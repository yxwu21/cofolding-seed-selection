#!/usr/bin/env python3
"""Paper v1: sampling-budget evaluation on the Runs N' Poses prediction table.

Implements docs/paper_v1_protocol.md (RQ1-RQ3). Reads only the public RNP
prediction table and annotations; writes CSV/JSON tables to --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import platform
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["af3", "af3_no_template", "boltz", "boltz1x", "boltz2", "chai", "protenix"]
LABELS = {
    "af3": "AF3", "af3_no_template": "AF3 (no templates)", "boltz": "Boltz-1",
    "boltz1x": "Boltz-1x", "boltz2": "Boltz-2", "chai": "Chai-1", "protenix": "Protenix",
}
PERMS = np.array(list(itertools.permutations(range(5))))  # 120 x 5, position -> seed rank
SPLIT_SALT = "cofoldpilot-paper-v1:"
FOLD_SALT = "cofoldpilot-paper-v1-fold:"
BOOT = 2000
BOOT_SEED = 20261015
MATERIAL = 0.01
LOSS_MAX = 0.005
SUCCESS_LDDT = 0.8
SUCCESS_RMSD = 2.0
QUANTILES = np.round(np.arange(0.05, 0.951, 0.05), 2)
NOVELTY_BINS = [0.0, 30.0, 50.0, 70.0, 100.0 + 1e-9]
NOVELTY_LABELS = ["[0,30)", "[30,50)", "[50,70)", "[70,100]"]


def split_of(cluster: str) -> str:
    return "dev" if hashlib.sha256((SPLIT_SALT + str(cluster)).encode()).digest()[0] % 2 == 0 else "test"


def fold_of(cluster: str) -> int:
    return hashlib.sha256((FOLD_SALT + str(cluster)).encode()).digest()[0] % 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- data

def load_inputs(pred_tar: Path, ann_csv: Path):
    ann = pd.read_csv(ann_csv, low_memory=False)
    proper = ann.loc[ann["ligand_is_proper"] == True,  # noqa: E712
                     ["system_id", "ligand_instance_chain", "cluster", "sucos_shape_pocket_qcov"]].copy()
    meta = proper.groupby("system_id").agg(
        cluster=("cluster", "first"), n_clusters=("cluster", "nunique"),
        novelty=("sucos_shape_pocket_qcov", "min"), n_proper=("ligand_instance_chain", "nunique"),
    )
    meta["split"] = meta["cluster"].map(split_of)
    meta["fold"] = meta["cluster"].map(fold_of)
    meta["novelty_bin"] = pd.cut(meta["novelty"], NOVELTY_BINS, labels=NOVELTY_LABELS, right=False)
    preds = {}
    match_cols = {"ligand_instance_chain_lddt_pli", "ligand_instance_chain_rmsd"}
    with tarfile.open(pred_tar, "r:gz") as archive:
        for method in METHODS:
            member = archive.extractfile(f"predictions/{method}.csv")
            preds[method] = pd.read_csv(
                io.BytesIO(member.read()), low_memory=False,
                usecols=lambda c: c in {"target", "seed", "sample", "ranking_score", "lddt_pli", "rmsd",
                                        "ligand_instance_chain"} | match_cols,
            )
    return meta, proper[["system_id", "ligand_instance_chain"]], preds


def ligand_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, seed, sample, ligand) with its own LDDT-PLI and RMSD.

    The RNP tables repeat a ligand once per reference ligand when label tables
    were joined. Where the table records which ligand chain each label was
    computed for (`ligand_instance_chain_lddt_pli` / `_rmsd`), only matching
    rows are used. Otherwise duplicate rows carry identical LDDT-PLI, and the
    RMSD is kept only if all duplicates agree (else it is ambiguous -> NaN).
    """
    key = ["target", "seed", "sample", "ligand_instance_chain"]
    base = df[np.isfinite(df["ranking_score"])]
    score = base.groupby(key)["ranking_score"].agg(["first", lambda s: float(s.max() - s.min())])
    score.columns = ["score", "score_span"]
    if "ligand_instance_chain_lddt_pli" in base.columns:
        lddt_rows = base[base["ligand_instance_chain_lddt_pli"].astype(str) == base["ligand_instance_chain"].astype(str)]
        rmsd_rows = base[base["ligand_instance_chain_rmsd"].astype(str) == base["ligand_instance_chain"].astype(str)]
        lddt = lddt_rows.groupby(key)["lddt_pli"].agg(["first", "nunique"])
        rmsd = rmsd_rows.groupby(key)["rmsd"].agg(["first", "nunique"])
    else:
        lddt = base.groupby(key)["lddt_pli"].agg(["first", "nunique"])
        rmsd = base.groupby(key)["rmsd"].agg(["first", "nunique"])
    out = score.join(lddt.rename(columns={"first": "lddt", "nunique": "lddt_n"}), how="left")
    out = out.join(rmsd.rename(columns={"first": "rmsd", "nunique": "rmsd_n"}), how="left")
    out.loc[out["lddt_n"] != 1, "lddt"] = np.nan
    out.loc[out["rmsd_n"] != 1, "rmsd"] = np.nan
    return out.reset_index()


def candidate_table(df: pd.DataFrame, proper: pd.DataFrame, meta: pd.DataFrame):
    lig = ligand_labels(df)
    d = lig.merge(proper, left_on=["target", "ligand_instance_chain"],
                  right_on=["system_id", "ligand_instance_chain"], how="inner")
    d = d[np.isfinite(d["lddt"])]
    grouped = d.groupby(["target", "seed", "sample"])
    cand = grouped.agg(
        score=("score", "first"), score_span=("score_span", "max"),
        q=("lddt", "mean"), qmin=("lddt", "min"), rmax=("rmsd", "max"),
        rmsd_missing=("rmsd", lambda s: int(s.isna().sum())),
        n_lig=("ligand_instance_chain", "nunique"),
    ).reset_index()
    cand = cand[cand["n_lig"] == cand["target"].map(meta["n_proper"])]
    cand.loc[cand["rmsd_missing"] > 0, "rmax"] = np.nan
    cand["succ"] = cand["qmin"] >= SUCCESS_LDDT
    cand["succ_rmsd"] = cand["rmax"] < SUCCESS_RMSD  # False where RMSD is ambiguous; masked by rmsd_ok
    cand["rmsd_ok"] = np.isfinite(cand["rmax"])
    return cand


def complete_systems(cand: pd.DataFrame) -> pd.Index:
    per_seed = cand.groupby(["target", "seed"]).size()
    agg = per_seed.groupby("target").agg(["size", "min", "max"])
    ok = (agg["size"] == 5) & (agg["min"] == 5) & (agg["max"] == 5)
    return agg.index[ok]


def system_arrays(cand: pd.DataFrame, systems: pd.Index):
    """Return per-system arrays ordered by ascending seed id and sample id."""
    c = cand[cand["target"].isin(systems)].sort_values(["target", "seed", "sample"])
    n = len(systems)
    order = c["target"].drop_duplicates().to_numpy()
    shape = (len(order), 5, 5)
    cs = c["score"].to_numpy().reshape(shape)
    cq = c["q"].to_numpy().reshape(shape)
    csu = c["succ"].to_numpy().reshape(shape)
    csr = c["succ_rmsd"].to_numpy().reshape(shape)
    rok = c["rmsd_ok"].to_numpy().reshape(shape).all(axis=(1, 2))
    assert len(order) == n
    # seed-level summaries: native top candidate within each seed (tie -> lower sample)
    within = cs - np.arange(5)[None, None, :] * 1e-12
    top = within.argmax(axis=2)
    take = lambda arr: np.take_along_axis(arr, top[..., None], axis=2)[..., 0]  # noqa: E731
    return {
        "systems": order, "cs": cs, "cq": cq, "csu": csu, "csr": csr, "rmsd_ok": rok,
        "S": take(cs), "QT": take(cq), "SU": take(csu), "SR": take(csr),
        "O": cq.max(axis=2), "OS": csu.any(axis=2), "OSR": csr.any(axis=2),
    }


# ------------------------------------------------------------------ permutations

def permutation_paths(arr: dict):
    """Returned/oracle quality after t+1 batches for every system x ordering."""
    S = arr["S"]
    key = S - np.arange(5)[None, :] * 1e-12  # ties across seeds -> lower seed id
    Kp = key[:, PERMS]                       # n x 120 x 5 (position order)
    idx = np.zeros(Kp.shape[:2], dtype=int)
    best = Kp[..., 0].copy()
    run_idx = np.zeros(Kp.shape, dtype=int)
    for t in range(5):
        if t:
            better = Kp[..., t] > best
            best = np.where(better, Kp[..., t], best)
            idx = np.where(better, t, idx)
        run_idx[..., t] = idx
    seeds_by_pos = np.broadcast_to(PERMS, Kp.shape)
    chosen = np.take_along_axis(seeds_by_pos, run_idx, axis=2)  # seed rank chosen after t+1 batches
    gather = lambda a: np.take_along_axis(np.broadcast_to(a[:, None, :], Kp.shape), chosen, axis=2)  # noqa: E731
    return {
        "ret_q": gather(arr["QT"]), "ret_s": gather(arr["SU"]).astype(float), "ret_r": gather(arr["SR"]).astype(float),
        "orc_q": np.maximum.accumulate(arr["O"][:, PERMS], axis=2),
        "orc_s": np.logical_or.accumulate(arr["OS"][:, PERMS], axis=2).astype(float),
        "orc_r": np.logical_or.accumulate(arr["OSR"][:, PERMS], axis=2).astype(float),
        "cms": np.maximum.accumulate(S[:, PERMS], axis=2),
    }


def stop_index(cms: np.ndarray, tau: float | None = None, delta: float | None = None) -> np.ndarray:
    cond = np.zeros(cms.shape, dtype=bool)
    if tau is not None:
        cond |= cms >= tau
    if delta is not None:
        inc = np.diff(cms, axis=2)
        cond[..., 1:] |= inc < delta
    cond[..., 4] = True
    return cond.argmax(axis=2)


# --------------------------------------------------------------------- statistics

def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator, reps: int = BOOT,
                      with_se: bool = False):
    """Percentile CI of the system-weighted mean, resampling clusters.

    With with_se=True the bootstrap standard error (the standard deviation of the resampled
    means) is returned as a fourth value; the resamples, and hence all other values, are the
    same either way (added 2026-10-22 for reporting, deviation D5).
    """
    values = np.asarray(values, dtype=float)
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    out = (float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return out + (float(np.std(boot, ddof=1)),) if with_se else out


def auc(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    penalty = np.full(Z.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - y) + penalty * w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def predict_logistic(model, X: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))


# ------------------------------------------------------------------- analyses

def budget_rows(method, paths, sysmeta, rng, rmsd_ok):
    rows = []
    for split in ("all", "dev", "test"):
        mask = np.ones(len(sysmeta), bool) if split == "all" else (sysmeta["split"].to_numpy() == split)
        clusters = sysmeta["cluster"].to_numpy()[mask]
        rmask = mask & rmsd_ok  # RMSD metrics only where every candidate's RMSD is unambiguous
        for k in range(1, 6):
            row = {"method": method, "split": split, "k": k, "n_systems": int(mask.sum()),
                   "n_rmsd_systems": int(rmask.sum())}
            for name in ("ret_q", "orc_q", "ret_s", "orc_s", "ret_r", "orc_r"):
                use = rmask if name in ("ret_r", "orc_r") else mask
                per_system = paths[name][use, :, k - 1].mean(axis=1)
                mean, lo, hi, se = cluster_bootstrap(per_system, sysmeta["cluster"].to_numpy()[use], rng, with_se=True)
                row[name] = mean
                row[name + "_lo"] = lo
                row[name + "_hi"] = hi
                row[name + "_sd"] = float(np.std(per_system, ddof=1))
                row[name + "_se"] = se
            gap = (paths["orc_q"][mask, :, k - 1] - paths["ret_q"][mask, :, k - 1]).mean(axis=1)
            row["gap_q"], row["gap_q_lo"], row["gap_q_hi"], row["gap_q_se"] = cluster_bootstrap(gap, clusters, rng,
                                                                                                 with_se=True)
            row["gap_q_sd"] = float(np.std(gap, ddof=1))
            rows.append(row)
    return rows


def change_rows(method, paths, sysmeta):
    rows = []
    for k in range(1, 5):
        for name, prefix in (("ret_q", "returned"), ("orc_q", "oracle")):
            delta = paths[name][:, :, k] - paths[name][:, :, k - 1]
            rows.append({
                "method": method, "transition": f"{k}->{k + 1}", "quantity": prefix,
                "n_systems": len(sysmeta),
                "frac_improve": float((delta > MATERIAL).mean()),
                "frac_worsen": float((delta < -MATERIAL).mean()),
                "frac_same": float((np.abs(delta) <= MATERIAL).mean()),
                "frac_any_change": float((delta != 0).mean()),
                "mean_delta": float(delta.mean()),
            })
    return rows


def novelty_rows(method, paths, sysmeta):
    rows = []
    for label in NOVELTY_LABELS:
        mask = (sysmeta["novelty_bin"] == label).to_numpy()
        if not mask.any():
            continue
        for k in (1, 2, 5):
            rows.append({
                "method": method, "novelty_bin": label, "k": k, "n_systems": int(mask.sum()),
                "ret_q": float(paths["ret_q"][mask, :, k - 1].mean()), "orc_q": float(paths["orc_q"][mask, :, k - 1].mean()),
                "ret_s": float(paths["ret_s"][mask, :, k - 1].mean()), "orc_s": float(paths["orc_s"][mask, :, k - 1].mean()),
            })
    return rows


def evaluate_rule(paths, mask, stop):
    q5 = paths["ret_q"][mask, :, 4]
    t = stop[mask]
    q = np.take_along_axis(paths["ret_q"][mask], t[..., None], axis=2)[..., 0]
    s = np.take_along_axis(paths["ret_s"][mask], t[..., None], axis=2)[..., 0]
    s5 = paths["ret_s"][mask, :, 4]
    return {
        "cost": (t + 1).mean(axis=1), "loss": np.maximum(q5 - q, 0).mean(axis=1),
        "signed": (q5 - q).mean(axis=1), "q": q.mean(axis=1), "succ": s.mean(axis=1),
        "succ_diff": (s5 - s).mean(axis=1),
    }


def stopping_rows(method, paths, sysmeta, arr, rng):
    split = sysmeta["split"].to_numpy()
    dev, test = split == "dev", split == "test"
    clusters_test = sysmeta["cluster"].to_numpy()[test]
    tau_grid = np.unique(np.quantile(arr["S"][dev].ravel(), QUANTILES))
    inc = np.diff(paths["cms"][dev], axis=2).ravel()
    delta_grid = np.unique(np.quantile(inc[inc > 0], QUANTILES)) if (inc > 0).any() else np.array([0.0])
    candidates = [("fixed", {"k": k}) for k in range(1, 6)]
    candidates += [("conf", {"tau": float(t)}) for t in tau_grid]
    candidates += [("plateau", {"delta": float(d)}) for d in delta_grid]
    candidates += [("conf_or_plateau", {"tau": float(t), "delta": float(d)}) for t in tau_grid for d in delta_grid]
    frontier, chosen = [], {}
    for family, params in candidates:
        if family == "fixed":
            stop = np.full(paths["cms"].shape[:2], params["k"] - 1)
        else:
            stop = stop_index(paths["cms"], params.get("tau"), params.get("delta"))
        dv, ts = evaluate_rule(paths, dev, stop), evaluate_rule(paths, test, stop)
        record = {"method": method, "family": family, **{f"param_{k}": v for k, v in params.items()},
                  "dev_cost": float(dv["cost"].mean()), "dev_loss": float(dv["loss"].mean()),
                  "test_cost": float(ts["cost"].mean()), "test_loss": float(ts["loss"].mean()),
                  "test_signed": float(ts["signed"].mean()), "test_q": float(ts["q"].mean()),
                  "test_succ": float(ts["succ"].mean()), "test_succ_diff": float(ts["succ_diff"].mean())}
        frontier.append(record)
        if family != "fixed" and record["dev_loss"] <= LOSS_MAX:
            key = (record["dev_cost"], -params.get("tau", 0.0), params.get("delta", 0.0))
            if family not in chosen or key < chosen[family][0]:
                chosen[family] = (key, params, stop)
    ops = []
    picks = [("fixed", {"k": k}, np.full(paths["cms"].shape[:2], k - 1)) for k in (1, 2, 3, 5)]
    picks += [(fam, params, stop) for fam, (_, params, stop) in sorted(chosen.items())]
    for family, params, stop in picks:
        dv, ts = evaluate_rule(paths, dev, stop), evaluate_rule(paths, test, stop)
        row = {"method": method, "family": family, **{f"param_{k}": v for k, v in params.items()},
               "dev_cost": float(dv["cost"].mean()), "dev_loss": float(dv["loss"].mean()),
               "n_test": int(test.sum())}
        for name in ("cost", "loss", "signed", "q", "succ", "succ_diff"):
            mean, lo, hi, se = cluster_bootstrap(ts[name], clusters_test, rng, with_se=True)
            row[f"test_{name}"], row[f"test_{name}_lo"], row[f"test_{name}_hi"] = mean, lo, hi
            row[f"test_{name}_sd"] = float(np.std(ts[name], ddof=1))
            row[f"test_{name}_se"] = se
        row["test_saving_vs_fixed5"] = 1 - row["test_cost"] / 5
        ops.append(row)
    for family in ("conf", "plateau", "conf_or_plateau"):
        if family not in chosen:
            ops.append({"method": method, "family": family, "note": "no dev threshold met the loss limit",
                        "n_test": int(test.sum())})
    return frontier, ops


def predictability_rows(method, arr, sysmeta):
    """Canonical (ascending seed) order; label: returned q improves by > MATERIAL."""
    cs, QT = arr["cs"], arr["QT"]
    key = arr["S"] - np.arange(5)[None, :] * 1e-12
    run_best = np.maximum.accumulate(key, axis=1)
    chosen = np.zeros(key.shape, dtype=int)
    idx = np.zeros(len(key), dtype=int)
    for t in range(5):
        if t:
            idx = np.where(key[:, t] > key[np.arange(len(key)), idx], t, idx)
        chosen[:, t] = idx
    ret = np.take_along_axis(QT, chosen, axis=1)
    feats, labels, rows_meta = [], [], []
    for k in range(1, 5):
        scores = cs[:, :k, :].reshape(len(cs), -1)
        srt = np.sort(scores, axis=1)[:, ::-1]
        best, second = srt[:, 0], srt[:, 1]
        feats.append(np.column_stack([
            np.full(len(cs), k), best, best - second, scores.mean(axis=1), scores.std(axis=1),
            (scores >= best[:, None] - 0.01).sum(axis=1),
        ]))
        labels.append(ret[:, k] - ret[:, k - 1] > MATERIAL)
        rows_meta.append(np.column_stack([np.arange(len(cs)), np.full(len(cs), k)]))
    X, y, M = np.vstack(feats), np.concatenate(labels).astype(float), np.vstack(rows_meta)
    names = ["k", "best_score", "top2_gap", "mean_score", "std_score", "n_within_0.01"]
    sys_idx = M[:, 0]
    split = sysmeta["split"].to_numpy()[sys_idx]
    fold = sysmeta["fold"].to_numpy()[sys_idx]
    dev, test = split == "dev", split == "test"
    out = []

    def prior_fit(mask):
        return {k: y[mask & (X[:, 0] == k)].mean() for k in range(1, 5)}

    def prior_pred(table, mask):
        return np.array([table[int(k)] for k in X[mask, 0]])

    # dev cross-validation
    cv_pred_lr, cv_pred_prior = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for f in range(5):
        tr, te = dev & (fold != f), dev & (fold == f)
        if te.sum() == 0 or y[tr].std() == 0:
            continue
        model = fit_logistic(X[tr], y[tr])
        cv_pred_lr[te] = predict_logistic(model, X[te])
        cv_pred_prior[te] = prior_pred(prior_fit(tr), te)
    have = dev & np.isfinite(cv_pred_lr)
    model = fit_logistic(X[dev], y[dev])
    test_lr = predict_logistic(model, X[test])
    test_prior = prior_pred(prior_fit(dev), test)
    base = {"method": method, "n_dev_rows": int(dev.sum()), "n_test_rows": int(test.sum()),
            "pos_rate_dev": float(y[dev].mean()), "pos_rate_test": float(y[test].mean())}
    out.append({**base, "model": "stage_prior", "dev_cv_auc": auc(cv_pred_prior[have], y[have]),
                "test_auc": auc(test_prior, y[test]),
                "dev_cv_brier": float(np.mean((cv_pred_prior[have] - y[have]) ** 2)),
                "test_brier": float(np.mean((test_prior - y[test]) ** 2))})
    out.append({**base, "model": "logistic_all_features", "dev_cv_auc": auc(cv_pred_lr[have], y[have]),
                "test_auc": auc(test_lr, y[test]),
                "dev_cv_brier": float(np.mean((cv_pred_lr[have] - y[have]) ** 2)),
                "test_brier": float(np.mean((test_lr - y[test]) ** 2))})
    for j, name in enumerate(names[1:], start=1):
        out.append({**base, "model": f"single:{name}", "dev_cv_auc": auc(X[dev, j], y[dev]),
                    "test_auc": auc(X[test, j], y[test]), "dev_cv_brier": float("nan"), "test_brier": float("nan")})
    return out


# ------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/tables")
    parser.add_argument("--dev-only", action="store_true",
                        help="debug mode: drop test-split systems and use dev folds 0-1 as a pseudo test split")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    meta, proper, preds = load_inputs(args.predictions, args.annotations)
    if args.dev_only:
        meta = meta[meta["split"] == "dev"].copy()
        meta["split"] = np.where(meta["fold"] <= 1, "test", "dev")
        proper = proper[proper["system_id"].isin(meta.index)]
    coverage, budget, changes, novelty, frontier, ops, pred = [], [], [], [], [], [], []
    for method in METHODS:
        cand = candidate_table(preds[method], proper, meta)
        systems = complete_systems(cand)
        arr = system_arrays(cand, systems)
        sysmeta = meta.loc[arr["systems"]]
        paths = permutation_paths(arr)
        coverage.append({
            "method": method, "label": LABELS[method],
            "systems_in_table": int(preds[method]["target"].nunique()),
            "systems_with_proper_labels": int(cand["target"].nunique()),
            "complete_systems": len(systems), "rmsd_evaluable_systems": int(arr["rmsd_ok"].sum()),
            "dev": int((sysmeta["split"] == "dev").sum()),
            "test": int((sysmeta["split"] == "test").sum()), "clusters": int(sysmeta["cluster"].nunique()),
            "max_score_span_within_candidate": float(cand["score_span"].max()),
            "multi_cluster_systems": int((sysmeta["n_clusters"] > 1).sum()),
        })
        budget += budget_rows(method, paths, sysmeta, rng, arr["rmsd_ok"])
        changes += change_rows(method, paths, sysmeta)
        novelty += novelty_rows(method, paths, sysmeta)
        fr, op = stopping_rows(method, paths, sysmeta, arr, rng)
        frontier += fr
        ops += op
        pred += predictability_rows(method, arr, sysmeta)
        print(f"{method}: {len(systems)} complete systems", flush=True)
    outputs = {
        "coverage.csv": coverage, "budget_curves.csv": budget, "next_batch_changes.csv": changes,
        "novelty_strata.csv": novelty, "stopping_frontier.csv": frontier,
        "stopping_operating_points.csv": ops, "predictability.csv": pred,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.output_dir / name, index=False)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    receipt = {
        "script": "scripts/paper_v1_budget_eval.py", "git_commit": commit,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "predictions_sha256": sha256_file(args.predictions), "annotations_sha256": sha256_file(args.annotations),
        "dev_only_debug": bool(args.dev_only),
        "bootstrap_reps": BOOT, "bootstrap_seed": BOOT_SEED, "wall_seconds": time.perf_counter() - started,
        "outputs": sorted(outputs),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
