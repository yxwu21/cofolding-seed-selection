#!/usr/bin/env python3
"""Paper v1 robustness and sensitivity analyses (post hoc, exploratory).

Added after an internal referee review (2026-10-16). None of these analyses
changes the frozen RQ1-RQ3 results; each is reported as exploratory.
Outputs (results/paper_v1/tables/robustness_*.csv):
  rmsd       pre-specified secondary endpoint (RMSD < 2 A success), paired gains
  excluded   characteristics of systems excluded by the complete-case rule
  available  available-case analysis: systems with >= 1 labelled sample in all 5 seeds
  subsets    gains on systems common to methods and on single-ligand systems
  samples    sample-level budgets: m samples of one seed vs m samples across seeds
  losssweep  operating points for loss limits 0.0025 / 0.005 / 0.01 (dev-chosen)
  stopdist   share of test runs stopping after each seed (confidence rule)
  predict    within-stage AUCs and bootstrap CI of the AUC gain over the stage prior
  material   improve/worsen shares at materiality thresholds 0 / 0.01 / 0.05 / 0.1
  headroom   headroom share and near-best share with random baselines and CIs
  perstudy   per-system spread of the 1-seed vs 5-seed difference
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_v1_budget_eval import (  # noqa: E402
    METHODS, PERMS, QUANTILES, auc, candidate_table, cluster_bootstrap, complete_systems,
    evaluate_rule, fit_logistic, load_inputs, permutation_paths, predict_logistic,
    stop_index, system_arrays,
)

SEED = 20261017
LIMITS = (0.0025, 0.005, 0.01)


def boot_ratio(num, den, clusters, rng, reps=2000):
    labels, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(labels))
    sd = np.bincount(inv, weights=den, minlength=len(labels))
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    boot = sn[draws].sum(1) / sd[draws].sum(1)
    # Fourth value: bootstrap standard error (added 2026-10-22 for reporting, deviation D5).
    return (float(num.sum() / den.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)),
            float(np.std(boot, ddof=1)))


def available_case(cand, meta):
    """Systems with 5 seeds and >= 1 labelled sample per seed; select among available samples."""
    per = cand.groupby(["target", "seed"]).size()
    nseed = per.groupby("target").size()
    ok = nseed[nseed == 5].index
    c = cand[cand["target"].isin(ok)].sort_values(["target", "seed", "sample"])
    systems = c["target"].unique()
    S = np.full((len(systems), 5), np.nan)
    QT = np.full((len(systems), 5), np.nan)
    O = np.full((len(systems), 5), np.nan)
    idx = {s: i for i, s in enumerate(systems)}
    for (t, seed), g in c.groupby(["target", "seed"], sort=False):
        i = idx[t]
        j = int(np.argmax(np.isnan(S[i])))  # next free seed slot (ascending seed order)
        top = g["score"].to_numpy().argmax()
        S[i, j], QT[i, j], O[i, j] = g["score"].iloc[top], g["q"].iloc[top], g["q"].max()
    key = S - np.arange(5)[None, :] * 1e-12
    Kp = key[:, PERMS]
    best, pick = Kp[..., 0].copy(), np.zeros(Kp.shape[:2], int)
    ret = np.zeros(Kp.shape)
    seeds_by_pos = np.broadcast_to(PERMS, Kp.shape)
    for t in range(5):
        if t:
            better = Kp[..., t] > best
            best, pick = np.where(better, Kp[..., t], best), np.where(better, t, pick)
        chosen = np.take_along_axis(seeds_by_pos[..., t:t + 1] * 0 + seeds_by_pos, pick[..., None], axis=2)[..., 0]
        ret[..., t] = np.take_along_axis(np.broadcast_to(QT[:, None, :], Kp.shape), chosen[..., None], axis=2)[..., 0]
    orc = np.maximum.accumulate(O[:, PERMS], axis=2)
    return systems, ret, orc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/tables")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    meta, proper, preds = load_inputs(ROOT / "data/raw/runs_n_poses_predictions.tar.gz",
                                      ROOT / "data/raw/runs_n_poses_annotations.csv")
    out = {k: [] for k in ["rmsd", "excluded", "available", "subsets", "samples", "losssweep", "stopdist",
                           "predict", "material", "headroom", "perstudy"]}
    cache = {}
    for method in METHODS:
        cand = candidate_table(preds[method], proper, meta)
        complete = complete_systems(cand)
        arr = system_arrays(cand, complete)
        sysmeta = meta.loc[arr["systems"]]
        paths = permutation_paths(arr)
        cache[method] = (arr, sysmeta, paths)
        clusters = sysmeta["cluster"].to_numpy()
        split = sysmeta["split"].to_numpy()
        test = split == "test"

        # --- RMSD endpoint (pre-specified secondary)
        rok = arr["rmsd_ok"]
        for scope, mask in (("all", rok), ("test", rok & test)):
            for name in ("ret_r", "orc_r"):
                per = paths[name][mask].mean(axis=1)
                m, lo, hi, se = cluster_bootstrap(per[:, 4] - per[:, 0], clusters[mask], rng, with_se=True)
                out["rmsd"].append({"method": method, "scope": scope, "quantity": name, "n": int(mask.sum()),
                                    "k1": float(per[:, 0].mean()), "k5": float(per[:, 4].mean()),
                                    "gain": m, "lo": lo, "hi": hi,
                                    "sd": float(np.std(per[:, 4] - per[:, 0], ddof=1)), "se": se})

        # --- excluded systems
        labelled = cand.groupby("target").agg(q=("q", "mean"), n=("q", "size"))
        excl = labelled.index.difference(complete)
        for label, ids in (("complete", complete), ("excluded", excl)):
            mm = meta.loc[meta.index.intersection(ids)]
            out["excluded"].append({"method": method, "group": label, "n_systems": len(ids),
                                    "mean_labelled_q": float(labelled.loc[ids, "q"].mean()),
                                    "mean_novelty": float(mm["novelty"].mean()),
                                    "multi_ligand_share": float((mm["n_proper"] > 1).mean())})

        # --- available-case analysis
        systems, ret, orc = available_case(cand, meta)
        cl = meta.loc[systems, "cluster"].to_numpy()
        for name, a in (("ret_q", ret), ("orc_q", orc)):
            per = np.nanmean(a, axis=1)
            m, lo, hi, se = cluster_bootstrap(per[:, 4] - per[:, 0], cl, rng, with_se=True)
            out["available"].append({"method": method, "quantity": name, "n_systems": len(systems),
                                     "k1": float(per[:, 0].mean()), "k5": float(per[:, 4].mean()),
                                     "gain": m, "lo": lo, "hi": hi,
                                    "sd": float(np.std(per[:, 4] - per[:, 0], ddof=1)), "se": se})

        # --- samples within one seed vs across seeds (oracle and native, m = 1..5)
        cs, cq = arr["cs"], arr["cq"]
        for m_ in range(1, 6):
            within_o, within_r, across_o, across_r = [], [], [], []
            for sub in itertools.combinations(range(5), m_):
                # m samples of one seed (averaged over seeds)
                s = cs[:, :, list(sub)]
                q = cq[:, :, list(sub)]
                top = s.argmax(axis=2)
                within_r.append(np.take_along_axis(q, top[..., None], 2)[..., 0].mean(axis=1))
                within_o.append(q.max(axis=2).mean(axis=1))
                # one sample from each of m seeds (sample index 0..4 averaged)
                for samp in range(5):
                    s2, q2 = cs[:, list(sub), samp], cq[:, list(sub), samp]
                    t2 = s2.argmax(axis=1)
                    across_r.append(q2[np.arange(len(q2)), t2])
                    across_o.append(q2.max(axis=1))
            out["samples"].append({"method": method, "m_samples": m_,
                                   "within_seed_returned": float(np.mean(within_r)),
                                   "within_seed_oracle": float(np.mean(within_o)),
                                   "across_seeds_returned": float(np.mean(across_r)),
                                   "across_seeds_oracle": float(np.mean(across_o))})

        # --- loss-limit sweep and stop distribution (confidence rule)
        dev = split == "dev"
        tau_grid = np.unique(np.quantile(arr["S"][dev].ravel(), QUANTILES))
        evals = []
        for tau in tau_grid:
            stop = stop_index(paths["cms"], float(tau), None)
            evals.append((float(tau), stop, evaluate_rule(paths, dev, stop)))
        for limit in LIMITS:
            feasible = [(e[2]["cost"].mean(), -e[0], e) for e in evals if e[2]["loss"].mean() <= limit]
            if not feasible:
                out["losssweep"].append({"method": method, "limit": limit, "note": "none feasible"})
                continue
            tau, stop, dv = min(feasible, key=lambda x: (x[0], x[1]))[2]
            ts = evaluate_rule(paths, test, stop)
            row = {"method": method, "limit": limit, "tau": tau, "dev_cost": float(dv["cost"].mean()),
                   "dev_saving": 1 - float(dv["cost"].mean()) / 5}
            for nm in ("cost", "loss", "signed", "succ_diff"):
                mm, lo, hi, se = cluster_bootstrap(ts[nm], clusters[test], rng, with_se=True)
                row.update({f"test_{nm}": mm, f"test_{nm}_lo": lo, f"test_{nm}_hi": hi,
                            f"test_{nm}_sd": float(np.std(ts[nm], ddof=1)), f"test_{nm}_se": se})
            row["test_saving"] = 1 - row["test_cost"] / 5
            out["losssweep"].append(row)
            if limit == 0.005:
                t = stop[test]
                for k in range(5):
                    out["stopdist"].append({"method": method, "stop_after_seed": k + 1,
                                            "share_of_test_runs": float((t == k).mean())})

        # --- per-system spread of the 1-seed vs 5-seed difference
        d15 = paths["ret_q"][:, :, 0].mean(axis=1) - paths["ret_q"][:, :, 4].mean(axis=1)
        out["perstudy"].append({"method": method, "n": len(d15),
                                "share_abs_gt_0.01": float((np.abs(d15) > 0.01).mean()),
                                "share_1seed_better_gt_0.01": float((d15 > 0.01).mean()),
                                "share_5seed_better_gt_0.01": float((d15 < -0.01).mean())})

        # --- materiality sweep (1 -> 2) and success flips
        d12 = paths["ret_q"][:, :, 1] - paths["ret_q"][:, :, 0]
        s12 = paths["ret_s"][:, :, 1] - paths["ret_s"][:, :, 0]
        for thr in (0.0, 0.01, 0.05, 0.1):
            out["material"].append({"method": method, "threshold": thr,
                                    "improve": float((d12 > thr).mean()), "worsen": float((d12 < -thr).mean())})
        out["material"].append({"method": method, "threshold": "success_flip",
                                "improve": float((s12 > 0).mean()), "worsen": float((s12 < 0).mean())})

        # --- headroom and near-best with random baselines
        csf, cqf = cs.reshape(len(cs), -1), cq.reshape(len(cq), -1)
        rnd = cqf.mean(axis=1)
        r5 = paths["ret_q"][:, :, 4].mean(axis=1)
        o5 = paths["orc_q"][:, :, 4].mean(axis=1)
        hm, hlo, hhi, hse = boot_ratio(r5 - rnd, o5 - rnd, clusters, rng)
        best = cqf.max(axis=1)
        top = (csf - np.arange(25)[None, :] * 1e-12).argmax(axis=1)
        near_native = (best - cqf[np.arange(len(cqf)), top] <= 0.01)
        near_random = (best[:, None] - cqf <= 0.01).mean(axis=1)
        spread = best - cqf.min(axis=1) > 0.1
        out["headroom"].append({
            "method": method, "headroom_share": hm, "headroom_lo": hlo, "headroom_hi": hhi,
            "near_best_native": float(near_native.mean()), "near_best_random": float(near_random.mean()),
            "near_best_native_spread": float(near_native[spread].mean()),
            "near_best_random_spread": float(near_random[spread].mean()),
            "headroom_se": hse,
        })

        # --- predictability: within-stage AUC and bootstrap CI of the AUC gain
        key = arr["S"] - np.arange(5)[None, :] * 1e-12
        idx = np.zeros(len(key), int)
        chosen = np.zeros(key.shape, int)
        for t in range(5):
            if t:
                idx = np.where(key[:, t] > key[np.arange(len(key)), idx], t, idx)
            chosen[:, t] = idx
        retc = np.take_along_axis(arr["QT"], chosen, axis=1)
        X, y, sidx = [], [], []
        for k in range(1, 5):
            sc = cs[:, :k, :].reshape(len(cs), -1)
            srt = np.sort(sc, axis=1)[:, ::-1]
            X.append(np.column_stack([np.full(len(cs), k), srt[:, 0], srt[:, 0] - srt[:, 1], sc.mean(1), sc.std(1),
                                      (sc >= srt[:, :1] - 0.01).sum(1)]))
            y.append(retc[:, k] - retc[:, k - 1] > 0.01)
            sidx.append(np.arange(len(cs)))
        X, y, sidx = np.vstack(X), np.concatenate(y).astype(float), np.concatenate(sidx)
        dmask, tmask = split[sidx] == "dev", split[sidx] == "test"
        model = fit_logistic(X[dmask], y[dmask])
        p_lr = predict_logistic(model, X[tmask])
        prior = {k: y[dmask & (X[:, 0] == k)].mean() for k in range(1, 5)}
        p_pr = np.array([prior[int(k)] for k in X[tmask, 0]])
        yt, kt, ct = y[tmask], X[tmask, 0], clusters[sidx[tmask]]
        for k in range(1, 5):
            mk = kt == k
            out["predict"].append({"method": method, "stage": k, "n": int(mk.sum()),
                                   "auc_logistic": auc(p_lr[mk], yt[mk]), "auc_best_score": auc(X[tmask][mk, 1], yt[mk])})
        labels, inv = np.unique(ct, return_inverse=True)
        gains = []
        for _ in range(500):
            pick = rng.integers(0, len(labels), len(labels))
            rows = np.concatenate([np.flatnonzero(inv == c) for c in pick])
            gains.append(auc(p_lr[rows], yt[rows]) - auc(p_pr[rows], yt[rows]))
        out["predict"].append({"method": method, "stage": "pooled", "n": int(len(yt)),
                               "auc_logistic": auc(p_lr, yt), "auc_prior": auc(p_pr, yt),
                               "auc_gain": auc(p_lr, yt) - auc(p_pr, yt),
                               "auc_gain_lo": float(np.nanpercentile(gains, 2.5)),
                               "auc_gain_hi": float(np.nanpercentile(gains, 97.5)),
                               "auc_gain_se": float(np.nanstd(gains, ddof=1))})
        print(method, "done", flush=True)

    # --- subsets: systems common to methods; single-ligand systems
    common6 = set.intersection(*[set(cache[m][0]["systems"]) for m in METHODS if m != "boltz2"])
    common7 = common6 & set(cache["boltz2"][0]["systems"])
    for method in METHODS:
        arr, sysmeta, paths = cache[method]
        ids = arr["systems"]
        for label, mask in (("common_6_methods", np.isin(ids, list(common6))),
                            ("common_7_methods", np.isin(ids, list(common7))),
                            ("single_proper_ligand", (sysmeta["n_proper"] == 1).to_numpy())):
            if label == "common_6_methods" and method == "boltz2":
                continue
            for name in ("ret_q", "orc_q"):
                per = paths[name][mask].mean(axis=1)
                m, lo, hi, se = cluster_bootstrap(per[:, 4] - per[:, 0], sysmeta["cluster"].to_numpy()[mask], rng,
                                                  with_se=True)
                out["subsets"].append({"method": method, "subset": label, "quantity": name, "n": int(mask.sum()),
                                       "gain": m, "lo": lo, "hi": hi,
                                    "sd": float(np.std(per[:, 4] - per[:, 0], ddof=1)), "se": se})
    for name, rows in out.items():
        pd.DataFrame(rows).to_csv(args.output_dir / f"robustness_{name}.csv", index=False)
    print("written", ", ".join(f"robustness_{n}.csv" for n in out))


if __name__ == "__main__":
    main()
