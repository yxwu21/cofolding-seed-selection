#!/usr/bin/env python3
"""E24 (seed extension v1): frozen analysis of Chai-1 with 20 seeds per system.

Implements the analysis plan of docs/seedext_v1_protocol.md, fixed before any E24
result existed. Inputs: configs/seedext_v1_systems.json, the OST scores of the new
candidates (C3) and of the archived RNP structures (C2), the G1 profiles, and the RNP
prediction table and annotations.

* Quality of a candidate: LDDT-PLI assigned to the proper ligand (reference file named
  by its ligand instance chain); success LDDT-PLI >= 0.8, RMSD success < 2 A.
* Per seed: the candidate with the highest ranking score (ties to the lower sample)
  gives the seed's top score and returned quality; the maximum gives the seed's best.
* After k seeds: expectations over random k-subsets of the 20 seeds, computed exactly
  from seed ranks (ties to the earlier seed in the configured order).
* Complete system: 20 seeds and 100 candidates with a valid quality. The primary
  analysis needs at least 81 complete systems.
* Primary: delta_ret(5->20), tiers weighted equally, 95% percentile interval from a
  cluster bootstrap stratified by tier (2,000 draws, seed 20261024). A: upper bound
  < 0.02; B: estimate >= 0.02 and lower bound > 0; C: otherwise.
* Secondary: delta_orc(5->20); per-tier deltas; gains of seeds 2..20 and k* (t = 0.005,
  sensitivity 0.002 and 0.01) per design tier, overall, and by the paper's first-seed
  confidence tiers; success rates; ranking over 100 candidates; the frozen Chai-1
  confidence rule with caps of 5 and 20 seeds (Monte Carlo over orderings); timing.
* Scoring parity (C2) and replication of the archived seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_v1_budget_eval as ev  # noqa: E402

N_SEEDS = 20
TIERS = ("low", "mid", "high")
THRESHOLDS = (0.002, 0.005, 0.01)
MARGIN = 0.02
MIN_COMPLETE = 81
PARITY_WITHIN, PARITY_SHARE, PARITY_MEAN = 0.01, 0.95, 0.005
BOOT, BOOT_SEED = 2000, 20261024
RULE_PERMUTATIONS, RULE_SEED = 4000, 20261025
OUTCOMES = ("ret_q", "orc_q", "ret_s", "orc_s", "ret_r", "orc_r")


# ------------------------------------------------------------------ expectations

def subset_weights(n: int, k: int) -> np.ndarray:
    """P(the r-th ranked of n items is the top item of a uniform random k-subset), r = 1..n."""
    return np.array([math.comb(n - r, k - 1) for r in range(1, n + 1)], dtype=float) / math.comb(n, k)


def rank_order(keys: np.ndarray) -> np.ndarray:
    """Indices sorted by key descending, ties to the lower index."""
    return np.lexsort((np.arange(len(keys)), -np.asarray(keys, dtype=float)))


def expected_top(keys: np.ndarray, values: np.ndarray, k: int) -> float:
    """E[value of the largest-key item in a uniform random k-subset]."""
    return float(subset_weights(len(keys), k) @ np.asarray(values, dtype=float)[rank_order(keys)])


def expected_top_given_first(keys: np.ndarray, values: np.ndarray, first: int, k: int) -> float:
    """As expected_top, for subsets that contain item `first` plus k - 1 random others."""
    keys, values = np.asarray(keys, dtype=float), np.asarray(values, dtype=float)
    if k == 1:
        return float(values[first])
    others = np.delete(np.arange(len(keys)), first)
    order = others[rank_order(keys[others])]
    w = subset_weights(len(others), k - 1)
    beats = (keys[order] > keys[first]) | ((keys[order] == keys[first]) & (order < first))
    return float(w[beats] @ values[order][beats] + (1.0 - w[beats].sum()) * values[first])


# ------------------------------------------------------------------ data

def proper_chains(annotations: Path) -> dict:
    ann = pd.read_csv(annotations, low_memory=False)
    proper = ann.loc[ann["ligand_is_proper"] == True, ["system_id", "ligand_instance_chain"]]  # noqa: E712
    return proper.groupby("system_id")["ligand_instance_chain"].apply(lambda s: sorted(set(s))).to_dict()


def candidate_quality(scores: pd.DataFrame, chains: dict) -> pd.DataFrame:
    """Mean LDDT-PLI and max RMSD over proper ligands; missing if any proper ligand is unscored."""
    out = scores.copy()
    q, r = [], []
    for row in out.itertuples():
        wanted = chains.get(row.system_id, [])
        lddt = json.loads(row.lddt_pli) if isinstance(row.lddt_pli, str) and row.lddt_pli else {}
        rmsd = json.loads(row.rmsd) if isinstance(row.rmsd, str) and row.rmsd else {}
        ok = row.status == "SUCCESS" and wanted and all(c in lddt for c in wanted)
        q.append(float(np.mean([lddt[c] for c in wanted])) if ok else np.nan)
        r.append(float(max(rmsd[c] for c in wanted)) if ok and all(c in rmsd for c in wanted) else np.nan)
    out["q"], out["rmsd_proper"] = q, r
    return out


def system_seed_arrays(cand: pd.DataFrame, seeds: list[int]) -> dict | None:
    """Seed-level arrays in configured seed order; None if the system is incomplete."""
    cand = cand.drop_duplicates(["seed", "sample"]).set_index(["seed", "sample"])
    idx = pd.MultiIndex.from_product([seeds, range(5)], names=["seed", "sample"])
    if not idx.isin(cand.index).all():
        return None
    c = cand.loc[idx]
    if c["q"].isna().any() or c["ranking_score"].isna().any():
        return None
    s = c["ranking_score"].to_numpy(float).reshape(len(seeds), 5)
    q = c["q"].to_numpy(float).reshape(len(seeds), 5)
    r = c["rmsd_proper"].to_numpy(float).reshape(len(seeds), 5)
    top = s.argmax(axis=1)  # first maximum, i.e. lower sample on ties
    take = lambda a: a[np.arange(len(seeds)), top]  # noqa: E731
    return {"S": take(s), "QT": take(q), "O": q.max(axis=1),
            "SU": (take(q) >= 0.8).astype(float), "OS": (q >= 0.8).any(axis=1).astype(float),
            "SR": (take(r) < 2).astype(float), "OSR": (r < 2).any(axis=1).astype(float),
            "rmsd_ok": bool(np.isfinite(r).all()), "scores": s, "q": q}


def curves(a: dict, n: int = N_SEEDS) -> dict:
    """Expected returned and best values after k = 1..n random seeds."""
    pairs = {"ret_q": ("S", "QT"), "orc_q": ("O", "O"), "ret_s": ("S", "SU"), "orc_s": ("OS", "OS"),
             "ret_r": ("S", "SR"), "orc_r": ("OSR", "OSR")}
    return {name: np.array([expected_top(a[key], a[val], k) for k in range(1, n + 1)])
            for name, (key, val) in pairs.items()}


def first_seed_curves(a: dict, n: int = N_SEEDS) -> dict:
    """Per first seed j: expected returned and best quality after k seeds (k = 1..n)."""
    out = {}
    for name, (key, val) in {"ret_q": ("S", "QT"), "orc_q": ("O", "O")}.items():
        out[name] = np.array([[expected_top_given_first(a[key], a[val], j, k) for k in range(1, n + 1)]
                              for j in range(n)])
    return out


# ------------------------------------------------------------------ statistics

def stratified_bootstrap(per_system: pd.DataFrame, column: str, rng: np.random.Generator):
    """Equal-weight mean over design tiers with a cluster bootstrap within each tier.

    Returns ((mean, lo, hi, se), {tier: (mean, lo, hi, n, se)}); se is the bootstrap standard
    error, added 2026-10-22 for reporting (paper v1 deviation D5). The resamples are unchanged.
    """
    boots, means, tier_rows = [], [], {}
    for tier in TIERS:
        sub = per_system[per_system["tier"] == tier]
        labels, inverse = np.unique(sub["cluster"].to_numpy(), return_inverse=True)
        values = np.vstack(sub[column].to_numpy())  # systems x K
        sums = np.zeros((len(labels), values.shape[1]))
        np.add.at(sums, inverse, values)
        counts = np.bincount(inverse, minlength=len(labels)).astype(float)
        draws = rng.integers(0, len(labels), size=(BOOT, len(labels)))
        weights = np.zeros((BOOT, len(labels)))
        np.add.at(weights, (np.arange(BOOT)[:, None], draws), 1.0)
        boot = (weights @ sums) / (weights @ counts)[:, None]
        mean = values.mean(axis=0)
        tier_rows[tier] = (mean, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0), len(sub),
                           boot.std(axis=0, ddof=1))
        boots.append(boot)
        means.append(mean)
    overall = np.mean(boots, axis=0)
    mean = np.mean(means, axis=0)
    return (mean, np.percentile(overall, 2.5, axis=0), np.percentile(overall, 97.5, axis=0),
            overall.std(axis=0, ddof=1)), tier_rows


def k_star(gains: np.ndarray, threshold: float):
    """gains[k-2] is the gain of seed k (k = 2..n). Returns (k*, reached)."""
    below = np.flatnonzero(gains < threshold)
    return (int(below[0]) + 1, True) if below.size else (len(gains) + 1, False)


def reading(mean: float, lo: float, hi: float) -> str:
    if hi < MARGIN:
        return "A"
    if mean >= MARGIN and lo > 0:
        return "B"
    return "C"


# ------------------------------------------------------------------ rule, timing

def rule_simulation(arrays: dict, tau: float, rng: np.random.Generator) -> dict:
    """Frozen confidence rule over random seed orderings with caps of 5 and 20 seeds."""
    out = {}
    for cap in (5, N_SEEDS):
        seeds, quality, fixed = [], [], []
        for _ in range(RULE_PERMUTATIONS):
            perm = rng.permutation(N_SEEDS)
            s, qt = arrays["S"][perm], arrays["QT"][perm]
            hits = np.flatnonzero(s[:cap] >= tau)
            stop = int(hits[0]) + 1 if hits.size else cap
            first = s[:stop]
            seeds.append(stop)
            quality.append(qt[:stop][rank_order(first)[0]])
            fixed.append(qt[:cap][rank_order(s[:cap])[0]])
        out[cap] = {"mean_seeds": float(np.mean(seeds)), "ret_rule": float(np.mean(quality)),
                    "ret_fixed": float(np.mean(fixed)),
                    "positive_loss": float(np.mean(np.maximum(np.array(fixed) - np.array(quality), 0)))}
    return out


def timing_rows(g1_run: Path, config: dict) -> pd.DataFrame:
    rows = []
    meta = {s["system_id"]: s for s in config["systems"]}
    for profile in sorted(g1_run.glob("task_*/*/profile.jsonl")):
        system = profile.parent.name
        events = [json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
        start = next((e for e in events if e["event"] == "start"), {})
        batches = [e for e in events if e["event"] == "batch" and e["status"] == "completed"]
        if not batches:
            continue
        walls = np.array([b["wall_seconds"] for b in batches])
        rows.append({"system_id": system, "task": profile.parent.parent.name, "gpu": start.get("gpu_name"),
                     "padded_tokens": meta.get(system, {}).get("padded_tokens"), "n_seeds": len(walls),
                     "startup_s": batches[0]["cumulative_process_wall_seconds"] - walls[0],
                     "first_s": walls[0], "steady_s": walls[1:].mean() if len(walls) > 1 else np.nan,
                     "steady_sd_s": walls[1:].std(ddof=1) if len(walls) > 2 else np.nan,
                     "steady_first_half_s": walls[1:10].mean() if len(walls) > 10 else np.nan,
                     "steady_second_half_s": walls[10:].mean() if len(walls) > 10 else np.nan,
                     "peak_reserved_gb": max(b["pytorch_peak_reserved_bytes"] for b in batches) / 1e9})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    parser.add_argument("--new-scores", type=Path, required=True, help="C3 scores.csv")
    parser.add_argument("--archived-scores", type=Path, help="C2 scores.csv")
    parser.add_argument("--g1-run", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--marginal-dir", type=Path, default=ROOT / "results/paper_v1/marginal")
    parser.add_argument("--operating-points", type=Path,
                        default=ROOT / "results/paper_v1/tables/stopping_operating_points.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/seedext_v1")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    systems = {s["system_id"]: s for s in config["systems"]}
    chains = proper_chains(args.annotations)
    ops = pd.read_csv(args.operating_points)
    tau = float(ops[(ops["method"] == "chai") & (ops["family"] == "conf")]["param_tau"].iloc[0])
    cuts = pd.read_csv(args.marginal_dir / "confidence_tiers.csv").set_index("method").loc["chai"]
    first_cuts = np.array([cuts["low_mid_cut"], cuts["mid_high_cut"]])

    new = candidate_quality(pd.read_csv(args.new_scores), chains)
    per_system, arrays = [], {}
    for system_id, row in systems.items():
        a = system_seed_arrays(new[new["system_id"] == system_id], row["seeds"])
        rec = {"system_id": system_id, "tier": row["tier"], "cluster": row["cluster"],
               "padded_tokens": row["padded_tokens"], "complete": a is not None}
        if a is not None:
            arrays[system_id] = a
            c = curves(a)
            rec.update({name: c[name] for name in OUTCOMES})
            rec["rmsd_ok"] = a["rmsd_ok"]
        per_system.append(rec)
    per = pd.DataFrame(per_system)
    done = per[per["complete"]].copy()
    n_complete = int(len(done))
    summary = {"n_systems": len(per), "n_complete": n_complete,
               "complete_by_tier": done["tier"].value_counts().to_dict(),
               "primary_analysis": n_complete >= MIN_COMPLETE}
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for name in OUTCOMES:
        (m, lo, hi, se), tiers = stratified_bootstrap(done, name, rng)
        for k in range(N_SEEDS):
            rows.append({"outcome": name, "tier": "pool", "kind": "level", "k": k + 1, "mean": m[k], "lo": lo[k],
                         "hi": hi[k], "n": n_complete, "se": se[k]})
            for t, (tm, tlo, thi, tn, tse) in tiers.items():
                rows.append({"outcome": name, "tier": t, "kind": "level", "k": k + 1, "mean": tm[k], "lo": tlo[k],
                             "hi": thi[k], "n": tn, "se": tse[k]})
        done[f"{name}_gain"] = [np.diff(v) for v in done[name]]
        (m, lo, hi, se), tiers = stratified_bootstrap(done, f"{name}_gain", rng)
        for k in range(N_SEEDS - 1):
            rows.append({"outcome": name, "tier": "pool", "kind": "gain", "k": k + 2, "mean": m[k], "lo": lo[k],
                         "hi": hi[k], "n": n_complete, "se": se[k]})
            for t, (tm, tlo, thi, tn, tse) in tiers.items():
                rows.append({"outcome": name, "tier": t, "kind": "gain", "k": k + 2, "mean": tm[k], "lo": tlo[k],
                             "hi": thi[k], "n": tn, "se": tse[k]})
        done[f"{name}_d5_20"] = [np.array([v[N_SEEDS - 1] - v[4]]) for v in done[name]]
        (m, lo, hi, se), tiers = stratified_bootstrap(done, f"{name}_d5_20", rng)
        summary[f"delta_5_20_{name}"] = {"pool": [m[0], lo[0], hi[0]],
                                          **{t: [v[0][0], v[1][0], v[2][0], v[3]] for t, v in tiers.items()}}
        summary[f"se_5_20_{name}"] = {"pool": se[0], **{t: v[4][0] for t, v in tiers.items()}}
    curve = pd.DataFrame(rows)
    d = summary["delta_5_20_ret_q"]["pool"]
    summary["primary"] = {"delta_ret_5_20": d[0], "lo": d[1], "hi": d[2], "margin": MARGIN,
                          "reading": reading(*d) if summary["primary_analysis"] else "not evaluated (incomplete)"}

    kstar_rows = []
    gains = curve[curve["kind"] == "gain"]
    for (outcome, tier), g in gains[gains["outcome"].isin(["ret_q", "orc_q"])].groupby(["outcome", "tier"]):
        for t in THRESHOLDS:
            k, ok = k_star(g.sort_values("k")["mean"].to_numpy(), t)
            kstar_rows.append({"scheme": "design", "outcome": outcome, "tier": tier, "threshold": t,
                               "k_star": k, "reached": ok})
    # First-seed confidence tiers (paper cut points), systems weighted to the pool composition.
    weights = {t: (1 / 3) / max(int((done["tier"] == t).sum()), 1) for t in TIERS}
    acc = {(o, t): [np.zeros(N_SEEDS - 1), 0.0] for o in ("ret_q", "orc_q") for t in TIERS}
    for system_id in done["system_id"]:
        a = arrays[system_id]
        w = weights[systems[system_id]["tier"]] / N_SEEDS
        fc = first_seed_curves(a)
        tier_idx = np.digitize(a["S"], first_cuts)
        for o in ("ret_q", "orc_q"):
            for j in range(N_SEEDS):
                key = (o, TIERS[tier_idx[j]])
                acc[key][0] += w * np.diff(fc[o][j])
                acc[key][1] += w
    first_rows = []
    for (o, t), (total, weight) in acc.items():
        if weight == 0:
            continue
        g = total / weight
        for k in range(N_SEEDS - 1):
            first_rows.append({"outcome": o, "tier": t, "k": k + 2, "gain": g[k], "weight": weight})
        for thr in THRESHOLDS:
            k, ok = k_star(g, thr)
            kstar_rows.append({"scheme": "first_seed", "outcome": o, "tier": t, "threshold": thr,
                               "k_star": k, "reached": ok})

    # Ranking over all 100 candidates of a system.
    rank_rows = []
    for system_id in done["system_id"]:
        a = arrays[system_id]
        s, q = a["scores"].ravel(), a["q"].ravel()
        best = q.max()
        rank_rows.append({"system_id": system_id, "tier": systems[system_id]["tier"],
                          "spearman": pd.Series(s).rank().corr(pd.Series(q).rank()),  # Spearman via average ranks
                          "top_near_best": float(q[rank_order(s)[0]] >= best - 0.01),
                          "random_near_best": float(np.mean(q >= best - 0.01)),
                          "spread": float(best - q.min())})
    ranking = pd.DataFrame(rank_rows)
    summary["ranking"] = {t: ranking[ranking["tier"] == t][["spearman", "top_near_best", "random_near_best"]]
                          .mean().to_dict() for t in TIERS}

    rule_rng = np.random.default_rng(RULE_SEED)
    rule = []
    for system_id in done["system_id"]:
        sim = rule_simulation(arrays[system_id], tau, rule_rng)
        for cap, v in sim.items():
            rule.append({"system_id": system_id, "tier": systems[system_id]["tier"], "cap": cap, **v})
    rule = pd.DataFrame(rule)
    summary["confidence_rule"] = {"tau": tau, **{
        f"cap_{cap}": rule[rule["cap"] == cap].groupby("tier")[["mean_seeds", "ret_rule", "ret_fixed",
                                                                "positive_loss"]].mean().to_dict("index")
        for cap in (5, N_SEEDS)}}

    # RNP labels and archived ranking scores.
    meta, proper, preds = ev.load_inputs(args.predictions, args.annotations)
    rnp = ev.candidate_table(preds["chai"], proper, meta)
    rnp = rnp[rnp["target"].isin(systems)][["target", "seed", "sample", "score", "q"]]
    rnp = rnp.rename(columns={"target": "system_id", "score": "rnp_score", "q": "rnp_q"})
    if args.archived_scores is not None and args.archived_scores.is_file():
        arch = candidate_quality(pd.read_csv(args.archived_scores), chains)
        par = arch.merge(rnp, on=["system_id", "seed", "sample"], how="left")
        diff = (par["q"] - par["rnp_q"]).dropna()
        summary["parity"] = {"n_candidates": int(len(par)), "n_compared": int(len(diff)),
                             "share_within": float((diff.abs() < PARITY_WITHIN).mean()) if len(diff) else None,
                             "mean_diff": float(diff.mean()) if len(diff) else None,
                             "median_abs_diff": float(diff.abs().median()) if len(diff) else None}
        summary["parity"]["passed"] = bool(len(diff) and summary["parity"]["share_within"] >= PARITY_SHARE
                                           and abs(summary["parity"]["mean_diff"]) <= PARITY_MEAN)
        par.to_csv(args.output_dir / "parity.csv", index=False)
    rep = new.merge(rnp, on=["system_id", "seed", "sample"], how="inner")
    if len(rep):
        dq = (rep["q"] - rep["rnp_q"]).dropna()
        summary["replication"] = {
            "n_candidates": int(len(rep)), "score_corr": float(rep["ranking_score"].corr(rep["rnp_score"])),
            "score_median_abs_diff": float((rep["ranking_score"] - rep["rnp_score"]).abs().median()),
            "q_median_abs_diff": float(dq.abs().median()), "q_share_within_0.01": float((dq.abs() < 0.01).mean()),
            "q_mean_diff": float(dq.mean())}
        rep.to_csv(args.output_dir / "replication.csv", index=False)

    timing = timing_rows(args.g1_run, config)
    if len(timing):
        summary["timing"] = timing.groupby("padded_tokens")[["steady_s", "steady_sd_s", "first_s", "startup_s"]] \
            .mean().round(2).to_dict("index")
        summary["timing_gpus"] = timing["gpu"].value_counts().to_dict()
        timing.to_csv(args.output_dir / "timing.csv", index=False)

    flat = done[["system_id", "tier", "cluster", "padded_tokens"]].copy()
    for name in OUTCOMES:
        for k in (1, 5, N_SEEDS):
            flat[f"{name}_{k}"] = [v[k - 1] for v in done[name]]
    flat.to_csv(args.output_dir / "per_system.csv", index=False)
    per[["system_id", "tier", "complete"]].to_csv(args.output_dir / "completeness.csv", index=False)
    curve.to_csv(args.output_dir / "curves.csv", index=False)
    pd.DataFrame(first_rows).to_csv(args.output_dir / "first_seed_gains.csv", index=False)
    pd.DataFrame(kstar_rows).to_csv(args.output_dir / "kstar.csv", index=False)
    ranking.to_csv(args.output_dir / "ranking.csv", index=False)
    rule.to_csv(args.output_dir / "rule.csv", index=False)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "configs"],
                           capture_output=True, text=True).stdout.strip()
    summary.update({"script": "scripts/seedext_v1_eval.py", "git_commit": commit, "scripts_dirty": bool(dirty),
                    "new_scores": str(args.new_scores), "archived_scores": str(args.archived_scores),
                    "g1_run": str(args.g1_run), "bootstrap": {"reps": BOOT, "seed": BOOT_SEED},
                    "rule_permutations": RULE_PERMUTATIONS, "wall_seconds": time.perf_counter() - started})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_complete", "primary_analysis", "primary")}, indent=2, default=float))


if __name__ == "__main__":
    main()
