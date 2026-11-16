#!/usr/bin/env python3
"""E25 (seed extension v2): frozen analysis of Protenix or Boltz-2 with 20 seeds per system.

The E24 analysis (scripts/seedext_v1_eval.py, whose expectation, bootstrap, k* and rule
functions are imported unchanged) applied to one method of configs/seedext_v2_systems.json,
following docs/seedext_v2_protocol.md, fixed before any E25 result existed:

* Quality of a candidate: LDDT-PLI of the proper ligand; success LDDT-PLI >= 0.8, RMSD < 2 A.
* Complete system: 20 seeds and 100 candidates with a valid quality and ranking score. The
  primary analysis needs at least 90% complete systems (ceil(0.9 n)).
* Primary: delta_ret(5->20) with the design tiers weighted equally, 95% percentile interval
  from a cluster bootstrap stratified by tier (2,000 draws, seed 20261024). A: upper bound
  < 0.02; B: estimate >= 0.02 and lower bound > 0; C: otherwise.
* Secondary: delta_orc(5->20); per-tier deltas by design tier and by the method's own tier;
  gains of seeds 2..20 and k* (t = 0.005; 0.002 and 0.01) per tier and overall, and by the
  paper's first-seed confidence tiers of the method; success rates; ranking over 100
  candidates; the paper's frozen confidence rule of the method with caps of 5 and 20 seeds;
  timing per seed by padded size from seedext_v2_run.py profiles.
* Scoring parity (C2) and replication of the archived seeds, against the RNP table of the method.

Run with the Chai environment (numpy, pandas):
  python -I scripts/seedext_v2_eval.py --method protenix ...
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_v1_budget_eval as ev  # noqa: E402
import seedext_v1_eval as v1  # noqa: E402
from seedext_v1_eval import (BOOT, BOOT_SEED, MARGIN, N_SEEDS, OUTCOMES, PARITY_MEAN, PARITY_SHARE,  # noqa: E402
                             PARITY_WITHIN, RULE_PERMUTATIONS, RULE_SEED, THRESHOLDS, TIERS, candidate_quality,
                             curves, first_seed_curves, k_star, proper_chains, rank_order, reading,
                             rule_simulation, stratified_bootstrap, system_seed_arrays)

COMPLETE_SHARE = 0.9


def timing_rows(g_run: Path, systems: dict, method: str | None = None) -> pd.DataFrame:
    """Per-system timing from seedext_v2_run.py profiles (task_*/<system>/profile.jsonl).

    With a method, profiles of other methods are skipped: the G run holds Protenix and Boltz-2
    tasks, and their 31 shared systems would otherwise enter each other's timing.
    """
    rows = []
    for profile in sorted(g_run.glob("task_*/*/profile.jsonl")):
        system = profile.parent.name
        events = [json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
        start = next((e for e in events if e["event"] == "start"), {})
        if method is not None and start.get("method", method) != method:  # profiles without a method are kept
            continue
        batches = [e for e in events if e["event"] == "batch" and e["status"] == "completed"]
        if not batches:
            continue
        walls = np.array([b["wall_seconds"] for b in batches])
        rows.append({"system_id": system, "task": profile.parent.parent.name, "gpu": start.get("gpu_name"),
                     "package_version": start.get("package_version"),
                     "padded_tokens": systems.get(system, {}).get("padded_tokens"), "n_seeds": len(walls),
                     "startup_s": batches[0]["cumulative_process_wall_seconds"] - walls[0],
                     "first_s": walls[0], "steady_s": walls[1:].mean() if len(walls) > 1 else np.nan,
                     "steady_sd_s": walls[1:].std(ddof=1) if len(walls) > 2 else np.nan,
                     "steady_first_half_s": walls[1:10].mean() if len(walls) > 10 else np.nan,
                     "steady_second_half_s": walls[10:].mean() if len(walls) > 10 else np.nan,
                     "peak_gpu_memory_gb": max(b.get("peak_gpu_memory_used_mib", 0) for b in batches) * 2**20 / 1e9})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["protenix", "boltz2", "boltz"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seedext_v2_systems.json")
    parser.add_argument("--new-scores", type=Path, required=True, help="C3 scores.csv of the method")
    parser.add_argument("--archived-scores", type=Path, help="C2 scores.csv of the method")
    parser.add_argument("--g-run", type=Path, required=True, help="G run directory (task_*/<system>/...)")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--marginal-dir", type=Path, default=ROOT / "results/paper_v1/marginal")
    parser.add_argument("--operating-points", type=Path,
                        default=ROOT / "results/paper_v1/tables/stopping_operating_points.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir or ROOT / "results/seedext_v2" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    block = json.loads(args.config.read_text(encoding="utf-8"))["methods"][args.method]
    systems = {s["system_id"]: s for s in block["systems"]}
    min_complete = math.ceil(COMPLETE_SHARE * len(systems))
    chains = proper_chains(args.annotations)
    ops = pd.read_csv(args.operating_points)
    tau = float(ops[(ops["method"] == args.method) & (ops["family"] == "conf")]["param_tau"].iloc[0])
    cuts = pd.read_csv(args.marginal_dir / "confidence_tiers.csv").set_index("method").loc[args.method]
    first_cuts = np.array([cuts["low_mid_cut"], cuts["mid_high_cut"]])

    new = candidate_quality(pd.read_csv(args.new_scores), chains)
    per_system, arrays = [], {}
    for system_id, row in systems.items():
        a = system_seed_arrays(new[new["system_id"] == system_id], row["seeds"])
        rec = {"system_id": system_id, "tier": row["tier"], "own_tier": row["own_tier"], "cluster": row["cluster"],
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
    summary = {"method": args.method, "n_systems": len(per), "n_complete": n_complete, "min_complete": min_complete,
               "complete_by_tier": done["tier"].value_counts().to_dict(),
               "complete_by_own_tier": done["own_tier"].value_counts().to_dict(),
               "primary_analysis": n_complete >= min_complete, "design": block["design"]}
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

    # Secondary: the same deltas with the method's own confidence tiers (same bootstrap stream continues).
    own = done.copy()
    own["tier"] = own["own_tier"]
    own_rows = []
    for name in ("ret_q", "orc_q"):
        (m, lo, hi, se), tiers = stratified_bootstrap(own, f"{name}_d5_20", rng)
        summary[f"own_tiers_delta_5_20_{name}"] = {"pool": [m[0], lo[0], hi[0], se[0]],
                                                    **{t: [v[0][0], v[1][0], v[2][0], v[3], v[4][0]] for t, v in tiers.items()}}
        (m, lo, hi, se), tiers = stratified_bootstrap(own, f"{name}_gain", rng)
        for k in range(N_SEEDS - 1):
            own_rows.append({"outcome": name, "tier": "pool", "kind": "gain", "k": k + 2, "mean": m[k], "lo": lo[k],
                             "hi": hi[k], "n": n_complete, "se": se[k]})
            for t, (tm, tlo, thi, tn, tse) in tiers.items():
                own_rows.append({"outcome": name, "tier": t, "kind": "gain", "k": k + 2, "mean": tm[k], "lo": tlo[k],
                                 "hi": thi[k], "n": tn, "se": tse[k]})
    own_curve = pd.DataFrame(own_rows)

    kstar_rows = []
    for scheme, frame in (("design", curve), ("own", own_curve)):
        gains = frame[frame["kind"] == "gain"]
        for (outcome, tier), g in gains[gains["outcome"].isin(["ret_q", "orc_q"])].groupby(["outcome", "tier"]):
            for t in THRESHOLDS:
                k, ok = k_star(g.sort_values("k")["mean"].to_numpy(), t)
                kstar_rows.append({"scheme": scheme, "outcome": outcome, "tier": tier, "threshold": t,
                                   "k_star": k, "reached": ok})
    # First-seed confidence tiers (paper cut points of the method), systems weighted to the pool composition.
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

    rank_rows = []
    for system_id in done["system_id"]:
        a = arrays[system_id]
        s, q = a["scores"].ravel(), a["q"].ravel()
        best = q.max()
        rank_rows.append({"system_id": system_id, "tier": systems[system_id]["tier"],
                          "own_tier": systems[system_id]["own_tier"],
                          "spearman": pd.Series(s).rank().corr(pd.Series(q).rank()),
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

    meta, proper, preds = ev.load_inputs(args.predictions, args.annotations)
    rnp = ev.candidate_table(preds[args.method], proper, meta)
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
        par.to_csv(output_dir / "parity.csv", index=False)
    rep = new.merge(rnp, on=["system_id", "seed", "sample"], how="inner")
    if len(rep):
        dq = (rep["q"] - rep["rnp_q"]).dropna()
        summary["replication"] = {
            "n_candidates": int(len(rep)), "score_corr": float(rep["ranking_score"].corr(rep["rnp_score"])),
            "score_median_abs_diff": float((rep["ranking_score"] - rep["rnp_score"]).abs().median()),
            "q_median_abs_diff": float(dq.abs().median()), "q_share_within_0.01": float((dq.abs() < 0.01).mean()),
            "q_mean_diff": float(dq.mean())}
        rep.to_csv(output_dir / "replication.csv", index=False)

    timing = timing_rows(args.g_run, systems, args.method)
    if len(timing):
        summary["timing"] = timing.groupby("padded_tokens")[["steady_s", "steady_sd_s", "first_s", "startup_s",
                                                             "peak_gpu_memory_gb"]].mean().round(2).to_dict("index")
        summary["timing_gpus"] = timing["gpu"].value_counts().to_dict()
        summary["package_versions"] = timing["package_version"].value_counts().to_dict()
        timing.to_csv(output_dir / "timing.csv", index=False)

    flat = done[["system_id", "tier", "own_tier", "cluster", "padded_tokens"]].copy()
    for name in OUTCOMES:
        for k in (1, 5, N_SEEDS):
            flat[f"{name}_{k}"] = [v[k - 1] for v in done[name]]
    flat.to_csv(output_dir / "per_system.csv", index=False)
    per[["system_id", "tier", "own_tier", "complete"]].to_csv(output_dir / "completeness.csv", index=False)
    curve.to_csv(output_dir / "curves.csv", index=False)
    own_curve.to_csv(output_dir / "curves_own_tiers.csv", index=False)
    pd.DataFrame(first_rows).to_csv(output_dir / "first_seed_gains.csv", index=False)
    pd.DataFrame(kstar_rows).to_csv(output_dir / "kstar.csv", index=False)
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    rule.to_csv(output_dir / "rule.csv", index=False)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts", "configs"],
                           capture_output=True, text=True).stdout.strip()
    summary.update({"script": "scripts/seedext_v2_eval.py", "v1_script": v1.__file__, "git_commit": commit,
                    "scripts_dirty": bool(dirty), "new_scores": str(args.new_scores),
                    "archived_scores": str(args.archived_scores), "g_run": str(args.g_run),
                    "bootstrap": {"reps": BOOT, "seed": BOOT_SEED}, "rule_permutations": RULE_PERMUTATIONS,
                    "wall_seconds": time.perf_counter() - started})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("method", "n_complete", "min_complete", "primary_analysis", "primary")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
