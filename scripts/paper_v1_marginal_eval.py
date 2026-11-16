#!/usr/bin/env python3
"""Paper v1 follow-up: marginal value of each seed by model and difficulty.

Post hoc and descriptive; nothing is
fitted and no threshold is tuned. For every co-folding variant and complete
system, the LDDT-PLI added by the k-th seed (k = 2..5) to the returned (native
top-1) and the best available (oracle) pose is averaged over the 120 seed
orderings, overall and in two difficulty strata:

* first-seed confidence: each (system, ordering) pair is placed in a tertile of
  the variant's seed-level top ranking scores by the score of its first seed,
  which is known after one seed;
* RNP novelty bin, which uses the reference pocket and is descriptive only.

Marginal-gain threshold budget k* for a threshold t: the smallest k (1-4) after which the
next seed adds less than t on average; k* = 5 with reached = False means the
fifth seed still added at least t. t = 0.005 LDDT-PLI is the primary threshold,
0.002 and 0.01 are sensitivity values (sensitivity analyses).

Chai-1 cost uses the E22 measurements (results/paper_v1/tables/timing_systems.csv):
T(k) = T0 + k * Tbar, with Tbar the steady per-seed time of the system's padded
input size and T0 the mean extra time of the warm processes (start-up plus the
excess of the first seed). Systems of a padded size that was not timed
(> 768 tokens) are left out of the cost table. The frozen confidence rules of
paper v1 are applied per stratum as reference points.

Run with the Chai environment, which provides RDKit:
  /usr/bin/python3.12 -S scripts/chai_site_bootstrap.py scripts/paper_v1_marginal_eval.py
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_v1_budget_eval as ev  # noqa: E402

THRESHOLDS = (0.002, 0.005, 0.01)
TIERS = ("low", "mid", "high")
CHAI_PAD_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]  # chai_lab 0.6.1 model sizes
OUTCOMES = {"returned": "ret_q", "oracle": "orc_q", "returned_success": "ret_s", "oracle_success": "orc_s"}
BOOT_SEED = 20261017


def confidence_cuts(arr: dict) -> np.ndarray:
    """Tertile cut points of the variant's seed-level top ranking scores."""
    return np.quantile(arr["S"].ravel(), [1 / 3, 2 / 3])


def strata(paths: dict, sysmeta: pd.DataFrame, cuts: np.ndarray):
    """Yield (stratifier, tier, mask) with a systems x orderings membership mask."""
    n, m = paths["cms"].shape[:2]
    yield "all", "all", np.ones((n, m), dtype=bool)
    tier = np.digitize(paths["cms"][:, :, 0], cuts)  # first-seed top score: 0 low, 1 mid, 2 high
    for i, name in enumerate(TIERS):
        yield "confidence", name, tier == i
    for label in ev.NOVELTY_LABELS:
        mask = (sysmeta["novelty_bin"] == label).to_numpy()
        yield "novelty", str(label), np.repeat(mask[:, None], m, axis=1)


def ratio_with_ci(sums: np.ndarray, counts: np.ndarray, weights: np.ndarray):
    """Ratio of summed values to summed counts, with percentile CIs from cluster weights."""
    mean = sums.sum(axis=0) / counts.sum()
    boot = (weights @ sums) / (weights @ counts)[:, None]
    return mean, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def summary_rows(method, paths, sysmeta, cuts, rng):
    clusters = sysmeta["cluster"].to_numpy()
    labels, inverse = np.unique(clusters, return_inverse=True)
    draws = rng.integers(0, len(labels), size=(ev.BOOT, len(labels)))
    weights = np.zeros((ev.BOOT, len(labels)))
    np.add.at(weights, (np.arange(ev.BOOT)[:, None], draws), 1.0)
    rows = []
    for stratifier, tier, mask in strata(paths, sysmeta, cuts):
        count_sys = mask.sum(axis=1).astype(float)
        if count_sys.sum() == 0:
            continue
        counts = np.bincount(inverse, weights=count_sys, minlength=len(labels))
        base = {"method": method, "stratifier": stratifier, "tier": tier,
                "n_systems": int((count_sys > 0).sum()), "n_pairs": int(count_sys.sum()),
                "n_clusters": int((counts > 0).sum())}
        for outcome, key in OUTCOMES.items():
            level = paths[key].astype(float)
            for kind, values, ks in (("level", level, range(1, 6)), ("increment", np.diff(level, axis=2), range(2, 6))):
                per_sys = (values * mask[:, :, None]).sum(axis=1)  # systems x K
                sums = np.zeros((len(labels), per_sys.shape[1]))
                np.add.at(sums, inverse, per_sys)
                mean, lo, hi = ratio_with_ci(sums, counts, weights)
                for j, k in enumerate(ks):
                    rows.append({**base, "outcome": outcome, "kind": kind, "k": k,
                                 "mean": mean[j], "lo": lo[j], "hi": hi[j]})
    return rows


def recommended_seeds(marginal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc = marginal[(marginal["kind"] == "increment") & marginal["outcome"].isin(["returned", "oracle"])]
    for key, g in inc.groupby(["method", "stratifier", "tier", "outcome"], sort=False):
        gains = g.sort_values("k")["mean"].to_numpy()  # seeds 2..5
        for t in THRESHOLDS:
            below = np.flatnonzero(gains < t)
            reached = below.size > 0
            k_star = int(below[0]) + 1 if reached else 5
            rows.append(dict(zip(["method", "stratifier", "tier", "outcome"], key), threshold=t, k_star=k_star,
                             reached=reached, next_gain=float(gains[below[0]]) if reached else np.nan))
    return pd.DataFrame(rows)


def rule_rows(method, paths, sysmeta, cuts, tau):
    """Frozen paper v1 confidence rule applied within each stratum (all splits, descriptive)."""
    stop = ev.stop_index(paths["cms"], tau=tau)  # 0-based index of the last seed run
    ret_stop = np.take_along_axis(paths["ret_q"], stop[..., None], axis=2)[..., 0]
    ret5 = paths["ret_q"][:, :, 4]
    rows = []
    for stratifier, tier, mask in strata(paths, sysmeta, cuts):
        if not mask.any():
            continue
        rows.append({"method": method, "stratifier": stratifier, "tier": tier, "tau": tau,
                     "n_pairs": int(mask.sum()), "mean_seeds": float((stop + 1)[mask].mean()),
                     "share_one_seed": float((stop == 0)[mask].mean()),
                     "share_five_seeds": float((stop == 4)[mask].mean()),
                     "ret_rule": float(ret_stop[mask].mean()), "ret_five": float(ret5[mask].mean()),
                     "positive_loss": float(np.maximum(ret5 - ret_stop, 0)[mask].mean())})
    return rows


def chai_tokens(inputs: dict, system_id: str) -> float:
    record = inputs.get(system_id, {})
    n = sum(len(seq.replace("-", "")) for seq in record.get("sequences", {}).values())
    for smiles in record.get("smiles", []):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        n += mol.GetNumHeavyAtoms()
    return float(n) if n else np.nan


def cost_rows(arr, paths, sysmeta, cuts, tau, inputs, timing):
    warm = timing[timing["overhead_in_batches"] < 1]
    t0 = float((warm["startup_s"] + warm["first_call_s"] - warm["steady_batch_s"]).mean())
    tbar = timing.groupby("padded_tokens")["steady_batch_s"].mean().to_dict()
    tokens = np.array([chai_tokens(inputs, s) for s in arr["systems"]])
    pad = np.array([min((p for p in CHAI_PAD_SIZES if p >= t), default=np.nan) if np.isfinite(t) else np.nan
                    for t in tokens])
    per_seed = np.array([tbar.get(p, np.nan) for p in pad])
    timed = np.isfinite(per_seed)
    stop = ev.stop_index(paths["cms"], tau=tau)
    ret_stop = np.take_along_axis(paths["ret_q"], stop[..., None], axis=2)[..., 0]
    rows = []
    for stratifier, tier, mask in strata(paths, sysmeta, cuts):
        if stratifier == "novelty":
            continue
        mask = mask & timed[:, None]
        w = mask.sum(axis=1).astype(float)
        if w.sum() == 0:
            continue
        base = {"stratifier": stratifier, "tier": tier, "n_systems": int((w > 0).sum()),
                "n_untimed_systems": int((~timed).sum()), "t0_s": t0}
        for k in range(1, 6):
            minutes = (t0 + k * per_seed[timed]) / 60
            rows.append({**base, "point": f"k={k}", "seeds": float(k),
                         "minutes": float((w[timed] * minutes).sum() / w.sum()),
                         "ret_q": float(paths["ret_q"][:, :, k - 1][mask].mean()),
                         "orc_q": float(paths["orc_q"][:, :, k - 1][mask].mean())})
        seeds = (stop + 1).astype(float)
        minutes = (t0 + seeds * per_seed[:, None]) / 60
        rows.append({**base, "point": "confidence rule", "seeds": float(seeds[mask].mean()),
                     "minutes": float(minutes[mask].mean()), "ret_q": float(ret_stop[mask].mean()),
                     "orc_q": np.nan, "tau": tau})
    pads = pd.Series(pad[np.isfinite(pad)]).value_counts().sort_index()
    return rows, {"t0_s": t0, "per_seed_s_by_pad": tbar, "systems_by_pad": {int(k): int(v) for k, v in pads.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--tables", type=Path, default=ROOT / "results/paper_v1/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/marginal")
    args = parser.parse_args()
    started = time.perf_counter()
    RDLogger.DisableLog("rdApp.*")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    meta, proper, preds = ev.load_inputs(args.predictions, args.annotations)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    timing = pd.read_csv(args.tables / "timing_systems.csv")
    ops = pd.read_csv(args.tables / "stopping_operating_points.csv")
    taus = ops[ops["family"] == "conf"].set_index("method")["param_tau"].to_dict()
    marginal, rules, cuts_rows, cost, cost_info = [], [], [], [], {}
    for method in ev.METHODS:
        cand = ev.candidate_table(preds[method], proper, meta)
        arr = ev.system_arrays(cand, ev.complete_systems(cand))
        sysmeta = meta.loc[arr["systems"]]
        paths = ev.permutation_paths(arr)
        cuts = confidence_cuts(arr)
        cuts_rows.append({"method": method, "low_mid_cut": cuts[0], "mid_high_cut": cuts[1], "tau": taus[method]})
        marginal += summary_rows(method, paths, sysmeta, cuts, rng)
        rules += rule_rows(method, paths, sysmeta, cuts, taus[method])
        if method == "chai":
            cost, cost_info = cost_rows(arr, paths, sysmeta, cuts, taus[method], inputs, timing)
        print(f"{method}: {len(arr['systems'])} complete systems", flush=True)
    marginal = pd.DataFrame(marginal)
    outputs = {
        "marginal_gains.csv": marginal, "recommended_seeds.csv": recommended_seeds(marginal),
        "confidence_tiers.csv": pd.DataFrame(cuts_rows), "rule_by_tier.csv": pd.DataFrame(rules),
        "chai_cost.csv": pd.DataFrame(cost),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "scripts"],
                           capture_output=True, text=True).stdout.strip()
    receipt = {
        "script": "scripts/paper_v1_marginal_eval.py", "git_commit": commit, "scripts_dirty": bool(dirty),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "predictions_sha256": ev.sha256_file(args.predictions), "annotations_sha256": ev.sha256_file(args.annotations),
        "inputs_sha256": ev.sha256_file(args.inputs),
        "timing_systems_sha256": ev.sha256_file(args.tables / "timing_systems.csv"),
        "operating_points_sha256": ev.sha256_file(args.tables / "stopping_operating_points.csv"),
        "thresholds": list(THRESHOLDS), "primary_threshold": 0.005,
        "bootstrap_reps": ev.BOOT, "bootstrap_seed": BOOT_SEED, "chai_cost": cost_info,
        "wall_seconds": time.perf_counter() - started, "outputs": sorted(outputs),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("git_commit", "scripts_dirty", "wall_seconds", "chai_cost")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
