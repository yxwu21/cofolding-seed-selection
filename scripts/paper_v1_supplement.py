#!/usr/bin/env python3
"""Paper v1 supplementary analyses (descriptive; no thresholds are tuned).

Adds to the frozen RQ1-RQ3 tables:
  * paired budget gains (k=5 minus k=1, k=2 minus k=1) with cluster-bootstrap CIs;
  * the protocol's ascending-seed-order sensitivity for the budget curves;
  * native-ranking diagnostics over all 25 candidates of a system;
  * candidate-quality spread, which bounds what extra sampling could offer.
Uses its own bootstrap seed so the frozen tables are unaffected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_v1_budget_eval import (  # noqa: E402
    LABELS, METHODS, candidate_table, cluster_bootstrap, complete_systems,
    load_inputs, permutation_paths, system_arrays,
)

SEED = 20261016


def row_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks within each row (ties share their mean rank)."""
    return pd.DataFrame(a).rank(axis=1, method="average").to_numpy()


def canonical_paths(arr):
    key = arr["S"] - np.arange(5)[None, :] * 1e-12
    idx = np.zeros(len(key), dtype=int)
    chosen = np.zeros(key.shape, dtype=int)
    for t in range(5):
        if t:
            idx = np.where(key[:, t] > key[np.arange(len(key)), idx], t, idx)
        chosen[:, t] = idx
    return {
        "ret_q": np.take_along_axis(arr["QT"], chosen, axis=1),
        "ret_s": np.take_along_axis(arr["SU"], chosen, axis=1).astype(float),
        "orc_q": np.maximum.accumulate(arr["O"], axis=1),
        "orc_s": np.logical_or.accumulate(arr["OS"], axis=1).astype(float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/tables")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    meta, proper, preds = load_inputs(ROOT / "data/raw/runs_n_poses_predictions.tar.gz",
                                      ROOT / "data/raw/runs_n_poses_annotations.csv")
    gains, canon, ranking = [], [], []
    for method in METHODS:
        cand = candidate_table(preds[method], proper, meta)
        arr = system_arrays(cand, complete_systems(cand))
        sysmeta = meta.loc[arr["systems"]]
        paths = permutation_paths(arr)
        split = sysmeta["split"].to_numpy()
        clusters = sysmeta["cluster"].to_numpy()
        for scope in ("all", "test"):
            mask = np.ones(len(split), bool) if scope == "all" else split == "test"
            for name in ("ret_q", "orc_q", "ret_s", "orc_s"):
                per = paths[name][mask].mean(axis=1)  # systems x k
                for (a, b) in ((1, 2), (1, 3), (1, 4), (1, 5), (2, 5)):
                    diff = per[:, b - 1] - per[:, a - 1]
                    mean, lo, hi, se = cluster_bootstrap(diff, clusters[mask], rng, with_se=True)
                    gains.append({"method": method, "scope": scope, "quantity": name, "from_k": a, "to_k": b,
                                  "n_systems": int(mask.sum()), "mean": mean, "lo": lo, "hi": hi,
                                  "sd": float(np.std(diff, ddof=1)), "se": se})
            # cluster-equal weighting of the 1->5 returned gain (sensitivity)
            per = paths["ret_q"][mask].mean(axis=1)
            diff = pd.Series(per[:, 4] - per[:, 0]).groupby(clusters[mask]).mean()
            gains.append({"method": method, "scope": scope, "quantity": "ret_q_cluster_weighted", "from_k": 1,
                          "to_k": 5, "n_systems": int(mask.sum()), "mean": float(diff.mean()),
                          "lo": float("nan"), "hi": float("nan"), "sd": float("nan"), "se": float("nan")})
        cp = canonical_paths(arr)
        for k in range(1, 6):
            canon.append({"method": method, "k": k, "order": "ascending_seed",
                          **{name: float(cp[name][:, k - 1].mean()) for name in cp},
                          "perm_ret_q": float(paths["ret_q"][:, :, k - 1].mean()),
                          "perm_orc_q": float(paths["orc_q"][:, :, k - 1].mean())})
        cs = arr["cs"].reshape(len(arr["cs"]), -1)
        cq = arr["cq"].reshape(len(arr["cq"]), -1)
        rs, rq = row_ranks(cs), row_ranks(cq)
        rs -= rs.mean(axis=1, keepdims=True)
        rq -= rq.mean(axis=1, keepdims=True)
        denom = np.sqrt((rs ** 2).sum(axis=1) * (rq ** 2).sum(axis=1))
        rho = np.where(denom > 0, (rs * rq).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)  # NaN if constant
        top = (cs - np.arange(25)[None, :] * 1e-12).argmax(axis=1)
        q_top = cq[np.arange(len(cq)), top]
        q_best, q_worst = cq.max(axis=1), cq.min(axis=1)
        spread = q_best - q_worst > 0.1
        ranking.append({
            "method": method, "label": LABELS[method], "n_systems": len(cq),
            "mean_spearman_score_vs_lddt": float(np.nanmean(rho)),
            "mean_spearman_if_spread_gt_0.1": float(np.nanmean(rho[spread])),
            "n_spread_gt_0.1": int(spread.sum()),
            "frac_top1_within_0.01_of_best25_if_spread": float((q_best - q_top <= 0.01)[spread].mean()),
            "median_spearman_score_vs_lddt": float(np.nanmedian(rho)),
            "frac_top1_within_0.01_of_best25": float((q_best - q_top <= 0.01).mean()),
            "frac_best25_minus_top1_gt_0.1": float((q_best - q_top > 0.1).mean()),
            "frac_spread_gt_0.1": float((q_best - q_worst > 0.1).mean()),
            "frac_all25_success": float(arr["csu"].reshape(len(cq), -1).all(axis=1).mean()),
            "frac_no25_success": float((~arr["csu"].reshape(len(cq), -1).any(axis=1)).mean()),
            "mean_best25_minus_top1": float((q_best - q_top).mean()),
            "random_pick_q": float(cq.mean()),
            "native_k5_minus_random": float((paths["ret_q"][:, :, 4].mean(axis=1) - cq.mean(axis=1)).mean()),
            "headroom_captured_k5": float((paths["ret_q"][:, :, 4].mean(axis=1) - cq.mean(axis=1)).mean()
                                          / (paths["orc_q"][:, :, 4].mean(axis=1) - cq.mean(axis=1)).mean()),
        })
        print(method, "done", flush=True)
    pd.DataFrame(gains).to_csv(args.output_dir / "budget_gains.csv", index=False)
    pd.DataFrame(canon).to_csv(args.output_dir / "order_sensitivity.csv", index=False)
    pd.DataFrame(ranking).to_csv(args.output_dir / "ranking_diagnostics.csv", index=False)
    (args.output_dir / "supplement_receipt.json").write_text(json.dumps(
        {"script": "scripts/paper_v1_supplement.py", "bootstrap_seed": SEED,
         "outputs": ["budget_gains.csv", "order_sensitivity.csv", "ranking_diagnostics.csv"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
