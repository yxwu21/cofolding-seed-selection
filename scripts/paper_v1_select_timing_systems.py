#!/usr/bin/env python3
"""Select RNP systems for the paper v1 Chai timing study (RQ4).

Deterministic rule: test-split Chai systems with the complete 5x5 design whose
public RNP input has exactly one protein sequence and one SMILES. For each
target protein length L in TARGET_LENGTHS (kept <= 600 residues so the padded
token count stays within 768 on a 48 GB GPU), take the eligible system whose
length is closest to L (ties: lexicographically smallest system id), without
reuse. Timing uses the system's first N_BATCHES archived RNP seeds (ascending).
Ground-truth structures are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from paper_v1_budget_eval import (  # noqa: E402
    candidate_table, complete_systems, load_inputs,
)

TARGET_LENGTHS = [100, 175, 250, 325, 400, 500, 600]
N_BATCHES = 3  # per system; first batch includes start-up, later batches give steady per-batch cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/paper_v1_timing_systems.json")
    args = parser.parse_args()
    meta, proper, preds = load_inputs(ROOT / "data/raw/runs_n_poses_predictions.tar.gz",
                                      ROOT / "data/raw/runs_n_poses_annotations.csv")
    cand = candidate_table(preds["chai"], proper, meta)
    complete = set(complete_systems(cand))
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    rows = []
    for system_id, record in inputs.items():
        seqs, smiles = record.get("sequences", {}), record.get("smiles", [])
        if len(seqs) != 1 or len(smiles) != 1 or system_id not in complete:
            continue
        if meta.loc[system_id, "split"] != "test":
            continue
        chain, seq = next(iter(seqs.items()))
        rows.append({"system_id": system_id, "chain": chain, "length": len(seq.replace("-", "")),
                     "smiles": smiles[0]})
    pool = pd.DataFrame(rows).sort_values("system_id")
    chosen = []
    for target in TARGET_LENGTHS:
        rest = pool[~pool["system_id"].isin([c["system_id"] for c in chosen])]
        dist = (rest["length"] - target).abs()
        best = rest.loc[dist == dist.min()].sort_values("system_id").iloc[0]
        record = {"target_length": target, **best.to_dict()}
        seeds = sorted(int(x) for x in cand.loc[cand["target"] == best["system_id"], "seed"].unique())
        record["rnp_seeds"] = seeds
        record["timing_seeds"] = seeds[:N_BATCHES]
        chosen.append(record)
    out = {
        "rule": " ".join(line.strip() for line in __doc__.strip().splitlines()[2:]),
        "eligible_pool": int(len(pool)),
        "systems": chosen,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(chosen)[["target_length", "system_id", "length", "timing_seeds"]].to_string(index=False))
    print("eligible pool:", len(pool), "length range:", int(pool["length"].min()), "-", int(pool["length"].max()))


if __name__ == "__main__":
    main()
