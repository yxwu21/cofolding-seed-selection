#!/usr/bin/env python3
"""E24 (seed extension v1): select 90 Chai-1 test systems stratified by native confidence.

Frozen selection rule:

* Pool: Chai-1 test-split systems with the complete 5 x 5 design whose public RNP
  input has exactly one protein sequence and one SMILES (the E22 rule), with at most
  768 Chai tokens after padding (residues plus ligand heavy atoms).
* Confidence: mean over the five archived seeds of the seed-level top ranking score.
* Tiers: tertiles of that confidence within the pool (low, mid, high).
* Allocation: 40 low, 30 mid and 20 high systems. Within a tier, systems are ordered
  by sha256("cofoldpilot-seedext-v1:" + system_id); the first system of each RNP
  cluster is kept and the first n systems are taken.
* Seeds: the five archived RNP seeds (ascending) followed by 15 new seeds,
  int(sha256(f"cofoldpilot-seedext-v1-seed:{system_id}:{j}").hexdigest()[:8], 16) % 2**31
  for j = 1, 2, ..., skipping values already used for the system.
* Array tasks: 12, assigned longest first by the E22 per-seed time of the padded size.

Ranking scores, clusters and splits are used; quality values and ground truth are not.
Run with the Chai environment, which provides RDKit:
  /usr/bin/python3.12 -S scripts/chai_site_bootstrap.py scripts/seedext_v1_select.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_v1_budget_eval as ev  # noqa: E402

PAD_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]  # chai_lab 0.6.1 model sizes
MAX_PAD = 768
SECONDS_PER_SEED = {256: 112.1, 384: 173.4, 512: 234.3, 768: 402.2}  # E22, RTX A6000
ALLOCATION = {"low": 40, "mid": 30, "high": 20}
N_NEW_SEEDS = 15
N_TASKS = 12
SALT = "cofoldpilot-seedext-v1"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_seeds(system_id: str, used: set[int]) -> list[int]:
    seeds, j = [], 1
    while len(seeds) < N_NEW_SEEDS:
        value = int(digest(f"{SALT}-seed:{system_id}:{j}")[:8], 16) % 2**31
        if value not in used:
            seeds.append(value)
            used.add(value)
        j += 1
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    meta, proper, preds = ev.load_inputs(args.predictions, args.annotations)
    cand = ev.candidate_table(preds["chai"], proper, meta)
    arr = ev.system_arrays(cand, ev.complete_systems(cand))
    confidence = dict(zip(arr["systems"], arr["S"].mean(axis=1)))
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    rows = []
    for system_id in arr["systems"]:
        record = inputs.get(system_id, {})
        sequences, smiles = record.get("sequences", {}), record.get("smiles", [])
        if meta.loc[system_id, "split"] != "test" or len(sequences) != 1 or len(smiles) != 1:
            continue
        chain, sequence = next(iter(sequences.items()))
        length = len(sequence.replace("-", ""))
        mol = Chem.MolFromSmiles(smiles[0])
        if mol is None:
            continue
        tokens = length + mol.GetNumHeavyAtoms()
        padded = min(p for p in PAD_SIZES if p >= tokens)
        if padded > MAX_PAD:
            continue
        rows.append({"system_id": system_id, "chain": chain, "length": length, "smiles": smiles[0],
                     "tokens": tokens, "padded_tokens": padded, "cluster": meta.loc[system_id, "cluster"],
                     "novelty": float(meta.loc[system_id, "novelty"]), "confidence": float(confidence[system_id]),
                     "hash": digest(f"{SALT}:{system_id}")})
    pool = pd.DataFrame(rows)
    cuts = np.quantile(pool["confidence"], [1 / 3, 2 / 3])
    pool["tier"] = np.array(["low", "mid", "high"])[np.digitize(pool["confidence"], cuts)]
    chosen = []
    for tier, n in ALLOCATION.items():
        sub = pool[pool["tier"] == tier].sort_values("hash")
        picks = sub.groupby("cluster", sort=False).head(1).sort_values("hash").head(n)
        if len(picks) != n:
            raise SystemExit(f"tier {tier}: only {len(picks)} clusters available for {n} systems")
        chosen.append(picks)
    selected = pd.concat(chosen)
    systems = []
    for _, row in selected.iterrows():
        archived = sorted(int(s) for s in cand.loc[cand["target"] == row["system_id"], "seed"].unique())
        if len(archived) != 5:
            raise SystemExit(f"{row['system_id']}: expected 5 archived seeds, found {len(archived)}")
        fresh = new_seeds(row["system_id"], set(archived))
        record = {k: row[k] for k in ("system_id", "chain", "length", "smiles", "tokens", "padded_tokens",
                                      "cluster", "novelty", "confidence", "tier")}
        record.update({k: int(record[k]) for k in ("length", "tokens", "padded_tokens")})
        record.update({"rnp_seeds": archived, "new_seeds": fresh, "seeds": archived + fresh,
                       "est_gpu_seconds": 20 * SECONDS_PER_SEED[record["padded_tokens"]]})
        systems.append(record)
    load = [0.0] * N_TASKS
    for record in sorted(systems, key=lambda r: (-r["est_gpu_seconds"], r["system_id"])):
        task = int(np.argmin(load))
        record["task"] = task
        load[task] += record["est_gpu_seconds"]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "experiment": "E24 seed extension v1", "selection_commit": commit,
        "rule": " ".join(line.strip() for line in __doc__.strip().splitlines()[2:17]),
        "pool_size": int(len(pool)), "pool_clusters": int(pool["cluster"].nunique()),
        "tier_cuts": [float(c) for c in cuts], "tier_pool_sizes": pool["tier"].value_counts().to_dict(),
        "allocation": ALLOCATION, "n_tasks": N_TASKS, "task_gpu_hours": [round(x / 3600, 2) for x in load],
        "systems": sorted(systems, key=lambda r: (r["task"], r["system_id"])),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    sel = pd.DataFrame(systems)
    print(f"pool {len(pool)} systems in {pool['cluster'].nunique()} clusters; cuts {np.round(cuts, 3)}")
    print(f"selected {len(sel)} systems in {sel['cluster'].nunique()} clusters; "
          f"GPU hours {sel['est_gpu_seconds'].sum() / 3600:.1f}; per task {out['task_gpu_hours']}")
    print(sel.groupby("tier")["padded_tokens"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
