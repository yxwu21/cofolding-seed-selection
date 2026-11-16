#!/usr/bin/env python3
"""E25 (seed extension v2): systems and seeds for Protenix and Boltz-2 at 20 seeds per system.

Frozen selection rule:

* Pool of a method: its test-split systems with the complete 5 x 5 design whose public RNP
  input has exactly one protein sequence and one SMILES, with at most 768 Chai tokens after
  padding (residues plus ligand heavy atoms); the E24 rule applied to the method's table.
* Confidence of a system: mean over the five archived seeds of the seed-level top ranking
  score of the method; own tiers are the tertiles of that confidence within the pool.
* Protenix: the 90 systems of E24 (configs/seedext_v1_systems.json), all of which are in
  the Protenix pool; the design tier is the E24 (Chai-1) tier so that the primary analysis
  has the same weighting as E24. The own tier is kept for the secondary analysis.
* Boltz-2: its whole pool (the systems released after its training cutoff leave 93 systems
  in 49 clusters); the design tier is the own tier. Only 31 of the 90 E24 systems are in
  the Boltz-2 table, so a one-per-cluster 40/30/20 draw is not possible.
* Seeds: the five archived RNP seeds of the method (ascending) followed by 15 new seeds,
  int(sha256(f"cofoldpilot-seedext-v2-{method}-seed:{system_id}:{j}").hexdigest()[:8], 16)
  % 2**31 for j = 1, 2, ..., skipping values already used for the system.
* Boltz-1 (method key "boltz" as in the RNP
  tables): the E24 systems that are complete in the Boltz-1 table (88 of 90), design tier =
  E24 tier; run with --methods boltz --output configs/seedext_v2_boltz1_systems.json.
* Array tasks are planned separately (scripts/seedext_v2_plan_tasks.py) from measured
  per-seed times, so that this file does not change after the pilot.

Ranking scores, clusters, splits and inputs are used; quality values and ground truth are
not. Run with the Chai environment, which provides pandas and RDKit:
  python -I scripts/seedext_v2_select.py
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

PAD_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]
MAX_PAD = 768
N_NEW_SEEDS = 15
SALT = "cofoldpilot-seedext-v2"
METHODS = ("protenix", "boltz2")  # default output configs/seedext_v2_systems.json
E24_METHODS = ("protenix", "boltz")  # methods drawn from the E24 systems
TIERS = ("low", "mid", "high")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_seeds(method: str, system_id: str, used: set[int]) -> list[int]:
    seeds, j = [], 1
    while len(seeds) < N_NEW_SEEDS:
        value = int(digest(f"{SALT}-{method}-seed:{system_id}:{j}")[:8], 16) % 2**31
        if value not in used:
            seeds.append(value)
            used.add(value)
        j += 1
    return seeds


def method_pool(method: str, meta, proper, preds, inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cand = ev.candidate_table(preds[method], proper, meta)
    arr = ev.system_arrays(cand, ev.complete_systems(cand))
    confidence = dict(zip(arr["systems"], arr["S"].mean(axis=1)))
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
                     "novelty": float(meta.loc[system_id, "novelty"]), "confidence": float(confidence[system_id])})
    pool = pd.DataFrame(rows)
    cuts = np.quantile(pool["confidence"], [1 / 3, 2 / 3])
    pool["own_tier"] = np.array(TIERS)[np.digitize(pool["confidence"], cuts)]
    return pool, cand, cuts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/raw/runs_n_poses_predictions.tar.gz")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/raw/runs_n_poses_annotations.csv")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--e24", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/seedext_v2_systems.json")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=["protenix", "boltz2", "boltz"])
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    meta, proper, preds = ev.load_inputs(args.predictions, args.annotations)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    e24 = {s["system_id"]: s for s in json.loads(args.e24.read_text(encoding="utf-8"))["systems"]}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"experiment": "E25 seed extension v2", "selection_commit": commit,
           "rule": " ".join(line.strip() for line in __doc__.strip().splitlines()[2:20]),
           "e24_config": str(args.e24.relative_to(ROOT)), "methods": {}}
    for method in args.methods:
        pool, cand, cuts = method_pool(method, meta, proper, preds, inputs)
        if method in E24_METHODS:
            missing = [s for s in e24 if s not in set(pool["system_id"])]
            if method == "protenix" and missing:
                raise SystemExit(f"protenix: {len(missing)} E24 systems not in the pool: {missing[:5]}")
            selected = pool[pool["system_id"].isin(e24)].copy()
            selected["tier"] = [e24[s]["tier"] for s in selected["system_id"]]
            gap = f" complete for the method (missing: {', '.join(sorted(missing))})" if missing else ""
            design = f"the {len(selected)} E24 systems{gap}; design tier = E24 (Chai-1) tier"
        else:
            selected = pool.copy()
            selected["tier"] = selected["own_tier"]
            design = "the whole pool; design tier = own tier"
        systems = []
        for _, row in selected.sort_values("system_id").iterrows():
            archived = sorted(int(s) for s in cand.loc[cand["target"] == row["system_id"], "seed"].unique())
            if len(archived) != 5:
                raise SystemExit(f"{method} {row['system_id']}: expected 5 archived seeds, found {len(archived)}")
            fresh = new_seeds(method, row["system_id"], set(archived))
            record = {k: row[k] for k in ("system_id", "chain", "length", "smiles", "tokens", "padded_tokens",
                                          "cluster", "novelty", "confidence", "own_tier", "tier")}
            record.update({k: int(record[k]) for k in ("length", "tokens", "padded_tokens")})
            record["e24_tier"] = e24[row["system_id"]]["tier"] if row["system_id"] in e24 else None
            record.update({"rnp_seeds": archived, "new_seeds": fresh, "seeds": archived + fresh})
            systems.append(record)
        sel = pd.DataFrame(systems)
        out["methods"][method] = {
            "design": design, "pool_size": int(len(pool)), "pool_clusters": int(pool["cluster"].nunique()),
            "tier_cuts": [float(c) for c in cuts], "tier_pool_sizes": pool["own_tier"].value_counts().to_dict(),
            "n_systems": int(len(sel)), "n_clusters": int(sel["cluster"].nunique()),
            "design_tier_sizes": sel["tier"].value_counts().to_dict(),
            "own_tier_sizes": sel["own_tier"].value_counts().to_dict(),
            "n_in_e24": int(sel["e24_tier"].notna().sum()), "systems": systems}
        print(f"{method}: pool {len(pool)} systems in {pool['cluster'].nunique()} clusters, cuts {np.round(cuts, 4)}; "
              f"selected {len(sel)} in {sel['cluster'].nunique()} clusters; design tiers "
              f"{sel['tier'].value_counts().to_dict()}; own tiers {sel['own_tier'].value_counts().to_dict()}; "
              f"in E24 {int(sel['e24_tier'].notna().sum())}")
        print(sel.groupby("tier")["padded_tokens"].value_counts().unstack(fill_value=0).to_string())
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
