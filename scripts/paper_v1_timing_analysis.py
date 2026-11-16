#!/usr/bin/env python3
"""Paper v1 RQ4: decompose measured Chai-1 wall time and convert batch savings.

Reads profile.jsonl files written by scripts/profile_chai.py inside one GPU job,
decomposes each system's run into start-up S (process start to the first call),
first call B1 (includes one-time model/ESM loading) and steady per-batch time B
(mean of later calls). A run that stops after k batches takes
T(k) = S + B1 + (k - 1) * B. Batch savings from the RQ3 test split are then
converted to wall-clock savings per timed system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["af3", "af3_no_template", "boltz", "boltz1x", "boltz2", "chai", "protenix"]
CHAI_PAD_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]  # chai_lab.data.collate.utils.AVAILABLE_MODEL_SIZES (0.6.1)


def load_runs(job_dir: Path, systems_cfg: Path) -> pd.DataFrame:
    cfg = {s["system_id"]: s for s in json.loads(systems_cfg.read_text())["systems"]}
    rows = []
    for profile in sorted(job_dir.glob("*/profile.jsonl")):
        system = profile.parent.name
        events = [json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
        start = next(e for e in events if e["event"] == "start")
        batches = [e for e in events if e["event"] == "batch" and e["status"] == "completed"]
        if not batches:
            continue
        b1 = batches[0]
        startup = b1["cumulative_process_wall_seconds"] - b1["wall_seconds"]
        later = [b["wall_seconds"] for b in batches[1:]]
        smiles = cfg[system]["smiles"]
        tokens = cfg[system]["length"] + Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()  # one per residue / ligand heavy atom
        rows.append({
            "system_id": system, "residues": cfg[system]["length"], "tokens": tokens,
            "padded_tokens": min(n for n in CHAI_PAD_SIZES if n >= tokens), "gpu": start["gpu_name"],
            "n_batches": len(batches), "startup_s": startup, "first_call_s": b1["wall_seconds"],
            "steady_batch_s": float(np.mean(later)) if later else np.nan,
            "steady_batch_sd_s": float(np.std(later)) if len(later) > 1 else np.nan,
            "reserved_gb_max": max(b["pytorch_peak_reserved_bytes"] for b in batches) / 1e9,
            "measured_total_s": batches[-1]["cumulative_process_wall_seconds"],
            "smiles": smiles,
        })
    df = pd.DataFrame(rows).sort_values("residues")
    df["fixed_s"] = df["startup_s"] + df["first_call_s"] - df["steady_batch_s"]
    df["overhead_in_batches"] = df["fixed_s"] / df["steady_batch_s"]
    for k in range(1, 6):
        df[f"T{k}_s"] = df["startup_s"] + df["first_call_s"] + (k - 1) * df["steady_batch_s"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--systems", type=Path, default=ROOT / "configs/paper_v1_timing_systems.json")
    parser.add_argument("--tables", type=Path, default=ROOT / "results/paper_v1/tables")
    args = parser.parse_args()
    runs = load_runs(args.job_dir, args.systems)
    runs.drop(columns=["smiles"]).to_csv(args.tables / "timing_systems.csv", index=False)
    ops = pd.read_csv(args.tables / "stopping_operating_points.csv")
    conv = []
    for _, r in runs.iterrows():
        if not np.isfinite(r["steady_batch_s"]):
            continue
        t5 = r["T5_s"]
        for _, o in ops.iterrows():
            if not np.isfinite(o.get("test_cost", np.nan)):
                continue
            kbar = o["test_cost"]  # mean batches; T(k) is linear in k, so E[T(k)] = T(kbar)
            t = r["startup_s"] + r["first_call_s"] + (kbar - 1) * r["steady_batch_s"]
            conv.append({"system_id": r["system_id"], "residues": r["residues"], "method": o["method"],
                         "family": o["family"], "param_k": o.get("param_k"), "mean_batches": kbar,
                         "batch_saving": 1 - kbar / 5, "time_saving": 1 - t / t5})
    conv = pd.DataFrame(conv)
    conv.to_csv(args.tables / "timing_conversion.csv", index=False)
    chai = conv[(conv.method == "chai") & conv.family.isin(["conf", "fixed"])]
    summary = chai.groupby(["family", "param_k"], dropna=False)["time_saving"].agg(["min", "max", "mean"]).reset_index()
    print(runs.drop(columns=["smiles"]).round(2).to_string(index=False))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
