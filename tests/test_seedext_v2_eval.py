"""E25 analysis script: timing profile parsing and a synthetic end-to-end run per method."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import seedext_v2_eval as v2  # noqa: E402

CONFIG = ROOT / "configs/seedext_v2_systems.json"
DATA = ROOT / "data/raw"


def write_profile(path: Path, seeds: list[int], wall: float = 100.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps({"event": "start", "gpu_name": "NVIDIA RTX A6000", "package_version": "x"}) + "\n")
        cum = 30.0
        for n, seed in enumerate(seeds, 1):
            cum += wall
            fh.write(json.dumps({"event": "batch", "batch_number": n, "seed": seed, "status": "completed",
                                 "wall_seconds": wall, "cumulative_process_wall_seconds": cum,
                                 "peak_gpu_memory_used_mib": 10240}) + "\n")


def test_timing_rows_keeps_only_the_requested_method(tmp_path):
    """The G run holds Protenix and Boltz-2 tasks; a shared system must not mix the two."""
    for task, method, wall in (("task_00", "protenix", 100.0), ("task_20", "boltz2", 150.0)):
        path = tmp_path / task / "sysA" / "profile.jsonl"
        write_profile(path, list(range(20)), wall=wall)
        lines = path.read_text().splitlines()
        start = json.loads(lines[0])
        start["method"] = method
        path.write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")
    systems = {"sysA": {"padded_tokens": 384}}
    assert len(v2.timing_rows(tmp_path, systems)) == 2
    for method, wall in (("protenix", 100.0), ("boltz2", 150.0)):
        rows = v2.timing_rows(tmp_path, systems, method)
        assert len(rows) == 1 and abs(rows.iloc[0]["steady_s"] - wall) < 1e-9


def test_timing_rows_reads_runner_profiles(tmp_path):
    write_profile(tmp_path / "task_00" / "sysA" / "profile.jsonl", list(range(20)))
    rows = v2.timing_rows(tmp_path, {"sysA": {"padded_tokens": 384}})
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["n_seeds"] == 20 and row["padded_tokens"] == 384
    assert abs(row["startup_s"] - 30.0) < 1e-9 and abs(row["steady_s"] - 100.0) < 1e-9
    assert abs(row["peak_gpu_memory_gb"] - 10240 * 2**20 / 1e9) < 1e-9


def test_uses_frozen_e24_functions():
    import seedext_v1_eval as v1
    assert v2.stratified_bootstrap is v1.stratified_bootstrap
    assert v2.curves is v1.curves and v2.k_star is v1.k_star and v2.reading is v1.reading


@pytest.mark.skipif(not (CONFIG.is_file() and (DATA / "runs_n_poses_predictions.tar.gz").is_file()),
                    reason="needs the selection config and the public RNP tables")
@pytest.mark.parametrize("method", ["protenix", "boltz2"])
def test_synthetic_end_to_end(tmp_path, method):
    import pandas as pd
    cfg = json.loads(CONFIG.read_text())["methods"][method]
    ann = pd.read_csv(DATA / "runs_n_poses_annotations.csv")
    proper = ann[ann["ligand_is_proper"] == True].groupby("system_id")["ligand_instance_chain"] \
        .apply(lambda s: sorted(set(s))).to_dict()  # noqa: E712
    rng = np.random.default_rng(0)
    fields = ["mode", "system_id", "seed", "sample", "ranking_score", "status", "lddt_pli", "rmsd", "n_ref_ligands",
              "seconds", "error", "model_path", "json_path"]
    scores = tmp_path / "scores.csv"
    with scores.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in cfg["systems"]:
            base = rng.uniform(0.3, 0.9)
            for seed in s["seeds"]:
                for sample in range(5):
                    q = float(np.clip(base + rng.normal(0, 0.1), 0, 1))
                    w.writerow({"mode": "new", "system_id": s["system_id"], "seed": seed, "sample": sample,
                                "ranking_score": float(np.clip(q + rng.normal(0, 0.15), 0, 1)), "status": "SUCCESS",
                                "lddt_pli": json.dumps({c: q for c in proper[s["system_id"]]}),
                                "rmsd": json.dumps({c: 1.0 for c in proper[s["system_id"]]}),
                                "n_ref_ligands": 1, "seconds": 1.0, "error": "", "model_path": "x", "json_path": "y"})
    write_profile(tmp_path / "g" / "task_00" / cfg["systems"][0]["system_id"] / "profile.jsonl", cfg["systems"][0]["seeds"])
    out = tmp_path / "results"
    # Same interpreter and sys.path as the test (numpy may come from PYTHONPATH), so no -I here.
    subprocess.run([sys.executable, str(ROOT / "scripts/seedext_v2_eval.py"), "--method", method,
                    "--new-scores", str(scores), "--g-run", str(tmp_path / "g"), "--output-dir", str(out)], check=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_complete"] == len(cfg["systems"]) and summary["primary_analysis"]
    assert summary["primary"]["reading"] in ("A", "B", "C")
    for name in ("curves.csv", "curves_own_tiers.csv", "kstar.csv", "per_system.csv", "ranking.csv", "rule.csv",
                 "timing.csv", "replication.csv"):
        assert (out / name).is_file()
