#!/usr/bin/env python3
"""E24 (seed extension v1), C2/C3: score Chai-1 candidates with OpenStructure as RNP did.

Every candidate is scored with the pinned OpenStructure 2.8.0 command that Runs N' Poses
used for Chai-1 (examples/utils/analyze_models_chai.sh at commit 197fafc):

  ost compare-ligand-structures -m <model.cif> -rl <ref>/ligand_files/*.sdf \
      -r <ref>/receptor.cif -o <json> --lddt-pli --rmsd --lddt-pli-amc

with the default 6 A LDDT-PLI radius. Calls run in parallel with a per-call timeout.
One row per candidate records the LDDT-PLI and RMSD of every assigned reference ligand
and the native ranking score; failures are kept as missing values with their reason.

Modes:
  archived  RNP Chai-1 structures of the five archived seeds (C2, scoring parity);
            models listed in <data-root>/refs_manifest.json
  new       E24 G1 outputs (C3) under --g1-run; ranking scores from scores.model_idx_*.npz

Run with the Chai environment (numpy):
  /usr/bin/python3.12 -S scripts/chai_site_bootstrap.py scripts/seedext_v1_score.py ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ost_cli_contract import verify_radius_default  # noqa: E402

STORAGE = Path(os.environ.get("COFOLD_STORAGE", str(Path(__file__).resolve().parents[1])))
OST_ENV = Path(os.environ.get("COFOLD_OST_ENV", str(STORAGE / "env/ost")))
OST = OST_ENV / "bin/ost"
CHEMLIB = OST_ENV / "share/openstructure/compounds.chemlib"
OST_SHA256 = "dd37387e3fb5682b5e3cf8ede50ea042e2d5ac3bf363ea011d4dc507021b3e5c"
CHEMLIB_SHA256 = "788dde3207cdfaa422e18672b97074069748eca0c8edc80d22e8ea43759f89bf"
FIELDS = ["mode", "system_id", "seed", "sample", "ranking_score", "status", "lddt_pli", "rmsd",
          "n_ref_ligands", "seconds", "error", "model_path", "json_path"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(mode: str, config: dict, data_root: Path, g1_run: Path | None) -> list[dict]:
    rows = []
    if mode == "archived":
        manifest = json.loads((data_root / "refs_manifest.json").read_text(encoding="utf-8"))
        for system_id, models in manifest["archived_models"].items():
            for m in models:
                path = data_root / m["path"]
                npz = path.with_name(f"scores.model_idx_{m['sample']}.npz")
                rows.append({"system_id": system_id, "seed": m["seed"], "sample": m["sample"],
                             "model": path, "npz": npz if npz.is_file() else None})
        return rows
    for row in config["systems"]:
        for batch in sorted(g1_run.glob(f"task_*/{row['system_id']}/batch_*_seed_*")):
            seed = int(re.search(r"seed_(\d+)$", batch.name).group(1))
            for sample in range(5):
                model = batch / f"pred.model_idx_{sample}.cif"
                if model.is_file():
                    rows.append({"system_id": row["system_id"], "seed": seed, "sample": sample, "model": model,
                                 "npz": batch / f"scores.model_idx_{sample}.npz"})
    return rows


def score(job: dict, mode: str, data_root: Path, out: Path, env: dict, timeout: float) -> dict:
    ref = data_root / "refs" / job["system_id"]
    ligands = sorted((ref / "ligand_files").glob("*.sdf"))
    json_path = out / "json" / job["system_id"] / f"{job['seed']}_{job['sample']}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"mode": mode, "system_id": job["system_id"], "seed": job["seed"], "sample": job["sample"],
           "ranking_score": np.nan, "status": "", "lddt_pli": "", "rmsd": "", "n_ref_ligands": len(ligands),
           "seconds": np.nan, "error": "", "model_path": str(job["model"]), "json_path": str(json_path)}
    if job["npz"] is not None and Path(job["npz"]).is_file():
        row["ranking_score"] = float(np.load(job["npz"])["aggregate_score"].ravel()[0])
    argv = [str(OST), "compare-ligand-structures", "-m", str(job["model"]), "-rl", *map(str, ligands),
            "-r", str(ref / "receptor.cif"), "-o", str(json_path), "--lddt-pli", "--rmsd", "--lddt-pli-amc"]
    log = json_path.with_suffix(".log")
    started = time.perf_counter()
    try:
        with log.open("w") as handle:
            done = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        row["seconds"] = time.perf_counter() - started
        if done.returncode != 0:
            row.update(status="ost_failed", error=f"exit {done.returncode}")
            return row
        result = json.loads(json_path.read_text())
        row["status"] = result.get("status", "missing_status")
        for metric in ("lddt_pli", "rmsd"):
            block = result.get(metric, {})
            assigned = {Path(a["reference_ligand"]).stem: a["score"] for a in block.get("assigned_scores", [])}
            row[metric] = json.dumps(assigned, sort_keys=True)
        unassigned = result.get("lddt_pli", {}).get("reference_ligand_unassigned_reason", {})
        if unassigned:
            row["error"] = json.dumps({Path(k).stem: v for k, v in unassigned.items()}, sort_keys=True)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", seconds=time.perf_counter() - started, error=f"over {timeout} s")
    except Exception as exc:  # recorded as missing, never as zero
        row.update(status="error", seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["archived", "new"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    parser.add_argument("--data-root", type=Path, default=STORAGE / "data/seedext_v1")
    parser.add_argument("--g1-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.mode == "new" and args.g1_run is None:
        parser.error("--g1-run is required in mode new")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "tmp").mkdir()
    identity = {"ost_sha256": sha256(OST), "chemlib_sha256": sha256(CHEMLIB)}
    if identity != {"ost_sha256": OST_SHA256, "chemlib_sha256": CHEMLIB_SHA256}:
        raise SystemExit(f"OST identity changed: {identity}")
    cli = verify_radius_default(OST)
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "OST_ROOT", "OST_COMPOUNDS_CHEMLIB", "LD_LIBRARY_PATH"):
        env.pop(name, None)
    tmp = str(args.output_dir / "tmp")
    env.update(TMPDIR=tmp, TMP=tmp, TEMP=tmp, OST_COMPOUNDS_CHEMLIB=str(CHEMLIB), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = candidates(args.mode, config, args.data_root, args.g1_run)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: score(j, args.mode, args.data_root, args.output_dir, env, args.timeout), jobs))
    with (args.output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    commit_file = ROOT / "COMMIT"
    commit = commit_file.read_text().strip() if commit_file.is_file() else subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    receipt = {"script": "scripts/seedext_v1_score.py", "commit": commit, "mode": args.mode,
               "g1_run": str(args.g1_run) if args.g1_run else None, "candidates": len(jobs),
               "statuses": statuses, "workers": args.workers, "timeout_s": args.timeout,
               "ost": {**identity, **cli}, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "wall_seconds": time.perf_counter() - started}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
