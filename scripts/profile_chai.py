#!/usr/bin/env python3
"""Profile complete Chai candidate batches using the Runs N' Poses settings.

Run with a GPU allocation, a writable output directory and an
already-installed Chai environment. This script intentionally does not install
packages or download inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

PROCESS_START = time.perf_counter()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-file", type=Path, required=True)
    parser.add_argument("--msa-directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--num-trunk-recycles", type=int, default=10)
    parser.add_argument("--num-diffn-timesteps", type=int, default=200)
    args = parser.parse_args()

    if not args.fasta_file.is_file() or not args.msa_directory.is_dir():
        parser.error("FASTA file and MSA directory must already exist")
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.num_trunk_recycles <= 0 or args.num_diffn_timesteps <= 0:
        parser.error("recycles and timesteps must be positive")
    cache = os.environ.get("CHAI_DOWNLOADS_DIR")
    scratch = os.environ.get("TMPDIR")
    if not cache or not Path(cache).is_dir():
        parser.error("CHAI_DOWNLOADS_DIR must name an existing approved cache")
    if not scratch or not Path(scratch).is_dir():
        parser.error("TMPDIR must name an existing approved project scratch directory")
    for label, path in (
        ("TMPDIR", Path(scratch)),
        ("CHAI_DOWNLOADS_DIR", Path(cache)),
        ("output directory", args.output_dir),
    ):
        if path.resolve().is_relative_to(Path("/tmp")):
            parser.error(f"server /tmp is prohibited for {label}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = args.output_dir / "profile.jsonl"

    import torch
    from chai_lab.chai1 import run_inference

    if not torch.cuda.is_available():
        parser.error("CUDA GPU is unavailable in this Slurm job")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    append_jsonl(log, {
        "event": "start", "schema_version": 1, "git_commit": git_commit(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "chai_lab_version": importlib.metadata.version("chai-lab"),
        "torch_version": torch.__version__, "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "fasta_file": str(args.fasta_file.resolve()),
        "fasta_sha256": sha256(args.fasta_file),
        "msa_directory": str(args.msa_directory.resolve()),
        "chai_downloads_dir": str(Path(cache).resolve()),
        "tmpdir": str(Path(scratch).resolve()), "seeds": args.seeds,
        "num_trunk_recycles": args.num_trunk_recycles,
        "num_diffn_timesteps": args.num_diffn_timesteps,
        "num_diffn_samples_per_seed": 5,
        "use_esm_embeddings": True, "use_msa_server": False,
        "measurement_scope": "process startup, imports, sequential run_inference calls, output writes; excludes Slurm queue, input/MSA preparation, and package installation",
    })
    for batch_number, seed in enumerate(args.seeds, start=1):
        batch_dir = args.output_dir / f"batch_{batch_number:02d}_seed_{seed}"
        batch_dir.mkdir()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            run_inference(
                fasta_file=args.fasta_file,
                output_dir=batch_dir,
                num_trunk_recycles=args.num_trunk_recycles,
                num_diffn_timesteps=args.num_diffn_timesteps,
                num_diffn_samples=5,
                seed=seed,
                device=device,
                use_esm_embeddings=True,
                use_msa_server=False,
                msa_directory=args.msa_directory,
            )
            torch.cuda.synchronize(device)
        except Exception as exc:
            append_jsonl(log, {
                "event": "batch", "batch_number": batch_number, "seed": seed,
                "status": "failed", "wall_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__, "error": str(exc),
            })
            raise
        append_jsonl(log, {
            "event": "batch", "batch_number": batch_number, "seed": seed,
            "status": "completed", "wall_seconds": time.perf_counter() - started,
            "cumulative_process_wall_seconds": time.perf_counter() - PROCESS_START,
            "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "output_dir": str(batch_dir.resolve()),
        })
    append_jsonl(log, {
        "event": "finish", "status": "completed",
        "total_process_wall_seconds": time.perf_counter() - PROCESS_START,
        "note": "PyTorch allocator peaks are not total device VRAM; report Slurm job accounting separately.",
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"profile_chai failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
