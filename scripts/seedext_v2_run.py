#!/usr/bin/env python3
"""E25 (seed extension v2): run Boltz-2 or Protenix seed by seed and time every seed.

Each seed runs in its own process of the method's command-line tool, as a user would run
it, with the Runs N' Poses settings (10 recycles, 200 diffusion steps, 5 samples per seed):

  boltz2     boltz predict <input.yaml> --out_dir <batch> --cache <cache> --recycling_steps 10
             --diffusion_samples 5 --seed <seed> --no_trifast          (boltz 2.0.3)
             (--no_trifast: the trifast triangular-attention kernel needs a triton JIT build
             that requires Python development headers on the execution host)
  boltz      as boltz2 with boltz 0.4.1 (Boltz-1, the RNP version; no --no_trifast option, float32)
  protenix   protenix predict --input <input.json> --out_dir <batch> --seeds <seed>
             (protenix 0.3.4; its predict command sets 10 cycles, 5 samples, 200 steps)

Outputs go to <output-dir>/batch_<nn>_seed_<seed>/ with the tool's stdout and stderr, and
<output-dir>/profile.jsonl records one event per seed (wall time, exit code, number of model
files found, peak GPU memory used sampled with nvidia-smi). A failed seed is recorded and the
next seed starts; nothing is retried. Standard library only; run with a GPU allocation:
  /usr/bin/python3.12 -I scripts/seedext_v2_run.py --method boltz2 --input <yaml> \
      --output-dir <dir> --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

PROCESS_START = time.perf_counter()
STORAGE = Path(os.environ.get("COFOLD_STORAGE", str(Path(__file__).resolve().parents[1])))
ENVS = {name: Path(os.environ.get("COFOLD_" + name.upper() + "_ENV", str(STORAGE / "env" / folder)))
        for name, folder in {"boltz2": "boltz2", "protenix": "protenix", "boltz": "boltz1"}.items()}
PACKAGES = {"boltz2": "boltz", "protenix": "protenix", "boltz": "boltz"}
CACHES = {"boltz2": STORAGE / "model_cache/boltz2", "boltz": STORAGE / "model_cache/boltz1"}
EXPECTED_MODELS = 5


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


class MemoryPoller:
    """Peak GPU memory used (MiB) over the lifetime of the poller, sampled every few seconds."""

    def __init__(self, interval: float = 5.0):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = nvidia_smi("memory.used")
            if value:
                try:
                    self.peak = max(self.peak, max(int(v) for v in value.splitlines() if v.strip()))
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)


def package_version(method: str) -> str | None:
    python = ENVS[method] / "bin/python"
    out = subprocess.run([str(python), "-c", f"import importlib.metadata as m; print(m.version('{PACKAGES[method]}'))"],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def command(method: str, args, seed: int, batch: Path) -> list[str]:
    if method in ("boltz2", "boltz"):
        argv = [str(ENVS[method] / "bin/boltz"), "predict", str(args.input), "--out_dir", str(batch),
                "--cache", str(args.cache or CACHES[method]), "--recycling_steps", "10", "--diffusion_samples", "5",
                "--seed", str(seed), "--output_format", "mmcif"]
        return argv + ["--no_trifast"] if method == "boltz2" else argv
    return [str(ENVS[method] / "bin/protenix"), "predict", "--input", str(args.input), "--out_dir", str(batch),
            "--seeds", str(seed)]


def model_files(method: str, batch: Path) -> list[Path]:
    if method in ("boltz2", "boltz"):
        return sorted(batch.glob("boltz_results_*/predictions/*/*_model_*.cif"))
    return sorted(batch.glob("*/seed_*/predictions/*_sample_*.cif"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(ENVS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, default=None, help="Boltz cache (default per method)")
    parser.add_argument("--system-id", default=None)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = args.output_dir / "profile.jsonl"
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if args.method in ("boltz2", "boltz"):
        env["BOLTZ_CACHE"] = str(args.cache or CACHES[args.method])
    else:
        env.pop("USE_DEEPSPEED_EVO_ATTTENTION", None)  # Protenix default: no DeepSpeed kernels
    append_jsonl(profile, {
        "event": "start", "method": args.method, "package_version": package_version(args.method),
        "system_id": args.system_id or args.input.stem, "input": str(args.input), "input_bytes": args.input.stat().st_size,
        "seeds": args.seeds, "hostname": socket.gethostname(), "platform": platform.platform(),
        "gpu_name": nvidia_smi("name"), "gpu_memory_total_mib": nvidia_smi("memory.total"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "settings": {"recycles": 10, "diffusion_steps": 200, "samples_per_seed": 5}, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    for number, seed in enumerate(args.seeds, start=1):
        batch = args.output_dir / f"batch_{number:02d}_seed_{seed}"
        if batch.exists():
            shutil.rmtree(batch)
        batch.mkdir(parents=True)
        argv = command(args.method, args, seed, batch)
        started = time.perf_counter()
        with MemoryPoller() as poller, (batch / "stdout.txt").open("w") as out, (batch / "stderr.txt").open("w") as err:
            try:
                done = subprocess.run(argv, env=env, stdout=out, stderr=err, cwd=str(batch), check=False)
                code = done.returncode
            except OSError as exc:
                err.write(f"{type(exc).__name__}: {exc}\n")
                code = -1
        wall = time.perf_counter() - started
        models = model_files(args.method, batch)
        append_jsonl(profile, {
            "event": "batch", "batch_number": number, "seed": seed, "returncode": code,
            "status": "completed" if code == 0 and len(models) == EXPECTED_MODELS else "failed",
            "n_model_files": len(models), "wall_seconds": wall,
            "cumulative_process_wall_seconds": time.perf_counter() - PROCESS_START,
            "peak_gpu_memory_used_mib": poller.peak, "command": argv, "batch_dir": str(batch),
        })
        print(f"seed {seed}: exit {code}, {len(models)} models, {wall:.0f} s", flush=True)
    append_jsonl(profile, {"event": "end", "total_process_wall_seconds": time.perf_counter() - PROCESS_START})


if __name__ == "__main__":
    main()
