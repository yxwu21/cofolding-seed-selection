#!/usr/bin/env python3
"""Stream the public RNP MSA archive once and build Chai inputs for the timing systems.

For each system in configs/paper_v1_timing_systems.json, writes
<output-root>/<system>/input.fasta (one protein, one ligand, from the public RNP
inputs.json) and <output-root>/<system>/msa/ (the system's archived MSA files),
plus a manifest with byte counts and SHA-256 of every file. Ground truth is
never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=ROOT / "configs/paper_v1_timing_systems.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/runs_n_poses_v6/msa_files.tar.gz")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    selected = {row["system_id"]: row for row in json.loads(args.systems.read_text())["systems"]}
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    manifest = {"archive": str(args.archive), "systems": {}}
    for system_id, row in selected.items():
        record = inputs[system_id]
        (chain, seq), = record["sequences"].items()
        smiles, = record["smiles"]
        out = args.output_root / system_id
        (out / "msa").mkdir(parents=True, exist_ok=True)
        fasta = out / "input.fasta"
        fasta.write_text(f">protein|name=A\n{seq.replace('-', '').upper()}\n>ligand|name=B\n{smiles}\n", encoding="utf-8")
        manifest["systems"][system_id] = {"length": row["length"], "chain": chain, "fasta_sha256": sha256(fasta), "msa_files": []}
    lowered = {s.lower(): s for s in selected}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if "msa_files" not in parts or not member.isfile():
                continue
            tail = parts[parts.index("msa_files") + 1:]
            if len(tail) < 2 or tail[0].lower() not in lowered:
                continue
            if any(p in {"", ".", ".."} for p in tail):
                raise ValueError(f"unsafe member {member.name}")
            system_id = lowered[tail[0].lower()]
            dest = args.output_root / system_id / "msa" / Path(*tail[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            tmp = dest.with_name(dest.name + ".part")
            with source, tmp.open("wb") as handle:
                for block in iter(lambda: source.read(1 << 20), b""):
                    handle.write(block)
            os.replace(tmp, dest)
            manifest["systems"][system_id]["msa_files"].append(
                {"member": member.name, "path": str(dest.relative_to(args.output_root)),
                 "bytes": dest.stat().st_size, "sha256": sha256(dest)})
    missing = [s for s, m in manifest["systems"].items() if not any(f["path"].endswith(".aligned.pqt") for f in m["msa_files"])]
    manifest["missing_aligned_pqt"] = missing
    manifest["wall_seconds"] = time.perf_counter() - started
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"systems": len(selected), "missing_aligned_pqt": missing, "wall_seconds": manifest["wall_seconds"]}))
    if missing:
        raise SystemExit(f"missing Chai MSA for {missing}")


if __name__ == "__main__":
    main()
