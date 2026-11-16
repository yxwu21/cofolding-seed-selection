#!/usr/bin/env python3
"""E24 (seed extension v1), C1b: extract references and archived Chai-1 structures.

Streams the public RNP ground-truth archive (receptor.cif, ligand_files/*.sdf,
system.cif and sequences.fasta of each selected system) into <output-root>/refs/<system>
and the public RNP prediction archive (the archived Chai-1 files of each selected
system) into <output-root>/archived/<system>/. Writes a manifest with sizes and
SHA-256 and exits non-zero if a system lacks its receptor, a ligand file or the 25
archived models of its five RNP seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("COFOLD_STORAGE", str(Path(__file__).resolve().parents[1])))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_member(archive, member, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    tmp = dest.with_name(dest.name + ".part")
    with source, tmp.open("wb") as handle:
        for block in iter(lambda: source.read(1 << 20), b""):
            handle.write(block)
    os.replace(tmp, dest)
    return {"member": member.name, "bytes": dest.stat().st_size, "sha256": sha256(dest)}


def safe(parts) -> None:
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"unsafe archive member {'/'.join(parts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    parser.add_argument("--ground-truth", type=Path, default=STORAGE / "data/runs_n_poses_v6/ground_truth.tar.gz")
    parser.add_argument("--predictions", type=Path, default=STORAGE / "data/runs_n_poses_v6/prediction_files.tar.gz")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.systems.read_text(encoding="utf-8"))
    systems = {row["system_id"]: row for row in config["systems"]}
    lowered = {s.lower(): s for s in systems}
    manifest = {"ground_truth": str(args.ground_truth), "predictions": str(args.predictions),
                "refs": {s: [] for s in systems}, "archived": {s: [] for s in systems}}
    with tarfile.open(args.ground_truth, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 3 or parts[0] != "ground_truth" or parts[1] not in systems:
                continue
            safe(parts)
            dest = args.output_root / "refs" / parts[1] / Path(*parts[2:])
            manifest["refs"][parts[1]].append({"path": str(dest.relative_to(args.output_root)),
                                               **copy_member(archive, member, dest)})
    print(f"references done after {time.perf_counter() - started:.0f} s", flush=True)
    with tarfile.open(args.predictions, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or "prediction_files" not in parts:
                continue
            tail = parts[parts.index("prediction_files") + 1:]
            if len(tail) < 3 or tail[0] != "chai" or tail[1].lower() not in lowered:
                continue
            safe(tail)
            system_id = lowered[tail[1].lower()]
            dest = args.output_root / "archived" / system_id / Path(*tail[2:])
            manifest["archived"][system_id].append({"path": str(dest.relative_to(args.output_root)),
                                                    **copy_member(archive, member, dest)})
    problems, models = {}, {}
    for system_id, row in systems.items():
        ref = args.output_root / "refs" / system_id
        issues = []
        if not (ref / "receptor.cif").is_file():
            issues.append("no receptor.cif")
        if not list((ref / "ligand_files").glob("*.sdf")):
            issues.append("no ligand sdf")
        found = []
        for path in sorted((args.output_root / "archived" / system_id).rglob("*.cif")):
            flat = re.fullmatch(r"seed-(\d+)_sample-(\d+)\.cif", path.name)  # RNP archive layout
            nested = re.fullmatch(r"pred\.model_idx_(\d+)\.cif", path.name)  # Chai output layout
            if flat:
                seed, sample = int(flat.group(1)), int(flat.group(2))
            elif nested:
                digits = "".join(ch for ch in path.parent.name if ch.isdigit())
                seed, sample = (int(digits) if digits else None), int(nested.group(1))
            else:
                continue
            found.append({"seed": seed, "sample": sample, "path": str(path.relative_to(args.output_root))})
        models[system_id] = found
        seeds = {m["seed"] for m in found}
        if len(found) != 25 or seeds != set(row["rnp_seeds"]):
            issues.append(f"archived models: {len(found)} files, seeds {sorted(s for s in seeds if s is not None)}")
        if issues:
            problems[system_id] = issues
    manifest["archived_models"] = models
    manifest["problems"] = problems
    manifest["wall_seconds"] = time.perf_counter() - started
    (args.output_root / "refs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"systems": len(systems), "problems": problems, "wall_seconds": manifest["wall_seconds"]}))
    if problems:
        raise SystemExit(f"{len(problems)} systems incomplete")


if __name__ == "__main__":
    main()
