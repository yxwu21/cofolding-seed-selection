#!/usr/bin/env python3
"""E25 (seed extension v2), C1: references, MSAs, model inputs and archived RNP structures.

For every system of configs/seedext_v2_systems.json (both methods), streams the public RNP
archives once each and writes under <output-root>:

  refs/<system>/                 receptor.cif, ligand_files/*.sdf, ... (ground_truth.tar.gz)
  msa/<system>/                  the system's archived MSA directory (msa_files.tar.gz)
  archived/<method>/<system>/    the archived RNP predictions of the method (prediction_files.tar.gz)
  inputs/boltz2/<system>/input.yaml     Boltz input (sequence, SMILES, the archived Boltz MSA csv)
  inputs/protenix/<system>.json         Protenix input (sequence, SMILES, the archived Protenix MSA dir)

and a manifest (sizes, SHA-256, problems). Exits non-zero if a system lacks its receptor, a
ligand file, its MSA for the method, or has no archived files. Ground-truth files are copied,
never read. Standard library only:
  /usr/bin/python3.12 -I scripts/seedext_v2_extract.py --output-root <storage>/data/seedext_v2
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
STORAGE = Path(os.environ.get("COFOLD_STORAGE", str(Path(__file__).resolve().parents[1])))
ARCHIVE_DIRS = {"boltz2": "boltz2", "protenix": "protenix", "boltz": "boltz"}  # method -> directory in prediction_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(parts) -> None:
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"unsafe archive member {'/'.join(parts)}")


def copy_member(archive, member, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    tmp = dest.with_name(dest.name + ".part")
    with source, tmp.open("wb") as handle:
        for block in iter(lambda: source.read(1 << 20), b""):
            handle.write(block)
    os.replace(tmp, dest)
    return {"member": member.name, "bytes": dest.stat().st_size, "sha256": sha256(dest)}


class _Empty:
    """Stand-in for a skipped archive: an empty iterable context manager."""

    def __enter__(self):
        return iter(())

    def __exit__(self, *exc):
        return False


def yaml_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", type=Path, default=ROOT / "configs/seedext_v2_systems.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/raw/runs_n_poses_inputs.json")
    parser.add_argument("--ground-truth", type=Path, default=STORAGE / "data/runs_n_poses_v6/ground_truth.tar.gz")
    parser.add_argument("--msa-archive", type=Path, default=STORAGE / "data/runs_n_poses_v6/msa_files.tar.gz")
    parser.add_argument("--predictions", type=Path, default=STORAGE / "data/runs_n_poses_v6/prediction_files.tar.gz")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-predictions", action="store_true", help="do not stream the prediction archive")
    parser.add_argument("--skip-ground-truth", action="store_true", help="reuse refs/ already under --output-root")
    parser.add_argument("--skip-msa", action="store_true", help="reuse msa/ already under --output-root")
    parser.add_argument("--manifest-name", default="extract_manifest.json")
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.systems.read_text(encoding="utf-8"))
    by_method = {m: {s["system_id"]: s for s in block["systems"]} for m, block in config["methods"].items()}
    systems = {s: row for rows in by_method.values() for s, row in rows.items()}
    lowered = {s.lower(): s for s in systems}
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    out = args.output_root
    manifest = {"config": str(args.systems), "ground_truth": str(args.ground_truth), "msa_archive": str(args.msa_archive),
                "predictions": str(args.predictions), "refs": {s: [] for s in systems}, "msa": {s: [] for s in systems},
                "archived": {m: {s: [] for s in rows} for m, rows in by_method.items()}, "inputs": {}}

    with (tarfile.open(args.ground_truth, "r|gz") if not args.skip_ground_truth else _Empty()) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 3 or parts[0] != "ground_truth" or parts[1] not in systems:
                continue
            safe(parts)
            dest = out / "refs" / parts[1] / Path(*parts[2:])
            manifest["refs"][parts[1]].append({"path": str(dest.relative_to(out)), **copy_member(archive, member, dest)})
    print(f"references done after {time.perf_counter() - started:.0f} s", flush=True)

    with (tarfile.open(args.msa_archive, "r|gz") if not args.skip_msa else _Empty()) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or "msa_files" not in parts:
                continue
            tail = parts[parts.index("msa_files") + 1:]
            if len(tail) < 2 or tail[0].lower() not in lowered:
                continue
            safe(tail)
            system_id = lowered[tail[0].lower()]
            dest = out / "msa" / system_id / Path(*tail[1:])
            manifest["msa"][system_id].append({"path": str(dest.relative_to(out)), **copy_member(archive, member, dest)})
    print(f"msa done after {time.perf_counter() - started:.0f} s", flush=True)

    if not args.skip_predictions:
        # Only the methods of this configuration; other methods' members are skipped (the Boltz-1
        # configuration has no Protenix entry, which raised a KeyError in job 336621).
        dir_to_method = {d: m for m, d in ARCHIVE_DIRS.items() if m in by_method}
        with tarfile.open(args.predictions, "r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or "prediction_files" not in parts:
                    continue
                tail = parts[parts.index("prediction_files") + 1:]
                if len(tail) < 3 or tail[0] not in dir_to_method:
                    continue
                method = dir_to_method[tail[0]]
                if tail[1].lower() not in lowered or lowered[tail[1].lower()] not in by_method[method]:
                    continue
                safe(tail)
                system_id = lowered[tail[1].lower()]
                dest = out / "archived" / method / system_id / Path(*tail[2:])
                manifest["archived"][method][system_id].append({"path": str(dest.relative_to(out)),
                                                                **copy_member(archive, member, dest)})
        print(f"archived done after {time.perf_counter() - started:.0f} s", flush=True)

    # Model inputs from the public sequences, SMILES and the archived MSAs.
    problems = {}
    for method, rows in by_method.items():
        for system_id, row in rows.items():
            issues = []
            record = inputs[system_id]
            (chain, seq), = record["sequences"].items()
            smiles, = record["smiles"]
            seq = seq.replace("-", "").upper()
            if seq != "".join(seq.split()) or len(seq) != row["length"] or smiles != row["smiles"]:
                issues.append("input sequence or SMILES differs from the selection config")
            ref = out / "refs" / system_id
            if not (ref / "receptor.cif").is_file():
                issues.append("no receptor.cif")
            if not list((ref / "ligand_files").glob("*.sdf")):
                issues.append("no ligand sdf")
            msa_dir = out / "msa" / system_id
            if method in ("boltz2", "boltz"):
                csv = msa_dir / f"{chain}.csv"
                if not csv.is_file():
                    found = sorted(msa_dir.glob("*.csv"))
                    csv = found[0] if len(found) == 1 else None
                if csv is None:
                    issues.append("no Boltz MSA csv")
                else:
                    dest = out / "inputs" / method / system_id / "input.yaml"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text("sequences:\n"
                                    "- protein:\n    id: [A]\n"
                                    f"    sequence: {seq}\n    msa: {csv.resolve()}\n"
                                    "- ligand:\n    id: [B]\n"
                                    f"    smiles: {yaml_quote(smiles)}\n", encoding="utf-8")
                    manifest["inputs"][f"{method}/{system_id}"] = {"path": str(dest.relative_to(out)), "msa": str(csv.resolve()),
                                                                 "sha256": sha256(dest)}
            else:
                dirs = sorted(d for d in msa_dir.iterdir() if d.is_dir() and (d / "pairing.a3m").is_file()
                              and (d / "non_pairing.a3m").is_file()) if msa_dir.is_dir() else []
                if len(dirs) != 1:
                    issues.append(f"Protenix MSA dirs found: {[d.name for d in dirs]}")
                else:
                    dest = out / "inputs" / "protenix" / f"{system_id}.json"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    payload = [{"name": system_id, "sequences": [
                        {"proteinChain": {"count": 1, "sequence": seq,
                                          "msa": {"precomputed_msa_dir": str(dirs[0].resolve()), "pairing_db": "uniprot"}}},
                        {"ligand": {"ligand": smiles, "count": 1}}]}]
                    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                    manifest["inputs"][f"protenix/{system_id}"] = {"path": str(dest.relative_to(out)),
                                                                   "msa": str(dirs[0].resolve()), "sha256": sha256(dest)}
            if not args.skip_predictions and not manifest["archived"][method][system_id]:
                issues.append("no archived RNP files")
            if issues:
                problems[f"{method}/{system_id}"] = issues
    manifest["problems"] = problems
    manifest["wall_seconds"] = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=True)
    (out / args.manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {m: {s: len(v) for s, v in rows.items()} for m, rows in manifest["archived"].items()}
    print(json.dumps({"systems": len(systems), "problems": problems,
                      "archived_files_per_system": {m: sorted(set(c.values())) for m, c in counts.items()},
                      "wall_seconds": manifest["wall_seconds"]}))
    if problems:
        raise SystemExit(f"{len(problems)} method/system pairs incomplete")


if __name__ == "__main__":
    main()
