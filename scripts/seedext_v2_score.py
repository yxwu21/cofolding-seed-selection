#!/usr/bin/env python3
"""E25 (seed extension v2), C2/C3: score Boltz-2 and Protenix candidates with OpenStructure as RNP did.

Every candidate is scored with the pinned OpenStructure 2.8.0 command of Runs N' Poses
(examples/utils/analyze_models_{boltz,protenix}.sh at commit 197fafc), default 6 A radius:

  boltz2, boltz (Boltz-1)
            ost compare-ligand-structures -m <model.cif> -rl <ref>/ligand_files/*.sdf \
                -r <ref>/receptor.cif -o <json> --lddt-pli --rmsd --lddt-pli-amc
  protenix  as above with the model split into a receptor mmCIF (-m) and a ligand SDF (-ml),
            the ligand's bond orders assigned from the input SMILES (RNP passed receptor and
            ligand files separately for Protenix); --protenix-ligand cif scores the model
            mmCIF directly instead (used where the archived RNP file already carries the ligand).
  boltz     as boltz2, after renaming the ligand residue LIG to LIG1 in _atom_site: boltz 0.4.1
            names SMILES ligands LIG, a real CCD component, so OST builds the ligand graph from
            the compound library and the graph match fails for every candidate; under LIG1 OST
            infers the bonds as for Boltz-2 (which writes LIG1). On the 25 archived RNP Boltz-1
            structures of 7fu6__1__1.A__1.C this reproduced the RNP labels within 0.0025, and
            receptor/ligand splitting gave the same values.

One row per candidate records the LDDT-PLI and RMSD of every assigned reference ligand and
the native ranking score (Boltz-2: confidence_score of confidence_*.json; Protenix:
ranking_score of *_summary_confidence_sample_*.json); failures are kept as missing values
with their reason, never as zero.

Modes:
  archived  RNP structures of the method under <data-root>/archived/<method>/<system>/
            (seed-<seed>_sample-<n>.cif), for the scoring parity check (C2)
  new       seedext_v2_run.py outputs under --run (<run>/[task_*/|<method>/]<system>/
            batch_*_seed_<seed>/...), for the analysis (C3)

Run with the Chai environment (numpy, gemmi, RDKit):
  python -I scripts/seedext_v2_score.py ...
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
FIELDS = ["method", "mode", "system_id", "seed", "sample", "ranking_score", "status", "lddt_pli", "rmsd",
          "n_ref_ligands", "seconds", "error", "ligand_mode", "model_path", "model_ligand_path", "json_path"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_dirs(run: Path, method: str, system_id: str) -> list[Path]:
    found = set()
    for pattern in (f"{system_id}/batch_*_seed_*", f"task_*/{system_id}/batch_*_seed_*",
                    f"{method}/{system_id}/batch_*_seed_*"):
        found.update(p for p in run.glob(pattern) if p.is_dir())
    return sorted(found)


def candidates(method: str, mode: str, config: dict, data_root: Path, run: Path | None) -> list[dict]:
    rows = []
    systems = config["methods"][method]["systems"]
    if mode == "archived":
        for row in systems:
            folder = data_root / "archived" / method / row["system_id"]
            archived_scores = {}
            if (folder / "ranking_scores.csv").is_file():  # RNP archive: seed,sample,ranking_score
                with (folder / "ranking_scores.csv").open(newline="") as handle:
                    for rec in csv.DictReader(handle):
                        archived_scores[(int(rec["seed"]), int(rec["sample"]))] = float(rec["ranking_score"])
            for path in sorted(folder.glob("seed-*_sample-*.cif")):
                m = re.fullmatch(r"seed-(\d+)_sample-(\d+)\.cif", path.name)
                seed, sample = int(m.group(1)), int(m.group(2))
                rows.append({"system_id": row["system_id"], "seed": seed, "sample": sample, "model": path,
                             "score_file": None, "archived_score": archived_scores.get((seed, sample), np.nan),
                             "smiles": row["smiles"]})
        return rows
    for row in systems:
        for batch in batch_dirs(run, method, row["system_id"]):
            seed = int(re.search(r"seed_(\d+)$", batch.name).group(1))
            if method in ("boltz2", "boltz"):
                for model in sorted(batch.glob("boltz_results_*/predictions/*/*_model_*.cif")):
                    sample = int(re.search(r"_model_(\d+)\.cif$", model.name).group(1))
                    conf = model.with_name(f"confidence_{model.name.replace('.cif', '.json')}")
                    rows.append({"system_id": row["system_id"], "seed": seed, "sample": sample, "model": model,
                                 "score_file": conf, "smiles": row["smiles"]})
            else:
                for model in sorted(batch.glob(f"*/seed_{seed}/predictions/*_sample_*.cif")):
                    if model.name.endswith("_wounresol.cif"):
                        continue
                    sample = int(re.search(r"_sample_(\d+)\.cif$", model.name).group(1))
                    conf = model.with_name(re.sub(r"_sample_(\d+)\.cif$", r"_summary_confidence_sample_\1.json", model.name))
                    rows.append({"system_id": row["system_id"], "seed": seed, "sample": sample, "model": model,
                                 "score_file": conf, "smiles": row["smiles"]})
    return rows


def ranking_score(method: str, score_file: Path | None) -> float:
    if score_file is None or not Path(score_file).is_file():
        return np.nan
    data = json.loads(Path(score_file).read_text())
    key = "confidence_score" if method in ("boltz2", "boltz") else "ranking_score"
    return float(data[key]) if key in data else np.nan


def split_protenix(model: Path, smiles: str, receptor_out: Path, ligand_out: Path) -> dict:
    """Receptor mmCIF without HETATM rows and the ligand as SDF with template bond orders.

    Works on the atom_site table itself (gemmi's structure builder rejects the Protenix mmCIF),
    so every other category of the model file is kept unchanged in the receptor file.
    """
    import gemmi
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    doc = gemmi.cif.read(str(model))
    block = doc.sole_block()
    table = block.find("_atom_site.", ["group_PDB", "type_symbol", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"])
    if not table:
        raise ValueError("no _atom_site table in the model")
    atoms, hetero_rows, residues = [], [], set()
    for i, row in enumerate(table):
        if row[0] == "HETATM" and row[2] not in ("HOH", "WAT"):
            hetero_rows.append(i)
            residues.add(row[2])
            if row[1].upper() != "H":
                atoms.append((row[1], float(row[3]), float(row[4]), float(row[5])))
    if not atoms:
        raise ValueError("no HETATM (ligand) atoms in the model")
    for i in reversed(hetero_rows):
        table.remove_row(i)
    receptor_out.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(receptor_out))
    xyz = f"{len(atoms)}\n\n" + "\n".join(f"{e} {x:.4f} {y:.4f} {z:.4f}" for e, x, y, z in atoms) + "\n"
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    template = Chem.MolFromSmiles(smiles)
    if template is None or template.GetNumHeavyAtoms() != mol.GetNumAtoms():
        raise ValueError(f"ligand atom count {mol.GetNumAtoms()} differs from the SMILES heavy atoms "
                         f"{template.GetNumHeavyAtoms() if template else None}")
    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    Chem.SanitizeMol(mol)
    Chem.MolToMolFile(mol, str(ligand_out))
    return {"n_ligand_residues": len(residues), "n_ligand_atoms": len(atoms)}


def rename_ligand(model: Path, out: Path, old: str = "LIG", new: str = "LIG1") -> int:
    """Copy of the model with the ligand residue renamed in _atom_site; returns the rows changed."""
    import gemmi

    doc = gemmi.cif.read(str(model))
    table = doc.sole_block().find("_atom_site.", ["group_PDB", "label_comp_id", "auth_comp_id"])
    changed = 0
    for row in table:
        if row[0] == "HETATM" and (row[1] == old or row[2] == old):
            row[1] = new if row[1] == old else row[1]
            row[2] = new if row[2] == old else row[2]
            changed += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(out))
    return changed


def score(job: dict, method: str, mode: str, ligand_mode: str, data_root: Path, out: Path, env: dict,
          timeout: float) -> dict:
    ref = data_root / "refs" / job["system_id"]
    ligands = sorted((ref / "ligand_files").glob("*.sdf"))
    stem = f"{job['seed']}_{job['sample']}"
    json_path = out / "json" / job["system_id"] / f"{stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"method": method, "mode": mode, "system_id": job["system_id"], "seed": job["seed"], "sample": job["sample"],
           "ranking_score": job.get("archived_score", ranking_score(method, job["score_file"])), "status": "",
           "lddt_pli": "", "rmsd": "",
           "n_ref_ligands": len(ligands), "seconds": np.nan, "error": "", "ligand_mode": ligand_mode,
           "model_path": str(job["model"]), "model_ligand_path": "", "json_path": str(json_path)}
    started = time.perf_counter()
    try:
        model_arg, extra = job["model"], []
        if method == "protenix" and ligand_mode == "split":
            receptor = json_path.with_name(f"{stem}_receptor.cif")
            ligand = json_path.with_name(f"{stem}_ligand.sdf")
            split_protenix(job["model"], job["smiles"], receptor, ligand)
            model_arg, extra = receptor, ["-ml", str(ligand)]
            row["model_ligand_path"] = str(ligand)
        elif method == "boltz" and ligand_mode == "rename":
            renamed = json_path.with_name(f"{stem}_lig1.cif")
            if rename_ligand(job["model"], renamed) == 0:
                raise ValueError("no HETATM rows named LIG in the Boltz-1 model")
            model_arg = renamed
            row["model_ligand_path"] = str(renamed)
        argv = [str(OST), "compare-ligand-structures", "-m", str(model_arg), *extra, "-rl", *map(str, ligands),
                "-r", str(ref / "receptor.cif"), "-o", str(json_path), "--lddt-pli", "--rmsd", "--lddt-pli-amc"]
        log = json_path.with_suffix(".log")
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
    parser.add_argument("--method", choices=["boltz2", "protenix", "boltz"], required=True)
    parser.add_argument("--mode", choices=["archived", "new"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seedext_v2_systems.json")
    parser.add_argument("--data-root", type=Path, default=STORAGE / "data/seedext_v2")
    parser.add_argument("--run", type=Path, help="seedext_v2_run.py output root (mode new)")
    parser.add_argument("--protenix-ligand", choices=["split", "cif"], default="split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--limit", type=int, default=None, help="score only the first n candidates (smoke test)")
    parser.add_argument("--systems", nargs="*", default=None, help="restrict to these system ids (smoke test)")
    args = parser.parse_args()
    if args.mode == "new" and args.run is None:
        parser.error("--run is required in mode new")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected in ((OST, OST_SHA256), (CHEMLIB, CHEMLIB_SHA256)):
        if sha256(path) != expected:
            raise SystemExit(f"{path} does not match the pinned sha256")
    cli = verify_radius_default(OST)
    env = dict(os.environ, OST_COMPOUNDS_CHEMLIB=str(CHEMLIB), OMP_NUM_THREADS="1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = candidates(args.method, args.mode, config, args.data_root, args.run)
    if args.systems:
        jobs = [j for j in jobs if j["system_id"] in set(args.systems)]
    if args.limit:
        jobs = jobs[:args.limit]
    ligand_mode = {"protenix": args.protenix_ligand, "boltz": "rename"}.get(args.method, "cif")
    print(json.dumps({"method": args.method, "mode": args.mode, "candidates": len(jobs), "ligand_mode": ligand_mode,
                      "workers": args.workers}), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score, job, args.method, args.mode, ligand_mode, args.data_root, args.output_dir, env,
                               args.timeout) for job in jobs]
        for i, future in enumerate(futures, start=1):
            rows.append(future.result())
            if i % 200 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} scored after {time.perf_counter() - started:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["system_id"], r["seed"], r["sample"]))
    with (args.output_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    manifest = {"method": args.method, "mode": args.mode, "ligand_mode": ligand_mode, "n_candidates": len(rows),
                "statuses": statuses, "n_with_error_field": sum(1 for r in rows if r["error"]),
                "ost": str(OST), "ost_sha256": OST_SHA256, "chemlib_sha256": CHEMLIB_SHA256, "ost_cli": cli,
                "config": str(args.config), "run": str(args.run), "wall_seconds": time.perf_counter() - started}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
