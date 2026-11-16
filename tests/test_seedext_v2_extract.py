"""E25 extraction: archived predictions of methods outside the configuration are skipped."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/seedext_v2_extract.py"


def add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def test_boltz1_config_skips_other_methods(tmp_path):
    """A Boltz-1-only configuration must not fail on Protenix or Boltz-2 members of the same system."""
    predictions = tmp_path / "prediction_files.tar.gz"
    with tarfile.open(predictions, "w:gz") as archive:
        add(archive, "rnp/prediction_files/protenix/sys1/seed-1_sample-0.cif", b"protenix")
        add(archive, "rnp/prediction_files/boltz/sys1/seed-1_sample-0.cif", b"boltz")
        add(archive, "rnp/prediction_files/boltz2/sys1/seed-1_sample-0.cif", b"boltz2")
        add(archive, "rnp/prediction_files/boltz/other/seed-1_sample-0.cif", b"not configured")
    config = tmp_path / "systems.json"
    config.write_text(json.dumps({"methods": {"boltz": {"systems": [
        {"system_id": "sys1", "length": 3, "smiles": "C"}]}}}))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"sys1": {"sequences": {"A": "ACD"}, "smiles": ["C"]}}))
    out = tmp_path / "out"
    (out / "refs/sys1/ligand_files").mkdir(parents=True)
    (out / "refs/sys1/receptor.cif").write_text("receptor")
    (out / "refs/sys1/ligand_files/lig.sdf").write_text("ligand")
    (out / "msa/sys1").mkdir(parents=True)
    (out / "msa/sys1/A.csv").write_text("key,sequence\n-1,ACD\n")

    done = subprocess.run([sys.executable, "-I", str(SCRIPT), "--systems", str(config), "--inputs", str(inputs),
                           "--predictions", str(predictions), "--output-root", str(out), "--skip-ground-truth",
                           "--skip-msa", "--manifest-name", "manifest.json"], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert (out / "archived/boltz/sys1/seed-1_sample-0.cif").read_bytes() == b"boltz"
    assert not (out / "archived/protenix").exists() and not (out / "archived/boltz2").exists()
    assert not (out / "archived/boltz/other").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["problems"] == {}
    assert [e["path"] for e in manifest["archived"]["boltz"]["sys1"]] == ["archived/boltz/sys1/seed-1_sample-0.cif"]
    assert (out / "inputs/boltz/sys1/input.yaml").is_file()
