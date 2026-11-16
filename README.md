# Cofolding seed selection

Code and compact numerical results for evaluating sampling budgets and native-confidence selection in protein–ligand cofolding. The analyses distinguish the quality of the returned pose from the best available pose and measure mean positive quality shortfall (MPQS) when reducing the seed budget.

## Contents

- `scripts/paper_v1_*.py`: seven-variant, one-to-five-seed evaluation, confidence stopping, sensitivity analyses, timing conversion and figure/table generation.
- `scripts/seedext_v1_*.py`: Chai-1 twenty-seed panel selection, extraction, scoring and analysis.
- `scripts/seedext_v2_*.py`: Protenix, Boltz-2 and Boltz-1 twenty-seed panels, inference, scoring and analysis.
- `scripts/profile_chai.py`: Chai-1 batch inference and timing.
- `configs/`: frozen system selections and seeds.
- `results/`: compact numerical tables used by the plotting scripts; runtime logs, raw coordinates and model weights are not included.
- `tests/`: analytical-expectation, timing-parser and archive-extraction regression tests.

## Environment

Use Python 3.12 for analysis and plotting. Install the analysis dependencies with `python -m pip install -r requirements.txt`. The exact plotting environment is frozen in `environments/paperfig/uv.lock`; with uv, use `uv sync --project environments/paperfig --frozen` and its `.venv/bin/python` for plotting.

Inference requires separate environments: Chai-1 `chai-lab==0.6.1`, Protenix `0.3.4`, Boltz-2 `boltz==2.0.3`, and Boltz-1 `boltz==0.4.1`. Environment definitions and locks for the latter three are in `environments/`. Obtain each model and its weights under its upstream terms. OST scoring requires OpenStructure 2.8.0; `ost_cli_contract.py` verifies the CLI source and its 6 Å LDDT-PLI default. The scoring scripts also retain the original executable and compound-library SHA-256 checks; a differently built OST installation will fail these checks and must be validated before adapting them.

## Recreate numerical figures and tables

From the repository root, with the plotting environment activated:

```bash
python scripts/paper_v1_figures.py
python scripts/paper_v1_marginal_figures.py
python scripts/seedext_v2_paper.py
mkdir -p paper/generated
python scripts/paper_v1_tables.py
python scripts/paper_v1_marginal_tables.py
python scripts/seedext_v1_tables.py
python scripts/seedext_v2_paper_details.py
```

Generated plots go to `results/paper_v1/figures/` and `results/seedext_v2/paper/`. The table generators write LaTeX fragments; the manuscript itself and illustrative overview/TOC artwork are not distributed here. Existing numerical result files are frozen inputs; use output-directory options when regenerating analyses to preserve them.

## Recompute the five-seed analyses

Download the following files from [Runs N’ Poses, Zenodo record 18366081](https://zenodo.org/records/18366081) into `data/raw/`:

| File | MD5 |
| --- | --- |
| `runs_n_poses_predictions.tar.gz` | `58648b836ff9434bc929b7fad25cd9bb` |
| `runs_n_poses_annotations.csv` | `793c711638ba20ae40518ed0e36b0a56` |
| `runs_n_poses_inputs.json` | `553f1723fee7f9f51de5764d66a6bbfa` |

These are upstream inputs and are not redistributed. Consult the upstream data and model-output terms, including the separate AlphaFold 3 output terms.

```bash
python scripts/paper_v1_budget_eval.py
python scripts/paper_v1_supplement.py
python scripts/paper_v1_robustness.py
python scripts/paper_v1_marginal_eval.py
```

Run these analyses in this order. The marginal and timing-conversion analyses use the included measured `results/paper_v1/tables/timing_systems.csv`; collecting new timing data is separate GPU work. `--help` lists input and output options. Cluster splits, candidate filtering, native-score selection and frozen numerical thresholds are implemented in `paper_v1_budget_eval.py`; changing them changes the evaluation.

## Twenty-seed inference and evaluation

The JSON configurations contain the original five archived seeds and fifteen extension seeds, selected systems and confidence tiers. `seedext_v1_select.py` and `seedext_v2_select.py` reproduce panel selection from the upstream inputs. Use the dedicated `seedext_v2_boltz1_systems.json` for the Boltz-1 panel.

Download the upstream `ground_truth.tar.gz`, `prediction_files.tar.gz` and `msa_files.tar.gz` archives separately when extracting structures/MSAs. Extraction scripts accept archive paths explicitly; run `--help` first. Their default archive directory is `data/runs_n_poses_v6/`.

Set `COFOLD_STORAGE` to a writable project data directory (default: this repository), `COFOLD_OST_ENV` to the validated OST installation prefix, and `COFOLD_PROTENIX_ENV`, `COFOLD_BOLTZ2_ENV`, `COFOLD_BOLTZ_ENV` to the respective inference environment prefixes. The latter default to `$COFOLD_STORAGE/env/protenix`, `env/boltz2` and `env/boltz1`. Set Chai's `CHAI_DOWNLOADS_DIR` and a writable `TMPDIR` before running `profile_chai.py` inside the Chai environment.

Run the extension seeds with `profile_chai.py` (Chai) or `seedext_v2_run.py` (other models). These generate prediction files and per-seed timing profiles. Score archived and new predictions with `seedext_v1_score.py` or `seedext_v2_score.py`, then pass the score CSVs and inference output directory to the matching `*_eval.py`. Run `--help` for required arguments. Model inference needs suitable GPU resources; archive extraction, scoring and analysis are separate steps. Raw extension coordinates and per-candidate score archives are not bundled in this code release, so a complete independent reanalysis of those panels requires regenerating or separately obtaining them.

The analysis uses reference-derived quality labels only for evaluation, not for native-confidence selection. Missing candidates and incomplete systems are tracked explicitly. An MPQS tolerance constrains a mean over systems and does not bound the loss for every individual system.

## Tests

```bash
python -m pytest -q tests
```

Tests requiring upstream prediction data skip when those files are absent. Inference runners record per-seed timings and errors.
