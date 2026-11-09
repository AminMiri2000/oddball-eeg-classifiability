# Quantifying Subject Classifiability in Oddball EEG

[![CI](https://github.com/AminMiri2000/oddball-eeg-classifiability/actions/workflows/ci.yml/badge.svg)](https://github.com/AminMiri2000/oddball-eeg-classifiability/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Reproducibility code for:

**Quantifying Subject Classifiability in Oddball EEG: A Cross-Dataset Methodological Comparison of Signal Quality and Classifier Effects**

Seyed Mohammad Amin Miri, Russell Butler, and Rachid Hedjam
Department of Computer Science, Bishop's University, Canada

## Overview

We study subject-level classifiability in oddball / event-related-potential (ERP)
EEG across four public OpenNeuro datasets (103 subject-recordings). Three
classifiers — shrinkage **LDA**, **logistic regression** and **EEGNet** — are
evaluated under grouped **within-subject** and **leave-one-subject-out (LOSO)**
cross-validation. The analyses then ask how much of the variation in decoding
performance is explained by the classifier versus by the recording itself, using
recording-level susceptibility features and a retrospective **Subject Quality
Index (SQI)**.

## Repository layout

```text
src/oddball_eeg/     installable analysis package (data loading, CV, models, metrics)
notebooks/           analysis notebooks, grouped by topic
  benchmark/         QC, within-subject and LOSO benchmarks, cross-dataset summary
  physiology/        ERP and time-resolved decoding plausibility checks
  susceptibility/    subject-level feature correlations, meta-model, FDR
  subject_quality/   SQI, relative importance, EEGNet gain vs subject quality
  eegnet/            EEGNet diagnostics, augmentation ablation, training curves
  figures/           final figures and tables
  development/       earlier 10/15-subject development stages (provenance)
tests/               unit and smoke tests (synthetic data; no download needed)
docs/                datasets, reproduction workflow, notebook index, results layout
results/             generated outputs (git-ignored except results/derived/)
```

See [docs/NOTEBOOK_INDEX.md](docs/NOTEBOOK_INDEX.md) for what each notebook does
and the order to run them in.

## Installation

```bash
git clone https://github.com/AminMiri2000/oddball-eeg-classifiability.git
cd oddball-eeg-classifiability

conda env create -f environment.yml          # Python 3.10 + pinned scientific stack
conda activate oddball-eeg-classifiability   # installs this package in editable mode
```

or, in any Python ≥ 3.10 environment:

```bash
pip install -e ".[notebooks]"
```

Verify the installation (no EEG data required):

```bash
oddball-eeg-check-config   # checks the final experiment registry (103 recordings)
pytest                     # unit and smoke tests on synthetic data
```

## Data

The four OpenNeuro datasets are **not** redistributed here.

| Dataset | Paradigm | Subjects |
|---|---|---:|
| [`ds003061`](https://openneuro.org/datasets/ds003061) | auditory oddball (P300) | 13 |
| [`ds005863`](https://openneuro.org/datasets/ds005863) | visual oddball | 30 |
| [`ds006466`](https://openneuro.org/datasets/ds006466) | active auditory oddball | 30 |
| [`ds006480`](https://openneuro.org/datasets/ds006480) | auditory oddball (pooled conditions) | 30 |

Place them under `datasets/openneuro/<dataset-id>/` in the repository, or point
`ODDBALL_EEG_DATA_DIR` at the folder that contains them. They can be fetched with

```bash
pip install -e ".[data]"
python -m oddball_eeg.cli download            # all four datasets
```

Subject lists and event-code mappings are in [docs/DATASETS.md](docs/DATASETS.md).

## Reproducing the study

The final configuration is the `max30_wide_timebin` experiment family:
50-ms time-binned 0–0.8 s features for LDA and logistic regression, raw 0–0.8 s
epochs for EEGNet, 0.1–12 Hz band-pass, 128 Hz, seeds 42/43/44.

```python
from oddball_eeg.config import JOURNAL_WITHIN_SUBJECT_EXPERIMENTS, get_experiment_family

family = get_experiment_family(JOURNAL_WITHIN_SUBJECT_EXPERIMENTS[0])
family["main"].result_dir()   # results/ds003061/P300/.../within_subject/standard
```

Run the notebooks in the order given in [docs/NOTEBOOK_INDEX.md](docs/NOTEBOOK_INDEX.md);
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) walks through each stage.
Generated outputs are written below `results/` (layout in
[docs/RESULTS_STRUCTURE.md](docs/RESULTS_STRUCTURE.md)); override with
`ODDBALL_EEG_RESULTS_DIR`.

### Using the package directly

```python
from oddball_eeg.bids import make_epoching_config_from_experiment
from oddball_eeg.config import get_experiment
from oddball_eeg.loader import load_subjects_grouped
from oddball_eeg.preprocessing import apply_crop_to_loaded_dataset
from oddball_eeg.splits import build_folds_by_subject
from oddball_eeg.training import evaluate_classical_benchmark

cfg = get_experiment("ds003061_within_subject_max30_wide_timebin")
data = load_subjects_grouped(
    dataset=cfg.dataset.dataset,
    task_name=cfg.dataset.task,
    subjects=cfg.subjects.subjects,
    epoching_cfg=make_epoching_config_from_experiment(cfg),
    chunk_size=cfg.splits.group_chunk_size,
)
data = apply_crop_to_loaded_dataset(data, *cfg.effective_crop_window())
folds = build_folds_by_subject(
    data, n_splits=cfg.splits.n_splits, purge_neighbors=cfg.splits.purge_neighbor_groups
)
results = evaluate_classical_benchmark(
    data, folds, feature_mode=cfg.feature.mode, bin_ms=cfg.feature.bin_ms
)
```

## Reproducibility notes

- Raw EEG, model checkpoints, logs and large intermediate outputs are not stored
  in the repository. The code regenerates the analyses from the public datasets.
- Some intermediate CSV files from the thesis working directory were not
  retained in version control and have **not** been reconstructed.
- Small redistributable derived tables may be placed in `results/derived/`.
- `environment.yml` pins the versions used for the reported analyses;
  `environment-full.yml` is the full development environment, kept for
  provenance.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff). The article DOI and the
archived software DOI will be added on publication.

## License

Code: BSD 3-Clause ([LICENSE](LICENSE)). The OpenNeuro datasets remain subject to
their own licenses and terms of use.
