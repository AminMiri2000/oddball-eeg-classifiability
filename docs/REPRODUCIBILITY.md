# Reproducing the analyses

This document describes the regeneration path for the journal analysis. Reusable code lives in the `oddball_eeg` package (`src/oddball_eeg/`); the analyses themselves are the notebooks under `notebooks/`, indexed in [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md).

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate oddball-eeg-classifiability
```

The environment installs this repository in editable mode (`pip install -e .`). The exact development environment is preserved in `environment-full.yml` for provenance.

Launch Jupyter from anywhere inside the repository. Each notebook calls `oddball_eeg.paths.use_project_root()` first, so relative `results/...` paths always resolve against the repository root.

## 2. Obtain the public datasets

Download the four OpenNeuro datasets listed in [DATASETS.md](DATASETS.md) (`python -m oddball_eeg.cli download` after `pip install -e ".[data]"`) and place them under:

```text
datasets/openneuro/<dataset-id>/
```

or set `ODDBALL_EEG_DATA_DIR` to the folder that contains the dataset directories.

## 3. Verify final experiment configuration

Before loading EEG data, run the lightweight registry check:

```bash
oddball-eeg-check-config
pytest
```

The first command verifies the four final within-subject and LOSO experiment keys and their expected subject counts without loading data; the test suite exercises loading, epoching, cross-validation and all three classifiers on synthetic data.

The final journal benchmark uses the `max30_wide_timebin` experiment family defined in `oddball_eeg.config`.

- LDA / Logistic Regression: 50-ms nominal time bins from 0.0 to 0.8 s
- EEGNet: raw 0.0--0.8 s post-stimulus epochs
- datasets: 13 + 30 + 30 + 30 = 103 subject-recordings
- neural base seeds: 42, 43, 44

The convenience tuples `JOURNAL_WITHIN_SUBJECT_EXPERIMENTS` and `JOURNAL_LOSO_EXPERIMENTS` identify these final experiment keys.

## 4. Cross-dataset classifier benchmark

Run `benchmark/01_data_qc_and_preprocessing.ipynb`, `benchmark/02_within_subject_benchmark.ipynb` and `benchmark/04_cross_subject_loso.ipynb` once per dataset, setting `EXPERIMENT_NAME` to each of the four `*_within_subject_max30_wide_timebin` keys. Then summarise with:

- `benchmark/14_cross_dataset_replication_comparison.ipynb`
- `benchmark/22_widetimebin_vs_neural_benchmark.ipynb`

`benchmark/22_widetimebin_vs_neural_benchmark.ipynb` additionally needs `eegnet/20_neural_model_diagnostics_and_baselines.ipynb`.

## 5. Physiological plausibility checks

Run:

- `physiology/15_erp_sanity_checks.ipynb`
- `physiology/16_time_resolved_decoding.ipynb`

These produce the target-minus-standard ERP and time-resolved decoding analyses used as physiological plausibility checks.

## 6. Subject-level susceptibility analyses

The relevant late-stage notebooks are:

- `susceptibility/17_classification_susceptibility_analysis.ipynb`
- `susceptibility/17b_multimodel_classification_susceptibility_analysis.ipynb`
- `susceptibility/18_ica_multimodel_classification_susceptibility_analysis.ipynb`
- `susceptibility/21_fdr_correction_susceptibility_results.ipynb`
- `susceptibility/21b_neural_model_susceptibility_fdr.ipynb`
- `susceptibility/23_loso_susceptibility_raw_erp_power_ica_fdr.ipynb`

Raw/ERP/power measures form the principal susceptibility family; ICA/source-quality measures are exploratory.

## 7. Subject Quality Index and relative importance

Run:

- `subject_quality/24_relative_importance_subject_quality_vs_classifier.ipynb`

This notebook contains the retrospective SQI construction and the relative-importance analyses, including cross-classifier subject consistency and subject-grouped cross-validation.

## 8. EEGNet diagnostics and subject-quality gain

Use:

- `eegnet/15b_representative_eegnet_training_loss.ipynb`
- `eegnet/20_neural_model_diagnostics_and_baselines.ipynb`
- `subject_quality/26_eegnet_gain_easy_hard_subjects.ipynb`

## 9. Gaussian-noise augmentation ablation

Use:

- `eegnet/27_eegnet_gaussian_augmentation_ablation.ipynb`

The principal benchmark itself is non-augmented; this is a secondary robustness analysis.

## 10. Final figures and tables

Use:

- `figures/25_dissertation_final_figures.ipynb`

The journal manuscript reuses selected final thesis figures in a condensed four-figure layout.

## Development-stage controls

`development/03_controls_permutation_and_prestim.ipynb` contains label-permutation and prestimulus controls from an earlier 10-subject `ds003061` development benchmark. These are supporting checks only and were not repeated across the complete 103-subject cross-dataset benchmark.

## Important reproducibility limitation

Large intermediate outputs and historical result directories were excluded from version control. Some generated CSV files used during thesis development were not retained. Therefore, this repository is designed to regenerate analyses from the original public datasets; it does not claim to contain every historical working-directory artifact.

No missing historical numerical tables have been reconstructed.
