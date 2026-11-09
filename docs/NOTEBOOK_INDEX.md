# Notebook index

The analyses are organised as Jupyter notebooks under [`notebooks/`](../notebooks),
grouped by topic. Notebook numbers are stable identifiers (they are referenced in
the thesis, in generated file names and between notebooks) and reflect the order
in which the analyses were developed; **run order is given by the dependency
column, not by folder.**

Every notebook starts with

```python
from oddball_eeg.paths import use_project_root
PROJECT_ROOT = use_project_root()
```

which switches the working directory to the repository root, so all relative
`results/...` paths resolve identically wherever the notebook lives.

Legend — **Raw**: loads the OpenNeuro EEG recordings (slow; GPU optional).
Otherwise the notebook only reads tables produced by earlier notebooks.

## Journal-facing analyses (103 subject-recordings, `max30_wide_timebin`)

| # | Notebook | Raw | Depends on | Produces |
|---|---|:-:|---|---|
| 01 | [`benchmark/01_data_qc_and_preprocessing`](../notebooks/benchmark/01_data_qc_and_preprocessing.ipynb) | ✓ | — | data-level QC per dataset |
| 02 | [`benchmark/02_within_subject_benchmark`](../notebooks/benchmark/02_within_subject_benchmark.ipynb) | ✓ | — | within-subject LDA / LogReg / EEGNet results (`<experiment>/tables/summaries`) |
| 04 | [`benchmark/04_cross_subject_loso`](../notebooks/benchmark/04_cross_subject_loso.ipynb) | ✓ | — | LOSO results |
| 14 | [`benchmark/14_cross_dataset_replication_comparison`](../notebooks/benchmark/14_cross_dataset_replication_comparison.ipynb) | | 02, 04 | cross-dataset summary tables |
| 15 | [`physiology/15_erp_sanity_checks`](../notebooks/physiology/15_erp_sanity_checks.ipynb) | ✓ | — | target-minus-standard ERPs (`thesis_artifacts/erp_sanity`) |
| 16 | [`physiology/16_time_resolved_decoding`](../notebooks/physiology/16_time_resolved_decoding.ipynb) | ✓ | — | time-resolved LDA decoding |
| 17 | [`susceptibility/17_classification_susceptibility_analysis`](../notebooks/susceptibility/17_classification_susceptibility_analysis.ipynb) | ✓ | 02 | subject-level raw/ERP/power features vs LDA AUC |
| 17b | [`susceptibility/17b_multimodel_classification_susceptibility_analysis`](../notebooks/susceptibility/17b_multimodel_classification_susceptibility_analysis.ipynb) | | 02, 17 | same features vs all three classifiers |
| 18 | [`susceptibility/18_ica_multimodel_classification_susceptibility_analysis`](../notebooks/susceptibility/18_ica_multimodel_classification_susceptibility_analysis.ipynb) | ✓ | 02 | exploratory ICA/source-quality features |
| 19 | [`susceptibility/19_susceptibility_meta_model`](../notebooks/susceptibility/19_susceptibility_meta_model.ipynb) | | 17b, 18 | meta-model of subject difficulty |
| 20 | [`eegnet/20_neural_model_diagnostics_and_baselines`](../notebooks/eegnet/20_neural_model_diagnostics_and_baselines.ipynb) | ✓ | — | EEGNet diagnostics and neural baselines |
| 21 | [`susceptibility/21_fdr_correction_susceptibility_results`](../notebooks/susceptibility/21_fdr_correction_susceptibility_results.ipynb) | | 17b, 18, 19 | within-subject FDR summaries |
| 21b | [`susceptibility/21b_neural_model_susceptibility_fdr`](../notebooks/susceptibility/21b_neural_model_susceptibility_fdr.ipynb) | | 17b, 18, 20, 21 | neural-model susceptibility with FDR |
| 22 | [`benchmark/22_widetimebin_vs_neural_benchmark`](../notebooks/benchmark/22_widetimebin_vs_neural_benchmark.ipynb) | ✓ | 20 | representation comparison behind the final benchmark choice |
| 23 | [`susceptibility/23_loso_susceptibility_raw_erp_power_ica_fdr`](../notebooks/susceptibility/23_loso_susceptibility_raw_erp_power_ica_fdr.ipynb) | | 04, 17, 17b, 18 | LOSO susceptibility with FDR |
| 24 | [`subject_quality/24_relative_importance_subject_quality_vs_classifier`](../notebooks/subject_quality/24_relative_importance_subject_quality_vs_classifier.ipynb) | | 02, 04, 17, 17b, 23 | **SQI**, cross-classifier consistency, relative importance / incremental R² |
| 26 | [`subject_quality/26_eegnet_gain_easy_hard_subjects`](../notebooks/subject_quality/26_eegnet_gain_easy_hard_subjects.ipynb) | | 02, 04, 24 | EEGNet gain vs subject quality |
| 27 | [`eegnet/27_eegnet_gaussian_augmentation_ablation`](../notebooks/eegnet/27_eegnet_gaussian_augmentation_ablation.ipynb) | ✓ | 24 | Gaussian-noise augmentation ablation (secondary) |
| 15b | [`eegnet/15b_representative_eegnet_training_loss`](../notebooks/eegnet/15b_representative_eegnet_training_loss.ipynb) | | 04 (or 27) | representative EEGNet training curve |
| 25 | [`figures/25_dissertation_final_figures`](../notebooks/figures/25_dissertation_final_figures.ipynb) | | all of the above | final figures and summary tables |

Notebooks 01, 02 and 04 are parameterised by `EXPERIMENT_NAME` near the top and
must be run once per dataset (four `*_within_subject_max30_wide_timebin` keys;
see `oddball_eeg.config.JOURNAL_WITHIN_SUBJECT_EXPERIMENTS`).

### Suggested run order

```text
01 → 02 → 04 → 14
15, 16, 20                       (independent; need raw data)
17 → 17b → 18 → 19 → 21 → 21b
20 → 22
04, 17, 17b, 18 → 23 → 24 → 26
24 → 27 → 15b
everything → 25
```

## Development / provenance notebooks (`notebooks/development/`)

These document earlier stages of the thesis on 10- and 15-subject subsets
(`*_main`, `*_max15` experiment keys). They are not confirmatory experiments for
the final 103-recording benchmark.

| # | Notebook | Role |
|---|---|---|
| 03 | `03_controls_permutation_and_prestim` | label-permutation and pre-stimulus controls (development benchmark; cited as supporting evidence only) |
| 05 | `05_final_model_selection` | model selection on the development benchmark |
| 06 | `06_interpretability_feature_importance` | broadband LDA/LogReg feature importance |
| 07 | `07_bandpower_feature_importance` | band-power feature importance and topographies |
| 08 | `08_thesis_results_figures_tables` | thesis-stage result figures |
| 09 | `09_interpretability_figures_tables` | interpretability figures (reads 06/07) |
| 10 | `10_thesis_artifacts` | thesis-stage figure/table export |
| 11 | `11_eegnet_interpretability` | EEGNet integrated-gradients interpretability |
| 12 | `12_eegnet_loso_export` | EEGNet LOSO model export |
| 13 | `13_eegnet_loso_interpretability` | interpretability of exported LOSO models (reads 12) |

## What changed relative to the thesis working copy

- Notebooks moved from the repository root into topic folders; the
  `15_erp_sanity_checks.py.ipynb`, `16_time_resolved_decoding.py.ipynb` and
  `12_eegnet_loso_export.ipynb.ipynb` suffixes were dropped, and the second
  notebook numbered 15 became `15b_representative_eegnet_training_loss`.
- Imports now come from the installable `oddball_eeg` package; the
  `sys.path`/`os.chdir`/`importlib.reload` boilerplate was replaced by
  `use_project_root()`.
- Helpers that several notebooks re-defined identically (`benjamini_hochberg`,
  `safe_corr`, `zscore_within_group`, `strength_word`, `valid_topomap_picks`)
  are imported from `oddball_eeg.stats` / `oddball_eeg.visualization`. Variants
  that differ in behaviour were left in their notebooks.
- Outputs were stripped. Analysis logic, parameters and output locations are
  unchanged.
