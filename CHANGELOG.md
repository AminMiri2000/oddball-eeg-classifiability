# Changelog

## 1.0.0 — public release preparation

### Structure

- Reusable code is now an installable package, `oddball_eeg` (`src/oddball_eeg/`),
  with a `pyproject.toml`. The former root-level modules map as follows:

  | Before | After |
  |---|---|
  | `config.py` | `oddball_eeg.config` |
  | `bss_utils.py` | `oddball_eeg.bids` (dataset discovery, event presets, preprocessing, epoching) |
  | `loader.py` | `oddball_eeg.loader` |
  | `preprocessing.py` | `oddball_eeg.preprocessing` |
  | `models.py` | `oddball_eeg.models` |
  | `utils.py` | split into `metrics`, `features`, `splits`, `training`, `summaries`, `io`, `reproducibility` |
  | `erp_utils.py` | `oddball_eeg.erp` |
  | `time_resolved.py` | `oddball_eeg.time_resolved` |
  | `visualization.py` | `oddball_eeg.visualization` |
  | `scripts/check_journal_config.py` | `oddball-eeg-check-config` command (`oddball_eeg.cli`) |

- New `oddball_eeg.paths` resolves the project root, results and data folders
  (overridable with `ODDBALL_EEG_ROOT`, `ODDBALL_EEG_RESULTS_DIR`,
  `ODDBALL_EEG_DATA_DIR`, `ODDBALL_EEG_LOGS_DIR`).
- New `oddball_eeg.stats` holds helpers previously re-defined in several
  notebooks (`benjamini_hochberg`, `safe_corr`, `zscore_within_group`,
  `strength_word`).
- Notebooks moved into `notebooks/<topic>/`, outputs stripped, legacy
  `sys.path`/`os.chdir`/`importlib.reload` boilerplate replaced by
  `use_project_root()`. Malformed names (`*.py.ipynb`, `*.ipynb.ipynb`) fixed;
  the second notebook 15 is now `15b_representative_eegnet_training_loss`.
- Tests (`tests/`), GitHub Actions CI, ruff configuration and a pre-commit
  hook that strips notebook outputs.
- `python -m oddball_eeg.cli download` fetches the OpenNeuro datasets.

### Fixes

- `get_experiment_family("<dataset>_within_subject_main")` raised because the
  development-stage control experiments were registered without the `_main`
  suffix. Notebooks 05–11 failed at their first configuration cell. The
  canonical `*_control_permuted_main` / `*_control_prestim_main` keys now exist;
  the old spellings remain as aliases.
- Event recoding in `make_epochs_std_tgt` is now done from masks computed
  before relabelling. Previously, standards were recoded to `1` before targets
  were recoded, so a target annotation that MNE happened to assign event id `1`
  would have turned every standard into a target. In that case a subject would
  have had a single class and the pipeline would have stopped with an error, so
  results produced by the old code are unaffected.
- `save_json` now serialises NumPy scalars and arrays.
- A local variable in `load_subject_epochs_from_bids` no longer shadows the
  module name it was imported under.

### Behaviour-preserving guarantees

- All 96 previously registered experiment keys resolve to identical
  configurations and identical result directories (checked against the
  previous `config.py`).
- Event-code presets, preprocessing, cross-validation, models and metrics are
  unchanged.
- Notebook analysis code, parameters and output locations are unchanged except
  for notebook 14, whose supervisor e-mail draft cell was removed and whose
  checklist file is now `replication_checklist.json`.

### Removed (unused)

- Sparse-autoencoder-era helpers in `bss_utils.py` (`EpochDataset`,
  `compute_channel_norm`, `make_weighted_train_loader`, `epochs_to_xy`,
  `baseline_latent`, `diff_from_latent`, `diff_from_epochs_mean`,
  `diff_from_ica_sources`, `snr_db_from_diff`, `topk_stats`, `cohen_d`,
  `trial_score`, `pick_roi_indices`) and the unused ICA cache
  (`ICACache`, `precompute_ica_epochs`, `ica_denoise_subject_epochs`).
  The exploratory ICA analysis (notebook 18) has its own implementation.
- Named module-level constants for individual experiments
  (`DS003061_WITHIN_SUBJECT_N15`, ...); use `get_experiment(key)`.
