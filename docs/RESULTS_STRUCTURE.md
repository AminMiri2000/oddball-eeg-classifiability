# Generated results structure

Generated results are intentionally excluded from version control by default because the full benchmark produces many tables, histories, checkpoints, figures, and intermediate artifacts.

`oddball_eeg.config.ExperimentConfig.result_dir()` constructs result paths under (override the root with `ODDBALL_EEG_RESULTS_DIR`):

```text
results/
└── <dataset>/
    └── <task>/
        └── events_<preset>/
            └── <signal-config>/
                └── <run-tag>/
                    └── <crop-config>/
                        └── <feature-config>/
                            └── <evaluation-mode>/
                                └── <control-mode>/
                                    ├── tables/
                                    ├── figures/
                                    ├── histories/
                                    ├── checkpoints/
                                    ├── artifacts/
                                    └── metadata/
```

Seed-specific EEGNet outputs may appear below an additional `seeds/seed_XXX/` component.

Notebooks 14–27 additionally write cross-dataset artifacts to `results/thesis_artifacts/<analysis-name>/{tables,figures}/`; the directory names are listed per notebook in [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md).

## Version-control policy

- Raw datasets are never committed.
- Logs, checkpoints, and large generated results are not committed.
- `results/derived/` is reserved for small, redistributable journal-support tables when they are regenerated.
- Historical generated result files that were not retained are not recreated manually.

The journal release emphasizes reproducible code, fixed configurations, and transparent provenance over reconstruction of missing historical working-directory artifacts.
