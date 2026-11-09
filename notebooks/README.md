# Notebooks

Analysis notebooks, grouped by topic. See
[docs/NOTEBOOK_INDEX.md](../docs/NOTEBOOK_INDEX.md) for the role of each notebook,
its inputs and the run order.

- Install the package first (`pip install -e ".[notebooks]"` from the repository root).
- Every notebook starts with `use_project_root()`, so it can be opened from any folder.
- Notebooks are committed **without outputs**. The pre-commit hook (`pre-commit install`)
  strips them automatically; or run `nbstripout notebooks/**/*.ipynb`.
