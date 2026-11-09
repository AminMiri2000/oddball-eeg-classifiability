"""Project-level path resolution.

All generated outputs and the default dataset location are resolved relative to
the *project root*, i.e. the repository checkout that contains
``pyproject.toml``. This keeps the result layout identical regardless of where a
notebook or script is launched from.

Resolution order
----------------
Project root
    1. ``ODDBALL_EEG_ROOT`` environment variable, if set.
    2. The nearest parent of the current working directory that contains the
       project marker files.
    3. The repository containing this source file (editable installs).
    4. The current working directory.

Results directory
    ``ODDBALL_EEG_RESULTS_DIR`` if set, otherwise ``<project root>/results``.

Data directory
    ``ODDBALL_EEG_DATA_DIR`` if set, otherwise
    ``<project root>/datasets/openneuro``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ENV_ROOT",
    "ENV_RESULTS",
    "ENV_DATA",
    "ENV_LOGS",
    "project_root",
    "results_dir",
    "logs_dir",
    "data_dir",
    "use_project_root",
]

ENV_ROOT = "ODDBALL_EEG_ROOT"
ENV_RESULTS = "ODDBALL_EEG_RESULTS_DIR"
ENV_DATA = "ODDBALL_EEG_DATA_DIR"
ENV_LOGS = "ODDBALL_EEG_LOGS_DIR"

_PACKAGE_NAME = "oddball_eeg"


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / _PACKAGE_NAME).is_dir()


def project_root() -> Path:
    """Return the repository root used to anchor results and data paths."""
    env = os.environ.get(ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_project_root(candidate):
            return candidate

    # Editable install: <root>/src/oddball_eeg/paths.py
    source_root = Path(__file__).resolve().parents[2]
    if _is_project_root(source_root):
        return source_root

    return cwd


def results_dir() -> Path:
    """Directory under which all generated tables, figures and artifacts are written."""
    env = os.environ.get(ENV_RESULTS)
    return Path(env).expanduser().resolve() if env else project_root() / "results"


def logs_dir() -> Path:
    """Directory for run logs."""
    env = os.environ.get(ENV_LOGS)
    return Path(env).expanduser().resolve() if env else project_root() / "logs"


def data_dir() -> Path:
    """Default parent directory of the OpenNeuro BIDS datasets."""
    env = os.environ.get(ENV_DATA)
    return Path(env).expanduser().resolve() if env else project_root() / "datasets" / "openneuro"


def use_project_root(verbose: bool = True) -> Path:
    """Change the working directory to the project root and return it.

    The analysis notebooks use paths relative to the project root, e.g.
    ``Path("results") / "thesis_artifacts"``. Calling this at the top of a
    notebook makes those paths resolve identically whichever folder the
    notebook lives in.
    """
    root = project_root()
    if not _is_project_root(root) and os.environ.get(ENV_ROOT) is None:
        raise FileNotFoundError(
            "Could not locate the project root (a directory containing "
            f"pyproject.toml and src/{_PACKAGE_NAME}). Launch Jupyter from inside the "
            f"repository or set the {ENV_ROOT} environment variable."
        )
    os.chdir(root)
    if verbose:
        print(f"Project root: {root}")
    return root
