"""Writing result tables, training histories and run metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_history_dict(
    histories: dict,
    out_dir: str | Path,
    filename_prefix: str = "history",
) -> None:
    """
    Save nested training-history DataFrames.

    Expected shape:
        histories[subject][fold_id] -> pd.DataFrame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject, fold_histories in histories.items():
        for fold_id, hist_df in fold_histories.items():
            path = out_dir / f"{filename_prefix}_subject-{subject}_fold-{fold_id}.csv"
            hist_df.to_csv(path, index=False)


def save_json(obj: dict, path: str | Path) -> None:
    """
    Save a JSON object, creating parent directories.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, Path):
            return str(o)
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def save_experiment_metadata(cfg, extra: dict | None = None) -> None:
    """
    Save experiment configuration and optional runtime notes.
    """
    payload = cfg.to_dict()

    if extra is not None:
        payload["extra"] = dict(extra)

    save_json(payload, cfg.metadata_dir() / "experiment_config.json")
