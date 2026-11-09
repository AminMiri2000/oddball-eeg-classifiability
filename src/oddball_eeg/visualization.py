"""Notebook display, table and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# ============================================================
# General display helpers
# ============================================================


def set_pandas_display_options() -> None:
    """
    Set friendlier pandas display defaults for notebook work.
    """
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def round_numeric_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Return a copy of the dataframe with numeric columns rounded.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def valid_topomap_picks(info: mne.Info) -> np.ndarray:
    """Indices of channels with a finite, non-zero 3-D position (plottable on a topomap)."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=float)
    valid = np.isfinite(pos).all(axis=1) & (np.linalg.norm(pos, axis=1) > 0)
    return np.where(valid)[0]


# ============================================================
# Table builders
# ============================================================


def make_subject_metric_table(
    df: pd.DataFrame,
    metric_col: str = "auc_mean",
    sort_desc: bool = True,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Build a subject x model pivot table for one metric.

    Expected input columns:
    - subject
    - model
    - metric_col
    """
    required = {"subject", "model", metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pivot = df.pivot(index="subject", columns="model", values=metric_col).sort_index()
    return pivot.round(decimals)


def make_model_ranking_table(
    df: pd.DataFrame,
    metric_col: str = "mean_subject_auc",
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Build a clean model ranking table sorted by a chosen metric.
    """
    if metric_col not in df.columns:
        raise ValueError(f"metric_col '{metric_col}' not found in dataframe")

    out = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    return round_numeric_columns(out, decimals=decimals)


def normalize_subject_id(x: object) -> str:
    s = str(x).strip()
    return s.zfill(3) if s.isdigit() else s


def make_main_vs_control_table(
    df_main: pd.DataFrame,
    df_control: pd.DataFrame,
    *,
    value_col_main: str = "auc_seed_mean",
    value_col_control: str = "auc_seed_mean",
    main_label: str = "main",
    control_label: str = "control",
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Compare main benchmark vs a control benchmark at the subject/model level.
    """

    required_main = {"subject", "model", value_col_main}
    required_control = {"subject", "model", value_col_control}

    missing_main = required_main - set(df_main.columns)
    missing_control = required_control - set(df_control.columns)

    if missing_main:
        raise ValueError(f"Main dataframe missing columns: {sorted(missing_main)}")
    if missing_control:
        raise ValueError(f"Control dataframe missing columns: {sorted(missing_control)}")

    left = df_main[["subject", "model", value_col_main]].copy()
    right = df_control[["subject", "model", value_col_control]].copy()

    # Normalize subject IDs to zero-padded strings
    left["subject"] = left["subject"].map(normalize_subject_id)
    right["subject"] = right["subject"].map(normalize_subject_id)

    # Normalize model names
    left["model"] = left["model"].astype(str).str.strip().str.lower()
    right["model"] = right["model"].astype(str).str.strip().str.lower()

    left = left.rename(columns={value_col_main: f"{main_label}_value"})
    right = right.rename(columns={value_col_control: f"{control_label}_value"})

    merged = pd.merge(left, right, on=["subject", "model"], how="inner")

    if merged.empty:
        raise ValueError(
            "Comparison table is empty after merging. "
            f"Main subjects={sorted(left['subject'].unique())}, "
            f"Control subjects={sorted(right['subject'].unique())}, "
            f"Main models={sorted(left['model'].unique())}, "
            f"Control models={sorted(right['model'].unique())}"
        )

    merged["delta"] = merged[f"{main_label}_value"] - merged[f"{control_label}_value"]

    return round_numeric_columns(
        merged.sort_values(["subject", "model"]).reset_index(drop=True),
        decimals=decimals,
    )


def make_seed_stability_table(
    df_subject_seed_summary: pd.DataFrame,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Build a clean subject-level seed stability table.

    Expected columns:
    - subject
    - model
    - auc_seed_mean
    - auc_seed_std
    """
    required = {"subject", "model", "auc_seed_mean", "auc_seed_std"}
    missing = required - set(df_subject_seed_summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df_subject_seed_summary.sort_values(["subject", "model"]).reset_index(drop=True)
    return round_numeric_columns(out, decimals=decimals)


# ============================================================
# Simple printing helpers
# ============================================================


def print_subject_model_summary(
    df: pd.DataFrame,
    subject_col: str = "subject",
    model_col: str = "model",
    metric_mean_col: str = "auc_mean",
    metric_std_col: str = "auc_std",
) -> None:
    """
    Print a simple subject-wise model summary.

    Example:
        Subject 001
            lda     mean AUC = ...
            logreg  mean AUC = ...
            eegnet  mean AUC = ...
    """
    required = {subject_col, model_col, metric_mean_col, metric_std_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subjects = list(pd.Series(df[subject_col]).drop_duplicates())

    for sub in subjects:
        print("=" * 70)
        print(f"Subject {sub}")

        sub_df = df[df[subject_col] == sub].copy()
        for _, row in sub_df.iterrows():
            print(
                f"{str(row[model_col]):>8s}  "
                f"mean = {row[metric_mean_col]:.4f}  "
                f"std = {row[metric_std_col]:.4f}"
            )


def print_model_ranking(
    df: pd.DataFrame,
    metric_col: str = "mean_subject_auc",
) -> None:
    """
    Print a simple model ranking.
    """
    if "model" not in df.columns or metric_col not in df.columns:
        raise ValueError(f"Dataframe must contain 'model' and '{metric_col}' columns")

    ranked = df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    print("=" * 70)
    print(f"Model ranking by {metric_col}")
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        print(f"{i:>2d}. {row['model']:<10s} {row[metric_col]:.4f}")


# ============================================================
# Plot helpers
# ============================================================


def plot_subject_metric_bars(
    df: pd.DataFrame,
    metric_col: str = "auc_mean",
    error_col: str | None = "auc_std",
    title: str = "Per-subject model comparison",
    ylabel: str = "ROC-AUC",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot a grouped bar chart:
        x-axis = subjects
        bar color/group = model
        y-axis = chosen metric
    """
    required = {"subject", "model", metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subjects = list(pd.Series(df["subject"]).drop_duplicates())
    models = list(pd.Series(df["model"]).drop_duplicates())

    x = np.arange(len(subjects))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        sub_df = df[df["model"] == model].set_index("subject").reindex(subjects)
        heights = sub_df[metric_col].values.astype(float)

        if error_col is not None and error_col in sub_df.columns:
            yerr = sub_df[error_col].values.astype(float)
        else:
            yerr = None

        ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            heights,
            width=width,
            label=model,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_ranking_bars(
    df: pd.DataFrame,
    metric_col: str = "mean_subject_auc",
    title: str = "Model ranking",
    ylabel: str = "Mean subject ROC-AUC",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Plot one bar per model using a model-level summary table.
    """
    required = {"model", metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ranked = df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(ranked["model"], ranked[metric_col].astype(float).values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_learning_curve(
    history_df: pd.DataFrame,
    title: str = "Learning curve",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Plot train/validation loss and validation AUC from one fold history.

    Expected columns:
    - epoch
    - train_loss
    - val_loss
    - val_auc
    """
    required = {"epoch", "train_loss", "val_loss", "val_auc"}
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} — losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_auc"], label="Val AUC")
    axes[1].set_title(f"{title} — validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_subject_seed_stability(
    df_subject_seed_summary: pd.DataFrame,
    metric_col: str = "auc_seed_mean",
    error_col: str = "auc_seed_std",
    title: str = "Seed stability by subject",
    ylabel: str = "ROC-AUC across seeds",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot subject-wise seed stability with error bars.
    """
    required = {"subject", "model", metric_col, error_col}
    missing = required - set(df_subject_seed_summary.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subjects = list(pd.Series(df_subject_seed_summary["subject"]).drop_duplicates())
    models = list(pd.Series(df_subject_seed_summary["model"]).drop_duplicates())

    x = np.arange(len(subjects))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        sub_df = (
            df_subject_seed_summary[df_subject_seed_summary["model"] == model]
            .set_index("subject")
            .reindex(subjects)
        )

        heights = sub_df[metric_col].astype(float).values
        yerr = sub_df[error_col].astype(float).values

        ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            heights,
            width=width,
            label=model,
            yerr=yerr,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_main_vs_control(
    df_compare: pd.DataFrame,
    *,
    main_col: str = "main_value",
    control_col: str = "control_value",
    title: str = "Main benchmark vs control",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot main vs control values side-by-side at the subject/model level.
    """

    required = {"subject", "model", main_col, control_col}
    missing = required - set(df_compare.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df_compare.empty:
        raise ValueError("df_compare is empty. Check the merge inputs before plotting.")

    subjects = list(pd.Series(df_compare["subject"]).drop_duplicates())
    models = list(pd.Series(df_compare["model"]).drop_duplicates())

    if len(models) == 0:
        raise ValueError("No models found in df_compare after merging.")

    n_rows = len(models)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(figsize[0], max(figsize[1], 3 * n_rows)), squeeze=False
    )

    for ax, model in zip(axes.ravel(), models):
        sub_df = df_compare[df_compare["model"] == model].set_index("subject").reindex(subjects)

        x = np.arange(len(subjects))
        width = 0.35

        ax.bar(x - width / 2, sub_df[main_col].astype(float).values, width=width, label="Main")
        ax.bar(
            x + width / 2, sub_df[control_col].astype(float).values, width=width, label="Control"
        )

        ax.set_title(f"{title} — {model}")
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# Convenience export helpers
# ============================================================


def save_table_csv(df: pd.DataFrame, path: str | Path, decimals: int = 4) -> None:
    """
    Save a rounded dataframe to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numeric_columns(df, decimals=decimals).to_csv(path, index=False)


def save_multiple_figures(
    figs: Sequence[tuple[str, plt.Figure]],
    out_dir: str | Path,
    dpi: int = 150,
) -> None:
    """
    Save multiple figures into one directory.

    Parameters
    ----------
    figs : sequence of (filename, figure)
    out_dir : path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, fig in figs:
        save_figure(fig, out_dir / filename, dpi=dpi)
