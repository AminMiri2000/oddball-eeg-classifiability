"""Aggregation of fold-level results into subject- and model-level tables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def summarize_fold_results(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("subject", "model"),
) -> pd.DataFrame:
    """
    Summarize fold-level results into mean/std tables.

    Expected columns include:
    - auc
    - pr_auc
    - bal_acc
    - acc
    - f1
    """
    required = {"auc", "pr_auc", "bal_acc", "acc", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")

    out = (
        df.groupby(list(group_cols), as_index=False)
        .agg(
            n_rows=("fold", "count"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            bal_acc_mean=("bal_acc", "mean"),
            bal_acc_std=("bal_acc", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )
    return out


def summarize_subject_results_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate subject-level summaries across seeds.

    Expected input:
    - one row per (subject, model, seed), typically after summarizing folds first
    """
    required = {"subject", "model", "seed", "auc_mean", "pr_auc_mean", "bal_acc_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for subject-across-seed summary: {sorted(missing)}"
        )

    out = (
        df.groupby(["subject", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            auc_seed_mean=("auc_mean", "mean"),
            auc_seed_std=("auc_mean", "std"),
            pr_auc_seed_mean=("pr_auc_mean", "mean"),
            pr_auc_seed_std=("pr_auc_mean", "std"),
            bal_acc_seed_mean=("bal_acc_mean", "mean"),
            bal_acc_seed_std=("bal_acc_mean", "std"),
            acc_seed_mean=("acc_mean", "mean"),
            acc_seed_std=("acc_mean", "std"),
            f1_seed_mean=("f1_mean", "mean"),
            f1_seed_std=("f1_mean", "std"),
        )
        .sort_values(["subject", "model"])
        .reset_index(drop=True)
    )
    return out


def summarize_model_results_across_subjects(df_subject_seed_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate subject-level seed summaries into model-level comparison tables.

    This is useful for ranking final candidate models.
    """
    required = {"subject", "model", "auc_seed_mean", "pr_auc_seed_mean", "bal_acc_seed_mean"}
    missing = required - set(df_subject_seed_summary.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for model summary across subjects: {sorted(missing)}"
        )

    out = (
        df_subject_seed_summary.groupby("model", as_index=False)
        .agg(
            n_subjects=("subject", "count"),
            mean_subject_auc=("auc_seed_mean", "mean"),
            std_subject_auc=("auc_seed_mean", "std"),
            median_subject_auc=("auc_seed_mean", "median"),
            worst_subject_auc=("auc_seed_mean", "min"),
            best_subject_auc=("auc_seed_mean", "max"),
            mean_subject_pr_auc=("pr_auc_seed_mean", "mean"),
            mean_subject_bal_acc=("bal_acc_seed_mean", "mean"),
            mean_subject_acc=("acc_seed_mean", "mean"),
            mean_subject_f1=("f1_seed_mean", "mean"),
        )
        .sort_values(
            by=["mean_subject_auc", "median_subject_auc", "worst_subject_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return out


def attach_experiment_metadata(
    df: pd.DataFrame,
    *,
    experiment_name: str,
    eval_mode: str,
    control_mode: str,
) -> pd.DataFrame:
    """
    Attach lightweight experiment metadata columns to a result table.
    """
    out = df.copy()
    out["experiment_name"] = experiment_name
    out["eval_mode"] = eval_mode
    out["control_mode"] = control_mode
    return out
