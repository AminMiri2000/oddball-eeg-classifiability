"""Sliding-window (time-resolved) decoding with the classical models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from oddball_eeg.metrics import classification_metric_dict
from oddball_eeg.models import get_model


def evaluate_time_resolved_classical_on_folds(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    model_name: str = "lda",
    subject: str | None = None,
    seed: int | None = None,
    window_ms: int = 50,
    step_ms: int = 25,
    sfreq: float | None = None,
) -> pd.DataFrame:
    """
    Sliding-window time-resolved decoding for classical models.

    For each time window, train/evaluate the model using only samples inside
    that window, under the same outer folds.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    times = np.asarray(times)

    if sfreq is None:
        sfreq = 1.0 / float(np.median(np.diff(times)))

    win_samples = max(1, int(round(window_ms / 1000.0 * sfreq)))
    step_samples = max(1, int(round(step_ms / 1000.0 * sfreq)))

    rows = []

    starts = range(0, X.shape[2] - win_samples + 1, step_samples)

    for start in starts:
        end = start + win_samples
        t_center = float(times[start:end].mean())
        t_start = float(times[start])
        t_end = float(times[end - 1])

        Xw = X[:, :, start:end]
        F = Xw.reshape(Xw.shape[0], -1)

        for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
            clf = get_model(model_name)

            clf.fit(F[tr_idx], y[tr_idx])

            if hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(F[te_idx])[:, 1]
            else:
                scores = clf.decision_function(F[te_idx])

            y_pred = clf.predict(F[te_idx])

            row = {
                "subject": subject,
                "seed": seed,
                "fold": int(fold_id),
                "model": model_name,
                "time_center_s": t_center,
                "time_start_s": t_start,
                "time_end_s": t_end,
                "window_ms": int(window_ms),
                "step_ms": int(step_ms),
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "train_pos_rate": float(y[tr_idx].mean()),
                "test_pos_rate": float(y[te_idx].mean()),
            }
            row.update(classification_metric_dict(y[te_idx], scores, y_pred))
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_time_resolved(df: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "model", "time_center_s", "auc"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        df.groupby(["subject", "model", "time_center_s"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            bal_acc_mean=("bal_acc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
        )
        .sort_values(["subject", "model", "time_center_s"])
        .reset_index(drop=True)
    )


def peak_time_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (subject, model), sub_df in df_summary.groupby(["subject", "model"]):
        sub_df = sub_df.sort_values("auc_mean", ascending=False)
        best = sub_df.iloc[0]

        rows.append(
            {
                "subject": subject,
                "model": model,
                "peak_time_s": float(best["time_center_s"]),
                "peak_auc": float(best["auc_mean"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["subject", "model"]).reset_index(drop=True)
