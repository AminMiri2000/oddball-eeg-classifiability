"""Binary classification metrics with safe handling of degenerate folds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def auc_safe(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC with safe handling when only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def pr_auc_safe(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) with safe handling.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def confusion_items(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Return tn, fp, fn, tp for binary classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_metric_dict(
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Build a standard metric dictionary for binary classification.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    out = {
        "auc": auc_safe(y_true, scores),
        "pr_auc": pr_auc_safe(y_true, scores),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out.update(confusion_items(y_true, y_pred))
    return out
