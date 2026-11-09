"""Feature construction for the classical (LDA / logistic regression) models."""

from __future__ import annotations

import numpy as np


def flatten_epochs(X: np.ndarray) -> np.ndarray:
    """
    Flatten EEG epochs from (N, C, T) to (N, C*T) for classical models.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, T), got {X.shape}")

    return X.reshape(X.shape[0], -1)


def time_bin_epochs(
    X: np.ndarray,
    times: np.ndarray | None = None,
    bin_ms: int = 50,
    sfreq: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert epochs from (N, C, T) to time-binned features (N, C, B).

    This is useful for LDA/logistic regression because it reduces feature
    dimensionality while preserving the broad ERP time course.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, T), got {X.shape}")
    if bin_ms <= 0:
        raise ValueError(f"bin_ms must be positive, got {bin_ms}")

    if sfreq is None:
        if times is None:
            raise ValueError("Need either sfreq or times to compute bin size.")
        times = np.asarray(times)
        if len(times) < 2:
            raise ValueError("Need at least two time samples to infer sfreq.")
        sfreq = 1.0 / float(np.median(np.diff(times)))

    bin_samples = max(1, int(round((bin_ms / 1000.0) * float(sfreq))))
    n_bins = X.shape[2] // bin_samples

    if n_bins < 1:
        raise ValueError("Time binning produced zero bins.")

    keep = n_bins * bin_samples
    X_trim = X[:, :, :keep]

    Xb = X_trim.reshape(X.shape[0], X.shape[1], n_bins, bin_samples).mean(axis=-1)

    if times is None:
        tb = None
    else:
        t_trim = np.asarray(times)[:keep]
        tb = t_trim.reshape(n_bins, bin_samples).mean(axis=-1)

    return Xb.astype(np.float32), tb


def make_classical_features(
    X: np.ndarray,
    *,
    feature_mode: str = "raw_flat",
    times: np.ndarray | None = None,
    sfreq: float | None = None,
    bin_ms: int = 50,
) -> np.ndarray:
    """
    Build features for classical models.
    """
    feature_mode = str(feature_mode).lower()

    if feature_mode == "raw_flat":
        return flatten_epochs(X)

    if feature_mode == "time_binned":
        Xb, _ = time_bin_epochs(
            X,
            times=times,
            sfreq=sfreq,
            bin_ms=bin_ms,
        )
        return flatten_epochs(Xb)

    raise ValueError(f"Unknown feature_mode={feature_mode!r}")
