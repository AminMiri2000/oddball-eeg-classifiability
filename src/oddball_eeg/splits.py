"""Chronological, purged, grouped cross-validation splits.

Trials are grouped into contiguous chunks (see
``oddball_eeg.loader.make_balanced_chunk_groups_from_y``). Each outer fold tests on
a block of consecutive chunks and removes ``purge_neighbors`` chunks on either side
from the training set to limit temporal leakage."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from oddball_eeg.loader import LoadedDataset


def build_purged_chunk_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    purge_neighbors: int = 2,
    min_test_targets: int = 1,
    min_train_targets: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build chronological purged grouped folds with basic class-balance safeguards.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    if len(y) != len(groups):
        raise ValueError(f"Length mismatch: len(y)={len(y)} vs len(groups)={len(groups)}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if purge_neighbors < 0:
        raise ValueError(f"purge_neighbors must be >= 0, got {purge_neighbors}")

    uniq = pd.Series(groups).drop_duplicates().to_numpy()

    if len(uniq) < 2:
        raise ValueError("Need at least 2 groups to build grouped folds.")

    test_blocks = np.array_split(np.arange(len(uniq)), int(n_splits))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = []

    for fold_id, block in enumerate(test_blocks, start=1):
        if len(block) == 0:
            skipped.append((fold_id, "empty block"))
            continue

        test_groups = uniq[block]

        left = max(0, int(block[0]) - int(purge_neighbors))
        right = min(len(uniq), int(block[-1]) + int(purge_neighbors) + 1)

        excluded_train_groups = set(uniq[left:right])

        tr_idx = np.where(~np.isin(groups, list(excluded_train_groups)))[0]
        te_idx = np.where(np.isin(groups, list(test_groups)))[0]

        if len(tr_idx) == 0 or len(te_idx) == 0:
            skipped.append((fold_id, "empty train or test"))
            continue

        train_targets = int(y[tr_idx].sum())
        test_targets = int(y[te_idx].sum())

        if len(np.unique(y[tr_idx])) < 2 or train_targets < min_train_targets:
            skipped.append((fold_id, f"training fold has too few targets: {train_targets}"))
            continue

        if len(np.unique(y[te_idx])) < 2 or test_targets < min_test_targets:
            skipped.append((fold_id, f"test fold has too few targets: {test_targets}"))
            continue

        splits.append((tr_idx, te_idx))

    if len(splits) < 2:
        raise ValueError(
            f"Purged chunk split produced too few valid folds. Skipped folds: {skipped}"
        )

    return splits


def build_folds_by_subject(
    data: LoadedDataset,
    n_splits: int = 5,
    purge_neighbors: int = 2,
    min_test_targets: int = 1,
    min_train_targets: int = 1,
) -> Dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Build purged grouped folds for every subject in a LoadedDataset.
    """
    folds_by_sub: Dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for sub in data.subjects:
        folds_by_sub[sub] = build_purged_chunk_splits(
            y=data.y_by_sub[sub],
            groups=data.groups_by_sub[sub],
            n_splits=n_splits,
            purge_neighbors=purge_neighbors,
            min_test_targets=min_test_targets,
            min_train_targets=min_train_targets,
        )

    return folds_by_sub


def make_group_train_val_split(
    y_train: np.ndarray,
    groups_train: np.ndarray,
    val_size: float = 0.15,
    seed: int = 42,
    n_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a group-aware train/validation split inside an outer training fold.

    This chooses whole groups for validation and tries to keep train/val class
    proportions close to the overall training-fold proportion.

    Returns
    -------
    train_sub_idx : np.ndarray
        Indices relative to the provided training fold

    val_sub_idx : np.ndarray
        Indices relative to the provided training fold
    """
    y_train = np.asarray(y_train).astype(int)
    groups_train = np.asarray(groups_train)

    if len(y_train) != len(groups_train):
        raise ValueError(
            f"Length mismatch: len(y_train)={len(y_train)} vs len(groups_train)={len(groups_train)}"
        )
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0,1), got {val_size}")
    if n_tries <= 0:
        raise ValueError(f"n_tries must be positive, got {n_tries}")

    uniq_groups = pd.Series(groups_train).drop_duplicates().to_numpy()

    if len(uniq_groups) < 2:
        raise ValueError("Need at least 2 groups for group-aware train/val split.")

    n_val_groups = max(1, int(round(val_size * len(uniq_groups))))
    n_val_groups = min(n_val_groups, len(uniq_groups) - 1)

    rng = np.random.default_rng(seed)
    global_rate = float(y_train.mean())
    best = None

    for _ in range(n_tries):
        val_groups = rng.choice(uniq_groups, size=n_val_groups, replace=False)
        val_mask = np.isin(groups_train, val_groups)

        tr_idx = np.where(~val_mask)[0]
        va_idx = np.where(val_mask)[0]

        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        if len(np.unique(y_train[tr_idx])) < 2:
            continue
        if len(np.unique(y_train[va_idx])) < 2:
            continue

        score = abs(float(y_train[tr_idx].mean()) - global_rate) + abs(
            float(y_train[va_idx].mean()) - global_rate
        )

        if best is None or score < best[0]:
            best = (score, tr_idx, va_idx)

    if best is None:
        raise ValueError("Could not build a valid group-aware train/val split.")

    return best[1], best[2]
