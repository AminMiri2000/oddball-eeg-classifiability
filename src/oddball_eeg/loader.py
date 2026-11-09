"""Subject loading: chronological trial reconstruction and chunk grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence  # noqa: UP035

import mne
import numpy as np
import pandas as pd

from oddball_eeg.bids import (
    EpochingConfig,
    find_bids_root,
    load_subject_epochs_from_bids,
    pick_task,
)

# ============================================================
# Data containers
# ============================================================


@dataclass
class SubjectData:
    """
    Container for one subject after chronological reconstruction.
    """

    subject: str
    X: np.ndarray  # (N, C, T)
    y: np.ndarray  # (N,)
    sample_idx: np.ndarray  # (N,) chronological sample indices
    groups: np.ndarray  # (N,) contiguous chunk groups


@dataclass
class LoadedDataset:
    """
    Container for a full loaded benchmark dataset.
    """

    subjects: list[str]
    X_by_sub: Dict[str, np.ndarray]
    y_by_sub: Dict[str, np.ndarray]
    groups_by_sub: Dict[str, np.ndarray]
    samples_by_sub: Dict[str, np.ndarray]
    times: np.ndarray
    ch_names: list[str]
    sfreq: float

    def subject_table(self) -> pd.DataFrame:
        """
        Build a simple subject-level QC table.
        """
        rows = []
        for sub in self.subjects:
            X = self.X_by_sub[sub]
            y = self.y_by_sub[sub]
            g = self.groups_by_sub[sub]

            rows.append(
                {
                    "subject": sub,
                    "n_trials": int(X.shape[0]),
                    "n_channels": int(X.shape[1]),
                    "n_times": int(X.shape[2]),
                    "n_standard": int((y == 0).sum()),
                    "n_target": int((y == 1).sum()),
                    "target_rate": float((y == 1).mean()),
                    "n_groups": int(pd.Series(g).nunique()),
                }
            )

        return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


# ============================================================
# Chronology + grouping helpers
# ============================================================


def make_balanced_chunk_groups_from_y(
    y: np.ndarray,
    chunk_size: int = 80,
    min_targets_per_group: int = 1,
) -> np.ndarray:
    """
    Chronological chunking with a simple target-balance safeguard.

    The chunk boundaries remain chronological, but a chunk is allowed to grow
    past chunk_size until it contains at least min_targets_per_group target
    trials. This reduces invalid or weak folds when targets are sparse.

    Parameters
    ----------
    y : np.ndarray
        Binary labels in chronological order.

    chunk_size : int
        Preferred chunk size.

    min_targets_per_group : int
        Minimum target trials per group when possible.

    Returns
    -------
    groups : np.ndarray
        Contiguous group IDs.
    """
    y = np.asarray(y).astype(int)

    if y.ndim != 1:
        raise ValueError(f"Expected y with shape (N,), got {y.shape}")
    if len(y) <= 0:
        raise ValueError("Need at least one trial.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if min_targets_per_group < 0:
        raise ValueError(f"min_targets_per_group must be >= 0, got {min_targets_per_group}")

    groups = np.empty(len(y), dtype=np.int64)

    group_id = 0
    start = 0

    while start < len(y):
        end = min(start + int(chunk_size), len(y))

        # Grow the chunk until it has enough targets, if possible.
        while end < len(y) and int(y[start:end].sum()) < int(min_targets_per_group):
            end += 1

        groups[start:end] = group_id
        group_id += 1
        start = end

    return groups


def build_chronological_from_std_tgt(
    epochs_std: mne.Epochs,
    epochs_tgt: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild a chronological trial stream from standard and target epochs.

    This preserves the original chronological order using event sample indices.

    Returns
    -------
    X_ord : np.ndarray
        Epoch tensor in chronological order, shape (N, C, T)

    y_ord : np.ndarray
        Labels in chronological order, shape (N,)
        standard -> 0
        target   -> 1

    sample_ord : np.ndarray
        Event sample indices in chronological order, shape (N,)
    """
    Xs = epochs_std.get_data().astype(np.float32)
    Xt = epochs_tgt.get_data().astype(np.float32)

    ys = np.zeros(len(Xs), dtype=np.int64)
    yt = np.ones(len(Xt), dtype=np.int64)

    ss = epochs_std.events[:, 0].astype(np.int64)
    st = epochs_tgt.events[:, 0].astype(np.int64)

    X_all = np.concatenate([Xs, Xt], axis=0)
    y_all = np.concatenate([ys, yt], axis=0)
    s_all = np.concatenate([ss, st], axis=0)

    # Track original source and row order so ties are deterministic
    source_order = np.concatenate(
        [
            np.zeros(len(Xs), dtype=np.int64),  # standard source
            np.ones(len(Xt), dtype=np.int64),  # target source
        ]
    )
    row_order = np.arange(len(s_all), dtype=np.int64)

    # Sort primarily by sample index, then source, then original row index
    order = np.lexsort((row_order, source_order, s_all))

    X_ord = X_all[order]
    y_ord = y_all[order]
    sample_ord = s_all[order]

    return X_ord, y_ord, sample_ord


# ============================================================
# Subject loading
# ============================================================


def load_one_subject_grouped(
    *,
    dataset: str,
    task_name: str,
    subject: str,
    epoching_cfg: EpochingConfig | None = None,
    chunk_size: int = 80,
    apply_baseline: bool = True,
    verbose: bool = True,
) -> tuple[SubjectData, np.ndarray, list[str], float]:
    """
    Load one subject, reconstruct chronological epochs, and assign chunk groups.

    Returns
    -------
    subject_data : SubjectData
    times : np.ndarray
    ch_names : list[str]
    sfreq : float
    """
    if epoching_cfg is None:
        epoching_cfg = EpochingConfig()

    bids_root = find_bids_root(dataset)
    task = pick_task(bids_root, task_name)

    if verbose:
        print(f"Loading sub-{subject} ...")

    epochs_std, epochs_tgt, _ = load_subject_epochs_from_bids(
        bids_root=bids_root,
        subject=subject,
        task=task,
        cfg=epoching_cfg,
        apply_baseline=apply_baseline,
    )

    if epochs_std is None or epochs_tgt is None:
        raise RuntimeError(f"Could not load epochs for subject {subject}")

    X, y, sample_idx = build_chronological_from_std_tgt(epochs_std, epochs_tgt)
    groups = make_balanced_chunk_groups_from_y(
        y=y,
        chunk_size=chunk_size,
        min_targets_per_group=1,
    )

    times = epochs_std.times.copy()
    ch_names = list(epochs_std.ch_names)
    sfreq = float(epochs_std.info["sfreq"])

    subject_data = SubjectData(
        subject=subject,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        sample_idx=sample_idx.astype(np.int64),
        groups=groups.astype(np.int64),
    )

    if verbose:
        print(
            f"sub-{subject}: "
            f"X={subject_data.X.shape}, "
            f"target_rate={subject_data.y.mean():.4f}, "
            f"n_groups={pd.Series(subject_data.groups).nunique()}"
        )

    return subject_data, times, ch_names, sfreq


def load_subjects_grouped(
    *,
    dataset: str,
    task_name: str,
    subjects: Sequence[str],
    epoching_cfg: EpochingConfig | None = None,
    chunk_size: int = 80,
    apply_baseline: bool = True,
    verbose: bool = True,
) -> LoadedDataset:
    """
    Load multiple subjects and return a single structured dataset object.

    This function:
    - loads each subject independently
    - rebuilds chronological trial order
    - assigns contiguous chunk groups
    - verifies consistency of time axis / channel order across subjects
    """
    if epoching_cfg is None:
        epoching_cfg = EpochingConfig()

    X_by_sub: Dict[str, np.ndarray] = {}
    y_by_sub: Dict[str, np.ndarray] = {}
    groups_by_sub: Dict[str, np.ndarray] = {}
    samples_by_sub: Dict[str, np.ndarray] = {}

    times_ref = None
    ch_names_ref = None
    sfreq_ref = None
    loaded_subjects: list[str] = []

    for subject in subjects:
        subj_data, times, ch_names, sfreq = load_one_subject_grouped(
            dataset=dataset,
            task_name=task_name,
            subject=subject,
            epoching_cfg=epoching_cfg,
            chunk_size=chunk_size,
            apply_baseline=apply_baseline,
            verbose=verbose,
        )

        if times_ref is None:
            times_ref = times.copy()
            ch_names_ref = list(ch_names)
            sfreq_ref = float(sfreq)
        else:
            if not np.allclose(times_ref, times):
                raise ValueError(f"Time axis mismatch for subject {subject}")
            if ch_names_ref != list(ch_names):
                raise ValueError(f"Channel order mismatch for subject {subject}")
            if abs(float(sfreq_ref) - float(sfreq)) > 1e-9:
                raise ValueError(f"Sampling frequency mismatch for subject {subject}")

        X_by_sub[subject] = subj_data.X
        y_by_sub[subject] = subj_data.y
        groups_by_sub[subject] = subj_data.groups
        samples_by_sub[subject] = subj_data.sample_idx
        loaded_subjects.append(subject)

    if times_ref is None or ch_names_ref is None or sfreq_ref is None:
        raise RuntimeError("No subjects were loaded successfully.")

    dataset_obj = LoadedDataset(
        subjects=loaded_subjects,
        X_by_sub=X_by_sub,
        y_by_sub=y_by_sub,
        groups_by_sub=groups_by_sub,
        samples_by_sub=samples_by_sub,
        times=np.asarray(times_ref),
        ch_names=list(ch_names_ref),
        sfreq=float(sfreq_ref),
    )

    return dataset_obj


# ============================================================
# Lightweight QC helpers
# ============================================================


def print_loaded_dataset_summary(data: LoadedDataset) -> None:
    """
    Print a readable summary of the loaded dataset.
    """
    for sub in data.subjects:
        X = data.X_by_sub[sub]
        y = data.y_by_sub[sub]
        groups = data.groups_by_sub[sub]

        print("=" * 70)
        print(f"Subject {sub}")
        print("X shape      :", X.shape)
        print("y shape      :", y.shape)
        print("n_trials     :", int(X.shape[0]))
        print("n_channels   :", int(X.shape[1]))
        print("n_times      :", int(X.shape[2]))
        print("n_standard   :", int((y == 0).sum()))
        print("n_target     :", int((y == 1).sum()))
        print("target rate  :", float((y == 1).mean()))
        print("n_groups     :", int(pd.Series(groups).nunique()))
        print("group counts :")
        print(pd.Series(groups).value_counts().sort_index().head())

    print("=" * 70)
    print("Times shape         :", data.times.shape)
    print("Time range          :", float(data.times[0]), "to", float(data.times[-1]))
    print("Sampling frequency  :", data.sfreq)
    print("Number of channels  :", len(data.ch_names))
    print("First 10 channels   :", data.ch_names[:10])


def make_subject_summary_table(data: LoadedDataset) -> pd.DataFrame:
    """
    Return a subject-level summary table for display or export.
    """
    return data.subject_table()
