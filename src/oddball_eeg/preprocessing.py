"""Epoch-level transforms: cropping, label permutation, normalization, PyTorch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from oddball_eeg.loader import LoadedDataset

# ============================================================
# Small containers
# ============================================================


@dataclass
class NormalizationStats:
    """
    Train-only normalization statistics for EEG tensors shaped (N, C, T).
    """

    mean: np.ndarray  # shape (1, C, 1)
    std: np.ndarray  # shape (1, C, 1)


@dataclass
class SplitArrays:
    """
    Simple container for one train/val/test split.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


# ============================================================
# Cropping
# ============================================================


def crop_time_window(
    X: np.ndarray,
    times: np.ndarray,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop an epoch tensor X=(N, C, T) to a requested time window.

    Parameters
    ----------
    X : np.ndarray
        Epoch tensor of shape (N, C, T)

    times : np.ndarray
        Time vector of shape (T,)

    tmin, tmax : float
        Inclusive crop window

    Returns
    -------
    X_crop : np.ndarray
        Cropped tensor of shape (N, C, T_crop)

    times_crop : np.ndarray
        Cropped time vector of shape (T_crop,)
    """
    X = np.asarray(X)
    times = np.asarray(times)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, T), got {X.shape}")
    if times.ndim != 1:
        raise ValueError(f"Expected times with shape (T,), got {times.shape}")
    if X.shape[2] != len(times):
        raise ValueError(
            f"Time dimension mismatch: X.shape[2]={X.shape[2]} vs len(times)={len(times)}"
        )
    if tmin >= tmax:
        raise ValueError(f"Invalid crop window: tmin={tmin} must be < tmax={tmax}")

    mask = (times >= float(tmin)) & (times <= float(tmax))

    if mask.sum() < 8:
        raise ValueError(f"Crop window [{tmin}, {tmax}] keeps too few samples: {mask.sum()}")

    X_crop = X[:, :, mask].astype(np.float32, copy=False)
    times_crop = times[mask].copy()

    return X_crop, times_crop


def apply_crop_to_loaded_dataset(
    data: LoadedDataset,
    tmin: float,
    tmax: float,
) -> LoadedDataset:
    """
    Apply the same crop window to every subject in a LoadedDataset.

    Returns a new LoadedDataset with cropped X and cropped times.
    Labels/groups/sample indices remain unchanged.
    """
    X_by_sub_new: Dict[str, np.ndarray] = {}
    times_new = None

    for sub in data.subjects:
        X_crop, times_crop = crop_time_window(
            X=data.X_by_sub[sub],
            times=data.times,
            tmin=tmin,
            tmax=tmax,
        )
        X_by_sub_new[sub] = X_crop

        if times_new is None:
            times_new = times_crop
        else:
            if not np.allclose(times_new, times_crop):
                raise ValueError(f"Cropped time mismatch for subject {sub}")

    if times_new is None:
        raise RuntimeError("No subjects available for cropping.")

    return LoadedDataset(
        subjects=list(data.subjects),
        X_by_sub=X_by_sub_new,
        y_by_sub={k: v.copy() for k, v in data.y_by_sub.items()},
        groups_by_sub={k: v.copy() for k, v in data.groups_by_sub.items()},
        samples_by_sub={k: v.copy() for k, v in data.samples_by_sub.items()},
        times=np.asarray(times_new),
        ch_names=list(data.ch_names),
        sfreq=float(data.sfreq),
    )


# ============================================================
# Label permutation control
# ============================================================


def permute_labels_within_subjects(
    y_by_sub: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Shuffle labels independently within each subject.

    This preserves class counts per subject while destroying the X-y mapping.
    Useful for a label-permutation control.
    """
    rng = np.random.default_rng(seed)
    y_perm: Dict[str, np.ndarray] = {}

    for sub, y in y_by_sub.items():
        y = np.asarray(y).astype(np.int64)
        y_perm[sub] = rng.permutation(y.copy()).astype(np.int64)

    return y_perm


def apply_permuted_labels_to_loaded_dataset(
    data: LoadedDataset,
    seed: int,
) -> LoadedDataset:
    """
    Return a copy of LoadedDataset with labels permuted independently per subject.
    """
    y_perm = permute_labels_within_subjects(data.y_by_sub, seed=seed)

    return LoadedDataset(
        subjects=list(data.subjects),
        X_by_sub={k: v.copy() for k, v in data.X_by_sub.items()},
        y_by_sub=y_perm,
        groups_by_sub={k: v.copy() for k, v in data.groups_by_sub.items()},
        samples_by_sub={k: v.copy() for k, v in data.samples_by_sub.items()},
        times=data.times.copy(),
        ch_names=list(data.ch_names),
        sfreq=float(data.sfreq),
    )


# ============================================================
# Normalization
# ============================================================


def compute_channel_normalization_stats(X_train: np.ndarray) -> NormalizationStats:
    """
    Compute train-only per-channel normalization stats.

    Parameters
    ----------
    X_train : np.ndarray
        Shape (N, C, T)

    Returns
    -------
    NormalizationStats
        mean/std shaped (1, C, 1)
    """
    X_train = np.asarray(X_train, dtype=np.float32)

    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train with shape (N, C, T), got {X_train.shape}")

    mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)

    return NormalizationStats(mean=mean, std=std)


def make_identity_normalization_stats(n_channels: int) -> NormalizationStats:
    """
    Return identity normalization stats (no-op normalization).
    """
    if n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {n_channels}")

    mean = np.zeros((1, n_channels, 1), dtype=np.float32)
    std = np.ones((1, n_channels, 1), dtype=np.float32)
    return NormalizationStats(mean=mean, std=std)


def normalize_epochs(
    X: np.ndarray,
    stats: NormalizationStats,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply per-channel normalization to an epoch tensor.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, T), got {X.shape}")

    Xn = (X - stats.mean) / (stats.std + eps)
    return Xn.astype(np.float32, copy=False)


# ============================================================
# Dataset + Dataloader helpers
# ============================================================


class EEGEpochDataset(Dataset):
    """
    PyTorch dataset for EEG epoch tensors.

    Expects:
        X -> (N, C, T)
        y -> (N,)
    Returns:
        x -> torch.FloatTensor, shape (C, T)
        y -> torch.LongTensor, shape ()
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm_stats: NormalizationStats,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.norm_stats = norm_stats

        if self.X.ndim != 3:
            raise ValueError(f"Expected X with shape (N, C, T), got {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"Expected y with shape (N,), got {self.y.shape}")
        if len(self.X) != len(self.y):
            raise ValueError(f"Length mismatch: len(X)={len(self.X)} vs len(y)={len(self.y)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (C, T)
        y = self.y[idx]

        x = (x - self.norm_stats.mean.squeeze(0)) / (self.norm_stats.std.squeeze(0) + 1e-8)

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y, dtype=torch.long),
        )


def make_class_counts(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """
    Count class frequencies.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=n_classes)
    return counts.astype(np.int64)


def make_class_weight_tensor(
    y_train: np.ndarray,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Build a class-weight tensor for binary CrossEntropyLoss / CrossEntropyLoss usage.

    Weighting scheme:
        w0 = 1.0
        w1 = n0 / n1
    """
    counts = make_class_counts(y_train, n_classes=2)
    n0 = max(int(counts[0]), 1)
    n1 = max(int(counts[1]), 1)

    w0 = 1.0
    w1 = float(n0 / n1)

    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def build_datasets_from_split(
    split: SplitArrays,
    normalize: bool = True,
) -> tuple[EEGEpochDataset, EEGEpochDataset, EEGEpochDataset, NormalizationStats]:
    """
    Build train/val/test datasets from split arrays.

    Normalization is always computed on TRAIN only.
    """
    if normalize:
        norm_stats = compute_channel_normalization_stats(split.X_train)
    else:
        norm_stats = make_identity_normalization_stats(split.X_train.shape[1])

    ds_train = EEGEpochDataset(split.X_train, split.y_train, norm_stats)
    ds_val = EEGEpochDataset(split.X_val, split.y_val, norm_stats)
    ds_test = EEGEpochDataset(split.X_test, split.y_test, norm_stats)

    return ds_train, ds_val, ds_test, norm_stats


def build_dataloaders(
    ds_train: Dataset,
    ds_val: Dataset,
    ds_test: Dataset,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build standard train/val/test dataloaders.

    Train loader uses shuffle=True.
    Validation/test loaders use shuffle=False.
    """
    pin = str(device).startswith("cuda")

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Convenience helpers for notebook usage
# ============================================================


def describe_crop_effect(
    original_times: np.ndarray,
    cropped_times: np.ndarray,
) -> dict:
    """
    Summarize how cropping changed the time axis.
    """
    original_times = np.asarray(original_times)
    cropped_times = np.asarray(cropped_times)

    return {
        "original_n_times": int(len(original_times)),
        "cropped_n_times": int(len(cropped_times)),
        "original_tmin": float(original_times[0]),
        "original_tmax": float(original_times[-1]),
        "cropped_tmin": float(cropped_times[0]),
        "cropped_tmax": float(cropped_times[-1]),
    }


def print_crop_effect(original_times: np.ndarray, cropped_times: np.ndarray) -> None:
    """
    Print a human-readable crop summary.
    """
    info = describe_crop_effect(original_times, cropped_times)
    print("=" * 70)
    print("Crop summary")
    print("original_n_times :", info["original_n_times"])
    print("cropped_n_times  :", info["cropped_n_times"])
    print("original_range   :", info["original_tmin"], "to", info["original_tmax"])
    print("cropped_range    :", info["cropped_tmin"], "to", info["cropped_tmax"])


def make_subject_label_table(data: LoadedDataset) -> np.ndarray:
    """
    Return a simple table-like array for quick notebook inspection of label counts.
    """
    rows = []
    for sub in data.subjects:
        y = data.y_by_sub[sub]
        rows.append(
            [
                sub,
                int((y == 0).sum()),
                int((y == 1).sum()),
                float((y == 1).mean()),
            ]
        )
    return np.array(rows, dtype=object)
