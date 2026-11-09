"""ERP summaries and plots (target-minus-standard difference waves, P300 topography)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from oddball_eeg.loader import LoadedDataset


def pick_roi_indices(ch_names: Sequence[str], roi_names: Sequence[str]) -> list[int]:
    name_to_idx = {str(ch).upper(): i for i, ch in enumerate(ch_names)}

    idx = []
    for roi in roi_names:
        key = str(roi).upper()
        if key in name_to_idx:
            idx.append(name_to_idx[key])

    if idx:
        return idx

    for fallback in ("PZ", "CPZ", "CZ"):
        if fallback in name_to_idx:
            return [name_to_idx[fallback]]

    return [0]


def compute_subject_erp_summary(
    data: LoadedDataset,
    *,
    roi_names: Sequence[str] = ("Pz", "CPz", "Cz", "CP1", "CP2"),
    p300_window: tuple[float, float] = (0.25, 0.50),
) -> pd.DataFrame:
    rows = []
    times = np.asarray(data.times)

    p3_mask = (times >= p300_window[0]) & (times <= p300_window[1])

    if p3_mask.sum() == 0:
        raise ValueError("P300 window has no samples.")

    roi_idx = pick_roi_indices(data.ch_names, roi_names)

    for sub in data.subjects:
        X = data.X_by_sub[sub]
        y = data.y_by_sub[sub]

        erp_std = X[y == 0].mean(axis=0)
        erp_tgt = X[y == 1].mean(axis=0)
        diff = erp_tgt - erp_std

        roi_diff = diff[roi_idx].mean(axis=0)
        p3_values = roi_diff[p3_mask]

        peak_local_idx = int(np.argmax(np.abs(p3_values)))
        p3_times = times[p3_mask]

        rows.append(
            {
                "subject": sub,
                "n_standard": int((y == 0).sum()),
                "n_target": int((y == 1).sum()),
                "roi_channels": ",".join([data.ch_names[i] for i in roi_idx]),
                "p300_mean_amp": float(p3_values.mean()),
                "p300_peak_amp_abs": float(p3_values[peak_local_idx]),
                "p300_peak_latency_s": float(p3_times[peak_local_idx]),
            }
        )

    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def grand_average_erps(data: LoadedDataset):
    X_std = []
    X_tgt = []

    for sub in data.subjects:
        X = data.X_by_sub[sub]
        y = data.y_by_sub[sub]
        X_std.append(X[y == 0].mean(axis=0))
        X_tgt.append(X[y == 1].mean(axis=0))

    erp_std = np.stack(X_std, axis=0).mean(axis=0)
    erp_tgt = np.stack(X_tgt, axis=0).mean(axis=0)
    diff = erp_tgt - erp_std

    return erp_std, erp_tgt, diff


def make_mne_info(ch_names: Sequence[str], sfreq: float) -> mne.Info:
    info = mne.create_info(list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    return info


def plot_grand_average_roi_erp(
    data: LoadedDataset,
    *,
    roi_names: Sequence[str] = ("Pz", "CPz", "Cz", "CP1", "CP2"),
    p300_window: tuple[float, float] = (0.25, 0.50),
):
    erp_std, erp_tgt, diff = grand_average_erps(data)
    roi_idx = pick_roi_indices(data.ch_names, roi_names)

    std_roi = erp_std[roi_idx].mean(axis=0)
    tgt_roi = erp_tgt[roi_idx].mean(axis=0)
    diff_roi = diff[roi_idx].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(data.times, std_roi, label="Standard")
    axes[0].plot(data.times, tgt_roi, label="Target")
    axes[0].axvline(0.0, linestyle="--", linewidth=1)
    axes[0].axvspan(p300_window[0], p300_window[1], alpha=0.15)
    axes[0].set_title("Grand-average ERP")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data.times, diff_roi, label="Target - Standard")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].axvline(0.0, linestyle="--", linewidth=1)
    axes[1].axvspan(p300_window[0], p300_window[1], alpha=0.15)
    axes[1].set_title("Difference waveform")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Difference amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_p300_topomap(
    data: LoadedDataset,
    *,
    p300_window: tuple[float, float] = (0.25, 0.50),
):
    _, _, diff = grand_average_erps(data)

    times = np.asarray(data.times)
    p3_mask = (times >= p300_window[0]) & (times <= p300_window[1])

    values = diff[:, p3_mask].mean(axis=1)

    info = make_mne_info(data.ch_names, data.sfreq)

    fig, ax = plt.subplots(figsize=(5, 4))
    mne.viz.plot_topomap(values, info, axes=ax, show=False, contours=6)
    ax.set_title("Target-minus-standard topography\nP300 window")
    fig.tight_layout()
    return fig
