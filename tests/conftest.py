from __future__ import annotations

import numpy as np
import pytest

from oddball_eeg.loader import LoadedDataset, make_balanced_chunk_groups_from_y


@pytest.fixture(autouse=True)
def _isolated_results(tmp_path, monkeypatch):
    """Never write into the repository's results/ or logs/ during tests."""
    monkeypatch.setenv("ODDBALL_EEG_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ODDBALL_EEG_LOGS_DIR", str(tmp_path / "logs"))


def make_synthetic_subject(
    rng: np.random.Generator,
    *,
    n_trials: int = 240,
    n_channels: int = 8,
    sfreq: float = 128.0,
    tmin: float = -0.2,
    tmax: float = 0.8,
    target_rate: float = 0.2,
    effect: float = 3.0,
):
    """Oddball-like epochs: targets carry a P300-shaped deflection on all channels."""
    times = np.arange(int(round((tmax - tmin) * sfreq)) + 1) / sfreq + tmin
    y = (rng.random(n_trials) < target_rate).astype(np.int64)
    y[:2] = [0, 1]  # guarantee both classes
    X = rng.normal(size=(n_trials, n_channels, times.size)).astype(np.float32)
    p300 = np.exp(-0.5 * ((times - 0.35) / 0.06) ** 2).astype(np.float32)
    X[y == 1] += effect * p300[None, None, :]
    return X, y, times


@pytest.fixture
def synthetic_dataset() -> LoadedDataset:
    rng = np.random.default_rng(0)
    subjects = ["001", "002", "003"]
    X_by_sub, y_by_sub, g_by_sub, s_by_sub = {}, {}, {}, {}
    times = None
    for sub in subjects:
        X, y, times = make_synthetic_subject(rng)
        X_by_sub[sub] = X
        y_by_sub[sub] = y
        g_by_sub[sub] = make_balanced_chunk_groups_from_y(y, chunk_size=20)
        s_by_sub[sub] = np.arange(len(y), dtype=np.int64) * 200
    return LoadedDataset(
        subjects=subjects,
        X_by_sub=X_by_sub,
        y_by_sub=y_by_sub,
        groups_by_sub=g_by_sub,
        samples_by_sub=s_by_sub,
        times=times,
        ch_names=[f"EEG{i:02d}" for i in range(8)],
        sfreq=128.0,
    )
