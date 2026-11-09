from __future__ import annotations

import mne
import numpy as np
import pytest

from oddball_eeg.loader import build_chronological_from_std_tgt, make_balanced_chunk_groups_from_y
from oddball_eeg.splits import build_purged_chunk_splits, make_group_train_val_split


def test_chunk_groups_are_contiguous_and_contain_targets():
    y = np.zeros(100, dtype=int)
    y[[5, 30, 31, 70, 95]] = 1
    groups = make_balanced_chunk_groups_from_y(y, chunk_size=10, min_targets_per_group=1)
    assert np.all(np.diff(groups) >= 0)  # contiguous, increasing
    for g in np.unique(groups)[:-1]:  # the final chunk may run out of targets
        assert y[groups == g].sum() >= 1


def test_purged_splits_do_not_leak_neighbouring_chunks():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.25).astype(int)
    groups = make_balanced_chunk_groups_from_y(y, chunk_size=20)
    purge = 1
    splits = build_purged_chunk_splits(y, groups, n_splits=5, purge_neighbors=purge)
    assert len(splits) == 5
    covered = np.zeros(len(y), dtype=bool)
    for tr, te in splits:
        assert np.intersect1d(tr, te).size == 0
        test_groups = np.unique(groups[te])
        train_groups = np.unique(groups[tr])
        lo, hi = test_groups.min() - purge, test_groups.max() + purge
        assert not np.any((train_groups >= lo) & (train_groups <= hi))
        covered[te] = True
    assert covered.all()


def test_group_train_val_split_keeps_groups_whole():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.3).astype(int)
    groups = np.repeat(np.arange(30), 10)
    tr, va = make_group_train_val_split(y, groups, val_size=0.2, seed=0)
    assert set(groups[tr]).isdisjoint(groups[va])
    assert len(set(groups[va])) == 6
    # Deterministic for a fixed seed.
    tr2, va2 = make_group_train_val_split(y, groups, val_size=0.2, seed=0)
    np.testing.assert_array_equal(va, va2)


def test_too_few_groups_raises():
    with pytest.raises(ValueError):
        build_purged_chunk_splits(np.array([0, 1, 0, 1]), np.zeros(4), n_splits=2)


def _epochs(samples, n_ch=2, n_times=5, value=0.0):
    info = mne.create_info([f"C{i}" for i in range(n_ch)], 100.0, "eeg")
    data = np.full((len(samples), n_ch, n_times), value)
    events = np.column_stack([samples, np.zeros(len(samples), int), np.ones(len(samples), int)])
    return mne.EpochsArray(data, info, events=events, verbose=False)


def test_chronological_reconstruction_interleaves_by_sample():
    std = _epochs([100, 300, 500], value=0.0)
    tgt = _epochs([200, 400], value=1.0)
    X, y, s = build_chronological_from_std_tgt(std, tgt)
    np.testing.assert_array_equal(s, [100, 200, 300, 400, 500])
    np.testing.assert_array_equal(y, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(X[:, 0, 0], y.astype(float))
