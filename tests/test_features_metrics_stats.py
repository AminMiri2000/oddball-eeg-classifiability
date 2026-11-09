from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oddball_eeg.features import flatten_epochs, make_classical_features, time_bin_epochs
from oddball_eeg.metrics import auc_safe, classification_metric_dict
from oddball_eeg.stats import benjamini_hochberg, safe_corr, strength_word, zscore_within_group


def test_time_binning_averages_consecutive_samples():
    sfreq = 100.0
    X = np.arange(2 * 3 * 20, dtype=np.float32).reshape(2, 3, 20)
    times = np.arange(20) / sfreq
    Xb, tb = time_bin_epochs(X, times=times, bin_ms=50)  # 5 samples per bin
    assert Xb.shape == (2, 3, 4)
    np.testing.assert_allclose(Xb[0, 0], X[0, 0].reshape(4, 5).mean(axis=1))
    np.testing.assert_allclose(tb, times.reshape(4, 5).mean(axis=1))


def test_classical_features_shapes():
    X = np.zeros((4, 8, 103), dtype=np.float32)
    assert flatten_epochs(X).shape == (4, 8 * 103)
    # 0.8 s at 128 Hz with 50-ms bins -> 6-sample bins -> 17 bins.
    F = make_classical_features(X, feature_mode="time_binned", sfreq=128.0, bin_ms=50)
    assert F.shape == (4, 8 * 17)
    with pytest.raises(ValueError):
        make_classical_features(X, feature_mode="unknown")


def test_auc_safe_handles_single_class():
    assert np.isnan(auc_safe(np.zeros(5), np.arange(5)))
    assert auc_safe([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_metric_dict_contains_confusion_counts():
    m = classification_metric_dict([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.1], [0, 1, 0, 0])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 1, 1)
    assert m["bal_acc"] == pytest.approx(0.75)


def test_benjamini_hochberg_matches_statsmodels():
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(0)
    p = rng.random(40) ** 2
    expected = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(benjamini_hochberg(p), expected)


def test_benjamini_hochberg_ignores_nan():
    q = benjamini_hochberg([0.01, np.nan, 0.04])
    assert np.isnan(q[1])
    np.testing.assert_allclose(q[[0, 2]], [0.02, 0.04])


def test_safe_corr_edge_cases():
    r, p, n = safe_corr([1, 2, np.nan, 4, 5], [2, 4, 1, 8, 10])
    assert n == 4 and r == pytest.approx(1.0)
    assert np.isnan(safe_corr([1, 1, 1, 1], [1, 2, 3, 4])[0])
    assert np.isnan(safe_corr([1, 2], [1, 2])[0])


def test_zscore_within_group_constant_is_nan():
    s = pd.Series([3.0, 3.0, 3.0])
    assert zscore_within_group(s).isna().all()
    z = zscore_within_group(pd.Series([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0) and z.std(ddof=0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "r,label", [(0.8, "strong"), (0.31, "moderate"), (0.05, "weak"), (np.nan, "unclear")]
)
def test_strength_word(r, label):
    assert strength_word(r) == label
