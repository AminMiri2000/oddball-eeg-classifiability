from __future__ import annotations

import numpy as np
import pytest
import torch

from oddball_eeg import config, io
from oddball_eeg.models import EEGNet, get_model
from oddball_eeg.preprocessing import (
    apply_crop_to_loaded_dataset,
    apply_permuted_labels_to_loaded_dataset,
)
from oddball_eeg.splits import build_folds_by_subject
from oddball_eeg.summaries import (
    summarize_fold_results,
    summarize_model_results_across_subjects,
    summarize_subject_results_across_seeds,
)
from oddball_eeg.training import evaluate_classical_benchmark, evaluate_eegnet_benchmark


@pytest.mark.parametrize("n_times", [103, 26, 8])
def test_eegnet_handles_full_and_short_windows(n_times):
    model = EEGNet(n_ch=8, n_times=n_times)
    out = model(torch.zeros(3, 1, 8, n_times))
    assert out.shape == (3, 2)


def test_get_model_requires_shape_for_eegnet():
    with pytest.raises(ValueError):
        get_model("eegnet")
    assert get_model("lda").named_steps["clf"].shrinkage == "auto"


def test_classical_benchmark_detects_synthetic_p300(synthetic_dataset):
    data = apply_crop_to_loaded_dataset(synthetic_dataset, 0.0, 0.8)
    folds = build_folds_by_subject(data, n_splits=5, purge_neighbors=1)
    df = evaluate_classical_benchmark(
        data, folds, seed=42, verbose=False, feature_mode="time_binned"
    )
    subj = summarize_fold_results(df.assign(seed=42), group_cols=("subject", "model", "seed"))
    across = summarize_subject_results_across_seeds(subj)
    models = summarize_model_results_across_subjects(across)
    assert set(models["model"]) == {"lda", "logreg"}
    assert (models["mean_subject_auc"] > 0.9).all()


def test_permuted_labels_destroy_signal(synthetic_dataset):
    data = apply_permuted_labels_to_loaded_dataset(
        apply_crop_to_loaded_dataset(synthetic_dataset, 0.0, 0.8), seed=0
    )
    folds = build_folds_by_subject(data, n_splits=5, purge_neighbors=1)
    df = evaluate_classical_benchmark(
        data, folds, model_names=("lda",), verbose=False, feature_mode="time_binned"
    )
    assert df["auc"].mean() < 0.7


def test_eegnet_benchmark_smoke(synthetic_dataset):
    data = apply_crop_to_loaded_dataset(synthetic_dataset, 0.0, 0.8)
    data.subjects = data.subjects[:1]
    folds = build_folds_by_subject(data, n_splits=3, purge_neighbors=1)
    df, histories = evaluate_eegnet_benchmark(data, folds, epochs=2, patience=1, verbose=False)
    assert len(df) == 3 and np.isfinite(df["auc"]).all()
    assert set(histories["001"]) == {1, 2, 3}


def test_save_json_handles_numpy_and_config(tmp_path):
    cfg = config.get_experiment("ds003061_within_subject_max30_wide_timebin")
    path = tmp_path / "meta.json"
    io.save_json({"cfg": cfg.to_dict(), "n": np.int64(3), "arr": np.arange(2)}, path)
    assert '"n": 3' in path.read_text()
