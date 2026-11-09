from __future__ import annotations

import pytest

from oddball_eeg import config
from oddball_eeg.cli import check_journal_config


def test_journal_registry_check_passes(capsys):
    assert check_journal_config() == 0
    assert "103 subject-recordings" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key", config.JOURNAL_WITHIN_SUBJECT_EXPERIMENTS + config.JOURNAL_LOSO_EXPERIMENTS
)
def test_journal_experiments_use_final_representation(key):
    cfg = config.get_experiment(key)
    assert cfg.effective_crop_window() == (0.0, 0.8)
    assert cfg.feature.mode == "time_binned"
    assert cfg.feature.bin_ms == 50
    assert cfg.seeds.seeds == (42, 43, 44)
    assert cfg.signal.l_freq == 0.1 and cfg.signal.h_freq == 12.0


def test_journal_result_path_layout(tmp_path):
    cfg = config.get_experiment("ds006480_loso_max30_wide_timebin")
    rel = cfg.result_dir().relative_to(tmp_path / "results")
    assert rel.as_posix() == (
        "ds006480/oddball/events_pooled/bp_0p1_12Hz_epoch_m0p2_0p8s_rs128Hz/"
        "max30_wide_timebin/crop_0_0p8s/feat_timebin_50ms/loso/standard"
    )
    assert cfg.result_dir(seed=43).name == "seed_043"


@pytest.mark.parametrize(
    "main_key",
    [
        "ds003061_within_subject_main",
        "ds005863_within_subject_max15",
        "ds006466_within_subject_n15",
        "ds006480_within_subject_max30_wide_timebin",
    ],
)
def test_experiment_family_resolves(main_key):
    family = config.get_experiment_family(main_key)
    assert set(family) == {"main", "permuted", "prestim", "loso"}
    assert family["main"].control_mode == "standard"
    assert family["permuted"].control_mode == "permuted_labels"
    assert family["prestim"].effective_crop_window() == (-0.2, 0.0)
    assert family["loso"].eval_mode == "loso"
    # All members share data, subject and signal settings.
    for cfg in family.values():
        assert cfg.dataset == family["main"].dataset
        assert cfg.subjects == family["main"].subjects


def test_legacy_aliases_point_to_same_config():
    assert config.get_experiment("ds003061_control_permuted") is config.get_experiment(
        "ds003061_control_permuted_main"
    )
    assert config.get_experiment("ds005863_within_subject_n15") is config.get_experiment(
        "ds005863_within_subject_max15"
    )
    assert config.get_experiment(
        "ds006480_active_within_subject_n15_timebin"
    ) is config.get_experiment("ds006480_active_within_subject_max15_timebin")


def test_unknown_experiment_lists_options():
    with pytest.raises(ValueError, match="Valid options"):
        config.get_experiment("does_not_exist")


def test_ds006466_final_subject_set_is_fixed():
    subjects = config.get_experiment("ds006466_within_subject_max30_wide_timebin").subjects.subjects
    assert len(subjects) == len(set(subjects)) == 30
    assert subjects[0] == "AgoBurn" and subjects[-1] == "HomeDye"
