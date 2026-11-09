from __future__ import annotations

import mne
import numpy as np
import pytest

from oddball_eeg import bids, paths


def _raw_with_annotations(descriptions, onsets, sfreq=128.0, duration=30.0):
    info = mne.create_info(["Fz", "Cz", "Pz"], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((3, int(duration * sfreq))), info, verbose=False)
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), descriptions))
    return raw


def test_event_presets_exclude_distractors():
    distractors = {108, 115, 127, 134}
    for std, tgt in bids.DS006480_EVENT_PRESETS.values():
        assert not distractors & (set(std) | set(tgt))
        assert not set(std) & set(tgt)


@pytest.mark.parametrize(
    "dataset,preset,std,tgt",
    [
        ("ds005863", "pooled", bids.DS005863_STD_CODES, bids.DS005863_TGT_CODES),
        ("ds006466", "pooled", bids.DS006466_STD_CODES, bids.DS006466_TGT_CODES),
        ("ds006480", "active", (113, 132), (114, 133)),
    ],
)
def test_numeric_code_configs(dataset, preset, std, tgt):
    cfg = bids.make_epoching_config_for_dataset(dataset, event_preset=preset)
    assert cfg.use_event_codes and cfg.std_codes == tuple(std) and cfg.tgt_codes == tuple(tgt)


def test_descriptor_config_for_ds003061():
    cfg = bids.make_epoching_config_for_dataset("ds003061", h_freq=12.0)
    assert not cfg.use_event_codes and cfg.h_freq == 12.0


def test_unknown_ds006480_preset_raises():
    with pytest.raises(ValueError, match="event_preset"):
        bids.make_epoching_config_for_dataset("ds006480", event_preset="nope")


def test_numeric_mode_epochs_and_ignores_distractors():
    onsets = [1, 2, 3, 4, 5, 6, 7]
    desc = ["113", "114", "113", "115", "113", "114", "boundary"]
    raw = _raw_with_annotations(desc, onsets)
    cfg = bids.make_epoching_config_for_dataset("ds006480", event_preset="active_control")
    std, tgt = bids.make_epochs_std_tgt(raw, cfg)
    assert len(std) == 3 and len(tgt) == 2


def test_target_with_lowest_event_id_is_not_mislabelled():
    # With event_id="auto", "stimulus/oddball" sorts first and receives MNE id 1,
    # the same value standards are recoded to. Standards must stay standards.
    desc = ["stimulus/standard", "stimulus/oddball", "stimulus/standard", "stimulus/standard"]
    raw = _raw_with_annotations(desc, [1, 2, 3, 4])
    std, tgt = bids.make_epochs_std_tgt(raw, bids.EpochingConfig())
    assert len(std) == 3 and len(tgt) == 1


def test_missing_events_return_none():
    raw = _raw_with_annotations(["foo", "bar"], [1, 2])
    assert bids.make_epochs_std_tgt(raw, bids.EpochingConfig()) == (None, None)


def test_parse_event_code():
    assert bids._parse_event_code_from_description("Stimulus/S 11") == 11
    assert bids._parse_event_code_from_description("boundary") is None


def test_find_bids_root_uses_data_dir_env(tmp_path, monkeypatch):
    ds = tmp_path / "ds003061"
    ds.mkdir()
    (ds / "dataset_description.json").write_text("{}")
    monkeypatch.setenv(paths.ENV_DATA, str(tmp_path))
    assert bids.find_bids_root("ds003061") == ds


def test_find_bids_root_error_mentions_env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=paths.ENV_DATA):
        bids.find_bids_root("ds999999")


def test_repair_brainvision_header(tmp_path):
    vhdr = tmp_path / "sub-01_task-x_eeg.vhdr"
    (tmp_path / "sub-01_task-x_eeg.eeg").write_bytes(b"")
    (tmp_path / "sub-01_task-x_eeg.vmrk").write_text("")
    vhdr.write_text("DataFile=old.eeg\nMarkerFile=old.vmrk\n")
    assert bids.repair_brainvision_header_links(vhdr)
    assert "DataFile=sub-01_task-x_eeg.eeg" in vhdr.read_text()
    assert (tmp_path / "sub-01_task-x_eeg.vhdr.bak").exists()
    assert not bids.repair_brainvision_header_links(vhdr)  # idempotent
