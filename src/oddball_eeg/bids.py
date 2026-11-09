"""Locating, reading and epoching the OpenNeuro BIDS datasets.

This module turns a BIDS recording into two :class:`mne.Epochs` objects
(``standard`` and ``target``). It contains the dataset-specific event
definitions for the four datasets used in the study (see ``docs/DATASETS.md``)
and the shared raw-signal preprocessing.

Preprocessing applied to every recording (:func:`preprocess_raw`):

1. Non-EEG auxiliary channels (``Temp``, ``Resp``, ``GSR*``, ``EXG*``, ...) are
   dropped.
2. Common average reference.
3. Band-pass filter (``l_freq``-``h_freq``), notch filter, resampling.
"""

from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mne
import mne_bids
import numpy as np
from mne_bids import read_raw_bids

from oddball_eeg import paths

# Keep notebook output readable; individual calls also pass verbose=False.
mne.set_log_level("ERROR")

__all__ = [
    "EEG_EXTS",
    "EpochingConfig",
    "DS005863_STD_CODES",
    "DS005863_TGT_CODES",
    "DS006466_STD_CODES",
    "DS006466_TGT_CODES",
    "DS006480_EVENT_PRESETS",
    "DS006480_STD_CODES",
    "DS006480_TGT_CODES",
    "bids_root_candidates",
    "find_bids_root",
    "pick_task",
    "find_eeg_data_paths",
    "repair_brainvision_header_links",
    "make_epoching_config_for_dataset",
    "make_epoching_config_from_experiment",
    "preprocess_raw",
    "make_epochs_std_tgt",
    "load_subject_epochs_from_bids",
]

EEG_EXTS = [".vhdr", ".edf", ".bdf", ".set"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Dataset discovery
# ============================================================


def bids_root_candidates(dataset_id: str) -> list[Path]:
    """Locations searched (in order) for a dataset's BIDS root."""
    cwd = Path.cwd()
    candidates = [
        paths.data_dir() / dataset_id,
        paths.project_root() / "datasets" / "openneuro" / dataset_id,
        paths.project_root() / "data" / "openneuro" / dataset_id,
        cwd / dataset_id,
        cwd / "datasets" / "openneuro" / dataset_id,
        cwd / "data" / "openneuro" / dataset_id,
        cwd.parent / dataset_id,
        cwd.parent / "datasets" / "openneuro" / dataset_id,
        Path("/mnt/data") / dataset_id,
        Path("/mnt/data") / "datasets" / "openneuro" / dataset_id,
        Path("/mnt/data") / "data" / "openneuro" / dataset_id,
    ]
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def find_bids_root(dataset_id: str) -> Path:
    """Return the directory holding ``dataset_description.json`` for ``dataset_id``.

    Set ``ODDBALL_EEG_DATA_DIR`` to the folder that contains the dataset
    directories to use a location outside the repository.
    """
    candidates = bids_root_candidates(dataset_id)
    for c in candidates:
        if (c / "dataset_description.json").exists():
            return c
    raise FileNotFoundError(
        f"Could not find BIDS root for {dataset_id}. Set {paths.ENV_DATA} or place the "
        "dataset in one of:\n" + "\n".join(str(x) for x in candidates)
    )


def pick_task(bids_root: Path, preferred: str) -> str:
    """Return ``preferred`` if the dataset has that task, else its first task."""
    tasks_all = sorted(mne_bids.get_entity_vals(bids_root, "task"))
    return preferred if preferred in tasks_all else (tasks_all[0] if tasks_all else preferred)


def find_eeg_data_paths(bids_root: Path, subject: str, task: str):
    if not hasattr(mne_bids, "find_matching_paths"):
        raise RuntimeError("mne-bids is too old: missing find_matching_paths().")

    kwargs = dict(
        root=bids_root,
        subjects=subject,
        tasks=task,
        datatypes="eeg",
        suffixes="eeg",
    )
    sig = inspect.signature(mne_bids.find_matching_paths)
    if "extensions" in sig.parameters:
        kwargs["extensions"] = EEG_EXTS
    if "ignore_json" in sig.parameters:
        kwargs["ignore_json"] = True

    keep = []
    for bp in mne_bids.find_matching_paths(**kwargs):
        fpath = bp.fpath if hasattr(bp, "fpath") else Path(str(bp))
        if fpath.suffix.lower() in EEG_EXTS:
            keep.append(bp)
    return keep


def repair_brainvision_header_links(
    vhdr_path: Path,
    *,
    make_backup: bool = True,
    verbose: bool = False,
) -> bool:
    """
    Repair BrainVision .vhdr files whose DataFile/MarkerFile entries point to
    old non-BIDS filenames.

    For a BIDS file like:
        sub-001_task-visualoddball_eeg.vhdr

    this function expects:
        sub-001_task-visualoddball_eeg.eeg
        sub-001_task-visualoddball_eeg.vmrk

    Returns
    -------
    changed : bool
        True if the header was modified.
    """
    vhdr_path = Path(vhdr_path)

    if vhdr_path.suffix.lower() != ".vhdr":
        return False

    if not vhdr_path.exists():
        return False

    text = vhdr_path.read_text(encoding="utf-8", errors="replace")

    data_match = re.search(r"^DataFile=(.+)$", text, flags=re.MULTILINE)
    marker_match = re.search(r"^MarkerFile=(.+)$", text, flags=re.MULTILINE)

    if data_match is None or marker_match is None:
        return False

    current_data_name = data_match.group(1).strip()
    current_marker_name = marker_match.group(1).strip()

    current_data_path = vhdr_path.parent / current_data_name
    current_marker_path = vhdr_path.parent / current_marker_name

    expected_data_name = vhdr_path.with_suffix(".eeg").name
    expected_marker_name = vhdr_path.with_suffix(".vmrk").name

    expected_data_path = vhdr_path.parent / expected_data_name
    expected_marker_path = vhdr_path.parent / expected_marker_name

    current_ok = current_data_path.exists() and current_marker_path.exists()
    expected_ok = expected_data_path.exists() and expected_marker_path.exists()

    if current_ok:
        return False

    if not expected_ok:
        if verbose:
            print("[repair_brainvision_header_links] Cannot repair:")
            print("  vhdr:", vhdr_path)
            print("  missing current data:", current_data_path)
            print("  missing current marker:", current_marker_path)
            print("  expected data exists:", expected_data_path.exists(), expected_data_path)
            print("  expected marker exists:", expected_marker_path.exists(), expected_marker_path)
        return False

    fixed = re.sub(
        r"^DataFile=.*$",
        f"DataFile={expected_data_name}",
        text,
        flags=re.MULTILINE,
    )
    fixed = re.sub(
        r"^MarkerFile=.*$",
        f"MarkerFile={expected_marker_name}",
        fixed,
        flags=re.MULTILINE,
    )

    if fixed == text:
        return False

    if make_backup:
        backup_path = vhdr_path.with_suffix(".vhdr.bak")
        if not backup_path.exists():
            backup_path.write_text(text, encoding="utf-8")

    vhdr_path.write_text(fixed, encoding="utf-8")

    if verbose:
        print("[repair_brainvision_header_links] Repaired:", vhdr_path.name)
        print("  DataFile  :", current_data_name, "->", expected_data_name)
        print("  MarkerFile:", current_marker_name, "->", expected_marker_name)

    return True


# ============================================================
# Epoching configuration and dataset event presets
# ============================================================


@dataclass
class EpochingConfig:
    """Signal-processing and event-selection settings for one dataset.

    Two event-selection modes are supported:

    * descriptor mode (``use_event_codes=False``; ds003061): annotations are
      matched by their exact description;
    * numeric-code mode (``use_event_codes=True``; ds005863, ds006466,
      ds006480): the trailing integer of each annotation description is
      matched against ``std_codes`` / ``tgt_codes``.
    """

    l_freq: float = 0.1
    h_freq: float = 30.0
    notch: float = 60.0
    resample_sfreq: float = 128.0
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)

    # Descriptor mode (ds003061). The misspelled "reponse" variant occurs in the
    # source annotations of some recordings.
    std_desc: str | None = "stimulus/standard"
    tgt_descs: tuple[str, ...] = (
        "stimulus/oddball",
        "stimulus/oddball_with_reponse",
        "stimulus/oddball_with_response",
    )

    # Numeric-code mode.
    use_event_codes: bool = False
    std_codes: tuple[int, ...] = ()
    tgt_codes: tuple[int, ...] = ()


# ds005863 (visual oddball): target = same letter pair, standard = different pair.
DS005863_STD_CODES: tuple[int, ...] = (
    12, 13, 14, 15,
    21, 23, 24, 25,
    31, 32, 34, 35,
    41, 42, 43, 45,
    51, 52, 53, 54,
)  # fmt: skip
DS005863_TGT_CODES: tuple[int, ...] = (11, 22, 33, 44, 55)

# ds006466 (older-adult auditory oddball). The analysed task is "active", in
# which only 113 (standard) / 114 (target) occur; the other condition codes are
# kept so that the preset also works for the passive/shock tasks.
#   passive control 106/107, active control 113/114,
#   passive shock   125/126, active shock   132/133
DS006466_STD_CODES: tuple[int, ...] = (106, 113, 125, 132)
DS006466_TGT_CODES: tuple[int, ...] = (107, 114, 126, 133)

# ds006480 (young-adult auditory oddball). Distractors (108, 115, 127, 134) are
# never included. The final benchmark uses the "pooled" preset.
DS006480_EVENT_PRESETS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "pooled": ((106, 113, 125, 132), (107, 114, 126, 133)),
    "control": ((106, 113), (107, 114)),
    "shock": ((125, 132), (126, 133)),
    "active": ((113, 132), (114, 133)),
    "passive": ((106, 125), (107, 126)),
    "active_control": ((113,), (114,)),
    "passive_control": ((106,), (107,)),
    "active_shock": ((132,), (133,)),
    "passive_shock": ((125,), (126,)),
}
DS006480_STD_CODES, DS006480_TGT_CODES = DS006480_EVENT_PRESETS["pooled"]


def make_epoching_config_for_dataset(
    dataset: str,
    *,
    l_freq: float = 0.1,
    h_freq: float = 12.0,
    notch: float = 60.0,
    resample_sfreq: float = 128.0,
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple[float, float] = (-0.2, 0.0),
    event_preset: str = "pooled",
) -> EpochingConfig:
    """Build the :class:`EpochingConfig` for a supported dataset.

    ds003061 uses descriptor mode; ds005863, ds006466 and ds006480 use
    numeric-code mode. ``event_preset`` only matters for ds006480.
    """
    dataset = str(dataset).strip().lower()
    event_preset = str(event_preset).strip().lower()

    signal = dict(
        l_freq=l_freq,
        h_freq=h_freq,
        notch=notch,
        resample_sfreq=resample_sfreq,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
    )

    if dataset == "ds005863":
        std_codes, tgt_codes = DS005863_STD_CODES, DS005863_TGT_CODES
    elif dataset == "ds006466":
        std_codes, tgt_codes = DS006466_STD_CODES, DS006466_TGT_CODES
    elif dataset == "ds006480":
        if event_preset not in DS006480_EVENT_PRESETS:
            raise ValueError(
                f"Unknown ds006480 event_preset={event_preset!r}. "
                f"Valid options: {sorted(DS006480_EVENT_PRESETS)}"
            )
        std_codes, tgt_codes = DS006480_EVENT_PRESETS[event_preset]
    else:
        # ds003061 and any other descriptor-annotated dataset.
        return EpochingConfig(**signal, use_event_codes=False)

    return EpochingConfig(
        **signal,
        use_event_codes=True,
        std_codes=tuple(std_codes),
        tgt_codes=tuple(tgt_codes),
    )


def make_epoching_config_from_experiment(cfg) -> EpochingConfig:
    """Build an :class:`EpochingConfig` from an ``ExperimentConfig``.

    This guarantees that the signal slug in the result path matches the
    preprocessing actually applied.
    """
    return make_epoching_config_for_dataset(
        cfg.dataset.dataset,
        l_freq=cfg.signal.l_freq,
        h_freq=cfg.signal.h_freq,
        notch=cfg.signal.notch,
        resample_sfreq=cfg.signal.resample_sfreq,
        tmin=cfg.signal.tmin,
        tmax=cfg.signal.tmax,
        baseline=cfg.signal.baseline,
        event_preset=getattr(cfg.dataset, "event_preset", "pooled"),
    )


# ============================================================
# Raw preprocessing and epoching
# ============================================================

_AUX_CHANNEL_NAMES = {"Temp", "Resp", "Plet", "Erg1", "Erg2", "GSR1", "GSR2"}


def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: EpochingConfig,
    *,
    keep_eog: bool = False,
) -> mne.io.BaseRaw:
    """Pick EEG, re-reference, filter and resample ``raw`` in place.

    With ``keep_eog=True`` the ``EXG*`` channels are retained as EOG instead of
    being discarded.
    """
    raw.load_data()

    to_types = {}
    for ch in raw.ch_names:
        up = ch.upper()
        if keep_eog and up.startswith("EXG"):
            to_types[ch] = "eog"
            continue
        if (
            ch in _AUX_CHANNEL_NAMES
            or up.startswith("GSR")
            or (up.startswith("EXG") and not keep_eog)
        ):
            to_types[ch] = "misc"

    if to_types:
        raw.set_channel_types(to_types)

    raw.pick_types(eeg=True, eog=keep_eog, meg=False, stim=False, misc=False)
    raw.set_eeg_reference("average", projection=False)

    if raw.get_montage() is None:
        raw.set_montage("standard_1020", on_missing="ignore")

    raw.filter(cfg.l_freq, cfg.h_freq, picks="eeg", verbose=False)
    raw.notch_filter(cfg.notch, picks="eeg", verbose=False)
    raw.resample(cfg.resample_sfreq, npad="auto", verbose=False)
    return raw


def _parse_event_code_from_description(desc) -> int | None:
    """Return the last integer in an annotation description.

    ``'S 11' -> 11``, ``'Stimulus/S 11' -> 11``, ``'S201' -> 201``,
    ``'boundary' -> None``.
    """
    nums = re.findall(r"\d+", str(desc))
    if not nums:
        return None
    return int(nums[-1])


def _print_available_annotations(ann_id: dict, parse_codes: bool) -> None:
    print("Available ann_id:")
    for k, v in sorted(ann_id.items(), key=lambda kv: kv[1]):
        if parse_codes:
            parsed = _parse_event_code_from_description(k)
            print(f"  event_id={int(v):6d}  parsed={parsed}  desc={k!r}")
        else:
            print(f"  {v:6d}  {k!r}")


def _binary_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    std_event_ids: list[int],
    tgt_event_ids: list[int],
    cfg: EpochingConfig,
    apply_baseline: bool,
) -> tuple[mne.Epochs, mne.Epochs] | tuple[None, None]:
    """Recode ``events`` to standard=1 / target=2 and epoch each class."""
    selected = np.isin(events[:, 2].astype(int), list(std_event_ids) + list(tgt_event_ids))
    new_events = events[selected].copy()
    new_events = new_events[np.argsort(new_events[:, 0])]

    if len(new_events) == 0:
        print("[make_epochs_std_tgt] Event IDs were matched in ann_id, but no events were found.")
        print("std_event_ids:", std_event_ids)
        print("tgt_event_ids:", tgt_event_ids)
        print("Available event IDs:", sorted(set(events[:, 2].astype(int))))
        return None, None

    is_std = np.isin(new_events[:, 2].astype(int), std_event_ids)
    is_tgt = np.isin(new_events[:, 2].astype(int), tgt_event_ids)
    new_events[is_std, 2] = 1
    new_events[is_tgt, 2] = 2

    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        print("[make_epochs_std_tgt] No EEG channels available after picking.")
        return None, None

    common = dict(
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline if apply_baseline else None,
        preload=True,
        picks=picks,
        reject_by_annotation=True,
        verbose=False,
    )
    epochs_std = mne.Epochs(raw, new_events, event_id={"standard": 1}, **common)
    epochs_tgt = mne.Epochs(raw, new_events, event_id={"target": 2}, **common)

    print(
        f"[make_epochs_std_tgt] Created epochs: standard={len(epochs_std)}, target={len(epochs_tgt)}"
    )

    if len(epochs_std) == 0 or len(epochs_tgt) == 0:
        return None, None
    return epochs_std, epochs_tgt


def make_epochs_std_tgt(raw: mne.io.BaseRaw, cfg: EpochingConfig, apply_baseline: bool = True):
    """Build ``(standard, target)`` epochs from the annotations of ``raw``.

    Returns ``(None, None)`` when the requested events are not present.
    """
    events, ann_id = mne.events_from_annotations(raw, event_id="auto", verbose=False)

    if len(events) == 0 or len(ann_id) == 0:
        print("[make_epochs_std_tgt] No events/annotations found.")
        return None, None

    if cfg.use_event_codes:
        std_codes = {int(c) for c in cfg.std_codes}
        tgt_codes = {int(c) for c in cfg.tgt_codes}

        if not std_codes or not tgt_codes:
            print("[make_epochs_std_tgt] use_event_codes=True but std_codes/tgt_codes are empty.")
            return None, None

        # MNE assigns arbitrary event IDs (10001, ...), so map each annotation
        # description back to the marker code it encodes.
        std_event_ids: set[int] = set()
        tgt_event_ids: set[int] = set()
        for desc, event_id in ann_id.items():
            code = _parse_event_code_from_description(desc)
            if code is None:
                continue
            if code in std_codes:
                std_event_ids.add(int(event_id))
            if code in tgt_codes:
                tgt_event_ids.add(int(event_id))

        if not std_event_ids or not tgt_event_ids:
            print(
                "[make_epochs_std_tgt] No annotation descriptions matched requested numeric codes."
            )
            print("Requested std_codes:", sorted(std_codes))
            print("Requested tgt_codes:", sorted(tgt_codes))
            _print_available_annotations(ann_id, parse_codes=True)
            return None, None

        return _binary_epochs(
            raw, events, sorted(std_event_ids), sorted(tgt_event_ids), cfg, apply_baseline
        )

    # Descriptor mode (ds003061).
    if cfg.std_desc not in ann_id:
        print("[make_epochs_std_tgt] std_desc not found:", cfg.std_desc)
        _print_available_annotations(ann_id, parse_codes=False)
        return None, None

    tgt_present = [k for k in cfg.tgt_descs if k in ann_id]
    if not tgt_present:
        print("[make_epochs_std_tgt] No target descriptors found.")
        print("Requested:", cfg.tgt_descs)
        _print_available_annotations(ann_id, parse_codes=False)
        return None, None

    return _binary_epochs(
        raw,
        events,
        [int(ann_id[cfg.std_desc])],
        [int(ann_id[k]) for k in tgt_present],
        cfg,
        apply_baseline,
    )


def load_subject_epochs_from_bids(
    bids_root: Path,
    subject: str,
    task: str,
    cfg: EpochingConfig,
    apply_baseline: bool = True,
    *,
    keep_eog: bool = False,
):
    """Load, preprocess and epoch every EEG run of one subject.

    Returns ``(epochs_std, epochs_tgt, epochs_all)`` with channels in
    alphabetical order, or ``(None, None, None)`` if no run produced both
    classes. Runs that fail to load are reported and skipped.

    Note
    ----
    BrainVision headers whose ``DataFile``/``MarkerFile`` entries point to
    pre-BIDS filenames are repaired *on disk* before reading (a ``.vhdr.bak``
    backup is written once). See :func:`repair_brainvision_header_links`.
    """
    run_paths = find_eeg_data_paths(bids_root, subject, task)
    std_list, tgt_list = [], []

    for bids_path in run_paths:
        try:
            fpath = bids_path.fpath if hasattr(bids_path, "fpath") else Path(str(bids_path))

            # BrainVision datasets sometimes contain stale internal references:
            # DataFile=old_name.eeg / MarkerFile=old_name.vmrk.
            # Repair before calling read_raw_bids().
            repair_brainvision_header_links(
                fpath,
                make_backup=True,
                verbose=False,
            )

            # Keep logs report-clean: suppress known noisy BIDS/channel warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Unknown types found, setting as type EEG*"
                )
                warnings.filterwarnings("ignore", message="Data will be preloaded*")
                warnings.filterwarnings("ignore", message="Not setting positions of*")
                warnings.filterwarnings("ignore", message="There are channels without locations*")
                warnings.filterwarnings("ignore", message="Unable to map the following column*")
                warnings.filterwarnings(
                    "ignore", message="No coordinate information found for channels*"
                )
                warnings.filterwarnings("ignore", message="Did not find any channels.tsv*")
                warnings.filterwarnings("ignore", message="The search_str was*")
                raw = read_raw_bids(bids_path=bids_path, verbose=False)

            raw = preprocess_raw(raw, cfg, keep_eog=keep_eog)
            e_std, e_tgt = make_epochs_std_tgt(raw, cfg, apply_baseline=apply_baseline)
            if e_std is None or e_tgt is None:
                continue
            if len(e_std) > 0:
                std_list.append(e_std)
            if len(e_tgt) > 0:
                tgt_list.append(e_tgt)
        except Exception as e:
            print(f"[load_subject_epochs_from_bids] Failed on {bids_path}")
            print(type(e).__name__, repr(e))
            continue

    if len(std_list) == 0 or len(tgt_list) == 0:
        return None, None, None

    epochs_std = mne.concatenate_epochs(std_list)
    epochs_tgt = mne.concatenate_epochs(tgt_list)
    epochs_all = mne.concatenate_epochs([epochs_std, epochs_tgt])

    order = sorted(epochs_std.ch_names)
    epochs_std.reorder_channels(order)
    epochs_tgt.reorder_channels(order)
    epochs_all.reorder_channels(order)

    return epochs_std, epochs_tgt, epochs_all
