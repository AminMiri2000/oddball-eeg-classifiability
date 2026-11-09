"""Experiment configuration and the experiment registry.

Every analysis is driven by an :class:`ExperimentConfig`, looked up by key from
the registry::

    from oddball_eeg.config import get_experiment, get_experiment_family

    cfg = get_experiment("ds003061_within_subject_max30_wide_timebin")
    family = get_experiment_family("ds003061_within_subject_max30_wide_timebin")
    family["loso"].result_dir()

The final journal benchmark is the ``max30_wide_timebin`` family, listed in
:data:`JOURNAL_WITHIN_SUBJECT_EXPERIMENTS` and :data:`JOURNAL_LOSO_EXPERIMENTS`.
The ``main`` (10-subject) and ``max15`` families are development-stage
configurations kept so that the provenance notebooks remain runnable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from oddball_eeg import paths

# ============================================================
# Types
# ============================================================

ControlMode = Literal["standard", "permuted_labels", "prestim_only"]
EvalMode = Literal["within_subject", "loso"]
ModelName = Literal["lda", "logreg", "eegnet"]
FeatureMode = Literal["raw_flat", "time_binned"]


def _fmt_number(x: float) -> str:
    """Format a number for use in directory names (``0.1 -> 0p1``, ``-0.2 -> m0p2``)."""
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return str(x).replace(".", "p").replace("-", "m")


# ============================================================
# Config dataclasses
# ============================================================


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "ds003061"
    task: str = "P300"
    # Event-code preset passed to ``bids.make_epoching_config_for_dataset``.
    # Ignored for descriptor-based ds003061.
    event_preset: str = "pooled"


@dataclass(frozen=True)
class SubjectConfig:
    subjects: tuple[str, ...] = tuple(f"{i:03d}" for i in range(1, 11))


@dataclass(frozen=True)
class SeedConfig:
    seeds: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True)
class SplitConfig:
    """Chronological grouped cross-validation settings."""

    n_splits: int = 5
    group_chunk_size: int = 80
    purge_neighbor_groups: int = 2
    val_size: float = 0.15
    val_n_tries: int = 200


@dataclass(frozen=True)
class CropConfig:
    use_crop: bool = True
    main_tmin: float = 0.0
    main_tmax: float = 0.6
    prestim_tmin: float = -0.2
    prestim_tmax: float = 0.0

    def slug(self, control_mode: str = "standard") -> str:
        if not self.use_crop:
            return "crop_full"
        if control_mode == "prestim_only":
            return f"crop_{_fmt_number(self.prestim_tmin)}_{_fmt_number(self.prestim_tmax)}s"
        return f"crop_{_fmt_number(self.main_tmin)}_{_fmt_number(self.main_tmax)}s"


@dataclass(frozen=True)
class FeatureConfig:
    mode: FeatureMode = "raw_flat"
    bin_ms: int = 50

    def slug(self) -> str:
        if self.mode == "raw_flat":
            return "feat_rawflat"
        return f"feat_timebin_{int(self.bin_ms)}ms"


@dataclass(frozen=True)
class SignalConfig:
    l_freq: float = 0.1
    h_freq: float = 12.0
    notch: float = 60.0
    resample_sfreq: float = 128.0
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)

    def band_slug(self) -> str:
        return f"bp_{_fmt_number(self.l_freq)}_{_fmt_number(self.h_freq)}Hz"

    def epoch_slug(self) -> str:
        return f"epoch_{_fmt_number(self.tmin)}_{_fmt_number(self.tmax)}s"

    def slug(self) -> str:
        return f"{self.band_slug()}_{self.epoch_slug()}_rs{_fmt_number(self.resample_sfreq)}Hz"


@dataclass(frozen=True)
class EEGNetConfig:
    n_classes: int = 2
    f1: int = 8
    d: int = 2
    f2: int = 16
    k1: int = 32
    k2: int = 16
    dropout: float = 0.25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    early_stopping_patience: int = 10
    normalize: bool = True
    use_class_weight: bool = True
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    eval_mode: EvalMode = "within_subject"
    control_mode: ControlMode = "standard"
    run_tag: str = "default"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    subjects: SubjectConfig = field(default_factory=SubjectConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    eegnet: EEGNetConfig = field(default_factory=EEGNetConfig)

    save_fold_predictions: bool = False
    save_learning_curves: bool = True
    verbose: bool = True

    # ------------------------------------------------------------------
    # Derived settings
    # ------------------------------------------------------------------

    def effective_crop_window(self) -> tuple[float, float] | None:
        if not self.crop.use_crop:
            return None
        if self.control_mode == "prestim_only":
            return (self.crop.prestim_tmin, self.crop.prestim_tmax)
        return (self.crop.main_tmin, self.crop.main_tmax)

    def _relative_run_path(self) -> Path:
        return (
            Path(self.dataset.dataset)
            / self.dataset.task
            / f"events_{self.dataset.event_preset}"
            / self.signal.slug()
            / self.run_tag
            / self.crop.slug(self.control_mode)
            / self.feature.slug()
            / self.eval_mode
            / self.control_mode
        )

    # ------------------------------------------------------------------
    # Output locations (created on access)
    # ------------------------------------------------------------------

    def result_dir(self, seed: int | None = None) -> Path:
        out_dir = paths.results_dir() / self._relative_run_path()
        if seed is not None:
            out_dir = out_dir / "seeds" / f"seed_{int(seed):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _subdir(self, kind: str, *parts: str, seed: int | None = None) -> Path:
        out_dir = self.result_dir(seed=seed) / kind
        for part in parts:
            out_dir = out_dir / str(part)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def table_dir(self, *parts: str, seed: int | None = None) -> Path:
        return self._subdir("tables", *parts, seed=seed)

    def figure_dir(self, *parts: str, seed: int | None = None) -> Path:
        return self._subdir("figures", *parts, seed=seed)

    def history_dir(self, *parts: str, seed: int | None = None) -> Path:
        return self._subdir("histories", *parts, seed=seed)

    def checkpoint_dir(self, *parts: str, seed: int | None = None) -> Path:
        return self._subdir("checkpoints", *parts, seed=seed)

    def artifact_dir(self, *parts: str) -> Path:
        return self._subdir("artifacts", *parts)

    def metadata_dir(self) -> Path:
        return self._subdir("metadata")

    def log_file(self, seed: int | None = None) -> Path:
        log_dir = paths.logs_dir() / self._relative_run_path()
        log_dir.mkdir(parents=True, exist_ok=True)
        if seed is None:
            return log_dir / "run.log"
        return log_dir / f"seed_{int(seed):03d}.log"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["project_root"] = str(paths.project_root())
        d["results_dir"] = str(paths.results_dir())
        d["logs_dir"] = str(paths.logs_dir())
        d["effective_crop_window"] = self.effective_crop_window()
        d["result_dir"] = str(self.result_dir())
        d["table_dir"] = str(self.table_dir())
        d["figure_dir"] = str(self.figure_dir())
        d["metadata_dir"] = str(self.metadata_dir())
        d["signal_slug"] = self.signal.slug()
        d["crop_slug"] = self.crop.slug(self.control_mode)
        d["feature_slug"] = self.feature.slug()
        d["event_preset"] = self.dataset.event_preset
        return d


# ============================================================
# Shared presets
# ============================================================

SIGNAL_0P1_12 = SignalConfig(l_freq=0.1, h_freq=12.0)
SIGNAL_0P1_30 = SignalConfig(l_freq=0.1, h_freq=30.0)
SIGNAL_0P1_45 = SignalConfig(l_freq=0.1, h_freq=45.0)

CROP_0_0P6 = CropConfig(use_crop=True, main_tmin=0.0, main_tmax=0.6)
CROP_0_0P8 = CropConfig(use_crop=True, main_tmin=0.0, main_tmax=0.8)

FEATURE_RAW_FLAT = FeatureConfig(mode="raw_flat", bin_ms=50)
FEATURE_TIME_BIN_50MS = FeatureConfig(mode="time_binned", bin_ms=50)

SEEDS_3 = SeedConfig(seeds=(42, 43, 44))

# Chunk sizes are dataset-specific because target density differs.
SPLITS_DEFAULT = SplitConfig(group_chunk_size=80, purge_neighbor_groups=2)
SPLITS_DS005863 = SplitConfig(group_chunk_size=20, purge_neighbor_groups=1)
SPLITS_DS006466 = SplitConfig(group_chunk_size=40, purge_neighbor_groups=1)
SPLITS_N15_DEFAULT = SplitConfig(group_chunk_size=40, purge_neighbor_groups=1)
SPLITS_N15_DENSE_EVENTS = SplitConfig(group_chunk_size=80, purge_neighbor_groups=1)
SPLITS_N15_SMALL_CHUNKS = SplitConfig(group_chunk_size=20, purge_neighbor_groups=1)

# ============================================================
# Datasets
# ============================================================

DS003061 = DatasetConfig(dataset="ds003061", task="P300", event_preset="pooled")
DS005863 = DatasetConfig(dataset="ds005863", task="visualoddball", event_preset="pooled")
DS006466_ACTIVE = DatasetConfig(dataset="ds006466", task="active", event_preset="pooled")
DS006480_POOLED = DatasetConfig(dataset="ds006480", task="oddball", event_preset="pooled")
DS006480_ACTIVE = DatasetConfig(dataset="ds006480", task="oddball", event_preset="active")
DS006480_PASSIVE = DatasetConfig(dataset="ds006480", task="oddball", event_preset="passive")
DS006480_CONTROL = DatasetConfig(dataset="ds006480", task="oddball", event_preset="control")
DS006480_SHOCK = DatasetConfig(dataset="ds006480", task="oddball", event_preset="shock")

# ============================================================
# Subject sets
# ============================================================


def _numbered(n: int, width: int) -> SubjectConfig:
    return SubjectConfig(subjects=tuple(f"{i:0{width}d}" for i in range(1, n + 1)))


_DS006466_ALL_30: tuple[str, ...] = (
    "AgoBurn", "AgoHake", "ArcoWeb", "AridWar", "AuldEve",
    "AvidJar", "AzoAgon", "BoneSip", "BossBob", "BossRip",
    "CalmJib", "CampHum", "CampOhm", "DearSop", "DrySolo",
    "EpicAwl", "EpicOla", "EvenBox", "EyedEgo", "FarTube",
    "FeyHeat", "FirmPhi", "FitHame", "FivePly", "FourSpy",
    "HeldSou", "HepBleu", "HepViol", "HewnMod", "HomeDye",
)  # fmt: skip

# Development-stage 10-subject subsets.
SUBJECTS_001_010 = _numbered(10, 3)
SUBJECTS_01_10 = _numbered(10, 2)
DS006466_SUBJECTS_10 = SubjectConfig(subjects=_DS006466_ALL_30[:10])

# "max15": up to 15 subjects per dataset (ds003061 only has 13).
DS003061_SUBJECTS_FINAL = _numbered(13, 3)
DS005863_SUBJECTS_FINAL = _numbered(15, 3)
DS006480_SUBJECTS_FINAL = _numbered(15, 2)
DS006466_SUBJECTS_15 = SubjectConfig(subjects=_DS006466_ALL_30[:15])

# "max30": the final journal benchmark (13 + 30 + 30 + 30 = 103 recordings).
DS003061_SUBJECTS_MAX30 = DS003061_SUBJECTS_FINAL
DS005863_SUBJECTS_MAX30 = _numbered(30, 3)
DS006480_SUBJECTS_MAX30 = _numbered(30, 2)
DS006466_SUBJECTS_MAX30 = SubjectConfig(subjects=_DS006466_ALL_30)

# ============================================================
# Registry
# ============================================================

EXPERIMENTS: dict[str, ExperimentConfig] = {}

#: Experiment kinds produced by :func:`register_family`.
FAMILY_KINDS: dict[str, tuple[EvalMode, ControlMode]] = {
    "within_subject": ("within_subject", "standard"),
    "control_permuted": ("within_subject", "permuted_labels"),
    "control_prestim": ("within_subject", "prestim_only"),
    "loso": ("loso", "standard"),
}

_FAMILY_ROLE_TO_KIND = {
    "main": "within_subject",
    "permuted": "control_permuted",
    "prestim": "control_prestim",
    "loso": "loso",
}


def _register(key: str, cfg: ExperimentConfig) -> ExperimentConfig:
    if key in EXPERIMENTS:
        raise ValueError(f"Duplicate experiment key: {key}")
    EXPERIMENTS[key] = cfg
    return cfg


def _alias(alias_key: str, target_key: str) -> None:
    """Register ``alias_key`` as another name for an existing experiment."""
    if target_key not in EXPERIMENTS:
        raise ValueError(f"Cannot alias missing target experiment: {target_key}")
    if alias_key in EXPERIMENTS:
        raise ValueError(f"Duplicate experiment alias/key: {alias_key}")
    EXPERIMENTS[alias_key] = EXPERIMENTS[target_key]


def make_experiment(
    *,
    name: str,
    dataset: DatasetConfig,
    subjects: SubjectConfig,
    splits: SplitConfig,
    eval_mode: EvalMode = "within_subject",
    control_mode: ControlMode = "standard",
    run_tag: str = "default",
    crop: CropConfig = CROP_0_0P6,
    signal: SignalConfig = SIGNAL_0P1_12,
    feature: FeatureConfig = FEATURE_RAW_FLAT,
    seeds: SeedConfig = SEEDS_3,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        eval_mode=eval_mode,
        control_mode=control_mode,
        run_tag=run_tag,
        dataset=dataset,
        subjects=subjects,
        seeds=seeds,
        splits=splits,
        crop=crop,
        signal=signal,
        feature=feature,
    )


def register_family(
    prefix: str,
    *,
    dataset: DatasetConfig,
    subjects: SubjectConfig,
    splits: SplitConfig,
    suffix: str,
    run_tag: str = "default",
    crop: CropConfig = CROP_0_0P6,
    signal: SignalConfig = SIGNAL_0P1_12,
    feature: FeatureConfig = FEATURE_RAW_FLAT,
) -> None:
    """Register the within-subject, control and LOSO experiments for one dataset.

    Keys follow ``<prefix>_<kind>_<suffix>``, e.g.
    ``ds003061_control_permuted_max30_wide_timebin``.
    """
    for kind, (eval_mode, control_mode) in FAMILY_KINDS.items():
        _register(
            f"{prefix}_{kind}_{suffix}",
            make_experiment(
                name=f"{kind}_{suffix}",
                dataset=dataset,
                subjects=subjects,
                splits=splits,
                eval_mode=eval_mode,
                control_mode=control_mode,
                run_tag=run_tag,
                crop=crop,
                signal=signal,
                feature=feature,
            ),
        )


def alias_family(prefix: str, *, target_suffix: str, alias_suffix: str) -> None:
    """Alias every member of a family, e.g. ``*_n15`` -> ``*_max15``."""
    for kind in FAMILY_KINDS:
        _alias(f"{prefix}_{kind}_{alias_suffix}", f"{prefix}_{kind}_{target_suffix}")


_DATASET_PREFIXES = ("ds003061", "ds005863", "ds006466", "ds006480")

# ------------------------------------------------------------
# Development stage: 10-subject "main" families
# ------------------------------------------------------------

for _prefix, _dataset, _subjects, _splits in [
    ("ds003061", DS003061, SUBJECTS_001_010, SPLITS_DEFAULT),
    ("ds005863", DS005863, SUBJECTS_001_010, SPLITS_DS005863),
    ("ds006466", DS006466_ACTIVE, DS006466_SUBJECTS_10, SPLITS_DS006466),
    ("ds006480", DS006480_POOLED, SUBJECTS_01_10, SPLITS_DEFAULT),
]:
    register_family(_prefix, dataset=_dataset, subjects=_subjects, splits=_splits, suffix="main")
    # Original key spellings used in the thesis notebooks.
    _alias(f"{_prefix}_control_permuted", f"{_prefix}_control_permuted_main")
    _alias(f"{_prefix}_control_prestim", f"{_prefix}_control_prestim_main")

# ------------------------------------------------------------
# Development stage: max15 families (raw-flat, 0-0.6 s)
# ------------------------------------------------------------

_MAX15 = [
    ("ds003061", DS003061, DS003061_SUBJECTS_FINAL, SPLITS_N15_DENSE_EVENTS),
    ("ds005863", DS005863, DS005863_SUBJECTS_FINAL, SPLITS_N15_SMALL_CHUNKS),
    ("ds006466", DS006466_ACTIVE, DS006466_SUBJECTS_15, SPLITS_N15_DEFAULT),
    ("ds006480", DS006480_POOLED, DS006480_SUBJECTS_FINAL, SPLITS_N15_DEFAULT),
]

for _prefix, _dataset, _subjects, _splits in _MAX15:
    register_family(
        _prefix, dataset=_dataset, subjects=_subjects, splits=_splits,
        run_tag="max15", suffix="max15",
    )  # fmt: skip
    alias_family(_prefix, target_suffix="max15", alias_suffix="n15")

# Time-binned within-subject refinements of the max15 families.
for _prefix, _dataset, _subjects, _splits in _MAX15:
    for _suffix, _crop in [("max15_timebin", CROP_0_0P6), ("max15_wide_timebin", CROP_0_0P8)]:
        _register(
            f"{_prefix}_within_subject_{_suffix}",
            make_experiment(
                name=f"within_subject_{_suffix}",
                dataset=_dataset,
                subjects=_subjects,
                splits=_splits,
                run_tag=_suffix,
                crop=_crop,
                feature=FEATURE_TIME_BIN_50MS,
            ),
        )
        _alias(
            f"{_prefix}_within_subject_{_suffix.replace('max15', 'n15')}",
            f"{_prefix}_within_subject_{_suffix}",
        )

# ds006480 separated-condition refinements.
for _name, _dataset in [
    ("active", DS006480_ACTIVE),
    ("passive", DS006480_PASSIVE),
    ("control", DS006480_CONTROL),
    ("shock", DS006480_SHOCK),
]:
    for _suffix, _feature in [
        ("max15", FEATURE_RAW_FLAT),
        ("max15_timebin", FEATURE_TIME_BIN_50MS),
    ]:
        _register(
            f"ds006480_{_name}_within_subject_{_suffix}",
            make_experiment(
                name=f"within_subject_max15_{_name}"
                + ("_timebin" if _suffix.endswith("timebin") else ""),
                dataset=_dataset,
                subjects=DS006480_SUBJECTS_FINAL,
                splits=SPLITS_N15_DEFAULT,
                run_tag=f"max15_{_name}" + ("_timebin" if _suffix.endswith("timebin") else ""),
                crop=CROP_0_0P6,
                feature=_feature,
            ),
        )
        _alias(
            f"ds006480_{_name}_within_subject_{_suffix.replace('max15', 'n15')}",
            f"ds006480_{_name}_within_subject_{_suffix}",
        )

# ------------------------------------------------------------
# Final journal benchmark: max30 wide-timebin families
# ------------------------------------------------------------
# - LDA / Logistic Regression: 50-ms time-binned features over 0.0-0.8 s
# - EEGNet: raw cropped epochs over 0.0-0.8 s

for _prefix, _dataset, _subjects, _splits in [
    ("ds003061", DS003061, DS003061_SUBJECTS_MAX30, SPLITS_N15_DENSE_EVENTS),
    ("ds005863", DS005863, DS005863_SUBJECTS_MAX30, SPLITS_N15_SMALL_CHUNKS),
    ("ds006466", DS006466_ACTIVE, DS006466_SUBJECTS_MAX30, SPLITS_N15_DEFAULT),
    ("ds006480", DS006480_POOLED, DS006480_SUBJECTS_MAX30, SPLITS_N15_DEFAULT),
]:
    register_family(
        _prefix,
        dataset=_dataset,
        subjects=_subjects,
        splits=_splits,
        run_tag="max30_wide_timebin",
        crop=CROP_0_0P8,
        feature=FEATURE_TIME_BIN_50MS,
        suffix="max30_wide_timebin",
    )

JOURNAL_FAMILY_SUFFIX = "max30_wide_timebin"

JOURNAL_WITHIN_SUBJECT_EXPERIMENTS: tuple[str, ...] = tuple(
    f"{p}_within_subject_{JOURNAL_FAMILY_SUFFIX}" for p in _DATASET_PREFIXES
)
JOURNAL_LOSO_EXPERIMENTS: tuple[str, ...] = tuple(
    f"{p}_loso_{JOURNAL_FAMILY_SUFFIX}" for p in _DATASET_PREFIXES
)

#: Subject counts of the final benchmark, used by sanity checks.
JOURNAL_EXPECTED_SUBJECT_COUNTS: dict[str, int] = {
    "ds003061": 13,
    "ds005863": 30,
    "ds006466": 30,
    "ds006480": 30,
}

# ============================================================
# Registry accessors
# ============================================================


def list_experiments() -> list[str]:
    """Return all registered experiment keys (including aliases), sorted."""
    return sorted(EXPERIMENTS)


def get_experiment(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        valid = "\n".join(f"- {k}" for k in sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment {name!r}. Valid options:\n{valid}")
    return EXPERIMENTS[name]


def get_experiment_family(main_experiment_name: str) -> dict[str, ExperimentConfig]:
    """Return the ``main``/``permuted``/``prestim``/``loso`` configs of a family.

    ``main_experiment_name`` must look like ``<prefix>_within_subject_<suffix>``,
    e.g. ``ds003061_within_subject_max30_wide_timebin``.
    """
    marker = "_within_subject_"
    if marker not in main_experiment_name:
        raise ValueError(
            "Unsupported experiment family key. Expected a key containing "
            f"{marker!r}. Got: {main_experiment_name!r}"
        )

    prefix, suffix = main_experiment_name.split(marker, maxsplit=1)
    names = {role: f"{prefix}_{kind}_{suffix}" for role, kind in _FAMILY_ROLE_TO_KIND.items()}

    missing = [name for name in names.values() if name not in EXPERIMENTS]
    if missing:
        raise ValueError(
            "Missing experiment family entries:\n" + "\n".join(f"- {m}" for m in missing)
        )
    return {role: EXPERIMENTS[name] for role, name in names.items()}


# ============================================================
# Default model sets
# ============================================================

DEFAULT_MODEL_NAMES: tuple[ModelName, ...] = ("lda", "logreg", "eegnet")
CLASSICAL_MODEL_NAMES: tuple[ModelName, ...] = ("lda", "logreg")
DEEP_MODEL_NAMES: tuple[ModelName, ...] = ("eegnet",)
