"""Command-line entry points.

``oddball-eeg-check-config``
    Verify the final journal experiment registry without loading EEG data.

``python -m oddball_eeg.cli download [--dataset ds003061 ...] [--tag ...]``
    Download the OpenNeuro datasets into the expected layout (requires the
    ``data`` extra: ``pip install -e ".[data]"``).
"""

from __future__ import annotations

import argparse
import sys

from oddball_eeg import paths
from oddball_eeg.config import (
    JOURNAL_EXPECTED_SUBJECT_COUNTS,
    JOURNAL_LOSO_EXPERIMENTS,
    JOURNAL_WITHIN_SUBJECT_EXPERIMENTS,
    get_experiment,
)

DATASET_IDS: tuple[str, ...] = tuple(JOURNAL_EXPECTED_SUBJECT_COUNTS)


def _check_family(keys: tuple[str, ...], label: str) -> None:
    print(f"\n{label}")
    for key in keys:
        cfg = get_experiment(key)
        dataset = cfg.dataset.dataset
        n_subjects = len(cfg.subjects.subjects)
        expected = JOURNAL_EXPECTED_SUBJECT_COUNTS[dataset]
        if n_subjects != expected:
            raise AssertionError(
                f"{key}: expected {expected} subjects for {dataset}, got {n_subjects}"
            )
        print(
            f"- {key}: dataset={dataset}, subjects={n_subjects}, "
            f"crop={cfg.effective_crop_window()}, feature={cfg.feature.mode}"
        )


def check_journal_config() -> int:
    """Check that the final experiment keys resolve with the expected subject counts."""
    _check_family(JOURNAL_WITHIN_SUBJECT_EXPERIMENTS, "Within-subject journal experiments")
    _check_family(JOURNAL_LOSO_EXPERIMENTS, "LOSO journal experiments")
    total = sum(JOURNAL_EXPECTED_SUBJECT_COUNTS.values())
    print(f"\nJournal experiment registry check passed ({total} subject-recordings).")
    return 0


def download(dataset_ids: list[str], tag: str | None = None) -> int:
    """Download OpenNeuro datasets into :func:`oddball_eeg.paths.data_dir`."""
    try:
        import openneuro
    except ImportError:
        print('openneuro-py is not installed. Run: pip install -e ".[data]"', file=sys.stderr)
        return 1

    target_root = paths.data_dir()
    for dataset_id in dataset_ids:
        target = target_root / dataset_id
        print(f"Downloading {dataset_id}{f' (tag {tag})' if tag else ''} -> {target}")
        kwargs = {"dataset": dataset_id, "target_dir": target}
        if tag:
            kwargs["tag"] = tag
        openneuro.download(**kwargs)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oddball_eeg")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-config", help="verify the journal experiment registry")

    p_dl = sub.add_parser("download", help="download the OpenNeuro datasets")
    p_dl.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_IDS,
        help="dataset to download (repeatable; default: all four)",
    )
    p_dl.add_argument("--tag", default=None, help="OpenNeuro snapshot tag (single dataset only)")

    args = parser.parse_args(argv)
    if args.command == "check-config":
        return check_journal_config()
    if args.command == "download":
        datasets = args.dataset or list(DATASET_IDS)
        if args.tag and len(datasets) != 1:
            parser.error("--tag can only be used together with a single --dataset")
        return download(datasets, tag=args.tag)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
