from __future__ import annotations

from pathlib import Path

from oddball_eeg import paths

REPO = Path(__file__).resolve().parents[1]


def test_project_root_found_from_subdirectory(monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.chdir(REPO / "notebooks")
    assert paths.project_root() == REPO


def test_use_project_root_changes_directory(monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.chdir(REPO / "src")
    assert paths.use_project_root(verbose=False) == REPO
    assert Path.cwd() == REPO


def test_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_ROOT, str(tmp_path))
    monkeypatch.delenv(paths.ENV_RESULTS, raising=False)
    monkeypatch.delenv(paths.ENV_DATA, raising=False)
    assert paths.project_root() == tmp_path
    assert paths.results_dir() == tmp_path / "results"
    assert paths.data_dir() == tmp_path / "datasets" / "openneuro"
