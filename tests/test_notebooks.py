"""Static checks on the analysis notebooks (no data or execution required)."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO / "notebooks").rglob("*.ipynb"))
FORBIDDEN_IMPORTS = {
    "config",
    "bss_utils",
    "loader",
    "preprocessing",
    "utils",
    "models",
    "visualization",
}


def _code(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _parse(src: str) -> ast.Module:
    no_magics = "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line for line in src.split("\n")
    )
    return ast.parse(no_magics)


def test_notebooks_found():
    assert len(NOTEBOOKS) >= 30


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_clean(nb_path):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "outputs must be stripped (run nbstripout)"
            assert cell["execution_count"] is None
    assert not nb_path.name.endswith((".py.ipynb", ".ipynb.ipynb"))


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_parses_and_imports_resolve(nb_path):
    cells = _code(nb_path)
    assert "use_project_root()" in cells[0], "first code cell must anchor the project root"
    for src in cells:
        tree = _parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {a.name for a in node.names} & FORBIDDEN_IMPORTS
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in FORBIDDEN_IMPORTS, f"legacy import: from {node.module}"
                if node.module.startswith("oddball_eeg"):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(module, alias.name), (
                            f"{node.module}.{alias.name} does not exist"
                        )
        joined = ast.unparse(tree)
        assert "sys.path" not in joined
        assert "importlib.reload" not in joined
