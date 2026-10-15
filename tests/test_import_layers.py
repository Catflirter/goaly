"""Contract test: import-linter catches imports that break the layering of spec §3.1."""

import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Literal

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_IMPORTS = Path(sysconfig.get_path("scripts")) / "lint-imports"
# The summary line of the locked import-linter, e.g. "Analyzed 8 files, 1 dependencies."
ANALYZED = re.compile(r"Analyzed (\d+) files, (\d+) dependencies")


def lint_imports(root: Path) -> subprocess.CompletedProcess[str]:
    """Run lint-imports on the package copy under ``root``, first on the import path."""
    # The arguments are fixed paths built here, not untrusted input.
    return subprocess.run(  # noqa: S603
        [LINT_IMPORTS, "--config", REPO_ROOT / "pyproject.toml", "--no-cache"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def analyzed(result: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    """Return how many files and imports lint-imports analysed."""
    match = ANALYZED.search(result.stdout)
    assert match, result.stdout + result.stderr
    return int(match[1]), int(match[2])


@pytest.mark.parametrize(
    ("importer", "imported", "expected"),
    [
        ("goaly.harness", "goaly.bench", "broken"),
        ("goaly.harness", "goaly.rsi", "broken"),
        ("goaly.bench", "goaly.rsi", "broken"),
        ("goaly.bench", "goaly.harness", "kept"),
        ("goaly.rsi", "goaly.bench", "kept"),
        ("goaly.contracts", "goaly.harness", "broken"),
        ("goaly.harness", "goaly.contracts", "kept"),
        # Only goaly.rsi may import mini-swe-agent (spec §3.1).
        ("goaly.contracts", "minisweagent", "broken"),
        ("goaly.harness", "minisweagent", "broken"),
        ("goaly.bench", "minisweagent", "broken"),
        ("goaly.rsi", "minisweagent", "kept"),
    ],
)
def test_layers_contract(
    tmp_path: Path, importer: str, imported: str, expected: Literal["broken", "kept"]
) -> None:
    # Analyse a copy of the package that has one extra import, never the installed package.
    shutil.copytree(
        REPO_ROOT / "src" / "goaly",
        tmp_path / "goaly",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    clean = lint_imports(tmp_path)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    files, imports = analyzed(clean)
    probe = tmp_path.joinpath(*importer.split("."), "_layer_probe.py")
    probe.write_text(f"import {imported}\n", encoding="utf-8")

    result = lint_imports(tmp_path)

    output = result.stdout + result.stderr
    if expected == "broken":
        assert result.returncode != 0, output
        # Naming the probe proves that the copy, not the installed package, was analysed.
        assert "_layer_probe" in output, output
    else:
        assert result.returncode == 0, output
        # One more file and one more import prove that the probe's import was analysed. A
        # third-party package also enters the graph as a module, unless the copy imported it.
        new_files, new_imports = analyzed(result)
        assert new_imports == imports + 1, output
        assert new_files - files in ((1,) if imported.startswith("goaly.") else (1, 2)), output
