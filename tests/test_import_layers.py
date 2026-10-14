"""Contract test: import-linter catches imports that break the layering of spec §3.1."""

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Literal

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_IMPORTS = Path(sysconfig.get_path("scripts")) / "lint-imports"


@pytest.mark.parametrize(
    ("importer", "imported", "expected"),
    [
        ("goaly.harness", "goaly.bench", "broken"),
        ("goaly.harness", "goaly.rsi", "broken"),
        ("goaly.bench", "goaly.rsi", "broken"),
        ("goaly.bench", "goaly.harness", "kept"),
        ("goaly.rsi", "goaly.bench", "kept"),
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
    probe = tmp_path.joinpath(*importer.split("."), "_layer_probe.py")
    probe.write_text(f"import {imported}\n", encoding="utf-8")

    # The arguments are fixed paths built above, not untrusted input.
    result = subprocess.run(  # noqa: S603
        [LINT_IMPORTS, "--config", REPO_ROOT / "pyproject.toml", "--no-cache"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    output = result.stdout + result.stderr
    if expected == "broken":
        assert result.returncode != 0, output
        # Naming the probe proves that the copy, not the installed package, was analysed.
        assert "_layer_probe" in output, output
    else:
        assert result.returncode == 0, output
