import pkgutil
import subprocess
import sys

import pytest

import sparjax

MODULES = sorted(m.name for m in pkgutil.walk_packages(sparjax.__path__, prefix="sparjax."))


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_fresh_interpreter(module: str) -> None:
    """Each module must import on its own (catches circular imports hidden by test order)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_console_script_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "sparjax.cli", "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
