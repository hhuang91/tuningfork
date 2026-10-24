"""`scripts/tuningfork`: tuningfork inside WSL from a checkout on Windows, from Git Bash.

There is no WSL here, so a stand-in `wsl.exe` runs what the wrapper hands it, as WSL would —
the program and its arguments, one by one, in the folder `--cd` names — and a stand-in `uv`
in the distro's `~/.local/bin` prints what reached it. Its twin, `scripts/tuningfork.ps1`,
needs Windows to be tried.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "tuningfork"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

WSL = """#!/usr/bin/env bash
# wsl.exe as the wrapper uses it: --list, or --distribution D --cd DIR --exec PROGRAM ARGS...
if [ "$1" = --list ]; then printf '  NAME      STATE      VERSION\\n* Distro-A  Running    2\\n'; exit 0; fi
[ "$1" = --distribution ] && [ "$3" = --cd ] && [ "$5" = --exec ] || { echo "unexpected: $*" >&2; exit 9; }
echo "distro=$2"
cd "$4"
shift 5
exec "$@"
"""

UV = """#!/usr/bin/env bash
echo "cwd=$PWD"
echo "venv=$UV_PROJECT_ENVIRONMENT"
for arg in "$@"; do echo "arg=$arg"; done
"""


def wrapper(tmp_path: Path, *args: str, distro: str | None = "Distro-A") -> subprocess.CompletedProcess:
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    for path, body in ((bin_dir / "wsl.exe", WSL), (home / ".local" / "bin" / "uv", UV)):
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    # A login shell resets PATH; the distro's own profile puts uv back, as uv's installer does.
    (home / ".profile").write_text('PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(home)}
    if distro:
        env["TUNINGFORK_WSL_DISTRO"] = distro
    return subprocess.run(["bash", str(WRAPPER), *args], capture_output=True, text=True, env=env, timeout=60)


def test_it_runs_nothing_until_a_distro_is_named(tmp_path):
    done = wrapper(tmp_path, "version", distro=None)
    assert done.returncode == 2
    assert "TUNINGFORK_WSL_DISTRO" in done.stderr and "Distro-A" in done.stderr  # the ones installed
    assert "arg=" not in done.stdout


def test_a_command_reaches_tuningfork_argument_by_argument(tmp_path):
    rules = '[{"if": "calls", "value": "write_file"}]'
    done = wrapper(tmp_path, "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[:3] == ["distro=Distro-A", f"cwd={ROOT}", f"venv={tmp_path / 'home'}/.local/share/tuningfork/venv"]
    args = [line.removeprefix("arg=") for line in lines if line.startswith("arg=")]
    assert args == ["run", "tuningfork", "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab"]


def test_run_runs_anything_else_in_its_venv(tmp_path):
    done = wrapper(tmp_path, "--run", "pytest", "-q", "tests/test_docs.py")
    assert done.returncode == 0, done.stderr
    args = [line.removeprefix("arg=") for line in done.stdout.splitlines() if line.startswith("arg=")]
    assert args == ["run", "pytest", "-q", "tests/test_docs.py"]
