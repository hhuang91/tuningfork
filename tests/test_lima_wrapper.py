"""`scripts/tuningfork` on a Mac: tuningfork inside a Lima VM, from a checkout on the Mac.

There is no Mac here, so a stand-in `uname` says Darwin, and a stand-in `limactl` runs what the
wrapper hands it as Lima would: the program and its arguments, one by one, in the folder
`--workdir` names — the checkout, which Lima shows the VM at the same path. The stand-in `uv`
in the VM's `~/.local/bin` prints what reached it. That Lima itself passes each argument on
intact, through ssh to the VM, only a Mac can show.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_wsl_wrapper import ROOT, UV, WRAPPER

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

UNAME = """#!/usr/bin/env bash
echo Darwin
"""

LIMACTL = """#!/usr/bin/env bash
# limactl as the wrapper uses it: shell --workdir DIR INSTANCE PROGRAM ARGS...
[ "$1" = shell ] && [ "$2" = --workdir ] || { echo "unexpected: $*" >&2; exit 9; }
echo "instance=$4"
cd "$3"
shift 4
exec "$@"
"""


def wrapper(
    tmp_path: Path, *args: str, lima: bool = True, instance: str | None = None,
) -> subprocess.CompletedProcess:
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    stand_ins = [(bin_dir / "uname", UNAME), (home / ".local" / "bin" / "uv", UV)]
    if lima:
        stand_ins.append((bin_dir / "limactl", LIMACTL))
    for path, body in stand_ins:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    # A login shell resets PATH; the VM's own profile puts uv back, as uv's installer does.
    (home / ".profile").write_text('PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(home)}
    if instance:
        env["TUNINGFORK_LIMA_INSTANCE"] = instance
    return subprocess.run(["bash", str(WRAPPER), *args], capture_output=True, text=True, env=env, timeout=60)


def test_a_command_reaches_tuningfork_in_the_vm_argument_by_argument(tmp_path):
    """In the VM named `tuningfork` unless another is, in the checkout as the Mac names it, and
    with no WSL distro asked for."""
    rules = '[{"if": "calls", "value": "write_file"}]'
    done = wrapper(tmp_path, "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab", "$HOME")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[:3] == ["instance=tuningfork", f"cwd={ROOT}", f"venv={tmp_path / 'home'}/.local/share/tuningfork/venv"]
    args = [line.removeprefix("arg=") for line in lines if line.startswith("arg=")]
    assert args == ["run", "tuningfork", "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab", "$HOME"]


def test_it_runs_in_the_vm_named(tmp_path):
    done = wrapper(tmp_path, "version", instance="lab")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "instance=lab"


def test_run_runs_anything_else_in_its_venv(tmp_path):
    done = wrapper(tmp_path, "--run", "pytest", "-q", "tests/test_docs.py")
    assert done.returncode == 0, done.stderr
    args = [line.removeprefix("arg=") for line in done.stdout.splitlines() if line.startswith("arg=")]
    assert args == ["run", "pytest", "-q", "tests/test_docs.py"]


@pytest.mark.skipif(shutil.which("limactl") is not None, reason="Lima is installed here")
def test_without_lima_it_says_how_to_get_it(tmp_path):
    done = wrapper(tmp_path, "version", lima=False)
    assert done.returncode == 127
    assert "brew install lima" in done.stderr and "arg=" not in done.stdout
