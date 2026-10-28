"""`scripts/tuningfork` on a Mac: tuningfork inside a Lima VM, from a checkout on the Mac.

There is no Mac here, so a stand-in `uname` says Darwin, a stand-in `limactl` says what Lima
knows of the VM (whether it runs, its ssh config, the folders it shows the VM), and a stand-in
`ssh` runs the command the wrapper hands the VM as sshd would: in a session of its own, from an
environment of its own. The stand-in `uv` in the VM's `~/.local/bin` prints what reached it.
That ssh passes each argument on intact, and that a wrapper killed takes tuningfork in the VM
with it, only a Mac can show.
"""

from __future__ import annotations

import os
import pty
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.test_wsl_wrapper import ROOT, UV, WRAPPER

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash and python3",
)

UNAME = """#!/usr/bin/env bash
echo Darwin
"""

LIMACTL = """#!/usr/bin/env bash
# limactl as the wrapper uses it: list INSTANCE --format F, the VM's state and ssh config, then
# the folder it shows the VM, at its own path.
[ "$1" = list ] || { echo "unexpected: $*" >&2; exit 9; }
[ "$2" = missing ] && exit 1
printf '%s\\t%s\\n%s\\t%s\\n' "${LIMA_STATUS:-Running}" /lima/ssh.config "$LIMA_SHOWS" "$LIMA_SHOWS"
"""

SSH = """#!/usr/bin/env bash
# ssh as the wrapper uses it: -t or -T, -F CONFIG and -o options, the VM's host, the command,
# run as sshd runs it: in a session of its own, from an environment of its own.
echo "through ssh $1" >&2
shift
while [ "$1" = -F ] || [ "$1" = -o ]; do shift 2; done
echo "instance=${1#lima-}"
exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \\
    env -i HOME="$HOME" PATH="$PATH" bash -c "$2"
"""


def stand_ins(tmp_path: Path, *, lima: bool = True, uv: str = UV) -> dict[str, str]:
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    made = [(bin_dir / "uname", UNAME), (bin_dir / "ssh", SSH), (home / ".local" / "bin" / "uv", uv)]
    if lima:
        made.append((bin_dir / "limactl", LIMACTL))
    for path, body in made:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    # A login shell resets PATH; the VM's own profile puts uv back, as uv's installer does.
    (home / ".profile").write_text('PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(home),
        "LIMA_SHOWS": str(ROOT.parent),
    }


def wrapper(
    tmp_path: Path, *args: str, lima: bool = True, terminal: bool = False, uv: str = UV,
    extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = {**stand_ins(tmp_path, lima=lima, uv=uv), **(extra or {})}
    if not terminal:
        return subprocess.run(
            ["bash", str(WRAPPER), *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            env=env, timeout=60,
        )
    # A terminal, both ways: the VM gets one only then.
    leader, follower = pty.openpty()
    try:
        done = subprocess.run(
            ["bash", str(WRAPPER), *args], stdin=follower, stdout=follower, stderr=subprocess.PIPE, text=True,
            env=env, timeout=60,
        )
        os.close(follower)
        follower = -1
        out = b""
        while True:
            try:
                chunk = os.read(leader, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
    finally:
        os.close(leader)
        if follower >= 0:
            os.close(follower)
    return subprocess.CompletedProcess(done.args, done.returncode, out.decode().replace("\r\n", "\n"), done.stderr)


@pytest.mark.parametrize("terminal", [False, True], ids=["no terminal", "terminal"])
def test_a_command_reaches_tuningfork_in_the_vm_argument_by_argument(tmp_path, terminal):
    """In the VM named `tuningfork` unless another is, in the checkout as the Mac names it, and
    with no WSL distro asked for; with a terminal in the VM only where there is one here."""
    rules = '[{"if": "calls", "value": "write_file"}]'
    done = wrapper(
        tmp_path, "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab", "$HOME", "it's",
        "", terminal=terminal,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr.strip() == ("through ssh -t" if terminal else "through ssh -T")
    lines = done.stdout.splitlines()
    assert lines[:3] == ["instance=tuningfork", f"cwd={ROOT}", f"venv={tmp_path / 'home'}/.local/share/tuningfork/venv"]
    args = [line.removeprefix("arg=") for line in lines if line.startswith("arg=")]
    assert args == [
        "run", "tuningfork", "refine", "start", "run1", "2", "--target-rules", rules, "--home", "/tmp/a lab", "$HOME",
        "it's", "",
    ]


def test_it_runs_in_the_vm_named(tmp_path):
    done = wrapper(tmp_path, "version", extra={"TUNINGFORK_LIMA_INSTANCE": "lab"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == "instance=lab"


def test_run_runs_anything_else_in_its_venv(tmp_path):
    done = wrapper(tmp_path, "--run", "pytest", "-q", "tests/test_docs.py")
    assert done.returncode == 0, done.stderr
    args = [line.removeprefix("arg=") for line in done.stdout.splitlines() if line.startswith("arg=")]
    assert args == ["run", "pytest", "-q", "tests/test_docs.py"]


def test_only_tuningforks_own_settings_cross(tmp_path):
    """A TUNINGFORK_ setting made on the Mac reaches the VM as written; nothing else does, nor the
    name of the VM, which is the Mac's own setting, nor a key, which on its way would be on ssh's
    command line for any account on the Mac to read: the wrapper says it stayed."""
    probe = (
        '#!/usr/bin/env bash\nprintf "clock=%s\\\\nother=%s\\\\ninstance=%s\\\\nkey=%s\\\\n" "$TUNINGFORK_CLOCK" '
        '"${OTHER-unset}" "${TUNINGFORK_LIMA_INSTANCE-unset}" "${TUNINGFORK_PROPOSER_KEY-unset}"\n'
    )
    done = wrapper(
        tmp_path, "version", uv=probe,
        extra={
            "TUNINGFORK_CLOCK": '2026-10-01 09:00:00 "$x"', "OTHER": "1", "TUNINGFORK_LIMA_INSTANCE": "tuningfork",
            "TUNINGFORK_PROPOSER_KEY": "not-a-real-key",
        },
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[1:] == [
        'clock=2026-10-01 09:00:00 "$x"', "other=unset", "instance=unset", "key=unset",
    ]
    assert "TUNINGFORK_PROPOSER_KEY is not passed to the VM" in done.stderr
    assert "not-a-real-key" not in done.stderr


@pytest.mark.parametrize("terminal", [False, True], ids=["no terminal", "terminal"])
@pytest.mark.parametrize(("ends", "status"), [("exit 7", 7), ("kill -TERM $$", 143)], ids=["exit", "killed"])
def test_its_exit_status_is_tuningforks(tmp_path, terminal, ends, status):
    done = wrapper(tmp_path, "version", terminal=terminal, uv=f"#!/usr/bin/env bash\n{ends}\n")
    assert done.returncode == status, done.stderr


def test_once_nothing_reads_it_tuningfork_is_interrupted(tmp_path):
    """The Mac side gone, its ssh with it: in the VM the watcher sees nothing read what tuningfork
    writes, and interrupts it, as Ctrl-C would."""
    started, interrupted = tmp_path / "started", tmp_path / "interrupted"
    # Python, not a shell's trap: a shell cannot trap a SIGINT ignored when it started, as it is
    # under a test runner started in the background.
    uv = "\n".join([
        "#!/usr/bin/env python3",
        "import signal, sys, time",
        "def stop(number, frame):",
        f"    open({str(interrupted)!r}, 'w').close()",
        "    sys.exit(130)",
        "signal.signal(signal.SIGINT, stop)",
        f"open({str(started)!r}, 'w').close()",
        "time.sleep(60)",
        "",
    ])
    env = stand_ins(tmp_path, uv=uv)
    running = subprocess.Popen(
        ["bash", str(WRAPPER), "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env,
    )
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            assert time.monotonic() < deadline and running.poll() is None, "it never started"
            time.sleep(0.05)
        running.stdout.close()
        assert running.wait(timeout=30) == 130
        assert interrupted.exists()
    finally:
        # The stand-in sshd's session is the watcher's: everything it started goes with it.
        try:
            os.killpg(running.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        running.wait()


def test_a_checkout_the_vm_cannot_see_is_refused(tmp_path):
    done = wrapper(tmp_path, "version", extra={"LIMA_SHOWS": str(tmp_path / "elsewhere")})
    assert done.returncode == 2
    assert "does not see this checkout" in done.stderr and "arg=" not in done.stdout


def test_a_vm_lima_does_not_have_is_named(tmp_path):
    done = wrapper(tmp_path, "version", extra={"TUNINGFORK_LIMA_INSTANCE": "missing"})
    assert done.returncode == 2
    assert "no Lima VM named missing" in done.stderr and "arg=" not in done.stdout


def test_a_stopped_vm_says_how_to_start_it(tmp_path):
    done = wrapper(tmp_path, "version", extra={"LIMA_STATUS": "Stopped"})
    assert done.returncode == 2
    assert "not running (Stopped)" in done.stderr and "limactl start tuningfork" in done.stderr
    assert "arg=" not in done.stdout


@pytest.mark.skipif(shutil.which("limactl") is not None, reason="Lima is installed here")
def test_without_lima_it_says_how_to_get_it(tmp_path):
    done = wrapper(tmp_path, "version", lima=False)
    assert done.returncode == 127
    assert "brew install lima" in done.stderr and "arg=" not in done.stdout
