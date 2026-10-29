"""A private view of the machine for every launch, with bubblewrap.

What a harness tuningfork launches sees is part of what is measured: the model reads the
paths in its prompt. In folders of its own under the recordings folder, every launch names
a different `.../batches/<id>/<n>/workspace`, and a few digits of that path were seen to
decide whether the model wrote a file or only said it did. So where
bubblewrap can run, every launch sees what an installation would, and the same every time:
its home is `/home/user`, it works in `/home/user/project`, and its harness keeps its files
where it installs them there.

Everything else is the machine as it is, read-only: the person's own home too, since the
harness and its tools live there. What tuningfork keeps is not in it: the recordings folder,
the folder the runs are made in, and the task's spec folder — its check, and any variant
kept beside it — are empty inside. `/tmp` is the launch's own, and so are its processes: when
the launch ends, nothing it started lives on.

`TUNINGFORK_SANDBOX=off` launches without it, as where bubblewrap cannot run: each launch
then sees its own folders.

The clock a launch reads can be pinned too (`TUNINGFORK_CLOCK`, an instant): libfaketime
under `LD_PRELOAD`, so the harness's "today" is the same whichever day a run is made — the
model reads the date in its prompt, and at a knife edge the date decides — and a fork's is
its source's.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from tuningfork.harness.base import LaunchSpec

#: What every launch's harness sees as its home, and as the folder it works in.
HOME = PurePosixPath("/home/user")
WORKSPACE = HOME / "project"
#: Where a check finds the run's final reply: beside the workspace, as a run's is, in no
#: snapshot and among none of the run's files.
REPLY = HOME / "reply.txt"

#: Every launch's own, and empty: nothing of the machine's there is seen inside.
TMP = Path("/tmp")

#: The environment variable that turns the sandbox off.
SWITCH = "TUNINGFORK_SANDBOX"
#: Why no launch runs in the sandbox, where one thing would let them (`remedy`).
TURNED_OFF = f"turned off ({SWITCH}=off)"
NOT_INSTALLED = "bubblewrap is not installed (Linux: the bubblewrap package)"
#: On a Mac, where bubblewrap cannot be installed: tuningfork runs there in a Linux VM.
NOT_LINUX = "bubblewrap runs only on Linux: on a Mac, run tuningfork in a Linux VM (docs/setup.md, On macOS)"
#: Folders every launch sees empty besides tuningfork's own, named by the person as PATH
#: is (`os.pathsep` between them): a benchmark's folder of tasks, say. A check sees them
#: empty too, but for its own task's folder, where its scripts and keys belong.
HIDE = "TUNINGFORK_SANDBOX_HIDE"

#: The instant every launch's clock starts at, "YYYY-MM-DD HH:MM:SS" in the machine's zone,
#: named by the person: libfaketime under `LD_PRELOAD`, which the harness's clock calls go
#: through, so a run's "today" is the same whichever day it is made, and a fork's is its
#: source's. Unset, or "off": the real clock.
CLOCK = "TUNINGFORK_CLOCK"
#: Where libfaketime is, when not at one of the usual places.
CLOCK_LIBRARY = "TUNINGFORK_CLOCK_LIBRARY"
CLOCK_LIBRARIES = (
    "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1",
    "/usr/lib/aarch64-linux-gnu/faketime/libfaketime.so.1",
    "/usr/lib64/faketime/libfaketime.so.1",
    "/usr/lib/faketime/libfaketime.so.1",
    "/usr/local/lib/faketime/libfaketime.so.1",
)
CLOCK_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
#: `for_launches`' default for `clock`: what the person set.
FROM_ENV = "from the environment"

#: A probe gives up after this long: bubblewrap that hangs must not hang a batch's plan.
PROBE_TIMEOUT = 10.0


@dataclass(frozen=True, slots=True)
class Status:
    """Whether launches run in the sandbox here, and if not, why not."""

    executable: str | None
    version: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reason is None

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "kind": "bubblewrap",
            "executable": self.executable,
            "version": self.version,
            "home": str(HOME),
            "workspace": str(WORKSPACE),
            "reason": self.reason,
        }


def hidden_by_the_person() -> list[Path]:
    """The folders `HIDE` names, if any: no launch may read them."""
    given = os.environ.get(HIDE, "")
    return [Path(part.strip()).expanduser() for part in given.split(os.pathsep) if part.strip()]


def pinned_clock() -> tuple[str | None, str | None]:
    """The instant the person pinned every launch's clock at (`CLOCK`), or None — and, if
    what they set is not an instant, why it is not used."""
    given = os.environ.get(CLOCK, "").strip()
    if not given or given.lower() == "off":
        return None, None
    if not CLOCK_FORMAT.match(given):
        return None, f"{CLOCK} is {given!r}, not an instant like 2026-01-05 09:00:00: runs keep the real clock"
    return given, None


def clock_library() -> str | None:
    """libfaketime, where it is here: the path the person named (`CLOCK_LIBRARY`), else the
    usual places."""
    given = os.environ.get(CLOCK_LIBRARY, "").strip()
    for candidate in ([given] if given else CLOCK_LIBRARIES):
        if Path(candidate).is_file():
            return candidate
    return None


def turned_off() -> bool:
    """Whether the person turned the sandbox off: then no plan warns that it is."""
    return os.environ.get(SWITCH, "").strip().lower() in {"off", "0", "no", "false"}


def status() -> Status:
    """Whether a launch can run in the sandbox now: bubblewrap found, able to make the
    namespaces it needs, and the person's own home not where a launch's would go."""
    if turned_off():
        return Status(None, None, TURNED_OFF)
    return _probe(shutil.which("bwrap"), str(Path.home()))


def remedy(reason: str | None) -> str | None:
    """What would let launches run in the sandbox, for the reason `status` gave that they
    cannot: the sandbox turned on again, bubblewrap installed, or, on a Mac, tuningfork run in
    a Linux VM. Else none: bubblewrap is there and could not make a sandbox, or the person's
    home is where a launch's goes, and the reason says which."""
    return {
        TURNED_OFF: f"turn it on again (unset {SWITCH})", NOT_INSTALLED: "install bubblewrap",
        NOT_LINUX: "run tuningfork in a Linux VM",
    }.get(reason or "")


@lru_cache(maxsize=8)
def _probe(executable: str | None, real_home: str) -> Status:
    if os.name != "posix" or executable is None:
        return Status(None, None, NOT_LINUX if sys.platform == "darwin" else NOT_INSTALLED)
    version = None
    try:
        done = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        version = done.stdout.strip().removeprefix("bubblewrap").strip() or None
        tried = subprocess.run(
            [executable, *_base_arguments(Path(real_home)), "--", "true"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return Status(executable, version, f"bubblewrap could not run: {error}")
    if tried.returncode != 0:
        said = (tried.stderr or tried.stdout).strip().splitlines()
        return Status(executable, version, f"bubblewrap could not make a sandbox here: {said[0] if said else f'exit {tried.returncode}'}")
    return Status(executable, version, home_conflict(real_home))


def home_conflict(real_home: str | Path) -> str | None:
    """Why the person's own home rules the sandbox out, if it does: a launch's home is seen
    where it is, and the person's — where the harness and its tools live — would be hidden."""
    home = PurePosixPath(real_home)
    if home == HOME or HOME in home.parents or home in HOME.parents:
        return f"your home, {home}, is where a launch's home would go ({HOME})"
    return None


def _base_arguments(real_home: Path) -> list[str]:
    """The machine as it is, read-only; a fresh /dev, /proc and /tmp; every home hidden but
    the person's own, read-only; the launch's processes its own, and gone with it."""
    arguments = [
        "--die-with-parent", "--unshare-pid",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--tmpfs", "/home",
    ]
    if PurePosixPath("/home") in PurePosixPath(real_home).parents:
        arguments += ["--ro-bind", str(real_home), str(real_home)]
    return arguments


@dataclass(frozen=True, slots=True)
class Sandbox:
    """Launches in bubblewrap, each seeing `HOME` and `WORKSPACE`. `hide`: folders of
    tuningfork's that no launch may read, made empty inside."""

    executable: str
    version: str | None
    hide: tuple[Path, ...] = ()
    #: The instant every launch's clock starts at, and the library that makes it so; None,
    #: the real clock — and if a pin was wanted and is not possible, why (the plans say).
    clock: str | None = None
    clock_library: str | None = None
    clock_problem: str | None = None
    #: Of `hide`, tuningfork's own folders — the recordings folder, the runs' — which no check
    #: is shown either, wherever its task's folder is.
    own: tuple[Path, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """As a launch's settings keep it: a fork of the run is sandboxed the same way, and
        its clock starts where its source's did. The folders hidden are kept too, so a record
        says what its launch could not see, and which of them were tuningfork's own — what a
        trial of its check, or a fork, hides as it did (`recorded_hiding`)."""
        return {
            "kind": "bubblewrap", "version": self.version, "home": str(HOME), "workspace": str(WORKSPACE),
            **({"hidden": [str(folder) for folder in self.hide]} if self.hide else {}),
            **({"own": [str(folder) for folder in self.own]} if self.own else {}),
            **({"clock": self.clock} if self.clock else {}),
        }

    def wrap(self, spec: LaunchSpec, *, home: Path, workspace: Path) -> LaunchSpec:
        """The harness's command, run in the sandbox: `home` seen as `HOME`, `workspace` as
        `WORKSPACE`, and every path to either in its arguments and environment written as
        the harness will see it."""
        if spec.argv:
            refuse_under_tmp(spec.argv[0], spec.env.get("PATH"))
        mounts = [(Path(workspace), WORKSPACE), (Path(home), HOME)]
        env = {key: translate(value, mounts) for key, value in spec.env.items()}
        real_home = Path.home()
        executable = _resolve(spec.argv[0], env.get("PATH")) if spec.argv else None
        hidden = hideable(self.hide, real_home=real_home, keep=[executable] if executable else [])
        if "PATH" in env:
            # tuningfork's own tools are not the harness's: a folder that is empty inside
            # leaves the harness's PATH.
            env["PATH"] = os.pathsep.join(
                entry for entry in env["PATH"].split(os.pathsep)
                if not any(_within(Path(entry), folder) for folder in hidden)
            )
        env.update({"HOME": str(HOME), "TMPDIR": "/tmp", "PWD": str(WORKSPACE)})
        arguments = [self.executable, *_base_arguments(real_home)]
        for folder in hidden:
            arguments += ["--tmpfs", str(folder)]
        arguments += [
            "--bind", str(Path(home).resolve()), str(HOME),
            "--bind", str(Path(workspace).resolve()), str(WORKSPACE),
            *self.clock_arguments(env.get("LD_PRELOAD")),
            "--chdir", str(WORKSPACE), "--",
        ]
        return LaunchSpec(
            argv=(*arguments, *(translate(part, mounts) for part in spec.argv)),
            env=env,
            cwd=Path(workspace),
        )

    def clock_arguments(self, preload: str | None = None) -> list[str]:
        """What bubblewrap is told for the pinned clock, or nothing: the harness's clock calls go
        through libfaketime, loaded before any library `preload` names, started at the instant,
        so its "today" is the same whichever day the launch is made. Monotonic clocks stay real,
        so its timeouts and tuningfork's timings do. Set with --setenv, for the program inside
        alone: in bubblewrap's own environment libfaketime would start in bubblewrap too and make
        its semaphore in the machine's /dev/shm, which the harness, in a /dev of its own, could
        not open — Claude Code's native binary stopped at once on arm64, naming a semaphore it had
        not made — and which a launch killed would leave there. A plan first runs the harness's
        `--version` with these (`ops.connections.clock_problem`)."""
        if not (self.clock and self.clock_library):
            return []
        given = {
            "LD_PRELOAD": ":".join(filter(None, [self.clock_library, preload])),
            "FAKETIME": f"@{self.clock}",
            "FAKETIME_DONT_FAKE_MONOTONIC": "1",
        }
        return [part for name, value in given.items() for part in ("--setenv", name, value)]

    def probe_arguments(self, home: str, *, pinned: bool = False) -> list[str]:
        """What a plan runs a harness's `--version` after, to see whether it runs with the clock
        pinned (`ops.connections.clock_problem`): the machine as it is, writable as it is, but a
        /dev and processes of its own, as a launch has, and, `pinned`, the clock as a launch's
        harness is given it. What libfaketime leaves in /dev/shm goes with it — run outside,
        Claude Code's native binary left a semaphore there on arm64 — and so does every process
        the harness started. Its throwaway `home` is shown where it is, and it starts there: one
        made under /dev, where TMPDIR=/dev/shm puts it, would be hidden."""
        return [
            self.executable, "--die-with-parent", "--unshare-pid",
            "--bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--bind", home, home,
            *(self.clock_arguments() if pinned else []), "--chdir", home, "--",
        ]

    def check_command(
        self, command: str, *, workspace: Path, environment: dict[str, str],
        reply_file: Path | None = None, spec_dir: Path | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """A task's shell check, run as its launch ran, and the environment it runs with: the
        workspace seen as `WORKSPACE`, the reply as `REPLY`, the machine read-only, tuningfork's
        folders and the ones the person hides empty, and a `/tmp` and processes of its own — a
        check can run code the model wrote, and none of its processes outlives it. Two kinds of
        folder show through what is hidden, read-only: the task's own (`spec_dir`, unless
        `shows` says not), where its check scripts are, which a launch never sees; and a Python
        environment on the check's `PATH` that a hidden folder holds — a venv kept in the
        recordings folder, its `python3` — which holds no secret. What is hidden inside a folder
        shown stays hidden: the recordings folder kept in the task's folder. Its clock is the
        machine's, as a check's always was."""
        real_home = Path.home()
        # Every folder hidden from the launch, one inside another too: a launch needs no word
        # for it (`hideable` leaves it out), but a check shown the outer folder does.
        hidden = [
            folder for folder in dict.fromkeys(Path(each).resolve() for each in self.hide)
            if folder.is_dir() and folder != Path("/") and not _within(real_home.resolve(), folder)
        ]
        path = environment.get("PATH", "")
        shown = _environments(path, hidden)
        if spec_dir is not None and self.shows(spec_dir):
            task = Path(spec_dir).resolve()
            shown.insert(0, task)
            hidden = [folder for folder in hidden if folder != task]
        mounts = [(Path(workspace), WORKSPACE), *([(Path(reply_file), REPLY)] if reply_file is not None else [])]
        env = {key: translate(value, mounts) for key, value in environment.items()}
        if "PATH" in env:
            # A folder of programs that is empty inside leaves its PATH, as it leaves a launch's.
            env["PATH"] = os.pathsep.join(
                entry for entry in path.split(os.pathsep)
                if any(_within(Path(entry).resolve(), folder) for folder in shown)
                or not any(_within(Path(entry).resolve(), folder) for folder in hidden)
            )
        env.update({"HOME": str(HOME), "TMPDIR": "/tmp", "PWD": str(WORKSPACE)})
        arguments = [self.executable, *_base_arguments(real_home)]
        # Outer folders before inner ones, a folder hidden before the same one shown: one shown
        # inside a hidden folder shows, and one hidden inside a shown folder stays hidden.
        ordered = sorted(
            [*((folder, False) for folder in hidden), *((folder, True) for folder in shown)],
            key=lambda pair: (len(pair[0].parts), pair[1]),
        )
        for folder, show in ordered:
            arguments += ["--ro-bind-try", str(folder), str(folder)] if show else ["--tmpfs", str(folder)]
        arguments += ["--bind", str(Path(workspace).resolve()), str(WORKSPACE)]
        if reply_file is not None:
            arguments += ["--ro-bind", str(Path(reply_file).resolve()), str(REPLY)]
        arguments += ["--chdir", str(WORKSPACE), "--", "/bin/sh", "-c", command]
        return arguments, env

    def shows(self, task: Path | str) -> bool:
        """Whether a check is shown its task's folder, which every launch is kept from: not
        when it is one of tuningfork's own (a task file kept in the recordings folder itself),
        nor when it would bring back what every launch is kept from (`holds_kept`)."""
        task = Path(task).resolve()
        return task not in {Path(folder).resolve() for folder in self.own} and not holds_kept(task)


def for_launches(
    hide: Iterable[Path | str | None] = (), *, own: Iterable[Path | str | None] = (), clock: str | None = FROM_ENV
) -> tuple[Sandbox | None, str | None]:
    """The sandbox launches run in here, or None and why not. `hide` is what the operation
    keeps from every launch; the folders the person names (`HIDE`) are hidden with them, and
    so are `own`, tuningfork's own folders, which no check is shown either. `clock` is the
    instant every launch starts at: the person's setting (`CLOCK`) unless given — a fork
    gives its source's, None where the source had none. Wanted and not possible, the sandbox
    says why (`clock_problem`) and runs on the real clock."""
    found = status()
    if not found.available or found.executable is None:
        return None, found.reason
    mine = tuple(dict.fromkeys(Path(folder).resolve() for folder in own if folder))
    folders = tuple(dict.fromkeys(
        Path(folder).resolve() for folder in (*mine, *hide, *hidden_by_the_person()) if folder
    ))
    wanted, problem = pinned_clock() if clock == FROM_ENV else (clock, None)
    library = clock_library() if wanted else None
    if wanted and library is None:
        problem = (f"the clock cannot be pinned at {wanted}: libfaketime is not installed "
                   f"(or name it with {CLOCK_LIBRARY}); runs keep the real clock")
        wanted = None
    return Sandbox(
        found.executable, found.version, folders, clock=wanted, clock_library=library, clock_problem=problem,
        own=mine,
    ), None


def mount_point(home: Path) -> Path:
    """Where the workspace is mounted inside a launch's home: made before the launch, so
    nothing is created in the home behind tuningfork's back."""
    return Path(home) / WORKSPACE.relative_to(HOME)


def translate(text: str, mounts: Sequence[tuple[Path, PurePosixPath]]) -> str:
    """`text` with every path to a mounted folder written as the sandbox shows it — the
    folder as given, and as resolved — ending at a path's end, not inside a longer name."""
    for host, inside in mounts:
        for form in dict.fromkeys((str(host), str(Path(host).resolve()))):
            text = re.sub(re.escape(form) + r"(?![\w.-])", str(inside), text)
    return text


def refuse_under_tmp(program: str, path: str | None) -> None:
    """Refused, with the reason, when the harness's program is under `/tmp`, which every
    launch sees empty. Found when a checkout cloned under /tmp failed its sandboxed runs with
    bubblewrap's own words: "execvp …: No such file or directory"."""
    from tuningfork.ops.common import OperationError

    found = shutil.which(program, path=path) if os.sep not in program else program
    for form in dict.fromkeys(Path(each) for each in (found, found and os.path.realpath(found)) if each):
        if _within(form, TMP):
            raise OperationError(
                f"the harness's program, {form}, is under {TMP}, which every sandboxed launch "
                "sees empty",
                hint=f"install the harness, and any Python it runs on, outside {TMP}; or launch "
                f"without the sandbox: {SWITCH}=off",
                code="sandbox_tmp",
            )


def hideable(
    folders: Iterable[Path], *, real_home: Path, keep: Sequence[Path] = ()
) -> list[Path]:
    """Of the folders tuningfork would hide, the ones a launch could otherwise read: not one
    the sandbox hides anyway (under `/tmp`, or another person's home), not one holding the
    person's home or the harness's program (`keep`), and not one inside another hidden."""
    found: list[Path] = []
    real_home = Path(real_home).resolve()
    for folder in sorted({Path(folder).resolve() for folder in folders}, key=lambda path: len(path.parts)):
        if not folder.is_dir() or folder == Path("/"):
            continue
        if _within(folder, TMP) or _within(real_home, folder):
            continue
        if _within(folder, Path("/home")) and not _within(folder, real_home):
            continue
        if any(_within(kept, folder) for kept in keep) or any(_within(folder, done) for done in found):
            continue
        found.append(folder)
    return found


def recorded_hiding(recorded: dict[str, Any] | None, *, task: Path | None) -> tuple[list[Path], list[Path]]:
    """What a launch's recorded sandbox (`Sandbox.to_json`) hid, and which of it was
    tuningfork's own: a trial of its check, or a fork, hides it as the launch did. A launch
    recorded before its own folders were has them guessed: all it hid but its task's folder
    and the folders the person hides now."""
    recorded = recorded or {}
    hidden = [Path(folder) for folder in recorded.get("hidden") or []]
    if "own" in recorded:
        return hidden, [Path(folder) for folder in recorded.get("own") or []]
    theirs = {folder.resolve() for folder in hidden_by_the_person()}
    return hidden, [folder for folder in hidden if folder != task and folder.resolve() not in theirs]


def holds_kept(folder: Path | str) -> bool:
    """Whether `folder` holds what the sandbox keeps from every launch, the machine's `/tmp`
    or other people's homes: a task's folder like that is never shown to its check, since
    shown whole it would bring them back — other programs' files and sockets among them."""
    folder = Path(folder).resolve()
    return any(_within(kept, folder) for kept in (TMP, Path("/home")))


def _environments(path: str, hidden: Sequence[Path]) -> list[Path]:
    """The Python environments on `path` that a hidden folder holds, inside it and never the
    hidden folder itself: a folder of programs whose parent keeps a `pyvenv.cfg`, as a venv's
    `bin` does."""
    found: list[Path] = []
    for entry in filter(None, path.split(os.pathsep)):
        folder = Path(entry).resolve().parent
        if (folder / "pyvenv.cfg").is_file() and any(folder != each and _within(folder, each) for each in hidden):
            found.append(folder)
    return list(dict.fromkeys(found))


def _within(path: Path, folder: Path) -> bool:
    """`path` is `folder` or inside it."""
    try:
        Path(path).relative_to(folder)
    except ValueError:
        return False
    return True


def _resolve(program: str, path: str | None) -> Path | None:
    """The file a program name runs, its links followed."""
    found = shutil.which(program, path=path) if os.sep not in program else program
    return Path(found).resolve() if found else None
