"""The sandbox: every launch sees the same home and folder (`tuningfork.sandbox`).

What a launch sees is what the model reads: in folders of its own, each run's prompt named a
different path, and a few digits of it decided whether the model wrote a file or only said it
did. The first tests build the sandbox's command without running it; the
rest run bubblewrap itself, and are skipped where it cannot run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import tuningfork.sandbox
from tuningfork.harness.base import LaunchSpec
from tuningfork.ops import OperationError
from tuningfork.runner import BatchRunner, layout_setting
from tuningfork.sandbox import (
    HOME,
    SWITCH,
    TMP,
    WORKSPACE,
    Sandbox,
    for_launches,
    hideable,
    home_conflict,
    recorded_hiding,
    status,
    translate,
)
from tests.helpers import FakeAgent, task_stack

#: A folder of this checkout, for the tests that run bubblewrap: outside /tmp there.
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sandboxed(monkeypatch):
    """The sandbox on, as it is by default — skipped where bubblewrap cannot run, and where
    this checkout or its Python is under /tmp, which every launch sees empty."""
    monkeypatch.delenv(SWITCH, raising=False)
    found = status()
    if not found.available:
        pytest.skip(f"no sandbox here: {found.reason}")
    for needed in (Path(sys.executable), Path(__file__)):
        if TMP in needed.parents:
            pytest.skip(f"{needed} is under {TMP}, which every sandboxed launch sees empty")
    return found


@pytest.fixture
def elsewhere(monkeypatch):
    """The sandbox's /tmp moved where nothing is, so folders a test makes — under pytest's
    own temporary folder, often in /tmp — count as the machine's, wherever the suite runs."""
    monkeypatch.setattr(tuningfork.sandbox, "TMP", Path("/nowhere-tmp"))


def folders(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "recordings" / "batches" / "ab12cd34" / "0"
    (root / "home").mkdir(parents=True)
    (root / "workspace").mkdir()
    return root / "home", root / "workspace"


def bubblewrap_stand_in(tmp_path: Path) -> str:
    """bubblewrap as the tests of the clock's probe need it, wherever the suite runs: what
    follows `--` run as it is, given what each `--setenv` sets."""
    program = tmp_path / "bwrap"
    program.write_text(
        '#!/bin/sh\nwhile [ "$1" != "--" ]; do\n'
        '    if [ "$1" = "--setenv" ]; then export "$2=$3"; shift 3; else shift; fi\n'
        'done\nshift\nexec "$@"\n',
        encoding="utf-8")
    program.chmod(0o755)
    return str(program)


def given_by_setenv(argv: Sequence[str]) -> dict[str, str]:
    """What bubblewrap's command sets for the program it runs (`--setenv NAME VALUE`)."""
    options = list(argv[:list(argv).index("--")])
    return {options[at + 1]: options[at + 2] for at, part in enumerate(options) if part == "--setenv"}


#: A check that says what it finds: run in a workspace, given the folder it should find empty
#: and the person's home.
LOOK = """\
import json, os, sys
hidden, real_home = sys.argv[1], sys.argv[2]
print(json.dumps({
    "cwd": os.getcwd(), "home": os.environ["HOME"], "files": sorted(os.listdir(".")),
    "reply": open(os.environ["TUNINGFORK_REPLY"]).read(), "key": os.environ.get("CLOUD_API_KEY"),
    "real_home_writable": os.access(real_home, os.W_OK), "hidden": sorted(os.listdir(hidden)),
    "task": sorted(os.listdir(os.environ["TUNINGFORK_SPEC_DIR"])),
}))
"""


class TestTheCommand:
    def test_paths_to_a_launchs_folders_are_written_as_it_sees_them(self):
        mounts = [(Path("/data/rec/batches/ab12cd34/0/workspace"), WORKSPACE),
                  (Path("/data/rec/batches/ab12cd34/0/home"), HOME)]
        assert translate("/data/rec/batches/ab12cd34/0/home/.hermes", mounts) == "/home/user/.hermes"
        assert translate("cd /data/rec/batches/ab12cd34/0/workspace && ls", mounts) == "cd /home/user/project && ls"
        # A longer name is another folder.
        assert translate("/data/rec/batches/ab12cd34/0/homework", mounts) == "/data/rec/batches/ab12cd34/0/homework"

    def test_the_harness_runs_in_bubblewrap_with_its_folders_seen_as_home_and_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/tester")))
        home, workspace = folders(tmp_path)
        spec = LaunchSpec(
            argv=("hermes", "chat", "--settings", json.dumps({"exclude": f"{workspace}/CLAUDE.md"})),
            env={"HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "PATH": "/usr/bin"},
            cwd=workspace,
        )
        wrapped = Sandbox("/usr/bin/bwrap", "0.11.1").wrap(spec, home=home, workspace=workspace)

        argv = list(wrapped.argv)
        assert argv[0] == "/usr/bin/bwrap"
        assert {"--die-with-parent", "--unshare-pid"} <= set(argv)
        assert argv[argv.index("--ro-bind"):argv.index("--ro-bind") + 3] == ["--ro-bind", "/", "/"]
        # Every home hidden but the person's own, read-only: the harness and its tools live there.
        assert ["--tmpfs", "/home", "--ro-bind", "/home/tester", "/home/tester"] == argv[argv.index("/home") - 1:argv.index("/home") + 4]
        bound = argv.index("--bind")
        assert argv[bound:bound + 6] == ["--bind", str(home.resolve()), "/home/user", "--bind", str(workspace.resolve()), "/home/user/project"]
        command = argv[argv.index("--") + 1:]
        assert command[:3] == ["hermes", "chat", "--settings"]
        assert json.loads(command[3]) == {"exclude": "/home/user/project/CLAUDE.md"}
        assert argv[argv.index("--chdir") + 1] == "/home/user/project"
        assert wrapped.env["HOME"] == "/home/user" and wrapped.env["HERMES_HOME"] == "/home/user/.hermes"
        assert wrapped.env["TMPDIR"] == "/tmp" and wrapped.env["PWD"] == "/home/user/project"

    def test_only_folders_a_launch_could_read_are_hidden(self, tmp_path, monkeypatch):
        person = tmp_path / "person"
        checkout = person / "tuningfork"  # the checkout sits in the person's home, as a clone does
        fixtures, temporary = checkout / "tests" / "fixtures", tmp_path / "tmp"
        for folder in (fixtures / "formats", temporary):
            folder.mkdir(parents=True)
        monkeypatch.setattr(tuningfork.sandbox, "TMP", temporary)
        found = hideable([checkout, fixtures, temporary], real_home=person)
        assert found == [checkout]  # inside another hidden, and hidden anyway, are left out
        # Never the folder that holds the harness's program, nor one holding the home itself.
        assert hideable([checkout], real_home=person, keep=[fixtures / "formats"]) == []
        assert hideable([person], real_home=person) == []

    def test_folders_the_person_names_are_hidden_from_every_launch(self, tmp_path, monkeypatch):
        from tuningfork.sandbox import HIDE, Status, for_launches

        keys, more, home = tmp_path / "keys", tmp_path / "more", tmp_path / "home"
        for folder in (keys, more, home):
            folder.mkdir()
        monkeypatch.setattr(tuningfork.sandbox, "status", lambda: Status("/usr/bin/bwrap", "0.11.1"))
        monkeypatch.setenv(HIDE, f"{keys}{os.pathsep} {more} {os.pathsep}")
        sandbox, why = for_launches([home])
        assert why is None and sandbox.hide == (home.resolve(), keys.resolve(), more.resolve())
        assert sandbox.to_json()["hidden"] == [str(folder) for folder in sandbox.hide]
        monkeypatch.delenv(HIDE)
        plain, _ = for_launches([home])
        assert plain.hide == (home.resolve(),)

    def test_the_hidden_folders_leave_the_harnesss_path(self, tmp_path, monkeypatch, elsewhere):
        person = tmp_path / "person"
        checkout = person / "tuningfork"
        (checkout / "bin").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: person))
        home, workspace = folders(tmp_path)
        spec = LaunchSpec(argv=("/usr/bin/true",), env={"PATH": f"{checkout}/bin:/usr/bin"}, cwd=workspace)
        wrapped = Sandbox("/usr/bin/bwrap", None, hide=(checkout,)).wrap(spec, home=home, workspace=workspace)
        assert wrapped.env["PATH"] == "/usr/bin"
        assert ["--tmpfs", str(checkout)] == list(wrapped.argv[wrapped.argv.index(str(checkout)) - 1:][:2])

    def test_a_harness_under_tmp_is_refused_with_the_reason(self, tmp_path, monkeypatch):
        """Found when a checkout cloned under /tmp failed every sandboxed run with bubblewrap's
        "execvp …: No such file or directory", which named no cause."""
        monkeypatch.setattr(tuningfork.sandbox, "TMP", tmp_path)
        home, workspace = folders(tmp_path)
        venv = tmp_path / "clone" / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        (venv / "python").chmod(0o755)
        for argv0, path in ((str(venv / "python"), "/usr/bin"), ("python", f"{venv}:/usr/bin")):
            spec = LaunchSpec(argv=(argv0, "agent.py"), env={"PATH": path}, cwd=workspace)
            with pytest.raises(OperationError) as refused:
                Sandbox("/usr/bin/bwrap", None).wrap(spec, home=home, workspace=workspace)
            assert refused.value.code == "sandbox_tmp" and str(venv / "python") in str(refused.value)
            assert "TUNINGFORK_SANDBOX=off" in refused.value.hint

    def test_a_person_whose_home_is_where_a_launchs_goes_runs_without_it(self):
        assert home_conflict("/home/user") and home_conflict("/home/user/x") and home_conflict("/home")
        assert home_conflict("/home/someone") is None

    def test_turned_off_it_says_so(self, monkeypatch):
        monkeypatch.setenv(SWITCH, "off")
        assert not status().available and SWITCH in status().reason
        sandbox, why = for_launches([])
        assert sandbox is None and SWITCH in why

    def test_on_a_mac_it_says_where_tuningfork_runs_instead(self, monkeypatch):
        """bubblewrap cannot be installed on a Mac, and "install bubblewrap" would send a person
        looking for it: tuningfork runs there in a Linux VM."""
        from tuningfork.sandbox import NOT_INSTALLED, NOT_LINUX, remedy

        monkeypatch.delenv(SWITCH, raising=False)
        monkeypatch.setattr(tuningfork.sandbox.shutil, "which", lambda name: None)
        found = {}
        for platform in ("darwin", "linux"):
            monkeypatch.setattr(sys, "platform", platform)
            tuningfork.sandbox._probe.cache_clear()
            found[platform] = status().reason
        tuningfork.sandbox._probe.cache_clear()
        assert found == {"darwin": NOT_LINUX, "linux": NOT_INSTALLED}
        assert "Linux VM" in NOT_LINUX and remedy(NOT_LINUX) == "run tuningfork in a Linux VM"
        assert remedy(NOT_INSTALLED) == "install bubblewrap"

    def test_a_launch_keeps_the_sandbox_it_ran_in(self):
        kept = layout_setting(FakeAgent(), Sandbox("/usr/bin/bwrap", "0.11.1"))
        assert kept["sandbox"] == {"kind": "bubblewrap", "version": "0.11.1", "home": "/home/user", "workspace": "/home/user/project"}
        assert "sandbox" not in layout_setting(FakeAgent())

    def test_a_launch_keeps_what_it_hid_and_which_of_it_was_tuningforks_own(self, tmp_path, monkeypatch):
        """Read back, a trial of its check and a fork hide it as it did; a launch kept before its
        own folders were has them guessed: all it hid but its task's folder and the person's."""
        monkeypatch.delenv("TUNINGFORK_SANDBOX_HIDE", raising=False)
        recordings, task = tmp_path / "recordings", tmp_path / "task"
        kept = Sandbox("/usr/bin/bwrap", "0.11.1", hide=(recordings, task), own=(recordings,)).to_json()
        assert kept["hidden"] == [str(recordings), str(task)] and kept["own"] == [str(recordings)]
        assert recorded_hiding(kept, task=task) == ([recordings, task], [recordings])
        del kept["own"]
        assert recorded_hiding(kept, task=task) == ([recordings, task], [recordings])
        assert recorded_hiding(None, task=None) == ([], [])


class TestTheCheck:
    """A run's check runs in the sandbox its launch ran in (`Sandbox.check_command`): it can run
    what the model left in the workspace."""

    def command(self, tmp_path: Path, *, hide=(), spec_dir=None, path="/usr/bin:/bin"):
        workspace = tmp_path / "runs" / "0" / "workspace"
        workspace.mkdir(parents=True)
        reply = workspace.parent / "reply.txt"
        reply.write_text("The report is written.")
        environment = {"PATH": path, "TUNINGFORK_REPLY": str(reply)}
        if spec_dir is not None:
            environment["TUNINGFORK_SPEC_DIR"] = str(spec_dir)
        argv, env = Sandbox("/usr/bin/bwrap", "0.11.1", hide=tuple(hide)).check_command(
            "python3 check.py", workspace=workspace, environment=environment, reply_file=reply, spec_dir=spec_dir,
        )
        return argv, env, workspace, reply

    def test_it_works_where_its_launch_worked_and_finds_the_reply_beside_it(self, tmp_path, elsewhere):
        argv, env, workspace, reply = self.command(tmp_path)
        assert argv[0] == "/usr/bin/bwrap" and argv[-4:] == ["--", "/bin/sh", "-c", "python3 check.py"]
        joined = " ".join(argv)
        assert f"--bind {workspace.resolve()} /home/user/project" in joined
        assert f"--ro-bind {reply.resolve()} /home/user/reply.txt" in joined
        assert "--chdir /home/user/project" in joined and "--unshare-pid" in joined
        assert env["TUNINGFORK_REPLY"] == "/home/user/reply.txt"
        assert (env["HOME"], env["TMPDIR"], env["PWD"]) == ("/home/user", "/tmp", "/home/user/project")

    def test_its_tasks_folder_shows_through_a_folder_hidden_from_launches(self, tmp_path, elsewhere):
        """A benchmark hides its whole folder from every launch, and its checks live in it."""
        bench = tmp_path / "bench"
        spec_dir = bench / "specs" / "design-md"
        spec_dir.mkdir(parents=True)
        argv, env, *_ = self.command(tmp_path, hide=[bench], spec_dir=spec_dir)
        hidden_at, shown_at = argv.index(str(bench.resolve())), argv.index(str(spec_dir.resolve()))
        assert argv[hidden_at - 1] == "--tmpfs" and argv[shown_at - 1] == "--ro-bind-try" and hidden_at < shown_at
        assert env["TUNINGFORK_SPEC_DIR"] == str(spec_dir)  # shown where it is

    @pytest.mark.parametrize("folder", [Path("/"), Path("/tmp"), Path("/home")])
    def test_a_tasks_folder_that_holds_what_the_sandbox_keeps_is_not_shown_whole(self, tmp_path, folder):
        """A task file at / or in /tmp or /home itself: shown whole, its folder would bring back
        the machine's /tmp — other programs' files and sockets — or other people's homes."""
        argv, *_ = self.command(tmp_path, spec_dir=folder)
        shown = [argv[at + 1] for at, part in enumerate(argv) if part == "--ro-bind-try"]
        assert str(folder.resolve()) not in shown

    def test_the_recordings_kept_in_its_tasks_folder_stay_hidden(self, tmp_path, elsewhere):
        """As a batch hides them: the task's folder, and the recordings and runs' folders inside
        it (`evals.json` in a project, `--home .tuningfork`). A launch needs a word for the outer
        folder alone; its check, shown that folder, needs one for each folder inside."""
        project = tmp_path / "project"
        recordings, runs = project / ".tuningfork", project / ".tuningfork" / "batches"
        runs.mkdir(parents=True)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sandbox = Sandbox("/usr/bin/bwrap", "0.11.1", hide=(recordings, runs, project), own=(recordings, runs))
        argv, _ = sandbox.check_command("true", workspace=workspace, environment={}, spec_dir=project)
        shown_at = argv.index(str(project.resolve()))
        assert argv[shown_at - 1] == "--ro-bind-try"
        for folder in (recordings, runs):
            hidden_at = argv.index(str(folder.resolve()))
            assert argv[hidden_at - 1] == "--tmpfs" and hidden_at > shown_at

    def test_a_tasks_folder_that_is_tuningforks_own_is_never_shown(self, tmp_path, elsewhere):
        """A task file kept in the recordings folder itself: shown, its folder would be the
        recordings, keys and all."""
        recordings = tmp_path / "recordings"
        recordings.mkdir()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sandbox = Sandbox("/usr/bin/bwrap", "0.11.1", hide=(recordings,), own=(recordings,))
        argv, _ = sandbox.check_command("true", workspace=workspace, environment={}, spec_dir=recordings)
        assert "--ro-bind-try" not in argv and argv[argv.index(str(recordings.resolve())) - 1] == "--tmpfs"
        assert not sandbox.shows(recordings) and sandbox.shows(recordings / "scenarios" / "one")

    def test_a_hidden_folder_that_is_a_python_environment_is_not_shown(self, tmp_path, elsewhere):
        recordings = tmp_path / "recordings"
        (recordings / "bin").mkdir(parents=True)
        (recordings / "pyvenv.cfg").write_text("home = /usr/bin\n")
        argv, env, *_ = self.command(tmp_path, hide=[recordings], path=f"{recordings / 'bin'}{os.pathsep}/usr/bin")
        assert "--ro-bind-try" not in argv and env["PATH"] == "/usr/bin"

    def test_a_folder_hidden_inside_its_tasks_folder_stays_hidden(self, tmp_path, elsewhere):
        spec_dir = tmp_path / "work"
        recordings = spec_dir / ".tuningfork"
        recordings.mkdir(parents=True)
        argv, *_ = self.command(tmp_path, hide=[recordings], spec_dir=spec_dir)
        shown_at, hidden_at = argv.index(str(spec_dir.resolve())), argv.index(str(recordings.resolve()))
        assert argv[shown_at - 1] == "--ro-bind-try" and argv[hidden_at - 1] == "--tmpfs" and shown_at < hidden_at

    def test_a_python_environment_a_hidden_folder_holds_stays_on_its_path(self, tmp_path, elsewhere):
        """A venv kept in the recordings folder: a check's `python3` is often its own."""
        recordings = tmp_path / "recordings"
        venv = recordings / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        tools = recordings / "tools"
        tools.mkdir()
        path = os.pathsep.join([str(venv / "bin"), str(tools), "/usr/bin"])
        argv, env, *_ = self.command(tmp_path, hide=[recordings], path=path)
        assert env["PATH"].split(os.pathsep) == [str(venv / "bin"), "/usr/bin"]
        assert argv[argv.index(str(venv.resolve())) - 1] == "--ro-bind-try"
        assert str(tools.resolve()) not in argv


class TestTheBatchPlan:
    @pytest.fixture
    def stack(self, tmp_path, mock_server, monkeypatch):
        with task_stack(tmp_path, mock_server, monkeypatch) as built:
            built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
            yield built

    def prepare(self, stack):
        from tuningfork.ops.batch import prepare_batch

        return prepare_batch(
            stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
            model="m", harness="fake", n=1,
        )

    def test_it_says_what_every_run_sees(self, stack, monkeypatch):
        import tuningfork.sandbox

        monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide, **_: (Sandbox("/usr/bin/bwrap", "0.11.1"), None))
        data = self.prepare(stack).to_json()
        assert data["sandbox"]["workspace"] == "/home/user/project" and data["notes"] == []

    def test_a_task_file_in_tmp_itself_is_warned_of(self, stack, monkeypatch):
        """Its check's scripts beside it are in the machine's /tmp, which a sandboxed check
        never sees: shown whole, it would bring back other programs' files and sockets."""
        import tuningfork.sandbox

        monkeypatch.setattr(tuningfork.sandbox, "TMP", stack.spec.path.parent.resolve())
        monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide, **_: (Sandbox("/usr/bin/bwrap", "0.11.1"), None))
        (note,) = self.prepare(stack).to_json()["notes"]
        assert note["level"] == "warning" and "never sees" in note["message"]
        # A check of the reply reads nothing beside the task file.
        stack.item.tuningfork.check = {"reply_contains": "61.5"}
        stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
        assert self.prepare(stack).to_json()["notes"] == []

    def test_without_it_a_warning_says_why_and_what_changes(self, stack, monkeypatch):
        import tuningfork.sandbox

        monkeypatch.delenv(SWITCH, raising=False)
        monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide, **_: (None, "bubblewrap is not installed"))
        data = self.prepare(stack).to_json()
        assert data["sandbox"] is None
        (note,) = data["notes"]
        assert note["level"] == "warning" and "bubblewrap is not installed" in note["message"]
        assert "differ from run to run" in note["message"]


class TestInBubblewrap:
    def test_a_launch_sees_its_home_and_project_and_nothing_hidden(self, sandboxed, tmp_path):
        home, workspace = folders(tmp_path)
        script = (
            "import json, os\n"
            "open('made.txt', 'w').write('x')\n"
            "open(os.path.expanduser('~/note.txt'), 'w').write('y')\n"
            f"print(json.dumps({{'cwd': os.getcwd(), 'home': os.environ['HOME'], "
            f"'hidden': os.listdir({str(FIXTURES)!r}), 'real_home_writable': os.access({str(Path.home())!r}, os.W_OK), "
            "'pids': sum(name.isdigit() for name in os.listdir('/proc'))}))\n"
        )
        spec = LaunchSpec(argv=(sys.executable, "-c", script), env={"PATH": os.environ.get("PATH", "")}, cwd=workspace)
        wrapped = Sandbox(sandboxed.executable, sandboxed.version, hide=(FIXTURES,)).wrap(spec, home=home, workspace=workspace)
        done = subprocess.run(list(wrapped.argv), env=wrapped.env, cwd=wrapped.cwd, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        seen = json.loads(done.stdout)
        assert seen["cwd"] == "/home/user/project" and seen["home"] == "/home/user"
        if hideable([FIXTURES], real_home=Path.home()):
            assert seen["hidden"] == []  # empty inside, untouched outside
            assert (FIXTURES / "formats").is_dir()
        assert seen["real_home_writable"] is False
        assert seen["pids"] < 10  # its own processes only
        # What it wrote is in the launch's own folders, where tuningfork snapshots and checks it.
        assert (workspace / "made.txt").read_text() == "x" and (home / "note.txt").read_text() == "y"

    def test_runs_of_a_batch_ask_the_same_thing_and_are_judged_as_before(self, sandboxed, tmp_path, mock_server, monkeypatch):
        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            sandbox, _ = for_launches([stack.store.home, stack.tmp / "batches"])
            runner = BatchRunner(
                stack.store, FakeAgent(), base_url=f"{stack.url}/v1", model="m",
                workdir=stack.tmp / "batches", settle=0.1, sandbox=sandbox,
            )
            outcomes = runner.run(stack.spec, stack.item, 2).outcomes
            # Judged as ever: the original wording's report has the wrong keys, and the check,
            # run in the sandbox too, found it where the harness wrote it.
            assert [(outcome.usable, outcome.passed) for outcome in outcomes] == [(True, False)] * 2, outcomes[0].stderr
            launch = stack.store.get_launch(stack.store.get_run(outcomes[0].run_id).launch_id)
            assert json.loads((Path(launch.workspace) / "report.json").read_text()) == {"count": 4, "total": 61.5}
            requests = [stack.store.harness_request(stack.store.steps(outcome.run_id)[0]) for outcome in outcomes]
            assert "Working directory: /home/user/project" in requests[0]["messages"][0]["content"]
            assert requests[0] == requests[1]  # nothing in the prompt tells one run from another
            assert launch.settings["sandbox"]["workspace"] == "/home/user/project"
            assert (Path(launch.home) / "project").is_dir()  # the mount point, made by tuningfork

    def test_a_check_runs_where_its_launch_ran_and_sees_its_tasks_folder(self, sandboxed, tmp_path, monkeypatch):
        from tuningfork.specs import run_check

        _, workspace = folders(tmp_path)
        (workspace / "look.py").write_text(LOOK)
        reply = workspace.parent / "reply.txt"
        reply.write_text("The report is written.")
        monkeypatch.setenv("CLOUD_API_KEY", "sk-not-for-a-check")
        spec_dir = FIXTURES / "formats"  # the task's folder, inside a folder hidden from launches
        sandbox = Sandbox(sandboxed.executable, sandboxed.version, hide=(FIXTURES,))
        result = run_check(
            f'python3 look.py "{FIXTURES}" "{Path.home()}"', workspace, 60,
            reply_file=reply, spec_dir=spec_dir, sandbox=sandbox, keep=None,
        )
        assert result.passed, result.output
        seen = json.loads(result.output)
        assert (seen["cwd"], seen["home"]) == ("/home/user/project", "/home/user")
        assert seen["files"] == ["look.py"] and seen["reply"] == "The report is written."
        assert seen["key"] is None and seen["real_home_writable"] is False
        assert seen["task"] == sorted(os.listdir(spec_dir))
        if hideable([FIXTURES], real_home=Path.home()):
            assert seen["hidden"] == ["formats"]  # empty inside, but for its task's folder

    def test_a_check_in_a_project_never_reads_the_recordings_kept_there(self, sandboxed, tmp_path):
        """The task file in a project folder, recorded into a folder inside it: the check reads
        what is kept beside the task file, and none of the recordings — not their keys, not
        another run's workspace."""
        from tuningfork.specs import run_check

        project = tmp_path / "project"
        recordings = project / ".tuningfork"
        other_run = recordings / "batches" / "b0" / "0" / "workspace"
        other_run.mkdir(parents=True)
        (recordings / ".access-token").write_text("the proxy's key")
        (other_run / "notes.txt").write_text("another run's file")
        (project / "check.sh").write_text("echo beside-the-task\n")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sandbox, _ = for_launches(
            [recordings, recordings / "batches", project], own=[recordings, recordings / "batches"],
        )
        result = run_check(
            f'sh "$TUNINGFORK_SPEC_DIR/check.sh"; cat "{recordings / ".access-token"}" "{other_run / "notes.txt"}"',
            workspace, 60, spec_dir=project, sandbox=sandbox,
        )
        assert "beside-the-task" in result.output
        assert "the proxy's key" not in result.output and "another run's file" not in result.output

    def test_a_check_keeps_a_python_environment_the_recordings_folder_holds(self, sandboxed, tmp_path, monkeypatch, elsewhere):
        """The recordings folder is empty inside, its keys with it — but for the venv on the
        check's PATH that it holds."""
        from tuningfork.specs import run_check

        recordings = tmp_path / "recordings"
        venv = recordings / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        judge = venv / "bin" / "judge"
        judge.write_text("#!/bin/sh\necho judged\n")
        judge.chmod(0o755)
        (recordings / ".access-token").write_text("the proxy's key")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("PATH", f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
        sandbox = Sandbox(sandboxed.executable, sandboxed.version, hide=(recordings,))
        result = run_check(f'judge && test ! -e "{recordings / ".access-token"}"', workspace, 60, sandbox=sandbox)
        assert result.passed and "judged" in result.output, result.output

    def test_a_fork_runs_where_its_source_ran_or_not_at_all(self, sandboxed, tmp_path, mock_server, monkeypatch):
        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            sandbox, _ = for_launches([stack.store.home, stack.tmp / "batches"])
            runner = BatchRunner(
                stack.store, FakeAgent(), base_url=f"{stack.url}/v1", model="m",
                workdir=stack.tmp / "batches", settle=0.1, sandbox=sandbox,
            )
            (source,) = runner.run(stack.spec, stack.item, 1).outcomes
            plan, (fork,) = stack.fork(source.run_id, "write_file")
            assert plan.sandbox is not None
            # The same paths as its source: every recorded call replayed, and live from the step.
            assert fork.usable and fork.departed is None and fork.reached
            assert fork.live_outcome["label"].startswith("write_file(")
            monkeypatch.setenv(SWITCH, "off")
            with pytest.raises(OperationError) as refused:
                stack.fork(source.run_id, "write_file")
            assert refused.value.code == "no_sandbox"


class TestTheClock:
    def test_a_launch_starts_at_the_pinned_instant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/tester")))
        home, workspace = folders(tmp_path)
        spec = LaunchSpec(argv=("hermes", "chat"), env={"PATH": "/usr/bin", "LD_PRELOAD": "/lib/other.so"}, cwd=workspace)
        pinned = Sandbox("/usr/bin/bwrap", "0.11.1", clock="2026-10-08 09:00:00", clock_library="/lib/libfaketime.so.1")
        wrapped = pinned.wrap(spec, home=home, workspace=workspace)
        assert given_by_setenv(wrapped.argv) == {
            "LD_PRELOAD": "/lib/libfaketime.so.1:/lib/other.so", "FAKETIME": "@2026-10-08 09:00:00",
            "FAKETIME_DONT_FAKE_MONOTONIC": "1",
        }
        # The harness's alone: in bubblewrap's own environment, libfaketime would start there too,
        # its semaphore in a /dev the harness cannot see (Claude Code stopped at once with 0.9.10).
        assert "FAKETIME" not in wrapped.env and wrapped.env["LD_PRELOAD"] == "/lib/other.so"
        assert pinned.to_json()["clock"] == "2026-10-08 09:00:00"
        plain = Sandbox("/usr/bin/bwrap", "0.11.1")
        unpinned = plain.wrap(spec, home=home, workspace=workspace)
        assert given_by_setenv(unpinned.argv) == {} and "FAKETIME" not in unpinned.env
        assert "clock" not in plain.to_json()

    def test_the_persons_setting_pins_it_where_the_library_is(self, tmp_path, monkeypatch):
        from tuningfork.sandbox import CLOCK, CLOCK_LIBRARY, Status

        monkeypatch.setattr(tuningfork.sandbox, "status", lambda: Status("/usr/bin/bwrap", "0.11.1"))
        library = tmp_path / "libfaketime.so.1"
        library.write_bytes(b"")
        monkeypatch.setenv(CLOCK, "2026-10-08 09:00:00")
        monkeypatch.setenv(CLOCK_LIBRARY, str(library))
        sandbox, why = for_launches([])
        assert why is None and sandbox.clock == "2026-10-08 09:00:00" and sandbox.clock_library == str(library)
        assert sandbox.clock_problem is None
        # A fork gives its source's clock, or none, whatever the person set now.
        assert for_launches([], clock=None)[0].clock is None
        assert for_launches([], clock="2026-01-05 09:00:00")[0].clock == "2026-01-05 09:00:00"
        # Without the library the runs keep the real clock, and the plans can say so.
        monkeypatch.setenv(CLOCK_LIBRARY, str(tmp_path / "missing.so"))
        monkeypatch.setattr(tuningfork.sandbox, "CLOCK_LIBRARIES", ())
        unpinned, _ = for_launches([])
        assert unpinned.clock is None and "libfaketime is not installed" in unpinned.clock_problem
        # Not an instant: the real clock, and why.
        monkeypatch.setenv(CLOCK, "tomorrow")
        odd, _ = for_launches([])
        assert odd.clock is None and "not an instant" in odd.clock_problem
        # Off, or unset: the real clock, nothing to say.
        monkeypatch.setenv(CLOCK, "off")
        assert for_launches([])[0].clock_problem is None
        monkeypatch.delenv(CLOCK)
        assert for_launches([])[0].clock is None

    def test_the_batch_plan_says_it_or_warns(self, tmp_path, mock_server, monkeypatch):
        from tuningfork.ops.batch import prepare_batch

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            kwargs = dict(base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches", model="m", harness="fake", n=1)
            monkeypatch.setattr(
                tuningfork.sandbox, "for_launches",
                lambda hide, **_: (Sandbox("/usr/bin/bwrap", "0.11.1", clock="2026-10-08 09:00:00", clock_library="/lib/x.so"), None),
            )
            data = prepare_batch(stack.store, stack.spec.path, **kwargs).to_json()
            assert data["sandbox"]["clock"] == "2026-10-08 09:00:00" and data["notes"] == []
            problem = "the clock cannot be pinned at 2026-10-08 09:00:00: libfaketime is not installed"
            monkeypatch.setattr(
                tuningfork.sandbox, "for_launches",
                lambda hide, **_: (Sandbox("/usr/bin/bwrap", "0.11.1", clock_problem=problem), None),
            )
            data = prepare_batch(stack.store, stack.spec.path, **kwargs).to_json()
            assert "clock" not in data["sandbox"]
            (note,) = data["notes"]
            assert note["level"] == "warning" and "libfaketime is not installed" in note["message"]

    def test_a_harness_the_pinned_clock_stops_is_found_by_its_own_version(self, tmp_path, monkeypatch):
        """With the libfaketime Ubuntu 26.04 ships, tried on x86-64, Claude Code's native binary
        hangs. Its own --version, run with the clock a launch's harness gets, says so before any
        run is launched, after the real clock's has answered, naming the library. It says so too
        of one that stops at once, as Claude Code did with 0.9.10 while bubblewrap loaded the
        library. A harness that fails without the clock too is not the clock's to refuse."""
        from types import SimpleNamespace

        import tuningfork.ops.connections as connections

        monkeypatch.setattr(connections, "CLOCK_TIMEOUT", 1.0)
        monkeypatch.setattr(connections, "_ON_THE_CLOCK", {})
        library = tmp_path / "libfaketime.so.1"
        library.write_bytes(b"")
        bwrap = bubblewrap_stand_in(tmp_path)
        pinned = Sandbox(bwrap, "0.11.1", clock="2026-10-08 09:00:00", clock_library=str(library))
        asked = tmp_path / "asked"

        def harness(name: str, body: str) -> SimpleNamespace:
            program = tmp_path / name
            program.write_text(f'#!/bin/sh\necho "{name} ${{FAKETIME:-real}}" >> \'{asked}\'\n{body}', encoding="utf-8")
            program.chmod(0o755)
            return SimpleNamespace(name=name, executable=str(program))

        def runs_of(name: str) -> list[str]:
            return [line.split(" ", 1)[1] for line in asked.read_text(encoding="utf-8").splitlines()
                    if line.split(" ", 1)[0] == name]

        hangs = harness("hangs", '[ -n "$FAKETIME" ] && exec sleep 30\necho 1.0\n')
        stops = harness("stops", '[ -n "$FAKETIME" ] && { echo "sem_open failed" >&2; exit 1; }\necho 1.0\n')
        runs = harness("runs", "echo 1.0\n")
        broken = harness("broken", "exit 3\n")

        assert connections.clock_problem(hangs, pinned) == (
            "hangs cannot run with the clock pinned at 2026-10-08 09:00:00: its --version answers on the real "
            f"clock, but with libfaketime loaded from {library} it did not answer in 1 s")
        assert runs_of("hangs") == ["real", "@2026-10-08 09:00:00"]
        assert (f"its --version answers on the real clock, but with libfaketime loaded from {library} it exited 1"
                in connections.clock_problem(stops, pinned))
        assert connections.clock_problem(runs, pinned) is None
        # Broken without the clock too: its launch says what is wrong, and the clock is not asked.
        assert connections.clock_problem(broken, pinned) is None
        assert runs_of("broken") == ["real"]

        # Asked once, each: the plans that follow wait for nothing, and quote their own instant.
        before = asked.read_text(encoding="utf-8")
        later = Sandbox(bwrap, "0.11.1", clock="2027-01-01 00:00:00", clock_library=str(library))
        assert "pinned at 2027-01-01 00:00:00" in connections.clock_problem(hangs, later)
        assert connections.clock_problem(runs, later) is None
        assert asked.read_text(encoding="utf-8") == before
        # No clock pinned, nothing asked.
        assert connections.clock_problem(hangs, Sandbox("/usr/bin/bwrap", "0.11.1")) is None
        assert connections.clock_problem(hangs, None) is None
        assert asked.read_text(encoding="utf-8") == before
        # Five minutes on, a refusal is asked again, and so is a harness that did not answer on
        # the real clock either: a machine too busy to answer once must not refuse a harness, or
        # spare it, for the life of a serve. A clean answer is kept.
        for key, (failed, until) in list(connections._ON_THE_CLOCK.items()):
            connections._ON_THE_CLOCK[key] = (failed, until - connections.CLOCK_PROBLEM_KEPT - 1)
        assert connections.clock_problem(hangs, pinned) and connections.clock_problem(runs, pinned) is None
        assert connections.clock_problem(broken, pinned) is None
        assert runs_of("hangs") == ["real", "@2026-10-08 09:00:00"] * 2 and runs_of("runs") == ["real", "@2026-10-08 09:00:00"]
        assert runs_of("broken") == ["real"] * 2

    def test_the_clock_is_asked_once_at_a_time_and_leaves_nothing_running(self, tmp_path, monkeypatch):
        """The web UI plans a form again at each change: plans made at once wait for the first
        one's answer rather than each starting the harness again, and a harness that hangs with
        the clock pinned is ended, with everything it started."""
        import threading
        from types import SimpleNamespace

        import tuningfork.ops.connections as connections

        monkeypatch.setattr(connections, "CLOCK_TIMEOUT", 1.0)
        monkeypatch.setattr(connections, "_ON_THE_CLOCK", {})
        library = tmp_path / "libfaketime.so.1"
        library.write_bytes(b"")
        pinned = Sandbox(bubblewrap_stand_in(tmp_path), "0.11.1", clock="2026-10-08 09:00:00", clock_library=str(library))
        asked, marker = tmp_path / "asked", "41.25"
        program = tmp_path / "hangs"
        program.write_text(
            f"#!/bin/sh\necho x >> '{asked}'\n[ -n \"$FAKETIME\" ] && {{ sleep {marker} & exec sleep 30; }}\necho 1.0\n",
            encoding="utf-8")
        program.chmod(0o755)
        hangs = SimpleNamespace(name="hangs", executable=str(program))

        said: list[str | None] = []
        plans = [threading.Thread(target=lambda: said.append(connections.clock_problem(hangs, pinned))) for _ in range(3)]
        for plan in plans:
            plan.start()
        for plan in plans:
            plan.join(timeout=30)
        assert len(said) == 3 and all(said)
        assert asked.read_text(encoding="utf-8").count("x") == 2

        def running(entry: Path) -> bool:
            try:
                return entry.name.isdigit() and marker.encode() in (entry / "cmdline").read_bytes()
            except OSError:
                return False

        assert [entry.name for entry in Path("/proc").iterdir() if running(entry)] == []

    def test_the_batch_plan_refuses_a_harness_the_pinned_clock_stops(self, tmp_path, mock_server, monkeypatch):
        """Refused before any run is launched, in the words its own --version gave, with how to
        run it anyway: every run would end before its first call, or hang until its timeout."""
        import tuningfork.ops.connections as connections
        from tuningfork.ops.batch import prepare_batch
        from tuningfork.sandbox import ANOTHER_LIBRARY, CLOCK

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            monkeypatch.setattr(
                tuningfork.sandbox, "for_launches",
                lambda hide, **_: (Sandbox("/usr/bin/bwrap", "0.11.1", clock="2026-10-08 09:00:00", clock_library="/lib/x.so"), None),
            )
            monkeypatch.setattr(connections, "clock_problem", lambda adapter, sandbox: (
                f"{adapter.name} cannot run with the clock pinned at {sandbox.clock}"))
            with pytest.raises(OperationError) as refused:
                prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                              model="m", harness="fake", n=1)
            assert refused.value.message == "fake cannot run with the clock pinned at 2026-10-08 09:00:00"
            assert refused.value.hint == f"{ANOTHER_LIBRARY}, or unset {CLOCK} to run it on the real clock"
            assert "name another build of libfaketime with TUNINGFORK_CLOCK_LIBRARY" in refused.value.hint
            assert refused.value.field == "harness"

    def test_a_launch_reads_the_pinned_clock(self, sandboxed, tmp_path):
        from tuningfork.sandbox import clock_library

        library = clock_library()
        if library is None:
            pytest.skip("libfaketime is not installed here")
        home, workspace = folders(tmp_path)
        script = "import datetime; print(datetime.datetime.now().strftime('%A, %B %d, %Y'))"
        spec = LaunchSpec(argv=(sys.executable, "-c", script), env={"PATH": os.environ.get("PATH", "")}, cwd=workspace)
        pinned = Sandbox(sandboxed.executable, sandboxed.version, clock="2026-10-08 09:00:00", clock_library=library)
        wrapped = pinned.wrap(spec, home=home, workspace=workspace)
        done = subprocess.run(list(wrapped.argv), env=wrapped.env, cwd=wrapped.cwd, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "Thursday, October 08, 2026"

    def test_a_killed_launch_leaves_nothing_of_the_clock_on_the_machine(self, sandboxed, tmp_path):
        """libfaketime is the harness's alone: in bubblewrap's own environment it would start
        there too, outside the sandbox's /dev, and leave its semaphore in the machine's /dev/shm
        whenever a launch is killed — a run cut at its task's timeout, a batch stopped."""
        import contextlib
        import signal
        import threading

        from tuningfork.sandbox import clock_library

        library = clock_library()
        if library is None:
            pytest.skip("libfaketime is not installed here")
        home, workspace = folders(tmp_path)
        spec = LaunchSpec(argv=("sh", "-c", "date; exec sleep 30"), env={"PATH": os.environ.get("PATH", "")}, cwd=workspace)
        pinned = Sandbox(sandboxed.executable, sandboxed.version, clock="2026-10-08 09:00:00", clock_library=library)
        wrapped = pinned.wrap(spec, home=home, workspace=workspace)
        before = set(os.listdir("/dev/shm"))
        running = subprocess.Popen(list(wrapped.argv), env=wrapped.env, cwd=wrapped.cwd, stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)

        def end() -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(running.pid, signal.SIGKILL)

        # A launch that never prints must not hold the suite.
        guard = threading.Timer(30, end)
        guard.start()
        try:
            started = running.stdout.readline()
        finally:
            guard.cancel()
            end()
            running.wait()
            running.stdout.close()
        assert "Oct" in started and "2026" in started
        # libfaketime names what it leaves by its process, and other launches make entries too.
        assert [name for name in set(os.listdir("/dev/shm")) - before if name.endswith(f"_{running.pid}")] == []

    def test_what_the_probe_leaves_in_dev_shm_goes_with_it(self, sandboxed, tmp_path, monkeypatch):
        """A harness given libfaketime can leave its semaphore in /dev/shm — run outside
        bubblewrap, Claude Code's native binary did — or hang and be ended, and a refusal is
        asked again every five minutes. The probe gives libfaketime to the harness alone, in
        bubblewrap with a /dev of its own, as a launch does: nothing it left there stays on the
        machine, nor anything of libfaketime's own."""
        from types import SimpleNamespace

        import tuningfork.ops.connections as connections
        from tuningfork.sandbox import clock_library

        library = clock_library()
        if library is None:
            pytest.skip("libfaketime is not installed here")
        monkeypatch.setattr(connections, "CLOCK_TIMEOUT", 1.0)
        monkeypatch.setattr(connections, "_ON_THE_CLOCK", {})
        started: list[int] = []

        class Started(subprocess.Popen):
            """Each process the probe starts, by its pid: libfaketime names what it leaves by it,
            and launches running beside the suite make and remove entries of their own."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                started.append(self.pid)

        monkeypatch.setattr(connections.subprocess, "Popen", Started)
        left = Path("/dev/shm") / f"sem.faketime_sem_tuningfork-test-{os.getpid()}"
        made = tmp_path / "made"

        def harness(name: str, body: str) -> SimpleNamespace:
            program = tmp_path / name
            program.write_text(f'#!/bin/sh\n[ -n "$FAKETIME" ] || {{ echo 1.0; exit 0; }}\n{body}', encoding="utf-8")
            program.chmod(0o755)
            return SimpleNamespace(name=name, executable=str(program))

        stops = harness("stops", f"touch '{left}' && echo stops >> '{made}'\necho 'sem_open failed' >&2\nexit 1\n")
        hangs = harness("hangs", f"echo hangs >> '{made}'\nexec sleep 30\n")
        pinned = Sandbox(sandboxed.executable, sandboxed.version, clock="2026-10-08 09:00:00", clock_library=library)
        before = set(os.listdir("/dev/shm"))
        assert left.name not in before
        try:
            assert f"loaded from {library} it exited 1" in (connections.clock_problem(stops, pinned) or "")
            assert f"loaded from {library} it did not answer in 1 s" in (connections.clock_problem(hangs, pinned) or "")
            assert made.read_text(encoding="utf-8") == "stops\nhangs\n" and len(started) == 4
            new = set(os.listdir("/dev/shm")) - before
            assert left.name not in new
            assert [name for name in new if name.endswith(tuple(f"_{pid}" for pid in started))] == []
        finally:
            left.unlink(missing_ok=True)

    def test_the_probe_shows_the_harness_its_home_wherever_it_is_made(self, sandboxed, tmp_path, monkeypatch):
        """The probe's /dev is its own, which would hide a throwaway home made under /dev, where
        TMPDIR=/dev/shm puts it: a harness whose --version needs its home would fail on the real
        clock too, and go unrefused. Its home is shown where it is, and it starts there."""
        import shutil
        import tempfile
        from types import SimpleNamespace

        import tuningfork.ops.connections as connections

        monkeypatch.setattr(connections, "_ON_THE_CLOCK", {})
        library = tmp_path / "libfaketime.so.1"
        library.write_bytes(b"")
        program = tmp_path / "needs-home"
        program.write_text(
            '#!/bin/sh\n[ -d "$HOME" ] && [ "$(pwd)" = "$HOME" ] || { echo "no home" >&2; exit 9; }\n'
            '[ -n "$FAKETIME" ] && exit 1\necho 1.0\n',
            encoding="utf-8")
        program.chmod(0o755)
        pinned = Sandbox(sandboxed.executable, sandboxed.version, clock="2026-10-08 09:00:00", clock_library=str(library))
        under_dev = tempfile.mkdtemp(prefix="tuningfork-test-", dir="/dev/shm")
        monkeypatch.setattr(tempfile, "tempdir", under_dev)
        try:
            said = connections.clock_problem(SimpleNamespace(name="needs-home", executable=str(program)), pinned)
        finally:
            shutil.rmtree(under_dev, ignore_errors=True)
        assert said is not None and f"with libfaketime loaded from {library} it exited 1" in said

