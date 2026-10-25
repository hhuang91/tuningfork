"""Forks: a recorded run replayed to a relaunched harness, then live from one step.

End to end, with no model: the demo's `agent.py` is a real agent process talking HTTP to a
real proxy on a socket, and the mock server plays the model — answering by what it is
shown, so a wording swapped in at a fork's step changes what it does, as a model's would.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from tuningfork.ops import OperationError
from tuningfork.ops.forks import execute_forks, prepare_fork
from tuningfork.store.db import Store
from tuningfork.demo.mock import MockTurn
from tests.helpers import VARIANT, control_headers, tally_model, task_stack


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        yield built


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settled(store: Store, run_id: str, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        steps = store.steps(run_id)
        if len(steps) >= count:
            return steps
        time.sleep(0.05)
    return store.steps(run_id)


def test_a_fork_is_kept_from_what_its_source_was_kept_from(stack, monkeypatch):
    """Its source batch's runs' folder outside the recordings, say: the source's launch and check
    never saw it, and the fork's are kept from it too, as tuningfork's own."""
    import tuningfork.sandbox
    from tuningfork.sandbox import Sandbox

    source = stack.record()
    runs = stack.tmp / "runs-elsewhere"
    launch_id = stack.store.get_run(source.run_id).launch_id
    stack.store._conn.execute(
        "UPDATE launches SET settings = json_set(settings, '$.sandbox', json(?)) WHERE id = ?",
        (json.dumps({"kind": "bubblewrap", "hidden": [str(runs)], "own": [str(runs)]}), launch_id),
    )
    asked: dict = {}

    def recorded(hide, **options):
        asked.update(hide=[Path(folder) for folder in hide], own=[Path(folder) for folder in options["own"]])
        return Sandbox("/usr/bin/bwrap", "0.11.1"), None

    monkeypatch.setattr(tuningfork.sandbox, "for_launches", recorded)
    prepare_fork(stack.store, source.run_id, "write_file", base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
    assert runs in asked["hide"] and runs in asked["own"]


def test_a_fork_reads_the_configs_list_once_and_only_for_a_check_that_runs_a_command(stack, monkeypatch):
    """A list that cannot be read refuses a fork of a task whose check runs a command, before
    its harness is launched; a fork of a task judged on its reply never reads it, and runs to its
    end."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(stack.tmp / "config"))
    machine = stack.tmp / "config" / "tuningfork" / "config.json"
    machine.parent.mkdir(parents=True)
    source = stack.record()
    stack.item.tuningfork.check = {"reply_contains": "written"}
    replied = stack.record()
    machine.write_text(json.dumps({"check_env": "https_proxy"}), encoding="utf-8")
    with pytest.raises(OperationError) as refused:
        prepare_fork(stack.store, source.run_id, "write_file", base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
    assert refused.value.code == "check_env"
    _, (fork,) = stack.fork(replied.run_id, "write_file")
    assert fork.usable and fork.passed


class TestReplay:
    def test_a_fork_is_answered_from_the_recording_then_goes_live(self, stack):
        source = stack.record()
        assert not source.passed  # the shipped wording invites the wrong keys
        seen = len(stack.mock.requests)

        _, (fork,) = stack.fork(source.run_id, "write_file")

        # The read was answered from the recording: only the write and the summary reached
        # the model.
        assert len(stack.mock.requests) - seen == 2
        steps = settled(stack.store, fork.run_id, 3)
        source_steps = stack.store.steps(source.run_id)
        assert steps[0].served and steps[0].replay["served_from"] == source_steps[0].id
        assert [s.served for s in steps[1:]] == [False, False]
        assert fork.departed is None and fork.reached and fork.usable
        assert "write_file" in fork.live_outcome["label"]
        assert not fork.passed  # same wording, same model: the same wrong keys

    def test_a_variant_is_swapped_in_from_the_step_on_and_only_there(self, stack):
        source = stack.record()
        variant = stack.tmp / "variants" / "SKILL.v2.md"
        variant.parent.mkdir()
        variant.write_text(VARIANT, encoding="utf-8")

        plan, (fork,) = stack.fork(source.run_id, "1", variant=variant)

        assert fork.passed and fork.usable
        steps = settled(stack.store, fork.run_id, 3)
        assert steps[0].served and not steps[0].substitution
        assert steps[1].substitution["spans"] >= 1 and not steps[1].edits
        # The model read the variant; the harness still sent the original.
        assert "row_count" in json.dumps(stack.store.rebuild_request(steps[1]))
        assert "row_count" not in json.dumps(stack.store.harness_request(steps[1]))
        assert plan.spans  # the original was found in the context at the step

    def test_served_answers_name_the_forks_own_workspace(self, stack):
        """The model wrote to an absolute path: replayed as recorded, every fork would
        write into the source run's workspace — and per-run path masking would hide it."""
        source = stack.record()
        launch = stack.store.get_launch(source.launch_id)
        source_report = Path(launch.workspace) / "report.json"
        before = digest(source_report), source_report.stat().st_mtime_ns

        _, (fork,) = stack.fork(source.run_id, "2")

        fork_launch = stack.store.get_launch(fork.launch_id)
        assert (Path(fork_launch.workspace) / "report.json").exists()
        assert (digest(source_report), source_report.stat().st_mtime_ns) == before
        steps = settled(stack.store, fork.run_id, 3)
        assert [s.served for s in steps] == [True, True, False]
        assert fork.usable

    def test_a_tool_whose_output_changes_takes_the_fork_off_the_recording(self, stack, monkeypatch):
        source = stack.record()
        monkeypatch.setenv("FAKE_NOISE", "1")

        _, (fork,) = stack.fork(source.run_id, "2")

        assert fork.departed and fork.departed["step"] == 1
        assert not fork.usable
        assert "departed from the recording at step 1" in fork.exclusion
        assert "µs" in fork.departed["got"]

    def test_many_forks_at_once_each_replay_the_whole_recording(self, stack):
        source = stack.record()
        seen = len(stack.mock.requests)
        plan = prepare_fork(
            stack.store, source.run_id, "1", base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks"
        )

        forks = execute_forks(stack.store, plan, 4, concurrency=4)

        assert len({fork.run_id for fork in forks}) == 4
        assert all(fork.usable for fork in forks)
        assert len(stack.mock.requests) - seen == 4 * 2  # the read, four times over, served


@pytest.mark.skipif(not __import__("tuningfork.snapshots").snapshots.available(), reason="needs git")
class TestSnapshots:
    def test_every_call_of_a_launched_run_has_its_workspace(self, stack):
        source = stack.record()
        steps = settled(stack.store, source.run_id, 3)
        launch = stack.store.get_launch(source.launch_id)

        assert all(step.workspace_commit for step in steps)
        assert launch.initial_commit and launch.final_commit and launch.skill_commit

    def test_a_fork_starts_from_what_its_source_started_from(self, stack):
        """Not from the spec as it is now: the task's input was edited after the run."""
        source = stack.record()
        stack.item.files[0]["content"] = "id,amount\nedited,0\n"

        _, (fork,) = stack.fork(source.run_id, "1")

        fork_launch = stack.store.get_launch(fork.launch_id)
        data = (Path(fork_launch.workspace) / "data.csv").read_text(encoding="utf-8")
        assert data.startswith("id,amount\n1,10")
        steps = settled(stack.store, fork.run_id, 3)
        # The replayed call and the call at the step found the workspace as the source did.
        assert steps[0].replay["workspace"] == "same"
        assert steps[1].replay["live_from"] and steps[1].replay["workspace"] == "same"

    def test_files_shows_the_workspace_a_call_found(self, stack):
        from rich.console import Console
        from typer.testing import CliRunner

        from tuningfork.cli import app

        source = stack.record()
        settled(stack.store, source.run_id, 3)
        runner = CliRunner()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("tuningfork.cli.console", Console(width=220))
            before = runner.invoke(app, ["files", source.run_id[:8], "write_file", "--home", str(stack.store.home)])
            after = runner.invoke(app, ["files", source.run_id[:8], "2", "--diff", "--home", str(stack.store.home)])
            end = runner.invoke(app, ["files", source.run_id[:8], "end", "--home", str(stack.store.home)])

        assert "before step 1" in before.output and "data.csv" in before.output
        assert "report.json" not in before.output
        # The write came between steps 1 and 2: step 2's call found the report there.
        assert "before step 2" in after.output and "report.json" in after.output
        assert '"count": 4' in after.output
        assert "at the end" in end.output and "report.json" in end.output


class TestBookkeeping:
    def test_a_fork_is_not_a_run_of_the_task(self, stack):
        source = stack.record()
        _, (fork,) = stack.fork(source.run_id, "1")

        assert [run.id for run in stack.store.runs()] == [source.run_id]
        assert fork.run_id in {run.id for run in stack.store.runs(forks=True)}
        run = stack.store.get_run(fork.run_id)
        assert run.is_fork and run.parity["fork_of"] == source.run_id

    def test_a_run_that_cannot_be_replayed_says_which_step(self, stack):
        source = stack.record()
        first = stack.store.steps(source.run_id)[0]
        with stack.store._lock:
            stack.store._conn.execute(
                "UPDATE steps SET raw_response_blob = NULL WHERE id = ?", (first.id,)
            )
            stack.store._conn.commit()

        with pytest.raises(OperationError) as refused:
            prepare_fork(stack.store, source.run_id, "2", base_url=f"{stack.url}/v1")
        assert "step 0 has no raw response" in refused.value.detail

    @pytest.mark.parametrize("how", ["rendered", "dropped"])
    def test_a_step_whose_bytes_were_redacted_beyond_the_text_says_so(self, stack, how):
        # The recorder wrote its bytes again around a secret, or kept none: a fork refuses it on
        # that word alone, whatever bytes are kept.
        source = stack.record()
        first = stack.store.steps(source.run_id)[0]
        params = json.dumps({**first.params, "raw_redacted": how})
        blob = first.raw_response_blob if how == "rendered" else None
        with stack.store._lock:
            stack.store._conn.execute(
                "UPDATE steps SET raw_response_blob = ?, params = ? WHERE id = ?", (blob, params, first.id)
            )
            stack.store._conn.commit()

        with pytest.raises(OperationError) as refused:
            prepare_fork(stack.store, source.run_id, "2", base_url=f"{stack.url}/v1")
        assert "step 0's answer had a secret redacted from it" in refused.value.detail

    def test_a_run_recorded_before_launches_needs_its_spec(self, stack):
        from tests.helpers import record_pilot

        run_id = record_pilot(stack.store)
        with pytest.raises(OperationError) as refused:
            prepare_fork(stack.store, run_id, "1", base_url=f"{stack.url}/v1")
        assert "--spec" in refused.value.hint


class TestToEnd:
    """Phase 4's done criterion: "resample from step k to completion, N times" reports
    end-to-end pass rates with intervals."""

    @pytest.fixture
    def cli(self, monkeypatch, stack):
        from rich.console import Console
        from typer.testing import CliRunner

        from tuningfork.cli import app

        monkeypatch.setenv("COLUMNS", "220")
        monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
        runner = CliRunner()

        def invoke(*args: str):
            proxy = ["--proxy", stack.url] if args[0] in ("resample", "ab") else []
            result = runner.invoke(app, [*args, *proxy, "--home", str(stack.store.home)])
            assert result.exception is None or isinstance(result.exception, SystemExit), (
                result.output
            )
            return result

        return invoke

    def test_resampling_to_the_end_reports_a_pass_rate_with_its_interval(self, stack, cli):
        source = stack.record()

        result = cli("resample", source.run_id[:8], "write_file", "-n", "3", "--to-end")

        assert result.exit_code == 0, result.output
        assert "check passed  0/3  0%  [0%-56%]" in result.output
        assert "what the model did at step 1" in result.output
        (experiment,) = stack.store.experiments(limit=5)
        assert experiment.settings["reach"] == "end"
        samples = stack.store.samples(experiment.id)
        assert [s.passed for s in samples] == [False, False, False]
        assert all(s.run_id and s.exclusion is None for s in samples)

    def test_a_forks_answer_at_its_step_is_kept_as_its_step_stored_it(self, stack, cli):
        """Masked once, as the step was: masked again, a command's second password falls too, and the
        sample reads another answer than the step."""
        source = stack.record()
        command = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"

        def model(payload):
            if sum(message["role"] == "tool" for message in payload["messages"]) == 1:
                return MockTurn(content=command)
            return tally_model(payload)

        stack.mock.responder = model
        cli("resample", source.run_id[:8], "write_file", "-n", "1", "--to-end")

        (experiment,) = stack.store.experiments(limit=5)
        (sample,) = stack.store.samples(experiment.id)
        step = next(s for s in stack.store.steps(sample.run_id) if s.idx == 1 and not s.served)
        assert stack.store.sample_response(sample) == stack.store.response(step)
        assert "-pZq8xK3vN5wLp" in json.dumps(stack.store.sample_response(sample))

    def test_runs_lists_the_task_and_says_what_it_left_out(self, stack, cli):
        """Found re-running the guide: the list showed 16 runs over a footer counting 57,
        the other 41 being forks it does not list."""
        source = stack.record()
        cli("resample", source.run_id[:8], "write_file", "-n", "2", "--to-end")

        listed = cli("runs").output

        assert source.run_id[:8] in listed
        assert "3 run(s)" in listed
        assert "not listed above: 2 fork(s): `tuningfork experiments`" in listed

    def test_the_api_names_each_fork_so_the_ui_can_link_it(self, stack, cli):
        """Forks are kept out of the list of runs, so an experiment is the way to them."""
        source = stack.record()
        cli("resample", source.run_id[:8], "write_file", "-n", "2", "--to-end")
        (experiment,) = stack.store.experiments(limit=5)

        body = httpx.get(f"{stack.url}/_tuningfork/api/experiments/{experiment.id}").json()

        forks = body["to_end"]["forks"]
        assert [fork["idx"] for fork in forks] == [0, 1]
        assert {fork["run_id"] for fork in forks} == {s.run_id for s in stack.store.samples(experiment.id)}
        assert all(fork["passed"] is False and fork["exclusion"] is None for fork in forks)

    def test_an_ab_to_the_end_compares_the_arms_on_the_check(self, stack, cli):
        source = stack.record()
        variant = stack.tmp / "SKILL.v2.md"
        variant.write_text(VARIANT, encoding="utf-8")

        result = cli("ab", source.run_id[:8], "1", "--variant", str(variant), "-n", "4", "--to-end")

        assert result.exit_code == 0, result.output
        assert "original  0/4" in result.output and "variant   4/4" in result.output
        assert "significant" in result.output and "not significant" not in result.output
        # Which way the pass rate moved, for the page to say it in words and colour it.
        arm = stack.store.experiments(limit=1)[0]
        body = httpx.get(f"{stack.url}/_tuningfork/api/experiments/{arm.id}").json()
        assert body["to_end"]["direction"] == "up" and body["to_end"]["significant"]
        # Each arm keeps the level it is judged at, as a step A/B's do.
        assert {e.arm: e.settings.get("alpha") for e in stack.store.experiments(limit=2)} == {"original": 0.05, "variant": 0.05}
        assert body["alpha"] == body["alpha_declared"] == 0.05
        # The edit it tested, as its forks were given it: the file has changed since.
        variant.write_text("another wording\n", encoding="utf-8")
        again = cli("experiment", stack.store.experiments(limit=1)[0].id[:8])
        assert "variant   4/4" in again.output
        assert "the edit  1 line changed" in again.output
        assert "+ Write the totals to report.json, using the keys row_count and amount_total." in again.output
        listing = cli("experiments")
        assert "ab to end" in listing.output

    def test_a_cancel_mid_arm_keeps_no_sample_from_the_fork_it_killed(self, stack):
        """Found in review: a fork killed by a cancel was judged like a finished one, and
        stored as a counted, failing sample."""
        from tuningfork.ops import Cancel, Cancelled
        from tuningfork.ops.forks import execute_ab_to_end

        source = stack.record()
        variant = stack.tmp / "SKILL.v2.md"
        variant.write_text(VARIANT, encoding="utf-8")
        original = prepare_fork(
            stack.store, source.run_id, "1", base_url=f"{stack.url}/v1",
            workdir=stack.tmp / "forks", breakpoints=False,
        )
        swapped = prepare_fork(
            stack.store, source.run_id, "1", base_url=f"{stack.url}/v1",
            workdir=stack.tmp / "forks", breakpoints=False, variant=variant,
        )
        live = stack.mock.responder

        def slow(payload):  # the second fork's live step takes long enough to be killed
            turn = live(payload)
            if len([m for m in payload["messages"] if m["role"] == "tool"]) == 1:
                slow.calls += 1
                if slow.calls == 2:
                    turn.stall = 30
            return turn

        slow.calls = 0
        stack.mock.responder = slow
        cancel = Cancel()
        seen = []

        def progress(event):
            if event.kind == "run":
                seen.append(event.subject)
                threading.Timer(0.5, cancel.cancel).start()

        with pytest.raises(Cancelled) as stopped:
            execute_ab_to_end(stack.store, original, swapped, 3, progress=progress, cancel=cancel)

        arms = {e.arm: e for e in stack.store.experiments(limit=5)}
        assert set(arms) == {"original"} and arms["original"].status == "cancelled"
        samples = stack.store.samples(arms["original"].id)
        assert [s.idx for s in samples] == [0] and len(seen) == 1
        assert stopped.value.partial.counted == 1
        statuses = sorted(
            row["status"] for row in stack.store._conn.execute("SELECT status FROM launches WHERE kind='fork'")
        )
        assert statuses == ["cancelled", "ended"]

    def test_more_forks_at_once_than_the_proxy_lets_through_is_refused(self, stack, cli):
        source = stack.record()
        result = cli("resample", source.run_id[:8], "1", "-n", "2", "--to-end", "--concurrency", "9")
        assert result.exit_code == 1
        assert "--max-inflight 9" in result.output

    def test_an_ab_to_the_end_is_planned_whole_before_any_fork(self, stack):
        from tuningfork.ops.forks import prepare_to_end

        source = stack.record()
        variant = stack.tmp / "SKILL.v2.md"
        variant.write_text(VARIANT, encoding="utf-8")
        options = dict(base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
        plan = prepare_to_end(stack.store, source.run_id, "1", n=3, variant=variant, **options)
        data = json.loads(json.dumps(plan.to_json()))
        assert data["kind"] == "ab" and data["forks"] == 6
        assert data["target"]["kind"] == "check"
        assert data["original"]["breakpoints"] is False and data["variant"]["breakpoints"] is False
        assert data["variant"]["variant"]["variant_path"] == str(variant)
        assert data["original"]["variant"] is None
        with pytest.raises(OperationError) as refused:
            prepare_to_end(stack.store, source.run_id, "1", n=3, concurrency=9, **options)
        assert refused.value.field == "concurrency"
        assert stack.store.experiments(limit=5) == []

    def test_a_fork_is_laid_out_as_its_source_was(self, stack, monkeypatch):
        """What a fork replays names the paths its source's launch had, so it is laid out the
        same way; a run recorded before launches recorded a layout was laid out flat."""
        from tuningfork.harness import ADAPTERS
        from tuningfork.runner import BatchRunner
        from tests.helpers import FakeAgent

        class LaidOut(FakeAgent):
            layouts = ("install", "flat")

            def __init__(self, executable=None, *, layout="install"):
                self.layout = layout

        monkeypatch.setitem(ADAPTERS, "fake", LaidOut)
        runner = BatchRunner(stack.store, LaidOut(), base_url=f"{stack.url}/v1", model="m",
                             workdir=stack.tmp / "batches", settle=0.1)
        (installed,) = runner.run(stack.spec, stack.item, 1).outcomes
        launch_of = lambda run_id: stack.store.get_launch(stack.store.get_run(run_id).launch_id)  # noqa: E731
        assert launch_of(installed.run_id).settings["layout"] == "install"
        assert prepare_fork(stack.store, installed.run_id, "1", base_url=f"{stack.url}/v1").adapter.layout == "install"

        older = stack.record()  # an adapter without layouts records none
        assert "layout" not in launch_of(older.run_id).settings
        assert prepare_fork(stack.store, older.run_id, "1", base_url=f"{stack.url}/v1").adapter.layout == "flat"

    def test_a_fork_is_set_up_as_its_source_was(self, stack, monkeypatch):
        """A harness option changes what the model is told (Hermes's self_improvement: a
        sentence of its system prompt), and a launch records its options: a fork of the run
        takes them, or its replay departs at the first step. An MCP server is recorded by
        name only, so a fork runs without it, and the plan says so."""
        from tuningfork.harness import ADAPTERS
        from tuningfork.runner import BatchRunner
        from tests.helpers import FakeAgent

        class SetUp(FakeAgent):
            layouts = ("install", "flat")

            def __init__(self, executable=None, *, layout="install", options=None):
                self.layout, self.options = layout, dict(options or {})

            @property
            def settings(self):
                return {"harness": dict(self.options)} if self.options else {}

        monkeypatch.setitem(ADAPTERS, "fake", SetUp)
        servers = {"tf": "http://127.0.0.1:1/mcp"}
        quiet = SetUp(options={"self_improvement": False, "mcp_servers": servers})
        runner = BatchRunner(stack.store, quiet, base_url=f"{stack.url}/v1", model="m",
                             workdir=stack.tmp / "batches", settle=0.1)
        (recorded,) = runner.run(stack.spec, stack.item, 1).outcomes
        launch = stack.store.get_launch(stack.store.get_run(recorded.run_id).launch_id)
        assert launch.settings["harness"] == {"self_improvement": False, "mcp_servers": servers}
        plan = prepare_fork(stack.store, recorded.run_id, "1", base_url=f"{stack.url}/v1")
        assert plan.adapter.options == plan.setup.options == {"self_improvement": False}
        assert plan.setup.dropped_servers == ["tf"]
        assert any("MCP server(s) tf" in warning and "without them" in warning for warning in plan.warnings), plan.warnings

        plain = stack.record()  # recorded with no options: forked with none
        assert prepare_fork(stack.store, plain.run_id, "1", base_url=f"{stack.url}/v1").adapter.options == {}

    def test_a_caller_inside_the_proxy_plans_a_fork_without_asking_it_over_http(self, stack, monkeypatch):
        from tuningfork.ops import launch

        source = stack.record()
        known = launch.check_proxy(f"{stack.url}/v1", stack.store.home)
        monkeypatch.setattr(launch, "check_proxy", lambda *a, **k: pytest.fail("asked over HTTP"))
        monkeypatch.setattr("tuningfork.ops.forks.check_proxy", launch.check_proxy)
        plan = prepare_fork(stack.store, source.run_id, "1", proxy=known, workdir=stack.tmp / "forks")
        assert plan.base_url == f"{stack.url}/v1" and plan.proxy is known


class TestBreakpoints:
    def test_a_forks_breakpoints_hold_only_its_live_calls(self, stack):
        """The debugger's restart: stop at the step, not on the way there."""
        source = stack.record()
        control = control_headers(stack.app)
        httpx.post(f"{stack.url}/_tuningfork/api/breakpoints", json={"phase": "request"}, headers=control)
        plan = prepare_fork(
            stack.store, source.run_id, "1", base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks"
        )
        done: list = []
        worker = threading.Thread(target=lambda: done.extend(execute_forks(stack.store, plan, 1)))
        worker.start()

        released = 0
        deadline = time.monotonic() + 30
        while worker.is_alive() and time.monotonic() < deadline:
            holds = httpx.get(f"{stack.url}/_tuningfork/api/holds").json()
            for hold in holds:
                if hold["pending"]:
                    assert hold["step"] >= 1  # never the replayed read
                    httpx.post(
                        f"{stack.url}/_tuningfork/api/holds/{hold['id']}/continue", json={}, headers=control
                    )
                    released += 1
            time.sleep(0.05)
        worker.join(5)

        (fork,) = done
        assert released == 2  # the write and the summary; the read was replayed
        assert fork.usable


class TestStoredMasks:
    """Found in review: a fork keys each call by its request as today's patterns mask it, and a
    step stored under other patterns was masked where today's keep the text — `token =
    tokenizer.encode(…)` — so every fork of a run recorded before they changed parted from it."""

    CODE = "ids = tokenizer.encode(text)\ntoken = tokenizer.encode(text)[0]\n"

    def recorded(self, store):
        from tests.helpers import record_pilot

        # As an earlier pattern stored it, before steps named their patterns: the second line's call masked.
        return record_pilot(store, read=self.CODE.replace("tokenizer.encode(text)[0]", "<redacted>(text)[0]"), params={})

    def test_a_call_fits_a_step_stored_under_other_masks(self, store):
        from tuningfork.replay import build_table
        from tuningfork.wire import get_wire

        run = store.get_run(self.recorded(store))
        stored = store.rebuild_request(store.steps(run.id)[1])
        live = json.loads(json.dumps(stored).replace("<redacted>", "tokenizer.encode"))
        table = build_table(store, run, 1)
        wire = get_wire("openai-chat")
        assert table.fitting(live, wire) == table.live_key
        other = json.loads(json.dumps(live).replace("[0]", "[1]"))
        assert table.fitting(other, wire) is None

    def test_a_call_that_parts_from_the_recording_after_a_mask_does_not_fit(self, store):
        """Found in review's second round: a mask stood for any text, so a call whose tool result
        went on, after the masked text, to say something else took the recorded answer."""
        from tuningfork.replay import build_table
        from tuningfork.wire import get_wire

        run = store.get_run(self.recorded(store))
        stored = store.rebuild_request(store.steps(run.id)[1])
        parted = json.loads(json.dumps(stored).replace(
            "<redacted>", "tokenizer.encode(text)[0]\\nERROR: publish.sh refused, nothing was published\\nx = tokenizer.encode"))
        assert build_table(store, run, 1).fitting(parted, get_wire("openai-chat")) is None

    @pytest.mark.parametrize(
        "recorded, live",
        [
            ('"GITHUB_TOKENS=<redacted>\\nexit 0"', '"GITHUB_TOKENS=A1b2C3d4,E5f6G7h8\\nexit 0"'),
            ('"GET /api?token=<redacted>&user=bob 200"', '"GET /api?token=A1b2C3d4&admin=true&user=bob 200"'),
            ('"GET /api?token=<redacted>&key=<redacted> 200"',
             '"GET /api?token=A1b2C3d4&admin=true&key=E5f6G7h8 200"'),
        ],
    )
    def test_a_mask_stands_for_one_key(self, recorded, live):
        """Found in review's third round: a mask that stood for any text but a space or a quote took
        in a second key, an extra parameter. The patterns 0.5.0 and before masked with took a
        key's characters only."""
        from tuningfork.replay import fits

        assert not fits(recorded, live)
        assert fits(recorded, live.replace(",E5f6G7h8", "").replace("&admin=true", ""))

    def test_an_older_mask_over_an_escape_fits(self):
        # 0.5.0 read JSON written inside strings: a mask there could take a key's letter written
        # as a `\u` escape.
        from tuningfork.replay import fits

        assert fits('"key=<redacted> ok"', '"key=ab\\\\u0041cd ok"')

    def test_a_mask_hides_whole_escapes(self):
        """Found in review's fourth round: the next piece could be found inside a `\\u` escape, the
        mask before it hiding a lone backslash."""
        from tuningfork.replay import fits

        escaped = "0" + "\\" + "u0041-10"
        assert not fits("<redacted><redacted>u<redacted><redacted>", escaped)
        assert fits("<redacted>-<redacted>", escaped)
        assert not fits("<redacted>u0041<redacted>", "ab" + "\\" + "u0041c")

    def test_a_megabyte_of_a_keys_characters_is_matched_quickly(self):
        """Found in review's fifth round: every place a mask could end was kept apart, so twelve
        masks in a megabyte of a key's characters took 3 s and 210 MB."""
        from tuningfork.replay import fits

        started = time.perf_counter()
        assert fits("<redacted>" + "a<redacted>" * 11, "a" * 1_000_000)
        assert time.perf_counter() - started < 1

    def test_each_place_a_piece_may_stand_is_tried(self):
        """Found in review's fourth round: a piece looked for at its leftmost place only can cut an
        escape there, where a later place leaves it whole for the mask before."""
        from tuningfork.replay import fits

        backslash = "\\"
        assert fits("<redacted>" + backslash + "<redacted>!", "a" + backslash + "u0041" + backslash + "b!")
        # And where a mask starts inside an escape, behind another start in its run, the escape's
        # letters are places it may end — but from a backslash there, the rest of it is one escape.
        assert fits("<redacted>" + backslash + "<redacted>42<redacted>", "ab" + backslash + "u0041" + backslash + "u0042x")
        assert not fits("<redacted>" + backslash + "<redacted>042<redacted>",
                        "a" + backslash * 2 + "u0041" + backslash * 2 + "u0042Z")

    def test_a_last_mask_hides_something(self):
        # Found in review's sixth round: no test held that the last mask stands for a character or more.
        from tuningfork.replay import fits

        assert not fits("a<redacted>b", "ab")
        assert fits("a<redacted>b", "axb")

    def test_a_call_whose_masks_could_start_in_too_many_places_parts(self):
        """Found in review's sixth round: in a run of escapes a backslash between two masks may stand
        at every one, and the places multiplied: twelve masks in a megabyte took 5 s, a hundred 49 s."""
        from tuningfork.replay import fits

        backslash = "\\"
        started = time.perf_counter()
        assert not fits("<redacted>" + (backslash + "<redacted>") * 11, (backslash + "u0041") * 170_000)
        assert time.perf_counter() - started < 1

    def test_a_step_masked_by_patterns_the_store_keeps_is_matched_as_they_mask(self, store):
        """Found in review's fourth round: a mask stood for a key's characters, as 0.5.0's masks
        hid, so a step masked by today's patterns where they took punctuation would part from a
        fork once they change. The store keeps the patterns each step names, and a fork masks a
        call with them again."""
        from tests.helpers import record_pilot
        from tuningfork.config import patterns_digest
        from tuningfork.replay import build_table
        from tuningfork.wire import get_wire

        own = (r"pw=(?P<secret>\S+)",)
        run = store.get_run(record_pilot(store, read="pw=<redacted> ok\n", params={"redaction": patterns_digest(own)}))
        stored = store.rebuild_request(store.steps(run.id)[1])
        live = json.loads(json.dumps(stored).replace("<redacted>", "Pa$$w0rd!x"))
        wire = get_wire("openai-chat")
        assert build_table(store, run, 1).fitting(live, wire) is None  # its masks hide a key's characters
        store.keep_masking(patterns_digest(own), own)
        table = build_table(store, run, 1)
        assert table.fitting(live, wire) == table.live_key
        parted = json.loads(json.dumps(stored).replace("<redacted> ok", "Pa$$w0rd!x ok\\nERROR: refused"))
        assert table.fitting(parted, wire) is None

    def test_a_call_is_matched_in_one_pass(self):
        """Found in review's third round: matched as a regex, eight masks in one run of letters took
        20 s, the proxy waiting."""
        from tuningfork.replay import fits

        recorded = '"' + "<redacted>a" * 12 + '"'
        live = '"' + "a" * 50 + " " + "a" * 50 + '"'
        started = time.perf_counter()
        assert not fits(recorded, live)
        assert time.perf_counter() - started < 1

    def test_the_patterns_name_says_how_they_mask(self, monkeypatch):
        from tuningfork import config

        before = config.patterns_digest(config.DEFAULT_REDACTIONS)
        monkeypatch.setattr(config, "MASKING_VERSION", config.MASKING_VERSION + 1)
        assert config.patterns_digest(config.DEFAULT_REDACTIONS) != before

    def test_a_step_masked_as_these_patterns_mask_is_found_by_its_key_alone(self, store):
        from tuningfork.config import DEFAULT_REDACTIONS, patterns_digest
        from tests.helpers import record_pilot
        from tuningfork.replay import build_table

        stamped = {"redaction": patterns_digest(DEFAULT_REDACTIONS)}
        run = store.get_run(record_pilot(store, read=self.CODE.replace("tokenizer.encode(text)[0]", "<redacted>(text)[0]"),
                                         params=stamped))
        assert build_table(store, run, 1).masked == []

    def test_a_step_masked_as_these_patterns_mask_is_keyed_as_it_is_stored(self, store):
        """Found in review's fifth round: masking is not idempotent — one pass masks a command's
        first `-p`, a second its next — so a step keyed by masking its stored request again parted
        from a fork whose call was masked once, as the step was."""
        from tests.helpers import record_pilot
        from tuningfork.config import DEFAULT_REDACTIONS, patterns_digest
        from tuningfork.replay import build_table, key_of
        from tuningfork.wire import get_wire

        today = {"redaction": patterns_digest(DEFAULT_REDACTIONS)}
        read = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db\n"
        run = store.get_run(record_pilot(store, read=read, params=today))
        stored = store.rebuild_request(store.steps(run.id)[1])
        call = json.loads(json.dumps(stored).replace("<redacted>", "Abcd1234X"))
        table = build_table(store, run, 1)
        assert key_of(table.text(call, get_wire("openai-chat"))) == table.live_key

    def test_a_fork_goes_live_at_its_step_though_its_mask_fell_elsewhere(self, store):
        from types import SimpleNamespace

        from tuningfork.config import DEFAULT_REDACTIONS
        from tuningfork.proxy.replay import ForkReplay
        from tuningfork.wire import get_wire

        run = store.get_run(self.recorded(store))
        launch = SimpleNamespace(id="f" * 32, source_run=run.id, source_step=1, workspace=None, home=None,
                                 variant=None)
        replay = ForkReplay(store, launch, DEFAULT_REDACTIONS, ())
        wire = get_wire("openai-chat")
        first = store.rebuild_request(store.steps(run.id)[0])
        stored = store.rebuild_request(store.steps(run.id)[1])
        second = json.loads(json.dumps(stored).replace("<redacted>", "tokenizer.encode"))
        main = SimpleNamespace(main=True, thread=0, idx=0)
        assert replay.decide(first, wire, main).served is not None
        decided = replay.decide(second, wire, SimpleNamespace(main=True, thread=0, idx=1))
        assert decided.at_step and decided.mismatch is None


def test_a_setup_says_how_another_one_differs():
    """What a screen's notes say of a run it passed over (`Setup.differences`): every part of
    how a launch was set up that changes what the model reads, the clock only where both ran in
    the sandbox, and nothing of which run each was read from."""
    from tuningfork.ops.launch import Setup

    own = Setup(harness="hermes", model="m", run="a" * 32, recorded={"self_improvement": False}, layout="install",
                sandbox={"kind": "bubblewrap", "clock": "2026-10-08 09:00:00"})
    assert own.differences(Setup(harness="hermes", model="m", run="b" * 32, recorded={"self_improvement": False},
                                 layout="install", sandbox={"kind": "bubblewrap", "version": "0.12",
                                                            "clock": "2026-10-08 09:00:00"})) == []
    other = Setup(harness="claude-code", model="n", recorded={}, layout=None, sandbox={"kind": "bubblewrap"})
    assert own.differences(other) == [
        "harness claude-code, not hermes", "model n, not m", 'harness options none, not {"self_improvement": false}',
        "layout flat, not install", "its clock at the real time, not 2026-10-08 09:00:00",
    ]
    assert own.differences(Setup(harness="hermes", model="m", recorded={"self_improvement": False},
                                 layout="install")) == ["without the sandbox, not in it"]
    servers = Setup(harness="hermes", model="m", recorded={"self_improvement": False, "mcp_servers": {"tf": "x", "a": "y"}})
    assert servers.options == {"self_improvement": False} and servers.dropped_servers == ["a", "tf"]
    assert servers.to_json() == {"harness_options": {"self_improvement": False}, "dropped_servers": ["a", "tf"],
                                 "layout": None, "sandboxed": False, "clock": None}
