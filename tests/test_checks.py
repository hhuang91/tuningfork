"""Trying a check before trusting it: a recorded run judged again — its workspace as it
ended, and its final reply — or a reply and files written here. In a throwaway folder, with
nothing recorded: a run's own verdict never changes.

End to end with no model: `FakeAgent` is a real harness process, so the run it leaves has a
snapshotted workspace and a recorded final reply ("The report is written.").
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tuningfork.ops import OperationError
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.checks import try_check
from tests.helpers import record_pilot, task_stack


@pytest.fixture
def launched(tmp_path, mock_server, monkeypatch):
    """A stack, and one run of its task that tuningfork launched: it wrote the report with
    the wrong keys, so its own check failed it."""
    with task_stack(tmp_path, mock_server, monkeypatch) as stack:
        stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
        plan = prepare_batch(
            stack.store, stack.spec.path, n=1, model="m", harness="fake",
            base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
        )
        (outcome,) = execute_batch(stack.store, plan).outcomes
        yield stack, outcome.run_id


def counts(store) -> tuple[int, ...]:
    return tuple(
        store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("runs", "steps", "experiments", "samples", "launches")
    )


class TestAgainstARun:
    def test_a_run_is_judged_again_by_its_own_check(self, launched):
        stack, run_id = launched
        trial = try_check(stack.store, run=run_id)
        assert trial.plan.own
        assert trial.passed is False and trial.agrees is True
        assert trial.plan.reply == "The report is written."
        assert {"data.csv", "report.json"} <= set(trial.found)

    def test_a_check_can_disagree_with_the_verdict_the_run_got(self, launched):
        """What trying is for: this check would have passed the run its own check failed."""
        stack, run_id = launched
        lenient = "python3 -c \"import json,sys; sys.exit(0 if json.load(open('report.json')).get('count') == 4 else 1)\""
        trial = try_check(stack.store, lenient, run=run_id)
        assert trial.passed is True and trial.agrees is False
        assert trial.to_json()["recorded"]["passed"] is False

    def test_the_reply_is_judged_directly_or_read_from_its_file(self, launched):
        stack, run_id = launched
        assert try_check(stack.store, {"reply_contains": "report is written"}, run=run_id).passed
        assert try_check(stack.store, {"reply_matches": r"^The report"}, run=run_id).passed
        assert try_check(stack.store, 'grep -q "report is written" "$TUNINGFORK_REPLY"', run=run_id).passed

    def test_a_check_finds_the_folder_of_the_spec_the_run_came_from(self, launched):
        stack, run_id = launched
        here = stack.spec.path.parent.resolve()
        trial = try_check(stack.store, f'test "$TUNINGFORK_SPEC_DIR" = "{here}"', run=run_id)
        assert trial.passed, trial.output

    def test_a_fork_finds_the_spec_folder_its_eval_keeps(self, launched):
        """Found in use: a fork's launch names no spec file, so trying its check ran with
        `$TUNINGFORK_SPEC_DIR` empty and could not find a check script kept beside the spec."""
        stack, run_id = launched
        launch_id = stack.store.get_run(run_id).launch_id
        stack.store._conn.execute("UPDATE launches SET spec_path = NULL WHERE id = ?", (launch_id,))
        here = stack.spec.path.parent.resolve()
        trial = try_check(stack.store, f'test "$TUNINGFORK_SPEC_DIR" = "{here}"', run=run_id)
        assert trial.passed, trial.output

    def test_nothing_is_recorded_and_the_verdict_stays(self, launched):
        stack, run_id = launched
        before, result = counts(stack.store), stack.store.get_run(run_id).result
        try_check(stack.store, "true", run=run_id)
        assert counts(stack.store) == before
        assert stack.store.get_run(run_id).result == result

    def test_a_command_runs_where_a_runs_check_does(self, launched, monkeypatch):
        """In the sandbox a launch's check runs in, its task's folder shown to it: a tried
        command can run what the model left in the workspace as much as the task's own can."""
        import tuningfork.sandbox

        seen: dict = {}

        class Recorded:
            def check_command(self, command, *, workspace, environment, reply_file=None, spec_dir=None):
                seen.update(command=command, spec_dir=spec_dir, reply_file=reply_file)
                return ["/bin/sh", "-c", command], environment

        monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide: (Recorded(), None))
        stack, run_id = launched
        trial = try_check(stack.store, "test -f report.json", run=run_id)
        assert trial.passed, trial.output
        assert trial.to_json()["sandboxed"] is True and "unsandboxed" not in trial.to_json()
        assert seen["command"] == "test -f report.json" and seen["reply_file"].name == "reply.txt"
        assert seen["spec_dir"].resolve() == stack.spec.path.parent.resolve()
        # A reply check runs nothing, so it needs no sandbox.
        assert try_check(stack.store, {"reply_contains": "written"}, run=run_id).to_json()["sandboxed"] is False


def test_a_check_reads_the_variables_its_task_names_and_no_others(tmp_path, mock_server, monkeypatch):
    """An LLM judge's key, say: named in the task (`check_env`), it reaches the batch's check
    and a check tried on the run; a variable the task does not name reaches neither."""
    monkeypatch.setenv("JUDGE_KEY", "for-the-judge")
    monkeypatch.setenv("CLOUD_API_KEY", "sk-not-for-a-check")
    with task_stack(tmp_path, mock_server, monkeypatch) as stack:
        stack.item.tuningfork.check = 'test "$JUDGE_KEY" = for-the-judge && test -z "${CLOUD_API_KEY:-}"'
        stack.item.tuningfork.check_env = ["JUDGE_KEY"]
        stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
        plan = prepare_batch(
            stack.store, stack.spec.path, n=1, model="m", harness="fake",
            base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
        )
        (outcome,) = execute_batch(stack.store, plan).outcomes
        assert outcome.passed, outcome.check.output
        trial = try_check(stack.store, run=outcome.run_id)
        assert trial.plan.check_env == ["JUDGE_KEY"] and trial.passed, trial.output
        # A check written here, not the task's, gets what the task lets its check read.
        assert try_check(stack.store, 'test "$JUDGE_KEY" = for-the-judge', run=outcome.run_id).passed
        assert not try_check(stack.store, 'test -n "${CLOUD_API_KEY:-}"', run=outcome.run_id).passed


class TestWrittenHere:
    def test_a_reply_and_files(self, store):
        check = 'test "$(cat out/report.json)" = "{}" && grep -q "the total" "$TUNINGFORK_REPLY"'
        trial = try_check(store, check, reply="Here is the total.", files=[{"path": "out/report.json", "content": "{}"}])
        assert trial.passed and trial.returncode == 0
        assert trial.found == ["out/report.json"]
        assert trial.agrees is None

    def test_a_reply_check_says_what_it_judged(self, store):
        trial = try_check(store, {"reply_contains": "61.5"}, reply="Done: 4 rows")
        assert not trial.passed and trial.returncode is None
        assert trial.output.startswith('the reply does not contain "61.5"')


class TestRefusals:
    @pytest.mark.parametrize(
        ("options", "field"),
        [
            ({"check": {"reply_matches": "(unclosed"}}, "check"),
            ({"check": {"reply_equals": "x"}}, "check"),
            ({"check": None}, "check"),
            ({"check": "  "}, "check"),
            ({"check": "true", "files": [{"path": "../escape.txt", "content": "x"}]}, "files"),
            ({"check": "true", "files": [{"path": "/etc/passwd", "content": "x"}]}, "files"),
        ],
    )
    def test_what_cannot_be_tried_is_refused_naming_the_field(self, store, options, field):
        with pytest.raises(OperationError) as refused:
            try_check(store, options.pop("check"), **options)
        assert refused.value.field == field

    def test_a_command_needs_a_workspace_the_run_kept(self, store):
        """A run recorded without a launch has no snapshot to run a command in; its reply
        can still be judged."""
        run_id = record_pilot(store)
        with pytest.raises(OperationError) as refused:
            try_check(store, "true", run=run_id)
        assert refused.value.field == "run" and "kept no workspace" in refused.value.message
        assert try_check(store, {"reply_contains": "x"}, run=run_id).passed is False

    def test_a_run_without_a_launch_has_no_check_of_its_own(self, store):
        with pytest.raises(OperationError) as refused:
            try_check(store, run=record_pilot(store))
        assert refused.value.field == "check"

    def test_files_are_for_a_trial_without_a_run(self, launched):
        stack, run_id = launched
        with pytest.raises(OperationError) as refused:
            try_check(stack.store, "true", run=run_id, files=[{"path": "a", "content": "b"}])
        assert refused.value.field == "files"


def test_a_trial_runs_as_a_job_through_the_proxy(launched):
    """What the scenario editor does: a dry run shows what will run, then the job runs it."""
    from tuningfork.control import ControlClient

    stack, run_id = launched
    with ControlClient(stack.url) as client:
        plan = client.plan("check-trial", {"check": {"reply_contains": "written"}, "run": run_id})["plan"]
        assert plan["check_kind"] == "reply_contains" and plan["run"]["id"] == run_id
        assert plan["recorded"]["passed"] is False
        job = client.submit("check-trial", {"check": {"reply_contains": "written"}, "run": run_id})
        done = client.wait_for_job(job["id"], timeout=60, poll=0.1)
    assert done["status"] == "done", done["error"]
    assert done["result"]["passed"] is True and done["result"]["agrees"] is False


def test_recheck_prints_the_verdict_beside_the_runs(launched):
    from tuningfork.cli import app

    stack, run_id = launched
    home = str(stack.store.home)
    own = CliRunner().invoke(app, ["recheck", run_id, "--home", home])
    assert own.exit_code == 1, own.output
    assert "(the run's own)" in own.output and "fail — as when it ran" in own.output
    other = CliRunner().invoke(app, ["recheck", run_id, "--reply-contains", "written", "--home", home])
    assert other.exit_code == 0, other.output
    assert "pass — the run was judged fail when it ran" in other.output
    sample = CliRunner().invoke(app, ["recheck", "--reply", "no", "--reply-contains", "yes", "--home", home])
    assert sample.exit_code == 1 and "a reply and files you gave" in sample.output


def test_recheck_says_when_the_check_cannot_run_in_the_sandbox(launched, monkeypatch):
    import tuningfork.sandbox
    from tuningfork.cli import app

    stack, run_id = launched
    home = str(stack.store.home)
    monkeypatch.delenv(tuningfork.sandbox.SWITCH, raising=False)
    monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide: (None, "bubblewrap is not installed"))
    shown = CliRunner().invoke(app, ["recheck", run_id, "--home", home])
    assert "not sandboxed: bubblewrap is not installed" in shown.output
    # Turned off by the person, it is said nowhere, as a batch's plan says nothing then.
    monkeypatch.setenv(tuningfork.sandbox.SWITCH, "off")
    quiet = CliRunner().invoke(app, ["recheck", run_id, "--home", home])
    assert "not sandboxed" not in quiet.output
