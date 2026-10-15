"""Batches as an operation: everything decided before a run is launched, the same for the
CLI, the web UI's jobs and later MCP.

End to end with no model: `FakeAgent` is a real harness process talking to a real proxy,
and the mock server plays the model (see `tests/helpers.task_stack`).
"""

from __future__ import annotations

import json
import threading

import pytest

from tuningfork.harness import ADAPTERS
from tuningfork.ops import Cancel, Cancelled, OperationError
from tuningfork.ops.batch import execute_batch, prepare_batch
from tests.helpers import task_stack


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def prepare(stack, **options):
    options.setdefault("model", "m")
    options.setdefault("harness", "fake")
    return prepare_batch(
        stack.store, stack.spec.path, base_url=f"{stack.url}/v1",
        workdir=stack.tmp / "batches", **options,
    )


def launches(store) -> int:
    return store._conn.execute("SELECT count(*) FROM launches").fetchone()[0]


class TestPrepare:
    def test_the_plan_says_what_will_run_and_shows_the_check_whole(self, stack):
        plan = prepare(stack, n=2)
        data = json.loads(json.dumps(plan.to_json()))
        assert data["task_id"] == "tally-report/1" and data["n"] == 2
        assert data["check"] == stack.item.tuningfork.check
        assert data["notes"] == []
        assert plan.parity["model"] == "m" and plan.parity["harness"] == "fake"
        assert launches(stack.store) == 0

    def test_the_model_defaults_to_the_one_the_server_lists(self, stack):
        stack.app.state.server = {"id": "s", "models": ["gemma"]}
        assert prepare(stack, model=None).model == "gemma"

    @pytest.mark.parametrize("models", [["a", "b"], []])
    def test_with_several_models_or_none_the_model_must_be_named(self, stack, models):
        stack.app.state.server = {"id": "s", "models": models}
        with pytest.raises(OperationError) as refused:
            prepare(stack, model=None)
        assert refused.value.field == "model"
        assert all(name in refused.value.hint for name in models)

    def test_a_missing_input_file_is_refused_before_any_launch(self, stack):
        data = stack.spec.to_dict()
        data["evals"][0]["files"].append("not-there.csv")
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError, match="not-there.csv") as refused:
            prepare(stack)
        assert refused.value.field == "spec"
        assert launches(stack.store) == 0

    def test_a_skill_folder_that_is_not_there_is_refused_before_any_launch(self, stack):
        """The runner would copy nothing, and every run would go without the skill."""
        data = stack.spec.to_dict()
        data["tuningfork"]["skill_dir"] = "no-such-skill"
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            prepare(stack)
        assert refused.value.field == "spec"
        assert "no-such-skill" in refused.value.message and "does not exist" in refused.value.message
        assert "resolved against the spec's folder" in refused.value.hint
        assert launches(stack.store) == 0

    def test_a_spec_with_problems_names_them(self, stack):
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["held_out"] = "false"
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError, match=r"evals\[0\]\.tuningfork\.held_out") as refused:
            prepare(stack)
        assert refused.value.field == "spec"

    def test_what_a_person_should_know_comes_as_notes(self, stack):
        (stack.tmp / "skills-src" / "tally-report" / "SKILL-v2.md").write_text("x")
        notes = prepare(stack, concurrency=9).notes
        assert [note.level for note in notes] == ["warning", "note"]
        assert notes[0].message == (
            "the skill folder also holds SKILL-v2.md. Every run is given a copy of the whole "
            "folder, so the model can read it. Keep variants outside the skill folder."
        )
        assert notes[1].message.startswith("the proxy lets 4 call(s) through at once, so 9 runs")

    def test_a_held_out_eval_runs_only_when_asked_for_by_name(self, stack):
        """It judges edits: never the default, and never shown whole in a plan."""
        data = stack.spec.to_dict()
        data["evals"].insert(0, {"id": "9", "prompt": "the held-out task's own prompt", "files": [],
                                 "tuningfork": {"held_out": True}})
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        plan = prepare(stack)
        assert plan.task_id == "tally-report/1"
        shown = json.dumps(plan.to_json())
        assert "the held-out task's own prompt" not in shown and plan.to_json()["held_out"] == ["9"]
        with pytest.raises(OperationError) as refused:
            prepare(stack, eval_id="9")
        assert refused.value.code == "held_out" and "refine validate" in refused.value.hint
        included = prepare(stack, eval_id="9", include_held_out=True)
        assert included.task_id == "tally-report/9"
        assert any("held out" in note.message for note in included.notes if note.level == "warning")
        assert launches(stack.store) == 0

    def test_a_spec_whose_evals_are_all_held_out_names_none_by_default(self, stack):
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["held_out"] = True
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError, match="every eval") as refused:
            prepare(stack)
        assert refused.value.code == "held_out"

    def test_a_harness_that_cannot_start_is_refused_before_any_launch(self, stack, monkeypatch):
        class Missing:
            name = "ghost"
            executable = "tuningfork-no-such-harness"

        monkeypatch.setitem(ADAPTERS, "ghost", Missing)
        with pytest.raises(OperationError, match="was not found") as refused:
            prepare(stack, harness="ghost")
        assert refused.value.field == "harness"
        with pytest.raises(OperationError, match="headlessly"):
            prepare(stack, harness="generic")
        with pytest.raises(OperationError, match="unknown harness"):
            prepare(stack, harness="nope")
        assert launches(stack.store) == 0


class TestExecute:
    def test_a_batch_reports_its_runs_as_they_go_and_its_rate(self, stack):
        seen = []
        report = execute_batch(stack.store, prepare(stack, n=2), progress=seen.append)

        assert [event.kind for event in seen].count("launch") == 2
        runs = [event for event in seen if event.kind == "run"]
        assert sorted(event.data["index"] for event in runs) == [0, 1]
        # The original wording: the report is written with the wrong keys, every time.
        assert report.rate.successes == 0 and report.rate.total == 2
        assert report.divergence is not None and not report.divergence.diverged
        data = json.loads(json.dumps(report.to_json()))
        assert data["rate"]["total"] == 2 and data["not_counted"] == 0
        assert [run["index"] for run in data["runs"]] == [0, 1]
        assert data["divergence"]["run_ids"] == report.run_ids

    def test_a_plan_lists_the_specs_tasks_by_what_each_asks(self, stack):
        """Found in use: the form's "Eval" field, hinted "the first", said neither what an
        eval was nor which there were. A form now offers the spec's tasks by their prompts."""
        data = stack.spec.to_dict()
        data["evals"].append({**data["evals"][0], "id": "2", "prompt": "Do it the other way."})
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")

        plan = prepare(stack, n=1).to_json()

        assert [item["id"] for item in plan["evals"]] == [data["evals"][0]["id"], "2"]
        assert plan["evals"][1]["prompt"] == "Do it the other way."
        assert plan["eval"] == data["evals"][0]["id"]

    def test_a_batch_can_be_judged_on_the_agents_final_reply(self, stack):
        """End to end: the fake agent's last turn says "The report is written." — recorded by
        the proxy, and judged by the check, with no shell command involved."""
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["check"] = {"reply_contains": "report is written"}
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")

        plan = prepare(stack, n=1)
        assert plan.to_json()["check_kind"] == "reply_contains"
        assert plan.to_json()["check_text"] == 'the reply contains "report is written"'
        report = execute_batch(stack.store, plan)

        assert report.rate.successes == 1 and report.rate.total == 1
        (outcome,) = report.outcomes
        stored = stack.store.get_run(outcome.run_id).result
        assert stored["check_output"].startswith('the reply contains "report is written"')

    def test_a_cancelled_batch_keeps_what_finished_and_judges_nothing_it_killed(self, stack):
        live = stack.mock.responder

        def slow(payload):  # the second run's write takes long enough to be killed
            turn = live(payload)
            if len([m for m in payload["messages"] if m["role"] == "tool"]) == 1:
                slow.calls += 1
                if slow.calls == 2:
                    turn.stall = 30
            return turn

        slow.calls = 0
        stack.mock.responder = slow
        cancel = Cancel()

        def progress(event):
            if event.kind == "run":
                threading.Timer(0.5, cancel.cancel).start()

        with pytest.raises(Cancelled) as stopped:
            execute_batch(stack.store, prepare(stack, n=3), progress=progress, cancel=cancel)

        partial = stopped.value.partial
        assert [outcome.index for outcome in partial.outcomes] == [0]
        assert partial.rate.total == 1
        statuses = sorted(
            row["status"] for row in stack.store._conn.execute("SELECT status FROM launches")
        )
        assert statuses == ["cancelled", "ended"]


def test_the_cli_prints_the_plan_and_the_rate(stack, monkeypatch):
    from rich.console import Console
    from typer.testing import CliRunner

    from tuningfork.cli import app

    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    stack.app.state.server = {"id": "s", "models": ["m"]}
    result = CliRunner().invoke(app, [
        "batch", str(stack.spec.path), "-n", "1", "--harness", "fake",
        "--base-url", f"{stack.url}/v1", "--workdir", str(stack.tmp / "batches"),
        "--home", str(stack.store.home),
    ])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "tally-report/1 — 1 run"
    assert lines[2] == f"  check:  {stack.item.tuningfork.check}"
    assert any(line.startswith("  run 0  fail") for line in lines)
    assert "0/1" in result.output and "passed" in result.output


class TestHarnessOptionsAndSkillFolder:
    def test_options_are_refused_for_a_harness_that_takes_none(self, stack):
        with pytest.raises(OperationError) as refused:
            prepare(stack, harness_options={"await_review": True})
        assert refused.value.field == "harness_options" and "takes no harness options" in refused.value.message
        assert launches(stack.store) == 0

    def test_a_servers_headers_and_environment_are_shown_masked(self):
        from tuningfork.ops.batch import shown_options

        given = {"self_improvement": False, "mcp_servers": {
            "tf": {"url": "http://127.0.0.1:1/_tuningfork/mcp", "headers": {"Authorization": "Bearer s3cret"}, "timeout": 300},
            "gh": {"command": "npx", "args": ["-y", "x"], "env": {"TOKEN": "ghp_x"}},
        }}
        shown = shown_options(given)
        assert shown["mcp_servers"]["tf"] == {"url": "http://127.0.0.1:1/_tuningfork/mcp", "headers": {"Authorization": "…"}, "timeout": 300}
        assert shown["mcp_servers"]["gh"] == {"command": "npx", "args": ["-y", "x"], "env": {"TOKEN": "…"}}
        assert shown["self_improvement"] is False and "s3cret" not in json.dumps(shown)
        # What was given is untouched: the launch is given the real header.
        assert given["mcp_servers"]["tf"]["headers"]["Authorization"] == "Bearer s3cret"
        assert shown_options(None) == {} and shown_options({"await_review": True}) == {"await_review": True}

    def test_a_skill_folder_other_than_the_specs_can_be_given(self, stack):
        other = stack.tmp / "other"
        other.mkdir()
        (other / "SKILL.md").write_text("# other\n", encoding="utf-8")
        plan = prepare(stack, skill_dir=other)
        assert plan.skill_dir == other and plan.to_json()["skill_dir"] == str(other)
        assert plan.to_json()["harness_options"] == {}
        with pytest.raises(OperationError) as refused:
            prepare(stack, skill_dir=stack.tmp / "missing")
        assert refused.value.field == "skill_dir" and "does not exist" in refused.value.message
        assert launches(stack.store) == 0
