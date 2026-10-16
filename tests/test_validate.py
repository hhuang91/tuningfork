"""Validating a screened proposal (`ops.validate`): forks from its checkpoint to the end, fresh
full runs of the training task, and fresh full runs of each held-out task, verdicts only —
with the fake agent as the harness and the mock as the model, which names the report's keys
only when the skill in its context does. Nothing of the held-out task reaches the recordings
folder, a job's result, or its progress."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from tuningfork.analysis.validation import judge
from tuningfork.control import ControlClient
from tuningfork.ops import refine, validate
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import Cancel, Cancelled, OperationError
from tuningfork.ops.private import PREFIX
from tuningfork.ops.screen import execute_screen
from tuningfork.specs import Eval, TuningforkOptions
from tests.helpers import CHECK, task_stack

MARKER = "HELD-OUT-MARKER"
RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEYS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total."}


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch, request):
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    # A test may ask for a proxy started with a cap on output: `indirect` parametrization.
    with task_stack(tmp_path, mock_server, monkeypatch, max_output=getattr(request, "param", None)) as built:
        built.spec.evals.append(Eval(
            id="2", prompt=f"Summarize data.csv into report.json. {MARKER}",
            files=[{"path": "data.csv", "content": f"id,amount\n1,10\n2,20\n3,30\n4,1.5\n# {MARKER}\n"}],
            tuningfork=TuningforkOptions(check=CHECK, held_out=True, skills=["tally-report"], max_turns=6, timeout=60),
        ))
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        built.temp = temp
        yield built


def screened(stack, n: int = 5):
    """A run of the training task, a refinement at its write step, the keys fix proposed, and
    a complete screen it survived."""
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id="tally-report/1")
    made = refine.create(stack.store, run.id, "write_file", target_rules=RULES)
    proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads row_count",
                                   proposer={"via": "person"})
    prepared = refine.prepare_refinement_screen(stack.store, made.id, [proposal.id], base_url=stack.url, n=n)
    result = asyncio.run(execute_screen(stack.store, prepared))
    assert [arm.name for arm in result.report.survivors()] == [proposal.id[:8]]
    return made, proposal


def mentions(home, text: str) -> list[str]:
    return [str(path) for path in home.rglob("*") if path.is_file() and text.encode() in path.read_bytes()]


def verdicts(part: dict, arm: str) -> list:
    return [verdict["passed"] for verdict in part[arm]]


def test_a_proposal_is_validated_three_ways_and_the_gate_reads_the_record(stack, monkeypatch):
    monkeypatch.setattr(validate, "MINIMUM", 2)  # the floor is tested below; here, speed
    made, proposal = screened(stack)
    plan = validate.prepare_validate(stack.store, made.id[:8], proposal.id[:8], n=2, runs=2, proxy_url=stack.url)
    shown = plan.to_json()
    assert plan.total == 2 * 2 + 2 * 2 * 2 and shown["held_out"] == [{"eval": "2", "runs_per_arm": 2}]
    assert shown["screen"]["survived"] and shown["screen"]["target"]["successes"] == 5
    assert MARKER not in json.dumps(shown)

    result = validate.execute_validate(stack.store, plan)
    record = result.record
    assert record["complete"] and record["status"] == "done" and record["task"] == "tally-report/1"
    assert (verdicts(record["forward"], "original"), verdicts(record["forward"], "edit")) == ([False] * 2, [True] * 2)
    assert (verdicts(record["training"], "original"), verdicts(record["training"], "edit")) == ([False] * 2, [True] * 2)
    (held,) = record["held_out"]
    assert held["eval"] == "2" and (verdicts(held, "original"), verdicts(held, "edit")) == ([False] * 2, [True] * 2)
    assert all("server_id" in verdict for part in (record["forward"], record["training"], held)
               for arm in ("original", "edit") for verdict in part[arm])
    # The private proxy probes its server as it starts (the test's own proxy runs no lifespan).
    assert all(verdict["server_id"] for arm in ("original", "edit") for verdict in held[arm])
    gate = judge(record, target_matches=True)
    assert gate.passed, [check.reason for check in gate.failed()]
    # Kept in the refinement, as the gate will read it again.
    assert validate.validations(stack.store, made.id) == [json.loads(json.dumps(record))]
    assert (made.folder / "validations" / f"{record['id']}.json").is_file()

    # The forks are found by the validation that ran them; the training runs had each wording.
    for experiment_id in record["forward"]["experiments"].values():
        settings = stack.store.get_experiment(experiment_id).settings
        assert settings["validation"] == record["id"] and settings["refinement"] == made.id
    for arm, text in (("original", made.baseline_text()), ("edit", proposal.text())):
        for run_id in record["training"]["runs"][arm]:
            launch = stack.store.get_launch(stack.store.get_run(run_id).launch_id)
            assert (Path(launch.skill_dir) / "SKILL.md").read_text(encoding="utf-8") == text

    # What a job would say: counts for the forward and training parts, the held-out task by id.
    answer = result.to_json()
    assert answer["held_out"] == ["2"] and answer["training"]["edit"]["passed"] == answer["training"]["edit"]["planned"] == 2
    assert MARKER not in json.dumps(answer) and "original" not in json.dumps(answer["held_out"])
    # Nothing of the held-out task is in the recordings folder, nor left in the temp folder.
    assert stack.store.runs(limit=500, task_id="tally-report/2") == [] and stack.store.held_out_tasks() == set()
    assert mentions(stack.store.home, MARKER) == []
    assert list(stack.temp.glob(f"{PREFIX}*")) == []


def test_refused_before_anything_runs_when_what_it_would_measure_has_changed(stack):
    made, proposal = screened(stack)
    runs = len(stack.store.runs(limit=500))

    def refused(**kwargs) -> OperationError:
        with pytest.raises(OperationError) as error:
            validate.prepare_validate(stack.store, made.id, kwargs.pop("proposal", proposal.id),
                                      proxy_url=stack.url, **kwargs)
        return error.value

    assert (refused(n=4).field, refused(runs=4).field) == ("n", "runs")
    other = refine.add_proposal(stack.store, made.id, text=made.baseline_text().replace(
        KEYS["old"], "Write the totals to report.json, compactly."), rationale="r", proposer={"via": "person"},
        strict=False)
    assert refused(proposal=other.id).code == "not_screened"

    source = stack.tmp / "skills-src" / "tally-report" / "SKILL.md"
    original = source.read_bytes()
    source.write_text(source.read_text(encoding="utf-8") + "\nOne more line.\n", encoding="utf-8")
    assert refused().code == "drifted"
    source.write_bytes(original.replace(b"\n", b"\r\n"))  # line endings alone are no change
    validate.prepare_validate(stack.store, made.id, proposal.id, proxy_url=stack.url)
    source.write_bytes(original)

    extra = source.parent / "notes.md"
    extra.write_text("a file the runs never had\n", encoding="utf-8")
    assert refused().code == "drifted" and "other files" in refused().message
    extra.unlink()

    spec = json.loads(stack.spec.path.read_text(encoding="utf-8"))
    spec["evals"][0]["prompt"] += " And be quick."
    stack.spec.path.write_text(json.dumps(spec), encoding="utf-8")
    changed = refused()
    assert changed.code == "drifted" and changed.field == "spec" and "tally-report/1" in changed.message
    spec["evals"][0]["prompt"] = spec["evals"][0]["prompt"].removesuffix(" And be quick.")
    stack.spec.path.write_text(json.dumps(spec), encoding="utf-8")
    validate.prepare_validate(stack.store, made.id, proposal.id, proxy_url=stack.url)

    refine.close(stack.store, made.id)
    assert refused().code == "closed"
    assert len(stack.store.runs(limit=500)) == runs


def test_a_spec_with_no_held_out_task_is_warned_and_can_never_pass_the_gate(stack):
    made, proposal = screened(stack)
    spec = json.loads(stack.spec.path.read_text(encoding="utf-8"))
    spec["evals"] = spec["evals"][:1]
    stack.spec.path.write_text(json.dumps(spec), encoding="utf-8")
    plan = validate.prepare_validate(stack.store, made.id, proposal.id, proxy_url=stack.url)
    assert plan.held_out == [] and plan.total == 4 * 5
    assert any("no held-out eval" in warning for warning in plan.warnings)


@pytest.mark.parametrize("stack", [50], indirect=True)
def test_the_held_out_runs_are_held_to_the_proxys_cap(stack, monkeypatch):
    """Every other run here goes through the proxy, held to its cap on output; the held-out runs
    go through a private proxy, which the plan gives the same cap."""
    from tuningfork.ops import private

    monkeypatch.setattr(validate, "MINIMUM", 1)
    made, proposal = screened(stack)
    plan = validate.prepare_validate(stack.store, made.id, proposal.id, n=1, runs=1, proxy_url=stack.url)
    assert plan.max_output == 50 and plan.to_json()["private"]["max_output"] == 50
    asked = []
    monkeypatch.setattr(private, "run_verdict_only", lambda *args, **kwargs: asked.append(kwargs["max_output"]) or [])
    validate.execute_validate(stack.store, plan)
    assert asked == [50, 50]  # the held-out task's two arms


def test_cancelled_it_keeps_what_finished_and_gives_no_verdict(stack, monkeypatch):
    monkeypatch.setattr(validate, "MINIMUM", 1)
    made, proposal = screened(stack)
    plan = validate.prepare_validate(stack.store, made.id, proposal.id, n=1, runs=1, proxy_url=stack.url)
    cancel = Cancel()

    def watch(event):
        if event.kind == "validation" and event.data.get("part") == "held_out":
            cancel.cancel()

    with pytest.raises(Cancelled) as stopped:
        validate.execute_validate(stack.store, plan, progress=watch, cancel=cancel)
    record = stopped.value.partial.record
    assert record["status"] == "cancelled" and not record["complete"] and record["held_out"] == []
    assert record["forward"] and record["training"]  # what finished is kept
    assert validate.validations(stack.store, made.id)[0]["status"] == "cancelled"
    assert not judge(record, target_matches=True).passed
    assert list(stack.temp.glob(f"{PREFIX}*")) == [] and mentions(stack.store.home, MARKER) == []


def test_as_a_job_every_run_counts_and_no_held_out_verdict_shows(stack):
    made, proposal = screened(stack)
    params = {"refinement": made.id[:8], "proposal": proposal.id[:8]}
    events = []
    with ControlClient(stack.url) as client:
        planned = client.plan("validate", params)
        job = client.submit("validate", params)
        done = client.wait_for_job(job["id"], timeout=240, poll=0.2, on_event=events.append)
    assert planned["total"] == 30 and planned["plan"]["held_out"] == [{"eval": "2", "runs_per_arm": 5}]
    assert done["status"] == "done", done["error"]
    assert done["progress"]["done"] == done["progress"]["total"] == 30
    assert done["result"]["held_out"] == ["2"] and done["result"]["complete"]
    held = [entry["event"] for entry in events if entry["event"]["kind"] == "held_out_run"]
    assert len(held) == 10 and all(set(event["data"]) <= {"label", "index"} for event in held)
    assert MARKER not in json.dumps(done) + json.dumps(events)
    (record,) = validate.validations(stack.store, made.id)
    assert judge(record, target_matches=True).passed
    assert mentions(stack.store.home, MARKER) == []


def a_record(made, proposal) -> dict:
    """A finished validation as `execute_validate` keeps it, one held-out verdict marked."""
    yes = {"passed": True, "exclusion": None, "timed_out": False, "server_id": "s1"}
    no = {**yes, "passed": False}
    secret = {**yes, "passed": None, "exclusion": "SECRET-EXCLUSION"}
    return {
        "id": "f" * 32, "refinement": made.id, "proposal": proposal.id, "created_at": 1.0,
        "task": "tally-report/1", "harness": "fake", "model": "m", "alpha": 0.05, "minimum": 5,
        "planned": {"forward": 5, "training": 5, "held_out": 5}, "rules": [],
        "screen": {"survived": True, "screen": "a" * 32}, "held_out_recorded": [],
        "forward": {"original": [no] * 5, "edit": [yes] * 5},
        "training": {"original": [no] * 5, "edit": [yes] * 5},
        "held_out": [{"eval": "2", "original": [no] * 5, "edit": [yes] * 5 + [secret]}],
        "complete": True, "status": "done",
    }


def test_only_the_persons_views_show_held_out_verdicts_and_the_gate(stack):
    import httpx

    from tuningfork.ops.packet import build_packet

    made, proposal = screened(stack)
    folder = made.folder / "validations"
    folder.mkdir()
    (folder / f"{'f' * 32}.json").write_text(json.dumps(a_record(made, proposal)), encoding="utf-8")

    (full,) = refine.view(stack.store, made.id, held_out=True)["validations"]
    assert full["held_out"][0]["edit"][-1]["exclusion"] == "SECRET-EXCLUSION"
    assert full["gate"]["passed"], [check["reason"] for check in full["gate"]["checks"] if not check["passed"]]
    plain = refine.view(stack.store, made.id)
    assert "SECRET" not in json.dumps(plain) and "gate" not in json.dumps(plain["validations"])
    assert plain["validations"][0]["held_out"] == ["2"] and plain["validations"][0]["forward"]["edit"]["passed"] == 5
    assert "SECRET" not in json.dumps(build_packet(stack.store, made.id, keep=False).data)
    # The person's page reads them all; it is theirs.
    with httpx.Client(base_url=stack.url, timeout=30) as http:
        page = http.get(f"/_tuningfork/api/refinements/{made.id}").json()
        listed = http.get(f"/_tuningfork/api/refinements/{made.id[:8]}/validations").json()
    assert page["validations"] == listed and listed[0]["held_out"][0]["edit"][-1]["exclusion"] == "SECRET-EXCLUSION"

    # The skill's own file changed: the gate says so, whatever the runs said.
    source = stack.tmp / "skills-src" / "tally-report" / "SKILL.md"
    source.write_text(source.read_text(encoding="utf-8") + "\nOne more line.\n", encoding="utf-8")
    (full,) = refine.view(stack.store, made.id, held_out=True)["validations"]
    (failed,) = [check for check in full["gate"]["checks"] if not check["passed"]]
    assert failed["id"] == "G6" and "has changed" in failed["reason"]


def test_an_agent_reading_the_refinement_sees_no_held_out_verdict(stack):
    pytest.importorskip("mcp")
    from tuningfork.mcp.server import ToolServer
    from tuningfork.mcp.tools import registry
    from tests.test_mcp_tools import NoJobs

    made, proposal = screened(stack)
    folder = made.folder / "validations"
    folder.mkdir()
    (folder / f"{'f' * 32}.json").write_text(json.dumps(a_record(made, proposal)), encoding="utf-8")
    server = ToolServer(store=stack.store, jobs=NoJobs(), toolset="refine")
    answer = asyncio.run(server.call(registry("refine")["show_refinement"], {"refinement": made.id[:8]}))
    text = json.dumps(answer.data) + "\n".join(answer.lines)
    assert "SECRET" not in text and "gate" not in json.dumps(answer.data)
    (shown,) = answer.data["validations"]
    assert shown["held_out"] == ["2"] and shown["training"]["edit"]["successes"] == 5
    assert any("the person's to read" in line for line in answer.lines)


def test_a_validation_whose_process_died_reads_as_interrupted(stack):
    made, proposal = screened(stack)
    record = {**a_record(made, proposal), "status": "running", "complete": False, "pid": 999999999}
    folder = made.folder / "validations"
    folder.mkdir()
    (folder / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")
    (shown,) = validate.validations(stack.store, made.id)
    assert shown["status"] == "interrupted" and not judge(shown, target_matches=True).passed
