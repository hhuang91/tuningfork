"""The tally of what the checks said: every failed run's check output read at once and counted, the
most common finding first — by condition for a task's runs, by arm for a screen's forks run to the
end — for a person (the CLI, the API) and for an agent (MCP), never of a held-out task."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.ops import refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.ops.screen import execute_screen_to_end
from tuningfork.ops.tally import finding_key, findings, tally_screen, tally_task
from tuningfork.ops.trash import trash
from tuningfork.specs import CHECK_KEPT
from tests.helpers import record_run, task_stack, text_message

API = "/_tuningfork/api"
TAGLINE = "below full marks: tagline 0"
OWNERS = "below full marks: owners 0.5"


def judged(store, label, task="t/1", *, passed, output="", skill="v1", usable=True):
    """A run of `task` with a verdict, as a batch stores one: its check's output, under a skill version."""
    run_id = record_run(store, label, [text_message("done")], task_id=task)
    store.annotate_run(run_id, harness="hermes", skill_hashes={"SKILL.md": skill},
                       result={"passed": passed, "check_output": output, "usable": usable})
    return run_id


def meeting(store, task="t/1"):
    """Four failed runs of one wording — the tagline missed three times, the owners twice, one check
    silent — a passing run, a timed-out one, and a failed run of another wording."""
    ids = [
        judged(store, "a", task, passed=False, output=f"{TAGLINE}\n{OWNERS}"),
        judged(store, "b", task, passed=False, output=TAGLINE),
        judged(store, "c", task, passed=False, output=f"below full marks: owners 0\n{TAGLINE}"),
        judged(store, "d", task, passed=False, output=""),
        judged(store, "e", task, passed=True, output="ok"),
        judged(store, "f", task, passed=False, output="timed out", usable=False),
        judged(store, "g", task, passed=False, output="below full marks: other 0", skill="v2"),
    ]
    return ids


def test_a_finding_is_a_line_with_its_numbers_masked():
    said = findings(f"{OWNERS}\n{TAGLINE}\n\n  ok \nbelow full marks: owners 0")
    assert list(said) == ["below full marks: owners #", "below full marks: tagline #"]
    assert said["below full marks: owners #"] == OWNERS
    assert findings("") == {} and findings(None) == {}
    assert finding_key("row_count is  5, expected 4.0") == "row_count is #, expected #"
    # A colour is masked whole: two shades are one finding, not a key for each run of digits.
    hover = "component button-primary-hover: <color> on <color> is #:#, below WCAG AA's #:#"
    assert finding_key("component button-primary-hover: #333333 on #EA580C is 3.55:1, below WCAG AA's 4.5:1") == hover
    assert finding_key("component button-primary-hover: #2D2D2D on #ea580c is 3.87:1, below WCAG AA's 4.5:1") == hover
    assert finding_key("#FFF on #abc, issue #123") == "<color> on <color>, issue ##"


def test_a_tasks_failed_runs_are_counted_the_most_common_first(store):
    ids = meeting(store)
    first, second = tally_task(store, "t/1").to_json()["groups"]
    # Counted runs only: the timed-out one describes neither wording.
    assert (first["judged"], first["failed"], first["silent"]) == (5, 4, 1)
    assert [(found["finding"], found["runs"]) for found in first["findings"]] == [
        ("below full marks: tagline #", 3), ("below full marks: owners #", 2)]
    tagline = first["findings"][0]
    assert tagline["example"] == TAGLINE and sorted(tagline["run_ids"]) == sorted(ids[:3])
    assert tagline["rate"]["successes"] == 3 and tagline["rate"]["total"] == 4 and 0 < tagline["rate"]["low"] < 0.75
    assert first["condition"]["skill_version"] and first["passed"]["successes"] == 1
    # Another wording is another condition: a finding pooled across wordings describes neither.
    assert (second["failed"], [found["finding"] for found in second["findings"]]) == (1, ["below full marks: other #"])
    # A run in the trash is out of every count.
    trash(store, "run", ids[1])
    again = tally_task(store, "t/1").to_json()["groups"][0]
    assert {found["finding"]: found["runs"] for found in again["findings"]} == {
        "below full marks: tagline #": 2, "below full marks: owners #": 2}
    assert again["failed"] == 3


def test_a_check_output_kept_from_its_end_is_not_counted_from_the_middle_of_a_line(store):
    # A verdict keeps the end of its check's output. Found in review: cut there in the middle
    # of a line, the line's end was counted as a finding of its own.
    printed = "\n".join(f"booked twice: delivery {n} on day {n}" for n in range(200))
    assert printed[-CHECK_KEPT - 1] != "\n"  # the cut falls inside a line
    judged(store, "a", passed=False, output=printed[-CHECK_KEPT:])
    judged(store, "b", passed=False, output="booked twice: delivery 1 on day 1")
    (group,) = tally_task(store, "t/1").to_json()["groups"]
    assert [(found["finding"], found["runs"]) for found in group["findings"]] == [("booked twice: delivery # on day #", 2)]


def test_what_is_not_there_is_refused(store):
    with pytest.raises(OperationError) as missing:
        tally_task(store, "t/9")
    assert missing.value.code == "not_found"
    judged(store, "a", "t/2", passed=False, output=TAGLINE)
    with pytest.raises(OperationError) as hidden:
        tally_task(store, "t/2", hidden={"t/2"})
    assert hidden.value.code == "held_out"
    # A resample's samples were drawn at a step, where no check runs.
    run = store.get_run(record_run(store, "r", [text_message("x")], task_id="t/3"))
    step = store.steps(run.id)[0]
    experiment = store.create_experiment(kind="resample", arm="original", run_id=run.id, step_idx=0, step_id=step.id,
                                         request={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                                         settings={"n": 1})
    with pytest.raises(OperationError) as at_a_step:
        tally_screen(store, experiment.id[:8])
    assert at_a_step.value.code == "bad_params" and "forks run to the end" in at_a_step.value.message
    with pytest.raises(OperationError) as unknown:
        tally_screen(store, "ffffffff")
    assert unknown.value.code == "not_found"


def test_a_person_reads_it_in_the_terminal(store):
    meeting(store)
    printed = CliRunner().invoke(app, ["tally", "--task", "t/1", "--home", str(store.home)], env={"COLUMNS": "120"})
    assert printed.exit_code == 0, printed.output
    lines = printed.output.splitlines()
    assert any("failed 4 of 5 judged" in line for line in lines)
    tagline = next(index for index, line in enumerate(lines) if "below full marks: tagline #" in line)
    owners = next(index for index, line in enumerate(lines) if "below full marks: owners #" in line)
    assert tagline < owners and "3/4" in lines[tagline]
    assert any("1 failed run(s) whose check printed nothing" in line for line in lines)
    neither = CliRunner().invoke(app, ["tally", "--home", str(store.home)])
    assert neither.exit_code == 1 and "give --task or --screen" in neither.output


async def test_the_web_ui_reads_it_from_the_api(proxy_client, recorder):
    meeting(recorder.store)
    found = (await proxy_client.get(f"{API}/tally", params={"task": "t/1"})).json()
    assert found["scope"] == "task" and found["groups"][0]["findings"][0]["finding"] == "below full marks: tagline #"
    assert (await proxy_client.get(f"{API}/tally", params={"task": "t/9"})).status_code == 404
    assert (await proxy_client.get(f"{API}/tally")).status_code == 400


async def test_an_agent_reads_it_in_sentences(store):
    pytest.importorskip("mcp")
    from tuningfork.mcp.server import ToolServer
    from tuningfork.mcp.tools import registry
    from tests.test_mcp_tools import NoJobs

    meeting(store)
    server = ToolServer(store=store, jobs=NoJobs(), toolset="refine")
    tools = registry("refine")
    answer = await server.call(tools["tally_failures"], {"task": "t/1"})
    assert answer.lines[0] == "t/1: what its failed runs' checks said, the most common first"
    assert "3/4 (75%, 95% CI " in answer.lines[2] and answer.lines[2].endswith("of the failed runs: below full marks: tagline #")
    assert any(line.startswith("aim at the most common finding first") for line in answer.lines)
    assert answer.data["groups"][0]["findings"][0]["rate"]["successes"] == 3
    with pytest.raises(OperationError):
        await server.call(tools["tally_failures"], {})


# --- a screen run to the end ------------------------------------------------------------------------

#: The task's check, saying what it found wrong.
SAYING = ("python3 -c \"import json,sys; d=json.load(open('report.json')); ok=d.get('row_count')==4;"
          " print('ok' if ok else 'no row_count: the report has ' + ', '.join(sorted(d))); sys.exit(0 if ok else 1)\"")
KEYS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total."}


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.item.tuningfork.check = SAYING
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def test_a_screen_to_the_end_is_tallied_arm_by_arm(stack):
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id="tally-report/1")
    assert run.result["passed"] is False
    made = refine.create(stack.store, run.id, "write_file", target_check=True)
    proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads row_count",
                                   proposer={"via": "person"})
    prepared = refine.prepare_refinement_screen(stack.store, made.id, [proposal.id], to_end=True, n=3,
                                                base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
    result = execute_screen_to_end(stack.store, prepared)

    # The task's own runs: the one the batch made.
    (group,) = tally_task(stack.store, "tally-report/1").to_json()["groups"]
    assert [(found["finding"], found["runs"]) for found in group["findings"]] == [
        ("no row_count: the report has count, total", 1)]
    # The screen's forks, by arm: what the original's still fail, beside what the proposal's do.
    for reference in (result.group_id, stack.store.experiment_group(result.group_id)[1].id[:8]):
        found = tally_screen(stack.store, reference).to_json()
        assert found["scope"] == "screen" and found["task_id"] == "tally-report/1" and found["run_id"] == run.id
        original, edited = found["groups"]
        assert (original["arm"], original["label"], original["judged"], original["failed"]) == ("original", "the original", 3, 3)
        assert [(f["finding"], f["runs"]) for f in original["findings"]] == [("no row_count: the report has count, total", 3)]
        assert (edited["arm"], edited["judged"], edited["failed"], edited["findings"]) == (proposal.id[:8], 3, 0, [])
    with pytest.raises(OperationError) as hidden:
        tally_screen(stack.store, result.group_id, hidden={"tally-report/1"})
    assert hidden.value.code == "held_out"
