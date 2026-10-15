"""Where a target holds along a task's runs: one question asked of every run — of the model's answers, at
every step, or of a file the runs wrote, at each state of the workspace — and the answers lined up with
the verdicts: the call that made it in each run, and the step that names that call in every run. For a
person (the CLI, the API) and for an agent (MCP), never of a held-out task; and what the other tools say
of it when a check prints only its verdict."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.ops.where import make_probe, seen_text, split_text, where_it_holds
from tests.helpers import record_run, task_stack, text_message, tool_call_message
from tests.mock_server import MockTurn

API = "/_tuningfork/api"
TASK = "t/1"
#: A script that drops the export's blank rows, and one that keeps them.
DROPS = "rows = [r for r in rows if any(c.strip() for c in r)]"
KEEPS = "rows = [r for r in rows]"


def scripted_run(store, label, steps, *, passed, task=TASK, skill="v1"):
    """A recorded run of `task`: each of `steps` a tool call, (name, arguments), then a reply; with its
    verdict as a batch stores one, its check silent."""
    messages = [tool_call_message(name, arguments, call_id=f"c{index}") for index, (name, arguments) in enumerate(steps)]
    run_id = record_run(store, label, [*messages, text_message("done")], task_id=task,
                        tools=["read_file", "write_file", "execute_code"])
    store.annotate_run(run_id, harness="hermes", skill_hashes={"SKILL.md": skill},
                       result={"passed": passed, "check_output": "", "usable": True})
    return run_id


def cleaning(store):
    """Three passing runs whose script drops the blank rows and three failing runs whose script keeps them;
    one of each reads the file first, its script a step later."""
    read = ("read_file", {"path": "data.csv"})
    return {
        "p1": scripted_run(store, "p1", [("execute_code", {"code": DROPS})], passed=True),
        "p2": scripted_run(store, "p2", [("execute_code", {"code": DROPS})], passed=True),
        "p3": scripted_run(store, "p3", [read, ("execute_code", {"code": DROPS})], passed=True),
        "f1": scripted_run(store, "f1", [("execute_code", {"code": KEEPS})], passed=False),
        "f2": scripted_run(store, "f2", [read, ("execute_code", {"code": KEEPS})], passed=False),
        "f3": scripted_run(store, "f3", [("execute_code", {"code": KEEPS})], passed=False),
    }


# --- of the model's answers -----------------------------------------------------------------------


def test_a_target_is_tried_at_every_step_of_every_run(store):
    ids = cleaning(store)
    found = where_it_holds(store, TASK, target_match=r"any\(c\.strip\(\)").to_json()
    assert found["probe"]["kind"] == "answer" and found["skipped"] == []
    (group,) = found["groups"]
    assert (group["passed"], group["failed"], group["unjudged"]) == (3, 3, 0)
    split = group["split"]
    assert (split["passing"]["successes"], split["failing"]["successes"]) == (3, 0)
    assert split["goes_with"] == "passing" and split["p"] == 0.1
    # Each run at its own step: the script came a step later where the file was read first.
    by_run = {entry["run"]: entry for entry in group["runs"]}
    assert (by_run[ids["p1"]]["step"], by_run[ids["p3"]]["step"], by_run[ids["p3"]]["steps"]) == (0, 1, [1])
    assert by_run[ids["p3"]]["tools"] == ["execute_code"]
    assert by_run[ids["f2"]]["held"] is False and by_run[ids["f2"]]["step"] is None
    assert group["seen_at"] == [[0, 2], [1, 1]]
    # The step that names that call in each run, whatever its index: its first call of the tool.
    assert group["reference"] == {"step": "execute_code", "names": 3, "of": 3}
    assert by_run[ids["p3"]]["reference"] == "execute_code" and by_run[ids["f2"]]["reference"] is None
    assert split_text(group, "answer") == ("at some step in 3 of 3 passing runs (44–100%) and 0 of 3 failing (0–56%): "
                                           "it goes with passing (Fisher p = 0.10)")
    assert seen_text(group, "answer") == "first at step 0 in 2, step 1 in 1"


def test_a_described_target_and_one_every_run_satisfies(store):
    cleaning(store)
    (every,) = where_it_holds(store, TASK, target_rules=[{"if": "calls", "value": "execute_code"}]).to_json()["groups"]
    assert (every["split"]["passing"]["successes"], every["split"]["failing"]["successes"]) == (3, 3)
    assert every["split"]["goes_with"] is None and every["split"]["p"] == 1.0
    assert split_text(every, "answer").endswith("no split (Fisher p = 1.00)")
    (read,) = where_it_holds(store, TASK, target_rules=[{"if": "calls", "value": "read_file"}]).to_json()["groups"]
    assert read["reference"] == {"step": "read_file", "names": 2, "of": 2}


def test_each_condition_apart_and_what_is_refused(store):
    cleaning(store)
    scripted_run(store, "v2", [("execute_code", {"code": DROPS})], passed=True, skill="v2")
    groups = where_it_holds(store, TASK, target_match="strip").to_json()["groups"]
    assert sorted(group["passed"] + group["failed"] for group in groups) == [1, 6]
    with pytest.raises(OperationError) as hidden:
        where_it_holds(store, TASK, target_match="strip", hidden={TASK})
    assert hidden.value.code == "held_out"
    with pytest.raises(OperationError) as missing:
        where_it_holds(store, "t/9", target_match="strip")
    assert missing.value.code == "not_found"
    for question, field in [
        ({}, "target_match"),
        ({"target_match": "a", "path": "x.csv"}, "path"),
        ({"target_match": "a", "target_rules": [{"if": "calls", "value": "x"}]}, "target_rules"),
        ({"target_match": "("}, "target_match"),
        ({"target_match": "a", "contains": "b"}, "contains"),
        ({"path": "/home/user/project/x.csv"}, "path"),
        ({"path": "x.csv", "contains": "("}, "contains"),
        ({"path": "x.csv", "at_least": 2}, "at_least"),
        ({"path": "x.csv", "contains": "a", "at_least": 0}, "at_least"),
    ]:
        with pytest.raises(OperationError) as refused:
            make_probe(**question)
        assert refused.value.field == field, question
    assert make_probe(path="./out/*.json", contains="a", at_least=2).description == "out/*.json contains /a/ at least 2 times"
    assert make_probe(path="report.json").description == "report.json exists"


def test_a_person_reads_it_in_the_terminal(store):
    cleaning(store)
    printed = CliRunner().invoke(app, ["where", "--task", TASK, "--match", r"any\(c\.strip", "--home", str(store.home)])
    assert printed.exit_code == 0, printed.output
    assert "t/1: behaviour matching /any\\(c\\.strip/, asked of 6 run(s)" in printed.output
    assert "it goes with passing (Fisher p = 0.10)" in printed.output
    assert "step \"execute_code\" names that call in 3 of the 3 run(s)" in printed.output
    assert "tuningfork refine start <run> 'execute_code'" in printed.output
    wrong = CliRunner().invoke(app, ["where", "--task", TASK, "--rules", "{nope", "--home", str(store.home)])
    assert wrong.exit_code == 1 and "--rules is not JSON" in wrong.output
    neither = CliRunner().invoke(app, ["where", "--task", TASK, "--home", str(store.home)])
    assert neither.exit_code == 1 and "ask one question" in neither.output


async def test_the_web_ui_reads_it_from_the_api(proxy_client, recorder):
    cleaning(recorder.store)
    found = (await proxy_client.get(f"{API}/where", params={"task": TASK, "target_match": "any"})).json()
    assert found["groups"][0]["split"]["goes_with"] == "passing"
    rules = json.dumps([{"if": "calls", "value": "read_file"}])
    found = (await proxy_client.get(f"{API}/where", params={"task": TASK, "target_rules": rules})).json()
    assert found["groups"][0]["reference"]["step"] == "read_file"
    assert (await proxy_client.get(f"{API}/where", params={"task": "t/9", "target_match": "a"})).status_code == 404
    assert (await proxy_client.get(f"{API}/where", params={"task": TASK})).status_code == 400
    assert (await proxy_client.get(f"{API}/where", params={"task": TASK, "path": "x", "at_least": "two"})).status_code == 400
    assert (await proxy_client.get(f"{API}/where")).status_code == 400


async def test_an_agent_reads_it_in_sentences_and_is_told_where_to_refine(store):
    pytest.importorskip("mcp")
    from tuningfork.mcp.server import ToolServer
    from tuningfork.mcp.tools import registry
    from tests.test_mcp_tools import NoJobs

    cleaning(store)
    assert "where_it_holds" in registry("evidence") and "where_it_holds" in registry("refine")
    server = ToolServer(store=store, jobs=NoJobs(), toolset="refine")
    tools = registry("refine")
    answer = await server.call(tools["where_it_holds"], {"task": TASK, "target_match": r"any\(c\.strip"})
    assert answer.lines[0] == "t/1: behaviour matching /any\\(c\\.strip/, asked of each run"
    assert any("it goes with passing (Fisher p = 0.10)" in line for line in answer.lines)
    assert answer.lines[-1].startswith("the passing runs' answers did it first at step \"execute_code\": "
                                       "start_refinement at step \"execute_code\" of a failing run")
    assert answer.data["groups"][0]["split"]["passing"]["successes"] == 3
    # A check that prints only its verdict: the tally says so, and where to look instead.
    tallied = await server.call(tools["tally_failures"], {"task": TASK})
    assert any(line.startswith("the check printed nothing to count, only its verdict: where_it_holds") for line in tallied.lines)
    with pytest.raises(OperationError):
        await server.call(tools["where_it_holds"], {"task": TASK})


# --- of what the runs wrote -----------------------------------------------------------------------

RIGHT, WRONG = {"row_count": 4}, {"count": 4}


def reports(*scripts):
    """The model: each run reads the data, writes report.json once with each of its script's contents in
    turn, and says so — the next run, the next script."""
    started = []

    def respond(payload):
        messages = payload["messages"]
        workdir = re.search(r"Working directory: (\S+)", messages[0]["content"]).group(1)
        done = sum(1 for message in messages if message["role"] == "tool")
        if not done:
            started.append(True)
            return MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": "data.csv"}}])
        writes = scripts[len(started) - 1]
        if done <= len(writes):
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/report.json", "content": json.dumps(writes[done - 1])}}])
        return MockTurn(content="The report is written.")

    return respond


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def batch(stack, *scripts):
    """A batch of the task, one run a script, in order; each run by its place in the batch."""
    stack.mock.responder = reports(*scripts)
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=len(scripts))
    execute_batch(stack.store, plan)
    runs = stack.store.runs(limit=50, task_id="tally-report/1")
    assert len(runs) == len(scripts)
    return {int(run.label.rsplit(".", 1)[1]): run for run in runs}


def test_a_file_is_asked_at_each_state_of_each_run(stack):
    # Some runs write the wrong key and stop; one writes it and mends it; the rest write it right.
    runs = batch(stack, [RIGHT], [WRONG], [WRONG, RIGHT], [WRONG], [RIGHT], [WRONG])
    assert [runs[index].result["passed"] for index in range(6)] == [True, False, True, False, True, False]
    found = where_it_holds(stack.store, "tally-report/1", path="report.json", contains='"count"').to_json()
    assert found["probe"]["description"] == 'report.json contains /"count"/'
    (group,) = found["groups"]
    by_place = {index: next(entry for entry in group["runs"] if entry["run"] == run.id) for index, run in runs.items()}
    # Written wrong by the call at step 1, the write, and left so: what a failing run left.
    wrong = by_place[1]
    assert (wrong["at_start"], wrong["made_by"], wrong["undone_by"], wrong["at_end"]) == (False, 1, None, True)
    assert (wrong["file"], wrong["count"], wrong["tools"], wrong["reference"]) == ("report.json", 1, ["write_file"],
                                                                                    "changed:report.json")
    # Written wrong at step 1 and mended at step 2: made, undone, not there at the end.
    mended = by_place[2]
    assert (mended["made_by"], mended["undone_by"], mended["at_end"], mended["held"]) == (1, 2, False, False)
    assert (by_place[0]["made_by"], by_place[0]["at_end"]) == (None, False)
    split = group["split"]
    assert (split["passing"]["successes"], split["failing"]["successes"]) == (0, 3) and split["goes_with"] == "failing"
    assert group["made"] == {"runs": 4, "of": 6, "passing": 1, "undone": 1, "from_start": 0, "every_run": False,
                             "shared_mistake": False}
    # Each run's first change of the file names the call that made it, in each of them.
    assert group["reference"] == {"step": "changed:report.json", "names": 4, "of": 4}
    assert seen_text(group, "file") == "made by a call in 4 of 6 judged run(s) (step 1 in 4); undone before the end in 1"
    # A file every run made and kept: no split, and no mistake every run made.
    (exists,) = where_it_holds(stack.store, "tally-report/1", path="*.json").to_json()["groups"]
    assert exists["split"]["p"] == 1.0 and exists["made"]["every_run"] and not exists["made"]["shared_mistake"]


def test_a_mistake_every_run_made_and_the_passing_runs_mended(stack):
    pytest.importorskip("mcp")
    from tuningfork.mcp.server import ToolServer
    from tuningfork.mcp.tools import registry
    from tests.test_mcp_tools import NoJobs

    batch(stack, [WRONG, RIGHT], [WRONG], [WRONG, RIGHT], [WRONG])
    (group,) = where_it_holds(stack.store, "tally-report/1", path="report.json", contains='"count"').to_json()["groups"]
    assert group["made"]["every_run"] and group["made"]["shared_mistake"] and group["made"]["passing"] == 2
    assert seen_text(group, "file").endswith("every run made it, the 2 that passed too, which undid it before the end")
    server = ToolServer(store=stack.store, jobs=NoJobs(), toolset="refine-to-end")
    answer = asyncio.run(server.call(registry("refine-to-end")["where_it_holds"],
                                     {"task": "tally-report/1", "path": "report.json", "contains": '"count"'}))
    assert answer.lines[-1].startswith("every run made it, the passing ones too, and undid it later")
    assert "step \"changed:report.json\"" in answer.lines[-1] and "target_check" in answer.lines[-1]
