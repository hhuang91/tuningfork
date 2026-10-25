"""The check run along a run — at its start, after each call that changed its files, at its end — and
what it gives a reader: which call made each finding (the tally, passing runs included, and a mistake
every run made named as one), a step named by the call that made a finding or first changed a file,
the files a run changed since it started, what each step's calls ran, and the warning that a fork
from a step replays every change made before it (findings §31.42)."""

from __future__ import annotations

import asyncio
import json

import pytest
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.ops import refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.checks import (
    ALONG, MAX_FINDINGS, check_along, first_made, kept_findings, made_by_calls, prepare_check_along,
)
from tuningfork.ops.common import OperationError
from tuningfork.ops.inspect import (
    first_change, hides_what_it_ran, ran_excerpt, replay_warning, run_detail, step_changes, workspace_at,
    workspace_patch,
)
from tuningfork.ops.tally import finding_words, shared_mistakes, tally_task
from tuningfork.specs import CHECK_KEPT
from tests.helpers import task_stack
from tests.test_tally import judged

#: The task's check, naming what it finds: a report not written yet, or one without row_count.
NAMING = ("python3 -c \"import json,os,sys; e=os.path.exists('report.json'); d=json.load(open('report.json')) if e "
          "else {}; ok=d.get('row_count')==4; print('ok' if ok else ('no row_count: the report has ' + "
          "', '.join(sorted(d)) if e else 'no report.json yet')); sys.exit(0 if ok else 1)\"")
TASK = "tally-report/1"
WRONG = "no row_count: the report has count, total"


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.item.tuningfork.check = NAMING
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def one_run(stack):
    """A run of the task: it reads data.csv (step 0), writes report.json without row_count (step 1),
    and says so (step 2)."""
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id=TASK)
    return run


def test_the_check_along_a_run_says_which_call_made_each_finding(stack):
    run = one_run(stack)
    assert run.result["passed"] is False
    # What each call changed: the write, at step 1.
    assert step_changes(stack.store, run) == {1: [("A", "report.json")]}
    assert first_change(stack.store, run, "report.json") == 1 == first_change(stack.store, run, "*.json")
    assert first_change(stack.store, run, "*.csv") is None
    # Its states: the start, the workspace the write left (the read changed nothing), the end.
    assert [before for before, _ in prepare_check_along(stack.store, run.id).states] == [0, 2, 3]
    made = {entry["line"]: entry for entry in check_along(stack.store, run.id)["made"]}
    # There from the start: work not done yet. Made by a call: a mistake, made there.
    assert (made["no report.json yet"]["made_by"], made["no report.json yet"]["at_end"]) == (None, False)
    assert (made[WRONG]["made_by"], made[WRONG]["at_end"]) == (1, True)
    # Kept with the run, its verdict as it was.
    kept = stack.store.get_run(run.id)
    assert kept.result["passed"] is False and kept.result[ALONG]["states"][0]["output"].strip() == "no report.json yet"
    assert kept.result[ALONG]["states"][0]["findings"] == ["no report.json yet"]
    # A step named by the call that made a finding, or that first changed a file: each run's own.
    from tuningfork.ops.experiments import find_step, step_index

    assert first_made(kept, "row_count") == 1 and first_made(kept, "nothing like it") is None
    assert step_index(stack.store, kept, "made:ROW_COUNT") == 1 == step_index(stack.store, kept, "changed:report.json")
    assert find_step(stack.store, kept, "changed:nope.txt") is None
    with pytest.raises(OperationError) as never:
        step_index(stack.store, kept, "changed:nope.txt")
    assert "never changed nope.txt" in never.value.message
    # The tally says which call made the finding, in how many of the runs the check was run along.
    (group,) = tally_task(stack.store, TASK).to_json()["groups"]
    (finding,) = group["findings"]
    assert (finding["made"]["steps"], finding["made"]["checked"], finding["made"]["every_run"]) == ([1], 1, False)
    assert finding["made_text"] == "made by a call in 1 of 1 run(s) the check was run along (step 1 in 1)"


def test_it_is_run_from_the_terminal_once_a_run(stack):
    run = one_run(stack)
    home = str(stack.store.home)
    printed = CliRunner().invoke(app, ["check-along", "--task", TASK, "--home", home], env={"COLUMNS": "200"})
    assert printed.exit_code == 0, printed.output
    assert f"step 1's call made: {WRONG}" in printed.output and "no report.json yet" not in printed.output
    again = CliRunner().invoke(app, ["check-along", "--task", TASK, "--home", home], env={"COLUMNS": "200"})
    assert again.exit_code == 0 and "run along before" in again.output and "nothing to run the check along" in again.output
    tallied = CliRunner().invoke(app, ["tally", "--task", TASK, "--home", home], env={"COLUMNS": "200"})
    assert tallied.exit_code == 0 and "made by a call in 1 of 1 run(s)" in tallied.output
    neither = CliRunner().invoke(app, ["check-along", "--home", home])
    assert neither.exit_code == 1 and "give --task or --run" in neither.output
    assert stack.store.get_run(run.id).result[ALONG]


def test_it_says_when_the_check_cannot_run_in_the_sandbox(stack, monkeypatch):
    import tuningfork.sandbox

    one_run(stack)
    monkeypatch.delenv(tuningfork.sandbox.SWITCH, raising=False)
    monkeypatch.setattr(tuningfork.sandbox, "for_launches", lambda hide, **_: (None, "bubblewrap is not installed"))
    printed = CliRunner().invoke(app, ["check-along", "--task", TASK, "--home", str(stack.store.home)],
                                 env={"COLUMNS": "200"})
    assert printed.exit_code == 0, printed.output
    assert "not sandboxed: bubblewrap is not installed" in printed.output


def test_a_check_of_the_reply_alone_has_nothing_to_read_before_the_end(stack):
    stack.item.tuningfork.check = {"reply_contains": "written"}
    stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
    run = one_run(stack)
    with pytest.raises(OperationError) as refused:
        prepare_check_along(stack.store, run.id)
    assert refused.value.code == "bad_params" and "reply" in refused.value.message


def test_a_run_shows_what_each_call_changed_and_a_fork_after_it_is_warned(stack):
    pytest.importorskip("mcp")
    from tests.test_mcp_tools import NoJobs, call, text

    from tuningfork.mcp.server import ToolServer

    run = one_run(stack)
    detail = run_detail(stack.store, run.id)
    assert [step["changed"] for step in detail["steps"]] == [[], [{"status": "A", "path": "report.json"}], []]
    # What the run page says of its launch: here unsandboxed (the suite turns the sandbox off).
    assert detail["launch"]["sandbox"] is None and "harness_settings" in detail["launch"]
    # At the end, what the whole run changed; before a step, what the step before changed.
    end = workspace_at(stack.store, run.id, "end")
    assert (end.since, end.changes) == ("the start", [("A", "report.json")])
    assert '"count": 4' in workspace_patch(stack.store, end, "report.json")
    before = workspace_at(stack.store, run.id, "2")
    assert (before.since, before.changes) == ("the step before", [("A", "report.json")])
    assert workspace_at(stack.store, run.id, "start").since is None
    # A fork from a step replays every change before it.
    assert replay_warning(stack.store, run, 1) is None
    warning = replay_warning(stack.store, run, 2)
    assert "step 1's call changed report.json before step 2" in warning and "changed:report.json" in warning
    # An agent reads the same.
    server = ToolServer(stack.store, NoJobs(), toolset="refine-to-end")
    shown = asyncio.run(call(server, "show_run", {"run": run.id[:8]}))
    assert "its calls that changed files: step 1 (report.json)" in text(shown)
    files = asyncio.run(call(server, "show_files", {"run": run.id[:8], "diff": True, "path": "report.json"}))
    assert "since the start: report.json added" in text(files)
    assert '"count": 4' in files.structured_content["diff"] and "text" not in files.structured_content
    started = asyncio.run(call(server, "start_refinement", {"run": run.id[:8], "step": "2", "target_check": True}))
    assert "note: run" in text(started) and "an edit can change only what happens from step 2 on" in text(started)
    # So does a screen's plan, run to the end from that step.
    made = refine.get(stack.store, started.structured_content["refinement"])
    proposal = refine.add_proposal(stack.store, made.id, edits=[{"old": "Write the totals to report.json.",
                                                                  "new": "Write row_count and amount_total."}],
                                   rationale="the check reads row_count", proposer={"via": "person"})
    plan = refine.prepare_refinement_screen(stack.store, made.id, [proposal.id], to_end=True, n=2,
                                            base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
    assert any("a call before the step changed the workspace" in warning for warning in plan.warnings)


def test_a_steps_label_hides_a_script_its_view_shows_what_it_ran():
    code = ("from hermes_tools import terminal\n\n# book it\ncommands = [\n    'python3 inv.py receive --sku A --qty 1',\n"
            "]\nfor c in commands:\n    terminal(c)\n")

    def response(name, **arguments):
        return {"choices": [{"message": {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    assert ran_excerpt(response("execute_code", code=code)) == (
        "execute_code: commands = [ 'python3 inv.py receive --sku A --qty 1', for c in commands: terminal(c)")
    # What a call writes, not where: the content before the longer path.
    assert ran_excerpt(response("write_file", path="/a/long/way/to/report.json", content="{}")) == "write_file: {}"
    assert hides_what_it_ran("execute_code(code='<ast:604d6998>')") and not hides_what_it_ran("read_file(path='a.md')")
    assert ran_excerpt({"choices": [{"message": {"content": "done"}}]}) is None
    long = ran_excerpt(response("terminal", command="echo " + "x" * 400))
    assert len(long) == 240 and long.endswith("…")


def along(states):
    """A check along a run as it is kept: (the step whose call found the workspace like this, what the
    check printed) for each state, the last its end."""
    return {"states": [{"before_step": before, "end": index == len(states) - 1, "commit": f"c{index}",
                        "passed": output == "ok", "output": output} for index, (before, output) in enumerate(states)],
            "steps": states[-1][0], "at": 0}


def kept_whole(states):
    """A check along a run as it is kept now: each state's findings whole beside its output's end."""
    return {"states": [{"before_step": before, "end": index == len(states) - 1, "commit": f"c{index}",
                        "passed": output == "ok", "output": output[-CHECK_KEPT:], **kept_findings(output)}
                       for index, (before, output) in enumerate(states)],
            "steps": states[-1][0], "at": 0}


def test_a_finding_cut_from_a_long_output_is_not_taken_for_one_a_later_call_made():
    """A check's output past what a verdict keeps loses its first lines: a finding there from the
    start, fixed last, would come back into what was kept and seem made by the call that fixed the
    others. Each state keeps its findings whole."""
    early = "report.json: no row_count"
    others = [f"line {index}: the total of this row is not the sum of its parts" for index in range(60)]
    start = "\n".join([early, *others])
    assert early not in start[-CHECK_KEPT:]
    states = [(0, start), (2, early)]  # the call at step 1 fixed every other line
    made = {entry["line"]: entry["made_by"] for entry in made_by_calls(kept_whole(states))}
    assert made[early] is None  # there from the start: work not done yet
    # Kept as before, the output's end alone, it was taken for step 1's call's.
    assert {entry["line"]: entry["made_by"] for entry in made_by_calls(along(
        [(before, output[-CHECK_KEPT:]) for before, output in states]))}[early] == 1


def test_a_check_that_prints_more_than_findings_places_none(store):
    """Past `MAX_FINDINGS` distinct lines a state is cut, a line missing from it might have been
    there, and no call can be named for any: the run is not counted among those placed."""
    flood = "\n".join(f"row {index} is wrong" for index in range(MAX_FINDINGS + 1))
    kept = kept_findings(flood)
    assert kept["findings_cut"] and len(kept["findings"]) == MAX_FINDINGS
    assert made_by_calls(kept_whole([(0, flood), (2, "row 0 is wrong")])) is None
    placed = judged(store, "a", passed=False, output="row 0 is wrong")
    flooded = judged(store, "b", passed=False, output="row 0 is wrong")
    for run_id, record in ((placed, kept_whole([(0, "ok"), (1, "row 0 is wrong")])),
                           (flooded, kept_whole([(0, flood), (1, "row 0 is wrong")]))):
        store.annotate_run(run_id, result={**store.get_run(run_id).result, ALONG: record})
    (finding,) = tally_task(store, "t/1").to_json()["groups"][0]["findings"]
    assert (finding["made"]["steps"], finding["made"]["checked"]) == ([0], 1)
    assert first_made(store.get_run(flooded), "row") is None


def test_a_line_every_run_kept_to_the_end_is_no_mistake_the_passing_ones_repaired(store):
    """Found in review: a line every run's check printed from the call that wrote the file to the
    end, the passing runs' too — its numbers masked, one finding — was named a mistake every run
    made and the passing ones repaired later, and an agent sent to its call; where_it_holds already
    asks that every passing run undo it."""
    failed = judged(store, "a", passed=False, output="owners 0.5")
    passing = judged(store, "b", passed=True, output="owners 1")
    for run_id, states in ((failed, [(0, "report.json: missing"), (2, "owners 0.5"), (3, "owners 0.5")]),
                           (passing, [(0, "report.json: missing"), (1, "owners 1"), (2, "owners 1")])):
        store.annotate_run(run_id, result={**store.get_run(run_id).result, ALONG: along(states)})
    report = tally_task(store, "t/1").to_json()
    (finding,) = report["groups"][0]["findings"]
    assert finding["made"]["every_run"] and finding["made"]["passing"] == 1
    assert shared_mistakes(report) == [] and "repaired" not in finding["made_text"]


def test_a_mistake_every_run_made_is_named_with_the_call_that_made_it(store):
    pytest.importorskip("mcp")
    from tuningfork.mcp.server import ToolServer
    from tuningfork.mcp.tools import registry
    from tests.test_mcp_tools import NoJobs

    not_yet = "SKU-310: 15 in stock, the document leads to 45"
    twice = "SKU-100: 28 in stock, the document leads to 18"
    # Every run books the first drop again, at its own step; two noticed later and repaired it.
    failed = judged(store, "a", passed=False, output=twice)
    first = judged(store, "b", passed=True, output="ok")
    second = judged(store, "c", passed=True, output="ok")
    for run_id, states in ((failed, [(0, not_yet), (2, twice), (4, twice)]),
                           (first, [(0, not_yet), (3, twice), (5, "ok"), (6, "ok")]),
                           (second, [(0, not_yet), (1, twice), (4, "ok")])):
        store.annotate_run(run_id, result={**store.get_run(run_id).result, ALONG: along(states)})
    report = tally_task(store, "t/1").to_json()
    (finding,) = report["groups"][0]["findings"]
    made = finding["made"]
    # Not booked yet is there from the start in every run, and is not counted as made.
    assert sorted(made["steps"]) == [0, 1, 2] and (made["checked"], made["repaired"], made["passing"]) == (3, 2, 2)
    assert made["every_run"] and shared_mistakes(report) == [finding]
    assert finding_words(finding["finding"]) == "in stock, the document leads to"
    assert "every run made it, the 2 that passed too, and repaired it later" in finding["made_text"]
    # The step it names is each run's own call that made it.
    assert first_made(store.get_run(first), "in stock, the document leads to") == 2
    # An agent is told so, and what follows from it: the check from the call that made it, where it
    # may screen to the end; else a behaviour no run showed.
    for toolset, toward in (("refine-to-end", "target_check"), ("refine", "unseen_ok")):
        server = ToolServer(store=store, jobs=NoJobs(), toolset=toolset)
        answer = asyncio.run(server.call(registry(toolset)["tally_failures"], {"task": "t/1"}))
        said = "\n".join(answer.lines)
        assert "every run made «SKU-#: # in stock, the document leads to #»" in said
        assert "made:in stock, the document leads to" in said and toward in said, toolset
    # A person too.
    printed = CliRunner().invoke(app, ["tally", "--task", "t/1", "--home", str(store.home)], env={"COLUMNS": "220"})
    assert printed.exit_code == 0 and "made:in stock, the document leads to" in printed.output


def test_the_check_along_a_run_leaves_its_result_as_it_was_kept(stack):
    """Kept masked once: masked again with the record beside it, a command's second password would fall
    too, as masking is not idempotent."""
    run = one_run(stack)
    command = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"
    stack.store.annotate_run(run.id, result={**run.result, "check_output": command})
    kept = stack.store.get_run(run.id).result["check_output"]
    check_along(stack.store, run.id)
    assert stack.store.get_run(run.id).result["check_output"] == kept and "-pZq8xK3vN5wLp" in kept
