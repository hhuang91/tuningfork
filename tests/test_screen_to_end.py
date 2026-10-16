"""A screen run to the end (`ops.screen.prepare_screen_to_end`): every arm's samples forks of
the run from the checkpoint to the end, the task's check deciding each — the screen for a
refinement whose target is the check, and open to any other. The fake agent is the harness
and the mock the model, which names the report's keys only when the skill in its context
does, so the original's forks fail the check and the edit's pass."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tuningfork.ops import compact, refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.ops.inspect import experiment_report
from tuningfork.ops.packet import build_packet
from tuningfork.ops.screen import execute_screen_to_end, screen_result
from tests.helpers import task_stack

KEYS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total."}
RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
CHECK = "the task's check passes"


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def recorded_run(stack):
    """One batch run of the training task, which fails: the original names no key."""
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id="tally-report/1")
    assert run.result["passed"] is False
    return run


def proposed(stack, **target):
    run = recorded_run(stack)
    made = refine.create(stack.store, run.id, "write_file", **target)
    proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads row_count",
                                   proposer={"via": "person"})
    return made, proposal


def test_a_refinement_toward_the_check_is_screened_to_the_end(stack):
    made, proposal = proposed(stack, target_check=True)
    assert made.target == {"kind": "check", "description": CHECK}
    # Nothing at the step can judge it.
    with pytest.raises(OperationError) as refused:
        refine.prepare_refinement_screen(stack.store, made.id, [proposal.id], base_url=stack.url, n=5)
    assert refused.value.code == "to_end_only" and "--to-end" in refused.value.hint

    prepared = refine.prepare_refinement_screen(
        stack.store, made.id, [proposal.id], to_end=True, n=5, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
    )
    shown = prepared.to_json()
    assert (shown["reach"], shown["n"], shown["total"], shown["family"], shown["target"]) == ("end", 5, 10, 1, CHECK)
    assert shown["arms"][0]["proposal"] == proposal.id and shown["arms"][0]["changed_lines"] == 1
    assert any("only a large shift" in warning for warning in shown["warnings"])

    result = execute_screen_to_end(stack.store, prepared)
    report = result.report
    assert report.reach == "end" and report.target == CHECK and report.complete
    assert [arm.name for arm in report.survivors()] == [proposal.id[:8]]
    assert report.baseline.target.successes == 0 and report.arms[0].target.successes == 5
    # Stored as one screen whose arms are experiments run to the end; the report recomputes.
    arms = stack.store.experiment_group(result.group_id)
    assert [arm.arm for arm in arms] == ["original", proposal.id[:8]]
    for arm in arms:
        assert arm.kind == "screen" and arm.status == "done" and arm.settings["reach"] == "end"
        assert arm.settings["refinement"] == made.id and arm.settings["screen"]["per_arm"] == 5
        assert arm.settings["screen"]["reach"] == "end"
    assert arms[1].variant["proposal"] == proposal.id
    assert screen_result(stack.store, arms[1].id[:8]).report.to_json() == report.to_json()
    # Each sample is a fork with a verdict: a run of the task, kept out of the list of runs.
    samples = stack.store.samples(arms[1].id)
    assert [sample.passed for sample in samples] == [True] * 5 and all(sample.run_id for sample in samples)
    assert stack.store.get_run(samples[0].run_id).is_fork

    # The refinement's record says what the screen measured.
    (screen,) = refine.screens(stack.store, made.id)
    assert screen["reach"] == "end" and screen["measure"] == CHECK and screen["report"]["survivors"] == [proposal.id[:8]]
    (verdict,) = refine.view(stack.store, made.id)["proposals"][0]["screens"]
    assert verdict["survives"] and verdict["reach"] == "end"
    # Reported like any screen, for a person and for an agent.
    found = experiment_report(stack.store, arms[0].id).to_json()
    assert found["kind"] == "screen" and found["reach"] == "end" and found["screen"]["survivors"] == [proposal.id[:8]]
    lines = compact.experiment_lines(compact.experiment(found))
    assert any("by the task's check" in line and "forks run from the checkpoint to the end" in line for line in lines), lines
    assert any(line.startswith(f"  {proposal.id[:8]} 5/5") and line.endswith("survives") for line in lines), lines
    # The packet sorts the answers seen by their forks' verdicts: the original arm's, and the run's own.
    answers = build_packet(stack.store, made.id).data["answers_here"]
    assert (answers["reached"], answers["of"]) == (0, 6)


def test_a_run_whose_verdict_does_not_count_says_nothing_of_the_check(stack):
    """Toward the check, the packet sorts the answers seen by their verdicts: the recorded run's
    own answer is one only where its verdict counts — not where the skill changed during it, or
    it timed out, as an excluded fork's is not."""
    made, _ = proposed(stack, target_check=True)
    assert build_packet(stack.store, made.id).data["answers_here"]["of"] == 1  # the run's own: it failed
    run = stack.store.get_run(made.run_id)
    stack.store.annotate_run(run.id, result={**run.result, "usable": False})
    answers = build_packet(stack.store, made.id).data["answers_here"]
    assert (answers["reached"], answers["of"]) == (0, 0)


def test_a_refinement_toward_a_behaviour_may_be_screened_to_the_end_too(stack, monkeypatch):
    from rich.console import Console

    from tuningfork.cli import app

    made, proposal = proposed(stack, target_rules=RULES)
    prepared = refine.prepare_refinement_screen(
        stack.store, made.id, [proposal.id], to_end=True, n=5, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
    )
    result = execute_screen_to_end(stack.store, prepared)
    assert [arm.name for arm in result.report.survivors()] == [proposal.id[:8]]
    (screen,) = refine.screens(stack.store, made.id)
    assert screen["reach"] == "end" and screen["measure"] == CHECK
    assert refine.get(stack.store, made.id).target["kind"] == "rules"  # its own target stands

    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    runner = CliRunner()
    shown = runner.invoke(app, ["refine", "show", made.id[:8], "--home", str(stack.store.home)])
    assert shown.exit_code == 0, shown.output
    assert "forks to the end by the task's check" in shown.output and "survives" in shown.output
    assert "by the task's check, forks to the end" in shown.output
    planned = runner.invoke(app, ["refine", "screen", made.id[:8], "--to-end", "-n", "3", "--proxy", stack.url,
                                 "--dry-run", "--home", str(stack.store.home)])
    assert planned.exit_code == 0, planned.output
    # The screen before drew the original's five forks here: they count again, and none is drawn.
    assert "the task's check passes" in planned.output and "forks     3 in all" in planned.output
    assert "original  5 fork(s) drawn by an earlier screen of this refinement" in planned.output
    afresh = runner.invoke(app, ["refine", "screen", made.id[:8], "--to-end", "-n", "3", "--proxy", stack.url,
                                "--fresh-original", "--dry-run", "--home", str(stack.store.home)])
    assert afresh.exit_code == 0, afresh.output
    assert "forks     6 in all: the original first" in afresh.output


def test_the_check_is_a_target_only_where_the_task_has_one(stack):
    from tests.helpers import SKILL, record_pilot

    run_id = record_pilot(stack.store)
    frozen = stack.tmp / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    stack.store.annotate_run(run_id, parity={**stack.store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
    with pytest.raises(OperationError) as refused:
        refine.create(stack.store, run_id, "1", target_check=True)
    assert refused.value.field == "target_check" and "no check" in str(refused.value)
    with pytest.raises(OperationError) as both:
        refine.create(stack.store, run_id, "1", target_check=True, target_match="x")
    assert both.value.field == "target" and "target_check" in str(both.value)


def test_a_screen_to_the_end_draws_its_forks_from_several_checkpoints(stack):
    """The same step of several recorded runs, the refinement's own first, each arm's forks
    spread evenly over them, one experiment per arm per checkpoint in one group; the pooled
    verdict as before, and where it came from beside it. The refinement finds all of them,
    and deleting it takes them all."""
    from typer.testing import CliRunner

    from tuningfork.cli import app
    from tuningfork.ops.experiments import load
    from tuningfork.ops.screen import choose_checkpoints

    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=3)
    execute_batch(stack.store, plan)
    runs = stack.store.runs(limit=5, task_id="tally-report/1")
    assert len(runs) == 3 and all(run.result["passed"] is False for run in runs)
    anchor = runs[0]
    made = refine.create(stack.store, anchor.id, "write_file", target_check=True)
    assert refine.get(stack.store, made.id).step_tools == ["write_file"]
    proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads row_count",
                                   proposer={"via": "person"})

    # The other two runs serve: the same step by its tool, the skill in context, replayable.
    chosen, notes = choose_checkpoints(stack.store, load(stack.store, anchor.id, "write_file"),
                                       4, original_text=refine.get(stack.store, made.id).baseline_text(), aligned=["write_file"])
    assert [found.run.id for found in chosen] == [anchor.id, runs[1].id, runs[2].id] and notes == []

    prepared = refine.prepare_refinement_screen(
        stack.store, made.id, [proposal.id], to_end=True, n=5, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
        checkpoints=4,
    )
    shown = prepared.to_json()
    assert [each["run"] for each in shown["checkpoints"]] == [anchor.id, runs[1].id, runs[2].id]
    assert [each["n"] for each in shown["checkpoints"]] == [2, 2, 1] and shown["total"] == 10
    assert any("3 checkpoint(s) of the 4 asked for" in warning for warning in shown["warnings"])
    assert prepared.per_checkpoint == [2, 2, 1] and len(prepared.baselines) == 3 and len(prepared.arms[0][2]) == 3

    result = execute_screen_to_end(stack.store, prepared)
    report = result.report
    assert report.complete and report.baseline.target.successes == 0 and report.arms[0].target.successes == 5
    assert [arm.name for arm in report.survivors()] == [proposal.id[:8]]
    # Six experiments in one group: each arm at each checkpoint, the pooled test over all.
    group = stack.store.experiment_group(result.group_id)
    assert sorted((arm.arm, arm.run_id) for arm in group) == sorted(
        (name, run.id) for name in ("original", proposal.id[:8]) for run in runs)
    assert {arm.settings["screen"]["per_arm"] for arm in group} == {5}
    assert [each["run"] for each in group[0].settings["screen"]["checkpoints"]] == [anchor.id, runs[1].id, runs[2].id]
    assert result.experiments["original"].run_id == anchor.id and len(result.by_arm["original"]) == 3
    assert [each["arms"] for each in report.checkpoints] == [
        {"original": {"passed": 0, "of": 2}, proposal.id[:8]: {"passed": 2, "of": 2}},
        {"original": {"passed": 0, "of": 2}, proposal.id[:8]: {"passed": 2, "of": 2}},
        {"original": {"passed": 0, "of": 1}, proposal.id[:8]: {"passed": 1, "of": 1}},
    ]
    assert report.across_checkpoints() == {proposal.id[:8]: {"above": 3, "below": 0, "same": 0, "of": 3}}
    again = screen_result(stack.store, group[-1].id[:8])
    assert again.report.to_json() == report.to_json()
    lines = compact.screen_lines(compact.screen(report.to_json()))
    assert any("forks drawn from 3 checkpoints" in line and "0 passing and 3 failing" in line for line in lines), lines
    assert any(line.strip().startswith(f"{proposal.id[:8]}: above the original at every checkpoint") for line in lines), lines

    # The refinement knows the screen, and everything it made.
    (screen,) = refine.screens(stack.store, made.id)
    assert screen["report"]["survivors"] == [proposal.id[:8]] and len(screen["report"]["checkpoints"]) == 3
    assert set(refine.made_by(stack.store, refine.get(stack.store, made.id)).experiments) == {arm.id for arm in group}
    cited = refine.citations(stack.store, runs=[runs[2].id])
    assert cited and "a checkpoint its screens drew from" in cited[0]["cites"]

    runner = CliRunner()
    planned = runner.invoke(app, ["refine", "screen", made.id[:8], "--to-end", "-n", "6", "--proxy", stack.url,
                                 "--dry-run", "--home", str(stack.store.home)])
    assert planned.exit_code == 0, planned.output
    assert "from      3 checkpoints" in planned.output and "2 an arm" in planned.output
    alone = runner.invoke(app, ["refine", "screen", made.id[:8], "--to-end", "-n", "4", "--checkpoints", "1",
                               "--proxy", stack.url, "--dry-run", "--home", str(stack.store.home)])
    assert alone.exit_code == 0 and "from      " not in alone.output, alone.output


def test_a_screen_at_the_step_samples_one_checkpoint(stack):
    made, proposal = proposed(stack, target_rules=RULES)
    with pytest.raises(OperationError) as refused:
        refine.prepare_refinement_screen(stack.store, made.id, [proposal.id], base_url=stack.url, n=5, checkpoints=3)
    assert refused.value.field == "checkpoints"


ROWS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json with the keys row_count and amount_total."}


def test_the_original_is_forked_once_a_refinement_and_a_quick_look_forks_the_proposals_alone(stack):
    """A later screen counts the original's forks an earlier one drew at the same checkpoints and
    draws only what its n still needs; a quick look draws the proposals' forks alone and gives
    rates, never a verdict, so the driver's rule cannot keep it; a fresh original is asked for."""
    from tuningfork.ops.screen import QUICK_N
    from tuningfork.ops.tally import tally_screen

    made, first = proposed(stack, target_check=True)
    full = refine.prepare_refinement_screen(
        stack.store, made.id, [first.id], to_end=True, n=4, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
    )
    assert full.to_json()["original"] == {"reused": 0, "from": [], "drawn": 4} and full.total == 8
    execute_screen_to_end(stack.store, full)
    second = refine.add_proposal(stack.store, made.id, edits=[ROWS], rationale="the keys, said again",
                                 proposer={"via": "person"})

    # A quick look: the proposal's eight forks, the original's four counted again.
    look = refine.prepare_refinement_screen(
        stack.store, made.id, [second.id], to_end=True, quick=True, base_url=f"{stack.url}/v1",
        workdir=stack.tmp / "forks",
    )
    shown = look.to_json()
    assert (shown["quick"], shown["n"], shown["total"]) == (True, QUICK_N, QUICK_N)
    assert shown["original"]["reused"] == 4 and shown["original"]["drawn"] == 0
    assert any("a quick look" in warning and "no verdict" in warning for warning in shown["warnings"])
    result = execute_screen_to_end(stack.store, look)
    report = result.report
    assert report.quick and report.complete and report.reused == 4 and not report.verdict
    assert report.survivors() == [] and report.arms[0].target.successes == QUICK_N
    assert report.baseline.target.successes == 0 and report.baseline.drawn == 4
    assert [arm.arm for arm in stack.store.experiment_group(result.group_id)] == [second.id[:8]]
    assert report.to_json()["arms"][0]["survives"] is None
    lines = compact.screen_lines(compact.screen(report.to_json()))
    assert lines[0].startswith("quick look at 1 proposal(s)") and "no verdict" in lines[0], lines
    # The refinement and an agent read it as a look; its tally counts the original's forks it counted.
    verdicts = {v["screen"]: v for p in refine.view(stack.store, made.id)["proposals"] for v in p["screens"]}
    assert [v["quick"] for v in verdicts.values()].count(True) == 1
    tally = tally_screen(stack.store, stack.store.experiment_group(result.group_id)[0].id)
    assert [(group.label, group.judged) for group in tally.groups] == [
        ("the original", 4), (f"proposal {second.id[:8]}", QUICK_N)]
    # The terminal reads it as a look too: no "n=0 an arm", no "against the original's 0/0".
    from tuningfork.cli import app

    shown = CliRunner().invoke(app, ["refine", "show", made.id, "--home", str(stack.store.home)], env={"COLUMNS": "220"})
    assert shown.exit_code == 0, shown.output
    assert f"quick look at 1 proposal, {QUICK_N} forks each" in shown.output and "rates only, no verdict" in shown.output
    assert f"quick look — {QUICK_N}/{QUICK_N} forks passed: rates, never a verdict; the original's 0/4 drawn before" in shown.output
    assert "n=0 an arm" not in shown.output

    # A full screen of it: the original's four count again, and only the proposal's four are drawn.
    again = refine.prepare_refinement_screen(
        stack.store, made.id, [second.id], to_end=True, n=4, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
    )
    assert again.original_counts == [0] and again.total == 4
    kept = execute_screen_to_end(stack.store, again).report
    assert kept.verdict and kept.reused == 4 and [arm.name for arm in kept.survivors()] == [second.id[:8]]
    # Asked for, the original is drawn anew; a quick look draws none of it, so it cannot.
    fresh = refine.prepare_refinement_screen(
        stack.store, made.id, [second.id], to_end=True, n=4, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
        fresh_original=True,
    )
    assert fresh.original_counts == [4] and fresh.total == 8 and fresh.to_json()["original"]["reused"] == 0
    with pytest.raises(OperationError) as both:
        refine.prepare_refinement_screen(stack.store, made.id, [second.id], to_end=True, quick=True,
                                         fresh_original=True, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks")
    assert both.value.field == "fresh_original"
    with pytest.raises(OperationError) as at_step:
        refine.prepare_refinement_screen(stack.store, made.id, [second.id], quick=True, base_url=stack.url)
    assert at_step.value.field in ("quick", "refinement")


def test_where_a_question_holds_among_a_screens_forks(stack):
    """Where none of the task's own runs passed, a screen's forks are asked instead, arm by arm:
    the original's forks wrote no row_count and failed, the proposal's wrote it and passed. And
    the answer for the task's runs points to the screen whose forks passed."""
    import asyncio

    from tuningfork.mcp.server import ToolServer
    from tuningfork.ops.where import passing_screens, where_it_holds
    from tests.test_mcp_tools import NoJobs, call, refusal, text

    made, proposal = proposed(stack, target_check=True)
    prepared = refine.prepare_refinement_screen(
        stack.store, made.id, [proposal.id], to_end=True, n=3, base_url=f"{stack.url}/v1", workdir=stack.tmp / "forks",
    )
    result = execute_screen_to_end(stack.store, prepared)
    screen = stack.store.experiment_group(result.group_id)[0].id
    found = where_it_holds(stack.store, screen=screen[:8], path="report.json", contains="row_count").to_json()
    assert found["screen"] == result.group_id and found["task_id"] == "tally-report/1"
    assert [(g["label"], g["passed"], g["failed"]) for g in found["groups"]] == [
        ("the original's forks", 0, 3), (f"proposal {proposal.id[:8]}'s forks", 3, 0)]
    assert [sum(1 for run in g["runs"] if run["held"]) for g in found["groups"]] == [0, 3]
    assert passing_screens(stack.store, "tally-report/1")[0]["passed"] == 3

    server = ToolServer(stack.store, NoJobs(), toolset="refine-to-end")
    asked = asyncio.run(call(server, "where_it_holds", {"screen": screen[:8], "path": "report.json", "contains": "row_count"}))
    assert not asked.is_error and "asked of each fork of screen" in text(asked), text(asked)
    of_task = asyncio.run(call(server, "where_it_holds", {"task": "tally-report/1", "path": "report.json"}))
    assert "no run of the task passed" in text(of_task) and "forks passed 3 of 6" in text(of_task), text(of_task)
    assert "where_it_holds with screen=" in text(of_task), text(of_task)
    neither = asyncio.run(call(server, "where_it_holds", {"path": "report.json"}))
    assert refusal(neither) and neither.structured_content["error"]["field"] == "task"
