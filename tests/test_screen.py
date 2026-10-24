"""A screen at a recorded checkpoint (`ops.screen`), against the mock model: a fresh
baseline arm and one arm per proposal, the plan stored before the first sample, the report
recomputed from the store, and no verdict from a screen cut short."""

from __future__ import annotations

import json

import httpx
import pytest

from tuningfork.analysis.distribution import target_by_rules
from tuningfork.ops import compact
from tuningfork.ops.common import Cancel, Cancelled, OperationError
from tuningfork.ops.inspect import experiment_report
from tuningfork.ops.screen import execute_screen, prepare_screen, screen_result
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot
from tests.mock_server import MockTurn
from tests.test_compact import assert_intervals

UPSTREAM = "http://upstream.test"
TARGET = target_by_rules([{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]).spec
NEUTRAL = SKILL.replace("Write the totals to report.json.", "Write the totals to report.json, compactly.")


def wording_sensitive(payload: dict) -> MockTurn:
    """Names the report's keys only when the skill in its context does."""
    keys = {"row_count": 4, "amount_total": 61.5} if "row_count and amount_total" in payload["messages"][0]["content"] \
        else {"count": 4, "total": 61.5}
    return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}])


@pytest.fixture
def served(monkeypatch, mock_server):
    mock_server.props = PROPS
    mock_server.responder = wording_sensitive
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw)
    )
    return mock_server


def prepare(store, run_id, arms, **kwargs):
    return prepare_screen(
        store, run_id, "1", original_text=SKILL, arms=arms, target_spec=TARGET, base_url=UPSTREAM,
        n=kwargs.pop("n", 6), **kwargs,
    )


def test_a_context_masked_as_it_was_recorded_is_said_in_the_plan(store):
    from tuningfork.config import REDACTED

    run_id = record_pilot(store, read=f"id,amount\nkey={REDACTED}\n")
    plan = prepare(store, run_id, [("p1", VARIANT, {"proposal": "a1"})])
    assert [warning for warning in plan.warnings if f"reads {REDACTED} in 1 place," in warning]


async def test_a_wording_that_names_the_keys_survives_and_a_neutral_one_does_not(store, served):
    run_id = record_pilot(store)
    plan = prepare(store, run_id, [("p1", VARIANT, {"proposal": "a1"}), ("p2", NEUTRAL, {"proposal": "b2"})], n=8)
    assert (plan.family, plan.total) == (2, 24)
    result = await execute_screen(store, plan)

    report = result.report
    assert report.complete and [arm.name for arm in report.survivors()] == ["p1"]
    assert report.baseline.target.successes == 0 and report.arms[0].target.successes == 8
    # The plan is stored with every arm, and the report recomputes from the store alone.
    arms = store.experiment_group(result.group_id)
    assert [arm.arm for arm in arms] == ["original", "p1", "p2"]
    for arm in arms:
        assert arm.kind == "screen" and arm.status == "done"
        assert arm.settings["screen"]["arms"] == ["p1", "p2"] and arm.settings["screen"]["per_arm"] == 8
    assert arms[1].variant["proposal"] == "a1" and arms[1].variant["spans"]
    assert screen_result(store, arms[2].id[:8]).report.to_json() == report.to_json()

    found = experiment_report(store, arms[0].id).to_json()
    assert found["kind"] == "screen" and found["screen"]["survivors"] == ["p1"]
    assert found["arms"]["p1"]["proposal"] == "a1"
    view = compact.experiment(found)
    assert assert_intervals(found) and assert_intervals(view)
    lines = compact.experiment_lines(view)
    assert any(line.startswith("  p1 8/8") and line.endswith("survives") for line in lines), lines
    assert any("p2" in line and "not significant" in line for line in lines)


async def test_the_baseline_is_drawn_first_and_each_arm_sees_its_own_wording(store, served):
    run_id = record_pilot(store)
    await execute_screen(store, prepare(store, run_id, [("p1", VARIANT, {})], n=2))
    systems = [payload["messages"][0]["content"] for payload in served.requests]
    assert ["row_count and amount_total" in system for system in systems] == [False, False, True, True]


async def test_a_screen_cut_short_keeps_its_samples_and_gives_no_verdict(store, served):
    run_id = record_pilot(store)
    plan = prepare(store, run_id, [("p1", VARIANT, {})], n=4)
    cancel, drawn = Cancel(), []

    def progress(event):
        if event.kind == "sample":
            drawn.append(event)
            if len(drawn) == 5:
                cancel.cancel()

    with pytest.raises(Cancelled) as stopped:
        await execute_screen(store, plan, progress=progress, cancel=cancel)
    partial = stopped.value.partial.report
    assert not partial.complete and partial.survivors() == []
    assert partial.baseline.drawn == 4 and partial.arms[0].drawn >= 1
    assert all(arm.status == "cancelled" for arm in store.experiment_group(stopped.value.partial.group_id))


class TestRefusedBeforeSampling:
    def test_an_arm_identical_to_the_original_or_to_another(self, store):
        run_id = record_pilot(store)
        with pytest.raises(OperationError, match="same text as original"):
            prepare(store, run_id, [("p1", SKILL, {})])
        with pytest.raises(OperationError, match="p2 is the same text as p1"):
            prepare(store, run_id, [("p1", VARIANT, {}), ("p2", VARIANT, {})])

    def test_names_and_how_many(self, store):
        run_id = record_pilot(store)
        with pytest.raises(OperationError, match="none may be called original"):
            prepare(store, run_id, [("original", VARIANT, {})])
        with pytest.raises(OperationError, match="at most 6"):
            prepare(store, run_id, [(f"p{k}", VARIANT + "\n" * k, {}) for k in range(1, 8)])
        with pytest.raises(OperationError, match="at least one"):
            prepare(store, run_id, [])

    def test_a_step_without_the_skill_in_its_context(self, store):
        run_id = record_pilot(store)
        with pytest.raises(OperationError) as refused:
            prepare_screen(store, run_id, "1", original_text="# another skill\n\nDo it.\n",
                           arms=[("p1", "# another skill\n\nDo it well.\n", {})], target_spec=TARGET, base_url=UPSTREAM)
        assert refused.value.field == "step"

    def test_nothing_could_survive_when_the_step_already_does_it(self, store):
        run_id = record_pilot(store)
        step = store.step(run_id, 1)
        resample = store.create_experiment(kind="resample", arm="original", run_id=run_id, step_idx=1,
                                           step_id=step.id, request={"messages": []}, settings={"n": 20})
        write = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                {"path": "report.json", "content": json.dumps({"row_count": 4})})}}]}}]}
        for index in range(20):
            store.add_sample(resample.id, index, seed=index, response=write, outcome_kind="tool_call",
                             outcome_key="k", outcome_label="write_file(…)")
        with pytest.raises(OperationError, match="no proposal could survive") as refused:
            prepare(store, run_id, [("p1", VARIANT, {}), ("p2", NEUTRAL, {})], n=20)
        assert refused.value.field == "n"

    def test_power_is_stated_before_any_sample(self, store):
        run_id = record_pilot(store)
        plan = prepare(store, run_id, [("p1", VARIANT, {}), ("p2", NEUTRAL, {})], n=20)
        assert plan.power["needed"] == 6 and "assumed" in plan.power  # no resample yet: the best case
        assert plan.to_json()["family"] == 2 and plan.to_json()["total"] == 60


async def test_the_cli_reports_a_screen_again(store, served, monkeypatch):
    from rich.console import Console
    from typer.testing import CliRunner

    from tuningfork.cli import app

    run_id = record_pilot(store)
    result = await execute_screen(store, prepare(store, run_id, [("p1", VARIANT, {}), ("p2", NEUTRAL, {})], n=6))
    monkeypatch.setattr("tuningfork.cli.console", Console(width=200))
    printed = CliRunner().invoke(app, ["experiment", result.experiments["p2"].id[:8], "--home", str(store.home)])
    assert printed.exit_code == 0, printed.output
    assert "target (declared before sampling)" in printed.output
    assert "survives" in printed.output and "not significant" in printed.output
    assert "Holm: corrected over the 2 planned" in printed.output
