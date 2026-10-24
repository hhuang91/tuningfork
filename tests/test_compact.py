"""Compact views (`ops.compact`): small enough for an agent, and never a rate without its
interval."""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.demo.mock import MockTurn
from tuningfork.ops import compact, inspect
from tuningfork.ops.timeline import step_detail
from tuningfork.store.db import Store
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot, record_run, tool_call_message

UPSTREAM = "http://upstream.test"


def rates(data: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every object in `data` that counts something: it has `successes` and `total`."""
    if isinstance(data, dict):
        if "successes" in data and "total" in data:
            yield path, data
        for key, value in data.items():
            yield from rates(value, f"{path}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from rates(value, f"{path}[{index}]")


def assert_intervals(data: Any) -> int:
    """No rate without its interval; returns how many rates there were."""
    found = list(rates(data))
    for path, value in found:
        assert "low" in value and "high" in value, f"a rate without its interval at {path}: {value}"
    return len(found)


@pytest.fixture
def lab(tmp_path, monkeypatch, mock_server):
    """A recorded run with a resample at step 1, and two A/Bs there: one declaring its target."""
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    mock_server.props = PROPS
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw)
    )
    path = tmp_path / "store"
    store = Store(path)
    run_id = record_pilot(store)
    store.close()
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")

    def write(keys: dict) -> MockTurn:
        return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "r.json", "content": json.dumps(keys)}}])

    counted, plain = write({"total": 10, "row_count": 1}), write({"total": 10})
    runner = CliRunner()

    def cli(*args: str) -> None:
        result = runner.invoke(app, [*args, "--base-url", UPSTREAM, "--home", str(path)])
        assert result.exit_code == 0, result.output

    mock_server.script(plain, plain, counted, plain)
    cli("resample", run_id[:8], "1", "-n", "4")
    mock_server.script(plain, plain, plain, plain, counted, counted, counted, plain)
    cli("ab", run_id[:8], "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "4", "--target-match", "row_count")
    mock_server.script(plain, counted, plain, plain, counted, counted, plain, plain)
    cli("ab", run_id[:8], "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "4")
    store = Store(path)
    yield store, run_id
    store.close()


def reports(store: Store) -> dict[str, dict[str, Any]]:
    """The lab's experiment reports by what they are."""
    found = {}
    for listed in inspect.experiment_list(store):
        report = inspect.experiment_report(store, listed["id"]).to_json()
        settings = report["settings"]
        if report["kind"] == "resample":
            found["resample"] = report
        elif listed["arm"] == "original":
            found["declared" if settings.get("target") else "exploratory"] = report
    return found


class TestExperiments:
    def test_a_resample_reads_as_behaviours_with_their_rates(self, lab):
        store, _ = lab
        report = reports(store)["resample"]
        view = compact.experiment(report)
        assert view["distribution"]["total"] == 4
        assert [b["letter"] for b in view["distribution"]["behaviours"]] == ["A", "B"]
        assert assert_intervals(report) and assert_intervals(view)
        lines = compact.experiment_lines(view)
        assert any("3/4 (75%, 95% CI" in line for line in lines), lines

    def test_a_declared_target_states_its_test(self, lab):
        store, _ = lab
        report = reports(store)["declared"]
        view = compact.experiment(report)
        target = view["comparison"]["target"]
        assert target["description"] and target["alpha"] == 0.05
        assert "samples_to_detect" in target
        assert assert_intervals(report) and assert_intervals(view)
        text = "\n".join(compact.experiment_lines(view))
        assert "declared target" in text and "Fisher's exact test" in text and "alpha 0.05" in text
        assert "exploratory" not in text

    def test_no_target_declared_reads_as_exploratory(self, lab):
        store, _ = lab
        view = compact.experiment(reports(store)["exploratory"])
        text = "\n".join(compact.experiment_lines(view))
        assert view["comparison"]["exploratory"] is True
        assert "exploratory" in text and "Bonferroni" in text
        assert_intervals(view)

    def test_the_list_says_what_each_is(self, lab):
        store, _ = lab
        listed = compact.experiments(inspect.experiment_list(store))
        assert {found["kind"] for found in listed} == {"resample", "ab"}
        assert all(len(found["id"]) == 8 for found in listed)


class TestRuns:
    def test_a_task_keeps_its_rate_and_interval(self, store):
        record_pilot(store)
        (summary,) = inspect.tasks(store)
        view = compact.task(summary)
        assert view["task_id"] == "tally-report/1" and view["conditions"][0]["runs"] == 1
        assert_intervals(view)
        assert compact.task_lines(view) == ["tally-report/1: 1 run(s), passed nothing counted"]

    def test_a_run_reads_step_by_step(self, store):
        record_pilot(store)
        view = compact.run_detail(inspect.run_detail(store, "pilot"))
        assert [step["did"]["kind"] for step in view["steps"]] == ["tool_call", "tool_call"]
        assert len(view["run"]["id"]) == 8

    def test_a_step_shows_its_skill_and_answer_not_its_whole_context(self, store):
        record_pilot(store)
        detail = step_detail(store, "pilot", 1).to_json()
        view = compact.step(detail)
        assert view["context"]["messages"] == len(detail["view"])
        assert view["answer"]["tool_calls"][0]["name"] == "write_file"
        assert view["context"]["skills"], "the skill in the system prompt is found"
        assert compact.size(view) < compact.size(detail)


class TestDivergence:
    def test_only_the_split_step_unless_asked(self, store):
        read = tool_call_message("read_file", {"path": "a"})
        record_run(store, "a", [read, tool_call_message("write_file", {"path": "r"})], task_id="t/1")
        record_run(store, "b", [read, tool_call_message("run_code", {"code": "1"})], task_id="t/1")
        report = inspect.divergence_report(store, task="t/1")
        view = compact.divergence(report)
        assert [found["index"] for found in view["steps"]] == [1]
        assert len(compact.divergence(report, all_steps=True)["steps"]) == 2
        assert assert_intervals(report) and assert_intervals(view)
        lines = compact.divergence_lines(view)
        assert lines[0] == "2 runs agree for 1 step(s), then split at step 1:"

    def test_mixed_skill_versions_are_flagged(self):
        report = {
            "run_ids": ["a" * 32, "b" * 32], "divergence_index": 0, "agreed_steps": 0,
            "mixed_skills": {"v1": ["a" * 32], "v2": ["b" * 32]},
            "steps": [{"index": 0, "diverged": True, "groups": []}],
        }
        view = compact.divergence(report)
        assert view["mixed_skills"]
        assert any("different skill versions" in line for line in compact.divergence_lines(view))


class TestBudget:
    def test_long_text_says_how_much_it_left_out(self):
        assert compact.clip("x" * 500, 400) == "x" * 400 + "… (+100 chars)"
        assert compact.clip("short") == "short"

    def test_a_long_list_keeps_its_count(self):
        assert compact.capped(list(range(25)), 20) == {"shown": list(range(20)), "total": 25}
        assert compact.capped([1, 2]) == [1, 2]

    def test_fit_brings_anything_under_budget_and_keeps_rates_whole(self):
        rate = {"successes": 3, "total": 20, "point": 0.15, "low": 0.05, "high": 0.36}
        huge = {"messages": [{"text": "y" * 5000, "rate": rate} for _ in range(200)]}
        fitted = compact.fit(huge)
        assert compact.size(fitted) <= compact.BUDGET
        assert fitted["messages"]["total"] == 200
        for _, found in rates(fitted):
            assert found == rate

    def test_a_step_with_a_huge_context_stays_small(self, store):
        store.begin_run("big", "t/big")
        messages = [{"role": "system", "content": "s" * 50_000}]
        for turn in range(150):
            messages.append({"role": "user", "content": f"turn {turn} " + "u" * 3000})
            messages.append({"role": "assistant", "content": "a" * 3000})
        from tests.helpers import as_response, text_message

        store.record_step(request={"model": "m", "messages": messages}, response=as_response(text_message("done")))
        view = compact.step(step_detail(store, "big", 0).to_json())
        assert compact.size(view) <= compact.BUDGET
        assert view["context"]["messages"] == len(messages) and len(view["context"]["last"]) == 6


class TestTokens:
    """A probe and a heatmap, from the toy llama.cpp (see tests/test_forking.py)."""

    @pytest.fixture
    def toy(self, monkeypatch, mock_server):
        from tuningfork.demo.toy import props as toy_props

        mock_server.props = toy_props()
        real = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw)
        )
        return mock_server

    async def test_a_probe_keeps_its_test_and_says_which_token_flips(self, store, toy):
        from tuningfork.ops.forking import execute_forking, prepare_forking
        from tests.test_forking import FORK_IN_ANSWER, recorded_and_scored

        run_id = await recorded_and_scored(store, toy, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        result = await execute_forking(store, prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM))
        report = inspect.experiment_report(store, result.experiment.id).to_json()
        view = compact.experiment(report)
        assert view["forking"]["flips"] == 1 and view["forking"]["family"] == 1
        assert assert_intervals(report) and assert_intervals(view)
        text = "\n".join(compact.experiment_lines(view))
        assert "Holm-corrected over 1 planned" in text
        assert "'search_files' instead of 'read_file'" in text

    async def test_a_heatmap_is_hotspots_unless_positions_are_asked_for(self, store, toy):
        from tuningfork.ops.tokens import step_tokens
        from tests.test_forking import FORK_IN_ANSWER, recorded_and_scored

        run_id = await recorded_and_scored(store, toy, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        found = step_tokens(store, run_id, "0").to_json()
        view = compact.tokens(found)
        assert view["scored"] and view["hotspots"] and "tokens" not in view
        position = view["hotspots"][0]["position"]
        picked = compact.tokens(found, positions=[position])
        assert [entry["i"] for entry in picked["tokens"]] == [position]
        assert compact.size(view) < compact.size(found)

    def test_a_step_with_no_tokens_says_how_to_get_them(self):
        view = compact.tokens({"run_id": "a" * 32, "step": 0, "view": None, "streams": [], "probes": []})
        assert view["scored"] is False and "start_tokens" in view["hint"]


class TestToEnd:
    """Forks run to the end, in the shape `ops.forks.EndToEnd.to_json` stores."""

    @staticmethod
    def arm(passed: int, counted: int, low: float, high: float) -> dict:
        return {
            "passed": passed, "counted": counted,
            "rate": {"successes": passed, "total": counted, "point": passed / counted, "low": low, "high": high},
            "exclusions": [{"run_id": "f" * 32, "why": "the harness timed out"}],
            "actions": {"total": counted, "errors": 0, "clusters": [{
                "letter": "A", "outcome": {"kind": "tool_call", "label": "read_file(…)", "key": "k1"},
                "rate": {"successes": counted, "total": counted, "point": 1.0, "low": 0.6, "high": 1.0}}]},
            "forks": [{"idx": 0, "run_id": "e" * 32, "passed": True, "exclusion": None}],
        }

    def test_an_ab_run_to_the_end_states_its_test(self):
        report = {"id": "x" * 32, "kind": "ab", "run_id": "r" * 32, "step_idx": 2, "status": "done",
                  "settings": {"n": 5, "reach": "end"}, "variant": "/v/SKILL.md",
                  "to_end": {"original": self.arm(1, 5, 0.04, 0.62), "variant": self.arm(5, 5, 0.57, 1.0),
                             "p_value": 0.048, "significant": True, "alpha": 0.05}}
        view = compact.experiment(report)
        assert view["to_end"]["variant"]["passed"]["successes"] == 5
        assert view["to_end"]["original"]["left_out"][0]["why"] == "the harness timed out"
        assert assert_intervals(report) and assert_intervals(view)
        text = "\n".join(compact.experiment_lines(view))
        assert "original 1/5 (20%, 95% CI 4–62%) → variant 5/5 (100%, 95% CI 57–100%)" in text
        assert "p = 0.048" in text and "significant" in text
