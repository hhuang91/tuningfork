"""Reporting a stored experiment again, whatever state it was left in."""

from __future__ import annotations

import json
import os

import httpx
import pytest
from rich.console import Console
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.experiment import SamplingPlan, load_checkpoint, run_resample
from tuningfork.ops.inspect import experiment_report
from tuningfork.store.db import Store
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot
from tests.mock_server import MockTurn

UPSTREAM = "http://upstream.test"


@pytest.fixture
def home(tmp_path, monkeypatch, mock_server):
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    mock_server.props = PROPS
    mock_server.responder = lambda payload: MockTurn(
        tool_calls=[{"name": "write_file", "arguments": {"path": "r.json", "content": "{}"}}]
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw),
    )
    path = tmp_path / "store"
    store = Store(path)
    record_pilot(store)
    store.close()
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")
    result = CliRunner().invoke(app, [
        "ab", "pilot", "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "4",
        "--target-match", "row_count", "--base-url", UPSTREAM, "--home", str(path),
    ])
    assert result.exit_code == 0, result.output
    return path


def ab_ids(path):
    store = Store(path)
    try:
        arms = {e.arm: e.id for e in store.experiments(limit=10)}
    finally:
        store.close()
    return arms


class TestWording:
    """An A/B's report says what it tested: the edit, as a line diff of its two texts, read
    back from what its arms were sent."""

    @staticmethod
    def changed(report) -> list[dict]:
        return [line for hunk in report.wording.diff["hunks"] for line in hunk["lines"] if line["op"] != " "]

    def test_the_report_shows_the_edit_numbered_as_the_files_lines(self, home):
        store = Store(home)
        report = experiment_report(store, ab_ids(home)["variant"])
        store.close()

        assert report.wording.source == "sent" and report.wording.form == "full"
        assert self.changed(report) == [
            {"op": "-", "old": 7, "text": "Write the totals to report.json."},
            {"op": "+", "new": 7, "text": "Write the totals to report.json, using the keys row_count and amount_total."},
        ]
        assert report.wording.variant["label"] == "skill/SKILL.v2.md"
        assert report.to_json()["wording"]["diff"]["changed"] == 1
        shown = CliRunner().invoke(app, ["experiment", ab_ids(home)["variant"][:8], "--home", str(home)])
        assert "the edit  1 line changed" in shown.output
        assert "+ Write the totals to report.json, using the keys row_count and amount_total." in shown.output

    def test_a_file_edited_since_does_not_change_what_was_tested(self, home, tmp_path):
        (tmp_path / "skill" / "SKILL.v2.md").write_text(VARIANT.replace("row_count", "rows"), encoding="utf-8")
        store = Store(home)
        report = experiment_report(store, ab_ids(home)["original"])
        store.close()

        assert report.wording.source == "sent"
        assert self.changed(report)[-1]["text"] == "Write the totals to report.json, using the keys row_count and amount_total."

    def test_before_its_variant_arm_its_files_say_it_while_they_hold_its_texts(self, home, tmp_path):
        arms = ab_ids(home)
        store = Store(home)
        with store._lock:
            store._conn.execute("DELETE FROM samples WHERE experiment_id = ?", (arms["variant"],))
            store._conn.execute("DELETE FROM experiments WHERE id = ?", (arms["variant"],))
            store._conn.commit()
        try:
            report = experiment_report(store, arms["original"])
            assert report.wording.source == "files" and report.wording.diff["changed"] == 1
            (tmp_path / "skill" / "SKILL.v2.md").write_text("another wording\n", encoding="utf-8")
            report = experiment_report(store, arms["original"])
        finally:
            store.close()

        assert report.wording.diff is None
        assert "hold the texts it tested" in report.wording.missing
        shown = CliRunner().invoke(app, ["experiment", arms["original"][:8], "--home", str(home)])
        assert "the edit  not shown: neither what its arms recorded" in shown.output


class TestPartial:
    def test_an_ab_stopped_in_its_variant_arm_says_so(self, home):
        """Found in review: an A/B interrupted part way through its second arm read as a
        complete comparison — the rates over two samples, reported as over four."""
        variant = ab_ids(home)["variant"]
        store = Store(home)
        with store._lock:
            store._conn.execute("DELETE FROM samples WHERE experiment_id = ? AND idx >= 2", (variant,))
            store._conn.commit()
        report = experiment_report(store, variant)
        store.close()

        assert report.partial == [("variant", 2, 4)]
        shown = CliRunner().invoke(app, ["experiment", variant[:8], "--home", str(home)])
        assert "stopped early: the variant arm has 2 of its 4 samples" in shown.output

    def test_an_experiment_whose_run_has_gone_still_reports(self, home):
        original = ab_ids(home)["original"]
        store = Store(home)
        with store._lock:
            store._conn.execute("DELETE FROM steps")
            store._conn.execute("DELETE FROM runs")
            store._conn.commit()
        store.close()

        shown = CliRunner().invoke(app, ["experiment", original[:8], "--home", str(home)])

        assert shown.exception is None or isinstance(shown.exception, SystemExit), shown.output
        assert "no longer in this store" in shown.output
        assert "0/4" in shown.output


class TestTransport:
    async def test_a_200_that_is_not_an_answer_costs_one_sample_not_the_run(self, tmp_path):
        """A JSON list, or an error body, at status 200: that sample failed in transport,
        and the rest are still drawn."""
        calls = {"n": 0}

        async def answer(request: Request):
            calls["n"] += 1
            if calls["n"] == 1:
                return JSONResponse(["not", "an", "answer"])
            if calls["n"] == 2:
                return JSONResponse({"error": {"message": "slot unavailable"}})
            return JSONResponse({
                "id": "c", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "fine"}}],
            })

        upstream = Starlette(routes=[
            Route("/v1/chat/completions", answer, methods=["POST"]),
            Route("/{path:path}", lambda request: JSONResponse({}), methods=["GET"]),
        ])
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        checkpoint = load_checkpoint(store, run_id, 0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream)) as client:
            experiment = await run_resample(
                store, checkpoint, SamplingPlan(n=4), base_url=UPSTREAM, client=client
            )
        samples = store.samples(experiment.id)
        store.close()

        assert len(samples) == 4
        errors = [s.error for s in samples if s.error]
        assert len(errors) == 2
        assert any("JSON list" in e for e in errors)
        assert any("slot unavailable" in e for e in errors)


class TestStatus:
    """Experiments say how they stand. Before they did, a running A/B read as one
    that had stopped early, and one still in its first arm as interrupted."""

    async def _resample(self, tmp_path, stall: float = 0.0, *, cancel_after: float | None = None,
                        reason: str = "cancelled"):
        from tuningfork.ops import Cancel, Cancelled
        from tuningfork.ops.experiments import ResamplePlan, execute_resample

        upstream = Starlette(routes=[
            Route("/v1/chat/completions", self._slow(stall), methods=["POST"]),
            Route("/{path:path}", lambda request: JSONResponse({}), methods=["GET"]),
        ])
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        prepared = ResamplePlan(
            checkpoint=load_checkpoint(store, run_id, 0), plan=SamplingPlan(n=3), base_url=UPSTREAM
        )
        cancel = Cancel()
        real = httpx.AsyncClient
        httpx.AsyncClient = lambda **kw: real(transport=httpx.ASGITransport(app=upstream), **kw)
        try:
            if cancel_after is not None:
                import asyncio

                asyncio.get_running_loop().call_later(cancel_after, cancel.cancel, reason)
            try:
                await execute_resample(store, prepared, cancel=cancel)
                stopped = None
            except Cancelled as caught:
                stopped = caught
        finally:
            httpx.AsyncClient = real
        return store, stopped

    @staticmethod
    def _slow(stall: float):
        import asyncio

        async def answer(request: Request):
            await asyncio.sleep(stall)
            return JSONResponse({
                "id": "c", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "fine"}}],
            })

        return answer

    async def test_an_experiment_that_ran_to_the_end_is_done(self, tmp_path):
        store, stopped = await self._resample(tmp_path)
        (experiment,) = store.experiments(limit=5)
        report = experiment_report(store, experiment.id)
        store.close()
        assert stopped is None
        assert experiment.status == "done" and experiment.ended_at
        assert report.status == "done" and report.partial == [] and report.running == []

    async def test_a_cancelled_experiment_says_so_and_keeps_what_it_drew(self, tmp_path):
        store, stopped = await self._resample(tmp_path, stall=0.3, cancel_after=0.45)
        (experiment,) = store.experiments(limit=5)
        report = experiment_report(store, experiment.id)
        store.close()
        assert stopped is not None and stopped.partial.experiment.id == experiment.id
        assert experiment.status == "cancelled"
        assert report.status == "cancelled"
        assert report.partial and report.partial[0][1] < 3

    async def test_one_stopped_by_a_shutdown_reads_as_interrupted(self, tmp_path):
        store, stopped = await self._resample(tmp_path, stall=0.3, cancel_after=0.45, reason="shutdown")
        (experiment,) = store.experiments(limit=5)
        store.close()
        assert stopped.reason == "shutdown"
        assert experiment.status == "interrupted"

    def test_an_ab_still_in_its_first_arm_is_running_not_interrupted(self, tmp_path):
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        step = store.steps(run_id)[0]
        arm = store.create_experiment(
            kind="ab", arm="original", run_id=run_id, step_idx=0, step_id=step.id,
            request={"messages": []}, settings={"n": 4}, group_id="g",
        )
        running = experiment_report(store, arm.id)

        store.set_experiment_status([arm.id], "running")
        with store._lock:  # the process that ran it is gone
            store._conn.execute(
                "UPDATE experiments SET owner = ? WHERE id = ?",
                (json.dumps({"pid": 2**22 + 99, "host": arm.owner["host"], "start": None}), arm.id),
            )
            store._conn.commit()
        orphaned = experiment_report(store, arm.id)
        store.close()

        assert running.kind == "ab_running" and running.status == "running"
        assert running.running == [("original", 0, 4)] and running.partial == []
        if os.name == "posix":
            assert orphaned.kind == "interrupted" and orphaned.status == "interrupted"

    def test_an_experiment_from_before_statuses_is_read_from_its_samples(self, tmp_path):
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        step = store.steps(run_id)[0]
        legacy = store.create_experiment(
            kind="resample", arm="original", run_id=run_id, step_idx=0, step_id=step.id,
            request={"messages": []}, settings={"n": 4}, status=None,
        )
        report = experiment_report(store, legacy.id)
        store.close()
        assert report.status == "interrupted"
        assert report.partial == [("original", 0, 4)]
