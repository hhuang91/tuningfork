"""What operations report as they work, and the JSON forms of what they return.

A front end that is not the CLI — the web UI through the job runner, later an MCP server —
sees an operation only through these: events as it goes, and a result that knows its JSON.
"""

from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest

from tuningfork.demo.mock import MockTurn
from tuningfork.experiment import SamplingPlan, load_checkpoint
from tuningfork.ops.experiments import ResamplePlan, execute_resample
from tuningfork.proxy.app import create_app
from tuningfork.store.db import Store
from tests.helpers import PROPS, record_pilot, serving


def events_of(seen, kind):
    return [event for event in seen if event.kind == kind]


class TestResampleEvents:
    async def test_the_experiment_is_named_before_any_sample(self, tmp_path, mock_server, monkeypatch):
        mock_server.props = PROPS
        mock_server.script(*[MockTurn(content=f"answer {i}") for i in range(3)])
        real = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw),
        )
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        prepared = ResamplePlan(
            checkpoint=load_checkpoint(store, run_id, 0), plan=SamplingPlan(n=3),
            base_url="http://upstream.test",
        )
        seen = []
        result = await execute_resample(store, prepared, progress=seen.append)
        store.close()

        assert seen[0].kind == "experiment"
        assert seen[0].data["id"] == result.experiment.id and seen[0].data["n"] == 3
        samples = events_of(seen, "sample")
        assert sorted(event.data["index"] for event in samples) == [0, 1, 2]
        assert all(event.data["experiment_id"] == result.experiment.id for event in samples)
        # Everything a front end sees is plain JSON; the typed objects stay in-process.
        wire = json.loads(json.dumps([event.to_json() for event in seen]))
        assert wire[1]["data"]["outcome"]["kind"] == "text"
        result_json = json.loads(json.dumps(result.to_json()))
        assert result_json["experiment"]["id"] == result.experiment.id
        assert result_json["distribution"]["total"] == 3
        assert json.loads(json.dumps(prepared.to_json()))["checkpoint"]["step"] == 0


class TestPause:
    def test_sampling_through_a_paused_proxy_waits_instead_of_failing(self, tmp_path, mock_server, settings):
        """A pause is a person looking at something. Before, each call made during it came
        back 503 and was stored as a failed sample."""
        mock_server.props = PROPS
        mock_server.script(*[MockTurn(content="fine") for _ in range(3)])
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(settings, client=upstream)
        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        app.state.gate.paused = True
        seen = []
        try:
            with serving(app) as url:
                prepared = ResamplePlan(
                    checkpoint=load_checkpoint(store, run_id, 0), plan=SamplingPlan(n=3),
                    base_url=url,
                )
                threading.Timer(1.5, lambda: setattr(app.state.gate, "paused", False)).start()
                result = asyncio.run(execute_resample(store, prepared, progress=seen.append))
            samples = store.samples(result.experiment.id)
        finally:
            store.close()
            app.state.recorder.store.close()
            asyncio.run(upstream.aclose())

        assert [event.data for event in events_of(seen, "waiting")] == [{"reason": "paused"}]
        assert [s.error for s in samples] == [None, None, None]
        assert result.distribution.total == 3


def test_a_finished_run_as_json_leaves_out_what_the_harness_printed():
    from tuningfork.runner import RunOutcome
    from tuningfork.specs import CheckResult

    outcome = RunOutcome(
        index=2, label="t#b.2", run_id="r", passed=True,
        check=CheckResult(passed=True, returncode=0, output="secret output"),
        duration_s=1.5, returncode=0, stdout="harness stdout", stderr="harness stderr",
        launch_id="l",
    )
    data = outcome.to_json()
    assert data["index"] == 2 and data["run_id"] == "r" and data["passed"] is True
    assert "stdout" not in data and "check_output" not in data
    assert "harness stdout" not in json.dumps(data) and "secret output" not in json.dumps(data)
