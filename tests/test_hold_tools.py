"""Tools at a held call: sampling it (evidence, filed under the step the call becomes) and
asking the model about it (a hypothesis, never recorded)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

import httpx
import pytest

from tuningfork.config import Settings
from tuningfork.control import ControlClient
from tuningfork.demo.mock import MockTurn
from tuningfork.experiment.sampling import step_resample
from tuningfork.proxy.app import create_app
from tuningfork.store.db import Store
from tuningfork.wire import get_wire
from tuningfork.wire.base import NOT_RUN
from tests.helpers import PROPS, record_pilot, serving

CHAT = "/v1/chat/completions"


def answer(content: str = "", calls: list[dict] | None = None) -> dict:
    message = {"role": "assistant", "content": content, "reasoning_content": "thinking…"}
    if calls:
        message["tool_calls"] = calls
    return {"object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}]}


WRITE = {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "report.json"}'}}


class TestAQuestionAtAHeldCall:
    def test_openai_before_the_model(self):
        request = {"model": "m", "stream": True, "stream_options": {"include_usage": True},
                   "messages": [{"role": "user", "content": "go"}]}
        sent = get_wire("openai-chat").with_user_message(request, "why?")
        assert sent["messages"][-1] == {"role": "user", "content": "why?"}
        assert sent["stream"] is False and "stream_options" not in sent
        assert request["stream"] is True  # the held request itself is untouched

    def test_openai_after_the_model_answers_its_tool_call_as_not_run(self):
        request = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        sent = get_wire("openai-chat").with_user_message(request, "why that key?", answer=answer("", [WRITE]))
        said, result, asked = sent["messages"][-3:]
        assert said["role"] == "assistant" and said["tool_calls"] == [WRITE] and "reasoning_content" not in said
        assert result == {"role": "tool", "tool_call_id": "call_1", "content": NOT_RUN}
        assert asked == {"role": "user", "content": "why that key?"}

    def test_anthropic_before_the_model_joins_the_turn_it_was_to_answer(self):
        request = {"model": "m", "stream": True, "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rows"}]}]}
        sent = get_wire("anthropic-messages").with_user_message(request, "why?")
        assert len(sent["messages"]) == 1 and sent["stream"] is False
        assert sent["messages"][0]["content"][-1] == {"type": "text", "text": "why?"}

    def test_anthropic_after_the_model(self):
        request = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
        sent = get_wire("anthropic-messages").with_user_message(request, "why?", answer=answer("ok", [WRITE]))
        said, asked = sent["messages"][-2:]
        assert [block["type"] for block in said["content"]] == ["text", "tool_use"]  # no thinking
        use_id = said["content"][1]["id"]
        assert asked["content"] == [
            {"type": "tool_result", "tool_use_id": use_id, "content": NOT_RUN},
            {"type": "text", "text": "why?"},
        ]


class TestFiledUnderTheStep:
    def create(self, store: Store, hold_id: str, phase: str = "request"):
        return store.create_experiment(
            kind="resample", arm="original", run_id=None, step_idx=None, step_id=None,
            request={"messages": []}, settings={"n": 1, "origin": {"hold": hold_id, "phase": phase}},
            hold_id=hold_id,
        )

    def test_sampled_first_then_recorded(self, store):
        run_id = record_pilot(store)
        step = store.step(run_id, 1)
        experiment = self.create(store, "abc123")
        assert experiment.run_id is None
        store.link_holds(["abc123"], step, edited=False)
        linked = store.get_experiment(experiment.id)
        assert (linked.run_id, linked.step_idx, linked.step_id) == (run_id, 1, step.id)
        assert linked.settings["origin"]["matches_step"] is True

    def test_recorded_first_then_sampled(self, store):
        run_id = record_pilot(store)
        store.link_holds(["def456"], store.step(run_id, 0), edited=False)
        experiment = self.create(store, "def456")
        assert (experiment.run_id, experiment.step_idx) == (run_id, 0)

    def test_a_request_edited_after_sampling_is_not_the_steps(self, store):
        """The model saw the edit; the samples were of the request as held."""
        from tuningfork.experiment import load_checkpoint

        run_id = record_pilot(store)
        experiment = self.create(store, "0ff1ce")
        store.link_holds(["0ff1ce"], store.step(run_id, 1), edited=True)
        assert store.get_experiment(experiment.id).settings["origin"]["matches_step"] is False
        # A lettered A/B target never comes from it.
        assert step_resample(store, load_checkpoint(store, run_id, 1)) is None

    def test_after_the_model_the_samples_are_the_steps_even_if_edited_before(self, store):
        run_id = record_pilot(store)
        experiment = self.create(store, "beef01", phase="response")
        store.link_holds(["beef01"], store.step(run_id, 1), edited=True)
        assert store.get_experiment(experiment.id).settings["origin"]["matches_step"] is True


def test_a_store_from_before_holds_is_rebuilt_with_every_row(tmp_path):
    home = tmp_path / "old"
    Store(home).close()
    db = sqlite3.connect(home / "tuningfork.db")
    db.executescript("""
        DROP TABLE experiments;
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, group_id TEXT, arm TEXT NOT NULL,
            run_id TEXT NOT NULL, step_idx INTEGER NOT NULL, step_id TEXT NOT NULL,
            request_blob TEXT NOT NULL, settings TEXT, changes TEXT, variant TEXT, base_url TEXT,
            server_defaults TEXT, notes TEXT, created_at REAL NOT NULL
        );
        INSERT INTO experiments VALUES ('e1', 'resample', NULL, 'original', 'r1', 2, 's1', 'b', '{"n": 3}', '{}', '{}', NULL, '{}', '[]', 1.0);
    """)
    db.commit()
    db.close()

    store = Store(home)
    try:
        columns = {row["name"]: row for row in store._conn.execute("PRAGMA table_info(experiments)")}
        indexes = {row["name"] for row in store._conn.execute("PRAGMA index_list(experiments)")}
        kept = store.get_experiment("e1")
    finally:
        store.close()
    assert columns["run_id"]["notnull"] == 0 and "hold_id" in columns
    assert {"experiments_checkpoint", "experiments_group", "experiments_job", "experiments_hold"} <= indexes
    assert (kept.run_id, kept.step_idx, kept.settings) == ("r1", 2, {"n": 3})


# --- at a real held call ---------------------------------------------------------------------


@pytest.fixture
def proxy(tmp_path, mock_server):
    mock_server.props = PROPS
    mock_server.responder = lambda payload: (
        MockTurn(content="Because the skill never names the keys.")
        if "why" in json.dumps(payload["messages"][-1:])
        else MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "report.json"}}])
    )
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    app = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home", max_inflight=2), client=upstream)
    store = app.state.recorder.store
    identity = {"upstream": "http://upstream.test", "n_ctx": 65536, "models": ["mock-model"], "defaults": {"temperature": 1.0}}
    app.state.server = {"id": store.add_server(identity), **identity}
    try:
        with serving(app) as url:
            yield app, url, store
    finally:
        store.close()
        asyncio.run(upstream.aclose())


def held_call(client: ControlClient, url: str, phase: str, results: list) -> dict:
    client.add_breakpoint(phase=phase)
    body = {"model": "mock-model", "stream": True, "messages": [{"role": "user", "content": "write the report"}]}
    threading.Thread(target=lambda: results.append(httpx.post(f"{url}{CHAT}", json=body, timeout=60)), daemon=True).start()
    deadline = time.monotonic() + 10
    while not (holds := [h for h in client.holds() if not h["action"]]):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return holds[0]


@pytest.mark.parametrize("phase", ["request", "response"])
def test_sampling_a_held_call_is_evidence_filed_under_its_step(proxy, phase):
    app, url, store = proxy
    results: list = []
    with ControlClient(url) as client:
        hold = held_call(client, url, phase, results)
        done = client.wait_for_job(client.submit("hold-sample", {"hold": hold["id"], "n": 3})["id"], timeout=30, poll=0.1)
        assert done["status"] == "done", done["error"]
        experiment = store.get_experiment(done["links"]["experiments"][0])
        assert experiment.hold_id == hold["id"] and experiment.run_id is None
        assert done["result"]["distribution"]["total"] == 3
        if phase == "response":
            assert done["result"]["held_letter"] == "A"  # the held answer is the sampled one
        client.decide(hold["id"], "continue")
        deadline = time.monotonic() + 10
        while not results:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    deadline = time.monotonic() + 10
    while store.get_experiment(experiment.id).run_id is None:
        assert time.monotonic() < deadline, "never filed under the step"
        time.sleep(0.05)
    filed = store.get_experiment(experiment.id)
    (run,) = store.runs(limit=5)
    assert (filed.run_id, filed.step_idx) == (run.id, 0)
    assert filed.settings["origin"]["matches_step"] is True
    assert [found.id for found in store.runs(limit=5)] == [run.id]  # samples are never runs


def test_the_console_answers_once_and_is_never_recorded(proxy, mock_server):
    app, url, store = proxy
    results: list = []
    with ControlClient(url) as client:
        hold = held_call(client, url, "response", results)
        before = len(mock_server.requests)
        done = client.wait_for_job(
            client.submit("hold-console", {"hold": hold["id"], "question": "why that path?"})["id"],
            timeout=30, poll=0.1,
        )
        client.decide(hold["id"], "continue")
    assert done["status"] == "done", done["error"]
    assert done["result"]["reply"]["text"] == "Because the skill never names the keys."
    assert done["result"]["hypothesis"] is True and "hypothesis" in done["result"]["note"].lower()
    asked = mock_server.requests[before]
    assert asked["stream"] is False and asked["messages"][-1] == {"role": "user", "content": "why that path?"}
    assert asked["messages"][-2]["role"] == "tool"  # the held tool call, answered as not run
    deadline = time.monotonic() + 10
    while not store.runs(limit=5):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    (run,) = store.runs(limit=5)
    assert run.step_count == 1  # the held call itself; the question is no step


def test_a_call_that_is_no_longer_held_is_refused(proxy):
    app, url, store = proxy
    from tuningfork.control import ControlError

    with ControlClient(url) as client, pytest.raises(ControlError) as refused:
        client.submit("hold-sample", {"hold": "abcdef12"})
    assert "hold" in str(refused.value)
