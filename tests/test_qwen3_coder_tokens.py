"""Scoring and forking tokens on a model that writes Qwen3-Coder's format, against the toy
(`tests/mock_server/toy.py`'s `Qwen3CoderToy`).

What the format's reading takes from the step, each exercised through the real path: there is
no reasoning to read, and the model may write its first call without the `<tool_call>` the
server reads it alike without, so only the recorded token count — or, without one, the model's
own likelihood — says which text it wrote (`renderings`).
"""

from __future__ import annotations

import httpx
import pytest

from tuningfork.backends.formats import get_format
from tuningfork.ops.forking import execute_forking, prepare_forking
from tuningfork.ops.tokens import execute_score, prepare_score
from tuningfork.wire import get_wire
from tests.helpers import as_response
from tests.mock_server.toy import Qwen3CoderToy, props as toy_props

UPSTREAM = "http://upstream.test"
CODER = get_format("qwen3-coder")
TOOLS = [
    {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "lines": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}}}},
]
REQUEST = {"model": "m", "tools": TOOLS, "max_tokens": 4096, "stream": True, "messages": [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Summarize data.csv into report.json."},
]}


def bare(name: str, **values: str) -> str:
    """A call as Qwen3-Coder sometimes writes its first: without `<tool_call>`."""
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values.items())
    return f"<function={name}>\n{body}</function>\n</tool_call>"


def call(name: str, **values: str) -> str:
    return "<tool_call>\n" + bare(name, **values)


# What the model writes after its turn opens: a line of content, then a call.
WRITE = "I will write it.\n" + call("write_file", path="report.json", lines="3")
WRITE_BARE = "I will write it.\n" + bare("write_file", path="report.json", lines="3")
READ_BARE = "I will write it.\n" + bare("read_file", path="data.csv")


@pytest.fixture
def served(monkeypatch, mock_server):
    mock_server.toy = Qwen3CoderToy()
    mock_server.props = toy_props(Qwen3CoderToy.template)
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


def record(store, served, raw: str, *, counted: bool = True) -> str:
    """A step whose answer was `raw`, as the server parsed it, with the counts it reported — or,
    not `counted`, without the answer's."""
    message = CODER.parse(raw, tools=True, offered=TOOLS)
    usage = {"prompt_tokens": served.toy.input_tokens(REQUEST), "completion_tokens": len(served.toy.tokenize(raw)) + 1}
    if not counted:
        del usage["completion_tokens"]
    store.begin_run("coder", "t/1")
    wire = get_wire("openai-chat")
    store.record_step(request=REQUEST, response={**as_response(message, "tool_calls"), "usage": usage},
                      wire=wire.name, normalize=wire.normalize_message)
    return store.get_run("coder").id


async def score(store, run_id):
    return await execute_score(store, prepare_score(store, run_id, "0", base_url=UPSTREAM))


async def test_an_answer_whose_call_has_its_marker_is_scored_exactly(store, served):
    served.toy.answers = [(WRITE, 0.6), ("I will write it.\n" + call("read_file", path="data.csv"), 0.4)]
    run_id = record(store, served, WRITE)

    found = await score(store, run_id)

    record_ = found.view.record
    assert (record_.method, record_.exact) == ("grammar", True)
    assert record_.notes["thinking"] == "model" and record_.notes["parse_agrees"] is True
    assert "texts_alike" not in record_.notes  # the count told the two texts apart
    stream = found.view.stream
    assert stream.text() == WRITE and stream.part(0) == "content"
    view = found.view.to_json()
    assert view["choice"]["chosen"] == "write_file"
    assert view["choice"]["chance"] == {"write_file": pytest.approx(0.6), "read_file": pytest.approx(0.4)}


async def test_a_first_call_without_its_marker_is_scored_as_the_model_wrote_it(store, served):
    served.toy.answers = [(WRITE_BARE, 0.6), (READ_BARE, 0.4)]
    run_id = record(store, served, WRITE_BARE)

    found = await score(store, run_id)

    # the server read it as the call with its marker; the recorded count says it had none
    assert found.view.record.exact is True and "texts_alike" not in found.view.record.notes
    stream = found.view.stream
    assert stream.text() == WRITE_BARE
    assert {part for part, _, _ in stream.parts} == {"content", "tool_call", "marker"}
    view = found.view.to_json()
    assert view["choice"]["chosen"] == "write_file"
    assert view["choice"]["chance"] == {"write_file": pytest.approx(0.6), "read_file": pytest.approx(0.4)}


async def test_without_the_count_the_text_the_model_finds_likelier_is_scored(store, served):
    served.toy.answers = [(WRITE_BARE, 0.7), (WRITE, 0.3)]
    run_id = record(store, served, WRITE_BARE, counted=False)

    found = await score(store, run_id)

    assert found.view.record.exact is None and found.view.stream.text() == WRITE_BARE
    served.toy.answers = [(WRITE_BARE, 0.3), (WRITE, 0.7)]
    assert (await score(store, run_id)).view.stream.text() == WRITE


async def test_forking_reads_a_continuation_without_its_marker_as_a_call(store, served):
    served.toy.answers = [(WRITE_BARE, 0.6), (READ_BARE, 0.4)]
    run_id = record(store, served, WRITE_BARE)
    await score(store, run_id)

    plan = prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM)

    (candidate,) = plan.candidates
    assert candidate.chosen.text == "write_file" and candidate.part == "tool_call"
    result = await execute_forking(store, plan)
    (position,) = result.report.positions
    assert (position.baseline.target.successes, position.baseline.target.total) == (4, 4)
    (forced,) = position.alternatives
    assert (forced.target.successes, forced.target.total) == (0, 4)
    responses = [store.sample_response(sample) for sample in store.samples(result.experiment.id)]
    calls = [r["choices"][0]["message"]["tool_calls"][0]["function"] for r in responses]
    assert {c["arguments"] for c in calls} == {'{"path":"report.json","lines":3}', '{"path":"data.csv"}'}
