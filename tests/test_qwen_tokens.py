"""Scoring and forking tokens on a model that writes Qwen3.5's format, against the toy
(`src/tuningfork/demo/toy.py`'s `QwenToy`).

Three things the format's reading takes from the step, each exercised here through the real
path: the prompt opened the reasoning (`thinking`), each argument is typed by the tool's schema
(`offered`), and the server folds a token's entry into the next one's (`aligned`).
"""

from __future__ import annotations

import httpx
import pytest

from tuningfork.backends.formats import get_format
from tuningfork.demo.toy import QwenToy, props as toy_props
from tuningfork.ops.forking import execute_forking, prepare_forking
from tuningfork.ops.tokens import execute_score, prepare_score
from tuningfork.wire import get_wire
from tests.helpers import as_response

UPSTREAM = "http://upstream.test"
QWEN = get_format("qwen3.5")
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


def call(name: str, **values: str) -> str:
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# What the model writes after a prompt that opened its reasoning. A path of "42" is a string
# only by the schema: read without it, it would be a number.
WRITE = "I will write it.\n</think>\n\n" + call("write_file", path="42", lines="3")
READ = "I will write it.\n</think>\n\n" + call("read_file", path="data.csv")


@pytest.fixture
def served(monkeypatch, mock_server):
    mock_server.toy = QwenToy()
    mock_server.props = toy_props(QwenToy.template)
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


def record(store, served, raw: str) -> str:
    """A step whose answer was `raw`, as the server parsed it, with the counts it reported."""
    message = QWEN.parse(raw, tools=True, thinking=True, offered=TOOLS)
    usage = {"prompt_tokens": served.toy.input_tokens(REQUEST), "completion_tokens": len(served.toy.tokenize(raw)) + 1}
    store.begin_run("qwen", "t/1")
    wire = get_wire("openai-chat")
    store.record_step(request=REQUEST, response={**as_response(message, "tool_calls"), "usage": usage},
                      wire=wire.name, normalize=wire.normalize_message)
    return store.get_run("qwen").id


async def score(store, run_id):
    return await execute_score(store, prepare_score(store, run_id, "0", base_url=UPSTREAM))


async def test_an_answer_begun_inside_the_reasoning_its_prompt_opened_is_scored_exactly(store, served):
    # the model's answers follow the turn, the prompt's opening included
    served.toy.answers = [("<think>\n" + WRITE, 0.6), ("<think>\n" + READ, 0.4)]
    run_id = record(store, served, WRITE)

    found = await score(store, run_id)

    record_ = found.view.record
    assert (record_.method, record_.exact) == ("grammar", True)
    # read back by the schema's types, the path "42" stays a string
    assert record_.notes["thinking"] is True and record_.notes["parse_agrees"] is True
    stream = found.view.stream
    assert stream.text() == WRITE and stream.thinking is True
    assert stream.part(0) == "reasoning" and {part for part, _, _ in stream.parts} >= {"marker", "tool_call"}
    name = next(token for token in found.view.to_json()["tokens"] if token["text"] == "write_file")
    assert [text for text, _ in name["alternatives"]] == ["write_file", "read_file"]


async def test_a_token_whose_entry_the_server_folded_into_the_next_is_scored_unknown(store, served):
    served.toy.answers = [("<think>\n" + WRITE, 1.0)]
    (folded,) = served.toy.tokenize(" will")
    served.toy.merged = {folded}
    run_id = record(store, served, WRITE)

    found = await score(store, run_id)

    assert found.view.record.exact is True
    tokens = found.view.to_json()["tokens"]
    will = next(token for token in tokens if token["text"] == " will")
    after = tokens[will["i"] + 1]
    assert will["p"] is None and will["alternatives"] == []  # nothing known of it
    assert after["text"] == " write" and after["p"] == pytest.approx(1.0)  # its own entry, its own piece
    assert found.view.stream.text() == WRITE


async def test_a_continuation_cut_off_while_reasoning_is_read_as_reasoning(store, served):
    # the prompt opened the reasoning: a continuation that stops before closing it is still
    # reasoning — read from its text alone, it would be a reply
    read = "I will read it.\n</think>\n\n" + call("read_file", path="data.csv")
    served.toy.answers = [("<think>\n" + WRITE, 0.6), ("<think>\n" + read, 0.4)]
    run_id = record(store, served, WRITE)
    await score(store, run_id)

    # forked at " write", two tokens allowed past it
    plan = prepare_forking(store, run_id, "0", positions=[2], n=2, max_tokens=4, base_url=UPSTREAM)

    (candidate,) = plan.candidates
    assert (candidate.chosen.text, candidate.part) == (" write", "reasoning")
    result = await execute_forking(store, plan)
    samples = store.samples(result.experiment.id)
    assert len(samples) == plan.planned
    for sample in samples:
        choice = store.sample_response(sample)["choices"][0]
        assert choice["finish_reason"] == "length"
        assert choice["message"]["content"] == "" and choice["message"]["reasoning_content"].startswith("I will ")


async def test_forking_reads_each_continuation_by_the_schemas(store, served):
    served.toy.answers = [("<think>\n" + WRITE, 0.6), ("<think>\n" + READ, 0.4)]
    run_id = record(store, served, WRITE)
    await score(store, run_id)

    plan = prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM)

    (candidate,) = plan.candidates
    assert candidate.chosen.text == "write_file" and candidate.part == "tool_call"
    result = await execute_forking(store, plan)
    (position,) = result.report.positions
    assert (position.baseline.target.successes, position.baseline.target.total) == (4, 4)
    (forced,) = position.alternatives
    assert (forced.target.successes, forced.target.total) == (0, 4)
    # each continuation read as the server would read it: a call, its arguments typed
    responses = [store.sample_response(sample) for sample in store.samples(result.experiment.id)]
    calls = [r["choices"][0]["message"]["tool_calls"][0]["function"] for r in responses]
    assert {c["arguments"] for c in calls} == {'{"path":"42","lines":3}', '{"path":"data.csv"}'}
