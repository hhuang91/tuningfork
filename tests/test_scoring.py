"""Scoring a recorded answer, against the mock's toy llama.cpp (see tests/mock_server/toy.py).

The toy model holds two answers that part at one token — `count` or `row_count` — so a scored
answer's heatmap has exactly one fork, and where it is, is known.
"""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.analysis.tokens import from_entries
from tuningfork.backends.formats import get_format
from tuningfork.backends.llamacpp.anthropic import to_chat
from tuningfork.backends.llamacpp.raw import aligned
from tuningfork.backends.raw import RawError
from tuningfork.cli import app
from tuningfork.config import EXPERIMENT_HEADER
from tuningfork.endpoints import ProtectedEndpoint
from tuningfork.ops.catalog import CATALOG
from tuningfork.ops.common import OperationError
from tuningfork.ops.tokens import execute_score, prepare_score
from tuningfork.wire import get_wire
from tests.helpers import PROPS, as_response
from tests.mock_server.toy import props as toy_props

UPSTREAM = "http://upstream.test"
GEMMA = get_format("gemma4")
TOOLS = [
    {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
]
REQUEST = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Summarize data.csv into report.json."},
    ],
    "tools": TOOLS,
    "max_tokens": 4096,
    "stream": True,
}


def answer(key: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I will write the report.",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "write_file", "arguments": f'{{"content":"{{\\"{key}\\": 4}}","path":"report.json"}}'}}],
    }


def raw(key: str) -> str:
    return GEMMA.render(answer(key), thinking=True)


@pytest.fixture
def served(monkeypatch, mock_server):
    """The mock as the only server there is: every client scoring opens reaches it."""
    mock_server.props = toy_props()
    mock_server.toy.answers = [(raw("count"), 0.6), (raw("row_count"), 0.4)]
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


def record(store, served, *, prompt_offset: int = 0, answer_offset: int = 0, wire: str = "openai-chat",
           request: dict | None = None) -> str:
    """A write step whose answer is the toy model's `count` answer, with the token counts the
    server would have reported for it."""
    request = request or REQUEST
    chat = to_chat(request) if wire == "anthropic-messages" else request
    usage = {
        "prompt_tokens": served.toy.input_tokens(chat) + prompt_offset,
        "completion_tokens": len(served.toy.tokenize(raw("count"))) + 1 + answer_offset,
    }
    store.begin_run("scored", "t/1")
    found = get_wire(wire)
    store.record_step(
        request=request, response={**as_response(answer("count"), "tool_calls"), "usage": usage},
        wire=found.name, normalize=found.normalize_message,
    )
    return store.get_run("scored").id


async def score(store, run_id, step="0", **kwargs):
    return await execute_score(store, prepare_score(store, run_id, step, base_url=UPSTREAM, **kwargs))


class TestScoring:
    async def test_the_answer_the_run_gave_scored_token_by_token(self, store, served):
        run_id = record(store, served)

        found = await score(store, run_id)

        view = found.view
        record_ = view.record
        assert (record_.source, record_.method, record_.exact) == ("scored", "grammar", True)
        assert record_.notes["prompt_tokens"] == record_.notes["recorded_prompt_tokens"]
        assert record_.notes["parse_agrees"] is True
        assert view.stream.text() == raw("count")  # the end token's piece is empty
        data = view.to_json()
        key = next(token for token in data["tokens"] if token["text"] == "count")
        assert key["part"] == "tool_call"
        assert [text for text, _ in key["alternatives"]] == ["count", "row_count"]
        assert [p for _, p in key["alternatives"]] == pytest.approx([0.6, 0.4], abs=1e-4)
        assert data["tokens"][-1]["part"] == "marker"  # the model's own end token
        assert data["hotspots"][0]["position"] == key["i"]
        # one request, every token of the answer forced, sent as an experiment's
        forced = served.toy.completions[-1]
        assert forced["grammar"].count("<[") == len(served.toy.tokenize(raw("count")))
        completion = [h for method, path, h in served.seen if path == "/completion"]
        assert completion and all(h.get(EXPERIMENT_HEADER) == "tokens" for h in completion)

    async def test_a_messages_step_is_rendered_as_the_server_converts_it(self, store, served):
        request = {
            "model": "m", "max_tokens": 32000, "stream": True,
            "system": [{"type": "text", "text": "You are an agent."}],
            "tools": [{"name": "write_file", "input_schema": {"type": "object"}},
                      {"name": "read_file", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize data.csv into report.json."}]}],
        }
        run_id = record(store, served, wire="anthropic-messages", request=request)
        found = await score(store, run_id)
        assert found.view.record.exact is True
        assert found.view.record.notes["prompt_tokens"] == found.view.record.notes["recorded_prompt_tokens"]

    async def test_a_different_prompt_is_refused(self, store, served):
        run_id = record(store, served, prompt_offset=3)
        with pytest.raises(OperationError, match="renders to .* tokens here, and the step's had"):
            await score(store, run_id)

    async def test_an_answer_rebuilt_approximately_says_so(self, store, served):
        run_id = record(store, served, answer_offset=1)
        found = await score(store, run_id)
        assert found.view.record.exact is False

    async def test_one_request_per_token_when_grammars_are_refused(self, store, served):
        served.toy.refuse_grammar = True
        run_id = record(store, served)
        found = await score(store, run_id)
        assert found.view.record.method == "per-token"
        key = next(token for token in found.view.to_json()["tokens"] if token["text"] == "count")
        assert [p for _, p in key["alternatives"]] == pytest.approx([0.6, 0.4], abs=1e-4)
        assert found.view.stream.text() == raw("count")

    async def test_a_model_whose_output_cannot_be_read_is_refused(self, store, served):
        served.props = PROPS  # no chat template
        run_id = record(store, served)
        with pytest.raises(OperationError, match="chat template is not one"):
            await score(store, run_id)

    async def test_a_server_that_is_not_llama_cpp_is_refused(self, store, served):
        served.absent_paths.add("/props")
        run_id = record(store, served)
        with pytest.raises(OperationError, match="no /props"):
            await score(store, run_id)

    def test_a_protected_endpoint_is_refused_before_anything_is_sent(self, store, served):
        run_id = record(store, served)
        with pytest.raises(ProtectedEndpoint):
            prepare_score(store, run_id, "0", base_url="http://127.0.0.1:1234")

    def test_the_job_kind(self):
        kind = CATALOG["tokens"]
        assert kind.parse({"run": "r", "step": "0"}).top_k == 8
        with pytest.raises(OperationError):
            kind.parse({"run": "r", "step": "0", "top_k": 21})


def test_a_character_the_server_merged_is_its_tokens_again():
    # llama.cpp b11108: Qwen3.5 writes " ≡" as tokens 20456 and 94, and reports them as one
    # entry, the second's, holding both tokens' bytes
    def entry(token, text, raw, logprob):
        return {"id": token, "token": text, "bytes": raw, "logprob": logprob,
                "top_logprobs": [{"id": token, "token": text, "bytes": raw, "logprob": logprob}]}

    # the alternatives keep their own bytes, as the server reports them
    merged = {**entry(94, " ≡", [32, 226, 137, 161], -0.2),
              "top_logprobs": [{"id": 94, "token": "�", "bytes": [161], "logprob": -0.2}]}
    entries = [entry(1, "a", [97], -0.1), merged, entry(2, "", [], -0.01)]
    out = aligned(entries, [1, 20456, 94, 2])
    assert [e["id"] for e in out] == [1, 20456, 94, 2]
    assert out[1]["logprob"] is None and out[1]["top_logprobs"] == []  # nothing known of it
    assert out[2]["bytes"] == [] and out[2]["logprob"] == -0.2  # its chance is its own; the bytes are not
    stream = from_entries(out)
    assert stream.pieces[94] == [161]  # its own piece, from its alternatives
    assert stream.pieces[20456] == ""  # read again from the tokenizer, where scoring has it
    stream.pieces[20456] = [32, 226, 137]
    assert stream.text() == "a ≡"
    # an entry for every token already is left as it is
    assert aligned(entries[:1], [1]) == entries[:1]
    # entries that follow no token are refused, never laid over the wrong ones
    with pytest.raises(RawError, match="do not follow its tokens"):
        aligned(entries, [1, 20456, 95, 2])


def test_the_command_scores_then_shows(store, served, monkeypatch):
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    run_id = record(store, served)
    shown = CliRunner().invoke(app, ["tokens", run_id[:8], "0", "--score", "--base-url", UPSTREAM, "--home", str(store.home)])
    assert shown.exit_code == 0, shown.output
    assert "scored · grammar · top 8" in shown.output and "rebuilt exactly" in shown.output
    assert "'count' 60% | 'row_count' 40%" in shown.output
