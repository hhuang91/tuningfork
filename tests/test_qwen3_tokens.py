"""Scoring and forking tokens on a model that writes Qwen3's format, or Qwen2.5's, against the
toy (`tests/mock_server/toy.py`'s `Qwen3Toy`).

What the format's reading takes from the step, each exercised through the real path: the model
opens its reasoning itself, or the prompt closed it; a reasoning the model left empty reads as
none, so only the recorded token count says it was written (`renderings`); and a call's
arguments are its JSON as the model wrote it.
"""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.backends.formats import get_format
from tuningfork.backends.llamacpp.anthropic import to_chat
from tuningfork.cli import app
from tuningfork.experiment.scoring import _chance, recorded_message
from tuningfork.ops.forking import execute_forking, prepare_forking
from tuningfork.ops.tokens import execute_score, prepare_score
from tuningfork.wire import get_wire
from tuningfork.wire.anthropic import AnthropicMessages
from tests.helpers import as_response, messages_stream
from tests.mock_server.toy import Qwen3Toy, props as toy_props

UPSTREAM = "http://upstream.test"
ANTHROPIC = AnthropicMessages()
QWEN3, QWEN25 = get_format("qwen3"), get_format("qwen2.5")
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
OFF = {**REQUEST, "chat_template_kwargs": {"enable_thinking": False}}


def call(name: str, arguments: str) -> str:
    return f'<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>'


# What the model writes after its turn opens: its own reasoning, then a call, the arguments
# spaced as it spaces them.
WRITE = "<think>\nI will write it.\n</think>\n\n" + call("write_file", '{"path": "report.json", "lines": 3}')
READ = "<think>\nI will write it.\n</think>\n\n" + call("read_file", '{"path": "data.csv"}')


def serve(monkeypatch, mock_server, toy: Qwen3Toy):
    mock_server.toy = toy
    mock_server.props = toy_props(toy.template)
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


@pytest.fixture
def served(monkeypatch, mock_server):
    return serve(monkeypatch, mock_server, Qwen3Toy())


def record(store, served, raw: str, *, request: dict = REQUEST, fmt=QWEN3, thinking: bool | str = "model",
           count: int | None = None, counted: bool = True) -> str:
    """A step whose answer was `raw`, as the server parsed it, with the counts it reported — or,
    not `counted`, without the answer's."""
    message = fmt.parse(raw, tools=True, thinking=thinking, offered=TOOLS)
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    answered = len(served.toy.tokenize(raw)) + 1 if count is None else count
    usage = {"prompt_tokens": served.toy.input_tokens(request), "completion_tokens": answered}
    if not counted:
        del usage["completion_tokens"]
    store.begin_run("qwen3", "t/1")
    wire = get_wire("openai-chat")
    store.record_step(request=request, response={**as_response(message, finish), "usage": usage},
                      wire=wire.name, normalize=wire.normalize_message)
    return store.get_run("qwen3").id


async def score(store, run_id):
    return await execute_score(store, prepare_score(store, run_id, "0", base_url=UPSTREAM))


async def test_an_answer_whose_reasoning_the_model_opened_is_scored_exactly(store, served):
    served.toy.answers = [(WRITE, 0.6), (READ, 0.4)]
    run_id = record(store, served, WRITE)

    found = await score(store, run_id)

    record_ = found.view.record
    assert (record_.method, record_.exact) == ("grammar", True)
    assert record_.notes["format"] == "qwen3" and record_.notes["parse_agrees"] is True
    # the prompt left the reasoning to the model, whose answer shows it
    assert record_.notes["thinking"] == "model"
    stream = found.view.stream
    assert stream.text() == WRITE and stream.thinking == "model"
    assert stream.part(0) == "marker" and {part for part, _, _ in stream.parts} >= {"reasoning", "tool_call"}
    name = next(token for token in found.view.to_json()["tokens"] if token["text"] == "write_file")
    assert [text for text, _ in name["alternatives"]] == ["write_file", "read_file"]
    # the tool's name is found after the prefix the call's part begins with, newline and all
    choice = found.view.to_json()["choice"]
    assert choice["chosen"] == "write_file"
    assert choice["chance"] == {"write_file": pytest.approx(0.6), "read_file": pytest.approx(0.4)}


async def test_an_empty_reasoning_is_scored_with_the_tokens_the_model_wrote(store, served):
    # A hybrid Qwen3 that chose not to reason still opened and closed its reasoning; the server
    # reads the empty one as none, so the recorded token count is what says it was written.
    raw = "<think>\n\n</think>\n\nI will write it."
    served.toy.answers = [(raw, 1.0)]
    run_id = record(store, served, raw)

    found = await score(store, run_id)

    assert found.view.record.exact is True
    assert found.view.stream.text() == raw


async def test_calls_a_blank_line_apart_are_scored_as_the_model_wrote_them(store, served):
    # A blank line is one token as a newline is: the count cannot tell which the model wrote
    # between two calls, and the server skips either. The model's own likelihood can.
    raw = "<think>\nBoth.\n</think>\n\n" + call("read_file", '{"path": "a.csv"}') + "\n\n" + call("read_file", '{"path": "b.csv"}')
    served.toy.answers = [(raw, 1.0)]
    run_id = record(store, served, raw)

    found = await score(store, run_id)

    assert found.view.record.exact is True and found.view.record.notes["texts_alike"] == 2
    assert found.view.stream.text() == raw


async def test_calls_a_line_apart_as_the_template_writes_them_keep_the_first_text(store, served):
    raw = "<think>\nBoth.\n</think>\n\n" + call("read_file", '{"path": "a.csv"}') + "\n" + call("read_file", '{"path": "b.csv"}')
    served.toy.answers = [(raw, 1.0)]
    run_id = record(store, served, raw)

    found = await score(store, run_id)

    assert found.view.record.exact is True and found.view.record.notes["texts_alike"] == 2
    assert found.view.stream.text() == raw


async def test_a_probe_of_a_base_chosen_among_texts_read_alike_says_so(store, served):
    head, first = "<think>\nBoth.\n</think>\n\n", call("read_file", '{"path": "a.csv"}')
    raw = head + first + "\n\n" + call("read_file", '{"path": "b.csv"}')
    served.toy.answers = [(raw, 0.6), (head + first + "\n\n" + call("read_file", '{"path": "c.csv"}'), 0.4)]
    run_id = record(store, served, raw)
    await score(store, run_id)

    plan = prepare_forking(store, run_id, "0", n=1, base_url=UPSTREAM)

    assert any("the likelier of texts the server reads" in warning for warning in plan.warnings)


async def test_a_tie_keeps_the_templates_own_spacing(store, served):
    # where the model finds both alike, nothing says which it wrote: the template's own is kept
    one, two = call("read_file", '{"path": "a.csv"}'), call("read_file", '{"path": "b.csv"}')
    head = "<think>\nBoth.\n</think>\n\n"
    served.toy.answers = [(head + one + "\n" + two, 0.5), (head + one + "\n\n" + two, 0.5)]
    run_id = record(store, served, head + one + "\n\n" + two)

    found = await score(store, run_id)

    assert found.view.stream.text() == head + one + "\n" + two


async def test_where_the_count_is_unknown_no_texts_are_said_alike(store, served):
    raw = "<think>\nBoth.\n</think>\n\n" + call("read_file", '{"path": "a.csv"}') + "\n\n" + call("read_file", '{"path": "b.csv"}')
    served.toy.answers = [(raw, 1.0)]
    run_id = record(store, served, raw, counted=False)

    found = await score(store, run_id)

    # the likelier is still kept, and nothing is claimed of the count
    assert found.view.record.exact is None and "texts_alike" not in found.view.record.notes
    assert found.view.stream.text() == raw


def test_a_token_outside_the_top_k_is_less_likely_than_any_inside_it():
    entry = {"id": 1, "logprob": -2.0, "top_logprobs": [{"id": 2, "logprob": -0.5}, {"id": 1, "logprob": -2.0}]}
    assert (_chance(entry, 1), _chance(entry, 2), _chance(entry, 3)) == (-2.0, -0.5, float("-inf"))


async def test_an_answer_after_a_prompt_that_closed_the_reasoning_is_scored_exactly(store, served):
    # with reasoning off the prompt closes it, and the answer is the call alone
    closed = "<think>\n\n</think>\n\n"
    answer = call("write_file", '{"path": "report.json", "lines": 3}')
    served.toy.answers = [(closed + answer, 1.0)]
    run_id = record(store, served, answer, request=OFF, thinking=False)

    found = await score(store, run_id)

    assert (found.view.record.exact, found.view.record.notes["parse_agrees"]) == (True, True)
    assert found.view.stream.text() == answer and found.view.stream.thinking is False


async def test_an_answer_after_a_prompt_that_closed_the_reasoning_keeps_its_first_space(store, served):
    # the server read the prompt's closed reasoning before the answer: the newline it begins
    # with is content, not space to skip
    answer = "\nI will write it."
    served.toy.answers = [("<think>\n\n</think>\n\n" + answer, 1.0)]
    run_id = record(store, served, answer, request=OFF, thinking=False)

    found = await score(store, run_id)

    assert (found.view.record.exact, found.view.record.notes["parse_agrees"]) == (True, True)
    assert found.view.stream.text() == answer and found.view.stream.part(0) == "content"


async def test_the_token_count_decides_before_the_likelihood(store, served):
    # the model finds an empty reasoning likelier, but the run's count says it wrote none
    served.toy.answers = [("<think>\n\n</think>\n\nHi.", 0.9), ("Hi.", 0.1)]
    run_id = record(store, served, "Hi.")

    found = await score(store, run_id)

    assert found.view.record.exact is True and found.view.stream.text() == "Hi."
    assert "texts_alike" not in found.view.record.notes


async def test_where_no_text_has_the_recorded_count_the_templates_own_is_scored(store, served):
    served.toy.answers = [("<think>\n\n</think>\n\nHi.", 0.9), ("Hi.", 0.1)]
    run_id = record(store, served, "Hi.", count=99)

    found = await score(store, run_id)

    assert found.view.record.exact is False and found.view.stream.text() == "Hi."


def test_the_command_says_which_text_it_scored(store, served, monkeypatch):
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    raw = "<think>\nBoth.\n</think>\n\n" + call("read_file", '{"path": "a.csv"}') + "\n\n" + call("read_file", '{"path": "b.csv"}')
    served.toy.answers = [(raw, 1.0)]
    run_id = record(store, served, raw)

    shown = CliRunner().invoke(app, ["tokens", run_id[:8], "0", "--score", "--base-url", UPSTREAM, "--home", str(store.home)])

    assert shown.exit_code == 0, shown.output
    assert "rebuilt as the likelier of 2 texts the server reads alike" in shown.output
    assert "rebuilt exactly" not in shown.output


async def test_a_qwen25_answer_is_scored_exactly(store, monkeypatch, mock_server):
    served = serve(monkeypatch, mock_server, Qwen3Toy(reasoning=False))
    answer = "Reading it first.\n" + call("read_file", '{"path": "data.csv"}')
    served.toy.answers = [(answer, 1.0)]
    run_id = record(store, served, answer, fmt=QWEN25)

    found = await score(store, run_id)

    record_ = found.view.record
    assert (record_.exact, record_.notes["format"], record_.notes["parse_agrees"]) == (True, "qwen2.5", True)
    assert {part for part, _, _ in found.view.stream.parts} == {"content", "marker", "tool_call"}
    assert found.view.to_json()["choice"]["chance"] == {"read_file": pytest.approx(1.0)}


#: The task as Claude Code sends it, over Anthropic Messages.
MESSAGES = {"model": "m", "max_tokens": 4096, "stream": True,
            "system": [{"type": "text", "text": "You are an agent."}],
            "tools": [{"name": t["function"]["name"], "input_schema": t["function"]["parameters"]} for t in TOOLS],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize data.csv into report.json."}]}]}
#: What the model wrote: its characters unescaped, as the server streamed them.
WRITTEN = '{"path": "café.json", "lines": 3}'
STREAMED = messages_stream(("write_file", [WRITTEN[:12], WRITTEN[12:]]), text="Writing it.\n")


def record_messages(store, served, *, stream_kept: bool) -> str:
    """A Messages step stored as tuningfork stored one before its wire kept the streamed text:
    each call's input written again, escaped — with the raw stream beside it, or not."""
    answer = "Writing it.\n" + call("write_file", WRITTEN)
    served.toy.answers = [(answer, 1.0)]
    stored = ANTHROPIC.canonical_response(ANTHROPIC.parse(STREAMED, event_stream=True))
    assert stored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] != WRITTEN
    usage = {"prompt_tokens": served.toy.input_tokens(to_chat(MESSAGES)),
             "completion_tokens": len(served.toy.tokenize(answer)) + 1}
    store.begin_run("messages", "t/1")
    wire = get_wire("anthropic-messages")
    store.record_step(request=MESSAGES, response={**stored, "usage": usage},
                      raw_response_blob=store.blobs.put(STREAMED) if stream_kept else None,
                      params={"streamed": stream_kept}, wire=wire.name, normalize=wire.normalize_message)
    return store.get_run("messages").id


async def test_a_messages_steps_arguments_are_scored_as_the_server_streamed_them(store, monkeypatch, mock_server):
    served = serve(monkeypatch, mock_server, Qwen3Toy(reasoning=False))
    run_id = record_messages(store, served, stream_kept=True)

    found = await score(store, run_id)

    # read again from the stream the step kept, they are the model's own text
    assert found.view.record.exact is True
    assert found.view.stream.text() == "Writing it.\n" + call("write_file", WRITTEN)


def stored_step(store, raw: bytes, *, arguments: list[str] | None = None):
    """A Messages step stored as before its wire kept the streamed text, its raw stream beside
    it — or, given `arguments`, with these as its calls' stored arguments."""
    stored = ANTHROPIC.canonical_response(ANTHROPIC.parse(raw, event_stream=True))
    for call_, text in zip(stored["choices"][0]["message"]["tool_calls"], arguments or []):
        call_["function"]["arguments"] = text
    store.begin_run("messages", "t/1")
    wire = get_wire("anthropic-messages")
    store.record_step(request=MESSAGES, response=stored, raw_response_blob=store.blobs.put(raw),
                      params={"streamed": True}, wire=wire.name, normalize=wire.normalize_message)
    return store.step(store.get_run("messages").id, 0)


def test_bytes_written_again_around_a_secret_are_not_read_for_the_arguments(store):
    # The recorder wrote them again from the masked answer: not the server's text, which the
    # stored arguments keep, masked where they stand.
    stored = ANTHROPIC.canonical_response(ANTHROPIC.parse(STREAMED, event_stream=True))
    store.begin_run("messages", "t/1")
    wire = get_wire("anthropic-messages")
    store.record_step(request=MESSAGES, response=stored, raw_response_blob=store.blobs.put(STREAMED),
                      params={"streamed": True, "raw_redacted": "rendered"}, wire=wire.name,
                      normalize=wire.normalize_message)

    message, _ = recorded_message(store, store.step(store.get_run("messages").id, 0))

    arguments = message["tool_calls"][0]["function"]["arguments"]
    assert arguments == stored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] != WRITTEN


def test_each_call_takes_its_own_streamed_text(store):
    texts = ['{"path": "café.json", "lines": 3}', '{"path": "naïve.csv"}']
    step = stored_step(store, messages_stream(("write_file", [texts[0]]), ("read_file", [texts[1]]), text="Both.\n"))

    message, _ = recorded_message(store, step)

    assert [c["function"]["arguments"] for c in message["tool_calls"]] == texts


#: A call whose text names a key twice: its values are the last's, and the text says more.
TWICE = messages_stream(("write_file", ['{"path": "sk-not-for-the-stream", "path": "café.json", "lines": 3}']))


def test_a_streamed_text_naming_a_key_twice_is_not_taken(store):
    # what the text says beyond its values may be what a redaction masked in the stored call
    message, _ = recorded_message(store, stored_step(store, TWICE))
    assert message["tool_calls"][0]["function"]["arguments"] == '{"path": "caf\\u00e9.json", "lines": 3}'


def test_a_streamed_text_naming_a_key_twice_is_not_taken_for_stored_arguments_that_are_no_json(store):
    message, _ = recorded_message(store, stored_step(store, TWICE, arguments=["not JSON"]))
    assert message["tool_calls"][0]["function"]["arguments"] == "not JSON"


def test_a_streamed_text_that_reads_otherwise_than_the_stored_call_is_not_taken(store):
    # the stored answer is the one scored: a stream that says something else does not replace it
    step = stored_step(store, STREAMED, arguments=['{"path": "[masked]", "lines": 3}'])

    message, _ = recorded_message(store, step)

    assert message["tool_calls"][0]["function"]["arguments"] == '{"path": "[masked]", "lines": 3}'


async def test_a_messages_step_without_its_stream_is_scored_with_its_input_written_again(store, monkeypatch, mock_server):
    served = serve(monkeypatch, mock_server, Qwen3Toy(reasoning=False))
    run_id = record_messages(store, served, stream_kept=False)

    found = await score(store, run_id)

    assert found.view.record.exact is False


async def test_forking_reads_each_continuation_as_the_server_would(store, served):
    served.toy.answers = [(WRITE, 0.6), (READ, 0.4)]
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
    # each continuation's call read with its arguments as the model wrote them
    responses = [store.sample_response(sample) for sample in store.samples(result.experiment.id)]
    calls = [r["choices"][0]["message"]["tool_calls"][0]["function"] for r in responses]
    assert {c["arguments"] for c in calls} == {'{"path": "report.json", "lines": 3}', '{"path": "data.csv"}'}


async def test_a_continuation_cut_off_while_reasoning_is_read_as_reasoning(store, served):
    read = "<think>\nI will read it.\n</think>\n\n" + call("read_file", '{"path": "data.csv"}')
    served.toy.answers = [(WRITE, 0.6), (read, 0.4)]
    run_id = record(store, served, WRITE)
    await score(store, run_id)

    # forked at " write" (after "<think>", "\n", "I", " will"), two tokens allowed past it: the
    # reasoning never closes
    plan = prepare_forking(store, run_id, "0", positions=[4], n=2, max_tokens=4, base_url=UPSTREAM)

    (candidate,) = plan.candidates
    assert (candidate.chosen.text, candidate.part) == (" write", "reasoning")
    result = await execute_forking(store, plan)
    samples = store.samples(result.experiment.id)
    assert len(samples) == plan.planned
    for sample in samples:
        choice = store.sample_response(sample)["choices"][0]
        assert choice["finish_reason"] == "length"
        assert choice["message"]["content"] == "" and choice["message"]["reasoning_content"].startswith("I will ")
