"""What token-level work stands on, measured on a real llama.cpp server: the 5b spike's gates.

Skipped unless TUNINGFORK_REAL_SERVER names a server — never a protected one, which is
refused before anything is sent:

    TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run pytest -m real_server tests/test_real_server.py

Each test is one thing the design relies on. The model must be one
whose output format tuningfork reads (`tuningfork.backends.formats`): Gemma 4, Qwen2.5, Qwen3,
Qwen3.5, Qwen3.6 or Qwen3-Coder.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tuningfork.analysis.tokens import probability

ROOT = os.environ.get("TUNINGFORK_REAL_SERVER", "").rstrip("/")

pytestmark = [
    pytest.mark.real_server,
    pytest.mark.skipif(not ROOT, reason="needs a real llama.cpp server: set TUNINGFORK_REAL_SERVER"),
]

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
BODY = {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": [WEATHER], "max_tokens": 1024}


@pytest.fixture
async def http(tmp_path):
    from tuningfork.endpoints import check_endpoint, guard_hooks

    check_endpoint(ROOT, home=tmp_path, purpose="real-server tests")
    async with httpx.AsyncClient(timeout=600, event_hooks=guard_hooks(home=tmp_path)) as client:
        yield client


async def post(http, path: str, body: dict) -> dict:
    response = await http.post(ROOT + path, json=body)
    assert response.status_code == 200, response.text[:400]
    return response.json()


async def rendered(http, body: dict) -> list[int]:
    prompt = (await post(http, "/apply-template", body))["prompt"]
    return (await post(http, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}))["tokens"]


def joined(entries: list[dict]) -> str:
    return b"".join(bytes(e.get("bytes") or []) for e in entries).decode("utf-8", "replace")


async def server_format(http):
    from tuningfork.backends.formats import format_for

    fmt = format_for((await http.get(ROOT + "/props")).json().get("chat_template"))
    assert fmt is not None, "the model's output format is not one tuningfork reads"
    return fmt


async def test_chat_logprobs_cover_every_generated_token(http):
    """G1: reasoning, channel markers and tool-call syntax each carry an entry, with its id."""
    fmt = await server_format(http)
    answer = await post(http, "/v1/chat/completions", {**BODY, "logprobs": True, "top_logprobs": 5})
    entries = answer["choices"][0]["logprobs"]["content"]
    assert len(entries) == answer["usage"]["completion_tokens"]
    assert all("id" in entry and "bytes" in entry for entry in entries)
    prompt = (await post(http, "/apply-template", BODY))["prompt"]
    parsed = fmt.parse(joined(entries), tools=True, thinking=fmt.thinking(prompt), offered=BODY["tools"])
    # the call's markers had entries: read back from them, the answer calls a tool as the server read it
    assert bool(parsed.get("tool_calls")) == bool(answer["choices"][0]["message"].get("tool_calls"))


async def test_the_raw_path_renders_the_chat_prompt_and_repeats_greedy_chat(http):
    """G2: the ids /apply-template and /tokenize give are the chat prompt, and a greedy raw
    completion from them is the greedy chat answer, token for token."""
    ids = await rendered(http, BODY)
    greedy = await post(http, "/v1/chat/completions", {**BODY, "temperature": 0, "logprobs": True, "top_logprobs": 1})
    assert len(ids) == greedy["usage"]["prompt_tokens"]
    chat = [entry["id"] for entry in greedy["choices"][0]["logprobs"]["content"]]
    raw = await post(http, "/completion", {"prompt": ids, "n_predict": len(chat) + 8, "temperature": 0, "return_tokens": True})
    assert raw["tokens"][: len(chat)] == chat
    # A forced prefix is processed in one batch, where the answer was decoded a token at a time,
    # and the numbers differ in the last places: on a server started with --no-cache-prompt the
    # greedy continuation parted from the chat's at its 8th token, at a near-tie (b11108). With
    # the prompt cache on, the forced tokens were the cached ones of the chat's own decoding.
    forced = await post(http, "/completion", {"prompt": ids + chat[:10], "n_predict": 10, "temperature": 0, "return_tokens": True})
    assert forced["tokens"][:5] == chat[10:15]


async def test_a_messages_request_renders_as_the_server_renders_it(http):
    """G3: the Anthropic conversion mirrored, rendered, has the tokens the server counts."""
    from tuningfork.backends.llamacpp.anthropic import to_chat

    request = {
        "model": "m", "max_tokens": 1024,
        "system": [{"type": "text", "text": "You are an agent."}, {"type": "text", "text": " Be brief."}],
        "tools": [{"name": "get_weather", "description": "Get the weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather in Oslo?"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Use the tool.", "signature": "s"},
                                              {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Rain, 8°C"},
                                         {"type": "text", "text": "And tomorrow?"}]},
        ],
    }
    counted = await post(http, "/v1/messages/count_tokens", request)
    assert len(await rendered(http, to_chat(request))) == counted["input_tokens"]


async def test_the_output_format_reads_and_writes_as_the_server_does(http):
    """G4: raw output parsed as the server parsed it, and a parsed answer written back as the
    raw text the model generated — one of the texts the server reads alike, where it reads
    several as one answer."""
    fmt = await server_format(http)
    answer = await post(http, "/v1/chat/completions", {**BODY, "logprobs": True, "top_logprobs": 1, "seed": 7})
    raw = joined(answer["choices"][0]["logprobs"]["content"])
    message = answer["choices"][0]["message"]
    prompt = (await post(http, "/apply-template", BODY))["prompt"]
    thinking = fmt.thinking(prompt)
    parsed = fmt.parse(raw, tools=True, thinking=thinking, offered=BODY["tools"])
    assert (parsed.get("content") or "") == (message.get("content") or "")
    assert (parsed.get("reasoning_content") or "") == (message.get("reasoning_content") or "")
    assert [c["function"]["arguments"] for c in parsed.get("tool_calls") or []] == [
        c["function"]["arguments"] for c in message.get("tool_calls") or []]
    assert any(raw.startswith(text) for text in fmt.renderings(message, thinking=thinking, offered=BODY["tools"]))


async def test_a_character_written_in_two_tokens_is_scored_token_by_token(http):
    """G6: an answer holding a character the model writes in two tokens or more is scored an
    entry a token. The server can give such a character one entry, the last token's (b11108,
    Qwen3.5's ` ≡`; Gemma 4's byte tokens kept their own); the raw path gives each token its own
    again, the ones merged away unknown."""
    from tuningfork.backends.llamacpp.raw import LlamaRaw

    raw = LlamaRaw(http, ROOT)
    for char in ("≡", "⟦", "𝔸", "🜂"):
        # a token that is no whole character comes back as bytes
        if any(isinstance(piece, list) for _, piece in await raw.pieces(f" {char}")):
            break
    else:
        pytest.skip("this model writes each candidate character whole")
    prompt = await raw.render({"messages": [{"role": "user", "content": "Say hi."}]}, "openai-chat")
    pieces = await raw.pieces(f"We have a {char} b.")
    answer = [token for token, _ in pieces]
    entries = await raw.score(prompt, answer, 3)
    assert [entry["id"] for entry in entries[: len(answer)]] == answer
    assert len(entries) == len(answer) + 1
    # only the character's tokens but its last can have been merged away, nothing known of them
    # (a forced token the server deems impossible has no chance either, but has its alternatives)
    split = [i for i, (_, piece) in enumerate(pieces) if isinstance(piece, list)]
    unknown = [i for i, entry in enumerate(entries[: len(answer)]) if entry["logprob"] is None and not entry["top_logprobs"]]
    assert unknown in ([], split[:-1])  # none merged (Gemma 4), or every one but the last (Qwen3.5)


async def test_a_grammar_forces_an_answer_and_scores_it_as_it_was_sampled(http):
    """G5: one request, a grammar of the answer's ids, gives the probabilities the answer was
    sampled with — read right after, on the same cache."""
    answer = await post(http, "/v1/chat/completions", {**BODY, "logprobs": True, "top_logprobs": 5, "seed": 11})
    entries = answer["choices"][0]["logprobs"]["content"]
    forced = [entry["id"] for entry in entries[:-1]]  # an end token cannot be forced
    ids = await rendered(http, BODY)
    scored = await post(http, "/completion", {
        "prompt": ids, "n_predict": len(forced) + 1, "n_probs": 5, "temperature": 0, "return_tokens": True,
        "grammar": "root ::= " + " ".join(f"<[{token}]>" for token in forced),
    })
    assert scored["tokens"][: len(forced)] == forced
    for sampled, again in zip(entries, scored["completion_probabilities"]):
        a = {t["id"]: probability(t["logprob"]) for t in sampled["top_logprobs"]}
        b = {t["id"]: probability(t["logprob"]) for t in again["top_logprobs"]}
        assert all(abs(a[k] - b[k]) < 0.05 for k in a.keys() & b.keys())
