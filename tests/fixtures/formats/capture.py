"""Capture output-format fixtures from a real server: raw generated text beside the server's
own parse of it.

Not a test. Run it against a llama.cpp server to (re)make the fixtures `tests/test_formats.py`
checks a format against:

    TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run python tests/fixtures/formats/capture.py gemma4
    TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run python tests/fixtures/formats/capture.py qwen3.5/qwen3.6-27b
    TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run python tests/fixtures/formats/capture.py qwen3/qwen3-30b-a3b

The second form keeps a format's fixtures a folder for each model that writes it.

Each case asks for an answer with logprobs, which cover every generated token — reasoning,
channel markers and tool-call syntax included — so the joined tokens are the raw text. The
server's parsed message comes in the same response. Protected endpoints are refused. A case
named for some formats only runs for them: one that bans a marker token, so that the model
writes what it sometimes writes without it — Qwen3-Coder's first call with no `<tool_call>`.

A fixture keeps the tools the case offered, which a format that types each argument by its
schema reads with, and the end of the rendered prompt, where a template that opens the
reasoning itself does so. What the prompt did with the reasoning is the format's reading of the
prompt (`thinking`), so a format is captured again once tuningfork knows it — or that reading
taken again from the prompt's end, where only the reading changed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from tuningfork.endpoints import check_endpoint, guard_hooks

ROOT = os.environ.get("TUNINGFORK_REAL_SERVER", "http://127.0.0.1:8081").rstrip("/")
HERE = Path(__file__).parent


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


WEATHER = _tool("get_weather", "Get the current weather for a city", {"city": {"type": "string"}}, ["city"])
WRITE = _tool("write_file", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
EVENT = _tool("create_event", "Create a calendar event", {
    "title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}},
    "location": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}},
    "priority": {"type": "integer"}, "confirmed": {"type": "boolean"}}, ["title"])
EXPENSE = _tool("record_expense", "Record an expense", {
    "amount": {"type": "number"}, "currency": {"type": "string"}, "note": {"type": ["string", "null"]}},
    ["amount", "currency"])

CASES = {
    "text": {"messages": [{"role": "user", "content": "What is 17 times 3? Answer in one short sentence."}]},
    "text-nothink": {"messages": [{"role": "user", "content": "Say hello in French, one word."}],
                     "chat_template_kwargs": {"enable_thinking": False}},
    "one-call": {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": [WEATHER]},
    "parallel": {"messages": [{"role": "user", "content": "Get the current weather in Paris and in Tokyo. Call the tool once for each city, both at once."}],
                 "tools": [WEATHER], "parallel_tool_calls": True},
    "braces": {"messages": [{"role": "user", "content": 'Write the file config.json containing exactly this JSON: {"a": {"b": [1, 2]}, "note": "x: y, {z}"}'}],
               "tools": [WRITE]},
    "nested": {"messages": [{"role": "user", "content": "Create an event titled 'Kickoff' with attendees Ann and Bo, located in Berlin, zip 10115, priority 2, confirmed."}],
               "tools": [EVENT]},
    "unicode": {"messages": [{"role": "user", "content": "Write a file named café.txt whose content is: naïve résumé — 日本語 ✓"}], "tools": [WRITE]},
    "numbers": {"messages": [{"role": "user", "content": "Record an expense of 61.50 USD with no note."}], "tools": [EXPENSE]},
    "text-then-call": {"messages": [{"role": "user", "content": "First tell me in one sentence what you will do, then check the weather in Oslo."}],
                       "tools": [WEATHER]},
    "no-call-with-tools": {"messages": [{"role": "user", "content": "Just say hi. Do not call any tool."}], "tools": [WEATHER]},
    "multiline": {"messages": [{"role": "user", "content": "Write the file notes.txt with three lines: alpha, beta and gamma, one a line."}],
                  "tools": [WRITE]},
    "call-nothink": {"messages": [{"role": "user", "content": "What's the weather in Lima right now?"}], "tools": [WEATHER],
                     "chat_template_kwargs": {"enable_thinking": False}},
    # A hybrid Qwen3 told so in the turn writes an empty reasoning, which the server drops.
    "no-think-switch": {"messages": [{"role": "user", "content": "Say hello in German, one word. /no_think"}]},
    # Qwen3-Coder sometimes opens its first call without its marker, which the server reads alike.
    # A model that does not, banned the marker, writes text — Qwen3-Coder-Next once in a loop: the
    # answer is held short.
    "no-marker": {"messages": [{"role": "user", "content": "What's the weather in Rome right now?"}], "tools": [WEATHER],
                  "ban": ["<tool_call>"], "formats": ["qwen3-coder"], "max_tokens": 1024},
    "text-then-no-marker": {"messages": [{"role": "user", "content": "First tell me in one sentence what you will do, then check the weather in Kyiv."}],
                            "tools": [WEATHER], "ban": ["<tool_call>"], "formats": ["qwen3-coder"], "max_tokens": 1024},
}


def piece(entry: dict) -> str:
    raw = entry.get("bytes")
    return bytes(raw).decode("utf-8", "replace") if isinstance(raw, list) else entry.get("token", "")


async def capture(fmt: str, seeds: tuple[int, ...] = (1, 2)) -> None:
    from tuningfork.backends.formats import format_for

    home = Path(os.environ.get("TUNINGFORK_HOME") or Path.home() / ".tuningfork")
    check_endpoint(ROOT, home=home, purpose="fixture capture")
    out = HERE / fmt
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=1800, event_hooks=guard_hooks(home=home)) as http:
        known = format_for((await http.get(f"{ROOT}/props")).json().get("chat_template"))
        print(f"the server's format: {known.name if known else 'none tuningfork knows yet'}")
        for name, case in CASES.items():
            if fmt.split("/")[0] not in case.get("formats", [fmt.split("/")[0]]):
                continue
            ban = case.get("ban", [])
            case = {key: value for key, value in case.items() if key not in ("ban", "formats")}
            bias = {}
            for text in ban:  # a marker is one token: banned, the model cannot write it
                (token,) = (await http.post(f"{ROOT}/tokenize", json={"content": text, "parse_special": True})).json()["tokens"]
                bias[str(token)] = -100
            prompt = (await http.post(f"{ROOT}/apply-template", json=case)).json()["prompt"]
            for seed in seeds:
                body = {**case, "max_tokens": case.get("max_tokens", 8192), "logprobs": True, "top_logprobs": 1, "seed": seed}
                if bias:
                    body["logit_bias"] = bias
                response = await http.post(f"{ROOT}/v1/chat/completions", json=body)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                entries = choice["logprobs"]["content"]
                message = choice["message"]
                fixture = {
                    "case": name,
                    "seed": seed,
                    "tools": bool(case.get("tools")),
                    "offered": case.get("tools") or [],
                    "parallel": bool(case.get("parallel_tool_calls", True)),
                    "thinking": known.thinking(prompt) if known else None,
                    "prompt_end": prompt[-64:],
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "pieces": [piece(entry) for entry in entries],
                    "message": {
                        "content": message.get("content") or "",
                        "reasoning_content": message.get("reasoning_content") or "",
                        "tool_calls": [
                            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                            for call in message.get("tool_calls") or []
                        ],
                    },
                }
                path = out / f"{name}-{seed}.json"
                path.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"{path.name}: {fixture['completion_tokens']} tokens")


if __name__ == "__main__":
    asyncio.run(capture(sys.argv[1] if len(sys.argv) > 1 else "gemma4"))
