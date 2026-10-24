"""A minimal harness, for tests and the demo: an agent loop over OpenAI chat-completions, as
a real process.

Launched the way tuningfork launches Hermes or Claude Code — a subprocess, in its own
workspace and home, talking HTTP to the proxy's base URL — so a batch run or a fork goes
through every part a real one does, with the mock server standing in for the model.

It preloads the skills in its home into its system prompt, as Hermes does, with the working
directory named there too; offers `read_file` and `write_file`, run in its working
directory; streams every call; and stops at the first answer with no tool call.

Environment: FAKE_BASE_URL, FAKE_MODEL, FAKE_HOME, FAKE_MAX_TURNS; FAKE_NOISE=1 makes
`read_file` say how long it took, as real tools do — output that differs on every run.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in keys},
                "required": list(keys),
            },
        },
    }
    for name, description, keys in (
        ("read_file", "Read a file.", ("path",)),
        ("write_file", "Write a file.", ("path", "content")),
    )
]


def system_prompt(home: Path) -> str:
    parts = [f"You are a test agent. Working directory: {Path.cwd()}"]
    for skill in sorted((home / "skills").glob("*/SKILL.md")):
        parts.append(f"[Skill {skill.parent.name} preloaded]\n\n{skill.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def call(base_url: str, body: dict) -> dict:
    """One streamed call, reassembled into a message."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer none"},
    )
    message: dict = {"role": "assistant", "content": ""}
    calls: dict[int, dict] = {}
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content"):
                    if isinstance(delta.get(key), str):
                        message[key] = message.get(key, "") + delta[key]
                for fragment in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        fragment.get("index", 0),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    slot["id"] = fragment.get("id") or slot["id"]
                    function = fragment.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message


def run_tool(name: str, arguments: dict) -> str:
    path = Path(arguments.get("path", ""))
    if name == "read_file":
        text = path.read_text(encoding="utf-8") if path.exists() else f"no such file: {path}"
        if os.environ.get("FAKE_NOISE") == "1":
            text += f"\n(read in {random.randint(1, 10_000)} µs)"
        return text
    if name == "write_file":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments.get("content", ""), encoding="utf-8")
        return f"wrote {len(arguments.get('content', ''))} bytes to {path}"
    return f"unknown tool {name}"


def main() -> None:
    base_url = os.environ["FAKE_BASE_URL"]
    model = os.environ.get("FAKE_MODEL", "m")
    home = Path(os.environ["FAKE_HOME"])
    turns = int(os.environ.get("FAKE_MAX_TURNS", "8"))
    messages = [
        {"role": "system", "content": system_prompt(home)},
        {"role": "user", "content": sys.argv[1]},
    ]
    # Seconds to wait before each model call: a test's slow harness.
    delay = float(os.environ.get("FAKE_DELAY", "0"))
    for _ in range(turns):
        if delay:
            time.sleep(delay)
        message = call(
            base_url,
            {"model": model, "messages": messages, "tools": TOOLS, "stream": True,
             "stream_options": {"include_usage": True}},
        )
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for tool_call in message["tool_calls"]:
            function = tool_call["function"]
            arguments = json.loads(function["arguments"] or "{}")
            messages.append(
                {"role": "tool", "tool_call_id": tool_call["id"],
                 "content": run_tool(function["name"], arguments)}
            )
    print(messages[-1].get("content") or "")


if __name__ == "__main__":
    main()
