"""Gemma 4's output format, as llama.cpp reads it (`common/parsers/gemma4.cpp`, b11108).

The raw text of an answer:

    <|channel>thought
    …reasoning…<channel|>…content…<|tool_call>call:NAME{key:value,…}<tool_call|>

then an end token: `<|tool_response>` after a call, the end of turn after text. With
reasoning on, the thought channel is always opened — an answer with nothing to think reads
`<|channel>thought\\n<channel|>` — and the server reports the empty reasoning as none. Tool
arguments are not JSON: strings are fenced by `<|"|>`, keys are bare, numbers, booleans and
null are written as JSON writes them, and nothing separates the elements. The server turns
them into compact JSON with each number as written.

Every channel and call delimiter is a single special token, so a stream's parts can be told
from its pieces alone.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from tuningfork.backends.formats.base import FormatError

CHANNEL, CHANNEL_END = "<|channel>", "<channel|>"
THOUGHT = "thought"
CALL, CALL_END = "<|tool_call>", "<tool_call|>"
CALL_PREFIX = CALL + "call:"
QUOTE = '<|"|>'
MODEL_TURN = "<|turn>model\n"
#: The pieces of the special tokens that end an answer (empty in llama.cpp's logprobs).
ENDS = ("", "<|tool_response>", "<turn|>", "<eos>")

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")
_KEY = re.compile(r"[^:}]+")
_SPACE = " \t\n\r"


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i] in _SPACE:
        i += 1
    return i


def _value(text: str, i: int) -> tuple[str, int]:
    if text.startswith(QUOTE, i):
        end = text.find(QUOTE, i + len(QUOTE))
        if end < 0:
            raise FormatError(f"a string opened at {i} is never closed")
        return json.dumps(text[i + len(QUOTE) : end], ensure_ascii=False), end + len(QUOTE)
    if text.startswith("{", i):
        return _dict(text, i)
    if text.startswith("[", i):
        return _array(text, i)
    match = _NUMBER.match(text, i)
    if not match:
        raise FormatError(f"no value at {i}: {text[i : i + 24]!r}")
    return match.group(0), match.end()


def _dict(text: str, i: int) -> tuple[str, int]:
    i = _skip_space(text, i + 1)
    if text.startswith("}", i):
        return "{}", i + 1
    members = []
    while True:
        match = _KEY.match(text, i)
        if not match or not text.startswith(":", match.end()):
            raise FormatError(f"no key at {i}: {text[i : i + 24]!r}")
        value, i = _value(text, _skip_space(text, match.end() + 1))
        members.append(json.dumps(match.group(0), ensure_ascii=False) + ":" + value)
        if text.startswith(",", i):
            i = _skip_space(text, i + 1)
            continue
        i = _skip_space(text, i)
        if text.startswith("}", i):
            return "{" + ",".join(members) + "}", i + 1
        raise FormatError(f"an object is not closed at {i}: {text[i : i + 24]!r}")


def _array(text: str, i: int) -> tuple[str, int]:
    i = _skip_space(text, i + 1)
    if text.startswith("]", i):
        return "[]", i + 1
    items = []
    while True:
        value, i = _value(text, i)
        items.append(value)
        if text.startswith(",", i):
            i = _skip_space(text, i + 1)
            continue
        i = _skip_space(text, i)
        if text.startswith("]", i):
            return "[" + ",".join(items) + "]", i + 1
        raise FormatError(f"an array is not closed at {i}: {text[i : i + 24]!r}")


class _Number(str):
    """A JSON number kept as written: `61.50` must come back as `61.50`, not `61.5`."""


def _written(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_written(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_written(item) for item in value) + "]"
    if isinstance(value, _Number):
        return str(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return QUOTE + str(value) + QUOTE


class Gemma4:
    name = "gemma4"
    models = ("Gemma 4",)
    call_name_prefix = CALL_PREFIX[len(CALL):]

    def detect(self, template: str) -> bool:
        # llama.cpp's own rule for choosing this parser.
        return CALL_PREFIX in template

    def thinking(self, prompt: str) -> bool:
        return "<|think|>" in prompt

    def parse(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool = True,
        thinking: bool | None = None,
        offered: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Mirrors the server's PEG parser: messages (a thought, then content) up to the first
        call, then the calls. Text between the last content and the first call that the
        grammar skips is skipped here too; without tools, a call is content. The model opens
        its own thought channel and writes every argument's type, so neither `thinking` nor
        the offered tools change what is read."""
        i = len(MODEL_TURN) if text.startswith(MODEL_TURN) else 0
        reasoning, content = "", ""
        stops = (CHANNEL, CHANNEL_END, CALL) if tools else (CHANNEL, CHANNEL_END)
        while i < len(text):
            start = i
            if text.startswith(CHANNEL, i):
                j = i
                while text.startswith(CHANNEL, j) and not text.startswith(THOUGHT, j + len(CHANNEL)):
                    j += len(CHANNEL)  # an empty channel
                if not text.startswith(CHANNEL + THOUGHT, j):
                    break
                j = _skip_space(text, j + len(CHANNEL + THOUGHT))
                end = text.find(CHANNEL_END, j)
                if end < 0:  # cut off while thinking
                    reasoning += text[j:]
                    i = len(text)
                    break
                reasoning += text[j:end]
                i = end + len(CHANNEL_END)
            found = [at for at in (text.find(stop, i) for stop in stops) if at >= 0]
            upto = min(found) if found else len(text)
            content += text[i:upto]
            i = upto
            if i == start:
                break
        calls: list[dict[str, Any]] = []
        if tools:
            i = text.find(CALL, i)
            while i >= 0 and text.startswith(CALL_PREFIX, i) and (parallel or not calls):
                name_start = i + len(CALL_PREFIX)
                brace = text.find("{", name_start)
                if brace < 0:
                    raise FormatError(f"a call at {i} has no arguments")
                arguments, i = _dict(text, brace)
                if not text.startswith(CALL_END, i):
                    raise FormatError(f"a call is not closed at {i}: {text[i : i + 24]!r}")
                calls.append({
                    "id": f"call_{len(calls)}",
                    "type": "function",
                    "function": {"name": text[name_start:brace], "arguments": arguments},
                })
                i += len(CALL_END)
                if not text.startswith(CALL, i):
                    break
            if i >= 0 and text.startswith(CALL, i) and not text.startswith(CALL_PREFIX, i):
                raise FormatError(f"a call at {i} does not start with 'call:'")
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return message

    def parts(self, pieces: Sequence[str], *, thinking: bool | None = None) -> list[tuple[str, int, int]]:
        runs: list[tuple[str, int, int]] = []

        def mark(kind: str, position: int) -> None:
            if runs and runs[-1][0] == kind and runs[-1][2] == position:
                runs[-1] = (kind, runs[-1][1], position + 1)
            else:
                runs.append((kind, position, position + 1))

        state, header = "content", False
        for position, piece in enumerate(pieces):
            if piece == CHANNEL:
                mark("marker", position)
                header = True
            elif header and piece == THOUGHT:
                mark("marker", position)
                state = "reasoning"
            elif header and state == "reasoning" and piece.strip(_SPACE) == "":
                mark("marker", position)  # the space before the reasoning, which it skips
            elif piece == CHANNEL_END:
                mark("marker", position)
                state, header = "content", False
            elif piece == CALL:
                mark("marker", position)
                state, header = "tool_call", False
            elif piece == CALL_END:
                mark("marker", position)
                state = "content"
            elif piece in ENDS:
                mark("marker", position)
            else:
                header = False
                mark(state, position)
        return runs

    def render(self, message: dict[str, Any], *, thinking: bool) -> str:
        text = ""
        reasoning = message.get("reasoning_content") or ""
        if reasoning or thinking:
            text += CHANNEL + THOUGHT + "\n" + reasoning + CHANNEL_END
        content = message.get("content")
        if isinstance(content, str):
            text += content
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}", parse_float=_Number, parse_int=_Number)
                except json.JSONDecodeError as error:
                    raise FormatError(f"a call's arguments are not JSON: {error}") from None
            text += CALL_PREFIX + (function.get("name") or "") + _written(arguments or {}) + CALL_END
        return text
