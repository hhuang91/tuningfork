"""Qwen3.5's output format, as llama.cpp reads it (`common/parsers/qwen3-coder.cpp`, b11108).

llama.cpp names the parser after Qwen3-Coder, which wrote the format first, and reads Qwen3.6,
Nemotron Nano 3 and StepFun-3.5-Flash with it too. The raw text of an answer, after a prompt
that opened the reasoning — the template's default, `<think>\\n` closing the prompt:

    …reasoning…
    </think>

    …content…

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

then an end token. With reasoning off the prompt closes the reasoning too
(`<think>\\n\\n</think>\\n\\n`), and the answer is the content and the calls. A value is typed
by the tool's schema, as the server types it: raw text where the parameter can only be a
string, JSON where it can be no string, and otherwise JSON where it reads as one of the
schema's other types, else text. The server builds the arguments from the values as written —
a string escaped into JSON, anything else kept as the model wrote it.

`<think>`, `</think>`, `<tool_call>` and `</tool_call>` are single tokens; the rest of a call is
ordinary text.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from tuningfork.backends.formats.base import FormatError

THINK, THINK_END = "<think>", "</think>"
CALL, CALL_END = "<tool_call>", "</tool_call>"
FUNCTION, FUNCTION_END = "<function=", "</function>"
PARAMETER, ARG_CLOSE = "<parameter=", "\n</parameter>\n"
ASSISTANT = "<|im_start|>assistant"
#: The pieces of the special tokens that end an answer (empty in llama.cpp's logprobs).
ENDS = ("", "<|im_end|>", "<|endoftext|>")

_SPACE = " \t\n\r"
_ALL = frozenset({"string", "number", "integer", "boolean", "null", "array", "object"})
_JSON = json.JSONDecoder()


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i] in _SPACE:
        i += 1
    return i


def _opened(text: str) -> bool:
    """Whether an answer read without its prompt started inside the reasoning: it closes a
    reasoning it never opened. An answer cut off before it closed reads as content."""
    end = text.find(THINK_END)
    return end >= 0 and THINK not in text[:end]


# --- the schema's types, as llama.cpp reads them (common/json-schema.cpp) ---------------------


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _types(schema: Any, root: dict[str, Any], seen: frozenset[str] = frozenset()) -> frozenset[str]:
    """The JSON types a schema admits: `common_chat_schema::value_types`."""
    if not isinstance(schema, dict):
        return _ALL
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or ref in seen:
            return frozenset()  # a cycle contributes no type
        target: Any = root
        for key in ref.removeprefix("#").strip("/").split("/") if ref.startswith("#") else ():
            target = target.get(key) if isinstance(target, dict) else None
        return _types(target, root, seen | {ref}) if isinstance(target, dict) and target is not root else _ALL
    if "oneOf" in schema or "anyOf" in schema:
        children = schema.get("oneOf") if "oneOf" in schema else schema.get("anyOf")
        return frozenset().union(*(_types(child, root, seen) for child in children or []))
    kind = schema.get("type")
    if isinstance(kind, list):
        return frozenset().union(*(_types({**schema, "type": each}, root, seen) for each in kind))
    if "const" in schema:
        return frozenset({_json_type(schema["const"])})
    if "enum" in schema and isinstance(schema["enum"], list):
        return frozenset(_json_type(value) for value in schema["enum"])

    def every(children: Any) -> frozenset[str]:
        found = _ALL
        for child in children or []:
            found &= _types(child, root, seen)
        return found

    has_properties = "properties" in schema or (
        "additionalProperties" in schema and schema["additionalProperties"] is not True
    )
    if not kind:
        if has_properties:
            return frozenset({"object"})
        if "allOf" in schema:
            return every(schema["allOf"])
        if "items" in schema or "prefixItems" in schema:
            return frozenset({"array"})
        if any(key in schema for key in ("pattern", "minLength", "maxLength", "format")):
            return frozenset({"string"})
        return _ALL
    if kind in ("object", "string") and "allOf" in schema and not (kind == "object" and has_properties):
        return every(schema["allOf"])
    return {
        "object": frozenset({"object"}), "string": frozenset({"string"}), "array": frozenset({"array"}),
        "integer": frozenset({"integer"}), "number": frozenset({"number", "integer"}),
        "boolean": frozenset({"boolean"}), "null": frozenset({"null"}),
    }.get(kind, _ALL)


def _parameters(offered: Sequence[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """A tool's parameters schema — its properties, and the root a `$ref` resolves in."""
    for tool in offered:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            root = function.get("parameters")
            root = root if isinstance(root, dict) else {}
            properties = root.get("properties")
            return (properties if isinstance(properties, dict) else {}), root
    return None


def _json_value(text: str, i: int, admitted: frozenset[str]) -> tuple[str, int] | None:
    """A JSON value of one of the admitted types at `i`, followed by the argument's close: its
    text as written, and where the close starts."""
    try:
        value, end = _JSON.raw_decode(text, i)
    except ValueError:
        return None
    kind = _json_type(value)
    allowed = kind in admitted or (kind == "integer" and "number" in admitted)
    if not allowed or kind == "string" or not text.startswith(ARG_CLOSE, end):
        return None
    return text[i:end], end


class Qwen3Coder:
    name = "qwen3-coder"
    models = ("Qwen3.5", "Qwen3.6")
    call_name_prefix = "\n" + FUNCTION

    def detect(self, template: str) -> bool:
        # llama.cpp's own rule for choosing this parser.
        return (
            CALL in template and FUNCTION in template and PARAMETER in template
            and "'<tool_call><function=' ~ tool_call.name ~ '>'" not in template
        )

    def thinking(self, prompt: str) -> bool:
        """Whether the prompt opened the reasoning: its assistant turn opens `<think>` and
        never closes it, as the template does unless reasoning is turned off."""
        start = prompt.rfind(ASSISTANT)
        tail = prompt[start:] if start >= 0 else prompt
        return THINK in tail and THINK_END not in tail

    def parse(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool = True,
        thinking: bool | None = None,
        offered: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Mirrors the server's PEG parser: an optional reasoning, then content up to the first
        call, then the calls. A call to a tool not offered, or a parameter its schema does not
        name, is read with any type — the server's grammar never lets one through, and what
        the model wrote without it is still worth reading."""
        if thinking is None:
            thinking = _opened(text)
        source = (THINK + "\n" if thinking else "") + text
        i, reasoning = 0, ""
        if source.startswith(THINK):
            start = _skip_space(source, len(THINK))
            stops = [at for at in (source.find(THINK_END, start), source.find(CALL, start)) if at >= 0]
            if not stops:  # cut off while thinking
                reasoning, i = source[start:], len(source)
            else:
                end = min(stops)
                reasoning = source[start:end]
                i = end + len(THINK_END) if source.startswith(THINK_END, end) else end
        i = _skip_space(source, i)
        upto = source.find(CALL, i) if tools else -1
        content = source[i:upto] if upto >= 0 else source[i:]
        calls: list[dict[str, Any]] = []
        i = upto
        while i >= 0 and source.startswith(CALL, i) and (parallel or not calls):
            call, i = self._call(source, i, offered)
            calls.append({"id": f"call_{len(calls)}", "type": "function", "function": call})
            i = _skip_space(source, i)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning.strip(_SPACE):
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return message

    def _call(self, text: str, i: int, offered: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        opening = CALL + "\n" + FUNCTION
        if not text.startswith(opening, i):
            raise FormatError(f"a call at {i} does not open with {opening!r}: {text[i : i + 32]!r}")
        name_start = i + len(opening)
        name_end = text.find(">\n", name_start)
        if name_end < 0:
            raise FormatError(f"a call at {i} never closes its function's name")
        name = text[name_start:name_end]
        schema = _parameters(offered, name)
        properties, root = schema if schema is not None else ({}, {})
        i = name_end + 2
        members = []
        while text.startswith(PARAMETER, i):
            key_start = i + len(PARAMETER)
            key_end = text.find(">\n", key_start)
            if key_end < 0:
                raise FormatError(f"a parameter at {i} never closes its name")
            key = text[key_start:key_end].strip(_SPACE)
            i = key_end + 2
            admitted = _types(properties[key], root) if key in properties else _ALL
            found = None if admitted == {"string"} else _json_value(text, i, admitted)
            if found is None:
                if "string" not in admitted:
                    raise FormatError(f"{name}'s {key} is not the JSON its schema asks for, at {i}")
                close = text.find(ARG_CLOSE, i)
                if close < 0:
                    raise FormatError(f"{name}'s {key} is never closed")
                value, i = json.dumps(text[i:close], ensure_ascii=False), close
            else:
                value, i = found
            members.append(json.dumps(key, ensure_ascii=False) + ":" + value)
            i += len(ARG_CLOSE)
        closing = FUNCTION_END + "\n" + CALL_END
        if not text.startswith(closing, i):
            raise FormatError(f"a call is not closed at {i}: {text[i : i + 32]!r}")
        return {"name": name, "arguments": "{" + ",".join(members) + "}"}, i + len(closing)

    def parts(self, pieces: Sequence[str], *, thinking: bool | None = None) -> list[tuple[str, int, int]]:
        runs: list[tuple[str, int, int]] = []

        def mark(kind: str, position: int) -> None:
            if runs and runs[-1][0] == kind and runs[-1][2] == position:
                runs[-1] = (kind, runs[-1][1], position + 1)
            else:
                runs.append((kind, position, position + 1))

        if thinking is None:
            thinking = _opened("".join(pieces))
        state, skipping = ("reasoning" if thinking else "content"), False
        for position, piece in enumerate(pieces):
            if piece == THINK:
                mark("marker", position)
                state, skipping = "reasoning", True
            elif piece == THINK_END:
                mark("marker", position)
                state, skipping = "content", True
            elif piece == CALL:
                mark("marker", position)
                state, skipping = "tool_call", False
            elif piece == CALL_END:
                mark("marker", position)
                state, skipping = "content", True
            elif piece in ENDS:
                mark("marker", position)
            elif skipping and piece.strip(_SPACE) == "":
                mark("marker", position)  # the space the server skips after a marker
            else:
                skipping = False
                mark(state, position)
        return runs

    def render(self, message: dict[str, Any], *, thinking: bool) -> str:
        # The server keeps the reasoning up to `</think>`, its last newline included, and skips
        # the space after it, which the model writes as a blank line.
        text = ""
        reasoning = message.get("reasoning_content") or ""
        if thinking:
            text += reasoning + THINK_END + "\n\n"
        elif reasoning:
            text += THINK + "\n" + reasoning + THINK_END + "\n\n"
        # Content before a call keeps the blank line that ends it; calls are a line apart, which
        # the server skips.
        content = message.get("content")
        text += content if isinstance(content, str) else ""
        for k, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            if k:
                text += "\n"
            text += CALL + "\n" + FUNCTION + (function.get("name") or "") + ">\n"
            for key, value in _members(function.get("arguments")):
                text += PARAMETER + key + ">\n" + value + ARG_CLOSE
            text += FUNCTION_END + "\n" + CALL_END
        return text


def _members(arguments: Any) -> list[tuple[str, str]]:
    """A call's arguments as (name, value as the model wrote it): a JSON string's text, any
    other value as it stands in the arguments, which keep it as written."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    text = arguments if isinstance(arguments, str) and arguments.strip() else "{}"
    i = _skip_space(text, 0)
    if not text.startswith("{", i):
        raise FormatError("a call's arguments are not a JSON object")
    members: list[tuple[str, str]] = []
    i = _skip_space(text, i + 1)
    while not text.startswith("}", i):
        try:
            key, i = _JSON.raw_decode(text, i)
            i = _skip_space(text, i)
            if not text.startswith(":", i):
                raise ValueError("no ':' after a key")
            i = _skip_space(text, i + 1)
            value, end = _JSON.raw_decode(text, i)
        except ValueError as error:
            raise FormatError(f"a call's arguments are not JSON: {error}") from None
        members.append((str(key), value if isinstance(value, str) else text[i:end]))
        i = _skip_space(text, end)
        if text.startswith(",", i):
            i = _skip_space(text, i + 1)
        elif not text.startswith("}", i):
            raise FormatError(f"a call's arguments are not closed at {i}")
    return members


__all__ = ["Qwen3Coder"]
