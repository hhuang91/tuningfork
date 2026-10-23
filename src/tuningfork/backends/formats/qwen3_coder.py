"""Qwen3.5's and Qwen3-Coder's output format, as llama.cpp reads it (`common/parsers/qwen3-coder.cpp`,
b11108).

llama.cpp names the parser after Qwen3-Coder, which wrote the format first, and reads Qwen3.5,
Qwen3.6, Nemotron Nano 3 and StepFun-3.5-Flash with it too. It reads a template that reasons —
Qwen3.5's — with its reasoning, and one that has none — Qwen3-Coder's — without it, where the
first call may begin without `<tool_call>`, as that model sometimes writes one. The raw text of a
Qwen3.5 answer, after a prompt that opened the reasoning — the template's default, `<think>\\n`
closing the prompt:

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
(`<think>\\n\\n</think>\\n\\n`), and the answer is the content and the calls. A Qwen3-Coder answer
is the content and the calls. There the server ends the content at `<tool_call>`, or at
`<function=` and the name of a tool it offered, and reads the first call with its marker or
without it; every later call has one.

The parser reads the prompt's opening of the turn and the answer as one text, and finds one
reasoning at most, at its start. Qwen3.5's and Qwen3.6's prompts always open or close one, so a
`<think>` the model writes is text. Where another template leaves the reasoning to the model, a
`<think>` the answer begins with is the reasoning. Where no prompt is at hand, an answer that
closes a reasoning it never opened is read as one the prompt opened, and any other as one it
closed, as these models' prompts do.

A value is typed by the tool's schema, as the server types it: raw text where the parameter can
only be a string, any JSON where it can be no string, and otherwise JSON where it reads as a
number or one of the schema's other types, else text. The server builds the arguments from the
values as written — a string escaped into JSON, anything else kept as the model wrote it.

Where the text ends partway into a marker or a function's opening, the server leaves that part
out of what it reads. It also leaves out what follows the calls, and a call it cannot read, with
all after it. A call cut off after its opening, or one that cannot be read, is an error here:
as the other formats do, tuningfork reads only calls the model finished, and says where one
went wrong.

`<think>`, `</think>`, `<tool_call>` and `</tool_call>` are single tokens; the rest of a call is
ordinary text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from tuningfork.backends.formats.base import FormatError, token_parts

THINK, THINK_END = "<think>", "</think>"
CALL, CALL_END = "<tool_call>", "</tool_call>"
#: A call's marker is a line of its own: the server reads the newline with it.
CALL_OPEN = CALL + "\n"
FUNCTION, FUNCTION_END = "<function=", "</function>"
#: The variant with no newline between its tags, which llama.cpp reads with another parser.
NO_NEWLINE = "'<tool_call><function=' ~ tool_call.name ~ '>'"
PARAMETER, ARG_CLOSE = "<parameter=", "\n</parameter>\n"
ASSISTANT = "<|im_start|>assistant"
#: What `thinking` says where the prompt left the reasoning to the model.
MODEL = "model"
#: The pieces of the special tokens that end an answer (empty in llama.cpp's logprobs).
ENDS = ("", "<|im_end|>", "<|endoftext|>")

_SPACE = " \t\n\r\v\f"  # std::isspace's
_ALL = frozenset({"string", "number", "integer", "boolean", "null", "array", "object"})
#: The formats llama.cpp reads as a string's (`get_format`); any other format says nothing.
_STRING_FORMATS = ("date", "time", "date-time", "uuid", "uuid1", "uuid2", "uuid3", "uuid4", "uuid5")


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


#: JSON as llama.cpp's parser reads it: a string may hold any character, and NaN and Infinity
#: are no numbers.
_JSON = json.JSONDecoder(strict=False, parse_constant=_no_constant)


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i] in _SPACE:
        i += 1
    return i


def _opened(text: str) -> bool:
    """Whether an answer read without its prompt started inside the reasoning: it closes a
    reasoning it never opened. An answer cut off before it closed reads as content."""
    end = text.find(THINK_END)
    return end >= 0 and THINK not in text[:end]


#: A function's opening, whatever its name: where the content ends when no tool is known.
_OPENER = re.compile(r"<function=[^<>\n]+>")


def _until(text: str, i: int, stops: Sequence[str], *, any_function: bool = False) -> int:
    """Where the server's reading from `i` stops: at the first of `stops` — or, `any_function`,
    of any function's opening — or where the text ends partway into one, which it leaves out, or
    at the end."""
    found = [at for at in (text.find(stop, i) for stop in stops) if at >= 0]
    opener = _OPENER.search(text, i) if any_function else None
    if opener is not None:
        found.append(opener.start())
    if found:
        return min(found)
    # each stop holds one "<", its first character: one cut off starts at the text's last
    at = text.rfind("<", i)
    tail = text[at:] if at >= 0 else ""
    if tail and (any(stop.startswith(tail) for stop in stops) or (any_function and _opening(tail))):
        return at
    return len(text)


def _opening(tail: str) -> bool:
    """Whether the end of a text is a function's opening cut off before its name closed."""
    return FUNCTION.startswith(tail) or (tail.startswith(FUNCTION) and ">" not in tail and "\n" not in tail)


def _names(offered: Sequence[dict[str, Any]]) -> list[str]:
    """The names of the tools offered."""
    return [
        function["name"] for tool in offered if isinstance(tool, dict)
        for function in [tool.get("function")] if isinstance(function, dict) and isinstance(function.get("name"), str)
    ]


def _expect(text: str, i: int, literal: str) -> int:
    if not text.startswith(literal, i):
        raise FormatError(f"a call has {text[i : i + 24]!r} at {i} where {literal!r} belongs")
    return i + len(literal)


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
        if any(key in schema for key in ("pattern", "minLength", "maxLength")) or schema.get("format") in _STRING_FORMATS:
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
    """The JSON value at `i`, followed by the argument's close: its text as written, and where
    the close starts. Where the schema admits a string, the server tries only the other types
    first — any number where it admits a number or an integer — and reads the rest as text;
    where it admits none, its parser takes any JSON (the grammar keeps to the schema)."""
    try:
        value, end = _JSON.raw_decode(text, i)
    except ValueError:
        return None
    if not text.startswith(ARG_CLOSE, end):
        return None
    if "string" in admitted:
        kind = _json_type(value)
        number = kind in ("integer", "number") and bool(admitted & {"integer", "number"})
        if kind == "string" or not (number or kind in admitted):
            return None
    return text[i:end], end


class Qwen3Coder:
    """One reading of these calls: with Qwen3.5's reasoning, or — for a template that has none,
    Qwen3-Coder's — without it, the first call's marker optional."""

    #: A call's own text starts at its function: `<tool_call>` and its newline are its marker.
    call_name_prefix = FUNCTION

    def __init__(self, name: str, models: tuple[str, ...], *, reasoning: bool) -> None:
        self.name = name
        self.models = models
        self.reasoning = reasoning

    def detect(self, template: str) -> bool:
        # llama.cpp's own rule for choosing this parser; a `<think>` in the template is what
        # makes it read a reasoning.
        return (
            CALL in template and FUNCTION in template and PARAMETER in template
            and NO_NEWLINE not in template
            and (THINK in template) == self.reasoning
        )

    def thinking(self, prompt: str) -> bool | str:
        """What the prompt's last assistant turn did with the reasoning: opened it (True), as the
        template does unless reasoning is turned off; closed it (False), as it does then; or
        neither (`MODEL`), leaving it to the model — as Qwen3-Coder's does, having none."""
        if not self.reasoning:
            return MODEL
        start = prompt.rfind(ASSISTANT)
        tail = prompt[start:] if start >= 0 else prompt
        if THINK_END in tail:
            return False
        return True if THINK in tail else MODEL

    def parse(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool = True,
        thinking: bool | str | None = None,
        offered: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Mirrors the server's PEG parser: the reasoning, then content up to the first call,
        then the calls; see the module for which reasoning. A call to a tool not offered, or a
        parameter its schema does not name, is read with any type — the server's grammar never
        lets one through, and what the model wrote without it is still worth reading. Only a
        first call without its marker must name a tool offered — any, where none is known — as
        the server's content ends only there."""
        message, _, error = self._read(text, tools=tools, parallel=parallel, thinking=thinking, offered=offered)
        if error is not None:
            raise error
        return message

    def _read(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool,
        thinking: bool | str | None,
        offered: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str], FormatError | None]:
        """The message, what each character of `text` is — marker, reasoning, content or
        tool_call — and the error that stopped the reading, if one did. The prompt's opening of
        the reasoning is read before the answer, as the server reads it."""
        if thinking is None:  # no prompt at hand: read from the answer — opened, or else closed,
            thinking = _opened(text) if self.reasoning else MODEL  # as these models leave it
        opening = THINK + "\n" if self.reasoning and thinking is True else ""
        source = opening + text
        kinds = ["marker"] * len(source)

        def mark(kind: str, start: int, end: int) -> None:
            kinds[start:end] = [kind] * (end - start)

        def done(error: FormatError | None = None) -> tuple[dict[str, Any], list[str], FormatError | None]:
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning.strip(_SPACE):
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = [
                    {"id": f"call_{k}", "type": "function", "function": call} for k, call in enumerate(calls)
                ]
            return message, kinds[len(opening):], error

        content, reasoning = "", ""
        calls: list[dict[str, Any]] = []
        i = 0
        if self.reasoning and (thinking is True or (thinking == MODEL and source.startswith(THINK))):
            start = _skip_space(source, len(THINK))
            end = _until(source, start, (THINK_END, CALL))
            reasoning = source[start:end]
            mark("reasoning", start, end)
            if source.startswith(THINK_END, end):
                i = end + len(THINK_END)
            elif source.startswith(CALL, end):
                i = end
            else:
                return done()  # cut off while thinking
        i = _skip_space(source, i)
        if not tools:
            content = source[i:]
            mark("content", i, len(source))
            return done()
        # Qwen3-Coder's content also ends where its first call opens without the marker: at a
        # tool offered, or at any, where none is known.
        names = [] if self.reasoning else _names(offered)
        openings = tuple(FUNCTION + name + ">" for name in names)

        def opens(at: int) -> bool:
            if self.reasoning:
                return False
            return source.startswith(openings, at) if names else _OPENER.match(source, at) is not None

        end = _until(source, i, (CALL, *openings), any_function=not self.reasoning and not names)
        content = source[i:end]
        mark("content", i, end)
        i = end
        try:
            while i < len(source) and (parallel or not calls):
                if source.startswith(CALL, i):
                    if source[i:] == CALL:
                        return done()  # the marker's line cut off: the server waits for the rest of it
                    i += len(CALL)
                    i = _expect(source, i, "\n")
                elif calls or not opens(i):
                    break  # a marker or an opening cut off, or text after the calls
                call, end = self._call(source, i, offered)
                mark("tool_call", i, end)
                i = _skip_space(source, _expect(source, end, CALL_END))
                calls.append(call)
        except FormatError as error:
            mark("tool_call", i, len(source))
            return done(error)
        if calls:  # text after the calls, which the server leaves out, is the model's text
            mark("content", i, len(source))
        return done()

    def _call(self, text: str, i: int, offered: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        """The call whose function opens at `i`, and where it ends: after the line that closes
        the function."""
        if not text.startswith(FUNCTION, i):
            raise FormatError(f"a call at {i} does not open with {FUNCTION!r}: {text[i : i + 32]!r}")
        name_start = i + len(FUNCTION)
        name_end = text.find(">", name_start)
        if name_end < 0 or not text.startswith(">\n", name_end):
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
                    raise FormatError(f"{name}'s {key} is no JSON, which its schema asks for, at {i}")
                close = text.find(ARG_CLOSE, i)
                if close < 0:
                    raise FormatError(f"{name}'s {key} is never closed")
                value, i = json.dumps(text[i:close], ensure_ascii=False), close
            else:
                value, i = found
            members.append(json.dumps(key, ensure_ascii=False) + ":" + value)
            i += len(ARG_CLOSE)
        closing = FUNCTION_END + "\n"
        if not text.startswith(closing, i):
            raise FormatError(f"a call is not closed at {i}: {text[i : i + 32]!r}")
        return {"name": name, "arguments": "{" + ",".join(members) + "}"}, i + len(closing)

    def parts(self, pieces: Sequence[str], *, thinking: bool | str | None = None) -> list[tuple[str, int, int]]:
        """Each token's part, from what the server's reading makes of each character. Read with
        tools offered, any tool's name, and calls in parallel: a stream does not say what was
        offered."""
        ends = len(pieces)
        while ends and pieces[ends - 1] in ENDS[1:]:
            ends -= 1
        _, kinds, _ = self._read("".join(pieces[:ends]), tools=True, parallel=True, thinking=thinking, offered=())
        return token_parts(pieces, kinds)

    def render(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> str:
        return self.renderings(message, thinking=thinking, offered=offered)[0]

    def renderings(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> list[str]:
        """The raw texts the server reads as this message, the template's own first. Qwen3-Coder
        may have written its first call without the marker, which only the answer's number of
        tokens tells."""
        # The server keeps the reasoning up to `</think>`, its last newline included, and skips
        # the space after it, which the model writes as a blank line.
        head = ""
        reasoning = message.get("reasoning_content") or ""
        if self.reasoning and thinking is True:
            head += reasoning + THINK_END + "\n\n"
        elif self.reasoning and reasoning:
            head += THINK + "\n" + reasoning + THINK_END + "\n\n"
        # Content before a call keeps the blank line that ends it; calls are a line apart, which
        # the server skips.
        content = message.get("content")
        head += content if isinstance(content, str) else ""
        calls = [self._written(call, offered) + CALL_END for call in message.get("tool_calls") or []]
        texts = [head + "\n".join(CALL_OPEN + call for call in calls)]
        if calls and not self.reasoning:
            texts.append(head + "\n".join([calls[0], *(CALL_OPEN + call for call in calls[1:])]))
        return texts

    def _written(self, call: dict[str, Any], offered: Sequence[dict[str, Any]]) -> str:
        """A call as the model writes it, from its function's opening to the line closing it."""
        function = call.get("function") or {}
        name = function.get("name") or ""
        properties, root = _parameters(offered, name) or ({}, {})
        text = FUNCTION + name + ">\n"
        for key, value, written in _members(function.get("arguments")):
            # a string the model wrote as text, unless its parameter takes none: then it was JSON
            admitted = _types(properties[key], root) if key in properties else _ALL
            raw = value if isinstance(value, str) and "string" in admitted else written
            text += PARAMETER + key + ">\n" + raw + ARG_CLOSE
        return text + FUNCTION_END + "\n"


def _members(arguments: Any) -> list[tuple[str, Any, str]]:
    """A call's arguments as (name, value, the value's JSON as it stands in the arguments, which
    on chat completions keep it as the model wrote it). A Messages step's arguments were written
    again from the parsed input (`wire.anthropic`): escapes added, the model's spacing lost. Such
    a value is rebuilt approximately, which scoring says where the token count changes."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    text = arguments if isinstance(arguments, str) and arguments.strip() else "{}"
    i = _skip_space(text, 0)
    if not text.startswith("{", i):
        raise FormatError("a call's arguments are not a JSON object")
    members: list[tuple[str, Any, str]] = []
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
        members.append((str(key), value, text[i:end]))
        i = _skip_space(text, end)
        if text.startswith(",", i):
            i = _skip_space(text, i + 1)
        elif not text.startswith("}", i):
            raise FormatError(f"a call's arguments are not closed at {i}")
    return members


__all__ = ["Qwen3Coder"]
