"""Qwen3's and Qwen2.5's output format, as llama.cpp reads it (b11108).

No parser of llama.cpp's is named for these models. Their templates fall through to its
autoparser (`common/chat-diff-analyzer.cpp`, `common/chat-auto-parser-generator.cpp`), which
renders the template with made-up messages, finds the markers in what changes, and builds a
parser from them. Every Qwen3 template tried gives the same one — Qwen's own, unsloth's, LM
Studio's, bartowski's, ggml-org's, the 2507 updates' — and Qwen2.5's give it without the
reasoning, as does ggml-org's Qwen3-2507 Instruct template, which renders none. The raw text of
a Qwen3 answer:

    <think>
    …reasoning…
    </think>

    …content…
    <tool_call>
    {"name": "NAME", "arguments": {…}}
    </tool_call>

then an end token. A hybrid Qwen3 opens its reasoning itself, and with reasoning off the prompt
closes it (`<think>\\n\\n</think>\\n\\n`); a Qwen3-2507 Thinking template opens it in the prompt
(`<think>\\n`) — where one does not, the model opens it itself; an Instruct model never reasons;
Qwen2.5 has no reasoning to read.

The server reads the prompt's own opening of the turn and the answer as one text, and finds one
reasoning at most, at its start. So where the prompt opened the reasoning, the answer begins
inside it; where the prompt closed it, the answer is read from its first character, space
included, and a `<think>` in it is content. Where the prompt left it to the model — as Qwen2.5's
does too, having no reasoning to read:

- the space an answer begins with is skipped — but for Qwen2.5's content, kept where no tool is
  offered;
- a reasoning is read only where the answer begins with `<think>`, after one newline, up to
  `</think>`, with the newline before it; up to two newlines after it are skipped. One cut off
  before its close is still reasoning; one that is blank is dropped.

Where no prompt is at hand, an answer that closes a reasoning it never opened is read as one the
prompt opened, and any other as one the prompt left to the model.

Then, in every case:

- the content runs up to `<tool_call>` and a newline, its own last newline kept — to the end,
  where no tool is offered;
- each call is a JSON object, the name before the arguments, any space between its tokens,
  between a pair of markers — several to a pair, separated by commas, where calls may run in
  parallel; the pairs are space apart, and nothing may follow them. The arguments are kept as the
  model wrote them: the JSON text, spacing and all, `{}` where they are not an object.
- a call cut off before its close is an error here, where the server keeps what it read of it:
  as the other formats do, tuningfork reads only calls the model finished.

`<think>`, `</think>`, `<tool_call>` and `</tool_call>` are single tokens; a call's JSON is
ordinary text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from tuningfork.backends.formats.base import FormatError

THINK, THINK_END = "<think>", "</think>"
CALL, CALL_END = "<tool_call>", "</tool_call>"
#: A call opens with its marker and a newline: the server waits for both.
CALL_OPEN = CALL + "\n"
#: How a prompt that turns the reasoning off closes it.
CLOSED = THINK + "\n\n" + THINK_END + "\n\n"
#: What `thinking` says where the prompt left the reasoning to the model.
MODEL = "model"
ASSISTANT = "<|im_start|>assistant"
#: The pieces of the special tokens that end an answer (empty in llama.cpp's logprobs).
ENDS = ("", "<|im_end|>", "<|endoftext|>")

_SPACE = " \t\n\r\v\f"  # std::isspace's: what the server's parser skips
_BLANK = " \t\n\r"  # a reasoning of only these the server drops
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_NUMBER_GOES_ON = frozenset("0123456789.eE+-")
_ESCAPES = frozenset('"\\/bfnrt')
_HEX = frozenset("0123456789abcdefABCDEF")
#: What a template writes, as its source has it, for the server to read calls this way...
_CALL_TEMPLATE = ('<tool_call>\\n{"name": "', '", "arguments": ', '}\\n</tool_call>')
#: ... and what Qwen3's write to find the reasoning between `<think>\n` and `\n</think>\n\n`.
_REASONING_TEMPLATE = ("\\n<think>\\n' + reasoning_content", "+ '\\n</think>\\n\\n'")


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i] in _SPACE:
        i += 1
    return i


def _until(text: str, i: int, delimiter: str) -> int:
    """Where the server stops reading up to `delimiter` from `i`: at the delimiter, or where the
    text ends partway into it — a marker cut off, which it leaves out — or at the end."""
    at = text.find(delimiter, i)
    if at >= 0:
        return at
    for at in range(max(i, len(text) - len(delimiter) + 1), len(text)):
        if delimiter.startswith(text[at:]):
            return at
    return len(text)


def _opened(text: str) -> bool:
    """Whether an answer read without its prompt started inside the reasoning: it closes a
    reasoning it never opened. One cut off before it closed reads as content — where the server,
    which saw the prompt open it, reads it as reasoning."""
    end = text.find(THINK_END)
    return end >= 0 and THINK not in text[:end]


def _expect(text: str, i: int, literal: str) -> int:
    if text.startswith(literal, i):
        return i + len(literal)
    if i >= len(text) or literal.startswith(text[i:]):
        raise FormatError(f"a call is cut off at {i}, before {literal!r}")
    raise FormatError(f"a call has {text[i : i + 24]!r} at {i} where {literal!r} belongs")


# --- JSON, as the server's grammar reads it (common/peg-parser.cpp's json rules) --------------
# Any character in a string, std::isspace's space between tokens, no NaN or Infinity, and a
# number never followed by what could go on with it.


def _json(text: str, i: int) -> int:
    """The end of the JSON value at `i`."""
    if text.startswith("{", i):
        return _container(text, i, "}", member=True)
    if text.startswith("[", i):
        return _container(text, i, "]", member=False)
    if text.startswith('"', i):
        return _string(text, i)
    for word in ("true", "false", "null"):
        if text.startswith(word, i):
            return i + len(word)
    found = _NUMBER.match(text, i)
    if found is None:
        if i >= len(text) or any(word.startswith(text[i:]) for word in ("true", "false", "null", "-")):
            raise FormatError(f"a value is cut off at {i}")
        raise FormatError(f"no JSON value at {i}: {text[i : i + 24]!r}")
    if found.end() >= len(text):
        raise FormatError(f"a number is cut off at {i}")
    if text[found.end()] in _NUMBER_GOES_ON:
        raise FormatError(f"a number at {i} is not JSON: {text[i : found.end() + 1]!r}")
    return found.end()


def _string(text: str, i: int) -> int:
    j = i + 1
    while j < len(text):
        c = text[j]
        if c == '"':
            return j + 1
        if c == "\\":
            escape = text[j + 1 : j + 2]
            if escape in _ESCAPES:
                j += 2
                continue
            if escape == "u":
                digits = text[j + 2 : j + 6]
                if all(d in _HEX for d in digits) and len(digits) == 4:
                    j += 6
                    continue
                if all(d in _HEX for d in digits) and j + 6 > len(text):
                    break  # cut off
            elif not escape:
                break  # cut off
            raise FormatError(f"a string has a bad escape at {j}: {text[j : j + 6]!r}")
        j += 1
    raise FormatError(f"a string opened at {i} is cut off")


def _container(text: str, i: int, close: str, *, member: bool) -> int:
    i = _skip_space(text, i + 1)
    if text.startswith(close, i):
        return i + 1
    while True:
        if member:
            if not text.startswith('"', i):
                _expect(text, i, '"')
            i = _expect(text, _skip_space(text, _string(text, i)), ":")
            i = _skip_space(text, i)
        i = _skip_space(text, _json(text, i))
        if text.startswith(",", i):
            i = _skip_space(text, i + 1)
            continue
        return _expect(text, i, close)


class QwenJson:
    """One reading of a call's JSON in `<tool_call>` markers: with Qwen3's reasoning, or — for
    templates that render none — without."""

    call_name_prefix = '\n{"name": "'

    def __init__(self, name: str, models: tuple[str, ...], *, reasoning: bool) -> None:
        self.name = name
        self.models = models
        self.reasoning = reasoning

    def detect(self, template: str) -> bool:
        # The template strings the server's analysis finds this reading's markers in; another
        # template that keeps them is read the same way.
        if "<|im_start|>" not in template or "<function=" in template:
            return False
        if not all(part in template for part in _CALL_TEMPLATE):
            return False
        if self.reasoning:
            return all(part in template for part in _REASONING_TEMPLATE)
        return "think>" not in template

    def thinking(self, prompt: str) -> bool | str:
        """What the prompt's last assistant turn did with the reasoning: opened it (True), as a
        Qwen3-2507 Thinking template does; closed it (False), as Qwen3's does with reasoning off;
        or neither (`MODEL`), leaving it to the model — as Qwen2.5's does, having none."""
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
        """Mirrors the parser the server builds from the template; see the module. A call to a
        tool not offered is read with its name — the server's grammar never lets one through,
        and what the model wrote without it is still worth reading."""
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
        tool_call — and the error that stopped the reading, if one did. The prompt's own opening
        of the turn is read before the answer, as the server reads it."""
        if thinking is None:  # no prompt at hand: read from the answer
            thinking = True if _opened(text) else MODEL
        if not self.reasoning or thinking == MODEL:
            opening = ""
        else:
            opening = THINK + "\n" if thinking else CLOSED
        source = opening + text
        kinds = ["marker"] * len(source)

        def mark(kind: str, start: int, end: int) -> None:
            kinds[start:end] = [kind] * (end - start)

        def done(error: FormatError | None = None) -> tuple[dict[str, Any], list[str], FormatError | None]:
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning.strip(_BLANK):
                message["reasoning_content"] = reasoning
            else:
                mark("marker", *reasoning_span)
            if calls:
                message["tool_calls"] = [
                    {"id": f"call_{k}", "type": "function", "function": call} for k, call in enumerate(calls)
                ]
            return message, kinds[len(opening):], error

        content, reasoning, reasoning_span = "", "", (0, 0)
        calls: list[dict[str, Any]] = []
        # Qwen2.5's content, without tools, is the whole answer: nothing skipped.
        i = _skip_space(source, 0) if self.reasoning or tools else 0
        if self.reasoning and source.startswith(THINK, i):
            start = i + len(THINK)
            if source.startswith("\n", start):
                start += 1
            end = _until(source, start, THINK_END)
            reasoning, reasoning_span = source[start:end], (start, end)
            mark("reasoning", start, end)
            if not source.startswith(THINK_END, end):
                return done()  # cut off before its close: the rest is reasoning
            i = end + len(THINK_END)
            for _ in range(2):
                if source.startswith("\n", i):
                    i += 1
        elif self.reasoning and i < len(source) and THINK.startswith(source[i:]):
            return done()  # the opening itself cut off
        if not tools:
            content = source[i:]
            mark("content", i, len(source))
            return done()
        end = _until(source, i, CALL_OPEN)
        content = source[i:end]
        mark("content", i, end)
        i = end
        try:
            while source.startswith(CALL_OPEN, i):
                i = _skip_space(source, self._block(source, i, parallel, offered, calls, mark))
        except FormatError as error:
            mark("tool_call", i + len(CALL), len(source))
            return done(error)
        if i < len(source) and not CALL_OPEN.startswith(source[i:]):  # a marker cut off is waited for
            return done(FormatError(f"text after the calls at {i}: {source[i : i + 32]!r}"))
        return done()

    def _block(
        self, text: str, i: int, parallel: bool, offered: Sequence[dict[str, Any]],
        calls: list[dict[str, Any]], mark: Any,
    ) -> int:
        """One `<tool_call>`'s calls; where it ends."""
        start = i + len(CALL)  # the newline after the marker starts the call, as its name's prefix
        j = _skip_space(text, i + len(CALL_OPEN))
        while True:
            call, j = self._call(text, j, offered)
            calls.append(call)
            k = _skip_space(text, j)
            if not (parallel and text.startswith(",", k)):
                break
            j = _skip_space(text, k + 1)
        j = _skip_space(text, j)
        mark("tool_call", start, j)
        return _expect(text, j, CALL_END)

    def _call(self, text: str, i: int, offered: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        i = _skip_space(text, _expect(text, i, "{"))
        i = _skip_space(text, _expect(text, i, '"name"'))
        i = _skip_space(text, _expect(text, i, ":"))
        names = [
            function.get("name") for tool in offered if isinstance(tool, dict)
            for function in [tool.get("function")] if isinstance(function, dict) and isinstance(function.get("name"), str)
        ]
        name = next((found for found in names if text.startswith(f'"{found}"', i)), None)
        if name is not None:
            i += len(name) + 2
        else:  # a tool not offered: its name as written
            if not text.startswith('"', i):
                _expect(text, i, '"')
            end = _string(text, i)
            name, i = text[i + 1 : end - 1], end
        i = _skip_space(text, _expect(text, _skip_space(text, i), ","))
        i = _skip_space(text, _expect(text, i, '"arguments"'))
        i = _skip_space(text, _expect(text, i, ":"))
        end = _json(text, i)
        arguments = text[i:end]
        i = _expect(text, _skip_space(text, end), "}")
        return {"name": name, "arguments": arguments if arguments.startswith("{") else "{}"}, i

    def parts(self, pieces: Sequence[str], *, thinking: bool | str | None = None) -> list[tuple[str, int, int]]:
        """Each token's part, from what the server's reading makes of each character: a token
        takes the part of its first character that is not a marker's, and is a marker where all
        are. Read with tools offered, any tool's name, and calls in parallel: a stream does not
        say what was offered."""
        ends = len(pieces)
        while ends and pieces[ends - 1] in ENDS[1:]:
            ends -= 1
        text = "".join(pieces[:ends])
        _, kinds, _ = self._read(text, tools=True, parallel=True, thinking=thinking, offered=())
        runs: list[tuple[str, int, int]] = []
        at = 0
        for position, piece in enumerate(pieces):
            own = kinds[at : at + len(piece)] if position < ends else []
            at += len(piece) if position < ends else 0
            kind = next((each for each in own if each != "marker"), "marker")
            if runs and runs[-1][0] == kind and runs[-1][2] == position:
                runs[-1] = (kind, runs[-1][1], position + 1)
            else:
                runs.append((kind, position, position + 1))
        return runs

    def render(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> str:
        return self.renderings(message, thinking=thinking, offered=offered)[0]

    def renderings(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> list[str]:
        """The raw texts the server reads as this message, the template's own first. What it
        skips leaves no trace in the message: a Qwen3 left to reason or not may have written an
        empty reasoning, which only the answer's number of tokens tells; and calls are a line
        apart as the template writes them, or a blank line, as Qwen3-2507 Thinking writes them —
        one token either way, which only the model's own likelihood tells."""
        reasoning = message.get("reasoning_content") or ""
        # The server keeps the reasoning up to `</think>`, its last newline included, and skips
        # the blank line the model writes after it.
        if not self.reasoning or thinking is False:
            heads = [""]  # no reasoning: none in the template, or the prompt closed it
        elif thinking is True:
            heads = [reasoning + THINK_END + "\n\n"]
        elif reasoning:  # left to the model, or not known
            heads = [THINK + "\n" + reasoning + THINK_END + "\n\n"]
        else:
            heads = ["", THINK + "\n\n" + THINK_END + "\n\n"]
        apart = ("\n", "\n\n") if len(message.get("tool_calls") or []) > 1 else ("\n",)
        return [head + self._body(message, between) for head in heads for between in apart]

    def _body(self, message: dict[str, Any], between: str) -> str:
        # Content before a call keeps the newline that ends it. The arguments are the text the
        # model wrote.
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        for k, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
            elif not arguments.strip():
                arguments = "{}"
            text += (between if k else "") + CALL_OPEN
            text += '{"name": "' + (function.get("name") or "") + '", "arguments": ' + arguments + "}\n" + CALL_END
        return text


__all__ = ["QwenJson"]
