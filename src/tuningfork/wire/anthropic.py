"""Anthropic Messages (`POST /v1/messages`), as Claude Code speaks it.

Verified against Claude Code 2.1.281 talking to llama.cpp's native `/v1/messages`
(b11108): every model call streams; a response is `thinking`, `text` and `tool_use`
content blocks; tool results come back as `tool_result` blocks inside a user message; and
Claude Code also sends `role: "system"` messages mid-conversation (a beta), which are kept
as they are.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from tuningfork.proxy.sse import (
    CUT_OFF,
    UPSTREAM_ERROR,
    SSEEvent,
    SSEParser,
    event_index,
    upstream_error,
)
from tuningfork.wire.base import NOT_RUN, Assembler, call_id

#: Anthropic stop reasons in the common shape's terms. Anything else passes through.
STOP_TO_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}
#: The way back, for an edit that changed `finish_reason`. Anything else passes through,
#: so an Anthropic stop reason (`refusal`, `pause_turn`, ...) can be written as it is.
FINISH_TO_STOP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class MessageAssembler:
    """Rebuild a complete `message` from its stream of events.

    A server may leave null what this expects filled in — `usage`, a block's opening text —
    and none of that may stop the rest of the message being read.
    """

    def __init__(self) -> None:
        self.chunk_count = 0
        self.message: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self._partial_json: dict[int, str] = {}
        #: The block opened last. Blocks stream one at a time, so a delta without a usable
        #: index belongs to it.
        self._open = 0
        #: The payload of an `error` event: the server gave up mid-stream.
        self.error: Any = None
        #: Whether `message_stop` arrived: the server's word that the message is whole.
        self.saw_stop = False

    def add_event(self, event: SSEEvent) -> None:
        data = event.json()
        if data is None:
            return
        self.chunk_count += 1
        kind = data.get("type") or event.event
        if kind == "message_start" and isinstance(data.get("message"), dict):
            self.message = copy.deepcopy(data["message"])
        elif kind == "content_block_start" and isinstance(data.get("content_block"), dict):
            self._open = event_index(data.get("index"), len(self.blocks))
            self.blocks[self._open] = copy.deepcopy(data["content_block"])
        elif kind == "content_block_delta" and isinstance(data.get("delta"), dict):
            self._apply(event_index(data.get("index"), self._open), data["delta"])
        elif kind == "message_delta":
            if isinstance(data.get("delta"), dict):
                self.message.update(data["delta"])
            if isinstance(data.get("usage"), dict):
                usage = self.message.get("usage")  # null if message_start said so
                self.message["usage"] = {**(usage if isinstance(usage, dict) else {}), **data["usage"]}
        elif kind == "message_stop":
            self.saw_stop = True
        elif kind == "error":
            self.error = data.get("error") or data

    def _apply(self, index: int, delta: dict[str, Any]) -> None:
        block = self.blocks.setdefault(index, {})
        kind = delta.get("type")
        if kind == "input_json_delta":
            fragment = delta.get("partial_json")
            if not isinstance(fragment, str):
                # Sent already parsed, as an object: the input stays JSON text until the end.
                fragment = "" if fragment is None else json.dumps(fragment)
            self._partial_json[index] = self._partial_json.get(index, "") + fragment
            return
        # text_delta → text, thinking_delta → thinking, signature_delta → signature, and
        # any future string delta by its own name: accumulated generically — onto nothing,
        # if the block opened with its text null rather than empty.
        for key, value in delta.items():
            if key != "type" and isinstance(value, str):
                block[key] = _string(block.get(key)) + value

    def failure(self) -> str | None:
        """Why the stream was not a whole message, or None if it was. Read once it has ended.

        An `error` event; or a stream cut off before the model finished — one that ended with
        neither `message_stop` nor a stop reason. As with OpenAI's `[DONE]` and finish reason,
        either one is enough to count as finished.
        """
        if self.error is not None:
            return upstream_error(self.error)
        if not self.saw_stop and not self.message.get("stop_reason"):
            return CUT_OFF
        return None

    def result(self) -> dict[str, Any]:
        message = dict(self.message)
        content = []
        for index in sorted(self.blocks):
            block = dict(self.blocks[index])
            if index in self._partial_json:
                raw = self._partial_json[index]
                try:
                    block["input"] = json.loads(raw) if raw else {}
                except (ValueError, RecursionError):
                    block["input"] = {"_unparsed": raw}
            content.append(block)
        message["content"] = content
        if self.error is not None:
            message["error"] = self.error  # kept beside what arrived before it, visibly
        return message


def _without_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_without_cache_control(v) for v in value]
    return value


def _string(value: Any) -> str:
    """A text field as text: a server, or an edit, may leave one null."""
    return value if isinstance(value, str) else ""


def _blocks(content: Any) -> list[dict[str, Any]]:
    """A message's content blocks, passing over anything that is not one."""
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _count(value: Any) -> int:
    """A token count, or 0 for one that is missing or not a number."""
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _finish_reason(stop: Any) -> Any:
    return STOP_TO_FINISH.get(stop, stop) if isinstance(stop, str) else stop


def _is_error(native: dict[str, Any]) -> bool:
    """Whether a body is an error rather than a message: Anthropic's `{"type": "error"}`,
    the `{"error": ...}` a gateway in front of the server may answer with, or a stream
    that failed before it produced anything."""
    if native.get("type") == "error":
        return True
    return native.get("error") is not None and not _blocks(native.get("content"))


def _tool_input(arguments: Any) -> dict[str, Any]:
    """An edited call's arguments as a tool_use `input`, which has to be an object.

    Normally JSON text; an object is taken as already parsed, since an edit written by hand
    may give one. Whatever is not an object is kept, visibly, rather than dropped.
    """
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except (ValueError, RecursionError):
            parsed = None
    elif arguments is None:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {"_unparsed": arguments}


def _stop_reason(finish: Any, calls: bool, original: dict[str, Any]) -> str:
    """The stop reason an edited response is delivered with, which the harness acts on.

    A `finish_reason` the edit left alone keeps the model's own stop reason — the common
    shape says "stop" for both end_turn and stop_sequence — unless the edit gave the
    message its first tool call, since a harness runs calls only in a tool_use turn. One
    the edit changed is what the person asked for, in Anthropic's terms. Either way, a turn
    left with no tool call in it is not tool_use.
    """
    before = original.get("stop_reason")
    if finish == _finish_reason(before):
        had_calls = any(b.get("type") == "tool_use" for b in _blocks(original.get("content")))
        stop = "tool_use" if calls and not had_calls else before
    else:
        stop = FINISH_TO_STOP.get(finish, finish) if isinstance(finish, str) else None
    if stop == "tool_use" and not calls:
        stop = "end_turn"
    if not isinstance(stop, str) or not stop:
        stop = "tool_use" if calls else "end_turn"
    return stop


def _text(content: Any) -> str:
    """Plain text of a content value that may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            _string(block.get("text")) for block in content if isinstance(block, dict)
        )
    return ""


def _assistant_view(blocks: list[Any]) -> dict[str, Any]:
    """Assistant content blocks as one OpenAI-style assistant message."""
    reasoning = [_string(b.get("thinking")) for b in blocks if b.get("type") == "thinking"]
    text = [_string(b.get("text")) for b in blocks if b.get("type") == "text"]
    calls = [
        {
            "id": b.get("id"),
            "type": "function",
            "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})},
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(text)}
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    if any(b.get("type") == "redacted_thinking" for b in blocks):
        message["redacted_thinking"] = True
    if calls:
        message["tool_calls"] = calls
    return message


def _parts(blocks: list[Any]) -> str | list[dict[str, Any]]:
    """Non-tool blocks of a message: a plain string when it is just one piece of text."""
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"type": "text", "text": _string(block.get("text"))})
        elif block.get("type") == "image":
            parts.append({"type": "image_url"})
        else:
            parts.append(block)
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


class AnthropicMessages:
    name = "anthropic-messages"
    endpoint = "/v1/messages"
    output_limits = ("max_tokens",)

    def matches(self, method: str, path: str) -> bool:
        return method == "POST" and path.endswith("/v1/messages")

    def assembler(self) -> MessageAssembler:
        return MessageAssembler()

    def parse(self, raw: bytes, event_stream: bool) -> dict[str, Any] | None:
        if event_stream:
            assembler = MessageAssembler()
            for event in SSEParser().feed(raw):
                assembler.add_event(event)
            return assembler.result() if assembler.chunk_count else None
        try:
            decoded = json.loads(raw)
        except (ValueError, RecursionError):  # not JSON, or not even UTF-8
            return None
        return decoded if isinstance(decoded, dict) else None

    def failure(
        self, status: int, response: dict[str, Any] | None, assembler: Assembler | None = None
    ) -> str | None:
        if status >= 400:
            return UPSTREAM_ERROR
        if assembler is not None:
            return assembler.failure()
        if isinstance(response, dict) and _is_error(response):
            return upstream_error(response.get("error"))
        return None

    # --- responses ------------------------------------------------------------------

    def canonical_response(self, native: dict[str, Any]) -> dict[str, Any]:
        if _is_error(native):
            # Not an answer, so no choices — as an OpenAI error body has none — and no
            # outcome to count as behaviour. The error itself stays, to be read.
            return {
                "id": native.get("id"),
                "object": "chat.completion",
                "model": native.get("model"),
                "choices": [],
                "error": native.get("error") or native,
            }
        canonical: dict[str, Any] = {
            "id": native.get("id"),
            "object": "chat.completion",
            "model": native.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_view(_blocks(native.get("content"))),
                    "finish_reason": _finish_reason(native.get("stop_reason")),
                }
            ],
        }
        usage = native.get("usage")
        if isinstance(usage, dict):
            cached = _count(usage.get("cache_read_input_tokens"))
            prompt = (
                _count(usage.get("input_tokens"))
                + cached
                + _count(usage.get("cache_creation_input_tokens"))
            )
            completion = _count(usage.get("output_tokens"))
            canonical["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
        if native.get("error"):
            canonical["error"] = native["error"]
        return canonical

    def native_response(
        self, canonical: dict[str, Any], original: dict[str, Any] | None
    ) -> dict[str, Any]:
        original = original or {}
        choices = canonical.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice = choice if isinstance(choice, dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        kept = _blocks(original.get("content"))

        blocks: list[dict[str, Any]] = []
        reasoning = _string(message.get("reasoning_content"))
        original_thinking = [b for b in kept if b.get("type") in ("thinking", "redacted_thinking")]
        if reasoning == _assistant_view(kept).get("reasoning_content", ""):
            blocks += original_thinking  # untouched: keep the blocks, signatures and all
        elif reasoning:
            blocks.append({"type": "thinking", "thinking": reasoning})
        if text := _text(message.get("content")):
            blocks.append({"type": "text", "text": text})
        calls = message.get("tool_calls")
        calls = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        for call in calls:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            blocks.append(
                {
                    "type": "tool_use",
                    # A call a person added has no id yet: the harness answers calls by id.
                    "id": call.get("id") or call_id("toolu_"),
                    "name": function.get("name"),
                    "input": _tool_input(function.get("arguments")),
                }
            )

        native = {
            key: original[key] for key in ("id", "type", "role", "model", "usage") if key in original
        }
        native.setdefault("type", "message")
        native.setdefault("role", "assistant")
        stop = _stop_reason(choice.get("finish_reason"), bool(calls), original)
        native["stop_reason"] = stop
        # Which stop sequence was hit means nothing once the message stopped some other way.
        native["stop_sequence"] = original.get("stop_sequence") if stop == "stop_sequence" else None
        native["content"] = blocks
        return native

    def render(self, native: dict[str, Any], *, request: dict[str, Any]) -> bytes:
        if not request.get("stream"):
            return json.dumps(native).encode()
        events: list[tuple[str, dict[str, Any]]] = []
        start = {k: v for k, v in native.items() if k not in ("content", "stop_reason")}
        # As a server opens a stream: how it stopped is only known at the end.
        start.update({"content": [], "stop_reason": None, "stop_sequence": None})
        usage = native.get("usage") if isinstance(native.get("usage"), dict) else {}
        start["usage"] = {k: v for k, v in usage.items() if k != "output_tokens"}
        events.append(("message_start", {"type": "message_start", "message": start}))
        for index, block in enumerate(native.get("content") or []):
            kind = block.get("type")
            if kind == "tool_use":
                opening = {k: v for k, v in block.items() if k != "input"}
                opening["input"] = {}
                deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}]
            elif kind == "thinking":
                opening = {"type": "thinking", "thinking": ""}
                deltas = [{"type": "thinking_delta", "thinking": block.get("thinking", "")}]
                if block.get("signature"):
                    deltas.append({"type": "signature_delta", "signature": block["signature"]})
            elif kind == "text":
                opening = {"type": "text", "text": ""}
                deltas = [{"type": "text_delta", "text": block.get("text", "")}]
            else:
                opening, deltas = block, []
            events.append(
                ("content_block_start", {"type": "content_block_start", "index": index, "content_block": opening})
            )
            for delta in deltas:
                events.append(
                    ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
                )
            events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": native.get("stop_reason"),
                        "stop_sequence": native.get("stop_sequence"),
                    },
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                },
            )
        )
        events.append(("message_stop", {"type": "message_stop"}))
        body = "".join(
            f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n" for name, data in events
        )
        return body.encode("utf-8")

    # --- requests, read-only ----------------------------------------------------------

    def view(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = request.get("system")
        if system:
            messages.append({"role": "system", "content": _parts(system) if isinstance(system, list) else system})
        entries = request.get("messages")
        for message in entries if isinstance(entries, list) else []:
            if not isinstance(message, dict):
                continue  # not a message: the request itself still shows it
            role, content = message.get("role"), message.get("content")
            if isinstance(content, str) or not isinstance(content, list):
                messages.append({"role": role, "content": content if isinstance(content, str) else ""})
                continue
            blocks = [b for b in content if isinstance(b, dict)]
            if role == "assistant":
                messages.append(_assistant_view(blocks))
                continue
            # A user turn: each tool result becomes a tool message, as in the OpenAI shape;
            # whatever else the turn carries follows as one user (or system) message.
            for block in blocks:
                if block.get("type") == "tool_result":
                    result: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _text(block.get("content")),
                    }
                    if block.get("is_error"):
                        result["is_error"] = True
                    messages.append(result)
            rest = [b for b in blocks if b.get("type") != "tool_result"]
            if rest:
                messages.append({"role": role, "content": _parts(rest)})
        return messages

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Without prompt-caching markers, and a lone text block as the string it stands for.

        Claude Code marks the last message of every request for caching: it turns string
        content into a one-block list carrying `cache_control`, and turns it back on the
        next request. The same message, re-encoded — verified on 2.1.281.
        """
        if not isinstance(message, dict):
            return message  # not a message: recorded as sent, and the server can say so
        stable = _without_cache_control(message)
        content = stable.get("content")
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and set(content[0]) == {"type", "text"}
            and content[0]["type"] == "text"
        ):
            stable = {**stable, "content": content[0]["text"]}
        return stable

    def fresh_request(
        self, system: str, user: str, *, model: str | None, max_tokens: int, temperature: float | None = None
    ) -> dict[str, Any]:
        # The instruction is top-level, not a message, and a limit is required.
        body: dict[str, Any] = {
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if model:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def auth_headers(self, key: str) -> dict[str, str]:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}

    def with_context(
        self, request: dict[str, Any], before: list[dict[str, Any]], in_turn: list[dict[str, Any]] = ()
    ) -> dict[str, Any] | None:
        """Not yet: a context is OpenAI-style messages, and these are content blocks."""
        return None

    def with_user_message(
        self, request: dict[str, Any], text: str, *, answer: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        sent = copy.deepcopy(request)
        messages = [m for m in sent.get("messages") or [] if isinstance(m, dict)]
        question: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if answer is not None:
            # Thinking is left out: a block without the server's signature is refused.
            blocks = [
                block for block in self.native_response(answer, None)["content"]
                if block.get("type") in ("text", "tool_use")
            ]
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            results = [
                {"type": "tool_result", "tool_use_id": block["id"], "content": NOT_RUN}
                for block in blocks
                if block.get("type") == "tool_use"
            ]
            messages.append({"role": "user", "content": [*results, *question]})
        elif messages and messages[-1].get("role") == "user":
            # One user turn, not two: the question joins the turn the model was to answer.
            last = dict(messages[-1])
            content = last.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            last["content"] = [*blocks, *question]
            messages[-1] = last
        else:
            messages.append({"role": "user", "content": question})
        sent["messages"] = messages
        sent["stream"] = False
        return sent

    def error(self, status: int, message: str, kind: str) -> dict[str, Any]:
        return {"type": "error", "error": {"type": kind, "message": message}}

    def error_event(self, error: dict[str, Any]) -> bytes:
        return f"event: error\ndata: {json.dumps(error)}\n\n".encode()
