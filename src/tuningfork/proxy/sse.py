"""Server-sent-event parsing and OpenAI chunk reassembly.

The proxy forwards upstream bytes untouched and reassembles a *copy* for the record.
Nothing here can change what the harness receives: a parse failure degrades the recording,
never the traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DONE = "[DONE]"

# Why a recorded step is not the model's answer — its `error`. A step with one is a
# transport failure, not behaviour. Shared by every wire format's assembler, so they live
# here, below `tuningfork.wire`, which imports this module.

#: The upstream reported an error: an error status, an error body, or an error mid-stream.
UPSTREAM_ERROR = "upstream_error"
#: A stream that ended before the model finished: its format's end never arrived.
CUT_OFF = "cut_off: the stream ended before the model finished"


def upstream_error(error: Any) -> str:
    """The step error for an error the upstream sent, with its message when it has one."""
    detail = (error.get("message") or error.get("type")) if isinstance(error, dict) else error
    if not isinstance(detail, str) or not detail.strip():
        return UPSTREAM_ERROR
    return f"{UPSTREAM_ERROR}: {detail[:500]}"


def unreadable(exc: BaseException) -> str:
    """The step error for a response tuningfork could not read. It still reached the
    harness exactly as it came, and its bytes are recorded."""
    return f"tuningfork could not read the response: {type(exc).__name__}: {exc}"


def read_as(headers: Any, asked: bool) -> tuple[bool, bool]:
    """(read at all, read as an event stream) for an answer being recorded, by its headers.

    A stream is read as one where the request asked for one (`asked`, its `stream`) whatever
    its label: one labelled `text/plain` and kept as text would keep whole a key the model sent
    in pieces. A body that turns out to be JSON — an answer sent at once, or an error the server
    sent instead — is read as JSON whatever its label (`json_body`). A content-encoded body is
    forwarded verbatim but not read: decoding a copy just to record it would be work for no
    benefit on a local server, which never compresses.
    """
    encoding = (headers.get("content-encoding") or "").strip().lower()
    if encoding and encoding != "identity":
        return False, False
    return True, asked or "text/event-stream" in (headers.get("content-type") or "").lower()


def json_body(raw: bytes) -> bool:
    """Whether a body read as a stream is JSON instead: one sent at once, where a stream was
    asked for. An event stream's first line is an event's field, or a comment."""
    return raw.lstrip()[:1] in (b"{", b"[")


def event_index(value: Any, default: int) -> int:
    """A stream event's `index`, which ties its fragments together. A server may send
    null, and anything but an integer would break putting the fragments in order."""
    return value if isinstance(value, int) and not isinstance(value, bool) else default


@dataclass(slots=True)
class SSEEvent:
    data: str
    event: str | None = None

    @property
    def is_done(self) -> bool:
        return self.data.strip() == DONE

    def json(self) -> dict[str, Any] | None:
        """Parsed payload, or None if this is the sentinel or not a JSON object."""
        if self.is_done:
            return None
        try:
            parsed = json.loads(self.data)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


class SSEParser:
    """Incremental SSE parser: feed bytes as they arrive, get complete events back.

    Chunk boundaries are arbitrary, so a single `data:` line may arrive split across several
    reads. Everything is buffered until a terminating newline is seen.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._data: list[str] = []
        self._event: str | None = None

    def feed(self, data: bytes) -> list[SSEEvent]:
        self._buf.extend(data)
        events: list[SSEEvent] = []
        while (idx := self._buf.find(b"\n")) != -1:
            line = bytes(self._buf[:idx]).rstrip(b"\r").decode("utf-8", "replace")
            del self._buf[: idx + 1]
            if (event := self._consume_line(line)) is not None:
                events.append(event)
        return events

    def _consume_line(self, line: str) -> SSEEvent | None:
        if not line:  # a blank line dispatches the accumulated event
            if not self._data:
                self._event = None
                return None
            event = SSEEvent(data="\n".join(self._data), event=self._event)
            self._data.clear()
            self._event = None
            return event
        if line.startswith(":"):
            return None  # comment, used as a keepalive while a breakpoint holds
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        return None


@dataclass(slots=True)
class _Choice:
    index: int
    role: str | None = None
    # Any string-valued delta field, accumulated by key: `content`, `reasoning_content`, ...
    text: dict[str, str] = field(default_factory=dict)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    logprobs: dict[str, list[Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class ChunkAssembler:
    """Rebuild a non-streaming-shaped response from `chat.completion.chunk` events.

    Delta fields are accumulated *generically*: any string-valued key is appended under its
    own name. Vendor extensions therefore survive without this module knowing about them,
    which is how `reasoning_content` is captured with no Gemma-specific branch.
    """

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.usage: dict[str, Any] | None = None
        self.chunk_count = 0
        self.saw_done = False
        #: An error the server sent mid-stream, in place of a chunk.
        self.error: Any = None
        self._choices: dict[int, _Choice] = {}

    def add_event(self, event: SSEEvent) -> None:
        if event.is_done:
            self.saw_done = True
            return
        if (chunk := event.json()) is not None:
            self.add_chunk(chunk)

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        self.chunk_count += 1
        for key in ("id", "model", "created", "system_fingerprint"):
            if chunk.get(key) is not None:
                self.meta[key] = chunk[key]
        # With stream_options={"include_usage": True} the usage arrives in a final chunk
        # that carries no choices at all.
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        if chunk.get("error") is not None:
            self.error = chunk["error"]

        choices = chunk.get("choices")
        for raw in choices if isinstance(choices, list) else ():
            if not isinstance(raw, dict):
                continue
            index = event_index(raw.get("index"), 0)
            choice = self._choices.setdefault(index, _Choice(index=index))
            if raw.get("finish_reason") is not None:
                choice.finish_reason = raw["finish_reason"]
            if isinstance(raw.get("logprobs"), dict):
                for key, value in raw["logprobs"].items():
                    if isinstance(value, list):
                        choice.logprobs.setdefault(key, []).extend(value)
            if isinstance(raw.get("delta"), dict):
                self._apply_delta(choice, raw["delta"])

    @staticmethod
    def _apply_delta(choice: _Choice, delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            if value is None:
                continue
            if key == "role":
                choice.role = value
            elif key == "tool_calls" and isinstance(value, list):
                for call in value:
                    ChunkAssembler._merge_tool_call(choice, call)
            elif isinstance(value, str):
                choice.text[key] = choice.text.get(key, "") + value
            else:
                choice.extra[key] = value

    @staticmethod
    def _merge_tool_call(choice: _Choice, call: Any) -> None:
        if not isinstance(call, dict):
            return
        # `index` ties fragments of one call together across chunks. A server that omits it
        # is assumed to emit calls in order.
        index = event_index(call.get("index"), len(choice.tool_calls))
        slot = choice.tool_calls.setdefault(index, {"index": index, "function": {}})
        for key, value in call.items():
            if key == "index" or value is None:
                continue
            if key != "function":
                slot[key] = value
            elif isinstance(value, dict):
                function = slot["function"]
                for fkey, fvalue in value.items():
                    if fvalue is None:
                        continue
                    if fkey == "arguments":
                        if not isinstance(fvalue, str):
                            # Sent already parsed, as an object. The common shape carries
                            # arguments as JSON text, as OpenAI's own does.
                            fvalue = json.dumps(fvalue, ensure_ascii=False)
                        function["arguments"] = function.get("arguments", "") + fvalue
                    else:
                        function[fkey] = fvalue

    def failure(self) -> str | None:
        """Why the stream was not a whole answer, or None if it was. Read once it has ended.

        An error sent mid-stream; or a stream cut off before the model finished — one with
        neither `[DONE]` nor any finish reason. Either one is enough to count as finished:
        some servers never send `[DONE]`.
        """
        if self.error is not None:
            return upstream_error(self.error)
        if not self.saw_done and not any(c.finish_reason for c in self._choices.values()):
            return CUT_OFF
        return None

    def result(self) -> dict[str, Any]:
        """The reassembled response, shaped like a non-streaming completion."""
        choices: list[dict[str, Any]] = []
        for index in sorted(self._choices):
            choice = self._choices[index]
            message: dict[str, Any] = {"role": choice.role or "assistant"}
            message.update(choice.text)
            message.update(choice.extra)
            if choice.tool_calls:
                message["tool_calls"] = [
                    choice.tool_calls[i] for i in sorted(choice.tool_calls)
                ]
            # Downstream code can rely on `content` existing; a pure tool call has none.
            message.setdefault("content", "")
            entry: dict[str, Any] = {
                "index": index,
                "message": message,
                "finish_reason": choice.finish_reason,
            }
            if choice.logprobs:
                entry["logprobs"] = choice.logprobs
            choices.append(entry)

        result = dict(self.meta)
        result["object"] = "chat.completion"
        result["choices"] = choices
        if self.usage is not None:
            result["usage"] = self.usage
        if self.error is not None:
            result["error"] = self.error  # kept beside what arrived before it, visibly
        return result


def render_sse(response: dict[str, Any], *, include_usage: bool = True) -> bytes:
    """Serialize a completion back into a stream a client can consume.

    Needed when a response is edited at a breakpoint: the upstream bytes no longer describe
    it. The stream mirrors what real servers were observed to send — role, then reasoning,
    then content, then each tool call as an opening fragment followed by its arguments —
    and ends with the finish reason, a usage chunk if the client asked for one, and
    `[DONE]`. `reassemble` of the result gives back the same response.
    """
    meta = {
        key: response[key]
        for key in ("id", "created", "model", "system_fingerprint")
        if response.get(key) is not None
    }
    meta["object"] = "chat.completion.chunk"
    events: list[dict[str, Any]] = []

    def emit(index: int, delta: dict[str, Any], finish: str | None = None) -> None:
        events.append(
            {**meta, "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}
        )

    for choice in response.get("choices") or []:
        index = choice.get("index", 0)
        message = choice.get("message") or {}
        emit(index, {"role": message.get("role") or "assistant"})
        text_keys = ["reasoning_content", "reasoning", "content"]
        text_keys += sorted(
            key
            for key, value in message.items()
            if isinstance(value, str) and key not in {*text_keys, "role"}
        )
        for key in text_keys:
            value = message.get(key)
            if isinstance(value, str) and value:
                emit(index, {key: value})
        calls = message.get("tool_calls") or []
        for position, call in enumerate(calls):
            function = call.get("function") or {}
            emit(index, {"tool_calls": [{
                "index": position,
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {"name": function.get("name"), "arguments": ""},
            }]})
            emit(index, {"tool_calls": [{
                "index": position,
                "function": {"arguments": function.get("arguments") or ""},
            }]})
        emit(index, {}, choice.get("finish_reason") or ("tool_calls" if calls else "stop"))

    if include_usage and isinstance(response.get("usage"), dict):
        events.append({**meta, "choices": [], "usage": response["usage"]})

    body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
    return (body + f"data: {DONE}\n\n").encode("utf-8")


def reassemble(raw: bytes) -> dict[str, Any]:
    """Convenience: parse a complete SSE body and return the assembled response."""
    parser = SSEParser()
    assembler = ChunkAssembler()
    for event in parser.feed(raw):
        assembler.add_event(event)
    return assembler.result()
