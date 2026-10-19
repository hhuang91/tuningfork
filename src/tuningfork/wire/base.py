"""The wire-format seam.

A harness talks to its model in some HTTP dialect — OpenAI chat-completions, Anthropic
Messages. The proxy forwards whatever bytes pass through untouched; a wire format is how
tuningfork *reads* a call of its dialect, and how it writes one back when a breakpoint edit
means the upstream bytes no longer describe what the harness should receive.

Requests are stored in their own format, exactly as sent, so a resample can resend them.
Responses are additionally kept in one common shape — an OpenAI `chat.completion` — which
is what outcomes, breakpoint conditions and the UI read. Harness requests are shown through
the same lens: `view` turns a request's messages into OpenAI-style messages for display
and matching, never for sending.

Supporting a new dialect means adding one of these; nothing under `tuningfork.proxy` should
need to change.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable


def call_id(prefix: str) -> str:
    """An id for a tool call a person added at a breakpoint without giving it one. The
    harness answers a call by its id, so a call without one could never be answered."""
    return f"{prefix}{uuid.uuid4().hex[:24]}"


@runtime_checkable
class Assembler(Protocol):
    """Rebuilds a complete response from a stream's events, as they arrive."""

    chunk_count: int

    def add_event(self, event: Any) -> None: ...

    def result(self) -> dict[str, Any]: ...

    def failure(self) -> str | None:
        """Why the stream was not a whole answer — it carried an error, or ended before the
        model finished — or None if it was one. Read once the stream has ended."""


#: What a tool call the model made at a held call is answered with, when a person asks the
#: model something there instead of letting the harness run it.
NOT_RUN = "[not run: tuningfork paused here so that a person could ask you something]"


@runtime_checkable
class WireFormat(Protocol):
    name: str
    #: Where an experiment sends a request of this format, relative to the server root.
    endpoint: str
    #: The request fields that limit how many tokens the model may write, the first the one
    #: written into a request that names none (`cap_output`).
    output_limits: tuple[str, ...]

    def matches(self, method: str, path: str) -> bool:
        """Whether a request to `path` is a model call in this format."""

    def assembler(self) -> Assembler:
        """For reassembling a streamed response."""

    def parse(self, raw: bytes, event_stream: bool) -> dict[str, Any] | None:
        """A complete response body, streamed or not, in this format."""

    def failure(
        self, status: int, response: dict[str, Any] | None, assembler: Assembler | None = None
    ) -> str | None:
        """Why a response is not an answer from the model, or None if it is one.

        An error status, or an error body; for a response that streamed, whatever its
        `assembler` found — an error mid-stream, or an end before the model finished. The
        proxy records it as the step's error: such a step is a transport failure, not
        behaviour, however much of an answer arrived before it.
        """

    def canonical_response(self, native: dict[str, Any]) -> dict[str, Any]:
        """The response in the common shape: an OpenAI `chat.completion`. An error body
        has no choices, and keeps its error under `error`."""

    def native_response(
        self, canonical: dict[str, Any], original: dict[str, Any] | None
    ) -> dict[str, Any]:
        """An edited common-shape response back in this format, keeping what the edit did
        not touch from `original`. A tool call's arguments may arrive as an object rather
        than JSON text, and a call a person added may have no id yet."""

    def render(self, native: dict[str, Any], *, request: dict[str, Any]) -> bytes:
        """The body the harness receives: a stream if its request asked for one."""

    def view(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """The request's conversation as OpenAI-style messages, for reading only. An entry
        that is not a message object is left out; the request itself still has it."""

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """A message in a stable form, for recognising it again in the next request.

        A harness may re-encode an earlier message from one request to the next without
        changing what it says; the message tree must still see one message. Exact forms are
        kept per step, so nothing a harness sent is lost. An entry that is not an object is
        returned as it is: a malformed request is still relayed, and recorded as sent.
        """

    def with_user_message(
        self, request: dict[str, Any], text: str, *, answer: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The request continued by a person's question — after the model's held `answer`
        (common shape), when there is one — to send once, not streamed. A tool call in that
        answer is answered as not run (`NOT_RUN`), so the conversation stays one the model's
        chat template accepts."""

    def with_context(self, request: dict[str, Any], context: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The request continuing a conversation it was not given — a scenario's context
        (`tuningfork.context`), OpenAI-style messages — after the harness's instructions and
        before its own first message: what the model is sent. None where this format cannot
        carry one yet: the call is refused rather than sent without it."""

    def fresh_request(
        self, system: str, user: str, *, model: str | None, max_tokens: int, temperature: float | None = None
    ) -> dict[str, Any]:
        """A new conversation — one instruction, one user message — not streamed: what
        tuningfork sends when it asks a model something of its own (`ops.proposer`)."""

    def auth_headers(self, key: str) -> dict[str, str]:
        """How an API of this format takes its key."""

    def error(self, status: int, message: str, kind: str) -> dict[str, Any]:
        """An error body a client of this format understands."""

    def error_event(self, error: dict[str, Any]) -> bytes:
        """The same error as a stream event, for a call whose response has begun."""


def cap_output(
    wire: WireFormat, request: dict[str, Any], cap: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """`request` held to at most `cap` output tokens, and a note of what was done: each limit
    it names above `cap`, or as no limit at all, lowered to `cap` — or, when it names none, the
    format's own limit written. A request that already asks for no more comes back as it was,
    with no note. The note keeps what the harness asked, so `uncapped` can give it back."""
    asked = {key: request[key] for key in wire.output_limits if key in request}
    over = [key for key, value in asked.items() if not _within(value, cap)]
    if asked and not over:
        return request, None
    written = over or [wire.output_limits[0]]
    return {**request, **dict.fromkeys(written, cap)}, {"cap": cap, "asked": asked, "written": written}


def _within(value: Any, cap: int) -> bool:
    """A limit of at most `cap` tokens: any number, a float too — llama.cpp reads one by its
    whole part, and found in review, 1024.0 read as no limit was raised to the cap. A negative
    one is none at all (llama.cpp's -1), and so is null, or anything that is not a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= cap


def uncapped(request: dict[str, Any], note: dict[str, Any] | None) -> dict[str, Any]:
    """The request as its harness sent it, before `cap_output` wrote what `note` says."""
    if not note:
        return request
    asked = note.get("asked") or {}
    restored = dict(request)
    for key in note.get("written") or ():
        if key in asked:
            restored[key] = asked[key]
        else:
            restored.pop(key, None)
    return restored
