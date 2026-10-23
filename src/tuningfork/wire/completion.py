"""Raw completions: a prompt in, text out — llama.cpp's `/completion` and `/infill`, and the
OpenAI-style `/completions`.

No harness talks this way; tuningfork's own token-level work does (`tuningfork.backends.raw`).
It is a wire format all the same, so that the proxy counts such a call as model load: it
takes the gate and waits out a pause like any other. It is never recorded — the proxy records
only a call with a list of messages, and tuningfork's own calls carry the experiment header —
and never held, so the methods only a recorded or held call would need are refused.
"""

from __future__ import annotations

import json
from typing import Any

from tuningfork.proxy.sse import UPSTREAM_ERROR, upstream_error

SUFFIXES = ("/completion", "/completions", "/infill")
#: How a raw completion stopped, as a chat completion's finish reason.
FINISH = {"eos": "stop", "word": "stop", "limit": "length"}


class _Assembler:
    """The text of a streamed raw completion, chunk by chunk."""

    def __init__(self) -> None:
        self.chunk_count = 0
        self.content = ""
        self.stop_type: str | None = None

    def add_event(self, event: Any) -> None:
        data = event.json() if hasattr(event, "json") else None
        if not isinstance(data, dict):
            return
        self.chunk_count += 1
        self.content += data.get("content") or ""
        if data.get("stop_type"):
            self.stop_type = data["stop_type"]

    def result(self) -> dict[str, Any]:
        return {"content": self.content, "stop_type": self.stop_type}

    def failure(self) -> str | None:
        return None


class RawCompletion:
    name = "completion"
    endpoint = "/completion"
    output_limits = ("n_predict", "max_tokens")

    def matches(self, method: str, path: str) -> bool:
        return (
            method == "POST"
            and path.endswith(SUFFIXES)
            and not path.endswith("/chat/completions")
        )

    def assembler(self) -> _Assembler:
        return _Assembler()

    def parse(self, raw: bytes, event_stream: bool) -> dict[str, Any] | None:
        if event_stream:
            from tuningfork.proxy.sse import SSEParser

            assembler = _Assembler()
            for event in SSEParser().feed(raw):
                assembler.add_event(event)
            return assembler.result() if assembler.chunk_count else None
        try:
            decoded = json.loads(raw)
        except (ValueError, RecursionError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def failure(
        self, status: int, response: dict[str, Any] | None, assembler: Any = None
    ) -> str | None:
        if status >= 400:
            return UPSTREAM_ERROR
        if isinstance(response, dict) and response.get("error") is not None:
            return upstream_error(response["error"])
        return None

    def canonical_response(self, native: dict[str, Any], *, stream: bytes | None = None) -> dict[str, Any]:
        if native.get("error") is not None and "content" not in native:
            return {"error": native["error"]}
        return {
            "object": "text_completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": native.get("content") or ""},
                "finish_reason": FINISH.get(native.get("stop_type") or "", None),
            }],
            "usage": {
                "prompt_tokens": native.get("tokens_evaluated"),
                "completion_tokens": native.get("tokens_predicted"),
            },
        }

    def native_response(self, canonical: dict[str, Any], original: dict[str, Any] | None) -> dict[str, Any]:
        raise NotImplementedError("a raw completion is never held, so never edited")

    def render(self, native: dict[str, Any], *, request: dict[str, Any]) -> bytes:
        return json.dumps(native).encode()

    def view(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        return []  # a prompt, not a conversation

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return message

    def with_context(
        self, request: dict[str, Any], before: list[dict[str, Any]], in_turn: list[dict[str, Any]] = ()
    ) -> dict[str, Any] | None:
        """Never: a raw completion is a prompt, not a conversation."""
        return None

    def with_user_message(
        self, request: dict[str, Any], text: str, *, answer: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError("a raw completion is never held, so never asked a question")

    def fresh_request(
        self, system: str, user: str, *, model: str | None, max_tokens: int, temperature: float | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError("a raw completion takes a prompt, not a conversation")

    def auth_headers(self, key: str) -> dict[str, str]:
        return {"authorization": f"Bearer {key}"}

    def error(self, status: int, message: str, kind: str) -> dict[str, Any]:
        return {"error": {"code": status, "message": message, "type": kind}}

    def error_event(self, error: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(error)}\n\n".encode()
