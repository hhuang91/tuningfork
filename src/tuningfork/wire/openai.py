"""OpenAI chat-completions: the common shape itself, so conversions are identities."""

from __future__ import annotations

import copy
import json
from typing import Any

from tuningfork.proxy.sse import (
    UPSTREAM_ERROR,
    ChunkAssembler,
    SSEParser,
    render_sse,
    upstream_error,
)
from tuningfork.wire.base import NOT_RUN, Assembler, call_id

SUFFIX = "/chat/completions"


class OpenAIChat:
    name = "openai-chat"
    endpoint = "/v1/chat/completions"
    #: OpenAI's two names, and llama.cpp's own, which it reads before `max_tokens`.
    output_limits = ("max_tokens", "max_completion_tokens", "n_predict")

    def matches(self, method: str, path: str) -> bool:
        return method == "POST" and path.endswith(SUFFIX)

    def assembler(self) -> ChunkAssembler:
        return ChunkAssembler()

    def parse(self, raw: bytes, event_stream: bool) -> dict[str, Any] | None:
        if event_stream:
            assembler = ChunkAssembler()
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
        # Some servers answer an error with a 200 and an error body.
        if isinstance(response, dict) and response.get("error") is not None:
            if not response.get("choices"):
                return upstream_error(response["error"])
        return None

    def canonical_response(self, native: dict[str, Any]) -> dict[str, Any]:
        return native

    def native_response(
        self, canonical: dict[str, Any], original: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The edit as it stands, but with each tool call in a form a harness can act on.

        `function.arguments` is JSON text on this wire: an object would reach the harness as
        an object, which an OpenAI client appends to a string, or fails to parse. And a call
        a person added has no id until it is given one.
        """
        native = copy.deepcopy(canonical)
        choices = native.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, dict):
                    continue
                if not call.get("id"):
                    call["id"] = call_id("call_")
                    call.setdefault("type", "function")
                function = call.get("function")
                if isinstance(function, dict) and function.get("arguments") is not None:
                    if not isinstance(function["arguments"], str):
                        function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
        return native

    def render(self, native: dict[str, Any], *, request: dict[str, Any]) -> bytes:
        if request.get("stream"):
            include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
            return render_sse(native, include_usage=include_usage)
        return json.dumps(native).encode()

    def view(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        messages = request.get("messages")
        if not isinstance(messages, list):
            return []
        return [message for message in messages if isinstance(message, dict)]

    def normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return message

    def with_user_message(
        self, request: dict[str, Any], text: str, *, answer: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        sent = copy.deepcopy(request)
        messages = [m for m in sent.get("messages") or [] if isinstance(m, dict)]
        if answer is not None:
            choices = answer.get("choices") or [{}]
            held = choices[0].get("message") if isinstance(choices[0], dict) else None
            held = held if isinstance(held, dict) else {}
            said: dict[str, Any] = {"role": "assistant", "content": held.get("content") or ""}
            calls = [c for c in held.get("tool_calls") or [] if isinstance(c, dict)]
            if calls:
                said["tool_calls"] = calls
            messages.append(said)
            messages += [{"role": "tool", "tool_call_id": c.get("id"), "content": NOT_RUN} for c in calls]
        messages.append({"role": "user", "content": text})
        sent["messages"] = messages
        sent["stream"] = False
        sent.pop("stream_options", None)
        return sent

    def with_context(
        self, request: dict[str, Any], before: list[dict[str, Any]], in_turn: list[dict[str, Any]] = ()
    ) -> dict[str, Any] | None:
        messages = list(request.get("messages") or [])
        # After the instructions: the leading system (or developer) messages.
        at = next((index for index, message in enumerate(messages)
                   if not (isinstance(message, dict) and message.get("role") in ("system", "developer"))), len(messages))
        rest = messages[at:]
        if in_turn:
            # The model's earlier steps of its turn go after the prompt: the harness's first
            # user message. A call without one has no turn to put them in.
            prompt = next((index for index, message in enumerate(rest)
                           if isinstance(message, dict) and message.get("role") == "user"), None)
            if prompt is None:
                return None
            rest = [*rest[:prompt + 1], *copy.deepcopy(list(in_turn)), *rest[prompt + 1:]]
        return {**request, "messages": [*messages[:at], *copy.deepcopy(before), *rest]}

    def fresh_request(
        self, system: str, user: str, *, model: str | None, max_tokens: int, temperature: float | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if model:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def auth_headers(self, key: str) -> dict[str, str]:
        return {"authorization": f"Bearer {key}"}

    def error(self, status: int, message: str, kind: str) -> dict[str, Any]:
        return {"error": {"message": message, "type": kind, "code": status}}

    def error_event(self, error: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(error)}\n\n".encode()
