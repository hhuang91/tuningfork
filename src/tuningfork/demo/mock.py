"""A scriptable OpenAI-compatible server.

Every tuningfork test runs against this instead of a real model, so the whole suite passes
with no GPU, no weights and no network. That is the contribution gate: an outside PR has to
be runnable by anyone. `tuningfork demo` puts it behind a real proxy, as its model.

It reproduces the wire behaviour observed from real servers, including the parts that are
easy to get wrong:

- `reasoning_content` arrives as its own delta field, interleaved before content.
- A tool call streams as an opening chunk carrying `id` and `function.name`, then the whole
  argument JSON in a single later chunk.
- `usage` arrives in a trailing chunk with an empty `choices` list, and only when the
  request asked for it via `stream_options`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tuningfork.demo.toy import ToyLlama

DEFAULT_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


@dataclass(slots=True)
class MockTurn:
    """One scripted response.

    `stall` delays the first byte; `chunk_delay` spaces the rest. Both exist for the Phase 3
    keepalive work, where the proxy must hold a response without the harness timing out.
    """

    content: str = ""
    reasoning: str = ""
    # Each entry: {"name": str, "arguments": dict, "id": str | None}, and on Anthropic
    # Messages "arguments_text": the JSON text a stream sends the input as (default: dumped).
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    status: int = 200
    error: dict[str, Any] | None = None
    stall: float = 0.0
    chunk_delay: float = 0.0
    logprobs: dict[str, Any] | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: How many choices to return, for servers that honour `n` (many local ones do not).
    n_choices: int = 1
    #: An error a stream sends in place of its end, at status 200: llama.cpp's
    #: `data: {"error": …}` when an exception is raised while it streams.
    error_event: dict[str, Any] | None = None

    @property
    def resolved_finish_reason(self) -> str:
        if self.finish_reason:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


def _pieces(text: str) -> list[str]:
    """Split text the way a server streams it: small fragments that reassemble exactly."""
    return re.findall(r"\s*\S+", text) if text else []


class MockServer:
    """Queue up `MockTurn`s, then point a client at `.app`.

    Received requests are recorded on `.requests` (parsed) and `.raw_bodies` (bytes) so
    tests can assert on exactly what the proxy forwarded.
    """

    def __init__(self, *turns: MockTurn, model: str = "mock-model") -> None:
        self.model = model
        self.turns: deque[MockTurn] = deque(turns)
        self.default_turn = MockTurn(content="ok")
        self.requests: list[dict[str, Any]] = []
        self.raw_bodies: list[bytes] = []
        self.seen: list[tuple[str, str, dict[str, str]]] = []  # method, path, headers
        # Peak concurrency, for asserting that the proxy honours its in-flight cap.
        # Counts the handler only, so measure it with non-streaming turns plus `stall`.
        self.inflight = 0
        self.peak_inflight = 0
        #: Paths that should 404, so probes for optional endpoints can be tested.
        self.absent_paths: set[str] = set()
        #: When set, chooses each response from the request instead of the script — for
        #: tests where the answer must depend on what was asked (a wording A/B, say).
        self.responder: Callable[[dict[str, Any]], MockTurn] | None = None
        #: When set, served at GET /props the way llama.cpp exposes its defaults.
        self.props: dict[str, Any] | None = None
        #: llama.cpp's raw endpoints, answered by a toy model (see `toy.py`).
        self.toy = ToyLlama()
        #: Delays each /completion, for asserting the proxy gates it.
        self.completion_stall = 0.0
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions/input_tokens", self._raw, methods=["POST"]),
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/v1/messages", self._messages, methods=["POST"]),
                Route("/v1/models", self._models, methods=["GET"]),
                Route("/apply-template", self._raw, methods=["POST"]),
                Route("/tokenize", self._raw, methods=["POST"]),
                Route("/detokenize", self._raw, methods=["POST"]),
                Route("/completion", self._raw, methods=["POST"]),
                Route(
                    "/{path:path}",
                    self._echo,
                    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                ),
            ]
        )

    def script(self, *turns: MockTurn) -> MockServer:
        self.turns.extend(turns)
        return self

    def _next_turn(self) -> MockTurn:
        return self.turns.popleft() if self.turns else self.default_turn

    # --- endpoints -----------------------------------------------------------------

    async def _chat(self, request: Request) -> Response:
        body = await request.body()
        self.raw_bodies.append(body)
        payload: dict[str, Any] = json.loads(body) if body else {}
        self.requests.append(payload)
        self.seen.append(("POST", request.url.path, dict(request.headers)))

        turn = self.responder(payload) if self.responder else self._next_turn()
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            if turn.error is not None or turn.status >= 400:
                return JSONResponse(
                    turn.error or {"error": {"message": "mock failure", "type": "mock"}},
                    status_code=turn.status if turn.status >= 400 else 500,
                    headers=turn.extra_headers,
                )

            if payload.get("stream"):
                return StreamingResponse(
                    self._stream(turn, payload),
                    media_type="text/event-stream",
                    headers=turn.extra_headers,
                )
            if turn.stall:
                await asyncio.sleep(turn.stall)
            return JSONResponse(self._complete(turn), headers=turn.extra_headers)
        finally:
            self.inflight -= 1

    async def _messages(self, request: Request) -> Response:
        """Anthropic Messages, shaped like llama.cpp's native `/v1/messages` answers."""
        body = await request.body()
        self.raw_bodies.append(body)
        payload: dict[str, Any] = json.loads(body) if body else {}
        self.requests.append(payload)
        self.seen.append(("POST", request.url.path, dict(request.headers)))
        turn = self.responder(payload) if self.responder else self._next_turn()
        if turn.error is not None or turn.status >= 400:
            return JSONResponse(
                turn.error
                or {"type": "error", "error": {"type": "api_error", "message": "mock failure"}},
                status_code=turn.status if turn.status >= 400 else 500,
            )
        if payload.get("stream"):
            return StreamingResponse(
                self._anthropic_stream(turn), media_type="text/event-stream"
            )
        return JSONResponse(self._anthropic_message(turn))

    def _anthropic_blocks(self, turn: MockTurn) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if turn.reasoning:
            blocks.append({"type": "thinking", "thinking": turn.reasoning, "signature": "sig-mock"})
        if turn.content:
            blocks.append({"type": "text", "text": turn.content})
        for i, call in enumerate(turn.tool_calls):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") or f"toolu_{i}",
                    "name": call["name"],
                    "input": call.get("arguments", {}),
                }
            )
        return blocks

    def _anthropic_message(self, turn: MockTurn) -> dict[str, Any]:
        stops = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}
        return {
            "id": f"msg_mock{len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": self._anthropic_blocks(turn),
            "stop_reason": stops.get(turn.resolved_finish_reason, turn.resolved_finish_reason),
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "cache_read_input_tokens": 30, "output_tokens": 7},
        }

    async def _anthropic_stream(self, turn: MockTurn):
        if turn.stall:
            await asyncio.sleep(turn.stall)
        message = self._anthropic_message(turn)

        def event(name: str, data: dict[str, Any]) -> bytes:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

        start = {**message, "content": [], "stop_reason": None}
        start["usage"] = {k: v for k, v in message["usage"].items() if k != "output_tokens"}
        yield event("message_start", {"type": "message_start", "message": start})
        yield event("ping", {"type": "ping"})
        texts = iter(call.get("arguments_text") for call in turn.tool_calls)
        for index, block in enumerate(message["content"]):
            if block["type"] == "tool_use":
                opening = {**block, "input": {}}
                raw = next(texts) or json.dumps(block["input"])
                # Split mid-JSON, as servers do: only the whole is parseable.
                deltas = [
                    {"type": "input_json_delta", "partial_json": raw[: len(raw) // 2]},
                    {"type": "input_json_delta", "partial_json": raw[len(raw) // 2 :]},
                ]
            elif block["type"] == "thinking":
                opening = {"type": "thinking", "thinking": ""}
                deltas = [{"type": "thinking_delta", "thinking": p} for p in _pieces(block["thinking"])]
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
            else:
                opening = {"type": "text", "text": ""}
                deltas = [{"type": "text_delta", "text": p} for p in _pieces(block["text"])]
            yield event("content_block_start", {"type": "content_block_start", "index": index, "content_block": opening})
            for delta in deltas:
                if turn.chunk_delay:
                    await asyncio.sleep(turn.chunk_delay)
                yield event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
            yield event("content_block_stop", {"type": "content_block_stop", "index": index})
        yield event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]},
            },
        )
        yield event("message_stop", {"type": "message_stop"})

    async def _raw(self, request: Request) -> Response:
        """llama.cpp's own endpoints, from the toy model: see `toy.py`."""
        path = request.url.path
        body = await request.body()
        self.seen.append(("POST", path, dict(request.headers)))
        if path in self.absent_paths:
            return JSONResponse({"error": {"code": 404, "message": "File Not Found"}}, status_code=404)
        payload: dict[str, Any] = json.loads(body) if body else {}
        if path == "/apply-template":
            return JSONResponse({"prompt": self.toy.apply_template(payload)})
        if path == "/tokenize":
            return JSONResponse(self.toy.tokenize_body(payload))
        if path == "/detokenize":
            return JSONResponse(self.toy.detokenize_body(payload))
        if path.endswith("/input_tokens"):
            return JSONResponse({"object": "response.input_tokens", "input_tokens": self.toy.input_tokens(payload)})
        if payload.get("grammar") and self.toy.refuse_grammar:
            return JSONResponse({"error": {"code": 400, "message": "failed to parse grammar"}}, status_code=400)
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            if self.completion_stall:
                await asyncio.sleep(self.completion_stall)
            return JSONResponse(self.toy.complete(payload))
        finally:
            self.inflight -= 1

    async def _models(self, request: Request) -> Response:
        self.seen.append(("GET", request.url.path, dict(request.headers)))
        return JSONResponse(
            {"object": "list", "data": [{"id": self.model, "object": "model"}]}
        )

    async def _echo(self, request: Request) -> Response:
        """Catch-all so passthrough tests can prove a request arrived unaltered."""
        body = await request.body()
        self.seen.append((request.method, request.url.path, dict(request.headers)))
        if request.url.path in self.absent_paths:
            return JSONResponse({"error": {"message": "not found"}}, status_code=404)
        if request.url.path == "/props" and self.props is not None:
            return JSONResponse(self.props)
        return JSONResponse(
            {
                "echo": True,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "body": body.decode("utf-8", "replace"),
            }
        )

    # --- response construction -----------------------------------------------------

    def _message(self, turn: MockTurn) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": turn.content}
        if turn.reasoning:
            message["reasoning_content"] = turn.reasoning
        if turn.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {})),
                    },
                }
                for i, call in enumerate(turn.tool_calls)
            ]
        return message

    def _complete(self, turn: MockTurn) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-mock{len(self.requests)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": index,
                    "message": self._message(turn),
                    "logprobs": turn.logprobs,
                    "finish_reason": turn.resolved_finish_reason,
                }
                for index in range(max(1, turn.n_choices))
            ],
            "usage": turn.usage or DEFAULT_USAGE,
        }

    async def _stream(self, turn: MockTurn, payload: dict[str, Any]):
        if turn.stall:
            await asyncio.sleep(turn.stall)

        completion_id = f"chatcmpl-mock{len(self.requests)}"
        created = int(time.time())

        def chunk(
            delta: dict[str, Any] | None = None,
            finish_reason: str | None = None,
            usage: dict[str, Any] | None = None,
            logprobs: dict[str, Any] | None = None,
        ) -> bytes:
            choices: list[dict[str, Any]] = []
            if delta is not None or finish_reason is not None:
                choices.append(
                    {
                        "index": 0,
                        "delta": delta if delta is not None else {},
                        "logprobs": logprobs,
                        "finish_reason": finish_reason,
                    }
                )
            body: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model,
                "choices": choices,
            }
            if usage is not None:
                body["usage"] = usage
            return f"data: {json.dumps(body)}\n\n".encode()

        async def pace() -> None:
            if turn.chunk_delay:
                await asyncio.sleep(turn.chunk_delay)

        yield chunk({"role": "assistant"})

        for piece in _pieces(turn.reasoning):
            await pace()
            yield chunk({"reasoning_content": piece})

        for piece in _pieces(turn.content):
            await pace()
            yield chunk({"content": piece})

        for i, call in enumerate(turn.tool_calls):
            await pace()
            # Opening fragment: identity and name, empty arguments.
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {"name": call["name"], "arguments": ""},
                        }
                    ]
                }
            )
            # Arguments land whole in one later chunk, as observed on real servers.
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(call.get("arguments", {}))
                            },
                        }
                    ]
                }
            )

        if turn.error_event is not None:
            yield f"data: {json.dumps({'error': turn.error_event})}\n\n".encode()
            return

        yield chunk({}, finish_reason=turn.resolved_finish_reason, logprobs=turn.logprobs)

        stream_options = payload.get("stream_options") or {}
        if stream_options.get("include_usage"):
            yield chunk(usage=turn.usage or DEFAULT_USAGE)

        yield b"data: [DONE]\n\n"
