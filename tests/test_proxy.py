"""Proxy behaviour: fidelity first, recording second.

The load-bearing property is that recording is a tee. Whatever the harness would have
received talking straight to the server, it must still receive through tuningfork — byte for
byte, and even when storage fails.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from tuningfork.analysis.normalize import outcome_for
from tuningfork.analysis.tokens import TokenStream
from tuningfork.breakpoints import Breakpoint
from tuningfork.config import DEFAULT_REDACTIONS, EXPERIMENT_HEADER, REDACTED, Settings, redact
from tuningfork.proxy.app import create_app, upstream_url
from tuningfork.proxy.recorder import stream_without_secrets
from tuningfork.proxy.sse import CUT_OFF, ChunkAssembler, reassemble
from tuningfork.wire.anthropic import AnthropicMessages
from tests.helpers import all_steps, only_step
from tests.mock_server import MockTurn
from tests.test_wire import (
    MESSAGE_DELTA,
    MESSAGE_STOP,
    OVERLOADED,
    claude_code_request,
    event,
    message_start,
    text_block,
)

CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"


def simple_request(**extra) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}


def openai_chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    body = {"id": "c1", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(body)}\n\n".encode()


def tool_block(name: str, partials: list[str], index: int = 0) -> str:
    """A Messages tool_use block whose input streams in these pieces."""
    opening = {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}}
    return (
        event("content_block_start", {"type": "content_block_start", "index": index, "content_block": opening})
        + "".join(
            event("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "input_json_delta", "partial_json": piece}})
            for piece in partials
        )
        + event("content_block_stop", {"type": "content_block_stop", "index": index})
    )


def thinking_block(pieces: list[str], index: int = 0) -> str:
    """A Messages thinking block streamed in these pieces, then signed."""
    opening = {"type": "thinking", "thinking": ""}
    deltas = [{"type": "thinking_delta", "thinking": piece} for piece in pieces]
    deltas.append({"type": "signature_delta", "signature": "c2lnbmVk"})
    return (
        event("content_block_start", {"type": "content_block_start", "index": index, "content_block": opening})
        + "".join(
            event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
            for delta in deltas
        )
        + event("content_block_stop", {"type": "content_block_stop", "index": index})
    )


def upstream_sending(
    *chunks: bytes | str, status: int = 200, media_type: str = "text/event-stream"
) -> Starlette:
    """An upstream answering every call with exactly these bytes, a chunk at a time: for
    what real servers send and the mock server does not. It keeps what it was sent."""
    parts = [chunk.encode() if isinstance(chunk, str) else chunk for chunk in chunks]
    received: list[bytes] = []

    async def answer(request: Request) -> Response:
        received.append(await request.body())

        async def body():
            for part in parts:
                yield part

        return StreamingResponse(body(), status_code=status, media_type=media_type)

    app = Starlette(routes=[Route("/{path:path}", answer, methods=["POST"])])
    app.state.received = received
    app.state.body = b"".join(parts)
    return app


@asynccontextmanager
async def proxy_before(upstream: Starlette, settings: Settings, recorder):
    """A proxy in front of `upstream`, and a harness client for it. Recordings have landed
    once the block is left."""
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream))
    app = create_app(settings, client=client, recorder=recorder)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as harness:
            yield app, harness
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
    finally:
        await client.aclose()


class TestRecording:
    async def test_logprobs_a_harness_asked_for_are_stored_apart_from_its_response(
        self, proxy_client, mock_server, recorder, settle
    ):
        # A 16-character token: redacted inside the response JSON, the logprobs' own "token"
        # key would have matched the `token: <12+ chars>` secret pattern.
        entries = [
            {"id": 7, "token": "abcdefghijklmnop", "bytes": list(b"abcdefghijklmnop"), "logprob": -0.1,
             "top_logprobs": [{"id": 7, "token": "abcdefghijklmnop", "bytes": list(b"abcdefghijklmnop"), "logprob": -0.1}]},
        ]
        mock_server.script(MockTurn(content="abcdefghijklmnop", logprobs={"content": entries}))

        response = await proxy_client.post(CHAT, json=simple_request(logprobs=True, top_logprobs=1))

        assert response.json()["choices"][0]["logprobs"]["content"] == entries  # relayed untouched
        await settle()
        step = only_step(recorder.store)
        assert "logprobs" not in recorder.store.response(step)["choices"][0]
        (record,) = recorder.store.token_streams(step_id=step.id)
        assert (record.source, record.method, record.tokens) == ("recorded", "chat-logprobs", 1)
        stream = TokenStream.from_json(recorder.store.token_stream_data(record))
        assert stream.text() == "abcdefghijklmnop"

    async def test_non_streaming_call_is_recorded(self, proxy_client, mock_server, recorder, settle):
        mock_server.script(MockTurn(content="hello there", reasoning="a brief thought"))

        response = await proxy_client.post(
            CHAT, json=simple_request(temperature=0.7, seed=42)
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello there"

        await settle()

        step = only_step(recorder.store)
        assert step.model == "m"
        assert step.params["temperature"] == 0.7
        assert step.params["seed"] == 42
        assert step.finish_reason == "stop"
        assert step.reasoning_chars == len("a brief thought")
        assert step.usage["total_tokens"] == 18
        assert step.timings["total_ms"] >= 0

    async def test_request_reconstructs_json_equal_to_the_original(
        self, proxy_client, recorder, settle
    ):
        """§6.2's requirement: a step must rebuild the exact request it recorded.

        The request is never stored whole — messages live in the tree, everything else in
        the extras blob — so this is what proves the tree is lossless.
        """
        payload = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "go"},
            ],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "temperature": 0.2,
            "top_p": 0.95,
        }

        await proxy_client.post(CHAT, json=payload)
        await settle()

        step = only_step(recorder.store)
        assert recorder.store.rebuild_request(step) == payload

    async def test_stream_is_forwarded_byte_for_byte(
        self, proxy_client, mock_server, recorder, settle
    ):
        mock_server.script(
            MockTurn(
                content="hi there",
                reasoning="thinking about it",
                tool_calls=[{"name": "write_file", "arguments": {"path": "gen.py"}}],
            )
        )

        received = bytearray()
        async with proxy_client.stream(
            "POST",
            CHAT,
            json=simple_request(stream=True, stream_options={"include_usage": True}),
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            async for chunk in response.aiter_raw():
                received.extend(chunk)

        assert bytes(received).endswith(b"data: [DONE]\n\n")

        await settle()

        step = only_step(recorder.store)
        assert step.params["streamed"] is True
        # The tee: what we stored is exactly what the client saw.
        assert recorder.store.blobs.get(step.raw_response_blob) == bytes(received)

    async def test_streamed_response_is_reassembled(
        self, proxy_client, mock_server, recorder, settle
    ):
        mock_server.script(
            MockTurn(
                content="done",
                reasoning="short thought",
                tool_calls=[{"name": "write_file", "arguments": {"path": "gen.py"}}],
            )
        )

        async with proxy_client.stream(
            "POST",
            CHAT,
            json=simple_request(stream=True, stream_options={"include_usage": True}),
        ) as response:
            async for _ in response.aiter_raw():
                pass

        await settle()

        step = only_step(recorder.store)
        message = recorder.store.response(step)["choices"][0]["message"]
        assert message["content"] == "done"
        assert message["reasoning_content"] == "short thought"
        (call,) = message["tool_calls"]
        assert call["function"]["name"] == "write_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "gen.py"}
        assert step.usage["total_tokens"] == 18

    async def test_recording_is_detached_from_the_request(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        """Guards the failure that made tuningfork silently record nothing for a real harness.

        The OpenAI SDK closes the connection the moment it sees `[DONE]`, which cancels the
        request task. Recording used to be awaited inside the response generator, so it was
        cancelled too — and since `CancelledError` is not an `Exception`, nothing was even
        logged. The proxy looked healthy and the database stayed empty.

        The invariant asserted here is structural, because it is the one a test can check
        reliably: an in-process transport never produces the real disconnect, so a test
        that merely abandons the stream passes either way — verified by reinstating the old
        code, which that test did not catch.
        """
        mock_server.script(MockTurn(content="hello", chunk_delay=0.01))

        async with proxy_client.stream(
            "POST", CHAT, json=simple_request(stream=True)
        ) as response:
            async for _ in response.aiter_raw():
                pass

        assert proxy_app.state.recording_stats["spawned"] == 1, (
            "recording was awaited inline; a harness hanging up would cancel it"
        )
        await settle()
        assert len(all_steps(recorder.store)) == 1

    async def test_recording_still_lands_when_the_client_abandons_the_stream(
        self, proxy_client, mock_server, recorder, settle
    ):
        mock_server.script(MockTurn(content="a long answer to abandon", chunk_delay=0.01))

        async with proxy_client.stream(
            "POST", CHAT, json=simple_request(stream=True)
        ) as response:
            async for _ in response.aiter_raw():
                break  # hang up mid-stream

        await settle()

        assert len(all_steps(recorder.store)) == 1

    async def test_multi_step_conversation_forms_one_run(
        self, proxy_client, mock_server, recorder, settle
    ):
        """A second request extending the first's history continues the same run."""
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"))

        first = simple_request()
        await proxy_client.post(CHAT, json=first)
        second = {
            "model": "m",
            "messages": [
                *first["messages"],
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "and now?"},
            ],
        }
        await proxy_client.post(CHAT, json=second)
        await settle()

        runs = recorder.store.runs()
        assert len(runs) == 1
        assert runs[0].step_count == 2
        assert [s.idx for s in recorder.store.steps(runs[0].id)] == [0, 1]

    async def test_steps_are_stored_in_the_order_their_responses_finished(
        self, proxy_client, mock_server, recorder, settle, monkeypatch
    ):
        """Found as a flaky test: each recording is a task of its own, and when the first
        call's raw bytes took longer to store than the second's, the second step was stored
        first — and a conversation outside any launch split into two runs."""
        import time

        store_raw = recorder._store_raw
        slow = [True]

        def slow_first(*args) -> str:
            if slow.pop() if slow else False:
                time.sleep(0.3)
            return store_raw(*args)

        monkeypatch.setattr(recorder, "_store_raw", slow_first)
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"))
        first = simple_request()
        await proxy_client.post(CHAT, json=first)
        second = {
            "model": "m",
            "messages": [
                *first["messages"],
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "and now?"},
            ],
        }
        await proxy_client.post(CHAT, json=second)
        await settle()

        runs = recorder.store.runs()
        assert len(runs) == 1
        assert [s.idx for s in recorder.store.steps(runs[0].id)] == [0, 1]


class TestExperimentTraffic:
    async def test_is_forwarded_but_not_recorded_as_a_run(
        self, proxy_client, mock_server, recorder, settle
    ):
        """A resample repeats a recorded step's path; recorded, it would pollute that run."""
        mock_server.script(MockTurn(content="sampled"))

        response = await proxy_client.post(
            CHAT, json=simple_request(), headers={"x-tuningfork-experiment": "exp-1"}
        )
        await settle()

        assert response.json()["choices"][0]["message"]["content"] == "sampled"
        assert all_steps(recorder.store) == []

    async def test_the_marker_is_not_sent_upstream(self, proxy_client, mock_server):
        await proxy_client.post(
            CHAT, json=simple_request(), headers={"x-tuningfork-experiment": "exp-1"}
        )
        (_, _, headers) = mock_server.seen[-1]
        assert "x-tuningfork-experiment" not in headers

    async def test_the_pause_switch_stops_it(self, proxy_client, proxy_app, mock_server):
        """Resampling is the load that pausing exists to stop; it must not slip past."""
        proxy_app.state.gate.paused = True

        response = await proxy_client.post(
            CHAT, json=simple_request(), headers={"x-tuningfork-experiment": "exp-1"}
        )

        assert response.status_code == 503
        assert mock_server.requests == []


class TestPassthrough:
    async def test_models_route_is_relayed_and_not_recorded(self, proxy_client, recorder):
        response = await proxy_client.get("/v1/models")

        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "mock-model"
        assert all_steps(recorder.store) == []

    async def test_unknown_route_keeps_method_path_and_body(self, proxy_client, recorder):
        response = await proxy_client.post("/v1/embeddings", json={"input": "x"})

        body = response.json()
        assert body["path"] == "/v1/embeddings"
        assert json.loads(body["body"]) == {"input": "x"}
        assert all_steps(recorder.store) == []

    async def test_query_string_survives(self, proxy_client):
        response = await proxy_client.get("/v1/things?limit=3&cursor=abc")
        assert response.json()["query"] == "limit=3&cursor=abc"

    async def test_health_endpoint_is_not_proxied(self, proxy_client):
        from tuningfork import __version__

        body = (await proxy_client.get("/_tuningfork/health")).json()
        assert body["status"] == "ok"
        assert body["max_inflight"] == 1
        # What a client of another version checks, and why the server could not be read.
        assert body["version"] == __version__ and "server_problem" in body


class TestFailureHandling:
    async def test_upstream_error_status_is_relayed_and_recorded(
        self, proxy_client, mock_server, recorder, settle
    ):
        mock_server.script(MockTurn(status=429, error={"error": {"message": "slow down"}}))

        response = await proxy_client.post(CHAT, json=simple_request())

        assert response.status_code == 429
        assert response.json()["error"]["message"] == "slow down"
        await settle()
        step = only_step(recorder.store)
        assert step.status == 429
        assert step.error == "upstream_error"

    async def test_unreachable_upstream_reports_502(self, settings, recorder):
        class Refusing(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                raise httpx.ConnectError("connection refused")

        app = create_app(
            settings, client=httpx.AsyncClient(transport=Refusing()), recorder=recorder
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            response = await client.post(CHAT, json=simple_request())

        assert response.status_code == 502
        assert response.json()["error"]["type"] == "tuningfork_upstream_error"
        assert "ConnectError" in only_step(recorder.store).error

    async def test_storage_failure_does_not_break_the_response(
        self, proxy_client, proxy_app, mock_server
    ):
        """Recording is best-effort; the harness must still get its answer."""
        mock_server.script(MockTurn(content="still fine"))

        async def explode(**kwargs):
            raise RuntimeError("disk on fire")

        proxy_app.state.recorder.record = explode

        response = await proxy_client.post(CHAT, json=simple_request())

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "still fine"


class TestUpstreamFailures:
    """An error is relayed as it came, and recorded as the step's error: not behaviour."""

    async def test_an_anthropic_error_status_is_not_an_empty_answer(
        self, proxy_client, mock_server, recorder, settle
    ):
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
        mock_server.script(MockTurn(status=400, error=error))

        response = await proxy_client.post(MESSAGES, json=claude_code_request())
        await settle()

        assert response.status_code == 400
        step = only_step(recorder.store)
        assert step.error == "upstream_error"
        stored = recorder.store.response(step)
        assert stored["choices"] == [] and stored["error"]["message"] == "prompt is too long"
        # Kept out of every rate, as the same error on OpenAI chat is.
        assert outcome_for(stored).is_error

    async def test_a_stream_that_ends_in_an_error_event(self, settings, recorder):
        """Overloaded mid-answer: a 200, and part of an answer, that is not an answer."""
        upstream = upstream_sending(message_start(), text_block("I will now wr"), OVERLOADED)
        async with proxy_before(upstream, settings, recorder) as (_, harness):
            response = await harness.post(MESSAGES, json=claude_code_request())

        assert response.content == upstream.state.body
        step = only_step(recorder.store)
        assert step.error == "upstream_error: Overloaded"
        stored = recorder.store.response(step)
        assert stored["choices"][0]["message"]["content"] == "I will now wr"  # what did arrive
        assert stored["error"]["type"] == "overloaded_error"

    async def test_an_anthropic_stream_that_just_stops(self, settings, recorder):
        upstream = upstream_sending(message_start(), text_block("I will now wr"))
        async with proxy_before(upstream, settings, recorder) as (_, harness):
            await harness.post(MESSAGES, json=claude_code_request())

        assert only_step(recorder.store).error == CUT_OFF

    @pytest.mark.parametrize(
        "ending, error",
        [
            (b"", CUT_OFF),
            # Some servers never send [DONE]: a finish reason is enough.
            (openai_chunk({}, "stop"), None),
            (openai_chunk({}, "stop") + b"data: [DONE]\n\n", None),
        ],
        ids=["cut-off", "finish-without-done", "whole"],
    )
    async def test_an_openai_stream_fails_only_if_cut_off(self, settings, recorder, ending, error):
        upstream = upstream_sending(
            openai_chunk({"role": "assistant"}), openai_chunk({"content": "partial"}), ending
        )
        async with proxy_before(upstream, settings, recorder) as (_, harness):
            response = await harness.post(CHAT, json=simple_request(stream=True))

        assert response.content == upstream.state.body
        assert only_step(recorder.store).error == error


class TestUnreadableResponses:
    """Whatever the recording's copy of a stream makes of it, the harness's bytes flow."""

    async def test_a_stream_the_recording_cannot_read_still_reaches_the_harness(
        self, proxy_client, mock_server, recorder, settle, monkeypatch
    ):
        def unreadable(self, chunk):
            raise RuntimeError("a shape nobody expected")

        monkeypatch.setattr(ChunkAssembler, "add_chunk", unreadable)
        mock_server.script(MockTurn(content="the whole answer", chunk_delay=0.001))

        received = bytearray()
        async with proxy_client.stream("POST", CHAT, json=simple_request(stream=True)) as response:
            async for part in response.aiter_raw():
                received.extend(part)
        await settle()

        assert bytes(received).endswith(b"data: [DONE]\n\n")
        step = only_step(recorder.store)
        assert step.error.startswith("tuningfork could not read the response: RuntimeError")
        assert recorder.store.blobs.get(step.raw_response_blob) == bytes(received)  # kept raw

    async def test_nulls_in_an_anthropic_stream_are_read(self, settings, recorder):
        upstream = upstream_sending(message_start(usage=None), text_block("hello"), MESSAGE_DELTA, MESSAGE_STOP)
        async with proxy_before(upstream, settings, recorder) as (_, harness):
            response = await harness.post(MESSAGES, json=claude_code_request())

        assert response.content == upstream.state.body
        step = only_step(recorder.store)
        assert step.error is None
        assert recorder.store.response(step)["choices"][0]["message"]["content"] == "hello"


class TestMalformedCalls:
    """A model call tuningfork cannot read is relayed untouched, like any body it does not
    understand, and left to the server to refuse. It is not recorded."""

    @pytest.mark.parametrize("armed", [False, True], ids=["unarmed", "breakpoint-armed"])
    @pytest.mark.parametrize(
        "body",
        [
            b'{"model": "m", "messages": [{"role": "user", "content": "\xff"}]}',
            b'{"model": "m", "messages": 5}',
        ],
        ids=["not-utf8", "messages-not-a-list"],
    )
    async def test_it_is_relayed_untouched(self, settings, recorder, body, armed):
        refusal = b'{"error": {"message": "invalid request"}}'
        upstream = upstream_sending(refusal, status=400, media_type="application/json")
        async with proxy_before(upstream, settings, recorder) as (app, harness):
            if armed:
                app.state.engine.add(Breakpoint(phase="request"))
            response = await harness.post(CHAT, content=body, headers={"content-type": "application/json"})

        assert upstream.state.received == [body]
        assert (response.status_code, response.content) == (400, refusal)
        assert all_steps(recorder.store) == []


class TestRedaction:
    async def test_secrets_are_masked_in_storage_but_not_in_transit(
        self, proxy_client, mock_server, recorder, settle
    ):
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUV"
        await proxy_client.post(
            CHAT,
            json={
                "model": "m",
                "messages": [{"role": "user", "content": f"my key is {secret}"}],
            },
        )

        # Forwarded untouched: redaction must never alter what the model sees.
        assert secret.encode() in mock_server.raw_bodies[0]

        await settle()

        stored = recorder.store.rebuild_request(only_step(recorder.store))
        assert secret not in json.dumps(stored)
        assert "<redacted>" in stored["messages"][0]["content"]
        assert stored["model"] == "m"

    async def test_a_step_counts_the_masks_it_made(self, proxy_client, mock_server, recorder, settle):
        """Found in review: a context that read `<redacted>` was taken for a masked one, though a
        client's debug log writes the word itself: a step counts the masks storage made."""
        from tuningfork.ops.inspect import masked_warning

        logged = "DEBUG Request options: {'headers': {'Authorization': '<redacted>'}}"
        await proxy_client.post(CHAT, json={"model": "m", "messages": [{"role": "user", "content": logged}]})
        await proxy_client.post(CHAT, json={"model": "m", "messages": [
            {"role": "user", "content": logged + "\nmy key is sk-ABCDEFGHIJKLMNOPQRSTUV"}]})
        await settle()

        steps = [step for run in recorder.store.runs() for step in recorder.store.steps(run.id)]
        quiet, masked = sorted(steps, key=lambda step: step.params["masked"])
        assert quiet.params["masked"] == 0
        assert masked_warning(recorder.store.rebuild_request(quiet), quiet) is None
        assert masked.params["masked"] == 1
        assert f"{REDACTED} in 1 place," in masked_warning(recorder.store.rebuild_request(masked), masked)

    async def test_a_number_under_a_secrets_name_keeps_a_claude_code_session_whole(
        self, proxy_client, mock_server, recorder, settle
    ):
        """Found in the redaction audit: a long number under `page_token` in a tool's input was
        masked bare, which broke the request's JSON. The request was dropped, and so was every
        later one of the session, which repeats the call: each became a run of its own. A number
        under a password's name is one a password may be, and stays only because it is JSON's."""
        first = claude_code_request()
        call = {"type": "tool_use", "id": "t2", "name": "list_items",
                "input": {"page_token": 1729712345678, "pin_password": 20261023123456}}
        first["messages"] += [
            {"role": "assistant", "content": [call]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "3 items"}]},
        ]
        second = json.loads(json.dumps(first))
        second["messages"] += [
            {"role": "assistant", "content": [{"type": "text", "text": "Three."}]},
            {"role": "user", "content": [{"type": "text", "text": "And the next page?"}]},
        ]
        mock_server.script(MockTurn(content="Three."), MockTurn(content="None left."))
        await proxy_client.post(MESSAGES, json=first)
        await proxy_client.post(MESSAGES, json=second)
        await settle()

        runs = recorder.store.runs()
        assert len(runs) == 1 and runs[0].step_count == 2
        stored = recorder.store.rebuild_request(recorder.store.steps(runs[0].id)[1])
        assert stored["messages"][4]["content"][0]["input"] == call["input"]

    async def keep(self, settings, recorder, path, request, *chunks):
        """What the recorder keeps of an answer the upstream streams in these chunks: the bytes,
        the step, and the body as it came."""
        upstream = upstream_sending(*chunks)
        async with proxy_before(upstream, settings, recorder) as (_, harness):
            relayed = await harness.post(path, json=request)
        assert relayed.content == upstream.state.body  # the harness got its answer as it came
        step = only_step(recorder.store)
        return recorder.store.blobs.get(step.raw_response_blob), step, upstream.state.body

    async def test_a_key_streamed_in_pieces_is_not_kept_in_the_raw_bytes(self, settings, recorder):
        """Found in the 0.5.0 review: a model writes a key a token an event, and no pattern sees
        the pieces whole in the bytes kept beside the answer, which a fork read back and gave the
        harness again. What is kept reads as the masked answer."""
        pieces = ["my key:\n", "sk", "-a1B2c3", "D4e5F6g7", "H8i9J0", " is set"]
        kept, step, _ = await self.keep(settings, recorder, CHAT, simple_request(stream=True),
                                        *(openai_chunk({"content": piece}) for piece in pieces),
                                        openai_chunk({}, "stop"), b"data: [DONE]\n\n")
        assert reassemble(kept)["choices"][0]["message"]["content"] == f"my key:\n{REDACTED} is set"
        assert recorder.store.response(step)["choices"][0]["message"]["content"] == f"my key:\n{REDACTED} is set"
        assert step.params["raw_redacted"] == "rendered"
        assert REDACTED.encode() in kept  # so a fork past it is refused (`replay_problems`)

    async def test_a_key_partly_in_one_event_is_not_kept_either(self, settings, recorder):
        # Found in review: an event holding enough of the key to match was masked as far as it
        # held it, and the key's tail, in the events after it, was kept.
        pieces = ["my key is sk-a1B2c3D4e5F6g7H8i9", "J0K1L2M3N4", "O5P6Q7R8", " ok"]
        kept, _, _ = await self.keep(settings, recorder, CHAT, simple_request(stream=True),
                                     *(openai_chunk({"content": piece}) for piece in pieces),
                                     openai_chunk({}, "stop"), b"data: [DONE]\n\n")
        assert reassemble(kept)["choices"][0]["message"]["content"] == f"my key is {REDACTED} ok"

    async def test_a_key_in_reasoning_streamed_in_pieces(self, settings, recorder):
        pieces = ["I will use ", "sk", "-a1B2c3D4", "e5F6g7H8i9J0", " now"]
        kept, _, _ = await self.keep(settings, recorder, CHAT, simple_request(stream=True),
                                     *(openai_chunk({"reasoning_content": piece}) for piece in pieces),
                                     openai_chunk({"content": "done"}), openai_chunk({}, "stop"), b"data: [DONE]\n\n")
        message = reassemble(kept)["choices"][0]["message"]
        assert (message["reasoning_content"], message["content"]) == (f"I will use {REDACTED} now", "done")

    async def test_a_call_written_again_keeps_its_id_name_and_finish(self, settings, recorder):
        opening = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "write_file", "arguments": ""}}]}
        pieces = ['{"content":"KEY=', "sk", "-a1B2c3D4e5F6g7H8i9J0", '"}']
        kept, _, _ = await self.keep(
            settings, recorder, CHAT, simple_request(stream=True), openai_chunk(opening),
            *(openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}) for piece in pieces),
            openai_chunk({}, "tool_calls"), b"data: [DONE]\n\n",
        )
        choice = reassemble(kept)["choices"][0]
        (call,) = choice["message"]["tool_calls"]
        assert (call["id"], call["function"]["name"], choice["finish_reason"]) == ("call_1", "write_file", "tool_calls")
        assert json.loads(call["function"]["arguments"]) == {"content": f"KEY={REDACTED}"}

    async def test_logprobs_that_spell_a_key_are_not_kept(self, settings, recorder):
        # Found in review: a server sending several tokens an event, the key whole in one and
        # masked there, and the event's logprobs spelling it a token at a time.
        tokens = ["key: sk-a1B2c3", "D4e5F6g7H8i9J0"]
        entries = [{"id": n, "token": t, "bytes": list(t.encode()), "logprob": -0.1, "top_logprobs": []}
                   for n, t in enumerate(tokens)]
        chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [
            {"index": 0, "delta": {"content": "".join(tokens)}, "finish_reason": None, "logprobs": {"content": entries}}]}
        kept, step, _ = await self.keep(settings, recorder, CHAT, simple_request(stream=True, logprobs=True),
                                        f"data: {json.dumps(chunk)}\n\n", openai_chunk({}, "stop"), b"data: [DONE]\n\n")
        assert b"a1B2c3" not in kept and b"D4e5F6g7H8i9J0" not in kept
        assert "logprobs" not in reassemble(kept)["choices"][0]  # their bytes spell it too
        assert step.params["raw_redacted"] == "rendered"

    async def test_a_key_in_a_claude_code_calls_streamed_input(self, settings, recorder):
        partials = ['{"command":"echo ', "sk", "-a1B2c3D4", 'e5F6g7H8i9J0"}']
        kept, step, _ = await self.keep(settings, recorder, MESSAGES, claude_code_request(), message_start(),
                                        tool_block("Bash", partials), MESSAGE_DELTA, MESSAGE_STOP)
        (block,) = AnthropicMessages().parse(kept, True)["content"]
        assert (block["id"], block["name"], block["input"]) == ("toolu_1", "Bash", {"command": f"echo {REDACTED}"})
        arguments = recorder.store.response(step)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(arguments) == {"command": f"echo {REDACTED}"}

    async def test_a_key_partly_in_one_input_event(self, settings, recorder):
        partials = ['{"command":"export OPENAI_API_KEY=sk-proj-a1B2c3D4e5F6', 'g7H8i9J0K1L2M3"}']
        kept, _, _ = await self.keep(settings, recorder, MESSAGES, claude_code_request(), message_start(),
                                     tool_block("Bash", partials), MESSAGE_DELTA, MESSAGE_STOP)
        (block,) = AnthropicMessages().parse(kept, True)["content"]
        assert block["input"] == {"command": f"export OPENAI_API_KEY={REDACTED}"}

    async def test_a_key_in_thinking_streamed_in_pieces(self, settings, recorder):
        pieces = ["I will use ", "sk", "-a1B2c3D4", "e5F6g7H8i9J0", " now"]
        kept, _, _ = await self.keep(settings, recorder, MESSAGES, claude_code_request(), message_start(),
                                     thinking_block(pieces), text_block("done", index=1), MESSAGE_DELTA, MESSAGE_STOP)
        thinking, text = AnthropicMessages().parse(kept, True)["content"]
        assert (thinking["thinking"], text["text"]) == (f"I will use {REDACTED} now", "done")

    async def test_a_stream_without_a_secret_is_kept_byte_for_byte(self, settings, recorder):
        kept, step, body = await self.keep(
            settings, recorder, MESSAGES, claude_code_request(), message_start(),
            thinking_block(["Reading ", "the ", "file."]),
            tool_block("Read", ['{"file_path":', '"/home/user/project/data.csv"}'], index=1),
            MESSAGE_DELTA, MESSAGE_STOP,
        )
        assert kept == body and "raw_redacted" not in step.params

    async def test_a_key_within_one_event_is_masked_where_it_stands(self, settings, recorder):
        # Nothing to read whole that the events do not show: the server's own bytes are kept.
        kept, step, body = await self.keep(
            settings, recorder, CHAT, simple_request(stream=True),
            openai_chunk({"content": "my key: sk-a1B2c3D4e5F6g7H8i9J0"}), openai_chunk({"content": " is set"}),
            openai_chunk({}, "stop"), b"data: [DONE]\n\n",
        )
        assert kept == redact(body.decode()).encode() and "raw_redacted" not in step.params

    async def test_bytes_that_are_not_utf8_are_redacted_too(self, settings, recorder):
        # Found in review: a body with one stray byte was kept as it came, unredacted.
        stray = openai_chunk({"content": "caf"}).replace(b'"caf"', b'"caf\xff"')
        kept, _, _ = await self.keep(settings, recorder, CHAT, simple_request(stream=True), stray,
                                     openai_chunk({"content": " key sk-a1B2c3D4e5F6g7H8i9J0"}),
                                     openai_chunk({}, "stop"), b"data: [DONE]\n\n")
        assert b"a1B2c3D4e5F6" not in kept
        assert b'"caf\xff"' in kept  # the rest as it came

    class Wire:
        """A wire whose reading and writing a test sets."""

        def __init__(self, read=None, render_fails=False, parse_fails=False):
            self.read, self.render_fails, self.parse_fails = read, render_fails, parse_fails

        def parse(self, raw, event_stream):
            if self.parse_fails:
                raise ValueError("unreadable")
            return self.read if self.read is not None else reassemble(raw)

        def canonical_response(self, native, *, stream=None):
            return native

        def native_response(self, canonical, original):
            return canonical

        def render(self, native, *, request):
            if self.render_fails:
                raise ValueError("no")
            return b"rendered"

    PIECES = b"".join([openai_chunk({"content": "sk"}), openai_chunk({"content": "-a1B2c3D4e5F6g7H8i9J0"}),
                       openai_chunk({}, "stop")])
    #: What `PIECES` read as when they came: a key.
    ANSWER = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "sk-a1B2c3D4e5F6g7H8i9J0"}}]}

    def check(self, answer, wire):
        return stream_without_secrets(self.PIECES, answer, wire, simple_request(stream=True), DEFAULT_REDACTIONS)

    def test_bytes_that_cannot_be_written_again_are_not_kept(self):
        assert self.check(self.ANSWER, self.Wire(render_fails=True)) == (None, "dropped")

    def test_an_answer_without_a_secret_is_kept_unread(self):
        quiet = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}]}
        neither = self.Wire(parse_fails=True, render_fails=True)  # read back or written again, it would fail
        assert self.check(quiet, neither) == (self.PIECES, None)
        assert self.check(None, neither) == (self.PIECES, None)  # an answer the proxy could not read

    def test_an_answer_that_masking_leaves_unreadable_is_not_kept(self):
        # The default patterns mask inside strings only. A pattern of a person's own can mask a bare
        # number, which breaks the answer's JSON: written again, it would be an empty answer.
        patterns = (*DEFAULT_REDACTIONS, r"12345678901234567")
        answer = {**self.ANSWER, "session_token": 12345678901234567}
        kept = stream_without_secrets(self.PIECES, answer, self.Wire(), simple_request(stream=True), patterns)
        assert kept == (None, "dropped")

    def test_bytes_that_cannot_be_read_back_are_written_again(self):
        assert self.check(self.ANSWER, self.Wire(parse_fails=True)) == (b"rendered", "rendered")

class TestGate:
    async def test_a_harness_gone_before_its_answer_began_leaves_the_gate_free(
        self, proxy_app, mock_server, recorder, settle
    ):
        """Found in review: the gate's slot was given back only where the answer's body ended,
        and Starlette starts no body for a response whose start cannot be sent — a harness gone
        before its answer began — so the slot stayed taken, and every later model call waited
        at the gate until the proxy restarted. Driven as raw ASGI: an in-process client never
        hangs up for real."""
        mock_server.script(MockTurn(content="hello"))
        body = json.dumps(simple_request()).encode()
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": CHAT, "raw_path": CHAT.encode(), "query_string": b"",
            "root_path": "", "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8787),
            "headers": [(b"host", b"localhost:8787"), (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        }
        incoming = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return incoming.pop(0) if incoming else {"type": "http.disconnect"}

        async def send(message):
            raise OSError("the client went away")  # uvicorn's ClientDisconnected is one

        try:
            await proxy_app(scope, receive, send)
        except Exception:  # noqa: BLE001 - however Starlette passes the hang-up on
            pass
        assert proxy_app.state.gate.inflight == 0
        await settle()
        assert len(all_steps(recorder.store)) == 1  # and the exchange is recorded, as one abandoned mid-answer is

    async def test_pause_rejects_chat_but_still_relays_other_routes(
        self, proxy_client, proxy_app, recorder
    ):
        proxy_app.state.gate.paused = True

        chat = await proxy_client.post(CHAT, json=simple_request())
        models = await proxy_client.get("/v1/models")

        assert chat.status_code == 503
        assert chat.json()["error"]["type"] == "tuningfork_paused"
        assert models.status_code == 200
        assert all_steps(recorder.store) == []

    async def test_in_flight_cap_serializes_upstream_requests(
        self, settings, mock_server, recorder
    ):
        """The guard that stops a resample fan-out starving another tenant."""
        mock_server.script(*[MockTurn(content="x", stall=0.05) for _ in range(4)])
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(settings, client=upstream, recorder=recorder)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            await asyncio.gather(
                *(client.post(CHAT, json=simple_request()) for _ in range(4))
            )
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
        await upstream.aclose()

        assert mock_server.peak_inflight == 1
        assert len(all_steps(recorder.store)) == 4

    async def test_raw_completions_take_the_gate_and_the_pause_but_are_never_recorded(
        self, settings, mock_server, recorder
    ):
        """Token-level work sends /completion: model load like any other, so the gate and the
        pause apply — but there is no conversation in it to record."""
        mock_server.completion_stall = 0.05
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(settings, client=upstream, recorder=recorder)
        body = {"prompt": [2, 105], "n_predict": 2}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            answers = await asyncio.gather(*(client.post("/completion", json=body) for _ in range(3)))
            assert all(answer.status_code == 200 for answer in answers)
            assert mock_server.peak_inflight == 1
            # the tokenizer is not model load: it answers even while paused
            app.state.gate.paused = True
            refused = await client.post("/completion", json=body, headers={"x-tuningfork-experiment": "t"})
            tokenized = await client.post("/tokenize", json={"content": "hi"})
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
        await upstream.aclose()

        assert refused.status_code == 503 and refused.headers.get("x-tuningfork-paused") == "1"
        assert tokenized.status_code == 200
        assert all_steps(recorder.store) == []

    async def test_raising_the_cap_allows_concurrency(self, settings, mock_server, recorder):
        relaxed = Settings(
            upstream_base_url=settings.upstream_base_url,
            home=settings.home,
            max_inflight=3,
        )
        mock_server.script(*[MockTurn(content="x", stall=0.05) for _ in range(3)])
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(relaxed, client=upstream, recorder=recorder)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            await asyncio.gather(
                *(client.post(CHAT, json=simple_request()) for _ in range(3))
            )
        await upstream.aclose()

        assert mock_server.peak_inflight > 1


class TestUpstreamUrl:
    @pytest.mark.parametrize(
        ("base", "path", "expected"),
        [
            ("http://h:8081", "/v1/chat/completions", "http://h:8081/v1/chat/completions"),
            ("http://h:8081/", "/v1/models", "http://h:8081/v1/models"),
            # A base URL pasted straight from a harness config already ends in /v1.
            ("http://h:8081/v1", "/v1/chat/completions", "http://h:8081/v1/chat/completions"),
            ("http://h:8081/v1", "/healthz", "http://h:8081/v1/healthz"),
        ],
    )
    def test_join(self, base, path, expected):
        assert upstream_url(base, path) == expected

    def test_query_is_appended(self):
        assert upstream_url("http://h", "/v1/models", "a=1") == "http://h/v1/models?a=1"


class TestOutputCap:
    """`serve --max-output`: every call a harness makes through the proxy is held to the cap.
    The step is what the model was sent; what the harness asked is kept beside it, so the
    harness's own request — what a fork's replay matches against — is still exact."""

    @pytest.mark.parametrize("asked", [{"max_tokens": 65536}, {}], ids=["more", "no-limit"])
    async def test_a_call_asking_for_more_is_sent_with_the_cap(self, settings, recorder, asked):
        upstream = upstream_sending(
            openai_chunk({"role": "assistant"}), openai_chunk({"content": "again and again"}),
            openai_chunk({}, "length"), b"data: [DONE]\n\n",
        )
        async with proxy_before(upstream, dataclasses.replace(settings, max_output=100), recorder) as (_, harness):
            response = await harness.post(CHAT, json=simple_request(**asked, stream=True))
            health = (await harness.get("/_tuningfork/health")).json()

        assert response.content == upstream.state.body  # the answer reaches the harness untouched
        (sent,) = upstream.state.received
        assert json.loads(sent)["max_tokens"] == 100
        assert health["max_output"] == 100
        step = only_step(recorder.store)
        assert step.finish_reason == "length" and step.params["max_tokens"] == 100
        assert step.params["output_cap"] == {"cap": 100, "asked": asked, "written": ["max_tokens"]}
        assert recorder.store.rebuild_request(step)["max_tokens"] == 100  # what the model was sent
        assert recorder.store.harness_request(step) == simple_request(**asked, stream=True)

    async def test_a_call_within_the_cap_is_relayed_as_sent(self, settings, recorder):
        upstream = upstream_sending(openai_chunk({"content": "hi"}, "stop"), b"data: [DONE]\n\n")
        body = json.dumps(simple_request(max_tokens=64, stream=True)).encode()
        async with proxy_before(upstream, dataclasses.replace(settings, max_output=100), recorder) as (_, harness):
            await harness.post(CHAT, content=body, headers={"content-type": "application/json"})

        assert upstream.state.received == [body]  # byte for byte
        assert "output_cap" not in only_step(recorder.store).params

    async def test_an_experiments_call_keeps_its_own_limit(self, settings, recorder):
        """Experiment traffic is tuningfork's own measurement, recorded by its experiment: the
        cap is for harnesses, and would change a sample without its record knowing."""
        upstream = upstream_sending(openai_chunk({"content": "hi"}, "stop"), b"data: [DONE]\n\n")
        async with proxy_before(upstream, dataclasses.replace(settings, max_output=100), recorder) as (_, harness):
            await harness.post(CHAT, json=simple_request(max_tokens=65536), headers={EXPERIMENT_HEADER: "e1"})

        assert json.loads(upstream.state.received[0])["max_tokens"] == 65536
