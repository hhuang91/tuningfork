"""Model calls held at breakpoints, end to end through the proxy.

A held call blocks its client, so each test starts the client call as a task, waits for the
hold to appear, decides it the way the UI or CLI would, then checks what the harness got
and what was recorded.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tuningfork.breakpoints import Breakpoint
from tuningfork.config import Settings
from tuningfork.proxy.app import create_app
from tuningfork.proxy.held import HUNG_UP
from tuningfork.proxy.sse import ChunkAssembler, reassemble
from tuningfork.wire.anthropic import AnthropicMessages
from tuningfork.wire.openai import OpenAIChat
from tests.helpers import all_steps, only_step, sse_payloads, until
from tests.mock_server import MockTurn
from tests.test_proxy import proxy_before, upstream_sending
from tests.test_wire import OVERLOADED, claude_code_request, message_start, text_block

CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"

WRITE_REPORT = MockTurn(
    reasoning="I will write the report.",
    tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": '{"count": 4}'}}],
)


def request(content: str = "hi", **extra) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}


def streaming(content: str = "hi") -> dict:
    return request(content, stream=True, stream_options={"include_usage": True})


@pytest.fixture
def engine(proxy_app):
    return proxy_app.state.engine


async def first_hold(engine):
    return (await until(lambda: engine.holds()))[0]


class TestBeforeTheModel:
    async def test_a_held_request_reaches_no_one_until_continued(
        self, proxy_client, engine, mock_server
    ):
        engine.add(Breakpoint(phase="request"))
        mock_server.script(MockTurn(content="answered"))

        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        hold = await first_hold(engine)

        assert mock_server.requests == []  # the model has not been asked
        assert hold.phase == "request"
        engine.resolve(hold.id, "continue")

        response = await asyncio.wait_for(call, 3)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "answered"

    async def test_an_edited_request_is_what_the_model_sees(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        engine.add(Breakpoint(phase="request"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("original wording")))
        hold = await first_hold(engine)

        edited = json.loads(json.dumps(hold.request))
        edited["messages"][0]["content"] = "edited wording"
        engine.resolve(hold.id, "continue", request=edited)
        await asyncio.wait_for(call, 3)
        await settle()

        assert mock_server.requests[0]["messages"][0]["content"] == "edited wording"
        step = only_step(recorder.store)
        assert recorder.store.rebuild_request(step)["messages"][0]["content"] == "edited wording"
        assert recorder.store.harness_request(step)["messages"][0]["content"] == "original wording"
        assert step.edits["request"]["messages"] == [0]

    async def test_an_edit_does_not_split_the_run(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        """The harness never saw the edit; its next request extends its own history."""
        engine.add(Breakpoint(phase="request", once=True))
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("go")))
        hold = await first_hold(engine)
        edited = json.loads(json.dumps(hold.request))
        edited["messages"][0]["content"] = "go, carefully"
        engine.resolve(hold.id, "continue", request=edited)
        await asyncio.wait_for(call, 3)
        await settle()

        follow_up = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "and then?"},
            ],
        }
        await proxy_client.post(CHAT, json=follow_up)
        await settle()

        (run,) = recorder.store.runs()
        assert run.step_count == 2

    async def test_the_harness_keeps_its_transport_even_if_an_edit_changes_it(
        self, proxy_client, engine, mock_server
    ):
        engine.add(Breakpoint(phase="request"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming()))
        hold = await first_hold(engine)
        edited = {**hold.request, "stream": False}
        engine.resolve(hold.id, "continue", request=edited)
        response = await asyncio.wait_for(call, 3)

        assert mock_server.requests[0]["stream"] is True
        assert sse_payloads(response.content)[-1] == "[DONE]"

    async def test_dropping_before_the_model(self, proxy_client, engine, mock_server, recorder, settle):
        engine.add(Breakpoint(phase="request"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        engine.resolve((await first_hold(engine)).id, "drop")

        response = await asyncio.wait_for(call, 3)
        await settle()

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "tuningfork_dropped"
        assert mock_server.requests == []
        assert all_steps(recorder.store) == []  # nothing happened, so nothing is recorded


class TestTheCapHoldsAnEdit:
    """`serve --max-output`: a held call is shown as it will be sent, at the cap, and an edit
    asking for more — or for no limit — is sent at the cap too, the step saying what it asked.
    What the harness asked stays its own: a fork's replay matches against it."""

    @pytest.fixture
    def settings(self, tmp_path):
        return Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home", max_output=100)

    @pytest.mark.parametrize("change", [{"max_tokens": 1000}, {"n_predict": 1000}, None],
                             ids=["more", "another-limit", "no-limit"])
    async def test_an_edit_asking_for_more_is_sent_at_the_cap(
        self, proxy_client, engine, mock_server, recorder, settle, change
    ):
        engine.add(Breakpoint(phase="request"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("go", max_tokens=65536)))
        hold = await first_hold(engine)
        assert hold.request["max_tokens"] == 100

        edited = json.loads(json.dumps(hold.request))
        edited["messages"][0]["content"] = "go, carefully"
        if change is None:
            edited.pop("max_tokens")
        else:
            edited.update(change)
        engine.resolve(hold.id, "continue", request=edited)
        await asyncio.wait_for(call, 3)
        await settle()

        (sent,) = mock_server.requests
        assert sent["messages"][0]["content"] == "go, carefully"
        assert sent["max_tokens"] == 100 and sent.get("n_predict", 100) == 100
        step = only_step(recorder.store)
        assert step.edits["output_cap"]["cap"] == 100
        assert recorder.store.harness_request(step) == request("go", max_tokens=65536)


class TestAfterTheModel:
    async def test_a_tool_call_is_held_before_the_harness_sees_it(
        self, proxy_client, engine, mock_server
    ):
        engine.add(Breakpoint(phase="response", tool="write_file", path_glob="*.json"))
        mock_server.script(WRITE_REPORT)

        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming()))
        hold = await first_hold(engine)

        assert hold.phase == "response"
        (tool_call,) = hold.response["choices"][0]["message"]["tool_calls"]
        assert tool_call["function"]["name"] == "write_file"
        assert not call.done()  # the harness is still waiting

        engine.resolve(hold.id, "continue")
        response = await asyncio.wait_for(call, 3)
        message = reassemble(response.content)["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "write_file"

    async def test_an_unedited_response_is_delivered_byte_for_byte(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        engine.add(Breakpoint(phase="response", tool="write_file"))
        mock_server.script(WRITE_REPORT)
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming()))
        engine.resolve((await first_hold(engine)).id, "continue")
        response = await asyncio.wait_for(call, 3)
        await settle()

        step = only_step(recorder.store)
        assert response.content == recorder.store.blobs.get(step.raw_response_blob)
        assert not step.edited

    async def test_an_edited_response_is_what_the_harness_receives(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        engine.add(Breakpoint(phase="response", tool="write_file"))
        mock_server.script(WRITE_REPORT)
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming()))
        hold = await first_hold(engine)

        edited = json.loads(json.dumps(hold.response))
        function = edited["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps(
            {"path": "report.json", "content": '{"row_count": 4, "amount_total": 61.5}'}
        )
        engine.resolve(hold.id, "continue", response=edited)
        response = await asyncio.wait_for(call, 3)
        await settle()

        delivered = reassemble(response.content)["choices"][0]["message"]["tool_calls"][0]
        assert "row_count" in delivered["function"]["arguments"]

        step = only_step(recorder.store)
        # The model's own answer stays the step's behaviour; the edit is recorded beside it.
        assert "count" in recorder.store.response(step)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert "row_count" in json.dumps(recorder.store.delivered_response(step))
        assert step.edits["response"]["message_fields"] == ["tool_calls"]

    async def test_an_edited_non_streaming_response(self, proxy_client, engine, mock_server):
        engine.add(Breakpoint(phase="response", output_regex="original"))
        mock_server.script(MockTurn(content="original text"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        hold = await first_hold(engine)
        edited = json.loads(json.dumps(hold.response))
        edited["choices"][0]["message"]["content"] = "replacement text"
        engine.resolve(hold.id, "continue", response=edited)

        response = await asyncio.wait_for(call, 3)
        assert response.json()["choices"][0]["message"]["content"] == "replacement text"

    async def test_dropping_after_the_model_records_what_it_said(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        engine.add(Breakpoint(phase="response", tool="write_file"))
        mock_server.script(WRITE_REPORT)
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming()))
        engine.resolve((await first_hold(engine)).id, "drop")
        response = await asyncio.wait_for(call, 3)
        await settle()

        (payload,) = sse_payloads(response.content)
        assert json.loads(payload)["error"]["type"] == "tuningfork_dropped"
        step = only_step(recorder.store)
        assert step.edits["response"] == {"dropped": True}
        assert step.error == "dropped at a tuningfork breakpoint"

    async def test_calls_no_breakpoint_could_catch_stream_live(
        self, proxy_client, engine, mock_server
    ):
        """No behaviour change unless a breakpoint could apply to this very call."""
        engine.add(Breakpoint(phase="response", tool="write_file", step_min=5))
        mock_server.script(MockTurn(content="streamed"))

        response = await proxy_client.post(CHAT, json=streaming())

        assert "x-tuningfork" not in response.headers
        assert engine.holds() == []


class TestStepping:
    async def test_step_from_before_to_after_the_same_call(self, proxy_client, engine, mock_server):
        engine.add(Breakpoint(phase="request", once=True))
        mock_server.script(MockTurn(content="answer"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))

        engine.resolve((await first_hold(engine)).id, "step")
        second = await first_hold(engine)
        assert second.phase == "response" and "stepped" in second.reason
        assert second.response["choices"][0]["message"]["content"] == "answer"

        engine.resolve(second.id, "continue")
        assert (await asyncio.wait_for(call, 3)).status_code == 200

    async def test_a_step_after_an_edit_shows_the_edited_call(
        self, proxy_client, engine, mock_server, settle
    ):
        """Found in review: the hold after the model still showed the request as the
        harness sent it, so a person reading it — and any condition on it — saw a call the
        model never answered."""
        engine.add(Breakpoint(phase="request", once=True))
        mock_server.script(MockTurn(content="answer"), MockTurn(content="later"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("original wording")))
        hold = await first_hold(engine)

        edited = json.loads(json.dumps(hold.request))
        edited["messages"][0]["content"] = "edited wording"
        engine.resolve(hold.id, "step", request=edited)
        after = await first_hold(engine)

        assert after.phase == "response"
        assert after.request["messages"][0]["content"] == "edited wording"
        assert after.view[0]["content"] == "edited wording"
        # Stepping on still stops the harness's next call, which extends what it sent.
        engine.resolve(after.id, "step")
        await asyncio.wait_for(call, 3)
        await settle()
        follow_up = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "original wording"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "next"},
            ],
        }
        call = asyncio.create_task(proxy_client.post(CHAT, json=follow_up))
        nxt = await first_hold(engine)
        assert nxt.phase == "request" and "stepped" in nxt.reason
        engine.resolve(nxt.id, "continue")
        await asyncio.wait_for(call, 3)

    async def test_step_from_after_one_call_to_before_the_next(
        self, proxy_client, engine, mock_server, settle
    ):
        engine.add(Breakpoint(phase="response", once=True))
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("go")))
        engine.resolve((await first_hold(engine)).id, "step")
        await asyncio.wait_for(call, 3)
        await settle()

        follow_up = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "next"},
            ],
        }
        call = asyncio.create_task(proxy_client.post(CHAT, json=follow_up))
        hold = await first_hold(engine)
        assert hold.phase == "request" and "stepped" in hold.reason
        engine.resolve(hold.id, "continue")
        await asyncio.wait_for(call, 3)


class TestHarnessSafety:
    async def test_a_long_hold_sends_keepalives_that_clients_ignore(self, settings, mock_server, recorder):
        quick = Settings(
            upstream_base_url=settings.upstream_base_url, home=settings.home, keepalive_interval=0.02
        )
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(quick, client=upstream, recorder=recorder)
        app.state.engine.add(Breakpoint(phase="request"))
        mock_server.script(MockTurn(content="after the wait"))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            call = asyncio.create_task(client.post(CHAT, json=streaming()))
            hold = (await until(lambda: app.state.engine.holds()))[0]
            await asyncio.sleep(0.15)
            app.state.engine.resolve(hold.id, "continue")
            response = await asyncio.wait_for(call, 3)
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
        await upstream.aclose()

        assert b": tuningfork: call held at a breakpoint" in response.content
        assert reassemble(response.content)["choices"][0]["message"]["content"] == "after the wait"

    async def test_a_held_call_does_not_occupy_the_gate(self, settings, mock_server, recorder):
        """Someone reading a held call must not starve other traffic on a shared server."""
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        app = create_app(settings, client=upstream, recorder=recorder)  # max_inflight=1
        app.state.engine.add(Breakpoint(phase="request", context_regex="hold me"))
        mock_server.script(MockTurn(content="free"), MockTurn(content="held one"))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787"
        ) as client:
            held = asyncio.create_task(client.post(CHAT, json=request("hold me")))
            hold = (await until(lambda: app.state.engine.holds()))[0]
            other = await asyncio.wait_for(client.post(CHAT, json=request("pass")), 3)
            assert other.json()["choices"][0]["message"]["content"] == "free"
            app.state.engine.resolve(hold.id, "continue")
            await asyncio.wait_for(held, 3)
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
        await upstream.aclose()

    async def test_experiment_traffic_is_never_held(self, proxy_client, engine, mock_server):
        engine.add(Breakpoint(phase="request"))
        response = await asyncio.wait_for(
            proxy_client.post(CHAT, json=request(), headers={"x-tuningfork-experiment": "e1"}), 3
        )
        assert response.status_code == 200
        assert engine.holds() == []

    async def test_a_harness_that_hangs_up_abandons_its_hold(self, proxy_client, engine):
        engine.add(Breakpoint(phase="request"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        hold = await first_hold(engine)

        call.cancel()
        await until(lambda: not engine.holds())

        assert engine.get_hold(hold.id).abandoned

    @pytest.mark.parametrize("stream", [True, False], ids=["streamed", "json"])
    async def test_hanging_up_on_a_held_response_still_records_the_step(
        self, proxy_client, engine, mock_server, recorder, settle, stream
    ):
        """The model did answer. Without the step, the run would be missing one it took."""
        engine.add(Breakpoint(phase="response", tool="write_file"))
        mock_server.script(WRITE_REPORT)
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming() if stream else request()))
        hold = await first_hold(engine)

        call.cancel()
        await until(lambda: not engine.holds())
        await settle()

        assert engine.get_hold(hold.id).abandoned
        step = only_step(recorder.store)
        assert step.error == HUNG_UP
        (made,) = recorder.store.response(step)["choices"][0]["message"]["tool_calls"]
        assert made["function"]["name"] == "write_file"

    @pytest.mark.parametrize("phase", ["request", "response"])
    async def test_a_dropped_call_is_not_retried_by_the_client(
        self, proxy_client, engine, mock_server, phase
    ):
        """The OpenAI and Anthropic SDKs resend a 409 unasked. The resent call would be a new
        one to tuningfork, past a `once` breakpoint already spent: the drop would do nothing."""
        engine.add(Breakpoint(phase=phase, once=True))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        engine.resolve((await first_hold(engine)).id, "drop")
        response = await asyncio.wait_for(call, 3)

        assert response.status_code == 409
        assert response.headers["x-should-retry"] == "false"


class TestResponsesThatGoWrong:
    """Once the model has answered, its answer is recorded, whatever happens next."""

    async def test_arguments_edited_into_an_object_reach_claude_code(
        self, proxy_client, engine, mock_server, recorder, settle
    ):
        engine.add(Breakpoint(phase="response", tool="Write"))
        mock_server.script(
            MockTurn(tool_calls=[{"name": "Write", "arguments": {"file_path": "a.json", "content": "{}"}}])
        )
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=claude_code_request()))
        hold = await first_hold(engine)
        edited = json.loads(json.dumps(hold.response))
        # An edit written by hand may give the arguments as the object itself.
        function = edited["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = {"file_path": "b.json", "content": "{}"}
        engine.resolve(hold.id, "continue", response=edited)
        response = await asyncio.wait_for(call, 3)
        await settle()

        received = AnthropicMessages().parse(response.content, event_stream=True)
        (write,) = [block for block in received["content"] if block["type"] == "tool_use"]
        assert write["input"] == {"file_path": "b.json", "content": "{}"}
        model_wrote = recorder.store.response(only_step(recorder.store))["choices"][0]["message"]
        assert json.loads(model_wrote["tool_calls"][0]["function"]["arguments"])["file_path"] == "a.json"

    @pytest.mark.parametrize("stream", [True, False], ids=["streamed", "json"])
    async def test_an_edit_that_cannot_be_delivered(
        self, proxy_client, engine, mock_server, recorder, settle, monkeypatch, stream
    ):
        def fails(self, native, *, request):
            raise RuntimeError("cannot render this")

        monkeypatch.setattr(OpenAIChat, "render", fails)
        engine.add(Breakpoint(phase="response", output_regex="original"))
        mock_server.script(MockTurn(content="original text"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=streaming() if stream else request()))
        hold = await first_hold(engine)
        edited = json.loads(json.dumps(hold.response))
        edited["choices"][0]["message"]["content"] = "replacement text"
        engine.resolve(hold.id, "continue", response=edited)
        response = await asyncio.wait_for(call, 3)
        await settle()

        # An error — never the answer as it came, which the person chose to change.
        assert b"original text" not in response.content
        if stream:
            (payload,) = sse_payloads(response.content)
            error = json.loads(payload)["error"]
        else:
            assert (response.status_code, response.headers["x-should-retry"]) == (500, "false")
            error = response.json()["error"]
        assert "could not deliver the edited response" in error["message"]

        step = only_step(recorder.store)
        assert recorder.store.response(step)["choices"][0]["message"]["content"] == "original text"
        assert step.error.startswith("tuningfork could not deliver the edited response")
        assert step.edits["response"]["message_fields"] == ["content"]  # what was attempted

    async def test_a_response_that_cannot_be_read_goes_on_as_it_came(
        self, proxy_client, engine, mock_server, recorder, settle, monkeypatch
    ):
        def unreadable(self, chunk):
            raise RuntimeError("a shape nobody expected")

        monkeypatch.setattr(ChunkAssembler, "add_chunk", unreadable)
        engine.add(Breakpoint(phase="response", tool="write_file"))
        mock_server.script(WRITE_REPORT)

        response = await asyncio.wait_for(proxy_client.post(CHAT, json=streaming()), 3)
        await settle()

        step = only_step(recorder.store)
        assert response.content == recorder.store.blobs.get(step.raw_response_blob)
        assert step.error.startswith("tuningfork could not read the response: RuntimeError")
        assert engine.holds() == []  # nothing to hold without reading it

    async def test_a_stream_that_errored_is_recorded_as_a_failure(self, settings, recorder):
        upstream = upstream_sending(message_start(), text_block("I will now wr"), OVERLOADED)
        async with proxy_before(upstream, settings, recorder) as (app, harness):
            # Buffers the response on the breakpoint path, then never matches it.
            app.state.engine.add(Breakpoint(phase="response", tool="Write"))
            response = await harness.post(MESSAGES, json=claude_code_request())

        assert response.headers["x-tuningfork"] == "breakpoint-path"
        assert response.content == upstream.state.body
        assert only_step(recorder.store).error == "upstream_error: Overloaded"

    async def test_an_answer_sent_at_once_for_a_stream_is_read_as_json(self, settings, recorder):
        answer = {"id": "x", "object": "chat.completion", "model": "m", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "sent whole"}, "finish_reason": "stop"}]}
        upstream = upstream_sending(json.dumps(answer), media_type="text/plain")
        async with proxy_before(upstream, settings, recorder) as (app, harness):
            app.state.engine.add(Breakpoint(phase="response", tool="Write"))
            response = await harness.post(CHAT, json=streaming())

        assert response.headers["x-tuningfork"] == "breakpoint-path"
        step = only_step(recorder.store)
        assert step.error is None and recorder.store.response(step)["choices"][0]["message"]["content"] == "sent whole"

    async def test_a_stream_asked_for_is_read_as_one_whatever_its_label(self, settings, recorder):
        """As on the ordinary path: a stream labelled `text/plain` is read as a stream, and a key it
        sent in pieces is not kept in its raw bytes."""
        from tests.test_proxy import openai_chunk

        pieces = ["my key: ", "sk", "-a1B2c3", "D4e5F6g7", "H8i9J0", " is set"]
        upstream = upstream_sending(*(openai_chunk({"content": piece}) for piece in pieces),
                                    openai_chunk({}, "stop"), b"data: [DONE]" + bytes([10, 10]), media_type="text/plain")
        async with proxy_before(upstream, settings, recorder) as (app, harness):
            # Buffers the response on the breakpoint path, then never matches it.
            app.state.engine.add(Breakpoint(phase="response", tool="Write"))
            response = await harness.post(CHAT, json=streaming())

        assert response.headers["x-tuningfork"] == "breakpoint-path"
        step = only_step(recorder.store)
        assert step.error is None and step.params["raw_redacted"] == "rendered"
        assert b"D4e5F6g7" not in recorder.store.blobs.get(step.raw_response_blob)


class TestMalformedCalls:
    @pytest.mark.parametrize(
        "definition",
        [{"phase": "response", "tool": "Write"}, {"phase": "request", "context_regex": "go"}],
        ids=["response", "request"],
    )
    async def test_a_message_that_is_not_an_object_does_not_fail_the_call(
        self, proxy_client, engine, mock_server, recorder, settle, definition
    ):
        """Relayed for the server to refuse or not, as it would be with no breakpoint."""
        engine.add(Breakpoint.from_dict(definition))
        payload = {"model": "m", "max_tokens": 5, "messages": ["stray", {"role": "user", "content": "go"}]}
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=payload))
        if definition["phase"] == "request":
            engine.resolve((await first_hold(engine)).id, "continue")
        response = await asyncio.wait_for(call, 3)
        await settle()

        assert response.status_code == 200
        assert mock_server.requests[0]["messages"][0] == "stray"
        assert recorder.store.rebuild_request(only_step(recorder.store)) == payload
