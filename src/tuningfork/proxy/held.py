"""Model calls that pass through a breakpoint.

A call only comes down this path when a breakpoint could actually apply to it; everything
else takes the proxy's ordinary streaming path, unchanged. That is the promise — no
behaviour change unless a breakpoint is armed — kept structurally rather than by care.

On this path the response is buffered rather than streamed, because a response-phase
breakpoint has to see the whole of the model's answer before the harness sees any of it.

Two properties matter for a harness on the other end:

- **Holds never occupy the gate.** A request is held before it takes an upstream slot, and
  a response after the slot is released, so a person taking their time at a breakpoint
  cannot starve other traffic on a shared server.
- **A held streaming call is kept alive.** Its response starts immediately, and an SSE
  comment line follows every `keepalive_interval` seconds until the body is ready. Clients
  ignore comment lines, but idle timeouts — in the client, or in anything in between — see
  traffic. The price is that such a call always answers 200 at the HTTP level, with any
  failure delivered as an SSE `error` event instead, which the OpenAI SDK raises as an
  `APIError`.

Once the model has answered, its answer is recorded whatever happens next — the harness
hanging up mid-hold, or an edit that cannot be delivered: it is still that step's behaviour.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from tuningfork.breakpoints import BreakpointEngine, Decision, ExchangeContext, Hold
from tuningfork.config import Settings
from tuningfork.proxy.recorder import Recorder, Timings
from tuningfork.proxy.sse import SSEParser, json_body, read_as, unreadable
from tuningfork.wire import Assembler, WireFormat
from tuningfork.wire.base import cap_output

#: Fields describing the harness's connection rather than the call's content. An edited
#: request keeps the harness's values, because the harness is still the one listening.
TRANSPORT_FIELDS = ("stream", "stream_options")

#: Step errors for a model answer the harness never received.
DROPPED = "dropped at a tuningfork breakpoint"
HUNG_UP = "the harness hung up while the response was held"

logger = logging.getLogger("tuningfork.proxy")


@dataclass(slots=True)
class Delivery:
    """What the harness receives at the end of a held exchange."""

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @classmethod
    def failure(
        cls, wire: WireFormat, status: int, message: str, kind: str, *, retry: bool = True
    ) -> Delivery:
        """An error in the shape the harness's client library understands.

        `retry=False` for a failure that is a decision, not a fault. The official OpenAI and
        Anthropic SDKs resend a 409 or a 5xx on their own — silently, as the same request —
        and a resent call is a new call to tuningfork: past a `once` breakpoint already
        spent, it would reach the model after all. Both SDKs obey `x-should-retry` before
        their status rules.
        """
        headers = {} if retry else {"x-should-retry": "false"}
        return cls(status=status, headers=headers, error=wire.error(status, message, kind))


def describe_edit(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A compact record of what an edit changed, for the step and the UI."""
    fields = sorted(key for key in {*before, *after} if before.get(key) != after.get(key))
    summary: dict[str, Any] = {"fields": fields}
    if "messages" in fields:
        old, new = before.get("messages") or [], after.get("messages") or []
        summary["messages"] = [
            index
            for index in range(max(len(old), len(new)))
            if index >= len(old) or index >= len(new) or old[index] != new[index]
        ]
    if "choices" in fields:
        old_message = ((before.get("choices") or [{}])[0].get("message")) or {}
        new_message = ((after.get("choices") or [{}])[0].get("message")) or {}
        summary["message_fields"] = sorted(
            key
            for key in {*old_message, *new_message}
            if old_message.get(key) != new_message.get(key)
        )
    extras_changed = [key for key in fields if key not in ("messages", "choices")]
    if extras_changed and "messages" in before:
        # Needed to rebuild the harness's own request later; see Store.harness_request.
        summary["original_extras"] = {k: v for k, v in before.items() if k != "messages"}
    return summary


def _parse(
    wire: WireFormat, raw: bytes, event_stream: bool
) -> tuple[dict[str, Any] | None, Assembler | None]:
    """A whole response in its own format, and for a stream the assembler that read it —
    which knows whether the stream ended the way a whole answer does."""
    if not event_stream:
        return wire.parse(raw, False), None
    assembler = wire.assembler()
    for event in SSEParser().feed(raw):
        assembler.add_event(event)
    return (assembler.result() if assembler.chunk_count else None), assembler


class BreakpointPath:
    def __init__(
        self,
        *,
        settings: Settings,
        gate: Any,
        engine: BreakpointEngine,
        recorder: Recorder,
        spawn: Callable[[Awaitable[Any]], None],
        settle: Callable[[], Awaitable[None]],
        forward_headers: Callable[..., dict[str, str]],
    ) -> None:
        self.settings = settings
        self.gate = gate
        self.engine = engine
        self.recorder = recorder
        self.spawn = spawn
        self.settle = settle
        self.forward_headers = forward_headers

    async def maybe_handle(
        self,
        request: Request,
        *,
        wire: WireFormat,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        body: bytes,
        payload: dict[str, Any],
        assignment: Any = None,
        release: Callable[[Any], None] | None = None,
        harness_payload: dict[str, Any] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> Response | None:
        """Take this call down the breakpoint path if a breakpoint could apply, else None.

        A launched call arrives with its run and step already `assignment`ed; `release`
        gives its step number back if it is dropped before it is recorded. In a fork testing
        a wording, `payload` is what the model will see and `harness_payload` what the
        harness sent; `notes` are the fork's to record with the step.
        """
        if assignment is None:
            # A recording still in flight could change which run and step this call
            # belongs to — and "step >= 3" has to be decided now, before it is recorded.
            await self.settle()
        ctx = await asyncio.to_thread(
            self.engine.context_for, harness_payload or payload, wire, assignment
        )
        if harness_payload is not None:
            # Conditions and a person read what the model will see; the path — what the
            # harness's next call extends — stays the harness's.
            ctx.request, ctx.view = payload, wire.view(payload)
        hold = self.engine.check_request(ctx)
        if hold is None and not self.engine.wants_response(ctx):
            return None

        run = self._run(
            ctx, hold, wire, request.method, request.url.path, url, headers, body, payload, client,
            assignment=assignment, release=release, harness_payload=harness_payload,
            notes=notes or {},
        )
        if payload.get("stream"):
            task = asyncio.ensure_future(run)
            return StreamingResponse(
                self._stream(task, wire),
                status_code=200,
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-tuningfork": "breakpoint-path"},
            )
        delivery = await run
        if delivery.error is not None:
            return JSONResponse(
                delivery.error, status_code=delivery.status, headers=delivery.headers
            )
        return Response(delivery.body, status_code=delivery.status, headers=delivery.headers)

    async def _stream(self, task: asyncio.Future[Delivery], wire: WireFormat):
        started = time.monotonic()
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=self.settings.keepalive_interval)
                if not done:
                    waited = int(time.monotonic() - started)
                    yield f": tuningfork: call held at a breakpoint for {waited}s\n\n".encode()
            try:
                delivery = task.result()
            except Exception as exc:  # noqa: BLE001 - report it to the client, not as a 500
                delivery = Delivery.failure(
                    wire,
                    500,
                    f"tuningfork failed while handling a held call: {exc}",
                    "tuningfork_error",
                )
            if delivery.error is not None:
                yield wire.error_event(delivery.error)
            else:
                yield delivery.body
        finally:
            if not task.done():
                # The harness hung up: stop waiting, which abandons any pending hold.
                task.cancel()

    async def _wait(self, hold: Hold) -> Decision:
        try:
            return await self.engine.wait(hold)
        except asyncio.CancelledError:
            self.engine.abandon(hold)
            raise

    async def _run(
        self,
        ctx: ExchangeContext,
        hold: Hold | None,
        wire: WireFormat,
        method: str,
        path: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        payload: dict[str, Any],
        client: httpx.AsyncClient,
        *,
        assignment: Any = None,
        release: Callable[[Any], None] | None = None,
        harness_payload: dict[str, Any] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> Delivery:
        notes = notes or {}
        sent = payload
        edits: dict[str, Any] = {}
        stepping = False
        #: Every hold this call stopped at: each is linked to the step the call becomes.
        held = [hold.id] if hold is not None else []

        # --- before the model ---------------------------------------------------------
        if hold is not None:
            decision = await self._wait(hold)
            if decision.action == "drop":
                # Never recorded, so its step number is free for the harness's retry.
                if release is not None and assignment is not None:
                    release(assignment)
                return Delivery.failure(
                    wire,
                    409,
                    "the call was dropped at a tuningfork breakpoint",
                    "tuningfork_dropped",
                    retry=False,
                )
            if decision.request is not None:
                sent = copy.deepcopy(decision.request)
                for key in TRANSPORT_FIELDS:
                    if key in payload:
                        sent[key] = payload[key]
                    else:
                        sent.pop(key, None)
                # The proxy's cap holds an edited call too: an edit asking for more output, or
                # for no limit, is sent at the cap, and the step says what the edit asked.
                # What the harness asked stays in the call's own note (`output_cap`).
                if self.settings.max_output is not None:
                    sent, held_to = cap_output(wire, sent, self.settings.max_output)
                    if held_to is not None:
                        edits["output_cap"] = held_to
                if sent != payload:
                    edits["request"] = describe_edit(payload, sent)
                    # What the model will see now. A response-phase condition, and a
                    # person stepping here, read the edited call — never the one it
                    # replaced. The path stays the harness's: its next call extends that.
                    ctx.request, ctx.view = sent, wire.view(sent)
                else:
                    sent = payload
            stepping = decision.action == "step"

        if self.gate.paused:
            # Never recorded: its step number is free for the retry, as a drop's is.
            if release is not None and assignment is not None:
                release(assignment)
            return Delivery.failure(
                wire,
                503,
                "tuningfork is paused; no requests are being sent upstream",
                "tuningfork_paused",
            )

        # --- the model ----------------------------------------------------------------
        outgoing = body if sent is payload else json.dumps(sent).encode()
        timings = Timings()
        recorded = False

        def record(status: int, error: str | None = None, **response: Any) -> None:
            """Record this exchange: the model's view as the step, plus any edits. Once."""
            nonlocal recorded
            if recorded:
                return
            recorded = True
            self.spawn(
                self.recorder.record(
                    method=method,
                    path=path,
                    request_body=outgoing,
                    request_payload=sent,
                    status=status,
                    timings=timings,
                    response_json=response.get("produced"),
                    raw_response=response.get("raw"),
                    streamed=response.get("event_stream", False),
                    error=error,
                    harness_request=(
                        harness_payload
                        if harness_payload is not None
                        else (payload if "request" in edits else None)
                    ),
                    delivered_response=response.get("delivered"),
                    edits=edits or None,
                    wire=wire,
                    assignment=assignment,
                    replay=notes.get("replay"),
                    substitution=notes.get("substitution"),
                    context=notes.get("context"),
                    output_cap=notes.get("output_cap"),
                    workspace_commit=notes.get("workspace_commit"),
                    holds=list(held),
                )
            )

        await self.gate.acquire()
        try:
            # Read now, with the gate taken: the upstream can be switched while idle.
            target = url() if callable(url) else url
            upstream = await client.send(
                client.build_request(method, target, content=outgoing, headers=headers)
            )
        except httpx.HTTPError as exc:
            timings.finish()
            record(502, f"{type(exc).__name__}: {exc}")
            return Delivery.failure(
                wire,
                502,
                f"tuningfork could not reach the upstream server: {exc}",
                "tuningfork_upstream_error",
            )
        finally:
            # Released before any response hold: a person reading a held response is not
            # using the server.
            self.gate.release()
        timings.finish()

        raw = upstream.content
        parse, event_stream = read_as(upstream.headers, bool(payload.get("stream")))
        event_stream = event_stream and not json_body(raw)  # sent at once as JSON, whatever its label
        # `produced` is in the call's own format. Conditions, holds and edits work on the
        # common shape; an edit is turned back into the harness's format to be delivered.
        try:
            produced, assembler = _parse(wire, raw, event_stream) if parse else (None, None)
            common = (
                wire.canonical_response(produced, stream=raw if event_stream else None)
                if produced is not None else None
            )
            failure = wire.failure(upstream.status_code, produced, assembler)
        except Exception as exc:  # noqa: BLE001 - the harness still gets the model's answer
            # Nothing can be held or edited without reading it; it goes on as it came, and
            # is recorded raw.
            logger.warning("tuningfork could not read a held call's response", exc_info=True)
            produced = common = None
            failure = unreadable(exc)
        delivered, delivered_raw = produced, raw
        answered = {"produced": produced, "raw": raw, "event_stream": event_stream}

        # --- after the model, before the harness --------------------------------------
        if upstream.status_code < 400 and common is not None:
            try:
                response_hold = self.engine.check_response(ctx, common, stepping=stepping)
                if response_hold is not None:
                    held.append(response_hold.id)
                    decision = await self._wait(response_hold)
                    if decision.action == "drop":
                        edits["response"] = {"dropped": True}
                        record(upstream.status_code, DROPPED, **answered)
                        return Delivery.failure(
                            wire,
                            409,
                            "the response was dropped at a tuningfork breakpoint",
                            "tuningfork_dropped",
                            retry=False,
                        )
                    if decision.response is not None and decision.response != common:
                        edits["response"] = describe_edit(common, decision.response)
                        delivered = wire.native_response(decision.response, produced)
                        delivered_raw = wire.render(delivered, request=payload)
                    if decision.action == "step":
                        self.engine.arm_next(ctx)
            except asyncio.CancelledError:
                # The harness hung up while its response was held. Nothing was delivered,
                # but the model did answer: that is still the step's behaviour.
                record(upstream.status_code, HUNG_UP, **answered)
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the harness instead
                # Never fall back to delivering the answer as it came: a breakpoint that was
                # meant to review it, or an edit meant to change it, would be skipped silently.
                logger.error("tuningfork could not deliver a held response", exc_info=True)
                what = "edited response" if "response" in edits else "response"
                message = f"tuningfork could not deliver the {what}: {type(exc).__name__}: {exc}"
                record(upstream.status_code, message, **answered)
                return Delivery.failure(wire, 500, message, "tuningfork_error", retry=False)

        if upstream.status_code >= 400:
            record(upstream.status_code, failure, **answered)
            error = wire.parse(raw, False) or wire.error(
                upstream.status_code, raw.decode("utf-8", "replace")[:500], "upstream_error"
            )
            return Delivery(status=upstream.status_code, error=error)

        record(
            upstream.status_code,
            failure,
            delivered=delivered if "response" in edits else None,
            **answered,
        )
        return Delivery(
            status=upstream.status_code,
            body=delivered_raw,
            headers=self.forward_headers(
                upstream.headers, drop=("content-length", "content-encoding")
            ),
        )
