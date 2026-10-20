"""The recording proxy.

Upstream bytes are forwarded verbatim and recorded by tee, so the recording can never
change what the harness receives. If parsing or storage fails, the traffic still flows.

The proxy is deliberately ignorant of which server is upstream and which harness is
downstream: it speaks the OpenAI chat-completions wire format and nothing else. Server and
harness knowledge lives in `tuningfork.backends` and `tuningfork.harness`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tuningfork.api import control_routes, data_routes
from tuningfork.api.connections import connection_routes
from tuningfork.api.imports import import_routes
from tuningfork.api.jobs import job_routes
from tuningfork.api.refinements import refinement_routes
from tuningfork.api.scenarios import scenario_routes
from tuningfork.api.skills import skill_routes
from tuningfork.api.trash import trash_routes
from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.access import Access
from tuningfork.api.guard import PREFIX, LocalOnly
from tuningfork.jobs import JobRunner
from tuningfork.proxy.local import LocalProxy
from tuningfork.breakpoints import BreakpointEngine, EventBus
from tuningfork.config import EXPERIMENT_HEADER, Settings
from tuningfork.endpoints import find_on_internet
from tuningfork.launches import split as split_launch
from tuningfork.proxy.held import BreakpointPath
from tuningfork.proxy.launches import LaunchRegistry, Refusal, context_note
from tuningfork.proxy.recorder import ANSWERED, Recorder, Timings, read_upstream
from tuningfork.proxy.sse import SSEParser, unreadable
from tuningfork.snapshots import SnapshotError, Snapshots
from tuningfork.snapshots import available as git_available
from tuningfork.ui import ui_routes
from tuningfork.wire import wire_for
from tuningfork.wire.base import cap_output

# Headers that describe a single hop and must not be relayed to the next one.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

#: How often the server's identity is read again, in seconds: a server restarted with
#: another model, context size or build must not be recorded as the one before it.
SERVER_REFRESH = 30.0

logger = logging.getLogger("tuningfork.proxy")


async def _no_such_endpoint(request: Request) -> Response:
    """tuningfork's own namespace is never relayed upstream.

    Not only tidiness: a browser's CORS preflight for a control endpoint would otherwise
    reach the inference server, and llama-server approves every preflight, echoing the
    asking page's origin back with credentials allowed — approving, on tuningfork's behalf,
    a cross-site request tuningfork would never approve itself.
    """
    return JSONResponse(
        {"error": {"message": f"no tuningfork endpoint {request.method} {request.url.path}"}},
        status_code=404,
    )


def _forward_headers(headers: Any, drop: Iterable[str] = ()) -> dict[str, str]:
    dropped = HOP_BY_HOP | {name.lower() for name in drop}
    return {key: value for key, value in headers.items() if key.lower() not in dropped}


def upstream_url(base: str, path: str, query: str = "") -> str:
    """Join the upstream base to the incoming path.

    `upstream_base_url` is meant to be the server root. A base that already ends in `/v1`
    is tolerated, because that is how harnesses are usually configured and pasting it here
    is the obvious mistake to make.
    """
    base = base.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1") :]
    url = f"{base}{path}"
    return f"{url}?{query}" if query else url


class Upstream:
    """Where model calls go. The settings say where the proxy started; a person can point
    it elsewhere while it is idle (see `switch_upstream` in `create_app`). Read only once a
    call has the gate — the gate is what makes the proxy idle."""

    def __init__(self, url: str) -> None:
        self.url = url


#: What a server identity says that makes two servers differ for an experiment.
IDENTITY_FIELDS = ("build", "model_file", "models", "n_ctx", "chat_template_sha256", "slots", "defaults")


class Gate:
    """Caps concurrent upstream requests, and can pause them entirely.

    Defaults to one in flight. On an inference server shared with anything else, a fan-out
    of resamples must never starve the other tenant.
    """

    def __init__(self, max_inflight: int) -> None:
        self.max_inflight = max(1, max_inflight)
        self.paused = False
        self.inflight = 0
        self._semaphore: Any = None

    def _ensure(self) -> Any:
        # Created lazily so the Gate can be built outside a running event loop.
        if self._semaphore is None:
            import asyncio

            self._semaphore = asyncio.Semaphore(self.max_inflight)
        return self._semaphore

    async def acquire(self) -> None:
        await self._ensure().acquire()
        self.inflight += 1

    def release(self) -> None:
        self.inflight -= 1
        self._ensure().release()


class _Finishing(StreamingResponse):
    """A forwarded answer that finishes its exchange however the response ends. Starlette
    starts no body for a response whose start could not be sent — a harness gone before its
    answer began, a cancelled sample — and a body never started never runs its `finally`:
    found in review, the gate's slot then stayed taken, and every later model call waited at
    the gate until the proxy restarted. `finish` is the body's own end, done at most once."""

    def __init__(self, content: Any, finish: Any, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._finish = finish

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._finish()


def _should_parse_body(headers: Any) -> tuple[bool, bool]:
    """(parse at all, is it an event stream) for a response we are recording.

    A content-encoded body is forwarded verbatim but not parsed: decoding a copy just to
    record it would be work for no benefit on a local server, which never compresses.
    """
    encoding = (headers.get("content-encoding") or "").strip().lower()
    if encoding and encoding != "identity":
        return False, False
    content_type = (headers.get("content-type") or "").lower()
    return True, "text/event-stream" in content_type


def create_app(
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    recorder: Recorder | None = None,
    engine: BreakpointEngine | None = None,
    api: bool = True,
) -> Starlette:
    """The recording proxy, with tuningfork's API and web UI — or, with `api=False`, only its
    health and the proxy itself: a private proxy recording a held-out task's runs serves no
    reads, so nothing on a loopback port shows them (`ops.private`)."""
    from tuningfork.endpoints import check_endpoint, guard_hooks

    # Before anything else: a proxy in front of a protected endpoint is never built.
    check_endpoint(settings.upstream_base_url, home=settings.home, purpose="the proxy")
    # The token every state change needs, and the MCP endpoint's own, written where this
    # user's CLI finds them.
    access = Access.start(settings.home, settings.token, settings.mcp_token)
    if api:
        # A held-out run's private folder left by a process that died mid-run: gone now, not
        # when the next validation happens to start one (`ops.private`).
        from tuningfork.ops.private import sweep_stale

        sweep_stale()
    #: A recorder made here is closed here; one passed in belongs to its caller.
    owns_recorder = recorder is None
    recorder = recorder or Recorder(settings)
    bus = engine.bus if engine is not None else EventBus()
    # The recorder's redactions: the engine predicts where a call lands in the tree the
    # recorder builds, from the request as it will be stored.
    engine = engine or BreakpointEngine(
        recorder.store, bus, redaction_patterns=settings.redaction_patterns
    )
    recorder.bus = bus
    gate = Gate(settings.max_inflight)
    server_address = Upstream(settings.upstream_base_url)
    launches = LaunchRegistry(recorder.store, settings.redaction_patterns)
    # Work started from the UI (or any front end) runs here, on threads of its own. What
    # an earlier proxy left running is marked as cut off before anything new starts.
    local = LocalProxy(None)
    jobs = JobRunner(recorder.store, bus, local, background=settings.jobs)
    jobs.sweep()
    snapshots = Snapshots(settings.home) if git_available() else None
    injected_client = client
    #: Recording tasks deliberately detached from the request that produced them.
    background: set[asyncio.Task] = set()
    #: Counts detachments. A completed task leaves `background` immediately, so this is
    #: what a test can assert on to prove recording was not awaited inline.
    recording_stats: dict[str, int] = {"spawned": 0}

    def _log_task_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("tuningfork failed to record a step", exc_info=error)

    def spawn_recording(coro) -> None:
        """Run a recording independently of the request that triggered it.

        A harness that closes the connection the moment it sees `[DONE]` — which the
        OpenAI SDK does — causes the server to cancel the request task. An `await` inside
        the response generator would be cancelled along with it, and since `CancelledError`
        is not an `Exception`, the recording would vanish without even an error. Detaching
        the work means the recording outlives the request.
        """
        task = asyncio.ensure_future(coro)
        recording_stats["spawned"] += 1
        background.add(task)
        task.add_done_callback(background.discard)
        task.add_done_callback(_log_task_error)

    async def settle_recordings() -> None:
        if background:
            await asyncio.gather(*list(background), return_exceptions=True)

    breakpoint_path = BreakpointPath(
        settings=settings,
        gate=gate,
        engine=engine,
        recorder=recorder,
        spawn=spawn_recording,
        settle=settle_recordings,
        forward_headers=_forward_headers,
    )

    #: Set when the upstream may have changed — a connection failed — to read it again now.
    server_stale = asyncio.Event()

    async def refresh_server(app: Starlette) -> None:
        """Read which server is upstream, and record every step as its from now on.

        The harness may send no sampling parameters at all (Hermes sends none), so the
        server's own defaults are part of the checkpoint, not background detail.
        """
        read = await read_upstream(app.state.client, server_address.url)
        identity, problem = read.identity, read.problem
        server_id = await asyncio.to_thread(recorder.store.add_server, identity)
        recorder.server_id = server_id
        recorder.server_defaults = identity.get("defaults", {})
        app.state.server = {"id": server_id, **identity}
        # Why it said nothing of itself, when it did not: down, or still loading its model.
        app.state.server_problem = problem
        # What the identity leaves out — a key asked for, llama.cpp's /props — and whether the
        # server is on the internet: what the reminder of a model run oneself reads, and the
        # token work (`ops.connections.reminder`, `token_work`).
        app.state.server_read = {**read.facts(), "internet": await find_on_internet(server_address.url)}

    async def busy() -> list[str]:
        """Why the upstream cannot be switched now: anything that is using it."""
        reasons = []
        if gate.inflight:
            reasons.append(f"{gate.inflight} model call(s) in flight")
        held = [hold for hold in engine.holds() if hold.pending]
        if held:
            reasons.append(f"{len(held)} call(s) held at a breakpoint")
        active = jobs.active()
        if active:
            reasons.append(f"{len(active)} job(s) queued or running")
        live = await asyncio.to_thread(launches.live)
        if live:
            reasons.append(f"{len(live)} launched harness(es) still running")
        return reasons

    async def switch_upstream(app: Starlette, url: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Point the proxy at another inference server, while nothing is using this one.

        Checked first: not a protected endpoint, and something answers there as a server.
        Then every gate permit is taken, so no call starts while the address changes; the
        new server's identity is recorded from then on, as a restart would.
        """
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.connections import reminder

        url = (url or "").strip().rstrip("/")
        if not re.match(r"https?://[^/]", url):
            raise OperationError(
                "a server's address starts with http:// or https://", field="url",
                hint="like http://127.0.0.1:8081",
            )
        check_endpoint(url, home=settings.home, purpose="the proxy")
        read = await read_upstream(app.state.client, url)
        identity = read.identity
        # One that refuses tuningfork for want of a key answered all the same: a hosted model
        # API, which records a harness's calls as `serve --upstream` does, and is said to be one.
        if not any(key in identity for key in ANSWERED) and not read.asks_key:
            raise OperationError(
                f"nothing answered at {url} as an inference server", field="url",
                hint="check the address, and that the server is running",
            )
        current = getattr(app.state, "server", None) or {}
        plan = {
            "url": url,
            "current": {"url": server_address.url, "server": current},
            "proposed": identity,
            "differences": [
                {"field": key, "current": current.get(key), "proposed": identity.get(key)}
                for key in IDENTITY_FIELDS
                if current.get(key) != identity.get(key)
            ],
            "busy": await busy(),
            "reminder": reminder({**read.facts(), "internet": await find_on_internet(url)}),
        }
        if dry_run:
            return plan
        if plan["busy"]:
            raise OperationError(
                "the proxy is in use: " + "; ".join(plan["busy"]), code="busy",
                hint="switch when nothing is running, so no run is recorded half on each server",
            )
        taken = 0
        try:
            for _ in range(gate.max_inflight):
                await asyncio.wait_for(gate.acquire(), timeout=10)
                taken += 1
            server_address.url = url
        except asyncio.TimeoutError:
            raise OperationError("a call started meanwhile; try again", code="busy") from None
        finally:
            for _ in range(taken):
                gate.release()
        await refresh_server(app)
        bus.publish("connections", upstream=url, server=app.state.server)
        logger.warning("tuningfork now forwards model calls to %s", url)
        return {**plan, "switched": True, "server": app.state.server}

    async def keep_server_current(app: Starlette) -> None:
        while True:
            try:
                await asyncio.wait_for(server_stale.wait(), timeout=SERVER_REFRESH)
            except asyncio.TimeoutError:
                pass
            server_stale.clear()
            try:
                await refresh_server(app)
            except Exception:  # noqa: BLE001 - a diagnostic; the next round tries again
                logger.warning("tuningfork could not read the server's identity", exc_info=True)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        owned = None
        if injected_client is None:
            owned = httpx.AsyncClient(
                timeout=settings.request_timeout, event_hooks=guard_hooks(home=settings.home)
            )
        app.state.client = injected_client or owned
        app.state.gate = gate
        app.state.recorder = recorder
        app.state.settings = settings
        app.state.loop = asyncio.get_running_loop()  # for `stop_live_work`
        try:
            await refresh_server(app)
        except Exception:  # noqa: BLE001 - never block startup on a diagnostic
            recorder.server_defaults = {}
        keeper = asyncio.ensure_future(keep_server_current(app))
        try:
            yield
        finally:
            keeper.cancel()
            if mcp is not None:
                mcp.stop()
            # Jobs were asked to stop by `stop_live_work`; they write to the store, so it
            # closes only once they have.
            jobs.stop()
            if not await asyncio.to_thread(jobs.join, 30.0):
                logger.warning("tuningfork is stopping with jobs still running")
            if background:
                # Let in-flight recordings finish rather than losing the last few steps.
                await asyncio.gather(*list(background), return_exceptions=True)
            if owned is not None:
                await owned.aclose()
            if owns_recorder:
                recorder.store.close()

    def health_json() -> dict[str, Any]:
        from tuningfork import __version__

        return (
            {
                "status": "ok",
                # What a client of another version checks before relying on a newer field.
                "version": __version__,
                "upstream": server_address.url,
                "started_with": settings.upstream_base_url,
                "max_inflight": gate.max_inflight,
                "max_output": settings.max_output,
                "inflight": gate.inflight,
                "paused": gate.paused,
                "home": str(settings.home),
                "breakpoints": len(engine.breakpoints()),
                "held": len(engine.holds()),
                # What a launcher checks before pointing a harness at `/_launch/<token>/`.
                "launches": True,
                # Where this user's CLI finds the access token: a path, never the token.
                "token_required": True,
                "token_file": str(access.file) if access.file else None,
                "mcp_token_file": str(access.mcp_file) if access.mcp_file else None,
                # Which server is answering, as recorded with every step.
                "server": getattr(app.state, "server", None),
                # Why it could not be read when last asked — down, or still loading.
                "server_problem": getattr(app.state, "server_problem", None),
                # What else that read found, and where the server is (`ServerRead`).
                "server_read": getattr(app.state, "server_read", None),
            }
        )

    async def health(request: Request) -> Response:
        return JSONResponse(health_json())

    def take_snapshot(launch: Any, assignment: Any) -> str | None:
        """The launch's workspace as this call arrives. None when it has no workspace, or
        the snapshot failed — which is logged, and never stops the call."""
        workspace = launch.workspace
        if snapshots is None or not workspace or not Path(workspace).is_dir():
            return None
        try:
            with launch.snapshot_lock:
                return snapshots.commit(
                    Path(workspace),
                    f"refs/launches/{launch.launch.id}/calls",
                    f"as call {assignment.idx} of thread {assignment.thread} arrived",
                )
        except SnapshotError:
            logger.warning("tuningfork could not snapshot a workspace", exc_info=True)
            return None

    def same_workspace(commit: str | None, recorded: str | None) -> str | None:
        """Whether a fork's workspace is what the recording's was at the same point."""
        if snapshots is None or not commit or not recorded:
            return None
        try:
            return "same" if snapshots.tree_of(commit) == snapshots.tree_of(recorded) else "differs"
        except SnapshotError:
            return None

    async def serve_recorded(
        request: Request, path: str, wire: Any, payload: dict[str, Any], assignment: Any,
        launch: Any, served: Any, workspace_commit: str | None,
    ) -> Response:
        """Answer a fork's call from the recording, and record that it was."""
        from tuningfork.replay import served_body

        timings = Timings()
        body, native = await asyncio.to_thread(
            served_body, recorder.store, served, payload, wire, launch.replay.pairs
        )
        timings.mark_first_byte()
        timings.finish()
        note = launch.replay.served_note(served)
        compared = await asyncio.to_thread(
            same_workspace, workspace_commit, served.step.workspace_commit
        )
        if compared:
            note["workspace"] = compared
        # A scenario's fork: what the model would have seen holds the conversation too.
        recorded, harness_request, context = payload, None, None
        if launch.in_context(assignment):
            continued = wire.with_context(payload, launch.context.before, launch.context.in_turn)
            if continued is not None:
                recorded, harness_request, context = continued, payload, context_note(launch.context)
        spawn_recording(
            recorder.record(
                method=request.method,
                path=path,
                request_body=b"",
                request_payload=recorded,
                harness_request=harness_request,
                context=context,
                status=200,
                timings=timings,
                response_json=native,
                raw_response=body,
                streamed=bool(payload.get("stream")),
                wire=wire,
                assignment=assignment,
                replay=note,
                workspace_commit=workspace_commit,
                served_by=served.step.server_id,
            )
        )
        media = "text/event-stream" if payload.get("stream") else "application/json"
        return Response(body, status_code=200, media_type=media)

    async def refuse(request: Request, path: str, refusal: Refusal) -> Response:
        wire = wire_for(request.method, path)
        if wire is not None:
            error = wire.error(refusal.status, refusal.message, refusal.kind)
        else:
            error = {"error": {"message": refusal.message, "type": refusal.kind}}
        # A refusal is a decision, not a fault: the client libraries must not resend it.
        return JSONResponse(error, status_code=refusal.status, headers={"x-should-retry": "false"})

    async def proxy(request: Request) -> Response:
        http = request.app.state.client
        # A launched harness talks to `/_launch/<token>/…`; the rest is the path it meant.
        token, path = split_launch(request.url.path)
        launch = None
        if token is not None:
            resolved = await asyncio.to_thread(launches.resolve, token)
            if isinstance(resolved, Refusal):
                return await refuse(request, path, resolved)
            launch = resolved
        body = await request.body()

        # Model traffic is subject to the gate and the pause switch, whoever sent it.
        # Experiment traffic in particular: it is the load that pausing exists to stop.
        # A call is a model call if some wire format recognises it (see tuningfork.wire).
        wire = wire_for(request.method, path)
        model_call = wire is not None
        # ...but experiment traffic is recorded by its experiment, never as a run step, and a
        # relay launch's calls are recorded by the private proxy that sent them, not here.
        recording = model_call and not request.headers.get(EXPERIMENT_HEADER) and (
            launch is None or launch.records
        )
        payload: dict[str, Any] | None = None
        if recording:
            try:
                parsed = json.loads(body) if body else {}
            except (ValueError, RecursionError):
                # Not JSON — or not UTF-8, which `json.loads` raises as a UnicodeDecodeError,
                # not a JSONDecodeError. Not something we understand; just relay it.
                recording = False
            else:
                # Nor is a call without a list of messages: there is no conversation to
                # record or hold. The server will say what is wrong with it.
                if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
                    payload = parsed
                recording = payload is not None

        # Where a launched call belongs is decided now, as it arrives, in arrival order.
        assignment = None
        if recording and payload is not None and launch is not None:
            assignment = launch.assign(payload, wire)

        # A fork's calls before its step are answered from the recording: never held, never
        # paused, never sent to the server. From its step on, a variant wording is swapped
        # into what the model sees; the harness's own request is recorded beside it.
        harness_payload: dict[str, Any] | None = None
        notes: dict[str, Any] = {}
        decision = None

        def give_back(assignment: Any) -> None:
            """A launched call never recorded: its step number is its retry's, and so, at a
            fork's step, is the step itself."""
            launch.release(assignment)
            if decision is not None and launch.replay is not None:
                launch.replay.give_back(decision)
        # The workspace as it stands when the call arrives: every earlier tool effect is on
        # disk by now. Taken before anything else — a hold, the model — can touch it.
        workspace_commit = None
        if assignment is not None and launch is not None:
            workspace_commit = await asyncio.to_thread(take_snapshot, launch, assignment)
        if assignment is not None and launch is not None and launch.replay is not None:
            decision = launch.replay.decide(payload, wire, assignment)
            if decision.served is not None:
                return await serve_recorded(
                    request, path, wire, payload, assignment, launch, decision.served,
                    workspace_commit,
                )
            if decision.mismatch:
                notes["replay"] = {"mismatch": decision.mismatch}
            if decision.at_step:
                live_step = launch.replay.table.live_step
                notes["replay"] = {"live_from": live_step.id, "source_idx": live_step.idx}
                compared = await asyncio.to_thread(
                    same_workspace, workspace_commit, live_step.workspace_commit
                )
                if compared:
                    notes["replay"]["workspace"] = compared

        # A scenario's run that stops at the next skill (`tuningfork.stop_at_next_skill`) ends
        # here, when the answer the harness sends back loaded a skill other than the one it
        # studies: the call is never sent, nor recorded, and the launch says where it stopped,
        # for its runner to end the harness and judge the workspace as it is.
        if assignment is not None and launch is not None and launch.studied:
            loaded = launch.next_skill(payload, wire, assignment)
            if loaded is not None or launch.stopped is not None:
                if launch.stopped is None:
                    launch.stopped = {"skill": loaded, "after_step": assignment.idx - 1}
                    await asyncio.to_thread(recorder.store.note_launch, launch.launch.id, stopped_at_skill=launch.stopped)
                give_back(assignment)
                return JSONResponse(
                    wire.error(400, f"this run ends where the model loads {launch.stopped['skill']}: its scenario "
                                    f"studies {launch.studied} alone", "tuningfork_stopped"),
                    status_code=400, headers={"x-should-retry": "false"},
                )

        # A scenario's run continues a conversation it was not given (`tuningfork.context`):
        # put after the harness's instructions in what the model sees, each call of the main
        # thread — and the model's earlier steps of the prompt's turn after the prompt; the
        # harness's own request is recorded beside it. Then a fork's variant, which replaces the
        # skill's text wherever the model would read it, the context included.
        if assignment is not None and launch is not None and launch.in_context(assignment):
            continued = wire.with_context(payload, launch.context.before, launch.context.in_turn)
            if continued is None:
                give_back(assignment)
                return JSONResponse(
                    wire.error(400, f"this run continues a conversation, which a {wire.name} call cannot "
                                    "carry yet", "tuningfork_context"),
                    status_code=400, headers={"x-should-retry": "false"},
                )
            harness_payload, payload = payload, continued
            notes["context"] = context_note(launch.context)
        if assignment is not None and launch is not None and launch.replay is not None and launch.replay.variant is not None:
            sent, info = launch.replay.variant.apply(payload)
            if info:
                notes["substitution"] = info
            if sent is not payload:
                harness_payload = payload if harness_payload is None else harness_payload
                payload = sent
        if harness_payload is not None:
            body = json.dumps(payload).encode()

        # A proxy started with a cap holds every call a harness makes through it to that many
        # output tokens — a retry asking for more included: a reply looping to the harness's own
        # limit is cut there (an agent's ran to 65,536 tokens for 35 minutes, findings §31.58).
        # What the harness asked is kept with the step, which records what the model was sent.
        if settings.max_output is not None and recording and payload is not None and wire is not None:
            capped, cap_note = cap_output(wire, payload, settings.max_output)
            if cap_note is not None:
                payload, body = capped, json.dumps(capped).encode()
                notes["output_cap"] = cap_note

        if wire is not None and gate.paused:
            # Never recorded, so its step number is free for the harness's retry, as a call
            # dropped at a breakpoint gives it back. Found in review: the retry took the next
            # number, and a fork of a later step, which counts steps by place, went live a call
            # late.
            if assignment is not None and launch is not None:
                give_back(assignment)
            return JSONResponse(
                wire.error(
                    503,
                    "tuningfork is paused; no requests are being sent upstream",
                    "tuningfork_paused",
                ),
                status_code=503,
                # An experiment's call waits this out rather than failing a sample.
                headers={"x-tuningfork-paused": "1"},
            )

        query = request.url.query

        def url() -> str:
            # Read when the call is sent, with the gate taken: see `Upstream`.
            return upstream_url(server_address.url, path, query)

        headers = _forward_headers(
            request.headers, drop=("host", "content-length", EXPERIMENT_HEADER, TOKEN_HEADER)
        )

        # Harness traffic a breakpoint could apply to takes the breakpoint path; nothing
        # else is touched. Experiment traffic is never held: it would just hang a resample.
        # Nor is a launch's that has breakpoints off — an experiment's forks.
        holdable = launch is None or launch.breakpoints
        if recording and payload is not None and wire is not None and engine.active and holdable:
            handled = await breakpoint_path.maybe_handle(
                request,
                wire=wire,
                client=http,
                url=url,
                headers=headers,
                body=body,
                payload=payload,
                assignment=assignment,
                release=give_back if launch is not None else None,
                harness_payload=harness_payload,
                notes={**notes, "workspace_commit": workspace_commit},
            )
            if handled is not None:
                return handled

        timings = Timings()

        # Only model calls take the gate. A token count or a props lookup is not load the
        # gate exists to limit, and must not queue behind a streaming answer.
        if model_call:
            await gate.acquire()
        released = not model_call
        try:
            upstream_request = http.build_request(
                request.method, url(), content=body, headers=headers
            )
            upstream = await http.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            if not released:
                gate.release()
                released = True
            server_stale.set()  # it may have gone, or come back as something else
            if recording:
                timings.finish()
                await recorder.record(
                    method=request.method,
                    path=path,
                    request_body=body,
                    request_payload=payload,
                    status=502,
                    timings=timings,
                    error=f"{type(exc).__name__}: {exc}",
                    wire=wire,
                    assignment=assignment,
                    harness_request=harness_payload,
                    replay=notes.get("replay"),
                    substitution=notes.get("substitution"),
                    context=notes.get("context"),
                    output_cap=notes.get("output_cap"),
                    workspace_commit=workspace_commit,
                )
            message = f"tuningfork could not reach the upstream server: {exc}"
            if wire is not None:
                error = wire.error(502, message, "tuningfork_upstream_error")
            else:
                error = {"error": {"message": message, "type": "tuningfork_upstream_error"}}
            return JSONResponse(error, status_code=502)
        except BaseException:
            if not released:
                gate.release()
            raise

        parse_body, is_event_stream = (
            _should_parse_body(upstream.headers) if recording else (False, False)
        )
        collected = bytearray() if parse_body else None
        parser = SSEParser() if is_event_stream else None
        assembler = wire.assembler() if is_event_stream and wire is not None else None
        #: Set if the recording's copy of the stream could not be read. Reading stops, the
        #: bytes still reach the harness, and the step is recorded raw with this as its error.
        copy_failed: str | None = None

        finished = False

        async def finish() -> None:
            """Once, however the exchange ends: the upstream closed, the gate's slot given back,
            the step recorded — at the body's end, or at the response's when its body never
            started (`_Finishing`)."""
            nonlocal released, finished
            if finished:
                return
            finished = True
            timings.finish()
            try:
                await upstream.aclose()
            finally:
                if not released:
                    released = True
                    gate.release()
            if recording:
                # Never awaited here: see `spawn_recording`. Recording must not be able to
                # break the response, and must not be killed by the harness hanging up either.
                spawn_recording(_record_response())

        async def stream_body():
            nonlocal copy_failed
            try:
                async for chunk in upstream.aiter_raw():
                    timings.mark_first_byte()
                    if collected is not None:
                        collected.extend(chunk)
                        if parser is not None and assembler is not None and copy_failed is None:
                            try:
                                for event in parser.feed(chunk):
                                    assembler.add_event(event)
                            except Exception as exc:  # noqa: BLE001 - see `copy_failed`
                                # Uncaught, this would end the harness's stream mid-answer.
                                copy_failed = unreadable(exc)
                                logger.warning(
                                    "tuningfork could not read a streamed response; "
                                    "recording it raw",
                                    exc_info=True,
                                )
                    yield chunk
            finally:
                await finish()

        async def _record_response() -> None:
            response_json: dict[str, Any] | None = None
            raw = bytes(collected) if collected is not None else None
            error = copy_failed
            if error is None:
                try:
                    if assembler is not None:
                        response_json = assembler.result() if assembler.chunk_count else None
                    elif raw is not None and not is_event_stream:
                        response_json = wire.parse(raw, event_stream=False)
                    # An error status is not the only failure: a 200 can carry an error
                    # body, or a stream that errored or was cut off partway.
                    error = wire.failure(upstream.status_code, response_json, assembler)
                except Exception as exc:  # noqa: BLE001 - still record the exchange, raw
                    logger.warning("tuningfork could not read a response", exc_info=True)
                    response_json, error = None, unreadable(exc)
            await recorder.record(
                method=request.method,
                path=path,
                request_body=body,
                request_payload=payload,
                status=upstream.status_code,
                timings=timings,
                response_json=response_json,
                raw_response=raw,
                streamed=is_event_stream,
                error=error,
                wire=wire,
                assignment=assignment,
                harness_request=harness_payload,
                replay=notes.get("replay"),
                substitution=notes.get("substitution"),
                context=notes.get("context"),
                output_cap=notes.get("output_cap"),
                workspace_commit=workspace_commit,
            )

        return _Finishing(
            stream_body(),
            finish,
            status_code=upstream.status_code,
            headers=_forward_headers(upstream.headers, drop=("content-length",)),
        )

    from tuningfork.harness import ADAPTERS

    mcp = None
    if api and settings.mcp_tools != "none":
        from tuningfork.mcp.http import McpRoute

        # The tools for an agent that connects (OpenJarvis's url, Claude Code's http): loaded
        # with the SDK on first use, behind the guard like everything under the prefix.
        mcp = McpRoute(store=recorder.store, runner=jobs, local=local, health=health_json, toolset=settings.mcp_tools,
                       results=settings.mcp_results, max_wait=settings.mcp_max_wait)
    served = [
        *([Route(f"{PREFIX}/mcp", mcp, methods=["GET", "POST", "DELETE"])] if mcp else []),
        *control_routes(engine=engine, gate=gate, bus=bus, access=access, jobs=jobs),
        *job_routes(store=recorder.store, runner=jobs),
        *connection_routes(),
        *scenario_routes(store=recorder.store, writable=True),
        *skill_routes(store=recorder.store, writable=True),
        *import_routes(store=recorder.store),
        *refinement_routes(store=recorder.store, writable=True),
        *trash_routes(store=recorder.store, writable=True),
        *data_routes(store=recorder.store, adapters=[a() for a in ADAPTERS.values()]),
        *ui_routes(),
    ] if api else []
    app = Starlette(
        routes=[
            Route("/_tuningfork/health", health, methods=["GET"]),
            *served,
            Route(PREFIX, _no_such_endpoint, methods=METHODS),
            Route(f"{PREFIX}/{{path:path}}", _no_such_endpoint, methods=METHODS),
            Route("/{path:path}", proxy, methods=METHODS),
        ],
        middleware=[Middleware(LocalOnly, bind_host=settings.host, access=access)],
        lifespan=lifespan,
    )
    # Also assigned eagerly, so the app works under test transports that skip lifespan.
    app.state.access = access
    local.app = app
    app.state.local = local
    app.state.jobs = jobs
    app.state.upstream = server_address
    app.state.switch_upstream = switch_upstream
    app.state.refresh_server = lambda: refresh_server(app)  # for in-process work: `LocalProxy`
    app.state.upstream_busy = busy
    app.state.client = injected_client
    app.state.launches = launches
    app.state.gate = gate
    app.state.recorder = recorder
    app.state.settings = settings
    app.state.background = background
    app.state.recording_stats = recording_stats
    app.state.engine = engine
    app.state.bus = bus
    app.state.mcp = mcp
    return app


def stop_live_work(app: Starlette) -> int:
    """Let the proxy stop while calls are held and UI tabs are watching.

    A server waits for its open responses to finish before it stops, and neither a held
    call nor an event stream ever finishes by itself: stopping would hang until a second
    Ctrl-C cut every connection mid-response. Instead, held calls are continued unedited —
    the harness gets the model's real answer — and event streams are ended.

    Must run on the app's event loop. Returns how many held calls were released.
    """
    # Jobs first: a job's harness waiting at a hold is killed rather than let go on.
    app.state.jobs.stop()
    released = app.state.engine.release_all()
    app.state.bus.close()
    # An MCP client's open stream would keep the server waiting just the same.
    if getattr(app.state, "mcp", None) is not None:
        app.state.mcp.stop()
    if released:
        logger.warning("tuningfork is stopping: continued %d held call(s) unedited", released)
    return released


class ProxyServer:
    """Runs the proxy under uvicorn, releasing live work on the first request to stop."""

    def __init__(self, app: Starlette, *, host: str, port: int) -> None:
        import uvicorn

        class _Server(uvicorn.Server):
            def handle_exit(self, sig: int, frame: Any) -> None:
                loop = getattr(app.state, "loop", None)
                if not self.should_exit and loop is not None:
                    loop.call_soon_threadsafe(stop_live_work, app)
                super().handle_exit(sig, frame)

        self.server = _Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))

    def run(self) -> None:
        self.server.run()


def create_browse_app(store: Any, *, host: str | None = None) -> Starlette:
    """The UI and data API with no proxy: for browsing recordings when nothing is running.

    Live controls report themselves unavailable rather than failing, so the same page works
    against either server.
    """
    from tuningfork.harness import ADAPTERS

    async def status(request: Request) -> Response:
        return JSONResponse({
            "live": False, "paused": False, "breakpoints": [], "holds": [],
            "auth": {"required": False, "valid": False},
        })

    @asynccontextmanager
    async def lifespan(app: Starlette):
        try:
            yield
        finally:
            store.close()

    return Starlette(
        routes=[
            Route("/_tuningfork/api/status", status, methods=["GET"]),
            *job_routes(store=store),
            *scenario_routes(store=store, writable=False),
            *skill_routes(store=store, writable=False),
            *refinement_routes(store=store, writable=False),
            *trash_routes(store=store, writable=False),
            *data_routes(store=store, adapters=[a() for a in ADAPTERS.values()]),
            *ui_routes(),
        ],
        middleware=[Middleware(LocalOnly, bind_host=host)],
        lifespan=lifespan,
    )
