"""The control API: live breakpoints and held calls, over HTTP.

Everything the web UI can do goes through here, and so does the CLI — so the UI never
holds a capability the command line lacks, which matters because a later phase has an LLM
driving the same operations.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tuningfork.access import Access
from tuningfork.breakpoints import BreakpointEngine, BreakpointError, EventBus, HoldError
from tuningfork.breakpoints.events import OVERFLOWED


#: Marks a refusal as this API's own, as opposed to a 404 from something that is not a
#: tuningfork proxy at all — an inference server, or `tuningfork ui`, which only browses.
CONTROL_HEADER = "x-tuningfork"


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message}}, status_code=status, headers={CONTROL_HEADER: "control"}
    )


def _status_for(error: Exception) -> int:
    message = str(error)
    if message.startswith("no single"):
        return 404
    if "already" in message:
        return 409
    return 400


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def control_routes(
    *,
    engine: BreakpointEngine,
    gate: Any,
    bus: EventBus,
    access: Access | None = None,
    jobs: Any = None,
    home: str | None = None,
) -> list[Route]:
    def snapshot() -> dict[str, Any]:
        return {
            "live": True,
            # The recordings folder: the pages show a path inside it from there.
            "home": home,
            "paused": gate.paused,
            "inflight": gate.inflight,
            "breakpoints": [bp.to_json() for bp in engine.breakpoints()],
            "holds": [hold.summary() for hold in engine.holds()],
            "jobs": jobs.active() if jobs is not None else [],
        }

    async def status(request: Request) -> Response:
        # Whether the asking page may act: the UI is read-only until this says so.
        auth = access.status(request.headers) if access else {"required": False, "valid": False}
        return JSONResponse({**snapshot(), "auth": auth})

    async def list_breakpoints(request: Request) -> Response:
        return JSONResponse([bp.to_json() for bp in engine.breakpoints()])

    async def create_breakpoint(request: Request) -> Response:
        try:
            definition = await _body(request)
            # Store lookups off the loop; the registry itself changes only on it, where the
            # proxy reads it while calls go through.
            bp = engine.install(await asyncio.to_thread(engine.prepare, definition))
        except (BreakpointError, ValueError, TypeError) as exc:
            return _error(str(exc), 400)
        return JSONResponse(bp.to_json(), status_code=201)

    async def update_breakpoint(request: Request) -> Response:
        try:
            changes = await _body(request)
            bp = engine.update(request.path_params["id"], enabled=changes.get("enabled"))
        except (BreakpointError, ValueError) as exc:
            return _error(str(exc), _status_for(exc))
        return JSONResponse(bp.to_json())

    async def delete_breakpoint(request: Request) -> Response:
        try:
            bp = engine.remove(request.path_params["id"])
        except BreakpointError as exc:
            return _error(str(exc), _status_for(exc))
        return JSONResponse(bp.to_json())

    async def list_holds(request: Request) -> Response:
        recent = request.query_params.get("recent") in ("1", "true", "yes")
        return JSONResponse([hold.summary() for hold in engine.holds(recent=recent)])

    async def get_hold(request: Request) -> Response:
        try:
            hold = engine.get_hold(request.path_params["id"])
        except HoldError as exc:
            return _error(str(exc), _status_for(exc))
        return JSONResponse(hold.summary(payload=True))

    async def act_on_hold(request: Request) -> Response:
        try:
            edits = await _body(request)
            hold = engine.resolve(
                request.path_params["id"],
                request.path_params["action"],
                request=edits.get("request"),
                response=edits.get("response"),
            )
        except (HoldError, ValueError) as exc:
            return _error(str(exc), _status_for(exc))
        return JSONResponse(hold.summary())

    async def pause(request: Request) -> Response:
        gate.paused = True
        bus.publish("paused", paused=True)
        return JSONResponse(snapshot())

    async def resume(request: Request) -> Response:
        gate.paused = False
        bus.publish("paused", paused=False)
        if jobs is not None:
            await asyncio.to_thread(jobs.kick)  # what waited for the pause to end starts now
        return JSONResponse(snapshot())

    async def events(request: Request) -> Response:
        async def stream():
            async with bus.subscribe() as queue:
                # A new subscriber first gets the whole current state, so a page opened in
                # the middle of a session is immediately correct.
                yield f"data: {json.dumps({'type': 'snapshot', **snapshot()})}\n\n".encode()
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield b": keepalive\n\n"
                        continue
                    if event is None:  # the bus closed: the server is stopping
                        return
                    if event is OVERFLOWED:
                        # Too far behind to trust event by event. Ending the stream makes
                        # the browser reconnect, and a new stream starts with a snapshot.
                        return
                    yield f"data: {json.dumps(event, default=str)}\n\n".encode()

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"cache-control": "no-cache"}
        )

    prefix = "/_tuningfork/api"
    return [
        Route(f"{prefix}/status", status, methods=["GET"]),
        Route(f"{prefix}/events", events, methods=["GET"]),
        Route(f"{prefix}/pause", pause, methods=["POST"]),
        Route(f"{prefix}/resume", resume, methods=["POST"]),
        Route(f"{prefix}/breakpoints", list_breakpoints, methods=["GET"]),
        Route(f"{prefix}/breakpoints", create_breakpoint, methods=["POST"]),
        Route(f"{prefix}/breakpoints/{{id}}", update_breakpoint, methods=["PATCH"]),
        Route(f"{prefix}/breakpoints/{{id}}", delete_breakpoint, methods=["DELETE"]),
        Route(f"{prefix}/holds", list_holds, methods=["GET"]),
        Route(f"{prefix}/holds/{{id}}", get_hold, methods=["GET"]),
        Route(f"{prefix}/holds/{{id}}/{{action}}", act_on_hold, methods=["POST"]),
    ]
