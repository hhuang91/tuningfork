"""The connections API: the inference server the proxy forwards to, the harnesses it can
launch, and the proxy itself.

The server can be switched here while the proxy is idle; a harness's executable can be
named here, kept in the recordings folder. What can only change with a restart — the port,
the gate's size, the number of jobs at once — is shown with the command that changes it.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.control import CONTROL_HEADER
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api/connections"
#: Harness detection runs each harness's `--version`: kept this long before it runs again.
DETECTION_KEPT = 60.0


def _refusal(error: OperationError) -> JSONResponse:
    status = {"not_found": 404, "busy": 409, "protected_endpoint": 403}.get(error.code or "", 400)
    return JSONResponse({"error": error.to_json()}, status_code=status, headers={CONTROL_HEADER: "control"})


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    return body


async def _harnesses(state: Any, *, fresh: bool = False) -> list[dict[str, Any]]:
    from tuningfork.ops.connections import detect_harnesses

    cached = getattr(state, "harnesses", None)
    if fresh or cached is None or time.monotonic() - cached[0] > DETECTION_KEPT:
        found = await asyncio.to_thread(detect_harnesses, state.settings.home)
        cached = (time.monotonic(), [info.to_json() for info in found])
        state.harnesses = cached
    return cached[1]


def _preflight(home: Any, name: str, server: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    from tuningfork.ops.connections import adapter_for, preflight

    try:
        return preflight(adapter_for(home, name), server, checks).to_json()
    except OperationError as error:
        return {"ok": False, "problems": [error.message], "notes": []}


def restart_command(settings: Any, upstream: str) -> str:
    """`tuningfork serve` as this proxy was started: a setting left at its default is left out."""
    parts = [
        "tuningfork", "serve", "--upstream", upstream, "--host", settings.host,
        "--port", str(settings.port), "--max-inflight", str(settings.max_inflight),
        "--jobs", str(settings.jobs), "--home", str(settings.home),
    ]
    if settings.mcp_tools != "evidence":
        parts += ["--mcp-tools", settings.mcp_tools]
    if settings.mcp_results != "both":
        parts += ["--mcp-results", settings.mcp_results]
    if settings.mcp_max_wait != 50:
        parts += ["--mcp-max-wait", str(settings.mcp_max_wait)]
    if settings.max_output is not None:
        parts += ["--max-output", str(settings.max_output)]
    return shlex.join(parts)


def connection_routes() -> list[Route]:
    async def show(request: Request) -> Response:
        from tuningfork.endpoints import protected

        state = request.app.state
        settings = state.settings
        server = getattr(state, "server", None) or {}
        checks = await asyncio.to_thread(state.recorder.store.server_checks, server.get("id"))
        harnesses = await _harnesses(state)
        # Each harness against this server: what stops it, what was never measured.
        harnesses = [
            {**harness, "preflight": _preflight(state.settings.home, harness["name"], server, checks)}
            for harness in harnesses
        ]
        # Why the server cannot be switched now, if it cannot: shown before anyone tries.
        busy = await state.upstream_busy()
        from tuningfork.sandbox import status as sandbox_status

        sandbox = (await asyncio.to_thread(sandbox_status)).to_json()
        return JSONResponse({
            "server": {
                "url": state.upstream.url,
                "started_with": settings.upstream_base_url,
                "identity": server,
                "checks": checks,
                # Why it is not answering, when it is not: down, or still loading its model.
                "problem": getattr(state, "server_problem", None),
            },
            "harnesses": harnesses,
            # Whether launches run in a private view of the machine, and if not, why not.
            "sandbox": sandbox,
            "proxy": {
                "home": str(settings.home),
                "host": settings.host,
                "port": settings.port,
                "max_inflight": state.gate.max_inflight,
                "max_output": settings.max_output,
                "jobs": settings.jobs,
                "protected": [str(endpoint) for endpoint in protected(settings.home)],
                "restart": restart_command(settings, state.upstream.url),
            },
            "busy": busy,
        })

    async def switch(request: Request) -> Response:
        state = request.app.state
        try:
            body = await _body(request)
            unknown = sorted(set(body) - {"url", "dry_run"})
            if unknown:
                raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
            result = await state.switch_upstream(
                request.app, str(body.get("url") or ""), dry_run=bool(body.get("dry_run"))
            )
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(result)

    async def detect(request: Request) -> Response:
        return JSONResponse(await _harnesses(request.app.state, fresh=True))

    async def set_harness(request: Request) -> Response:
        from tuningfork.ops.connections import set_override

        state = request.app.state
        name = request.path_params["name"]
        try:
            body = await _body(request)
            unknown = sorted(set(body) - {"executable"})
            if unknown:
                raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
            executable = body.get("executable")
            if executable is not None and not isinstance(executable, str):
                raise OperationError("executable must be a path, or null", code="bad_params", field="executable")
            await asyncio.to_thread(set_override, state.settings.home, name, executable or None)
        except OperationError as error:
            return _refusal(error)
        found = await _harnesses(state, fresh=True)
        return JSONResponse(next((h for h in found if h["name"] == name), {}))

    return [
        Route(PREFIX, show, methods=["GET"]),
        Route(f"{PREFIX}/upstream", switch, methods=["POST"]),
        Route(f"{PREFIX}/harnesses/detect", detect, methods=["POST"]),
        Route(f"{PREFIX}/harnesses/{{name}}", set_harness, methods=["PUT"]),
    ]
