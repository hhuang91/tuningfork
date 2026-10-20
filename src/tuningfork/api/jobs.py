"""The jobs API: start work, watch it, stop it.

`POST /jobs` with `dry_run` answers with the plan and starts nothing — what the web UI shows
before its Start button is enabled. Without it, the job is queued and answered 202. The
browsing server (`tuningfork ui`) serves the stored history read-only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.control import CONTROL_HEADER
from tuningfork.ops.catalog import kinds_json
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api"


def _refusal(error: OperationError, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": error.to_json()}, status_code=status, headers={CONTROL_HEADER: "control"})


def _status_of(error: OperationError) -> int:
    return {"not_found": 404, "stopping": 503}.get(error.code or "", 400)


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    return body


def job_routes(*, store: Any, runner: Any = None) -> list[Route]:
    from tuningfork.jobs import job_json

    async def list_jobs(request: Request) -> Response:
        if runner is not None:
            return JSONResponse(await asyncio.to_thread(runner.list))
        records = await asyncio.to_thread(store.jobs)
        return JSONResponse([job_json(record) for record in records])

    async def kinds(request: Request) -> Response:
        return JSONResponse(runner.kinds() if runner is not None else kinds_json())

    async def get_job(request: Request) -> Response:
        reference = request.path_params["id"]
        raw_after = request.query_params.get("after")
        try:
            after = int(raw_after) if raw_after is not None else None
        except ValueError:
            return _refusal(OperationError("after must be a whole number", code="bad_params", field="after"))
        if runner is not None:
            found = await asyncio.to_thread(runner.get, reference, after=after)
        else:
            record = await asyncio.to_thread(store.get_job, reference)
            found = job_json(record) if record else None
        if found is None:
            return _refusal(OperationError(f"no job matching {reference!r}", code="not_found"), 404)
        return JSONResponse(found)

    routes = [
        Route(f"{PREFIX}/jobs", list_jobs, methods=["GET"]),
        Route(f"{PREFIX}/jobs/kinds", kinds, methods=["GET"]),
        Route(f"{PREFIX}/jobs/{{id}}", get_job, methods=["GET"]),
    ]
    if runner is None:
        return routes

    async def submit(request: Request) -> Response:
        try:
            body = await _json_body(request)
            unknown = sorted(set(body) - {"kind", "params", "dry_run", "quiet", "request_id", "started_by", "dedupe"})
            if unknown:
                raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
            # The address this request reached is the proxy's own: where its jobs send
            # their model calls, and point the harnesses they launch.
            runner.proxy.attach(asyncio.get_running_loop(), str(request.base_url))
            kind, params = body.get("kind"), body.get("params", {})
            if body.get("dry_run"):
                try:
                    return JSONResponse(await asyncio.to_thread(runner.plan, kind, params))
                except OperationError as error:
                    if not body.get("quiet"):
                        raise
                    # A form asking as it is filled in: the refusal is its answer, never an error
                    # a browser logs (found in review: an ordinary visit logged them in the console).
                    return JSONResponse({"refused": error.to_json()})
            job = await asyncio.to_thread(
                runner.submit, kind, params, request_id=body.get("request_id"),
                started_by=body.get("started_by"), dedupe=bool(body.get("dedupe")),
            )
        except OperationError as error:
            return _refusal(error, _status_of(error))
        return JSONResponse(job, status_code=202)

    async def cancel(request: Request) -> Response:
        try:
            job = await asyncio.to_thread(runner.cancel, request.path_params["id"])
        except OperationError as error:
            return _refusal(error, _status_of(error))
        return JSONResponse(job)

    return [
        *routes,
        Route(f"{PREFIX}/jobs", submit, methods=["POST"]),
        Route(f"{PREFIX}/jobs/{{id}}/cancel", cancel, methods=["POST"]),
    ]
