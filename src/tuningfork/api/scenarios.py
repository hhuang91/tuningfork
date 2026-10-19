"""The scenarios API: task specs kept in the recordings folder, edited from the browser.

Reading works on the browsing server too; writing needs a running proxy (and its token, as
every change does). An input file is uploaded as raw bytes, `PUT /scenarios/{name}/inputs/
{path}`, read with a cap: a file too large is refused as it arrives, not after.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.control import CONTROL_HEADER
from tuningfork.ops import scenarios
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api/scenarios"
STATUS = {"not_found": 404, "conflict": 409, "too_large": 413}


def _refusal(error: OperationError) -> JSONResponse:
    return JSONResponse(
        {"error": error.to_json()}, status_code=STATUS.get(error.code or "", 400),
        headers={CONTROL_HEADER: "control"},
    )


async def _json(request: Request) -> dict[str, Any]:
    raw = await _capped(request, scenarios.MAX_SPEC)
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    return body


async def _capped(request: Request, cap: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise OperationError(f"larger than {cap // (1024 * 1024)} MiB", code="too_large")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise OperationError(f"larger than {cap // (1024 * 1024)} MiB", code="too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def scenario_routes(*, store: Any, writable: bool) -> list[Route]:
    home = store.home

    def answer(function: Any, *args: Any, status: int = 200) -> Any:
        async def run() -> Response:
            try:
                found = await asyncio.to_thread(function, *args)
            except OperationError as error:
                return _refusal(error)
            data = found.to_json() if hasattr(found, "to_json") else found
            return JSONResponse(data, status_code=status)

        return run()

    async def listing(request: Request) -> Response:
        return await answer(lambda: [summary.to_json() for summary in scenarios.list_scenarios(store)])

    async def read(request: Request) -> Response:
        return await answer(scenarios.read_scenario, home, request.path_params["name"])

    routes = [
        Route(PREFIX, listing, methods=["GET"]),
        Route(f"{PREFIX}/{{name}}", read, methods=["GET"]),
    ]
    if not writable:
        return routes

    async def validate(request: Request) -> Response:
        try:
            body = await _json(request)
        except OperationError as error:
            return _refusal(error)
        name = body.get("name") if isinstance(body.get("name"), str) else None
        return await answer(lambda: scenarios.check_spec(body.get("spec"), home=home, name=name))

    async def save(request: Request) -> Response:
        try:
            body = await _json(request)
            unknown = sorted(set(body) - {"spec", "etag"})
            if unknown:
                raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
        except OperationError as error:
            return _refusal(error)
        return await answer(
            lambda: scenarios.save_scenario(home, request.path_params["name"], body.get("spec"), etag=body.get("etag"))
        )

    async def delete(request: Request) -> Response:
        return await answer(lambda: {"trash": scenarios.delete_scenario(home, request.path_params["name"])})

    async def upload(request: Request) -> Response:
        try:
            data = await _capped(request, scenarios.MAX_FILE)
        except OperationError as error:
            return _refusal(error)
        return await answer(
            scenarios.put_input, home, request.path_params["name"], request.path_params["path"], data,
            status=201,
        )

    async def remove_input(request: Request) -> Response:
        return await answer(
            lambda: scenarios.delete_input(home, request.path_params["name"], request.path_params["path"]) or {"deleted": True}
        )

    async def import_(request: Request) -> Response:
        try:
            body = await _json(request)
        except OperationError as error:
            return _refusal(error)
        return await answer(
            lambda: scenarios.import_spec(home, str(body.get("path") or ""), str(body.get("name") or "")),
            status=201,
        )

    async def context(request: Request) -> Response:
        """The conversation a scenario continues, edited and saved whole: `{"messages": [...]}`."""
        try:
            raw = await _capped(request, scenarios.MAX_FILE)
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                raise OperationError("the body is not JSON", code="bad_params") from None
            if not isinstance(body, dict) or set(body) != {"messages"}:
                raise OperationError('the body is {"messages": [...]}, or {"messages": {"before": [...], '
                                     '"in_turn": [...]}}', code="bad_params")
        except OperationError as error:
            return _refusal(error)
        return await answer(
            lambda: {"messages": scenarios.put_context(home, request.path_params["name"], body["messages"])}
        )

    return [
        Route(f"{PREFIX}/validate", validate, methods=["POST"]),
        Route(f"{PREFIX}/import", import_, methods=["POST"]),
        *routes,
        Route(f"{PREFIX}/{{name}}/context", context, methods=["PUT"]),
        Route(f"{PREFIX}/{{name}}", save, methods=["PUT"]),
        Route(f"{PREFIX}/{{name}}", delete, methods=["DELETE"]),
        Route(f"{PREFIX}/{{name}}/inputs/{{path:path}}", upload, methods=["PUT"]),
        Route(f"{PREFIX}/{{name}}/inputs/{{path:path}}", remove_input, methods=["DELETE"]),
    ]
