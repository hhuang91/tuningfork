"""The scenarios API: task specs kept in the recordings folder, edited from the browser.

Reading works on the browsing server too; writing needs a running proxy (and its token, as
every change does). An input file is uploaded as raw bytes, `PUT /scenarios/{name}/inputs/
{path}`, read with a cap: a file too large is refused as it arrives, not after. A task file
is kept as a scenario by `POST /scenarios/import`: one on this machine by its path, or its
folder uploaded — planned first (`dry_run`), as the page shows it, then made.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
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


#: What an import's body may hold: a path, or a folder's files; whether only to plan it.
IMPORT_FIELDS = {"path", "name", "files", "spec", "folder", "dry_run", "quiet"}


def _upload_cap() -> int:
    """An import's body: every file a scenario may copy in, as base64, and room for the JSON."""
    return scenarios.MAX_TOTAL * 4 // 3 + 1024 * 1024


def _import_plan(home: Any, body: dict[str, Any]) -> scenarios.ImportPlan:
    for name in ("path", "name", "spec", "folder"):
        if body.get(name) is not None and not isinstance(body[name], str):
            raise OperationError(f"{name} is text", code="bad_params", field=name)
    for name in ("dry_run", "quiet"):
        if not isinstance(body.get(name, False), bool):
            raise OperationError(f"{name} is true or false", code="bad_params", field=name)
    name = str(body.get("name") or "")
    if body.get("files") is None:
        return scenarios.prepare_import(home, str(body.get("path") or ""), name)
    if body.get("path"):
        raise OperationError("a path or a folder's files, not both", code="bad_params", field="path")
    files = body["files"]
    if not isinstance(files, list):
        raise OperationError("files is a list of {path, data}, data in base64", code="bad_params", field="files")
    decoded: dict[str, bytes] = {}
    for each in files:
        if not isinstance(each, dict) or not isinstance(each.get("path"), str) or not isinstance(each.get("data"), str):
            raise OperationError("each file is {path, data}, data in base64", code="bad_params", field="files")
        try:
            decoded[each["path"]] = base64.b64decode(each["data"], validate=True)
        except (binascii.Error, ValueError):
            raise OperationError(f"{each['path']}: its data is not base64", code="bad_params", field="files") from None
    return scenarios.prepare_upload(home, name, decoded, spec=body.get("spec") or None,
                                    folder_name=body.get("folder") or None)


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
        """A task file kept as a scenario: one on this machine by its path, `{"path", "name"}`, or
        a folder uploaded, `{"name", "files": [{"path", "data": <base64>}], "spec"?, "folder"?}` —
        `dry_run` its plan, writing nothing, and `quiet` a dry run's refusal as `{"refused": …}`,
        for a form asking as it is filled in."""
        body: Any = {}
        try:
            raw = await _capped(request, _upload_cap())
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                raise OperationError("the body is not JSON", code="bad_params") from None
            if not isinstance(body, dict):
                raise OperationError("the body must be a JSON object", code="bad_params")
            unknown = sorted(set(body) - IMPORT_FIELDS)
            if unknown:
                raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
            plan = await asyncio.to_thread(_import_plan, home, body)
            if body.get("dry_run") is True:
                return JSONResponse({"plan": plan.to_json()})
            made = await asyncio.to_thread(scenarios.execute_import, home, plan)
        except OperationError as error:
            if isinstance(body, dict) and body.get("dry_run") is True and body.get("quiet") is True:
                return JSONResponse({"refused": error.to_json()})
            return _refusal(error)
        return JSONResponse({"scenario": made.to_json(), "plan": plan.to_json()}, status_code=201)

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
