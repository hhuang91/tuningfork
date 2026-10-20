"""Adding a skill from the browser (`ops.addskill`): a folder on this machine, an uploaded
folder, or the skill a run had — each planned first (`dry_run`), then added; and forgetting one.

    POST   /skills/add            {"kind": "folder", "path": …}
                                  {"kind": "upload", "name": …, "files": [{"path": …, "data": <base64>}]}
                                  {"kind": "run", "run": …, "skill"?: …, "name"?: …}
                                  each with "dry_run" (the plan, nothing written) and "quiet"
                                  (a dry run's refusal as {"refused": …}, for a form asked as it is filled in)
    DELETE /skills/added/{name}   forget it: a folder named stays; a copy is moved aside
    GET    /runs/{run}/skills     the skills a run can give, for a form to choose from

Writing needs a running proxy (and its token, as every change does); a run's skills read on the
browsing server too.
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
from tuningfork.api.scenarios import _capped
from tuningfork.ops import addskill
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api"
STATUS = {"not_found": 404, "conflict": 409, "too_large": 413}
KINDS = {"folder": {"path"}, "upload": {"name", "files"}, "run": {"run", "skill", "name"}}


def _refusal(error: OperationError) -> JSONResponse:
    return JSONResponse({"error": error.to_json()}, status_code=STATUS.get(error.code or "", 400),
                        headers={CONTROL_HEADER: "control"})


def _cap() -> int:
    """An upload's body: every file a copy may hold, as base64, and room for the JSON around it."""
    _, _, total = addskill._limits()
    return total * 4 // 3 + 1024 * 1024


def _plan(store: Any, body: dict[str, Any]) -> addskill.AddPlan:
    kind = body.get("kind")
    if kind not in KINDS:
        raise OperationError(f"kind is one of {', '.join(KINDS)}", code="bad_params", field="kind")
    unknown = sorted(set(body) - KINDS[kind] - {"kind", "dry_run", "quiet"})
    if unknown:
        raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
    if kind == "folder":
        return addskill.prepare_folder(store, _text(body, "path"))
    if kind == "run":
        return addskill.prepare_from_run(store, _text(body, "run"), skill=_text(body, "skill", optional=True),
                                         name=_text(body, "name", optional=True))
    files = body.get("files")
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
    return addskill.prepare_upload(store, _text(body, "name"), decoded)


def _text(body: dict[str, Any], name: str, *, optional: bool = False) -> Any:
    value = body.get(name)
    if value is None or value == "":
        if optional:
            return None
        raise OperationError(f"{name} is needed", code="bad_params", field=name)
    if not isinstance(value, str):
        raise OperationError(f"{name} is text", code="bad_params", field=name)
    return value


def skill_routes(*, store: Any, writable: bool) -> list[Route]:
    async def run_skills(request: Request) -> Response:
        try:
            found = await asyncio.to_thread(addskill.run_skills, store, request.path_params["run"])
        except OperationError as error:
            return _refusal(error)
        return JSONResponse({"skills": found})

    routes = [Route(f"{PREFIX}/runs/{{run}}/skills", run_skills, methods=["GET"])]
    if not writable:
        return routes

    async def add(request: Request) -> Response:
        body: Any = {}
        try:
            raw = await _capped(request, _cap())
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                raise OperationError("the body is not JSON", code="bad_params") from None
            if not isinstance(body, dict):
                raise OperationError("the body must be a JSON object", code="bad_params")
            plan = await asyncio.to_thread(_plan, store, body)
            if body.get("dry_run"):
                return JSONResponse({"plan": plan.to_json()})
            added = await asyncio.to_thread(addskill.execute_add, store, plan)
        except OperationError as error:
            # A form asks as it is filled in: a refusal then is an answer, not a failed request.
            if isinstance(body, dict) and body.get("dry_run") and body.get("quiet"):
                return JSONResponse({"refused": error.to_json()})
            return _refusal(error)
        return JSONResponse({"added": added.to_json(), "plan": plan.to_json()}, status_code=201)

    async def forget(request: Request) -> Response:
        try:
            found = await asyncio.to_thread(addskill.forget, store, request.path_params["name"])
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found)

    return [
        *routes,
        Route(f"{PREFIX}/skills/add", add, methods=["POST"]),
        Route(f"{PREFIX}/skills/added/{{name}}", forget, methods=["DELETE"]),
    ]
