"""The imports API: a conversation an agent had elsewhere, read and imported as a run
(`ops.imports`), for the web UI's import page.

`POST /imports/preview` shows what an import would record — the conversation, the skills it
shows, what the file lacked — and records nothing; `POST /imports` records it. The body is the
file's text with the import's options, read with a cap. Both take the access token, as every
change does: a preview reads a person's file, and the running proxy's store is the only one an
import may write to.

`GET /runs/{run}/scenario` says where a scenario of any run can start, and `POST` makes it
(`ops.imports.prepare_scenario`); `/imports/{run}/scenario`, the address an imported run's had
first, answers the same.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.scenarios import _capped, _refusal
from tuningfork.ops import imports
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api/imports"
#: A scenario of any run is made at `/runs/{run}/scenario`.
RUNS = "/_tuningfork/api/runs"
#: What a body may hold besides the file's text: `prepare_import`'s options.
OPTIONS = ("session", "skill", "skill_dir", "harness", "label", "task_id", "tools_from")
#: What a scenario's body may hold: `prepare_scenario`'s options.
SCENARIO_FIELDS = ("name", "cut", "check", "skill", "skill_dir", "harness", "task_check", "prompt")


async def _body(request: Request) -> tuple[str, dict[str, Any]]:
    raw = await _capped(request, imports.MAX_BYTES * 2)  # the text inside JSON, escaped
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    unknown = sorted(set(body) - {"content", *OPTIONS})
    if unknown:
        raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
    content = body.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OperationError("the file's text is missing", code="bad_params", field="content")
    options = {}
    for name in OPTIONS:
        value = body.get(name)
        # A skill's folder for each skill that has one: one, or a list.
        if name == "skill_dir" and isinstance(value, list):
            if not all(isinstance(each, str) for each in value):
                raise OperationError("skill_dir is a folder, or a list of folders", code="bad_params", field=name)
            value = [each for each in value if each.strip()] or None
        elif value is not None and not isinstance(value, str):
            raise OperationError(f"{name} is text", code="bad_params", field=name)
        if value:
            options[name] = value
    return content, options


def import_routes(*, store: Any) -> list[Route]:
    async def preview(request: Request) -> Response:
        def look(content: str, options: dict[str, Any]) -> dict[str, Any]:
            try:
                return imports.prepare_import(store, content, **options).to_json()
            except OperationError as error:
                if error.code != "choose_session":
                    raise
                # A file of several sessions: the page asks which, from these.
                return {"choose_session": imports.sessions_in(content)}

        try:
            content, options = await _body(request)
            found = await asyncio.to_thread(look, content, options)
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found)

    async def create(request: Request) -> Response:
        def run(content: str, options: dict[str, Any]) -> Any:
            return imports.execute_import(store, imports.prepare_import(store, content, **options))

        try:
            content, options = await _body(request)
            result = await asyncio.to_thread(run, content, options)
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(result.to_json(), status_code=201)

    async def starts(request: Request) -> Response:
        """Where a scenario of a run can start (`imports.scenario_choices`): its person's turns,
        the skills it kept or showed — `?skill=` the one studied, where there are several — and
        the turn it starts from, `?cut=` or the one before the skill came in, with its words and
        the conversation's steps from there. `?harness=`: the harness that had it, where the run
        does not say."""
        try:
            cut = request.query_params.get("cut") or None
            if cut is not None and not cut.isdigit():
                raise OperationError("cut is a message's index", code="bad_params", field="cut")
            found = await asyncio.to_thread(
                imports.scenario_choices, store, request.path_params["run"],
                skill=request.query_params.get("skill") or None, harness=request.query_params.get("harness") or None,
                cut=int(cut) if cut is not None else None,
            )
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found)

    async def scenario(request: Request) -> Response:
        def make(body: dict[str, Any]) -> dict[str, Any]:
            cut = body.get("cut")
            if cut is not None and (isinstance(cut, bool) or not isinstance(cut, int)):
                raise OperationError("cut is a message's index", code="bad_params", field="cut")
            for name in ("skill", "skill_dir", "harness", "prompt"):
                if body.get(name) is not None and not isinstance(body[name], str):
                    raise OperationError(f"{name} is text", code="bad_params", field=name)
            if not isinstance(body.get("task_check", True), bool):
                raise OperationError("task_check is true or false", code="bad_params", field="task_check")
            plan = imports.prepare_scenario(store, request.path_params["run"], str(body.get("name") or ""), cut=cut,
                                            check=body.get("check"), skill=body.get("skill") or None,
                                            skill_dir=body.get("skill_dir") or None, harness=body.get("harness") or None,
                                            task_check=body.get("task_check", True), prompt=body.get("prompt"))
            made = imports.make_scenario(store, plan)
            return {"scenario": made.to_json(), "warnings": [*plan.warnings, *made.checked.warnings],
                    "harness": plan.harness}

        try:
            raw = await _capped(request, 1024 * 1024)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict) or set(body) - set(SCENARIO_FIELDS):
                raise OperationError(f"the body is {{{', '.join(SCENARIO_FIELDS)}}}", code="bad_params")
            found = await asyncio.to_thread(make, body)
        except ValueError:
            return _refusal(OperationError("the body is not JSON", code="bad_params"))
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found, status_code=201)

    return [
        Route(f"{PREFIX}/preview", preview, methods=["POST"]),
        Route(PREFIX, create, methods=["POST"]),
        # A scenario of any run; the imported runs' first address kept.
        *(Route(f"{prefix}/{{run}}/scenario", starts, methods=["GET"]) for prefix in (RUNS, PREFIX)),
        *(Route(f"{prefix}/{{run}}/scenario", scenario, methods=["POST"]) for prefix in (RUNS, PREFIX)),
    ]
