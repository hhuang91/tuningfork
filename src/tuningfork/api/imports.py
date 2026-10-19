"""The imports API: a conversation an agent had elsewhere, read and imported as a run
(`ops.imports`), for the web UI's import page.

`POST /imports/preview` shows what an import would record — the conversation, the skills it
shows, what the file lacked — and records nothing; `POST /imports` records it. The body is the
file's text with the import's options, read with a cap. Both take the access token, as every
change does: a preview reads a person's file, and the running proxy's store is the only one an
import may write to.
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
#: What a body may hold besides the file's text: `prepare_import`'s options.
OPTIONS = ("session", "skill", "skill_dir", "harness", "label", "task_id", "tools_from")


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
        """Where a scenario of an imported run can start: its user turns, and the one before
        the skill came in — `?skill=`, where it kept several — which it starts from unless told
        otherwise."""
        from tuningfork.ops.experiments import kept_skills, study_skill

        def look() -> dict[str, Any]:
            run = store.get_run(request.path_params["run"])
            if run is None:
                raise OperationError(f"no run matching {request.path_params['run']!r}", code="not_found")
            chat = imports.chat_of(store, run)
            try:
                studied = study_skill(run, request.query_params.get("skill") or None)
            except OperationError as error:
                if error.code != "choose_skill":
                    raise
                studied = None
            skill = studied[0] if studied else None
            try:
                start = imports.scenario_start(chat, skill or "", run.harness or "generic")
            except OperationError:
                start = None
            return {"run": run.id, "skill": skill, "skills": list(kept_skills(run)), "turns": imports.turns(chat),
                    "default": start.cut if start else None,
                    "context": len(start.context.before) if start else 0,
                    "in_turn": len(start.context.in_turn) if start else 0,
                    "next_skill": start.next_skill if start else None}

        try:
            return JSONResponse(await asyncio.to_thread(look))
        except OperationError as error:
            return _refusal(error)

    async def scenario(request: Request) -> Response:
        def make(body: dict[str, Any]) -> dict[str, Any]:
            cut = body.get("cut")
            if cut is not None and (isinstance(cut, bool) or not isinstance(cut, int)):
                raise OperationError("cut is a message's index", code="bad_params", field="cut")
            skill = body.get("skill")
            if skill is not None and not isinstance(skill, str):
                raise OperationError("skill is a skill's name", code="bad_params", field="skill")
            plan = imports.prepare_scenario(store, request.path_params["run"], str(body.get("name") or ""),
                                            cut=cut, check=body.get("check"), skill=skill or None)
            made = imports.make_scenario(store, plan)
            return {"scenario": made.to_json(), "warnings": plan.warnings}

        try:
            raw = await _capped(request, 1024 * 1024)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict) or set(body) - {"name", "cut", "check", "skill"}:
                raise OperationError("the body is {name, cut, check, skill}", code="bad_params")
            found = await asyncio.to_thread(make, body)
        except ValueError:
            return _refusal(OperationError("the body is not JSON", code="bad_params"))
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found, status_code=201)

    return [
        Route(f"{PREFIX}/preview", preview, methods=["POST"]),
        Route(PREFIX, create, methods=["POST"]),
        Route(f"{PREFIX}/{{run}}/scenario", starts, methods=["GET"]),
        Route(f"{PREFIX}/{{run}}/scenario", scenario, methods=["POST"]),
    ]
