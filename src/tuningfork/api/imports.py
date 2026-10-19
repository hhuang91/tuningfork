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
        if value is not None and not isinstance(value, str):
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

    return [
        Route(f"{PREFIX}/preview", preview, methods=["POST"]),
        Route(PREFIX, create, methods=["POST"]),
    ]
