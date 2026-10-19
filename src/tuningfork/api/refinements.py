"""The refinements API: a skill refined at a checkpoint — its evidence, its proposals, its
screens, its validations — and the built-in proposer's settings.

Reading works on the browsing server too. Starting a refinement, proposing an edit, closing
one, and changing the proposer need a running proxy and its token, as every change does.
Asking the built-in proposer, screening proposals and validating one are jobs (`POST /jobs`,
kinds `propose`, `screen` and `validate`). The proposer's key can be set or cleared here,
never read back.

These pages are the person's: a validation is shown with its held-out verdicts and the gate
they feed, which no MCP tool shows. Approving writes the skill's own file, so it is a POST
with the access token, and only with the code its plan shows (`ops.approve`). An agent that can fetch from this port, or read the
recordings folder, can read them too — keep such agents away from it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.control import CONTROL_HEADER
from tuningfork.ops import proposer, refine
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api/refinements"
PROPOSER = "/_tuningfork/api/proposer"
STATUS = {
    "not_found": 404, "closed": 409, "held_out": 409, "remote_not_allowed": 403, "too_large": 413,
    # Approving: refused because of where things stand, not because of what was asked.
    "approved": 409, "drifted": 409, "locked": 409, "gate": 409, "moved": 409, "not_validated": 409,
    "no_change": 409, "permission": 403,
}
#: A body here is a few edits, or a skill's whole text: never near this.
MAX_BODY = 1024 * 1024


def _refusal(error: OperationError) -> JSONResponse:
    return JSONResponse(
        {"error": error.to_json()}, status_code=STATUS.get(error.code or "", 400),
        headers={CONTROL_HEADER: "control"},
    )


async def _body(request: Request, allowed: set[str]) -> dict[str, Any]:
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY:
            raise OperationError("larger than 1 MiB", code="too_large")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise OperationError(f"unknown field {unknown[0]!r}", code="bad_params", field=unknown[0])
    return body


def refinement_routes(*, store: Any, writable: bool) -> list[Route]:
    def answer(function: Callable[[], Any], status: int = 200) -> Any:
        async def run() -> Response:
            try:
                found = await asyncio.to_thread(function)
            except OperationError as error:
                return _refusal(error)
            return JSONResponse(found.to_json() if hasattr(found, "to_json") else found, status_code=status)

        return run()

    async def listing(request: Request) -> Response:
        run = request.query_params.get("run") or None
        closed = request.query_params.get("closed") == "1"

        def build() -> list[dict[str, Any]]:
            found = []
            for each in refine.list_refinements(store, run=run, include_closed=closed):
                found.append({**each.to_json(), "proposals": len(refine.proposals(store, each.id))})
            return found

        return await answer(build)

    async def detail(request: Request) -> Response:
        # The person's page: every validation with its held-out verdicts and its gate.
        return await answer(lambda: refine.view(store, request.path_params["id"], held_out=True))

    async def validations(request: Request) -> Response:
        from tuningfork.ops.validate import validation_views

        return await answer(lambda: validation_views(store, refine.get(store, request.path_params["id"]), held_out=True))

    async def approval_plan(request: Request) -> Response:
        """What approving a proposal would write, where, on what evidence — and the code that
        approves exactly that. Writing it is the POST beside, with the access token."""
        from tuningfork.ops.approve import prepare_approve

        query = request.query_params
        return await answer(lambda: prepare_approve(
            store, request.path_params["id"], query.get("proposal") or "", validation=query.get("validation") or None,
            spec=query.get("spec") or None, target=query.get("target") or None,
        ))

    async def packet(request: Request) -> Response:
        """The evidence as a proposer would be shown it now: built, not kept."""
        from tuningfork.ops.packet import build_packet

        return await answer(lambda: build_packet(store, request.path_params["id"], keep=False))

    async def settings(request: Request) -> Response:
        return await answer(lambda: {**proposer.config(store.home), "key": proposer.key_status(store.home)})

    routes = [
        Route(PREFIX, listing, methods=["GET"]),
        Route(f"{PREFIX}/{{id}}", detail, methods=["GET"]),
        Route(f"{PREFIX}/{{id}}/validations", validations, methods=["GET"]),
        Route(f"{PREFIX}/{{id}}/approve", approval_plan, methods=["GET"]),
        Route(f"{PREFIX}/{{id}}/packet", packet, methods=["GET"]),
        Route(PROPOSER, settings, methods=["GET"]),
    ]
    if not writable:
        return routes

    async def create(request: Request) -> Response:
        try:
            body = await _body(request, {"run", "step", "target", "target_match", "target_rules", "target_check", "skill"})
        except OperationError as error:
            return _refusal(error)
        return await answer(lambda: refine.create(
            store, str(body.get("run") or ""), str(body.get("step") if body.get("step") is not None else ""),
            target=body.get("target") or None, target_match=body.get("target_match") or None,
            target_rules=body.get("target_rules") or None, target_check=body.get("target_check") is True,
            created_by={"via": "person"}, skill=str(body["skill"]) if body.get("skill") else None,
        ), status=201)

    async def propose(request: Request) -> Response:
        try:
            body = await _body(request, {"edits", "text", "rationale"})
        except OperationError as error:
            return _refusal(error)
        return await answer(lambda: refine.add_proposal(
            store, request.path_params["id"], edits=body.get("edits"), text=body.get("text"),
            rationale=body.get("rationale") or "", proposer={"via": "person"},
            strict=body.get("text") is None,
        ), status=201)

    async def close(request: Request) -> Response:
        return await answer(lambda: refine.close(store, request.path_params["id"]))

    async def approve_edit(request: Request) -> Response:
        """Approve a proposal, as the person at this page: worked out again here, and written
        only if the code typed is the code of the approval as it stands now."""
        from tuningfork.ops.approve import execute_approve, prepare_approve

        try:
            body = await _body(request, {"proposal", "confirm", "validation", "spec", "target"})
        except OperationError as error:
            return _refusal(error)

        def run() -> Any:
            plan = prepare_approve(
                store, request.path_params["id"], str(body.get("proposal") or ""),
                validation=body.get("validation") or None, spec=body.get("spec") or None,
                target=body.get("target") or None,
            )
            return execute_approve(store, plan, confirm=str(body.get("confirm") or ""), front_end="ui")

        return await answer(run)

    async def repair(request: Request) -> Response:
        from tuningfork.ops.approve import repair as finish

        return await answer(lambda: finish(store, request.path_params["id"]))

    async def change(request: Request) -> Response:
        try:
            body = await _body(request, set(proposer.DEFAULTS))
        except OperationError as error:
            return _refusal(error)
        return await answer(lambda: {**proposer.set_config(store.home, **body), "key": proposer.key_status(store.home)})

    async def key(request: Request) -> Response:
        try:
            body = await _body(request, {"key"})
        except OperationError as error:
            return _refusal(error)
        value = body.get("key")
        if value is not None and not isinstance(value, str):
            return _refusal(OperationError("key is text, or null to clear it", code="bad_params", field="key"))
        return await answer(lambda: proposer.set_key(store.home, value))

    return [
        *routes,
        Route(PREFIX, create, methods=["POST"]),
        Route(f"{PREFIX}/{{id}}/proposals", propose, methods=["POST"]),
        Route(f"{PREFIX}/{{id}}/close", close, methods=["POST"]),
        Route(f"{PREFIX}/{{id}}/approve", approve_edit, methods=["POST"]),
        Route(f"{PREFIX}/{{id}}/repair", repair, methods=["POST"]),
        Route(PROPOSER, change, methods=["PUT"]),
        Route(f"{PROPOSER}/key", key, methods=["PUT"]),
    ]
