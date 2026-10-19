"""The data API: recorded runs, steps and experiments, read-only.

Served alongside the live control API by the proxy, and on its own by `tuningfork ui` for
browsing recordings with no proxy running. Every answer is computed from the store the same
way the CLI computes it, so the two can never disagree about a result.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Sequence

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.analysis.normalize import validate_normalizers
from tuningfork.ops import inspect
from tuningfork.ops.common import OperationError
from tuningfork.ops.timeline import run_json
from tuningfork.store.db import Run, Store


class _Text:
    """A saved variant's text, as the editor loads it."""

    def __init__(self, skill: str, name: str, path: Any, text: str) -> None:
        self.skill, self.name, self.path, self.text = skill, name, path, text

    def to_json(self) -> dict[str, Any]:
        return {"skill": self.skill, "name": self.name, "path": str(self.path), "text": self.text}


def _limit(request: Request, default: int) -> int:
    """A `?limit=` parameter; a malformed one is the client's mistake, answered with a 400."""
    raw = request.query_params.get("limit", str(default))
    try:
        limit = int(raw)
    except ValueError:
        raise ValueError(f"limit must be a whole number, not {raw!r}") from None
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return limit


def data_routes(*, store: Store, adapters: Sequence[Any]) -> list[Route]:
    async def threaded(function: Callable[[], Any]) -> Response:
        try:
            return JSONResponse(await asyncio.to_thread(function))
        except LookupError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

    def find_run(reference: str) -> Run:
        run = store.get_run(reference)
        if run is None:
            raise LookupError(f"no run matching {reference!r}")
        return run

    def compared(run_ids: list[str], normalizers: Sequence[str] = ()) -> dict[str, Any]:
        from tuningfork.ops import OperationError
        from tuningfork.ops.inspect import divergence as diverged

        try:
            report, results = diverged(store, run_ids, normalizers)
        except OperationError as error:
            raise LookupError(error.message) from None
        return report.to_json(results)

    async def runs(request: Request) -> Response:
        def build() -> list[dict[str, Any]]:
            limit = _limit(request, 100)
            task = request.query_params.get("task") or None
            return [run_json(run) for run in store.runs(limit=limit, task_id=task)]

        return await threaded(build)

    async def tasks(request: Request) -> Response:
        return await threaded(lambda: inspect.tasks(store))

    async def run(request: Request) -> Response:
        def build() -> dict[str, Any]:
            return inspect.run_detail(store, find_run(request.path_params["run"]).id)

        return await threaded(build)

    async def files(request: Request) -> Response:
        """The workspace as a step's call found it; `at` is a step, `start` or `end`."""
        from tuningfork.ops import OperationError
        from tuningfork.ops.inspect import workspace_at

        def build() -> dict[str, Any]:
            try:
                return workspace_at(
                    store, request.path_params["run"], request.query_params.get("at", "end")
                ).to_json()
            except OperationError as error:
                return {"error": error.message, "hint": error.hint}

        return await threaded(build)

    async def step(request: Request) -> Response:
        from tuningfork.ops.timeline import step_detail

        context = request.query_params.get("context", "1") not in ("0", "false")
        return await threaded(op(lambda: step_detail(
            store, request.path_params["run"], int(request.path_params["idx"]),
            adapters=adapters, context=context,
        )))

    async def timeline(request: Request) -> Response:
        """The run as one conversation: its last context once, each step where it happened."""
        from tuningfork.ops.timeline import run_timeline

        return await threaded(op(lambda: run_timeline(store, request.path_params["run"], adapters=adapters)))

    async def tokens(request: Request) -> Response:
        """A step's answer as tokens — its own, scored, or a sample's with ?stream=<id>."""
        from tuningfork.ops.tokens import step_tokens

        return await threaded(op(lambda: step_tokens(
            store, request.path_params["run"], str(request.path_params["idx"]),
            stream=request.query_params.get("stream") or None,
        )))

    async def sample_answer(request: Request) -> Response:
        """One sample as the model gave it: its text, its tool calls, its reasoning."""
        return await threaded(lambda: _view(lambda: inspect.sample_view(
            store, request.path_params["id"], int(request.path_params["idx"])
        )))

    def _view(function: Callable[[], Any]) -> Any:
        try:
            return function()
        except OperationError as error:
            raise LookupError(error.message) from None

    async def sample(request: Request) -> Response:
        """One sample of an experiment, with its tokens if it kept them."""
        from tuningfork.ops.tokens import sample_tokens

        return await threaded(op(lambda: sample_tokens(
            store, request.path_params["id"], int(request.path_params["idx"])
        )))

    async def diff(request: Request) -> Response:
        def build() -> dict[str, Any]:
            first = find_run(request.query_params.get("a", ""))
            second = find_run(request.query_params.get("b", ""))
            return compared([first.id, second.id])

        return await threaded(build)

    async def divergence(request: Request) -> Response:
        def build() -> dict[str, Any]:
            task = request.query_params.get("task")
            listed = [ref for ref in request.query_params.get("runs", "").split(",") if ref]
            lens = validate_normalizers(request.query_params.getlist("normalizer"))
            if not listed and not task:
                raise ValueError("pass ?task=<task id> or ?runs=<id>,<id>,...")
            for ref in listed:
                find_run(ref)  # an unknown run is a 404, not a 400
            limit = _limit(request, 20) if task else 20
            try:
                return inspect.divergence_report(
                    store, refs=listed, task=task, normalizers=lens, limit=limit
                )
            except OperationError as error:
                raise LookupError(error.message) from None

        return await threaded(build)

    async def tally(request: Request) -> Response:
        """What the checks said, counted: `?task=<id>` for a task's failed runs by condition, or
        `?screen=<experiment id>` for each arm's failed forks of a screen run to the end."""
        from tuningfork.ops.tally import tally_screen, tally_task

        def build() -> Any:
            task, screen = request.query_params.get("task"), request.query_params.get("screen")
            if bool(task) == bool(screen):
                raise ValueError("pass ?task=<task id> or ?screen=<experiment id>")
            try:
                found = tally_task(store, task) if task else tally_screen(store, screen)
            except OperationError as error:
                if error.code == "not_found":
                    raise LookupError(error.message) from None
                raise ValueError(error.message) from None
            return found.to_json()

        return await threaded(build)

    async def chains(request: Request) -> Response:
        """The chains kept, newest first: each one's task, harness, links and rate."""
        from tuningfork.ops.chain import list_chains

        keys = ("id", "status", "task_id", "harness", "model", "n", "tell", "started_at", "elapsed", "rate",
                "not_counted", "stopped", "budget_seconds", "spent_seconds")

        def build() -> Any:
            limit = int(request.query_params.get("limit") or 50)
            return [{**{key: data.get(key) for key in keys}, "links": len(data.get("links") or [])}
                    for data in list_chains(store, limit=limit)]

        return await threaded(build)

    async def chain(request: Request) -> Response:
        """One chain, link by link: each run's verdict, who changed the skill, the lines changed."""
        from tuningfork.ops.chain import get_chain

        def build() -> Any:
            try:
                return get_chain(store, request.path_params["id"])
            except OperationError as error:
                if error.code == "not_found":
                    raise LookupError(error.message) from None
                raise ValueError(error.message) from None

        return await threaded(build)

    async def spend(request: Request) -> Response:
        """What work cost, counted from the records: `?run=`, `?experiment=`, `?job=`, `?chain=` or
        `?task=`, each one id or several, comma-separated — a row for each, in that order."""
        from tuningfork.ops.spend import spend_report

        def build() -> Any:
            named = {key: [part for part in (request.query_params.get(key) or "").split(",") if part]
                     for key in ("run", "experiment", "job", "chain", "task")}
            if not any(named.values()):
                raise ValueError("pass ?run=, ?experiment=, ?job=, ?chain= or ?task=")
            try:
                rows = spend_report(store, runs=named["run"], experiments=named["experiment"], jobs=named["job"],
                                    chains=named["chain"], tasks=named["task"])
            except OperationError as error:
                if error.code == "not_found":
                    raise LookupError(error.message) from None
                raise ValueError(error.message) from None
            return {"rows": [row.to_json() for row in rows]}

        return await threaded(build)

    async def where(request: Request) -> Response:
        """Where a question holds along a task's runs: `?task=<id>` with `target_match` or
        `target_rules` (JSON) for the model's answers, or `path` with `contains` and `at_least` for
        what the runs wrote; `?screen=<id>` instead asks it of a screen run to the end's forks."""
        from tuningfork.ops.where import where_it_holds

        query = request.query_params

        def build() -> Any:
            if bool(query.get("task")) == bool(query.get("screen")):
                raise ValueError("pass ?task=<task id> or ?screen=<screen id>")
            rules = None
            if query.get("target_rules"):
                try:
                    rules = json.loads(query["target_rules"])
                except ValueError:
                    raise ValueError("target_rules is not JSON") from None
            try:
                at_least = int(query.get("at_least") or 1)
            except ValueError:
                raise ValueError("at_least must be a whole number") from None
            try:
                found = where_it_holds(
                    store, query.get("task") or None, target_match=query.get("target_match") or None,
                    target_rules=rules, path=query.get("path") or None, contains=query.get("contains") or None,
                    at_least=at_least, limit=_limit(request, 200), screen=query.get("screen") or None,
                )
            except OperationError as error:
                if error.code == "not_found":
                    raise LookupError(error.message) from None
                raise ValueError(error.message) from None
            return found.to_json()

        return await threaded(build)

    async def experiments(request: Request) -> Response:
        def build() -> list[dict[str, Any]]:
            reference = request.query_params.get("run")
            return inspect.experiment_list(store, find_run(reference).id if reference else None)

        return await threaded(build)

    async def experiment(request: Request) -> Response:
        from tuningfork.ops import OperationError
        from tuningfork.ops.inspect import experiment_report

        def build() -> dict[str, Any]:
            try:
                report = experiment_report(store, request.path_params["id"])
            except OperationError as error:
                raise LookupError(error.message) from None
            return report.to_json()

        return await threaded(build)

    def op(function: Callable[[], Any]) -> Callable[[], Any]:
        """An operation's result as JSON; its refusal as a 404 naming what was missing."""
        from tuningfork.ops import OperationError

        def build() -> Any:
            try:
                return function().to_json()
            except OperationError as error:
                raise LookupError(error.message) from None

        return build

    async def skill(request: Request) -> Response:
        """The skill file a run was given: the baseline a variant is edited from — `?skill=`
        names it where the run kept several (an imported chat)."""
        from tuningfork.ops.experiments import recorded_skill

        named = request.query_params.get("skill") or None
        return await threaded(op(lambda: recorded_skill(store, request.path_params["run"], named)))

    async def targets(request: Request) -> Response:
        """What an A/B at this step can declare as its target: the lettered behaviours of
        the latest resample of the step itself."""
        from tuningfork.ops.experiments import step_targets

        return await threaded(op(lambda: step_targets(
            store, request.path_params["run"], str(request.path_params["idx"])
        )))

    async def behaviours(request: Request) -> Response:
        """The behaviours seen at this step, each marked as a target would count them: what a
        form shows of a target before it is declared."""
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.targets import seen_behaviours

        query = request.query_params

        def build() -> Any:
            rules = None
            if query.get("target_rules"):
                try:
                    rules = json.loads(query["target_rules"])
                except ValueError:
                    raise OperationError("target_rules is not JSON", field="target_rules") from None
            return seen_behaviours(
                store, request.path_params["run"], str(request.path_params["idx"]),
                normalizers=[name for name in (query.get("normalizers") or "").split(",") if name.strip()],
                target_match=query.get("target_match") or None,
                target_rules=rules,
                base=query.get("base") == "1",
            )

        return await threaded(op(build))

    async def skill_list(request: Request) -> Response:
        """Every skill the recordings know, from runs, scenarios, variants and refinements."""
        from tuningfork.ops.skills import skills

        return await threaded(lambda: [found.to_json() for found in skills(store)])

    async def skill_detail(request: Request) -> Response:
        """One skill: its versions with their texts and pass rates, its own folder now."""
        from tuningfork.ops.skills import skill

        return await threaded(op(lambda: skill(store, request.path_params["name"])))

    async def skill_diff(request: Request) -> Response:
        """Two texts of a skill, line by line (`ops.skills.REFS` names them)."""
        from tuningfork.ops.skills import skill_diff

        query = request.query_params
        return await threaded(op(lambda: skill_diff(
            store, request.path_params["name"], query.get("before") or "", query.get("after") or "",
        )))

    async def skill_text(request: Request) -> Response:
        """One text of a skill: a version's, its own file's, a variant's, an old version's."""
        from tuningfork.ops.skills import skill_text as text_of

        return await threaded(op(lambda: text_of(store, request.path_params["name"], request.query_params.get("ref") or "")))

    async def glossary(request: Request) -> Response:
        """tuningfork's words, each defined once."""
        from tuningfork.ops.glossary import glossary as words

        return await threaded(words)

    async def saved_variants(request: Request) -> Response:
        from tuningfork.ops.variants import list_variants

        skill = request.query_params.get("skill") or None
        return await threaded(lambda: [v.to_json() for v in list_variants(store.home, skill)])

    async def saved_variant(request: Request) -> Response:
        from tuningfork.ops.variants import path_for, read_variant

        skill, name = request.path_params["skill"], request.path_params["name"]
        return await threaded(op(lambda: _Text(
            skill, name, path_for(store.home, skill, name), read_variant(store.home, skill, name)
        )))

    prefix = "/_tuningfork/api"
    return [
        Route(f"{prefix}/tasks", tasks, methods=["GET"]),
        Route(f"{prefix}/variants", saved_variants, methods=["GET"]),
        Route(f"{prefix}/skills", skill_list, methods=["GET"]),
        Route(f"{prefix}/skills/{{name}}", skill_detail, methods=["GET"]),
        Route(f"{prefix}/skills/{{name}}/diff", skill_diff, methods=["GET"]),
        Route(f"{prefix}/skills/{{name}}/text", skill_text, methods=["GET"]),
        Route(f"{prefix}/glossary", glossary, methods=["GET"]),
        Route(f"{prefix}/variants/{{skill}}/{{name}}", saved_variant, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/skill", skill, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/steps/{{idx:int}}/targets", targets, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/steps/{{idx:int}}/behaviours", behaviours, methods=["GET"]),
        Route(f"{prefix}/runs", runs, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}", run, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/files", files, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/timeline", timeline, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/steps/{{idx:int}}", step, methods=["GET"]),
        Route(f"{prefix}/runs/{{run}}/steps/{{idx:int}}/tokens", tokens, methods=["GET"]),
        Route(f"{prefix}/experiments/{{id}}/samples/{{idx:int}}", sample, methods=["GET"]),
        Route(f"{prefix}/experiments/{{id}}/samples/{{idx:int}}/answer", sample_answer, methods=["GET"]),
        Route(f"{prefix}/diff", diff, methods=["GET"]),
        Route(f"{prefix}/divergence", divergence, methods=["GET"]),
        Route(f"{prefix}/tally", tally, methods=["GET"]),
        Route(f"{prefix}/spend", spend, methods=["GET"]),
        Route(f"{prefix}/chains", chains, methods=["GET"]),
        Route(f"{prefix}/chains/{{id}}", chain, methods=["GET"]),
        Route(f"{prefix}/where", where, methods=["GET"]),
        Route(f"{prefix}/experiments", experiments, methods=["GET"]),
        Route(f"{prefix}/experiments/{{id}}", experiment, methods=["GET"]),
    ]
