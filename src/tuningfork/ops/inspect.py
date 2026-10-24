"""Reading recordings: what every front end shows of a run, computed once.

The CLI prints these and the HTTP API serves them as JSON; neither computes a result of its
own, so the two cannot disagree about one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.ops.experiments import resolve_run, step_index
from tuningfork.store.db import Run, Step, Store


# --- runs -----------------------------------------------------------------------------------


@dataclass(slots=True)
class StepLine:
    """One step of a run, with what the model did there."""

    step: Step
    outcome: Any
    #: "none" where the model answered without reasoning though it reasons at other steps of
    #: the run, "reasoned", or None (`analysis.reasoning`).
    reasoning: str | None = None


def step_lines(store: Store, run: Run) -> list[StepLine]:
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.analysis.reasoning import ACTED, reasons, status

    steps = store.steps(run.id)
    run_reasons = reasons(step.reasoning_chars for step in steps if not step.served)
    lines = []
    for step in steps:
        outcome = recorded_outcome(store.response(step), step.error)
        lines.append(StepLine(step, outcome, status(
            step.reasoning_chars, reasons=run_reasons, acted=outcome.kind in ACTED and not step.served,
        )))
    return lines


def runs_judged(runs: list[Run], verdict: str) -> list[Run]:
    """The runs the task's check `verdict` ("passed" or "failed"), counted. Found writing the
    design-md guide: a run to study is a failing one, and nothing picked one. A run no check
    judged, cancelled, or not counted — a timeout, an exclusion — has no verdict here."""
    wanted = verdict == "passed"

    def judged(run: Run) -> bool:
        result = run.result or {}
        return (result.get("passed") is wanted and result.get("usable", True)
                and not result.get("cancelled"))

    return [run for run in runs if judged(run)]


def runs_calling(
    store: Store, tool: str, *, task: str | None = None, step: int | None = None, limit: int = 20
) -> list[Run]:
    """Runs, newest first, where the model called `tool` — at `step`, if one is given. Not
    every run takes the same path: one that wrote its file with a script never called the
    write tool at all, and one that ran code first may search later."""
    from tuningfork.ops.experiments import first_call, tools_called

    def called(run: Run) -> bool:
        if step is None:
            return first_call(store, run, tool) is not None
        recorded = store.step(run.id, step)
        return recorded is not None and tool in tools_called(store, recorded)

    return [run for run in store.runs(limit=1_000_000, task_id=task) if called(run)][:limit]


#: How much of what a step's calls ran a run's view shows: enough to tell one script from another.
RAN_LIMIT = 240
#: Lines of a script or a command that say nothing of what it does.
_QUIET = re.compile(r"^(#|import\s|from\s\S+\s+import\s|[\[\](){},;]+$)")
#: Arguments that hold what a call runs or writes, by the names harnesses give them, in order of
#: preference; a call with none of them shows its longest value.
_RUNS = ("code", "command", "cmd", "script", "content", "text", "patch")


def hides_what_it_ran(label: str) -> bool:
    """Whether a step's label shows what its call ran only as a digest (a script's syntax tree,
    `<ast:…>`; a long text, `<text:…>`) or cut short — where the run's view adds `ran`."""
    return "<ast:" in label or "<text:" in label or label.endswith("…)") or label.endswith("…")


def ran_excerpt(response: dict[str, Any] | None, limit: int = RAN_LIMIT) -> str | None:
    """What a step's calls ran, in short: each call's tool, then what it ran or wrote — its code,
    command or content (`_RUNS`), else its longest value — imports, comments and blank lines left
    out, its lines run together. A call's label keeps a script only as a digest, so steps that ran different scripts
    looked alike (findings §31.42: the step that booked a delivery and the step that tried a test
    entry were both `execute_code(code='<ast:…>')`). None for a step without calls."""
    from tuningfork.analysis.normalize import mask_workspace

    message = ((response or {}).get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    parts = []
    for call in calls:
        function = call.get("function") or {}
        raw = function.get("arguments")
        try:
            arguments: Any = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            arguments = raw
        values: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, str):
                values.append(value)

        collect(arguments)
        named = [arguments[key] for key in _RUNS if isinstance(arguments, dict) and isinstance(arguments.get(key), str)]
        longest = named[0] if named else max(values, key=len, default="")
        kept = [line.strip() for line in longest.splitlines() if line.strip() and not _QUIET.match(line.strip())]
        body = " ".join(" ".join(kept or [longest]).split())
        name = str(function.get("name") or "?")
        parts.append(f"{name}: {body}" if body else name)
    text = mask_workspace(" · ".join(parts))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_detail(store: Store, reference: str) -> dict[str, Any]:
    """A run as its page shows it: each step with what the model did there — what its calls ran,
    where the label shows only a digest of it, and the files they changed — its threads and the
    launch that made it."""
    from tuningfork.ops.timeline import launch_json, run_json

    found = resolve_run(store, reference)
    changed = step_changes(store, found)
    steps = [
        {
            "idx": line.step.idx,
            "outcome": line.outcome.to_json(),
            "ran": ran_excerpt(store.response(line.step)) if hides_what_it_ran(line.outcome.label) else None,
            "changed": [{"status": status, "path": path} for status, path in changed.get(line.step.idx, [])],
            "finish_reason": line.step.finish_reason,
            "timings": line.step.timings,
            "usage": line.step.usage,
            "reasoning_chars": line.step.reasoning_chars,
            "reasoning": line.reasoning,
            "edited": line.step.edited,
            "error": line.step.error,
            # A fork's replayed step, where a variant was swapped in, the workspace.
            "served": line.step.served,
            "replay": line.step.replay or None,
            "substitution": line.step.substitution or None,
            "workspace_commit": line.step.workspace_commit,
        }
        for line in step_lines(store, found)
    ]
    threads = [] if found.is_thread else store.threads_of(found.id)
    launch = store.get_launch(found.launch_id) if found.launch_id else None
    return {
        "run": run_json(found),
        "steps": steps,
        "threads": [run_json(thread) for thread in threads],
        "launch": launch_json(launch),
    }


# --- tasks ----------------------------------------------------------------------------------


def condition_json(condition: Any) -> dict[str, Any]:
    """One condition a task ran under — harness, model, skill version, server — with its
    runs' check pass rate."""
    import hashlib
    import json

    passed = condition.passed
    parts = [condition.harness, condition.model, condition.skill_version]
    if condition.server_id:
        # Only when known, so a condition recorded before servers were keeps its id.
        parts.append(condition.server_id)
    identity = json.dumps(parts)
    return {
        # Stable across reloads, so a page can link to one condition of a task.
        "id": hashlib.sha256(identity.encode()).hexdigest()[:8],
        "harness": condition.harness,
        "model": condition.model,
        "skill_version": condition.skill_version,
        "server_id": condition.server_id,
        "skill_file": condition.skill_file,
        "runs": len(condition.runs),
        "run_ids": condition.run_ids,
        "passed": passed.to_json() if passed else None,
        "uncounted": condition.uncounted,
        "first": condition.first,
        "latest": condition.latest,
    }


def task_summaries(runs: Sequence[Run], checks: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    """Runs grouped by task, with a check pass rate per condition.

    A task run under two wordings, models or harnesses has one rate for each: `passed` is
    given for the task as a whole only when all its runs share one condition. `checks` are
    each task's checks, the latest first (`Store.task_checks`): how its runs pass.
    """
    from tuningfork.analysis.conditions import conditions
    from tuningfork.specs import check_kind, describe_check

    checks = checks or {}
    groups: dict[str | None, list[Run]] = {}
    for found in runs:
        groups.setdefault(found.task_id, []).append(found)
    summaries = []
    for task_id, members in groups.items():
        found_conditions = [condition_json(found) for found in conditions(members)]
        summaries.append(
            {
                "task_id": task_id,
                "runs": len(members),
                "latest": max(found.started_at for found in members),
                "models": sorted({found.model for found in members if found.model}),
                "passed": found_conditions[0]["passed"] if len(found_conditions) == 1 else None,
                "uncounted": sum(found["uncounted"] for found in found_conditions),
                "conditions": found_conditions,
                "check": (
                    {"kind": check_kind(checks[task_id][0]), "text": describe_check(checks[task_id][0])}
                    if checks.get(task_id) else None
                ),
                "checks": len(checks.get(task_id) or []),
            }
        )
    return sorted(summaries, key=lambda summary: summary["latest"], reverse=True)


def tasks(store: Store) -> list[dict[str, Any]]:
    """Every task recorded, newest first, with its pass rate under each condition."""
    return task_summaries(store.runs(limit=100_000), store.task_checks())


# --- divergence -----------------------------------------------------------------------------


def divergence_runs(
    store: Store, *, refs: Sequence[str] = (), task: str | None = None, limit: int = 20
) -> tuple[list[str], str]:
    """The runs a divergence report compares, and what to call them."""
    if refs:
        return [resolve_run(store, ref).id for ref in refs], "selected runs"
    if task:
        return [run.id for run in store.runs(limit=limit, task_id=task)], task
    raise OperationError("pass run ids, or a task to compare every run of")


def divergence(
    store: Store, run_ids: Sequence[str], normalizers: Sequence[str] = ()
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """The divergence report, and each run's stored verdict, for which behaviour passed."""
    from tuningfork.analysis.divergence import compare_runs

    if len(run_ids) < 2:
        raise OperationError("need at least two runs to compare")
    results = {}
    for run_id in run_ids:
        found = store.get_run(run_id)
        results[run_id] = found.result if found else {}
    return compare_runs(store, list(run_ids), normalizers), results


def divergence_report(
    store: Store,
    *,
    refs: Sequence[str] = (),
    task: str | None = None,
    normalizers: Sequence[str] = (),
    limit: int = 20,
) -> dict[str, Any]:
    """Where the runs named, or a task's latest runs, stopped agreeing, as JSON: each step's
    behaviours, and how often the runs behind each went on to pass."""
    from tuningfork.analysis.normalize import validate_normalizers

    try:
        lens = validate_normalizers(normalizers)
    except ValueError as error:
        raise OperationError(str(error), field="normalizers") from None
    if not refs and not task:
        raise OperationError("name a task, or the runs to compare", field="task")
    run_ids, _ = divergence_runs(store, refs=refs, task=task, limit=limit)
    if len(run_ids) < 2:
        raise OperationError(
            "a comparison needs at least two runs",
            hint="run the task again: `tuningfork batch <spec> -n 5`",
        )
    report, results = divergence(store, run_ids, lens)
    out = {**report.to_json(results), "tool_split": tool_split(store, run_ids, results)}
    out["verdict_split"] = verdict_split(out)
    return out


def behaviour_pass_rates(comparison: Any, results: dict[str, dict[str, Any]]) -> list[Any]:
    """For each behaviour at a step, how often the runs that did it went on to pass."""
    from tuningfork.analysis.conditions import pass_rate

    return [
        pass_rate(results.get(run_id, {}) for run_id in group.run_ids)
        for group in comparison.groups
    ]


def verdict_split(report: dict[str, Any], *, alpha: float = 0.2, minimum: int = 2) -> dict[str, Any]:
    """The step whose behaviour best predicts the runs' verdicts, from a divergence report's
    JSON: at each step, each behaviour that at least `minimum` judged runs took, against the
    rest, by Fisher's exact test on passes and fails; the smallest p wins when it is under
    `alpha`. `judged` is how many runs had a verdict; `best` is None when no behaviour
    separates them — the difference is then in what the runs wrote, not in which behaviour
    they chose, which the first split never shows. `best` names a run that took the behaviour,
    to look at (`run_id`), and its kind: "ended" for runs that had ended before the step."""
    from tuningfork.analysis.stats import fisher_exact

    best: dict[str, Any] | None = None
    judged = 0
    for step in report.get("steps") or []:
        groups = [group for group in step.get("groups") or [] if (group.get("passed") or {}).get("total")]
        all_passed = sum(group["passed"]["successes"] for group in groups)
        all_total = sum(group["passed"]["total"] for group in groups)
        judged = max(judged, all_total)
        for group in groups:
            passed, total = group["passed"]["successes"], group["passed"]["total"]
            others_passed, others_total = all_passed - passed, all_total - total
            if total < minimum or others_total < minimum:
                continue
            p = fisher_exact(passed, total - passed, others_passed, others_total - others_passed)
            if best is None or p < best["p"]:
                outcome = group.get("outcome") or {}
                best = {
                    "index": step.get("index"), "label": outcome.get("label"), "kind": outcome.get("kind"),
                    "run_id": (group.get("run_ids") or [None])[0],
                    "passed": passed, "total": total, "others_passed": others_passed,
                    "others_total": others_total, "p": round(p, 4),
                }
    return {"judged": judged, "best": best if best is not None and best["p"] < alpha else None}


def tool_split(
    store: Store, run_ids: Sequence[str], results: dict[str, dict[str, Any]] | None = None,
    *, alpha: float = 0.2, minimum: int = 2,
) -> dict[str, Any]:
    """By the tool each run used, anywhere in it: for every tool some of these runs called,
    the runs that called it against the runs that never did, on their verdicts, by Fisher's
    exact test. A split no step index shows — the step where a run calls a tool depends on
    the path it took, and a run that wrote with a script never called the write tool at
    all — and the one the write path hides: on meeting-action-items the runs that called
    write_file passed 35 of 41, and the runs that scripted the file through execute_code
    58 of 104, whichever step each did it at. `best` is the tool that best predicts the
    verdict, when one does (p under `alpha`, at least `minimum` judged runs on each side),
    with which way it goes: `with_passing`, or the tool of the runs that failed."""
    from tuningfork.analysis.stats import fisher_exact
    from tuningfork.ops.experiments import tools_called
    from tuningfork.ops.targets import verdict_from

    judged: dict[str, str] = {}
    callers: dict[str, set[str]] = {}
    for run_id in run_ids:
        result = results.get(run_id) if results is not None else getattr(store.get_run(run_id), "result", None)
        verdict = verdict_from(result)
        if verdict is None:
            continue
        judged[run_id] = verdict
        for step in store.steps(run_id):
            for name in tools_called(store, step):
                if name:
                    callers.setdefault(name, set()).add(run_id)
    passed_all = sum(1 for verdict in judged.values() if verdict == "passed")
    tools = []
    for name in sorted(callers):
        called = [run_id for run_id in judged if run_id in callers[name]]
        passed = sum(1 for run_id in called if judged[run_id] == "passed")
        never = len(judged) - len(called)
        entry: dict[str, Any] = {
            "tool": name,
            "called": {"passed": passed, "total": len(called)},
            "never": {"passed": passed_all - passed, "total": never},
            "p": None,
            "with_passing": None,
        }
        if len(called) >= minimum and never >= minimum:
            entry["p"] = round(fisher_exact(passed, len(called) - passed, passed_all - passed, never - (passed_all - passed)), 4)
            entry["with_passing"] = passed / len(called) > (passed_all - passed) / never
        tools.append(entry)
    tested = [entry for entry in tools if entry["p"] is not None]
    # Two tools that split the runs the same way have the same p: the one the passing runs
    # used is the one a target names, since a target is a behaviour to raise.
    best = min(tested, key=lambda entry: (entry["p"], not entry["with_passing"], entry["tool"])) if tested else None
    return {"judged": len(judged), "tools": tools, "best": best if best is not None and best["p"] < alpha else None}


# --- experiments ----------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentReport:
    """A stored experiment's result, as it reads again without drawing a sample.

    `kind` is how to show it: `resample`, `ab`, `interrupted` (an A/B with one arm),
    `ab_running` (an A/B whose second arm has not begun), `to_end` or `ab_to_end` (forks
    run to the end, judged by the task's check). `status` is how the experiment as a whole
    stands: running, done, cancelled, interrupted or failed.
    """

    experiment: Any
    kind: str
    checkpoint: Any = None
    distribution: Any = None
    comparison: Any = None
    #: Why a target declared before sampling cannot be re-tested now, if it cannot.
    target_note: str | None = None
    arms: dict[str, Any] = field(default_factory=dict)
    #: Arms that stopped short of the samples they were meant to draw: (arm, drawn, meant).
    partial: list[tuple[str, int, int]] = field(default_factory=list)
    to_end: Any = None
    status: str = "done"
    #: Arms still drawing samples: (arm, drawn, meant). Not stopped short — not yet done.
    running: list[tuple[str, int, int]] = field(default_factory=list)
    #: A forking-token probe's report (`tuningfork.analysis.forking`).
    forking: Any = None
    #: A screen's report (`tuningfork.analysis.screen`): each proposal against one baseline.
    screen: Any = None
    #: An A/B's edit: its two texts as a line diff (`ops.experiments.ab_wording`).
    wording: Any = None
    #: An A/B's significance level: the one it is judged at here, and the one it was declared
    #: with (`declared_alpha`; None where nothing kept it).
    alpha: float | None = None
    alpha_declared: float | None = None
    #: Of a resample that asked for each sample's tokens, how many samples kept them: none
    #: where the server sends no logprobs.
    tokens_kept: int | None = None

    @property
    def interrupted_arm(self) -> str | None:
        return next(iter(self.arms), None) if self.kind == "interrupted" else None

    def to_json(self) -> dict[str, Any]:
        """The report as the API serves it; the web UI and, later, MCP read this."""
        report, found = self, self.experiment
        base = {
            "id": found.id,
            "kind": found.kind,
            "run_id": found.run_id,
            "step_idx": found.step_idx,
            "notes": found.notes,
            "changes": found.changes,
            "settings": found.settings,
            "reach": found.settings.get("reach", "step"),
            # Arms that stopped short of the samples they were meant to draw.
            "partial": [
                {"arm": arm, "drawn": drawn, "meant": meant}
                for arm, drawn, meant in report.partial
            ],
            "status": report.status,
            # Arms still drawing: not stopped short, not done yet.
            "running": [
                {"arm": arm, "drawn": drawn, "meant": meant}
                for arm, drawn, meant in report.running
            ],
            "job_id": found.job_id,
            "hold_id": found.hold_id,
            "origin": found.settings.get("origin"),
            "trashed_at": found.trashed_at,
            "alpha": report.alpha,
            "alpha_declared": report.alpha_declared,
            "tokens_kept": report.tokens_kept,
        }
        if report.wording is not None:
            base["wording"] = report.wording.to_json()
        if report.kind == "to_end":
            base["to_end"] = report.to_end.to_json()
            return base
        if report.kind == "screen":
            base["screen"] = report.screen.to_json()
            base["arms"] = {
                name: {
                    "id": arm.id,
                    "proposal": (arm.variant or {}).get("proposal"),
                    "variant_path": (arm.variant or {}).get("variant_path"),
                    "samples": arm.sample_count,
                    "status": arm_status(arm),
                }
                for name, arm in report.arms.items()
            }
            return base
        if report.kind == "forking":
            base["forking"] = report.forking.to_json()
            return base
        if report.kind == "resample":
            base["distribution"] = report.distribution.to_json()
            return base
        if report.kind in ("interrupted", "ab_running"):
            # Stopped before the second arm, or not at it yet: show what was sampled.
            if report.kind == "interrupted":
                base["interrupted"] = report.interrupted_arm
            else:
                base["running_arm"] = report.only_arm
            if report.to_end is not None:
                base["to_end"] = report.to_end.to_json()
            else:
                base["distribution"] = report.distribution.to_json()
            return base
        base["variant"] = report.arms["variant"].variant
        if report.kind == "ab_to_end":
            base["to_end"] = report.to_end.to_json()
            return base
        if report.target_note:
            base["notes"] = [*found.notes, report.target_note]
        base["comparison"] = report.comparison.to_json()
        return base
    @property
    def only_arm(self) -> str | None:
        """The one arm an A/B has so far: interrupted before its second, or still in its
        first."""
        return next(iter(self.arms), None) if self.kind in ("interrupted", "ab_running") else None


def declared_alpha(store: Store, experiment: Any) -> float | None:
    """The significance level an A/B was declared with: kept with its arms — or, for one run
    before it was kept, its job's own, where a job started it. None where neither says."""
    def level(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    kept = level(experiment.settings.get("alpha"))
    if kept is not None:
        return kept
    job = store.get_job(experiment.job_id) if experiment.job_id else None
    if job is None or job.kind not in ("ab", "ab-to-end"):
        return None
    # A job that named none ran at its kind's default.
    return level((job.params or {}).get("alpha", 0.05))


def experiment_report(
    store: Store, reference: str, *, lens: Sequence[str] | None = None, alpha: float | None = None
) -> ExperimentReport:
    """A stored experiment's result, as it reads again. An A/B is judged at the significance
    level it was declared with, unless `alpha` names another (found in review: one declared at
    0.01 was judged at 0.05 on its own page, while its job's page said 0.01)."""
    from tuningfork.analysis.distribution import compare_arms
    from tuningfork.experiment import (
        CheckpointError,
        declared_target,
        experiment_distribution,
        experiment_outcomes,
        experiment_reasoning,
        load_checkpoint,
    )
    from tuningfork.experiment.sampling import target_letters
    from tuningfork.ops.forks import arm_summary, compare_to_end

    found = store.get_experiment(reference)
    if found is None:
        raise OperationError(f"no experiment matching {reference!r}")
    try:
        # None while it samples a call held at a breakpoint that is not recorded yet.
        checkpoint = (
            load_checkpoint(store, found.run_id, found.step_idx) if found.run_id is not None else None
        )
    except CheckpointError:
        checkpoint = None  # the run it sampled has gone; its samples have not
    arms = (
        {arm.arm: arm for arm in store.experiment_group(found.group_id)}
        if found.group_id
        else {found.arm: found}
    )
    statuses = {name: arm_status(arm) for name, arm in arms.items()}
    counts = {
        name: (arm.arm, arm.sample_count, int(arm.settings.get("n") or 0))
        for name, arm in arms.items()
    }
    running = [counts[name] for name in arms if statuses[name] == "running"]
    partial = [
        counts[name] for name in arms
        if statuses[name] != "running" and counts[name][1] < counts[name][2]
    ]
    report = ExperimentReport(
        experiment=found, kind="resample", checkpoint=checkpoint, arms=arms, partial=partial,
        status=overall_status(statuses.values()), running=running,
    )
    if found.settings.get("logprobs"):
        samples = [sample.id for sample in store.samples(found.id)]
        report.tokens_kept = len(store.token_streams(sample_ids=samples)) if samples else 0
    lens = list(lens) if lens else None
    if found.kind == "screen":
        from tuningfork.ops.screen import screen_result

        if lens:
            raise OperationError(
                "a screen is reported under the normalizers it was planned with: its target was "
                "declared under them"
            )
        report.kind, report.screen = "screen", screen_result(store, found.id).report
        return report
    if found.kind == "forking-tokens":
        from tuningfork.ops.forking import forking_result

        # Its lens and target are the ones declared before sampling: the report reads them.
        report.kind, report.forking = "forking", forking_result(store, found)
        return report
    to_end = found.settings.get("reach") == "end"
    if found.kind == "ab":
        from tuningfork.ops.experiments import ab_wording

        report.wording = ab_wording(store, arms)
    if found.kind != "ab" or not found.group_id:
        if to_end:
            report.kind, report.to_end = "to_end", arm_summary(store, found)
        else:
            report.distribution = experiment_distribution(store, found, lens)
        return report
    if set(arms) != {"original", "variant"}:
        (arm,) = arms.values()
        report.kind = "ab_running" if statuses[arm.arm] == "running" else "interrupted"
        if to_end:
            report.to_end = arm_summary(store, arm)
        else:
            report.distribution = experiment_distribution(store, arm, lens)
        return report
    # Judged at the level it was declared with, unless asked to judge at another.
    report.alpha_declared = declared_alpha(store, arms["original"])
    report.alpha = alpha if alpha is not None else report.alpha_declared if report.alpha_declared is not None else 0.05
    if to_end:
        report.kind = "ab_to_end"
        report.to_end = compare_to_end(
            arm_summary(store, arms["original"]), arm_summary(store, arms["variant"]), report.alpha
        )
        return report
    declared, report.target_note = declared_target(arms["original"], lens)
    report.kind = "ab"
    report.comparison = compare_arms(
        experiment_outcomes(store, arms["original"], lens),
        experiment_outcomes(store, arms["variant"], lens),
        target=declared,
        alpha=report.alpha,
        reasoning=(experiment_reasoning(store, arms["original"]), experiment_reasoning(store, arms["variant"])),
        letters=target_letters(store, declared.spec, lens) if declared else None,
    )
    return report

def arm_status(experiment: Any) -> str:
    """How one experiment stands.

    Its stored status; for an experiment recorded before statuses existed, done if it drew
    every sample it was meant to and interrupted if not. A running one whose process has
    gone was cut off: it reads as interrupted.
    """
    from tuningfork.owners import owner_alive

    status = experiment.status
    if status is None:
        meant = int(experiment.settings.get("n") or 0)
        return "done" if experiment.sample_count >= meant else "interrupted"
    if status == "running" and owner_alive(experiment.owner) is False:
        return "interrupted"
    return status


def launch_status(launch: Any) -> str:
    """How a launch stands: its stored status, but a running one whose process has gone was
    cut off, and reads as interrupted. Found in review: a batch killed hard left its launches
    running for good — none of their runs could go to the trash, and the run a fork of them
    replayed could never be deleted for good."""
    from tuningfork.owners import owner_alive

    if launch.status == "running" and owner_alive(launch.owner) is False:
        return "interrupted"
    return launch.status


def overall_status(statuses: Any) -> str:
    """An A/B's status from its arms': running while any arm is, else the worst ending."""
    statuses = list(statuses)
    for status in ("running", "failed", "cancelled", "interrupted"):
        if status in statuses:
            return status
    return "done"


def sample_view(store: Store, experiment: str, idx: int) -> dict[str, Any]:
    """One sample as the model gave it: what it said, the tools it called, its reasoning —
    which often shows the instruction it was weighing — and what that counts as."""
    found = store.get_experiment(experiment)
    if found is None:
        raise OperationError(f"no experiment matching {experiment!r}", code="not_found")
    sample = next((each for each in store.samples(found.id) if each.idx == idx), None)
    if sample is None:
        raise OperationError(f"experiment {found.id[:8]} has no sample {idx}", code="not_found")
    response = store.sample_response(sample) or {}
    message = ((response.get("choices") or [{}])[0].get("message")) or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "experiment": found.id,
        "kind": found.kind,
        "arm": found.arm,
        "idx": sample.idx,
        "seed": sample.seed,
        "outcome": {"kind": sample.outcome_kind, "label": sample.outcome_label},
        "finish_reason": sample.finish_reason,
        "error": sample.error,
        "content": content or None,
        "tool_calls": [
            {"name": (call.get("function") or {}).get("name"), "arguments": (call.get("function") or {}).get("arguments")}
            for call in message.get("tool_calls") or []
        ],
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or None,
        "usage": sample.usage or None,
        # Run to the end, a sample is a fork: whether the task's check passed there.
        "run_id": sample.run_id,
        "passed": sample.passed,
        "forced": sample.forced,
    }


def experiment_list(store: Store, run: str | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
    """Experiments, newest first — every one, or those at a run's steps."""
    run_id = resolve_run(store, run).id if run else None
    return [
        {
            "id": experiment.id,
            "kind": experiment.kind,
            "reach": experiment.settings.get("reach", "step"),
            "arm": experiment.arm,
            "group_id": experiment.group_id,
            "run_id": experiment.run_id,
            "step_idx": experiment.step_idx,
            "hold_id": experiment.hold_id,
            "samples": experiment.sample_count,
            "n": experiment.settings.get("n"),
            "target": (experiment.settings.get("target") or {}).get("description"),
            "created_at": experiment.created_at,
            "status": arm_status(experiment),
            "job_id": experiment.job_id,
        }
        for experiment in store.experiments(run_id=run_id, limit=limit)
    ]


#: Names for the workspace at the ends of a launch, rather than before one of its calls.
START, END = ("start", "initial"), ("end", "final")


@dataclass(slots=True)
class WorkspaceView:
    """A run's workspace at one point, and what changed since the snapshot it is compared with:
    before a step, the one the step before found — what that step's call changed; at the end, the
    start — what the whole run changed (findings §31.42: an end view of the last step's changes
    read "changed: []", and hid that the run had booked a delivery a second time)."""

    run: Run
    #: The step whose call found the workspace like this; None at the start or the end.
    step: int | None
    #: Where, in words: "before step 2", "at the start", "at the end".
    where: str
    commit: str
    files: list[Any]
    #: The snapshot compared with, and each file that changed since, as (status, path).
    before: str | None = None
    changes: list[tuple[str, str]] = field(default_factory=list)
    #: What `before` is, in words: "the step before", "the start"; None at the start.
    since: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "step": self.step,
            "where": self.where,
            "commit": self.commit,
            "files": [{"path": e.path, "mode": e.mode, "size": e.size} for e in self.files],
            "before": self.before,
            "since": self.since,
            "changes": [{"status": s, "path": p} for s, p in self.changes],
        }


def workspace_at(store: Store, run: str, where: str) -> WorkspaceView:
    """The workspace a run's call at `where` found — a step, by number, tool or what its call
    did (`ops.experiments.step_index`) — or the one it started with (`start`) or ended with
    (`end`). Its changes are since the step before; at the end, since the start."""
    from tuningfork.snapshots import Snapshots, SnapshotError, available

    found = resolve_run(store, run)
    launch = store.get_launch(found.launch_id) if found.launch_id else None
    hint = "workspaces are snapshotted for runs tuningfork launches, from Phase 4 on"
    if not available():
        raise OperationError("workspace snapshots need git, which is not on PATH")
    step: int | None = None
    since: str | None = "the step before"
    if where in START:
        commit, before, words, since = (launch.initial_commit if launch else None), None, "at the start", None
    elif where in END:
        commit = launch.final_commit if launch else None
        before = launch.initial_commit if launch else None
        words, since = "at the end", "the start"
    else:
        step = step_index(store, found, where)
        recorded = store.step(found.id, step)
        if recorded is None:
            raise OperationError(f"run {found.id[:8]} has no step {step}")
        commit, words = recorded.workspace_commit, f"before step {step}"
        previous = store.step(found.id, step - 1) if step > 0 else None
        if previous is not None and previous.workspace_commit:
            before = previous.workspace_commit
        else:
            before, since = (launch.initial_commit if launch else None), "the start"
    if not commit:
        raise OperationError(f"run {found.id[:8]} has no snapshot of its workspace {words}", hint=hint)
    snapshots = Snapshots(store.home)
    try:
        return WorkspaceView(
            run=found,
            step=step,
            where=words,
            commit=commit,
            files=snapshots.files(commit),
            before=before,
            changes=snapshots.diff(before, commit) if before else [],
            since=since if before else None,
        )
    except SnapshotError as exc:
        raise OperationError(f"cannot read the snapshot: {exc}") from None


def workspace_patch(store: Store, view: WorkspaceView, path: str | None = None) -> str:
    """What changed since the snapshot the view is compared with (`view.since`), as a unified
    diff."""
    from tuningfork.snapshots import Snapshots

    return Snapshots(store.home).patch(view.before, view.commit, path)


def workspace_states(store: Store, run: Run) -> list[tuple[int, str]] | None:
    """The states of a run's workspace, each a change from the one before — the start, each that a
    call left changed, and the end — as (the step whose call found the workspace like this, its
    snapshot): 0 the start, the number of steps the end, the start and the end always. What a state
    at step k shows changed was changed by the call at step k − 1. None for a run that kept no
    snapshots of its workspace."""
    from tuningfork.snapshots import Snapshots, SnapshotError

    launch = store.get_launch(run.launch_id) if run.launch_id else None
    if launch is None or not launch.initial_commit or not launch.final_commit:
        return None
    steps = store.steps(run.id)
    # A snapshot is taken at every call, a new commit each time: the files are what tell one state
    # from the next.
    snapshots = Snapshots(store.home)

    def files_of(commit: str) -> str:
        try:
            return snapshots.tree_of(commit)
        except SnapshotError:
            return commit

    states: list[tuple[int, str]] = [(0, launch.initial_commit)]
    last = files_of(launch.initial_commit)
    for step in steps:
        if step.idx > 0 and step.workspace_commit:
            tree = files_of(step.workspace_commit)
            if tree != last:
                states.append((step.idx, step.workspace_commit))
                last = tree
    states.append((len(steps), launch.final_commit))
    return states


def step_changes(store: Store, run: Run) -> dict[int, list[tuple[str, str]]]:
    """What each step's call changed in the workspace, as (status, path) by the step's index:
    the snapshot its call found beside the next one (the run's end, after its last step). Only
    steps that changed something; nothing for a run with no snapshots, or without git."""
    from tuningfork.snapshots import Snapshots, SnapshotError, available

    launch = store.get_launch(run.launch_id) if run.launch_id else None
    if launch is None or not available():
        return {}
    steps = store.steps(run.id)
    commits = [step.workspace_commit for step in steps] + [launch.final_commit]
    snapshots = Snapshots(store.home)
    changed: dict[int, list[tuple[str, str]]] = {}
    for index, step in enumerate(steps):
        before = commits[index]
        after = next((commit for commit in commits[index + 1:] if commit), None)
        if not before or not after or before == after:
            continue
        try:
            found = snapshots.diff(before, after)
        except SnapshotError:
            continue
        if found:
            changed[step.idx] = found
    return changed


def first_change(store: Store, run: Run, pattern: str = "*") -> int | None:
    """The step whose call first changed a file matching `pattern` (a path, or a glob: "*.json")
    in the run's workspace; None when no call did."""
    from fnmatch import fnmatchcase

    for index, changes in sorted(step_changes(store, run).items()):
        if any(fnmatchcase(path, pattern) for _, path in changes):
            return index
    return None


def replayed_changes(store: Store, run: Run, step: int) -> tuple[int, list[str]] | None:
    """The first call before `step` that changed the run's workspace, and the files it changed:
    what a fork from `step` replays as recorded, so that an edit can change only what happens
    from `step` on. None when no call before it changed a file."""
    changes = step_changes(store, run)
    earlier = sorted(index for index in changes if index < step)
    if not earlier:
        return None
    return earlier[0], [path for _, path in changes[earlier[0]]]


def replay_warning(store: Store, run: Run, step: int) -> str | None:
    """A sentence for a person or an agent about to measure an edit from `step` of `run`, when a
    call before it changed the workspace (findings §31.42: a screen from the step after a
    delivery was booked could see neither an edit to the booking's gain nor its harm)."""
    found = replayed_changes(store, run, step)
    if found is None:
        return None
    first, paths = found
    shown = ", ".join(paths[:3]) + (", …" if len(paths) > 3 else "")
    return (f"run {run.id[:8]}: step {first}'s call changed {shown} before step {step}; a fork from step {step} "
            f"replays that as recorded, so an edit can change only what happens from step {step} on — to measure "
            f"one to what step {first} did, start there (or name the step changed:{paths[0]}, each run's own "
            "first change of it)")


def masked_warning(request: dict[str, Any], step: int) -> str | None:
    """A sentence for a person or an agent about to sample at `step`, when its context reads
    `<redacted>`: where text was masked as the step was recorded, each sample is sent the mask
    where the model saw the text, and answers another prompt than the recorded one."""
    from tuningfork.config import REDACTED

    found = json.dumps(request, ensure_ascii=False).count(REDACTED)
    if not found:
        return None
    return (f"step {step}'s context reads {REDACTED} in {found} place{'s' if found != 1 else ''}, where text was "
            "masked as it was recorded: each sample is sent the mask where the model saw the text, so it answers "
            "another prompt than the recorded one")


def restore_workspace(store: Store, view: WorkspaceView, target: Path) -> None:
    """Write the snapshot's files into `target`."""
    from tuningfork.snapshots import Snapshots

    target = Path(target)
    if target.exists() and any(target.iterdir()):
        raise OperationError(f"{target} is not empty", hint="restore into a new or empty folder")
    Snapshots(store.home).restore(view.commit, target)
