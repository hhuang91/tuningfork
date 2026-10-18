"""Validating a proposal with full runs (brief §7, step 5).

A screen showed that a proposal shifts its target at one step, with the history before that
step written under the original wording. Validation asks what that is worth to the task,
three ways, each arm against the original and drawn fresh:

- **forward:** forks of the checkpoint's run, replayed to the step and live from there to
  the end, with and without the edit (`ops.forks`): does hitting the target carry the task?
- **training:** full runs of the task the refinement came from, with the edit in place from
  the first step (a batch with `skill_file`): does it disturb an earlier step?
- **held out:** full runs of each held-out eval of the spec, verdicts only (`ops.private`):
  does it break a task it was not made for?

n is fixed in the plan, and a validation stands alone: it is never topped up or pooled with
another. Its record is kept in the refinement's folder, `validations/<id>.json`: every
verdict, which the gate reads (`analysis.validation`). A job's plan and result carry no
held-out verdict — whoever follows a job may be an agent — so the person reads them in the
refinement (`validations`, `refine show`).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress, emit
from tuningfork.store.db import Store

FOLDER = "validations"
#: Runs an arm draws, at least: below this a verdict says next to nothing.
MINIMUM = 5


@dataclass(slots=True)
class ValidatePlan:
    refinement: Any
    proposal: Any
    spec: Any
    training: Any
    held_out: list[Any]
    n: int
    runs: int
    alpha: float
    concurrency: int
    harness: str
    model: str | None
    proxy_url: str
    forward: Any
    training_plans: dict[str, Any]
    #: The screen it survived, as it stood when validation began: a purge can delete it.
    screen: dict[str, Any]
    #: Held-out tasks of the spec with runs in the recordings: an agent may have read them.
    held_out_recorded: list[str]
    #: Where the held-out runs' private folders are made, each deleted when its runs end.
    private_base: Path | None = None
    keep_private: Path | None = None
    #: The proxy's cap on a call's output: the held-out runs, which it never records, are
    #: held to it by their private proxy, as every other run here is by the proxy itself.
    max_output: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return f"{self.spec.skill_name}/{self.training.id}"

    @property
    def total(self) -> int:
        return 2 * self.n + 2 * self.runs * (1 + len(self.held_out))

    def to_json(self) -> dict[str, Any]:
        """The plan as anyone may see it: held-out tasks by their ids, nothing else of them."""
        return {
            "refinement": self.refinement.id,
            "proposal": self.proposal.id,
            "checkpoint": {"run_id": self.refinement.run_id, "step": self.refinement.step_idx},
            "task": self.task_id,
            "forward": {"forks_per_arm": self.n},
            "training": {"runs_per_arm": self.runs},
            "held_out": [{"eval": entry.id, "runs_per_arm": self.runs} for entry in self.held_out],
            "total": self.total,
            "alpha": self.alpha,
            "minimum": MINIMUM,
            "harness": self.harness,
            "model": self.model,
            "screen": self.screen,
            "private": {
                "under": str(self.private_base) if self.private_base else None,
                "kept_at": str(self.keep_private) if self.keep_private else None,
                "says": "held-out runs are recorded in a temporary folder outside the recordings folder, "
                        "deleted when they end, and only their verdicts kept"
                        + (f"; a copy is kept at {self.keep_private}" if self.keep_private else ""),
                "max_output": self.max_output,
            },
            "warnings": self.warnings,
        }


def latest_screen(store: Store, refinement: Any, proposal_id: str) -> dict[str, Any] | None:
    """The proposal's latest complete screen, and whether it kept the proposal: what the gate's
    G0 reads. A quick look, which never has a verdict, and an unfinished screen are not it; a
    screen in the trash is out of every list. None when no complete screen has screened it."""
    from tuningfork.ops import refine

    for screen in refine.screens(store, refinement.id):  # newest first
        report = screen["report"]
        if not report.get("complete") or report.get("quick"):
            continue
        names = {name for name, pid in (screen.get("proposals") or {}).items() if pid == proposal_id}
        arm = next((arm for arm in report["arms"] if arm["name"] in names), None)
        if arm is not None:
            return {
                "survived": bool(arm.get("survives")), "screen": screen["experiment"], "target": arm["target"],
                "baseline": report["baseline"]["target"], "adjusted_p": arm.get("adjusted_p"),
                "alpha": report.get("alpha"), "family": report.get("family"),
            }
    return None


def surviving_screen(store: Store, refinement: Any, proposal: Any) -> dict[str, Any]:
    """The proposal's latest complete screen, as the gate keeps it: refused unless it survived
    there. Found in review: an earlier screen it survived was taken, though a later one — with
    more proposals in its family, or run to the end — had not kept it."""
    found = latest_screen(store, refinement, proposal.id)
    if found is None:
        raise OperationError(
            f"proposal {proposal.id[:8]} has not survived a complete screen",
            hint="screen it first: `tuningfork refine screen <refinement> --proposals <proposal>`",
            code="not_screened", field="proposal",
        )
    if not found["survived"]:
        raise OperationError(
            f"proposal {proposal.id[:8]} did not survive its latest complete screen, {found['screen'][:8]}",
            hint="the latest screen of a proposal is its evidence: screen it again "
                 "(`tuningfork refine screen <refinement> --proposals <proposal>`)",
            code="not_screened", field="proposal",
        )
    return found


def _others(tree: dict[str, str]) -> dict[str, str]:
    """A skill folder's files but its SKILL.md — the file the edit replaces."""
    return {path: digest for path, digest in tree.items() if path != "SKILL.md"}


def prepare_validate(
    store: Store,
    refinement_ref: str,
    proposal_ref: str,
    *,
    n: int = MINIMUM,
    runs: int = MINIMUM,
    alpha: float = 0.05,
    concurrency: int = 1,
    proxy_url: str | None = None,
    proxy: Any = None,
    keep_private: Path | None = None,
) -> ValidatePlan:
    """A validation, checked before anything runs. Refused when what it would measure is no
    longer what was refined: the skill's file, its other files, or the training task.

    `proxy` is for a caller inside the proxy itself (a job), which knows it without asking."""
    from tuningfork.control import default_proxy
    from tuningfork.experiment.skill import text_sha256
    from tuningfork.ops import refine
    from tuningfork.ops.batch import prepare_batch
    from tuningfork.ops.experiments import load
    from tuningfork.ops.forks import prepare_to_end
    from tuningfork.runner import eval_record
    from tuningfork.specs import SpecError, hash_tree, load_spec, missing_files

    if n < MINIMUM:
        raise OperationError(f"at least {MINIMUM} forks an arm", code="bad_params", field="n")
    if runs < MINIMUM:
        raise OperationError(f"at least {MINIMUM} runs an arm", code="bad_params", field="runs")
    if not 0 < alpha < 1:
        raise OperationError("alpha is between 0 and 1", code="bad_params", field="alpha")
    if keep_private is not None:
        keep_private = Path(keep_private).resolve()
        if keep_private == store.home.resolve() or store.home.resolve() in keep_private.parents:
            raise OperationError(
                "the held-out runs' copy would be inside the recordings folder, where agents read",
                hint="keep it somewhere else", code="bad_params", field="keep_private",
            )
    refinement = refine.get(store, refinement_ref)
    refine.refuse_closed(refinement)
    proposal = refine.proposal(store, refinement.id, proposal_ref)
    screen = surviving_screen(store, refinement, proposal)

    checkpoint = load(store, refinement.run_id, str(refinement.step_idx))
    run = checkpoint.run
    launch = refine._launch_of(store, run)
    if launch is None or not launch.spec_path:
        raise OperationError("the refinement's run was not launched from a task spec", field="refinement")
    try:
        spec = load_spec(Path(launch.spec_path))
    except (OSError, SpecError) as error:
        raise OperationError(f"the task spec cannot be read: {error}", hint=launch.spec_path, field="spec") from None
    recorded = dict(launch.eval or {})
    try:
        training = spec.get(recorded.get("id"))
    except (KeyError, ValueError):
        raise OperationError(f"the spec no longer holds eval {recorded.get('id')!r}", field="spec") from None
    current = eval_record(training, spec.directory)
    if {k: v for k, v in current.items() if k != "spec_dir"} != {k: v for k, v in recorded.items() if k != "spec_dir"}:
        raise OperationError(
            f"the training task, {spec.skill_name}/{training.id}, has changed since the checkpoint's run: "
            "its prompt, files or check differ, so its runs would not measure the same thing",
            hint="refine again from a run of the task as it is now", code="drifted", field="spec",
        )
    source = refine.source_skill(store, run)
    if source is None:
        raise OperationError("the spec names no skill folder holding a SKILL.md", field="spec")
    if text_sha256(refine._text(source)) != refinement.data["baseline"]["sha256"]:
        raise OperationError(
            f"{source} has changed since the refinement started: the edit was made to the skill as it was",
            hint="refine again from a run of the skill as it is now", code="drifted", field="refinement",
        )
    snapshot = (run.parity or {}).get("skill_snapshot") or {}
    if snapshot and _others(hash_tree([source.parent])) != _others(snapshot):
        raise OperationError(
            f"the skill folder's other files have changed since the checkpoint's run ({source.parent})",
            hint="refine again from a run of the skill as it is now", code="drifted", field="refinement",
        )
    from tuningfork.ops.private import check_private_base

    held = spec.held_out_evals()
    for entry in held:
        missing = missing_files(entry, spec.directory)
        if missing:
            raise OperationError(f"held-out eval {entry.id} names input files that are not there", field="spec")
    private_base = check_private_base(store.home) if held else None
    warnings = []
    if not held:
        warnings.append("the spec holds no held-out eval: nothing will test the edit on a task it was not "
                        "made for, and the gate will refuse approval")
    if n < 20 or runs < 20:
        warnings.append("at a few runs an arm only a large difference shows: the not-worse checks are low-powered")
    held_recorded = sorted(store.held_out_tasks() & {f"{spec.skill_name}/{entry.id}" for entry in held})

    proxy_url = (proxy_url or (proxy.url if proxy is not None else None) or default_proxy()).rstrip("/")
    base_url = f"{proxy_url}/v1"
    variant = Path(proposal.data["variant_path"])
    baseline = refinement.folder / "baseline.md"
    forward = prepare_to_end(
        store, run.id, str(refinement.step_idx), n=n, concurrency=concurrency, variant=variant,
        alpha=alpha, base_url=base_url, proxy=proxy,
    )
    # Every part runs the harness and model the checkpoint's run had: the forks' own.
    harness, model = forward.original.harness, forward.original.model
    training_plans = {
        arm: prepare_batch(
            store, spec, eval_id=training.id, n=runs, model=model, harness=harness, skill_file=path,
            concurrency=concurrency, base_url=base_url, proxy=forward.original.proxy,
        )
        for arm, path in (("original", baseline), ("edit", variant))
    }
    for found in [*forward.original.warnings, *(note.message for note in training_plans["edit"].notes
                                                if note.level == "warning")]:
        if found not in warnings:
            warnings.append(found)
    return ValidatePlan(
        refinement=refinement, proposal=proposal, spec=spec, training=training, held_out=held,
        n=n, runs=runs, alpha=alpha, concurrency=concurrency, harness=harness, model=model,
        proxy_url=proxy_url, forward=forward, training_plans=training_plans, screen=screen,
        held_out_recorded=held_recorded, private_base=private_base, keep_private=keep_private,
        max_output=forward.original.proxy.raw.get("max_output"), warnings=warnings,
    )


def _verdict(store: Store, run_id: str | None, passed: bool | None, exclusion: str | None,
             timed_out: bool | None = None) -> dict[str, Any]:
    run = store.get_run(run_id) if run_id else None
    if timed_out is None:
        timed_out = bool(((run.result if run is not None else None) or {}).get("timed_out"))
    return {"run_id": run_id, "passed": passed, "exclusion": exclusion, "timed_out": bool(timed_out),
            "server_id": run.server_id if run is not None else None}


def _forks(store: Store, arm: Any) -> list[dict[str, Any]]:
    return [_verdict(store, run_id, passed, why) for _, run_id, passed, why in arm.forks]


def _batch(store: Store, report: Any) -> list[dict[str, Any]]:
    return [
        _verdict(store, outcome.run_id, outcome.passed if outcome.checked else None, outcome.exclusion,
                 outcome.timed_out)
        for outcome in report.outcomes
    ]


@dataclass(slots=True)
class ValidateResult:
    record: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """What a job says it found: the forward and training counts, and only that each
        held-out task ran — its verdicts are the person's, in the refinement."""
        from tuningfork.analysis.validation import count

        found = self.record
        planned = found.get("planned") or {}

        def counts(name: str) -> dict[str, Any] | None:
            part = found.get(name)
            if not part:
                return None
            return {arm: count(part.get(arm) or [], int(planned.get(name) or 0)).to_json() for arm in ("original", "edit")}

        return {
            "validation": found["id"],
            "refinement": found["refinement"],
            "proposal": found["proposal"],
            "status": found["status"],
            "complete": found["complete"],
            "forward": counts("forward"),
            "training": counts("training"),
            "held_out": [entry["eval"] for entry in found.get("held_out") or []],
            "note": "the held-out tasks' verdicts are in the refinement, for the person: "
            f"`tuningfork refine show {found['refinement'][:8]}`",
        }


def execute_validate(
    store: Store,
    plan: ValidatePlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ValidateResult:
    from tuningfork.analysis.validation import RULES
    from tuningfork.ops.batch import execute_batch
    from tuningfork.ops.forks import execute_to_end_plan
    from tuningfork.ops.private import run_verdict_only
    from tuningfork.ops.refine import _replace

    vid = uuid.uuid4().hex
    path = plan.refinement.folder / FOLDER / f"{vid}.json"
    tag = {"refinement": plan.refinement.id, "validation": vid}
    record: dict[str, Any] = {
        "id": vid, "refinement": plan.refinement.id, "proposal": plan.proposal.id,
        "created_at": time.time(), "task": plan.task_id, "harness": plan.harness, "model": plan.model,
        # The process running it: a record still "running" after that process died was cut off.
        "pid": os.getpid(),
        "alpha": plan.alpha, "minimum": MINIMUM,
        "planned": {"forward": plan.n, "training": plan.runs, "held_out": plan.runs},
        "screen": plan.screen, "held_out_recorded": plan.held_out_recorded, "rules": list(RULES),
        "held_out_planned": [entry.id for entry in plan.held_out],
        "forward": None, "training": None, "held_out": [], "complete": False, "status": "running",
    }
    _replace(path, record)
    variant = Path(plan.proposal.data["variant_path"])
    baseline = plan.refinement.folder / "baseline.md"
    try:
        emit(progress, "validation", part="forward", id=vid)
        ab = execute_to_end_plan(store, plan.forward, progress=progress, cancel=cancel, tag=tag)
        record["forward"] = {
            "original": _forks(store, ab.original), "edit": _forks(store, ab.variant),
            "experiments": {"original": ab.original.experiment.id, "edit": ab.variant.experiment.id},
        }
        _replace(path, record)

        training: dict[str, Any] = {"runs": {}}
        for arm in ("original", "edit"):
            emit(progress, "validation", part="training", id=vid, arm=arm)
            report = execute_batch(store, plan.training_plans[arm], progress=progress, cancel=cancel)
            training[arm] = _batch(store, report)
            training["runs"][arm] = report.run_ids
        record["training"] = training
        _replace(path, record)

        for entry in plan.held_out:
            arms: dict[str, Any] = {"eval": entry.id}
            for arm, skill in (("original", baseline), ("edit", variant)):
                emit(progress, "validation", part="held_out", id=vid, eval=entry.id, arm=arm)
                keep = plan.keep_private / f"{entry.id}-{arm}" if plan.keep_private else None
                arms[arm] = [verdict.to_json() for verdict in run_verdict_only(
                    store, plan.spec, entry, n=plan.runs, harness=plan.harness, model=plan.model,
                    proxy_url=plan.proxy_url, skill_file=skill, concurrency=plan.concurrency, keep=keep,
                    max_output=plan.max_output, label=f"held-out {entry.id}, {arm}", progress=progress, cancel=cancel,
                )]
            record["held_out"].append(arms)
            _replace(path, record)
        record["complete"], record["status"] = True, "done"
    except Cancelled as stopped:
        record["status"] = "interrupted" if stopped.reason == "shutdown" else "cancelled"
        record["finished_at"] = time.time()
        _replace(path, record)
        raise Cancelled(stopped.reason, partial=ValidateResult(record)) from None
    except BaseException as error:
        record["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        record["finished_at"] = time.time()
        _replace(path, record)
        raise
    record["finished_at"] = time.time()
    _replace(path, record)
    return ValidateResult(record)


def validations(store: Store, refinement_ref: str) -> list[dict[str, Any]]:
    """A refinement's validations, newest first, as recorded: for the person's views."""
    from tuningfork.ops import refine

    from tuningfork.ops.private import _alive

    refinement = refine.get(store, refinement_ref)
    folder = refinement.folder / FOLDER
    found = [json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("*.json")] if folder.is_dir() else []
    for record in found:
        # Killed mid-run — a `serve` stopped hard — it never said how it ended: it was cut off.
        if record.get("status") == "running" and record.get("pid") and not _alive(int(record["pid"])):
            record["status"] = "interrupted"
    return sorted(found, key=lambda record: record.get("created_at", 0), reverse=True)


def target_matches(refinement: Any) -> bool | None:
    """Whether the skill's own file, where an approved edit is written, still holds the text the
    refinement started from (line endings aside); None when it named no such file."""
    from tuningfork.experiment.skill import text_sha256
    from tuningfork.ops import refine

    path = (refinement.data.get("source") or {}).get("path")
    if not path:
        return None
    try:
        return text_sha256(refine._text(Path(path))) == refinement.data["baseline"]["sha256"]
    except OSError:
        return False


def gate_for(store: Store, refinement: Any, record: dict[str, Any]) -> Any:
    """The gate a validation meets now: its record, the skill's file as it is, the proposal's
    latest screen, and where the refinement stands (`analysis.validation.judge`)."""
    from tuningfork.analysis.validation import judge

    return judge(
        record, target_matches=target_matches(refinement),
        refinement_open=refinement.status == "open", approved=refinement.status == "approved",
        latest_screen=latest_screen(store, refinement, record.get("proposal") or ""),
    )


def validation_views(store: Store, refinement: Any, *, held_out: bool = False) -> list[dict[str, Any]]:
    """A refinement's validations, newest first. With `held_out`, for the person: every verdict,
    and the gate each meets now. Without, for anyone: what a job would say of each — the
    forward and training counts, and the held-out tasks by id."""
    from tuningfork.ops.approve import approval

    found = validations(store, refinement.id)
    approved = approval(store, refinement.id)
    if held_out:
        views = []
        for record in found:
            if approved and approved.get("validation") == record["id"]:
                # Judged again now, it would fail: the file holds the edit, the refinement is
                # approved. It is shown with the gate it met when it was approved.
                views.append({**record, "gate": approved["evidence"]["gate"], "counts": counts(record),
                              "approved": True})
            else:
                views.append({**record, "gate": gate_for(store, refinement, record).to_json(), "counts": counts(record)})
        return views
    return [{**ValidateResult(record).to_json(), "approved": bool(approved and approved.get("validation") == record["id"])}
            for record in found]


def counts(record: dict[str, Any]) -> dict[str, Any]:
    """Each part's arms as counted (`analysis.validation.COUNTING`), with their intervals: for
    the person's views, which draw them rather than count again."""
    from tuningfork.analysis.validation import count

    planned = record.get("planned") or {}

    def arms(part: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
        if not part:
            return None
        return {arm: count(part.get(arm) or [], int(planned.get(name) or 0)).to_json() for arm in ("edit", "original")}

    return {
        "forward": arms(record.get("forward"), "forward"),
        "training": arms(record.get("training"), "training"),
        "held_out": [{"eval": entry["eval"], **(arms(entry, "held_out") or {})} for entry in record.get("held_out") or []],
    }
