"""What a target would count, shown before any sample is drawn.

A target is declared before sampling. This sets it against the behaviours already seen at the
step — the run's own, other runs of the task at the same step, and every sample drawn there —
each marked as counting or not, and each with how the recorded runs that took it fared: a
check of the declaration, so a person sees what it means before trusting it. A target a
screen can move counts the passing runs' answers and not the failing runs'. Never a result:
these were seen before, and are not the samples the test will be run on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Store

#: Where a behaviour was seen, in the order a page lists them.
SOURCES = ("this run", "other runs", "samples")


@dataclass(slots=True)
class Seen:
    outcome: Any
    #: How often it was seen, by where: `SOURCES`.
    counts: dict[str, int] = field(default_factory=dict)
    #: Whether the target counts it; None when no target was given.
    matches: bool | None = None
    #: How the recorded runs that took it fared — {"passed": n, "failed": m} — over those with
    #: a verdict that counts. A sample has none.
    verdicts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            **self.outcome.to_json(),
            "seen": self.counts,
            "count": sum(self.counts.values()),
            "matches": self.matches,
            "verdicts": self.verdicts,
        }


@dataclass(slots=True)
class SeenBehaviours:
    run_id: str
    step_idx: int
    #: The tools the step offered the model, for a target naming one.
    tools: list[str]
    normalizers: list[str]
    target: str | None
    behaviours: list[Seen]
    #: The recorded runs with a verdict that counts: {"passed": P, "failed": F}.
    judged: dict[str, int] = field(default_factory=dict)
    #: The step was named by the tools called there, and other runs are compared at each
    #: one's own first call of any of them — not at the same index, which is the write in
    #: one run and the final reply in another. None: compared by index.
    aligned: list[str] | None = None
    #: Other runs of the task with no such step, by verdict ("passed", "failed", "unjudged"):
    #: a run that wrote by a tool the name leaves out, or never wrote.
    not_aligned: dict[str, int] = field(default_factory=dict)
    #: The model's choice of tool at the step as a number, where the recorded answers were
    #: scored (`ops.tokens.step_choice`): each scored run's, and the mean chance of each tool
    #: over them. `scored` is how many of the `of` recorded runs compared have one.
    choices: dict[str, Any] = field(default_factory=dict)

    @property
    def recorded(self) -> int:
        """Answers seen at the step, from every source."""
        return sum(sum(entry.counts.values()) for entry in self.behaviours)

    def split(self) -> dict[str, Any] | None:
        """With a target: of the passing runs' answers, how many it counts, and of the failing
        runs' — whether it separates them, which is what a screen needs. `kind` is universal
        when every answer seen satisfies it (an edit could only lower it), unseen when none
        does, mixed otherwise."""
        if self.target is None:
            return None
        counted = [entry for entry in self.behaviours if entry.matches]
        passing = {"counted": sum(entry.verdicts.get("passed", 0) for entry in counted), "of": self.judged.get("passed", 0)}
        failing = {"counted": sum(entry.verdicts.get("failed", 0) for entry in counted), "of": self.judged.get("failed", 0)}
        if not self.behaviours:
            kind = "nothing seen"
        elif len(counted) == len(self.behaviours):
            kind = "universal"
        elif not counted:
            kind = "unseen"
        else:
            kind = "mixed"
        separates = None
        if passing["of"] and failing["of"]:
            separates = passing["counted"] / passing["of"] > failing["counted"] / failing["of"]
        return {"kind": kind, "passing": passing, "failing": failing, "separates": separates}

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step_idx,
            "tools": self.tools,
            "normalizers": self.normalizers,
            "target": self.target,
            "behaviours": [seen.to_json() for seen in self.behaviours],
            "judged": self.judged,
            "recorded": self.recorded,
            "split": self.split(),
            "aligned": self.aligned,
            "not_aligned": self.not_aligned,
            "choices": self.choices,
        }


def offered_tools(request: dict[str, Any]) -> list[str]:
    """The tools a request offered, by name, in either wire format."""
    names = []
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            name = (tool.get("function") or {}).get("name") or tool.get("name")
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def verdict_from(result: dict[str, Any] | None) -> str | None:
    """How a run fared, from its stored result, where its check's verdict counts: "passed",
    "failed", or None."""
    result = result or {}
    if not result.get("usable", True) or result.get("passed") is None:
        return None
    return "passed" if result["passed"] else "failed"


def verdict_of(run: Any) -> str | None:
    """How a run fared, where its check's verdict counts: "passed", "failed", or None."""
    return verdict_from(getattr(run, "result", None))


def seen_behaviours(
    store: Store,
    run: str,
    step: str,
    *,
    normalizers: Sequence[str] = (),
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    base: bool = False,
    runs: int = 200,
) -> SeenBehaviours:
    """The behaviours seen at a step, each marked as the target would count it: one described
    (`target_rules`), a pattern (`target_match`), or the run's own behaviour (`base`).

    A step named by what the run did there — a tool ("write_file", or "write_file|execute_code"
    for whichever came first), a file's first change (`changed:<path>`), a finding's making
    (`made:<words>`) — is compared across runs at each run's own such step: the same point in
    every run, whatever path each took to it. A step named by its index is compared at that
    index, which is the write in one run and the final reply in another."""
    from tuningfork.analysis.distribution import target_by_key, target_by_pattern, target_by_rules
    from tuningfork.analysis.normalize import outcome_for, recorded_outcome, validate_normalizers
    from tuningfork.experiment.forking import forking_outcome
    from tuningfork.ops.experiments import find_step, load, named_by_doing, step_tools

    try:
        lens = validate_normalizers(normalizers)
    except ValueError as error:
        raise OperationError(str(error), field="normalizers") from None
    from tuningfork.ops.tokens import step_choice

    checkpoint = load(store, run, step)
    aligned = step if named_by_doing(step) else None
    seen: dict[str, Seen] = {}
    judged: dict[str, int] = {}
    not_aligned: dict[str, int] = {}
    compared = 0
    scored: list[dict[str, Any]] = []

    def note(outcome: Any, source: str, verdict: str | None = None) -> None:
        if outcome.is_error:
            return
        entry = seen.setdefault(outcome.key, Seen(outcome))
        entry.counts[source] = entry.counts.get(source, 0) + 1
        if verdict:
            entry.verdicts[verdict] = entry.verdicts.get(verdict, 0) + 1
            judged[verdict] = judged.get(verdict, 0) + 1

    def choice(run_id: str, recorded: Any, request: dict[str, Any] | None, verdict: str | None) -> None:
        nonlocal compared
        compared += 1
        found = step_choice(store, recorded, request)
        if found is not None:
            scored.append({"run": run_id, "verdict": verdict, **found})

    own = recorded_outcome(store.response(checkpoint.step), checkpoint.step.error, lens)
    note(own, "this run", verdict_of(checkpoint.run))
    choice(checkpoint.run.id, checkpoint.step, checkpoint.request, verdict_of(checkpoint.run))
    if checkpoint.run.task_id:
        for other in store.runs(limit=runs, task_id=checkpoint.run.task_id):
            if other.id == checkpoint.run.id or other.is_thread or other.is_fork:
                continue
            index = find_step(store, other, aligned) if aligned else checkpoint.step.idx
            if index is None:
                key = verdict_of(other) or "unjudged"
                not_aligned[key] = not_aligned.get(key, 0) + 1
                continue
            found = store.step(other.id, index)
            if found is not None:
                note(recorded_outcome(store.response(found), found.error, lens), "other runs", verdict_of(other))
                choice(other.id, found, None, verdict_of(other))
    for experiment in store.experiments(run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, limit=50):
        read = forking_outcome if experiment.kind == "forking-tokens" else outcome_for
        for sample in store.samples(experiment.id):
            if not sample.error:
                note(read(store.sample_response(sample), lens), "samples")

    try:
        target = (
            target_by_rules(target_rules) if target_rules
            else target_by_pattern(target_match) if target_match
            else target_by_key(own.key, f"the run's own behaviour: {own.label}") if base
            else None
        )
    except (ValueError, re.error) as error:
        raise OperationError(str(error), field="target_rules" if target_rules else "target_match") from None
    if target is not None:
        for entry in seen.values():
            entry.matches = target.matches(entry.outcome)
    return SeenBehaviours(
        run_id=checkpoint.run.id,
        step_idx=checkpoint.step.idx,
        tools=offered_tools(checkpoint.request),
        normalizers=lens,
        target=target.description if target else None,
        behaviours=sorted(seen.values(), key=lambda entry: -sum(entry.counts.values())),
        judged=judged,
        # The tools the step was named by; a step named by a file's change or a finding's
        # making is its reference whole.
        aligned=(step_tools(aligned) or [aligned]) if aligned else None,
        not_aligned=not_aligned,
        choices=choices_summary(scored, compared),
    )


def choices_summary(scored: Sequence[dict[str, Any]], compared: int) -> dict[str, Any]:
    """The scored runs' tool choices, and the mean chance of each tool over them."""
    if not scored:
        return {"scored": 0, "of": compared}
    totals: dict[str, float] = {}
    for found in scored:
        for name, p in (found.get("chance") or {}).items():
            totals[name] = totals.get(name, 0.0) + p
    mean = {name: round(total / len(scored), 4) for name, total in sorted(totals.items(), key=lambda item: -item[1])}
    return {"scored": len(scored), "of": compared, "chance": mean, "runs": list(scored)}
