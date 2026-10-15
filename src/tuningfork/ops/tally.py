"""What the checks said, counted: across a task's failed runs, or across the failed forks of each arm
of a screen run to the end, how often each thing a check printed came up.

A reader who opens failed runs one at a time aims at the first failure read, not the most common
one: an agent did, in meeting-pb (findings §31.35), at the third most common. The tally reads every
failed run's check output at once. Each line a check prints is one finding — a check that names
what it found wrong, one thing a line, gets one count a thing — and lines that differ only in
their numbers ("owners 0.5", "owners 0") or their colours ("#333333 on #EA580C") are one finding.
A finding is counted once a run, so its count is the number of failed runs that said it, with its
Wilson interval over the failed runs.

A task's runs are tallied by condition — harness, model, skill version, server — as their pass
rates are (`analysis.conditions`): a finding pooled across wordings describes neither. A screen's
forks are tallied by arm: the original's beside each proposal's, so what a proposal left failing
shows next to what the original fails. A held-out task's runs are a person's evidence; the front
ends an agent reaches pass them in `hidden`, and they are refused.

Where the check was run along a task's runs (`ops.checks.check_along`), each finding also says
which call made it in each run, the passing ones included (`made_along`). One every run made, and
the passing runs repaired later, is a mistake no passing run's answer shows the fix for
(`shared_mistakes`): comparing passing with failing runs points at the repair (findings §31.42).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tuningfork.analysis.stats import rate
from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store

#: Numbers, as a check prints scores, counts and ratios: masked so that one finding is one key.
NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: Hex colours, masked whole before the numbers: "#333333 on #EA580C" and "#2D2D2D on #EA580C" are
#: one finding. Three digits are a colour only with a letter among them; "#123" stays a number.
COLOR = re.compile(r"#(?:[0-9A-Fa-f]{8}|[0-9A-Fa-f]{6}|(?=[0-9]*[A-Fa-f])[0-9A-Fa-f]{3})\b")
#: A finding is read up to this many characters: a check's line, not its whole report.
MAX_LINE = 200


def finding_key(line: str) -> str:
    """The finding a line states: its colours and numbers masked, its spaces collapsed."""
    return re.sub(r"\s+", " ", NUMBER.sub("#", COLOR.sub("<color>", line.strip())))[:MAX_LINE]


def finding_lines(output: str | None) -> list[str]:
    """Each line one check's output says something wrong in, as printed, in order: a check that
    names what it found wrong prints one finding a line."""
    lines = []
    for line in str(output or "").splitlines():
        text = line.strip()
        if text and text.lower() != "ok":
            lines.append(text)
    return lines


def findings(output: str | None) -> dict[str, str]:
    """Each distinct finding in one check's output, by its key, with the line as printed first."""
    found: dict[str, str] = {}
    for text in finding_lines(output):
        found.setdefault(finding_key(text), text[:MAX_LINE])
    return found


@dataclass(slots=True)
class Finding:
    key: str
    #: The line as one check printed it, numbers and all.
    example: str
    run_ids: list[str]
    #: The failed runs (or forks) counted: the denominator of the rate.
    of: int
    #: Where the check was run along the condition's runs (`ops.checks.check_along`): which call
    #: made this finding in each, passing runs included (`made_along`).
    made: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "finding": self.key,
            "example": self.example,
            "runs": len(self.run_ids),
            "rate": rate(len(self.run_ids), self.of).to_json(),
            "run_ids": self.run_ids,
            "made": self.made,
            "made_text": made_text(self.made) if self.made else None,
        }


def made_along(runs: Iterable[Run]) -> dict[str, dict[str, Any]]:
    """For each finding, over the runs whose check was run along them — passing and failing — in
    how many a call made it, at which step of each, how many still had it at the end, and how many
    of those runs passed: a finding every run made, the passing ones too, is a mistake they all
    made and some repaired later — where no passing run's answer shows the fix (findings §31.42)."""
    from tuningfork.ops.checks import ALONG, made_by_calls

    checked = [run for run in runs if (run.result or {}).get(ALONG)]
    found: dict[str, dict[str, Any]] = {}
    for run in checked:
        first: dict[str, dict[str, Any]] = {}
        for entry in made_by_calls(run.result[ALONG]):
            if entry["made_by"] is None:
                continue
            seen = first.setdefault(entry["finding"], {"step": entry["made_by"], "at_end": False})
            seen["step"] = min(seen["step"], entry["made_by"])
            seen["at_end"] = seen["at_end"] or entry["at_end"]
        for key, seen in first.items():
            stat = found.setdefault(key, {"checked": len(checked), "runs": [], "steps": [], "repaired": 0,
                                          "passing": 0})
            stat["runs"].append(run.id)
            stat["steps"].append(seen["step"])
            stat["repaired"] += 0 if seen["at_end"] else 1
            stat["passing"] += 1 if (run.result or {}).get("passed") else 0
    for stat in found.values():
        stat["every_run"] = len(stat["runs"]) == stat["checked"] and stat["checked"] >= 2
    return found


def made_steps(made: dict[str, Any]) -> str:
    """The steps whose calls made a finding, with how many runs each: "step 2 in 6, step 1 in 2"."""
    from collections import Counter

    counts = Counter(made.get("steps") or [])
    return ", ".join(f"step {step} in {count}" for step, count in sorted(counts.items()))


def made_text(made: dict[str, Any]) -> str:
    """Where the check along the runs says a finding was made, in a sentence."""
    runs = len(made.get("runs") or [])
    text = f"made by a call in {runs} of {made.get('checked')} run(s) the check was run along ({made_steps(made)})"
    if made.get("repaired"):
        text += f"; gone by the end in {made['repaired']}"
    if made.get("every_run") and made.get("passing"):
        text += f" — every run made it, the {made['passing']} that passed too, and repaired it later"
    return text


def finding_words(key: str) -> str:
    """The longest stretch of a finding's key that no number or colour was masked in — words every
    line of the finding contains, whatever its numbers: what `made:<words>` names a step by."""
    pieces = [piece.strip(" ,.:;-") for piece in re.split(r"#|<color>", key)]
    return max(pieces, key=len, default="") or key


def shared_mistakes(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The findings of a task's tally that every run made — the passing ones too, repairing them
    later — where the check was run along the runs: no passing run's answer shows the fix."""
    found = []
    for group in report.get("groups") or []:
        for finding in group.get("findings") or []:
            made = finding.get("made") or {}
            if made.get("every_run") and made.get("passing"):
                found.append(finding)
    return found


@dataclass(slots=True)
class TallyGroup:
    #: What the runs share: a condition of the task, or an arm of a screen.
    label: str
    judged: int
    failed: int
    #: Failed runs whose check printed nothing to count.
    silent: int
    findings: list[Finding] = field(default_factory=list)
    #: For a task's runs: the condition, as `inspect.condition_json` gives it.
    condition: dict[str, Any] | None = None
    #: For a screen's forks: the arm's name — "original", or a proposal's id.
    arm: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "judged": self.judged,
            "failed": self.failed,
            "passed": rate(self.judged - self.failed, self.judged).to_json() if self.judged else None,
            "silent": self.silent,
            "findings": [finding.to_json() for finding in self.findings],
            "condition": self.condition,
            "arm": self.arm,
        }


@dataclass(slots=True)
class Tally:
    #: "task": a task's recorded runs; "screen": the forks of a screen (or an A/B) run to the end.
    scope: str
    task_id: str | None
    groups: list[TallyGroup]
    #: For a screen: its group id, and the checkpoint run it started from.
    screen: str | None = None
    run_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "task_id": self.task_id,
            "screen": self.screen,
            "run_id": self.run_id,
            "groups": [group.to_json() for group in self.groups],
        }


def counted(label: str, judged: int, failed: Sequence[tuple[str, str | None]], **where: Any) -> TallyGroup:
    """One group's findings: each failed run's (id, check output), counted; the most common first,
    ties in the order first seen."""
    by_key: dict[str, Finding] = {}
    silent = 0
    for run_id, output in failed:
        said = findings(output)
        if not said:
            silent += 1
        for key, example in said.items():
            found = by_key.get(key)
            if found is None:
                by_key[key] = Finding(key=key, example=example, run_ids=[run_id], of=len(failed))
            else:
                found.run_ids.append(run_id)
    ordered = sorted(by_key.values(), key=lambda finding: -len(finding.run_ids))
    return TallyGroup(label=label, judged=judged, failed=len(failed), silent=silent, findings=ordered, **where)


def judged_runs(runs: Iterable[Run]) -> list[Run]:
    """Runs with a verdict that counts, as a pass rate counts them (`analysis.conditions.pass_rate`):
    a run that timed out, was edited or whose skill changed describes neither wording."""
    return [run for run in runs if run.result and run.result.get("usable") and run.result.get("passed") is not None]


def refuse_hidden_task(task_id: str | None, hidden: Iterable[str]) -> None:
    if task_id is not None and task_id in set(hidden):
        raise OperationError(
            f"{task_id} is a held-out task: its runs are the person's evidence, never an agent's",
            code="held_out", field="task",
        )


def tally_task(store: Store, task_id: str, *, hidden: Iterable[str] = ()) -> Tally:
    """What a task's failed runs' checks said, by condition, the most common finding first."""
    from tuningfork.analysis.conditions import conditions
    from tuningfork.ops.inspect import condition_json

    refuse_hidden_task(task_id, hidden)
    runs = store.runs(limit=100_000, task_id=task_id)
    if not runs:
        raise OperationError(f"no runs of task {task_id!r}", code="not_found", field="task")
    groups = []
    for condition in conditions(runs):
        judged = judged_runs(condition.runs)
        failed = [(run.id, run.result.get("check_output")) for run in judged if run.result.get("passed") is False]
        about = condition_json(condition)
        label = " · ".join(str(part) for part in (about["harness"], about["model"], about["skill_version"]) if part)
        group = counted(label, len(judged), failed, condition=about)
        along = made_along(judged)
        for finding in group.findings:
            finding.made = along.get(finding.key)
        groups.append(group)
    return Tally(scope="task", task_id=task_id, groups=groups)


def tally_screen(store: Store, reference: str, *, hidden: Iterable[str] = ()) -> Tally:
    """What the failed forks' checks said in each arm of a screen (or an A/B) run to the end: the
    original's beside each proposal's. `reference` is any of its arms' experiments."""
    from tuningfork.ops.screen import BASELINE

    found = store.get_experiment(reference)
    grouped = store.experiment_group(found.group_id if found is not None and found.group_id else reference)
    if found is None and grouped:
        found = grouped[0]
    if found is None:
        raise OperationError(f"no screen or experiment matching {reference!r}", code="not_found", field="screen")
    group_id = found.group_id or found.id
    experiments = grouped or [found]
    run = store.get_run(found.run_id) if found.run_id else None
    task_id = run.task_id if run is not None else None
    # A held-out task's experiment is refused before anything else is said of it.
    refuse_hidden_task(task_id, hidden)
    if found.settings.get("reach") != "end":
        raise OperationError(
            f"{found.id[:8]} drew its samples at a step, where no check runs: a tally needs forks run to the end "
            "(a screen to the end, or an A/B with --to-end)",
            code="bad_params", field="screen",
        )
    by_arm: dict[str, list[Any]] = {}
    if found.kind == "screen":
        from tuningfork.ops.screen import screen_arms

        # The original's forks it counts from an earlier screen of its refinement are its own here too.
        _, by_arm, _ = screen_arms(store, found.id)
    else:
        for experiment in experiments:
            by_arm.setdefault(experiment.arm or "original", []).append(experiment)
    planned = list((next((experiment.settings.get("screen") for experiment in experiments
                          if experiment.settings.get("screen")), None) or {}).get("arms") or [])
    names = [name for name in (BASELINE, *planned, *by_arm) if name in by_arm]
    groups = []
    for name in dict.fromkeys(names):
        samples = [sample for experiment in by_arm[name] for sample in store.samples(experiment.id)]
        kept = [sample for sample in samples if sample.exclusion is None and sample.passed is not None]
        failed = []
        for sample in kept:
            if sample.passed is False and sample.run_id:
                fork = store.get_run(sample.run_id)
                failed.append((sample.run_id, (fork.result or {}).get("check_output") if fork else None))
        label = "the original" if name == BASELINE else f"proposal {name[:8]}" if name != "variant" else "the variant"
        groups.append(counted(label, len(kept), failed, arm=name))
    return Tally(scope="screen", task_id=task_id, groups=groups, screen=group_id, run_id=found.run_id)
