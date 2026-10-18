"""Where a target holds along a task's runs: in which runs, made at which call, and whether that goes
with passing.

A check that names what it found wrong says where to look: the tally counts its findings, and run
along the runs it says which call made each (`ops.tally`, `ops.checks.check_along`). A check that
prints only its verdict says neither, and a step's behaviours (`ops.targets`) are read at one step,
which the reader has to know already. This asks one question of every run instead and lines the
answers up with the verdicts. The question is of either side of a run:

- what the model did: a target as a refinement declares one (`target_match`, `target_rules`), tried
  on the recorded answer at every step of every run — the steps whose answer satisfies it, and the
  first of them;
- what the run wrote: a file, by path or glob, whose text matches a pattern at least so many times
  (or that exists, with no pattern), tried on the workspace as the run started, after each call that
  changed it, and at its end, as the check is run along a run (`ops.inspect.workspace_states`) —
  whether it held at the start, the call that made it hold, the call that undid it, and whether it
  held at the end.

Counted by verdict over a task's runs, by condition as pass rates are: in how many passing and how
many failing runs it held — at some step, of an answer; at the end, of a file — with Fisher's exact
test; the steps where it was first seen or made; and the step reference that names that call in each
run, a tool's first call or a file's first change (`ops.experiments.find_step`), so that a refinement
and its screens start at each run's own such call. A diagnostic, never a verdict: it reads the
recordings and runs nothing. A held-out task's runs are the person's evidence: the front ends an
agent reaches pass them in `hidden`, and they are refused.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Iterable, Sequence

from tuningfork.analysis.stats import fisher_exact, p_short, rate
from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store

#: A split is said to go with passing or with failing under this p, as the step that best predicts
#: a verdict is named (`ops.inspect.verdict_split`): a lead to read, not a result.
ALPHA = 0.2
#: A file larger than this is not read for a pattern: a workspace's text, not a dataset's whole.
MAX_BYTES = 20_000_000
#: Steps listed per run where an answer satisfied the target.
STEPS_KEPT = 20


@dataclass(slots=True)
class Probe:
    """The question asked of every run: of its answers (`target`), or of its files (`path`)."""

    description: str
    #: A target over what the model did at a step (`analysis.distribution.Target`).
    target: Any = None
    normalizers: list[str] = field(default_factory=list)
    #: A file of the workspace, by its path there or a glob; with `pattern`, its text must match
    #: it at least `at_least` times — each line's ends are ^ and $ — and without, it must exist.
    path: str | None = None
    pattern: re.Pattern[str] | None = None
    at_least: int = 1

    @property
    def kind(self) -> str:
        return "file" if self.path is not None else "answer"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "path": self.path,
            "contains": self.pattern.pattern if self.pattern is not None else None,
            "at_least": self.at_least,
            "normalizers": self.normalizers,
        }


def make_probe(
    *,
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    path: str | None = None,
    contains: str | None = None,
    at_least: int = 1,
    normalizers: Sequence[str] = (),
) -> Probe:
    """The question, checked before any run is read: one of the model's answers or of the runs'
    files, never both, its pattern compiled."""
    from tuningfork.analysis.distribution import target_by_pattern, target_by_rules
    from tuningfork.analysis.normalize import validate_normalizers

    asks_answer = bool(target_match) or bool(target_rules)
    asks_file = bool(path and path.strip())
    if asks_answer == asks_file:
        raise OperationError(
            "ask one question: of the model's answers (target_match or target_rules), or of what the runs wrote "
            "(path, with contains for a pattern its text must match)",
            code="bad_params", field="path" if asks_answer else "target_match",
        )
    if target_match and target_rules:
        raise OperationError("declare the target once: target_match or target_rules", code="bad_params",
                             field="target_rules")
    if at_least < 1:
        raise OperationError("at_least counts matches in a file: 1 or more", code="bad_params", field="at_least")
    if asks_answer:
        if contains is not None or at_least != 1:
            raise OperationError("contains and at_least ask of a file: name it with path", code="bad_params",
                                 field="contains")
        try:
            lens = validate_normalizers(normalizers)
        except ValueError as error:
            raise OperationError(str(error), code="bad_params", field="normalizers") from None
        try:
            target = target_by_rules(target_rules) if target_rules else target_by_pattern(target_match or "")
        except (ValueError, re.error) as error:
            raise OperationError(str(error), code="bad_params",
                                 field="target_rules" if target_rules else "target_match") from None
        return Probe(description=target.description, target=target, normalizers=lens)
    where = (path or "").strip()
    if where.startswith("/"):
        raise OperationError(
            f"{where!r} is a path on a machine: name the file by its path in the workspace, as show_files lists it "
            "(clean.csv, out/report.json), or by a glob (*.json)",
            code="bad_params", field="path",
        )
    where = where.removeprefix("./")
    pattern = None
    if contains is not None:
        try:
            pattern = re.compile(contains, re.MULTILINE)
        except re.error as error:
            raise OperationError(f"contains is not a regular expression: {error}", code="bad_params",
                                 field="contains") from None
    if pattern is None:
        if at_least != 1:
            raise OperationError("at_least counts matches of contains: give the pattern", code="bad_params",
                                 field="at_least")
        description = f"{where} exists"
    else:
        times = "" if at_least == 1 else f" at least {at_least} times"
        description = f"{where} contains /{contains}/{times}"
    return Probe(description=description, path=where, pattern=pattern, at_least=at_least)


@dataclass(slots=True)
class RunHeld:
    """Where the probe held along one run."""

    run_id: str
    #: "passed", "failed", or None where the run's verdict does not count (`ops.targets.verdict_of`).
    verdict: str | None
    #: An answer's: at some step. A file's: at the end, what the run left.
    held: bool
    #: Where it was first seen: the first step whose answer satisfied the target, or the call that
    #: made the file hold — None for a file there from the start, or never made.
    step: int | None = None
    #: The tools the model called at `step`.
    tools: list[str] = field(default_factory=list)
    #: The step reference that names `step` in this run, where the group's reference does.
    reference: str | None = None
    #: An answer's: every step whose answer satisfied it, the first `STEPS_KEPT`.
    steps: list[int] = field(default_factory=list)
    #: A file's: whether it held at the start and at the end, and the call that first undid it.
    at_start: bool | None = None
    at_end: bool | None = None
    undone_by: int | None = None
    #: The file it held in, and how many times the pattern matched there, when it was made (or at
    #: the end, or the start).
    file: str | None = None
    count: int | None = None
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = {"run": self.run_id, "verdict": self.verdict, "held": self.held, "step": self.step,
                                 "tools": self.tools, "reference": self.reference}
        if self.at_start is None:
            found["steps"] = self.steps
        else:
            found.update({"at_start": self.at_start, "made_by": self.step, "undone_by": self.undone_by,
                          "at_end": self.at_end, "file": self.file, "count": self.count})
        if self.note:
            found["note"] = self.note
        return found


@dataclass(slots=True)
class HeldGroup:
    """One condition's runs, each with where the probe held."""

    label: str
    condition: dict[str, Any]
    runs: list[RunHeld]
    kind: str
    #: The step reference naming, in the most runs, the call where it was first seen or made:
    #: {"step": "changed:clean.csv", "names": 5, "of": 5}.
    reference: dict[str, Any] | None = None

    def judged(self, verdict: str) -> list[RunHeld]:
        return [found for found in self.runs if found.verdict == verdict]

    def split(self) -> dict[str, Any]:
        """In how many passing runs and how many failing runs it held, Fisher's exact test between
        them, and which way it goes when p is under `ALPHA`."""
        passing, failing = self.judged("passed"), self.judged("failed")
        held_passing = sum(1 for found in passing if found.held)
        held_failing = sum(1 for found in failing if found.held)
        p = goes_with = None
        if passing and failing:
            p = round(fisher_exact(held_passing, len(passing) - held_passing,
                                   held_failing, len(failing) - held_failing), 4)
            if p < ALPHA:
                goes_with = "passing" if held_passing / len(passing) > held_failing / len(failing) else "failing"
        return {
            "passing": rate(held_passing, len(passing)).to_json() if passing else None,
            "failing": rate(held_failing, len(failing)).to_json() if failing else None,
            "p": p,
            "goes_with": goes_with,
        }

    def seen_at(self) -> list[list[int]]:
        """[step, runs] for each step where it was first seen or made, the earliest first."""
        counts = Counter(found.step for found in self.runs if found.step is not None)
        return [[step, count] for step, count in sorted(counts.items())]

    def made(self) -> dict[str, Any] | None:
        """A file's: in how many runs a call made it hold — the passing ones among them — how many
        undid it before the end, and how many had it from the start."""
        if self.kind != "file":
            return None
        judged = [found for found in self.runs if found.verdict is not None]
        made = [found for found in judged if found.step is not None]
        passing = [found for found in made if found.verdict == "passed"]
        every_run = len(made) == len(judged) and len(judged) >= 2
        return {
            "runs": len(made),
            "of": len(judged),
            "passing": len(passing),
            "undone": sum(1 for found in made if found.undone_by is not None and not found.at_end),
            "from_start": sum(1 for found in judged if found.at_start),
            "every_run": every_run,
            # Every run made it and every passing run undid it before the end: a mistake they all made,
            # where no passing run's answer shows the fix (findings §31.42) — not what every run made and kept.
            "shared_mistake": every_run and bool(passing) and not any(found.at_end for found in passing),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "condition": self.condition,
            "judged": len(self.judged("passed")) + len(self.judged("failed")),
            "passed": len(self.judged("passed")),
            "failed": len(self.judged("failed")),
            "unjudged": sum(1 for found in self.runs if found.verdict is None),
            "split": self.split(),
            "seen_at": self.seen_at(),
            "reference": self.reference,
            "made": self.made(),
            "runs": [found.to_json() for found in self.runs],
        }


@dataclass(slots=True)
class WhereItHolds:
    task_id: str
    probe: Probe
    groups: list[HeldGroup]
    #: Runs passed over, and why: a run that kept no snapshots has no files to ask of.
    skipped: list[str] = field(default_factory=list)
    #: Asked of a screen's forks, arm by arm, rather than of the task's own runs: the screen.
    screen: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "screen": self.screen,
            "probe": self.probe.to_json(),
            "groups": [group.to_json() for group in self.groups],
            "skipped": self.skipped[:20],
            "skipped_more": max(0, len(self.skipped) - 20),
        }


# --- one run ----------------------------------------------------------------------------------


def answers_along(store: Store, run: Run, probe: Probe) -> RunHeld:
    """The steps of `run` whose recorded answer satisfies the probe's target."""
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.ops.experiments import tools_called
    from tuningfork.ops.targets import verdict_of

    steps: list[int] = []
    tools: list[str] = []
    for step in store.steps(run.id):
        outcome = recorded_outcome(store.response(step), step.error, probe.normalizers)
        if outcome.is_error or not probe.target.matches(outcome):
            continue
        if not steps:
            tools = [name for name in tools_called(store, step) if name]
        steps.append(step.idx)
    return RunHeld(run_id=run.id, verdict=verdict_of(run), held=bool(steps), step=steps[0] if steps else None,
                   tools=tools, steps=steps[:STEPS_KEPT])


class FileReader:
    """A probe's file read once a content: the same text in two snapshots, or in two runs, is one
    object in the store, and is counted once."""

    def __init__(self, store: Store, probe: Probe) -> None:
        from tuningfork.snapshots import Snapshots

        self.snapshots = Snapshots(store.home)
        self.probe = probe
        self.counts: dict[str, int | None] = {}

    def count(self, commit: str, entry: Any) -> int | None:
        """How many times the pattern matches the file's text; None when it is too large to read."""
        key = entry.sha or f"{commit}:{entry.path}"
        if key not in self.counts:
            if entry.size is not None and entry.size > MAX_BYTES:
                self.counts[key] = None
            else:
                raw = self.snapshots.blob(entry.sha) if entry.sha else self.snapshots.read(commit, entry.path)
                self.counts[key] = sum(1 for _ in self.probe.pattern.finditer(raw.decode("utf-8", "replace")))
        return self.counts[key]

    def holds(self, commit: str) -> tuple[bool, str | None, int | None, str | None]:
        """Whether the probe holds in a snapshot: (held, the file it held in, its count there, a note)."""
        matching = [entry for entry in self.snapshots.files(commit) if fnmatchcase(entry.path, self.probe.path)]
        if self.probe.pattern is None:
            return (bool(matching), matching[0].path if matching else None, None, None)
        note = None
        for entry in matching:
            found = self.count(commit, entry)
            if found is None:
                note = f"{entry.path} is over {MAX_BYTES // 1_000_000} MB: not read"
                continue
            if found >= self.probe.at_least:
                return (True, entry.path, found, None)
        return (False, None, None, note)


def files_along(store: Store, run: Run, reader: FileReader) -> RunHeld | None:
    """Whether the probe's file held at each state of `run`'s workspace: at the start, the call that
    made it hold, the call that undid it, and at the end. None for a run that kept no snapshots."""
    from tuningfork.ops.experiments import tools_called
    from tuningfork.ops.inspect import workspace_states
    from tuningfork.ops.targets import verdict_of
    from tuningfork.snapshots import SnapshotError

    states = workspace_states(store, run)
    if states is None:
        return None
    try:
        held_at = [reader.holds(commit) for _, commit in states]
    except SnapshotError:
        return None
    at_start, file, count, note = held_at[0]
    made_by = undone_by = None
    before = at_start
    for (index, _), (held, where, times, said) in zip(states[1:], held_at[1:]):
        note = note or said
        # A state at step k is what the call at step k - 1 left.
        if held and not before and made_by is None:
            made_by, file, count = index - 1, where, times
        elif before and not held and undone_by is None:
            undone_by = index - 1
        before = held
    at_end = held_at[-1][0]
    if file is None and at_end:
        file, count = held_at[-1][1], held_at[-1][2]
    tools: list[str] = []
    if made_by is not None:
        found = store.step(run.id, made_by)
        tools = [name for name in tools_called(store, found) if name] if found is not None else []
    return RunHeld(run_id=run.id, verdict=verdict_of(run), held=at_end, step=made_by, tools=tools,
                   at_start=at_start, at_end=at_end, undone_by=undone_by, file=file, count=count, note=note)


# --- the step that names it in each run -------------------------------------------------------


def name_the_step(store: Store, runs: Sequence[tuple[Run, RunHeld]], kind: str) -> dict[str, Any] | None:
    """The step reference that names, in the most runs, the call where the probe was first seen or
    made: the file's first change (`changed:<path>`), the first call of the tool called there, or
    of any of the tools called there across the runs (`write_file|execute_code`). Each run's entry
    gets the reference where it names its own call."""
    from tuningfork.ops.experiments import CHANGED, EITHER, find_step

    placed = [(run, found) for run, found in runs if found.step is not None]
    if not placed:
        return None
    candidates: list[str] = []
    if kind == "file":
        candidates += [f"{CHANGED}{found.file}" for _, found in placed if found.file]
    tools = [name for _, found in placed for name in found.tools]
    candidates += tools
    either = sorted(set(tools))
    if len(either) > 1:
        candidates.append(EITHER.join(either))
    names: dict[str, set[str]] = {}
    for reference in dict.fromkeys(candidates):
        names[reference] = {run.id for run, found in placed if find_step(store, run, reference) == found.step}
    # The most runs named; a tie goes to the candidate first listed: a file's change before a tool.
    best = max(names, key=lambda reference: len(names[reference]), default=None)
    if best is None or not names[best]:
        return None
    for run, found in placed:
        if run.id in names[best]:
            found.reference = best
    return {"step": best, "names": len(names[best]), "of": len(placed)}


# --- a task's runs ----------------------------------------------------------------------------


def where_it_holds(
    store: Store,
    task: str | None = None,
    *,
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    path: str | None = None,
    contains: str | None = None,
    at_least: int = 1,
    normalizers: Sequence[str] = (),
    limit: int = 200,
    hidden: Iterable[str] = (),
    screen: str | None = None,
) -> WhereItHolds:
    """Ask the probe of each of a task's runs — its own, not forks or threads, the newest `limit` —
    and count the answers by verdict, by condition. With `screen`, of a screen run to the end's
    forks instead, arm by arm (`screen_forks`): where none of a broken skill's own runs passed, a
    proposal's passing forks beside its failing ones are the comparison there is."""
    from tuningfork.analysis.conditions import conditions
    from tuningfork.ops.inspect import condition_json
    from tuningfork.ops.tally import refuse_hidden_task

    probe = make_probe(target_match=target_match, target_rules=target_rules, path=path, contains=contains,
                       at_least=at_least, normalizers=normalizers)
    if screen is not None:
        return where_in_screen(store, screen, probe, hidden=hidden)
    if not task:
        raise OperationError("give a task, or a screen run to the end, to ask of", code="bad_params", field="task")
    refuse_hidden_task(task, hidden)
    runs = [found for found in store.runs(limit=limit, task_id=task) if not found.is_thread and not found.is_fork]
    if not runs:
        raise OperationError(f"no runs of task {task!r}", code="not_found", field="task")
    reader = FileReader(store, probe) if probe.kind == "file" else None
    asked: dict[str, RunHeld] = {}
    skipped = []
    for run in runs:
        found = answers_along(store, run, probe) if reader is None else files_along(store, run, reader)
        if found is None:
            skipped.append(f"run {run.id[:8]}: kept no snapshots of its workspace")
            continue
        asked[run.id] = found
    groups = []
    for condition in conditions([run for run in runs if run.id in asked]):
        about = condition_json(condition)
        label = " · ".join(str(part) for part in (about["harness"], about["model"], about["skill_version"]) if part)
        members = [(run, asked[run.id]) for run in condition.runs if run.id in asked]
        group = HeldGroup(label=label or "runs", condition=about, runs=[found for _, found in members],
                          kind=probe.kind)
        group.reference = name_the_step(store, members, probe.kind)
        groups.append(group)
    return WhereItHolds(task_id=task, probe=probe, groups=groups, skipped=skipped)


def screen_forks(store: Store, reference: str, *, hidden: Iterable[str] = ()) -> tuple[str, str, list[tuple[str, str, list[Run]]]]:
    """A screen run to the end, by any of its arms' experiments or its group: its task, its group,
    and each arm's forks — (name, label, the fork runs), the original's first, counting those of an
    earlier screen of its refinement it counted (`ops.screen.screen_arms`)."""
    from tuningfork.ops.screen import BASELINE, screen_arms
    from tuningfork.ops.tally import refuse_hidden_task

    group_id, by_arm, planned = screen_arms(store, reference)
    first = next(arm for arms in by_arm.values() for arm in arms)
    if first.settings.get("reach") != "end":
        raise OperationError(
            f"{reference[:8]} drew its samples at a step, where nothing ran to the end: ask of a screen run to the end",
            code="bad_params", field="screen",
        )
    run = store.get_run(first.run_id) if first.run_id else None
    task_id = run.task_id if run is not None else None
    refuse_hidden_task(task_id, hidden)
    order = [name for name in dict.fromkeys([BASELINE, *(planned.get("arms") or []), *by_arm]) if name in by_arm]
    arms = []
    for name in order:
        runs = [store.get_run(sample.run_id) for experiment in by_arm[name] for sample in store.samples(experiment.id)
                if sample.run_id]
        label = "the original's forks" if name == BASELINE else f"proposal {name[:8]}'s forks"
        arms.append((name, label, [found for found in runs if found is not None]))
    return task_id or "", group_id, arms


def where_in_screen(store: Store, reference: str, probe: Probe, *, hidden: Iterable[str] = ()) -> WhereItHolds:
    """The probe asked of a screen's forks, each arm a group: within an arm the edit and the prefixes
    are shared, so what its passing forks' files or answers have that its failing forks' lack is what
    the check saw — the comparison a task none of whose own runs passed does not offer."""
    task_id, group_id, arms = screen_forks(store, reference, hidden=hidden)
    reader = FileReader(store, probe) if probe.kind == "file" else None
    groups, skipped = [], []
    for name, label, runs in arms:
        members = []
        for run in runs:
            found = answers_along(store, run, probe) if reader is None else files_along(store, run, reader)
            if found is None:
                skipped.append(f"fork {run.id[:8]}: kept no snapshots of its workspace")
                continue
            members.append((run, found))
        group = HeldGroup(label=label, condition={"arm": name, "screen": group_id}, runs=[found for _, found in members],
                          kind=probe.kind)
        group.reference = name_the_step(store, members, probe.kind)
        groups.append(group)
    return WhereItHolds(task_id=task_id, probe=probe, groups=groups, skipped=skipped, screen=group_id)


def passing_screens(store: Store, task: str, *, limit: int = 2000) -> list[dict[str, Any]]:
    """The screens run to the end from this task's runs whose forks passed its check: each with its
    id (an experiment of it) and its passing and judged forks, the most passes first. Where none of
    the task's own runs passed, these hold the passing runs there are."""
    tasks: dict[str, str | None] = {}
    found: dict[str, dict[str, Any]] = {}
    for experiment in store.experiments(kind="screen", limit=limit):
        if experiment.settings.get("reach") != "end" or not experiment.group_id or not experiment.run_id:
            continue
        if experiment.run_id not in tasks:
            run = store.get_run(experiment.run_id)
            tasks[experiment.run_id] = run.task_id if run is not None else None
        if tasks[experiment.run_id] != task:
            continue
        entry = found.setdefault(experiment.group_id, {"screen": experiment.id, "passed": 0, "judged": 0})
        for sample in store.samples(experiment.id):
            if sample.exclusion is None and sample.passed is not None:
                entry["judged"] += 1
                entry["passed"] += 1 if sample.passed else 0
    return sorted((entry for entry in found.values() if entry["passed"]), key=lambda entry: -entry["passed"])


# --- in words -----------------------------------------------------------------------------------


def rate_words(value: dict[str, Any] | None, what: str) -> str:
    """"3 of 7 failing (16–75%)": a count with its Wilson interval, never a bare percentage."""
    if not value:
        return f"no {what}"
    return (f"{value['successes']} of {value['total']} {what} "
            f"({value['low'] * 100:.0f}–{value['high'] * 100:.0f}%)")


def split_text(group: dict[str, Any], kind: str) -> str:
    """Where it held, by verdict, in a sentence: "at the end in 0 of 3 passing runs (0–56%) and 5 of
    7 failing (36–92%): it goes with failing (Fisher p = 0.08)"."""
    split = group.get("split") or {}
    when = "at the end" if kind == "file" else "at some step"
    text = f"{when} in {rate_words(split.get('passing'), 'passing runs')} and {rate_words(split.get('failing'), 'failing')}"
    if split.get("p") is None:
        return text + ": nothing to compare, with " + ("no failing run" if split.get("passing") else "no passing run")
    if split.get("goes_with"):
        return text + f": it goes with {split['goes_with']} (Fisher p = {p_short(split['p'])})"
    return text + f": no split (Fisher p = {p_short(split['p'])})"


def seen_text(group: dict[str, Any], kind: str) -> str | None:
    """Where it was first seen, or made, in a sentence."""
    seen = group.get("seen_at") or []
    if not seen:
        return None
    steps = ", ".join(f"step {step} in {count}" for step, count in seen)
    if kind == "file":
        made = group.get("made") or {}
        text = f"made by a call in {made.get('runs')} of {made.get('of')} judged run(s) ({steps})"
        if made.get("undone"):
            text += f"; undone before the end in {made['undone']}"
        if made.get("shared_mistake"):
            text += f" — every run made it, the {made['passing']} that passed too, which undid it before the end"
        return text
    return f"first at {steps}"


def reference_text(group: dict[str, Any]) -> str | None:
    """The step reference that names that call in each run, in a sentence."""
    from tuningfork.ops.experiments import CHANGED

    found = group.get("reference")
    if not found:
        return None
    text = f"step \"{found['step']}\" names that call in {found['names']} of the {found['of']} run(s)"
    if found["step"].startswith(CHANGED):
        return text + ", each run's first change of the file"
    return text + ", each run's first call of it"
