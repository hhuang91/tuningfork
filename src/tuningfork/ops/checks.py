"""Trying a check before trusting it.

A task's check decides every verdict a batch or a fork counts, and a check that is subtly
wrong — a key misspelled, a pattern that matches too much, a command that passes whatever
happens — skews every pass rate built on it, silently. So a check can be tried as it stands,
unsaved: against what a recorded run left behind (its workspace as the run ended, restored
from its snapshot, and its final reply), or against a reply and files a person writes. It
runs in a throwaway folder, and nothing is recorded: a run's own verdict never changes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.specs import CHECK_KEPT
from tuningfork.store.db import Run, Store
from tuningfork.store.masking import Masked

#: How long a tried check may run, unless asked otherwise: one that takes longer is worth
#: knowing about before a batch runs it once a run.
DEFAULT_TIMEOUT = 60.0
#: How much of a reply, and of a workspace's file list, a trial reports.
REPLY_SHOWN = 4000
FILES_SHOWN = 200


@dataclass(slots=True)
class CheckTrialPlan:
    """A check, and what it will be tried against: decided before anything runs."""

    check: Any
    #: The recorded run it is tried against; None for a reply and files written here.
    run: Run | None = None
    #: The run's workspace as it ended, in the snapshot store; None when it kept none.
    commit: str | None = None
    #: What is judged as the reply: the run's final reply, or the one written.
    reply: str = ""
    #: Files for the workspace, for a trial without a run.
    files: list[dict[str, str]] = field(default_factory=list)
    timeout: float = DEFAULT_TIMEOUT
    #: The verdict the run got when it ran, set beside this one.
    recorded: dict[str, Any] | None = None
    #: Whether the check is the run's own, as it was launched.
    own: bool = False
    #: The folder of the spec the run was launched from: `$TUNINGFORK_SPEC_DIR` for the check.
    spec_dir: str | None = None
    #: The variables the run's task lets its check read (`check_env`).
    check_env: list[str] = field(default_factory=list)
    #: Where a shell check runs: a `tuningfork.sandbox.Sandbox`, as a launch does, or None —
    #: and then why not (`unsandboxed`), unless the person turned it off.
    sandbox: Any = None
    unsandboxed: str | None = None

    def to_json(self) -> dict[str, Any]:
        from tuningfork.specs import check_kind, describe_check

        return {
            "check": self.check,
            "check_kind": check_kind(self.check),
            "check_text": describe_check(self.check),
            "own": self.own,
            "sandboxed": self.sandbox is not None,
            **({"unsandboxed": self.unsandboxed} if self.unsandboxed else {}),
            "run": {
                "id": self.run.id, "label": self.run.label, "task_id": self.run.task_id,
                "harness": self.run.harness,
            } if self.run else None,
            "workspace": "run" if self.commit else "files" if self.files else "empty",
            "files": [entry["path"] for entry in self.files],
            "reply_chars": len(self.reply),
            "recorded": self.recorded,
            "timeout": self.timeout,
        }


@dataclass(slots=True)
class CheckTrial:
    """How a check judged what it was tried against."""

    plan: CheckTrialPlan
    passed: bool
    #: The command's exit code; None for a reply check, which runs nothing.
    returncode: int | None
    output: str
    #: The workspace's files as the check found them.
    found: list[str]
    #: Characters cut from the output's start, as a verdict cuts them (`specs.CHECK_KEPT`).
    cut: int = 0

    @property
    def agrees(self) -> bool | None:
        """Whether this is the verdict the run got when it ran; None when it got none."""
        recorded = (self.plan.recorded or {}).get("passed")
        return None if recorded is None else recorded == self.passed

    def to_json(self) -> dict[str, Any]:
        reply = self.plan.reply
        return {
            **self.plan.to_json(),
            "passed": self.passed,
            "returncode": self.returncode,
            "output": self.output,
            "output_cut": self.cut,
            "found": self.found[:FILES_SHOWN],
            "found_more": max(0, len(self.found) - FILES_SHOWN),
            "reply": reply if len(reply) <= REPLY_SHOWN else f"{reply[:REPLY_SHOWN]}… (+{len(reply) - REPLY_SHOWN} chars)",
            "agrees": self.agrees,
        }


def prepare_check_trial(
    store: Store,
    check: Any = None,
    *,
    run: str | None = None,
    reply: str = "",
    files: Sequence[dict[str, str]] = (),
    timeout: float = DEFAULT_TIMEOUT,
) -> CheckTrialPlan:
    """What a check will be tried against, checked before it runs: a recorded run (with no
    check named, the run's own), or a reply and files written here."""
    from tuningfork.ops.experiments import resolve_run
    from tuningfork.runner import final_reply
    from tuningfork.snapshots import available
    from tuningfork.specs import Eval, SpecError, check_kind, check_problems, inside

    found: Run | None = None
    commit = None
    recorded = None
    own = False
    spec_dir = None
    check_env: list[str] = []
    if run:
        found = resolve_run(store, run)
        if found.is_thread:
            raise OperationError(
                f"run {found.id[:8]} is a {found.role} thread: a check judges the run its launch made",
                hint=f"try it on run {found.parent_run_id[:8]}" if found.parent_run_id else None,
                field="run",
            )
        if files:
            raise OperationError(
                "a trial against a run uses the run's own workspace: files are for a trial without one",
                field="files",
            )
        launch = store.get_launch(found.launch_id) if found.launch_id else None
        if check is None:
            if launch is None or not launch.eval:
                raise OperationError(
                    f"run {found.id[:8]} was not launched by tuningfork, so it has no check of its own",
                    hint="name the check to try", field="check",
                )
            check = Eval.from_dict(launch.eval).tuningfork.check
            if check is None:
                raise OperationError(f"run {found.id[:8]}'s task has no check", hint="name one to try", field="check")
            own = True
        commit = launch.final_commit if launch else None
        # A fork's launch names no spec file; its eval keeps the folder, as a batch's does.
        if launch is not None and launch.spec_path:
            spec_dir = str(Path(launch.spec_path).parent)
        elif launch is not None and isinstance(launch.eval, dict) and launch.eval.get("spec_dir"):
            spec_dir = str(launch.eval["spec_dir"])
        if launch is not None and launch.eval:
            # What its task lets a check read, whichever check is tried on its workspace.
            check_env = Eval.from_dict(launch.eval).tuningfork.check_env
        reply = final_reply(store, found.id)
        if found.result:
            recorded = {
                "passed": found.result.get("passed"),
                "usable": found.result.get("usable"),
                "check_output": (found.result.get("check_output") or "")[-2000:],
            }
    problems = check_problems(check)
    if problems:
        raise OperationError(f"the check {problems[0].message}", field="check")
    kind = check_kind(check)
    if kind is None:
        raise OperationError("name the check to try: a shell command, or what the reply should hold", field="check")
    if kind == "shell" and found is not None and commit is None:
        raise OperationError(
            f"run {found.id[:8]} kept no workspace for a command to run in",
            hint="only runs tuningfork launched keep one, from Phase 4 on; a reply check needs none",
            field="run",
        )
    if commit and not available():
        raise OperationError("a run's workspace is restored with git, which is not on PATH", field="run")
    written = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("content", ""), str):
            raise OperationError("each file is {path, content}, both text", field="files")
        try:
            inside(Path("/workspace"), entry["path"], what="file")
        except SpecError as exc:
            raise OperationError(str(exc), field="files") from None
        written.append({"path": entry["path"], "content": entry.get("content", "")})
    from tuningfork.sandbox import for_launches, turned_off

    # As a launch's check runs: a shell check can run what the model left in the workspace.
    sandbox, unsandboxed = for_launches([store.home]) if kind == "shell" else (None, None)
    return CheckTrialPlan(
        check=check, run=found, commit=commit, reply=reply or "", files=written,
        timeout=timeout, recorded=recorded, own=own, spec_dir=spec_dir, check_env=check_env,
        sandbox=sandbox, unsandboxed=None if turned_off() else unsandboxed,
    )


def execute_check_trial(store: Store, plan: CheckTrialPlan, *, keep: int | None = CHECK_KEPT) -> CheckTrial:
    """Run the check in a throwaway folder laid out as a run's: the workspace, and the reply
    in a file beside it (`$TUNINGFORK_REPLY`). Nothing is kept, and nothing recorded. The
    output is its last `keep` characters, as a verdict keeps it; None, all of it."""
    from tuningfork.snapshots import Snapshots
    from tuningfork.specs import inside, run_check

    with tempfile.TemporaryDirectory(prefix="tuningfork-check-") as scratch:
        root = Path(scratch)
        workspace = root / "workspace"
        if plan.commit:
            Snapshots(store.home).restore(plan.commit, workspace)
        else:
            workspace.mkdir()
            for entry in plan.files:
                target = inside(workspace, entry["path"], what="file")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(entry["content"], encoding="utf-8")
        found = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file())
        reply_file = root / "reply.txt"
        reply_file.write_text(plan.reply, encoding="utf-8")
        result = run_check(plan.check, workspace, plan.timeout, reply=plan.reply, reply_file=reply_file,
                           spec_dir=Path(plan.spec_dir) if plan.spec_dir else None, check_env=plan.check_env,
                           sandbox=plan.sandbox, keep=keep)
    return CheckTrial(plan=plan, passed=result.passed, returncode=result.returncode, output=result.output, found=found,
                      cut=result.cut)


def try_check(store: Store, check: Any = None, **options: Any) -> CheckTrial:
    """Prepare a trial and run it: what the CLI does in one go."""
    return execute_check_trial(store, prepare_check_trial(store, check, **options))


# --- the check along a run -------------------------------------------------------------------
#
# A check judges a run's end. Run on the workspace as each call left it, it says when each thing
# it complains about came to be: a finding there from the start is work not done yet; one a call
# made is a mistake, and that call is where it was made. Where every run made the same mistake
# and some repaired it later, comparing passing with failing runs points at the repair; this
# points at the mistake (findings §31.42: all ten runs booked a delivery's first drop again, and
# the four that passed had noticed later, by accident). A diagnostic, never a verdict: part-way
# through a run a check fails on work not done yet, and nothing here changes a run's own verdict.

#: Where the check along a run is kept, in the run's result.
ALONG = "check_along"
#: The findings a state keeps whole, each to so many characters: a call is named for a finding
#: by comparing them, and a finding cut from one state's output would read as made by the next
#: call. A state with more is not placed at all (`made_by_calls`): its check prints more than
#: findings.
MAX_FINDINGS = 400
FINDING_CHARS = 500


def kept_findings(output: str) -> dict[str, Any]:
    """A state's findings, each distinct one once in the order printed — all of them, or a state
    whose check printed more than `MAX_FINDINGS` marked cut."""
    from tuningfork.ops.tally import finding_lines

    lines = list(dict.fromkeys(line[:FINDING_CHARS] for line in finding_lines(output)))
    if len(lines) > MAX_FINDINGS:
        return {"findings": lines[:MAX_FINDINGS], "findings_cut": True}
    return {"findings": lines}


@dataclass(slots=True)
class CheckAlongPlan:
    """A run, and the states of its workspace its check will be run on: the start, each that a
    call left changed, and the end."""

    trial: CheckTrialPlan
    #: (the step whose call found the workspace like this — 0 the start, the number of steps the
    #: end — and its snapshot), each a change from the one before, the start and the end always.
    states: list[tuple[int, str]]
    steps: int

    def to_json(self) -> dict[str, Any]:
        return {**self.trial.to_json(), "states": len(self.states), "steps": self.steps}


def prepare_check_along(store: Store, run: str, *, timeout: float = DEFAULT_TIMEOUT) -> CheckAlongPlan:
    """What the run's own check will be run on, checked before anything runs: a shell check — a
    reply check has nothing to read before the reply — and a run whose workspace was snapshotted
    at each call."""
    from tuningfork.specs import check_kind

    trial = prepare_check_trial(store, run=run, timeout=timeout)
    found = trial.run
    if check_kind(trial.check) != "shell":
        raise OperationError(
            f"run {found.id[:8]}'s check reads the reply alone, and there is no reply before the end: only a "
            "check that reads the files can be run along a run",
            code="bad_params", field="run",
        )
    from tuningfork.ops.inspect import workspace_states

    states = workspace_states(store, found)
    if states is None:
        raise OperationError(f"run {found.id[:8]} kept no snapshots of its workspace to run the check on",
                             field="run")
    # The last state is the end, which the run's last step left: its number is the run's steps.
    return CheckAlongPlan(trial=trial, states=states, steps=states[-1][0])


def execute_check_along(
    store: Store, plan: CheckAlongPlan, *, progress: Any = None, cancel: Any = None,
) -> dict[str, Any]:
    """Run the check on each state, and keep what it printed with the run (`ALONG` in its
    result): its own verdict is left as it was."""
    import time
    from dataclasses import replace

    from tuningfork.ops.common import emit

    states = []
    seen: dict[tuple[str, bool], CheckTrial] = {}
    for index, (before, commit) in enumerate(plan.states):
        if cancel is not None:
            cancel.check()
        end = index == len(plan.states) - 1
        key = (commit, end)
        if key not in seen:
            # The reply is the run's own at the end; before it, there is none. The output whole:
            # its findings are compared state by state, and only their end is shown.
            reply = plan.trial.reply if end else ""
            seen[key] = execute_check_trial(store, replace(plan.trial, commit=commit, reply=reply), keep=None)
        trial = seen[key]
        states.append({"before_step": before, "end": end, "commit": commit, "passed": trial.passed,
                       "output": trial.output[-CHECK_KEPT:], **kept_findings(trial.output)})
        emit(progress, "state", done=index + 1, total=len(plan.states))
    record = {"states": states, "steps": plan.steps, "at": time.time()}
    run = store.get_run(plan.trial.run.id)
    # The result as it is stored, masked already, and the record masked once beside it: masked
    # again, a command's second password would fall too.
    store.annotate_run(run.id, result=Masked({**(run.result or {}), ALONG: store.masking.value(record)}))
    return {"run": run.id, **record, "made": made_by_calls(record) or []}


def check_along(store: Store, run: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Prepare and run the check along one run."""
    return execute_check_along(store, prepare_check_along(store, run, timeout=timeout))


@dataclass(slots=True)
class CheckAlongRunsPlan:
    """The runs the check will be run along — a task's, or one — and those passed over, and why."""

    task_id: str | None
    plans: list[CheckAlongPlan]
    skipped: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runs": [plan.trial.run.id for plan in self.plans],
            "states": sum(len(plan.states) for plan in self.plans),
            "skipped": self.skipped[:20],
            "skipped_more": max(0, len(self.skipped) - 20),
        }


@dataclass(slots=True)
class CheckAlongRuns:
    """What the check said along each run: the findings a call made, with the step that made each."""

    task_id: str | None
    runs: list[dict[str, Any]]
    skipped: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "runs": self.runs, "skipped": self.skipped[:20],
                "skipped_more": max(0, len(self.skipped) - 20)}


def prepare_check_along_runs(
    store: Store,
    *,
    task: str | None = None,
    run: str | None = None,
    limit: int = 50,
    again: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    hidden: Sequence[str] = (),
) -> CheckAlongRunsPlan:
    """The runs of a task — its own, not forks or threads, the newest first, each not run along
    before unless `again` — or one run, each checked before anything runs. `hidden` are tasks an
    agent may not reach (the held-out ones)."""
    from tuningfork.ops.tally import refuse_hidden_task

    if (task is None) == (run is None):
        raise OperationError("give a task, or one run", code="bad_params", field="task")
    if run is not None:
        plan = prepare_check_along(store, run, timeout=timeout)
        refuse_hidden_task(plan.trial.run.task_id, hidden)
        return CheckAlongRunsPlan(task_id=plan.trial.run.task_id, plans=[plan])
    refuse_hidden_task(task, hidden)
    runs = [found for found in store.runs(limit=limit, task_id=task) if not found.is_thread and not found.is_fork]
    if not runs:
        raise OperationError(f"no runs of task {task!r}", code="not_found", field="task")
    plans, skipped = [], []
    for found in runs:
        if not again and (found.result or {}).get(ALONG):
            skipped.append(f"run {found.id[:8]}: run along before (again runs it anew)")
            continue
        try:
            plans.append(prepare_check_along(store, found.id, timeout=timeout))
        except OperationError as error:
            skipped.append(f"run {found.id[:8]}: {error.message}")
    return CheckAlongRunsPlan(task_id=task, plans=plans, skipped=skipped)


def execute_check_along_runs(
    store: Store, plan: CheckAlongRunsPlan, *, progress: Any = None, cancel: Any = None,
) -> CheckAlongRuns:
    """Run the check along each planned run, one after another, keeping each with its run."""
    from tuningfork.ops.common import emit

    done = []
    for index, each in enumerate(plan.plans):
        if cancel is not None:
            cancel.check()
        record = execute_check_along(store, each, cancel=cancel)
        done.append({
            "run": each.trial.run.id,
            "passed": (each.trial.run.result or {}).get("passed"),
            "states": len(record["states"]),
            "made": [entry for entry in record["made"] if entry["made_by"] is not None][:20],
        })
        emit(progress, "run", done=index + 1, total=len(plan.plans), run=each.trial.run.id)
    return CheckAlongRuns(task_id=plan.task_id, runs=done, skipped=plan.skipped)


def made_by_calls(record: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Each line the check printed along a run, with the step whose call first made it — None
    when it was there from the start: work not done yet — and whether it was still there at the
    end. Lines are followed by their exact text: masked as the tally masks them, "SKU-310: 15 in
    stock" (not booked yet) and "SKU-100: 28 in stock" (booked twice) would be one finding.

    None when a state's findings were cut (`kept_findings`): a line missing from a state might
    have been there, and no call can be named for it — the run is not placed at all."""
    from tuningfork.ops.tally import MAX_LINE, finding_key, finding_lines

    states = (record or {}).get("states") or []
    if not states:
        return []
    if any(state.get("findings_cut") for state in states):
        return None
    # A record from before states kept their findings whole has only its output's end.
    lines = [state["findings"] if "findings" in state else finding_lines(state.get("output")) for state in states]
    first: dict[str, int] = {}
    for index, found in enumerate(lines):
        for line in found:
            first.setdefault(line, index)
    end = set(lines[-1])
    made = []
    for line, index in first.items():
        # A line first seen in a state a call made: the call at the step before that state.
        by = None if index == 0 else int(states[index]["before_step"]) - 1
        made.append({"line": line[:MAX_LINE], "finding": finding_key(line), "made_by": by, "at_end": line in end})
    return made


def first_made(run: Run, words: str) -> int | None:
    """The step whose call first made a finding with these words (case aside) along `run`, from
    its check along the run; None when none did, or the check was not run along it, or could not
    place what it found."""
    wanted = words.strip().lower()
    steps = [entry["made_by"] for entry in made_by_calls((run.result or {}).get(ALONG)) or []
             if entry["made_by"] is not None and wanted in f"{entry['line']}\n{entry['finding']}".lower()]
    return min(steps) if steps else None
