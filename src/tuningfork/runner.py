"""Batch runner: execute a task N times through a headless harness.

Every iteration gets a fresh workspace *and* a fresh harness home. That is stronger than it
may look: a shared home means memory injected by run 3 is present for run 7, and the runs
are no longer independent samples — which quietly invalidates every interval and
significance test computed over them.

Skill files are hashed before and after each run. If they moved, the run is flagged: the
thing under test changed while it was being measured, so the sample does not belong in the
comparison.

Each run is a *launch* (see `tuningfork.launches`): its harness talks to the proxy under its
own token, so every call it makes is attributed to it exactly, as it arrives. That is what
lets runs go in parallel — before, runs were told apart by their message paths and a queue
of labels, which parallel copies of one task would have crossed.

Where bubblewrap runs, each launch runs in a sandbox (`tuningfork.sandbox`): its folders,
wherever they are on disk, are seen as one home and one project folder, the same in every
launch, so no run's prompt differs from another's by the folders it was given.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from tuningfork.analysis.conditions import exclusion_from
from tuningfork.analysis.conditions import pass_rate  # noqa: F401 - long imported from here
from tuningfork.harness.base import HarnessError, LaunchSpec
from tuningfork.launches import launch_url
from tuningfork.ops.common import Cancel, Cancelled
from tuningfork.snapshots import SnapshotError, Snapshots
from tuningfork.snapshots import available as snapshots_available
from tuningfork.specs import (
    CHECK_KEPT,
    CheckResult,
    Eval,
    load_context,
    TaskSpec,
    hash_tree,
    materialize_files,
    missing_files,
    run_check,
    skill_bodies,
)
from tuningfork.store.db import Launch, Run, Step, Store

logger = logging.getLogger("tuningfork.runner")

#: How often a run that stops at the next skill asks whether it has: the harness's call after
#: that answer is refused, so it ends within this, or on its own.
STOP_POLL = 0.5


@dataclass(slots=True)
class RunOutcome:
    index: int
    label: str
    run_id: str | None
    passed: bool
    check: CheckResult
    duration_s: float
    returncode: int
    stdout: str
    stderr: str
    skills_changed: list[str] = field(default_factory=list)
    timed_out: bool = False
    #: False when the task defines no check command: the run has no verdict at all.
    checked: bool = True
    #: Steps a person changed at a breakpoint. Such a run's verdict measures the edit as
    #: much as the skill, so it is not a sample of the skill's behaviour.
    edited_steps: list[int] = field(default_factory=list)
    #: Steps whose model call failed — the server errored, the stream broke off, or the
    #: call was dropped at a breakpoint. The run then measured the infrastructure.
    failed_steps: list[int] = field(default_factory=list)
    #: The same, in the launch's other threads — a subagent's calls, say — as
    #: `<role> <run> step <k>`.
    thread_trouble: list[str] = field(default_factory=list)
    #: The launch this run was; see `tuningfork.launches`.
    launch_id: str | None = None
    #: For a fork: the step it went live at, and how its replay went. `departed` is the
    #: first call the recording had no answer for; `leaked`, where the original wording
    #: still reached a model testing the variant; `reached`, whether the harness got as
    #: far as the step at all.
    fork_step: int | None = None
    departed: dict[str, Any] | None = None
    leaked: list[int] = field(default_factory=list)
    reached: bool = True
    #: For a fork: what the model did at the step it went live at.
    live_outcome: dict[str, Any] | None = None
    #: What the harness was told after the check, when it was (`Launcher.tell`): the message,
    #: and how that turn's process ended.
    told: dict[str, Any] | None = None
    #: Where a scenario's run ended beside its skill (`tuningfork.stop_at_next_skill`): the
    #: skill the model loaded next, and the step whose answer loaded it. Counted: the scenario
    #: ends there, and the check judged the workspace as it was.
    stopped: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        """Whether this run may be counted in a statistic.

        A run that never reached the proxy measured the harness failing to start, not the
        skill; so did one whose model calls failed. A fork whose replay departed from the
        recording has a history other than the one it claims to continue.
        """
        return self.exclusion is None

    @property
    def exclusion(self) -> str | None:
        """Why the run is not counted, in words; None when it is."""
        if self.run_id is None:
            return "no model call was recorded"
        return exclusion_from({
            "skills_changed": self.skills_changed, "timed_out": self.timed_out,
            "edited_steps": self.edited_steps, "failed_steps": self.failed_steps,
            "thread_trouble": self.thread_trouble, "departed": self.departed,
            "leaked": self.leaked, "reached": self.reached, "fork_step": self.fork_step,
        })

    def to_json(self) -> dict[str, Any]:
        """How a run went, for a front end: its verdict and bookkeeping, without the
        harness's output."""
        result = self.to_result()
        result.pop("check_output", None)
        return {
            **result,
            "index": self.index,
            "label": self.label,
            "run_id": self.run_id,
            "launch_id": self.launch_id,
            "exclusion": self.exclusion,
            "live_outcome": self.live_outcome,
        }

    def to_result(self) -> dict[str, Any]:
        """What the store keeps about how the run ended, for reading back later."""
        result = {
            "passed": self.passed if self.checked else None,
            "check_output": self.check.output[-CHECK_KEPT:],
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 2),
            "timed_out": self.timed_out,
            "skills_changed": self.skills_changed,
            "edited_steps": self.edited_steps,
            "failed_steps": self.failed_steps,
            "thread_trouble": self.thread_trouble,
            "usable": self.usable,
        }
        # What a verdict does not keep of a longer output, said wherever it is shown: found in
        # review, nothing said a finding printed early in one was never tallied.
        cut = self.check.cut + max(0, len(self.check.output) - CHECK_KEPT)
        if cut:
            result["check_output_cut"] = cut
        if self.fork_step is not None:
            result.update(
                fork_step=self.fork_step,
                departed=self.departed,
                leaked=self.leaked,
                reached=self.reached,
                live_outcome=self.live_outcome,
                exclusion=self.exclusion,
            )
        if self.told is not None:
            result["told"] = self.told
        if self.stopped is not None:
            result["stopped_at_skill"] = self.stopped
        return result


def told_message(template: str, check: CheckResult) -> str:
    """What the harness is told of the check: the template with the verdict and the check's
    output, as a person who ran the check would put it."""
    verdict = "It passed." if check.passed else "It failed."
    return template.format(verdict=verdict, output=(check.output or "").strip()).rstrip()


def final_reply(store: Store, run_id: str) -> str:
    """A run's final reply: what its last step said, as the harness received it — what a
    reply check judges. Empty when the run ended without saying anything (on a tool call,
    or an error). Redacted as the recording is.

    A harness's own side call is never the final reply. A launch files one in a thread of
    its own as it arrives (`tuningfork.launches`); a run recorded before that may end on
    one — Hermes's background review, cut off when Hermes exited — so it is skipped.
    """
    from tuningfork.analysis.normalize import reply_text
    from tuningfork.proxy.launches import side_call_for

    steps = store.steps(run_id)
    run = store.get_run(run_id)
    side_call = side_call_for(run.harness if run else None)
    while steps and side_call and side_call(store.harness_request(steps[-1])):
        steps.pop()
    return reply_text(store.delivered_response(steps[-1])) if steps else ""


def named_side_call(store: Store, adapter: Any, run: Any) -> bool:
    """Whether a thread is a side call its harness names (`side_call`): its own work, beside
    the task's conversation and never fed back into it. Its failure says nothing of the run —
    Hermes's background review is cut off whenever Hermes exits after its answer."""
    side_call = getattr(adapter, "side_call", None)
    if run.role != "side" or side_call is None:
        return False
    first = store.steps(run.id)[:1]
    return bool(first) and bool(side_call(store.harness_request(first[0])))


def lookalike_skill_files(skill_dir: Path) -> list[str]:
    """Other `SKILL*.md` files in a skill folder: variants kept beside the skill.

    Every run is given a copy of the whole folder, and a model that looks around it can
    read them — an improved wording beside the one under test is an answer key. This
    project's own example kept its variant there from Phase 1 until it was noticed.
    """
    return sorted(
        path.name
        for path in skill_dir.glob("*.md")
        if path.name.lower().startswith("skill") and path.name != "SKILL.md"
    )


@dataclass(slots=True)
class Exit:
    """How a harness process ended."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    #: Killed because the work was cancelled: how it would have ended is unknown.
    cancelled: bool = False
    #: Ended where the model loaded a skill other than the one its scenario studies, as the
    #: proxy marked it (`tuningfork.stop_at_next_skill`): which skill, after which step.
    stopped: dict[str, Any] | None = None


@dataclass(slots=True)
class BatchResult:
    task_id: str
    outcomes: list[RunOutcome]

    @property
    def checked(self) -> bool:
        """Whether the task has a check at all. Without one there is no pass rate."""
        return all(outcome.checked for outcome in self.outcomes)

    @property
    def usable(self) -> list[RunOutcome]:
        return [outcome for outcome in self.outcomes if outcome.usable]

    @property
    def run_ids(self) -> list[str]:
        return [o.run_id for o in self.outcomes if o.run_id]

    @property
    def passes(self) -> int:
        return sum(1 for outcome in self.usable if outcome.passed)

    @property
    def invalidated(self) -> list[RunOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.usable]


class Launcher:
    """Starts harnesses as launches, and judges how each went. Batch runs and forks both
    run through here: a fork is a batch run whose calls the proxy answers from a recording
    until the step it goes live at."""

    def __init__(
        self,
        store: Store,
        adapter: Any,
        *,
        base_url: str,
        model: str,
        workdir: Path,
        settle: float = 0.25,
        concurrency: int = 1,
        sandbox: Any = None,
        tell: str | None = None,
        launch_seconds: float | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        #: Where each launch runs: a `tuningfork.sandbox.Sandbox`, in which every launch sees
        #: the same home and folder, or None for each in its own folders.
        self.sandbox = sandbox
        #: A template with `{verdict}` and `{output}`: after the check, the harness is given
        #: its result as one more user turn of the same session (`continue_launch`), before
        #: its own review — what a person who looked at the result would say. The verdict
        #: stays the task's: judged before the turn.
        self.tell = tell
        #: The most seconds a launch's processes may take together — its run, and the told
        #: turn after it — when something larger has a budget for it (a chain's last link):
        #: each process's timeout is cut to what is left of this. The processes' wall time
        #: bounds their model calls' time from above.
        self.launch_seconds = launch_seconds
        #: The proxy's base URL as a harness would be given it; each launch gets its own
        #: token inserted into it (see `tuningfork.launches.launch_url`).
        self.base_url = base_url
        self.model = model
        self.workdir = Path(workdir)
        #: The proxy records a step just after the harness has its response, and the
        #: harness can exit first. Steps are read once they stop arriving (see `_settle`).
        self.settle = settle
        #: Runs in flight at once. Each is its own launch, so parallel runs cannot be
        #: confused; the proxy's `--max-inflight` still caps calls reaching the server.
        self.concurrency = max(1, concurrency)
        #: Stops the work: runs not yet started are skipped, running harnesses killed.
        self.cancel = Cancel()
        self.cancel.on_cancel(self._kill_all)
        self._live: set[subprocess.Popen] = set()
        #: Harnesses killed by a cancel, rather than ending by themselves or timing out.
        self._killed: set[subprocess.Popen] = set()
        self._live_lock = threading.Lock()
        #: Called with each launch as it is created, before its harness starts: how a job
        #: learns which launches are its own.
        self.on_launch: Callable[[Launch, int], None] | None = None
        #: Where workspace snapshots go; None without git, and then there are none.
        self.snapshots = Snapshots(store.home) if snapshots_available() else None

    def _snap(self, directory: Path | None, launch: Launch, name: str) -> str | None:
        """Snapshot a directory as `refs/launches/<launch>/<name>`; None if it cannot be."""
        if self.snapshots is None or directory is None or not Path(directory).is_dir():
            return None
        try:
            return self.snapshots.commit(
                Path(directory), f"refs/launches/{launch.id}/{name}", f"{name} of {launch.label}"
            )
        except SnapshotError:
            logger.warning("tuningfork could not snapshot %s", directory, exc_info=True)
            return None

    def _many(
        self,
        n: int,
        one: Callable[[int], RunOutcome],
        on_progress: Callable[[RunOutcome], None] | None,
    ) -> list[RunOutcome]:
        """`one(i)` for i in range(n), `concurrency` at a time, in index order.

        Cancelled, it raises `Cancelled` whose `partial` is every run that finished — each
        reported to `on_progress` first, even one that finished while another was being
        killed. A run killed part way through is not among them: it has no verdict.
        """

        with _terminate_as_interrupt():
            return self._run_many(n, one, on_progress)

    def _run_many(
        self,
        n: int,
        one: Callable[[int], RunOutcome],
        on_progress: Callable[[RunOutcome], None] | None,
    ) -> list[RunOutcome]:
        def guarded(index: int) -> RunOutcome:
            self.cancel.check()
            return one(index)

        outcomes: list[RunOutcome] = []

        def deliver(outcome: RunOutcome) -> None:
            outcomes.append(outcome)
            if on_progress:
                on_progress(outcome)

        try:
            if self.concurrency == 1:
                for index in range(n):
                    deliver(guarded(index))
            else:
                pool = ThreadPoolExecutor(max_workers=self.concurrency)
                futures = [pool.submit(guarded, index) for index in range(n)]
                delivered: set[Any] = set()
                try:
                    for future in as_completed(futures):
                        outcome = future.result()
                        delivered.add(future)
                        deliver(outcome)
                except BaseException:
                    self.cancel.cancel()
                    for future in futures:
                        future.cancel()
                    pool.shutdown(wait=True)
                    for future in futures:
                        # Runs that finished while the others were being stopped.
                        if future in delivered or future.cancelled() or future.exception():
                            continue
                        deliver(future.result())
                    raise
                finally:
                    pool.shutdown(wait=True)
        except KeyboardInterrupt:
            # Kill every harness still running; what finished stays recorded.
            self.cancel.cancel()
            raise
        except Cancelled as stopped:
            self.cancel.cancel(stopped.reason)
            raise Cancelled(
                stopped.reason, partial=sorted(outcomes, key=lambda o: o.index)
            ) from None
        return sorted(outcomes, key=lambda o: o.index)

    def _go(
        self,
        launch: Launch,
        item: Eval,
        spec_dir: Path,
        skill: Path | None,
        workspace: Path,
        home: Path,
        index: int,
        *,
        annotate: dict[str, Any],
        restore: str | None = None,
        restore_skill: str | None = None,
    ) -> RunOutcome:
        """Prepare a launch's workspace and home, run its harness, and judge the result.

        `restore` starts the workspace from a snapshot rather than from the eval's files —
        a fork starts from exactly what its source did; `restore_skill` is a snapshot of
        the skill folder, for when the frozen copy on disk has gone.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        prepare_home(
            self.adapter, home, base_url=launch_url(self.base_url, launch.id), model=self.model,
            max_turns=item.tuningfork.max_turns,
        )
        if self.sandbox is not None:
            from tuningfork.sandbox import mount_point

            mount_point(home).mkdir(parents=True, exist_ok=True)
        if skill is not None and skill.exists():
            for place in skill_places(self.adapter, home, skill):
                shutil.copytree(skill, place, dirs_exist_ok=True)
        elif skill is not None and restore_skill and self.snapshots is not None:
            # Restored beside the home first: where it goes can depend on its own SKILL.md.
            staging = Path(tempfile.mkdtemp(prefix=".skill-", dir=home.parent)) / skill.name
            try:
                self.snapshots.restore(restore_skill, staging)
                for place in skill_places(self.adapter, home, staging):
                    shutil.copytree(staging, place, dirs_exist_ok=True)
            finally:
                shutil.rmtree(staging.parent, ignore_errors=True)
        if restore and self.snapshots is not None:
            self.snapshots.restore(restore, workspace)
        else:
            materialize_files(item, workspace, spec_dir)
        self.store.set_launch_snapshots(
            launch.id,
            initial_commit=self._snap(workspace, launch, "initial"),
            skill_commit=self._snap(skill, launch, "skill") if skill and skill.exists() else restore_skill,
        )

        skill_roots = list(self.adapter.skill_roots(home))
        before = hash_tree(skill_roots)
        bodies = skill_bodies(skill_roots)

        started = time.monotonic()
        try:
            self.cancel.check()
            spec_ = self.adapter.launch(
                item.prompt,
                home=home,
                workspace=workspace,
                skills=item.tuningfork.skills,
                max_turns=item.tuningfork.max_turns,
            )
            if self.sandbox is not None:
                spec_ = self.sandbox.wrap(spec_, home=home, workspace=workspace)
            exited = self._execute(spec_, workspace, item, timeout=self._limit(item, started),
                                   stopped=self._stop_mark(launch))
        except Cancelled as stopped:
            self._stopped(launch, workspace, stopped.reason, time.monotonic() - started, annotate)
            raise
        except BaseException:
            self.store.end_launch(launch.id, status="interrupted")
            raise
        duration = time.monotonic() - started
        if exited.cancelled:
            reason = self.cancel.reason or "cancelled"
            self._stopped(launch, workspace, reason, duration, annotate)
            raise Cancelled(reason)
        returncode, stdout, stderr, timed_out = (
            exited.returncode, exited.stdout, exited.stderr, exited.timed_out
        )
        # Every step recorded before the check, which may judge the final reply.
        self._settle(launch.id)
        reply = self._final_reply(self.store.launch_runs(launch.id))
        # Beside the workspace, not in it: in no snapshot, and not among the run's files.
        reply_file = workspace.parent / "reply.txt"
        reply_file.write_text(reply, encoding="utf-8")
        check = run_check(item.tuningfork.check, workspace, reply=reply, reply_file=reply_file, spec_dir=spec_dir)
        told = None
        if (
            self.tell and item.tuningfork.check and not timed_out and exited.stopped is None
            and getattr(self.adapter, "continue_launch", None) is not None
        ):
            told = self._tell(launch, item, check, home, workspace, annotate, started)
            duration = time.monotonic() - started
        self.store.end_launch(
            launch.id,
            exit={
                "returncode": returncode, "timed_out": timed_out, "duration_s": round(duration, 2),
                **({"told": told} if told else {}),
                **({"stopped_at_skill": exited.stopped} if exited.stopped else {}),
                **({"cap_seconds": self.launch_seconds} if self.launch_seconds is not None else {}),
            },
        )
        self.store.set_launch_snapshots(
            launch.id, final_commit=self._snap(workspace, launch, "final")
        )

        after = hash_tree(skill_roots)
        changed = sorted(
            {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
        )
        self._settle(launch.id)
        self.store.finalize_threads(launch.id, getattr(self.adapter, "subagent_tools", ()))
        runs = self.store.launch_runs(launch.id)
        parity = {**(annotate.pop("parity", None) or {}), "skill_bodies": bodies}
        return self._outcome(
            launch, index, check, duration, returncode, stdout, stderr, timed_out, changed,
            runs=runs,
            checked=bool(item.tuningfork.check),
            annotate={**annotate, "skill_hashes": before, "parity": parity},
            told=told,
            stopped=exited.stopped,
        )

    def _tell(
        self,
        launch: Launch,
        item: Eval,
        check: CheckResult,
        home: Path,
        workspace: Path,
        annotate: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        """One more turn of the launch's session: the check's verdict and output as the
        user's next message, run by the harness as it resumes the session it just ended.
        Paths in the check's output are written as the sandbox shows them."""
        message = told_message(self.tell or "", check)
        if self.sandbox is not None:
            from tuningfork.sandbox import HOME, WORKSPACE, translate

            message = translate(message, [(Path(workspace), WORKSPACE), (Path(home), HOME)])
        turn_started = time.monotonic()
        try:
            self.cancel.check()
            spec_ = self.adapter.continue_launch(
                message, home=home, workspace=workspace, max_turns=item.tuningfork.max_turns,
            )
            if self.sandbox is not None:
                spec_ = self.sandbox.wrap(spec_, home=home, workspace=workspace)
            exited = self._execute(spec_, workspace, item, timeout=self._limit(item, started))
        except Cancelled as stopped:
            self._stopped(launch, workspace, stopped.reason, time.monotonic() - started, annotate)
            raise
        except BaseException:
            self.store.end_launch(launch.id, status="interrupted")
            raise
        if exited.cancelled:
            reason = self.cancel.reason or "cancelled"
            self._stopped(launch, workspace, reason, time.monotonic() - started, annotate)
            raise Cancelled(reason)
        return {
            "message": message,
            "returncode": exited.returncode,
            "timed_out": exited.timed_out,
            "duration_s": round(time.monotonic() - turn_started, 2),
        }

    def _final_reply(self, runs: list[Any]) -> str:
        main = next((run for run in runs if not run.is_thread), None)
        return final_reply(self.store, main.id) if main is not None else ""

    def _stopped(
        self,
        launch: Launch,
        workspace: Path,
        reason: str,
        duration: float,
        annotate: dict[str, Any],
    ) -> None:
        """A launch cancelled before it finished: ended as such, and never judged.

        How it would have ended is unknown, so its check is not run and it has no verdict:
        judging a killed harness would count a failure the skill never made. Its run is kept
        on file, marked as not counted.
        """
        status = "interrupted" if reason == "shutdown" else "cancelled"
        self.store.end_launch(
            launch.id,
            status=status,
            exit={"cancelled": True, "reason": reason, "duration_s": round(duration, 2)},
        )
        self.store.set_launch_snapshots(
            launch.id, final_commit=self._snap(workspace, launch, "final")
        )
        self.store.finalize_threads(launch.id, getattr(self.adapter, "subagent_tools", ()))
        main = next((run for run in self.store.launch_runs(launch.id) if not run.is_thread), None)
        if main is not None:
            self.store.annotate_run(
                main.id,
                harness=self.adapter.name,
                task_id=annotate.get("task_id"),
                result={
                    "passed": None,
                    "usable": False,
                    "cancelled": True,
                    "exclusion": f"{status} before it finished",
                    "duration_s": round(duration, 2),
                },
            )

    def _outcome(
        self,
        launch: Launch,
        index: int,
        check: CheckResult,
        duration: float,
        returncode: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
        changed: list[str],
        *,
        runs: list[Any],
        checked: bool,
        annotate: dict[str, Any],
        told: dict[str, Any] | None = None,
        stopped: dict[str, Any] | None = None,
    ) -> RunOutcome:
        """How a finished launch went, filed on its main run. `runs` are the launch's runs,
        every step recorded (`_settle`) and its threads classified."""
        main = next((run for run in runs if not run.is_thread), None)
        steps = self.store.steps(main.id) if main else []
        every = [(run, step) for run in runs for step in self.store.steps(run.id)]
        trouble = [
            f"{run.role} {run.id[:8]} step {step.idx}"
            for run, step in every
            if run.is_thread and not named_side_call(self.store, self.adapter, run)
            and (step.edits or step.error or (step.status or 200) >= 400)
        ]
        outcome = RunOutcome(
            index=index,
            label=launch.label or launch.id,
            run_id=main.id if main else None,
            passed=check.passed,
            check=check,
            duration_s=duration,
            returncode=returncode,
            stdout=stdout[-4000:],
            stderr=stderr[-4000:],
            skills_changed=changed,
            timed_out=timed_out,
            checked=checked,
            edited_steps=[step.idx for step in steps if step.edits],
            failed_steps=[
                step.idx for step in steps if step.error or (step.status or 200) >= 400
            ],
            thread_trouble=trouble,
            launch_id=launch.id,
            told=told,
            stopped=stopped,
        )
        if launch.kind == "fork" and launch.source_step is not None:
            k = launch.source_step
            outcome.fork_step = k
            outcome.departed = next(
                (step.replay["mismatch"] for _, step in every if step.replay.get("mismatch")),
                None,
            )
            outcome.leaked = [step.idx for step in steps if step.substitution.get("leaked")]
            live = next((step for step in steps if step.idx == k and not step.served), None)
            outcome.reached = any(step.idx >= k for step in steps)
            if live is not None:
                from tuningfork.analysis.normalize import recorded_outcome

                found = recorded_outcome(self.store.response(live), live.error)
                outcome.live_outcome = {"kind": found.kind, "key": found.key, "label": found.label}
        if main is not None:
            self.store.annotate_run(
                main.id,
                harness=self.adapter.name,
                result=outcome.to_result(),
                **annotate,
            )
        return outcome

    def _limit(self, item: Eval, started: float) -> float | None:
        """The timeout for one of a launch's processes: the eval's, cut to what is left of
        `launch_seconds` since the launch `started` — never under a second, so the process
        runs and its end is recorded as a timeout."""
        if self.launch_seconds is None:
            return None
        return max(1.0, min(item.tuningfork.timeout, self.launch_seconds - (time.monotonic() - started)))

    def _stop_mark(self, launch: Launch) -> Callable[[], dict[str, Any] | None] | None:
        """For a run that stops at the next skill: where the proxy said it stopped, once it
        has (`Store.note_launch`). None for any other run."""
        if not launch.settings.get("stop_at_next_skill"):
            return None

        def mark() -> dict[str, Any] | None:
            found = self.store.get_launch(launch.id)
            return found.settings.get("stopped_at_skill") if found is not None else None

        return mark

    def _execute(
        self, launch: LaunchSpec, workspace: Path, item: Eval, *, timeout: float | None = None,
        stopped: Callable[[], dict[str, Any] | None] | None = None,
    ) -> Exit:
        """Run the harness once, and say how it ended.

        The harness gets a process group of its own, and a timeout or a cancel kills the
        whole group: a harness's children — a shell it started, a subagent's tool — would
        otherwise outlive it, still writing into the workspace. With `stopped`, it is asked
        every half second whether the run has ended where its scenario stops (the model loaded
        another skill, `tuningfork.stop_at_next_skill`); then the harness goes the same way.
        """
        limit = item.tuningfork.timeout if timeout is None else timeout
        group = {"start_new_session": True} if os.name == "posix" else {}
        try:
            process = subprocess.Popen(
                list(launch.argv),
                cwd=str(launch.cwd or workspace),
                env=launch.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **group,
            )
        except FileNotFoundError:
            raise HarnessError(
                f"{launch.argv[0]!r} was not found. Is {self.adapter.name} installed, and "
                "on PATH for this shell?"
            ) from None
        with self._live_lock:
            self._live.add(process)
            if self.cancel.cancelled:
                # Cancelled between the check and the start: stop it at once.
                self._killed.add(process)
                _kill(process)
        try:
            deadline = time.monotonic() + limit
            while True:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    stdout, stderr = process.communicate(
                        timeout=remaining if stopped is None else min(STOP_POLL, remaining)
                    )
                    break
                except subprocess.TimeoutExpired:
                    mark = stopped() if stopped is not None else None
                    if mark is not None:
                        _kill(process)
                        stdout, _ = process.communicate()
                        return Exit(
                            -1, stdout or "", f"stopped where the model loaded {mark.get('skill')}",
                            cancelled=self._was_killed(process), stopped=mark,
                        )
                    if time.monotonic() >= deadline:
                        _kill(process)
                        stdout, _ = process.communicate()
                        return Exit(
                            -1, stdout or "", f"timed out after {limit:g}s",
                            timed_out=True, cancelled=self._was_killed(process),
                        )
            return Exit(
                process.returncode, stdout or "", stderr or "",
                cancelled=self._was_killed(process), stopped=stopped() if stopped is not None else None,
            )
        except BaseException:
            # Interrupted while waiting on it: Ctrl-C reaches this process alone — the
            # harness has a session of its own — so stop it here, before it leaves the live
            # set below and nothing is left to stop it. Found in use: it went on calling the
            # model after `tuningfork batch` was gone.
            _kill(process)
            try:
                process.wait(timeout=5)
            except BaseException:
                pass
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            raise
        finally:
            with self._live_lock:
                self._live.discard(process)
                self._killed.discard(process)

    def _was_killed(self, process: subprocess.Popen) -> bool:
        with self._live_lock:
            return process in self._killed

    def _kill_all(self) -> None:
        with self._live_lock:
            live = list(self._live)
            self._killed.update(live)
        for process in live:
            _kill(process)

    def _settle(self, launch_id: str) -> None:
        """Wait until the proxy has finished recording the launch's calls: the harness can
        exit a moment before its last step is stored."""
        def count() -> int:
            return sum(run.step_count for run in self.store.launch_runs(launch_id))

        seen = count()
        deadline = time.monotonic() + 20 * self.settle
        while self.settle and time.monotonic() < deadline:
            time.sleep(self.settle)
            now = count()
            if now == seen and now:
                break
            seen = now


class BatchRunner(Launcher):
    """Runs one eval N times, each run a launch of its own, recorded through the proxy."""

    def __init__(
        self,
        store: Store,
        adapter: Any,
        *,
        base_url: str,
        model: str,
        workdir: Path,
        keep_workspaces: bool = True,
        skill_file: Path | None = None,
        settle: float = 0.25,
        concurrency: int = 1,
        sandbox: Any = None,
        tell: str | None = None,
    ) -> None:
        super().__init__(
            store, adapter, base_url=base_url, model=model, workdir=workdir, settle=settle,
            concurrency=concurrency, sandbox=sandbox, tell=tell,
        )
        self.keep_workspaces = keep_workspaces
        #: Installed as the skill's SKILL.md in every run, in place of the spec's own — how
        #: an edit that won a checkpoint A/B is confirmed with full runs.
        self.skill_file = Path(skill_file) if skill_file else None

    def run(
        self,
        spec: TaskSpec,
        item: Eval,
        n: int,
        *,
        parity: dict[str, Any] | None = None,
        on_progress: Callable[[RunOutcome], None] | None = None,
        group: str | None = None,
        start: int = 0,
        skill_dir: Path | None = None,
        skill_snapshot: str = "skill-snapshot",
        settings: dict[str, Any] | None = None,
        launch_seconds: float | None = None,
    ) -> BatchResult:
        """Run the eval `n` times. The rest is for an operation that runs a batch's runs as
        part of something larger — a chain gives each of its links a skill folder of its own
        (`skill_dir`, frozen under `skill_snapshot`) in one group folder, numbered in order
        from `start`, with `settings` recorded on each launch beside the batch's own, and
        `launch_seconds` the most each launch's processes may take together (a budget's
        last link)."""
        self.launch_seconds = launch_seconds
        task_id = f"{spec.skill_name}/{item.id}"
        missing = missing_files(item, spec.directory)
        if missing:
            raise ValueError(
                f"eval {item.id} names input files that are not next to the spec "
                f"({spec.directory}): {', '.join(missing)}"
            )
        batch = group or uuid.uuid4().hex[:8]
        skill = self._snapshot_skill(spec, batch, source=skill_dir, name=skill_snapshot)
        parity = {**(parity or {}), "skill_snapshot": hash_tree([skill]) if skill else {}}
        if skill:
            # The frozen copy every run was given: the true "original" for a later A/B,
            # however the source has been edited since.
            parity["skill_snapshot_dir"] = str(skill.resolve())
        if self.skill_file:
            parity["skill_file"] = str(self.skill_file)

        def one(index: int) -> RunOutcome:
            return self._run_once(
                spec, item, start + index, task_id, batch, parity, skill, settings or {}
            )

        return BatchResult(task_id=task_id, outcomes=self._many(n, one, on_progress))

    def _snapshot_skill(
        self,
        spec: TaskSpec,
        batch: str,
        *,
        source: Path | None = None,
        name: str = "skill-snapshot",
    ) -> Path | None:
        """Freeze the skill for the whole batch.

        Every run copies from this snapshot rather than from the source. Editing the skill
        while a batch runs — easy to do, and done during Phase 1 — would otherwise put
        different wordings into different runs of what claims to be one sample, and the
        pass rate would describe neither. `source` is a folder other than the spec's, given
        by the operation running the batch.
        """
        source = Path(source) if source else spec.resolved_skill_dir()
        if not source or not source.exists():
            if self.skill_file:
                raise ValueError(
                    "the spec names no skill folder (tuningfork.skill_dir) for the "
                    "replacement SKILL.md to go in"
                )
            return None
        snapshot = self.workdir / batch / name / source.name
        shutil.copytree(source, snapshot, dirs_exist_ok=True)
        if self.skill_file:
            shutil.copyfile(self.skill_file, snapshot / "SKILL.md")
        return snapshot

    def _context(self, item: Eval, spec_dir: Path) -> dict[str, Any]:
        """The conversation a run continues (`tuningfork.context`), kept in the store for the
        proxy to put before the harness's own messages: its blob, how many messages, and the
        file it was read from. A fork carries its source's (`ForkSource.context`), never the file
        as it reads now."""
        if not item.tuningfork.context:
            return {}
        context = load_context(spec_dir, item.tuningfork.context)
        return {"context": {"blob": self.store.put_given(context.to_json()), "messages": len(context.before),
                            **({"in_turn": len(context.in_turn)} if context.in_turn else {}),
                            "file": item.tuningfork.context}}

    @staticmethod
    def _stop(spec: TaskSpec, item: Eval) -> dict[str, Any]:
        """Where a scenario's run ends, when it stops at the next skill
        (`tuningfork.stop_at_next_skill`): beside the skill its spec studies."""
        if not item.tuningfork.stop_at_next_skill or not spec.skill_name:
            return {}
        return {"stop_at_next_skill": {"skill": spec.skill_name}}

    def _run_once(
        self,
        spec: TaskSpec,
        item: Eval,
        index: int,
        task_id: str,
        batch: str,
        parity: dict[str, Any] | None,
        skill: Path | None,
        settings: dict[str, Any],
    ) -> RunOutcome:
        root = self.workdir / batch / str(index)
        workspace, home = root / "workspace", root / "home"
        launch = self.store.create_launch(
            kind="batch",
            label=f"{task_id}#{batch}.{index}",
            harness=self.adapter.name,
            model=self.model,
            task_id=task_id,
            eval=eval_record(item, spec.directory),
            spec_path=str(spec.path.resolve()) if spec.path else None,
            skill_dir=str(skill.resolve()) if skill else None,
            workspace=str(workspace.resolve()),
            home=str(home.resolve()),
            settings={"breakpoints": True, **layout_setting(self.adapter, self.sandbox), **settings,
                      **self._context(item, spec.directory), **self._stop(spec, item)},
        )
        if self.on_launch:
            self.on_launch(launch, index)
        return self._go(
            launch, item, spec.directory, skill, workspace, home, index,
            annotate={"task_id": task_id, "parity": parity},
        )


@dataclass(slots=True)
class ForkSource:
    """What a fork relaunches: a recorded run, and how it was run."""

    run: Run
    #: The step the fork goes live at.
    step: int
    item: Eval
    spec_dir: Path
    #: The frozen copy of the skill folder the run was given.
    skill: Path | None
    #: A wording to swap in from the step on, as `tuningfork.replay.Variant.to_json`.
    variant: dict[str, Any] = field(default_factory=dict)
    #: Whether breakpoints may hold the fork's live calls: yes for a single fork being
    #: debugged, no for an experiment's, which one would hang.
    breakpoints: bool = True
    #: Snapshots of the source's starting workspace and skill folder, when it has them: a
    #: fork starts from exactly what its source did, whatever the spec says now.
    initial_commit: str | None = None
    skill_commit: str | None = None
    #: The conversation the source continued (its launch's `context` setting), when it did.
    context: dict[str, Any] | None = None
    #: Where the source stopped beside its skill (its launch's `stop_at_next_skill`), when it did.
    stop: dict[str, Any] | None = None


class ForkRunner(Launcher):
    """Replays a recorded run to a relaunched harness, N times, live from one step."""

    def fork(
        self,
        source: ForkSource,
        n: int,
        *,
        on_progress: Callable[[RunOutcome], None] | None = None,
        experiment_id: str | None = None,
    ) -> list[RunOutcome]:
        group = uuid.uuid4().hex[:8]

        def one(index: int) -> RunOutcome:
            return self._fork_once(source, group, index, experiment_id)

        return self._many(n, one, on_progress)

    def _fork_once(
        self, source: ForkSource, group: str, index: int, experiment_id: str | None
    ) -> RunOutcome:
        # Laid out like a batch's, <group>/<index>/{workspace,home}, so per-run paths in
        # what the harness sends are masked the same way (analysis.normalize.RUN_PATH).
        root = self.workdir / group / str(index)
        workspace, home = root / "workspace", root / "home"
        run = source.run
        launch = self.store.create_launch(
            kind="fork",
            label=f"{run.label or run.id[:8]}@{source.step}.{index}",
            harness=self.adapter.name,
            model=self.model,
            task_id=run.task_id,
            eval=eval_record(source.item, source.spec_dir),
            skill_dir=str(source.skill.resolve()) if source.skill else None,
            workspace=str(workspace.resolve()),
            home=str(home.resolve()),
            settings={"breakpoints": source.breakpoints, **layout_setting(self.adapter, self.sandbox),
                      **({"context": source.context} if source.context else {}),
                      **({"stop_at_next_skill": source.stop} if source.stop else {})},
            source_run=run.id,
            source_step=source.step,
            variant=source.variant,
            experiment_id=experiment_id,
            sample_idx=index,
        )
        if self.on_launch:
            self.on_launch(launch, index)
        parity = {"fork_of": run.id, "fork_step": source.step}
        if source.variant:
            parity["variant"] = {k: v for k, v in source.variant.items() if not k.endswith("_text")}
        return self._go(
            launch, source.item, source.spec_dir, source.skill, workspace, home, index,
            annotate={"task_id": run.task_id, "parity": parity},
            restore=source.initial_commit,
            restore_skill=source.skill_commit,
        )


@contextmanager
def _terminate_as_interrupt() -> Iterator[None]:
    """While harnesses run from this process's main thread, a SIGTERM stops them as Ctrl-C
    does, through the interrupt's cleanup: each runs in a session of its own, so a `kill` of
    tuningfork alone would leave them running. Elsewhere — a job's worker thread inside
    `serve` — the job's own cancel does it."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _kill(process: subprocess.Popen) -> None:
    """Kill a harness and everything it started."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def skill_places(adapter: Any, home: Path, skill: Path) -> list[Path]:
    """Where a launch's harness is given a skill folder: where its adapter puts it — Hermes,
    in the category folder the skill declares — or, for an adapter that names only the
    folders its harness reads skills from, a folder of the skill's name in each."""
    place = getattr(adapter, "skill_folder", None)
    if place is not None:
        return [place(home, skill)]
    return [root / skill.name for root in adapter.skill_roots(home)]


def layout_setting(adapter: Any, sandbox: Any = None) -> dict[str, Any]:
    """How a launch is laid out, for its settings: how the adapter lays out its files, how
    its harness was set up (an adapter's `settings`: Hermes's options), and the sandbox it
    ran in, if any. A fork of the run is laid out the same way, so the paths it replays
    exist (see `HermesAdapter.layout`, `tuningfork.sandbox`)."""
    layout = getattr(adapter, "layout", None)
    found: dict[str, Any] = {"layout": layout} if layout else {}
    found.update(getattr(adapter, "settings", None) or {})
    if sandbox is not None:
        found["sandbox"] = sandbox.to_json()
    return found


def prepare_home(adapter: Any, home: Path, *, base_url: str, model: str, max_turns: int) -> Path:
    """`adapter.prepare_home`, passing the turn limit to an adapter that takes one.

    Hermes writes it into its config as well as taking it on the command line; an adapter
    written before the argument existed is still called the old way.
    """
    parameters = inspect.signature(adapter.prepare_home).parameters
    if "max_turns" in parameters:
        return adapter.prepare_home(home, base_url=base_url, model=model, max_turns=max_turns)
    return adapter.prepare_home(home, base_url=base_url, model=model)


def eval_record(item: Eval, spec_dir: Path) -> dict[str, Any]:
    """The eval as a launch was given it: enough to launch it again without the spec."""
    return {
        "id": item.id,
        "prompt": item.prompt,
        "expected_output": item.expected_output,
        "files": item.files,
        "expectations": item.expectations,
        "tuningfork": item.tuningfork.to_dict(),
        # Where a file named by path, not given inline, is found.
        "spec_dir": str(Path(spec_dir).resolve()),
    }
