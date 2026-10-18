"""Pass rates by condition: runs of one task, under one harness, model, skill version and
server.

A rate pooled across wordings or models measures none of them. Once a task has been run
before and after an edit — which is the whole workflow — its runs are two samples, not one,
and each gets its own rate. So do runs answered by different servers: LM Studio and
llama-server can both report the same model name for one GGUF, with chat templates that
need not agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from tuningfork.analysis.stats import Rate, rate
from tuningfork.store.db import Run


def pass_rate(results: Iterable[dict[str, Any]]) -> Rate | None:
    """The check pass rate over stored run results (`runner.RunOutcome.to_result`).

    None when no run has a verdict. Runs the batch marked unusable are left out: a run
    whose skill changed underneath it, that timed out, was edited at a breakpoint or whose
    model calls failed describes neither wording.
    """
    verdicts = [
        result["passed"]
        for result in results
        if result.get("usable") and result.get("passed") is not None
    ]
    return rate(sum(verdicts), len(verdicts)) if verdicts else None


def exclusion_from(result: dict[str, Any] | None) -> str | None:
    """Why a run with this stored result (`runner.RunOutcome.to_result`) is not counted, in
    words; None when nothing in it says so."""
    result = result or {}

    def steps(key: str) -> str:
        return ", ".join(map(str, result[key]))

    if result.get("skills_changed"):
        return f"skills changed: {', '.join(result['skills_changed'])}"
    if result.get("timed_out"):
        return "timed out"
    if result.get("edited_steps"):
        return f"edited at a breakpoint (step {steps('edited_steps')})"
    if result.get("failed_steps"):
        return f"model call failed (step {steps('failed_steps')})"
    if result.get("thread_trouble"):
        return f"a call failed or was edited in {', '.join(result['thread_trouble'])}"
    departed = result.get("departed")
    if departed:
        where = f"step {departed.get('step')}"
        if departed.get("thread"):
            where = f"thread {departed['thread']} {where}"
        return f"the replay departed from the recording at {where}"
    if result.get("leaked"):
        return f"the original wording reached the model (step {steps('leaked')})"
    if result.get("reached") is False:
        return f"the harness never reached step {result.get('fork_step')}"
    return None


def counted_verdict(result: dict[str, Any] | None) -> str | None:
    """How one run fared, as `pass_rate` counts it: "passed" or "failed"; "not counted" for a
    verdict every rate leaves out; None for a run with no verdict."""
    result = result or {}
    if result.get("passed") is None:
        return None
    if not result.get("usable"):
        return "not counted"
    return "passed" if result["passed"] else "failed"


@dataclass(slots=True)
class Condition:
    task_id: str | None
    harness: str | None
    model: str | None
    #: `Run.skill_version`: a fingerprint of the skill files the runs were given.
    skill_version: str | None
    #: The server the runs were answered by (`Store.server`); None for runs recorded before
    #: servers were, which form a condition of their own.
    server_id: str | None = None
    runs: list[Run] = field(default_factory=list)

    @property
    def run_ids(self) -> list[str]:
        return [run.id for run in self.runs]

    @property
    def skill_file(self) -> str | None:
        """The file a batch installed as SKILL.md in place of the spec's own, if every run had
        it. Runs given the same text another way share the fingerprint, and the condition: a
        validation's original arm installs a copy of the skill's own file."""
        files = {run.parity.get("skill_file") or None for run in self.runs}
        return files.pop() if len(files) == 1 else None

    @property
    def passed(self) -> Rate | None:
        return pass_rate(run.result for run in self.runs if run.result)

    @property
    def uncounted(self) -> int:
        """Runs with a verdict that may not be counted: timed out, skill changed, edited."""
        return sum(1 for run in self.runs if run.result and not run.result.get("usable"))

    @property
    def first(self) -> float:
        return min(run.started_at for run in self.runs)

    @property
    def latest(self) -> float:
        return max(run.started_at for run in self.runs)


def conditions(runs: Iterable[Run]) -> list[Condition]:
    """Runs grouped by task, harness, model, skill version and server, oldest first.

    The skill version is the content the runs were given, not how it got there: a variant
    installed with `batch --skill` and the same text later copied into SKILL.md are one
    condition. A launch's subagent threads and an experiment's forks are not runs of the
    task, and are expected to have been left out already (`Store.runs` does).
    """
    groups: dict[tuple[str | None, ...], Condition] = {}
    for run in runs:
        key = (run.task_id, run.harness, run.model, run.skill_version, run.server_id)
        if key not in groups:
            groups[key] = Condition(*key)
        groups[key].runs.append(run)
    return sorted(groups.values(), key=lambda condition: condition.first)
