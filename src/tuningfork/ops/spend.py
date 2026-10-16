"""What work cost: the model calls behind a run, an experiment, a job, a chain or a task,
counted from the records — calls, tokens in and out, and time. Time is each call's wall time
as the proxy or the experiment measured it, and the server's own where it reported it
(llama.cpp puts `timings` in its answer); a step answered from a recording cost nothing.

Two arms compared need equal budgets, or their spend reported beside their result: this is the
same count for both — a harness's steps, side threads and subagents included, and an
experiment's samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store


@dataclass(slots=True)
class Spend:
    """Model calls counted: how many, how many failed, the tokens in and out, and the time."""

    calls: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Milliseconds each call took, as the proxy or the experiment measured it, summed.
    wall_ms: float = 0.0
    #: Milliseconds the server said it spent, summed over the calls that reported it.
    server_ms: float = 0.0
    #: Calls whose server reported no timings: `server_ms` leaves them out.
    unmeasured: int = 0

    def __add__(self, other: Spend) -> Spend:
        return Spend(
            calls=self.calls + other.calls,
            failed=self.failed + other.failed,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            wall_ms=self.wall_ms + other.wall_ms,
            server_ms=self.server_ms + other.server_ms,
            unmeasured=self.unmeasured + other.unmeasured,
        )

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_json(self) -> dict[str, Any]:
        measured = self.calls - self.unmeasured
        return {
            "calls": self.calls,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens": self.tokens,
            "wall_s": round(self.wall_ms / 1000, 1),
            "server_s": round(self.server_ms / 1000, 1) if measured > 0 else None,
            "unmeasured": self.unmeasured,
        }


def count_call(usage: Any, timings: Any, response: Any, error: Any) -> Spend:
    """One model call's cost, from what its record keeps: OpenAI's `usage` or Anthropic's
    (`input_tokens`, `output_tokens`), the proxy's `total_ms`, and the server's `timings`."""
    usage = usage if isinstance(usage, dict) else {}
    timings = timings if isinstance(timings, dict) else {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens")) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens")) or 0
    served = response.get("timings") if isinstance(response, dict) else None
    server_ms: float | None = None
    if isinstance(served, dict) and any(key in served for key in ("prompt_ms", "predicted_ms")):
        server_ms = float(served.get("prompt_ms") or 0) + float(served.get("predicted_ms") or 0)
    return Spend(
        calls=1,
        failed=1 if error else 0,
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
        wall_ms=float(timings.get("total_ms") or 0),
        server_ms=server_ms or 0.0,
        unmeasured=0 if server_ms is not None else 1,
    )


def run_spend(store: Store, run_id: str) -> Spend:
    """One run's steps — this thread's own. A step answered from a recording cost nothing."""
    total = Spend()
    for step in store.steps(run_id):
        if step.served:
            continue
        total += count_call(step.usage, step.timings, store.response(step), step.error)
    return total


def launch_spend(store: Store, launch_id: str) -> Spend:
    """Every thread of a launch: the task's conversation, its subagents, the harness's review."""
    total = Spend()
    for run in store.launch_runs(launch_id):
        total += run_spend(store, run.id)
    return total


def main_run_spend(store: Store, run: Run) -> Spend:
    """A run with its launch's other threads, when tuningfork launched it; alone otherwise."""
    return launch_spend(store, run.launch_id) if run.launch_id else run_spend(store, run.id)


def sampled_spend(store: Store, experiment_id: str) -> Spend:
    """The model calls an experiment made itself: one a sample, answered at the step. A sample
    run to the end is a fork, whose calls its launch's runs recorded; one with no run made no
    model call — its harness ended before its first — and its time is the harness's, never a
    call's."""
    experiment = store.get_experiment(experiment_id)
    if experiment is None or (experiment.settings or {}).get("reach") == "end":
        return Spend()
    total = Spend()
    for sample in store.samples(experiment_id):
        if not sample.run_id:
            total += count_call(sample.usage, sample.timings, store.sample_response(sample), sample.error)
    return total


def experiment_spend(store: Store, experiment_id: str) -> Spend:
    """An experiment's samples. A sample run to the end is a fork: counted as its launch."""
    total = sampled_spend(store, experiment_id)
    for sample in store.samples(experiment_id):
        if sample.run_id:
            run = store.get_run(sample.run_id)
            if run is not None:
                total += main_run_spend(store, run)
    return total


def job_spend(store: Store, links: dict[str, Any]) -> Spend:
    """What a job launched and sampled, from the links it recorded. A to-the-end experiment's
    samples are forks the job also launched: counted once, as launches."""
    total = Spend()
    for launch_id in links.get("launches") or []:
        total += launch_spend(store, launch_id)
    for experiment_id in links.get("experiments") or []:
        total += sampled_spend(store, experiment_id)
    return total


def chain_spend(store: Store, chain: dict[str, Any]) -> tuple[list[Spend], Spend]:
    """Each link's launch, and the whole chain."""
    links = [
        launch_spend(store, link["launch_id"]) if link.get("launch_id") else Spend()
        for link in chain.get("links") or []
    ]
    return links, sum(links, Spend())


@dataclass(slots=True)
class SpendRow:
    what: str
    spend: Spend

    def to_json(self) -> dict[str, Any]:
        return {"what": self.what, **self.spend.to_json()}


def spend_report(
    store: Store,
    *,
    runs: Sequence[str] = (),
    experiments: Sequence[str] = (),
    jobs: Sequence[str] = (),
    chains: Sequence[str] = (),
    tasks: Sequence[str] = (),
) -> list[SpendRow]:
    """A row for each thing named — by id, a prefix of it, or a run's label — in the order
    given; one not found is refused."""
    from tuningfork.ops.chain import get_chain

    rows: list[SpendRow] = []
    for reference in runs:
        run = store.get_run(reference)
        if run is None:
            raise OperationError(f"no run matching {reference!r}", code="not_found", field="run")
        label = f" ({run.label})" if run.label else ""
        rows.append(SpendRow(f"run {run.id[:8]}{label}", main_run_spend(store, run)))
    for reference in experiments:
        experiment = store.get_experiment(reference)
        if experiment is None:
            raise OperationError(f"no experiment matching {reference!r}", code="not_found", field="experiment")
        rows.append(SpendRow(
            f"experiment {experiment.id[:8]} ({experiment.kind}, {experiment.arm})",
            experiment_spend(store, experiment.id),
        ))
    for reference in jobs:
        record = store.get_job(reference)
        if record is None:
            raise OperationError(f"no job matching {reference!r}", code="not_found", field="job")
        rows.append(SpendRow(f"job {record.id[:8]} ({record.kind})", job_spend(store, record.links or {})))
    for reference in chains:
        chain = get_chain(store, reference)
        _, total = chain_spend(store, chain)
        rows.append(SpendRow(f"chain {chain['id'][:8]} ({len(chain.get('links') or [])} links)", total))
    for task_id in tasks:
        found = store.runs(limit=1_000_000, task_id=task_id)
        if not found:
            raise OperationError(f"no runs of task {task_id!r}", code="not_found", field="task")
        total = Spend()
        for run in found:
            total += main_run_spend(store, run)
        rows.append(SpendRow(f"task {task_id} ({len(found)} runs)", total))
    return rows
