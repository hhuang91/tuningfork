"""Localizing where runs stop agreeing.

Given several runs of the same task, find the first step at which they split into different
normalized behaviours, and show the competing behaviours side by side. That step is where
the instruction text is worth arguing about — and in Phase 2 it becomes the checkpoint an
A/B test resamples from.

Alignment is by step index. This is the honest first cut: it is correct whenever runs
proceed in lockstep, which is the common case, and it degrades predictably when they do not
— a retry shifts a run by one and shows up as a divergence one step early. Tolerating
retries and reordered parallel calls needs real sequence alignment, which is deferred
rather than faked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tuningfork.analysis.normalize import Outcome, recorded_outcome
from tuningfork.analysis.stats import Rate, rate
from tuningfork.store.db import Store

ENDED = Outcome("ended", "ended", "run ended")


@dataclass(slots=True)
class OutcomeGroup:
    """One competing behaviour at a step, and which runs took it."""

    outcome: Outcome
    run_ids: list[str]
    rate: Rate

    @property
    def count(self) -> int:
        return len(self.run_ids)


@dataclass(slots=True)
class StepComparison:
    index: int
    groups: list[OutcomeGroup]

    @property
    def diverged(self) -> bool:
        return len(self.groups) > 1

    @property
    def majority(self) -> OutcomeGroup:
        return self.groups[0]


@dataclass(slots=True)
class DivergenceReport:
    run_ids: list[str]
    steps: list[StepComparison] = field(default_factory=list)
    divergence_index: int | None = None
    #: Runs grouped by the skill files they were recorded with. Empty signature: unknown.
    skill_versions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def mixed_skills(self) -> bool:
        """Whether the compared runs were recorded under different skill files.

        A comparison across wordings measures the wording change, not the model's
        variance. Phase 1 published exactly this mistake — one of the "same" runs had been
        recorded under an edited skill — and the recorded hashes are what exposed it.
        """
        return sum(1 for signature in self.skill_versions if signature) > 1

    @property
    def compared(self) -> int:
        return len(self.run_ids)

    @property
    def agreed_steps(self) -> int:
        """How many leading steps every run performed identically."""
        return self.divergence_index if self.divergence_index is not None else len(self.steps)

    @property
    def diverged(self) -> bool:
        return self.divergence_index is not None

    def at_divergence(self) -> StepComparison | None:
        if self.divergence_index is None:
            return None
        return self.steps[self.divergence_index]

    def to_json(self, results: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """The report, with each behaviour's check pass rate when `results` gives the runs'
        verdicts — which of the competing behaviours at the divergence step goes on to pass."""
        from tuningfork.analysis.conditions import counted_verdict, pass_rate

        results = results or {}

        def passed(run_ids: list[str]) -> dict[str, Any] | None:
            found = pass_rate([results.get(run_id, {}) for run_id in run_ids])
            return found.to_json() if found is not None else None

        return {
            "run_ids": self.run_ids,
            # Each run's own verdict, counted as the behaviours' pass rates count it.
            "verdicts": {run_id: counted_verdict(results.get(run_id)) for run_id in self.run_ids},
            "divergence_index": self.divergence_index,
            "agreed_steps": self.agreed_steps,
            "mixed_skills": self.mixed_skills,
            "steps": [
                {
                    "index": comparison.index,
                    "diverged": comparison.diverged,
                    "groups": [
                        {
                            "outcome": group.outcome.to_json(),
                            "run_ids": group.run_ids,
                            "rate": group.rate.to_json(),
                            "passed": passed(group.run_ids),
                        }
                        for group in comparison.groups
                    ],
                }
                for comparison in self.steps
            ],
        }


def compare_runs(
    store: Store,
    run_ids: Sequence[str],
    normalizers: Iterable[str] = (),
) -> DivergenceReport:
    """Compare runs step by step and locate the first divergence."""
    if len(run_ids) < 2:
        raise ValueError("comparing runs needs at least two of them")

    steps_by_run = {run_id: store.steps(run_id) for run_id in run_ids}
    longest = max((len(steps) for steps in steps_by_run.values()), default=0)
    report = DivergenceReport(run_ids=list(run_ids))
    normalizer_names = list(normalizers)

    versions: dict[str, list[str]] = defaultdict(list)
    for run_id in run_ids:
        run = store.get_run(run_id)
        versions[(run.skill_version if run else None) or ""].append(run_id)
    report.skill_versions = dict(versions)

    for index in range(longest):
        members: dict[str, list[str]] = defaultdict(list)
        outcomes: dict[str, Outcome] = {}
        for run_id, steps in steps_by_run.items():
            if index < len(steps):
                step = steps[index]
                outcome = recorded_outcome(store.response(step), step.error, normalizer_names)
            else:
                # A run that stopped early is a real difference in behaviour, not a gap.
                outcome = ENDED
            members[outcome.key].append(run_id)
            outcomes.setdefault(outcome.key, outcome)

        groups = [
            OutcomeGroup(
                outcome=outcomes[key],
                run_ids=run_list,
                rate=rate(len(run_list), len(run_ids)),
            )
            for key, run_list in members.items()
        ]
        # Largest cluster first: the report reads as "most runs did X, but N did Y".
        groups.sort(key=lambda group: (-group.count, group.outcome.label))

        comparison = StepComparison(index=index, groups=groups)
        report.steps.append(comparison)
        if comparison.diverged and report.divergence_index is None:
            report.divergence_index = index

    return report
