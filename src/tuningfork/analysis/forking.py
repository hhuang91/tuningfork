"""The forking-token probe's statistics: which forced tokens changed what the model did.

Each probed position has arms: the base's own token forced there (the baseline), and each
alternative forced instead. Every arm is sampled n times from the same prefix, and each sample
ends in a behaviour. The question at a position is whether an alternative changed how often
the model reached the target behaviour, against the baseline at that position — the same
prefix, differing in one token.

- **One baseline per position**, never one pooled baseline: the chance of the target drifts
  along the path, and the contrast is only clean between arms that share their prefix.
- **Fisher's exact test** per alternative, against its position's baseline.
- **Holm's correction** over every alternative planned before sampling. The family is fixed
  by the plan and never shrinks: an arm that was not drawn — the probe was stopped — counts
  as p = 1.
- **The target was declared before sampling**: by default the base's own behaviour.

Beside the tests, not tested: how often the baseline reproduced the base's behaviour, and the
chance of the target from each position weighted by how likely each probed token was — Bigelow
et al.'s outcome estimate along the path. Where the run's own token seldom led back to its
behaviour, what it led to instead says why: the same tools with other arguments, and the target
is too fine a behaviour to be reproduced; something else, and the step had not decided yet — a
later token decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from tuningfork.analysis.distribution import Target, distribution
from tuningfork.analysis.normalize import Outcome, called_tools
from tuningfork.analysis.stats import Rate, fisher_exact, holm, rate


@dataclass(slots=True)
class ArmResult:
    arm: dict[str, Any]
    outcomes: list[Outcome]
    #: How often the arm reached the target: errors aside.
    target: Rate
    #: Samples meant for this arm, and drawn.
    planned: int
    p_value: float | None = None
    adjusted_p: float | None = None

    @property
    def drawn(self) -> int:
        return len(self.outcomes)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "target": self.target.to_json(),
            "drawn": self.drawn,
            "planned": self.planned,
            "errors": sum(1 for outcome in self.outcomes if outcome.is_error),
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
        }


@dataclass(slots=True)
class PositionResult:
    position: int
    part: str | None
    #: The text just before the position, and the base's token there, for reading.
    context: str
    baseline: ArmResult
    alternatives: list[ArmResult]
    alpha: float
    #: Why the run's own token here seldom led back to the target, when it did: "too_fine" or
    #: "undecided" (see `_why_missed`).
    why_missed: str | None = None

    def flips(self) -> list[ArmResult]:
        return [
            arm for arm in self.alternatives
            if arm.adjusted_p is not None and arm.adjusted_p < self.alpha
        ]

    @property
    def weighted(self) -> float | None:
        """The chance of the target from here, each probed token weighted by its probability:
        Bigelow et al.'s outcome estimate at this position, over the tokens probed."""
        arms = [arm for arm in (self.baseline, *self.alternatives) if arm.target.total]
        weight = sum(arm.arm.get("p") or 0.0 for arm in arms)
        if not arms or weight <= 0:
            return None
        return sum((arm.arm.get("p") or 0.0) * arm.target.point for arm in arms) / weight

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "part": self.part,
            "context": self.context,
            "baseline": self.baseline.to_json(),
            "alternatives": [
                {**arm.to_json(), "flips": arm in self.flips(),
                 "direction": _direction(self.baseline.target, arm.target)}
                for arm in self.alternatives
            ],
            "weighted": self.weighted,
            "why_missed": self.why_missed,
        }


def _direction(baseline: Rate, arm: Rate) -> str:
    if arm.point > baseline.point:
        return "toward"
    if arm.point < baseline.point:
        return "away"
    return "unchanged"


@dataclass(slots=True)
class ForkingReport:
    positions: list[PositionResult]
    target: str
    alpha: float
    #: Alternatives planned: the size of the family Holm corrects over.
    family: int
    #: Every sample's behaviour, lettered across the whole probe.
    behaviours: Any
    letters: dict[str, str] = field(default_factory=dict)
    #: How often the baseline arms, taken together, reproduced the base's behaviour.
    reproduced: Rate | None = None
    #: Continuations planned for each arm.
    per_arm: int = 0

    @property
    def detectable(self) -> int | None:
        """How many of an arm's continuations must change for a forced token to show, at best:
        a probe that found nothing says so, never that nothing changed. Measured on
        find-the-file: 'Wait' led to the search 4 times in 10 and 'According' once, and the
        page said no forced token had changed what the model did."""
        return detectable(self.per_arm, self.family, self.alpha) if self.per_arm else None

    def missed(self, why: str) -> list[int]:
        """The positions whose own token seldom led back to the target, for that reason."""
        return [position.position for position in self.positions if position.why_missed == why]

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "alpha": self.alpha,
            "family": self.family,
            "positions": [position.to_json() for position in self.positions],
            "flips": sum(len(position.flips()) for position in self.positions),
            "behaviours": self.behaviours.to_json(),
            "letters": self.letters,
            "reproduced": self.reproduced.to_json() if self.reproduced else None,
            "missed": {why: self.missed(why) for why in ("too_fine", "undecided")},
            "per_arm": self.per_arm,
            "detectable": self.detectable,
        }


def forking_report(
    arms: Sequence[dict[str, Any]],
    outcomes_by_arm: Sequence[Sequence[Outcome]],
    *,
    per_arm: int,
    target: Target,
    alpha: float = 0.05,
    contexts: dict[int, str] | None = None,
    parts: dict[int, str | None] | None = None,
) -> ForkingReport:
    """Compare every alternative arm with its position's baseline arm.

    `arms` are the plan's, each {"position", "id", "text", "p", "original"}; `outcomes_by_arm`
    holds the behaviours drawn for each, in the same order.
    """
    results = []
    for arm, outcomes in zip(arms, outcomes_by_arm):
        kept = [outcome for outcome in outcomes if not outcome.is_error]
        hits = sum(1 for outcome in kept if target.matches(outcome))
        results.append(ArmResult(arm=dict(arm), outcomes=list(outcomes), target=rate(hits, len(kept)), planned=per_arm))

    by_position: dict[int, list[ArmResult]] = {}
    for result in results:
        by_position.setdefault(int(result.arm["position"]), []).append(result)

    tests: list[ArmResult] = []
    positions = []
    for position in sorted(by_position):
        group = by_position[position]
        baseline = next((r for r in group if r.arm.get("original")), None)
        if baseline is None:
            continue
        alternatives = [r for r in group if not r.arm.get("original")]
        for arm in alternatives:
            if arm.target.total and baseline.target.total:
                arm.p_value = fisher_exact(
                    baseline.target.successes, baseline.target.total - baseline.target.successes,
                    arm.target.successes, arm.target.total - arm.target.successes,
                )
            else:
                arm.p_value = 1.0  # not drawn: counted, never dropped from the family
            tests.append(arm)
        positions.append(PositionResult(
            position=position,
            part=(parts or {}).get(position),
            context=(contexts or {}).get(position, ""),
            baseline=baseline,
            alternatives=alternatives,
            alpha=alpha,
        ))
    for arm, adjusted in zip(tests, holm([arm.p_value or 1.0 for arm in tests])):
        arm.adjusted_p = adjusted

    every = [outcome for result in results for outcome in result.outcomes]
    reached = [outcome for outcome in every if target.matches(outcome)]
    if target.spec.get("kind") == "key":
        reached.append(Outcome("tool_call", target.spec["value"], ""))
    target_tools = {tools for tools in map(called_tools, reached) if tools}
    for position in positions:
        position.why_missed = _why_missed(position.baseline, target, target_tools)
    behaviours = distribution(every)
    baselines = [outcome for position in positions for outcome in position.baseline.outcomes if not outcome.is_error]
    reproduced = rate(sum(1 for outcome in baselines if target.matches(outcome)), len(baselines)) if baselines else None
    return ForkingReport(
        positions=positions,
        target=target.description,
        alpha=alpha,
        family=len(tests),
        behaviours=behaviours,
        letters={cluster.outcome.key: cluster.letter for cluster in behaviours.clusters},
        reproduced=reproduced,
        per_arm=per_arm,
    )


def _why_missed(baseline: ArmResult, target: Target, target_tools: set[tuple[str, ...]]) -> str | None:
    """Why the run's own token seldom led back to the target, when it did.

    Found on the real stack: one warning covered two different things. After a step that ran
    code, the run's own token led to code 10/10, every script different from the base's:
    "too_fine", the target is too fine a behaviour to be reproduced. Early in a reasoning, it
    led to reading the file instead of searching for it, as often as not: "undecided", the step
    had not decided yet, and a later token decided it. None when the target's tools are
    unknown: then the misses cannot say which.
    """
    if not baseline.target.total or baseline.target.point >= 0.5 or not target_tools:
        return None
    misses = [outcome for outcome in baseline.outcomes if not outcome.is_error and not target.matches(outcome)]
    near = sum(1 for outcome in misses if called_tools(outcome) in target_tools)
    return "too_fine" if 2 * near > len(misses) else "undecided"


def detectable(n: int, family: int, alpha: float = 0.05) -> int | None:
    """The smallest number of an arm's n samples that must change — from all reaching the
    target to not, against a baseline that always does — for a single test to survive the
    correction as the first of `family`. A statement of power, before any sample is drawn."""
    for moved in range(1, n + 1):
        if fisher_exact(n, 0, n - moved, moved) * max(1, family) < alpha:
            return moved
    return None
