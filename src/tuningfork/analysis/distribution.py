"""Outcome distributions at a checkpoint, and A/B comparisons between two of them.

Two statistical commitments shape this module.

**No bare rates.** Every count comes with its Wilson interval, including the counts that
look decisive — 0/20 is consistent with a true rate of up to 16%.

**The target is declared before the data is seen.** An A/B asks whether one *specific*
behaviour became more or less common. If the target is picked after looking — "cluster C
moved the most, so that's the finding" — the test is answering a different, easier
question, and its p-value is wrong. So a comparison either carries a pre-declared target
and gets a single exact test, or it is explicitly exploratory and every cluster's p-value
is Bonferroni-corrected for the number of clusters examined.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tuningfork.analysis.normalize import Outcome, called_tools
from tuningfork.analysis.stats import Rate, fisher_exact, rate, samples_needed

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def cluster_letter(index: int) -> str:
    """A, B, ... Z, then AA, AB ... ZZ, AAA — many clusters is rare, but must not crash."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, len(LETTERS))
        letters = LETTERS[remainder] + letters
    return letters


def _ordered_keys(counts: dict[str, int], outcomes: dict[str, Outcome]) -> list[str]:
    # Largest first; ties broken by label so the lettering is stable between runs.
    return sorted(counts, key=lambda key: (-counts[key], outcomes[key].label))


@dataclass(slots=True)
class Cluster:
    letter: str
    outcome: Outcome
    members: list[int]
    rate: Rate
    #: How many of its samples came with reasoning; None where no sample here had any
    #: (`analysis.reasoning`): then a count says nothing.
    reasoned: int | None = None

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass(slots=True)
class Distribution:
    """Behaviours observed at one checkpoint, largest cluster first."""

    clusters: list[Cluster]
    #: Samples that produced a behaviour. Transport errors are excluded from every rate.
    total: int
    errors: int

    def by_letter(self, letter: str) -> Cluster | None:
        return next((c for c in self.clusters if c.letter == letter.upper()), None)

    def by_key(self, key: str) -> Cluster | None:
        return next((c for c in self.clusters if c.outcome.key == key), None)

    def to_json(self) -> dict:
        return {
            "total": self.total,
            "errors": self.errors,
            "clusters": [
                {
                    "letter": cluster.letter,
                    "outcome": cluster.outcome.to_json(),
                    "rate": cluster.rate.to_json(),
                    **({"reasoned": cluster.reasoned} if cluster.reasoned is not None else {}),
                }
                for cluster in self.clusters
            ],
        }


def _reasoned(indices: Sequence[int], reasoning: Sequence[int | None] | None) -> int | None:
    """How many of these samples came with reasoning — when any sample at all did."""
    from tuningfork.analysis.reasoning import reasons

    if reasoning is None or not reasons(reasoning):
        return None
    return sum(1 for index in indices if (reasoning[index] or 0) > 0)


def distribution(outcomes: Sequence[Outcome], reasoning: Sequence[int | None] | None = None) -> Distribution:
    """`reasoning`, aligned with `outcomes`, is each sample's reasoning in characters: with it,
    each cluster says how many of its samples reasoned."""
    members: dict[str, list[int]] = defaultdict(list)
    first: dict[str, Outcome] = {}
    errors = 0
    for index, outcome in enumerate(outcomes):
        if outcome.is_error:
            errors += 1
            continue
        members[outcome.key].append(index)
        first.setdefault(outcome.key, outcome)

    total = sum(len(indices) for indices in members.values())
    counts = {key: len(indices) for key, indices in members.items()}
    clusters = [
        Cluster(cluster_letter(position), first[key], members[key], rate(counts[key], total),
                _reasoned(members[key], reasoning))
        for position, key in enumerate(_ordered_keys(counts, first))
    ]
    return Distribution(clusters=clusters, total=total, errors=errors)


@dataclass(frozen=True, slots=True)
class Target:
    """The behaviour an A/B is about, declared before any sample is drawn."""

    description: str
    matches: Callable[[Outcome], bool]
    #: Serializable form, stored with the experiment so it can be re-reported against
    #: exactly the test it was run for.
    spec: dict[str, Any] = field(default_factory=dict)


def target_by_key(key: str, description: str) -> Target:
    return Target(
        description,
        lambda outcome: outcome.key == key,
        {"kind": "key", "value": key, "description": description},
    )


def target_by_pattern(pattern: str, *, code: bool = True) -> Target:
    """Match on the readable label, the full key, or — `code` — what the behaviour says.

    The key matters because labels truncate long arguments: a pattern naming something deep
    inside a written file would otherwise miss. What it says matters because a key keeps only
    a script's digest: a pattern naming what a script writes meets it there.
    A target stored before targets read it (no `code` in its spec) matches as declared.
    """
    compiled = re.compile(pattern)
    description = f"behaviour matching /{pattern}/"

    def matches(outcome: Outcome) -> bool:
        return bool(
            compiled.search(outcome.label) or compiled.search(outcome.key)
            or (code and outcome.text and compiled.search(outcome.text))
        )

    spec: dict[str, Any] = {"kind": "pattern", "value": pattern, "description": description}
    if code:
        spec["code"] = True
    return Target(description, matches, spec)


#: What a described target's conditions can say of a behaviour, as each reads.
RULES = {
    "calls": "calls {}",
    "not_calls": "does not call {}",
    "contains": "mentions {}",
    "not_contains": "does not mention {}",
}


def behaviour_text(outcome: Outcome) -> str:
    """What a behaviour says, as one text, whitespace collapsed: a reply's words, or each tool
    call's name and the values of its arguments, decoded. A regular expression meets these as
    a truncated label or a JSON-escaped key; a described target never does."""
    parts: list[str] = []
    calls: Any = None
    if outcome.kind == "tool_call" and outcome.key.startswith("tools:"):
        parts.extend(outcome.key[len("tools:"):].split(","))
    elif outcome.kind == "tool_call":
        try:
            calls = json.loads(outcome.key)
        except json.JSONDecodeError:
            calls = None
        for call in calls if isinstance(calls, list) else ():
            if isinstance(call, dict):
                parts.append(str(call.get("name") or ""))
                _values(call.get("arguments"), parts)
        if not isinstance(calls, list):
            parts.append(outcome.label)
    elif outcome.key.startswith("text:"):
        parts.append(outcome.key[len("text:"):])
    else:
        parts.append(outcome.label)
    return " ".join(" ".join(parts).split())


def _values(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _values(item, out)
    elif isinstance(value, list):
        for item in value:
            _values(item, out)
    elif value is not None:
        out.append(str(value))


def check_rules(rules: Any) -> list[dict[str, str]]:
    """A described target's conditions, checked: each one of `RULES` with a value."""
    if not isinstance(rules, (list, tuple)) or not rules:
        raise ValueError("a described target needs at least one condition")
    checked = []
    for number, rule in enumerate(rules, 1):
        kind = rule.get("if") if isinstance(rule, dict) else None
        if kind not in RULES:
            raise ValueError(f"condition {number}: {kind!r} is not one of {', '.join(RULES)}")
        value = rule.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"condition {number} ({kind}) needs a value")
        checked.append({"if": kind, "value": value.strip()})
    return checked


def target_by_rules(rules: Any, *, code: bool = True) -> Target:
    """A behaviour described by conditions, all of which must hold: it calls (or does not
    call) a tool; it mentions (or does not mention) a phrase, ignoring case and whitespace,
    in a reply's words or a tool call's arguments.

    Built in the web UI for a person who would rather not write a regular expression. Found in
    use: a pattern meets a label truncated at 90 characters or a JSON-escaped key, so "does not
    mention" cannot be written as one, and "A and B" needs lookaheads that stop at a newline.
    With `code`, "mentions" reads a script's own code; a target stored before (no `code` in
    its spec) reads what it did when declared.
    """
    checked = check_rules(rules)
    description = "a behaviour that " + ", and ".join(
        RULES[rule["if"]].format(json.dumps(rule["value"], ensure_ascii=False) if "contains" in rule["if"] else rule["value"])
        for rule in checked
    )

    def matches(outcome: Outcome) -> bool:
        tools = set(called_tools(outcome) or ())
        # As its key has it — a written JSON file canonical — and, with `code`, as the model
        # wrote it, a script's code whole: a phrase is mentioned if it is in either.
        texts = [behaviour_text(outcome).casefold()]
        if code and outcome.text:
            texts.append(" ".join(outcome.text.split()).casefold())
        for rule in checked:
            kind, value = rule["if"], rule["value"]
            phrase = " ".join(value.split()).casefold()
            said = any(phrase in found for found in texts)
            if (
                (kind == "calls" and value not in tools)
                or (kind == "not_calls" and value in tools)
                or (kind == "contains" and not said)
                or (kind == "not_contains" and said)
            ):
                return False
        return True

    spec: dict[str, Any] = {"kind": "rules", "rules": checked, "description": description}
    if code:
        spec["code"] = True
    return Target(description, matches, spec)


#: The task's own check as a target: judged by forks run to the end, never on an answer at
#: the step. For a failure only the check can see — a computed property of what the run
#: wrote, which no condition over an answer and no pattern expresses.
CHECK_TARGET = {"kind": "check", "description": "the task's check passes"}


def target_check() -> Target:
    """The task's check as the target. It counts no answer at a step: a screen of it runs
    each sample as a fork from the checkpoint to the end, and the check decides."""
    return Target(CHECK_TARGET["description"], lambda outcome: False, dict(CHECK_TARGET))


def target_from_spec(spec: dict[str, Any] | None) -> Target | None:
    if not spec:
        return None
    if spec.get("kind") == "check":
        return target_check()
    if spec.get("kind") == "key":
        return target_by_key(spec["value"], spec.get("description", spec["value"]))
    if spec.get("kind") == "pattern":
        return target_by_pattern(spec["value"], code=bool(spec.get("code")))
    if spec.get("kind") == "rules":
        return target_by_rules(spec["rules"], code=bool(spec.get("code")))
    raise ValueError(f"unknown target kind {spec.get('kind')!r}")


@dataclass(slots=True)
class TargetTest:
    description: str
    original: Rate
    variant: Rate
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_json(self) -> dict:
        return {
            "description": self.description,
            "original": self.original.to_json(),
            "variant": self.variant.to_json(),
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "direction": self.direction,
            # Beside a result that is not significant: how far from measuring it this was.
            "samples_to_detect": None if self.significant else self.samples_to_detect(),
        }

    @property
    def direction(self) -> str:
        if self.variant.point > self.original.point:
            return "up"
        if self.variant.point < self.original.point:
            return "down"
        return "unchanged"

    def samples_to_detect(self) -> int | None:
        """Per-arm samples needed to detect the observed difference at 80% power.

        Reported alongside a non-significant result, because "not significant" at twenty
        samples mostly means "not measured precisely enough", not "no effect".
        """
        low, high = self.original.point, self.variant.point
        if low == high:
            return None

        def clamp(p: float) -> float:
            return min(max(p, 0.01), 0.99)

        return samples_needed(clamp(low), clamp(high))


@dataclass(slots=True)
class ClusterShift:
    """One behaviour's rate in each arm, lettered by its pooled frequency."""

    letter: str
    outcome: Outcome
    original: Rate
    variant: Rate
    p_value: float
    #: Bonferroni-corrected for the number of clusters examined.
    adjusted_p: float
    #: How many of each arm's samples of it came with reasoning; None where neither arm
    #: shows any reasoning.
    original_reasoned: int | None = None
    variant_reasoned: int | None = None
    #: How many of its samples, both arms together, the declared target counted: all of them
    #: for a behaviour that is the target, none for one that is not — some, where a pattern
    #: meets what one script of the behaviour says and not another's. None with no target.
    target_counted: int | None = None


@dataclass(slots=True)
class ABComparison:
    original: Distribution
    variant: Distribution
    shifts: list[ClusterShift]
    target: TargetTest | None
    alpha: float

    @property
    def exploratory(self) -> bool:
        return self.target is None

    def significant_shifts(self) -> list[ClusterShift]:
        return [shift for shift in self.shifts if shift.adjusted_p < self.alpha]

    def to_json(self) -> dict:
        return {
            "exploratory": self.exploratory,
            "shifts": [
                {
                    "letter": shift.letter,
                    "outcome": shift.outcome.to_json(),
                    "original": shift.original.to_json(),
                    "variant": shift.variant.to_json(),
                    "p_value": shift.p_value,
                    "adjusted_p": shift.adjusted_p,
                    **({"original_reasoned": shift.original_reasoned, "variant_reasoned": shift.variant_reasoned}
                       if shift.original_reasoned is not None else {}),
                    **({"target_counted": shift.target_counted} if shift.target_counted is not None else {}),
                }
                for shift in self.shifts
            ],
            "target": self.target.to_json() if self.target else None,
        }


def compare_arms(
    original: Sequence[Outcome],
    variant: Sequence[Outcome],
    *,
    target: Target | None = None,
    alpha: float = 0.05,
    reasoning: tuple[Sequence[int | None], Sequence[int | None]] | None = None,
) -> ABComparison:
    """`reasoning`, each arm's samples' reasoning aligned with its outcomes: with it, each
    behaviour says how many of each arm's samples of it reasoned."""
    from tuningfork.analysis.reasoning import reasons

    shown = reasoning is not None and (reasons(reasoning[0]) or reasons(reasoning[1]))

    def reasoned(outcomes: Sequence[Outcome], chars: Sequence[int | None], key: str) -> int:
        return sum(1 for outcome, count in zip(outcomes, chars) if not outcome.is_error
                   and outcome.key == key and (count or 0) > 0)

    kept_original = [outcome for outcome in original if not outcome.is_error]
    kept_variant = [outcome for outcome in variant if not outcome.is_error]
    n_original, n_variant = len(kept_original), len(kept_variant)

    first: dict[str, Outcome] = {}
    for outcome in (*kept_original, *kept_variant):
        first.setdefault(outcome.key, outcome)
    count_original = Counter(outcome.key for outcome in kept_original)
    count_variant = Counter(outcome.key for outcome in kept_variant)
    pooled = {key: count_original[key] + count_variant[key] for key in first}

    # Each sample the target counts, by its behaviour: the rows the target's own rates add up.
    hits_original = [target.matches(outcome) for outcome in kept_original] if target is not None else []
    hits_variant = [target.matches(outcome) for outcome in kept_variant] if target is not None else []
    counted = Counter(
        outcome.key
        for outcome, hit in (*zip(kept_original, hits_original), *zip(kept_variant, hits_variant))
        if hit
    )

    ordered = _ordered_keys(pooled, first)
    tests = max(1, len(ordered))
    shifts = []
    for position, key in enumerate(ordered):
        a, c = count_original[key], count_variant[key]
        p_value = fisher_exact(a, n_original - a, c, n_variant - c)
        shifts.append(
            ClusterShift(
                letter=cluster_letter(position),
                outcome=first[key],
                original=rate(a, n_original),
                variant=rate(c, n_variant),
                p_value=p_value,
                adjusted_p=min(1.0, p_value * tests),
                original_reasoned=reasoned(original, reasoning[0], key) if shown else None,
                variant_reasoned=reasoned(variant, reasoning[1], key) if shown else None,
                target_counted=counted[key] if target is not None else None,
            )
        )

    target_test = None
    if target is not None:
        a, c = sum(hits_original), sum(hits_variant)
        target_test = TargetTest(
            description=target.description,
            original=rate(a, n_original),
            variant=rate(c, n_variant),
            p_value=fisher_exact(a, n_original - a, c, n_variant - c),
            alpha=alpha,
        )

    return ABComparison(
        original=distribution(original, reasoning[0] if reasoning else None),
        variant=distribution(variant, reasoning[1] if reasoning else None),
        shifts=shifts,
        target=target_test,
        alpha=alpha,
    )
