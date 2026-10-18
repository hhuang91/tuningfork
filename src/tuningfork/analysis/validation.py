"""Judging a validated edit: whether it may be approved, check by check.

A validation (`ops.validate`) runs a proposal three ways, each arm against the original:
forks from the refinement's checkpoint to the end (does hitting the target carry the task?),
fresh full runs of the training task (does the edit disturb earlier steps?), and fresh full
runs of each held-out task (does it break what it was not made for?). This module reads the
verdicts and nothing else: it is pure, and every check it makes carries the raw counts and
the name of the rule that judged it, so a record can be judged again under a later rule.

The rules are placeholders until the statistics are rethought: at five runs an arm, "not
significantly worse" passes nearly anything, so it says so ("low-powered"), with how many
runs it would take to tell a real drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from tuningfork.analysis.stats import Rate, fisher_exact, rate, samples_needed

#: How runs are counted: a run that timed out failed — an edit that makes the model loop must
#: not drop out of its own count — and a run the infrastructure broke is left out, and shown.
COUNTING = "timeouts-fail@1"
#: Not significantly worse: a two-sided Fisher test, worse only when significant and down.
NOT_WORSE = "fisher-two-sided@1"
#: The edit's own task, from the target forward: passes at least as often, and at least once.
AT_LEAST_AS_OFTEN = "at-least-as-often@1"
#: A held-out task the original fails more often than not: the edit must pass it more often.
#: An edit that fixes only the training task — one that writes in its answers — leaves it
#: failing as before, and only this catches it, as it caught one in use. Not a significance
#: test: the person judges the size of the gain, as with G3.
LIFTS_WHERE_FAILING = "lifts-where-original-fails@1"
RULES = (COUNTING, NOT_WORSE, AT_LEAST_AS_OFTEN, LIFTS_WHERE_FAILING)
#: Below this pass rate the original "fails" a task: a held-out task it fails must improve, and
#: from a step it passes at least this often, the step may not decide the task.
FAILING = 0.5


@dataclass(slots=True)
class Arm:
    """One arm's runs, counted under `COUNTING`."""

    passed: int
    counted: int
    timeouts: int
    planned: int
    drawn: int
    #: Runs left out, and why: what the infrastructure broke, not the edit.
    excluded: list[str] = field(default_factory=list)
    servers: list[str] = field(default_factory=list)

    @property
    def rate(self) -> Rate | None:
        return rate(self.passed, self.counted) if self.counted else None

    def to_json(self) -> dict[str, Any]:
        found = self.rate
        return {
            "passed": self.passed, "counted": self.counted, "timeouts": self.timeouts,
            "planned": self.planned, "drawn": self.drawn, "excluded": self.excluded,
            "servers": self.servers, "rate": found.to_json() if found else None,
        }


def count(verdicts: Sequence[dict[str, Any]], planned: int) -> Arm:
    passed = counted = timeouts = 0
    excluded, servers = [], []
    for verdict in verdicts:
        if verdict.get("server_id") and verdict["server_id"] not in servers:
            servers.append(verdict["server_id"])
        if verdict.get("timed_out"):
            timeouts += 1
            counted += 1
            continue
        if verdict.get("exclusion") or verdict.get("passed") is None:
            excluded.append(verdict.get("exclusion") or "no verdict")
            continue
        counted += 1
        passed += bool(verdict["passed"])
    return Arm(passed, counted, timeouts, planned, len(verdicts), excluded, servers)


@dataclass(slots=True)
class Comparison:
    original: Arm
    edit: Arm
    p_value: float | None
    #: up, down or same: the edit's pass rate against the original's.
    direction: str | None

    def to_json(self) -> dict[str, Any]:
        return {"original": self.original.to_json(), "edit": self.edit.to_json(),
                "p_value": self.p_value, "direction": self.direction}


def compare(original: Arm, edit: Arm) -> Comparison:
    if not original.counted or not edit.counted:
        return Comparison(original, edit, None, None)
    p = fisher_exact(edit.passed, edit.counted - edit.passed, original.passed, original.counted - original.passed)
    a, b = edit.passed / edit.counted, original.passed / original.counted
    return Comparison(original, edit, p, "up" if a > b else "down" if a < b else "same")


def runs_to_tell(original: Arm, drop: float = 0.3) -> int | None:
    """Runs an arm would need to tell a drop of `drop` from the original's rate: what a
    "not significantly worse" at this size cannot see."""
    if not original.counted:
        return None
    base = original.passed / original.counted
    target = base - drop if base - drop > 0 else None
    if target is None:
        return None
    needed = samples_needed(base, target)
    return needed or None


@dataclass(slots=True)
class Check:
    id: str
    name: str
    passed: bool
    reason: str
    rule: str | None = None
    low_powered: bool = False
    numbers: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "passed": self.passed, "reason": self.reason,
                "rule": self.rule, "low_powered": self.low_powered, "numbers": self.numbers}


@dataclass(slots=True)
class Gate:
    checks: list[Check]
    #: What a person should weigh that no check decides.
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def to_json(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [check.to_json() for check in self.checks],
                "notes": self.notes, "rules": list(RULES)}


def _pair(part: dict[str, Any] | None, planned: int) -> Comparison | None:
    if not part:
        return None
    return compare(count(part.get("original") or [], planned), count(part.get("edit") or [], planned))


def _not_worse(check_id: str, name: str, comparison: Comparison | None, alpha: float) -> Check:
    if comparison is None or comparison.p_value is None:
        return Check(check_id, name, False, "no runs were counted in one of the arms", NOT_WORSE)
    worse = comparison.direction == "down" and comparison.p_value < alpha
    numbers = comparison.to_json()
    needed = runs_to_tell(comparison.original)
    if needed:
        numbers["runs_to_tell_a_drop_of_30_points"] = needed
    if worse:
        reason = f"significantly worse (Fisher p = {comparison.p_value:.3g} < {alpha})"
    elif comparison.direction == "up" and comparison.p_value < alpha:
        reason = f"significantly better (Fisher p = {comparison.p_value:.3g})"
    else:
        reason = (f"not significantly worse (Fisher p = {comparison.p_value:.3g}); at "
                  f"{comparison.edit.counted} runs an arm only a large drop would show")
    return Check(check_id, name, not worse, reason, NOT_WORSE, low_powered=True, numbers=numbers)


def _lifts(check_id: str, name: str, comparison: Comparison | None) -> Check:
    """A held-out task the original fails must do better with the edit."""
    if comparison is None or comparison.p_value is None:
        return Check(check_id, name, False, "no runs were counted in one of the arms", LIFTS_WHERE_FAILING)
    o, e = comparison.original, comparison.edit
    counts = f"with the edit {e.passed}/{e.counted}, without it {o.passed}/{o.counted}"
    numbers = comparison.to_json()
    if o.passed / o.counted >= FAILING:
        return Check(check_id, name, True, f"the original already passes it {o.passed}/{o.counted}: nothing to lift",
                     LIFTS_WHERE_FAILING, numbers=numbers)
    if e.passed / e.counted > o.passed / o.counted:
        return Check(check_id, name, True, counts, LIFTS_WHERE_FAILING, numbers=numbers)
    return Check(
        check_id, name, False,
        f"{counts}: a held-out task the original fails must pass more often with the edit. An edit that "
        "fixes only the training task — one that writes in its answers — leaves it failing",
        LIFTS_WHERE_FAILING, numbers=numbers,
    )


def judge(
    record: dict[str, Any],
    *,
    target_matches: bool | None = None,
    refinement_open: bool = True,
    approved: bool = False,
    latest_screen: dict[str, Any] | None = None,
) -> Gate:
    """The gate a validation record meets, or not, check by check (G0–G8).

    `target_matches` is whether the skill's file still holds the text the refinement started
    from (None: not looked at); `refinement_open` and `approved` say where the refinement
    stands; `latest_screen` is the proposal's latest complete screen now
    (`ops.validate.latest_screen`; None: not looked at, or none left), which G0 reads over the
    one the record kept when validation began. Everything else is read from the record."""
    alpha = float(record.get("alpha", 0.05))
    minimum = int(record.get("minimum", 5))
    planned = record.get("planned") or {}
    checks: list[Check] = []
    notes: list[str] = []

    # The latest screen of the proposal is the evidence: found in review, one it survived passed
    # G0 after a later screen — more proposals in its family, or run to the end — had not kept it.
    kept = record.get("screen") or {}
    screen = latest_screen or kept
    named = str(screen.get("screen") or "?")[:8]
    if screen.get("survived"):
        reason = f"it survived screen {named}" + (", its latest" if latest_screen else "")
    elif latest_screen:
        reason = f"its latest complete screen, {named}, did not keep it" + (
            f" (it had survived {str(kept.get('screen') or '?')[:8]}, the latest when it was validated)"
            if kept.get("survived") and kept.get("screen") != latest_screen.get("screen") else "")
    else:
        reason = "no complete screen found it to shift the target"
    checks.append(Check(
        "G0", "survived a screen", bool(screen.get("survived")), reason,
        numbers={key: screen.get(key) for key in ("target", "baseline", "adjusted_p") if key in screen},
    ))

    held = record.get("held_out") or []
    planned_held = record.get("held_out_planned") or []
    checks.append(Check(
        "G1", "a held-out task was validated", bool(held),
        f"{len(held)} held-out task(s) validated" if held
        else f"the validation stopped before its held-out task(s) ran ({', '.join(planned_held)})" if planned_held
        else "the refinement's spec holds no held-out eval: nothing tested it on a task it was not made for",
    ))
    if record.get("held_out_recorded"):
        notes.append(f"runs of a held-out task are in the recordings ({', '.join(record['held_out_recorded'])}): "
                     "an agent reading them earlier could have fitted the edit to them")

    parts = [("forward", record.get("forward")), ("training", record.get("training"))]
    parts += [(f"held-out {entry.get('eval')}", entry) for entry in held]
    short, servers = [], []
    for name, part in parts:
        if not part:
            continue
        for arm in ("original", "edit"):
            # Runs counted, not drawn (found in review): the ones the infrastructure broke are
            # left out of every later check, so one counted run an arm met this and every later
            # check judged that one run.
            verdicts = part.get(arm) or []
            counted = count(verdicts, minimum)
            if counted.counted < minimum:
                left = len(verdicts) - counted.counted
                short.append(f"{name} {arm}: {counted.counted} of at least {minimum}"
                             + (f" counted, {left} left out ({', '.join(sorted(set(counted.excluded)))})" if left else ""))
        found = {verdict.get("server_id") for arm in ("original", "edit") for verdict in part.get(arm) or []} - {None}
        if len(found) > 1:
            servers.append(name)
    complete = bool(record.get("complete")) and not short and not servers
    reason = "every arm ran its runs on one server" if complete else "; ".join(
        (["the validation did not finish"] if not record.get("complete") else [])
        + short + [f"{name}: more than one server answered" for name in servers]
    )
    checks.append(Check("G2", "finished, with enough runs", complete, reason, numbers={"minimum": minimum}))

    forward = _pair(record.get("forward"), int(planned.get("forward") or minimum))
    if forward is None or forward.edit.counted == 0:
        checks.append(Check("G3", "its own task succeeds from the target forward", False,
                            "no forks of the edit were counted", AT_LEAST_AS_OFTEN))
    else:
        e, o = forward.edit, forward.original
        enough = e.passed >= 1 and (not o.counted or e.passed / e.counted >= o.passed / o.counted)
        checks.append(Check(
            "G3", "its own task succeeds from the target forward", enough,
            f"with the edit {e.passed}/{e.counted} forks passed, without it {o.passed}/{o.counted}"
            + ("" if enough else ": the edit must pass at least as often, and at least once"),
            AT_LEAST_AS_OFTEN, numbers=forward.to_json(),
        ))
        if o.counted and o.passed / o.counted >= FAILING:
            notes.append(f"from this step the original already passes {o.passed}/{o.counted}: the step may not "
                         "decide the task, and the screen's shift may not matter for it")

    training = _pair(record.get("training"), int(planned.get("training") or minimum))
    checks.append(_not_worse("G4", "the training task's full runs are not worse", training, alpha))

    held_checks = [
        _not_worse("G5", f"held-out task {entry.get('eval')} is not worse",
                   _pair(entry, int(planned.get("held_out") or minimum)), alpha)
        for entry in held
    ]
    checks += held_checks or [Check("G5", "held-out tasks are not worse", False, "no held-out task was validated", NOT_WORSE)]
    lifts = []
    for entry in held:
        comparison = _pair(entry, int(planned.get("held_out") or minimum))
        lifts.append(_lifts("G8", f"held-out task {entry.get('eval')} improves where the original fails", comparison))
        # A task the original passes is no test of a fix, only of harm: a drop G5 cannot call
        # significant at this size is still said.
        if (comparison is not None and comparison.direction == "down"
                and comparison.original.passed / comparison.original.counted >= FAILING):
            notes.append(f"held-out task {entry.get('eval')}: lower with the edit ({comparison.edit.passed}/"
                         f"{comparison.edit.counted} against {comparison.original.passed}/"
                         f"{comparison.original.counted}), though not significantly at this size")

    checks.append(Check(
        "G6", "the skill's file is still the one refined", target_matches is True,
        "it holds the text the refinement started from (line endings aside)" if target_matches
        else "not looked at" if target_matches is None
        else "the file has changed since the refinement started: refine the skill as it is now",
    ))
    checks.append(Check(
        "G7", "the refinement is open, and nothing in it is approved", refinement_open and not approved,
        "open" if refinement_open and not approved else "already approved" if approved else "closed",
    ))
    # Numbered after the others, which records and docs already name.
    checks += lifts or [Check("G8", "held-out tasks improve where the original fails", False,
                              "no held-out task was validated", LIFTS_WHERE_FAILING)]
    return Gate(checks, notes)
