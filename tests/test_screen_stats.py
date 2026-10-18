"""Edits screened against one shared baseline (`analysis.screen`): Fisher per arm, Holm over
the family planned before sampling, and no verdict from a screen that stopped short."""

from __future__ import annotations

import pytest

from tuningfork.analysis.distribution import target_by_key
from tuningfork.analysis.normalize import Outcome
from tuningfork.analysis.screen import compare_to_baseline, needed_to_survive
from tuningfork.analysis.stats import fisher_exact, holm

HIT = Outcome("tool_call", "write:row_count", "write_file(row_count…)")
MISS = Outcome("tool_call", "write:count", "write_file(count…)")
ERROR = Outcome("error", "error:upstream", "the server failed")
TARGET = target_by_key(HIT.key, "the report names row_count")


def arm(hits: int, misses: int, errors: int = 0) -> list[Outcome]:
    return [HIT] * hits + [MISS] * misses + [ERROR] * errors


class TestTheScreen:
    def test_each_arm_is_tested_against_the_shared_baseline_with_holm_over_the_family(self):
        report = compare_to_baseline(arm(0, 20), {"p1": arm(20, 0), "p2": arm(10, 10), "p3": arm(0, 20)},
                                     target=TARGET, planned=20)
        raw = [fisher_exact(20, 0, 0, 20), fisher_exact(10, 10, 0, 20), fisher_exact(0, 20, 0, 20)]
        assert [a.p_value for a in report.arms] == raw
        assert [a.adjusted_p for a in report.arms] == holm(raw)  # in the order planned
        assert [a.name for a in report.survivors()] == ["p1", "p2"]
        assert report.arms[2].direction == "same" and report.worse() == []
        assert report.complete and report.family == 3

    def test_an_arm_never_drawn_still_counts_in_the_family(self):
        report = compare_to_baseline(arm(0, 20), {"p1": arm(7, 13), "p2": []}, target=TARGET, planned=20)
        first, never = report.arms
        assert never.p_value is None and never.adjusted_p is None and never.drawn == 0
        assert first.adjusted_p == pytest.approx(holm([first.p_value, 1.0])[0])
        assert first.adjusted_p == pytest.approx(min(1.0, 2 * first.p_value))

    def test_a_screen_that_stopped_short_has_rates_and_no_verdict(self):
        report = compare_to_baseline(arm(0, 10), {"p1": arm(10, 0)}, target=TARGET, planned=20)
        assert not report.complete
        assert report.arms[0].adjusted_p < 0.05  # it looks like a win...
        assert report.survivors() == []  # ...and is not called one
        data = report.to_json()
        assert data["arms"][0]["survives"] is None and data["survivors"] == []

    def test_a_wording_that_made_things_worse_says_so(self):
        report = compare_to_baseline(arm(20, 0), {"p1": arm(0, 20)}, target=TARGET, planned=20)
        assert [a.name for a in report.worse()] == ["p1"] and report.survivors() == []
        assert report.to_json()["arms"][0]["worse"] is True

    def test_transport_errors_are_drawn_but_never_counted(self):
        report = compare_to_baseline(arm(0, 18, errors=2), {"p1": arm(9, 9, errors=2)}, target=TARGET, planned=20)
        assert report.baseline.target.total == 18 and report.baseline.errors == 2 and report.baseline.drawn == 20
        assert report.arms[0].target.successes == 9 and report.complete

    def test_behaviours_are_lettered_across_every_arm(self):
        report = compare_to_baseline(arm(0, 20), {"p1": arm(15, 5)}, target=TARGET, planned=20)
        (first, second) = report.behaviours
        assert (first["letter"], first["count"], first["by_arm"]) == ("A", 25, {"original": 20, "p1": 5})
        assert (second["letter"], second["by_arm"]) == ("B", {"original": 0, "p1": 15})

    def test_each_behaviour_says_how_many_of_its_samples_the_target_counted(self):
        """The page badges each row target or not target: a row's own counts say how often a
        behaviour happened, never whether it was the one wanted."""
        report = compare_to_baseline(arm(0, 20), {"p1": arm(15, 5)}, target=TARGET, planned=20)
        assert [(found["outcome"]["label"], found["target_counted"]) for found in report.behaviours] == [
            (MISS.label, 0), (HIT.label, 15),
        ]


@pytest.mark.parametrize(
    "baseline, family, needed",
    [
        (0, 1, 5),    # from nothing, one proposal must reach 5 of 20
        (0, 3, 7),    # three proposals: the strongest is tested at alpha / 3
        (6, 3, 15),   # 14 of 20 would give an adjusted p near 0.077
        (19, 3, None),  # nothing could survive: refuse the screen before sampling
    ],
)
def test_what_an_arm_must_reach_to_survive(baseline, family, needed):
    assert needed_to_survive(baseline, 20, family) == needed
    if needed:
        p = fisher_exact(needed - 1, 20 - needed + 1, baseline, 20 - baseline)
        assert p * family >= 0.05  # one fewer would not have done


class TestForksToTheEnd:
    def test_each_arm_is_judged_by_the_check_with_the_same_correction(self):
        """A screen run to the end: a fork that passed is a hit, one that failed a miss, and one
        left out counts for neither — then the test is the step screen's."""
        from types import SimpleNamespace

        from tuningfork.analysis.screen import compare_verdicts

        def forks(passed: int, failed: int, excluded: int = 0) -> list:
            return ([SimpleNamespace(passed=True, exclusion=None)] * passed
                    + [SimpleNamespace(passed=False, exclusion=None)] * failed
                    + [SimpleNamespace(passed=None, exclusion="timed out")] * excluded)

        report = compare_verdicts(forks(2, 8), {"p1": forks(9, 1), "p2": forks(3, 6, 1)}, planned=10)
        raw = [fisher_exact(9, 1, 2, 8), fisher_exact(3, 6, 2, 8)]
        assert report.reach == "end" and report.target == "the task's check passes" and report.complete
        assert [arm.p_value for arm in report.arms] == raw and [arm.adjusted_p for arm in report.arms] == holm(raw)
        assert [arm.name for arm in report.survivors()] == ["p1"]
        assert report.arms[1].errors == 1 and report.arms[1].target.total == 9 and report.arms[1].drawn == 10
        assert report.to_json()["reach"] == "end" and report.baseline.target.successes == 2
        short = compare_verdicts(forks(2, 8), {"p1": forks(5, 0)}, planned=10)
        assert not short.complete and short.survivors() == []
