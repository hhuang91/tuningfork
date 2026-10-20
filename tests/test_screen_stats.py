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

    def test_a_letter_target_s_behaviours_keep_its_resample_s_letters(self):
        """A refinement whose target was a resample's B screens it as B, whichever behaviour
        the screen's own samples did most."""
        report = compare_to_baseline(arm(0, 20), {"p1": arm(15, 5)}, target=TARGET, planned=20,
                                     letters={HIT.key: "B", MISS.key: "A", "write:other": "C"})
        assert [(found["outcome"]["label"], found["letter"]) for found in report.behaviours] == [
            (MISS.label, "A"), (HIT.label, "B"),
        ]
        report = compare_to_baseline(arm(20, 0), {"p1": arm(20, 0)}, target=TARGET, planned=20,
                                     letters={HIT.key: "C"})
        assert [found["letter"] for found in report.behaviours] == ["C"]

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
                    + [SimpleNamespace(passed=None, exclusion="the replay departed from the recording at step 2")] * excluded)

        report = compare_verdicts(forks(2, 8), {"p1": forks(9, 1), "p2": forks(3, 6, 1)}, planned=10)
        raw = [fisher_exact(9, 1, 2, 8), fisher_exact(3, 6, 2, 8)]
        assert report.reach == "end" and report.target == "the task's check passes" and report.complete
        assert [arm.p_value for arm in report.arms] == raw and [arm.adjusted_p for arm in report.arms] == holm(raw)
        assert [arm.name for arm in report.survivors()] == ["p1"]
        assert report.arms[1].errors == 1 and report.arms[1].target.total == 9 and report.arms[1].drawn == 10
        assert report.to_json()["reach"] == "end" and report.baseline.target.successes == 2
        short = compare_verdicts(forks(2, 8), {"p1": forks(5, 0)}, planned=10)
        assert not short.complete and short.survivors() == []

    def test_a_fork_that_timed_out_failed_as_the_gate_counts_it(self):
        """An edit that makes the agent loop until it is stopped must not drop out of its own
        count: a fork that timed out is a failure, whatever the check made of what it left.
        Found in review: left out, such an arm counted only the forks that finished, and an
        edit whose agent ran out the clock half the time survived its screen."""
        from types import SimpleNamespace

        from tuningfork.analysis.screen import compare_verdicts

        original = [SimpleNamespace(passed=True, exclusion=None)] * 2 + [SimpleNamespace(passed=False, exclusion=None)] * 8
        looping = ([SimpleNamespace(passed=True, exclusion=None)] * 5
                   + [SimpleNamespace(passed=True, exclusion="timed out")] * 3
                   + [SimpleNamespace(passed=False, exclusion="timed out")] * 2)
        report = compare_verdicts(original, {"p1": looping}, planned=10)
        (arm,) = report.arms
        assert (arm.target.successes, arm.target.total, arm.errors, arm.timeouts) == (5, 10, 0, 5)
        assert report.complete and report.survivors() == [] and report.to_json()["arms"][0]["timeouts"] == 5


def test_forks_to_the_end_count_a_timeout_as_a_failure_wherever_they_are_read(store):
    """The same count in an A/B or a resample to the end (`ops.forks.arm_summary`), at each of a
    screen's checkpoints, and among the original's forks a later screen counts again."""
    from tuningfork.ops.forks import arm_summary
    from tuningfork.ops.screen import checkpoint_breakdown, judged_forks

    experiment = store.create_experiment(kind="resample", arm="original", run_id="r", step_idx=1, step_id=None,
                                         request={"messages": []}, settings={"n": 4, "reach": "end"}, status="done")
    departed = "the replay departed from the recording at step 0"
    for idx, (passed, exclusion) in enumerate([(True, None), (False, None), (True, "timed out"), (None, departed)]):
        store.add_sample(experiment.id, idx, seed=None, response=None, outcome_kind="error", outcome_key=None,
                         outcome_label=None, error=exclusion, run_id=f"fork{idx}", passed=passed, exclusion=exclusion)

    summary = arm_summary(store, experiment)
    assert (summary.passed, summary.counted, summary.timeouts) == (1, 3, 1)
    shown = summary.to_json()
    assert [entry["why"] for entry in shown["exclusions"]] == [departed]
    assert [fork["timed_out"] for fork in shown["forks"]] == [False, False, True, False]
    assert judged_forks(store, experiment) == 3
    here, elsewhere = checkpoint_breakdown(store, {"original": [experiment]}, [{"run": "r", "step": 1}, {"run": "s", "step": 1}])
    assert here["arms"] == {"original": {"passed": 1, "of": 3}} and elsewhere["arms"] == {}

    # What an agent reads says so, the fork that timed out a failure and not left out.
    from tuningfork.ops import compact

    seen = compact.to_end(shown)
    assert seen["timed_out"] == 1 and [entry["why"] for entry in seen["left_out"]] == [departed]
    assert seen["forks"][2] == {"idx": 2, "run": "fork2", "passed": True, "timed_out": "counted as a failure"}
