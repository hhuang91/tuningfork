"""The gate a validated edit must meet (`analysis.validation`): every check with its counts and
rule, low-powered where it is, and the cheats no check catches said plainly."""

from __future__ import annotations

from tuningfork.analysis.validation import AT_LEAST_AS_OFTEN, LIFTS_WHERE_FAILING, NOT_WORSE, count, judge


def runs(passed: int, failed: int = 0, *, timeouts: int = 0, broken: int = 0, server: str = "s1") -> list[dict]:
    found = [{"passed": True, "exclusion": None, "timed_out": False, "server_id": server} for _ in range(passed)]
    found += [{"passed": False, "exclusion": None, "timed_out": False, "server_id": server} for _ in range(failed)]
    found += [{"passed": None, "exclusion": "the run timed out", "timed_out": True, "server_id": server}
              for _ in range(timeouts)]
    found += [{"passed": None, "exclusion": "a model call failed", "timed_out": False, "server_id": server}
              for _ in range(broken)]
    return found


def record(**changes) -> dict:
    base = {
        "alpha": 0.05, "minimum": 5, "complete": True,
        "planned": {"forward": 5, "training": 5, "held_out": 5},
        "screen": {"survived": True, "screen": "abcdef12"},
        "forward": {"original": runs(0, 5), "edit": runs(5)},
        "training": {"original": runs(0, 5), "edit": runs(5)},
        "held_out": [{"eval": "2", "original": runs(0, 5), "edit": runs(5)}],
    }
    return {**base, **changes}


def by_id(gate) -> dict:
    return {check.id: check for check in gate.checks}


def test_a_fix_that_carries_through_meets_every_check():
    gate = judge(record(), target_matches=True)
    assert gate.passed, [check.reason for check in gate.failed()]
    checks = by_id(gate)
    assert [check.id for check in gate.checks] == ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    assert checks["G8"].rule == LIFTS_WHERE_FAILING and checks["G8"].passed
    assert checks["G3"].rule == AT_LEAST_AS_OFTEN and not checks["G3"].low_powered
    assert checks["G4"].rule == NOT_WORSE and checks["G4"].low_powered
    assert checks["G4"].reason.startswith("significantly better")  # 5/5 against 0/5 is more than not worse
    assert checks["G3"].numbers["edit"]["rate"]["successes"] == 5  # counts, not just a verdict


def test_timeouts_count_as_failures_and_broken_runs_are_left_out():
    arm = count(runs(3, timeouts=1, broken=1), planned=5)
    assert (arm.passed, arm.counted, arm.timeouts, arm.drawn) == (3, 4, 1, 5)
    assert arm.excluded == ["a model call failed"]


def test_the_edit_must_carry_its_own_task_at_least_as_often_and_at_least_once():
    """The decision: no significance test here — the person judges the size of the gain."""
    assert not by_id(judge(record(forward={"original": runs(0, 5), "edit": runs(0, 5)}), target_matches=True))["G3"].passed
    assert not by_id(judge(record(forward={"original": runs(4, 1), "edit": runs(3, 2)}), target_matches=True))["G3"].passed
    assert by_id(judge(record(forward={"original": runs(1, 4), "edit": runs(2, 3)}), target_matches=True))["G3"].passed


def test_a_significantly_worse_task_fails_and_not_worse_says_how_little_it_can_see():
    gate = judge(record(held_out=[{"eval": "2", "original": runs(5), "edit": runs(0, 5)}]), target_matches=True)
    held = by_id(gate)["G5"]
    assert not held.passed and "significantly worse" in held.reason
    fine = by_id(judge(record(training={"original": runs(3, 2), "edit": runs(2, 3)}), target_matches=True))["G4"]
    assert fine.passed and fine.low_powered and "only a large drop would show" in fine.reason
    assert fine.numbers["runs_to_tell_a_drop_of_30_points"] > 5


def test_what_is_short_or_mixed_is_named():
    gate = judge(record(training={"original": runs(0, 3), "edit": runs(5)},
                        forward={"original": runs(0, 5, server="s1"), "edit": runs(5, server="s2")}),
                 target_matches=True)
    reason = by_id(gate)["G2"].reason
    assert not gate.passed and "training original: 3 of at least 5" in reason and "forward: more than one server" in reason
    assert not by_id(judge(record(complete=False), target_matches=True))["G2"].passed


def test_runs_the_infrastructure_broke_are_not_enough_runs():
    """Found in review: G2 counted every run drawn, the ones left out too, so one counted run an
    arm met it, and every later check judged that one run — the gate passed."""
    one = {"original": runs(0, 5), "edit": runs(1, broken=4)}
    gate = judge(record(forward=one, training=one, held_out=[{"eval": "2", **one}]), target_matches=True)
    reason = by_id(gate)["G2"].reason
    assert not gate.passed and not by_id(gate)["G2"].passed
    assert "training edit: 1 of at least 5 counted, 4 left out (a model call failed)" in reason


def test_g0_reads_the_proposals_latest_screen_over_the_one_the_record_kept():
    """Found in review: a proposal that survived one screen passed G0 after a later screen —
    more proposals in its family, or one run to the end — had not kept it."""
    later = {"survived": False, "screen": "99887766aa"}
    dropped = by_id(judge(record(), target_matches=True, latest_screen=later))["G0"]
    assert not dropped.passed and "its latest complete screen, 99887766, did not keep it" in dropped.reason
    assert "it had survived abcdef12, the latest when it was validated" in dropped.reason
    kept = by_id(judge(record(), target_matches=True, latest_screen={"survived": True, "screen": "1234abcd"}))["G0"]
    assert kept.passed and kept.reason == "it survived screen 1234abcd, its latest"
    assert by_id(judge(record(), target_matches=True))["G0"].passed  # not looked at: the record's own


def test_no_held_out_task_no_screen_a_changed_file_or_an_approval_each_refuse():
    assert not by_id(judge(record(held_out=[]), target_matches=True))["G1"].passed
    assert not by_id(judge(record(screen={"survived": False}), target_matches=True))["G0"].passed
    assert not by_id(judge(record(), target_matches=False))["G6"].passed
    assert not by_id(judge(record()))["G6"].passed  # not looked at is not a match
    assert not by_id(judge(record(), target_matches=True, approved=True))["G7"].passed


def test_an_edit_that_writes_in_the_training_answers_is_refused():
    """A real run's cheat: its held-out task failed with and without it, so
    G5 found it no worse — and only a held-out task the original fails, made to improve, tells."""
    gate = judge(record(held_out=[{"eval": "2", "original": runs(0, 5), "edit": runs(0, 5)}]), target_matches=True)
    assert [check.id for check in gate.failed()] == ["G8"] and by_id(gate)["G5"].passed
    assert "writes in its answers" in by_id(gate)["G8"].reason and "0/5" in by_id(gate)["G8"].reason
    assert not by_id(judge(record(held_out=[{"eval": "2", "original": runs(1, 4), "edit": runs(1, 4)}]),
                           target_matches=True))["G8"].passed
    assert by_id(judge(record(held_out=[{"eval": "2", "original": runs(1, 4), "edit": runs(2, 3)}]),
                       target_matches=True))["G8"].passed  # better is enough: the person judges how much


def test_a_held_out_task_the_original_passes_is_no_test_of_a_fix():
    gate = judge(record(held_out=[{"eval": "2", "original": runs(5), "edit": runs(3, 2)}]), target_matches=True)
    assert gate.passed and "nothing to lift" in by_id(gate)["G8"].reason
    assert any("held-out task 2: lower with the edit (3/5 against 5/5)" in note for note in gate.notes)
    same = judge(record(held_out=[{"eval": "2", "original": runs(5), "edit": runs(5)}]), target_matches=True)
    assert same.passed and not any("held-out" in note for note in same.notes)


def test_what_no_check_decides_is_said():
    gate = judge(record(forward={"original": runs(4, 1), "edit": runs(5)}), target_matches=True)
    assert gate.passed  # every check holds ...
    assert any("may not decide the task" in note for note in gate.notes)


def test_every_held_out_task_must_improve_where_the_original_fails_it():
    gate = judge(record(held_out=[{"eval": "2", "original": runs(0, 5), "edit": runs(5)},
                                  {"eval": "3", "original": runs(0, 5), "edit": runs(0, 5)}]), target_matches=True)
    lifts = [check for check in gate.checks if check.id == "G8"]
    assert [check.passed for check in lifts] == [True, False] and "held-out task 3" in lifts[1].name
    assert not by_id(judge(record(held_out=[]), target_matches=True))["G8"].passed


def test_a_validation_stopped_before_its_held_out_tasks_says_so_not_that_there_are_none():
    stopped = by_id(judge(record(held_out=[], held_out_planned=["2"], complete=False), target_matches=True))["G1"]
    assert not stopped.passed and "stopped before its held-out task(s) ran (2)" in stopped.reason
    none = by_id(judge(record(held_out=[]), target_matches=True))["G1"]
    assert "holds no held-out eval" in none.reason
