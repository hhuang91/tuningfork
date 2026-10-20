"""Fisher's exact test, JSON normalization, distributions, and A/B comparison.

The Fisher tests check the float implementation against exact rational arithmetic — an
independent path to the same number — and against a worked example.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb

import pytest

from tuningfork.analysis.distribution import (
    agreeing_letters,
    cluster_letter,
    compare_arms,
    distribution,
    target_by_key,
    target_by_pattern,
)
from tuningfork.analysis.normalize import (
    Outcome,
    json_canonical,
    outcome_for,
    step_outcome,
    validate_normalizers,
)
from tuningfork.analysis.stats import fisher_exact
from tests.helpers import as_response, text_message, tool_call_message


def exact_fisher(a: int, b: int, c: int, d: int) -> float:
    """Reference implementation in rational arithmetic."""
    row1, row2, col1 = a + b, c + d, a + c
    denominator = comb(row1 + row2, col1)

    def probability(x: int) -> Fraction:
        return Fraction(comb(row1, x) * comb(row2, col1 - x), denominator)

    observed = probability(a)
    return float(
        sum(
            (
                probability(x)
                for x in range(max(0, col1 - row2), min(row1, col1) + 1)
                if probability(x) <= observed
            ),
            Fraction(0),
        )
    )


def outcome(label: str) -> Outcome:
    return Outcome("tool_call", f"key:{label}", label)


ERROR = Outcome("error", "error:x", "transport failure")


class TestFisher:
    def test_a_worked_example(self):
        """18/30 against 27/30: p = 0.015."""
        assert fisher_exact(18, 12, 27, 3) == pytest.approx(0.015332, abs=1e-6)

    @pytest.mark.parametrize(
        "table",
        [(18, 12, 27, 3), (3, 17, 12, 8), (0, 20, 19, 1), (5, 5, 5, 5), (1, 0, 0, 1), (7, 13, 2, 18)],
    )
    def test_agrees_with_exact_rational_arithmetic(self, table):
        assert fisher_exact(*table) == pytest.approx(exact_fisher(*table), rel=1e-9)

    def test_identical_arms_are_not_significant(self):
        assert fisher_exact(10, 10, 10, 10) == pytest.approx(1.0)

    def test_complete_separation_is_highly_significant(self):
        assert fisher_exact(0, 20, 20, 0) < 1e-9

    def test_empty_table(self):
        assert fisher_exact(0, 0, 0, 0) == 1.0

    def test_negative_counts_are_rejected(self):
        with pytest.raises(ValueError):
            fisher_exact(-1, 2, 3, 4)


class TestJsonNormalization:
    def test_formatting_and_number_spelling_do_not_split_a_behaviour(self):
        compact = tool_call_message(
            "write_file", {"path": "report.json", "content": '{"count":4,"total":61.5}'}
        )
        pretty = tool_call_message(
            "write_file",
            {"path": "report.json", "content": '{\n  "total": 61.50,\n  "count": 4\n}'},
        )
        assert step_outcome(as_response(compact)).key == step_outcome(as_response(pretty)).key

    def test_different_keys_are_different_behaviours(self):
        """The whole point of the tally-report fixture."""
        wrong = tool_call_message(
            "write_file", {"path": "report.json", "content": '{"count": 4, "total": 61.5}'}
        )
        right = tool_call_message(
            "write_file",
            {"path": "report.json", "content": '{"row_count": 4, "amount_total": 61.5}'},
        )
        assert step_outcome(as_response(wrong)).key != step_outcome(as_response(right)).key

    def test_label_shows_the_canonical_document(self):
        message = tool_call_message(
            "write_file", {"path": "report.json", "content": '{ "b": 1, "a": 2 }'}
        )
        assert '{"a":2,"b":1}' in step_outcome(as_response(message)).label

    def test_unparseable_json_is_left_alone(self):
        assert json_canonical("{not json") is None
        message = tool_call_message("write_file", {"path": "r.json", "content": "{not json"})
        assert "{not json" in step_outcome(as_response(message)).key

    def test_only_json_files_are_parsed(self):
        message = tool_call_message(
            "write_file", {"path": "notes.txt", "content": '{ "a": 1 }'}
        )
        assert '{ \\"a\\": 1 }' in step_outcome(as_response(message)).key

    def test_can_be_switched_off_by_naming_other_normalizers(self):
        pretty = tool_call_message(
            "write_file", {"path": "r.json", "content": '{\n "a": 1\n}'}
        )
        compact = tool_call_message("write_file", {"path": "r.json", "content": '{"a":1}'})
        assert (
            outcome_for(as_response(pretty), ["tool_call_json"]).key
            != outcome_for(as_response(compact), ["tool_call_json"]).key
        )

    def test_unknown_normalizer_names_are_rejected(self):
        with pytest.raises(ValueError, match="known"):
            validate_normalizers(["python_ast", "yaml_magic"])


class TestToolNameNormalizer:
    def test_groups_calls_by_tool_regardless_of_arguments(self):
        """Three different programs are one behaviour when the question is *which tool*."""
        first = tool_call_message("execute_code", {"code": "import csv\nprint(1)"})
        second = tool_call_message("execute_code", {"code": "open('data.csv').read()"})

        assert outcome_for(as_response(first)).key != outcome_for(as_response(second)).key
        coarse = [outcome_for(as_response(m), ["tool_name"]) for m in (first, second)]
        assert coarse[0].key == coarse[1].key
        assert coarse[0].label == "execute_code(…)"

    def test_is_never_on_by_default(self):
        message = tool_call_message("read_file", {"path": "a.csv"})
        assert "a.csv" in outcome_for(as_response(message)).key

    def test_different_tools_stay_apart(self):
        read = outcome_for(as_response(tool_call_message("read_file", {})), ["tool_name"])
        run = outcome_for(as_response(tool_call_message("execute_code", {})), ["tool_name"])
        assert read.key != run.key


class TestTruncation:
    def test_a_length_cutoff_is_its_own_outcome(self):
        """A fragment is not a behaviour; clustering fragments invents distinctions."""
        response = as_response(text_message("partial answ"), finish_reason="length")
        result = step_outcome(response)
        assert result.kind == "truncated"
        assert "max_tokens" in result.label


class TestDistribution:
    def test_clusters_are_lettered_largest_first(self):
        result = distribution([outcome("b"), outcome("a"), outcome("b"), outcome("b")])

        assert [c.letter for c in result.clusters] == ["A", "B"]
        assert result.clusters[0].outcome.label == "b"
        assert result.clusters[0].count == 3
        assert result.total == 4

    def test_errors_are_counted_but_never_rated(self):
        result = distribution([outcome("a"), ERROR, outcome("a")])

        assert result.errors == 1
        assert result.total == 2
        assert result.clusters[0].rate.total == 2

    def test_every_cluster_carries_an_interval(self):
        (only,) = distribution([outcome("a")] * 20).clusters
        assert only.rate.point == 1.0
        assert only.rate.low < 0.85  # 20/20 is not proof of certainty

    def test_lookup_by_letter_and_key(self):
        result = distribution([outcome("x"), outcome("y"), outcome("x")])
        assert result.by_letter("a").outcome.label == "x"
        assert result.by_key("key:y").letter == "B"

    def test_letters_beyond_z(self):
        assert cluster_letter(25) == "Z"
        assert cluster_letter(26) == "AA"


class TestAgreeingLetters:
    def test_a_resample_s_behaviours_keep_their_letters_and_new_ones_take_unused_ones(self):
        known = {"key:x": "A", "key:y": "B", "key:z": "C"}
        assert agreeing_letters(["key:y", "key:new", "key:x", "key:other"], known) == ["B", "D", "A", "E"]
        assert agreeing_letters(["key:a", "key:b"]) == ["A", "B"]
        assert [c.letter for c in distribution([outcome("y"), outcome("y"), outcome("w")], letters=known).clusters] == ["B", "D"]

    def test_an_a_b_letters_its_behaviours_as_the_resample_its_target_came_from(self):
        """Found in use: an A/B whose target was a resample's B lettered its own table afresh,
        the target its row A, and only the target badge said which row it was."""
        resample = {"key:count/total": "A", "key:row_count": "B"}
        original = [outcome("row_count")] * 2 + [outcome("count/total")] * 3
        variant = [outcome("row_count")] * 20
        target = target_by_key("key:row_count", "B from resample 12345678: row_count")

        afresh = compare_arms(original, variant, target=target)
        agreeing = compare_arms(original, variant, target=target, letters=resample)

        assert {shift.outcome.label: shift.letter for shift in afresh.shifts} == {"row_count": "A", "count/total": "B"}
        assert {shift.outcome.label: shift.letter for shift in agreeing.shifts} == {"row_count": "B", "count/total": "A"}
        # Each arm's own behaviours take the table's letters, whichever arm did them more.
        assert agreeing.original.by_key("key:count/total").letter == "A"
        assert agreeing.variant.by_key("key:row_count").letter == "B"
        assert afresh.original.by_key("key:row_count").letter == "A"


class TestABComparison:
    def test_pre_declared_target_gets_one_exact_test(self):
        original = [outcome("count/total")] * 18 + [outcome("row_count")] * 2
        variant = [outcome("row_count")] * 19 + [outcome("count/total")] * 1

        result = compare_arms(original, variant, target=target_by_pattern("row_count"))

        assert result.target.original.successes == 2
        assert result.target.variant.successes == 19
        assert result.target.significant
        assert result.target.direction == "up"
        assert result.target.p_value == pytest.approx(exact_fisher(2, 18, 19, 1), rel=1e-9)

    def test_each_behaviour_says_how_many_of_its_samples_the_target_counted(self):
        """The rows the target's own rates add up, as the A/B's page marks them: all of the
        behaviour the target names, none of the others."""
        original = [outcome("count/total")] * 18 + [outcome("row_count")] * 2
        variant = [outcome("row_count")] * 19 + [outcome("count/total")] * 1

        result = compare_arms(original, variant, target=target_by_pattern("row_count"))

        counted = {shift.outcome.label: shift.target_counted for shift in result.shifts}
        assert counted == {"row_count": 21, "count/total": 0}
        assert {row["outcome"]["label"]: row["target_counted"] for row in result.to_json()["shifts"]} == counted

    def test_a_pattern_can_count_part_of_one_behaviour(self):
        """Two scripts that differ in a comment are one behaviour, and a pattern reading what
        each says can meet one and not the other: the row says how many it counted."""
        plain = Outcome("tool_call", "key:script", "script", text="print(1)")
        noted = Outcome("tool_call", "key:script", "script", text="print(1)  # row_count")

        result = compare_arms([plain] * 3, [noted] * 2 + [plain], target=target_by_pattern("row_count"))

        (shift,) = result.shifts
        assert shift.target_counted == 2 and result.target.variant.successes == 2

    def test_without_a_target_no_behaviour_says_what_it_counted(self):
        result = compare_arms([outcome("a")] * 3, [outcome("b")] * 3)
        assert all(shift.target_counted is None for shift in result.shifts)
        assert all("target_counted" not in row for row in result.to_json()["shifts"])

    def test_no_real_shift_is_not_significant(self):
        original = [outcome("read")] * 14 + [outcome("exec")] * 6
        variant = [outcome("read")] * 13 + [outcome("exec")] * 7

        result = compare_arms(original, variant, target=target_by_pattern("^read$"))

        assert not result.target.significant
        # "Not significant" must come with how much data a real answer would take.
        assert result.target.samples_to_detect() > 100

    def test_without_a_target_the_comparison_is_exploratory_and_corrected(self):
        original = [outcome("a")] * 10 + [outcome("b")] * 10
        variant = [outcome("a")] * 10 + [outcome("b")] * 10

        result = compare_arms(original, variant)

        assert result.exploratory
        assert result.target is None
        assert all(shift.adjusted_p >= shift.p_value for shift in result.shifts)
        assert result.significant_shifts() == []

    def test_bonferroni_scales_with_the_number_of_clusters(self):
        original = [outcome("a")] * 6 + [outcome("b")] * 7 + [outcome("c")] * 7
        variant = [outcome("a")] * 20
        shifts = {s.outcome.label: s for s in compare_arms(original, variant).shifts}
        assert shifts["a"].adjusted_p == pytest.approx(min(1.0, shifts["a"].p_value * 3))

    def test_pooled_letters_are_shared_by_both_arms(self):
        original = [outcome("only-original")] * 3
        variant = [outcome("only-variant")] * 5

        shifts = compare_arms(original, variant).shifts

        assert [s.letter for s in shifts] == ["A", "B"]
        assert shifts[0].outcome.label == "only-variant"  # larger pooled count
        assert shifts[0].original.successes == 0
        assert shifts[0].variant.successes == 5

    def test_errors_are_excluded_from_both_arms(self):
        original = [outcome("a")] * 5 + [ERROR] * 3
        variant = [outcome("a")] * 5
        result = compare_arms(original, variant, target=target_by_key("key:a", "a"))
        assert result.target.original.total == 5
        assert result.original.errors == 3

    def test_pattern_target_searches_the_full_key(self):
        """Labels truncate long arguments; the key does not."""
        long = Outcome("tool_call", 'key:{"padding":"' + "x" * 200 + '","deep":1}', "short label")
        assert target_by_pattern("deep").matches(long)


def test_letters_never_run_out():
    """Found in review: 703 clusters (a long free-text resample) crashed the report."""
    assert [cluster_letter(i) for i in (0, 25, 26, 51, 701, 702, 18277)] == [
        "A", "Z", "AA", "AZ", "ZZ", "AAA", "ZZZ",
    ]
