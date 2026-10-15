"""Normalization, statistics, and divergence localization.

The clustering tests are the ones that matter: if incidental differences survive
normalization, every sample forms its own cluster and the divergence report degenerates
into "all runs differ", which is true and useless.
"""

from __future__ import annotations

import pytest

from tuningfork.analysis.divergence import compare_runs
from tuningfork.analysis.normalize import (
    outcome_for,
    python_ast_digest,
    step_outcome,
)
from tuningfork.analysis.stats import rate, samples_needed, wilson_interval
from tests.helpers import as_response, record_run, text_message, tool_call_message


class TestWilson:
    def test_matches_the_worked_example(self):
        """3/5 is consistent with anywhere from ~23% to ~88%."""
        low, high = wilson_interval(3, 5)
        assert round(low, 2) == 0.23
        assert round(high, 2) == 0.88

    def test_extremes_do_not_claim_certainty(self):
        low, high = wilson_interval(5, 5)
        assert high == 1.0
        assert low < 0.6  # 5/5 is not proof of a perfect skill

        low, high = wilson_interval(0, 5)
        assert low == 0.0
        assert high > 0.4

    def test_interval_narrows_with_more_samples(self):
        small = wilson_interval(6, 10)
        large = wilson_interval(60, 100)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_zero_samples_is_total_ignorance(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_format_always_shows_the_count(self):
        assert rate(3, 5).format().startswith("3/5")

    def test_sample_size_helper_is_in_the_right_ballpark(self):
        """Separating 60% from 80% takes tens of runs per arm, not five."""
        needed = samples_needed(0.6, 0.8)
        assert 60 <= needed <= 110

    def test_smaller_differences_need_more_samples(self):
        assert samples_needed(0.6, 0.65) > samples_needed(0.6, 0.9)


class TestPythonNormalization:
    def test_comments_and_formatting_are_ignored(self):
        first = "def f(x):\n    return x + 1\n"
        second = "# adds one\ndef f(x):\n\n    return x+1  # here\n"
        assert python_ast_digest(first) == python_ast_digest(second)

    def test_docstrings_are_ignored(self):
        bare = "def f():\n    return 1\n"
        documented = 'def f():\n    """Explain."""\n    return 1\n'
        assert python_ast_digest(bare) == python_ast_digest(documented)

    def test_real_differences_are_kept(self):
        assert python_ast_digest("x = 1") != python_ast_digest("x = 2")

    def test_unparseable_source_returns_none(self):
        assert python_ast_digest("def (:") is None


class TestOutcomes:
    def test_tool_call_ids_do_not_split_identical_behaviour(self):
        first = step_outcome(as_response(tool_call_message("w", {"p": "a"}, call_id="1")))
        second = step_outcome(as_response(tool_call_message("w", {"p": "a"}, call_id="2")))
        assert first.key == second.key

    def test_different_arguments_are_different_outcomes(self):
        first = step_outcome(as_response(tool_call_message("w", {"p": "a"})))
        second = step_outcome(as_response(tool_call_message("w", {"p": "b"})))
        assert first.key != second.key

    def test_python_payloads_cluster_structurally(self):
        """Two scripts differing only in comments are the same behaviour."""
        a = tool_call_message("write_file", {"path": "g.py", "content": "x = 1\n"})
        b = tool_call_message(
            "write_file", {"path": "g.py", "content": "# note\nx  =  1\n"}
        )
        assert step_outcome(as_response(a)).key == step_outcome(as_response(b)).key

    def test_python_normalization_can_be_switched_off(self):
        a = tool_call_message("write_file", {"path": "g.py", "content": "x = 1\n"})
        b = tool_call_message("write_file", {"path": "g.py", "content": "# n\nx = 1\n"})
        first = step_outcome(as_response(a), python_ast=False)
        second = step_outcome(as_response(b), python_ast=False)
        assert first.key != second.key

    def test_text_whitespace_is_collapsed(self):
        first = step_outcome(as_response(text_message("hello   world")))
        second = step_outcome(as_response(text_message("hello world\n")))
        assert first.key == second.key

    def test_wording_differences_are_preserved(self):
        first = step_outcome(as_response(text_message("hello world")))
        second = step_outcome(as_response(text_message("hi world")))
        assert first.key != second.key

    def test_missing_response_is_an_error_outcome(self):
        assert step_outcome(None).kind == "error"

    def test_label_names_the_tool_and_arguments(self):
        outcome = step_outcome(as_response(tool_call_message("write_file", {"path": "g.py"})))
        assert "write_file" in outcome.label
        assert "g.py" in outcome.label

    def test_normalizer_selection_by_name(self):
        message = tool_call_message("write_file", {"path": "g.py", "content": "# c\nx=1"})
        plain = outcome_for(as_response(message), ["tool_call_json"])
        structural = outcome_for(as_response(message), ["python_ast"])
        assert "<ast:" in structural.key
        assert "<ast:" not in plain.key


class TestDivergence:
    def test_identical_runs_do_not_diverge(self, store):
        messages = [text_message("step one"), text_message("step two")]
        ids = [record_run(store, f"r{i}", messages) for i in range(3)]

        report = compare_runs(store, ids)

        assert not report.diverged
        assert report.divergence_index is None
        assert report.compared == 3

    def test_first_divergent_step_is_located(self, store):
        shared = [text_message("read the file"), text_message("parse it")]
        good = [*shared, text_message("write report.json")]
        bad = [*shared, text_message("ask the user what to do")]
        ids = [record_run(store, f"g{i}", good) for i in range(3)]
        ids += [record_run(store, f"b{i}", bad) for i in range(2)]

        report = compare_runs(store, ids)

        assert report.divergence_index == 2
        assert report.agreed_steps == 2

    def test_competing_behaviours_are_grouped_and_rated(self, store):
        shared = [text_message("same start")]
        ids = [record_run(store, f"g{i}", [*shared, text_message("A")]) for i in range(3)]
        ids += [record_run(store, f"b{i}", [*shared, text_message("B")]) for i in range(2)]

        comparison = compare_runs(store, ids).at_divergence()

        assert [group.count for group in comparison.groups] == [3, 2]  # largest first
        majority = comparison.majority
        assert majority.rate.total == 5
        assert round(majority.rate.point, 2) == 0.6
        # Intervals are always attached, never a bare percentage.
        assert 0.0 < majority.rate.low < majority.rate.point < majority.rate.high < 1.0

    def test_a_run_that_stops_early_counts_as_a_difference(self, store):
        long_run = [text_message("one"), text_message("two")]
        short_run = [text_message("one")]
        ids = [record_run(store, "long", long_run), record_run(store, "short", short_run)]

        report = compare_runs(store, ids)

        assert report.divergence_index == 1
        labels = {group.outcome.label for group in report.at_divergence().groups}
        assert "run ended" in labels

    def test_volatile_ids_alone_do_not_look_like_divergence(self, store):
        """Without masking, every run would diverge at its first tool call."""
        ids = [
            record_run(store, f"r{i}", [tool_call_message("w", {"p": "a"}, call_id=str(i))])
            for i in range(4)
        ]

        assert not compare_runs(store, ids).diverged

    def test_runs_recorded_under_different_skill_files_are_flagged(self, store):
        """Would have caught the contaminated sample Phase 1 published."""
        ids = [record_run(store, f"r{i}", [text_message("x")]) for i in range(3)]
        store.annotate_run(ids[0], skill_hashes={"SKILL.md": "aaa"})
        store.annotate_run(ids[1], skill_hashes={"SKILL.md": "aaa"})
        store.annotate_run(ids[2], skill_hashes={"SKILL.md": "bbb"})

        report = compare_runs(store, ids)

        assert report.mixed_skills
        assert sorted(len(runs) for runs in report.skill_versions.values()) == [1, 2]

    def test_consistent_or_unknown_skills_are_not_flagged(self, store):
        ids = [record_run(store, f"r{i}", [text_message("x")]) for i in range(3)]
        store.annotate_run(ids[0], skill_hashes={"SKILL.md": "aaa"})
        store.annotate_run(ids[1], skill_hashes={"SKILL.md": "aaa"})
        # ids[2] has no recorded hashes: unknown, which is not the same as different

        assert not compare_runs(store, ids).mixed_skills

    def test_comparing_fewer_than_two_runs_is_rejected(self, store):
        run_id = record_run(store, "solo", [text_message("x")])
        with pytest.raises(ValueError, match="at least two"):
            compare_runs(store, [run_id])


class TestRunPaths:
    """A batch run's workspace and home differ per run; what the model did there must not."""

    RUN = "/home/u/.tuningfork/batches/0a2f84ad/3"

    @staticmethod
    def call(name: str, **arguments) -> dict:
        return as_response(tool_call_message(name, arguments), "tool_calls")

    def key(self, name: str, **arguments) -> str:
        return outcome_for(self.call(name, **arguments)).key

    def test_a_shell_command_naming_the_workspace_matches_across_runs(self):
        other = self.RUN.replace("0a2f84ad/3", "4bb1283e/0")
        assert self.key("Bash", command=f"cat {self.RUN}/workspace/data.csv") == self.key(
            "Bash", command=f"cat {other}/workspace/data.csv"
        )

    def test_the_home_is_masked_too(self):
        """Claude Code's skills live under the run's home: reading a bundled file names it."""
        label = outcome_for(self.call("Read", file_path=f"{self.RUN}/home/config/skills/x/ref.md")).label
        assert "<home>/config/skills/x/ref.md" in label

    @pytest.mark.parametrize("after", [".", ")", ";", ":", ",", "/"])
    def test_the_root_followed_by_punctuation_is_masked(self, after):
        text = f"(cd {self.RUN}/workspace{after} ls"
        assert f"<workspace>{after}" in outcome_for(self.call("Bash", command=text)).label

    def test_nested_arguments_are_masked(self):
        label = outcome_for(self.call("Edit", edits=[{"path": f"{self.RUN}/workspace/a.txt"}])).label
        assert "<workspace>/a.txt" in label

    def test_code_naming_the_workspace_has_one_structure(self):
        first = self.key("execute_code", code=f"open('{self.RUN}/workspace/data.csv').read()")
        other = self.RUN.replace("0a2f84ad/3", "4bb1283e/0")
        second = self.key("execute_code", code=f"open('{other}/workspace/data.csv').read()")
        assert first == second

    def test_a_similar_name_is_left_alone(self):
        label = outcome_for(self.call("Bash", command=f"ls {self.RUN}/workspaces")).label
        assert "<workspace>" not in label

    def test_the_sandboxs_folders_are_masked_as_a_launchs_own(self):
        """In the sandbox every launch works in /home/user/project: a run there and one in its
        own folders did the same thing when they wrote the same file."""
        assert self.key("Write", file_path="/home/user/project/report.json", content="x") == self.key(
            "Write", file_path=f"{self.RUN}/workspace/report.json", content="x"
        ) == self.key("Write", file_path="report.json", content="x")
        label = outcome_for(self.call("terminal", command="python /home/user/.hermes/skills/x/run.py")).label
        assert "<home>/.hermes/skills/x/run.py" in label
        assert "<home>" not in outcome_for(self.call("Bash", command="ls /home/username /home/user-files")).label


class TestFailedCalls:
    """A stream that broke off keeps the partial answer it assembled. Clustered as a
    behaviour, it would be one the model never finished."""

    def test_a_transport_failure_is_an_error(self):
        from tuningfork.analysis.normalize import recorded_outcome
        from tuningfork.proxy.sse import CUT_OFF, upstream_error, unreadable

        partial = as_response(text_message("I will now wr"))
        for error in (CUT_OFF, upstream_error({"message": "overloaded"}), unreadable(ValueError("x"))):
            assert recorded_outcome(partial, error).is_error
        assert recorded_outcome(partial, None).kind == "text"

    def test_a_dropped_call_still_shows_what_the_model_did(self):
        from tuningfork.analysis.normalize import recorded_outcome
        from tuningfork.proxy.held import DROPPED, HUNG_UP

        answer = as_response(text_message("done"))
        assert recorded_outcome(answer, DROPPED).kind == "text"
        assert recorded_outcome(answer, HUNG_UP).kind == "text"

    def test_divergence_reads_a_failed_step_as_an_error(self, store):
        from tuningfork.proxy.sse import CUT_OFF

        whole = record_run(store, "whole", [text_message("I will now write it")], task_id="t/9")
        store.begin_run("cut", "t/9")
        store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "do the task"}]},
                          response=as_response(text_message("I will now wr")), error=CUT_OFF)
        cut = store.get_run("cut").id

        report = compare_runs(store, [whole, cut])

        kinds = {group.outcome.kind for group in report.steps[0].groups}
        assert kinds == {"text", "error"}


def test_a_recorded_script_with_an_invalid_escape_is_digested_without_a_warning():
    """The source is the model's: its `"\\s"` is a SyntaxWarning in Python 3.12, and whoever reads
    the recording must not print it (the benchmark's driver did, three times in one stage)."""
    import warnings

    from tuningfork.analysis.normalize import python_ast_digest

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        digest = python_ast_digest('import re\nprint(re.sub("\\s+", " ", "a  b"))\n')
    assert digest

