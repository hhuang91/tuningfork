"""The reads every front end shares (`ops.inspect`): the HTTP API serves them as they are,
and the MCP tools read the same functions."""

from __future__ import annotations

import pytest

from tuningfork.ops import inspect
from tuningfork.ops.common import OperationError
from tests.helpers import record_pilot, record_run, tool_call_message


def test_a_run_reads_as_its_steps_with_what_the_model_did(store):
    run_id = record_pilot(store)
    detail = inspect.run_detail(store, "pilot")
    assert detail["run"]["id"] == run_id
    assert [step["idx"] for step in detail["steps"]] == [0, 1]
    assert [step["outcome"]["kind"] for step in detail["steps"]] == ["tool_call", "tool_call"]
    assert detail["threads"] == [] and detail["launch"] is None


def test_an_unknown_run_is_refused_by_name(store):
    with pytest.raises(OperationError, match="no run matching 'nope'"):
        inspect.run_detail(store, "nope")


def test_tasks_group_runs_with_their_conditions(store):
    record_pilot(store)
    (task,) = inspect.tasks(store)
    assert task["task_id"] == "tally-report/1" and task["runs"] == 1
    (condition,) = task["conditions"]
    assert condition["runs"] == 1 and condition["model"] == "mock-model"
    # No check has judged the run: there is no rate to give, and none is made up.
    assert task["passed"] is None and task["uncounted"] == 0


def test_experiments_list_newest_first_and_by_run(store):
    record_pilot(store)
    assert inspect.experiment_list(store) == []
    assert inspect.experiment_list(store, "pilot") == []
    with pytest.raises(OperationError):
        inspect.experiment_list(store, "nope")


class TestDivergence:
    def test_two_runs_that_split_say_where(self, store):
        read = tool_call_message("read_file", {"path": "a"})
        record_run(store, "a", [read, tool_call_message("write_file", {"path": "r"})], task_id="t/1")
        record_run(store, "b", [read, tool_call_message("run_code", {"code": "1"})], task_id="t/1")
        report = inspect.divergence_report(store, task="t/1")
        assert report["divergence_index"] == 1
        assert inspect.divergence_report(store, refs=["a", "b"])["divergence_index"] == 1

    def test_each_run_says_its_verdict_as_the_pass_rates_count_it(self, store):
        """The page colours each run by its own verdict: a run the rates leave out says so,
        whatever its check said."""
        read = tool_call_message("read_file", {"path": "a"})
        runs = {name: record_run(store, name, [read, tool_call_message("write_file", {"path": name})], task_id="t/1")
                for name in ("a", "b", "c", "d")}
        store.annotate_run(runs["a"], result={"passed": True, "usable": True})
        store.annotate_run(runs["b"], result={"passed": False, "usable": True})
        store.annotate_run(runs["c"], result={"passed": True, "usable": False, "timed_out": True})

        report = inspect.divergence_report(store, task="t/1")

        assert report["verdicts"] == {runs["a"]: "passed", runs["b"]: "failed", runs["c"]: "not counted", runs["d"]: None}

    def test_it_needs_something_to_compare(self, store):
        with pytest.raises(OperationError) as refused:
            inspect.divergence_report(store)
        assert refused.value.field == "task"

    def test_one_run_is_not_a_comparison_and_says_what_to_do(self, store):
        record_pilot(store)
        with pytest.raises(OperationError) as refused:
            inspect.divergence_report(store, task="tally-report/1")
        assert "at least two runs" in refused.value.message and "batch" in refused.value.hint

    def test_an_unknown_lens_names_the_field(self, store):
        with pytest.raises(OperationError) as refused:
            inspect.divergence_report(store, task="t/1", normalizers=["nope"])
        assert refused.value.field == "normalizers"


def test_a_sample_reads_as_the_model_gave_it(store):
    """What a reflection packet quotes: the answer, its tool calls, and its reasoning."""
    run_id = record_pilot(store)
    step = store.steps(run_id)[1]
    experiment = store.create_experiment(
        kind="resample", arm="original", run_id=run_id, step_idx=1, step_id=step.id,
        request={"messages": []}, settings={"n": 1},
    )
    message = {**tool_call_message("write_file", {"path": "report.json"}),
               "reasoning_content": "The skill says to write the totals."}
    store.add_sample(experiment.id, 0, seed=1000, response={"choices": [{"message": message, "finish_reason": "tool_calls"}]},
                     outcome_kind="tool_call", outcome_key="k", outcome_label="write_file(path='report.json')")
    view = inspect.sample_view(store, experiment.id[:8], 0)
    assert view["tool_calls"][0]["name"] == "write_file"
    assert view["reasoning"] == "The skill says to write the totals."
    assert view["outcome"]["label"] == "write_file(path='report.json')" and view["seed"] == 1000
    with pytest.raises(OperationError, match="has no sample 5"):
        inspect.sample_view(store, experiment.id, 5)
