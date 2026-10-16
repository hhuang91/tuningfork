"""What work cost, counted from the records: calls, tokens, and time."""

from __future__ import annotations

import pytest

from tuningfork.ops import OperationError
from tuningfork.ops.spend import Spend, count_call, run_spend, spend_report
from tuningfork.store.db import Store
from tests.helpers import as_response, text_message


def test_a_runs_calls_are_counted_from_their_records(tmp_path):
    store = Store(tmp_path / "store")
    try:
        store.begin_run("r1", "tally-report/1")
        first = as_response(text_message("one"))
        first["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
        first["timings"] = {"prompt_ms": 50.0, "predicted_ms": 25.0}  # llama.cpp's own
        store.record_step(
            request={"model": "m", "messages": [{"role": "user", "content": "a"}]},
            response=first, timings={"total_ms": 100.0},
        )
        second = as_response(text_message("two"))
        second["usage"] = {"input_tokens": 5, "output_tokens": 1}  # as the Messages API counts
        store.record_step(
            request={"model": "m", "messages": [
                {"role": "user", "content": "a"}, {"role": "assistant", "content": "one"},
                {"role": "user", "content": "b"},
            ]},
            response=second, timings={"total_ms": 40.0},
        )
        run = store.get_run("r1")

        found = run_spend(store, run.id)
        assert found.calls == 2 and found.failed == 0
        assert found.prompt_tokens == 105 and found.completion_tokens == 11 and found.tokens == 116
        assert found.wall_ms == 140.0
        # Only the first call's server said what it spent.
        assert found.server_ms == 75.0 and found.unmeasured == 1
        data = found.to_json()
        assert data["wall_s"] == 0.1 and data["server_s"] == 0.1 and data["unmeasured"] == 1

        rows = spend_report(store, runs=["r1"])
        assert rows[0].what == f"run {run.id[:8]} (r1)" and rows[0].spend == found
        assert rows[0].to_json()["calls"] == 2
        with pytest.raises(OperationError) as refused:
            spend_report(store, runs=["nope"])
        assert refused.value.code == "not_found" and refused.value.field == "run"
        with pytest.raises(OperationError) as refused:
            spend_report(store, tasks=["no/task"])
        assert refused.value.field == "task"
        assert spend_report(store, tasks=["tally-report/1"])[0].spend == found
    finally:
        store.close()


def test_a_failed_call_and_one_without_timings_still_count():
    failed = count_call({"prompt_tokens": 3}, {"total_ms": 7}, None, "boom")
    assert failed.failed == 1 and failed.calls == 1 and failed.prompt_tokens == 3 and failed.unmeasured == 1
    bare = count_call(None, None, {"timings": {"prompt_ms": 2}}, None)
    assert bare.wall_ms == 0 and bare.server_ms == 2 and bare.unmeasured == 0
    total = failed + bare
    assert total.calls == 2 and total.to_json()["server_s"] == 0.0
    assert Spend().to_json()["server_s"] is None


def test_a_fork_that_made_no_call_cost_no_call(tmp_path):
    """A sample run to the end is a fork, counted as its launch. One whose harness ended before its
    first model call has no run, and its time is the harness's: never a failed call. A sample at
    the step is a call of its own."""
    from tuningfork.ops.spend import experiment_spend, job_spend

    store = Store(tmp_path / "store")
    try:
        made = {}
        for name, settings in (("forks", {"n": 1, "reach": "end"}), ("step", {"n": 1})):
            experiment = store.create_experiment(
                kind="resample", arm="original", run_id=None, step_idx=None, step_id=None,
                request={"model": "m", "messages": []}, settings=settings,
            )
            store.add_sample(experiment.id, 0, seed=None, response=None, outcome_kind="error", outcome_key=None,
                             outcome_label=None, timings={"total_ms": 30_000.0}, error="the harness exited",
                             usage={"prompt_tokens": 7} if name == "step" else None)
            made[name] = experiment.id

        assert experiment_spend(store, made["forks"]) == Spend()
        step = experiment_spend(store, made["step"])
        assert (step.calls, step.failed, step.prompt_tokens, step.wall_ms) == (1, 1, 7, 30_000.0)
        assert job_spend(store, {"experiments": [made["forks"], made["step"]]}) == step
    finally:
        store.close()
