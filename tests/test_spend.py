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
