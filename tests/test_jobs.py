"""The job runner, with a job kind of the tests' own: lanes, the pause, cancelling,
the sweep after a restart, and shutting down with work in flight.

What a job does is an operation's business (see test_jobs_e2e.py for real ones); here a
job only counts, so every property of the runner can be pinned down exactly.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from tuningfork.jobs import KEEP, JobRunner
from tuningfork.ops.catalog import BACKGROUND, INTERACTIVE, Kind
from tuningfork.ops.common import Cancelled, OperationError, emit


@dataclass
class CountParams:
    steps: int = field(default=3, metadata={"minimum": 0})
    #: Seconds per step; the job waits on its cancel, so a cancel ends it at once.
    delay: float = 0.0
    fail: str | None = None
    refuse: bool = False


@dataclass
class OptionsParams:
    harness_options: dict[str, Any] = field(default_factory=dict)


class Counted:
    def __init__(self, params: CountParams) -> None:
        self.params = params

    def to_json(self) -> dict:
        return {"steps": self.params.steps}


def prepare(context, params: CountParams) -> Counted:
    if params.refuse:
        raise OperationError("this job cannot run", field="refuse")
    return Counted(params)


def execute(context, prepared: Counted, *, progress, cancel):
    params = prepared.params
    for index in range(params.steps):
        if cancel.wait(params.delay):
            raise Cancelled(cancel.reason, partial={"counted": index})
        emit(progress, "sample", index=index)
    if params.fail == "operation":
        raise OperationError("it went wrong", hint="a hint")
    if params.fail == "bug":
        raise RuntimeError("a bug")
    return {"counted": params.steps}


async def execute_async(context, prepared: Counted, *, progress, cancel):
    for index in range(prepared.params.steps):
        await asyncio.sleep(0)
        emit(progress, "sample", index=index)
    return {"counted": prepared.params.steps, "async": True}


def kinds(**extra: Kind) -> dict[str, Kind]:
    made = {
        "count": Kind("count", BACKGROUND, "Count.", CountParams, prepare, execute, total=lambda p: p.params.steps),
        "quick": Kind("quick", INTERACTIVE, "Count, quickly.", CountParams, prepare, execute),
        "async": Kind("async", BACKGROUND, "Count, asynchronously.", CountParams, prepare, execute_async),
    }
    made.update(extra)
    return made


class Proxy:
    """What the runner needs of the proxy: a pause switch, somewhere to schedule, holds."""

    def __init__(self) -> None:
        self.paused = False
        self.url = "http://127.0.0.1:1"
        self.dropped: list[set] = []

    def call_soon(self, callback, *args, **kwargs) -> None:
        callback(*args, **kwargs)

    def drop_holds_of(self, launch_ids) -> None:
        self.dropped.append(set(launch_ids))


class Bus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, kind: str, **payload) -> None:
        self.events.append((kind, payload))


@pytest.fixture
def runner(store):
    made = JobRunner(store, Bus(), Proxy(), catalog=kinds())
    yield made
    made.shutdown(10)


def settled(runner: JobRunner, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not end: {runner.get(job_id)['status']}")


def until(check, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.01)
    raise AssertionError("timed out")


class TestLifecycle:
    def test_a_job_runs_reports_as_it_goes_and_is_stored(self, runner, store):
        job = runner.submit("count", {"steps": 3})
        done = settled(runner, job["id"])

        assert done["status"] == "done" and done["result"] == {"counted": 3}
        assert done["plan"] == {"steps": 3}
        assert done["progress"] == {"done": 3, "total": 3, "waiting": None}
        stored = store.get_job(job["id"])
        assert (stored.status, stored.result, stored.params["steps"]) == ("done", {"counted": 3}, 3)
        assert stored.started_at and stored.ended_at >= stored.started_at
        statuses = [p["job"]["status"] for kind, p in runner.bus.events if kind == "job"]
        assert statuses[0] == "queued" and statuses[-1] == "done" and "running" in statuses
        seqs = [p["seq"] for kind, p in runner.bus.events if kind == "job_event"]
        assert seqs == [1, 2, 3]

    def test_an_async_operation_runs_in_the_jobs_own_thread(self, runner):
        done = settled(runner, runner.submit("async", {"steps": 2})["id"])
        assert done["result"] == {"counted": 2, "async": True}

    def test_a_page_catching_up_gets_only_what_it_missed(self, runner):
        job_id = runner.submit("count", {"steps": 5})["id"]
        settled(runner, job_id)
        caught_up = runner.get(job_id, after=3)
        assert [entry["seq"] for entry in caught_up["events"]] == [4, 5]
        assert caught_up["events"][0]["event"] == {"kind": "sample", "data": {"index": 3}}

    def test_only_the_last_events_are_kept_and_the_gap_is_said(self, runner):
        job_id = runner.submit("count", {"steps": KEEP + 10})["id"]
        settled(runner, job_id)
        caught_up = runner.get(job_id, after=0)
        assert caught_up["first_seq"] == 11 and len(caught_up["events"]) == KEEP

    def test_a_failure_is_the_operations_own_words(self, runner):
        failed = settled(runner, runner.submit("count", {"steps": 1, "fail": "operation"})["id"])
        assert failed["status"] == "failed"
        assert failed["error"]["message"] == "it went wrong" and failed["error"]["hint"] == "a hint"

    def test_a_bug_in_a_job_fails_it_and_nothing_else(self, runner):
        broken = settled(runner, runner.submit("count", {"steps": 0, "fail": "bug"})["id"])
        assert broken["status"] == "failed" and broken["error"]["code"] == "internal"
        assert settled(runner, runner.submit("count", {"steps": 1})["id"])["status"] == "done"


class TestAsking:
    def test_parameters_are_refused_before_anything_is_queued(self, runner, store):
        with pytest.raises(OperationError) as refused:
            runner.submit("count", {"steps": "3"})
        assert refused.value.field == "steps"
        with pytest.raises(OperationError) as unknown:
            runner.submit("count", {"stepz": 3})
        assert unknown.value.field == "stepz"
        with pytest.raises(OperationError) as kind:
            runner.submit("nope", {})
        assert kind.value.field == "kind" and "count" in kind.value.hint
        assert store.jobs() == []

    def test_a_job_that_cannot_run_is_refused_now_not_when_its_turn_comes(self, runner, store):
        with pytest.raises(OperationError, match="cannot run") as refused:
            runner.submit("count", {"refuse": True})
        assert refused.value.field == "refuse"
        assert store.jobs() == []

    def test_a_dry_run_says_what_would_run_and_starts_nothing(self, runner, store):
        assert runner.plan("count", {"steps": 4}) == {
            "kind": "count", "lane": "background", "plan": {"steps": 4}, "total": 4,
        }
        assert store.jobs() == []

    def test_who_asked_is_kept_beside_the_process_that_runs_it(self, runner, store):
        asked = {"via": "mcp", "client": "claude-code", "version": "2.1.281"}
        job = runner.submit("count", {"steps": 1}, started_by=asked)
        assert job["started_by"] == asked
        settled(runner, job["id"])
        record = store.get_job(job["id"])
        # The sweep reads `owner`: still this runner's process, whoever asked.
        assert record.started_by == asked and record.owner == runner.owner
        assert runner.list()[0]["started_by"] == asked

    @pytest.mark.parametrize("asked", ["mcp", {"client": "x"}, {"via": 3}, {"via": "mcp", "client": "x" * 101}])
    def test_who_asked_is_short_text_or_refused(self, runner, store, asked):
        with pytest.raises(OperationError) as refused:
            runner.submit("count", {"steps": 1}, started_by=asked)
        assert refused.value.field == "started_by"
        assert store.jobs() == []

    def test_the_same_work_still_under_way_is_given_back_not_started_again(self, runner, store):
        first = runner.submit("count", {"steps": 100, "delay": 0.05}, dedupe=True)
        again = runner.submit("count", {"steps": 100, "delay": 0.05}, dedupe=True)
        assert again["id"] == first["id"] and again["deduplicated"] is True
        other = runner.submit("count", {"steps": 99, "delay": 0.05}, dedupe=True)
        assert other["id"] != first["id"] and "deduplicated" not in other
        # Without asking for it, the same work twice is two jobs, as from the web UI.
        assert runner.submit("count", {"steps": 100, "delay": 0.05})["id"] != first["id"]
        for job in runner.list():
            runner.cancel(job["id"])
        settled(runner, first["id"])
        # Once it has ended, asking again is new work.
        assert runner.submit("count", {"steps": 100, "delay": 0.05}, dedupe=True)["id"] != first["id"]
        for job in runner.list():
            runner.cancel(job["id"])

    def test_the_same_request_twice_is_one_job(self, runner, store):
        first = runner.submit("count", {"steps": 1}, request_id="form-7")
        again = runner.submit("count", {"steps": 1}, request_id="form-7")
        assert again["id"] == first["id"]
        settled(runner, first["id"])
        assert runner.submit("count", {"steps": 1}, request_id="form-7")["id"] == first["id"]
        assert len(store.jobs()) == 1

    def test_an_mcp_servers_secrets_are_kept_masked_and_given_to_the_job(self, store):
        """A header or an environment variable of an MCP server is a secret whatever its name, which
        no pattern would know: kept and shown masked — on the Jobs page, to whoever reads the jobs
        API — and the job runs with it as given."""
        from tuningfork.mcp.tools import same_params

        given = []

        def execute_options(context, prepared, *, progress, cancel):
            given.append(prepared.harness_options)
            cancel.wait(10)  # under way until cancelled
            raise Cancelled(cancel.reason)

        kind = Kind("options", BACKGROUND, "Options.", OptionsParams, lambda context, params: params, execute_options)
        runner = JobRunner(store, Bus(), Proxy(), catalog=kinds(options=kind))
        try:
            options = {"mcp_servers": {"x": {"command": "x", "env": {"CREDENTIAL": "plain-words-1"},
                                             "headers": {"X-Credential": "plain-words-2"}}}}
            job = runner.submit("options", {"harness_options": options}, dedupe=True)
            until(lambda: given)
            kept = store.get_job(job["id"]).params
            assert kept == runner.get(job["id"])["params"] == same_params(kind, {"harness_options": options})
            assert kept["harness_options"]["mcp_servers"]["x"] == {
                "command": "x", "env": {"CREDENTIAL": "…"}, "headers": {"X-Credential": "…"}}
            assert given == [options]
            # The same work asked for while it runs is found by the view kept of it.
            assert runner.submit("options", {"harness_options": options}, dedupe=True)["deduplicated"] is True
            runner.cancel(job["id"])
            settled(runner, job["id"])
        finally:
            runner.shutdown(10)

    def test_the_kinds_carry_their_parameters(self, runner):
        (count, *_) = runner.kinds()
        assert count["name"] == "count" and count["lane"] == "background"
        assert count["params"]["properties"]["steps"] == {"type": "integer", "default": 3, "minimum": 0}


class TestLanes:
    def test_background_jobs_wait_their_turn(self, runner):
        first = runner.submit("count", {"steps": 1, "delay": 5})
        second = runner.submit("count", {"steps": 1})
        until(lambda: runner.get(first["id"])["status"] == "running")
        assert runner.get(second["id"])["status"] == "queued"
        assert runner.get(second["id"])["queue_position"] == 0
        runner.cancel(first["id"])
        assert settled(runner, second["id"])["status"] == "done"

    def test_an_interactive_job_does_not_queue_behind_a_batch(self, runner):
        long = runner.submit("count", {"steps": 1, "delay": 5})
        until(lambda: runner.get(long["id"])["status"] == "running")
        quick = runner.submit("quick", {"steps": 1})
        assert settled(runner, quick["id"])["status"] == "done"
        assert runner.get(long["id"])["status"] == "running"
        runner.cancel(long["id"])

    def test_nothing_starts_while_paused(self, runner):
        runner.proxy.paused = True
        job = runner.submit("count", {"steps": 1})
        time.sleep(0.2)
        assert runner.get(job["id"])["status"] == "queued"
        runner.proxy.paused = False
        runner.kick()  # what `resume` does
        assert settled(runner, job["id"])["status"] == "done"


class TestCancelling:
    def test_a_queued_job_ends_at_once_and_never_runs(self, runner):
        blocker = runner.submit("count", {"steps": 1, "delay": 5})
        waiting = runner.submit("count", {"steps": 1})
        cancelled = runner.cancel(waiting["id"])
        assert cancelled["status"] == "cancelled"
        runner.cancel(blocker["id"])
        settled(runner, blocker["id"])
        assert runner.get(waiting["id"])["progress"] == {}

    def test_a_running_job_stops_and_keeps_what_it_finished(self, runner):
        job = runner.submit("count", {"steps": 100, "delay": 0.05})
        until(lambda: (runner.get(job["id"])["progress"] or {}).get("done", 0) >= 2)
        runner.cancel(job["id"])
        stopped = settled(runner, job["id"])
        assert stopped["status"] == "cancelled"
        assert 2 <= stopped["result"]["counted"] < 100

    def test_a_queued_job_cancelled_twice_at_once_ends_once(self, runner, monkeypatch):
        # The second cancel arrives after the first has taken the job out of its queue, before
        # it has ended it: found in review, it took the job for queued still, and failed
        # taking it out again.
        runner.proxy.paused = True
        job = runner.submit("count", {"steps": 1})
        finish = runner._finish
        second = []

        def finish_after_another_cancel(held, status, **kwargs):
            if not second:
                second.append(runner.cancel(held.id))
            finish(held, status, **kwargs)

        monkeypatch.setattr(runner, "_finish", finish_after_another_cancel)
        assert runner.cancel(job["id"])["status"] == "cancelled"
        assert second[0]["status"] == "cancelled"
        assert [kind for kind, payload in runner.bus.events if payload["job"]["status"] == "cancelled"] == ["job"]

    def test_cancelling_twice_or_after_the_end_changes_nothing(self, runner):
        job = runner.submit("count", {"steps": 1})
        settled(runner, job["id"])
        assert runner.cancel(job["id"])["status"] == "done"
        with pytest.raises(OperationError) as missing:
            runner.cancel("f" * 32)
        assert missing.value.code == "not_found"


class TestStopping:
    def test_shutting_down_interrupts_work_and_waits_for_it(self, store):
        runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
        running = runner.submit("count", {"steps": 100, "delay": 0.05})
        queued = runner.submit("count", {"steps": 1})
        until(lambda: runner.get(running["id"])["status"] == "running")
        assert runner.shutdown(10)
        assert store.get_job(running["id"]).status == "interrupted"
        assert store.get_job(queued["id"]).status == "interrupted"
        with pytest.raises(OperationError, match="stopping"):
            runner.submit("count", {"steps": 1})

    def test_work_an_earlier_proxy_left_running_is_marked_cut_off(self, store):
        dead = {"pid": 2**22 + 12345, "host": _host(), "start": 1}
        job = store.create_job(kind="count", lane="background", params={})
        store.update_job(job.id, status="running", owner=dead)
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        store.set_launch_job(launch.id, job.id)
        mine = store.create_job(kind="count", lane="background", params={})
        store.update_job(mine.id, status="running", owner=JobRunner(store, Bus(), Proxy()).owner)

        swept = JobRunner(store, Bus(), Proxy(), catalog=kinds()).sweep()

        assert swept == 1
        assert store.get_job(job.id).status == "interrupted"
        assert store.get_launch(launch.id).status == "interrupted"
        assert store.get_job(mine.id).status == "running"  # its process is alive: this one


def _host() -> str:
    import socket

    return socket.gethostname()


def test_many_jobs_at_once_are_each_run_exactly_once(store):
    runner = JobRunner(store, Bus(), Proxy(), catalog=kinds(), background=3)
    try:
        submitted = []
        threads = [
            threading.Thread(target=lambda: submitted.append(runner.submit("count", {"steps": 2})))
            for _ in range(12)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        results = [settled(runner, job["id"]) for job in submitted]
        assert [job["result"] for job in results] == [{"counted": 2}] * 12
    finally:
        runner.shutdown(10)
