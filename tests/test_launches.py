"""Launches: a harness process's calls, attributed to it exactly and as they arrive.

Before launches, runs were told apart by their message paths and a first-in, first-out
queue of labels. Two copies of one task open with byte-identical requests, so running them
at once could cross their steps, and a queued run boundary captured whatever call came
next, from anywhere. These tests pin the properties that replaced that.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

from tuningfork.config import Settings
from tuningfork.demo.mock import MockTurn
from tuningfork.harness import HermesAdapter
from tuningfork.launches import LaunchThreads, launch_url, proxy_root, request_path, split
from tuningfork.ops import OperationError
from tuningfork.ops.launch import check_proxy
from tuningfork.proxy.app import create_app
from tuningfork.runner import BatchRunner, prepare_home
from tests.helpers import as_response, control_headers, mcp_headers, serving, token_in
from tests.test_runner import GOOD_REPORT, RecordingHarness, ScriptedHarness, make_spec

CHAT = "/v1/chat/completions"


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def chat(*messages: dict, **extra) -> dict:
    return {"model": "m", "messages": list(messages), **extra}


def launched(token: str, path: str = CHAT) -> str:
    return f"/_launch/{token}{path}"


class TestAddresses:
    def test_a_launch_url_keeps_the_path_the_harness_expects(self):
        token = "a" * 32
        assert launch_url("http://127.0.0.1:8787/v1", token) == f"http://127.0.0.1:8787/_launch/{token}/v1"
        assert launch_url("http://127.0.0.1:8787", token) == f"http://127.0.0.1:8787/_launch/{token}"
        assert launch_url("http://127.0.0.1:8787/", token) == f"http://127.0.0.1:8787/_launch/{token}"
        assert proxy_root(f"http://127.0.0.1:8787/_launch/{token}/v1") == "http://127.0.0.1:8787"

    def test_the_prefix_is_split_off_the_path(self):
        token = "0123456789abcdef" * 2
        assert split(f"/_launch/{token}/v1/messages") == (token, "/v1/messages")
        assert split(f"/_launch/{token}") == (token, "/")
        assert split("/v1/chat/completions") == (None, "/v1/chat/completions")
        # Not a token: left alone, and relayed as the path it is.
        assert split("/_launch/xyz/v1/models") == (None, "/_launch/xyz/v1/models")


class TestThreads:
    def test_a_call_extending_the_last_one_continues_its_thread(self):
        threads = LaunchThreads("t" * 32, label="task#b.0", task_id="task")
        first = threads.assign(request_path(chat(user("go"))))
        second = threads.assign(
            request_path(chat(user("go"), {"role": "assistant", "content": "ok"}, user("more")))
        )
        assert (first.thread, first.idx) == (0, 0)
        assert (second.thread, second.idx, second.run_id) == (0, 1, first.run_id)
        assert first.label == "task#b.0" and first.main

    def test_a_retry_repeats_its_thread_and_a_new_root_starts_another(self):
        threads = LaunchThreads("t" * 32, label="task#b.0", task_id="task")
        threads.assign(request_path(chat(user("go"))))
        retry = threads.assign(request_path(chat(user("go"))))
        child = threads.assign(request_path(chat(user("a subagent's own task"))))
        assert (retry.thread, retry.idx) == (0, 1)
        assert (child.thread, child.idx) == (1, 0)
        assert child.label == "task#b.0~1" and not child.main
        assert child.task_id == "task"  # breakpoints scoped to the task stop it too

    def test_a_side_call_never_extends_the_tasks_own_thread(self):
        threads = LaunchThreads("t" * 32, label="task#b.0")
        threads.assign(request_path(chat(user("go"))))
        answered = (user("go"), {"role": "assistant", "content": "done"})
        review = threads.assign(request_path(chat(*answered, user("review"))), side=True)
        more = threads.assign(
            request_path(chat(*answered, user("review"), {"role": "assistant", "content": "x"}, user("y"))),
            side=True,
        )
        assert (review.thread, review.idx, review.main, review.role) == (1, 0, False, "side")
        assert (more.thread, more.idx, more.role) == (1, 1, "side")

    def test_a_call_dropped_before_the_model_gives_its_step_back(self):
        threads = LaunchThreads("t" * 32)
        dropped = threads.assign(request_path(chat(user("go"))))
        threads.release(dropped)
        retried = threads.assign(request_path(chat(user("go"))))
        assert retried.idx == dropped.idx == 0


class TestStore:
    def test_a_launched_call_is_recorded_where_it_was_assigned(self, store):
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)
        request = chat(user("go"))
        store.record_step(request=request, assignment=threads.assign(request_path(request)))

        (run,) = store.launch_runs(launch.id)
        assert (run.label, run.task_id, run.role, run.launch_id) == ("t/1#b.0", "t/1", "main", launch.id)

    def test_a_queued_run_boundary_does_not_capture_launched_calls(self, store):
        """A person's `label-next` waits for *their* next run, not a batch's."""
        store.begin_run("manual", "t/9")
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)
        for text in ("one", "two"):
            request = chat(user(text))
            store.record_step(request=request, assignment=threads.assign(request_path(request)))

        assert store.pending_labels() == ["manual"]
        assert all(run.launch_id == launch.id for run in store.launch_runs(launch.id))

    def test_unlaunched_traffic_never_joins_a_launched_run(self, store):
        """Even with identical messages: a call without the prefix is not the launch's."""
        launch = store.create_launch(kind="batch", label="t/1#b.0")
        threads = LaunchThreads(launch.id, label=launch.label)
        request = chat(user("go"))
        store.record_step(request=request, assignment=threads.assign(request_path(request)))

        stray = store.record_step(request=chat(user("go"), {"role": "assistant", "content": "x"}, user("y")))

        launched_run = store.launch_runs(launch.id)[0]
        assert stray.run_id != launched_run.id
        assert store.get_run(stray.run_id).launch_id is None
        assert store.get_run(stray.run_id).forked_from_run is None

    def test_hermes_s_background_review_is_a_side_thread_and_never_the_final_reply(self, store):
        """Found on a real Hermes batch: after its answer, Hermes replayed the conversation with
        a review prompt, and was cut off when it exited. Filed as the task's next step, that
        empty reply was judged as the run's, and a run whose answer passed failed its check."""
        from tests.test_harness import REVIEW
        from tuningfork.proxy.launches import LaunchState
        from tuningfork.runner import final_reply, named_side_call

        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1", harness="hermes")
        state = LaunchState(launch)
        answered = (user("go"), {"role": "assistant", "content": "The total is 212.00."})
        for request, response in (
            (chat(user("go")), {"role": "assistant", "content": "The total is 212.00."}),
            (chat(*answered, user(REVIEW)), None),  # cut off as Hermes exited
        ):
            store.record_step(
                request=request, response=None if response is None else as_response(response),
                error=None if response else "the client went away", harness="hermes",
                assignment=state.assign(request, None),
            )
        store.finalize_threads(launch.id, HermesAdapter.subagent_tools)

        main, review = store.launch_runs(launch.id)
        assert (main.role, main.step_count, review.role) == ("main", 1, "side")
        assert final_reply(store, main.id) == "The total is 212.00."
        # its failure — cut off as Hermes exited — says nothing of the run
        assert named_side_call(store, HermesAdapter(), review)
        assert not named_side_call(store, HermesAdapter(), main)

    def test_a_run_recorded_before_ends_on_its_review_and_its_final_reply_skips_it(self, store):
        from tests.test_harness import REVIEW
        from tuningfork.runner import final_reply

        answered = (user("go"), {"role": "assistant", "content": "The total is 212.00."})
        store.record_step(request=chat(user("go")), response=as_response(answered[1]), harness="hermes")
        store.record_step(request=chat(*answered, user(REVIEW)), error="the client went away", harness="hermes")
        (run,) = store.runs()
        assert run.step_count == 2 and final_reply(store, run.id) == "The total is 212.00."

    def test_a_launchs_threads_are_told_apart_after_it_ends(self, store):
        """A thread that began while an `Agent` call was outstanding is its subagent; one
        that began later is the harness's own side call."""
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)

        def call(request, arrived, response=None):
            return store.record_step(
                request=request,
                response=response,
                timings={"arrived_at": arrived},
                assignment=threads.assign(request_path(request)),
            )

        spawn = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Agent", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]}
        main0 = call(chat(user("task")), 100.0, spawn)
        child = call(chat(user("the subagent's goal")), 105.0)
        call(chat(user("task"), spawn["choices"][0]["message"],
                  {"role": "tool", "tool_call_id": "c1", "content": "done"}), 110.0)
        side = call(chat(user("summarise this title")), 120.0)

        store.finalize_threads(launch.id, {"Agent"})

        subagent = store.get_run(child.run_id)
        assert (subagent.role, subagent.parent_run_id, subagent.spawned_at) == (
            "subagent", main0.run_id, 0
        )
        assert store.get_run(side.run_id).role == "side"
        # Listings show the main run only; its threads belong to it.
        assert [run.id for run in store.runs()] == [main0.run_id]
        assert {run.id for run in store.threads_of(main0.run_id)} == {child.run_id, side.run_id}
        assert store.get_run(child.run_id).task_id is None

    def test_a_subagent_started_in_the_background_is_known_by_its_prompt(self, store):
        """Found on the real stack: both harnesses can delegate in the background, so the
        parent moves on at once and its child begins afterwards — outside any waiting time.
        The child's first request carries the prompt the spawning call passed."""
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)

        def call(request, arrived, response=None):
            return store.record_step(
                request=request, response=response, timings={"arrived_at": arrived},
                assignment=threads.assign(request_path(request)),
            )

        prompt = "Count the data rows in data.csv and add up the amount column."
        spawn = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "delegate_task", "arguments": json.dumps({"goal": prompt})}}]},
            "finish_reason": "tool_calls"}]}
        main0 = call(chat(user("task")), 100.0, spawn)
        call(chat(user("task"), spawn["choices"][0]["message"],
                  {"role": "tool", "tool_call_id": "c1", "content": "dispatched"}), 101.0)
        child = call(chat({"role": "system", "content": f"You are a subagent. Goal: {prompt}"},
                          user("begin")), 103.0)
        other = call(chat(user("unrelated housekeeping")), 104.0)

        store.finalize_threads(launch.id, {"delegate_task"})

        found = store.get_run(child.run_id)
        assert (found.role, found.parent_run_id, found.spawned_at) == ("subagent", main0.run_id, 0)
        assert store.get_run(other.run_id).role == "side"


class TestProxy:
    async def test_a_launched_call_is_relayed_without_its_prefix_and_recorded_under_it(
        self, proxy_client, mock_server, recorder, settle
    ):
        launch = recorder.store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        mock_server.script(MockTurn(content="hello"))

        response = await proxy_client.post(launched(launch.id), json=chat(user("hi")))
        await settle()

        assert response.status_code == 200
        assert mock_server.seen[-1][1] == CHAT
        (run,) = recorder.store.launch_runs(launch.id)
        assert run.label == "t/1#b.0" and run.task_id == "t/1"

    async def test_a_call_refused_while_paused_gives_its_step_number_to_its_retry(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        """Found in review: a call refused while tuningfork was paused kept its step number, so
        its retry was recorded a step later, and a fork of a later step, which counts steps by
        place, went live a call late."""
        launch = recorder.store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"))
        assert (await proxy_client.post(launched(launch.id), json=chat(user("hi")))).status_code == 200
        later = chat(user("hi"), {"role": "assistant", "content": "first"}, user("and then"))
        proxy_app.state.gate.paused = True
        try:
            refused = await proxy_client.post(launched(launch.id), json=later)
        finally:
            proxy_app.state.gate.paused = False
        assert refused.status_code == 503
        assert (await proxy_client.post(launched(launch.id), json=later)).status_code == 200
        await settle()
        (run,) = recorder.store.launch_runs(launch.id)
        assert [step.idx for step in recorder.store.steps(run.id)] == [0, 1]

    async def test_a_forks_call_at_its_step_refused_while_paused_leaves_its_retry_the_step(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        """Found in review: the call at a fork's step, refused while tuningfork was paused, had
        already marked the fork live, so its retry was recorded with no word that the fork went
        live there, nor how its workspace compared with the source's."""
        source = recorder.store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        first = chat(user("hi"))
        later = chat(user("hi"), {"role": "assistant", "content": "first"}, user("and then"))
        mock_server.script(MockTurn(content="first"), MockTurn(content="second"), MockTurn(content="live"))
        for request in (first, later):
            assert (await proxy_client.post(launched(source.id), json=request)).status_code == 200
        await settle()
        (run,) = recorder.store.launch_runs(source.id)
        fork = recorder.store.create_launch(kind="fork", label="t/1#b.0@1.0", task_id="t/1", source_run=run.id, source_step=1)
        assert (await proxy_client.post(launched(fork.id), json=first)).status_code == 200
        proxy_app.state.gate.paused = True
        try:
            refused = await proxy_client.post(launched(fork.id), json=later)
        finally:
            proxy_app.state.gate.paused = False
        assert refused.status_code == 503
        assert (await proxy_client.post(launched(fork.id), json=later)).status_code == 200
        await settle()
        (forked,) = recorder.store.launch_runs(fork.id)
        steps = recorder.store.steps(forked.id)
        assert [(step.idx, step.served) for step in steps] == [(0, True), (1, False)]
        assert (steps[1].replay or {}).get("live_from") == recorder.store.steps(run.id)[1].id

    async def test_a_relay_launch_is_forwarded_and_gated_but_never_recorded(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        """It carries a private proxy's calls — a held-out task's — through this proxy's gate."""
        relay = recorder.store.create_launch(kind="relay", label="relay", settings={"record": False, "breakpoints": False})
        mock_server.script(MockTurn(content="hello"))

        response = await proxy_client.post(launched(relay.id), json=chat(user("held-out task")))
        await settle()

        assert response.status_code == 200 and mock_server.seen[-1][1] == CHAT
        assert recorder.store.launch_runs(relay.id) == []
        assert recorder.store.runs(limit=10) == []
        assert recorder.store.pending_labels() == []
        # Paused, it is refused like any model call: the pause is the person's, for all load.
        proxy_app.state.gate.paused = True
        try:
            refused = await proxy_client.post(launched(relay.id), json=chat(user("again")))
        finally:
            proxy_app.state.gate.paused = False
        assert refused.status_code == 503

    async def test_a_proxy_without_its_api_serves_no_reads(self, settings, upstream_client, recorder):
        """A private proxy's recordings are a held-out task's: nothing on its port shows them."""
        app = create_app(settings, client=upstream_client, recorder=recorder, api=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787", headers=control_headers(app)
        ) as client:
            assert (await client.get("/_tuningfork/health")).status_code == 200
            for path in ("/_tuningfork/api/runs", "/_tuningfork/api/tasks", "/_tuningfork/ui/"):
                assert (await client.get(path)).status_code == 404, path
            assert (await client.get("/_tuningfork/mcp", headers=mcp_headers(app))).status_code == 404

    @pytest.mark.parametrize("path", [CHAT, "/v1/messages"])
    async def test_an_unknown_launch_is_refused_in_the_harness_dialect(
        self, proxy_client, mock_server, path
    ):
        request = chat(user("hi")) if path == CHAT else {
            "model": "m", "max_tokens": 10, "messages": [user("hi")]
        }
        response = await proxy_client.post(launched("f" * 32, path), json=request)

        assert response.status_code == 404
        assert response.headers["x-should-retry"] == "false"
        body = response.json()
        assert "same home" in json.dumps(body)
        assert mock_server.requests == []

    async def test_a_launch_begun_under_another_proxy_is_interrupted_not_guessed(
        self, settings, upstream_client, recorder, mock_server, settle
    ):
        """This proxy cannot know where the next call of a launch it never saw belongs."""
        launch = recorder.store.create_launch(kind="batch", label="t/1#b.0")
        mock_server.script(MockTurn(content="one"), MockTurn(content="two"))
        first = create_app(settings, client=upstream_client, recorder=recorder)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=first), base_url="http://localhost:8787") as harness:
            assert (await harness.post(launched(launch.id), json=chat(user("hi")))).status_code == 200
        await settle(first)

        restarted = create_app(settings, client=upstream_client, recorder=recorder)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted), base_url="http://localhost:8787") as harness:
            again = await harness.post(
                launched(launch.id),
                json=chat(user("hi"), {"role": "assistant", "content": "one"}, user("next")),
            )

        assert again.status_code == 409
        assert recorder.store.get_launch(launch.id).status == "interrupted"

    async def test_identical_launches_at_once_never_cross(
        self, proxy_client, mock_server, recorder, settle
    ):
        """The case that kept batches serial: the same task, twice, interleaved."""
        mock_server.responder = lambda payload: MockTurn(content="ok")
        a = recorder.store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        b = recorder.store.create_launch(kind="batch", label="t/1#b.1", task_id="t/1")
        recorder.store.begin_run("someone else's", None)  # must not be claimed either
        opening = chat(user("go"))
        follow = chat(user("go"), {"role": "assistant", "content": "ok"}, user("and then"))

        for token in (a.id, b.id):
            await proxy_client.post(launched(token), json=opening)
        for token in (b.id, a.id):
            await proxy_client.post(launched(token), json=follow)
        await settle()

        for launch in (a, b):
            (run,) = recorder.store.launch_runs(launch.id)
            assert run.label == launch.label
            assert [step.idx for step in recorder.store.steps(run.id)] == [0, 1]
        assert recorder.store.pending_labels() == ["someone else's"]

    async def test_a_launch_with_breakpoints_off_is_never_held(
        self, proxy_app, proxy_client, mock_server, recorder, settle
    ):
        """How an experiment's forks run: a breakpoint armed for something else must not
        hang them."""
        proxy_app.state.engine.add({"phase": "request"})
        launch = recorder.store.create_launch(kind="batch", settings={"breakpoints": False})
        mock_server.script(MockTurn(content="straight through"))

        response = await asyncio.wait_for(
            proxy_client.post(launched(launch.id), json=chat(user("hi"))), timeout=5
        )
        await settle()

        assert response.status_code == 200
        assert proxy_app.state.engine.holds() == []

    async def test_a_held_launched_call_knows_its_run_and_step_before_it_is_recorded(
        self, proxy_app, proxy_client, mock_server, recorder, settle
    ):
        engine = proxy_app.state.engine
        engine.add({"phase": "request", "step_min": 1})
        launch = recorder.store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
        mock_server.responder = lambda payload: MockTurn(content="ok")

        await proxy_client.post(launched(launch.id), json=chat(user("go")))
        await settle()  # the first call's run is in the store; the second's step is not
        second = asyncio.ensure_future(proxy_client.post(
            launched(launch.id),
            json=chat(user("go"), {"role": "assistant", "content": "ok"}, user("next")),
        ))
        for _ in range(200):
            if engine.holds():
                break
            await asyncio.sleep(0.01)
        (hold,) = engine.holds()
        run = recorder.store.launch_runs(launch.id)[0]
        assert (hold.run_id, hold.step, hold.task_id) == (run.id, 1, "t/1")
        engine.resolve(hold.id, "continue")
        assert (await second).status_code == 200
        await settle()

    async def test_only_model_calls_wait_at_the_gate(self, settings, upstream_client, recorder, mock_server):
        """A token count or a models lookup is not what the gate limits."""
        assert settings.max_inflight == 1
        app = create_app(settings, client=upstream_client, recorder=recorder)
        mock_server.script(MockTurn(content="slow", stall=1.5))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787") as harness:
            slow = asyncio.ensure_future(harness.post(CHAT, json=chat(user("hi"))))
            await asyncio.sleep(0.2)
            started = time.monotonic()
            models = await harness.get("/v1/models")
            waited = time.monotonic() - started
            await slow
        assert models.status_code in (200, 404)
        assert waited < 1.0

    async def test_health_says_launches_are_understood(self, proxy_client):
        health = (await proxy_client.get("/_tuningfork/health")).json()
        assert health["launches"] is True


class TestPreflight:
    def test_a_proxy_recording_elsewhere_is_refused_before_anything_runs(
        self, tmp_path, upstream_client
    ):
        settings = Settings(upstream_base_url="http://upstream.test", home=tmp_path / "proxy-home")
        app = create_app(settings, client=upstream_client)
        try:
            with serving(app) as url:
                assert check_proxy(f"{url}/v1", tmp_path / "proxy-home").max_inflight == 1
                with pytest.raises(OperationError, match="records into") as refused:
                    check_proxy(f"{url}/v1", tmp_path / "somewhere-else")
        finally:
            app.state.recorder.store.close()
        assert "--home" in refused.value.hint

    def test_no_proxy_at_all_is_refused(self, tmp_path):
        with pytest.raises(OperationError, match="no tuningfork proxy"):
            check_proxy("http://127.0.0.1:9/v1", tmp_path)


class TestRunner:
    def test_parallel_runs_are_each_their_own_launch(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = BatchRunner(
            store, RecordingHarness(GOOD_REPORT, store), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", settle=0, concurrency=3,
        )

        result = runner.run(spec, item, 4)

        assert [o.index for o in result.outcomes] == [0, 1, 2, 3]
        assert len(set(result.run_ids)) == 4
        for outcome in result.outcomes:
            run = store.get_run(outcome.run_id)
            assert run.launch_id == outcome.launch_id and run.label == outcome.label
            assert store.get_launch(outcome.launch_id).status == "ended"
        assert result.passes == 4

    def test_each_harness_is_given_its_own_launch_url(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = BatchRunner(
            store, ScriptedHarness(GOOD_REPORT), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", settle=0,
        )
        runner.run(spec, item, 2)
        tokens = {token_in(url) for url in runner.adapter.base_urls.values()}
        assert len(tokens) == 2
        assert all(url.endswith("/v1") for url in runner.adapter.base_urls.values())

    def test_a_missing_input_file_stops_the_batch_before_it_starts(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        item.files.append("not-there.csv")
        runner = BatchRunner(
            store, ScriptedHarness(GOOD_REPORT), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", settle=0,
        )
        with pytest.raises(ValueError, match="not-there.csv"):
            runner.run(spec, item, 1)
        assert runner.adapter.homes == []

    @pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
    def test_a_timeout_kills_what_the_harness_started_too(self, store, tmp_path):
        """A harness's own children — a shell, a subagent's tool — would otherwise outlive
        it and keep writing into the workspace."""
        spec, item = make_spec(tmp_path, timeout=0.5)
        late = (
            "import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(1.5); open(\"late.txt\", \"w\").write(\"x\")']);"
            "time.sleep(10)"
        )
        runner = BatchRunner(
            store, ScriptedHarness(late), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", settle=0,
        )
        (outcome,) = runner.run(spec, item, 1).outcomes
        time.sleep(2.0)
        assert outcome.timed_out
        assert not (runner.adapter.workspaces[0] / "late.txt").exists()

    def test_hermes_is_given_the_turn_limit_in_its_config_and_its_own_home(self, tmp_path):
        adapter = HermesAdapter()
        home = prepare_home(
            adapter, tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="m",
            max_turns=8,
        )
        assert "max_turns: 8" in (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert adapter.environment(home)["HOME"] == str(home.resolve())
