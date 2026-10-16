"""The HTTP API the CLI and the web UI share: live control, recorded data, and the page."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

from tuningfork.proxy.app import create_browse_app
from tests.helpers import SKILL, record_pilot, record_run, text_message, until
from tests.mock_server import MockTurn

API = "/_tuningfork/api"
CHAT = "/v1/chat/completions"


def request(content: str = "hi") -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


class TestBreakpointsAPI:
    async def test_create_list_disable_delete(self, proxy_client):
        created = await proxy_client.post(
            f"{API}/breakpoints", json={"phase": "response", "tool": "write_file"}
        )
        assert created.status_code == 201
        bp_id = created.json()["id"]

        listed = (await proxy_client.get(f"{API}/breakpoints")).json()
        assert [bp["id"] for bp in listed] == [bp_id]

        disabled = await proxy_client.patch(f"{API}/breakpoints/{bp_id}", json={"enabled": False})
        assert disabled.json()["enabled"] is False

        deleted = await proxy_client.delete(f"{API}/breakpoints/{bp_id}")
        assert deleted.status_code == 200
        assert (await proxy_client.get(f"{API}/breakpoints")).json() == []

    async def test_an_impossible_breakpoint_is_explained(self, proxy_client):
        response = await proxy_client.post(
            f"{API}/breakpoints", json={"phase": "request", "tool": "write_file"}
        )
        assert response.status_code == 400
        assert "response phase" in response.json()["error"]["message"]

    async def test_unknown_breakpoint_is_404(self, proxy_client):
        response = await proxy_client.delete(f"{API}/breakpoints/nope")
        assert response.status_code == 404


class TestHoldsAPI:
    async def test_inspect_edit_and_continue_a_held_call(self, proxy_client, proxy_app, mock_server):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        mock_server.script(MockTurn(content="done"))
        call = asyncio.create_task(proxy_client.post(CHAT, json=request("original")))

        listed = await until(lambda: proxy_app.state.engine.holds())
        hold_id = listed[0].id
        summary = (await proxy_client.get(f"{API}/holds")).json()
        assert [h["id"] for h in summary] == [hold_id]
        assert "request" not in summary[0]  # lists stay light

        detail = (await proxy_client.get(f"{API}/holds/{hold_id}")).json()
        edited = detail["request"]
        edited["messages"][0]["content"] = "edited"
        resolved = await proxy_client.post(
            f"{API}/holds/{hold_id}/continue", json={"request": edited}
        )
        assert resolved.json()["action"] == "continue"

        assert (await asyncio.wait_for(call, 3)).status_code == 200
        assert mock_server.requests[0]["messages"][0]["content"] == "edited"

    async def test_deciding_twice_is_a_conflict(self, proxy_client, proxy_app):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        hold_id = (await until(lambda: proxy_app.state.engine.holds()))[0].id

        await proxy_client.post(f"{API}/holds/{hold_id}/drop")
        again = await proxy_client.post(f"{API}/holds/{hold_id}/continue")

        assert again.status_code == 409
        await asyncio.wait_for(call, 3)

    async def test_an_unknown_action_is_refused(self, proxy_client, proxy_app):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(proxy_client.post(CHAT, json=request()))
        hold_id = (await until(lambda: proxy_app.state.engine.holds()))[0].id

        response = await proxy_client.post(f"{API}/holds/{hold_id}/teleport")

        assert response.status_code == 400
        await proxy_client.post(f"{API}/holds/{hold_id}/continue")
        await asyncio.wait_for(call, 3)

    async def test_an_unknown_hold_is_404(self, proxy_client):
        assert (await proxy_client.get(f"{API}/holds/nothing")).status_code == 404


class TestLiveControls:
    async def test_pause_and_resume(self, proxy_client):
        paused = (await proxy_client.post(f"{API}/pause")).json()
        assert paused["paused"] is True
        assert (await proxy_client.post(CHAT, json=request())).status_code == 503

        resumed = (await proxy_client.post(f"{API}/resume")).json()
        assert resumed["paused"] is False
        assert (await proxy_client.post(CHAT, json=request())).status_code == 200

    async def test_status_says_this_is_a_live_proxy(self, proxy_client):
        status = (await proxy_client.get(f"{API}/status")).json()
        assert status["live"] is True
        assert status["holds"] == []



class TestDataAPI:
    async def test_runs_and_their_steps(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)

        runs = (await proxy_client.get(f"{API}/runs")).json()
        assert [run["id"] for run in runs] == [run_id]

        detail = (await proxy_client.get(f"{API}/runs/{run_id[:8]}")).json()
        assert [step["idx"] for step in detail["steps"]] == [0, 1]
        assert "read_file" in detail["steps"][0]["outcome"]["label"]

    async def test_a_step_shows_the_skill_the_model_read(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)
        recorder.store.annotate_run(
            run_id,
            skill_hashes={"tally-report/SKILL.md": hashlib.sha256(SKILL.encode()).hexdigest()},
        )

        step = (await proxy_client.get(f"{API}/runs/{run_id}/steps/0")).json()

        (skill,) = step["skills"]
        assert skill["name"] == "tally-report"
        assert skill["source"] == "preloaded"
        assert skill["text"] == SKILL
        # Mapped back to the file the run recorded, by hash.
        assert skill["file"] == "tally-report/SKILL.md"
        system = step["request"]["messages"][0]["content"]
        assert system[skill["start"]:skill["end"]] == SKILL

    async def test_a_missing_step_is_404(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)
        assert (await proxy_client.get(f"{API}/runs/{run_id}/steps/9")).status_code == 404

    async def test_a_run_as_one_conversation(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)
        found = (await proxy_client.get(f"{API}/runs/{run_id[:8]}/timeline")).json()
        assert [entry["origin"] for entry in found["entries"]] == ["start", "start", "copy", "harness"]
        assert [step["position"] for step in found["steps"]] == [2, 4]
        assert (await proxy_client.get(f"{API}/runs/nothing/timeline")).status_code == 404

    async def test_a_steps_details_without_its_context(self, proxy_client, recorder):
        """What a step shows beside its conversation: everything but the request itself."""
        run_id = record_pilot(recorder.store)
        slim = (await proxy_client.get(f"{API}/runs/{run_id}/steps/1?context=0")).json()
        assert "request" not in slim and "view" not in slim
        assert slim["context_messages"] == 4 and slim["outcome"]["kind"] == "tool_call"

    async def test_diff_and_divergence(self, proxy_client, recorder):
        store = recorder.store
        a = record_run(store, "a", [text_message("same"), text_message("left")], task_id="t/1")
        b = record_run(store, "b", [text_message("same"), text_message("right")], task_id="t/1")

        diff = (await proxy_client.get(f"{API}/diff", params={"a": a, "b": b})).json()
        assert diff["divergence_index"] == 1

        report = (await proxy_client.get(f"{API}/divergence", params={"task": "t/1"})).json()
        assert report["divergence_index"] == 1
        assert len(report["steps"][1]["groups"]) == 2
        # What goes with passing, for the page beside the first split: no verdicts here, so nothing.
        assert report["verdict_split"] == {"judged": 0, "best": None}
        assert report["tool_split"]["best"] is None

    async def test_what_work_cost(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)

        found = (await proxy_client.get(f"{API}/spend", params={"run": run_id[:8]})).json()

        (row,) = found["rows"]
        assert row["what"].startswith(f"run {run_id[:8]}")
        assert row["calls"] == 2 and row["failed"] == 0
        assert (await proxy_client.get(f"{API}/spend")).status_code == 400
        assert (await proxy_client.get(f"{API}/spend", params={"run": "nothing"})).status_code == 404

    async def test_a_chain_link_by_link(self, proxy_client, recorder):
        from tuningfork.ops.chain import root

        folder = root(recorder.store.home)
        folder.mkdir(parents=True, exist_ok=True)
        record = {"id": "c0ffee11" + "0" * 24, "status": "done", "task_id": "t/1", "harness": "hermes", "n": 2,
                  "started_at": 1.0, "rate": None, "links": [{"index": 0, "passed": False}, {"index": 1, "passed": True}]}
        (folder / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")

        (listed,) = (await proxy_client.get(f"{API}/chains")).json()
        assert (listed["id"], listed["links"], listed["task_id"]) == (record["id"], 2, "t/1")
        shown = (await proxy_client.get(f"{API}/chains/c0ffee11")).json()
        assert [link["passed"] for link in shown["links"]] == [False, True]
        assert (await proxy_client.get(f"{API}/chains/deadbeef")).status_code == 404

    async def test_tasks_carry_a_pass_rate_with_its_interval(self, proxy_client, recorder):
        store = recorder.store
        for index, passed in enumerate([True, True, False]):
            run_id = record_run(store, f"r{index}", [text_message("x")], task_id="t/1")
            store.annotate_run(run_id, result={"passed": passed, "usable": True})
        uncounted = record_run(store, "r3", [text_message("x")], task_id="t/1")
        store.annotate_run(uncounted, result={"passed": True, "usable": False})

        (task,) = (await proxy_client.get(f"{API}/tasks")).json()

        assert task["task_id"] == "t/1"
        assert task["runs"] == 4
        assert task["uncounted"] == 1
        assert (task["passed"]["successes"], task["passed"]["total"]) == (2, 3)
        assert 0 < task["passed"]["low"] < task["passed"]["high"] < 1

    async def test_a_task_says_how_its_runs_pass(self, proxy_client, recorder):
        """What "check passed" means, beside it; and a check that changed between batches."""
        import time

        store = recorder.store
        record_run(store, "r0", [text_message("x")], task_id="t/1")
        for check in ("python3 old_check.py", {"reply_contains": "61.5"}):
            store.create_launch(kind="batch", task_id="t/1", eval={"id": "1", "prompt": "go", "tuningfork": {"check": check}})
            time.sleep(0.01)

        (task,) = (await proxy_client.get(f"{API}/tasks")).json()

        assert task["check"] == {"kind": "reply_contains", "text": 'the reply contains "61.5"'}
        assert task["checks"] == 2

    async def test_a_task_run_under_two_wordings_has_a_rate_for_each(self, proxy_client, recorder):
        """The workflow is a batch, an edit, another batch. Pooled, 0/3 and 3/3 read as a
        coin flip, 3/6 — a rate that describes neither wording."""
        store = recorder.store
        for index in range(6):
            run_id = record_run(store, f"r{index}", [text_message("x")], task_id="t/1")
            edited = index >= 3
            store.annotate_run(
                run_id,
                harness="hermes",
                skill_hashes={"t/SKILL.md": "v2" if edited else "v1"},
                parity={"skill_file": "variants/SKILL.v2.md"} if edited else {},
                result={"passed": edited, "usable": True},
            )

        (task,) = (await proxy_client.get(f"{API}/tasks")).json()

        assert task["passed"] is None
        first, second = task["conditions"]
        assert (first["passed"]["successes"], first["passed"]["total"]) == (0, 3)
        assert (second["passed"]["successes"], second["passed"]["total"]) == (3, 3)
        assert first["skill_version"] != second["skill_version"]
        assert (first["skill_file"], second["skill_file"]) == (None, "variants/SKILL.v2.md")
        assert first["harness"] == "hermes"
        assert len(second["run_ids"]) == 3

    async def test_divergence_over_chosen_runs_and_through_a_lens(self, proxy_client, recorder):
        store = recorder.store

        def call(name: str, path: str) -> dict:
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": name, "arguments": json.dumps({"path": path})}}]}

        a = record_run(store, "a", [call("read_file", "a.csv")], task_id="t/3")
        b = record_run(store, "b", [call("read_file", "b.csv")], task_id="t/3")
        record_run(store, "c", [call("terminal", "a.csv")], task_id="t/3")

        chosen = await proxy_client.get(f"{API}/divergence", params={"runs": f"{a},{b}"})
        assert chosen.json()["run_ids"] == [a, b]
        assert chosen.json()["divergence_index"] == 0

        lens = await proxy_client.get(
            f"{API}/divergence", params={"runs": f"{a},{b}", "normalizer": "tool_name"}
        )
        assert lens.json()["divergence_index"] is None
        bad = await proxy_client.get(f"{API}/divergence", params={"task": "t/3", "normalizer": "x"})
        assert bad.status_code == 400

    async def test_divergence_says_which_behaviour_went_on_to_pass(self, proxy_client, recorder):
        store = recorder.store
        for index, (last, passed) in enumerate([("left", True), ("left", True), ("right", False)]):
            run_id = record_run(
                store, f"r{index}", [text_message("same"), text_message(last)], task_id="t/2"
            )
            store.annotate_run(run_id, result={"passed": passed, "usable": True})

        report = (await proxy_client.get(f"{API}/divergence", params={"task": "t/2"})).json()

        groups = report["steps"][report["divergence_index"]]["groups"]
        passed = {group["outcome"]["label"]: group["passed"] for group in groups}
        assert (passed["left"]["successes"], passed["left"]["total"]) == (2, 2)
        assert (passed["right"]["successes"], passed["right"]["total"]) == (0, 1)

    async def test_divergence_needs_a_task(self, proxy_client):
        assert (await proxy_client.get(f"{API}/divergence")).status_code == 400

    async def test_an_experiment_distribution(self, proxy_client, recorder):
        store = recorder.store
        run_id = record_pilot(store)
        run_step = store.step(run_id, 0)
        experiment = store.create_experiment(
            kind="resample", arm="original", run_id=run_id, step_idx=0, step_id=run_step.id,
            request={"messages": []}, settings={"n": 2, "normalizers": []},
        )
        for index in range(2):
            store.add_sample(
                experiment.id, index, seed=index,
                response={"choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "ok"}}]},
                outcome_kind="text", outcome_key="text:ok", outcome_label="ok",
            )

        listed = (await proxy_client.get(f"{API}/experiments")).json()
        assert [e["id"] for e in listed] == [experiment.id]
        detail = (await proxy_client.get(f"{API}/experiments/{experiment.id[:8]}")).json()
        (cluster,) = detail["distribution"]["clusters"]
        assert cluster["rate"]["successes"] == 2
        # One sample as the model gave it: what a reflection packet quotes.
        answer = (await proxy_client.get(f"{API}/experiments/{experiment.id[:8]}/samples/1/answer")).json()
        assert answer["content"] == "ok" and answer["idx"] == 1
        missing = await proxy_client.get(f"{API}/experiments/{experiment.id[:8]}/samples/9/answer")
        assert missing.status_code == 404


class TestThePage:
    async def test_the_ui_is_served(self, proxy_client):
        redirect = await proxy_client.get("/_tuningfork/ui")
        assert redirect.status_code in (301, 302, 307)
        page = await proxy_client.get("/_tuningfork/ui/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]


class TestBrowseMode:
    """`tuningfork ui` with no proxy: recordings are browsable, live controls absent."""

    async def test_data_without_a_proxy(self, store):
        run_id = record_pilot(store)
        app = create_browse_app(store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8788"
        ) as client:
            status = (await client.get(f"{API}/status")).json()
            runs = (await client.get(f"{API}/runs")).json()
            page = await client.get("/_tuningfork/ui/")
        assert status["live"] is False
        assert [run["id"] for run in runs] == [run_id]
        assert page.status_code == 200


async def test_a_malformed_limit_is_a_client_error(proxy_client):
    response = await proxy_client.get(f"{API}/runs", params={"limit": "abc"})
    assert response.status_code == 400
    assert "whole number" in response.json()["error"]["message"]


def test_an_ambiguous_experiment_prefix_matches_nothing(store):
    run_id = record_pilot(store)
    step = store.step(run_id, 0)
    made = [
        store.create_experiment(kind="resample", arm="original", run_id=run_id, step_idx=0,
                                step_id=step.id, request={"messages": []}, settings={"n": 1})
        for _ in range(40)
    ]
    shared = next(
        a.id[0] for a in made if sum(1 for b in made if b.id[0] == a.id[0]) > 1
    )
    assert store.get_experiment(shared) is None  # ambiguous: refuse to guess
    assert store.get_experiment(made[0].id[:12]).id == made[0].id
    assert store.get_experiment("%") is None
