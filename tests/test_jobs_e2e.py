"""Jobs through the proxy on a real socket: started over HTTP, their model calls sent to the
proxy itself, what they make stored and linked as if made from the CLI.

The mock server plays the model, following the key names only when the skill states them —
so an A/B from the browser shows the known shift, as it does on the real stack.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from tuningfork.config import Settings
from tuningfork.control import ControlClient, ControlError
from tuningfork.demo.mock import MockTurn
from tuningfork.experiment.skill import text_sha256
from tuningfork.ops import OperationError
from tuningfork.ops.variants import save_variant
from tuningfork.proxy.app import create_app
from tests.helpers import PROPS, SKILL, VARIANT, control_headers, record_pilot, serving

API = "/_tuningfork/api"
#: The variant edited again: another text.
EDITED = VARIANT + "\nSay which keys you used.\n"
TAKEN = "a variant named '{}' of tally-report already exists, with other text"


def write(keys: dict) -> MockTurn:
    return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}])


def wording_sensitive(payload: dict) -> MockTurn:
    if "row_count and amount_total" in payload["messages"][0]["content"]:
        return write({"row_count": 4, "amount_total": 61.5})
    return write({"count": 4, "total": 61.5})


class Proxy:
    def __init__(self, app, url: str, store, run_id: str) -> None:
        self.app, self.url, self.store, self.run_id = app, url, store, run_id

    def client(self) -> ControlClient:
        return ControlClient(self.url)


@pytest.fixture
def proxy(tmp_path, mock_server):
    mock_server.props = PROPS
    mock_server.responder = wording_sensitive
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    app = create_app(
        Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home", max_inflight=2),
        client=upstream,
    )
    store = app.state.recorder.store
    run_id = record_pilot(store)
    try:
        with serving(app) as url:
            yield Proxy(app, url, store, run_id)
    finally:
        store.close()
        asyncio.run(upstream.aclose())


class TestResample:
    def test_a_resample_started_over_http_runs_through_the_proxy(self, proxy, mock_server):
        runs_before = len(proxy.store.runs(limit=100))
        with proxy.client() as client:
            job = client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 4, "concurrency": 2})
            assert job["status"] in ("queued", "running")
            done = client.wait_for_job(job["id"], timeout=30, poll=0.1)

        assert done["status"] == "done", done["error"]
        assert done["progress"]["done"] == 4 and done["progress"]["total"] == 4
        (experiment_id,) = done["links"]["experiments"]
        experiment = proxy.store.get_experiment(experiment_id)
        assert experiment.job_id == done["id"] and experiment.status == "done"
        assert experiment.base_url == proxy.url  # its calls went through the proxy
        assert done["result"]["distribution"]["total"] == 4
        # Sampled, not recorded: a resample is never a run of the task.
        assert len(proxy.store.runs(limit=100)) == runs_before
        calls = [seen for seen in mock_server.seen if seen[1] == "/v1/chat/completions"]
        assert len(calls) == 4
        assert all("x-tuningfork-experiment" not in headers for _, _, headers in calls)

    def test_a_dry_run_shows_the_plan_and_starts_nothing(self, proxy):
        with proxy.client() as client:
            planned = client.plan("resample", {"run": proxy.run_id, "step": "write_file", "n": 5})
            assert client.jobs() == []
        assert planned["kind"] == "resample"
        assert planned["plan"]["checkpoint"]["step"] == 1
        assert planned["plan"]["sampling"]["n"] == 5
        assert planned["total"] == 5  # what a budget is counted in

    def test_who_asked_travels_with_the_job_and_a_repeat_is_given_back(self, proxy):
        asked = {"via": "mcp", "client": "an agent", "version": "1"}
        params = {"run": proxy.run_id, "step": "1", "n": 4}
        with proxy.client() as client:
            client.pause()
            try:
                first = client.submit("resample", params, started_by=asked, dedupe=True)
                again = client.submit("resample", params, started_by=asked, dedupe=True)
                assert again["id"] == first["id"] and again["deduplicated"] is True
                assert first["started_by"] == asked
            finally:
                client.resume()
            done = client.wait_for_job(first["id"], timeout=30, poll=0.1)
        assert done["status"] == "done" and done["started_by"] == asked
        bad = httpx.post(
            f"{proxy.url}{API}/jobs", headers=control_headers(proxy.app),
            json={"kind": "resample", "params": params, "started_by": "me"},
        )
        assert bad.status_code == 400 and bad.json()["error"]["field"] == "started_by"

    def test_a_refusal_keeps_its_field_for_the_caller(self, proxy):
        with proxy.client() as client, pytest.raises(ControlError) as refused:
            client.submit("resample", {"run": proxy.run_id, "step": "1", "n": "many"})
        assert refused.value.error["field"] == "n" and refused.value.status == 400

    def test_a_bad_request_is_refused_with_the_field_named(self, proxy):
        response = httpx.post(
            f"{proxy.url}{API}/jobs", headers=control_headers(proxy.app),
            json={"kind": "resample", "params": {"run": proxy.run_id, "step": "1", "n": "many"}},
        )
        assert response.status_code == 400
        assert response.json()["error"]["field"] == "n"
        missing = httpx.post(
            f"{proxy.url}{API}/jobs", headers=control_headers(proxy.app),
            json={"kind": "resample", "params": {"run": "nothing-like-it", "step": "1"}},
        )
        assert missing.status_code == 400 and "no run matching" in missing.json()["error"]["message"]

    def test_starting_work_needs_the_token(self, proxy):
        response = httpx.post(
            f"{proxy.url}{API}/jobs",
            json={"kind": "resample", "params": {"run": proxy.run_id, "step": "1"}},
        )
        assert response.status_code == 401
        assert proxy.store.jobs() == []

    def test_a_job_waits_out_a_pause_then_runs(self, proxy):
        with proxy.client() as client:
            client.pause()
            job = client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 2})
            time.sleep(0.3)
            assert client.job(job["id"])["status"] == "queued"
            client.resume()
            assert client.wait_for_job(job["id"], timeout=30, poll=0.1)["status"] == "done"

    def test_a_cancelled_resample_keeps_its_samples_and_says_so(self, proxy, mock_server):
        def slow(payload):
            turn = wording_sensitive(payload)
            turn.stall = 0.3
            return turn

        mock_server.responder = slow
        with proxy.client() as client:
            job = client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 50})
            deadline = time.monotonic() + 20
            while (client.job(job["id"])["progress"] or {}).get("done", 0) < 2:
                assert time.monotonic() < deadline
                time.sleep(0.05)
            client.cancel_job(job["id"])
            stopped = client.wait_for_job(job["id"], timeout=30, poll=0.1)

        assert stopped["status"] == "cancelled"
        experiment = proxy.store.get_experiment(stopped["links"]["experiments"][0])
        assert experiment.status == "cancelled"
        assert 2 <= len(proxy.store.samples(experiment.id)) < 50
        assert stopped["result"]["experiment"]["id"] == experiment.id


class TestAB:
    def test_an_ab_with_a_wording_edited_in_the_browser(self, proxy, tmp_path):
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        params = {
            "run": proxy.run_id, "step": "1", "n": 6,
            "original": str(original), "variant_text": VARIANT, "variant_name": "keys-named",
            "target_match": "row_count",
        }
        with proxy.client() as client:
            planned = client.plan("ab", params)["plan"]
            saved = proxy.store.home / "variants" / "tally-report" / "keys-named.md"
            assert not saved.exists()  # a draft leaves nothing behind
            assert planned["variant"] == str(saved) and planned["target"]
            done = client.wait_for_job(client.submit("ab", params)["id"], timeout=60, poll=0.1)

        assert done["status"] == "done", done["error"]
        assert saved.read_text(encoding="utf-8") == VARIANT
        comparison = done["result"]["comparison"]
        assert comparison["target"]["original"]["successes"] == 0
        assert comparison["target"]["variant"]["successes"] == 6
        assert comparison["target"]["significant"]
        assert len(done["links"]["experiments"]) == 2
        assert {proxy.store.get_experiment(e).status for e in done["links"]["experiments"]} == {"done"}

    def test_a_name_holding_another_text_is_refused_before_anything_is_sampled(self, proxy, tmp_path):
        # Found in review of #79: the A/B editor's "Start from" fills in the name of the variant it
        # loads, and an edit of it started under that name tested the edit while both arms' records
        # named the file, which held the text before it; the edit was saved nowhere.
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        params = {
            "run": proxy.run_id, "step": "1", "n": 2,
            "original": str(original), "variant_text": VARIANT, "variant_name": "keys-named",
            "target_match": "row_count",
        }
        saved = proxy.store.home / "variants" / "tally-report" / "keys-named.md"
        with proxy.client() as client:
            first = client.wait_for_job(client.submit("ab", params)["id"], timeout=60, poll=0.1)
            assert first["status"] == "done", first["error"]
            for ask in (client.plan, client.submit):
                with pytest.raises(ControlError, match=TAKEN.format("keys-named")) as refused:
                    ask("ab", {**params, "variant_text": EDITED})
                assert refused.value.error["field"] == "variant_name" and refused.value.status == 400
            assert saved.read_text(encoding="utf-8") == VARIANT
            saved.write_bytes(b"\xff not UTF-8")  # holds no text
            with pytest.raises(ControlError, match=TAKEN.format("keys-named")):
                client.plan("ab", params)
        assert [job.id for job in proxy.store.jobs()] == [first["id"]]

    def test_a_variant_is_called_saved_under_its_name_only_while_its_file_holds_the_text_tested(self, proxy, tmp_path):
        # An A/B started under a name another text had taken, before that was refused, named that
        # file: its page said the variant was saved there.
        from tuningfork.ops import compact
        from tuningfork.ops.inspect import experiment_report

        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        params = {
            "run": proxy.run_id, "step": "1", "n": 2,
            "original": str(original), "variant_text": VARIANT, "variant_name": "keys-named",
            "target_match": "row_count",
        }
        with proxy.client() as client:
            done = client.wait_for_job(client.submit("ab", params)["id"], timeout=60, poll=0.1)
        assert done["status"] == "done", done["error"]
        saved = proxy.store.home / "variants" / "tally-report" / "keys-named.md"

        def wording() -> dict:
            return experiment_report(proxy.store, done["links"]["experiments"][0]).to_json()["wording"]

        assert wording()["variant"]["name"] == "keys-named" and "file" not in wording()["variant"]
        saved.write_text(EDITED, encoding="utf-8")
        found = wording()
        assert "name" not in found["variant"] and found["variant"]["label"] == "tally-report/keys-named.md"
        assert found["variant"]["file"] == "holds another text than the one tested"
        assert found["source"] == "sent" and found["diff"]["changed"] == 1  # the edit it tested
        lines = compact.wording_lines(compact.wording(found))
        assert "the variant's file holds another text than the one tested" in lines
        saved.unlink()
        assert wording()["variant"]["file"] == "is not there"

    def test_the_text_a_name_holds_is_tested_under_it_again_its_file_kept(self, proxy, tmp_path):
        # A browser gives a text box's text with LF line endings, whatever the file's: a variant
        # saved with CRLF and loaded into the editor again is the text its records name.
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        saved = proxy.store.home / "variants" / "tally-report" / "keys-named.md"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(VARIANT.replace("\n", "\r\n").encode("utf-8"))
        params = {
            "run": proxy.run_id, "step": "1", "n": 2,
            "original": str(original), "variant_text": VARIANT, "variant_name": "keys-named",
            "target_match": "row_count",
        }
        with proxy.client() as client:
            done = client.wait_for_job(client.submit("ab", params)["id"], timeout=60, poll=0.1)
        assert done["status"] == "done", done["error"]
        assert saved.read_bytes() == VARIANT.replace("\n", "\r\n").encode("utf-8")
        (variant,) = [e for e in map(proxy.store.get_experiment, done["links"]["experiments"]) if e.arm == "variant"]
        assert variant.variant["variant_path"] == str(saved)
        assert variant.variant["variant_sha256"] == text_sha256(VARIANT)

    def test_a_name_a_queued_ab_holds_is_refused_for_another_text_and_taken_for_its_own(self, proxy, tmp_path):
        # Each is planned while no file is there: the later would be refused only as it starts,
        # perhaps hours on, behind the work queued before it (found in review).
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        params = {
            "run": proxy.run_id, "step": "1", "n": 2,
            "original": str(original), "variant_text": VARIANT, "variant_name": "v9",
            "target_match": "row_count",
        }
        with proxy.client() as client:
            client.pause()
            try:
                first = client.submit("ab", params)
                for ask in (client.plan, client.submit):
                    with pytest.raises(ControlError, match=f"is taken by job {first['id'][:8]}, queued with another text") as refused:
                        ask("ab", {**params, "variant_text": EDITED})
                    assert refused.value.error["field"] == "variant_name"
                again = client.submit("ab", {**params, "variant_text": VARIANT.replace("\n", "\r\n")})  # the same text
            finally:
                client.resume()
            first = client.wait_for_job(first["id"], timeout=60, poll=0.1)
            again = client.wait_for_job(again["id"], timeout=60, poll=0.1)
        assert first["status"] == "done" and again["status"] == "done", (first["error"], again["error"])
        assert (proxy.store.home / "variants" / "tally-report" / "v9.md").read_text(encoding="utf-8") == VARIANT
        assert len(proxy.store.jobs()) == 2

    def test_an_ab_whose_name_was_taken_while_it_waited_is_refused_as_it_starts(self, proxy, tmp_path):
        # A job is prepared again as its turn comes: the file there then decides.
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        with proxy.client() as client:
            client.pause()
            try:
                job = client.submit("ab", {
                    "run": proxy.run_id, "step": "1", "n": 2,
                    "original": str(original), "variant_text": VARIANT, "variant_name": "v9",
                    "target_match": "row_count",
                })
                saved = save_variant(proxy.store.home, "tally-report", "v9", EDITED)
            finally:
                client.resume()
            done = client.wait_for_job(job["id"], timeout=60, poll=0.1)
        assert done["status"] == "failed" and TAKEN.format("v9") in done["error"]["message"]
        assert not done["links"].get("experiments") and proxy.store.experiments() == []
        assert saved.read_text(encoding="utf-8") == EDITED

    def test_a_name_taken_after_the_ab_was_prepared_is_refused_as_it_starts(self, proxy, tmp_path):
        # Two A/Bs started at once are each prepared before either saves its text.
        from tuningfork.ops.catalog import CATALOG, JobContext

        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        params = {
            "run": proxy.run_id, "step": "1", "n": 2,
            "original": str(original), "variant_text": VARIANT, "variant_name": "v8",
            "target_match": "row_count",
        }
        with proxy.client() as client:
            client.plan("ab", params)  # the proxy learns its address from a request for work
        kind = CATALOG["ab"]
        context = JobContext(store=proxy.store, proxy=proxy.app.state.jobs.proxy, job_id="")
        prepared = kind.prepare(context, kind.parse(params))
        saved = save_variant(proxy.store.home, "tally-report", "v8", EDITED)
        with pytest.raises(OperationError, match=TAKEN.format("v8")):
            asyncio.run(kind.execute(context, prepared, progress=None, cancel=None))
        assert proxy.store.experiments() == []
        assert saved.read_text(encoding="utf-8") == EDITED

    def test_an_ab_must_say_what_it_is_about_before_sampling(self, proxy, tmp_path):
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        with proxy.client() as client, pytest.raises(ControlError, match="declare what the A/B is about"):
            client.submit("ab", {
                "run": proxy.run_id, "step": "1", "original": str(original),
                "variant_text": VARIANT, "variant_name": "v2",
            })
        assert proxy.store.jobs() == []

    def test_a_lettered_target_comes_from_the_steps_own_resample(self, proxy, tmp_path):
        original = tmp_path / "SKILL.md"
        original.write_text(SKILL, encoding="utf-8")
        with proxy.client() as client:
            resample = client.wait_for_job(
                client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 3})["id"],
                timeout=30, poll=0.1,
            )
        targets = httpx.get(f"{proxy.url}{API}/runs/{proxy.run_id}/steps/1/targets").json()
        assert targets["resample"]["id"] == resample["links"]["experiments"][0]
        (only,) = targets["behaviours"]["clusters"]
        assert only["letter"] == "A" and "count" in only["outcome"]["label"]


class TestHistory:
    def test_the_event_stream_announces_jobs(self, proxy):
        seen: list[dict] = []
        with httpx.Client(base_url=proxy.url, timeout=30) as reader:
            with reader.stream("GET", f"{API}/events") as stream:
                lines = stream.iter_lines()
                snapshot = json.loads(next(line for line in lines if line.startswith("data: "))[6:])
                assert snapshot["jobs"] == []
                with proxy.client() as client:
                    client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 2})
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    seen.append(event)
                    if event["type"] == "job" and event["job"]["status"] == "done":
                        break
        kinds = [event["type"] for event in seen]
        assert "job_event" in kinds
        samples = [e for e in seen if e["type"] == "job_event" and e["event"]["kind"] == "sample"]
        assert [e["seq"] for e in samples] == sorted(e["seq"] for e in samples)

    def test_the_browsing_server_shows_jobs_and_starts_none(self, proxy):
        from tuningfork.proxy.app import create_browse_app

        with proxy.client() as client:
            client.wait_for_job(
                client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 1})["id"],
                timeout=30, poll=0.1,
            )
        browse = create_browse_app(proxy.store)
        with serving(browse) as url:
            listed = httpx.get(f"{url}{API}/jobs").json()
            refused = httpx.post(f"{url}{API}/jobs", json={"kind": "resample", "params": {}})
        assert [job["status"] for job in listed] == ["done"]
        assert refused.status_code == 405


def test_the_cli_follows_a_job(proxy):
    from typer.testing import CliRunner

    from tuningfork.cli import app as cli

    with proxy.client() as client:
        # As the web UI starts work: saying from which page.
        job = client.submit("resample", {"run": proxy.run_id, "step": "1", "n": 2}, started_by={"via": "web", "page": "step"})
    waited = CliRunner().invoke(cli, ["jobs", "wait", job["id"][:8], "--proxy", proxy.url])
    assert waited.exit_code == 0, waited.output
    assert f"job {job['id'][:8]}" in waited.output and "done" in waited.output
    listed = CliRunner().invoke(cli, ["jobs", "list", "--proxy", proxy.url])
    assert job["id"][:8] in listed.output and "resample" in listed.output and "web · step" in listed.output
