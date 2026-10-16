"""Refinements through their front ends: the HTTP API behind the web UI — every change behind
the token, the proposer's key set but never read back — and the CLI."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from typer.testing import CliRunner

from tuningfork.cli import app as cli
from tuningfork.ops import refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.proxy.app import create_browse_app
from tests.helpers import control_headers, tally_model, task_stack
from tests.mock_server import MockTurn

API = "/_tuningfork/api/refinements"
PROPOSER = "/_tuningfork/api/proposer"
RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEYS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total."}
KEY = "sk-front-0123456789abcdefXYZ"


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        plan = prepare_batch(built.store, built.spec.path, base_url=f"{built.url}/v1", workdir=built.tmp / "batches",
                             model="m", harness="fake", n=1)
        execute_batch(built.store, plan)
        (run,) = built.store.runs(limit=5, task_id="tally-report/1")
        built.run_id = run.id
        yield built


def client(stack, token: bool = True) -> httpx.Client:
    return httpx.Client(base_url=stack.url, timeout=60, headers=control_headers(stack.app) if token else {})


class TestOverHttp:
    def test_a_refinement_is_started_proposed_to_and_closed(self, stack):
        with client(stack) as http:
            made = http.post(API, json={"run": stack.run_id, "step": "write_file", "target_rules": RULES})
            assert made.status_code == 201, made.text
            rid = made.json()["id"]
            again = http.post(API, json={"run": stack.run_id, "step": "write_file", "target_rules": RULES})
            assert again.json()["id"] == rid  # the same checkpoint and target: the one there
            proposed = http.post(f"{API}/{rid[:8]}/proposals", json={"edits": [KEYS], "rationale": "the check reads row_count"})
            assert proposed.status_code == 201, proposed.text
            (listed,) = http.get(API, params={"run": stack.run_id}).json()
            assert listed["id"] == rid and listed["proposals"] == 1
            shown = http.get(f"{API}/{rid}").json()
            (proposal,) = shown["proposals"]
            assert proposal["proposers"][0]["via"] == "person"
            assert [line["op"] for hunk in proposal["diff"] for line in hunk["lines"] if line["op"] != " "] == ["-", "+"]
            assert http.post(f"{API}/{rid}/close").json()["status"] == "closed"
            late = http.post(f"{API}/{rid}/proposals", json={"edits": [KEYS], "rationale": "r"})
            assert late.status_code == 409 and late.json()["error"]["field"] == "refinement"

    def test_the_check_as_a_refinements_target(self, stack):
        """From the page as from the terminal: the task's check, for a failure only it can see."""
        with client(stack) as http:
            made = http.post(API, json={"run": stack.run_id, "step": "write_file", "target_check": True})
            assert made.status_code == 201, made.text
            shown = http.get(f"{API}/{made.json()['id']}").json()
        assert shown["refinement"]["target"]["kind"] == "check"

    def test_a_persons_whole_text_may_change_more_but_is_warned(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        text = made.baseline_text() + "".join(f"\nA new line {i}." for i in range(12))
        with client(stack) as http:
            proposed = http.post(f"{API}/{made.id}/proposals", json={"text": text, "rationale": "more guidance"})
        assert proposed.status_code == 201, proposed.text
        assert "a screen will not say which of them mattered" in proposed.json()["warnings"][0]

    def test_refusals_name_their_field(self, stack):
        with client(stack) as http:
            nothing = http.post(API, json={"run": stack.run_id, "step": "write_file"})
            unknown = http.post(API, json={"run": stack.run_id, "step": "1", "target_rules": RULES, "extra": 1})
            missing = http.get(f"{API}/nosuch")
        assert nothing.status_code == 400 and nothing.json()["error"]["field"] == "target"
        assert unknown.status_code == 400 and unknown.json()["error"]["field"] == "extra"
        assert missing.status_code == 404

    def test_a_look_at_the_evidence_leaves_nothing_behind(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        with client(stack) as http:
            packet = http.get(f"{API}/{made.id}/packet").json()
        assert packet["packet"]["answers_here"]["reached"] == 0 and packet["size"] > 0
        assert not (made.folder / "packets").exists()

    def test_every_change_needs_the_token(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        with client(stack, token=False) as http:
            assert http.get(API).status_code == 200  # reads stay open
            assert http.post(API, json={"run": stack.run_id, "step": "1", "target_rules": RULES}).status_code == 401
            assert http.post(f"{API}/{made.id}/proposals", json={"edits": [KEYS], "rationale": "r"}).status_code == 401
            assert http.put(f"{PROPOSER}/key", json={"key": KEY}).status_code == 401
        assert refine.proposals(stack.store, made.id) == []

    def test_the_proposers_key_is_set_and_never_read_back(self, stack):
        with client(stack) as http:
            changed = http.put(PROPOSER, json={"url": "https://api.example.test", "model": "big", "allow_remote": True})
            assert changed.status_code == 200, changed.text
            stored = http.put(f"{PROPOSER}/key", json={"key": KEY})
            shown = http.get(PROPOSER)
            bad = http.put(PROPOSER, json={"wire": "smoke-signals"})
            cleared = http.put(f"{PROPOSER}/key", json={"key": None})
        assert stored.json() == {"set": True, "from": "file"}
        assert shown.json()["key"] == {"set": True, "from": "file"} and shown.json()["url"] == "https://api.example.test"
        assert all(KEY not in response.text for response in (changed, stored, shown))
        assert bad.status_code == 400 and bad.json()["error"]["field"] == "wire"
        assert cleared.json() == {"set": False, "from": None}

    async def test_the_browsing_server_reads_and_changes_nothing(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_browse_app(stack.store)), base_url="http://localhost:8788"
        ) as http:
            listed = (await http.get(API)).json()
            refused = await http.post(f"{API}/{made.id}/proposals", json={"edits": [KEYS], "rationale": "r"})
            proposer = (await http.get(PROPOSER)).json()
        assert [each["id"] for each in listed] == [made.id]
        assert refused.status_code == 405 and proposer["key"] == {"set": False, "from": None}


class TestAsJobs:
    def wait(self, http, job_id: str) -> dict:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            job = http.get(f"/_tuningfork/api/jobs/{job_id}").json()
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(0.1)
        raise AssertionError(f"job {job_id} still running")

    def test_the_built_in_proposer_runs_through_the_proxy_unrecorded(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        answer = json.dumps({"proposals": [{"edits": [KEYS], "rationale": "the check reads row_count"}]})

        def respond(payload: dict) -> MockTurn:
            system = (payload.get("messages") or [{}])[0].get("content") or ""
            return MockTurn(content=answer) if system.startswith("You improve an agent skill") else tally_model(payload)

        stack.mock.responder = respond
        runs = len(stack.store.runs(limit=100))
        with client(stack) as http:
            plan = http.post("/_tuningfork/api/jobs", json={"kind": "propose", "params": {"refinement": made.id}, "dry_run": True})
            assert plan.status_code == 200, plan.text
            assert plan.json()["plan"]["through_the_proxy"] is True and plan.json()["plan"]["remote"] is False
            assert not (made.folder / "packets").exists()  # a plan looked at leaves nothing
            started = http.post("/_tuningfork/api/jobs", json={"kind": "propose", "params": {"refinement": made.id}})
            job = self.wait(http, started.json()["id"])
        assert job["status"] == "done", job
        (accepted,) = job["result"]["accepted"]
        assert accepted["proposers"][0]["via"] == "builtin" and accepted["proposers"][0]["packet"]
        assert (made.folder / "packets" / f"{accepted['proposers'][0]['packet']}.json").is_file()
        assert len(stack.store.runs(limit=100)) == runs  # the proposer's call is no run

    def test_a_screen_job_names_its_family(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        pid = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"}).id
        with client(stack) as http:
            empty = http.post("/_tuningfork/api/jobs", json={"kind": "screen", "params": {"refinement": made.id}, "dry_run": True})
            started = http.post("/_tuningfork/api/jobs", json={"kind": "screen", "params": {"refinement": made.id, "proposals": [pid[:8]], "n": 4}})
            job = self.wait(http, started.json()["id"])
            shown = http.get(f"{API}/{made.id}").json()
        assert empty.status_code == 400 and empty.json()["error"]["field"] == "proposals"
        assert job["status"] == "done" and job["result"]["screen"]["survivors"] == [pid[:8]]
        (verdict,) = shown["proposals"][0]["screens"]
        assert verdict["survives"] is True and shown["screens"][0]["status"] == "done"


class TestTheCli:
    def invoke(self, stack, *args: str, input: str | None = None):
        result = CliRunner().invoke(cli, [*args, "--home", str(stack.store.home)], input=input)
        return result

    def test_the_loop_from_a_terminal(self, stack, tmp_path):
        started = self.invoke(stack, "refine", "start", stack.run_id[:8], "write_file", "--target-rules", json.dumps(RULES))
        assert started.exit_code == 0, started.output
        assert 'calls write_file, and mentions "row_count"' in started.output, started.output
        (made,) = refine.list_refinements(stack.store)
        edited = tmp_path / "mine.md"
        edited.write_text(made.baseline_text().replace(KEYS["old"], KEYS["new"]), encoding="utf-8")
        proposed = self.invoke(stack, "refine", "propose", made.id[:8], "--file", str(edited), "--why", "the check reads row_count")
        assert proposed.exit_code == 0, proposed.output
        (pid,) = [each.id for each in refine.proposals(stack.store, made.id)]
        screened = self.invoke(stack, "refine", "screen", made.id[:8], "-n", "4", "--base-url", stack.url)
        assert screened.exit_code == 0, screened.output
        assert "survives" in screened.output and pid[:8] in screened.output
        # What qualifies its samples, as a resample says it — here, a server that is not
        # llama.cpp's — each said once, though every arm keeps it.
        said = [line for line in screened.output.splitlines() if "parity:" in line]
        assert said and len(said) == len(set(said)) and "not llama.cpp" in screened.output
        shown = self.invoke(stack, "refine", "show", made.id[:8])
        assert shown.exit_code == 0, shown.output
        assert f"- {KEYS['old']}" in shown.output and f"+ {KEYS['new']}" in shown.output
        assert "a person" in shown.output and "survives — 4/4 against the original's 0/4" in shown.output
        assert f"{pid[:8]} survived" in shown.output
        listed = self.invoke(stack, "refine", "list")
        assert made.id[:8] in listed.output
        assert self.invoke(stack, "refine", "list", "-q", "-n", "1").output.strip() == made.id
        closed = self.invoke(stack, "refine", "close", made.id[:8])
        assert closed.exit_code == 0 and "closed" in closed.output

    def test_the_packet_is_printed_and_not_kept(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        printed = self.invoke(stack, "refine", "show", made.id[:8], "--packet", "--no-task")
        assert printed.exit_code == 0, printed.output
        packet = json.loads(printed.output)
        assert packet["included"] == {"task_criteria": False} and "task" not in packet
        assert not (made.folder / "packets").exists()

    def test_a_refusal_is_one_readable_line(self, stack):
        frozen = refine.run_skill_file(stack.store, stack.store.get_run(stack.run_id))
        frozen.write_text("# a skill this run never saw\n", encoding="utf-8")
        refused = self.invoke(stack, "refine", "start", stack.run_id[:8], "1", "--target-match", "row_count")
        assert refused.exit_code == 1
        assert "not in the context at step 1" in refused.output and "no step of this run" in refused.output
        both = self.invoke(stack, "refine", "propose", "x", "--builtin", "--file", "f.md")
        assert both.exit_code == 1 and "--file with --why, or --builtin" in both.output

    def test_the_proposers_settings_and_key(self, stack):
        made = refine.create(stack.store, stack.run_id, "write_file", target_rules=RULES)
        remote = self.invoke(stack, "connections", "proposer", "set", "--url", "https://api.example.test", "--model", "big")
        assert remote.exit_code == 0 and "not allowed yet" in remote.output
        refused = self.invoke(stack, "refine", "propose", made.id[:8], "--builtin", "--dry-run")
        assert refused.exit_code == 1 and "--allow-remote" in refused.output
        allowed = self.invoke(stack, "connections", "proposer", "set", "--allow-remote", "--key-stdin", input=KEY + "\n")
        assert allowed.exit_code == 0, allowed.output
        assert "allowed" in allowed.output and "key          set, in the recordings folder" in allowed.output
        planned = self.invoke(stack, "refine", "propose", made.id[:8], "--builtin", "--dry-run", "-m", "2")
        assert planned.exit_code == 0, planned.output
        assert "api.example.test:443, which is not this machine" in planned.output
        shown = self.invoke(stack, "connections", "proposer", "show")
        assert all(KEY not in output for output in (allowed.output, planned.output, shown.output))
        cleared = self.invoke(stack, "connections", "proposer", "set", "--clear-key", "--url", "none")
        assert "key          not set" in cleared.output and "the proxy's own server" in cleared.output


def test_a_validation_says_a_timed_out_run_counts_as_a_failure():
    """The gate counts a validation's timeout as a failure, so an edit that makes the model
    loop cannot drop out of its own count: the line printed as the run lands says so, rather
    than the batch's "not counted"."""
    from types import SimpleNamespace

    from tuningfork.cli import _fork_line

    timed_out = SimpleNamespace(checked=True, passed=False, run_id="35e787c5aa", exclusion="timed out",
                                duration_s=600.0, live_outcome=None)
    assert "timed out — counted as a failure" in _fork_line(timed_out, timeouts_fail=True)
    assert "timed out — not counted" in _fork_line(timed_out)
