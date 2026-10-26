"""The MCP tools through the SDK's own client, in memory: reads on recorded data, and jobs —
started, waited on, deduplicated, capped, cancelled — on a job runner with the tests' own
kinds (tests/test_jobs.py)."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402

from tuningfork.cli import app  # noqa: E402
from tuningfork.config import REDACTED  # noqa: E402
from tuningfork.demo.mock import MockTurn  # noqa: E402
from tuningfork.jobs import JobRunner  # noqa: E402
from tuningfork.mcp.backend import InProcessJobs  # noqa: E402
from tuningfork.mcp.server import ToolServer  # noqa: E402
from tuningfork.mcp.tools import Answer, job_tool  # noqa: E402
from tuningfork.ops.common import OperationError  # noqa: E402
from tuningfork.store.db import Store  # noqa: E402
from tests.helpers import KEY, PROPS, SKILL, VARIANT, record_pilot  # noqa: E402
from tests.test_jobs import Bus, Proxy, kinds, settled  # noqa: E402
from tests.test_store_masking import NAMED, PASSWORD, PRIVATE  # noqa: E402

UPSTREAM = "http://upstream.test"
AGENT = types.Implementation(name="test-agent", version="0.1")


def refusal(result) -> bool:
    """A refusal is an answer, not an error result (Hermes counts those toward a breaker)."""
    return not result.is_error and "error" in (result.structured_content or {}) and text(result).startswith("refused")


def text(result) -> str:
    return result.content[0].text


def no_proxy() -> dict:
    raise OperationError("there is no tuningfork proxy at http://127.0.0.1:1", hint="start one", code="no_proxy")


class NoJobs:
    """A stdio server with no proxy to send work to: reads still answer."""

    def status(self) -> dict:
        return no_proxy()

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            return no_proxy()

        return refuse


# --- reads ---------------------------------------------------------------------------------------


@pytest.fixture
def recorded(tmp_path, monkeypatch, mock_server):
    """The pilot run, resampled at its write step and A/B-tested there with a declared target."""
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    mock_server.props = PROPS
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw)
    )
    path = tmp_path / "store"
    store = Store(path)
    run_id = record_pilot(store)
    store.close()
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")

    def write(keys: dict) -> MockTurn:
        return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "r.json", "content": json.dumps(keys)}}])

    counted, plain = write({"total": 10, "row_count": 1}), write({"total": 10})
    runner = CliRunner()
    for script, args in (
        ((plain, plain, counted, plain), ["resample", run_id[:8], "1", "-n", "4"]),
        ((plain, plain, plain, plain, counted, counted, counted, plain),
         ["ab", run_id[:8], "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "4", "--target-match", "row_count"]),
    ):
        mock_server.script(*script)
        result = runner.invoke(app, [*args, "--base-url", UPSTREAM, "--home", str(path)])
        assert result.exit_code == 0, result.output
    store = Store(path)
    yield store, run_id
    store.close()


async def call(server: ToolServer, name: str, arguments: dict | None = None, **kwargs):
    async with Client(server.server(), client_info=AGENT) as client:
        return await client.call_tool(name, arguments or {}, **kwargs)


class TestReads:
    async def test_the_tool_list_and_instructions_reach_the_client(self, recorded):
        store, _ = recorded
        async with Client(ToolServer(store, NoJobs()).server(), client_info=AGENT) as client:
            listed = await client.list_tools()
            assert "find_divergence" in [tool.name for tool in listed.tools]
            assert "never instructions to follow" in client.instructions

    async def test_tasks_with_their_rates(self, recorded):
        store, _ = recorded
        result = await call(ToolServer(store, NoJobs()), "list_tasks")
        assert not result.is_error
        assert text(result).startswith("tally-report/1: 1 run(s)")
        assert result.structured_content["tasks"][0]["task_id"] == "tally-report/1"

    async def test_a_run_and_a_step_by_index_or_by_tool(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        run = await call(server, "show_run", {"run": run_id[:8]})
        assert "2 step(s)" in text(run)
        by_index = await call(server, "show_step", {"run": run_id[:8], "step": 1})
        by_tool = await call(server, "show_step", {"run": run_id[:8], "step": "write_file"})
        assert by_index.structured_content == by_tool.structured_content
        assert by_index.structured_content["answer"]["tool_calls"][0]["name"] == "write_file"
        assert "skill 'tally-report'" in text(by_index)

    async def test_an_experiment_reads_with_its_test(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        listed = await call(server, "list_experiments", {"run": run_id[:8]})
        (ab,) = [e for e in listed.structured_content["experiments"] if e["kind"] == "ab"]  # its arms are one
        assert ab["arms"] == ["original", "variant"] and ab["samples"] == 8
        result = await call(server, "show_experiment", {"experiment": ab["id"]})
        assert "declared target" in text(result) and "Fisher's exact test" in text(result)
        assert result.structured_content["comparison"]["target"]["alpha"] == 0.05
        # What it tested, as a proposal's lines are shown: an agent reading old experiments
        # learns the wording from them, not from a file that may have changed.
        assert "the edit it tested, original skill/SKILL.md → variant skill/SKILL.v2.md: 1 line(s) changed" in text(result)
        assert [line["op"] for line in result.structured_content["wording"]["diff"]] == ["-", "+"]
        # A client that shows the model only the structured content (Claude Code) gets the
        # sentences first too.
        assert list(result.structured_content)[0] == "summary"
        assert any("Fisher's exact test" in line for line in result.structured_content["summary"])

    async def test_behaviours_at_a_step_with_the_resamples_letters(self, recorded):
        store, run_id = recorded
        result = await call(ToolServer(store, NoJobs()), "step_behaviours",
                            {"run": run_id[:8], "step": 1, "target_match": "row_count"})
        data = result.structured_content
        assert data["letters"]["behaviours"][0]["letter"] == "A"
        assert any(found["matches"] for found in data["behaviours"])
        assert "row_count" in text(result) and "counts 1 of them" in text(result)

    async def test_a_step_with_no_tokens_says_how_to_get_them(self, recorded):
        store, run_id = recorded
        result = await call(ToolServer(store, NoJobs()), "step_tokens", {"run": run_id[:8], "step": 1})
        assert "start_tokens" in text(result)

    async def test_one_run_is_no_comparison_and_the_refusal_says_why(self, recorded):
        store, _ = recorded
        result = await call(ToolServer(store, NoJobs()), "find_divergence", {"task": "tally-report/1"})
        assert refusal(result)
        assert "at least two runs" in text(result) and result.structured_content["error"]["hint"]

    async def test_arguments_are_parsed_strictly_and_refusals_name_the_field(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        wrong = await call(server, "show_step", {"run": run_id[:8], "step": 1, "last": "six"})
        assert refusal(wrong) and wrong.structured_content["error"]["field"] == "last"
        unknown = await call(server, "list_runs", {"tsak": "x"})
        assert refusal(unknown) and unknown.structured_content["error"]["field"] == "tsak"
        missing = await call(server, "nope")
        assert refusal(missing) and "no tool called 'nope'" in text(missing)

    async def test_without_a_proxy_reads_answer_and_work_says_what_to_do(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        status = await call(server, "server_status")
        assert not status.is_error and status.structured_content["proxy"] is None
        assert status.structured_content["recorded_servers"] == []  # the pilot recorded no server
        started = await call(server, "start_resample", {"run": run_id[:8], "step": 1, "n": 4})
        assert refusal(started) and "start one" in text(started)

    async def test_the_token_work_a_check_ruled_out_is_said(self, recorded):
        """An agent learns before it starts one that start_tokens and the rest would be refused."""
        store, _ = recorded
        server_id = store.add_server({"upstream": "http://upstream.test", "models": ["m"]})
        store.set_server_checks(server_id, {"results": {
            "logprobs": {"support": "no", "detail": "request accepted but `logprobs` came back null"},
        }})
        runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
        try:
            jobs = InProcessJobs(runner, status=lambda: {"upstream": "http://upstream.test", "server": {"id": server_id}})
            status = await call(ToolServer(store, jobs), "server_status")
        finally:
            runner.shutdown(10)
        reason = status.structured_content["cannot"]["logprobs"]
        assert reason.startswith("Keeping each sample's tokens needs token probabilities") and reason in text(status)
        assert set(status.structured_content["cannot"]) == {"logprobs"}
        assert status.structured_content["reminder"] is None

    async def test_a_hosted_model_api_is_said_to_be_one(self, recorded):
        """An agent told what its own work can't reach there, before it starts any."""
        store, _ = recorded
        runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
        read = {"internet": True, "asks_key": True, "props": False}
        try:
            jobs = InProcessJobs(runner, status=lambda: {"upstream": "https://api.example.com", "server": {},
                                                        "server_read": read})
            status = await call(ToolServer(store, jobs), "server_status")
        finally:
            runner.shutdown(10)
        noted = status.structured_content["reminder"]
        assert noted["headline"] == "This server is on the internet and asks for an API key."
        assert noted["headline"] in text(status) and "built for a model you run yourself" in text(status)
        assert set(status.structured_content["cannot"]) == {"score", "forking"}


# --- jobs ------------------------------------------------------------------------------------------


@pytest.fixture
def lab(store):
    """A job runner with the tests' own kinds, behind the tools: `start_count` counts."""
    runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
    jobs = InProcessJobs(runner, status=lambda: {"upstream": "http://upstream.test", "server": {}})
    server = ToolServer(store, jobs, limits={"count": 5000}, max_jobs=2, max_wait=5)
    server.tools["start_count"] = job_tool(kinds()["count"], max_wait=5)
    yield server, runner
    runner.shutdown(10)


class TestJobs:
    async def test_a_dry_run_plans_and_starts_nothing(self, lab, store):
        server, _ = lab
        result = await call(server, "start_count", {"steps": 3, "dry_run": True})
        assert not result.is_error and "count would make 3" in text(result)
        assert store.jobs() == []

    async def test_a_wait_returns_the_result_and_reports_progress(self, lab, store):
        server, _ = lab
        seen = []

        async def progress(done, total, message):
            seen.append((done, total))

        result = await call(server, "start_count", {"steps": 3, "delay": 0.2, "wait_seconds": 5}, progress_callback=progress)
        data = result.structured_content
        assert data["status"] == "done" and data["result"] == {"counted": 3}
        assert data["started_by"] == {"via": "mcp", "client": "test-agent", "version": "0.1"}
        assert seen and seen[-1][1] == 3
        assert store.jobs()[0].started_by["client"] == "test-agent"

    async def test_no_wait_returns_the_job_and_job_status_follows_it(self, lab, store):
        from tests.helpers import until

        server, runner = lab
        started = await call(server, "start_count", {"steps": 6, "delay": 0.1, "wait_seconds": 0})
        job = started.structured_content["id"]
        assert started.structured_content["status"] in ("queued", "running") and "job_status" in text(started)
        # The answer says how to wait — the most this route allows — and never to start it again.
        assert f'job_status(job="{job}", wait_seconds=5)' in text(started) and "never start the same work again" in text(started)
        assert "left" in text(started) or "it starts when" in text(started)
        # Once a unit has landed, how long is left at that pace, in the progress and the sentence.
        await until(lambda: ((runner.get(job) or {}).get("progress") or {}).get("remaining_s") is not None, timeout=3)
        running = await call(server, "job_status", {"job": job})
        if running.structured_content["status"] == "running":
            assert running.structured_content["progress"]["remaining_s"] >= 0
            assert "minute(s) left at its pace so far" in text(running)
        done = await call(server, "job_status", {"job": job, "wait_seconds": 5, "after": 0})
        assert done.structured_content["status"] == "done" and "remaining_s" not in (done.structured_content.get("progress") or {})
        assert [entry["kind"] for entry in done.structured_content["events"]][-1] == "sample"

    async def test_a_plans_warning_reaches_the_agent(self, lab, store):
        """Found in review: an agent saw a plan's warnings on a dry run only, so a context that
        read `<redacted>` went unsaid once its resample started."""
        server, _ = lab
        said = "step 1's context reads <redacted> in 1 place"
        result = await call(server, "start_count", {"steps": 2, "warn": said, "wait_seconds": 5})
        assert f"warning: {said}" in text(result)
        assert result.structured_content["warnings"] == [said]

    async def test_the_same_work_asked_again_is_not_started_again(self, lab, store):
        server, runner = lab
        first = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        again = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        assert again.structured_content["id"] == first.structured_content["id"]
        assert again.structured_content["deduplicated"] is True and "not started again" in text(again)
        assert len(store.jobs()) == 1
        runner.cancel(store.jobs()[0].id)

    async def test_more_than_a_job_may_make_is_refused(self, lab, store):
        server, _ = lab
        server.limits["count"] = 5
        result = await call(server, "start_count", {"steps": 10})
        assert refusal(result) and result.structured_content["error"]["code"] == "over_limit"
        assert "--limit count=N" in text(result) and store.jobs() == []

    async def test_agents_cannot_queue_work_without_end(self, lab, store):
        server, runner = lab
        for steps in (100, 101):
            await call(server, "start_count", {"steps": steps, "delay": 0.05, "wait_seconds": 0})
        third = await call(server, "start_count", {"steps": 102, "delay": 0.05, "wait_seconds": 0})
        assert refusal(third) and third.structured_content["error"]["code"] == "too_many_jobs"
        for job in store.jobs():
            runner.cancel(job.id)

    async def test_an_agent_cancels_its_own_work_and_no_one_elses(self, lab, store):
        server, runner = lab
        mine = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        stopped = await call(server, "cancel_job", {"job": mine.structured_content["id"]})
        assert not stopped.is_error
        settled(runner, store.jobs()[0].id)
        theirs = runner.submit("count", {"steps": 100, "delay": 0.05})
        refused = await call(server, "cancel_job", {"job": theirs["id"][:8]})
        assert refusal(refused) and refused.structured_content["error"]["code"] == "not_yours"
        runner.cancel(theirs["id"])
        bad = await call(server, "job_status", {"job": "not-an-id"})
        assert refusal(bad) and bad.structured_content["error"]["field"] == "job"

    async def test_a_refused_parameter_is_named_for_the_model(self, lab):
        server, _ = lab
        result = await call(server, "start_count", {"steps": "three"})
        assert refusal(result) and result.structured_content["error"]["field"] == "steps"

    async def test_the_jobs_list_says_who_started_each(self, lab):
        server, runner = lab
        await call(server, "start_count", {"steps": 1, "wait_seconds": 5})
        runner.submit("count", {"steps": 1})
        listed = await call(server, "list_jobs")
        assert sorted(bool(job.get("started_by")) for job in listed.structured_content["jobs"]) == [False, True]


def test_over_stdio_the_jobs_list_keeps_the_work_under_way_too(tmp_path):
    # As the runner lists them in the proxy's own MCP (tests/test_jobs.py): the newest, then every
    # job still under way, however many came after it.
    from types import SimpleNamespace

    from tuningfork.mcp.backend import HttpJobs

    jobs = HttpJobs("http://127.0.0.1:1", tmp_path)
    jobs._client = SimpleNamespace(jobs=lambda: [
        {"id": "a", "status": "done"}, {"id": "b", "status": "queued"}, {"id": "c", "status": "failed"},
        {"id": "d", "status": "done"}, {"id": "e", "status": "running"},
    ])
    assert [job["id"] for job in jobs.list(limit=2)] == ["a", "b", "e"]


async def test_a_legacy_client_names_itself_too(store):
    """A 2025 client (Hermes's, OpenJarvis's) gives its name at the handshake."""
    runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
    try:
        server = ToolServer(store, InProcessJobs(runner, status=dict), limits={"count": 10}, max_wait=5)
        server.tools["start_count"] = job_tool(kinds()["count"], max_wait=5)
        async with Client(server.server(), mode="legacy", client_info=AGENT) as client:
            result = await client.call_tool("start_count", {"steps": 1, "wait_seconds": 5})
            assert client.protocol_version == "2025-11-25"
        assert result.structured_content["started_by"]["client"] == "test-agent"
    finally:
        runner.shutdown(10)


def test_a_skills_text_where_its_path_goes_is_refused_for_what_it_is(store):
    """What an agent did on the real stack: the whole skill passed as `original`."""
    from tuningfork.ops.experiments import prepare_ab

    record_pilot(store)
    with pytest.raises(OperationError) as refused:
        prepare_ab(store, "pilot", "1", variant="unused.md", variant_text="edited", original="---\nname: x\n---\n",
                   target_match="x")
    assert refused.value.field == "original"
    assert "not the skill's text" in refused.value.message and "variant_text" in refused.value.hint


def test_a_dry_run_says_how_long_it_may_take():
    """A resample's bound is the recorded step's time on a cold cache; a probe's, decode time
    and, when the step's timing gave it, each continuation's prompt processing."""
    from tuningfork.mcp.tools import plan_answer
    from tuningfork.ops.catalog import CATALOG

    resample = plan_answer(CATALOG["resample"], {"total": 20, "plan": {"estimate": {"recorded_seconds": 7.8, "worst_minutes": 2.6}}})
    assert resample.lines[0] == "resample would make 20 samples; nothing was started"
    assert "at most about 3 minute(s)" in resample.lines[1]
    probe = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {"tokens": 4000, "minutes": 1.2}}})
    assert probe.lines[1] == "about 1 minute(s) at the step's recorded speed"
    counted = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {
        "tokens": 4000, "minutes": 2.3, "prompt_tokens": 13729, "prefill_seconds": 1.4}}})
    assert counted.lines[1] == "about 2 minute(s) at the step's recorded speed, its whole prompt processed for each continuation"
    uncounted = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {
        "tokens": 4000, "minutes": 1.2, "prompt_tokens": 13729}}})
    assert uncounted.lines[1].endswith("; each continuation also processes the whole prompt first")


def test_an_agent_proposes_a_bounded_number_of_edits(store, tmp_path):
    """Twelve proposals from agents to one refinement; the thirteenth is refused."""
    from tuningfork.mcp.tools import AGENT_PROPOSALS, Context, ProposeEditArgs, propose_edit
    from tuningfork.ops import refine
    from tests.helpers import SKILL

    run_id = record_pilot(store)
    frozen = tmp_path / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
    made = refine.create(store, run_id, "1", target_match="row_count")
    context, who = Context(store=store, jobs=None), {"via": "mcp", "client": "an agent"}

    def edit(k: int) -> ProposeEditArgs:
        return ProposeEditArgs(refinement=made.id, rationale="r", edits=[
            {"old": "Write the totals to report.json.", "new": f"Write the totals to report.json ({k})."}])

    for k in range(AGENT_PROPOSALS):
        propose_edit(context, edit(k), who)
    with pytest.raises(OperationError) as refused:
        propose_edit(context, edit(99), who)
    assert refused.value.code == "too_many_proposals"
    # A person is not counted against agents.
    refine.add_proposal(store, made.id, edits=edit(100).edits, rationale="mine", proposer={"via": "person"})


def test_an_agent_proposes_by_line_number_and_is_told_what_it_changed(store, tmp_path):
    """The line-addressed form: a number read off show_skill and the line as it should read —
    nothing to copy. The answer says what each edit resolved to."""
    from tuningfork.mcp.tools import Context, ProposeEditArgs, RunArgs, propose_edit, show_skill
    from tuningfork.ops import refine
    from tests.helpers import SKILL

    run_id = record_pilot(store)
    frozen = tmp_path / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
    made = refine.create(store, run_id, "1", target_match="row_count")
    context, who = Context(store=store, jobs=None), {"via": "mcp", "client": "an agent"}
    assert show_skill(context, RunArgs(run=run_id)).data["lines"][6] == "7: Write the totals to report.json."
    answer = propose_edit(context, ProposeEditArgs(refinement=made.id, rationale="r", edits=[
        {"line": 7, "new": "Write the totals to report.json, using the keys row_count and amount_total."},
        {"after": 7, "insert": "Check the counts against the input."}]), who)
    assert answer.data["changed"] == 2
    assert answer.lines[1] == ("edit 1, line 7: 'Write the totals to report.json.' → "
                               "'Write the totals to report.json, using the keys row_count and amount_total.'")
    assert answer.lines[2] == "edit 2, after line 7: adds 'Check the counts against the input.'"


def test_a_refinement_at_a_passing_run_is_told_so(store, tmp_path):
    """A 31B agent refined at the one passing run's step and screened eight sound edits against an
    original arm at 17–20 of 20: nothing could show. The answer now says where to start instead."""
    from tuningfork.mcp.tools import Context, RefinementArgs, start_refinement
    from tests.helpers import SKILL

    run_id = record_pilot(store)
    frozen = tmp_path / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)},
                       result={"passed": True, "check_output": "ok"})
    context, who = Context(store=store, jobs=None), {"via": "mcp", "client": "an agent"}
    answer = start_refinement(context, RefinementArgs(run=run_id, step="1", target_match="row_count", unseen_ok=True), who)
    note = next(line for line in answer.lines if line.startswith("note: run "))
    assert "passed its check" in note and "failed runs: none recorded yet" in note
    assert "a screen to the end from several checkpoints" in note


def test_a_step_screen_at_the_ceiling_says_why_nothing_could_show():
    from tuningfork.ops import compact

    def found(reach: str) -> dict:
        return {"family": 1, "complete": True, "reach": reach, "alpha": 0.05, "survivors": [],
                "baseline": {"target": {"successes": 18, "total": 20, "low": 0.7, "high": 0.97}},
                "arms": [{"name": "abcdef12", "target": {"successes": 20, "total": 20, "low": 0.84, "high": 1.0},
                          "adjusted_p": 0.49, "survives": False, "worse": False}]}

    at_step = compact.screen_lines(found("step"))
    assert any("the original reaches the target 18/20 at this checkpoint: no edit can show a gain here" in line
               for line in at_step)
    assert not any("no edit can show a gain here" in line for line in compact.screen_lines(found("end")))


# --- the surface a weak reader needs -------------------------------------------------------------------


def test_the_refine_toolset_orients_an_agent_in_its_descriptions():
    """Hermes shows a client no server instructions: what an agent must know is in the tools."""
    from tuningfork.mcp.tools import registry

    refine, evidence = registry("refine"), registry("evidence")
    assert "show_run of a failed run" in refine["list_tasks"].description and "show_skill" in refine["list_tasks"].description
    assert evidence["list_tasks"].description.startswith("Start here. Every task recorded")
    assert "not a file on your machine" in refine["show_skill"].description
    assert "what its check said" in refine["show_run"].description
    assert "best predicts the verdict" in refine["find_divergence"].description
    assert "unseen_ok" in refine["start_refinement"].description
    assert "show_files" in refine and "show_files" in evidence


def test_the_step_whose_behaviour_predicts_the_verdict():
    from tuningfork.ops.inspect import verdict_split

    def group(label: str, passed: int, total: int, kind: str = "tool_call") -> dict:
        return {"outcome": {"label": label, "kind": kind}, "run_ids": [f"{label}-{i}" for i in range(total)],
                "passed": {"successes": passed, "total": total}}

    report = {"divergence_index": 0, "steps": [
        {"index": 0, "groups": [group("search_files", 4, 7), group("skills_list", 2, 2)]},
        {"index": 3, "groups": [group("write_file(row_count)", 5, 5), group("write_file(rows)", 0, 4)]},
    ]}
    found = verdict_split(report)
    assert found["judged"] == 9 and found["best"]["index"] == 3 and found["best"]["label"] == "write_file(row_count)"
    assert found["best"]["passed"] == 5 and found["best"]["others_passed"] == 0 and found["best"]["p"] < 0.01
    # A run that took the behaviour, to look at — never just the first run compared.
    assert (found["best"]["run_id"], found["best"]["kind"]) == ("write_file(row_count)-0", "tool_call")
    # Where having ended is what predicts the verdict, the run is shown, there being no step to show.
    ended = {"divergence_index": 0, "steps": [
        {"index": 4, "groups": [group("run ended", 0, 4, "ended"), group("write_file(rows)", 5, 5)]},
    ]}
    best = verdict_split(ended)["best"]  # the two split alike; the first found stands
    assert (best["index"], best["kind"], best["run_id"]) == (4, "ended", "run ended-0")
    # Every run's answer its own: no behaviour predicts anything; the difference is in the content.
    singles = {"divergence_index": 0, "steps": [{"index": 0, "groups": [group(f"w{i}", i % 2, 1) for i in range(6)]}]}
    assert verdict_split(singles) == {"judged": 6, "best": None}
    assert verdict_split({"steps": []}) == {"judged": 0, "best": None}


class TestWhatAFailedRunShows:
    async def test_a_failed_run_leads_with_what_its_check_said(self, recorded):
        store, run_id = recorded
        store.annotate_run(run_id, result={"passed": False, "check_output": "report.json lacks row_count\nand more"})
        server = ToolServer(store, NoJobs(), toolset="refine")
        shown = await call(server, "show_run", {"run": run_id[:8]})
        assert "it failed its check" in text(shown)
        assert "its check said: report.json lacks row_count | and more" in text(shown)
        # The step to refine is where the check's complaint was made, not the last call: naming
        # the last call sent an agent two steps past the mistake (findings §31.42).
        assert "its last tool call" not in text(shown) and "show_files" in text(shown)
        assert "the first whose call made what the check complains about — not the last call" in text(shown)
        listed = await call(server, "list_runs", {"task": "tally-report/1"})
        assert f"failed {run_id[:8]}: report.json lacks row_count" in text(listed)
        # No snapshot was kept of this run's workspace: the refusal says what would have one.
        files = await call(server, "show_files", {"run": run_id[:8]})
        assert refusal(files) and "snapshot" in text(files)

    async def test_a_target_no_recorded_answer_satisfies_is_refused_unless_it_is_new(self, recorded, tmp_path):
        from tests.helpers import SKILL

        store, run_id = recorded
        frozen = tmp_path / "frozen" / "tally-report"
        frozen.mkdir(parents=True)
        (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
        store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
        server = ToolServer(store, NoJobs(), toolset="refine")
        refused = await call(server, "start_refinement", {"run": run_id[:8], "step": 1, "target_match": "zzz_never"})
        assert refusal(refused) and refused.structured_content["error"]["code"] == "unseen_target"
        assert "unseen_ok" in text(refused) and "step_behaviours" in text(refused)
        # Seen among the step's samples: allowed; asked for again: it says so; declared new: allowed.
        seen = await call(server, "start_refinement", {"run": run_id[:8], "step": 1, "target_match": "row_count"})
        assert not seen.is_error and "open already" not in text(seen)
        # Accepted, the target's evidence is said too: how it splits the runs recorded at the step.
        assert "the target, against the answers recorded at step 1: it counts the answers of" in text(seen)
        again = await call(server, "start_refinement", {"run": run_id[:8], "step": 1, "target_match": "row_count"})
        assert not again.is_error and "open already" in text(again)
        new = await call(server, "start_refinement",
                         {"run": run_id[:8], "step": 1, "target_match": "zzz_never", "unseen_ok": True})
        assert not new.is_error

    def test_what_a_run_wrote(self, tmp_path, mock_server, monkeypatch):
        # The task fixture serves a real proxy and closes its client with asyncio.run: a sync test.
        from tuningfork.ops.batch import execute_batch, prepare_batch
        from tests.helpers import task_stack

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1",
                                 workdir=stack.tmp / "batches", model="m", harness="fake", n=1)
            run_id = execute_batch(stack.store, plan).outcomes[0].run_id
            server = ToolServer(stack.store, NoJobs(), toolset="refine")
            listed = asyncio.run(call(server, "show_files", {"run": run_id[:8]}))
            assert not listed.is_error, text(listed)
            assert "report.json" in [entry["path"] for entry in listed.structured_content["files"]]
            assert "a passing run's" in text(listed)
            one = asyncio.run(call(server, "show_files", {"run": run_id[:8], "path": "report.json"}))
            assert not one.is_error and isinstance(json.loads(one.structured_content["text"]), dict)
            assert "report.json (" in text(one)
            whole = one.structured_content["text"]
            assert one.structured_content["page"] == {"offset": 0, "to": len(whole), "chars": len(whole), "next": None}
            # Read from a character on, and never past the end.
            later = asyncio.run(call(server, "show_files", {"run": run_id[:8], "path": "report.json", "offset": 5}))
            assert later.structured_content["text"] == whole[5:] and f"characters 5–{len(whole)}" in text(later)
            past = asyncio.run(call(server, "show_files", {"run": run_id[:8], "path": "report.json", "offset": 10_000}))
            assert refusal(past) and past.structured_content["error"]["field"] == "offset"
            # One message of a step whole: the answer's call with its arguments as sent, or a message of the context.
            step = list(stack.store.steps(run_id))[-2].idx
            answer = asyncio.run(call(server, "show_step", {"run": run_id[:8], "step": step, "message": "answer"}))
            assert not answer.is_error and "[call " in answer.structured_content["text"], text(answer)
            first = asyncio.run(call(server, "show_step", {"run": run_id[:8], "step": step, "message": "1"}))
            assert not first.is_error and first.structured_content["role"] and "message 1" in text(first), text(first)
            nowhere = asyncio.run(call(server, "show_step", {"run": run_id[:8], "step": step, "message": "99"}))
            assert refusal(nowhere) and nowhere.structured_content["error"]["field"] == "message"
            clipped = asyncio.run(call(server, "show_step", {"run": run_id[:8], "step": step}))
            assert "reads one whole, a page at a time" in text(clipped)
            missing = asyncio.run(call(server, "show_files", {"run": run_id[:8], "path": "nope.txt"}))
            assert refusal(missing) and missing.structured_content["error"]["field"] == "path"
            # A 26B asked for 'before step 4', 'at step 1', 'after step 5', eighteen times running:
            # the refusal says what `where` is, in one.
            worded = asyncio.run(call(server, "show_files", {"run": run_id[:8], "where": "before step 1"}))
            assert refusal(worded) and worded.structured_content["error"]["field"] == "where"
            assert "its number, or the name of the tool" in text(worded) and "'before step 1'" in text(worded)
            before = asyncio.run(call(server, "show_files", {"run": run_id[:8], "where": "1"}))
            assert not before.is_error and "error" not in (before.structured_content or {}), text(before)


async def test_a_client_that_shows_both_may_be_sent_the_text_alone(recorded):
    """Hermes puts a result's text and its structured content side by side in the tool
    message — every answer twice — and a 26B in a 64k context spent it in twenty-two calls.
    With results "text", an answer and a refusal carry the text alone, the JSON still in it."""
    from tuningfork.mcp.server import RESULTS

    store, run_id = recorded
    both = await call(ToolServer(store, NoJobs()), "show_run", {"run": run_id[:8]})
    once = await call(ToolServer(store, NoJobs(), results="text"), "show_run", {"run": run_id[:8]})
    assert both.structured_content and once.structured_content is None
    assert text(once) == text(both) and text(once).startswith(f"run {run_id[:8]}") and '{"run":' in text(once)
    refused = await call(ToolServer(store, NoJobs(), results="text"), "show_run", {"run": "nope"})
    assert refused.structured_content is None and not refused.is_error and text(refused).startswith("refused")
    with pytest.raises(OperationError, match="both or text"):
        ToolServer(store, NoJobs(), results="json")
    assert RESULTS == ("both", "text")


async def test_the_same_refusal_again_and_again_carries_the_tool_itself(recorded):
    """A 26B read each of twenty-seven identical refusals as the first: from the third, the
    refusal says how many times, and the tool's description again; from the sixth, to stop."""
    store, run_id = recorded
    server = ToolServer(store, NoJobs())
    first = await call(server, "show_run", {"run": "nope"})
    second = await call(server, "show_run", {"run": "nope"})
    assert refusal(first) and refusal(second) and "times running" not in text(second)
    third = await call(server, "show_run", {"run": "nope"})
    assert refusal(third) and "the same refusal 3 times running" in text(third)
    assert "One run step by step" in text(third) and "stop sending this call" not in text(third)
    assert third.structured_content["error"]["message"] == second.structured_content["error"]["message"]
    for _ in range(3):
        sixth = await call(server, "show_run", {"run": "nope"})
    assert "6 times running" in text(sixth) and "stop sending this call as it is" in text(sixth)
    assert "try another edit, another step or another tool" in text(sixth)
    # An answer, or a different refusal, ends the streak.
    assert not refusal(await call(server, "show_run", {"run": run_id[:8]}))
    again = await call(server, "show_run", {"run": "nope"})
    assert refusal(again) and "times running" not in text(again)
    other = await call(server, "show_step", {"run": "nope", "step": 1})
    assert refusal(other) and "times running" not in text(other)


def test_a_refusal_is_an_answer_and_a_failure_an_error():
    """Hermes counts error results toward a circuit breaker: a refusal must not be one."""
    from tuningfork.mcp.server import refused

    answer = refused(OperationError("its old text is not in the skill", code="bad_params", field="edits"))
    assert not answer.is_error and answer.structured_content["error"]["field"] == "edits"
    assert answer.content[0].text.startswith("refused: its old text is not in the skill")
    failure = refused(OperationError("tuningfork failed: KeyError: x", code="internal"))
    assert failure.is_error and failure.content[0].text.startswith("refused: tuningfork failed")


class TestTargetsThatCannotMove:
    """What a 26B did on the real stack: a target every answer satisfied, screened 20/20 against
    20/20, and one no answer could contain — the check's own sentence. Both are refused with what
    a target is, and the tryout says it before a screen is spent."""

    async def test_a_target_every_recorded_answer_satisfies_is_refused(self, recorded, tmp_path):
        from tests.helpers import SKILL

        store, run_id = recorded
        frozen = tmp_path / "frozen" / "tally-report"
        frozen.mkdir(parents=True)
        (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
        store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
        # Every answer at step 1 — the run's own and every sample drawn there — is a write_file call.
        server = ToolServer(store, NoJobs(), toolset="refine")
        refused = await call(server, "start_refinement", {"run": run_id[:8], "step": 1, "target_match": "write_file"})
        assert refusal(refused) and refused.structured_content["error"]["code"] == "universal_target"
        assert "could only show an edit lowering it" in text(refused) and "step_behaviours" in text(refused)
        assert "target_check" not in text(refused)  # not offered by this toolset
        again = await call(ToolServer(store, NoJobs(), toolset="refine-to-end"), "start_refinement",
                           {"run": run_id[:8], "step": 1, "target_match": "write_file"})
        assert refusal(again) and "target_check" in text(again)
        # The twin refusal says it too.
        unseen = await call(server, "start_refinement", {"run": run_id[:8], "step": 1, "target_match": "zzz_never"})
        assert refusal(unseen) and "passing runs' answers have that the failing runs' lack" in text(unseen)

    def test_a_target_the_failing_runs_satisfy_as_readily_as_the_passing_is_refused(self, tmp_path, mock_server, monkeypatch):
        """What the 26B declared next, on the real stack: "mentions Action Items", which 3 of 4
        passing runs' answers had and 4 of 5 failing runs' — screened 19/20 against 20/20."""
        from tuningfork.ops.batch import execute_batch, prepare_batch
        from tests.helpers import task_stack

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1",
                                 workdir=stack.tmp / "batches", model="m", harness="fake", n=2)
            failed, other = [outcome.run_id for outcome in execute_batch(stack.store, plan).outcomes]
            # Both wrote the file the same way; one is marked as having passed, so the target
            # counts a passing run's answer and a failing run's alike.
            stack.store.annotate_run(other, result={**stack.store.get_run(other).result, "passed": True})
            server = ToolServer(stack.store, NoJobs(), toolset="refine")
            target = {"run": failed[:8], "step": "write_file", "target_rules": [{"if": "calls", "value": "write_file"}]}
            tried = asyncio.run(call(server, "step_behaviours", target))
            assert "1 of 1 passing run(s) and 1 of 1 failing run(s)" in text(tried), text(tried)
            assert tried.structured_content["split"] == {
                "kind": "universal", "passing": {"counted": 1, "of": 1}, "failing": {"counted": 1, "of": 1},
                "separates": False}
            refused = asyncio.run(call(server, "start_refinement", target))
            assert refusal(refused) and refused.structured_content["error"]["code"] == "target_does_not_separate"
            assert "does not go with passing" in text(refused) and "would not move the verdict" in text(refused)
            assert "passing runs' answers have that the failing runs' lack" in text(refused)
            assert stack.store.experiments(limit=5) == []

    async def test_the_check_as_a_target_needs_the_toolset_that_screens_to_the_end(self, recorded):
        store, run_id = recorded
        refused = await call(ToolServer(store, NoJobs(), toolset="refine"), "start_refinement",
                             {"run": run_id[:8], "step": 1, "target_check": True})
        assert refusal(refused) and refused.structured_content["error"]["code"] == "not_offered"
        assert "refine-to-end" in text(refused)
        both = await call(ToolServer(store, NoJobs(), toolset="refine-to-end"), "start_refinement",
                          {"run": run_id[:8], "step": 1, "target_check": True, "target_match": "x"})
        assert refusal(both) and both.structured_content["error"]["field"] == "target"

    def test_the_tryout_and_the_check_as_a_target_on_a_task_with_a_check(self, tmp_path, mock_server, monkeypatch):
        from tuningfork.ops.batch import execute_batch, prepare_batch
        from tests.helpers import task_stack

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1",
                                 workdir=stack.tmp / "batches", model="m", harness="fake", n=1)
            run_id = execute_batch(stack.store, plan).outcomes[0].run_id
            server = ToolServer(stack.store, NoJobs(), toolset="refine-to-end")
            # The tryout: a target against the recorded answers, split by verdict.
            tried = asyncio.run(call(server, "step_behaviours", {
                "run": run_id[:8], "step": "write_file", "target_rules": [{"if": "calls", "value": "write_file"}]}))
            assert not tried.is_error, text(tried)
            assert "0 passed and 1 failed" in text(tried)
            assert "1 of 1 failing run(s): every recorded answer satisfies it" in text(tried)
            assert tried.structured_content["split"]["kind"] == "universal"
            assert tried.structured_content["behaviours"][0]["verdicts"] == {"failed": 1}
            unseen = asyncio.run(call(server, "step_behaviours", {
                "run": run_id[:8], "step": "write_file", "target_rules": [{"if": "contains", "value": "row_count"}]}))
            assert "no recorded answer does" in text(unseen)
            # Below five recorded answers, a target every one satisfies is let through: one says little.
            few = asyncio.run(call(server, "start_refinement", {
                "run": run_id[:8], "step": "write_file", "target_rules": [{"if": "calls", "value": "write_file"}]}))
            assert not few.is_error and "error" not in (few.structured_content or {}), text(few)
            started = asyncio.run(call(server, "start_refinement", {"run": run_id[:8], "step": "write_file", "target_check": True}))
            assert not started.is_error and "error" not in (started.structured_content or {}), text(started)
            assert "the task's check passes" in text(started) and "start_screen_to_end" in text(started)


class TestTheToolARunUsed:
    """The tool a run used, at whatever step, is a split no step index shows: a run that
    wrote with a script never called the write tool at all. find_divergence says when the
    tool goes with the verdict, and names the target and the step that follow from it;
    step_behaviours compares runs at each one's own such step when the step is named by it."""

    def runs(self, tmp_path):
        from tests.helpers import record_run, text_message, tool_call_message

        store = Store(tmp_path / "split")
        read = tool_call_message("read_file", {"path": "notes.md"})
        search = tool_call_message("search_files", {"pattern": "*.md"})
        write = tool_call_message("write_file", {"path": "out.md", "content": "x"})
        script = tool_call_message("execute_code", {"code": "write_file('out.md', 'x')"})
        paths = [
            ([read, write, text_message("done")], True),
            ([search, read, write, text_message("done")], True),
            ([read, write, text_message("done")], True),
            ([search, write, text_message("done")], True),
            ([read, script, text_message("done")], False),
            ([search, read, script, text_message("done")], False),
            ([read, script, script, text_message("done")], False),
        ]
        for index, (steps, passed) in enumerate(paths):
            run_id = record_run(store, f"w{index}", steps, task_id="notes/1")
            store.annotate_run(run_id, result={"passed": passed, "usable": True})
        return store

    def test_find_divergence_names_the_tool_that_goes_with_passing(self, tmp_path):
        from tuningfork.ops.inspect import tool_split

        store = self.runs(tmp_path)
        try:
            run_ids = [run.id for run in store.runs(limit=10, task_id="notes/1")]
            split = tool_split(store, run_ids)
            assert split["judged"] == 7
            by_name = {entry["tool"]: entry for entry in split["tools"]}
            assert by_name["write_file"]["called"] == {"passed": 4, "total": 4}
            assert by_name["write_file"]["never"] == {"passed": 0, "total": 3}
            assert by_name["write_file"]["with_passing"] is True and by_name["execute_code"]["with_passing"] is False
            # read_file: one run never called it, too few to compare the callers with.
            assert by_name["read_file"]["p"] is None and by_name["read_file"]["never"]["total"] == 1
            assert split["best"]["tool"] == "write_file" and split["best"]["p"] < 0.2

            result = asyncio.run(call(ToolServer(store, NoJobs(), toolset="refine"), "find_divergence", {"task": "notes/1"}))
            assert not refusal(result), text(result)
            assert "runs that called write_file passed 4/4, runs that never did 0/3" in text(result)
            assert 'target_rules [{"if": "calls", "value": "write_file"}]' in text(result)
            assert 'step "write_file"' in text(result) and "naming the tool to use" in text(result)
            assert result.structured_content["tool_split"]["tool"] == "write_file"
        finally:
            store.close()

    def test_step_behaviours_compares_runs_at_their_own_call_of_the_tool(self, tmp_path):
        store = self.runs(tmp_path)
        try:
            writer = next(run for run in store.runs(limit=10, task_id="notes/1") if run.label == "w1")
            server = ToolServer(store, NoJobs())
            by_tool = asyncio.run(call(server, "step_behaviours", {"run": writer.id[:8], "step": "write_file"}))
            assert "other runs compared at each one's own first call of write_file" in text(by_tool)
            assert "3 run(s) of the task never called one (3 failed)" in text(by_tool) and "joined by |" in text(by_tool)
            assert by_tool.structured_content["aligned"] == ["write_file"]
            assert by_tool.structured_content["not_aligned"] == {"failed": 3}
            either = asyncio.run(call(server, "step_behaviours", {
                "run": writer.id[:8], "step": "write_file|execute_code",
                "target_rules": [{"if": "calls", "value": "write_file"}]}))
            assert "it separates them, a target a screen can move" in text(either)
            assert either.structured_content["not_aligned"] is None
            by_index = asyncio.run(call(server, "step_behaviours", {"run": writer.id[:8], "step": 2}))
            assert "compared at the same step index" in text(by_index) and "write_file|execute_code" in text(by_index)
        finally:
            store.close()


class TestTheChanceOfEachTool:
    """step_behaviours says the model's own chance of each tool at the step where the
    recorded answers were scored, and how to get it where they were not."""

    def test_scored_or_not(self, tmp_path):
        from tests.helpers import record_run, text_message, tool_call_message
        from tests.test_tokens_ops import ANSWER, entry, stream_json

        store = Store(tmp_path / "chance")
        try:
            tools = ("read_file", "write_file", "execute_code")
            read = tool_call_message("read_file", {"path": "notes.md"})
            write = tool_call_message("write_file", {"path": "out.md", "content": "x"})
            script = tool_call_message("execute_code", {"code": "x"})
            writer = record_run(store, "writer", [read, write, text_message("done")], task_id="c/1", tools=tools)
            scripted = record_run(store, "scripted", [read, script, text_message("done")], task_id="c/1", tools=tools)
            for run_id, passed in ((writer, True), (scripted, False)):
                store.annotate_run(run_id, result={"passed": passed, "usable": True})
            server = ToolServer(store, NoJobs(), toolset="refine")
            unscored = asyncio.run(call(server, "step_behaviours", {"run": writer[:8], "step": "write_file|execute_code"}))
            assert "not scored at this step" in text(unscored) and "tuningfork tokens <run> <step> --score" in text(unscored)
            offered = ToolServer(store, NoJobs(), toolset="evidence")
            assert "start_tokens at this step scores it" in text(asyncio.run(call(offered, "step_behaviours", {"run": writer[:8], "step": "1"})))

            step = store.step(writer, 1)
            answer = [*ANSWER[:8], entry(7, "write", 0.4, [(12, "execute", 0.6)]), *ANSWER[9:]]
            store.add_token_stream(stream_json(answer), source="scored", step_id=step.id, method="grammar", exact=True)
            scored = asyncio.run(call(server, "step_behaviours", {"run": writer[:8], "step": "write_file|execute_code"}))
            assert ("the model's own chance of each tool at this step, from its scored tokens in 1 of the 2 recorded "
                    "run(s) compared: execute_code 60%, write_file 40% — a choice this open is one a few words of the "
                    "skill can move") in text(scored), text(scored)
            choices = scored.structured_content["choices"]
            assert choices["scored"] == 1 and choices["of"] == 2 and choices["runs"][0]["chosen"] == "write_file"
            # A reply in text offers no choice: nothing said.
            assert "chance of each tool" not in text(asyncio.run(call(server, "step_behaviours", {"run": writer[:8], "step": "2"})))
        finally:
            store.close()


def test_a_long_text_is_read_page_by_page():
    """A file or a message longer than a page comes a page at a time, each saying where the next
    starts; one whose escaping would overflow an answer comes in smaller pages, never cut to a
    few hundred characters by the answer's fit (findings §31.57)."""
    from tuningfork.mcp.tools import FILE_PAGE, page
    from tuningfork.ops import compact

    text = "".join(f"{i:05d}\n" for i in range(4000))  # 24,000 characters
    shown, where, words = page(text, 0)
    assert shown == text[:FILE_PAGE] and where == {"offset": 0, "to": FILE_PAGE, "chars": len(text), "next": FILE_PAGE}
    assert f"the next page: offset={FILE_PAGE}" in words
    pages, offset = [], 0
    while offset is not None:
        shown, where, _ = page(text, offset)
        pages.append(shown)
        offset = where["next"]
    assert "".join(pages) == text and len(pages) == 3
    wide = "—" * 30_000  # six characters each in JSON as compact measures it
    shown, where, _ = page(wide, 0, {"path": "x"})
    assert 500 <= len(shown) < FILE_PAGE and compact.size({"path": "x", "text": shown, "page": where}) <= compact.BUDGET
    with pytest.raises(OperationError) as past:
        page(text, len(text) + 1)
    assert past.value.field == "offset"


# --- what an agent is shown --------------------------------------------------------------------


#: The skill a run was given, holding a key, and a private key's lines: an agent's model may not
#: be shown them.
KEYED = (SKILL + f"\nLog in with {KEY}.\nUse password={PASSWORD} first.\nIf asked again, answer {PASSWORD}.\n\n"
         + "\n".join(PRIVATE) + "\n\nThen write the report.\n")


class TestWhatAnAgentIsShown:
    """Every answer an agent is given over MCP, and every refusal, is masked as it leaves the server
    (`ToolServer.shown`): an agent's model may be hosted, and what the store keeps as given — a
    task's check, a workspace's files, the skill a run was given — goes no further unmasked."""

    async def test_an_answer_s_text_and_json_are_masked_alike(self, recorded):
        store, _ = recorded
        server = ToolServer(store, NoJobs())
        server.tools["list_tasks"] = dataclasses.replace(server.tools["list_tasks"], read=lambda context, args: Answer(
            {"check": f"curl -H 'Authorization: Bearer {KEY}' https://api.test", "said": NAMED},
            [f"its check sends {KEY}", f"and the password, {PASSWORD}"]))
        result = await call(server, "list_tasks")
        said = text(result) + json.dumps(result.structured_content)
        assert KEY not in said and PASSWORD not in said and REDACTED in said

    async def test_a_refusal_is_masked_where_it_quotes_what_it_was_asked(self, recorded):
        store, _ = recorded
        result = await call(ToolServer(store, NoJobs()), "show_run", {"run": KEY})
        assert refusal(result) and REDACTED in text(result)
        assert KEY not in text(result) + json.dumps(result.structured_content)

    async def test_the_skill_a_run_was_given_is_shown_masked_each_line_at_its_number(self, store, tmp_path):
        """Masked whole before its lines are numbered: line by line, a private key's lines pass. An
        edit still names a line by the number it is shown at, and is resolved against the file."""
        from tuningfork.ops import refine

        run_id = record_pilot(store, skill=KEYED)
        frozen = tmp_path / "frozen" / "tally-report"
        frozen.mkdir(parents=True)
        (frozen / "SKILL.md").write_text(KEYED, encoding="utf-8")
        store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
        made = refine.create(store, run_id, "1", target_match="row_count")
        server = ToolServer(store, NoJobs(), toolset="refine")
        lines = KEYED.splitlines()
        keyed, after = lines.index(f"Log in with {KEY}.") + 1, lines.index("Then write the report.") + 1
        shown = (await call(server, "show_skill", {"run": run_id})).structured_content["lines"]
        assert shown[keyed - 1] == f"{keyed}: Log in with {REDACTED}."
        assert shown[after - 1] == f"{after}: Then write the report."
        assert not any(PRIVATE[1] in line or PRIVATE[2] in line for line in shown)
        # That line rewritten as it was shown would write the mask into the skill: refused.
        copied = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"line": keyed, "new": f"Log in with {REDACTED}, once."}]})
        assert refusal(copied) and f"would write {REDACTED} into the skill" in text(copied)
        # Rewritten whole, it is resolved against the file, and its answer quotes the line masked.
        rewritten = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"line": keyed, "new": "Log in with the key in the environment."}]})
        assert not refusal(rewritten) and f"'Log in with {REDACTED}.' →" in text(rewritten)
        assert KEY not in text(rewritten) + json.dumps(rewritten.structured_content)
        (proposal,) = refine.proposals(store, made.id)
        assert PRIVATE[1] in proposal.text() and KEY not in proposal.text()  # the file's, but for the edit
        # The private key's lines taken out: each is a line the proposal changed, as the refinement
        # shows them — from the two texts masked whole and alike, never one line at a time.
        first, last = lines.index(PRIVATE[0]) + 1, lines.index(PRIVATE[-1]) + 1
        removed = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"line": first, "through": last, "new": "Keep keys out of the skill."}]})
        assert not refusal(removed) and PRIVATE[1] not in text(removed), text(removed)
        # The private key's first line taken away, or a line put after it: what is left of it reads as
        # no key on its own, and is shown as the skill shows it (found in review).
        for edit in ({"line": first, "new": "The deploy key is kept in the vault."},
                     {"after": first, "insert": "(the deploy key)"}):
            assert not refusal(await call(server, "propose_edit", {"refinement": made.id, "rationale": "r",
                                                                   "edits": [edit]}))
        # A line saying bare a password another line names is quoted as the skill shows it, and so are
        # the lines a refusal finds closest (found in review).
        bare = lines.index(f"If asked again, answer {PASSWORD}.") + 1
        quoted = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"line": bare, "new": "If asked again, answer it."}]})
        assert f"'If asked again, answer {REDACTED}.' →" in text(quoted)
        missed = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"old": "If asked again, answer it.", "new": "x"}]})
        assert refusal(missed) and f"answer {REDACTED}." in text(missed)
        refined = await call(server, "show_refinement", {"refinement": made.id, "evidence": True})
        said = text(refined) + json.dumps(refined.structured_content)
        assert "Keep keys out of the skill." in said and "tried_before" in said
        for each in (refined, quoted, missed, removed):
            said = text(each) + json.dumps(each.structured_content)
            assert KEY not in said and PASSWORD not in said and PRIVATE[1] not in said and PRIVATE[2] not in said

    async def test_what_a_tool_cuts_short_is_masked_whole_before_it_is_cut(self, store, tmp_path):
        """A tool fits a long answer to its budget, cutting its texts: what is kept as given — a
        scenario's spec, the line an edit replaced — is masked before, so a cut through a key leaves
        none of it, where the part left would read as no key."""
        from tuningfork.ops import refine, scenarios

        spec = {"skill_name": "tally-report", "evals": [{"id": "1", "prompt": "go", "expected_output": "y" * 20_000,
                                                         "tuningfork": {"check": "x" * 389 + " " + KEY + " && true"}}]}
        place = scenarios.folder(store.home, "long")
        place.mkdir(parents=True)
        (place / scenarios.SPEC).write_text(json.dumps(spec), encoding="utf-8")
        server = ToolServer(store, NoJobs(), toolset="all")
        result = await call(server, "list_scenarios", {"name": "long"})
        said = text(result) + json.dumps(result.structured_content)
        assert "x" * 389 in said and KEY[:10] not in said  # cut where the key stood, and none of it left

        long = "x" * 109 + " " + KEY + " and then the totals."
        run_id = record_pilot(store, skill=SKILL + f"\n{long}\n")
        frozen = tmp_path / "frozen" / "tally-report"
        frozen.mkdir(parents=True)
        (frozen / "SKILL.md").write_text(SKILL + f"\n{long}\n", encoding="utf-8")
        store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
        made = refine.create(store, run_id, "1", target_match="row_count")
        line = (SKILL + f"\n{long}\n").splitlines().index(long) + 1
        rewritten = await call(server, "propose_edit", {"refinement": made.id, "rationale": "r", "edits": [
            {"line": line, "new": "Then the totals."}]})
        said = text(rewritten) + json.dumps(rewritten.structured_content)
        assert not refusal(rewritten) and "x" * 109 in said and KEY[:10] not in said

    def test_a_page_cut_through_a_key_leaves_none_of_it(self, tmp_path, mock_server, monkeypatch):
        """A file, and what a run changed, are read a page at a time: each is masked whole before it is
        paged, so a page's edge through a key leaves none of it, where the part left would read as no
        key. A sync test: the task fixture's proxy closes its client with asyncio.run."""
        import re

        from tuningfork.ops.batch import execute_batch, prepare_batch
        from tuningfork.ops.inspect import workspace_at, workspace_patch
        from tests.helpers import task_stack

        notes = "Totals checked by hand.\n" * 3 + f"Log in with {KEY}.\n" + "\n".join(PRIVATE) + "\nDone.\n"
        # A file the run started with and rewrote: in a diff, each of its lines removed reads after a
        # `-`, where a key's pattern no longer reads it (found in review).
        deploy = "\n".join(PRIVATE) + f"\n{KEY}\n"

        def writes(payload: dict) -> MockTurn:
            messages = payload["messages"]
            workdir = re.search(r"Working directory: (\S+)", messages[0]["content"]).group(1)
            if not any(message["role"] == "tool" for message in messages):
                return MockTurn(tool_calls=[
                    {"name": "write_file", "arguments": {"path": f"{workdir}/notes.txt", "content": notes}},
                    {"name": "write_file", "arguments": {"path": f"{workdir}/deploy.pem", "content": "rotated\n"}}])
            return MockTurn(content="Noted.")

        with task_stack(tmp_path, mock_server, monkeypatch) as stack:
            stack.item.files.append({"path": "deploy.pem", "content": deploy})
            stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
            mock_server.responder = writes
            plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1",
                                 workdir=stack.tmp / "batches", model="m", harness="fake", n=1)
            run_id = execute_batch(stack.store, plan).outcomes[0].run_id
            server = ToolServer(stack.store, NoJobs(), toolset="refine")
            patch = workspace_patch(stack.store, workspace_at(stack.store, run_id, "end"), "notes.txt")
            for arguments, whole in (({"path": "notes.txt"}, notes), ({"path": "notes.txt", "diff": True}, patch)):
                # The page's edge ten characters into the key, as the file and the diff stand.
                monkeypatch.setattr("tuningfork.mcp.tools.FILE_PAGE", whole.index(KEY) + 10)
                first = asyncio.run(call(server, "show_files", {"run": run_id[:8], **arguments}))
                said = text(first) + json.dumps(first.structured_content)
                assert not refusal(first) and "Totals checked by hand." in said and KEY[:10] not in said, arguments
            monkeypatch.setattr("tuningfork.mcp.tools.FILE_PAGE", 8000)
            # A whole file keeps each line at its number; a diff of the whole run shows no line of a key.
            read = asyncio.run(call(server, "show_files", {"run": run_id[:8], "path": "notes.txt"}))
            assert len(read.structured_content["text"].splitlines()) == len(notes.splitlines())
            changed = asyncio.run(call(server, "show_files", {"run": run_id[:8], "diff": True}))
            said = text(changed) + json.dumps(changed.structured_content)
            assert "deploy.pem" in said and "+rotated" in said and "Totals checked by hand." in said
            assert KEY[:10] not in said and PRIVATE[1] not in said and PRIVATE[2] not in said

    def test_an_ab_s_wording_read_from_its_files_is_shown_masked_whole(self, tmp_path, monkeypatch, mock_server):
        """Where an A/B's texts are read from the files it names, kept as given, an agent is shown its
        edit from both masked whole: line by line, a private key's lines pass."""
        from tests.test_reports import ab_home, ab_ids

        variant = VARIANT + f"\nLog in with {KEY}.\n\n" + "\n".join(PRIVATE) + "\n"
        home = ab_home(tmp_path, monkeypatch, mock_server, variant)
        arms = ab_ids(home)
        store = Store(home)
        try:
            with store._lock:
                store._conn.execute("DELETE FROM samples WHERE experiment_id = ?", (arms["variant"],))
                store._conn.execute("DELETE FROM experiments WHERE id = ?", (arms["variant"],))
                store._conn.commit()
            shown = asyncio.run(call(ToolServer(store, NoJobs()), "show_experiment", {"experiment": arms["original"]}))
            said = text(shown) + json.dumps(shown.structured_content)
            assert "Log in with" in said and KEY not in said and PRIVATE[1] not in said and PRIVATE[2] not in said
        finally:
            store.close()

    async def test_a_dry_run_s_plan_is_masked_whole_before_it_is_fitted(self, lab, monkeypatch):
        """A plan is not kept, so nothing masked it: an agent is shown it masked whole, before it is
        fitted to the answer, which can cut a key apart."""
        server, runner = lab
        monkeypatch.setattr(runner, "plan", lambda kind, params: {"total": 3, "plan": {
            "spec": {"check": "x" * 389 + " " + KEY + " && true", "expected_output": "y" * 20_000}}})
        result = await call(server, "start_count", {"steps": 3, "dry_run": True})
        said = text(result) + json.dumps(result.structured_content)
        assert "count would make 3" in said and "x" * 389 in said and KEY[:10] not in said

    async def test_a_proposal_s_rationale_is_masked_whole_before_it_is_clipped(self, store, tmp_path):
        from tuningfork.ops import refine

        run_id = record_pilot(store)
        frozen = tmp_path / "frozen" / "tally-report"
        frozen.mkdir(parents=True)
        (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
        store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
        made = refine.create(store, run_id, "1", target_match="row_count")
        refine.add_proposal(store, made.id, edits=[{"line": 7, "new": "Write the totals to report.json, checked."}],
                            rationale="r" * 289 + " " + KEY + " was in the way.", proposer={"via": "person"})
        shown = await call(ToolServer(store, NoJobs(), toolset="refine"), "show_refinement", {"refinement": made.id})
        said = text(shown) + json.dumps(shown.structured_content)
        assert "r" * 289 in said and KEY[:10] not in said
