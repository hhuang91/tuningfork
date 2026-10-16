"""`tuningfork mcp` end to end: a real subprocess on stdio, driven by the SDK's client and by
a 2025 client written by hand, reading recordings and starting work on a real proxy.

The token comes from the file the proxy writes, as for the CLI: nothing secret is passed.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import anyio
import httpx
import pytest

pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from tuningfork.config import Settings  # noqa: E402
from tuningfork.proxy.app import create_app  # noqa: E402
from tuningfork.store.db import Store  # noqa: E402
from tests.helpers import PROPS, free_port, record_pilot, serving  # noqa: E402
from tests.mock_server import MockTurn  # noqa: E402

AGENT = types.Implementation(name="stdio-test", version="1")


def write(keys: dict) -> MockTurn:
    return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}])


@pytest.fixture
def proxy(tmp_path, mock_server):
    mock_server.props = PROPS
    # Names the report's keys only when the skill in its context does.
    mock_server.responder = lambda payload: write(
        {"row_count": 4} if "row_count and amount_total" in payload["messages"][0]["content"] else {"total": 61.5}
    )
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    app = create_app(
        Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home", max_inflight=2), client=upstream
    )
    store = app.state.recorder.store
    run_id = record_pilot(store)
    try:
        with serving(app) as url:
            yield url, tmp_path / "home", run_id
    finally:
        store.close()
        asyncio.run(upstream.aclose())


def environment() -> dict[str, str]:
    """What the client passes the server: no token — it reads the proxy's file."""
    return {key: value for key, value in os.environ.items() if key != "TUNINGFORK_TOKEN"}


def server(home, proxy_url: str, *extra: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-W", "error", "-m", "tuningfork", "mcp", "--home", str(home), "--proxy", proxy_url, *extra],
        env=environment(),
    )


@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_reads_and_a_resample_over_stdio(proxy, mode):
    url, home, run_id = proxy
    with anyio.fail_after(60):
        async with Client(server(home, url), mode=mode, client_info=AGENT) as client:
            runs = await client.call_tool("list_runs", {})
            assert runs.structured_content["runs"][0]["id"] == run_id[:8]
            started = await client.call_tool(
                "start_resample", {"run": run_id[:8], "step": 1, "n": 4, "wait_seconds": 30}
            )
            assert not started.is_error, started.content[0].text
            text = started.content[0].text
            assert "4 samples:" in text and "4/4 (100%, 95% CI 51–100%)" in text
    store = Store(home)
    try:
        (job,) = store.jobs()
        assert job.started_by == {"via": "mcp", "client": "stdio-test", "version": "1"}
    finally:
        store.close()


async def test_with_no_proxy_reads_answer_and_work_says_what_to_do(proxy):
    _, home, run_id = proxy
    nobody = f"http://127.0.0.1:{free_port()}"
    with anyio.fail_after(60):
        async with Client(server(home, nobody), client_info=AGENT) as client:
            step = await client.call_tool("show_step", {"run": run_id[:8], "step": 1})
            assert not step.is_error
            started = await client.call_tool("start_resample", {"run": run_id[:8], "step": 1, "n": 4})
            assert not started.is_error and "error" in started.structured_content  # a refusal, answered
            assert "tuningfork serve --home" in started.content[0].text


async def test_a_proxy_recording_elsewhere_is_refused(proxy, tmp_path):
    url, _, run_id = proxy
    other = tmp_path / "other"
    store = Store(other)
    record_pilot(store)
    store.close()
    with anyio.fail_after(60):
        async with Client(server(other, url), client_info=AGENT) as client:
            started = await client.call_tool("start_resample", {"run": "pilot", "step": 1, "n": 4})
            assert not started.is_error and "records into" in started.content[0].text  # a refusal, answered


def test_a_2025_client_that_reads_only_text_gets_counts_and_intervals(proxy):
    """What OpenJarvis's client does: initialize at 2025-03-26, read `content`, ignore the rest."""
    url, home, run_id = proxy
    process = subprocess.Popen(
        [sys.executable, "-W", "error", "-m", "tuningfork", "mcp", "--home", str(home), "--proxy", url],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment(),
    )
    try:
        def send(message: dict) -> None:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        def answer(request_id: int) -> dict:
            while True:
                message = json.loads(process.stdout.readline())
                if message.get("id") == request_id:
                    return message

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "jarvis", "version": "0"}}})
        assert answer(1)["result"]["protocolVersion"] == "2025-03-26"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "start_resample", "arguments": {"run": run_id[:8], "step": "1", "n": 4, "wait_seconds": 30}}})
        result = answer(2)["result"]
        text = result["content"][0]["text"]
        assert "4/4 (100%, 95% CI 51–100%)" in text
        assert json.loads(text.split("\n\n", 1)[1])["status"] == "done"  # the data, after the sentences
    finally:
        process.stdin.close()
        code = process.wait(timeout=30)
        stderr = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert code == 0, stderr  # stdin closed: the server ends, cleanly
    assert "Traceback" not in stderr


def test_without_recordings_it_refuses_and_says_where(tmp_path):
    found = subprocess.run(
        [sys.executable, "-m", "tuningfork", "mcp", "--home", str(tmp_path / "nothing"),
         "--proxy", f"http://127.0.0.1:{free_port()}"],
        capture_output=True, text=True, timeout=60, env=environment(), stdin=subprocess.DEVNULL,
    )
    assert found.returncode == 1
    assert found.stdout == ""  # stdout is the protocol's, even when it refuses
    assert "no recordings at" in found.stderr and "--home" in found.stderr
    assert not (tmp_path / "nothing").exists()


async def test_the_refinement_loop_over_stdio(proxy, tmp_path):
    """start_refinement → show_refinement → propose_edit → start_screen, with --tools refine."""
    from tests.helpers import SKILL

    app_url, home, run_id = proxy
    # The pilot run as a batch would have left it: its skill's frozen copy beside it.
    frozen = tmp_path / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    store = Store(home)
    try:
        run = store.get_run(run_id)
        store.annotate_run(run_id, parity={**run.parity, "skill_snapshot_dir": str(frozen)})
    finally:
        store.close()
    rules = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
    edit = {"old": "Write the totals to report.json.",
            "new": "Write the totals to report.json, using the keys row_count and amount_total."}
    with anyio.fail_after(90):
        async with Client(server(home, app_url, "--tools", "refine"), client_info=AGENT) as client:
            # The fixture's one recorded answer does none of this: the target is new behaviour, said so.
            started = await client.call_tool("start_refinement",
                                              {"run": run_id[:8], "step": 1, "target_rules": rules, "unseen_ok": True})
            assert not started.is_error, started.content[0].text
            rid = started.structured_content["refinement"]
            shown = await client.call_tool("show_refinement", {"refinement": rid})
            assert shown.structured_content["packet"]["answers_here"]["reached"] == 0
            proposed = await client.call_tool("propose_edit", {"refinement": rid, "edits": [edit],
                                                               "rationale": "the check reads row_count"})
            assert not proposed.is_error, proposed.content[0].text
            pid = proposed.structured_content["proposal"]
            screened = await client.call_tool("start_screen", {"refinement": rid, "proposals": [pid], "n": 4,
                                                               "wait_seconds": 60})
            text = screened.content[0].text
            assert not screened.is_error, text
            assert f"{pid} 4/4" in text and "survives" in text, text
            after = await client.call_tool("show_refinement", {"refinement": rid, "evidence": False})
            (proposal,) = after.structured_content["proposals"]
            assert proposal["screens"][0]["survives"] is True
            assert proposal["proposers"] == [{"via": "mcp", "client": "stdio-test"}]
