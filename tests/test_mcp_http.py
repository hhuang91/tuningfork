"""MCP over Streamable HTTP, on the proxy: behind the guard on every method, both protocol
eras, jobs sent through the proxy itself, and a stop that never waits on a connected client."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import anyio
import httpx
import pytest

pytest.importorskip("mcp")

import httpx2  # noqa: E402
import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from tuningfork.access import HEADER  # noqa: E402
from tuningfork.config import Settings  # noqa: E402
from tuningfork.proxy.app import create_app, stop_live_work  # noqa: E402
from tests.helpers import PROPS, record_pilot, serving  # noqa: E402
from tests.mock_server import MockTurn  # noqa: E402

AGENT = types.Implementation(name="http-test", version="2")
MCP = "/_tuningfork/mcp"


def write(keys: dict) -> MockTurn:
    return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}])


@pytest.fixture
def proxy(tmp_path, mock_server):
    # sse-starlette, under the SDK, patches uvicorn's exit handler to mark the whole process
    # as stopping, for good: once any test has stopped a server (test_live_server does, as
    # Ctrl-C would), every SSE stream after it would end at once.
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit = False
    mock_server.props = PROPS
    mock_server.responder = lambda payload: write({"total": 61.5})
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    app = create_app(
        Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home", max_inflight=2), client=upstream
    )
    store = app.state.recorder.store
    run_id = record_pilot(store)
    try:
        with serving(app) as url:
            yield app, url, run_id
    finally:
        store.close()
        asyncio.run(upstream.aclose())


def connect(url: str, headers: dict[str, str], mode: str = "auto"):
    http = httpx2.AsyncClient(headers=headers, timeout=60)
    return http, Client(streamable_http_client(f"{url}{MCP}", http_client=http), mode=mode, client_info=AGENT)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_both_eras_read_and_resample_through_the_proxy(proxy, mode):
    app, url, run_id = proxy
    http, client = connect(url, {HEADER: app.state.access.mcp_token}, mode)
    with anyio.fail_after(60):
        async with http, client:
            runs = await client.call_tool("list_runs", {})
            assert runs.structured_content["runs"][0]["id"] == run_id[:8]
            started = await client.call_tool("start_resample", {"run": run_id[:8], "step": 1, "n": 4, "wait_seconds": 30})
            assert "4/4 (100%, 95% CI 51–100%)" in started.content[0].text, started.content[0].text
    (job,) = app.state.recorder.store.jobs()
    assert job.started_by == {"via": "mcp", "client": "http-test", "version": "2"}
    experiment = app.state.recorder.store.get_experiment(job.links["experiments"][0])
    assert experiment.base_url == url  # its model calls went through the proxy's own gate


async def test_a_bearer_token_is_the_token_too(proxy):
    app, url, _ = proxy
    http, client = connect(url, {"authorization": f"Bearer {app.state.access.mcp_token}"})
    with anyio.fail_after(30):
        async with http, client:
            tasks = await client.call_tool("list_tasks", {})
            assert not tasks.is_error


def test_without_the_token_nothing_is_answered_not_even_a_read(proxy):
    app, url, _ = proxy
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    accept = {"accept": "application/json, text/event-stream"}
    for method in ("GET", "POST", "DELETE"):
        refused = httpx.request(method, f"{url}{MCP}", json=body if method == "POST" else None, headers=accept)
        assert refused.status_code == 401, method
        assert refused.json()["error"]["code"] == "token_missing"
    wrong = httpx.post(f"{url}{MCP}", json=body, headers={**accept, "authorization": "Bearer nope"})
    assert wrong.status_code == 403
    token = {HEADER: app.state.access.mcp_token, **accept}
    assert httpx.post(f"{url}{MCP}", json=body, headers={**token, "host": "evil.example"}).status_code == 403
    assert httpx.post(f"{url}{MCP}", json=body, headers={**token, "origin": "http://evil.example"}).status_code == 403


def test_each_token_opens_its_own_door_and_no_other(proxy):
    """An MCP client holds the MCP token: with it, the tools — and never what they leave out,
    such as approving an edit. The access token, the person's, is no key to the tools."""
    app, url, _ = proxy
    access = app.state.access
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    accept = {"accept": "application/json, text/event-stream"}
    person = httpx.post(f"{url}{MCP}", json=body, headers={**accept, "authorization": f"Bearer {access.token}"})
    assert person.status_code == 403 and "the access token" in person.json()["error"]["message"]
    assert "tuningfork token --mcp" in person.json()["error"]["hint"]
    agent = httpx.post(f"{url}/_tuningfork/api/pause", headers={"authorization": f"Bearer {access.mcp_token}"})
    assert agent.status_code == 403 and "the MCP token" in agent.json()["error"]["message"]
    assert not app.state.gate.paused
    assert access.mcp_token != access.token and access.mcp_file.read_text().strip() == access.mcp_token
    health = httpx.get(f"{url}/_tuningfork/health").json()
    assert health["mcp_token_file"] == str(access.mcp_file) and access.mcp_token not in json.dumps(health)


def test_stopping_does_not_wait_on_a_connected_client(proxy):
    """A 2025 session keeps a stream open for the server's messages; stopping ends it."""
    app, url, _ = proxy
    token = app.state.access.mcp_token
    headers = {HEADER: token, "accept": "application/json, text/event-stream"}
    opened = httpx.post(f"{url}{MCP}", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "raw", "version": "0"}}})
    session = {"mcp-session-id": opened.headers["mcp-session-id"], "mcp-protocol-version": "2025-06-18"}
    httpx.post(f"{url}{MCP}", headers={**headers, **session}, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    ended: list[float] = []

    def listen() -> None:
        started = time.monotonic()
        with httpx.stream("GET", f"{url}{MCP}", headers={HEADER: token, "accept": "text/event-stream", **session},
                          timeout=30) as stream:
            assert stream.status_code == 200
            for _ in stream.iter_bytes():
                pass
        ended.append(time.monotonic() - started)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    time.sleep(0.5)
    loop = app.state.mcp._loop
    loop.call_soon_threadsafe(stop_live_work, app)
    listener.join(5)
    assert ended and ended[0] < 5, "the stream outlived the stop"
    again = httpx.post(f"{url}{MCP}", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert again.status_code == 503


def test_the_route_waits_as_long_as_the_proxy_was_told(tmp_path):
    """How long one call may wait for a job is the proxy's setting (`serve --mcp-max-wait`):
    50 s unless told, for a client that reads an answer for 60 s; a client that gives a call
    300 s (Hermes) may be given about 110, so a twenty-minute screen is a dozen polls."""
    from tuningfork.config import Settings
    from tuningfork.proxy.app import create_app

    default = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "a"))
    patient = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "b", mcp_max_wait=110))
    try:
        assert default.state.mcp.max_wait == 50 and patient.state.mcp.max_wait == 110
    finally:
        default.state.recorder.store.close()
        patient.state.recorder.store.close()
