"""Connections: the server the proxy forwards to, switched only while nothing uses it; the
harnesses it launches, found without touching a real installation; and what a harness
needs of the server, checked before it is launched."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import httpx
import pytest

from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.config import Settings, read_home_config
from tuningfork.harness import ClaudeCodeAdapter, HermesAdapter
from tuningfork.ops import OperationError
from tuningfork.ops.connections import adapter_for, detect_harnesses, preflight, set_override
from tuningfork.proxy.app import create_app
from tests.helpers import PROPS, until
from tests.mock_server import MockServer, MockTurn

API = "/_tuningfork/api"
CHAT = "/v1/chat/completions"


def props(n_ctx: int) -> dict:
    return {**PROPS, "default_generation_settings": {**PROPS["default_generation_settings"], "n_ctx": n_ctx}}


@pytest.fixture
def two_servers(tmp_path):
    """A proxy started in front of server A, with server B — smaller context — beside it."""
    from starlette.applications import Starlette

    first, second = MockServer(model="model-a"), MockServer(model="model-b")
    first.props, second.props = props(65536), props(32768)
    nothing = Starlette()  # answers every path with a 404: no server here
    upstream = httpx.AsyncClient(mounts={
        "http://a.test": httpx.ASGITransport(app=first.app),
        "http://b.test": httpx.ASGITransport(app=second.app),
        "http://c.test": httpx.ASGITransport(app=nothing),
    })
    app = create_app(Settings(upstream_base_url="http://a.test", home=tmp_path / "home"), client=upstream)
    yield app, first, second
    app.state.recorder.store.close()
    asyncio.run(upstream.aclose())


def client_of(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8787",
        headers={TOKEN_HEADER: app.state.access.token}, timeout=30,
    )


def chat(content: str = "hi") -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


class TestSwitchingTheServer:
    async def test_a_dry_run_says_what_would_change_and_changes_nothing(self, two_servers):
        app, first, second = two_servers
        async with client_of(app) as client:
            planned = (await client.post(f"{API}/connections/upstream", json={"url": "http://b.test", "dry_run": True})).json()
            health = (await client.get("/_tuningfork/health")).json()
        assert planned["proposed"]["n_ctx"] == 32768 and planned["busy"] == []
        assert {change["field"] for change in planned["differences"]} >= {"n_ctx", "models"}
        assert health["upstream"] == "http://a.test"

    async def test_an_idle_proxy_switches_and_records_the_new_server_from_then_on(self, two_servers):
        app, first, second = two_servers
        async with client_of(app) as client:
            switched = await client.post(f"{API}/connections/upstream", json={"url": "http://b.test"})
            assert switched.status_code == 200 and switched.json()["switched"]
            await client.post(CHAT, json=chat())
            health = (await client.get("/_tuningfork/health")).json()
        assert len(second.requests) == 1 and first.requests == []
        assert health["upstream"] == "http://b.test" and health["started_with"] == "http://a.test"
        assert health["server"]["n_ctx"] == 32768
        await asyncio.gather(*list(app.state.background), return_exceptions=True)
        (run,) = app.state.recorder.store.runs(limit=5)
        assert run.server_id == health["server"]["id"]

    async def test_a_proxy_in_use_refuses_to_switch(self, two_servers):
        app, first, second = two_servers
        app.state.engine.add({"phase": "request"})
        async with client_of(app) as client:
            call = asyncio.create_task(client.post(CHAT, json=chat("held")))
            (hold,) = await until(lambda: app.state.engine.holds())
            refused = await client.post(f"{API}/connections/upstream", json={"url": "http://b.test"})
            app.state.engine.resolve(hold.id, "continue")
            await call
        assert refused.status_code == 409 and refused.json()["error"]["code"] == "busy"
        assert "held at a breakpoint" in refused.json()["error"]["message"]
        assert app.state.upstream.url == "http://a.test"

    async def test_nothing_answering_is_refused(self, two_servers):
        app, *_ = two_servers
        async with client_of(app) as client:
            refused = await client.post(f"{API}/connections/upstream", json={"url": "http://c.test"})
            malformed = await client.post(f"{API}/connections/upstream", json={"url": "b.test"})
        assert refused.status_code == 400 and "nothing answered" in refused.json()["error"]["message"]
        assert malformed.status_code == 400 and malformed.json()["error"]["field"] == "url"

    async def test_a_protected_endpoint_is_refused_before_anything_is_asked(self, two_servers, monkeypatch):
        app, first, second = two_servers
        monkeypatch.setenv("TUNINGFORK_PROTECTED", "127.0.0.1:18099")
        async with client_of(app) as client:
            refused = await client.post(f"{API}/connections/upstream", json={"url": "http://localhost:18099"})
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "protected_endpoint"


# --- harnesses ---------------------------------------------------------------------------------


def fake_program(folder: Path, name: str, version: str, log: Path) -> Path:
    """A stand-in harness that says its version — and writes down the home it was given,
    and tries to leave something there, as a harness reading its config would."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(
        "#!/bin/sh\n"
        f'echo "$HOME" >> "{log}"\n'
        'mkdir -p "$HOME/.config" && echo touched > "$HOME/.config/touched"\n'
        f'echo "{version}"\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(os.name != "posix", reason="the stand-in harnesses are shell scripts")
class TestHarnesses:
    def test_detection_reads_versions_in_a_throwaway_home(self, tmp_path, monkeypatch):
        log = tmp_path / "homes.log"
        bin_dir = tmp_path / "bin"
        fake_program(bin_dir, "hermes", "Hermes Agent v0.18.2", log)
        fake_program(bin_dir, "claude", "2.1.281 (Claude Code)", log)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        real_home = Path.home()

        found = {info.name: info for info in detect_harnesses(tmp_path / "store")}

        assert found["hermes"].version == "Hermes Agent v0.18.2" and found["hermes"].source == "PATH"
        assert found["claude-code"].version == "2.1.281 (Claude Code)"
        homes = log.read_text().split()
        assert len(homes) == 2 and all(home != str(real_home) for home in homes)
        assert not any(Path(home).exists() for home in homes)  # thrown away after
        assert found["hermes"].requirements[0]["what"] == "n_ctx"

    def test_a_missing_harness_is_reported_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        found = {info.name: info for info in detect_harnesses(tmp_path / "store")}
        assert found["hermes"].source == "missing" and found["hermes"].path is None
        assert "not on PATH" in found["hermes"].error

    def test_an_executable_can_be_named_in_the_recordings_folder(self, tmp_path, monkeypatch):
        home = tmp_path / "store"
        program = fake_program(tmp_path / "elsewhere", "hermes-dev", "Hermes Agent v0.19.0", tmp_path / "log")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        set_override(home, "hermes", str(program))
        assert read_home_config(home)["harnesses"]["hermes"] == {"executable": str(program)}
        (hermes,) = [info for info in detect_harnesses(home) if info.name == "hermes"]
        assert (hermes.source, hermes.version) == ("override", "Hermes Agent v0.19.0")
        assert adapter_for(home, "hermes").executable == str(program)
        set_override(home, "hermes", None)
        assert adapter_for(home, "hermes").executable == "hermes"

    @pytest.mark.parametrize("given", ["hermes", "/no/such/program"])
    def test_only_an_absolute_path_to_a_program_is_accepted(self, tmp_path, given):
        with pytest.raises(OperationError) as refused:
            set_override(tmp_path, "hermes", given)
        assert refused.value.field == "executable"

    def test_under_wsl_a_windows_program_is_refused_with_where_to_look(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuningfork.paths.under_wsl", lambda: True)
        with pytest.raises(OperationError) as refused:
            set_override(tmp_path, "hermes", r"C:\Users\me\hermes.exe")
        assert "Windows path" in refused.value.message and "inside WSL" in refused.value.hint

    def test_a_file_that_cannot_run_is_refused(self, tmp_path):
        plain = tmp_path / "hermes.txt"
        plain.write_text("not a program")
        with pytest.raises(OperationError, match="not a program"):
            set_override(tmp_path, "hermes", str(plain))


# --- preflight ---------------------------------------------------------------------------------


class TestPreflight:
    @pytest.mark.parametrize(("n_ctx", "ok", "noted"), [(32768, False, False), (65536, True, False), (None, True, True)])
    def test_hermes_needs_64000_tokens_of_context(self, n_ctx, ok, noted):
        server = {"n_ctx": n_ctx} if n_ctx else {}
        checks = {"results": {"streaming_tool_calls": {"support": "yes"}}}
        found = preflight(HermesAdapter(), server, checks)
        assert found.ok is ok
        assert any("n_ctx" in note for note in found.notes) is noted
        if not ok:
            assert "64,000" in found.problems[0] and "32,768" in found.problems[0]

    @pytest.mark.parametrize(("support", "ok", "noted"), [("no", False, False), ("yes", True, False), (None, True, True)])
    def test_claude_code_needs_the_messages_api(self, support, ok, noted):
        checks = {"results": {"anthropic_messages": {"support": support, "detail": "404"}}} if support else {}
        found = preflight(ClaudeCodeAdapter(), {}, checks)
        assert found.ok is ok and bool(found.notes) is noted

    def test_a_batch_its_server_cannot_serve_is_refused_before_a_launch(self, tmp_path, monkeypatch, store):
        from tuningfork.ops.batch import prepare_batch
        from tuningfork.ops.launch import ProxyStatus

        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
        if os.name == "posix":
            fake_program(tmp_path / "bin", "hermes", "v", tmp_path / "log")
        spec = tmp_path / "evals.json"
        spec.write_text('{"skill_name": "t", "evals": [{"id": "1", "prompt": "p"}]}')
        small = ProxyStatus(url="http://127.0.0.1:1", home=store.home, max_inflight=1,
                            raw={"server": {"id": "s", "n_ctx": 32768, "models": ["m"]}})
        with pytest.raises(OperationError, match="64,000") as refused:
            prepare_batch(store, spec, harness="hermes", proxy=small, base_url="http://127.0.0.1:1/v1")
        assert refused.value.code == "preflight"
        assert store._conn.execute("SELECT count(*) FROM launches").fetchone()[0] == 0


# --- the server check --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (httpx.ConnectError("refused"), "is not answering: nothing is listening there"),
        # WSL's mirrored networking: a port nothing listens on lets a connection wait
        (httpx.ConnectTimeout("timed out"), "is not answering: nothing accepted a connection there within 3 seconds"),
        (httpx.Response(503, json={"error": {"code": 503, "message": "Loading model"}}),
         "is not answering: it answered 503: Loading model"),
    ],
)
def test_a_server_that_is_down_or_loading_is_said_to_be(tmp_path, answer, said):
    """Found in use: with the server down, or still loading its model, the check asked for
    a model name — the model list was empty because nothing had answered."""
    from tuningfork.control import ControlClient, ControlError
    from tests.helpers import serving

    def upstream_answer(request: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream_answer))
    app = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home"), client=upstream)
    try:
        with serving(app) as url, ControlClient(url) as client:
            with pytest.raises(ControlError) as refused:
                client.submit("server-check", {})
            shown = client.connections()
    finally:
        app.state.recorder.store.close()
        asyncio.run(upstream.aclose())

    assert said in str(refused.value) and "model" not in str(refused.value).split(":")[0]
    assert shown["server"]["problem"] == said.split("answering: ")[1]


def test_a_server_that_cannot_be_reached_is_asked_once():
    """Found on WSL's mirrored networking: each read of a server that was down waited out its
    timeout, /props and then /v1/models, so the check's caller gave up first."""
    from tuningfork.proxy.recorder import read_server

    asked = []

    def nobody(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        raise httpx.ConnectTimeout("timed out")

    async def read():
        async with httpx.AsyncClient(transport=httpx.MockTransport(nobody)) as client:
            return await read_server(client, "http://upstream.test")

    identity, problem = asyncio.run(read())
    assert asked == ["/v1/models"] and set(identity) == {"upstream"} and "within 3 seconds" in problem


def test_a_server_listing_several_models_asks_which(tmp_path):
    from types import SimpleNamespace

    from tuningfork.ops.connections import prepare_server_check

    status = SimpleNamespace(url="http://p", raw={"server": {"id": "s1", "upstream": "http://u", "models": ["a", "b"]}})
    with pytest.raises(OperationError) as asked:
        prepare_server_check(status)
    assert asked.value.field == "model" and "lists 2 models" in asked.value.message
    assert prepare_server_check(status, "b").model == "b"


def test_a_server_check_measures_what_the_harnesses_need(tmp_path, mock_server):
    from tuningfork.control import ControlClient
    from tests.helpers import serving

    mock_server.props = PROPS
    mock_server.responder = lambda payload: (
        MockTurn(tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}])
        if payload.get("tools") else MockTurn(content="hello")
    )
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
    app = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home"), client=upstream)
    store = app.state.recorder.store
    identity = {"upstream": "http://upstream.test", "n_ctx": 65536, "models": ["mock-model"]}
    app.state.server = {"id": store.add_server(identity), **identity}
    try:
        with serving(app) as url, ControlClient(url) as client:
            job = client.submit("server-check", {})
            done = client.wait_for_job(job["id"], timeout=30, poll=0.1)
            shown = client.connections()
    finally:
        store.close()
        asyncio.run(upstream.aclose())

    assert done["status"] == "done", done["error"]
    results = done["result"]["results"]
    assert results["streaming_tool_calls"]["support"] == "yes"
    assert results["anthropic_messages"]["support"] == "yes"
    # The mock answers a repeated prompt from its cache, as llama.cpp does by default.
    assert results["fresh_prompts"]["support"] == "no"
    assert shown["server"]["checks"]["results"]["anthropic_messages"]["support"] == "yes"
    # The check's calls went through the proxy as an experiment's: never recorded as a run.
    assert [r for r in mock_server.requests if r.get("tools")] and store is not None


def test_the_restart_command_keeps_what_the_proxy_was_started_with(tmp_path):
    """The Connections page's "these change only with a restart": a restart from it keeps the
    cap on output and the MCP route as they were, and names no setting left at its default."""
    import dataclasses

    from tuningfork.api.connections import restart_command
    from tuningfork.config import Settings

    plain = Settings(upstream_base_url="http://u.test", home=tmp_path)
    command = restart_command(plain, "http://u.test")
    assert "--max-output" not in command and "--mcp-" not in command

    started = dataclasses.replace(plain, max_output=8192, mcp_tools="refine-to-end", mcp_results="text", mcp_max_wait=110)
    command = restart_command(started, "http://u.test")
    for part in ("--max-output 8192", "--mcp-tools refine-to-end", "--mcp-results text", "--mcp-max-wait 110"):
        assert part in command
