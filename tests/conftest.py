"""Shared fixtures.

Everything runs in-process: the test client talks to the proxy app, which talks to the mock
server app, with no sockets, no ports and no model. That is what keeps the suite runnable
by a contributor with no GPU.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.config import Settings
from tuningfork.demo.mock import MockServer
from tuningfork.proxy.app import create_app
from tuningfork.proxy.recorder import Recorder


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Every test sees the CLI at one width, whatever the shell exports. Rich fixes a
    console's width when it is created, from COLUMNS if that is exported then: found when a
    guide's `pytest -q` failed one CLI test in a shell exporting COLUMNS=100, its sentence
    wrapped where the assertion looked for it."""
    from rich.console import Console

    import tuningfork.cli

    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr(tuningfork.cli, "console", Console(width=220))


@pytest.fixture(autouse=True)
def protected_endpoints(monkeypatch, tmp_path_factory):
    """No test reads the machine's own config, and none sends anything to port 1234, LM
    Studio's default, whatever serves there: some tests name it as the endpoint they expect
    refused. `tests/test_endpoints.py` sets the list itself."""
    from tuningfork.endpoints import ENV

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.getbasetemp() / "no-machine-config"))
    monkeypatch.setenv(ENV, "localhost:1234")


@pytest.fixture(autouse=True)
def unsandboxed(monkeypatch):
    """Launches run in their own folders unless a test asks for the sandbox: the suite runs
    where bubblewrap does not, and a test's expectations should not depend on whether it
    does. `tests/test_sandbox.py` turns it on where it can run."""
    from tuningfork.sandbox import SWITCH

    monkeypatch.setenv(SWITCH, "off")


@pytest.fixture(autouse=True)
def unmeasured_prompt_cache():
    """Each test measures a server's prompt cache afresh: the measurement is kept a while,
    by the server's address, and every mock server answers at the same one."""
    from tuningfork.backends import llamacpp

    llamacpp._reuse.clear()
    yield
    llamacpp._reuse.clear()


@pytest.fixture
def mock_server() -> MockServer:
    return MockServer()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home")


@pytest.fixture
def recorder(settings: Settings):
    recorder = Recorder(settings)
    yield recorder
    recorder.store.close()


@pytest.fixture
def store(tmp_path: Path):
    from tuningfork.store.db import Store

    store = Store(tmp_path / "store")
    yield store
    store.close()


@pytest.fixture
def upstream_client(mock_server: MockServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_server.app), timeout=30
    )


@pytest.fixture
def proxy_app(settings: Settings, upstream_client: httpx.AsyncClient, recorder: Recorder):
    return create_app(settings, client=upstream_client, recorder=recorder)


@pytest.fixture
def settle(proxy_app):
    """Wait for detached recording tasks to finish.

    Recording is deliberately not awaited by the request that triggers it, so a harness
    hanging up cannot cancel it. Tests therefore have to wait for it explicitly.
    """

    async def _settle(app=None) -> None:
        import asyncio

        target = app if app is not None else proxy_app
        await asyncio.gather(*list(target.state.background), return_exceptions=True)

    return _settle


@pytest.fixture
async def proxy_client(proxy_app, upstream_client: httpx.AsyncClient):
    """A client of the proxy that holds its access token, as the CLI and the UI do."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app),
        base_url="http://localhost:8787",
        timeout=30,
        headers={TOKEN_HEADER: proxy_app.state.access.token},
    ) as client:
        yield client
    await upstream_client.aclose()
