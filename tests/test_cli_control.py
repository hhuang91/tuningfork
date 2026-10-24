"""The breakpoint and hold commands, against a proxy on a real socket.

The CLI talks HTTP to a proxy running somewhere else, so these tests run one: the proxy app
on a local port, forwarding to the mock server, with the harness's call made from a thread
that blocks at the breakpoint while the commands inspect and decide it.
"""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.config import Settings
from tuningfork.control import ControlClient, ControlError
from tuningfork.demo.mock import MockServer, MockTurn
from tuningfork.proxy.app import create_app, create_browse_app
from tuningfork.proxy.recorder import Recorder
from tuningfork.store.db import Store
from tests.helpers import free_port, serving

CHAT = "/v1/chat/completions"


@pytest.fixture
def cli(monkeypatch):
    # Wide enough that Rich does not wrap the lines the assertions look for. The CLI's
    # console is replaced, not just COLUMNS set: Rich fixes a console's width when it is
    # created if COLUMNS is exported then, so a shell exporting it would fail these tests.
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    runner = CliRunner()

    def invoke(*args: str):
        result = runner.invoke(app, list(args))
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            result.output
        )
        return result

    return invoke


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A proxy on a real port, forwarding to the mock server; the CLI pointed at it."""
    mock = MockServer()
    settings = Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home")
    recorder = Recorder(settings)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock.app), timeout=30)
    with serving(create_app(settings, client=upstream, recorder=recorder)) as base:
        monkeypatch.setenv("TUNINGFORK_PROXY", base)
        yield base, mock
    asyncio.run(upstream.aclose())
    recorder.store.close()


@pytest.fixture
def control(live):
    with ControlClient() as client:
        yield client


def harness_call(base: str, content: str = "hello", **extra) -> Future:
    """The harness's model call, made in the background: it blocks while held."""
    body = {"model": "m", "messages": [{"role": "user", "content": content}], **extra}
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(httpx.post, f"{base}{CHAT}", json=body, timeout=15)
    pool.shutdown(wait=False)
    return future


def hold_id_in(output: str) -> str:
    found = re.search(r"hold ([0-9a-f]{8})", output)
    assert found, output
    return found.group(1)


class TestBreakCommands:
    def test_a_tool_condition_means_after_the_model(self, live, cli):
        result = cli("break", "add", "--tool", "write_file", "--path", "*.json")
        assert result.exit_code == 0, result.output
        assert "after the model: write_file on *.json" in result.output
        assert "write_file on *.json" in cli("break", "list").output

    def test_without_one_it_stops_before_the_model(self, live, cli):
        assert "before the model: step >= 2" in cli("break", "add", "--from-step", "2").output
        assert "before the model: step 3" in cli("break", "add", "--step", "3").output

    def test_step_and_from_step_conflict(self, live, cli):
        result = cli("break", "add", "--step", "1", "--from-step", "2")
        assert result.exit_code == 1
        assert "not both" in result.output

    def test_off_on_and_rm(self, live, cli, control):
        bp = control.add_breakpoint(phase="request")
        cli("break", "off", bp["id"])
        assert control.breakpoints()[0]["enabled"] is False
        cli("break", "on", bp["id"][:4])
        assert control.breakpoints()[0]["enabled"] is True
        cli("break", "rm", bp["id"])
        assert control.breakpoints() == []

    def test_an_impossible_breakpoint_is_refused_readably(self, live, cli):
        result = cli("break", "add", "--phase", "request", "--tool", "write_file")
        assert result.exit_code == 1
        assert "response phase" in result.output

    def test_an_unknown_id_is_refused_readably(self, live, cli):
        result = cli("break", "rm", "nope")
        assert result.exit_code == 1
        assert "no single breakpoint" in result.output


class TestHoldCommands:
    def test_edit_a_request_before_the_model_sees_it(self, live, cli, tmp_path):
        base, mock = live
        mock.script(MockTurn(content="done"))
        cli("break", "add")
        call = harness_call(base, "the original question")

        waited = cli("hold", "wait", "--timeout", "5")
        assert waited.exit_code == 0, waited.output
        assert "before the model" in waited.output
        assert "the original question" in waited.output
        hold_id = hold_id_in(waited.output)
        assert "waiting" in cli("hold", "list").output

        request = json.loads(cli("hold", "show", hold_id, "--json").output)
        request["messages"][0]["content"] = "an edited question"
        edited = tmp_path / "request.json"
        edited.write_text(json.dumps(request), encoding="utf-8")
        assert "continue with your edit" in cli(
            "hold", "continue", hold_id, "--edited", str(edited)
        ).output

        response = call.result(timeout=10)
        assert response.json()["choices"][0]["message"]["content"] == "done"
        assert mock.requests[-1]["messages"][0]["content"] == "an edited question"
        assert "continue" in cli("hold", "list", "--recent").output

    def test_edit_what_the_model_decided_before_the_harness_acts(self, live, cli, tmp_path):
        base, mock = live
        mock.script(
            MockTurn(
                tool_calls=[
                    {
                        "name": "write_file",
                        "arguments": {"path": "out.json", "content": '{"count": 3}'},
                    }
                ]
            )
        )
        cli("break", "add", "--tool", "write_file", "--path", "*.json")
        call = harness_call(base, "write the report")

        waited = cli("hold", "wait", "--timeout", "5").output
        assert "after the model" in waited
        assert "write_file" in waited
        hold_id = hold_id_in(waited)

        response = json.loads(cli("hold", "show", hold_id, "--json").output)
        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        arguments = json.loads(function["arguments"])
        arguments["content"] = '{"count": 4}'
        function["arguments"] = json.dumps(arguments)
        edited = tmp_path / "response.json"
        edited.write_text(json.dumps(response), encoding="utf-8")
        cli("hold", "continue", hold_id, "--edited", str(edited))

        delivered = call.result(timeout=10).json()
        written = delivered["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(written)["content"] == '{"count": 4}'

    def test_step_stops_again_once_the_model_has_answered(self, live, cli):
        base, mock = live
        mock.script(MockTurn(content="the answer"))
        cli("break", "add", "--once")
        call = harness_call(base)

        first = hold_id_in(cli("hold", "wait", "--timeout", "5").output)
        cli("hold", "step", first)
        after = cli("hold", "wait", "--timeout", "5").output
        assert "after the model" in after
        assert "stepped here from before the call" in after
        assert "the answer" in after
        assert hold_id_in(after) != first

        cli("hold", "continue", hold_id_in(after))
        assert call.result(timeout=10).json()["choices"][0]["message"]["content"] == "the answer"

    def test_drop_fails_the_call_before_the_model_runs(self, live, cli):
        base, mock = live
        cli("break", "add")
        call = harness_call(base)
        hold_id = hold_id_in(cli("hold", "wait", "--timeout", "5").output)

        assert "drop" in cli("hold", "drop", hold_id).output
        assert call.result(timeout=10).status_code == 409
        assert mock.requests == []

    def test_a_broken_edit_file_leaves_the_call_held(self, live, cli, tmp_path):
        base, _ = live
        cli("break", "add")
        call = harness_call(base)
        hold_id = hold_id_in(cli("hold", "wait", "--timeout", "5").output)

        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        result = cli("hold", "continue", hold_id, "--edited", str(broken))
        assert result.exit_code == 1
        assert "could not read" in result.output

        # Nothing was decided, so the call can still go ahead.
        cli("hold", "continue", hold_id)
        assert call.result(timeout=10).status_code == 200

    def test_wait_gives_up(self, live, cli):
        result = cli("hold", "wait", "--timeout", "0.3")
        assert result.exit_code == 1
        assert "nothing was held" in result.output


def test_pause_refuses_model_calls_until_resume(live, cli, control):
    base, _ = live
    cli("pause")
    assert control.status()["paused"] is True
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert httpx.post(f"{base}{CHAT}", json=body, timeout=5).status_code == 503

    cli("resume")
    assert httpx.post(f"{base}{CHAT}", json=body, timeout=5).status_code == 200


def test_no_proxy_is_explained(cli, monkeypatch):
    monkeypatch.setenv("TUNINGFORK_PROXY", f"http://127.0.0.1:{free_port()}")
    result = cli("break", "list")
    assert result.exit_code == 1
    assert "no tuningfork proxy is answering" in result.output


def test_a_browse_only_server_has_no_live_controls(tmp_path):
    store = Store(tmp_path / "store")
    with serving(create_browse_app(store)) as base, ControlClient(base) as client:
        with pytest.raises(ControlError, match="browsing recordings only"):
            client.status()
        with pytest.raises(ControlError, match="no live controls"):
            client.breakpoints()
    store.close()


def test_an_id_with_a_slash_is_reported_as_unknown(live, cli):
    """A phrase pasted as an id once read as "this server has no live controls"."""
    result = cli("hold", "continue", "/not an id/")
    assert result.exit_code == 1
    assert "no single hold" in result.output


def test_a_proxy_address_without_a_scheme_is_explained(cli):
    """`--proxy localhost:8787` once ended in httpx's UnsupportedProtocol traceback."""
    result = cli("hold", "list", "--proxy", "localhost:8787")
    assert result.exit_code == 1
    assert "not a URL" in result.output


def test_hold_list_quiet_prints_only_waiting_ids(live, cli, control):
    base, _ = live
    control.add_breakpoint(phase="request")
    body = {"model": "m", "messages": [{"role": "user", "content": "quiet"}]}
    with ThreadPoolExecutor(1) as pool:
        call = pool.submit(httpx.post, f"{base}{CHAT}", json=body, timeout=30)
        held = control.wait_for_hold(timeout=10)
        result = cli("hold", "list", "-q")
        assert result.output.split() == [held["id"]]
        control.decide(held["id"], "continue")
        assert call.result().status_code == 200
