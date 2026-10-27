"""The access token: every state change needs it, reads do not, and the CLI finds it
without being told.

A web page open in the user's browser can send requests to a loopback port but cannot read
the token, so it cannot change anything. What changes state is about to include starting
harnesses that run shell commands, which is why this is a structural check in one place.
"""

from __future__ import annotations

import json

import os
import stat

import pytest

from tuningfork.access import ENV, HEADER, Access, matches, read_token, token_path, write_token
from tuningfork.config import Settings
from tuningfork.control import ControlClient, ControlError
from tuningfork.proxy.app import create_app
from tuningfork.specs import run_check
from tests.helpers import serving

API = "/_tuningfork/api"


class TestTheTokenFile:
    @pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
    def test_it_is_readable_by_this_user_only(self, tmp_path):
        path = write_token(tmp_path / "home", "secret")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert read_token(path) == "secret"
        assert [p.name for p in path.parent.iterdir()] == [".access-token"]  # nothing left over

    def test_a_missing_or_empty_file_holds_no_token(self, tmp_path):
        assert read_token(tmp_path / "nothing") is None
        (tmp_path / "empty").write_text("\n")
        assert read_token(tmp_path / "empty") is None

    def test_each_start_makes_a_new_token_unless_one_is_pinned(self, tmp_path):
        first, second = Access.start(tmp_path), Access.start(tmp_path)
        assert first.token != second.token and len(first.token) >= 32
        assert read_token(token_path(tmp_path)) == second.token
        assert Access.start(tmp_path, "pinned").token == "pinned"

    def test_matching(self):
        assert matches("a", "a")
        assert not matches("a", "b") and not matches("", "") and not matches(None, "a")

    def test_settings_never_show_it(self, tmp_path):
        settings = Settings(upstream_base_url="http://x", home=tmp_path, token="tok-5f3a9c")
        assert "tok-5f3a9c" not in repr(settings)


class TestTheProxy:
    async def test_health_says_where_the_token_is_never_what_it_is(self, proxy_client, proxy_app):
        health = (await proxy_client.get("/_tuningfork/health")).json()
        assert health["token_required"] is True
        assert read_token(health["token_file"]) == proxy_app.state.access.token
        assert proxy_app.state.access.token not in str(health)

    async def test_status_says_whether_the_page_may_act(self, proxy_client):
        assert (await proxy_client.get(f"{API}/status")).json()["auth"] == {"required": True, "valid": True}
        without = await proxy_client.get(f"{API}/status", headers={HEADER: ""})
        assert without.json()["auth"] == {"required": True, "valid": False}
        wrong = await proxy_client.get(f"{API}/status", headers={HEADER: "guess"})
        assert wrong.json()["auth"]["valid"] is False


class TestTheCli:
    @pytest.fixture
    def proxy(self, settings, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        app = create_app(settings)
        try:
            with serving(app) as url:
                yield app, url
        finally:
            app.state.recorder.store.close()

    def test_the_client_finds_the_token_in_the_proxys_home(self, proxy):
        app, url = proxy
        with ControlClient(url) as client:
            assert client.pause()["paused"] is True
            assert client.token == app.state.access.token

    def test_a_token_left_in_the_environment_from_an_earlier_proxy_is_replaced(self, proxy, monkeypatch):
        app, url = proxy
        monkeypatch.setenv(ENV, "from-yesterday")
        with ControlClient(url) as client:
            assert client.pause()["paused"] is True
            assert client.token == app.state.access.token

    def test_a_token_given_outright_is_not_second_guessed(self, proxy):
        app, url = proxy
        with ControlClient(url, token="wrong") as client, pytest.raises(ControlError) as refused:
            client.pause()
        assert "not this proxy's" in str(refused.value) and "tuningfork token" in str(refused.value)
        assert app.state.gate.paused is False

    def test_tuningfork_token_prints_the_address_with_it(self, proxy, monkeypatch):
        from typer.testing import CliRunner

        from tuningfork.cli import app as cli

        app, url = proxy
        printed = CliRunner().invoke(cli, ["token", "--proxy", url])
        assert printed.exit_code == 0, printed.output
        assert printed.output.strip() == f"{url}/_tuningfork/ui/?token={app.state.access.token}"
        raw = CliRunner().invoke(cli, ["token", "--proxy", url, "--raw"])
        assert raw.output.strip() == app.state.access.token
        # For an MCP client: the MCP token, which opens the MCP endpoint alone — alone, or as
        # the header for a client that runs a command for it, such as Claude Code's headersHelper.
        mcp = CliRunner().invoke(cli, ["token", "--proxy", url, "--mcp"])
        assert mcp.exit_code == 0 and mcp.output.strip() == app.state.access.mcp_token
        header = CliRunner().invoke(cli, ["token", "--proxy", url, "--header-json"])
        assert json.loads(header.output) == {"Authorization": f"Bearer {app.state.access.mcp_token}"}


def test_the_client_reads_only_a_token_file_of_a_proxy_on_this_machine(tmp_path, monkeypatch):
    """Found in review: the client read whatever file a server's /health named, and sent what
    it held back as the token — any file, to any server it was pointed at."""
    import httpx

    monkeypatch.delenv(ENV, raising=False)
    secret = tmp_path / "id_ed25519"
    secret.write_text("PRIVATE KEY", encoding="utf-8")
    own = tmp_path / "home" / ".access-token"
    own.parent.mkdir()
    own.write_text("the-token", encoding="utf-8")

    def client_for(url, named):
        client = ControlClient(url)
        client._http.close()
        client._http = httpx.Client(base_url=f"{url}/_tuningfork/api", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"token_file": str(named)})))
        return client

    assert client_for("http://127.0.0.1:8787", secret).token is None
    assert client_for("http://198.51.100.7:8787", own).token is None  # not a proxy on this machine
    assert client_for("http://127.0.0.1:8787", own).token == "the-token"


def test_a_tasks_check_never_sees_the_token(tmp_path, monkeypatch):
    """A check is the spec's own shell command, and a spec can come from anywhere."""
    monkeypatch.setenv(ENV, "secret")
    result = run_check('test -z "${TUNINGFORK_TOKEN:-}"', tmp_path)
    assert result.passed, result.output


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_a_secret_file_is_this_users_alone(tmp_path):
    from tuningfork.access import write_secret

    path = write_secret(tmp_path / "home" / ".proposer-key", "sk-test")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and read_token(path) == "sk-test"
    assert [p.name for p in path.parent.iterdir()] == [".proposer-key"]


def test_a_tasks_check_never_sees_the_proposers_key(tmp_path, monkeypatch):
    from tuningfork.access import PROPOSER_KEY_ENV

    monkeypatch.setenv(PROPOSER_KEY_ENV, "sk-secret")
    result = run_check('test -z "${TUNINGFORK_PROPOSER_KEY:-}"', tmp_path)
    assert result.passed, result.output


class TestTheMcpToken:
    """An MCP client's own token: new each start unless pinned, never the access token, and
    kept like it — this user's alone, and never handed to a task's check."""

    @pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
    def test_it_is_written_beside_the_access_token_and_differs(self, tmp_path):
        access = Access.start(tmp_path / "home")
        assert access.mcp_token != access.token and access.mcp_file.name == ".mcp-token"
        assert stat.S_IMODE(access.mcp_file.stat().st_mode) == 0o600
        assert read_token(access.mcp_file) == access.mcp_token
        assert access.allows_mcp(access.mcp_token) and not access.allows_mcp(access.token)
        assert access.allows(access.token) and not access.allows(access.mcp_token)

    def test_pinned_it_stays_and_one_token_for_both_doors_is_refused(self, tmp_path):
        assert Access.start(tmp_path / "a", "control", "pinned").mcp_token == "pinned"
        with pytest.raises(ValueError, match="must differ"):
            Access.start(tmp_path / "b", "same", "same")

    def test_a_tasks_check_never_sees_it(self, tmp_path, monkeypatch):
        from tuningfork.access import MCP_ENV

        monkeypatch.setenv(MCP_ENV, "mcp-secret")
        result = run_check('test -z "${TUNINGFORK_MCP_TOKEN:-}"', tmp_path)
        assert result.passed, result.output


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
class TestTheRecordingsFolder:
    def test_one_made_here_is_this_users_alone_and_one_made_before_keeps_its_mode(self, tmp_path):
        from tuningfork.store.db import Store

        Store(tmp_path / "new" / "home").close()
        assert stat.S_IMODE((tmp_path / "new" / "home").stat().st_mode) == 0o700
        before = tmp_path / "before"
        before.mkdir()
        before.chmod(0o755)
        Store(before).close()
        assert stat.S_IMODE(before.stat().st_mode) == 0o755  # told, never changed

    def test_who_else_can_open_it_is_said_with_how_to_make_it_private(self, tmp_path):
        from tuningfork.access import open_to_others

        folder = tmp_path / "home"
        folder.mkdir()
        said = {}
        for mode in (0o700, 0o750, 0o705, 0o755):
            folder.chmod(mode)
            said[mode] = open_to_others(folder)
        assert said[0o700] is None
        assert said[0o750].startswith("its group on this machine can open it (mode 750)")
        assert said[0o705].startswith("other users on this machine can open it (mode 705)")
        assert said[0o755].startswith("its group and other users on this machine")
        assert said[0o755].endswith(f"`chmod 700 {folder}` makes it yours alone")

    def test_every_way_one_is_first_made_makes_it_this_users_alone(self, tmp_path):
        from tuningfork.access import write_secret
        from tuningfork.config import write_home_config
        from tuningfork.demo import lab

        write_home_config(tmp_path / "configured", {"protected": []})
        write_secret(tmp_path / "keyed" / ".proposer-key", "sk-test")
        lab.demo_home(tmp_path / "demo", "lab")
        for made in ("configured", "keyed", "demo"):
            assert stat.S_IMODE((tmp_path / made).stat().st_mode) == 0o700, made

    def test_its_advice_quotes_the_folder_and_holds_on_a_windows_drive(self, tmp_path, monkeypatch):
        from tuningfork import access

        spaced = tmp_path / "my recordings"
        spaced.mkdir()
        spaced.chmod(0o755)
        assert f"`chmod 700 '{spaced}'`" in access.open_to_others(spaced)

        monkeypatch.setattr("tuningfork.paths.under_wsl", lambda: True)
        monkeypatch.setattr("tuningfork.paths.drive_root", lambda: str(tmp_path) + "/")
        drive = tmp_path / "c"
        drive.mkdir()
        drive.chmod(0o777)
        said = access.open_to_others(drive)
        assert "on a Windows drive" in said and "chmod 700" not in said
