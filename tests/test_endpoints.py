"""Protected endpoints: tuningfork sends nothing to them, from anywhere.

A model server something else depends on — a production agent's, say — must never receive
experiment load. Nothing is protected until a person says so; then these tests hold it in
code — at every place a URL is chosen, and at the HTTP clients as a backstop.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from tuningfork.config import Settings, machine_config_path, write_home_config
from tuningfork.ops.common import OperationError
from tuningfork.endpoints import (
    ProtectedEndpoint,
    check_endpoint,
    guard_hooks,
    is_loopback,
    parse_endpoint,
    protected,
)


@pytest.fixture
def unprotected(monkeypatch, tmp_path):
    """No list at all: no environment variable, and a machine config that does not exist."""
    monkeypatch.delenv("TUNINGFORK_PROTECTED", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def protect_machine(*specs: str) -> None:
    write_home_config(machine_config_path().parent, {"protected": list(specs)})


@pytest.mark.usefixtures("unprotected")
class TestTheList:
    def test_nothing_is_protected_until_a_person_says_so(self):
        """LM Studio's default port is as likely a person's own experiment as anything."""
        assert protected() == ()
        assert check_endpoint("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:1234/v1",
            "http://localhost:1234",
            "http://LOCALHOST:1234/v1/chat/completions",
            "http://[::1]:1234/v1",
            "http://0.0.0.0:1234",
            "http://127.0.1.1:1234",
            "http://[::ffff:127.0.0.1]:1234",
            "localhost:1234",
        ],
    )
    def test_every_spelling_of_this_machine_is_refused(self, url):
        protect_machine("localhost:1234")
        with pytest.raises(ProtectedEndpoint) as refused:
            check_endpoint(url)
        assert refused.value.code == "protected_endpoint"
        assert str(machine_config_path()) in refused.value.hint  # where to change it

    def test_a_refusal_shows_the_address_without_a_key(self):
        protect_machine("localhost:1234")
        with pytest.raises(ProtectedEndpoint) as refused:
            check_endpoint("http://tf:Sup3rS3cr3t@localhost:1234/v1?key=abc1")
        assert str(refused.value).startswith("http://<redacted>@localhost:1234/v1?key=<redacted> is a protected")

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:8081", "http://192.168.1.5:1234", "https://localhost/v1"]
    )
    def test_other_endpoints_are_left_alone(self, url):
        protect_machine("localhost:1234")
        assert check_endpoint(url) == url

    def test_the_machine_s_list_holds_whatever_the_environment_says(self, monkeypatch):
        """A shell exporting an empty TUNINGFORK_PROTECTED once unprotected everything."""
        protect_machine("localhost:1234")
        monkeypatch.setenv("TUNINGFORK_PROTECTED", "")
        with pytest.raises(ProtectedEndpoint):
            check_endpoint("http://127.0.0.1:1234")

    def test_the_environment_adds_to_it(self, monkeypatch):
        monkeypatch.setenv("TUNINGFORK_PROTECTED", "*:9000, gpu-box:*")
        with pytest.raises(ProtectedEndpoint) as refused:
            check_endpoint("http://anywhere:9000")
        assert "TUNINGFORK_PROTECTED" in refused.value.hint
        with pytest.raises(ProtectedEndpoint):
            check_endpoint("http://GPU-BOX:8081")
        check_endpoint("http://127.0.0.1:1234")

    def test_a_recordings_folder_adds_to_it(self, tmp_path):
        protect_machine("localhost:1234")
        write_home_config(tmp_path, {"protected": ["127.0.0.1:8090"]})
        with pytest.raises(ProtectedEndpoint) as refused:
            check_endpoint("http://localhost:8090", home=tmp_path)
        assert str(tmp_path / "config.json") in refused.value.hint
        with pytest.raises(ProtectedEndpoint):
            check_endpoint("http://localhost:1234", home=tmp_path)

    @pytest.mark.parametrize("text", ["{not json", '{"protected": "localhost:1234"}', "[]"])
    def test_a_machine_config_that_cannot_be_read_stops_all_traffic(self, text):
        """Not protecting nothing without a word: a typo there once would have."""
        machine_config_path().parent.mkdir(parents=True)
        machine_config_path().write_text(text, encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            check_endpoint("http://127.0.0.1:8081")
        assert refused.value.code == "machine_config" and str(machine_config_path()) in str(refused.value)

    def test_an_entry_that_cannot_be_read_names_where_it_is(self, monkeypatch):
        monkeypatch.setenv("TUNINGFORK_PROTECTED", "localhost")
        with pytest.raises(OperationError) as refused:
            check_endpoint("http://127.0.0.1:8081")
        assert refused.value.code == "protected_list" and "TUNINGFORK_PROTECTED" in str(refused.value)

    def test_parsing(self):
        assert str(parse_endpoint("[::1]:1234")) == "[::1]:1234"
        assert str(parse_endpoint("*:80")) == "*:80"
        with pytest.raises(ValueError):
            parse_endpoint("no-port")
        assert is_loopback("127.5.5.5") and not is_loopback("10.0.0.1")


class TestEveryPlaceAUrlIsChosen:
    """Each refuses before anything is sent: the mock server behind them never hears a word."""

    PROTECTED = "http://127.0.0.1:1234"

    def test_a_proxy_is_never_built_in_front_of_one(self, tmp_path):
        from tuningfork.proxy.app import create_app

        with pytest.raises(ProtectedEndpoint):
            create_app(Settings(upstream_base_url=f"{self.PROTECTED}/v1", home=tmp_path))

    def test_the_probe_refuses(self, mock_server):
        from tuningfork.backends.probe import probe_backend

        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        try:
            with pytest.raises(ProtectedEndpoint):
                asyncio.run(probe_backend(self.PROTECTED, client=client))
        finally:
            asyncio.run(client.aclose())
        assert mock_server.requests == []

    def test_resampling_refuses(self, tmp_path, mock_server):
        from tuningfork.experiment import SamplingPlan, load_checkpoint, run_resample
        from tuningfork.ops.experiments import prepare_resample
        from tuningfork.store.db import Store
        from tests.helpers import record_pilot

        store = Store(tmp_path / "store")
        run_id = record_pilot(store)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))
        try:
            with pytest.raises(ProtectedEndpoint):
                prepare_resample(store, run_id, "0", base_url=self.PROTECTED)
            # And below the operation: the sampling engine itself.
            with pytest.raises(ProtectedEndpoint):
                asyncio.run(
                    run_resample(
                        store, load_checkpoint(store, run_id, 0), SamplingPlan(n=2),
                        base_url=self.PROTECTED, client=client,
                    )
                )
            assert store.experiments(limit=5) == []
        finally:
            asyncio.run(client.aclose())
            store.close()
        assert mock_server.requests == []

    def test_a_launched_harness_is_never_pointed_at_one(self, tmp_path):
        from tuningfork.ops.launch import check_proxy

        with pytest.raises(ProtectedEndpoint):
            check_proxy(f"{self.PROTECTED}/v1", tmp_path)

    def test_the_clients_refuse_even_when_nothing_else_checked(self, mock_server):
        """The backstop: an httpx hook on tuningfork's own clients."""

        async def send() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mock_server.app), event_hooks=guard_hooks()
            ) as client:
                await client.get(f"{self.PROTECTED}/v1/models")

        with pytest.raises(ProtectedEndpoint):
            asyncio.run(send())
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)),
                          event_hooks=guard_hooks(sync=True)) as client:
            with pytest.raises(ProtectedEndpoint):
                client.get(f"{self.PROTECTED}/v1/models")
        assert mock_server.requests == []

    def test_serve_and_probe_refuse_at_the_command_line(self, tmp_path):
        from tuningfork.cli import app

        runner = CliRunner()
        served = runner.invoke(app, ["serve", "-u", f"{self.PROTECTED}/v1", "--home", str(tmp_path)])
        probed = runner.invoke(app, ["probe", "http://localhost:1234"])

        assert served.exit_code == 1 and "protected endpoint" in served.output
        assert probed.exit_code == 1 and "protected endpoint" in probed.output
        assert not (tmp_path / "tuningfork.db").exists()  # refused before the store opened


class TestWhereAServerIs:
    """On the internet or not, read from every address the server's host resolves to: what a
    reminder that tuningfork is built for a model run oneself goes by, never a provider's name."""

    @staticmethod
    def resolving(*addresses):
        def resolve(host, port, type=0):
            if not addresses:
                raise OSError("no such host")
            return [(2, 1, 6, "", (address, port)) for address in addresses]

        return resolve

    @pytest.mark.parametrize(("addresses", "internet"), [
        (("127.0.0.1",), False),  # this machine
        (("192.168.1.20",), False),  # a box on the network
        (("10.0.0.7",), False),
        (("100.101.102.103",), False),  # a GPU of one's own on Tailscale
        (("fd7a:115c:a1e0::1",), False),
        (("8.8.8.8",), True),
        (("2001:4860:4860::8888",), True),
        (("8.8.8.8", "10.0.0.7"), False),  # a name that reaches a private address too
        ((), None),  # it does not resolve
    ])
    def test_on_the_internet_only_where_every_address_is_public(self, addresses, internet):
        from tuningfork.endpoints import on_internet

        assert on_internet("https://api.example.com/v1", resolve=self.resolving(*addresses)) is internet


def test_the_home_config_is_written_atomically_and_read_leniently(tmp_path):
    from tuningfork.config import read_home_config

    assert read_home_config(tmp_path) == {}
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert read_home_config(tmp_path) == {}
    write_home_config(tmp_path, {"protected": ["a:1"]})
    assert json.loads((tmp_path / "config.json").read_text())["protected"] == ["a:1"]
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]
