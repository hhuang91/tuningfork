"""The proxy on a real socket.

Some behaviour only exists over a real connection. The event stream is the clearest case:
it never ends, by design, and an in-process test transport waits for a complete body — so
testing it in-process just hangs. The browser talks to a real server; so does this.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import httpx

from tuningfork.config import Settings
from tuningfork.demo.mock import MockServer, MockTurn
from tuningfork.proxy.app import ProxyServer, create_app
from tuningfork.proxy.recorder import Recorder
from tests.helpers import control_headers, free_port, serving

API = "/_tuningfork/api"
CHAT = "/v1/chat/completions"


def events(response: httpx.Response) -> Iterator[dict]:
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield json.loads(line[len("data: "):])


def test_the_event_stream_delivers_a_snapshot_then_live_changes(tmp_path):
    recorder = Recorder(Settings(upstream_base_url="http://127.0.0.1:9", home=tmp_path))
    app = create_app(recorder.settings, recorder=recorder)

    with serving(app) as base, httpx.Client(
        base_url=base, timeout=5, headers=control_headers(app)
    ) as client:
        with client.stream("GET", f"{API}/events") as stream:
            received = events(stream)
            snapshot = next(received)
            assert snapshot["type"] == "snapshot" and snapshot["live"] is True

            # A change made through the API reaches the open stream.
            created = client.post(f"{API}/breakpoints", json={"phase": "request"})
            assert created.status_code == 201
            update = next(received)
            assert update["type"] == "breakpoints"
            assert update["breakpoints"][0]["id"] == created.json()["id"]

    recorder.store.close()


def test_stopping_releases_held_calls_and_ends_event_streams(tmp_path):
    """A server waits for open responses before it stops, and neither a held call nor an
    event stream ever finishes by itself. Stopping used to hang until a second Ctrl-C cut
    every connection; now the held call is answered and the stream ends."""
    mock = MockServer(MockTurn(content="the model's answer"))
    settings = Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home")
    recorder = Recorder(settings)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock.app))
    port = free_port()
    app = create_app(settings, client=upstream, recorder=recorder)
    proxy = ProxyServer(app, host="127.0.0.1", port=port)
    server = threading.Thread(target=proxy.run, daemon=True)
    server.start()
    deadline = time.monotonic() + 5
    while not proxy.server.started:
        assert time.monotonic() < deadline, "the proxy did not start"
        time.sleep(0.01)
    base = f"http://127.0.0.1:{port}"

    def watch_events() -> list[dict]:
        with httpx.Client(base_url=base, timeout=10) as client:
            with client.stream("GET", f"{API}/events") as stream:
                return list(events(stream))  # returns only once the stream ends

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(
        base_url=base, timeout=10, headers=control_headers(app)
    ) as control, ThreadPoolExecutor(2) as pool:
        control.post(f"{API}/breakpoints", json={"phase": "request"})
        watching = pool.submit(watch_events)
        call = pool.submit(httpx.post, f"{base}{CHAT}", json=body, timeout=10)
        deadline = time.monotonic() + 5
        while not control.get(f"{API}/holds").json():
            assert time.monotonic() < deadline, "the call was not held"
            time.sleep(0.02)

        proxy.server.handle_exit(signal.SIGINT, None)  # what Ctrl-C does

        answered = call.result(timeout=10)
        assert answered.status_code == 200
        assert answered.json()["choices"][0]["message"]["content"] == "the model's answer"
        seen = watching.result(timeout=10)
        assert seen[0]["type"] == "snapshot"

    server.join(10)
    assert not server.is_alive(), "the proxy did not stop"
    recorder.store.close()
