"""Small helpers shared by tests. The tally task's skill, model and harness are the demo's own
(`tuningfork.demo.task`), imported here for the tests that run it."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import uvicorn

from tuningfork.demo.task import (  # noqa: F401 - re-exported
    CHECK,
    SKILL,
    VARIANT,
    FakeAgent,
    as_response,
    system_prompt,
    tally_model,
    tool_call_message,
)
from tuningfork.launches import LaunchThreads, request_path, split
from tuningfork.ops.forks import execute_forks, prepare_fork
from tuningfork.runner import BatchRunner
from tuningfork.specs import Eval, TaskSpec, TuningforkOptions
from tuningfork.store.db import Step, Store

#: A key the store masks whole, as `<redacted>`: what a step "masked as it was recorded" was given.
KEY = "sk-a1B2c3D4e5F6g7H8i9J0"

#: Each launch's threads, as the proxy would track them, for harnesses that record directly.
_THREADS: dict[str, LaunchThreads] = {}


def token_in(base_url: str) -> str:
    """The launch token in a launched harness's base URL."""
    from urllib.parse import urlsplit

    token, _ = split(urlsplit(base_url).path)
    assert token, f"not a launch URL: {base_url}"
    return token


def record_launched(store: Store, token: str, request: dict[str, Any], **step: Any) -> Step:
    """Record a call the way the proxy does for a launched harness: under its launch, in the
    thread its messages continue."""
    launch = store.get_launch(token)
    assert launch is not None, f"no launch {token}"
    threads = _THREADS.setdefault(
        token,
        LaunchThreads(
            token, label=launch.label, task_id=launch.task_id,
            main_role="fork" if launch.kind == "fork" else "main",
        ),
    )
    assignment = threads.assign(request_path(request))
    return store.record_step(request=request, assignment=assignment, **step)


def all_steps(store: Store) -> list[Step]:
    """Every recorded step, oldest run first, in step order."""
    runs = sorted(store.runs(limit=1000, threads=True), key=lambda run: run.started_at)
    return [step for run in runs for step in store.steps(run.id)]


def only_step(store: Store) -> Step:
    steps = all_steps(store)
    assert len(steps) == 1, f"expected exactly one step, found {len(steps)}"
    return steps[0]


def text_message(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def messages_stream(*calls: tuple[str, Sequence[str]], text: str = "") -> bytes:
    """An Anthropic Messages answer as a server streams it: some text, then tool calls, each
    input sent as these pieces of JSON text."""
    events: list[tuple[str, dict[str, Any]]] = []
    if text:
        events += [
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ]
    for index, (name, pieces) in enumerate(calls, start=1 if text else 0):
        events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
            "type": "tool_use", "id": f"toolu_{index}", "name": name, "input": {}}}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": index,
                                            "delta": {"type": "input_json_delta", "partial_json": piece}})
                   for piece in pieces]
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


# --- a recorded pilot run, shaped like a real Hermes one ---------------------------------

DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "n_ctx": 65536}
PROPS = {
    "default_generation_settings": {
        "n_ctx": 65536,
        "params": {"temperature": 1.0, "top_p": 0.95},
    }
}
READ = tool_call_message("read_file", {"path": "data.csv"}, call_id="c1")


def record_pilot(
    store: Store, *, model: str = "mock-model", server_defaults=DEFAULTS, read: str = "id,amount\n1,10\n",
    skill: str = SKILL, params: dict[str, Any] | None = None,
) -> str:
    """Record a two-step run: read the CSV (`read`, what the tool gave back), then write the report,
    `skill` preloaded — each request masked as the store masks it — and each step's parameters, where
    given, replaced by `params`: a step as it was recorded otherwise, before steps named the patterns
    that masked them (`{}`), or under other patterns."""
    store.begin_run("pilot", "tally-report/1")
    base = {
        "model": model,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "max_tokens": 65536,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first = [
        {"role": "system", "content": system_prompt(skill)},
        {"role": "user", "content": "Use the tally-report skill on data.csv."},
    ]
    store.record_step(
        request={**base, "messages": first},
        response=as_response(READ, finish_reason="tool_calls"),
        server_defaults=server_defaults,
    )
    second = [*first, READ, {"role": "tool", "tool_call_id": "c1", "content": read}]
    store.record_step(
        request={**base, "messages": second},
        response=as_response(tool_call_message("write_file", {"path": "report.json"})),
        server_defaults=server_defaults,
    )
    run = store.get_run("pilot")
    assert run is not None
    if params is not None:
        restamp(store, run.id, params)
    return run.id


def restamp(store: Store, run_id: str, params: dict[str, Any]) -> None:
    """Each of a run's steps given `params` in place of what the store stamped it with: a step
    recorded before steps named the patterns that masked them (`{}`), or under other patterns."""
    with store._lock:
        store._conn.execute("UPDATE steps SET params = ? WHERE run_id = ?", (json.dumps(params), run_id))
        store._conn.commit()


def record_run(
    store: Store,
    label: str,
    assistant_messages: Sequence[dict[str, Any]],
    *,
    prompt: str = "do the task",
    task_id: str | None = None,
    tools: Sequence[str] = (),
) -> str:
    """Record a synthetic multi-step run and return its id.

    Each assistant message is followed by a user turn, so successive requests extend the
    same path through the tree exactly as a real conversation would. `tools` are offered
    in every request, by name.
    """
    store.begin_run(label, task_id)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    offered = {"tools": [{"type": "function", "function": {"name": name}} for name in tools]} if tools else {}
    for message in assistant_messages:
        store.record_step(
            request={"model": "m", "messages": list(messages), **offered},
            response=as_response(message),
        )
        messages.append(message)
        messages.append({"role": "user", "content": "continue"})
    run = store.get_run(label)
    assert run is not None
    return run.id


async def until(predicate: Callable[[], Any], timeout: float = 3.0) -> Any:
    """Poll until `predicate()` is truthy. For waiting on a call to reach a breakpoint."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for the condition")


def sse_payloads(body: bytes) -> list[str]:
    """The `data:` payloads of an SSE body, keepalive comments excluded."""
    return [
        line[len("data: "):]
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def control_headers(app) -> dict[str, str]:
    """What a client sends to change a proxy's state: its access token."""
    from tuningfork.access import HEADER

    return {HEADER: app.state.access.token}


def mcp_headers(app) -> dict[str, str]:
    """What an MCP client sends to the proxy's MCP endpoint: the MCP token, which opens it alone."""
    from tuningfork.access import HEADER

    return {HEADER: app.state.access.mcp_token}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def serving(app) -> Iterator[str]:
    """Run an app on a real local socket, in a thread, for as long as the block lasts.

    For what only exists over a real connection: a never-ending event stream, and the CLI,
    which talks HTTP to a proxy in another process.
    """
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        # Jobs first, while the server still answers them: they write to the store, which
        # the caller closes next.
        jobs = getattr(getattr(app, "state", None), "jobs", None)
        if jobs is not None:
            jobs.shutdown(10)
        server.should_exit = True
        thread.join(5)


# --- a real harness process, a task, and a proxy to run it through --------------------------

#: The skill as the task's folder holds it; `VARIANT` is its edit.
ORIGINAL = SKILL


@dataclass
class Stack:
    """A real proxy on a socket, the mock server as its model, a task, and a store."""

    url: str
    store: Store
    app: object
    mock: object
    tmp: Path
    spec: TaskSpec
    item: Eval

    def record(self):
        """One batch run of the task, as the source a fork replays."""
        runner = BatchRunner(
            self.store, FakeAgent(), base_url=f"{self.url}/v1", model="m",
            workdir=self.tmp / "batches", settle=0.1,
        )
        (outcome,) = runner.run(self.spec, self.item, 1).outcomes
        assert outcome.run_id, outcome.stderr
        return outcome

    def fork(self, run_id: str, step: str, n: int = 1, **options):
        plan = prepare_fork(
            self.store, run_id, step, base_url=f"{self.url}/v1",
            workdir=self.tmp / "forks", **options,
        )
        return plan, execute_forks(self.store, plan, n)


@contextmanager
def task_stack(
    tmp_path: Path, mock_server: Any, monkeypatch: Any, *, max_inflight: int = 4, max_output: int | None = None,
) -> Iterator[Stack]:
    """The tally task, runnable end to end with no model: `FakeAgent` as the harness (as
    "fake" in ADAPTERS), `tally_model` as the mock server's responder, a real proxy on a
    socket, and a skill folder beside the spec."""
    import httpx

    from tuningfork.config import Settings
    from tuningfork.harness import ADAPTERS
    from tuningfork.proxy.app import create_app

    monkeypatch.setitem(ADAPTERS, "fake", FakeAgent)
    mock_server.responder = tally_model
    home = tmp_path / "store"
    skill = tmp_path / "skills-src" / "tally-report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(ORIGINAL, encoding="utf-8")
    item = Eval(
        id="1",
        prompt="Summarize data.csv into report.json.",
        files=[{"path": "data.csv", "content": "id,amount\n1,10\n2,20\n3,30\n4,1.5\n"}],
        tuningfork=TuningforkOptions(check=CHECK, skills=["tally-report"], max_turns=6, timeout=60),
    )
    spec = TaskSpec(
        skill_name="tally-report", evals=[item], path=tmp_path / "spec.json",
        skill_dir="skills-src/tally-report",
    )
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    app = create_app(
        Settings(upstream_base_url="http://upstream.test", home=home, max_inflight=max_inflight, max_output=max_output),
        client=upstream,
    )
    store = Store(home)
    try:
        with serving(app) as url:
            yield Stack(url, store, app, mock_server, tmp_path, spec, item)
    finally:
        store.close()
        app.state.recorder.store.close()
        asyncio.run(upstream.aclose())
