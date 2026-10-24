"""The canary sweep: a key planted in every input a run takes, the loop run as a person runs it —
a batch through the proxy with the fake agent, jobs, a held call edited, an import, a server's
error — and every row and blob the recordings folder's store holds searched for each key.

Only what is kept as given by design may hold one: a launch's eval and variant, which a fork runs
again as they were written.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sqlite3
import string
import threading
from pathlib import Path

import httpx
import pytest

from tuningfork.config import Settings
from tuningfork.control import ControlClient
from tuningfork.harness import ADAPTERS
from tuningfork.proxy.app import create_app
from tuningfork.specs import Eval, TaskSpec, TuningforkOptions
from tests.helpers import FakeAgent, serving
from tests.mock_server import MockTurn

#: Where a key may stand by design: what a fork runs again as it was written.
KEPT_AS_GIVEN = {("launches", "eval"), ("launches", "variant")}

#: Each input a key is planted in.
PLANTED = (
    "input file", "written by the model", "final reply", "check command", "environment", "prompt",
    "sampled answer", "imported chat", "check trial reply", "check trial file", "server error",
    "server error event", "edit at a breakpoint", "upstream password",
)

SKILL = "---\nname: tally-report\n---\n\n# Tally report\n\nWrite the totals to config.json.\n"


def canaries() -> dict[str, str]:
    """A key a name, each its own, in a shape every pattern set masks whole."""
    draw = random.Random()
    return {
        name: f"sk-tfcanary{number:02d}" + "".join(draw.choices(string.ascii_letters + string.digits, k=20))
        for number, name in enumerate(PLANTED)
    }


def model(keys: dict[str, str]):
    """The model: read the input, write the config with a key in it, then say so with another —
    and a sample its own key; a call starting "fail" a refusal quoting one, one starting "cut" a stream
    that ends in an error quoting one, as llama.cpp's do."""

    def answer(payload: dict) -> MockTurn:
        messages = payload.get("messages") or []
        last = str(messages[-1].get("content") or "") if messages else ""
        if last.startswith("fail"):
            return MockTurn(status=400, error={"error": {
                "message": f"invalid api key {keys['server error']}", "type": "invalid_request_error"}})
        if last.startswith("cut"):
            return MockTurn(content="Writing", error_event={
                "code": 500, "message": f"Invalid diff: {keys['server error event']} not found", "type": "server_error"})
        if last.startswith("edit me"):
            return MockTurn(content="Edited.")
        system = str(messages[0].get("content") or "") if messages else ""
        found = re.search(r"Working directory: (\S+)", system)
        workdir = found.group(1) if found else "."
        results = [message for message in messages if message.get("role") == "tool"]
        if payload.get("seed") is not None and results:
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/config.json", "content": json.dumps({"api_key": keys["sampled answer"]})}}])
        if not results:
            return MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": f"{workdir}/.env"}}])
        if len(results) == 1:
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/config.json",
                "content": json.dumps({"api_key": keys["written by the model"], "row_count": 4})}}])
        return MockTurn(content=f"Done. I used the key {keys['final reply']}.")

    return answer


def found_in_rows(database: Path, keys: dict[str, str]) -> list[str]:
    """Each place a key stands in the database, as `<key> in <table>.<column>`, but where kept as given."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        places = []
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            for column in [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]:
                if (table, column) in KEPT_AS_GIVEN:
                    continue
                for name, key in keys.items():
                    if connection.execute(
                        f"SELECT 1 FROM {table} WHERE instr(CAST({column} AS TEXT), ?) > 0 LIMIT 1", (key,)
                    ).fetchone():
                        places.append(f"{name} in {table}.{column}")
        return places
    finally:
        connection.close()


def found_in_blobs(blobs: Path, keys: dict[str, str]) -> list[str]:
    return [
        f"{name} in blob {path.parent.name}{path.name}"
        for path in sorted(blobs.rglob("*")) if path.is_file()
        for name, key in keys.items() if key.encode() in path.read_bytes()
    ]


def swept(tmp_path: Path, mock_server, monkeypatch, keys: dict[str, str], patterns=None) -> list[str]:
    """The loop run once, a key in each input, recorded by a store masking with `patterns` (its
    own where None): where each key stands in its rows and blobs afterwards."""
    from tuningfork.proxy.recorder import Recorder
    from tuningfork.store.db import Store

    monkeypatch.setitem(ADAPTERS, "fake", FakeAgent)
    monkeypatch.setenv("TF_CANARY_ENVIRONMENT", keys["environment"])
    mock_server.responder = model(keys)
    mock_server.props = {"default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 1.0}}}
    skill = tmp_path / "spec" / "skills-src" / "tally-report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL, encoding="utf-8")
    check = (
        f"cat config.json; echo; echo env=$TF_CANARY_ENVIRONMENT; echo literal={keys['check command']}; "
        "python3 -c \"import json,sys; d=json.load(open('config.json')); sys.exit(0 if d.get('row_count')==4 else 1)\""
    )
    item = Eval(
        id="1",
        prompt=f"Read .env and write config.json with row_count 4. My key is {keys['prompt']}.",
        files=[{"path": ".env", "content": f"OPENAI_API_KEY={keys['input file']}\n"}],
        tuningfork=TuningforkOptions(check=check, skills=["tally-report"], max_turns=6, timeout=60),
    )
    spec = TaskSpec(skill_name="tally-report", evals=[item], path=tmp_path / "spec" / "spec.json",
                    skill_dir="skills-src/tally-report")
    spec.path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    home = tmp_path / "home"
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app), timeout=30)
    settings = Settings(upstream_base_url=f"http://tuningfork:{keys['upstream password']}@upstream.test", home=home,
                        max_inflight=4)
    store = Store(home, patterns)
    app = create_app(settings, client=upstream, recorder=Recorder(settings, store))
    try:
        with serving(app) as url, ControlClient(url, timeout=60) as client:
            monkeypatch.setenv("TUNINGFORK_PROXY", url)
            monkeypatch.setenv("TUNINGFORK_UPSTREAM", url)

            def job(kind: str, params: dict) -> dict:
                done = client.wait_for_job(client.submit(kind, params)["id"], timeout=120, poll=0.1)
                assert done["status"] == "done", (kind, done.get("error"))
                return done

            # The server the proxy sends its calls to, read and kept, its address with a password in it.
            assert client.switch_upstream(settings.upstream_base_url)["switched"]
            # A batch run through the proxy: the ordinary recording, and its check.
            (run_id,) = job("batch", {"spec": str(spec.path), "n": 1, "harness": "fake", "model": "m"})["links"]["runs"]
            # Jobs on it: a resample, a fork to the end, the check tried and run along it.
            job("resample", {"run": run_id, "step": "1", "n": 2})
            job("resample-to-end", {"run": run_id, "step": "1", "n": 1})
            job("check-trial", {"check": "cat notes.txt", "reply": f"my reply quotes {keys['check trial reply']}",
                                "files": [{"path": "notes.txt", "content": f"token {keys['check trial file']}"}]})
            job("check-along", {"run": run_id})
            # Calls the server refuses, or cuts off, quoting a key.
            httpx.post(f"{url}/v1/chat/completions", timeout=30,
                       json={"model": "m", "messages": [{"role": "user", "content": "fail, please"}]})
            httpx.post(f"{url}/v1/chat/completions", timeout=30,
                       json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "cut it"}]})
            # A call held before the model, and a key typed into it.
            client.add_breakpoint(phase="request", context_regex="edit me")
            sent: dict = {}
            caller = threading.Thread(target=lambda: sent.setdefault("answer", httpx.post(
                f"{url}/v1/chat/completions", timeout=60,
                json={"model": "m", "messages": [{"role": "user", "content": "edit me"}]})))
            caller.start()
            held = client.wait_for_hold(timeout=30, poll=0.1)
            assert held is not None
            edited = {"model": "m", "messages": [{"role": "user", "content": f"edit me: {keys['edit at a breakpoint']}"}]}
            client.decide(held["id"], "continue", request=edited)
            caller.join(60)
            assert sent["answer"].status_code == 200
        # A chat had elsewhere, imported.
        from tests.test_chat_import import SKILL as CHAT_SKILL, conversation, export
        from tuningfork.ops import imports

        rows = [{"id": 1001, "role": "user", "content": f"Remember my key {keys['imported chat']}.", "timestamp": 0.5},
                {"id": 1002, "role": "assistant", "content": "Noted.", "finish_reason": "stop", "timestamp": 0.7}]
        imports.execute_import(store, imports.prepare_import(store, export(rows + conversation(("tally-report", CHAT_SKILL)))))
    finally:
        store.close()
        asyncio.run(upstream.aclose())

    return found_in_rows(home / "tuningfork.db", keys) + found_in_blobs(home / "blobs", keys)


def test_no_key_planted_in_a_run_is_kept_by_the_store(tmp_path, mock_server, monkeypatch):
    found = swept(tmp_path, mock_server, monkeypatch, canaries())
    assert found == [], "\n".join(found)


def test_every_key_planted_reaches_the_store_where_nothing_masks_it(tmp_path, mock_server, monkeypatch):
    """The sweep's control: with no patterns, each key is found — so each input does reach the store,
    and the sweep finding none means the store masked it. An upstream's password is taken from its
    address by its place, whatever the patterns."""
    found = swept(tmp_path, mock_server, monkeypatch, canaries(), patterns=())
    assert {place.split(" in ")[0] for place in found} == set(PLANTED) - {"upstream password"}
