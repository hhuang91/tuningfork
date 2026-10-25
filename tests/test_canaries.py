"""The canary sweep: a key planted in each input of the loop, run as a person runs it — batches
through the proxy with the fake agent, a scenario's among them, a chain told what its check
printed, jobs, a held call edited and asked about, a label queued, an import, a server's errors —
and every row and blob the recordings folder's store holds, and a chain's record beside it,
searched for each key's start and end.

Only what a run is given, kept as it was given by design, may hold one: a launch's eval and
variant, and a scenario's context, which a fork runs again as they were written.
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

from tuningfork.access import PROPOSER_KEY_FILE, write_secret
from tuningfork.config import Settings
from tuningfork.control import ControlClient
from tuningfork.harness import ADAPTERS
from tuningfork.proxy.app import create_app
from tuningfork.ops.checks import REPLY_SHOWN
from tuningfork.specs import CHECK_KEPT, REPLY_QUOTED, Eval, TaskSpec, TuningforkOptions
from tuningfork.demo.mock import MockTurn
from tests.helpers import FakeAgent, serving

#: Where a key may stand by design: what a fork runs again as it was written.
KEPT_AS_GIVEN = {("launches", "eval"), ("launches", "variant")}

#: Each input a key is planted in.
PLANTED = (
    "input file", "written by the model", "final reply", "check command", "environment", "prompt",
    "sampled answer", "imported chat", "check trial reply", "check trial file", "server error",
    "server error event", "edit at a breakpoint", "upstream password", "label queued", "wording tried",
    "question at a held call", "scenario's context", "password a server's error quotes",
    "check output its cut falls inside", "variable named as a secret", "key of tuningfork's own",
    "reply cut to its quote", "reply cut to what a trial shows", "check output given from inside it",
)
#: Keys no pattern reads alone: masked by a password's name in a call, then quoted bare by the
#: server's error — masked there only as a value the call's own mask took.
NAMED_ONLY = ("password a server's error quotes",)
#: Keys no pattern reads at all: masked by where they stand — a check's variable whose name is a
#: secret's, printed bare; a key tuningfork holds, which a check found in its file.
BY_PLACE = ("variable named as a secret", "key of tuningfork's own")
#: What of a key the sweep looks for: its first characters and its last, each enough to name it.
#: Where a cut fell inside a key, its start or its end is what stands.
STARTS, ENDS = 16, 20


def pieces(key: str) -> tuple[str, str]:
    return key[:STARTS], key[-ENDS:]

SKILL = "---\nname: tally-report\n---\n\n# Tally report\n\nWrite the totals to config.json.\n"


def canaries() -> dict[str, str]:
    """A key a name, each its own: in a shape every pattern set masks whole, or, named only, a
    password's letters and digits, masked under a password's name. A draw the patterns would not
    mask so — found in review: one in a few thousand reads as a name — is drawn again."""
    from tuningfork.config import redact

    draw = random.Random()
    keys = {}
    for number, name in enumerate(PLANTED):
        while True:
            key = ("Pw" if name in NAMED_ONLY + BY_PLACE else "sk-tfcanary") + f"{number:02d}" + "".join(
                draw.choices(string.ascii_letters + string.digits, k=ENDS))
            if name in BY_PLACE:
                if redact(f"said={key}") == f"said={key}":
                    break
            elif key not in redact(f"password={key}" if name in NAMED_ONLY else key):
                break
        keys[name] = key
    return keys


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
        if last.startswith("deny"):
            return MockTurn(content="Checking", error_event={
                "code": 500, "message": f"login refused for {keys[NAMED_ONLY[0]]}", "type": "server_error"})
        if last.startswith("edit me"):
            return MockTurn(content="Edited.")
        if last.startswith("I ran my check"):
            return MockTurn(content="Noted.")
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
    """Each place a key's start or end stands in the database, as `<key> in <table>.<column>`, but
    where kept as given."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        places = []
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            for column in [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]:
                if (table, column) in KEPT_AS_GIVEN:
                    continue
                for name, key in keys.items():
                    if any(connection.execute(
                        f"SELECT 1 FROM {table} WHERE instr(CAST({column} AS TEXT), ?) > 0 LIMIT 1", (piece,)
                    ).fetchone() for piece in pieces(key)):
                        places.append(f"{name} in {table}.{column}")
        return places
    finally:
        connection.close()


def found_in_blobs(home: Path, keys: dict[str, str]) -> list[str]:
    """Each blob a key's start or end stands in, but a scenario's context, which its launches name."""
    connection = sqlite3.connect(f"file:{home / 'tuningfork.db'}?mode=ro", uri=True)
    try:
        given = {row[0] for row in connection.execute(
            "SELECT json_extract(settings, '$.context.blob') FROM launches"
            " WHERE json_extract(settings, '$.context.blob') IS NOT NULL")}
    finally:
        connection.close()
    return [
        f"{name} in blob {path.parent.name}{path.name}"
        for path in sorted((home / "blobs").rglob("*")) if path.is_file() and path.parent.name + path.name not in given
        for name, key in keys.items() if any(piece.encode() in path.read_bytes() for piece in pieces(key))
    ]


def found_in_files(home: Path, keys: dict[str, str]) -> list[str]:
    """Each file tuningfork writes beside its store from what was recorded that a key's start or end
    stands in: a chain's record."""
    return [
        f"{name} in {path.relative_to(home).as_posix()}"
        for path in sorted((home / "chains").glob("*.json"))
        for name, key in keys.items() if any(piece.encode() in path.read_bytes() for piece in pieces(key))
    ]


def swept(tmp_path: Path, mock_server, monkeypatch, keys: dict[str, str], patterns=None) -> list[str]:
    """The loop run once, a key in each input, recorded by a store masking with `patterns` (its
    own where None): where each key stands in its rows and blobs afterwards."""
    from tuningfork.proxy.recorder import Recorder
    from tuningfork.store.db import Store

    monkeypatch.setitem(ADAPTERS, "fake", FakeAgent)
    monkeypatch.setenv("TF_CANARY_ENVIRONMENT", keys["environment"])
    monkeypatch.setenv("TF_CANARY_TOKEN", keys["variable named as a secret"])
    mock_server.responder = model(keys)
    mock_server.props = {"default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 1.0}}}
    skill = tmp_path / "spec" / "skills-src" / "tally-report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL, encoding="utf-8")
    # A key tuningfork holds, which no check is given, but one finds in its file.
    home = tmp_path / "home"
    held = write_secret(home / PROPOSER_KEY_FILE, keys["key of tuningfork's own"])
    # The check starts on a key and ends on one, each with just enough around it that a place its
    # output is cut falls inside it: where the part of its output masked to keep its end starts
    # (`specs.MASKED_MOST`, set here to what that part holds of the run's last check: the check along
    # the run is given it whole), and the cut a verdict makes.
    window, cut = keys["check output given from inside it"], keys["check output its cut falls inside"]
    after_window, after_cut = 100, CHECK_KEPT - (len(cut) - 6)
    printed = "".join(f"{line}\n" for line in (
        json.dumps({"api_key": keys["written by the model"], "row_count": 4}), f"env={keys['environment']}",
        f"said={keys['variable named as a secret']}", keys["key of tuningfork's own"], f"literal={keys['check command']}"))
    monkeypatch.setattr("tuningfork.specs.MASKED_MOST",
                        len(window) - 6 + after_window + 1 + len(printed) + len(cut) + after_cut)
    check = (
        f"printf '%s' {window}; python3 -c \"import sys; sys.stdout.write('w' * {after_window})\"; echo; "
        f"cat config.json; echo; echo env=$TF_CANARY_ENVIRONMENT; echo said=$TF_CANARY_TOKEN; cat {held}; "
        f"echo literal={keys['check command']}; printf '%s' {cut}; "
        f"python3 -c \"import sys; sys.stdout.write('x' * {after_cut})\"; "
        "python3 -c \"import json,sys; d=json.load(open('config.json')); sys.exit(0 if d.get('row_count')==4 else 1)\""
    )
    # The check reads the planted variables as its task lets it (`check_env`): it gets no other
    # of the person's.
    item = Eval(
        id="1",
        prompt=f"Read .env and write config.json with row_count 4. My key is {keys['prompt']}.",
        files=[{"path": ".env", "content": f"OPENAI_API_KEY={keys['input file']}\n"}],
        tuningfork=TuningforkOptions(
            check=check, check_env=["TF_CANARY_ENVIRONMENT", "TF_CANARY_TOKEN"], skills=["tally-report"],
            max_turns=6, timeout=60,
        ),
    )
    spec = TaskSpec(skill_name="tally-report", evals=[item], path=tmp_path / "spec" / "spec.json",
                    skill_dir="skills-src/tally-report")
    spec.path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    # A scenario: the same task, continuing a conversation that holds a key.
    earlier = keys["scenario's context"]
    (tmp_path / "spec" / "context.json").write_text(json.dumps([
        {"role": "user", "content": f"Earlier: my API token is {earlier}."},
        {"role": "assistant", "content": "Noted."},
    ]), encoding="utf-8")
    scenario = TaskSpec(
        skill_name="tally-report", path=tmp_path / "spec" / "scenario.json", skill_dir="skills-src/tally-report",
        evals=[Eval(id="2", prompt=item.prompt, files=item.files, tuningfork=TuningforkOptions(
            check=check, check_env=["TF_CANARY_ENVIRONMENT", "TF_CANARY_TOKEN"], skills=["tally-report"],
            max_turns=6, timeout=60, context="context.json"))],
    )
    scenario.path.write_text(json.dumps(scenario.to_dict()), encoding="utf-8")
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
            # A batch run through the proxy: the ordinary recording, and its check; and a scenario's.
            (run_id,) = job("batch", {"spec": str(spec.path), "n": 1, "harness": "fake", "model": "m"})["links"]["runs"]
            job("batch", {"spec": str(scenario.path), "n": 1, "harness": "fake", "model": "m"})
            # A chain whose harness is told what the check printed.
            job("chain", {"spec": str(spec.path), "n": 1, "harness": "fake", "model": "m", "tell": True})
            # Jobs on it: a resample, an A/B of a wording, a fork to the end, the check tried and run along it.
            job("resample", {"run": run_id, "step": "1", "n": 2})
            job("ab", {"run": run_id, "step": "1", "n": 1, "exploratory": True, "variant_name": "canary",
                       "variant_text": SKILL + f"Never print {keys['wording tried']}.\n"})
            job("resample-to-end", {"run": run_id, "step": "1", "n": 1})
            job("check-trial", {"check": "cat notes.txt", "reply": f"my reply quotes {keys['check trial reply']}",
                                "files": [{"path": "notes.txt", "content": f"token {keys['check trial file']}"}]})
            # A reply check tried on a reply typed here, a cut falling inside a key each time: the
            # quote of the reply the check's output keeps, and what a trial shows of the reply.
            for cut_at, name in ((REPLY_QUOTED, "reply cut to its quote"), (REPLY_SHOWN, "reply cut to what a trial shows")):
                job("check-trial", {"check": {"reply_contains": "done"},
                                    "reply": "y" * (cut_at - 19) + " " + keys[name] + " done"})
            job("check-along", {"run": run_id})
            # Calls the server refuses, or cuts off, quoting a key.
            httpx.post(f"{url}/v1/chat/completions", timeout=30,
                       json={"model": "m", "messages": [{"role": "user", "content": "fail, please"}]})
            httpx.post(f"{url}/v1/chat/completions", timeout=30,
                       json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "cut it"}]})
            httpx.post(f"{url}/v1/chat/completions", timeout=30, json={"model": "m", "stream": True, "messages": [
                {"role": "user", "content": f"deny me, password={keys[NAMED_ONLY[0]]}"}]})
            # A label queued for the next run.
            store.label_next(f"tried with {keys['label queued']}")
            httpx.post(f"{url}/v1/chat/completions", timeout=30,
                       json={"model": "m", "messages": [{"role": "user", "content": "labelled"}]})
            # A call held before the model, and a key typed into it.
            client.add_breakpoint(phase="request", context_regex="edit me")
            sent: dict = {}
            caller = threading.Thread(target=lambda: sent.setdefault("answer", httpx.post(
                f"{url}/v1/chat/completions", timeout=60,
                json={"model": "m", "messages": [{"role": "user", "content": "edit me"}]})))
            caller.start()
            held = client.wait_for_hold(timeout=30, poll=0.1)
            assert held is not None
            job("hold-console", {"hold": held["id"], "question": f"What would you do with {keys['question at a held call']}?"})
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

    return found_in_rows(home / "tuningfork.db", keys) + found_in_blobs(home, keys) + found_in_files(home, keys)


def test_no_key_planted_in_a_run_is_kept_by_the_store(tmp_path, mock_server, monkeypatch):
    found = swept(tmp_path, mock_server, monkeypatch, canaries())
    assert found == [], "\n".join(found)


def test_every_key_planted_reaches_the_store_where_nothing_masks_it(tmp_path, mock_server, monkeypatch):
    """The sweep's control: with no patterns, and nothing masked by its place — a check's variable
    named as a secret's, a key tuningfork holds — each key is found, so each input does reach the
    store, and the sweep finding none means it was masked. An upstream's password is taken from its
    address by its place, whatever the patterns."""
    monkeypatch.setattr("tuningfork.config.secret_name", lambda name: False)
    monkeypatch.setattr("tuningfork.access.held_secrets", lambda home=None: [])
    monkeypatch.setattr("tuningfork.runner.held_secrets", lambda home=None: [])
    keys = canaries()
    found = swept(tmp_path, mock_server, monkeypatch, keys, patterns=())
    assert {place.split(" in ")[0] for place in found} == set(PLANTED) - {"upstream password"}
    # And each reply's cut does fall inside its key: where a trial keeps the cut part, its start
    # stands and its end does not.
    connection = sqlite3.connect(f"file:{tmp_path / 'home' / 'tuningfork.db'}?mode=ro", uri=True)
    try:
        trials = [json.loads(row[0]) for row in connection.execute(
            "SELECT result FROM jobs WHERE kind = 'check-trial' AND result IS NOT NULL")]
    finally:
        connection.close()
    for name, field in (("reply cut to its quote", "output"), ("reply cut to what a trial shows", "reply")):
        start, end = pieces(keys[name])
        cut = [trial[field] for trial in trials if start in (trial.get(field) or "")]
        assert cut and not any(end in text for text in cut), name
