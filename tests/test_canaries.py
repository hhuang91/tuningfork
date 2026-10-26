"""The canary sweep: a key planted in each input of the loop, run as a person runs it — batches
through the proxy with the fake agent, a scenario's among them, a chain told what its check
printed whose first link changes the skill its second is given, jobs, a held call edited and
asked about, a label queued, an import, a server's errors, a scenario made of a run, of the chat
and of the chain's second link, the chat's skill added and the link's — and every row and blob
the recordings folder's store holds, and every file beside it, searched for each key's start and
end.

Only what a run is given, kept as it was given by design, may hold one: a launch's eval and
variant, and a scenario's context, which a fork runs again as they were written; the files kept
as they were (`KEPT_AS_IT_WAS`): a launch's own folders, the workspace's snapshots; and, in a
copy of the skill a chain's second link was given (`GIVEN_COPIES`), the line its first wrote.
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
    "skill a chain's link edited", "skill an imported chat shows", "turn a scenario starts from",
    "password an imported answer repeats",
)
#: Keys no pattern reads alone: masked by a password's name in a call, then quoted bare by the
#: server's error — masked there only as a value the call's own mask took.
NAMED_ONLY = ("password a server's error quotes",)
#: Keys no pattern reads at all: masked by where they stand — a check's variable whose name is a
#: secret's, printed bare; a key tuningfork holds, which a check found in its file.
BY_PLACE = ("variable named as a secret", "key of tuningfork's own")
#: A line a chain's link wrote into its skill: what the chain's record shows of each link.
NOTED = "skill a chain's link edited"
#: A password a person gave an imported chat by its name, which the model's reasoning — never sent
#: back — repeats bare: masked there only as a value the masks before it took.
REPEATED = ("password an imported answer repeats",)
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
            key = ("Pw" if name in NAMED_ONLY + BY_PLACE + REPEATED else "sk-tfcanary") + f"{number:02d}" + "".join(
                draw.choices(string.ascii_letters + string.digits, k=ENDS))
            if name in BY_PLACE:
                if redact(f"said={key}") == f"said={key}":
                    break
            elif name in REPEATED:
                if key not in redact(f"password={key}") and redact(f"said {key}") == f"said {key}":
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
        # A chain's link asked to: its skill, in its home beside its workspace, given a line with a key.
        if len(results) == 2 and "note it in your skill" in str(messages[1].get("content") or ""):
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/../home/skills/tally-report/SKILL.md",
                "content": SKILL + f"\nThe key is {keys[NOTED]}.\n"}}])
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


#: What the recordings folder keeps beside its rows and blobs as it was, by design, and why: each
#: folder by its place in the recordings folder, a scenario's by its place in the scenario's.
KEPT_AS_IT_WAS = {
    "tuningfork.db": "the store's rows, searched row by row",
    "blobs": "the store's blobs, searched blob by blob",
    "snapshots.git": "the workspace at every call",
    "batches": "each launch's own folders, its workspace and its harness's home, and the skill each was "
               "given or left, the next link's start",
    "forks": "each fork's own folders, as a launch's",
    "variants": "each wording tried, as it was given",
    ".proposer-key": "tuningfork's own key",
    "scenarios/*/inputs": "a scenario's input files: a run's workspace where the scenario starts",
}
#: What a scenario's spec takes from the task of the run it was made of, as the task's file wrote it
#: (`launches.eval`, kept as given): its check, and the output and expectations it names. The rest
#: of the spec is searched.
GIVEN_TO_A_SCENARIO = ("check", "expected_output", "expectations")
#: What the sweep copies of the skill a launched run was given, kept as it was given, as the run's
#: own copy is (`batches`): a scenario's copy of what the run had (`copy_skill`), and a skill added
#: from the run — of a chain's second link, its skill holding the line its first wrote (`NOTED`),
#: and nothing else planted.
GIVEN_COPIES = ("scenarios/from-link/skill", "skills/from-link")


def kept_as_it_was(relative: Path) -> bool:
    top, *rest = relative.parts
    if top.startswith("tuningfork.db"):  # its -wal and -shm with it
        top = "tuningfork.db"
    return top in KEPT_AS_IT_WAS or (top == "scenarios" and len(rest) > 1 and f"scenarios/*/{rest[1]}" in KEPT_AS_IT_WAS)


def given_copy(relative: Path, name: str) -> bool:
    """Whether the key `name` stands in a file by design: the line a chain's first link wrote, in
    a copy of the skill its second was given (`GIVEN_COPIES`)."""
    return name == NOTED and any(relative.as_posix().startswith(f"{copy}/") for copy in GIVEN_COPIES)


def searched(home: Path, path: Path) -> bytes:
    """What the sweep reads of a file: all of it, but of a scenario's spec what it takes from its
    run's task as written (`GIVEN_TO_A_SCENARIO`)."""
    parts = path.relative_to(home).parts
    if len(parts) != 3 or parts[0] != "scenarios" or parts[2] != "evals.json":
        return path.read_bytes()
    spec = json.loads(path.read_text(encoding="utf-8"))
    for item in spec.get("evals") or []:
        for given in GIVEN_TO_A_SCENARIO:
            item.pop(given, None)
            (item.get("tuningfork") or {}).pop(given, None)
    return json.dumps(spec, ensure_ascii=False).encode("utf-8")


def found_in_files(home: Path, keys: dict[str, str]) -> list[str]:
    """Each file of the recordings folder a key's start or end stands in, but where kept as it was,
    or given as it stands there."""
    return [
        f"{name} in {path.relative_to(home).as_posix()}"
        for path in sorted(home.rglob("*")) if path.is_file() and not kept_as_it_was(path.relative_to(home))
        for name, key in keys.items()
        if not given_copy(path.relative_to(home), name) and any(piece.encode() in searched(home, path) for piece in pieces(key))
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
    # A chain's task: the same, its link asked to note it in the skill it hands on.
    chained = TaskSpec(skill_name="tally-report", path=tmp_path / "spec" / "chain.json", skill_dir="skills-src/tally-report",
                       evals=[Eval(id="3", prompt=f"{item.prompt} Then note it in your skill.", files=item.files,
                                   tuningfork=item.tuningfork)])
    chained.path.write_text(json.dumps(chained.to_dict()), encoding="utf-8")
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
            # A chain whose harness is told what the check printed: its first link changes the skill
            # its second is given.
            chain = job("chain", {"spec": str(chained.path), "n": 2, "harness": "fake", "model": "m", "tell": True})
            linked = chain["result"]["links"][1]["run_id"]
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
        # A chat had elsewhere, imported: a key said before it, one in the turn the skill came in
        # after, and one in the skill the model read.
        from tests.test_chat_import import SKILL as CHAT_SKILL, conversation, export
        from tuningfork.ops import addskill, imports

        rows = [{"id": 1001, "role": "user", "content": f"Remember my key {keys['imported chat']}.", "timestamp": 0.5},
                {"id": 1002, "role": "assistant", "content": "Noted.", "finish_reason": "stop", "timestamp": 0.6},
                {"id": 1003, "role": "user", "content": f"Log in with password={keys[REPEATED[0]]}.", "timestamp": 0.7},
                {"id": 1004, "role": "assistant", "content": "Logged in.", "finish_reason": "stop", "timestamp": 0.8,
                 "reasoning_content": f"The password is {keys[REPEATED[0]]}."}]
        chat = conversation(("tally-report", CHAT_SKILL + f"\nThe key is {keys['skill an imported chat shows']}.\n"))
        chat[0] = {**chat[0], "content": f"{chat[0]['content']} My key is {keys['turn a scenario starts from']}."}
        imported = imports.execute_import(store, imports.prepare_import(store, export(rows + chat)))
        # What is made of recordings: a scenario of the chat, and of the batch's run; the chat's skill added.
        imports.make_scenario(store, imports.prepare_scenario(store, imported.run_id, "from-import"))
        imports.make_scenario(store, imports.prepare_scenario(store, run_id, "from-run"))
        addskill.execute_add(store, addskill.prepare_from_run(store, imported.run_id, name="from-import"))
        # And of the chain's second link: a copy of what it had, and its skill added.
        imports.make_scenario(store, imports.prepare_scenario(store, linked, "from-link", copy_skill=True))
        addskill.execute_add(store, addskill.prepare_from_run(store, linked, name="from-link"))
    finally:
        store.close()
        asyncio.run(upstream.aclose())

    return found_in_rows(home / "tuningfork.db", keys) + found_in_blobs(home, keys) + found_in_files(home, keys)


def test_no_key_planted_in_a_run_is_kept_by_the_store(tmp_path, mock_server, monkeypatch):
    keys = canaries()
    found = swept(tmp_path, mock_server, monkeypatch, keys)
    assert found == [], "\n".join(found)
    # A copy of the skill a launched run was given holds it as the run was given it: the chain's
    # second link's, the line its first wrote in it. Found in review: the sweep made no such copy.
    for copy in GIVEN_COPIES:
        held = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "home" / copy).rglob("*") if path.is_file())
        assert keys[NOTED] in held, copy


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
    # Each file made of what was recorded holds what it was made of: the chain's record the line
    # its link wrote, the imported chat and its skill each its keys, and so does what was made of them.
    made = {re.sub(r"/[0-9a-f]{8,}", "/<id>", place) for place in found}
    assert {
        f"{NOTED} in chains/<id>.json",
        *(f"{name} in imports/<id>/chat.json"
          for name in ("imported chat", "skill an imported chat shows", "turn a scenario starts from", *REPEATED)),
        "skill an imported chat shows in imports/<id>/skills/tally-report/SKILL.md",
        "imported chat in scenarios/from-import/context.json",
        "turn a scenario starts from in scenarios/from-import/evals.json",
        "skill an imported chat shows in scenarios/from-import/skill/tally-report/SKILL.md",
        "skill an imported chat shows in skills/from-import/SKILL.md",
        "prompt in scenarios/from-run/evals.json",
    } <= made, sorted(made)
    # And each reply's cut does fall inside its key: where a trial keeps the cut part, its start
    # stands and its end does not.
    connection = sqlite3.connect(f"file:{tmp_path / 'home' / 'tuningfork.db'}?mode=ro", uri=True)
    try:
        trials = [json.loads(row[0]) for row in connection.execute(
            "SELECT result FROM jobs WHERE kind = 'check-trial' AND result IS NOT NULL")]
        (along,) = [json.loads(row[0])["check_along"] for row in connection.execute(
            "SELECT result FROM runs WHERE json_extract(result, '$.check_along') IS NOT NULL")]
    finally:
        connection.close()
    for name, field in (("reply cut to its quote", "output"), ("reply cut to what a trial shows", "reply")):
        start, end = pieces(keys[name])
        cut = [trial[field] for trial in trials if start in (trial.get(field) or "")]
        assert cut and not any(end in text for text in cut), name
    # So does where the run's last check's output is given from: check-along, given it whole, keeps
    # the key's end and not its start.
    start, end = pieces(keys["check output given from inside it"])
    last = "\n".join(along["states"][-1]["findings"])
    assert end in last and start not in last
