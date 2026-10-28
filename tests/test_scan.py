"""The scan (`ops.scan`): a recordings folder read for what today's masking would mask — what an
older tuningfork kept in clear — and where it stands, run by run, never the key itself; run only
when a person asks, changing nothing but its mark. And the mark the store keeps of the first time
a tuningfork that keeps marks opened a folder, which `serve` and `ui` read to say, from one count,
that the folder holds runs an older one recorded."""

from __future__ import annotations

import json
import shlex
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.config import REDACTED
from tuningfork.ops.common import Cancel, Cancelled
from tuningfork.ops.scan import NOT_RECORDED, OWNED, _Reader, older, older_note, scan
from tuningfork.store.db import OPENED, SCANNED, Store
from tests.helpers import KEY, as_response, record_run, text_message
from tests.test_canaries import canaries, pieces, swept

#: What the scan never reads, whatever made it: a copy of the skill a launched run was given, kept as
#: it was given — a skill added from the chain's second link, and a scenario's copy of what it had.
NOT_READ = ("skills/from-link/", "scenarios/from-link/skill/")
#: Keys no pattern reads: a check's variable named as a secret's, printed bare, which only its
#: place masks; and what a check's cut left of a key it fell inside, its end alone — as 0.6 kept it.
NO_PATTERN_READS = ("variable named as a secret", "check output its cut falls inside")


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    runner = CliRunner()

    def invoke(*args: str):
        result = runner.invoke(app, [str(arg) for arg in args])
        assert result.exception is None or isinstance(result.exception, SystemExit), result.output
        return result

    return invoke


def read_whole(home: Path) -> _Reader:
    with Store(home) as store:
        reader = _Reader(store, None, None)
        reader.run()
        return reader


def as_if_older(home: Path) -> None:
    """The folder as a tuningfork that kept no marks left it: what it holds is older than the
    first time one that keeps them opens it."""
    connection = sqlite3.connect(home / "tuningfork.db")
    try:
        connection.execute("DELETE FROM marks")
        connection.commit()
    finally:
        connection.close()


def older_folder(home: Path, *contents: str) -> str:
    """A folder an older tuningfork recorded, its masking reaching nothing: one run, each of
    `contents` an answer of it. Its id."""
    with Store(home, patterns=()) as store:
        run = record_run(store, "old", [text_message(content) for content in contents], prompt="tidy the csv")
    as_if_older(home)
    return run


def test_a_folder_this_tuningfork_recorded_holds_nothing_the_scan_finds(tmp_path, mock_server, monkeypatch):
    """The whole loop, a key in each of its inputs, recorded as this tuningfork records: the store
    masked each, so the scan finds none — and nothing else, where nothing else holds one."""
    assert swept(tmp_path, mock_server, monkeypatch, canaries()) == []
    with Store(tmp_path / "home") as store:
        report = scan(store)
    assert not report.found, report.to_json()
    assert report.unread == []


def test_every_key_an_older_tuningfork_kept_is_found_where_it_stands_and_never_said(
    tmp_path, mock_server, monkeypatch, cli,
):
    """The same loop recorded by a store that masked nothing, as an older tuningfork left what its
    masking did not reach: every place a key stands in a row, a blob or a file the store writes is
    one the scan finds — but a key no pattern reads (`NO_PATTERN_READS`). What it says, as text or
    JSON, holds none of them: where, never what."""
    monkeypatch.setattr("tuningfork.config.secret_name", lambda name: False)
    monkeypatch.setattr("tuningfork.access.held_secrets", lambda home=None: [])
    monkeypatch.setattr("tuningfork.runner.held_secrets", lambda home=None: [])
    keys = canaries()
    found = swept(tmp_path, mock_server, monkeypatch, keys, patterns=())
    monkeypatch.undo()
    home = tmp_path / "home"
    reader = read_whole(home)
    rows = {(key[1], key[3]) for key in reader.hits if key[0] == "row"}
    blobs = {key[1] for key in reader.hits if key[0] in ("blob", "message")}
    files = {Path(key[1]).relative_to(home).as_posix() for key in reader.hits if key[0] == "file"}
    missed = []
    for place in found:
        name, where = place.split(" in ", 1)
        if " in MCP " in place or name in NO_PATTERN_READS:
            continue
        if where.startswith("blob "):
            covered = where[len("blob "):] in blobs
        elif "." in where and "/" not in where and not where.endswith(".json"):
            covered = tuple(where.split(".", 1)) in rows
        else:
            covered = where.startswith(NOT_READ) or where in files
        if not covered:
            missed.append(place)
    assert missed == [], "\n".join(missed)

    told = cli("scan", "--home", home).output + cli("scan", "--home", home, "--json").output
    said = [name for name, key in keys.items() if any(piece in told for piece in pieces(key))]
    assert said == []
    report = json.loads(cli("scan", "--home", home, "--json").output)
    by_how = {}
    for run in report["runs"]:
        by_how.setdefault(run["recorded"], []).append(run)
    # A launched run's answer holding a key is one a fork past its step gives its harness again;
    # what nothing launched forks nowhere.
    assert any(run["replays_from"] is not None for run in by_how["launched"])
    assert by_how["live"] and all(run["replays_from"] is None for how in ("imported", "live") for run in by_how[how])
    (chat,) = by_how["imported"]
    assert {"chat", "messages", "answers"} <= set(chat["places"])
    # A launch's check, told and kept, is its run's; an experiment's samples are the run's they sampled.
    assert any({"result", "experiments"} <= set(run["places"]) for run in by_how["launched"])
    owners = {found["owner"] for found in report["elsewhere"]}
    assert owners >= {"job", "chain", "refinement", "scenario"} and not owners & {"launch", "experiment", "record"}
    printed = cli("scan", "--home", home).output
    assert "chains/" in printed
    # Each command it prints is one the trash takes: a fork an experiment drew goes with its experiment.
    commands = [shlex.split(line.split("tuningfork ", 1)[1]) for line in printed.splitlines() if line.startswith("set these")]
    assert {command[1] for command in commands} >= {"run", "experiment", "job", "refinement", "scenario"}
    for command in commands:
        done = cli(*command)
        assert done.exit_code == 0, done.output


def test_every_table_is_read_or_said_to_hold_no_recording(tmp_path):
    """A table the store gains is read, or named here as holding no recording, before it ships."""
    with Store(tmp_path / "home") as store:
        assert set(store.tables()) == set(OWNED) | NOT_RECORDED


def test_the_scan_changes_nothing_but_its_mark(tmp_path):
    home = tmp_path / "home"
    older_folder(home, f"Done. I used the key {KEY}.")

    def everything() -> tuple[dict, dict]:
        connection = sqlite3.connect(f"file:{home / 'tuningfork.db'}?mode=ro", uri=True)
        try:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
            rows = {table: sorted(map(repr, connection.execute(f"SELECT * FROM {table}")))
                    for table in tables if table != "marks"}
        finally:
            connection.close()
        files = {path.relative_to(home).as_posix(): path.read_bytes()
                 for path in home.rglob("*") if path.is_file() and not path.name.startswith("tuningfork.db")}
        return rows, files

    before = everything()
    with Store(home) as store:
        report = scan(store)
        assert report.found and store.mark(SCANNED) is not None
    assert everything() == before


#: A key a stream sent a piece an event, as the model wrote it.
CHUNKS = ["The key is ", KEY[:9], KEY[9:16], KEY[16:]]


def in_pieces(chunks: list[str]) -> bytes:
    """A chat completion streamed a piece an event."""
    return ("".join(
        "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": "m", "choices": [
            {"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}) + "\n\n" for chunk in chunks
    ) + "data: [DONE]\n\n").encode()


def test_a_key_a_stream_sent_in_pieces_is_found_in_the_answer_s_raw_bytes(tmp_path):
    """Before 0.6.0 an answer's text was masked, and its raw bytes kept as they came: a key the
    stream sent a piece an event stood in no event whole. Read back as the wire reads them, the
    bytes hold it — a stream known by its bytes where its step does not say it streamed."""
    from tests.helpers import restamp

    home = tmp_path / "home"
    raw = in_pieces(CHUNKS).decode()
    with Store(home, patterns=()) as store:
        step = store.record_step(
            request={"model": "m", "stream": True, "messages": [{"role": "user", "content": "the key?"}]},
            response=as_response(text_message("The key is <redacted>")),
            raw_response=raw.encode(), streamed=True, wire="openai-chat",
        )
    assert KEY not in raw and all(chunk in raw for chunk in CHUNKS)
    reader = read_whole(home)
    assert set(reader.hits) == {("blob", step.raw_response_blob, "raw")}
    with Store(home) as store:
        (run,) = scan(store).runs
        assert run["places"] == {"answers": 1} and run["first_step"] == {"answers": 0}
        restamp(store, step.run_id, {})
    assert set(read_whole(home).hits) == {("blob", step.raw_response_blob, "raw")}
    # A stream a comment opens, as a server's keep-alive does.
    commented = b": ping\n\n" + raw.encode()
    with Store(home, patterns=()) as store:
        other = store.record_step(
            request={"model": "m", "stream": True, "messages": [{"role": "user", "content": "and again?"}]},
            response=as_response(text_message("The key is <redacted>")),
            raw_response=commented, streamed=True, wire="openai-chat",
        )
        restamp(store, other.run_id, {})
    assert ("blob", other.raw_response_blob, "raw") in read_whole(home).hits


def test_a_run_says_where_it_holds_a_key_and_from_which_step(tmp_path):
    """An answer holding a key, and every request after it that carries it as a message."""
    home = tmp_path / "home"
    older_folder(home, f"The key is {KEY}.", "Done.", "Done again.")
    with Store(home) as store:
        (found,) = scan(store).runs
    assert found["places"] == {"messages": 1, "answers": 1}
    assert found["first_step"] == {"answers": 0, "messages": 1}
    assert found["replays_from"] is None and found["recorded"] == "live"  # nothing launched it: it forks nowhere


def test_a_fork_past_a_launched_run_s_answer_holding_a_key_gives_it_again(tmp_path):
    """Of a launched run, the first step whose answer holds a key — in its raw bytes alone, or in
    its text alone — is one a fork past it gives its harness again."""
    from tests.helpers import record_launched

    home = tmp_path / "home"
    first = {"model": "m", "messages": [{"role": "user", "content": "go"}]}
    second = {**first, "messages": [*first["messages"], text_message("Reading."), {"role": "user", "content": "and?"}]}
    with Store(home, patterns=()) as store:
        raw_only, text_only = (store.create_launch(kind="batch", task_id="t/1") for _ in range(2))
        for launch in (raw_only, text_only):
            record_launched(store, launch.id, first, response=as_response(text_message("Reading.")))
        record_launched(store, raw_only.id, {**second, "stream": True}, response=as_response(text_message("It is <redacted>")),
                        raw_response=in_pieces(CHUNKS), streamed=True, wire="openai-chat")
        record_launched(store, text_only.id, second, response=as_response(text_message(f"It is {KEY}")))
    with Store(home) as store:
        report = scan(store)
    assert [(run["recorded"], run["replays_from"]) for run in report.runs] == [("launched", 1), ("launched", 1)]
    # Their launches still running, the trash would refuse them: no command yet.
    assert all(run["running"] for run in report.runs) and report.commands() == []


def test_a_launch_s_record_is_its_main_run_s_whichever_began_first(tmp_path):
    with Store(tmp_path / "home") as store:
        reader = _Reader(store, None, None)
        for run, role in (("thread", "side"), ("main", "main"), ("later", "subagent")):
            reader._names("runs", {"id": run, "launch_id": "L", "role": role}, ("run", run), None)
    assert reader.run_of_launch == {"L": "main"}


def test_a_thread_is_set_aside_with_its_launch_s_main_run(cli, tmp_path):
    """The trash takes a thread only with its run: the command the scan prints names that run."""
    from tests.helpers import record_launched

    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        launch = store.create_launch(kind="batch", task_id="t/1")
        record_launched(store, launch.id, {"model": "m", "messages": [{"role": "user", "content": "go"}]},
                        response=as_response(text_message("Delegating.")))
        record_launched(store, launch.id, {"model": "m", "messages": [{"role": "user", "content": f"use {KEY}"}]},
                        response=as_response(text_message("Done.")))
        store.end_launch(launch.id)
        main, thread = store.launch_runs(launch.id)
    assert thread.is_thread and not main.is_thread
    printed = cli("scan", "--home", home).output
    assert f"run {thread.id[:8]}" in printed
    (line,) = [line for line in printed.splitlines() if line.startswith("set these")]
    assert line.endswith(f"tuningfork trash run {main.id} --home {home}")
    done = cli(*shlex.split(line.split("tuningfork ", 1)[1]))
    assert done.exit_code == 0, done.output


def test_what_a_launch_was_told_is_its_run_s(tmp_path):
    """A launch's record — what its harness was told of the check — is its run's check result."""
    from tests.helpers import record_launched

    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        launch = store.create_launch(kind="batch", task_id="t/1")
        record_launched(store, launch.id, {"model": "m", "messages": [{"role": "user", "content": "go"}]},
                        response=as_response(text_message("Done.")))
        store.end_launch(launch.id, exit={"told": f"the check printed {KEY}"})
    with Store(home) as store:
        report = scan(store)
    (found,) = report.runs
    assert found["places"] == {"result": 1} and report.elsewhere == []


def test_a_key_tuningfork_holds_is_found_wherever_it_stands(tmp_path):
    """No pattern reads it: it is found as the key it is, as the store masks it."""
    from tuningfork.access import PROPOSER_KEY_FILE, write_secret
    from tuningfork.config import redact

    held = "Pw00heldbytuningfork77Zq"
    assert redact(f"Using {held} now.") == f"Using {held} now."
    home = tmp_path / "home"
    older_folder(home, f"Using {held} now.")
    write_secret(home / PROPOSER_KEY_FILE, held)
    with Store(home) as store:
        (found,) = scan(store).runs
    assert found["places"] == {"answers": 1}


def test_a_key_in_a_server_s_address_kept_before_is_found(tmp_path, monkeypatch):
    """An older tuningfork kept a server's address with its query as it was given — a server read,
    an experiment sent there: masked whole today (`config.masked_address`), what it held is found
    though no pattern reads it, and so is the value wherever else it stands, as a mask's is. Kept
    today, nothing is."""
    from tests.test_store import request_with

    # Under a name no pattern reads as a secret's, and each its own, so neither is found as the
    # other's value: the job names the server's alone.
    address = "http://127.0.0.1:8081/v1?k={}"

    def keep(home: Path) -> tuple[str, str, str]:
        with Store(home, patterns=()) as store:
            server = store.add_server({"upstream": address.format("Q7bX2kLm9Zr4"), "models": ["m"]})
            experiment = store.create_experiment(kind="resample", arm="original", run_id=None, step_idx=None,
                                                 step_id=None, request=request_with([text_message("hi")]),
                                                 settings={}, base_url=address.format("W3nT8vR2pL6x"))
            job = store.create_job(kind="server-check", lane="interactive", params={"said": "with Q7bX2kLm9Zr4"})
        return server, experiment.id, job.id

    with monkeypatch.context() as before:
        before.setattr("tuningfork.config.without_credentials", lambda url: url)
        server, experiment, job = keep(tmp_path / "older")
    as_if_older(tmp_path / "older")
    with Store(tmp_path / "older") as store:
        found = {(each["owner"], each["id"]) for each in scan(store).elsewhere}
    assert found == {("server", server), ("experiment", experiment), ("job", job)}
    keep(tmp_path / "now")
    with Store(tmp_path / "now") as store:
        assert scan(store).elsewhere == []


def test_an_address_kept_masked_before_makes_no_mask_a_key(tmp_path, monkeypatch):
    """An address 0.6 or 0.7 kept had its user and password masked: today's masking takes the rest
    of it, and gives the scan no mask to look for (found in review: every run holding a mask read
    as holding a key)."""
    home = tmp_path / "home"
    run = older_folder(home, f"the key is {REDACTED} now")
    with monkeypatch.context() as before:
        before.setattr("tuningfork.config.without_credentials", lambda url: url)
        with Store(home, patterns=()) as store:
            server = store.add_server({"upstream": f"http://{REDACTED}@127.0.0.1:8081/v1?x=1", "models": ["m"]})
    as_if_older(home)
    with Store(home) as store:
        report = scan(store)
    assert [(each["owner"], each["id"]) for each in report.elsewhere] == [("server", server)]
    assert report.runs == [] and run


def test_a_word_or_a_date_an_address_held_is_looked_for_nowhere_else(tmp_path, monkeypatch):
    """Masked whole in the address, as a query's values and a password are, but looked for
    elsewhere only where it reads as a key (`config.key_like`): a run that names the same word or
    date holds no key (found in review: both runs below read as holding one)."""
    home = tmp_path / "home"
    older_folder(home, "it is the sunshine state", "released on 2024-02-01", "write to me@example.com")
    with monkeypatch.context() as before:
        before.setattr("tuningfork.config.without_credentials", lambda url: url)
        with Store(home, patterns=()) as store:
            server = store.add_server({"upstream": "http://admin:sunshine@127.0.0.1:8081/v1?api-version=2024-02-01"
                                                   "&user=me@example.com", "models": ["m"]})
    as_if_older(home)
    with Store(home) as store:
        report = scan(store)
    assert [(each["owner"], each["id"]) for each in report.elsewhere] == [("server", server)] and report.runs == []


def test_a_value_in_an_address_the_patterns_read_is_looked_for_elsewhere(tmp_path, monkeypatch):
    """Where the patterns find a key in an address kept before, its query's values are looked for
    too, as wherever else a mask's value is."""
    address = f"http://127.0.0.1:8081/{KEY}/v1?k=Q7bX2kLm9Zr4"
    home = tmp_path / "home"
    with monkeypatch.context() as before:
        before.setattr("tuningfork.config.without_credentials", lambda url: url)
        with Store(home, patterns=()) as store:
            server = store.add_server({"upstream": address, "models": ["m"]})
            job = store.create_job(kind="server-check", lane="interactive", params={"said": "with Q7bX2kLm9Zr4"})
    as_if_older(home)
    with Store(home) as store:
        found = {(each["owner"], each["id"]) for each in scan(store).elsewhere}
    assert found == {("server", server), ("job", job.id)}


def test_a_run_s_label_is_said_masked_of_what_the_masks_took(cli, tmp_path):
    """Found in review: a password a prompt named, bare in the run's label, was said as it is."""
    password = "Tr0ub4dorXyz9"
    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        record_run(store, f"retry {password} login", [text_message("Trying.")], prompt=f"log in with password={password}")
    as_if_older(home)
    told = cli("scan", "--home", home).output + cli("scan", "--home", home, "--json").output
    assert "retry " in told and password not in told


def test_a_message_every_run_holds_is_found_in_each(tmp_path):
    """A key in the prompt every run of a task shares: each run's messages hold it, from its first
    step. Found in review: the query read the message tree again for every step."""
    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        made = [record_run(store, f"run {number}", [text_message(f"Step {step}.") for step in range(4)],
                           prompt=f"tidy the csv with {KEY}") for number in range(5)]
    with Store(home) as store:
        found = scan(store).runs
    assert sorted(run["run"] for run in found) == sorted(made)
    assert all(run["places"] == {"messages": 1} and run["first_step"] == {"messages": 0} for run in found)


def test_a_skill_made_from_a_chat_is_read_and_one_a_person_brought_is_not(cli, tmp_path):
    """Before 0.7.0 the SKILL.md an import extracted from its chat, and a skill added from it, were
    written as the chat showed them, unmasked: each is read, and the skill added still is once the
    chat is deleted for good. A skill a person brought, or a folder they named at the import, is kept
    as given: never read."""
    from tuningfork.ops import addskill, imports, trash
    from tests.test_chat_import import SKILL as CHAT_SKILL, conversation, export

    home = tmp_path / "home"
    named = tmp_path / "named" / "csv-tidy"
    named.mkdir(parents=True)
    (named / "SKILL.md").write_text(f"---\nname: csv-tidy\n---\n\nSort the rows with {KEY}.\n", encoding="utf-8")
    mine = tmp_path / "mine" / "own-skill"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text(f"---\nname: own-skill\n---\n\nUse {KEY} in the demo.\n", encoding="utf-8")
    with Store(home, patterns=()) as store:
        chat = conversation(("tally-report", CHAT_SKILL + f"\nThe key is {KEY}.\n"), ("csv-tidy", "Sort the rows."))
        imported = imports.execute_import(store, imports.prepare_import(
            store, export(chat), skill="tally-report", skill_dir=[named]))
        addskill.execute_add(store, addskill.prepare_from_run(store, imported.run_id, skill="tally-report", name="from-chat"))
        addskill.execute_add(store, addskill.prepare_copy(store, mine, name="own-skill"))
    reader = read_whole(home)
    files = {Path(key[1]).relative_to(home).as_posix() for key in reader.hits if key[0] == "file"}
    run = imported.run_id
    assert f"imports/{run}/chat.json" in files and "skills/from-chat/SKILL.md" in files
    assert any(path.startswith(f"imports/{run}/skills/") and "tally" in path for path in files)
    assert not any("csv-tidy" in path or "own-skill" in path for path in files), files
    with Store(home) as store:
        trash.trash(store, "run", run)
        trash.purge(store, "run", run)
        report = scan(store)
    assert [(found["owner"], found["id"], found["by_hand"]) for found in report.elsewhere] == [
        ("skill", "from-chat", ["skills/from-chat/", "skills/from-chat.json"])]
    # Nothing the trash takes is left, so no deletion from it is offered.
    told = cli("scan", "--home", home).output
    assert "remove skills/from-chat/, skills/from-chat.json by hand" in told and "trash purge" not in told
    # An approval took the key out of it: the text it replaced, and the changelog's lines, kept
    # beside it as an approval keeps them, hold it still.
    skills = home / "skills"
    old = (skills / "from-chat" / "SKILL.md").read_text(encoding="utf-8")
    (skills / "from-chat.history").mkdir()
    (skills / "from-chat.history" / "SKILL.20261027-000000-abcd1234.md").write_text(old, encoding="utf-8")
    (skills / "from-chat.changelog.md").write_text(f"- The key is {KEY}.\n", encoding="utf-8")
    (skills / "from-chat" / "SKILL.md").write_text(old.replace(KEY, "a key"), encoding="utf-8")
    with Store(home) as store:
        (skill,) = scan(store).elsewhere
    assert skill["places"] == 2 and skill["by_hand"] == [
        "skills/from-chat/", "skills/from-chat.json", "skills/from-chat.history/", "skills/from-chat.changelog.md"]
    # Forgotten, it is moved aside whole, its record beside it: still read.
    with Store(home) as store:
        addskill.forget(store, "from-chat")
        report = scan(store)
    ((owner, ident, places, by_hand),) = [(found["owner"], found["id"], found["places"], found["by_hand"])
                                          for found in report.elsewhere]
    assert owner == "skill" and ident.startswith(".forgotten/from-chat-") and places == 2 and by_hand == [f"skills/{ident}/"]


def test_what_a_person_gave_a_refinement_is_not_read(tmp_path):
    """A refinement keeps the skill it refines, the edit a person approved and their own proposals
    as given: read, only what a proposing model or an agent made of the recordings is."""
    home = tmp_path / "home"
    Store(home).close()
    folder = home / "refinements" / "9f8e7d6c5b4a"
    (folder / "proposals").mkdir(parents=True)
    for name in ("baseline.md", "approval.json", "applied.json"):
        (folder / name).write_text(f"api_key = {KEY}\n", encoding="utf-8")
    (folder / "proposals" / "aaaa.json").write_text(json.dumps(
        {"proposers": [{"via": "person"}], "rationale": f"keep {KEY}"}), encoding="utf-8")
    (folder / "proposals" / "bbbb.json").write_text(json.dumps(
        {"proposers": [{"via": "builtin"}], "rationale": f"keep {KEY}"}), encoding="utf-8")
    reader = read_whole(home)
    assert {Path(key[1]).name for key in reader.hits if key[0] == "file"} == {"bbbb.json"}


def test_a_scenario_in_the_trash_is_read_and_said_to_be_there(cli, tmp_path):
    home = tmp_path / "home"
    Store(home).close()
    for folder in (home / "scenarios" / "s2", home / "scenarios" / ".trash" / "s1-20261027-185722"):
        folder.mkdir(parents=True)
        (folder / "context.json").write_text(json.dumps([{"role": "user", "content": f"key {KEY}"}]), encoding="utf-8")
    with Store(home) as store:
        report = scan(store)
    assert [(found["id"], found["trashed"]) for found in report.elsewhere] == [("s1-20261027-185722", True), ("s2", False)]
    assert report.commands() == [{"kind": "scenario", "ids": ["s2"]}]
    told = cli("scan", "--home", home).output
    assert "scenario s1-20261027-185722" in told and "(in the trash)" in told


def test_a_job_in_the_trash_or_still_running_gets_no_command(tmp_path):
    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        waiting = store.create_job(kind="resample", lane="interactive", params={"note": f"key {KEY}"})
        done = store.create_job(kind="resample", lane="interactive", params={"note": f"key {KEY}"})
        store.update_job(done.id, status="done")
        thrown = store.create_job(kind="resample", lane="interactive", params={"note": f"key {KEY}"})
        store.update_job(thrown.id, status="done")
        store.trash_job(thrown.id)
    with Store(home) as store:
        report = scan(store)
    states = {found["id"]: (found["trashed"], found["running"]) for found in report.elsewhere}
    assert states == {waiting.id: (False, True), done.id: (False, False), thrown.id: (True, False)}
    assert report.commands() == [{"kind": "job", "ids": [done.id]}]


def test_a_local_default_under_a_secret_s_name_makes_no_word_a_key(tmp_path, monkeypatch):
    """Found in review: the values of this process's variables named as secrets' were looked for as
    keys, and `PGPASSWORD=postgres` made a word one in every recording that said it."""
    for name, value in (("PGPASSWORD", "postgres"), ("MYSQL_PWD", "password"), ("DB_PASSWORD", "changeme"),
                        ("OPENAI_API_KEY", "sk-no-key-required")):
        monkeypatch.setenv(name, value)
    home = tmp_path / "home"
    with Store(home) as store:
        record_run(store, "db setup", [text_message("Set up postgres; the server takes sk-no-key-required.")],
                   prompt="Set up the postgres database for the app; its password is changeme.")
    with Store(home) as store:
        assert scan(store).runs == []


def test_only_the_store_s_own_tables_are_read(tmp_path):
    with Store(tmp_path / "home") as store, pytest.raises(ValueError):
        next(store.table_rows("runs; DROP TABLE runs"))


def test_the_store_is_read_a_page_at_a_time_to_its_end(tmp_path, monkeypatch):
    home = tmp_path / "home"
    older_folder(home, *(f"Key {number}: {KEY}" for number in range(5)))
    monkeypatch.setattr(Store, "PAGE", 2)
    with Store(home) as store:
        report = scan(store)
    (found,) = report.runs
    assert found["places"]["answers"] == 5
    connection = sqlite3.connect(home / "tuningfork.db")
    try:
        rows = sum(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in OWNED)
    finally:
        connection.close()
    assert report.read["records"] == rows > 5


def test_a_key_only_the_harness_sent_is_found_in_its_own_request(tmp_path):
    """An edit at a breakpoint took a key out of what the model was sent: the harness's own
    request, kept beside it, holds it still."""
    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        step = store.record_step(
            request={"model": "m", "messages": [{"role": "user", "content": "log in"}]},
            harness_request={"model": "m", "messages": [{"role": "user", "content": f"log in with {KEY}"}]},
            response=as_response(text_message("Done.")),
        )
    with Store(home) as store:
        (found,) = scan(store).runs
    assert found["places"] == {"messages": 1} and found["first_step"] == {"messages": step.idx}


def test_a_key_the_model_wrote_over_several_tokens_is_found_in_its_token_stream(tmp_path):
    from tuningfork.analysis.tokens import from_entries
    from tests.test_store_masking import entries

    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        step = store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "the key?"}]},
                                 response=as_response(text_message("It is <redacted>")))
        record = store.add_token_stream(from_entries(entries("It is ", KEY[:9], KEY[9:16], KEY[16:])),
                                        source="logprobs", step_id=step.id)
    reader = read_whole(home)
    assert set(reader.hits) == {("blob", record.blob, "tokens")}


def test_a_value_a_mask_took_is_found_where_it_stands_bare(tmp_path):
    """A password masked where its name stands before it, and bare where the server's error
    quotes it: found there too, as the store masks a step's other forms of what its masks took."""
    home = tmp_path / "home"
    with Store(home, patterns=()) as store:
        store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "use password=Tr0ub4dor3xyz"}]},
                          response=as_response(text_message("Trying.")))
        failed = store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "try again"}]},
                                   error="login refused for Tr0ub4dor3xyz", status=401)
    reader = read_whole(home)
    (error,) = [unit for unit, _ in reader.hits.values() if unit.key[:2] == ("row", "steps") and unit.key[3] == "error"]
    assert next(iter(error.places)).step == failed.idx
    with Store(home) as store:
        assert sorted(tuple(found["places"]) for found in scan(store).runs) == [("errors",), ("messages",)]


def test_what_cannot_be_read_is_said_and_the_rest_is_read(tmp_path):
    home = tmp_path / "home"
    run = older_folder(home, f"Done. I used the key {KEY}.", "And nothing else.")
    with Store(home) as store:
        steps = store.steps(run)
        lost = steps[1].response_blob
        (home / "blobs" / lost[:2] / lost[2:]).unlink()
        report = scan(store)
        assert store.mark(SCANNED) is not None
    assert report.unread == [f"blob {lost[:12]} (run {run[:8]}, step 1): missing"]
    assert [found["run"] for found in report.runs] == [run]


def test_a_scan_cancelled_leaves_no_mark(tmp_path):
    home = tmp_path / "home"
    older_folder(home, f"Done. I used the key {KEY}.")
    cancel = Cancel()
    cancel.cancel()
    with Store(home) as store:
        with pytest.raises(Cancelled):
            scan(store, cancel=cancel)
        assert store.mark(SCANNED) is None and older(store) is not None


def test_a_chat_kept_after_its_run_is_said_as_an_import(tmp_path):
    home = tmp_path / "home"
    Store(home).close()
    (home / "imports" / "1a2b3c4d").mkdir(parents=True)
    (home / "imports" / "1a2b3c4d" / "chat.json").write_text(json.dumps([{"role": "user", "content": f"key {KEY}"}]))
    with Store(home) as store:
        report = scan(store)
    assert report.runs == [] and report.elsewhere == [{"owner": "import", "id": "1a2b3c4d", "places": 1, "trashed": False,
                                                       "running": False, "aside": None, "by_hand": ["imports/1a2b3c4d/"]}]


def test_a_folder_says_it_holds_runs_an_older_tuningfork_recorded_until_it_is_scanned(tmp_path):
    home = tmp_path / "home"
    with Store(home) as store:
        assert store.mark(OPENED) is not None and older(store) is None
        record_run(store, "new", [text_message("Done.")])
        assert older(store) is None  # recorded by this one: nothing older
    with Store(home) as store:
        assert older(store) is None  # opened again: the mark is the first time's
    as_if_older(home)
    with Store(home) as store:
        found = older(store)
        assert (found["runs"], found["jobs"]) == (1, 0)
        assert older_note(store, command="tuningfork scan --home x") == (
            "1 run here was recorded by an older tuningfork, which masked fewer keys: "
            "`tuningfork scan --home x` looks for what today's masking would mask in them, and changes no recording.")
        scan(store)
        assert older(store) is None and older_note(store) is None


def test_the_note_goes_once_the_older_runs_are_gone(tmp_path):
    from tuningfork.ops import trash

    home = tmp_path / "home"
    run = older_folder(home, "Done.")
    with Store(home) as store:
        assert older(store)["runs"] == 1
        trash.trash(store, "run", run)
        assert older(store)["runs"] == 1  # in the trash, still kept
        trash.purge(store, "run", run)
        assert older(store) is None


def test_the_note_counts_jobs_and_reads_no_recording(tmp_path, monkeypatch):
    home = tmp_path / "home"
    older_folder(home, "Done.", "And again.")
    with Store(home) as store:
        store.create_job(kind="resample", lane="interactive", params={})
    as_if_older(home)

    def refused(*args, **kwargs):
        raise AssertionError("the note reads a recording")

    monkeypatch.setattr("tuningfork.store.db.Store.table_rows", refused)
    monkeypatch.setattr("tuningfork.store.db.Store.message_blobs", refused)
    monkeypatch.setattr("tuningfork.store.blobs.BlobStore.get", refused)
    with Store(home) as store:
        assert older_note(store) == (
            "1 run and 1 job here were recorded by an older tuningfork, which masked fewer keys: "
            "`tuningfork scan` looks for what today's masking would mask in them, and changes no recording.")


def test_serve_and_ui_say_the_folder_holds_older_runs_until_a_scan(cli, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import uvicorn

    class Server:
        def __init__(self, app, *, host, port):
            pass

        def run(self):
            pass

    monkeypatch.setattr("tuningfork.proxy.app.create_app", lambda settings: SimpleNamespace(
        settings=settings, state=SimpleNamespace(access=SimpleNamespace(token="t"))))
    monkeypatch.setattr("tuningfork.proxy.app.ProxyServer", Server)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr("tuningfork.proxy.app.create_browse_app", lambda store, host: store.close())
    home = tmp_path / "home"
    older_folder(home, f"Done. I used the key {KEY}.")
    note = f"recorded by an older tuningfork, which masked fewer keys: `tuningfork scan --home {home}`"
    assert note in cli("serve", "--upstream", "http://127.0.0.1:1", "--home", home).output
    assert note in cli("ui", "--home", home).output
    scanned = cli("scan", "--home", home).output
    assert "stands in 1 run." in scanned and "tuningfork trash run" in scanned and "no longer say" in scanned
    assert "older tuningfork" not in cli("ui", "--home", home).output
    # The folder a person runs with no --home is named so: by nothing.
    other = tmp_path / "other"
    older_folder(other, "Done.")
    monkeypatch.setenv("TUNINGFORK_HOME", str(other))
    told = cli("ui").output
    assert "`tuningfork scan` looks" in told and "--home" not in told


def test_a_run_in_the_trash_is_said_so_and_not_set_aside_again(cli, tmp_path):
    from tuningfork.ops import trash

    home = tmp_path / "home"
    run = older_folder(home, f"The key is {KEY}.")
    with Store(home) as store:
        trash.trash(store, "run", run)
    told = cli("scan", "--home", home).output
    assert "(in the trash)" in told and "set these runs aside" not in told and "trash purge <id>" in told


def test_scan_reads_only_a_recordings_folder_there_is(cli, tmp_path):
    missing = tmp_path / "nothing-here"
    result = cli("scan", "--home", missing)
    assert result.exit_code == 1 and "no recordings folder at" in result.output
    assert not missing.exists()
