"""`tuningfork demo`: the made-up runs and the scripted model, started by one command, from
tuningfork's own package — the README's way in for a reader with no GPU."""

from __future__ import annotations

import errno
import json
import re
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.harness import ADAPTERS
from tuningfork.store.db import Store

SRC = Path(__file__).resolve().parents[1] / "src" / "tuningfork"


@pytest.fixture
def served(monkeypatch):
    """What the demo would serve, caught instead of served: the proxy's arguments."""
    from tuningfork.demo import lab

    calls = []
    monkeypatch.setattr(lab, "proxy", lambda home, port, **options: calls.append((home, port, options)))
    # Whatever this machine serves on the demo's port, the demo is never served here.
    monkeypatch.setattr(lab, "check_port", lambda port: None)
    # The lab names its harness for the jobs it runs; this test takes the name back after.
    monkeypatch.setitem(ADAPTERS, "fake", None)
    return calls


def demo(*args: str):
    return CliRunner().invoke(app, ["demo", *args])


def refused(*args: str) -> str:
    result = demo(*args)
    assert result.exit_code == 1, result.output
    return result.output


def labels(home: Path) -> list[str]:
    store = Store(home)
    try:
        return sorted(run.label or "" for run in store.runs(limit=100))
    finally:
        store.close()


def test_seed_records_the_made_up_runs(tmp_path):
    result = demo("seed", "--home", str(tmp_path / "demo"))
    assert result.exit_code == 0, result.output
    assert "seeded" in result.output and "Try: tuningfork runs" in result.output
    assert {f"tally-report/1#demo.{index}" for index in range(9)} <= set(labels(tmp_path / "demo"))


def test_the_lab_is_the_default_and_its_model_follows_the_wording(tmp_path, served):
    result = demo("--home", str(tmp_path / "lab"), "--port", "8799")
    assert result.exit_code == 0, result.output
    # The web UI it prints is what to try: no command a reader of `uvx` would not have.
    assert "seeded" in result.output and "Try:" not in result.output
    ((home, port, options),) = served
    assert (home, port) == (tmp_path / "lab", 8799)
    assert (home / "lab-task" / "evals.json").is_file() and ADAPTERS["fake"].name == "fake"
    assert "tokens-lab" in labels(home)

    def keys(system: str) -> dict:
        turn = options["responder"]({"messages": [{"role": "system", "content": system}]})
        return json.loads(turn.tool_calls[0]["arguments"]["content"])

    # The guide's walkthrough: the report's keys only for a skill that names them in these words.
    assert set(keys("Write the totals to report.json.")) == {"count", "total"}
    assert set(keys("Write the totals, using the keys row_count and amount_total.")) == {"row_count", "amount_total"}


def test_a_demo_records_in_a_new_folder_never_your_own(tmp_path, monkeypatch, served):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    monkeypatch.setenv("TUNINGFORK_HOME", str(tmp_path / "mine"))
    for _ in range(2):
        assert demo().exit_code == 0
    homes = [home for home, _, _ in served]
    assert len(set(homes)) == 2 and all(re.fullmatch(r"tuningfork-lab-\w+", home.name) for home in homes)
    assert all(home.parent == tmp_path / "temp" for home in homes)
    assert not (tmp_path / "mine").exists()
    # A new folder it cannot make, or cannot write in, is named, with what to do instead.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "gone"))
    said = refused()
    assert f"cannot record in {tmp_path / 'gone'}" in said and "name a folder with --home" in said
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))

    def full(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full)
    said = refused()
    assert f"cannot record in {tmp_path / 'temp'}" in said and "No space left" in said
    assert "name a folder with --home" in said and "leave --home out" not in said


def test_a_demo_never_records_among_your_recordings(tmp_path, monkeypatch, served):
    """Its runs are made up under a real task's id: among real runs they would count in its
    pass rates, and the lab would point every run of that task at its own copy of the skill."""
    monkeypatch.setenv("HOME", str(tmp_path / "you"))
    mine = tmp_path / "you" / ".tuningfork"
    assert "where tuningfork keeps your recordings by default" in refused("--home", str(mine))
    assert not mine.exists()
    Store(tmp_path / "other").close()
    before = sorted((tmp_path / "other").rglob("*"))
    assert "not marked as the demo's" in refused("seed", "--home", str(tmp_path / "other"))
    assert sorted((tmp_path / "other").rglob("*")) == before
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "todo.txt").write_text("mine")
    assert "not empty" in refused("--home", str(tmp_path / "notes"))
    assert sorted(path.name for path in (tmp_path / "notes").iterdir()) == ["todo.txt"]
    assert "a file, not a folder" in refused("--home", str(tmp_path / "notes" / "todo.txt"))
    assert "cannot record in" in refused("--home", str(tmp_path / "notes" / "todo.txt" / "inside"))
    assert not served


def test_a_demo_goes_on_in_its_own_folder_and_seeds_it_once(tmp_path, monkeypatch, served):
    """As the guide uses them: each folder named with --home, and in its shell, once seeded,
    `$TUNINGFORK_HOME` set to it."""
    seeded, live = tmp_path / "tuningfork-demo", tmp_path / "tuningfork-live"
    assert demo("seed", "--home", str(seeded)).exit_code == 0
    monkeypatch.setenv("TUNINGFORK_HOME", str(seeded))
    assert "seeded already" in refused("seed", "--home", str(seeded))
    assert demo("--home", str(seeded)).exit_code == 0
    assert labels(seeded).count("tally-report/1#demo.0") == 1

    # A proxy's folder holds no seeded runs: the lab seeds it, and only then is it seeded. The
    # guide's second terminal names it in $TUNINGFORK_HOME, perhaps before the first makes it.
    monkeypatch.setenv("TUNINGFORK_HOME", str(live))
    assert demo("proxy", "--home", str(live)).exit_code == 0
    Store(live).close()
    assert demo("live", "--home", str(live)).exit_code == 0
    assert demo("--home", str(live)).exit_code == 0
    assert "tally-report/1#demo.0" in labels(live)
    assert "seeded already" in refused("seed", "--home", str(live))
    assert [home.name for home, _, _ in served] == ["tuningfork-demo", "tuningfork-live"] + ["tuningfork-live"] * 2


def test_a_port_in_use_is_refused_before_anything_is_made(tmp_path):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert f"port {port} is in use" in refused("--home", str(tmp_path / "lab"), "--port", str(port))
    assert not (tmp_path / "lab").exists()
    assert demo("--port", "99999").exit_code == 2  # out of range, refused by its option


def test_live_asks_for_its_held_call_and_ctrl_c_ends_it_quietly(tmp_path, monkeypatch):
    from tuningfork.demo import lab

    seen = {}

    def interrupted(home, port, **options):
        seen.update(options)
        raise KeyboardInterrupt

    monkeypatch.setattr(lab, "proxy", interrupted)
    monkeypatch.setattr(lab, "check_port", lambda port: None)
    result = demo("live", "--home", str(tmp_path / "live"))
    assert (result.exit_code, result.output, seen) == (130, "", {"demo_call": True})


def test_an_unknown_demo_is_refused(tmp_path):
    assert "lab, seed, proxy or live" in refused("browse", "--home", str(tmp_path / "x"))
    assert not (tmp_path / "x").exists()


def test_tuningfork_imports_nothing_from_the_test_suite():
    """The demo ships with tuningfork, so nothing it imports may live in `tests/`, which an
    install does not have."""
    found = [
        f"{path.relative_to(SRC)}:{number}"
        for path in SRC.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if re.match(r"\s*(from|import) tests\b", line)
    ]
    assert not found, found
    from tuningfork.demo.task import FAKE_AGENT

    assert FAKE_AGENT.is_file() and FAKE_AGENT.parent == SRC / "demo"
