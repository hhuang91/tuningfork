"""Wordings edited in the browser: where they are kept, and that a name never points at two
texts — an experiment's record names the file."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import httpx
import pytest

from tuningfork.ops import OperationError
from tuningfork.ops.variants import keep, list_variants, path_for, read_variant, save_variant
from tuningfork.proxy.app import create_browse_app
from tests.helpers import record_pilot


@pytest.mark.parametrize("name", ["../escape", "A-capital", "", "with space", "ends.md", ".hidden", "x" * 70])
def test_a_name_is_one_plain_path_component(tmp_path, name):
    with pytest.raises(OperationError):
        path_for(tmp_path, "tally-report", name)


def test_variants_live_in_the_recordings_folder_never_in_a_skill_folder(tmp_path):
    path = save_variant(tmp_path, "tally-report", "keys-named", "text")
    assert path == tmp_path / "variants" / "tally-report" / "keys-named.md"
    assert read_variant(tmp_path, "tally-report", "keys-named") == "text"
    assert [p.name for p in path.parent.iterdir()] == ["keys-named.md"]  # no temporary left


def test_a_name_holding_other_text_is_refused_not_overwritten(tmp_path):
    save_variant(tmp_path, "tally-report", "v2", "first")
    assert save_variant(tmp_path, "tally-report", "v2", "first").read_text() == "first"
    with pytest.raises(OperationError, match="already exists") as refused:
        save_variant(tmp_path, "tally-report", "v2", "second")
    assert refused.value.field == "variant_name"
    assert read_variant(tmp_path, "tally-report", "v2") == "first"


def test_the_text_a_name_holds_is_its_bytes_line_endings_and_all(tmp_path):
    # Read back as Python reads text, CRLF as LF, a text whose lines end with CRLF was refused as
    # another when saved again (found in review of issue #76's fix), and the same lines ended with
    # LF were said to be what the file holds: a run given that file is given other bytes.
    path = save_variant(tmp_path, "tally-report", "v2", "first\r\nsecond\r\n")
    assert save_variant(tmp_path, "tally-report", "v2", "first\r\nsecond\r\n") == path
    assert read_variant(tmp_path, "tally-report", "v2") == "first\r\nsecond\r\n"
    with pytest.raises(OperationError, match="already exists"):
        save_variant(tmp_path, "tally-report", "v2", "first\nsecond\n")
    assert path.read_bytes() == b"first\r\nsecond\r\n"
    path.write_bytes(b"\xff not UTF-8")
    with pytest.raises(OperationError, match="already exists"):
        save_variant(tmp_path, "tally-report", "v2", "first\r\nsecond\r\n")


@pytest.mark.parametrize("links", [True, False], ids=["linked-into-place", "no-links"])
def test_a_save_never_overwrites_a_file_another_save_made_meanwhile(tmp_path, monkeypatch, links):
    # Two A/Bs under one name started at once each look for the name's file before either saves:
    # the later must find the earlier's file as it writes, never replace it. The race is staged
    # where `save_variant` looks, `Path.exists`: a save that looked otherwise would race unseen here.
    if not links:
        def unlinkable(source, target):
            raise PermissionError(1, "Operation not permitted")  # a file system without hard links

        monkeypatch.setattr(os, "link", unlinkable)
    path = path_for(tmp_path, "tally-report", "v2")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first\r\n")
    seen = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self, **options: self != path and seen(self, **options))
    assert save_variant(tmp_path, "tally-report", "v2", "first\r\n") == path
    with pytest.raises(OperationError, match="already exists"):
        save_variant(tmp_path, "tally-report", "v2", "second\r\n")
    monkeypatch.undo()
    assert path.read_bytes() == b"first\r\n"
    assert [p.name for p in path.parent.iterdir()] == ["v2.md"]  # no temporary left
    if not links:
        monkeypatch.setattr(os, "link", unlinkable)
    assert save_variant(tmp_path, "tally-report", "v3", "third\r\n").read_bytes() == b"third\r\n"


def test_two_threads_saving_one_name_never_write_into_each_others_file(tmp_path, monkeypatch):
    # The jobs of one proxy are threads of one process. A temporary named by the process alone is
    # the other thread's too: written there once this thread had linked it into place, it changed
    # the file this thread had saved.
    path = path_for(tmp_path, "tally-report", "v2")
    linked, seen = os.link, Path.exists
    outcome: dict[str, object] = {}

    def save(text: str) -> None:
        try:
            outcome[text] = save_variant(tmp_path, "tally-report", "v2", text)
        except OperationError as refused:
            outcome[text] = refused

    def exists(self, **options):  # the other thread looked before this one's file was there
        if self == path and threading.current_thread() is not threading.main_thread():
            return False
        return seen(self, **options)

    def link(source, target):
        linked(source, target)
        if threading.current_thread() is threading.main_thread():
            other = threading.Thread(target=save, args=("second",))
            other.start()
            other.join()

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(os, "link", link)
    save("first")
    monkeypatch.undo()
    assert outcome["first"] == path and path.read_bytes() == b"first"
    assert isinstance(outcome["second"], OperationError) and "already exists" in str(outcome["second"])
    assert [p.name for p in path.parent.iterdir()] == ["v2.md"]


def test_the_same_text_saved_meanwhile_its_lines_ended_otherwise_is_the_one_kept(tmp_path, monkeypatch):
    # The start of an A/B takes a file holding its text with other line endings as its own: so it
    # must when another save made that file between its look and its own save (found in review).
    path = path_for(tmp_path, "tally-report", "v2")
    linked = os.link

    def link(source, target):
        Path(target).write_bytes(b"first\r\nsecond\r\n")  # another save of the text lands first
        linked(source, target)

    monkeypatch.setattr(os, "link", link)
    assert keep(tmp_path, path, "first\nsecond\n") == path
    monkeypatch.undo()
    assert path.read_bytes() == b"first\r\nsecond\r\n"
    with pytest.raises(OperationError, match="already exists"):
        keep(tmp_path, path, "another\n")


def test_a_queued_ab_holds_its_name_for_its_own_skill_alone(store):
    # Another skill's variant of the same name is another file (found in review: nothing pinned which
    # file a queued A/B holds); the same text, its lines ended otherwise, is that A/B's own.
    from tuningfork.ops.catalog import CATALOG, JobContext, _refuse_queued

    run_id = record_pilot(store)
    queued = store.create_job(kind="ab", lane="background", params={
        "run": run_id, "step": "1", "variant_text": "first\n", "variant_name": "v9"})
    context = JobContext(store=store, proxy=None, job_id="")

    def asked(text: str):
        return CATALOG["ab"].parse({"run": run_id, "step": "1", "variant_text": text, "variant_name": "v9"})

    with pytest.raises(OperationError, match=f"is taken by job {queued.id[:8]}, queued with another text"):
        _refuse_queued(context, asked("second\n"), path_for(store.home, "tally-report", "v9"))
    _refuse_queued(context, asked("second\n"), path_for(store.home, "other-skill", "v9"))
    _refuse_queued(context, asked("first\r\n"), path_for(store.home, "tally-report", "v9"))


def test_saved_variants_are_listed_newest_first(tmp_path):
    older = save_variant(tmp_path, "tally-report", "older", "a")
    os.utime(older, (time.time() - 60, time.time() - 60))
    save_variant(tmp_path, "tally-report", "newer", "b")
    save_variant(tmp_path, "other-skill", "x", "c")
    assert [v.name for v in list_variants(tmp_path, "tally-report")] == ["newer", "older"]
    assert len(list_variants(tmp_path)) == 3


async def test_the_skill_a_run_was_given_is_served_as_the_editors_baseline(store, tmp_path):
    run_id = record_pilot(store)
    snapshot = tmp_path / "snapshot" / "tally-report"
    snapshot.mkdir(parents=True)
    (snapshot / "SKILL.md").write_text("the recorded wording\n", encoding="utf-8")
    store.annotate_run(run_id, parity={"skill_snapshot_dir": str(snapshot)})
    save_variant(store.home, "tally-report", "mine", "an edit")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_browse_app(store)), base_url="http://localhost:8788"
    ) as client:
        skill = (await client.get(f"/_tuningfork/api/runs/{run_id}/skill")).json()
        saved = (await client.get("/_tuningfork/api/variants?skill=tally-report")).json()
        missing = await client.get("/_tuningfork/api/runs/nothing-like-it/skill")

    assert skill["skill"] == "tally-report" and skill["text"] == "the recorded wording\n"
    assert skill["path"] == str(snapshot / "SKILL.md") and len(skill["sha256"]) == 64
    assert [v["name"] for v in saved] == ["mine"]
    assert missing.status_code == 404
