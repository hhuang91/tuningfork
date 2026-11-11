"""Wordings edited in the browser: where they are kept, and that a name never points at two
texts — an experiment's record names the file."""

from __future__ import annotations

import os
import time

import httpx
import pytest

from tuningfork.ops import OperationError
from tuningfork.ops.variants import list_variants, path_for, read_variant, save_variant
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
