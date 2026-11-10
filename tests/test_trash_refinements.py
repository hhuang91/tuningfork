"""A refinement in the trash (`ops.trash`, kind "refinement"): out of the list, taking no new
work, shown as it was and back as it went; deleted for good from there, with what it made —
its screens, its validations' forks and fresh runs, its proposals' saved texts, its folder —
and nothing else: never the run it studies.

Asked for by the user (2026-10-05): refinements can be deleted, as everything else can.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tuningfork.ops import approve, refine, trash, validate
from tuningfork.ops.common import OperationError
from tuningfork.store import Store
from tests.helpers import control_headers
from tests.test_validate import screened, stack  # noqa: F401 — `stack` is the fixture

API = "/_tuningfork/api"


def refused(call, *args, **kwargs) -> OperationError:
    with pytest.raises(OperationError) as caught:
        call(*args, **kwargs)
    return caught.value


class TestInTheTrash:
    def test_it_leaves_the_list_takes_no_new_work_and_comes_back_as_it_went(self, stack):
        store = stack.store
        made, proposal = screened(stack)
        (screen,) = refine.screens(store, made.id)

        moved = trash.trash(store, "refinement", made.id[:8])

        assert moved["moved"] == [made.id] and moved["label"] == "tally-report"
        assert trash.trash(store, "refinement", made.id)["already"]
        assert refine.list_refinements(store, include_closed=True) == []
        shown = refine.view(store, made.id)  # still shown as it was, marked
        assert shown["refinement"]["trashed_at"] and [item["id"] for item in shown["proposals"]] == [proposal.id]
        # No new work: whatever it made would be deleted with it.
        assert refused(refine.add_proposal, store, made.id, edits=[{"old": "report.json", "new": "report.json!"}],
                       rationale="r", proposer={"via": "person"}).code == "trashed"
        assert refused(refine.prepare_refinement_screen, store, made.id, [proposal.id], base_url=stack.url).code == "trashed"
        assert refused(validate.prepare_validate, store, made.id, proposal.id, proxy_url=stack.url).code == "trashed"
        assert refused(approve.prepare_approve, store, made.id, proposal.id).code == "trashed"
        # What it made stays where it is until it is deleted for good.
        assert store.get_experiment(screen["experiment"]) is not None
        assert Path(proposal.data["variant_path"]).is_file()

        (listed,) = trash.listing(store).refinements
        assert listed["id"] == made.id and listed["proposals"] == 1 and listed["status"] == "open"
        goes = listed["goes"]
        assert (goes["refinements"], goes["variants"], goes["experiments"], goes["samples"]) == (1, 1, 2, 10)
        assert goes["runs"] == 0 and goes["cited_by"] == []  # its own screens cite nothing it keeps
        assert trash.find(store, made.id[:6]) == ("refinement", made.id)

        assert trash.restore(store, "refinement", made.id[:8])["restored"]
        assert [each.id for each in refine.list_refinements(store)] == [made.id]
        assert refused(trash.restore, store, "refinement", made.id).code == "not_found"
        again = refine.add_proposal(store, made.id, edits=[{"old": "Write the totals", "new": "Write all the totals"}],
                                    rationale="r", proposer={"via": "person"})
        assert again.id != proposal.id

    def test_work_still_running_keeps_it_out(self, stack):
        store = stack.store
        made, proposal = screened(stack, n=5)
        record = {"id": "f" * 32, "refinement": made.id, "proposal": proposal.id, "status": "running",
                  "pid": os.getpid(), "created_at": 1.0}
        (made.folder / validate.FOLDER).mkdir(exist_ok=True)
        (made.folder / validate.FOLDER / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")
        assert refused(trash.trash, store, "refinement", made.id).code == "busy"
        assert refine.get(store, made.id).trashed_at is None


class TestDeletedForGood:
    def test_it_goes_with_what_it_made_and_nothing_else(self, stack, monkeypatch):
        monkeypatch.setattr(validate, "MINIMUM", 2)
        store = stack.store
        made, proposal = screened(stack)
        plan = validate.prepare_validate(store, made.id, proposal.id, n=2, runs=2, proxy_url=stack.url)
        record = validate.execute_validate(store, plan).record
        screen_arms = [experiment.id for experiment in store.experiments(kind="screen", limit=50)]
        forward = list(record["forward"]["experiments"].values())
        forks = [verdict["run_id"] for arm in ("original", "edit") for verdict in record["forward"][arm]]
        training = [run for arm in ("original", "edit") for run in record["training"]["runs"][arm]]
        launch_folders = [Path(store.get_launch(store.get_run(run).launch_id).workspace).parent for run in training]
        assert all(folder.is_dir() for folder in launch_folders)

        found = refine.made_by(store, refine.get(store, made.id))
        assert set(found.experiments) == set(screen_arms + forward) and set(found.runs) == set(training)
        assert refused(trash.purge, store, "refinement", made.id).code == "bad_params"  # from the trash only

        trash.trash(store, "refinement", made.id)
        goes = trash.preview(store, "refinement", made.id)
        assert goes["experiments"] == len(screen_arms) + len(forward)
        assert goes["runs"] >= len(set(forks + training)) and goes["cited_by"] == [] and goes["notes"] == []

        removed = trash.purge(store, "refinement", made.id[:8]).to_json()

        assert removed["removed"]["refinements"] == 1 and removed["removed"]["variants"] == 1
        assert removed["items"] == [{"kind": "refinement", "id": made.id, "label": "tally-report"}]
        assert not made.folder.exists() and not Path(proposal.data["variant_path"]).exists()
        assert refused(refine.get, store, made.id).code == "not_found"
        assert all(store.get_experiment(each) is None for each in screen_arms + forward)
        assert all(store.get_run(each) is None for each in forks + training)
        assert not any(folder.exists() for folder in launch_folders)
        # Never the run it studied, nor the skill's own folder.
        assert store.get_run(made.run_id) is not None
        assert (stack.tmp / "skills-src" / "tally-report" / "SKILL.md").is_file()
        assert trash.listing(store).empty

    def test_from_a_copy_it_takes_the_copys_own_saved_texts_never_the_originals(self, stack):
        """Issue #76: a proposal's record names its text where it was saved, in the folder it was
        made in. Deleted for good from a copy of that folder, a refinement takes the copy's own."""
        made, proposal = screened(stack)
        screens = [experiment.id for experiment in stack.store.experiments(kind="screen", limit=50)]
        copy = stack.tmp / "copy"
        shutil.copytree(stack.store.home, copy, symlinks=True)
        own, original = copy / "variants" / "tally-report" / f"{proposal.data['variant']}.md", Path(proposal.data["variant_path"])
        store = Store(copy)
        try:
            assert refine.made_by(store, refine.get(store, made.id)).variants == [own]
            trash.trash(store, "refinement", made.id)
            removed = trash.purge(store, "refinement", made.id).to_json()
            assert removed["removed"]["refinements"] == 1 and removed["removed"]["variants"] == 1 and not own.exists()
        finally:
            store.close()
        assert original.is_file() and refine.get(stack.store, made.id).folder.is_dir()
        assert all(stack.store.get_experiment(each) is not None for each in screens)

    def test_emptying_the_trash_takes_refinements_too(self, stack):
        store = stack.store
        made, proposal = screened(stack)
        trash.trash(store, "refinement", made.id)
        removed = trash.empty(store).to_json()
        assert {"kind": "refinement", "id": made.id, "label": "tally-report"} in removed["items"]
        assert removed["removed"]["refinements"] == 1 and removed["removed"]["experiments"] == 2
        assert not made.folder.exists() and store.experiments(kind="screen", include_trashed=True) == []


class TestFrontEnds:
    def test_from_the_command_line(self, stack):
        from tuningfork.cli import app as cli

        store = stack.store
        made, _ = screened(stack)
        home = ["--home", str(store.home)]
        runner = CliRunner()
        moved = runner.invoke(cli, ["trash", "refinement", made.id[:8], *home])
        assert moved.exit_code == 0 and f"moved refinement {made.id[:8]} (tally-report) to the trash" in moved.output, moved.output
        listed = runner.invoke(cli, ["trash", "list", *home])
        assert "refinement" in listed.output and "1 refinement" in listed.output, listed.output
        assert "no refinements yet" in runner.invoke(cli, ["refine", "list", *home]).output
        shown = runner.invoke(cli, ["refine", "show", made.id[:8], *home])
        assert shown.exit_code == 0 and "in the trash since" in shown.output, shown.output
        purged = runner.invoke(cli, ["trash", "purge", made.id[:8], "--yes", *home])
        assert purged.exit_code == 0 and "deleted for good: 1 refinement" in purged.output, purged.output
        assert not made.folder.exists()

    def test_over_http(self, stack):
        store = stack.store
        made, _ = screened(stack)
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            assert http.post(f"{API}/trash", json={"kind": "refinement", "id": made.id}).status_code == 401  # no token
            moved = http.post(f"{API}/trash", json={"kind": "refinement", "id": made.id},
                              headers=control_headers(stack.app))
            assert moved.status_code == 200 and moved.json()["moved"] == [made.id]
            assert http.get(f"{API}/refinements").json() == []
            assert http.get(f"{API}/refinements/{made.id}").json()["refinement"]["trashed_at"]
            assert [each["id"] for each in http.get(f"{API}/trash").json()["refinements"]] == [made.id]
            preview = http.get(f"{API}/trash/preview", params={"kind": "refinement", "id": made.id}).json()
            assert preview["refinements"] == 1
            restored = http.post(f"{API}/trash/restore", json={"kind": "refinement", "id": made.id},
                                 headers=control_headers(stack.app))
            assert restored.status_code == 200 and refine.get(store, made.id).trashed_at is None
