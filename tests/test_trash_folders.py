"""Deleting for good, on disk: the folder a batch or a set of forks was laid out in goes with the
last of its runs, the skill copy its runs were given with it, wherever tuningfork made it and
marked it; a folder it did not mark, and a chain's, stay. What a deleted row held is
overwritten, the database's log emptied, and what the two-minute grace kept goes once it has
passed — whichever process deleted it.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from tuningfork.ops import trash
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork import runner
from tests.test_approve import stack  # noqa: F401 — the fixture: a task, a proxy, a skill folder
from tests.test_private import mentions

#: Written into the task's prompt and the skill: the launch's row, the skill's copy, and every
#: request that carried either.
MARKER = "plum orchard nineteen lanterns"


def marked(stack) -> None:
    data = json.loads(stack.spec.path.read_text(encoding="utf-8"))
    data["evals"][0]["prompt"] += f" {MARKER}."
    stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
    stack.skill.write_text(stack.skill.read_text(encoding="utf-8") + f"\nKept as written: {MARKER}.\n",
                           encoding="utf-8")


def batch(stack, workdir: Path, n: int = 2) -> list[str]:
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=workdir,
                         model="m", harness="fake", n=n)
    runs = [outcome.run_id for outcome in execute_batch(stack.store, plan).outcomes]
    assert all(runs)
    return runs


def group_of(store, run_id: str) -> Path:
    return Path(store.get_launch(store.get_run(run_id).launch_id).workspace).parent.parent


def deleted(store, run_id: str) -> dict:
    trash.trash(store, "run", run_id)
    return trash.purge(store, "run", run_id).to_json()


def aged(home: Path) -> None:
    """Every blob and snapshot object as though written long ago: past the grace."""
    for root in (home / "blobs", home / "snapshots.git" / "objects"):
        for path in root.rglob("*"):
            if path.is_file():
                os.utime(path, (0, 0))


class TestTheFolderABatchRanIn:
    def test_it_goes_with_its_last_run_and_the_skill_copy_with_it(self, stack):
        store = stack.store
        first, second = batch(stack, store.home / "batches")
        group = group_of(store, first)
        assert {entry.name for entry in group.iterdir()} == {"0", "1", "skill-snapshot", runner.GROUP_MARK}

        deleted(store, first)
        # The other run was given the same copy, and its forks would read it: both stay.
        assert {entry.name for entry in group.iterdir()} == {"1", "skill-snapshot", runner.GROUP_MARK}

        removed = deleted(store, second)
        assert not group.exists() and removed["folders"] == 2  # its launch's folder, then the batch's
        assert (store.home / "batches").is_dir()

    def test_outside_the_recordings_folder_it_goes_where_tuningfork_marked_it(self, stack):
        store = stack.store
        runs = batch(stack, stack.tmp / "elsewhere")
        group = group_of(store, runs[0])
        assert stat.S_IMODE(group.stat().st_mode) == 0o700 and (group / runner.GROUP_MARK).is_file()

        for run in runs:
            deleted(store, run)

        assert not group.exists() and (stack.tmp / "elsewhere").is_dir()

    def test_outside_it_without_the_mark_nothing_is_removed(self, stack):
        # A folder laid out before tuningfork marked its own, or one a person keeps there.
        store = stack.store
        runs = batch(stack, stack.tmp / "elsewhere")
        group = group_of(store, runs[0])
        (group / runner.GROUP_MARK).unlink()

        removed = [deleted(store, run) for run in runs]

        assert {entry.name for entry in group.iterdir()} == {"0", "1", "skill-snapshot"}
        assert [each["folders"] for each in removed] == [0, 0]

    def test_outside_it_without_the_mark_it_stays_once_its_launches_folders_are_gone(self, stack):
        # Cleared out by hand, it holds only the skill copy: still not tuningfork's to remove.
        import shutil

        store = stack.store
        runs = batch(stack, stack.tmp / "elsewhere")
        group = group_of(store, runs[0])
        (group / runner.GROUP_MARK).unlink()
        for launch in ("0", "1"):
            shutil.rmtree(group / launch)

        for run in runs:
            deleted(store, run)

        assert {entry.name for entry in group.iterdir()} == {"skill-snapshot"}

    def test_one_holding_anything_else_stays(self, stack):
        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        group = group_of(store, run)
        (group / "notes.txt").write_text("a person's own file\n", encoding="utf-8")

        deleted(store, run)

        assert {entry.name for entry in group.iterdir()} == {"notes.txt", "skill-snapshot", runner.GROUP_MARK}

    def test_a_set_of_forks_goes_with_its_last_fork_and_never_its_sources(self, stack):
        store = stack.store
        (source,) = batch(stack, store.home / "batches", n=1)
        _, (fork,) = stack.fork(source, "write_file")
        forks = group_of(store, fork.run_id)
        assert forks.parent == stack.tmp / "forks" and (forks / runner.GROUP_MARK).is_file()

        deleted(store, fork.run_id)

        assert not forks.exists()
        assert (group_of(store, source) / "skill-snapshot").is_dir()  # the source was given it, and stays

    def test_a_skill_copy_an_older_fork_was_given_stays_until_that_fork_goes(self, stack):
        # A fork recorded before forks named their source is not deleted with it: the copy it
        # was given stays while it is left, and goes with it.
        store = stack.store
        (source,) = batch(stack, store.home / "batches", n=1)
        _, (fork,) = stack.fork(source, "write_file")
        group = group_of(store, source)
        with store._lock:
            store._conn.execute("UPDATE launches SET source_run = NULL WHERE id = ?",
                                (store.get_run(fork.run_id).launch_id,))
            store._conn.commit()

        deleted(store, source)
        assert {entry.name for entry in group.iterdir()} == {"skill-snapshot", runner.GROUP_MARK}

        deleted(store, fork.run_id)
        assert not group.exists()

    def test_a_chains_folder_stays_with_every_skill_in_it(self, stack):
        # A chain's record and the skill each link left are the chain's (decided 2026-10-27):
        # deleting its links' runs takes their own folders and nothing of the chain's.
        from tuningfork.ops.chain import execute_chain, prepare_chain

        store = stack.store
        report = execute_chain(store, prepare_chain(
            store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=store.home / "batches",
            model="m", harness="fake", n=2,
        ))
        folder = group_of(store, report.links[0].run_id)
        for link in report.links:
            deleted(store, link.run_id)

        names = [entry.name for entry in folder.iterdir()]
        assert sum(name.startswith("skill-snapshot") for name in names) == 2
        assert sum(name.startswith("skill-left") for name in names) == 2
        assert all((Path(link.left) / "SKILL.md").is_file() for link in report.links)
        assert (store.home / "chains" / f"{report.plan.id}.json").is_file()

    def test_a_chains_folder_stays_before_any_link_left_a_skill(self, stack):
        # Cut off before its first link ended, a chain's folder holds only what it gave: still its.
        import shutil

        from tuningfork.ops.chain import execute_chain, prepare_chain

        store = stack.store
        report = execute_chain(store, prepare_chain(
            store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=store.home / "batches",
            model="m", harness="fake", n=1,
        ))
        (link,) = report.links
        folder = group_of(store, link.run_id)
        shutil.rmtree(Path(link.left).parent)  # skill-left-<i>, whole

        deleted(store, link.run_id)

        assert folder.is_dir() and any(entry.name.startswith("skill-snapshot") for entry in folder.iterdir())


class TestNothingLeft:
    def test_a_deleted_batch_leaves_nothing_of_it_in_the_recordings_folder(self, stack):
        store = stack.store
        marked(stack)
        runs = batch(stack, store.home / "batches")
        found = mentions(store.home, MARKER)
        assert any(name.endswith("SKILL.md") for name in found)
        assert any("tuningfork.db" in name for name in found)  # the launch's row: its task as given
        aged(store.home)

        for run in runs:
            deleted(store, run)

        # Not in a file, a blob, the database or its log: rows went overwritten, the log emptied.
        assert mentions(store.home, MARKER) == []
        assert store._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1

    def test_what_the_grace_kept_goes_once_it_has_passed(self, stack):
        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)

        removed = deleted(store, run)  # at once: what the run wrote is younger than the grace

        assert removed["held"] > 0
        held = [path for path in (store.home / "blobs").rglob("*") if path.is_file()]
        due = store.home / trash.SWEEP_DUE
        assert due.is_file()
        assert trash.sweep_if_due(store) is None  # not yet
        aged(store.home)

        swept = trash.sweep_if_due(store, now=time.time() + 2 * trash.GRACE).to_json()

        assert swept["blobs"] > 0 and not due.exists()
        assert any(not path.exists() for path in held)
        assert trash.sweep_if_due(store, now=time.time() + 2 * trash.GRACE) is None

    def test_a_running_proxy_sweeps_it_whichever_process_deleted(self, tmp_path, monkeypatch):
        import asyncio

        import httpx

        from tuningfork.config import Settings
        from tuningfork.proxy.app import create_app

        monkeypatch.setattr(trash, "SWEEP_EVERY", 0.05)
        home = tmp_path / "home"
        nowhere = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
        app = create_app(Settings(upstream_base_url="http://upstream.test", home=home), client=nowhere)
        blobs = app.state.recorder.store.blobs
        orphan = blobs.put(b"left by a deletion")
        os.utime(blobs._path(orphan), (0, 0))
        trash._write_due(home, time.time() - 1)  # as another process's deletion left it

        async def serving() -> None:
            async with app.router.lifespan_context(app):  # what `serve` runs around its routes
                deadline = time.monotonic() + 10
                while (home / trash.SWEEP_DUE).exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            await nowhere.aclose()

        asyncio.run(serving())

        assert not (home / trash.SWEEP_DUE).exists() and orphan not in blobs


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_a_folder_made_for_launches_is_this_users_alone(stack):
    (run,) = batch(stack, stack.tmp / "batches", n=1)
    assert stat.S_IMODE(group_of(stack.store, run).stat().st_mode) == 0o700
