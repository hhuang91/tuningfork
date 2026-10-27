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


def in_snapshots(home: Path, text: str) -> list[str]:
    """The workspace snapshots' objects holding `text`: git keeps each one compressed, out of a
    search of the bytes (found in review)."""
    import zlib

    found = []
    for path in (home / "snapshots.git" / "objects").glob("??/*"):
        try:
            if text.encode() in zlib.decompress(path.read_bytes()):
                found.append(str(path))
        except zlib.error:
            continue
    return found


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
        assert in_snapshots(store.home, MARKER)  # the skill each launch was given, committed
        aged(store.home)

        for run in runs:
            deleted(store, run)

        # Not in a file, a blob, a snapshot, the database or its log: rows went overwritten, the
        # log emptied.
        assert mentions(store.home, MARKER) == [] and in_snapshots(store.home, MARKER) == []
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


class TestWhatIsNotItsOwn:
    def test_deleting_in_a_copy_of_the_recordings_folder_leaves_the_originals_folders(self, stack):
        # Found in review: a copy's launches name the original's folders, whose mark said only
        # "tuningfork's"; deleting a run in the copy removed them from under the original.
        import shutil

        from tuningfork.store.db import Store

        store = stack.store
        runs = batch(stack, store.home / "batches", n=1) + batch(stack, stack.tmp / "elsewhere", n=1)
        groups = [group_of(store, run) for run in runs]
        copied = Store(Path(shutil.copytree(store.home, stack.tmp / "copy")))
        try:
            assert trash.preview(copied, "run", runs[1])["notes"] == []
            removed = [deleted(copied, run)["folders"] for run in runs]
        finally:
            copied.close()

        assert removed == [0, 0]
        assert all((group / "0").is_dir() and (group / "skill-snapshot").is_dir() for group in groups)
        assert all(store.get_run(run) is not None for run in runs)

    def test_a_skill_copy_added_as_a_skill_stays(self, stack):
        # Found in review: a person can add a batch's skill copy as a skill remembered where it
        # is, and an approval then writes there.
        from tuningfork.ops import addskill

        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        copy = group_of(store, run) / "skill-snapshot" / "tally-report"
        addskill.execute_add(store, addskill.prepare_folder(store, copy))

        deleted(store, run)

        assert (copy / "SKILL.md").is_file()

    def test_a_skill_copy_a_scenario_names_stays(self, stack):
        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        copy = group_of(store, run) / "skill-snapshot" / "tally-report"
        scenario = store.home / "scenarios" / "named"
        scenario.mkdir(parents=True)
        (scenario / "evals.json").write_text(
            json.dumps({"skill_name": "tally-report", "tuningfork": {"skill_dir": str(copy)}, "evals": []}),
            encoding="utf-8")

        deleted(store, run)

        assert (copy / "SKILL.md").is_file()

    def test_a_new_group_is_never_a_folder_that_was_there(self, tmp_path):
        runner.make_group(tmp_path, "one", tmp_path / "home")
        with pytest.raises(FileExistsError):
            runner.make_group(tmp_path, "one", tmp_path / "home")
        (tmp_path / "theirs").mkdir()
        runner.make_group(tmp_path, "theirs", tmp_path / "home", fresh=False)  # a chain's later link
        assert runner.made_for(tmp_path / "theirs", tmp_path / "home")
        assert not runner.made_for(tmp_path / "theirs", tmp_path / "elsewhere")


class TestSaidBeforeAndAfter:
    def test_its_preview_says_where_it_reaches_outside_the_recordings_folder(self, stack):
        store = stack.store
        (run,) = batch(stack, stack.tmp / "elsewhere", n=1)
        trash.trash(store, "run", run)

        (note,) = trash.preview(store, "run", run)["notes"]

        assert "outside the recordings folder" in note and str(group_of(store, run)) in note

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="file modes, as a user they bind")
    def test_a_folder_it_cannot_read_stays_and_the_sweep_still_runs(self, stack):
        # Found in review: a folder it could not read stopped the deletion after the rows went,
        # so nothing was swept and the log stayed.
        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        group = group_of(store, run)
        group.chmod(0)
        try:
            deleted(store, run)
            assert (store.home / trash.SWEEP_DUE).is_file()
        finally:
            group.chmod(0o700)
        assert (group / "0").is_dir()


def test_a_folder_named_itself_stays_whole_and_one_it_cannot_list_is_left(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    (folder / "workspace").mkdir(parents=True)
    trash._clear(folder, [folder])
    assert (folder / "workspace").is_dir()

    def refused(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", refused)
    trash._clear(folder, [folder / "workspace" / "kept"])  # raises nothing
    assert (folder / "workspace").is_dir()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX")
def test_a_task_file_that_is_a_pipe_holds_no_deletion(stack):
    # A pipe at a task file's path, opened, waits for a writer that never comes.
    import threading

    store = stack.store
    (run,) = batch(stack, store.home / "batches", n=1)
    pipe = stack.tmp / "piped.json"
    os.mkfifo(pipe)
    with store._lock:
        store._conn.execute("UPDATE launches SET spec_path = ?", (str(pipe),))
        store._conn.commit()
    trash.trash(store, "run", run)
    previewed = threading.Thread(target=trash.preview, args=(store, "run", run), daemon=True)
    previewed.start()
    previewed.join(10)
    assert not previewed.is_alive()


def test_a_launch_folder_it_cannot_read_is_left(tmp_path, monkeypatch):
    # Python 3.12 and 3.13 raise from `is_dir` where a folder cannot be searched; 3.14 says False.
    home = tmp_path / "home"
    folder = home / "batches" / "g" / "0"
    (folder / "workspace").mkdir(parents=True)

    def refused(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "is_dir", refused)
    assert trash._launch_folder(home.resolve(), str(folder / "workspace"), str(folder / "home")) is None


class TestTheLogAndTheMark:
    def test_a_checkpoint_another_reader_holds_is_tried_again(self, stack):
        import sqlite3

        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        reader = sqlite3.connect(store.home / "tuningfork.db")
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM runs").fetchone()
            started = time.monotonic()
            assert store.checkpoint() is False and time.monotonic() - started < 2
            (store.home / trash.SWEEP_DUE).unlink(missing_ok=True)
            aged(store.home)
            trash._write_due(store.home, time.time() - 1)
            trash.sweep_if_due(store, now=time.time() + 1)
            assert (store.home / trash.SWEEP_DUE).is_file()  # due again, for the log
        finally:
            reader.close()
        assert store.checkpoint() is True
        trash.sweep_if_due(store, now=time.time() + 1)
        assert not (store.home / trash.SWEEP_DUE).exists()

    def test_a_mark_it_cannot_read_sweeps_nothing(self, stack):
        (stack.store.home / trash.SWEEP_DUE).mkdir()
        assert trash.sweep_if_due(stack.store, now=time.time() + 1000) is None

    def test_a_sweep_due_later_is_never_made_earlier(self, tmp_path):
        later = time.time() + trash.GRACE / 2  # later, within the grace
        trash._write_due(tmp_path, later)
        trash._write_due(tmp_path, time.time())
        assert float((tmp_path / trash.SWEEP_DUE).read_text(encoding="utf-8")) == pytest.approx(later, abs=0.01)

    def test_a_time_due_now_is_never_read_as_the_future(self, tmp_path):
        # Found in review: written to the millisecond, it rounded up, and a sweep asked at once
        # found it not due yet, one run in seven.
        for _ in range(200):
            now = time.time()
            trash._write_due(tmp_path, now)
            assert trash._read_due(tmp_path / trash.SWEEP_DUE)[0] <= now
            (tmp_path / trash.SWEEP_DUE).unlink()

    def test_a_busy_log_is_tried_again_without_sweeping_again(self, stack, monkeypatch):
        import sqlite3

        store = stack.store
        batch(stack, store.home / "batches", n=1)
        sweeps = []
        sweep = trash._sweep
        monkeypatch.setattr(trash, "_sweep", lambda *args, **kwargs: sweeps.append(1) or sweep(*args, **kwargs))
        reader = sqlite3.connect(store.home / "tuningfork.db")
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM runs").fetchone()
            trash._put_due(store.home / trash.SWEEP_DUE, time.time() - 1, False)
            trash.sweep_if_due(store, now=time.time() + 1)
            trash.sweep_if_due(store, now=time.time() + 1)
            assert len(sweeps) == 1 and trash._read_due(store.home / trash.SWEEP_DUE)[1]  # the log alone
        finally:
            reader.close()
        trash.sweep_if_due(store, now=time.time() + 1)
        assert len(sweeps) == 1 and not (store.home / trash.SWEEP_DUE).exists()

    def test_a_whole_sweep_asked_for_stays_asked_for(self, tmp_path):
        trash._write_due(tmp_path, time.time())
        trash._write_due(tmp_path, time.time(), only_the_log=True)  # a job's purge, the log busy
        assert trash._read_due(tmp_path / trash.SWEEP_DUE)[1] is False
        (tmp_path / trash.SWEEP_DUE).unlink()
        trash._write_due(tmp_path, time.time(), only_the_log=True)
        assert trash._read_due(tmp_path / trash.SWEEP_DUE)[1] is True

    def test_a_mark_set_far_ahead_is_kept_no_later_than_the_grace(self, tmp_path):
        trash._put_due(tmp_path / trash.SWEEP_DUE, float("inf"), False)
        now = time.time()
        trash._write_due(tmp_path, now)
        assert trash._read_due(tmp_path / trash.SWEEP_DUE)[0] <= now + trash.GRACE + 1

    def test_a_mark_that_cannot_be_written_fails_no_deletion(self, stack):
        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        (store.home / trash.SWEEP_DUE).mkdir()

        deleted(store, run)

        assert store.get_run(run) is None and not list(store.home.glob(f"{trash.SWEEP_DUE}.*.tmp"))


class TestNamedElsewhere:
    def test_a_skill_copy_a_scenario_in_the_trash_names_stays(self, stack):
        # Its skill folder written relative to its own folder, where it comes back (found in
        # review: read against the trash's, the copy went, and the scenario came back without it).
        from tuningfork.ops.scenarios import delete_scenario

        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        copy = group_of(store, run) / "skill-snapshot" / "tally-report"
        scenario = store.home / "scenarios" / "named"
        scenario.mkdir(parents=True)
        (scenario / "evals.json").write_text(json.dumps({
            "skill_name": "tally-report", "tuningfork": {"skill_dir": os.path.relpath(copy, scenario)}, "evals": [],
        }), encoding="utf-8")
        delete_scenario(store.home, "named")

        deleted(store, run)

        assert (copy / "SKILL.md").is_file()

    def test_a_skill_copy_a_task_file_names_stays_while_a_run_of_it_is_left(self, stack):
        store = stack.store
        (first,) = batch(stack, store.home / "batches", n=1)
        copy = group_of(store, first) / "skill-snapshot" / "tally-report"
        data = json.loads(stack.spec.path.read_text(encoding="utf-8"))
        data["tuningfork"]["skill_dir"] = str(copy)
        other = stack.tmp / "other.json"
        other.write_text(json.dumps(data), encoding="utf-8")
        execute_batch(store, prepare_batch(store, other, base_url=f"{stack.url}/v1",
                                           workdir=store.home / "batches", model="m", harness="fake", n=1))

        deleted(store, first)

        assert (copy / "SKILL.md").is_file()

    def test_a_skill_added_from_a_runs_own_folder_stays_and_the_rest_of_it_goes(self, stack):
        # Found in review: the whole folder stayed, the run's conversation and reply with it.
        from tuningfork.ops import addskill

        store = stack.store
        (run,) = batch(stack, stack.tmp / "elsewhere", n=1)
        folder = group_of(store, run) / "0"
        kept = folder / "kept" / "kept-skill"
        kept.mkdir(parents=True)
        (kept / "SKILL.md").write_text("---\nname: kept-skill\ndescription: a person's own\n---\n", encoding="utf-8")
        (folder / "kept" / "beside.txt").write_text("not named\n", encoding="utf-8")
        addskill.execute_add(store, addskill.prepare_folder(store, kept))
        trash.trash(store, "run", run)
        assert any(str(kept) in note for note in trash.preview(store, "run", run)["notes"])

        trash.purge(store, "run", run)

        assert (kept / "SKILL.md").is_file()
        assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*")) == [
            "kept", "kept/kept-skill", "kept/kept-skill/SKILL.md",
        ]

    def test_a_skill_added_from_an_imports_folder_stays_and_the_rest_of_it_goes(self, stack):
        from tuningfork.ops import addskill
        from tuningfork.ops.imports import root as imports

        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        folder = imports(store.home) / run
        kept = folder / "skills" / "kept-skill"
        kept.mkdir(parents=True)
        (kept / "SKILL.md").write_text("---\nname: kept-skill\ndescription: a person's own\n---\n", encoding="utf-8")
        (folder / "chat.json").write_text("[]\n", encoding="utf-8")
        addskill.execute_add(store, addskill.prepare_folder(store, kept))

        deleted(store, run)

        assert (kept / "SKILL.md").is_file() and not (folder / "chat.json").exists()

    def test_a_skill_added_from_a_refinements_folder_stays_and_the_rest_of_it_goes(self, stack):
        from tuningfork.ops import addskill, refine
        from tests.test_approve import RULES

        store = stack.store
        (run,) = batch(stack, store.home / "batches", n=1)
        made = refine.create(store, run, "write_file", target_rules=RULES)
        kept = made.folder / "kept-skill"
        kept.mkdir()
        (kept / "SKILL.md").write_text("---\nname: kept-skill\ndescription: a person's own\n---\n", encoding="utf-8")
        addskill.execute_add(store, addskill.prepare_folder(store, kept))
        trash.trash(store, "refinement", made.id)

        trash.purge(store, "refinement", made.id)

        assert (kept / "SKILL.md").is_file() and not (made.folder / "refinement.json").exists()

    def test_a_task_file_it_cannot_read_stops_no_deletion(self, stack):
        # Found in review: JSON nested too deep raised past every reader, and the trash could
        # neither list nor delete anything with a launch.
        store = stack.store
        (first,) = batch(stack, store.home / "batches", n=1)
        other = stack.tmp / "other.json"
        other.write_text(stack.spec.path.read_text(encoding="utf-8"), encoding="utf-8")
        execute_batch(store, prepare_batch(store, other, base_url=f"{stack.url}/v1",
                                           workdir=store.home / "batches", model="m", harness="fake", n=1))
        other.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")

        trash.trash(store, "run", first)
        trash.preview(store, "run", first)
        trash.purge(store, "run", first)

        assert store.get_run(first) is None

    def test_a_chains_first_link_never_takes_a_folder_that_was_there(self, stack):
        store = stack.store
        (stack.tmp / "chained" / "taken").mkdir(parents=True)
        plan = prepare_batch(store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "chained",
                             model="m", harness="fake", n=1)
        with pytest.raises(FileExistsError):
            execute_batch(store, plan, group="taken", start=0)
