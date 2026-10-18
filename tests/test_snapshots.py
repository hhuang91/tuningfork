"""Workspace snapshots, in a git object store of tuningfork's own."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tuningfork.snapshots import Snapshots, available

pytestmark = pytest.mark.skipif(not available(), reason="snapshots need git on PATH")


@pytest.fixture
def snapshots(tmp_path) -> Snapshots:
    return Snapshots(tmp_path / "home")


def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    (root / "sub").mkdir(parents=True)
    (root / "data.csv").write_text("id,amount\n1,10\n", encoding="utf-8")
    (root / "sub" / "notes.txt").write_text("hello\n", encoding="utf-8")
    script = root / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)
    return root


class TestTheStore:
    def test_two_launches_starting_at_once_on_a_fresh_home_both_snapshot(self, tmp_path):
        """Found in use: a batch of two at a time on a new recordings folder lost its first
        run's snapshot — both launches found no store, and the second `git init` failed on
        the first's half-made directory. Here the second launch's store is made while the
        first is still making its own, and both go on to commit."""
        home = tmp_path / "home"
        root = workspace(tmp_path)

        class Racing(Snapshots):
            def _init_aside(self):
                staging = super()._init_aside()
                Snapshots(home).ensure()  # the other launch wins the race meanwhile
                return staging

        first = Racing(home)
        first.ensure()
        assert (home / "snapshots.git" / "HEAD").exists()
        a = first.commit(root, "refs/launches/a/start")
        b = Snapshots(home).commit(root, "refs/launches/b/start")
        # Both committed the same workspace. A commit's id carries the second it was made in, so
        # two made across a second boundary differ (seen once, on CI); what they hold does not.
        assert first.resolve("refs/launches/a/start") == a and first.resolve("refs/launches/b/start") == b
        assert first.tree_of(a) == first.tree_of(b)
        assert [p.name for p in home.iterdir()] == ["snapshots.git"]  # no staging left beside it


class TestTaking:
    def test_an_empty_workspace_restores_as_empty(self, snapshots, tmp_path):
        """Found in use: a task judged on its reply alone leaves its workspace empty, and its
        snapshot — a pax header and no files — could not be opened, so no such run could
        be rechecked."""
        empty = tmp_path / "empty"
        empty.mkdir()
        commit = snapshots.commit(empty, "refs/launches/t/end")

        snapshots.restore(commit, tmp_path / "restored")

        assert (tmp_path / "restored").is_dir() and not any((tmp_path / "restored").iterdir())

    def test_a_snapshot_restores_as_it_was(self, snapshots, tmp_path):
        root = workspace(tmp_path)
        os.symlink("data.csv", root / "link.csv")
        commit = snapshots.commit(root, "refs/launches/t/calls")

        target = tmp_path / "restored"
        snapshots.restore(commit, target)

        assert (target / "data.csv").read_text(encoding="utf-8") == "id,amount\n1,10\n"
        assert (target / "sub" / "notes.txt").read_text(encoding="utf-8") == "hello\n"
        assert os.access(target / "run.sh", os.X_OK)
        assert os.readlink(target / "link.csv") == "data.csv"

    def test_nothing_the_workspace_ignores_is_left_out_and_no_git_is_left_in_it(
        self, snapshots, tmp_path
    ):
        """A task's own `.gitignore` or repository is the task's business, not the
        snapshot's: every file is kept, and nothing is written into the workspace."""
        root = workspace(tmp_path)
        (root / ".gitignore").write_text("*.csv\nsub/\n", encoding="utf-8")
        nested = root / "project"
        nested.mkdir()
        subprocess.run(["git", "init", "-q", str(nested)], check=True, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull})
        (nested / "code.py").write_text("print(1)\n", encoding="utf-8")

        commit = snapshots.commit(root, "refs/launches/t/calls")

        paths = {entry.path for entry in snapshots.files(commit)}
        assert {"data.csv", "sub/notes.txt", ".gitignore", "project/code.py"} <= paths
        assert not any(path.startswith("project/.git/") for path in paths)
        assert not (root / ".git").exists()

    def test_the_users_git_configuration_is_never_read(self, snapshots, tmp_path, monkeypatch):
        home = tmp_path / "user"
        home.mkdir()
        (home / ".gitconfig").write_text("[user]\n\tname = somebody\n[core]\n\thooksPath = /nope\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))

        commit = snapshots.commit(workspace(tmp_path), "refs/launches/t/calls")

        author = subprocess.run(
            ["git", "--git-dir", str(snapshots.git_dir), "log", "-1", "--format=%an", commit],
            capture_output=True, text=True, check=True, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull},
        ).stdout.strip()
        assert author == "tuningfork"

    def test_snapshots_on_a_ref_chain_and_diff(self, snapshots, tmp_path):
        root = workspace(tmp_path)
        first = snapshots.commit(root, "refs/launches/t/calls")
        (root / "report.json").write_text('{"row_count": 4}', encoding="utf-8")
        (root / "data.csv").write_text("changed\n", encoding="utf-8")
        second = snapshots.commit(root, "refs/launches/t/calls")

        assert snapshots.resolve("refs/launches/t/calls") == second
        assert sorted(snapshots.diff(first, second)) == [("A", "report.json"), ("M", "data.csv")]
        assert "+changed" in snapshots.patch(first, second, "data.csv")
        assert snapshots.read(second, "report.json") == b'{"row_count": 4}'
        # The same contents make the same tree, whatever the commit.
        again = snapshots.commit(root, "refs/launches/u/calls")
        assert snapshots.tree_of(again) == snapshots.tree_of(second)
