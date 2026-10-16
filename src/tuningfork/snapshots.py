"""Workspace snapshots: what a run's workspace held when each of its calls arrived.

A checkpoint's environment is the workspace at the moment call k arrives — every earlier
tool effect is on disk by then. So the proxy commits a launched run's workspace as each call
arrives, and the step keeps the commit; the runner commits the workspace a run started from,
the one it ended with, and the frozen skill it was given. A fork starts from its source's
initial snapshot, and at every replayed call its workspace can be compared with the
source's at the same point.

Everything goes into one git object store under the tuningfork home (`snapshots.git`),
never into the workspace: a `.git` there would change what the harness sees — Claude Code
reports git status to the model — and a task's own repository must be left alone. The
store deduplicates across runs, so N forks of one run cost little more than one.

A snapshot is built from a walk of the directory, not with `git add`: `git add` would skip
what the workspace's own `.gitignore` names, and record a nested repository as a bare
pointer to its HEAD rather than the files in it. Only `.git` directories themselves are
left out. Each commit uses a throwaway index, so the proxy and the runner — separate
processes — never contend for a lock, and a user's git configuration is never read.

What a snapshot cannot see is anything outside the workspace: installed packages, `/tmp`,
the network. A failed snapshot is noted, and never stops the call it was taken for.
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: A snapshot that takes longer than this is abandoned, and the call goes ahead without it.
TIMEOUT = 30.0
#: Fixed so that a commit's identity says nothing about who ran tuningfork.
IDENTITY = {
    "GIT_AUTHOR_NAME": "tuningfork",
    "GIT_AUTHOR_EMAIL": "snapshots@tuningfork.invalid",
    "GIT_COMMITTER_NAME": "tuningfork",
    "GIT_COMMITTER_EMAIL": "snapshots@tuningfork.invalid",
}


class SnapshotError(RuntimeError):
    """A snapshot could not be taken or read."""


def available() -> bool:
    return shutil.which("git") is not None


@dataclass(frozen=True, slots=True)
class Entry:
    path: str
    mode: str
    size: int | None
    #: The file's content, by its object id: the same text in two snapshots is one object.
    sha: str | None = None


class Snapshots:
    """The snapshot store of one tuningfork home."""

    def __init__(self, home: Path) -> None:
        self.git_dir = Path(home) / "snapshots.git"

    # --- plumbing -------------------------------------------------------------------

    def _env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "GIT_DIR": str(self.git_dir),
            # Never a user's configuration: hooks, a global excludes file, signing.
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C",
            **IDENTITY,
        }
        env.update(extra)
        return env

    def _git(self, *args: str, input: bytes | None = None, **env: str) -> bytes:
        try:
            done = subprocess.run(
                ["git", *args],
                input=input,
                capture_output=True,
                env=self._env(**env),
                timeout=TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SnapshotError(f"git {args[0]}: {exc}") from None
        if done.returncode != 0:
            raise SnapshotError(
                f"git {args[0]} failed: {done.stderr.decode('utf-8', 'replace').strip()}"
            )
        return done.stdout

    def ensure(self) -> None:
        """Make the store on first use: beside it, then renamed into place. Found in use: a
        batch of two runs at a time on a fresh home had both launches find no store, and the
        second `git init` failed on the first's half-made directory, losing a snapshot. A
        rename is atomic, and one that loses the race finds the winner's store complete."""
        if (self.git_dir / "HEAD").exists():
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self._init_aside()
        try:
            os.rename(staging, self.git_dir)
        except OSError as exc:
            if not (self.git_dir / "HEAD").exists():
                raise SnapshotError(f"cannot create {self.git_dir}: {exc}") from None
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _init_aside(self) -> Path:
        staging = Path(tempfile.mkdtemp(prefix=".snapshots-", dir=self.git_dir.parent))
        self._git("init", "--bare", "--quiet", str(staging), GIT_DIR=str(staging))
        return staging

    # --- taking ---------------------------------------------------------------------

    def tree(self, directory: Path) -> str:
        """Store a directory's contents; return the git tree of them."""
        self.ensure()
        directory = Path(directory)
        files = list(_walk(directory))
        paths = [str(directory / path) for path, _ in files if not _is_link(directory / path)]
        blobs = self._git("hash-object", "-w", "--stdin-paths", input="\n".join(paths).encode()).split() if paths else []
        shas = iter(blob.decode() for blob in blobs)
        lines = []
        for path, mode in files:
            if mode == "120000":
                target = os.readlink(directory / path).encode()
                sha = self._git("hash-object", "-w", "--stdin", input=target).decode().strip()
            else:
                sha = next(shas)
            lines.append(f"{mode} {sha}\t{path}")
        with tempfile.TemporaryDirectory() as scratch:
            index = str(Path(scratch) / "index")
            if lines:
                self._git(
                    "update-index", "--add", "--index-info",
                    input=("\n".join(lines) + "\n").encode(), GIT_INDEX_FILE=index,
                )
            return self._git("write-tree", GIT_INDEX_FILE=index).decode().strip()

    def commit(self, directory: Path, ref: str, message: str = "snapshot") -> str:
        """Commit a directory's contents on top of `ref`, and move `ref` to it.

        Commits chain by parent, so a ref keeps every snapshot it was given alive.
        """
        tree = self.tree(directory)
        parent = self.resolve(ref)
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        sha = self._git(*args).decode().strip()
        self._git("update-ref", ref, sha, *([parent] if parent else []))
        return sha

    def drop(self, launch_id: str) -> int:
        """Forget a launch's snapshots: its refs go, and what only they reached goes at the
        next `prune`. Returns how many refs."""
        if not (self.git_dir / "HEAD").exists():
            return 0
        refs = self._git("for-each-ref", "--format=%(refname)", f"refs/launches/{launch_id}/").decode().split()
        for ref in refs:
            self._git("update-ref", "-d", ref)
        return len(refs)

    def prune(self, grace: str = "2.minutes.ago") -> None:
        """Delete the objects no ref reaches, but those written within `grace`: a snapshot
        in flight writes its objects before the ref that keeps them."""
        if (self.git_dir / "HEAD").exists():
            self._git("prune", f"--expire={grace}")

    def resolve(self, ref: str) -> str | None:
        if not (self.git_dir / "HEAD").exists():
            return None
        try:
            return self._git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").decode().strip() or None
        except SnapshotError:
            return None

    # --- reading --------------------------------------------------------------------

    def tree_of(self, commit: str) -> str:
        return self._git("rev-parse", f"{commit}^{{tree}}").decode().strip()

    def files(self, commit: str) -> list[Entry]:
        out = self._git("ls-tree", "-r", "-l", "-z", commit)
        entries = []
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, _kind, sha, size = meta.split()
            entries.append(Entry(
                path=path.decode(), mode=mode.decode(), size=None if size == b"-" else int(size), sha=sha.decode(),
            ))
        return entries

    def diff(self, before: str | None, after: str) -> list[tuple[str, str]]:
        """(status, path) for each file that changed: A added, M modified, D deleted."""
        if before is None:
            return [("A", entry.path) for entry in self.files(after)]
        out = self._git("diff-tree", "-r", "--no-renames", "--name-status", "-z", before, after)
        fields = [field.decode() for field in out.split(b"\0") if field]
        return list(zip(fields[0::2], fields[1::2]))

    def patch(self, before: str | None, after: str, path: str | None = None) -> str:
        """A unified diff between two snapshots, of one file or all of them."""
        empty = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree
        args = ["diff", "--no-color", before or empty, after]
        if path:
            args += ["--", path]
        return self._git(*args).decode("utf-8", "replace")

    def read(self, commit: str, path: str) -> bytes:
        return self._git("cat-file", "blob", f"{commit}:{path}")

    def blob(self, sha: str) -> bytes:
        """A file's content by its object id (`Entry.sha`), whichever snapshot holds it."""
        return self._git("cat-file", "blob", sha)

    def restore(self, commit: str, target: Path) -> None:
        """Write a snapshot's files into `target`, which is created if need be."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        # An empty workspace — a task judged on its reply alone — archives as a pax header
        # and nothing else, which tarfile cannot open (found in use: no such run could be
        # rechecked). There is nothing to write.
        if not self._git("ls-tree", "--name-only", commit).strip():
            return
        archive = self._git("archive", "--format=tar", commit)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(target, filter="data")


def _is_link(path: Path) -> bool:
    return path.is_symlink()


def _walk(root: Path) -> Iterator[tuple[str, str]]:
    """(relative path, git mode) for every file under `root`, `.git` directories left out."""
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d != ".git")
        for name in sorted(names):
            if "\n" in name or "\t" in name:
                continue  # git's line-based plumbing cannot name it; vanishingly rare
            full = Path(current) / name
            relative = full.relative_to(root).as_posix()
            info = full.lstat()
            if stat.S_ISLNK(info.st_mode):
                yield relative, "120000"
            elif stat.S_ISREG(info.st_mode):
                yield relative, "100755" if info.st_mode & stat.S_IXUSR else "100644"
        # A symlink to a directory is listed with the names (followlinks=False), not walked.
        for name in list(directories):
            if (Path(current) / name).is_symlink():
                directories.remove(name)
                relative = (Path(current) / name).relative_to(root).as_posix()
                yield relative, "120000"
