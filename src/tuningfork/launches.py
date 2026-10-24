"""Launches: every harness process tuningfork starts, and the calls it makes.

A batch run, or a fork of one, is a *launch*. It gets a token of its own, and the harness is
pointed at `http://<proxy>/_launch/<token>/…` instead of the proxy's root. The proxy strips
the prefix before relaying, so the harness and the server see nothing different — but every
call now says which launch it came from, subagents included, since both harnesses give a
subagent the parent's base URL.

That makes attribution exact, and decided when a call *arrives*. Without it, runs are told
apart by their message paths alone: N copies of one task open with byte-identical requests,
a fork replaying a recording sends the same requests as every other fork of it, and a step's
place in its run was only known once it had been recorded — after the model answered.

Within a launch, calls fall into threads. A call that extends a thread's last request joins
that thread; one that repeats it is a retry in the same thread; a call that continues none
is a new thread. The first thread is the main one — the task's own conversation; later ones
are a subagent's or a harness's side call, told apart once the launch has ended
(`Store.finalize_threads`), from when each began. A call its harness names a side call
(`side_call` on the adapter) never extends the main thread, though it continues its
conversation: Hermes's background review replays the whole conversation after the answer,
and filed as the task's next step, its reply — cut off when Hermes exits — was judged as
the run's.
"""

from __future__ import annotations

import re
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from tuningfork.store.tree import node_path

PREFIX = "/_launch/"
_PATH = re.compile(r"^/_launch/([0-9a-f]{32})(/.*)?$")


def new_token() -> str:
    """128 random bits. Model traffic is not behind the loopback guard (it belongs to the
    harness), so a token must not be guessable: it selects replay plans and snapshots."""
    return secrets.token_hex(16)


def split(path: str) -> tuple[str | None, str]:
    """(token, the path the harness meant) for a request path; (None, path) for no prefix."""
    match = _PATH.match(path)
    if not match:
        return None, path
    return match.group(1), match.group(2) or "/"


def launch_url(base_url: str, token: str) -> str:
    """The base URL a launched harness is given: the proxy's, with the launch in its path.

    `http://127.0.0.1:8787/v1` becomes `http://127.0.0.1:8787/_launch/<token>/v1`; whatever
    path the harness expects is kept after the prefix.
    """
    parts = urlsplit(base_url)
    path = f"{PREFIX}{token}{parts.path.rstrip('/')}" if parts.path.strip("/") else f"{PREFIX}{token}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def proxy_root(base_url: str) -> str:
    """The proxy's own root, from the base URL a harness is given."""
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def request_path(request: dict[str, Any], wire: Any = None, masking: Any = None) -> list[str]:
    """The request's node path, exactly as `Store.record_step` will build it: masked as the
    store masks (`masking`, the store's `Masking`), then in the wire format's stable form.

    A path built from the raw request would never be found once a mask had changed a message
    — and neither would every step after it.
    """
    from tuningfork.store.masking import Masking

    stored = (masking or Masking()).value({"messages": request.get("messages") or []})
    messages = stored.get("messages") or []
    if wire is not None:
        messages = [wire.normalize_message(m) for m in messages]
    return node_path(messages)


@dataclass(frozen=True, slots=True)
class Assignment:
    """Where a launched call belongs, fixed when it arrived."""

    launch_id: str
    run_id: str
    idx: int
    #: 0 for the launch's first thread: the task's own conversation.
    thread: int
    #: The run's label, for a thread the store has not seen yet.
    label: str | None = None
    #: The launch's task. Every thread works on it, so a breakpoint scoped to the task stops
    #: a subagent's calls too; only the main thread's run is filed under it.
    task_id: str | None = None
    #: The run's role (see `tuningfork.store.db`): the main thread's is the launch's own —
    #: `main`, or `fork` for a replay — and any other thread's is decided at the end.
    role: str = "main"

    @property
    def main(self) -> bool:
        return self.thread == 0


@dataclass(slots=True)
class _Thread:
    run_id: str
    head: tuple[str, ...]
    next_idx: int = 0
    #: `side` for a side call its harness named, filed so as it arrives: the end of the
    #: launch must not take it for a subagent because it replays a delegated goal.
    role: str = "thread"


@dataclass
class LaunchThreads:
    """One launch's threads, as its calls arrive. Safe to use from several threads."""

    launch_id: str
    label: str | None = None
    task_id: str | None = None
    #: The main thread's role: `main` for a batch run, `fork` for a replay.
    main_role: str = "main"
    threads: list[_Thread] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def assign(self, path: Sequence[str], *, side: bool = False) -> Assignment:
        path = tuple(path)
        with self._lock:
            chosen = self._thread_for(path)
            if chosen is None or (side and chosen == 0):
                chosen = len(self.threads)
                self.threads.append(
                    _Thread(run_id=uuid.uuid4().hex, head=path, role="side" if side else "thread")
                )
            thread = self.threads[chosen]
            thread.head = path
            idx = thread.next_idx
            thread.next_idx += 1
        return Assignment(
            launch_id=self.launch_id,
            run_id=thread.run_id,
            idx=idx,
            thread=chosen,
            label=self.label if chosen == 0 else _thread_label(self.label, chosen),
            task_id=self.task_id,
            role=self.main_role if chosen == 0 else thread.role,
        )

    def release(self, assignment: Assignment) -> None:
        """Give back the index of a call that will never be recorded — one dropped before
        the model saw it. The harness's retry then takes the same step number."""
        with self._lock:
            if assignment.thread >= len(self.threads):
                return
            thread = self.threads[assignment.thread]
            if thread.next_idx == assignment.idx + 1:
                thread.next_idx = assignment.idx

    def _thread_for(self, path: tuple[str, ...]) -> int | None:
        """Which thread a call continues: the one whose last request it extends the
        furthest, else one it repeats (a retry), else none."""
        extends: list[tuple[int, int]] = []
        repeats: list[int] = []
        for index, thread in enumerate(self.threads):
            head = thread.head
            if not head:
                continue
            if len(head) < len(path) and path[: len(head)] == head:
                extends.append((len(head), index))
            elif path == head:
                repeats.append(index)
        if extends:
            return max(extends)[1]
        if repeats:
            return repeats[-1]
        return None


def _thread_label(label: str | None, thread: int) -> str | None:
    return f"{label}~{thread}" if label else None
