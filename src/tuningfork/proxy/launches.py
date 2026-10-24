"""Launches as the proxy sees them: which launch a call came from, and where in it it goes.

The harness-facing half of `tuningfork.launches`. A launch's threads are tracked here, in
memory, as its calls arrive; the store is told where each call belongs when it is recorded.
Because that state lives only in this process, a launch whose calls began under an earlier
proxy cannot be continued by this one: nothing here knows where its next call belongs, so it
is refused and marked interrupted rather than guessed at.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from tuningfork.launches import Assignment, LaunchThreads, request_path
from tuningfork.proxy.replay import ForkReplay
from tuningfork.specs import Context
from tuningfork.store.db import Launch, Store

#: Launches kept in memory before finished ones are dropped.
KEEP = 256


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why a launched call cannot be served, as an HTTP status and a message."""

    status: int
    message: str
    kind: str


def volatile_for(harness: str | None) -> tuple[tuple[str, str], ...]:
    """What the harness declares differs between two launches of one task."""
    from tuningfork.harness import ADAPTERS

    adapter = ADAPTERS.get(harness or "")
    return tuple(getattr(adapter(), "volatile", ())) if adapter else ()


def side_call_for(harness: str | None) -> Callable[[dict[str, Any]], str | None] | None:
    """How the harness names its own side calls, if it has any (see `HarnessAdapter`)."""
    from tuningfork.harness import ADAPTERS

    adapter = ADAPTERS.get(harness or "")
    return getattr(adapter(), "side_call", None) if adapter else None


def skill_load_for(harness: str | None) -> Callable[[dict[str, Any]], str | None] | None:
    """How the harness's calls name the skill they load, if it says (see `HarnessAdapter`)."""
    from tuningfork.harness import ADAPTERS

    adapter = ADAPTERS.get(harness or "")
    return getattr(adapter(), "skill_load", None) if adapter else None


def context_note(context: Context) -> dict[str, Any]:
    """What a step records of the conversation put into its call: how many messages before
    the prompt, and in the prompt's turn."""
    return {"messages": len(context.before), **({"in_turn": len(context.in_turn)} if context.in_turn else {})}


def _skill_key(name: str) -> str:
    """A skill's name as two calls may give it: a category before it, or another case."""
    return name.strip().rsplit("/", 1)[-1].lower()


class LaunchState:
    def __init__(self, launch: Launch, store: Store | None = None) -> None:
        self.launch = launch
        #: The store's masking: what a call's path is predicted with (`request_path`).
        self.masking = store.masking if store is not None else None
        self.threads = LaunchThreads(
            launch.id,
            label=launch.label,
            task_id=launch.task_id,
            main_role="fork" if launch.kind == "fork" else "main",
        )
        self.side_call = side_call_for(launch.harness)
        #: For a fork: the recording it answers from, until it goes live.
        self.replay: ForkReplay | None = None
        if launch.kind == "fork" and store is not None:
            self.replay = ForkReplay(store, launch, volatile=volatile_for(launch.harness))
        #: For a scenario's run: the conversation it continues (`tuningfork.context`), put
        #: before the harness's own messages in each call of its main thread — and the model's
        #: earlier steps of the prompt's turn, after the prompt.
        self.context: Context | None = None
        kept = launch.settings.get("context")
        if isinstance(kept, dict) and kept.get("blob") and store is not None:
            self.context = Context.read(store.blobs.get_json(kept["blob"]))
        #: For a scenario's run that ends where the model loads another skill
        #: (`tuningfork.stop_at_next_skill`): the skill it studies, how the harness's calls
        #: name the skill they load, and — once it has ended there — where.
        stop = launch.settings.get("stop_at_next_skill")
        self.studied: str | None = stop.get("skill") if isinstance(stop, dict) else None
        self.skill_load = skill_load_for(launch.harness) if self.studied else None
        self.stopped: dict[str, Any] | None = launch.settings.get("stopped_at_skill")
        #: One snapshot at a time: a launch's threads can call at once, and each snapshot
        #: moves the launch's ref.
        self.snapshot_lock = threading.Lock()

    def in_context(self, assignment: Assignment) -> bool:
        """Whether this call continues the scenario's conversation: one of the main thread's.
        A subagent's has its own task, and the harness's side calls (Hermes's review) are its
        own, not the task's next step."""
        return self.context is not None and assignment.main

    def next_skill(self, payload: dict[str, Any], wire: Any, assignment: Assignment) -> str | None:
        """The skill other than the one studied that the model's latest answer loads, as the
        harness sends it back in this call of the main thread: where a run that stops at the
        next skill ends. None for any other call."""
        if self.skill_load is None or not assignment.main:
            return None
        view = wire.view(payload)
        latest = next((message for message in reversed(view)
                       if isinstance(message, dict) and message.get("role") == "assistant"), None)
        for call in (latest or {}).get("tool_calls") or []:
            loaded = self.skill_load(call)
            if loaded and _skill_key(loaded) != _skill_key(self.studied or ""):
                return loaded
        return None

    @property
    def workspace(self) -> str | None:
        """The workspace to snapshot as calls arrive, when the launch has one and wants it."""
        if not self.launch.settings.get("snapshots", True):
            return None
        return self.launch.workspace

    @property
    def breakpoints(self) -> bool:
        return self.launch.breakpoints

    @property
    def records(self) -> bool:
        """Whether its calls are recorded. A relay's are not: it carries another proxy's
        calls — a held-out task's, recorded only where no agent can read them — through this
        proxy's gate and pause, and nothing else."""
        return bool(self.launch.settings.get("record", True))

    def assign(self, payload: dict[str, Any], wire: Any) -> Assignment:
        side = bool(self.side_call and self.side_call(payload))
        return self.threads.assign(request_path(payload, wire, self.masking), side=side)

    def release(self, assignment: Assignment) -> None:
        self.threads.release(assignment)


class LaunchRegistry:
    """The launches this proxy is serving. `resolve` reads the store: call it off the loop."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._states: dict[str, LaunchState] = {}
        self._lock = threading.Lock()

    def resolve(self, token: str) -> LaunchState | Refusal:
        with self._lock:
            state = self._states.get(token)
        if state is not None:
            return state
        launch = self.store.get_launch(token)
        if launch is None:
            return Refusal(
                404,
                f"tuningfork has no launch {token[:8]} in {self.store.home}. Is this proxy "
                "recording into the same home (--home) as the command that started the harness?",
                "tuningfork_unknown_launch",
            )
        if launch.status != "running":
            return Refusal(409, f"launch {token[:8]} has {launch.status}", "tuningfork_launch_over")
        if self.store.launch_runs(token):
            # Calls of it were recorded, but not through this process: the proxy restarted
            # while the launch ran, and cannot know where its next call belongs.
            self.store.end_launch(
                token, status="interrupted", exit={"reason": "the proxy restarted while it ran"}
            )
            return Refusal(
                409,
                f"launch {token[:8]} was interrupted: the proxy restarted while it ran",
                "tuningfork_launch_interrupted",
            )
        try:
            fresh = LaunchState(launch, self.store)
        except ValueError as exc:
            self.store.end_launch(token, status="interrupted", exit={"reason": str(exc)})
            return Refusal(409, f"launch {token[:8]} cannot be replayed: {exc}", "tuningfork_replay")
        with self._lock:
            state = self._states.setdefault(token, fresh)
            if len(self._states) > KEEP:
                self._forget_finished()
        return state

    def live(self) -> list[str]:
        """Launches this proxy has served that are still running. Reads the store."""
        with self._lock:
            tokens = list(self._states)
        running = []
        for token in tokens:
            launch = self.store.get_launch(token)
            if launch is not None and launch.status == "running":
                running.append(token)
        return running

    def _forget_finished(self) -> None:
        for token in list(self._states):
            launch = self.store.get_launch(token)
            if launch is None or launch.status != "running":
                del self._states[token]
