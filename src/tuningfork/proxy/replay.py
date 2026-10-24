"""A fork's calls, as the proxy answers them: from the recording until the step it goes live
at, from the model after. See `tuningfork.replay` for what a fork is and why."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from tuningfork.replay import (
    ReplayTable,
    Served,
    Variant,
    build_table,
    first_difference,
    key_of,
    run_paths,
)
from tuningfork.store.db import Launch, Store


@dataclass(slots=True)
class Decision:
    """What to do with one call of a fork."""

    #: Answer it with this recorded answer.
    served: Served | None = None
    #: Send it to the model. Set for every call from the fork's step on — and for a call
    #: the recording has no answer to, which is then also a `mismatch`.
    live: bool = False
    #: How a call that should have been answered from the recording differed from it.
    mismatch: dict[str, Any] | None = None
    #: The call at the fork's step itself: where the recording hands over to the model.
    at_step: bool = False


class ForkReplay:
    """One fork launch's replay state. Built when its first call arrives; lives as long
    as the proxy does, like the rest of a launch's state."""

    def __init__(self, store: Store, launch: Launch, patterns: Any = None, volatile: Any = ()) -> None:
        source = store.get_run(launch.source_run or "")
        if source is None or launch.source_step is None:
            raise ValueError(f"fork {launch.id[:8]} names no recorded run to replay")
        self.source = source
        self.table: ReplayTable = build_table(
            store, source, launch.source_step, patterns=patterns, volatile=volatile
        )
        workspace, home = run_paths(store, source)
        self.pairs: list[tuple[str, str]] = []
        if workspace and launch.workspace:
            self.pairs.append((str(workspace), launch.workspace))
        if home and launch.home:
            self.pairs.append((str(home), launch.home))
        self.variant = Variant.from_json(launch.variant)
        self.live = False
        self.departed: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def decide(self, payload: dict[str, Any], wire: Any, assignment: Any) -> Decision:
        text = self.table.text(payload, wire)
        key = key_of(text)
        with self._lock:
            if key != self.table.live_key and not self.table.entries.get(key):
                # No recorded call has its key: one stored under other masks may still be it.
                key = self.table.fitting(payload, wire) or key
            if not self.live and key == self.table.live_key and assignment.main:
                self.live = True
                return Decision(live=True, at_step=True)
            served = self.table.take(key)
            if served is not None:
                return Decision(served=served)
            if self.live:
                return Decision(live=True)
            # Before the fork's step, every call should have been answered from the
            # recording. One that was not asked for something the recording never
            # answered: from here the fork's history is not the recorded one.
            expected = self.table.expected.get(assignment.idx) if assignment.main else None
            mismatch: dict[str, Any] = {"thread": assignment.thread, "step": assignment.idx}
            if expected is not None:
                mismatch.update(first_difference(expected, text))
            self.live = True
            self.departed = self.departed or mismatch
            return Decision(live=True, mismatch=mismatch)

    def give_back(self, decision: Decision) -> None:
        """A call decided at the fork's step, then never recorded (refused while paused, or
        dropped at a breakpoint): its retry is the step again. Found in review: the fork was
        live already, and the retry was recorded with no word that it went live there."""
        if decision.at_step:
            with self._lock:
                self.live = False

    def served_note(self, served: Served) -> dict[str, Any]:
        return {
            "served_from": served.step.id,
            "source_run": served.step.run_id,
            "source_idx": served.step.idx,
        }
