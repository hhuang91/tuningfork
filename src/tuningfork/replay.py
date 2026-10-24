"""Replaying a recorded run to a relaunched harness: what a fork is made of.

A harness cannot be resumed from a list of messages, so a fork rebuilds its state the only
general way there is. It relaunches the harness on the same task, in a fresh workspace, and
answers the harness's calls from the recording — the harness re-runs every recorded tool
call for real — until the call at the step the fork goes live at. From there the model
answers.

Which recorded answer a call gets is decided by content: every recorded call before that
step, in any of the run's threads, is keyed by its request with the per-launch detail masked
out, and a call gets the next answer recorded for its key. So a subagent's calls, a retry,
and calls arriving in a different order all find their answers, and a call that finds none
is a *mismatch*: the harness asked something the recording never answered, so the fork's
history is no longer the recorded one. Such a fork is not a sample of "from step k".

Served answers are the recorded ones with the source run's workspace and home paths swapped
for the fork's. Claude Code names absolute paths in its tool calls; replayed verbatim, those
would make every fork read and write the *source* run's workspace — and mask the fact, since
per-run paths are masked when requests are compared.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from tuningfork.analysis.normalize import mask_workspace
from tuningfork.config import DEFAULT_REDACTIONS, REDACTED, patterns_digest, redact_json
from tuningfork.store.db import Run, Step, Store, arrived
from tuningfork.wire import get_wire

#: How much of each side of a mismatch a report quotes.
CONTEXT = 90


def masked_request(
    request: dict[str, Any],
    wire: Any,
    patterns: Sequence[str] = DEFAULT_REDACTIONS,
    volatile: Iterable[tuple[str, str]] = (),
) -> str:
    """A harness request as text, with what differs between two launches of one task
    masked: redacted as the recorder stores it, messages in their stable form, per-run
    paths, and the fields the harness declares volatile (a session id, today's date)."""
    stored = redact_json(request, tuple(patterns))
    stored = dict(stored) if isinstance(stored, dict) else {"request": stored}
    if isinstance(stored.get("messages"), list):
        stored["messages"] = [wire.normalize_message(m) for m in stored["messages"]]
    text = mask_workspace(json.dumps(stored, sort_keys=True, ensure_ascii=False))
    for pattern, replacement in volatile:
        text = re.sub(pattern, replacement, text)
    return text


def key_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fits(recorded: str, text: str) -> bool:
    """Whether `text` reads as `recorded` does but where `recorded` holds a mask, each standing
    for what a mask hides: one or more characters, no space and no quote — a call keyed as
    today's patterns mask it, against a request stored under others. Found in review, every
    fork of a run recorded before the patterns changed parted from its recording where they had
    masked code they now keep; and a mask that stood for any text let a call that parted from
    the recording right after one take its answer."""
    if REDACTED not in recorded:
        return recorded == text
    hidden = r'[^\s"]+?'
    return re.fullmatch(hidden.join(re.escape(piece) for piece in recorded.split(REDACTED)), text) is not None


@dataclass(slots=True)
class Served:
    """A recorded answer, ready to be given again."""

    step: Step
    #: The thread it came from in the source run (0: the main thread).
    thread: int


@dataclass
class ReplayTable:
    """The recorded answers a fork may give, by masked request."""

    entries: dict[str, deque[Served]]
    #: The key of the call at the step the fork goes live at, and that recorded call.
    live_key: str
    live_step: Step
    #: The source's main thread, masked, by step: what a mismatch is compared against.
    expected: dict[int, str]
    step: int
    volatile: tuple[tuple[str, str], ...] = ()
    patterns: tuple[str, ...] = DEFAULT_REDACTIONS
    #: The recorded calls masked under other patterns than these (`config.patterns_digest`,
    #: kept as `redaction`) whose request holds a mask, as (key, its text unredacted): what a
    #: call matches when its key is no recorded one's (`fitting`).
    masked: list[tuple[str, str]] = field(default_factory=list)

    def text(self, request: dict[str, Any], wire: Any) -> str:
        return masked_request(request, wire, self.patterns, self.volatile)

    def fitting(self, request: dict[str, Any], wire: Any) -> str | None:
        """The key of a recorded call, still to answer or the fork's own step, whose stored
        request this one fits where its masks stand (`fits`); None where none does."""
        if not self.masked:
            return None
        text = masked_request(request, wire, (), self.volatile)
        return next(
            (key for key, recorded in self.masked
             if (key == self.live_key or self.entries.get(key)) and fits(recorded, text)),
            None,
        )

    def take(self, key: str) -> Served | None:
        queue = self.entries.get(key)
        if not queue:
            return None
        return queue.popleft()

    @property
    def remaining(self) -> int:
        return sum(len(queue) for queue in self.entries.values())


def replayed_steps(store: Store, run: Run, step: int) -> list[tuple[Step, int]]:
    """Every recorded call a fork of `run` going live at `step` may be answered from: the
    main thread's calls before it, and every call of any thread that began before it — a
    subagent started earlier keeps its own conversation, whatever the main thread does."""
    main = store.steps(run.id)
    if step >= len(main):
        raise ValueError(f"run {run.id[:8]} has no step {step}; it has steps 0-{len(main) - 1}")
    cutoff = arrived(main[step])
    found = [(recorded, 0) for recorded in main[:step]]
    if run.launch_id:
        for position, thread in enumerate(store.threads_of(run.id), start=1):
            steps = store.steps(thread.id)
            if steps and arrived(steps[0]) < cutoff:
                found += [(recorded, position) for recorded in steps]
    return found


def build_table(
    store: Store,
    run: Run,
    step: int,
    *,
    patterns: Sequence[str] = DEFAULT_REDACTIONS,
    volatile: Iterable[tuple[str, str]] = (),
) -> ReplayTable:
    volatile = tuple(volatile)
    entries: dict[str, deque[Served]] = {}
    expected: dict[int, str] = {}
    masked: list[tuple[str, str]] = []
    today = patterns_digest(tuple(patterns))
    main = store.steps(run.id)
    live = main[step]
    for recorded, thread in [*replayed_steps(store, run, step), (live, None)]:
        wire = get_wire(recorded.wire)
        request = store.harness_request(recorded)
        text = masked_request(request, wire, patterns, volatile)
        if thread is not None:
            entries.setdefault(key_of(text), deque()).append(Served(recorded, thread))
        if not thread:
            expected[recorded.idx] = text
        if (recorded.params or {}).get("redaction") == today:
            continue  # masked as these patterns mask: a call that is it has its key
        plain = masked_request(request, wire, (), volatile)
        if REDACTED in plain:
            masked.append((key_of(text), plain))
    live_text = expected[step]
    return ReplayTable(
        entries=entries,
        live_key=key_of(live_text),
        live_step=live,
        expected=expected,
        step=step,
        volatile=volatile,
        patterns=tuple(patterns),
        masked=masked,
    )


def replay_problems(store: Store, run: Run, step: int) -> list[str]:
    """Why a fork of `run` at `step` could not be served faithfully, one sentence each.

    A step with no raw bytes cannot be given again as it was; one holding a redaction
    placeholder would give the harness the placeholder; one that failed, or was dropped or
    hung up at a breakpoint, gave the harness no answer to repeat.
    """
    problems: list[str] = []
    for recorded, thread in replayed_steps(store, run, step):
        where = f"step {recorded.idx}" if thread == 0 else f"thread {thread} step {recorded.idx}"
        if recorded.error or (recorded.status or 200) >= 400:
            problems.append(f"{where} failed ({recorded.error or recorded.status}), so there is no answer to replay")
            continue
        if (recorded.params or {}).get("raw_redacted"):
            # Its bytes were written again around a secret, or not kept (`recorder.stream_without_secrets`).
            problems.append(f"{where}'s answer had a secret redacted from it, so it cannot be replayed as it was")
            continue
        if not recorded.raw_response_blob:
            problems.append(f"{where} has no raw response recorded to replay")
            continue
        raw = store.blobs.get(recorded.raw_response_blob)
        if REDACTED.encode() in raw:
            problems.append(f"{where}'s answer had a secret redacted from it, so it cannot be replayed as it was")
    return problems


def run_paths(store: Store, run: Run) -> tuple[Path | None, Path | None]:
    """The workspace and home a recorded run was given."""
    if run.launch_id:
        launch = store.get_launch(run.launch_id)
        if launch and launch.workspace and launch.home:
            return Path(launch.workspace), Path(launch.home)
    # A batch run recorded before launches: <workdir>/<batch>/<index>/{workspace,home},
    # with the batch's frozen skill at <workdir>/<batch>/skill-snapshot/<name>.
    snapshot = run.parity.get("skill_snapshot_dir")
    if snapshot and run.label and "." in run.label:
        index = run.label.rsplit(".", 1)[1]
        if index.isdigit():
            root = Path(snapshot).parents[1] / index
            return root / "workspace", root / "home"
    return None, None


def rewrite(value: Any, pairs: Sequence[tuple[str, str]]) -> Any:
    """Every string in a JSON value, with each (old, new) prefix swapped."""
    if isinstance(value, str):
        for old, new in pairs:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite(item, pairs) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, pairs) for key, item in value.items()}
    return value


def served_body(
    store: Store,
    served: Served,
    request: dict[str, Any],
    wire: Any,
    pairs: Sequence[tuple[str, str]],
) -> tuple[bytes, dict[str, Any]]:
    """What the harness gets for a replayed call, and the answer as a native response.

    Always rendered afresh, never passed through as bytes: a path the answer names can be
    split across stream chunks, and a find-and-replace on raw bytes misses it.
    """
    recorded = served.step
    streamed = bool(recorded.params.get("streamed"))
    raw = store.blobs.get(recorded.raw_response_blob) if recorded.raw_response_blob else b""
    native = wire.parse(raw, streamed) if raw else None
    if native is None:
        # Nothing parseable in the raw bytes (a stream the proxy could not read, say): the
        # common shape is all there is.
        native = wire.native_response(store.response(recorded) or {}, None)
    delivered = store.delivered_response(recorded) if recorded.delivered_response_blob else None
    if delivered is not None:
        # The harness was given an edit, not the model's answer: that is what it acted on.
        native = wire.native_response(delivered, native)
    native = rewrite(native, pairs)
    return wire.render(native, request=request), native


def first_difference(expected: str, actual: str) -> dict[str, Any]:
    """Where two masked requests part, quoted from each side."""
    at = next(
        (n for n, (a, b) in enumerate(zip(expected, actual)) if a != b),
        min(len(expected), len(actual)),
    )
    start = max(0, at - CONTEXT // 3)
    return {
        "at": at,
        "expected": expected[start : at + CONTEXT],
        "got": actual[start : at + CONTEXT],
    }


def unique_lines(original: str, variant: str, shortest: int = 20) -> list[str]:
    """Lines of the original wording that the variant does not have: if the model sees one
    in a fork testing the variant, the original reached it — a leak. A line the variant
    still holds — as a line, or inside a longer one, since an edit that extends a line keeps
    its words — cannot tell a leak from the variant itself, and is left out: twenty forks of
    one screen were marked unusable, the variant in place, for "- explicit commitments"
    inside "- explicit commitments (including …)"."""
    return [
        line.strip()
        for line in original.splitlines()
        if len(line.strip()) >= shortest and line.strip() not in variant
    ]


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings(item)]
    return []


def leaked(request: dict[str, Any], lines: Sequence[str]) -> list[str]:
    """The original's lines still present where the harness put text for the model, in a
    request meant to carry only the variant. The model's own earlier turns are left out, as
    `experiment.skill` leaves them out when it swaps the variant in: they are the history a
    fork inherits, written under the original wording — a model that quoted its skill while
    reasoning, before the step, quoted it as it then read."""
    texts = strings(request.get("system"))
    for message in request.get("messages") or []:
        if isinstance(message, dict) and message.get("role") != "assistant":
            texts += strings(message.get("content"))
    text = "\n".join(texts)
    return [line for line in lines if line in text]


@dataclass(slots=True)
class Variant:
    """A wording swapped in from the step a fork goes live at."""

    original_text: str
    variant_text: str
    original_path: str | None = None
    variant_path: str | None = None
    unique: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.unique:
            self.unique = unique_lines(self.original_text, self.variant_text)

    def to_json(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "variant_text": self.variant_text,
            "original_path": self.original_path,
            "variant_path": self.variant_path,
            "original_sha256": hashlib.sha256(self.original_text.encode()).hexdigest(),
            "variant_sha256": hashlib.sha256(self.variant_text.encode()).hexdigest(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Variant | None:
        if not data or "variant_text" not in data:
            return None
        return cls(
            original_text=data["original_text"],
            variant_text=data["variant_text"],
            original_path=data.get("original_path"),
            variant_path=data.get("variant_path"),
        )

    def apply(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """The request with the variant swapped in, and what was done, or the request
        unchanged and None when it holds no copy of the skill."""
        from tuningfork.experiment.skill import NothingToCompare, SkillNotFound, substitute

        try:
            sent, spans = substitute(request, self.original_text, self.variant_text)
        except (SkillNotFound, NothingToCompare):
            sent, spans = request, []
        info: dict[str, Any] | None = None
        if spans:
            info = {"spans": len(spans), "variant_sha256": self.to_json()["variant_sha256"]}
        found = leaked(sent, self.unique)
        if found:
            info = {**(info or {}), "leaked": found[:3]}
        return sent, info
