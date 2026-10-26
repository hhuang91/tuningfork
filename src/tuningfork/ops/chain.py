"""A chain: a task run N times in order, each run given the skill folder as the run before
left it — how a harness's own self-improvement accumulates, measured link by link.

A batch's runs are independent samples: each is given the skill fresh, and one that changed it
is not counted. A chain is the opposite, on purpose: whatever a run leaves of the skill —
patched by the model during the task, or by Hermes's background review after it — is what the
next run starts from. Every link is still a launch of its own, recorded and judged as a batch's
run is (one whose skill changed is still not counted toward a rate, and still says whether it
passed), and the chain records every version of the skill it handed on: its fingerprint, the
lines that changed, and whose calls changed them. The folder the last link leaves is the
chain's final skill: what the harness's loop made of it, as it would stand in use.

A chain is kept as `<home>/chains/<id>.json`, written after every link, so one interrupted
still says how far it got; its links' launches are a batch's, under one group folder.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tuningfork.ops.batch import BatchPlan, Note, execute_batch, prepare_batch, shown_options
from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress, emit
from tuningfork.ops.launch import ProxyStatus
from tuningfork.store.db import Store

#: Where chains are kept, under the recordings folder.
CHAINS = "chains"
#: Beside the links' folders, numbered as they are: the skill as each link's harness left it,
#: the next link's start. Never inside a link's own folder, which goes with its run when that
#: is deleted for good (found in review: so did the chain's final skill, with the last link's).
LEFT = "skill-left"
#: What a told chain says to the harness after each run, as one more user turn of the same
#: session: the check's verdict and its output, nothing more — what a person who ran the
#: check would say, in the same words for every skill. The harness's own loop decides what to
#: make of it; its review, due after the turn, sees the task and this.
TOLD_TEMPLATE = "I ran my check on what you produced. {verdict}\n\n{output}"


def root(home: Path | str) -> Path:
    return Path(home) / CHAINS


@dataclass(slots=True)
class ChainPlan:
    """A chain checked before any link runs: a batch plan for one run, taken N times in order."""

    batch: BatchPlan
    n: int
    id: str
    notes: list[Note] = field(default_factory=list)
    #: Whether each link's harness is told the check's result (`TOLD_TEMPLATE`).
    tell: bool = False
    #: Stop before a link once the links' model calls have taken this long, in seconds of wall
    #: time at the proxy (`ops.spend`): how two loops are given equal budgets.
    budget_seconds: float | None = None

    @property
    def task_id(self) -> str:
        return self.batch.task_id

    @property
    def skill_dir(self) -> Path:
        assert self.batch.skill_dir is not None
        return self.batch.skill_dir

    def to_json(self) -> dict[str, Any]:
        data = self.batch.to_json()
        data.update(
            kind="chain", chain=self.id, n=self.n, tell=self.tell,
            told_template=TOLD_TEMPLATE if self.tell else None,
            budget_seconds=self.budget_seconds,
            notes=[note.to_json() for note in [*self.batch.notes, *self.notes]],
        )
        return data


def prepare_chain(
    store: Store,
    spec: Path | Any,
    *,
    eval_id: str | None = None,
    n: int = 10,
    model: str | None = None,
    harness: str = "hermes",
    workdir: Path | None = None,
    base_url: str | None = None,
    proxy: ProxyStatus | None = None,
    harness_options: dict[str, Any] | None = None,
    skill_dir: Path | None = None,
    tell: bool = False,
    budget_seconds: float | None = None,
) -> ChainPlan:
    """Everything a chain needs, checked as a batch's is. The links run one at a time, in
    order: each needs the one before it. `skill_dir` starts the chain from a folder other
    than the spec's — a chain continued from where another left off. `tell` gives each link's
    harness the check's result as one more user turn of the same session, before its own
    review: the signal a person who looked at the result would give. `budget_seconds` stops
    the chain before a link once its model calls have taken that long, at the proxy."""
    if n < 1:
        raise OperationError("a chain needs at least one link", field="n")
    if budget_seconds is not None and budget_seconds <= 0:
        raise OperationError("a budget is a number of seconds above 0", field="budget_seconds")
    batch = prepare_batch(
        store, spec, eval_id=eval_id, n=1, model=model, harness=harness, workdir=workdir,
        concurrency=1, base_url=base_url, proxy=proxy, harness_options=harness_options,
        skill_dir=skill_dir,
    )
    if batch.skill_dir is None:
        raise OperationError(
            "a chain carries a skill forward, and the spec names no skill folder (tuningfork.skill_dir)",
            field="spec",
        )
    if tell and getattr(batch.adapter, "continue_launch", None) is None:
        raise OperationError(
            f"{harness} cannot be given one more turn of a session it ran, so it cannot be told "
            "the check's result",
            field="tell",
        )
    if tell and not batch.item.tuningfork.check:
        raise OperationError(
            f"eval {batch.item.id} has no check, so there is no result to tell the harness",
            field="tell",
        )
    notes = [Note(
        "note",
        "a chain's runs are not independent samples: each starts from the skill the one before "
        "left, so their pass rate describes the loop, not one wording",
    )]
    if tell:
        notes.append(Note(
            "note",
            "after each run, the harness is told the check's verdict and output as the user's "
            "next turn of the same session, before its review; the run's verdict stays the task's",
        ))
    if budget_seconds is not None:
        notes.append(Note(
            "note",
            f"the chain stops before a link once its model calls have taken {budget_seconds:g} s, and the "
            "link that could outrun it is cut to what is left "
            "at the proxy, the first link always run",
        ))
    notes += [Note("note", text) for text in getattr(batch.adapter, "chain_notes", lambda: [])()]
    return ChainPlan(
        batch=batch, n=n, id=uuid.uuid4().hex[:8], notes=notes, tell=tell, budget_seconds=budget_seconds,
    )


@dataclass(slots=True)
class Link:
    """One run of the chain, and what it did to the skill."""

    index: int
    launch_id: str | None
    run_id: str | None
    passed: bool | None
    usable: bool
    exclusion: str | None
    duration_s: float
    #: Fingerprints (`specs.fingerprint`) of the skill the link was given, and of what it left.
    skill_before: str | None
    skill_after: str | None
    #: The skill's files that changed, relative to the harness's skills folder.
    changed: list[str]
    #: Every call of a skill-editing tool in the launch: the thread that made it (task, review,
    #: subagent), at which step, with what action on which skill.
    edits: list[dict[str, Any]]
    #: The harness's own review after the answer, when it ran: its run, how many steps, whether
    #: it ended, and how many skill-editing calls it made.
    review: dict[str, Any] | None
    #: The lines of SKILL.md that changed, when they did (`ops.diff.line_diff`), of each text
    #: masked whole.
    diff: dict[str, Any] | None
    #: Where the skill was left, copied: the next link's start.
    left: str | None
    #: What the harness was told after the check, when the chain tells: the message, and how
    #: that turn's process ended.
    told: dict[str, Any] | None = None
    #: Seconds the link's model calls took at the proxy, every thread of its launch.
    spent_seconds: float = 0.0
    #: The most seconds the link's processes were allowed, when the budget cut them: the last
    #: link of a chain that would otherwise have outrun it.
    cap_seconds: float | None = None

    @property
    def callers(self) -> list[str]:
        """The threads that called a skill-editing tool, in the order seen."""
        return list(dict.fromkeys(edit["thread"] for edit in self.edits))

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index, "launch_id": self.launch_id, "run_id": self.run_id,
            "passed": self.passed, "usable": self.usable, "exclusion": self.exclusion,
            "duration_s": round(self.duration_s, 2), "skill_before": self.skill_before,
            "skill_after": self.skill_after, "changed": list(self.changed), "edits": list(self.edits),
            "callers": self.callers, "review": self.review, "diff": self.diff, "left": self.left,
            "told": self.told, "spent_seconds": round(self.spent_seconds, 3), "cap_seconds": self.cap_seconds,
        }


@dataclass(slots=True)
class ChainReport:
    plan: ChainPlan
    links: list[Link]
    #: The folder the last finished link left: the chain's final skill. The start, with no link.
    final_skill: Path | None
    elapsed: float
    status: str  # running | done | cancelled | failed
    started_at: float
    #: Why it ran fewer links than planned, when it did: "budget".
    stopped: str | None = None

    @property
    def spent_seconds(self) -> float:
        return sum(link.spent_seconds for link in self.links)

    @property
    def counted(self) -> list[Link]:
        return [link for link in self.links if link.usable and link.passed is not None]

    @property
    def passes(self) -> int:
        return sum(1 for link in self.counted if link.passed)

    @property
    def versions(self) -> list[dict[str, Any]]:
        """Every version of the skill the chain handed on, in order: the one it was given,
        then each that a link left different from what it was given."""
        found: list[dict[str, Any]] = []
        if self.links:
            found.append({"skill": self.links[0].skill_before, "from": "start"})
        for link in self.links:
            if link.skill_after and link.skill_after != link.skill_before:
                found.append({
                    "skill": link.skill_after, "from": "link", "link": link.index,
                    "changed": list(link.changed), "by": link.callers,
                })
        return found

    def to_json(self) -> dict[str, Any]:
        from tuningfork.analysis.stats import rate

        batch = self.plan.batch
        counted = self.counted
        return {
            "id": self.plan.id,
            "status": self.status,
            "task_id": self.plan.task_id,
            "spec": str(batch.spec.path) if batch.spec.path else None,
            "eval": batch.item.id,
            "harness": batch.harness,
            "model": batch.model,
            "n": self.plan.n,
            "harness_options": shown_options(batch.harness_options),
            "tell": self.plan.tell,
            "told_template": TOLD_TEMPLATE if self.plan.tell else None,
            "budget_seconds": self.plan.budget_seconds,
            "spent_seconds": round(self.spent_seconds, 3),
            "stopped": self.stopped,
            "started_at": self.started_at,
            "elapsed": round(self.elapsed, 2),
            "workdir": str(batch.workdir / self.plan.id),
            "skill_start": str(self.plan.skill_dir),
            "final_skill": str(self.final_skill) if self.final_skill else None,
            "links": [link.to_json() for link in self.links],
            "versions": self.versions,
            "rate": rate(self.passes, len(counted)).to_json() if counted else None,
            "not_counted": len(self.links) - len(counted),
        }


def execute_chain(
    store: Store,
    plan: ChainPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ChainReport:
    """Run the links in order. Reports the batch's own `launch` and `run` events for each,
    then `link` with the link's record. Cancelled, it raises `Cancelled` whose `partial` is the
    report over the links that finished; the chain's file says so too."""
    from tuningfork.ops.spend import launch_spend

    adapter = plan.batch.adapter
    folder = plan.batch.workdir / plan.id
    current = Path(plan.skill_dir)
    links: list[Link] = []
    started = time.monotonic()
    started_at = time.time()
    stopped: str | None = None

    def report(status: str) -> ChainReport:
        found = ChainReport(
            plan=plan, links=list(links), final_skill=current,
            elapsed=time.monotonic() - started, status=status, started_at=started_at, stopped=stopped,
        )
        _write(store, found)
        return found

    report("running")
    try:
        for index in range(plan.n):
            if cancel is not None:
                cancel.check()
            spent = sum(link.spent_seconds for link in links)
            if plan.budget_seconds is not None and links and spent >= plan.budget_seconds:
                stopped = "budget"
                emit(progress, "budget", spent_seconds=round(spent, 1), budget_seconds=plan.budget_seconds, links=len(links))
                break
            # The first link runs whole, whatever the budget. After it, a link that could outrun
            # the budget — its timeout longer than what is left — is cut to what is left: its
            # processes' wall time bounds their model calls' time from above. Killed at the cut,
            # it is the last; finished within it, the chain goes on, the next link cut to less.
            cap: float | None = None
            if plan.budget_seconds is not None and links:
                left_seconds = plan.budget_seconds - spent
                if left_seconds < plan.batch.item.tuningfork.timeout:
                    cap = max(1.0, left_seconds)
            batch = execute_batch(
                store, plan.batch, progress=progress, cancel=cancel, group=plan.id, start=index,
                skill_dir=current, skill_snapshot=f"skill-snapshot-{index}",
                settings={"chain": {"id": plan.id, "link": index, "tell": plan.tell}},
                tell=TOLD_TEMPLATE if plan.tell else None, launch_seconds=cap,
            )
            link, left = _link(store, adapter, batch.outcomes[0], index, current, folder / f"{LEFT}-{index}")
            link.cap_seconds = cap
            if link.launch_id:
                link.spent_seconds = launch_spend(store, link.launch_id).wall_ms / 1000
            links.append(link)
            if left is not None:
                current = left
            emit(progress, "link", subject=link, **link.to_json())
            report("running")
            if cap is not None and batch.outcomes[0].timed_out:
                stopped = "budget"
                spent = sum(link.spent_seconds for link in links)
                emit(progress, "budget", spent_seconds=round(spent, 1), budget_seconds=plan.budget_seconds,
                     links=len(links), cap_seconds=cap)
                break
    except Cancelled as halted:
        raise Cancelled(halted.reason, partial=report("cancelled")) from None
    except OperationError:
        report("failed")
        raise
    return report("done")


def _link(
    store: Store, adapter: Any, outcome: Any, index: int, given: Path, folder: Path
) -> tuple[Link, Path | None]:
    """What a finished link did to the skill: the fingerprints before and after, the files
    changed, who called a skill-editing tool, and a copy of the skill as it was left."""
    from tuningfork.ops.diff import line_diff
    from tuningfork.runner import skill_places
    from tuningfork.specs import fingerprint, hash_tree

    launch = store.get_launch(outcome.launch_id) if outcome.launch_id else None
    run = store.get_run(outcome.run_id) if outcome.run_id else None
    before = run.skill_version if run is not None else None
    after = before
    left: Path | None = None
    diff: dict[str, Any] | None = None
    changed = list(outcome.skills_changed)
    if launch is not None and launch.home:
        home = Path(launch.home)
        after = fingerprint(hash_tree(adapter.skill_roots(home)))
        place = skill_places(adapter, home, given)[0]
        if (place / "SKILL.md").is_file():
            left = folder / given.name
            shutil.copytree(
                place, left, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*", "__pycache__"),
            )
            old, new = _text(given / "SKILL.md"), _text(left / "SKILL.md")
            if old != new:
                # The lines that changed, of each text masked whole, as the store masks the call that
                # wrote it: found in review, masked line by line, a private key's lines and a block's
                # stood, and so did a value the skill names a password, said bare in a line added. Both
                # masked alike, of what the masks took in either, or a line holding such a value bare
                # would differ though the link never changed it; each mask keeps its line breaks, so
                # the lines keep their numbers in the file.
                found = store.masking.text(old).found | store.masking.text(new).found
                was, now = (store.masking.text(text, also=found, lines=True).value for text in (old, new))
                diff = line_diff(was, now)
        else:
            changed = sorted({*changed, "SKILL.md (gone: the chain goes on from the skill before)"})
    told = getattr(outcome, "told", None) or {}
    edits, review = (
        skill_edits(store, adapter, launch.id, told=told.get("message")) if launch is not None else ([], None)
    )
    link = Link(
        index=index, launch_id=launch.id if launch else None, run_id=run.id if run else None,
        passed=outcome.passed if outcome.checked else None, usable=outcome.usable,
        exclusion=outcome.exclusion, duration_s=outcome.duration_s,
        skill_before=before, skill_after=after, changed=changed, edits=edits, review=review,
        diff=diff, left=str(left) if left else None, told=getattr(outcome, "told", None),
    )
    return link, left


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def skill_edits(
    store: Store, adapter: Any, launch_id: str, *, told: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Every call of a skill-editing tool (the adapter's `skill_tools`) in a launch, each
    with who made it — judged step by step, since one thread can hold more than one: the
    task's own conversation; the harness's review after an answer (`side_call`); the turn the
    harness was told the check's result (`told`, the message); else a subagent. And a summary
    of the review's calls, when the harness made any."""
    tools = frozenset(getattr(adapter, "skill_tools", ()) or ())
    side_call = getattr(adapter, "side_call", None)
    edits: list[dict[str, Any]] = []
    review: dict[str, Any] | None = None
    # The message is the check's output masked where it ran; the request that carried it is kept
    # masked again, which can mask more (masking is not idempotent).
    told_as = {told, _as_requested(store, told)} if told else set()
    for run in store.launch_runs(launch_id):
        for step in store.steps(run.id):
            request = store.harness_request(step)
            if side_call is not None and side_call(request):
                thread = "review"
                review = review or {"steps": 0, "ended": True, "edit_calls": 0}
                review["steps"] += 1
                review["ended"] = not step.error and (step.status or 200) < 400
            elif not run.is_thread:
                thread = "task"
            elif _last_user(request) in told_as:
                thread = "told"
            else:
                thread = run.role or "thread"
            message = ((store.response(step) or {}).get("choices") or [{}])[0].get("message") or {}
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") not in tools:
                    continue
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                edits.append({
                    "thread": thread, "run": run.id, "step": step.idx, "tool": function.get("name"),
                    "action": arguments.get("action"), "name": arguments.get("name"),
                    "file": arguments.get("file_path"),
                })
                if thread == "review" and review is not None:
                    review["edit_calls"] += 1
    return edits, review


def _as_requested(store: Store, message: str) -> str:
    """`message` as the store keeps a request that carries it as the user's turn."""
    kept = store.masking.value({"messages": [{"role": "user", "content": message}]})
    try:
        return str(kept["messages"][0]["content"])
    except (KeyError, IndexError, TypeError):
        return message


def _last_user(request: dict[str, Any]) -> str:
    from tuningfork.analysis.normalize import content_text

    users = [m for m in request.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return content_text(users[-1].get("content")) if users else ""


# --- the chains kept ---------------------------------------------------------------------------


#: What a chain's record keeps of each link as it was masked where it was made: what its harness
#: was told, masked where its check ran (`runner.MASKED_WHERE_MADE`); its skill-editing calls, read
#: from the store as it keeps them; and the lines it changed in the skill, of each text masked
#: whole (`_link`). Masked again, a command's second password would fall too.
LINK_MASKED = ("told", "edits", "diff")


def _write(store: Store, report: ChainReport) -> None:
    """The chain's record, masked as the store masks what it keeps (`Store.keep_file`), once. The
    copy of the skill a link left, the next link's start, is kept as it was left."""
    data = report.to_json()
    links = [store.masking.of(link, masked=LINK_MASKED).value for link in data["links"]]
    store.keep_file(root(store.home) / f"{report.plan.id}.json",
                    store.masking.of({**data, "links": links}, masked=("links",)), indent=2)


def list_chains(store: Store, limit: int = 20) -> list[dict[str, Any]]:
    """The chains kept, newest first, each as its file holds it."""
    folder = root(store.home)
    found: list[dict[str, Any]] = []
    if folder.is_dir():
        for path in folder.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("id"):
                found.append(data)
    found.sort(key=lambda data: data.get("started_at") or 0, reverse=True)
    return found[:limit] if limit else found


def get_chain(store: Store, reference: str) -> dict[str, Any]:
    """A chain by its id, or a prefix of it."""
    reference = (reference or "").strip()
    if not reference:
        raise OperationError("name the chain, by its id", code="bad_params", field="chain")
    matches = [data for data in list_chains(store, limit=0) if str(data["id"]).startswith(reference)]
    if not matches:
        raise OperationError(
            f"no chain matching {reference!r}", code="not_found", field="chain",
            hint="`tuningfork chain list` names them",
        )
    if len(matches) > 1:
        raise OperationError(
            f"{len(matches)} chains match {reference!r}: give more of the id", code="ambiguous", field="chain",
        )
    return matches[0]
