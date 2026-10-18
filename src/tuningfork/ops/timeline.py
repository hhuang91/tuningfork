"""A run as one conversation, and one step as it was sent.

Within a run each step's context is the one before it, extended: a harness sends its whole
history with every call, and run grouping files anything else — a rewritten history, a
subagent — as a thread of its own. So a run's last context holds the conversation once, and
each step is a place in it: where the model was asked, and answered. The timeline is that
context, read once, with each step's own output set where it happened — taken from the
recording, because a harness can send an answer back without the reasoning that produced it
(Hermes does, verified on recorded runs; Claude Code sends its thinking back).

`step_detail` is one step exactly as it was sent and answered: what a resample resends, and
what a step's exact-context page shows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from tuningfork.analysis.conditions import exclusion_from
from tuningfork.ops.common import OperationError
from tuningfork.ops.experiments import resolve_run
from tuningfork.store.db import Launch, Run, Step, Store, arrived


def run_json(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "label": run.label,
        "task_id": run.task_id,
        "model": run.model,
        "harness": run.harness,
        "step_count": run.step_count,
        "started_at": run.started_at,
        "updated_at": run.updated_at,
        "forked_from_run": run.forked_from_run,
        "forked_at_depth": run.forked_at_depth,
        "skill_hashes": run.skill_hashes,
        "result": run.result or None,
        # Why every rate leaves its verdict out, where one does.
        "exclusion": exclusion_from(run.result) if (run.result or {}).get("usable") is False else None,
        "launch_id": run.launch_id,
        "role": run.role,
        "parent_run_id": run.parent_run_id,
        "spawned_at": run.spawned_at,
        "server_id": run.server_id,
        # For a fork: the run it replayed, and the step it went live at.
        "fork_of": run.parity.get("fork_of") if run.is_fork else None,
        "fork_step": run.parity.get("fork_step") if run.is_fork else None,
        # In the trash: when, and what it went with (itself, its run, or an experiment).
        "trashed_at": run.trashed_at,
        "trashed_with": run.trashed_with,
    }


def launch_json(launch: Launch | None) -> dict[str, Any] | None:
    """How a run was launched, when tuningfork launched it: what running the task again, or
    forking it, starts from — and where its harness worked, and how it passes."""
    if launch is None:
        return None
    from tuningfork.ops.inspect import launch_status
    from tuningfork.specs import check_kind, describe_check

    check = ((launch.eval or {}).get("tuningfork") or {}).get("check")
    settings = launch.settings or {}
    return {
        "id": launch.id,
        "kind": launch.kind,
        "harness": launch.harness,
        "model": launch.model,
        "task_id": launch.task_id,
        "eval_id": (launch.eval or {}).get("id"),
        "spec_path": launch.spec_path,
        "status": launch_status(launch),
        "job_id": launch.job_id,
        # A fork an experiment drew is one of its samples, and goes to the trash with it.
        "experiment_id": launch.experiment_id,
        "workspace": launch.workspace,
        "check": describe_check(check),
        "check_kind": check_kind(check),
        # What the harness saw of the machine: the sandbox's paths, the folders hidden from it and
        # the instant its clock started at — and how it was set up (Hermes's options).
        "sandbox": settings.get("sandbox"),
        "harness_settings": settings.get("harness"),
    }


def changed_positions(
    view: Sequence[dict[str, Any]], harness_view: Sequence[dict[str, Any]] | None
) -> list[int]:
    """Where the model's view of a request differs from the harness's: the edited messages."""
    if harness_view is None:
        return []
    length = max(len(view), len(harness_view))
    return [
        index
        for index in range(length)
        if index >= len(view) or index >= len(harness_view) or view[index] != harness_view[index]
    ]


def _adapters(adapters: Sequence[Any] | None) -> list[Any]:
    if adapters is not None:
        return list(adapters)
    from tuningfork.harness import ADAPTERS

    return [adapter() for adapter in ADAPTERS.values()]


def mapped_skills(
    view: Sequence[dict[str, Any]], run: Run, adapters: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    """Skills in a context, found on its OpenAI-style view, each mapped back to the file the
    run recorded when the text matches."""
    from tuningfork.experiment.skill import text_sha256

    bodies = run.parity.get("skill_bodies") or {}
    spans = []
    for adapter in _adapters(adapters):
        for span in adapter.skill_spans({"messages": list(view)}):
            data = span.to_dict()
            digest = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
            data["file"], data["first_line"], data["match"] = None, 1, None
            exact = next((p for p, h in run.skill_hashes.items() if h == digest), None)
            if exact:
                data["file"], data["match"] = exact, "file"
            else:
                # A harness that shows only the body (frontmatter stripped) still maps, with
                # the body's own line numbers in the file; so does one that read a CRLF file
                # in text mode.
                normalized = text_sha256(span.text.strip())
                for path, entry in bodies.items():
                    if entry.get("full_sha256") == normalized:
                        data["file"], data["match"] = path, "file"
                        break
                    if entry.get("sha256") == normalized:
                        data["file"], data["match"] = path, "body"
                        data["first_line"] = entry.get("first_line", 1)
                        break
            spans.append(data)
    return spans


# --- one step ------------------------------------------------------------------------------


@dataclass(slots=True)
class StepDetail:
    """One step as it was sent and answered, and as the harness saw it where that differed."""

    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return self.data


def changed_by(step: Step) -> str | None:
    """Why the harness's request differs from the model's at a step: a person's edit at a
    breakpoint, or a fork's variant wording. None when it does not."""
    if not step.harness_request_node_id:
        return None
    return "variant" if step.substitution and not step.edits.get("request") else "breakpoint"


def step_detail(
    store: Store,
    run_ref: str,
    idx: int,
    *,
    adapters: Sequence[Any] | None = None,
    context: bool = True,
) -> StepDetail:
    """A step exactly as it was sent and answered. `context=False` leaves the request and its
    views out — what a page showing the step's details beside its conversation needs."""
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.analysis.reasoning import ACTED, after_failed_call, reasons, status
    from tuningfork.wire import get_wire

    found = resolve_run(store, run_ref)
    recorded = store.step(found.id, idx)
    if recorded is None:
        raise OperationError(f"run {found.id[:8]} has no step {idx}", code="not_found")
    model_request = store.rebuild_request(recorded)
    response = store.response(recorded)
    wire = get_wire(recorded.wire)
    view = wire.view(model_request)
    outcome = recorded_outcome(response, recorded.error)
    run_reasons = reasons(step.reasoning_chars for step in store.steps(found.id) if not step.served)
    harness_request = store.harness_request(recorded) if recorded.harness_request_node_id else None
    harness_view = wire.view(harness_request) if harness_request else None
    data: dict[str, Any] = {
        "run": run_json(found),
        "idx": recorded.idx,
        "model": recorded.model,
        "params": recorded.params,
        "server_defaults": recorded.server_defaults,
        "timings": recorded.timings,
        "usage": recorded.usage,
        "finish_reason": recorded.finish_reason,
        "error": recorded.error,
        "wire": wire.name,
        # `request` is exactly as sent, in its own format; `view` is the same conversation
        # as OpenAI-style messages, which is what the UI shows.
        "request": model_request,
        "view": view,
        "response": response,
        "outcome": outcome.to_json(),
        # "none": the model answered without reasoning, where it reasons at other steps of
        # the run; with whether a tool call it was about to read about had failed.
        "reasoning": status(recorded.reasoning_chars, reasons=run_reasons,
                            acted=outcome.kind in ACTED and not recorded.served),
        "after_failure": after_failed_call(view),
        "skills": mapped_skills(view, found, adapters),
        "edits": recorded.edits,
        # Positions in `view`, not in the request: an edit's own indices count the wire
        # format's messages, which an Anthropic view splits and reorders.
        "edited_messages": changed_positions(view, harness_view),
        "request_changed_by": changed_by(recorded),
        "harness_request": harness_request,
        "harness_view": harness_view,
        # A fork's step: answered from the recording, or swapped a variant in.
        "replay": recorded.replay or None,
        "substitution": recorded.substitution or None,
        "workspace_commit": recorded.workspace_commit,
        "server_id": recorded.server_id,
        "delivered_response": (
            store.delivered_response(recorded) if recorded.delivered_response_blob else None
        ),
    }
    if not context:
        for key in ("request", "view", "harness_request", "harness_view"):
            data.pop(key)
        data["context_messages"] = len(view)
    return StepDetail(data)


# --- the run ------------------------------------------------------------------------------


@dataclass(slots=True)
class RunTimeline:
    """A run as one conversation: its last context once, and a mark at each step.

    `entries` is the last step's context as the harness sent it, in view form. Each entry's
    `origin` says how it got there: `start`, the context the run began with; `harness`, added
    by the harness after step `after` (tool results, reminders, a skill's body); `copy`, the
    harness's copy of step `after`'s output — shown in its place from the recording instead.
    Each of `steps` sits before `entries[position]`: its context was the entries above it.
    """

    run: Run
    launch: dict[str, Any] | None
    wire: str | None
    tools: list[str]
    entries: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    #: Step indices with no recording: a call refused while the proxy was paused.
    missing: list[int]
    skills: list[dict[str, Any]]
    threads: list[dict[str, Any]]
    #: Whether the run is over: launched and ended, or given a verdict.
    ended: bool
    #: The last step's reply, as the harness received it; None when it said nothing.
    reply: str | None

    @property
    def made_by(self) -> list[dict[str, Any]] | None:
        """Where the check was run along the run (`ops.checks.check_along`): each line it
        printed, with the step whose call made it (`made_by_calls`); None where it was not, or
        could not place what it found."""
        from tuningfork.ops.checks import ALONG, made_by_calls

        along = (self.run.result or {}).get(ALONG)
        return made_by_calls(along) if along else None

    def to_json(self) -> dict[str, Any]:
        return {
            "run": run_json(self.run),
            "launch": self.launch,
            "wire": self.wire,
            "tools": self.tools,
            "entries": self.entries,
            "steps": self.steps,
            "missing": self.missing,
            "skills": self.skills,
            "threads": self.threads,
            "ended": self.ended,
            "reply": self.reply,
            "made_by": self.made_by,
        }


def _message(response: dict[str, Any] | None) -> dict[str, Any] | None:
    choices = (response or {}).get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("message") or None


def _said(message: dict[str, Any] | None) -> tuple[str, tuple[tuple[Any, str], ...]]:
    """What a message says, for telling whether the harness kept an answer as it was given:
    its text and its tool calls — not its reasoning, which a harness may drop, nor the ids
    tool calls are given anew."""
    from tuningfork.analysis.normalize import content_text

    if not message:
        return ("", ())
    calls = []
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            parsed = arguments
        calls.append((function.get("name"), json.dumps(parsed, sort_keys=True)))
    return (content_text(message.get("content")).strip(), tuple(calls))


def _tool_names(extras: dict[str, Any]) -> list[str]:
    names = []
    for tool in extras.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _extras_signature(step: Step) -> str:
    changed = step.edits.get("request") or step.substitution or {}
    if step.harness_request_node_id and changed.get("original_extras") is not None:
        return "harness:" + hashlib.sha256(
            json.dumps(changed["original_extras"], sort_keys=True).encode()
        ).hexdigest()
    return step.extras_blob


def _edited(step: Step) -> dict[str, Any]:
    response = step.edits.get("response") or {}
    return {
        "request": bool(step.edits.get("request")),
        "response": "dropped" if response.get("dropped") else "edited" if response else None,
    }


def run_timeline(store: Store, run_ref: str, *, adapters: Sequence[Any] | None = None) -> RunTimeline:
    """A run as one conversation, from one walk of its last context: each message read once."""
    from tuningfork.analysis.normalize import recorded_outcome, reply_text
    from tuningfork.analysis.reasoning import ACTED, after_failed_call, reasons, status
    from tuningfork.ops.inspect import arm_status, launch_status
    from tuningfork.wire import get_wire

    run = resolve_run(store, run_ref)
    steps = store.steps(run.id)
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    ended = bool(run.result) or (launch is not None and launch_status(launch) not in (None, "running"))
    threads = _threads(store, run, steps)
    if not steps:
        return RunTimeline(run, launch_json(launch), None, [], [], [], [], [], threads, ended, None)

    # The harness's history, not the model's: what the next call extends.
    leaves = [step.harness_request_node_id or step.request_node_id for step in steps]
    depths = store.node_depths(leaves)
    final = max(range(len(steps)), key=lambda i: (depths.get(leaves[i], -1), i))
    rows = store.path_rows(leaves[final])
    on_path = {row["id"]: row["depth"] for row in rows}
    messages = [store.blobs.get_json(row["message_blob"]) for row in rows]

    wire = get_wire(steps[final].wire)
    extras = store.harness_extras(steps[final])
    head = wire.view({**extras, "messages": []})
    # A view converts each message on its own (a contract `tests/test_timeline.py` pins), so
    # native message i starts at view position prefix[i].
    chunks = [wire.view({"messages": [message]}) for message in messages]
    prefix = [len(head)]
    for chunk in chunks:
        prefix.append(prefix[-1] + len(chunk))
    entries: list[dict[str, Any]] = [
        {"index": index, "native": None, "message": message, "origin": "start", "after": None}
        for index, message in enumerate(head)
    ]
    for native, chunk in enumerate(chunks):
        for message in chunk:
            entries.append({
                "index": len(entries), "native": native, "message": message,
                "origin": "start", "after": None,
            })

    positions: list[int] = []
    off_path: list[bool] = []
    for step, leaf in zip(steps, leaves):
        depth = on_path.get(leaf) if leaf else -1
        if depth is None:
            # Recorded before run grouping kept this promise: placed after the step before it.
            positions.append(positions[-1] if positions else len(head))
            off_path.append(True)
        else:
            positions.append(prefix[depth + 1])
            off_path.append(False)

    responses = [store.response(step) for step in steps]
    outputs = [_message(response) for response in responses]
    delivered = [
        _message(store.delivered_response(step)) if step.delivered_response_blob else None
        for step in steps
    ]

    # Each step's region runs from its position to the next step's. A step sent again from
    # the same place (a retry) left no answer behind; the last step there owns the region.
    kept: list[str | None] = [None] * len(steps)
    copies: list[int | None] = [None] * len(steps)
    retried_by: list[int | None] = [None] * len(steps)
    for i in range(len(steps) - 1):
        if positions[i + 1] == positions[i] and not off_path[i + 1]:
            retried_by[i] = steps[i + 1].idx
            kept[i] = "retried"
    for i, step in enumerate(steps):
        if retried_by[i] is not None or off_path[i]:
            continue
        start = positions[i]
        later = [p for p in positions[i + 1:] if p > start]
        end = min(later) if later else len(entries)
        region = range(start, end)
        copy = next((j for j in region if entries[j]["message"].get("role") == "assistant"), None)
        for j in region:
            entries[j]["origin"] = "copy" if j == copy else "harness"
            entries[j]["after"] = step.idx
        copies[i] = copy
        if i == len(steps) - 1 and copy is None:
            continue  # the last step: nothing after it has sent its answer back yet
        if copy is None:
            kept[i] = "missing"
        elif _said(entries[copy]["message"]) == _said(outputs[i]):
            kept[i] = "same"
        elif delivered[i] is not None and _said(entries[copy]["message"]) == _said(delivered[i]):
            kept[i] = "edited"
        else:
            kept[i] = "differs"

    changed = [_changed_for_model(store, step, leaf, rows, prefix, head) for step, leaf in zip(steps, leaves)]
    for step, found in zip(steps, changed):
        for index in (found or {}).get("entries", []):
            if index < len(entries):
                marks = entries[index].setdefault("changed_for_model", [])
                mark = next((m for m in marks if m["by"] == found["by"]), None)
                if mark is None:
                    marks.append({"by": found["by"], "steps": [step.idx]})
                else:
                    mark["steps"].append(step.idx)

    extras_changed = _extras_changes(store, steps)
    experiments = _experiments(store, run, arm_status)
    # Which answers have been scored — one query for the run; the tokens load when opened.
    streams = store.steps_with_token_streams([step.id for step in steps])
    run_reasons = reasons(step.reasoning_chars for step in steps if not step.served)
    said = [entry["message"] for entry in entries]
    anchors = []
    for i, step in enumerate(steps):
        outcome = recorded_outcome(responses[i], step.error)
        anchors.append({
            "idx": step.idx,
            "id": step.id,
            "position": positions[i],
            "off_path": off_path[i],
            "outcome": outcome.to_json(),
            "reasoning": status(step.reasoning_chars, reasons=run_reasons,
                                acted=outcome.kind in ACTED and not step.served),
            "after_failure": after_failed_call(said[:positions[i]]),
            "finish_reason": step.finish_reason,
            "error": step.error,
            "status": step.status,
            "output": outputs[i],
            "delivered": delivered[i],
            "kept": kept[i],
            "copy": copies[i],
            "retried_by": retried_by[i],
            "last": i == len(steps) - 1,
            "served": step.served,
            "replay": step.replay or None,
            "variant": bool(step.substitution),
            "leaked": bool(step.substitution.get("leaked")),
            "edited": _edited(step),
            "changed_for_model": changed[i],
            "extras_changed": extras_changed[i],
            "model": step.model,
            "usage": step.usage,
            "timings": step.timings,
            "reasoning_chars": step.reasoning_chars,
            "arrived_at": arrived(step),
            "workspace_commit": step.workspace_commit,
            "experiments": experiments.get(step.idx, []),
            "tokens": _tokens_json(streams.get(step.id), step),
        })

    recorded = {step.idx for step in steps}
    last = steps[-1]
    reply = reply_text(store.delivered_response(last)) or None
    return RunTimeline(
        run=run,
        launch=launch_json(launch),
        wire=wire.name,
        tools=_tool_names(extras),
        entries=entries,
        steps=anchors,
        missing=[idx for idx in range(last.idx + 1) if idx not in recorded],
        skills=mapped_skills([entry["message"] for entry in entries], run, adapters),
        threads=threads,
        ended=ended,
        reply=reply,
    )


def _changed_for_model(
    store: Store,
    step: Step,
    leaf: str,
    rows: list[dict[str, Any]],
    prefix: list[int],
    head: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """What the model saw that the harness's history does not hold, at a step whose request
    was changed for the model: by a person at a breakpoint, or by a fork's variant."""
    by = changed_by(step)
    if by is None:
        return None
    model = store.path_rows(step.request_node_id)
    depth = next((row["depth"] for row in rows if row["id"] == leaf), None)
    # The harness's history up to this step; off the run's path, its own (and no entries).
    harness = rows[: depth + 1] if depth is not None else store.path_rows(leaf)
    natives = [
        i for i in range(min(len(model), len(harness)))
        if model[i]["exact_hash"] != harness[i]["exact_hash"]
    ]
    entries = [] if depth is None else [
        index for native in natives if native + 1 < len(prefix)
        for index in range(prefix[native], prefix[native + 1])
    ]
    edits = step.edits.get("request") or {}
    fields = [field for field in edits.get("fields") or [] if field != "messages"]
    if by == "variant" and step.substitution.get("original_extras") is not None:
        own = store.blobs.get_json(step.extras_blob)
        original = step.substitution["original_extras"]
        fields = sorted(
            key for key in set(own) | set(original)
            if key != "messages" and own.get(key) != original.get(key)
        )
    if "system" in fields and head:
        entries = [0, *entries]
    return {
        "by": by,
        "entries": sorted(set(entries)),
        "added": max(0, len(model) - len(harness)),
        "fields": fields,
    }


def _extras_changes(store: Store, steps: Sequence[Step]) -> list[dict[str, Any] | None]:
    """Where the request's other keys (a system prompt, the tools offered) changed from the
    step before: loaded only where their stored form differs."""
    cache: dict[str, dict[str, Any]] = {}

    def extras(step: Step) -> dict[str, Any]:
        signature = _extras_signature(step)
        if signature not in cache:
            cache[signature] = store.harness_extras(step)
        return cache[signature]

    changes: list[dict[str, Any] | None] = [None]
    for before, step in zip(steps, steps[1:]):
        if _extras_signature(before) == _extras_signature(step):
            changes.append(None)
            continue
        old, new = extras(before), extras(step)
        fields = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        if not fields:
            changes.append(None)
            continue
        old_tools, new_tools = set(_tool_names(old)), set(_tool_names(new))
        changes.append({
            "fields": fields,
            "tools_added": sorted(new_tools - old_tools),
            "tools_removed": sorted(old_tools - new_tools),
        })
    return changes


def _tokens_json(record: Any, step: Any) -> dict[str, Any] | None:
    """Whether the step's answer has tokens, and how they were had: not the tokens themselves."""
    if record is None:
        return None
    return {
        "stream_id": record.id,
        "source": record.source,
        "method": record.method,
        "tokens": record.tokens,
        "exact": record.exact,
        "same_server": None if record.server_id is None or step.server_id is None else record.server_id == step.server_id,
    }


def _experiments(store: Store, run: Run, arm_status: Any) -> dict[int, list[dict[str, Any]]]:
    """The experiments at each step; an A/B's two arms as one."""
    found: dict[int, list[dict[str, Any]]] = {}
    groups: set[str] = set()
    for experiment in sorted(store.experiments(run_id=run.id, limit=10_000), key=lambda e: e.created_at):
        if experiment.step_idx is None:
            continue
        if experiment.group_id:
            if experiment.group_id in groups:
                continue
            groups.add(experiment.group_id)
        found.setdefault(experiment.step_idx, []).append({
            "id": experiment.id,
            "kind": experiment.kind,
            "reach": experiment.settings.get("reach", "step"),
            "n": experiment.settings.get("n"),
            "samples": experiment.sample_count,
            "status": arm_status(experiment),
            "target": (experiment.settings.get("target") or {}).get("description"),
        })
    return found


def _threads(store: Store, run: Run, steps: Sequence[Step]) -> list[dict[str, Any]]:
    """The other conversations of the run's launch, each placed after the step it followed:
    a subagent after the step that spawned it, anything else by when its first call came."""
    if run.is_thread or not run.launch_id:
        return []
    arrivals = [(arrived(step), step.idx) for step in steps]
    placed = []
    for thread in store.threads_of(run.id):
        after, how = None, None
        if thread.role == "subagent" and thread.parent_run_id == run.id and thread.spawned_at is not None:
            after, how = thread.spawned_at, "spawn"
        else:
            first = store.steps(thread.id)[:1]
            if first:
                began = arrived(first[0])
                before = [idx for when, idx in arrivals if when <= began]
                after, how = (before[-1] if before else None), "time"
        placed.append({
            "run_id": thread.id,
            "role": thread.role,
            "label": thread.label,
            "steps": thread.step_count,
            "parent_run_id": thread.parent_run_id,
            "spawned_at": thread.spawned_at,
            "after": after,
            "placed": how,
        })
    return placed
