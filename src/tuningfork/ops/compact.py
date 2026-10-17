"""Results made small enough for a reader with a short context.

An agent driving tuningfork over MCP — and, in Phase 6b, the reflection packet a proposer
reads — sees the same results the web UI does, but can spend a few thousand tokens on each,
not a step's whole context. Each view here takes an operation's JSON and keeps what such a
reader needs; the evidence sentences say a result in words, statistics first.

JSON in, JSON out: a stored job result compacts exactly as a live one does. Nothing is cut
silently — a clipped text says how much it left out, a capped list how many items it had —
and no rate ever loses its interval: every object with `successes` and `total` keeps `low`
and `high` (`tests/test_compact.py` walks every view to check).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Callable, Sequence

#: Characters kept of a long text: a message, a check's output, a behaviour's label.
TEXT = 400
#: Characters kept of a label naming a behaviour.
LABEL = 160
#: Items kept of a long list.
ITEMS = 20
#: Bytes of JSON a view aims to stay under; `fit` tightens a view that does not.
BUDGET = 24_000
#: Sampling parameters worth showing of a step; the rest of its request is context.
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "seed", "n_probs", "logprobs")


# --- the pieces -------------------------------------------------------------------------------


def clip(text: Any, limit: int = TEXT) -> Any:
    """`text` cut to `limit` characters, saying how many it left out."""
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars)"


def short(value: Any) -> Any:
    """A run, experiment or job id cut to 8 characters: every lookup takes a prefix back."""
    return value[:8] if isinstance(value, str) and len(value) > 8 else value


def when(timestamp: Any) -> Any:
    """A recorded time as a reader writes it, in UTC."""
    if not isinstance(timestamp, (int, float)):
        return timestamp
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%MZ")


def capped(items: Sequence[Any], limit: int = ITEMS) -> dict[str, Any] | list[Any]:
    """A list as it is when short; the first `limit` items and the count when long."""
    items = list(items)
    if len(items) <= limit:
        return items
    return {"shown": items[:limit], "total": len(items)}


def rate(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """A rate with its interval, rounded for reading; never without `low` and `high`."""
    if not value:
        return None
    successes, total = value["successes"], value["total"]
    point = value.get("point", successes / total if total else 0.0)
    return {
        "successes": successes,
        "total": total,
        "point": round(point, 3),
        "low": round(value["low"], 3),
        "high": round(value["high"], 3),
    }


def rate_text(value: dict[str, Any] | None) -> str:
    """`3/20 (15%, 95% CI 5–36%)` — a count and its interval, never a bare percentage."""
    if not value:
        return "nothing counted"
    successes, total = value["successes"], value["total"]
    point = value.get("point", successes / total if total else 0.0)
    return f"{successes}/{total} ({point:.0%}, 95% CI {value['low'] * 100:.0f}–{value['high'] * 100:.0f}%)"


def outcome(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a step did, labelled for reading; `key` is what samples are grouped on."""
    if not value:
        return None
    return {"kind": value.get("kind"), "label": clip(value.get("label"), LABEL), "key": value.get("key")}


def p_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "p n/a"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3g}"


# --- tasks and runs ---------------------------------------------------------------------------


def task(summary: dict[str, Any]) -> dict[str, Any]:
    """A task with its pass rate under each condition: harness, model, skill version, server."""
    check = summary.get("check")
    return {
        "task_id": summary.get("task_id"),
        "runs": summary.get("runs"),
        "latest": when(summary.get("latest")),
        "models": summary.get("models"),
        "passed": rate(summary.get("passed")),
        "uncounted": summary.get("uncounted"),
        "check": {"kind": check.get("kind"), "text": clip(check.get("text"), 200)} if check else None,
        "conditions": [
            {
                "id": condition.get("id"),
                "harness": condition.get("harness"),
                "model": condition.get("model"),
                "skill_version": short(condition.get("skill_version")),
                "server_id": short(condition.get("server_id")),
                "skill_file": condition.get("skill_file"),
                "runs": condition.get("runs"),
                "run_ids": capped([short(found) for found in condition.get("run_ids") or []], 10),
                "passed": rate(condition.get("passed")),
                "uncounted": condition.get("uncounted"),
            }
            for condition in summary.get("conditions") or []
        ],
    }


def verdict(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run's verdict from its task's check, without the check's whole output."""
    if not result:
        return None
    kept = {key: result[key] for key in ("passed", "usable", "reason", "exit_code") if key in result}
    for key in ("output", "detail", "why"):
        if result.get(key):
            kept[key] = clip(result[key], 300)
    return kept


def run(found: dict[str, Any]) -> dict[str, Any]:
    """A run as a list shows it."""
    data = {
        "id": short(found.get("id")),
        "label": found.get("label"),
        "task_id": found.get("task_id"),
        "harness": found.get("harness"),
        "model": found.get("model"),
        "steps": found.get("step_count"),
        "started": when(found.get("started_at")),
        "result": verdict(found.get("result")),
    }
    for key in ("role", "fork_of", "fork_step", "parent_run_id", "forked_from_run", "trashed_at"):
        if found.get(key) is not None:
            data[key] = short(found[key]) if key != "fork_step" else found[key]
    return data


def usage(found: dict[str, Any] | None) -> dict[str, Any] | None:
    if not found:
        return None
    return {key: found[key] for key in ("prompt_tokens", "completion_tokens") if key in found} or None


def run_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """A run step by step: what the model did at each, in a line."""
    launch = detail.get("launch") or None
    return {
        "run": run(detail["run"]),
        "steps": capped(
            [
                {
                    "idx": step.get("idx"),
                    "did": outcome(step.get("outcome")),
                    # What its calls ran, where `did` shows a digest of it; the files they changed.
                    "ran": clip(step.get("ran"), 240),
                    "changed": [change.get("path") for change in step.get("changed") or []][:8] or None,
                    "finish_reason": step.get("finish_reason"),
                    "tokens": usage(step.get("usage")),
                    "reasoning_chars": step.get("reasoning_chars") or None,
                    # Answered without reasoning, though the model reasons at other steps.
                    "no_reasoning": True if step.get("reasoning") == "none" else None,
                    "error": clip(step.get("error"), 200),
                    "replayed": bool(step.get("replay")) or None,
                    "variant": bool(step.get("substitution")) or None,
                }
                for step in detail.get("steps") or []
            ],
            60,
        ),
        "threads": [run(thread) for thread in detail.get("threads") or []],
        "launch": (
            {
                key: (clip(launch[key], 300) if isinstance(launch[key], str) else launch[key])
                for key in ("harness", "spec", "task_id", "check", "check_kind", "workspace")
                if launch.get(key) is not None
            }
            if launch
            else None
        ),
    }


# --- one step --------------------------------------------------------------------------------


def _content(content: Any) -> str:
    """A message's content as text: parts joined, images and the like named."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") if isinstance(part.get("text"), str) else f"[{part.get('type', 'part')}]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return "" if content is None else json.dumps(content)


def _calls(message: dict[str, Any], limit: int) -> list[dict[str, Any]] | None:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append({"name": function.get("name"), "arguments": clip(arguments, limit)})
    return calls or None


def message(found: dict[str, Any], limit: int = TEXT) -> dict[str, Any]:
    """One message of a context: who said it and what, clipped."""
    data: dict[str, Any] = {"role": found.get("role"), "content": clip(_content(found.get("content")), limit)}
    calls = _calls(found, limit)
    if calls:
        data["tool_calls"] = calls
    if found.get("tool_call_id"):
        data["tool_call_id"] = found["tool_call_id"]
    reasoning = found.get("reasoning_content") or found.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        data["reasoning_chars"] = len(reasoning)
    return data


def step(detail: dict[str, Any], *, full: bool = False, last: int = 6) -> dict[str, Any]:
    """A step as an agent reads it: where the skill sits in its context, the last few
    messages before the model answered, and the answer. `full` keeps every message, with
    longer clips."""
    limit = 4000 if full else TEXT
    view = detail.get("view") or []
    system = next((entry for entry in view if entry.get("role") == "system"), None)
    shown = view if full else view[-last:]
    response = detail.get("response") or {}
    answer = ((response.get("choices") or [{}])[0].get("message")) or {}
    reasoning = answer.get("reasoning_content") or answer.get("reasoning") or ""
    params = detail.get("params") or {}
    return {
        "run": short((detail.get("run") or {}).get("id")),
        "idx": detail.get("idx"),
        "model": detail.get("model"),
        "wire": detail.get("wire"),
        "sampling": {key: params[key] for key in SAMPLING if key in params} or None,
        "tokens": usage(detail.get("usage")),
        "finish_reason": detail.get("finish_reason"),
        "error": clip(detail.get("error"), 300),
        "did": outcome(detail.get("outcome")),
        "context": {
            "messages": len(view),
            "system": clip(_content(system.get("content")), 1500 if not full else 8000) if system else None,
            # Where the skill's text sits, and which file and lines it came from.
            "skills": [
                {
                    "name": span.get("name"),
                    "source": span.get("source"),
                    "message_index": span.get("message_index"),
                    "lines": span.get("lines"),
                    "file": span.get("file"),
                    "first_line": span.get("first_line"),
                    "match": span.get("match"),
                }
                for span in detail.get("skills") or []
            ],
            "shown_from": len(view) - len(shown),
            "last": [message(entry, limit) for entry in shown],
        },
        "answer": {
            "content": clip(_content(answer.get("content")), 1500 if not full else 8000),
            "tool_calls": _calls(answer, 1500 if not full else 8000),
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
            # Answered without reasoning, though the model reasons at other steps of the run;
            # and whether a tool result it was about to read reported failure.
            "no_reasoning": True if detail.get("reasoning") == "none" else None,
            "after_failed_tool_call": True if detail.get("after_failure") else None,
        },
        "edited": bool(detail.get("edits")) or None,
        "replayed": bool(detail.get("replay")) or None,
        "variant": bool(detail.get("substitution")) or None,
    }


def skill(recorded: dict[str, Any], variants: Sequence[dict[str, Any]] = (), *, lines: int = 400) -> dict[str, Any]:
    """The skill file a run was given, its lines numbered as an edit names them, and the
    variants saved of it."""
    text = recorded.get("text")
    numbered = None
    if isinstance(text, str):
        found = text.splitlines()
        numbered = capped([f"{number}: {line}" for number, line in enumerate(found, 1)], lines)
    return {
        "run": short(recorded.get("run_id")),
        "skill": recorded.get("skill"),
        "path": recorded.get("path"),
        "sha256": recorded.get("sha256"),
        "lines": numbered,
        "missing": None if isinstance(text, str) else "the batch's frozen copy of this skill is gone",
        "variants": [
            {key: variant.get(key) for key in ("name", "path", "skill", "modified_at") if variant.get(key) is not None}
            for variant in variants
        ],
    }


# --- what runs and samples did ------------------------------------------------------------------


def distribution(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """Behaviours, lettered by frequency, each with its rate."""
    if not found:
        return None

    def unreasoned(cluster: dict[str, Any]) -> bool:
        """Some of the behaviour's samples came without reasoning, where others reasoned."""
        reasoned = cluster.get("reasoned")
        return reasoned is not None and reasoned < ((cluster.get("rate") or {}).get("successes") or 0)

    return {
        "total": found.get("total"),
        "errors": found.get("errors"),
        "behaviours": [
            {"letter": cluster.get("letter"), **(outcome(cluster.get("outcome")) or {}), "rate": rate(cluster.get("rate")),
             **({"reasoned": cluster["reasoned"]} if unreasoned(cluster) else {})}
            for cluster in found.get("clusters") or []
        ],
    }


def divergence(report: dict[str, Any], *, all_steps: bool = False) -> dict[str, Any]:
    """Where runs stopped agreeing: the behaviours at that step, how many runs took each,
    and how often those runs went on to pass. Other steps only when asked for."""
    index = report.get("divergence_index")
    steps = report.get("steps") or []
    if not all_steps:
        steps = [found for found in steps if found.get("index") == index]
    mixed = report.get("mixed_skills")
    return {
        "runs": len(report.get("run_ids") or []),
        "run_ids": capped([short(found) for found in report.get("run_ids") or []], 30),
        "divergence_index": index,
        "agreed_steps": report.get("agreed_steps"),
        "mixed_skills": mixed or None,
        "steps": [
            {
                "index": found.get("index"),
                "diverged": found.get("diverged"),
                "groups": [
                    {
                        **(outcome(group.get("outcome")) or {}),
                        "runs": len(group.get("run_ids") or []),
                        "run_ids": capped([short(member) for member in group.get("run_ids") or []], 10),
                        "share": rate(group.get("rate")),
                        "passed": rate(group.get("passed")),
                    }
                    for group in found.get("groups") or []
                ],
            }
            for found in steps
        ],
    }


def behaviours(seen: dict[str, Any], targets: dict[str, Any] | None = None) -> dict[str, Any]:
    """The behaviours seen at a step — in recorded runs, forks and samples — each marked as a
    target would count it, and the letters of the step's latest resample, by which an A/B
    can declare its target."""
    latest = (targets or {}).get("resample")
    return {
        "run": short(seen.get("run_id")),
        "step": seen.get("step"),
        "tools": seen.get("tools"),
        "normalizers": seen.get("normalizers"),
        "target": seen.get("target"),
        "judged": seen.get("judged") or None,
        "split": seen.get("split"),
        "aligned": seen.get("aligned"),
        "not_aligned": seen.get("not_aligned") or None,
        "choices": _choices(seen.get("choices")),
        "behaviours": capped(
            [
                {**(outcome(found) or {}), "count": found.get("count"), "seen": found.get("seen"),
                 "matches": found.get("matches"), "verdicts": found.get("verdicts") or None}
                for found in seen.get("behaviours") or []
            ],
            30,
        ),
        "letters": (
            {
                "resample": short(latest.get("id")),
                "normalizers": (targets or {}).get("normalizers"),
                **(distribution((targets or {}).get("behaviours")) or {}),
            }
            if latest
            else None
        ),
    }


# --- experiments --------------------------------------------------------------------------------


def _choices(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """The scored runs' tool choices: the mean chance of each tool, and each run's, capped."""
    if not found or not found.get("scored"):
        return None
    return {
        "scored": found.get("scored"),
        "of": found.get("of"),
        "chance": found.get("chance"),
        "runs": capped([
            {"run": short(entry.get("run")), "verdict": entry.get("verdict"), "chosen": entry.get("chosen"),
             "chance": entry.get("chance")}
            for entry in found.get("runs") or []
        ], 10),
    }


def choice_text(chance: dict[str, Any] | None) -> str:
    """Each tool's chance, likeliest first: `execute_code 58%, write_file 40%`."""
    return ", ".join(f"{name} {float(p):.0%}" for name, p in (chance or {}).items())


def target(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """A declared target's test: each arm's rate, Fisher's p, the threshold, the verdict."""
    if not found:
        return None
    data = {
        "description": found.get("description"),
        "original": rate(found.get("original")),
        "variant": rate(found.get("variant")),
        "p_value": found.get("p_value"),
        "alpha": found.get("alpha"),
        "significant": found.get("significant"),
    }
    if "samples_to_detect" in found:
        data["samples_to_detect"] = found["samples_to_detect"]
    return data


def comparison(found: dict[str, Any] | None) -> dict[str, Any] | None:
    if not found:
        return None
    return {
        "exploratory": found.get("exploratory"),
        "target": target(found.get("target")),
        "shifts": [
            {
                "letter": shift.get("letter"),
                **(outcome(shift.get("outcome")) or {}),
                "original": rate(shift.get("original")),
                "variant": rate(shift.get("variant")),
                "p_value": shift.get("p_value"),
                "adjusted_p": shift.get("adjusted_p"),
            }
            for shift in found.get("shifts") or []
        ],
    }


def to_end(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """Forks run to the end, judged by the task's check: the pass rate over the forks that
    count, what was left out and why, and what the model did at the step."""
    if not found:
        return None
    if "original" in found and "variant" in found:
        return {
            "original": to_end(found["original"]),
            "variant": to_end(found["variant"]),
            "p_value": found.get("p_value"),
            "alpha": found.get("alpha"),
            "significant": found.get("significant"),
        }
    return {
        "passed": rate(found.get("rate")),
        "counted": found.get("counted"),
        "left_out": capped(
            [{"run": short(entry.get("run_id")), "why": clip(entry.get("why"), 200)} for entry in found.get("exclusions") or []],
            10,
        ) or None,
        "did_at_step": distribution(found.get("actions")),
        "forks": capped(
            [
                {"idx": fork.get("idx"), "run": short(fork.get("run_id")), "passed": fork.get("passed"),
                 "left_out": clip(fork.get("exclusion"), 120)}
                for fork in found.get("forks") or []
            ],
            20,
        ),
    }


def forking(found: dict[str, Any] | None, *, positions: int = 10) -> dict[str, Any] | None:
    """A forking-token probe: at each position, the base's token against each alternative
    forced there, Holm-corrected over the family planned before sampling."""
    if not found:
        return None

    def arm(entry: dict[str, Any]) -> dict[str, Any]:
        forced = entry.get("arm") or {}
        data = {
            "text": forced.get("text"),
            "p": forced.get("p"),
            "target": rate(entry.get("target")),
            "drawn": entry.get("drawn"),
            "planned": entry.get("planned"),
            "errors": entry.get("errors") or None,
        }
        for key in ("p_value", "adjusted_p", "flips", "direction"):
            if entry.get(key) is not None:
                data[key] = entry[key]
        return data

    ranked = sorted(
        found.get("positions") or [],
        key=lambda position: min(
            (alt.get("adjusted_p") for alt in position.get("alternatives") or [] if alt.get("adjusted_p") is not None),
            default=1.0,
        ),
    )
    return {
        "target": found.get("target"),
        "alpha": found.get("alpha"),
        "family": found.get("family"),
        "flips": found.get("flips"),
        "reproduced": rate(found.get("reproduced")),
        "missed": {why: listed for why, listed in (found.get("missed") or {}).items() if listed} or None,
        "positions": capped(
            [
                {
                    "position": position.get("position"),
                    "part": position.get("part"),
                    "context": clip(position.get("context"), 120),
                    "baseline": arm(position.get("baseline") or {}),
                    "alternatives": [arm(alt) for alt in position.get("alternatives") or []],
                    "weighted": position.get("weighted"),
                    "why_missed": position.get("why_missed"),
                }
                for position in ranked
            ],
            positions,
        ),
        "behaviours": distribution(found.get("behaviours")),
    }


def _screen_arm(arm: dict[str, Any] | None) -> dict[str, Any] | None:
    if not arm:
        return None
    data = {
        "name": arm.get("name"),
        "target": rate(arm.get("target")),
        "drawn": arm.get("drawn"),
        "planned": arm.get("planned"),
        "errors": arm.get("errors") or None,
        "p_value": arm.get("p_value"),
        "adjusted_p": arm.get("adjusted_p"),
        "direction": arm.get("direction"),
        "survives": arm.get("survives"),
        "worse": arm.get("worse"),
    }
    return {key: value for key, value in data.items() if value is not None}


def screen(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """Several proposals, each against one fresh baseline, Holm-corrected over the family
    planned before sampling. No verdict until every arm has drawn what was planned."""
    if not found:
        return None
    return {
        "target": found.get("target"),
        "reach": found.get("reach"),
        "alpha": found.get("alpha"),
        "family": found.get("family"),
        "complete": found.get("complete"),
        "quick": found.get("quick") or None,
        "reused": found.get("reused") or None,
        "baseline": _screen_arm(found.get("baseline")),
        "arms": [_screen_arm(arm) for arm in found.get("arms") or []],
        "survivors": found.get("survivors"),
        "checkpoints": [
            {"run": short(entry.get("run")), "step": entry.get("step"), "verdict": entry.get("verdict"),
             "arms": entry.get("arms")}
            for entry in found.get("checkpoints") or []
        ] or None,
        "across_checkpoints": found.get("across_checkpoints") or None,
        "behaviours": capped(
            [
                {"letter": entry.get("letter"), **(outcome(entry.get("outcome")) or {}),
                 "count": entry.get("count"), "by_arm": entry.get("by_arm")}
                for entry in found.get("behaviours") or []
            ],
            12,
        ),
    }


def experiment(report: dict[str, Any]) -> dict[str, Any]:
    """A stored experiment's result: a distribution, an A/B, forks to the end, or a probe."""
    settings = report.get("settings") or {}
    data: dict[str, Any] = {
        "id": short(report.get("id")),
        "kind": report.get("kind"),
        "run": short(report.get("run_id")),
        "step": report.get("step_idx"),
        "reach": report.get("reach"),
        "status": report.get("status"),
        "n": settings.get("n"),
        "normalizers": settings.get("normalizers") or None,
        "target": (settings.get("target") or {}).get("description"),
        "partial": report.get("partial") or None,
        "running": report.get("running") or None,
        "notes": [clip(note, 300) for note in report.get("notes") or []] or None,
        "job": short(report.get("job_id")),
    }
    for key in ("interrupted", "running_arm"):
        if report.get(key):
            data[key] = report[key]
    if report.get("variant"):
        data["variant"] = report["variant"]
    if report.get("wording"):
        data["wording"] = wording(report["wording"])
    if "distribution" in report:
        data["distribution"] = distribution(report["distribution"])
    if "comparison" in report:
        data["comparison"] = comparison(report["comparison"])
    if "to_end" in report:
        data["to_end"] = to_end(report["to_end"])
    if "forking" in report:
        data["forking"] = forking(report["forking"])
    if "screen" in report:
        data["screen"] = screen(report["screen"])
        data["arms"] = {
            name: {key: value for key, value in arm.items() if key in ("proposal", "samples", "status") and value is not None}
            | {"id": short(arm.get("id"))}
            for name, arm in (report.get("arms") or {}).items()
        }
    return {key: value for key, value in data.items() if value is not None}


def wording(found: dict[str, Any]) -> dict[str, Any]:
    """The edit an A/B tested: the lines it changed, as a refinement's proposals show theirs."""
    diff = found.get("diff")
    data = {
        "original": (found.get("original") or {}).get("label"),
        "variant": (found.get("variant") or {}).get("label"),
        "changed": diff.get("changed") if diff else None,
        "diff": [line for hunk in diff["hunks"] for line in hunk["lines"] if line["op"] != " "][:12] if diff else None,
        "form": found.get("form"),
        "missing": found.get("missing"),
    }
    return {key: value for key, value in data.items() if value is not None}


def wording_lines(found: dict[str, Any] | None) -> list[str]:
    if not found:
        return []
    if found.get("changed") is None:
        return [f"the edit it tested is not shown: {found.get('missing')}"]
    lines = [f"the edit it tested, original {found.get('original')} → variant {found.get('variant')}: "
             f"{found['changed']} line(s) changed"]
    lines += [f"  {line['op']} {clip(line['text'], 200)}" for line in (found.get("diff") or [])[:6]]
    return lines


def experiments(listed: Sequence[dict[str, Any]], *, limit: int = 30) -> dict[str, Any] | list[Any]:
    """Experiments, newest first, an A/B once: its arms are one experiment to a reader, and
    either arm's id shows the comparison."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for found in listed:
        if found.get("group_id"):
            arms.setdefault(found["group_id"], []).append(found)
    shown, seen = [], set()
    for found in listed:
        group = found.get("group_id")
        if group in seen:
            continue
        entry = found
        if group:
            seen.add(group)
            entry = next((arm for arm in arms[group] if arm.get("arm") == "original"), found)
        shown.append({
            "id": short(entry.get("id")),
            "kind": entry.get("kind"),
            "reach": entry.get("reach"),
            "arms": sorted(arm.get("arm") for arm in arms[group]) if group else None,
            "run": short(entry.get("run_id")),
            "step": entry.get("step_idx"),
            "samples": sum(arm.get("samples") or 0 for arm in arms[group]) if group else entry.get("samples"),
            "n": entry.get("n"),
            "target": entry.get("target"),
            "status": entry.get("status"),
            "created": when(entry.get("created_at")),
        })
    return capped([{key: value for key, value in entry.items() if value is not None} for entry in shown], limit)


# --- tokens -------------------------------------------------------------------------------------


def tokens(found: dict[str, Any], *, positions: Sequence[int] = (), hotspots: int = 10) -> dict[str, Any]:
    """A step's answer as tokens: where the model was least sure, and of what. Every token's
    numbers only at the positions asked for."""
    view = found.get("view") or {}
    stream = view.get("stream") or {}
    data: dict[str, Any] = {
        "run": short(found.get("run_id")),
        "step": found.get("step"),
        "scored": bool(view),
        "streams": len(found.get("streams") or []),
    }
    if not view:
        data["hint"] = "no token data at this step yet: start_tokens scores the answer"
        return data
    wanted = set(positions)
    data.update({
        "stream": {key: stream.get(key) for key in ("id", "source", "method", "exact", "top_k", "notes") if stream.get(key) is not None},
        "same_server": view.get("same_server"),
        "format": view.get("format"),
        "sampler": view.get("sampler"),
        "choice": view.get("choice"),
        "text": clip(view.get("text"), 1500),
        "hotspots": [
            {
                "position": spot.get("position"),
                "part": spot.get("part"),
                "chosen": spot.get("chosen"),
                "alternatives": (spot.get("alternatives") or [])[:3],
                "runner_up": spot.get("runner_up"),
                "entropy": spot.get("entropy"),
            }
            for spot in (view.get("hotspots") or [])[:hotspots]
        ],
        "probes": [
            {key: probe.get(key) for key in ("experiment_id", "positions", "status", "flips") if probe.get(key) is not None}
            for probe in found.get("probes") or []
        ] or None,
    })
    if wanted:
        data["tokens"] = [
            {key: entry.get(key) for key in ("i", "text", "part", "p", "entropy", "margin", "alternatives")}
            for entry in view.get("tokens") or []
            if entry.get("i") in wanted
        ]
    return data


# --- fitting a budget ---------------------------------------------------------------------------


def size(data: Any) -> int:
    return len(json.dumps(data, separators=(",", ":"), default=str))


def _trim(data: Any, text: int, items: int) -> Any:
    if isinstance(data, str):
        return clip(data, text)
    if isinstance(data, list):
        kept = [_trim(item, text, items) for item in data[:items]]
        return kept if len(data) <= items else {"shown": kept, "total": len(data)}
    if isinstance(data, dict):
        if "shown" in data and "total" in data and isinstance(data["shown"], list):
            kept = data["shown"][:items]
            return {"shown": [_trim(item, text, items) for item in kept], "total": data["total"]}
        return {key: _trim(value, text, items) for key, value in data.items()}
    return data


def fit(data: Any, budget: int = BUDGET) -> Any:
    """`data` as it is when it fits the budget; otherwise with texts clipped and lists capped
    harder and harder until it does. Rates are never cut: their numbers are not texts."""
    if size(data) <= budget:
        return data
    text, items = TEXT, ITEMS
    while text > 40 or items > 3:
        data = _trim(data, text, items)
        if size(data) <= budget:
            return data
        text, items = max(40, text // 2), max(3, items // 2)
    return data


# --- evidence, in words -------------------------------------------------------------------------


def _label(found: dict[str, Any]) -> str:
    return found.get("label") or found.get("kind") or "?"


def distribution_lines(found: dict[str, Any] | None, *, what: str = "samples") -> list[str]:
    if not found:
        return []
    head = f"{found.get('total', 0)} {what}"
    if found.get("errors"):
        head += f" (+{found['errors']} transport errors, not counted)"
    lines = [head + ":"]
    for cluster in found.get("behaviours") or []:
        reasoned = cluster.get("reasoned")
        said = f" ({reasoned} of them reasoned)" if reasoned is not None else ""
        lines.append(f"  {cluster.get('letter')}: {rate_text(cluster.get('rate'))} {_label(cluster)}{said}")
    return lines


def target_lines(found: dict[str, Any]) -> list[str]:
    verdict = "significant" if found.get("significant") else "not significant"
    lines = [
        f"declared target: {found.get('description')}",
        f"  original {rate_text(found.get('original'))} → variant {rate_text(found.get('variant'))}",
        f"  Fisher's exact test {p_text(found.get('p_value'))}, alpha {found.get('alpha')}: {verdict}",
    ]
    if not found.get("significant") and found.get("samples_to_detect"):
        lines.append(f"  about {found['samples_to_detect']} samples per arm would detect a difference this size (80% power)")
    return lines


def comparison_lines(found: dict[str, Any] | None) -> list[str]:
    if not found:
        return []
    if found.get("target"):
        return target_lines(found["target"])
    lines = ["exploratory: no target was declared before sampling, so every shift is Bonferroni-adjusted"]
    for shift in found.get("shifts") or []:
        lines.append(
            f"  {shift.get('letter')}: original {rate_text(shift.get('original'))} → variant "
            f"{rate_text(shift.get('variant'))}, adjusted {p_text(shift.get('adjusted_p'))} {_label(shift)}"
        )
    return lines


def to_end_lines(found: dict[str, Any] | None) -> list[str]:
    if not found:
        return []
    if "original" in found and "variant" in found:
        verdict = "significant" if found.get("significant") else "not significant"
        return [
            f"run to the end, judged by the task's check: original {rate_text(found['original'].get('passed'))}"
            f" → variant {rate_text(found['variant'].get('passed'))}",
            f"  Fisher's exact test {p_text(found.get('p_value'))}, alpha {found.get('alpha')}: {verdict}",
        ]
    return [f"run to the end, judged by the task's check: passed {rate_text(found.get('passed'))}"]


def forking_lines(found: dict[str, Any] | None) -> list[str]:
    if not found:
        return []
    lines = [
        f"forking-token probe on target {found.get('target')!r}: {found.get('flips') or 0} alternative(s) flip it"
        f" (Holm-corrected over {found.get('family')} planned, alpha {found.get('alpha')})",
    ]
    if found.get("reproduced"):
        lines.append(f"  the base's own tokens led back to its behaviour {rate_text(found['reproduced'])}")
    positions = found.get("positions") or []
    for position in positions.get("shown", positions) if isinstance(positions, dict) else positions:
        for alt in position.get("alternatives") or []:
            if alt.get("flips"):
                lines.append(
                    f"  position {position.get('position')}: {alt.get('text')!r} instead of "
                    f"{(position.get('baseline') or {}).get('text')!r} → target {rate_text(alt.get('target'))}"
                    f" vs {rate_text((position.get('baseline') or {}).get('target'))}, adjusted {p_text(alt.get('adjusted_p'))}"
                )
    return lines


def screen_lines(found: dict[str, Any] | None) -> list[str]:
    if not found:
        return []
    family, complete = found.get("family"), found.get("complete")
    measure = (
        "by the task's check, each arm's samples forks run from the checkpoint to the end"
        if found.get("reach") == "end" else f"on target {found.get('target')!r}"
    )
    baseline = (found.get("baseline") or {}).get("target") or {}
    reused = found.get("reused") or 0
    if found.get("quick"):
        lines = [
            f"quick look at {family} proposal(s) {measure}: rates only, no verdict — a lead for the next edit, and "
            "only a full screen can keep one",
            (f"  original {rate_text(baseline)}, the forks this refinement drew before" if baseline.get("total")
             else "  original: none of its forks drawn in this refinement yet — set the rates beside the task's "
                  "recorded runs (list_tasks)"),
        ]
        for arm in found.get("arms") or []:
            lines.append(f"  {arm.get('name')} {rate_text(arm.get('target'))}"
                         + ("" if complete else " so far"))
        if not complete:
            lines.append("  unfinished: rates so far")
        lines.extend(checkpoint_lines(found))
        return lines
    against = "the original's forks" if found.get("reach") == "end" else "a fresh baseline"
    lines = [
        f"screen of {family} proposal(s) {measure}, each against {against}: Fisher's exact test, "
        f"Holm-corrected over the {family} planned (alpha {found.get('alpha')})",
        f"  original {rate_text(baseline)}"
        + (f" ({reused} of its forks drawn by an earlier screen of this refinement, counted again)" if reused else ""),
    ]
    for arm in found.get("arms") or []:
        if not complete:
            verdict = "no verdict yet"
        elif arm.get("survives"):
            verdict = "survives"
        elif arm.get("worse"):
            verdict = "significantly worse"
        else:
            verdict = "not significant"
        lines.append(
            f"  {arm.get('name')} {rate_text(arm.get('target'))}, adjusted {p_text(arm.get('adjusted_p'))}: {verdict}"
        )
    if not complete:
        lines.append("  unfinished: rates only — there is no verdict until every arm has drawn what was planned")
    elif not found.get("survivors"):
        lines.append("  no proposal survived the correction")
        baseline = (found.get("baseline") or {}).get("target") or {}
        if found.get("reach") != "end" and baseline.get("total") and baseline["successes"] / baseline["total"] >= 0.8:
            # Nothing can beat an original arm at the ceiling: the checkpoint is where the model
            # went right, usually a passing run's step.
            lines.append(
                f"  the original reaches the target {baseline['successes']}/{baseline['total']} at this checkpoint: "
                "no edit can show a gain here — a failed run's step, or a screen to the end from several "
                "checkpoints, measures it where the model went wrong"
            )
    lines.extend(checkpoint_lines(found))
    return lines


def checkpoint_lines(found: dict[str, Any]) -> list[str]:
    """Where a screen to the end drew its forks from, when from several checkpoints, and
    whether each arm's effect held at every one of them."""
    checkpoints = found.get("checkpoints") or []
    if len(checkpoints) < 2:
        return []
    verdicts = [entry.get("verdict") for entry in checkpoints]
    where = ", ".join(f"{entry.get('run')} step {entry.get('step')} ({entry.get('verdict') or 'unjudged'})" for entry in checkpoints)
    lines = [f"  forks drawn from {len(checkpoints)} checkpoints — the same step of {verdicts.count('passed')} passing and "
             f"{verdicts.count('failed')} failing run(s): {where}"]
    for entry in checkpoints:
        arms = entry.get("arms") or {}
        counts = ", ".join(f"{name} {found_['passed']}/{found_['of']}" for name, found_ in arms.items())
        lines.append(f"    at {entry.get('run')} step {entry.get('step')}: {counts}")
    for name, tally in (found.get("across_checkpoints") or {}).items():
        of = tally.get("of") or 0
        if tally.get("above") == of:
            held = "above the original at every checkpoint: an effect that holds across prefixes"
        elif tally.get("above", 0) <= 1 and of > 1:
            held = (f"above the original at {tally.get('above', 0)} of {of} checkpoints: an effect of one prefix, which "
                    "fresh runs draw anew")
        else:
            held = f"above the original at {tally.get('above', 0)} of {of} checkpoints, below at {tally.get('below', 0)}"
        lines.append(f"    {name}: {held}")
    return lines


def experiment_lines(found: dict[str, Any]) -> list[str]:
    head = f"experiment {found.get('id')} ({found.get('kind')}) at run {found.get('run')} step {found.get('step')}: {found.get('status')}"
    lines = [head, *wording_lines(found.get("wording"))]
    if found.get("screen"):
        return lines + screen_lines(found["screen"])
    if found.get("forking"):
        return lines + forking_lines(found["forking"])
    if found.get("comparison"):
        return lines + comparison_lines(found["comparison"])
    if found.get("to_end"):
        return lines + to_end_lines(found["to_end"])
    return lines + distribution_lines(found.get("distribution"))


def divergence_lines(found: dict[str, Any]) -> list[str]:
    index = found.get("divergence_index")
    if index is None:
        return [f"{found.get('runs')} runs never stopped agreeing, step for step"]
    lines = [f"{found.get('runs')} runs agree for {found.get('agreed_steps')} step(s), then split at step {index}:"]
    for entry in found.get("steps") or []:
        if entry.get("index") != index:
            continue
        for group in entry.get("groups") or []:
            passed = group.get("passed")
            lines.append(
                f"  {group.get('runs')} run(s) {_label(group)}"
                + (f" — went on to pass {rate_text(passed)}" if passed else " — no verdicts")
            )
    if found.get("mixed_skills"):
        lines.append("  warning: these runs were given different skill versions; compare runs of one version")
    return lines


def task_lines(found: dict[str, Any]) -> list[str]:
    conditions = found.get("conditions") or []
    if len(conditions) <= 1:
        return [f"{found.get('task_id')}: {found.get('runs')} run(s), passed {rate_text(found.get('passed'))}"]
    lines = [f"{found.get('task_id')}: {found.get('runs')} run(s) under {len(conditions)} conditions:"]
    for condition in conditions:
        lines.append(
            f"  {condition.get('harness')} · {condition.get('model')} · skill {condition.get('skill_version')}:"
            f" passed {rate_text(condition.get('passed'))}"
        )
    return lines


def tally(report: dict[str, Any], show: int = 10) -> dict[str, Any]:
    """A tally of what the checks said, small enough for an agent: each group's most common
    findings, each with its count over the failed runs and its interval, and a few of the runs."""
    groups = []
    for group in report.get("groups") or []:
        findings = group.get("findings") or []
        groups.append({
            "label": group.get("label"),
            "arm": group.get("arm"),
            "condition": short((group.get("condition") or {}).get("id")),
            "judged": group.get("judged"),
            "failed": group.get("failed"),
            "silent": group.get("silent"),
            "findings": capped([{
                "finding": clip(finding.get("finding"), LABEL),
                "rate": rate(finding.get("rate")),
                "runs": [short(run_id) for run_id in (finding.get("run_ids") or [])[:3]],
                # Where the check was run along the runs: which call made it, passing runs included.
                "made": made(finding.get("made")),
            } for finding in findings], show),
        })
    return {"scope": report.get("scope"), "task_id": report.get("task_id"), "screen": short(report.get("screen")),
            "run_id": short(report.get("run_id")), "groups": groups}


def made(found: dict[str, Any] | None) -> dict[str, Any] | None:
    """Where the check along the runs says a finding was made: in how many of the runs it was run
    along, at which steps, gone by the end in how many, and whether every run made it."""
    if not found:
        return None
    from tuningfork.ops.tally import made_steps, made_text

    return {"runs": len(found.get("runs") or []), "of": found.get("checked"), "steps": made_steps(found),
            "repaired": found.get("repaired"), "passing": found.get("passing"), "every_run": found.get("every_run"),
            "text": made_text(found)}


def tally_lines(view: dict[str, Any]) -> list[str]:
    """The tally in sentences: each group, then its findings, the most common first — with, where
    the check was run along the runs, which call made each."""
    if view.get("scope") == "screen":
        lines = [f"screen {view.get('screen')} from run {view.get('run_id')}: what each arm's failed forks' "
                 "checks said, the most common first"]
    else:
        lines = [f"{view.get('task_id')}: what its failed runs' checks said, the most common first"]
    for group in view.get("groups") or []:
        lines.append(f"{group.get('label')}: {group.get('failed')} of {group.get('judged')} judged failed")
        found = group.get("findings") or []
        shown = found.get("shown") if isinstance(found, dict) else found
        for finding in shown:
            lines.append(f"  {rate_text(finding.get('rate'))} of the failed runs: {finding.get('finding')}")
            along = finding.get("made")
            if along:
                lines.append(f"    {along['text']}")
        if isinstance(found, dict):
            lines.append(f"  … {found['total'] - len(shown)} more finding(s)")
        if group.get("silent"):
            lines.append(f"  {group['silent']} failed run(s) whose check printed nothing")
    lines.append("each line a check printed is one finding, numbers shown as #, colours as <color>; a run counts once a finding")
    return lines


def where(report: dict[str, Any], show: int = 12) -> dict[str, Any]:
    """Where a question held along a task's runs, small enough for an agent: each condition's split
    by verdict with its intervals, where it was first seen or made, the step that names that call in
    each run, and a few of the runs."""
    kind = (report.get("probe") or {}).get("kind")
    groups = []
    for group in report.get("groups") or []:
        split = group.get("split") or {}
        runs = []
        for found in group.get("runs") or []:
            entry = {"run": short(found.get("run")), "verdict": found.get("verdict"), "held": found.get("held"),
                     "step": found.get("step"), "tools": found.get("tools"), "reference": found.get("reference")}
            if kind == "file":
                entry.update({key: found.get(key) for key in ("at_start", "undone_by", "at_end", "file", "count")})
            else:
                entry["steps"] = (found.get("steps") or [])[:5]
            runs.append(entry)
        groups.append({
            "label": group.get("label"),
            "condition": short((group.get("condition") or {}).get("id")),
            "passed": group.get("passed"),
            "failed": group.get("failed"),
            "unjudged": group.get("unjudged"),
            "split": {"passing": rate(split.get("passing")), "failing": rate(split.get("failing")),
                      "p": split.get("p"), "goes_with": split.get("goes_with")},
            "seen_at": group.get("seen_at"),
            "reference": group.get("reference"),
            "made": group.get("made"),
            "runs": capped(runs, show),
        })
    return {"task_id": report.get("task_id"), "screen": short(report.get("screen")), "probe": report.get("probe"),
            "groups": groups, "skipped": report.get("skipped"), "skipped_more": report.get("skipped_more")}


def where_lines(view: dict[str, Any]) -> list[str]:
    """Where a question held, in sentences: the question, then each condition's split by verdict,
    where it was first seen or made, and the step that names that call in each run."""
    from tuningfork.ops.where import reference_text, seen_text, split_text

    probe = view.get("probe") or {}
    kind = probe.get("kind")
    lines = [f"{view.get('task_id')}: {probe.get('description')}, asked of each "
             + (f"fork of screen {view['screen']}, arm by arm" if view.get("screen") else "run")]
    for group in view.get("groups") or []:
        lines.append(f"{group.get('label')}: {group.get('passed')} passed, {group.get('failed')} failed"
                     + (f", {group['unjudged']} without a verdict that counts" if group.get("unjudged") else ""))
        for line in (split_text(group, kind), seen_text(group, kind), reference_text(group)):
            if line:
                lines.append(f"  {line}")
    if view.get("skipped"):
        lines.append("passed over: " + "; ".join(view["skipped"][:3]))
    lines.append("a lead to read, never a verdict: a split is said to go one way under p 0.2; intervals are Wilson 95%")
    return lines


#: The sentences for each view, by name: what an MCP result's text opens with.
LINES: dict[str, Callable[[Any], list[str]]] = {
    "experiment": experiment_lines,
    "divergence": divergence_lines,
    "task": task_lines,
    "tally": tally_lines,
    "where": where_lines,
}
