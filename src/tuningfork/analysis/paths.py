"""A task's runs as paths that part.

A divergence report says, step by step, which runs did what. Read along the runs instead, the
same report is a tree: the runs start together, and where they did different things their path
forks. A node is the runs that did the same things so far; its children are what they did next.

What "the same thing" is decides what the tree shows. Two replies that differ in a word, or two
searches with another pattern, are one move here: a node is the tools a step called, whatever
their arguments, or a reply, whatever its words. Compared exactly, as the divergence report
compares them, twenty runs of a task were twelve behaviours at the step that decided them, nine
of them the same refusal reworded; by move they were two, and the two were the passing runs and
the failing ones. A tool's calls are drawn apart only where they came in a few distinct forms
(`forms`): the demo's two ways of writing its report are two branches where its runs part, and
are drawn as two.

The tree is where the turning point is looked for: the split whose branches best separate the
runs that passed from the runs that failed, each branch against the rest of the runs that got
there. A step-by-step comparison of every run (`ops.inspect.verdict_split`) cannot name it once
runs have ended: the runs that gave up at step 1 are "ended" at every step after, and that is
what it finds.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from tuningfork.analysis.stats import fisher_exact, p_short, rate

#: How a tool's calls are told apart: never ("tools"), where they came in a few distinct forms
#: ("auto"), or always ("calls"). A reply is one move under each.
FORMS = ("tools", "auto", "calls")

#: What a step that called no tool and said nothing is called, by its kind.
MOVE_LABELS = {"empty": "no answer", "truncated": "cut off", "error": "call failed"}
#: And what its runs did, as a sentence says it: "runs that were cut off".
MOVE_DID = {"empty": "gave no answer", "truncated": "were cut off", "error": "had a failed call"}

#: A form of a call is drawn apart when at least this many runs took it …
FORM_RUNS = 2
#: … and the forms that many runs took are at least this share of the calls.
FORM_SHARE = 0.8


def _move(group: dict[str, Any]) -> tuple[str, str, str, list[str]] | None:
    """A behaviour as a move: its key, kind, label and tools. None for a run that had ended."""
    outcome = group.get("outcome") or {}
    kind = outcome.get("kind") or "text"
    if kind == "ended":
        return None
    if kind == "tool_call":
        tools = list(group.get("tools") or []) or [str(outcome.get("label") or "?").split("(")[0]]
        return ("tool:" + "+".join(tools), kind, " + ".join(tools), tools)
    if kind == "text":
        return ("reply", kind, "reply", [])
    return (kind, kind, MOVE_LABELS.get(kind, kind), [])


def outcome_move(outcome: Any) -> tuple[str, str, str, list[str], str]:
    """An outcome (`analysis.normalize.Outcome`) as a move: its key, kind, label and tools, and
    the behaviour's own label."""
    from tuningfork.analysis.normalize import called_tools

    exact = str(outcome.label)
    if outcome.kind == "tool_call":
        tools = list(called_tools(outcome) or ()) or [exact.split("(")[0]]
        return ("tool:" + "+".join(tools), "tool_call", " + ".join(tools), tools, exact)
    if outcome.kind == "text":
        return ("reply", "text", "reply", [], exact)
    return (outcome.kind, outcome.kind, MOVE_LABELS.get(outcome.kind, outcome.kind), [], exact)


def json_move(row: dict[str, Any]) -> tuple[str, str, str, list[str], str]:
    """A behaviour's JSON — its `outcome`, and the `tools` it called — as a move."""
    found = _move(row)
    assert found is not None, "a behaviour seen at a step is never a run that had ended"
    return (*found, str((row.get("outcome") or {}).get("label")))


def ended_json(tally: Iterable[int]) -> dict[str, Any]:
    """How the forks that did one thing at a step ended, each run on to the end: how many passed
    the task's check, failed it, and had no verdict — no check judged them — and the pass rate
    over those judged, with its interval; none where none was. A fork left out of the step's
    rates (it timed out, or measured nothing) did nothing there to count, and is in no tally."""
    passed, failed, unjudged = tally
    return {
        "passed": passed, "failed": failed, "no_verdict": unjudged,
        "rate": rate(passed, passed + failed).to_json() if passed + failed else None,
    }


def gather_moves(
    rows: Iterable[tuple[Any, ...]], totals: Mapping[str, int],
    move_of: Callable[[Any], tuple[str, str, str, list[str], str]] = outcome_move,
) -> list[dict[str, Any]]:
    """Behaviours gathered by move, for what one step led to: `rows` are each behaviour, its
    letter, and how many times each arm did it; `totals`, each arm's samples. Where each sample
    was a fork run to the end, a row's fourth item is how each arm's forks that did it ended —
    `{arm: (passed, failed, no verdict)}` — and each move then says it of its own (`ended`,
    `ended_json` for each arm): a branch of a fan ends in its forks, as a path of runs does.

    A move is the tools a behaviour called, or a reply, whatever its words: sixteen behaviours
    at a step, fifteen of them one refusal reworded, are two moves. A tool's calls are told
    apart as a tree of paths tells them (`_parts`): where they came in a few distinct forms,
    counted over every arm together, each form is a move of its own, with what tells it from
    the others (`hint`). Each move says its behaviours' letters, how many forms they were — over
    every arm (`forms`) and in each arm's own samples (`forms_by_arm`: an arm that never replied
    took none of a reply's eleven phrasings) — and for each arm how many of its samples made the
    move, as a rate with its interval. The most made first. `move_of` reads a row's behaviour:
    an Outcome, or (`json_move`) its JSON."""
    moves: dict[str, dict[str, Any]] = {}
    for behaviour, letter, by_arm, *more in rows:
        key, kind, label, tools, exact = move_of(behaviour)
        move = moves.setdefault(key, {"kind": kind, "label": label, "tools": tools, "variants": []})
        pooled = sum(by_arm.values())
        # `runs`: one for each sample, so that a form is counted as the tree counts a form's runs.
        move["variants"].append({
            "outcome": {"label": exact}, "letter": letter, "counts": dict(by_arm), "runs": [None] * pooled,
            "ended": dict(more[0]) if more and more[0] is not None else None,
        })
    gathered: list[dict[str, Any]] = []
    for move in moves.values():
        variants = sorted(move["variants"], key=lambda variant: (-len(variant["runs"]), variant["outcome"]["label"]))
        parts = _parts(move["kind"], variants, "auto")
        labels = [part["variants"][0]["outcome"]["label"] for part in parts if part["apart"] and not part["others"]]
        for part in parts:
            counts = {arm: sum(variant["counts"].get(arm, 0) for variant in part["variants"]) for arm in totals}
            hint = None
            if part["others"]:
                hint = "1 other form" if len(part["variants"]) == 1 else f"{len(part['variants'])} other forms"
            elif part["apart"]:
                hint = _hint(part["variants"][0]["outcome"]["label"], labels)
            ended = {}
            if any(variant["ended"] is not None for variant in part["variants"]):
                tallies = [variant["ended"] for variant in part["variants"] if variant["ended"] is not None]
                ended = {"ended": {
                    arm: ended_json(sum(tally.get(arm, (0, 0, 0))[at] for tally in tallies) for at in range(3))
                    for arm, total in totals.items() if total
                }}
            gathered.append({
                **ended,
                "kind": move["kind"], "label": move["label"], "tools": move["tools"],
                "letters": [variant["letter"] for variant in part["variants"] if variant["letter"]],
                "forms": len(part["variants"]), "apart": part["apart"], "others": part["others"], "hint": hint,
                "forms_by_arm": {
                    arm: sum(1 for variant in part["variants"] if variant["counts"].get(arm, 0))
                    for arm, total in totals.items() if total
                },
                "by_arm": {arm: rate(counts[arm], total).to_json() for arm, total in totals.items() if total},
                "_made": sum(counts.values()),
            })
    gathered.sort(key=lambda move: (-move.pop("_made"), move["label"], move["hint"] or ""))
    return gathered


def _parts(kind: str, variants: list[dict[str, Any]], forms: str) -> list[dict[str, Any]]:
    """One move's runs as the nodes drawn for it: together, or each form of the call apart."""
    whole = [{"variants": variants, "apart": False, "others": False}]
    if kind != "tool_call" or len(variants) < 2 or forms == "tools":
        return whole
    if forms == "calls":
        return [{"variants": [variant], "apart": True, "others": False} for variant in variants]
    clean = [variant for variant in variants if len(variant["runs"]) >= FORM_RUNS]
    total = sum(len(variant["runs"]) for variant in variants)
    if len(clean) < 2 or sum(len(variant["runs"]) for variant in clean) < FORM_SHARE * total:
        return whole
    rest = [variant for variant in variants if len(variant["runs"]) < FORM_RUNS]
    parts = [{"variants": [variant], "apart": True, "others": False} for variant in clean]
    if rest:
        parts.append({"variants": rest, "apart": True, "others": True})
    return parts


def _hint(label: str, siblings: list[str], limit: int = 26) -> str:
    """What tells one form of a call from the others, in a few words: its label from where the
    labels first differ, back to the start of that argument."""
    common = 0
    while all(common < len(other) and common < len(label) and other[common] == label[common] for other in siblings):
        common += 1
    start = max(label.rfind("=", 0, common + 1), label.rfind("(", 0, common + 1), label.rfind(" ", 0, common + 1)) + 1
    start = min(start, common)
    text = label[start:]
    if text.endswith(")"):
        text = text[:-1]
    return text if len(text) <= limit else f"{text[:limit]}…"


def path_tree(report: dict[str, Any], *, forms: str = "auto", alpha: float = 0.2, minimum: int = 2) -> dict[str, Any]:
    """The runs of a divergence report (its JSON) as a tree of paths.

    `nodes` lists every node, the root first: its `step` (-1 for the root, every run), its move
    (`kind`, `label`, `tools`), the runs that came this way and those that ended here (`runs`,
    `ends`: indexes into the report's `run_ids`), how many of its judged runs passed (`passed`, a
    rate with its interval), the exact behaviours among its runs (`forms`), and its `children`,
    the most runs first. `apart` marks a form of a call drawn on its own, `hint` what tells it
    from the others, `others` the forms too rare to draw one by one. `turning` is the turning
    point (`turning_point`), or None."""
    if forms not in FORMS:
        raise ValueError(f"forms is one of {', '.join(FORMS)}: not {forms!r}")
    run_ids = list(report.get("run_ids") or [])
    index = {run_id: position for position, run_id in enumerate(run_ids)}
    verdicts = report.get("verdicts") or {}
    steps = report.get("steps") or []
    at: list[dict[int, dict[str, Any]]] = []
    for step in steps:
        by_run: dict[int, dict[str, Any]] = {}
        for group in step.get("groups") or []:
            for run_id in group.get("run_ids") or []:
                if run_id in index:
                    by_run[index[run_id]] = group
        at.append(by_run)

    def fared(runs: list[int]) -> tuple[int, int]:
        passed = sum(1 for run in runs if verdicts.get(run_ids[run]) == "passed")
        failed = sum(1 for run in runs if verdicts.get(run_ids[run]) == "failed")
        return passed, passed + failed

    def passed_rate(runs: list[int]) -> dict[str, Any] | None:
        passed, judged = fared(runs)
        return rate(passed, judged).to_json() if judged else None

    nodes: list[dict[str, Any]] = []

    def add(parent: int | None, step: int, runs: list[int], **more: Any) -> dict[str, Any]:
        node = {
            "id": len(nodes), "parent": parent, "step": step, "kind": None, "label": None, "tools": [],
            "runs": runs, "ends": [], "passed": passed_rate(runs), "ended": None, "forms": [],
            "apart": False, "others": False, "hint": None, "children": [], **more,
        }
        nodes.append(node)
        return node

    def grow(node: dict[str, Any]) -> None:
        depth = node["step"] + 1
        moves: dict[str, dict[str, Any]] = {}
        for run in node["runs"]:
            group = at[depth].get(run) if depth < len(at) else None
            move = _move(group) if group else None
            if move is None:
                node["ends"].append(run)
                continue
            key, kind, label, tools = move
            entry = moves.setdefault(key, {"kind": kind, "label": label, "tools": tools, "exact": {}})
            outcome = group["outcome"]
            variant = entry["exact"].setdefault(outcome.get("key") or outcome.get("label"), {"outcome": outcome, "runs": []})
            variant["runs"].append(run)
        made: list[dict[str, Any]] = []
        for entry in moves.values():
            variants = sorted(entry["exact"].values(), key=lambda variant: (-len(variant["runs"]), str(variant["outcome"].get("label"))))
            parts = _parts(entry["kind"], variants, forms)
            labels = [str(part["variants"][0]["outcome"].get("label")) for part in parts if part["apart"] and not part["others"]]
            for part in parts:
                runs = [run for variant in part["variants"] for run in variant["runs"]]
                hint = None
                if part["others"]:
                    hint = "1 other form" if len(part["variants"]) == 1 else f"{len(part['variants'])} other forms"
                elif part["apart"]:
                    hint = _hint(str(part["variants"][0]["outcome"].get("label")), labels)
                made.append(add(
                    node["id"], depth, runs, kind=entry["kind"], label=entry["label"], tools=entry["tools"],
                    forms=part["variants"], apart=part["apart"], others=part["others"], hint=hint,
                ))
        # The most runs first: a split's first branch is the way most of its runs went.
        made.sort(key=lambda child: (-len(child["runs"]), child["label"], child["hint"] or ""))
        node["children"] = [child["id"] for child in made]
        # Runs that ended here while others went on are a branch of the split too.
        if node["ends"] and made and node["parent"] is not None:
            node["ended"] = passed_rate(node["ends"])
        for child in made:
            grow(child)

    root = add(None, -1, list(range(len(run_ids))))
    grow(root)
    passed, judged = fared(root["runs"])
    tree = {"forms": forms, "runs": len(run_ids), "judged": judged, "passed": passed, "nodes": nodes}
    tree["turning"] = turning_point(tree, run_ids, verdicts, alpha=alpha, minimum=minimum)
    return tree


def turning_point(
    tree: dict[str, Any], run_ids: list[str], verdicts: dict[str, Any], *, alpha: float = 0.2, minimum: int = 2
) -> dict[str, Any] | None:
    """The split whose branches best separate the runs that passed from the runs that failed.

    At each place the runs parted, each branch — a move, or having ended there — against the
    rest of the runs that got there, by Fisher's exact test on passes and fails, with at least
    `minimum` judged runs on each side. The smallest p wins when it is under `alpha`; of equal
    ones, the earliest step, and there the branch the passing runs took, since that is the
    behaviour a target would name. A lead to read, never a verdict: with many splits compared,
    chance alone can make one. None when the runs all passed, all failed, or no split tells them
    apart clearly enough — no branch under `alpha`, or too few judged runs on a side: the
    difference may then be in what they wrote."""
    nodes = tree["nodes"]

    def fared(runs: list[int]) -> tuple[int, int]:
        passed = sum(1 for run in runs if verdicts.get(run_ids[run]) == "passed")
        failed = sum(1 for run in runs if verdicts.get(run_ids[run]) == "failed")
        return passed, failed

    found: list[tuple[tuple[float, int, int, int], dict[str, Any]]] = []
    for node in nodes:
        branches: list[tuple[dict[str, Any] | None, list[int]]] = [(nodes[child], nodes[child]["runs"]) for child in node["children"]]
        if node["ends"] and node["children"] and node["parent"] is not None:
            branches.append((None, node["ends"]))
        if len(branches) < 2:
            continue
        all_passed, all_failed = fared(node["runs"])
        for branch, runs in branches:
            passed, failed = fared(runs)
            others_passed, others_failed = all_passed - passed, all_failed - failed
            if passed + failed < minimum or others_passed + others_failed < minimum:
                continue
            p = fisher_exact(passed, failed, others_passed, others_failed)
            with_passing = passed / (passed + failed) > others_passed / (others_passed + others_failed)
            entry = {
                "parent": node["id"], "step": node["step"] + 1, "after": node["label"],
                "branch": branch["id"] if branch else None, "stopped": branch is None,
                "kind": branch["kind"] if branch else "ended", "label": branch["label"] if branch else "ended",
                "hint": branch["hint"] if branch and branch["apart"] and not branch["others"] else None,
                "passed": passed, "total": passed + failed,
                "others_passed": others_passed, "others_total": others_passed + others_failed,
                "p": round(p, 4), "with_passing": with_passing,
                "run_id": run_ids[runs[0]] if runs else None,
            }
            # Equal p's are equal to the last place a float keeps: a tie is one in ten places.
            found.append(((round(p, 10), node["step"], 0 if with_passing else 1, -len(runs)), entry))
    if not found:
        return None
    key, best = min(found, key=lambda each: each[0])
    return best if key[0] < alpha else None


def outline(tree: dict[str, Any], *, most: int = 40) -> list[str]:
    """The tree as lines of text, for a terminal and an agent: a split's branches indented under
    it, a stretch with no split on one line, each path ending in how its runs fared. A form of a
    call drawn apart says in brackets what tells it from the others. Runs that each went their
    own way from one place are counted in one line, not followed: a run's page has its steps."""
    nodes = tree["nodes"]

    def fared(rate_json: dict[str, Any] | None) -> str:
        if not rate_json:
            return "no verdicts"
        return f"passed {rate_json['successes']}/{rate_json['total']}"

    def name(node: dict[str, Any]) -> str:
        label = str(node["label"])
        if node["apart"] and node["hint"]:
            label = f"{label} [{node['hint']}]"
        return label

    lines: list[str] = []

    def walk(node: dict[str, Any], indent: int) -> None:
        # A stretch with no split, on one line: each move, its runs said where they change.
        parts = [f"{name(node)} ×{len(node['runs'])}"]
        at = node
        while len(at["children"]) == 1 and not at["ends"]:
            at = nodes[at["children"][0]]
            parts.append(name(at))
        text = "  " * indent + " → ".join(parts)
        children = [nodes[child] for child in at["children"]]
        if not children:
            lines.append(f"{text}  — {fared(at['passed'])}")
            return
        lines.append(text)
        shared = [child for child in children if len(child["runs"]) > 1]
        alone = [child for child in children if len(child["runs"]) == 1]
        for child in shared:
            walk(child, indent + 1)
        if len(alone) == 1:
            walk(alone[0], indent + 1)
        elif alone:
            passed = sum(1 for child in alone if (child["passed"] or {}).get("successes"))
            judged = sum(1 for child in alone if child["passed"])
            moves = sorted({name(child) for child in alone})
            went = ", ".join(moves[:4]) + (", …" if len(moves) > 4 else "")
            how = f"passed {passed}/{judged}" if judged else "no verdicts"
            lines.append("  " * (indent + 1) + f"{len(alone)} runs, each its own way ({went})  — {how}")
        if at["ends"]:
            lines.append("  " * (indent + 1) + f"ended here ×{len(at['ends'])}  — {fared(at['ended'])}")

    for child in nodes[0]["children"]:
        walk(nodes[child], 0)
    if len(lines) > most:
        lines = lines[:most] + [f"… {len(lines) - most} more line(s)"]
    return lines


def turning_text(turning: dict[str, Any] | None) -> str | None:
    """The turning point in a sentence, as a terminal and an agent are given it."""
    if not turning:
        return None
    if turning["stopped"]:
        did = "ended there"
    elif turning["kind"] == "text":
        did = "replied"
    elif turning["kind"] == "tool_call":
        did = f"called {turning['label']}" + (f" with {turning['hint']}" if turning["hint"] else "")
    else:
        did = MOVE_DID.get(turning["kind"], f"had {turning['label']}")
    after = f"after {turning['after']}, " if turning["after"] else ""
    # As the lines beside it say a p (`p_short`): two decimals, or its first significant figure.
    p = "< 0.0001" if turning["p"] < 0.0001 else f"= {p_short(turning['p'])}"
    return (f"step {turning['step']}: {after}runs that {did} passed {turning['passed']}/{turning['total']}, "
            f"the rest {turning['others_passed']}/{turning['others_total']} (Fisher p {p})")
