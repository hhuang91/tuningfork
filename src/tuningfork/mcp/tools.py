"""The tools an agent is offered: what each is for, its arguments, and what it answers.

Reads answer from the recordings, in this process — no proxy needed. Each `start_*` tool
is a job kind from the catalog (`mcp.schema`), sent to a proxy's job runner; `server.py`
starts it and waits. Every answer is a compact view (`ops.compact`) opened by sentences
that say it statistics first: an agent should never mistake noise for an improvement.

Nothing here needs the MCP SDK: the registry and its handlers are plain Python, and
`server.py` puts them on the wire.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tuningfork.analysis.stats import p_short
from tuningfork.ops import compact
from tuningfork.ops.catalog import RULE_LIST, Kind
from tuningfork.ops.common import OperationError, params_schema, parse_params

STEP_HELP = ("The step: its index, or what the run did there — a tool's name for its first call "
             "(write_file|execute_code: whichever came first), changed:<path> for the call that first changed "
             "that file, made:<words> for the call that first made a finding of the check.")
RUN_HELP = "The run: its id, an 8-character prefix of it, or its label."
#: A job id or a prefix of one: hexadecimal, as every id tuningfork makes.
JOB_ID = re.compile(r"[0-9a-f]{4,32}")
#: Characters of a file, or of one message of a step, an answer shows at a time; a longer one is
#: read page by page from `offset`. A cut with no way past it hid the part of a 12,000-character
#: file the check failed (findings §31.57: the legend began at character 9,407).
FILE_PAGE = 8000


def page(text: str, offset: int, data: dict[str, Any] | None = None) -> tuple[str, dict[str, Any], str]:
    """One page of `text` from `offset`: the page, where it lies (characters from and to, of how
    many, and the offset of the next page), and that said in words. `FILE_PAGE` characters, fewer
    where escaping would push the answer beside `data` past what an answer may hold — trimmed to
    fit, the page itself would be cut to a few hundred characters (`compact.fit`)."""
    if offset > len(text):
        raise OperationError(f"offset {offset} is past the end: the text has {len(text)} characters",
                             code="bad_params", field="offset")
    limit = FILE_PAGE
    while True:
        end = min(len(text), offset + limit)
        found = {"offset": offset, "to": end, "chars": len(text), "next": end if end < len(text) else None}
        if limit <= 500 or compact.size({**(data or {}), "text": text[offset:end], "page": found}) <= compact.BUDGET - 500:
            break
        limit = limit * 3 // 4
    words = (f"characters {offset}–{end} of {len(text)}"
             + (f"; the next page: offset={end}" if end < len(text) else (" (the end)" if offset else "")))
    return text[offset:end], found, words


def _help(text: str, **extra: Any) -> dict[str, Any]:
    return {"help": text, **extra}


@dataclass(slots=True)
class Answer:
    """What a tool answers: JSON, and the sentences a reader should see first."""

    data: dict[str, Any]
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Context:
    """What a tool can use: the recordings, and where work is sent."""

    store: Any
    jobs: Any  # mcp.backend.Jobs
    max_wait: int = 120
    #: The toolset the agent was given: the refine loop's hints lead elsewhere than the
    #: evidence loop's.
    toolset: str = "evidence"


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    title: str
    description: str
    #: Its arguments, strictly parsed (a dataclass), for every tool but `start_*`.
    args: type | None = None
    #: A read, answered in a worker thread: (context, parsed arguments) -> Answer.
    read: Callable[[Context, Any], Answer] | None = None
    #: A write that calls no model, in a worker thread: (context, arguments, who) -> Answer.
    act: Callable[[Context, Any, dict[str, Any]], Answer] | None = None
    #: The job kind a `start_*` tool starts.
    kind: Kind | None = None
    #: Calls the model, runs nothing on the machine — or (`machine`) runs a harness or check.
    machine: bool = False
    #: Changes something: starts or stops work.
    acts: bool = False

    def annotations(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "readOnlyHint": not self.acts,
            "destructiveHint": False,
            "idempotentHint": not self.acts or self.kind is not None,
            "openWorldHint": self.machine,
        }


def read_args(cls: type, arguments: dict[str, Any] | None) -> Any:
    """A tool's arguments, strictly: unknown names and wrong types refused, by field. A
    model writes a step as 3; the parser takes "3"."""
    data = dict(arguments or {})
    if isinstance(data.get("step"), int) and not isinstance(data.get("step"), bool):
        data["step"] = str(data["step"])
    return parse_params(cls, data)


def args_schema(cls: type) -> dict[str, Any]:
    schema = params_schema(cls)
    if "step" in schema["properties"]:
        schema["properties"]["step"] = {"type": ["integer", "string"], "description": STEP_HELP}
    return schema


# --- arguments ---------------------------------------------------------------------------------


@dataclass
class NoArgs:
    pass


@dataclass
class RunsArgs:
    task: str | None = field(default=None, metadata=_help("Only this task's runs, by its id."))
    called: str | None = field(default=None, metadata=_help("Only runs whose model called this tool somewhere."))
    at_step: int | None = field(default=None, metadata=_help("With `called`: only runs that called it at this step.", minimum=0))
    limit: int = field(default=20, metadata=_help("How many, newest first.", minimum=1, maximum=100))


@dataclass
class RunArgs:
    run: str = field(metadata=_help(RUN_HELP))


@dataclass
class StepArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    full: bool = field(default=False, metadata=_help("Every message of the context, clipped less: much longer."))
    last: int = field(default=6, metadata=_help("How many messages before the answer to show.", minimum=1, maximum=50))
    message: str | None = field(default=None, metadata=_help(
        "One message whole, page by page: its number in the context, or \"answer\"."))
    offset: int = field(default=0, metadata=_help("With message: the character to read from.", minimum=0))


@dataclass
class DivergenceArgs:
    task: str | None = field(default=None, metadata=_help("Compare this task's latest runs."))
    runs: list[str] = field(default_factory=list, metadata=_help("Or compare these runs, by id or prefix."))
    normalizers: list[str] = field(default_factory=list, metadata=_help(
        "How behaviours are compared: tool_call_json, python_ast, json_content, or tool_name "
        "(by which tools were called, ignoring arguments). None named: the defaults."))
    limit: int = field(default=20, metadata=_help("With `task`: how many of its latest runs.", minimum=2, maximum=200))
    all_steps: bool = field(default=False, metadata=_help("Every step, not only where the runs split."))


@dataclass
class TallyArgs:
    task: str | None = field(default=None, metadata=_help("A task: its failed runs, by condition."))
    screen: str | None = field(default=None, metadata=_help("Or a screen run to the end (any arm's id): each arm's failed forks."))


@dataclass
class BehavioursArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    normalizers: list[str] = field(default_factory=list, metadata=_help("How behaviours are compared; none named: the defaults."))
    target_match: str | None = field(default=None, metadata=_help(
        "A regular expression over each behaviour's label: which ones a target would count."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "Or a target described: conditions that must all hold, each {if: calls | not_calls | contains | "
        "not_contains, value}.", schema=RULE_LIST))


@dataclass
class WhereArgs:
    task: str | None = field(default=None, metadata=_help("The task: asked of its own runs."))
    screen: str | None = field(default=None, metadata=_help("Or a screen run to the end: asked of its forks, arm by arm."))
    target_match: str | None = field(default=None, metadata=_help(
        "Of the model's answers: a regular expression over what a step did — a call's name and arguments, a "
        "script's code, a reply's words."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "Or the answer described: conditions that must all hold, each {if, value}.", schema=RULE_LIST))
    path: str | None = field(default=None, metadata=_help(
        "Of what the runs wrote: a file in the workspace, by path or glob (clean.csv, *.json)."))
    contains: str | None = field(default=None, metadata=_help(
        "With path: a regular expression its text must match (^ $ at each line's ends); none: it exists."))
    at_least: int = field(default=1, metadata=_help("With contains: the matches it takes.", minimum=1))


@dataclass
class ExperimentsArgs:
    run: str | None = field(default=None, metadata=_help("Only experiments at this run's steps."))
    limit: int = field(default=30, metadata=_help("How many, newest first.", minimum=1, maximum=200))


@dataclass
class ExperimentArgs:
    experiment: str = field(metadata=_help("The experiment: its id or an 8-character prefix."))


@dataclass
class TokensArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    positions: list[int] = field(default_factory=list, metadata=_help("Token positions to show in full, as the hotspots number them."))
    hotspots: int = field(default=10, metadata=_help("How many of the least certain tokens to show.", minimum=1, maximum=50))
    stream: str | None = field(default=None, metadata=_help("A sample's token stream, by its full id. Default: the answer's own."))


@dataclass
class ScenariosArgs:
    name: str | None = field(default=None, metadata=_help("One scenario in full, by name."))


@dataclass
class RefinementArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    target: str | None = field(default=None, metadata=_help("What the model should do here: a behaviour's letter from this step's latest resample."))
    target_match: str | None = field(default=None, metadata=_help("Or a regular expression over what the model did."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "Or conditions in words that must all hold, each {if: calls | not_calls | contains | not_contains, value}.",
        schema=RULE_LIST))
    unseen_ok: bool = field(default=False, metadata=_help(
        "The target is behaviour no recorded answer at this step shows yet — new behaviour the edit should "
        "bring about — and that is intended. Without it, such a target is refused: the step is usually wrong."))
    target_check: bool = field(default=False, metadata=_help(
        "Or the task's own check as the target, for a failure only the check can see (a computed property "
        "of what the run wrote, which no condition or pattern expresses): its screens run forks to the end "
        "(start_screen_to_end), each sample a harness run — costlier than a screen at the step."))


@dataclass
class FilesArgs:
    run: str = field(metadata=_help(RUN_HELP))
    path: str | None = field(default=None, metadata=_help("One file to read, by its path in the workspace; none: the list of files."))
    where: str = field(default="end", metadata=_help(
        "At the run's end (the default), its start, or before a step (its number, a tool name, or "
        "changed:<path>)."))
    diff: bool = field(default=False, metadata=_help(
        "What changed, as a diff: at the end, since the run started; before a step, since the step before. "
        "With path, that file only."))
    offset: int = field(default=0, metadata=_help("With path or diff: the character to read from.", minimum=0))


@dataclass
class RefinementRef:
    refinement: str = field(metadata=_help("The refinement: its id or an 8-character prefix."))
    evidence: bool = field(default=True, metadata=_help("With the packet of evidence a proposal should cite."))


@dataclass
class ProposeEditArgs:
    refinement: str = field(metadata=_help("The refinement: its id or an 8-character prefix."))
    edits: list[dict[str, Any]] = field(default_factory=list, metadata=_help(
        "1 to 3 edits, each naming lines by number as show_skill prints them: {line, new} replaces that line "
        "with new, the line as it should read after the change; {line, through, new} replaces lines line..through; "
        "{after, insert} adds lines after that line. Or by text: {old, new}, old copied exactly from show_skill "
        "and occurring once, new the same lines as they should read after the change. new and insert are what "
        "the skill should say, never a copy of what it says.",
        schema={"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "properties": {
                "line": {"type": "integer", "description": "a line number, as show_skill prints them"},
                "through": {"type": "integer", "description": "with line: the last line replaced"},
                "new": {"type": "string", "description": "the line(s) as they should read after the change"},
                "after": {"type": "integer", "description": "the line to add lines after"},
                "insert": {"type": "string", "description": "with after: the lines to add"},
                "old": {"type": "string", "description": "text copied exactly from show_skill, occurring once"}},
            "additionalProperties": False}}))
    rationale: str = field(default="", metadata=_help("Why: cite the evidence in the packet."))
    packet: str | None = field(default=None, metadata=_help("The packet the edit came from, as show_refinement named it."))


@dataclass
class JobArgs:
    job: str = field(metadata=_help("The job: its id or an 8-character prefix."))
    wait_seconds: int = field(default=0, metadata=_help("How long to wait for it to end.", minimum=0, maximum=120))
    after: int | None = field(default=None, metadata=_help("Also its progress events after this sequence number.", minimum=0))


@dataclass
class JobsArgs:
    limit: int = field(default=20, metadata=_help("How many, newest first, plus all under way.", minimum=1, maximum=100))


@dataclass
class CancelArgs:
    job: str = field(metadata=_help("The job: its id or an 8-character prefix."))


# --- what an agent is never shown ------------------------------------------------------------------
#
# Runs of a held-out task judge an edit. A person's validation never records them, but runs
# recorded before it existed, or by a person's explicit --include-held-out, may be in the
# recordings: every read below leaves them out, and refuses to open one by name.


def hidden_tasks(context: Context) -> set[str]:
    return context.store.held_out_tasks()


def refuse_hidden(context: Context, run: Any, hidden: set[str] | None = None) -> Any:
    """The run, unless it is of a held-out task."""
    if run is not None and run.task_id in (hidden_tasks(context) if hidden is None else hidden):
        raise OperationError(
            f"run {run.id[:8]} is of a held-out task: its runs are the person's evidence for judging "
            "an edit, never an agent's",
            code="held_out", field="run",
        )
    return run


def visible_run(context: Context, reference: str) -> Any:
    from tuningfork.ops.experiments import resolve_run

    return refuse_hidden(context, resolve_run(context.store, reference))


def hidden_run_ids(context: Context, run_ids: Any, hidden: set[str]) -> set[str]:
    found = set()
    for run_id in {run_id for run_id in run_ids if run_id}:
        run = context.store.get_run(run_id)
        if run is not None and run.task_id in hidden:
            found.add(run_id)
    return found


def without_held_out_evals(spec: Any) -> Any:
    """A spec as an agent sees it: its held-out evals left out, but for a count."""
    if not isinstance(spec, dict) or not isinstance(spec.get("evals"), list):
        return spec
    kept = [item for item in spec["evals"]
            if not (isinstance(item, dict) and (item.get("tuningfork") or {}).get("held_out") is True)]
    shown = {**spec, "evals": kept}
    if len(kept) != len(spec["evals"]):
        shown["held_out_evals"] = len(spec["evals"]) - len(kept)
    return shown


def check_said(result: dict[str, Any] | None) -> str | None:
    """The first line a failed run's check printed; None for a run that did not fail."""
    if not result or result.get("passed") is not False:
        return None
    text = str(result.get("check_output") or "")
    cut = bool(result.get("check_output_cut"))
    if cut and "\n" in text:
        text = text.partition("\n")[2]  # its start cut, the first line kept is the end of a longer one
    text = text.strip()
    if not text:
        return "(its check printed nothing)"
    return ("… " if cut else "") + text.splitlines()[0][:200]


#: The toolsets that are the refine loop: its reads only, and its hints.
REFINE_TOOLSETS = ("refine", "refine-to-end")
#: Answers seen at a step before a target every one of them satisfies is refused: fewer say
#: little about what the step does.
UNIVERSAL_MIN = 5


def refining(context: Context) -> bool:
    """Whether the agent was given the refine loop: its hints lead to it."""
    return context.toolset in (*REFINE_TOOLSETS, "all")


def to_end_offered(context: Context) -> bool:
    """Whether the agent may screen to the end: forks through the harness, the check deciding."""
    return context.toolset in ("refine-to-end", "all")


def target_advice(context: Context, run: str) -> str:
    """What a target is, for an agent whose target was refused: a difference between the
    passing runs' answers and the failing runs', with what the failing runs' check said."""
    found = context.store.get_run(run)
    if found is None:
        found = next((each for each in context.store.runs(limit=500) if each.id.startswith(run)), None)
    complaints: list[str] = []
    if found is not None and found.task_id:
        for other in context.store.runs(limit=100, task_id=found.task_id):
            said = check_said(other.result)
            if said and said not in complaints:
                complaints.append(said)
            if len(complaints) == 2:
                break
    advice = "a target names what the passing runs' answers have that the failing runs' lack"
    if complaints:
        advice += " — their check said: " + " | ".join(repr(said) for said in complaints)
    advice += (". step_behaviours tries a target_match or target_rules against the recorded answers, split "
               "by verdict, before a screen is spent; find_divergence names the step whose behaviour predicts "
               "the verdict, and the tool that goes with passing when one does — a tool the model calls is a "
               "behaviour a target can name ({\"if\": \"calls\", \"value\": \"write_file\"}), at the step "
               "where the passing runs called it")
    if to_end_offered(context):
        advice += ("; for a failure only the check can see, start_refinement with target_check and screen it "
                   "to the end")
    return advice


def choice_lines(context: Context, seen: dict[str, Any]) -> list[str]:
    """The model's choice of tool at the step as a number, where the recorded answers were
    scored — the chance of each tool, exact for each prompt — or how to get it. Only for a
    step where some recorded answer called a tool: a reply in text chose none."""
    if not seen.get("tools") or not any(entry.get("kind") == "tool_call" for entry in seen.get("behaviours") or []):
        return []
    choices = seen.get("choices") or {}
    if choices.get("scored"):
        mean = compact.choice_text(choices.get("chance"))
        line = (f"the model's own chance of each tool at this step, from its scored tokens in {choices['scored']} "
                f"of the {choices.get('of', 0)} recorded run(s) compared: {mean}")
        likeliest = max((float(p) for p in (choices.get("chance") or {}).values()), default=0.0)
        if likeliest < 0.75:
            line += " — a choice this open is one a few words of the skill can move"
        else:
            line += " — a choice this settled takes more than a few words to move"
        return [line]
    from tuningfork.mcp.schema import offered

    how = ("start_tokens at this step scores it: one request"
           if any(kind.name == "tokens" for kind in offered(context.toolset))
           else "the person can score it: `tuningfork tokens <run> <step> --score`")
    return [f"not scored at this step: the model's own chance of each tool is not known here — {how}"]


def split_lines(split: dict[str, Any]) -> list[str]:
    """What a target's split by verdict says of it, before a screen is spent."""
    if not split:
        return []
    passing, failing = split.get("passing") or {}, split.get("failing") or {}
    line = (f"it counts the answers of {passing.get('counted', 0)} of {passing.get('of', 0)} passing run(s) and "
            f"{failing.get('counted', 0)} of {failing.get('of', 0)} failing run(s)")
    kind = split.get("kind")
    if kind == "universal":
        line += (": every recorded answer satisfies it, so a screen could only show an edit lowering it — name "
                 "what the passing runs' answers have that the failing runs' lack")
    elif kind == "unseen":
        line += ": no recorded answer does — if the edit should bring it about, start_refinement with unseen_ok"
    elif split.get("separates") is True:
        line += ": it separates them, a target a screen can move"
    elif split.get("separates") is False:
        line += (": it does not go with passing — a target that counts failing runs' answers as readily as "
                 "passing runs' measures the wrong thing, and start_refinement refuses it")
    return [line]


# --- reads -------------------------------------------------------------------------------------


def list_tasks(context: Context, args: NoArgs) -> Answer:
    from tuningfork.ops import inspect

    hidden = hidden_tasks(context)
    found = [compact.task(summary) for summary in inspect.tasks(context.store)]
    found = [summary for summary in found if summary.get("task_id") not in hidden]
    if not found:
        return Answer({"tasks": []}, [
            "no runs are recorded yet: `tuningfork batch <spec> -n 5`, or the web UI's Run a task, makes some",
        ])
    lines = [line for summary in found[:10] for line in compact.task_lines(summary)]
    if len(found) > 10:
        lines.append(f"… and {len(found) - 10} more task(s) below")
    return Answer({"tasks": compact.capped(found, 30)}, lines)


def list_runs(context: Context, args: RunsArgs) -> Answer:
    from tuningfork.ops.inspect import runs_calling
    from tuningfork.ops.timeline import run_json

    if args.at_step is not None and not args.called:
        raise OperationError("at_step narrows `called`: name the tool too", code="bad_params", field="at_step")
    hidden = hidden_tasks(context)
    if args.task in hidden:
        raise OperationError(f"{args.task} is a held-out task: its runs are the person's evidence, never an agent's",
                             code="held_out", field="task")
    if args.called:
        runs = runs_calling(context.store, args.called, task=args.task, step=args.at_step, limit=args.limit)
    else:
        runs = context.store.runs(limit=args.limit, task_id=args.task)
    views = [compact.run(run_json(found)) for found in runs if found.task_id not in hidden]
    judged = [view for view in views if (view.get("result") or {}).get("passed") is not None]
    passed = sum(1 for view in judged if view["result"]["passed"])
    lines = [f"{len(views)} run(s), newest first" + (f"; of those judged by their check, {passed}/{len(judged)} passed" if judged else "")]
    failed = [(compact.short(found.id), check_said(found.result)) for found in runs
              if found.task_id not in hidden and check_said(found.result)]
    for run_id, said in failed[:5]:
        lines.append(f"  failed {run_id}: {said}")
    if len(failed) > 5:
        lines.append(f"  … and {len(failed) - 5} more failed run(s); show_run says what each check said")
    return Answer({"runs": views}, lines)


def show_run(context: Context, args: RunArgs) -> Answer:
    from tuningfork.ops import inspect

    found = visible_run(context, args.run)
    view = compact.run_detail(inspect.run_detail(context.store, found.id))
    # Where the spec is would lead an agent with file access to the evals held out from it.
    if view.get("launch"):
        view["launch"].pop("spec", None)
    run = view["run"]
    result = found.result or {}
    verdict = {True: "passed its check", False: "failed its check"}.get(result.get("passed"), "has no verdict")
    steps = view["steps"]
    count = steps["total"] if isinstance(steps, dict) else len(steps)
    lines = [f"run {run['id']} of {run.get('task_id') or 'no task'}: {count} step(s); it {verdict}"]
    if result.get("passed") is False:
        output, cut = str(result.get("check_output") or ""), result.get("check_output_cut")
        if cut:
            output = output.partition("\n")[2]  # the first line kept is the end of a longer one
        said = [line[:160] for line in output.strip().splitlines()[:3]]
        lines.append("its check said: " + " | ".join(said) if said else "its check printed nothing")
        if cut:
            from tuningfork.specs import CHECK_KEPT

            lines.append(f"its check printed {cut:,} characters more before that, not kept: a verdict keeps the last "
                         f"{CHECK_KEPT:,}, and only they are tallied")
    shown = steps["shown"] if isinstance(steps, dict) else steps
    # Which calls changed the files, and what they ran: where a run did what its check saw. Its
    # last call was named here once, as the step to refine, and an agent refined a run's last
    # call — a side experiment — two steps after the call that made the failure (findings §31.42).
    changed = [step for step in shown if step.get("changed")]
    if changed:
        lines.append("its calls that changed files: " + "; ".join(
            f"step {step.get('idx')} ({', '.join(step['changed'][:3])})" for step in changed[:8]
        ) + ("; …" if len(changed) > 8 else ""))
    if refining(context):
        lines.append("each step's `ran` says what its calls ran; show_files shows the files at the start, before "
                     "any step and at the end (diff: what changed). The step to refine is the first whose call "
                     "made what the check complains about — not the last call, nor where tool choices first "
                     "differed: a screen replays everything before its step as recorded, the files included")
    return Answer(view, lines)


def show_files(context: Context, args: FilesArgs) -> Answer:
    from tuningfork.ops.experiments import CHANGED, MADE
    from tuningfork.ops.inspect import workspace_at, workspace_patch
    from tuningfork.snapshots import Snapshots

    found = visible_run(context, args.run)
    named = args.where.startswith((CHANGED, MADE))
    if args.where not in ("start", "end") and (not args.where or (not named and any(ch.isspace() for ch in args.where))):
        # A 26B wrote 'before step 4', 'at step 1', 'after step 5' eighteen times running.
        raise OperationError(
            f"where is 'end' (the default), 'start', or a step — its number, or the name of the tool the model "
            f"called there, or changed:<path> — for the workspace before that step: not {args.where!r}",
            code="bad_params", field="where",
        )
    view = workspace_at(context.store, found.id, args.where)
    statuses = {"A": "added", "M": "changed", "D": "deleted"}
    data: dict[str, Any] = {
        "run": compact.short(found.id), "where": view.where,
        "files": compact.capped([{"path": entry.path, "size": entry.size} for entry in view.files], 50),
        "changed": [{"status": status, "path": path} for status, path in view.changes][:50],
        "changed_since": view.since,
    }
    names = ", ".join(entry.path for entry in view.files[:12]) + (", …" if len(view.files) > 12 else "")
    lines = [f"run {found.id[:8]} {view.where}: {len(view.files)} file(s)" + (f": {names}" if names else "")]
    if view.since:
        lines.append(f"since {view.since}: " + (", ".join(
            f"{path} {statuses.get(status, status)}" for status, path in view.changes[:12]
        ) if view.changes else "nothing changed"))
    if args.diff:
        patch = workspace_patch(context.store, view, args.path) if view.before else ""
        shown, where, words = page(patch, min(args.offset, len(patch)), data)
        data["diff"] = shown
        if patch:
            data["page"] = where
        lines.append((f"what changed since {view.since}" + (f" in {args.path}" if args.path else "")
                      + (f", {words}: the page is `diff`" + (f" — read on with offset={where['next']}"
                                                             if where["next"] else "") if patch else ": nothing"))
                     if view.since else "nothing to compare: this is how the run started")
    if args.path and not args.diff:
        try:
            raw = Snapshots(context.store.home).read(view.commit, args.path)
        except Exception:  # noqa: BLE001 - git says no such path in many words
            raise OperationError(
                f"no file {args.path!r} in run {found.id[:8]}'s workspace {view.where}",
                hint="show_files without a path lists them", code="no_such_file", field="path",
            ) from None
        text = raw.decode("utf-8", errors="replace")
        data.update({"path": args.path, "bytes": len(raw)})
        shown, where, words = page(text, args.offset, data)
        data.update({"text": shown, "page": where})
        lines.append(f"{args.path} ({len(raw)} bytes), {words}: the page is `text`"
                     + (f" — read on with show_files(run, path, offset={where['next']})" if where["next"] else ""))
    elif refining(context) and not args.path:
        lines.append("read one with `path`; diff shows what changed. What a run changed beside what it started with "
                     "(where: start) is what it did; a failed run's file beside a passing run's is the difference "
                     "the check saw")
    return Answer(data, lines)


def show_step(context: Context, args: StepArgs) -> Answer:
    from tuningfork.ops.experiments import step_index
    from tuningfork.ops.timeline import step_detail

    run = visible_run(context, args.run)
    index = step_index(context.store, run, args.step)
    detail = step_detail(context.store, run.id, index).to_json()
    if args.message is not None:
        return step_message(run, index, detail, args.message, args.offset)
    view = compact.step(detail, full=args.full, last=args.last)
    lines = [f"step {index} of run {run.id[:8]}: {view['context']['messages']} message(s) in its context;"
             f" the model did: {(view.get('did') or {}).get('label')}"]
    for span in view["context"]["skills"]:
        where = f"{span['file']}, from line {span['first_line']}" if span.get("file") else "not matched to a file"
        lines.append(f"  skill {span['name']!r}: {span['lines']} line(s) in message {span['message_index']} ({where})")
    if view["context"]["skills"]:
        lines.append("the skill's text, line by line: show_skill(run) — it is not a file on your machine")
    lines.append(f"each message is clipped: show_step with message (its number, {view['context']['shown_from']} the first "
                 "shown, or \"answer\") reads one whole, a page at a time")
    return Answer(view, lines)


def message_text(found: dict[str, Any]) -> str:
    """A message whole: its text, then each call it made with its arguments as sent."""
    parts = [compact._content(found.get("content"))]
    for call in found.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        parts.append(f"[call {function.get('name')}] {arguments}")
    return "\n".join(part for part in parts if part)


def step_message(run: Any, index: int, detail: dict[str, Any], which: str, offset: int) -> Answer:
    """One message of a step whole, a page at a time: a context message by its number, or the answer."""
    view = detail.get("view") or []
    if str(which).strip().lower() == "answer":
        response = detail.get("response") or {}
        found = ((response.get("choices") or [{}])[0].get("message")) or {}
        role, label = "assistant", "its answer"
    else:
        try:
            number = int(str(which).strip())
        except ValueError:
            raise OperationError(f"message is a number from 0 to {len(view) - 1}, or \"answer\": not {which!r}",
                                 code="bad_params", field="message") from None
        if not 0 <= number < len(view):
            raise OperationError(f"step {index} has {len(view)} message(s) in its context: 0 to {len(view) - 1}",
                                 code="bad_params", field="message")
        found = view[number]
        role, label = found.get("role"), f"message {number}"
    data: dict[str, Any] = {"run": compact.short(run.id), "idx": index, "message": which, "role": role}
    shown, where, words = page(message_text(found), offset, data)
    data.update({"text": shown, "page": where})
    lines = [f"step {index} of run {run.id[:8]}, {label} ({role}), {words}: the page is `text`"
             + (f" — read on with offset={where['next']}" if where["next"] else "")]
    return Answer(data, lines)


def find_divergence(context: Context, args: DivergenceArgs) -> Answer:
    from tuningfork.ops import inspect

    hidden = hidden_tasks(context)
    if args.task in hidden:
        raise OperationError(f"{args.task} is a held-out task: its runs are the person's evidence, never an agent's",
                             code="held_out", field="task")
    for reference in args.runs or []:
        visible_run(context, reference)
    report = inspect.divergence_report(
        context.store, refs=args.runs, task=args.task, normalizers=args.normalizers, limit=args.limit
    )
    split = report["verdict_split"]
    view = compact.divergence(report, all_steps=args.all_steps)
    if view.get("task_id") in hidden or view.get("task") in hidden:
        raise OperationError("those runs are of a held-out task: the person's evidence, never an agent's",
                             code="held_out", field="task")
    lines = compact.divergence_lines(view)
    best = split.get("best")
    if best is not None:
        view["verdict_split"] = best
        where = "this split" if best["index"] == view.get("divergence_index") else f"step {best['index']}"
        lines.append(
            f"the behaviour that best predicts the verdict is at {where}: {best['label']} — those runs passed "
            f"{best['passed']}/{best['total']}, the others {best['others_passed']}/{best['others_total']} "
            f"(Fisher p = {p_short(best['p'])})"
        )
    by_tool = (report.get("tool_split") or {}).get("best")
    if by_tool is not None:
        view["tool_split"] = by_tool
        called, never, tool = by_tool["called"], by_tool["never"], by_tool["tool"]
        line = (f"by the tool a run used, at whatever step: runs that called {tool} passed "
                f"{called['passed']}/{called['total']}, runs that never did {never['passed']}/{never['total']} "
                f"(Fisher p = {p_short(by_tool['p'])})")
        if by_tool.get("with_passing"):
            line += (f" — the tool goes with passing, and is a behaviour a target can name: start_refinement at "
                     f"step \"{tool}\" of a run that called it (its first such step), with target_rules "
                     f"[{{\"if\": \"calls\", \"value\": \"{tool}\"}}]; a skill sentence naming the tool to use "
                     f"is the edit to try")
        else:
            line += f" — the tool goes with failing: a target names what the passing runs did instead"
        lines.append(line)
    if best is None and by_tool is None and split.get("judged", 0) >= 4 and view.get("divergence_index") is not None:
        lines.append("no step's behaviour predicts the verdict (every split's p is 0.2 or more), nor does the tool a "
                     "run used: the difference is in what the runs wrote — show_run of a failed run, then "
                     "show_files beside a passing run's; where_it_holds then asks every run's files at once "
                     "(path, contains) and names the call that made the difference"
                     + ("; a target names what the passing runs' writes contain and the failing runs' lack, and "
                        "where only the check can tell them apart, start_refinement with target_check and screen "
                        "to the end" if to_end_offered(context) else ""))
    if view.get("divergence_index") is not None:
        if refining(context):
            lines.append("next: show_run of a failed run (what its check said) and show_skill; then start_refinement "
                         "at the step to change, with a target a passing run's answer satisfies")
        else:
            lines.append("next: show_step at that step, then start_resample there to measure each behaviour")
    return Answer(view, lines)


def tally_failures(context: Context, args: TallyArgs) -> Answer:
    from tuningfork.ops.tally import tally_screen, tally_task

    if (args.task is None) == (args.screen is None):
        raise OperationError("give a task, or a screen run to the end — one of them", code="bad_params", field="task")
    hidden = hidden_tasks(context)
    found = (tally_task(context.store, args.task, hidden=hidden) if args.task
             else tally_screen(context.store, args.screen, hidden=hidden))
    report = found.to_json()
    view = compact.tally(report)
    lines = compact.tally_lines(view)
    failed = [group for group in report["groups"] if group.get("failed")]
    if failed and all(group.get("silent") == group.get("failed") for group in failed):
        # A check that prints only its verdict: nothing here says what went wrong, or where.
        lines.append("the check printed nothing to count, only its verdict: where_it_holds asks one question "
                     "of every run — what the passing runs' files have that the failing runs' lack, or what "
                     "their answers did — and says which call made the difference")
    if refining(context):
        from tuningfork.ops.tally import finding_words, shared_mistakes

        shared = shared_mistakes(report) if found.scope == "task" else []
        if shared:
            # Every run made it, and the passes are later repairs: comparing passing with failing
            # runs would point at the repair (findings §31.42).
            toward = ("toward the task's check (target_check), and screen it to the end" if to_end_offered(context)
                      else "toward a behaviour none of them showed there (unseen_ok)")
            lines.append(f"every run made «{shared[0]['finding']}» — the passing ones too, which repaired it later — so "
                         "no passing run's answer shows the fix: start the refinement at the call that made it "
                         f"(step made:{finding_words(shared[0]['finding'])} names each run's own), {toward}")
        if found.scope == "task":
            lines.append("aim at the most common finding first, not the first one read: show_run of a run that said it, "
                         "and show_files of what it wrote")
        else:
            lines.append("what the original's forks said and a proposal's did not is what the proposal fixed; what both "
                         "still say is what the next edit, built on that proposal's, should aim at")
    return Answer(view, lines)


def step_behaviours(context: Context, args: BehavioursArgs) -> Answer:
    from tuningfork.ops.experiments import step_targets
    from tuningfork.ops.targets import seen_behaviours

    visible_run(context, args.run)
    seen = seen_behaviours(
        context.store, args.run, args.step, normalizers=args.normalizers,
        target_match=args.target_match, target_rules=args.target_rules or None,
    ).to_json()
    try:
        targets = step_targets(context.store, args.run, args.step).to_json()
    except OperationError:
        targets = None
    view = compact.behaviours(seen, targets)
    found = seen.get("behaviours") or []
    lines = [f"{len(found)} distinct behaviour(s) seen at step {seen.get('step')} (in runs, forks and samples)"]
    aligned = seen.get("aligned")
    if aligned:
        from tuningfork.ops.experiments import CHANGED, MADE, not_done

        named = aligned[0] if len(aligned) == 1 and aligned[0].startswith((CHANGED, MADE)) else None
        if named and named.startswith(CHANGED):
            line = f"other runs compared at each one's own call that first changed {named[len(CHANGED):] or 'a file'}"
        elif named:
            line = f"other runs compared at each one's own call that first made a finding with {named[len(MADE):]!r}"
        else:
            line = f"other runs compared at each one's own first call of {' or '.join(aligned)}"
        missing = seen.get("not_aligned") or {}
        if missing:
            fared = ", ".join(f"{count} {verdict}" for verdict, count in sorted(missing.items()))
            line += (f"; {sum(missing.values())} run(s) of the task {not_done(named)} ({fared})" if named else
                     f"; {sum(missing.values())} run(s) of the task never called one ({fared}) — a step named by every "
                     "tool the runs wrote with, joined by |, compares them too")
        lines.append(line)
    elif seen.get("step") is not None and (seen.get("judged") or {}):
        lines.append("other runs compared at the same step index, which may be another point in a run that took "
                     "another path: a step named by its tool (write_file, or write_file|execute_code) compares each "
                     "run at its own such step")
    judged = seen.get("judged") or {}
    if judged:
        lines.append(f"of the recorded runs with a verdict, {judged.get('passed', 0)} passed and "
                     f"{judged.get('failed', 0)} failed")
    lines.extend(choice_lines(context, seen))
    if seen.get("target"):
        counted = sum(1 for entry in found if entry.get("matches"))
        lines.append(f"the target {seen['target']!r} counts {counted} of them")
        lines.extend(split_lines(seen.get("split") or {}))
    if view.get("letters"):
        lines.append(f"resample {view['letters']['resample']} lettered them: an A/B can declare `target` as one of those letters")
    else:
        lines.append("no resample at this step yet: start_resample letters its behaviours, for an A/B's target")
    return Answer(view, lines)


def where_holds(context: Context, args: WhereArgs) -> Answer:
    from tuningfork.ops.where import where_it_holds

    if (args.task is None) == (args.screen is None):
        raise OperationError("give a task, or a screen run to the end — one of them", code="bad_params", field="task")
    found = where_it_holds(
        context.store, args.task, target_match=args.target_match, target_rules=args.target_rules or None,
        path=args.path, contains=args.contains, at_least=args.at_least, hidden=hidden_tasks(context),
        screen=args.screen,
    )
    view = compact.where(found.to_json())
    lines = compact.where_lines(view)
    if refining(context):
        lines.extend(where_advice(context, view))
    return Answer(view, lines)


def where_advice(context: Context, view: dict[str, Any]) -> list[str]:
    """What a refining agent can do with where a question held: of the condition with the most
    judged runs, the call to refine and toward what — or another question to ask."""
    groups = view.get("groups") or []
    if not groups:
        return []
    group = max(groups, key=lambda found: (found.get("passed") or 0) + (found.get("failed") or 0))
    kind = (view.get("probe") or {}).get("kind")
    goes = (group.get("split") or {}).get("goes_with")
    step = (group.get("reference") or {}).get("step")
    made = group.get("made") or {}
    toward_check = ("toward the task's check (target_check), screened to the end" if to_end_offered(context)
                    else f"toward what the passing runs' answers did there (step_behaviours at step \"{step}\" "
                         "tries a target)")
    if made.get("shared_mistake"):
        return [f"every run made it, the passing ones too, and undid it later, so no passing run's answer shows the "
                f"fix: start_refinement at the call that made it (step \"{step}\"), "
                + ("toward the task's check (target_check), screened to the end" if to_end_offered(context)
                   else "toward a behaviour none of them showed there (unseen_ok)")]
    if goes is None and not view.get("screen") and not any(found.get("passed") for found in groups):
        # A broken skill: no run of the task passed, and its screens' forks are the passing runs there are.
        from tuningfork.ops.where import passing_screens

        found = passing_screens(context.store, view.get("task_id") or "")
        if found:
            best = found[0]
            return [f"no run of the task passed, so nothing here can split: screen {best['screen'][:8]}'s forks passed "
                    f"{best['passed']} of {best['judged']} — where_it_holds with screen=\"{best['screen'][:8]}\" asks the "
                    "same question of its forks, a proposal's passing ones beside its failing ones, and show_files "
                    "reads a passing fork's files"]
        return ["no run of the task passed, so nothing here can split: set a failed run's files against the skill's "
                "rules and the task's criteria (show_files reads a file page by page with offset), or screen a "
                "proposal — its passing forks are then the comparison (where_it_holds with screen)"]
    if goes is None:
        return ["it does not separate passing from failing runs: ask another question — what a passing run's files "
                "have that a failing run's lack (show_files of each, with diff), or what their answers did"]
    if not step:
        return ["no one step names that call in every run: show_run of a failing run says which call it was"]
    if kind == "file" and goes == "failing":
        return [f"the failing runs' call at step \"{step}\" made it: start_refinement there, in a failing run, "
                + toward_check]
    if kind == "file":
        return [f"the passing runs' call at step \"{step}\" made it and the failing runs' same call did not: "
                f"start_refinement at step \"{step}\" of a failing run, " + toward_check]
    if goes == "passing":
        return [f"the passing runs' answers did it first at step \"{step}\": start_refinement at step \"{step}\" of "
                "a failing run with this target (step_behaviours there says how it splits at that one step)"]
    return [f"the failing runs' answers did it first at step \"{step}\": a target names what the passing runs' "
            f"answers did there instead (step_behaviours at step \"{step}\" lists them)"]


def show_skill(context: Context, args: RunArgs) -> Answer:
    from tuningfork.ops.experiments import recorded_skill
    from tuningfork.ops.variants import list_variants

    visible_run(context, args.run)
    recorded = recorded_skill(context.store, args.run).to_json()
    variants = [variant.to_json() for variant in list_variants(context.store.home, recorded.get("skill"))]
    view = compact.skill(recorded, variants)
    lines = [f"skill {recorded.get('skill')!r} as run {view['run']} was given it: {recorded.get('path') or 'no file kept'}"]
    if variants:
        lines.append(f"{len(variants)} saved variant(s): " + ", ".join(v.get("name", "?") for v in view["variants"]))
    if refining(context):
        lines.append("propose_edit names these lines by their numbers: {line, new} replaces one with the line as it "
                     "should read, {after, insert} adds lines after one; or {old, new} copies one exactly. This is "
                     "the only way to read the skill: it is not a file on your machine")
    else:
        lines.append("to test an edit: start_ab with variant_text (the whole edited file) and variant_name")
    return Answer(view, lines)


def list_experiments(context: Context, args: ExperimentsArgs) -> Answer:
    from tuningfork.ops import inspect

    if args.run:
        visible_run(context, args.run)
    listed = inspect.experiment_list(context.store, args.run, limit=args.limit)
    hidden = hidden_tasks(context)
    if hidden:
        held = hidden_run_ids(context, (entry.get("run_id") for entry in listed), hidden)
        listed = [entry for entry in listed if entry.get("run_id") not in held]
    shown = compact.experiments(listed, limit=args.limit)
    count = shown["total"] if isinstance(shown, dict) else len(shown)
    return Answer({"experiments": shown}, [f"{count} experiment(s), newest first; an A/B's two arms are one"])


def show_experiment(context: Context, args: ExperimentArgs) -> Answer:
    from tuningfork.ops.inspect import experiment_report

    report = experiment_report(context.store, args.experiment).to_json()
    if report.get("run_id"):
        refuse_hidden(context, context.store.get_run(report["run_id"]))
    view = compact.experiment(report)
    return Answer(view, compact.experiment_lines(view))


def show_tokens(context: Context, args: TokensArgs) -> Answer:
    from tuningfork.ops.tokens import step_tokens

    visible_run(context, args.run)
    found = step_tokens(context.store, args.run, args.step, stream=args.stream).to_json(hotspots=args.hotspots)
    view = compact.tokens(found, positions=args.positions, hotspots=args.hotspots)
    if not view.get("scored"):
        return Answer(view, [view["hint"]])
    spots = view.get("hotspots") or []
    lines = [f"{len(spots)} least certain token(s) of step {view.get('step')}'s answer, the runner-up's chance first"]
    for spot in spots[:5]:
        chosen = spot.get("chosen") or {}
        lines.append(f"  position {spot.get('position')} ({spot.get('part')}): {chosen.get('text')!r} at "
                     f"{chosen.get('p', 0):.0%}, runner-up {spot.get('runner_up', 0):.0%}")
    lines.append("start_forking_tokens forces alternatives there and tests which change what the model does")
    return Answer(view, lines)


def server_status(context: Context, args: NoArgs) -> Answer:
    store = context.store
    data: dict[str, Any] = {}
    lines = []
    try:
        health = context.jobs.status()
    except OperationError as error:
        health = None
        data["proxy"] = None
        lines.append(f"no proxy to start work through: {error.message}" + (f" — {error.hint}" if error.hint else ""))
    if health:
        server = health.get("server") or {}
        data["proxy"] = {key: health.get(key) for key in ("version", "upstream", "paused", "inflight", "max_inflight", "home")}
        data["server"] = {
            "id": compact.short(server.get("id")),
            "models": server.get("models"),
            "build": server.get("build"),
            "n_ctx": server.get("n_ctx"),
        }
        data["problem"] = health.get("server_problem")
        checks = store.server_checks(server.get("id")) or {}
        data["checks"] = compact.fit(checks, 6000) or None
        lines.append(f"the proxy sends model calls to {health.get('upstream')}"
                     + (" — paused" if health.get("paused") else ""))
        if health.get("server_problem"):
            lines.append(f"the server could not be read: {health['server_problem']}")
        elif server.get("models"):
            lines.append(f"it serves {', '.join(map(str, server['models']))}")
        from tuningfork.ops.connections import reminder, token_work

        # A server on the internet answering as a hosted API does: what the agent's own work can't reach.
        read = health.get("server_read")
        noted = reminder(read)
        data["reminder"] = noted
        if noted:
            lines.append(f"{noted['headline']} {noted['body']}")
        # The token work its check rules out, which start_tokens and the rest would be refused.
        ruled_out = {work: reason for work, reason in token_work(checks, read).items() if reason}
        data["cannot"] = ruled_out or None
        lines.extend(ruled_out.values())
    data["recorded_servers"] = [
        {"id": compact.short(server_id), "models": identity.get("models"), "build": identity.get("build"),
         "first_seen": compact.when(first_seen), "runs": runs}
        for server_id, identity, first_seen, runs in store.servers()[:5]
    ]
    return Answer(data, lines)


def show_scenarios(context: Context, args: ScenariosArgs) -> Answer:
    from tuningfork.ops.scenarios import list_scenarios, read_scenario

    def shown(found: dict[str, Any]) -> dict[str, Any]:
        return {**found, "spec": without_held_out_evals(found.get("spec"))} if "spec" in found else found

    if args.name:
        found = shown(read_scenario(context.store.home, args.name).to_json())
        return Answer(compact.fit(found, 16_000), [f"scenario {args.name!r}: a task spec start_batch can run"])
    hidden = hidden_tasks(context)

    def summary(found: dict[str, Any]) -> dict[str, Any]:
        # A held-out task's pass rates are its verdicts, counted (found in review).
        return {**found, "results": [result for result in found.get("results") or [] if result.get("task_id") not in hidden]}

    listed = [summary(scenario.to_json()) for scenario in list_scenarios(context.store)]
    return Answer({"scenarios": compact.fit(listed, 16_000)}, [f"{len(listed)} saved scenario(s)"])


# --- the refinement loop -----------------------------------------------------------------------------

#: Proposals an agent may make to one refinement: a screen tests a few, not a flood.
AGENT_PROPOSALS = 12


def start_refinement(context: Context, args: RefinementArgs, who: dict[str, Any]) -> Answer:
    from tuningfork.ops import refine
    from tuningfork.ops.targets import seen_behaviours

    if args.target_check:
        if args.target or args.target_match or args.target_rules:
            raise OperationError("declare one target: the task's check (target_check), or a behaviour at the step",
                                 code="bad_params", field="target")
        if not to_end_offered(context):
            raise OperationError(
                "the task's check as a target needs screens run to the end, which this toolset does not offer",
                hint="name what the passing runs' answers have that the failing runs' lack, as target_rules or "
                     "target_match (step_behaviours tries one); the person can start the proxy with "
                     "--mcp-tools refine-to-end",
                code="not_offered", field="target_check",
            )
    seen = None
    if (args.target_match or args.target_rules) and not args.unseen_ok:
        # A target nothing recorded at the step satisfies was, every time it was seen, a
        # target at the wrong step: the behaviour happened elsewhere, or is the failure itself.
        # One every recorded answer satisfies cannot show an edit's effect either: its rate
        # is already everything, and a screen could only show it falling.
        seen = seen_behaviours(
            context.store, args.run, args.step, target_match=args.target_match, target_rules=args.target_rules or None,
        ).to_json()
        entries = seen.get("behaviours") or []
        if entries and not any(entry.get("matches") for entry in entries):
            raise OperationError(
                f"no recorded answer at step {seen.get('step')} does that: the target describes none of the "
                f"{len(entries)} behaviour(s) seen there (step_behaviours lists them)",
                hint="refine at the step where a passing run did it (show_run), or, if this is new behaviour "
                     "the edit should bring about, say so with unseen_ok. " + target_advice(context, args.run),
                code="unseen_target", field="target",
            )
        recorded = int(seen.get("recorded") or 0)
        if entries and all(entry.get("matches") for entry in entries) and recorded >= UNIVERSAL_MIN:
            split = seen.get("split") or {}
            raise OperationError(
                f"every recorded answer at step {seen.get('step')} already does that: all {recorded} seen there, "
                f"from {(split.get('passing') or {}).get('of', 0)} passing and "
                f"{(split.get('failing') or {}).get('of', 0)} failing runs alike, so a screen could only show "
                "an edit lowering it",
                hint=target_advice(context, args.run),
                code="universal_target", field="target",
            )
        # One the failing runs' answers satisfy as readily as the passing runs' measures the
        # wrong thing: moving it would not move the verdict.
        split = seen.get("split") or {}
        passing, failing = split.get("passing") or {}, split.get("failing") or {}
        if (failing.get("counted", 0) and passing.get("of", 0)
                and failing["counted"] / failing["of"] >= passing.get("counted", 0) / passing["of"]):
            raise OperationError(
                f"the target does not go with passing: at step {seen.get('step')} it counts the answers of "
                f"{passing.get('counted', 0)} of {passing['of']} passing run(s) and {failing['counted']} of "
                f"{failing['of']} failing run(s), so a screen moving it would not move the verdict",
                hint=target_advice(context, args.run),
                code="target_does_not_separate", field="target",
            )
    open_before = {found.id for found in refine.list_refinements(context.store, run=args.run)}
    made = refine.create(
        context.store, args.run, args.step, target=args.target, target_match=args.target_match,
        target_rules=args.target_rules or None, target_check=args.target_check, created_by=who,
    )
    data = made.to_json()
    source = data.get("source") or {}
    by_check = data["target"].get("kind") == "check"
    lines = [
        f"refinement {made.id[:8]} at run {made.run_id[:8]} step {made.step_idx}, toward: "
        f"{data['target'].get('description')}",
        "next: show_refinement for the evidence, then propose_edit, then start_screen_to_end (each sample a "
        "fork run to the end, the check deciding)" if by_check else
        "next: show_refinement for the evidence, then propose_edit, then start_screen",
    ]
    if made.id in open_before:
        lines.insert(1, f"it was open already, with {len(refine.proposals(context.store, made.id))} proposal(s): "
                        "propose_edit is next; starting it again changes nothing")
    if by_check:
        from tuningfork.ops.inspect import replay_warning

        checkpoint_run = context.store.get_run(made.run_id)
        warning = replay_warning(context.store, checkpoint_run, made.step_idx) if checkpoint_run else None
        if warning:
            lines.append(f"note: {warning}")
    if not by_check:
        # A refinement at a passing run's step measures an edit where the model went right: the
        # original arm of a step screen there starts near the target's ceiling. A 31B agent
        # screened eight sound edits at the one passing run's step, 17–20 of 20 against 20 of 20
        # each time, read "not significant" eight times, and kept none.
        checkpoint_run = context.store.get_run(made.run_id)
        if checkpoint_run is not None and (checkpoint_run.result or {}).get("passed"):
            failed = [found for found in context.store.runs(limit=200, task_id=checkpoint_run.task_id)
                      if (found.result or {}).get("passed") is False
                      and found.forked_from_run is None and found.parent_run_id is None][:3]
            where = ", ".join(compact.short(found.id) for found in failed) or "none recorded yet"
            lines.append(
                f"note: run {compact.short(made.run_id)} passed its check, so this step is where the model went "
                "right: a step screen here starts near the target's ceiling and can show no gain. Prefer a failed "
                f"run's step (failed runs: {where}), or a screen to the end from several checkpoints"
            )
    if seen is not None and seen.get("split"):
        # The target's evidence, said when it is accepted too: how it splits the recorded runs there.
        lines.extend(f"the target, against the answers recorded at step {seen.get('step')}: {line}"
                     for line in split_lines(seen["split"]))
    if source and not source.get("matches"):
        lines.append("note: the skill's file has changed since this run was given it; proposals edit the run's copy")
    return Answer({"refinement": compact.short(made.id), "run": compact.short(made.run_id), "step": made.step_idx,
                   "target": data["target"].get("description"), "skill": made.skill}, lines)


def show_refinement(context: Context, args: RefinementRef) -> Answer:
    from tuningfork.ops import refine
    from tuningfork.ops.packet import build_packet

    shown = refine.view(context.store, args.refinement)
    found = shown["refinement"]
    data: dict[str, Any] = {
        "refinement": compact.short(found["id"]),
        "status": found["status"],
        "in_trash": True if found.get("trashed_at") else None,
        "target": found["target"].get("description"),
        "skill": found["skill"],
        "proposals": capped_proposals(shown["proposals"]),
        "screens": [
            {"experiment": compact.short(screen["experiment"]), "status": screen["status"],
             **(compact.screen(screen["report"]) or {})}
            for screen in shown["screens"][:5]
        ],
        # Without the held-out verdicts, or the gate they feed: those are the person's.
        "validations": [
            {"validation": compact.short(each["validation"]), "proposal": each["proposal"][:8],
             "status": each["status"], "forward": _arms(each["forward"]), "training": _arms(each["training"]),
             "held_out": each["held_out"]}
            for each in shown["validations"][:5]
        ] or None,
    }
    lines = [f"refinement {compact.short(found['id'])}: {len(shown['proposals'])} proposal(s), "
             f"{len(shown['screens'])} screen(s); target: {data['target']}"]
    if found.get("trashed_at"):
        lines.append("it is in the trash: it takes no new proposals, screens or validations until the person restores it")
    for screen in shown["screens"][:1]:
        lines += compact.screen_lines(compact.screen(screen["report"]))
    if shown["validations"]:
        lines.append(f"{len(shown['validations'])} validation(s); their held-out verdicts and the gate are the "
                     "person's to read, and approving is theirs")
    if args.evidence and found["status"] == "open" and not found.get("trashed_at"):
        packet = build_packet(context.store, found["id"])
        data["packet"] = {"sha": packet.sha, **packet.data}
        lines.append(f"packet {packet.sha}: {packet.data.get('answers_here', {}).get('summary', '')}")
        lines.append("propose_edit by line number from show_skill ({line, new}, {after, insert}) or exact text "
                     "({old, new}); cite this packet")
    return Answer(data, lines)


def _arms(part: dict[str, Any] | None) -> dict[str, Any] | None:
    """A validation part's pass rates, the edit's and the original's."""
    if not part:
        return None
    return {arm: compact.rate(part[arm].get("rate")) for arm in ("edit", "original")}


def capped_proposals(items: list[dict[str, Any]]) -> Any:
    return compact.capped([
        {
            "id": item["id"][:8],
            "changed": item["changed"],
            "diff": [line for hunk in item["diff"] for line in hunk["lines"] if line["op"] != " "][:12],
            "proposers": [{key: p.get(key) for key in ("via", "client", "model") if p.get(key)} for p in item["proposers"]],
            "rationale": compact.clip(item["proposers"][0].get("rationale"), 300),
            "screens": [
                {"survives": v.get("survives"), "worse": v.get("worse"), "complete": v.get("complete"),
                 "quick": v.get("quick") or None,
                 "target": compact.rate(v.get("target")), "baseline": compact.rate(v.get("baseline")),
                 "adjusted_p": v.get("adjusted_p")}
                for v in item["screens"]
            ] or None,
        }
        for item in items
    ], 20)


def propose_edit(context: Context, args: ProposeEditArgs, who: dict[str, Any]) -> Answer:
    from tuningfork.ops import refine

    mine = [p for p in refine.proposals(context.store, args.refinement)
            if any(entry.get("via") == "mcp" for entry in p.data.get("proposers", []))]
    if len(mine) >= AGENT_PROPOSALS:
        raise OperationError(
            f"agents have made {len(mine)} proposals to this refinement already",
            hint="screen the ones there: start_screen with their ids",
            code="too_many_proposals", field="refinement",
        )
    made = refine.add_proposal(
        context.store, args.refinement, edits=args.edits, rationale=args.rationale,
        proposer=who, packet=args.packet,
    )
    data = made.to_json()
    lines = [
        f"proposal {made.id[:8]}: {data['changed']} line(s) changed"
        + (" — the same text was proposed before; you are added to its proposers" if made.duplicate else ""),
    ]
    # What each edit resolved to, so a model that named a line sees the line it changed.
    for k, each in enumerate((made.data.get("proposers") or [{}])[-1].get("edits") or [], 1):
        old, new = compact.clip(each.get("old"), 120), compact.clip(each.get("new"), 120)
        lines.append(f"edit {k}, {each.get('lines')}: " + (f"adds {new!r}" if not old else f"{old!r} → {new!r}"))
    lines.append(f"a hypothesis until screened: start_screen(refinement, proposals=[\"{made.id[:8]}\", …])")
    return Answer({"proposal": made.id[:8], "changed": data["changed"], "duplicate": made.duplicate,
                   "variant": data["variant"]}, lines)


# --- jobs, as answers ------------------------------------------------------------------------------


def job_brief(job: dict[str, Any]) -> dict[str, Any]:
    """A job as a list shows it: what, how far, who asked."""
    progress = job.get("progress") or {}
    error = job.get("error") or None
    data = {
        "id": compact.short(job.get("id")),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "progress": {key: progress.get(key) for key in ("done", "total", "waiting", "remaining_s")
                     if progress.get(key) is not None} or None,
        "queue_position": job.get("queue_position"),
        "started_by": job.get("started_by"),
        "created": compact.when(job.get("created_at")),
        "ended": compact.when(job.get("ended_at")),
        "error": {key: error.get(key) for key in ("message", "hint", "field", "code") if error.get(key)} if error else None,
    }
    return {key: value for key, value in data.items() if value is not None}


def held_out_job(context: Context, job: dict[str, Any]) -> bool:
    """Whether a job's work is of a held-out task: a batch a person ran with its held-out
    evals included, the task it names, the run it was started at, any run it made or any
    experiment it drew. Found in review: only the run it was started at and its first
    experiment were looked at, so a held-out batch and a task's check along its runs were
    shown whole."""
    params = job.get("params") or {}
    if params.get("include_held_out"):
        return True
    hidden = hidden_tasks(context)
    if not hidden:
        return False
    if params.get("task") in hidden:
        return True
    from tuningfork.ops.experiments import resolve_run

    store = context.store
    links = job.get("links") or {}
    runs = []
    for reference in (params.get("run"), *(links.get("runs") or [])):
        if reference:
            try:
                runs.append(resolve_run(store, str(reference)))
            except OperationError:
                continue
    for experiment in links.get("experiments") or []:
        found = store.get_experiment(str(experiment))
        if found is not None and found.run_id:
            runs.append(store.get_run(found.run_id))
    return any(run is not None and run.task_id in hidden for run in runs)


def job_answer(context: Context, job: dict[str, Any]) -> Answer:
    """A job with its result, read the way `show_experiment` reads it, when it has one."""
    data = job_brief(job)
    if job.get("deduplicated"):
        data["deduplicated"] = True
    progress = job.get("progress") or {}
    head = f"job {data['id']} ({job.get('kind')}): {job.get('status')}"
    if progress.get("total"):
        head += f", {progress.get('done') or 0} of {progress['total']}"
    if held_out_job(context, job):
        # Nothing of its work: no runs, no result, no error's words, no events (`held_out`).
        data.pop("error", None)
        data["held_out"] = True
        return Answer(data, [head, "its results are of a held-out task: the person's evidence, never an agent's"])
    links = {name: [compact.short(found) for found in ids] for name, ids in (job.get("links") or {}).items() if ids}
    if links:
        data["links"] = links
    lines = [head + (" — the same work was already under way, so it was not started again" if job.get("deduplicated") else "")]
    # What its plan warned of — a context that reads `<redacted>`, say — which a dry run alone showed.
    plan = job.get("plan") if isinstance(job.get("plan"), dict) else {}
    warnings = [warning for warning in plan.get("warnings") or [] if isinstance(warning, str)]
    if warnings:
        data["warnings"] = warnings
        lines += [f"warning: {warning}" for warning in warnings]
    status = job.get("status")
    if status in ("queued", "running"):
        left = progress.get("remaining_s")
        pace = (f"about {max(1, round(left / 60))} minute(s) left at its pace so far — keep waiting" if isinstance(left, (int, float))
                else "how long is left is not known until its first unit lands" if status == "running"
                else "it starts when the jobs before it end")
        lines.append(f"still {status}, {pace}: job_status(job=\"{data['id']}\", wait_seconds={context.max_wait}) "
                     "follows it; never start the same work again")
        return Answer(data, lines)
    if job.get("error"):
        error = job["error"]
        lines.append(f"failed: {error.get('message')}" + (f" — {error['hint']}" if error.get("hint") else ""))
    experiments = (job.get("links") or {}).get("experiments") or []
    result = job.get("result")
    if experiments and status in ("done", "cancelled", "interrupted"):
        from tuningfork.ops.inspect import experiment_report

        view = compact.experiment(experiment_report(context.store, experiments[0]).to_json())
        data["result"] = view
        lines += [line for line in compact.experiment_lines(view)[1:] if line not in lines]
    elif job.get("kind") == "tokens" and isinstance(result, dict):
        data["result"] = compact.tokens(result)
        choice = data["result"].get("choice") or {}
        if choice.get("chance"):
            lines.append(f"the model's own chance of each tool at this step: {compact.choice_text(choice['chance'])}"
                         " — a choice near even is one a few words of the skill can move")
        lines.append("step_tokens shows where the model was least sure; start_forking_tokens tests those tokens")
    elif result is not None:
        data["result"] = compact.fit(result, 12_000)
    if status in ("cancelled", "interrupted"):
        lines.append("it stopped before the end: what it finished is kept and counted")
    return Answer(data, lines)


#: What a job's total counts, by kind, for a dry run's first sentence.
UNITS = {
    "resample": "samples", "ab": "samples", "screen": "samples", "forking-tokens": "samples",
    "resample-to-end": "forks, each run to the end", "ab-to-end": "forks, each run to the end",
    "screen-to-end": "forks, each run to the end",
    "fork": "forks", "batch": "runs", "server-check": "checks",
}


def plan_answer(kind: Kind, planned: dict[str, Any]) -> Answer:
    """A dry run: what the job would do, and what it would cost."""
    plan = planned.get("plan") or {}
    total = planned.get("total")
    made = f"{total} {UNITS.get(kind.name, 'units of work')}" if total is not None else "its work"
    lines = [f"{kind.name} would make {made}; nothing was started"]
    estimate = plan.get("estimate") if isinstance(plan, dict) else None
    if isinstance(estimate, dict):
        # A resample's or an A/B's bound is the recorded step's own time, on a cold cache;
        # the forking probe's is decode time over the tokens it would generate, with each
        # continuation's prompt processing when the step's timing gave it.
        if estimate.get("fresh"):
            tokens = estimate.get("prompt_tokens")
            lines.append("each sample processes its whole prompt" + (f", {tokens:,} tokens" if isinstance(tokens, int) else "")
                         + ", then answers: allow for that on top of the step's own time")
        elif isinstance(estimate.get("worst_minutes"), (int, float)):
            lines.append(f"at most about {max(1, round(estimate['worst_minutes']))} minute(s): the recorded "
                         "step's time for every sample" + ("; a warm cache is usually faster" if estimate.get("cached") else ""))
        elif isinstance(estimate.get("minutes"), (int, float)):
            lines.append(f"about {max(1, round(estimate['minutes']))} minute(s) at the step's recorded speed"
                         + (", its whole prompt processed for each continuation" if estimate.get("prefill_seconds")
                            else "; each continuation also processes the whole prompt first"
                            if estimate.get("prompt_tokens") else ""))
    for warning in (plan.get("warnings") or []) if isinstance(plan, dict) else []:
        lines.append(f"warning: {warning}")
    return Answer({"kind": kind.name, "total": total, "plan": compact.fit(plan, 12_000)}, lines)


# --- the registry -----------------------------------------------------------------------------


READS = (
    Tool("list_tasks", "Tasks and pass rates",
         "Start here. Every task recorded, with its check's pass rate (and 95% interval) under each "
         "condition it ran under: harness, model, skill version, server.", NoArgs, list_tasks),
    Tool("list_runs", "Runs",
         "Recorded runs, newest first: of a task, or where the model called a tool (at a step). Each with "
         "its verdict; a failed run's with the first line its check printed.",
         RunsArgs, list_runs),
    Tool("show_run", "A run step by step",
         "One run step by step: what the model did at each — what its calls ran, and the files they changed "
         "— its verdict and, when it failed, what its check said — the first thing to read about a failed "
         "run — and its threads.", RunArgs, show_run),
    Tool("show_files", "What a run wrote",
         "The files in a run's workspace at its end (or its start, or before a step), one of them by path, "
         "page by page (offset), and with diff what changed — at the end, everything the run changed since it started. What a "
         "run changed beside what it started with is what it did; what a failed run wrote beside what a "
         "passing run wrote is the difference the check saw.", FilesArgs, show_files),
    Tool("find_divergence", "Where runs split",
         "The first step where a task's runs stopped agreeing — the behaviours there, how many runs took "
         "each, how often those went on to pass — the step whose behaviour best predicts the verdict, "
         "when one does (the first split often does not), and the tool whose use goes with passing, when "
         "one does: runs that called it against runs that never did, at whatever step.",
         DivergenceArgs, find_divergence),
    Tool("tally_failures", "What failed checks said, counted",
         "Every failed run's check output, counted: a finding a line (lines differing only in numbers are "
         "one), how many failed runs said it, with its interval, the most common first: the failure to fix "
         "first. By condition for a task; by arm for a screen run to the end.", TallyArgs, tally_failures),
    Tool("show_step", "One step's context",
         "One step as the model saw it: where the skill's text sits (file and lines), the last "
         "messages before it answered, and its answer, each clipped: message reads one whole, page by "
         "page (offset).", StepArgs, show_step),
    Tool("step_behaviours", "Behaviours at a step; try a target",
         "What the model did at a step, across runs, forks and samples, each with how the runs that did it "
         "fared. A step named by its tool (write_file, or write_file|execute_code for whichever came first) "
         "compares every run at its own such step; an index compares the same number, another point in a "
         "run that took another path. Give target_match or target_rules to try a target before a screen: "
         "it should count the passing runs' answers and not the failing runs'. Also the letters of the "
         "step's latest resample.",
         BehavioursArgs, step_behaviours),
    Tool("where_it_holds", "Where a target holds, along every run",
         "One question asked of every run of a task: of the model's answers, at every step, or of a file the runs "
         "wrote, at each state of the workspace. Says in how many passing and failing runs it held, the call that "
         "did it or made it in each, and the step naming that call in every run, for start_refinement. Where the "
         "check prints only its verdict, ask what the passing runs' files have that the failing runs' lack; where "
         "no run passed, a screen's forks (screen).",
         WhereArgs, where_holds),
    Tool("show_skill", "The skill a run was given",
         "The SKILL.md a run was given, its lines numbered, and the variants saved of it. The only way to "
         "read the skill — it is not a file on your machine — and the text propose_edit's old text is "
         "copied from, or an A/B's variant starts from.", RunArgs, show_skill),
    Tool("list_experiments", "Experiments",
         "Resamples, A/Bs and probes, newest first: of a run, or all.", ExperimentsArgs, list_experiments),
    Tool("show_experiment", "An experiment's result",
         "An experiment's result again: behaviours with their rates and intervals; an A/B's declared "
         "target, Fisher's p and its verdict, or that it was exploratory; a probe's flipping tokens.",
         ExperimentArgs, show_experiment),
    Tool("step_tokens", "Where the model wavered",
         "A step's answer as tokens, once scored (start_tokens): the tokens the model was least sure "
         "of, and its alternatives there.", TokensArgs, show_tokens),
    Tool("server_status", "The inference server",
         "The proxy and the server answering it: what it serves, whether it can be read, what it "
         "can do (token analysis needs llama.cpp), and the servers recorded before.", NoArgs, server_status),
    Tool("list_scenarios", "Scenarios",
         "Task specs saved from the web UI, or one in full: what start_batch can run.",
         ScenariosArgs, show_scenarios),
)

REFINE_READS = (
    "list_tasks", "list_runs", "show_run", "show_files", "find_divergence", "tally_failures", "show_step",
    "step_behaviours", "where_it_holds", "show_skill", "list_experiments", "show_experiment",
)
#: list_tasks as the refine toolset offers it: the loop in one place, since a client may show the
#: model no server instructions (Hermes shows none) and the descriptions are then all it has.
REFINE_LIST_TASKS = Tool(
    "list_tasks", "Tasks and pass rates",
    "Start here. Every task recorded, with its check's pass rate (and 95% interval) under each condition it "
    "ran under. The loop from here: tally_failures (what every failed run's check said, counted: the most "
    "common failure first); show_run of a failed run that said it and show_files of what it wrote, beside a "
    "passing run's — where_it_holds when the check prints only its verdict; show_skill for the skill's "
    "text (it is not a file on your machine); "
    "start_refinement at the first step where the failed runs went wrong (a screen replays everything before "
    "its step as recorded, the files included), with a target the passing runs' answers satisfy and the "
    "failing runs' do not (step_behaviours tries one first) — or, when every run made the same mistake, the "
    "task's check; propose_edit with old text copied from show_skill; "
    "start_screen. After a failed screen, a different edit. Proposals are the output, never an edited file.",
    NoArgs, list_tasks,
)
LOOP_READ = Tool(
    "show_refinement", "A refinement and its evidence",
    "A refinement's proposals with their line diffs and screen verdicts, and the evidence packet a "
    "proposal should cite: the step's context, answers that did and did not reach the target, how "
    "runs fared, the task's criteria, edits tried before.", RefinementRef, show_refinement,
)
LOOP_ACTS = (
    Tool("start_refinement", "Start refining a skill",
         "Start refining the skill a run was given, at one of its steps, toward a target declared now: what "
         "the passing runs' answers do there that the failing runs' do not — the tool they call (target_rules "
         "[{\"if\": \"calls\", \"value\": \"write_file\"}], when find_divergence says a tool goes with passing), "
         "or words their answers contain. Name the step by that tool, so a screen's checkpoints are each "
         "run's own such step. Choose the first step where the failed runs went wrong: a screen replays "
         "everything before its step as recorded, the files included. When every run made the same mistake, "
         "the passing ones too (tally_failures says so), start where it was made (changed:<path> or "
         "made:<words> names each run's own) with target_check. Refused when no recorded answer "
         "does it (unless unseen_ok says the behaviour is new), when every one does (nothing to raise), and "
         "when the failing runs' answers do it as readily as the passing runs' (step_behaviours tries a "
         "target first). Returns the open one if it exists.",
         RefinementArgs, act=start_refinement, acts=True),
    Tool("propose_edit", "Propose an edit",
         "Propose a small edit to the refinement's skill: 1 to 3 edits, each a line number from show_skill "
         "with the line as it should read ({line, new}), lines added after one ({after, insert}), or exact "
         "text replaced ({old, new}), and a rationale citing the packet. A hypothesis until start_screen "
         "tests it.",
         ProposeEditArgs, act=propose_edit, acts=True),
)

JOB_TOOLS = (
    Tool("job_status", "A job's progress and result",
         "A job's progress, and its result once it ends — read like show_experiment. Wait on a "
         "running job here; never start the same work again.", JobArgs, acts=False),
    Tool("list_jobs", "Jobs", "Recent jobs, newest first, with who started each.", JobsArgs),
    Tool("cancel_job", "Cancel a job",
         "Stop a job an agent started. What it finished is kept.", CancelArgs, acts=True),
)

#: What each kind is for in the loop, for its tool's description.
KIND_GUIDE = {
    "resample": "How often each behaviour happens at a step, its history held fixed: N samples, "
                "each behaviour lettered with its rate and 95% interval. Run it at the divergence step.",
    "ab": "Does an edited wording change what the model does at a step? Give the whole edited skill as "
          "variant_text with a variant_name. Declare the target before sampling — `target` (a letter "
          "from that step's resample), `target_match` or `target_rules` — or the result is exploratory.",
    "tokens": "Score a step's answer token by token, so step_tokens can show where the model wavered.",
    "forking-tokens": "At the step's least certain tokens, force each likely alternative and let the "
                      "model go on: which tokens change what it does (Holm-corrected).",
    "server-check": "Measure what the server can do: harness needs, token analysis.",
    "batch": "Run a task N times through its harness, each run in a fresh workspace: pass rates with "
             "intervals. Runs a harness, and the task's check, on this machine.",
    "chain": "A task run N times in order, the skill each run leaves given to the next: what the harness's "
             "own loop makes of it. Runs a harness, and the task's check, on this machine.",
    "fork": "Replay a run up to a step and go live from there. Runs a harness on this machine.",
    "resample-to-end": "Fork a run from a step N times, each to the end: how often the task's check "
                       "passes from there. Runs a harness and the check on this machine.",
    "ab-to-end": "The A/B run to the end: each wording's forks judged by the task's check, compared "
                 "with Fisher's test. Runs a harness and the check on this machine.",
    "check-trial": "Try a check on a recorded run's end state, or on a reply: nothing is recorded.",
    "screen": "Screen proposals at the refinement's checkpoint: a fresh sample of the original and of "
              "each proposal (n each), each tested against the original on the target, Holm-corrected "
              "over the proposals named. Everything before the step is held as recorded: the edit is "
              "measured on the step's answer alone. A survivor is evidence to confirm with full runs, not proof.",
    "screen-to-end": "Screen proposals by the task's own check: each arm's samples are forks of the run from "
                     "the checkpoint to the end (n each, the original's included) and the check decides each; "
                     "Fisher against the original, Holm over the proposals named. The original's forks are drawn "
                     "once per refinement and counted again by its later screens. quick: true is a quick look — 8 "
                     "forks of each proposal, rates and no verdict: choose the next edit by it, then screen the "
                     "best in full. A fork replays everything "
                     "before its step as recorded, the files included: an edit to an earlier call shows nothing "
                     "(the plan warns). For a refinement toward the check (target_check), or a "
                     "failure only the check can see. Each sample runs the harness to the end: costlier than "
                     "start_screen. Runs a harness and the check on this machine.",
}


def job_tool(kind: Kind, *, max_wait: int) -> Tool:
    from tuningfork.mcp.schema import HARNESS, tool_name

    guide = KIND_GUIDE.get(kind.name, kind.summary)
    return Tool(
        tool_name(kind.name),
        kind.summary.split(":")[0].split(",")[0][:60],
        f"{guide} A job: returns its result if it ends within wait_seconds, else its id for "
        "job_status. dry_run plans it without starting.",
        kind=kind, machine=kind.name in HARNESS, acts=True,
    )


def registry(toolset: str = "evidence", *, max_wait: int = 120) -> dict[str, Tool]:
    """Every tool an agent is offered, in a fixed order: reads, the loop's own tools (with
    the refine and all toolsets), the jobs it may start, and the tools that follow jobs up."""
    from tuningfork.mcp.schema import offered

    kinds = offered(toolset)
    reads = [tool for tool in READS if toolset not in REFINE_TOOLSETS or tool.name in REFINE_READS]
    if toolset in (*REFINE_TOOLSETS, "all"):
        reads = [REFINE_LIST_TASKS if tool.name == "list_tasks" else tool for tool in reads]
    loop = [LOOP_READ, *LOOP_ACTS] if toolset in (*REFINE_TOOLSETS, "all") else []
    tools = [*reads, *loop, *(job_tool(kind, max_wait=max_wait) for kind in kinds), *JOB_TOOLS]
    return {tool.name: tool for tool in tools}


def input_schema(tool: Tool, *, max_wait: int) -> dict[str, Any]:
    from tuningfork.mcp.schema import job_schema

    if tool.kind is not None:
        return job_schema(tool.kind, max_wait=max_wait)
    schema = args_schema(tool.args or NoArgs)
    if "wait_seconds" in schema["properties"]:
        schema["properties"]["wait_seconds"]["maximum"] = max_wait
    return schema


def check_job_id(value: str) -> str:
    if not JOB_ID.fullmatch(value):
        raise OperationError(f"no job matching {value!r}: job ids are hexadecimal", code="bad_params", field="job")
    return value


def same_params(kind: Kind, params: dict[str, Any], masking: Any) -> dict[str, Any]:
    """Parameters as the job runner stores them — parsed, defaults filled, masked as the store
    masks (`masking`, its own) — so an identical job already under way can be found."""
    from tuningfork.jobs import kept_params

    return kept_params(kind.parse(params), masking)
