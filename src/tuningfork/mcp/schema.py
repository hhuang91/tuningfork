"""The job catalog as MCP tools: which kinds an agent may start, and their arguments.

Each kind's schema is the catalog's own (`ops.common.params_schema`), changed only where an
agent differs from a person at a form:

- `step` takes a number as well as text — a model writes 3, the parser wants "3";
- a fork never stops for a person: there is nobody at the breakpoint to continue it;
- a check is a run's own, or one that reads the reply — never a shell command, which would
  hand the agent a shell;
- every tool adds `dry_run` (the plan: what it would cost, what it can detect) and
  `wait_seconds` (wait for the result, rather than come back for it).

The strict parser stays the gate: these schemas only say what it accepts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from tuningfork.ops.catalog import CATALOG, Kind
from tuningfork.ops.common import OperationError, params_schema

#: Kinds that call the model and nothing else: their risk class, and the default toolset.
EVIDENCE = ("resample", "ab", "tokens", "forking-tokens", "server-check", "screen")
#: Kinds that run a harness, or a task's check, on this machine — as the user, perhaps with
#: skill text the agent wrote. Offered only with `--tools all`.
HARNESS = ("batch", "chain", "fork", "resample-to-end", "ab-to-end", "check-trial", "check-along", "screen-to-end")
#: Kinds an agent is never offered, and why: each is a person's to start.
EXCLUDED = {
    "hold-sample": "samples a call held at a breakpoint: live debugging, for a person",
    "hold-console": "asks the model at a held call: live debugging, for a person",
    "propose": "sends recorded prompts, answers and the skill to the proposer's endpoint — perhaps "
               "a cloud host on the person's key: the person's decision. An agent proposes its own "
               "edits with propose_edit",
    "validate": "runs the held-out tasks that judge an edit: their verdicts are the person's, and an "
                "agent that saw them could fit its edits to them",
}
#: Kinds whose parameters name a refinement: offered with the loop's own tools.
REFINE_ONLY = ("screen", "screen-to-end")
TOOLSETS = {
    "evidence": tuple(kind for kind in EVIDENCE if kind not in REFINE_ONLY),
    # The refinement loop: evidence at a step, proposals, and screens of them.
    "refine": ("resample", "ab", "screen"),
    # The loop with the task's check as a measure too: screens whose samples are forks run to
    # the end through the harness on this machine, for a failure only the check can see.
    "refine-to-end": ("resample", "ab", "screen", "screen-to-end"),
    "all": EVIDENCE + HARNESS,
}

#: How long a call may wait for a job, at most, by transport. Hermes gives a tool call 300 s
#: by default; OpenJarvis reads an HTTP answer for 60 s.
MAX_WAIT = {"stdio": 120, "http": 50}
#: A job's `total` (samples, forks, runs) an agent may ask for, at most, unless the person
#: running `tuningfork mcp` says otherwise: a checkpoint sample costs seconds, a harness run
#: minutes.
DEFAULT_LIMITS = {kind: 100 for kind in EVIDENCE} | {kind: 20 for kind in HARNESS} | {"screen-to-end": 60}

#: What an agent may say of a check: one that reads the reply, or (with `run`) the run's own.
AGENT_CHECK = {
    "type": "object",
    "properties": {"reply_contains": {"type": "string"}, "reply_matches": {"type": "string"}},
    "additionalProperties": False,
    "minProperties": 1,
    "maxProperties": 1,
}


def tool_name(kind: str) -> str:
    return "start_" + kind.replace("-", "_")


@dataclass(frozen=True, slots=True)
class JobCall:
    """A `start_*` call's arguments, read: the kind's parameters, and how to call it."""

    params: dict[str, Any]
    dry_run: bool
    wait_seconds: int


def job_schema(kind: Kind, *, max_wait: int) -> dict[str, Any]:
    schema = copy.deepcopy(params_schema(kind.params))
    properties = schema["properties"]
    if "step" in properties:
        step = properties["step"]
        properties["step"] = {
            "type": ["integer", "string"],
            "description": step.get("description")
            or "The step: its index, or a tool's name for the first step that called it.",
        }
    if kind.name == "fork":
        properties.pop("hold", None)
    # A held-out eval judges an edit: an agent never runs one, so it is never offered the way.
    properties.pop("include_held_out", None)
    # Which of an imported chat's skills an A/B varies: imports are a person's, not an agent's,
    # and every run tuningfork records keeps one.
    properties.pop("skill", None)
    if kind.name == "check-trial" and "check" in properties:
        properties["check"] = {
            **AGENT_CHECK,
            "description": "A check that reads the reply: {\"reply_contains\": …} or "
            "{\"reply_matches\": …}. With a run and none named, the run's own check.",
        }
    schema["required"] = [name for name in schema["required"] if name in properties]
    properties["dry_run"] = {
        "type": "boolean",
        "default": False,
        "description": "Plan only: what it would run, cost and be able to detect. Starts nothing.",
    }
    properties["wait_seconds"] = {
        "type": "integer",
        "minimum": 0,
        "maximum": max_wait,
        "default": min(30, max_wait),
        "description": "How long to wait for the result. 0 returns the job id at once; "
        "job_status follows it up.",
    }
    return schema


def read_job_call(kind: Kind, arguments: dict[str, Any] | None, *, max_wait: int) -> JobCall:
    """A `start_*` call's arguments as the job runner takes them, or a refusal naming the
    field — which the agent reads, and corrects."""
    params = dict(arguments or {})
    dry_run = params.pop("dry_run", False)
    if not isinstance(dry_run, bool):
        raise OperationError("dry_run must be true or false", code="bad_params", field="dry_run")
    wait = params.pop("wait_seconds", min(30, max_wait))
    if isinstance(wait, bool) or not isinstance(wait, int) or not 0 <= wait <= max_wait:
        raise OperationError(
            f"wait_seconds must be a whole number from 0 to {max_wait}", code="bad_params", field="wait_seconds"
        )
    step = params.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        params["step"] = str(step)
    if kind.name == "fork":
        if "hold" in params:
            raise OperationError(
                "a fork started by an agent never stops at a breakpoint: nobody is there to continue it",
                code="bad_params", field="hold",
            )
        params["hold"] = "none"
    if kind.name == "check-trial" and isinstance(params.get("check"), str):
        raise OperationError(
            "an agent's check reads the reply ({\"reply_contains\": …} or {\"reply_matches\": …}), "
            "or is the run's own: a shell command is for the CLI and the web UI",
            code="bad_params", field="check",
        )
    if "include_held_out" in params:
        raise OperationError(
            "held-out evals judge an edit: an agent never runs one",
            code="held_out", field="include_held_out",
        )
    return JobCall(params=params, dry_run=dry_run, wait_seconds=wait)


def offered(toolset: str) -> list[Kind]:
    """The kinds a toolset offers, in the catalog's order."""
    if toolset not in TOOLSETS:
        raise OperationError(f"no toolset called {toolset!r}", hint=f"known: {', '.join(TOOLSETS)}")
    names = set(TOOLSETS[toolset])
    return [kind for kind in CATALOG.values() if kind.name in names]
