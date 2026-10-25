"""Task specifications.

Reuses the `evals.json` convention from Anthropic's skill-creator (`id`, `prompt`,
`expected_output`, `files`, `expectations`) so specs stay usable by existing skill tooling,
and keeps tuningfork's additions under their own `tuningfork` key rather than colonizing the
shared schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence


_OPTION_KEYS = (
    "check", "check_env", "normalizers", "held_out", "skills", "timeout", "max_turns", "context", "stop_at_next_skill",
)
_EVAL_KEYS = ("id", "prompt", "expected_output", "files", "expectations", "tuningfork")

#: The checks tuningfork judges itself, on the agent's final reply, besides a shell command:
#: `{"reply_contains": "61.5"}` passes when the reply holds that text, `{"reply_matches":
#: "(?i)row.count"}` when a regular expression finds a match in it.
REPLY_CHECKS = ("reply_contains", "reply_matches")
#: Names the file holding the agent's final reply, for a shell check that wants it.
REPLY_ENV = "TUNINGFORK_REPLY"
#: The folder of the spec a check belongs to: where a check script can live, beside the
#: spec and out of the workspace the agent reads.
SPEC_DIR_ENV = "TUNINGFORK_SPEC_DIR"
#: What a shell check keeps of the machine's environment, as a launch does: where programs
#: are, the language, the terminal, the temporary folder, who runs it. Nothing else of the
#: person's but what its task names (`check_env`) and what the machine's config names for every
#: check (`configured_check_env`): a check can run code the model wrote, and a variable can hold
#: a key.
CHECK_INHERITED = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")
#: A variable's name, as a shell writes one.
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_kind(check: Any) -> str | None:
    """What kind of check this is: `"shell"`, one of `REPLY_CHECKS`, or None for none."""
    if isinstance(check, str):
        return "shell" if check.strip() else None
    if isinstance(check, dict) and len(check) == 1:
        (key,) = check
        return key if key in REPLY_CHECKS else None
    return None


def describe_check(check: Any) -> str | None:
    """A check as a person reads it: a shell command exactly as it runs, a reply check in
    words. What every front end shows."""
    kind = check_kind(check)
    if kind == "shell":
        return check
    if kind == "reply_contains":
        return f"the reply contains {json.dumps(check[kind], ensure_ascii=False)}"
    if kind == "reply_matches":
        return f"the reply matches /{check[kind]}/"
    return None


@dataclass(slots=True)
class TuningforkOptions:
    #: What decides pass/fail once the agent has finished: a shell command run in the
    #: workspace, or a check of the final reply (`REPLY_CHECKS`).
    check: str | dict[str, str] | None = None
    #: The person's environment variables a shell check may read, by name — an LLM judge's key,
    #: say. It gets no other of theirs (`check_environment`).
    check_env: list[str] = field(default_factory=list)
    normalizers: list[str] = field(default_factory=list)
    held_out: bool = False
    #: Skills to preload for the run (`hermes chat -s ...`).
    skills: list[str] = field(default_factory=list)
    timeout: float = 900.0
    max_turns: int = 12
    #: A conversation each run continues: a JSON file of OpenAI-style messages beside the spec
    #: (`load_context`), which the proxy puts after the harness's instructions and before its
    #: first message, in every call of the run's own thread — a chat imported as a scenario —
    #: and the model's earlier steps of the prompt's own turn, if any, after the prompt.
    context: str | None = None
    #: A run ends where the model loads a skill other than the one studied: the check judges
    #: the workspace then. For a scenario of a chat that went on to another skill, which no run
    #: of this one has.
    stop_at_next_skill: bool = False
    #: Keys this version does not know, kept so a spec rewritten by tuningfork loses none.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TuningforkOptions:
        data = data or {}
        return cls(
            check=data.get("check"),
            check_env=[name for name in data["check_env"] if isinstance(name, str)]
            if isinstance(data.get("check_env"), list) else [],
            normalizers=list(data.get("normalizers") or []),
            held_out=bool(data.get("held_out", False)),
            skills=list(data.get("skills") or []),
            timeout=float(data.get("timeout", 900.0)),
            max_turns=int(data.get("max_turns", 12)),
            context=data.get("context") if isinstance(data.get("context"), str) else None,
            stop_at_next_skill=data.get("stop_at_next_skill") is True,
            extra={k: v for k, v in data.items() if k not in _OPTION_KEYS},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.check is not None:
            data["check"] = self.check
        if self.check_env:
            data["check_env"] = list(self.check_env)
        data.update(
            normalizers=list(self.normalizers),
            held_out=self.held_out,
            skills=list(self.skills),
            timeout=self.timeout,
            max_turns=self.max_turns,
        )
        if self.context is not None:
            data["context"] = self.context
        if self.stop_at_next_skill:
            data["stop_at_next_skill"] = True
        return {**data, **self.extra}


@dataclass(slots=True)
class Eval:
    id: str
    prompt: str
    expected_output: str = ""
    files: list[Any] = field(default_factory=list)
    expectations: list[str] = field(default_factory=list)
    tuningfork: TuningforkOptions = field(default_factory=TuningforkOptions)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Eval:
        """Lenient: a launch's stored eval record (`runner.eval_record`) is read back with
        this, and it carries a key of its own."""
        return cls(
            id=str(data["id"]),
            prompt=data["prompt"],
            expected_output=data.get("expected_output", ""),
            files=list(data.get("files") or []),
            expectations=list(data.get("expectations") or []),
            tuningfork=TuningforkOptions.from_dict(data.get("tuningfork")),
            extra={k: v for k, v in data.items() if k not in _EVAL_KEYS},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "files": list(self.files),
            "expectations": list(self.expectations),
            "tuningfork": self.tuningfork.to_dict(),
            **self.extra,
        }


@dataclass(slots=True)
class TaskSpec:
    skill_name: str
    evals: list[Eval]
    path: Path | None = None
    #: Directory holding the skill under test, relative to the spec.
    skill_dir: str | None = None
    #: Top-level keys, and keys of the top-level `tuningfork` block, this version does not
    #: know: kept, so rewriting a spec loses nothing.
    extra: dict[str, Any] = field(default_factory=dict)
    tuningfork_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent if self.path else Path.cwd()

    def get(self, eval_id: str | None = None) -> Eval:
        if eval_id is None:
            if not self.evals:
                raise ValueError(f"spec {self.skill_name} has no evals")
            return self.evals[0]
        for item in self.evals:
            if item.id == str(eval_id):
                return item
        raise KeyError(f"no eval {eval_id!r} in {self.skill_name}")

    def default_eval(self) -> Eval | None:
        """The eval a batch runs when none is named: the first that is not held out. Held-out
        evals judge an edit, and only run when asked for by name."""
        return next((item for item in self.evals if not item.tuningfork.held_out), None)

    def held_out_evals(self) -> list[Eval]:
        return [item for item in self.evals if item.tuningfork.held_out]

    def resolved_skill_dir(self) -> Path | None:
        """The skill folder as this machine opens it: relative to the spec's folder, and a
        Windows path read the way WSL mounts it (`paths.resolve_path`)."""
        if not self.skill_dir:
            return None
        from tuningfork.paths import resolve_path

        return resolve_path(self.skill_dir, self.directory).path

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> TaskSpec:
        top = data.get("tuningfork") or {}
        return cls(
            skill_name=data.get("skill_name") or (path.stem if path else "task"),
            evals=[Eval.from_dict(item) for item in data.get("evals") or []],
            path=path,
            skill_dir=top.get("skill_dir"),
            extra={k: v for k, v in data.items() if k not in ("skill_name", "evals", "tuningfork")},
            tuningfork_extra={k: v for k, v in top.items() if k != "skill_dir"},
        )

    def to_dict(self) -> dict[str, Any]:
        """The spec as `evals.json` holds it: what tuningfork writes when it saves one."""
        top = {**({"skill_dir": self.skill_dir} if self.skill_dir else {}), **self.tuningfork_extra}
        return {
            "skill_name": self.skill_name,
            **({"tuningfork": top} if top else {}),
            "evals": [item.to_dict() for item in self.evals],
            **self.extra,
        }


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a spec: where, and what."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}" if self.where else self.message


def validate_spec(data: Any) -> list[Problem]:
    """Everything wrong with a spec, before anything runs.

    Loading used to be forgiving in ways that change what is measured: `"held_out": "false"`
    read as true, an unknown normalizer failed only at sampling time, and a file entry could
    point outside the workspace.
    """
    from tuningfork.analysis.normalize import KNOWN_NORMALIZERS

    if not isinstance(data, dict):
        return [Problem("", "a spec is a JSON object")]
    problems: list[Problem] = []

    def add(where: str, message: str) -> None:
        problems.append(Problem(where, message))

    if "skill_name" in data and not isinstance(data["skill_name"], str):
        add("skill_name", "must be text")
    top = data.get("tuningfork")
    if top is not None:
        if not isinstance(top, dict):
            add("tuningfork", "must be an object")
        elif "skill_dir" in top and not isinstance(top["skill_dir"], str):
            add("tuningfork.skill_dir", "must be text")
    evals = data.get("evals")
    if not isinstance(evals, list) or not evals:
        add("evals", "must be a list with at least one eval")
        return problems
    seen: set[str] = set()
    for index, item in enumerate(evals):
        at = f"evals[{index}]"
        if not isinstance(item, dict):
            add(at, "must be an object")
            continue
        identity = item.get("id")
        if isinstance(identity, bool) or not isinstance(identity, (str, int)) or str(identity) == "":
            add(f"{at}.id", "is required: text or a whole number")
        elif str(identity) in seen:
            add(f"{at}.id", f"{identity!r} is used by another eval")
        else:
            seen.add(str(identity))
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            add(f"{at}.prompt", "is required: the task as the harness is given it")
        if "expected_output" in item and not isinstance(item["expected_output"], str):
            add(f"{at}.expected_output", "must be text")
        expectations = item.get("expectations", [])
        if not isinstance(expectations, list) or not all(isinstance(e, str) for e in expectations):
            add(f"{at}.expectations", "must be a list of text")
        files = item.get("files", [])
        if not isinstance(files, list):
            add(f"{at}.files", "must be a list")
            files = []
        for position, entry in enumerate(files):
            problem = _file_problem(entry)
            if problem:
                add(f"{at}.files[{position}]", problem)
        options = item.get("tuningfork", {})
        if not isinstance(options, dict):
            add(f"{at}.tuningfork", "must be an object")
            continue
        where = f"{at}.tuningfork"
        problems.extend(check_problems(options.get("check"), f"{where}.check"))
        if "check_env" in options:
            problem = check_env_problem(options["check_env"])
            if problem:
                add(f"{where}.check_env", problem)
        normalizers = options.get("normalizers", [])
        if not isinstance(normalizers, list) or not all(isinstance(n, str) for n in normalizers):
            add(f"{where}.normalizers", "must be a list of names")
        else:
            unknown = [n for n in normalizers if n not in KNOWN_NORMALIZERS]
            if unknown:
                add(f"{where}.normalizers", f"unknown: {', '.join(unknown)}; known: {', '.join(KNOWN_NORMALIZERS)}")
        if "held_out" in options and not isinstance(options["held_out"], bool):
            add(f"{where}.held_out", "must be true or false")
        skills = options.get("skills", [])
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            add(f"{where}.skills", "must be a list of skill names")
        timeout = options.get("timeout", 900.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            add(f"{where}.timeout", "must be a number of seconds above 0")
        turns = options.get("max_turns", 12)
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
            add(f"{where}.max_turns", "must be a whole number of at least 1")
        if "context" in options:
            context = options["context"]
            if not isinstance(context, str) or not context.endswith(".json") or _climbs(context):
                add(f"{where}.context", "must be the path of a .json file beside the spec, inside its folder")
        if "stop_at_next_skill" in options:
            if not isinstance(options["stop_at_next_skill"], bool):
                add(f"{where}.stop_at_next_skill", "must be true or false")
            elif options["stop_at_next_skill"] and not data.get("skill_name"):
                add(f"{where}.stop_at_next_skill", "names no skill to stop beside: the spec needs its skill_name")
    return problems


def _climbs(path: str) -> bool:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return not parts or path.startswith(("/", "\\")) or ":" in parts[0] or any(part in ("", ".", "..") for part in parts)


#: The roles a context may hold: a conversation, not the harness's instructions.
CONTEXT_ROLES = ("user", "assistant", "tool")


@dataclass(slots=True)
class Context:
    """The conversation a scenario's runs continue (`tuningfork.context`): what came before
    its prompt, and what the model had already done in the prompt's own turn — its calls and
    their results, an earlier skill among them — which its first live answer follows. Kept as
    a list of messages, the part before the prompt, or as {"before": [...], "in_turn": [...]}."""

    before: list[dict[str, Any]] = field(default_factory=list)
    in_turn: list[dict[str, Any]] = field(default_factory=list)

    @property
    def messages(self) -> int:
        return len(self.before) + len(self.in_turn)

    def to_json(self) -> list[dict[str, Any]] | dict[str, Any]:
        return list(self.before) if not self.in_turn else {"before": list(self.before), "in_turn": list(self.in_turn)}

    @classmethod
    def read(cls, data: Any) -> Context:
        """A context as `to_json` keeps it, read back without checking it."""
        if isinstance(data, dict):
            return cls(list(data.get("before") or []), list(data.get("in_turn") or []))
        return cls(list(data or []))


def load_context(spec_dir: Path | str, path: str) -> Context:
    """The conversation a spec's `tuningfork.context` names, checked as one a harness can
    continue (`check_context`)."""
    if _climbs(path):
        raise SpecError(f"context {path!r} is not a path inside the spec's folder")
    file = Path(spec_dir) / path
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"cannot read the context {file}: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise SpecError(f"the context {file} is not JSON: {exc}") from None
    return check_context(data, f"the context {file}")


def check_context(data: Any, what: str = "the context") -> Context:
    """`data` as a context a harness can continue, or a SpecError saying what is wrong with it.

    Before the prompt: user, assistant and tool messages (the harness brings its own
    instructions), every tool call answered, and never ending on a user turn — the run's own
    prompt follows it. In the prompt's turn: the model's answers and their results, from its
    first answer on — a note the harness sent between them, too — every tool call answered, and
    never ending on an answer that called nothing, which would have ended the turn."""
    if isinstance(data, list):
        if not data:
            raise SpecError(f"{what} must be a list of messages, not empty")
        return Context(_check_messages(data, "context message"))
    if not isinstance(data, dict) or set(data) - {"before", "in_turn"}:
        raise SpecError(f"{what} must be a list of messages, or {{\"before\": [...], \"in_turn\": [...]}}")
    for part in ("before", "in_turn"):
        if not isinstance(data.get(part, []), list):
            raise SpecError(f"{what}'s {part} must be a list of messages")
    before, in_turn = list(data.get("before") or []), list(data.get("in_turn") or [])
    if not before and not in_turn:
        raise SpecError(f"{what} holds no message")
    if in_turn:
        if not isinstance(in_turn[0], dict) or in_turn[0].get("role") != "assistant":
            raise SpecError(f"{what}'s in_turn starts with the model's first answer: an assistant message")
        _check_messages(in_turn, "in_turn message", in_turn=True)
    return Context(_check_messages(before, "context message") if before else [], in_turn)


def _check_messages(messages: list[Any], name: str, *, in_turn: bool = False) -> list[dict[str, Any]]:
    open_calls: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in CONTEXT_ROLES:
            raise SpecError(f"{name} {index} must be a user, assistant or tool message")
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not call.get("id") or not isinstance(call.get("function"), dict):
                    raise SpecError(f"{name} {index} holds a tool call without an id and a function")
                open_calls.add(str(call["id"]))
        elif message["role"] == "tool":
            if str(message.get("tool_call_id")) not in open_calls:
                raise SpecError(f"{name} {index} answers a tool call no earlier message made")
            open_calls.discard(str(message.get("tool_call_id")))
        if message["role"] == "user" and open_calls:
            raise SpecError(f"{name} {index} comes before the tool calls above it were answered")
    if open_calls:
        raise SpecError(f"the {'in_turn' if in_turn else 'context'} ends with a tool call not answered")
    last = messages[-1]
    if not in_turn and last["role"] == "user":
        raise SpecError("the context ends on a user turn: the run's own prompt follows it, two in a row")
    if in_turn and last["role"] == "assistant":
        raise SpecError("the in_turn ends on an answer that called nothing: the turn was over, and no answer follows")
    return messages


def check_problems(check: Any, where: str = "check") -> list[Problem]:
    """What is wrong with a check, if anything: a shell command (text), or one reply check
    whose text is there and, for a pattern, compiles. None is no check, which is allowed."""
    if check is None or isinstance(check, str):
        return []
    if not isinstance(check, dict) or len(check) != 1 or next(iter(check)) not in REPLY_CHECKS:
        return [Problem(
            where,
            'must be a shell command, as text, or {"reply_contains": "<text>"} or '
            '{"reply_matches": "<regular expression>"}',
        )]
    ((key, value),) = check.items()
    if not isinstance(value, str) or not value:
        return [Problem(f"{where}.{key}", "must be text, not empty")]
    if key == "reply_matches":
        try:
            re.compile(value)
        except re.error as exc:
            return [Problem(f"{where}.reply_matches", f"is not a regular expression: {exc}")]
    return []


def check_env_problem(names: Any) -> str | None:
    """What is wrong with a check's `check_env`, if anything: a list of variable names, none
    of them tuningfork's own — its keys, which no check gets, or one it sets for the check."""
    from tuningfork.access import SECRET_ENVS

    if not isinstance(names, list) or not all(isinstance(name, str) and ENV_NAME.match(name) for name in names):
        return "must be a list of environment variable names"
    own = [name for name in names if name in (*SECRET_ENVS, REPLY_ENV, SPEC_DIR_ENV)]
    if own:
        return f"names a variable tuningfork keeps for itself: {', '.join(own)}"
    return None


def _file_problem(entry: Any) -> str | None:
    def bad_path(path: Any) -> str | None:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            return "names no usable path"
        # Judged as Windows would too: a spec is portable, and C:/x is absolute somewhere.
        windows = PureWindowsPath(path)
        if Path(path).is_absolute() or path.startswith(("/", "\\")) or windows.drive:
            return f"{path!r} must be a path inside the workspace, not an absolute one"
        if ".." in Path(path).parts or ".." in windows.parts or Path(path).name in ("", "."):
            return f"{path!r} would land outside the workspace"
        return None

    if isinstance(entry, str):
        if Path(entry).name in ("", ".", ".."):
            return f"{entry!r} names no file"
        return None
    if not isinstance(entry, dict):
        return "must be a file name, or an object with a path"
    problem = bad_path(entry.get("path"))
    if problem:
        return problem
    if "source" in entry:
        return None if isinstance(entry["source"], str) and entry["source"] else "source must name a file beside the spec"
    if not isinstance(entry.get("content", ""), str):
        return "content must be text; name a file with source for anything else"
    return None


def load_spec(path: Path, *, strict: bool = True) -> TaskSpec:
    """A spec from its file. Strict by default: a spec with problems is refused, each named,
    rather than half-read."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SpecError(f"{path} is not valid JSON: {exc}") from None
    if strict:
        problems = validate_spec(data)
        if problems:
            listed = "; ".join(str(p) for p in problems[:6])
            more = f" (and {len(problems) - 6} more)" if len(problems) > 6 else ""
            raise SpecError(f"{path} has problems: {listed}{more}", problems)
    return TaskSpec.from_dict(data, path)


def missing_files(item: Eval, spec_dir: Path) -> list[str]:
    """Input files an eval names by path that are not next to its spec."""
    named = [
        entry if isinstance(entry, str) else entry.get("source")
        for entry in item.files
        if isinstance(entry, str) or (isinstance(entry, dict) and "source" in entry)
    ]
    return [name for name in named if not isinstance(name, str) or not (Path(spec_dir) / name).exists()]


class SpecError(ValueError):
    """A task spec that cannot be used as written."""

    def __init__(self, message: str, problems: Sequence[Problem] = ()) -> None:
        super().__init__(message)
        self.problems = list(problems)


def inside(root: Path, relative: str, what: str = "input file") -> Path:
    """`root / relative`, refusing anything that would land outside `root`.

    A spec's file entries name where in the workspace a file goes. Found in review: an
    absolute path, or one climbing out with `..`, was written wherever it pointed — outside
    the run's workspace, as the user running tuningfork.
    """
    if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
        raise SpecError(f"{what} {relative!r} is not a usable file name")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or relative.startswith(("/", "\\")):
        raise SpecError(f"{what} {relative!r} must be a path inside the workspace, not an absolute one")
    base = Path(root).resolve()
    target = (base / candidate).resolve()
    if target == base or not target.is_relative_to(base):
        raise SpecError(f"{what} {relative!r} would land outside the workspace")
    return target


def materialize_files(item: Eval, workspace: Path, spec_dir: Path) -> None:
    """Place an eval's input files into a fresh workspace.

    Entries may be `{"path": ..., "content": ...}` for inline text, `{"path": ...,
    "source": ...}` for a file next to the spec copied as bytes, or a plain string naming a
    file or folder next to the spec. A named file that is not there is an error: a run
    without its input would still run, and fail for a reason that has nothing to do with
    the skill. Nothing is ever written outside the workspace.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for entry in item.files:
        if isinstance(entry, dict):
            target = inside(workspace, entry.get("path"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if "source" in entry:
                source = Path(spec_dir) / entry["source"]
                if not source.is_file():
                    raise FileNotFoundError(f"input file {entry['source']!r} is not in {spec_dir}")
                shutil.copyfile(source, target)
            else:
                target.write_text(entry.get("content", ""), encoding="utf-8")
        elif isinstance(entry, str):
            source = Path(spec_dir) / entry
            if not source.exists():
                raise FileNotFoundError(f"input file {entry!r} is not in {spec_dir}")
            # Copied in under its own name: an entry of `..` would otherwise copy into the
            # workspace's parent.
            target = inside(workspace, source.name)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)


def is_bookkeeping(path: Path) -> bool:
    """Whether a file under a skills directory is harness bookkeeping, not skill content.

    Hermes writes `.usage.json` and `.usage.json.lock` into the skills directory as it
    runs. Hashing those would flag *every* run as having modified the skill under test,
    which would invalidate an entire experiment for no reason.
    """
    if path.name.startswith("."):
        return True
    if path.suffix in {".lock", ".pyc", ".log", ".tmp"}:
        return True
    return "__pycache__" in path.parts


def hash_tree(roots: Iterable[Path]) -> dict[str, str]:
    """Content hashes for skill files under `roots`, keyed by path relative to each root.

    Used to detect a harness rewriting the skill under test mid-experiment. Hermes's own
    system prompt tells the model to patch skills it finds wanting, so this is a live
    hazard rather than a theoretical one.
    """
    hashes: dict[str, str] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for file in sorted(root.rglob("*")):
            if file.is_file() and not is_bookkeeping(file.relative_to(root)):
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                hashes[str(file.relative_to(root))] = digest
    return hashes


def fingerprint(hashes: dict[str, str]) -> str | None:
    """A short fingerprint of a skill's files from their hashes (`hash_tree`): the same for
    two folders whose files match byte for byte, and None for no files. How a run names the
    skill version it was given (`Run.skill_version`)."""
    if not hashes:
        return None
    signature = json.dumps(hashes, sort_keys=True)
    return hashlib.sha256(signature.encode()).hexdigest()[:8]


def skill_bodies(roots: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """For each SKILL.md under `roots`: the hash of its body with the frontmatter removed,
    and the file line the body starts on, keyed like `hash_tree`.

    Some harnesses show the model only the body — Claude Code strips the frontmatter — so
    the text in the payload matches no file byte for byte. This is what still ties it to
    its file, with the file's own line numbers. `full_sha256` is the whole file with its
    line endings normalized, for a harness that read a CRLF file in text mode. A file with
    no frontmatter is all body.
    """
    from tuningfork.experiment.skill import normalize_newlines, split_frontmatter, text_sha256

    bodies: dict[str, dict[str, Any]] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for file in sorted(root.rglob("SKILL.md")):
            text = normalize_newlines(file.read_text(encoding="utf-8", errors="replace"))
            frontmatter, body = split_frontmatter(text)
            if not body.strip():
                continue
            skipped = body[: len(body) - len(body.lstrip())]
            first_line = frontmatter.count("\n") + skipped.count("\n") + 1
            bodies[str(file.relative_to(root))] = {
                "sha256": text_sha256(body.strip()),
                "first_line": first_line,
                "full_sha256": text_sha256(text.strip()),
            }
    return bodies


@dataclass(slots=True)
class CheckResult:
    passed: bool
    output: str
    returncode: int | None = None
    #: Characters cut from the output's start to keep its end: said wherever it is shown.
    cut: int = 0


#: How much of the reply a reply check's output quotes.
REPLY_QUOTED = 500


#: The end of a check's output a verdict keeps, in characters.
CHECK_KEPT = 2000


def configured_check_env(home: Path | None = None) -> list[str]:
    """The variables every check here may read besides those its task names: `"check_env"` in
    the machine's config (`config.machine_config_path()`) and in the recordings folder's
    `config.json`, together — a proxy's settings, say, which belong to the machine and to no
    task file. A list that cannot be read is refused, saying where it is written."""
    from tuningfork.config import HOME_CONFIG, machine_config_path, read_home_config, read_machine_config
    from tuningfork.ops.common import OperationError

    places = [(machine_config_path(), read_machine_config())]
    if home is not None:
        places.append((Path(home) / HOME_CONFIG, read_home_config(home)))
    names: list[str] = []
    for where, data in places:
        if "check_env" not in data:
            continue
        problem = check_env_problem(data["check_env"])
        if problem:
            raise OperationError(
                f'"check_env" in {where} {problem}',
                hint='write it as a list of variable names: {"check_env": ["https_proxy", "SSL_CERT_FILE"]}',
                code="check_env", field="check_env",
            )
        names += data["check_env"]
    return list(dict.fromkeys(names))


def check_environment(
    names: Sequence[str] = (), *, reply_file: Path | None = None, spec_dir: Path | None = None,
) -> dict[str, str]:
    """A shell check's whole environment, made from scratch, with the sandbox or without it:
    the few variables a launch keeps (`CHECK_INHERITED`), the ones its task names (`names`,
    its `check_env`) that are set — never one of tuningfork's own keys, however a spec names
    it — and where the reply and the spec's folder are."""
    from tuningfork.access import SECRET_ENVS

    wanted = (*CHECK_INHERITED, *(name for name in names if name not in SECRET_ENVS))
    environment = {name: os.environ[name] for name in wanted if name in os.environ}
    if reply_file is not None:
        environment[REPLY_ENV] = str(reply_file)
    if spec_dir is not None:
        environment[SPEC_DIR_ENV] = str(Path(spec_dir).resolve())
    return environment


def run_check(
    command: str | dict[str, str] | None,
    workspace: Path,
    timeout: float = 300.0,
    *,
    reply: str | None = None,
    reply_file: Path | None = None,
    spec_dir: Path | None = None,
    check_env: Sequence[str] = (),
    sandbox: Any = None,
    keep: int | None = CHECK_KEPT,
) -> CheckResult:
    """Judge a finished run: a shell command run in its workspace, or a reply check.

    `reply` is the agent's final reply (empty when it ended without one). A reply check
    judges it here, in Python; a shell command finds it in the file `REPLY_ENV` names, and
    its spec's folder in `SPEC_DIR_ENV` — where a check script can sit, out of the agent's
    sight: `python3 "$TUNINGFORK_SPEC_DIR/checks/report.py"`. Its output is kept to its last
    `keep` characters; None keeps it whole, for a caller that reads every line of it.

    A shell check can run what the model left in the workspace — a `conftest.py` its `pytest`
    finds, a module its `python -m` imports — so it gets an environment made from scratch
    (`check_environment`: the variables its task names in `check_env`, and nothing else of the
    person's), and, given the `sandbox` its launch ran in, runs there
    (`tuningfork.sandbox.Sandbox.check_command`).

    No check means no verdict — recorded as `None`-ish rather than silently passing, since
    a check that does not run must never look like a success.
    """
    kind = check_kind(command)
    if kind is None:
        return CheckResult(passed=False, output="no check command defined", returncode=None)
    if kind in REPLY_CHECKS:
        return _reply_check(kind, command[kind], reply or "")
    environment = check_environment(check_env, reply_file=reply_file, spec_dir=spec_dir)
    if sandbox is not None:
        argv, environment = sandbox.check_command(
            command, workspace=Path(workspace), environment=environment, reply_file=reply_file, spec_dir=spec_dir,
        )
        how: dict[str, Any] = {"args": argv, "shell": False}
    else:
        how = {"args": command, "shell": True}
    try:
        completed = subprocess.run(
            **how,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(passed=False, output=f"check timed out after {timeout}s")
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    kept = output[-keep:] if keep is not None else output
    return CheckResult(
        passed=completed.returncode == 0,
        output=kept,
        returncode=completed.returncode,
        cut=len(output) - len(kept),
    )


def _reply_check(kind: str, wanted: str, reply: str) -> CheckResult:
    """A reply check, judged here: its output says what was judged, and quotes the reply."""
    if kind == "reply_contains":
        passed = wanted in reply
        verdict = f"the reply {'contains' if passed else 'does not contain'} {json.dumps(wanted, ensure_ascii=False)}"
    else:
        # `.` crosses line breaks: a reply is one text, and "A and B anywhere" is written
        # (?=.*A)(?=.*B). Found on a real check: without it, A and B had to share a line.
        passed = re.search(wanted, reply, re.DOTALL) is not None
        verdict = f"the reply {'matches' if passed else 'does not match'} /{wanted}/"
    if not reply:
        quoted = "(none: the run ended without a reply)"
    elif len(reply) > REPLY_QUOTED:
        quoted = f"{reply[:REPLY_QUOTED]}… (+{len(reply) - REPLY_QUOTED} chars)"
    else:
        quoted = reply
    return CheckResult(passed=passed, output=f"{verdict}\n\nthe reply:\n{quoted}", returncode=None)
