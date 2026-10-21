"""A conversation an agent had outside tuningfork, imported as a run: recorded as if it had come
through the proxy, so the step tools work on it — resample, A/B a wording, the forking-token
probe, a refinement's screens at a step — each step with the chat's earlier turns as its context.

A history keeps what the agent kept, not the requests it sent, so each step's request is rebuilt
from it as its harness sends one: the system prompt, every message before the step, and the
tools where the file names them (Hermes's export does not). An assistant turn's reasoning is in
its own step's answer, never in a later request: Hermes sends none back to a server that does not
ask for it. Each message is as the harness changes its own copy before a call
(`HarnessAdapter.as_sent`: Hermes strips each one and writes tool-call arguments compactly),
while each step's answer stays as the model gave it. The run says it was imported and what the
file lacked (`parity.imported`); nothing launched it, so nothing can fork it to the end.

Read: Hermes's session snapshot (`~/.hermes/sessions/session_<id>.json`: the system prompt, the
tools and the messages), Hermes's export (`hermes sessions export`: one session a line, without
tools), and a plain OpenAI-style message list, or an object holding one. The skill under study is
found where its harness shows skills (`HarnessAdapter.skill_spans`; SKILL.md frontmatter where
the harness is not one tuningfork knows), and kept beside the run (`<home>/imports/<run>/`):
the skill folder the person names, or the text the model read, extracted — SKILL.md alone, since
the chat never showed the rest of the folder.

Any run — imported, launched, or recorded as it happened — becomes a scenario (`prepare_scenario`):
a person's turn of its conversation the prompt, the conversation before it the context each run
continues, its skill, its check, and the files the runs need: a launched run's workspace where the
scenario starts, from its snapshots; for any other, the files its conversation names, for a person
to add.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.specs import Context
from tuningfork.store.db import Store

FOLDER = "imports"
#: A history is text; one past this is a mistake, or more than one session's worth.
MAX_BYTES = 64 * 1024 * 1024
ROLES = ("system", "user", "assistant", "tool")
#: A skill's text where the harness is not one tuningfork knows: a SKILL.md, frontmatter first.
FRONTMATTER = re.compile(r"(?ms)^---[ \t]*\n(?:[^\n]*\n)*?name:[ \t]*(?P<name>[^\n]+?)[ \t]*\n(?:[^\n]*\n)*?---[ \t]*\n")


def root(home: Path | str) -> Path:
    return Path(home) / FOLDER


@dataclass(slots=True)
class Chat:
    """One conversation as a file kept it, its messages in OpenAI's chat shape, the system
    prompt first where the file kept one."""

    format: str
    messages: list[dict[str, Any]]
    session_id: str | None = None
    title: str | None = None
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    #: The harness whose file it is, where the format says.
    harness: str | None = None
    started_at: float | None = None
    #: How each assistant turn ended, by its index in `messages`, where the file says.
    finish: dict[int, str] = field(default_factory=dict)
    #: A recording's step at each message its answer is, where the run's calls may have carried
    #: more than its own steps — a scenario's conversation, spliced in. None: each assistant turn
    #: is a step, in order, as an import records them.
    numbers: dict[int, int] | None = None

    @property
    def steps(self) -> list[int]:
        """Each assistant turn's index in `messages`: the steps an import records."""
        return [index for index, message in enumerate(self.messages) if message["role"] == "assistant"]

    def step_of(self, index: int) -> int | None:
        """The run's step whose answer is the message at `index`; None for one that is not."""
        if self.numbers is not None:
            return self.numbers.get(index)
        steps = self.steps
        return steps.index(index) if index in steps else None

    @property
    def system_prompt(self) -> bool:
        return bool(self.messages) and self.messages[0]["role"] == "system"

    def request(self, index: int) -> dict[str, Any]:
        """The request the assistant turn at `index` answered, as its harness sends one."""
        request: dict[str, Any] = {"model": self.model or "imported", "messages": as_sent(self.harness, self.messages[:index])}
        if self.tools:
            request["tools"] = self.tools
        return request

    def response(self, index: int) -> dict[str, Any]:
        """The turn at `index` as tuningfork keeps an answer (an OpenAI `chat.completion`)."""
        message = dict(self.messages[index])
        finish = self.finish.get(index) or ("tool_calls" if message.get("tool_calls") else "stop")
        return {
            "id": f"imported-{index}", "object": "chat.completion", "model": self.model or "imported",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }

    def to_json(self) -> dict[str, Any]:
        """The conversation as the import keeps it beside the run: what a scenario starts from."""
        return {
            "format": self.format, "session_id": self.session_id, "title": self.title, "model": self.model,
            "harness": self.harness, "started_at": self.started_at, "tools": self.tools,
            "messages": self.messages, "finish": {str(index): reason for index, reason in self.finish.items()},
        }


def sent(message: dict[str, Any]) -> dict[str, Any]:
    """A message as a later request carries it: without the turn's reasoning."""
    return {key: value for key, value in message.items() if key not in ("reasoning_content", "reasoning")}


def as_sent(harness: str | None, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A kept history as `harness` sends it in a call: without each turn's reasoning, and as the
    harness changes its own copy before every call (`HarnessAdapter.as_sent`)."""
    from tuningfork.harness import ADAPTERS

    kept = [sent(message) for message in messages]
    adapter = ADAPTERS.get(harness or "generic")
    shape = getattr(adapter(), "as_sent", None) if adapter is not None else None
    return shape(kept) if shape is not None else kept


# --- reading a file ------------------------------------------------------------------------------


def read_chat(data: bytes | str, *, session: str | None = None) -> Chat:
    """The conversation a file holds. A file holding several sessions — Hermes's export of more
    than one — needs `session`: an id, or its first characters (`sessions_in` lists them)."""
    rows = _rows(data)
    if len(rows) == 1 and isinstance(rows[0], list):
        return _plain(rows[0])
    sessions = _sessions(rows)
    found = _choose(sessions, session)
    if "session_id" in found and ("system_prompt" in found or "tools" in found):
        return _hermes(found, "hermes-snapshot", found.get("session_id"), _when(found.get("session_start")))
    if "id" in found and "source" in found:
        return _hermes(found, "hermes-export", found.get("id"), _when(found.get("started_at")))
    return _plain(found["messages"], tools=found.get("tools"), model=found.get("model"))


def sessions_in(data: bytes | str) -> list[dict[str, Any]]:
    """The sessions a file holds, to choose one from: each one's id, title and messages. None
    for a file of one conversation."""
    rows = _rows(data)
    if len(rows) == 1 and isinstance(rows[0], list):
        return []
    return [
        {"id": str(found.get("session_id") or found.get("id") or ""), "title": found.get("title") or None,
         "messages": len(found.get("messages") or []), "started_at": _when(found.get("started_at") or found.get("session_start"))}
        for found in _sessions(rows)
    ]


def _rows(data: bytes | str) -> list[Any]:
    """The file's JSON: one value, or one a line (JSON lines)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_BYTES:
        raise OperationError(f"the file is {len(raw):,} bytes: at most {MAX_BYTES:,}", field="file")
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        raise OperationError("the file is not text in UTF-8", field="file") from None
    if not text:
        raise OperationError("the file is empty", field="file")
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        try:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise OperationError(f"neither JSON nor JSON lines: {error.msg}, line {error.lineno}", field="file") from None


def _sessions(rows: list[Any]) -> list[dict[str, Any]]:
    sessions = [row for row in rows if isinstance(row, dict) and isinstance(row.get("messages"), list)]
    if not sessions:
        raise OperationError(
            "no conversation in the file: it holds no list of messages",
            hint="a Hermes session file (~/.hermes/sessions/session_<id>.json), an export (`hermes sessions "
                 "export`), or an OpenAI-style list of messages",
            field="file",
        )
    return sessions


def _choose(sessions: list[dict[str, Any]], session: str | None) -> dict[str, Any]:
    def ident(found: dict[str, Any]) -> str:
        return str(found.get("session_id") or found.get("id") or "")

    if session:
        matching = [found for found in sessions if ident(found).startswith(session)]
        if len(matching) != 1:
            raise OperationError(f"{'no' if not matching else 'more than one'} session in the file matches {session!r}",
                                 hint=_sessions_hint(sessions), field="session")
        return matching[0]
    if len(sessions) > 1:
        raise OperationError(f"the file holds {len(sessions)} sessions: name the one to import",
                             hint=_sessions_hint(sessions), code="choose_session", field="session")
    return sessions[0]


def _sessions_hint(sessions: list[dict[str, Any]]) -> str:
    shown = [
        f"{found.get('session_id') or found.get('id')} ({len(found.get('messages') or [])} messages"
        + (f": {str(found['title'])[:40]}" if found.get("title") else "") + ")"
        for found in sessions[:8]
    ]
    return "the sessions in it: " + "; ".join(shown) + ("; …" if len(sessions) > 8 else "")


def _when(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime

        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _hermes(found: dict[str, Any], kind: str, session_id: Any, started: float | None) -> Chat:
    messages, finish = _messages(found["messages"])
    prompt = found.get("system_prompt")
    if isinstance(prompt, str) and prompt and not (messages and messages[0]["role"] == "system"):
        messages.insert(0, {"role": "system", "content": prompt})
        finish = {index + 1: reason for index, reason in finish.items()}
    tools = found.get("tools") if isinstance(found.get("tools"), list) and found.get("tools") else None
    return Chat(format=kind, messages=messages, session_id=str(session_id) if session_id else None,
                title=found.get("title") or None, model=found.get("model") or None, tools=tools,
                harness="hermes", started_at=started, finish=finish)


def _plain(rows: list[Any], *, tools: Any = None, model: Any = None) -> Chat:
    messages, finish = _messages(rows)
    return Chat(format="messages", messages=messages, model=model if isinstance(model, str) else None,
                tools=tools if isinstance(tools, list) and tools else None, finish=finish)


def _messages(rows: list[Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """The rows that are messages, in OpenAI's chat shape, and how each assistant turn ended."""
    messages: list[dict[str, Any]] = []
    finish: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in ROLES:
            continue
        role = row["role"]
        content = row.get("content")
        message: dict[str, Any] = {"role": role, "content": "" if content is None and role != "assistant" else content}
        if role == "assistant":
            calls = [_call(call) for call in row.get("tool_calls") or [] if isinstance(call, dict)]
            if calls:
                message["tool_calls"] = calls
            elif content is None:
                message["content"] = ""
            reasoning = row.get("reasoning_content") or row.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                message["reasoning_content"] = reasoning
            if isinstance(row.get("finish_reason"), str):
                finish[len(messages)] = row["finish_reason"]
        elif role == "tool":
            message["tool_call_id"] = row.get("tool_call_id")
            # Hermes sends the tool's name with its result; its export keeps it as `tool_name`
            # (found comparing an import with the same run recorded through the proxy).
            name = row.get("name") or row.get("tool_name")
            if isinstance(name, str) and name:
                message["name"] = name
        messages.append(message)
    return messages, finish


def _call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", "")
    return {
        "id": call.get("id"), "type": "function",
        "function": {"name": function.get("name"),
                     "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)},
    }


# --- the skills in it ----------------------------------------------------------------------------


@dataclass(slots=True)
class FoundSkill:
    """A skill the conversation shows: where it came in, and the text the model read."""

    name: str
    #: preloaded (in the system prompt), tool (loaded by a call), or frontmatter (found by its look).
    source: str
    text: str
    #: Its message's index in the conversation.
    message: int
    #: The first step whose request holds it; None when no assistant turn came after it.
    step: int | None

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "message": self.message, "step": self.step,
                "chars": len(self.text)}


def find_skills(chat: Chat, harness: str) -> list[FoundSkill]:
    """Every skill the conversation shows, once each, where it first came in."""
    from tuningfork.harness import ADAPTERS

    spans = ADAPTERS[harness]().skill_spans({"messages": chat.messages})
    found: dict[str, FoundSkill] = {}
    steps = chat.steps
    for span in spans or _frontmatter(chat.messages):
        if span.name in found:
            continue
        step = next((number for number, index in enumerate(steps) if index > span.message_index), None)
        found[span.name] = FoundSkill(span.name, span.source, span.text, span.message_index, step)
    return list(found.values())


def _frontmatter(messages: list[dict[str, Any]]) -> list[Any]:
    from tuningfork.harness.base import SkillSpan

    spans = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if message["role"] == "assistant" or not isinstance(content, str):
            continue
        for match in FRONTMATTER.finditer(content):
            spans.append(SkillSpan(name=match["name"].strip().strip("'\""), message_index=index,
                                   source="frontmatter", text=content[match.start():]))
            break
    return spans


# --- the import ----------------------------------------------------------------------------------


@dataclass(slots=True)
class ImportPlan:
    """What an import will record, shown before it does. It keeps every skill the conversation
    shows; which one an A/B, a refinement or a scenario studies is named when it starts."""

    chat: Chat
    label: str
    task_id: str | None
    harness: str
    skills: list[FoundSkill]
    #: The folders the person named, by the skill each holds: kept in place of the SKILL.md the
    #: model read, its scripts and references with it.
    folders: dict[str, Path] = field(default_factory=dict)
    #: The skill studied where none is named: the one the person named, or the only one. None
    #: where the conversation shows several and none was named.
    study: str | None = None
    #: What the file lacked or the import cannot do: a caution, each.
    warnings: list[str] = field(default_factory=list)
    #: What is so and needs no care: whose answers they are, how many skills it keeps.
    notes: list[str] = field(default_factory=list)
    #: Where the tools came from, when the file named none and a recorded run's were taken.
    tools_from: str | None = None

    @property
    def names(self) -> list[str]:
        """Every skill it keeps: those the conversation shows, as they came in, then any folder
        named for a skill it does not show."""
        shown = [skill.name for skill in self.skills]
        return shown + [name for name in self.folders if name not in shown]

    def to_json(self) -> dict[str, Any]:
        chat = self.chat
        shown = {skill.name: skill for skill in self.skills}
        return {
            "format": chat.format, "session_id": chat.session_id, "title": chat.title, "model": chat.model,
            "harness": self.harness, "label": self.label, "task_id": self.task_id,
            "messages": len(chat.messages), "steps": len(chat.steps), "system_prompt": chat.system_prompt,
            "tools": len(chat.tools or []), "tools_from": self.tools_from,
            "skills": [
                {**(shown[name].to_json() if name in shown else {"name": name, "source": None, "message": None,
                                                                   "step": None, "chars": 0}),
                 "folder": str(self.folders[name]) if name in self.folders else None}
                for name in self.names
            ],
            "skill": self.study,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def prepare_import(
    store: Store,
    data: bytes | str,
    *,
    session: str | None = None,
    skill: str | None = None,
    skill_dir: str | Path | Sequence[str | Path] | None = None,
    harness: str | None = None,
    label: str | None = None,
    task_id: str | None = None,
    tools_from: str | None = None,
) -> ImportPlan:
    """An import, checked and shown before anything is recorded: the conversation, the skills it
    shows — each kept, extracted from what the model read or from a folder the person names
    (`skill_dir`, one or several) — the one studied where none is named (`skill`), and what the
    file lacked."""
    from tuningfork.harness import ADAPTERS
    from tuningfork.paths import resolve_path

    chat = read_chat(data, session=session)
    harness = harness or chat.harness or "generic"
    if harness not in ADAPTERS:
        raise OperationError(f"no harness called {harness!r}", hint=f"one of {', '.join(ADAPTERS)}", field="harness")
    # The harness that had the conversation sends its history its own way: each step's request
    # is rebuilt as it would have sent it.
    chat.harness = harness
    if not chat.steps:
        raise OperationError("the conversation has no assistant turn: nothing the model did to study", field="file")
    warnings: list[str] = []
    notes: list[str] = []
    origin = None
    if not chat.tools and tools_from:
        chat.tools, origin = _tools_of(store, tools_from)
    if not chat.system_prompt:
        warnings.append("the file kept no system prompt: each step is rebuilt without one, which is not what the "
                        "model read")
    if not chat.tools:
        warnings.append("the file names no tools" + (" (Hermes's export keeps none)" if chat.format == "hermes-export" else "")
                        + ": each step is rebuilt without them; name a run recorded through the same harness to "
                        "take its tools from (--tools-from, or the import page's “Take its tools from”)")
    if chat.model:
        notes.append(f"its answers are {chat.model}'s: an experiment here samples your server's model, given "
                     "this conversation")

    skills = find_skills(chat, harness)
    folders: dict[str, Path] = {}
    for given in [skill_dir] if isinstance(skill_dir, (str, Path)) else list(skill_dir or []):
        folder = resolve_path(str(given)).path
        if not (folder / "SKILL.md").is_file():
            raise OperationError(f"{folder} holds no SKILL.md", field="skill_dir")
        held = _held_by(folder, skills)
        if held in folders:
            raise OperationError(f"two folders hold {held}: {folders[held]} and {folder}", field="skill_dir")
        folders[held] = folder
    plan = ImportPlan(chat=chat, label="", task_id=task_id or None, harness=harness, skills=skills, folders=folders,
                      warnings=warnings, notes=notes, tools_from=origin)
    names = plan.names
    if skill and skill not in names:
        raise OperationError(f"no skill called {skill!r} shows in the conversation",
                             hint=f"the skills in it: {', '.join(names) or 'none'}", field="skill")
    plan.study = skill or (names[0] if len(names) == 1 else None)

    if not names:
        warnings.append("no skill shows in the conversation: an A/B or a refinement needs one; name the skill's folder "
                        "(--skill-dir, or the import page's folder field)")
    for held, folder in folders.items():
        shown = next((found for found in skills if found.name == held), None)
        if shown is None:
            warnings.append(f"{held} does not show in the conversation: nothing at a step reads it")
        elif not _locatable(chat, shown, (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")):
            warnings.append(f"{folder}/SKILL.md is not the text the model read in the conversation: an A/B or a "
                            "refinement finds the skill by its text, and would not find it")
    for found in skills:
        if found.step is None:
            warnings.append(f"{found.name} came in after the last assistant turn: no step reads it")
    extracted = [found.name for found in skills if found.name not in folders]
    if extracted:
        warnings.append(f"the SKILL.md of {_listed(extracted)} is extracted from what the model read: the rest of "
                        "its folder — scripts, templates, references — the chat never showed; name the skill's folder "
                        "(--skill-dir, or the import page's folder field) to keep it whole")
    if plan.study is None and len(names) > 1:
        notes.append(f"it keeps {len(names)} skills, {_listed(names)}: an A/B, a refinement or a scenario names "
                     "the one it studies")
    name = chat.title or (f"session {chat.session_id}" if chat.session_id else "a chat")
    plan.label = label or f"imported: {name}"[:120]
    return plan


def _held_by(folder: Path, skills: list[FoundSkill]) -> str:
    """The skill a folder holds: the one of those shown that its SKILL.md's name, or its own
    name, matches — else that name."""
    match = FRONTMATTER.match((folder / "SKILL.md").read_text(encoding="utf-8", errors="replace"))
    names = [match["name"].strip().strip("'\"")] if match else []
    names.append(folder.name)
    shown = {skill.name for skill in skills}
    return next((name for name in names if name in shown), names[0])


def _listed(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _folder_name(skill: str) -> str:
    """A skill's name as a folder's: a name a harness was given can hold a slash."""
    return re.sub(r"[^\w.-]+", "-", skill).strip(".-") or "skill"


def _locatable(chat: Chat, skill: FoundSkill, text: str) -> bool:
    from tuningfork.experiment.skill import locate

    if skill.step is None:
        return False
    return bool(locate(chat.request(chat.steps[skill.step]), text))


def _tools_of(store: Store, reference: str) -> tuple[list[dict[str, Any]], str]:
    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}", field="tools_from")
    steps = store.steps(run.id)
    tools = store.rebuild_request(steps[0]).get("tools") if steps else None
    if not isinstance(tools, list) or not tools:
        raise OperationError(f"run {run.id[:8]}'s first call offered no tools", field="tools_from")
    return tools, run.id


@dataclass(slots=True)
class ImportResult:
    run_id: str
    steps: int
    #: The skill studied where none is named; None where it kept several and none was named.
    skill: str | None
    folder: Path
    warnings: list[str]
    #: Every skill it kept, as the conversation showed them.
    skills: list[str] = field(default_factory=list)
    #: What its plan said is so and needs no care, apart from its warnings.
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"run": self.run_id, "steps": self.steps, "skill": self.skill, "skills": self.skills,
                "folder": str(self.folder), "warnings": self.warnings, "notes": self.notes}


def execute_import(store: Store, plan: ImportPlan) -> ImportResult:
    """Record the conversation as a run, step by step, and keep beside it what it was imported
    from and every skill it shows (`parity.skill_dirs`), the one studied where none is named as
    the run's own (`parity.skill_snapshot_dir`). The run is its own from the first step: it never
    waits on the label a live call could take (`Store.begin_run`)."""
    from tuningfork.launches import Assignment
    from tuningfork.specs import hash_tree

    chat = plan.chat
    run_id = uuid.uuid4().hex
    folder = root(store.home) / run_id
    folder.mkdir(parents=True)
    shown = {skill.name: skill for skill in plan.skills}
    kept: dict[str, Path] = {}
    try:
        (folder / "chat.json").write_text(json.dumps(chat.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        skills = folder / "skills"
        for name in plan.names:
            place = skills / _folder_name(name)
            if name in plan.folders:
                shutil.copytree(plan.folders[name], place, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            else:
                place.mkdir(parents=True)
                (place / "SKILL.md").write_text(shown[name].text, encoding="utf-8")
            kept[name] = place
        for number, index in enumerate(chat.steps):
            store.record_step(
                request=chat.request(index), response=chat.response(index), harness=plan.harness,
                assignment=Assignment(launch_id=None, run_id=run_id, idx=number, thread=0, label=plan.label,  # type: ignore[arg-type]
                                      task_id=plan.task_id, role=None),  # type: ignore[arg-type]
            )
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    study = plan.study
    imported = {
        "format": chat.format, "session_id": chat.session_id, "title": chat.title, "model": chat.model,
        "at": time.time(), "tools_from": plan.tools_from,
        # Each skill kept, and the folder it was copied from — the skill a scenario made from it
        # tunes, while it is there — or none, its SKILL.md extracted from what the model read.
        "skills": [{"name": name, "source": "folder" if name in plan.folders else "extracted",
                    "dir": str(plan.folders[name]) if name in plan.folders else None} for name in plan.names],
        "skill": study,
        # The studied one's, as an import kept before it kept several: what older readers look for.
        "skill_source": ("folder" if study in plan.folders else "extracted") if study else None,
        "skill_dir": str(plan.folders[study]) if study in plan.folders else None,
        "warnings": plan.warnings,
    }
    parity: dict[str, Any] = {"imported": imported, "skill_dirs": {name: str(place) for name, place in kept.items()}}
    if study:
        parity["skill_snapshot_dir"] = str(kept[study])
    store.annotate_run(
        run_id, harness=plan.harness, task_id=plan.task_id, label=plan.label,
        skill_hashes=hash_tree([skills]) if kept else None, parity=parity,
    )
    return ImportResult(run_id=run_id, steps=len(chat.steps), skill=study, folder=folder, warnings=plan.warnings,
                        skills=list(kept), notes=plan.notes)


# --- a scenario from a run -------------------------------------------------------------------------

#: Where a scenario keeps a skill of its own, beside its spec — the copy a run kept, or the text
#: its model read — when no folder of the skill's own was found.
SKILL_FOLDER = "skill"
#: What a scenario decides itself, never taken from the task a launched run was of.
OWN_OPTIONS = ("check", "context", "stop_at_next_skill", "held_out")
#: A tool call's arguments that name a file.
PATH_ARGUMENTS = ("path", "file_path", "filepath", "filename", "file", "notebook_path")
#: What a check reads from its task's folder: `$TUNINGFORK_SPEC_DIR/<path>`.
SPEC_DIR_PATH = re.compile(r"\$\{?TUNINGFORK_SPEC_DIR\}?/([^\s\"'`;|&()<>]+)")
#: What a scenario's runs do with each step of the conversation from its prompt on: take it as
#: done (in their context, as the chat had it), make it first or as they go on — each its own
#: answer, the chat's path only one — or never reach it: past where each run stops, or after the
#: person's next turn, which no run is sent.
STEP_PARTS = ("done", "first", "live", "past", "later")


def chat_of(store: Store, run: Any, harness: str | None = None) -> Chat:
    """A run's conversation: an imported run's as the import kept it beside the run; any other's
    as its recording holds it — its last step's request as the model was sent it (where a
    scenario's run holds the conversation it continued, which its harness never sent) and that
    step's answer as the harness was given it. `harness`: the one that had it, where the run does
    not say."""
    from tuningfork.harness import ADAPTERS
    from tuningfork.wire import get_wire

    if harness is not None and harness not in ADAPTERS:
        raise OperationError(f"no harness called {harness!r}", hint=f"one of {', '.join(ADAPTERS)}", field="harness")
    path = root(store.home) / run.id / "chat.json"
    if (run.parity or {}).get("imported") and path.is_file():
        kept = json.loads(path.read_text(encoding="utf-8"))
        return Chat(format=kept["format"], messages=kept["messages"], session_id=kept.get("session_id"),
                    title=kept.get("title"), model=kept.get("model"), tools=kept.get("tools"),
                    harness=harness or kept.get("harness"), started_at=kept.get("started_at"),
                    finish={int(index): reason for index, reason in (kept.get("finish") or {}).items()})
    steps = store.steps(run.id)
    if not steps:
        raise OperationError(f"run {run.id[:8]} has no step: no conversation to start from", field="run")
    last = steps[-1]
    wire = get_wire(last.wire)
    request = store.rebuild_request(last)
    messages = [dict(message) for message in wire.view(request)
                if isinstance(message, dict) and isinstance(message.get("role"), str)]
    answer = ((store.delivered_response(last) or {}).get("choices") or [{}])[0].get("message")
    if isinstance(answer, dict):
        messages.append(sent({"role": "assistant", **{key: value for key, value in answer.items()
                                                      if key in ("content", "tool_calls")}}))
    tools = request.get("tools")
    chat = Chat(format="recorded", messages=messages, model=run.model, started_at=run.started_at,
                tools=tools if isinstance(tools, list) else None)
    chat.harness = harness or run.harness or _guessed_harness(chat, wire.name)
    # Each step's answer comes after the messages its call held. A scenario's run had its
    # conversation spliced into every call (OpenAI chat only), whose turns are none of its steps.
    if all(get_wire(step.wire).name == "openai-chat" for step in steps):
        depths = store.node_depths([step.request_node_id for step in steps])
        chat.numbers = {depths[step.request_node_id] + 1: step.idx for step in steps if step.request_node_id in depths}
    return chat


def _guessed_harness(chat: Chat, wire: str) -> str:
    """The harness a run recorded as it happened came from, where it does not say: by its wire,
    and by the tool its skills load with."""
    if wire == "anthropic-messages":
        return "claude-code"
    called = {call.get("function", {}).get("name") for message in chat.messages for call in message.get("tool_calls") or []}
    return "hermes" if "skill_view" in called else "generic"


def turns(chat: Chat) -> list[dict[str, Any]]:
    """Each person's turn of the conversation — a user message the harness wrote itself is not
    one (`harness_note`) — its message's index, its first line, and the step that answers it:
    where a scenario could start, the skill it studies allowing (`Start.starts`)."""
    note = _adapter_method(chat.harness, "harness_note")
    steps = chat.steps
    found = []
    for index, message in enumerate(chat.messages):
        if message["role"] != "user" or (note is not None and note(message)):
            continue
        words = _words(message, note).strip()
        found.append({"message": index, "text": words.splitlines()[0][:160] if words else "",
                      "step": next((number for at in steps if at > index for number in [chat.step_of(at)]
                                    if number is not None), None)})
    return found


def _adapter_method(harness: str | None, name: str) -> Any:
    from tuningfork.harness import ADAPTERS

    adapter = ADAPTERS.get(harness or "")
    return getattr(adapter(), name, None) if adapter is not None else None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _words(message: dict[str, Any], note: Any = None) -> str:
    """A person's turn in their own words: without a part the harness put beside them
    (`harness_note`: Claude Code's reminders, which it writes again in every run)."""
    content = message.get("content")
    if isinstance(content, list) and note is not None:
        content = [part for part in content if not (isinstance(part, dict) and isinstance(part.get("text"), str)
                                                    and note({"role": "user", "content": part["text"]}))]
    return _text(content)


def _bare(name: str) -> str:
    """A skill's name without the category a harness may give before it."""
    return name.strip().rsplit("/", 1)[-1]


def _key(name: str) -> str:
    """A skill's name as two calls may give it: a category before it, or another case."""
    return _bare(name).lower()


@dataclass(slots=True)
class ScenarioPlan:
    """A scenario made from a run, shown before it is written: the conversation it continues, its
    prompt, the skill folder, the check and the input files — and where its runs stop, when the
    conversation went on to another skill."""

    run_id: str
    name: str
    #: The user turn it starts from, by its message's index: the scenario's prompt.
    cut: int
    prompt: str
    context: Context
    skill: str
    skill_dir: Path
    check: Any
    #: What a person should look at before making it: a caution, each.
    warnings: list[str] = field(default_factory=list)
    #: What is so and needs no care: where its check and its inputs come from, where each run
    #: starts and stops.
    notes: list[str] = field(default_factory=list)
    #: The skill the conversation loaded next after the one studied, and the step whose answer
    #: loaded it: each run stops there (`tuningfork.stop_at_next_skill`). None where none did.
    next_skill: dict[str, Any] | None = None
    #: The skill came preloaded, in the harness's instructions: each run preloads it too.
    preload: bool = False
    #: Its input files (`inputs/`), each by its path in the workspace: a launched run's workspace
    #: where the scenario starts, from its snapshots.
    files: dict[str, bytes] = field(default_factory=dict)
    #: A folder copied in as the scenario's own skill (`skill/<name>/`), where no folder of the
    #: skill's own was found: the copy the run kept.
    copy_from: Path | None = None
    #: The SKILL.md written into the scenario's own skill folder: the text the model read, where
    #: the run kept no copy, or a copy's, given the frontmatter its harness had not shown.
    extract: str | None = None
    #: The harness that had the conversation: the one its runs are launched with.
    harness: str | None = None
    #: The eval a launched run was given (`Launch.eval`): its expected output, expectations and
    #: options, kept in the scenario but for what a scenario decides itself (`OWN_OPTIONS`).
    task: dict[str, Any] = field(default_factory=dict)
    #: What its check reads from its task's folder (`$TUNINGFORK_SPEC_DIR`), by its place beside
    #: the scenario's spec: copied there, where no run sees it.
    check_files: dict[str, Path] = field(default_factory=dict)
    #: The conversation's steps from its prompt on, as the chat went, each with what a run does
    #: with it (`Start.steps`).
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def own_skill(self) -> bool:
        """Whether the scenario keeps its skill in its own folder."""
        return self.copy_from is not None or self.extract is not None

    def spec(self) -> dict[str, Any]:
        """The scenario's evals.json, every field written as the Scenarios page writes them. Its
        eval is named by the scenario, so its runs are a task of their own (`skill/<scenario>`),
        never counted with the task of the run it came from."""
        options: dict[str, Any] = {"normalizers": [], "held_out": False, "skills": [], "timeout": 900, "max_turns": 30}
        options.update({key: value for key, value in (self.task.get("tuningfork") or {}).items() if key not in OWN_OPTIONS})
        if self.preload and self.skill not in options["skills"]:
            options["skills"] = [*options["skills"], self.skill]
        if self.context.messages:
            options["context"] = "context.json"
        if self.next_skill is not None:
            options["stop_at_next_skill"] = True
        if self.check is not None:
            options = {"check": self.check, **options}
        files = [{"path": path, "source": f"inputs/{path}"} for path in sorted(self.files)]
        skill_dir = f"{SKILL_FOLDER}/{self.skill_dir.name}" if self.own_skill else str(self.skill_dir)
        return {"skill_name": self.skill, "tuningfork": {"skill_dir": skill_dir},
                "evals": [{"id": self.name, "prompt": self.prompt, "expected_output": self.task.get("expected_output") or "",
                           "files": files, "expectations": list(self.task.get("expectations") or []),
                           "tuningfork": options}]}

    def to_json(self) -> dict[str, Any]:
        return {"run": self.run_id, "name": self.name, "cut": self.cut, "prompt": self.prompt,
                "context": len(self.context.before), "in_turn": len(self.context.in_turn), "next_skill": self.next_skill,
                "skill": self.skill, "skill_dir": str(self.skill_dir), "own_skill": self.own_skill,
                "check": self.check, "check_files": sorted(self.check_files), "harness": self.harness,
                "files": sorted(self.files), "steps": self.steps, "spec": self.spec(), "warnings": self.warnings,
                "notes": self.notes}


@dataclass(slots=True)
class Start:
    """Where a scenario of a conversation starts, and where its runs stop."""

    cut: int
    context: Context
    prompt: str
    #: The skill the conversation loaded next after the one studied, and the step whose answer
    #: loaded it; None where none did.
    next_skill: dict[str, Any] | None = None
    #: The studied skill came in before the turn it starts from: in its context as the chat had
    #: it, where a variant never reaches it.
    skill_before: bool = False
    #: The studied skill came preloaded, in the harness's instructions.
    preloaded: bool = False
    #: The studied skill shows in the conversation at all.
    came_in: bool = True
    #: The conversation's answer that each run's first stands for, by its message's index: the
    #: first after the prompt and the steps taken as done. None where the conversation has none.
    live: int | None = None
    #: The person's turns a scenario can start from, by their messages' indexes: the one the
    #: skill came in after — every one, where it never came in or came with the harness's own
    #: instructions — so that it comes in live wherever a run starts.
    starts: list[int] = field(default_factory=list)
    #: The conversation's steps from the prompt on, as the chat went: each by the run's step
    #: (`step`; None for a message its calls carried that was none of its steps), what the model
    #: did there as a run's page names it (`label`), what each run does with it (`part`, one of
    #: `STEP_PARTS`), and `stop` on the answer that loaded the skill each run stops at.
    steps: list[dict[str, Any]] = field(default_factory=list)


def scenario_start(chat: Chat, skill: str | None, harness: str, cut: int | None = None) -> Start:
    """Where a scenario from this conversation starts: a person's turn `cut` (by default the last
    one before the skill first came in, or the first where it never did) as the prompt, the
    conversation before it as each run's context — as its harness sends history, without the
    instructions it brings itself — and, when the skill came in later in that same turn, the
    model's steps of the turn before it (`Context.in_turn`): done already in each run, an earlier
    skill's text and results among them, so the skill itself comes in live. Where the
    conversation went on to load another skill, each run stops there (`Start.next_skill`). A turn
    where the skill could not come in live is refused: one before the turn it came in, whose
    words no run is sent, or one after, where it is in each run's context as the chat had it.
    `skill` None: one the conversation never shows, its folder not named yet."""
    from tuningfork.specs import SpecError, check_context

    people = [entry["message"] for entry in turns(chat)]
    found = None if skill is None else next(
        (each for each in find_skills(chat, harness) if _key(each.name) == _key(skill)), None)
    load = _load_of(chat, found, harness) if found is not None else None
    came = load if load is not None else (found.message if found is not None else None)
    # In the harness's instructions, however it was found there: each run's harness brings it.
    preloaded = found is not None and (found.source == "preloaded" or chat.messages[found.message]["role"] == "system")
    before = [index for index in people if came is not None and index < came]
    starts = people if found is None or preloaded else (before[-1:] or people[:1])
    if cut is None:
        cut = starts[0] if starts else None
    if cut is None or cut not in people:
        raise OperationError(f"message {cut} is not a user turn of the conversation", field="cut",
                             hint=f"the user turns, by message: {', '.join(map(str, people)) or 'none'}")
    if cut not in starts:
        raise OperationError(_not_live(skill, cut, starts[0]), field="cut",
                             hint=f"start from message {starts[0]}, where {skill} comes in live")
    after = next((index for index in people if index > cut), len(chat.messages))
    in_turn = chat.messages[cut + 1:load] if load is not None and cut < load < after else []
    # The proxy puts it in after the harness has made its own copy ready: it goes as the harness
    # would have sent it — without the instructions it writes itself, Claude Code's among the
    # messages too, which each run's harness writes again.
    context = Context(as_sent(harness, [message for message in chat.messages[:cut] if message["role"] != "system"]),
                      as_sent(harness, [message for message in in_turn if message["role"] != "system"]))
    prompt = _words(chat.messages[cut], _adapter_method(harness, "harness_note"))
    if not prompt.strip():
        raise OperationError(f"the user turn at message {cut} holds no text to start from", field="cut")
    if context.messages:
        try:
            check_context(context.to_json() if context.in_turn else context.before)
        except SpecError as error:
            raise OperationError(f"the conversation before message {cut}: {error}", field="cut") from None
    live = load if in_turn else next((index for index in chat.steps if index > cut), None)
    loads = _next_load(chat, harness, skill or "", came if came is not None else cut)
    return Start(cut=cut, context=context, prompt=prompt,
                 next_skill={"skill": loads[0], "step": chat.step_of(loads[1])} if loads else None,
                 skill_before=came is not None and came < cut and not preloaded, preloaded=preloaded,
                 came_in=found is not None, live=live, starts=starts,
                 steps=_parts(chat, cut, after, live, loads[1] if loads else None))


def _not_live(skill: str, cut: int, start: int) -> str:
    """Why a run starting from the person's turn at message `cut` would never vary the skill,
    which came in after their turn at `start`."""
    if cut < start:
        return (f"{skill} came in after message {start}, which a run starting from message {cut} is never sent: the "
                "skill would come in only if the model loaded it unasked")
    return (f"{skill} came in before message {cut}: a run starting there has it in its context as the chat had it, "
            "where a variant never reaches it")


def _parts(chat: Chat, cut: int, after: int, live: int | None, stop: int | None) -> list[dict[str, Any]]:
    """The conversation's steps from the prompt at message `cut` on, as the chat went, each with
    what a run does with it (`Start.steps`): the person's next turn is at `after`, each run's
    first call stands for the answer at `live`, and each stops after one loading another skill,
    as the answer at `stop` did."""
    from tuningfork.analysis.normalize import outcome_for

    shown = []
    for index in chat.steps:
        if index <= cut:
            continue
        if index > after:
            part = "later"
        elif stop is not None and index > stop:
            part = "past"
        elif live is not None and index < live:
            part = "done"
        else:
            part = "first" if index == live else "live"
        shown.append({"step": chat.step_of(index), "message": index, "part": part,
                      "label": outcome_for(chat.response(index)).label,
                      "stop": index == stop and part in ("first", "live")})
    return shown


def _load_of(chat: Chat, found: FoundSkill, harness: str) -> int | None:
    """The answer whose call loaded a skill the conversation shows: the last assistant message
    before its text that loads it by name, or that made the call its result answers. None for
    a skill that was not loaded by a call."""
    if found.source != "tool":
        return None
    named = _adapter_method(harness, "skill_load")
    answered = chat.messages[found.message].get("tool_call_id")
    for index in range(found.message - 1, -1, -1):
        message = chat.messages[index]
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if (named is not None and named(call) == found.name) or (answered and call.get("id") == answered):
                return index
    return None


def _next_load(chat: Chat, harness: str, skill: str, after: int) -> tuple[str, int] | None:
    """The first skill other than `skill` the conversation's answers load after message
    `after`, and the answer that loads it, by its message's index."""
    named = _adapter_method(harness, "skill_load")
    if named is None:
        return None
    for index in chat.steps:
        if index <= after:
            continue
        for call in chat.messages[index].get("tool_calls") or []:
            loaded = named(call)
            if loaded and _key(loaded) != _key(skill):
                return loaded, index
    return None


@dataclass(slots=True)
class Studied:
    """The skill a scenario of a run studies: its name — as its harness loads it, where the
    conversation shows it — where the conversation shows it, and the SKILL.md the run kept of it,
    by the name it kept it under."""

    name: str
    found: FoundSkill | None = None
    kept: Path | None = None
    kept_name: str | None = None

    @property
    def shown_as(self) -> str:
        """Its name as the conversation shows it: what tells where it came in."""
        return self.found.name if self.found is not None else self.name


def skill_names(run: Any, chat: Chat, harness: str) -> list[str]:
    """Every skill a scenario of the run can study: each it kept a copy of, then each its
    conversation shows besides."""
    from tuningfork.ops.experiments import kept_skills

    kept = kept_skills(run)
    named = {_key(name) for name in kept}
    return [*kept, *(found.name for found in find_skills(chat, harness) if _key(found.name) not in named)]


def _studied(run: Any, chat: Chat, harness: str, skill: str | None = None, folder: Path | None = None) -> Studied:
    """The skill a scenario of `run` studies: `skill`, else the one `folder` holds, else the run's
    own (`experiments.study_skill`: a batch run's, the one an import studies, the only one it
    kept), else the only one its conversation shows. Named as its harness loads it, where the
    conversation shows it: a run that stops at the next skill tells the two apart by name."""
    from tuningfork.ops.experiments import kept_skills, study_skill

    shown = find_skills(chat, harness)
    kept = kept_skills(run)

    def keys(name: str, path: Path | None = None) -> set[str]:
        named = {_key(name)}
        match = FRONTMATTER.match(path.read_text(encoding="utf-8", errors="replace")) if path is not None else None
        if match:
            named.add(_key(match["name"].strip().strip("'\"")))
        return named

    def shown_as(name: str, path: Path | None = None) -> FoundSkill | None:
        wanted = keys(name, path)
        return next((found for found in shown if _key(found.name) in wanted), None)

    if folder is not None and not skill:
        skill = _held_by(folder, shown)
    if skill:
        own = next(((name, path) for name, path in kept.items() if _key(skill) in keys(name, path)), None)
        found = shown_as(skill) or (shown_as(*own) if own else None)
        if own is None and found is None and folder is None:
            raise OperationError(f"run {run.id[:8]} shows no skill called {skill!r}", field="skill",
                                 hint=f"the skills in it: {', '.join(skill_names(run, chat, harness)) or 'none'}")
        named = _bare(found.name) if found else (own[0] if own else _bare(skill))
        return Studied(named, found, own[1] if own else None, own[0] if own else None)
    mine = study_skill(run)  # refused where it kept several and none is named
    if mine is not None:
        found = shown_as(*mine)
        return Studied(_bare(found.name) if found else mine[0], found, mine[1], mine[0])
    if len(shown) == 1:
        return Studied(_bare(shown[0].name), shown[0])
    if not shown:
        raise OperationError(f"no skill shows in run {run.id[:8]}'s conversation, and it kept none: a scenario studies "
                             "one — name its folder (--skill-dir, or “Its folder” on the scenario's form)", field="skill_dir")
    raise OperationError(f"run {run.id[:8]}'s conversation shows {len(shown)} skills: name the one the scenario studies",
                         hint=f"the skills in it: {', '.join(found.name for found in shown)}", code="choose_skill",
                         field="skill")


def _scenario_run(store: Store, reference: str) -> Any:
    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}", code="not_found", field="run")
    if run.is_thread:
        raise OperationError(f"run {run.id[:8]} is a {run.role} thread of run {(run.parent_run_id or '?')[:8]}: its "
                             "calls are its own, never a task's — make the scenario of the run it belongs to", field="run")
    return run


def scenario_choices(store: Store, reference: str, *, skill: str | None = None, harness: str | None = None,
                     cut: int | None = None) -> dict[str, Any]:
    """What a page offers before a scenario of a run is made: the skills it kept or showed and the
    one studied unless another is chosen; the person's turns, each `offered` where the skill would
    come in live in a run starting there; the one it starts from — `cut`, else the turn before
    the skill came in — its words, the starting prompt unless the person writes another, and the
    conversation's steps from there on, each with what a run does with it; where its runs stop;
    and the check its task had, for a run tuningfork launched. Beside them, what the chat's calls
    held before the prompt, for the whole of what each run is given: its system prompt and tools
    — never sent, each run's harness bringing its own — and the conversation before it, in brief."""
    run = _scenario_run(store, reference)
    chat = chat_of(store, run, harness)
    harness = chat.harness or "generic"
    unseen = False
    try:
        studied: Studied | None = _studied(run, chat, harness, skill)
    except OperationError as error:
        if error.code != "choose_skill" and error.field != "skill_dir":
            raise
        studied = None
        unseen = error.field == "skill_dir"  # none shows: the one a folder will name never came in
    try:
        start = (scenario_start(chat, studied.shown_as, harness, cut) if studied is not None
                 else scenario_start(chat, None, harness, cut) if unseen else None)
    except OperationError:
        if cut is not None:
            raise  # a turn asked for: why it cannot start there
        start = None
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    task = launch.eval if launch is not None and isinstance(launch.eval, dict) else {}
    offered = set(start.starts) if start else set()
    return {
        "run": run.id, "harness": harness, "recorded_harness": run.harness,
        "imported": bool((run.parity or {}).get("imported")), "launched": launch is not None,
        "skill": studied.name if studied is not None else None, "skills": skill_names(run, chat, harness),
        "turns": [{**turn, "offered": turn["message"] in offered} for turn in turns(chat)],
        "cut": start.cut if start else None, "prompt": start.prompt if start else None,
        "steps": start.steps if start else [],
        "context": len(start.context.before) if start else 0, "in_turn": len(start.context.in_turn) if start else 0,
        "next_skill": start.next_skill if start else None, "came_in": start.came_in if start else None,
        "preloaded": bool(start and start.preloaded),
        "system": _text(chat.messages[0].get("content")) if chat.system_prompt else None,
        "tools": chat.tools or [],
        "before": [_gist(message) for message in start.context.before] if start else [],
        "check": (task.get("tuningfork") or {}).get("check"),
    }


def _gist(message: dict[str, Any]) -> dict[str, Any]:
    """A message of the conversation before a prompt, on one line: an answer's calls as a run's
    page names a step's, any other message the first line of what it says."""
    from tuningfork.analysis.normalize import outcome_for

    if message.get("role") == "assistant" and message.get("tool_calls"):
        said = outcome_for({"choices": [{"message": message, "finish_reason": "tool_calls"}]}).label
    else:
        text = _text(message.get("content")).strip()
        said = text.splitlines()[0][:160] if text else ""
    return {"role": message.get("role"), "said": said}


def prepare_scenario(
    store: Store, reference: str, name: str, *, cut: int | None = None, check: Any = None, skill: str | None = None,
    skill_dir: str | Path | None = None, harness: str | None = None, task_check: bool = True,
    prompt: str | None = None,
) -> ScenarioPlan:
    """A scenario of a run — imported, launched by tuningfork, or recorded as it happened —
    checked and shown before anything is written. The conversation before the person's turn
    `cut` is the context each run continues — by default before the turn that led to the skill,
    which then comes in live, through the harness, as a variant would — and that turn is the
    prompt, in the person's words, or `prompt` (a file the chat named where the scenario's inputs
    put it, say). The skill, `skill` where the run kept or showed several: in `skill_dir`, else the
    folder an import named, else its task's own (a launched run's spec), else a copy in the
    scenario's own folder — the one the run kept, or the text its model read. Its inputs: a
    launched run's workspace where the scenario starts, from its snapshots, else its task's input
    files; for any other run, the files its conversation names, for a person to add. Its check:
    `check`, else a launched run's task's own (unless `task_check` is False), with what it reads
    from the task's folder."""
    from tuningfork.harness import ADAPTERS
    from tuningfork.ops import scenarios
    from tuningfork.paths import resolve_path
    from tuningfork.specs import check_problems
    from tuningfork.wire import get_wire

    run = _scenario_run(store, reference)
    scenarios.check_name(name)
    place = scenarios.folder(store.home, name)
    if (place / scenarios.SPEC).exists():
        raise OperationError(f"a scenario called {name!r} exists already", code="conflict", field="name")
    if place.exists():
        raise OperationError(f"{place} is there already, holding no scenario: move it away to use the name",
                             code="conflict", field="name")
    chat = chat_of(store, run, harness)
    harness = chat.harness or "generic"
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    task = launch.eval if launch is not None and isinstance(launch.eval, dict) else {}
    folder = None
    if skill_dir:
        folder = resolve_path(str(skill_dir)).path
        if not (folder / "SKILL.md").is_file():
            raise OperationError(f"{folder} holds no SKILL.md", field="skill_dir")
    studied = _studied(run, chat, harness, skill, folder)
    from_task = check is None and task_check and (task.get("tuningfork") or {}).get("check") is not None
    if from_task:
        check = task["tuningfork"]["check"]
    problems = check_problems(check, "check")
    if problems:
        raise OperationError(f"the check {problems[0].message}", field="check")
    start = scenario_start(chat, studied.shown_as, harness, cut)
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise OperationError("the starting prompt is empty: each run is sent it as the person's turn", field="prompt")
        start.prompt = prompt
    if start.context.messages:
        speaks = getattr(ADAPTERS[harness](), "wire", "openai-chat")
        probe = {"messages": [{"role": "user", "content": start.prompt}]}
        if get_wire(speaks).with_context(probe, start.context.before, start.context.in_turn) is None:
            what = (f"the conversation before message {start.cut}" if start.context.before
                    else f"the model's steps of that turn before {studied.name} came in")
            raise OperationError(f"{harness}'s calls ({speaks}) cannot carry {what} yet: a scenario of its run starts "
                                 "at its first turn, before the model did anything", field="cut")
    warnings: list[str] = []
    notes: list[str] = []

    own = place / SKILL_FOLDER / _folder_name(studied.name)
    copy_from: Path | None = None
    extract: str | None = None
    source = imported_source(run, studied.kept_name) if studied.kept_name is not None else None
    task_folder = _task_skill_dir(run, launch, studied.kept)
    if folder is not None:
        skill_folder = folder
        text = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        if studied.found is not None and not _locatable(chat, studied.found, text):
            warnings.append(f"{folder}/SKILL.md is not the text the model read in the conversation: the scenario's "
                            "runs read it as the folder has it")
    elif source is not None:
        skill_folder = source
    elif task_folder is not None:
        skill_folder = task_folder
        if studied.kept is not None and (task_folder / "SKILL.md").read_bytes() != studied.kept.read_bytes():
            warnings.append(f"{task_folder}/SKILL.md has changed since run {run.id[:8]} was given it: the scenario's "
                            "runs read it as it is now")
    elif studied.kept is not None or studied.found is not None:
        skill_folder = own
        if studied.kept is not None:
            copy_from = studied.kept.parent
            text = studied.kept.read_text(encoding="utf-8", errors="replace")
        else:
            text = studied.found.text if studied.found is not None else ""
        written, made_up = _skill_file(text, studied.name, chat)
        if made_up or copy_from is None:
            extract = written
        what = (f"the copy run {run.id[:8]} kept" if copy_from is not None
                else "the SKILL.md the model read — the rest of its folder, scripts and references, the chat never showed")
        warnings.append(f"no folder of {studied.name}'s own was found: the scenario keeps {what}, in its own folder "
                        f"({own}), where an approved edit is written; name the skill's folder (--skill-dir, or “Its folder” "
                        "on the scenario's form) to tune that one")
        if made_up:
            notes.append(f"the model read {studied.name} without its frontmatter: the scenario's SKILL.md is given "
                         "one, its name and a description, so that the harness lists it")
    else:
        raise OperationError(f"no copy of {studied.name} to give the scenario's runs: name its folder (--skill-dir, "
                             "or “Its folder” on the scenario's form)",
                             field="skill_dir")

    check_files: dict[str, Path] = {}
    if from_task:
        spec_dir = Path(task["spec_dir"]) if task.get("spec_dir") else None
        check_files, unread = _check_files(check, spec_dir)
        notes.append(f"its check is task {run.task_id or '-'}'s own, as run {run.id[:8]} was launched with it")
        if check_files:
            notes.append(f"its check reads {', '.join(sorted(check_files))} from the task's folder ({spec_dir}): "
                         "copied beside the scenario's spec, where no run sees it")
        if unread:
            warnings.append(f"its check reads {', '.join(unread)} from the task's folder, which the scenario does not "
                            "copy: put what it needs beside the scenario's spec")
    cautions, said = start_notes(start, studied.name, check)
    warnings.extend(cautions)
    notes.extend(said)

    found_files, cautions, said = _inputs(store, run, launch, chat, start)
    files: dict[str, bytes] = {}
    for path, data in found_files.items():
        try:
            scenarios.input_path(store.home, name, path)
        except OperationError:
            cautions.append(f"{path} cannot be one of a scenario's inputs, by its name: left out")
            continue
        files[path] = data
    warnings.extend(cautions)
    notes.extend(said)
    if (launch is not None and not launch.settings.get("sandbox") and launch.workspace and start.context.messages
            and launch.workspace in json.dumps(start.context.to_json(), ensure_ascii=False)):
        warnings.append(f"the conversation names run {run.id[:8]}'s own workspace, {launch.workspace}: each run of the "
                        "scenario works in a folder of its own, the same in every run only in the sandbox")
    if (task.get("tuningfork") or {}).get("held_out"):
        warnings.append(f"run {run.id[:8]} is of a held-out eval, which judges edits: the scenario is not held out, and "
                        "an edit it shapes was shaped by what the held-out eval was to judge")
    return ScenarioPlan(run_id=run.id, name=name, cut=start.cut, prompt=start.prompt, context=start.context,
                        skill=studied.name, skill_dir=skill_folder, check=check, warnings=warnings, notes=notes,
                        next_skill=start.next_skill, preload=start.preloaded, files=files, copy_from=copy_from,
                        extract=extract, harness=harness, task=task, check_files=check_files, steps=start.steps)


def start_notes(start: Start, skill: str, check: Any) -> tuple[list[str], list[str]]:
    """What a scenario starting there should say before it is made: what no run will have, a
    caution each, then what it takes as done and where its runs stop."""
    cautions, notes = [], []
    if check is None:
        cautions.append("no check: its runs have no verdict until it has one (the scenario's page, tuningfork.check)")
    if not start.came_in:
        cautions.append(f"{skill} never came into the conversation: each run starts from the turn chosen, and may not "
                        "load it either")
    if start.context.in_turn:
        done = [entry["step"] for entry in start.steps if entry["part"] == "done" and entry["step"] is not None]
        first = next((entry["step"] for entry in start.steps if entry["part"] == "first"), None)
        notes.append((f"each run starts at step {first}, where the chat loaded {skill}, " if first is not None
                      else f"each run starts where the chat loaded {skill}, ")
                     + (f"with the turn's {named_steps(done)} taken as done" if done
                        else f"with the model's first {len(start.context.in_turn)} message(s) of the turn taken as done")
                     + f" — so {skill} comes in live")
    if start.skill_before:
        cautions.append(f"{skill} came in before the person's first turn: it is in each run's context as the chat had "
                        "it, where a variant never reaches it")
    if start.next_skill is not None:
        at = f", at step {start.next_skill['step']}" if start.next_skill["step"] is not None else ""
        # A check of the final reply has nothing to read where a run stops: that is a caution.
        (cautions if isinstance(check, dict) else notes).append(
            f"the chat loads {start.next_skill['skill']} after {skill}{at}: each run stops where the model "
            f"loads a skill other than {skill}, and the check judges the workspace as it is then"
            + (" — a check of the final reply finds none there" if isinstance(check, dict) else ""))
    return cautions, notes


def named_steps(numbers: list[int]) -> str:
    """Steps by their numbers, as a sentence names them: step 2, steps 0–1, steps 0, 2 and 5."""
    if len(numbers) == 1:
        return f"step {numbers[0]}"
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"steps {numbers[0]}–{numbers[-1]}"
    return f"steps {', '.join(map(str, numbers[:-1]))} and {numbers[-1]}"


def imported_source(run: Any, skill: str) -> Path | None:
    """The folder an imported run's copy of `skill` was made from, while it is still there: the
    one the person named at import. None where its SKILL.md was extracted from the chat."""
    imported = (run.parity or {}).get("imported") or {}
    named = next((each.get("dir") for each in imported.get("skills") or [] if each.get("name") == skill), None)
    if named is None and imported.get("skill") == skill:
        named = imported.get("skill_dir")  # an import from before it kept several
    return Path(named) if named and (Path(named) / "SKILL.md").is_file() else None


def _task_skill_dir(run: Any, launch: Any, kept: Path | None) -> Path | None:
    """The skill folder of the task a launched run was of — its spec's, while the spec and the
    folder are there — where the skill studied is the one the run was given."""
    from tuningfork.ops.experiments import recorded_skill_file
    from tuningfork.specs import SpecError, load_spec

    if launch is None or not launch.spec_path or kept is None or recorded_skill_file(run) != kept:
        return None
    try:
        folder = load_spec(Path(launch.spec_path), strict=False).resolved_skill_dir()
    except (OSError, SpecError, ValueError):
        return None
    return folder if folder is not None and (folder / "SKILL.md").is_file() else None


def _skill_file(text: str, name: str, chat: Chat) -> tuple[str, bool]:
    """A SKILL.md of a skill as the model read it: its text, given the frontmatter its harness may
    not have shown (Claude Code strips it) — its name, and the description the conversation lists
    it by where it does, else its first line — so a harness lists it. Whether it was given one."""
    if text.lstrip().startswith("---"):
        return text, False
    listed = re.compile(rf"(?m)^[ \t]*[-*][ \t]*\**{re.escape(name)}\**[ \t]*:[ \t]*(\S.*?)[ \t]*$")
    found = next((match[1] for message in chat.messages if message["role"] != "assistant"
                  for match in [listed.search(_text(message.get("content")))] if match), None)
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), name)
    described = json.dumps((found or first or name)[:300], ensure_ascii=False)
    return f"---\nname: {name}\ndescription: {described}\n---\n\n{text}", True


def _check_files(check: Any, spec_dir: Path | None) -> tuple[dict[str, Path], list[str]]:
    """What a shell check reads from its task's folder (`$TUNINGFORK_SPEC_DIR/<path>`), each by
    the place beside the scenario's spec it is copied to — the first part of its path, so that a
    script finds what lies beside it — and each it names that cannot be copied."""
    from tuningfork.ops import scenarios

    if not isinstance(check, str) or spec_dir is None:
        return {}, []
    found: dict[str, Path] = {}
    unread: list[str] = []
    for relative in SPEC_DIR_PATH.findall(check):
        parts = PurePosixPath(relative).parts
        top = parts[0] if parts else ""
        if top == scenarios.INPUTS:
            continue  # the scenario's own inputs
        if top in ("", ".", "..", scenarios.SPEC, scenarios.CONTEXT, SKILL_FOLDER) or not (spec_dir / top).exists():
            unread.append(relative)
        else:
            found[top] = spec_dir / top
    if "TUNINGFORK_SPEC_DIR" in check and not found and not unread:
        unread.append("$TUNINGFORK_SPEC_DIR")
    return found, unread


def _inputs(store: Store, run: Any, launch: Any, chat: Chat,
            start: Start) -> tuple[dict[str, bytes], list[str], list[str]]:
    """A scenario's input files, and what to say of them, the cautions apart from the notes: a
    launched run's workspace as it was when the call the scenario's first stands for came, from
    its snapshots — else its task's own input files, while its spec's folder has them; for any
    other run, none, and the files its conversation's calls name, for a person to add."""
    if launch is None:
        named = _named_paths(chat.messages)
        if not named:
            return {}, [], []
        shown = ", ".join(named[:8]) + (f" and {len(named) - 8} more" if len(named) > 8 else "")
        return {}, [f"its conversation's calls name {shown}: no run of the scenario has a file it was not given — add "
                    "those the model needs as the scenario's inputs"], []
    step = _step_at(store, run, start.live) if start.live is not None else None
    commit = (step.workspace_commit or (launch.initial_commit if step.idx == 0 else None)) if step is not None else None
    if step is not None and commit:
        from tuningfork.snapshots import SnapshotError, Snapshots, available

        try:
            files, left = _snapshot_files(Snapshots(store.home), commit) if available() else (None, [])
        except SnapshotError:
            files, left = None, []
        if files is not None:
            notes = [f"its inputs are run {run.id[:8]}'s workspace as its step {step.idx} found it: {len(files)} "
                     "file(s)" + (", what the steps before wrote among them" if step.idx else "")]
            return files, [_left_out(left)] if left else [], notes
    return _task_files(run, launch, step)


def _step_at(store: Store, run: Any, index: int) -> Any:
    """The run's step whose call came at message `index` of its conversation: the first whose
    request held the messages before it."""
    from tuningfork.wire import get_wire

    for step in store.steps(run.id):
        if len(get_wire(step.wire).view(store.rebuild_request(step))) == index:
            return step
    return None


def _snapshot_files(snapshots: Any, commit: str) -> tuple[dict[str, bytes], list[str]]:
    """A snapshot's files, as many as a scenario's inputs hold, and the paths of those left out:
    a link, or one past the limits."""
    from tuningfork.ops.scenarios import MAX_FILE, MAX_FILES, MAX_TOTAL

    files: dict[str, bytes] = {}
    left: list[str] = []
    total = 0
    for entry in snapshots.files(commit):
        size = entry.size or 0
        if entry.mode == "120000" or size > MAX_FILE or total + size > MAX_TOTAL or len(files) >= MAX_FILES:
            left.append(entry.path)
            continue
        files[entry.path] = snapshots.blob(entry.sha) if entry.sha else snapshots.read(commit, entry.path)
        total += size
    return files, left


def _left_out(left: list[str]) -> str:
    from tuningfork.ops.scenarios import MAX_FILE, MAX_FILES, MAX_TOTAL

    shown = ", ".join(left[:8]) + (f" and {len(left) - 8} more" if len(left) > 8 else "")
    return (f"left out of its inputs, a link or past what a scenario holds ({MAX_FILES} files of at most "
            f"{MAX_FILE // (1024 * 1024)} MiB, {MAX_TOTAL // (1024 * 1024)} MiB in all): {shown}")


def _task_files(run: Any, launch: Any, step: Any) -> tuple[dict[str, bytes], list[str], list[str]]:
    """The input files a launched run's task gave it, while its spec's folder has them, where no
    snapshot shows its workspace; the cautions, then the notes."""
    import tempfile

    from tuningfork.specs import Eval, SpecError, materialize_files

    item = launch.eval or {}
    cautions = []
    if step is None or step.idx > 0:
        cautions.append(f"no snapshot shows run {run.id[:8]}'s workspace where the scenario starts: what its steps "
                        "before then wrote, no run of the scenario has — add it as the scenario's inputs")
    if not item.get("files"):
        return {}, cautions, []
    try:
        with tempfile.TemporaryDirectory() as scratch:
            workspace = Path(scratch)
            materialize_files(Eval.from_dict(item), workspace, Path(item.get("spec_dir") or "."))
            files = {path.relative_to(workspace).as_posix(): path.read_bytes()
                     for path in sorted(workspace.rglob("*")) if path.is_file()}
    except (OSError, SpecError, KeyError, ValueError) as error:
        return {}, [*cautions, f"its task's input files could not be read ({error}): add them as the scenario's inputs"], []
    return files, cautions, [f"its inputs are its task's own input files, from {item.get('spec_dir')}"]


def _named_paths(messages: list[dict[str, Any]]) -> list[str]:
    """The files a conversation's tool calls name, in the order they first did."""
    named: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(arguments, dict):
                continue
            for key in PATH_ARGUMENTS:
                value = arguments.get(key)
                if isinstance(value, str) and value.strip() and value.strip() not in named:
                    named.append(value.strip())
    return named


def make_scenario(store: Store, plan: ScenarioPlan) -> Any:
    """Write the scenario: its skill where it keeps its own, what its check reads, its
    conversation, then its spec, which names them, and its input files. Its folder is made new,
    and taken away whole when anything fails."""
    from tuningfork.ops import scenarios

    place = scenarios.folder(store.home, plan.name)
    try:
        place.mkdir(parents=True)
    except FileExistsError:
        raise OperationError(f"a scenario called {plan.name!r} exists already", code="conflict", field="name") from None
    try:
        if plan.copy_from is not None:
            shutil.copytree(plan.copy_from, plan.skill_dir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        if plan.extract is not None:
            plan.skill_dir.mkdir(parents=True, exist_ok=True)
            (plan.skill_dir / "SKILL.md").write_text(plan.extract, encoding="utf-8")
        for name, source in plan.check_files.items():
            if source.is_dir():
                shutil.copytree(source, place / name, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            else:
                shutil.copyfile(source, place / name)
        if plan.context.messages:
            scenarios.put_context(store.home, plan.name, plan.context.to_json(), new=True)
        scenarios.save_scenario(store.home, plan.name, plan.spec(), etag=None)
        for path, data in sorted(plan.files.items()):
            scenarios.put_input(store.home, plan.name, path, data)
    except BaseException:
        shutil.rmtree(place, ignore_errors=True)
        raise
    return scenarios.read_scenario(store.home, plan.name)
