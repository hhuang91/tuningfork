"""A conversation an agent had outside tuningfork, imported as a run: recorded as if it had come
through the proxy, so the step tools work on it — resample, A/B a wording, the forking-token
probe, a refinement's screens at a step — each step with the chat's earlier turns as its context.

A history keeps what the agent kept, not the requests it sent, so each step's request is rebuilt
from it as its harness sends one: the system prompt, every message before the step, and the
tools where the file names them (Hermes's export does not). An assistant turn's reasoning is in
its own step's answer, never in a later request: Hermes sends none back to a server that does not
ask for it. The run says it was imported and what the file lacked (`parity.imported`); nothing
launched it, so nothing can fork it to the end.

Read: Hermes's session snapshot (`~/.hermes/sessions/session_<id>.json`: the system prompt, the
tools and the messages), Hermes's export (`hermes sessions export`: one session a line, without
tools), and a plain OpenAI-style message list, or an object holding one. The skill under study is
found where its harness shows skills (`HarnessAdapter.skill_spans`; SKILL.md frontmatter where
the harness is not one tuningfork knows), and kept beside the run (`<home>/imports/<run>/`):
the skill folder the person names, or the text the model read, extracted — SKILL.md alone, since
the chat never showed the rest of the folder.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tuningfork.ops.common import OperationError
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

    @property
    def steps(self) -> list[int]:
        """Each assistant turn's index in `messages`: the steps an import records."""
        return [index for index, message in enumerate(self.messages) if message["role"] == "assistant"]

    @property
    def system_prompt(self) -> bool:
        return bool(self.messages) and self.messages[0]["role"] == "system"

    def request(self, index: int) -> dict[str, Any]:
        """The request the assistant turn at `index` answered, as its harness sends one."""
        request: dict[str, Any] = {"model": self.model or "imported", "messages": [sent(m) for m in self.messages[:index]]}
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
    """What an import will record, shown before it does."""

    chat: Chat
    label: str
    task_id: str | None
    harness: str
    skills: list[FoundSkill]
    #: The skill under study, and its folder when the person named one (else its text is kept).
    skill: FoundSkill | None
    skill_dir: Path | None
    warnings: list[str] = field(default_factory=list)
    #: Where the tools came from, when the file named none and a recorded run's were taken.
    tools_from: str | None = None

    @property
    def choose(self) -> bool:
        """Whether the person must name the skill to study first: the conversation shows several,
        and a run studies one."""
        return self.skill is None and self.skill_dir is None and len(self.skills) > 1

    def to_json(self) -> dict[str, Any]:
        chat = self.chat
        return {
            "format": chat.format, "session_id": chat.session_id, "title": chat.title, "model": chat.model,
            "harness": self.harness, "label": self.label, "task_id": self.task_id,
            "messages": len(chat.messages), "steps": len(chat.steps), "system_prompt": chat.system_prompt,
            "tools": len(chat.tools or []), "tools_from": self.tools_from,
            "skills": [skill.to_json() for skill in self.skills],
            "skill": self.skill.name if self.skill else None,
            "skill_dir": str(self.skill_dir) if self.skill_dir else None,
            "choose_skill": self.choose,
            "warnings": self.warnings,
        }


def prepare_import(
    store: Store,
    data: bytes | str,
    *,
    session: str | None = None,
    skill: str | None = None,
    skill_dir: str | Path | None = None,
    harness: str | None = None,
    label: str | None = None,
    task_id: str | None = None,
    tools_from: str | None = None,
) -> ImportPlan:
    """An import, checked and shown before anything is recorded: the conversation, the skills it
    shows, the one under study, and what the file lacked."""
    from tuningfork.harness import ADAPTERS
    from tuningfork.paths import resolve_path

    chat = read_chat(data, session=session)
    harness = harness or chat.harness or "generic"
    if harness not in ADAPTERS:
        raise OperationError(f"no harness called {harness!r}", hint=f"one of {', '.join(ADAPTERS)}", field="harness")
    if not chat.steps:
        raise OperationError("the conversation has no assistant turn: nothing the model did to study", field="file")
    warnings: list[str] = []
    origin = None
    if not chat.tools and tools_from:
        chat.tools, origin = _tools_of(store, tools_from)
    if not chat.system_prompt:
        warnings.append("the file kept no system prompt: each step is rebuilt without one, which is not what the "
                        "model read")
    if not chat.tools:
        warnings.append("the file names no tools" + (" (Hermes's export keeps none)" if chat.format == "hermes-export" else "")
                        + ": each step is rebuilt without them; name a run recorded through the same harness to "
                        "take its tools (tools_from)")
    if chat.model:
        warnings.append(f"its answers are {chat.model}'s: an experiment here samples your server's model, given "
                        "this conversation")

    skills = find_skills(chat, harness)
    folder = resolve_path(str(skill_dir)).path if skill_dir else None
    if folder is not None and not (folder / "SKILL.md").is_file():
        raise OperationError(f"{folder} holds no SKILL.md", field="skill_dir")
    chosen = _chosen(skills, skill, folder)
    if chosen is None and folder is None:
        if not skills:
            warnings.append("no skill shows in the conversation: an A/B or a refinement needs one (skill_dir)")
        elif len(skills) > 1:
            warnings.append(f"the conversation shows {len(skills)} skills: name the one to study — a run studies one; "
                            "the others stay in its steps as they were")
    elif folder is not None:
        text = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        if chosen is None:
            warnings.append(f"{folder.name} does not show in the conversation: nothing at a step reads it")
        elif not _locatable(chat, chosen, text):
            warnings.append(f"{folder}/SKILL.md is not the text the model read in the conversation: an A/B or a "
                            "refinement finds the skill by its text, and would not find it")
    elif chosen.step is None:
        warnings.append(f"{chosen.name} came in after the last assistant turn: no step reads it")
    if chosen is not None and folder is None:
        warnings.append(f"{chosen.name}'s SKILL.md is extracted from what the model read: the rest of its folder — "
                        "scripts, templates, references — the chat never showed (skill_dir names the folder)")
    name = chat.title or (f"session {chat.session_id}" if chat.session_id else "a chat")
    return ImportPlan(
        chat=chat, label=label or f"imported: {name}"[:120], task_id=task_id or None, harness=harness,
        skills=skills, skill=chosen, skill_dir=folder, warnings=warnings, tools_from=origin,
    )


def _chosen(skills: list[FoundSkill], name: str | None, folder: Path | None) -> FoundSkill | None:
    wanted = name or (folder.name if folder is not None else None)
    if wanted:
        found = next((skill for skill in skills if skill.name == wanted), None)
        if found is None and name:
            shown = ", ".join(skill.name for skill in skills) or "none"
            raise OperationError(f"no skill called {name!r} shows in the conversation", hint=f"the skills in it: {shown}",
                                 field="skill")
        return found
    return skills[0] if len(skills) == 1 else None


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
    skill: str | None
    folder: Path
    warnings: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"run": self.run_id, "steps": self.steps, "skill": self.skill, "folder": str(self.folder),
                "warnings": self.warnings}


def execute_import(store: Store, plan: ImportPlan) -> ImportResult:
    """Record the conversation as a run, step by step, and keep beside it what it was imported
    from and the skill under study. The run is its own from the first step: it never waits on
    the label a live call could take (`Store.begin_run`)."""
    from tuningfork.launches import Assignment
    from tuningfork.specs import hash_tree

    if plan.choose:
        raise OperationError(
            f"the conversation shows {len(plan.skills)} skills: name the one to study — a run studies one",
            hint="the skills in it: " + ", ".join(f"{skill.name} (in context from step {skill.step})" for skill in plan.skills),
            code="choose_skill", field="skill",
        )
    chat = plan.chat
    run_id = uuid.uuid4().hex
    folder = root(store.home) / run_id
    folder.mkdir(parents=True)
    try:
        (folder / "chat.json").write_text(json.dumps(chat.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        skills = folder / "skills"
        kept = None
        if plan.skill_dir is not None:
            kept = skills / plan.skill_dir.name
            shutil.copytree(plan.skill_dir, kept, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        elif plan.skill is not None:
            kept = skills / plan.skill.name
            kept.mkdir(parents=True)
            (kept / "SKILL.md").write_text(plan.skill.text, encoding="utf-8")
        for number, index in enumerate(chat.steps):
            store.record_step(
                request=chat.request(index), response=chat.response(index), harness=plan.harness,
                assignment=Assignment(launch_id=None, run_id=run_id, idx=number, thread=0, label=plan.label,  # type: ignore[arg-type]
                                      task_id=plan.task_id, role=None),  # type: ignore[arg-type]
            )
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    imported = {
        "format": chat.format, "session_id": chat.session_id, "title": chat.title, "model": chat.model,
        "at": time.time(), "tools_from": plan.tools_from, "skill": kept.name if kept else None,
        "skill_source": ("folder" if plan.skill_dir is not None else "extracted") if kept else None,
        "warnings": plan.warnings,
    }
    store.annotate_run(
        run_id, harness=plan.harness, task_id=plan.task_id, label=plan.label,
        skill_hashes=hash_tree([skills]) if kept else None,
        parity={"imported": imported, **({"skill_snapshot_dir": str(kept)} if kept else {})},
    )
    return ImportResult(run_id=run_id, steps=len(chat.steps), skill=kept.name if kept else None, folder=folder,
                        warnings=plan.warnings)
