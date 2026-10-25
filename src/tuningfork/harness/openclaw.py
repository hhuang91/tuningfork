"""OpenClaw adapter.

Verified against OpenClaw 2026.10.7 (157d8c2) on Node 26.12.1, talking to llama-server
through the tuningfork proxy. Everything here is OpenClaw-specific by design; the rest of
tuningfork must not need to know any of it.

A launch runs `openclaw agent exec`: one prompt, headless, with no Gateway. OpenClaw keeps
its state under `~/.openclaw` (`OPENCLAW_STATE_DIR`'s default), and a launch's `HOME` is its
own directory, so the config, the skills and what OpenClaw writes there — a state database,
links to its plugins' skills — are the launch's alone: no real installation's settings,
credentials or memory reach a run, and nothing of a run is left for the next. `agent exec`
also gives each run a temporary state folder of its own for its session, deleted when it
ends.

On the wire (`tuningfork.wire.openai`): OpenAI chat completions, every call streamed. The
skill reaches the model as a tool result: the system prompt lists each skill's name,
description and location (`<available_skills>`), and the model reads the location with
`read` (Tool Search on, as shipped) or calls `skills_read` (Tool Search off), each giving the
SKILL.md's text, frontmatter included.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from tuningfork.harness.base import LaunchSpec, Requirement, SkillSpan
from tuningfork.ops.common import OperationError

#: The provider a launch's config names, and the key it sends. OpenClaw takes `custom-local`
#: as a non-secret marker for a custom provider on this machine; the only server it ever
#: reaches is tuningfork's proxy.
PROVIDER = "local"
API_KEY = "custom-local"

#: The smallest context window OpenClaw runs a model with, in tokens: under it, `agent exec`
#: exits before its first call (`src/agents/context-window-guard.ts`).
SMALLEST_WINDOW = 4_000

#: How long OpenClaw lets one model call take, in seconds (`models.providers.<id>.timeoutSeconds`).
#: Without it, a call to a local endpoint is given up after 300 s without a first streamed
#: event, and a call held at a breakpoint sends none: its keep-alive comments are not
#: events. With it, both that wait and the wait between events follow this. The task's own
#: timeout, which tuningfork enforces, ends a run long before.
CALL_TIMEOUT = 86_400

#: What a launch of OpenClaw may be asked beside the seam's own arguments (`OpenClawOptions`),
#: each recorded in the launch's settings under `harness`, so a run says how its harness was
#: set up, and a fork of it is set up the same:
#:
#: - `tool_search`: on, as OpenClaw ships it: the model is given its core tools and
#:   `tool_search`, `tool_describe` and `tool_call`; OpenClaw's other tools (`skills_read`,
#:   `sessions_spawn`…) are a line each in a directory in the system prompt, reached through
#:   `tool_call`, and an MCP server's are not named at all, found only by `tool_search`. Off
#:   (`tools.toolSearch: false`), every tool's schema is in the request, and called directly.
#: - `skill_workshop`: on, as shipped: the system prompt has a Skill Workshop section, and the
#:   `skill_workshop` tool saves and patches the model's *learned* skills — in a folder of
#:   their own, never the skill under test. Off, the tool is denied (`tools.deny`), and the
#:   section goes with it: OpenClaw writes it only beside the tool. (`skills.workshop.
#:   autonomous.mode: "off"` stops only the background review, which never runs in a launch.)
#: - `memory_flush`: off by default, unlike OpenClaw: every launch's config says
#:   `agents.defaults.compaction.memoryFlush.enabled: false`. The flush — a silent turn before
#:   a compaction, writing what the model should remember to `memory/` in the workspace —
#:   prepares a later turn of a kept session, and every `agent exec` is a new one, so today it
#:   never runs in a launch either way; on, the launch keeps OpenClaw's own setting.
#: - `mcp_servers`: MCP servers for the launch's config (`mcp.servers`), by name, as OpenClaw
#:   takes them (`command`, `args`, `env`, or `url` with its `transport`): how a launched
#:   OpenClaw gets tuningfork's own tools.
#: - `context_window`: the context window OpenClaw is told the model has, in tokens. OpenClaw
#:   does not ask the server, and assumes 200,000 tokens of a model its config does not size;
#:   it compacts a conversation, and bounds an answer, by this. Unless given, a batch sets it
#:   to the server's own (`for_server`), and a fork to its source's.
OPTIONS = ("tool_search", "skill_workshop", "memory_flush", "mcp_servers", "context_window")

#: Carried into the child from the parent; everything else is dropped, so a stray
#: OPENCLAW_* in the parent shell — a state folder, a config path, a profile — cannot change
#: what is being measured, nor point the run at a real installation.
INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")

#: How OpenClaw 2026.10.7 opens the system prompt of its compaction summary: when a
#: conversation outgrows the context window, a call of its own — no tools, the conversation so
#: far as one user message — summarizes it, and the conversation goes on from the summary. Seen
#: in a recording, two such calls to each compaction.
SUMMARIZER = "You are a context summarization assistant."

#: The user messages OpenClaw 2026.10.7 writes itself, by how each opens once the time it puts
#: before some is left out, and what each is: after a compaction, the summary that stands for
#: the conversation before it, and the turn that tells the model to go on. Seen in a
#: recording. Any other is a person's turn.
NOTES = (
    ("The conversation history before this point was compacted into the following summary:", "compaction summary"),
    ("Continue the current task from the existing transcript, preserving completed work.",
     "continue after a compaction"),
)

#: The time OpenClaw writes before a user message it sends: `[Sun 2026-10-25 02:32 EDT] `.
_TIME = re.compile(r"\A\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}[^\]]*\] ")

#: The Node OpenClaw 2026.10.7 runs on (`package.json`'s engines).
NODE = "24.22.1 or later in 24, or 26.12.1 or later"
#: What `openclaw --version` adds when no supported `node` is found: it answers anyway, where
#: any other command exits before it starts.
_UNSUPPORTED = re.compile(r"Running on an unsupported Node \(([^)\s]+)\)")

#: A server's name, as a folder name is: nothing a config path could be built from.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: What `read` answers in place of a skill OpenClaw has loaded — listed in the system prompt or
#: not — in OpenClaw 2026.10.7 (`src/agents/agent-tools.read.ts`): refused, where the whole file
#: would not fit the call's budget — such a skill is never served in part — and read again in
#: the same context.
NOT_SERVED = (
    "Skill instructions cannot be partially served:",
    "Skill instructions were already served whole earlier in the current model context.",
)
#: What `read` writes after a part of a longer file, in each of its three wordings
#: (`src/agents/sessions/tools/read-page.ts`): OpenClaw's words, not the file's. A skill
#: OpenClaw has loaded is never cut so, but any other SKILL.md is.
_MORE = re.compile(
    r"\n\n\[(?:\d+ more lines? in file|Read output capped at [^\]\n]+ for this call|Showing [^\]\n]+ limit\))\. "
    r"Use (?:read with )?offset=\d+(?:, cursor=\d+)?(?:, limit=\d+)? to continue\.\]\Z"
)


@dataclass(frozen=True, slots=True)
class OpenClawOptions:
    """A launch's options (`OPTIONS`), read strictly from what a person or a job gave."""

    tool_search: bool = True
    skill_workshop: bool = True
    memory_flush: bool = False
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_window: int | None = None

    @classmethod
    def read(cls, data: Any) -> OpenClawOptions:
        if data is None:
            return cls()
        if isinstance(data, OpenClawOptions):
            return data
        if not isinstance(data, dict):
            raise OperationError("harness options must be an object", code="bad_params", field="harness_options")
        unknown = sorted(set(data) - set(OPTIONS))
        if unknown:
            raise OperationError(
                f"openclaw takes no option called {unknown[0]!r}",
                hint=f"its options: {', '.join(OPTIONS)}", code="bad_params", field="harness_options",
            )
        values: dict[str, Any] = {}
        for name in ("tool_search", "skill_workshop", "memory_flush"):
            if name in data:
                if not isinstance(data[name], bool):
                    raise OperationError(f"{name} must be true or false", code="bad_params", field="harness_options")
                values[name] = data[name]
        window = data.get("context_window")
        if window is not None:
            if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                raise OperationError(
                    "context_window is a number of tokens", code="bad_params", field="harness_options",
                )
            if window < SMALLEST_WINDOW:
                # Seen: "Model context window too small (3000 tokens …). Minimum is 4000."
                raise OperationError(
                    f"context_window must be at least {SMALLEST_WINDOW:,} tokens: OpenClaw refuses a smaller "
                    "one before its first call",
                    code="bad_params", field="harness_options",
                )
            values["context_window"] = window
        servers = data.get("mcp_servers")
        if servers is not None:
            if not isinstance(servers, dict) or not all(
                isinstance(name, str) and _NAME.match(name) and isinstance(server, dict)
                for name, server in servers.items()
            ):
                raise OperationError(
                    "mcp_servers maps each server's name to an object, as OpenClaw's mcp.servers has it",
                    code="bad_params", field="harness_options",
                )
            for name, server in servers.items():
                if not (isinstance(server.get("command"), str) or isinstance(server.get("url"), str)):
                    raise OperationError(
                        f"MCP server {name!r} needs a command (stdio) or a url",
                        code="bad_params", field="harness_options",
                    )
            values["mcp_servers"] = {name: dict(server) for name, server in servers.items()}
        return cls(**values)

    def to_json(self) -> dict[str, Any]:
        """What a launch records of them: every option set otherwise than its default — of an
        MCP server its command or URL, never its environment or headers, which may hold a key."""
        found: dict[str, Any] = {}
        for name in ("tool_search", "skill_workshop"):
            if not getattr(self, name):
                found[name] = False
        if self.memory_flush:
            found["memory_flush"] = True
        if self.mcp_servers:
            found["mcp_servers"] = {
                name: server.get("command") or server.get("url") for name, server in self.mcp_servers.items()
            }
        if self.context_window is not None:
            found["context_window"] = self.context_window
        return found


class OpenClawAdapter:
    name = "openclaw"
    #: The wire format its calls take.
    wire = "openai-chat"
    #: The tool that starts a subagent. Under `agent exec` none starts: 2026.10.7 refuses every
    #: spawn without a Gateway ("published reply runtime missing for main"). Under Tool Search
    #: a spawn would come as a `tool_call`, which this name does not catch.
    subagent_tools = frozenset({"sessions_spawn"})
    #: What differs between two launches of one task besides per-run paths, as (pattern,
    #: replacement) over a request's JSON text; see `tuningfork.replay`. Found by replaying
    #: recorded runs: the time OpenClaw writes before the prompt, the date in the system prompt,
    #: and the session key its `Runtime:` line names — a key of any kind, as a tool's result
    #: names one too, with a run's id: a fork of a run whose `sessions_spawn` had failed
    #: departed at the step after, its result naming the child session and run it made. A
    #: command left running in the background is named by a slug and pid of its own (source).
    #: Each pattern holds whatever escaping a result's JSON inside the request's has.
    volatile = (
        (r"\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}[^\]\\\"]*\]", "[<time>]"),
        (r"Current date: \d{4}-\d{2}-\d{2}", "Current date: <date>"),
        (r"(agent:[a-z0-9_-]+:[a-z0-9_-]+:)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", r"\1<session>"),
        (r'(runId\\*"\s*:\s*\\*")[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', r"\1<run>"),
        (r"Command still running \(session [a-z0-9-]+, pid (?:\d+|n/a)\)", "Command still running (session <process>)"),
    )

    #: How a launch lays OpenClaw's files out, recorded with each launch so a fork is laid out
    #: as its source was: "install", as installed — the state and config in `~/.openclaw`,
    #: a skill in `~/.openclaw/skills/<name>`, the paths the model reads.
    layouts = ("install",)

    def __init__(self, executable: str = "openclaw", *, layout: str = "install", options: Any = None) -> None:
        if layout not in self.layouts:
            raise ValueError(f"unknown layout {layout!r}; known: {', '.join(self.layouts)}")
        self.executable = executable
        self.layout = layout
        self.options = OpenClawOptions.read(options)

    @property
    def settings(self) -> dict[str, Any]:
        """What a launch records of how its harness was set up, beside its layout."""
        found = self.options.to_json()
        return {"harness": found} if found else {}

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        """What OpenClaw needs of the server, with Tool Search on or off: a window its first
        request and an answer fit in. Measured on Gemma 4, the first request alone is about
        9,000 tokens with Tool Search and 23,000 without, whatever the task."""
        window, first = (16_384, "9,000") if self.options.tool_search else (32_768, "23,000")
        return (
            Requirement(
                "streaming_tool_calls",
                why="OpenClaw streams every call, and acts on the tool calls in the stream",
            ),
            Requirement(
                "n_ctx", minimum=window,
                why=f"OpenClaw's first request alone is about {first} tokens"
                + ("" if self.options.tool_search else " with Tool Search off") + ", before its answer",
            ),
        )

    def chain_notes(self) -> list[str]:
        """What a person should know before OpenClaw's runs are chained, the skill carried
        forward: nothing of OpenClaw's own changes it."""
        return [
            "OpenClaw's Skill Workshop saves and patches learned skills only, never the skill a chain "
            "carries, and its background review never runs in a launch: each link gets the skill as "
            "the one before left it, changed only if the model edits the file itself"
        ]

    def for_server(self, server: dict[str, Any]) -> OpenClawAdapter:
        """This adapter, with OpenClaw told the server's context window (`n_ctx`) unless a
        person gave one: OpenClaw never asks the server, and of a model its config does not
        size it assumes 200,000 tokens — past what a smaller server holds, it would never
        compact before the server refused the prompt."""
        window = server.get("n_ctx") if isinstance(server, dict) else None
        known = isinstance(window, int) and not isinstance(window, bool) and window >= SMALLEST_WINDOW
        if self.options.context_window is not None or not known:
            return self
        options = {**self.options.to_json(), "context_window": window}
        if self.options.mcp_servers:
            options["mcp_servers"] = self.options.mcp_servers
        return type(self)(self.executable, layout=self.layout, options=options)

    def version_problem(self, output: str) -> str | None:
        """Why an OpenClaw that answers `--version` still cannot launch, from what it printed:
        on a Node it does not support, it runs itself again with a supported `node` it finds on
        PATH, and without one, `--version` answers with a warning while `agent exec` exits before
        its first call. Found here: Ubuntu's Node is 22."""
        found = _UNSUPPORTED.search(output)
        if found is None:
            return None
        return (f"OpenClaw found only Node {found[1]}, which it does not run on: a launch would end before its "
                f"first call. Put Node {NODE} on PATH")

    def state_dir(self, home: Path) -> Path:
        """OpenClaw's own state in a launch's home: `~/.openclaw`, as installed."""
        return Path(home) / ".openclaw"

    def config_path(self, home: Path) -> Path:
        return self.state_dir(home) / "openclaw.json"

    def config(self, *, base_url: str, model: str) -> dict[str, Any]:
        """The launch's `openclaw.json`: one provider, tuningfork's proxy, serving the model
        under test, with what the options ask."""
        entry: dict[str, Any] = {
            "id": model,
            "name": model,
            # llama-server compiles a tool's schema into a grammar: OpenClaw's own llama.cpp
            # profile drops the long `pattern` and `maxLength` values it cannot compile, as
            # OpenClaw's docs say a custom provider pointed at llama-server must ask.
            "compat": {"supportsTools": True, "toolSchemaProfile": "llamacpp"},
        }
        if self.options.context_window is not None:
            entry["contextWindow"] = self.options.context_window
        config: dict[str, Any] = {
            "models": {"providers": {PROVIDER: {
                "baseUrl": base_url,
                "apiKey": API_KEY,
                "api": "openai-completions",
                "timeoutSeconds": CALL_TIMEOUT,
                "models": [entry],
            }}},
            "agents": {"defaults": {"model": {"primary": f"{PROVIDER}/{model}"}}},
        }
        if not self.options.memory_flush:
            config["agents"]["defaults"]["compaction"] = {"memoryFlush": {"enabled": False}}
        tools: dict[str, Any] = {}
        if not self.options.tool_search:
            tools["toolSearch"] = False
        if not self.options.skill_workshop:
            tools["deny"] = ["skill_workshop"]
        if tools:
            config["tools"] = tools
        if self.options.mcp_servers:
            config["mcp"] = {"servers": self.options.mcp_servers}
        return config

    def prepare_home(self, home: Path, *, base_url: str, model: str, max_turns: int = 30) -> Path:
        """`max_turns` is not OpenClaw's: `agent exec` takes no turn limit, and a run ends
        where the model stops, or at the task's timeout."""
        home = Path(home)
        (self.state_dir(home) / "skills").mkdir(parents=True, exist_ok=True)
        self.config_path(home).write_text(
            "// Written by tuningfork: a disposable, isolated OpenClaw home for one launch.\n"
            + json.dumps(self.config(base_url=base_url, model=model), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        return home

    def environment(self, home: Path) -> dict[str, str]:
        """The child's whole environment, built from scratch. `HOME` is the run's own: it is
        where OpenClaw finds its state, and the model's shell commands run with it."""
        env = {key: os.environ[key] for key in INHERITED_ENV if key in os.environ}
        env["HOME"] = str(Path(home).resolve())
        return env

    def launch(
        self,
        prompt: str,
        *,
        home: Path,
        workspace: Path,
        skills: Sequence[str] = (),
        extra_args: Sequence[str] = (),
        max_turns: int = 30,
    ) -> LaunchSpec:
        """A single non-interactive run: `agent exec`, one prompt with no Gateway.

        There is no flag to preload a skill: the skills in `~/.openclaw/skills` are listed to
        the model, which reads one when the task calls for it, so `skills` is only what the
        runner installs. `--config` pins the launch's own config. `--timeout 0` leaves the end
        to tuningfork, which stops a run at its task's timeout as it stops any harness: with
        OpenClaw's own 600 s, a run would end itself first and be judged. The prompt goes
        after `--`, so one that starts with a dash is not read as a flag. Both paths are given
        whole: OpenClaw reads a relative one from the workspace, where it starts.
        """
        home, workspace = Path(home).resolve(), Path(workspace).resolve()
        argv = [
            self.executable,
            "agent",
            "exec",
            "--config",
            str(self.config_path(home)),
            "--cwd",
            str(workspace),
            "--json",
            "--timeout",
            "0",
            *extra_args,
            "--",
            prompt,
        ]
        return LaunchSpec(argv=tuple(argv), env=self.environment(home), cwd=Path(workspace))

    def skill_roots(self, home: Path) -> tuple[Path, ...]:
        return (self.state_dir(home) / "skills",)

    def isolation_warnings(self, workdir: Path) -> list[str]:
        """What could still reach every run from outside it, without the sandbox: OpenClaw
        reads no instructions from above its working directory — verified, with every file it
        loads planted there — but walks up to the git repository it is in, and names its root
        in every prompt's `Runtime:` line (`repo=…`)."""
        start = Path(workdir).resolve()
        for folder in (start, *start.parents):
            if (folder / ".git").exists():
                return [
                    f"the batch folder is inside the git repository at {folder}, so OpenClaw names it in "
                    f"every run's prompt (repo={folder}). Put --workdir outside it."
                ]
        return []

    def side_call(self, request: dict[str, Any]) -> str | None:
        """OpenClaw's compaction summary: a call of its own beside the task's conversation, with
        its own system prompt (`SUMMARIZER`), never the task's next step nor its final reply."""
        from tuningfork.analysis.normalize import content_text

        messages = request.get("messages") or []
        first = messages[0] if messages and isinstance(messages[0], dict) else {}
        if first.get("role") == "system" and content_text(first.get("content")).startswith(SUMMARIZER):
            return "compaction summary"
        return None

    def harness_note(self, message: dict[str, Any]) -> str | None:
        """A user message OpenClaw wrote itself (`NOTES`), named in a few words; None for a
        person's turn — which OpenClaw also sends with the time before it and its `Runtime:`
        line after it, and which a note of another version's wording reads as."""
        from tuningfork.analysis.normalize import content_text

        text = _TIME.sub("", content_text(message.get("content")).strip(), count=1)
        return next((name for opening, name in NOTES if text.startswith(opening)), None)

    def skill_load(self, call: dict[str, Any]) -> str | None:
        """The skill a tool call loads, by name, on the request's OpenAI-style view: a `read` of
        a skill's SKILL.md, by its folder's name — the name tuningfork installs it under — or a
        `skills_read`, called directly or through `tool_call`, by the name it gives. The same
        names `skill_spans` gives, so a scenario that stops at the next skill knows its own."""
        function = call.get("function") if isinstance(call, dict) else None
        loads = _loads(function) if isinstance(function, dict) else None
        return loads[1] if loads else None

    def skill_spans(self, request: dict[str, Any]) -> list[SkillSpan]:
        """Skills loaded in an OpenClaw payload, each a tool result holding the SKILL.md's text:

        - a `read` of the file, as the model finds it listed with Tool Search on. A skill
          OpenClaw has loaded, listed or not, is served whole — any paging the call asks for
          dropped — its line endings made LF and a byte-order mark dropped, as `read` gives any
          file. Too long for the call's budget, it is refused, and read again in the same
          context, OpenClaw says it was already served: neither notice is the skill
          (`NOT_SERVED`). Any other SKILL.md is read as any file is, a part at a time, a note
          after each saying where the rest begins, which is left out (`_MORE`);
        - `skills_read`, with Tool Search off — the file's text as it is;
        - `skills_read` through `tool_call` — the same text, inside the JSON `tool_call` returns.

        A call that failed returns OpenClaw's error as JSON, which is no skill either. A `read`'s
        skill is named by its folder, a `skills_read`'s by the name it was given, as `skill_load`
        names them."""
        messages = request.get("messages") or []
        loads: dict[str, tuple[str, str]] = {}  # tool_call_id -> (how, the skill's name)
        spans: list[SkillSpan] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                found = _loads(function) if isinstance(function, dict) else None
                if found:
                    loads[call.get("id")] = (found[0], found[1])
            if message.get("role") != "tool" or not isinstance(content, str):
                continue
            if message.get("tool_call_id") not in loads:
                continue
            how, name = loads[message["tool_call_id"]]
            text = _wrapped_text(content) if how == "tool_call" else content
            if text is None or _not_served(text):
                continue
            if how == "read":
                more = _MORE.search(text)
                end = more.start() if more else len(text)
                spans.append(SkillSpan(name=name, message_index=index, source="tool", text=text[:end], start=0, end=end))
            elif how == "skills_read":
                spans.append(SkillSpan(name=name, message_index=index, source="tool", text=text, start=0, end=len(text)))
            else:
                # Decoded out of `tool_call`'s JSON: no offsets in the message.
                spans.append(SkillSpan(name=name, message_index=index, source="tool", text=text))
        return spans


def _loads(function: dict[str, Any]) -> tuple[str, str, str] | None:
    """How a tool call loads a skill, if it does: ("read", the skill's name, the path),
    ("skills_read", name, name), or ("tool_call", name, name) for `skills_read` called through
    `tool_call` (`{"id": "skills_read", "args": {"name": …}}`, or the tool's whole id)."""
    name = function.get("name")
    if name == "read":
        path = _argument(function, "path")
        skill = _skill_of(path) if path else None
        return ("read", skill, path) if skill and path else None
    if name == "skills_read":
        skill = _argument(function, "name")
        return ("skills_read", skill, skill) if skill else None
    if name == "tool_call":
        arguments = _arguments(function)
        tool = arguments.get("id")
        inner = arguments.get("args")
        if isinstance(tool, str) and tool.rsplit(":", 1)[-1] == "skills_read" and isinstance(inner, dict):
            skill = inner.get("name")
            if isinstance(skill, str) and skill.strip():
                return ("tool_call", skill.strip(), skill.strip())
    return None


def _not_served(text: str) -> bool:
    """Whether a tool result holds no skill: one of OpenClaw's notices in its place
    (`NOT_SERVED`), or the JSON of a call that failed (`{"status": "error", "tool": …,
    "error": …}`)."""
    if text.startswith(NOT_SERVED):
        return True
    try:
        found = json.loads(text)
    except ValueError:
        return False
    return isinstance(found, dict) and found.get("status") == "error" and "error" in found


def _wrapped_text(content: str) -> str | None:
    """The text a tool called through `tool_call` returned: OpenClaw's JSON around it is
    `{"tool": {…}, "result": {"content": [{"type": "text", "text": …}], …}}`."""
    try:
        wrapped = json.loads(content)
    except ValueError:
        return None
    result = wrapped.get("result") if isinstance(wrapped, dict) else None
    parts = result.get("content") if isinstance(result, dict) else None
    texts = [part["text"] for part in parts or [] if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return "".join(texts) if texts else None


def _arguments(function: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments, if they parse as an object; else none."""
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (TypeError, ValueError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _argument(function: dict[str, Any], name: str) -> str | None:
    """One string argument of a tool call, if its arguments parse and hold it."""
    value = _arguments(function).get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _skill_of(path: str) -> str | None:
    """The skill a path names, by its folder: `…/<name>/SKILL.md`."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return parts[-2] if len(parts) >= 2 and parts[-1] == "SKILL.md" else None
