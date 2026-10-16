"""Hermes Agent adapter.

Verified against Hermes Agent v0.18.2. Everything here is Hermes-specific by design; the
rest of tuningfork must not need to know any of it.

Isolation uses `HERMES_HOME`, which Hermes supports as a first-class override of its entire
home directory — config, memory, skills, curator state, sessions, cron and gateway state
all move together. That matters more than it sounds: Hermes can rewrite its own skills
during a run and injects memory into later sessions, so an experiment sharing a home with a
real installation is neither isolated nor reproducible.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.harness.base import LaunchSpec, Requirement, SkillSpan
from tuningfork.ops.common import OperationError

#: How Hermes 0.18.2 opens the last user turn of its background review
#: (`agent/background_review.py`): after a turn it may replay the whole conversation with one
#: of three review prompts — memory, skills, or both — each beginning so. Verified from a
#: recording, where the review was cut off when Hermes exited after its answer.
REVIEW_PROMPT = "Review the conversation above and "

#: How Hermes 0.18.2 inlines a skill preloaded with `-s`: its own marker, a blank line, the
#: complete SKILL.md, then a line naming the skill directory. Verified from a recording.
PRELOADED_SKILL = re.compile(
    r'\[IMPORTANT: The user launched this CLI session with the "(?P<name>[^"]+)" skill '
    r"preloaded\.[^\]]*\]\n\n(?P<text>.*?)\n\[Skill directory: ",
    re.DOTALL,
)

CONFIG_TEMPLATE = """\
# Written by tuningfork. This is a disposable, isolated HERMES_HOME used only for
# experiments. It is not connected to any other Hermes installation.
model:
  default: {model}
  provider: custom
  base_url: {base_url}
providers: {{}}
fallback_providers: []
toolsets:
- hermes-cli
agent:
  max_turns: {max_turns}
security:
  # Hermes downloads its tirith command scanner from GitHub into any home that lacks it,
  # in the background. Every run's home is fresh, so each run would fetch 38 MB from
  # outside the proxy, and whether the scanner arrived before the model's first command
  # would vary from run to run. Off, it is the same in every run.
  tirith_enabled: false
"""

#: What a launch of Hermes may be asked beside the seam's own arguments (`HermesOptions`),
#: each recorded in the launch's settings under `harness`, so a run says how its harness was
#: set up:
#:
#: - `self_improvement`: on, Hermes is as shipped: its system prompt tells the model to patch
#:   a skill it finds wanting with `skill_manage`, and its background review may do the same
#:   after the answer. Off, a plugin written into the launch's home removes `skill_manage`
#:   from Hermes's tool registry at start-up (`registry.deregister`, with the operator opt-in
#:   Hermes asks for), so nothing in the run can change the skill it is measured with: the
#:   patching guidance is not injected (Hermes adds it only when the tool is there), and the
#:   skill review's trigger needs the tool too. `skill_view` and `skills_list` stay.
#: - `await_review`: let Hermes's background review finish before the process exits, as an
#:   open session would. After a turn Hermes may replay the conversation on a daemon thread
#:   with a review prompt (`REVIEW_PROMPT`) and patch a skill from it; in one-shot mode
#:   (`chat -q`) the interpreter's exit cuts that thread off. The same plugin joins it from
#:   Hermes's own session hooks, which fire before the exit path arms its watchdog. Nothing
#:   of Hermes is patched.
#: - `review_every_turn`: make the skill review due after every turn
#:   (`skills.creation_nudge_interval: 1`). By default it is due once ten model calls have
#:   passed in the session, which a short task never reaches.
#: - `mcp_servers`: MCP servers for the launch's `config.yaml`, by name, as Hermes takes
#:   them (`command`, `args`, `env`, `timeout`…): how a launched Hermes gets tuningfork's own
#:   tools. Needs Hermes's `mcp` extra.
OPTIONS = ("self_improvement", "await_review", "review_every_turn", "mcp_servers")

PLUGIN_NAME = "tuningfork-launch"
PLUGIN_MANIFEST = """\
name: tuningfork-launch
version: "1"
description: What tuningfork asked of this launch — its self-improvement off, or its background review let finish. Written by tuningfork.
provides_hooks:
- on_session_end
- on_session_finalize
"""
PLUGIN_MODULE = '''\
"""Written by tuningfork into this launch's home, with `options.json` beside it saying what
the launch was asked. Two things, each only when asked: remove Hermes's skill-editing tool,
so the run cannot change the skill it is measured with (self_improvement off); and let
Hermes's background review finish before a one-shot session exits (await_review). After a
turn, Hermes may review the conversation on a daemon thread and patch a skill from it; a
one-shot session exits first and cuts the thread off, and joining it from Hermes's own
session hooks lets it finish, as it would in an open session. Nothing else is changed."""

import json
import threading
from pathlib import Path

SKILL_TOOL = "skill_manage"
REVIEW_THREAD = "bg-review"
#: A bound only: the launch's own timeout is what ends a run whose review never finishes.
WAIT_SECONDS = 3600.0


def asked():
    try:
        return json.loads((Path(__file__).parent / "options.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def wait_for_review(**_kwargs):
    for thread in threading.enumerate():
        if thread.name == REVIEW_THREAD and thread is not threading.current_thread():
            thread.join(WAIT_SECONDS)


def register(ctx):
    options = asked()
    if options.get("self_improvement") is False:
        from tools.registry import discover_builtin_tools, registry

        # Plugins load before Hermes imports its built-in tool modules, which register
        # themselves: import them now, so the tool is there to remove, and the later import
        # (Python's cache) registers nothing again.
        discover_builtin_tools()
        registry.deregister(SKILL_TOOL)
    if options.get("await_review"):
        ctx.register_hook("on_session_end", wait_for_review)
        ctx.register_hook("on_session_finalize", wait_for_review)
'''


@dataclass(frozen=True, slots=True)
class HermesOptions:
    """A launch's options (`OPTIONS`), read strictly from what a person or a job gave."""

    self_improvement: bool = True
    await_review: bool = False
    review_every_turn: bool = False
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def needs_plugin(self) -> bool:
        """Whether the launch's home gets the plugin: for what only Hermes's own process can
        do — remove a tool, or wait on a thread."""
        return self.await_review or not self.self_improvement

    @classmethod
    def read(cls, data: Any) -> HermesOptions:
        if data is None:
            return cls()
        if isinstance(data, HermesOptions):
            return data
        if not isinstance(data, dict):
            raise OperationError("harness options must be an object", code="bad_params", field="harness_options")
        unknown = sorted(set(data) - set(OPTIONS))
        if unknown:
            raise OperationError(
                f"hermes takes no option called {unknown[0]!r}",
                hint=f"its options: {', '.join(OPTIONS)}", code="bad_params", field="harness_options",
            )
        values: dict[str, Any] = {}
        for name in ("self_improvement", "await_review", "review_every_turn"):
            if name in data:
                if not isinstance(data[name], bool):
                    raise OperationError(f"{name} must be true or false", code="bad_params", field="harness_options")
                values[name] = data[name]
        servers = data.get("mcp_servers")
        if servers is not None:
            if not isinstance(servers, dict) or not all(
                isinstance(name, str) and _FOLDER_NAME.match(name) and isinstance(server, dict)
                for name, server in servers.items()
            ):
                raise OperationError(
                    "mcp_servers maps each server's name to an object, as Hermes's config.yaml has it",
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
        """What a launch records of them: every option set — of an MCP server its command,
        never its environment, which may hold a key."""
        found: dict[str, Any] = {}
        if not self.self_improvement:
            found["self_improvement"] = False
        if self.await_review:
            found["await_review"] = True
        if self.review_every_turn:
            found["review_every_turn"] = True
        if self.mcp_servers:
            found["mcp_servers"] = {
                name: server.get("command") or server.get("url") for name, server in self.mcp_servers.items()
            }
        return found

    def config(self) -> str:
        """What the options add to the launch's `config.yaml`."""
        parts: list[str] = []
        if self.review_every_turn:
            parts.append(
                "skills:\n"
                "  # The skill review after every turn, not once ten model calls have passed.\n"
                "  creation_nudge_interval: 1\n"
            )
        if self.needs_plugin:
            parts.append(f"plugins:\n  enabled:\n  - {PLUGIN_NAME}\n")
            if not self.self_improvement:
                # Hermes lets a plugin remove a tool it does not own only with this opt-in.
                parts.append(f"  entries:\n    {PLUGIN_NAME}:\n      allow_tool_override: true\n")
        if self.mcp_servers:
            # JSON is YAML: one flow mapping, read as the nested mapping Hermes expects.
            parts.append(f"mcp_servers: {json.dumps(self.mcp_servers, ensure_ascii=False)}\n")
        return "".join(parts)


#: Environment variables worth carrying into the child process. Everything else is dropped,
#: so a stray HERMES_* in the parent shell cannot quietly change what is being measured.
INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")

#: How a launch lays Hermes's files out, recorded with each launch so a fork is laid out as
#: its source was. "install" is how an installation has them: Hermes's state in `~/.hermes`,
#: and a skill in the category folder it declares, as Hermes's own sync puts its bundled
#: skills (`~/.hermes/skills/productivity/xlsx`). The model reads those paths: laid out as
#: launches were before — "flat", the home itself as HERMES_HOME and every skill directly in
#: `skills/` — the same step called a skill's script another way.
LAYOUTS = ("install", "flat")

#: A category is a folder name: nothing that could climb out of `skills/`.
_FOLDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def skill_category(skill_md: Path) -> str | None:
    """The category a Hermes skill declares in its frontmatter — `metadata: hermes:
    category:`, Hermes's own convention, or a top-level `category:` — if it is a plain
    folder name. Read line by line: the frontmatter is a few nested keys, and tuningfork
    carries no YAML parser."""
    try:
        lines = Path(skill_md).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or lines[0].strip() != "---":
        return None
    inside: list[tuple[int, str]] = []  # the mappings the current line is nested in
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        key, colon, value = line.strip().partition(":")
        if not colon:
            continue
        indent = len(line) - len(line.lstrip(" "))
        while inside and inside[-1][0] >= indent:
            inside.pop()
        keys = [name for _, name in inside] + [key.strip()]
        value = value.strip().strip("'\"")
        if keys in (["metadata", "hermes", "category"], ["category"]) and _FOLDER_NAME.match(value):
            return value
        if not value:
            inside.append((indent, key.strip()))
    return None


class HermesAdapter:
    name = "hermes"
    requirements = (
        Requirement(
            "n_ctx", minimum=64000,
            why="Hermes refuses a model reporting under 64,000 tokens of context, before its first request",
        ),
        Requirement(
            "streaming_tool_calls",
            why="Hermes streams every call, and acts on the tool calls in the stream",
        ),
    )
    #: In-process children with a fresh conversation, on the parent's base URL (0.18.2).
    subagent_tools = frozenset({"delegate_task"})
    #: What differs between two launches of one task besides per-run paths, as (pattern,
    #: replacement) over a request's JSON text; see `tuningfork.replay`. Found by replaying
    #: recorded runs and reading their tool results: the session date in the system prompt,
    #: and how long a tool took, which `execute_code` and `delegate_task` both report.
    #: And the folder `execute_code` runs a script in, which a traceback names: found when every
    #: fork of a run whose script had failed departed at the step after.
    volatile = (
        (r'Conversation started: [^\\"]*', "Conversation started: <date>"),
        (r'(\\"(?:total_)?duration_seconds\\": )[0-9.eE+-]+', r"\1<seconds>"),
        (r"hermes_sandbox_[0-9a-z_]+", "hermes_sandbox_<id>"),
    )

    layouts = LAYOUTS
    #: The tool whose calls edit a skill, for a chain's record of who changed one.
    skill_tools = frozenset({"skill_manage"})

    def __init__(
        self, executable: str = "hermes", *, layout: str = "install", options: Any = None
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; known: {', '.join(LAYOUTS)}")
        self.executable = executable
        self.layout = layout
        self.options = HermesOptions.read(options)

    @property
    def settings(self) -> dict[str, Any]:
        """What a launch records of how its harness was set up, beside its layout."""
        found = self.options.to_json()
        return {"harness": found} if found else {}

    def chain_notes(self) -> list[str]:
        """What a person should know before Hermes's runs are chained, the skill carried
        forward: what its own loop needs to act at all."""
        notes: list[str] = []
        if not self.options.self_improvement:
            notes.append(
                "Hermes's skill-editing tool is removed from these launches (self_improvement off): "
                "the chain will carry the skill on unchanged"
            )
        if not self.options.await_review:
            notes.append(
                "Hermes's background review is cut off when a one-shot session exits: the harness "
                "option await_review lets it finish, so its edits reach the next link"
            )
        if not self.options.review_every_turn:
            notes.append(
                "Hermes's skill review is due once ten model calls have passed in a session, which "
                "a short task never reaches: review_every_turn makes it due after every run"
            )
        return notes

    def hermes_home(self, home: Path) -> Path:
        """Hermes's own state in a launch's home: `~/.hermes`, as installed, or — laid out
        flat — the home itself."""
        return Path(home) / ".hermes" if self.layout == "install" else Path(home)

    def prepare_home(
        self,
        home: Path,
        *,
        base_url: str,
        model: str,
        max_turns: int = 30,
    ) -> Path:
        home = Path(home)
        state = self.hermes_home(home)
        (state / "skills").mkdir(parents=True, exist_ok=True)
        (state / "config.yaml").write_text(
            CONFIG_TEMPLATE.format(model=model, base_url=base_url, max_turns=max_turns)
            + self.options.config(),
            encoding="utf-8",
        )
        if self.options.needs_plugin:
            plugin = state / "plugins" / PLUGIN_NAME
            plugin.mkdir(parents=True, exist_ok=True)
            (plugin / "plugin.yaml").write_text(PLUGIN_MANIFEST, encoding="utf-8")
            (plugin / "__init__.py").write_text(PLUGIN_MODULE, encoding="utf-8")
            (plugin / "options.json").write_text(
                json.dumps({
                    "self_improvement": self.options.self_improvement,
                    "await_review": self.options.await_review,
                }),
                encoding="utf-8",
            )
        return home

    def environment(self, home: Path) -> dict[str, str]:
        """The child's whole environment. `HOME` is the run's own too, as it is for Claude
        Code: the model's shell commands and code run with it, and a real home directory is
        state one run could leave for the next, and that no two machines share."""
        env = {key: os.environ[key] for key in INHERITED_ENV if key in os.environ}
        env["HERMES_HOME"] = str(self.hermes_home(home).resolve())
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
        yolo: bool = True,
    ) -> LaunchSpec:
        """A single non-interactive run.

        `-Q` keeps stdout to the final response, which is what a batch runner wants.
        `--yolo` auto-approves tool use: acceptable only because the workspace is
        disposable, and necessary because there is no TTY to approve anything.
        """
        argv: list[str] = [
            self.executable,
            "chat",
            "-q",
            prompt,
            "-Q",
            "--max-turns",
            str(max_turns),
        ]
        if skills:
            argv += ["-s", ",".join(skills)]
        if yolo:
            argv.append("--yolo")
        argv += list(extra_args)

        return LaunchSpec(
            argv=tuple(argv),
            env=self.environment(home),
            cwd=Path(workspace),
        )

    def continue_launch(
        self,
        prompt: str,
        *,
        home: Path,
        workspace: Path,
        max_turns: int = 30,
        yolo: bool = True,
    ) -> LaunchSpec:
        """One more user turn of the session the launch just ran: `--continue` resumes the
        most recent session in the home — a launch's home has one — with its whole
        conversation, so the review after this turn sees the task and what was said of it."""
        argv: list[str] = [
            self.executable, "chat", "--continue", "-q", prompt, "-Q", "--max-turns", str(max_turns),
        ]
        if yolo:
            argv.append("--yolo")
        return LaunchSpec(argv=tuple(argv), env=self.environment(home), cwd=Path(workspace))

    def skill_roots(self, home: Path) -> tuple[Path, ...]:
        return (self.hermes_home(home) / "skills",)

    def skill_folder(self, home: Path, skill: Path) -> Path:
        """Where a skill folder goes in a launch: in the category folder its SKILL.md
        declares, as Hermes keeps its bundled skills — `skill_view` then reports its path as
        `<category>/<name>/SKILL.md` — or directly under `skills/` when it declares none, or
        when the launch is laid out flat."""
        root = self.skill_roots(home)[0]
        category = skill_category(Path(skill) / "SKILL.md") if self.layout == "install" else None
        return root / category / Path(skill).name if category else root / Path(skill).name

    def side_call(self, request: dict[str, Any]) -> str | None:
        """Hermes's background review: the conversation so far, replayed with a review prompt
        as its last user turn, to decide what to save to memory or a skill. It continues the
        task's conversation, but it is the harness's own call, not the task's next step."""
        from tuningfork.analysis.normalize import content_text

        users = [m for m in request.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
        if len(users) > 1 and content_text(users[-1].get("content")).startswith(REVIEW_PROMPT):
            return "background review"
        return None

    def skill_spans(self, request: dict[str, Any]) -> list[SkillSpan]:
        """Skills in a Hermes payload: preloaded ones in the system prompt, and any
        loaded mid-run through `skill_view`, whose result is a JSON object carrying the
        skill's text under `content`."""
        messages = request.get("messages") or []
        spans: list[SkillSpan] = []
        for index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "system":
                for match in PRELOADED_SKILL.finditer(content):
                    spans.append(
                        SkillSpan(
                            name=match["name"],
                            message_index=index,
                            source="preloaded",
                            text=match["text"],
                            start=match.start("text"),
                            end=match.end("text"),
                        )
                    )
            elif message.get("role") == "tool" and self._is_skill_view(
                messages, message.get("tool_call_id")
            ):
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(result, dict)
                    and result.get("success")
                    and isinstance(result.get("content"), str)
                ):
                    spans.append(
                        SkillSpan(
                            name=str(result.get("name") or "?"),
                            message_index=index,
                            source="tool",
                            text=result["content"],
                        )
                    )
        return spans

    @staticmethod
    def _is_skill_view(messages: list[dict[str, Any]], call_id: str | None) -> bool:
        for message in messages:
            for call in message.get("tool_calls") or []:
                if call.get("id") == call_id:
                    return (call.get("function") or {}).get("name") == "skill_view"
        return False
