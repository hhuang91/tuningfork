"""Claude Code adapter.

Verified against Claude Code 2.1.281, talking to llama.cpp's native Anthropic endpoint
(`/v1/messages`) with no router in between. Everything here is Claude Code-specific by
design; the rest of tuningfork must not need to know any of it.

Isolation: `CLAUDE_CONFIG_DIR` relocates Claude Code's whole state — `.claude.json`,
settings, sessions, skills — and a run's `HOME` is its own directory too, so neither a real
installation's settings nor its user memory can leak into an experiment. Claude Code also
loads every `CLAUDE.md` in the directories *above* its working directory — verified: one in
the recordings folder reached every run — so those are excluded by path at launch. The API
key is a placeholder: the only server it is ever sent to is tuningfork's proxy.

On the wire (see `tuningfork.wire.anthropic`): every call streams; the model's reasoning
arrives as `thinking` blocks; and a skill is loaded through the `Skill` tool, whose result
is followed by a user message that starts `Base directory for this skill: <dir>` and then
carries the SKILL.md body — with its frontmatter stripped, and followed by an
`ARGUMENTS: ...` paragraph when the model passed the skill arguments.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from tuningfork.config import server_root
from tuningfork.harness.base import LaunchSpec, Requirement, SkillSpan

SKILL_PREFIX = "Base directory for this skill: "
ARGUMENTS_MARKER = "\n\nARGUMENTS: "
#: What Claude Code adds to a user message for the model, not from a person: a reminder of its
#: own (the date, a file changed, its to-do list), wrapped so.
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

#: Settings that keep a headless run to the task and off every network but the proxy's.
QUIET = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}

#: Carried into the child from the parent; everything else is dropped, so a stray
#: ANTHROPIC_* or CLAUDE_* in the parent shell cannot change what is being measured.
INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")


def anthropic_base(base_url: str) -> str:
    """Claude Code appends `/v1/messages` itself, so it wants the server root."""
    return server_root(base_url)


#: The memory files Claude Code looks for in each directory it walks up through.
MEMORY_FILES = ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md", ".claude/rules/**")


def ancestor_memory(workspace: Path) -> list[str]:
    """`claudeMdExcludes` patterns for every memory file above the workspace.

    Claude Code matches them against absolute paths. The workspace itself is left alone: a
    task may ship a CLAUDE.md as part of what is being tested.
    """
    above = Path(workspace).resolve().parents
    return [f"{folder.as_posix().rstrip('/')}/{name}" for folder in above for name in MEMORY_FILES]


class ClaudeCodeAdapter:
    name = "claude-code"
    #: The wire format its calls take.
    wire = "anthropic-messages"
    requirements = (
        Requirement(
            "anthropic_messages",
            why="Claude Code speaks only the Anthropic Messages API, /v1/messages",
        ),
    )
    #: `Agent` in 2.1.281 (optionally in the background), `Task` in earlier versions: both
    #: run in-process, on the parent's base URL.
    subagent_tools = frozenset({"Agent", "Task"})
    #: What differs between two launches of one task besides per-run paths, as (pattern,
    #: replacement) over a request's JSON text; see `tuningfork.replay`. Found by replaying
    #: recorded runs: a device and session id in `metadata`, and the memory directory, which
    #: names the working directory with its slashes turned into dashes. A subagent adds
    #: more: the new agent's id; for one in the background, the file its transcript goes to,
    #: under the working directory's dashed name and the session; and, in what it reports
    #: back, how long it took and its token count, which counts the per-run paths it read.
    #: And the date in the system prompt: found when a validation's forks, the day after
    #: their run was recorded, all departed from it at step 0.
    volatile = (
        (r"Today's date is \d{4}-\d{2}-\d{2}", "Today's date is <date>"),
        (r'"user_id": "(?:[^"\\]|\\.)*"', '"user_id": "<session>"'),
        (r'projects/-[^/"\\]*?-[0-9a-f]{8}-\d+-(?:workspace|home)', "projects/<workspace>"),
        (r"(agentId: |with to: '|<task-id>)[0-9a-z]{8,}", r"\1<agent>"),
        (r'/tmp/claude-\d+/[^"\\\s<]*?\.output', "<agent transcript>"),
        (r"(<duration_ms>|duration_ms: |<subagent_tokens>|subagent_tokens: )\d+", r"\1<n>"),
    )

    #: How a launch lays Claude Code's files out, recorded with each launch so a fork is laid
    #: out as its source was: "install" as installed, the config in `~/.claude` and a skill in
    #: `~/.claude/skills/<name>` — the paths the model reads; "flat", as launches were before,
    #: the config in `config/`.
    layouts = ("install", "flat")

    def __init__(self, executable: str = "claude", *, layout: str = "install") -> None:
        if layout not in self.layouts:
            raise ValueError(f"unknown layout {layout!r}; known: {', '.join(self.layouts)}")
        self.executable = executable
        self.layout = layout

    def config_dir(self, home: Path) -> Path:
        return Path(home) / (".claude" if self.layout == "install" else "config")

    def prepare_home(
        self, home: Path, *, base_url: str, model: str, max_turns: int = 30
    ) -> Path:
        """`max_turns` goes on the command line only (see `launch`)."""
        home = Path(home)
        (self.config_dir(home) / "skills").mkdir(parents=True, exist_ok=True)
        # Every model tier Claude Code might reach for is the one model under test.
        env = {
            "ANTHROPIC_BASE_URL": anthropic_base(base_url),
            "ANTHROPIC_API_KEY": "sk-tuningfork-local",
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            **QUIET,
        }
        settings = {
            "$comment": "Written by tuningfork: a disposable, isolated Claude Code home.",
            "env": env,
        }
        (self.config_dir(home) / "settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )
        return home

    def environment(self, home: Path) -> dict[str, str]:
        home = Path(home).resolve()
        settings = json.loads((self.config_dir(home) / "settings.json").read_text(encoding="utf-8"))
        env = {key: os.environ[key] for key in INHERITED_ENV if key in os.environ}
        # Also set directly, not only through settings.json: the first request must
        # already go to the proxy, before any settings are applied.
        env.update(settings["env"])
        env["HOME"] = str(home)
        env["CLAUDE_CONFIG_DIR"] = str(self.config_dir(home))
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
        """A single non-interactive run.

        Claude Code has no flag to preload a skill: skills in the config directory are
        offered to the model, which loads one through the `Skill` tool when the task calls
        for it. `skills` is therefore only what the runner installs, not a launch option.
        `--dangerously-skip-permissions` is acceptable only because the workspace is
        disposable, and necessary because there is no one to approve anything.
        """
        env = self.environment(home)
        settings = {"claudeMdExcludes": ancestor_memory(Path(workspace))}
        argv = [
            self.executable,
            "-p",
            prompt,
            "--model",
            env["ANTHROPIC_MODEL"],
            "--max-turns",
            str(max_turns),
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--settings",
            json.dumps(settings),
            *extra_args,
        ]
        return LaunchSpec(argv=tuple(argv), env=env, cwd=Path(workspace))

    def isolation_warnings(self, workdir: Path) -> list[str]:
        """What could still reach every run from outside it.

        Inside a git repository, Claude Code also offers the skills, commands and agents in
        `.claude` folders between the working directory and the repository's root. Those
        cannot be excluded the way memory files are.
        """
        start = Path(workdir).resolve()
        chain = [start, *start.parents]
        for position, folder in enumerate(chain):
            if not (folder / ".git").exists():
                continue
            found = [str(above / ".claude") for above in chain[: position + 1]
                     if (above / ".claude").is_dir()]
            if not found:
                return []
            return [
                f"the batch folder is inside the git repository at {folder}, so Claude Code "
                f"will offer every run what is in {', '.join(found)}. Put --workdir outside it."
            ]
        return []

    def skill_roots(self, home: Path) -> tuple[Path, ...]:
        return (self.config_dir(home) / "skills",)

    def harness_note(self, message: dict[str, Any]) -> str | None:
        """A user message Claude Code wrote itself, on the request's OpenAI-style view: the text
        of a skill the model loaded, or nothing but its `<system-reminder>`s. A person's words,
        reminders beside them or not, are their turn: None."""
        content = message.get("content")
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        else:
            texts = []
        if any(text.startswith(SKILL_PREFIX) for text in texts):
            return "skill text"
        if texts and not REMINDER.sub("", "\n".join(texts)).strip():
            return "reminder"
        return None

    def skill_spans(self, request: dict[str, Any]) -> list[SkillSpan]:
        """Skills loaded through the `Skill` tool, found on the request's OpenAI-style view.

        The skill's text is the user message that follows the tool's result, after its
        `Base directory for this skill:` line.
        """
        messages = request.get("messages") or []
        requested: dict[str, tuple[str, str]] = {}  # tool_call_id -> (skill name, its args)
        spans: list[SkillSpan] = []
        last_result: tuple[str, str] | None = None
        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != "Skill":
                    continue
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                requested[call.get("id")] = (
                    str(arguments.get("skill") or "?"),
                    str(arguments.get("args") or ""),
                )
            if message.get("role") == "tool":
                last_result = requested.get(message.get("tool_call_id"))
                continue
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    span = _skill_in(content, index, last_result, inline=True)
                    spans.extend([span] if span else [])
                elif isinstance(content, list):
                    # The turn carries other blocks too (a reminder, say): the skill is one
                    # text block among them, so it has no offset in a single string.
                    for block in content:
                        text = block.get("text") if isinstance(block, dict) else None
                        span = _skill_in(text, index, last_result, inline=False) if isinstance(text, str) else None
                        spans.extend([span] if span else [])
            last_result = None
        return spans


def _skill_in(
    text: str, index: int, loaded: tuple[str, str] | None, *, inline: bool
) -> SkillSpan | None:
    """The skill in a `Base directory for this skill:` text, if it is one."""
    if not text.startswith(SKILL_PREFIX):
        return None
    header, separator, _ = text.partition("\n\n")
    if not separator:
        return None
    start = len(header) + len(separator)
    end = len(text.rstrip())
    name, args = loaded or (Path(header[len(SKILL_PREFIX):].strip()).name, "")
    # A `Skill` call with arguments gets them appended after the body, as a final
    # `ARGUMENTS: ...` paragraph: the harness's text, not the skill's. Only that exact
    # paragraph is cut — a skill may well contain the marker itself.
    suffix = f"{ARGUMENTS_MARKER}{args}"
    if args and text.rstrip().endswith(suffix.rstrip()):
        end = len(text.rstrip()) - len(suffix.rstrip())
        end = start + len(text[start:end].rstrip())
    return SkillSpan(
        name=name,
        message_index=index,
        source="tool",
        text=text[start:end],
        start=start if inline else None,
        end=end if inline else None,
    )
