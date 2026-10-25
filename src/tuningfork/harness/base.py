"""The harness seam.

A harness is whatever drives the model: Hermes, Claude Code through a router, a script
somebody wrote. tuningfork only needs three things from one — how to launch it headless
against a chosen base URL, how to give it an isolated home so experiments cannot disturb a
real installation, and where its skills live.

Everything harness-specific belongs behind this interface. If supporting a new harness
requires editing `tuningfork.proxy`, the seam is in the wrong place.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable


class HarnessError(RuntimeError):
    """Raised when a harness cannot do what was asked of it."""


@dataclass(frozen=True, slots=True)
class SkillSpan:
    """A skill as it appears in a payload — the text the model actually read.

    `start`/`end` are character offsets into the message content when the skill is
    literally there (a preloaded skill inlined into a system prompt). A skill that arrived
    inside a structured tool result has no such offsets: its text had to be decoded out.
    """

    name: str
    message_index: int
    source: str  # preloaded | tool
    text: str
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "message_index": self.message_index,
            "source": self.source,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "lines": self.text.count("\n") + (0 if self.text.endswith("\n") else 1),
        }


@dataclass(frozen=True, slots=True)
class Requirement:
    """Something a harness needs of the inference server, checked before it is launched —
    not found out from a failed run. `what` is a server identity field (`n_ctx`) or a
    capability a server check measures (see `backends.base.Capability`)."""

    what: str
    #: For a number: the least that works.
    minimum: int | None = None
    #: Why, in a sentence a refusal can quote.
    why: str = ""


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """A headless invocation, ready to hand to `subprocess`."""

    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def command_line(self) -> str:
        """Shell-quoted form, for logs and for pasting into a terminal."""
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items()))
        command = shlex.join(self.argv)
        return f"{prefix} {command}".strip()


@runtime_checkable
class HarnessAdapter(Protocol):
    """What tuningfork needs from a harness.

    Optionally, an adapter also declares:

    - `executable`: the program it runs, found on PATH unless the recordings folder's
      config names another (see `ops.connections`).
    - `requirements`: what it needs of the inference server (`Requirement`), checked
      before a batch or fork launches it.

    - `subagent_tools`: the tools whose calls start a subagent (Hermes's `delegate_task`,
      Claude Code's `Agent`), so a launch's threads can be told apart — see
      `Store.finalize_threads`.
    - `volatile`: (pattern, replacement) pairs over a request's JSON text for what differs
      between two launches of one task besides per-run paths — a session id, the date — so
      a fork can recognise the calls it replays; see `tuningfork.replay`.
    - `side_call(request)`: a name for a call the harness makes for itself that continues
      the task's conversation without being its next step — Hermes's background review —
      or None. Such a call starts a side thread of its launch instead of extending the task's
      run, and a run's final reply is never one (see `tuningfork.launches`).
    - `wire`: the wire format its calls take (`tuningfork.wire`), where tuningfork must know
      it before a call arrives: whether a scenario's context can be carried (`with_context`).
    - `continue_launch(prompt, *, home, workspace, max_turns)`: the command that gives the
      session the launch just ran one more user turn, in the same home and workspace — how a
      chain tells the harness what the task's check made of its result, as a person who
      looked would (`ops.chain`, `tell`). Hermes resumes its most recent session.
    - `as_sent(messages)`: a history as the harness kept it, as it sends it in a call — what it
      does to its own copy before every one (Hermes strips each message and writes tool-call
      arguments compactly) — for a request rebuilt from a kept history (`ops.imports`).
      Without one, a history is sent as kept.
    - `skill_load(call)`: the skill a tool call on a request's OpenAI-style view loads, by name
      (Hermes's `skill_view`, Claude Code's `Skill`), or None: where a scenario's run that stops
      at the next skill ends (`tuningfork.stop_at_next_skill`), and where a scenario made from
      a chat finds it does.
    - `harness_note(message)`: a user message the harness wrote itself, named in a few words —
      Hermes's nudge to go on after a cut reply, Claude Code's reminders — or None for a
      person's turn, which a run's page shows as a new exchange (`ops.timeline`). Without
      one, every user message is a person's.
    - `for_server(server)`: the adapter set up for the server it will talk to, from the
      server's identity (`n_ctx`, …) — for a harness that does not ask the server itself:
      OpenClaw is told its context window. A batch calls it (`ops.batch`); what it sets is
      the adapter's `settings`, so a fork, set up from its source's, needs none.
    - `version_problem(output)`: why a harness that answers `--version` still cannot launch,
      read from what that printed (OpenClaw on a Node it does not run on), or None — what the
      Connections page and `connections show` say beside its version (`ops.connections`).
    """

    name: str

    def prepare_home(
        self, home: Path, *, base_url: str, model: str, max_turns: int = 30
    ) -> Path:
        """Create an isolated home directory configured to talk to `base_url`.

        Isolation is the point: an experiment must not read or write the state of a real
        installation, which may be in use. `max_turns` is for a harness whose config holds
        its own turn limit, so it agrees with the one on its command line.
        """

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
        """Build the command that runs `prompt` once, non-interactively."""

    def skill_roots(self, home: Path) -> tuple[Path, ...]:
        """Directories holding SKILL.md files for this harness."""

    def skill_spans(self, request: dict) -> list[SkillSpan]:
        """Where this harness put skill text in a request, found from the payload alone."""


class GenericHarness:
    """For harnesses tuningfork cannot launch itself.

    Recording still works — that only needs the harness pointed at the proxy — but batch
    running and isolation are the user's responsibility.
    """

    name = "generic"
    subagent_tools: frozenset[str] = frozenset()
    volatile: tuple[tuple[str, str], ...] = ()

    def prepare_home(
        self, home: Path, *, base_url: str, model: str, max_turns: int = 30
    ) -> Path:
        raise HarnessError(
            "tuningfork cannot configure this harness automatically. Point its "
            f"OpenAI base URL at {base_url} yourself, then record as usual."
        )

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
        raise HarnessError(
            "tuningfork does not know how to launch this harness headlessly. "
            "Run it yourself against the proxy, or add an adapter in "
            "tuningfork/harness/ — see docs/adapters/TEMPLATE.md."
        )

    def skill_roots(self, home: Path) -> tuple[Path, ...]:
        return ()

    def skill_spans(self, request: dict) -> list[SkillSpan]:
        return []

    def side_call(self, request: dict) -> str | None:
        return None
