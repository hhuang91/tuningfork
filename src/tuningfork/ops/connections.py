"""What tuningfork talks to: the inference server, and the harnesses it launches.

A harness is found on PATH, or at an executable the recordings folder's `config.json`
names — never by reading or writing the harness's own configuration, which belongs to a
real installation that may be in use. Its version is read with `--version` in a throwaway
home, so even that leaves nothing behind.

Before a batch or a fork launches a harness, its `requirements` of the server are checked
(`preflight`): a context too small for Hermes, or no `/v1/messages` for Claude Code, is
refused before a launch is recorded, not found out from a failed run.

Token work is held to the same check (`token_work`): scoring an answer, forking tokens and
keeping each sample's tokens each need something a server check measures, and one the check
found the server without is refused before anything is sent, saying what was missing — the
web UI greys its way in, with the same words. What was never measured refuses nothing.

tuningfork is built for a model one runs oneself. A server on the internet that answers as a
hosted model API does — asking for a key tuningfork does not send, or without llama.cpp's
/props — is said to be one wherever the server is shown (`reminder`). A harness's calls are
recorded through it, its own key passed on; only a launch is refused where a key is asked for,
since a harness tuningfork launches carries none (`preflight`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tuningfork.backends.formats import known_models
from tuningfork.config import read_home_config, write_home_config
from tuningfork.ops.common import OperationError

#: Harness version checks give up after this long: a harness that waits for input, or for
#: a network, must not hang the page that asked.
VERSION_TIMEOUT = 15.0


def overrides(home: Path | str) -> dict[str, str]:
    """Harness executables named in the recordings folder's config, by harness name."""
    found = read_home_config(home).get("harnesses")
    if not isinstance(found, dict):
        return {}
    return {
        name: entry["executable"]
        for name, entry in found.items()
        if isinstance(entry, dict) and isinstance(entry.get("executable"), str)
    }


def set_override(home: Path | str, name: str, executable: str | None) -> None:
    """Name the executable to run for a harness, or with None go back to PATH. Refused
    unless it is an absolute path to a file this user can run."""
    from tuningfork.harness import ADAPTERS

    if name not in ADAPTERS or not hasattr(ADAPTERS[name](), "executable"):
        raise OperationError(f"no harness called {name!r} to point at an executable", code="not_found")
    config = read_home_config(home)
    harnesses = config.get("harnesses") if isinstance(config.get("harnesses"), dict) else {}
    if executable is None:
        harnesses.pop(name, None)
    else:
        from tuningfork.paths import under_wsl, windows_path

        if windows_path(executable) and under_wsl():
            # Not translated: a Windows program, even one reachable under /mnt/c, is not a
            # harness tuningfork can run in its own isolated home here.
            raise OperationError(
                f"{executable} is a Windows path, and tuningfork runs in WSL", field="executable",
                hint="give the program's path inside WSL, like /home/you/.local/bin/hermes",
            )
        path = Path(executable.strip().strip("\"'"))
        if not path.is_absolute():
            raise OperationError(
                f"{executable!r} is not an absolute path", field="executable",
                hint="name the program by its full path, like /home/you/.local/bin/hermes",
            )
        if not path.is_file() or not os.access(path, os.X_OK):
            raise OperationError(f"{executable} is not a program this user can run", field="executable")
        harnesses[name] = {"executable": str(path)}
    config["harnesses"] = harnesses
    write_home_config(home, config)


def adapter_for(
    home: Path | str, name: str, *, layout: str | None = None, options: Any = None
) -> Any:
    """The adapter for a harness, running the executable the recordings folder names, and
    laying its launches out as `layout` says — a fork's as its source's were — when the
    harness has layouts; new launches are laid out as installed. `options` are the harness's
    own, which its adapter reads strictly (`HermesAdapter`'s `OPTIONS`); a harness that takes
    none refuses them."""
    import inspect

    from tuningfork.harness import ADAPTERS

    if name not in ADAPTERS:
        raise OperationError(
            f"unknown harness {name!r}; known: {', '.join(ADAPTERS)}", field="harness"
        )
    cls = ADAPTERS[name]
    executable = overrides(home).get(name)
    arguments: dict[str, Any] = (
        {"layout": layout} if layout and layout in getattr(cls, "layouts", ()) else {}
    )
    if options:
        if "options" not in inspect.signature(cls.__init__).parameters:
            raise OperationError(
                f"{name} takes no harness options", code="bad_params", field="harness_options"
            )
        arguments["options"] = options
    return cls(executable, **arguments) if executable else cls(**arguments)


@dataclass(slots=True)
class HarnessInfo:
    name: str
    #: What a launch runs: the override, or the bare name looked up on PATH.
    executable: str
    #: Where that is, or None when it cannot be found.
    path: str | None
    #: override | PATH | missing
    source: str
    version: str | None
    error: str | None
    requirements: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executable": self.executable,
            "path": self.path,
            "source": self.source,
            "version": self.version,
            "error": self.error,
            "requirements": self.requirements,
        }


def _version(path: str, problem: Any = None) -> tuple[str | None, str | None]:
    """A harness's `--version`, run where it can touch nothing real: a throwaway home, its
    own config dirs pointed there, no input. `problem` is its adapter's reading of what it
    printed (`version_problem`): what answers --version and still cannot launch — OpenClaw on
    a Node it does not support."""
    with tempfile.TemporaryDirectory(prefix="tuningfork-detect-") as scratch:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": scratch,
            "HERMES_HOME": scratch,
            "CLAUDE_CONFIG_DIR": scratch,
            "XDG_CONFIG_HOME": scratch,
            "XDG_DATA_HOME": scratch,
            "XDG_CACHE_HOME": scratch,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        try:
            done = subprocess.run(
                [path, "--version"], cwd=scratch, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=VERSION_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None, f"--version did not answer in {VERSION_TIMEOUT:g} s"
        except OSError as exc:
            return None, f"could not run it: {exc}"
    lines = [line.strip() for line in (done.stdout or done.stderr or "").splitlines() if line.strip()]
    if done.returncode != 0:
        return None, f"--version exited {done.returncode}: {lines[0] if lines else 'no output'}"
    said = problem(f"{done.stdout or ''}\n{done.stderr or ''}") if problem is not None else None
    return (lines[0] if lines else None), said


def detect_harnesses(home: Path | str) -> list[HarnessInfo]:
    """Every harness tuningfork can launch: where its program is, and which version."""
    from tuningfork.harness import ADAPTERS

    named = overrides(home)
    found: list[HarnessInfo] = []
    for name, cls in ADAPTERS.items():
        adapter = cls()
        executable = getattr(adapter, "executable", None)
        if not executable:
            continue  # it cannot be launched: `generic`
        requirements = [
            {"what": r.what, "minimum": r.minimum, "why": r.why}
            for r in getattr(adapter, "requirements", ())
        ]
        chosen = named.get(name, executable)
        path = chosen if name in named else shutil.which(executable)
        if path is None or not Path(path).is_file():
            found.append(HarnessInfo(
                name, chosen, None, "missing", None,
                f"{chosen!r} is not on PATH" if name not in named else f"{chosen} does not exist",
                requirements,
            ))
            continue
        version, error = _version(path, getattr(adapter, "version_problem", None))
        found.append(HarnessInfo(
            name, chosen, path, "override" if name in named else "PATH", version, error, requirements,
        ))
    return found


# --- preflight --------------------------------------------------------------------------------


@dataclass(slots=True)
class Preflight:
    """What stops a harness from running against this server, and what could not be told."""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": self.problems, "notes": self.notes}


def preflight(
    adapter: Any, server: dict[str, Any] | None, checks: dict[str, Any] | None, read: dict[str, Any] | None = None,
) -> Preflight:
    """Check a harness's requirements against the server's identity and what a server
    check measured. What was never measured is a note, not a refusal. A server that asks for
    an API key (`read`, the proxy's own read of it) refuses every call a launched harness
    makes, since one carries none of the person's: that is the one thing said of it — no
    check could measure the rest there."""
    result = Preflight()
    server = server or {}
    measured = (checks or {}).get("results") or {}
    name = getattr(adapter, "name", "the harness")
    if (read or {}).get("asks_key"):
        result.problems.append(
            f"this server asks for an API key, and a harness tuningfork launches carries none of yours, so every "
            f"call {name} made would be refused"
        )
        return result
    for requirement in getattr(adapter, "requirements", ()):
        if requirement.minimum is not None:
            value = server.get(requirement.what)
            if not isinstance(value, (int, float)):
                result.notes.append(
                    f"the server does not report its {requirement.what}; {name} needs at least "
                    f"{requirement.minimum:,} ({requirement.why})"
                )
            elif value < requirement.minimum:
                result.problems.append(
                    f"{name} needs {requirement.what} of at least {requirement.minimum:,}, and the "
                    f"server has {value:,}: {requirement.why}"
                )
            continue
        found = measured.get(requirement.what)
        support = found.get("support") if isinstance(found, dict) else None
        if support == "no":
            result.problems.append(
                f"{name} needs {requirement.what.replace('_', ' ')}, and the server check found "
                f"none ({found.get('detail') or 'no detail'}): {requirement.why}"
            )
        elif support != "yes":
            result.notes.append(
                f"{requirement.what.replace('_', ' ')} was not measured on this server; {name} "
                f"needs it ({requirement.why}) — a server check measures it"
            )
    return result


def cached_prompts_warning(checks: dict[str, Any] | None) -> str | None:
    """What to say when a server check found the server answering prompts from its cache."""
    found = ((checks or {}).get("results") or {}).get("fresh_prompts")
    if isinstance(found, dict) and found.get("support") == "no":
        return ("the server reuses the prompts it processes: each run's calls, and every fork's, depend on "
                "what it ran before (start llama-server with --no-cache-prompt)")
    return None


#: Where a refusal for want of a key points: why tuningfork wants a model run oneself.
LOCAL_MODEL_HINT = ("tuningfork is built for a model you run yourself: see “local model” on the Words page, or "
                    "`tuningfork glossary local model`")


def check_harness(store: Any, adapter: Any, proxy: Any) -> list[str]:
    """Refuse a launch its server cannot serve; return the notes that do not stop it."""
    server = proxy.raw.get("server") or {}
    read = proxy.raw.get("server_read") or {}
    found = preflight(adapter, server, store.server_checks(server.get("id")), read)
    if found.problems:
        raise OperationError(
            "; ".join(found.problems),
            hint=LOCAL_MODEL_HINT if read.get("asks_key")
            else "point the proxy at a server that has it (Connections), or choose another harness",
            code="preflight",
            field="harness",
        )
    return found.notes


# --- the server check -------------------------------------------------------------------------


@dataclass(slots=True)
class ServerCheckPlan:
    server_id: str
    upstream: str
    model: str
    #: Where the check's calls go: the proxy, which sends them on as an experiment's.
    base_url: str

    #: What a harness needs, then what token-level work needs.
    checks: tuple[str, ...] = (
        "streaming_tool_calls", "anthropic_messages", "fresh_prompts", "output_format", "prompt_equivalence",
        "raw_completion", "logprobs",
    )

    def to_json(self) -> dict[str, Any]:
        return {"server_id": self.server_id, "upstream": self.upstream, "model": self.model,
                "checks": list(self.checks)}


def prepare_server_check(proxy: Any, model: str | None = None) -> ServerCheckPlan:
    server = proxy.raw.get("server") or {}
    problem = proxy.raw.get("server_problem")
    if (proxy.raw.get("server_read") or {}).get("asks_key"):
        # Up, and refusing tuningfork: not "start it", which a server down or loading needs.
        raise OperationError(
            f"the server at {server.get('upstream') or 'the upstream'} asks for an API key, and tuningfork sends "
            "none of its own: it cannot be checked",
            hint=LOCAL_MODEL_HINT,
            code="asks_key",
        )
    if problem:
        # Found in use: a server that was down, or still loading, was reported as a missing
        # model name — the model list was empty because nothing had answered.
        raise OperationError(
            f"the server at {server.get('upstream') or 'the upstream'} is not answering: {problem}",
            hint="start it, or let it finish loading its model, then check again",
            code="server_down",
        )
    if not server.get("id"):
        raise OperationError("the proxy has not identified its server yet", hint="try again in a moment")
    models = [m for m in server.get("models") or [] if isinstance(m, str)]
    if not model:
        if len(models) != 1:
            raise OperationError(
                f"the server lists {len(models)} models: choose the one to check with" if models
                else "the server lists no model: name the one to check with",
                field="model", hint=", ".join(models) if models else None,
            )
        model = models[0]
    return ServerCheckPlan(server_id=server["id"], upstream=server.get("upstream") or "", model=model, base_url=proxy.url)


async def execute_server_check(store: Any, plan: ServerCheckPlan, *, progress: Any = None) -> dict[str, Any]:
    """Small calls, one at a time: what the harnesses need — a streaming tool call and a
    Messages call — then what token-level work needs: a known output format, a prompt rendered
    as the chat endpoint renders it, a raw completion with forced tokens, and the chat
    endpoint's own token probabilities."""
    import httpx

    from tuningfork.backends.probe import (
        TOKEN_PROBES,
        ProbeContext,
        probe_anthropic_messages,
        probe_fresh_prompts,
        probe_streaming_tool_calls,
    )
    from tuningfork.config import EXPERIMENT_HEADER
    from tuningfork.endpoints import guard_hooks
    from tuningfork.ops.common import emit

    probes = [("streaming_tool_calls", probe_streaming_tool_calls), ("anthropic_messages", probe_anthropic_messages),
              ("fresh_prompts", probe_fresh_prompts)]
    probes += [(name, probe) for name, probe in TOKEN_PROBES if name in plan.checks]
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(
        headers={EXPERIMENT_HEADER: "server-check"}, event_hooks=guard_hooks(home=store.home),
    ) as client:
        context = ProbeContext(client=client, root=plan.base_url, model=plan.model, timeout=120.0)
        for name, probe in probes:
            try:
                found = await probe(context)
            except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
                results[name] = {"support": "unknown", "detail": f"{type(exc).__name__}: {exc}"}
                emit(progress, "check", name=name, support="unknown")
                continue
            for result in found:
                results[str(result.capability)] = {
                    "support": result.support.value, "detail": result.detail, "evidence": result.evidence,
                }
                emit(progress, "check", name=str(result.capability), support=result.support.value)
    checks = {"checked_at": time.time(), "model": plan.model, "results": results}
    store.set_server_checks(plan.server_id, checks)
    return {"server_id": plan.server_id, **checks}


# --- token work, held to the check ------------------------------------------------------------

#: What each of a server check's measurements is, and what needs it: the Connections page's
#: table, and the words a refusal of token work uses (`token_work`).
MEASURES: dict[str, tuple[str, str]] = {
    "streaming_tool_calls": ("tool calls while streaming", "Hermes, which streams every call and acts on its tool calls"),
    "anthropic_messages": ("the Messages endpoint, /v1/messages", "Claude Code, which speaks only /v1/messages"),
    "fresh_prompts": ("every prompt processed whole", "samples and forks that do not depend on what the server ran before"),
    "known_output_format": ("a model whose raw output tuningfork reads", "scoring an answer and forking tokens: each token read as the model wrote it"),
    "prompt_equivalence": ("prompts rendered as the chat endpoint renders them", "scoring an answer against the exact prompt its step was sent"),
    "raw_completion": ("a raw completion with each token's probabilities", "scoring an answer (its heatmap) and forking tokens: tokens forced, each one's chance read"),
    "logprobs": ("token probabilities from the chat endpoint (logprobs)", "keeping each sample's tokens in a resample, for its heatmap"),
    "top_logprobs": ("the alternatives at each token (top_logprobs)", "what else the model weighed at each of a sample's tokens"),
    "logprobs_all_tokens": ("probabilities for every token generated", "a sample's heatmap over its reasoning and tool calls, not only its reply"),
}

#: Each kind of token work: what it is called, the measurements it needs, and the server that
#: has them. Scoring renders the step's prompt and forces the answer through llama.cpp's own
#: endpoints; forking tokens forces tokens the same way; a resample keeps each sample's tokens
#: from the chat endpoint's logprobs.
TOKEN_WORK: dict[str, tuple[str, tuple[str, ...], str]] = {
    "score": ("Scoring an answer", ("raw_completion", "prompt_equivalence", "known_output_format"),
              f"llama.cpp's server running a model tuningfork reads ({known_models()})"),
    "forking": ("Forking tokens", ("raw_completion", "known_output_format"),
                f"llama.cpp's server running a model tuningfork reads ({known_models()})"),
    "logprobs": ("Keeping each sample's tokens", ("logprobs",), "a server whose chat endpoint returns logprobs, as llama.cpp's does"),
}


def token_work(checks: dict[str, Any] | None, read: dict[str, Any] | None = None) -> dict[str, str | None]:
    """Why the server's check says each kind of token work cannot be done on it, or None.
    Only a measurement that found the server without something is a reason: one never made,
    or that could not tell, leaves the work to start and the server to say. The proxy's own
    read of its server (`read`, every half minute) is one too: a server that answered without
    llama.cpp's /props cannot score or force tokens, checked or not."""
    measured = (checks or {}).get("results") or {}
    no_props = (read or {}).get("props") is False
    reasons: dict[str, str | None] = {}
    for work, (what, needs, _) in TOKEN_WORK.items():
        missing = [name for name in needs if (measured.get(name) or {}).get("support") == "no"]
        if missing:
            details = "; ".join(measured[name].get("detail") or "no detail" for name in missing)
            reasons[work] = (
                f"{what} needs {' and '.join(MEASURES[name][0] for name in missing)}: the server check found this "
                f"server without {'it' if len(missing) == 1 else 'them'} ({details})"
            )
        elif no_props and "raw_completion" in needs:
            reasons[work] = f"{what} needs llama.cpp's server: this one has no /props"
        else:
            reasons[work] = None
    return reasons


def refuse_token_work(store: Any, proxy: Any, work: str, *, field: str | None = None) -> None:
    """Refuse token work the proxy's server was measured unable to do, before anything is sent."""
    server = proxy.raw.get("server") or {}
    reason = token_work(store.server_checks(server.get("id")), proxy.raw.get("server_read")).get(work)
    if reason:
        raise OperationError(
            reason, code="unsupported", field=field,
            hint=f"on Connections, point the proxy at {TOKEN_WORK[work][2]}, or check this server again if it has changed",
        )


# --- a model run oneself ------------------------------------------------------------------------


def reminder(read: dict[str, Any] | None) -> dict[str, str] | None:
    """What to say of a server on the internet that answers as a hosted model API does —
    asking for a key tuningfork does not send, or without llama.cpp's /props — or None.
    Never a refusal: a harness's calls are recorded through any server, its own key passed on.
    Measured, never a list of providers: a llama.cpp server rented on the internet, answering
    without a key, is a model run oneself, and is said nothing of."""
    read = read or {}
    if read.get("internet") is not True:
        return None
    if read.get("asks_key"):
        headline = "This server is on the internet and asks for an API key."
    elif read.get("props") is False:
        headline = "This server is on the internet, and is not llama.cpp's."
    else:
        return None
    return {
        "headline": headline,
        "body": "tuningfork is built for a model you run yourself. It records your agent's calls through this "
                "server — your agent sends its own key — but what tuningfork runs itself (resamples, A/Bs, forks, "
                "batches) sends none, and scoring and forking tokens need llama.cpp.",
    }


async def read_reminder(url: str, *, client: Any = None) -> dict[str, str] | None:
    """The reminder for a server no proxy reads yet — what `serve` says before it starts. The
    server is asked only when it is on the internet."""
    import httpx

    from tuningfork.endpoints import find_on_internet, guard_hooks
    from tuningfork.proxy.recorder import read_upstream

    internet = await find_on_internet(url)
    if not internet:
        return None
    owned = client is None
    client = client or httpx.AsyncClient(event_hooks=guard_hooks())
    try:
        read = await read_upstream(client, url)
    finally:
        if owned:
            await client.aclose()
    return reminder({**read.facts(), "internet": internet})
