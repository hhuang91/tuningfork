"""What tuningfork talks to: the inference server, and the harnesses it launches.

A harness is found on PATH, or at an executable the recordings folder's `config.json`
names — never by reading or writing the harness's own configuration, which belongs to a
real installation that may be in use. Its version is read with `--version` in a throwaway
home, so even that leaves nothing behind.

Before a batch or a fork launches a harness, its `requirements` of the server are checked
(`preflight`): a context too small for Hermes, or no `/v1/messages` for Claude Code, is
refused before a launch is recorded, not found out from a failed run.
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


def _version(path: str) -> tuple[str | None, str | None]:
    """A harness's `--version`, run where it can touch nothing real: a throwaway home, its
    own config dirs pointed there, no input."""
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
    return (lines[0] if lines else None), None


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
        version, error = _version(path)
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


def preflight(adapter: Any, server: dict[str, Any] | None, checks: dict[str, Any] | None) -> Preflight:
    """Check a harness's requirements against the server's identity and what a server
    check measured. What was never measured is a note, not a refusal."""
    result = Preflight()
    server = server or {}
    measured = (checks or {}).get("results") or {}
    for requirement in getattr(adapter, "requirements", ()):
        name = getattr(adapter, "name", "the harness")
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


def check_harness(store: Any, adapter: Any, proxy: Any) -> list[str]:
    """Refuse a launch its server cannot serve; return the notes that do not stop it."""
    server = proxy.raw.get("server") or {}
    found = preflight(adapter, server, store.server_checks(server.get("id")))
    if found.problems:
        raise OperationError(
            "; ".join(found.problems),
            hint="point the proxy at a server that has it (Connections), or choose another harness",
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
        "raw_completion",
    )

    def to_json(self) -> dict[str, Any]:
        return {"server_id": self.server_id, "upstream": self.upstream, "model": self.model,
                "checks": list(self.checks)}


def prepare_server_check(proxy: Any, model: str | None = None) -> ServerCheckPlan:
    server = proxy.raw.get("server") or {}
    problem = proxy.raw.get("server_problem")
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
    as the chat endpoint renders it, and a raw completion with forced tokens."""
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
