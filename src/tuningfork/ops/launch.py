"""Launching harnesses: checks made before any harness is started, and how a recorded run's
harness was set up, for the runs set up as it was."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from tuningfork.launches import proxy_root
from tuningfork.ops.common import OperationError


@dataclass(frozen=True, slots=True)
class Setup:
    """How a recorded run's harness was set up, read from its launch as a fork of the run is
    set up (`ops.forks.prepare_fork`). Each part of it changes what the model reads, so the
    runs measured beside the forks are set up the same way — a validation's fresh runs
    (`prepare_batch`'s `setup`) — and a screen pools the forks of runs set up alike
    (`ops.screen.choose_checkpoints`)."""

    harness: str | None
    model: str | None
    #: The run it was read from, for what is said of it.
    run: str | None = None
    #: The harness's options as the launch recorded them (`settings["harness"]`), its MCP
    #: servers among them, which a launch records by name only.
    recorded: dict[str, Any] = field(default_factory=dict)
    #: How the adapter laid the launch out (`settings["layout"]`): None for an adapter that
    #: has one layout, or a launch recorded before layouts were, which was laid out flat.
    layout: str | None = None
    #: The sandbox the launch ran in, as it recorded it (`Sandbox.to_json`), its clock among
    #: it; None: it ran in its own folders.
    sandbox: dict[str, Any] | None = None

    @property
    def options(self) -> dict[str, Any]:
        """The harness options a launch can be given again: all but the MCP servers."""
        return {key: value for key, value in self.recorded.items() if key != "mcp_servers"}

    @property
    def dropped_servers(self) -> list[str]:
        """The MCP servers it had, by name, which no launch set up as it was is given."""
        return sorted(self.recorded.get("mcp_servers") or ())

    @property
    def clock(self) -> str | None:
        """The instant its clock started at, or None: the real clock."""
        return (self.sandbox or {}).get("clock")

    def differences(self, other: Setup) -> list[str]:
        """What `other` was set up with that this was not, each in a few words; none when
        they are alike."""
        found: list[str] = []
        if other.harness != self.harness:
            found.append(f"harness {other.harness}, not {self.harness}")
        if other.model != self.model:
            found.append(f"model {other.model}, not {self.model}")
        if other.recorded != self.recorded:
            found.append(f"harness options {_options(other.recorded)}, not {_options(self.recorded)}")
        if other.layout != self.layout:
            found.append(f"layout {other.layout or 'flat'}, not {self.layout or 'flat'}")
        if (other.sandbox is None) != (self.sandbox is None):
            found.append("in the sandbox, not without it" if other.sandbox is not None
                         else "without the sandbox, not in it")
        elif other.clock != self.clock:
            found.append(f"its clock at {other.clock or 'the real time'}, not {self.clock or 'the real time'}")
        return found

    def to_json(self) -> dict[str, Any]:
        """As the launch recorded it, beside the harness and the model: a validation's plan and
        record give the clock as its runs will read it instead (`ValidatePlan.setup_json`)."""
        return {
            "harness_options": self.options, "dropped_servers": self.dropped_servers, "layout": self.layout,
            "sandboxed": self.sandbox is not None, "clock": self.clock,
        }


def _options(options: dict[str, Any]) -> str:
    return json.dumps(options, sort_keys=True) if options else "none"


def setup_of(store: Any, run: Any) -> Setup:
    """How a run's harness was set up, from its launch, as a fork of the run reads it: a run
    whose launch kept no task — recorded before launches kept what each was launched with —
    had the harness the run names."""
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    settings = (launch.settings if launch is not None else None) or {}
    return Setup(
        harness=launch.harness if launch is not None and launch.eval else run.harness,
        model=run.model,
        run=run.id,
        recorded=dict(settings.get("harness") or {}),
        layout=settings.get("layout"),
        sandbox=settings.get("sandbox") or None,
    )


@dataclass(slots=True)
class ProxyStatus:
    url: str
    home: Path
    max_inflight: int
    raw: dict[str, Any]


def check_proxy(base_url: str, home: Path, *, timeout: float = 5.0) -> ProxyStatus:
    """The proxy a launched harness will talk to: that it is one, that it serves launches,
    and that it records into the same home this command reads.

    A launched run is found again by its launch, which the proxy must have recorded where
    this command looks. With different homes every run used to come back as "no model call
    was recorded", after the whole batch had run.
    """
    from tuningfork.endpoints import check_endpoint

    check_endpoint(base_url, home=home, purpose="a harness launched by tuningfork")
    root = proxy_root(base_url)
    try:
        response = httpx.get(f"{root}/_tuningfork/health", timeout=timeout)
        health = response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        health = None
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise OperationError(
            f"there is no tuningfork proxy at {root}",
            hint="start one with `tuningfork serve`, or point --base-url at it",
        )
    if not health.get("launches"):
        raise OperationError(
            f"the proxy at {root} is an older tuningfork that cannot tell launches apart",
            hint="restart it with this version",
        )
    served = Path(str(health.get("home") or ""))
    if served.resolve() != Path(home).resolve():
        raise OperationError(
            f"the proxy at {root} records into {served}, but this command reads {home}",
            hint=f"pass --home {served}, or restart the proxy with --home {home}",
        )
    return ProxyStatus(
        url=root, home=served, max_inflight=int(health.get("max_inflight") or 1), raw=health
    )


def check_launchable(adapter: Any) -> None:
    """That tuningfork can start this harness here: an adapter that launches headlessly, an
    executable that resolves, and nothing its `--version` says that stops it launching (OpenClaw
    on a Node it does not run on) — refused before a launch is recorded, not by its first run."""
    import shutil

    from tuningfork.harness.base import GenericHarness
    from tuningfork.ops.connections import launch_problem

    if isinstance(adapter, GenericHarness):
        raise OperationError(
            "tuningfork does not know how to launch this harness headlessly",
            hint="run it yourself against the proxy, or add an adapter (docs/adapters/TEMPLATE.md)",
            field="harness",
        )
    executable = getattr(adapter, "executable", None)
    if executable and shutil.which(executable) is None:
        raise OperationError(
            f"{executable!r} was not found. Is {adapter.name} installed, and on PATH for this "
            "shell?",
            field="harness",
        )
    said = launch_problem(adapter)
    if said:
        raise OperationError(said, field="harness")


def check_clock(adapter: Any, sandbox: Any, *, hint: str) -> None:
    """That the harness runs with the clock the sandbox pins, refused before a launch is
    recorded if it does not (`ops.connections.clock_problem`): Claude Code's native binary on
    x86-64, whose every run would hang until its task's timeout."""
    from tuningfork.ops.connections import clock_problem

    said = clock_problem(adapter, sandbox)
    if said:
        raise OperationError(said, hint=hint, field="harness")
