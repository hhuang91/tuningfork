"""Launching harnesses: checks made before any harness is started."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from tuningfork.launches import proxy_root
from tuningfork.ops.common import OperationError


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
