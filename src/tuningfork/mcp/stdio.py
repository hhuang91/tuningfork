"""`tuningfork mcp`: the tools over stdio, for a client that starts its own servers.

Reads open the recordings directly, so an agent can study them with no proxy running. Work
goes to the proxy that records into the same folder (`backend.HttpJobs`). Nothing but the
protocol reaches stdout: while it serves, the SDK points file descriptor 1 at stderr, so a
stray print anywhere — ours, a library's, a child process's — cannot corrupt the wire.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from tuningfork.ops.common import OperationError


def proxy_home(proxy_url: str) -> Path | None:
    """The folder the proxy at `proxy_url` records into, if one answers."""
    try:
        response = httpx.get(f"{proxy_url.rstrip('/')}/_tuningfork/health", timeout=httpx.Timeout(5.0, connect=3.0))
        health = response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
    home = health.get("home") if isinstance(health, dict) else None
    return Path(home) if home else None


def resolve_home(home: Path | None, proxy_url: str) -> Path:
    """The recordings to read: `home` as given, else the running proxy's, else the default
    folder. One with no recordings is refused — a store opened there would be made empty."""
    from tuningfork.config import default_home

    if home is None:
        home = proxy_home(proxy_url) or default_home()
    home = Path(home).expanduser()
    if not (home / "tuningfork.db").is_file():
        raise OperationError(
            f"there are no recordings at {home}",
            hint="pass --home: the folder `tuningfork serve` records into",
        )
    return home


def serve(
    *,
    home: Path | None,
    proxy_url: str,
    toolset: str = "evidence",
    limits: dict[str, int] | None = None,
    max_jobs: int = 4,
    max_wait: int = 120,
    results: str = "both",
) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    from tuningfork.mcp.backend import HttpJobs
    from tuningfork.mcp.server import ToolServer
    from tuningfork.store.db import Store

    logging.basicConfig(level=logging.WARNING, format="tuningfork mcp: %(message)s")
    found = resolve_home(home, proxy_url)
    store = Store(found)
    jobs = HttpJobs(proxy_url, found)
    try:
        tools = ToolServer(store, jobs, toolset=toolset, limits=limits, max_jobs=max_jobs, max_wait=max_wait,
                           results=results)
        server = tools.server()

        async def main() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        anyio.run(main)
    finally:
        jobs.close()
        store.close()


def parse_limits(values: list[str]) -> dict[str, int]:
    """`--limit resample=200` as {kind: N}, for a kind an agent can be offered."""
    from tuningfork.mcp.schema import EVIDENCE, HARNESS

    limits: dict[str, Any] = {}
    for value in values:
        kind, _, number = value.partition("=")
        if kind not in (*EVIDENCE, *HARNESS):
            raise OperationError(
                f"--limit {value!r}: no job kind {kind!r} an agent can start",
                hint=f"known: {', '.join((*EVIDENCE, *HARNESS))}",
            )
        if not number.isdigit() or int(number) < 1:
            raise OperationError(f"--limit {value!r}: the limit is a whole number of at least 1")
        limits[kind] = int(number)
    return limits
