"""The tools over Streamable HTTP, on the running proxy, at `/_tuningfork/mcp`.

For a client that connects to a server rather than starting one: OpenJarvis's `url` and
`token`, Claude Code's `--transport http`. The route sits behind the proxy's guard like
every endpoint under `/_tuningfork/` — a loopback Host, no cross-site Origin — and needs the
access token on every request, GET included, as `x-tuningfork-token` or a bearer token:
MCP is a channel, not a page to read. The SDK's own host check is left off; the guard's is
stricter.

The SDK's session manager runs in a task of its own, started by the first request, so the
route works whether or not the app's lifespan ran, and loads the SDK only when used.
`stop()` ends every open stream at once: a connected client cannot keep the proxy from
stopping.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse


class McpRoute:
    """The MCP endpoint, as one ASGI app for one route."""

    def __init__(
        self,
        *,
        store: Any,
        runner: Any,
        local: Any,
        health: Callable[[], dict[str, Any]],
        toolset: str = "evidence",
        max_wait: int = 50,
        results: str = "both",
    ) -> None:
        self.store, self.runner, self.local, self.health = store, runner, local, health
        self.toolset, self.max_wait, self.results = toolset, max_wait, results
        self._app: Any = None
        self._manager: Any = None
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Event | None = None
        self._stopping: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._base_url = ""

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if importlib.util.find_spec("mcp") is None:
            await _refuse(501, "this tuningfork has no MCP SDK: install tuningfork[mcp]", "mcp_unavailable")(scope, receive, send)
            return
        self._loop = asyncio.get_running_loop()
        # The address the client reached is the proxy's own: where a job's model calls go.
        self._base_url = str(Request(scope).base_url).rstrip("/")
        if self._task is None:
            self._begin()
        assert self._ready is not None and self._stopping is not None and self._task is not None
        await self._ready.wait()
        if self._stopping.is_set() or self._task.done():
            await _refuse(503, "tuningfork's MCP server is not running: the proxy is stopping", "stopping")(
                scope, receive, send
            )
            return
        await self._app(scope, receive, send)

    def _begin(self) -> None:
        from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager

        from tuningfork.mcp.backend import InProcessJobs
        from tuningfork.mcp.server import ToolServer

        jobs = InProcessJobs(
            self.runner, status=self.health, attach=lambda: self.local.attach(self._loop, self._base_url)
        )
        tools = ToolServer(self.store, jobs, toolset=self.toolset, max_wait=self.max_wait, results=self.results)
        # No security settings: the SDK's host check is off, and the proxy's guard does it.
        self._manager = StreamableHTTPSessionManager(app=tools.server(), security_settings=None)
        self._app = StreamableHTTPASGIApp(self._manager)
        self._ready, self._stopping = asyncio.Event(), asyncio.Event()
        self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        assert self._ready is not None and self._stopping is not None
        try:
            async with self._manager.run():
                self._ready.set()
                await self._stopping.wait()
        finally:
            self._ready.set()  # a request waiting on a manager that failed to start is answered

    def stop(self) -> None:
        """End every open stream. Call it on the proxy's event loop."""
        if self._stopping is not None:
            self._stopping.set()


def _refuse(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": code}}, status_code=status)
