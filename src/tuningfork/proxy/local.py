"""The running proxy, as work started inside it sees it.

A job runs on a worker thread of the proxy's own process. It must not ask the proxy things
over HTTP that it can read directly, and it must not touch what belongs to the event loop —
held calls, the event bus — except by scheduling onto that loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from tuningfork.ops.common import OperationError
from tuningfork.ops.launch import ProxyStatus

logger = logging.getLogger("tuningfork.proxy")


class LocalProxy:
    def __init__(self, app: Any) -> None:
        self.app = app
        #: The address the proxy answers at, and the loop it runs on: learned from the
        #: request that starts work, which reached it there. Tests bind random ports, and
        #: a wildcard bind names no address, so neither is taken from the settings.
        self.url: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop, url: str) -> None:
        self.loop, self.url = loop, url.rstrip("/")

    @property
    def paused(self) -> bool:
        return bool(self.app.state.gate.paused)

    @property
    def home(self) -> Path:
        return Path(self.app.state.settings.home)

    def status(self) -> ProxyStatus:
        """What `ops.launch.check_proxy` would learn over HTTP, read in-process."""
        if not self.url:
            raise OperationError("the proxy does not know its own address yet")
        state = self.app.state
        raw = {
            "status": "ok",
            "home": str(self.home),
            "max_inflight": state.gate.max_inflight,
            "max_output": state.settings.max_output,
            "paused": state.gate.paused,
            "launches": True,
            "server": getattr(state, "server", None),
            "server_problem": getattr(state, "server_problem", None),
        }
        return ProxyStatus(url=self.url, home=self.home, max_inflight=state.gate.max_inflight, raw=raw)

    def refresh_server(self, timeout: float = 20.0) -> None:
        """Read the server's identity again now, as the proxy does every half minute: work
        about to measure the server must not go by a reading that old. From a worker thread
        only; a failure leaves the last reading."""
        loop, refresh = self.loop, getattr(self.app.state, "refresh_server", None)
        if loop is None or loop.is_closed() or refresh is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(refresh(), loop).result(timeout)
        except Exception:  # noqa: BLE001 - a diagnostic; the last reading stands
            logger.warning("tuningfork could not read the server's identity", exc_info=True)

    def call_soon(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run `callback` on the proxy's event loop, from any thread. Dropped once the loop
        has closed: the proxy is gone, and so is anyone to tell."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(functools.partial(callback, *args, **kwargs))
        except RuntimeError:
            pass

    def run_on_loop(self, callback: Callable[..., Any], *args: Any, timeout: float = 10.0) -> Any:
        """Run `callback` on the proxy's event loop and wait for what it returns. From a
        worker thread only: on the loop itself this would wait on itself."""
        loop = self.loop
        if loop is None or loop.is_closed():
            raise OperationError("the proxy is not running")
        done: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def call() -> None:
            try:
                done.set_result(callback(*args))
            except BaseException as error:  # noqa: BLE001 - handed to the waiting thread
                done.set_exception(error)

        loop.call_soon_threadsafe(call)
        return done.result(timeout)

    def held_call(self, hold_id: str) -> dict[str, Any]:
        """A call held at a breakpoint, as it stands: a copy, read on the proxy's loop."""
        import copy

        from tuningfork.breakpoints import HoldError

        engine = self.app.state.engine

        def read() -> dict[str, Any]:
            hold = engine.get_hold(hold_id)
            return {
                **hold.summary(),
                "request": copy.deepcopy(hold.request),
                "response": copy.deepcopy(hold.response),
            }

        try:
            return self.run_on_loop(read)
        except HoldError as error:
            raise OperationError(str(error), code="not_found", field="hold") from None

    def drop_holds_of(self, launch_ids: Iterable[str]) -> None:
        """Fail the held calls of these launches: a cancelled fork waiting at a breakpoint
        would otherwise wait for a decision no one is left to make."""
        wanted = set(launch_ids)
        if not wanted:
            return
        engine = self.app.state.engine

        def drop() -> None:
            for hold in engine.holds():
                if hold.launch_id in wanted:
                    try:
                        engine.resolve(hold.id, "drop")
                    except Exception:  # noqa: BLE001 - decided meanwhile
                        logger.debug("hold %s was decided before it could be dropped", hold.id)

        self.call_soon(drop)
