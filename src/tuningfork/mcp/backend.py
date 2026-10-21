"""Where an MCP server sends work: the job runner of the proxy it runs in, or of one it talks to.

Reads never come here — they run on the recordings, in the MCP server's own process, and
need no proxy at all. Work that calls the model goes to a proxy's job runner, so it takes
the proxy's gate and pause switch, and shows on its Jobs page like work started there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from tuningfork.ops.common import OperationError


class Jobs(Protocol):
    """A proxy's job runner, as the MCP tools use it. Every method blocks: call it in a
    worker thread."""

    def plan(self, kind: str, params: dict[str, Any]) -> dict[str, Any]: ...

    def submit(self, kind: str, params: dict[str, Any], *, started_by: dict[str, Any]) -> dict[str, Any]: ...

    def get(self, job_id: str, *, after: int | None = None) -> dict[str, Any] | None: ...

    def list(self, limit: int = 50) -> list[dict[str, Any]]: ...

    def cancel(self, job_id: str) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]:
        """The proxy as `/health` describes it: its upstream, the server answering, why the
        server could not be read, whether it is paused."""
        ...


class InProcessJobs:
    """Inside the proxy (the HTTP route, and tests): its own job runner.

    `attach` tells the runner the proxy's address before work is planned, as `POST /jobs`
    does — a job's model calls go back to the proxy itself."""

    def __init__(
        self,
        runner: Any,
        *,
        status: Callable[[], dict[str, Any]],
        attach: Callable[[], None] | None = None,
    ) -> None:
        self.runner = runner
        self._status = status
        self._attach = attach or (lambda: None)

    def plan(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        self._attach()
        return self.runner.plan(kind, params)

    def submit(self, kind: str, params: dict[str, Any], *, started_by: dict[str, Any]) -> dict[str, Any]:
        self._attach()
        return self.runner.submit(kind, params, started_by=started_by, dedupe=True)

    def get(self, job_id: str, *, after: int | None = None) -> dict[str, Any] | None:
        return self.runner.get(job_id, after=after)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.runner.list(limit)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.runner.cancel(job_id)

    def status(self) -> dict[str, Any]:
        return self._status()


def refusal(error: Any) -> OperationError:
    """A proxy's refusal (`control.ControlError`) as the operation's own, field and all."""
    found = getattr(error, "error", None) or {}
    return OperationError(
        found.get("message") or str(error),
        hint=found.get("hint"),
        detail=found.get("detail"),
        code=found.get("code") or ("no_proxy" if getattr(error, "status", None) is None else None),
        field=found.get("field"),
    )


class HttpJobs:
    """Over stdio: the jobs API of a running proxy that records into the same home.

    The proxy is checked when work is first asked for, not when the MCP server starts: a
    person may start `tuningfork serve` after the agent, and reading recordings needs no
    proxy. While there is none, or it serves other recordings, every job tool says so and
    what to do; the next call looks again."""

    def __init__(self, proxy_url: str, home: Path | str) -> None:
        self.proxy_url = proxy_url.rstrip("/")
        self.home = Path(home)
        self._client: Any = None

    def _connected(self) -> Any:
        from tuningfork.control import ControlClient
        from tuningfork.ops.launch import check_proxy

        if self._client is None:
            try:
                check_proxy(self.proxy_url, self.home)
            except OperationError as error:
                none = error.message.startswith("there is no tuningfork proxy")
                raise OperationError(
                    error.message,
                    hint=f"start one with `tuningfork serve --home {self.home}`" if none else error.hint,
                    code="no_proxy",
                ) from None
            self._client = ControlClient(self.proxy_url)
        return self._client

    def _call(self, use: Callable[[Any], Any]) -> Any:
        from tuningfork.control import ControlError

        client = self._connected()
        try:
            return use(client)
        except ControlError as error:
            if error.status is None:  # the proxy went away: look again next time
                client.close()
                self._client = None
            raise refusal(error) from None

    def plan(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._call(lambda client: client.plan(kind, params))

    def submit(self, kind: str, params: dict[str, Any], *, started_by: dict[str, Any]) -> dict[str, Any]:
        return self._call(lambda client: client.submit(kind, params, started_by=started_by, dedupe=True))

    def get(self, job_id: str, *, after: int | None = None) -> dict[str, Any] | None:
        from tuningfork.control import ControlError

        client = self._connected()
        try:
            return client.job(job_id, after=after)
        except ControlError as error:
            if error.status == 404 or (error.error or {}).get("code") == "not_found":
                return None
            if error.status is None:
                client.close()
                self._client = None
            raise refusal(error) from None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        # As the proxy's runner lists them in process: the newest, then every job still under way.
        jobs = self._call(lambda client: client.jobs())
        return jobs[:limit] + [job for job in jobs[limit:] if job.get("status") in ("queued", "running")]

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._call(lambda client: client.cancel_job(job_id))

    def status(self) -> dict[str, Any]:
        from tuningfork.ops.launch import check_proxy

        try:
            return check_proxy(self.proxy_url, self.home).raw
        except OperationError as error:
            raise OperationError(error.message, hint=error.hint, code="no_proxy") from None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
