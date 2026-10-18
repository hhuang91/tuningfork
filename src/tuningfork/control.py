"""A Python client for a running proxy's control API.

The CLI's breakpoint and hold commands are thin wrappers around this, and it is the
intended surface for scripts — and for the later phase where an LLM drives tuningfork.
Everything the web UI can do is reachable from here.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from tuningfork.access import ENV as TOKEN_ENV
from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.access import MCP_ENV, MCP_TOKEN_FILE, TOKEN_FILE, read_token

DEFAULT_PROXY = "http://127.0.0.1:8787"
#: How long to wait for a connection. Locally one is made at once; under WSL's mirrored
#: networking a port nothing listens on hangs instead of refusing, until this runs out.
CONNECT_TIMEOUT = 3.0
#: Planning or starting a job prepares it first, which can take a while: a server check
#: reads the server again, a fork checks its whole replay.
PREPARE_TIMEOUT = 60.0


def _id(kind: str, value: str) -> str:
    """Breakpoint and hold ids (and their prefixes) are hexadecimal. Anything else — a phrase
    pasted by mistake — is refused here, where the message can say so plainly: in a URL, a
    stray `/` would turn it into a different route altogether."""
    if not re.fullmatch(r"[0-9a-f]+", str(value)):
        raise ControlError(f"no single {kind} matching {value!r}: ids are hexadecimal")
    return str(value)


class ControlError(RuntimeError):
    """The proxy refused a request, or there is no proxy to talk to.

    `error` is the refusal as the proxy gave it — message, hint, code, field — so a front
    end can point at the parameter at fault; None when there was no proxy to refuse."""

    def __init__(self, message: str, *, error: dict[str, Any] | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.status = status


def _loopback(base_url: str) -> bool:
    """Whether the proxy at `base_url` is on this machine, where its recordings folder is."""
    from tuningfork.endpoints import is_loopback

    return is_loopback(httpx.URL(base_url).host or "")


def _code(response: httpx.Response) -> str | None:
    """A refusal's machine-readable reason, when the proxy gave one."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None
    return error.get("code") if isinstance(error, dict) else None


def default_proxy() -> str:
    return os.environ.get("TUNINGFORK_PROXY", DEFAULT_PROXY)


class ControlClient:
    """Talks to a running proxy. Changing anything needs the proxy's access token: given
    here, else `TUNINGFORK_TOKEN`, else read from the file the proxy names at `/health` —
    which only the user it runs as can read."""

    def __init__(
        self, base_url: str | None = None, timeout: float = 10.0, *, token: str | None = None
    ) -> None:
        self.base_url = (base_url or default_proxy()).rstrip("/")
        self._given = token
        self._token: str | None = None
        self._looked = False
        if not re.match(r"https?://", self.base_url):
            # `localhost:8787` would otherwise surface as httpx's UnsupportedProtocol.
            raise ControlError(
                f"{self.base_url!r} is not a URL — write it with its scheme, like "
                "http://127.0.0.1:8787"
            )
        from tuningfork.endpoints import guard_hooks

        self._http = httpx.Client(
            base_url=f"{self.base_url}/_tuningfork/api",
            timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT)),
            event_hooks=guard_hooks(sync=True),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ControlClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def token(self) -> str | None:
        """The access token this client sends, found the first time it is needed."""
        if not self._looked:
            self._looked = True
            self._token = self._given or os.environ.get(TOKEN_ENV) or self._discover()
        return self._token

    @property
    def mcp_token(self) -> str | None:
        """The MCP endpoint's token: `TUNINGFORK_MCP_TOKEN`, else read from the file the proxy
        names at `/health`. It opens `/_tuningfork/mcp`, and nothing this client calls."""
        return os.environ.get(MCP_ENV) or self._discover("mcp_token_file")

    def _discover(self, name: str = "token_file") -> str | None:
        """The token in the file the proxy names at `/health`: only a proxy on this machine,
        and only a file of the name a recordings folder keeps that token under. Found in
        review: a server anywhere else could name any file, and be sent what it holds."""
        if not _loopback(self.base_url):
            return None
        try:
            health = self._http.get(f"{self.base_url}/_tuningfork/health").json()
        except (httpx.HTTPError, ValueError):
            return None
        path = health.get(name) if isinstance(health, dict) else None
        expected = MCP_TOKEN_FILE if name == "mcp_token_file" else TOKEN_FILE
        if not isinstance(path, str) or os.path.basename(path) != expected:
            return None
        return read_token(path)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._send(method, path, **kwargs)
        if response.status_code in (401, 403) and self._given is None and _code(response) in (
            "token_missing", "token_wrong"
        ):
            # The proxy restarted since the token was read, or TUNINGFORK_TOKEN is left over
            # from an earlier one: the file always holds the current token.
            found = self._discover()
            if found and found != self._token:
                self._token = found
                response = self._send(method, path, **kwargs)
        return self._answer(response)

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        token = self.token
        if token:
            kwargs["headers"] = {**kwargs.get("headers", {}), TOKEN_HEADER: token}
        try:
            return self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise ControlError(
                f"no tuningfork proxy is answering at {self.base_url} — start one with "
                "`tuningfork serve`, or point TUNINGFORK_PROXY at it"
            ) from None
        except httpx.TimeoutException:
            raise ControlError(f"the proxy at {self.base_url} did not answer in time") from None
        except httpx.HTTPError as exc:
            raise ControlError(f"could not talk to {self.base_url}: {exc}") from None

    def _answer(self, response: httpx.Response) -> Any:
        ours = response.headers.get("x-tuningfork") == "control"
        if response.status_code in (404, 405) and not ours:
            raise ControlError(
                f"{self.base_url} has no live controls — point at a running `tuningfork serve` "
                "proxy (`tuningfork ui` only browses recordings)"
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code >= 400:
            error = body.get("error") if isinstance(body, dict) else None
            error = error if isinstance(error, dict) else None
            message = (error or {}).get("message") or f"HTTP {response.status_code}"
            hint = (error or {}).get("hint")
            raise ControlError(
                f"{message} — {hint}" if hint else message, error=error, status=response.status_code
            )
        return body

    # --- state --------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        status = self._call("GET", "/status")
        if not status.get("live"):
            raise ControlError(
                f"{self.base_url} is browsing recordings only (`tuningfork ui`); live "
                "controls need `tuningfork serve`"
            )
        return status

    def pause(self) -> dict[str, Any]:
        return self._call("POST", "/pause")

    def resume(self) -> dict[str, Any]:
        return self._call("POST", "/resume")

    # --- breakpoints --------------------------------------------------------------

    def breakpoints(self) -> list[dict[str, Any]]:
        return self._call("GET", "/breakpoints")

    def add_breakpoint(self, **definition: Any) -> dict[str, Any]:
        fields = {key: value for key, value in definition.items() if value is not None}
        return self._call("POST", "/breakpoints", json=fields)

    def set_enabled(self, breakpoint_id: str, enabled: bool) -> dict[str, Any]:
        return self._call(
            "PATCH", f"/breakpoints/{_id('breakpoint', breakpoint_id)}", json={"enabled": enabled}
        )

    def remove_breakpoint(self, breakpoint_id: str) -> dict[str, Any]:
        return self._call("DELETE", f"/breakpoints/{_id('breakpoint', breakpoint_id)}")

    # --- holds --------------------------------------------------------------------

    def holds(self, recent: bool = False) -> list[dict[str, Any]]:
        return self._call("GET", "/holds", params={"recent": "1"} if recent else None)

    def hold(self, hold_id: str) -> dict[str, Any]:
        return self._call("GET", f"/holds/{_id('hold', hold_id)}")

    def decide(
        self,
        hold_id: str,
        action: str,
        *,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        edits = {
            key: value
            for key, value in (("request", request), ("response", response))
            if value is not None
        }
        return self._call("POST", f"/holds/{_id('hold', hold_id)}/{action}", json=edits)

    def wait_for_hold(self, timeout: float = 300.0, poll: float = 0.5) -> dict[str, Any] | None:
        """Block until a call is held; return it with its payload, or None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pending = self.holds()
            if pending:
                return self.hold(pending[0]["id"])
            time.sleep(poll)
        return None

    # --- connections --------------------------------------------------------------

    def connections(self) -> dict[str, Any]:
        """The server the proxy forwards to, the harnesses it can launch, the proxy."""
        return self._call("GET", "/connections")

    def switch_upstream(self, url: str, *, dry_run: bool = False) -> dict[str, Any]:
        return self._call("POST", "/connections/upstream", json={"url": url, "dry_run": dry_run})

    def detect_harnesses(self) -> list[dict[str, Any]]:
        return self._call("POST", "/connections/harnesses/detect")

    def set_harness(self, name: str, executable: str | None) -> dict[str, Any]:
        """Run `executable` for this harness; None goes back to the one on PATH."""
        return self._call("PUT", f"/connections/harnesses/{name}", json={"executable": executable})

    # --- jobs ---------------------------------------------------------------------

    def kinds(self) -> list[dict[str, Any]]:
        """The kinds of work a job can do, each with its parameters' schema."""
        return self._call("GET", "/jobs/kinds")

    def plan(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """What a job would do, checked as starting it would be — nothing starts."""
        return self._call(
            "POST", "/jobs", json={"kind": kind, "params": params, "dry_run": True},
            timeout=httpx.Timeout(PREPARE_TIMEOUT, connect=CONNECT_TIMEOUT),
        )

    def submit(
        self,
        kind: str,
        params: dict[str, Any],
        *,
        request_id: str | None = None,
        started_by: dict[str, Any] | None = None,
        dedupe: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": kind, "params": params}
        if request_id is not None:
            body["request_id"] = request_id
        if started_by is not None:
            body["started_by"] = started_by
        if dedupe:
            body["dedupe"] = True
        return self._call(
            "POST", "/jobs", json=body, timeout=httpx.Timeout(PREPARE_TIMEOUT, connect=CONNECT_TIMEOUT)
        )

    def jobs(self) -> list[dict[str, Any]]:
        return self._call("GET", "/jobs")

    def job(self, job_id: str, *, after: int | None = None) -> dict[str, Any]:
        params = {"after": after} if after is not None else None
        return self._call("GET", f"/jobs/{_id('job', job_id)}", params=params)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/jobs/{_id('job', job_id)}/cancel")

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll: float = 1.0,
        on_event: Any = None,
    ) -> dict[str, Any]:
        """Block until a job has ended, handing each progress event to `on_event` once.
        Returns the job; raises `ControlError` on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        seen = 0
        while True:
            job = self.job(job_id, after=seen)
            for entry in job.get("events") or []:
                seen = max(seen, entry["seq"])
                if on_event is not None:
                    on_event(entry)
            if job["status"] not in ("queued", "running"):
                return job
            if deadline is not None and time.monotonic() >= deadline:
                raise ControlError(f"job {job['id'][:8]} is still {job['status']}")
            time.sleep(poll)
