"""Endpoints tuningfork never sends anything to.

A model server something else depends on — a production agent's, say — must never receive
experiment load: not a resample, not a probe, not a harness's calls through the proxy. Such
endpoints are listed here, and every place tuningfork chooses where to send traffic checks
the list, with a request hook on its own HTTP clients as a backstop.

Nothing is protected until a person says so, in any of three places, each adding to the
others: `TUNINGFORK_PROTECTED` (a comma-separated list of `host:port`, where either side may
be `*`), the machine's config (`config.machine_config_path()`, `"protected"`), which nothing
else can take back, and a recordings folder's `config.json`, under `"protected"` too. A
list that cannot be read stops all traffic until it is fixed.
"""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tuningfork.ops.common import OperationError

ENV = "TUNINGFORK_PROTECTED"

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


class ProtectedEndpoint(OperationError):
    """Traffic that would reach a protected endpoint."""


def is_loopback(host: str) -> bool:
    """Whether `host` means this machine: any spelling of localhost, 127/8, ::1, an
    IPv4-mapped 127 address, or the unspecified address (which connects locally)."""
    host = host.strip("[]").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


@dataclass(frozen=True, slots=True)
class Endpoint:
    #: None is any host. A loopback host stands for every loopback spelling.
    host: str | None
    #: None is any port.
    port: int | None

    def matches(self, host: str, port: int) -> bool:
        if self.port is not None and self.port != port:
            return False
        if self.host is None:
            return True
        if is_loopback(self.host):
            return is_loopback(host)
        return self.host.lower() == host.strip("[]").lower()

    def __str__(self) -> str:
        host = self.host if self.host is not None else "*"
        if host and ":" in host:
            host = f"[{host}]"
        return f"{host}:{self.port if self.port is not None else '*'}"


def parse_endpoint(spec: str) -> Endpoint:
    """`host:port`, `*:port`, `host:*`, or `[v6]:port`."""
    spec = spec.strip()
    if spec.startswith("["):
        host, _, rest = spec[1:].partition("]")
        port = rest.lstrip(":") or "*"
    else:
        host, sep, port = spec.rpartition(":")
        if not sep:
            raise ValueError(f"{spec!r} is not host:port")
    if not host:
        raise ValueError(f"{spec!r} names no host")
    try:
        number = None if port == "*" else int(port)
    except ValueError:
        raise ValueError(f"{spec!r} has no port number") from None
    return Endpoint(None if host == "*" else host, number)


def _machine_specs() -> list[str]:
    """The machine's config's list, read strictly: a file there that cannot be read stops
    all traffic until it is fixed, rather than protecting nothing without a word."""
    from tuningfork.config import machine_config_path

    path = machine_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as error:
        problem: str = str(error)
    else:
        specs = data.get("protected", []) if isinstance(data, dict) else None
        if isinstance(specs, list) and all(isinstance(spec, str) for spec in specs):
            return specs
        problem = 'it holds no "protected" list of host:port strings'
    raise OperationError(
        f"the machine's config, {path}, cannot be read: {problem}",
        hint="tuningfork sends nothing anywhere until it can: fix the file, since removing it would drop "
             "the endpoints it protects too",
        code="machine_config",
    )


def _sources(home: Path | None) -> list[tuple[Endpoint, str]]:
    """Each protected endpoint, with where it was written: TUNINGFORK_PROTECTED, the
    machine's config, the recordings folder's."""
    from tuningfork.config import HOME_CONFIG, machine_config_path, read_home_config

    written = [(part, ENV) for part in os.environ.get(ENV, "").split(",") if part.strip()]
    written += [(spec, str(machine_config_path())) for spec in _machine_specs()]
    if home is not None:
        extra = read_home_config(home).get("protected") or []
        written += [(spec, str(Path(home) / HOME_CONFIG)) for spec in extra if isinstance(spec, str)]
    found = []
    for spec, where in written:
        try:
            found.append((parse_endpoint(spec), where))
        except ValueError as error:
            raise OperationError(
                f"a protected endpoint in {where} cannot be read: {error}",
                hint="write it as host:port, where either side may be *", code="protected_list",
            ) from None
    return found


def protected(home: Path | None = None) -> tuple[Endpoint, ...]:
    """The protected list: TUNINGFORK_PROTECTED's, the machine's config's and the recordings
    folder's, together."""
    return tuple(endpoint for endpoint, _ in _sources(home))


def address(url: str) -> tuple[str, int]:
    parts = urlsplit(url if "://" in url else f"http://{url}")
    host = parts.hostname or ""
    port = parts.port or _DEFAULT_PORTS.get(parts.scheme, 80)
    return host, port


def on_internet(url: str, *, resolve: Any = None) -> bool | None:
    """Whether the server at `url` is out on the internet: every address its host resolves to
    is a public one — not this machine, a private network, or a VPN's shared range (a GPU of
    one's own reached over Tailscale is not). None when the host does not resolve."""
    import socket

    host, port = address(url)
    try:
        found = {info[4][0] for info in (resolve or socket.getaddrinfo)(host, port, type=socket.SOCK_STREAM)}
    except (OSError, UnicodeError, ValueError):
        return None
    addresses = []
    for text in found:
        try:
            ip = ipaddress.ip_address(str(text).split("%", 1)[0])
        except ValueError:
            return None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        addresses.append(ip)
    return all(ip.is_global for ip in addresses) if addresses else None


async def find_on_internet(url: str, *, timeout: float = 5.0) -> bool | None:
    """`on_internet` off the event loop, given up on after `timeout` seconds."""
    import asyncio

    try:
        return await asyncio.wait_for(asyncio.to_thread(on_internet, url), timeout)
    except asyncio.TimeoutError:
        return None


def check_endpoint(url: str, *, home: Path | None = None, purpose: str = "tuningfork") -> str:
    """`url`, unless it would reach a protected endpoint — then a refusal saying how the
    list is changed."""
    host, port = address(url)
    for endpoint, where in _sources(home):
        if endpoint.matches(host, port):
            raise ProtectedEndpoint(
                f"{url} is a protected endpoint: {purpose} sends nothing there",
                hint=(
                    f"{endpoint} is protected by {where}. If this server is meant for "
                    f"experiments, take it off the list there."
                ),
                code="protected_endpoint",
            )
    return url


def guard_hooks(*, home: Path | None = None, sync: bool = False) -> dict[str, list[Any]]:
    """httpx `event_hooks` that refuse a request to a protected endpoint before it is sent:
    the backstop behind every check made where a URL is chosen."""

    def refuse(request: Any) -> None:
        check_endpoint(str(request.url), home=home)

    async def refuse_async(request: Any) -> None:
        refuse(request)

    return {"request": [refuse if sync else refuse_async]}
