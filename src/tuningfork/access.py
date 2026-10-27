"""The access token: what a request needs to change tuningfork's state.

Reading stays open: to the person at the machine, and to anything else that can reach the
port — another user's process, or, under WSL's mirrored networking, a Windows program. The
recordings folder itself is this user's alone when tuningfork makes it (0700), and `serve` says
when one made before is not (`open_to_others`). Anything that changes state — pausing,
a breakpoint, a decision at a held call, and from Phase 5 starting work that launches
harnesses — needs the token a running proxy was started with. A web page open in the same
browser can send requests to a loopback port, but it cannot read the token: it is in the
address `tuningfork serve` printed, in this browser tab's session, and in a file only this
user can read.

An MCP client that connects to the proxy (`/_tuningfork/mcp`) holds a token of its own, the
MCP token, which opens that endpoint and nothing else: an agent offered tools there cannot
use its client's token to call what the tools leave out, such as approving an edit.

The token is not a sandbox for the harness. A launched harness runs shell commands as this
user, and can read anything this user can; what the token keeps out is everything that is
not this user at this machine.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Where a request carries the token.
HEADER = "x-tuningfork-token"
#: Pins the token a proxy starts with, and gives the CLI one without the file.
ENV = "TUNINGFORK_TOKEN"
#: The proxy's current token, in its recordings folder, readable by this user only.
TOKEN_FILE = ".access-token"
#: The MCP endpoint's own token, beside it: pinned by `MCP_ENV`, else new each start.
MCP_ENV = "TUNINGFORK_MCP_TOKEN"
MCP_TOKEN_FILE = ".mcp-token"
#: The built-in proposer's API key, for an endpoint that needs one (`ops.proposer`); else
#: the file `PROPOSER_KEY_FILE` in the recordings folder.
PROPOSER_KEY_ENV = "TUNINGFORK_PROPOSER_KEY"
PROPOSER_KEY_FILE = ".proposer-key"
#: Never handed to anything tuningfork runs for a task: a check runs as this user, in a
#: workspace a model wrote.
SECRET_ENVS = (ENV, MCP_ENV, PROPOSER_KEY_ENV)


def new_token() -> str:
    return secrets.token_urlsafe(24)


def held_secrets(home: Path | str | None = None) -> list[str]:
    """The keys tuningfork holds here — in this process's environment (`SECRET_ENVS`), and in the
    recordings folder's files — masked wherever one turns up: in a check's output, say, which
    may print what the model's code read, though nothing tuningfork runs is given one."""
    values = [os.environ.get(name, "") for name in SECRET_ENVS]
    if home is not None:
        for path in (token_path(home), mcp_token_path(home), Path(home) / PROPOSER_KEY_FILE):
            try:
                values.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def token_path(home: Path | str) -> Path:
    return Path(home) / TOKEN_FILE


def mcp_token_path(home: Path | str) -> Path:
    return Path(home) / MCP_TOKEN_FILE


def write_token(home: Path | str, token: str) -> Path:
    """Write the token where the CLI looks for it."""
    return write_secret(token_path(home), token)


def open_to_others(folder: Path | str) -> str | None:
    """Who besides this user can open a recordings folder, in words, with how to make it this
    user's alone — or None where no one else can. A folder tuningfork makes is 0700; one made
    before, or by a person, keeps its mode, which tuningfork never changes by itself."""
    import shlex

    from tuningfork.paths import drive_root, under_wsl

    if os.name != "posix":
        return None
    try:
        mode = Path(folder).stat().st_mode & 0o777
    except OSError:
        return None
    if not mode & 0o077:
        return None
    who = ("its group and other users" if mode & 0o070 and mode & 0o007
           else "its group" if mode & 0o070 else "other users")
    if under_wsl() and re.match(re.escape(drive_root()) + r"[A-Za-z](/|$)", str(Path(folder).resolve())):
        # A Windows drive, which keeps no file modes unless mounted with `metadata`.
        return (f"{who} on this machine can open it (mode {mode:o}): it is on a Windows drive, where "
                "chmod may not hold; keep recordings on the Linux file system")
    return (f"{who} on this machine can open it (mode {mode:o}): `chmod 700 {shlex.quote(str(folder))}` "
            "makes it yours alone")


def write_secret(path: Path | str, text: str) -> Path:
    """Write a secret atomically, mode 0600 from the moment the file exists — a secret
    briefly readable by others would be a secret given away."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{text}\n")
        os.chmod(temporary, 0o600)  # whatever the umask made of the mode above
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_token(path: Path | str) -> str | None:
    """The token in a token file, or None when there is none to read."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def matches(given: str | None, expected: str | None) -> bool:
    """Compared in constant time: how long a refusal takes says nothing about the token."""
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Access:
    """How a server decides who may change its state."""

    token: str
    #: Where the token was written, for the CLI to find; reported by `/health`.
    file: Path | None = None
    #: The MCP endpoint's token: it opens `/_tuningfork/mcp`, and nothing else.
    mcp_token: str | None = None
    mcp_file: Path | None = None

    @classmethod
    def start(cls, home: Path | str, token: str | None = None, mcp_token: str | None = None) -> Access:
        """A proxy's access: the tokens it was given, or new ones, written into its home."""
        token = token or new_token()
        mcp_token = mcp_token or new_token()
        if matches(mcp_token, token):
            raise ValueError(f"{MCP_ENV} must differ from {ENV}: the MCP token opens the MCP endpoint alone")
        return cls(token=token, file=write_token(home, token), mcp_token=mcp_token,
                   mcp_file=write_secret(mcp_token_path(home), mcp_token))

    def allows(self, given: str | None) -> bool:
        """Whether `given` is the access token, which every change but MCP needs."""
        return matches(given, self.token)

    def allows_mcp(self, given: str | None) -> bool:
        """Whether `given` is the MCP token, which the MCP endpoint alone takes."""
        return matches(given, self.mcp_token)

    def status(self, headers: Mapping[str, Any]) -> dict[str, Any]:
        """What `/status` says about the asking page's token: whether one is needed, and
        whether the one it sent (if any) is it."""
        return {"required": True, "valid": self.allows(headers.get(HEADER))}
