"""Runtime configuration, filesystem layout, and storage-time redaction."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Payloads contain whatever the agent's tools read: shell output, files, environment.
# Redaction runs at storage time, never on forwarded traffic, so a bad pattern can spoil a
# recording but can never alter what the harness receives.
#
# Each pattern masks its ``secret`` group if it has one, otherwise the whole match.
#
# The key prefixes must start a token: without that, `sk-` matched inside ordinary words —
# "task-management-system" was stored as "ta<redacted>".
DEFAULT_REDACTIONS: tuple[str, ...] = (
    r"(?<![A-Za-z0-9_\-])(?P<secret>sk-[A-Za-z0-9_\-]{16,})",
    r"(?i)\bbearer\s+(?P<secret>[A-Za-z0-9._\-]{16,})",
    r"(?i)(?:api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"']?(?P<secret>[A-Za-z0-9._\-]{12,})",
    r"(?P<secret>gh[pousr]_[A-Za-z0-9]{16,})",
    r"(?P<secret>AKIA[0-9A-Z]{16})",
    r"(?P<secret>xox[baprs]-[A-Za-z0-9-]{10,})",
)

REDACTED = "<redacted>"

#: Marks a request as experiment traffic. A tuningfork proxy on the path forwards it through
#: its gate but does not record it as a run step: a resampled request carries the same
#: message path as the step it came from, and would otherwise look like the harness
#: repeating that call.
EXPERIMENT_HEADER = "x-tuningfork-experiment"


def server_root(base_url: str) -> str:
    """A server's root, from a base URL with or without the `/v1` harnesses are given."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def redact(text: str, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> str:
    """Mask secrets in `text`. The replacement is quote-free, so JSON stays parseable."""
    for pattern in patterns:
        compiled = re.compile(pattern)
        has_secret_group = "secret" in compiled.groupindex

        def _sub(match: re.Match[str], _has_group: bool = has_secret_group) -> str:
            if not _has_group or match.group("secret") is None:
                return REDACTED
            # Keep the surrounding match (the `api_key=` prefix, say) and mask only the
            # secret itself, so redacted recordings stay readable.
            whole = match.group(0)
            start, end = match.span("secret")
            offset = match.start()
            return whole[: start - offset] + REDACTED + whole[end - offset :]

        text = compiled.sub(_sub, text)
    return text


def _masked_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    spans = []
    for pattern in patterns:
        compiled = re.compile(pattern)
        for match in compiled.finditer(text):
            if "secret" in compiled.groupindex and match.group("secret") is not None:
                spans.append(match.span("secret"))
            else:
                spans.append(match.span())
    return spans


def _json_escaped(text: str) -> tuple[str, list[int]]:
    """`text` as a JSON string holds it (between the quotes), and for each character of that
    the index of the character of `text` it came from."""
    short = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    out: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        piece = short.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 else char)
        out.append(piece)
        origin.extend([index] * len(piece))
    return "".join(out), origin


def redact_spans(text: str, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> list[tuple[int, int]]:
    """The spans of `text` a redaction masks, merged and in order.

    For text that is stored in pieces — an answer as its tokens — where masking the joined text
    is not possible. Patterns run over the text as it reads and as JSON escapes it: storage
    redacts serialized JSON, where a newline is `\\n` and a pattern's `\\s*` no longer matches
    it, so a secret either form would mask is masked here.
    """
    spans = _masked_spans(text, patterns)
    escaped, origin = _json_escaped(text)
    for start, end in _masked_spans(escaped, patterns):
        if end > start:
            spans.append((origin[start], origin[end - 1] + 1))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class Settings:
    """Where tuningfork keeps state and how the proxy behaves."""

    upstream_base_url: str
    home: Path
    host: str = "127.0.0.1"
    port: int = 8787
    # Cap on concurrent upstream requests. Defaults to 1: on a shared inference server, a
    # fan-out of resamples must never starve whatever else is using the model.
    max_inflight: int = 1
    request_timeout: float = 600.0
    #: While a streaming call is held at a breakpoint, an SSE comment is sent this often so
    #: that a client or intermediary with an idle timeout keeps the connection open.
    keepalive_interval: float = 10.0
    redaction_patterns: tuple[str, ...] = DEFAULT_REDACTIONS
    #: Background jobs run at once (`tuningfork.jobs`). On one GPU, more is slower.
    jobs: int = 1
    #: The access token to require for state changes (`tuningfork.access`); a new one is
    #: made when none is given. Never shown in a repr.
    token: str | None = field(default=None, repr=False)
    #: The MCP endpoint's own token (`access.MCP_ENV`), which opens nothing else.
    mcp_token: str | None = field(default=None, repr=False)
    #: The tools the proxy's MCP route offers (`tuningfork.mcp`): evidence, all, or none —
    #: no route at all.
    mcp_tools: str = "evidence"
    #: What a tool result carries: "both" its text and its structured content, or the "text"
    #: alone, for a client that shows the model both (Hermes).
    mcp_results: str = "both"
    #: How long one MCP call may wait for a job, at most, in seconds: under what the client
    #: gives a tool call. OpenJarvis reads an HTTP answer for 60 s, Hermes gives a call 300 s
    #: by default — for it, about 110 keeps a twenty-minute screen to a dozen polls.
    mcp_max_wait: int = 50
    #: The most output tokens any call a harness makes through this proxy may ask for: one that
    #: asks for more, or names no limit, is sent with this one (`wire.base.cap_output`), and what
    #: it asked is recorded with the step. None leaves every call as its harness sent it.
    max_output: int | None = None

    @property
    def blob_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def step_log(self) -> Path:
        """Phase 0 step log. Replaced by the SQLite message tree in Phase 1."""
        return self.home / "steps.jsonl"


def default_home() -> Path:
    return Path(os.environ.get("TUNINGFORK_HOME") or Path.home() / ".tuningfork")


#: Settings kept with a recordings folder: harness executables, extra protected endpoints,
#: recent upstreams. Never a harness's own configuration.
HOME_CONFIG = "config.json"


def _read_object(path: Path) -> dict[str, Any]:
    """A JSON object from `path`, or an empty one. A file that cannot be read is treated as
    empty rather than stopping every command."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_home_config(home: Path) -> dict[str, Any]:
    """The recordings folder's `config.json`, or an empty one."""
    return _read_object(Path(home) / HOME_CONFIG)


def machine_config_path() -> Path:
    """The machine's own settings, whichever recordings folder is in use:
    `$XDG_CONFIG_HOME/tuningfork/config.json`, by default under `~/.config`. Read, strictly,
    for the endpoints to protect on this machine (`endpoints`)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "tuningfork" / HOME_CONFIG


def write_home_config(home: Path, data: dict[str, Any]) -> None:
    """Replace the recordings folder's `config.json` atomically."""
    path = Path(home) / HOME_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{HOME_CONFIG}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def redact_json(obj: Any, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> Any:
    """Redact a JSON-serializable object by round-tripping it through `redact`.

    Replacements are quote-free, so the result still parses. Applied to payloads on their
    way into storage, which means a recording containing a secret will not reconstruct
    byte-identically — that is the intended trade.
    """
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return obj
    redacted = redact(text, patterns)
    if redacted == text:
        return obj
    try:
        return json.loads(redacted)
    except json.JSONDecodeError:
        # A pattern broke the structure. Losing the recording beats storing the secret.
        return {"tuningfork_error": "redaction produced invalid JSON; payload dropped"}
