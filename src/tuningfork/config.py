"""Runtime configuration, filesystem layout, and storage-time redaction."""

from __future__ import annotations

import bisect
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Payloads contain whatever the agent's tools read: shell output, files, environment.
# Redaction runs at storage time, never on forwarded traffic, so a bad pattern can spoil a
# recording but can never alter what the harness receives.
#
# Each pattern masks its ``secret`` group if it has one, otherwise the whole match.
#
# A mask costs the recording: a step whose context holds one can no longer be scored or
# probed (its prompt no longer renders to the recorded token count), a fork past an answer
# holding one is refused, and a resample sends `<redacted>` where the model saw text. So a
# pattern masks what has a key's shape, and a value under a secret's name only where it reads
# as a key — never code, never words: the generic pattern once took `tokenizer.encode` for a
# token, and masked one Hermes source file in six an agent read.
#
# The key prefixes must start a token: without that, `sk-` matched inside ordinary words —
# "task-management-system" was stored as "ta<redacted>".

#: The characters a value is written in, as each kind of pattern reads one: bare, as `.env`
#: files, YAML, headers and command lines write it; in quotes, as code and JSON do (anything
#: but spaces, quotes and the brackets placeholders and markup are written in); a bearer token;
#: Base64; a password in a URL; a key alone.
_BARE = r"A-Za-z0-9_\-.~+/=!@#$%^&*"
_QUOTED = r"^\s\"'\\<>|\[\]{}(),;"
_TOKEN68 = r"A-Za-z0-9_\-.~+/"
_BASE64 = r"A-Za-z0-9+/"
_URL_PASSWORD = r"^\s/@\"'<>\\"
_KEY_CHARS = r"A-Za-z0-9_\-"


def _not_a_key(chars: str) -> str:
    """A lookahead: the value ahead, written in `chars`, is no key though it stands where one
    could. Not a dotted name (`settings.SECRET_KEY`); not letters only, a word or words joined
    (`Initialization`, `resolved_api_key`, `_RUNTIME_KEY`, `resolvedApiKey`,
    `token-based-authentication`, `io.modelcontextprotocol/related-task`) — a key of letters
    only is read as a name and kept, as real keys hold digits; not a snake_case name with digits
    (`oauth2_scheme`); not numbers between dots (a version, an address, an OID); not a path to a
    file that holds one (`~/.ssh/id_ed25519`), nor a reference to where one is kept (`op://…`,
    `vault://…`); not a mask's stars, nor an example cut short (`sk-...`); not a variable or a
    placeholder (`$TOKEN`, `${{ secrets.X }}`, `<your-key>`, `%KEY%`)."""
    end = "[.!]?(?![" + chars + "])"
    return (
        r"(?!_*[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+" + end
        + r"|_*[A-Za-z]+(?:[_\-./:#]+[A-Za-z]+)*_*" + end
        + r"|_*[a-z][a-z0-9]*(?:_[a-z0-9]+)+_*" + end
        + r"|[0-9]+(?:\.[0-9]+)+" + end
        + r"|(?:~|\.{1,2})?/[A-Za-z0-9_.\-]"
        + r"|[A-Za-z][A-Za-z0-9+.\-]*://"
        + r"|\*{3}"
        + "|[" + chars + r"]*?(?:\.\.\.|…)"
        + r"|[$%{<])"
    )


def _either(word: str) -> str:
    """`word` in either case, letter by letter. Each pattern starts with a letter or a set of
    them, and looks behind only after it: the engine then skips to where one stands, where a
    pattern starting with a lookbehind, or under `(?i)`, is tried at every character."""
    return "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else re.escape(c) for c in word)


#: Starts a token: no character a key or a name is written in stands before it.
_STARTS = r"(?<![A-Za-z0-9_\-]{})"

#: Starts a word.
_WORD = r"(?<![A-Za-z0-9_]{})"

#: How the name of a secret ends: `OPENAI_API_KEY`, `db_password`, `aws_secret_access_key`,
#: `clientSecret`, `GITHUB_TOKEN`, `DB_PASS`, `dbPass`, and an upper-case `…_KEY`, `…_AUTH` or
#: `…_PAT` as a `.env` file writes one. Not a page's token (`next_page_token`, `nextPageToken`):
#: opaque, but no key, and an agent sends it back in its next call.
_SECRET_NAME = (
    r"[AaSsPpEeMmTt_](?:(?<=[Aa])(?i:pi[_\-]?key|ccess[_\-]?key|ccount[_\-]?key)"
    r"|(?<=[Ss])(?i:ecret[_\-]?key|igning[_\-]?key|ecret)"
    r"|(?<=[Pp])(?i:rivate[_\-]?key|assword|asswd|assphrase|ass(?<![a-z]pass))"
    r"|(?<=P)ass(?<=[a-z0-9]Pass)"
    r"|(?<=[Ee])(?i:ncryption[_\-]?key)|(?<=[Mm])(?i:aster[_\-]?key)"
    r"|(?<=[Tt])(?i:oken" + "".join(
        f"(?<!{page}{joint}token)"
        for page in ("page", "next", "cursor", "sync", "continuation", "pagination", "resume")
        for joint in ("", "_", "-")
    )
    + r")|(?<=_)(?:KEY(?<=[A-Z0-9]_KEY)|AUTH(?<=[A-Z0-9]_AUTH)|PAT(?<=[A-Z0-9]_PAT)"
    r"|CREDENTIALS(?<=[A-Z0-9]_CREDENTIALS)))"
)

#: A long flag named for a secret: `--api-key`, `--secret-key`, `--azure-openai-key`, `--db-pass`.
_SECRET_FLAG = r"--(?i:(?:[a-z0-9]+-)*(?:key|token|secret|password|passwd|pass))(?![A-Za-z0-9_\-])"

DEFAULT_REDACTIONS: tuple[str, ...] = (
    # Keys known by their shape, wherever they stand. `sk-` is no key where lower-case words
    # follow it (`sk-toggleable__content`, `sk-ecdsa-sha2-nistp256@openssh.com`); the generic
    # prefixes need a digit or a capital, which an identifier such as `fw_update_settings`
    # never has.
    r"(?P<secret>s" + _STARTS.format("s")
    + r"k(?:-(?![a-z0-9]{1,12}(?:[_\-]+[a-z0-9]{1,12})+(?![A-Za-z0-9_\-]))[A-Za-z0-9_\-]{16,}+(?!@)"
    r"|_(?:live|test)_[A-Za-z0-9]{20,})"
    r"|g" + _STARTS.format("g") + r"(?:h[pousr]_[A-Za-z0-9]{16,}|ithub_pat_[A-Za-z0-9_]{22,}"
    r"|(?:lpat-|sk_)(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,})"
    r"|h" + _STARTS.format("h") + r"f_(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|f" + _STARTS.format("f") + r"w_(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|n" + _STARTS.format("n") + r"pm_(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|t" + _STARTS.format("t") + r"gp_v1_(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|A" + _STARTS.format("A") + r"Iza[0-9A-Za-z_\-]{35}"
    r"|A(?<![A-Za-z0-9_\-+/]A)(?:KIA|SIA)[0-9A-Z]{16}(?![A-Za-z0-9+/])"
    r"|p" + _STARTS.format("p") + r"ypi-[A-Za-z0-9_\-]{40,}"
    r"|r" + _STARTS.format("r") + r"k_(?:live|test)_[A-Za-z0-9]{20,}"
    r"|w" + _STARTS.format("w") + r"hsec_[A-Za-z0-9+/=]{24,}"
    r"|x" + _STARTS.format("x") + r"(?:ox[abprs]-[A-Za-z0-9\-]{10,}|app-[0-9]-[A-Za-z0-9\-]{10,})"
    r")",
    # A JSON Web Token; a Discord bot's token; a Telegram bot's, after its bot's number.
    r"(?P<secret>eyJ" + _STARTS.format("eyJ") + r"[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})",
    r"(?P<secret>[MNO]" + _STARTS.format("[MNO]") + r"(?=[A-Za-z_\-]*[0-9])[A-Za-z0-9_\-]{23,27}"
    r"\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,40})(?![A-Za-z0-9_\-])",
    r"[0-9](?<![0-9][0-9])[0-9]{7,9}:(?P<secret>AA[A-Za-z0-9_\-]{33})(?![A-Za-z0-9_\-])",
    # A private key's body, from its first line to its last, or to where the text stops; a
    # body that starts as one does (Base64, or an encrypted key's header), not a mention. Its end
    # is looked for within a key's length: many beginnings without one would each read to the end.
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?=(?:\s|\\[nrt])*(?:Proc-Type:|[A-Za-z0-9+/]{16}))"
    r"(?P<secret>[\s\S]{1,10000}?(?=-----END [A-Z ]*PRIVATE KEY-----)|(?:\\[nrt]|[A-Za-z0-9+/=\s])+)",
    # A password in a URL, in HTTP Basic credentials, curl's `-u`, MySQL's `-p`, requests'
    # `auth=`, AWS's secret key, an Azure connection string or signature, and a Slack webhook's
    # path. None takes a backslash: at the JSON's own level it is half of an escape, and masked
    # alone it breaks the JSON.
    r"://[^\s/:@\"'<>\\]+:(?P<secret>" + _not_a_key(_URL_PASSWORD) + "[" + _URL_PASSWORD + "]+)@",
    r"[Bb]" + _WORD.format("[Bb]") + _either("asic") + r"[ \t]+(?P<secret>" + _not_a_key(_BASE64)
    + r"[A-Za-z0-9+/]{12,}+={0,2})(?![A-Za-z0-9+/=])",
    r"[Cc]" + _WORD.format("[Cc]") + _either("url")
    + r"(?![A-Za-z0-9_])[^\n]{0,300}?[ \t](?:-u|--user)[ \t=]+[\"']?[^\s:\"'\\]+:(?P<secret>[^\s\"'\\]+)",
    r"[Mm]" + _WORD.format("[Mm]") + r"(?:" + _either("ysql") + r"(?:" + _either("dump") + "|" + _either("admin")
    + r")?|" + _either("ariadb") + r")(?![A-Za-z0-9_])[^\n]{0,300}?[ \t]-p(?P<secret>[^\s\"'\\]{4,})",
    r"[AaSs](?:(?<=[Aa])(?i:ws_secret_access_key)|(?<=[Ss])(?i:ecretaccesskey))[\"']?[ \t]*[:=\t ][ \t]*[\"']?"
    r"(?P<secret>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
    r"[Aa]" + _WORD.format("[Aa]") + _either("uth") + r"[ \t]*=[ \t]*\([ \t]*(?P<u>[\"'])[^\"'\\]*(?P=u)[ \t]*,"
    r"[ \t]*(?P<q>[\"'])(?P<secret>[^\"'\\]+)(?P=q)",
    r"[?&;Ss](?:(?<=[?&;])[Ss][Ii][Gg]|(?<=[Ss])(?i:hared[_\-]?access[_\-]?signature))"
    r"=(?P<secret>[A-Za-z0-9%+/=]{20,})",
    r"hooks\.slack\.com/services/(?P<secret>T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+)",
    # A bearer token, and a key given to a command's flag. Neither crosses a line.
    r"[Bb]" + _WORD.format("[Bb]") + _either("earer") + r"[ \t]+(?P<secret>" + _not_a_key(_TOKEN68)
    + "[" + _TOKEN68 + "]{16,}+=*)",
    _SECRET_FLAG + r"(?:[ \t]+|=)[\"']?(?P<secret>" + _not_a_key(_BARE) + "[" + _BARE + "]{8,}+)",
    # A value under a secret's name, in quotes or bare. A quoted name takes a quoted value only
    # — `"page_token": 1729712345678` is a number, and masked bare it would no longer be JSON —
    # and neither crosses a line: `if not api_key:` is no assignment of the next line. A bare
    # value before `,` or `)` is a type, an argument or a call (`private_key: PKCS7Types,`).
    _SECRET_NAME + r"(?P<kq>[\"'])?[ \t]*(?:=>|:=|[:=])[ \t]*"
    r"(?:(?P<q>[\"'])(?P<secret>" + _not_a_key(_QUOTED) + "[" + _QUOTED + "]{8,}+)(?P=q)"
    r"|(?(kq)(?!)|(?P<secret_bare>" + _not_a_key(_BARE) + "[" + _BARE + r"]{8,}+)(?![(\[,)\]}])))",
    # A value under any other name ending in `key` (`azure_openai_key`, `mistralKey`), where it
    # reads as a key alone: digits and letters, twenty or more, ending where a key ends. Not a
    # UUID, a path, a cache's or a table's key, nor a public one.
    r"[_\-K](?:(?<=[_\-])[Kk][Ee][Yy](?<=[A-Za-z0-9][_\-][Kk][Ee][Yy])|(?<=K)ey(?<=[a-z0-9]Key))"
    r"(?<![Pp][Uu][Bb][Ll][Ii][Cc][_\-][Kk][Ee][Yy])(?<!publicKey)"
    r"(?P<kq>[\"'])?[ \t]*[:=][ \t]*(?(kq)[\"']|[\"']?)"
    r"(?P<secret>(?=[A-Za-z0-9_\-]*[0-9])(?=[A-Za-z0-9_\-]*[A-Za-z])" + _not_a_key(_KEY_CHARS)
    + r"(?![0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![A-Za-z0-9_\-]))"
    "[" + _KEY_CHARS + r"]{20,}+)(?![A-Za-z0-9_\-./:+@,(\[)\]}])",
)

REDACTED = "<redacted>"

#: How deep redaction reads JSON written inside JSON strings: a stored payload's own strings,
#: a call's arguments or a tool's JSON result inside them, and what that result quotes.
JSON_DEPTH = 3

#: A JSON escape: how JSON text writes a character of a string.
_ESCAPE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])')
_SHORT_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

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
    """Mask secrets in `text`, as it stands and as its JSON escapes read (`_read_spans`). The
    replacement is quote-free, and replaces whole escapes, so JSON stays parseable."""
    spans = _merged(_read_spans(text, patterns))
    if not spans:
        return text
    pieces, at = [], 0
    for start, end in spans:
        pieces += [text[at:start], REDACTED]
        at = end
    return "".join(pieces) + text[at:]


def _masked_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    # A pattern with a `secret` group masks that group only — the `api_key=` before it stays,
    # so a redacted recording stays readable — and one without masks its whole match. One with
    # several ways to a secret names each group `secret…`, and masks the one that matched.
    spans = []
    for pattern in patterns:
        compiled = re.compile(pattern)
        groups = [name for name in compiled.groupindex if name.startswith("secret")]
        for match in compiled.finditer(text):
            spans.append(next((match.span(name) for name in groups if match.group(name) is not None), match.span()))
    return spans


def _merged(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _json_unescaped(text: str) -> tuple[str, Callable[[int], int]] | None:
    """`text` with its JSON escapes read once, and a map from an index of that back to `text`'s:
    a character read from an escape stands for the whole escape. None where there is none."""
    pieces: list[str] = []
    read_at: list[int] = []  # where each escape's character sits in what is read
    shifts: list[int] = []  # how far `text` runs ahead of what is read, after each escape
    at = shift = 0
    for match in _ESCAPE.finditer(text):
        escape = match.group()
        char = chr(int(escape[2:], 16)) if escape[1] == "u" else _SHORT_ESCAPES[escape[1]]
        pieces += [text[at : match.start()], char]
        read_at.append(match.start() - shift)
        shift += len(escape) - 1
        shifts.append(shift)
        at = match.end()
    if not read_at:
        return None

    def back(index: int) -> int:
        before = bisect.bisect_left(read_at, index)
        return index + (shifts[before - 1] if before else 0)

    return "".join(pieces) + text[at:], back


def _read_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    """The spans of `text` the patterns mask, read as it stands and as its JSON escapes read,
    `JSON_DEPTH` deep. Stored payloads are JSON, where a newline is a backslash and an `n`: a key
    on a line of its own follows a letter there, which an `sk-` key must not, and `api_key: …`
    inside a call's arguments or a tool's JSON result has its quotes escaped."""
    spans = _masked_spans(text, patterns)
    view, back = text, (lambda index: index)
    for _ in range(JSON_DEPTH):
        read = _json_unescaped(view)
        if read is None:
            break
        view, inner = read
        back = (lambda outer, inner: lambda index: outer(inner(index)))(back, inner)
        spans += [(back(start), back(end)) for start, end in _masked_spans(view, patterns)]
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
    is not possible. Patterns run over the text as storage reads it: as it stands and as its
    JSON escapes read (a model that writes JSON writes them), and as JSON escapes it, as it is
    stored — so a secret any of these would mask is masked here.
    """
    spans = _read_spans(text, patterns)
    escaped, origin = _json_escaped(text)
    for start, end in _masked_spans(escaped, patterns):
        if end > start:
            spans.append((origin[start], origin[end - 1] + 1))
    return _merged(spans)


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

    Replacements are quote-free and fall inside strings, so the result still parses. Applied to
    payloads on their way into storage, which means a recording containing a secret will not
    reconstruct byte-identically — that is the intended trade. Two keys of one object masked
    alike stay two: the second is `<redacted> (2)`, and so on.
    """
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return obj
    redacted = redact(text, patterns)
    if redacted == text:
        return obj
    try:
        return json.loads(redacted, object_pairs_hook=_apart)
    except json.JSONDecodeError:
        # A pattern broke the structure. Losing the recording beats storing the secret.
        return {"tuningfork_error": "redaction produced invalid JSON; payload dropped"}


def _apart(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """An object's pairs, a key masked as another already is numbered: none is lost."""
    kept: dict[str, Any] = {}
    for key, value in pairs:
        if key in kept and REDACTED in key:
            number = 2
            while f"{key} ({number})" in kept:
                number += 1
            key = f"{key} ({number})"
        kept[key] = value
    return kept
