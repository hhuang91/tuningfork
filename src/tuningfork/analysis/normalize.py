"""Turning a model response into something comparable.

Clustering outcomes is only meaningful once incidental differences are removed. Two samples
that did the same thing must produce the same key, or every sample forms its own cluster and
the statistics say nothing.

Phase 0 measured the concrete hazard: tool-call ids are a fresh number per call, so on raw
payloads two identical behaviours differ from the first tool call onward. The same goes for
a file's formatting: `{"count": 4}` and `{\n  "count": 4\n}` are one behaviour.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tuningfork.proxy.sse import CUT_OFF, UPSTREAM_ERROR
from tuningfork.sandbox import HOME as SANDBOX_HOME
from tuningfork.sandbox import WORKSPACE as SANDBOX_WORKSPACE
from tuningfork.store.tree import mask_volatile

WHITESPACE = re.compile(r"\s+")

#: The batch runner gives every run its own workspace and home,
#: `<workdir>/<batch>/<index>/workspace` and `.../home`. A harness that writes absolute paths
#: (Claude Code does) puts that per-run prefix into file arguments, shell commands and
#: file contents, so it is masked wherever it appears: two runs that read the same file did
#: the same thing. The name must end there — followed by `/`, punctuation, or nothing.
RUN_PATH = re.compile(r"(?:/[^/\s\"'`()]+)*?/[0-9a-f]{8}/\d+/(workspace|home)(?![\w-])")

#: A launch in the sandbox (`tuningfork.sandbox`) sees the same home and folder every time;
#: they are masked as a launch's own folders are, so runs laid out either way compare.
SANDBOX_PATHS = (
    (re.compile(re.escape(str(SANDBOX_WORKSPACE)) + r"(?![\w-])"), "<workspace>"),
    (re.compile(re.escape(str(SANDBOX_HOME)) + r"(?![\w-])"), "<home>"),
)

#: Bumped whenever a change can give an outcome a different key. A key stored earlier — an
#: A/B's declared target — then names a cluster that no longer exists, so a stored target
#: is only re-tested under the version it was declared under.
#: 2: per-run paths masked in every argument, the home masked too, workspace paths relative.
#: 3: a sandboxed launch's home and folder masked as a launch's own are.
NORMALIZER_VERSION = 3

#: Argument keys that name a file, used to decide how to treat a sibling payload.
PATH_KEYS = ("path", "file_path", "filename", "file")

#: Every normalizer a task spec may name. `tool_call_json` is the baseline and always on.
#: `tool_name` is a deliberate coarsening, so it is only ever on when asked for by name.
KNOWN_NORMALIZERS = ("tool_call_json", "python_ast", "json_content", "tool_name")
#: What naming no normalizers means.
DEFAULT_NORMALIZERS = ("tool_call_json", "python_ast", "json_content")


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a step did, in comparable form."""

    kind: str  # tool_call | text | empty | truncated | error
    key: str  # equal keys mean equivalent behaviour
    label: str  # human-readable, for reports
    #: What the behaviour says, whole: a reply's words, or each call's tool and its arguments'
    #: values as the model wrote them — a script's code, not its digest — with per-run paths
    #: masked. What a target reads (`analysis.distribution`); never part of the key, so two
    #: scripts differing in a comment are still one behaviour.
    text: str = field(default="", compare=False)

    @property
    def short_key(self) -> str:
        return hashlib.sha256(self.key.encode()).hexdigest()[:8]

    @property
    def is_error(self) -> bool:
        """Transport failures are not behaviour, and are kept out of every rate."""
        return self.kind == "error"

    def to_json(self) -> dict:
        return {"kind": self.kind, "label": self.label, "key": self.short_key}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]


def python_ast_digest(source: str) -> str | None:
    """Structural digest of Python source, or None if it does not parse.

    Comments and formatting vanish at parse time; docstrings are removed explicitly. Two
    scripts that differ only in wording or layout therefore land in the same cluster, which
    is the point — the question is whether the model wrote *different code*, not whether it
    phrased a comment differently.
    """
    try:
        # The source is the model's: an invalid escape in one of its strings is its own
        # business, not a warning on whoever reads the recording.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    _strip_docstrings(tree)
    return _digest(ast.dump(tree, annotate_fields=False, include_attributes=False))


def json_canonical(text: str) -> str | None:
    """Canonical form of a JSON document, or None if it does not parse.

    Keys are sorted and whitespace removed; numbers go through a parse, so `61.50` and
    `61.5` agree. The result stays readable, which matters here: in a report, the canonical
    JSON *is* the most useful description of what the model wrote.
    """
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mask_workspace(text: str) -> str:
    """Replace a run's own workspace and home paths with `<workspace>` / `<home>`: a
    launch's folders, or the sandbox's."""
    text = RUN_PATH.sub(lambda match: f"<{match.group(1)}>", text)
    for pattern, placeholder in SANDBOX_PATHS:
        text = pattern.sub(placeholder, text)
    return text


def workspace_relative(path: str) -> str:
    """A path argument as the file it names inside the run's workspace.

    A batch run's harness starts in its workspace, so `data.csv`, `./data.csv` and the
    absolute path to it name one file. Claude Code alternates between the two forms from
    run to run; left alone, that splits every step that touches a file.
    """
    masked = mask_workspace(path)
    if masked == "<workspace>":
        return "."
    if masked.startswith("<workspace>/"):
        masked = masked[len("<workspace>/") :] or "."
    while masked.startswith("./"):
        masked = masked[2:] or "."
    return masked


def normalize_text(text: str) -> str:
    """Collapse whitespace. Deliberately conservative: wording differences are the subject."""
    return WHITESPACE.sub(" ", mask_workspace(text)).strip()


def _sibling_path(arguments: dict[str, Any]) -> str | None:
    for key in PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return None


def _is_python_payload(arguments: dict[str, Any], key: str) -> bool:
    if key in {"code", "source"}:
        return True
    path = _sibling_path(arguments)
    return key in {"content", "text"} and bool(path) and path.endswith(".py")


def _is_json_payload(arguments: dict[str, Any], key: str) -> bool:
    path = _sibling_path(arguments)
    return key in {"content", "text", "data", "body"} and bool(path) and path.endswith(".json")


def canonical_arguments(
    arguments: Any, *, python_ast: bool = True, json_content: bool = True
) -> Any:
    """Canonicalize tool-call arguments, summarizing file payloads by their structure."""
    if not isinstance(arguments, dict):
        return _masked(arguments)
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and key in PATH_KEYS:
            value = workspace_relative(value)
        if not isinstance(value, str):
            out[key] = _masked(value)
        elif python_ast and _is_python_payload(arguments, key):
            source = mask_workspace(value)
            digest = python_ast_digest(source)
            out[key] = f"<ast:{digest}>" if digest else f"<text:{_digest(source)}>"
        elif json_content and _is_json_payload(arguments, key):
            canonical = json_canonical(value)
            out[key] = mask_workspace(canonical if canonical is not None else value)
        else:
            out[key] = mask_workspace(value)
    return out


def _masked(value: Any) -> Any:
    """Per-run paths masked in every string inside an argument, however deeply nested."""
    if isinstance(value, str):
        return mask_workspace(value)
    if isinstance(value, list):
        return [_masked(item) for item in value]
    if isinstance(value, dict):
        return {key: _masked(item) for key, item in value.items()}
    return value


def canonical_tool_call(
    call: dict[str, Any], *, python_ast: bool = True, json_content: bool = True
) -> dict[str, Any]:
    function = call.get("function") or {}
    raw = function.get("arguments")
    if isinstance(raw, str):
        try:
            arguments: Any = json.loads(raw)
        except json.JSONDecodeError:
            arguments = raw
    else:
        arguments = raw
    return {
        "name": function.get("name"),
        "arguments": canonical_arguments(
            arguments, python_ast=python_ast, json_content=json_content
        ),
    }


def describe_tool_calls(calls: Sequence[dict[str, Any]], limit: int = 90) -> str:
    parts = []
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(arguments.items()))
        else:
            rendered = str(arguments)
        if len(rendered) > limit:
            rendered = f"{rendered[:limit]}…"
        parts.append(f"{name}({rendered})")
    return " | ".join(parts)


def step_outcome(
    response: dict[str, Any] | None,
    *,
    python_ast: bool = True,
    json_content: bool = True,
    tool_name: bool = False,
    text_limit: int = 240,
) -> Outcome:
    """Classify what a step did, in a form two runs can be compared on.

    `tool_name` clusters tool calls by which tools were called, ignoring every argument.
    Measured need: at a real checkpoint, three samples each chose to run code, and each
    wrote a structurally different program — three clusters of one, where the question
    being asked was simply "read the file, or run code?".
    """
    if not response:
        return Outcome("error", "error:no-response", "no response recorded")

    choices = response.get("choices") or []
    if not choices:
        return Outcome("error", "error:no-choices", "response had no choices")

    finish = choices[0].get("finish_reason")
    if finish == "length":
        # Cut off by max_tokens: whatever was produced is a fragment, not a behaviour, and
        # clustering the fragments would invent distinctions that are not there.
        return Outcome("truncated", "truncated", "truncated at max_tokens")

    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if calls and tool_name:
        names = [(call.get("function") or {}).get("name") or "?" for call in calls]
        return Outcome(
            "tool_call",
            "tools:" + ",".join(names),
            " | ".join(f"{name}(…)" for name in names),
            calls_text(calls),
        )
    if calls:
        canonical = [
            canonical_tool_call(
                mask_volatile(call), python_ast=python_ast, json_content=json_content
            )
            for call in calls
        ]
        key = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return Outcome("tool_call", key, describe_tool_calls(canonical), calls_text(calls))

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        normalized = normalize_text(content)
        label = normalized if len(normalized) <= text_limit else f"{normalized[:text_limit]}…"
        return Outcome("text", f"text:{normalized}", label, mask_workspace(content))

    return Outcome("empty", f"empty:{finish or 'unknown'}", f"no content (finish_reason={finish})")


def calls_text(calls: Sequence[dict[str, Any]]) -> str:
    """Tool calls as what they say: each call's tool, then the values of its arguments as the
    model wrote them, one call a line, per-run paths masked."""
    lines = []
    for call in calls:
        function = call.get("function") or {}
        raw = function.get("arguments")
        try:
            arguments: Any = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            arguments = raw
        values: list[str] = []
        _argument_values(arguments, values)
        lines.append(" ".join([str(function.get("name") or "?"), *values]))
    return mask_workspace("\n".join(lines))


def _argument_values(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _argument_values(item, out)
    elif isinstance(value, list):
        for item in value:
            _argument_values(item, out)
    elif value is not None:
        out.append(str(value))


def called_tools(outcome: Outcome) -> tuple[str, ...] | None:
    """The tools an outcome's calls named, in order, read back from its key; None when it
    called none."""
    if outcome.kind != "tool_call":
        return None
    if outcome.key.startswith("tools:"):
        return tuple(outcome.key[len("tools:"):].split(","))
    try:
        calls = json.loads(outcome.key)
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list):
        return None
    return tuple(str(call.get("name") or "?") for call in calls if isinstance(call, dict))


def validate_normalizers(names: Iterable[str]) -> list[str]:
    names = list(names)
    unknown = [name for name in names if name not in KNOWN_NORMALIZERS]
    if unknown:
        raise ValueError(
            f"unknown normalizer(s) {', '.join(unknown)}; known: {', '.join(KNOWN_NORMALIZERS)}"
        )
    return names


#: A recorded step whose error starts with one of these holds no answer from the model — the
#: server failed, the stream broke off part-way, or the response could not be read — so it
#: is an error, never a behaviour. A call dropped at a breakpoint, or whose harness hung up
#: while it was held, does hold the answer the model gave: that stays its behaviour.
TRANSPORT_FAILURES = (UPSTREAM_ERROR, CUT_OFF.split(":", 1)[0], "tuningfork could not read")


def recorded_outcome(
    response: dict[str, Any] | None, error: str | None = None, normalizers: Iterable[str] = ()
) -> Outcome:
    """A recorded step's outcome: an error when its call failed, else what the model did.

    A cut-off stream keeps the partial answer it assembled, which would otherwise be
    clustered as a behaviour the model never finished.
    """
    if error and error.startswith(TRANSPORT_FAILURES):
        return Outcome("error", f"error:{error.split(':', 1)[0]}", error)
    return outcome_for(response, normalizers)


def content_text(content: Any) -> str:
    """A message's content as text: a list of parts joined, a missing content empty."""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content if isinstance(content, str) else ""


def reply_text(response: dict[str, Any] | None) -> str:
    """The text a response says to its reader: the message's content, without reasoning or
    tool calls. Empty when the model answered with tool calls only, or not at all — so a
    run's last step gives its final reply, or nothing when it ended mid-task."""
    choices = (response or {}).get("choices") or []
    message = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
    return content_text(message.get("content")).strip()


def outcome_for(response: dict[str, Any] | None, normalizers: Iterable[str] = ()) -> Outcome:
    """Apply the named normalizers. Naming none turns every normalizer on."""
    names = list(normalizers)
    if not names:
        return step_outcome(response)
    return step_outcome(
        response,
        python_ast="python_ast" in names,
        json_content="json_content" in names,
        tool_name="tool_name" in names,
    )
