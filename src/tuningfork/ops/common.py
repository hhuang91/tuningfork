"""What every operation shares: refusals, cancellation, progress events, typed parameters."""

from __future__ import annotations

import dataclasses
import logging
import threading
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from tuningfork.owners import current_owner, owner_alive  # noqa: F401 - re-exported

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationError(Exception):
    """A refusal a person can act on: what is wrong, and, when there is one, what to do.

    Raised instead of printing: a front end decides how to show it. `hint` is a short
    remedy that belongs beside the problem; `detail` is a longer explanation that follows
    it, such as a diagnosis of where a skill's text went. `code` names the refusal for a
    program; `field` names the parameter at fault, so a form can point at it.
    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        detail: str | None = None,
        *,
        code: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail
        self.code = code
        self.field = field

    def to_json(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "hint": self.hint,
            "detail": self.detail,
            "code": self.code,
            "field": self.field,
        }


class Cancelled(Exception):
    """An operation stopped because it was asked to. What it finished is kept.

    `reason` is `cancelled` when a person asked, or `shutdown` when tuningfork itself is
    stopping. `partial` is the operation's result over what finished, when it has one.
    """

    def __init__(self, reason: str = "cancelled", *, partial: Any = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial = partial


class Cancel:
    """Cooperative cancellation, safe to trigger from any thread.

    Operations check it between units of work — a sample, a run — and stop there, so what
    was already recorded stays consistent. `on_cancel` callbacks run once, when cancelled:
    for work that cannot poll, such as a harness process to be killed. A callback that
    fails is logged, and the others still run.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next = 0
        self.reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self._event.set()
            callbacks, self._callbacks = list(self._callbacks.values()), {}
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - one broken callback must not stop the rest
                logger.exception("a cancellation callback failed")

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Run `callback` when cancelled, or now if already cancelled. Returns a function
        that unregisters it, for work that finishes before any cancel comes."""
        with self._lock:
            if not self._event.is_set():
                key = self._next
                self._next += 1
                self._callbacks[key] = callback

                def unregister() -> None:
                    with self._lock:
                        self._callbacks.pop(key, None)

                return unregister
        callback()
        return lambda: None

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled(self.reason or "cancelled")

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`, waking early if cancelled. True if cancelled."""
        return self._event.wait(max(0.0, seconds))


@dataclass(slots=True)
class Event:
    """A progress event: `kind` names it, `data` is its JSON-ready content.

    `subject` carries the typed object behind it — a sample, a finished run — for a front end
    in the same process, such as the CLI's printers. It never leaves the process.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    subject: Any = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data}


#: Where an operation reports progress. Called from whichever thread did the work.
Progress = Callable[[Event], None]


def emit(progress: Progress | None, kind: str, /, subject: Any = None, **data: Any) -> None:
    """Report an event, if anyone listens. `kind` is positional, so an event's data may
    carry a field of that name too."""
    if progress is not None:
        progress(Event(kind, data, subject))


# --- typed parameters ----------------------------------------------------------------------


def kept_params(parsed: Any, masking: Any) -> dict[str, Any]:
    """A job's parameters as they are kept and shown — on the Jobs page, to whoever reads the
    jobs API — masked as the store masks (`masking`, its own), and a harness's options as its
    plan shows them: every value under an MCP server's `headers` or `env` masked, whatever its
    name, which no pattern would know for a secret (`shown_options`). The job runs with them as
    given; what is kept is never run."""
    from tuningfork.ops.batch import shown_options

    data = dataclasses.asdict(parsed)
    if isinstance(data.get("harness_options"), dict):
        data["harness_options"] = shown_options(data["harness_options"])
    return masking.value(data)


def parse_params(cls: type[T], data: Any) -> T:
    """Build `cls` (a dataclass) from untrusted JSON, strictly.

    Unknown keys are refused, and so are near-misses of a type — `"false"` is not a bool, a
    bool is not an int — because a parameter silently read the wrong way changes what an
    experiment measures. Each refusal names the field.
    """
    if not isinstance(data, dict):
        raise OperationError("parameters must be a JSON object", code="bad_params")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise OperationError(
            f"unknown parameter {unknown[0]!r}",
            hint=f"accepted: {', '.join(sorted(fields))}",
            code="bad_params",
            field=unknown[0],
        )
    values: dict[str, Any] = {}
    for name, spec in fields.items():
        if name not in data:
            if spec.default is dataclasses.MISSING and spec.default_factory is dataclasses.MISSING:
                raise OperationError(f"{name} is required", code="bad_params", field=name)
            continue
        values[name] = _coerce(data[name], hints[name], name)
        _within(values[name], spec, name)
    return cls(**values)


def _within(value: Any, spec: dataclasses.Field, name: str) -> None:
    """A number inside the bounds its field declares (`metadata` minimum and maximum)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    low, high = spec.metadata.get("minimum"), spec.metadata.get("maximum")
    if low is not None and value < low:
        raise OperationError(f"{name} must be at least {low}", code="bad_params", field=name)
    if high is not None and value > high:
        raise OperationError(f"{name} must be at most {high}", code="bad_params", field=name)


def _coerce(value: Any, hint: Any, name: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if hint is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, option, name)
            except OperationError as exc:
                errors.append(exc)
        raise errors[0] if errors else _wrong(name, hint, value)
    if origin is typing.Literal:
        if value in args:
            return value
        raise OperationError(
            f"{name} must be one of {', '.join(repr(a) for a in args)}", code="bad_params", field=name
        )
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise _wrong(name, hint, value)
        inner = args[0] if args else Any
        items = [_coerce(item, inner, name) for item in value]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise _wrong(name, hint, value)
        inner = args[1] if len(args) == 2 else Any
        return {k: _coerce(v, inner, name) for k, v in value.items()}
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise _wrong(name, hint, value)
    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise _wrong(name, hint, value)
    if hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise _wrong(name, hint, value)
    if hint is str:
        if isinstance(value, str):
            return value
        raise _wrong(name, hint, value)
    raise OperationError(f"{name} has a type parameters cannot carry", code="bad_params", field=name)


def _wrong(name: str, hint: Any, value: Any) -> OperationError:
    return OperationError(
        f"{name} must be {_type_name(hint)}, not {type(value).__name__}",
        code="bad_params",
        field=name,
    )


def _type_name(hint: Any) -> str:
    origin = typing.get_origin(hint)
    if origin in (list, tuple):
        return "a list"
    if origin is dict:
        return "an object"
    return {bool: "true or false", int: "a whole number", float: "a number", str: "text"}.get(
        hint, str(hint)
    )


def params_schema(cls: type) -> dict[str, Any]:
    """A JSON Schema (draft 2020-12) of a parameters dataclass: what a form shows, and what
    an agent reads in its tool list — llama.cpp even builds the grammar of a tool call from
    it, so it says exactly what `parse_params` accepts, no more.

    Each field's help text comes from `field(metadata={"help": ...})`. A field whose shape
    its type cannot say — a list of conditions, files to write — gives its own schema as
    `metadata["schema"]`. An optional field is its plain type with a default of null: the
    parser takes null for it, and a reader sees one type rather than a union."""
    hints = typing.get_type_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for spec in dataclasses.fields(cls):
        if not spec.init:
            continue
        entry = dict(spec.metadata["schema"]) if "schema" in spec.metadata else _schema(hints[spec.name])
        if spec.metadata.get("help"):
            entry["description"] = spec.metadata["help"]
        if spec.default is not dataclasses.MISSING:
            entry["default"] = spec.default
        elif spec.default_factory is not dataclasses.MISSING:
            entry["default"] = spec.default_factory()
        else:
            required.append(spec.name)
        for key in ("minimum", "maximum"):
            if key in spec.metadata:
                entry[key] = spec.metadata[key]
        properties[spec.name] = entry
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _schema(hint: Any) -> dict[str, Any]:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _schema(options[0]) if len(options) == 1 else {"anyOf": [_schema(a) for a in options]}
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        named = {str: "string", int: "integer"}.get(kinds.pop()) if len(kinds) == 1 else None
        return {"type": named, "enum": list(args)} if named else {"enum": list(args)}
    if origin in (list, tuple):
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if origin is dict:
        inner = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": _schema(inner)} if inner is not Any else {"type": "object"}
    return {
        bool: {"type": "boolean"},
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
    }.get(hint, {})
