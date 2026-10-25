"""What the store masks, and how: a write into the recordings folder's database and blobs masks
what it is given here, once, with the store's one set of patterns — masking is not idempotent (a
command's second `-p` falls only to a second pass), and a fork matches a call masked once against
a step stored masked once. What a run is given is kept as it was given, since a fork runs it
again: a launch's eval and variant, and a scenario's context (`Store.put_given`).

A writer that must read a value as it will be stored first — an outcome computed from a
sample's answer, so a report recomputing it from the store agrees — asks for it masked
(`Masking.of`) and hands the store the `Masked` it got: a write stores that as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from tuningfork.config import (
    _FOUND, DEFAULT_REDACTIONS, REDACTED, _masking, patterns_digest, redact_json_found, redact_spans,
)


@dataclass(frozen=True, slots=True)
class Masked:
    """A value as the store masks it: how many masks that made, and the values they took that
    are masked wherever else they stand in the same step (`config._FOUND`)."""

    value: Any
    made: int = 0
    found: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Masking:
    """The store's patterns, and what each kind of content is masked as."""

    patterns: tuple[str, ...] = DEFAULT_REDACTIONS

    @property
    def digest(self) -> str:
        return patterns_digest(self.patterns)

    def of(self, value: Any, *, also: Iterable[str] = (), masked: Iterable[str] = ()) -> Masked:
        """`value` masked — text as text, anything else as the JSON it is written as — and every
        place a value `also` names stands. A `Masked` is kept as it is, and so are the fields of an
        object that `masked` names: masked where they were made — a check's output, before it was
        cut — and masked again, a command's second password would fall too."""
        if isinstance(value, Masked) or value is None:
            return value if isinstance(value, Masked) else Masked(None)
        if isinstance(value, str):
            text, made, taken = _masking(value, self.patterns, tuple(also))
            return Masked(text, made, frozenset(taken))
        if isinstance(value, dict) and masked:
            kept = {key: value[key] for key in masked if key in value}
            rest = self.of({key: item for key, item in value.items() if key not in kept}, also=also)
            # In the object's own order; a key masking took apart, or a drop's marker, after it.
            ordered = {key: kept[key] if key in kept else rest.value[key]
                       for key in value if key in kept or key in rest.value}
            return Masked({**ordered, **rest.value, **kept}, rest.made, rest.found)
        written, made, found = redact_json_found(value, self.patterns, tuple(also))
        return Masked(written, made, found)

    def text(self, text: str, *, also: Iterable[str] = ()) -> Masked:
        """`text` masked every way storage reads a text (`config.redact_spans`): as it stands and as
        its escapes read, as JSON writes it, its quotes escaped, and wherever a value `also` names
        stands. Read as text alone (`of`), a password before a typed declaration's quoted key stands;
        read as a string in JSON — a run's result — a key three escapes deep does, and a value `also`
        names written as a number, or a pattern took part of. A check's output is masked so."""
        spans = redact_spans(text, self.patterns, tuple(also))
        pieces, taken, at = [], [], 0
        for start, end in spans:
            pieces += [text[at:start], REDACTED]
            if _FOUND.fullmatch(text, start, end):
                taken.append(text[start:end])
            at = end
        return Masked("".join(pieces) + text[at:], len(spans), frozenset(taken))

    def value(self, value: Any, *, also: Iterable[str] = ()) -> Any:
        """`value` masked, for a write that keeps no count."""
        return self.of(value, also=also).value

    def stream(self, stream: Any, *, also: Iterable[str] = ()) -> Any:
        """A token stream, a `TokenStream` or its JSON form, masked token-aware (`redact_stream`);
        anything else that is not one, masked as JSON."""
        from tuningfork.analysis.tokens import TokenStream, redact_stream

        if isinstance(stream, Masked):
            return stream.value
        try:
            tokens = stream if isinstance(stream, TokenStream) else TokenStream.from_json(stream)
        except (KeyError, TypeError, ValueError, AttributeError):
            return self.value(stream, also=also)
        return redact_stream(tokens, self.patterns, tuple(also)).to_json()

    def raw(
        self,
        body: bytes,
        *,
        answer: dict[str, Any] | None,
        masked: dict[str, Any] | None,
        wire: Any,
        request: dict[str, Any],
        streamed: bool,
        also: Iterable[str] = (),
    ) -> tuple[bytes | None, str | None]:
        """An answer's raw bytes as they are kept, and how they were masked beyond the text
        (`bytes_without_secrets`). They are masked as text by the patterns — bytes that are not
        UTF-8 too, and otherwise kept as they came — but not of what masks took elsewhere: a
        number holding its digits would break; the bytes are read back whole for it instead."""
        text = body.decode("utf-8", "surrogateescape")
        kept = _masking(text, self.patterns, ())[0].encode("utf-8", "surrogateescape")
        return bytes_without_secrets(
            kept, answer, wire, request, self.patterns, masked=masked, streamed=streamed, also=also
        )


def bytes_without_secrets(
    body: bytes,
    answer: dict[str, Any] | None,
    wire: Any,
    request: dict[str, Any],
    patterns: tuple[str, ...],
    *,
    masked: dict[str, Any] | None = None,
    streamed: bool = True,
    also: Iterable[str] = (),
) -> tuple[bytes | None, str | None]:
    """The bytes to keep of an answer, and how they were redacted beyond the text, if they were:
    "rendered" or "dropped" (a step's `raw_redacted`).

    `body` is the answer's bytes redacted as text, and `answer` what they read as before that, in
    the common shape, logprobs apart; `masked`, that answer redacted, where it already is.

    A model's answer streams about a token an event, so a key it writes arrives in pieces that no
    pattern sees whole, or masked only as far as one event holds it, and a fork reading the bytes
    back would give it again. An answer sent at once keeps what a stream does besides its text:
    logprobs, whose tokens spell a secret, and fields the common shape leaves out — a thinking
    block's signature. So the bytes are read back whole, as the proxy reads an answer, and kept
    as they came only where nothing left in them masks — every answer without a secret, and one
    whose secrets each sat whole in an event, masked there — and carry no logprobs where the
    answer held a secret. Otherwise they are the answer written again (`wire.render`) from its
    masked form; the step says so, and a fork refuses to replay it. That drops what no reader of
    the bytes needs — logprobs, a cut-off end or an error event (the step keeps its error), usage
    the request did not ask for, a thinking block's signature, the server's timings. Where the
    masked answer cannot be written again, no bytes are kept: losing them beats storing the
    secret. Bytes that cannot be read back, nor an answer the proxy could not read at all, are
    kept as the text left them, with every place a value `also` names masked as text: they hold
    no JSON a mask could break. Such a step has an error, which scoring and a fork refuse first.
    """
    from tuningfork.analysis.tokens import split_logprobs

    found = tuple(also)

    def as_text(unread: bytes) -> bytes:
        return _masking(unread.decode("utf-8", "surrogateescape"), (), found)[0].encode("utf-8", "surrogateescape")

    if not answer:
        return as_text(body), None
    if masked is None:
        masked = redact_json_found(answer, patterns, found)[0]
    native: dict[str, Any] | None = None
    try:
        native = wire.parse(body, streamed)
        kept = wire.canonical_response(native, stream=body if streamed else None) if native else None
    except Exception:  # noqa: BLE001 - bytes that cannot be read back are written again
        kept = None
    if isinstance(kept, dict) and redact_json_found(native, patterns, found)[1] == 0:
        if masked == answer:
            return body, None
        spelled = any(isinstance(choice, dict) and "logprobs" in choice for choice in kept.get("choices") or [])
        if split_logprobs(kept)[0] == masked and not spelled:
            return body, None
    elif not isinstance(kept, dict) and masked == answer:
        return as_text(body), None
    if "tuningfork_error" in masked:
        return None, "dropped"  # masked, the answer no longer reads: written again, it would be an empty one
    try:
        return wire.render(wire.native_response(masked, native), request={**request, "stream": streamed}), "rendered"
    except Exception:  # noqa: BLE001 - see above
        return None, "dropped"


__all__ = ["Masked", "Masking", "bytes_without_secrets"]
