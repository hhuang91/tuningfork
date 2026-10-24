"""What the store masks, and how: every write into the recordings folder masks what it is given
here, with the store's one set of patterns, so nothing is stored unmasked and nothing is masked
twice — masking is not idempotent (a command's second `-p` falls only to a second pass), and a
fork matches a call masked once against a step stored masked once.

A writer that must read a value as it will be stored first — an outcome computed from a
sample's answer, so a report recomputing it from the store agrees — asks for it masked
(`Masking.of`) and hands the store the `Masked` it got: a write stores that as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from tuningfork.config import DEFAULT_REDACTIONS, _masking, patterns_digest, redact_json_found


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

    def of(self, value: Any, *, also: Iterable[str] = ()) -> Masked:
        """`value` masked — text as text, anything else as the JSON it is written as — and every
        place a value `also` names stands. A `Masked` is kept as it is."""
        if isinstance(value, Masked) or value is None:
            return value if isinstance(value, Masked) else Masked(None)
        if isinstance(value, str):
            text, made, taken = _masking(value, self.patterns, tuple(also))
            return Masked(text, made, frozenset(taken))
        masked, made, found = redact_json_found(value, self.patterns, tuple(also))
        return Masked(masked, made, found)

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
        (`bytes_without_secrets`). Bytes that are not UTF-8 are masked as text too, and
        otherwise kept as they came."""
        found = tuple(also)
        text = body.decode("utf-8", "surrogateescape")
        kept = _masking(text, self.patterns, found)[0].encode("utf-8", "surrogateescape")
        return bytes_without_secrets(
            kept, answer, wire, request, self.patterns, masked=masked, streamed=streamed, also=found
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
    secret. An answer the proxy could not read at all keeps its bytes as the text left them: its
    step has an error, which scoring and a fork refuse first.
    """
    from tuningfork.analysis.tokens import split_logprobs

    if not answer:
        return body, None
    found = tuple(also)
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
        return body, None
    if "tuningfork_error" in masked:
        return None, "dropped"  # masked, the answer no longer reads: written again, it would be an empty one
    try:
        return wire.render(wire.native_response(masked, native), request={**request, "stream": streamed}), "rendered"
    except Exception:  # noqa: BLE001 - see above
        return None, "dropped"


__all__ = ["Masked", "Masking", "bytes_without_secrets"]
