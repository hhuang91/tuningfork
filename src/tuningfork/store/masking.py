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
from typing import Any, Callable, Iterable, Sequence

from tuningfork.config import (
    _FOUND, DEFAULT_REDACTIONS, REDACTED, _masking, patterns_digest, redact_json_found, text_spans,
)


#: What a value is shown as itself: a number, a truth, nothing. Any other is shown as its text.
_PLAIN = (bool, int, float, type(None))


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

    def text(self, text: str, *, also: Iterable[str] = (), lines: bool = False) -> Masked:
        """`text` masked every way storage reads a text (`config.text_spans`): as it stands and as
        its escapes read, as JSON writes it, its quotes escaped, and wherever a value `also` names
        stands, as it is written or as its escapes read. Read as text alone (`of`), a password before
        a typed declaration's quoted key stands; read as a string in JSON — a run's result — a key
        three escapes deep does, and a value `also` names written as a number, or a pattern took part
        of. A check's output is masked so. `lines`: each mask followed by the line breaks it took, so
        that every line keeps its number — a private key's lines masked whole, say, in a diff."""
        return _applied(text, text_spans(text, self.patterns, tuple(also)), 0, len(text), lines=lines)[0]

    def part(
        self, text: str, start: int = 0, end: int | None = None, *, also: Iterable[str] = (),
    ) -> tuple[str, int, int]:
        """`text` masked every way (`text`), and its part from `start` to `end` given: what is given,
        and where in `text` it runs from and to — each edge moved past a mask that covers it. Read on
        to either side, a key the edge falls inside is found whole: what is given holds neither its
        end nor half of its mask. Where a check's output is long, its end is given so."""
        spans = text_spans(text, self.patterns, tuple(also))
        end = len(text) if end is None else end
        for span_start, span_end in spans:
            if span_start < start < span_end:
                start = span_end
            if span_start < end < span_end:
                end = span_start
        given, start, end = _applied(text, spans, start, max(start, end))
        return given.value, start, end

    def quoted(self, text: str, *, also: Iterable[str] = ()) -> Callable[[int, int], str]:
        """`text` as it is shown — masked whole, of what its own masks took too — quoted a part at a
        time, from `start` to `end`: each part as `part` gives it, an edge moved past a mask that
        covers it. Masked alone, a part can read as no key: a line of a private key, or a line saying
        bare a password another line names."""
        also = tuple(also)
        spans = text_spans(text, self.patterns, (*also, *self.text(text, also=also).found))

        def quote(start: int, end: int) -> str:
            for span_start, span_end in spans:
                if span_start < start < span_end:
                    start = span_end
                if span_start < end < span_end:
                    end = span_start
            return _applied(text, spans, start, max(start, end))[0].value

        return quote

    def alike(self, texts: Sequence[str]) -> list[str]:
        """Texts shown side by side — a text and what was made of it, the two sides of a diff — each
        masked whole and alike (`shown`, each line at its number), and each line the first holds
        shown as the first shows it, wherever it stands in the others: an edit can take away what a
        key's pattern needs, a private key's first line, and leave its body to read as no key."""
        texts = list(texts)
        shown = self.shown(texts, lines=True).value
        if not texts:
            return shown

        def lines_of(text: str) -> list[tuple[str, str]]:
            """Each line, and the break after it."""
            found = []
            for line in text.splitlines(keepends=True):
                body = (line.splitlines() or [""])[0]
                found.append((body, line[len(body):]))
            return found

        first, first_shown = lines_of(texts[0]), lines_of(shown[0])
        held = {raw: said for (raw, _), (said, _) in zip(first, first_shown) if raw != said}
        if not held or len(first) != len(first_shown):
            return shown
        result = [shown[0]]
        for text, said in zip(texts[1:], shown[1:]):
            raw_lines, said_lines = lines_of(text), lines_of(said)
            if len(raw_lines) != len(said_lines):  # each mask keeps its breaks: never
                result.append(said)
                continue
            result.append("".join(held.get(raw, body) + end for (raw, _), (body, end) in zip(raw_lines, said_lines)))
        return result

    def value(self, value: Any, *, also: Iterable[str] = ()) -> Any:
        """`value` masked, for a write that keeps no count."""
        return self.of(value, also=also).value

    def shown(self, value: Any, *, also: Iterable[str] = (), lines: bool = False) -> Masked:
        """`value` as it is shown outside tuningfork — to an agent over MCP, whose model may be
        hosted, or to a proposing model: every text in it, a key's too, masked as text is (`text`),
        and wherever a value any of those masks took stands, in that text or another — a password
        named in one field and bare in another. What the store keeps is masked once already, and
        masked again it loses no more than a second command's password; what it keeps as given — a
        launch's eval, a workspace's files, a skill — is masked here. Its numbers and its shape are
        kept. A text a view splits — into numbered lines, pages, a diff — is shown so whole first,
        with `lines` where its line numbers must hold: masked line by line, a private key's lines
        each read as no key. A value JSON cannot write is read as the text it would be written as
        (`default=str`), and masked so: an answer is written so."""
        also = tuple(also)
        strings: dict[str, Masked] = {}

        def read(item: Any) -> None:
            if isinstance(item, dict):
                for key, each in item.items():
                    read(key)
                    read(each)
            elif isinstance(item, (list, tuple)):
                for each in item:
                    read(each)
            elif not isinstance(item, _PLAIN) and str(item) not in strings:
                strings[str(item)] = self.text(str(item), also=also, lines=lines)

        read(value)
        found = frozenset().union(*(masked.found for masked in strings.values()))
        if found - set(also):
            strings = {text: self.text(text, also=(*also, *found), lines=lines) for text in strings}

        def written(item: Any) -> Any:
            if isinstance(item, dict):
                kept: dict[Any, Any] = {}
                for key, each in item.items():
                    name = written(key)
                    while name in kept:  # two keys a mask made one: each kept
                        name = f"{name} "
                    kept[name] = written(each)
                return kept
            if isinstance(item, (list, tuple)):
                return [written(each) for each in item]
            return item if isinstance(item, _PLAIN) else strings[str(item)].value

        return Masked(written(value), sum(masked.made for masked in strings.values()), found)

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


def _applied(
    text: str, spans: list[tuple[int, int]], start: int, end: int, *, lines: bool = False,
) -> tuple[Masked, int, int]:
    """`text` from `start` to `end`, each of `spans` in it a mask — none of which an edge falls inside;
    `lines`, each mask followed by as many line breaks as it took, each as a line is split
    (`str.splitlines`: a carriage return alone, U+2028 and the rest too)."""
    pieces, taken, at, made = [], [], start, 0
    for span_start, span_end in spans:
        if span_end <= start or span_start >= end:
            continue
        breaks = len(f"x{text[span_start:span_end]}x".splitlines()) - 1 if lines else 0
        pieces += [text[at:span_start], REDACTED + "\n" * breaks]
        if _FOUND.fullmatch(text, span_start, span_end):
            taken.append(text[span_start:span_end])
        at, made = span_end, made + 1
    return Masked("".join(pieces) + text[at:end], made, frozenset(taken)), start, end


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
