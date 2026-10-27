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

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tuningfork.config import (
    _FOUND, DEFAULT_REDACTIONS, REDACTED, _masking, _merged, patterns_digest, redact_json_found, text_spans,
)


#: What a value is shown as itself: a number, a truth, nothing. Any other is shown as its text.
_PLAIN = (bool, int, float, type(None))
#: Where in a text: from, to.
Span = tuple[int, int]


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

    def spans(self, text: str, *, also: Iterable[str] = ()) -> list[tuple[int, int]]:
        """Where `text` is masked as it is shown (`shown`): every way storage reads a text, and of
        what its own masks took too. An edit that cuts into one would change part of a key its
        reader was shown masked (`ops.diff.resolve_edits`)."""
        also = tuple(also)
        return text_spans(text, self.patterns, (*also, *self.text(text, also=also).found))

    def alike(self, texts: Sequence[str]) -> list[str]:
        """Texts shown side by side — a text and what was made of it, the two sides of a diff — each
        masked whole, each line at its number, of what the masks took in any of them; each masked
        wherever it holds, unchanged, text another masks — a line they share, the characters a
        changed line shares (`_carried`) — and, on its changed lines, wherever a mask's value
        stands as a word of its own, or glued as its mask was (`_values`, `_found_in`). A key's
        pattern reads its context, which an edit can take away — a private key's first line, the
        `-u` before a password — leaving what it masked to read as no key on its own. The first is
        masked as each other is shown beside it, and each other as the first: a line two of them
        share reads alike."""
        texts = list(texts)
        found = frozenset().union(*(self.text(text).found for text in texts))
        spans = [text_spans(text, self.patterns, tuple(found)) for text in texts]
        if len(texts) > 1:
            shared = [_shared(texts[0], other) for other in texts[1:]]
            values = _values(texts, spans, found)
            back = [span for k, (blocks, _, _) in enumerate(shared, 1)
                    for span in _carried(spans[k], [(theirs, ours, size, lines) for ours, theirs, size, lines in blocks],
                                         texts[0])]
            changed = _merged([span for _, ours, _ in shared for span in ours])
            spans[0] = _merged(spans[0] + back + _found_in(texts[0], values, changed))
            for k, (blocks, _, theirs) in enumerate(shared, 1):
                spans[k] = _merged(spans[k] + _carried(spans[0], blocks, texts[k]) + _found_in(texts[k], values, theirs))
        return [_applied(text, masks, 0, len(text), lines=True)[0].value for text, masks in zip(texts, spans)]

    def bared(self, before: str, after: str) -> list[str]:
        """Each value `before` is shown masked that `after` holds unmasked more often than `before`
        does: an edit can take away what a key's pattern reads — `password=`, the `-u` before a
        password, a private key's first line — and leave the key to read as none wherever the
        edited text is kept or shown on its own, the steps of every run given it among them."""
        masks_before, masks_after = self.spans(before), self.spans(after)
        bare: list[str] = []
        for start, end in masks_before:
            value = before[start:end]
            if value not in bare and _bare(after, value, masks_after) > _bare(before, value, masks_before):
                bare.append(value)
        return bare

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
        pieces += [text[at:span_start], _marked(text[span_start:span_end]) if lines else REDACTED]
        if _FOUND.fullmatch(text, span_start, span_end):
            taken.append(text[span_start:span_end])
        at, made = span_end, made + 1
    return Masked("".join(pieces) + text[at:end], made, frozenset(taken)), start, end


def _marked(taken: str) -> str:
    """A mask over `taken`, in a text whose lines keep their numbers: `<redacted>` on each line it
    covers, each followed by the break it took (a carriage return alone, U+2028 and the rest too,
    as `str.splitlines` splits them). Shown blank, a private key's lines read as lines with
    nothing on them, which an edit could take."""
    pieces = []
    for piece in taken.splitlines(keepends=True):
        body = (piece.splitlines() or [""])[0]
        pieces.append(REDACTED + piece[len(body):])
    return "".join(pieces) or REDACTED


#: The most characters of two runs of changed lines compared one by one (their lengths
#: multiplied): beyond it, only their lines are matched.
CHARS_COMPARED = 2_000_000
#: The fewest characters of a mask's value found by its value on changed lines: fewer stand
#: anywhere by chance.
VALUE_LEAST = 4


def _shared(first: str, other: str) -> tuple[list[tuple[int, int, int, bool]], list[Span], list[Span]]:
    """What two texts share, as (where in the first, where in the other, how long, whether whole
    lines): their lines matched, then the characters of each run of lines that changed — a line
    can keep a password and lose the `-u` before it. With the lines that changed, where they stand
    in the first and in the other."""
    import difflib

    def offsets(lines: list[str]) -> list[int]:
        at = [0]
        for line in lines:
            at.append(at[-1] + len(line))
        return at

    ours, theirs = first.splitlines(keepends=True), other.splitlines(keepends=True)
    at_ours, at_theirs = offsets(ours), offsets(theirs)
    blocks: list[tuple[int, int, int, bool]] = []
    changed: list[Span] = []
    changed_other: list[Span] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ours, theirs).get_opcodes():
        start, begin = at_ours[i1], at_theirs[j1]
        if tag == "equal":
            blocks.append((start, begin, at_ours[i2] - start, True))
            continue
        changed += [(start, at_ours[i2])] if i1 < i2 else []
        changed_other += [(begin, at_theirs[j2])] if j1 < j2 else []
        if tag == "replace" and (at_ours[i2] - start) * (at_theirs[j2] - begin) <= CHARS_COMPARED:
            matcher = difflib.SequenceMatcher(None, first[start:at_ours[i2]], other[begin:at_theirs[j2]], autojunk=False)
            blocks += [(start + i, begin + j, size, False) for i, j, size in matcher.get_matching_blocks() if size]
    return blocks, changed, changed_other


#: The fewest characters of a mask a run of shared characters carries into another text, but for a
#: mask it holds whole: fewer are as likely chance — the letters a word shares with a key.
CARRIED_LEAST = 8


def _carried(spans: list[Span], blocks: list[tuple[int, int, int, bool]], into: str) -> list[Span]:
    """`spans` of one text — sorted and apart, as `text_spans` and `_merged` give them — carried into
    another, `into`, through what they share (`blocks`, as `_shared` gives them): each part of a span
    inside a run of shared lines, where it stands in the other; inside a run of shared characters,
    `CARRIED_LEAST` characters of one, or a shorter span it holds whole where it may land (`_lands`;
    found in review: a key's line rewritten as `rotated` read `rot<redacted>t<redacted>d`). Each run
    reads only the spans that reach it (found in review: every span against every run took seconds
    for a few thousand lines)."""
    starts = [start for start, _ in spans]
    carried = []
    for ours, theirs, size, lines in blocks:
        k = max(bisect_right(starts, ours) - 1, 0)
        while k < len(spans) and spans[k][0] < ours + size:
            start, end = spans[k]
            low, high = max(start, ours), min(end, ours + size)
            there = (low - ours + theirs, high - ours + theirs)
            if low < high and (lines or high - low >= CARRIED_LEAST
                               or ((low, high) == (start, end) and _lands(into, *there, start > ours, end < ours + size))):
                carried.append(there)
            k += 1
    return carried


def _values(texts: Sequence[str], spans: Sequence[list[Span]], found: frozenset[str]) -> list[str]:
    """What the masks of `texts` took that is found by its value on changed lines (`_found_in`):
    each value once, with the characters before and after its masks, `VALUE_LEAST` characters or
    more, and never one `found` masks wherever it stands already, nor one without a letter —
    digits, a PIN or a year, or punctuation, `####`, stand for any number or rule in a text, and
    masking those that read the same would say, beside the text shown alone, which of them is the
    key (found in review: `1234` masked in `51234`, `1,234` in a count, `####` in a heading)."""
    values: dict[str, tuple[set[str], set[str]]] = {}
    for text, masks in zip(texts, spans):
        for start, end in masks:
            value = text[start:end]
            if len(value) >= VALUE_LEAST and any(char.isalpha() for char in value) and value not in found:
                before, after = values.setdefault(value, (set(), set()))
                before.update(text[start - 1:start])
                after.update(text[end:end + 1])
    return [(value, frozenset(before), frozenset(after)) for value, (before, after) in values.items()]


def _found_in(text: str, values: list[tuple[str, frozenset[str], frozenset[str]]], changed: list[Span]) -> list[Span]:
    """Where each of `values` (as `_values` gives them) stands on `text`'s changed lines (`changed`): a
    run of changed lines too long to compare by characters (`CHARS_COMPARED`), or a line moved, carries
    no mask onto them (found in review: a four-character password whose `mysql -p` was rewritten among
    eighty lines changed showed on the other side). Every text's changed lines are read alike, so a
    line two texts hold unchanged reads alike in both. One glued to a letter, a digit or `_` is
    another word (found in review: `run_1234.csv`) — but one its own mask was glued to: the same
    option, mysql's `-pZ9q4` (found in review: `mycli -pZ9q4`)."""
    carried = []
    for value, before, after in values:
        for begin, stop in changed:
            at = text.find(value, begin, stop)
            while at >= 0:
                glued_before, glued_after = _glued(text, at, at + len(value))
                if not (glued_before and text[at - 1] not in before) and not (glued_after and text[at + len(value)] not in after):
                    carried.append((at, at + len(value)))
                at = text.find(value, at + 1, stop)
    return carried


def _lands(into: str, start: int, end: int, kept_before: bool, kept_after: bool) -> bool:
    """Where a short mask carried whole through a run of shared characters may land in `into`: one with a
    letter wherever the run puts it, glued to its option or not (mysql's `-pZ9q4`, rewritten `-PZ9q4`); one
    without a letter only as a word of its own, or glued to a character the run kept (found in review:
    `-p1234` rewritten as `port 51234` read `port 5<redacted>`; as `mycli -p1234`, the run kept the `p`)."""
    if any(char.isalpha() for char in into[start:end]):
        return True
    glued_before, glued_after = _glued(into, start, end)
    return not (glued_before and not kept_before) and not (glued_after and not kept_after)


def _glued(text: str, start: int, end: int) -> tuple[bool, bool]:
    """Whether `text[start:end]` is glued, before it and after it, to a letter, a digit or `_` where
    its own edge is one."""
    def word(char: str) -> bool:
        return char.isalnum() or char == "_"

    return (start > 0 and word(text[start]) and word(text[start - 1]),
            end < len(text) and word(text[end - 1]) and word(text[end]))


def _bare(text: str, value: str, spans: list[tuple[int, int]]) -> int:
    """How often `value` stands in `text` outside every span."""
    count, at = 0, text.find(value)
    while at >= 0:
        count += not any(low <= at and at + len(value) <= high for low, high in spans)
        at = text.find(value, at + 1)
    return count


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
