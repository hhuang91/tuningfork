"""What the model chose, token by token, and how sure it was.

A *token stream* is an answer as the model's own tokens. At every position it keeps the token
chosen, its log-probability, and the K most likely tokens the model weighed there: the top of
its distribution *before* sampling, a softmax of the raw logits, as llama.cpp reports it. From
those come the numbers a heatmap shows:

- **entropy**, in bits: how spread out the choice was. Only the top K are known, so the mass
  beyond them is counted as one more outcome. That makes the figure a lower bound, and it is
  reported as one: a position whose tail holds a third of the mass is at least as uncertain as
  shown, never less.
- **margin**: the top probability minus the runner-up's. Near 0, two tokens were nearly tied.
- **surprise**: how unlikely the chosen token was, in bits.

Where the model *could* have gone another way is a different question from how spread out it
was: a sampler cuts the tail (top-k, top-p, min-p) before it draws. `effective` applies a
step's own sampler settings to the known tokens, and `candidates` ranks positions by how likely
the runner-up was under them — the positions a forking-token probe tests.

A stream is stored columnar, with a table of token pieces: the same pieces recur through an
answer and its alternatives. A piece that is not valid UTF-8 on its own — a character split
into byte tokens — is kept as its bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tuningfork.config import DEFAULT_REDACTIONS, REDACTED, redact_spans

VERSION = 1

#: What a position is, as the model's output format marks it (see `tuningfork.backends.formats`).
MARKER, REASONING, CONTENT, TOOL_CALL = "marker", "reasoning", "content", "tool_call"
#: The parts a probe region covers. Markers — channel and call delimiters — are never probed.
REGIONS: dict[str, tuple[str, ...]] = {
    "answer": (CONTENT, TOOL_CALL),
    "reasoning": (REASONING,),
    "all": (REASONING, CONTENT, TOOL_CALL),
}

#: llama.cpp reports a token of probability 0 with this logprob rather than -infinity.
FLOOR = -1e30


def probability(logprob: float | None) -> float:
    if logprob is None or logprob < FLOOR:
        return 0.0
    return math.exp(logprob)


def entropy_bits(logprobs: Iterable[float]) -> float:
    """Entropy of a top-K distribution, in bits, counting the unseen tail as one outcome."""
    ps = [probability(lp) for lp in logprobs]
    rest = max(0.0, 1.0 - sum(ps))
    h = -sum(p * math.log2(p) for p in ps if p > 0)
    if rest > 1e-12:
        h -= rest * math.log2(rest)
    return max(0.0, h)


def margin(logprobs: Iterable[float]) -> float:
    """The top probability minus the runner-up's."""
    ps = sorted((probability(lp) for lp in logprobs), reverse=True)
    if not ps:
        return 0.0
    return ps[0] - (ps[1] if len(ps) > 1 else 0.0)


#: Where a heatmap's eight levels begin: entropy in bits, and closeness (1 - margin).
ENTROPY_EDGES = (0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
CLOSENESS_EDGES = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)


def heat(value: float | None, edges: Sequence[float] = ENTROPY_EDGES) -> int:
    """A value's level on a heatmap, 0 (sure) to len(edges) (as unsure as the scale goes)."""
    if value is None:
        return 0
    return sum(1 for edge in edges if value >= edge)


def surprise_bits(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    p = probability(logprob)
    return -math.log2(p) if p > 0 else math.inf


# --- the stream -----------------------------------------------------------------------------


Piece = str | list[int]


def _piece_bytes(piece: Piece | None) -> bytes:
    if piece is None:
        return b""
    return bytes(piece) if isinstance(piece, list) else piece.encode("utf-8")


def _as_piece(raw: bytes) -> Piece:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return list(raw)


@dataclass(slots=True)
class TokenStream:
    """An answer as tokens. Position i holds `ids[i]`, chosen with `logprobs[i]`, and the top
    tokens the model weighed there, `top_ids[i]` with `top_logprobs[i]`. A redacted position
    has no id, no piece and no alternatives: only that something was there."""

    ids: list[int | None]
    logprobs: list[float | None]
    top_ids: list[list[int]]
    top_logprobs: list[list[float]]
    pieces: dict[int, Piece]
    top_k: int
    #: (kind, start, end) over positions, from the model's output format; empty if unknown.
    parts: list[tuple[str, int, int]] = field(default_factory=list)
    #: Masked positions, as (start, end) ranges.
    redacted: list[tuple[int, int]] = field(default_factory=list)
    #: The output format that marked the parts (`tuningfork.backends.formats`), if one did.
    format: str | None = None
    #: What the prompt did with the model's reasoning, the format's `thinking`, where it was
    #: known when the parts were marked: a template that opens the reasoning itself starts the
    #: answer inside it. None where the prompt was not known.
    thinking: bool | str | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def piece(self, token_id: int | None) -> Piece | None:
        return None if token_id is None else self.pieces.get(token_id)

    def token_bytes(self, position: int) -> bytes:
        return _piece_bytes(self.piece(self.ids[position]))

    def token_text(self, position: int) -> str:
        """One position's piece as text; empty when redacted."""
        return self.token_bytes(position).decode("utf-8", "replace")

    def token_text_of(self, token_id: int | None) -> str:
        """A token's piece as text, by its id."""
        return _piece_bytes(self.piece(token_id)).decode("utf-8", "replace")

    def mark_parts(self, fmt: Any, *, thinking: bool | str | None = None) -> TokenStream:
        """Mark what each position is, by an output format (`tuningfork.backends.formats`).
        `thinking`: what the prompt did with the reasoning, as the format reads it."""
        if fmt is not None:
            pieces = [self.token_text(position) for position in range(len(self))]
            self.parts = fmt.parts(pieces, thinking=thinking)
            self.format = fmt.name
            self.thinking = thinking
        return self

    def text(self, start: int = 0, end: int | None = None) -> str:
        """The positions' pieces joined, as text; a redacted position reads as `<redacted>`."""
        out = b""
        for position in range(start, len(self) if end is None else end):
            out += REDACTED.encode() if self.ids[position] is None else self.token_bytes(position)
        return out.decode("utf-8", "replace")

    def known(self, position: int) -> dict[int, float]:
        """Every token whose probability is known here: the top K, and the chosen one."""
        found = dict(zip(self.top_ids[position], self.top_logprobs[position]))
        chosen, logprob = self.ids[position], self.logprobs[position]
        if chosen is not None and logprob is not None:
            found.setdefault(chosen, logprob)
        return found

    def entropy(self, position: int) -> float | None:
        if not self.top_logprobs[position]:
            return None
        return entropy_bits(self.known(position).values())

    def margin(self, position: int) -> float | None:
        if not self.top_logprobs[position]:
            return None
        return margin(self.known(position).values())

    def part(self, position: int) -> str | None:
        for kind, start, end in self.parts:
            if start <= position < end:
                return kind
        return None

    def is_redacted(self, position: int) -> bool:
        return self.ids[position] is None

    def to_json(self) -> dict[str, Any]:
        data = {
            "v": VERSION,
            "k": self.top_k,
            "format": self.format,
            "ids": self.ids,
            "lp": [None if lp is None else round(lp, 5) for lp in self.logprobs],
            "top_ids": self.top_ids,
            "top_lp": [[round(lp, 5) for lp in row] for row in self.top_logprobs],
            "pieces": {str(token): piece for token, piece in self.pieces.items()},
            "parts": [list(part) for part in self.parts],
            "redacted": [list(span) for span in self.redacted],
        }
        if self.thinking is not None:
            data["thinking"] = self.thinking
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TokenStream:
        return cls(
            ids=list(data["ids"]),
            logprobs=list(data["lp"]),
            top_ids=[list(row) for row in data["top_ids"]],
            top_logprobs=[list(row) for row in data["top_lp"]],
            pieces={int(token): piece for token, piece in (data.get("pieces") or {}).items()},
            top_k=int(data.get("k") or 0),
            parts=[(kind, int(start), int(end)) for kind, start, end in data.get("parts") or []],
            redacted=[(int(start), int(end)) for start, end in data.get("redacted") or []],
            format=data.get("format"),
            thinking=data.get("thinking"),
        )


def from_entries(entries: Sequence[dict[str, Any]], *, top_k: int | None = None) -> TokenStream:
    """A stream from llama.cpp's per-token entries: chat `logprobs.content` or `/completion`
    `completion_probabilities`, which share a shape — id, token, bytes, logprob and
    top_logprobs of the same. `bytes` is preferred to `token`, which llama.cpp trims to valid
    UTF-8."""
    pieces: dict[int, Piece] = {}

    def remember(entry: dict[str, Any]) -> int | None:
        token_id = entry.get("id")
        if token_id is None:
            return None
        raw = entry.get("bytes")
        piece = _as_piece(bytes(raw)) if isinstance(raw, list) else (entry.get("token") or "")
        # An entry the server merged keeps no piece (`backends.llamacpp.raw.aligned`): the
        # token's own, where another entry shows it, wins.
        if token_id not in pieces or (piece and not pieces[token_id]):
            pieces[token_id] = piece
        return token_id

    ids, logprobs, top_ids, top_logprobs = [], [], [], []
    widest = 0
    for entry in entries:
        ids.append(remember(entry))
        logprobs.append(entry.get("logprob"))
        row = [(remember(alt), alt.get("logprob")) for alt in entry.get("top_logprobs") or []]
        row = [(token, lp) for token, lp in row if token is not None and lp is not None]
        top_ids.append([token for token, _ in row])
        top_logprobs.append([lp for _, lp in row])
        widest = max(widest, len(row))
    return TokenStream(
        ids=ids, logprobs=logprobs, top_ids=top_ids, top_logprobs=top_logprobs, pieces=pieces,
        top_k=top_k if top_k is not None else widest,
    )


def split_logprobs(response: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """A chat response without its logprobs, and the first choice's token entries.

    Taken out before a response is redacted: redaction works on serialized JSON, where the
    generic `token: …` pattern matches the logprobs' own `"token"` key and masks ordinary
    tokens, while a secret split across tokens slips past it. Token data is redacted by
    `redact_stream` instead, and stored apart from the response.
    """
    if not isinstance(response, dict):
        return response, None
    choices = response.get("choices")
    if not isinstance(choices, list) or not any(
        isinstance(choice, dict) and "logprobs" in choice for choice in choices
    ):
        return response, None
    entries = None
    stripped = []
    for index, choice in enumerate(choices):
        if isinstance(choice, dict) and "logprobs" in choice:
            choice = dict(choice)
            logprobs = choice.pop("logprobs")
            if index == 0 and isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                entries = logprobs["content"]
        stripped.append(choice)
    return {**response, "choices": stripped}, entries


# --- display ---------------------------------------------------------------------------------


def display_groups(stream: TokenStream) -> list[tuple[int, int, str]]:
    """Positions as (start, end, text) groups whose bytes decode: a character split into byte
    tokens is shown once, for the whole group. Nearly every group is one token."""
    groups: list[tuple[int, int, str]] = []
    start, pending = 0, b""
    for position in range(len(stream)):
        if stream.ids[position] is None:
            if pending:
                groups.append((start, position, pending.decode("utf-8", "replace")))
            groups.append((position, position + 1, REDACTED))
            start, pending = position + 1, b""
            continue
        pending += stream.token_bytes(position)
        try:
            text = pending.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data" and position + 1 < len(stream):
                continue  # the character goes on in the next token
            text = pending.decode("utf-8", "replace")
        groups.append((start, position + 1, text))
        start, pending = position + 1, b""
    if pending:
        groups.append((start, len(stream), pending.decode("utf-8", "replace")))
    return groups


# --- where the sampler could have gone another way ---------------------------------------


#: llama.cpp's default sampler order; only the samplers that shape the distribution are applied.
DEFAULT_ORDER = ("top_k", "typ_p", "top_p", "min_p", "temperature")


@dataclass(frozen=True, slots=True)
class Sampler:
    """The sampling settings a step ran under: the request's own, else the server's defaults."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    order: tuple[str, ...] = DEFAULT_ORDER

    @classmethod
    def of(cls, request: dict[str, Any] | None, defaults: dict[str, Any] | None) -> Sampler:
        request, defaults = request or {}, defaults or {}

        def pick(key: str, fallback: Any) -> Any:
            value = request.get(key)
            if value is None:
                value = defaults.get(key)
            return fallback if value is None else value

        order = defaults.get("samplers")
        return cls(
            temperature=float(pick("temperature", 1.0)),
            top_k=int(pick("top_k", 0) or 0),
            top_p=float(pick("top_p", 1.0)),
            min_p=float(pick("min_p", 0.0)),
            order=tuple(order) if isinstance(order, list) and order else DEFAULT_ORDER,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p,
            "min_p": self.min_p, "order": list(self.order),
        }


def effective(known: dict[int, float], sampler: Sampler) -> dict[int, float]:
    """The probability each known token has of being drawn under `sampler`.

    An approximation, and stated as one: only the top K are known, the unseen tail is taken
    to be sampled never, and the samplers are applied in the server's order to the known
    tokens alone. It is exact where it matters most — for the likely tokens.
    """
    ranked = sorted(((probability(lp), token) for token, lp in known.items()), reverse=True)
    kept = [(p, token) for p, token in ranked if p > 0]
    for name in sampler.order:
        if not kept:
            break
        if name == "top_k" and sampler.top_k > 0:
            kept = kept[: sampler.top_k]
        elif name == "top_p" and sampler.top_p < 1.0:
            total, cut = 0.0, []
            for p, token in kept:
                cut.append((p, token))
                total += p
                if total >= sampler.top_p:
                    break
            kept = cut
        elif name == "min_p" and sampler.min_p > 0:
            floor = sampler.min_p * kept[0][0]
            kept = [(p, token) for p, token in kept if p >= floor]
        elif name == "temperature":
            if sampler.temperature <= 0:
                kept = kept[:1]
            elif sampler.temperature != 1.0:
                kept = [(p ** (1.0 / sampler.temperature), token) for p, token in kept]
    total = sum(p for p, _ in kept)
    return {token: p / total for p, token in kept} if total > 0 else {}


@dataclass(frozen=True, slots=True)
class Choice:
    """A token at a position: its id, its text, the model's probability for it, and the
    probability the step's sampler would have drawn it with."""

    id: int
    text: str
    p: float
    p_sampled: float

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "p": round(self.p, 6), "p_sampled": round(self.p_sampled, 6)}


@dataclass(frozen=True, slots=True)
class Candidate:
    """A position a forking-token probe tests, and the tokens it forces there."""

    position: int
    part: str | None
    chosen: Choice
    alternatives: tuple[Choice, ...]
    #: The sampled probability of the likeliest token other than the chosen one.
    runner_up: float
    entropy: float | None
    #: Picked by a person, rather than ranked.
    picked: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "part": self.part,
            "chosen": self.chosen.to_json(),
            "alternatives": [alt.to_json() for alt in self.alternatives],
            "runner_up": round(self.runner_up, 6),
            "entropy": None if self.entropy is None else round(self.entropy, 4),
            "picked": self.picked,
        }


def _text(stream: TokenStream, token: int) -> str:
    return _piece_bytes(stream.piece(token)).decode("utf-8", "replace")


def candidate_at(
    stream: TokenStream, position: int, sampler: Sampler, *, alternatives: int = 3,
    min_p: float = 0.05, picked: bool = False,
) -> Candidate | None:
    """The position as a probe would test it, or None if it cannot be tested: redacted, or
    with nothing known about the tokens weighed there."""
    chosen = stream.ids[position]
    if chosen is None or not stream.top_ids[position]:
        return None
    known = stream.known(position)
    sampled = effective(known, sampler)
    chosen_text = _text(stream, chosen)
    distinct = []
    for token, lp in sorted(known.items(), key=lambda item: -item[1]):
        if token == chosen:
            continue
        text = _text(stream, token)
        # A token that is the chosen one's beginning mostly re-spells it: " c" then "ount".
        # It is not another way to go, so it is neither probed nor counted as the runner-up.
        if text and chosen_text != text and chosen_text.startswith(text):
            continue
        distinct.append(Choice(token, text, probability(lp), sampled.get(token, 0.0)))
    others = [choice for choice in distinct if picked or choice.p_sampled >= min_p]
    runner_up = max((choice.p_sampled for choice in distinct), default=0.0)
    return Candidate(
        position=position,
        part=stream.part(position),
        chosen=Choice(chosen, chosen_text, probability(stream.logprobs[position]), sampled.get(chosen, 0.0)),
        alternatives=tuple(others[:alternatives]),
        runner_up=runner_up,
        entropy=stream.entropy(position),
        picked=picked,
    )


def candidates(
    stream: TokenStream,
    sampler: Sampler,
    *,
    top: int = 5,
    region: str = "answer",
    min_p: float = 0.05,
    alternatives: int = 3,
) -> list[Candidate]:
    """The `top` positions in `region` whose runner-up the step's sampler would most likely
    have drawn, at least `min_p` — the forks Bigelow et al. test, at their 5% threshold.

    A stream whose format is unknown has no parts, and then every position counts as the
    answer. Markers and redacted positions are never candidates.
    """
    if region not in REGIONS:
        raise ValueError(f"region is one of {', '.join(REGIONS)}, not {region!r}")
    kinds = REGIONS[region]
    found = []
    for position in range(len(stream)):
        part = stream.part(position)
        if stream.parts and part not in kinds:
            continue
        candidate = candidate_at(stream, position, sampler, alternatives=alternatives, min_p=min_p)
        if candidate is None or not candidate.alternatives or candidate.runner_up < min_p:
            continue
        found.append(candidate)
    found.sort(key=lambda c: (-c.runner_up, c.position))
    return found[:top]


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """The model's choice of tool at a step, as a number: at the first token of the first
    call's name, how likely each tool offered was under the step's sampler. Exact for this
    prompt, from one scoring; twenty samples would estimate it to within twenty points."""

    position: int
    #: The tool the recorded answer called; tools that share a first token are joined by "/".
    chosen: str | None
    #: Each tool's chance of being the one called — "other" for tokens naming none offered.
    chance: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "chosen": self.chosen,
            "chance": {name: round(p, 4) for name, p in sorted(self.chance.items(), key=lambda item: -item[1])},
        }


def tool_choice(
    stream: TokenStream, sampler: Sampler, *, tools: Sequence[str] = (), prefix: str = "",
) -> ToolChoice | None:
    """What the step's tool choice was, from its scored tokens: at the first token of the
    first call's name — after `prefix`, what the format writes before a name — each known
    token's sampled probability, summed by the offered tool its text begins. None for an
    answer with no call, a stream whose format is unknown, or a position nothing is known at.

    A choice near even is one a few words of the skill can move, as a screen then shows; one
    near certain needs another kind of change. On meeting-action-items the write path — a
    script through execute_code, or write_file — was the task's largest effect, decided at
    this one position."""
    part = next(((start, end) for kind, start, end in stream.parts if kind == TOOL_CALL), None)
    if part is None:
        return None
    start, end = part
    consumed, position = "", None
    for i in range(start, end):
        piece = stream.token_text(i)
        if len(consumed) + len(piece) > len(prefix):
            position = i
            break
        consumed += piece
    if position is None or stream.ids[position] is None or not stream.top_ids[position]:
        return None
    head = prefix[len(consumed):]  # what is left of the prefix, if this token carries it too

    def tool_of(text: str, chosen: bool = False) -> str | None:
        if head and text.startswith(head):
            text = text[len(head):]
        text = text.strip()
        if not text:
            return None
        matched = sorted(name for name in tools if name.startswith(text))
        if matched:
            return "/".join(matched)
        # The name the model wrote, when no tool offered begins with it — or none is known.
        return text if chosen or not tools else None

    sampled = effective(stream.known(position), sampler)
    chance: dict[str, float] = {}
    for token, p in sampled.items():
        key = tool_of(stream.token_text_of(token), chosen=token == stream.ids[position]) or "other"
        chance[key] = chance.get(key, 0.0) + p
    return ToolChoice(position=position, chosen=tool_of(stream.token_text(position), chosen=True), chance=chance)


# --- redaction ------------------------------------------------------------------------------


def redact_stream(stream: TokenStream, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> TokenStream:
    """The stream with every token that overlaps a secret masked — its id, piece and the
    alternatives weighed there — so a secret split across tokens is not stored piece by piece.
    The redaction patterns are the ones storage applies to text."""
    groups = display_groups(stream)
    text, spans_by_group = "", []
    for start, end, piece in groups:
        spans_by_group.append((len(text), len(text) + len(piece), start, end))
        text += piece
    masked: set[int] = set()
    for secret_start, secret_end in redact_spans(text, patterns):
        for char_start, char_end, start, end in spans_by_group:
            if char_start < secret_end and secret_start < char_end:
                masked.update(range(start, end))
    if not masked:
        return stream
    ids = [None if position in masked else token for position, token in enumerate(stream.ids)]
    top_ids = [[] if position in masked else row for position, row in enumerate(stream.top_ids)]
    top_logprobs = [[] if position in masked else row for position, row in enumerate(stream.top_logprobs)]
    used = {token for token in ids if token is not None} | {token for row in top_ids for token in row}
    ranges: list[tuple[int, int]] = []
    for position in sorted(masked):
        if ranges and ranges[-1][1] == position:
            ranges[-1] = (ranges[-1][0], position + 1)
        else:
            ranges.append((position, position + 1))
    return TokenStream(
        ids=ids,
        logprobs=list(stream.logprobs),
        top_ids=top_ids,
        top_logprobs=top_logprobs,
        pieces={token: piece for token, piece in stream.pieces.items() if token in used},
        top_k=stream.top_k,
        parts=list(stream.parts),
        redacted=sorted(set(stream.redacted) | set(ranges)),
        format=stream.format,
        thinking=stream.thinking,
    )
