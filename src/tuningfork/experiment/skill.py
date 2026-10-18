"""Finding a skill's text in a recorded payload, and swapping a variant in its place.

A wording A/B needs two requests that differ in the skill text and in nothing else. That
depends on locating the skill *exactly*: a near-miss that substitutes the wrong span, or
only some of the occurrences, produces a comparison between two things that were never the
ones being compared.

What harnesses actually do: Hermes 0.18.2 inlines a preloaded skill (`-s name`) verbatim —
full file, frontmatter included — into the system prompt, and its `skill_view` tool returns
the file inside a JSON result, newlines and quotes escaped. Claude Code 2.1.281 shows the
body only, frontmatter stripped, in the user turn after its `Skill` tool's result. So the
skill is searched for in its full form first, then trimmed, then as the body alone, each
both literally and JSON-escaped; anything else fails loudly with a diagnosis rather than
guessing.

Only text the harness placed is searched: the system prompt, user turns and tool results.
The model's own earlier turns are history, and history is held fixed — a model that quoted
the skill back wrote that under the original wording.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)

#: Where a text lives in a request: the keys and indices leading to it.
Path = tuple[str | int, ...]


class SkillNotFound(LookupError):
    """The skill text is not in the context at this step, or not in a recognizable form."""


class NothingToCompare(SkillNotFound):
    """The variant changes nothing the model sees at this step: both arms would be one."""


def normalize_newlines(text: str) -> str:
    # A skill checked out with CRLF must still match a payload built with LF.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_frontmatter(text: str) -> tuple[str, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return "", text
    return match.group(0), text[match.end() :]


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_newlines(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Span:
    """One occurrence of the skill in the payload, and the file lines it covers."""

    message_index: int
    role: str
    part_index: int | None  # set when content is a list of parts
    start: int
    end: int
    form: str  # full | full-trimmed | body
    line_start: int
    line_end: int
    #: `json` when the skill sits inside a JSON-encoded tool result, escaped.
    encoding: str = "raw"

    def describe(self) -> str:
        where = f"message {self.message_index} ({self.role})"
        if self.part_index is not None:
            where += f", part {self.part_index}"
        escaped = ", JSON-escaped" if self.encoding == "json" else ""
        return (
            f"{self.form.replace('-', ' ')}{escaped}, lines {self.line_start}-{self.line_end}, "
            f"in {where} at chars {self.start}-{self.end}"
        )


@dataclass(frozen=True, slots=True)
class _Form:
    name: str
    needle: str
    render: Callable[[str], str]  # variant text -> replacement in this same form


@dataclass(frozen=True, slots=True)
class _Needle:
    form: _Form
    encoding: str  # raw | json
    ascii: bool  # json only: whether non-ASCII characters were \u-escaped
    text: str

    def encode(self, replacement: str) -> str:
        if self.encoding == "raw":
            return replacement
        return json.dumps(replacement, ensure_ascii=self.ascii)[1:-1]


def _forms(skill_text: str) -> list[_Form]:
    text = normalize_newlines(skill_text)
    frontmatter, body = split_frontmatter(text)
    trailing = text[len(text.rstrip("\n")) :]

    forms = [
        # Keep the original's trailing newlines, so the text around the span is unchanged.
        _Form("full", text, lambda v: normalize_newlines(v).rstrip("\n") + trailing),
        _Form("full-trimmed", text.strip(), lambda v: normalize_newlines(v).strip()),
    ]
    if frontmatter and body.strip():
        forms.append(
            _Form(
                "body",
                body.strip(),
                lambda v: split_frontmatter(normalize_newlines(v))[1].strip(),
            )
        )
    # A harness may keep the file's own CRLF line endings — Claude Code inlines a skill's body
    # just as the file has it — so each form is looked for that way too, and an edit is
    # swapped in with the same line endings.
    forms += [
        _Form(form.name, form.needle.replace("\n", "\r\n"),
              lambda v, render=form.render: render(v).replace("\n", "\r\n"))
        for form in forms
    ]
    # Distinct needles only, most specific first.
    seen, unique = set(), []
    for form in forms:
        if form.needle and form.needle not in seen:
            seen.add(form.needle)
            unique.append(form)
    return unique


def _needles(skill_text: str) -> list[_Needle]:
    """Every form, literally and as it reads inside a JSON string, most specific first."""
    needles, seen = [], set()
    for form in _forms(skill_text):
        for encoding, ascii_ in (("raw", False), ("json", False), ("json", True)):
            text = (
                form.needle
                if encoding == "raw"
                else json.dumps(form.needle, ensure_ascii=ascii_)[1:-1]
            )
            if text not in seen:
                seen.add(text)
                needles.append(_Needle(form, encoding, ascii_, text))
    return needles


def _fields(request: dict[str, Any]) -> Iterator[tuple[Path, int, str, int | None, str]]:
    """Every text the harness placed, as (path, message index, role, part index, text).

    An Anthropic request keeps its system prompt in a top-level `system` field, a string
    or a list of text blocks, rather than as a message; it is searched as message -1.
    """
    if "system" in request:
        yield from _content_fields(("system",), request["system"], -1, "system", None)
    for index, message in enumerate(request.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") == "assistant":
            continue
        role = str(message.get("role"))
        path: Path = ("messages", index, "content")
        yield from _content_fields(path, message.get("content"), index, role, None)


def _content_fields(
    path: Path, content: Any, index: int, role: str, part: int | None
) -> Iterator[tuple[Path, int, str, int | None, str]]:
    if isinstance(content, str):
        yield path, index, role, part, content
    elif isinstance(content, list):
        for position, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            where = position if part is None else part
            if isinstance(block.get("text"), str):
                yield (*path, position, "text"), index, role, where, block["text"]
            elif block.get("type") == "tool_result":
                # Anthropic nests a tool's output inside the user turn that carries it.
                nested: Path = (*path, position, "content")
                yield from _content_fields(nested, block.get("content"), index, role, where)


def _line_range(skill_text: str, needle: str) -> tuple[int, int]:
    text = normalize_newlines(skill_text)
    offset = text.find(needle)
    if offset < 0:
        return 1, text.count("\n") + 1
    start = text.count("\n", 0, offset) + 1
    return start, start + needle.rstrip("\n").count("\n")


def _find_all(haystack: str, needle: str) -> Iterator[int]:
    position = haystack.find(needle)
    while position >= 0:
        yield position
        position = haystack.find(needle, position + len(needle))


def _occurrences(request: dict[str, Any], skill_text: str) -> list[tuple[Path, Span, _Needle]]:
    """Every occurrence, in whichever form it takes where it is found.

    One copy can be the full file (preloaded) and another only the body (shown by a tool),
    so every form is searched everywhere; where two forms overlap, the more specific one
    was found first and is kept.
    """
    fields = list(_fields(request))
    taken: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    found: list[tuple[Path, Span, _Needle]] = []
    for needle in _needles(skill_text):
        line_start, line_end = _line_range(skill_text, normalize_newlines(needle.form.needle))
        for path, index, role, part, text in fields:
            for start in _find_all(text, needle.text):
                end = start + len(needle.text)
                if any(start < last and first < end for first, last in taken[path]):
                    continue
                taken[path].append((start, end))
                span = Span(
                    index, role, part, start, end,
                    needle.form.name, line_start, line_end, needle.encoding,
                )
                found.append((path, span, needle))
    found.sort(key=lambda item: (item[1].message_index, item[0], item[1].start))
    return found


def locate(request: dict[str, Any], skill_text: str) -> list[Span]:
    """Every occurrence of the skill in the request, each in the most specific form it takes."""
    return [span for _, span, _ in _occurrences(request, skill_text)]


def diagnose(request: dict[str, Any], skill_text: str) -> str:
    """Explain why the skill could not be found, pointing at the likely cause."""
    _, body = split_frontmatter(normalize_newlines(skill_text))
    probe = next((line.strip() for line in body.splitlines() if line.strip()), "")
    if probe:
        for _, index, role, _, text in _fields(request):
            position = text.find(probe)
            if position >= 0:
                return (
                    f"The skill's first line ({probe!r}) appears in message {index} "
                    f"({role}) at char {position}, but the full text does not match. Either "
                    "the file changed after this run was recorded, or the harness reformatted "
                    "it when inlining. Pass the exact text that was loaded with --original."
                )
    return (
        "No part of the skill appears in the context at this step. A preloaded skill is "
        "present from step 0; a skill loaded by a tool call only appears after that call's "
        "result. Check that the skill was actually in play at this checkpoint."
    )


def unapplied(original_text: str, variant_text: str, spans: list[Span]) -> list[str]:
    """Parts of the edit the model will not see at this step, so the A/B cannot test them."""
    if not spans or any(span.form != "body" for span in spans):
        return []
    before, _ = split_frontmatter(normalize_newlines(original_text))
    after, _ = split_frontmatter(normalize_newlines(variant_text))
    if before == after:
        return []
    return [
        "the variant's frontmatter differs, but at this step the model sees only the skill's "
        "body, so that part of the edit is not being tested (a harness may also list each "
        "skill's description elsewhere, e.g. in a tool's description; that is not swapped)"
    ]


def substitute(
    request: dict[str, Any], original_text: str, variant_text: str
) -> tuple[dict[str, Any], list[Span]]:
    """A copy of `request` with every occurrence of the original skill replaced.

    Every occurrence, not the first: if the skill appears twice (preloaded, then viewed
    again through a tool), replacing one would leave the model reading both wordings.
    Raises `NothingToCompare` if the result is the request unchanged.
    """
    found = _occurrences(request, original_text)
    if not found:
        raise SkillNotFound(diagnose(request, original_text))

    changed = copy.deepcopy(request)
    by_path: dict[Path, list[tuple[Span, _Needle]]] = defaultdict(list)
    for path, span, needle in found:
        by_path[path].append((span, needle))
    for path, items in by_path.items():
        text = _get(changed, path)
        # From the end, so earlier offsets stay valid.
        for span, needle in sorted(items, key=lambda item: item[0].start, reverse=True):
            replacement = needle.encode(needle.form.render(variant_text))
            text = text[: span.start] + replacement + text[span.end :]
        _set(changed, path, text)

    spans = [span for _, span, _ in found]
    if changed == request:
        detail = " ".join(unapplied(original_text, variant_text, spans)) or (
            "the variant's text, in the form the model sees it here, is identical"
        )
        raise NothingToCompare(
            f"Both arms would send the same request: {detail}. Nothing would be measured."
        )
    return changed, spans


def swapped(
    original: dict[str, Any], variant: dict[str, Any], spans: list[dict[str, Any]]
) -> tuple[str, str, dict[str, Any]] | None:
    """The skill texts two requests differ by, as the model read each — an A/B's arms, the
    variant put where `spans` (as `substitute` recorded them) found the original: its inverse.

    Exact or nothing. A text that differs only inside one recorded span gives the original's
    text there and what took its place, with that span; so do two or more copies in one text,
    each swapped for the same text (`_swapped_together`). None when no text does — the arms
    were sent the same texts, or what differs is not one swap.
    """
    before_fields, after_fields = list(_fields(original)), list(_fields(variant))
    if len(before_fields) != len(after_fields):
        return None
    for (path, index, _, part, before), (other, _, _, _, after) in zip(before_fields, after_fields):
        if path != other or before == after:
            continue
        head = len(os.path.commonprefix([before, after]))
        room = min(len(before), len(after)) - head
        tail = len(os.path.commonprefix([before[::-1], after[::-1]])[:room])
        here = [span for span in spans if span.get("message_index") == index and span.get("part_index") == part]
        for span in here:
            start, end = span.get("start"), span.get("end")
            # Everything that differs lies inside the span: the rest is the same in both.
            if not (isinstance(start, int) and isinstance(end, int)
                    and 0 <= start <= head and len(before) - tail <= end <= len(before)):
                continue
            texts = [before[start:end], after[start : len(after) - (len(before) - end)]]
            if span.get("encoding") == "json":
                try:
                    texts = [json.loads(f'"{text}"') for text in texts]
                except ValueError:
                    continue
            return normalize_newlines(texts[0]), normalize_newlines(texts[1]), span
        together = _swapped_together(before, after, here)
        if together is not None:
            return together
    return None


def _swapped_together(
    before: str, after: str, spans: list[dict[str, Any]]
) -> tuple[str, str, dict[str, Any]] | None:
    """Two or more copies in one text, every one swapped for the same text: what differs runs
    from the first copy to the last, so no one span holds it.

    Copies of one form and encoding hold one text and took one text's place, and the text
    between them is unchanged: the replacement's length follows from the two texts' lengths,
    and the text rebuilt with it must be the variant's exactly. Copies of mixed forms give
    None.
    """
    spans = sorted(spans, key=lambda span: span.get("start") if isinstance(span.get("start"), int) else -1)
    if len(spans) < 2 or len({(span.get("form"), span.get("encoding")) for span in spans}) != 1:
        return None
    bounds = [(span.get("start"), span.get("end")) for span in spans]
    if not all(isinstance(start, int) and isinstance(end, int) and 0 <= start <= end for start, end in bounds):
        return None
    if any(bounds[i][1] > bounds[i + 1][0] for i in range(len(bounds) - 1)) or bounds[-1][1] > len(before):
        return None
    copies = {before[start:end] for start, end in bounds}
    size, left = divmod(len(after) - len(before) + sum(end - start for start, end in bounds), len(bounds))
    if len(copies) != 1 or left or size < 0:
        return None
    replacement = after[bounds[0][0] : bounds[0][0] + size]
    rebuilt, at = [], 0
    for start, end in bounds:
        rebuilt += [before[at:start], replacement]
        at = end
    if "".join(rebuilt) + before[at:] != after:
        return None
    texts = [copies.pop(), replacement]
    if spans[0].get("encoding") == "json":
        try:
            texts = [json.loads(f'"{text}"') for text in texts]
        except ValueError:
            return None
    return normalize_newlines(texts[0]), normalize_newlines(texts[1]), spans[0]


def _get(obj: Any, path: Path) -> Any:
    for key in path:
        obj = obj[key]
    return obj


def _set(obj: Any, path: Path, value: Any) -> None:
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value
