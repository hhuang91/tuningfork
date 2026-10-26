"""A skill's text, edited and compared line by line.

A proposed change to a skill is a few small edits, each resolved to an exact replacement
before it is applied. An edit names a line by its number, as show_skill prints them —
`{line, new}`, the line as it should read after the change; `{line, through, new}` for
several; `{after, insert}`, lines added after one — or copies exact text found once in the
skill, `{old, new}`. Line numbers do not drift here: a refinement's copy of the skill never
changes under an edit, every edit in a proposal names the text as it was shown, and a
number is read off show_skill's output rather than counted. What changed is then shown as a
line diff: the same JSON for the CLI, the web UI and an agent.

What an agent or a proposing model reads of a skill is masked (`Masking.shown`), its lines
keeping their numbers, while the skill's file keeps what it holds: an edit is resolved against
the file, and one whose new text holds the mask more often than the text it replaces is
refused: a key the skill holds would be lost to `<redacted>` in its file.
"""

from __future__ import annotations

import difflib
from bisect import bisect_right
from typing import Any, Sequence

from tuningfork.config import REDACTED
from tuningfork.ops.common import OperationError

#: The shapes an edit takes, as a refusal names them.
SHAPES = (
    '{"line": <its number, as show_skill prints it>, "new": <that line as it should read>}; '
    '{"line": <first>, "through": <last>, "new": <those lines as they should read>}; '
    '{"after": <a line number>, "insert": <the lines to add after it>}; or '
    '{"old": <text copied exactly from the skill>, "new": <its replacement>}'
)
SHORT_SHAPES = "{line, new}, {line, through, new}, {after, insert} or {old, new}"
_KEYS = {"old", "new", "line", "through", "after", "insert"}


def line_diff(before: str, after: str, *, context: int = 2) -> dict[str, Any]:
    """What changed from `before` to `after`, line by line.

    Each hunk lists its lines with `op` " " (unchanged), "-" (removed) or "+" (added), and
    their numbers in the old and new text. `changed` is the larger of the lines added and
    removed: a line reworded counts once.
    """
    old, new = before.splitlines(), after.splitlines()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    hunks = []
    for group in matcher.get_grouped_opcodes(context):
        lines: list[dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                lines.extend(
                    {"op": " ", "old": i1 + k + 1, "new": j1 + k + 1, "text": old[i1 + k]} for k in range(i2 - i1)
                )
                continue
            lines.extend({"op": "-", "old": k + 1, "text": old[k]} for k in range(i1, i2))
            lines.extend({"op": "+", "new": k + 1, "text": new[k]} for k in range(j1, j2))
        hunks.append({"old_start": group[0][1] + 1, "new_start": group[0][3] + 1, "lines": lines})
    opcodes = matcher.get_opcodes()
    removed = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("replace", "delete"))
    added = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("replace", "insert"))
    return {"hunks": hunks, "added": added, "removed": removed, "changed": max(added, removed)}


def apply_edits(text: str, edits: Sequence[Any]) -> str:
    """`text` with every edit applied: each resolved against `text` as it stands, then
    spliced in from the bottom up, so that none shifts another."""
    return apply_resolved(text, resolve_edits(text, edits))


def apply_resolved(text: str, resolved: Sequence[dict[str, Any]]) -> str:
    """`text` with each resolved edit's exact splice made, the last in the text first."""
    for found in sorted(resolved, key=lambda each: (each["start"], each["end"]), reverse=True):
        text = text[:found["start"]] + found["replacement"] + text[found["end"]:]
    return text


def resolve_edits(text: str, edits: Sequence[Any]) -> list[dict[str, Any]]:
    """Each edit as an exact replacement of `text`: `old`, the text it changes (empty for
    lines added), `new`, what takes its place, `lines`, the lines it names, `given`, the edit
    as it was sent — and `start`, `end` and `replacement`, the exact splice.

    Every edit is located in `text` as it stands, since a proposal's edits all name the text
    as show_skill printed it; two that touch the same lines are refused. A text written with
    Windows line endings takes edits written with plain ones.
    """
    if not isinstance(edits, (list, tuple)) or not edits:
        raise OperationError(
            f"give at least one edit: {SHORT_SHAPES}", hint=f"each edit is {SHAPES}", code="bad_params", field="edits",
        )
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    resolved = []
    for number, edit in enumerate(edits, 1):
        keys = set(edit) if isinstance(edit, dict) else set()
        if keys and "old" in keys and keys <= {"old", "new"}:
            found = _by_text(text, starts, number, edit, newline)
        elif keys and "line" in keys and keys <= {"line", "through", "new"}:
            found = _by_line(text, lines, starts, number, edit, newline)
        elif keys and "after" in keys and keys <= {"after", "insert"}:
            found = _after_line(lines, starts, number, edit, newline)
        else:
            raise _not_an_edit(text, number, edit)
        if found["new"].count(REDACTED) > found["old"].count(REDACTED):
            raise OperationError(
                f"edit {number} would write {REDACTED} into the skill: what is shown of a skill reads {REDACTED} "
                "where it masks a key the skill holds, and the skill's file keeps the key",
                hint="leave that text as it is: change the lines around it, or only the words you mean with {old, new}",
                code="bad_params", field="edits",
            )
        resolved.append(found)
    ordered = sorted(range(len(resolved)), key=lambda i: (resolved[i]["start"], resolved[i]["end"]))
    for a, b in zip(ordered, ordered[1:]):
        first, second = resolved[a], resolved[b]
        if second["start"] < first["end"] or first["start"] == first["end"] == second["start"] == second["end"]:
            raise OperationError(
                f"edits {min(a, b) + 1} and {max(a, b) + 1} touch the same lines ({first['lines']}; "
                f"{second['lines']}): make them one edit",
                code="bad_params", field="edits",
            )
    return resolved


def _by_text(text: str, starts: list[int], number: int, edit: dict[str, Any], newline: str) -> dict[str, Any]:
    """`{old, new}`: `old` found exactly once — the way an editor that replaces by text
    refuses an ambiguous edit."""
    old, new = edit.get("old"), edit.get("new")
    if not isinstance(old, str):
        raise _not_an_edit(text, number, edit)
    if not isinstance(new, str):
        raise OperationError(
            f"edit {number} has no new: `old` is the text as it is, `new` the same lines as they should read "
            "after the change",
            hint=f"each edit is {SHAPES}", code="bad_params", field="edits",
        )
    new = _unescaped(new)
    if newline == "\r\n":
        old = old.replace("\r\n", "\n").replace("\n", "\r\n")
        new = new.replace("\r\n", "\n").replace("\n", "\r\n")
    if not old.strip():
        raise OperationError(
            f"edit {number} names no text to replace: `old` is the exact text it changes",
            code="bad_params", field="edits",
        )
    if old == new:
        # A 26B sent the table it meant to change, unchanged, nine times in eleven calls — the
        # change left in its rationale — and read "changes nothing" as a fault of the tool:
        # say what `new` is.
        raise OperationError(
            f"edit {number} changes nothing: its `new` is its `old` word for word. `old` is the text as "
            "it is, `new` the same lines as they should read after the change",
            code="bad_params", field="edits",
        )
    found = text.count(old)
    if found == 0:
        # Indentation alone is no difference: an agent that copies the lines but not
        # their leading spaces means the same lines. The edit is applied with the
        # skill's indentation, the new text's own shape kept.
        matched = _ignoring_indentation(text, old)
        if not matched:
            raise OperationError(
                f"edit {number}: its old text is not in the skill",
                hint=(f"what you read of the skill shows {REDACTED} where it masks a key the skill holds: copy "
                      "the text beside it, or name the line by its number"
                      if REDACTED in old and REDACTED not in text else _closest(text, old)),
                code="bad_params", field="edits",
            )
        if len(matched) > 1:
            raise OperationError(
                f"edit {number}: its old text occurs {len(matched)} times once indentation is ignored",
                hint="include more of the text around it, so that it occurs once",
                code="bad_params", field="edits",
            )
        old, new = matched[0][0], _reindented(new, matched[0][1])
    elif found > 1:
        raise OperationError(
            f"edit {number}: its old text occurs {found} times",
            hint="include more of the text around it, so that it occurs once",
            code="bad_params", field="edits",
        )
    start = text.index(old)
    end = start + len(old)
    first, last = bisect_right(starts, start), bisect_right(starts, max(start, end - 1))
    return {"given": edit, "lines": _span(first, last), "old": old, "new": new,
            "start": start, "end": end, "replacement": new}


def _by_line(text: str, lines: list[str], starts: list[int], number: int, edit: dict[str, Any],
             newline: str) -> dict[str, Any]:
    """`{line, new}` or `{line, through, new}`: those lines replaced by `new`, which may hold
    more lines or fewer; an empty `new` removes them."""
    first = _line_number(number, edit, "line", len(lines))
    last = _line_number(number, edit, "through", len(lines)) if "through" in edit else first
    if last < first:
        raise OperationError(
            f"edit {number}: `through` ({last}) is before `line` ({first})", code="bad_params", field="edits",
        )
    several = last > first
    new = edit.get("new")
    if not isinstance(new, str):
        raise OperationError(
            f"edit {number} has no new: {_span(first, last)} as {'they' if several else 'it'} should read after "
            f"the change (\"\" removes {'them' if several else 'it'})",
            hint=f"each edit is {SHAPES}", code="bad_params", field="edits",
        )
    start, end = starts[first - 1], starts[last - 1] + len(lines[last - 1])
    old = text[start:end]
    ending = _ending(old)
    body = old[:len(old) - len(ending)]
    wanted = _plain(new, newline)
    if new != "" and wanted == body:
        raise OperationError(
            f"edit {number} changes nothing: {_span(first, last)} already read{'' if several else 's'} "
            f"{_clip(body)!r}. `new` is the line as it should read after the change",
            code="bad_params", field="edits",
        )
    return {"given": edit, "lines": _span(first, last), "old": body, "new": wanted,
            "start": start, "end": end, "replacement": "" if new == "" else wanted + ending}


def _after_line(lines: list[str], starts: list[int], number: int, edit: dict[str, Any], newline: str) -> dict[str, Any]:
    """`{after, insert}`: lines added after the one named (after 0: at the top)."""
    at = _line_number(number, edit, "after", len(lines), top=True)
    insert = edit.get("insert")
    if not isinstance(insert, str) or not insert.strip():
        raise OperationError(
            f"edit {number} adds nothing: `insert` is the line(s) to add after line {at}",
            hint=f"each edit is {SHAPES}", code="bad_params", field="edits",
        )
    if at == 0:
        position, prefix = 0, ""
    else:
        position = starts[at - 1] + len(lines[at - 1])
        prefix = "" if _ending(lines[at - 1]) else newline
    body = _plain(insert, newline)
    return {"given": edit, "lines": f"after line {at}", "old": "", "new": body,
            "start": position, "end": position, "replacement": prefix + body + newline}


def _line_number(number: int, edit: dict[str, Any], key: str, count: int, *, top: bool = False) -> int:
    value = edit.get(key)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperationError(
            f"edit {number}: `{key}` is a line number, as show_skill prints them",
            hint=f"each edit is {SHAPES}", code="bad_params", field="edits",
        )
    if value < (0 if top else 1) or value > count:
        raise OperationError(
            f"edit {number} names line {value}: the skill has {count} lines, numbered as show_skill prints them"
            + (" (after 0 adds at the top)" if top else ""),
            code="bad_params", field="edits",
        )
    return value


def _not_an_edit(text: str, number: int, edit: Any) -> OperationError:
    hint = f"each edit is {SHAPES}"
    if not isinstance(edit, dict):
        return OperationError(
            f"edit {number} is not an object: it must be {SHORT_SHAPES}", hint=hint, code="bad_params", field="edits",
        )
    unknown = sorted(set(edit) - _KEYS)
    if unknown:
        what = f"edit {number} has a key tuningfork does not take ({', '.join(unknown)}): it must be {SHORT_SHAPES}"
    elif not set(edit) & {"old", "line", "after"}:
        what = f"edit {number} names nothing to change (no line, after or old): it must be {SHORT_SHAPES}"
        if isinstance(edit.get("new"), str) and edit["new"].strip():
            hint += f"; for this new text, {_closest(text, edit['new'])}"
    else:
        what = f"edit {number} mixes shapes: it must be {SHORT_SHAPES}"
    return OperationError(what, hint=hint, code="bad_params", field="edits")


def _span(first: int, last: int) -> str:
    return f"line {first}" if first == last else f"lines {first}–{last}"


def _ending(line: str) -> str:
    return "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")


def _plain(new: str, newline: str) -> str:
    """`new` in the text's own line endings, one trailing line break dropped: a line as a
    model writes it, with or without the break the text keeps for it. Line breaks escaped
    twice — the two characters backslash-n, in a text with no line break of its own — are
    read as line breaks: a 26B sent a two-line insert so, and it went in as one line."""
    plain = _unescaped(new).replace("\r\n", "\n")
    if plain.endswith("\n"):
        plain = plain[:-1]
    return plain.replace("\n", newline)


def _unescaped(text: str) -> str:
    if "\\n" in text and "\n" not in text:
        return text.replace("\\n", "\n")
    return text


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _ignoring_indentation(text: str, old: str) -> list[tuple[str, str]]:
    """Where `old`'s lines occur in `text` once each line's leading and trailing whitespace is
    ignored: each hit as the text's own lines, with the first line's indentation."""
    wanted = [line.strip() for line in old.split("\n")]
    while wanted and not wanted[0]:
        wanted.pop(0)
    while wanted and not wanted[-1]:
        wanted.pop()
    if not wanted:
        return []
    lines = text.split("\n")
    stripped = [line.strip() for line in lines]
    hits = []
    for start in range(len(lines) - len(wanted) + 1):
        if stripped[start:start + len(wanted)] == wanted:
            first = lines[start]
            block = lines[start:start + len(wanted)]
            # The last line's carriage return is its line ending's, not its text: left out of the
            # match, it stays when the lines are replaced, and a Windows file keeps its endings.
            block[-1] = block[-1].removesuffix("\r")
            hits.append(("\n".join(block), first[:len(first) - len(first.lstrip())]))
    return hits


def _reindented(new: str, indent: str) -> str:
    """`new` with its lines' common indentation replaced by `indent`: its own shape, moved."""
    lines = new.split("\n")
    filled = [line for line in lines if line.strip()]
    if not filled:
        return new
    common = min(len(line) - len(line.lstrip()) for line in filled)
    return "\n".join((indent + line[common:]) if line.strip() else line for line in lines)


def _closest(text: str, old: str) -> str:
    wanted = next((line for line in old.splitlines() if line.strip()), old).strip()
    by_stripped: dict[str, str] = {}
    for line in text.split("\n"):
        if line.strip():
            by_stripped.setdefault(line.strip(), line.rstrip("\r"))
    close = difflib.get_close_matches(wanted, list(by_stripped), n=2, cutoff=0.5)
    if not close:
        return "copy `old` from the skill exactly, whitespace and punctuation included"
    return ("the closest lines in the skill, as they stand: " + " | ".join(repr(by_stripped[found]) for found in close)
            + " — copy one exactly; leading spaces alone do not matter")
