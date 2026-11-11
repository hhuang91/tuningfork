"""Skill wordings edited in the browser, kept in the recordings folder.

A variant is saved as `<home>/variants/<skill>/<name>.md`, and only when work that uses it
starts: a draft abandoned in the editor leaves nothing behind. A name holds one text, which
every experiment run with it names by that file: work given another text under it is refused,
when it is planned and again as it starts (`refuse_taken`, `keep`). Never inside a skill folder —
every run is given a copy of that whole folder, and a better wording beside the one under
test is an answer key the model can read (see `runner.lookalike_skill_files`).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tuningfork.ops.common import OperationError

#: A skill's or a variant's name: one path component, nothing a shell or a URL mangles.
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
FOLDER = "variants"
#: A skill file is text a model reads; far beyond this is a mistake, not a wording.
MAX_BYTES = 1024 * 1024


def root(home: Path | str) -> Path:
    return Path(home) / FOLDER


def check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value) or value.endswith((".", ".md")):
        raise OperationError(
            f"{what} {value!r} is not a usable name",
            hint="lowercase letters, digits, and - _ . — starting with a letter or digit",
            field="variant_name" if what == "a variant name" else None,
        )
    return value


def path_for(home: Path | str, skill: str, name: str) -> Path:
    """Where a variant lives; the names are checked, so it is always inside the folder."""
    return root(home) / check_name(skill, "a skill name") / f"{check_name(name, 'a variant name')}.md"


@dataclass(frozen=True, slots=True)
class Variant:
    skill: str
    name: str
    path: Path
    sha256: str
    size: int
    modified: float

    def to_json(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "modified": self.modified,
        }


def _variant(skill: str, path: Path) -> Variant:
    data = path.read_bytes()
    return Variant(
        skill=skill,
        name=path.stem,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        modified=path.stat().st_mtime,
    )


def list_variants(home: Path | str, skill: str | None = None) -> list[Variant]:
    """Saved variants, newest first."""
    found: list[Variant] = []
    base = root(home)
    if not base.is_dir():
        return found
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or not NAME.fullmatch(folder.name):
            continue
        if skill is not None and folder.name != skill:
            continue
        for path in folder.glob("*.md"):
            if path.is_file() and NAME.fullmatch(path.stem):
                found.append(_variant(folder.name, path))
    return sorted(found, key=lambda variant: variant.modified, reverse=True)


def read_variant(home: Path | str, skill: str, name: str) -> str:
    path = path_for(home, skill, name)
    if not path.is_file():
        raise OperationError(f"no variant {name!r} of {skill}")
    # As saved, line endings and all: the text `save_variant` takes back under its name.
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def save_variant(home: Path | str, skill: str, name: str, text: str) -> Path:
    """Write a variant, never over another file: two saves under one name at once leave the
    first one's (`_write_new`). A name already holding other text is refused rather than
    overwritten: an earlier experiment's record points at that file. Only the same bytes are
    the same text: read as Python reads a text, a line ended with CRLF ends with LF."""
    data = text.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise OperationError("the variant is larger than 1 MiB", field="variant_text")
    path = path_for(home, skill, name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_new(path, data)
            return path
        except FileExistsError:
            pass  # another save made it since it was looked for: two A/Bs started at once
    if path.read_bytes() == data:
        return path
    raise _taken(skill, name)


def _write_new(path: Path, data: bytes) -> None:
    """Create `path` holding `data`, failing if it exists: written beside it and linked into
    place, there whole or not at all, as `refine._write_once` places a record (without its
    fsync); where the file system makes no links, written in place, still never over another."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        except OSError:
            with open(path, "xb") as handle:  # a file system without links: in place, never over another
                handle.write(data)
    finally:
        temporary.unlink(missing_ok=True)


def _taken(skill: str, name: str) -> OperationError:
    return OperationError(
        f"a variant named {name!r} of {skill} already exists, with other text",
        hint="choose another name: experiments already run name that file",
        field="variant_name",
    )


def holds(path: Path, text: str) -> bool:
    """Whether the variant file at `path` holds `text`, every line's ending read as one, as an A/B
    at a step digests the texts it records (`text_sha256`). A browser gives a text box's text with
    LF line endings, whatever the file's; a file not UTF-8 holds none."""
    from tuningfork.experiment.skill import text_sha256

    try:
        held = path.read_bytes().decode("utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return False
    return text_sha256(held) == text_sha256(text)


def refuse_taken(path: Path, text: str) -> None:
    """Refused where `path`, where a variant given as text is saved as the work that tests it
    starts, holds another text: a name holds one text, which every record of it names."""
    if path.exists() and not holds(path, text):
        raise _taken(path.parent.name, path.stem)


def keep(home: Path | str, path: Path, text: str) -> Path:
    """A variant given as text, as the work that tests it starts: the file at `path` where it
    holds the text, its lines ended as they may be; else saved now, and refused where the
    name took another text since the work was planned."""
    if holds(path, text):
        return path
    try:
        return save_variant(home, path.parent.name, path.stem, text)
    except OperationError:
        if holds(path, text):  # another save made it meanwhile, its lines ended otherwise
            return path
        raise


def skill_name(run: Any, skill_file: Path | None) -> str:
    """The skill a run was given, by its folder's name — where its variants are kept."""
    candidates = []
    if skill_file is not None:
        candidates.append(skill_file.parent.name)
    if run.task_id:
        candidates.append(run.task_id.split("/", 1)[0])
    for candidate in candidates:
        cleaned = re.sub(r"[^a-z0-9_.-]+", "-", candidate.lower()).strip("-.")
        if NAME.fullmatch(cleaned or ""):
            return cleaned
    return "skill"
