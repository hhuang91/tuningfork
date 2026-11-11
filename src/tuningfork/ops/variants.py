"""Skill wordings edited in the browser, kept in the recordings folder.

A variant is saved as `<home>/variants/<skill>/<name>.md`, and only when work that uses it
starts: a draft abandoned in the editor leaves nothing behind. Never inside a skill folder —
every run is given a copy of that whole folder, and a better wording beside the one under
test is an answer key the model can read (see `runner.lookalike_skill_files`).
"""

from __future__ import annotations

import hashlib
import os
import re
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
    """Write a variant, atomically. A name already holding other text is refused rather
    than overwritten: an earlier experiment's record points at that file. Only the same bytes
    are the same text: read as Python reads a text, a line ended with CRLF ends with LF."""
    data = text.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise OperationError("the variant is larger than 1 MiB", field="variant_text")
    path = path_for(home, skill, name)
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise OperationError(
            f"a variant named {name!r} of {skill} already exists, with other text",
            hint="choose another name: experiments already run name that file",
            field="variant_name",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)
    return path


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
