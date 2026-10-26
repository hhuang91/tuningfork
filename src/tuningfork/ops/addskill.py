"""Skills a person adds: `tuningfork add-skill`, `tuningfork forget-skill`, the Skills page's
Add a skill….

A skill is a folder its person keeps — where their harness reads it, or in a repository — and an
approved edit is written back to that folder (`ops.approve`). It is added one of three ways:

- **a folder on this machine**: remembered where it is, nothing copied, so approvals write to it;
- **uploaded**, or copied from a folder (`--copy`): its files kept in the recordings folder,
  `<home>/skills/<name>/` — a copy of tuningfork's own, which approvals write to, never back to
  where it came from;
- **from a run**: the copy the run kept of it, else the SKILL.md its model read, given the
  frontmatter its harness did not show — a copy of tuningfork's own too.

Each is a record, `<home>/skills/<name>.json`, beside the copy where there is one. The Skills
page then lists it with that folder as one of its own (`ops.skills`), and a scenario can name it.
Each is planned before anything is written, and the plan is all that is written. Forgetting one
drops its record; a copy is moved aside whole, never deleted — to `<home>/skills/.forgotten/`,
with what an approval kept beside it — and refused while a scenario runs from it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Store

ROOT = "skills"
FORGOTTEN = ".forgotten"
#: Never copied: a repository's history, caches, environments.
IGNORED = frozenset({".git", "__pycache__", "node_modules", ".venv", ".DS_Store"})


def root(home: Path | str) -> Path:
    return Path(home) / ROOT


def _limits() -> tuple[int, int, int]:
    """What a copy may hold: a scenario's inputs' limits — files, bytes each, bytes in all."""
    from tuningfork.ops.scenarios import MAX_FILE, MAX_FILES, MAX_TOTAL

    return MAX_FILES, MAX_FILE, MAX_TOTAL


@dataclass(slots=True)
class Added:
    """A skill a person added, as its record says."""

    name: str
    #: folder: remembered where it is | copy: tuningfork's own, in the recordings folder.
    kind: str
    #: Its own folder: the one named, or the copy.
    path: Path
    #: Where it came from: `{"folder": …}`, `{"upload": …}` or `{"run": …, "skill": …, "how": …}`.
    origin: dict[str, Any]
    added_at: float

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import windows_form

        return {"name": self.name, "kind": self.kind, "path": str(self.path), "windows": windows_form(self.path),
                "origin": self.origin, "added_at": self.added_at}


def added(home: Path | str) -> list[Added]:
    """Every skill added, by name: those whose record can be read."""
    found = []
    base = root(home)
    for record in sorted(base.glob("*.json")) if base.is_dir() else ():
        each = _read_record(record)
        if each is not None:
            found.append(each)
    return found


def added_skill(home: Path | str, name: str) -> Added | None:
    record = root(home) / f"{name}.json"
    return _read_record(record) if record.is_file() else None


def _read_record(record: Path) -> Added | None:
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        return Added(name=str(data["name"]), kind=str(data["kind"]), path=Path(data["path"]),
                     origin=dict(data.get("origin") or {}), added_at=float(data.get("added_at") or 0))
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --- a plan, then what it writes ---------------------------------------------------------------------


@dataclass(slots=True)
class AddPlan:
    """A skill to add, checked and shown before anything is written."""

    name: str
    kind: str
    path: Path
    origin: dict[str, Any]
    #: A copy's files, by their path in it.
    files: dict[str, bytes] = field(default_factory=dict)
    #: What was left out of a copy, and why: a link, a cache, past the limits.
    left_out: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import windows_form

        return {
            "name": self.name, "kind": self.kind, "path": str(self.path), "windows": windows_form(self.path),
            "origin": self.origin, "files": sorted(self.files), "bytes": sum(len(data) for data in self.files.values()),
            "left_out": self.left_out, "warnings": self.warnings, "notes": self.notes,
        }


def _check_name(home: Path | str, name: Any, *, field_name: str = "name") -> str:
    """A name a new skill can take: a skill's folder name, as runs and variants name it, and
    not one added already."""
    from tuningfork.ops.variants import NAME

    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise OperationError(f"{name!r} is not a skill's name: lowercase letters, digits, '.', '-' and '_', "
                             "starting with a letter or digit", field=field_name)
    base = root(home)
    if (base / f"{name}.json").exists():
        raise OperationError(f"a skill called {name!r} is added already: forget it first to add another",
                             code="conflict", field=field_name)
    if (base / name).exists():
        raise OperationError(f"{base / name} is there already, holding no skill tuningfork added: move it away "
                             "to use the name", code="conflict", field=field_name)
    return name


def _known_note(store: Store, name: str, path: Path | None = None) -> list[str]:
    """Where a skill of that name is known already: the added one will be one of its folders —
    or, a folder a spec names already, stays one when the spec goes."""
    from tuningfork.ops.skills import skills

    known = next((summary for summary in skills(store) if summary.name == name), None)
    if known is None:
        return []
    if path is not None and str(path) in known.folders:
        return [f"{name} lists this folder already, as a spec names it: added, it stays listed whatever the spec does"]
    return [f"a skill called {name} is known already: this becomes one more of its own folders, beside the "
            "others the Skills page lists"]


def _skill_folder(folder: str | Path) -> Path:
    """A skill's folder named on this machine, as tuningfork will open it: one holding a SKILL.md."""
    from tuningfork.paths import resolve_path

    found = resolve_path(str(folder))
    if not found.path.is_dir():
        given = f" (given as {found.translated_from})" if found.translated_from else ""
        raise OperationError(f"{found.path}{given} does not exist, as tuningfork sees it", field="path")
    if not (found.path / "SKILL.md").is_file():
        raise OperationError(f"{found.path} holds no SKILL.md", field="path")
    return found.path


def prepare_folder(store: Store, folder: str | Path) -> AddPlan:
    """A folder on this machine, remembered where it is: nothing copied, so an approved edit is
    written to it. Named by the folder, as the Skills page names a skill."""
    from tuningfork.ops.skills import _clean

    path = _skill_folder(folder)
    base = root(store.home).resolve()
    if path.resolve() == base or base in path.resolve().parents:
        raise OperationError(f"{path} is a copy tuningfork keeps already", field="path")
    for each in added(store.home):
        if each.kind == "folder" and each.path.resolve() == path.resolve():
            raise OperationError(f"{path} is added already, as {each.name}", code="conflict", field="path")
    name = _clean(path.name)
    if name is None:
        raise OperationError(f"{path.name!r} cannot name a skill: rename its folder", field="path")
    _check_name(store.home, name, field_name="path")
    return AddPlan(name=name, kind="folder", path=path, origin={"folder": str(path)},
                   notes=[*_known_note(store, name, path), "an approved edit is written to this folder"])


def prepare_upload(store: Store, name: str, files: dict[str, bytes]) -> AddPlan:
    """A skill's files sent from elsewhere — a browser's folder upload — kept as tuningfork's own
    copy, `<home>/skills/<name>/`."""
    name = _check_name(store.home, name)
    kept, left_out = _checked_files(files)
    return AddPlan(name=name, kind="copy", path=root(store.home) / name, origin={"upload": {"files": len(kept)}},
                   files=kept, left_out=left_out, warnings=_copy_warnings(left_out), notes=[
                       *_known_note(store, name),
                       "a copy of tuningfork's own: an approved edit is written to it, never back to where it came from"])


def prepare_copy(store: Store, folder: str | Path, name: str | None = None) -> AddPlan:
    """A folder on this machine copied in, as an upload is: `add-skill --copy`."""
    from tuningfork.ops.skills import _clean

    path = _skill_folder(folder)
    files, left_out = _read_folder(path)
    plan = prepare_upload(store, name or _clean(path.name) or "", files)
    plan.origin = {"upload": {"files": len(plan.files), "from": str(path)}}
    plan.left_out = [*left_out, *plan.left_out]
    plan.warnings = _copy_warnings(plan.left_out)
    return plan


def prepare_from_run(store: Store, reference: str, *, skill: str | None = None, name: str | None = None) -> AddPlan:
    """The skill a run had, kept as tuningfork's own copy: the copy the run kept of it — its
    folder whole — else the SKILL.md its model read, given the frontmatter its harness did not
    show. `skill` where the run kept or showed several."""
    from tuningfork.ops.imports import _skill_file, _studied, chat_of, extracted_unmasked
    from tuningfork.ops.skills import _clean

    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}", code="not_found", field="run")
    chat = chat_of(store, run)
    try:
        studied = _studied(run, chat, chat.harness or "generic", skill)
    except OperationError as error:
        # Said as a scenario of the run would say it: here, of the skill to keep.
        if error.code == "choose_skill":
            raise OperationError(f"run {run.id[:8]} has several skills: name the one to keep", hint=error.hint,
                                 code="choose_skill", field="skill") from None
        if error.field == "skill_dir":
            raise OperationError(f"run {run.id[:8]} kept no skill, and its conversation shows none", field="run") from None
        raise
    name = _check_name(store.home, name if name is not None else (_clean(studied.name) or ""))
    if studied.kept is not None and extracted_unmasked(run, studied.kept_name):
        # Taken from its chat before an import was masked: masked as it is read, as the chat is.
        text = store.masking.text(studied.kept.read_text(encoding="utf-8", errors="replace")).value
        files, left_out = {"SKILL.md": text.encode("utf-8")}, []
        how = "the copy it kept"
    elif studied.kept is not None:
        files, left_out = _read_folder(studied.kept.parent)
        how = "the copy it kept"
    elif studied.found is not None:
        text, made_up = _skill_file(studied.found.text, studied.name, chat)
        files, left_out = {"SKILL.md": text.encode("utf-8")}, []
        how = "the SKILL.md its model read" + (", given frontmatter" if made_up else "")
    else:
        raise OperationError(f"run {run.id[:8]} kept no copy of {studied.name}, and its conversation does not show it",
                             field="skill")
    kept, too_many = _checked_files(files)
    left_out = [*left_out, *too_many]
    notes = [*_known_note(store, name), f"{studied.name} as run {run.id[:8]} had it: {how}",
             "a copy of tuningfork's own: an approved edit is written to it, never back to where it came from"]
    if studied.found is not None and studied.kept is None:
        notes.append("only its SKILL.md: the rest of its folder — scripts, templates, references — the run never showed")
    return AddPlan(name=name, kind="copy", path=root(store.home) / name,
                   origin={"run": run.id, "skill": studied.name, "how": how},
                   files=kept, left_out=left_out, warnings=_copy_warnings(left_out), notes=notes)


def run_skills(store: Store, reference: str) -> list[dict[str, Any]]:
    """The skills a run can give: each it kept a copy of, then each its conversation shows."""
    from tuningfork.ops.experiments import kept_skills
    from tuningfork.ops.imports import chat_of, skill_names

    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}", code="not_found", field="run")
    chat = chat_of(store, run)
    kept = kept_skills(run)
    return [{"name": name, "kept": name in kept} for name in skill_names(run, chat, chat.harness or "generic")]


def execute_add(store: Store, plan: AddPlan) -> Added:
    """Write what the plan says: a copy's files into a folder of its own, made whole before it
    takes its name, then the record."""
    base = root(store.home)
    base.mkdir(parents=True, exist_ok=True)
    _check_name(store.home, plan.name)
    made = None
    if plan.kind == "copy":
        staging = base / f".incoming-{uuid.uuid4().hex[:12]}"
        try:
            for relative, data in plan.files.items():
                target = staging.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            os.rename(staging, plan.path)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        made = plan.path
    found = Added(name=plan.name, kind=plan.kind, path=plan.path, origin=plan.origin, added_at=time.time())
    try:
        _write_record(base / f"{plan.name}.json", found)
    except BaseException:
        if made is not None:
            shutil.rmtree(made, ignore_errors=True)
        raise
    return found


def _write_record(record: Path, found: Added) -> None:
    staged = record.with_name(f".{record.name}.{uuid.uuid4().hex[:8]}")
    staged.write_text(json.dumps({"name": found.name, "kind": found.kind, "path": str(found.path),
                                  "origin": found.origin, "added_at": found.added_at}, indent=2), encoding="utf-8")
    os.replace(staged, record)


# --- forgetting one ----------------------------------------------------------------------------------


def forget(store: Store, name: str) -> dict[str, Any]:
    """Drop an added skill's record. A folder named stays where it is; a copy is moved aside
    whole — with the changelog and the old versions an approval kept beside it — to
    `<home>/skills/.forgotten/<name>-<when>/`, and refused while a scenario runs from it."""
    from tuningfork.ops.skills import _scenarios, _spec_folder

    found = added_skill(store.home, name)
    if found is None:
        raise OperationError(f"no skill called {name!r} was added", code="not_found", field="name")
    base = root(store.home)
    moved = None
    if found.kind == "copy":
        mine = found.path.resolve()
        using = sorted(scenario for named in _scenarios(store).values() for scenario, spec in named
                       if (folder := _spec_folder(spec)) is not None and folder.resolve() == mine)
        if using:
            raise OperationError(f"scenario {', '.join(using)} runs from this copy: give it another skill folder, or "
                                 "delete it, first", code="conflict", field="name")
        moved = base / FORGOTTEN / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        if moved.exists():  # forgotten twice in one second: added again in between
            moved = moved.with_name(f"{moved.name}-{uuid.uuid4().hex[:6]}")
        moved.mkdir(parents=True)
        for kept in (found.path, base / f"{name}.history", base / f"{name}.changelog.md"):
            if kept.exists():
                shutil.move(str(kept), str(moved / kept.name))
        shutil.move(str(base / f"{name}.json"), str(moved / "record.json"))
    else:
        (base / f"{name}.json").unlink()
    return {"name": name, "kind": found.kind, "path": str(found.path), "moved_to": str(moved) if moved else None}


# --- a copy's files ------------------------------------------------------------------------------------


def _read_folder(folder: Path) -> tuple[dict[str, bytes], list[str]]:
    """A folder's files, by their path in it, and what was left out: links, caches, a
    repository's history."""
    files: dict[str, bytes] = {}
    left_out: list[str] = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if IGNORED.intersection(relative.parts):
            continue
        if path.is_symlink():
            left_out.append(f"{relative.as_posix()}: a link")
            continue
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
    return files, left_out


def _checked_files(files: dict[str, bytes]) -> tuple[dict[str, bytes], list[str]]:
    """A copy's files as they are kept: each path inside the copy, a SKILL.md at its top, and no
    more than a scenario's inputs may hold — what is past the limits left out, and said."""
    most, each_max, total_max = _limits()
    kept: dict[str, bytes] = {}
    left_out: list[str] = []
    total = 0
    for relative in sorted(files):
        clean = _relative(relative)
        if IGNORED.intersection(PurePosixPath(clean).parts):
            continue
        data = files[relative]
        if len(data) > each_max:
            left_out.append(f"{clean}: larger than {each_max // (1024 * 1024)} MiB")
            continue
        if len(kept) >= most or total + len(data) > total_max:
            left_out.append(f"{clean}: past what a copy holds ({most} files, {total_max // (1024 * 1024)} MiB in all)")
            continue
        kept[clean] = data
        total += len(data)
    if "SKILL.md" not in kept:
        raise OperationError("no SKILL.md at the top of what was sent: a skill's folder holds one", field="files")
    return kept, left_out


def _relative(text: str) -> str:
    """A file's path inside a copy, refused where it would land outside it."""
    if not isinstance(text, str):
        raise OperationError("a file's path is text", field="files")
    parts = PurePosixPath(text.replace("\\", "/")).parts
    if not parts or any(part in ("", ".", "..") for part in parts) or text.startswith(("/", "\\")) or ":" in parts[0]:
        raise OperationError(f"{text!r} is not a path inside a skill's folder", field="files")
    return PurePosixPath(*parts).as_posix()


def _copy_warnings(left_out: list[str]) -> list[str]:
    return [f"left out of the copy: {', '.join(left_out[:6])}" + (f" and {len(left_out) - 6} more" if len(left_out) > 6 else "")] if left_out else []
