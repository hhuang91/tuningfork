"""Scenarios: task specs made and kept in the recordings folder, from the browser.

A scenario is `<home>/scenarios/<name>/evals.json` — the same evals.json convention as any
spec — with its input files in `inputs/` beside it. Its skill folder is a reference to a
folder elsewhere, never copied in — but for one made from a run whose skill has no folder of
its own to be found (`ops.imports.prepare_scenario`), and a task file's folder uploaded with
its skill inside (`prepare_upload`), which keep a copy in `skill/`. A task file kept as a
scenario brings what its check reads from its folder (`$TUNINGFORK_SPEC_DIR`), beside its spec.
Everything a scenario can be told to write stays inside
its own folder: names are checked, input paths may not climb out, and uploads have limits.
A later phase lets a model drive this same API, which is why it is a jail.

Saving checks the spec strictly (`specs.validate_spec`) and refuses to overwrite a version
the saver did not read (an etag, as HTTP has). Deleting moves the folder to `.trash/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tuningfork.ops.common import OperationError

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
FOLDER = "scenarios"
SPEC = "evals.json"
INPUTS = "inputs"
#: The conversation a scenario made from a chat continues (`tuningfork.context`), beside its spec.
CONTEXT = "context.json"
TRASH = ".trash"
#: Limits on what can be uploaded into a scenario.
MAX_FILE = 10 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
MAX_FILES = 200
#: A spec is text; anything near this is a mistake.
MAX_SPEC = 2 * 1024 * 1024
#: Said of a task file kept as a scenario that names no skill folder.
NO_SKILL = "it names no skill folder: choose the skill on the scenario's page, or each run is given none"


def root(home: Path | str) -> Path:
    return Path(home) / FOLDER


def check_name(name: Any) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise OperationError(
            f"{name!r} is not a usable scenario name", field="name",
            hint="lowercase letters, digits, - and _, starting with a letter or digit; at most 63",
        )
    return name


def folder(home: Path | str, name: str) -> Path:
    return root(home) / check_name(name)


def _etag(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def input_path(home: Path | str, name: str, relative: str) -> Path:
    """Where an input file lives: inside the scenario's `inputs/`, always."""
    parts = PurePosixPath(relative.replace("\\", "/")).parts if isinstance(relative, str) else ()
    if not parts or any(part in ("", ".", "..") for part in parts) or relative.startswith(("/", "\\")) or ":" in parts[0]:
        raise OperationError(f"{relative!r} is not a path inside the scenario's inputs", field="path")
    base = (folder(home, name) / INPUTS).resolve()
    target = base.joinpath(*parts).resolve()
    if base != target and base not in target.parents:
        raise OperationError(f"{relative!r} would land outside the scenario's inputs", field="path")
    return target


@dataclass(slots=True)
class InputFile:
    path: str
    size: int

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size}


def inputs(home: Path | str, name: str) -> list[InputFile]:
    base = folder(home, name) / INPUTS
    if not base.is_dir():
        return []
    return sorted(
        (
            InputFile(path.relative_to(base).as_posix(), path.stat().st_size)
            for path in base.rglob("*")
            if path.is_file()
        ),
        key=lambda found: found.path,
    )


# --- checking a spec --------------------------------------------------------------------------


@dataclass(slots=True)
class Checked:
    problems: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Each path the spec names, as tuningfork will open it (`paths.ResolvedPath.to_json`,
    #: plus what was found there): what a page shows beside the field it came from.
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The scenario's own folder, where its evals.json is saved and a relative skill folder is
    #: taken from: `<name>` in it, before the scenario has one.
    folder: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import under_wsl

        return {
            "ok": self.ok, "problems": self.problems, "warnings": self.warnings, "paths": self.paths,
            "folder": self.folder,
            # How a typed path is read here: under WSL, a Windows path is translated.
            "wsl": under_wsl(),
        }


def check_spec(data: Any, *, home: Path | str | None = None, name: str | None = None) -> Checked:
    """Everything wrong with a scenario's spec, and what a person should know about it."""
    from tuningfork.paths import host_path, resolve_path
    from tuningfork.runner import lookalike_skill_files
    from tuningfork.specs import validate_spec

    checked = Checked(problems=[{"where": p.where, "message": p.message} for p in validate_spec(data)])
    named = home is not None and bool(name) and NAME.fullmatch(name or "") is not None
    if home is not None:
        checked.folder = str(folder(home, name)) if named else f"{root(home)}/<name>"
    if not isinstance(data, dict):
        return checked
    top = data.get("tuningfork") if isinstance(data.get("tuningfork"), dict) else {}
    skill_dir = top.get("skill_dir")
    skill: Path | None = None
    if not isinstance(skill_dir, str) or not skill_dir.strip():
        checked.warnings.append("no skill folder: each run is given no skill")
    else:
        # A relative path is resolved against the scenario's own folder, where its evals.json
        # lives, exactly as a run resolves it. A new scenario has no folder until it has a name.
        if not Path(host_path(skill_dir)).is_absolute() and not named:
            where = f"{root(home)}/<name>" if home is not None else "the scenario's folder"
            checked.paths["skill_dir"] = {
                "given": skill_dir, "path": None, "relative_to": where,
                "translated_from": None, "exists": False, "has_skill_md": False,
            }
            checked.warnings.append(
                f"the skill folder is a relative path: it is resolved against the scenario's "
                f"own folder, {where}, so it needs the scenario's name first"
            )
        else:
            found = resolve_path(skill_dir, folder(home, name) if named else None)
            skill = found.path
            has_skill = (skill / "SKILL.md").is_file()
            checked.paths["skill_dir"] = {**found.to_json(), "has_skill_md": has_skill}
            if not has_skill:
                message = (
                    f"{skill} holds no SKILL.md" if skill.is_dir()
                    else f"{skill} does not exist, as tuningfork sees it"
                )
                checked.problems.append({"where": "tuningfork.skill_dir", "message": message})
                skill = None
    lookalikes = lookalike_skill_files(skill) if skill is not None else []
    if lookalikes:
        checked.warnings.append(
            f"the skill folder also holds {', '.join(lookalikes)}: every run is given a copy of "
            "the whole folder, so the model can read it. Keep variants outside the skill folder."
        )
    for index, item in enumerate(data.get("evals") or []):
        if not isinstance(item, dict):
            continue
        options = item.get("tuningfork") if isinstance(item.get("tuningfork"), dict) else {}
        if isinstance(options.get("check"), str) and options["check"].strip():
            checked.warnings.append(
                f"task {item.get('id', index)}: its check runs as a shell command in each run's "
                "workspace, as you — read it as you would a script"
            )
            # What it reads from its task's folder, $TUNINGFORK_SPEC_DIR — this scenario's own.
            missing = [relative for relative in _check_reads(options["check"])
                       if named and not (folder(home, name) / relative).exists()]
            if missing:
                checked.warnings.append(
                    f"task {item.get('id', index)}: its check reads {', '.join(missing)} from the scenario's "
                    "folder ($TUNINGFORK_SPEC_DIR), which holds no such file: put it beside the scenario's evals.json"
                )
        for preloaded in options.get("skills") or []:
            if skill is not None and isinstance(preloaded, str) and preloaded != skill.name:
                checked.warnings.append(
                    f"task {item.get('id', index)} preloads a skill called {preloaded!r}, and the "
                    f"skill folder is {skill.name!r}: the harness finds skills by folder name"
                )
        context = options.get("context")
        if home is not None and name and NAME.fullmatch(name) and isinstance(context, str) and context:
            from tuningfork.specs import SpecError, load_context

            try:
                load_context(folder(home, name), context)
            except SpecError as error:
                checked.problems.append({"where": f"evals[{index}].tuningfork.context", "message": str(error)})
        if home is not None and name:
            for entry in item.get("files") or []:
                source = entry.get("source") if isinstance(entry, dict) else None
                if isinstance(source, str) and source.startswith(f"{INPUTS}/"):
                    try:
                        present = input_path(home, name, source[len(INPUTS) + 1:]).is_file()
                    except OperationError as error:
                        checked.problems.append({"where": f"evals[{index}].files", "message": error.message})
                        continue
                    if not present:
                        # Not a problem yet: a new scenario is saved before files go in.
                        # A batch refuses to start while an input is missing.
                        checked.warnings.append(
                            f"task {item.get('id', index)}: {source} has not been uploaded yet"
                        )
    return checked


# --- reading and writing ----------------------------------------------------------------------


@dataclass(slots=True)
class Scenario:
    name: str
    spec: dict[str, Any]
    etag: str
    inputs: list[InputFile]
    checked: Checked
    modified: float
    #: The conversation it continues, as `context.json` holds it, for a scenario made from a chat.
    context: list[dict[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "etag": self.etag,
            "inputs": [found.to_json() for found in self.inputs],
            "checked": self.checked.to_json(),
            "modified": self.modified,
            "context": self.context,
        }


def read_scenario(home: Path | str, name: str) -> Scenario:
    path = folder(home, name) / SPEC
    if not path.is_file():
        raise OperationError(f"no scenario called {name!r}", code="not_found")
    raw = path.read_bytes()
    try:
        spec = json.loads(raw)
    except ValueError as exc:
        raise OperationError(f"{path} is not valid JSON: {exc}") from None
    try:
        context = read_context(home, name)
    except OperationError:
        context = None  # its problem is the spec check's to say
    return Scenario(
        name=name, spec=spec, etag=_etag(raw), inputs=inputs(home, name),
        checked=check_spec(spec, home=home, name=name), modified=path.stat().st_mtime, context=context,
    )


def spec_path(home: Path | str, name: str) -> Path:
    path = folder(home, name) / SPEC
    if not path.is_file():
        raise OperationError(f"no scenario called {name!r}", code="not_found", field="scenario")
    return path


def save_scenario(home: Path | str, name: str, spec: Any, *, etag: str | None) -> Scenario:
    """Write a scenario's spec, if it is sound and the saver saw the version it replaces.
    `etag` None means "new": refused if the scenario exists already."""
    target = folder(home, name)
    path = target / SPEC
    if path.exists():
        current = _etag(path.read_bytes())
        if etag != current:
            raise OperationError(
                f"scenario {name!r} changed since it was opened" if etag else f"a scenario called {name!r} exists already",
                code="conflict",
                hint="open it again to see the other change" if etag else "choose another name",
            )
    elif etag is not None:
        raise OperationError(f"no scenario called {name!r}", code="not_found")
    spec = _as_opened(spec)
    checked = check_spec(spec, home=home, name=name)
    if not checked.ok:
        listed = "; ".join(f"{p['where']}: {p['message']}" if p["where"] else p["message"] for p in checked.problems[:6])
        raise OperationError(f"the spec has problems: {listed}", code="invalid", detail=json.dumps(checked.problems))
    data = (json.dumps(spec, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > MAX_SPEC:
        raise OperationError("the spec is larger than 2 MiB", code="too_large")
    (target / INPUTS).mkdir(parents=True, exist_ok=True)
    temporary = target / f".{SPEC}.{os.getpid()}.tmp"
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return read_scenario(home, name)


def _as_opened(spec: Any) -> Any:
    """The spec with its skill folder written the way tuningfork opens it: a path pasted
    from Windows is saved translated (`paths.host_path`), so the file says what runs use."""
    from tuningfork.paths import host_path

    top = spec.get("tuningfork") if isinstance(spec, dict) else None
    given = top.get("skill_dir") if isinstance(top, dict) else None
    if not isinstance(given, str) or not given.strip() or host_path(given) == given:
        return spec
    return {**spec, "tuningfork": {**top, "skill_dir": host_path(given)}}


def delete_scenario(home: Path | str, name: str) -> str:
    """Move a scenario to `.trash/`, from where a person can bring it back by hand."""
    target = folder(home, name)
    if not target.is_dir():
        raise OperationError(f"no scenario called {name!r}", code="not_found")
    trash = root(home) / TRASH
    trash.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = trash / f"{name}-{stamp}"
    # Deleted again within the second: a folder of its own beside the first, never inside it,
    # where `shutil.move` puts what is moved onto a folder (found in review).
    again = 1
    while moved.exists():
        again += 1
        moved = trash / f"{name}-{stamp}-{again}"
    shutil.move(str(target), str(moved))
    return str(moved)


def put_input(home: Path | str, name: str, relative: str, data: bytes) -> InputFile:
    if not (folder(home, name) / SPEC).is_file():
        raise OperationError(f"no scenario called {name!r}", code="not_found")
    if len(data) > MAX_FILE:
        raise OperationError(f"{relative} is larger than {MAX_FILE // (1024 * 1024)} MiB", code="too_large")
    target = input_path(home, name, relative)
    existing = inputs(home, name)
    others = [found for found in existing if found.path != PurePosixPath(relative.replace("\\", "/")).as_posix()]
    if len(others) + 1 > MAX_FILES:
        raise OperationError(f"a scenario holds at most {MAX_FILES} input files", code="too_large")
    if sum(found.size for found in others) + len(data) > MAX_TOTAL:
        raise OperationError(f"a scenario's inputs total at most {MAX_TOTAL // (1024 * 1024)} MiB", code="too_large")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return InputFile(target.relative_to(folder(home, name) / INPUTS).as_posix(), len(data))


def read_context(home: Path | str, name: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    """The conversation the scenario continues, as its `context.json` holds it — a list of the
    messages before its prompt, or {"before": [...], "in_turn": [...]} — None without one."""
    path = folder(home, name) / CONTEXT
    if not path.is_file():
        return None
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise OperationError(f"{path} is not valid JSON: {exc}") from None
    return found if isinstance(found, (list, dict)) else None


def put_context(home: Path | str, name: str, messages: Any, *, new: bool = False) -> int:
    """Write the conversation the scenario continues — a list of messages before its prompt, or
    {"before": [...], "in_turn": [...]} — checked as one a harness can continue
    (`specs.check_context`); how many messages it holds. `new`: for a scenario whose spec is
    written next, which names it."""
    from tuningfork.specs import SpecError, check_context

    target = folder(home, name)
    if not new and not (target / SPEC).is_file():
        raise OperationError(f"no scenario called {name!r}", code="not_found")
    try:
        checked = check_context(messages)
    except SpecError as error:
        raise OperationError(str(error), code="invalid", field="context") from None
    data = (json.dumps(checked.to_json(), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > MAX_FILE:
        raise OperationError(f"the conversation is larger than {MAX_FILE // (1024 * 1024)} MiB", code="too_large")
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f".{CONTEXT}.{os.getpid()}.tmp"
    temporary.write_bytes(data)
    os.replace(temporary, target / CONTEXT)
    return checked.messages


def delete_input(home: Path | str, name: str, relative: str) -> None:
    target = input_path(home, name, relative)
    if not target.is_file():
        raise OperationError(f"no input file {relative!r} in {name}", code="not_found")
    target.unlink()


def _check_reads(check: str) -> list[str]:
    """What a shell check reads from its task's folder, `$TUNINGFORK_SPEC_DIR/<path>`: each path
    inside it, in the order the check names them."""
    from tuningfork.ops.imports import SPEC_DIR_PATH

    found: list[str] = []
    for relative in SPEC_DIR_PATH.findall(check):
        parts = PurePosixPath(relative).parts
        if not parts or any(part in ("", ".", "..") for part in parts):
            continue
        if (cleaned := PurePosixPath(*parts).as_posix()) not in found:
            found.append(cleaned)
    return found


# --- a task file kept as a scenario -----------------------------------------------------------


@dataclass(slots=True)
class ImportPlan:
    """A task file to keep as a scenario — one on this machine, or a folder uploaded — shown
    before anything is written: its spec as it will be saved, and the files that go in with it."""

    name: str
    #: Where it comes from: the task file's path, or the folder uploaded and the file in it.
    source: str
    spec: dict[str, Any]
    #: Its input files, by their path in the scenario's `inputs/`.
    inputs: dict[str, bytes] = field(default_factory=dict)
    #: An uploaded folder's skill, copied into the scenario's own folder: by its path there.
    skill: dict[str, bytes] = field(default_factory=dict)
    #: What its check reads from its task's folder (`$TUNINGFORK_SPEC_DIR`), by its path beside
    #: the scenario's evals.json.
    beside: dict[str, bytes] = field(default_factory=dict)
    #: Its skill folder as tuningfork will open it, and whether a SKILL.md is there.
    skill_dir: str | None = None
    has_skill: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        everything = [*self.inputs.values(), *self.skill.values(), *self.beside.values()]
        return {
            "name": self.name, "source": self.source, "skill_name": self.spec.get("skill_name"),
            "tasks": [str(item.get("id")) for item in self.spec.get("evals") or [] if isinstance(item, dict)],
            "skill_dir": self.skill_dir, "has_skill": self.has_skill, "skill_copied": bool(self.skill),
            "inputs": sorted(self.inputs), "beside": sorted(self.beside),
            "files": len(everything), "bytes": sum(len(data) for data in everything),
            "warnings": self.warnings, "notes": self.notes,
        }


class _Beside:
    """The files beside a task file, wherever they are: in its folder on this machine, or in a
    folder uploaded — by their paths in it, the task file's own folder at `base` — which keeps
    the paths taken (`used`)."""

    def __init__(self, *, directory: Path | None = None, files: dict[str, bytes] | None = None,
                 base: str = "", where: str = "") -> None:
        self.directory = directory
        self.files = files or {}
        self.base = base
        #: How a sentence names where they are looked for.
        self.where = where
        self.used: set[str] = set()

    def tree(self, relative: str, *, take: bool = True) -> dict[str, bytes]:
        """The file at `relative` — {"": its bytes} — or the files of the folder there, by their
        paths in it: nothing where there is neither. `take`: what is found counts as used."""
        from tuningfork.ops.addskill import _read_folder

        if self.directory is not None:
            path = self.directory / relative
            if path.is_file():
                return {"": path.read_bytes()}
            return _read_folder(path)[0] if path.is_dir() else {}
        parts = PurePosixPath(self.base, relative.replace("\\", "/")).parts
        if ".." in parts:
            return {}
        at = PurePosixPath(*parts).as_posix() if parts else ""
        if at in self.files:
            found, prefix = {"": self.files[at]}, at
        else:
            prefix = f"{at}/" if at else ""
            found = {path[len(prefix):]: data for path, data in self.files.items() if path.startswith(prefix)}
        if take:
            self.used.update(f"{prefix}{sub}" for sub in found)
        return found


def _placed(name: str, sub: str) -> str:
    return f"{name}/{sub}" if sub else name


def _new_name(home: Path | str, name: str) -> None:
    """A name a new scenario can take: its own, and no folder there yet."""
    place = folder(home, name)
    if (place / SPEC).exists():
        raise OperationError(f"a scenario called {name!r} exists already", code="conflict", field="name")
    if place.exists():
        raise OperationError(f"{place} is there already, holding no scenario: move it away to use the name",
                             code="conflict", field="name")


def _take_files(plan: ImportPlan, beside: _Beside) -> None:
    """Each input file the spec names beside it, as the scenario keeps one — in its `inputs/`,
    named by the spec as `{"path", "source"}` — and what its check reads from its task's folder,
    beside the scenario's evals.json, or in its `inputs/` where the check reads one there."""
    from tuningfork.ops.imports import SKILL_FOLDER

    for item in plan.spec["evals"]:
        entries = []
        for entry in item.get("files") or []:
            if isinstance(entry, str):
                # A file or a folder beside the spec, put in the workspace under its own name.
                found = beside.tree(entry)
                if not found:
                    raise OperationError(f"{entry} is not {beside.where}: the task file names it as an input file",
                                         field="path")
                own = PurePosixPath(entry).name
                for sub, data in sorted(found.items()):
                    plan.inputs[_placed(own, sub)] = data
                    entries.append({"path": _placed(own, sub), "source": f"{INPUTS}/{_placed(own, sub)}"})
            elif isinstance(entry, dict) and "source" in entry:
                data = beside.tree(entry["source"]).get("")
                if data is None:
                    raise OperationError(f"{entry['source']} is not {beside.where}: the task file names it as an "
                                         "input file", field="path")
                relative = PurePosixPath(entry["path"]).as_posix()
                plan.inputs[relative] = data
                entries.append({**entry, "source": f"{INPUTS}/{relative}"})
            else:
                entries.append(entry)
        item["files"] = entries
    unread: list[str] = []
    for item in plan.spec["evals"]:
        check = (item.get("tuningfork") or {}).get("check")
        for relative in _check_reads(check) if isinstance(check, str) else []:
            top = PurePosixPath(relative).parts[0]
            if top == INPUTS:
                within = relative[len(INPUTS) + 1:]
                for sub, data in beside.tree(relative).items():
                    plan.inputs.setdefault(_placed(within, sub) if within else sub, data)
                continue
            found = beside.tree(top) if top not in (SPEC, CONTEXT, SKILL_FOLDER, TRASH) else {}
            if not found:
                unread.append(relative)
            for sub, data in found.items():
                plan.beside[_placed(top, sub)] = data
    if plan.beside:
        tops = sorted({PurePosixPath(path).parts[0] for path in plan.beside})
        plan.notes.append(f"its check reads {', '.join(tops)} from its task's folder: copied beside the scenario's "
                          "evals.json, where no run sees it")
    if unread:
        plan.warnings.append(f"its check reads {', '.join(dict.fromkeys(unread))} from its task's folder, which is not "
                             f"{beside.where}: put it beside the scenario's evals.json")
    if plan.inputs:
        plan.notes.append(f"{len(plan.inputs)} input file(s) copied into the scenario's inputs/")


def _limited(plan: ImportPlan) -> None:
    """A plan no larger than a scenario holds: each file, and the files it copies in all."""
    for part in (plan.inputs, plan.skill, plan.beside):
        for relative, data in part.items():
            if len(data) > MAX_FILE:
                raise OperationError(f"{relative} is larger than {MAX_FILE // (1024 * 1024)} MiB", code="too_large")
        if len(part) > MAX_FILES:
            raise OperationError(f"a scenario holds at most {MAX_FILES} input files, and as many of a skill's or its "
                                 "check's", code="too_large")
    if sum(len(data) for part in (plan.inputs, plan.skill, plan.beside) for data in part.values()) > MAX_TOTAL:
        raise OperationError(f"what a scenario copies in totals at most {MAX_TOTAL // (1024 * 1024)} MiB", code="too_large")


def prepare_import(home: Path | str, source: Path | str, name: str) -> ImportPlan:
    """A task file on this machine, as a scenario: its input files, and what its check reads from
    its folder, copied in; its skill folder kept as a reference to where it is."""
    from tuningfork.paths import resolve_path
    from tuningfork.specs import SpecError, load_spec

    check_name(name)
    _new_name(home, name)
    found = resolve_path(str(source))
    try:
        spec = load_spec(found.path)
    except (OSError, SpecError) as exc:
        read_as = f" (read as {found.path})" if found.translated_from or found.base else ""
        raise OperationError(f"cannot import {source}{read_as}: {exc}", field="path") from None
    data = spec.to_dict()
    plan = ImportPlan(name=name, source=str(found.path), spec=data)
    skill = spec.resolved_skill_dir()
    if skill is not None:
        data["tuningfork"] = {**(data.get("tuningfork") or {}), "skill_dir": str(skill)}
        plan.skill_dir, plan.has_skill = str(skill), (skill / "SKILL.md").is_file()
        if plan.has_skill:
            plan.notes.append(f"its skill folder stays where it is, {skill}: each run gets a copy, and an approved "
                              "edit is written to it")
        else:
            plan.warnings.append(f"its skill folder, {skill}, holds no SKILL.md: choose the skill on the scenario's page")
    else:
        plan.warnings.append(NO_SKILL)
    _take_files(plan, _Beside(directory=spec.directory, where=f"beside it, in {spec.directory}"))
    _limited(plan)
    return plan


def prepare_upload(home: Path | str, name: str, files: dict[str, bytes], *, spec: str | None = None,
                   folder_name: str | None = None) -> ImportPlan:
    """A task file in a folder uploaded from elsewhere, as a scenario: the input files it names,
    and what its check reads from its folder, copied in, and the skill folder it names, where the
    folder uploaded holds it, copied into the scenario's own (`skill/<its name>/`) — the copy an
    approved edit is written to. `spec`: the task file's path in the folder, where its top holds
    no evals.json or several task files."""
    from tuningfork.ops.addskill import IGNORED
    from tuningfork.ops.imports import SKILL_FOLDER, _folder_name
    from tuningfork.paths import resolve_path
    from tuningfork.specs import TaskSpec, validate_spec

    check_name(name)
    _new_name(home, name)
    cleaned = {}
    for path, data in files.items():
        parts = PurePosixPath(path.replace("\\", "/")).parts if isinstance(path, str) else ()
        if not parts or any(part in ("", ".", "..") for part in parts) or path.startswith(("/", "\\")) or ":" in parts[0]:
            raise OperationError(f"{path!r} is not a path inside the folder uploaded", field="files")
        if not IGNORED.intersection(parts):
            cleaned[PurePosixPath(*parts).as_posix()] = data
    tops = sorted(path for path in cleaned if "/" not in path and path.endswith(".json"))
    if spec is None:
        spec = SPEC if SPEC in cleaned else tops[0] if len(tops) == 1 else None
    if spec is None or spec not in cleaned:
        raise OperationError(
            f"{spec} is not in the folder uploaded" if spec else "no evals.json at the top of the folder uploaded",
            hint=f"the task files at its top: {', '.join(tops)}" if tops else "upload the folder that holds the task file",
            field="spec")
    uploaded = folder_name or "the folder uploaded"
    try:
        raw = json.loads(cleaned[spec].decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OperationError(f"{spec} is not valid JSON: {exc}", field="spec") from None
    problems = validate_spec(raw)
    if problems:
        listed = "; ".join(str(problem) for problem in problems[:6])
        raise OperationError(f"{spec} has problems: {listed}", field="spec")
    data = TaskSpec.from_dict(raw, Path(spec)).to_dict()
    plan = ImportPlan(name=name, source=f"{uploaded}/{spec}", spec=data)
    base = PurePosixPath(spec).parent.as_posix()
    beside = _Beside(files=cleaned, base="" if base == "." else base, where=f"in {uploaded}")
    given = (data.get("tuningfork") or {}).get("skill_dir")
    if isinstance(given, str) and given.strip():
        given = given.strip()
        if "SKILL.md" in beside.tree(given, take=False):
            own = _folder_name(PurePosixPath(given.replace("\\", "/")).name or folder_name or "skill")
            plan.skill = {f"{SKILL_FOLDER}/{own}/{sub}": each for sub, each in beside.tree(given).items()}
            data["tuningfork"]["skill_dir"] = f"{SKILL_FOLDER}/{own}"
            plan.skill_dir, plan.has_skill = f"{SKILL_FOLDER}/{own}", True
            plan.notes.append(f"its skill folder, {given}, is copied into the scenario's own folder ({SKILL_FOLDER}/{own}/): "
                              "an approved edit is written to that copy, never back to yours")
        else:
            # Not in what was uploaded: a folder on this machine, where its path names one.
            where = resolve_path(given, folder(home, name)).path
            plan.skill_dir, plan.has_skill = str(where), (where / "SKILL.md").is_file()
            if plan.has_skill:
                plan.notes.append(f"its skill folder stays where it is, {where}: each run gets a copy, and an approved "
                                  "edit is written to it")
            else:
                plan.warnings.append(f"its skill folder, {given}, is not {beside.where}, and the scenario would read it at "
                                     f"{where}, which holds no SKILL.md: choose the skill on the scenario's page")
    else:
        plan.warnings.append(NO_SKILL)
    _take_files(plan, beside)
    left = len(set(cleaned) - beside.used - {spec})
    if left:
        plan.notes.append(f"{left} other file(s) in {uploaded} are left out: the task file does not name them")
    _limited(plan)
    return plan


def execute_import(home: Path | str, plan: ImportPlan) -> Scenario:
    """Write what the plan says: the scenario's folder made new — a skill's copy and what its
    check reads, then its evals.json, then its input files — and taken away whole when anything
    fails."""
    target = folder(home, plan.name)
    try:
        target.mkdir(parents=True)
    except FileExistsError:
        raise OperationError(f"a scenario called {plan.name!r} exists already", code="conflict", field="name") from None
    try:
        for relative, data in sorted({**plan.skill, **plan.beside}.items()):
            path = target.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (target / INPUTS).mkdir(exist_ok=True)
        # The spec goes in first, so each input lands in an existing scenario.
        (target / SPEC).write_text(json.dumps(plan.spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for relative, data in sorted(plan.inputs.items()):
            put_input(home, plan.name, relative, data)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return read_scenario(home, plan.name)


def import_spec(home: Path | str, source: Path | str, name: str) -> Scenario:
    """A spec from elsewhere, as a scenario: its input files, and what its check reads from its
    folder, copied in; its skill folder kept as a reference to where it is."""
    return execute_import(home, prepare_import(home, source, name))


# --- the list ---------------------------------------------------------------------------------


@dataclass(slots=True)
class ScenarioSummary:
    name: str
    skill_name: str | None
    evals: int
    modified: float
    problems: int
    #: Per task (skill_name/eval id): each condition's pass rate, newest runs first.
    results: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name, "skill_name": self.skill_name, "evals": self.evals,
            "modified": self.modified, "problems": self.problems, "results": self.results,
        }


def _results(runs: list[Any], tasks: set[str]) -> list[dict[str, Any]]:
    """Each task's pass rate under each condition it was run under, newest first."""
    from tuningfork.analysis.conditions import conditions

    found = []
    for task in sorted(tasks):
        for condition in conditions([run for run in runs if run.task_id == task]):
            passed = condition.passed
            found.append({
                "task_id": task,
                "harness": condition.harness,
                "model": condition.model,
                "skill_version": condition.skill_version,
                "skill_file": condition.skill_file,
                "runs": len(condition.runs),
                "passed": passed.to_json() if passed else None,
                "uncounted": condition.uncounted,
                "latest": max(run.started_at for run in condition.runs),
            })
    return sorted(found, key=lambda result: result["latest"], reverse=True)


def list_scenarios(store: Any) -> list[ScenarioSummary]:
    base = root(store.home)
    if not base.is_dir():
        return []
    found = []
    runs = None
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not NAME.fullmatch(entry.name) or not (entry / SPEC).is_file():
            continue
        try:
            scenario = read_scenario(store.home, entry.name)
        except OperationError:
            found.append(ScenarioSummary(entry.name, None, 0, (entry / SPEC).stat().st_mtime, 1, []))
            continue
        spec = scenario.spec
        skill_name = spec.get("skill_name") if isinstance(spec.get("skill_name"), str) else None
        tasks = {
            f"{skill_name}/{item.get('id')}"
            for item in spec.get("evals") or []
            if isinstance(item, dict) and skill_name
        }
        if runs is None:
            runs = store.runs(limit=100_000)
        results = _results(runs, tasks)
        found.append(ScenarioSummary(
            name=entry.name, skill_name=skill_name, evals=len(spec.get("evals") or []),
            modified=scenario.modified, problems=len(scenario.checked.problems), results=results,
        ))
    return sorted(found, key=lambda summary: summary.modified, reverse=True)
