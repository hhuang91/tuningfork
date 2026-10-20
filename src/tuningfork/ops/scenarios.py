"""Scenarios: task specs made and kept in the recordings folder, from the browser.

A scenario is `<home>/scenarios/<name>/evals.json` — the same evals.json convention as any
spec — with its input files in `inputs/` beside it. Its skill folder is a reference to a
folder elsewhere, never copied in — but for one made from a run whose skill has no folder of
its own to be found (`ops.imports.prepare_scenario`), which keeps a copy in `skill/`.
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

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import under_wsl

        return {
            "ok": self.ok, "problems": self.problems, "warnings": self.warnings, "paths": self.paths,
            # How a typed path is read here: under WSL, a Windows path is translated.
            "wsl": under_wsl(),
        }


def check_spec(data: Any, *, home: Path | str | None = None, name: str | None = None) -> Checked:
    """Everything wrong with a scenario's spec, and what a person should know about it."""
    from tuningfork.paths import host_path, resolve_path
    from tuningfork.runner import lookalike_skill_files
    from tuningfork.specs import validate_spec

    checked = Checked(problems=[{"where": p.where, "message": p.message} for p in validate_spec(data)])
    if not isinstance(data, dict):
        return checked
    top = data.get("tuningfork") if isinstance(data.get("tuningfork"), dict) else {}
    skill_dir = top.get("skill_dir")
    skill: Path | None = None
    if isinstance(skill_dir, str) and skill_dir.strip():
        # A relative path is resolved against the scenario's own folder, where its evals.json
        # lives, exactly as a run resolves it. A new scenario has no folder until it has a name.
        named = home is not None and bool(name) and NAME.fullmatch(name or "") is not None
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


def import_spec(home: Path | str, source: Path | str, name: str) -> Scenario:
    """A spec from elsewhere, as a scenario: its input files copied into `inputs/`, its
    skill folder kept as a reference to where it is."""
    from tuningfork.paths import resolve_path
    from tuningfork.specs import SpecError, load_spec

    check_name(name)
    if (folder(home, name) / SPEC).exists():
        raise OperationError(f"a scenario called {name!r} exists already", code="conflict", field="name")
    found = resolve_path(str(source))
    try:
        spec = load_spec(found.path)
    except (OSError, SpecError) as exc:
        read_as = f" (read as {found.path})" if found.translated_from or found.base else ""
        raise OperationError(f"cannot import {source}{read_as}: {exc}", field="path") from None
    data = spec.to_dict()
    top = data.get("tuningfork") or {}
    skill = spec.resolved_skill_dir()
    if skill is not None:
        top["skill_dir"] = str(skill)
        data["tuningfork"] = top
    copies: list[tuple[str, Path]] = []
    for item in data["evals"]:
        files = []
        for entry in item.get("files") or []:
            if isinstance(entry, str):
                copies.append((PurePosixPath(entry).name, spec.directory / entry))
                files.append({"path": PurePosixPath(entry).name, "source": f"{INPUTS}/{PurePosixPath(entry).name}"})
            elif isinstance(entry, dict) and "source" in entry:
                relative = PurePosixPath(entry["path"]).as_posix()
                copies.append((relative, spec.directory / entry["source"]))
                files.append({"path": entry["path"], "source": f"{INPUTS}/{relative}"})
            else:
                files.append(entry)
        item["files"] = files
    for relative, origin in copies:
        if not origin.is_file():
            raise OperationError(f"{origin} does not exist: the spec names it as an input file", field="path")
    target = folder(home, name)
    (target / INPUTS).mkdir(parents=True, exist_ok=True)
    # The spec goes in first, so each input lands in an existing scenario.
    (target / SPEC).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for relative, origin in copies:
        put_input(home, name, relative, origin.read_bytes())
    return read_scenario(home, name)


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
