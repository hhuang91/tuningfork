"""Skills as the recordings know them: `tuningfork skills`, `tuningfork skill`, the Skills page.

A skill is named by its folder, as its variants and refinements are (`ops.variants.skill_name`):
the folder a run's skill files were recorded under, else the first part of its task's id.
Each version of it is a fingerprint of the files its runs were given (`Run.skill_version`):
two runs with one fingerprint read the same skill, byte for byte. Each version comes with its SKILL.md as the
runs read it (the launch's snapshot of the skill folder, or the batch's frozen copy), its
pass rate under each condition it ran under, and what is known of where its text came from:
the skill's own file as it is now, a saved variant, a refinement's proposal, a version an
approval replaced. Beside the versions: the skill's own folder as specs name it, with the
changelog and the old versions an approval keeps beside it (`ops.approve`); its saved
variants; its refinements.

Read-only: nothing here writes the skill's folder, or anything else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store

#: The texts a diff compares, as a front end names them.
REFS = "a version's fingerprint, `file` (its own SKILL.md now), `variant:<name>` or `history:<file>`"
#: A refinement's proposal, saved as a variant (`ops.refine.add_proposal`).
PROPOSAL = re.compile(r"r([0-9a-f]{8})-([0-9a-f]{8})")


def _clean(candidate: str) -> str | None:
    """A name as `ops.variants.skill_name` writes one, so a skill's runs, variants and
    refinements agree on it."""
    from tuningfork.ops.variants import NAME

    cleaned = re.sub(r"[^a-z0-9_.-]+", "-", candidate.lower()).strip("-.")
    return cleaned if NAME.fullmatch(cleaned or "") else None


def skill_of(run: Run) -> str | None:
    """The skill a run was given, by its folder's name — the folder its top SKILL.md was
    recorded in, below any category folder a harness keeps skills in (Hermes:
    `productivity/xlsx/SKILL.md`), else the one folder its files were under — else by the
    first part of its task's id."""
    paths = [path.replace("\\", "/") for path in run.skill_hashes]
    tops = sorted((path.count("/"), path) for path in paths if path.endswith("/SKILL.md"))
    shallowest = [path for depth, path in tops if depth == tops[0][0]] if tops else []
    folders = ({path.rsplit("/", 2)[-2] for path in shallowest}
               or {path.split("/", 1)[0] for path in paths if "/" in path})
    candidates = [next(iter(folders))] if len(folders) == 1 else []
    if run.task_id and "/" in run.task_id:
        candidates.append(run.task_id.split("/", 1)[0])
    return next((name for name in map(_clean, candidates) if name), None)


def _by_skill(store: Store) -> dict[str, list[Run]]:
    """Runs that recorded the skill files they were given, by skill: those in the trash, and
    forks and threads, left out as `Store.runs` leaves them."""
    groups: dict[str, list[Run]] = {}
    for run in store.runs(limit=100_000):
        name = skill_of(run)
        if name and run.skill_version:
            groups.setdefault(name, []).append(run)
    return groups


def _read(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()
    except OSError:
        return None


def text_of(store: Store, run: Run) -> str | None:
    """SKILL.md as a run read it: from its launch's snapshot of the skill folder, or the
    batch's frozen copy while that is still on disk."""
    from tuningfork.ops.refine import run_skill_file
    from tuningfork.snapshots import SnapshotError, Snapshots

    launch = store.get_launch(run.launch_id) if run.launch_id else None
    if launch is not None and launch.skill_commit:
        try:
            return Snapshots(store.home).read(launch.skill_commit, "SKILL.md").decode("utf-8", "replace")
        except SnapshotError:
            pass
    try:
        found = run_skill_file(store, run)
    except OperationError:
        return None  # an imported chat that kept several skills, none of them its own
    return _read(found) if found is not None else None


# --- the skill's own folder ------------------------------------------------------------------------


@dataclass(slots=True)
class Folder:
    """The skill's own folder, as a spec names it, with what an approval keeps beside it."""

    path: Path
    #: The specs that name it: a scenario's, or the spec a launch ran from.
    specs: list[str]
    #: Its SKILL.md now; None when it cannot be read.
    text: str | None
    sha256: str | None
    #: The version whose SKILL.md is this text, if any run has read it.
    version: str | None = None
    #: `<skill>.changelog.md`, beside the folder: what approvals wrote, and why.
    changelog: str | None = None
    #: `<skill>.history/`, beside the folder: the files approvals replaced, oldest first.
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import windows_form

        return {
            "path": str(self.path), "windows": windows_form(self.path), "exists": self.text is not None,
            "specs": self.specs, "text": self.text, "sha256": self.sha256, "version": self.version,
            "changelog": self.changelog, "history": self.history,
        }


def _folder(path: Path, specs: list[str]) -> Folder:
    from tuningfork.experiment.skill import text_sha256

    text = _read(path / "SKILL.md")
    history_dir = path.parent / f"{path.name}.history"
    history = []
    for old in sorted(history_dir.glob("SKILL.*.md")) if history_dir.is_dir() else ():
        old_text = _read(old)
        if old_text is not None:
            history.append({"name": old.name, "path": str(old), "sha256": text_sha256(old_text),
                            "modified": old.stat().st_mtime})
    return Folder(
        path=path, specs=sorted(set(specs)), text=text, sha256=text_sha256(text) if text is not None else None,
        changelog=_read(path.parent / f"{path.name}.changelog.md"), history=history,
    )


def _scenarios(store: Store) -> dict[str, list[tuple[str, Path]]]:
    """Each skill's scenarios, as (name, spec path), by the skill's folder — else the skill
    the spec names: read for that, not run."""
    from tuningfork.ops.scenarios import NAME, SPEC, root

    found: dict[str, list[tuple[str, Path]]] = {}
    base = root(store.home)
    for entry in sorted(base.iterdir()) if base.is_dir() else ():
        spec = entry / SPEC
        if not entry.is_dir() or not NAME.fullmatch(entry.name) or not spec.is_file():
            continue
        try:
            data = json.loads(spec.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        folder = _spec_folder(spec)
        named = data.get("skill_name") if isinstance(data.get("skill_name"), str) else ""
        name = next((each for each in map(_clean, [folder.name if folder else "", named]) if each), None)
        if name:
            found.setdefault(name, []).append((entry.name, spec))
    return found


def _spec_folder(spec_path: str | Path) -> Path | None:
    from tuningfork.specs import load_spec

    try:
        spec = load_spec(Path(spec_path), strict=False)
    except (OSError, ValueError, OperationError):
        return None
    return spec.resolved_skill_dir()


def _folders(store: Store, name: str, runs: Iterable[Run]) -> list[Folder]:
    """The skill's own folders: those its scenarios name, and those the specs its latest runs
    ran from name — one per folder, however many specs share it."""
    specs: dict[str, None] = {str(spec): None for _, spec in _scenarios(store).get(name, [])}
    latest: dict[str, Run] = {}
    for run in runs:  # one run per version is enough to find the specs it came from
        if run.skill_version not in latest or run.started_at > latest[run.skill_version].started_at:
            latest[run.skill_version] = run
    for run in latest.values():
        launch = store.get_launch(run.launch_id) if run.launch_id else None
        if launch is not None and launch.spec_path:
            specs[launch.spec_path] = None
    folders: dict[Path, list[str]] = {}
    for spec in specs:
        found = _spec_folder(spec)
        if found is not None:
            folders.setdefault(found, []).append(spec)
    return [_folder(path, named) for path, named in folders.items()]


# --- versions ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class Version:
    """One version of a skill: every run that read these files, byte for byte."""

    fingerprint: str
    runs: int
    first: float
    latest: float
    #: Each skill file the runs were given, with its digest (`Run.skill_hashes`).
    files: dict[str, str]
    #: SKILL.md as they read it; None when neither a snapshot nor a frozen copy is left.
    text: str | None
    sha256: str | None
    #: Where this text is known from: the skill's own file now, a variant, a proposal…
    known_as: list[str]
    #: Its pass rate under each condition it ran under: task, harness, model, server.
    conditions: list[dict[str, Any]]

    def to_json(self, *, text: bool = True) -> dict[str, Any]:
        found = {
            "fingerprint": self.fingerprint, "runs": self.runs, "first": self.first, "latest": self.latest,
            "files": self.files, "sha256": self.sha256, "known_as": self.known_as, "conditions": self.conditions,
        }
        if text:
            found["text"] = self.text
        return found


def _conditions(runs: list[Run]) -> list[dict[str, Any]]:
    from tuningfork.analysis.conditions import conditions

    found = []
    for condition in conditions(runs):
        passed = condition.passed
        found.append({
            "task_id": condition.task_id, "harness": condition.harness, "model": condition.model,
            "server_id": condition.server_id, "skill_file": condition.skill_file, "runs": len(condition.runs),
            "passed": passed.to_json() if passed else None, "uncounted": condition.uncounted,
            "first": condition.first, "latest": condition.latest,
        })
    return sorted(found, key=lambda each: each["latest"], reverse=True)


def _variants(store: Store, name: str | None = None) -> list[Any]:
    """A skill's saved variants (every skill's, without `name`), less the proposals of a
    refinement in the trash: the trash leaves it out of every list, and what it made with it.
    Found in use: a refinement set aside still listed its proposals' texts here."""
    from tuningfork.ops.refine import list_refinements
    from tuningfork.ops.variants import list_variants

    trashed = {each.id[:8] for each in list_refinements(store, include_closed=True, include_trashed=True)
               if each.trashed_at is not None}

    def kept(variant) -> bool:
        proposal = PROPOSAL.fullmatch(variant.name)
        return not (proposal and proposal.group(1) in trashed)

    return [variant for variant in list_variants(store.home, name) if kept(variant)]


def _known(store: Store, name: str, folders: list[Folder]) -> dict[str, list[str]]:
    """What is known of where a text came from, by its digest (line endings aside)."""
    from tuningfork.experiment.skill import text_sha256

    known: dict[str, list[str]] = {}
    for folder in folders:
        if folder.sha256:
            where = f" ({folder.path})" if len(folders) > 1 else ""
            known.setdefault(folder.sha256, []).append(f"its own file now{where}")
        for old in folder.history:
            known.setdefault(old["sha256"], []).append(f"replaced by an approval: {old['name']}")
    for variant in _variants(store, name):
        text = _read(variant.path)
        if text is None:
            continue
        proposal = PROPOSAL.fullmatch(variant.name)
        label = (f"proposal {proposal.group(2)} of refinement {proposal.group(1)}" if proposal
                 else f"saved variant {variant.name}")
        known.setdefault(text_sha256(text), []).append(label)
    return known


def _versions(store: Store, runs: list[Run], known: dict[str, list[str]]) -> list[Version]:
    from tuningfork.experiment.skill import text_sha256

    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.skill_version or "", []).append(run)
    versions = []
    for fingerprint, members in groups.items():
        newest = max(members, key=lambda run: run.started_at)
        text = None
        # The newest run whose text can still be read: snapshots outlive frozen copies.
        for run in sorted(members, key=lambda each: each.started_at, reverse=True)[:5]:
            text = text_of(store, run)
            if text is not None:
                break
        sha = text_sha256(text) if text is not None else None
        versions.append(Version(
            fingerprint=fingerprint, runs=len(members),
            first=min(run.started_at for run in members), latest=newest.started_at,
            files=dict(newest.skill_hashes), text=text, sha256=sha,
            known_as=list(known.get(sha, [])) if sha else [], conditions=_conditions(members),
        ))
    return sorted(versions, key=lambda version: version.latest, reverse=True)


# --- a skill, and the list ----------------------------------------------------------------------------


@dataclass(slots=True)
class SkillDetail:
    name: str
    tasks: list[str]
    #: Newest first.
    versions: list[Version]
    folders: list[Folder]
    variants: list[dict[str, Any]]
    refinements: list[dict[str, Any]]

    def to_json(self, *, text: bool = True) -> dict[str, Any]:
        return {
            "name": self.name, "tasks": self.tasks,
            "versions": [version.to_json(text=text) for version in self.versions],
            "folders": [folder.to_json() for folder in self.folders],
            "variants": self.variants, "refinements": self.refinements,
        }


def _refinements(store: Store, name: str) -> list[dict[str, Any]]:
    from tuningfork.ops.refine import list_refinements, proposals

    return [
        {"id": each.id, "status": each.status, "run": each.run_id, "step": each.step_idx,
         "task_id": each.data.get("task_id"), "target": (each.target or {}).get("description"),
         "proposals": len(proposals(store, each.id)), "created_at": each.data.get("created_at")}
        for each in list_refinements(store, include_closed=True) if each.skill == name
    ]


def skill(store: Store, name: str) -> SkillDetail:
    """One skill: its versions with their texts and pass rates, its own folder now — with
    its changelog and the files approvals replaced — its saved variants and its refinements."""
    from tuningfork.experiment.skill import text_sha256

    runs = _by_skill(store).get(name, [])
    folders = _folders(store, name, runs)
    variants = _variants(store, name)
    refinements = _refinements(store, name)
    if not (runs or folders or variants or refinements):
        raise OperationError(
            f"no skill {name!r} in these recordings", hint="`tuningfork skills` lists the ones there are",
            code="not_found", field="skill",
        )
    known = _known(store, name, folders)
    versions = _versions(store, runs, known)
    by_text = {version.sha256: version.fingerprint for version in versions if version.sha256}
    for folder in folders:
        folder.version = by_text.get(folder.sha256) if folder.sha256 else None
    variant_rows = []
    for variant in variants:
        text = _read(variant.path)
        sha = text_sha256(text) if text is not None else None
        proposal = PROPOSAL.fullmatch(variant.name)
        variant_rows.append({
            **variant.to_json(), "text_sha256": sha, "version": by_text.get(sha) if sha else None,
            "proposal": {"refinement": proposal.group(1), "proposal": proposal.group(2)} if proposal else None,
        })
    return SkillDetail(
        name=name, tasks=sorted({run.task_id for run in runs if run.task_id}), versions=versions,
        folders=folders, variants=variant_rows, refinements=refinements,
    )


@dataclass(slots=True)
class SkillSummary:
    name: str
    tasks: list[str]
    runs: int
    versions: int
    latest: float | None
    folders: list[str]
    scenarios: list[str]
    variants: int
    #: Refinements not closed: open, or approved.
    refinements: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name, "tasks": self.tasks, "runs": self.runs, "versions": self.versions,
            "latest": self.latest, "folders": self.folders, "scenarios": self.scenarios,
            "variants": self.variants, "refinements": self.refinements,
        }


def skills(store: Store) -> list[SkillSummary]:
    """Every skill the recordings know — from runs, scenarios, saved variants and
    refinements — the latest run first, then by name."""
    from tuningfork.ops.refine import list_refinements

    runs = _by_skill(store)
    scenarios: dict[str, list[str]] = {}
    folders: dict[str, dict[str, None]] = {}
    for name, named in _scenarios(store).items():
        for scenario, spec in named:
            scenarios.setdefault(name, []).append(scenario)
            found = _spec_folder(spec)
            if found is not None:
                folders.setdefault(name, {})[str(found)] = None
    for name, members in runs.items():
        newest = max(members, key=lambda run: run.started_at)
        launch = store.get_launch(newest.launch_id) if newest.launch_id else None
        found = _spec_folder(launch.spec_path) if launch is not None and launch.spec_path else None
        if found is not None:
            folders.setdefault(name, {})[str(found)] = None
    variants: dict[str, int] = {}
    for variant in _variants(store):
        variants[variant.skill] = variants.get(variant.skill, 0) + 1
    refinements: dict[str, dict[str, int]] = {}
    for each in list_refinements(store, include_closed=True):
        counts = refinements.setdefault(each.skill, {"open": 0, "approved": 0, "closed": 0})
        counts[each.status] = counts.get(each.status, 0) + 1
    names = set(runs) | set(scenarios) | set(variants) | set(refinements)
    found = [
        SkillSummary(
            name=name,
            tasks=sorted({run.task_id for run in runs.get(name, []) if run.task_id}),
            runs=len(runs.get(name, [])),
            versions=len({run.skill_version for run in runs.get(name, [])}),
            latest=max((run.started_at for run in runs.get(name, [])), default=None),
            folders=list(folders.get(name, {})),
            scenarios=sorted(scenarios.get(name, [])),
            variants=variants.get(name, 0),
            refinements=refinements.get(name, {"open": 0, "approved": 0, "closed": 0}),
        )
        for name in names
    ]
    return sorted(found, key=lambda summary: (-(summary.latest or 0), summary.name))


# --- two texts, line by line ----------------------------------------------------------------------------


@dataclass(slots=True)
class SkillDiff:
    name: str
    before: dict[str, Any]
    after: dict[str, Any]
    diff: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"skill": self.name, "before": self.before, "after": self.after, **self.diff}


def _text_for(store: Store, name: str, ref: str, found: SkillDetail) -> tuple[str, str]:
    """A text of the skill, by how a front end names it: (what to call it, the text)."""
    from tuningfork.ops.variants import read_variant

    if ref == "file" or ref.startswith("file:"):
        index = int(ref.split(":", 1)[1]) if ":" in ref and ref.split(":", 1)[1].isdigit() else 0
        folders = [folder for folder in found.folders if folder.text is not None]
        if index >= len(folders):
            raise OperationError(f"skill {name} has no SKILL.md of its own to read", code="not_found", field="ref")
        return f"its own file now ({folders[index].path})", folders[index].text or ""
    if ref.startswith("variant:"):
        variant = ref.split(":", 1)[1]
        return f"variant {variant}", read_variant(store.home, name, variant)
    if ref.startswith("history:"):
        wanted = ref.split(":", 1)[1]
        for folder in found.folders:
            for old in folder.history:
                if old["name"] == wanted:
                    return f"replaced by an approval: {wanted}", _read(Path(old["path"])) or ""
        raise OperationError(f"no old version {wanted!r} beside skill {name}'s folder", code="not_found", field="ref")
    matches = [version for version in found.versions if version.fingerprint.startswith(ref)] if ref else []
    if len(matches) != 1:
        raise OperationError(f"no single version of {name} matching {ref!r}", hint=f"name {REFS}",
                             code="not_found", field="ref")
    if matches[0].text is None:
        raise OperationError(f"version {matches[0].fingerprint}'s SKILL.md is no longer on disk or in a snapshot",
                             code="not_found", field="ref")
    return f"version {matches[0].fingerprint}", matches[0].text


@dataclass(slots=True)
class SkillText:
    name: str
    ref: str
    label: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"skill": self.name, "ref": self.ref, "label": self.label, "text": self.text}


def skill_text(store: Store, name: str, ref: str) -> SkillText:
    """One text of a skill, named as `REFS` says."""
    found = skill(store, name)
    label, text = _text_for(store, name, ref, found)
    return SkillText(name=name, ref=ref, label=label, text=text)


def skill_diff(store: Store, name: str, before: str, after: str) -> SkillDiff:
    """Two texts of a skill line by line: each named as `REFS` says."""
    from tuningfork.ops.diff import line_diff

    found = skill(store, name)
    before_label, before_text = _text_for(store, name, before, found)
    after_label, after_text = _text_for(store, name, after, found)
    return SkillDiff(
        name=name, before={"ref": before, "label": before_label}, after={"ref": after, "label": after_label},
        diff=line_diff(before_text, after_text),
    )
