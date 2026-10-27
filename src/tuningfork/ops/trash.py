"""The trash: runs, experiments, jobs, scenarios and refinements set aside, and deleted for
good from there.

Moving something to the trash changes a mark and nothing else: it leaves every list and every
statistic, and whatever is tied to it stays where it is — experiments at a run's steps, their
samples, the forks that replayed it — so it comes back as it went. No new work starts on a run
or a refinement in the trash. Deleting from the trash is permanent: the item goes with
everything tied to it (`Store.purge_closure`), and then whatever nothing left refers to —
message content, blobs, workspace snapshots, the folders its launches ran in, and the folder of
a batch or of a set of forks, with the skill copy its runs were given, once none of its runs is
left and nothing else names it. A launch's folder outside the recordings folder goes only where
tuningfork marked its group as made for this recordings folder (`runner.GROUP_MARK`): a copy of
the recordings folder, whose launches name the original's folders, removes none of them, nor its
own copies of them. A path in a folder it removes that something a person keeps names
(`_named`) stays where it is, and the rest goes (`_clear`). What a deleted row held is
overwritten, and the database's log emptied once no other process reads it
(`Store.checkpoint`). What was written in the last two minutes, which a recording in flight may
still need, is swept once they have passed (`sweep_if_due`: a running proxy, or the next
deletion).

A job keeps its record of work deleted since, and a chain its record and the skill each link
left: each is a record of its own, deleted only as itself. What was made of a run stays too — a
scenario's conversation and input files, a skill added from it — and so do the quotes a
refinement that cites it keeps.

A refinement goes with what it made (`ops.refine.made_by`): its screens, its validations'
forks and fresh training runs, its proposals' saved texts, and its own folder. Never the run
it studies, nor what an approval wrote beside the skill's folder — its changelog and the
skill's old versions stay.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.blobs import GRACE
from tuningfork.store.db import Purge, Store

KINDS = ("run", "experiment", "job", "scenario", "refinement")


def refuse_trashed(run: Any) -> None:
    """Refuse new work on a run in the trash: whatever it made would be deleted with it."""
    if getattr(run, "trashed_at", None) is not None:
        raise OperationError(
            f"run {run.id[:8]} is in the trash",
            hint=f"restore it first: `tuningfork trash restore {run.id[:8]}`",
            code="trashed",
        )


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise OperationError(f"no {kind!r} to trash: one of {', '.join(KINDS)}", code="bad_params", field="kind")
    return kind


def _run(store: Store, reference: str) -> Any:
    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}", code="not_found")
    if run.is_thread:
        parent = run.parent_run_id[:8] if run.parent_run_id else "its launch's main run"
        raise OperationError(
            f"run {run.id[:8]} is a thread of {parent}: it goes to the trash with that run",
            code="bad_params",
        )
    return run


def experiment_of_fork(store: Store, run: Any) -> str | None:
    """The experiment a run is a sample of — a fork it drew, run to the end — or None. Such a
    run goes to the trash with its experiment, never alone, as a thread goes with its run: its
    verdict is the experiment's, kept in its sample. Found in review: a fork trashed, even
    deleted for good, still counted in its screen; and taking single forks out of a finished
    experiment would choose which of them count."""
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    return launch.experiment_id if launch is not None and launch.experiment_id else None


def _fork_alone(run: Any, experiment: str) -> OperationError:
    return OperationError(
        f"run {run.id[:8]} is a fork experiment {experiment[:8]} drew, one of its samples: it goes to the trash, "
        "and is deleted, with that experiment",
        hint=f"`tuningfork trash experiment {experiment[:8]}`", code="bad_params",
    )


def _experiments(store: Store, reference: str) -> tuple[str, list[Any]]:
    """An experiment, with the other arm of its A/B: (the id they are filed under, the arms)."""
    found = store.get_experiment(reference)
    if found is None:
        arms = store.experiment_group(reference)  # an A/B by its group, as the trash lists it
        if arms:
            return reference, arms
        raise OperationError(f"no experiment matching {reference!r}", code="not_found")
    if found.group_id:
        return found.group_id, store.experiment_group(found.group_id)
    return found.id, [found]


def _job(store: Store, reference: str) -> Any:
    found = store.get_job(reference)
    if found is None:
        raise OperationError(f"no job matching {reference!r}", code="not_found")
    return found


def _refinement(store: Store, reference: str) -> Any:
    from tuningfork.ops import refine

    return refine.get(store, reference)


def _refinement_busy(store: Store, refinement: Any) -> None:
    """Refused while work on a refinement runs — a proposer asked, a screen, a validation —
    since what it made would be left without its refinement."""
    from tuningfork.ops.inspect import arm_status
    from tuningfork.ops.refine import made_by
    from tuningfork.ops.validate import validations

    for job in store.jobs(limit=10_000, unfinished=True):
        reference = (job.params or {}).get("refinement")
        if isinstance(reference, str) and reference and refinement.id.startswith(reference):
            raise OperationError(
                f"a {job.kind} job on refinement {refinement.id[:8]} is still {job.status}",
                hint="wait for it, or cancel it", code="busy",
            )
    if any(record.get("status") == "running" for record in validations(store, refinement.id)):
        raise OperationError(
            f"a validation of refinement {refinement.id[:8]} is still running", hint="wait for it", code="busy",
        )
    for experiment_id in made_by(store, refinement).experiments:
        experiment = store.get_experiment(experiment_id)
        if experiment is not None and arm_status(experiment) == "running":
            raise OperationError(
                f"a {experiment.kind} of refinement {refinement.id[:8]} is still running", hint="wait for it",
                code="busy",
            )


# --- moving to the trash, and back -------------------------------------------------------------


def trash(store: Store, kind: str, reference: str) -> dict[str, Any]:
    """Move one item to the trash. Refused while it, or work on it, is still running — by a
    process still alive: work whose process died was cut off (`ops.inspect.launch_status`)."""
    from tuningfork.ops.inspect import arm_status, launch_status

    kind = _kind(kind)
    if kind == "run":
        run = _run(store, reference)
        if run.trashed_at is not None:
            return {"kind": kind, "id": run.id, "moved": [], "already": True}
        experiment = experiment_of_fork(store, run)
        if experiment:
            raise _fork_alone(run, experiment)
        launch = store.get_launch(run.launch_id) if run.launch_id else None
        if launch is not None and launch_status(launch) == "running":
            raise OperationError(f"run {run.id[:8]} is still running", hint="wait for it, or cancel its job", code="busy")
        running = [e for e in store.experiments(run_id=run.id, limit=10_000) if arm_status(e) == "running"]
        if running:
            raise OperationError(
                f"an experiment is running at step {running[0].step_idx} of run {run.id[:8]}",
                hint="wait for it, or cancel its job", code="busy",
            )
        moved = store.trash_run(run.id)
        return {"kind": kind, "id": run.id, "label": run.label, "moved": moved}
    if kind == "experiment":
        root, arms = _experiments(store, reference)
        if any(arm_status(arm) == "running" for arm in arms):
            raise OperationError(f"experiment {arms[0].id[:8]} is still running", hint="wait for it, or cancel its job", code="busy")
        if all(arm.trashed_at is not None for arm in arms):
            return {"kind": kind, "id": root, "moved": [], "already": True}
        moved = store.trash_experiments(root, [arm.id for arm in arms])
        return {"kind": kind, "id": root, "moved": moved}
    if kind == "job":
        job = _job(store, reference)
        if not job.finished:
            raise OperationError(f"job {job.id[:8]} is still {job.status}", hint="cancel it first", code="busy")
        store.trash_job(job.id)
        return {"kind": kind, "id": job.id, "moved": [job.id]}
    if kind == "refinement":
        from tuningfork.ops.refine import TRASHED, _replace

        found = _refinement(store, reference)
        if found.trashed_at is not None:
            return {"kind": kind, "id": found.id, "moved": [], "already": True}
        _refinement_busy(store, found)
        _replace(found.folder / TRASHED, {"trashed_at": time.time()})
        return {"kind": kind, "id": found.id, "label": found.skill, "moved": [found.id]}
    from tuningfork.ops.scenarios import delete_scenario

    moved = delete_scenario(store.home, reference)
    return {"kind": kind, "id": Path(moved).name, "moved": [moved]}


def restore(store: Store, kind: str, reference: str) -> dict[str, Any]:
    """Bring an item back from the trash, as it went."""
    kind = _kind(kind)
    if kind == "scenario":
        return _restore_scenario(store.home, reference)
    if kind == "refinement":
        from tuningfork.ops.refine import TRASHED

        found = _refinement(store, reference)
        if found.trashed_at is None:
            raise OperationError("that refinement is not in the trash", code="not_found")
        (found.folder / TRASHED).unlink(missing_ok=True)
        return {"kind": kind, "id": found.id, "restored": True}
    if kind == "run":
        root = _run(store, reference).id
    elif kind == "experiment":
        root = _experiments(store, reference)[0]
    else:
        root = _job(store, reference).id
    if not store.restore_trashed(root):
        raise OperationError(f"that {kind} is not in the trash", code="not_found")
    return {"kind": kind, "id": root, "restored": True}


# --- deleting for good --------------------------------------------------------------------------


@dataclass(slots=True)
class Removal:
    """What a permanent deletion removed."""

    items: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    nodes: int = 0
    blobs: int = 0
    bytes: int = 0
    folders: int = 0
    #: Blobs nothing refers to that were written in the last two minutes: swept once those have
    #: passed (`sweep_if_due`).
    held: int = 0
    #: The refinements that cited what was removed (`ops.refine.citations`).
    cited_by: list[dict[str, Any]] = field(default_factory=list)

    def add(self, counts: dict[str, int]) -> None:
        for name, value in counts.items():
            self.counts[name] = self.counts.get(name, 0) + value

    def to_json(self) -> dict[str, Any]:
        return {
            "items": self.items, "removed": self.counts, "message_nodes": self.nodes,
            "blobs": self.blobs, "bytes": self.bytes, "folders": self.folders, "held": self.held,
            "cited_by": self.cited_by,
        }


def _goes(store: Store, plan: Purge, excluding: Sequence[str] = ()) -> dict[str, Any]:
    """What a deletion removes, counted, and the refinements that cite it — but those going
    with it (`excluding`)."""
    from tuningfork.ops.refine import citations

    return {**plan.to_json(), "cited_by": citations(store, runs=plan.runs, experiments=plan.experiments,
                                                    excluding=excluding),
            "notes": _outside(store, plan)}


def _outside(store: Store, plan: Purge) -> list[str]:
    """That deleting `plan` reaches outside the recordings folder, and where: the folders
    tuningfork made there for its launches, marked as made for this recordings folder. And what
    stays in the folders it removes: a path something a person keeps names (`_named`)."""
    from tuningfork.ops.imports import root as imports

    if not plan.launches and not plan.runs:
        return []
    home = store.home.resolve()
    launched = [folder for workspace, harness_home in plan.launches.values()
                if (folder := _launch_folder(home, workspace, harness_home)) is not None]
    groups = sorted({str(folder.parent) for folder in launched if not _inside(home, folder, 1)})
    notes = [f"the folders tuningfork made for its launches outside the recordings folder go too, in "
             f"{', '.join(groups)}"] if groups else []
    named = _named(store, home)
    folders = launched + [imports(home) / run for run in plan.runs]
    kept = sorted({str(path) for folder in folders for path in named if path == folder or folder in path.parents})
    if kept:
        notes.append(f"what a skill added, a scenario or a task file names in them stays, and the rest goes: "
                     f"{', '.join(kept)}")
    return notes


def _refinement_goes(store: Store, refinement: Any) -> dict[str, Any]:
    """What deleting a refinement for good removes: what it made, counted, and its folder."""
    from tuningfork.ops.refine import made_by

    made = made_by(store, refinement)
    goes = _goes(store, store.purge_closure(runs=made.runs, experiments=made.experiments), [refinement.id])
    notes = list(goes["notes"])
    if refinement.status == "approved":
        notes.append(f"refinement {refinement.id[:8]} is approved: its approval and the evidence kept in it go "
                     "too; the skill's changelog and its old versions, beside the skill's folder, stay")
    return {**goes, "refinements": 1, "variants": len(made.variants), "notes": notes}


def preview(store: Store, kind: str, reference: str) -> dict[str, Any]:
    """What deleting an item for good would remove, before anyone confirms it — and which
    refinements cite it."""
    kind = _kind(kind)
    if kind == "run":
        return _goes(store, store.purge_closure(runs=[_run(store, reference).id]))
    if kind == "experiment":
        return _goes(store, store.purge_closure(experiments=[arm.id for arm in _experiments(store, reference)[1]]))
    if kind == "job":
        return {"jobs": 1 if _job(store, reference) else 0, "notes": []}
    if kind == "refinement":
        return _refinement_goes(store, _refinement(store, reference))
    return {"scenarios": 1 if _trashed_scenario(store.home, reference) else 0, "notes": []}


def purge(store: Store, kind: str, reference: str) -> Removal:
    """Delete an item in the trash for good, with everything tied to it."""
    kind = _kind(kind)
    removal = Removal()
    if kind == "scenario":
        shutil.rmtree(_trashed_scenario(store.home, reference))
        removal.items.append({"kind": kind, "id": reference})
        removal.add({"scenarios": 1})
        return removal
    if kind == "job":
        job = _job(store, reference)
        if job.trashed_at is None:
            raise _not_in_trash(kind, job.id)
        store.purge_job(job.id)
        if not store.checkpoint():
            _write_due(store.home, time.time(), only_the_log=True)  # emptied when the proxy next looks
        removal.items.append({"kind": kind, "id": job.id})
        removal.add({"jobs": 1})
        return removal
    if kind == "refinement":
        from tuningfork.ops.refine import made_by

        found = _refinement(store, reference)
        if found.trashed_at is None:
            raise _not_in_trash(kind, found.id)
        _refinement_busy(store, found)
        made = made_by(store, found)
        plan = store.purge_closure(runs=made.runs, experiments=made.experiments)
        removal.items.append({"kind": kind, "id": found.id, "label": found.skill})
        removal.cited_by = _goes(store, plan, [found.id])["cited_by"]
        _remove(store, plan, removal)
        _remove_refinement(store, found, made.variants, removal)
        _sweep(store, removal)
        return removal
    if kind == "run":
        run = _run(store, reference)
        if run.trashed_with != run.id:
            raise _not_in_trash(kind, run.id)
        # One that went to the trash alone before forks went only with their experiments: its
        # verdict would stay in the experiment. It comes back with `restore`.
        experiment = experiment_of_fork(store, run)
        if experiment:
            raise _fork_alone(run, experiment)
        plan = store.purge_closure(runs=[run.id])
        removal.items.append({"kind": kind, "id": run.id, "label": run.label})
    else:
        root, arms = _experiments(store, reference)
        if any(arm.trashed_at is None for arm in arms):
            raise _not_in_trash(kind, arms[0].id)
        plan = store.purge_closure(experiments=[arm.id for arm in arms])
        removal.items.append({"kind": kind, "id": root})
    removal.cited_by = _goes(store, plan)["cited_by"]
    _remove(store, plan, removal)
    _sweep(store, removal)
    return removal


def empty(store: Store) -> Removal:
    """Delete everything in the trash for good."""
    from tuningfork.ops.refine import made_by

    removal = Removal()
    found = store.trashed()
    runs = [run.id for run in found["runs"]]
    experiments = [experiment.id for experiment in found["experiments"]]
    # A fork that went to the trash without the experiment that drew it (before forks went only
    # with theirs) would leave its verdict standing there: it is restored, or its experiment goes too.
    alone = [run for run in found["runs"] if (drew := experiment_of_fork(store, run)) and drew not in experiments]
    if alone:
        raise OperationError(
            f"{len(alone)} fork run(s) in the trash went there without the experiment that drew them, whose "
            f"verdict counts them: {', '.join(run.id[:8] for run in alone)}",
            hint="restore each (`tuningfork trash restore <run>`), or move its experiment to the trash too",
            code="bad_params",
        )
    refinements = _trashed_refinements(store)
    for refinement in refinements:
        _refinement_busy(store, refinement)
    made = {refinement.id: made_by(store, refinement) for refinement in refinements}
    tied_runs = runs + [run for each in made.values() for run in each.runs]
    tied_experiments = experiments + [experiment for each in made.values() for experiment in each.experiments]
    if tied_runs or tied_experiments:
        plan = store.purge_closure(runs=tied_runs, experiments=tied_experiments)
        removal.cited_by = _goes(store, plan, list(made))["cited_by"]
        _remove(store, plan, removal)
    for refinement in refinements:
        _remove_refinement(store, refinement, made[refinement.id].variants, removal)
    for job in found["jobs"]:
        store.purge_job(job.id)
    removal.add({"jobs": len(found["jobs"])})
    scenarios = _trashed_scenarios(store.home)
    for entry in scenarios:
        shutil.rmtree(entry["path"])
    removal.add({"scenarios": len(scenarios)})
    removal.items = (
        [{"kind": "run", "id": r} for r in runs] + [{"kind": "experiment", "id": e} for e in experiments]
        + [{"kind": "job", "id": j.id} for j in found["jobs"]] + [{"kind": "scenario", "id": s["id"]} for s in scenarios]
        + [{"kind": "refinement", "id": r.id, "label": r.skill} for r in refinements]
    )
    _sweep(store, removal)
    return removal


def _not_in_trash(kind: str, item: str) -> OperationError:
    return OperationError(
        f"{kind} {item[:8]} is not in the trash", hint="move it to the trash first: deletion is from there",
        code="bad_params",
    )


def _refuse_running(store: Store, plan: Purge) -> None:
    """Refused while anything that would go is still at work. A fork started before its
    source went to the trash runs on, and so may an experiment at one of its steps: found in
    review, deleting the source took them with it, the fork's folders included, while its
    harness was still working there. Work whose process died was cut off, and holds nothing."""
    from tuningfork.ops.inspect import arm_status, launch_status

    for launch_id in plan.launches:
        launch = store.get_launch(launch_id)
        if launch is not None and launch_status(launch) == "running":
            raise OperationError(
                f"{launch.kind} {launch.label or launch_id[:8]} is still running, and would go with it",
                hint="wait for it, or cancel its job", code="busy",
            )
    for experiment_id in plan.experiments:
        experiment = store.get_experiment(experiment_id)
        if experiment is not None and arm_status(experiment) == "running":
            raise OperationError(
                f"a {experiment.kind} at step {experiment.step_idx} of run {(experiment.run_id or '?')[:8]} is still "
                "running, and would go with it", hint="wait for it, or cancel its job", code="busy",
            )


def _remove(store: Store, plan: Purge, removal: Removal) -> None:
    from tuningfork.snapshots import SnapshotError, Snapshots

    _refuse_running(store, plan)
    groups = _groups(store, plan)  # read from the launches, before their rows go
    store.purge(plan)
    removal.add(plan.to_json())
    snapshots = Snapshots(store.home)
    home = store.home.resolve()
    named = _named(store, home)  # by what is left
    for launch_id, (workspace, harness_home) in plan.launches.items():
        try:
            snapshots.drop(launch_id)
        except SnapshotError:
            pass  # a store git cannot read keeps what it holds; the rows are gone either way
        folder = _launch_folder(home, workspace, harness_home)
        if folder is not None:
            _clear(folder, named)
            removal.folders += 1
    # An imported run's conversation and skill, kept beside it (`ops.imports`).
    from tuningfork.ops.imports import root as imports

    for run_id in plan.runs:
        kept = imports(home) / run_id
        if kept.is_dir():
            _clear(kept, named)
            removal.folders += 1
    _remove_groups(store, home, groups, removal, named)


def _clear(folder: Path, named: Sequence[Path]) -> None:
    """Remove `folder`, but any path `named` inside it, and the folders leading to one: what a
    person keeps a name of stays where it is, and the rest of the run goes. One named itself stays
    whole. What cannot be read is left."""
    if folder in named:
        return
    kept = [path for path in named if folder in path.parents]
    try:
        if not kept:
            shutil.rmtree(folder, ignore_errors=True)
            return
        for entry in folder.iterdir():
            if entry in kept:
                continue
            if entry.is_dir() and not entry.is_symlink():
                if any(entry in path.parents for path in kept):
                    _clear(entry, kept)
                else:
                    shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    except OSError:
        pass


def _groups(store: Store, plan: Purge) -> dict[Path, bool]:
    """The folders the launches going were laid out in (`<group>/<i>/`), and those holding the
    skill copy they were given (`<group>/skill-snapshot*/<skill>`), each said to be a chain's or
    not. A chain's folder holds the skill each link left, which the chain keeps."""
    from tuningfork.runner import SNAPSHOT

    groups: dict[Path, bool] = {}
    for launch_id in plan.launches:
        launch = store.get_launch(launch_id)
        if launch is None:
            continue
        chain = bool((launch.settings or {}).get("chain"))
        found = []
        try:
            workspace = Path(launch.workspace).resolve() if launch.workspace else None
            if workspace is not None and workspace.name == "workspace" and workspace.parent.name.isdigit():
                found.append(workspace.parent.parent)
            if launch.skill_dir:
                copy = Path(launch.skill_dir).resolve().parent
                if copy.name.startswith(SNAPSHOT):
                    found.append(copy.parent)
        except OSError:
            continue  # a path it cannot read: its folders stay
        for group in found:
            groups[group] = groups.get(group, False) or chain
    return groups


def _remove_groups(store: Store, home: Path, groups: dict[Path, bool], removal: Removal,
                   named: Sequence[Path] = ()) -> None:
    """A batch's folder, or a set of forks', once none of its launches is left: the skill copy
    its runs were given goes with it. Only one tuningfork made for this recordings folder — inside
    it two levels down or more, or marked as made for it, wherever that is — holding nothing but
    its skill copies, to which no launch left refers, and inside which nothing a person keeps
    names a path (`_named`). A chain's stays, with the skill each link left; so does one that
    cannot be read."""
    from tuningfork.runner import GROUP_MARK, SNAPSHOT, made_for

    for group, chain in groups.items():
        try:
            if chain or group.is_symlink() or not group.is_dir():
                continue
            if not _inside(home, group, 2) and not made_for(group, home):
                continue
            if any(entry.name != GROUP_MARK and not (entry.is_dir() and entry.name.startswith(SNAPSHOT))
                   for entry in group.iterdir()):
                continue
            if store.refers_into(group) or _holds(group, named):
                continue
        except OSError:
            continue
        shutil.rmtree(group, ignore_errors=True)
        removal.folders += 1


def _inside(home: Path, folder: Path, depth: int) -> bool:
    """Whether `folder` lies inside the recordings folder, `depth` levels down or more."""
    try:
        return len(folder.relative_to(home).parts) >= depth
    except ValueError:
        return False


def _named(store: Store, home: Path) -> list[Path]:
    """The skill folders something a person keeps names, as a run would open them: a skill added
    and remembered where it is, a scenario's (one in the trash too), and the task file's of each
    launch left in the store. A person can name a batch's skill copy, or a skill as a run's
    harness left it, as any of them, and an approval then writes there: what holds one stays."""
    from tuningfork.ops.addskill import added
    from tuningfork.ops.scenarios import SPEC
    from tuningfork.ops.scenarios import root as scenarios
    from tuningfork.paths import resolve_path

    found = [each.path.resolve() for each in added(home) if each.kind == "folder"]
    # Each task file, with the folder its skill folder is read against: a scenario in the trash
    # comes back to its own name, which a relative skill folder was written from.
    specs = [*((spec, spec.parent) for spec in scenarios(home).glob(f"*/{SPEC}")),
             *((entry["path"] / SPEC, scenarios(home) / entry["name"]) for entry in _trashed_scenarios(home)),
             *((Path(path), Path(path).parent) for path in store.spec_paths())]
    for spec, base in specs:
        try:
            if not spec.is_file():  # a regular file only: never a pipe, which would hold every deletion
                continue
            skill = json.loads(spec.read_text(encoding="utf-8")).get("tuningfork", {}).get("skill_dir")
            if isinstance(skill, str) and skill:
                found.append(resolve_path(skill, base).path.resolve())
        except Exception:  # noqa: BLE001 - a task file it cannot read names nothing; it stops no deletion
            continue
    return found


def _holds(folder: Path, named: Sequence[Path]) -> bool:
    """Whether `folder` is, or holds, one of the paths `named`."""
    return any(path == folder or folder in path.parents for path in named)


def _remove_refinement(store: Store, refinement: Any, variants: Sequence[Path], removal: Removal) -> None:
    """A refinement's own files, once what it made is gone from the store: its proposals'
    saved texts, then its folder, but a path in it something a person keeps names."""
    for path in variants:
        path.unlink(missing_ok=True)
    _clear(Path(refinement.folder).resolve(), _named(store, store.home.resolve()))
    removal.add({"refinements": 1, "variants": len(variants)})


def _launch_folder(home: Path, workspace: str | None, harness_home: str | None) -> Path | None:
    """The folder a launch ran in — its workspace's and harness home's parent — if tuningfork
    made it for this recordings folder: inside it, two levels down or more, or numbered in a folder
    marked as made for it (`runner.made_for`), wherever that is. Anything else is left alone — a
    copy of the recordings folder's launches name the original's folders — and so is what cannot
    be read."""
    from tuningfork.runner import made_for

    if not workspace:
        return None
    folder = Path(workspace).resolve().parent
    if harness_home and Path(harness_home).resolve().parent != folder:
        return None
    try:
        if folder.is_dir() and (_inside(home, folder, 2)
                                or (folder.name.isdigit() and made_for(folder.parent, home))):
            return folder
    except OSError:
        pass
    return None


def _sweep(store: Store, removal: Removal, *, again: bool = True) -> bool:
    """What nothing left refers to: message-tree nodes, then blobs, then snapshot objects; then
    the database's log emptied, which another process reading at that moment keeps — whether it
    was is returned. What the grace kept is swept again once it has passed, `again`, unless this
    is that sweep."""
    from tuningfork.snapshots import SnapshotError, Snapshots

    removal.nodes += store.collect_garbage()
    held: list[str] = []
    deleted, freed = store.blobs.sweep(store.live_blobs(), held=held)
    removal.blobs += deleted
    removal.bytes += freed
    removal.held += len(held)
    try:
        Snapshots(store.home).prune()
    except SnapshotError:
        pass
    emptied = store.checkpoint()
    if again:
        # Written whatever the blobs held: snapshot objects that young are kept too, uncounted.
        # That sweep empties the log too, where this one could not.
        _write_due(store.home, time.time() + GRACE)
    return emptied


#: Beside the database, when a sweep is due: the time, after a deletion for good, at which what
#: the grace kept may go — and, on a second line `LOG_ONLY`, that only the database's log is left
#: to empty. The proxy looks every `SWEEP_EVERY` seconds. A later deletion for good sweeps
#: everything nothing refers to anyway, so a mark that cannot be written fails nothing.
SWEEP_DUE = ".sweep-due"
SWEEP_EVERY = 30.0
LOG_ONLY = "log"


def _read_due(path: Path) -> tuple[float, bool, str] | None:
    """A mark's time, whether it asks only for the log, and its text; None where there is no
    mark, or it cannot be read. A time that is not one is due now."""
    try:
        written = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = written.splitlines()
    try:
        when = float(lines[0])
    except (IndexError, ValueError):
        when = 0.0
    return (0.0 if when != when else when), lines[1:2] == [LOG_ONLY], written


def _put_due(path: Path, when: float, only_the_log: bool) -> None:
    """Write a mark whole, its time exactly as it is: rounded, a time due now read as the future."""
    temporary = path.with_name(f"{SWEEP_DUE}.{os.getpid()}.{uuid.uuid4().hex}.tmp")  # threads of one process too
    try:
        temporary.write_text(f"{when!r}\n" + (f"{LOG_ONLY}\n" if only_the_log else ""), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _write_due(home: Path, when: float, *, only_the_log: bool = False) -> None:
    """Set when a sweep is due: `when`, or the later time written already — but never later than
    the grace from now, which a clock set wrong once would otherwise keep. A whole sweep asked for
    before stays asked for."""
    path = home / SWEEP_DUE
    found = _read_due(path)
    if found is not None:
        when = max(when, min(found[0], time.time() + GRACE))
        only_the_log = only_the_log and found[1]
    _put_due(path, when, only_the_log)


def sweep_if_due(store: Store, *, now: float | None = None) -> Removal | None:
    """Sweep what a deletion for good left for the grace, once the grace has passed: its blobs
    and snapshot objects, and the database's log — or only the log, where that is all the mark
    asks for. None where nothing is due yet, or the mark cannot be read."""
    path = store.home / SWEEP_DUE
    found = _read_due(path)
    if found is None or (time.time() if now is None else now) < found[0]:
        return None
    when, only_the_log, written = found
    removal = Removal()
    emptied = store.checkpoint() if only_the_log else _sweep(store, removal, again=False)
    try:
        # Unless a deletion since wrote another mark, which its own sweep meets: taken away, or
        # left asking only for the log another process still reads. In the moment between reading
        # it and this, a mark written is lost: its leftovers wait for the next deletion.
        if path.read_text(encoding="utf-8") == written:
            if emptied:
                path.unlink()
            elif not only_the_log:
                _put_due(path, time.time(), True)
    except OSError:
        pass
    return removal


# --- what is in it ----------------------------------------------------------------------------


def find(store: Store, reference: str) -> tuple[str, str]:
    """Which item in the trash a reference names — an id or its prefix, a run's label, a
    scenario's name — as (kind, id). Refused when it names none, or more than one."""
    found = store.trashed()
    candidates = {("run", run.id) for run in found["runs"] if run.id.startswith(reference) or run.label == reference}
    candidates |= {
        ("experiment", experiment.group_id or experiment.id) for experiment in found["experiments"]
        if experiment.id.startswith(reference) or (experiment.group_id or "").startswith(reference)
    }
    candidates |= {("job", job.id) for job in found["jobs"] if job.id.startswith(reference)}
    candidates |= {
        ("scenario", entry["id"]) for entry in _trashed_scenarios(store.home)
        if reference in (entry["id"], entry["name"])
    }
    candidates |= {("refinement", found.id) for found in _trashed_refinements(store) if found.id.startswith(reference)}
    if not reference or not candidates:
        raise OperationError(
            f"nothing in the trash matches {reference!r}", hint="`tuningfork trash list` shows what is there",
            code="not_found",
        )
    if len(candidates) > 1:
        raise OperationError(f"{reference!r} matches {len(candidates)} things in the trash: give more of the id", code="bad_params")
    return next(iter(candidates))


@dataclass(slots=True)
class TrashListing:
    runs: list[dict[str, Any]]
    experiments: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    scenarios: list[dict[str, Any]]
    refinements: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"runs": self.runs, "experiments": self.experiments, "jobs": self.jobs, "scenarios": self.scenarios,
                "refinements": self.refinements}

    @property
    def empty(self) -> bool:
        return not (self.runs or self.experiments or self.jobs or self.scenarios or self.refinements)

    @property
    def total(self) -> int:
        return len(self.runs) + len(self.experiments) + len(self.jobs) + len(self.scenarios) + len(self.refinements)


def listing(store: Store) -> TrashListing:
    """What is in the trash, each with what deleting it for good would remove."""
    found = store.trashed()
    runs = [
        {
            "id": run.id, "label": run.label, "task_id": run.task_id, "harness": run.harness,
            "steps": run.step_count, "trashed_at": run.trashed_at,
            "goes": _goes(store, store.purge_closure(runs=[run.id])),
        }
        for run in found["runs"]
    ]
    groups: dict[str, list[Any]] = {}
    for experiment in found["experiments"]:
        groups.setdefault(experiment.group_id or experiment.id, []).append(experiment)
    experiments = [
        {
            "id": root, "kind": arms[0].kind, "run_id": arms[0].run_id, "step_idx": arms[0].step_idx,
            "arms": len(arms), "samples": sum(arm.sample_count for arm in arms), "trashed_at": arms[0].trashed_at,
            "goes": _goes(store, store.purge_closure(experiments=[arm.id for arm in arms])),
        }
        for root, arms in groups.items()
    ]
    jobs = [
        {"id": job.id, "kind": job.kind, "status": job.status, "created_at": job.created_at, "trashed_at": job.trashed_at}
        for job in found["jobs"]
    ]
    scenarios = [
        {key: entry[key] for key in ("id", "name", "trashed_at")} for entry in _trashed_scenarios(store.home)
    ]
    refinements = [
        {
            "id": found.id, "skill": found.skill, "run_id": found.run_id, "step_idx": found.step_idx,
            "task_id": found.data.get("task_id"), "status": found.status, "trashed_at": found.trashed_at,
            "proposals": len(list((found.folder / "proposals").glob("*.json"))),
            "goes": _refinement_goes(store, found),
        }
        for found in _trashed_refinements(store)
    ]
    return TrashListing(runs=runs, experiments=experiments, jobs=jobs, scenarios=scenarios, refinements=refinements)


def _trashed_refinements(store: Store) -> list[Any]:
    from tuningfork.ops.refine import list_refinements

    found = list_refinements(store, include_closed=True, include_trashed=True)
    return sorted((each for each in found if each.trashed_at is not None), key=lambda each: each.trashed_at, reverse=True)


# --- scenarios: folders in the scenarios' own `.trash/` -----------------------------------------


def _trashed_scenarios(home: Path) -> list[dict[str, Any]]:
    from tuningfork.ops.scenarios import TRASH, root

    folder = root(home) / TRASH
    entries = []
    for path in sorted(folder.iterdir() if folder.is_dir() else (), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_dir():
            # `<name>-<YYYYmmdd-HHMMSS>`, as `ops.scenarios.delete_scenario` names it.
            name = path.name.rsplit("-", 2)[0] if path.name.count("-") >= 2 else path.name
            entries.append({"id": path.name, "name": name, "trashed_at": path.stat().st_mtime, "path": path})
    return entries


def _trashed_scenario(home: Path, reference: str) -> Path:
    found = next((entry["path"] for entry in _trashed_scenarios(home) if entry["id"] == reference), None)
    if found is None:
        raise OperationError(f"no scenario {reference!r} in the trash", code="not_found")
    return found


def _restore_scenario(home: Path, reference: str) -> dict[str, Any]:
    from tuningfork.ops.scenarios import folder

    source = _trashed_scenario(home, reference)
    entry = next(entry for entry in _trashed_scenarios(home) if entry["path"] == source)
    target = folder(home, entry["name"])
    if target.exists():
        raise OperationError(
            f"a scenario called {entry['name']!r} exists already", hint="rename or delete it first", code="busy"
        )
    shutil.move(str(source), str(target))
    return {"kind": "scenario", "id": reference, "name": entry["name"], "restored": True}


def when(seconds: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds)) if seconds else "-"
