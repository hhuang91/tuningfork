"""A refinement: one skill at one checkpoint, a target declared before anything was
sampled, and every edit proposed for it — brief §7, steps 1 to 4.

A refinement is a folder in the recordings folder, `refinements/<id>/`:

    refinement.json      written once: the checkpoint, the skill as the run was given it and
                         as its source file is now, the target, the normalizers, who started it
    baseline.md          the skill as the run was given it: its own copy, kept if the run goes
    status.json          open, closed, or approved (`ops.approve`: approval.json and
                         applied.json beside it)
    trashed.json         there while it is in the trash (`ops.trash`): left out of the list,
                         taking no new work, and shown as it was
    proposals/<pid>.json each proposal — its edits, and who proposed it and why. The text is
                         a saved variant (`ops.variants`), so an A/B or a batch takes it as is
    packets/<sha>.json   what a proposer was shown, by its digest (`ops.packet`)
    proposer/            each call of the built-in proposer: its prompt and its redacted answer

Written once wherever it can be, so an agent and the built-in proposer can add to the same
refinement at once without colliding, and the store's schema does not change. Its screens
are experiments (`ops.screen`), found by the refinement's id in their settings.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store

FOLDER = "refinements"
#: There while a refinement is in the trash.
TRASHED = "trashed.json"
#: A proposal is a few small edits: what a screen at one step can attribute.
MAX_EDITS = 3
MAX_CHANGED_LINES = 10


def root(home: Path | str) -> Path:
    return Path(home) / FOLDER


def _write_once(path: Path, data: Any) -> None:
    """Create `path` with `data`, failing if it exists: two writers never overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _replace(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --- the skill a run was given, and where it came from --------------------------------------------


def run_skill_file(store: Store, run: Run) -> Path | None:
    """The SKILL.md a run was given: its batch's frozen copy, or a fork's launch folder."""
    from tuningfork.ops.experiments import recorded_skill_file

    found = recorded_skill_file(run)
    if found is None and run.launch_id:
        launch = store.get_launch(run.launch_id)
        if launch is not None and launch.skill_dir:
            candidate = Path(launch.skill_dir) / "SKILL.md"
            found = candidate if candidate.is_file() else None
    return found


def _launch_of(store: Store, run: Run) -> Any:
    """The launch that ran the task: a fork's own, else the one its source run had."""
    launch = store.get_launch(run.launch_id) if run.launch_id else None
    if launch is not None and launch.spec_path:
        return launch
    source = run.parity.get("fork_of") if run.parity else None
    if source:
        found = store.get_run(source)
        if found is not None and found.launch_id:
            return store.get_launch(found.launch_id)
    return launch


def held_out(store: Store, run: Run) -> bool:
    """Whether the run is of a task its spec holds out (`tuningfork.held_out`)."""
    launch = _launch_of(store, run)
    options = ((launch.eval if launch is not None else None) or {}).get("tuningfork") or {}
    return options.get("held_out") is True


def source_skill(store: Store, run: Run) -> Path | None:
    """The skill's own SKILL.md, in the spec's skill folder — what 6c would write back to."""
    from tuningfork.specs import load_spec

    launch = _launch_of(store, run)
    if launch is None or not launch.spec_path:
        return None
    try:
        spec = load_spec(Path(launch.spec_path), strict=False)
    except (OSError, ValueError, OperationError):
        return None
    folder = spec.resolved_skill_dir()
    candidate = folder / "SKILL.md" if folder else None
    return candidate if candidate is not None and candidate.is_file() else None


def _first_step_with(store: Store, run: Run, text: str) -> int | None:
    from tuningfork.experiment.skill import locate

    for step in store.steps(run.id):
        if locate(store.rebuild_request(step), text):
            return step.idx
    return None


def _text(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


# --- refinements ------------------------------------------------------------------------------------


@dataclass(slots=True)
class Refinement:
    id: str
    folder: Path
    data: dict[str, Any]
    status: str
    #: When it was moved to the trash (`ops.trash`); None while it is not there.
    trashed_at: float | None = None

    @property
    def run_id(self) -> str:
        return self.data["run"]

    @property
    def step_idx(self) -> int:
        return self.data["step"]

    @property
    def step_tools(self) -> list[str] | None:
        """The tools the step was named by, when it was ("write_file", or
        "write_file|execute_code"): the same step of another run is then that run's own
        first call of one, where a screen draws from several checkpoints."""
        from tuningfork.ops.experiments import step_tools

        named = self.data.get("step_named")
        return step_tools(named) if isinstance(named, str) else None

    @property
    def step_reference(self) -> str | None:
        """The step as it was named, when by what the run did there — a tool, a file's first
        change (`changed:<path>`), a finding's making (`made:<words>`): another run's same
        step is then its own such step. None for a step named by its index."""
        named = self.data.get("step_named")
        return named if isinstance(named, str) else None

    @property
    def target(self) -> dict[str, Any]:
        return self.data["target"]

    @property
    def normalizers(self) -> list[str]:
        return list(self.data.get("normalizers") or [])

    @property
    def skill(self) -> str:
        return self.data["skill"]

    def baseline_text(self) -> str:
        return _text(self.folder / "baseline.md")

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, **self.data, "status": self.status, "trashed_at": self.trashed_at}


def _load(folder: Path) -> Refinement:
    data = _read(folder / "refinement.json")
    status_file = folder / "status.json"
    status = _read(status_file).get("status", "open") if status_file.is_file() else "open"
    trashed = folder / TRASHED
    trashed_at = _read(trashed).get("trashed_at") if trashed.is_file() else None
    return Refinement(id=folder.name, folder=folder, data=data, status=status, trashed_at=trashed_at)


def refuse_closed(refinement: Refinement) -> None:
    """Refuse new work on a refinement that takes none: one in the trash — whatever it made
    would be deleted with it — or one closed, or approved."""
    if refinement.trashed_at is not None:
        raise OperationError(
            f"refinement {refinement.id[:8]} is in the trash",
            hint=f"restore it first: `tuningfork trash restore {refinement.id[:8]}`",
            code="trashed", field="refinement",
        )
    if refinement.status != "open":
        raise OperationError(f"refinement {refinement.id[:8]} is {refinement.status}", code="closed", field="refinement")


def get(store: Store, reference: str) -> Refinement:
    """A refinement by its id or an unambiguous prefix of it."""
    base = root(store.home)
    if reference and (base / reference / "refinement.json").is_file():
        return _load(base / reference)
    matches = [path for path in base.glob(f"{reference}*") if (path / "refinement.json").is_file()] if reference and base.is_dir() else []
    if len(matches) != 1:
        raise OperationError(f"no single refinement matching {reference!r}", code="not_found", field="refinement")
    return _load(matches[0])


def list_refinements(
    store: Store, *, run: str | None = None, include_closed: bool = False, include_trashed: bool = False,
) -> list[Refinement]:
    """Refinements, newest first: of one run (by its id or a prefix of it), or every one. Those
    in the trash are left out, as the trash leaves anything out of every list."""
    base = root(store.home)
    if not base.is_dir():
        return []
    found = []
    for folder in base.iterdir():
        if not (folder / "refinement.json").is_file():
            continue
        refinement = _load(folder)
        if run is not None and not refinement.run_id.startswith(run):
            continue
        if refinement.status == "closed" and not include_closed:
            continue
        if refinement.trashed_at is not None and not include_trashed:
            continue
        found.append(refinement)
    return sorted(found, key=lambda each: each.data.get("created_at", 0), reverse=True)


def create(
    store: Store,
    run: str,
    step: str,
    *,
    target: str | None = None,
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    target_check: bool = False,
    created_by: dict[str, Any] | None = None,
) -> Refinement:
    """Start refining the skill a run was given, at one of its steps, toward a target
    declared now — before any proposal is made or sampled. The same checkpoint, target and
    skill give back the open refinement already there. `target_check` makes the task's own
    check the target, for a failure only the check can see: its screens run forks to the
    end."""
    from tuningfork.experiment import resolve_target
    from tuningfork.experiment.sampling import CheckpointError
    from tuningfork.experiment.skill import diagnose, locate, text_sha256
    from tuningfork.ops.experiments import load
    from tuningfork.ops.trash import refuse_trashed
    from tuningfork.ops.variants import skill_name

    declared = [value for value in (target, target_match, target_rules, target_check) if value]
    if len(declared) != 1:
        raise OperationError(
            "declare what the model should do at this step, before anything is proposed or "
            "sampled: a behaviour's letter from this step's resample (target), a pattern "
            "(target_match), conditions in words (target_rules), or the task's check "
            "(target_check) — exactly one",
            code="bad_params", field="target",
        )
    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    if held_out(store, checkpoint.run):
        raise OperationError(
            "this run is of a held-out task: held-out tasks judge an edit, they never shape it",
            hint="refine from a run of one of the skill's training tasks",
            code="held_out", field="run",
        )
    found = run_skill_file(store, checkpoint.run)
    if found is None:
        raise OperationError(
            f"run {checkpoint.run.id[:8]} kept no copy of the skill it was given",
            hint="refine from a run tuningfork launched: a batch's, or a fork of one",
            field="run",
        )
    baseline = _text(found)
    if not locate(checkpoint.request, baseline):
        first = _first_step_with(store, checkpoint.run, baseline)
        raise OperationError(
            f"the skill's text is not in the context at step {checkpoint.step.idx}: no edit of it "
            "could change what the model saw there",
            hint=(f"step {first} is the first whose context has it" if first is not None
                  else "no step of this run had the skill in its context"),
            detail=diagnose(checkpoint.request, baseline),
            code="no_skill_here", field="step",
        )
    if target_check:
        # Judged by forks run to the end: the run must come from a task with a check.
        from tuningfork.analysis.distribution import target_check as check_target

        launch = _launch_of(store, checkpoint.run)
        check = (((launch.eval or {}).get("tuningfork") or {}).get("check")) if launch is not None else None
        if not check:
            raise OperationError(
                f"run {checkpoint.run.id[:8]}'s task has no check to judge forks by",
                hint="the task's check is a target only for a run launched from a spec whose eval names one",
                code="bad_params", field="target_check",
            )
        declared_target, bound = check_target(), None
    else:
        try:
            declared_target, bound = resolve_target(
                store, checkpoint, letter=target, pattern=target_match, rules=target_rules
            )
        except (ValueError, CheckpointError) as error:
            raise OperationError(str(error), code="bad_params", field="target") from None
    baseline_sha = text_sha256(baseline)
    source = source_skill(store, checkpoint.run)
    source_data = None
    if source is not None:
        source_sha = text_sha256(_text(source))
        source_data = {"path": str(source), "sha256": source_sha, "matches": source_sha == baseline_sha}
    for existing in list_refinements(store, run=checkpoint.run.id):
        if (existing.step_idx == checkpoint.step.idx and existing.target == declared_target.spec
                and existing.data["baseline"]["sha256"] == baseline_sha):
            return existing
    from tuningfork.ops.experiments import named_by_doing

    rid = uuid.uuid4().hex
    folder = root(store.home) / rid
    data = {
        "run": checkpoint.run.id,
        "step": checkpoint.step.idx,
        # The step as it was named, when by what the run did there: how the same step is found
        # in other runs.
        "step_named": step if named_by_doing(step) else None,
        "task_id": checkpoint.run.task_id,
        "skill": skill_name(checkpoint.run, found),
        "baseline": {"path": str(found), "sha256": baseline_sha},
        "source": source_data,
        "target": declared_target.spec,
        "normalizers": list(bound or []),
        "created_by": created_by or {"via": "person"},
        "created_at": time.time(),
    }
    _write_once(folder / "refinement.json", data)
    with open(folder / "baseline.md", "w", encoding="utf-8", newline="") as handle:
        handle.write(baseline)
    _replace(folder / "status.json", {"status": "open"})
    return _load(folder)


def close(store: Store, reference: str) -> Refinement:
    refinement = get(store, reference)
    if refinement.status == "approved":
        return refinement  # it takes nothing more already, and stays approved
    _replace(refinement.folder / "status.json", {"status": "closed", "closed_at": time.time()})
    return _load(refinement.folder)


# --- proposals --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Proposal:
    id: str
    data: dict[str, Any]
    #: Given back for a text already proposed: the proposer was added to the one there.
    duplicate: bool = False

    def text(self) -> str:
        return _text(Path(self.data["variant_path"]))

    def to_json(self) -> dict[str, Any]:
        return {**self.data, "duplicate": self.duplicate}


def proposals(store: Store, reference: str) -> list[Proposal]:
    refinement = get(store, reference)
    folder = refinement.folder / "proposals"
    found = [Proposal(path.stem, _read(path)) for path in folder.glob("*.json")] if folder.is_dir() else []
    return sorted(found, key=lambda proposal: proposal.data.get("created_at", 0))


def add_proposal(
    store: Store,
    reference: str,
    *,
    edits: Sequence[Any] | None = None,
    text: str | None = None,
    rationale: str,
    proposer: dict[str, Any],
    packet: str | None = None,
    prompt: str | None = None,
    strict: bool = True,
) -> Proposal:
    """A proposed edit of the refinement's skill: a few edits (`edits`: a line replaced by its
    number, lines added after one, or exact text replaced — each resolved to an exact
    replacement, and recorded so), or — from a person — a whole text. Refused unless it can
    be screened at the checkpoint.

    `strict` holds a proposal to a few small edits; a person's own text only warns."""
    from tuningfork.experiment.skill import NothingToCompare, split_frontmatter, substitute, text_sha256
    from tuningfork.ops.diff import apply_resolved, line_diff, resolve_edits
    from tuningfork.ops.experiments import load
    from tuningfork.ops.variants import save_variant

    refinement = get(store, reference)
    refuse_closed(refinement)
    if (edits is None) == (text is None):
        raise OperationError(
            "give the edits ({line, new}, {after, insert} or {old, new}), or a whole text",
            code="bad_params", field="edits",
        )
    if not isinstance(rationale, str) or not rationale.strip():
        raise OperationError("say why: a rationale that cites the evidence", code="bad_params", field="rationale")
    baseline = refinement.baseline_text()
    if edits is not None:
        if len(edits) > MAX_EDITS:
            raise OperationError(
                f"at most {MAX_EDITS} edits in one proposal: small changes are what a screen can attribute",
                code="bad_params", field="edits",
            )
        resolved = resolve_edits(baseline, edits)
        text = apply_resolved(baseline, resolved)
        kept = [{key: found[key] for key in ("lines", "old", "new", "given")} for found in resolved]
    else:
        kept = None
    assert text is not None
    if text_sha256(text) == text_sha256(baseline):
        raise OperationError("the proposal changes nothing", code="bad_params", field="edits")
    if split_frontmatter(text)[0].strip() != split_frontmatter(baseline)[0].strip():
        raise OperationError(
            "the frontmatter is how a harness chooses the skill, which a screen at one step "
            "cannot measure: edit the skill's body",
            code="bad_params", field="edits",
        )
    diff = line_diff(baseline, text)
    warnings = []
    if diff["changed"] > MAX_CHANGED_LINES:
        if strict:
            raise OperationError(
                f"{diff['changed']} lines changed; a proposal changes at most {MAX_CHANGED_LINES}, "
                "so that a screen can say what made the difference",
                code="bad_params", field="edits",
            )
        warnings.append(f"{diff['changed']} lines changed: a screen will not say which of them mattered")
    checkpoint = load(store, refinement.run_id, str(refinement.step_idx))
    try:
        substitute(checkpoint.request, baseline, text)
    except NothingToCompare:
        raise OperationError(
            f"the proposal changes nothing the model sees at step {refinement.step_idx}",
            code="bad_params", field="edits",
        ) from None
    pid = text_sha256(text)[:12]
    entry = {
        **proposer,
        "rationale": rationale.strip(),
        "packet": packet,
        "prompt": prompt,
        "edits": kept,
        "at": time.time(),
    }
    name = f"r{refinement.id[:8]}-{pid[:8]}"
    path = save_variant(store.home, refinement.skill, name, text)
    record = refinement.folder / "proposals" / f"{pid}.json"
    data = {
        "id": pid,
        "refinement": refinement.id,
        "variant": name,
        "variant_path": str(path),
        "changed": diff["changed"],
        "added": diff["added"],
        "removed": diff["removed"],
        "warnings": warnings,
        "proposers": [entry],
        "created_at": entry["at"],
    }
    try:
        _write_once(record, data)
    except FileExistsError:
        # The same text proposed again, by another proposer or the same one: one proposal,
        # every proposer kept with it.
        data = _read(record)
        data["proposers"].append(entry)
        _replace(record, data)
        return Proposal(pid, data, duplicate=True)
    return Proposal(pid, data)


def proposal(store: Store, reference: str, pid: str) -> Proposal:
    found = [each for each in proposals(store, reference) if each.id.startswith(pid)] if pid else []
    if len(found) != 1:
        raise OperationError(f"no single proposal matching {pid!r}", code="not_found", field="proposals")
    return found[0]


# --- screening proposals -----------------------------------------------------------------------------


def prepare_refinement_screen(
    store: Store,
    reference: str,
    proposals: Sequence[str],
    *,
    base_url: str | None = None,
    alpha: float = 0.05,
    to_end: bool = False,
    proxy: Any = None,
    spec: Path | None = None,
    workdir: Path | None = None,
    checkpoints: int = 1,
    quick: bool = False,
    fresh_original: bool = False,
    **plan: Any,
) -> Any:
    """A screen of some of a refinement's proposals at its checkpoint, toward its target:
    the family the correction counts is the proposals named here, fixed before sampling.
    With `to_end`, each arm's samples are forks run to the end and the task's check decides:
    the only screen for a refinement whose target is the check, and open to any other, whose
    own target then stands aside for this screen — the record says which measured. Its
    forks are drawn from `checkpoints` runs — this one, and others of its task at the same
    step, passing and failing by turns — so an effect of one prefix shows up as one. The
    original's forks its earlier screens drew there count again (`fresh_original` draws them
    anew); `quick` is a quick look, the proposals' forks alone and no verdict."""
    from tuningfork.ops.screen import prepare_screen, prepare_screen_to_end

    refinement = get(store, reference)
    refuse_closed(refinement)
    if not proposals:
        raise OperationError(
            "name the proposals to screen: the family the correction counts is fixed before sampling",
            code="bad_params", field="proposals",
        )
    chosen: list[Proposal] = []
    for each in proposals:
        found = proposal(store, refinement.id, each)
        if found.id not in [already.id for already in chosen]:
            chosen.append(found)
    arms = [(found.id[:8], found.text(), {"proposal": found.id, "variant_path": found.data["variant_path"]})
            for found in chosen]
    if refinement.target.get("kind") == "check" and not to_end:
        raise OperationError(
            "this refinement's target is the task's check, which no answer at the step can be judged by",
            hint="screen it to the end — each arm's samples forks run through the harness, the check "
                 "deciding: `tuningfork refine screen <id> --to-end`, or start_screen_to_end",
            code="to_end_only", field="refinement",
        )
    if to_end:
        return prepare_screen_to_end(
            store, refinement.run_id, str(refinement.step_idx), original_text=refinement.baseline_text(),
            arms=arms, alpha=alpha, base_url=base_url, spec=spec, workdir=workdir, proxy=proxy,
            refinement=refinement.id, checkpoints=checkpoints, aligned=refinement.step_reference,
            quick=quick, fresh_original=fresh_original, **plan,
        )
    if quick or fresh_original:
        raise OperationError(
            "a quick look, and the original's forks drawn anew, are for a screen to the end: a screen at the step "
            "draws its own original arm, one call a sample",
            code="bad_params", field="quick" if quick else "fresh_original",
        )
    if checkpoints != 1:
        raise OperationError(
            "a screen at the step samples the refinement's own checkpoint: several are for a screen to the end",
            code="bad_params", field="checkpoints",
        )
    return prepare_screen(
        store, refinement.run_id, str(refinement.step_idx),
        original_text=refinement.baseline_text(), arms=arms, target_spec=refinement.target,
        normalizers=refinement.normalizers, alpha=alpha, base_url=base_url,
        refinement=refinement.id, **plan,
    )


# --- what a refinement holds, for a front end --------------------------------------------------------


def screen_experiments(store: Store, refinement: Refinement, *, include_trashed: bool = False) -> list[Any]:
    """Every experiment of the refinement's screens: those at its own checkpoint, found by
    its id in their settings, and the rest of each one's group — a screen to the end drawn
    from several checkpoints keeps an experiment at each, under other runs' ids."""
    found: dict[str, Any] = {}
    for experiment in store.experiments(
        run_id=refinement.run_id, step_idx=refinement.step_idx, kind="screen", limit=100_000,
        include_trashed=include_trashed,
    ):
        if experiment.settings.get("refinement") != refinement.id:
            continue
        found.setdefault(experiment.id, experiment)
        if experiment.group_id:
            for member in store.experiment_group(experiment.group_id):
                if include_trashed or member.trashed_at is None:
                    found.setdefault(member.id, member)
    return list(found.values())


def screens(store: Store, reference: str) -> list[dict[str, Any]]:
    """The refinement's screens, newest first, each reported again from the store."""
    from tuningfork.ops.inspect import overall_status, arm_status
    from tuningfork.ops.screen import screen_result

    refinement = get(store, reference)
    groups: dict[str, list[Any]] = {}
    for experiment in screen_experiments(store, refinement):
        if experiment.group_id:
            groups.setdefault(experiment.group_id, []).append(experiment)
    found = []
    for group_id, arms in groups.items():
        result = screen_result(store, group_id)
        planned = (arms[0].settings.get("screen") or {})
        found.append({
            "group_id": group_id,
            "experiment": next((arm.id for arm in arms if arm.arm == "original"), arms[0].id),
            "created_at": min(arm.created_at for arm in arms),
            "status": overall_status(arm_status(arm) for arm in arms),
            "proposals": planned.get("proposals") or {},
            # What it measured: a target at the step, or the task's check over forks to the end.
            "reach": result.report.reach,
            "measure": result.report.target,
            # A quick look's rates are a lead, never a verdict.
            "quick": result.report.quick,
            "report": result.report.to_json(),
        })
    return sorted(found, key=lambda screen: screen["created_at"], reverse=True)


def citations(
    store: Store, *, runs: Sequence[str] = (), experiments: Sequence[str] = (), excluding: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """The refinements that cite any of these runs or experiments: what deleting them for good
    takes from each, said before it is done. An approved refinement keeps, in its approval,
    the evidence it stood on; any other loses what is deleted. `excluding`: refinements going
    with them, which lose nothing they would still have."""
    from tuningfork.ops.validate import validations

    runs, experiments = set(runs), set(experiments)
    base = root(store.home)
    if not (runs or experiments) or not base.is_dir():
        return []
    found = []
    for folder in sorted(base.iterdir()):
        if not (folder / "refinement.json").is_file() or folder.name in excluding:
            continue
        refinement = _load(folder)
        cites = ["its checkpoint"] if refinement.run_id in runs else []
        screened = {experiment.id for experiment in screen_experiments(store, refinement, include_trashed=True)}
        if screened & experiments:
            cites.append("its screens")
        elif any(experiment.run_id in runs for experiment in screen_experiments(store, refinement, include_trashed=True)):
            cites.append("a checkpoint its screens drew from")
        for record in validations(store, refinement.id):
            ran = {verdict.get("run_id") for part in ("forward", "training") for arm in ("original", "edit")
                   for verdict in (record.get(part) or {}).get(arm) or []}
            forks = set(((record.get("forward") or {}).get("experiments") or {}).values())
            if ran & runs or forks & experiments:
                cites.append(f"validation {record['id'][:8]}")
        if not cites:
            continue
        approved = refinement.status == "approved"
        consequence = ("its approval keeps the evidence it stood on" if approved
                       else "it loses that evidence" + (", and can no longer screen or validate from its checkpoint"
                                                        if "its checkpoint" in cites else ""))
        where = ", in the trash" if refinement.trashed_at is not None else ""
        found.append({
            "refinement": refinement.id, "skill": refinement.skill, "status": refinement.status, "cites": cites,
            "kept": approved, "trashed": refinement.trashed_at is not None,
            "says": f"refinement {refinement.id[:8]} ({refinement.skill}, {refinement.status}{where}) cites "
                    f"{', '.join(cites)}: {consequence}",
        })
    return found


@dataclass(slots=True)
class Made:
    """What a refinement made: what deleting it for good takes with it (`ops.trash`)."""

    #: Its screens, and its validations' forks from the checkpoint to the end — those in the
    #: trash too.
    experiments: list[str]
    #: Its validations' fresh runs of the training task.
    runs: list[str]
    #: Its proposals' texts, saved as variants (`ops.variants`), inside the recordings folder.
    variants: list[Path]


def made_by(store: Store, refinement: Refinement) -> Made:
    """What a refinement made, found by its id: its screens, its validations' forks and
    training runs, and its proposals' variants. Never its checkpoint — the run it studies —
    and never what an approval wrote beside the skill's own folder: those are not its own."""
    from tuningfork.ops.validate import validations

    experiments = {experiment.id for experiment in store.experiments(
        run_id=refinement.run_id, step_idx=refinement.step_idx, limit=100_000, include_trashed=True,
    ) if experiment.settings.get("refinement") == refinement.id}
    # A screen to the end drawn from several checkpoints: its experiments at the other runs.
    experiments.update(experiment.id for experiment in screen_experiments(store, refinement, include_trashed=True))
    runs: set[str] = set()
    for record in validations(store, refinement.id):
        experiments.update(((record.get("forward") or {}).get("experiments") or {}).values())
        for ids in ((record.get("training") or {}).get("runs") or {}).values():
            runs.update(ids or [])
    home = store.home.resolve()
    variants = []
    for proposal_data in (_read(path) for path in (refinement.folder / "proposals").glob("*.json")):
        path = Path(proposal_data.get("variant_path") or "")
        try:
            path.resolve().relative_to(home)  # its own saved text, never a file elsewhere
        except ValueError:
            continue
        if path.is_file():
            variants.append(path)
    return Made(experiments=sorted(experiments), runs=sorted(runs), variants=sorted(variants))


def view(store: Store, reference: str, *, held_out: bool = False) -> dict[str, Any]:
    """A refinement with everything it holds: its proposals with their line diffs, its
    screens with their verdicts, and its validations.

    A validation's held-out verdicts, and the gate they feed, are the person's: shown only
    with `held_out`, for their own views. An agent shown them could fit its edits to them."""
    from tuningfork.ops.approve import state as approval_state
    from tuningfork.ops.diff import line_diff
    from tuningfork.ops.validate import validation_views

    refinement = get(store, reference)
    baseline = refinement.baseline_text()
    found_screens = screens(store, refinement.id)
    verdicts: dict[str, list[dict[str, Any]]] = {}
    for screen in found_screens:
        report = screen["report"]
        for arm in report["arms"]:
            pid = screen["proposals"].get(arm["name"])
            if pid:
                verdicts.setdefault(pid, []).append({
                    "screen": screen["experiment"], "arm": arm["name"], "target": arm["target"],
                    "reach": screen.get("reach"), "quick": bool(report.get("quick")),
                    "baseline": report["baseline"]["target"], "adjusted_p": arm.get("adjusted_p"),
                    "survives": arm.get("survives"), "worse": arm.get("worse"), "complete": report["complete"],
                })
    items = []
    for each in proposals(store, refinement.id):
        items.append({**each.to_json(), "diff": line_diff(baseline, each.text())["hunks"],
                      "screens": verdicts.get(each.id, [])})
    packets = sorted(path.stem for path in (refinement.folder / "packets").glob("*.json")) \
        if (refinement.folder / "packets").is_dir() else []
    return {
        "refinement": refinement.to_json(),
        "baseline": baseline,
        "proposals": items,
        "screens": found_screens,
        "validations": validation_views(store, refinement, held_out=held_out),
        # Where an approval stands, and where it wrote: never its evidence.
        "approval": approval_state(store, refinement.id),
        "packets": packets,
    }
