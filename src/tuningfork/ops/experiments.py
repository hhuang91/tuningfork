"""Sampling a recorded step: resample it, or A/B a wording change at it.

Each comes as `prepare_*`, which checks everything a person should see before any sample is
drawn and has no side effects, and `execute_*`, which draws the samples. A front end shows
the prepared plan, then runs it.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress
from tuningfork.ops.trash import refuse_trashed
from tuningfork.store.db import Run, Step, Store


def default_upstream() -> str:
    """Where experiments sample from: the inference server itself, not the proxy."""
    return os.environ.get("TUNINGFORK_UPSTREAM", "http://127.0.0.1:8081")


def resolve_run(store: Store, reference: str) -> Run:
    run = store.get_run(reference)
    if run is None:
        raise OperationError(f"no run matching {reference!r}")
    return run


def tools_called(store: Store, step: Step) -> list[str]:
    """The tools the model called at `step`, in the order it called them."""
    message = ((store.response(step) or {}).get("choices") or [{}])[0].get("message") or {}
    return [(call.get("function") or {}).get("name") for call in message.get("tool_calls") or []]


#: Joins tool names in a step reference: "write_file|execute_code" is the first step whose
#: model called either — the step where a run wrote, whichever way it wrote.
EITHER = "|"
#: A step named by the call that first changed a file of the workspace — "changed:inventory.json",
#: or a glob, "changed:*.json" — found from the run's snapshots (`ops.inspect.first_change`).
CHANGED = "changed:"
#: A step named by the call that first made a finding its task's check prints — "made:in stock" —
#: found from the check run along the run (`ops.checks.check_along`, `first_made`).
MADE = "made:"


def step_tools(reference: str) -> list[str] | None:
    """The tools a step reference names, as `step_index` reads it; None for an index, or for a
    step named by a file's first change or a finding's making."""
    if reference.isdigit() or reference.startswith((CHANGED, MADE)):
        return None
    return [name for name in reference.split(EITHER) if name]


def named_by_doing(reference: str) -> bool:
    """Whether a step reference names the step by what the run did there — a tool it called, a
    file its call first changed, a finding its call first made — so that another run's same step
    is that run's own such step, not the same index."""
    return not reference.isdigit()


def first_call(store: Store, run: Run, tool: str | Sequence[str]) -> int | None:
    """The first step of `run` whose model called `tool` — any of them, given several."""
    wanted = {tool} if isinstance(tool, str) else set(tool)
    for step in store.steps(run.id):
        if wanted.intersection(tools_called(store, step)):
            return step.idx
    return None


def find_step(store: Store, run: Run, reference: str | Sequence[str]) -> int | None:
    """The step `reference` names in `run`, by what the run did there, or None when it did no such
    thing; an index is itself. A sequence of tool names is the first call of any of them."""
    if not isinstance(reference, str):
        return first_call(store, run, reference)
    if reference.isdigit():
        return int(reference)
    if reference.startswith(CHANGED):
        from tuningfork.ops.inspect import first_change

        return first_change(store, run, reference[len(CHANGED):] or "*")
    if reference.startswith(MADE):
        from tuningfork.ops.checks import first_made

        return first_made(run, reference[len(MADE):])
    tools = step_tools(reference)
    return first_call(store, run, tools) if tools else None


def not_done(reference: str | Sequence[str]) -> str:
    """What a run did not do, when `find_step` finds no step of it: "never called write_file"."""
    if not isinstance(reference, str):
        return f"never called {' or '.join(reference)}"
    if reference.startswith(CHANGED):
        return f"never changed {reference[len(CHANGED):] or 'a file'}"
    if reference.startswith(MADE):
        return f"made no finding with {reference[len(MADE):]!r} (or its check was not run along it)"
    return f"never called {' or '.join(step_tools(reference) or [reference])}"


def step_index(store: Store, run: Run, reference: str) -> int:
    """A step of `run` by its index, as the first step whose model called that tool — or any
    of several tools, joined by `|` — or as the step whose call first changed a file
    (`changed:<path>`) or first made a finding of the task's check (`made:<words>`).

    "The write" is how a person thinks of a step; its number depends on the path the run
    took — a run that listed the directory first writes a step later. And not every run
    writes the same way: one calls write_file, another a script through execute_code, so
    "write_file|execute_code" names the write in both. Where every run made the same mistake
    with one call among several — a delivery booked a second time — "changed:inventory.json"
    names that call in each run, whatever tool made it.
    """
    if reference.isdigit():
        return int(reference)
    if reference.startswith(CHANGED):
        found = find_step(store, run, reference)
        if found is None:
            raise OperationError(
                f"run {run.id[:8]} {not_done(reference)}: no call of it changed that in its workspace",
                hint="show_files where=end lists what the run changed (`tuningfork files <run> end`)",
            )
        return found
    if reference.startswith(MADE):
        if not reference[len(MADE):].strip():
            raise OperationError("made: names a finding by words it contains: made:in stock", code="bad_params",
                                 field="step")
        found = find_step(store, run, reference)
        if found is None:
            raise OperationError(
                f"run {run.id[:8]} {not_done(reference)}",
                hint="`tuningfork check-along --run <run>` runs the task's check along it; the tally then says "
                     "which call made each finding",
            )
        return found
    tools = step_tools(reference)
    if not tools or any(ch.isspace() for ch in reference):
        raise OperationError(
            "a step is its number, or the name of the tool the model called there (write_file; "
            "write_file|execute_code for whichever came first), or changed:<path> or made:<words>: "
            f"not {reference!r}",
            code="bad_params", field="step",
        )
    found = first_call(store, run, tools)
    if found is None:
        raise OperationError(
            f"run {run.id[:8]} has no step where the model called {' or '.join(tools)}",
            hint=f"`tuningfork runs --called {tools[0]}` lists runs that do",
        )
    return found


def recorded_skill_file(run: Run) -> Path | None:
    """The SKILL.md a batch run was given: the batch's frozen copy, if it still exists."""
    folder = run.parity.get("skill_snapshot_dir")
    candidate = Path(folder) / "SKILL.md" if folder else None
    return candidate if candidate and candidate.is_file() else None


def kept_skills(run: Run) -> dict[str, Path]:
    """Each skill a run kept a copy of, by name, with its SKILL.md: every one an imported chat
    showed (`parity.skill_dirs`), else the one a batch run was given."""
    from tuningfork.ops.variants import skill_name

    kept = {
        name: Path(folder) / "SKILL.md"
        for name, folder in (run.parity.get("skill_dirs") or {}).items()
        if (Path(folder) / "SKILL.md").is_file()
    }
    own = recorded_skill_file(run)
    if not kept and own is not None:
        kept[(run.parity.get("imported") or {}).get("skill") or skill_name(run, own)] = own
    return kept


def study_skill(run: Run, skill: str | None = None) -> tuple[str, Path] | None:
    """The skill to study in `run`, by name, and its SKILL.md: the one named, else the run's own —
    a batch run's, the one an import was told to study, or the only one it kept. None where it
    kept none; refused where it kept several and none is named, or none by the name given."""
    kept = kept_skills(run)
    if skill:
        found = kept.get(skill)
        if found is None:
            raise OperationError(
                f"run {run.id[:8]} kept no skill called {skill!r}",
                hint=f"the skills it kept: {', '.join(kept) or 'none'}", field="skill",
            )
        return skill, found
    own = recorded_skill_file(run)
    if own is not None:
        return next((name for name, path in kept.items() if path == own), own.parent.name), own
    if len(kept) > 1:
        raise OperationError(
            f"run {run.id[:8]} kept {len(kept)} skills: name the one to study",
            hint=f"the skills it kept: {', '.join(kept)}", code="choose_skill", field="skill",
        )
    return next(iter(kept.items()), None)


def make_plan(
    *,
    n: int = 20,
    concurrency: int = 1,
    max_tokens: int = 4096,
    seed_base: int | None = None,
    prompt_cache: bool = False,
    normalizers: Sequence[str] = (),
    overrides: dict[str, Any] | None = None,
    logprobs: int | None = None,
):
    from tuningfork.experiment import SamplingPlan

    try:
        return SamplingPlan(
            n=n,
            concurrency=concurrency,
            max_tokens=max_tokens,
            seed_base=seed_base,
            prompt_cache=prompt_cache,
            normalizers=list(normalizers),
            overrides={k: v for k, v in (overrides or {}).items() if v is not None},
            logprobs=logprobs,
        )
    except ValueError as exc:
        raise OperationError(str(exc)) from None


def load(store: Store, run_reference: str, step_reference: str):
    """The checkpoint a run and step name, whichever way each is named."""
    from tuningfork.experiment import CheckpointError, load_checkpoint

    run = resolve_run(store, run_reference)
    index = step_index(store, run, step_reference)
    try:
        return load_checkpoint(store, run_reference, index)
    except CheckpointError as exc:
        raise OperationError(str(exc)) from None


@dataclass(slots=True)
class Estimate:
    """How long sampling may take, from how long the recorded step took."""

    recorded_seconds: float
    #: The recorded step's time for every sample: an upper bound when samples reuse the
    #: server's cache of the prompt. None when each processes its whole prompt instead —
    #: work the recorded step, mostly cached, did not do — so its time bounds nothing.
    worst_minutes: float | None
    #: Each sample processes its whole prompt, this many tokens when the step recorded it.
    fresh: bool = False
    prompt_tokens: int | None = None
    #: Whether the recorded step read most of its prompt from the server's cache: on a server
    #: started with --no-cache-prompt it read none, and its time is a whole prompt's.
    cached: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "recorded_seconds": round(self.recorded_seconds, 2),
            "worst_minutes": round(self.worst_minutes, 2) if self.worst_minutes is not None else None,
            "fresh": self.fresh,
            "prompt_tokens": self.prompt_tokens,
            "cached": self.cached,
        }


def estimate(checkpoint, plan, *, arms: int = 1) -> Estimate | None:
    recorded_ms = (checkpoint.step.timings or {}).get("total_ms")
    if not recorded_ms:
        return None
    from tuningfork.backends.llamacpp import CACHE_PROMPT_ENDPOINTS

    wire = getattr(checkpoint, "wire", None)
    usage = checkpoint.step.usage or {}
    tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cached = isinstance(read, int) and isinstance(tokens, int) and tokens > 0 and read > tokens / 2
    if not getattr(plan, "prompt_cache", True) and getattr(wire, "endpoint", None) in CACHE_PROMPT_ENDPOINTS:
        return Estimate(recorded_seconds=recorded_ms / 1000, worst_minutes=None, fresh=True,
                        prompt_tokens=tokens if isinstance(tokens, int) else None, cached=cached)
    # The recorded step's time for every sample: an upper bound when samples reuse a cached
    # prefix, and about right when the server processes every prompt whole.
    worst = recorded_ms / 1000 * plan.n * arms / plan.concurrency
    return Estimate(recorded_seconds=recorded_ms / 1000, worst_minutes=worst / 60, cached=cached)


# --- resample -----------------------------------------------------------------------------


@dataclass(slots=True)
class ResamplePlan:
    checkpoint: Any
    plan: Any
    base_url: str

    @property
    def estimate(self) -> Estimate | None:
        return estimate(self.checkpoint, self.plan)

    def to_json(self) -> dict[str, Any]:
        guess = self.estimate
        return {
            "checkpoint": self.checkpoint.to_json(),
            "sampling": self.plan.as_settings(),
            "base_url": self.base_url,
            "estimate": guess.to_json() if guess else None,
        }


def experiment_json(experiment: Any) -> dict[str, Any]:
    """An experiment as a result names it: enough to find and link it."""
    return {
        "id": experiment.id,
        "kind": experiment.kind,
        "arm": experiment.arm,
        "group_id": experiment.group_id,
        "run_id": experiment.run_id,
        "step_idx": experiment.step_idx,
        "hold_id": experiment.hold_id,
        "samples": experiment.sample_count,
    }


@dataclass(slots=True)
class ResampleResult:
    experiment: Any
    distribution: Any
    elapsed: float

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment": experiment_json(self.experiment),
            "distribution": self.distribution.to_json(),
            "elapsed": round(self.elapsed, 2),
        }


def prepare_resample(
    store: Store,
    run: str,
    step: str,
    *,
    base_url: str | None = None,
    **plan: Any,
) -> ResamplePlan:
    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    from tuningfork.endpoints import check_endpoint

    sampling = make_plan(**plan)
    if sampling.logprobs is not None and checkpoint.wire.name != "openai-chat":
        raise OperationError(
            f"step {checkpoint.step.idx} was a {checkpoint.wire.endpoint} call, and that API has "
            "no logprobs",
            hint=f"score the answer the run gave instead: `tuningfork tokens {checkpoint.run.id[:8]} "
            f"{checkpoint.step.idx} --score`",
            field="logprobs",
        )
    return ResamplePlan(
        checkpoint=checkpoint,
        plan=sampling,
        base_url=check_endpoint(base_url or default_upstream(), home=store.home, purpose="sampling"),
    )


async def execute_resample(
    store: Store,
    prepared: ResamplePlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ResampleResult:
    from tuningfork.experiment import experiment_distribution, run_resample

    started = time.monotonic()
    created: list[Any] = []
    try:
        experiment = await _cancellable(
            run_resample(
                store, prepared.checkpoint, prepared.plan,
                base_url=prepared.base_url, progress=_tracking(created, progress),
            ),
            cancel,
        )
    except Cancelled as stopped:
        _stopped(store, created, stopped.reason)
        partial = (
            ResampleResult(
                experiment=created[0],
                distribution=experiment_distribution(store, created[0]),
                elapsed=time.monotonic() - started,
            )
            if created
            else None
        )
        raise Cancelled(stopped.reason, partial=partial) from None
    return ResampleResult(
        experiment=experiment,
        distribution=experiment_distribution(store, experiment),
        elapsed=time.monotonic() - started,
    )


# --- A/B ----------------------------------------------------------------------------------


@dataclass(slots=True)
class StepTargets:
    """What an A/B at a step can declare as its target before sampling: a behaviour of the
    latest resample of that step, by its letter."""

    checkpoint: Any
    resample: Any
    behaviours: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint.to_json(),
            "resample": experiment_json(self.resample) if self.resample else None,
            # A letter names a behaviour only under the normalizers it was clustered with.
            "normalizers": self.resample.normalizers if self.resample else [],
            "behaviours": self.behaviours.to_json() if self.behaviours else None,
        }


def step_targets(store: Store, run: str, step: str) -> StepTargets:
    from tuningfork.experiment import experiment_distribution
    from tuningfork.experiment.sampling import step_resample

    checkpoint = load(store, run, step)
    resample = step_resample(store, checkpoint)
    behaviours = experiment_distribution(store, resample) if resample else None
    return StepTargets(checkpoint=checkpoint, resample=resample, behaviours=behaviours)


@dataclass(slots=True)
class RecordedSkill:
    """The skill file a run was given, as its batch kept it: the baseline of a variant."""

    run_id: str
    skill: str
    path: Path | None
    text: str | None
    #: Every skill the run kept, by name: more than one for an imported chat that showed several.
    kept: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        from tuningfork.experiment.skill import text_sha256

        return {
            "run_id": self.run_id,
            "skill": self.skill,
            "kept": self.kept,
            "path": str(self.path) if self.path else None,
            "text": self.text,
            "sha256": text_sha256(self.text) if self.text is not None else None,
        }


def recorded_skill(store: Store, run: str, skill: str | None = None) -> RecordedSkill:
    """The skill a run kept, to vary: `skill` where it kept several (`study_skill`). Not yet
    named where it kept several, it says which (`kept`) and gives no text."""
    from tuningfork.ops.variants import skill_name

    found = resolve_run(store, run)
    try:
        studied = study_skill(found, skill)
    except OperationError as error:
        if error.code != "choose_skill":
            raise
        studied = None
    path = studied[1] if studied is not None else None
    return RecordedSkill(
        run_id=found.id,
        skill=skill_name(found, path),
        path=path,
        text=path.read_text(encoding="utf-8") if path else None,
        kept=list(kept_skills(found)),
    )


@dataclass(slots=True)
class ABPlan:
    checkpoint: Any
    plan: Any
    base_url: str
    original: Path
    variant: Path
    original_text: str
    variant_text: str
    #: Where the original wording sits in the checkpoint's context.
    spans: list[Any]
    #: Parts of the edit that cannot reach the model, as sentences.
    unapplied: list[str]
    #: The declared target, if any: what the confirmatory test is about.
    target: Any
    alpha: float = 0.05
    #: Things a person should know before sampling, which do not stop it.
    warnings: list[str] = field(default_factory=list)

    @property
    def estimate(self) -> Estimate | None:
        return estimate(self.checkpoint, self.plan, arms=2)

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        from tuningfork.experiment.skill import text_sha256

        guess = self.estimate
        return {
            "checkpoint": self.checkpoint.to_json(),
            "sampling": self.plan.as_settings(),
            "base_url": self.base_url,
            "original": str(self.original),
            "variant": str(self.variant),
            "original_sha256": text_sha256(self.original_text),
            "variant_sha256": text_sha256(self.variant_text),
            "spans": [asdict(span) for span in self.spans],
            "unapplied": self.unapplied,
            "target": self.target.description if self.target else None,
            "alpha": self.alpha,
            "warnings": self.warnings,
            "estimate": guess.to_json() if guess else None,
        }


@dataclass(slots=True)
class ABResult:
    run: Any
    comparison: Any
    #: The history caveat (`experiment.sampling.history_caveat`), sharpened by what this
    #: checkpoint holds.
    caveat: str
    elapsed: float

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "experiments": {
                "original": experiment_json(self.run.original),
                "variant": experiment_json(self.run.variant),
            },
            "spans": [asdict(span) for span in self.run.spans],
            "comparison": self.comparison.to_json(),
            "caveat": self.caveat,
            "elapsed": round(self.elapsed, 2),
        }


def prepare_ab(
    store: Store,
    run: str,
    step: str,
    *,
    variant: Path,
    original: Path | None = None,
    target: str | None = None,
    target_match: str | None = None,
    alpha: float = 0.05,
    base_url: str | None = None,
    normalizers: Sequence[str] = (),
    variant_text: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    skill: str | None = None,
    **plan: Any,
) -> ABPlan:
    """An A/B at a checkpoint, checked before any sample is drawn. Without `original`, the
    original is the skill the run kept — `skill` names it where the run kept several.

    `variant` is the variant's file. With `variant_text`, that is where the text will be
    saved when the A/B starts — a wording edited in the browser is written only then.
    """
    from tuningfork.experiment import (
        CheckpointError,
        SkillNotFound,
        locate,
        resolve_target,
        substitute,
        unapplied,
    )
    from tuningfork.experiment.skill import diagnose
    from tuningfork.paths import host_path

    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    for name, value in (("original", original), ("variant", variant)):
        # An agent once passed the skill's whole text here: say what the field is for, rather
        # than that a file named after the text does not exist.
        if value is not None and "\n" in str(value):
            raise OperationError(
                f"{name} is the path of a skill file, not the skill's text",
                hint="give the edited wording as variant_text; leave original out to compare "
                "with the skill the run was given",
                field=name,
            )
    variant = Path(host_path(str(variant)))
    studied = None if original else study_skill(checkpoint.run, skill)
    recorded = studied[1] if studied is not None else None
    if original:
        original = Path(host_path(str(original)))
    elif recorded is not None:
        original = recorded
    elif variant_text is not None:
        raise OperationError(
            f"run {checkpoint.run.id[:8]} kept no copy of the skill file it was given",
            hint="name the original: the skill file as it was when the run was recorded",
            field="original",
        )
    else:
        original = variant.with_name("SKILL.md")
    if original.resolve() == variant.resolve():
        raise OperationError("the variant and the original are the same file")
    for path in (original, variant):
        if path == variant and variant_text is not None:
            continue
        if not path.exists():
            detail = None
            if path == original:
                detail = "name the original: the skill file as it was when the run was recorded"
            raise OperationError(
                f"{path} does not exist", detail=detail,
                field="original" if path == original else "variant",
            )
    original_text = original.read_text(encoding="utf-8")
    if variant_text is None:
        variant_text = variant.read_text(encoding="utf-8")
    if variant_text == original_text:
        raise OperationError("the variant and the original are the same text", field="variant_text")

    warnings: list[str] = []
    try:
        declared, target_normalizers = resolve_target(
            store, checkpoint, letter=target, pattern=target_match, rules=target_rules
        )
    except (CheckpointError, ValueError) as exc:
        raise OperationError(str(exc), field="target_rules" if target_rules else None) from None
    normalizers = list(normalizers)
    if target_normalizers is not None and normalizers != target_normalizers:
        if normalizers:
            warnings.append(
                "--normalizer ignored: a lettered target is only meaningful under the "
                "normalizers of the resample it came from"
            )
        normalizers = target_normalizers

    sampling = make_plan(normalizers=normalizers, **plan)
    spans = locate(checkpoint.request, original_text)
    if not spans:
        raise OperationError(
            f"{original} is not in the context at this step.",
            detail=diagnose(checkpoint.request, original_text),
        )
    try:
        substitute(checkpoint.request, original_text, variant_text)
    except SkillNotFound as exc:
        raise OperationError(str(exc)) from None
    from tuningfork.endpoints import check_endpoint

    check_endpoint(base_url or default_upstream(), home=store.home, purpose="sampling")
    return ABPlan(
        checkpoint=checkpoint,
        plan=sampling,
        base_url=base_url or default_upstream(),
        original=original,
        variant=variant,
        original_text=original_text,
        variant_text=variant_text,
        spans=spans,
        unapplied=unapplied(original_text, variant_text, spans),
        target=declared,
        alpha=alpha,
        warnings=warnings,
    )


async def execute_ab(
    store: Store,
    prepared: ABPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ABResult:
    from tuningfork.analysis.distribution import compare_arms
    from tuningfork.experiment import SkillNotFound, experiment_outcomes, experiment_reasoning, history_caveat, run_ab

    started = time.monotonic()
    created: list[Any] = []
    try:
        result = await _cancellable(
            run_ab(
                store,
                prepared.checkpoint,
                prepared.plan,
                base_url=prepared.base_url,
                original_text=prepared.original_text,
                variant_text=prepared.variant_text,
                original_path=str(prepared.original),
                variant_path=str(prepared.variant),
                target=prepared.target,
                progress=_tracking(created, progress),
            ),
            cancel,
        )
    except SkillNotFound as exc:
        raise OperationError(str(exc)) from None
    except Cancelled as stopped:
        _stopped(store, created, stopped.reason)
        # What was sampled is stored and can be re-reported; there is no comparison yet.
        raise Cancelled(stopped.reason, partial=created) from None
    from tuningfork.experiment.sampling import target_letters

    comparison = compare_arms(
        experiment_outcomes(store, result.original),
        experiment_outcomes(store, result.variant),
        target=prepared.target,
        alpha=prepared.alpha,
        reasoning=(experiment_reasoning(store, result.original), experiment_reasoning(store, result.variant)),
        letters=target_letters(store, prepared.target.spec if prepared.target else None),
    )
    return ABResult(
        run=result,
        comparison=comparison,
        caveat=history_caveat(prepared.checkpoint),
        elapsed=time.monotonic() - started,
    )


# --- what an A/B tested ----------------------------------------------------------------------


@dataclass(slots=True)
class Wording:
    """The edit an A/B tested: its two texts, and the lines between them."""

    #: Each side's file as the A/B named it (`path`), and a short name for it (`label`); a
    #: variant saved in this home's variants folder adds its `skill` and `name`.
    original: dict[str, Any]
    variant: dict[str, Any]
    #: `ops.diff.line_diff` from the original to the variant, numbered as the skill file's
    #: lines; None when neither the records nor the files hold the texts it tested.
    diff: dict[str, Any] | None = None
    #: Where the texts were read: `sent` (its arms' requests), `fork` (a fork's launch) or
    #: `files` (the files it names, each still holding the text it tested).
    source: str | None = None
    #: How the model read the skill, when read from what was sent: full, full-trimmed or body.
    form: str | None = None
    #: Why there is no diff.
    missing: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "original": self.original, "variant": self.variant, "diff": self.diff,
            "source": self.source, "form": self.form, "missing": self.missing,
        }


def ab_wording(store: Store, arms: dict[str, Any]) -> Wording | None:
    """What an A/B changed, read back from what it recorded rather than from files that can
    have changed since: a fork's launch keeps both texts whole, and a checkpoint A/B's arms
    the requests they sent, the variant where the original was (`experiment.skill.swapped`).
    Else the files the A/B names, each only while it still holds the text it tested. None for
    an experiment that swapped no wording."""
    from tuningfork.experiment.skill import swapped
    from tuningfork.ops.diff import line_diff

    info = next((arm.variant for arm in arms.values() if arm.variant), None)
    if not info:
        return None
    found = Wording(original=_side(store, info, "original"), variant=_side(store, info, "variant"))
    offset = 0
    texts = _fork_texts(store, arms.get("variant"))
    if texts is not None:
        found.source = "fork"
    elif {"original", "variant"} <= set(arms) and arms["variant"].settings.get("reach") != "end":
        try:
            read = swapped(
                store.experiment_request(arms["original"]), store.experiment_request(arms["variant"]),
                info.get("spans") or [],
            )
        except (OSError, KeyError, ValueError):
            read = None
        if read is not None:
            before, after, span = read
            texts = (before, after)
            found.source, found.form = "sent", span.get("form")
            # The skill as the model read it may start below the file's first line — its body,
            # without the frontmatter: numbered as the file's lines, as show_skill numbers them.
            offset = max(int(span.get("line_start") or 1) - 1, 0)
    if texts is None:
        texts = _file_texts(info)
        found.source = "files" if texts is not None else None
    if texts is None:
        found.missing = "neither what its arms recorded nor the files it names hold the texts it tested"
        return found
    found.diff = _renumbered(line_diff(*texts), offset)
    return found


def _side(store: Store, info: dict[str, Any], which: str) -> dict[str, Any]:
    """One side of an A/B: its file, and a short name — a saved variant's own, else the file's
    folder and name."""
    from tuningfork.ops.variants import NAME, root

    path = info.get(f"{which}_path")
    side: dict[str, Any] = {"path": path, "label": "/".join(Path(path).parts[-2:]) if path else "—"}
    if not path:
        return side
    try:
        skill, name = Path(path).resolve().relative_to(root(store.home).resolve()).parts
    except (OSError, ValueError):
        return side
    stem = name[: -len(".md")] if name.endswith(".md") else ""
    if NAME.fullmatch(skill) and NAME.fullmatch(stem):
        side.update(label=stem, skill=skill, name=stem)
    return side


def _fork_texts(store: Store, arm: Any) -> tuple[str, str] | None:
    """Both texts, as a fork of an A/B's variant arm was given them: its launch keeps them."""
    if arm is None or arm.settings.get("reach") != "end":
        return None
    for sample in store.samples(arm.id):
        run = store.get_run(sample.run_id) if sample.run_id else None
        launch = store.get_launch(run.launch_id) if run is not None and run.launch_id else None
        kept = (launch.variant or {}) if launch is not None else {}
        if isinstance(kept.get("original_text"), str) and isinstance(kept.get("variant_text"), str):
            return kept["original_text"], kept["variant_text"]
    return None


def _file_texts(info: dict[str, Any]) -> tuple[str, str] | None:
    """The texts in the files an A/B names, each only if it is still the text the A/B
    recorded (its digest: an A/B at a step digests it with its line endings plain, a fork as
    read)."""
    import hashlib

    from tuningfork.experiment.skill import text_sha256

    texts = []
    for which in ("original", "variant"):
        path = info.get(f"{which}_path")
        try:
            text = Path(path).read_text(encoding="utf-8") if path else None
        except (OSError, ValueError):
            text = None
        digests = {text_sha256(text), hashlib.sha256(text.encode()).hexdigest()} if text is not None else set()
        if info.get(f"{which}_sha256") not in digests:
            return None
        texts.append(text)
    return texts[0], texts[1]


def _renumbered(diff: dict[str, Any], offset: int) -> dict[str, Any]:
    """A line diff's numbers moved down by `offset` lines."""
    if offset:
        for hunk in diff["hunks"]:
            hunk["old_start"] += offset
            hunk["new_start"] += offset
            for line in hunk["lines"]:
                for key in ("old", "new"):
                    if key in line:
                        line[key] += offset
    return diff


def _tracking(created: list[Any], progress: Progress | None) -> Progress:
    """Forward progress, keeping the experiments it reports as they are created."""

    def track(event: Any) -> None:
        if event.kind == "experiment":
            created.append(event.subject)
        if progress is not None:
            progress(event)

    return track


def _stopped(store: Store, experiments: list[Any], reason: str) -> None:
    """Experiments stopped by tuningfork shutting down read as interrupted, not cancelled:
    no one asked for them to stop."""
    if reason == "shutdown":
        store.set_experiment_status([e.id for e in experiments], "interrupted")


async def _cancellable(work, cancel: Cancel | None):
    """Run `work`, stopping it — samples already drawn stay recorded — if `cancel` fires.

    Stopped by `cancel`, it raises `Cancelled`; cancelled from outside (Ctrl-C in the CLI,
    say), the `CancelledError` goes on as it came.
    """
    if cancel is None:
        return await work
    task = asyncio.ensure_future(work)
    loop = asyncio.get_running_loop()
    unregister = cancel.on_cancel(lambda: loop.call_soon_threadsafe(task.cancel))
    try:
        return await task
    except asyncio.CancelledError:
        if cancel.cancelled:
            raise Cancelled(cancel.reason or "cancelled") from None
        raise
    finally:
        unregister()
