"""Several wordings of a skill sampled at one checkpoint, each against the original.

A screen draws a fresh baseline arm — the checkpoint as recorded — then one arm per
proposal, the skill's text swapped for the proposal's wherever the model saw it
(`experiment.skill.substitute`). Everything a verdict depends on is fixed before the first
sample: the arms and their order, the target, alpha and the samples per arm are stored with
every arm, so the report is always recomputed from the store against exactly the test
planned (`analysis.screen` holds the statistics).

The baseline arm is drawn first and the arms one after another, in blocks, as an A/B's
are: on a server with prefix caching, interleaving them would reprocess the context from
the skill onward at every switch.

A screen run to the end measures the original once per refinement: its forks at a checkpoint
count again in the refinement's later screens there, which draw only what their n still needs
(`original_forks`). A quick look forks the proposals alone, and reports rates without a verdict.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress
from tuningfork.ops.experiments import _cancellable, _stopped, _tracking, default_upstream, estimate, load, make_plan
from tuningfork.store.db import Store

#: Proposals one screen may hold. Each costs n samples, and each widens the correction.
MAX_ARMS = 6
BASELINE = "original"
#: Forks a quick look draws of each proposal — two at each of four checkpoints: enough to see an
#: edit that moves the pass rate a long way, never enough to keep one.
QUICK_N = 8


@dataclass(slots=True)
class ScreenArm:
    """One proposal in a screen: its wording, and the request the model will see."""

    name: str
    text: str
    #: Where it came from: {"proposal", "variant_path", ...}, stored with its arm.
    meta: dict[str, Any]
    request: dict[str, Any]
    spans: list[Any]
    unapplied: list[str]
    changed: int

    def variant_info(self, original_text: str) -> dict[str, Any]:
        from tuningfork.experiment.skill import text_sha256

        return {
            **self.meta,
            "original_sha256": text_sha256(original_text),
            "variant_sha256": text_sha256(self.text),
            "spans": [asdict(span) for span in self.spans],
            "unapplied": self.unapplied,
        }


@dataclass(slots=True)
class ScreenPlan:
    checkpoint: Any
    plan: Any
    base_url: str
    original_text: str
    arms: list[ScreenArm]
    target: Any
    normalizers: list[str]
    alpha: float
    refinement: str | None = None
    #: Before any sample: what an arm must reach to survive, against what the baseline did.
    power: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def family(self) -> int:
        return len(self.arms)

    @property
    def total(self) -> int:
        return self.plan.n * (self.family + 1)

    def annotations(self) -> dict[str, Any]:
        """Stored with every arm: the screen as planned, before its first sample."""
        return {
            "target": self.target.spec,
            "refinement": self.refinement,
            "screen": {
                "arms": [arm.name for arm in self.arms],
                "per_arm": self.plan.n,
                "alpha": self.alpha,
                "family": self.family,
                "proposals": {arm.name: arm.meta.get("proposal") for arm in self.arms},
            },
        }

    def to_json(self) -> dict[str, Any]:
        guess = estimate(self.checkpoint, self.plan, arms=self.family + 1)
        return {
            "checkpoint": self.checkpoint.to_json(),
            "refinement": self.refinement,
            "base_url": self.base_url,
            "n": self.plan.n,
            "total": self.total,
            "target": self.target.description,
            "normalizers": self.normalizers,
            "alpha": self.alpha,
            "family": self.family,
            "arms": [
                {"name": arm.name, **{k: v for k, v in arm.meta.items() if k in ("proposal", "variant_path")},
                 "changed_lines": arm.changed, "unapplied": arm.unapplied}
                for arm in self.arms
            ],
            "power": self.power,
            "estimate": guess.to_json() if guess else None,
            "warnings": self.warnings,
        }


def prepare_screen(
    store: Store,
    run: str,
    step: str,
    *,
    original_text: str,
    arms: Sequence[tuple[str, str, dict[str, Any]]],
    target_spec: dict[str, Any],
    normalizers: Sequence[str] = (),
    alpha: float = 0.05,
    base_url: str | None = None,
    refinement: str | None = None,
    **plan: Any,
) -> ScreenPlan:
    """A screen at a checkpoint, checked before any sample is drawn: every arm must change
    what the model sees there, and at least one must be able to survive the correction.

    `arms` are (name, the proposal's whole text, where it came from), in the order the
    family is planned."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.analysis.screen import needed_to_survive
    from tuningfork.endpoints import check_endpoint
    from tuningfork.experiment import experiment_outcomes
    from tuningfork.experiment.sampling import step_resample
    from tuningfork.experiment.skill import NothingToCompare, diagnose, locate, substitute, text_sha256, unapplied
    from tuningfork.ops.diff import line_diff
    from tuningfork.ops.trash import refuse_trashed

    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    if not arms:
        raise OperationError("a screen needs at least one proposal", code="bad_params", field="proposals")
    if len(arms) > MAX_ARMS:
        raise OperationError(
            f"a screen holds at most {MAX_ARMS} proposals: each costs n samples and widens the correction",
            code="bad_params", field="proposals",
        )
    names = [name for name, _, _ in arms]
    if len(set(names)) != len(names) or BASELINE in names:
        raise OperationError("each proposal needs its own name, and none may be called original", field="proposals")
    if not 0 < alpha < 1:
        raise OperationError("alpha is between 0 and 1", code="bad_params", field="alpha")
    if not locate(checkpoint.request, original_text):
        raise OperationError(
            f"the skill's text is not in the context at step {checkpoint.step.idx}, so no wording "
            "of it can change what the model saw there",
            detail=diagnose(checkpoint.request, original_text),
            field="step",
        )
    seen = {text_sha256(original_text): BASELINE}
    prepared: list[ScreenArm] = []
    warnings: list[str] = []
    for name, text, meta in arms:
        digest = text_sha256(text)
        if digest in seen:
            raise OperationError(f"{name} is the same text as {seen[digest]}", field="proposals")
        seen[digest] = name
        try:
            request, spans = substitute(checkpoint.request, original_text, text)
        except NothingToCompare:
            raise OperationError(
                f"{name} changes nothing the model sees at step {checkpoint.step.idx}",
                hint="an edit to the frontmatter, or to text the harness never sends, cannot be screened",
                field="proposals",
            ) from None
        left = unapplied(original_text, text, spans)
        if left:
            warnings.append(f"{name}: part of the edit cannot reach the model here — " + "; ".join(left))
        changed = line_diff(original_text, text)["changed"]
        prepared.append(ScreenArm(name, text, dict(meta), request, spans, left, changed))

    if target_spec.get("kind") == "check":
        raise OperationError(
            "this refinement's target is the task's check, which no answer at the step can be judged by",
            hint="screen it to the end — each arm's samples forks run through the harness, the check "
                 "deciding: `tuningfork refine screen <id> --to-end`, or start_screen_to_end",
            code="to_end_only", field="refinement",
        )
    target = target_from_spec(target_spec)
    sampling = make_plan(normalizers=normalizers, **plan)
    n, family = sampling.n, len(prepared)
    # How the baseline did at this step before: a resample's count of the target, if any.
    power: dict[str, Any] = {"family": family, "alpha": alpha, "n": n}
    resample = step_resample(store, checkpoint)
    if resample is not None:
        outcomes = [o for o in experiment_outcomes(store, resample, list(normalizers)) if not o.is_error]
        if outcomes:
            hits = sum(1 for outcome in outcomes if target.matches(outcome))
            expected = round(hits / len(outcomes) * n)
            power.update(baseline={"hits": hits, "of": len(outcomes), "resample": resample.id},
                         needed=needed_to_survive(expected, n, family, alpha))
            if power["needed"] is None:
                raise OperationError(
                    f"no proposal could survive: the step already does this {hits} times in "
                    f"{len(outcomes)}, so even {n} of {n} would not pass the correction over {family}",
                    hint="screen fewer proposals, draw more samples per arm, or declare a target "
                    "the step seldom reaches",
                    field="n",
                )
    if "needed" not in power:
        power["needed"] = needed_to_survive(0, n, family, alpha)
        power["assumed"] = "a baseline that never reaches the target: the most an arm could be helped"
    if n < 20:
        warnings.append(f"at {n} samples an arm, only a large shift can survive the correction")
    return ScreenPlan(
        checkpoint=checkpoint,
        plan=sampling,
        base_url=check_endpoint(base_url or default_upstream(), home=store.home, purpose="sampling"),
        original_text=original_text,
        arms=prepared,
        target=target,
        normalizers=list(normalizers),
        alpha=alpha,
        refinement=refinement,
        power=power,
        warnings=warnings,
    )


@dataclass(slots=True)
class ScreenResult:
    group_id: str
    #: Each arm's experiment at the refinement's own checkpoint, by the arm's name.
    experiments: dict[str, Any]
    report: Any
    elapsed: float = 0.0
    #: Each arm's experiments at every checkpoint, the refinement's own first: one for a
    #: screen at a step, or to the end from one checkpoint.
    by_arm: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def notes(self) -> list[str]:
        """What qualifies the samples, each said once though every arm keeps it: on Claude
        Code's Messages calls, that llama.cpp drew them from its cache of the prompt."""
        arms = [arm for arms in (self.by_arm or {name: [found] for name, found in self.experiments.items()}).values() for arm in arms]
        return list(dict.fromkeys(note for arm in arms for note in arm.notes or []))

    def to_json(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "experiments": {name: experiment.id for name, experiment in self.experiments.items()},
            "all_experiments": {name: [found.id for found in arms] for name, arms in self.by_arm.items()},
            "screen": self.report.to_json(),
            "notes": self.notes,
            "elapsed": round(self.elapsed, 1),
        }


async def execute_screen(
    store: Store,
    prepared: ScreenPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ScreenResult:
    import httpx

    from tuningfork.endpoints import guard_hooks
    from tuningfork.experiment import check_parity
    from tuningfork.experiment.sampling import finished, run_arm

    started = time.monotonic()
    created: list[Any] = []
    group_id = uuid.uuid4().hex
    annotations = prepared.annotations()

    async def work() -> None:
        async with httpx.AsyncClient(timeout=prepared.plan.timeout, event_hooks=guard_hooks(home=store.home)) as http:
            parity = await check_parity(http, prepared.base_url, prepared.checkpoint, prepared.plan)
            # Every arm stays running until the whole screen has: a finished baseline with
            # its proposals still to come is a screen under way, not one cut off.
            with finished(store, _tracking(created, progress)) as ended:
                for name, request, variant in (
                    (BASELINE, prepared.checkpoint.request, None),
                    *((arm.name, arm.request, arm.variant_info(prepared.original_text)) for arm in prepared.arms),
                ):
                    await run_arm(
                        store, prepared.checkpoint, prepared.plan,
                        base_url=prepared.base_url, request=request, kind="screen", arm=name,
                        client=http, group_id=group_id, variant=variant, parity=parity,
                        annotations=annotations, progress=ended,
                    )

    try:
        await _cancellable(work(), cancel)
    except Cancelled as stopped:
        _stopped(store, created, stopped.reason)
        # What was sampled is stored; the report says it is incomplete, with no verdict.
        raise Cancelled(stopped.reason, partial=screen_result(store, group_id) if created else None) from None
    result = screen_result(store, group_id)
    result.elapsed = time.monotonic() - started
    return result


# --- to the end: every arm's samples forks, the task's check deciding ---------------------------


@dataclass(slots=True)
class ScreenToEndPlan:
    """A screen judged by the task's check: each arm's samples are forks of the run from the
    checkpoint to the end — the original's wording, then each proposal's swapped in — and
    the check decides each. For a failure only the check can see: a computed property of
    what the run wrote, which no target over an answer expresses. Each sample runs the
    harness to the end, so it costs what a fork costs, not what a model call costs.

    The forks may be drawn from several checkpoints — the same step of several recorded
    runs of the task, the refinement's own first, passing and failing by turns — each arm's
    n spread evenly over them. A fork measures the edit's effect given the prefix it
    replays, and one prefix is one draw of a path the model takes nine times in ten or one:
    an effect that holds across prefixes is one fresh runs keep, and one that lives in a
    single prefix shows up as disagreement between checkpoints. The arms share the same
    checkpoints in the same counts, so the pooled test compares like with like."""

    checkpoint: Any
    baseline: Any  # ops.forks.ForkPlan: the original's wording at the refinement's own checkpoint
    #: (name, where it came from, its fork plans with the proposal's wording, one per
    #: checkpoint), in family order.
    arms: list[tuple[str, dict[str, Any], list[Any]]]
    n: int
    concurrency: int
    alpha: float
    refinement: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: The checkpoints the forks are drawn from, the refinement's own first.
    checkpoints: list[Any] = field(default_factory=list)
    #: The original's fork plan at each checkpoint, in the same order.
    baselines: list[Any] = field(default_factory=list)
    #: Each arm's forks at each checkpoint: n spread evenly, the remainder to the first.
    per_checkpoint: list[int] = field(default_factory=list)
    #: Why each candidate checkpoint was passed over, for the plan.
    notes: list[str] = field(default_factory=list)
    #: A quick look: the proposals' forks alone, rates and no verdict.
    quick: bool = False
    #: At each checkpoint, the original's arms of the refinement's earlier screens counted here
    #: (`original_forks`), and how many judged forks they hold.
    reused: list[list[Any]] = field(default_factory=list)
    reused_judged: list[int] = field(default_factory=list)
    #: The original's forks this screen draws at each checkpoint: what its count still needs.
    original_counts: list[int] = field(default_factory=list)

    @property
    def family(self) -> int:
        return len(self.arms)

    @property
    def drawn_original(self) -> list[int]:
        return self.original_counts or self.per_checkpoint

    @property
    def total(self) -> int:
        return sum(self.drawn_original) + self.n * self.family

    @property
    def forks(self) -> int:
        return self.total

    @property
    def original_planned(self) -> int:
        """The original's forks the report counts on: a full screen's n, a quick look's whatever
        the refinement drew before."""
        return sum(self.reused_judged) if self.quick else self.n

    def checkpoints_json(self) -> list[dict[str, Any]]:
        from tuningfork.ops.targets import verdict_of

        return [
            {"run": found.run.id, "step": found.step.idx, "verdict": verdict_of(found.run), "n": count}
            for found, count in zip(self.checkpoints, self.per_checkpoint)
        ]

    def annotations(self) -> dict[str, Any]:
        """Stored with every arm: the screen as planned, before its first fork."""
        return {
            "refinement": self.refinement,
            "screen": {
                "arms": [name for name, _, _ in self.arms],
                "per_arm": self.n,
                "alpha": self.alpha,
                "family": self.family,
                "proposals": {name: meta.get("proposal") for name, meta, _ in self.arms},
                "reach": "end",
                "checkpoints": self.checkpoints_json(),
                "quick": self.quick,
                # The original's arms of earlier screens it counts, fixed before its first fork.
                "original_reused": [arm.id for arms in self.reused for arm in arms],
                "original_planned": self.original_planned,
            },
        }

    def to_json(self) -> dict[str, Any]:
        from tuningfork.analysis.distribution import CHECK_TARGET

        return {
            "checkpoint": self.checkpoint.to_json(),
            "checkpoints": self.checkpoints_json(),
            "notes": self.notes,
            "refinement": self.refinement,
            "reach": "end",
            "quick": self.quick,
            "n": self.n,
            "total": self.total,
            "forks": self.forks,
            "original": {
                "reused": sum(self.reused_judged),
                "from": sorted({arm.group_id or arm.id for arms in self.reused for arm in arms}),
                "drawn": sum(self.drawn_original),
            },
            "concurrency": self.concurrency,
            "target": CHECK_TARGET["description"],
            "alpha": self.alpha,
            "family": self.family,
            "arms": [
                {"name": name, **{k: v for k, v in meta.items() if k in ("proposal", "variant_path")},
                 "changed_lines": meta.get("changed_lines")}
                for name, meta, _ in self.arms
            ],
            "harness": self.baseline.harness,
            "model": self.baseline.model,
            "proxy": self.baseline.proxy.url,
            "sandbox": self.baseline.sandbox.to_json() if self.baseline.sandbox is not None else None,
            "warnings": self.warnings,
        }


def split_evenly(n: int, parts: int) -> list[int]:
    """`n` forks over `parts` checkpoints, the remainder to the first — the refinement's own."""
    base, extra = divmod(n, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def choose_checkpoints(
    store: Store,
    checkpoint: Any,
    count: int,
    *,
    original_text: str,
    aligned: str | Sequence[str] | None = None,
    runs: int = 200,
) -> tuple[list[Any], list[str]]:
    """The checkpoints a screen to the end draws its forks from: the refinement's own, then
    up to `count - 1` others — the same step of other recorded runs of its task, passing
    and failing by turns, starting with the verdict the refinement's run did not get, and
    preferring runs that reached the step by a different path (the tools they called
    before it). With `aligned` — the step as the refinement named it by what the run did
    there: a tool, a file's first change, a finding's making (`ops.experiments.find_step`),
    or a list of tools — the step is each run's own such step; else the same index.

    A run serves when its step exists and is the same kind of answer as the refinement's
    (a call, or a reply), has the skill's text in its context — so the proposals' wording
    can be swapped in there — and can be replayed up to it. The notes say which were
    passed over, and why."""
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.experiment.skill import locate
    from tuningfork.ops.experiments import find_step, not_done, tools_called
    from tuningfork.ops.targets import verdict_of
    from tuningfork.replay import replay_problems

    chosen: list[Any] = [checkpoint]
    notes: list[str] = []
    if count <= 1 or not checkpoint.run.task_id:
        return chosen, notes
    own_kind = recorded_outcome(store.response(checkpoint.step), checkpoint.step.error).kind
    own_verdict = verdict_of(checkpoint.run)

    def prefix(run_id: str, index: int) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(tools_called(store, found)) for found in store.steps(run_id) if found.idx < index)

    candidates: dict[str, list[tuple[Any, tuple[Any, ...]]]] = {"passed": [], "failed": []}
    for other in store.runs(limit=runs, task_id=checkpoint.run.task_id):
        if other.id == checkpoint.run.id or other.is_thread or other.is_fork:
            continue
        verdict = verdict_of(other)
        if verdict is None:
            continue
        index = find_step(store, other, aligned) if aligned else checkpoint.step.idx
        if index is None:
            notes.append(f"run {other.id[:8]} {not_done(aligned)}")
            continue
        found = store.step(other.id, index)
        if found is None:
            notes.append(f"run {other.id[:8]} has no step {index}")
            continue
        kind = recorded_outcome(store.response(found), found.error).kind
        if kind != own_kind:
            notes.append(f"run {other.id[:8]}'s step {index} is {'a reply' if kind == 'text' else kind}, "
                         f"not {'a reply' if own_kind == 'text' else own_kind} as the refinement's is")
            continue
        request = store.rebuild_request(found)
        if not locate(request, original_text):
            notes.append(f"the skill's text is not in run {other.id[:8]}'s context at step {index}")
            continue
        problems = replay_problems(store, other, index)
        if problems:
            notes.append(f"run {other.id[:8]} cannot be replayed up to step {index}: {problems[0]}")
            continue
        candidates[verdict].append((load(store, other.id, str(index)), prefix(other.id, index)))

    taken = {prefix(checkpoint.run.id, checkpoint.step.idx)}
    order = ["failed", "passed"] if own_verdict != "failed" else ["passed", "failed"]
    turn = 0
    while len(chosen) < count and (candidates["passed"] or candidates["failed"]):
        verdict = order[turn % 2]
        turn += 1
        pool = candidates[verdict] or candidates[order[turn % 2]]
        # A path not drawn from yet, when there is one; else the newest left.
        pick = next((entry for entry in pool if entry[1] not in taken), pool[0])
        pool.remove(pick)
        candidate, signature = pick
        taken.add(signature)
        chosen.append(candidate)
    return chosen, notes


def judged_forks(store: Store, experiment: Any) -> int:
    """An arm's forks the check decided: neither left out nor without a verdict."""
    return sum(1 for sample in store.samples(experiment.id) if sample.exclusion is None and sample.passed is not None)


def original_forks(store: Store, refinement: str | None, checkpoint: Any, plan: Any) -> list[Any]:
    """The original's forks a refinement's earlier screens to the end drew at `checkpoint`: the
    original arms there that have ended, forked through the same harness, model and proxy as
    `plan` (`ops.forks.ForkPlan`), out of the trash. The original's wording is the refinement's
    own, the same in all its screens, so a later screen counts these and draws only what its n
    still needs — a fork costs what a run costs, and two screens of one broken skill spent forty
    forks measuring an original at 0 of 20 twice (findings §31.57). None without a refinement."""
    from tuningfork.ops.inspect import arm_status

    if not refinement:
        return []
    found = []
    for experiment in store.experiments(run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, kind="screen",
                                        limit=10_000):
        settings = experiment.settings or {}
        if (experiment.arm == BASELINE and settings.get("reach") == "end" and settings.get("refinement") == refinement
                and settings.get("harness") == plan.harness and settings.get("model") == plan.model
                and experiment.base_url == plan.proxy.url
                and arm_status(experiment) in ("done", "cancelled", "interrupted")):
            found.append(experiment)
    return sorted(found, key=lambda experiment: experiment.created_at)


def earlier_checkpoints(store: Store, checkpoint: Any, refinement: str | None, count: int) -> list[tuple[str, int]] | None:
    """The checkpoints the refinement's latest screen to the end drew from, when it drew from
    `count` of them, the refinement's own first: a later screen draws from the same ones, where
    the original's forks already are. None when there is none to follow."""
    if not refinement:
        return None
    for experiment in store.experiments(run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, kind="screen",
                                        limit=10_000):
        settings = experiment.settings or {}
        if settings.get("refinement") != refinement or settings.get("reach") != "end":
            continue
        planned = (settings.get("screen") or {}).get("checkpoints") or []
        if (len(planned) == count and planned[0].get("run") == checkpoint.run.id
                and planned[0].get("step") == checkpoint.step.idx):
            return [(str(entry["run"]), int(entry["step"])) for entry in planned]
        return None  # the latest screen decides, newest first
    return None


def prepare_screen_to_end(
    store: Store,
    run: str,
    step: str,
    *,
    original_text: str,
    arms: Sequence[tuple[str, str, dict[str, Any]]],
    n: int = 20,
    concurrency: int = 1,
    alpha: float = 0.05,
    base_url: str | None = None,
    spec: Path | None = None,
    workdir: Path | None = None,
    proxy: Any = None,
    refinement: str | None = None,
    checkpoints: int = 1,
    aligned: str | Sequence[str] | None = None,
    quick: bool = False,
    fresh_original: bool = False,
) -> ScreenToEndPlan:
    """A screen run to the end, checked before any fork starts: a fork plan for the original
    and one for each proposal at each checkpoint, its saved file swapped in wherever the
    model saw the skill, the task's check required, and no more forks at once than the
    proxy lets through.

    The original's forks the refinement's earlier screens drew at the same checkpoints are
    counted (`original_forks`), and only what n still needs is drawn — none, after a full
    screen — unless `fresh_original`; a later screen draws from the checkpoints the latest one
    did (`earlier_checkpoints`). `quick` is a quick look: `QUICK_N` forks of each proposal and
    none of the original, its rates set beside the original's forks drawn before, and no
    verdict — a lead to choose the next edit by, never one to keep.

    `arms` are (name, the proposal's whole text, where it came from — with the
    `variant_path` of its saved file), in the order the family is planned. `checkpoints`
    is how many runs to draw the forks from — this one and others of its task at the same
    step (`choose_checkpoints`), each arm's n spread evenly over them; fewer when the
    recordings hold fewer that serve. `aligned` is the step as it was named by what the run
    did there (a tool, `changed:<path>`, `made:<words>`): the same step is then each run's
    own such step.

    A fork replays everything before its step as recorded — the model's answers and what its
    tools did to the files — so the plan warns, for each checkpoint, of a call before it that
    changed the workspace: an edit to what that call did cannot show there (findings §31.42)."""
    from tuningfork.experiment.skill import text_sha256
    from tuningfork.ops.diff import line_diff
    from tuningfork.ops.forks import check_fork_concurrency, prepare_fork
    from tuningfork.ops.inspect import replay_warning
    from tuningfork.ops.trash import refuse_trashed

    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    if not arms:
        raise OperationError("a screen needs at least one proposal", code="bad_params", field="proposals")
    if len(arms) > MAX_ARMS:
        raise OperationError(
            f"a screen holds at most {MAX_ARMS} proposals: each costs n forks and widens the correction",
            code="bad_params", field="proposals",
        )
    names = [name for name, _, _ in arms]
    if len(set(names)) != len(names) or BASELINE in names:
        raise OperationError("each proposal needs its own name, and none may be called original", field="proposals")
    if not 0 < alpha < 1:
        raise OperationError("alpha is between 0 and 1", code="bad_params", field="alpha")
    if quick and fresh_original:
        raise OperationError("a quick look draws none of the original's forks: fresh_original is for a full screen",
                             code="bad_params", field="fresh_original")
    if quick:
        n = QUICK_N
    if n < 1:
        raise OperationError("n must be at least 1", code="bad_params", field="n")
    if checkpoints < 1:
        raise OperationError("checkpoints is at least 1: the refinement's own", code="bad_params", field="checkpoints")
    options: dict[str, Any] = dict(base_url=base_url, spec=spec, workdir=workdir, breakpoints=False, need_check=True)
    seen = {text_sha256(original_text): BASELINE}
    warnings: list[str] = []
    for name, text, meta in arms:
        digest = text_sha256(text)
        if digest in seen:
            raise OperationError(f"{name} is the same text as {seen[digest]}", field="proposals")
        seen[digest] = name
        if not meta.get("variant_path"):
            raise OperationError(f"{name} has no saved file to fork with", field="proposals")

    # The checkpoints, the refinement's own first: it must serve, and another that cannot be
    # forked is passed over with a note.
    wanted = min(checkpoints, n)
    if checkpoints > n:
        warnings.append(f"{checkpoints} checkpoints asked for and {n} forks an arm: at most one checkpoint a fork")
    chosen: list[Any] = []
    notes: list[str] = []
    again = None if fresh_original else earlier_checkpoints(store, checkpoint, refinement, wanted)
    if again:
        # Where the latest screen drew from, so that the original's forks there count again.
        try:
            chosen = [checkpoint, *(load(store, run_id, str(index)) for run_id, index in again[1:])]
        except OperationError:
            chosen = []
    if not chosen:
        chosen, notes = choose_checkpoints(store, checkpoint, wanted, original_text=original_text, aligned=aligned)
    prepared: dict[str, list[Any]] = {name: [] for name, _, _ in arms}
    baselines: list[Any] = []
    kept: list[Any] = []
    for found in chosen:
        plans: dict[str, Any] = {}
        try:
            for name, text, meta in arms:
                plan = prepare_fork(store, found.run.id, str(found.step.idx), variant=Path(meta["variant_path"]),
                                    proxy=proxy, **options)
                proxy = plan.proxy
                plans[name] = plan
            plans[BASELINE] = prepare_fork(store, found.run.id, str(found.step.idx), proxy=proxy, **options)
        except OperationError as refused:
            if found is checkpoint:
                raise
            notes.append(f"run {found.run.id[:8]} step {found.step.idx} cannot be forked: {refused}")
            continue
        kept.append(found)
        baselines.append(plans[BASELINE])
        for name, _, _ in arms:
            prepared[name].append(plans[name])
            if found is checkpoint:
                warnings.extend(f"{name}: {warning}" for warning in plans[name].warnings)
    if wanted > 1 and len(kept) < wanted:
        warnings.append(f"{len(kept)} checkpoint(s) of the {wanted} asked for: the task's other recorded runs "
                        "serve only where the same step exists, has the skill in its context and replays"
                        + (" (" + "; ".join(notes[:3]) + ")" if notes else ""))
    replayed = [warning for warning in (replay_warning(store, found.run, found.step.idx) for found in kept) if warning]
    if replayed:
        warnings.append(f"at {len(replayed)} of the {len(kept)} checkpoint(s) a call before the step changed the "
                        "workspace, and the forks replay it: " + "; ".join(replayed[:2])
                        + ("; …" if len(replayed) > 2 else ""))
    baseline = baselines[0]
    check_fork_concurrency(baseline.proxy, concurrency)
    per_checkpoint = split_evenly(n, len(kept))
    reused: list[list[Any]] = []
    reused_judged: list[int] = []
    original_counts: list[int] = []
    for found, plan, count in zip(kept, baselines, per_checkpoint):
        earlier = [] if fresh_original else original_forks(store, refinement, found, plan)
        if len(kept) > 1 and not quick:
            # Pooled over several checkpoints, the arms must weigh each alike: never more of the
            # original's forks at one than this screen draws of each proposal there, or the
            # original's rate leans to that prefix (found in review: twenty drawn at the first by
            # an earlier screen of it alone, against five of the proposal's). Whole arms, oldest
            # first, while they fit; what is short is drawn. At one checkpoint, all count.
            fitted, judged = [], 0
            for arm in earlier:
                forks = judged_forks(store, arm)
                if judged + forks <= count:
                    fitted.append(arm)
                    judged += forks
            earlier = fitted
        have = sum(judged_forks(store, arm) for arm in earlier)
        reused.append(earlier)
        reused_judged.append(have)
        original_counts.append(0 if quick else max(0, count - have))
    if quick:
        warnings.append(
            f"a quick look: {n} forks of each proposal and none of the original — rates beside the "
            + (f"{sum(reused_judged)} original fork(s) this refinement drew before" if sum(reused_judged)
               else "task's recorded runs, the original's forks not drawn here yet")
            + "; no verdict, and it keeps nothing: a full screen decides")
    elif n < 10:
        warnings.append(f"at {n} forks an arm, only a large shift in the pass rate can survive the correction")
    return ScreenToEndPlan(
        checkpoint=checkpoint, baseline=baseline,
        arms=[(name, {**meta, "changed_lines": line_diff(original_text, text)["changed"]}, prepared[name])
              for name, text, meta in arms],
        n=n, concurrency=concurrency, alpha=alpha, refinement=refinement, warnings=warnings,
        checkpoints=kept, baselines=baselines, per_checkpoint=per_checkpoint, notes=notes,
        quick=quick, reused=reused, reused_judged=reused_judged, original_counts=original_counts,
    )


def execute_screen_to_end(
    store: Store,
    prepared: ScreenToEndPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ScreenResult:
    """Every arm's forks, the original's first, each arm stored as one experiment of the
    screen at each checkpoint — the same group, the arm's name on each — and the report
    recomputed from the store against the test planned before the first fork."""
    from tuningfork.ops.forks import execute_to_end, finish

    started = time.monotonic()
    group_id = uuid.uuid4().hex
    tag = prepared.annotations()
    created: list[str] = []

    def note(event: Any) -> None:
        if event.kind == "experiment":
            created.append(event.subject.id)
        if progress is not None:
            progress(event)

    try:
        for name, meta, plans in ((BASELINE, {}, prepared.baselines), *prepared.arms):
            # The original's forks: only what the refinement's earlier screens did not draw.
            counts = prepared.drawn_original if name == BASELINE else prepared.per_checkpoint
            for plan, count in zip(plans, counts):
                if count < 1:
                    continue
                execute_to_end(
                    store, plan, count, kind="screen", arm=name, group_id=group_id,
                    concurrency=prepared.concurrency, progress=note, cancel=cancel, finalize=False, tag=tag,
                    variant_meta={k: v for k, v in meta.items() if k in ("proposal", "variant_path")} or None,
                )
    except Cancelled as stopped:
        # Every arm stays running until the whole screen has: what was forked is stored, and
        # the report says it is incomplete, with no verdict.
        finish(store, created, stopped)
        raise Cancelled(stopped.reason, partial=screen_result(store, group_id) if created else None) from None
    except BaseException as error:
        finish(store, created, error)
        raise
    finish(store, created, None)
    result = screen_result(store, group_id)
    result.elapsed = time.monotonic() - started
    return result


def screen_arms(store: Store, reference: str) -> tuple[str, dict[str, list[Any]], dict[str, Any]]:
    """A screen's arms from the store, by its group or any of its arms: each arm's experiments
    by its name — the original's own first, then those of earlier screens of its refinement it
    counts (`original_reused`, fixed when it was planned) — and the screen as planned."""
    found = store.get_experiment(reference)
    group_id = found.group_id if found is not None and found.group_id else reference
    by_arm: dict[str, list[Any]] = {}
    own = store.experiment_group(group_id)
    for arm in own:
        by_arm.setdefault(arm.arm, []).append(arm)
    planned = next((arm.settings.get("screen") for arm in own if arm.settings.get("screen")), None)
    if BASELINE not in by_arm and planned is None:
        raise OperationError(f"no screen matching {reference!r}", code="not_found")
    for experiment_id in (planned or {}).get("original_reused") or []:
        earlier = store.get_experiment(experiment_id)
        if earlier is not None and earlier.trashed_at is None:
            by_arm.setdefault(BASELINE, []).append(earlier)
    return group_id, by_arm, planned or {}


def screen_result(store: Store, reference: str) -> ScreenResult:
    """A screen's report from the store, by its group or any of its arms: recomputed against
    the arms, target and alpha planned before its first sample. A screen run to the end is
    reported on its forks' verdicts, the check having decided each — the original's those it
    drew and those of earlier screens of its refinement it counted (`screen_arms`)."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.analysis.screen import compare_to_baseline
    from tuningfork.experiment import experiment_outcomes

    group_id, by_arm, planned = screen_arms(store, reference)
    arms = {name: found[0] for name, found in by_arm.items()}
    own = next(iter(store.experiment_group(group_id)), None) or next(iter(arms.values()))
    names = planned.get("arms", [])
    alpha = float(planned.get("alpha", 0.05))
    per_arm = int(planned.get("per_arm") or own.settings.get("n") or 0)
    if own.settings.get("reach") == "end":
        from tuningfork.analysis.screen import compare_verdicts

        def samples(name: str) -> list[Any]:
            return [sample for arm in by_arm.get(name, []) for sample in store.samples(arm.id)]

        reused = set(planned.get("original_reused") or [])
        report = compare_verdicts(
            samples(BASELINE), {name: samples(name) for name in names}, alpha=alpha, planned=per_arm,
            behaviours={name: [outcome for arm in by_arm[name] for outcome in experiment_outcomes(store, arm)]
                        for name in (BASELINE, *names) if name in by_arm},
            baseline_planned=planned.get("original_planned"), quick=bool(planned.get("quick")),
        )
        report.reused = sum(judged_forks(store, arm) for arm in by_arm.get(BASELINE, []) if arm.id in reused)
        report.checkpoints = checkpoint_breakdown(store, by_arm, planned.get("checkpoints") or [])
        return ScreenResult(group_id=group_id, experiments=arms, report=report, by_arm=by_arm)
    target = target_from_spec(arms[BASELINE].settings["target"])
    report = compare_to_baseline(
        experiment_outcomes(store, arms[BASELINE]),
        {name: experiment_outcomes(store, arms[name]) if name in arms else [] for name in names},
        target=target, alpha=alpha, planned=per_arm,
    )
    return ScreenResult(group_id=group_id, experiments=arms, report=report, by_arm=by_arm)


def checkpoint_breakdown(store: Store, by_arm: dict[str, list[Any]], planned: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each checkpoint a screen to the end drew from, with every arm's passes there: the
    pooled test is the verdict, and this is where it came from — an effect the arms agree
    on at every checkpoint, or one checkpoint's alone. Empty for a screen from one."""
    if len(planned) < 2:
        return []
    found = []
    for entry in planned:
        arms: dict[str, dict[str, int]] = {}
        for name, experiments in by_arm.items():
            # The original's may be several there: its own, and an earlier screen's it counts.
            at = [arm for arm in experiments if arm.run_id == entry.get("run") and arm.step_idx == entry.get("step")]
            if not at:
                continue
            judged = [sample for arm in at for sample in store.samples(arm.id)
                      if sample.exclusion is None and sample.passed is not None]
            arms[name] = {"passed": sum(1 for sample in judged if sample.passed), "of": len(judged)}
        found.append({"run": entry.get("run"), "step": entry.get("step"), "verdict": entry.get("verdict"),
                      "n": entry.get("n"), "arms": arms})
    return found
