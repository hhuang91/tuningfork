"""Forks: relaunch a recorded run's harness, replay the recording up to a step, go live.

`prepare_fork` checks everything before a harness starts — that the run can be relaunched
at all, that every call before the step can be answered from the recording as it was, that
the proxy serves launches into this store — and `execute_forks` runs one or many.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tuningfork.analysis.conditions import TIMED_OUT, fork_counted
from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress, emit
from tuningfork.ops.trash import refuse_trashed
from tuningfork.ops.experiments import load
from tuningfork.ops.launch import ProxyStatus, check_launchable, check_proxy
from tuningfork.store.db import Store


@dataclass(slots=True)
class ForkPlan:
    source: Any  # runner.ForkSource
    checkpoint: Any
    harness: str
    model: str
    proxy: ProxyStatus
    base_url: str
    workdir: Path
    #: Recorded calls the fork will be answered from, all threads counted.
    replayed: int
    variant: Any = None  # replay.Variant
    #: Where the original wording sits at the step, when a variant is swapped in.
    spans: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The harness adapter, running the executable the recordings folder names, if any.
    adapter: Any = None
    #: The sandbox the fork runs in, as its source did (`tuningfork.sandbox.Sandbox`), or
    #: None: the source ran in its own folders, and so does the fork.
    sandbox: Any = None
    #: The variables its check may read besides those its task names, read from the configs
    #: here, once (`specs.configured_check_env`).
    check_env: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        from dataclasses import asdict

        variant = self.source.variant or {}
        return {
            "checkpoint": self.checkpoint.to_json(),
            "harness": self.harness,
            "model": self.model,
            "proxy": self.proxy.url,
            "max_inflight": self.proxy.max_inflight,
            "replayed": self.replayed,
            "breakpoints": self.source.breakpoints,
            "variant": {k: v for k, v in variant.items() if not k.endswith("_text")} or None,
            "spans": [asdict(span) for span in self.spans],
            "sandbox": self.sandbox.to_json() if self.sandbox is not None else None,
            "warnings": self.warnings,
        }


def prepare_fork(
    store: Store,
    run: str,
    step: str,
    *,
    base_url: str | None = None,
    spec: Path | None = None,
    variant: Path | None = None,
    original: Path | None = None,
    breakpoints: bool = True,
    workdir: Path | None = None,
    need_check: bool = False,
    proxy: ProxyStatus | None = None,
    variant_text: str | None = None,
) -> ForkPlan:
    """Everything a fork needs, checked before a harness starts. `proxy` is for a caller
    inside the proxy itself, which knows it without asking over HTTP. With `variant_text`,
    `variant` is where that text will be saved when the fork starts."""
    from tuningfork.control import default_proxy
    from tuningfork.experiment import locate
    from tuningfork.harness import ADAPTERS
    from tuningfork.paths import host_path
    from tuningfork.replay import Variant, replay_problems, replayed_steps
    from tuningfork.runner import ForkSource
    from tuningfork.specs import Eval, check_kind, configured_check_env, load_spec

    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    source_run, k = checkpoint.run, checkpoint.step.idx
    if source_run.is_thread:
        raise OperationError(
            f"run {source_run.id[:8]} is a {source_run.role} thread",
            hint=f"fork its run, {source_run.parent_run_id[:8] if source_run.parent_run_id else 'the main one'}",
        )

    launch = store.get_launch(source_run.launch_id) if source_run.launch_id else None
    initial_commit = skill_commit = None
    if launch is not None and launch.eval:
        item = Eval.from_dict(launch.eval)
        spec_dir = Path(launch.eval.get("spec_dir") or ".")
        skill = Path(launch.skill_dir) if launch.skill_dir else None
        harness = launch.harness
        initial_commit, skill_commit = launch.initial_commit, launch.skill_commit
    else:
        if spec is None:
            raise OperationError(
                f"run {source_run.id[:8]} was recorded before tuningfork kept what each run "
                "was launched with",
                hint="pass --spec with the task spec it was run from",
            )
        try:
            task_spec = load_spec(Path(host_path(str(spec))))
            item = task_spec.get((source_run.task_id or "").split("/", 1)[-1] or None)
        except (OSError, ValueError, KeyError) as exc:
            raise OperationError(f"cannot use {spec}: {exc}") from None
        spec_dir = task_spec.directory
        snapshot = source_run.parity.get("skill_snapshot_dir")
        skill = Path(snapshot) if snapshot else task_spec.resolved_skill_dir()
        harness = source_run.harness
    if not harness or harness not in ADAPTERS or harness == "generic":
        raise OperationError(
            f"run {source_run.id[:8]} was not run by a harness tuningfork can launch ({harness or 'unknown'})"
        )
    from tuningfork.ops.connections import adapter_for, cached_prompts_warning, check_harness

    # Laid out as the source run was: what it replays names its paths. A launch from before
    # layouts were recorded was laid out flat. Set up as it was, too: a harness option
    # changes what the model is told (Hermes's self_improvement, its system prompt), and a
    # fork told otherwise departs from the recording at its first step.
    recorded = dict((launch.settings.get("harness") if launch else None) or {})
    servers = recorded.pop("mcp_servers", None)
    adapter = adapter_for(
        store.home, harness, layout=(launch.settings.get("layout") if launch else None) or "flat",
        options=recorded or None,
    )
    check_launchable(adapter)
    workdir = Path(workdir) if workdir else store.home / "forks"
    sandbox = None
    if launch is not None and launch.settings.get("sandbox"):
        # What it replays names the sandbox's paths, and what the model saw was the same
        # home and folder: a fork of it runs there too, or not at all.
        from tuningfork.sandbox import for_launches, recorded_hiding

        # And its clock starts where its source's did, so its live steps read the
        # recording's date; a source made on the real clock forks on the real clock. It is
        # kept from what its source was kept from, too: the source batch's runs among them.
        hid, mine = recorded_hiding(launch.settings.get("sandbox"), task=spec_dir.resolve())
        sandbox, unsandboxed = for_launches(
            [store.home, workdir, spec_dir, *hid], own=[store.home, workdir, *mine],
            clock=(launch.settings.get("sandbox") or {}).get("clock"),
        )
        if sandbox is None:
            raise OperationError(
                f"run {source_run.id[:8]} ran in the sandbox, so its forks must: {unsandboxed}",
                hint="install bubblewrap, or fork a run launched without it",
                code="no_sandbox",
            )
    if skill is not None and not skill.exists() and not skill_commit:
        raise OperationError(f"the skill folder the run was given, {skill}, no longer exists")
    # Read once, here, for a check that runs a command: a list that cannot be read is refused
    # before a fork is launched, and every fork is judged with the same.
    every_check = configured_check_env(store.home) if check_kind(item.tuningfork.check) == "shell" else []
    if need_check and not item.tuningfork.check:
        raise OperationError(
            "the task defines no check, so a fork has no verdict to count",
            hint="add tuningfork.check to the eval",
        )

    problems = replay_problems(store, source_run, k)
    if problems:
        raise OperationError(
            f"run {source_run.id[:8]} cannot be replayed up to step {k}",
            detail="\n".join(problems),
        )

    swapped = None
    spans: list[Any] = []
    warnings: list[str] = []
    if sandbox is not None and sandbox.clock_problem:
        warnings.append(
            f"{sandbox.clock_problem}: the forks read today's date where the recording read "
            f"{(launch.settings.get('sandbox') or {}).get('clock') if launch else None}"
        )
    if servers:
        warnings.append(
            f"run {source_run.id[:8]} had MCP server(s) {', '.join(sorted(servers))}, which a launch "
            "records by name only: its forks run without them, and may depart from the recording"
        )
    if variant is not None:
        variant = Path(host_path(str(variant)))
        original = Path(host_path(str(original))) if original else (skill / "SKILL.md" if skill else None)
        if original is None or not original.exists():
            raise OperationError(
                f"{original} does not exist" if original else "no original skill file to swap out",
                detail="pass --original: the skill file as it was when the run was recorded",
                field="original",
            )
        if variant_text is None and not variant.exists():
            raise OperationError(f"{variant} does not exist", field="variant")
        original_text = original.read_text(encoding="utf-8")
        if variant_text is None:
            variant_text = variant.read_text(encoding="utf-8")
        if original_text == variant_text:
            raise OperationError("the variant and the original are the same text", field="variant_text")
        swapped = Variant(
            original_text=original_text,
            variant_text=variant_text,
            original_path=str(original),
            variant_path=str(variant),
        )
        spans = locate(checkpoint.request, original_text)
        if not spans:
            warnings.append(
                f"{original.name} is not in the context at step {k}; it will be swapped in "
                "wherever it appears after that"
            )

    if proxy is not None:
        base = base_url or f"{proxy.url}/v1"
    else:
        base = base_url or f"{default_proxy()}/v1"
        proxy = check_proxy(base, store.home)
    # What the harness needs of the server: refused here, or noted when never measured.
    warnings.extend(check_harness(store, adapter, proxy))
    cached = cached_prompts_warning(store.server_checks((proxy.raw.get("server") or {}).get("id")))
    if cached:
        warnings.append(cached)
    serving = (proxy.raw.get("server") or {}).get("id")
    recorded = checkpoint.step.server_id
    if serving and recorded and serving != recorded:
        was, now = store.server(recorded) or {}, proxy.raw.get("server") or {}
        changed = [
            f"{key} {was.get(key)!r} → {now.get(key)!r}"
            for key in ("build", "model_file", "n_ctx", "chat_template_sha256", "slots", "defaults")
            if was.get(key) != now.get(key)
        ]
        warnings.append(
            "the run was recorded on another server configuration"
            + (f" ({'; '.join(changed)})" if changed else "")
            + ": calls the harness makes may not match the recording, and each that does not "
            "takes its fork off it"
        )
    return ForkPlan(
        source=ForkSource(
            run=source_run,
            step=k,
            item=item,
            spec_dir=spec_dir,
            skill=skill,
            variant=swapped.to_json() if swapped else {},
            breakpoints=breakpoints,
            initial_commit=initial_commit,
            skill_commit=skill_commit,
            context=(launch.settings.get("context") if launch is not None else None) or None,
            stop=(launch.settings.get("stop_at_next_skill") if launch is not None else None) or None,
        ),
        checkpoint=checkpoint,
        harness=harness,
        model=source_run.model or "",
        proxy=proxy,
        base_url=base,
        workdir=workdir,
        replayed=len(replayed_steps(store, source_run, k)),
        variant=swapped,
        spans=spans,
        warnings=warnings,
        adapter=adapter,
        sandbox=sandbox,
        check_env=every_check,
    )


def execute_forks(
    store: Store,
    plan: ForkPlan,
    n: int = 1,
    *,
    concurrency: int = 1,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    experiment_id: str | None = None,
    arm: str | None = None,
    on_outcome: Callable[[Any], None] | None = None,
) -> list[Any]:
    """Run `n` forks of the plan; each is a `runner.RunOutcome` whose run is the fork.

    Reports `launch` as each fork's harness is launched and `run` as each finishes, after
    `on_outcome` has seen it (where an experiment stores it as a sample).
    """
    from tuningfork.harness import ADAPTERS
    from tuningfork.runner import ForkRunner

    from tuningfork.harness.base import HarnessError

    runner = ForkRunner(
        store,
        plan.adapter or ADAPTERS[plan.harness](),
        base_url=plan.base_url,
        model=plan.model,
        workdir=plan.workdir,
        concurrency=concurrency,
        sandbox=plan.sandbox,
        check_env=plan.check_env,
    )
    unregister = (
        cancel.on_cancel(lambda: runner.cancel.cancel(cancel.reason or "cancelled"))
        if cancel is not None
        else None
    )
    runner.on_launch = lambda launch, index: emit(
        progress, "launch", subject=launch,
        launch_id=launch.id, index=index, label=launch.label, arm=arm,
    )

    def finished_one(outcome: Any) -> None:
        if on_outcome is not None:
            on_outcome(outcome)
        emit(progress, "run", subject=outcome, arm=arm, **outcome.to_json())

    try:
        return runner.fork(plan.source, n, on_progress=finished_one, experiment_id=experiment_id)
    except HarnessError as exc:
        raise OperationError(str(exc)) from None
    finally:
        if unregister is not None:
            unregister()


# --- to the end: the done criterion of Phase 4 ----------------------------------------------


def check_fork_concurrency(proxy: ProxyStatus, concurrency: int) -> None:
    """Forks at once must fit through the proxy's gate: a fork left waiting there past its
    harness's own timeout fails, and would be judged."""
    if concurrency < 1:
        raise OperationError("concurrency must be at least 1", field="concurrency")
    if concurrency > proxy.max_inflight:
        raise OperationError(
            f"the proxy lets {proxy.max_inflight} call(s) through at once, fewer than "
            f"--concurrency {concurrency}: forks would queue at its gate and time out",
            hint=f"restart it with --max-inflight {concurrency}",
            field="concurrency",
        )


@dataclass(slots=True)
class ToEndPlan:
    """Forks from a step to the end, judged by the task's check: one arm, or an A/B's two."""

    original: ForkPlan
    #: The arm with the variant wording swapped in; None for a resample.
    variant: ForkPlan | None
    n: int
    concurrency: int = 1
    alpha: float = 0.05

    @property
    def checkpoint(self) -> Any:
        return self.original.checkpoint

    @property
    def forks(self) -> int:
        return self.n * (2 if self.variant is not None else 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "ab" if self.variant is not None else "resample",
            "reach": "end",
            "n": self.n,
            "forks": self.forks,
            "concurrency": self.concurrency,
            "alpha": self.alpha if self.variant is not None else None,
            "target": {"kind": "check", "description": "the task's check passes"},
            "original": self.original.to_json(),
            "variant": self.variant.to_json() if self.variant is not None else None,
        }


def prepare_to_end(
    store: Store,
    run: str,
    step: str,
    *,
    n: int,
    concurrency: int = 1,
    variant: Path | None = None,
    original: Path | None = None,
    alpha: float = 0.05,
    base_url: str | None = None,
    spec: Path | None = None,
    workdir: Path | None = None,
    proxy: ProxyStatus | None = None,
    variant_text: str | None = None,
) -> ToEndPlan:
    """A resample (or, with `variant`, an A/B) from a step to the end: breakpoints off, the
    task's check required, and no more forks at once than the proxy lets through."""
    if n < 1:
        raise OperationError("n must be at least 1", field="n")
    options: dict[str, Any] = dict(
        base_url=base_url, spec=spec, workdir=workdir, breakpoints=False, need_check=True,
    )
    swapped = None
    if variant is not None:
        swapped = prepare_fork(
            store, run, step, variant=variant, original=original, proxy=proxy,
            variant_text=variant_text, **options
        )
        proxy = swapped.proxy
    plain = prepare_fork(store, run, step, proxy=proxy, **options)
    check_fork_concurrency(plain.proxy, concurrency)
    return ToEndPlan(original=plain, variant=swapped, n=n, concurrency=concurrency, alpha=alpha)


def execute_to_end_plan(
    store: Store,
    plan: ToEndPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    tag: dict[str, Any] | None = None,
) -> EndToEnd | ABEndToEnd:
    """`tag` is kept in each experiment's settings: what started it, so it can be found."""
    if plan.variant is None:
        return execute_to_end(
            store, plan.original, plan.n, concurrency=plan.concurrency,
            progress=progress, cancel=cancel, tag=tag,
        )
    return execute_ab_to_end(
        store, plan.original, plan.variant, plan.n, alpha=plan.alpha,
        concurrency=plan.concurrency, progress=progress, cancel=cancel, tag=tag,
    )


@dataclass(slots=True)
class EndToEnd:
    """One arm of forks run to the end, judged by the task's check."""

    experiment: Any
    passed: int
    counted: int
    #: The pass rate over the forks that count, with its Wilson interval.
    rate: Any
    #: (fork run, why it does not count) for each fork left out.
    exclusions: list[tuple[str | None, str]]
    #: What the model did at the step, over the forks that count.
    actions: Any
    #: The forks as just run; empty when read back from the store.
    outcomes: list[Any] = field(default_factory=list)
    elapsed: float = 0.0
    #: Every fork of the arm, in order, as stored: (sample index, its run, whether the
    #: check passed, why it does not count).
    forks: list[tuple[int, str | None, bool | None, str | None]] = field(default_factory=list)
    #: Forks that timed out: counted among the failures, as the gate counts them.
    timeouts: int = 0

    def to_json(self) -> dict[str, Any]:
        """One arm of forks run to the end: its pass rate, what it left out, what the model
        did at the step, and each fork — forks are not in the list of runs."""
        return {
            "passed": self.passed,
            "counted": self.counted,
            "rate": self.rate.to_json() if self.rate else None,
            "exclusions": [{"run_id": run_id, "why": why} for run_id, why in self.exclusions],
            "timeouts": self.timeouts,
            "actions": self.actions.to_json(),
            "forks": [
                {"idx": idx, "run_id": run_id, "passed": passed, "exclusion": why, "timed_out": why == TIMED_OUT}
                for idx, run_id, passed, why in self.forks
            ],
        }


@dataclass(slots=True)
class ABEndToEnd:
    original: EndToEnd
    variant: EndToEnd
    #: Fisher's exact test on pass counts; None when an arm has no fork that counts.
    p_value: float | None
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < self.alpha

    @property
    def direction(self) -> str | None:
        """Which way the variant moved the pass rate — "up", "down" or "unchanged" — as a
        step's A/B says of its target; None when an arm has no fork that counts."""
        if self.original.rate is None or self.variant.rate is None:
            return None
        if self.variant.rate.point > self.original.rate.point:
            return "up"
        if self.variant.rate.point < self.original.rate.point:
            return "down"
        return "unchanged"

    def to_json(self) -> dict[str, Any]:
        return {
            "original": self.original.to_json(),
            "variant": self.variant.to_json(),
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "direction": self.direction,
        }


def arm_summary(store: Store, experiment: Any, outcomes: list[Any] = ()) -> EndToEnd:
    """An arm's result, from its stored samples — the same whether just run or read back."""
    from tuningfork.analysis.stats import rate
    from tuningfork.experiment import experiment_distribution

    samples = store.samples(experiment.id)
    # A fork that timed out failed, as the gate counts it; the rest the check decided, or left out.
    verdicts = [fork_counted(s.passed, s.exclusion) for s in samples]
    counted = [verdict for verdict in verdicts if verdict is not None]
    passed = sum(counted)
    return EndToEnd(
        experiment=experiment,
        passed=passed,
        counted=len(counted),
        rate=rate(passed, len(counted)) if counted else None,
        exclusions=[(s.run_id, s.exclusion) for s, verdict in zip(samples, verdicts) if s.exclusion and verdict is None],
        actions=experiment_distribution(store, experiment),
        outcomes=list(outcomes),
        forks=[(s.idx, s.run_id, s.passed, s.exclusion) for s in samples],
        timeouts=sum(1 for s in samples if s.exclusion == TIMED_OUT),
    )


def compare_to_end(original: EndToEnd, variant: EndToEnd, alpha: float = 0.05) -> ABEndToEnd:
    from tuningfork.analysis.stats import fisher_exact

    p_value = None
    if original.counted and variant.counted:
        p_value = fisher_exact(
            original.passed, original.counted - original.passed,
            variant.passed, variant.counted - variant.passed,
        )
    return ABEndToEnd(original=original, variant=variant, p_value=p_value, alpha=alpha)


def execute_to_end(
    store: Store,
    plan: ForkPlan,
    n: int,
    *,
    kind: str = "resample",
    arm: str = "original",
    group_id: str | None = None,
    concurrency: int = 1,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    finalize: bool = True,
    tag: dict[str, Any] | None = None,
    variant_meta: dict[str, Any] | None = None,
) -> EndToEnd:
    """`n` forks from the plan's step to the end, stored as one arm of an experiment.
    `variant_meta` is kept with the arm's variant: where its wording came from, as a screen
    names its proposal.

    Each fork is a sample: whether the task's check passed, and — so the report can say
    what the model did where the forks began — its answer at the step itself. A fork that
    does not count keeps its reason, and is left out of both.
    """
    import time

    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.store.masking import Masked

    checkpoint = plan.checkpoint
    variant = plan.variant.to_json() if plan.variant else {}
    experiment = store.create_experiment(
        kind=kind,
        arm=arm,
        run_id=checkpoint.run.id,
        step_idx=checkpoint.step.idx,
        step_id=checkpoint.step.id,
        # Read back from the store, masked as it stores it.
        request=Masked(checkpoint.request),
        settings={
            "n": n,
            "concurrency": concurrency,
            "reach": "end",
            "harness": plan.harness,
            "model": plan.model,
            "target": {"kind": "check", "description": "the task's check passes"},
            **(tag or {}),
        },
        variant={**{k: v for k, v in variant.items() if not k.endswith("_text")}, **(variant_meta or {})},
        base_url=plan.proxy.url,
        notes=list(plan.warnings),
        group_id=group_id,
    )
    emit(
        progress, "experiment", subject=experiment,
        id=experiment.id, kind=kind, arm=arm, group_id=group_id, n=n, reach="end",
    )
    k = checkpoint.step.idx

    def record(outcome: Any) -> None:
        live = None
        if outcome.run_id:
            live = next(
                (s for s in store.steps(outcome.run_id) if s.idx == k and not s.served), None
            )
        response = store.response(live) if live else None
        found = recorded_outcome(response, live.error if live else "never reached the step")
        store.add_sample(
            experiment.id,
            outcome.index,
            seed=None,
            # The fork's step, as the store keeps it: masked already.
            response=Masked(response),
            outcome_kind=found.kind,
            outcome_key=found.key,
            outcome_label=found.label,
            finish_reason=live.finish_reason if live else None,
            usage=live.usage if live else None,
            timings={"total_ms": round(outcome.duration_s * 1000, 1)},
            reasoning_chars=live.reasoning_chars if live else 0,
            # An excluded fork is an error to every rate, the step's included.
            error=outcome.exclusion,
            run_id=outcome.run_id,
            passed=outcome.passed if outcome.checked else None,
            exclusion=outcome.exclusion,
        )

    started = time.monotonic()
    try:
        outcomes = execute_forks(
            store, plan, n, concurrency=concurrency, progress=progress, cancel=cancel,
            experiment_id=experiment.id, arm=arm, on_outcome=record,
        )
    except Cancelled as stopped:
        # The forks that finished are stored samples; the one killed has none.
        if finalize:
            finish(store, [experiment.id], stopped)
        summary = arm_summary(store, experiment, stopped.partial or [])
        summary.elapsed = time.monotonic() - started
        raise Cancelled(stopped.reason, partial=summary) from None
    except BaseException as error:
        if finalize:
            finish(store, [experiment.id], error)
        raise
    if finalize:
        finish(store, [experiment.id], None)
    summary = arm_summary(store, experiment, outcomes)
    summary.elapsed = time.monotonic() - started
    return summary


def finish(store: Store, ids: list[str], error: BaseException | None) -> None:
    """Say how experiments ended, from how the work that ran them did."""
    if error is None:
        status = "done"
    elif isinstance(error, Cancelled):
        status = "interrupted" if error.reason == "shutdown" else "cancelled"
    elif isinstance(error, KeyboardInterrupt):
        status = "cancelled"
    else:
        status = "failed"
    store.finish_experiments(ids, status)


def execute_ab_to_end(
    store: Store,
    original: ForkPlan,
    variant: ForkPlan,
    n: int,
    *,
    alpha: float = 0.05,
    concurrency: int = 1,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    tag: dict[str, Any] | None = None,
) -> ABEndToEnd:
    """Both arms to the end, the original first, compared on the check's pass rate —
    the target the check declared before any fork ran."""
    import uuid

    group = uuid.uuid4().hex
    arms: list[EndToEnd] = []
    # Both arms stay running until the whole A/B has ended, as a checkpoint A/B's do.
    created: list[str] = []

    def note(event: Any) -> None:
        if event.kind == "experiment":
            created.append(event.subject.id)
        if progress is not None:
            progress(event)

    for name, plan in (("original", original), ("variant", variant)):
        try:
            arms.append(
                execute_to_end(
                    store, plan, n, kind="ab", arm=name, group_id=group,
                    concurrency=concurrency, progress=note, cancel=cancel, finalize=False,
                    # The level it is judged at, kept with both arms as a step A/B keeps it.
                    tag={**(tag or {}), "alpha": alpha},
                )
            )
        except Cancelled as stopped:
            finish(store, created, stopped)
            done = [*arms, stopped.partial] if stopped.partial is not None else arms
            partial = compare_to_end(done[0], done[1], alpha) if len(done) == 2 else (
                done[0] if done else None
            )
            raise Cancelled(stopped.reason, partial=partial) from None
        except BaseException as error:
            finish(store, created, error)
            raise
    finish(store, created, None)
    return compare_to_end(arms[0], arms[1], alpha)
