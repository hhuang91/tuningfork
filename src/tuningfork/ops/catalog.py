"""The operations a front end can start as jobs: what each is called, what it takes, and
how it is prepared and run.

The job runner (`tuningfork.jobs`) never knows what it runs; it looks a kind up here. The
web UI renders its forms' defaults from `GET /jobs/kinds`, which is this catalog's JSON, and
the MCP server of Phase 6 takes the same list as its tools. Parameters are dataclasses read
strictly (`ops.common.parse_params`): an unknown key, or a value of a near-miss type, is
refused with the field named, never guessed at.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from tuningfork.ops.common import OperationError, params_schema, parse_params

BACKGROUND, INTERACTIVE = "background", "interactive"


@dataclass(frozen=True, slots=True)
class JobContext:
    """What a job's operation can use: the store, and the proxy it runs inside."""

    store: Any  # store.db.Store
    proxy: Any  # proxy.local.LocalProxy
    job_id: str

    @property
    def home(self) -> Any:
        return self.store.home

    @property
    def model_url(self) -> str:
        """Where a job's model calls go: the proxy itself, so they share its gate and its
        pause switch, marked as an experiment's so they are never recorded or held."""
        if not self.proxy.url:
            raise OperationError("the proxy does not know its own address yet")
        return self.proxy.url


@dataclass(frozen=True, slots=True)
class Kind:
    name: str
    lane: str
    #: One line for a menu, and for a tool list.
    summary: str
    params: type
    prepare: Callable[[JobContext, Any], Any]
    #: Returns the result, or a coroutine the job runner runs to get it.
    execute: Callable[..., Any]
    #: How many units of progress (samples, runs) the prepared plan will report.
    total: Callable[[Any], int | None] = lambda prepared: None

    def parse(self, data: Any) -> Any:
        return parse_params(self.params, data)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lane": self.lane,
            "summary": self.summary,
            "params": params_schema(self.params),
        }


def _help(text: str, **bounds: Any) -> dict[str, Any]:
    return {"help": text, **bounds}


#: Shapes a type cannot say, for `params_schema`. The rules are `analysis.distribution.RULES`
#: (a test keeps them the same).
RULE_LIST = {"type": "array", "items": {
    "type": "object",
    "properties": {
        "if": {"type": "string", "enum": ["calls", "not_calls", "contains", "not_contains"]},
        "value": {"type": "string"},
    },
    "required": ["if", "value"],
    "additionalProperties": False,
}}
FILE_LIST = {"type": "array", "items": {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"],
    "additionalProperties": False,
}}
CHECK = {"anyOf": [
    {"type": "string"},
    {
        "type": "object",
        "properties": {"reply_contains": {"type": "string"}, "reply_matches": {"type": "string"}},
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 1,
    },
]}


# --- sampling at a step --------------------------------------------------------------------


@dataclass
class AtStep:
    """A recorded step, as a person names it."""

    run: str = field(metadata=_help("The run: its id, a prefix of it, or its label."))
    step: str = field(metadata=_help("The step: its number, or the tool its model called first."))


#: Said once for every kind that samples: what the seeds and the prompt cache do.
SEED_HELP = ("Sample i is sent with seed base+i. None: a new random base, recorded with the experiment, so a "
             "second experiment draws new samples; an earlier base draws those again.")
CACHE_HELP = ("Let the server reuse its cache of the prompt between samples: faster, but on llama.cpp a sample's "
              "behaviour can then depend on what the server ran before. Off: each prompt is processed whole.")


@dataclass
class Sampling(AtStep):
    """How a step is sampled; shared by the kinds that sample one."""

    n: int = field(default=20, metadata=_help("Samples to draw (per arm, for an A/B).", minimum=1, maximum=1000))
    concurrency: int = field(
        default=1,
        metadata=_help("Samples in flight at once. The proxy's --max-inflight still caps what reaches the server.", minimum=1, maximum=64),
    )
    max_tokens: int = field(default=4096, metadata=_help("Cap per sample; hitting it counts as truncated.", minimum=1, maximum=262144))
    seed_base: int | None = field(default=None, metadata=_help(SEED_HELP, minimum=0))
    prompt_cache: bool = field(default=False, metadata=_help(CACHE_HELP))
    normalizers: list[str] = field(default_factory=list, metadata=_help("Restrict normalizers; none named means the defaults."))


@dataclass
class ResampleParams(Sampling):
    temperature: float | None = field(default=None, metadata=_help("Override the recorded temperature (recorded as a change).", minimum=0, maximum=5))
    top_p: float | None = field(default=None, metadata=_help("Override top_p (recorded as a change).", minimum=0, maximum=1))
    top_k: int | None = field(default=None, metadata=_help("Override top_k (recorded as a change).", minimum=0))
    logprobs: int | None = field(default=None, metadata=_help("Keep each sample's tokens with this many alternatives each, for its heatmap. Chat-completions steps only.", minimum=1, maximum=20))


def _checkpoint(params: Any) -> None:
    for name in ("run", "step"):
        if not getattr(params, name):
            raise OperationError(f"{name} is required", code="bad_params", field=name)


def _sampling(params: Any) -> dict[str, Any]:
    return {
        "n": params.n,
        "concurrency": params.concurrency,
        "max_tokens": params.max_tokens,
        "seed_base": params.seed_base,
        "prompt_cache": params.prompt_cache,
    }


def prepare_resample_job(context: JobContext, params: ResampleParams) -> Any:
    from tuningfork.ops.experiments import prepare_resample

    _checkpoint(params)
    return prepare_resample(
        context.store, params.run, params.step, base_url=context.model_url,
        normalizers=params.normalizers,
        overrides={"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k},
        logprobs=params.logprobs,
        **_sampling(params),
    )


def execute_resample_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.experiments import execute_resample

    return execute_resample(context.store, prepared, progress=progress, cancel=cancel)


# --- tokens: a recorded answer scored -----------------------------------------------------


@dataclass
class TokensParams(AtStep):
    top_k: int = field(default=8, metadata=_help("Alternatives to keep at each token: the heatmap's entropy counts the rest as one.", minimum=1, maximum=20))


def prepare_tokens_job(context: JobContext, params: TokensParams) -> Any:
    from tuningfork.ops.tokens import prepare_score

    _checkpoint(params)
    return prepare_score(context.store, params.run, params.step, top_k=params.top_k, base_url=context.model_url)


def execute_tokens_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.tokens import execute_score

    return execute_score(context.store, prepared, progress=progress, cancel=cancel)


# --- forking tokens: which forced tokens change what the model does ------------------------


@dataclass
class ForkingParams(AtStep):
    positions: list[int] = field(default_factory=list, metadata=_help("Positions to probe, as the heatmap numbers them. None: the most likely forks in the region."))
    stream: str | None = field(default=None, metadata=_help("The token stream to fork from, by id: a sample's. Default: the step's own answer, scored."))
    region: Literal["answer", "reasoning", "all"] = field(default="answer", metadata=_help("Where to look for forks when no positions are given."))
    top: int = field(default=5, metadata=_help("Positions to probe when none are given.", minimum=1, maximum=20))
    alternatives: int = field(default=3, metadata=_help("Other tokens to force at each position.", minimum=1, maximum=10))
    min_p: float = field(default=0.05, metadata=_help("Least chance the step's sampler gave an alternative, to count as a fork.", minimum=0.001, maximum=0.5))
    n: int = field(default=10, metadata=_help("Continuations per forced token.", minimum=1, maximum=200))
    max_tokens: int = field(default=4096, metadata=_help("Cap per continuation; hitting it counts as truncated.", minimum=1, maximum=262144))
    normalizers: list[str] = field(default_factory=list, metadata=_help("Restrict normalizers; none named means the defaults."))
    target: str | None = field(default=None, metadata=_help("The target: a behaviour's letter from an earlier resample of this step. Default: the base's own behaviour."))
    target_match: str | None = field(default=None, metadata=_help("The target: a regex over what the model did."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help("The target, described: conditions that must all hold, each {if: calls | not_calls | contains | not_contains, value}.", schema=RULE_LIST))
    alpha: float = field(default=0.05, metadata=_help("Significance level, Holm-corrected over the planned tests.", minimum=0.0001, maximum=0.5))
    seed_base: int | None = field(default=None, metadata=_help(SEED_HELP, minimum=0))
    concurrency: int = field(default=1, metadata=_help("Continuations at once. On one GPU, one at a time keeps the prompt cache.", minimum=1, maximum=16))


def prepare_forking_job(context: JobContext, params: ForkingParams) -> Any:
    from tuningfork.ops.forking import prepare_forking

    _checkpoint(params)
    return prepare_forking(
        context.store, params.run, params.step, stream=params.stream, positions=params.positions,
        top=params.top, region=params.region, alternatives=params.alternatives, min_p=params.min_p,
        n=params.n, max_tokens=params.max_tokens, normalizers=params.normalizers, target=params.target,
        target_match=params.target_match, target_rules=params.target_rules or None, alpha=params.alpha,
        base_url=context.model_url, seed_base=params.seed_base, concurrency=params.concurrency,
    )


def execute_forking_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.forking import execute_forking

    return execute_forking(context.store, prepared, progress=progress, cancel=cancel)


# --- A/B of a wording ---------------------------------------------------------------------


@dataclass
class ABParams(Sampling):
    variant: str | None = field(default=None, metadata=_help("The variant skill file, a path on the proxy's machine."))
    variant_text: str | None = field(default=None, metadata=_help("The variant's text, when edited in the browser: saved as a variant when the A/B starts."))
    variant_name: str | None = field(default=None, metadata=_help("The name to save variant_text under."))
    original: str | None = field(default=None, metadata=_help("The skill file as the run was given it. Default: the copy its batch kept."))
    skill: str | None = field(default=None, metadata=_help("Which skill the variant replaces, by name, where the run kept several: an imported chat."))
    target: str | None = field(default=None, metadata=_help("The target: a behaviour's letter from an earlier resample of this step."))
    target_match: str | None = field(default=None, metadata=_help("The target: a regex over what the model did."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help("The target, described: conditions that must all hold, each {if: calls | not_calls | contains | not_contains, value}.", schema=RULE_LIST))
    exploratory: bool = field(default=False, metadata=_help("No target: every behaviour is compared, Bonferroni-corrected."))
    alpha: float = field(default=0.05, metadata=_help("Significance level.", minimum=0.0001, maximum=0.5))


def prepare_ab_job(context: JobContext, params: ABParams) -> Any:
    from tuningfork.ops import variants
    from tuningfork.ops.experiments import load, prepare_ab

    _checkpoint(params)
    declared = [name for name in ("target", "target_match", "target_rules") if getattr(params, name)]
    if params.exploratory:
        declared.append("exploratory")
    if len(declared) != 1:
        raise OperationError(
            "declare what the A/B is about before sampling: a behaviour's letter from an "
            "earlier resample, a pattern, a behaviour described — or say it is exploratory"
            if not declared
            else f"declare one target, not {' and '.join(declared)}",
            code="bad_params",
            field="target",
        )
    if bool(params.variant) == bool(params.variant_text):
        raise OperationError(
            "give the variant either as a file (variant) or as text (variant_text)",
            code="bad_params",
            field="variant",
        )
    variant = params.variant
    if params.variant_text is not None:
        if not params.variant_name:
            raise OperationError("name the variant, to save it under", code="bad_params", field="variant_name")
        checkpoint = load(context.store, params.run, params.step)
        variant = str(variants.path_for(context.home, _skill_of(context, checkpoint.run, params.skill), params.variant_name))
    return prepare_ab(
        context.store, params.run, params.step,
        variant=variant, variant_text=params.variant_text, original=params.original, skill=params.skill,
        target=params.target, target_match=params.target_match, target_rules=params.target_rules or None,
        alpha=params.alpha, base_url=context.model_url, normalizers=params.normalizers, **_sampling(params),
    )


def execute_ab_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops import variants
    from tuningfork.ops.experiments import execute_ab

    # A wording edited in the browser is written now, and only now.
    if not prepared.variant.exists():
        skill, name = prepared.variant.parent.name, prepared.variant.stem
        variants.save_variant(context.home, skill, name, prepared.variant_text)
    return execute_ab(context.store, prepared, progress=progress, cancel=cancel)


# --- to the end: forks judged by the task's check ------------------------------------------


@dataclass
class ToEndParams(AtStep):
    n: int = field(default=5, metadata=_help("Forks to run, each to the end.", minimum=1, maximum=200))
    concurrency: int = field(
        default=1,
        metadata=_help("Forks at once; no more than the proxy's --max-inflight. On one GPU, one at a time is fastest.", minimum=1, maximum=16),
    )


@dataclass
class ABToEndParams(ToEndParams):
    variant: str | None = field(default=None, metadata=_help("The variant skill file, a path on the proxy's machine."))
    variant_text: str | None = field(default=None, metadata=_help("The variant's text, when edited in the browser: saved as a variant when the A/B starts."))
    variant_name: str | None = field(default=None, metadata=_help("The name to save variant_text under."))
    original: str | None = field(default=None, metadata=_help("The skill file as the run was given it. Default: the copy its launch kept."))
    alpha: float = field(default=0.05, metadata=_help("Significance level.", minimum=0.0001, maximum=0.5))


def prepare_to_end_job(context: JobContext, params: ToEndParams) -> Any:
    from tuningfork.ops.forks import prepare_to_end

    _checkpoint(params)
    return prepare_to_end(
        context.store, params.run, params.step, n=params.n, concurrency=params.concurrency,
        proxy=context.proxy.status(),
    )


def prepare_ab_to_end_job(context: JobContext, params: ABToEndParams) -> Any:
    from tuningfork.ops import variants
    from tuningfork.ops.experiments import load
    from tuningfork.ops.forks import prepare_to_end

    _checkpoint(params)
    if bool(params.variant) == bool(params.variant_text):
        raise OperationError(
            "give the variant either as a file (variant) or as text (variant_text)",
            code="bad_params", field="variant",
        )
    variant = params.variant
    if params.variant_text is not None:
        if not params.variant_name:
            raise OperationError("name the variant, to save it under", code="bad_params", field="variant_name")
        checkpoint = load(context.store, params.run, params.step)
        variant = str(variants.path_for(context.home, _skill_of(context, checkpoint.run), params.variant_name))
    return prepare_to_end(
        context.store, params.run, params.step, n=params.n, concurrency=params.concurrency,
        variant=variant, variant_text=params.variant_text, original=params.original,
        alpha=params.alpha, proxy=context.proxy.status(),
    )


def _skill_of(context: JobContext, run: Any, skill: str | None = None) -> str:
    """The skill a run was given, by name — `skill`, where it kept several: where a variant of it
    is kept."""
    from tuningfork.ops import variants
    from tuningfork.ops.experiments import study_skill

    studied = study_skill(run, skill)
    found = studied[1] if studied is not None else None
    if found is None and run.launch_id:
        launch = context.store.get_launch(run.launch_id)
        if launch is not None and launch.skill_dir:
            from pathlib import Path

            candidate = Path(launch.skill_dir) / "SKILL.md"
            found = candidate if candidate.is_file() else None
    return variants.skill_name(run, found)


def execute_to_end_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from pathlib import Path

    from tuningfork.ops import variants
    from tuningfork.ops.forks import execute_to_end_plan

    swapped = prepared.variant.variant if prepared.variant is not None else None
    if swapped is not None and swapped.variant_path and not Path(swapped.variant_path).exists():
        # A wording edited in the browser is written now, and only now.
        target = Path(swapped.variant_path)
        variants.save_variant(context.home, target.parent.name, target.stem, swapped.variant_text)
    return execute_to_end_plan(context.store, prepared, progress=progress, cancel=cancel)


# --- a fork: the run replayed to a step, live from there ----------------------------------


@dataclass
class ForkParams(AtStep):
    n: int = field(default=1, metadata=_help("Forks to launch.", minimum=1, maximum=20))
    hold: Literal["step", "breakpoints", "none"] = field(
        default="step",
        metadata=_help(
            "step: stop before the model answers at the step, for you to look, edit, step on — "
            "breakpoints armed on the proxy apply too. breakpoints: only those. none: run to "
            "the end unattended."
        ),
    )
    concurrency: int = field(default=1, metadata=_help("Forks at once; no more than the proxy's --max-inflight.", minimum=1, maximum=16))


@dataclass(slots=True)
class ForkJob:
    plan: Any  # ops.forks.ForkPlan
    n: int
    concurrency: int
    hold: str

    def to_json(self) -> dict[str, Any]:
        return {**self.plan.to_json(), "n": self.n, "concurrency": self.concurrency, "hold": self.hold}


def prepare_fork_job(context: JobContext, params: ForkParams) -> ForkJob:
    from tuningfork.ops.forks import check_fork_concurrency, prepare_fork

    _checkpoint(params)
    plan = prepare_fork(
        context.store, params.run, params.step, proxy=context.proxy.status(),
        breakpoints=params.hold != "none",
    )
    check_fork_concurrency(plan.proxy, params.concurrency)
    if params.hold != "none":
        plan.warnings.append(
            f"the harness is stopped {plan.source.item.tuningfork.timeout:g} s after it starts, "
            "time held at a breakpoint included (the task's timeout)"
        )
    return ForkJob(plan=plan, n=params.n, concurrency=params.concurrency, hold=params.hold)


def execute_fork_job(context: JobContext, prepared: ForkJob, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.forks import execute_forks

    k = prepared.plan.checkpoint.step.idx

    def watch(event: Any) -> None:
        if event.kind == "launch" and prepared.hold == "step":
            # Armed for this launch alone, once, before its harness starts: it stops the
            # fork where it goes live, and nothing else.
            launch_id = event.data["launch_id"]
            context.proxy.run_on_loop(context.proxy.app.state.engine.add, {
                "phase": "request", "launch": launch_id, "step_min": k, "step_max": k,
                "once": True, "label": f"fork {launch_id[:8]} at step {k}",
            })
        if progress is not None:
            progress(event)

    return execute_forks(
        context.store, prepared.plan, prepared.n, concurrency=prepared.concurrency,
        progress=watch, cancel=cancel,
    )


# --- a batch: a task run N times through its harness ---------------------------------------

HARNESS_OPTIONS_HELP = (
    "Harness options, recorded with each run (Hermes: self_improvement, await_review, "
    "review_every_turn, mcp_servers)."
)


@dataclass
class BatchParams:
    scenario: str | None = field(default=None, metadata=_help("A scenario kept in the recordings folder, by name."))
    spec: str | None = field(default=None, metadata=_help("Or a task spec (evals.json), a path on the proxy's machine."))
    eval: str | None = field(default=None, metadata=_help("Which eval of the spec; default the first that is not held out."))
    n: int = field(default=5, metadata=_help("Runs.", minimum=1, maximum=200))
    harness: str = field(default="hermes", metadata=_help("The harness to launch."))
    model: str | None = field(default=None, metadata=_help("The model the harness requests; default the one the server lists."))
    skill_file: str | None = field(default=None, metadata=_help("A file to use as the skill's SKILL.md in every run: to confirm a wording with full runs."))
    skill_dir: str | None = field(default=None, metadata=_help("A skill folder for every run instead of the spec's."))
    concurrency: int = field(default=1, metadata=_help("Runs at once. On one GPU, one at a time is fastest.", minimum=1, maximum=16))
    include_held_out: bool = field(default=False, metadata=_help("Run a held-out eval named in `eval` anyway: its runs are then recorded where agents can read them."))
    harness_options: dict[str, Any] = field(default_factory=dict, metadata=_help(HARNESS_OPTIONS_HELP))


def _spec_of(context: JobContext, params: Any) -> Any:
    from pathlib import Path

    from tuningfork.ops.scenarios import spec_path

    if bool(params.spec) == bool(params.scenario):
        raise OperationError("name a scenario, or a task spec — one of them", code="bad_params", field="scenario")
    return spec_path(context.home, params.scenario) if params.scenario else Path(params.spec)


def prepare_batch_job(context: JobContext, params: BatchParams) -> Any:
    from pathlib import Path

    from tuningfork.ops.batch import prepare_batch

    spec = _spec_of(context, params)
    proxy = context.proxy.status()
    return prepare_batch(
        context.store, spec, eval_id=params.eval, n=params.n, model=params.model,
        harness=params.harness, skill_file=Path(params.skill_file) if params.skill_file else None,
        concurrency=params.concurrency, base_url=f"{proxy.url}/v1", proxy=proxy,
        include_held_out=params.include_held_out, harness_options=params.harness_options or None,
        skill_dir=Path(params.skill_dir) if params.skill_dir else None,
    )


def execute_batch_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.batch import execute_batch

    return execute_batch(context.store, prepared, progress=progress, cancel=cancel)


# --- a chain: a task run N times in order, the skill carried from each run to the next -----


@dataclass
class ChainParams:
    scenario: str | None = field(default=None, metadata=_help("A scenario kept in the recordings folder, by name."))
    spec: str | None = field(default=None, metadata=_help("Or a task spec (evals.json), a path on the proxy's machine."))
    eval: str | None = field(default=None, metadata=_help("Which eval of the spec; default the first that is not held out."))
    n: int = field(default=10, metadata=_help("Links: runs in order, the skill carried forward.", minimum=1, maximum=200))
    harness: str = field(default="hermes", metadata=_help("The harness to launch."))
    model: str | None = field(default=None, metadata=_help("The model the harness requests; default the one the server lists."))
    harness_options: dict[str, Any] = field(default_factory=dict, metadata=_help(HARNESS_OPTIONS_HELP))
    tell: bool = field(default=False, metadata=_help("Tell the harness each run's check result as the user's next turn."))
    budget_seconds: float | None = field(default=None, metadata=_help("Stop once the links' calls have taken this many seconds.", minimum=1))


def prepare_chain_job(context: JobContext, params: ChainParams) -> Any:
    from tuningfork.ops.chain import prepare_chain

    spec = _spec_of(context, params)
    proxy = context.proxy.status()
    return prepare_chain(
        context.store, spec, eval_id=params.eval, n=params.n, model=params.model,
        harness=params.harness, base_url=f"{proxy.url}/v1", proxy=proxy,
        harness_options=params.harness_options or None, tell=params.tell,
        budget_seconds=params.budget_seconds,
    )


def execute_chain_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.chain import execute_chain

    return execute_chain(context.store, prepared, progress=progress, cancel=cancel)


# --- a server check: what a harness needs, measured -------------------------------------------


@dataclass
class ServerCheckParams:
    model: str | None = field(default=None, metadata=_help("The model to check with; default the one the server lists."))


def prepare_server_check_job(context: JobContext, params: ServerCheckParams) -> Any:
    from tuningfork.ops.connections import prepare_server_check

    refresh = getattr(context.proxy, "refresh_server", None)
    if refresh is not None:
        refresh()
    return prepare_server_check(context.proxy.status(), params.model)


def execute_server_check_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.connections import execute_server_check

    return execute_server_check(context.store, prepared, progress=progress)


@dataclass(slots=True)
class CheckTrialParams:
    check: str | dict[str, str] | None = field(default=None, metadata=_help(
        'The check to try, as a spec writes it: a shell command, or {"reply_contains": …} or '
        '{"reply_matches": …}. With a run and none named, the run\'s own.', schema=CHECK))
    run: str | None = field(default=None, metadata=_help("A recorded run to judge again: its workspace as it ended, and its final reply."))
    reply: str = field(default="", metadata=_help("Without a run: the reply to judge."))
    files: list[dict[str, str]] = field(default_factory=list, metadata=_help("Without a run: files for the workspace, each {path, content}.", schema=FILE_LIST))
    timeout: float = field(default=60.0, metadata=_help("Seconds the check may take.", minimum=1, maximum=600))


def prepare_check_trial_job(context: JobContext, params: CheckTrialParams) -> Any:
    from tuningfork.ops.checks import prepare_check_trial

    return prepare_check_trial(
        context.store, params.check, run=params.run, reply=params.reply, files=params.files,
        timeout=params.timeout,
    )


def execute_check_trial_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.checks import execute_check_trial

    return execute_check_trial(context.store, prepared)


@dataclass(slots=True)
class CheckAlongParams:
    task: str | None = field(default=None, metadata=_help("A task id: its recorded runs, not forks, newest first."))
    run: str | None = field(default=None, metadata=_help("Or one run."))
    limit: int = field(default=50, metadata=_help("With a task: at most this many runs.", minimum=1, maximum=500))
    again: bool = field(default=False, metadata=_help("Anew along runs it was run along before."))
    timeout: float = field(default=60.0, metadata=_help("Seconds the check may take, each time.", minimum=1, maximum=600))


def prepare_check_along_job(context: JobContext, params: CheckAlongParams) -> Any:
    from tuningfork.ops.checks import prepare_check_along_runs

    return prepare_check_along_runs(
        context.store, task=params.task, run=params.run, limit=params.limit, again=params.again,
        timeout=params.timeout,
    )


def execute_check_along_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.checks import execute_check_along_runs

    return execute_check_along_runs(context.store, prepared, progress=progress, cancel=cancel)


# --- tools at a held call: evidence, and hypotheses ------------------------------------------


def _hold_id(value: str) -> str:
    import re

    if not re.fullmatch(r"[0-9a-f]{1,32}", value or ""):
        raise OperationError("hold is a held call's id", code="bad_params", field="hold")
    return value


@dataclass
class HoldSampleParams:
    hold: str = field(metadata=_help("The held call, by its id."))
    n: int = field(default=10, metadata=_help("Samples to draw.", minimum=1, maximum=200))
    concurrency: int = field(default=1, metadata=_help("Samples at once; no more than the proxy's --max-inflight.", minimum=1, maximum=16))
    max_tokens: int = field(default=4096, metadata=_help("Cap per sample; hitting it counts as truncated.", minimum=1, maximum=262144))
    seed_base: int | None = field(default=None, metadata=_help(SEED_HELP, minimum=0))
    prompt_cache: bool = field(default=False, metadata=_help(CACHE_HELP))
    normalizers: list[str] = field(default_factory=list, metadata=_help("Restrict normalizers; none named means the defaults."))


def prepare_hold_sample_job(context: JobContext, params: HoldSampleParams) -> Any:
    from tuningfork.ops.holds import prepare_hold_sample

    return prepare_hold_sample(
        context.store, context.proxy, _hold_id(params.hold), n=params.n,
        concurrency=params.concurrency, max_tokens=params.max_tokens,
        seed_base=params.seed_base, prompt_cache=params.prompt_cache, normalizers=params.normalizers,
    )


def execute_hold_sample_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.holds import execute_hold_sample

    return execute_hold_sample(context.store, prepared, progress=progress, cancel=cancel)


@dataclass
class HoldConsoleParams:
    hold: str = field(metadata=_help("The held call, by its id."))
    question: str = field(metadata=_help("One question for the model, at the held call."))


def prepare_console_job(context: JobContext, params: HoldConsoleParams) -> Any:
    from tuningfork.ops.holds import prepare_console

    return prepare_console(context.proxy, _hold_id(params.hold), params.question)


def execute_console_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.holds import execute_console

    return execute_console(context.store, prepared, progress=progress, cancel=cancel)


# --- refining a skill: proposals from the built-in proposer, and screens of them ---------------


@dataclass
class ProposeParams:
    refinement: str = field(metadata=_help("The refinement, by its id or a prefix of it."))
    m: int = field(default=3, metadata=_help("Proposals to ask for: different ideas, each a few small edits.", minimum=1, maximum=6))
    include_task: bool = field(default=True, metadata=_help("Show the proposer the task's own criteria — its expectations and check — beside the step's evidence."))


def prepare_propose_job(context: JobContext, params: ProposeParams) -> Any:
    from tuningfork.ops.proposer import prepare_propose

    return prepare_propose(
        context.store, params.refinement, m=params.m, proxy_url=context.model_url, include_task=params.include_task
    )


def execute_propose_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.proposer import execute_propose

    return execute_propose(context.store, prepared, progress=progress, cancel=cancel)


@dataclass
class ScreenParams:
    refinement: str = field(metadata=_help("The refinement, by its id or a prefix of it."))
    proposals: list[str] = field(default_factory=list, metadata=_help("The proposals to screen, by id or prefix: the family the correction counts, fixed before sampling."))
    n: int = field(default=20, metadata=_help("Samples per arm, the original's included: 20 by default; fewer than 10 can show only a large shift.", minimum=2, maximum=200))
    alpha: float = field(default=0.05, metadata=_help("Family-wise significance level.", minimum=0.0001, maximum=0.5))
    concurrency: int = field(default=1, metadata=_help("Samples in flight at once; the proxy's --max-inflight still caps them.", minimum=1, maximum=64))
    max_tokens: int = field(default=4096, metadata=_help("Cap per sample; hitting it counts as truncated.", minimum=1, maximum=262144))
    seed_base: int | None = field(default=None, metadata=_help(SEED_HELP, minimum=0))
    prompt_cache: bool = field(default=False, metadata=_help(CACHE_HELP))


def prepare_screen_job(context: JobContext, params: ScreenParams) -> Any:
    from tuningfork.ops.refine import prepare_refinement_screen

    return prepare_refinement_screen(
        context.store, params.refinement, params.proposals, base_url=context.model_url,
        alpha=params.alpha, n=params.n, concurrency=params.concurrency,
        max_tokens=params.max_tokens, seed_base=params.seed_base, prompt_cache=params.prompt_cache,
    )


def execute_screen_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.screen import execute_screen

    return execute_screen(context.store, prepared, progress=progress, cancel=cancel)


@dataclass
class ScreenToEndParams:
    refinement: str = field(metadata=_help("The refinement, by its id or a prefix of it."))
    proposals: list[str] = field(default_factory=list, metadata=_help("The proposals to screen, by id or prefix: the family the correction counts, fixed before any fork runs."))
    n: int = field(default=20, metadata=_help("Forks per arm, each run from the checkpoint to the end, the original's included: 20 by default; fewer than 10 can show only a large shift.", minimum=2, maximum=100))
    alpha: float = field(default=0.05, metadata=_help("Family-wise significance level.", minimum=0.0001, maximum=0.5))
    concurrency: int = field(default=1, metadata=_help("Forks at once; no more than the proxy's --max-inflight. On one GPU, one at a time is fastest.", minimum=1, maximum=16))
    checkpoints: int = field(default=4, metadata=_help(
        "Runs the forks are drawn from: the refinement's own, and others of its task at the same step, passing and "
        "failing by turns, each arm's n spread evenly over them (4 by default; fewer where fewer serve). A fork "
        "measures the edit given the prefix it replays, and one prefix is one draw of the path: an effect that holds "
        "across checkpoints is one fresh runs keep. 1 forks the refinement's run alone.", minimum=1, maximum=10))
    quick: bool = field(default=False, metadata=_help(
        "A quick look: 8 forks of each proposal alone, beside the original's forks drawn before; rates, no verdict."))
    fresh_original: bool = field(default=False, metadata=_help(
        "Draw the original's forks anew, rather than count those this refinement drew before."))


def prepare_screen_to_end_job(context: JobContext, params: ScreenToEndParams) -> Any:
    from tuningfork.ops.refine import prepare_refinement_screen

    return prepare_refinement_screen(
        context.store, params.refinement, params.proposals, to_end=True, alpha=params.alpha,
        n=params.n, concurrency=params.concurrency, proxy=context.proxy.status(), checkpoints=params.checkpoints,
        quick=params.quick, fresh_original=params.fresh_original,
    )


def execute_screen_to_end_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.screen import execute_screen_to_end

    return execute_screen_to_end(context.store, prepared, progress=progress, cancel=cancel)


@dataclass
class ValidateParams:
    refinement: str = field(metadata=_help("The refinement, by its id or a prefix of it."))
    proposal: str = field(metadata=_help("The proposal to validate, by id or prefix: one that survived a screen."))
    n: int = field(default=5, metadata=_help("Forks per arm from the checkpoint to the end, the original's included.", minimum=5, maximum=200))
    runs: int = field(default=5, metadata=_help("Full runs per arm, of the training task and of each held-out task.", minimum=5, maximum=200))
    alpha: float = field(default=0.05, metadata=_help("Significance level for the not-worse checks.", minimum=0.0001, maximum=0.5))
    concurrency: int = field(default=1, metadata=_help("Runs at once; no more than the proxy's --max-inflight. On one GPU, one at a time is fastest.", minimum=1, maximum=16))


def prepare_validate_job(context: JobContext, params: ValidateParams) -> Any:
    from tuningfork.ops.validate import prepare_validate

    return prepare_validate(
        context.store, params.refinement, params.proposal, n=params.n, runs=params.runs,
        alpha=params.alpha, concurrency=params.concurrency, proxy=context.proxy.status(),
    )


def execute_validate_job(context: JobContext, prepared: Any, *, progress: Any, cancel: Any) -> Any:
    from tuningfork.ops.validate import execute_validate

    return execute_validate(context.store, prepared, progress=progress, cancel=cancel)


# --- the catalog ---------------------------------------------------------------------------

CATALOG: dict[str, Kind] = {
    kind.name: kind
    for kind in (
        Kind(
            name="resample",
            lane=BACKGROUND,
            summary="Re-run one step N times with its history held fixed, and cluster what the model did.",
            params=ResampleParams,
            prepare=prepare_resample_job,
            execute=execute_resample_job,
            total=lambda prepared: prepared.plan.n,
        ),
        Kind(
            name="tokens",
            lane=BACKGROUND,
            summary="Score the answer a step gave, token by token: how sure the model was of each, for its heatmap.",
            params=TokensParams,
            prepare=prepare_tokens_job,
            execute=execute_tokens_job,
        ),
        Kind(
            name="forking-tokens",
            lane=BACKGROUND,
            summary="Force each likely alternative at a step's uncertain tokens, let the model go on, and test which change what it does.",
            params=ForkingParams,
            prepare=prepare_forking_job,
            execute=execute_forking_job,
            total=lambda prepared: prepared.planned,
        ),
        Kind(
            name="ab",
            lane=BACKGROUND,
            summary="Swap an edited skill in at a step, sample both wordings, and test whether behaviour shifts.",
            params=ABParams,
            prepare=prepare_ab_job,
            execute=execute_ab_job,
            total=lambda prepared: prepared.plan.n * 2,
        ),
        Kind(
            name="propose",
            lane=BACKGROUND,
            summary="Ask a model for small edits to a skill from a refinement's evidence: hypotheses until a screen tests them.",
            params=ProposeParams,
            prepare=prepare_propose_job,
            execute=execute_propose_job,
            # No total: one question to a model, no sample, fork or run to count — a total of one
            # was never counted, and a finished proposer read "0 of 1" (found in review).
        ),
        Kind(
            name="screen",
            lane=BACKGROUND,
            summary="Sample several proposed wordings at a refinement's checkpoint, each against the original, Holm-corrected.",
            params=ScreenParams,
            prepare=prepare_screen_job,
            execute=execute_screen_job,
            total=lambda prepared: prepared.total,
        ),
        Kind(
            name="screen-to-end",
            lane=BACKGROUND,
            summary="Screen proposals by the task's check: every arm's forks run from the checkpoint to the end, each against the original, Holm-corrected.",
            params=ScreenToEndParams,
            prepare=prepare_screen_to_end_job,
            execute=execute_screen_to_end_job,
            total=lambda prepared: prepared.total,
        ),
        Kind(
            name="validate",
            lane=BACKGROUND,
            summary="Validate a screened proposal with full runs: forks from its checkpoint to the end, the training task, and each held-out task — verdicts only.",
            params=ValidateParams,
            prepare=prepare_validate_job,
            execute=execute_validate_job,
            total=lambda prepared: prepared.total,
        ),
        Kind(
            name="resample-to-end",
            lane=BACKGROUND,
            summary="Fork the run from a step N times, each run to the end, and count the task's check passing.",
            params=ToEndParams,
            prepare=prepare_to_end_job,
            execute=execute_to_end_job,
            total=lambda prepared: prepared.forks,
        ),
        Kind(
            name="ab-to-end",
            lane=BACKGROUND,
            summary="Fork the run from a step under each wording, run every fork to the end, and compare the check's pass rates.",
            params=ABToEndParams,
            prepare=prepare_ab_to_end_job,
            execute=execute_to_end_job,
            total=lambda prepared: prepared.forks,
        ),
        Kind(
            name="fork",
            lane=BACKGROUND,
            summary="Replay a run up to a step and go live from there — stopping at the step for you, by default.",
            params=ForkParams,
            prepare=prepare_fork_job,
            execute=execute_fork_job,
            total=lambda prepared: prepared.n,
        ),
        Kind(
            name="batch",
            lane=BACKGROUND,
            summary="Run a task N times through its harness, each run in a fresh workspace and home.",
            params=BatchParams,
            prepare=prepare_batch_job,
            execute=execute_batch_job,
            total=lambda prepared: prepared.n,
        ),
        Kind(
            name="chain",
            lane=BACKGROUND,
            summary="Run a task N times in order, each run starting from the skill the one before left: what the harness's own self-improvement makes of it.",
            params=ChainParams,
            prepare=prepare_chain_job,
            execute=execute_chain_job,
            total=lambda prepared: prepared.n,
        ),
        Kind(
            name="server-check",
            lane=INTERACTIVE,
            summary="Measure what the harnesses need of the server — a streaming tool call, /v1/messages — and what token heatmaps and forking tokens need.",
            params=ServerCheckParams,
            prepare=prepare_server_check_job,
            execute=execute_server_check_job,
            total=lambda prepared: len(prepared.checks),
        ),
        Kind(
            name="check-trial",
            lane=INTERACTIVE,
            summary="Try a task's check before trusting it: judge a recorded run again, or a reply and files you write. Nothing is recorded.",
            params=CheckTrialParams,
            prepare=prepare_check_trial_job,
            execute=execute_check_trial_job,
        ),
        Kind(
            name="check-along",
            lane=BACKGROUND,
            summary="Run a task's check along its runs — at the start, after each call that changed the files, at the end — so the tally says which call made each finding. Kept with each run; no verdict changes.",
            params=CheckAlongParams,
            prepare=prepare_check_along_job,
            execute=execute_check_along_job,
            total=lambda prepared: len(prepared.plans),
        ),
        Kind(
            name="hold-sample",
            lane=INTERACTIVE,
            summary="Sample a call held at a breakpoint N times: what the model does there, as evidence.",
            params=HoldSampleParams,
            prepare=prepare_hold_sample_job,
            execute=execute_hold_sample_job,
            total=lambda prepared: prepared.plan.n,
        ),
        Kind(
            name="hold-console",
            lane=INTERACTIVE,
            summary="Ask the model one question at a held call. Its answer is a hypothesis, never evidence.",
            params=HoldConsoleParams,
            prepare=prepare_console_job,
            execute=execute_console_job,
        ),
    )
}


def kinds_json(catalog: dict[str, Kind] | None = None) -> list[dict[str, Any]]:
    return [kind.to_json() for kind in (catalog or CATALOG).values()]


def params_json(params: Any) -> dict[str, Any]:
    """Parameters as stored with a job: what was asked, as JSON."""
    return dataclasses.asdict(params)
