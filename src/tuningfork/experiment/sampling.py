"""Resampling from a recorded checkpoint: the first way to "continue from here".

Take step k's exact request, send it N times, and see what the model does next. No tools
run: the recorded history already contains every tool result up to step k. That makes one
sample cost a single model call rather than a whole agent run, which is the point — samples
go where the variance is.

Three things are easy to get wrong here, and each is handled explicitly:

- **Sampling conditions.** Hermes sends no sampling parameters, so the original run was
  sampled under the *server's* defaults. A resample is only a resample of that run if the
  server still has the same defaults; the recording holds them, so they are compared.
- **What actually changed.** A resample cannot send the recorded request byte-for-byte: it
  drops streaming, caps `max_tokens` (the harness asks for the whole context window), and
  varies the seed. Every such change is recorded with the experiment.
- **Where the traffic goes.** Requests carry an experiment header, so a tuningfork proxy on
  the path forwards them without recording them as run steps — a resampled request has the
  same message path as its step and would otherwise look like the harness repeating it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Sequence

import httpx

from tuningfork.analysis.distribution import (
    Target,
    distribution,
    target_by_key,
    target_by_pattern,
    target_from_spec,
)
from tuningfork.analysis.normalize import (
    DEFAULT_NORMALIZERS,
    NORMALIZER_VERSION,
    Outcome,
    outcome_for,
    validate_normalizers,
)
from tuningfork.analysis.tokens import split_logprobs
from tuningfork.backends.probe import server_root
from tuningfork.config import EXPERIMENT_HEADER, redact_json
from tuningfork.ops.common import Event, Progress, emit
from tuningfork.experiment.skill import Span, substitute, text_sha256, unapplied
from tuningfork.proxy.recorder import fetch_server_defaults
from tuningfork.store.db import Experiment, Run, Step, Store
from tuningfork.store.tree import describe_message
from tuningfork.wire import WireFormat, get_wire

#: Server defaults compared between the recording and now. `seed` is random by design and
#: `n_predict` is overridden by every request, so neither says anything about parity.
PARITY_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "n_ctx",
    "samplers",
)

#: Request fields that set sampling behaviour directly, overriding any server default.
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "presence_penalty",
    "frequency_penalty",
    # llama.cpp's name, and the one OpenAI-compatible servers such as vLLM use. Missing
    # one meant a request that set it was checked against the server's default anyway.
    "repeat_penalty",
    "repetition_penalty",
)


class CheckpointError(LookupError):
    """The checkpoint, or something an experiment refers to, does not exist."""


@dataclass(slots=True)
class SamplingPlan:
    n: int = 20
    #: Concurrent requests. 1 by default: the server may be shared.
    concurrency: int = 1
    #: Sample i is sent with seed base+i. None draws a new random base for each arm, recorded
    #: with it, so a second experiment draws new samples: the same seeds draw the same ones
    #: again wherever the server reproduces them. An earlier base draws
    #: those again on purpose.
    seed_base: int | None = None
    #: Let the server reuse its cache of the prompt from one sample to the next. Off, each
    #: sample's prompt is processed whole (llama.cpp's `cache_prompt: false`), so every
    #: sample is drawn from the same state, whatever the server ran before: on llama.cpp a
    #: reused cache made one request's behaviour depend on the server's history.
    prompt_cache: bool = False
    #: The harness typically asks for the whole context window. A cap bounds the cost of a
    #: runaway sample; anything that hits it is reported as truncated, never clustered.
    max_tokens: int = 4096
    overrides: dict[str, Any] = field(default_factory=dict)
    normalizers: list[str] = field(default_factory=list)
    timeout: float = 600.0
    #: Ask for each token's top alternatives, this many, and keep them with each sample
    #: (`tuningfork.analysis.tokens`). Chat completions only.
    logprobs: int | None = None
    #: The model's output format (`tuningfork.backends.formats`), to mark what each token is.
    format: str | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.logprobs is not None and not 1 <= self.logprobs <= 20:
            raise ValueError("logprobs is the number of alternatives to keep, from 1 to 20")
        validate_normalizers(self.normalizers)

    def as_settings(self) -> dict[str, Any]:
        # The version travels with the settings: a cluster key stored with the experiment
        # (a declared target) is only meaningful under the normalizers that produced it.
        return {**asdict(self), "normalizer_version": NORMALIZER_VERSION}

    def seeded(self) -> SamplingPlan:
        """This plan with a seed base: its own, or a new random one that leaves room for
        every sample's seed below 2**31."""
        if self.seed_base is not None:
            return self
        import secrets

        return replace(self, seed_base=secrets.randbelow(2**31 - 1 - self.n))


@dataclass(slots=True)
class Checkpoint:
    run: Run
    step: Step
    #: Exactly as recorded, in the step's own wire format: what a resample resends.
    request: dict[str, Any]

    @property
    def wire(self) -> WireFormat:
        return get_wire(self.step.wire)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The conversation as OpenAI-style messages, for describing it."""
        return self.wire.view(self.request)

    @property
    def generated_history(self) -> int:
        """How many messages before this step were written by the model."""
        return sum(1 for message in self.messages if message.get("role") == "assistant")

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "run_label": self.run.label,
            "step": self.step.idx,
            "wire": self.wire.name,
            "messages": len(self.messages),
            "describe": self.describe(),
        }

    def describe(self) -> str:
        """What the model had just seen when it produced this step."""
        if not self.messages:
            return "empty context"
        # A harness may follow every tool result with bookkeeping of its own — Claude Code
        # appends a token budget as a system message. The model is reacting to what came
        # before that.
        position = len(self.messages) - 1
        while position > 0 and self.messages[position].get("role") == "system":
            position -= 1
        last = self.messages[position]
        if last.get("role") == "tool":
            call_id = last.get("tool_call_id")
            for message in reversed(self.messages):
                for call in message.get("tool_calls") or []:
                    if call.get("id") == call_id:
                        return f"after the result of {call['function']['name']}"
            return "after a tool result"
        return f"after {describe_message(last, 70)}"


def load_checkpoint(store: Store, run_reference: str, index: int) -> Checkpoint:
    run = store.get_run(run_reference)
    if run is None:
        raise CheckpointError(f"no run matching {run_reference!r}")
    step = store.step(run.id, index)
    if step is None:
        raise CheckpointError(
            f"run {run.id[:8]} has no step {index}; it has steps 0-{run.step_count - 1}"
        )
    return Checkpoint(run=run, step=step, request=store.rebuild_request(step))


def history_caveat(checkpoint: Checkpoint) -> str:
    """The caveat every A/B at a checkpoint carries — both arms inherit a history the
    original wording wrote — sharpened by what this checkpoint actually holds."""
    if checkpoint.generated_history == 0:
        return (
            "No model-written history precedes this step, so both arms saw identical "
            "context apart from the skill text itself."
        )
    return (
        f"The {checkpoint.generated_history} model-written message(s) before this step were "
        "generated under the original wording, and both arms inherit them. A checkpoint "
        "A/B is a fast screen; confirm a winner with full runs."
    )


def prepare_request(request: dict[str, Any], plan: SamplingPlan) -> tuple[dict, dict]:
    """The request a resample sends, and every way it differs from the recorded one."""
    sent = copy.deepcopy(request)
    changes: dict[str, Any] = {}

    def change(key: str, recorded: Any, now: Any) -> None:
        changes[key] = {"recorded": recorded, "sent": now}

    for key in ("stream", "stream_options", "n"):
        if key in sent:
            change(key, sent.pop(key), None)

    if "max_completion_tokens" in sent:
        # The newer OpenAI name; local servers generally only honour `max_tokens`.
        recorded = sent.pop("max_completion_tokens")
        change("max_completion_tokens", recorded, None)
        sent.setdefault("max_tokens", recorded)
    recorded_max = sent.get("max_tokens")
    if recorded_max is None or recorded_max > plan.max_tokens:
        sent["max_tokens"] = plan.max_tokens
        change("max_tokens", recorded_max, plan.max_tokens)

    if "seed" in sent:
        change("seed", sent.pop("seed"), "per sample")
    for key, value in plan.overrides.items():
        change(key, sent.get(key), value)
        sent[key] = value
    if plan.logprobs is not None:
        # The distribution is read before sampling: asking for it changes no token drawn.
        for key, value in (("logprobs", True), ("top_logprobs", plan.logprobs)):
            if sent.get(key) != value:
                change(key, sent.get(key), value)
            sent[key] = value
    return sent, changes


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-9)
    return first == second


async def _served_models(client: httpx.AsyncClient, base_url: str) -> list[str]:
    try:
        response = await client.get(f"{server_root(base_url)}/v1/models", timeout=15)
        if response.status_code != 200:
            return []
        entries = response.json().get("data") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return []
    return [entry["id"] for entry in entries if isinstance(entry, dict) and "id" in entry]


@dataclass(slots=True)
class Anchor:
    """What an experiment samples: a recorded step, or a call held at a breakpoint that
    has not been recorded yet — no run, no step, but a request and a server all the same."""

    run_id: str | None
    step_idx: int | None
    step_id: str | None
    request: dict[str, Any]
    wire: WireFormat
    #: The server's generation defaults the call was made under.
    server_defaults: dict[str, Any]
    model: str | None
    server_id: str | None
    hold_id: str | None = None

    @classmethod
    def of(cls, checkpoint: Checkpoint) -> Anchor:
        return cls(
            run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, step_id=checkpoint.step.id,
            request=checkpoint.request, wire=checkpoint.wire,
            server_defaults=checkpoint.step.server_defaults or {}, model=checkpoint.step.model,
            server_id=checkpoint.step.server_id,
        )


async def check_parity(
    client: httpx.AsyncClient, base_url: str, anchor: Checkpoint | Anchor, plan: SamplingPlan
) -> tuple[dict[str, Any], list[str]]:
    """Current server defaults, plus warnings wherever they no longer match the recording."""
    if not isinstance(anchor, Anchor):
        anchor = Anchor.of(anchor)
    notes: list[str] = []
    current = await fetch_server_defaults(client, base_url)
    recorded = anchor.server_defaults or {}
    # A field the request sets itself does not depend on the server's default.
    fixed = set(plan.overrides) | {key for key in SAMPLING_FIELDS if key in anchor.request}

    if not recorded:
        notes.append(
            "the original run's server defaults were not recorded, so this cannot confirm "
            "it samples under the same settings"
        )
    elif not current:
        notes.append("the server does not expose its defaults, so parity cannot be checked")
    else:
        for key in PARITY_KEYS:
            if key in fixed or key not in recorded or key not in current:
                continue
            if not _same(recorded[key], current[key]):
                notes.append(
                    f"server default {key} is {current[key]!r} now, but was "
                    f"{recorded[key]!r} when the run was recorded"
                )

    models = await _served_models(client, base_url)
    model = anchor.model
    if models and model and model not in models:
        notes.append(
            f"the step was recorded against model {model!r}, but the server serves "
            f"{', '.join(models)}"
        )
    if anchor.server_id:
        from tuningfork.proxy.recorder import fetch_server_identity
        from tuningfork.store.db import identity_id

        now = identity_id(await fetch_server_identity(client, base_url))
        if now != anchor.server_id:
            notes.append(
                f"the server is not the one the step was recorded on (server "
                f"{anchor.server_id[:8]} then, {now[:8]} now): a different build, "
                "model file, context size, chat template or defaults — `tuningfork servers` "
                "lists both"
            )
    return current, notes


@dataclass(slots=True)
class SampleEvent:
    """Reported as each sample lands, for progress output."""

    arm: str
    index: int
    seed: int
    outcome: Outcome
    seconds: float
    completion_tokens: int | None
    experiment_id: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            "index": self.index,
            "seed": self.seed,
            "outcome": self.outcome.to_json(),
            "seconds": round(self.seconds, 3),
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


#: Set on the proxy's 503 while it is paused: a person is looking at something, and the
#: call should wait rather than count as a failure.
PAUSED_HEADER = "x-tuningfork-paused"


async def send_experiment_call(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    *,
    tag: str,
    timeout: float,
    progress: Progress | None = None,
) -> tuple[httpx.Response, float]:
    """POST an experiment call and return the response with how long the answering attempt
    took. Through a paused proxy it waits — reported once as `waiting` — and tries again,
    rather than storing a failed sample for every call made during the pause."""
    delay, told = 1.0, False
    while True:
        started = time.monotonic()
        response = await client.post(
            url, json=body, headers={EXPERIMENT_HEADER: tag}, timeout=timeout
        )
        if response.status_code != 503 or not response.headers.get(PAUSED_HEADER):
            return response, time.monotonic() - started
        if not told:
            emit(progress, "waiting", reason="paused")
            told = True
        await asyncio.sleep(delay)
        delay = min(delay * 2, 10.0)


def _summarize_response(response: dict[str, Any] | None) -> tuple[str | None, dict, int]:
    if not response:
        return None, {}, 0
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    reasoning = sum(
        len(message[key]) for key in ("reasoning_content", "reasoning")
        if isinstance(message.get(key), str)
    )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return choices[0].get("finish_reason"), usage, reasoning


def store_sample_tokens(store: Store, sample: Any, entries: list[dict[str, Any]], plan: SamplingPlan) -> None:
    """A sample's tokens, as the server reported them with it: parts marked by the model's
    output format when it is known, redacted token-aware, stored beside the sample."""
    from tuningfork.analysis.tokens import from_entries, redact_stream
    from tuningfork.backends.formats import get_format

    stream = from_entries(entries, top_k=plan.logprobs).mark_parts(get_format(plan.format))
    store.add_token_stream(
        redact_stream(stream).to_json(), source="sampled", sample_id=sample.id, method="chat-logprobs",
    )


async def draw_samples(
    store: Store,
    experiment: Experiment,
    client: httpx.AsyncClient,
    base_url: str,
    request: dict[str, Any],
    plan: SamplingPlan,
    progress: Progress | None = None,
    wire: WireFormat | None = None,
) -> list[Outcome]:
    wire = wire or get_wire(None)
    # `run_arm` seeds the plan before the experiment records it; this only keeps a direct
    # caller from failing.
    plan = plan.seeded()
    # The recorded request is resent in its own format, to the endpoint for that format.
    url = f"{server_root(base_url)}{wire.endpoint}"
    semaphore = asyncio.Semaphore(plan.concurrency)
    outcomes: list[Outcome] = [Outcome("error", "error", "not drawn")] * plan.n

    async def one(index: int) -> None:
        # Stored and reported before the next sample may start, so at concurrency 1 the
        # samples land, and are told, in the order they were drawn.
        async with semaphore:
            await draw(index)

    async def draw(index: int) -> None:
        seed = plan.seed_base + index
        response_json: dict[str, Any] | None = None
        error: str | None = None
        started = time.monotonic()
        try:
            response, seconds = await send_experiment_call(
                client, url, {**request, "seed": seed},
                tag=experiment.id, timeout=plan.timeout, progress=progress,
            )
            started = time.monotonic() - seconds
            if response.status_code >= 400:
                error = f"HTTP {response.status_code}: {response.text[:300]}"
            else:
                parsed = response.json()
                # A 200 can still be no answer: a body that is not an object at all, or
                # an error body. Either is a transport failure for this sample — never
                # a reason to abandon the samples still being drawn.
                if not isinstance(parsed, dict):
                    error = f"the server answered 200 with a JSON {type(parsed).__name__}"
                elif failure := wire.failure(response.status_code, parsed):
                    detail = json.dumps(parsed.get("error"))[:200] if parsed.get("error") else ""
                    error = f"{failure}{': ' + detail if detail else ''}"
                else:
                    response_json = parsed
        except (httpx.HTTPError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - started

        # Token data comes out before the response is redacted, and is redacted token-aware.
        response_json, entries = split_logprobs(response_json)
        # Outcomes are computed from what is stored, so a later report recomputing them
        # from the database can never disagree with what was printed now.
        stored = (
            redact_json(wire.canonical_response(response_json))
            if response_json is not None
            else None
        )
        outcome = (
            Outcome("error", "error", error)
            if error is not None
            else outcome_for(stored, plan.normalizers)
        )
        finish_reason, usage, reasoning = _summarize_response(stored)
        sample = await asyncio.to_thread(
            store.add_sample,
            experiment.id,
            index,
            seed=seed,
            response=stored,
            outcome_kind=outcome.kind,
            outcome_key=outcome.key,
            outcome_label=outcome.label,
            finish_reason=finish_reason,
            usage=usage,
            timings={"total_ms": round(seconds * 1000, 2)},
            reasoning_chars=reasoning,
            error=error,
        )
        if entries:
            await asyncio.to_thread(store_sample_tokens, store, sample, entries, plan)
        outcomes[index] = outcome
        if progress is not None:
            event = SampleEvent(
                arm=experiment.arm,
                index=index,
                seed=seed,
                outcome=outcome,
                seconds=seconds,
                completion_tokens=usage.get("completion_tokens"),
                experiment_id=experiment.id,
                error=error,
            )
            emit(progress, "sample", subject=event, **event.to_json())

    await asyncio.gather(*(one(index) for index in range(plan.n)))
    return outcomes


async def run_arm(
    store: Store,
    checkpoint: Checkpoint | Anchor,
    plan: SamplingPlan,
    *,
    base_url: str,
    request: dict[str, Any],
    kind: str,
    arm: str,
    client: httpx.AsyncClient,
    group_id: str | None = None,
    variant: dict[str, Any] | None = None,
    parity: tuple[dict[str, Any], list[str]] | None = None,
    annotations: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> Experiment:
    """Create an experiment for one arm and draw its samples. It stays `running`: the caller
    says how it ended, once every arm of it has (`finished`).

    Reports `experiment` when it is created — so work in progress can be linked to it —
    then `sample` as each sample lands.
    """
    from tuningfork.endpoints import check_endpoint

    check_endpoint(base_url, home=store.home, purpose="sampling")
    anchor = checkpoint if isinstance(checkpoint, Anchor) else Anchor.of(checkpoint)
    # Each arm its own random seeds, unless the caller named them: arms drawn alike would
    # not be independent samples.
    plan = plan.seeded()
    sent, changes = prepare_request(request, plan)
    if plan.logprobs is not None and plan.format is None:
        from tuningfork.backends.llamacpp import server_format

        found = await server_format(client, base_url)
        plan.format = found.name if found is not None else None
    server_defaults, notes = parity or await check_parity(client, base_url, anchor, plan)
    if not plan.prompt_cache:
        from tuningfork.backends.llamacpp import CACHE_PROMPT_ENDPOINTS, is_llamacpp, reuses_prompts

        endpoint = anchor.wire.endpoint
        if not await is_llamacpp(client, base_url):
            notes = [*notes, "the server is not llama.cpp's, so it was not asked to process each "
                     "sample's prompt whole: samples after the first may come from its cache of it"]
        elif endpoint not in CACHE_PROMPT_ENDPOINTS:
            # The request cannot ask for it; the server may do it anyway, started with
            # --no-cache-prompt. Measured, not assumed: then every sample is processed whole.
            reuses = await reuses_prompts(client, base_url, endpoint, model=str(request.get("model") or "tuningfork"))
            if reuses:
                notes = [*notes, f"llama.cpp's {endpoint} drops cache_prompt, and this server reuses the prompts "
                         "it processes: samples after the first come from its cache of this one, and depend on "
                         "what it ran before (start llama-server with --no-cache-prompt)"]
            elif reuses is None:
                notes = [*notes, f"llama.cpp's {endpoint} drops cache_prompt, so each sample's prompt "
                         "could not be processed whole: samples after the first come from its cache of it"]
        else:
            changes["cache_prompt"] = {"recorded": sent.get("cache_prompt"), "sent": False}
            sent["cache_prompt"] = False
    experiment = await asyncio.to_thread(
        store.create_experiment,
        kind=kind,
        arm=arm,
        run_id=anchor.run_id,
        step_idx=anchor.step_idx,
        step_id=anchor.step_id,
        hold_id=anchor.hold_id,
        request=redact_json(sent),
        settings={**plan.as_settings(), **(annotations or {})},
        changes=changes,
        variant=variant,
        base_url=base_url,
        server_defaults=server_defaults,
        notes=notes,
        group_id=group_id,
    )
    emit(
        progress, "experiment", subject=experiment,
        id=experiment.id, kind=kind, arm=arm, group_id=group_id, n=plan.n, reach="step",
    )
    await draw_samples(
        store, experiment, client, base_url, sent, plan, progress, wire=anchor.wire
    )
    return experiment


class finished:
    """Collects the experiments a piece of work creates, and says how they ended.

    As a context manager around the work: `done` if it returns, `cancelled` if it is
    cancelled — Ctrl-C included — and `failed` if it raises. The store is written
    synchronously, since a cancelled task may not get to await again.
    """

    def __init__(self, store: Store, progress: Progress | None) -> None:
        self.store = store
        self.forward = progress
        self.created: list[Experiment] = []

    def __call__(self, event: Event) -> None:
        if event.kind == "experiment":
            self.created.append(event.subject)
        if self.forward:
            self.forward(event)

    def __enter__(self) -> "finished":
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if kind is None:
            status = "done"
        elif issubclass(kind, (asyncio.CancelledError, KeyboardInterrupt)):
            status = "cancelled"
        else:
            status = "failed"
        self.store.finish_experiments([e.id for e in self.created], status)


async def run_resample(
    store: Store,
    checkpoint: Checkpoint,
    plan: SamplingPlan,
    *,
    base_url: str,
    client: httpx.AsyncClient | None = None,
    progress: Progress | None = None,
) -> Experiment:
    """Sample the next step N times from a checkpoint, with history held fixed."""
    owned = client is None
    http = client or httpx.AsyncClient(timeout=plan.timeout, event_hooks=guard_hooks(home=store.home))
    try:
        with finished(store, progress) as ended:
            return await run_arm(
                store,
                checkpoint,
                plan,
                base_url=base_url,
                request=checkpoint.request,
                kind="resample",
                arm="original",
                client=http,
                progress=ended,
            )
    finally:
        if owned:
            await http.aclose()


@dataclass(slots=True)
class ABRun:
    original: Experiment
    variant: Experiment
    spans: list[Span]


async def run_ab(
    store: Store,
    checkpoint: Checkpoint,
    plan: SamplingPlan,
    *,
    base_url: str,
    original_text: str,
    variant_text: str,
    original_path: str | None = None,
    variant_path: str | None = None,
    target: Target | None = None,
    alpha: float | None = None,
    client: httpx.AsyncClient | None = None,
    progress: Progress | None = None,
) -> ABRun:
    """Resample the checkpoint as recorded, then with the variant skill substituted.

    `alpha`, the significance level the A/B was declared with, is kept with both arms beside
    its target: a re-report judges at it unless asked otherwise.

    The arms run one after the other rather than interleaved. On a server with prefix
    caching, alternating arms would invalidate the cache at every switch and re-process the
    context from the skill onward each time; running them in blocks costs two cold starts.
    Sampling from a fixed model does not drift, so blocking buys speed at no cost in bias.
    """
    variant_request, spans = substitute(checkpoint.request, original_text, variant_text)
    variant_info = {
        "original_path": original_path,
        "variant_path": variant_path,
        "original_sha256": text_sha256(original_text),
        "variant_sha256": text_sha256(variant_text),
        "spans": [asdict(span) for span in spans],
        "unapplied": unapplied(original_text, variant_text, spans),
    }
    group_id = uuid.uuid4().hex
    owned = client is None
    http = client or httpx.AsyncClient(timeout=plan.timeout, event_hooks=guard_hooks(home=store.home))
    try:
        parity = await check_parity(http, base_url, checkpoint, plan)
        arms = {}
        # Both arms stay running until the whole A/B has: a finished original arm with no
        # variant arm yet is an A/B under way, not one that was cut off.
        ended = finished(store, progress)
        with ended:
            for arm, request in (("original", checkpoint.request), ("variant", variant_request)):
                arms[arm] = await run_arm(
                    store,
                    checkpoint,
                    plan,
                    base_url=base_url,
                    request=request,
                    kind="ab",
                    arm=arm,
                    client=http,
                    group_id=group_id,
                    variant=variant_info,
                    parity=parity,
                    # Stored with both arms, so the result can always be re-reported
                    # against the test declared before sampling — not one chosen afterwards:
                    # its target, and the level it is judged at (found in review: an A/B
                    # declared at α 0.01 was judged at 0.05 on its own page).
                    annotations={**({"target": target.spec} if target else {}),
                                 **({"alpha": alpha} if alpha is not None else {})} or None,
                    progress=ended,
                )
        return ABRun(original=arms["original"], variant=arms["variant"], spans=spans)
    finally:
        if owned:
            await http.aclose()


def guard_hooks(**kwargs: Any) -> dict[str, Any]:
    from tuningfork.endpoints import guard_hooks as hooks

    return hooks(**kwargs)


def experiment_outcomes(
    store: Store, experiment: Experiment, normalizers: list[str] | None = None
) -> list[Outcome]:
    """Recompute outcomes from stored responses, by default with the experiment's normalizers."""
    names = experiment.normalizers if normalizers is None else normalizers
    outcomes = []
    for sample in store.samples(experiment.id):
        if sample.error:
            outcomes.append(Outcome("error", "error", sample.error))
        else:
            outcomes.append(outcome_for(store.sample_response(sample), names))
    return outcomes


def experiment_reasoning(store: Store, experiment: Experiment) -> list[int]:
    """Each sample's reasoning in characters, aligned with `experiment_outcomes`."""
    return [sample.reasoning_chars or 0 for sample in store.samples(experiment.id)]


def experiment_distribution(
    store: Store, experiment: Experiment, normalizers: list[str] | None = None
) -> Any:
    """The behaviours an experiment saw, each with how many of its samples reasoned."""
    from tuningfork.analysis.distribution import distribution

    return distribution(experiment_outcomes(store, experiment, normalizers), experiment_reasoning(store, experiment))


def declared_target(
    experiment: Experiment, lens: list[str] | None = None
) -> tuple[Target | None, str | None]:
    """The target an A/B declared before sampling, if it can still be tested as declared.

    Returns the target, or None and the reason it cannot be. A cluster key names a cluster
    only under the normalizers, and the normalizer version, that produced it. A pattern is
    matched against behaviour labels, which a different lens rewrites: under `tool_name`,
    `/row_count/` would match nothing in either arm and read as "no difference".
    """
    spec = experiment.settings.get("target")
    if not spec:
        return None, None
    if lens is not None and _effective(lens) != _effective(experiment.normalizers):
        return None, (
            "the declared target was defined under the original normalizers, so it cannot be "
            "re-tested under a different lens; showing the exploratory view instead"
        )
    version = experiment.settings.get("normalizer_version", 1)
    if spec.get("kind") == "key" and version != NORMALIZER_VERSION:
        return None, (
            "the declared target names a cluster as tuningfork's normalizers keyed it when the "
            "A/B ran; they have changed since, so it is shown as an exploratory comparison"
        )
    return target_from_spec(spec), None


def _effective(normalizers: list[str]) -> frozenset[str]:
    """The normalizers actually applied: naming none applies every default one."""
    return frozenset(normalizers or DEFAULT_NORMALIZERS)


def step_resample(store: Store, checkpoint: Checkpoint) -> Any:
    """The most recent resample of the step itself, whose lettered behaviours a target can
    name — not one run to the end, whose behaviours are its forks' at the step."""
    for experiment in store.experiments(
        run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, kind="resample", limit=20
    ):
        origin = experiment.settings.get("origin") or {}
        if experiment.settings.get("reach") == "end" or origin.get("matches_step") is False:
            continue  # its forks' behaviour, or a request the model never saw at this step
        return experiment
    return None


def resolve_target(
    store: Store,
    checkpoint: Checkpoint,
    *,
    letter: str | None = None,
    pattern: str | None = None,
    rules: Sequence[dict[str, str]] | None = None,
) -> tuple[Target | None, list[str] | None]:
    """Turn a user's target declaration into a predicate, before any A/B sample is drawn.

    A letter names a cluster from the most recent *resample* at this checkpoint — an
    earlier, separate sample, which is what makes it a legitimate pre-declared target. The
    normalizers used there are returned too, since cluster keys are only comparable under
    the same normalizers.
    """
    given = [name for name, value in (("letter", letter), ("pattern", pattern), ("description", rules)) if value]
    if len(given) > 1:
        raise ValueError(f"declare the target by {' or by '.join(given)}, not both")
    if rules:
        from tuningfork.analysis.distribution import target_by_rules

        return target_by_rules(rules), None
    if pattern:
        return target_by_pattern(pattern), None
    if not letter:
        return None, None

    experiment = step_resample(store, checkpoint)
    if experiment is None:
        raise CheckpointError(
            f"--target {letter} names a cluster from an earlier `tuningfork resample` at this "
            "checkpoint, and there is none. Run a resample first, or declare the target "
            "with --target-match."
        )
    cluster = distribution(experiment_outcomes(store, experiment)).by_letter(letter)
    if cluster is None:
        raise CheckpointError(f"resample {experiment.id[:8]} has no cluster {letter.upper()}")
    description = f"{cluster.letter} from resample {experiment.id[:8]}: {cluster.outcome.label}"
    return (target_by_key(cluster.outcome.key, description, letter=cluster.letter, resample=experiment.id),
            experiment.normalizers)


def target_letters(store: Store, spec: dict[str, Any] | None, lens: list[str] | None = None) -> dict[str, str] | None:
    """The letters, by behaviour key, of the resample a letter target came from — for a report
    of that target to letter its behaviours as the resample did (`agreeing_letters`). None for
    any other target, and where they would not agree: under another lens, or with the resample
    gone. A target stored before targets kept their resample names it in its description."""
    import re

    from tuningfork.analysis.distribution import distribution

    if not spec or spec.get("kind") != "key":
        return None
    reference = spec.get("resample")
    if not reference:
        named = re.match(r"([A-Z]+) from resample ([0-9a-f]{8}): ", spec.get("description") or "")
        reference = named.group(2) if named else None
    resample = store.get_experiment(reference) if reference else None
    if resample is None or resample.kind != "resample":
        return None
    if lens is not None and _effective(lens) != _effective(resample.normalizers):
        return None
    found = distribution(experiment_outcomes(store, resample))
    return {cluster.outcome.key: cluster.letter for cluster in found.clusters}
