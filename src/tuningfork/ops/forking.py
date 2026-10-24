"""Forking tokens at a step: which forced tokens change what the model does.

`prepare_forking` plans a probe from a stored base stream — the step's own answer, scored, or
a sample's — with no network calls: the positions, the tokens forced at each, the target
declared before sampling, the size of the family of tests and what it can detect, and an
estimate. `execute_forking` draws it (`tuningfork.experiment.forking`), and `forking_result`
reports it again from the store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress
from tuningfork.ops.trash import refuse_trashed
from tuningfork.store.db import Store


def _base(store: Store, checkpoint: Any, stream: str | None) -> Any:
    own = store.token_streams(step_id=checkpoint.step.id)
    if stream is None:
        found = next((record for record in own if record.source in ("scored", "recorded")), None)
        if found is None:
            raise OperationError(
                f"step {checkpoint.step.idx}'s answer has no tokens to fork from yet",
                hint=f"score it first: `tuningfork tokens {checkpoint.run.id[:8]} {checkpoint.step.idx} --score` "
                "(or --score here)",
            )
        return found
    from tuningfork.ops.tokens import step_tokens

    view = step_tokens(store, checkpoint.run.id, str(checkpoint.step.idx), stream=stream).view
    return view.record


def _context(stream: Any, position: int, width: int = 40) -> str:
    before = stream.text(max(0, position - 12), position)
    return f"…{before[-width:]}⟦{stream.token_text(position)}⟧"


def _seconds_per_token(step: Any) -> float | None:
    timings, usage = step.timings or {}, step.usage or {}
    total, first, generated = timings.get("total_ms"), timings.get("ttft_ms"), usage.get("completion_tokens")
    if not total or not generated:
        return None
    decoding = total - (first or 0)
    return max(decoding, 1.0) / 1000 / generated


def _prefill_seconds(step: Any) -> float | None:
    """Seconds the step's server took to process its whole prompt, read from the time to its
    first token and scaled up for whatever the step read from a cache: every continuation
    processes the prompt again (`backends.llamacpp.raw`). Measured on Gemma 4 26B, a probe
    of 150 continuations of a 13,729-token prompt spent most of its six minutes there."""
    timings, usage = step.timings or {}, step.usage or {}
    first, prompt = timings.get("ttft_ms"), usage.get("prompt_tokens")
    if not first or not prompt:
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    fresh = prompt - cached
    return first / 1000 * prompt / fresh if fresh > 0 else None


def prepare_forking(
    store: Store,
    run: str,
    step: str,
    *,
    stream: str | None = None,
    positions: Sequence[int] = (),
    top: int = 5,
    region: str = "answer",
    alternatives: int = 3,
    min_p: float = 0.05,
    n: int = 10,
    max_tokens: int = 4096,
    normalizers: Sequence[str] = (),
    target: str | None = None,
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    alpha: float = 0.05,
    base_url: str | None = None,
    seed_base: int | None = None,
    concurrency: int = 1,
) -> Any:
    from tuningfork.analysis.distribution import target_by_key
    from tuningfork.analysis.normalize import validate_normalizers
    from tuningfork.analysis.tokens import REGIONS, Sampler, TokenStream, candidate_at, candidates
    from tuningfork.backends.formats import FormatError, get_format
    from tuningfork.endpoints import check_endpoint
    from tuningfork.experiment import CheckpointError, resolve_target
    from tuningfork.experiment.forking import ForkingPlan, forking_outcome
    from tuningfork.ops.experiments import default_upstream, load

    if region not in REGIONS:
        raise OperationError(f"region is one of {', '.join(REGIONS)}", field="region")
    if not 1 <= n <= 200:
        raise OperationError("n is the samples per arm, from 1 to 200", field="n")
    if not 1 <= alternatives <= 10:
        raise OperationError("alternatives is from 1 to 10", field="alternatives")
    try:
        normalizers = validate_normalizers(normalizers)
    except ValueError as error:
        raise OperationError(str(error), field="normalizers") from None
    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    base = _base(store, checkpoint, stream)
    tokens = TokenStream.from_json(store.token_stream_data(base))
    fmt = get_format(tokens.format)
    if fmt is None:
        raise OperationError(
            "the base's tokens were not read by a known output format, so a continuation from them "
            "could not be read either", hint="score the answer again on a server whose model tuningfork reads",
        )
    sampler = Sampler.of(checkpoint.request, checkpoint.step.server_defaults)
    # A position after a redacted token cannot be forked: its prefix is not stored.
    first_masked = next((i for i, token in enumerate(tokens.ids) if token is None), len(tokens))
    from tuningfork.ops.inspect import masked_warning

    masked = masked_warning(checkpoint.request, checkpoint.step)
    warnings: list[str] = [masked] if masked else []
    if positions:
        found = []
        for position in positions:
            if not 0 <= position < len(tokens):
                raise OperationError(f"the base has positions 0-{len(tokens) - 1}, not {position}", field="positions")
            if position >= first_masked:
                raise OperationError(f"position {position} comes after a redacted token", field="positions")
            candidate = candidate_at(tokens, position, sampler, alternatives=alternatives, picked=True)
            if candidate is None or not candidate.alternatives:
                raise OperationError(f"nothing is known of what else the model weighed at position {position}",
                                     field="positions")
            found.append(candidate)
        # Named positions are probed as named: where to look, how many and the least chance
        # choose positions only when none are.
        unused = [name for name, value, default in (("region", region, "answer"), ("top", top, 5), ("min_p", min_p, 0.05))
                  if value != default]
        if unused:
            warnings.append(f"{' and '.join(unused)} {'chooses' if len(unused) == 1 else 'choose'} positions only "
                            "when none are named: not used")
    else:
        found = [c for c in candidates(tokens, sampler, top=top, region=region, min_p=min_p, alternatives=alternatives)
                 if c.position < first_masked]
        if not found:
            elsewhere = "reasoning" if region == "answer" else "answer"
            others = [c for c in candidates(tokens, sampler, top=5, region=elsewhere, min_p=min_p, alternatives=alternatives)
                      if c.position < first_masked]
            listing = "; ".join(
                f"[{c.position}] {c.chosen.text!r} or " + " / ".join(repr(a.text) for a in c.alternatives)
                + f" (~{len(tokens) - c.position} tokens to regenerate)"
                for c in others
            )
            raise OperationError(
                f"no token in the {region} had a runner-up the step's sampler would draw at least "
                f"{min_p:.0%} of the time: the model had decided before it"
                + (" — while reasoning, most likely" if region == "answer" else ""),
                detail=f"the best in the {elsewhere}: {listing}" if listing else None,
                hint=f"probe the {elsewhere} instead (`--in {elsewhere}`), or pick the positions yourself (`--at`)",
                field="region",
            )
        found.sort(key=lambda c: c.position)
    arms = []
    for candidate in sorted(found, key=lambda c: c.position):
        for alt in candidate.alternatives:
            arms.append({"position": candidate.position, "id": alt.id, "text": alt.text, "p": alt.p, "original": False})
        chosen = candidate.chosen
        arms.append({"position": candidate.position, "id": chosen.id, "text": chosen.text, "p": chosen.p, "original": True})

    try:
        declared, lens = resolve_target(store, checkpoint, letter=target, pattern=target_match, rules=target_rules)
    except (CheckpointError, ValueError) as error:
        raise OperationError(str(error), field="target_rules" if target_rules else "target") from None
    if lens is not None:
        normalizers = lens
    if declared is None:
        # The base's own behaviour, read by the same parser that reads every continuation.
        from tuningfork.backends.llamacpp.raw import offered_tools

        offered = offered_tools(checkpoint.request, checkpoint.wire.name)
        try:
            parsed = fmt.parse(tokens.text(), tools=bool(offered), thinking=tokens.thinking, offered=offered)
        except FormatError as error:
            raise OperationError(f"the base answer cannot be read back: {error}") from None
        finish = "tool_calls" if parsed.get("tool_calls") else "stop"
        outcome = forking_outcome({"choices": [{"message": parsed, "finish_reason": finish}]}, normalizers)
        declared = target_by_key(outcome.key, f"the base's own behaviour: {outcome.label}")
        if "<ast:" in outcome.key or "<text:" in outcome.key:
            # Found on the real stack: every continuation of a step that ran code wrote its own
            # code, so the base's was never reached again, and a real fork went unseen.
            warnings.append(
                "the target holds the base's code, compared whole: continuations seldom write the same "
                "code, so few may reach it (the tool_name lens compares only which tool was called)"
            )
        if base.notes.get("parse_agrees") is False:
            raise OperationError(
                "the base answer, read the way tuningfork reads a continuation, is not the behaviour the "
                "server recorded: continuations could not be compared with it",
                hint="declare the target yourself (a behaviour's letter, or a pattern)",
            )
    if base.exact is False:
        warnings.append("the base was rebuilt approximately: its tokens are close to, not exactly, the run's own")
    elif (base.notes or {}).get("texts_alike"):
        warnings.append("the base is the likelier of texts the server reads as the run's answer: where they part, "
                        "its token may not be the run's own")
    if any(c.part == "reasoning" for c in found):
        warnings.append("reasoning positions regenerate the rest of the reasoning: each continuation is long")
    if seed_base is None:
        # A new random base, recorded with the probe: one drawn with an earlier probe's
        # seeds would draw its continuations again rather than new ones.
        import secrets

        seed_base = secrets.randbelow(2**31 - 1 - len(arms) * n)
    return ForkingPlan(
        checkpoint=checkpoint,
        base=base,
        stream=tokens,
        candidates=found,
        arms=arms,
        per_arm=n,
        max_tokens=max_tokens,
        seed_base=seed_base,
        concurrency=concurrency,
        normalizers=list(normalizers),
        target=declared,
        alpha=alpha,
        region="picked" if positions else region,
        base_url=check_endpoint(base_url or default_upstream(), home=store.home, purpose="forking tokens"),
        format=fmt.name,
        warnings=warnings,
        seconds_per_token=_seconds_per_token(checkpoint.step),
        prefill_seconds=_prefill_seconds(checkpoint.step),
    )


@dataclass(slots=True)
class ForkingResult:
    experiment: Any
    report: Any
    elapsed: float

    def to_json(self) -> dict[str, Any]:
        from tuningfork.ops.experiments import experiment_json

        return {
            "experiment": experiment_json(self.experiment),
            "forking": self.report.to_json(),
            "elapsed": round(self.elapsed, 2),
        }


def forking_result(store: Store, experiment: Any) -> Any:
    """A stored probe's report, recomputed from its samples."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.analysis.forking import forking_report
    from tuningfork.analysis.normalize import Outcome
    from tuningfork.analysis.tokens import TokenStream
    from tuningfork.experiment.forking import forking_outcome
    from tuningfork.experiment.sampling import target_letters

    settings = experiment.settings
    arms, per_arm = settings.get("arms") or [], int(settings.get("per_arm") or 1)
    normalizers = list(settings.get("normalizers") or [])
    outcomes: list[list[Any]] = [[] for _ in arms]
    for sample in store.samples(experiment.id):
        slot = sample.idx // per_arm
        if slot >= len(outcomes):
            continue
        outcomes[slot].append(
            Outcome("error", "error", sample.error) if sample.error
            else forking_outcome(store.sample_response(sample), normalizers)
        )
    contexts, parts = {}, {}
    record = store.token_stream(settings.get("base_stream") or "")
    if record is not None:
        stream = TokenStream.from_json(store.token_stream_data(record))
        for arm in arms:
            position = arm["position"]
            if position < len(stream):
                contexts[position] = _context(stream, position)
                parts[position] = stream.part(position)
    return forking_report(
        arms, outcomes, per_arm=per_arm, target=target_from_spec(settings.get("target")),
        alpha=float(settings.get("alpha") or 0.05), contexts=contexts, parts=parts,
        letters=target_letters(store, settings.get("target")),
    )


async def execute_forking(
    store: Store, prepared: Any, *, progress: Progress | None = None, cancel: Cancel | None = None
) -> ForkingResult:
    from tuningfork.backends.raw import RawError
    from tuningfork.experiment.forking import ForkingError, run_forking
    from tuningfork.experiment.sampling import finished
    from tuningfork.ops.experiments import _cancellable, _stopped, _tracking

    started = time.monotonic()
    created: list[Any] = []

    async def run() -> Any:
        # Inside the task that is cancelled, so the experiment reads as cancelled, not failed.
        with finished(store, _tracking(created, progress)) as ended:
            return await run_forking(store, prepared, progress=ended)

    try:
        experiment = await _cancellable(run(), cancel)
    except Cancelled as stopped:
        _stopped(store, created, stopped.reason)
        partial = (
            ForkingResult(created[0], forking_result(store, created[0]), time.monotonic() - started)
            if created else None
        )
        raise Cancelled(stopped.reason, partial=partial) from None
    except (RawError, ForkingError) as error:
        raise OperationError(str(error)) from None
    return ForkingResult(experiment, forking_result(store, experiment), time.monotonic() - started)
