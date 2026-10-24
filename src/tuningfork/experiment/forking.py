"""The forking-token probe (after Bigelow et al., "Forking Paths in Neural Text Generation",
ICLR 2025): force one token of an answer, let the model go on, and see whether what it does
changes.

The base is an answer as tokens (a stored token stream: the step's own answer scored, or a
sample's). At each probed position the model is given the step's exact prompt and the base's
tokens up to the position, then one token forced — the base's own (the baseline arm) or an
alternative the model weighed there — and continues, n times per arm, at the step's own
sampling settings. Every continuation is read back as the server would read it
(`tuningfork.backends.formats`) into a behaviour; `tuningfork.analysis.forking` compares the
arms.

Arms are drawn position by position, ascending, alternatives before the baseline: the prompt
cache keeps the longest prefix the next arm shares. The prompt is sent as token ids and never
stored; the experiment keeps a reference to the base stream and the arms instead.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from tuningfork.analysis.distribution import Target
from tuningfork.analysis.normalize import NORMALIZER_VERSION, Outcome, outcome_for
from tuningfork.analysis.tokens import TokenStream, from_entries, redact_stream
from tuningfork.backends.formats import FormatError, get_format
from tuningfork.backends.raw import RawError
from tuningfork.config import redact_json
from tuningfork.ops.common import emit
from tuningfork.store.db import Store

#: A sample whose answer the output format could not read keeps its text under this key.
UNPARSED = "tuningfork_unparsed"


class ForkingError(RuntimeError):
    """The probe cannot run as planned: said why."""


def forking_outcome(response: dict[str, Any] | None, normalizers: list[str]) -> Outcome:
    """A probe sample's behaviour, from what is stored. A continuation cut off by its token
    budget is truncated, whatever it was writing; one that ended but that the format could not
    read is a behaviour of its own — the model wrote it — not a transport error."""
    if isinstance(response, dict) and response.get(UNPARSED):
        choices = response.get("choices") or [{}]
        if choices[0].get("finish_reason") == "length":
            return outcome_for(response, normalizers)
        return Outcome("unparseable", "unparseable", f"not readable as the server would read it: {response[UNPARSED]}")
    return outcome_for(response, normalizers)


@dataclass(slots=True)
class ForkingPlan:
    checkpoint: Any
    #: The base stream's record, and its tokens.
    base: Any
    stream: TokenStream
    candidates: list[Any]
    #: {"position", "id", "text", "p", "original"}, in the order they are drawn.
    arms: list[dict[str, Any]]
    per_arm: int
    max_tokens: int
    seed_base: int
    concurrency: int
    normalizers: list[str]
    target: Target
    alpha: float
    region: str
    base_url: str
    format: str
    warnings: list[str] = field(default_factory=list)
    #: Seconds per generated token, from the step's own timing, when known.
    seconds_per_token: float | None = None
    #: Seconds to process the step's whole prompt, which every continuation does first.
    prefill_seconds: float | None = None

    @property
    def planned(self) -> int:
        return len(self.arms) * self.per_arm

    @property
    def family(self) -> int:
        return sum(1 for arm in self.arms if not arm["original"])

    def budget(self, position: int) -> int:
        """Tokens a continuation from `position` may generate: room for a different, longer
        path than the base's — a script where the base read a file — capped by the plan's cap
        per sample. Hitting it counts as truncated. Measured: at 256, a tenth of one real
        probe's continuations were cut off writing code."""
        remaining = len(self.stream) - position
        return max(1, min(self.max_tokens - position, max(1024, 4 * remaining)))

    def expected_tokens(self) -> int:
        return sum(self.per_arm * max(1, len(self.stream) - arm["position"]) for arm in self.arms)

    def settings(self) -> dict[str, Any]:
        return {
            "n": self.planned,
            "per_arm": self.per_arm,
            "arms": self.arms,
            "base_stream": self.base.id,
            "target": self.target.spec,
            "alpha": self.alpha,
            "region": self.region,
            "max_tokens": self.max_tokens,
            "seed_base": self.seed_base,
            "concurrency": self.concurrency,
            "normalizers": self.normalizers,
            "normalizer_version": NORMALIZER_VERSION,
            "format": self.format,
        }

    def to_json(self) -> dict[str, Any]:
        from tuningfork.analysis.forking import detectable

        expected = self.expected_tokens()
        return {
            "checkpoint": self.checkpoint.to_json(),
            "base": self.base.to_json(),
            "positions": [candidate.to_json() for candidate in self.candidates],
            "arms": self.arms,
            "per_arm": self.per_arm,
            "planned": self.planned,
            "family": self.family,
            "target": self.target.description,
            "alpha": self.alpha,
            "region": self.region,
            "base_url": self.base_url,
            "warnings": self.warnings,
            "power": {
                "detectable": detectable(self.per_arm, self.family, self.alpha),
                "of": self.per_arm,
            },
            "estimate": {
                "tokens": expected,
                "minutes": self.minutes(expected),
                # Each continuation processes the step's whole prompt first (backends.llamacpp.raw):
                # counted in the minutes when the step's time to first token says how long it takes.
                "prompt_tokens": (self.checkpoint.step.usage or {}).get("prompt_tokens"),
                "prefill_seconds": round(self.prefill_seconds, 1) if self.prefill_seconds else None,
            },
        }

    def minutes(self, expected: int) -> float | None:
        if not self.seconds_per_token:
            return None
        seconds = expected * self.seconds_per_token + self.planned * (self.prefill_seconds or 0.0)
        return round(seconds / 60, 1)


def _arm_label(arm: dict[str, Any]) -> str:
    return f"{arm['position']}:{arm['text']!r}"


async def run_forking(store: Store, plan: ForkingPlan, *, progress: Any = None) -> Any:
    """Draw every arm's continuations and store them. Reports `experiment` once it exists,
    then `sample` as each lands. Returns the experiment, `running` until the caller says how
    it ended (`sampling.finished`)."""
    from tuningfork.experiment.sampling import Anchor, SampleEvent, SamplingPlan, check_parity
    from tuningfork.experiment.scoring import other_context, raw_session

    checkpoint = plan.checkpoint
    fmt = get_format(plan.format)
    if fmt is None:
        raise ForkingError(f"no output format {plan.format!r}")
    async with raw_session(store, plan.base_url, tag="forking-tokens", progress=progress) as (client, raw, server):
        prompt = await raw.render(checkpoint.request, checkpoint.wire.name)
        recorded = (checkpoint.step.usage or {}).get("prompt_tokens")
        if recorded is not None and recorded != len(prompt.ids):
            raise ForkingError(
                f"the prompt renders to {len(prompt.ids)} tokens here, and the step's had {recorded}: "
                "the model would be asked about a different context. " + other_context(checkpoint.request, checkpoint.step.params)
            )
        thinking = fmt.thinking(prompt.text)
        server_defaults, notes = await check_parity(
            client, plan.base_url, Anchor.of(checkpoint), SamplingPlan(n=1, max_tokens=plan.max_tokens)
        )
        experiment = await asyncio.to_thread(
            store.create_experiment,
            kind="forking-tokens",
            arm="probe",
            run_id=checkpoint.run.id,
            step_idx=checkpoint.step.idx,
            step_id=checkpoint.step.id,
            # A reference, never the prompt's token ids: those would bypass redaction.
            request=redact_json({"base_stream": plan.base.id, "arms": plan.arms}),
            settings={**plan.settings(), "sampling": prompt.sampling, "server": server, "warnings": plan.warnings},
            base_url=plan.base_url,
            server_defaults=server_defaults,
            notes=notes,
        )
        emit(progress, "experiment", subject=experiment, id=experiment.id, kind="forking-tokens",
             arm="probe", group_id=None, n=plan.planned, reach="step")
        semaphore = asyncio.Semaphore(plan.concurrency)

        async def draw(arm_index: int, arm: dict[str, Any], j: int) -> None:
            index = arm_index * plan.per_arm + j
            seed = plan.seed_base + index
            position = arm["position"]
            prefix = [token for token in plan.stream.ids[:position]] + [arm["id"]]
            started = time.monotonic()
            generation, error = None, None
            async with semaphore:
                try:
                    generation = await raw.complete(
                        prompt, prefix, n_predict=plan.budget(position), seed=seed, n_probs=1
                    )
                except RawError as failed:
                    error = str(failed)
            seconds = time.monotonic() - started
            stored, finish = None, None
            if generation is not None:
                text = plan.stream.text(0, position) + arm["text"] + generation.text
                finish = "length" if generation.stop == "limit" else None
                usage = {"completion_tokens": position + 1 + len(generation.ids)}
                try:
                    message = fmt.parse(text, tools=prompt.tools, parallel=prompt.parallel,
                                        thinking=thinking, offered=prompt.offered)
                    finish = finish or ("tool_calls" if message.get("tool_calls") else "stop")
                    stored = {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
                except FormatError as unreadable:
                    finish = finish or "stop"
                    stored = {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                           "finish_reason": finish}], "usage": usage, UNPARSED: str(unreadable)}
                stored = redact_json(stored)
            outcome = Outcome("error", "error", error) if error else forking_outcome(stored, plan.normalizers)
            sample = await asyncio.to_thread(
                store.add_sample, experiment.id, index, seed=seed, response=stored,
                outcome_kind=outcome.kind, outcome_key=outcome.key, outcome_label=outcome.label,
                finish_reason=finish, usage=(stored or {}).get("usage") or {},
                timings={"total_ms": round(seconds * 1000, 2)}, error=error, forced=dict(arm),
            )
            if generation is not None and generation.entries:
                stream = redact_stream(from_entries(generation.entries, top_k=1))
                await asyncio.to_thread(
                    store.add_token_stream, stream.to_json(), source="continuation",
                    sample_id=sample.id, server_id=server, method="completion",
                )
            event = SampleEvent(
                arm=_arm_label(arm), index=index, seed=seed, outcome=outcome, seconds=seconds,
                completion_tokens=len(generation.ids) if generation else None,
                experiment_id=experiment.id, error=error,
            )
            emit(progress, "sample", subject=event, **event.to_json(),
                 position=position, token=arm["text"], original=arm["original"])

        for arm_index, arm in enumerate(plan.arms):
            await asyncio.gather(*(draw(arm_index, arm, j) for j in range(plan.per_arm)))
        return experiment
