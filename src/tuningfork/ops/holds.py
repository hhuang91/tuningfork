"""Tools at a held call: evidence, and hypotheses — kept apart.

**Sampling** a held call is an ordinary experiment: the held request, sent N times through
the proxy, its behaviours clustered. It has no step yet — the call has not been recorded —
so it is anchored to the hold, and filed under the step the call becomes once it is (see
`Store.link_holds`). If a person then edits the request before the model sees it, the
samples are not of that step: the experiment says so (`origin.matches_step`), and a
lettered A/B target will not come from it.

**The console** asks the model one question at the held call: the held context, the
model's held answer when there is one, and the question — once, not streamed, never
recorded. The reply is the model's account of itself: a hypothesis, not evidence of what
it does. It is kept only in its job's result. Resampling is the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tuningfork.ops.common import Cancel, OperationError, Progress

#: A question longer than this is a paste gone wrong.
MAX_QUESTION = 4000
HYPOTHESIS = (
    "A hypothesis: the model's account of itself, which need not be what drives it. "
    "Resampling is the evidence."
)


def _held(proxy: Any, hold_id: str) -> dict[str, Any]:
    held = proxy.held_call(hold_id)
    if not held["pending"]:
        raise OperationError(
            f"the call held as {held['id']} is no longer waiting", code="not_found", field="hold",
        )
    return held


def _hold_json(held: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "phase", "reason", "run_id", "step", "task_id", "launch_id", "held_seconds", "wire")
    return {key: held.get(key) for key in keys}


# --- evidence: sampling the held call ---------------------------------------------------------


@dataclass(slots=True)
class HoldSamplePlan:
    hold: dict[str, Any]
    anchor: Any  # experiment.sampling.Anchor
    plan: Any  # experiment.SamplingPlan
    base_url: str
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "hold": _hold_json(self.hold),
            "sampling": self.plan.as_settings(),
            "base_url": self.base_url,
            "warnings": self.warnings,
        }


def prepare_hold_sample(
    store: Any,
    proxy: Any,
    hold_id: str,
    *,
    n: int = 10,
    concurrency: int = 1,
    max_tokens: int = 4096,
    seed_base: int | None = None,
    prompt_cache: bool = False,
    normalizers: list[str] | None = None,
) -> HoldSamplePlan:
    from tuningfork.experiment.sampling import Anchor
    from tuningfork.ops.experiments import make_plan
    from tuningfork.wire import get_wire

    held = _held(proxy, hold_id)
    status = proxy.status()
    server = status.raw.get("server") or {}
    warnings: list[str] = []
    if concurrency > status.max_inflight:
        warnings.append(
            f"{concurrency} at once asked, and the proxy lets {status.max_inflight} through: "
            f"sampling {status.max_inflight} at a time"
        )
        concurrency = status.max_inflight
    request = held["request"]
    if not request.get("stream"):
        warnings.append(
            "the harness asked for a whole answer, not a stream: nothing keeps its connection "
            "alive while it waits, so it may give up before the samples are drawn"
        )
    if held.get("launch_id"):
        launch = store.get_launch(held["launch_id"])
        timeout = ((launch.eval or {}).get("tuningfork") or {}).get("timeout") if launch else None
        if isinstance(timeout, (int, float)):
            left = timeout - (time.time() - launch.created_at)
            warnings.append(
                f"its harness is stopped about {max(0, left):.0f} s from now (the task's timeout, "
                "held time included)"
            )
    anchor = Anchor(
        run_id=None, step_idx=None, step_id=None, request=request, wire=get_wire(held["wire"]),
        # The call is live: the server answering it now is the one to compare with.
        server_defaults=server.get("defaults") or {}, model=request.get("model"),
        server_id=server.get("id"), hold_id=held["id"],
    )
    plan = make_plan(
        n=n, concurrency=concurrency, max_tokens=max_tokens, seed_base=seed_base,
        prompt_cache=prompt_cache, normalizers=normalizers or [],
    )
    return HoldSamplePlan(hold=held, anchor=anchor, plan=plan, base_url=proxy.url, warnings=warnings)


@dataclass(slots=True)
class HoldSampleResult:
    experiment: Any
    distribution: Any
    #: The held answer's behaviour, after the model: which of the sampled ones it is.
    held: Any = None
    held_letter: str | None = None

    def to_json(self) -> dict[str, Any]:
        from tuningfork.ops.experiments import experiment_json

        return {
            "experiment": experiment_json(self.experiment),
            "distribution": self.distribution.to_json(),
            "held": self.held.to_json() if self.held is not None else None,
            "held_letter": self.held_letter,
        }


async def execute_hold_sample(
    store: Any, prepared: HoldSamplePlan, *, progress: Progress | None = None, cancel: Cancel | None = None,
) -> HoldSampleResult:
    import httpx

    from tuningfork.analysis.normalize import outcome_for
    from tuningfork.endpoints import guard_hooks
    from tuningfork.experiment import experiment_distribution
    from tuningfork.experiment.sampling import finished, run_arm
    from tuningfork.ops.experiments import _cancellable, _tracking

    created: list[Any] = []
    held = prepared.hold
    annotations = {"origin": {"hold": held["id"], "phase": held["phase"], "reason": held.get("reason")}}
    async with httpx.AsyncClient(
        timeout=prepared.plan.timeout, event_hooks=guard_hooks(home=store.home)
    ) as client:

        async def sample() -> Any:
            with finished(store, _tracking(created, progress)) as ended:
                return await run_arm(
                    store, prepared.anchor, prepared.plan, base_url=prepared.base_url,
                    request=prepared.anchor.request, kind="resample", arm="original",
                    client=client, annotations=annotations, progress=ended,
                )

        experiment = await _cancellable(sample(), cancel)
    found = experiment_distribution(store, experiment)
    result = HoldSampleResult(experiment=store.get_experiment(experiment.id) or experiment, distribution=found)
    if held["phase"] == "response" and held.get("response"):
        result.held = outcome_for(held["response"], experiment.normalizers)
        result.held_letter = next(
            (cluster.letter for cluster in found.clusters if cluster.outcome.key == result.held.key), None
        )
    return result


# --- hypotheses: asking the model ------------------------------------------------------------


@dataclass(slots=True)
class ConsolePlan:
    hold: dict[str, Any]
    question: str
    request: dict[str, Any]
    wire: Any
    base_url: str

    def to_json(self) -> dict[str, Any]:
        return {
            "hold": _hold_json(self.hold),
            "question": self.question,
            "sends": (
                "the held context, the model's held answer, and your question"
                if self.hold["phase"] == "response"
                else "the held context and your question"
            ) + " — once, not streamed, never recorded",
            "note": HYPOTHESIS,
        }


def prepare_console(proxy: Any, hold_id: str, question: str) -> ConsolePlan:
    from tuningfork.wire import get_wire

    question = (question or "").strip()
    if not question:
        raise OperationError("ask the model something", field="question")
    if len(question) > MAX_QUESTION:
        raise OperationError(f"a question is at most {MAX_QUESTION} characters", field="question")
    held = _held(proxy, hold_id)
    wire = get_wire(held["wire"])
    answer = held.get("response") if held["phase"] == "response" else None
    request = wire.with_user_message(held["request"], question, answer=answer)
    return ConsolePlan(hold=held, question=question, request=request, wire=wire, base_url=proxy.url)


@dataclass(slots=True)
class ConsoleReply:
    hold: dict[str, Any]
    question: str
    text: str
    reasoning: str
    tool_calls: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "hold": _hold_json(self.hold),
            "question": self.question,
            "reply": {"text": self.text, "reasoning": self.reasoning, "tool_calls": self.tool_calls},
            "hypothesis": True,
            "note": HYPOTHESIS,
        }


async def execute_console(
    store: Any, prepared: ConsolePlan, *, progress: Progress | None = None, cancel: Cancel | None = None,
) -> ConsoleReply:
    import httpx

    from tuningfork.config import EXPERIMENT_HEADER
    from tuningfork.endpoints import guard_hooks
    from tuningfork.ops.experiments import _cancellable

    # Sent as an experiment's call: through the gate and the pause switch, never recorded
    # or held.
    async with httpx.AsyncClient(
        headers={EXPERIMENT_HEADER: "console"}, timeout=600, event_hooks=guard_hooks(home=store.home)
    ) as client:
        response = await _cancellable(
            client.post(f"{prepared.base_url}{prepared.wire.endpoint}", json=prepared.request), cancel
        )
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400 or not isinstance(body, dict):
        detail = (body or {}).get("error") if isinstance(body, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else response.text[:300]
        raise OperationError(f"the model's server answered {response.status_code}: {message}")
    from tuningfork.analysis.tokens import split_logprobs

    # A held harness request may have asked for logprobs: the answer's text is what is kept —
    # masked with the job's result, as the store keeps it.
    common, _ = split_logprobs(prepared.wire.canonical_response(body))
    choices = common.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    return ConsoleReply(
        hold=prepared.hold,
        question=prepared.question,
        text=message.get("content") or "",
        reasoning=message.get("reasoning_content") or "",
        tool_calls=[call for call in message.get("tool_calls") or [] if isinstance(call, dict)],
    )
