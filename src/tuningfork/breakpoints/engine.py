"""The breakpoint engine: which model calls to pause, and what to do with them.

The engine lives in the proxy process — the one place live traffic passes — and is driven
from outside through the control API, which the CLI, the web UI, and eventually an LLM all
share. It never touches the network itself: the proxy asks it whether to hold, waits on
the hold it gets back, and acts on the decision.

A decision is one of three things:

- **continue** — let the call proceed, optionally with an edited request (before the model)
  or an edited response (before the harness).
- **step** — continue, and stop again at the next stop point: from before a call to after
  it, or from after a call to before the harness's next one.
- **drop** — fail the call. The harness receives an error, as it would from a server fault.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from tuningfork.analysis.normalize import outcome_for, recorded_outcome
from tuningfork.breakpoints.conditions import (
    Breakpoint,
    BreakpointError,
    ExchangeContext,
    request_matches,
    response_matches,
)
from tuningfork.breakpoints.events import EventBus
from tuningfork.store.masking import Masking
from tuningfork.store.db import Store
from tuningfork.store.tree import node_path
from tuningfork.wire import DEFAULT as DEFAULT_WIRE
from tuningfork.wire import WireFormat

ACTIONS = ("continue", "step", "drop")


class HoldError(LookupError):
    """No such hold, or a decision that cannot apply to it."""


@dataclass(slots=True)
class Decision:
    action: str
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


@dataclass(slots=True)
class Hold:
    """A model call paused at a stop point, waiting for a decision."""

    id: str
    phase: str
    reason: str
    breakpoint_id: str | None
    run_id: str | None
    task_id: str | None
    step: int
    request: dict[str, Any]
    response: dict[str, Any] | None
    created_at: float
    future: asyncio.Future = field(repr=False)
    decision: Decision | None = None
    resolved_at: float | None = None
    abandoned: bool = False
    #: The request's conversation as OpenAI-style messages, for display.
    view: list[dict[str, Any]] = field(default_factory=list)
    wire: str = DEFAULT_WIRE.name
    #: The launch the call belongs to: a job that started it can find its holds.
    launch_id: str | None = None

    @property
    def pending(self) -> bool:
        return self.decision is None and not self.abandoned

    def summary(self, payload: bool = False) -> dict[str, Any]:
        end = self.resolved_at or time.time()
        data: dict[str, Any] = {
            "id": self.id,
            "phase": self.phase,
            "reason": self.reason,
            "breakpoint_id": self.breakpoint_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": self.step,
            "created_at": self.created_at,
            "held_seconds": round(end - self.created_at, 1),
            "pending": self.pending,
            "action": self.decision.action if self.decision else None,
            # Whether the decision carried an edit. The proxy ignores an "edit" identical
            # to the original, so a recorded step is the authority on what really changed.
            "edited": bool(
                self.decision
                and (self.decision.request is not None or self.decision.response is not None)
            ),
            "abandoned": self.abandoned,
            "launch_id": self.launch_id,
        }
        data["wire"] = self.wire
        if payload:
            # `request` is in the call's own format, and is what an edit replaces;
            # `response` is in the common shape.
            data["request"] = self.request
            data["view"] = self.view
            data["response"] = self.response
        return data


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise HoldError("an edited request must be an object with a `messages` list")
    return request


def _validate_response(response: Any) -> dict[str, Any]:
    """Check an edited response before any of it reaches the harness.

    Tool-call arguments travel as JSON *text*. Hand-editing that escaped string is awkward
    enough that a person will write an object instead; that is read as the object they
    meant. Anything else malformed is refused, leaving the call held: rendering it would
    fail after the hold was released, losing both the harness's answer and the recording.
    """
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise HoldError("an edited response must be an object with a non-empty `choices` list")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise HoldError("an edited response's first choice needs a `message` object")
    content = message.get("content")
    if content is not None and not isinstance(content, (str, list)):
        raise HoldError("a message's `content` must be text")
    calls = message.get("tool_calls")
    if calls is None:
        return response
    if not isinstance(calls, list):
        raise HoldError("`tool_calls` must be a list")
    for position, call in enumerate(calls):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise HoldError(f"tool call {position} needs a `function` with a `name`")
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            raise HoldError(
                f"tool call {position}: `arguments` must be a JSON object, or the JSON text of one"
            )
    return response


class BreakpointEngine:
    def __init__(
        self,
        store: Store | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.store = store
        self.bus = bus or EventBus()
        #: The store's: a request is masked as it is stored, so a prediction of where it lands
        #: must mask it the same way (see `context_for`).
        self.masking = store.masking if store is not None else Masking()
        self._breakpoints: dict[str, Breakpoint] = {}
        self._pending: dict[str, Hold] = {}
        self._recent: deque[Hold] = deque(maxlen=50)
        #: Node ids to stop after: the next request continuing from one of them is held.
        #: Keyed on the message path rather than a run id because the first call of a new
        #: run has no id yet when it is held — but the harness's next request always
        #: extends the request before it.
        self._armed: set[str] = set()
        self._golden: dict[tuple[str, int], str | None] = {}

    # --- registry -----------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether any call could be held. When False, the proxy takes its usual path."""
        return bool(self._armed) or any(bp.enabled for bp in self._breakpoints.values())

    def breakpoints(self) -> list[Breakpoint]:
        return sorted(self._breakpoints.values(), key=lambda bp: bp.created_at)

    def get(self, breakpoint_id: str) -> Breakpoint:
        matches = [bp for key, bp in self._breakpoints.items() if key.startswith(breakpoint_id)]
        if len(matches) != 1:
            raise BreakpointError(f"no single breakpoint matching {breakpoint_id!r}")
        return matches[0]

    def add(self, definition: Breakpoint | dict[str, Any]) -> Breakpoint:
        return self.install(self.prepare(definition))

    def prepare(self, definition: Breakpoint | dict[str, Any]) -> Breakpoint:
        """Validate a definition and resolve its run references, without installing it.

        Reads the store, so the proxy calls it off the event loop; `install` then runs on
        the loop, where the breakpoints are read while calls go through.
        """
        bp = definition if isinstance(definition, Breakpoint) else Breakpoint.from_dict(definition)
        # Resolve run references now, so a typo fails here rather than never matching.
        for name in ("run", "diverges_from"):
            reference = getattr(bp, name)
            if reference and self.store is not None:
                run = self.store.get_run(reference)
                if run is None:
                    raise BreakpointError(f"{name}: no run matching {reference!r}")
                setattr(bp, name, run.id)
        return bp

    def install(self, bp: Breakpoint) -> Breakpoint:
        if bp.id in self._breakpoints:
            raise BreakpointError(f"breakpoint {bp.id} already exists")
        self._breakpoints[bp.id] = bp
        self._publish_breakpoints()
        return bp

    def update(self, breakpoint_id: str, *, enabled: bool | None = None) -> Breakpoint:
        if enabled is not None and not isinstance(enabled, bool):
            raise BreakpointError("enabled must be true or false")
        bp = self.get(breakpoint_id)
        if enabled is not None:
            bp.enabled = enabled
        self._publish_breakpoints()
        return bp

    def remove(self, breakpoint_id: str) -> Breakpoint:
        bp = self.get(breakpoint_id)
        del self._breakpoints[bp.id]
        self._publish_breakpoints()
        return bp

    def _publish_breakpoints(self) -> None:
        self.bus.publish("breakpoints", breakpoints=[bp.to_json() for bp in self.breakpoints()])

    # --- deciding whether to hold ---------------------------------------------------

    def context_for(
        self, request: dict[str, Any], wire: WireFormat | None = None, assignment: Any = None
    ) -> ExchangeContext:
        """Where a request will land. A launched call already knows (`assignment`, see
        `tuningfork.launches`); any other is predicted from the store, so call this off the
        event loop."""
        wire = wire or DEFAULT_WIRE
        # As `record_step` will store it: masked, then in the tree's stable form. A path built
        # from the raw request is never found once a mask has changed a message, and every
        # step-scoped condition for the rest of that run misfires.
        stored = self.masking.value({"messages": request.get("messages") or []})
        messages = [wire.normalize_message(m) for m in stored.get("messages") or []]
        run_id, step, task_id = None, 0, None
        if assignment is not None:
            run_id, step, task_id = assignment.run_id, assignment.idx, assignment.task_id
            if task_id is None and self.store is not None:
                run = self.store.get_run(assignment.run_id)
                task_id = run.task_id if run else None
        elif self.store is not None:
            run_id, step = self.store.predict_attachment(messages)
            if run_id:
                run = self.store.get_run(run_id)
                task_id = run.task_id if run else None
            else:
                pending = self.store.next_pending()
                task_id = pending[1] if pending else None
        return ExchangeContext(
            request=request,
            run_id=run_id,
            task_id=task_id,
            step=step,
            path=node_path(messages),
            view=wire.view(request),
            wire=wire.name,
            launch_id=assignment.launch_id if assignment is not None else None,
        )

    def check_request(self, ctx: ExchangeContext) -> Hold | None:
        armed = self._armed.intersection(ctx.path)
        if armed:
            self._armed -= armed
            return self._open(ctx, "request", reason="stepped here from the previous call")
        for bp in self.breakpoints():
            if bp.phase == "request" and request_matches(bp, ctx):
                self._hit(bp)
                return self._open(ctx, "request", reason=bp.describe(), breakpoint=bp)
        return None

    def wants_response(self, ctx: ExchangeContext) -> bool:
        """Whether this call's response might be held, so must be buffered, not streamed.

        Only response breakpoints whose request-side conditions already hold count: a
        breakpoint scoped to another run should not stop this one from streaming live.
        """
        return any(
            bp.phase == "response" and request_matches(bp, ctx)
            for bp in self._breakpoints.values()
        )

    def check_response(
        self, ctx: ExchangeContext, response: dict[str, Any] | None, *, stepping: bool = False
    ) -> Hold | None:
        if stepping:
            return self._open(
                ctx, "response", reason="stepped here from before the call", response=response
            )
        for bp in self.breakpoints():
            if bp.phase == "response" and response_matches(bp, ctx, response, self._diverged):
                self._hit(bp)
                return self._open(
                    ctx, "response", reason=bp.describe(), breakpoint=bp, response=response
                )
        return None

    def arm_next(self, ctx: ExchangeContext) -> None:
        """Stop before the harness's next call that continues from this one."""
        if ctx.path:
            self._armed.add(ctx.path[-1])

    def _diverged(self, golden_run: str, step: int, response: dict[str, Any] | None) -> bool:
        key = (golden_run, step)
        if key not in self._golden:
            golden = self.store.step(golden_run, step) if self.store else None
            self._golden[key] = (
                recorded_outcome(self.store.response(golden), golden.error).key
                if golden and self.store
                else None
            )
        # A golden run that has no step here counts as divergence: one run has ended and
        # the other has not.
        return outcome_for(response).key != self._golden[key]

    def _hit(self, bp: Breakpoint) -> None:
        bp.hits += 1
        if bp.once:
            bp.enabled = False
        self._publish_breakpoints()

    def _open(
        self,
        ctx: ExchangeContext,
        phase: str,
        *,
        reason: str,
        breakpoint: Breakpoint | None = None,
        response: dict[str, Any] | None = None,
    ) -> Hold:
        hold = Hold(
            id=uuid.uuid4().hex[:8],
            phase=phase,
            reason=reason,
            breakpoint_id=breakpoint.id if breakpoint else None,
            run_id=ctx.run_id,
            task_id=ctx.task_id,
            step=ctx.step,
            request=copy.deepcopy(ctx.request),
            response=copy.deepcopy(response),
            created_at=time.time(),
            view=copy.deepcopy(ctx.messages),
            wire=ctx.wire,
            launch_id=ctx.launch_id,
            future=asyncio.get_running_loop().create_future(),
        )
        self._pending[hold.id] = hold
        self.bus.publish("hold", hold=hold.summary())
        return hold

    # --- holds ----------------------------------------------------------------------

    def holds(self, *, recent: bool = False) -> list[Hold]:
        pending = sorted(self._pending.values(), key=lambda hold: hold.created_at)
        return pending + list(self._recent) if recent else pending

    def get_hold(self, hold_id: str) -> Hold:
        candidates = [*self._pending.values(), *self._recent]
        matches = [hold for hold in candidates if hold.id.startswith(hold_id)]
        if len(matches) != 1:
            raise HoldError(f"no single hold matching {hold_id!r}")
        return matches[0]

    def resolve(
        self,
        hold_id: str,
        action: str,
        *,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
    ) -> Hold:
        hold = self.get_hold(hold_id)
        if not hold.pending:
            state = hold.decision.action if hold.decision else "abandoned"
            raise HoldError(f"hold {hold.id} was already {state}")
        if action not in ACTIONS:
            raise HoldError(f"action must be one of {', '.join(ACTIONS)}")
        if request is not None:
            if hold.phase != "request":
                raise HoldError(
                    "the request can only be edited before the model sees it; this hold is "
                    "after the model answered — edit the response instead"
                )
            _validate_request(request)
        if response is not None:
            if hold.phase != "response":
                raise HoldError("there is no response to edit yet: the model has not answered")
            _validate_response(response)

        hold.decision = Decision(action=action, request=request, response=response)
        hold.resolved_at = time.time()
        self._retire(hold)
        if not hold.future.done():
            hold.future.set_result(hold.decision)
        self.bus.publish("resolved", hold=hold.summary())
        return hold

    async def wait(self, hold: Hold) -> Decision:
        """Until someone decides. Cancelling the waiter — the harness hung up — propagates,
        and the caller is expected to `abandon` the hold."""
        return await hold.future

    def abandon(self, hold: Hold) -> None:
        """The harness hung up while its call was held; nothing is waiting any more."""
        if not hold.pending:
            return
        hold.abandoned = True
        hold.resolved_at = time.time()
        self._retire(hold)
        self.bus.publish("resolved", hold=hold.summary())

    def release_all(self) -> int:
        """Continue every pending hold, unedited. Used on shutdown and by `resume all`."""
        pending = list(self._pending)
        for hold_id in pending:
            self.resolve(hold_id, "continue")
        return len(pending)

    def _retire(self, hold: Hold) -> None:
        self._pending.pop(hold.id, None)
        self._recent.appendleft(hold)
