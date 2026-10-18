"""Breakpoint conditions and the engine that holds calls.

The condition tests pin down the cases that are easy to get subtly wrong: "write_file on
*.json" must be about one tool call, not any mix of calls; request-phase breakpoints must
refuse conditions that only exist after the model answers.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tuningfork.breakpoints import (
    Breakpoint,
    BreakpointEngine,
    BreakpointError,
    EventBus,
    ExchangeContext,
    HoldError,
    request_matches,
    response_matches,
)
from tuningfork.breakpoints.events import OVERFLOWED
from tuningfork.store.tree import node_path
from tests.helpers import as_response, record_run, text_message, tool_call_message


def ctx(messages=None, *, step=0, run_id=None, task_id=None) -> ExchangeContext:
    messages = messages or [{"role": "user", "content": "go"}]
    return ExchangeContext(
        request={"model": "m", "messages": messages},
        run_id=run_id,
        task_id=task_id,
        step=step,
        path=node_path(messages),
    )


def write(path: str, content: str = "{}") -> dict:
    return as_response(tool_call_message("write_file", {"path": path, "content": content}))


READ_THEN_RESULT = [
    {"role": "user", "content": "go"},
    tool_call_message("read_file", {"path": "data.csv"}, call_id="c1"),
    {"role": "tool", "tool_call_id": "c1", "content": "id,amount"},
]


class TestDefinitions:
    def test_request_phase_refuses_response_conditions(self):
        with pytest.raises(BreakpointError, match="response phase"):
            Breakpoint(phase="request", tool="write_file")

    def test_invalid_regex_is_refused_up_front(self):
        with pytest.raises(BreakpointError, match="valid regex"):
            Breakpoint(phase="response", output_regex="(unclosed")

    def test_unknown_fields_are_refused(self):
        with pytest.raises(BreakpointError, match="unknown"):
            Breakpoint.from_dict({"phase": "request", "stepp_min": 2})

    def test_a_phase_is_required(self):
        with pytest.raises(BreakpointError, match="phase"):
            Breakpoint.from_dict({"step_min": 2})

    def test_inverted_step_range_is_refused(self):
        with pytest.raises(BreakpointError):
            Breakpoint(phase="request", step_min=4, step_max=2)

    @pytest.mark.parametrize("field", ["id", "hits", "created_at"])
    def test_the_engines_own_fields_cannot_be_set(self, field):
        """Found in review: `created_at: "x"` broke the sort behind every listing, and
        with the breakpoint installed every model call through the proxy failed."""
        with pytest.raises(BreakpointError, match="set by tuningfork"):
            Breakpoint.from_dict({"phase": "request", field: "x"})

    @pytest.mark.parametrize(
        "field, value",
        [("step_min", "3"), ("step_min", -1), ("step_min", True), ("after_tool", 5),
         ("tool", 5), ("path_glob", ["*.json"]), ("once", "yes"), ("enabled", 1)],
    )
    def test_wrong_types_are_refused_before_anything_is_installed(self, field, value):
        phase = "response" if field in ("tool", "path_glob") else "request"
        with pytest.raises(BreakpointError):
            Breakpoint.from_dict({"phase": phase, field: value})

    async def test_a_refused_definition_leaves_the_engine_working(self):
        engine = BreakpointEngine()
        engine.add({"phase": "request", "step_min": 2})
        with pytest.raises(BreakpointError):
            engine.add({"phase": "request", "created_at": "x"})
        assert len(engine.breakpoints()) == 1
        assert engine.check_request(ctx(step=3)) is not None

    def test_description_reads_like_the_intent(self):
        bp = Breakpoint(phase="response", tool="write_file", path_glob="*.json", once=True)
        assert bp.describe() == "after the model: write_file on *.json (once)"
        assert Breakpoint(phase="request", step_min=2).describe() == "before the model: step >= 2"
        assert Breakpoint(phase="request").describe() == "before the model: every call"


class TestRequestConditions:
    def test_step_range(self):
        bp = Breakpoint(phase="request", step_min=1, step_max=2)
        assert not request_matches(bp, ctx(step=0))
        assert request_matches(bp, ctx(step=1))
        assert not request_matches(bp, ctx(step=3))

    def test_after_a_particular_tool_returns(self):
        bp = Breakpoint(phase="request", after_tool="read_*")
        assert request_matches(bp, ctx(READ_THEN_RESULT))
        assert not request_matches(bp, ctx())

    def test_context_regex_sees_tool_calls_too(self):
        bp = Breakpoint(phase="request", context_regex=r"data\.csv")
        assert request_matches(bp, ctx(READ_THEN_RESULT))

    def test_scope_to_a_run_and_a_task(self):
        bp = Breakpoint(phase="request", run="abc", task="tally/1")
        assert request_matches(bp, ctx(run_id="abcdef", task_id="tally/1"))
        assert not request_matches(bp, ctx(run_id="zzz", task_id="tally/1"))
        assert not request_matches(bp, ctx(run_id="abcdef", task_id="other"))

    def test_a_disabled_breakpoint_never_matches(self):
        assert not request_matches(Breakpoint(phase="request", enabled=False), ctx())


class TestResponseConditions:
    def test_tool_and_path_glob(self):
        bp = Breakpoint(phase="response", tool="write_file", path_glob="*.json")
        assert response_matches(bp, ctx(), write("report.json"))
        assert response_matches(bp, ctx(), write("/abs/dir/report.json"))  # basename match
        assert not response_matches(bp, ctx(), write("gen.py"))

    def test_tool_path_and_arguments_must_hold_for_the_same_call(self):
        """A read of a .json plus a write of a .py is not 'write_file on *.json'."""
        mixed = as_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "type": "function",
                     "function": {"name": "read_file", "arguments": '{"path": "a.json"}'}},
                    {"id": "2", "type": "function",
                     "function": {"name": "write_file", "arguments": '{"path": "b.py"}'}},
                ],
            }
        )
        bp = Breakpoint(phase="response", tool="write_file", path_glob="*.json")
        assert not response_matches(bp, ctx(), mixed)

    def test_arguments_regex_matches_values_as_they_read(self):
        """A JSON file's content must not have to be matched in its escaped form."""
        bp = Breakpoint(phase="response", tool="write_file", args_regex=r'"count"')
        assert response_matches(bp, ctx(), write("r.json", '{"count": 4}'))
        assert not response_matches(bp, ctx(), write("r.json", '{"row_count": 4}'))

    def test_arguments_regex_can_name_the_argument(self):
        """Arguments render as `key=value` lines, so a pattern can target one argument."""
        bp = Breakpoint(phase="response", args_regex=r"path=report\.json")
        assert response_matches(bp, ctx(), write("report.json"))
        assert not response_matches(bp, ctx(), write("other.json"))

    def test_output_regex_reads_the_reasoning(self):
        response = as_response(
            {"role": "assistant", "content": "ok", "reasoning_content": "I will ask the user"}
        )
        assert response_matches(Breakpoint(phase="response", output_regex="ask the user"), ctx(), response)

    def test_request_conditions_still_apply_in_the_response_phase(self):
        bp = Breakpoint(phase="response", tool="write_file", step_min=2)
        assert not response_matches(bp, ctx(step=0), write("r.json"))
        assert response_matches(bp, ctx(step=2), write("r.json"))

    def test_divergence_is_delegated(self):
        bp = Breakpoint(phase="response", diverges_from="golden")
        assert response_matches(bp, ctx(), write("r.json"), lambda run, step, resp: True)
        assert not response_matches(bp, ctx(), write("r.json"), lambda run, step, resp: False)


class TestEngine:
    async def test_a_matching_request_is_held_until_decided(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))

        hold = engine.check_request(ctx())
        assert hold is not None and hold.pending

        engine.resolve(hold.id, "continue")
        decision = await asyncio.wait_for(hold.future, 1)
        assert decision.action == "continue"
        assert not hold.pending

    async def test_nothing_is_held_without_a_match(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request", step_min=5))
        assert engine.check_request(ctx(step=0)) is None

    async def test_once_disables_after_the_first_hit(self):
        engine = BreakpointEngine()
        bp = engine.add(Breakpoint(phase="request", once=True))

        engine.resolve(engine.check_request(ctx()).id, "continue")

        assert bp.hits == 1 and not bp.enabled
        assert engine.check_request(ctx()) is None
        assert not engine.active

    async def test_step_from_after_a_call_stops_before_the_next(self):
        engine = BreakpointEngine()
        current = ctx([{"role": "user", "content": "go"}])
        engine.arm_next(current)
        assert engine.active

        following = ctx([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ])
        hold = engine.check_request(following)

        assert hold is not None and "stepped" in hold.reason
        assert not engine.active  # consumed

    async def test_unrelated_calls_do_not_trip_an_armed_step(self):
        engine = BreakpointEngine()
        engine.arm_next(ctx([{"role": "user", "content": "go"}]))
        assert engine.check_request(ctx([{"role": "user", "content": "elsewhere"}])) is None

    async def test_wants_response_only_for_breakpoints_that_could_fire(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="response", tool="write_file", step_min=3))
        assert not engine.wants_response(ctx(step=0))
        assert engine.wants_response(ctx(step=3))

    async def test_a_held_response_can_be_stepped_to(self):
        engine = BreakpointEngine()
        hold = engine.check_response(ctx(), write("r.json"), stepping=True)
        assert hold.phase == "response" and "stepped" in hold.reason

    async def test_edits_are_validated_against_the_phase(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        hold = engine.check_request(ctx())

        with pytest.raises(HoldError, match="no response to edit yet"):
            engine.resolve(hold.id, "continue", response=write("r.json"))
        with pytest.raises(HoldError, match="messages"):
            engine.resolve(hold.id, "continue", request={"model": "m"})
        engine.resolve(hold.id, "continue", request={"model": "m", "messages": []})

    async def test_tool_arguments_written_as_an_object_are_taken_as_meant(self):
        """Found in review: an object here crashed the held call after its release, and
        the model's response was never recorded."""
        engine = BreakpointEngine()
        hold = engine.check_response(ctx(), write("r.json"), stepping=True)
        edited = write("r.json")
        edited["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {
            "path": "r.json", "content": '{"row_count": 4}'}

        engine.resolve(hold.id, "continue", response=edited)

        decided = engine.get_hold(hold.id).decision.response
        arguments = decided["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, str)
        assert json.loads(arguments) == {"path": "r.json", "content": '{"row_count": 4}'}

    @pytest.mark.parametrize("arguments", [5, None, ["a"]])
    async def test_malformed_tool_arguments_leave_the_call_held(self, arguments):
        engine = BreakpointEngine()
        hold = engine.check_response(ctx(), write("r.json"), stepping=True)
        edited = write("r.json")
        edited["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
        with pytest.raises(HoldError, match="arguments"):
            engine.resolve(hold.id, "continue", response=edited)
        assert engine.get_hold(hold.id).pending

    async def test_a_held_response_cannot_have_its_request_edited(self):
        engine = BreakpointEngine()
        hold = engine.check_response(ctx(), write("r.json"), stepping=True)
        with pytest.raises(HoldError, match="edit the response instead"):
            engine.resolve(hold.id, "continue", request={"messages": []})
        with pytest.raises(HoldError, match="choices"):
            engine.resolve(hold.id, "continue", response={"nope": 1})

    async def test_a_hold_cannot_be_decided_twice(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        hold = engine.check_request(ctx())
        engine.resolve(hold.id, "drop")
        with pytest.raises(HoldError, match="already drop"):
            engine.resolve(hold.id, "continue")

    async def test_unknown_actions_are_refused(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        hold = engine.check_request(ctx())
        with pytest.raises(HoldError, match="action"):
            engine.resolve(hold.id, "teleport")

    async def test_abandoned_holds_leave_the_pending_list(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        hold = engine.check_request(ctx())
        engine.abandon(hold)
        assert engine.holds() == []
        assert engine.get_hold(hold.id).abandoned

    async def test_release_all_continues_everything(self):
        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        holds = [engine.check_request(ctx()) for _ in range(3)]
        assert engine.release_all() == 3
        assert all(hold.decision.action == "continue" for hold in holds)

    async def test_stopping_drops_a_cancelled_jobs_held_calls_before_continuing_the_rest(self):
        """Ctrl-C stops the jobs, then continues every call still held (`stop_live_work`).
        Found in review: a cancelled job's holds were dropped on the loop's next turn, after
        every hold had been continued — a fork killed at a breakpoint still sent its call to
        the model, and its answer was recorded into the interrupted run."""
        from dataclasses import replace
        from types import SimpleNamespace

        from tuningfork.proxy.local import LocalProxy

        engine = BreakpointEngine()
        engine.add(Breakpoint(phase="request"))
        forked = engine.check_request(replace(ctx(), launch_id="a-cancelled-fork"))
        person = engine.check_request(ctx([{"role": "user", "content": "a person's own harness"}]))
        proxy = LocalProxy(SimpleNamespace(state=SimpleNamespace(engine=engine)))
        proxy.attach(asyncio.get_running_loop(), "http://127.0.0.1:1")

        proxy.drop_holds_of(["a-cancelled-fork"])  # what stopping the jobs does, on the loop
        assert engine.release_all() == 1
        assert forked.decision.action == "drop" and person.decision.action == "continue"

    async def test_events_are_published(self):
        engine = BreakpointEngine(bus=EventBus())
        async with engine.bus.subscribe() as queue:
            engine.add(Breakpoint(phase="request"))
            hold = engine.check_request(ctx())
            engine.resolve(hold.id, "continue")
            kinds = [queue.get_nowait()["type"] for _ in range(queue.qsize())]
        assert kinds == ["breakpoints", "breakpoints", "hold", "resolved"]

    async def test_a_subscriber_too_far_behind_is_cut_off_not_misled(self):
        """Found in review: the oldest event was dropped silently — a lost "hold" left a
        call waiting that no page showed. Now the stream ends and the client resyncs."""
        bus = EventBus(maxsize=3)
        async with bus.subscribe() as slow, bus.subscribe() as fast:
            for index in range(4):
                bus.publish("step", n=index)
                if not fast.empty():
                    fast.get_nowait()
            assert slow.get_nowait() is OVERFLOWED
            assert slow.empty()
            assert bus.subscribers == 1  # the reader keeping up is unaffected


class TestEngineWithAStore:
    async def test_context_predicts_the_run_and_step(self, store):
        run_id = record_run(store, "r", [text_message("a")], task_id="tally/1")
        engine = BreakpointEngine(store)
        extended = [
            {"role": "user", "content": "do the task"},
            text_message("a"),
            {"role": "user", "content": "continue"},
        ]

        context = engine.context_for({"model": "m", "messages": extended})

        assert (context.run_id, context.step, context.task_id) == (run_id, 1, "tally/1")

    async def test_a_new_run_takes_the_task_of_its_pending_label(self, store):
        store.begin_run("batch#0", "tally/1")
        context = BreakpointEngine(store).context_for({"messages": [{"role": "user", "content": "x"}]})
        assert (context.run_id, context.task_id) == (None, "tally/1")

    async def test_run_references_are_resolved_when_added(self, store):
        run_id = record_run(store, "findable", [text_message("a")])
        engine = BreakpointEngine(store)
        bp = engine.add({"phase": "request", "run": "findable"})
        assert bp.run == run_id
        with pytest.raises(BreakpointError, match="no run"):
            engine.add({"phase": "request", "run": "missing"})

    async def test_divergence_from_a_golden_run(self, store):
        golden = record_run(store, "golden", [tool_call_message("read_file", {"path": "data.csv"})])
        engine = BreakpointEngine(store)
        engine.add({"phase": "response", "diverges_from": golden})

        same = as_response(tool_call_message("read_file", {"path": "data.csv"}, call_id="other"))
        different = as_response(tool_call_message("execute_code", {"code": "x"}))

        assert engine.check_response(ctx(step=0), same) is None
        assert engine.check_response(ctx(step=0), different) is not None


class TestEventBusClosing:
    async def test_closing_ends_every_subscription(self):
        from tuningfork.breakpoints import EventBus

        bus = EventBus()
        async with bus.subscribe() as queue:
            bus.publish("hold", hold={})
            bus.close()
            assert (await queue.get())["type"] == "hold"
            assert await queue.get() is None

    async def test_a_late_subscriber_to_a_closed_bus_ends_at_once(self):
        from tuningfork.breakpoints import EventBus

        bus = EventBus()
        bus.close()
        async with bus.subscribe() as queue:
            assert await queue.get() is None


async def test_a_redacted_message_does_not_throw_the_prediction_off(store):
    """Found in review: the recorder stores requests redacted, but the engine predicted from
    the raw request, so once any message was redacted, every later call of that run was
    predicted as a new run — and step-scoped breakpoints misfired."""
    from tuningfork.config import redact_json

    secret = "the key is sk-abcdefghijklmnopqrstuvwxyz"
    first = [{"role": "user", "content": secret}]
    store.record_step(request=redact_json({"model": "m", "messages": first}),
                      response=as_response(text_message("ok")))
    later = [*first, text_message("ok"), {"role": "user", "content": "go on"}]

    context = BreakpointEngine(store).context_for({"model": "m", "messages": later})

    assert context.run_id == store.runs()[0].id
    assert context.step == 1


def test_a_launch_scoped_breakpoint_stops_that_launch_alone():
    """What a fork started from the browser arms: its own step, and no one else's."""
    bp = Breakpoint.from_dict({"phase": "request", "launch": "f00d", "step_min": 2, "step_max": 2, "once": True})
    mine = ctx(step=2)
    mine.launch_id = "f00d"
    other = ctx(step=2)
    other.launch_id = "beef"
    assert request_matches(bp, mine)
    assert not request_matches(bp, other)
    assert not request_matches(bp, ctx(step=2))  # a call outside any launch
    assert "launch f00d" in bp.describe()
