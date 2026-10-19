"""A run as one conversation: its last context read once, each step where it happened.

The timeline rests on one promise run grouping keeps — within a run, each step's context
extends the one before — and on one a wire format keeps: its view converts each message on
its own. Both are pinned here, with each thing that bends the plain picture: an answer the
harness sent back without its reasoning, a retry, a call refused while paused, an edit at a
breakpoint, a fork's variant, a replayed step, a subagent.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.launches import LaunchThreads, request_path
from tuningfork.ops.timeline import run_timeline, step_detail
from tuningfork.wire import get_wire
from tests.helpers import as_response, record_pilot, text_message, tool_call_message
from tests.test_wire import anthropic_message, claude_code_request

ANTHROPIC = get_wire("anthropic-messages")


def chat(*messages, **extras) -> dict:
    return {"model": "m", **extras, "messages": list(messages)}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def origins(found: dict) -> list[tuple]:
    return [(entry["index"], entry["origin"], entry["after"]) for entry in found["entries"]]


def test_a_run_reads_as_one_conversation_with_each_step_in_its_place(store):
    found = run_timeline(store, record_pilot(store)).to_json()
    assert [entry["message"]["role"] for entry in found["entries"]] == ["system", "user", "assistant", "tool"]
    # The context the run began with; the harness's copy of step 0's answer; what came after.
    assert origins(found) == [(0, "start", None), (1, "start", None), (2, "copy", 0), (3, "harness", 0)]
    first, second = found["steps"]
    assert (first["position"], first["copy"], first["kept"]) == (2, 2, "same")
    assert (second["position"], second["last"], second["kept"]) == (4, True, None)
    assert found["wire"] == "openai-chat" and found["tools"] == ["read_file"]
    assert [span["message_index"] for span in found["skills"]] == [0]
    json.dumps(found)  # everything a page gets is JSON


def test_an_answer_sent_back_without_its_reasoning_keeps_it_here(store):
    """Hermes sends an earlier answer back without its reasoning (seen in real recordings):
    the timeline shows the answer the model gave — reasoning and all — and counts it kept."""
    given = tool_call_message("write_file", {"path": "r.json"}, call_id="c1")
    given["reasoning_content"] = "Four rows; the total is 61.5."
    sent_back = tool_call_message("write_file", {"path": "r.json"}, call_id="c1")
    sent_back["tool_calls"][0]["function"]["arguments"] = '{"path":"r.json"}'  # re-serialized
    first = store.record_step(request=chat(user("go")), response=as_response(given, "tool_calls"))
    store.record_step(
        request=chat(user("go"), sent_back, {"role": "tool", "tool_call_id": "c1", "content": "ok"}),
        response=as_response(text_message("Done.")),
    )

    found = run_timeline(store, first.run_id).to_json()

    step = found["steps"][0]
    assert step["output"]["reasoning_content"] == "Four rows; the total is 61.5."
    assert step["kept"] == "same"
    assert found["reply"] == "Done."


def test_claude_codes_turns_are_placed_in_the_views_positions(store):
    """Anthropic's `system` is a request key, and a tool result is split out of its user
    turn: positions count the view, not the wire format's messages."""
    whole = claude_code_request()
    opening = {**whole, "messages": whole["messages"][:2]}
    native = {
        "id": "msg_0", "type": "message", "role": "assistant", "model": "gemma-4-31b",
        "content": whole["messages"][2]["content"], "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    first = store.record_step(
        request=opening, response=ANTHROPIC.canonical_response(native),
        wire="anthropic-messages", normalize=ANTHROPIC.normalize_message,
    )
    store.record_step(
        request=whole, response=ANTHROPIC.canonical_response(anthropic_message()),
        wire="anthropic-messages", normalize=ANTHROPIC.normalize_message,
    )

    found = run_timeline(store, first.run_id).to_json()

    roles = [entry["message"]["role"] for entry in found["entries"]]
    assert roles == ["system", "user", "system", "assistant", "tool", "user"]
    assert [step["position"] for step in found["steps"]] == [3, 6]
    assert found["steps"][0]["copy"] == 3 and found["steps"][0]["kept"] == "same"
    assert found["steps"][1]["output"]["reasoning_content"] == "Four rows."
    assert found["tools"] == ["Skill", "Write"]
    # Offered whole, an Anthropic tool's `input_schema` as its parameters.
    assert found["tools_offered"] == [{"name": "Skill", "description": "", "parameters": {}},
                                      {"name": "Write", "description": "", "parameters": {}}]
    assert [span["message_index"] for span in found["skills"]] == [5]


def test_a_call_sent_again_from_the_same_place_left_no_answer(store):
    first = store.record_step(request=chat(user("go")), error="upstream_error: 502 Bad Gateway")
    store.record_step(request=chat(user("go")), response=as_response(text_message("hi")))

    steps = run_timeline(store, first.run_id).to_json()["steps"]

    assert [step["position"] for step in steps] == [1, 1]
    assert (steps[0]["kept"], steps[0]["retried_by"]) == ("retried", 1)
    assert steps[0]["outcome"]["kind"] == "error"


def test_a_call_refused_while_paused_leaves_its_index_unrecorded(store):
    launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
    threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)
    opening = chat(user("go"))
    first = store.record_step(
        request=opening, response=as_response(text_message("a")), assignment=threads.assign(request_path(opening)),
    )
    later = chat(user("go"), text_message("a"), user("more"))
    threads.assign(request_path(later))  # refused while paused: its index is used, nothing recorded
    store.record_step(request=later, response=as_response(text_message("b")), assignment=threads.assign(request_path(later)))

    found = run_timeline(store, first.run_id).to_json()

    assert [step["idx"] for step in found["steps"]] == [0, 2]
    assert found["missing"] == [1]


def test_a_context_edited_at_a_breakpoint_is_marked_where_the_model_saw_it(store):
    original, answer = chat(user("go")), text_message("done")
    first = store.record_step(
        request=chat(user("go, carefully")), harness_request=original, response=as_response(answer),
        edits={"request": {"fields": ["messages"], "messages": [0]}},
    )
    store.record_step(request=chat(user("go"), answer, user("next")), response=as_response(text_message("ok")))

    found = run_timeline(store, first.run_id).to_json()

    # The history is the harness's: the model's edited view is marked, one click away.
    assert found["entries"][0]["message"]["content"] == "go"
    assert found["entries"][0]["changed_for_model"] == [{"by": "breakpoint", "steps": [0]}]
    assert found["steps"][0]["changed_for_model"] == {"by": "breakpoint", "entries": [0], "added": 0, "fields": []}
    assert found["steps"][0]["edited"] == {"request": True, "response": None}
    assert step_detail(store, first.run_id, 0).to_json()["request_changed_by"] == "breakpoint"


def test_an_answer_edited_at_a_breakpoint_is_what_the_harness_kept(store):
    produced, delivered = text_message("count/total"), text_message("row_count/amount_total")
    first = store.record_step(
        request=chat(user("go")), response=as_response(produced), delivered_response=as_response(delivered),
        edits={"response": {"fields": ["content"]}},
    )
    store.record_step(request=chat(user("go"), delivered, user("next")), response=as_response(text_message("ok")))

    step = run_timeline(store, first.run_id).to_json()["steps"][0]

    assert step["kept"] == "edited"
    assert step["output"]["content"] == "count/total"
    assert step["delivered"]["content"] == "row_count/amount_total"
    assert step["edited"] == {"request": False, "response": "edited"}


def test_a_forks_variant_is_marked_where_the_model_saw_it(store):
    original = chat({"role": "system", "content": "Write count."}, user("go"))
    variant = chat({"role": "system", "content": "Write row_count."}, user("go"))
    first = store.record_step(
        request=variant, harness_request=original, response=as_response(text_message("ok")),
        substitution={"spans": 1, "variant_sha256": "v"},
    )

    found = run_timeline(store, first.run_id).to_json()

    step = found["steps"][0]
    assert step["variant"] and step["changed_for_model"]["by"] == "variant"
    assert step["changed_for_model"]["entries"] == [0]
    assert found["entries"][0]["message"]["content"] == "Write count."
    detail = step_detail(store, first.run_id, 0).to_json()
    assert detail["request_changed_by"] == "variant"
    assert detail["harness_view"][0]["content"] == "Write count."
    assert detail["view"][0]["content"] == "Write row_count."


def test_a_replayed_step_says_where_its_answer_came_from(store):
    replay = {"served_from": "s1", "source_run": "a" * 32, "source_idx": 0}
    first = store.record_step(request=chat(user("go")), response=as_response(text_message("a")), replay=replay)

    step = run_timeline(store, first.run_id).to_json()["steps"][0]

    assert step["served"] and step["replay"]["source_idx"] == 0


def test_a_subagent_is_placed_after_the_step_that_spawned_it(store):
    launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1")
    threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id)

    def call(request, arrived, response=None):
        return store.record_step(
            request=request, response=response, timings={"arrived_at": arrived},
            assignment=threads.assign(request_path(request)),
        )

    spawn = as_response(tool_call_message("Agent", {}, call_id="c1"), "tool_calls")
    main = call(chat(user("task")), 100.0, spawn)
    child = call(chat(user("the subagent's goal")), 105.0)
    call(chat(user("task"), spawn["choices"][0]["message"], {"role": "tool", "tool_call_id": "c1", "content": "done"}), 110.0)
    store.finalize_threads(launch.id, {"Agent"})

    found = run_timeline(store, main.run_id).to_json()

    (thread,) = found["threads"]
    assert (thread["run_id"], thread["role"], thread["after"], thread["placed"]) == (child.run_id, "subagent", 0, "spawn")


def test_tools_offered_that_change_between_steps_are_noted(store):
    read = {"type": "function", "function": {"name": "read", "description": "Read a file.",
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    first = store.record_step(request=chat(user("go"), tools=[read]), response=as_response(text_message("a")))
    store.record_step(
        request=chat(user("go"), text_message("a"), user("more"), tools=[read, tool("Agent")]),
        response=as_response(text_message("b")),
    )

    found = run_timeline(store, first.run_id).to_json()
    steps = found["steps"]

    assert steps[0]["extras_changed"] is None
    assert steps[1]["extras_changed"] == {"fields": ["tools"], "tools_added": ["Agent"], "tools_removed": []}
    # What the first call offered, whole: the model reads it, though no message holds it.
    assert found["tools_offered"] == [{"name": "read", "description": "Read a file.",
                                       "parameters": read["function"]["parameters"]}]


def test_a_note_the_harness_wrote_itself_is_told_from_a_persons_turn(store):
    """After a step, a user message is a person's next turn, or the harness's own: Hermes's
    nudge to go on after a cut reply is named, by the run's adapter — or, for a run recorded as
    it happened, by whichever adapter knows it — and a person's turn is not."""
    task = user("Summarize data.csv into report.json.")
    nudge = user("[System: Your previous response was truncated by the output length limit. Continue exactly "
                 "where you left off. Do not restart or repeat prior text. Finish the answer directly.]")
    thanks = user("Thanks. Now the same for march.csv.")
    first = store.record_step(request=chat(task), response=as_response(text_message("Writing the rep"), "length"))
    store.record_step(request=chat(task, text_message("Writing the rep"), nudge),
                      response=as_response(text_message("Done: report.json.")))
    store.record_step(request=chat(task, text_message("Writing the rep"), nudge, text_message("Done: report.json."), thanks),
                      response=as_response(text_message("On it.")))

    def notes() -> list[tuple[str, int, str | None]]:
        found = run_timeline(store, first.run_id).to_json()
        return [(entry["message"]["content"][:9], entry["after"], entry.get("note")) for entry in found["entries"]
                if entry["origin"] == "harness" and entry["message"]["role"] == "user"]

    assert store.get_run(first.run_id).harness is None
    assert notes() == [("[System: ", 0, "continue after a cut reply"), ("Thanks. N", 1, None)]
    store.annotate_run(first.run_id, harness="hermes")
    assert notes() == [("[System: ", 0, "continue after a cut reply"), ("Thanks. N", 1, None)]
    # A harness's own adapter alone names its notes; one tuningfork does not know names none.
    for harness in ("claude-code", "generic"):
        store.annotate_run(first.run_id, harness=harness)
        assert notes() == [("[System: ", 0, None), ("Thanks. N", 1, None)]


def test_the_conversation_a_first_call_continued_is_read_as_the_runs_own_turns_are(store):
    """A run recorded live may continue a conversation already going on, a session of many
    turns: they are told apart as the run's own are — a person's turn carries their own words,
    a note the harness wrote itself is named — so the page lists them before the run's first
    step, as the conversation it continued."""
    nudge = user("[System: Your previous response was truncated by the output length limit. Continue exactly "
                 "where you left off. Do not restart or repeat prior text. Finish the answer directly.]")
    earlier = [user("Process receipt A."), text_message("Filing r"), nudge, text_message("Filed A."), user("Now receipt B.")]
    first = store.record_step(request=chat(*earlier), response=as_response(text_message("Filed B.")))
    store.annotate_run(first.run_id, harness="hermes")
    found = run_timeline(store, first.run_id).to_json()
    assert [(entry["origin"], entry["message"]["role"], entry.get("note"), entry.get("words")) for entry in found["entries"]] == [
        ("start", "user", None, "Process receipt A."), ("start", "assistant", None, None),
        ("start", "user", "continue after a cut reply", None), ("start", "assistant", None, None),
        ("start", "user", None, "Now receipt B.")]
    assert found["run"]["recorded"] == "live" and [step["position"] for step in found["steps"]] == [5]

    # Claude Code's reminders beside a person's words are its own: the words are theirs alone.
    claude = store.record_step(request=claude_code_request(), wire="anthropic-messages", harness="claude-code",
                               response=as_response(tool_call_message("Write", {"file_path": "report.json", "content": "{}"})))
    said = [(entry.get("note"), entry.get("words")) for entry in run_timeline(store, claude.run_id).to_json()["entries"]
            if entry["message"]["role"] == "user"]
    assert said[0] == (None, "Use the tally-report skill on data.csv.") and said[-1] == ("skill text", None)


def test_an_ab_is_one_experiment_at_its_step(store):
    run_id = record_pilot(store)
    step = store.step(run_id, 1)
    for arm in ("original", "variant"):
        store.create_experiment(
            kind="ab", arm=arm, run_id=run_id, step_idx=1, step_id=step.id, request={},
            settings={"n": 3}, group_id="g1", status="done",
        )
    store.create_experiment(
        kind="resample", arm="original", run_id=run_id, step_idx=0, step_id=store.step(run_id, 0).id,
        request={}, settings={"n": 4}, status="done",
    )

    steps = run_timeline(store, run_id).to_json()["steps"]

    assert [e["kind"] for e in steps[0]["experiments"]] == ["resample"]
    assert [e["kind"] for e in steps[1]["experiments"]] == ["ab"]


def test_a_step_off_the_runs_path_is_placed_not_refused(store):
    run_id = record_pilot(store)
    elsewhere = store.record_step(request=chat(user("something else")))
    with store._transaction():
        store._conn.execute(
            "UPDATE steps SET request_node_id = ? WHERE run_id = ? AND idx = 0",
            (elsewhere.request_node_id, run_id),
        )

    steps = run_timeline(store, run_id).to_json()["steps"]

    assert steps[0]["off_path"] and not steps[1]["off_path"]


def test_each_message_is_read_once(store, monkeypatch):
    run_id = record_pilot(store)
    reads: list[str] = []
    real = store.blobs.get

    def counted(digest):
        reads.append(digest)
        return real(digest)

    monkeypatch.setattr(store.blobs, "get", counted)
    found = run_timeline(store, run_id).to_json()
    # 4 messages, 2 responses, the extras once, and the last response again for the reply.
    assert len(reads) <= len(found["entries"]) + len(found["steps"]) + 2


@pytest.mark.parametrize("wire", ["openai-chat", "anthropic-messages"])
def test_a_view_converts_each_message_on_its_own(wire):
    """What the timeline's positions rest on: a whole request's view is its extras' view
    followed by each message's own."""
    converter = get_wire(wire)
    request = claude_code_request() if wire == "anthropic-messages" else chat(
        {"role": "system", "content": "s"}, user("go"),
        tool_call_message("read", {"p": 1}), {"role": "tool", "tool_call_id": "call_1", "content": "x"},
    )
    head = converter.view({**{k: v for k, v in request.items() if k != "messages"}, "messages": []})
    pieces = [piece for message in request["messages"] for piece in converter.view({"messages": [message]})]
    assert head + pieces == converter.view(request)


def test_an_empty_run_has_an_empty_timeline(store):
    store.begin_run("nothing", "t/1")
    first = store.record_step(request=chat(user("go")))
    found = run_timeline(store, first.run_id).to_json()
    assert found["steps"][0]["position"] == 1 and found["reply"] is None


def test_a_run_not_counted_says_why_and_where_its_checks_lines_came_from(store):
    """The run's verdict closes its page: why the rates leave it out, and — where the check was
    run along the run — the call that made each line it printed."""
    run_id = record_pilot(store)
    counted = run_timeline(store, run_id).to_json()
    assert counted["run"]["exclusion"] is None and counted["made_by"] is None

    along = {"steps": 2, "states": [
        {"before_step": 0, "end": False, "findings": ["report.json: no total"]},
        {"before_step": 2, "end": True, "findings": ["report.json: no total", "row_count: wrong"]},
    ]}
    store.annotate_run(run_id, result={"passed": False, "usable": False, "timed_out": True, "check_along": along})
    found = run_timeline(store, run_id).to_json()

    assert found["run"]["exclusion"] == "timed out"
    assert [(line["line"], line["made_by"], line["at_end"]) for line in found["made_by"]] == [
        ("report.json: no total", None, True),  # there from the start: work not done
        ("row_count: wrong", 1, True),  # made by step 1's call
    ]
