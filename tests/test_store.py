"""The message tree and run grouping.

Two properties carry the design. Messages are stored once and requests rebuild losslessly
from the tree — that is what makes recording a long agent context affordable. And requests
are grouped into runs by their path, with an explicit boundary available for the case the
tree genuinely cannot resolve.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.store.tree import (
    exact_hash,
    mask_volatile,
    node_id,
    node_path,
    normal_hash,
)
from tests.helpers import as_response, record_run, tool_call_message


def request_with(messages: list[dict], **extra) -> dict:
    return {"model": "m", "messages": messages, **extra}


SYSTEM = {"role": "system", "content": "You are helpful."}


class TestHashing:
    def test_exact_hash_is_order_independent_for_keys(self):
        assert exact_hash({"role": "user", "content": "hi"}) == exact_hash(
            {"content": "hi", "role": "user"}
        )

    def test_exact_hash_separates_different_content(self):
        assert exact_hash({"role": "user", "content": "a"}) != exact_hash(
            {"role": "user", "content": "b"}
        )

    def test_normal_hash_ignores_volatile_tool_call_ids(self):
        """The Phase 0 finding: ids are a fresh number per call on real servers."""
        first = tool_call_message("write_file", {"path": "a.py"}, call_id="385339560")
        second = tool_call_message("write_file", {"path": "a.py"}, call_id="990114723")

        assert exact_hash(first) != exact_hash(second)
        assert normal_hash(first) == normal_hash(second)

    def test_normal_hash_still_separates_real_differences(self):
        first = tool_call_message("write_file", {"path": "a.py"}, call_id="1")
        second = tool_call_message("write_file", {"path": "b.py"}, call_id="1")
        assert normal_hash(first) != normal_hash(second)

    def test_masking_leaves_content_alone(self):
        masked = mask_volatile({"id": "abc", "content": "keep me", "nested": {"id": "x"}})
        assert masked["content"] == "keep me"
        assert masked["id"] == "<masked>"
        assert masked["nested"]["id"] == "<masked>"

    def test_same_message_after_different_history_is_a_different_node(self):
        shared = {"role": "user", "content": "go"}
        a = node_path([{"role": "system", "content": "A"}, shared])
        b = node_path([{"role": "system", "content": "B"}, shared])
        assert a[0] != b[0]
        assert a[1] != b[1]  # position matters, not just content

    def test_node_path_is_a_chain(self):
        messages = [SYSTEM, {"role": "user", "content": "go"}]
        path = node_path(messages)
        assert path[0] == node_id(None, exact_hash(SYSTEM))
        assert path[1] == node_id(path[0], exact_hash(messages[1]))


class TestStorage:
    def test_shared_context_is_stored_once(self, store):
        """The reason recording a 13k-token context per step is affordable."""
        store.record_step(request=request_with([SYSTEM, {"role": "user", "content": "one"}]))
        store.begin_run("second")
        store.record_step(request=request_with([SYSTEM, {"role": "user", "content": "two"}]))

        # One shared system node plus two distinct user nodes.
        assert store.stats()["nodes"] == 3

    def test_request_rebuilds_exactly(self, store):
        request = request_with(
            [SYSTEM, {"role": "user", "content": "go"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            temperature=0.4,
            stream=True,
        )
        step = store.record_step(request=request)

        assert store.rebuild_request(step) == request

    def test_request_rebuilds_byte_for_byte_key_order_included(self, store):
        """Found in review: every object came back with sorted keys. A server renders tool
        schemas and past tool-call arguments in the order it receives them, so a resample
        was sampling a different prompt from the one recorded."""
        write = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "Write", "arguments": '{"file_path": "r.json", "content": "{}"}'},
        }]}
        request = {
            "model": "m",
            "stream": True,
            "messages": [SYSTEM, {"role": "user", "content": "go"}, write],
            "tools": [{"type": "function", "function": {"name": "Edit", "parameters": {
                "type": "object",
                "properties": {"file_path": {}, "old_string": {}, "new_string": {}},
            }}}],
            "max_tokens": 32000,
        }
        step = store.record_step(request=request)

        assert json.dumps(store.rebuild_request(step)) == json.dumps(request)

    def test_the_harness_view_keeps_the_harness_encoding(self, store):
        """An edit before the model: the harness's own messages come back as it sent them,
        cache markers included, not in the tree's stable form."""

        def stable(message: dict) -> dict:
            return {k: v for k, v in message.items() if k != "cache_control"}

        sent = request_with([SYSTEM, {"role": "user", "content": "go", "cache_control": "x"}])
        edited = request_with([SYSTEM, {"role": "user", "content": "go, carefully"}])
        step = store.record_step(
            request=edited, harness_request=sent, normalize=stable,
            edits={"request": {"messages": [1]}},
        )

        assert store.harness_request(step) == sent
        assert store.rebuild_request(step) == edited

    def test_messages_for_node_returns_them_in_order(self, store):
        messages = [SYSTEM, {"role": "user", "content": "go"}]
        step = store.record_step(request=request_with(messages))
        assert store.messages_for_node(step.request_node_id) == messages

    def test_reasoning_is_measured_from_the_text(self, store):
        """llama.cpp reports no reasoning_tokens, so the counter comes from the content."""
        step = store.record_step(
            request=request_with([{"role": "user", "content": "go"}]),
            response={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning_content": "twelve chars",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        assert step.reasoning_chars == len("twelve chars")


class TestRunGrouping:
    def test_extending_a_path_continues_the_run(self, store):
        first = [SYSTEM, {"role": "user", "content": "go"}]
        store.record_step(request=request_with(first))
        store.record_step(
            request=request_with([*first, {"role": "assistant", "content": "ok"}])
        )

        runs = store.runs()
        assert len(runs) == 1
        assert runs[0].step_count == 2

    def test_unrelated_history_starts_a_new_run(self, store):
        store.record_step(request=request_with([{"role": "user", "content": "task one"}]))
        store.record_step(request=request_with([{"role": "user", "content": "task two"}]))

        assert len(store.runs()) == 2

    def test_branching_from_a_known_root_is_recorded_as_a_fork(self, store):
        base = [SYSTEM, {"role": "user", "content": "go"}]
        store.record_step(request=request_with([*base, {"role": "assistant", "content": "A"}]))
        store.record_step(request=request_with([*base, {"role": "assistant", "content": "B"}]))

        runs = sorted(store.runs(), key=lambda run: run.started_at)
        assert len(runs) == 2
        assert runs[1].forked_from_run == runs[0].id
        assert runs[1].forked_at_depth == 2

    def test_a_branch_is_marked_from_the_run_it_shares_most_with_where_the_two_part(self, store):
        """A new run is marked a branch of the run whose history holds the most of its own — at
        the message where the two part, not at its own length — and two conversations sharing
        only their system prompt, as every session of one harness may, are no branch at all."""
        def say(text: str, role: str = "user") -> dict:
            return {"role": role, "content": text}

        opening = [SYSTEM, say("go"), say("a1", "assistant")]
        a = store.record_step(request=request_with([*opening, say("u1")])).run_id
        b = store.record_step(request=request_with([*opening, say("u2")])).run_id
        assert store.record_step(request=request_with([*opening, say("u2"), say("b1", "assistant"), say("u3")])).run_id == b
        c = store.record_step(request=request_with([*opening, say("u2"), say("b1", "assistant"), say("u4"),
                                                    say("c1", "assistant"), say("u5")])).run_id
        other = store.record_step(request=request_with([SYSTEM, say("something else")])).run_id

        marked = {run: (store.get_run(run).forked_from_run, store.get_run(run).forked_at_depth) for run in (a, b, c, other)}
        assert marked == {a: (None, None), b: (a, 3), c: (b, 5), other: (None, None)}

    def test_identical_requests_collapse_without_a_boundary(self, store):
        """Documents the limitation `begin_run` exists to solve."""
        request = request_with([{"role": "user", "content": "same"}])
        store.record_step(request=request)
        store.record_step(request=request)

        assert len(store.runs()) == 1

    def test_begin_run_separates_identical_requests(self, store):
        request = request_with([{"role": "user", "content": "same"}])
        store.begin_run("batch#0")
        store.record_step(request=request)
        store.begin_run("batch#1")
        store.record_step(request=request)

        runs = store.runs()
        assert len(runs) == 2
        assert {run.label for run in runs} == {"batch#0", "batch#1"}
        assert all(run.step_count == 1 for run in runs)


class TestLabelsAndAnnotation:
    def test_label_next_applies_to_the_next_run_only(self, store):
        store.label_next("the-one", task_id="t1")
        store.record_step(request=request_with([{"role": "user", "content": "first"}]))
        store.record_step(request=request_with([{"role": "user", "content": "second"}]))

        runs = sorted(store.runs(), key=lambda run: run.started_at)
        assert runs[0].label == "the-one"
        assert runs[0].task_id == "t1"
        assert runs[1].label is None

    def test_pending_labels_are_visible_before_use(self, store):
        store.label_next("queued")
        assert store.pending_labels() == ["queued"]

    def test_annotate_run_records_skill_hashes_and_parity(self, store):
        step = store.record_step(request=request_with([{"role": "user", "content": "go"}]))
        store.annotate_run(
            step.run_id,
            skill_hashes={"SKILL.md": "abc"},
            parity={"kv_cache": "q4_0", "n_ctx": 65536},
        )

        run = store.get_run(step.run_id)
        assert run.skill_hashes == {"SKILL.md": "abc"}
        assert run.parity["n_ctx"] == 65536

    def test_run_lookup_by_prefix_and_label(self, store):
        store.begin_run("findable")
        step = store.record_step(request=request_with([{"role": "user", "content": "go"}]))

        assert store.get_run("findable").id == step.run_id
        assert store.get_run(step.run_id[:8]).id == step.run_id
        assert store.get_run("nonexistent") is None


class TestBreakpointEdits:
    """At a breakpoint the model's view and the harness's view can differ."""

    def test_an_edited_request_does_not_split_the_run(self, store):
        """The harness never sees the edit, so its next request extends its own history."""
        original = request_with([SYSTEM, {"role": "user", "content": "go"}])
        edited = request_with([SYSTEM, {"role": "user", "content": "go, carefully"}])
        answer = {"role": "assistant", "content": "done"}

        first = store.record_step(
            request=edited,
            response=as_response(answer),
            harness_request=original,
            edits={"request": {"fields": ["messages"]}},
        )
        second = store.record_step(
            request=request_with([*original["messages"], answer, {"role": "user", "content": "next"}])
        )

        assert second.run_id == first.run_id
        assert second.idx == 1

    def test_the_model_view_and_the_harness_view_are_both_kept(self, store):
        original = request_with([{"role": "user", "content": "go"}])
        edited = request_with([{"role": "user", "content": "go, carefully"}])

        step = store.record_step(
            request=edited, harness_request=original, edits={"request": {"fields": ["messages"]}}
        )

        assert store.rebuild_request(step) == edited
        assert store.harness_request(step) == original
        assert step.edited

    def test_an_edited_response_keeps_the_models_output_as_the_behaviour(self, store):
        produced = as_response({"role": "assistant", "content": "count/total"})
        delivered = as_response({"role": "assistant", "content": "row_count/amount_total"})

        step = store.record_step(
            request=request_with([{"role": "user", "content": "go"}]),
            response=produced,
            delivered_response=delivered,
            edits={"response": {"fields": ["content"]}},
        )

        assert store.response(step) == produced
        assert store.delivered_response(step) == delivered

    def test_a_variant_swapped_into_the_extras_is_what_the_model_saw(self, store):
        """A fork's variant in an Anthropic `system` field: the step keeps the model's
        request exactly — key order included — as a resample resends it, and the harness's
        beside it. It once kept the harness's extras as the model's."""
        original = {"model": "m", "system": "Write count and total.", "messages": [{"role": "user", "content": "go"}], "stream": True}
        variant = {"model": "m", "system": "Write row_count and amount_total.", "messages": [{"role": "user", "content": "go"}], "stream": True}

        step = store.record_step(
            request=variant, harness_request=original, substitution={"spans": 1, "variant_sha256": "v"},
        )

        rebuilt = store.rebuild_request(step)
        assert json.dumps(rebuilt) == json.dumps(variant)
        assert store.harness_request(step)["system"] == "Write count and total."
        assert store.harness_extras(step) == {"model": "m", "system": "Write count and total.", "stream": True}
        assert not step.edited

    def test_an_unedited_step_delivers_what_the_model_produced(self, store):
        response = as_response({"role": "assistant", "content": "x"})
        step = store.record_step(
            request=request_with([{"role": "user", "content": "go"}]), response=response
        )
        assert store.delivered_response(step) == response
        assert store.harness_request(step) == store.rebuild_request(step)
        assert not step.edited


class TestPredictAttachment:
    def test_prediction_matches_where_the_step_is_then_recorded(self, store):
        first = request_with([SYSTEM, {"role": "user", "content": "go"}])
        step0 = store.record_step(request=first)
        second = request_with([*first["messages"], {"role": "assistant", "content": "ok"}])

        predicted = store.predict_attachment(second["messages"])
        step1 = store.record_step(request=second)

        assert predicted == (step0.run_id, 1)
        assert (step1.run_id, step1.idx) == predicted

    def test_an_unknown_request_would_start_a_new_run(self, store):
        assert store.predict_attachment([{"role": "user", "content": "new"}]) == (None, 0)

    def test_an_asserted_boundary_is_respected(self, store):
        request = request_with([{"role": "user", "content": "same"}])
        store.record_step(request=request)
        store.begin_run("next")
        assert store.predict_attachment(request["messages"]) == (None, 0)

    def test_prediction_records_nothing(self, store):
        store.predict_attachment([{"role": "user", "content": "just looking"}])
        assert store.stats()["runs"] == 0
        assert store.stats()["nodes"] == 0


def test_a_database_from_an_earlier_version_gains_the_new_columns(tmp_path):
    """CREATE TABLE IF NOT EXISTS never alters a table, so old databases need migrating."""
    import sqlite3

    from tuningfork.store.db import Store

    home = tmp_path / "old"
    home.mkdir()
    old = sqlite3.connect(home / "tuningfork.db")
    old.execute(
        "CREATE TABLE steps (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, idx INTEGER NOT NULL,"
        " request_node_id TEXT NOT NULL, extras_blob TEXT NOT NULL, response_blob TEXT,"
        " raw_response_blob TEXT, params TEXT, server_defaults TEXT, model TEXT,"
        " status INTEGER, finish_reason TEXT, timings TEXT, usage TEXT,"
        " reasoning_chars INTEGER NOT NULL DEFAULT 0, error TEXT, recorded_at REAL NOT NULL,"
        " UNIQUE(run_id, idx))"
    )
    old.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, root_node_id TEXT NOT NULL,"
        " head_node_id TEXT NOT NULL, label TEXT, task_id TEXT, harness TEXT, model TEXT,"
        " forked_from_run TEXT, forked_at_depth INTEGER, skill_hashes TEXT, parity TEXT,"
        " started_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    old.commit()
    old.close()

    with Store(home) as store:
        step = store.record_step(
            request=request_with([{"role": "user", "content": "go"}]),
            edits={"request": {"fields": ["messages"]}},
        )
        assert store.steps(step.run_id)[0].edits == {"request": {"fields": ["messages"]}}
        store.annotate_run(step.run_id, result={"passed": True, "usable": True})
        assert store.get_run(step.run_id).result == {"passed": True, "usable": True}


def test_an_older_samples_table_gains_the_forced_column(tmp_path):
    import sqlite3

    from tuningfork.store.db import Store

    home = tmp_path / "old"
    home.mkdir()
    old = sqlite3.connect(home / "tuningfork.db")
    old.execute(
        "CREATE TABLE samples (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, idx INTEGER NOT NULL,"
        " seed INTEGER, response_blob TEXT, outcome_kind TEXT, outcome_key TEXT, outcome_label TEXT,"
        " finish_reason TEXT, usage TEXT, timings TEXT, reasoning_chars INTEGER NOT NULL DEFAULT 0,"
        " error TEXT, created_at REAL NOT NULL, UNIQUE(experiment_id, idx))"
    )
    old.commit()
    old.close()

    with Store(home) as store:
        forced = {"position": 4, "id": 17, "text": " row", "p": 0.38, "original": False}
        store.add_sample("e", 0, seed=1, response=None, outcome_kind="text", outcome_key="k",
                         outcome_label="l", forced=forced)
        assert store.samples("e")[0].forced == forced


def test_an_older_jobs_table_gains_who_asked(tmp_path):
    import sqlite3

    from tuningfork.store.db import Store

    home = tmp_path / "old"
    home.mkdir()
    old = sqlite3.connect(home / "tuningfork.db")
    old.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, lane TEXT NOT NULL,"
        " status TEXT NOT NULL, params TEXT NOT NULL, plan TEXT, progress TEXT, result TEXT,"
        " error TEXT, links TEXT, owner TEXT, request_id TEXT, created_at REAL NOT NULL,"
        " started_at REAL, ended_at REAL)"
    )
    old.execute(
        "INSERT INTO jobs (id, kind, lane, status, params, created_at)"
        " VALUES ('a1', 'resample', 'background', 'done', '{}', 1)"
    )
    old.commit()
    old.close()

    with Store(home) as store:
        assert store.get_job("a1").started_by is None
        made = store.create_job(kind="resample", lane="background", params={}, started_by={"via": "mcp"})
        assert store.get_job(made.id).started_by == {"via": "mcp"}


class TestTokenStreams:
    STREAM = {"v": 1, "k": 2, "ids": [1, 2], "lp": [-0.1, -0.2], "top_ids": [[1], [2]],
              "top_lp": [[-0.1], [-0.2]], "pieces": {"1": "a", "2": "b"}, "parts": [], "redacted": []}

    def test_a_steps_streams_newest_first(self, store):
        step = store.record_step(request=request_with([{"role": "user", "content": "go"}]))
        first = store.add_token_stream(self.STREAM, source="scored", step_id=step.id, method="grammar",
                                       exact=True, notes={"rebuilt": 2})
        second = store.add_token_stream(self.STREAM, source="scored", step_id=step.id, method="grammar")
        assert [r.id for r in store.token_streams(step_id=step.id)] == [second.id, first.id]
        found = store.token_stream(first.id)
        assert (found.tokens, found.top_k, found.exact, found.notes) == (2, 2, True, {"rebuilt": 2})
        assert store.token_stream_data(found) == self.STREAM
        assert store.steps_with_token_streams([step.id])[step.id].id == second.id

    def test_a_samples_streams(self, store):
        record = store.add_token_stream(self.STREAM, source="sampled", sample_id="s1")
        assert [r.id for r in store.token_streams(sample_ids=["s1", "s2"])] == [record.id]
        assert store.token_streams(sample_ids=[]) == []
        # a sampled stream is not a step's own
        assert store.steps_with_token_streams(["s1"]) == {}


class TestRunLookup:
    """Ids are hex, so labels and id prefixes overlap; lookup must not guess between them."""

    def test_a_label_wins_over_another_runs_id_prefix(self, store):
        first = record_run(store, "first", [{"role": "assistant", "content": "x"}])
        second = record_run(store, "second", [{"role": "assistant", "content": "y"}])
        # Label the second run with a prefix of the first run's id.
        store.annotate_run(second, label=first[:3])

        assert store.get_run(first[:3]).id == second
        assert store.get_run(first).id == first

    def test_an_unambiguous_prefix_finds_its_run(self, store):
        run_id = record_run(store, "only", [{"role": "assistant", "content": "x"}])
        assert store.get_run(run_id[:6]).id == run_id

    def test_an_ambiguous_prefix_finds_nothing(self, store):
        ids = [
            record_run(store, f"r{index}", [{"role": "assistant", "content": str(index)}])
            for index in range(17)
        ]
        # Seventeen hex ids: at least two share a first character.
        shared = next(c for c in "0123456789abcdef" if sum(i.startswith(c) for i in ids) > 1)
        assert store.get_run(shared) is None

    def test_wildcards_in_a_reference_are_literal(self, store):
        record_run(store, "only", [{"role": "assistant", "content": "x"}])
        assert store.get_run("%") is None
        assert store.get_run("_") is None


class TestRetrieval:
    def test_steps_are_ordered_and_indexed(self, store):
        run_id = record_run(
            store,
            "ordered",
            [{"role": "assistant", "content": c} for c in ("a", "b", "c")],
        )
        steps = store.steps(run_id)
        assert [step.idx for step in steps] == [0, 1, 2]
        assert store.step(run_id, 1).id == steps[1].id

    def test_response_roundtrips(self, store):
        run_id = record_run(store, "resp", [tool_call_message("write_file", {"path": "x"})])
        response = store.response(store.steps(run_id)[0])
        call = response["choices"][0]["message"]["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"path": "x"}


def test_a_step_that_fails_part_way_leaves_nothing_behind(store, monkeypatch):
    """Found in review: a failure after the run was created left the transaction open, and
    the next write committed a run with no steps — holding the label meant for another."""
    store.begin_run("mine", "t/1")
    real = store.blobs.put_json
    calls = []

    def broken(value):
        calls.append(value)
        if len(calls) == 2:  # the step's extras: after its node, its run and its label
            raise OSError("disk full")
        return real(value)

    monkeypatch.setattr(store.blobs, "put_json", broken)
    with pytest.raises(OSError):
        store.record_step(request=request_with([{"role": "user", "content": "go"}]))
    monkeypatch.undo()
    store.label_next("unrelated")  # any later write commits

    assert store.runs() == []
    assert store.pending_labels() == ["mine", "unrelated"]
