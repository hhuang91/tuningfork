"""Refinements (`ops.refine`): one skill at one checkpoint toward a declared target, and the
edits proposed for it — on runs a real batch made (the fake agent, the mock model)."""

from __future__ import annotations

import json
import os

import pytest

from tuningfork.ops import refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.specs import hash_tree
from tests.helpers import task_stack

RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEYS = {"old": "Write the totals to report.json.", "new": "Write the totals to report.json, using the keys row_count and amount_total."}


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def a_run(stack, **options) -> str:
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1, **options)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id="tally-report/1")
    return run.id


def start(stack, run_id, step="write_file", **target):
    return refine.create(stack.store, run_id, step, **(target or {"target_rules": RULES}))


def test_a_record_written_once_is_there_whole_or_not_at_all(tmp_path):
    # Cut off half-written (a crash, a Ctrl-C, a value that will not serialize), a record read
    # as made. Found in review: a truncated approval.json refused every later approval, and
    # the refinement's page with it.
    path = tmp_path / "refinement" / "approval.json"
    with pytest.raises(TypeError):
        refine._write_once(path, {"a": "x" * 100_000, "z": object()})
    assert not path.exists() and (not path.parent.exists() or os.listdir(path.parent) == [])
    refine._write_once(path, {"a": 1})
    with pytest.raises(FileExistsError):
        refine._write_once(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1} and os.listdir(path.parent) == ["approval.json"]


class TestCreate:
    def test_a_refinement_holds_the_checkpoint_the_skill_and_its_source(self, stack):
        run_id = a_run(stack)
        made = start(stack, run_id)
        assert made.step_idx == 1 and made.skill == "tally-report" and made.status == "open"
        assert made.baseline_text() == (stack.tmp / "skills-src" / "tally-report" / "SKILL.md").read_text()
        assert made.data["source"]["path"].endswith("skills-src/tally-report/SKILL.md")
        assert made.data["source"]["matches"] is True
        assert made.target["kind"] == "rules" and made.data["created_by"] == {"via": "person"}
        assert (stack.store.home / "refinements" / made.id / "refinement.json").is_file()

    def test_the_same_checkpoint_and_target_give_back_the_open_one(self, stack):
        run_id = a_run(stack)
        first = start(stack, run_id)
        assert start(stack, run_id).id == first.id
        assert start(stack, run_id, target_match="row_count").id != first.id
        assert [each.id for each in refine.list_refinements(stack.store)] != []
        refine.close(stack.store, first.id[:8])
        assert first.id not in [each.id for each in refine.list_refinements(stack.store)]
        assert start(stack, run_id).id != first.id  # a closed refinement is not given back

    @pytest.mark.parametrize("target", [{}, {"target_match": "row_count", "target_rules": RULES}])
    def test_the_target_is_declared_once_before_anything_else(self, stack, target):
        run_id = a_run(stack)
        with pytest.raises(OperationError) as refused:
            refine.create(stack.store, run_id, "write_file", **target)
        assert refused.value.field == "target"

    def test_a_held_out_task_never_shapes_an_edit(self, stack):
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["held_out"] = True
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        run_id = a_run(stack, eval_id="1", include_held_out=True)  # recorded anyway, as a person may
        with pytest.raises(OperationError) as refused:
            start(stack, run_id)
        assert refused.value.code == "held_out"

    def test_a_step_without_the_skill_says_so(self, stack):
        run_id = a_run(stack)
        run = stack.store.get_run(run_id)
        frozen = refine.run_skill_file(stack.store, run)
        frozen.write_text("# a skill this run never saw\n", encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            start(stack, run_id)
        assert refused.value.code == "no_skill_here" and "no step of this run" in refused.value.hint


class TestProposals:
    def test_an_edit_is_saved_as_a_variant_with_who_proposed_it(self, stack):
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check wants these keys",
                                       proposer={"via": "mcp", "client": "claude-code"})
        assert proposal.data["changed"] == 1 and not proposal.duplicate
        text = proposal.text()
        assert "row_count and amount_total" in text and proposal.data["variant"].startswith(f"r{made.id[:8]}-")
        assert (stack.store.home / "variants" / "tally-report" / f"{proposal.data['variant']}.md").is_file()
        (entry,) = proposal.data["proposers"]
        assert entry["client"] == "claude-code" and entry["rationale"] == "the check wants these keys"

    def test_an_edit_by_line_number_is_resolved_to_the_text_it_names(self, stack):
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(
            stack.store, made.id, rationale="the check wants these keys", proposer={"via": "mcp"},
            edits=[{"line": 7, "new": KEYS["new"]}, {"after": 7, "insert": "Check the counts against the input."}],
        )
        assert "row_count and amount_total" in proposal.text() and "Check the counts" in proposal.text()
        (entry,) = proposal.data["proposers"]
        assert [(e["lines"], e["old"]) for e in entry["edits"]] == [("line 7", KEYS["old"]), ("after line 7", "")]
        assert entry["edits"][0]["given"] == {"line": 7, "new": KEYS["new"]}

    def test_the_same_text_twice_is_one_proposal_with_both_proposers(self, stack):
        made = start(stack, a_run(stack))
        first = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="a", proposer={"via": "mcp"})
        again = refine.add_proposal(stack.store, made.id, text=first.text(), rationale="b", proposer={"via": "person"},
                                    strict=False)
        assert again.duplicate and again.id == first.id
        assert [p["via"] for p in again.data["proposers"]] == ["mcp", "person"]
        assert len(refine.proposals(stack.store, made.id)) == 1

    def test_what_a_proposal_may_not_do(self, stack):
        made = start(stack, a_run(stack))
        baseline = made.baseline_text()

        def refused(**kwargs):
            kwargs.setdefault("rationale", "why")
            kwargs.setdefault("proposer", {"via": "mcp"})
            with pytest.raises(OperationError) as error:
                refine.add_proposal(stack.store, made.id, **kwargs)
            return error.value

        assert "frontmatter" in refused(edits=[{"old": "name: tally-report", "new": "name: tally"}]).message
        assert "frontmatter" in refused(edits=[{"line": 2, "new": "name: tally"}]).message
        assert "at most 3 edits" in refused(edits=[KEYS] * 4).message
        many = baseline + "".join(f"line {k}\n" for k in range(12))
        assert "12 lines changed" in refused(text=many).message
        assert refused(edits=[KEYS], rationale="  ").field == "rationale"
        assert refused(edits=[KEYS], text="x").field == "edits"
        # From a person, a long edit only warns.
        loose = refine.add_proposal(stack.store, made.id, text=many, rationale="mine", proposer={"via": "person"},
                                    strict=False)
        assert loose.data["warnings"]

    def test_nothing_is_written_in_the_skills_own_folder(self, stack):
        source = stack.tmp / "skills-src"
        before = hash_tree([source])
        made = start(stack, a_run(stack))
        refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "mcp"})
        refine.close(stack.store, made.id)
        assert hash_tree([source]) == before

    def test_a_closed_refinement_takes_no_more(self, stack):
        made = start(stack, a_run(stack))
        refine.close(stack.store, made.id)
        with pytest.raises(OperationError) as refused:
            refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "mcp"})
        assert refused.value.code == "closed"

    def test_the_view_shows_each_proposal_as_a_line_diff(self, stack):
        made = start(stack, a_run(stack))
        refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "mcp"})
        shown = refine.view(stack.store, made.id[:8])
        (proposal,) = shown["proposals"]
        changed = [line for hunk in proposal["diff"] for line in hunk["lines"] if line["op"] != " "]
        assert [line["op"] for line in changed] == ["-", "+"]
        assert shown["screens"] == [] and shown["refinement"]["id"] == made.id
