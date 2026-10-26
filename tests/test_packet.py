"""The reflection packet (`ops.packet`): what a proposer is shown, built from one task's
records, kept by its digest — on runs a real batch made (the fake agent, the mock model)."""

from __future__ import annotations

import json

import pytest

from tuningfork.config import REDACTED
from tuningfork.ops import packet as packets
from tuningfork.ops import refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tests.helpers import KEY, SKILL, VARIANT, task_stack
from tests.test_store_masking import PRIVATE

RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEYS = {"old": "Write the totals to report.json.", "new": "Write the totals to report.json, using the keys row_count and amount_total."}


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def runs(stack, n, skill_file=None, model="m"):
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model=model, harness="fake", n=n, skill_file=skill_file)
    execute_batch(stack.store, plan)
    return [run.id for run in stack.store.runs(limit=50, task_id="tally-report/1")]


def test_a_packet_shows_the_skill_the_step_and_the_tasks_own_criteria(stack):
    run_id = runs(stack, 1)[0]
    made = refine.create(stack.store, run_id, "write_file", target_rules=RULES)
    found = packets.build_packet(stack.store, made.id)
    data = found.data
    assert data["target"]["description"].startswith("a behaviour that calls write_file")
    assert data["skill"]["name"] == "tally-report" and data["skill"]["in_context"]
    assert data["step"]["answer"]["tool_calls"][0]["name"] == "write_file"
    # The run wrote count and total: no answer seen here reached the target, and it says so.
    assert data["answers_here"]["summary"] == "the one answer seen at this checkpoint did not reach the target"
    assert data["answers_here"]["not_reaching"][0]["source"] == "the recorded run"
    assert "row_count" in data["task"]["check"] and data["included"] == {"task_criteria": True}
    assert found.size <= packets.BUDGET


def test_the_same_evidence_gives_the_same_digest_and_one_file(stack):
    made = refine.create(stack.store, runs(stack, 1)[0], "write_file", target_rules=RULES)
    first, again = packets.build_packet(stack.store, made.id), packets.build_packet(stack.store, made.id)
    assert first.sha == again.sha and len(first.sha) == 16
    assert [path.stem for path in (made.folder / "packets").glob("*.json")] == [first.sha]
    assert packets.read_packet(stack.store, made.id, first.sha).data == first.data


def test_only_runs_like_this_one_are_compared(stack, tmp_path):
    """The same skill version, harness and model: a fixed wording's runs could quote it, and
    another model's are other evidence."""
    original = runs(stack, 2)
    variant = tmp_path / "variant.md"
    variant.write_text(VARIANT, encoding="utf-8")
    runs(stack, 1, skill_file=str(variant))  # a run of the fixed wording: never shown
    runs(stack, 1, model="another-model")  # nor one of another model
    made = refine.create(stack.store, original[-1], "write_file", target_rules=RULES)
    here = packets.build_packet(stack.store, made.id).data["runs_here"]
    assert here["summary"] == "2 runs of this task with this harness, model and skill version"


def test_without_the_task_criteria_when_asked(stack):
    made = refine.create(stack.store, runs(stack, 1)[0], "write_file", target_rules=RULES)
    data = packets.build_packet(stack.store, made.id, include_task=False).data
    assert "task" not in data and data["included"] == {"task_criteria": False}


def test_what_was_tried_before_is_in_the_next_packet(stack):
    made = refine.create(stack.store, runs(stack, 1)[0], "write_file", target_rules=RULES)
    refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads these keys",
                        proposer={"via": "mcp"})
    (tried,) = packets.build_packet(stack.store, made.id).data["tried_before"]
    assert tried["changed"][0].startswith("- Write the totals") and tried["screened"] is None
    assert tried["rationale"] == "the check reads these keys"


def test_what_a_proposer_is_shown_of_the_skill_and_the_task_is_masked_each_line_at_its_number(stack):
    """The run's skill is kept as it was given, and its task as its file wrote it: a proposer's model
    may be hosted, and is shown them masked — the skill whole before its lines are numbered, as
    show_skill numbers them (line by line, a private key's lines pass). Kept, the packet is as built."""
    keyed = SKILL + f"\nLog in with {KEY}.\n\n" + "\n".join(PRIVATE) + "\n\nThen write the report.\n"
    (stack.tmp / "skills-src" / "tally-report" / "SKILL.md").write_text(keyed, encoding="utf-8")
    stack.item.prompt += f" My key is {KEY}."
    stack.item.tuningfork.check = f": {KEY}; {stack.item.tuningfork.check}"
    stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
    made = refine.create(stack.store, runs(stack, 1)[0], "write_file", target_rules=RULES)
    found = packets.build_packet(stack.store, made.id)
    kept = (made.folder / "packets" / f"{found.sha}.json").read_text(encoding="utf-8")
    for said in (json.dumps(found.data), kept):
        assert KEY not in said and PRIVATE[1] not in said and PRIVATE[2] not in said
    lines, numbered = keyed.splitlines(), found.data["skill"]["lines"]
    at, after = lines.index(f"Log in with {KEY}.") + 1, lines.index("Then write the report.") + 1
    assert numbered[at - 1] == f"{at}: Log in with {REDACTED}." and numbered[after - 1] == f"{after}: Then write the report."
    assert REDACTED in found.data["task"]["prompt"] and REDACTED in found.data["task"]["check"]
