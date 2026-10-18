"""Approving a validated edit and writing it back (`ops.approve`): in place, with the file's
line endings and byte-order mark, the old file and a changelog beside the skill folder, the
evidence in the refinement — and refused, with nothing written, whenever what was shown is
no longer what would be written. An approval a crash interrupted is reported and repaired."""

from __future__ import annotations

import json
import shutil

import pytest

from tuningfork.ops import approve, refine
from tuningfork.ops.approve import execute_approve, prepare_approve, render
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tests.helpers import task_stack

RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEYS = {"old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total."}
BOM = b"\xef\xbb\xbf"


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        built.skill = tmp_path / "skills-src" / "tally-report" / "SKILL.md"
        yield built


def verdicts(passed: int, failed: int) -> list[dict]:
    one = {"passed": True, "exclusion": None, "timed_out": False, "server_id": "s1"}
    return [one] * passed + [{**one, "passed": False}] * failed


def validated(stack, *, held_out: tuple[int, int] = (5, 0)) -> tuple:
    """A run of the task, a refinement at its write step, the keys fix proposed, and a
    finished validation of it — meeting the gate unless told otherwise."""
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1)
    execute_batch(stack.store, plan)
    (run,) = stack.store.runs(limit=5, task_id="tally-report/1")
    made = refine.create(stack.store, run.id, "write_file", target_rules=RULES)
    proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="the check reads row_count",
                                   proposer={"via": "person"})
    record = {
        "id": "e" * 32, "refinement": made.id, "proposal": proposal.id, "created_at": 1.0, "task": "tally-report/1",
        "alpha": 0.05, "minimum": 5, "planned": {"forward": 5, "training": 5, "held_out": 5}, "rules": [],
        "screen": {"survived": True, "screen": "a" * 32}, "held_out_recorded": [],
        "forward": {"original": verdicts(0, 5), "edit": verdicts(5, 0)},
        "training": {"original": verdicts(0, 5), "edit": verdicts(5, 0)},
        "held_out": [{"eval": "2", "original": verdicts(5, 0), "edit": verdicts(*held_out)}],
        "complete": True, "status": "done",
    }
    (made.folder / "validations").mkdir()
    (made.folder / "validations" / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")
    return made, proposal


def refused(call, *args, **kwargs) -> OperationError:
    with pytest.raises(OperationError) as error:
        call(*args, **kwargs)
    return error.value


def beside(stack) -> list[str]:
    """What sits beside the skill folder, and in it."""
    root = stack.skill.parent.parent
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_approved_the_edit_is_written_in_place_with_its_history_and_changelog_beside(stack):
    made, proposal = validated(stack)
    before = stack.skill.read_bytes()
    plan = prepare_approve(stack.store, made.id[:8], proposal.id[:8])
    shown = plan.to_json()
    assert shown["target"]["path"] == str(stack.skill) and "windows" in shown["target"]
    assert shown["target"]["history"].endswith("skills-src/tally-report.history")
    assert shown["target"]["changelog"].endswith("skills-src/tally-report.changelog.md")
    assert shown["gate"]["passed"] and len(plan.code) == 8 and shown["line_endings"] == "LF"
    changed = [line["op"] + line["text"] for hunk in shown["diff"]["hunks"] for line in hunk["lines"] if line["op"] != " "]
    assert changed == ["-" + KEYS["old"], "+" + KEYS["new"]]
    assert stack.skill.read_bytes() == before  # shown, not written

    result = execute_approve(stack.store, plan, confirm=plan.code.upper(), front_end="cli").to_json()
    assert stack.skill.read_text(encoding="utf-8") == proposal.text()
    # The skill folder holds what it held; the rest is beside it.
    (backup,) = (stack.skill.parent.parent / "tally-report.history").glob("SKILL.*.md")
    assert backup.read_bytes() == before and result["backup"] == str(backup)
    assert [name for name in beside(stack) if name.startswith("tally-report/")] == ["tally-report/SKILL.md"]
    changelog = (stack.skill.parent.parent / "tally-report.changelog.md").read_text(encoding="utf-8")
    assert f"+{KEYS['new']}" in changelog and f"-{KEYS['old']}" in changelog
    assert "the check reads row_count" in changelog and "G5" in changelog and "low-powered" in changelog
    # The approval carries its evidence, whatever is purged later.
    approval = json.loads((made.folder / "approval.json").read_text(encoding="utf-8"))
    assert approval["approver"]["front_end"] == "cli" and approval["evidence"]["gate"]["passed"]
    assert [each["id"] for each in approval["evidence"]["validations"]] == ["e" * 32]
    assert approval["evidence"]["proposal"]["proposers"][0]["rationale"] == "the check reads row_count"
    assert approval["intent"]["after"]["text"] == proposal.text()
    shown = refine.view(stack.store, made.id)
    assert shown["refinement"]["status"] == "approved" and shown["approval"]["state"] == "applied"
    # Judged again now it would fail (the file holds the edit): shown with the gate it met.
    (approved,) = refine.view(stack.store, made.id, held_out=True)["validations"]
    assert approved["approved"] and approved["gate"]["passed"]
    assert refused(prepare_approve, stack.store, made.id, proposal.id).code == "approved"
    assert refine.close(stack.store, made.id).status == "approved"


def test_line_endings_and_a_byte_order_mark_are_kept(stack):
    stack.skill.write_bytes(BOM + stack.skill.read_bytes().replace(b"\n", b"\r\n"))
    made, proposal = validated(stack)
    before = stack.skill.read_bytes()
    plan = prepare_approve(stack.store, made.id, proposal.id)
    assert plan.to_json()["line_endings"] == "CRLF" and plan.to_json()["bom"]
    execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    after = stack.skill.read_bytes()
    assert after.startswith(BOM) and after.count(BOM) == 1
    assert b"\r\n" in after and b"\n" not in after.replace(b"\r\n", b"")
    assert KEYS["new"].encode() in after
    (backup,) = (stack.skill.parent.parent / "tally-report.history").glob("SKILL.*.md")
    assert backup.read_bytes() == before


def test_rendering_follows_the_file_not_the_proposal():
    assert render("a\nb\n", b"x\r\ny\r\n") == b"a\r\nb\r\n"
    assert render("a\r\nb\n", b"x\ny\n") == b"a\nb\n"
    assert render("﻿a\n", b"x\n") == b"a\n"
    assert render("a\n", BOM + b"x\n") == BOM + b"a\n"


def test_a_wrong_code_or_a_file_changed_since_it_was_shown_writes_nothing(stack):
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    assert refused(execute_approve, stack.store, plan, confirm="deadbeef", front_end="cli").code == "confirm"
    stack.skill.write_text(stack.skill.read_text(encoding="utf-8") + "\nA line someone added.\n", encoding="utf-8")
    edited = stack.skill.read_bytes()
    assert refused(execute_approve, stack.store, plan, confirm=plan.code, front_end="cli").code == "drifted"
    assert stack.skill.read_bytes() == edited and not (made.folder / "approval.json").exists()
    assert list((stack.skill.parent.parent / "tally-report.history").glob("SKILL.*.md")) == []
    gate = refused(prepare_approve, stack.store, made.id, proposal.id)
    assert gate.code == "gate" and "G6" in gate.message


def test_the_gate_must_be_met(stack):
    made, proposal = validated(stack, held_out=(0, 5))
    found = refused(prepare_approve, stack.store, made.id, proposal.id)
    assert found.code == "gate" and "G5" in found.message and "significantly worse" in found.message


def test_another_checkout_or_a_moved_spec_is_refused_unless_named(stack):
    made, proposal = validated(stack)
    clone = stack.tmp / "clone"
    shutil.copytree(stack.tmp / "skills-src", clone / "skills-src")
    shutil.copyfile(stack.spec.path, clone / "spec.json")
    # This checkout moved on; the other still holds the skill as refined.
    stack.skill.write_text(stack.skill.read_text(encoding="utf-8") + "\nMoved on.\n", encoding="utf-8")
    moved_on = stack.skill.read_bytes()
    assert "G6" in refused(prepare_approve, stack.store, made.id, proposal.id).message
    plan = prepare_approve(stack.store, made.id, proposal.id, spec=str(clone / "spec.json"))
    assert plan.target.path == clone / "skills-src" / "tally-report" / "SKILL.md"
    execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    assert KEYS["new"] in plan.target.path.read_text(encoding="utf-8") and stack.skill.read_bytes() == moved_on


def test_a_spec_moved_or_leading_elsewhere_is_named_by_the_person(stack):
    made, proposal = validated(stack)
    moved = stack.tmp / "moved.json"
    stack.spec.path.rename(moved)
    found = refused(prepare_approve, stack.store, made.id, proposal.id)
    assert found.code == "moved" and found.field == "spec" and "--spec" in found.hint
    assert prepare_approve(stack.store, made.id, proposal.id, spec=str(moved)).target.path == stack.skill
    # The spec now leads to another folder: not the file the refinement began from.
    other = stack.tmp / "elsewhere" / "tally-report"
    shutil.copytree(stack.skill.parent, other)
    data = json.loads(moved.read_text(encoding="utf-8"))
    data["tuningfork"]["skill_dir"] = "elsewhere/tally-report"
    stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
    found = refused(prepare_approve, stack.store, made.id, proposal.id)
    assert found.code == "moved" and found.field == "target"
    assert prepare_approve(stack.store, made.id, proposal.id, target=str(stack.skill)).target.path == stack.skill
    assert refused(prepare_approve, stack.store, made.id, proposal.id, target=str(stack.tmp / "spec.json")).field == "target"


def test_a_second_approval_of_the_same_file_waits_its_turn(stack):
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    with approve._Lock(plan.target):
        assert refused(execute_approve, stack.store, plan, confirm=plan.code, front_end="cli").code == "locked"
    assert not (made.folder / "approval.json").exists()
    execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    assert refused(execute_approve, stack.store, plan, confirm=plan.code, front_end="ui").code == "approved"


def test_a_file_that_cannot_be_written_is_refused_and_nothing_is_kept(stack, monkeypatch):
    made, proposal = validated(stack)
    before = stack.skill.read_bytes()
    plan = prepare_approve(stack.store, made.id, proposal.id)

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(approve.os, "replace", denied)
    found = refused(execute_approve, stack.store, plan, confirm=plan.code, front_end="cli")
    assert found.code == "permission" and "Nothing was changed" in found.hint
    assert stack.skill.read_bytes() == before and not (made.folder / "approval.json").exists()
    assert list((stack.skill.parent.parent / "tally-report.history").glob("SKILL.*.md")) == []


def test_interrupted_after_the_file_was_written_it_is_finished(stack, monkeypatch):
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    real = approve._append_changelog
    monkeypatch.setattr(approve, "_append_changelog", lambda *args: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError):
        execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    assert approve.state(stack.store, made.id)["state"] == "written"
    assert refine.view(stack.store, made.id)["approval"]["state"] == "written"
    monkeypatch.setattr(approve, "_append_changelog", real)
    assert approve.repair(stack.store, made.id)["state"] == "applied"
    assert approve.repair(stack.store, made.id)["state"] == "applied"  # twice is once
    changelog = (stack.skill.parent.parent / "tally-report.changelog.md").read_text(encoding="utf-8")
    assert changelog.count("<!-- tuningfork approval") == 1
    assert refine.get(stack.store, made.id).status == "approved"


def test_interrupted_between_its_last_two_marks_it_is_reported_and_marked(stack, monkeypatch):
    # Applied, the refinement not yet marked approved: found in review, it read as applied, so
    # nothing said so and repair did nothing, and the refinement stayed open for good.
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    real = refine._replace

    def killed_at_the_mark(path, data):
        if path.name == "status.json":
            raise KeyboardInterrupt()
        real(path, data)

    monkeypatch.setattr(refine, "_replace", killed_at_the_mark)
    with pytest.raises(KeyboardInterrupt):
        execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    monkeypatch.setattr(refine, "_replace", real)
    assert (made.folder / "applied.json").is_file() and refine.get(stack.store, made.id).status == "open"
    assert approve.state(stack.store, made.id)["state"] == "unmarked"
    assert refine.view(stack.store, made.id)["approval"]["state"] == "unmarked"
    assert approve.repair(stack.store, made.id)["state"] == "applied"
    assert refine.get(stack.store, made.id).status == "approved"
    changelog = (stack.skill.parent.parent / "tally-report.changelog.md").read_text(encoding="utf-8")
    assert changelog.count("<!-- tuningfork approval") == 1


def test_interrupted_before_the_file_was_written_it_is_written_as_approved(stack, monkeypatch):
    made, proposal = validated(stack)
    before = stack.skill.read_bytes()
    plan = prepare_approve(stack.store, made.id, proposal.id)
    real = approve._replace_file
    monkeypatch.setattr(approve, "_replace_file", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    assert approve.state(stack.store, made.id)["state"] == "not_written" and stack.skill.read_bytes() == before
    monkeypatch.setattr(approve, "_replace_file", real)
    assert approve.repair(stack.store, made.id)["state"] == "applied"
    assert stack.skill.read_text(encoding="utf-8") == proposal.text()
    (backup,) = (stack.skill.parent.parent / "tally-report.history").glob("SKILL.*.md")
    assert backup.read_bytes() == before


def test_an_interrupted_approval_whose_file_was_changed_since_is_left_alone(stack, monkeypatch):
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    monkeypatch.setattr(approve, "_replace_file", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
    stack.skill.write_text("# someone's own rewrite\n", encoding="utf-8")
    assert approve.state(stack.store, made.id)["state"] == "diverged"
    assert refused(approve.repair, stack.store, made.id).code == "drifted"
    assert stack.skill.read_text(encoding="utf-8") == "# someone's own rewrite\n"


def test_a_refused_code_never_tells_the_right_one(stack):
    made, proposal = validated(stack)
    plan = prepare_approve(stack.store, made.id, proposal.id)
    found = refused(execute_approve, stack.store, plan, confirm="00000000", front_end="ui")
    assert plan.code not in found.message + (found.hint or "") + json.dumps(found.to_json())
