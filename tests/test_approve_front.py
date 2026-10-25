"""Validating and approving from a terminal and over HTTP: `refine validate`, `refine approve`
and `refine show`, and the approve route — which takes the access token and the code, and
never the MCP token."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from tuningfork.cli import app as cli
from tuningfork.ops import approve, validate
from tests.helpers import control_headers, mcp_headers
from tests.test_approve import KEYS, stack, validated  # noqa: F401 — the fixture
from tests.test_validate import MARKER, SetUp, screened
from tests.test_validate import stack as held_out_stack  # noqa: F401 — the fixture with a held-out eval

API = "/_tuningfork/api/refinements"


def invoke(home, *args: str, input: str | None = None):
    return CliRunner().invoke(cli, [*args, "--home", str(home)], input=input)


class TestValidateFromATerminal:
    def test_the_plan_names_every_part_and_no_held_out_text(self, held_out_stack):
        made, proposal = screened(held_out_stack)
        shown = invoke(held_out_stack.store.home, "refine", "validate", made.id[:8], proposal.id[:8],
                       "--proxy", held_out_stack.url, "--dry-run")
        assert shown.exit_code == 0, shown.output
        for line in ("survived screen", "forward   5 forks an arm", "training  5 full runs an arm of tally-report/1",
                     "held out  5 full runs an arm of each of: 2", "only their verdicts are kept", "30 in all"):
            assert line in shown.output, (line, shown.output)
        assert MARKER not in shown.output
        assert "  options   " not in shown.output  # the run was launched with none

    def test_the_plan_names_the_options_every_run_is_launched_with(self, held_out_stack, monkeypatch):
        from tuningfork.harness import ADAPTERS

        monkeypatch.setitem(ADAPTERS, "fake", SetUp)
        made, proposal = screened(held_out_stack, harness_options={"self_improvement": False})
        shown = invoke(held_out_stack.store.home, "refine", "validate", made.id[:8], proposal.id[:8],
                       "--proxy", held_out_stack.url, "--dry-run")
        assert shown.exit_code == 0, shown.output
        assert ("  options   every run launched with the harness options the checkpoint's run had: "
                '{"self_improvement": false}') in shown.output, shown.output

    def test_a_validation_prints_each_run_and_the_gate(self, held_out_stack, monkeypatch):
        monkeypatch.setattr(validate, "MINIMUM", 1)
        made, proposal = screened(held_out_stack)
        ran = invoke(held_out_stack.store.home, "refine", "validate", made.id[:8], proposal.id[:8],
                     "--proxy", held_out_stack.url, "-n", "1", "--runs", "1")
        assert ran.exit_code == 0, ran.output
        for line in ("forward — forks from step 1 to the end", "training — full runs of tally-report/1",
                     "held out 2 — verdicts only", "ended", "meets the gate", "G5 met", "low-powered",
                     f"tuningfork refine approve {made.id[:8]} {proposal.id[:8]}"):
            assert line in ran.output, (line, ran.output)
        assert MARKER not in ran.output
        forward = ran.output.split("training —")[0]
        assert "  edit " in forward and "variant" not in forward  # a validation's word for it, throughout
        shown = invoke(held_out_stack.store.home, "refine", "show", made.id[:8])
        assert "validations (1)" in shown.output and "held out 2: edit 1/1" in shown.output, shown.output

    def test_a_copy_of_the_private_runs_is_never_kept_among_the_recordings(self, held_out_stack):
        made, proposal = screened(held_out_stack)
        inside = held_out_stack.store.home / "kept"
        refused = invoke(held_out_stack.store.home, "refine", "validate", made.id[:8], proposal.id[:8],
                         "--proxy", held_out_stack.url, "--keep-private", str(inside), "--dry-run")
        assert refused.exit_code == 1 and "inside the recordings folder" in refused.output


class TestApproveFromATerminal:
    def test_shown_then_written_once_the_code_is_typed(self, stack):
        made, proposal = validated(stack)
        before = stack.skill.read_bytes()
        cancelled = invoke(stack.store.home, "refine", "approve", made.id[:8], proposal.id[:8], input="no\n")
        assert cancelled.exit_code == 1 and "nothing written" in cancelled.output
        assert stack.skill.read_bytes() == before
        code = approve.prepare_approve(stack.store, made.id, proposal.id).code
        for line in (f"writes     {stack.skill}", "LF line endings", "tally-report.history", "tally-report.changelog.md",
                     "every check met", "G6 met", f"- {KEYS['old']}", f"+ {KEYS['new']}", f"code  {code}"):
            assert line in cancelled.output, (line, cancelled.output)
        written = invoke(stack.store.home, "refine", "approve", made.id[:8], proposal.id[:8], input=f"{code}\n")
        assert written.exit_code == 0, written.output
        assert f"written: {stack.skill}" in written.output and "the file before" in written.output
        assert KEYS["new"] in stack.skill.read_text(encoding="utf-8")
        again = invoke(stack.store.home, "refine", "approve", made.id[:8], proposal.id[:8], "--confirm", code)
        assert again.exit_code == 1 and "approved edit already" in again.output
        shown = invoke(stack.store.home, "refine", "show", made.id[:8])
        assert "approval  proposal" in shown.output and "applied" in shown.output, shown.output

    def test_a_code_given_in_advance_must_be_this_approvals(self, stack):
        made, proposal = validated(stack)
        wrong = invoke(stack.store.home, "refine", "approve", made.id[:8], proposal.id[:8], "--confirm", "abcdef12")
        assert wrong.exit_code == 1 and "nothing written" in wrong.output
        code = approve.prepare_approve(stack.store, made.id, proposal.id).code
        right = invoke(stack.store.home, "refine", "approve", made.id[:8], proposal.id[:8], "--confirm", code)
        assert right.exit_code == 0, right.output

    def test_an_interrupted_approval_is_shown_and_repaired(self, stack, monkeypatch):
        made, proposal = validated(stack)
        plan = approve.prepare_approve(stack.store, made.id, proposal.id)
        real = approve._append_changelog
        monkeypatch.setattr(approve, "_append_changelog", lambda *args: (_ for _ in ()).throw(RuntimeError("killed")))
        with pytest.raises(RuntimeError):
            approve.execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
        monkeypatch.setattr(approve, "_append_changelog", real)
        shown = invoke(stack.store.home, "refine", "show", made.id[:8])
        assert "written" in shown.output and "--repair" in shown.output, shown.output
        repaired = invoke(stack.store.home, "refine", "show", made.id[:8], "--repair")
        assert repaired.exit_code == 0 and "approval applied" in repaired.output


class TestApproveOverHttp:
    def test_the_plan_is_read_and_the_approval_takes_the_token_and_the_code(self, stack):
        made, proposal = validated(stack)
        before = stack.skill.read_bytes()
        with httpx.Client(base_url=stack.url, timeout=60) as http:
            plan = http.get(f"{API}/{made.id}/approve", params={"proposal": proposal.id[:8]})
            assert plan.status_code == 200, plan.text
            code = plan.json()["code"]
            assert plan.json()["target"]["path"] == str(stack.skill) and plan.json()["gate"]["passed"]
            body = {"proposal": proposal.id, "confirm": code}
            assert http.post(f"{API}/{made.id}/approve", json=body).status_code == 401
            agent = http.post(f"{API}/{made.id}/approve", json=body, headers=mcp_headers(stack.app))
            assert agent.status_code == 403  # an MCP client's token approves nothing
            wrong = http.post(f"{API}/{made.id}/approve", json={**body, "confirm": "00000000"},
                              headers=control_headers(stack.app))
            assert wrong.status_code == 400 and wrong.json()["error"]["code"] == "confirm"
            assert code not in json.dumps(wrong.json())
            assert stack.skill.read_bytes() == before
            done = http.post(f"{API}/{made.id}/approve", json=body, headers=control_headers(stack.app))
            assert done.status_code == 200, done.text
            assert done.json()["state"] == "applied" and done.json()["target"] == str(stack.skill)
            again = http.post(f"{API}/{made.id}/approve", json=body, headers=control_headers(stack.app))
            assert again.status_code == 409 and again.json()["error"]["code"] == "approved"
            page = http.get(f"{API}/{made.id}").json()
        assert page["approval"]["state"] == "applied" and page["refinement"]["status"] == "approved"
        approval = json.loads((made.folder / "approval.json").read_text(encoding="utf-8"))
        assert approval["approver"]["front_end"] == "ui"

    def test_a_file_changed_after_the_plan_was_read_is_not_approved_with_its_code(self, stack):
        made, proposal = validated(stack)
        with httpx.Client(base_url=stack.url, timeout=60, headers=control_headers(stack.app)) as http:
            code = http.get(f"{API}/{made.id}/approve", params={"proposal": proposal.id}).json()["code"]
            stack.skill.write_bytes(stack.skill.read_bytes().replace(b"\n", b"\r\n"))  # the same text, new bytes
            stale = http.post(f"{API}/{made.id}/approve", json={"proposal": proposal.id, "confirm": code})
        assert stale.status_code == 400 and stale.json()["error"]["code"] == "confirm"
        assert not (made.folder / "approval.json").exists()

    def test_an_interrupted_approval_is_repaired_from_the_page(self, stack, monkeypatch):
        made, proposal = validated(stack)
        plan = approve.prepare_approve(stack.store, made.id, proposal.id)
        real = approve._append_changelog
        monkeypatch.setattr(approve, "_append_changelog", lambda *args: (_ for _ in ()).throw(RuntimeError("killed")))
        with pytest.raises(RuntimeError):
            approve.execute_approve(stack.store, plan, confirm=plan.code, front_end="cli")
        monkeypatch.setattr(approve, "_append_changelog", real)
        with httpx.Client(base_url=stack.url, timeout=60) as http:
            assert http.get(f"{API}/{made.id}").json()["approval"]["state"] == "written"
            assert http.post(f"{API}/{made.id}/repair").status_code == 401
            repaired = http.post(f"{API}/{made.id}/repair", headers=control_headers(stack.app))
        assert repaired.status_code == 200 and repaired.json()["state"] == "applied"


def test_an_unfinished_validation_is_shown_without_a_verdict(stack):
    made, proposal = validated(stack)
    path = next((made.folder / "validations").glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(complete=False, status="cancelled", held_out=[], held_out_planned=["2"])
    path.write_text(json.dumps(record), encoding="utf-8")
    shown = invoke(stack.store.home, "refine", "show", made.id[:8])
    assert "unfinished: no verdict" in shown.output and "G1" not in shown.output, shown.output
