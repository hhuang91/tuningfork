"""Refinements (`ops.refine`): one skill at one checkpoint toward a declared target, and the
edits proposed for it — on runs a real batch made (the fake agent, the mock model)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tuningfork.ops import refine, validate
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.proxy.app import create_browse_app
from tuningfork.specs import hash_tree
from tuningfork.store import Store
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


def run_of(stack, **options) -> str:
    """One run of a batch of the task, with `prepare_batch`'s options: another skill folder or file."""
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=1, **options)
    (outcome,) = execute_batch(stack.store, plan).outcomes
    return outcome.run_id


def older(parity: dict, stack, run_id: str) -> None:
    """The run's parity as an older tuningfork recorded it."""
    with stack.store._lock:
        stack.store._conn.execute("UPDATE runs SET parity = ? WHERE id = ?", (json.dumps(parity), run_id))
        stack.store._conn.commit()


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
        assert {key: made.data["source"][key] for key in ("origin", "given", "others_match")} == {
            "origin": None, "given": None, "others_match": True}
        assert refine.source_note(made.data["source"], run_id) is None
        assert made.target["kind"] == "rules" and made.data["created_by"] == {"via": "person"}
        assert (stack.store.home / "refinements" / made.id / "refinement.json").is_file()

    def test_a_run_given_another_folder_says_so_from_the_start(self, stack):
        """Issue #55: a run of `batch --skill-dir <folder>` records the folder it was given
        (`parity.skill_dir`) and whether it held what the task's folder held then
        (`parity.skill_as_task`). A refinement of it says the run was given another text where it
        was, never that the task's file has changed since; a copy that held the task's text, the
        task's folder edited since, is said to have changed since; and an older run, whose record
        says neither, is said to differ."""
        import shutil

        theirs = stack.tmp / "skills-src" / "tally-report"
        mine = stack.tmp / "elsewhere" / "tally-report"
        shutil.copytree(theirs, mine)
        plain = stack.store.get_run(run_of(stack))
        assert (plain.parity["skill_dir"], plain.parity["skill_as_task"]) == (str(theirs.resolve()), True)
        # The task's folder in a repository: what is in a dot-folder, which a chain's copy of a
        # link's skill leaves out, never tells a copy from the folder (`specs.kept_alike`).
        (theirs / ".git").mkdir()
        (theirs / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        alike = run_of(stack, skill_dir=mine)
        assert stack.store.get_run(alike).parity["skill_dir"] == str(mine.resolve())
        assert stack.store.get_run(alike).parity["skill_as_task"] is True
        assert start(stack, alike).data["source"]["origin"] is None  # a copy of the task's, as it is
        # Held the task's text then, the task's folder edited since: changed since, never another text.
        had = (theirs / "SKILL.md").read_bytes()
        (theirs / "SKILL.md").write_bytes(had + b"\nAn edit approved since.\n")
        since = start(stack, alike, target_match="row_count").data["source"]
        assert (since["origin"], since["given"]) == ("task", None)
        assert refine.source_note(since, alike).startswith("the skill's file has changed since this run was given it")
        (theirs / "SKILL.md").write_bytes(had)

        (mine / "SKILL.md").write_text(had.decode("utf-8") + "\nIts own line.\n", encoding="utf-8")
        run_id = run_of(stack, skill_dir=mine)
        source = start(stack, run_id).data["source"]
        assert source["matches"] is False and (source["origin"], source["given"]) == ("other", f"another folder, {mine.resolve()}")
        assert refine.source_note(source, run_id).startswith(f"run {run_id[:8]} was given another skill text")
        # Recorded before a run kept what it was given: said to differ, never to have changed.
        older({key: value for key, value in stack.store.get_run(run_id).parity.items()
               if key not in ("skill_dir", "skill_as_task")}, stack, run_id)
        unknown = start(stack, run_id, target_match="row_count").data["source"]
        assert unknown["origin"] == "unknown"
        assert refine.source_note(unknown, run_id) == ("the skill's file does not hold the text this run had: proposals "
                                                       "edit the run's copy")
        # A file of its own beside the same SKILL.md: the folder still differs, and says so.
        (mine / "SKILL.md").write_bytes(had)
        (mine / "notes.md").write_text("a file the task's folder never held\n", encoding="utf-8")
        run_id = run_of(stack, skill_dir=mine)
        source = start(stack, run_id).data["source"]
        assert source["matches"] is True and source["others_match"] is False and source["origin"] == "other"

    def test_a_fork_of_a_run_given_another_text_answers_for_its_source(self, stack):
        """A fork replays its source and is given its source's copy of the skill, recording neither
        what the source was given nor a copy of its own: a refinement of it says what its source
        was given (found in review)."""
        variant = stack.tmp / "variant.md"
        variant.write_text((stack.tmp / "skills-src" / "tally-report" / "SKILL.md").read_text(encoding="utf-8")
                           + "\nIts own line.\n", encoding="utf-8")
        source = run_of(stack, skill_file=str(variant))
        _, (fork,) = stack.fork(source, "write_file")
        assert stack.store.get_run(fork.run_id).parity.get("fork_of") == source
        made = start(stack, fork.run_id).data["source"]
        assert (made["origin"], made["given"]) == ("other", f"{variant} as its SKILL.md")

    def test_a_refinement_recorded_before_says_what_its_run_was_given_where_it_is_shown(self, stack):
        """One recorded before the refinement kept what its run was given: its view finds it, and
        writes nothing (issue #55: the refinement of a chain's later link said its file changed)."""
        variant = stack.tmp / "variant.md"
        variant.write_text((stack.tmp / "skills-src" / "tally-report" / "SKILL.md").read_text(encoding="utf-8")
                           + "\nIts own line.\n", encoding="utf-8")
        run_id = run_of(stack, skill_file=str(variant))
        made = start(stack, run_id)
        record = stack.store.home / "refinements" / made.id / "refinement.json"
        data = json.loads(record.read_text(encoding="utf-8"))
        data["source"] = {key: data["source"][key] for key in ("path", "sha256", "matches")}
        record.write_text(json.dumps(data), encoding="utf-8")
        shown = refine.view(stack.store, made.id)["refinement"]["source"]
        assert (shown["origin"], shown["given"]) == ("other", f"{variant} as its SKILL.md")
        assert "origin" not in json.loads(record.read_text(encoding="utf-8"))["source"]

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


@contextmanager
def copied(stack, *, keep_original: bool = False):
    """The recordings folder copied whole, as a person copies one to keep or share it, and opened
    there; the folder it was copied from no longer holding its proposals' saved texts, unless kept."""
    copy = stack.tmp / "copy"
    shutil.copytree(stack.store.home, copy, symlinks=True)
    if not keep_original:
        shutil.rmtree(stack.store.home / "variants")
    store = Store(copy)
    try:
        yield store
    finally:
        store.close()


def kept_in(store, proposal) -> Path:
    """Where a recordings folder keeps a proposal's saved text, by the name its record keeps."""
    return store.home / "variants" / "tally-report" / f"{proposal.data['variant']}.md"


async def fetch(store, path: str) -> httpx.Response:
    """A page's data as the web UI asks for it, from `tuningfork ui` on that recordings folder."""
    transport = httpx.ASGITransport(app=create_browse_app(store), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8788") as client:
        return await client.get(path)


class TestFromACopy:
    """Issue #76: a refinement's page answered HTTP 500 from a copy of the recordings folder, or
    the folder moved, once the folder it was made in no longer held its proposals' texts: each
    proposal's record names the file where it was saved. Each is read where the folder it is opened
    in keeps it, and a proposal whose text is gone is shown without it, taking no work."""

    def test_a_copy_reads_its_own_saved_texts(self, stack):
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        with copied(stack) as store:
            found = refine.proposal(store, made.id, proposal.id)
            assert found.saved() == (kept_in(store, proposal), found.text()) and KEYS["new"] in found.text()
            answer = asyncio.run(fetch(store, f"/_tuningfork/api/refinements/{made.id}"))
            assert answer.status_code == 200, answer.text
            (item,) = answer.json()["proposals"]
            changed = [line["op"] for hunk in item["diff"] for line in hunk["lines"] if line["op"] != " "]
            assert item["missing"] is None and changed == ["-", "+"]
            # A screen's arms name the copy's own file: a fork to the end is given it.
            plan = refine.prepare_refinement_screen(store, made.id, [proposal.id], base_url=stack.url, n=2)
            assert [arm.meta["variant_path"] for arm in plan.arms] == [str(kept_in(store, proposal))]

    def test_where_it_was_saved_serves_a_copy_without_its_own(self, stack):
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        with copied(stack, keep_original=True) as store:
            shutil.rmtree(store.home / "variants")
            assert refine.proposal(store, made.id, proposal.id).saved()[0] == Path(proposal.data["variant_path"])

    def test_a_file_holding_another_text_is_never_taken_for_it(self, stack):
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        with copied(stack, keep_original=True) as store:
            own, original = kept_in(store, proposal), Path(proposal.data["variant_path"])
            own.write_text("Another text.\n", encoding="utf-8")
            found = refine.proposal(store, made.id, proposal.id)
            assert found.saved()[0] == original
            original.write_text("Another text again.\n", encoding="utf-8")
            assert found.saved() is None
            assert found.missing() == (f"proposal {proposal.id[:8]}'s saved text is missing: {own} holds another text, "
                                       f"and {original}, where it was saved, holds another text")
            # Two places: the hint names this folder's, where the text belongs.
            assert found.remedy() == f"put its file back at {own}, from a copy of the recordings folder that still holds it"

    @pytest.mark.skipif(os.name == "posix" and os.geteuid() == 0, reason="root reads a file whatever its mode")
    def test_the_hint_is_for_this_folders_own_file(self, stack):
        """A copy whose records name a folder this user cannot read, another person's say: the text
        put back in the copy still brings it back, so the hint stays. Where the copy's own file is
        there but unreadable, its permissions are the problem, and no file put back is the cure."""
        made = start(stack, a_run(stack))
        proposal = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        with copied(stack, keep_original=True) as store:
            own, original = kept_in(store, proposal), Path(proposal.data["variant_path"])
            own.unlink()
            original.chmod(0)
            try:
                found = refine.proposal(store, made.id, proposal.id)
                assert found.missing().endswith(f"{original}, where it was saved, cannot be read (Permission denied)")
                assert found.remedy() == f"put its file back at {own}, from a copy of the recordings folder that still holds it"
                own.write_text("Another text.\n", encoding="utf-8")
                own.chmod(0)
                assert found.remedy() is None
            finally:
                original.chmod(0o644)
                own.chmod(0o644)

    def test_a_text_two_of_its_places_name_is_taken_once_when_deleted(self, stack):
        """A recordings folder reached through a link, its proposal saved under the link's path: both
        of the proposal's places name one file, which deleting the refinement for good takes once."""
        made = start(stack, a_run(stack))
        alias = stack.tmp / "alias"
        alias.symlink_to(stack.store.home, target_is_directory=True)
        through = Store(alias)
        try:
            proposal = refine.add_proposal(through, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        finally:
            through.close()
        assert Path(proposal.data["variant_path"]).parent.parent.parent == alias
        assert refine.made_by(stack.store, refine.get(stack.store, made.id)).variants == [kept_in(stack.store, proposal)]

    def test_a_proposal_whose_text_is_gone_is_shown_without_it_and_takes_no_work(self, stack):
        made = start(stack, a_run(stack))
        # The gone one first: the other's diff must still be its own, read after a text that is not there.
        gone = refine.add_proposal(stack.store, made.id, edits=[{"old": KEYS["old"], "new": "Name the keys in report.json."}],
                                   rationale="r", proposer={"via": "person"})
        kept = refine.add_proposal(stack.store, made.id, edits=[KEYS], rationale="r", proposer={"via": "person"})
        Path(gone.data["variant_path"]).unlink()
        said = f"proposal {gone.id[:8]}'s saved text is missing: {gone.data['variant_path']} is not there"
        for masking in (None, stack.store.masking):  # as the page shows it, and as an agent is shown it
            items = {item["id"]: item for item in refine.view(stack.store, made.id, masking=masking)["proposals"]}
            added = [line["text"] for hunk in items[kept.id]["diff"] for line in hunk["lines"] if line["op"] == "+"]
            assert items[kept.id]["missing"] is None and added == [KEYS["new"]]
            assert items[gone.id]["missing"] == said and items[gone.id]["diff"] == []
        answer = asyncio.run(fetch(stack.store, f"/_tuningfork/api/refinements/{made.id}"))
        assert answer.status_code == 200 and len(answer.json()["proposals"]) == 2
        for attempt in (
            lambda: refine.prepare_refinement_screen(stack.store, made.id, [kept.id, gone.id], base_url=stack.url, n=2),
            lambda: validate.prepare_validate(stack.store, made.id, gone.id, n=validate.MINIMUM, runs=validate.MINIMUM,
                                              proxy_url=stack.url),
        ):
            with pytest.raises(OperationError) as refused:
                attempt()
            assert (refused.value.code, refused.value.field, refused.value.message) == ("text_missing", "proposals", said)
            # One place, named in the sentence already.
            assert refused.value.hint == "put its file back, from a copy of the recordings folder that still holds it"
        # The other is screened as before.
        assert refine.prepare_refinement_screen(stack.store, made.id, [kept.id], base_url=stack.url, n=2).arms
