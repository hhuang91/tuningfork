"""Skills a person adds (`ops.addskill`): a folder named where it is, a folder uploaded or copied
in, the skill a run had — each then one of the skill's own folders on the Skills page — and
forgetting one, which never deletes a copy."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.helpers import as_response
from tests.test_chat_import import OTHER, PRELOADED, SKILL, SYSTEM, conversation, export
from tuningfork.ops import addskill, imports, skills
from tuningfork.ops.common import OperationError


def skill_folder(tmp_path: Path, name: str = "tally-report") -> Path:
    path = tmp_path / "mine" / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(SKILL, encoding="utf-8")
    return path


class TestAFolderOnThisMachine:
    def test_it_is_remembered_where_it_is_and_listed_as_the_skills_own(self, store, tmp_path):
        mine = skill_folder(tmp_path)
        plan = addskill.prepare_folder(store, mine)
        assert (plan.name, plan.kind, plan.path, plan.files) == ("tally-report", "folder", mine, {})
        assert "an approved edit is written to this folder" in plan.notes
        addskill.execute_add(store, plan)
        # Nothing copied: approvals write to the folder itself.
        assert not (addskill.root(store.home) / "tally-report").exists()
        (summary,) = skills.skills(store)
        assert (summary.name, summary.folders, summary.added) == ("tally-report", [str(mine)], "folder")
        (own,) = skills.skill(store, "tally-report").folders
        assert own.path == mine and own.text == SKILL and own.added["kind"] == "folder"
        assert own.to_json()["added"]["origin"] == {"folder": str(mine)}

    def test_a_folder_without_its_skill_md_twice_or_one_tuningfork_keeps_is_refused(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(OperationError, match="holds no SKILL.md"):
            addskill.prepare_folder(store, empty)
        with pytest.raises(OperationError, match="does not exist, as tuningfork sees it"):
            addskill.prepare_copy(store, tmp_path / "nowhere")
        mine = skill_folder(tmp_path)
        addskill.execute_add(store, addskill.prepare_folder(store, mine))
        with pytest.raises(OperationError, match="added already") as twice:
            addskill.prepare_folder(store, mine)
        assert twice.value.code == "conflict"
        copy = addskill.execute_add(store, addskill.prepare_upload(store, "other", {"SKILL.md": OTHER.encode()}))
        with pytest.raises(OperationError, match="a copy tuningfork keeps already"):
            addskill.prepare_folder(store, copy.path)


class TestUploaded:
    def test_its_files_are_kept_as_a_copy_of_tuningforks_own(self, store):
        files = {"SKILL.md": SKILL.encode(), "scripts/total.py": b"print(1)\n", ".git/HEAD": b"ref",
                 "__pycache__/total.pyc": b"\0"}
        plan = addskill.prepare_upload(store, "tally-copy", files)
        # A repository's history and caches are never copied.
        assert plan.kind == "copy" and sorted(plan.files) == ["SKILL.md", "scripts/total.py"]
        assert any("never back to where it came from" in note for note in plan.notes)
        found = addskill.execute_add(store, plan)
        assert found.path == addskill.root(store.home) / "tally-copy"
        assert (found.path / "scripts" / "total.py").read_bytes() == b"print(1)\n"
        assert not list(addskill.root(store.home).glob(".incoming-*"))
        (own,) = skills.skill(store, "tally-copy").folders
        assert own.added["kind"] == "copy" and own.added["origin"] == {"upload": {"files": 2}} and own.text == SKILL

    def test_a_path_outside_the_copy_no_skill_md_or_a_bad_name_is_refused(self, store):
        for outside in ("../evil.md", "/etc/x", "C:/x"):
            with pytest.raises(OperationError, match="not a path inside"):
                addskill.prepare_upload(store, "fine", {outside: b"x", "SKILL.md": b"x"})
        with pytest.raises(OperationError, match="no SKILL.md at the top"):
            addskill.prepare_upload(store, "fine", {"docs/SKILL.md": b"x"})
        with pytest.raises(OperationError, match="not a skill's name"):
            addskill.prepare_upload(store, "Bad Name", {"SKILL.md": b"x"})
        assert not addskill.root(store.home).exists()

    def test_what_is_past_the_limits_is_left_out_and_said(self, store, monkeypatch):
        monkeypatch.setattr(addskill, "_limits", lambda: (3, 10, 25))
        plan = addskill.prepare_upload(store, "big", {"SKILL.md": b"12345", "a.txt": b"x" * 11, "b.txt": b"x" * 9,
                                                      "c.txt": b"x" * 9, "d.txt": b"x"})
        assert sorted(plan.files) == ["SKILL.md", "b.txt", "c.txt"]
        assert [line.split(":")[0] for line in plan.left_out] == ["a.txt", "d.txt"]
        assert plan.warnings and "a.txt" in plan.warnings[0]

    def test_a_folder_copied_in_from_the_terminal_leaves_its_history_out(self, store, tmp_path):
        mine = skill_folder(tmp_path)
        (mine / ".git").mkdir()
        (mine / ".git" / "HEAD").write_text("ref", encoding="utf-8")
        (mine / "references.md").write_text("see", encoding="utf-8")
        plan = addskill.prepare_copy(store, mine)
        assert plan.name == "tally-report" and sorted(plan.files) == ["SKILL.md", "references.md"]
        assert plan.origin == {"upload": {"files": 2, "from": str(mine)}}


class TestFromARun:
    def test_the_copy_an_import_kept_and_the_one_named_of_several(self, store):
        made = imports.execute_import(store, imports.prepare_import(store, export(conversation(("tally-report", SKILL), ("csv-tidy", OTHER)))))
        assert addskill.run_skills(store, made.run_id) == [{"name": "tally-report", "kept": True}, {"name": "csv-tidy", "kept": True}]
        with pytest.raises(OperationError, match="has several skills: name the one to keep") as several:
            addskill.prepare_from_run(store, made.run_id)
        assert several.value.code == "choose_skill"
        plan = addskill.prepare_from_run(store, made.run_id, skill="csv-tidy")
        assert (plan.name, plan.kind, plan.files["SKILL.md"].decode()) == ("csv-tidy", "copy", OTHER)
        assert plan.origin == {"run": made.run_id, "skill": "csv-tidy", "how": "the copy it kept"}
        found = addskill.execute_add(store, plan)
        assert (found.path / "SKILL.md").read_text(encoding="utf-8") == OTHER
        assert skills.skill(store, "csv-tidy").folders[0].added["origin"]["run"] == made.run_id

    def test_a_run_recorded_live_gives_the_skill_md_its_model_read(self, store):
        store.begin_run("live", None)
        store.record_step(request={"model": "m", "messages": [{"role": "system", "content": SYSTEM + PRELOADED},
                                                              {"role": "user", "content": "Tally data.csv."}]},
                          response=as_response({"role": "assistant", "content": "Done."}), harness="hermes")
        run = store.get_run("live")
        plan = addskill.prepare_from_run(store, run.id, name="tally-kept")
        assert plan.files == {"SKILL.md": SKILL.encode()} and plan.origin["how"] == "the SKILL.md its model read"
        assert any("the run never showed" in note for note in plan.notes)

    def test_a_run_with_no_skill_says_so(self, store):
        store.begin_run("bare", None)
        store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                          response=as_response({"role": "assistant", "content": "hello"}))
        with pytest.raises(OperationError, match="kept no skill, and its conversation shows none"):
            addskill.prepare_from_run(store, store.get_run("bare").id)


class TestForgetting:
    def test_a_folder_named_stays_and_a_copy_is_moved_aside_whole(self, store, tmp_path):
        mine = skill_folder(tmp_path)
        addskill.execute_add(store, addskill.prepare_folder(store, mine))
        assert addskill.forget(store, "tally-report")["moved_to"] is None
        assert (mine / "SKILL.md").is_file() and addskill.added(store.home) == []
        copy = addskill.execute_add(store, addskill.prepare_upload(store, "kept", {"SKILL.md": SKILL.encode()}))
        # What an approval keeps beside a copy goes with it.
        (copy.path.parent / "kept.changelog.md").write_text("# approvals\n", encoding="utf-8")
        gone = addskill.forget(store, "kept")
        moved = Path(gone["moved_to"])
        assert moved.parent == addskill.root(store.home) / addskill.FORGOTTEN
        assert (moved / "kept" / "SKILL.md").is_file() and (moved / "kept.changelog.md").is_file() and (moved / "record.json").is_file()
        assert not copy.path.exists() and addskill.added(store.home) == []
        with pytest.raises(OperationError, match="no skill called") as again:
            addskill.forget(store, "kept")
        assert again.value.code == "not_found"

    def test_a_copy_forgotten_twice_in_one_second_keeps_both(self, store, monkeypatch):
        monkeypatch.setattr(addskill.time, "strftime", lambda *_: "20261020-010203")
        moved = []
        for _ in range(2):
            addskill.execute_add(store, addskill.prepare_upload(store, "kept", {"SKILL.md": SKILL.encode()}))
            moved.append(Path(addskill.forget(store, "kept")["moved_to"]))
        assert moved[0] != moved[1] and all((each / "kept" / "SKILL.md").is_file() for each in moved)

    def test_a_copy_a_scenario_runs_from_is_kept(self, store):
        copy = addskill.execute_add(store, addskill.prepare_upload(store, "kept", {"SKILL.md": SKILL.encode()}))
        scenario = store.home / "scenarios" / "uses-it"
        scenario.mkdir(parents=True)
        (scenario / "evals.json").write_text(json.dumps(
            {"skill_name": "kept", "tuningfork": {"skill_dir": str(copy.path)}, "evals": []}), encoding="utf-8")
        with pytest.raises(OperationError, match="scenario uses-it runs from this copy") as refused:
            addskill.forget(store, "kept")
        assert refused.value.code == "conflict" and copy.path.is_dir()
        assert skills.skill(store, "kept").scenarios == ["uses-it"]


def test_a_person_adds_and_forgets_a_skill_from_the_terminal(store, tmp_path):
    from tuningfork.cli import app

    runner = CliRunner()
    home = ["--home", str(store.home)]
    mine = skill_folder(tmp_path)
    looked = runner.invoke(app, ["add-skill", str(mine), "--dry-run", *home], env={"COLUMNS": "160"})
    assert looked.exit_code == 0, looked.output
    assert "a folder on this machine, remembered where it is" in looked.output and "nothing added" in looked.output
    assert addskill.added(store.home) == []
    done = runner.invoke(app, ["add-skill", str(mine), *home], env={"COLUMNS": "160"})
    assert done.exit_code == 0 and "added: `tuningfork skill tally-report` shows it" in done.output, done.output
    copied = runner.invoke(app, ["add-skill", str(mine), "--copy", "--name", "tally-copy", *home], env={"COLUMNS": "160"})
    assert copied.exit_code == 0 and "a copy kept in the recordings folder" in copied.output, copied.output
    # Its page in the terminal says how each was added, as the web page does.
    shown = runner.invoke(app, ["skill", "tally-copy", *home], env={"COLUMNS": "200"})
    assert f"added: tuningfork's copy, copied from {mine}" in shown.output, shown.output
    shown = runner.invoke(app, ["skill", "tally-report", *home], env={"COLUMNS": "200"})
    assert "added: a folder on this machine, remembered where it is" in shown.output, shown.output
    both = runner.invoke(app, ["add-skill", *home], env={"COLUMNS": "160"})
    assert both.exit_code != 0 and "name the skill's folder, or a run with --run" in both.output
    forgot = runner.invoke(app, ["forget-skill", "tally-copy", *home], env={"COLUMNS": "160"})
    assert forgot.exit_code == 0 and "its copy is moved aside to" in forgot.output, forgot.output
    assert [each.name for each in addskill.added(store.home)] == ["tally-report"]


async def test_the_web_ui_plans_adds_and_forgets(proxy_client, recorder, tmp_path):
    api = "/_tuningfork/api"
    mine = skill_folder(tmp_path)
    planned = (await proxy_client.post(f"{api}/skills/add", json={"kind": "folder", "path": str(mine), "dry_run": True})).json()
    assert planned["plan"]["name"] == "tally-report" and planned["plan"]["kind"] == "folder"
    # A form asking as it is filled in hears a refusal as an answer.
    quiet = await proxy_client.post(f"{api}/skills/add", json={"kind": "folder", "path": str(tmp_path), "dry_run": True, "quiet": True})
    assert quiet.status_code == 200 and "holds no SKILL.md" in quiet.json()["refused"]["message"]
    loud = await proxy_client.post(f"{api}/skills/add", json={"kind": "folder", "path": str(tmp_path)})
    assert loud.status_code == 400 and loud.json()["error"]["field"] == "path"
    files = [{"path": "SKILL.md", "data": base64.b64encode(SKILL.encode()).decode()},
             {"path": "scripts/total.py", "data": base64.b64encode(b"print(1)\n").decode()}]
    made = await proxy_client.post(f"{api}/skills/add", json={"kind": "upload", "name": "uploaded", "files": files})
    assert made.status_code == 201 and made.json()["added"]["kind"] == "copy"
    assert (addskill.root(recorder.store.home) / "uploaded" / "scripts" / "total.py").is_file()
    odd = await proxy_client.post(f"{api}/skills/add", json={"kind": "upload", "name": "x", "files": [{"path": "SKILL.md", "data": "%%"}]})
    assert odd.status_code == 400 and "not base64" in odd.json()["error"]["message"]
    extra = await proxy_client.post(f"{api}/skills/add", json={"kind": "folder", "path": str(mine), "name": "y"})
    assert extra.status_code == 400 and extra.json()["error"]["field"] == "name"
    listed = (await proxy_client.get(f"{api}/skills")).json()
    assert {each["name"]: each["added"] for each in listed} == {"uploaded": "copy"}
    gone = await proxy_client.delete(f"{api}/skills/added/uploaded")
    assert gone.status_code == 200 and gone.json()["moved_to"]
    missing = await proxy_client.delete(f"{api}/skills/added/uploaded")
    assert missing.status_code == 404
