"""Scenarios: specs made and kept in the recordings folder from the browser — saved only
over the version the saver read, checked strictly, never writing outside their own folder,
and runnable as a batch."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.ops import OperationError
from tuningfork.ops import scenarios
from tuningfork.proxy.app import create_browse_app

EXAMPLE = Path(__file__).parents[1] / "examples" / "tally-report.evals.json"
API = "/_tuningfork/api/scenarios"


def spec(skill_dir: str | None = None, **options) -> dict:
    data = {
        "skill_name": "tally",
        "evals": [{"id": "1", "prompt": "Summarize data.csv.", "files": [], "tuningfork": {"check": "true", **options}}],
    }
    if skill_dir:
        data["tuningfork"] = {"skill_dir": skill_dir}
    return data


@pytest.fixture
def skill(tmp_path) -> Path:
    folder = tmp_path / "skills" / "tally"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: tally\n---\nWrite report.json.\n", encoding="utf-8")
    return folder


class TestSaving:
    def test_a_scenario_is_its_own_folder_with_an_evals_json(self, tmp_path, skill):
        saved = scenarios.save_scenario(tmp_path, "tally-1", spec(str(skill)), etag=None)
        folder = tmp_path / "scenarios" / "tally-1"
        assert json.loads((folder / "evals.json").read_text())["skill_name"] == "tally"
        assert (folder / "inputs").is_dir()
        assert saved.checked.ok and len(saved.etag) == 32

    def test_saving_over_a_version_the_saver_did_not_read_is_refused(self, tmp_path, skill):
        first = scenarios.save_scenario(tmp_path, "t", spec(str(skill)), etag=None)
        scenarios.save_scenario(tmp_path, "t", spec(str(skill), max_turns=4), etag=first.etag)
        with pytest.raises(OperationError) as refused:
            scenarios.save_scenario(tmp_path, "t", spec(str(skill), max_turns=9), etag=first.etag)
        assert refused.value.code == "conflict"
        with pytest.raises(OperationError) as taken:
            scenarios.save_scenario(tmp_path, "t", spec(str(skill)), etag=None)
        assert taken.value.code == "conflict"
        assert scenarios.read_scenario(tmp_path, "t").spec["evals"][0]["tuningfork"]["max_turns"] == 4

    def test_a_spec_with_problems_is_refused_with_them_named(self, tmp_path, skill):
        bad = spec(str(skill), held_out="false")
        with pytest.raises(OperationError) as refused:
            scenarios.save_scenario(tmp_path, "t", bad, etag=None)
        assert refused.value.code == "invalid"
        assert "held_out" in refused.value.message
        assert not (tmp_path / "scenarios" / "t" / "evals.json").exists()

    def test_what_to_know_before_running_it(self, tmp_path, skill):
        (skill / "SKILL.v2.md").write_text("a variant")
        checked = scenarios.check_spec(spec(str(skill), skills=["tally-report"]), home=tmp_path, name="t")
        text = " ".join(checked.warnings)
        assert "shell command" in text and "SKILL.v2.md" in text and "'tally-report'" in text
        assert scenarios.check_spec(spec(str(tmp_path / "nowhere")), home=tmp_path, name="t").problems

    @pytest.mark.parametrize("name", ["Tally", "../up", "", "a" * 64, "with space", ".trash"])
    def test_names_are_plain(self, tmp_path, name):
        with pytest.raises(OperationError):
            scenarios.save_scenario(tmp_path, name, spec(), etag=None)

    def test_deleting_moves_it_to_the_trash(self, tmp_path, skill):
        scenarios.save_scenario(tmp_path, "gone", spec(str(skill)), etag=None)
        moved = Path(scenarios.delete_scenario(tmp_path, "gone"))
        assert moved.parent == tmp_path / "scenarios" / ".trash" and (moved / "evals.json").is_file()
        assert scenarios.list_scenarios(type("S", (), {"home": tmp_path, "runs": lambda self, limit: []})()) == []

    def test_deleted_twice_in_one_second_it_is_kept_twice_side_by_side(self, tmp_path, skill, monkeypatch):
        # Found in review: the second went inside the first's folder in the trash, its name
        # taken by the second, and a third would have been refused.
        monkeypatch.setattr(scenarios.time, "strftime", lambda *args: "20261018-061500")
        moved = []
        for prompt in ("first", "second", "third"):
            scenarios.save_scenario(tmp_path, "gone", {**spec(str(skill)), "skill_name": prompt}, etag=None)
            moved.append(Path(scenarios.delete_scenario(tmp_path, "gone")))
        assert len(set(moved)) == 3 and all(path.parent == tmp_path / "scenarios" / ".trash" for path in moved)
        assert [json.loads((path / "evals.json").read_text(encoding="utf-8"))["skill_name"] for path in moved] == ["first", "second", "third"]


class TestInputs:
    @pytest.fixture
    def made(self, tmp_path, skill):
        scenarios.save_scenario(tmp_path, "t", spec(str(skill)), etag=None)
        return tmp_path

    def test_an_input_lands_inside_the_scenario(self, made):
        scenarios.put_input(made, "t", "data/rows.csv", b"id\n1\n")
        assert (made / "scenarios" / "t" / "inputs" / "data" / "rows.csv").read_bytes() == b"id\n1\n"
        assert [f.path for f in scenarios.inputs(made, "t")] == ["data/rows.csv"]
        scenarios.delete_input(made, "t", "data/rows.csv")
        assert scenarios.inputs(made, "t") == []

    @pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../b.txt", "C:/x.txt", "", "."])
    def test_nothing_is_written_outside_it(self, made, path):
        with pytest.raises(OperationError):
            scenarios.put_input(made, "t", path, b"x")
        assert not (made / "scenarios" / "escape.txt").exists()

    def test_uploads_have_limits(self, made, monkeypatch):
        monkeypatch.setattr(scenarios, "MAX_FILE", 10)
        with pytest.raises(OperationError) as big:
            scenarios.put_input(made, "t", "big.bin", b"x" * 11)
        assert big.value.code == "too_large"
        monkeypatch.setattr(scenarios, "MAX_FILES", 2)
        scenarios.put_input(made, "t", "a", b"1")
        scenarios.put_input(made, "t", "b", b"2")
        scenarios.put_input(made, "t", "a", b"3")  # replacing one is not one more
        with pytest.raises(OperationError, match="at most 2"):
            scenarios.put_input(made, "t", "c", b"4")


class TestPaths:
    """What the skill folder field names, as tuningfork will open it — said beside it."""

    def test_a_relative_folder_is_resolved_against_the_scenarios_own_folder(self, tmp_path):
        folder = tmp_path / "scenarios" / "t" / "skill"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text("x")
        checked = scenarios.check_spec(spec("skill"), home=tmp_path, name="t")
        assert checked.ok, checked.problems
        found = checked.to_json()["paths"]["skill_dir"]
        assert found["path"] == str(folder)
        assert found["relative_to"] == str(tmp_path / "scenarios" / "t")
        assert found["has_skill_md"] and found["exists"] and found["translated_from"] is None

    def test_before_a_scenario_has_a_name_a_relative_folder_waits_for_it(self, tmp_path):
        checked = scenarios.check_spec(spec("skill"), home=tmp_path, name=None)
        assert checked.paths["skill_dir"]["path"] is None
        assert checked.paths["skill_dir"]["relative_to"] == f"{tmp_path / 'scenarios'}/<name>"
        assert any("needs the scenario's name" in text for text in checked.warnings)

    def test_a_missing_folder_says_so(self, tmp_path):
        missing = tmp_path / "nowhere"
        checked = scenarios.check_spec(spec(str(missing)), home=tmp_path, name="t")
        assert checked.problems == [
            {"where": "tuningfork.skill_dir", "message": f"{missing} does not exist, as tuningfork sees it"},
        ]
        assert checked.paths["skill_dir"]["exists"] is False

    def test_a_windows_path_is_read_and_saved_the_way_wsl_mounts_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuningfork.paths.under_wsl", lambda: True)
        monkeypatch.setattr("tuningfork.paths.drive_root", lambda: f"{tmp_path}/mnt/")
        mounted = tmp_path / "mnt" / "c" / "Users" / "me" / "tally"
        mounted.mkdir(parents=True)
        (mounted / "SKILL.md").write_text("x")
        given = '"C:\\Users\\me\\tally"'
        checked = scenarios.check_spec(spec(given), home=tmp_path, name="t")
        assert checked.ok, checked.problems
        assert checked.paths["skill_dir"]["path"] == str(mounted)
        assert checked.paths["skill_dir"]["translated_from"] == given
        assert checked.to_json()["wsl"] is True
        saved = scenarios.save_scenario(tmp_path, "t", spec(given), etag=None)
        assert saved.spec["tuningfork"]["skill_dir"] == str(mounted)

    def test_an_import_path_may_come_quoted_and_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(EXAMPLE.parent)
        imported = scenarios.import_spec(tmp_path, f'"{EXAMPLE.name}"', "quoted")
        assert imported.checked.ok, imported.checked.problems


def test_an_existing_spec_is_imported_with_its_skill_kept_where_it_is(tmp_path):
    imported = scenarios.import_spec(tmp_path, EXAMPLE, "tally-report")
    assert imported.checked.ok, imported.checked.problems
    assert imported.spec["tuningfork"]["skill_dir"] == str((EXAMPLE.parent / "tally-report").resolve())
    assert imported.spec["evals"][0]["files"][0]["content"].startswith("id,item,amount")
    with pytest.raises(OperationError) as twice:
        scenarios.import_spec(tmp_path, EXAMPLE, "tally-report")
    assert twice.value.code == "conflict"


def test_an_imported_spec_brings_its_input_files(tmp_path):
    source = tmp_path / "elsewhere"
    source.mkdir()
    (source / "rows.csv").write_text("id\n1\n")
    (source / "evals.json").write_text(json.dumps({
        "skill_name": "t", "evals": [{"id": "1", "prompt": "p", "files": ["rows.csv"]}],
    }))
    imported = scenarios.import_spec(tmp_path / "home", source / "evals.json", "t")
    assert imported.spec["evals"][0]["files"] == [{"path": "rows.csv", "source": "inputs/rows.csv"}]
    assert [f.path for f in imported.inputs] == ["rows.csv"]


class TestOverHttp:
    async def test_the_proxy_serves_and_saves_them(self, proxy_client, proxy_app, skill):
        created = await proxy_client.put(f"{API}/t", json={"spec": spec(str(skill)), "etag": None})
        assert created.status_code == 200
        etag = created.json()["etag"]
        stale = await proxy_client.put(f"{API}/t", json={"spec": spec(str(skill), max_turns=3), "etag": "0" * 32})
        assert stale.status_code == 409
        uploaded = await proxy_client.put(f"{API}/t/inputs/rows.csv", content=b"id\n1\n")
        assert uploaded.status_code == 201 and uploaded.json() == {"path": "rows.csv", "size": 5}
        escape = await proxy_client.put(f"{API}/t/inputs/..%2Fescape.txt", content=b"x")
        assert escape.status_code == 400
        read = (await proxy_client.get(f"{API}/t")).json()
        assert read["etag"] == etag and read["inputs"] == [{"path": "rows.csv", "size": 5}]
        listed = (await proxy_client.get(API)).json()
        assert [found["name"] for found in listed] == ["t"]

    async def test_a_reply_check_is_kept_as_saved_and_runs_no_shell(self, proxy_client, skill):
        data = spec(str(skill))
        data["evals"][0]["tuningfork"]["check"] = {"reply_contains": "61.5"}
        saved = await proxy_client.put(f"{API}/t", json={"spec": data, "etag": None})
        assert saved.status_code == 200, saved.text
        read = (await proxy_client.get(f"{API}/t")).json()
        assert read["spec"]["evals"][0]["tuningfork"]["check"] == {"reply_contains": "61.5"}
        assert not any("shell command" in text for text in read["checked"]["warnings"])

    async def test_an_upload_too_large_is_refused_as_it_arrives(self, proxy_client, monkeypatch, skill):
        await proxy_client.put(f"{API}/t", json={"spec": spec(str(skill)), "etag": None})
        monkeypatch.setattr(scenarios, "MAX_FILE", 8)
        refused = await proxy_client.put(f"{API}/t/inputs/big.bin", content=b"x" * 64)
        assert refused.status_code == 413

    async def test_changing_one_needs_the_token(self, proxy_client, skill):
        refused = await proxy_client.put(f"{API}/t", json={"spec": spec(str(skill)), "etag": None}, headers={TOKEN_HEADER: ""})
        assert refused.status_code == 401

    async def test_the_browsing_server_reads_them_and_changes_nothing(self, store, skill):
        scenarios.save_scenario(store.home, "t", spec(str(skill)), etag=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_browse_app(store)), base_url="http://localhost:8788"
        ) as client:
            listed = (await client.get(API)).json()
            refused = await client.put(f"{API}/t", json={"spec": spec(str(skill)), "etag": None})
        assert [found["name"] for found in listed] == ["t"]
        assert refused.status_code == 405


def test_a_scenario_runs_as_a_batch(tmp_path, mock_server, monkeypatch):
    from tuningfork.control import ControlClient
    from tests.helpers import task_stack

    with task_stack(tmp_path, mock_server, monkeypatch) as stack:
        data = stack.spec.to_dict()
        data["tuningfork"]["skill_dir"] = str(stack.spec.resolved_skill_dir())
        scenarios.save_scenario(stack.store.home, "tally", data, etag=None)
        with ControlClient(stack.url) as client:
            job = client.submit("batch", {"scenario": "tally", "n": 1, "harness": "fake", "model": "m"})
            done = client.wait_for_job(job["id"], timeout=60, poll=0.1)
        assert done["status"] == "done", done["error"]
        assert done["result"]["task_id"] == "tally-report/1"
        (summary,) = scenarios.list_scenarios(stack.store)
        (result,) = summary.results
        assert result["runs"] == 1 and result["passed"]["total"] == 1
