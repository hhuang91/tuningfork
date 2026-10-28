"""Task specs and the batch runner.

Driven by a scripted harness rather than a model: the runner's job is workspace and home
isolation, skill hashing, check execution and run bookkeeping, none of which needs
inference to verify.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tuningfork.harness.base import HarnessError, LaunchSpec
from tuningfork.runner import BatchRunner, lookalike_skill_files, pass_rate
from tuningfork.specs import (
    Eval,
    TuningforkOptions,
    TaskSpec,
    hash_tree,
    load_spec,
    materialize_files,
    run_check,
)
from tests.helpers import TWICE, as_response, record_launched, text_message, token_in

REPO = Path(__file__).resolve().parents[1]

GOOD_REPORT = (
    "import json;"
    "json.dump({'row_count': 4, 'amount_total': 61.5}, open('report.json','w'))"
)
BAD_REPORT = "import json; json.dump({'count': 4, 'total': 61.5}, open('report.json','w'))"
MUTATES_SKILL = (
    "import json, os, pathlib;"
    "json.dump({'row_count': 4, 'amount_total': 61.5}, open('report.json','w'));"
    "p = pathlib.Path(os.environ['SKILLS_DIR']) / 'tally-report' / 'SKILL.md';"
    "p.write_text(p.read_text() + '\\nimproved by the agent\\n')"
)
SLEEPS = "import time; time.sleep(5)"


class ScriptedHarness:
    """Runs a fixed Python snippet where a real harness would run an agent."""

    name = "scripted"

    def __init__(self, script: str) -> None:
        self.script = script
        self.homes: list[Path] = []
        self.workspaces: list[Path] = []
        #: The base URL each home was given: a launch URL, carrying the run's token.
        self.base_urls: dict[Path, str] = {}

    def prepare_home(self, home, *, base_url: str, model: str) -> Path:
        home = Path(home)
        (home / "skills").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(f"base_url: {base_url}\n", encoding="utf-8")
        self.homes.append(home)
        self.base_urls[home] = base_url
        return home

    def skill_roots(self, home) -> tuple[Path, ...]:
        return (Path(home) / "skills",)

    def launch(self, prompt, *, home, workspace, skills=(), extra_args=(), max_turns=12, yolo=True):
        self.workspaces.append(Path(workspace))
        return LaunchSpec(
            argv=("python3", "-c", self.script),
            env={
                "PATH": os.environ.get("PATH", ""),
                "SKILLS_DIR": str(Path(home) / "skills"),
            },
            cwd=Path(workspace),
        )


CHECK = (
    "python3 -c \"import json,sys; d=json.load(open('report.json'));"
    " sys.exit(0 if d.get('row_count')==4 else 1)\""
)


class RecordingHarness(ScriptedHarness):
    """Also records a model call as it launches, the way the proxy would in a real run:
    under the launch its base URL names."""

    def __init__(self, script: str, store, edits: dict | None = None, reply: str | None = None) -> None:
        super().__init__(script)
        self.store = store
        self.edits = edits
        #: What the model says at the recorded step: the run's final reply.
        self.reply = reply

    def record(self, home, prompt: str, **step) -> None:
        record_launched(
            self.store,
            token_in(self.base_urls[Path(home)]),
            {"model": "m", "messages": [{"role": "user", "content": prompt}]},
            **step,
        )

    def launch(self, prompt, **kwargs):
        response = as_response(text_message(self.reply)) if self.reply is not None else None
        self.record(kwargs["home"], prompt, edits=self.edits, response=response)
        return super().launch(prompt, **kwargs)


def make_spec(tmp_path: Path, check: str | None = CHECK, **options) -> tuple[TaskSpec, Eval]:
    skill = tmp_path / "tally-report"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text("# tally-report\n", encoding="utf-8")
    item = Eval(
        id="1",
        prompt="tally data.csv",
        files=[{"path": "data.csv", "content": "id,amount\n1,10\n"}],
        tuningfork=TuningforkOptions(check=check, **options),
    )
    spec = TaskSpec(
        skill_name="tally-report",
        evals=[item],
        path=tmp_path / "spec.json",
        skill_dir="tally-report",
    )
    return spec, item


def make_runner(store, tmp_path: Path, script: str, *, recording: bool = True) -> BatchRunner:
    """A runner whose harness records a model call, as a real one does through the proxy.

    With `recording=False` nothing reaches the store: a harness that never got going.
    """
    return BatchRunner(
        store,
        RecordingHarness(script, store) if recording else ScriptedHarness(script),
        base_url="http://127.0.0.1:8787/v1",
        model="m",
        workdir=tmp_path / "batches",
        settle=0,
    )


class TestSpecs:
    def test_the_shipped_example_parses(self):
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")

        assert spec.skill_name == "tally-report"
        item = spec.get()
        assert item.tuningfork.check
        assert item.tuningfork.skills == ["tally-report"]
        assert spec.resolved_skill_dir().name == "tally-report"
        assert (spec.resolved_skill_dir() / "SKILL.md").exists()

    def test_the_example_check_accepts_the_intended_report(self, tmp_path):
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")
        (tmp_path / "report.json").write_text(
            json.dumps({"row_count": 4, "amount_total": 61.5}), encoding="utf-8"
        )
        assert run_check(spec.get().tuningfork.check, tmp_path).passed

    def test_the_shipped_example_keeps_its_variant_outside_the_skill_folder(self):
        """Every run gets a copy of the whole folder. The variant sat in it from Phase 1
        until it was noticed: an answer key beside the skill under test."""
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")
        assert lookalike_skill_files(spec.resolved_skill_dir()) == []
        assert (REPO / "examples" / "variants" / "tally-report" / "SKILL.v2.md").exists()

    def test_variants_beside_a_skill_are_noticed(self, tmp_path):
        for name in ("SKILL.md", "SKILL.v2.md", "skill-old.md", "README.md", "notes.txt"):
            (tmp_path / name).write_text(name, encoding="utf-8")
        assert lookalike_skill_files(tmp_path) == ["SKILL.v2.md", "skill-old.md"]

    def test_the_example_check_rejects_the_keys_the_wording_invites(self, tmp_path):
        """`count`/`total` is what the model wrote in every clean run of the shipped skill."""
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")
        (tmp_path / "report.json").write_text(
            json.dumps({"count": 4, "total": 61.5}), encoding="utf-8"
        )
        assert not run_check(spec.get().tuningfork.check, tmp_path).passed

    def test_eval_lookup_by_id(self):
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")
        assert spec.get("1").id == "1"
        with pytest.raises(KeyError):
            spec.get("99")

    def test_the_example_holds_out_a_second_task_with_the_same_contract(self, tmp_path):
        """Held out from refinement: other data, the same keys. A batch runs eval 1 unless told."""
        spec = load_spec(REPO / "examples" / "tally-report.evals.json")
        assert spec.default_eval().id == "1"
        (held,) = spec.held_out_evals()
        assert held.id == "2" and held.files and held.tuningfork.held_out
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"row_count": 6, "amount_total": 82.5}), encoding="utf-8")
        assert run_check(held.tuningfork.check, tmp_path).passed
        report.write_text(json.dumps({"count": 6, "total": 82.5}), encoding="utf-8")
        assert not run_check(held.tuningfork.check, tmp_path).passed
        report.write_text(json.dumps({"row_count": 4, "amount_total": 61.5}), encoding="utf-8")
        assert not run_check(held.tuningfork.check, tmp_path).passed  # the training task's answer

    def test_inline_files_are_written(self, tmp_path):
        item = Eval(id="1", prompt="p", files=[{"path": "a/b.txt", "content": "hi"}])
        materialize_files(item, tmp_path, tmp_path)
        assert (tmp_path / "a" / "b.txt").read_text() == "hi"

    def test_a_file_copied_from_beside_the_spec_keeps_its_bytes(self, tmp_path):
        (tmp_path / "spec" / "inputs").mkdir(parents=True)
        (tmp_path / "spec" / "inputs" / "photo.png").write_bytes(b"\x89PNG\x00\xff")
        item = Eval(id="1", prompt="p", files=[{"path": "img/photo.png", "source": "inputs/photo.png"}])
        materialize_files(item, tmp_path / "ws", tmp_path / "spec")
        assert (tmp_path / "ws" / "img" / "photo.png").read_bytes() == b"\x89PNG\x00\xff"

    @pytest.mark.parametrize(
        "entry",
        [
            {"path": "../escaped.txt", "content": "x"},
            {"path": "/tmp/escaped.txt", "content": "x"},
            {"path": "a/../../escaped.txt", "content": "x"},
            {"path": "", "content": "x"},
            {"path": ".", "content": "x"},
            "..",
        ],
    )
    def test_nothing_is_written_outside_the_workspace(self, tmp_path, entry):
        """Found in review: an absolute path, or one climbing out with `..`, was written
        wherever it pointed; an entry of `..` copied the spec's parent folder into the
        workspace's."""
        from tuningfork.specs import SpecError

        (tmp_path / "spec").mkdir()
        workspace = tmp_path / "runs" / "0" / "workspace"
        item = Eval(id="1", prompt="p", files=[entry])

        with pytest.raises(SpecError):
            materialize_files(item, workspace, tmp_path / "spec")

        assert not (tmp_path / "runs" / "0" / "escaped.txt").exists()
        assert sorted(p.name for p in (tmp_path / "runs" / "0").iterdir()) == ["workspace"]

    def test_missing_check_is_not_a_pass(self, tmp_path):
        result = run_check(None, tmp_path)
        assert result.passed is False
        assert "no check" in result.output

    def test_hash_tree_notices_an_edit(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("one", encoding="utf-8")
        before = hash_tree([tmp_path])
        (tmp_path / "SKILL.md").write_text("two", encoding="utf-8")
        assert hash_tree([tmp_path]) != before

    def test_harness_bookkeeping_is_not_mistaken_for_a_skill_edit(self, tmp_path):
        """Hermes writes `.usage.json` into the skills dir on every run."""
        (tmp_path / "SKILL.md").write_text("content", encoding="utf-8")
        before = hash_tree([tmp_path])

        (tmp_path / ".usage.json").write_text('{"runs": 1}', encoding="utf-8")
        (tmp_path / ".usage.json.lock").write_text("", encoding="utf-8")
        (tmp_path / "notes.log").write_text("chatter", encoding="utf-8")

        assert hash_tree([tmp_path]) == before


class TestBatchRunner:
    def test_each_run_gets_a_fresh_workspace_and_home(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)

        result = runner.run(spec, item, 3)

        workspaces = runner.adapter.workspaces
        homes = runner.adapter.homes
        assert len(set(workspaces)) == 3
        assert len(set(homes)) == 3
        assert len(result.outcomes) == 3

    def test_passing_runs_are_counted(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        result = make_runner(store, tmp_path, GOOD_REPORT).run(spec, item, 3)

        assert result.passes == 3
        assert all(outcome.passed for outcome in result.outcomes)

    def test_failing_check_is_reported_as_failure(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        result = make_runner(store, tmp_path, BAD_REPORT).run(spec, item, 2)

        assert result.passes == 0
        assert all(not outcome.passed for outcome in result.outcomes)

    def test_input_files_reach_the_workspace(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.run(spec, item, 1)

        assert (runner.adapter.workspaces[0] / "data.csv").read_text().startswith("id,")

    def test_the_skill_is_copied_into_each_home(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.run(spec, item, 1)

        copied = runner.adapter.homes[0] / "skills" / "tally-report" / "SKILL.md"
        assert copied.exists()
        (run,) = store.runs(limit=5)
        assert "layout" not in store.get_launch(run.launch_id).settings  # an adapter without layouts

    def test_a_skill_goes_where_its_adapter_lays_it_out_and_the_launch_says_how(self, store, tmp_path):
        """Laid out as installed, a skill sits where an installation keeps it — Hermes, in the
        category folder it declares — and the launch records the layout, so a fork of the
        run is laid out the same way."""

        class Installed(RecordingHarness):
            layout = "install"

            def skill_folder(self, home, skill):
                return Path(home) / ".agent" / "skills" / "tools" / Path(skill).name

        spec, item = make_spec(tmp_path)
        runner = BatchRunner(store, Installed(GOOD_REPORT, store), base_url="http://127.0.0.1:8787/v1",
                             model="m", workdir=tmp_path / "batches", settle=0)
        runner.run(spec, item, 1)

        home = runner.adapter.homes[0]
        assert (home / ".agent" / "skills" / "tools" / "tally-report" / "SKILL.md").exists()
        assert not (home / "skills" / "tally-report").exists()
        (run,) = store.runs(limit=5)
        assert store.get_launch(run.launch_id).settings["layout"] == "install"

    def test_a_run_that_edits_the_skill_is_excluded(self, store, tmp_path):
        """Hermes tells the model to patch skills it dislikes; such a sample is void."""
        spec, item = make_spec(tmp_path)
        result = make_runner(store, tmp_path, MUTATES_SKILL).run(spec, item, 2)

        assert result.usable == []
        assert len(result.invalidated) == 2
        assert all("SKILL.md" in o.skills_changed[0] for o in result.invalidated)
        # The check passed, but the run still must not count.
        assert all(outcome.passed for outcome in result.outcomes)
        assert result.passes == 0

    def test_a_timeout_is_excluded(self, store, tmp_path):
        spec, item = make_spec(tmp_path, timeout=0.5)
        result = make_runner(store, tmp_path, SLEEPS).run(spec, item, 1)

        assert result.outcomes[0].timed_out
        assert result.usable == []

    def test_the_verdict_is_kept_with_the_recording(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store)

        result = runner.run(spec, item, 2)

        results = [store.get_run(run_id).result for run_id in result.run_ids]
        assert [r["passed"] for r in results] == [True, True]
        assert all(r["usable"] for r in results)
        assert pass_rate(results).format().startswith("2/2")

    def test_a_run_that_cannot_be_counted_says_so(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, MUTATES_SKILL)
        runner.adapter = RecordingHarness(MUTATES_SKILL, store)

        result = runner.run(spec, item, 1)

        stored = store.get_run(result.run_ids[0]).result
        assert stored["passed"] is True
        assert stored["usable"] is False
        assert stored["skills_changed"]
        assert pass_rate([stored]) is None

    def test_a_run_edited_at_a_breakpoint_is_not_counted(self, store, tmp_path):
        """Its verdict measures the edit as much as the skill: in Phase 3 a run passed only
        because its report was fixed at a breakpoint, and the batch counted it as a pass."""
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        edit = {"response": {"fields": ["choices"], "message_fields": ["tool_calls"]}}
        runner.adapter = RecordingHarness(GOOD_REPORT, store, edits=edit)

        result = runner.run(spec, item, 1)

        (outcome,) = result.outcomes
        assert outcome.passed
        assert outcome.edited_steps == [0]
        assert result.usable == []
        assert result.passes == 0
        stored = store.get_run(outcome.run_id).result
        assert stored["usable"] is False
        assert stored["edited_steps"] == [0]
        assert pass_rate([stored]) is None

    @pytest.mark.parametrize(
        ("check", "passed"),
        [
            ({"reply_contains": "61.5"}, True),
            ({"reply_contains": "61.50"}, False),
            ({"reply_matches": r"(?i)TOTAL\W+61\.5"}, True),
            ({"reply_matches": r"^Done$"}, False),
        ],
    )
    def test_a_check_can_judge_the_final_reply(self, store, tmp_path, check, passed):
        spec, item = make_spec(tmp_path, check=check)
        runner = make_runner(store, tmp_path, BAD_REPORT)
        runner.adapter = RecordingHarness(BAD_REPORT, store, reply="Done. Total: 61.5 over 4 rows.")

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert outcome.passed is passed
        assert outcome.check.returncode is None
        assert "the reply:\nDone. Total: 61.5 over 4 rows." in outcome.check.output
        stored = store.get_run(outcome.run_id).result
        assert stored["passed"] is passed and stored["check_output"] == outcome.check.output

    def test_what_the_check_printed_is_kept_as_it_was_masked_where_it_ran(self, store, tmp_path):
        """Masked once, where the check ran, before it was cut: masked again as the run's result is
        kept, a command's last password would fall too."""
        command = TWICE
        key = "sk-" "tfcheckprints4Q7xZ2mN8pL4vR6tY"
        spec, item = make_spec(tmp_path, check=f"echo '{command}'; echo 'key {key}'; exit 1")
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store)

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert key not in outcome.check.output and "-pZq8xK3vN5wLp" in outcome.check.output
        assert store.get_run(outcome.run_id).result["check_output"] == outcome.check.output

    def test_a_shell_check_finds_the_reply_in_a_file_beside_the_workspace(self, store, tmp_path):
        spec, item = make_spec(tmp_path, check='grep -q "Total: 61.5" "$TUNINGFORK_REPLY"')
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store, reply="Done. Total: 61.5")

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert outcome.passed
        workspace = runner.adapter.workspaces[0]
        assert (workspace.parent / "reply.txt").read_text() == "Done. Total: 61.5"
        assert not (workspace / "reply.txt").exists()

    def test_a_check_script_beside_the_spec_is_found_by_its_folder(self, store, tmp_path):
        """A check longer than a line lives beside its spec, where the agent cannot read it:
        `$TUNINGFORK_SPEC_DIR` names that folder."""
        checks = tmp_path / "checks"
        checks.mkdir()
        (checks / "report.py").write_text(
            "import json, sys; sys.exit(0 if json.load(open('report.json')).get('amount_total') == 61.5 else 1)\n",
            encoding="utf-8",
        )
        spec, item = make_spec(tmp_path, check='python3 "$TUNINGFORK_SPEC_DIR/checks/report.py"')
        runner = make_runner(store, tmp_path, GOOD_REPORT)

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert outcome.passed, outcome.check.output
        assert not list(runner.adapter.workspaces[0].rglob("report.py"))

    def test_a_run_that_ends_without_a_reply_gives_an_empty_one(self, store, tmp_path):
        spec, item = make_spec(tmp_path, check={"reply_contains": "61.5"})
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store)

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert outcome.passed is False
        assert "(none: the run ended without a reply)" in outcome.check.output
        assert (runner.adapter.workspaces[0].parent / "reply.txt").read_text() == ""

    def test_no_check_means_no_verdict_rather_than_a_failure(self, store, tmp_path):
        spec, item = make_spec(tmp_path, check=None)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store)

        result = runner.run(spec, item, 1)

        stored = store.get_run(result.run_ids[0]).result
        assert stored["passed"] is None
        assert pass_rate([stored]) is None

    def test_unclaimed_labels_do_not_leak_to_later_runs(self, store, tmp_path):
        """The scripted harness never reaches the proxy, so no run claims its label."""
        spec, item = make_spec(tmp_path)
        make_runner(store, tmp_path, GOOD_REPORT, recording=False).run(spec, item, 2)

        assert store.pending_labels() == []

    def test_a_replacement_skill_file_is_what_every_run_gets(self, store, tmp_path):
        """`batch --skill`: an edit that won a checkpoint A/B, confirmed with full runs."""
        spec, item = make_spec(tmp_path)
        (spec.resolved_skill_dir() / "helper.py").write_text("print()\n", encoding="utf-8")
        variant = tmp_path / "variants" / "SKILL.v2.md"
        variant.parent.mkdir()
        variant.write_text("# tally-report, v2\n", encoding="utf-8")

        baseline = make_runner(store, tmp_path, GOOD_REPORT)
        baseline.adapter = RecordingHarness(GOOD_REPORT, store)
        before = baseline.run(spec, item, 1)
        runner = BatchRunner(
            store, RecordingHarness(GOOD_REPORT, store), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", skill_file=variant, settle=0,
        )
        result = runner.run(spec, item, 2)

        for home in runner.adapter.homes:
            installed = home / "skills" / "tally-report"
            assert (installed / "SKILL.md").read_text(encoding="utf-8") == "# tally-report, v2\n"
            assert (installed / "helper.py").exists()  # the rest of the folder comes along
        source = spec.resolved_skill_dir() / "SKILL.md"
        assert source.read_text(encoding="utf-8") == "# tally-report\n"

        runs = [store.get_run(run_id) for run_id in result.run_ids]
        assert {run.parity["skill_file"] for run in runs} == {str(variant)}
        assert len({run.skill_version for run in runs}) == 1
        assert runs[0].skill_version != store.get_run(before.run_ids[0]).skill_version

    def test_a_replacement_needs_a_skill_folder_to_go_in(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        spec.skill_dir = None
        variant = tmp_path / "SKILL.v2.md"
        variant.write_text("v2", encoding="utf-8")
        runner = BatchRunner(
            store, ScriptedHarness(GOOD_REPORT), base_url="http://127.0.0.1:8787/v1",
            model="m", workdir=tmp_path / "batches", skill_file=variant, settle=0,
        )
        with pytest.raises(ValueError, match="skill folder"):
            runner.run(spec, item, 1)

    def test_each_run_records_the_frozen_skill_it_was_given(self, store, tmp_path):
        """What a later A/B swaps out, however the source has been edited since."""
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = RecordingHarness(GOOD_REPORT, store)
        result = runner.run(spec, item, 1)
        (spec.resolved_skill_dir() / "SKILL.md").write_text("# edited later\n", encoding="utf-8")

        snapshot = Path(store.get_run(result.run_ids[0]).parity["skill_snapshot_dir"])
        assert (snapshot / "SKILL.md").read_text(encoding="utf-8") == "# tally-report\n"

    def test_editing_the_skill_mid_batch_cannot_reach_later_runs(self, store, tmp_path):
        """The Phase 1 contamination: a fixture edited while its batch was running."""
        spec, item = make_spec(tmp_path)
        source = spec.resolved_skill_dir() / "SKILL.md"
        original = source.read_text(encoding="utf-8")

        def edit_after_first_run(outcome):
            if outcome.index == 0:
                source.write_text("# edited mid-batch\n", encoding="utf-8")

        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.run(spec, item, 3, on_progress=edit_after_first_run)

        copies = [home / "skills" / "tally-report" / "SKILL.md" for home in runner.adapter.homes]
        assert all(copy.read_text(encoding="utf-8") == original for copy in copies)

    def test_a_harness_that_is_not_installed_leaves_nothing_running(self, store, tmp_path):
        """Found in review, back when runs were claimed through a queue: a boundary queued
        for a harness that never started split the next batch's first run in two."""
        spec, item = make_spec(tmp_path)

        class Missing(ScriptedHarness):
            def launch(self, prompt, **kwargs):
                spec_ = super().launch(prompt, **kwargs)
                return LaunchSpec(argv=("no-such-harness-tf",), env=spec_.env, cwd=spec_.cwd)

        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = Missing(GOOD_REPORT)
        with pytest.raises(HarnessError, match="no-such-harness-tf"):
            runner.run(spec, item, 1)
        assert store.pending_labels() == []
        (home,) = runner.adapter.homes
        launch = store.get_launch(token_in(runner.adapter.base_urls[home]))
        assert launch.status == "interrupted"

    def test_labels_queued_for_other_runs_are_left_alone(self, store, tmp_path):
        """A batch's runs are named by their launches, so a person's `label-next` — or a
        run boundary a killed batch left behind — is neither taken nor thrown away."""
        spec, item = make_spec(tmp_path)
        store.begin_run("tally-report/1#dead.0", "tally-report/1")
        store.label_next("my next manual run")

        result = make_runner(store, tmp_path, GOOD_REPORT).run(spec, item, 1)

        assert store.pending_labels() == ["tally-report/1#dead.0", "my next manual run"]
        (outcome,) = result.outcomes
        assert store.get_run(outcome.label).id == outcome.run_id
        assert store.get_run(outcome.run_id).launch_id == outcome.launch_id

    def test_a_run_that_never_reached_the_proxy_is_not_counted(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT, recording=False)

        (outcome,) = runner.run(spec, item, 1).outcomes

        assert outcome.passed  # the check passed...
        assert not outcome.usable  # ...but nothing was measured
        assert outcome.exclusion == "no model call was recorded"

    def test_a_run_whose_model_call_failed_is_not_counted(self, store, tmp_path):
        spec, item = make_spec(tmp_path)

        class Failing(RecordingHarness):
            def launch(self, prompt, **kwargs):
                self.record(kwargs["home"], prompt, status=502, error="upstream_error")
                return ScriptedHarness.launch(self, prompt, **kwargs)

        runner = make_runner(store, tmp_path, BAD_REPORT)
        runner.adapter = Failing(BAD_REPORT, store)
        result = runner.run(spec, item, 1)

        (outcome,) = result.outcomes
        assert outcome.failed_steps == [0]
        assert result.usable == []
        assert store.get_run(outcome.run_id).result["usable"] is False

    def test_a_task_without_a_check_has_no_pass_rate_at_all(self, store, tmp_path):
        spec, item = make_spec(tmp_path, check=None)
        result = make_runner(store, tmp_path, GOOD_REPORT).run(spec, item, 2)
        assert result.checked is False

    def test_progress_callback_sees_each_run(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        seen = []
        make_runner(store, tmp_path, GOOD_REPORT).run(
            spec, item, 3, on_progress=seen.append
        )
        assert [outcome.index for outcome in seen] == [0, 1, 2]


class TestCancel:
    """A cancel kills what is running and keeps what finished. Found in review: a harness
    killed by a cancel returned like any other, and was judged — a failing check the skill
    never made, stored as a counted sample."""

    def launches(self, store) -> list[dict]:
        with store._lock:
            return [dict(row) for row in store._conn.execute("SELECT id, status FROM launches")]

    def test_a_harness_killed_by_a_cancel_is_never_judged(self, store, tmp_path):
        import threading

        from tuningfork.ops import Cancelled

        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, SLEEPS)
        seen = []
        threading.Timer(0.5, runner.cancel.cancel).start()

        with pytest.raises(Cancelled) as stopped:
            runner.run(spec, item, 1, on_progress=seen.append)

        assert stopped.value.partial == [] and seen == []
        (launch,) = self.launches(store)
        assert launch["status"] == "cancelled"
        (run,) = store.runs(limit=5)
        assert run.result["cancelled"] is True
        assert run.result["usable"] is False and run.result["passed"] is None
        assert pass_rate([run.result]) is None

    def test_runs_that_finished_are_kept_when_a_cancel_stops_the_rest(self, store, tmp_path):
        import threading

        from tuningfork.ops import Cancelled

        import dataclasses

        class FastThenSlow(RecordingHarness):
            started = 0
            lock = threading.Lock()

            def launch(self, prompt, **kwargs):
                with self.lock:
                    first = FastThenSlow.started == 0
                    FastThenSlow.started += 1
                spec = super().launch(prompt, **kwargs)
                script = GOOD_REPORT if first else SLEEPS
                return dataclasses.replace(spec, argv=("python3", "-c", script))

        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.adapter = FastThenSlow(GOOD_REPORT, store)
        runner.concurrency = 2
        seen = []
        threading.Timer(2.0, runner.cancel.cancel).start()

        with pytest.raises(Cancelled) as stopped:
            runner.run(spec, item, 2, on_progress=seen.append)

        (finished,) = stopped.value.partial
        assert finished.passed and finished.usable
        assert seen == [finished]
        assert sorted(launch["status"] for launch in self.launches(store)) == ["cancelled", "ended"]

    def interrupted(self, store, tmp_path, signum: int) -> int:
        """Send this process `signum` while the runner waits on a harness that names its own
        pid; return that pid. The harness runs in a session of its own, as every one does."""
        import os
        import threading
        import time

        pidfile = tmp_path / "harness.pid"
        script = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, script)

        def send() -> None:
            for _ in range(200):
                if pidfile.exists() and pidfile.read_text():
                    break
                time.sleep(0.05)
            os.kill(os.getpid(), signum)

        threading.Thread(target=send, daemon=True).start()
        with pytest.raises(KeyboardInterrupt):
            runner.run(spec, item, 1)
        return int(pidfile.read_text())

    def test_an_interrupt_kills_the_harness_it_was_waiting_on(self, store, tmp_path):
        """Ctrl-C reaches tuningfork alone, while it waits on the harness. Found in use: the
        harness left the live set before the interrupt's cleanup looked for it, and went on
        calling the model after `tuningfork batch` was gone."""
        import os
        import signal

        pid = self.interrupted(store, tmp_path, signal.SIGINT)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        (launch,) = self.launches(store)
        assert launch["status"] == "interrupted"

    def test_a_sigterm_stops_the_harness_as_an_interrupt_does(self, store, tmp_path):
        """A `kill` of tuningfork goes through the same cleanup, and the handler it had
        before comes back."""
        import os
        import signal

        class Terminated(Exception):
            """Raised by the test's own handler: what a SIGTERM does if the runner never
            installs its own — a failed test, rather than this process killed."""

        def ours(signum, frame):
            raise Terminated

        previous = signal.signal(signal.SIGTERM, ours)
        try:
            pid = self.interrupted(store, tmp_path, signal.SIGTERM)
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert signal.getsignal(signal.SIGTERM) is ours
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_a_cancel_before_a_run_starts_launches_nothing(self, store, tmp_path):
        from tuningfork.ops import Cancelled

        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, GOOD_REPORT)
        runner.cancel.cancel()

        with pytest.raises(Cancelled) as stopped:
            runner.run(spec, item, 2)

        assert stopped.value.partial == []
        assert self.launches(store) == []

    def test_a_shutdown_leaves_the_launch_interrupted(self, store, tmp_path):
        import threading

        from tuningfork.ops import Cancelled

        spec, item = make_spec(tmp_path)
        runner = make_runner(store, tmp_path, SLEEPS)
        threading.Timer(0.5, lambda: runner.cancel.cancel("shutdown")).start()

        with pytest.raises(Cancelled) as stopped:
            runner.run(spec, item, 1)

        assert stopped.value.reason == "shutdown"
        assert [launch["status"] for launch in self.launches(store)] == ["interrupted"]


class TestRunsAsPartOfSomethingLarger:
    """`BatchRunner.run`'s group, start, skill_dir, skill_snapshot and settings: what a chain
    asks of a batch's runs."""

    def test_a_given_skill_folder_and_a_group_number_the_runs_in_order(self, store, tmp_path):
        spec, item = make_spec(tmp_path)
        other = tmp_path / "other-skill"
        other.mkdir()
        (other / "SKILL.md").write_text("# other\n", encoding="utf-8")
        runner = make_runner(store, tmp_path, GOOD_REPORT)

        result = runner.run(
            spec, item, 2, group="grp12345", start=3, skill_dir=other, skill_snapshot="skill-snapshot-3",
            settings={"chain": {"id": "grp12345", "link": 3}},
        )

        assert [outcome.index for outcome in result.outcomes] == [3, 4]
        launch = store.get_launch(store.get_run(result.outcomes[0].run_id).launch_id)
        assert launch.label == "tally-report/1#grp12345.3"
        assert launch.settings["chain"] == {"id": "grp12345", "link": 3} and launch.settings["breakpoints"]
        snapshot = (tmp_path / "batches" / "grp12345" / "skill-snapshot-3" / "other-skill").resolve()
        assert Path(launch.skill_dir) == snapshot
        assert (runner.adapter.homes[0] / "skills" / "other-skill" / "SKILL.md").read_text() == "# other\n"
        assert runner.adapter.workspaces[0] == tmp_path / "batches" / "grp12345" / "3" / "workspace"
        assert store.get_run(result.outcomes[0].run_id).parity["skill_snapshot_dir"] == str(snapshot)

    def test_an_adapters_settings_are_recorded_with_the_launch(self, store, tmp_path):
        class Configured(RecordingHarness):
            settings = {"harness": {"await_review": True}}

        spec, item = make_spec(tmp_path)
        runner = BatchRunner(store, Configured(GOOD_REPORT, store), base_url="http://127.0.0.1:8787/v1",
                             model="m", workdir=tmp_path / "batches", settle=0)
        (outcome,) = runner.run(spec, item, 1).outcomes
        launch = store.get_launch(store.get_run(outcome.run_id).launch_id)
        assert launch.settings["harness"] == {"await_review": True}


@pytest.mark.parametrize("changes, why", [
    ({}, None),
    ({"skills_changed": ["SKILL.md"]}, "skills changed: SKILL.md"),
    ({"timed_out": True}, "timed out"),
    ({"edited_steps": [2]}, "edited at a breakpoint (step 2)"),
    ({"failed_steps": [0, 3]}, "model call failed (step 0, 3)"),
    ({"thread_trouble": ["subagent 1a2b3c4d step 1"]}, "a call failed or was edited in subagent 1a2b3c4d step 1"),
    ({"fork_step": 2, "departed": {"step": 1, "thread": "side"}}, "the replay departed from the recording at thread side step 1"),
    ({"fork_step": 2, "leaked": [2]}, "the original wording reached the model (step 2)"),
    ({"fork_step": 2, "reached": False}, "the harness never reached step 2"),
])
def test_why_a_run_is_not_counted_reads_the_same_from_what_is_stored(changes, why):
    """A run's page says why its verdict is left out from the result the store kept: the same
    words the batch said when it ran."""
    from tuningfork.analysis.conditions import exclusion_from
    from tuningfork.runner import RunOutcome
    from tuningfork.specs import CheckResult

    outcome = RunOutcome(index=0, label="t#b.0", run_id="r", passed=False, check=CheckResult(False, "KeyError"),
                         duration_s=1.0, returncode=0, stdout="", stderr="", **changes)

    assert outcome.exclusion == why
    assert exclusion_from(outcome.to_result()) == why
    assert outcome.to_result()["usable"] is (why is None)
