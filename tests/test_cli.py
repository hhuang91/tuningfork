"""The CLI, end to end against the mock server.

These exist because the engine underneath being correct did not stop the CLI printing model
output unescaped: Rich parses square brackets as markup, and payloads are full of them —
JSON arrays, and Hermes's own `[IMPORTANT: ...]` markers. Only running the real commands
over realistic text catches that.
"""

from __future__ import annotations

import json

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.store.db import Store
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot, record_run, text_message
from tests.mock_server import MockTurn

UPSTREAM = "http://upstream.test"


@pytest.fixture
def cli(monkeypatch):
    # Wide enough that Rich does not wrap table cells the assertions look for. The CLI's
    # console is replaced, not just COLUMNS set: Rich fixes a console's width when it is
    # created if COLUMNS is exported then, so a shell exporting it would fail these tests.
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    runner = CliRunner()

    def invoke(*args: str):
        result = runner.invoke(app, list(args))
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            result.output
        )
        return result

    return invoke


@pytest.fixture
def home(tmp_path):
    path = tmp_path / "store"
    store = Store(path)
    run_id = record_pilot(store)
    store.close()
    return path, run_id


@pytest.fixture
def served(monkeypatch, mock_server):
    """Route the engine's own HTTP client to the mock server."""
    mock_server.props = PROPS
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


def write_keys(keys: dict) -> MockTurn:
    return MockTurn(
        tool_calls=[
            {
                "name": "write_file",
                "arguments": {"path": "report.json", "content": json.dumps(keys)},
            }
        ]
    )


class TestResample:
    def test_draws_samples_and_prints_the_distribution(self, cli, home, served):
        path, run_id = home
        read = MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": "data.csv"}}])
        served.script(read, read, read, MockTurn(content="I will ask first."))

        result = cli("resample", run_id[:8], "0", "-n", "4",
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "3/4" in result.output
        assert "read_file(path='data.csv')" in result.output
        assert "I will ask first." in result.output

    def test_square_brackets_in_model_output_survive(self, cli, home, served):
        """The escaping bug: `[1, 2]` would otherwise be eaten as Rich markup."""
        path, run_id = home
        served.script(write_keys({"values": [1, 2]}), write_keys({"values": [1, 2]}))

        result = cli("resample", run_id[:8], "1", "-n", "2",
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert '{"values":[1,2]}' in result.output

    def test_a_missing_step_fails_cleanly(self, cli, home, served):
        path, run_id = home
        result = cli("resample", run_id[:8], "9", "--home", str(path))
        assert result.exit_code == 1
        assert "has no step 9" in result.output


class TestAB:
    @staticmethod
    def wording_sensitive(payload: dict) -> MockTurn:
        if "row_count and amount_total" in payload["messages"][0]["content"]:
            return write_keys({"row_count": 4, "amount_total": 61.5})
        return write_keys({"count": 4, "total": 61.5})

    @pytest.fixture
    def skill_files(self, tmp_path):
        folder = tmp_path / "skill"
        folder.mkdir()
        (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")
        return folder

    def test_a_declared_target_gets_a_verdict(self, cli, home, served, skill_files):
        path, run_id = home
        served.responder = self.wording_sensitive

        result = cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
                     "-n", "6", "--target-match", "row_count",
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "found: full, lines 1-7" in result.output
        assert "declared before sampling" in result.output
        assert "significant" in result.output and "not significant" not in result.output
        assert "0/6" in result.output and "6/6" in result.output

    def test_a_described_target_gets_a_verdict(self, cli, home, served, skill_files):
        """A reply can mention a key without writing it: a target can ask for both."""
        path, run_id = home
        served.responder = self.wording_sensitive
        rules = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]

        result = cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
                     "-n", "6", "--target-rules", json.dumps(rules),
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert 'calls write_file, and mentions "row_count"' in result.output
        assert "0/6" in result.output and "6/6" in result.output
        refused = cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
                      "--target-rules", "not json", "--base-url", UPSTREAM, "--home", str(path))
        assert refused.exit_code == 1 and "not JSON" in refused.output

    def test_without_a_target_the_result_is_labelled_exploratory(
        self, cli, home, served, skill_files
    ):
        path, run_id = home
        served.responder = self.wording_sensitive

        result = cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
                     "-n", "4", "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "exploratory" in result.output
        assert "Bonferroni" in result.output

    def test_a_skill_absent_from_the_context_is_diagnosed_before_sampling(
        self, cli, home, served, tmp_path
    ):
        path, run_id = home
        stray = tmp_path / "other"
        stray.mkdir()
        (stray / "SKILL.md").write_text("# Another skill\n\nNot loaded.\n", encoding="utf-8")
        (stray / "SKILL.v2.md").write_text("# Another skill\n\nStill not.\n", encoding="utf-8")

        result = cli("ab", run_id[:8], "1", "--variant", str(stray / "SKILL.v2.md"),
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 1
        assert "not in the context" in result.output
        assert served.requests == []

    def test_the_original_defaults_to_the_copy_the_batch_gave_the_run(
        self, cli, home, served, skill_files, tmp_path
    ):
        """The variant need not sit beside the skill — and should not: every run gets a
        copy of the skill's whole folder."""
        path, run_id = home
        snapshot = tmp_path / "batch" / "skill-snapshot" / "tally-report"
        snapshot.mkdir(parents=True)
        (snapshot / "SKILL.md").write_text(SKILL, encoding="utf-8")
        store = Store(path)
        store.annotate_run(run_id, parity={"skill_snapshot_dir": str(snapshot)})
        store.close()
        elsewhere = tmp_path / "variants"
        elsewhere.mkdir()
        (elsewhere / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")
        served.responder = self.wording_sensitive

        result = cli("ab", run_id[:8], "1", "--variant", str(elsewhere / "SKILL.v2.md"),
                     "-n", "2", "--target-match", "row_count",
                     "--base-url", UPSTREAM, "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "skill-snapshot" in result.output

    def test_a_missing_original_says_how_to_name_it(self, cli, home, served, tmp_path):
        path, run_id = home
        (tmp_path / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")
        result = cli("ab", run_id[:8], "1", "--variant", str(tmp_path / "SKILL.v2.md"),
                     "--base-url", UPSTREAM, "--home", str(path))
        assert result.exit_code == 1
        assert "name the original" in result.output

    def test_a_letter_target_needs_an_earlier_resample(self, cli, home, served, skill_files):
        path, run_id = home
        result = cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
                     "--target", "A", "--base-url", UPSTREAM, "--home", str(path))
        assert result.exit_code == 1
        assert "Run a resample first" in result.output

    def test_a_stored_ab_can_be_reported_again(self, cli, home, served, skill_files):
        path, run_id = home
        served.responder = self.wording_sensitive
        cli("ab", run_id[:8], "1", "--variant", str(skill_files / "SKILL.v2.md"),
            "-n", "4", "--target-match", "row_count",
            "--base-url", UPSTREAM, "--home", str(path))

        listing = cli("experiments", "--home", str(path))
        experiment_id = listing.output.split("\n")[1].split()[0]
        again = cli("experiment", experiment_id, "--home", str(path))

        assert again.exit_code == 0, again.output
        # The target declared at the time is the one re-tested, not a new choice.
        assert "row_count/" in again.output
        assert "declared before sampling" in again.output


class TestReanalysis:
    def test_a_stored_resample_can_be_viewed_through_another_lens(self, cli, home, served):
        """Different programs are different clusters by default, one under tool_name."""
        path, run_id = home
        served.script(
            *[MockTurn(tool_calls=[{"name": "execute_code", "arguments": {"code": f"x = {i}"}}])
              for i in range(3)]
        )
        cli("resample", run_id[:8], "0", "-n", "3", "--base-url", UPSTREAM, "--home", str(path))
        experiment_id = cli("experiments", "--home", str(path)).output.split("\n")[1].split()[0]

        coarse = cli("experiment", experiment_id, "--normalizer", "tool_name",
                     "--home", str(path))

        assert coarse.exit_code == 0, coarse.output
        assert "3/3" in coarse.output
        assert "execute_code(…)" in coarse.output

    def test_a_key_target_is_not_carried_into_a_different_lens(
        self, cli, home, served, tmp_path
    ):
        path, run_id = home
        folder = tmp_path / "skill"
        folder.mkdir()
        (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")
        served.responder = TestAB.wording_sensitive
        cli("resample", run_id[:8], "1", "-n", "2", "--base-url", UPSTREAM, "--home", str(path))
        cli("ab", run_id[:8], "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "2",
            "--target", "A", "--base-url", UPSTREAM, "--home", str(path))
        ab_id = next(
            line.split()[0]
            for line in cli("experiments", "--home", str(path)).output.split("\n")
            if " ab " in line
        )

        result = cli("experiment", ab_id, "--normalizer", "tool_name", "--home", str(path))

        assert "cannot be re-tested under a different lens" in result.output
        assert "exploratory" in result.output

    def test_an_unknown_lens_is_refused(self, cli, home):
        path, _ = home
        result = cli("experiment", "anything", "--normalizer", "vibes", "--home", str(path))
        assert result.exit_code == 1
        assert "unknown normalizer" in result.output


class TestInspection:
    def test_step_prints_harness_markers_literally(self, cli, home):
        path, run_id = home
        result = cli("step", run_id[:8], "0", "--full", "--home", str(path))
        assert result.exit_code == 0, result.output
        assert "[IMPORTANT: The user launched" in result.output

    def test_divergence_warns_when_runs_used_different_skills(self, cli, tmp_path):
        path = tmp_path / "mixed"
        store = Store(path)
        ids = [record_run(store, f"r{i}", [text_message("x")], task_id="t/1") for i in range(3)]
        for run_id, digest in zip(ids, ("aaa", "aaa", "bbb")):
            store.annotate_run(run_id, skill_hashes={"SKILL.md": digest})
        store.close()

        result = cli("divergence", "--task", "t/1", "--home", str(path))

        assert "2 different versions of the skill" in result.output

    def test_divergence_says_which_behaviour_went_on_to_pass(self, cli, tmp_path):
        path = tmp_path / "verdicts"
        store = Store(path)
        for index, (last, passed) in enumerate([("left", True), ("left", True), ("right", False)]):
            run_id = record_run(
                store, f"r{index}", [text_message("same"), text_message(last)], task_id="t/2"
            )
            store.annotate_run(run_id, result={"passed": passed, "usable": True})
        store.close()

        result = cli("divergence", "--task", "t/2", "--home", str(path))

        assert "then passed" in result.output
        assert "2/2" in result.output
        assert "0/1" in result.output

    def test_tasks_give_each_wording_its_own_rate(self, cli, tmp_path):
        path = tmp_path / "conditions"
        store = Store(path)
        for index in range(4):
            run_id = record_run(store, f"r{index}", [text_message("x")], task_id="t/4")
            v2 = index >= 2
            store.annotate_run(
                run_id, harness="hermes", skill_hashes={"SKILL.md": "v2" if v2 else "v1"},
                parity={"skill_file": "variants/SKILL.v2.md"} if v2 else {},
                result={"passed": v2, "usable": True},
            )
        store.close()

        result = cli("tasks", "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "t/4 · hermes · m" in result.output  # one table for the task, harness, model
        lines = [line for line in result.output.splitlines() if "/2 " in line]
        assert len(lines) == 2
        assert "0/2" in lines[0] and "2/2" in lines[1]
        assert "SKILL.md from SKILL.v2.md" in lines[1]

    def test_a_file_only_some_runs_were_given_does_not_name_the_whole_row(self, cli, tmp_path):
        # Found in the guide's capture: a validation's original arm installs a copy of the
        # skill's own text, and joined the batch run with the folder's SKILL.md under one
        # fingerprint — and the row read "SKILL.md from baseline.md" for all of them.
        path = tmp_path / "mixed"
        store = Store(path)
        for index in range(3):
            run_id = record_run(store, f"m{index}", [text_message("x")], task_id="t/7")
            store.annotate_run(
                run_id, harness="hermes", skill_hashes={"SKILL.md": "v1"},
                parity={"skill_file": "refinements/r/baseline.md"} if index == 2 else {},
                result={"passed": False, "usable": True},
            )
        store.close()

        result = cli("tasks", "--home", str(path))

        assert result.exit_code == 0, result.output
        assert "0/3" in result.output
        assert "baseline.md" not in result.output

    def test_runs_quiet_prints_only_ids_newest_first(self, cli, tmp_path):
        path = tmp_path / "quiet"
        store = Store(path)
        ids = [record_run(store, f"q{i}", [text_message(f"x{i}")], task_id="t/5") for i in range(3)]
        store.close()

        result = cli("runs", "-q", "-n", "2", "--home", str(path))

        assert result.output.split() == ids[::-1][:2]

    def test_divergence_through_a_lens(self, cli, tmp_path):
        path = tmp_path / "lens"
        store = Store(path)
        for index, target in enumerate(("a.csv", "b.csv")):
            call = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": target})}}]}
            record_run(store, f"l{index}", [call], task_id="t/6")
        store.close()

        plain = cli("divergence", "--task", "t/6", "--home", str(path))
        lens = cli("divergence", "--task", "t/6", "--normalizer", "tool_name", "--home", str(path))
        unknown = cli("divergence", "--task", "t/6", "--normalizer", "nope", "--home", str(path))

        assert "diverged at step 0" in plain.output
        assert "compared under tool_name" in lens.output
        assert "behaved identically" in lens.output
        assert unknown.exit_code == 1 and "unknown normalizer" in unknown.output

    def test_a_run_recorded_live_says_so_where_a_label_would_name_it(self, cli, tmp_path):
        """A harness's calls through the proxy, nothing launching it: no label of its own, and
        the listing and its page say how it was recorded instead."""
        from tests.helpers import as_response

        path = tmp_path / "live"
        store = Store(path)
        step = store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                                 response=as_response(text_message("hello")))
        assert store.get_run(step.run_id).recorded == "live" and store.get_run(step.run_id).label is None
        store.close()

        listed = cli("runs", "--home", str(path))
        shown = cli("show", step.run_id[:8], "--home", str(path))
        assert listed.exit_code == 0 and "(recorded live)" in listed.output
        assert shown.exit_code == 0 and "(recorded live)" in shown.output.splitlines()[0]

    def test_runs_show_the_check_verdict(self, cli, tmp_path):
        path = tmp_path / "listed"
        store = Store(path)
        passing = record_run(store, "p", [text_message("x")], task_id="t/3")
        store.annotate_run(passing, result={"passed": True, "usable": True})
        void = record_run(store, "v", [text_message("x")], task_id="t/3")
        store.annotate_run(void, result={"passed": False, "usable": False})
        store.close()

        result = cli("runs", "--home", str(path))

        assert "pass" in result.output
        assert "fail (not counted)" in result.output


class TestStepsByTool:
    """A step can be named by the tool its model called: "the write" is how a person thinks
    of it, and its number depends on the path the run took."""

    def test_the_first_step_calling_a_tool(self, cli, home):
        path, run_id = home
        by_name = cli("step", run_id[:8], "write_file", "--home", str(path))
        by_number = cli("step", run_id[:8], "1", "--home", str(path))
        assert by_name.exit_code == 0, by_name.output
        assert "step 1" in by_name.output
        assert by_name.output == by_number.output

    def test_resample_takes_one_too(self, cli, home, served):
        path, run_id = home
        served.script(*[write_keys({"count": 4})] * 2)
        result = cli("resample", run_id[:8], "write_file", "-n", "2",
                     "--base-url", UPSTREAM, "--home", str(path))
        assert result.exit_code == 0, result.output
        assert "step 1" in result.output

    def test_a_tool_never_called_is_refused(self, cli, home):
        path, run_id = home
        result = cli("step", run_id[:8], "Write", "--home", str(path))
        assert result.exit_code == 1
        assert "no step where the model called Write" in result.output


def test_runs_can_be_chosen_by_a_tool_they_called(cli, tmp_path):
    """Found on a real Hermes batch: the latest run wrote its report with a script, so it had
    no write_file step to resample. `--called` picks a run that did."""
    from tests.helpers import tool_call_message

    path = tmp_path / "called"
    store = Store(path)
    wrote = record_run(store, "w", [tool_call_message("write_file", {"path": "r.json"})], task_id="t/7")
    record_run(store, "s", [tool_call_message("execute_code", {"code": "open('r.json','w')"})], task_id="t/7")
    store.close()

    newest = cli("runs", "-q", "-n", "1", "--home", str(path)).output.split()
    chosen = cli("runs", "-q", "-n", "1", "--called", "write_file", "--home", str(path)).output.split()
    refused = cli("step", newest[0][:8], "write_file", "--home", str(path))

    assert chosen == [wrote] != newest
    assert refused.exit_code == 1 and "runs --called write_file" in refused.output


def test_runs_can_be_chosen_by_what_they_called_at_a_step(cli, tmp_path):
    """Found writing the guide: runs split at step 0 between reading the file, searching for
    it and running code, and `--called search_files` picked a run that ran code first and
    searched later. `--at-step` picks a run by what it did at the step being studied."""
    from tests.helpers import tool_call_message

    path = tmp_path / "at-step"
    store = Store(path)
    searched_first = record_run(store, "a", [
        tool_call_message("search_files", {"pattern": "data.csv"}),
        tool_call_message("read_file", {"path": "data.csv"}),
    ], task_id="t/7")
    searched_later = record_run(store, "b", [
        tool_call_message("execute_code", {"code": "print(1)"}),
        tool_call_message("search_files", {"pattern": "data.csv"}),
    ], task_id="t/7")
    store.close()

    anywhere = cli("runs", "-q", "--called", "search_files", "--home", str(path)).output.split()
    at_zero = cli("runs", "-q", "--called", "search_files", "--at-step", "0", "--home", str(path)).output.split()
    beyond = cli("runs", "-q", "--called", "search_files", "--at-step", "5", "--home", str(path)).output.split()
    alone = cli("runs", "--at-step", "0", "--home", str(path))

    assert anywhere == [searched_later, searched_first]
    assert at_zero == [searched_first]
    assert beyond == []
    assert alone.exit_code == 1 and "--called" in alone.output


def test_runs_can_be_chosen_by_their_verdict(cli, tmp_path):
    """Found writing the design-md guide: the run to study is one the check failed, and
    nothing picked one. A run not counted — a timeout — has no verdict to choose by."""
    from tests.helpers import tool_call_message

    path = tmp_path / "verdicts"
    store = Store(path)
    failed = record_run(store, "f", [tool_call_message("write_file", {"path": "a"})], task_id="t/1")
    passed = record_run(store, "p", [tool_call_message("write_file", {"path": "b"})], task_id="t/1")
    timed_out = record_run(store, "x", [tool_call_message("write_file", {"path": "c"})], task_id="t/1")
    unjudged = record_run(store, "u", [tool_call_message("write_file", {"path": "d"})], task_id="t/1")
    store.annotate_run(failed, result={"passed": False, "usable": True})
    store.annotate_run(passed, result={"passed": True, "usable": True})
    store.annotate_run(timed_out, result={"passed": False, "usable": False})
    store.close()

    newest_failed = cli("runs", "-q", "-n", "1", "--failed", "--home", str(path)).output.split()
    every_passed = cli("runs", "-q", "--passed", "--home", str(path)).output.split()
    both = cli("runs", "--failed", "--passed", "--home", str(path))

    assert newest_failed == [failed]  # the newest run is unjudged, and the one before it not counted
    assert every_passed == [passed]
    assert both.exit_code == 1 and "--failed or --passed" in both.output
    assert unjudged not in newest_failed + every_passed


def test_serve_says_where_it_forwards_without_a_password(cli, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tuningfork.config import REDACTED
    from tests.helpers import KEY

    class Server:
        def __init__(self, app, *, host, port):
            pass

        def run(self):
            pass

    def create_app(settings):
        return SimpleNamespace(settings=settings, state=SimpleNamespace(access=SimpleNamespace(token="t")))

    monkeypatch.setattr("tuningfork.proxy.app.create_app", create_app)
    monkeypatch.setattr("tuningfork.proxy.app.ProxyServer", Server)
    result = cli("serve", "--upstream", f"http://tuningfork:{KEY}@127.0.0.1:1", "--home", str(tmp_path / "home"))
    assert KEY not in result.output and f"forwarding to http://{REDACTED}@127.0.0.1:1" in result.output


def test_serve_reads_its_settings_from_the_environment(cli, tmp_path, monkeypatch):
    """`TUNINGFORK_HOST`, `TUNINGFORK_PORT` and `TUNINGFORK_MAX_INFLIGHT` were read only by a
    settings function nothing called, so setting them changed nothing."""
    from types import SimpleNamespace

    seen = {}

    class Server:
        def __init__(self, app, *, host, port):
            seen.update(settings=app.settings, host=host, port=port)

        def run(self):
            pass

    def create_app(settings):
        return SimpleNamespace(settings=settings, state=SimpleNamespace(access=SimpleNamespace(token=settings.token)))

    monkeypatch.setattr("tuningfork.proxy.app.create_app", create_app)
    monkeypatch.setattr("tuningfork.proxy.app.ProxyServer", Server)
    monkeypatch.setenv("TUNINGFORK_HOST", "127.0.0.2")
    monkeypatch.setenv("TUNINGFORK_PORT", "9797")
    monkeypatch.setenv("TUNINGFORK_MAX_INFLIGHT", "3")
    monkeypatch.setenv("TUNINGFORK_TOKEN", "pinned")

    result = cli("serve", "--upstream", "http://127.0.0.1:1", "--home", str(tmp_path / "home"))

    assert (seen["host"], seen["port"], seen["settings"].max_inflight) == ("127.0.0.2", 9797, 3)
    # The token can be pinned, and the address printed carries it.
    assert seen["settings"].token == "pinned"
    assert "http://127.0.0.2:9797/_tuningfork/ui/?token=pinned" in result.output
