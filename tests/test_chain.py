"""A chain: a task run N times in order, each run given the skill the run before left.

End to end with no model: `FakeAgent` is a real harness process talking to a real proxy, and
the mock server plays a model that improves its own skill on every run (see
`tests.helpers.task_stack`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.config import REDACTED
from tuningfork.control import ControlClient
from tuningfork.demo.mock import MockTurn
from tuningfork.ops import Cancel, Cancelled, OperationError
from tuningfork.ops.chain import execute_chain, get_chain, list_chains, prepare_chain
from tests.helpers import FakeAgent, task_stack


def self_improving(payload: dict) -> MockTurn:
    """The model: write the report, and patch its own skill with what it learned — one line
    more each run — then say so. The skill lives in the run's home, beside the workspace."""
    messages = payload["messages"]
    system = messages[0]["content"]
    if messages[1]["content"].startswith("I ran my check"):
        return MockTurn(content="Noted: " + messages[1]["content"].splitlines()[0])
    workdir = re.search(r"Working directory: (\S+)", system).group(1)
    skill = re.search(r"\[Skill tally-report preloaded\]\n\n(.*)", system, re.S).group(1)
    results = [m for m in messages if m["role"] == "tool"]
    if results:
        return MockTurn(content="The report is written, and the skill improved.")
    home = Path(workdir).parent / "home"
    learned = skill.count("learned:") + 1
    return MockTurn(tool_calls=[
        {"name": "write_file", "arguments": {
            "path": f"{workdir}/report.json", "content": json.dumps({"row_count": 4, "amount_total": 61.5})}},
        {"name": "write_file", "arguments": {
            "path": str(home / "skills" / "tally-report" / "SKILL.md"),
            "content": skill.rstrip("\n") + f"\nlearned: round {learned}\n"}},
    ])


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        mock_server.responder = self_improving
        # The fake agent edits a skill with its one writing tool; a chain then knows who did.
        monkeypatch.setattr(FakeAgent, "skill_tools", frozenset({"write_file"}), raising=False)
        yield built


def prepare(stack, **options):
    options.setdefault("model", "m")
    options.setdefault("harness", "fake")
    options.setdefault("n", 3)
    return prepare_chain(
        stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "chains", **options,
    )


class TestChain:
    def test_each_link_starts_from_the_skill_the_one_before_left(self, stack):
        plan = prepare(stack)
        events = []
        report = execute_chain(stack.store, plan, progress=events.append)

        links = report.links
        assert [link.index for link in links] == [0, 1, 2]
        # Every run patched the skill, so none counts toward a rate; each still has a verdict.
        assert all(link.passed and not link.usable for link in links)
        assert all(link.exclusion.startswith("skills changed") for link in links)
        assert all(link.changed == ["tally-report/SKILL.md"] for link in links)
        # The skill handed on: what one link left is what the next was given.
        assert all(link.skill_after != link.skill_before for link in links)
        assert links[1].skill_before == links[0].skill_after
        assert links[2].skill_before == links[1].skill_after
        final = (report.final_skill / "SKILL.md").read_text(encoding="utf-8")
        assert final.count("learned:") == 3 and final.endswith("learned: round 3\n")
        assert [version["from"] for version in report.versions] == ["start", "link", "link", "link"]
        assert [version.get("link") for version in report.versions][1:] == [0, 1, 2]
        # Who changed it: the task's own conversation, with the tool the adapter names.
        assert all(link.callers == ["task"] for link in links)
        assert links[0].edits[0]["tool"] == "write_file" and links[0].edits[0]["step"] == 0
        assert all(link.review is None for link in links)
        assert all(link.diff and link.diff["added"] == 1 for link in links)
        assert report.status == "done" and report.counted == []
        kinds = [event.kind for event in events]
        assert kinds.count("link") == 3 and kinds.count("run") == 3 and kinds.count("launch") == 3

    def test_a_link_deleted_for_good_leaves_the_skill_it_left(self, stack):
        # What a link left is the chain's: the next link's start, and the last link's the
        # chain's final skill. Found in review: kept in the link's launch folder, it went when
        # the link's run was deleted for good, the chain's final skill with the last link's.
        from tuningfork.ops import trash

        report = execute_chain(stack.store, prepare_chain(
            stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.store.home / "batches",
            model="m", harness="fake", n=2,
        ))
        for link in report.links:
            trash.trash(stack.store, "run", link.run_id)
            assert trash.purge(stack.store, "run", link.run_id).to_json()["folders"] == 1
        assert all((Path(link.left) / "SKILL.md").is_file() for link in report.links)
        assert (report.final_skill / "SKILL.md").read_text(encoding="utf-8").endswith("learned: round 2\n")

    def test_the_links_are_launches_of_one_group_that_say_which_chain(self, stack):
        plan = prepare(stack, n=2)
        report = execute_chain(stack.store, plan)
        for link in report.links:
            launch = stack.store.get_launch(link.launch_id)
            assert launch.label == f"tally-report/1#{plan.id}.{link.index}"
            assert launch.settings["chain"] == {"id": plan.id, "link": link.index, "tell": False}
            # The frozen copy each link was given is its own, under the chain's folder.
            assert Path(launch.skill_dir).parent.name == f"skill-snapshot-{link.index}"
            assert Path(launch.skill_dir).parent.parent.name == plan.id
        given = (Path(stack.store.get_launch(report.links[1].launch_id).skill_dir) / "SKILL.md").read_text()
        assert given.endswith("learned: round 1\n")

    def test_the_chain_is_kept_and_found_again(self, stack):
        plan = prepare(stack, n=2)
        report = execute_chain(stack.store, plan)
        kept = get_chain(stack.store, plan.id[:4])
        assert kept == json.loads(json.dumps(report.to_json()))
        assert kept["status"] == "done" and len(kept["links"]) == 2 and kept["not_counted"] == 2
        assert kept["rate"] is None and kept["final_skill"] == str(report.final_skill)
        assert [chain["id"] for chain in list_chains(stack.store)] == [plan.id]
        with pytest.raises(OperationError) as refused:
            get_chain(stack.store, "nope")
        assert refused.value.code == "not_found"

    def test_a_cancelled_chain_keeps_the_links_that_finished(self, stack):
        plan = prepare(stack, n=3)
        cancel = Cancel()

        def stop_after_one(event) -> None:
            if event.kind == "link":
                cancel.cancel("cancelled")

        with pytest.raises(Cancelled) as stopped:
            execute_chain(stack.store, plan, progress=stop_after_one, cancel=cancel)
        partial = stopped.value.partial
        assert partial.status == "cancelled" and len(partial.links) == 1
        assert get_chain(stack.store, plan.id)["status"] == "cancelled"
        assert (partial.final_skill / "SKILL.md").read_text().endswith("learned: round 1\n")

    def test_a_chain_needs_a_skill_folder_and_a_link(self, stack):
        data = stack.spec.to_dict()
        del data["tuningfork"]
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            prepare(stack)
        assert refused.value.field == "spec" and "skill folder" in refused.value.message
        with pytest.raises(OperationError) as refused:
            prepare(stack, n=0)
        assert refused.value.field == "n"

    def test_the_plan_says_what_a_chain_is_and_what_its_harness_needs(self, stack, monkeypatch):
        monkeypatch.setattr(FakeAgent, "chain_notes", lambda self: ["the review is off"], raising=False)
        plan = prepare(stack)
        data = json.loads(json.dumps(plan.to_json()))
        assert data["kind"] == "chain" and data["chain"] == plan.id and data["n"] == 3
        notes = [note["message"] for note in data["notes"]]
        assert any("not independent samples" in note for note in notes)
        assert notes[-1] == "the review is off"
        assert data["concurrency"] == 1

    def test_a_told_chain_gives_the_harness_the_checks_result_as_one_more_turn(self, stack):
        plan = prepare(stack, n=2, tell=True)
        assert plan.tell and plan.to_json()["told_template"].startswith("I ran my check")
        assert any("told the check's verdict" in note.message for note in plan.notes)
        report = execute_chain(stack.store, plan)

        for link in report.links:
            assert link.passed is True and link.told["returncode"] == 0 and not link.told["timed_out"]
            assert link.told["message"] == "I ran my check on what you produced. It passed."
            launch = stack.store.get_launch(link.launch_id)
            assert launch.settings["chain"]["tell"] is True and launch.exit["told"] == link.told
            # The task's two calls, then the told turn's one: all the launch's, every thread.
            steps = sum(len(stack.store.steps(run.id)) for run in stack.store.launch_runs(launch.id))
            assert steps == 3
            told_step = next(
                step for run in stack.store.launch_runs(launch.id) for step in stack.store.steps(run.id)
                if "I ran my check" in json.dumps(stack.store.rebuild_request(step))
            )
            reply = stack.store.response(told_step)["choices"][0]["message"]["content"]
            assert reply == "Noted: I ran my check on what you produced. It passed."
        kept = get_chain(stack.store, plan.id)
        assert kept["tell"] is True and kept["links"][0]["told"]["message"].endswith("It passed.")

    def test_the_harness_is_told_what_the_check_printed_masked(self, stack):
        """Masked where the check ran: the harness, its command line and its own session files are
        given the masked text, and the run's result and the chain keep it as it was told."""
        from tuningfork.demo.task import CHECK

        key = "sk-" "tftoldkey4Q7xZ2mN8pL4vR6tY0a"
        command = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["check"] = f"echo 'the key is {key}'; echo '{command}'; {CHECK}"
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        report = execute_chain(stack.store, prepare(stack, n=1, tell=True))

        (link,) = report.links
        assert "It passed.\n\nthe key is <redacted>\n" in link.told["message"]
        sent = [payload for payload in stack.mock.requests
                if str(payload["messages"][-1].get("content") or "").startswith("I ran my check")]
        assert len(sent) == 1 and key not in json.dumps(sent) and "<redacted>" in json.dumps(sent)
        # Kept as it was told: masked again, the command's second password would fall too.
        assert "-pZq8xK3vN5wLp" in link.told["message"]
        assert stack.store.get_run(link.run_id).result["told"] == link.told
        assert stack.store.get_launch(link.launch_id).exit["told"] == link.told

    def test_the_chains_record_is_masked_but_what_was_masked_where_it_was_made(self, stack):
        """The lines a link wrote into its skill are masked in the chain's record; the copy it left,
        the next link's start, keeps them. What its harness was told, masked where the check ran,
        and its skill-editing calls, read from the store, are kept as they were masked: masked
        again, a command's second password would fall too."""
        from tuningfork.demo.task import CHECK

        key = "sk-" "tfchainkey7Hq2Lm9Xw3Rv5Pz8B"
        command = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"

        def noting(payload: dict) -> MockTurn:
            messages = payload["messages"]
            if messages[1]["content"].startswith("I ran my check"):
                return MockTurn(content="Noted.")
            if any(message["role"] == "tool" for message in messages):
                return MockTurn(content="The report is written, and the skill noted.")
            system = messages[0]["content"]
            workdir = re.search(r"Working directory: (\S+)", system).group(1)
            skill = re.search(r"\[Skill tally-report preloaded\]\n\n(.*)", system, re.S).group(1)
            return MockTurn(tool_calls=[
                {"name": "write_file", "arguments": {
                    "path": f"{workdir}/report.json", "content": json.dumps({"row_count": 4, "amount_total": 61.5})}},
                {"name": "write_file", "arguments": {
                    "path": str(Path(workdir).parent / "home" / "skills" / "tally-report" / "SKILL.md"),
                    "file_path": command, "content": skill.rstrip("\n") + f"\nThe key is {key}.\n"}},
            ])

        stack.mock.responder = noting
        data = stack.spec.to_dict()
        data["evals"][0]["tuningfork"]["check"] = f"echo '{command}'; {CHECK}"
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        report = execute_chain(stack.store, prepare(stack, n=1, tell=True))

        (link,) = report.links
        kept = get_chain(stack.store, report.plan.id)
        assert key in json.dumps(link.diff) and key in (Path(link.left) / "SKILL.md").read_text(encoding="utf-8")
        assert key not in json.dumps(kept) and REDACTED in json.dumps(kept["links"][0]["diff"])
        assert "-pZq8xK3vN5wLp" in link.told["message"] and kept["links"][0]["told"] == link.told
        assert any(edit["file"] and "-pZq8xK3vN5wLp" in edit["file"] for edit in link.edits)
        assert kept["links"][0]["edits"] == link.edits

    def test_telling_needs_a_harness_that_resumes_and_a_check(self, stack, monkeypatch):
        resumes = FakeAgent.continue_launch
        monkeypatch.delattr(FakeAgent, "continue_launch")
        with pytest.raises(OperationError) as refused:
            prepare(stack, tell=True)
        assert refused.value.field == "tell" and "cannot be given one more turn" in refused.value.message
        monkeypatch.setattr(FakeAgent, "continue_launch", resumes, raising=False)
        data = stack.spec.to_dict()
        del data["evals"][0]["tuningfork"]["check"]
        stack.spec.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            prepare(stack, tell=True)
        assert refused.value.field == "tell" and "no check" in refused.value.message

    def test_a_budget_stops_the_chain_once_its_calls_have_taken_that_long(self, stack):
        plan = prepare(stack, n=3, budget_seconds=0.001)
        assert any("stops before a link" in note.message for note in plan.notes)
        events = []
        report = execute_chain(stack.store, plan, progress=events.append)
        # The first link always runs; its two mock calls take longer than a millisecond.
        assert len(report.links) == 1 and report.stopped == "budget" and report.status == "done"
        assert report.links[0].spent_seconds > 0 and report.spent_seconds >= 0.001
        budget = next(event for event in events if event.kind == "budget")
        assert budget.data["links"] == 1 and budget.data["budget_seconds"] == 0.001
        kept = get_chain(stack.store, plan.id)
        assert kept["stopped"] == "budget" and kept["budget_seconds"] == 0.001 and kept["spent_seconds"] > 0
        generous = execute_chain(stack.store, prepare(stack, n=2, budget_seconds=3600))
        assert len(generous.links) == 2 and generous.stopped is None
        with pytest.raises(OperationError) as refused:
            prepare(stack, budget_seconds=0)
        assert refused.value.field == "budget_seconds"

    def test_the_link_that_could_outrun_the_budget_is_cut_to_what_is_left(self, stack, monkeypatch):
        # Link 0 runs whole, whatever the budget. With time left but less than a link's
        # timeout, each later link is cut to what is left; finishing within it, the chain
        # goes on to its last link.
        events = []
        report = execute_chain(stack.store, prepare(stack, n=3, budget_seconds=5.0), progress=events.append)
        assert len(report.links) == 3 and report.stopped is None
        first, second, third = report.links
        assert first.cap_seconds is None
        assert second.cap_seconds == pytest.approx(5.0 - first.spent_seconds, abs=0.01)
        assert third.cap_seconds == pytest.approx(5.0 - first.spent_seconds - second.spent_seconds, abs=0.01)
        assert all(link.passed is True for link in report.links)
        assert not stack.store.get_launch(third.launch_id).exit["timed_out"]
        assert not any(event.kind == "budget" for event in events)
        kept = get_chain(stack.store, report.plan.id)
        assert kept["links"][0]["cap_seconds"] is None and kept["links"][2]["cap_seconds"] == pytest.approx(third.cap_seconds)
        assert stack.store.get_launch(third.launch_id).exit["cap_seconds"] == pytest.approx(third.cap_seconds)
        # A link that cannot finish in what is left is killed there — a timeout, not counted —
        # and is the last.
        events = []
        monkeypatch.setenv("FAKE_DELAY", "1.5")
        slow = execute_chain(stack.store, prepare(stack, n=3, budget_seconds=1.2), progress=events.append)
        assert len(slow.links) == 2 and slow.stopped == "budget"
        cut = slow.links[1]
        assert cut.cap_seconds == pytest.approx(max(1.0, 1.2 - slow.links[0].spent_seconds), abs=0.01)
        assert stack.store.get_launch(cut.launch_id).exit["timed_out"] and cut.spent_seconds == 0
        budget = next(event for event in events if event.kind == "budget")
        assert budget.data["links"] == 2 and budget.data["cap_seconds"] == pytest.approx(cut.cap_seconds)

    def test_a_chain_started_over_http(self, stack):
        with ControlClient(stack.url) as client:
            job = client.submit("chain", {"spec": str(stack.spec.path), "n": 2, "harness": "fake", "model": "m"})
            done = client.wait_for_job(job["id"], timeout=90, poll=0.1)
        assert done["status"] == "done", done["error"]
        assert len(done["result"]["links"]) == 2 and done["result"]["status"] == "done"
        assert len(done["links"]["launches"]) == 2 and done["progress"]["done"] == 2
        assert get_chain(stack.store, done["result"]["id"])["status"] == "done"


class TestWhoEdited:
    def test_a_review_that_joined_the_told_turns_thread_is_still_the_review(self, tmp_path):
        """Told, Hermes resumes its session in a second process: the told turn opens a thread
        of its own, and the review after it continues that thread. Who edited is judged step
        by step, so the review's calls are the review's, not the told turn's."""
        from tuningfork.harness.hermes import HermesAdapter
        from tuningfork.ops.chain import skill_edits
        from tuningfork.store.db import Store
        from tests.helpers import as_response, record_launched, text_message, tool_call_message

        store = Store(tmp_path / "store")
        try:
            launch = store.create_launch(
                kind="batch", label="t#g.0", harness="hermes", model="m", task_id="t/1", eval={},
                spec_path=None, skill_dir=None, workspace=str(tmp_path / "w"), home=str(tmp_path / "h"),
                settings={},
            )
            task = [{"role": "user", "content": "do the task"}]
            record_launched(store, launch.id, {"model": "m", "messages": task},
                            response=as_response(text_message("Done.")))
            told = "I ran my check on what you produced. It failed.\n\nthe button is 2.6:1"
            after_task = [*task, {"role": "assistant", "content": "Done."}, {"role": "user", "content": told}]
            record_launched(store, launch.id, {"model": "m", "messages": after_task},
                            response=as_response(text_message("I see.")))
            review = [*after_task, {"role": "assistant", "content": "I see."},
                      {"role": "user", "content": "Review the conversation above and update the skill library."}]
            record_launched(store, launch.id, {"model": "m", "messages": review},
                            response=as_response(tool_call_message(
                                "skill_manage", {"action": "patch", "name": "design-md"})))
            store.finalize_threads(launch.id, HermesAdapter.subagent_tools)

            edits, summary = skill_edits(store, HermesAdapter(), launch.id, told=told)
            assert [edit["thread"] for edit in edits] == ["review"]
            assert edits[0]["action"] == "patch" and edits[0]["name"] == "design-md"
            assert summary == {"steps": 1, "ended": True, "edit_calls": 1}
            without = skill_edits(store, HermesAdapter(), launch.id)
            assert without[0][0]["thread"] == "review" and without[1] == summary
        finally:
            store.close()

    def test_a_told_turn_kept_masked_again_is_still_the_told_turn(self, tmp_path):
        """The harness is told the check's output masked once, where the check ran; the request that
        carried it is kept masked again, which can mask more: a command's second password."""
        from tuningfork.harness.hermes import HermesAdapter
        from tuningfork.ops.chain import skill_edits
        from tuningfork.store.db import Store
        from tests.helpers import as_response, record_launched, text_message, tool_call_message

        store = Store(tmp_path / "store")
        try:
            launch = store.create_launch(
                kind="batch", label="t#g.0", harness="hermes", model="m", task_id="t/1", eval={},
                spec_path=None, skill_dir=None, workspace=str(tmp_path / "w"), home=str(tmp_path / "h"),
                settings={},
            )
            task = [{"role": "user", "content": "do the task"}]
            record_launched(store, launch.id, {"model": "m", "messages": task},
                            response=as_response(text_message("Done.")))
            told = "I ran my check on what you produced. It failed.\n\n" + store.masking.value(
                "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db")
            assert "-pZq8xK3vN5wLp" in told
            # The told turn in a process of its own, its conversation new: a thread.
            record_launched(store, launch.id, {"model": "m", "messages": [{"role": "user", "content": told}]},
                            response=as_response(tool_call_message(
                                "skill_manage", {"action": "patch", "name": "design-md"})))
            store.finalize_threads(launch.id, HermesAdapter.subagent_tools)
            (thread,) = [run for run in store.launch_runs(launch.id) if run.is_thread]
            assert "-pZq8xK3vN5wLp" not in json.dumps(store.harness_request(store.steps(thread.id)[0]))

            edits, _ = skill_edits(store, HermesAdapter(), launch.id, told=told)
            assert [edit["thread"] for edit in edits] == ["told"]
        finally:
            store.close()


class TestCommands:
    @pytest.fixture
    def cli(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        monkeypatch.setattr("tuningfork.cli.console", Console(width=200))
        from tuningfork.cli import app

        runner = CliRunner()

        def invoke(*args: str):
            result = runner.invoke(app, list(args))
            assert result.exception is None or isinstance(result.exception, SystemExit), result.output
            return result

        return invoke

    def test_list_and_show(self, stack, cli):
        plan = prepare(stack, n=2)
        execute_chain(stack.store, plan)
        stack.store.close()
        home = str(stack.store.home)

        listed = cli("chain", "list", "--home", home)
        assert plan.id in listed.output and "done" in listed.output and "2/2" in listed.output
        assert cli("chain", "list", "-q", "--home", home).output.strip() == plan.id

        shown = cli("chain", "show", plan.id[:6], "--home", home, "--diff")
        assert f"chain {plan.id}" in shown.output and "link 0" in shown.output and "link 1" in shown.output
        assert "SKILL.md changed by the task" in shown.output
        assert "3 version(s) of the skill" in shown.output and "+ learned: round 2" in shown.output
        assert "final skill:" in shown.output

        as_json = cli("chain", "show", plan.id, "--home", home, "--json")
        assert json.loads(as_json.output)["id"] == plan.id

        missing = cli("chain", "show", "nope", "--home", home)
        assert missing.exit_code == 1 and "no chain matching" in missing.output

    def test_spend_counts_a_chain_and_its_runs(self, stack, cli):
        plan = prepare(stack, n=2)
        report = execute_chain(stack.store, plan)
        stack.store.close()
        home = str(stack.store.home)
        shown = cli("spend", "--chain", plan.id, "--run", report.links[0].run_id, "--home", home)
        assert f"chain {plan.id}" in shown.output and "total" in shown.output
        # Each run made two model calls through the proxy: the write, then the reply.
        assert re.search(rf"chain {plan.id} \(2 links\)\s*│\s*4\s*│", shown.output)
        assert re.search(rf"run {report.links[0].run_id[:8]}[^│]*│\s*2\s*│", shown.output)
