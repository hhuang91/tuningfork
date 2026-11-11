"""A task's runs as paths that part (`analysis.paths`): the tree a divergence report makes when it
is read along its runs, the turning point found on it, and each front end's way to them.

The tree is by move — the tools a step called, a reply whatever its words — because that is where
runs part: compared exactly, find-the-file's twenty runs were ten behaviours at the step that
decided them, and the step-by-step comparison named a later step, where the runs that gave up had
"ended"."""

from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

from tuningfork.analysis.paths import FORMS, outline, path_tree, turning_text
from tuningfork.cli import app
from tuningfork.ops import inspect
from tuningfork.store.db import Store
from tests.helpers import record_run, text_message, tool_call_message


def group(kind: str, label: str, runs: list[str], tools: list[str] | None = None) -> dict:
    return {"outcome": {"kind": kind, "label": label, "key": f"{kind}:{label}"}, "tools": tools or [], "run_ids": runs}


def call(tool: str, arguments: str, runs: list[str]) -> dict:
    return group("tool_call", f"{tool}({arguments})", runs, [tool])


def ended(runs: list[str]) -> dict:
    return group("ended", "run ended", runs)


def find_the_file() -> dict:
    """Twenty runs as find-the-file's went: nineteen read a file that is not there; fifteen of
    them say so, each in its own words, and fail; four search, find it and pass; one scripts."""
    gave_up = [f"g{i}" for i in range(15)]
    searched = [f"s{i}" for i in range(4)]
    scripted = ["x0"]
    return {
        "run_ids": gave_up + searched + scripted,
        "verdicts": {**{run: "failed" for run in gave_up}, **{run: "passed" for run in searched + scripted}},
        "steps": [
            {"index": 0, "groups": [call("read_file", "path='data.csv'", gave_up + searched), call("execute_code", "code='<ast:1>'", scripted)]},
            {"index": 1, "groups": [
                *[group("text", f"The file was not found, wording {i}.", [run]) for i, run in enumerate(gave_up)],
                call("search_files", "pattern='data.csv'", searched), call("search_files", "pattern='*.csv'", scripted)]},
            {"index": 2, "groups": [ended(gave_up), call("read_file", "path='inputs/data.csv'", searched), call("execute_code", "code='<ast:2>'", scripted)]},
            {"index": 3, "groups": [ended(gave_up), call("write_file", "path='report.json'", searched), group("text", "Done.", scripted)]},
            {"index": 4, "groups": [ended(gave_up + scripted), group("text", "Done.", searched)]},
        ],
    }


def children(tree: dict, parent: int) -> list[dict]:
    return [tree["nodes"][child] for child in tree["nodes"][parent]["children"]]


def by_label(tree: dict, parent: int) -> dict[str, dict]:
    return {node["label"]: node for node in children(tree, parent)}


class TestTheTree:
    def test_runs_part_by_the_move_they_made(self):
        tree = path_tree(find_the_file())
        root = tree["nodes"][0]
        assert (root["step"], len(root["runs"]), root["passed"]["successes"], root["passed"]["total"]) == (-1, 20, 5, 20)
        first = by_label(tree, 0)
        assert sorted(first) == ["execute_code", "read_file"] and len(first["read_file"]["runs"]) == 19
        # The most runs first: a split's first branch is the way most of its runs went.
        assert children(tree, 0)[0]["label"] == "read_file"
        second = by_label(tree, first["read_file"]["id"])
        # Fifteen refusals, each in its own words, are one move; so are the searches.
        assert {label: len(node["runs"]) for label, node in second.items()} == {"reply": 15, "search_files": 4}
        assert len(second["reply"]["forms"]) == 15 and second["reply"]["kind"] == "text"
        assert second["reply"]["passed"]["successes"] == 0 and second["search_files"]["passed"]["successes"] == 4
        # A path ends where its runs do: the refusals at step 1, the searchers' at their reply.
        assert sorted(second["reply"]["ends"]) == sorted(second["reply"]["runs"]) and not second["reply"]["children"]
        deepest = max(tree["nodes"], key=lambda node: node["step"])
        assert (deepest["step"], deepest["label"], len(deepest["ends"])) == (4, "reply", 4)

    def test_runs_are_indexes_into_the_reports(self):
        report = find_the_file()
        scripted = by_label(path_tree(report), 0)["execute_code"]
        assert [report["run_ids"][run] for run in scripted["runs"]] == ["x0"]

    def test_a_tools_calls_are_apart_where_they_came_in_a_few_forms(self):
        good, bad = [f"a{i}" for i in range(6)], [f"b{i}" for i in range(3)]
        report = {
            "run_ids": good + bad,
            "verdicts": {**{run: "passed" for run in good}, **{run: "failed" for run in bad}},
            "steps": [
                {"index": 0, "groups": [call("read_file", "path='data.csv'", good + bad)]},
                {"index": 1, "groups": [
                    call("write_file", "content='{\"amount_total\":61.5,\"row_count\":4}', path='report.json'", good),
                    call("write_file", "content='{\"count\":4,\"total\":61.5}', path='report.json'", bad)]},
            ],
        }
        auto = path_tree(report, forms="auto")
        writes = children(auto, by_label(auto, 0)["read_file"]["id"])
        assert [(node["label"], len(node["runs"]), node["apart"]) for node in writes] == [("write_file", 6, True), ("write_file", 3, True)]
        # What tells one form from the other, from the start of the argument where they differ.
        assert writes[0]["hint"].startswith("'{\"amount_total\"") and writes[1]["hint"].startswith("'{\"count\"")
        # By tool alone they are one node, and nothing parts: the difference is in what was written.
        tools = path_tree(report, forms="tools")
        assert [len(node["runs"]) for node in children(tools, by_label(tools, 0)["read_file"]["id"])] == [9]
        assert tools["turning"] is None and auto["turning"]["hint"].startswith("'{\"amount_total\"")

    def test_forms_one_run_each_stay_together(self):
        runs = [f"r{i}" for i in range(5)]
        report = {"run_ids": runs, "verdicts": {}, "steps": [
            {"index": 0, "groups": [call("execute_code", f"code='<ast:{i}>'", [run]) for i, run in enumerate(runs)]}]}
        assert [len(node["runs"]) for node in children(path_tree(report), 0)] == [5]
        assert len(children(path_tree(report, forms="calls"), 0)) == 5

    def test_rare_forms_are_one_node_beside_the_common_ones(self):
        runs = [f"r{i}" for i in range(10)]
        report = {"run_ids": runs, "verdicts": {}, "steps": [{"index": 0, "groups": [
            call("search", "q='a'", runs[:5]), call("search", "q='b'", runs[5:9]), call("search", "q='c'", runs[9:])]}]}
        found = [(len(node["runs"]), node["others"], node["hint"]) for node in children(path_tree(report), 0)]
        assert found == [(5, False, "'a'"), (4, False, "'b'"), (1, True, "1 other form")]

    def test_an_unknown_way_of_telling_calls_apart_is_refused(self):
        with pytest.raises(ValueError, match="forms is one of"):
            path_tree(find_the_file(), forms="exact")


class TestTheTurningPoint:
    def test_it_is_the_split_that_separates_passing_from_failing(self):
        tree = path_tree(find_the_file())
        turning = tree["turning"]
        read = by_label(tree, 0)["read_file"]
        assert (turning["step"], turning["after"], turning["label"], turning["kind"]) == (1, "read_file", "search_files", "tool_call")
        assert (turning["passed"], turning["total"], turning["others_passed"], turning["others_total"]) == (4, 4, 0, 15)
        assert turning["parent"] == read["id"] and tree["nodes"][turning["branch"]]["label"] == "search_files"
        # Of two branches that split the runs alike, the one the passing runs took: a target names it.
        assert turning["with_passing"] is True and turning["p"] < 0.001 and turning["run_id"] == "s0"
        assert turning_text(turning).startswith("step 1: after read_file, runs that called search_files passed 4/4, the rest 0/15")

    def test_the_earliest_of_equal_splits_stands(self):
        x, y = [f"x{i}" for i in range(6)], [f"y{i}" for i in range(6)]
        passed = x[:3] + y[:3]
        report = {
            # The runs whose split comes later are met first in the report: the order must not decide.
            "run_ids": y + x,
            "verdicts": {run: ("passed" if run in passed else "failed") for run in x + y},
            "steps": [
                {"index": 0, "groups": [call("plan", "", x), call("look", "", y)]},
                {"index": 1, "groups": [call("check", "", x[:3]), call("skip", "", x[3:]), call("wait", "", y)]},
                {"index": 2, "groups": [ended(x), call("check", "", y[:3]), call("skip", "", y[3:])]},
            ],
        }
        turning = path_tree(report)["turning"]
        # Two splits part their runs alike, 3 of 3 against 0 of 3, one at step 1 and one at step 2:
        # the earlier stands, and of its two branches the one the passing runs took.
        assert (turning["step"], turning["after"], turning["label"], turning["passed"], turning["total"]) == (1, "plan", "check", 3, 3)

    def test_ending_there_is_a_branch_too(self):
        stopped, went_on = [f"s{i}" for i in range(3)], [f"w{i}" for i in range(3)]
        report = {
            "run_ids": stopped + went_on,
            "verdicts": {**{run: "failed" for run in stopped}, **{run: "passed" for run in went_on}},
            "steps": [
                {"index": 0, "groups": [call("read_file", "", stopped + went_on)]},
                {"index": 1, "groups": [ended(stopped), call("write_file", "", went_on)]},
            ],
        }
        tree = path_tree(report)
        read = by_label(tree, 0)["read_file"]
        assert sorted(read["ends"]) == [0, 1, 2] and (read["ended"]["successes"], read["ended"]["total"]) == (0, 3)
        # The branch the passing runs took is named; the runs that ended there are "the rest".
        assert (tree["turning"]["label"], tree["turning"]["passed"], tree["turning"]["others_total"]) == ("write_file", 3, 3)

    @pytest.mark.parametrize("verdict", ["passed", "failed", None])
    def test_none_where_the_runs_fared_alike(self, verdict):
        report = find_the_file()
        report["verdicts"] = {run: verdict for run in report["run_ids"]}
        assert path_tree(report)["turning"] is None

    def test_none_where_too_few_runs_took_a_branch(self):
        report = {"run_ids": ["a", "b", "c"], "verdicts": {"a": "passed", "b": "failed", "c": "failed"}, "steps": [
            {"index": 0, "groups": [call("plan", "", ["a"]), call("guess", "", ["b", "c"])]}]}
        assert path_tree(report)["turning"] is None
        assert turning_text(None) is None


class TestItsLines:
    def test_the_tree_in_lines(self):
        assert outline(path_tree(find_the_file())) == [
            "read_file ×19",
            "  reply ×15  — passed 0/15",
            "  search_files ×4 → read_file → write_file → reply  — passed 4/4",
            "execute_code ×1 → search_files → execute_code → reply  — passed 1/1",
        ]

    def test_runs_gone_their_own_way_are_counted_not_followed(self):
        runs = [f"r{i}" for i in range(4)]
        report = {"run_ids": runs, "verdicts": {run: "failed" for run in runs}, "steps": [
            {"index": 0, "groups": [call("read_file", "", runs)]},
            {"index": 1, "groups": [call("plan", "", runs[:2]), call("guess", "", runs[2:3]), call("give_up", "", runs[3:])]},
        ]}
        assert outline(path_tree(report)) == [
            "read_file ×4",
            "  plan ×2  — passed 0/2",
            "  2 runs, each its own way (give_up, guess)  — passed 0/2",
        ]

    def test_a_long_tree_is_cut_and_says_so(self):
        runs = [f"r{i}" for i in range(12)]
        report = {"run_ids": runs, "verdicts": {}, "steps": [
            {"index": 0, "groups": [call(f"tool{i // 2}", "", runs[i:i + 2]) for i in range(0, 12, 2)]}]}
        lines = outline(path_tree(report), most=3)
        assert len(lines) == 4 and lines[-1] == "… 3 more line(s)"


class TestTheSplitByStep:
    def test_the_first_of_equal_splits_stands(self):
        """A split's p is kept to four places for a reader. Compared against that rounded figure,
        a later split as good as the first replaced it: 0.00006 is under the 0.0001 it had been
        rounded to. find-the-file's page said "ending before step 3"; the first was step 2."""
        found = inspect.verdict_split({"steps": [
            {"index": step, "groups": [
                {"outcome": {"label": "run ended", "kind": "ended"}, "run_ids": [f"g{i}" for i in range(15)], "passed": {"successes": 0, "total": 15}},
                {"outcome": {"label": "read_file", "kind": "tool_call"}, "run_ids": [f"s{i}" for i in range(5)], "passed": {"successes": 5, "total": 5}},
            ]} for step in (2, 3)]})
        assert found["best"]["index"] == 2 and found["best"]["p"] == 0.0001


class TestEachFrontEnd:
    def store(self, tmp_path) -> Store:
        store = Store(tmp_path / "paths")
        read = tool_call_message("read_file", {"path": "data.csv"})
        search = tool_call_message("search_files", {"pattern": "data.csv"})
        write = tool_call_message("write_file", {"path": "report.json", "content": "{}"})
        for index in range(4):
            run_id = record_run(store, f"g{index}", [read, text_message(f"Not found, in words {index}.")], task_id="find/1")
            store.annotate_run(run_id, result={"passed": False, "usable": True})
        for index in range(3):
            run_id = record_run(store, f"s{index}", [read, search, write, text_message("done")], task_id="find/1")
            store.annotate_run(run_id, result={"passed": True, "usable": True})
        return store

    def test_the_report_carries_a_tree_for_each_detail(self, tmp_path):
        store = self.store(tmp_path)
        try:
            report = inspect.divergence_report(store, task="find/1")
            assert sorted(report["paths"]) == sorted(FORMS)
            tree = report["paths"]["auto"]
            assert (tree["runs"], tree["judged"], tree["passed"]) == (7, 7, 3)
            assert (tree["turning"]["step"], tree["turning"]["label"]) == (1, "search_files")
            # A behaviour's tools are in the report: what a tree draws a node by.
            assert report["steps"][0]["groups"][0]["tools"] == ["read_file"]
        finally:
            store.close()

    def test_the_terminal_prints_it_when_asked(self, tmp_path):
        self.store(tmp_path).close()
        runner = CliRunner()
        plain = runner.invoke(app, ["divergence", "--task", "find/1", "--home", str(tmp_path / "paths")])
        assert plain.exit_code == 0 and "turning point" not in plain.output
        result = runner.invoke(app, ["divergence", "--task", "find/1", "--paths", "--home", str(tmp_path / "paths")])
        assert result.exit_code == 0, result.output
        assert "read_file ×7" in result.output and "search_files ×3 → write_file → reply" in result.output
        assert "turning point  step 1: after read_file, runs that called search_files passed 3/3, the rest 0/4" in result.output
        assert "A lead to read, never a verdict" in result.output

    def test_an_agent_is_given_the_turning_point(self, tmp_path):
        pytest.importorskip("mcp")
        from tuningfork.mcp.server import ToolServer
        from tests.test_mcp_tools import NoJobs, call as call_tool, refusal, text

        store = self.store(tmp_path)
        try:
            result = asyncio.run(call_tool(ToolServer(store, NoJobs(), toolset="evidence"), "find_divergence", {"task": "find/1"}))
            assert not refusal(result), text(result)
            said = text(result)
            # Said as the lead it is.
            assert ("the turning point — where passing and failing part along the runs' paths; a lead, under p 0.2 — "
                    "is at step 1: after read_file, runs that called search_files passed 3/3") in said
            # The comparison step by step names its step by number; where that is another step than
            # the turning point's, it says which of the two to act on, and why they differ.
            other = next(line for line in said.splitlines() if "best predicts the verdict is at step" in line)
            named = int(other.split("is at step ")[1].split(":")[0].split(" ")[0])
            ended = result.structured_content["verdict_split"]["kind"] == "ended"
            # Where it is only that some runs had ended, the turning point is the one to refine at;
            # where it is an exact behaviour at another step, the two are said to compare differently.
            assert ("refine at the turning point, where passing and failing parted" in other) == (named != 1 and ended)
            assert ("this compares exact behaviours at one step" in other) == (named != 1 and not ended)
            turning = result.structured_content["turning_point"]
            assert (turning["step"], turning["label"], turning["passed"], turning["others_total"]) == (1, "search_files", 3, 4)
            assert "parent" not in turning and len(turning["run_id"]) == 8
        finally:
            store.close()

    def divergence(self, tmp_path, runs: list[tuple[list, bool]]) -> str:
        """What an agent is told of runs recorded as given: each its answers, and whether it passed."""
        from tuningfork.mcp.server import ToolServer
        from tests.test_mcp_tools import NoJobs, call as call_tool, refusal, text

        store = Store(tmp_path / "leads")
        try:
            for index, (answers, passed) in enumerate(runs):
                run_id = record_run(store, f"r{index}", answers, task_id="leads/1")
                store.annotate_run(run_id, result={"passed": passed, "usable": True})
            result = asyncio.run(call_tool(ToolServer(store, NoJobs(), toolset="evidence"), "find_divergence", {"task": "leads/1"}))
            assert not refusal(result), text(result)
            return text(result)
        finally:
            store.close()

    def test_where_the_comparison_names_runs_that_had_ended_the_turning_point_is_the_one_to_refine(self, tmp_path):
        read = tool_call_message("read_file", {"path": "data.csv"})
        write = tool_call_message("write_file", {"path": "report.json", "content": "{}"})
        # At step 1 every run does its own exact thing — each refusal reworded, each search with
        # its own pattern — so, compared exactly, the first step that tells is the next one, where
        # the runs that gave up had ended. Along their paths, the runs parted at step 1.
        gave_up = [([read, text_message(f"Not found, in words {index}.")], False) for index in range(4)]
        searched = [([read, tool_call_message("search_files", {"pattern": f"data{index}.csv"}), write, text_message("done")], True)
                    for index in range(3)]
        said = self.divergence(tmp_path, gave_up + searched)
        assert "is at step 1: after read_file, runs that called search_files passed 3/3, the rest 0/4" in said
        other = next(line for line in said.splitlines() if "best predicts the verdict is at step" in line)
        assert "is at step 2: run ended" in other
        assert other.endswith("this comparison counts runs that had ended as one behaviour at every later step: refine at "
                              "the turning point, where passing and failing parted")

    def test_where_it_names_an_exact_behaviour_at_another_step_the_two_are_said_to_compare_differently(self, tmp_path):
        # The passing runs read one path at step 0, the failing ones each their own: one box on the
        # map, read_file, and a behaviour that tells when compared exactly. The paths part at step 1.
        passing = [([tool_call_message("read_file", {"path": "data.csv"}), tool_call_message("search_files", {"pattern": "data.csv"}),
                     text_message("done")], True) for _ in range(3)]
        failing = [([tool_call_message("read_file", {"path": f"other{index}.csv"}), text_message(f"Not found, in words {index}.")], False)
                   for index in range(4)]
        said = self.divergence(tmp_path, passing + failing)
        assert "is at step 1: after read_file, runs that called search_files passed 3/3, the rest 0/4" in said
        other = next(line for line in said.splitlines() if "best predicts the verdict is at step" in line)
        assert "is at step 0 (the first split): read_file(path='data.csv')" in other
        assert other.endswith("another step than the turning point's: this compares exact behaviours at one step, the "
                              "turning point what the runs did along their paths")
        assert "refine at the turning point" not in other


class TestWhatAStepLedTo:
    """Behaviours gathered by move, for an experiment's page: a reply one move whatever its words."""

    def outcomes(self):
        from tuningfork.analysis.normalize import step_outcome

        def response(message: dict) -> dict:
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

        search = step_outcome(response(tool_call_message("search_files", {"pattern": "data.csv"})))
        other = step_outcome(response(tool_call_message("search_files", {"pattern": "*.csv"})))
        replies = [step_outcome(response(text_message(f"Not found, in words {index}."))) for index in range(3)]
        return search, other, replies

    def test_a_distribution_says_its_moves(self):
        from tuningfork.analysis.distribution import distribution

        search, other, replies = self.outcomes()
        found = distribution([search, search, other, *replies, replies[0]]).to_json()
        assert len(found["clusters"]) == 5
        moves = {move["label"]: move for move in found["moves"]}
        # Four replies in three phrasings are one move; so are the searches, whatever their pattern.
        assert (moves["reply"]["forms"], moves["reply"]["by_arm"]["all"]["successes"], moves["reply"]["by_arm"]["all"]["total"]) == (3, 4, 7)
        # Two searches with one pattern and one with another: too few of the second to draw apart.
        assert (moves["search_files"]["forms"], moves["search_files"]["by_arm"]["all"]["successes"], moves["search_files"]["tools"]) == (2, 3, ["search_files"])
        assert moves["search_files"]["apart"] is False and moves["search_files"]["hint"] is None
        assert (moves["reply"]["forms_by_arm"], moves["reply"]["others"]) == ({"all": 3}, False)
        # Each move's rate has its interval, and its behaviours' letters lead back to them.
        assert 0 < moves["reply"]["by_arm"]["all"]["low"] < moves["reply"]["by_arm"]["all"]["high"] < 1
        assert sorted(letter for move in found["moves"] for letter in move["letters"]) == sorted(cluster["letter"] for cluster in found["clusters"])
        # The most made first.
        assert [move["label"] for move in found["moves"]] == ["reply", "search_files"]

    def test_a_tools_calls_in_a_few_forms_are_moves_of_their_own(self):
        from tuningfork.analysis.distribution import distribution

        search, other, replies = self.outcomes()
        found = distribution([search, search, search, other, other, replies[0]]).to_json()
        searches = [move for move in found["moves"] if move["label"] == "search_files"]
        # Three searches with one pattern, two with another: each form a move, told apart by its pattern.
        assert [(move["by_arm"]["all"]["successes"], move["apart"], move["hint"]) for move in searches] == [
            (3, True, "'data.csv'"), (2, True, "'*.csv'")]
        assert [move["letters"] for move in searches] == [["A"], ["B"]]

    def test_an_ab_says_what_each_wording_led_to(self):
        from tuningfork.analysis.distribution import compare_arms

        search, other, replies = self.outcomes()
        found = compare_arms([search, *replies], [search, search, other, replies[0]]).to_json()
        moves = {move["label"]: move["by_arm"] for move in found["moves"]}
        assert (moves["reply"]["original"]["successes"], moves["reply"]["original"]["total"]) == (3, 4)
        assert (moves["reply"]["variant"]["successes"], moves["reply"]["variant"]["total"]) == (1, 4)
        assert (moves["search_files"]["original"]["successes"], moves["search_files"]["variant"]["successes"]) == (1, 3)
        # How many forms a move took is each wording's own: three phrasings of a reply over both,
        # one of them the variant's; two forms of the search, the original's samples in one.
        forms = {move["label"]: (move["forms"], move["forms_by_arm"]) for move in found["moves"]}
        assert forms["reply"] == (3, {"original": 3, "variant": 1})
        assert forms["search_files"] == (2, {"original": 1, "variant": 2})

    def test_a_screen_says_its_moves_by_wording(self):
        from tuningfork.analysis.distribution import target_by_rules
        from tuningfork.analysis.screen import compare_to_baseline

        search, other, replies = self.outcomes()
        target = target_by_rules([{"if": "calls", "value": "search_files"}])
        found = compare_to_baseline(replies, {"p1": [search, other, search]}, target=target, planned=3).to_json()
        assert found["behaviours"][0]["tools"] in (["search_files"], [])
        moves = {move["label"]: move["by_arm"] for move in found["moves"]}
        assert (moves["search_files"]["p1"]["successes"], moves["search_files"]["p1"]["total"]) == (3, 3)
        assert (moves["reply"]["original"]["successes"], moves["reply"]["p1"]["successes"]) == (3, 0)
        # A wording that never replied took none of the reply's phrasings.
        reply = next(move for move in found["moves"] if move["label"] == "reply")
        assert reply["forms_by_arm"] == {"original": 3, "p1": 0}


class TestHowForksEnded:
    """Forks run to the end: what each did at the step, set against how it ended. A fan's branch
    ends in its forks, as a path of runs does — and an agent is told which behaviour went on to
    pass."""

    DEPARTED = "the replay departed from the recording at step 0"

    def arm(self, store, name: str, forks: list[tuple], *, kind: str = "resample"):
        """One arm of forks: each its answer at the step (a message, or None where it never got
        there), whether its check passed, and why it does not count."""
        from tuningfork.analysis.normalize import step_outcome

        experiment = store.create_experiment(
            kind=kind, arm=name, run_id="r", step_idx=1, step_id=None, request={"messages": []},
            settings={"n": len(forks), "reach": "end"}, status="done")
        for idx, (message, passed, exclusion) in enumerate(forks):
            response = {"choices": [{"message": message, "finish_reason": "stop"}]} if message else None
            found = step_outcome(response) if response else None
            store.add_sample(
                experiment.id, idx, seed=None, response=response, outcome_kind=found.kind if found else "error",
                outcome_key=found.key if found else None, outcome_label=found.label if found else None,
                error=exclusion, run_id=f"{name}{idx}", passed=passed, exclusion=exclusion)
        return experiment

    def forks(self) -> list[tuple]:
        search = tool_call_message("search_files", {"pattern": "data.csv"})
        replies = [text_message(f"Not found, in words {index}.") for index in range(3)]
        return [
            (search, True, None), (search, True, None), (search, False, None),
            (replies[0], False, None), (replies[1], False, None),
            (replies[2], None, None),                # no check judged it: not counted
            (search, True, "timed out"),             # a failure in the arm, and out of the step's rates
            (None, None, self.DEPARTED),             # never measured the wording
        ]

    def test_an_arm_says_how_each_behaviours_forks_ended(self, store):
        from tuningfork.ops.forks import arm_summary

        summary = arm_summary(store, self.arm(store, "original", self.forks()))
        assert (summary.passed, summary.counted, summary.timeouts) == (2, 6, 1)
        shown = summary.to_json()
        # Each fork says what it did at the step, by its behaviour's letter: none where the
        # step's rates leave the fork out.
        assert [fork["letter"] for fork in shown["forks"]] == ["A", "A", "A", "B", "C", "D", None, None]
        ended = {cluster["letter"]: cluster["ended"] for cluster in shown["actions"]["clusters"]}
        assert (ended["A"]["passed"], ended["A"]["failed"], ended["A"]["no_verdict"]) == (2, 1, 0)
        # A rate with its interval, over the forks that count; none where none counts.
        assert (ended["A"]["rate"]["successes"], ended["A"]["rate"]["total"]) == (2, 3)
        assert 0 < ended["A"]["rate"]["low"] < ended["A"]["rate"]["high"] < 1
        assert ended["D"] == {"passed": 0, "failed": 0, "no_verdict": 1, "rate": None}
        # By move, the three phrasings of the reply are one, and its forks are theirs together.
        moves = {move["label"]: move["ended"]["all"] for move in shown["actions"]["moves"]}
        assert (moves["search_files"]["passed"], moves["search_files"]["failed"]) == (2, 1)
        assert (moves["reply"]["passed"], moves["reply"]["failed"], moves["reply"]["no_verdict"]) == (0, 2, 1)
        assert (moves["reply"]["rate"]["successes"], moves["reply"]["rate"]["total"]) == (0, 2)

    def test_samples_at_a_step_say_nothing_of_an_end(self):
        from tuningfork.analysis.distribution import distribution
        from tuningfork.analysis.normalize import step_outcome

        answers = [step_outcome({"choices": [{"message": message, "finish_reason": "stop"}]})
                   for message, _, _ in self.forks()[:5]]
        found = distribution(answers).to_json()
        assert all("ended" not in cluster for cluster in found["clusters"])
        assert all("ended" not in move for move in found["moves"])

    def test_an_ab_says_it_of_each_wording(self, store):
        from tuningfork.ops.forks import arm_summary, compare_to_end

        search = tool_call_message("search_files", {"pattern": "data.csv"})
        original = arm_summary(store, self.arm(store, "original", self.forks(), kind="ab"))
        variant = arm_summary(store, self.arm(store, "variant", [(search, True, None)] * 4, kind="ab"))
        shown = compare_to_end(original, variant).to_json()
        mine = {move["label"]: move["ended"]["all"] for move in shown["variant"]["actions"]["moves"]}
        assert list(mine) == ["search_files"]
        assert (mine["search_files"]["passed"], mine["search_files"]["failed"], mine["search_files"]["no_verdict"]) == (4, 0, 0)
        assert (mine["search_files"]["rate"]["successes"], mine["search_files"]["rate"]["total"]) == (4, 4)
        theirs = {move["label"]: move["ended"]["all"] for move in shown["original"]["actions"]["moves"]}
        assert (theirs["reply"]["failed"], theirs["search_files"]["passed"]) == (2, 2)
        # And of both together, for one picture: each move with each wording's forks that made it.
        both = {move["label"]: move for move in shown["moves"]}
        search = both["search_files"]
        assert (search["by_arm"]["original"]["successes"], search["by_arm"]["variant"]["successes"]) == (3, 4)
        assert (search["ended"]["original"]["passed"], search["ended"]["original"]["failed"], search["ended"]["variant"]["passed"]) == (2, 1, 4)
        assert both["reply"]["ended"]["variant"] == {"passed": 0, "failed": 0, "no_verdict": 0, "rate": None}
        assert (both["reply"]["forms_by_arm"], both["reply"]["by_arm"]["variant"]["successes"]) == ({"original": 3, "variant": 0}, 0)
        # Each arm letters its own behaviours, so the moves of both carry no letter.
        assert all(move["letters"] == [] for move in shown["moves"])

    def test_an_ab_tells_apart_the_forms_each_wording_made(self, store):
        from tuningfork.ops.forks import arm_summary, compare_to_end

        # Each arm made one form of the call: by itself one move, and together the two they are.
        old = tool_call_message("write_file", {"path": "report.json", "content": '{"count":4}'})
        new = tool_call_message("write_file", {"path": "report.json", "content": '{"row_count":4}'})
        original = arm_summary(store, self.arm(store, "original", [(old, False, None)] * 2, kind="ab"))
        variant = arm_summary(store, self.arm(store, "variant", [(new, True, None)] * 2, kind="ab"))
        shown = compare_to_end(original, variant).to_json()
        assert [move["apart"] for move in shown["original"]["actions"]["moves"]] == [False]
        moves = shown["moves"]
        assert [(move["label"], move["apart"]) for move in moves] == [("write_file", True), ("write_file", True)]
        assert len({move["hint"] for move in moves}) == 2 and all("count" in move["hint"] for move in moves)
        made = sorted((move["by_arm"]["original"]["successes"], move["by_arm"]["variant"]["successes"],
                       move["ended"]["original"]["failed"], move["ended"]["variant"]["passed"]) for move in moves)
        assert made == [(0, 2, 0, 2), (2, 0, 2, 0)]

    def test_an_agent_is_told_which_behaviour_went_on_to_pass(self, store):
        from tuningfork.ops import compact
        from tuningfork.ops.forks import arm_summary

        seen = compact.to_end(arm_summary(store, self.arm(store, "original", self.forks())).to_json())
        behaviours = {entry["letter"]: entry for entry in seen["did_at_step"]["behaviours"]}
        assert (behaviours["A"]["passed"]["successes"], behaviours["A"]["passed"]["total"]) == (2, 3)
        assert behaviours["A"]["passed"]["low"] < behaviours["A"]["passed"]["high"] and "no_verdict" not in behaviours["A"]
        # A fork no check judged has no verdict: it is no fork "not counted", which is one left out.
        assert behaviours["D"]["passed"] is None and behaviours["D"]["no_verdict"] == 1
        lines = compact.to_end_lines(seen)
        assert lines[0].startswith("run to the end, judged by the task's check: passed 2/6")
        assert lines[1] == "  what the forks did at the step, and how those forks ended:"
        assert lines[2].startswith("    A: search_files(") and "— went on to pass 2/3 (67%, 95% CI " in lines[2]
        assert lines[5].endswith("— no verdict") and len(lines) == 6
        # One behaviour at the step parts nothing: the pass rate above says all there is.
        search = tool_call_message("search_files", {"pattern": "data.csv"})
        alike = compact.to_end(arm_summary(store, self.arm(store, "variant", [(search, True, None)] * 3)).to_json())
        assert compact.to_end_lines(alike) == ["run to the end, judged by the task's check: passed 3/3 (100%, 95% CI 44–100%)"]

    def test_the_terminal_says_it_where_the_forks_did_more_than_one_thing(self, tmp_path):
        store = Store(tmp_path / "forks")
        search = tool_call_message("search_files", {"pattern": "data.csv"})
        try:
            mixed = self.arm(store, "original", self.forks()).id
            alike = self.arm(store, "original", [(search, True, None)] * 3).id
        finally:
            store.close()
        runner = CliRunner()
        shown = runner.invoke(app, ["experiment", mixed, "--home", str(tmp_path / "forks")])
        assert shown.exit_code == 0, shown.output
        assert "then passed" in shown.output
        row = next(line for line in shown.output.splitlines() if line.startswith("A "))
        assert "3/6" in row and row.rstrip().endswith("2/3  67%  [21%-94%]") and "search_files(" in row
        # No fork of the behaviour counted: no rate to say.
        assert next(line for line in shown.output.splitlines() if line.startswith("D ")).rstrip().endswith("  -")
        # The two forks in no row are said as what they are: none failed in transport.
        assert "2 fork(s) are in none of these rows: they timed out, or were not counted" in shown.output
        assert "failed in transport" not in shown.output
        # One behaviour parts nothing: the table is as it was, and a guide's capture of it holds.
        plain = runner.invoke(app, ["experiment", alike, "--home", str(tmp_path / "forks")])
        assert plain.exit_code == 0, plain.output
        assert "then passed" not in plain.output and "3/3" in plain.output

    def screen(self, **changes):
        from types import SimpleNamespace as Fork

        from tuningfork.analysis.normalize import step_outcome
        from tuningfork.analysis.screen import compare_verdicts

        def answer(message: dict):
            return step_outcome({"choices": [{"message": message, "finish_reason": "stop"}]})

        search, other = (answer(tool_call_message("search_files", {"pattern": pattern})) for pattern in ("data.csv", "*.csv"))
        replies = [answer(text_message(f"Not found, in words {index}.")) for index in range(3)]
        given = {
            "baseline": [Fork(passed=False, exclusion=None) for _ in replies],
            "arms": {"p1": [Fork(passed=True, exclusion=None), Fork(passed=True, exclusion=None), Fork(passed=False, exclusion=None)]},
            "behaviours": {"original": replies, "p1": [search, search, other]},
            **changes,
        }
        return compare_verdicts(given["baseline"], given["arms"], planned=3, behaviours=given["behaviours"]).to_json()

    def test_a_screen_to_the_end_says_it_arm_by_arm(self):
        found = self.screen()
        moves = {move["label"]: move["ended"] for move in found["moves"]}
        assert (moves["search_files"]["p1"]["passed"], moves["search_files"]["p1"]["failed"]) == (2, 1)
        assert (moves["search_files"]["p1"]["rate"]["successes"], moves["search_files"]["p1"]["rate"]["total"]) == (2, 3)
        assert moves["search_files"]["original"] == {"passed": 0, "failed": 0, "no_verdict": 0, "rate": None}
        assert (moves["reply"]["original"]["failed"], moves["reply"]["p1"]["rate"]) == (3, None)
        # Each behaviour says it too, and what an agent reads of the screen keeps the counts.
        from tuningfork.ops import compact

        rows = {row["outcome"]["label"]: row["ended"] for row in found["behaviours"]}
        assert rows["search_files(pattern='data.csv')"]["p1"]["passed"] == 2
        # One key and one shape with a resample's behaviours: `passed`, a rate with its interval,
        # here for each arm whose forks that did it were judged.
        shown = {row["label"]: row["passed"] for row in compact.screen(found)["behaviours"]}
        searched = shown["search_files(pattern='data.csv')"]
        assert list(searched) == ["p1"] and (searched["p1"]["successes"], searched["p1"]["total"]) == (2, 2)
        assert 0 < searched["p1"]["low"] < searched["p1"]["high"] <= 1
        assert (shown["search_files(pattern='*.csv')"]["p1"]["successes"], shown["search_files(pattern='*.csv')"]["p1"]["total"]) == (0, 1)

    def test_answers_that_are_not_one_a_fork_say_nothing(self):
        from types import SimpleNamespace as Fork

        # An answer short of the forks: which fork did what is not known, so nothing is said of it.
        short = self.screen(baseline=[Fork(passed=False, exclusion=None)] * 2)
        assert short["moves"] and all("ended" not in move for move in short["moves"])
        assert all("ended" not in row for row in short["behaviours"])
        # And a screen at a step, whose samples have no end, says none.
        from tuningfork.analysis.distribution import target_by_rules
        from tuningfork.analysis.normalize import step_outcome
        from tuningfork.analysis.screen import compare_to_baseline

        search = step_outcome({"choices": [{"message": tool_call_message("search_files", {"pattern": "data.csv"}), "finish_reason": "stop"}]})
        target = target_by_rules([{"if": "calls", "value": "search_files"}])
        at_step = compare_to_baseline([search], {"p1": [search]}, target=target, planned=1).to_json()
        assert all("ended" not in move for move in at_step["moves"]) and all("ended" not in row for row in at_step["behaviours"])
