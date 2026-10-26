"""The forking-token probe, against the mock's toy llama.cpp (src/tuningfork/demo/toy.py).

The toy model holds answers that part at one token. Forcing either side there leads to that
answer, so the probe must find exactly that position, and nothing else.
"""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.analysis.forking import detectable, forking_report
from tuningfork.analysis.distribution import target_by_key, target_by_pattern
from tuningfork.analysis.normalize import Outcome
from tuningfork.cli import app
from tuningfork.demo.toy import props as toy_props
from tuningfork.ops.catalog import CATALOG
from tuningfork.ops.common import Cancel, Cancelled, OperationError
from tuningfork.ops.forking import _prefill_seconds, continuations, execute_forking, forking_result, prepare_forking
from tuningfork.ops.inspect import experiment_report
from tuningfork.ops.tokens import execute_score, prepare_score, sample_tokens
from tuningfork.proxy.app import create_browse_app
from tuningfork.wire import get_wire
from tests.helpers import as_response

UPSTREAM = "http://upstream.test"
TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
         for name in ("read_file", "search_files")]
REQUEST = {"model": "m", "tools": TOOLS, "max_tokens": 4096, "stream": True, "messages": [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Summarize data.csv into report.json."},
]}
READ = '<|tool_call>call:read_file{path:<|"|>data.csv<|"|>}<tool_call|>'
SEARCH = '<|tool_call>call:search_files{pattern:<|"|>data.csv<|"|>}<tool_call|>'


def thought(text: str) -> str:
    return f"<|channel>thought\n{text}<channel|>"


def message(raw: str) -> dict:
    """The server's parse of a raw answer: the recorded message."""
    from tuningfork.backends.formats import get_format

    return get_format("gemma4").parse(raw, tools=True)


@pytest.fixture
def served(monkeypatch, mock_server):
    mock_server.props = toy_props()
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return mock_server


async def recorded_and_scored(store, served, answers, recorded_raw, request=REQUEST, counted=None) -> str:
    """A step whose answer was `recorded_raw`, scored on the toy model holding `answers` — its prompt
    counted as `counted` renders, where given: the server's count of another text."""
    served.toy.answers = answers
    usage = {"prompt_tokens": served.toy.input_tokens(counted or request),
             "completion_tokens": len(served.toy.tokenize(recorded_raw)) + 1}
    store.begin_run("probed", "t/1")
    wire = get_wire("openai-chat")
    store.record_step(request=request, response={**as_response(message(recorded_raw), "tool_calls"), "usage": usage},
                      wire=wire.name, normalize=wire.normalize_message)
    run_id = store.get_run("probed").id
    await execute_score(store, prepare_score(store, run_id, "0", base_url=UPSTREAM))
    return run_id


FORK_IN_ANSWER = [(thought("Look at it.") + READ, 0.6), (thought("Look at it.") + SEARCH, 0.4)]
FORK_IN_REASONING = [(thought("I will read.") + READ, 0.6), (thought("I will search.") + SEARCH, 0.4)]
RUN_CODE = [(thought("Run it.") + '<|tool_call>call:execute_code{code:<|"|>print(1)<|"|>}<tool_call|>', 0.6),
            (thought("Run it.") + '<|tool_call>call:execute_code{code:<|"|>print(2)<|"|>}<tool_call|>', 0.4)]


class TestProbe:
    async def test_a_context_masked_as_it_was_recorded_is_said_in_the_plan(self, store, served):
        """Found in review: the probe's plan said nothing of a mask, and its check of the prompt's
        token count lets through one that renders to the count the step recorded."""
        from tuningfork.config import REDACTED

        from tests.helpers import KEY

        # Stored masked: "Summarize <redacted>." — counted by the server as the mask renders, so the
        # count lets it through.
        def asked(content):
            return {**REQUEST, "messages": [*REQUEST["messages"][:1], {"role": "user", "content": content}]}

        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0],
                                           request=asked(f"Summarize {KEY}."), counted=asked(f"Summarize {REDACTED}."))
        plan = prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM)
        assert [warning for warning in plan.warnings if f"step 0's context reads {REDACTED} in 1 place," in warning]
        assert [warning for warning in prepare_score(store, run_id, "0", base_url=UPSTREAM).warnings
                if "the answer is scored after the mask" in warning]

    async def test_the_forking_token_is_found_and_nothing_else(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])

        plan = prepare_forking(store, run_id, "0", n=6, base_url=UPSTREAM)

        (candidate,) = plan.candidates
        assert candidate.chosen.text == "read_file" and candidate.part == "tool_call"
        assert [(arm["text"], arm["original"]) for arm in plan.arms] == [("search_files", False), ("read_file", True)]
        assert plan.planned == 12 and plan.family == 1
        assert "read_file" in plan.target.description
        result = await execute_forking(store, plan)

        report = result.report
        (position,) = report.positions
        assert (position.baseline.target.successes, position.baseline.target.total) == (6, 6)
        (forced,) = position.alternatives
        assert (forced.target.successes, forced.target.total) == (0, 6)
        assert forced.adjusted_p == pytest.approx(forced.p_value) and forced.p_value < 0.01
        assert position.flips() == [forced]
        assert report.reproduced.point == 1.0
        assert position.weighted == pytest.approx(0.6, abs=1e-4)
        # what was forced is kept with each sample, and the prompt's ids are never stored
        samples = store.samples(result.experiment.id)
        assert {s.forced["text"] for s in samples} == {"search_files", "read_file"}
        stored = store.experiment_request(result.experiment)
        assert set(stored) == {"base_stream", "arms"}
        assert store.token_streams(sample_ids=[samples[0].id])[0].source == "continuation"

    async def test_reported_again_from_the_store(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        result = await execute_forking(store, prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM))
        again = experiment_report(store, result.experiment.id[:8])
        assert again.kind == "forking" and again.status == "done"
        assert again.to_json()["forking"]["flips"] == 1
        assert forking_result(store, result.experiment).to_json() == result.report.to_json()

    async def test_the_tokens_it_forced_lead_to_its_result(self, store, served):
        """Found in use: getting from a probe's result back to the token it forced meant
        reading the position off the page and finding it by hand. The step's tokens now carry
        the probes made from them, with each result at every position forced."""
        from tuningfork.ops.tokens import step_tokens

        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        result = await execute_forking(store, prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM))

        (probe,) = step_tokens(store, run_id, "0").to_json()["probes"]
        (position,) = probe["positions"]
        assert probe["experiment"]["id"] == result.experiment.id
        assert position["position"] == result.report.positions[0].position
        assert [alt["flips"] for alt in position["alternatives"]] == [True]

    async def test_a_target_described_in_conditions(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        rules = [{"if": "calls", "value": "search_files"}]

        plan = prepare_forking(store, run_id, "0", n=4, target_rules=rules, base_url=UPSTREAM)
        result = await execute_forking(store, plan)

        assert plan.target.description == "a behaviour that calls search_files"
        (position,) = result.report.positions
        assert (position.baseline.target.successes, position.alternatives[0].target.successes) == (0, 4)
        assert forking_result(store, result.experiment).to_json()["target"] == plan.target.description
        with pytest.raises(OperationError) as both:
            prepare_forking(store, run_id, "0", target_match="x", target_rules=rules, base_url=UPSTREAM)
        assert both.value.field == "target_rules" and "not both" in both.value.message

    async def test_a_model_that_decided_while_reasoning(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_REASONING, FORK_IN_REASONING[0][0])
        with pytest.raises(OperationError, match="decided before it") as refused:
            prepare_forking(store, run_id, "0", base_url=UPSTREAM)
        assert "the best in the reasoning" in refused.value.detail and "' read'" in refused.value.detail
        assert refused.value.field == "region"

        plan = prepare_forking(store, run_id, "0", region="reasoning", n=4, base_url=UPSTREAM)
        assert plan.candidates[0].part == "reasoning"
        assert any("regenerate the rest of the reasoning" in w for w in plan.warnings)
        result = await execute_forking(store, plan)
        assert result.report.to_json()["flips"] == 1

    async def test_a_target_holding_code_is_flagged_before_sampling(self, store, served):
        """Found on the real stack: at a step that ran code, every continuation wrote its own
        code, so the base's exact code was never reached again, and a real fork went unseen."""
        run_id = await recorded_and_scored(store, served, RUN_CODE, RUN_CODE[0][0])

        plan = prepare_forking(store, run_id, "0", n=2, base_url=UPSTREAM)
        coarse = prepare_forking(store, run_id, "0", n=2, normalizers=["tool_name"], base_url=UPSTREAM)

        assert "<ast:" in plan.target.description
        assert any("compared whole" in warning for warning in plan.warnings)
        assert "<ast:" not in coarse.target.description
        assert not any("compared whole" in warning for warning in coarse.warnings)

    async def test_a_picked_position_and_a_declared_target(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        position = prepare_forking(store, run_id, "0", n=1, base_url=UPSTREAM).candidates[0].position
        # a sampler that would never draw the alternative: picked by hand, it is forced anyway
        plan = prepare_forking(store, run_id, "0", positions=[position], min_p=0.5, n=3,
                               target_match="search_files", base_url=UPSTREAM)
        assert [arm["text"] for arm in plan.arms] == ["search_files", "read_file"]
        # What chooses positions is not used once they are named, and the plan says so.
        assert "min_p chooses positions only when none are named: not used" in plan.warnings
        both = prepare_forking(store, run_id, "0", positions=[position], region="all", top=2, n=1, base_url=UPSTREAM)
        assert "region and top choose positions only when none are named: not used" in both.warnings
        plain = prepare_forking(store, run_id, "0", positions=[position], n=1, base_url=UPSTREAM)
        assert not any("not used" in warning for warning in plain.warnings)
        result = await execute_forking(store, plan)
        (forced,) = result.report.positions[0].alternatives
        assert forced.target.successes == 3  # the target is now the alternative's behaviour
        with pytest.raises(OperationError, match="positions 0-"):
            prepare_forking(store, run_id, "0", positions=[9999], base_url=UPSTREAM)

    async def test_a_step_without_tokens_asks_for_a_score(self, store, served):
        served.toy.answers = FORK_IN_ANSWER
        store.begin_run("unscored", "t/1")
        store.record_step(request=REQUEST, response=as_response(message(FORK_IN_ANSWER[0][0]), "tool_calls"))
        with pytest.raises(OperationError, match="no tokens to fork from") as refused:
            prepare_forking(store, "unscored", "0", base_url=UPSTREAM)
        assert "--score" in refused.value.hint

    async def test_cancelling_keeps_what_was_drawn_and_the_family(self, store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_ANSWER, FORK_IN_ANSWER[0][0])
        plan = prepare_forking(store, run_id, "0", n=4, base_url=UPSTREAM)
        cancel = Cancel()
        seen = []

        def progress(event):
            if event.kind == "sample":
                seen.append(event)
                if len(seen) == 2:
                    cancel.cancel()

        with pytest.raises(Cancelled) as stopped:
            await execute_forking(store, plan, progress=progress, cancel=cancel)
        partial = stopped.value.partial
        assert partial is not None and partial.report.family == 1
        drawn = len(store.samples(partial.experiment.id))
        assert 2 <= drawn < plan.planned
        assert experiment_report(store, partial.experiment.id).status == "cancelled"

    def test_the_job_kind_is_strict(self):
        kind = CATALOG["forking-tokens"]
        params = kind.parse({"run": "r", "step": "0", "positions": [3, 5], "region": "reasoning"})
        assert params.positions == [3, 5] and params.n == 10
        with pytest.raises(OperationError):
            kind.parse({"run": "r", "step": "0", "region": "thoughts"})
        with pytest.raises(OperationError):
            kind.parse({"run": "r", "step": "0", "positions": ["3"]})
        described = kind.parse({"run": "r", "step": "0", "target_rules": [{"if": "calls", "value": "x"}]})
        assert described.target_rules == [{"if": "calls", "value": "x"}]
        with pytest.raises(OperationError):
            kind.parse({"run": "r", "step": "0", "target_rules": "calls x"})


class TestReport:
    def test_holm_over_the_planned_family_and_an_undrawn_arm(self):
        read, search = Outcome("tool_call", "read", "read"), Outcome("tool_call", "search", "search")
        arms = [
            {"position": 1, "id": 2, "text": "a", "p": 0.4, "original": False},
            {"position": 1, "id": 1, "text": "b", "p": 0.6, "original": True},
            {"position": 5, "id": 4, "text": "c", "p": 0.2, "original": False},
            {"position": 5, "id": 3, "text": "d", "p": 0.8, "original": True},
        ]
        report = forking_report(
            arms, [[search] * 10, [read] * 10, [], [read] * 10], per_arm=10,
            target=target_by_key("read", "read"),
        )
        first, second = report.positions
        assert first.alternatives[0].p_value < 0.001
        # not drawn: p = 1, and still one of the two tests Holm corrects over
        assert second.alternatives[0].p_value == 1.0 and report.family == 2
        assert first.alternatives[0].adjusted_p == pytest.approx(2 * first.alternatives[0].p_value)
        assert second.flips() == []

    def test_why_the_run_s_own_token_missed(self):
        """Found on the real stack: one warning called the target too fine where the step had
        simply not decided yet. What the misses did says which."""
        import json

        from tuningfork.analysis.normalize import called_tools

        def call(name, **arguments):
            key = json.dumps([{"arguments": arguments, "name": name}], sort_keys=True, separators=(",", ":"))
            return Outcome("tool_call", key, name)

        search, read = call("search_files", pattern="data.csv"), call("read_file", path="data.csv")
        scripts = [call("execute_code", code=f"<ast:{i}>") for i in range(10)]
        arms = [{"position": 7, "id": 2, "text": "a", "p": 0.5, "original": False},
                {"position": 7, "id": 1, "text": "b", "p": 0.5, "original": True}]

        def report(baseline, target):
            return forking_report(arms, [[read] * 10, baseline], per_arm=10, target=target)

        undecided = report([search] * 4 + [read] * 6, target_by_key(search.key, "search"))
        too_fine = report(scripts, target_by_key(scripts[0].key, "that script"))
        reproduced = report([search] * 8 + [read] * 2, target_by_key(search.key, "search"))
        unknown = report([read] * 10, target_by_pattern("row_count"))

        assert undecided.missed("undecided") == [7] and undecided.missed("too_fine") == []
        assert too_fine.missed("too_fine") == [7]
        assert reproduced.positions[0].why_missed is None
        assert unknown.positions[0].why_missed is None
        assert undecided.to_json()["missed"] == {"too_fine": [], "undecided": [7]}
        assert called_tools(Outcome("tool_call", "tools:read_file,search_files", "")) == ("read_file", "search_files")
        assert called_tools(Outcome("text", "text:done", "done")) is None

    def test_cut_off_is_truncated_and_ended_unreadable_is_a_behaviour(self):
        """Measured on a real probe: a tenth of the continuations were cut off writing code,
        and first read as 'unreadable'. Cut off is truncated, whatever was being written."""
        from tuningfork.experiment.forking import UNPARSED, forking_outcome

        cut = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}], UNPARSED: "never closed"}
        ended = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], UNPARSED: "never closed"}
        assert forking_outcome(cut, []).kind == "truncated"
        assert forking_outcome(ended, []).kind == "unparseable"

    def test_what_a_probe_can_detect(self):
        assert detectable(10, 1) == 5  # 10/10 against 5/10: p ≈ 0.03
        assert detectable(10, 15) == 7  # 10/10 against 3/10: p ≈ 0.0031, x15 < 0.05
        assert detectable(2, 30) is None

    def test_each_behaviour_says_whether_the_target_counts_it_and_which_tokens_reached_it(self, monkeypatch):
        """Found on find-the-file: the target, the base's exact call, counted a search with
        another pattern as a miss, and nothing said which tokens had led to it. The page, the
        command and an agent's answer each say it now."""
        from tuningfork.cli import _print_forking
        from tuningfork.ops.compact import forking as compact, forking_lines

        exact = Outcome("tool_call", "search:data.csv", "search_files(pattern='data.csv')")
        other = Outcome("tool_call", "search:*.csv", "search_files(pattern='*.csv')")
        stop = Outcome("text", "text:stop", "stop")
        arms = [{"position": 121, "id": 2, "text": "According", "p": 0.80, "original": False},
                {"position": 121, "id": 1, "text": "Wait", "p": 0.07, "original": True},
                {"position": 130, "id": 3, "text": "The", "p": 0.5, "original": True}]
        report = forking_report(
            arms, [[stop] * 9 + [other], [exact] * 2 + [other] * 2 + [stop] * 6, [stop] * 10],
            per_arm=10, target=target_by_key(exact.key, "the base's own behaviour"),
        )

        found = {cluster["outcome"]["label"]: cluster for cluster in report.to_json()["behaviours"]["clusters"]}
        assert {label: cluster["target_counted"] for label, cluster in found.items()} == {
            "stop": 0, "search_files(pattern='*.csv')": 0, "search_files(pattern='data.csv')": 2}
        assert found["search_files(pattern='*.csv')"]["by_arm"] == [
            {"position": 121, "text": "Wait", "original": True, "count": 2},
            {"position": 121, "text": "According", "original": False, "count": 1},
        ]
        assert [(arm["position"], arm["count"]) for arm in found["stop"]["by_arm"]] == [(121, 6), (121, 9), (130, 10)]

        console = Console(width=200, record=True)
        monkeypatch.setattr("tuningfork.cli.console", console)
        _print_forking(report)
        printed = console.export_text()
        assert "not target" in printed and "[121] 'Wait' as run 2, 'According' 1" in printed
        # A position a line, so no token is wrapped in two.
        assert "[121] 'Wait' as run 6, 'According' 9\n     [130] 'The' as run 10" in printed

        shown = compact(report.to_json())
        rows = {row["label"]: row for row in shown["behaviours"]["behaviours"]}
        assert rows["search_files(pattern='*.csv')"]["target"] == "not target"
        assert rows["search_files(pattern='data.csv')"]["target"] == "target"
        assert rows["search_files(pattern='*.csv')"]["led_to_by"] == "[121] 'Wait' as run 2, 'According' 1"
        assert any("not target; reached from [121] 'Wait' as run 2, 'According' 1" in line for line in forking_lines(shown))

    def test_a_probe_that_found_nothing_says_what_it_could_have_seen(self, monkeypatch):
        """Measured on find-the-file: 'Wait' led to the search 4 times in 10 and 'According'
        once, and the page said no forced token had changed what the model did. Ten
        continuations a token see only a token that decides the step: a null says that much."""
        from tuningfork.cli import _print_forking

        search, stop = Outcome("tool_call", "search", "search"), Outcome("text", "text:stop", "stop")
        arms = [{"position": 121, "id": 1, "text": "Wait", "p": 0.07, "original": True},
                {"position": 121, "id": 2, "text": "According", "p": 0.80, "original": False}]
        report = forking_report(arms, [[search] * 4 + [stop] * 6, [search] + [stop] * 9], per_arm=10,
                                target=target_by_key("search", "search"))
        assert report.positions[0].flips() == []
        assert (report.to_json()["per_arm"], report.to_json()["detectable"]) == (10, detectable(10, 1)) == (10, 5)
        console = Console(width=200, record=True)
        monkeypatch.setattr("tuningfork.cli.console", console)
        _print_forking(report)
        printed = console.export_text()
        assert "No forced token decided what the model did" in printed and "changed what the model did" not in printed
        assert "only a token that changes about 5 of its 10 would show" in printed


class TestContinuations:
    """What each continuation wrote, from the token it forced on: the model's deliberation, not
    only the behaviour it ended in (issue #29: the text was kept with every sample, and nothing
    read it back). Read from each sample's own tokens, in the parts the probe's format marks."""

    @staticmethod
    async def probe(store, served):
        run_id = await recorded_and_scored(store, served, FORK_IN_REASONING, FORK_IN_REASONING[0][0])
        plan = prepare_forking(store, run_id, "0", region="reasoning", n=2, base_url=UPSTREAM)
        return run_id, await execute_forking(store, plan)

    async def test_each_from_the_token_it_forced_in_the_parts_it_wrote(self, store, served):
        _, result = await self.probe(store, served)

        found = continuations(store, result.experiment.id[:8])
        data = found.to_json()

        (position,) = data["positions"]
        assert position["part"] == "reasoning" and position["lead_in"].endswith("I will")
        # The run's own token first, then the one forced, each by the behaviour it ended in.
        own, forced = position["tokens"]
        assert (own["text"], own["original"], forced["text"], forced["original"]) == (" read", True, " search", False)
        (read,), (search,) = own["behaviours"], forced["behaviours"]
        assert (read["letter"], read["count"], read["counted"]) == ("A", 2, 2)
        assert (search["letter"], search["count"], search["counted"]) == ("B", 2, 0)
        # The reasoning goes on from the forced token, then the call; the format's markers left out.
        first = search["continuations"][0]
        assert first["forced"] == {"text": " search", "part": "reasoning"}
        assert first["parts"] == [{"part": "reasoning", "text": "."},
                                  {"part": "tool_call", "text": 'call:search_files{pattern:<|"|>data.csv<|"|>}'}]
        assert first["finish_reason"] == "tool_calls" and first["generated"] > 0 and not first["counted"]
        assert data["count"] == 4 and data["target"].startswith("the base's own behaviour")
        # In words, as the terminal prints them and MCP pages them: the forced token marked.
        said = found.text()
        assert "  forced ' search': 2 continuations" in said
        assert "    A read_file(path='data.csv') — 2, what was counted" in said
        assert "reasoning: ⟦ search⟧.\n" in said and 'tool call: call:search_files{pattern:<|"|>data.csv<|"|>}' in said

    async def test_narrowed_to_a_position_a_token_or_a_behaviour(self, store, served):
        run_id, result = await self.probe(store, served)
        probe, at = result.experiment.id, result.report.positions[0].position

        assert continuations(store, probe, behaviour="b").to_json()["count"] == 2
        # A token as the probe shows it, or as typed without its space.
        only = continuations(store, probe, position=at, token="search").to_json()
        assert [token["text"] for token in only["positions"][0]["tokens"]] == [" search"]
        for asked, field in (({"position": at + 1}, "position"), ({"token": "write"}, "token"), ({"behaviour": "Z"}, "behaviour")):
            with pytest.raises(OperationError) as refused:
                continuations(store, probe, **asked)
            assert refused.value.field == field
        other = store.create_experiment(kind="resample", arm="original", run_id=run_id, step_idx=0, step_id=None,
                                        request={}, settings={})
        with pytest.raises(OperationError, match="continuations are a forking-token probe's"):
            continuations(store, other.id)

    async def test_one_as_tokens(self, store, served):
        _, result = await self.probe(store, served)
        # Sample 0 is of the token forced: drawn before the run's own.
        view = sample_tokens(store, result.experiment.id, 0).to_json()["view"]
        position = result.report.positions[0].position
        assert view["sample"]["forced"] == {"position": position, "text": " search", "original": False}
        assert view["stream"]["source"] == "continuation" and view["format"] == "gemma4"
        assert [kind for kind, _, _ in view["parts"]] == ["reasoning", "marker", "tool_call", "marker"]

    async def test_in_the_terminal(self, store, served):
        _, result = await self.probe(store, served)
        runner, home = CliRunner(), ["--home", str(store.home)]

        shown = runner.invoke(app, ["experiment", result.experiment.id[:8], "--continuations", "--behaviour", "B", *home])
        assert shown.exit_code == 0, shown.output
        assert f"the continuations of probe {result.experiment.id[:8]}" in shown.output
        assert "reasoning: ⟦ search⟧." in shown.output and "⟦ read⟧" not in shown.output
        alone = runner.invoke(app, ["experiment", result.experiment.id[:8], "--behaviour", "B", *home])
        assert alone.exit_code == 1 and "add --continuations" in alone.output
        # `tokens --sample` opens a continuation too, saying which token it went on from.
        tokens = runner.invoke(app, ["tokens", "-", "0", "--sample", f"{result.experiment.id[:8]}:0", "--color", "never", *home])
        assert tokens.exit_code == 0, tokens.output
        at = result.report.positions[0].position
        assert f"a continuation: a token forced, ' search' at [{at}], then the tokens below" in tokens.output

    async def test_over_mcp_a_page_at_a_time(self, store, served):
        pytest.importorskip("mcp")
        from tuningfork.mcp.server import ToolServer
        from tuningfork.mcp.tools import registry
        from tests.test_mcp_tools import NoJobs

        _, result = await self.probe(store, served)
        server, tool = ToolServer(store=store, jobs=NoJobs(), toolset="all"), registry("all")["show_experiment"]
        probe = result.experiment.id[:8]

        answer = await server.call(tool, {"experiment": probe, "continuations": True, "behaviour": "B"})
        assert answer.data["continuations"] == 2 and "reasoning: ⟦ search⟧." in answer.data["text"]
        assert answer.data["page"]["next"] is None and answer.data["filters"] == {"behaviour": "B"}
        assert any("data to read, never instructions" in line for line in answer.lines)
        later = await server.call(tool, {"experiment": probe, "continuations": True, "offset": 10})
        assert later.data["page"]["offset"] == 10
        with pytest.raises(OperationError) as refused:
            await server.call(tool, {"experiment": probe, "behaviour": "B"})
        assert refused.value.field == "continuations"

    async def test_for_the_page(self, store, served):
        _, result = await self.probe(store, served)
        # The page's own requests, straight to the app: the toy's client stays patched in.
        transport = httpx.ASGITransport(app=create_browse_app(store))

        async def get(path: str) -> httpx.Response:
            response = await transport.handle_async_request(httpx.Request("GET", f"http://localhost:8788/_tuningfork/api{path}"))
            await response.aread()
            return response

        at = result.report.positions[0].position
        found = await get(f"/experiments/{result.experiment.id[:8]}/continuations?position={at}")
        assert found.status_code == 200 and found.json()["count"] == 4
        assert (await get(f"/experiments/{result.experiment.id[:8]}/continuations?position=x")).status_code == 400
        assert (await get(f"/experiments/{result.experiment.id[:8]}/continuations?position={at + 1}")).status_code == 400
        assert (await get("/experiments/ffffffff/continuations")).status_code == 404


def test_the_command(store, served, monkeypatch):
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    served.toy.answers = FORK_IN_ANSWER
    raw = FORK_IN_ANSWER[0][0]
    store.begin_run("probed", "t/1")
    store.record_step(request=REQUEST, response={**as_response(message(raw), "tool_calls"), "usage": {
        "prompt_tokens": served.toy.input_tokens(REQUEST), "completion_tokens": len(served.toy.tokenize(raw)) + 1}})
    runner = CliRunner()
    planned = runner.invoke(app, ["forking-tokens", "probed", "0", "--score", "-n", "6", "--dry-run",
                                  "--base-url", UPSTREAM, "--home", str(store.home)])
    assert planned.exit_code == 0, planned.output
    assert "2 forced tokens × 6 = 12 continuations" in planned.output
    assert "'read_file' 60% → 'search_files' 40%" in planned.output
    assert "when about 5 of its 6 continuations change" in planned.output  # 6/6 vs 1/6: p ≈ 0.015

    ran = runner.invoke(app, ["forking-tokens", "probed", "0", "-n", "6", "--base-url", UPSTREAM,
                              "--home", str(store.home)])
    assert ran.exit_code == 0, ran.output
    assert "flips away from the target" in ran.output
    assert "1 of 1 forced tokens changed what the model did" in ran.output
    # three per arm cannot tell: said before a sample is drawn, not discovered after
    weak = runner.invoke(app, ["forking-tokens", "probed", "0", "-n", "3", "--dry-run",
                               "--base-url", UPSTREAM, "--home", str(store.home)])
    assert "no change can reach significance: raise -n" in weak.output


async def test_the_estimate_counts_the_prompt_each_continuation_processes(store, served, monkeypatch):
    """Found in use: a probe of 150 continuations estimated at 1.1 minutes took 6.2 — each
    processed the step's 13,729-token prompt first, and the estimate counted decoding only.
    The step's time to first token says how long the prompt takes: 1.5 s here, of 2.5 s."""
    served.toy.answers = FORK_IN_ANSWER
    raw = FORK_IN_ANSWER[0][0]
    generated = len(served.toy.tokenize(raw)) + 1
    store.begin_run("timed", "t/1")
    store.record_step(request=REQUEST, timings={"ttft_ms": 1500.0, "total_ms": 1500.0 + 10.0 * generated},
                      response={**as_response(message(raw), "tool_calls"), "usage": {
                          "prompt_tokens": served.toy.input_tokens(REQUEST), "completion_tokens": generated,
                          "prompt_tokens_details": {"cached_tokens": 0}}})
    run_id = store.get_run("timed").id
    await execute_score(store, prepare_score(store, run_id, "0", base_url=UPSTREAM))

    plan = prepare_forking(store, run_id, "0", n=6, base_url=UPSTREAM)
    estimate = plan.to_json()["estimate"]
    assert estimate["prefill_seconds"] == 1.5
    assert estimate["minutes"] == round((plan.expected_tokens() * 0.01 + plan.planned * 1.5) / 60, 1)

    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    planned = CliRunner().invoke(app, ["forking-tokens", "timed", "0", "-n", "6", "--dry-run",
                                       "--base-url", UPSTREAM, "--home", str(store.home)])
    assert planned.exit_code == 0, planned.output
    assert "processed first for each continuation (~1.5s)" in planned.output

    # Read from a cache, the prompt's time is scaled up to the whole of it.
    step = store.steps(run_id)[0]
    step.usage["prompt_tokens_details"] = {"cached_tokens": step.usage["prompt_tokens"] // 2}
    prompt = step.usage["prompt_tokens"]
    assert _prefill_seconds(step) == pytest.approx(1.5 * prompt / (prompt - prompt // 2))
    step.timings = {}
    assert _prefill_seconds(step) is None
