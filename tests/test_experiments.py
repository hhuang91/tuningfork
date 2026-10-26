"""Resampling and wording A/B tests from a recorded checkpoint.

Everything runs against the mock server. The A/B test uses a mock whose answer depends on
the skill text it was sent, so a correct substitution produces a real, detectable shift —
and a broken one would not.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tuningfork.analysis.distribution import compare_arms, distribution, target_by_pattern
from tuningfork.analysis.normalize import NORMALIZER_VERSION
from tuningfork.demo.mock import MockTurn
from tuningfork.experiment import (
    EXPERIMENT_HEADER,
    CheckpointError,
    NothingToCompare,
    SamplingPlan,
    SkillNotFound,
    declared_target,
    experiment_outcomes,
    history_caveat,
    load_checkpoint,
    prepare_request,
    resolve_target,
    run_ab,
    run_resample,
    substitute,
)
from tuningfork.analysis.tokens import TokenStream
from tuningfork.config import REDACTED
from tests.helpers import KEY, PROPS, SKILL, VARIANT, as_response, record_pilot, text_message

UPSTREAM = "http://upstream.test"


def write(keys: dict) -> MockTurn:
    return MockTurn(
        tool_calls=[
            {"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}
        ]
    )


@pytest.fixture
async def client(mock_server):
    mock_server.props = PROPS
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as http:
        yield http


class TestPrepareRequest:
    RECORDED = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 65536,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    def test_drops_streaming_and_caps_the_generation(self):
        sent, changes = prepare_request(self.RECORDED, SamplingPlan(max_tokens=4096))

        assert "stream" not in sent and "stream_options" not in sent
        assert sent["max_tokens"] == 4096
        assert changes["max_tokens"] == {"recorded": 65536, "sent": 4096}
        assert changes["stream"]["recorded"] is True

    def test_a_smaller_recorded_budget_is_kept(self):
        sent, changes = prepare_request({**self.RECORDED, "max_tokens": 512}, SamplingPlan())
        assert sent["max_tokens"] == 512
        assert "max_tokens" not in changes

    def test_overrides_are_applied_and_recorded(self):
        sent, changes = prepare_request(
            self.RECORDED, SamplingPlan(overrides={"temperature": 0.2})
        )
        assert sent["temperature"] == 0.2
        assert changes["temperature"] == {"recorded": None, "sent": 0.2}

    def test_a_recorded_seed_is_replaced_and_noted(self):
        sent, changes = prepare_request({**self.RECORDED, "seed": 42}, SamplingPlan())
        assert "seed" not in sent
        assert changes["seed"] == {"recorded": 42, "sent": "per sample"}

    def test_the_recorded_request_is_not_mutated(self):
        recorded = json.loads(json.dumps(self.RECORDED))
        prepare_request(recorded, SamplingPlan(overrides={"temperature": 0.1}))
        assert recorded == self.RECORDED

    def test_invalid_plans_are_rejected(self):
        with pytest.raises(ValueError):
            SamplingPlan(n=0)
        with pytest.raises(ValueError):
            SamplingPlan(normalizers=["nonsense"])


class TestCheckpoint:
    def test_loads_and_describes_what_the_model_had_just_seen(self, store):
        run_id = record_pilot(store)

        checkpoint = load_checkpoint(store, run_id, 1)

        assert checkpoint.describe() == "after the result of read_file"
        assert checkpoint.generated_history == 1

    def test_missing_step_is_a_clear_error(self, store):
        run_id = record_pilot(store)
        with pytest.raises(CheckpointError, match="steps 0-1"):
            load_checkpoint(store, run_id, 7)

    def test_caveat_distinguishes_a_clean_checkpoint(self, store):
        run_id = record_pilot(store)
        assert "identical context" in history_caveat(load_checkpoint(store, run_id, 0))
        assert "fast screen" in history_caveat(load_checkpoint(store, run_id, 1))


class TestResample:
    async def test_samples_are_drawn_stored_and_clustered(self, store, mock_server, client):
        exec_call = MockTurn(tool_calls=[{"name": "execute_code", "arguments": {"code": "x=1"}}])
        mock_server.script(
            *[MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": "data.csv"}}])] * 3,
            exec_call,
            exec_call,
        )
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=5), base_url=UPSTREAM, client=client
        )

        result = distribution(experiment_outcomes(store, experiment))
        assert [(c.letter, c.count) for c in result.clusters] == [("A", 3), ("B", 2)]
        assert "read_file" in result.clusters[0].outcome.label
        samples = store.samples(experiment.id)
        # A random base, recorded with the experiment: each sample's seed is base + i.
        base = experiment.settings["seed_base"]
        assert isinstance(base, int) and [s.seed for s in samples] == [base + i for i in range(5)]

    async def test_each_experiment_draws_new_seeds_unless_told_which(self, store, mock_server, client):
        """The same seeds would draw the same samples wherever the server reproduces them, so
        a second experiment would be no second sample."""
        checkpoint = load_checkpoint(store, record_pilot(store), 0)
        drawn = [await run_resample(store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM, client=client)
                 for _ in range(2)]
        assert drawn[0].settings["seed_base"] != drawn[1].settings["seed_base"]

        again = await run_resample(store, checkpoint, SamplingPlan(n=2, seed_base=drawn[0].settings["seed_base"]),
                                   base_url=UPSTREAM, client=client)
        assert [s.seed for s in store.samples(again.id)] == [s.seed for s in store.samples(drawn[0].id)]

    async def test_each_prompt_is_processed_whole_on_llama_cpp(self, store, mock_server, client):
        """A reused prompt cache made one request's behaviour depend on what the server ran
        before: every sample is asked for its whole prompt, and it says so."""
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        fresh = await run_resample(store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM, client=client)
        assert all(request.get("cache_prompt") is False for request in mock_server.requests[-2:])
        assert fresh.changes["cache_prompt"] == {"recorded": None, "sent": False}

        reused = await run_resample(store, checkpoint, SamplingPlan(n=2, prompt_cache=True), base_url=UPSTREAM, client=client)
        assert all("cache_prompt" not in request for request in mock_server.requests[-2:])
        assert "cache_prompt" not in reused.changes

        # llama.cpp's Messages endpoint drops the field (measured on b11108): not sent, and said.
        from tuningfork.wire import get_wire

        wire = get_wire("anthropic-messages")
        store.begin_run("cc", "t")
        store.record_step(
            request={"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
            response=as_response(text_message("ok")), wire=wire.name, normalize=wire.normalize_message,
        )
        messages = await run_resample(store, load_checkpoint(store, "cc", 0), SamplingPlan(n=1),
                                      base_url=UPSTREAM, client=client)
        assert "cache_prompt" not in mock_server.requests[-1] and "cache_prompt" not in messages.changes
        # This server answers a repeated prompt from its cache (measured): said, with the cure.
        assert any("drops cache_prompt" in note and "--no-cache-prompt" in note for note in messages.notes)

        # Started with --no-cache-prompt, it processes every prompt whole: nothing to qualify.
        import tuningfork.backends.llamacpp as llamacpp

        async def never(*args, **kwargs):
            return False

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(llamacpp, "reuses_prompts", never)
            whole = await run_resample(store, load_checkpoint(store, "cc", 0), SamplingPlan(n=1),
                                       base_url=UPSTREAM, client=client)
        assert not any("cache" in note for note in whole.notes)

        # Another server is not sent llama.cpp's field; the experiment says what that means.
        mock_server.props = None
        other = await run_resample(store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client)
        assert "cache_prompt" not in mock_server.requests[-1] and "cache_prompt" not in other.changes
        assert any("not llama.cpp's" in note for note in other.notes)

    async def test_requests_are_non_streaming_capped_and_tagged(
        self, store, mock_server, client
    ):
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=3), base_url=UPSTREAM, client=client
        )

        for payload in mock_server.requests:
            assert "stream" not in payload
            assert payload["max_tokens"] == 4096
            # History is held fixed: the messages are exactly the recorded ones.
            assert payload["messages"] == checkpoint.request["messages"]
        assert len({payload["seed"] for payload in mock_server.requests}) == 3
        chat_headers = [h for method, path, h in mock_server.seen if path.endswith("/completions")]
        assert all(h.get(EXPERIMENT_HEADER) == experiment.id for h in chat_headers)

    async def test_transport_errors_are_reported_but_not_rated(
        self, store, mock_server, client
    ):
        mock_server.script(MockTurn(content="ok"), MockTurn(status=500), MockTurn(content="ok"))
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=3), base_url=UPSTREAM, client=client
        )

        result = distribution(experiment_outcomes(store, experiment))
        assert result.errors == 1
        assert result.total == 2

    @pytest.mark.parametrize("concurrency", [1, 2])
    async def test_concurrency_is_bounded(self, store, mock_server, client, concurrency):
        mock_server.script(*[MockTurn(content="x", stall=0.03)] * 6)
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        await run_resample(
            store,
            checkpoint,
            SamplingPlan(n=6, concurrency=concurrency),
            base_url=UPSTREAM,
            client=client,
        )

        assert mock_server.peak_inflight <= concurrency

    async def test_truncated_samples_are_their_own_outcome(self, store, mock_server, client):
        mock_server.script(MockTurn(content="half a", finish_reason="length"))
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client
        )

        (outcome,) = experiment_outcomes(store, experiment)
        assert outcome.kind == "truncated"


class TestAMaskedContext:
    """Found in the redaction audit: a resample or an A/B sends the step's request as it is stored,
    and where text was masked as the step was recorded, every sample is sent `<redacted>` where the
    model saw the text — another prompt than the recorded one — and nothing said so."""

    #: What a tool gave back, a key in it: stored masked, `key=<redacted>`.
    MASKED = f"id,amount\nkey={KEY}\n"

    def test_a_resample_says_so(self, store):
        from tuningfork.ops.experiments import prepare_resample

        plan = prepare_resample(store, record_pilot(store, read=self.MASKED), "1", base_url=UPSTREAM)
        assert [warning for warning in plan.warnings if f"reads {REDACTED} in 1 place," in warning]
        assert plan.to_json()["warnings"] == plan.warnings

    def test_a_context_without_a_mask_says_nothing(self, store):
        from tuningfork.ops.experiments import prepare_resample

        assert prepare_resample(store, record_pilot(store), "1", base_url=UPSTREAM).warnings == []

    def test_an_ab_says_so(self, store, tmp_path):
        from tuningfork.ops.experiments import prepare_ab

        (tmp_path / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (tmp_path / "variant.md").write_text(VARIANT, encoding="utf-8")
        plan = prepare_ab(
            store, record_pilot(store, read=self.MASKED), "1", variant=tmp_path / "variant.md",
            original=tmp_path / "SKILL.md", base_url=UPSTREAM,
        )
        assert [warning for warning in plan.warnings if f"reads {REDACTED}" in warning]

    async def test_its_report_says_so_where_its_plan_did(self, store, mock_server, client):
        """Found in review: the plan's warning was the only word of it, and an agent over MCP saw
        a plan only on a dry run: the report says it again, as its page, the CLI and MCP read it."""
        from tuningfork.ops import compact
        from tuningfork.ops.inspect import experiment_report

        mock_server.script(MockTurn(content="abc"))
        checkpoint = load_checkpoint(store, record_pilot(store, read=self.MASKED), 1)
        experiment = await run_resample(store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client)
        report = experiment_report(store, experiment.id).to_json()
        assert f"reads {REDACTED} in 1 place," in report["masked"]
        assert f"warning: {report['masked']}" in compact.experiment_lines(compact.experiment(report))

    SOURCE = "Read the rows from postgres://report:Tr0ub4dor9@db.internal/app.\n"

    def test_an_ab_of_a_skill_masked_as_it_was_recorded_sends_each_wording_masked(self, store, tmp_path):
        """Found in review: a skill holding an example credential is stored masked, and an A/B of it
        was refused, the skill "not in the context at this step"."""
        from tuningfork.experiment.skill import MASKED_SKILL
        from tuningfork.ops.experiments import prepare_ab

        skill = SKILL.replace("Write the totals", self.SOURCE + "Write the totals")
        (tmp_path / "SKILL.md").write_text(skill, encoding="utf-8")
        (tmp_path / "variant.md").write_text(VARIANT.replace("Write the totals", self.SOURCE + "Write the totals"),
                                             encoding="utf-8")
        run = record_pilot(store, skill=skill)
        plan = prepare_ab(store, run, "1", variant=tmp_path / "variant.md", original=tmp_path / "SKILL.md",
                          base_url=UPSTREAM)
        assert MASKED_SKILL in plan.warnings
        sent, _ = substitute(plan.checkpoint.request, plan.original_text, plan.variant_text)
        system = sent["messages"][0]["content"]
        assert f"postgres://report:{REDACTED}@db.internal" in system and "row_count and amount_total" in system
        assert "Tr0ub4dor9" not in json.dumps(sent)

    def test_a_variant_written_from_the_skill_as_shown_is_set_on_the_file(self, store, tmp_path):
        """An agent writes a variant from show_skill, which shows the skill masked, each line at its
        number: the A/B tests it set on the file as it stands — each line it kept the file's, a key
        and a private key's lines with them — never the masks (found in review). A line it rewrote
        that still reads the mask is refused."""
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.experiments import prepare_ab
        from tests.test_store_masking import PRIVATE

        skill = SKILL + f"\nLog in with {KEY}.\n\n" + "\n".join(PRIVATE) + "\n"
        (tmp_path / "SKILL.md").write_text(skill, encoding="utf-8")
        run = record_pilot(store, skill=skill)
        shown = store.masking.shown(skill, lines=True).value
        assert KEY not in shown and PRIVATE[1] not in shown
        line = "Write the totals to report.json."
        better = "Write the totals to report.json, using the keys row_count and amount_total."
        plan = prepare_ab(store, run, "1", variant=tmp_path / "variant.md", variant_text=shown.replace(line, better),
                          original=tmp_path / "SKILL.md", base_url=UPSTREAM)
        assert plan.variant_text == skill.replace(line, better)
        with pytest.raises(OperationError) as refused:
            prepare_ab(store, run, "1", variant=tmp_path / "variant.md", original=tmp_path / "SKILL.md",
                       variant_text=shown.replace(f"Log in with {REDACTED}.", f"Log in once with {REDACTED}."),
                       base_url=UPSTREAM)
        assert refused.value.field == "variant_text" and REDACTED in refused.value.message

    def test_a_variant_keeps_a_key_whole_or_not_at_all_and_a_person_s_text_is_as_given(self, store, tmp_path):
        """A private key's lines each read `<redacted>` as show_skill shows them: a variant written
        from it that keeps some of them would test a broken key, and is refused; one that leaves them
        all out leaves the key out. A person's text, written from the file, reads the mask no more
        often than the file does, and is tested as given (found in review: set on the file as shown,
        the key a person had taken out was put back)."""
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.experiments import prepare_ab
        from tests.test_store_masking import PASSWORD, PRIVATE

        skill = (SKILL + f"\nLog in with {KEY}.\nUse password={PASSWORD} first.\nIf asked again, answer {PASSWORD}.\n\n"
                 + "\n".join(PRIVATE) + "\n\nThen send it.\n")
        (tmp_path / "SKILL.md").write_text(skill, encoding="utf-8")
        run = record_pilot(store, skill=skill)
        shown = store.masking.shown(skill, lines=True).value.splitlines(keepends=True)
        body = shown.index(f"{REDACTED}\n")  # the key's first line shown, then its body, a line apiece

        def tested(text: str) -> str:
            return prepare_ab(store, run, "1", variant=tmp_path / "variant.md", variant_text=text,
                              original=tmp_path / "SKILL.md", base_url=UPSTREAM).variant_text

        with pytest.raises(OperationError) as refused:
            tested("".join(shown[:body] + shown[body + 1:]))
        assert refused.value.field == "variant_text" and "kept whole or not at all" in refused.value.message
        out = tested("".join(shown[:body - 1] + shown[body + 3:]))
        assert PRIVATE[1] not in out and PRIVATE[2] not in out and f"Log in with {KEY}." in out
        written = skill.replace("\n".join(PRIVATE) + "\n", "")
        assert tested(written) == written
        # A password named on one line and said bare on another: the line that names it rewritten, the
        # other kept as shown, its runs would be recorded with the password bare.
        with pytest.raises(OperationError) as bared:
            tested("".join(shown).replace(f"Use password={REDACTED} first.", "Use the stored password first."))
        assert bared.value.field == "variant_text" and "to read as no key" in bared.value.message

    def test_a_person_s_variant_is_tested_as_given(self, store):
        """A text that reads the mask no more often than the skill's file was written from the file: a
        person's. It is tested as given, never matched against the skill as shown nor given the file's
        line breaks (found in review: matched so, a key the person took out was put back)."""
        from tuningfork.ops.experiments import variant_onto
        from tests.test_store_masking import PRIVATE

        original = (SKILL + f"\nLog in with {KEY}.\n\n" + "\n".join(PRIVATE) + "\n").replace("\n", "\r\n")
        written = SKILL + f"\nLog in with {KEY}, once.\n"
        assert variant_onto(store, original, written) == written

    def test_an_edit_inside_masked_text_is_said_to_be_one(self, store, tmp_path):
        from tuningfork.experiment.skill import NothingToCompare

        skill = SKILL.replace("Write the totals", self.SOURCE + "Write the totals")
        run = record_pilot(store, skill=skill)
        checkpoint = load_checkpoint(store, run, 1)
        with pytest.raises(NothingToCompare, match="the edit falls inside text storage masks"):
            substitute(checkpoint.request, skill, skill.replace("Tr0ub4dor9", "Tr0ub4dor8"))

    def test_a_skill_masked_otherwise_than_storage_masks_it_now_is_said_to_be(self, store, tmp_path):
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.experiments import prepare_ab

        skill = SKILL.replace("Write the totals", self.SOURCE + "Write the totals")
        (tmp_path / "SKILL.md").write_text(skill, encoding="utf-8")
        (tmp_path / "variant.md").write_text(VARIANT, encoding="utf-8")
        # Stored under other patterns: masked where today's would not mask it.
        run = record_pilot(store, skill=skill.replace("db.internal/app", f"{REDACTED}/app"))
        with pytest.raises(OperationError) as refused:
            prepare_ab(store, run, "1", variant=tmp_path / "variant.md", original=tmp_path / "SKILL.md",
                       base_url=UPSTREAM)
        assert "holds text storage masks" in refused.value.detail


class TestATargetKeptAsDeclared:
    async def test_a_target_read_from_masked_answers_still_names_its_behaviour(self, store, mock_server, client):
        """Found in review: an experiment's settings were masked as they were stored, and a target keyed
        from an answer masked once, masked again — a command's second password — named no behaviour."""
        from tuningfork.analysis.distribution import target_by_key
        from tuningfork.analysis.normalize import DEFAULT_NORMALIZERS, outcome_for

        command = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"
        mock_server.script(*[MockTurn(content=command)] * 4)
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        key = outcome_for(store.masking.of(as_response(text_message(command))).value, list(DEFAULT_NORMALIZERS)).key
        target = target_by_key(key, "A from an earlier resample: says the command")
        ab = await run_ab(store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM, original_text=SKILL,
                          variant_text=VARIANT, target=target, client=client)

        arm = store.get_experiment(ab.original.id)
        stored, _ = declared_target(arm)
        assert [stored.matches(outcome) for outcome in experiment_outcomes(store, arm)] == [True, True]


class TestLogprobs:
    """A resample can keep each sample's tokens, for a heatmap of where it wavered."""

    @staticmethod
    def entry(token_id: int, token: str, logprob: float = -0.01) -> dict:
        piece = {"id": token_id, "token": token, "bytes": list(token.encode()), "logprob": logprob}
        return {**piece, "top_logprobs": [piece, {"id": 99, "token": "x", "bytes": [120], "logprob": -5.0}]}

    async def test_a_samples_tokens_are_kept_apart_from_its_response(self, store, mock_server, client):
        mock_server.props = {**PROPS, "chat_template": "{{ '<|tool_call>call:' }}"}
        entries = [
            self.entry(100, "<|channel>"), self.entry(1, "thought"), self.entry(2, "\n"),
            self.entry(3, "hm"), self.entry(101, "<channel|>"), self.entry(4, "abcdefghijklmnop"),
            self.entry(50, ""),
        ]
        mock_server.script(MockTurn(content="abcdefghijklmnop", reasoning="hm", logprobs={"content": entries}))
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1, logprobs=5), base_url=UPSTREAM, client=client
        )

        sent = mock_server.requests[-1]
        assert (sent["logprobs"], sent["top_logprobs"]) == (True, 5)
        assert experiment.changes["top_logprobs"] == {"recorded": None, "sent": 5}
        assert experiment.settings["format"] == "gemma4"
        (sample,) = store.samples(experiment.id)
        assert "logprobs" not in store.sample_response(sample)["choices"][0]
        (record,) = store.token_streams(sample_ids=[sample.id])
        assert (record.source, record.method, record.top_k) == ("sampled", "chat-logprobs", 5)
        stream = TokenStream.from_json(store.token_stream_data(record))
        assert stream.text() == "<|channel>thought\nhm<channel|>abcdefghijklmnop"
        assert [kind for kind, _, _ in stream.parts] == ["marker", "reasoning", "marker", "content", "marker"]
        from tuningfork.ops.inspect import experiment_report

        assert experiment_report(store, experiment.id).tokens_kept == 1
        assert not any("sent none back" in note for note in experiment.notes)

    async def test_a_samples_tokens_are_masked_of_what_its_answer_masked(self, store, mock_server, client):
        """The model says a password back bare, a few characters a token: no pattern reads it there,
        and the answer's own mask names it."""
        mock_server.props = {**PROPS, "chat_template": "{{ '<|tool_call>call:' }}"}
        pieces = ["password=", "Tr0u", "b4do", "r3xyz", " so", " Tr0u", "b4do", "r3xyz"]
        entries = [self.entry(number, piece) for number, piece in enumerate(pieces, start=1)]
        mock_server.script(MockTurn(content="".join(pieces), logprobs={"content": entries}))
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1, logprobs=5), base_url=UPSTREAM, client=client
        )

        (sample,) = store.samples(experiment.id)
        (record,) = store.token_streams(sample_ids=[sample.id])
        kept = json.dumps(store.token_stream_data(record))
        assert "b4do" not in kept and '" so"' in kept

    async def test_a_server_that_sends_none_back_is_said_to(self, store, mock_server, client):
        """Found in review: asked for and accepted with nothing back — a server without logprobs —
        the resample kept no tokens and said nothing, and its page said each sample had kept them."""
        from tuningfork.ops.inspect import experiment_report

        mock_server.script(MockTurn(content="abc"), MockTurn(content="abd"))
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=2, logprobs=5), base_url=UPSTREAM, client=client
        )

        said = "each sample's tokens were asked for, and the server sent none back: its chat endpoint returns no logprobs"
        assert said in experiment.notes and said in store.get_experiment(experiment.id).notes
        report = experiment_report(store, experiment.id)
        assert report.tokens_kept == 0 and said in report.to_json()["notes"]

    async def test_without_logprobs_nothing_changes(self, store, mock_server, client):
        checkpoint = load_checkpoint(store, record_pilot(store), 0)
        experiment = await run_resample(store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client)
        assert "logprobs" not in mock_server.requests[-1]
        assert store.token_streams(sample_ids=[s.id for s in store.samples(experiment.id)]) == []

    def test_a_messages_step_has_no_logprobs_to_ask_for(self, store):
        from tuningfork.ops.common import OperationError
        from tuningfork.ops.experiments import prepare_resample
        from tuningfork.wire import get_wire

        wire = get_wire("anthropic-messages")
        store.begin_run("cc", "t")
        store.record_step(
            request={"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
            response=as_response(text_message("ok")), wire=wire.name, normalize=wire.normalize_message,
        )
        with pytest.raises(OperationError, match="no logprobs") as refused:
            prepare_resample(store, "cc", "0", base_url=UPSTREAM, logprobs=5)
        assert "--score" in refused.value.hint

    def test_logprobs_are_bounded(self):
        with pytest.raises(ValueError, match="logprobs"):
            SamplingPlan(logprobs=0)
        with pytest.raises(ValueError, match="logprobs"):
            SamplingPlan(logprobs=21)


class TestParity:
    async def test_matching_defaults_raise_no_notes(self, store, client):
        checkpoint = load_checkpoint(store, record_pilot(store), 0)
        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client
        )
        assert experiment.notes == []
        assert experiment.server_defaults["temperature"] == 1.0

    async def test_a_changed_server_default_is_flagged(self, store, mock_server, client):
        mock_server.props = {"default_generation_settings": {"params": {"temperature": 0.7}}}
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client
        )

        assert any("temperature is 0.7 now" in note for note in experiment.notes)

    async def test_an_overridden_parameter_is_not_compared(self, store, mock_server, client):
        mock_server.props = {"default_generation_settings": {"params": {"temperature": 0.7}}}
        checkpoint = load_checkpoint(store, record_pilot(store), 0)

        experiment = await run_resample(
            store,
            checkpoint,
            SamplingPlan(n=1, overrides={"temperature": 1.0}),
            base_url=UPSTREAM,
            client=client,
        )

        assert not any("temperature" in note for note in experiment.notes)

    async def test_unrecorded_defaults_are_admitted(self, store, client):
        checkpoint = load_checkpoint(store, record_pilot(store, server_defaults={}), 0)
        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client
        )
        assert any("not recorded" in note for note in experiment.notes)

    async def test_a_different_model_is_flagged(self, store, client):
        checkpoint = load_checkpoint(store, record_pilot(store, model="other-model"), 0)
        experiment = await run_resample(
            store, checkpoint, SamplingPlan(n=1), base_url=UPSTREAM, client=client
        )
        assert any("other-model" in note for note in experiment.notes)


class TestAB:
    @staticmethod
    def wording_sensitive(payload: dict) -> MockTurn:
        """A model that follows the key names only when the skill states them."""
        if "row_count and amount_total" in payload["messages"][0]["content"]:
            return write({"row_count": 4, "amount_total": 61.5})
        return write({"count": 4, "total": 61.5})

    async def test_a_real_wording_effect_is_detected(self, store, mock_server, client):
        mock_server.responder = self.wording_sensitive
        checkpoint = load_checkpoint(store, record_pilot(store), 1)

        ab = await run_ab(
            store,
            checkpoint,
            SamplingPlan(n=10),
            base_url=UPSTREAM,
            original_text=SKILL,
            variant_text=VARIANT,
            client=client,
        )
        result = compare_arms(
            experiment_outcomes(store, ab.original),
            experiment_outcomes(store, ab.variant),
            target=target_by_pattern("row_count"),
        )

        assert result.target.original.successes == 0
        assert result.target.variant.successes == 10
        assert result.target.significant

    async def test_no_effect_is_not_reported_as_one(self, store, mock_server, client):
        mock_server.responder = lambda payload: write({"count": 4, "total": 61.5})
        checkpoint = load_checkpoint(store, record_pilot(store), 1)

        ab = await run_ab(
            store, checkpoint, SamplingPlan(n=10), base_url=UPSTREAM,
            original_text=SKILL, variant_text=VARIANT, client=client,
        )
        result = compare_arms(
            experiment_outcomes(store, ab.original),
            experiment_outcomes(store, ab.variant),
            target=target_by_pattern("row_count"),
        )

        assert not result.target.significant

    async def test_the_arms_differ_only_in_the_skill(self, store, mock_server, client):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)

        await run_ab(
            store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM,
            original_text=SKILL, variant_text=VARIANT, client=client,
        )

        original, variant = mock_server.requests[0], mock_server.requests[-1]
        assert SKILL in original["messages"][0]["content"]
        assert VARIANT in variant["messages"][0]["content"]
        assert SKILL not in variant["messages"][0]["content"]
        assert original["messages"][1:] == variant["messages"][1:]
        assert {k: v for k, v in original.items() if k not in ("messages", "seed")} == {
            k: v for k, v in variant.items() if k not in ("messages", "seed")
        }

    async def test_the_arms_are_stored_as_one_group(self, store, client):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)

        ab = await run_ab(
            store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM,
            original_text=SKILL, variant_text=VARIANT,
            original_path="SKILL.md", variant_path="SKILL.v2.md", client=client,
        )

        group = store.experiment_group(ab.original.group_id)
        assert [e.arm for e in group] == ["original", "variant"]
        assert group[1].variant["variant_path"] == "SKILL.v2.md"
        assert group[1].variant["spans"][0]["role"] == "system"

    async def test_a_skill_that_is_not_in_the_context_fails_before_sampling(
        self, store, mock_server, client
    ):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)

        with pytest.raises(SkillNotFound):
            await run_ab(
                store, checkpoint, SamplingPlan(n=5), base_url=UPSTREAM,
                original_text="# A skill nobody loaded\n\nDo things.\n",
                variant_text=VARIANT, client=client,
            )

        assert mock_server.requests == []  # nothing was spent on a doomed experiment


class TestTargets:
    async def test_a_letter_names_a_cluster_from_an_earlier_resample(
        self, store, mock_server, client
    ):
        mock_server.script(write({"count": 4, "total": 61.5}), write({"count": 4, "total": 61.5}))
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        await run_resample(
            store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM, client=client
        )

        target, normalizers = resolve_target(store, checkpoint, letter="a")

        assert "count" in target.description
        assert normalizers == []

    async def test_a_letter_target_keeps_its_resample_so_reports_letter_as_it_did(
        self, store, mock_server, client
    ):
        """An A/B, a screen or a probe of a resample's B letters that behaviour B too. A target
        stored before targets kept their resample still names it in its description."""
        from tuningfork.experiment.sampling import target_letters

        mock_server.script(write({"count": 4, "total": 61.5}), write({"row_count": 4}), write({"row_count": 4}))
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        resample = await run_resample(store, checkpoint, SamplingPlan(n=3), base_url=UPSTREAM, client=client)

        target, _ = resolve_target(store, checkpoint, letter="B")

        assert (target.spec["letter"], target.spec["resample"]) == ("B", resample.id)
        letters = target_letters(store, target.spec)
        assert letters[target.spec["value"]] == "B" and sorted(letters.values()) == ["A", "B"]
        older = {key: value for key, value in target.spec.items() if key not in ("letter", "resample")}
        assert target_letters(store, older) == letters
        assert target_letters(store, older, ["tool_name"]) is None  # another lens keys them otherwise
        assert target_letters(store, {"kind": "pattern", "value": "row_count"}) is None

    def test_a_letter_without_an_earlier_resample_is_refused(self, store):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        with pytest.raises(CheckpointError, match="Run a resample first"):
            resolve_target(store, checkpoint, letter="A")

    def test_declaring_both_forms_is_refused(self, store):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        with pytest.raises(ValueError):
            resolve_target(store, checkpoint, letter="A", pattern="x")


class TestStoredTargets:
    """A target declared before sampling is re-tested only as it was declared."""

    @staticmethod
    def experiment(store, target: dict, **settings):
        run_id = record_pilot(store)
        return store.create_experiment(
            kind="ab", arm="original", run_id=run_id, step_idx=1,
            step_id=store.step(run_id, 1).id, request={"messages": []},
            settings={"n": 2, "normalizers": [], "target": target, **settings},
        )

    def test_a_target_is_kept_under_its_own_normalizers(self, store):
        found = self.experiment(store, {"kind": "pattern", "value": "row_count"},
                                normalizer_version=NORMALIZER_VERSION)
        target, why_not = declared_target(found)
        assert target is not None and why_not is None
        same = ["tool_call_json", "python_ast", "json_content"]
        assert declared_target(found, same)[0] is not None

    def test_a_pattern_is_not_carried_into_another_lens(self, store):
        """Found in review: under tool_name, /row_count/ matched nothing in either arm, and
        a 0/20 → 20/20 result re-reported as 'not significant'."""
        found = self.experiment(store, {"kind": "pattern", "value": "row_count"})
        target, why_not = declared_target(found, ["tool_name"])
        assert target is None
        assert "different lens" in why_not

    def test_a_key_from_older_normalizers_is_not_re_tested(self, store):
        found = self.experiment(store, {"kind": "key", "value": "k", "description": "A"})
        target, why_not = declared_target(found)  # stored before versions were recorded
        assert target is None
        assert "changed since" in why_not

    def test_new_experiments_record_the_normalizer_version(self):
        assert SamplingPlan().as_settings()["normalizer_version"] == NORMALIZER_VERSION


class TestNothingToCompareEndToEnd:
    async def test_an_ab_that_would_compare_a_request_with_itself_draws_nothing(
        self, store, mock_server, client
    ):
        checkpoint = load_checkpoint(store, record_pilot(store), 1)
        with pytest.raises(NothingToCompare):
            await run_ab(store, checkpoint, SamplingPlan(n=2), base_url=UPSTREAM,
                         original_text=SKILL, variant_text=SKILL, client=client)
        assert mock_server.requests == []


def test_a_checkpoint_after_harness_bookkeeping_describes_what_came_before(store):
    """Claude Code follows each tool result with a token budget as a system message; every
    checkpoint read 'after system: <total_tokens>…'."""
    run_id = record_pilot(store)
    checkpoint = load_checkpoint(store, run_id, 1)
    checkpoint.request["messages"].append({"role": "system", "content": "<total_tokens>9</total_tokens>"})
    assert checkpoint.describe() == "after the result of read_file"
