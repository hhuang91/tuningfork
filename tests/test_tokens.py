"""Token streams: the numbers a heatmap shows, where a probe could fork, and redaction."""

from __future__ import annotations

import json
import math

import pytest

from tuningfork.analysis.stats import holm
from tuningfork.analysis.tokens import (
    CONTENT,
    MARKER,
    REASONING,
    TOOL_CALL,
    Sampler,
    TokenStream,
    candidate_at,
    candidates,
    display_groups,
    effective,
    entropy_bits,
    from_entries,
    margin,
    probability,
    redact_stream,
    split_logprobs,
    surprise_bits,
)
from tuningfork.config import REDACTED, redact_json, redact_spans


def entry(token_id, token, p, alternatives=(), raw=None):
    """One llama.cpp token entry: chat `logprobs.content` and `/completion` share the shape."""
    top = [(token_id, token, p), *alternatives]
    return {
        "id": token_id,
        "token": token,
        "bytes": list(raw if raw is not None else token.encode()),
        "logprob": math.log(p) if p > 0 else -3.4e38,
        "top_logprobs": [
            {"id": i, "token": t, "bytes": list(t.encode()), "logprob": math.log(q) if q > 0 else -3.4e38}
            for i, t, q in sorted(top, key=lambda item: -item[2])
        ],
    }


class TestNumbers:
    def test_a_fair_coin_is_one_bit(self):
        assert entropy_bits([math.log(0.5), math.log(0.5)]) == pytest.approx(1.0)

    def test_three_ways(self):
        assert entropy_bits([math.log(0.5), math.log(0.25), math.log(0.25)]) == pytest.approx(1.5)

    def test_certainty_is_zero(self):
        assert entropy_bits([0.0]) == pytest.approx(0.0)

    def test_the_unseen_tail_counts_as_one_more_outcome(self):
        # 0.5 and 0.3 known; the other 0.2 is one outcome: a lower bound on the true entropy
        expected = -(0.5 * math.log2(0.5) + 0.3 * math.log2(0.3) + 0.2 * math.log2(0.2))
        assert entropy_bits([math.log(0.5), math.log(0.3)]) == pytest.approx(expected)

    def test_llama_cpp_floor_reads_as_impossible(self):
        assert probability(-3.4e38) == 0.0
        assert entropy_bits([0.0, -3.4e38]) == pytest.approx(0.0)

    def test_margin_is_top_minus_runner_up(self):
        assert margin([math.log(0.3), math.log(0.6), math.log(0.1)]) == pytest.approx(0.3)

    def test_surprise(self):
        assert surprise_bits(math.log(0.25)) == pytest.approx(2.0)
        assert surprise_bits(None) is None


class TestStream:
    def stream(self):
        return from_entries([
            entry(1, "The", 0.9, [(2, "A", 0.1)]),
            entry(3, " count", 0.54, [(4, " row", 0.38), (5, " total", 0.03), (6, " c", 0.02)]),
            entry(7, "", 1.0),
        ])

    def test_from_entries_keeps_ids_pieces_and_alternatives(self):
        stream = self.stream()
        assert stream.ids == [1, 3, 7]
        assert stream.top_k == 4
        assert stream.pieces[4] == " row"
        assert stream.text() == "The count"
        assert stream.entropy(0) == pytest.approx(entropy_bits([math.log(0.9), math.log(0.1)]))

    def test_round_trips_through_json(self):
        stream = self.stream()
        stream.parts = [(CONTENT, 0, 2), (MARKER, 2, 3)]
        stream.thinking = True
        again = TokenStream.from_json(json.loads(json.dumps(stream.to_json())))
        assert again.ids == stream.ids
        assert again.pieces == stream.pieces
        assert again.parts == stream.parts
        assert again.top_ids == stream.top_ids
        assert again.thinking is True
        # what was never known stays unknown, and is not written
        stream.thinking = None
        assert "thinking" not in stream.to_json() and TokenStream.from_json(stream.to_json()).thinking is None
        # rounded to 5 places when stored: far finer than any heatmap shows
        assert again.logprobs == pytest.approx(stream.logprobs, abs=1e-5)

    def test_a_character_split_into_byte_tokens_is_shown_once(self):
        e_acute = "é".encode()
        stream = from_entries([
            entry(1, "caf", 1.0),
            entry(2, "", 1.0, raw=e_acute[:1]),
            entry(3, "", 1.0, raw=e_acute[1:]),
            entry(4, "!", 1.0),
        ])
        assert display_groups(stream) == [(0, 1, "caf"), (1, 3, "é"), (3, 4, "!")]
        assert stream.pieces[2] == [e_acute[0]]
        assert stream.text() == "café!"


class TestSampler:
    known = {1: math.log(0.6), 2: math.log(0.3), 3: math.log(0.08), 4: math.log(0.02)}

    def test_min_p_cuts_relative_to_the_top(self):
        cut = effective(self.known, Sampler(min_p=0.05))
        assert set(cut) == {1, 2, 3}  # 0.02 < 0.05 * 0.6
        assert sum(cut.values()) == pytest.approx(1.0)

    def test_top_k_and_top_p(self):
        assert set(effective(self.known, Sampler(top_k=2))) == {1, 2}
        assert set(effective(self.known, Sampler(top_p=0.85))) == {1, 2}

    def test_temperature_zero_is_greedy(self):
        assert effective(self.known, Sampler(temperature=0)) == {1: 1.0}

    def test_a_lower_temperature_sharpens(self):
        cold = effective(self.known, Sampler(temperature=0.5))
        assert cold[1] > 0.6

    def test_the_step_settings_win_over_the_server_defaults(self):
        sampler = Sampler.of({"temperature": 0.7}, {"temperature": 1.0, "top_k": 64, "samplers": ["top_k", "min_p", "temperature"]})
        assert sampler.temperature == 0.7
        assert sampler.top_k == 64
        assert sampler.order == ("top_k", "min_p", "temperature")


class TestCandidates:
    def stream(self):
        stream = from_entries([
            entry(10, "<|channel>", 1.0),
            entry(11, "I", 0.55, [(12, "Let", 0.45)]),
            entry(13, "<channel|>", 1.0),
            entry(14, "search", 0.92, [(15, "read", 0.08)]),
            entry(16, " count", 0.54, [(17, " row", 0.38), (18, " c", 0.05), (19, " total", 0.03)]),
            entry(20, "}", 1.0),
        ])
        stream.parts = [(MARKER, 0, 1), (REASONING, 1, 2), (MARKER, 2, 3), (TOOL_CALL, 3, 6)]
        return stream

    def test_answer_positions_ranked_by_the_runner_up(self):
        found = candidates(self.stream(), Sampler(), top=5, region="answer")
        assert [c.position for c in found] == [4, 3]
        assert found[0].chosen.text == " count"
        # " c" re-spells " count", and " total" is under the 5% the sampler would draw
        assert [alt.text for alt in found[0].alternatives] == [" row"]

    def test_the_reasoning_region(self):
        found = candidates(self.stream(), Sampler(), region="reasoning")
        assert [c.position for c in found] == [1]

    def test_the_sampler_decides_what_could_have_been_drawn(self):
        # min_p 0.1 of 0.92 drops "read" at position 3
        found = candidates(self.stream(), Sampler(min_p=0.1), region="answer")
        assert [c.position for c in found] == [4]

    def test_a_re_spelling_is_not_the_runner_up(self):
        """Found on a real sample: " contents" chosen at 21%, " content" at 72% — the same
        word begun, not another way to go. It is neither listed nor the runner-up."""
        stream = from_entries([entry(1, " contents", 0.21, [(2, " content", 0.72), (3, " skill", 0.05)])])
        found = candidate_at(stream, 0, Sampler())
        assert [alt.text for alt in found.alternatives] == [" skill"]
        assert found.runner_up == pytest.approx(0.05 / (0.21 + 0.72 + 0.05), abs=1e-3)

    def test_a_picked_position_keeps_its_unlikely_alternatives(self):
        picked = candidate_at(self.stream(), 4, Sampler(min_p=0.9), picked=True)
        assert [alt.text for alt in picked.alternatives] == [" row", " total"]
        assert picked.picked

    def test_without_parts_everything_is_the_answer(self):
        stream = self.stream()
        stream.parts = []
        assert {c.position for c in candidates(stream, Sampler(), region="answer")} == {1, 3, 4}

    def test_an_unknown_region_is_refused(self):
        with pytest.raises(ValueError, match="region"):
            candidates(self.stream(), Sampler(), region="thoughts")


class TestLogprobsAndRedaction:
    def response(self, entries):
        return {
            "choices": [{"index": 0, "finish_reason": "stop", "logprobs": {"content": entries},
                         "message": {"role": "assistant", "content": "".join(e["token"] for e in entries)}}],
        }

    def test_logprobs_come_out_before_redaction(self):
        entries = [entry(1, "abcdefghijklmnop", 1.0)]
        response, found = split_logprobs(self.response(entries))
        assert found == entries
        assert "logprobs" not in response["choices"][0]
        # Redacted in place, the logprobs' own "token" key would have matched `token: <12+>`
        assert REDACTED in json.dumps(redact_json(self.response(entries))["choices"][0]["logprobs"])
        assert redact_json(response)["choices"][0]["message"]["content"] == "abcdefghijklmnop"
        assert redact_stream(from_entries(found)).pieces[1] == "abcdefghijklmnop"

    def test_a_response_without_logprobs_is_left_as_it_is(self):
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert split_logprobs(response) == (response, None)

    def test_a_secret_split_across_tokens_is_masked_token_by_token(self):
        stream = from_entries([
            entry(1, "key ", 1.0),
            entry(2, "sk", 0.9, [(9, "the", 0.1)]),
            entry(3, "-abcdefgh", 1.0),
            entry(4, "ijklmnopqrst", 1.0, [(8, "ijklmnopqrsu", 0.0001)]),
            entry(5, " done", 1.0),
        ])
        masked = redact_stream(stream)
        assert masked.ids == [1, None, None, None, 5]
        assert masked.redacted == [(1, 4)]
        assert masked.top_ids[1] == [] and masked.top_ids[3] == []
        assert set(masked.pieces) == {1, 5}
        assert "abcdefgh" not in json.dumps(masked.to_json())
        assert masked.text() == f"key {REDACTED}{REDACTED}{REDACTED} done"

    def test_redacted_positions_are_never_candidates(self):
        stream = from_entries([
            entry(1, "sk-abcdefghijklmnopqrst", 0.5, [(2, "x", 0.5)]),
            entry(3, " ok", 0.5, [(4, " no", 0.5)]),
        ])
        masked = redact_stream(stream)
        assert [c.position for c in candidates(masked, Sampler())] == [1]

    def test_spans_are_found_in_the_text_and_in_its_json_form(self):
        text = 'line\n"api_key": "abcdefghijklmnop" tail'
        (start, end), = redact_spans(text)
        assert text[start:end] == "abcdefghijklmnop"

    def test_nothing_to_mask(self):
        stream = from_entries([entry(1, "hello", 1.0)])
        assert redact_stream(stream) is stream


class TestHolm:
    def test_matches_r(self):
        # p.adjust(c(0.01, 0.04, 0.03, 0.005), "holm") in R
        assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_capped_and_monotone(self):
        assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
        assert holm([0.001, 0.002, 0.9]) == pytest.approx([0.003, 0.004, 0.9])

    def test_empty(self):
        assert holm([]) == []


class TestToolChoice:
    """The model's choice of tool as a number: at the first token of the call's name, each
    offered tool's sampled chance. The write path on meeting-action-items — a script through
    execute_code, or write_file — was decided at this one position."""

    TOOLS = ("read_file", "write_file", "execute_code", "search_files")

    def stream(self, name_entry):
        from tuningfork.backends.formats import get_format

        return from_entries([
            entry(100, "<|channel>", 1.0), entry(1, "thought", 1.0), entry(2, "\n", 1.0),
            entry(3, "Write", 0.7, [(4, "Save", 0.3)]),
            entry(101, "<channel|>", 1.0), entry(48, "<|tool_call>", 1.0), entry(5, "call", 1.0), entry(6, ":", 1.0),
            name_entry,
            entry(8, "{", 1.0), entry(9, "path", 1.0), entry(11, ":x}", 1.0), entry(49, "<tool_call|>", 1.0), entry(50, "", 1.0),
        ]).mark_parts(get_format("gemma4"))

    def test_each_tools_chance_at_the_names_first_token(self):
        from tuningfork.analysis.tokens import tool_choice

        stream = self.stream(entry(7, "write", 0.4, [(12, "execute", 0.55), (13, "read", 0.03), (14, "{", 0.02)]))
        found = tool_choice(stream, Sampler(), tools=self.TOOLS, prefix="call:")
        assert found is not None and found.position == 8 and found.chosen == "write_file"
        assert found.chance["write_file"] == pytest.approx(0.4) and found.chance["execute_code"] == pytest.approx(0.55)
        assert found.chance["read_file"] == pytest.approx(0.03) and found.chance["other"] == pytest.approx(0.02)
        shown = found.to_json()
        assert list(shown["chance"]) == ["execute_code", "write_file", "read_file", "other"]
        # The step's sampler shapes it: greedy, the likeliest token is the only one drawn.
        greedy = tool_choice(stream, Sampler(temperature=0.0), tools=self.TOOLS, prefix="call:")
        assert greedy.chance == {"execute_code": pytest.approx(1.0)}

    def test_a_name_no_tool_begins_and_a_whole_name_in_one_token(self):
        from tuningfork.analysis.tokens import tool_choice

        whole = tool_choice(self.stream(entry(7, "write_file", 0.9, [(12, "execute_code", 0.1)])),
                            Sampler(), tools=self.TOOLS, prefix="call:")
        assert whole.chosen == "write_file" and whole.chance == {"write_file": pytest.approx(0.9), "execute_code": pytest.approx(0.1)}
        made_up = tool_choice(self.stream(entry(7, "summon", 0.9, [(12, "write", 0.1)])),
                              Sampler(), tools=self.TOOLS, prefix="call:")
        assert made_up.chosen == "summon" and made_up.chance == {"summon": pytest.approx(0.9), "write_file": pytest.approx(0.1)}
        unknown_tools = tool_choice(self.stream(entry(7, "write", 0.6, [(12, "execute", 0.4)])), Sampler(), prefix="call:")
        assert unknown_tools.chosen == "write" and set(unknown_tools.chance) == {"write", "execute"}

    def test_no_call_or_no_parts_is_no_choice(self):
        from tuningfork.analysis.tokens import tool_choice

        reply = from_entries([entry(1, "Done", 1.0), entry(50, "", 1.0)])
        assert tool_choice(reply, Sampler(), tools=self.TOOLS, prefix="call:") is None
        unmarked = from_entries([entry(48, "<|tool_call>", 1.0), entry(5, "call", 1.0), entry(6, ":", 1.0), entry(7, "write", 1.0)])
        assert tool_choice(unmarked, Sampler(), tools=self.TOOLS, prefix="call:") is None

    def test_the_name_is_found_after_the_prefix_by_its_text(self):
        # A Qwen3.5 stream scored before the newline after <tool_call> was the marker's keeps
        # its parts: its call's part begins with that newline. And a token carries the
        # prefix's end with the name's start.
        from tuningfork.analysis.tokens import tool_choice

        stream = from_entries([
            entry(1, "<tool_call>", 1.0), entry(2, "\n", 1.0), entry(3, "<", 1.0), entry(4, "function", 1.0),
            entry(5, "=write", 0.7, [(6, "=read", 0.3)]), entry(7, "_file", 1.0), entry(8, ">\n", 1.0),
            entry(9, "</function>", 1.0), entry(50, "", 1.0),
        ])
        for start in (1, 2):  # the newline the part's, or the marker's
            stream.parts = [("marker", 0, start), ("tool_call", start, 8), ("marker", 8, 9)]
            found = tool_choice(stream, Sampler(), tools=self.TOOLS, prefix="<function=")
            assert found is not None and (found.position, found.chosen) == (4, "write_file")
            assert found.chance == {"write_file": pytest.approx(0.7), "read_file": pytest.approx(0.3)}
        # a part without the prefix has no name to find
        stream.parts = [("marker", 0, 4), ("tool_call", 4, 8), ("marker", 8, 9)]
        assert tool_choice(stream, Sampler(), tools=self.TOOLS, prefix="<function=") is None

    def test_the_prefix_is_found_where_the_part_begins_only(self):
        # A bare call's "<" can sit in the content's last token (".<" is one of Qwen's tokens):
        # the part then begins past it, and an opening further on is no name of this call.
        from tuningfork.analysis.tokens import tool_choice

        stream = from_entries([
            entry(1, "Checking", 1.0), entry(2, ".<", 1.0), entry(3, "function", 1.0), entry(4, "=get", 0.9, [(5, "=read", 0.1)]),
            entry(6, "_weather", 1.0), entry(7, ">\n<", 1.0), entry(8, "parameter=city>\n<", 1.0), entry(9, "function", 1.0),
            entry(5, "=read", 0.6, [(4, "=get", 0.4)]), entry(10, "_file", 1.0), entry(11, "</tool_call>", 1.0), entry(50, "", 1.0),
        ])
        stream.parts = [("content", 0, 2), ("tool_call", 2, 10), ("marker", 10, 12)]
        assert tool_choice(stream, Sampler(), tools=self.TOOLS, prefix="<function=") is None
