"""Wire formats: reading and writing each dialect of model call.

The Anthropic fixtures are shaped on a recorded Claude Code 2.1.281 session against
llama.cpp's native `/v1/messages` — system blocks, a `Skill` tool call, tool results inside
user turns, thinking blocks, and mid-conversation `role: system` messages.
"""

from __future__ import annotations

import copy
import json

import pytest

from tuningfork.analysis.normalize import outcome_for
from tuningfork.proxy.sse import CUT_OFF, SSEParser
from tuningfork.wire import get_wire, wire_for
from tuningfork.wire.anthropic import AnthropicMessages
from tuningfork.wire.base import cap_output, uncapped
from tuningfork.wire.openai import OpenAIChat

ANTHROPIC = AnthropicMessages()
OPENAI = OpenAIChat()

SKILL_BODY = "# Tally report\n\nWrite the totals to report.json.\n"


def claude_code_request() -> dict:
    return {
        "model": "gemma-4-31b",
        "max_tokens": 32000,
        "stream": True,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.281;"},
            {"type": "text", "text": "You are a Claude agent.", "cache_control": {"type": "ephemeral"}},
        ],
        "tools": [{"name": "Skill", "input_schema": {}}, {"name": "Write", "input_schema": {}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>be brief</system-reminder>"},
                {"type": "text", "text": "Use the tally-report skill on data.csv."},
            ]},
            {"role": "system", "content": "# Environment\nPlatform: linux"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Load the skill.", "signature": "s1"},
                {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "tally-report"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "Launching skill: tally-report"},
                {"type": "text", "text": "Base directory for this skill: /h/skills/tally-report\n\n" + SKILL_BODY},
            ]},
        ],
    }


def anthropic_message() -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "gemma-4-31b",
        "content": [
            {"type": "thinking", "thinking": "Four rows.", "signature": "sig"},
            {"type": "text", "text": "Writing it."},
            {"type": "tool_use", "id": "t2", "name": "Write",
             "input": {"file_path": "report.json", "content": '{"count": 4}'}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "cache_read_input_tokens": 99, "output_tokens": 20},
    }


class TestRouting:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/v1/chat/completions", "openai-chat"),
            ("/v1/messages", "anthropic-messages"),
            ("/v1/messages/count_tokens", None),
            ("/v1/models", None),
            # raw completions are model load too: the gate and the pause apply
            ("/completion", "completion"),
            ("/completions", "completion"),
            ("/v1/completions", "completion"),
            ("/infill", "completion"),
            ("/_launch/abc/completion", "completion"),
            ("/v1/chat/completions/input_tokens", None),
            ("/tokenize", None),
            ("/apply-template", None),
        ],
    )
    def test_a_path_selects_its_format(self, path, expected):
        wire = wire_for("POST", path)
        assert (wire.name if wire else None) == expected

    def test_only_posts_are_model_calls(self):
        assert wire_for("GET", "/v1/messages") is None

    def test_steps_recorded_before_formats_are_openai(self):
        assert get_wire(None).name == "openai-chat"

    def test_a_raw_completion_reads_as_a_chat_completion(self):
        wire = get_wire("completion")
        canonical = wire.canonical_response({"content": "<|channel>thought\nhm", "stop_type": "limit",
                                             "tokens_evaluated": 12, "tokens_predicted": 4})
        assert canonical["choices"][0]["message"]["content"] == "<|channel>thought\nhm"
        assert canonical["choices"][0]["finish_reason"] == "length"
        assert wire.failure(500, None) is not None
        assert wire.error(503, "paused", "tuningfork_paused")["error"]["code"] == 503
        assert wire.view({"prompt": [1, 2]}) == []
        with pytest.raises(NotImplementedError):
            wire.with_user_message({"prompt": [1]}, "why?")


class TestAnthropicResponses:
    def test_a_stream_reassembles_into_the_message(self):
        message = anthropic_message()
        body = ANTHROPIC.render(message, request={"stream": True})
        assert ANTHROPIC.parse(body, event_stream=True) == message

    def test_partial_tool_json_is_joined_before_parsing(self):
        assembler = ANTHROPIC.assembler()
        raw = (
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"tool_use","id":"t","name":"Read","input":{}}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"input_json_delta","partial_json":"{\\"file_pa"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"input_json_delta","partial_json":"th\\": \\"a.csv\\"}"}}\n\n'
        )
        for event in SSEParser().feed(raw.encode()):
            assembler.add_event(event)
        assert assembler.result()["content"][0]["input"] == {"file_path": "a.csv"}

    def test_the_common_shape_carries_reasoning_calls_and_usage(self):
        canonical = ANTHROPIC.canonical_response(anthropic_message())
        choice = canonical["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["reasoning_content"] == "Four rows."
        assert choice["message"]["content"] == "Writing it."
        (call,) = choice["message"]["tool_calls"]
        assert call["function"]["name"] == "Write"
        assert json.loads(call["function"]["arguments"])["file_path"] == "report.json"
        # Cached prompt tokens are still prompt tokens.
        assert canonical["usage"]["prompt_tokens"] == 100
        assert canonical["usage"]["completion_tokens"] == 20

    def test_an_edit_goes_back_keeping_what_it_did_not_touch(self):
        original = anthropic_message()
        canonical = ANTHROPIC.canonical_response(original)
        function = canonical["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps({"file_path": "report.json", "content": '{"row_count": 4}'})

        edited = ANTHROPIC.native_response(canonical, original)

        assert edited["content"][0] == original["content"][0]  # thinking, signature intact
        assert edited["content"][2]["input"]["content"] == '{"row_count": 4}'
        assert edited["id"] == "msg_1" and edited["stop_reason"] == "tool_use"
        assert edited["usage"] == original["usage"]

    def test_an_unedited_round_trip_is_exact(self):
        original = anthropic_message()
        assert ANTHROPIC.native_response(ANTHROPIC.canonical_response(original), original) == original

    def test_edited_reasoning_becomes_a_new_block(self):
        original = anthropic_message()
        canonical = ANTHROPIC.canonical_response(original)
        canonical["choices"][0]["message"]["reasoning_content"] = "Think again."
        edited = ANTHROPIC.native_response(canonical, original)
        assert edited["content"][0] == {"type": "thinking", "thinking": "Think again."}

    def test_a_non_streaming_request_gets_json(self):
        body = ANTHROPIC.render(anthropic_message(), request={"stream": False})
        assert json.loads(body)["id"] == "msg_1"

    def test_errors_take_the_anthropic_shape(self):
        error = ANTHROPIC.error(409, "dropped", "tuningfork_dropped")
        assert error == {"type": "error", "error": {"type": "tuningfork_dropped", "message": "dropped"}}
        assert ANTHROPIC.error_event(error).startswith(b"event: error\ndata: ")


def event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def message_start(**overrides) -> str:
    message = {"id": "msg_s", "type": "message", "role": "assistant", "model": "m", "content": [],
               "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1}}
    return event("message_start", {"type": "message_start", "message": {**message, **overrides}})


def text_block(text: str, index: int = 0) -> str:
    return (
        event("content_block_start", {"type": "content_block_start", "index": index,
                                       "content_block": {"type": "text", "text": ""}})
        + event("content_block_delta", {"type": "content_block_delta", "index": index,
                                        "delta": {"type": "text_delta", "text": text}})
    )


MESSAGE_DELTA = event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                        "stop_sequence": None}, "usage": {"output_tokens": 2}})
MESSAGE_STOP = event("message_stop", {"type": "message_stop"})
OVERLOADED = event("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


def assembled(body: str):
    assembler = ANTHROPIC.assembler()
    for sse_event in SSEParser().feed(body.encode()):
        assembler.add_event(sse_event)
    return assembler


class TestAnthropicFailures:
    """An error, or a stream that stopped early, is not the model's behaviour."""

    def test_an_error_body_has_no_choices_and_keeps_its_error(self):
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
        canonical = ANTHROPIC.canonical_response(body)

        assert canonical["choices"] == []
        assert canonical["error"]["message"] == "prompt is too long"
        # The same as an OpenAI error body: kept out of every rate, not "empty".
        assert outcome_for(canonical).key == outcome_for(OPENAI.canonical_response({"error": {}})).key
        assert outcome_for(canonical).is_error

    def test_a_gateways_openai_style_error_is_an_error_too(self):
        assert ANTHROPIC.canonical_response({"error": {"message": "bad gateway"}})["choices"] == []

    @pytest.mark.parametrize(
        "status, body, expected",
        [
            (400, {"type": "error", "error": {"message": "too long"}}, "upstream_error"),
            (200, {"type": "error", "error": {"message": "too long"}}, "upstream_error: too long"),
            (200, {"type": "message", "content": [], "stop_reason": "end_turn"}, None),
        ],
    )
    def test_a_body_is_a_failure_by_status_or_by_shape(self, status, body, expected):
        assert ANTHROPIC.failure(status, body) == expected

    def test_a_stream_that_ends_in_an_error_event_is_a_failure(self):
        stream = assembled(message_start() + text_block("I will now wr") + OVERLOADED)

        assert ANTHROPIC.failure(200, stream.result(), stream) == "upstream_error: Overloaded"
        # What did arrive stays readable, beside the error.
        canonical = ANTHROPIC.canonical_response(stream.result())
        assert canonical["choices"][0]["message"]["content"] == "I will now wr"
        assert canonical["error"]["type"] == "overloaded_error"

    def test_an_error_before_any_content_has_no_choices(self):
        stream = assembled(message_start() + OVERLOADED)
        assert ANTHROPIC.canonical_response(stream.result())["choices"] == []

    @pytest.mark.parametrize(
        "ending, expected",
        [
            (MESSAGE_DELTA + MESSAGE_STOP, None),
            # The stop reason arrived; only the last event went missing. The answer is whole.
            (MESSAGE_DELTA, None),
            # The server said it was done, if without a stop reason.
            (MESSAGE_STOP, None),
            ("", CUT_OFF),
        ],
    )
    def test_a_stream_is_whole_if_it_ended_the_way_messages_end(self, ending, expected):
        stream = assembled(message_start() + text_block("hello") + ending)
        assert ANTHROPIC.failure(200, stream.result(), stream) == expected


class TestAnthropicStreamShapes:
    """Nulls where the assembler expected values: each once raised, which ended the
    harness's own stream along with the recording's copy."""

    def test_usage_may_open_null(self):
        stream = assembled(message_start(usage=None) + text_block("hello") + MESSAGE_DELTA + MESSAGE_STOP)
        assert stream.result()["usage"] == {"output_tokens": 2}

    def test_a_block_may_open_with_null_text(self):
        body = (
            message_start()
            + event("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": None}})
            + event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "text_delta", "text": "hello"}})
        )
        assert assembled(body).result()["content"] == [{"type": "text", "text": "hello"}]

    def test_a_null_index_does_not_break_ordering(self):
        body = message_start() + text_block("first") + text_block("second").replace('"index": 0', '"index": null')
        texts = [block["text"] for block in assembled(body).result()["content"]]
        assert texts == ["first", "second"]

    def test_tool_input_sent_as_an_object(self):
        opening = {"type": "tool_use", "id": "t", "name": "Read", "input": {}}
        parsed = {"type": "input_json_delta", "partial_json": {"file_path": "a"}}
        body = (
            message_start()
            + event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": opening})
            + event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": parsed})
        )
        assert assembled(body).result()["content"][0]["input"] == {"file_path": "a"}

    def test_null_text_in_a_complete_message_reads_as_empty(self):
        message = {**anthropic_message(), "content": [{"type": "text", "text": None}]}
        assert ANTHROPIC.canonical_response(message)["choices"][0]["message"]["content"] == ""


class TestAnthropicEdits:
    """An edited response goes back to Claude Code; what it says has to hold together."""

    @staticmethod
    def edit(original: dict, change) -> dict:
        canonical = ANTHROPIC.canonical_response(original)
        change(canonical["choices"][0])
        return ANTHROPIC.native_response(canonical, original)

    def test_arguments_given_as_an_object_are_the_input(self):
        def change(choice):
            choice["message"]["tool_calls"][0]["function"]["arguments"] = {"file_path": "b.json"}

        edited = self.edit(anthropic_message(), change)
        assert edited["content"][2]["input"] == {"file_path": "b.json"}

    @pytest.mark.parametrize("arguments", [["a", "b"], "[1, 2]", "not json", 7])
    def test_arguments_that_are_not_an_object_are_kept_visibly(self, arguments):
        def change(choice):
            choice["message"]["tool_calls"][0]["function"]["arguments"] = arguments

        edited = self.edit(anthropic_message(), change)
        assert edited["content"][2]["input"] == {"_unparsed": arguments}

    def test_a_call_added_without_an_id_is_given_one(self):
        def change(choice):
            choice["message"]["tool_calls"].append({"function": {"name": "Read", "arguments": "{}"}})

        edited = self.edit(anthropic_message(), change)
        added = edited["content"][-1]
        assert added["name"] == "Read" and added["id"].startswith("toolu_")
        assert edited["content"][2]["id"] == "t2"  # the model's own ids are kept

    @staticmethod
    def stopped_at_a_sequence() -> dict:
        return {**anthropic_message(), "content": [{"type": "text", "text": "abc"}],
                "stop_reason": "stop_sequence", "stop_sequence": "###"}

    def test_a_stop_sequence_survives_an_edit_that_leaves_the_finish_reason_alone(self):
        def change(choice):
            choice["message"]["content"] = "abd"

        edited = self.edit(self.stopped_at_a_sequence(), change)
        assert (edited["stop_reason"], edited["stop_sequence"]) == ("stop_sequence", "###")

    @pytest.mark.parametrize(
        "finish, stop",
        [
            ("length", "max_tokens"),
            ("content_filter", "refusal"),
            ("refusal", "refusal"),  # already Anthropic's: as it is
            ("pause_turn", "pause_turn"),
        ],
    )
    def test_a_changed_finish_reason_is_what_is_delivered(self, finish, stop):
        def change(choice):
            choice["finish_reason"] = finish

        for original in (self.stopped_at_a_sequence(), anthropic_message()):
            edited = self.edit(original, change)
            assert edited["stop_reason"] == stop
            assert edited["stop_sequence"] is None  # no sequence was what stopped it

    def test_a_tool_call_added_to_an_answer_makes_it_tool_use(self):
        def change(choice):
            choice["message"]["tool_calls"] = [{"id": "t9", "function": {"name": "Read", "arguments": "{}"}}]

        edited = self.edit(self.stopped_at_a_sequence(), change)
        assert (edited["stop_reason"], edited["stop_sequence"]) == ("tool_use", None)

    def test_taking_out_every_tool_call_ends_the_turn(self):
        def change(choice):
            choice["message"]["tool_calls"] = []

        edited = self.edit(anthropic_message(), change)
        assert edited["stop_reason"] == "end_turn"
        assert [block["type"] for block in edited["content"]] == ["thinking", "text"]

    def test_an_edit_with_no_finish_reason_still_says_how_it_stopped(self):
        def change(choice):
            choice["finish_reason"] = None

        assert self.edit(anthropic_message(), change)["stop_reason"] == "tool_use"


class TestMalformedRequests:
    """A request tuningfork cannot make sense of is still relayed; reading it must not raise."""

    @pytest.mark.parametrize("wire", [ANTHROPIC, OPENAI], ids=lambda wire: wire.name)
    def test_an_entry_that_is_not_a_message_is_kept_as_it_is(self, wire):
        assert wire.normalize_message("hi") == "hi"
        assert wire.normalize_message(None) is None

    @pytest.mark.parametrize("wire", [ANTHROPIC, OPENAI], ids=lambda wire: wire.name)
    def test_the_view_leaves_it_out(self, wire):
        request = {"messages": ["hi", {"role": "user", "content": "go"}, 5]}
        assert wire.view(request) == [{"role": "user", "content": "go"}]

    @pytest.mark.parametrize("wire", [ANTHROPIC, OPENAI], ids=lambda wire: wire.name)
    def test_messages_that_are_not_a_list_view_as_none(self, wire):
        assert wire.view({"messages": 5}) == []

    def test_a_system_list_with_a_stray_entry(self):
        view = ANTHROPIC.view({"system": ["stray", {"type": "text", "text": "Be brief."}], "messages": []})
        assert view == [{"role": "system", "content": "Be brief."}]


class TestAnthropicView:
    def test_the_conversation_reads_as_openai_messages(self):
        view = ANTHROPIC.view(claude_code_request())
        roles = [message["role"] for message in view]
        assert roles == ["system", "user", "system", "assistant", "tool", "user"]

        system, user, _, assistant, tool, skill = view
        assert [part["text"] for part in system["content"]][1] == "You are a Claude agent."
        assert len(user["content"]) == 2  # two text blocks stay two parts
        assert assistant["reasoning_content"] == "Load the skill."
        assert assistant["tool_calls"][0]["function"]["name"] == "Skill"
        assert tool == {"role": "tool", "tool_call_id": "t1", "content": "Launching skill: tally-report"}
        assert skill["content"].endswith(SKILL_BODY)  # one text block reads as a string

    def test_the_view_is_for_reading_only(self):
        request = claude_code_request()
        before = json.dumps(request, sort_keys=True)
        ANTHROPIC.view(request)
        assert json.dumps(request, sort_keys=True) == before


class TestOpenAI:
    def test_the_common_shape_is_its_own(self):
        wire = OpenAIChat()
        response = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        assert wire.canonical_response(response) is response
        assert wire.view({"messages": [{"role": "user", "content": "x"}]}) == [
            {"role": "user", "content": "x"}
        ]

    @staticmethod
    def with_call(call: dict) -> dict:
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return {"id": "c", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}

    def test_edited_arguments_given_as_an_object_go_back_as_text(self):
        edit = self.with_call({"id": "call_1", "type": "function",
                               "function": {"name": "write_file", "arguments": {"path": "a.json"}}})
        before = copy.deepcopy(edit)

        native = OPENAI.native_response(edit, None)

        (call,) = native["choices"][0]["message"]["tool_calls"]
        # An object here would reach the harness as an object: OpenAI clients expect text.
        assert json.loads(call["function"]["arguments"]) == {"path": "a.json"}
        assert edit == before  # the decision as the person made it is left alone

    def test_a_call_added_without_an_id_is_given_one(self):
        edit = self.with_call({"function": {"name": "read_file", "arguments": "{}"}})
        (call,) = OPENAI.native_response(edit, None)["choices"][0]["message"]["tool_calls"]
        assert call["id"].startswith("call_") and call["type"] == "function"

    @pytest.mark.parametrize(
        "status, body, expected",
        [
            (429, {"error": {"message": "slow down"}}, "upstream_error"),
            (200, {"error": {"message": "context full"}}, "upstream_error: context full"),
            (200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}, None),
        ],
    )
    def test_a_body_is_a_failure_by_status_or_by_shape(self, status, body, expected):
        assert OPENAI.failure(status, body) == expected


class TestOutputCap:
    """A proxy's cap on output tokens (`serve --max-output`), as each format names the limit."""

    def test_a_limit_above_the_cap_is_lowered_and_what_was_asked_kept(self):
        request = {"model": "m", "max_tokens": 65536, "messages": [], "stream": True}

        capped, note = cap_output(OPENAI, request, 8192)

        assert capped == {**request, "max_tokens": 8192}
        assert list(capped) == list(request)  # every key where the harness put it
        assert note == {"cap": 8192, "asked": {"max_tokens": 65536}, "written": ["max_tokens"]}
        assert request["max_tokens"] == 65536  # the harness's own request is left alone
        assert uncapped(capped, note) == request

    def test_a_request_naming_no_limit_is_given_the_formats_own(self):
        request = {"model": "m", "messages": []}

        capped, note = cap_output(OPENAI, request, 100)

        assert capped["max_tokens"] == 100 and note["asked"] == {}
        assert uncapped(capped, note) == request  # the written limit taken out again

    @pytest.mark.parametrize("limit", [-1, None, "lots", 2.5, True])
    def test_no_limit_at_all_is_more_than_the_cap(self, limit):
        capped, note = cap_output(OPENAI, {"n_predict": limit, "messages": []}, 100)
        assert capped["n_predict"] == 100 and note["asked"] == {"n_predict": limit}

    def test_each_limit_named_is_held_and_one_within_it_left(self):
        request = {"max_tokens": 50, "n_predict": -1, "messages": []}

        capped, note = cap_output(OPENAI, request, 100)

        # llama.cpp reads n_predict before max_tokens: lowering max_tokens alone would not do.
        assert (capped["max_tokens"], capped["n_predict"]) == (50, 100)
        assert note["written"] == ["n_predict"]
        assert uncapped(capped, note) == request

    def test_a_request_within_the_cap_comes_back_as_it_was(self):
        request = {"max_tokens": 100, "messages": []}
        assert cap_output(OPENAI, request, 100) == (request, None)
        assert cap_output(AnthropicMessages(), {"max_tokens": 64, "messages": []}, 100)[1] is None

    def test_anthropic_messages_names_its_one_limit(self):
        capped, note = cap_output(AnthropicMessages(), {"max_tokens": 32000, "messages": []}, 8192)
        assert capped["max_tokens"] == 8192 and note["asked"] == {"max_tokens": 32000}
