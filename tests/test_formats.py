"""Model output formats, checked against the server's own parse of real output.

The fixtures (`tests/fixtures/formats/<format>/`) are raw generated text as llama.cpp's
logprobs gave it, token by token, beside the message the same server parsed from it; see
`tests/fixtures/formats/capture.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuningfork.analysis.normalize import outcome_for
from tuningfork.backends.formats import FormatError, format_for, get_format, known_models
from tuningfork.backends.formats.gemma4 import Gemma4
from tuningfork.backends.formats.qwen3_coder import Qwen3Coder

HERE = Path(__file__).parent / "fixtures" / "formats"
#: Each format's fixtures: Gemma 4's from one model; the Qwen formats' from each model that
#: writes them, a folder each.
FIXTURES = {
    "gemma4": sorted((HERE / "gemma4").glob("*.json")),
    "qwen3-coder": sorted((HERE / "qwen3-coder").glob("*/*.json")),
    "qwen3": sorted((HERE / "qwen3").glob("*/*.json")),
    "qwen2.5": sorted((HERE / "qwen2.5").glob("*/*.json")),
}
#: The pieces each format marks as markers wherever they stand.
MARKERS = {
    "gemma4": ("<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", ""),
    "qwen3-coder": ("<think>", "</think>", "<tool_call>", "</tool_call>", ""),
    "qwen3": ("<think>", "</think>", "<tool_call>", "</tool_call>", ""),
    "qwen2.5": ("<tool_call>", "</tool_call>", ""),
}
CASES = [(name, path) for name, paths in FIXTURES.items() for path in paths]
GEMMA = Gemma4()
QWEN = Qwen3Coder()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_seen(message: dict) -> dict:
    """A message as the fixtures keep the server's: ids and empty fields aside."""
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": [
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for call in message.get("tool_calls") or []
        ],
    }


def response(message: dict, finish: str) -> dict:
    return {"choices": [{"finish_reason": finish, "message": message}]}


def server_message(fixture: dict) -> dict:
    seen = fixture["message"]
    message = {"role": "assistant", "content": seen["content"]}
    if seen["tool_calls"]:
        message["tool_calls"] = [
            {"id": f"x{i}", "type": "function", "function": call}
            for i, call in enumerate(seen["tool_calls"])
        ]
    return message


def read(fmt, fixture: dict, **known) -> dict:
    """A fixture's raw text read by its format, with what a caller knows of its request."""
    return fmt.parse("".join(fixture["pieces"]), tools=fixture["tools"], parallel=fixture["parallel"],
                     offered=fixture.get("offered") or (), **known)


@pytest.mark.parametrize("name", FIXTURES)
def test_there_are_fixtures_for_every_case(name):
    assert len(FIXTURES[name]) >= 20
    assert {load(path)["case"] for path in FIXTURES[name]} >= {"text", "parallel", "braces", "unicode", "text-nothink"}


@pytest.mark.parametrize(("name", "path"), CASES,
                         ids=lambda case: f"{case.parent.name}/{case.stem}" if isinstance(case, Path) else case)
class TestFixtures:
    """Each format against the server's parse of real output: read as the server read it, the
    same behaviour, written back as the model wrote it, each token's part."""

    def test_parses_as_the_server_did(self, name, path):
        fixture = load(path)
        assert as_seen(read(get_format(name), fixture, thinking=fixture["thinking"])) == fixture["message"]

    def test_parses_alike_without_the_prompt(self, name, path):
        # A resample's kept tokens are read without the prompt that opened the reasoning.
        fixture = load(path)
        assert as_seen(read(get_format(name), fixture)) == fixture["message"]

    def test_same_behaviour_as_the_server_saw(self, name, path):
        fixture = load(path)
        parsed = read(get_format(name), fixture, thinking=fixture["thinking"])
        finish = fixture["finish_reason"]
        assert outcome_for(response(parsed, finish)).key == outcome_for(response(server_message(fixture), finish)).key

    def test_renders_back_to_the_raw_text(self, name, path):
        fixture = load(path)
        fmt = get_format(name)
        known = {"thinking": fixture["thinking"], "offered": fixture.get("offered") or ()}
        message = read(fmt, fixture, thinking=fixture["thinking"])
        renderings = fmt.renderings(message, **known)
        assert renderings[0] == fmt.render(message, **known)
        # the model's end token has an empty piece: the rest is the text rendered back — or, where
        # the server reads several texts alike, one of them, which the token count picks
        assert fixture["pieces"][-1] == ""
        assert "".join(fixture["pieces"][:-1]) in renderings

    @pytest.mark.parametrize("told", [True, False], ids=["told", "inferred"])
    def test_parts_follow_the_markers(self, name, path, told):
        fixture = load(path)
        fmt = get_format(name)
        pieces = fixture["pieces"]
        parts = fmt.parts(pieces, thinking=fixture["thinking"] if told else None)
        assert parts[0][1] == 0 and parts[-1][2] == len(pieces)
        assert all(a[2] == b[1] for a, b in zip(parts, parts[1:]))  # contiguous
        kind = {position: part for part, start, end in parts for position in range(start, end)}
        for position, piece in enumerate(pieces):
            # A marker the model writes while it reasons is reasoning where the server reads a
            # reasoning up to its close alone (Qwen3's): the joined reasoning below says which.
            if piece in MARKERS[name] and kind[position] != "reasoning":
                assert kind[position] == "marker", (position, piece)
        assert kind[len(pieces) - 1] == "marker"  # the end token
        reasoning = "".join(p for i, p in enumerate(pieces) if kind[i] == "reasoning")
        assert reasoning == fixture["message"]["reasoning_content"]
        if fixture["message"]["tool_calls"]:
            call = "".join(p for i, p in enumerate(pieces) if kind[i] == "tool_call")
            assert call.startswith(fmt.call_name_prefix)


class TestGemma4:
    def test_detected_by_the_servers_own_rule(self):
        assert format_for("{% if tools %}<|tool_call>call:{{ name }}{% endif %}").name == "gemma4"
        assert format_for("<|im_start|>assistant") is None
        assert format_for(None) is None
        assert get_format("gemma4").name == "gemma4"

    def test_thinking_is_on_when_the_prompt_says_so(self):
        assert GEMMA.thinking("<|turn>system\n<|think|>\nSYS<turn|>")
        assert not GEMMA.thinking("<|turn>user\nhi<turn|>")

    def test_a_number_is_written_back_as_it_was(self):
        message = {"tool_calls": [{"function": {"name": "record_expense", "arguments": '{"amount":61.50,"note":null,"ok":true}'}}]}
        text = GEMMA.render(message, thinking=False)
        assert text == '<|tool_call>call:record_expense{amount:61.50,note:null,ok:true}<tool_call|>'
        assert GEMMA.parse(text, tools=True)["tool_calls"][0]["function"]["arguments"] == '{"amount":61.50,"note":null,"ok":true}'

    def test_an_empty_thought_is_opened_when_thinking_is_on(self):
        assert GEMMA.render({"content": "hi"}, thinking=True) == "<|channel>thought\n<channel|>hi"
        assert GEMMA.render({"content": "hi"}, thinking=False) == "hi"

    def test_one_call_only_when_parallel_calls_are_off(self):
        text = '<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>'
        assert len(GEMMA.parse(text, tools=True)["tool_calls"]) == 2
        assert len(GEMMA.parse(text, tools=True, parallel=False)["tool_calls"]) == 1

    def test_without_tools_a_call_is_content(self):
        text = "<|channel>thought\nhm<channel|>see <|tool_call>call:a{x:1}<tool_call|>"
        parsed = GEMMA.parse(text, tools=False)
        assert parsed["reasoning_content"] == "hm"
        assert parsed["content"] == "see <|tool_call>call:a{x:1}<tool_call|>"

    @pytest.mark.parametrize("text", [
        '<|tool_call>call:a{x:<|"|>open}<tool_call|>',
        "<|tool_call>call:a{x:1<tool_call|>",
        "<|tool_call>call:a{x:}<tool_call|>",
        "<|tool_call>call:a{x:1}",
    ])
    def test_a_malformed_call_is_an_error(self, text):
        with pytest.raises(FormatError):
            GEMMA.parse(text, tools=True)

    def test_cut_off_while_thinking(self):
        parsed = GEMMA.parse("<|channel>thought\nstill going", tools=True)
        assert parsed["reasoning_content"] == "still going"
        assert parsed["content"] == ""


def tool(name: str, properties: dict, **schema) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": properties, **schema}}}


def call(name: str, **values: str) -> str:
    """A call as Qwen3.5 writes one: each value as raw text."""
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def arguments(text: str, offered: list, **known) -> str:
    return QWEN.parse(text, tools=True, offered=offered, **known)["tool_calls"][0]["function"]["arguments"]


class TestQwen3Coder:
    def test_detected_by_the_servers_own_rule(self):
        coder = "{{ '<tool_call>\n<function=' + name + '>\n' }}{{ '<parameter=' + key + '>\n' }}"
        template = "{{ '<think>\n' }}" + coder
        assert format_for(template).name == "qwen3-coder"
        # the variant with no newline between its tags is another parser's
        assert format_for(template + "{{ '<tool_call><function=' ~ tool_call.name ~ '>' }}") is None
        assert format_for("{{ '<tool_call>' }}{\"name\": ...}") is None
        # Qwen3-Coder's own template, which never reasons, the server reads otherwise: not this format
        assert format_for(coder) is None
        assert get_format("qwen3-coder").name == "qwen3-coder"

    def test_the_known_models_are_named_once(self):
        assert known_models() == "Gemma 4, Qwen2.5, Qwen3, Qwen3.5 and Qwen3.6"

    @pytest.mark.parametrize(("prompt", "on"), [
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", True),
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", False),
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", False),
        # an earlier turn's reasoning, closed, is history: only the last turn's opening counts
        ("<|im_start|>assistant\n<think>\nx\n</think>\n\nok<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", True),
    ])
    def test_thinking_is_on_when_the_prompt_opens_it(self, prompt, on):
        assert QWEN.thinking(prompt) is on

    def test_an_answer_starts_inside_the_reasoning_the_prompt_opened(self):
        text = "Plan it.\n</think>\n\nDone."
        for thinking in (True, None):
            parsed = QWEN.parse(text, tools=False, thinking=thinking)
            assert parsed["reasoning_content"] == "Plan it.\n" and parsed["content"] == "Done."
        assert QWEN.render(QWEN.parse(text, tools=False, thinking=True), thinking=True) == text

    def test_cut_off_while_thinking(self):
        parsed = QWEN.parse("still going", tools=True, thinking=True)
        assert parsed["reasoning_content"] == "still going" and parsed["content"] == ""
        # read without its prompt, an answer that never closed its reasoning reads as content
        assert QWEN.parse("still going", tools=True)["content"] == "still going"

    @pytest.mark.parametrize(("schema", "raw", "written"), [
        ({"type": "string"}, "42", '"42"'),
        ({"type": "string"}, '{"a": 1}', '"{\\"a\\": 1}"'),
        ({"type": "number"}, "61.50", "61.50"),
        ({"type": "integer"}, "7", "7"),
        ({"type": ["string", "null"]}, "null", "null"),
        ({"type": ["string", "null"]}, "none", '"none"'),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "7", "7"),
        # any number, where an integer is admitted beside a string: the server's alternative is a number
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "7.5", "7.5"),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "seven", '"seven"'),
        ({"type": ["number", "string"]}, "NaN", '"NaN"'),  # no JSON number
        ({"type": ["object", "string"]}, '{"a": "x\ty"}', '{"a": "x\ty"}'),  # a string holds any character
        # where no string is admitted the parser takes any JSON; the grammar keeps to the schema
        ({"type": "integer"}, '"7"', '"7"'),
        ({"type": "integer"}, "7.5", "7.5"),
        ({"format": "uri"}, "42", "42"),  # a format llama.cpp does not read says nothing
        ({"format": "date"}, "2026", '"2026"'),
        # a JSON string is never one of the alternatives: the text is the string
        ({"type": ["string", "integer"]}, '"7"', '"\\"7\\""'),
        ({"type": "object"}, '{"city": "Berlin"}', '{"city": "Berlin"}'),
        ({"type": "array", "items": {"type": "string"}}, '["Ann", "Bo"]', '["Ann", "Bo"]'),
        ({"enum": ["a", "b"]}, "a", '"a"'),
        ({"enum": [1, 2]}, "2", "2"),
        ({"$ref": "#/$defs/n"}, "3", "3"),
        ({}, "true", "true"),
        ({}, "hello", '"hello"'),
    ])
    def test_a_value_is_typed_by_the_schema_as_the_server_types_it(self, schema, raw, written):
        offered = [tool("f", {"x": schema}, **{"$defs": {"n": {"type": "integer"}}})]
        text = call("f", x=raw)
        assert arguments(text, offered) == "{\"x\":" + written + "}"
        # and written back as the model wrote it
        assert QWEN.render(QWEN.parse(text, tools=True, offered=offered), thinking=False, offered=offered) == text

    def test_a_parameter_or_tool_not_offered_is_read_with_any_type(self):
        offered = [tool("f", {"x": {"type": "string"}})]
        assert arguments(call("f", y="true", x="1"), offered) == '{"y":true,"x":"1"}'
        assert arguments(call("g", n="3"), offered) == '{"n":3}'

    def test_a_value_that_is_no_json_where_the_schema_asks_for_json_is_an_error(self):
        for raw in ("three", "NaN"):
            with pytest.raises(FormatError):
                QWEN.parse(call("f", n=raw), tools=True, offered=[tool("f", {"n": {"type": "integer"}})])

    def test_content_before_a_call_keeps_its_blank_line(self):
        text = "Checking.\n\n" + call("get_weather", city="Oslo")
        parsed = QWEN.parse(text, tools=True, offered=[tool("get_weather", {"city": {"type": "string"}})])
        assert parsed["content"] == "Checking.\n\n"
        assert QWEN.render(parsed, thinking=False) == text

    def test_one_call_only_when_parallel_calls_are_off(self):
        text = call("a", x="1") + "\n" + call("b", y="2")
        assert len(QWEN.parse(text, tools=True)["tool_calls"]) == 2
        assert len(QWEN.parse(text, tools=True, parallel=False)["tool_calls"]) == 1

    def test_without_tools_a_call_is_content(self):
        text = "see " + call("a", x="1")
        assert QWEN.parse(text, tools=False)["content"] == text

    @pytest.mark.parametrize("text", [
        "<tool_call><function=a>\n</function>\n</tool_call>",  # the line after the opening is missing
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</function>\n</tool_call>",  # never closed
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</tool_call>",  # no </function>
        "<tool_call>\n<function=a",
    ])
    def test_a_malformed_call_is_an_error(self, text):
        with pytest.raises(FormatError):
            QWEN.parse(text, tools=True)

    def test_parts_mark_the_space_the_server_skips(self):
        pieces = ["\n", "Plan", ".\n", "</think>", "\n\n", "Hi", "<tool_call>", "\n", "<", "function", "=", "a", ">\n",
                  "</function>", "\n", "</tool_call>", "\n", "<tool_call>", "\n", "</tool_call>", ""]
        parts = QWEN.parts(pieces, thinking=True)
        kind = {position: part for part, start, end in parts for position in range(start, end)}
        # the space an answer begins with is skipped as the space after a marker is
        assert [kind[i] for i in range(6)] == ["marker", "reasoning", "reasoning", "marker", "marker", "content"]
        assert kind[7] == "tool_call" and kind[16] == "marker" and kind[20] == "marker"
        assert QWEN.parse("".join(pieces[:5]), tools=False, thinking=True)["reasoning_content"] == "Plan.\n"


QWEN3, QWEN25 = get_format("qwen3"), get_format("qwen2.5")
#: The lines of a Qwen3 template the server finds the markers in, as its source writes them.
QWEN3_LINES = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n')"
    " + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
    "{{- '<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}"
    "{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}"
    "{%- endfor %}"
)
#: Qwen2.5's renders no reasoning.
QWEN25_LINES = QWEN3_LINES.replace(
    " + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n')", " + content"
)
WEATHER = tool("get_weather", {"city": {"type": "string"}})


def json_call(name: str, arguments: str) -> str:
    """A call as Qwen3 and Qwen2.5 write one."""
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + arguments + "}\n</tool_call>"


def first_arguments(text: str) -> str:
    return QWEN3.parse(text, tools=True)["tool_calls"][0]["function"]["arguments"]


class TestQwenJson:
    def test_detected_by_the_template_lines_the_server_finds_the_markers_in(self):
        assert format_for(QWEN3_LINES).name == "qwen3"
        assert format_for(QWEN25_LINES).name == "qwen2.5"
        # QwQ cuts an answer at its close without rendering a reasoning: the server reads it otherwise
        assert format_for(QWEN25_LINES + "{%- set content = message.content.split('</think>')[-1] %}") is None
        assert format_for(QWEN3_LINES.replace("<|im_start|>", "<|turn>")) is None
        # Qwen3.5's calls are XML, another format's
        assert format_for(QWEN3_LINES + "{{ '<function=' }}") is None

    @pytest.mark.parametrize(("prompt", "on"), [
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", False),  # a hybrid Qwen3 opens it itself
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", True),  # a Qwen3-2507 Thinking template
        ("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", False),  # reasoning off
    ])
    def test_thinking_is_on_when_the_prompt_opens_it(self, prompt, on):
        assert QWEN3.thinking(prompt) is on
        assert QWEN25.thinking(prompt) is False

    def test_a_reasoning_is_read_as_the_server_reads_it(self):
        # one newline after the opening, the one before the close kept, two after it skipped
        parsed = QWEN3.parse("<think>\nPlan.\n</think>\n\n\nHi.", tools=False)
        assert (parsed["reasoning_content"], parsed["content"]) == ("Plan.\n", "\nHi.")
        # one the answer does not begin with is content
        assert QWEN3.parse("Hi. <think>\nx\n</think>", tools=False) == {"role": "assistant", "content": "Hi. <think>\nx\n</think>"}
        # a blank one is dropped
        assert QWEN3.parse("<think>\n\n</think>\n\nHi.", tools=False) == {"role": "assistant", "content": "Hi."}

    def test_an_answer_starts_inside_the_reasoning_the_prompt_opened(self):
        text = "\nPlan it.\n</think>\n\nDone."
        for thinking in (True, None):
            parsed = QWEN3.parse(text, tools=False, thinking=thinking)
            # the prompt's opening had its newline: the answer's own is the reasoning's
            assert (parsed["reasoning_content"], parsed["content"]) == ("\nPlan it.\n", "Done.")
        assert QWEN3.render(QWEN3.parse(text, tools=False, thinking=True), thinking=True) == text

    def test_qwen25_reads_no_reasoning(self):
        text = "<think>\nPlan.\n</think>\n\nHi."
        assert QWEN25.parse(text, tools=False) == {"role": "assistant", "content": text}

    @pytest.mark.parametrize(("fmt", "tools", "content"), [
        (QWEN3, False, "Hi."), (QWEN3, True, "Hi."), (QWEN25, True, "Hi."),
        (QWEN25, False, "\n\nHi."),  # Qwen2.5's content without tools: the whole answer
    ], ids=["qwen3", "qwen3-tools", "qwen2.5-tools", "qwen2.5"])
    def test_the_space_an_answer_begins_with_is_skipped(self, fmt, tools, content):
        assert fmt.parse("\n\nHi.", tools=tools)["content"] == content

    def test_cut_off_while_thinking(self):
        for text, thinking in (("<think>\nstill going", None), ("still going", True)):
            parsed = QWEN3.parse(text, tools=True, thinking=thinking)
            assert (parsed["reasoning_content"], parsed["content"]) == ("still going", "")
        # a close cut off is left out, as the server waits for the rest of it
        assert QWEN3.parse("<think>\nstill</thi", tools=True)["reasoning_content"] == "still"

    def test_content_runs_to_a_call_and_keeps_its_newline(self):
        text = "<think>\nx\n</think>\n\nChecking.\n" + json_call("get_weather", '{"city": "Oslo"}')
        parsed = QWEN3.parse(text, tools=True, offered=[WEATHER])
        assert parsed["content"] == "Checking.\n"
        assert parsed["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Oslo"}'}
        assert QWEN3.render(parsed, thinking=False, offered=[WEATHER]) == text
        # a call's marker cut off at the end is left out of the content
        assert QWEN3.parse("Checking.\n<tool_call>", tools=True)["content"] == "Checking.\n"

    @pytest.mark.parametrize(("arguments", "kept"), [
        # as written: the spacing, and a string holding any character
        ('{ "path" :\n"a\tb.txt" }', '{ "path" :\n"a\tb.txt" }'),
        ('{"n": -1.5e3, "ok": true, "none": null, "list": ["\\u00e9", {}]}',
         '{"n": -1.5e3, "ok": true, "none": null, "list": ["\\u00e9", {}]}'),
        ('{"a":\v1}', '{"a":\v1}'),  # any of std::isspace's spaces between tokens
        ('"a.txt"', "{}"),  # no object: the server keeps none of it
        ("[1, 2]", "{}"),
    ])
    def test_the_arguments_are_kept_as_written(self, arguments, kept):
        assert first_arguments(json_call("read_file", arguments)) == kept

    @pytest.mark.parametrize("arguments", [
        '{"n": 01}', '{"n": NaN}', '{"s": "\\q"}', '{"n": 1,}', '{"n" 1}', "{'n': 1}", '{"s": "\\u12"}',
    ])
    def test_arguments_that_are_no_json_are_an_error(self, arguments):
        with pytest.raises(FormatError):
            QWEN3.parse(json_call("read_file", arguments), tools=True)

    def test_calls_in_parallel(self):
        two = json_call("a", '{"x": 1}') + "\n\n" + json_call("b", '{"y": 2}')
        # calls in markers of their own are read whether calls may run in parallel or not
        for parallel in (True, False):
            assert [c["function"]["name"] for c in QWEN3.parse(two, tools=True, parallel=parallel)["tool_calls"]] == ["a", "b"]
        # in one marker, separated by commas, only where they may
        one = '<tool_call>\n{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}\n</tool_call>'
        assert len(QWEN3.parse(one, tools=True)["tool_calls"]) == 2
        with pytest.raises(FormatError):
            QWEN3.parse(one, tools=True, parallel=False)

    def test_a_tool_not_offered_is_read_with_its_name(self):
        parsed = QWEN3.parse(json_call("nope", "{}"), tools=True, offered=[WEATHER])
        assert parsed["tool_calls"][0]["function"] == {"name": "nope", "arguments": "{}"}

    def test_without_tools_a_call_is_content(self):
        text = "see " + json_call("a", "{}")
        assert QWEN3.parse(text, tools=False)["content"] == text
        assert QWEN25.parse(text, tools=False)["content"] == text

    @pytest.mark.parametrize("text", [
        json_call("a", "{}") + "\nDone.",  # nothing may follow the calls
        '<tool_call>\n{"name": "a", "arguments": {"x": "cut',  # cut off inside a call
        '<tool_call>\n{"arguments": {}, "name": "a"}\n</tool_call>',  # the name comes first
        '<tool_call>\n{"name": "a", "arguments": {}}\n',  # never closed
    ])
    def test_a_malformed_call_is_an_error(self, text):
        with pytest.raises(FormatError):
            QWEN3.parse(text, tools=True)

    def test_an_empty_reasoning_is_one_of_the_texts_read_alike(self):
        parsed = QWEN3.parse("<think>\n\n</think>\n\nHallo.", tools=False)
        assert QWEN3.renderings(parsed, thinking=False) == ["Hallo.", "<think>\n\n</think>\n\nHallo."]
        # a reasoning the prompt opened is closed whatever it holds; Qwen2.5 writes none
        assert QWEN3.renderings({"content": "Hi."}, thinking=True) == ["</think>\n\nHi."]
        assert QWEN25.renderings({"content": "Hi."}, thinking=False) == ["Hi."]

    def test_calls_a_line_or_a_blank_line_apart_are_read_alike(self):
        one, two = json_call("a", "{}"), json_call("b", "{}")
        head = "<think>\nx\n</think>\n\n"
        parsed = QWEN3.parse(head + one + "\n\n" + two, tools=True)
        assert QWEN3.renderings(parsed, thinking=False) == [head + one + "\n" + two, head + one + "\n\n" + two]

    def test_parts_follow_the_servers_reading(self):
        pieces = ["\n", "<think>", "\n", "Plan", ".\n", "</think>", "\n\n", "Hi", ".\n", "<tool_call>", "\n", '{"', "name",
                  '":', ' "', "a", '",', ' "', "arguments", '":', " {}}", "\n", "</tool_call>", "\n", ""]
        kind = [part for part, start, end in QWEN3.parts(pieces) for _ in range(start, end)]
        expected = ["marker"] * 3 + ["reasoning"] * 2 + ["marker"] * 2 + ["content"] * 2 + ["marker"]
        assert kind == expected + ["tool_call"] * 12 + ["marker"] * 3
        call = "".join(piece for piece, part in zip(pieces, kind) if part == "tool_call")
        assert call.startswith(QWEN3.call_name_prefix)

    def test_parts_of_a_malformed_call_are_still_marked(self):
        parts = QWEN3.parts(["<tool_call>", "\n", "{", "oops", ""])
        assert [part for part, _, _ in parts] == ["marker", "tool_call", "marker"]
