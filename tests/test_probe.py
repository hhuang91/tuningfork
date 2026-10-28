"""Capability probing.

These tests pin the *interpretation* of a server's answers, which is where the subtlety
lives. The case that matters most is the quiet one: a server that accepts `logprobs`,
returns HTTP 200, and hands back null. Read only the status code and you would record that
as working.
"""

from __future__ import annotations

import httpx
import pytest

from tuningfork.backends.base import (
    BackendProfile,
    Capability,
    CapabilityResult,
    Support,
    UnsupportedCapability,
    anonymize_paths,
)
from tuningfork.backends.probe import (
    ProbeContext,
    probe_backend,
    probe_chat_template_endpoint,
    probe_logprobs,
    probe_output_format,
    probe_partial_assistant,
    probe_prompt_equivalence,
    probe_raw_completion,
    probe_reasoning,
    probe_seed_determinism,
    probe_server_side_n,
    probe_streaming_tool_calls,
    probe_tool_calls,
    probe_usage_in_stream,
    server_root,
)
from tuningfork.demo.mock import MockTurn
from tuningfork.demo.toy import props as toy_props

UPSTREAM = "http://upstream.test"


@pytest.fixture
async def probe_ctx(mock_server):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_server.app)
    ) as client:
        yield ProbeContext(client=client, root=UPSTREAM, model="mock-model")


def by_capability(results: list[CapabilityResult]) -> dict[str, CapabilityResult]:
    return {str(result.capability): result for result in results}


class TestLogprobs:
    async def test_accepted_but_null_is_not_support(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="hello"))

        results = by_capability(await probe_logprobs(probe_ctx))

        assert results[Capability.LOGPROBS].support is Support.NO
        assert "null" in results[Capability.LOGPROBS].detail
        assert results[Capability.TOP_LOGPROBS].support is Support.NO

    async def test_present_logprobs_with_alternatives(self, mock_server, probe_ctx):
        mock_server.script(
            MockTurn(
                content="hello",
                logprobs={
                    "content": [
                        {"token": "hel", "logprob": -0.1, "top_logprobs": [{"token": "hel"}]}
                    ]
                },
            )
        )

        results = by_capability(await probe_logprobs(probe_ctx))

        assert results[Capability.LOGPROBS].support is Support.YES
        assert results[Capability.TOP_LOGPROBS].support is Support.YES

    async def test_rejected_parameter_is_reported_verbatim(self, mock_server, probe_ctx):
        mock_server.script(
            MockTurn(
                status=400,
                error={"error": {"message": "Unsupported parameter: logprobs"}},
            )
        )

        results = by_capability(await probe_logprobs(probe_ctx))

        assert results[Capability.LOGPROBS].support is Support.NO
        assert "Unsupported parameter" in results[Capability.LOGPROBS].detail


class TestServerSideN:
    async def test_silently_ignored_n(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="blue", n_choices=1))

        (result,) = await probe_server_side_n(probe_ctx)

        assert result.support is Support.NO
        assert "concurrent requests" in result.detail

    async def test_honoured_n(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="blue", n_choices=3))

        (result,) = await probe_server_side_n(probe_ctx)

        assert result.support is Support.YES


class TestSeedDeterminism:
    async def test_identical_output_is_support(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="the sea"), MockTurn(content="the sea"))

        (result,) = await probe_seed_determinism(probe_ctx)

        assert result.support is Support.YES

    async def test_diverging_output_recommends_substitution(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="the sea"), MockTurn(content="the ocean"))

        (result,) = await probe_seed_determinism(probe_ctx)

        assert result.support is Support.NO
        assert "substitute from the recording" in result.detail


class TestToolCalls:
    async def test_falls_back_when_tool_choice_is_rejected(self, mock_server, probe_ctx):
        mock_server.script(
            MockTurn(status=400, error={"error": {"message": "tool_choice unsupported"}}),
            MockTurn(tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}]),
        )

        (result,) = await probe_tool_calls(probe_ctx)

        assert result.support is Support.YES
        assert "auto" in result.detail

    async def test_no_call_is_not_blamed_on_the_server(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="It is sunny in Paris."))

        (result,) = await probe_tool_calls(probe_ctx)

        assert result.support is Support.UNKNOWN
        assert "model behaviour" in result.detail

    async def test_streaming_requires_structured_deltas(self, mock_server, probe_ctx):
        mock_server.script(
            MockTurn(tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}])
        )

        (result,) = await probe_streaming_tool_calls(probe_ctx)

        assert result.support is Support.YES
        assert result.evidence["structured_tool_call_deltas"] is True

    async def test_streaming_without_a_call(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="no tools today"))

        (result,) = await probe_streaming_tool_calls(probe_ctx)

        assert result.support is Support.NO


class TestOtherProbes:
    async def test_reasoning_detected(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="hello", reasoning="the user wants a greeting"))

        (result,) = await probe_reasoning(probe_ctx)

        assert result.support is Support.YES
        assert "reasoning_content" in result.detail

    async def test_reasoning_absent(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="hello"))

        (result,) = await probe_reasoning(probe_ctx)

        assert result.support is Support.NO

    async def test_usage_in_stream(self, mock_server, probe_ctx):
        mock_server.script(MockTurn(content="hello"))

        (result,) = await probe_usage_in_stream(probe_ctx)

        assert result.support is Support.YES
        assert result.evidence["usage"]["total_tokens"] == 18

    async def test_missing_chat_template_endpoint(self, mock_server, probe_ctx):
        mock_server.absent_paths.add("/apply-template")

        (result,) = await probe_chat_template_endpoint(probe_ctx)

        assert result.support is Support.NO

    @pytest.mark.parametrize("content, support", [
        # llama.cpp returns the prefill with what it continued with
        ("Counting in French, one word each: un, deux, trois, quatre, cinq.", Support.YES),
        # others return only the continuation
        (" trois, quatre, cinq.", Support.YES),
        # a restart counts again from the start
        ("Un, deux, trois, quatre, cinq.", Support.NO),
    ])
    async def test_partial_assistant_is_decided(self, mock_server, probe_ctx, content, support):
        mock_server.script(MockTurn(content=content))

        (result,) = await probe_partial_assistant(probe_ctx)

        assert result.support is support
        assert mock_server.requests[-1]["messages"][-1]["role"] == "assistant"


class TestFreshPrompts:
    """Whether the server processes every prompt whole. With llama.cpp's prompt cache on, a
    Claude Code step's samples swung from 16/20 to 0/20 on one behaviour with what the server
    had run before; started with --no-cache-prompt, three resamples agreed."""

    @staticmethod
    def server(second_read: int | None, *, messages: bool = True):
        """A server whose second answer to a repeated prompt reads `second_read` tokens from
        its cache (None: it says nothing of a cache)."""
        calls: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/v1/messages":
                if not messages:
                    return httpx.Response(404, json={"error": {"message": "no"}})
                usage = {"input_tokens": 600, "output_tokens": 1}
                if second_read is not None:
                    usage["cache_read_input_tokens"] = second_read if calls.count("/v1/messages") > 1 else 0
                return httpx.Response(200, json={"type": "message", "content": [], "usage": usage})
            body = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 600}}
            if second_read is not None:
                body["timings"] = {"cache_n": second_read if calls.count(request.url.path) > 1 else 0}
            return httpx.Response(200, json=body)

        return httpx.AsyncClient(transport=httpx.MockTransport(answer)), calls

    @pytest.mark.parametrize("read, support", [(590, Support.NO), (0, Support.YES), (None, Support.UNKNOWN)])
    async def test_a_repeated_prompt_tells(self, read, support):
        from tuningfork.backends.probe import probe_fresh_prompts

        client, calls = self.server(read)
        async with client:
            (result,) = await probe_fresh_prompts(ProbeContext(client=client, root=UPSTREAM, model="m"))
        assert result.support is support
        if support is Support.NO:
            assert "--no-cache-prompt" in result.detail
        assert calls[:2] == ["/v1/messages", "/v1/messages"]  # Claude Code's endpoint first

    async def test_without_messages_the_chat_endpoint_tells(self):
        from tuningfork.backends.probe import probe_fresh_prompts

        client, calls = self.server(590, messages=False)
        async with client:
            (result,) = await probe_fresh_prompts(ProbeContext(client=client, root=UPSTREAM, model="m"))
        assert result.support is Support.NO and result.evidence["endpoint"] == "/v1/chat/completions"

    async def test_a_measurement_is_kept_a_while(self):
        from tuningfork.backends.llamacpp import reuses_prompts

        client, calls = self.server(0)
        async with client:
            assert await reuses_prompts(client, "http://kept.test", fresh=True) is False
            assert await reuses_prompts(client, "http://kept.test") is False
        assert len(calls) == 2

    def test_a_plan_warns_when_the_check_found_a_cache(self):
        from tuningfork.ops.connections import cached_prompts_warning

        assert "--no-cache-prompt" in cached_prompts_warning({"results": {"fresh_prompts": {"support": "no"}}})
        assert cached_prompts_warning({"results": {"fresh_prompts": {"support": "yes"}}}) is None
        assert cached_prompts_warning(None) is None


class TestTokenLevel:
    """What scoring an answer and forcing a token need, measured on the toy llama.cpp."""

    async def test_logprobs_that_cover_every_generated_token(self, mock_server, probe_ctx):
        entries = [{"id": 1, "token": "hi", "logprob": -0.1, "top_logprobs": [{"id": 1, "token": "hi", "logprob": -0.1}]}]
        mock_server.script(MockTurn(content="hi", logprobs={"content": entries},
                                    usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}))
        results = by_capability(await probe_logprobs(probe_ctx))
        assert results["logprobs_all_tokens"].support is Support.YES

    async def test_logprobs_for_the_reply_alone(self, mock_server, probe_ctx):
        entries = [{"id": 1, "token": "hi", "logprob": -0.1, "top_logprobs": [{"id": 1, "token": "hi", "logprob": -0.1}]}]
        mock_server.script(MockTurn(content="hi", logprobs={"content": entries},
                                    usage={"prompt_tokens": 3, "completion_tokens": 40, "total_tokens": 43}))
        results = by_capability(await probe_logprobs(probe_ctx))
        assert results["logprobs_all_tokens"].support is Support.NO
        assert "1 entries for 40" in results["logprobs_all_tokens"].detail

    async def test_a_prompt_rendered_as_the_chat_endpoint_renders_it(self, probe_ctx):
        (result,) = await probe_prompt_equivalence(probe_ctx)
        assert result.support is Support.YES

    async def test_forced_tokens_with_their_probabilities(self, probe_ctx):
        (result,) = await probe_raw_completion(probe_ctx)
        assert result.support is Support.YES

    async def test_no_raw_endpoints(self, mock_server, probe_ctx):
        mock_server.absent_paths.add("/apply-template")
        (result,) = await probe_raw_completion(probe_ctx)
        assert result.support is Support.NO

    async def test_the_output_format(self, mock_server, probe_ctx):
        mock_server.props = toy_props()
        (result,) = await probe_output_format(probe_ctx)
        assert (result.support, result.detail) == (Support.YES, "gemma4")
        mock_server.props = {"chat_template": "{{ '<think>' }}{{ '<tool_call>\\n<function=' }}{{ '<parameter=' }}"}
        (result,) = await probe_output_format(probe_ctx)
        assert (result.support, result.detail) == (Support.YES, "qwen3.5")
        mock_server.props = {"chat_template": "{{ '<tool_call>\\n<function=' }}{{ '<parameter=' }}"}
        (result,) = await probe_output_format(probe_ctx)
        assert (result.support, result.detail) == (Support.YES, "qwen3-coder")
        mock_server.props = {"chat_template": "<|im_start|>"}
        (result,) = await probe_output_format(probe_ctx)
        assert result.support is Support.NO


class TestProfile:
    async def test_end_to_end_probe_builds_a_profile(self, mock_server):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_server.app)
        ) as client:
            profile = await probe_backend(
                f"{UPSTREAM}/v1",
                client=client,
                include_prefix_cache=False,
                performance_rounds=1,
            )

        assert profile.model == "mock-model"
        assert profile.supports(Capability.USAGE_IN_STREAM)
        assert not profile.supports(Capability.LOGPROBS)
        assert profile.performance["mean_total_ms"] is not None

    async def test_a_profile_keeps_its_address_without_a_key(self, mock_server):
        """A profile is written to be shared: a user and password in the address probed is kept
        masked, as wherever an address is kept or shown (`config.without_credentials`)."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as client:
            profile = await probe_backend("http://tf:Sup3rS3cr3t@upstream.test/v1", client=client,
                                          include_prefix_cache=False, performance_rounds=1)
        assert profile.base_url == "http://<redacted>@upstream.test/v1" and profile.model == "mock-model"

    async def test_a_profile_keeps_what_a_refused_call_said_without_the_key(self, mock_server):
        """A key the server refuses is just when a profile records errors, and each quotes the
        address its call went to: the profile, written to be shared, keeps none of the key (found in
        review). No model listed there, a probe says so of the address masked too."""
        import json

        def refusing(message: str) -> None:  # a probe reports a server's message verbatim
            mock_server.responder = lambda payload: MockTurn(status=401, error={"error": {"message": message}})

        async def kept(address: str) -> str:
            profile = await probe_backend(address, client=client, include_prefix_cache=False, performance_rounds=1)
            return json.dumps(profile.to_dict())

        key = "sk-" "a1B2c3D4e5F6g7H8i9J0"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as client:
            # A password no pattern reads as a key, a word, and shorter than a value looked for
            # elsewhere: masked where its address stands, as given or as httpx spells it (found in
            # review: an error quotes a host in lower case, and no `:80`), or not at all.
            refusing(f"bad key {key}")
            given = await kept("http://tf:hunt@upstream.test/v1")
            spelled = await kept("http://tf:hunt@UPSTREAM.test:80/v1")
            # And a password the refusal quotes back alone.
            refusing("bad password hunter2xyz")
            echoed = await kept("http://tf:hunter2xyz@upstream.test/v1")
            # A query cuts the path each call is sent to: no model is listed there.
            with pytest.raises(RuntimeError) as listed_none:
                await probe_backend("http://upstream.test/v1?k=abc1longer9", client=client,
                                    include_prefix_cache=False, performance_rounds=1)
        for profile in (given, spelled, echoed):
            assert "401" in profile and "<redacted>" in profile and "hunt" not in profile
        assert "bad key" in given and key not in given + spelled and "bad password" in echoed
        assert "abc1longer9" not in str(listed_none.value) and "k=<redacted>" in str(listed_none.value)

    async def test_permissive_catch_all_is_not_mistaken_for_lm_studio(self, mock_server):
        """A server that answers every path must not be fingerprinted from a 200 alone."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_server.app)
        ) as client:
            profile = await probe_backend(
                UPSTREAM, client=client, include_prefix_cache=False, performance_rounds=1
            )

        assert profile.server != "lmstudio"

    def test_roundtrip_and_actionable_error(self, tmp_path):
        profile = BackendProfile(
            name="local",
            base_url="http://h:8081",
            server="lmstudio",
            model="m",
            probed_at="2026-09-22T12:00:00+00:00",
            capabilities={
                Capability.LOGPROBS: CapabilityResult(
                    Capability.LOGPROBS, Support.NO, "came back null"
                )
            },
        )

        restored = BackendProfile.load(profile.save(tmp_path / "p.json"))

        assert restored.capabilities[Capability.LOGPROBS].support is Support.NO
        with pytest.raises(UnsupportedCapability) as excinfo:
            restored.require(Capability.LOGPROBS)
        message = str(excinfo.value)
        assert "token-level analysis" in message  # says what it was needed for
        assert "came back null" in message  # and why it is unavailable

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("/home/alice/models/m.gguf", "<home>/models/m.gguf"),
            ("/Users/bob/models/m.gguf", "<home>/models/m.gguf"),
            ("/mnt/c/Users/carol/models/m.gguf", "<home>/models/m.gguf"),
            ("http://127.0.0.1:8081", "http://127.0.0.1:8081"),
        ],
    )
    def test_home_directories_are_stripped(self, raw, expected):
        assert anonymize_paths({"a": [raw]})["a"][0] == expected

    def test_saved_profile_is_shareable_by_default(self, tmp_path):
        """A profile is the obvious first contribution, so it must carry no username."""
        profile = BackendProfile(
            name="local",
            base_url="http://127.0.0.1:8081",
            server="llama.cpp",
            model="/home/someone/models/gemma.gguf",
            probed_at="2026-09-22T12:00:00+00:00",
            notes=["model_path: /home/someone/models/gemma.gguf"],
        )

        text = profile.save(tmp_path / "p.json").read_text(encoding="utf-8")

        assert "someone" not in text
        assert "<home>/models/gemma.gguf" in text

    def test_anonymizing_can_be_declined(self, tmp_path):
        profile = BackendProfile(
            name="local",
            base_url="u",
            server="s",
            model="/home/someone/m.gguf",
            probed_at="t",
        )
        text = profile.save(tmp_path / "p.json", anonymize=False).read_text()
        assert "someone" in text

    def test_unprobed_capability_is_not_assumed(self):
        profile = BackendProfile(
            name="x", base_url="u", server="s", model="m", probed_at="t"
        )
        with pytest.raises(UnsupportedCapability, match="never probed"):
            profile.require(Capability.LOGPROBS)

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("http://h:8081", "http://h:8081"),
            ("http://h:8081/", "http://h:8081"),
            ("http://h:8081/v1", "http://h:8081"),
            ("http://h:8081/v1/", "http://h:8081"),
        ],
    )
    def test_server_root_normalisation(self, given, expected):
        assert server_root(given) == expected
