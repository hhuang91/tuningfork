"""Runtime capability probing.

Ask the server what it can do rather than inferring it from a name and version. Probes run
strictly one at a time: the server being probed may be shared with something that matters,
and a burst of parallel requests is precisely the thing tuningfork must not do.

Every probe is failure-tolerant. A server that rejects an unknown parameter, returns a
surprising shape, or drops the connection yields `UNKNOWN` with the reason attached, never
an exception that aborts the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

import httpx

from tuningfork.backends.base import (
    BackendProfile,
    Capability,
    CapabilityResult,
    Support,
)
from tuningfork.config import server_root  # noqa: F401 - also imported from here
from tuningfork.proxy.sse import ChunkAssembler, SSEParser

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Reasoning models spend tokens before they act: on the first real probe of this stack the
# model burned 44 tokens of reasoning before emitting a tool call. A tight max_tokens would
# have looked exactly like "this server cannot do tool calls".
TOOL_PROBE_MAX_TOKENS = 384
SHORT_PROBE_MAX_TOKENS = 192


@dataclass(slots=True)
class ProbeContext:
    client: httpx.AsyncClient
    root: str
    model: str
    timeout: float = 180.0

    @property
    def chat_url(self) -> str:
        return f"{self.root}/v1/chat/completions"

    async def chat(self, **payload: Any) -> httpx.Response:
        return await self.client.post(
            self.chat_url, json={"model": self.model, **payload}, timeout=self.timeout
        )

    async def collect_stream(self, **payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a streaming request; return (assembled response, stream statistics)."""
        parser, assembler = SSEParser(), ChunkAssembler()
        started = time.monotonic()
        ttft: float | None = None
        saw_tool_call_delta = False

        async with self.client.stream(
            "POST",
            self.chat_url,
            json={"model": self.model, "stream": True, **payload},
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            async for raw in response.aiter_raw():
                if ttft is None:
                    ttft = time.monotonic() - started
                for event in parser.feed(raw):
                    chunk = event.json()
                    if chunk:
                        for choice in chunk.get("choices") or []:
                            if (choice.get("delta") or {}).get("tool_calls"):
                                saw_tool_call_delta = True
                    assembler.add_event(event)

        stats = {
            "ttft_ms": None if ttft is None else round(ttft * 1000, 2),
            "total_ms": round((time.monotonic() - started) * 1000, 2),
            "chunks": assembler.chunk_count,
            "saw_done": assembler.saw_done,
            "structured_tool_call_deltas": saw_tool_call_delta,
        }
        return assembler.result(), stats


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return f"HTTP {response.status_code}: {error['message'][:200]}"
    return f"HTTP {response.status_code}"


# --- individual probes --------------------------------------------------------------


async def probe_tool_calls(ctx: ProbeContext) -> list[CapabilityResult]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "max_tokens": TOOL_PROBE_MAX_TOKENS,
        "tool_choice": "required",
    }
    response = await ctx.chat(**payload)
    forced = True
    if response.status_code >= 400:
        # Not every server implements tool_choice; fall back before concluding anything.
        forced = False
        payload.pop("tool_choice")
        response = await ctx.chat(**payload)
    if response.status_code >= 400:
        return [
            CapabilityResult(
                Capability.TOOL_CALLS, Support.NO, _error_detail(response)
            )
        ]

    message = _message(response.json())
    calls = message.get("tool_calls") or []
    if calls:
        return [
            CapabilityResult(
                Capability.TOOL_CALLS,
                Support.YES,
                f"tool_choice={'required' if forced else 'auto'}",
                {"name": calls[0].get("function", {}).get("name")},
            )
        ]
    return [
        CapabilityResult(
            Capability.TOOL_CALLS,
            Support.UNKNOWN,
            "server accepted tools but the model did not call one; "
            "this may be model behaviour rather than a server limit",
            {"finish_reason": (response.json().get("choices") or [{}])[0].get("finish_reason")},
        )
    ]


async def probe_streaming_tool_calls(ctx: ProbeContext) -> list[CapabilityResult]:
    result, stats = await ctx.collect_stream(
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        max_tokens=TOOL_PROBE_MAX_TOKENS,
    )
    calls = _message(result).get("tool_calls") or []
    if calls and stats["structured_tool_call_deltas"]:
        return [
            CapabilityResult(
                Capability.STREAMING_TOOL_CALLS,
                Support.YES,
                "structured delta.tool_calls observed",
                stats,
            )
        ]
    if calls:
        return [
            CapabilityResult(
                Capability.STREAMING_TOOL_CALLS,
                Support.UNKNOWN,
                "tool call reassembled but no structured delta seen",
                stats,
            )
        ]
    return [
        CapabilityResult(
            Capability.STREAMING_TOOL_CALLS,
            Support.NO,
            "no tool call in the stream",
            stats,
        )
    ]


async def probe_anthropic_messages(ctx: ProbeContext) -> list[CapabilityResult]:
    """Whether the server answers the Anthropic Messages API: one short, non-streaming
    call. What Claude Code needs; llama-server has it, an OpenAI-only server does not."""
    response = await ctx.client.post(
        f"{ctx.root}/v1/messages",
        json={
            "model": ctx.model,
            "max_tokens": SHORT_PROBE_MAX_TOKENS,
            "messages": [{"role": "user", "content": "Say exactly: hello"}],
        },
        headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-tuningfork-probe"},
        timeout=ctx.timeout,
    )
    if response.status_code >= 400:
        detail = "the server has no /v1/messages" if response.status_code == 404 else _error_detail(response)
        return [CapabilityResult(Capability.ANTHROPIC_MESSAGES, Support.NO, detail)]
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("type") == "message" and isinstance(body.get("content"), list):
        return [
            CapabilityResult(
                Capability.ANTHROPIC_MESSAGES, Support.YES, "a Messages response",
                {"stop_reason": body.get("stop_reason")},
            )
        ]
    return [
        CapabilityResult(
            Capability.ANTHROPIC_MESSAGES, Support.UNKNOWN,
            "answered, but not with a Messages response",
        )
    ]


async def probe_reasoning(ctx: ProbeContext) -> list[CapabilityResult]:
    response = await ctx.chat(
        messages=[{"role": "user", "content": "Say exactly: hello"}],
        max_tokens=SHORT_PROBE_MAX_TOKENS,
    )
    if response.status_code >= 400:
        return [
            CapabilityResult(
                Capability.REASONING_CONTENT, Support.UNKNOWN, _error_detail(response)
            )
        ]
    message = _message(response.json())
    for key in ("reasoning_content", "reasoning"):
        if message.get(key):
            return [
                CapabilityResult(
                    Capability.REASONING_CONTENT,
                    Support.YES,
                    f"field `{key}`",
                    {"sample": str(message[key])[:160]},
                )
            ]
    return [
        CapabilityResult(
            Capability.REASONING_CONTENT, Support.NO, "no reasoning field in the response"
        )
    ]


async def probe_logprobs(ctx: ProbeContext) -> list[CapabilityResult]:
    response = await ctx.chat(
        messages=[{"role": "user", "content": "Say exactly: hello"}],
        max_tokens=SHORT_PROBE_MAX_TOKENS,
        logprobs=True,
        top_logprobs=3,
    )
    if response.status_code >= 400:
        detail = _error_detail(response)
        return [
            CapabilityResult(Capability.LOGPROBS, Support.NO, detail),
            CapabilityResult(Capability.TOP_LOGPROBS, Support.NO, detail),
            CapabilityResult(Capability.LOGPROBS_ALL_TOKENS, Support.NO, detail),
        ]

    body = response.json()
    choices = body.get("choices") or []
    logprobs = choices[0].get("logprobs") if choices else None
    if not logprobs:
        # The quiet failure mode: accepted, ignored, null. Worth naming explicitly, since
        # it looks like success from the status code alone.
        detail = "request accepted but `logprobs` came back null"
        return [
            CapabilityResult(Capability.LOGPROBS, Support.NO, detail),
            CapabilityResult(Capability.TOP_LOGPROBS, Support.NO, detail),
            CapabilityResult(Capability.LOGPROBS_ALL_TOKENS, Support.NO, detail),
        ]

    entries = logprobs.get("content") or []
    has_top = bool(entries and entries[0].get("top_logprobs"))
    # A reasoning model generates far more than its visible reply: llama.cpp's entries cover
    # every token — reasoning, channel markers, tool-call syntax — others may give the reply's.
    generated = (body.get("usage") or {}).get("completion_tokens")
    if generated is None:
        coverage = CapabilityResult(
            Capability.LOGPROBS_ALL_TOKENS, Support.UNKNOWN,
            f"{len(entries)} entries, and no token count to compare them with",
        )
    else:
        covers = len(entries) == generated
        coverage = CapabilityResult(
            Capability.LOGPROBS_ALL_TOKENS,
            Support.YES if covers else Support.NO,
            f"{len(entries)} entries for {generated} generated tokens"
            + ("" if covers else ": some tokens (reasoning, markup) carry none"),
        )
    return [
        CapabilityResult(
            Capability.LOGPROBS, Support.YES, f"{len(entries)} token entries"
        ),
        CapabilityResult(
            Capability.TOP_LOGPROBS,
            Support.YES if has_top else Support.NO,
            "top_logprobs present" if has_top else "no top_logprobs in entries",
        ),
        coverage,
    ]


async def probe_server_side_n(ctx: ProbeContext) -> list[CapabilityResult]:
    response = await ctx.chat(
        messages=[{"role": "user", "content": "Name one colour."}],
        max_tokens=SHORT_PROBE_MAX_TOKENS,
        n=3,
    )
    if response.status_code >= 400:
        return [
            CapabilityResult(
                Capability.SERVER_SIDE_N, Support.NO, _error_detail(response)
            )
        ]
    count = len(response.json().get("choices") or [])
    if count >= 2:
        return [
            CapabilityResult(
                Capability.SERVER_SIDE_N, Support.YES, f"{count} choices for n=3"
            )
        ]
    return [
        CapabilityResult(
            Capability.SERVER_SIDE_N,
            Support.NO,
            f"n=3 silently returned {count} choice; fan-out must use concurrent requests",
        )
    ]


async def probe_seed_determinism(ctx: ProbeContext) -> list[CapabilityResult]:
    prompt = "Write one short sentence about the sea."
    outputs = []
    for _ in range(2):
        response = await ctx.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=SHORT_PROBE_MAX_TOKENS,
            temperature=0.9,
            seed=1234,
        )
        if response.status_code >= 400:
            return [
                CapabilityResult(
                    Capability.SEED_DETERMINISM, Support.NO, _error_detail(response)
                )
            ]
        message = _message(response.json())
        # A reasoning model can spend the whole budget thinking: compare what it thought too,
        # or two empty replies read as "identical".
        outputs.append((message.get("reasoning_content") or "") + (message.get("content") or ""))

    if outputs[0] and outputs[0] == outputs[1]:
        return [
            CapabilityResult(
                Capability.SEED_DETERMINISM,
                Support.YES,
                "two seeded calls produced identical text",
            )
        ]
    return [
        CapabilityResult(
            Capability.SEED_DETERMINISM,
            Support.NO,
            "same seed produced different text; replay must substitute from the recording "
            "rather than regenerate",
            {"first": outputs[0][:80], "second": outputs[1][:80]},
        )
    ]


async def probe_usage_in_stream(ctx: ProbeContext) -> list[CapabilityResult]:
    result, stats = await ctx.collect_stream(
        messages=[{"role": "user", "content": "Say exactly: hello"}],
        max_tokens=SHORT_PROBE_MAX_TOKENS,
        stream_options={"include_usage": True},
    )
    usage = result.get("usage")
    return [
        CapabilityResult(
            Capability.USAGE_IN_STREAM,
            Support.YES if usage else Support.NO,
            "usage chunk present" if usage else "no usage chunk in the stream",
            {"usage": usage} if usage else stats,
        )
    ]


#: A beginning no model would start a reply with on its own: a reply that restarts cannot look
#: like one that continues.
PREFILL = "Counting in French, one word each: un, deux,"


async def probe_partial_assistant(ctx: ProbeContext) -> list[CapabilityResult]:
    messages = [
        {"role": "user", "content": "Count from one to five in French."},
        {"role": "assistant", "content": PREFILL},
    ]
    response = await ctx.chat(messages=messages, max_tokens=SHORT_PROBE_MAX_TOKENS, temperature=0)
    used_flag = False
    if response.status_code >= 400:
        # vLLM gates this behind an explicit flag.
        response = await ctx.chat(
            messages=messages,
            max_tokens=SHORT_PROBE_MAX_TOKENS,
            temperature=0,
            continue_final_message=True,
            add_generation_prompt=False,
        )
        used_flag = True
    if response.status_code >= 400:
        return [
            CapabilityResult(
                Capability.PARTIAL_ASSISTANT_CONTINUATION,
                Support.NO,
                _error_detail(response),
            )
        ]
    content = (_message(response.json()).get("content") or "").strip()
    evidence = {"continuation": content[:160], "used_flag": used_flag}
    # llama.cpp returns the prefill with its continuation; others return the continuation. A
    # restart counts again from "un", which neither begins with.
    lowered = content.lower()
    if (content.startswith(PREFILL) and "trois" in lowered[len(PREFILL) : len(PREFILL) + 24]) or lowered.startswith("trois"):
        return [CapabilityResult(
            Capability.PARTIAL_ASSISTANT_CONTINUATION, Support.YES,
            "a trailing assistant message is continued" + (" with continue_final_message" if used_flag else ""),
            evidence,
        )]
    return [CapabilityResult(
        Capability.PARTIAL_ASSISTANT_CONTINUATION, Support.NO,
        "a trailing assistant message was answered afresh, not continued", evidence,
    )]


async def probe_chat_template_endpoint(ctx: ProbeContext) -> list[CapabilityResult]:
    response = await ctx.client.post(
        f"{ctx.root}/apply-template",
        json={"messages": [{"role": "user", "content": "hi"}]},
        timeout=ctx.timeout,
    )
    if response.status_code >= 400:
        return [
            CapabilityResult(
                Capability.CHAT_TEMPLATE_ENDPOINT,
                Support.NO,
                f"/apply-template returned HTTP {response.status_code}",
            )
        ]
    try:
        prompt = response.json().get("prompt")
    except ValueError:
        prompt = None
    return [
        CapabilityResult(
            Capability.CHAT_TEMPLATE_ENDPOINT,
            Support.YES if prompt else Support.UNKNOWN,
            "/apply-template renders a prompt" if prompt else "unexpected response shape",
            {"prompt_preview": str(prompt)[:160] if prompt else None},
        )
    ]


async def _render_ids(ctx: ProbeContext, body: dict[str, Any]) -> list[int] | None:
    """A chat body as the server renders and tokenizes it, as ids; None if it cannot."""
    rendered = await ctx.client.post(f"{ctx.root}/apply-template", json=body, timeout=ctx.timeout)
    if rendered.status_code >= 400:
        return None
    prompt = rendered.json().get("prompt")
    tokenized = await ctx.client.post(
        f"{ctx.root}/tokenize",
        json={"content": prompt, "add_special": True, "parse_special": True},
        timeout=ctx.timeout,
    )
    if tokenized.status_code >= 400:
        return None
    tokens = tokenized.json().get("tokens")
    return [t if isinstance(t, int) else t.get("id") for t in tokens] if isinstance(tokens, list) else None


async def probe_prompt_equivalence(ctx: ProbeContext) -> list[CapabilityResult]:
    """Rendering and tokenizing a chat body gives the prompt the chat endpoint would use:
    the same number of tokens it reports, tools and all."""
    body = {"model": ctx.model, "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [WEATHER_TOOL]}
    ids = await _render_ids(ctx, body)
    if ids is None:
        return [CapabilityResult(Capability.PROMPT_EQUIVALENCE, Support.NO,
                                 "no /apply-template and /tokenize to render a prompt with")]
    counted = await ctx.client.post(f"{ctx.root}/v1/chat/completions/input_tokens", json=body, timeout=ctx.timeout)
    expected = counted.json().get("input_tokens") if counted.status_code < 400 else None
    if expected is None:
        answered = await ctx.chat(messages=body["messages"], tools=body["tools"], max_tokens=1)
        expected = (answered.json().get("usage") or {}).get("prompt_tokens") if answered.status_code < 400 else None
    if expected is None:
        return [CapabilityResult(Capability.PROMPT_EQUIVALENCE, Support.UNKNOWN,
                                 f"rendered {len(ids)} tokens; the server reported no count to compare")]
    same = len(ids) == expected
    return [CapabilityResult(
        Capability.PROMPT_EQUIVALENCE, Support.YES if same else Support.NO,
        f"rendered {len(ids)} tokens, the chat endpoint's prompt has {expected}",
    )]


async def probe_raw_completion(ctx: ProbeContext) -> list[CapabilityResult]:
    """Continuing from token ids with each token's probabilities, with tokens forced by a
    grammar of their ids: what scoring an answer and forcing a token need."""
    ids = await _render_ids(ctx, {"messages": [{"role": "user", "content": "Say exactly: hello"}]})
    if ids is None:
        return [CapabilityResult(Capability.RAW_COMPLETION, Support.NO,
                                 "no /apply-template and /tokenize to render a prompt with")]
    tokenized = await ctx.client.post(
        f"{ctx.root}/tokenize", json={"content": "Hello there", "add_special": False, "parse_special": True},
        timeout=ctx.timeout,
    )
    forced = [t if isinstance(t, int) else t.get("id") for t in tokenized.json().get("tokens") or []]
    response = await ctx.client.post(
        f"{ctx.root}/completion",
        json={"prompt": ids, "n_predict": len(forced) + 1, "n_probs": 2, "temperature": 0,
              "grammar": "root ::= " + " ".join(f"<[{t}]>" for t in forced), "return_tokens": True},
        timeout=ctx.timeout,
    )
    if response.status_code >= 400:
        return [CapabilityResult(Capability.RAW_COMPLETION, Support.NO, _error_detail(response))]
    data = response.json()
    entries = data.get("completion_probabilities") or []
    kept = (data.get("tokens") or [])[: len(forced)] == forced
    probabilities = bool(entries and entries[0].get("top_logprobs"))
    if kept and probabilities:
        return [CapabilityResult(Capability.RAW_COMPLETION, Support.YES,
                                 f"{len(forced)} tokens forced by a grammar of ids, with their top alternatives")]
    return [CapabilityResult(
        Capability.RAW_COMPLETION, Support.NO,
        "the forced tokens were not kept" if not kept else "no per-token probabilities came back",
    )]


async def probe_output_format(ctx: ProbeContext) -> list[CapabilityResult]:
    from tuningfork.backends.formats import format_for

    response = await ctx.client.get(f"{ctx.root}/props", timeout=ctx.timeout)
    if response.status_code >= 400:
        return [CapabilityResult(Capability.KNOWN_OUTPUT_FORMAT, Support.UNKNOWN,
                                 "no /props to read the chat template from")]
    template = response.json().get("chat_template")
    found = format_for(template if isinstance(template, str) else None)
    if found is None:
        return [CapabilityResult(Capability.KNOWN_OUTPUT_FORMAT, Support.NO,
                                 "the chat template is not one tuningfork reads raw output of")]
    return [CapabilityResult(Capability.KNOWN_OUTPUT_FORMAT, Support.YES, f"{found.name}")]


async def probe_fresh_prompts(ctx: ProbeContext) -> list[CapabilityResult]:
    """Whether every request's prompt is processed whole. llama-server reuses what it cached of
    an earlier prompt unless it was started with --no-cache-prompt, and its Messages endpoint,
    which Claude Code uses, ignores a request's own `cache_prompt`: then Claude Code's samples,
    and every fork's calls, depend on what the server ran before. Measured with
    the same call sent twice, on the Messages endpoint where there is one."""
    from tuningfork.backends.llamacpp import reuses_prompts

    endpoint = "/v1/messages"
    reuses = await reuses_prompts(ctx.client, ctx.root, endpoint, model=ctx.model, fresh=True)
    if reuses is None:
        endpoint = "/v1/chat/completions"
        reuses = await reuses_prompts(ctx.client, ctx.root, endpoint, model=ctx.model, fresh=True)
    if reuses is None:
        return [CapabilityResult(Capability.FRESH_PROMPTS, Support.UNKNOWN,
                                 "the server reported no cached tokens to tell from")]
    if reuses:
        return [CapabilityResult(
            Capability.FRESH_PROMPTS, Support.NO,
            "a prompt sent twice was answered from the cache the second time: Claude Code's samples and every "
            "fork depend on what the server ran before — start llama-server with --no-cache-prompt",
            {"endpoint": endpoint},
        )]
    return [CapabilityResult(Capability.FRESH_PROMPTS, Support.YES,
                             "a prompt sent twice was processed whole both times", {"endpoint": endpoint})]


async def probe_prefix_cache(ctx: ProbeContext) -> list[CapabilityResult]:
    """Time the same long prefix twice. A cached second pass is dramatically faster."""
    filler = " ".join(f"item-{i}" for i in range(1500))
    messages = [
        {"role": "system", "content": f"Reference data follows.\n{filler}"},
        {"role": "user", "content": "Reply with the single word: ok"},
    ]
    timings = []
    for _ in range(2):
        _, stats = await ctx.collect_stream(messages=messages, max_tokens=8)
        timings.append(stats["ttft_ms"] or stats["total_ms"])

    first, second = timings
    if not first or not second:
        return [
            CapabilityResult(
                Capability.PREFIX_CACHE, Support.UNKNOWN, "could not time the requests"
            )
        ]
    ratio = second / first
    evidence = {"first_ttft_ms": first, "second_ttft_ms": second, "ratio": round(ratio, 3)}
    if ratio < 0.6:
        return [
            CapabilityResult(
                Capability.PREFIX_CACHE,
                Support.YES,
                f"second pass took {ratio:.0%} of the first",
                evidence,
            )
        ]
    return [
        CapabilityResult(
            Capability.PREFIX_CACHE,
            Support.NO,
            f"no speedup on a repeated prefix (second pass {ratio:.0%} of the first)",
            evidence,
        )
    ]


# --- server identification and performance -------------------------------------------


def _model_entries(response: httpx.Response) -> list[dict[str, Any]] | None:
    """Model-list entries if this really is a model-list response, else None."""
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    entries = body.get("data") if isinstance(body, dict) else None
    if isinstance(entries, list) and any(
        isinstance(entry, dict) and "id" in entry for entry in entries
    ):
        return [entry for entry in entries if isinstance(entry, dict)]
    return None


async def identify_server(ctx: ProbeContext) -> tuple[str, list[str]]:
    """Best-effort server fingerprint, plus notes worth recording."""
    notes: list[str] = []
    server = "openai-compatible (unidentified)"

    try:
        response = await ctx.client.get(f"{ctx.root}/api/v0/models", timeout=15)
        # A 200 alone proves nothing: plenty of servers answer every path. Require the
        # response to actually look like the endpoint we are fingerprinting.
        entries = _model_entries(response)
        if entries is not None:
            for entry in entries:
                if entry.get("id") == ctx.model:
                    notes.append(
                        "model: arch={arch} quant={quant} ctx={ctx} type={type}".format(
                            arch=entry.get("arch"),
                            quant=entry.get("quantization"),
                            ctx=entry.get("max_context_length"),
                            type=entry.get("type"),
                        )
                    )
            return "lmstudio", notes
    except httpx.HTTPError:
        pass

    try:
        response = await ctx.client.get(f"{ctx.root}/props", timeout=15)
        body = response.json() if response.status_code == 200 else None
        if isinstance(body, dict) and (
            {"model_path", "chat_template", "default_generation_settings"} & body.keys()
        ):
            server = "llama.cpp"
            if body.get("model_path"):
                notes.append(f"model_path: {body['model_path']}")
            if body.get("chat_template"):
                notes.append("chat_template exposed via /props")
    except (httpx.HTTPError, ValueError):
        pass

    return server, notes


async def measure_performance(ctx: ProbeContext, rounds: int = 3) -> dict[str, Any]:
    """Latency and throughput, so sampling budgets rest on numbers rather than guesses."""
    samples: list[dict[str, Any]] = []
    for _ in range(rounds):
        result, stats = await ctx.collect_stream(
            messages=[{"role": "user", "content": "Name one colour."}],
            max_tokens=SHORT_PROBE_MAX_TOKENS,
            stream_options={"include_usage": True},
        )
        usage = result.get("usage") or {}
        completion = usage.get("completion_tokens") or 0
        details = usage.get("completion_tokens_details") or {}
        samples.append(
            {
                "ttft_ms": stats["ttft_ms"],
                "total_ms": stats["total_ms"],
                "completion_tokens": completion,
                "reasoning_tokens": details.get("reasoning_tokens") or 0,
            }
        )

    def mean(key: str) -> float | None:
        values = [s[key] for s in samples if s[key] is not None]
        return round(sum(values) / len(values), 2) if values else None

    total_ms = mean("total_ms") or 0.0
    completion_tokens = mean("completion_tokens") or 0.0
    reasoning_tokens = mean("reasoning_tokens") or 0.0
    performance: dict[str, Any] = {
        "rounds": rounds,
        "mean_ttft_ms": mean("ttft_ms"),
        "mean_total_ms": mean("total_ms"),
        "mean_completion_tokens": completion_tokens,
        "mean_reasoning_tokens": reasoning_tokens,
        "samples": samples,
    }
    if total_ms and completion_tokens:
        performance["tokens_per_second"] = round(completion_tokens / (total_ms / 1000), 2)
    if completion_tokens:
        # How much of a resample is spent thinking rather than acting. Directly inflates
        # the cost of every experiment.
        performance["reasoning_share"] = round(reasoning_tokens / completion_tokens, 3)
    return performance


# --- orchestration -------------------------------------------------------------------

ProbeFn = Callable[[ProbeContext], Awaitable[list[CapabilityResult]]]

CORE_PROBES: tuple[tuple[str, ProbeFn], ...] = (
    ("tool_calls", probe_tool_calls),
    ("streaming_tool_calls", probe_streaming_tool_calls),
    ("reasoning", probe_reasoning),
    ("logprobs", probe_logprobs),
    ("server_side_n", probe_server_side_n),
    ("usage_in_stream", probe_usage_in_stream),
    ("seed_determinism", probe_seed_determinism),
    ("partial_assistant", probe_partial_assistant),
    ("chat_template_endpoint", probe_chat_template_endpoint),
    ("prompt_equivalence", probe_prompt_equivalence),
    ("raw_completion", probe_raw_completion),
    ("output_format", probe_output_format),
)

#: What token-level work needs, measured by the server check (the Connections page): scoring
#: and forking tokens, through llama.cpp's own endpoints; a resample's tokens, through the chat
#: endpoint's logprobs.
TOKEN_PROBES: tuple[tuple[str, ProbeFn], ...] = (
    ("output_format", probe_output_format),
    ("prompt_equivalence", probe_prompt_equivalence),
    ("raw_completion", probe_raw_completion),
    ("logprobs", probe_logprobs),
)


async def discover_model(ctx: ProbeContext) -> str | None:
    try:
        response = await ctx.client.get(f"{ctx.root}/v1/models", timeout=15)
        response.raise_for_status()
        entries = response.json().get("data") or []
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    return entries[0].get("id") if entries else None


async def probe_backend(
    base_url: str,
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
    name: str | None = None,
    include_prefix_cache: bool = True,
    performance_rounds: int = 3,
) -> BackendProfile:
    """Probe a live server. Runs sequentially; never more than one request in flight."""
    from tuningfork.endpoints import check_endpoint, guard_hooks

    check_endpoint(base_url, purpose="the probe")
    root = server_root(base_url)
    owned = client is None
    http = client or httpx.AsyncClient(timeout=180.0, event_hooks=guard_hooks())
    try:
        ctx = ProbeContext(client=http, root=root, model=model or "")
        if not ctx.model:
            ctx.model = await discover_model(ctx) or ""
        if not ctx.model:
            raise RuntimeError(
                f"no model id given and {root}/v1/models did not list one; pass --model"
            )

        server, notes = await identify_server(ctx)
        capabilities: dict[str, CapabilityResult] = {}

        probes = list(CORE_PROBES)
        if include_prefix_cache:
            probes.append(("prefix_cache", probe_prefix_cache))

        for label, probe in probes:
            try:
                results = await probe(ctx)
            except Exception as exc:  # noqa: BLE001 - a probe must never abort the run
                results = [
                    CapabilityResult(
                        label, Support.UNKNOWN, f"probe raised {type(exc).__name__}: {exc}"
                    )
                ]
            for result in results:
                capabilities[str(result.capability)] = result

        try:
            performance = await measure_performance(ctx, rounds=performance_rounds)
        except Exception as exc:  # noqa: BLE001
            performance = {"error": f"{type(exc).__name__}: {exc}"}

        return BackendProfile(
            name=name or server,
            base_url=base_url,
            server=server,
            model=ctx.model,
            probed_at=datetime.now(UTC).isoformat(timespec="seconds"),
            capabilities=capabilities,
            performance=performance,
            notes=notes,
        )
    finally:
        if owned:
            await http.aclose()
