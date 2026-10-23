"""Scoring a recorded answer: the model's own distribution at every token of it.

No recorded step carries its tokens — Hermes never asks for logprobs, and the Messages API has
none — so a step's heatmap comes from scoring its answer afterwards. The answer is written back
as the raw text the model generated (the output format's `render`), tokenized, and forced
after the step's exact prompt; the model's top-k at each position is read on the way.

Two checks say whether that is the run's own answer, token for token, and both are kept:

- **the prompt**: rendered now, it must have the number of tokens the step recorded, or the
  model is being asked about a different context. A mismatch is refused.
- **the answer**: rebuilt, it must have the recorded number of tokens (plus the end token the
  model added). If it does not — a number written differently, a secret the recording masked —
  the heatmap is still of this text, and says it was rebuilt approximately.

A call's arguments are written back as the server sent them: where the step kept its raw
stream, read again from it (`recorded_message`), since a Messages step recorded before its wire
kept that text has its calls' inputs only parsed.

Where the server reads several texts as the same answer — an empty reasoning it drops, a blank
line between calls (`renderings`) — the one with the recorded number of tokens is scored. Where
several have it, the one the model finds likelier where they part is scored, and the record says
how many there were (`texts_alike`): the heatmap is then of the likelier text, not surely the
run's own.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from tuningfork.analysis.normalize import outcome_for, recorded_outcome
from tuningfork.analysis.tokens import from_entries, redact_stream
from tuningfork.backends.formats import FormatError, format_for, known_models
from tuningfork.store.db import Store


class ScoringError(RuntimeError):
    """The answer cannot be scored as the run's own: said why."""


@asynccontextmanager
async def raw_session(
    store: Store, base_url: str, *, tag: str, timeout: float = 600.0, progress: Any = None
) -> AsyncIterator[tuple[httpx.AsyncClient, Any, str]]:
    """A client, the raw path over it, and the id of the server it reaches.

    Every model call is tagged as an experiment's — a proxy on the way gates it and records
    nothing — and waits out a paused proxy rather than failing. A protected endpoint is refused
    before anything is sent.
    """
    from tuningfork.backends.raw import raw_server
    from tuningfork.endpoints import guard_hooks
    from tuningfork.experiment.sampling import send_experiment_call
    from tuningfork.proxy.recorder import fetch_server_identity
    from tuningfork.store.db import identity_id

    async with httpx.AsyncClient(timeout=timeout, event_hooks=guard_hooks(home=store.home)) as client:

        async def send(url: str, body: dict[str, Any]) -> httpx.Response:
            response, _ = await send_experiment_call(
                client, url, body, tag=tag, timeout=timeout, progress=progress
            )
            return response

        raw = await raw_server(client, base_url, send=send)
        server = identity_id(await fetch_server_identity(client, base_url))
        yield client, raw, server


def _chance(entry: dict[str, Any], token: int) -> float:
    """A token's log-probability at a scored position: the forced token's own, or an
    alternative's among the top k. One not known there is less likely than any that is."""
    if entry.get("id") == token and entry.get("logprob") is not None:
        return entry["logprob"]
    seen = {alt.get("id"): alt.get("logprob") for alt in entry.get("top_logprobs") or []}
    found = seen.get(token)
    return found if found is not None else float("-inf")


def _parting(a: list[int], b: list[int]) -> int | None:
    """The first position where two answers' tokens differ; None where one begins the other."""
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)


def recorded_message(store: Store, step: Any) -> tuple[dict[str, Any], str | None]:
    """The step's recorded answer in the common shape, and how it finished."""
    response = store.response(step) or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    return _as_streamed(store, step, choice.get("message") or {}), choice.get("finish_reason")


def _as_streamed(store: Store, step: Any, message: dict[str, Any]) -> dict[str, Any]:
    """The message with each call's arguments as the server streamed them, read again from the
    raw stream the step keeps, wherever they read as the same values. A Messages step recorded
    before its wire kept that text stored its calls' inputs written again from the parsed object
    (`wire.anthropic`): the model's own text is in the stream alone."""
    calls = message.get("tool_calls")
    if not calls or not step.raw_response_blob or not (step.params or {}).get("streamed"):
        return message
    from tuningfork.wire import get_wire

    wire = get_wire(step.wire)
    raw = store.blobs.get(step.raw_response_blob)
    try:
        native = wire.parse(raw, True)
        again = wire.canonical_response(native, stream=raw) if native else {}
        choices = again.get("choices") or [{}]
        texts = [
            (call.get("function") or {}).get("arguments")
            for call in (choices[0].get("message") or {}).get("tool_calls") or []
        ]
    except Exception:  # noqa: BLE001 - a stream that cannot be read leaves the message as stored
        return message
    if len(texts) != len(calls):
        return message
    kept = []
    for call, text in zip(calls, texts):
        function = call.get("function") or {}
        if isinstance(text, str) and _values(text) is not None and _values(text) == _values(function.get("arguments")):
            call = {**call, "function": {**function, "arguments": text}}
        kept.append(call)
    return {**message, "tool_calls": kept}


def _values(arguments: Any) -> Any:
    """A call's arguments as values, or None where they are no JSON text, or name a key twice:
    then the values hide part of what the text says, which may be what a redaction masked."""
    if not isinstance(arguments, str):
        return None
    try:
        return json.loads(arguments, object_pairs_hook=_once)
    except (ValueError, RecursionError):
        return None


def _once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) < len(pairs):
        raise ValueError("a key named twice")
    return dict(pairs)


async def score_answer(
    store: Store,
    checkpoint: Any,
    raw: Any,
    *,
    top_k: int = 8,
    server_id: str | None = None,
) -> Any:
    """Score the step's recorded answer on `raw` (`tuningfork.backends.raw`) and store it."""
    step = checkpoint.step
    message, finish = recorded_message(store, step)
    if step.error and recorded_outcome(store.response(step), step.error).is_error:
        raise ScoringError(f"step {step.idx} has no answer to score: {step.error}")
    if not message:
        raise ScoringError(f"step {step.idx} has no answer to score")

    fmt = format_for(await raw.chat_template())
    if fmt is None:
        raise ScoringError(
            "the server's chat template is not one tuningfork can read raw output of; it reads the "
            f"output of {known_models()} (backends/formats)"
        )
    prompt = await raw.render(checkpoint.request, checkpoint.wire.name)
    recorded_prompt = (step.usage or {}).get("prompt_tokens")
    if recorded_prompt is not None and recorded_prompt != len(prompt.ids):
        raise ScoringError(
            f"the prompt renders to {len(prompt.ids)} tokens here, and the step's had "
            f"{recorded_prompt}: the model would be asked about a different context. The server's "
            "chat template or settings may have changed since the step was recorded."
        )
    thinking = fmt.thinking(prompt.text)
    try:
        texts = fmt.renderings(message, thinking=thinking, offered=prompt.offered)
    except FormatError as error:
        raise ScoringError(f"the recorded answer cannot be written back as raw text: {error}") from None
    recorded_answer = (step.usage or {}).get("completion_tokens")
    ended = finish in ("stop", "tool_calls")
    # Where the server reads several texts as this answer, those with the recorded number of
    # tokens — every one, where the number is unknown, and the first where none has it. Of
    # those, the one the model finds likelier where they part: a blank line between two calls
    # is one token, as a newline is. The texts agree up to there, so one scoring weighs both.
    tokenized = [(candidate, await raw.pieces(candidate)) for candidate in texts]
    alike = [
        (candidate, found) for candidate, found in tokenized
        if recorded_answer is None or len(found) + ended == recorded_answer
    ] or tokenized[:1]
    text, pieces = alike[0]
    entries = await raw.score(prompt, [token for token, _ in pieces], top_k)
    for candidate, found in alike[1:]:
        ours, theirs = [token for token, _ in pieces], [token for token, _ in found]
        at = _parting(ours, theirs)
        if at is not None and _chance(entries[at], theirs[at]) > _chance(entries[at], ours[at]):
            text, pieces = candidate, found
            entries = await raw.score(prompt, theirs, top_k)
    answer = [token for token, _ in pieces]

    stream = from_entries(entries, top_k=top_k)
    for token, piece in pieces:
        if not stream.pieces.get(token):
            stream.pieces[token] = piece  # the per-token method may not have seen it among the top k
    stream.mark_parts(fmt, thinking=thinking)

    rebuilt = len(answer) + (1 if ended else 0)
    exact = None if recorded_answer is None else rebuilt == recorded_answer
    try:
        parsed = fmt.parse(text, tools=prompt.tools, parallel=prompt.parallel, thinking=thinking, offered=prompt.offered)
        agrees = (
            outcome_for({"choices": [{"message": parsed, "finish_reason": finish}]}).key
            == outcome_for(store.response(step)).key
        )
    except FormatError:
        agrees = False
    notes = {
        "format": fmt.name,
        "prompt_tokens": len(prompt.ids),
        "recorded_prompt_tokens": recorded_prompt,
        "answer_tokens": rebuilt,
        "recorded_answer_tokens": recorded_answer,
        "thinking": thinking,
        "parse_agrees": agrees,
    }
    if recorded_answer is not None and len(alike) > 1:
        notes["texts_alike"] = len(alike)  # with the recorded count; the likelier kept
    return store.add_token_stream(
        redact_stream(stream).to_json(),
        source="scored",
        step_id=step.id,
        server_id=server_id,
        method=raw.method,
        exact=exact,
        notes=notes,
    )
