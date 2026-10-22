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
"""

from __future__ import annotations

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


def recorded_message(store: Store, step: Any) -> tuple[dict[str, Any], str | None]:
    response = store.response(step) or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    return choice.get("message") or {}, choice.get("finish_reason")


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
            "the server's chat template is not one tuningfork can read raw output of; it reads "
            f"{known_models()}'s (backends/formats)"
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
        text = fmt.render(message, thinking=thinking)
    except FormatError as error:
        raise ScoringError(f"the recorded answer cannot be written back as raw text: {error}") from None
    pieces = await raw.pieces(text)
    answer = [token for token, _ in pieces]
    entries = await raw.score(prompt, answer, top_k)

    stream = from_entries(entries, top_k=top_k)
    for token, piece in pieces:
        if not stream.pieces.get(token):
            stream.pieces[token] = piece  # the per-token method may not have seen it among the top k
    stream.mark_parts(fmt, thinking=thinking)

    recorded_answer = (step.usage or {}).get("completion_tokens")
    ended = finish in ("stop", "tool_calls")
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
    return store.add_token_stream(
        redact_stream(stream).to_json(),
        source="scored",
        step_id=step.id,
        server_id=server_id,
        method=raw.method,
        exact=exact,
        notes=notes,
    )
