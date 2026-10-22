"""The raw path: what token-level work needs of a server beyond chat completions.

Three things. Rendering a recorded request into the exact prompt the chat endpoint gave the
model, as token ids, without generating. Scoring an answer: the model's distribution at every
token of it, with the answer forced. And continuing from token ids — a prompt plus a forced
prefix — which is how a forking-token probe forces one token and lets the model go on.

Token ids, not text: a forced prefix sent back as text would be tokenized again, and not
always into the tokens the model chose. llama.cpp's server does all three
(`tuningfork.backends.llamacpp.raw`). A server that cannot is refused with the reason, never
guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RawError(RuntimeError):
    """The server could not do what the raw path asked, or answered in a way it cannot use."""


@dataclass(slots=True)
class Prompt:
    """A recorded request as the model saw it: the rendered prompt and its token ids."""

    ids: list[int]
    text: str
    #: Whether the request offered tools, and allowed more than one call at once: how the
    #: server parses what comes back.
    tools: bool
    parallel: bool
    #: The request's own sampling settings; the server's defaults apply to the rest.
    sampling: dict[str, Any] = field(default_factory=dict)
    #: Stop strings the request set.
    stop: list[str] = field(default_factory=list)
    #: The tools offered, as chat-completions definitions, while `tools`: a format that types
    #: each argument by its schema reads an answer with them, as the server did.
    offered: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Generation:
    """What came back from a prompt and a forced prefix: the generated ids after the prefix,
    their text, and how generation stopped (eos | limit | word)."""

    ids: list[int]
    text: str
    stop: str
    #: Per-token entries (`tuningfork.analysis.tokens.from_entries`), when asked for.
    entries: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)


async def raw_server(client: Any, base_url: str, *, send: Any = None) -> RawServer:
    """The raw path on the server at `base_url`, or `RawError` saying why there is none.

    Only llama.cpp's server has one today; it is told by its `/props`. Everything else is
    measured as it is used: a missing endpoint is refused with its name.
    """
    from tuningfork.backends.llamacpp import fetch_props
    from tuningfork.backends.llamacpp.raw import LlamaRaw

    props = await fetch_props(client, base_url)
    if props is None:
        raise RawError(
            "the server has no /props: token-level work needs llama.cpp's server — "
            "/apply-template, /tokenize and /completion — and this one does not look like it"
        )
    return LlamaRaw(client, base_url, send=send, props=props)


class RawServer(Protocol):
    #: How the last `score` forced the answer: "grammar", or "per-token" requests.
    method: str

    async def chat_template(self) -> str | None:
        """The chat template the server applies, for telling the model's output format."""

    async def render(self, request: dict[str, Any], wire: str) -> Prompt:
        """A recorded request, in its own wire format, as the prompt the model was given."""

    async def tokenize(self, text: str) -> list[int]:
        """Text as token ids, special tokens parsed, no beginning-of-text token added."""

    async def pieces(self, text: str) -> list[tuple[int, Any]]:
        """Text as (id, piece) pairs, the same tokenization as `tokenize`."""

    async def score(self, prompt: Prompt, answer: list[int], top_k: int) -> list[dict[str, Any]]:
        """The model's top-k at every position of `answer`, forced after `prompt`, and at the
        position after it — where the model's own end token shows. Per-token entries."""

    async def complete(
        self,
        prompt: Prompt,
        prefix: list[int],
        *,
        n_predict: int,
        seed: int,
        n_probs: int = 1,
    ) -> Generation:
        """Generate from the prompt plus a forced prefix, at the request's own sampling."""
