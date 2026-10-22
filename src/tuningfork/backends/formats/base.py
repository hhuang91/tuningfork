"""The output-format protocol; see `tuningfork.backends.formats`."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class FormatError(ValueError):
    """Raw text that the format cannot read: a malformed tool call, say."""


class Format(Protocol):
    name: str
    #: The models that write this format, as a sentence names them: ("Gemma 4",).
    models: tuple[str, ...]
    #: What the model writes between a call's opening marker and the tool's name — Gemma 4's
    #: `call:` — so the first token of the name can be found in a stream's tool-call part.
    call_name_prefix: str

    def detect(self, template: str) -> bool:
        """Whether a chat template is this format's."""

    def thinking(self, prompt: str) -> bool:
        """Whether a rendered prompt turns the model's reasoning on."""

    def parse(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool = True,
        thinking: bool | None = None,
        offered: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Raw generated text as a common-shape assistant message: content, reasoning_content
        and tool_calls, as the server's own parser gives them. Raises `FormatError`.

        `thinking` is `thinking(prompt)` where the prompt is known: a template that opens the
        reasoning in the prompt itself leaves the answer starting inside it. None: read from
        the text, where it can be. `offered` is the request's tools, as chat-completions
        definitions, for a format that types each argument by its schema."""

    def parts(self, pieces: Sequence[str], *, thinking: bool | None = None) -> list[tuple[str, int, int]]:
        """What each token is — marker, reasoning, content or tool_call — as (kind, start,
        end) runs over positions, from the tokens' pieces. `thinking` as for `parse`."""

    def render(self, message: dict[str, Any], *, thinking: bool, offered: Sequence[dict[str, Any]] = ()) -> str:
        """A parsed assistant message as the raw text the model generated: the inverse of
        `parse`, given what `parse` is given. Without the end token, which the model adds
        itself."""

    def renderings(
        self, message: dict[str, Any], *, thinking: bool, offered: Sequence[dict[str, Any]] = ()
    ) -> list[str]:
        """Every text the model may have generated for this message where the server reads
        several the same, the likeliest — `render`'s — first. Scoring keeps the one with the
        recorded number of tokens, and where several have it, the one the model finds
        likeliest."""
