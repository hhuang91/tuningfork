"""The output-format protocol; see `tuningfork.backends.formats`."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class FormatError(ValueError):
    """Raw text that the format cannot read: a malformed tool call, say."""


class Format(Protocol):
    name: str
    #: What the model writes between a call's opening marker and the tool's name — Gemma 4's
    #: `call:` — so the first token of the name can be found in a stream's tool-call part.
    call_name_prefix: str

    def detect(self, template: str) -> bool:
        """Whether a chat template is this format's."""

    def thinking(self, prompt: str) -> bool:
        """Whether a rendered prompt turns the model's reasoning on."""

    def parse(self, text: str, *, tools: bool) -> dict[str, Any]:
        """Raw generated text as a common-shape assistant message: content, reasoning_content
        and tool_calls, as the server's own parser gives them. Raises `FormatError`."""

    def parts(self, pieces: Sequence[str]) -> list[tuple[str, int, int]]:
        """What each token is — marker, reasoning, content or tool_call — as (kind, start,
        end) runs over positions, from the tokens' pieces."""

    def render(self, message: dict[str, Any], *, thinking: bool) -> str:
        """A parsed assistant message as the raw text the model generated: the inverse of
        `parse`. Without the end token, which the model adds itself."""
