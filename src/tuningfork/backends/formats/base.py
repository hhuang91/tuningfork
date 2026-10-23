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

    def thinking(self, prompt: str) -> bool | str:
        """What a rendered prompt did with the model's reasoning: True where it turned it on,
        or opened it, so that the answer starts inside it; False where it turned it off, or
        closed it; `"model"` where it left it to the model, which may open one itself."""

    def parse(
        self,
        text: str,
        *,
        tools: bool,
        parallel: bool = True,
        thinking: bool | str | None = None,
        offered: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Raw generated text as a common-shape assistant message: content, reasoning_content
        and tool_calls, as the server's own parser gives them. Raises `FormatError`.

        `thinking` is `thinking(prompt)` where the prompt is known: a template that opens the
        reasoning in the prompt itself leaves the answer starting inside it. None, where the
        prompt is not known: read from the text, where it can be. `offered` is the request's
        tools, as chat-completions definitions, for a format that types each argument by its
        schema."""

    def parts(self, pieces: Sequence[str], *, thinking: bool | str | None = None) -> list[tuple[str, int, int]]:
        """What each token is — marker, reasoning, content or tool_call — as (kind, start,
        end) runs over positions, from the tokens' pieces. `thinking` as for `parse`."""

    def render(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> str:
        """A parsed assistant message as the raw text the model generated: the inverse of
        `parse`, given what `parse` is given. Without the end token, which the model adds
        itself."""

    def renderings(
        self, message: dict[str, Any], *, thinking: bool | str | None, offered: Sequence[dict[str, Any]] = ()
    ) -> list[str]:
        """Every text the model may have generated for this message where the server reads
        several the same, `render`'s first. Scoring keeps the one with the recorded number of
        tokens, and where several have it, the one the model finds likelier where they part."""


def token_parts(pieces: Sequence[str], kinds: Sequence[str]) -> list[tuple[str, int, int]]:
    """`parts` from what each character of the joined pieces is (`kinds`; any past its end are
    markers): a token takes the part of its first character that is not a marker's, and is a
    marker where all are."""
    runs: list[tuple[str, int, int]] = []
    at = 0
    for position, piece in enumerate(pieces):
        own = kinds[at : at + len(piece)]
        at += len(piece)
        kind = next((each for each in own if each != "marker"), "marker")
        if runs and runs[-1][0] == kind and runs[-1][2] == position:
            runs[-1] = (kind, runs[-1][1], position + 1)
        else:
            runs.append((kind, position, position + 1))
    return runs
