"""The raw path on llama.cpp's server (`tuningfork.backends.raw`), as measured on b11108.

- **render**: `/apply-template` with the step's own chat body — a Messages step converted the
  way the server converts it (`anthropic.to_chat`) — then `/tokenize` with the beginning-of-text
  token added, as the chat endpoint tokenizes. The ids equal the prompt tokens the chat call
  reported, and a greedy completion from them repeats the greedy chat answer token for token.
- **score**: one `/completion` with a grammar of the answer's ids, `root ::= <[id]> <[id]> …`.
  Every id is forced; the probabilities reported are before sampling, so the grammar does not
  bend them. The answer's end token is left out: once the grammar is done only an end token
  may follow, and the model's own shows there. If the grammar is refused, one request per
  token instead — slower (about 170 ms a token against 20), with the same numbers.
- **complete**: `/completion` from the prompt plus a forced prefix, at the request's own
  sampling settings; the server's defaults apply to the rest, as they did to the chat call.

Every request processes its whole prompt (`cache_prompt: false`): what the server cached of
an earlier one changed the numbers. Scored right after other work, a step's write gave its tool
`Bash` 76% and `Write` 23% from 14,061 cached tokens; processed whole, `Write` 100% — as its
twenty samples did. A request's own setting overrides the server's.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import httpx

from tuningfork.backends.llamacpp import fetch_props
from tuningfork.backends.raw import Generation, Prompt, RawError
from tuningfork.config import EXPERIMENT_HEADER, server_root

#: Request fields that set sampling and that `/completion` reads by the same name.
SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "typical_p", "presence_penalty",
    "frequency_penalty", "repeat_penalty",
)
#: Chat-request fields that shape only the answer, never the prompt.
_ANSWER_ONLY = ("stream", "stream_options", "logprobs", "top_logprobs", "n", "seed")

Send = Callable[[str, dict[str, Any]], Awaitable[httpx.Response]]


def chat_body(request: dict[str, Any], wire: str) -> dict[str, Any]:
    """The recorded request as the chat body the server renders its prompt from."""
    if wire == "anthropic-messages":
        from tuningfork.backends.llamacpp.anthropic import to_chat

        body = to_chat(request)
    else:
        body = dict(request)
    for key in _ANSWER_ONLY:
        body.pop(key, None)
    return body


def offered_tools(request: dict[str, Any], wire: str) -> list[dict[str, Any]]:
    """The tools a recorded request offered, as the chat-completions definitions the server
    read its answer with; none where it offered none, or said not to call one."""
    body = chat_body(request, wire)
    if body.get("tool_choice") == "none":
        return []
    return [tool for tool in body.get("tools") or [] if isinstance(tool, dict)]


def aligned(entries: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    """Per-token entries, one a token again. llama.cpp gives a character the model wrote in
    two tokens or more one entry: the last token's, holding the whole character's bytes (b11108:
    Qwen3.5 writes ` ≡` as two tokens). Each token it merged away gets an entry of its own,
    nothing known of it, and the merged entry loses the bytes, which are not its token's alone —
    its piece is read again from the tokenizer, where its caller has one. Entries that do not
    follow the tokens are refused, never laid over the wrong ones."""
    out: list[dict[str, Any]] = []
    k, skipped = 0, False
    for token in ids:
        if k < len(entries) and entries[k].get("id") == token:
            entry = entries[k]
            k += 1
            if skipped:
                entry = {**entry, "token": "", "bytes": []}
            out.append(entry)
            skipped = False
        else:
            out.append({"id": token, "token": "", "bytes": [], "logprob": None, "top_logprobs": []})
            skipped = True
    if k < len(entries):
        raise RawError(f"the server's probability entries do not follow its tokens, from entry {k}")
    return out


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or body)[:300]


class LlamaRaw:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        send: Send | None = None,
        props: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.client = client
        self.root = server_root(base_url)
        self.timeout = timeout
        #: How a model call is sent: by default tagged as an experiment's, so a proxy on the
        #: way gates it and records nothing. The experiment layer passes one that also waits
        #: out a paused proxy.
        self._send = send
        self._props = props
        self.method = "grammar"

    async def _post(self, path: str, body: dict[str, Any], *, model_call: bool = False) -> dict[str, Any]:
        url = f"{self.root}{path}"
        try:
            if model_call and self._send is not None:
                response = await self._send(url, body)
            else:
                headers = {EXPERIMENT_HEADER: "tokens"} if model_call else None
                response = await self.client.post(url, json=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as error:
            raise RawError(f"{path}: {type(error).__name__}: {error}") from None
        if response.status_code == 404:
            raise RawError(
                f"the server has no {path}: token-level work needs llama.cpp's server, which has "
                "/apply-template, /tokenize and /completion"
            )
        if response.status_code >= 400:
            raise RawError(f"{path} answered {response.status_code}: {_detail(response)}")
        try:
            data = response.json()
        except ValueError:
            raise RawError(f"{path} answered something that is not JSON") from None
        if not isinstance(data, dict):
            raise RawError(f"{path} answered a JSON {type(data).__name__}")
        return data

    async def props(self) -> dict[str, Any]:
        if self._props is None:
            self._props = await fetch_props(self.client, self.root) or {}
        return self._props

    async def chat_template(self) -> str | None:
        template = (await self.props()).get("chat_template")
        return template if isinstance(template, str) else None

    async def render(self, request: dict[str, Any], wire: str) -> Prompt:
        body = chat_body(request, wire)
        rendered = (await self._post("/apply-template", body)).get("prompt")
        if not isinstance(rendered, str):
            raise RawError("/apply-template answered without a prompt")
        ids = await self._tokenize(rendered, bos=True)
        caps = (await self.props()).get("chat_template_caps") or {}
        tools = bool(body.get("tools")) and body.get("tool_choice") != "none"
        stop = body.get("stop")
        return Prompt(
            ids=ids,
            text=rendered,
            tools=tools,
            parallel=bool(body.get("parallel_tool_calls", caps.get("supports_parallel_tool_calls", True))),
            sampling={key: body[key] for key in SAMPLING_FIELDS if body.get(key) is not None},
            stop=[stop] if isinstance(stop, str) else list(stop or []),
            offered=offered_tools(request, wire),
        )

    async def _tokenize(self, text: str, *, bos: bool, pieces: bool = False) -> list[Any]:
        data = await self._post(
            "/tokenize", {"content": text, "add_special": bos, "parse_special": True, "with_pieces": pieces}
        )
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise RawError("/tokenize answered without tokens")
        return tokens

    async def tokenize(self, text: str) -> list[int]:
        return [token if isinstance(token, int) else token["id"] for token in await self._tokenize(text, bos=False)]

    async def pieces(self, text: str) -> list[tuple[int, Any]]:
        """Text as (id, piece) pairs; a piece is text, or bytes as a list when not valid UTF-8."""
        return [(token["id"], token.get("piece", "")) for token in await self._tokenize(text, bos=False, pieces=True)]

    async def score(self, prompt: Prompt, answer: list[int], top_k: int) -> list[dict[str, Any]]:
        if not answer:
            raise RawError("there is no answer to score")
        body = {
            "prompt": prompt.ids,
            "n_predict": len(answer) + 1,
            "n_probs": top_k,
            "grammar": "root ::= " + " ".join(f"<[{token}]>" for token in answer),
            "temperature": 0,
            "return_tokens": True,
            "cache_prompt": False,
        }
        try:
            data = await self._post("/completion", body, model_call=True)
        except RawError as error:
            if "grammar" not in str(error).lower():
                raise
            return await self._score_per_token(prompt, answer, top_k)
        entries = data.get("completion_probabilities") or []
        tokens = data.get("tokens")
        if isinstance(tokens, list) and tokens:
            entries = aligned(entries, tokens)
        if [entry.get("id") for entry in entries[: len(answer)]] != answer:
            raise RawError("the server did not keep to the forced answer")
        self.method = "grammar"
        return entries[: len(answer) + 1]

    async def _score_per_token(self, prompt: Prompt, answer: list[int], top_k: int) -> list[dict[str, Any]]:
        """One request per position: the distribution after the prompt and the answer so far.
        The position's own token is the answer's, found among the top k when it is there."""
        entries = []
        for position in range(len(answer) + 1):
            data = await self._post(
                "/completion",
                {"prompt": prompt.ids + answer[:position], "n_predict": 1, "n_probs": top_k,
                 "temperature": 0, "cache_prompt": False},
                model_call=True,
            )
            found = (data.get("completion_probabilities") or [{}])[0]
            if position < len(answer):
                token = answer[position]
                known = {alt.get("id"): alt for alt in found.get("top_logprobs") or []}
                chosen = known.get(token) or {"id": token, "token": "", "bytes": [], "logprob": None}
                found = {**chosen, "top_logprobs": found.get("top_logprobs") or []}
            entries.append(found)
        self.method = "per-token"
        return entries

    async def complete(
        self,
        prompt: Prompt,
        prefix: list[int],
        *,
        n_predict: int,
        seed: int,
        n_probs: int = 1,
    ) -> Generation:
        body: dict[str, Any] = {
            **prompt.sampling,
            "prompt": prompt.ids + prefix,
            "n_predict": n_predict,
            "seed": seed,
            "n_probs": n_probs,
            "return_tokens": True,
            "cache_prompt": False,
        }
        if prompt.stop:
            body["stop"] = prompt.stop
        data = await self._post("/completion", body, model_call=True)
        ids = list(data.get("tokens") or [])
        entries = list(data.get("completion_probabilities") or [])
        if ids and entries:
            try:
                entries = aligned(entries, ids)
            except RawError:
                pass  # kept for the continuation's stream only, which shows them as they came
        return Generation(
            ids=ids,
            text=data.get("content") or "",
            stop=data.get("stop_type") or "none",
            entries=entries,
            timings=data.get("timings") or {},
        )
