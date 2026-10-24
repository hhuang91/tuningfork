"""Turn observed traffic into a durable record.

Writes into the SQLite message tree. Redaction happens here, on the way to disk, and never
on forwarded traffic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from tuningfork.analysis.tokens import from_entries, redact_stream, split_logprobs
from tuningfork.config import Settings, patterns_digest, redact_json, redact_json_counted, server_root
from tuningfork.store.db import Step, Store
from tuningfork.wire import DEFAULT as DEFAULT_WIRE
from tuningfork.wire import WireFormat

# Non-message request fields worth surfacing without opening a blob. Deliberately a superset
# of the OpenAI spec: local servers accept vendor sampling knobs such as `top_k` and `min_p`,
# and an experiment that ignored them would not reproduce the original run.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    # llama.cpp's name, and the one OpenAI-compatible servers such as vLLM use.
    "repeat_penalty",
    "repetition_penalty",
    "logprobs",
    "top_logprobs",
    "response_format",
    "reasoning_effort",
    "stream",
)

#: Server-side defaults worth capturing. Phase 0 found that Hermes sends *no* sampling
#: parameters at all, so without these a checkpoint cannot be reproduced: the resample
#: would silently run under whatever the server happened to default to.
SERVER_DEFAULT_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "seed",
    "n_ctx",
    "n_predict",
    "samplers",
)


@dataclass(slots=True)
class Timings:
    started: float = field(default_factory=time.monotonic)
    first_byte: float | None = None
    completed: float | None = None
    #: When the call reached the proxy, by the wall clock: how a launch's threads are told
    #: apart afterwards (see `Store.finalize_threads`).
    arrived: float = field(default_factory=time.time)

    def mark_first_byte(self) -> None:
        if self.first_byte is None:
            self.first_byte = time.monotonic()

    def finish(self) -> None:
        self.completed = time.monotonic()

    def as_dict(self) -> dict[str, float | None]:
        def ms(value: float | None) -> float | None:
            return None if value is None else round((value - self.started) * 1000, 2)

        return {
            "ttft_ms": ms(self.first_byte),
            "total_ms": ms(self.completed),
            "arrived_at": round(self.arrived, 3),
        }


def summarize_params(payload: dict[str, Any]) -> dict[str, Any]:
    """The sampling parameters actually sent, so a resample can reproduce them."""
    params = {key: payload[key] for key in SAMPLING_KEYS if key in payload}
    if isinstance(payload.get("tools"), list):
        params["tool_count"] = len(payload["tools"])
    if isinstance(payload.get("messages"), list):
        params["message_count"] = len(payload["messages"])
    return params


def defaults_in(props: dict[str, Any]) -> dict[str, Any]:
    """The generation defaults in a `/props` body."""
    settings = props.get("default_generation_settings")
    source = settings if isinstance(settings, dict) else props
    defaults = {key: source[key] for key in SERVER_DEFAULT_KEYS if key in source}
    if isinstance(source.get("params"), dict):
        params = source["params"]
        defaults.update({k: params[k] for k in SERVER_DEFAULT_KEYS if k in params})
    return defaults


async def _props(client: httpx.AsyncClient, base_url: str, timeout: Any = 15) -> dict[str, Any] | None:
    try:
        response = await client.get(f"{server_root(base_url)}/props", timeout=timeout)
        if response.status_code != 200:
            return None
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return body if isinstance(body, dict) else None


async def fetch_server_defaults(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Read the server's own generation defaults, if it exposes them.

    llama.cpp publishes them at `/props`. Servers that do not are recorded as empty, which
    is honest: it means a resample cannot claim to reproduce the original sampling.
    """
    body = await _props(client, base_url)
    return defaults_in(body) if body else {}


async def fetch_server_identity(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """What the server behind the proxy is: enough to tell two servers apart when they
    report the same model name — which LM Studio and llama-server both do for one GGUF,
    with chat templates that need not agree.

    A whitelist of what `/props` says (build, model file, context size, slots, a hash of
    the chat template, the sampling defaults) and the model ids `/v1/models` lists. What a
    server does not say is left out; an empty identity is still an identity: "unknown".
    """
    identity, _ = await read_server(client, base_url)
    return identity


#: What an identity holds only when something answered as an inference server.
ANSWERED = ("build", "models", "n_ctx", "model_file")

#: How long a server gets to accept a connection while it is asked what it is: one that is
#: up accepts at once. Found on WSL's mirrored networking, where a port nothing listens on
#: does not refuse a connection but lets it wait: two 15-second waits made a server that was
#: down read, half a minute later, as one that had said nothing of itself.
CONNECT_TIMEOUT = 3.0
_READ = httpx.Timeout(15.0, connect=CONNECT_TIMEOUT)


#: Answers that say a server wants a key tuningfork does not send: a hosted model API's, most often.
ASKS_KEY = (401, 403)


@dataclass(slots=True)
class ServerRead:
    """One read of the upstream: its identity, why it said nothing of itself when it did not,
    and two things the identity leaves out — whether it refused tuningfork for want of an API
    key, and whether llama.cpp's `/props` answered (None where nothing answered at all, or the
    server was not ready: a 5xx is no answer to that)."""

    identity: dict[str, Any]
    problem: str | None = None
    asks_key: bool = False
    props: bool | None = None

    def facts(self) -> dict[str, Any]:
        return {"asks_key": self.asks_key, "props": self.props}


async def read_server(client: httpx.AsyncClient, base_url: str) -> tuple[dict[str, Any], str | None]:
    """The server's identity (see `fetch_server_identity`), and — when it said nothing of
    itself — why, in words: nothing listening there, nothing accepting a connection, or its
    own answer (llama.cpp answers 503 "Loading model" while it loads)."""
    read = await read_upstream(client, base_url)
    return read.identity, read.problem


async def read_upstream(client: httpx.AsyncClient, base_url: str) -> ServerRead:
    """`read_server`, with what the read found beyond the identity (`ServerRead`)."""
    import hashlib
    from pathlib import PurePosixPath

    root = server_root(base_url)
    identity: dict[str, Any] = {"upstream": root}
    problem: str | None = None
    try:
        response = await client.get(f"{root}/v1/models", timeout=_READ)
    except httpx.ConnectTimeout:
        return ServerRead(identity, f"nothing accepted a connection there within {CONNECT_TIMEOUT:g} seconds")
    except httpx.ConnectError:
        return ServerRead(identity, "nothing is listening there")
    except httpx.HTTPError as exc:
        response, problem = None, f"it did not answer ({type(exc).__name__})"
    asks_key = response is not None and response.status_code in ASKS_KEY
    if response is not None and response.status_code == 200:
        try:
            listed = response.json().get("data")
        except (ValueError, AttributeError):
            listed = None
        if isinstance(listed, list):
            identity["models"] = sorted(
                str(entry.get("id")) for entry in listed if isinstance(entry, dict) and entry.get("id")
            )
    elif response is not None:
        problem = _answered(response)
    body, status = await _read_props(client, root)
    # Told no — not found, or refused without a key — is a server with no /props; a 5xx, or
    # no answer, says nothing either way (llama.cpp answers 503 while it loads its model).
    props = True if body else False if status in (*ASKS_KEY, 404, 405) else None
    asks_key = asks_key or status in ASKS_KEY
    if body:
        settings = body.get("default_generation_settings")
        settings = settings if isinstance(settings, dict) else {}
        if body.get("build_info"):
            identity["build"] = body["build_info"]
        if body.get("model_path"):
            identity["model_file"] = PurePosixPath(str(body["model_path"]).replace("\\", "/")).name
        n_ctx = settings.get("n_ctx", body.get("n_ctx"))
        if n_ctx is not None:
            identity["n_ctx"] = n_ctx
        if body.get("total_slots") is not None:
            identity["slots"] = body["total_slots"]
        if isinstance(body.get("chat_template"), str):
            identity["chat_template_sha256"] = hashlib.sha256(
                body["chat_template"].encode("utf-8")
            ).hexdigest()[:16]
        defaults = defaults_in(body)
        if defaults:
            identity["defaults"] = defaults
    if any(key in identity for key in ANSWERED):
        return ServerRead(identity, None, asks_key, props)
    return ServerRead(identity, problem or "it answered, but said nothing of what it is", asks_key, props)


async def _read_props(client: httpx.AsyncClient, root: str) -> tuple[dict[str, Any] | None, int | None]:
    """llama.cpp's `/props`, and the status it was answered with (None: no answer)."""
    try:
        response = await client.get(f"{root}/props", timeout=_READ)
    except httpx.HTTPError:
        return None, None
    if response.status_code != 200:
        return None, response.status_code
    try:
        body = response.json()
    except ValueError:
        return None, response.status_code
    return (body if isinstance(body, dict) else None), response.status_code


def stream_without_secrets(
    body: bytes,
    answer: dict[str, Any] | None,
    wire: WireFormat,
    request: dict[str, Any],
    patterns: tuple[str, ...],
    *,
    masked: dict[str, Any] | None = None,
) -> tuple[bytes | None, str | None]:
    """The bytes to keep of a streamed answer, and how they were redacted beyond the text, if
    they were: "rendered" or "dropped" (a step's `raw_redacted`).

    `body` is the stream redacted as text, and `answer` what the stream read as before that, in
    the common shape, logprobs apart; `masked`, that answer redacted, where it already is.

    A model's answer streams about a token an event, so a key it writes arrives in pieces that no
    pattern sees whole, or masked only as far as one event holds it, and a fork reading the bytes
    back would give it again. So the bytes are kept as they came only where they read as the
    masked answer — every answer without a secret, and one whose secrets each sat whole in an
    event, masked there — and carry no logprobs, whose tokens spell a secret too. Otherwise they
    are the answer written again (`wire.render`) from its masked form; the step says so, and a
    fork refuses to replay it. That drops what no reader of the bytes needs — logprobs, a cut-off
    end or an error event (the step keeps its error), usage the request did not ask for, a
    thinking block's signature, the server's timings. Where the masked answer cannot be written
    again, no bytes are kept: losing them beats storing the secret. The check sees what the
    common shape keeps, not a field it leaves out. An answer the proxy could not read at all
    keeps its bytes as the text left them: its step has an error, which scoring and a fork
    refuse first.
    """
    if not answer:
        return body, None
    if masked is None:
        masked = redact_json(answer, patterns)
    if masked == answer:
        return body, None
    native: dict[str, Any] | None = None
    try:
        native = wire.parse(body, True)
        kept = wire.canonical_response(native, stream=body) if native else None
    except Exception:  # noqa: BLE001 - bytes that cannot be read back are written again
        kept = None
    if isinstance(kept, dict):
        spelled = any(isinstance(choice, dict) and "logprobs" in choice for choice in kept.get("choices") or [])
        if split_logprobs(kept)[0] == masked and not spelled:
            return body, None
    if "tuningfork_error" in masked:
        return None, "dropped"  # masked, the answer no longer reads: written again, it would be an empty one
    try:
        return wire.render(wire.native_response(masked, native), request={**request, "stream": True}), "rendered"
    except Exception:  # noqa: BLE001 - see above
        return None, "dropped"


def _answered(response: httpx.Response) -> str:
    """A refusal as the server put it: its status, and its own message if it gave one."""
    message = ""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        message = str((error.get("message") if isinstance(error, dict) else error) or body.get("message") or "")
    except (ValueError, AttributeError):
        message = response.text[:200]
    return f"it answered {response.status_code}" + (f": {message}" if message else "")


class Recorder:
    def __init__(self, settings: Settings, store: Store | None = None) -> None:
        self.settings = settings
        self.store = store or Store(settings.home)
        #: The server's generation defaults; see `fetch_server_defaults`.
        self.server_defaults: dict[str, Any] = {}
        #: Which server is answering, as its identity's id (see `fetch_server_identity`);
        #: kept current by the proxy, and recorded with every step.
        self.server_id: str | None = None
        #: Set by the proxy, so each recorded step reaches the UI as it happens.
        self.bus: Any = None
        #: Steps are stored in the order recording began, which is the order the responses
        #: finished in. Each recording is a task of its own, with two hops to a thread; left
        #: to race, a slow blob write let the next call's step reach the store first, and a
        #: call outside any launch — placed by the message tree as it is stored — then
        #: started a run of its own.
        self._in_order = asyncio.Lock()

    @property
    def blobs(self):
        return self.store.blobs

    async def record(
        self,
        *,
        method: str,
        path: str,
        request_body: bytes,
        request_payload: dict[str, Any] | None,
        status: int,
        timings: Timings,
        response_json: dict[str, Any] | None = None,
        raw_response: bytes | None = None,
        streamed: bool = False,
        error: str | None = None,
        harness: str | None = None,
        harness_request: dict[str, Any] | None = None,
        delivered_response: dict[str, Any] | None = None,
        edits: dict[str, Any] | None = None,
        wire: WireFormat | None = None,
        assignment: Any = None,
        replay: dict[str, Any] | None = None,
        substitution: dict[str, Any] | None = None,
        workspace_commit: str | None = None,
        served_by: str | None = None,
        context: dict[str, Any] | None = None,
        holds: list[str] | None = None,
        output_cap: dict[str, Any] | None = None,
    ) -> Step | None:
        """Store one exchange. The edit arguments are only set when a breakpoint changed it.

        Responses arrive in the call's own wire format and are stored in the common shape;
        the raw bytes are kept too. `assignment` places a launched call where it was
        assigned when it arrived (see `tuningfork.launches`). `output_cap` is what the proxy's
        cap changed in the request (`wire.base.cap_output`), kept with its parameters: the
        request is stored as the model was sent it.
        """
        if not request_payload:
            return None
        wire = wire or DEFAULT_WIRE
        if response_json:
            response_json = wire.canonical_response(response_json, stream=raw_response if streamed else None)
        if delivered_response is not None:
            delivered_response = wire.canonical_response(delivered_response)
        # A harness that asked for logprobs gets them; they are stored as the step's tokens,
        # redacted token-aware, never through JSON redaction (see `split_logprobs`).
        response_json, token_entries = split_logprobs(response_json)
        delivered_response, _ = split_logprobs(delivered_response)

        patterns = self.settings.redaction_patterns

        def clean(value: dict[str, Any] | None) -> dict[str, Any] | None:
            return redact_json(value, patterns) if value is not None else None

        params = summarize_params(request_payload)
        params["streamed"] = streamed
        if output_cap:
            params["output_cap"] = output_cap

        def record_step(raw_blob: str | None, response: dict[str, Any] | None) -> Any:
            # Redacted in the worker thread, as the step is stored: redaction reads every
            # payload whole, its JSON escapes read too, and the event loop has calls to serve.
            # The step counts its masks — a request can read `<redacted>` where nothing was masked —
            # and names the patterns that made them, which the store keeps: a fork masks a call
            # with them again, to match a step masked otherwise than the patterns it has.
            request, params["masked"] = redact_json_counted(request_payload, patterns)
            params["redaction"] = patterns_digest(patterns)
            self.store.keep_masking(params["redaction"], patterns)
            return self.store.record_step(
                request=request,
                response=response,
                raw_response_blob=raw_blob,
                params=params,
                server_defaults=self.server_defaults,
                status=status,
                timings=timings.as_dict(),
                error=error,
                harness=harness,
                harness_request=clean(harness_request),
                delivered_response=clean(delivered_response),
                # An edit's record holds the harness's own fields (`original_extras`), which a
                # fork rebuilds the harness's request from: masked as that request is.
                edits=clean(edits),
                wire=wire.name,
                normalize=wire.normalize_message,
                assignment=assignment,
                replay=replay,
                substitution=substitution,
                context=context,
                workspace_commit=workspace_commit,
                # A fork's replayed answer came from the server its source was recorded on.
                server_id=served_by or self.server_id,
            )

        async with self._in_order:
            # The answer masked once, off the loop: what is stored, and what its bytes must read as.
            response = await asyncio.to_thread(clean, response_json) if response_json else None
            raw_blob = None
            if raw_response is not None:
                raw_blob, raw_redacted = await asyncio.to_thread(
                    self._store_raw, raw_response, wire, streamed, request_payload, response_json, response
                )
                if raw_redacted:
                    # Its bytes are not the server's: scoring keeps to the answer, a fork refuses it.
                    params["raw_redacted"] = raw_redacted
            step = await asyncio.to_thread(record_step, raw_blob, response)
            if holds and step is not None:
                # What was sampled at these holds is filed under this step from now on.
                await asyncio.to_thread(
                    self.store.link_holds, holds, step, edited=bool((edits or {}).get("request"))
                )
            if token_entries and step is not None:
                await asyncio.to_thread(self._store_tokens, step, token_entries)
        if self.bus is not None and step is not None:
            self.bus.publish(
                "step", run_id=step.run_id, idx=step.idx, edited=step.edited, error=step.error
            )
        return step

    def _store_tokens(self, step: Step, entries: list[dict[str, Any]]) -> None:
        stream = redact_stream(from_entries(entries), self.settings.redaction_patterns)
        self.store.add_token_stream(
            stream.to_json(), source="recorded", step_id=step.id, server_id=step.server_id,
            method="chat-logprobs",
        )

    def _store_raw(
        self,
        raw: bytes,
        wire: WireFormat,
        streamed: bool,
        request: dict[str, Any],
        answer: dict[str, Any] | None,
        masked: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        """The raw bytes kept, and how they were redacted beyond the text (`stream_without_secrets`).
        Bytes that are not UTF-8 are redacted as text too, and otherwise kept as they came."""
        from tuningfork.config import redact

        patterns = self.settings.redaction_patterns
        body: bytes | None = redact(raw.decode("utf-8", "surrogateescape"), patterns).encode("utf-8", "surrogateescape")
        how = None
        if streamed:
            body, how = stream_without_secrets(body, answer, wire, request, patterns, masked=masked)
        return (self.store.blobs.put(body) if body is not None else None), how
