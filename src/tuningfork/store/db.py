"""SQLite-backed recording store.

Replaces the Phase 0 JSONL log. Messages live in the content-addressed tree, so the large
constant part of an agent's context — a 13k-token system prompt plus tool schemas — is
stored once no matter how many steps or runs share it. Only the per-step *extras* (tools,
sampling parameters) and the response are stored per step.

A request is therefore never stored whole. It is rebuilt from its node path plus the extras
blob, and `rebuild_request` is asserted to be JSON-equal to the original.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from tuningfork.store.blobs import BlobStore
from tuningfork.store.tree import exact_hash, node_id, node_path, normal_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id           TEXT PRIMARY KEY,
    parent_id    TEXT,
    exact_hash   TEXT NOT NULL,
    normal_hash  TEXT NOT NULL,
    message_blob TEXT NOT NULL,
    depth        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    root_node_id     TEXT NOT NULL,
    head_node_id     TEXT NOT NULL,
    label            TEXT,
    task_id          TEXT,
    harness          TEXT,
    model            TEXT,
    forked_from_run  TEXT,
    forked_at_depth  INTEGER,
    skill_hashes     TEXT,
    parity           TEXT,
    started_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    result           TEXT
);
CREATE INDEX IF NOT EXISTS runs_head ON runs(head_node_id);
CREATE INDEX IF NOT EXISTS runs_root ON runs(root_node_id);

CREATE TABLE IF NOT EXISTS steps (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    request_node_id  TEXT NOT NULL,
    extras_blob      TEXT NOT NULL,
    response_blob    TEXT,
    raw_response_blob TEXT,
    params           TEXT,
    server_defaults  TEXT,
    model            TEXT,
    status           INTEGER,
    finish_reason    TEXT,
    timings          TEXT,
    usage            TEXT,
    reasoning_chars  INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    recorded_at      REAL NOT NULL,
    -- Set when a breakpoint changed this exchange: see Store.record_step.
    edits            TEXT,
    harness_request_node_id TEXT,
    delivered_response_blob TEXT,
    -- The wire format of the call (tuningfork.wire); NULL for steps recorded before formats.
    wire             TEXT,
    -- Messages whose exact form differs from the stable one in the tree: {index: message}.
    message_overrides TEXT,
    UNIQUE(run_id, idx)
);
CREATE INDEX IF NOT EXISTS steps_run ON steps(run_id, idx);

-- Labels queued by `tuningfork label-next`, consumed by the next run that appears.
-- `force` additionally declares a run boundary: see `begin_run`.
CREATE TABLE IF NOT EXISTS pending_labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT NOT NULL,
    task_id    TEXT,
    force      INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

-- One arm of sampling from a recorded checkpoint. A resample is one experiment; an A/B is
-- two, sharing a group_id. Experiments live apart from runs on purpose: a resampled request
-- has the same message path as the step it came from, and recorded as a run it would be
-- indistinguishable from the harness repeating that call.
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    group_id        TEXT,
    arm             TEXT NOT NULL,
    -- The step it samples. NULL while it samples a call held at a breakpoint that has not
    -- been recorded yet: `hold_steps` names the step once it has.
    run_id          TEXT,
    step_idx        INTEGER,
    step_id         TEXT,
    request_blob    TEXT NOT NULL,
    settings        TEXT,
    changes         TEXT,
    variant         TEXT,
    base_url        TEXT,
    server_defaults TEXT,
    notes           TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_checkpoint ON experiments(run_id, step_idx);
CREATE INDEX IF NOT EXISTS experiments_group ON experiments(group_id);

CREATE TABLE IF NOT EXISTS samples (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    idx             INTEGER NOT NULL,
    seed            INTEGER,
    response_blob   TEXT,
    outcome_kind    TEXT,
    outcome_key     TEXT,
    outcome_label   TEXT,
    finish_reason   TEXT,
    usage           TEXT,
    timings         TEXT,
    reasoning_chars INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      REAL NOT NULL,
    UNIQUE(experiment_id, idx)
);
CREATE INDEX IF NOT EXISTS samples_experiment ON samples(experiment_id, idx);

-- A harness process tuningfork started: a batch run, or a fork of one. Its id is the token
-- every call it makes carries (see tuningfork.launches), and what it was given to run is
-- kept here, so it can be launched again.
CREATE TABLE IF NOT EXISTS launches (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    label         TEXT,
    harness       TEXT,
    model         TEXT,
    task_id       TEXT,
    eval          TEXT,
    spec_path     TEXT,
    skill_dir     TEXT,
    workspace     TEXT,
    home          TEXT,
    settings      TEXT,
    source_run    TEXT,
    source_step   INTEGER,
    variant       TEXT,
    experiment_id TEXT,
    sample_idx    INTEGER,
    status        TEXT NOT NULL,
    exit          TEXT,
    created_at    REAL NOT NULL,
    ended_at      REAL
);

-- The servers that answered: what each said about itself (see
-- tuningfork.proxy.recorder.fetch_server_identity), by a short hash of that.
CREATE TABLE IF NOT EXISTS servers (
    id         TEXT PRIMARY KEY,
    identity   TEXT NOT NULL,
    first_seen REAL NOT NULL
);

-- Work a running proxy did for someone (see tuningfork.jobs): a resample, an A/B, a batch,
-- a fork. What it was asked, the plan it ran, how it ended, and what it made. Its progress
-- event by event lives in memory only; what it made is linked here, and stored as usual.
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    lane        TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT NOT NULL,
    plan        TEXT,
    progress    TEXT,
    result      TEXT,
    error       TEXT,
    links       TEXT,
    owner       TEXT,
    request_id  TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    ended_at    REAL
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS jobs_request ON jobs(request_id);

-- The step a call held at a breakpoint became, once it was recorded — so what was sampled
-- at the hold is filed under that step. `edited`: the model saw an edited request.
CREATE TABLE IF NOT EXISTS hold_steps (
    hold_id   TEXT PRIMARY KEY,
    step_id   TEXT NOT NULL,
    run_id    TEXT NOT NULL,
    step_idx  INTEGER NOT NULL,
    edited    INTEGER NOT NULL DEFAULT 0
);

-- An answer as the model's tokens, with the probabilities it weighed at each (see
-- tuningfork.analysis.tokens): a recorded step's answer scored afterwards, or a sample drawn
-- with logprobs, or a forced continuation. Derived data, and server-dependent: kept apart from
-- the responses, with the server that computed it. `exact`: the answer was rebuilt token for
-- token, as its recorded token count confirms.
CREATE TABLE IF NOT EXISTS token_streams (
    id          TEXT PRIMARY KEY,
    step_id     TEXT,
    sample_id   TEXT,
    server_id   TEXT,
    source      TEXT NOT NULL,
    method      TEXT,
    top_k       INTEGER,
    blob        TEXT NOT NULL,
    tokens      INTEGER NOT NULL,
    exact       INTEGER,
    notes       TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS token_streams_step ON token_streams(step_id, created_at);
CREATE INDEX IF NOT EXISTS token_streams_sample ON token_streams(sample_id);
"""

#: Roles of a launched run. `main` is a batch run's own conversation, and `fork` a fork's:
#: a replay of a recorded run, live from one of its steps, which belongs to an experiment
#: rather than beside the runs it replays. `subagent` and `side` are a launch's other
#: threads, and `thread` one not yet told apart (see `Store.finalize_threads`). A run
#: recorded outside any launch has no role.
MAIN, FORK, SUBAGENT, SIDE, THREAD = "main", "fork", "subagent", "side", "thread"


@dataclass(slots=True)
class Run:
    id: str
    root_node_id: str
    head_node_id: str
    label: str | None
    task_id: str | None
    harness: str | None
    model: str | None
    forked_from_run: str | None
    forked_at_depth: int | None
    skill_hashes: dict[str, str]
    parity: dict[str, Any]
    started_at: float
    updated_at: float
    step_count: int = 0
    #: How the run ended, when a batch ran it: the task check's verdict and whether the
    #: run may be counted in a statistic. Empty for a run recorded any other way.
    result: dict[str, Any] = field(default_factory=dict)
    #: The launch it came from, when tuningfork started the harness; see `Launch`.
    launch_id: str | None = None
    #: Its thread's role in that launch (`MAIN`, `SUBAGENT`, `SIDE`, `THREAD`).
    role: str | None = None
    #: For a subagent or side thread: the thread it came from, and that thread's step.
    parent_run_id: str | None = None
    spawned_at: int | None = None
    #: The server its first call was answered by (see `Store.server`); None if unrecorded.
    server_id: str | None = None
    #: When it was moved to the trash, and what it went with (see `Store.trash_run`).
    trashed_at: float | None = None
    trashed_with: str | None = None

    @property
    def is_thread(self) -> bool:
        """Whether this is a launch's secondary thread rather than a run in its own right."""
        return self.role in (SUBAGENT, SIDE, THREAD)

    @property
    def is_fork(self) -> bool:
        return self.role == FORK

    @property
    def skill_version(self) -> str | None:
        """A short fingerprint of the skill files the run was given; None if unrecorded.

        Two runs with the same fingerprint read the same skill, byte for byte. Runs that
        differ in it measure different wordings, and a rate pooled across them measures
        neither.
        """
        from tuningfork.specs import fingerprint

        return fingerprint(self.skill_hashes)


@dataclass(slots=True)
class Step:
    id: str
    run_id: str
    idx: int
    request_node_id: str
    extras_blob: str
    response_blob: str | None
    raw_response_blob: str | None
    params: dict[str, Any]
    server_defaults: dict[str, Any]
    model: str | None
    status: int | None
    finish_reason: str | None
    timings: dict[str, Any]
    usage: dict[str, Any]
    reasoning_chars: int
    error: str | None
    recorded_at: float
    #: What a breakpoint changed, if anything: {"request": {...}, "response": {...}}.
    edits: dict[str, Any] = field(default_factory=dict)
    #: The harness's own request, when it differs from what the model was shown.
    harness_request_node_id: str | None = None
    #: What the harness received, when it differs from what the model produced.
    delivered_response_blob: str | None = None
    #: The wire format the call used (`tuningfork.wire`); None means OpenAI chat.
    wire: str | None = None
    #: Exact forms of messages the tree stores normalized, by index; see `record_step`.
    message_overrides: dict[str, Any] = field(default_factory=dict)
    #: In a fork: the recorded step this one was answered from, or why a call that should
    #: have been answered from the recording was not — see `tuningfork.replay`.
    replay: dict[str, Any] = field(default_factory=dict)
    #: In a fork testing a wording: the skill text swapped into what the model saw. Not an
    #: edit — the run is still a sample — so kept apart from `edits`.
    substitution: dict[str, Any] = field(default_factory=dict)
    #: The workspace as the call arrived, in the home's snapshot store (`tuningfork.snapshots`).
    workspace_commit: str | None = None
    #: The server that answered, by the id of its identity (see `Store.server`).
    server_id: str | None = None
    #: In a scenario's run: the conversation it continues (`tuningfork.context`), which the
    #: proxy put before the harness's own messages in what the model saw. Not an edit — the run
    #: is still a sample — so kept apart from `edits`, as a variant is.
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def edited(self) -> bool:
        return bool(self.edits)

    @property
    def served(self) -> bool:
        """Answered from a recording, not by the model."""
        return bool(self.replay.get("served_from"))


@dataclass(slots=True)
class Launch:
    """A harness process tuningfork started, and what it was given to run."""

    id: str
    kind: str  # batch | fork
    label: str | None
    harness: str | None
    model: str | None
    task_id: str | None
    #: The eval as launched: prompt, input files, preloaded skills, turn limit, check.
    eval: dict[str, Any]
    spec_path: str | None
    #: The frozen copy of the skill folder the harness was given.
    skill_dir: str | None
    workspace: str | None
    home: str | None
    #: Launch-wide settings: whether breakpoints apply, and harness settings that change
    #: what is measured.
    settings: dict[str, Any]
    #: For a fork: the run it replays, and the step it goes live at.
    source_run: str | None
    source_step: int | None
    variant: dict[str, Any]
    experiment_id: str | None
    sample_idx: int | None
    status: str  # running | ended | interrupted
    exit: dict[str, Any]
    created_at: float
    ended_at: float | None
    #: Snapshots (see `tuningfork.snapshots`): the workspace before the harness started and
    #: after it ended, and the skill folder it was given.
    initial_commit: str | None = None
    final_commit: str | None = None
    skill_commit: str | None = None
    #: The job that started it, when one did (see tuningfork.jobs).
    job_id: str | None = None
    #: The process running it (`tuningfork.owners`): one still "running" after that process
    #: died was cut off, and never said so (`ops.inspect.launch_status`).
    owner: dict[str, Any] | None = None

    @property
    def breakpoints(self) -> bool:
        """Whether breakpoints may hold this launch's calls. Off for an experiment's forks,
        which a breakpoint armed for something else would otherwise hang."""
        return bool(self.settings.get("breakpoints", True))


@dataclass(slots=True)
class Experiment:
    id: str
    kind: str  # resample | ab
    group_id: str | None
    arm: str  # original | variant
    #: The step it samples; None for one sampled at a held call not recorded yet.
    run_id: str | None
    step_idx: int | None
    step_id: str | None
    #: The exact request sent, minus the per-sample seed.
    request_blob: str
    #: How it was sampled: n, concurrency, seed base, max_tokens, overrides, normalizers.
    settings: dict[str, Any]
    #: Every way the sent request differs from the recorded one.
    changes: dict[str, Any]
    #: For a variant arm: which files were swapped, and where in the payload.
    variant: dict[str, Any]
    base_url: str | None
    server_defaults: dict[str, Any]
    #: Warnings raised while setting up, e.g. server defaults that no longer match.
    notes: list[str]
    created_at: float
    sample_count: int = 0
    #: running | done | cancelled | interrupted | failed. None for an experiment recorded
    #: before experiments had a status, which is read from its samples instead.
    status: str | None = None
    #: The job that ran it, when one did (see `tuningfork.jobs`).
    job_id: str | None = None
    ended_at: float | None = None
    #: The process that ran it (`tuningfork.owners`): a running experiment whose process
    #: is gone was cut off.
    owner: dict[str, Any] | None = None
    #: The call held at a breakpoint it sampled, when it sampled one.
    hold_id: str | None = None
    #: When it was moved to the trash, with the other arm of its A/B if it has one.
    trashed_at: float | None = None

    @property
    def normalizers(self) -> list[str]:
        return list(self.settings.get("normalizers") or [])


@dataclass(slots=True)
class Sample:
    id: str
    experiment_id: str
    idx: int
    seed: int | None
    response_blob: str | None
    outcome_kind: str | None
    outcome_key: str | None
    outcome_label: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    timings: dict[str, Any]
    reasoning_chars: int
    error: str | None
    created_at: float
    #: For a sample run to the end: the fork it was, whether the task's check passed, and
    #: why it does not count, if it does not.
    run_id: str | None = None
    passed: bool | None = None
    exclusion: str | None = None
    #: For a forking-token probe's sample: the position forced, and the token forced there —
    #: {"position", "id", "text", "p", "original"}. See `tuningfork.experiment.forking`.
    forced: dict[str, Any] | None = None


@dataclass(slots=True)
class TokenStreamRecord:
    """A stored token stream: see the `token_streams` table and `tuningfork.analysis.tokens`."""

    id: str
    step_id: str | None
    sample_id: str | None
    server_id: str | None
    #: scored | sampled | continuation | recorded
    source: str
    #: grammar | per-token | chat-logprobs | completion
    method: str | None
    top_k: int | None
    blob: str
    tokens: int
    exact: bool | None
    notes: dict[str, Any]
    created_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "sample_id": self.sample_id,
            "server_id": self.server_id,
            "source": self.source,
            "method": self.method,
            "top_k": self.top_k,
            "tokens": self.tokens,
            "exact": self.exact,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class JobRecord:
    """A job as stored: see `tuningfork.jobs`."""

    id: str
    kind: str
    lane: str
    #: queued | running | done | failed | cancelled | interrupted
    status: str
    params: dict[str, Any]
    plan: dict[str, Any] | None
    progress: dict[str, Any]
    result: Any
    error: dict[str, Any] | None
    #: What it made: {"experiments": [...], "launches": [...], "runs": [...]}.
    links: dict[str, list[str]]
    owner: dict[str, Any] | None
    request_id: str | None
    created_at: float
    started_at: float | None
    ended_at: float | None
    #: When it was moved to the trash.
    trashed_at: float | None = None
    #: Who asked for it, when a front end said: `{"via": "mcp", "client": ..., "version": ...}`.
    started_by: dict[str, Any] | None = None

    @property
    def finished(self) -> bool:
        return self.status not in ("queued", "running")


#: A job's columns that change as it goes, and how each is stored.
_JOB_JSON = ("params", "plan", "progress", "result", "error", "links", "owner", "started_by")
_JOB_PLAIN = ("status", "started_at", "ended_at")


def _overrides(exact: Sequence[Any], stable: Sequence[Any]) -> dict[str, Any]:
    """Messages whose exact form differs from the stable one the tree stores, by index."""
    return {str(i): m for i, (m, s) in enumerate(zip(exact, stable)) if m != s}


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass(slots=True)
class Purge:
    """What a permanent deletion removes (see `Store.purge_closure`)."""

    runs: list[str]
    experiments: list[str]
    steps: int
    samples: list[str]
    streams: list[str]
    #: Each launch's id, with its workspace and harness home: their folders go too.
    launches: dict[str, tuple[str | None, str | None]]

    def to_json(self) -> dict[str, Any]:
        return {
            "runs": len(self.runs), "steps": self.steps, "experiments": len(self.experiments),
            "samples": len(self.samples), "token_streams": len(self.streams), "launches": len(self.launches),
        }


def _chunks(values: list[str], size: int = 500) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def identity_id(identity: dict[str, Any]) -> str:
    """A short id for a server identity: equal identities, equal ids.

    Where the server was reached is kept with the identity but is not part of it: two
    servers configured alike are one condition whichever port they are on, and
    `localhost` and `127.0.0.1` must not split one.
    """
    what = {key: value for key, value in identity.items() if key != "upstream"}
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def arrived(step: Step) -> float:
    """When a call reached the proxy, by the wall clock. A step recorded before arrival
    times were kept is estimated from when it was recorded and how long the call took."""
    timings = step.timings or {}
    at = timings.get("arrived_at")
    if isinstance(at, (int, float)):
        return float(at)
    return step.recorded_at - (timings.get("total_ms") or 0) / 1000


def _calls_to(response: dict[str, Any] | None, tools: set[str]) -> list[dict[str, Any]]:
    """The calls to any of `tools` in a common-shape response."""
    choices = (response or {}).get("choices") or []
    message = (choices[0].get("message") if choices and isinstance(choices[0], dict) else None) or {}
    return [
        call
        for call in message.get("tool_calls") or []
        if isinstance(call, dict) and (call.get("function") or {}).get("name") in tools
    ]


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


def _argument_texts(call: dict[str, Any], shortest: int = 16) -> list[str]:
    """The substantial strings a tool call passed: a subagent's goal, its prompt."""
    arguments = (call.get("function") or {}).get("arguments")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        parsed = arguments
    return [text for text in _strings(parsed) if len(text) >= shortest]


class Store:
    """Thread-safe SQLite store. One connection, guarded by a lock."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.home / "blobs")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.home / "tuningfork.db", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    #: Columns added after a table first shipped. `CREATE TABLE IF NOT EXISTS` never alters
    #: an existing table, so a database recorded by an earlier version gains them here.
    _MIGRATIONS = (
        ("steps", "edits", "TEXT"),
        ("steps", "harness_request_node_id", "TEXT"),
        ("steps", "delivered_response_blob", "TEXT"),
        ("runs", "result", "TEXT"),
        ("steps", "wire", "TEXT"),
        ("steps", "message_overrides", "TEXT"),
        ("runs", "launch_id", "TEXT"),
        ("runs", "role", "TEXT"),
        ("runs", "parent_run_id", "TEXT"),
        ("runs", "spawned_at", "INTEGER"),
        ("steps", "replay", "TEXT"),
        ("steps", "substitution", "TEXT"),
        ("samples", "run_id", "TEXT"),
        ("samples", "passed", "INTEGER"),
        ("samples", "exclusion", "TEXT"),
        ("steps", "workspace_commit", "TEXT"),
        ("launches", "initial_commit", "TEXT"),
        ("launches", "final_commit", "TEXT"),
        ("launches", "skill_commit", "TEXT"),
        ("steps", "server_id", "TEXT"),
        ("runs", "server_id", "TEXT"),
        ("experiments", "status", "TEXT"),
        ("experiments", "job_id", "TEXT"),
        ("experiments", "ended_at", "REAL"),
        ("experiments", "owner", "TEXT"),
        ("launches", "job_id", "TEXT"),
        ("servers", "checks", "TEXT"),
        ("experiments", "hold_id", "TEXT"),
        ("samples", "forced", "TEXT"),
        # The trash (see `Store.trash_run`): when a record was moved there, and for a run,
        # what it went with — itself, its launch's main run, or an experiment.
        ("runs", "trashed_at", "REAL"),
        ("runs", "trashed_with", "TEXT"),
        ("experiments", "trashed_at", "REAL"),
        ("jobs", "trashed_at", "REAL"),
        # Who asked for a job: `{"via": "mcp", "client": ...}`. Not `owner`, which is the
        # process that ran it, read by the sweep to tell live work from work cut off.
        ("jobs", "started_by", "TEXT"),
        # The process running a launch, as for a job or an experiment: a batch killed hard
        # never says how its launches ended.
        ("launches", "owner", "TEXT"),
        # The conversation a scenario's run continues, spliced into what the model saw.
        ("steps", "context", "TEXT"),
    )

    #: Indexes on migrated columns. They live here, not in SCHEMA: SCHEMA runs first, and on
    #: a database from before the column existed, an index on it would stop the store from
    #: opening at all.
    _MIGRATED_INDEXES = (
        "CREATE INDEX IF NOT EXISTS runs_launch ON runs(launch_id)",
        "CREATE INDEX IF NOT EXISTS experiments_job ON experiments(job_id)",
        "CREATE INDEX IF NOT EXISTS launches_job ON launches(job_id)",
        "CREATE INDEX IF NOT EXISTS experiments_hold ON experiments(hold_id)",
        "CREATE INDEX IF NOT EXISTS runs_trashed ON runs(trashed_with)",
    )

    #: The experiments table as it is now, for rebuilding one from before its anchors could
    #: be NULL: SQLite cannot drop a NOT NULL in place.
    _EXPERIMENTS_NOW = """
        CREATE TABLE experiments_rebuilt (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, group_id TEXT, arm TEXT NOT NULL,
            run_id TEXT, step_idx INTEGER, step_id TEXT, request_blob TEXT NOT NULL,
            settings TEXT, changes TEXT, variant TEXT, base_url TEXT, server_defaults TEXT,
            notes TEXT, created_at REAL NOT NULL, status TEXT, job_id TEXT, ended_at REAL,
            owner TEXT, hold_id TEXT, trashed_at REAL
        )"""
    _EXPERIMENT_INDEXES = (
        "CREATE INDEX IF NOT EXISTS experiments_checkpoint ON experiments(run_id, step_idx)",
        "CREATE INDEX IF NOT EXISTS experiments_group ON experiments(group_id)",
        "CREATE INDEX IF NOT EXISTS experiments_job ON experiments(job_id)",
        "CREATE INDEX IF NOT EXISTS experiments_hold ON experiments(hold_id)",
    )

    def _migrate(self) -> None:
        for table, column, kind in self._MIGRATIONS:
            columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        for statement in self._MIGRATED_INDEXES:
            self._conn.execute(statement)
        self._relax_experiment_anchors()

    def _relax_experiment_anchors(self) -> None:
        """A database from before an experiment could sample a held call has NOT NULL
        anchors: rebuild the table with them nullable, every row and index kept."""
        info = {row["name"]: row for row in self._conn.execute("PRAGMA table_info(experiments)")}
        if not info["run_id"]["notnull"]:
            return
        columns = ", ".join(info)
        if self._conn.in_transaction:
            self._conn.commit()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(self._EXPERIMENTS_NOW)
            self._conn.execute(
                f"INSERT INTO experiments_rebuilt ({columns}) SELECT {columns} FROM experiments"
            )
            self._conn.execute("DROP TABLE experiments")
            self._conn.execute("ALTER TABLE experiments_rebuilt RENAME TO experiments")
            for statement in self._EXPERIMENT_INDEXES:
                self._conn.execute(statement)
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Everything inside is stored, or nothing is.

        A step touches several tables — its nodes, its run, the label it claims. Half of
        that left in an open transaction would be committed by whatever wrote next: a run
        with no steps, holding a label no longer queued for the run it was meant for.
        """
        with self._lock:
            try:
                yield
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()

    # --- tree ---------------------------------------------------------------------

    def _insert_path(self, messages: Sequence[dict[str, Any]]) -> list[str]:
        """Insert every message of a request as tree nodes; return the node path."""
        path: list[str] = []
        parent: str | None = None
        for depth, message in enumerate(messages):
            digest = exact_hash(message)
            current = node_id(parent, digest)
            blob = self.blobs.put_json(message)
            self._conn.execute(
                "INSERT OR IGNORE INTO nodes"
                " (id, parent_id, exact_hash, normal_hash, message_blob, depth)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (current, parent, digest, normal_hash(message), blob, depth),
            )
            path.append(current)
            parent = current
        return path

    def messages_for_node(self, node: str) -> list[dict[str, Any]]:
        """Walk from a node back to the root, returning messages in order."""
        messages: list[dict[str, Any]] = []
        current: str | None = node
        with self._lock:
            while current:
                row = self._conn.execute(
                    "SELECT parent_id, message_blob FROM nodes WHERE id = ?", (current,)
                ).fetchone()
                if row is None:
                    break
                messages.append(self.blobs.get_json(row["message_blob"]))
                current = row["parent_id"]
        messages.reverse()
        return messages

    def path_rows(self, node: str | None) -> list[dict[str, Any]]:
        """A path's nodes from the root down — `id`, `exact_hash`, `message_blob`, `depth` —
        in one query, reading no message: for comparing paths, and loading each blob once."""
        if not node:
            return []
        with self._lock:
            rows = self._conn.execute(
                "WITH RECURSIVE path(id, parent_id, exact_hash, message_blob, depth) AS ("
                " SELECT id, parent_id, exact_hash, message_blob, depth FROM nodes WHERE id = ?"
                " UNION ALL SELECT n.id, n.parent_id, n.exact_hash, n.message_blob, n.depth"
                " FROM nodes n JOIN path p ON n.id = p.parent_id)"
                " SELECT id, exact_hash, message_blob, depth FROM path ORDER BY depth",
                (node,),
            ).fetchall()
        return [dict(row) for row in rows]

    def node_depths(self, nodes: Sequence[str]) -> dict[str, int]:
        """Each node's depth — its path's length less one — in one query."""
        wanted = sorted({node for node in nodes if node})
        if not wanted:
            return {}
        marks = ",".join("?" * len(wanted))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, depth FROM nodes WHERE id IN ({marks})", wanted
            ).fetchall()
        return {row["id"]: row["depth"] for row in rows}

    def harness_extras(self, step: Step) -> dict[str, Any]:
        """The request's keys besides its messages, as the harness sent them: the model's,
        unless a breakpoint edit or a fork's variant changed them for the model."""
        changed = step.edits.get("request") or step.substitution or step.context or {}
        if step.harness_request_node_id and changed.get("original_extras") is not None:
            return {k: v for k, v in changed["original_extras"].items() if k != "messages"}
        extras = self.blobs.get_json(step.extras_blob)
        return {k: v for k, v in extras.items() if k != "messages"}

    def normal_hashes_for_node(self, node: str) -> list[str]:
        """Normalized hashes along a path, for comparing runs that did the same thing."""
        hashes: list[str] = []
        current: str | None = node
        with self._lock:
            while current:
                row = self._conn.execute(
                    "SELECT parent_id, normal_hash FROM nodes WHERE id = ?", (current,)
                ).fetchone()
                if row is None:
                    break
                hashes.append(row["normal_hash"])
                current = row["parent_id"]
        hashes.reverse()
        return hashes

    # --- recording ----------------------------------------------------------------

    def record_step(
        self,
        *,
        request: dict[str, Any],
        response: dict[str, Any] | None = None,
        raw_response_blob: str | None = None,
        params: dict[str, Any] | None = None,
        server_defaults: dict[str, Any] | None = None,
        status: int | None = None,
        timings: dict[str, Any] | None = None,
        error: str | None = None,
        harness: str | None = None,
        harness_request: dict[str, Any] | None = None,
        delivered_response: dict[str, Any] | None = None,
        edits: dict[str, Any] | None = None,
        wire: str | None = None,
        normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        assignment: Any = None,
        replay: dict[str, Any] | None = None,
        substitution: dict[str, Any] | None = None,
        workspace_commit: str | None = None,
        server_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Step:
        """Record one model call.

        `assignment` (a `tuningfork.launches.Assignment`) is where a launched call belongs,
        decided when it arrived; such a call is recorded there, and neither labels nor path
        matching are consulted. Without one, the call is attached by its message path.

        `request` and `response` are what the *model* saw and produced. At a breakpoint a
        person can change either, and then the harness's view differs from the model's:

        - `harness_request` is the request as the harness sent it, before an edit. The
          harness keeps its own history, so its next request extends *this* path, not the
          edited one — run grouping must follow it, or every edit would split the run.
        - `delivered_response` is what the harness received, if the response was edited.
          The model's own output stays the step's response: it is the behaviour.

        The request is stored in its own wire format, exactly as sent; responses are in the
        common shape (an OpenAI `chat.completion`), whatever the format — see `wire`.

        `normalize` gives a message the stable form the tree identifies it by, when a
        harness re-encodes earlier messages from one request to the next. Messages whose
        exact form differs are kept on the step, so `rebuild_request` is still exact.
        """
        exact = request.get("messages") or []
        messages = [normalize(m) for m in exact] if normalize else list(exact)
        overrides = _overrides(exact, messages)
        # `messages` keeps its place among the keys, as null: a rebuilt request has every
        # key where the harness put it.
        extras = {k: (None if k == "messages" else v) for k, v in request.items()}

        with self._transaction():
            path = self._insert_path(messages)
            if harness_request is not None:
                harness_exact = harness_request.get("messages") or []
                harness_messages = (
                    [normalize(m) for m in harness_exact] if normalize else list(harness_exact)
                )
                attach_path = self._insert_path(harness_messages)
                harness_overrides = _overrides(harness_exact, harness_messages)
                kept = substitution if substitution is not None else context
                if kept is not None and not edits:
                    # A fork's variant, or a scenario's context — not a person's edit: kept
                    # with it, so the step does not read as edited (which would void the run).
                    # The step's own extras stay the model's, as with an edit: what a
                    # resample resends.
                    kept = copy.deepcopy(kept)
                    if harness_overrides:
                        kept["harness_overrides"] = harness_overrides
                    harness_extras = {k: v for k, v in harness_request.items() if k != "messages"}
                    if harness_extras != {k: v for k, v in request.items() if k != "messages"}:
                        kept["original_extras"] = harness_extras
                    if substitution is not None:
                        substitution = kept
                    else:
                        context = kept
                elif harness_overrides:
                    edits = copy.deepcopy(edits or {})
                    edits.setdefault("request", {})["harness_overrides"] = harness_overrides
            else:
                attach_path = path
            if assignment is not None:
                run = self._launched_run(assignment, attach_path, request.get("model"), harness)
                idx = assignment.idx
            else:
                run = self._attach_to_run(attach_path, request.get("model"), harness)
                idx = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (run,)
                ).fetchone()["n"]
            if server_id:
                # A run is filed under the server its first recorded call reached.
                self._conn.execute(
                    "UPDATE runs SET server_id = ? WHERE id = ? AND server_id IS NULL",
                    (server_id, run),
                )

            finish_reason = None
            reasoning_chars = 0
            usage: dict[str, Any] = {}
            response_blob = None
            if response is not None:
                response_blob = self.blobs.put_json(response)
                choices = response.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    message = choices[0].get("message") or {}
                    # Counted from the text, not from usage: llama.cpp returns the
                    # reasoning but no reasoning_tokens counter.
                    for key in ("reasoning_content", "reasoning"):
                        if isinstance(message.get(key), str):
                            reasoning_chars += len(message[key])
                if isinstance(response.get("usage"), dict):
                    usage = response["usage"]

            step = Step(
                id=uuid.uuid4().hex,
                run_id=run,
                idx=idx,
                request_node_id=path[-1] if path else "",
                extras_blob=self.blobs.put_json(extras),
                response_blob=response_blob,
                raw_response_blob=raw_response_blob,
                params=params or {},
                server_defaults=server_defaults or {},
                model=request.get("model"),
                status=status,
                finish_reason=finish_reason,
                timings=timings or {},
                usage=usage,
                reasoning_chars=reasoning_chars,
                error=error,
                recorded_at=time.time(),
                edits=edits or {},
                harness_request_node_id=(
                    attach_path[-1] if harness_request and attach_path else None
                ),
                delivered_response_blob=(
                    self.blobs.put_json(delivered_response)
                    if delivered_response is not None
                    else None
                ),
                wire=wire,
                message_overrides=overrides,
                replay=replay or {},
                substitution=substitution or {},
                workspace_commit=workspace_commit,
                server_id=server_id,
                context=context or {},
            )
            self._conn.execute(
                "INSERT INTO steps (id, run_id, idx, request_node_id, extras_blob,"
                " response_blob, raw_response_blob, params, server_defaults, model,"
                " status, finish_reason, timings, usage, reasoning_chars, error,"
                " recorded_at, edits, harness_request_node_id, delivered_response_blob,"
                " wire, message_overrides, replay, substitution, workspace_commit, server_id, context)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    step.id, step.run_id, step.idx, step.request_node_id,
                    step.extras_blob, step.response_blob, step.raw_response_blob,
                    json.dumps(step.params), json.dumps(step.server_defaults),
                    step.model, step.status, step.finish_reason,
                    json.dumps(step.timings), json.dumps(step.usage),
                    step.reasoning_chars, step.error, step.recorded_at,
                    json.dumps(step.edits), step.harness_request_node_id,
                    step.delivered_response_blob, step.wire,
                    json.dumps(step.message_overrides) if step.message_overrides else None,
                    json.dumps(step.replay) if step.replay else None,
                    json.dumps(step.substitution) if step.substitution else None,
                    step.workspace_commit,
                    step.server_id,
                    json.dumps(step.context) if step.context else None,
                ),
            )
            # The head follows the harness's view: its next request extends what *it*
            # sent, whatever the model was actually shown. A launched run's steps can be
            # recorded out of order, so its head is only moved forward.
            self._conn.execute(
                "UPDATE runs SET head_node_id = ?, updated_at = ? WHERE id = ?"
                " AND NOT EXISTS (SELECT 1 FROM steps WHERE run_id = ? AND idx > ?)",
                (attach_path[-1] if attach_path else "", step.recorded_at, run, run, idx),
            )
        return step

    def _launched_run(
        self, assignment: Any, path: Sequence[str], model: str | None, harness: str | None
    ) -> str:
        """The run a launched call was assigned to, created with its first recorded step."""
        exists = self._conn.execute(
            "SELECT 1 FROM runs WHERE id = ?", (assignment.run_id,)
        ).fetchone()
        if exists is None:
            now = time.time()
            self._conn.execute(
                "INSERT INTO runs (id, root_node_id, head_node_id, label, task_id, harness,"
                " model, started_at, updated_at, launch_id, role)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    assignment.run_id,
                    path[0] if path else "",
                    path[-1] if path else "",
                    assignment.label,
                    # Only the task's own conversation is filed under the task: a subagent's
                    # thread is part of a run, not a run of the task.
                    assignment.task_id if assignment.main else None,
                    harness,
                    model,
                    now,
                    now,
                    assignment.launch_id,
                    assignment.role,
                ),
            )
        return assignment.run_id

    def predict_attachment(self, messages: Sequence[dict[str, Any]]) -> tuple[str | None, int]:
        """Which run a request *would* join, and at which step, without recording anything.

        Request-phase breakpoints need this: a condition like "step >= 3" has to be decided
        before the request goes upstream, while the step is not yet recorded. Mirrors
        `_attach_to_run`, read-only. Returns (None, 0) for a request that would start a
        new run.
        """
        path = node_path(messages)
        with self._lock:
            run_id = self._existing_run_for(path)
            if run_id is None:
                return None, 0
            count = self._conn.execute(
                "SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (run_id,)
            ).fetchone()["n"]
        return run_id, count

    def _attach_to_run(
        self, path: Sequence[str], model: str | None, harness: str | None
    ) -> str:
        """Decide which run this request belongs to.

        Extends a run whose head lies on this path; branches off a known root as a fork;
        otherwise starts a new run.
        """
        existing = self._existing_run_for(path)
        if existing is not None:
            return existing
        if path and not self._peek_pending_force():
            forked = self._conn.execute(
                "SELECT id FROM runs WHERE root_node_id = ? AND launch_id IS NULL"
                " AND trashed_at IS NULL ORDER BY started_at LIMIT 1",
                (path[0],),
            ).fetchone()
            if forked:
                return self._new_run(path, model, harness, forked_from=forked["id"])
        return self._new_run(path, model, harness)

    def _existing_run_for(self, path: Sequence[str]) -> str | None:
        """The run this path continues, if any. Read-only; callers hold the lock.

        Shared by recording and by `predict_attachment`, so a breakpoint's idea of which
        step a request is can never drift from where it is then recorded.
        """
        # An asserted boundary always wins over path matching.
        if not path or self._peek_pending_force():
            return None

        placeholders = ",".join("?" for _ in path)
        # A launched run is attributed by its launch, never by path: a call without the
        # launch's prefix is not one of its calls, whatever its messages.
        rows = self._conn.execute(
            f"SELECT id, head_node_id, updated_at FROM runs"
            f" WHERE head_node_id IN ({placeholders}) AND launch_id IS NULL AND trashed_at IS NULL",
            tuple(path),
        ).fetchall()
        if not rows:
            return None
        # Several runs of one task share their whole message path, so more than one run's
        # head can sit on this path. Distinguish by *where*:
        #
        #   head before the last message  -> this request extends that run
        #   head at the last message      -> this request repeats that run's last call
        #
        # An extension is the stronger claim, so it wins. Without this, the second run of
        # a batch would have its later steps absorbed by the first run, which shares the
        # identical path and has a deeper head.
        index = {node: i for i, node in enumerate(path)}
        last = len(path) - 1
        extends = [row for row in rows if index[row["head_node_id"]] < last]
        if extends:
            return max(
                extends, key=lambda row: (index[row["head_node_id"]], row["updated_at"])
            )["id"]
        return max(rows, key=lambda row: row["updated_at"])["id"]

    def _new_run(
        self,
        path: Sequence[str],
        model: str | None,
        harness: str | None,
        forked_from: str | None = None,
    ) -> str:
        now = time.time()
        run_id = uuid.uuid4().hex
        label, task_id = self._take_pending_label()
        self._conn.execute(
            "INSERT INTO runs (id, root_node_id, head_node_id, label, task_id, harness,"
            " model, forked_from_run, forked_at_depth, skill_hashes, parity,"
            " started_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                path[0] if path else "",
                path[-1] if path else "",
                label,
                task_id,
                harness,
                model,
                forked_from,
                len(path) - 1 if forked_from else None,
                None,
                None,
                now,
                now,
            ),
        )
        return run_id

    # --- labels -------------------------------------------------------------------

    def label_next(
        self, label: str, task_id: str | None = None, force: bool = False
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_labels (label, task_id, force, created_at)"
                " VALUES (?,?,?,?)",
                (label, task_id, 1 if force else 0, time.time()),
            )
            self._conn.commit()

    def begin_run(self, label: str, task_id: str | None = None) -> None:
        """Declare that the next request starts a new run, whatever its history.

        Repeated runs of one task open with byte-identical requests, so the tree alone
        cannot tell a fresh run from a retry of the previous one — they share a path and
        would collapse into a single run. A batch runner knows where the boundaries are,
        so it asserts them here rather than leaving it to a heuristic.
        """
        self.label_next(label, task_id, force=True)

    def _peek_pending_force(self) -> bool:
        row = self._conn.execute(
            "SELECT force FROM pending_labels ORDER BY id LIMIT 1"
        ).fetchone()
        return bool(row and row["force"])

    def _take_pending_label(self) -> tuple[str | None, str | None]:
        row = self._conn.execute(
            "SELECT id, label, task_id FROM pending_labels ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return None, None
        self._conn.execute("DELETE FROM pending_labels WHERE id = ?", (row["id"],))
        return row["label"], row["task_id"]

    def next_pending(self) -> tuple[str, str | None] | None:
        """The (label, task) the next new run will take, without taking it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT label, task_id FROM pending_labels ORDER BY id LIMIT 1"
            ).fetchone()
        return (row["label"], row["task_id"]) if row else None

    def discard_pending_label(self, label: str) -> None:
        """Drop a queued label that was never claimed.

        A harness can exit before issuing a single request — Hermes aborts outright if the
        model's context window is too small, for instance. The queued label would otherwise
        be picked up by whatever ran next and quietly mislabel it.
        """
        with self._lock:
            self._conn.execute("DELETE FROM pending_labels WHERE label = ?", (label,))
            self._conn.commit()

    def discard_pending_labels(self) -> list[str]:
        """Drop every queued label, returning them.

        For a batch about to start: labels are claimed first in, first out, so anything
        already queued would be taken by the batch's first run. A run boundary still queued
        was left by a batch that died before its run reached the proxy; a label queued with
        `label_next` was meant for some other run.
        """
        with self._lock:
            rows = self._conn.execute("SELECT label FROM pending_labels ORDER BY id").fetchall()
            self._conn.execute("DELETE FROM pending_labels")
            self._conn.commit()
        return [row["label"] for row in rows]

    def pending_labels(self) -> list[str]:
        with self._lock:
            return [
                row["label"]
                for row in self._conn.execute(
                    "SELECT label FROM pending_labels ORDER BY id"
                )
            ]

    def annotate_run(
        self,
        run_id: str,
        *,
        label: str | None = None,
        task_id: str | None = None,
        skill_hashes: dict[str, str] | None = None,
        parity: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        harness: str | None = None,
    ) -> None:
        sets, values = [], []
        for column, value in (
            ("label", label),
            ("task_id", task_id),
            ("harness", harness),
            ("skill_hashes", json.dumps(skill_hashes) if skill_hashes is not None else None),
            ("parity", json.dumps(parity) if parity is not None else None),
            ("result", json.dumps(result) if result is not None else None),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                values.append(value)
        if not sets:
            return
        values.append(run_id)
        with self._lock:
            self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", values)
            self._conn.commit()

    # --- reading ------------------------------------------------------------------

    def _run_from_row(self, row: sqlite3.Row) -> Run:
        return Run(
            id=row["id"],
            root_node_id=row["root_node_id"],
            head_node_id=row["head_node_id"],
            label=row["label"],
            task_id=row["task_id"],
            harness=row["harness"],
            model=row["model"],
            forked_from_run=row["forked_from_run"],
            forked_at_depth=row["forked_at_depth"],
            skill_hashes=_loads(row["skill_hashes"], {}),
            parity=_loads(row["parity"], {}),
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            step_count=row["step_count"] if "step_count" in row.keys() else 0,
            result=_loads(row["result"], {}),
            launch_id=row["launch_id"],
            role=row["role"],
            parent_run_id=row["parent_run_id"],
            spawned_at=row["spawned_at"],
            server_id=row["server_id"],
            trashed_at=row["trashed_at"],
            trashed_with=row["trashed_with"],
        )

    # --- the trash ----------------------------------------------------------------
    #
    # A run, an experiment or a job moved to the trash is left out of every list and every
    # statistic, and changed in nothing else: whatever is tied to it stays in place, so
    # restoring it is clearing a mark. Deleting it from the trash is permanent: it goes, with
    # everything tied to it (`purge_closure`), and then whatever no remaining record refers
    # to (`collect_garbage`, `live_blobs`).

    def trash_run(self, run_id: str) -> list[str]:
        """Move a run to the trash with its launch's threads; return the runs moved."""
        now = time.time()
        with self._transaction():
            row = self._conn.execute("SELECT launch_id FROM runs WHERE id = ?", (run_id,)).fetchone()
            ids = [run_id]
            if row is not None and row["launch_id"]:
                ids += [
                    found["id"] for found in self._conn.execute(
                        "SELECT id FROM runs WHERE launch_id = ? AND id != ? AND role IN (?, ?, ?)",
                        (row["launch_id"], run_id, SUBAGENT, SIDE, THREAD),
                    )
                ]
            self._conn.executemany(
                "UPDATE runs SET trashed_at = ?, trashed_with = ? WHERE id = ? AND trashed_at IS NULL",
                [(now, run_id, found) for found in ids],
            )
        return ids

    def trash_experiments(self, root: str, experiment_ids: Sequence[str]) -> list[str]:
        """Move experiments to the trash — an A/B's two arms go together — with the forks
        their samples ran to the end, all filed under `root` to come back together."""
        now = time.time()
        with self._transaction():
            self._conn.executemany(
                "UPDATE experiments SET trashed_at = ? WHERE id = ? AND trashed_at IS NULL",
                [(now, found) for found in experiment_ids],
            )
            forks = self._sample_fork_runs(experiment_ids)
            self._conn.executemany(
                "UPDATE runs SET trashed_at = ?, trashed_with = ? WHERE id = ? AND trashed_at IS NULL",
                [(now, root, found) for found in forks],
            )
        return list(experiment_ids)

    def trash_job(self, job_id: str) -> None:
        with self._transaction():
            self._conn.execute(
                "UPDATE jobs SET trashed_at = ? WHERE id = ? AND trashed_at IS NULL", (time.time(), job_id)
            )

    def restore_trashed(self, root: str) -> int:
        """Bring back what went to the trash under `root` — a run with its threads, an
        experiment or an A/B's arms with their forks, a job. Returns how many records."""
        with self._transaction():
            count = self._conn.execute(
                "UPDATE runs SET trashed_at = NULL, trashed_with = NULL WHERE trashed_with = ?", (root,)
            ).rowcount
            count += self._conn.execute(
                "UPDATE experiments SET trashed_at = NULL WHERE trashed_at IS NOT NULL AND (id = ? OR group_id = ?)",
                (root, root),
            ).rowcount
            count += self._conn.execute(
                "UPDATE jobs SET trashed_at = NULL WHERE id = ? AND trashed_at IS NOT NULL", (root,)
            ).rowcount
        return count

    def trashed(self) -> dict[str, list[Any]]:
        """What is in the trash, the latest first: the runs moved there themselves (their
        threads went with them), experiments, and jobs."""
        with self._lock:
            runs = [
                self._run_from_row(row) for row in self._conn.execute(
                    "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id) AS step_count"
                    " FROM runs r WHERE r.trashed_with = r.id ORDER BY r.trashed_at DESC"
                )
            ]
            experiments = [
                self._experiment_from_row(row) for row in self._conn.execute(
                    self._EXPERIMENT_SELECT + " WHERE e.trashed_at IS NOT NULL ORDER BY e.trashed_at DESC"
                )
            ]
            jobs = [
                self._job_from_row(row) for row in self._conn.execute(
                    "SELECT * FROM jobs WHERE trashed_at IS NOT NULL ORDER BY trashed_at DESC"
                )
            ]
        return {"runs": runs, "experiments": experiments, "jobs": jobs}

    def _sample_fork_runs(self, experiment_ids: Sequence[str]) -> list[str]:
        """The runs an experiment's samples ran to the end — forks — with their threads."""
        launches = set()
        runs = set()
        for chunk in _chunks(list(experiment_ids)):
            marks = ",".join("?" for _ in chunk)
            for row in self._conn.execute(
                f"SELECT DISTINCT run_id FROM samples WHERE experiment_id IN ({marks}) AND run_id IS NOT NULL", chunk
            ):
                runs.add(row["run_id"])
            for row in self._conn.execute(f"SELECT id FROM launches WHERE experiment_id IN ({marks})", chunk):
                launches.add(row["id"])
        for chunk in _chunks(sorted(launches)):
            marks = ",".join("?" for _ in chunk)
            runs.update(row["id"] for row in self._conn.execute(
                f"SELECT id FROM runs WHERE launch_id IN ({marks})", chunk
            ))
        return sorted(runs)

    def purge_closure(self, *, runs: Sequence[str] = (), experiments: Sequence[str] = ()) -> Purge:
        """Everything that goes when these runs and experiments are deleted for good. A run
        takes its launch's threads, its steps, the experiments at its steps, and the forks
        tuningfork launched from it (they replayed its recording); an experiment takes the
        other arm of its A/B, its samples, and the forks they ran to the end — each of which
        takes what a run takes. Token streams, launches and held calls filed under any of
        these go too. A run recorded ad hoc that only shares a beginning with one of them
        (`forked_from_run`) stays: it is a run of its own. Jobs stay: a job is a record of
        work, not the work. Read-only."""
        run_set, experiment_set = set(runs), set(experiments)
        with self._lock:
            while True:
                before = (len(run_set), len(experiment_set))
                launch_set = self._ids("SELECT DISTINCT launch_id AS id FROM runs WHERE id IN ({}) AND launch_id IS NOT NULL", run_set)
                launch_set |= self._ids("SELECT id FROM launches WHERE source_run IN ({})", run_set)
                launch_set |= self._ids("SELECT id FROM launches WHERE experiment_id IN ({})", experiment_set)
                run_set |= self._ids("SELECT id FROM runs WHERE launch_id IN ({})", launch_set)
                run_set |= self._ids("SELECT DISTINCT run_id AS id FROM samples WHERE experiment_id IN ({}) AND run_id IS NOT NULL", experiment_set)
                experiment_set |= self._ids("SELECT id FROM experiments WHERE run_id IN ({})", run_set)
                experiment_set |= self._ids(
                    "SELECT e.id FROM experiments e JOIN hold_steps h ON e.hold_id = h.hold_id WHERE h.run_id IN ({})", run_set
                )
                groups = self._ids("SELECT DISTINCT group_id AS id FROM experiments WHERE id IN ({}) AND group_id IS NOT NULL", experiment_set)
                experiment_set |= self._ids("SELECT id FROM experiments WHERE group_id IN ({})", groups)
                if (len(run_set), len(experiment_set)) == before:
                    break
            steps = self._ids("SELECT id FROM steps WHERE run_id IN ({})", run_set)
            samples = self._ids("SELECT id FROM samples WHERE experiment_id IN ({})", experiment_set)
            streams = self._ids("SELECT id FROM token_streams WHERE step_id IN ({})", steps)
            streams |= self._ids("SELECT id FROM token_streams WHERE sample_id IN ({})", samples)
            launches = {
                row["id"]: (row["workspace"], row["home"]) for chunk in _chunks(sorted(launch_set))
                for row in self._conn.execute(
                    f"SELECT id, workspace, home FROM launches WHERE id IN ({','.join('?' for _ in chunk)})", chunk
                )
            }
        return Purge(
            runs=sorted(run_set), experiments=sorted(experiment_set), steps=len(steps),
            samples=sorted(samples), streams=sorted(streams), launches=launches,
        )

    def _ids(self, query: str, values: set[str]) -> set[str]:
        """Ids a query returns for a set of ids, asked in chunks SQLite can take."""
        found: set[str] = set()
        for chunk in _chunks(sorted(values)):
            found.update(row["id"] for row in self._conn.execute(query.format(",".join("?" for _ in chunk)), chunk))
        return found

    def purge(self, plan: Purge) -> None:
        """Delete what `purge_closure` named, in one transaction: all of it, or none."""
        with self._transaction():
            # A run that only shared its beginning with a deleted one is no longer its fork.
            for chunk in _chunks(list(plan.runs)):
                self._conn.execute(
                    "UPDATE runs SET forked_from_run = NULL, forked_at_depth = NULL"
                    f" WHERE forked_from_run IN ({','.join('?' for _ in chunk)})", chunk,
                )
            for table, column, ids in (
                ("token_streams", "id", plan.streams),
                ("samples", "id", plan.samples),
                ("experiments", "id", plan.experiments),
                ("hold_steps", "run_id", plan.runs),
                ("steps", "run_id", plan.runs),
                ("runs", "id", plan.runs),
                ("launches", "id", sorted(plan.launches)),
            ):
                for chunk in _chunks(list(ids)):
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({','.join('?' for _ in chunk)})", chunk
                    )

    def purge_job(self, job_id: str) -> None:
        with self._transaction():
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def collect_garbage(self) -> int:
        """Remove the message-tree nodes no remaining step or run reaches; return how many.
        A node shared with a remaining run is reached through it, and stays."""
        with self._transaction():
            self._conn.execute(
                """
                WITH RECURSIVE live(id) AS (
                    SELECT request_node_id FROM steps
                    UNION SELECT harness_request_node_id FROM steps WHERE harness_request_node_id IS NOT NULL
                    UNION SELECT root_node_id FROM runs
                    UNION SELECT head_node_id FROM runs
                    UNION SELECT n.parent_id FROM nodes n JOIN live ON n.id = live.id WHERE n.parent_id IS NOT NULL
                )
                DELETE FROM nodes WHERE id NOT IN (SELECT id FROM live WHERE id IS NOT NULL)
                """
            )
            # Not `rowcount`: Python's sqlite3 gives -1 for a DELETE that opens with WITH.
            return self._conn.execute("SELECT changes()").fetchone()[0]

    def live_blobs(self) -> set[str]:
        """Every blob a remaining record refers to: what a sweep of the blob store keeps."""
        columns = (
            ("nodes", "message_blob"), ("steps", "extras_blob"), ("steps", "response_blob"),
            ("steps", "raw_response_blob"), ("steps", "delivered_response_blob"),
            ("experiments", "request_blob"), ("samples", "response_blob"), ("token_streams", "blob"),
        )
        live: set[str] = set()
        with self._lock:
            for table, column in columns:
                live.update(
                    row[0] for row in self._conn.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL")
                )
            # A scenario's context, kept once for every launch of it and every fork after.
            live.update(
                row[0] for row in self._conn.execute(
                    "SELECT json_extract(settings, '$.context.blob') FROM launches"
                    " WHERE json_extract(settings, '$.context.blob') IS NOT NULL"
                )
            )
        return live

    # --- servers ------------------------------------------------------------------

    def add_server(self, identity: dict[str, Any]) -> str:
        """Keep a server identity; return its id. The same identity gets the same id."""
        server_id = identity_id(identity)
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO servers (id, identity, first_seen) VALUES (?,?,?)",
                (server_id, json.dumps(identity, sort_keys=True), time.time()),
            )
            self._conn.commit()
        return server_id

    def server(self, server_id: str | None) -> dict[str, Any] | None:
        if not server_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT identity FROM servers WHERE id = ?", (server_id,)
            ).fetchone()
        return _loads(row["identity"], {}) if row else None

    def set_server_checks(self, server_id: str, checks: dict[str, Any]) -> None:
        """What a server check found about this server, measured rather than assumed."""
        with self._lock:
            self._conn.execute(
                "UPDATE servers SET checks = ? WHERE id = ?", (json.dumps(checks), server_id)
            )
            self._conn.commit()

    def server_checks(self, server_id: str | None) -> dict[str, Any]:
        if not server_id:
            return {}
        with self._lock:
            row = self._conn.execute("SELECT checks FROM servers WHERE id = ?", (server_id,)).fetchone()
        return _loads(row["checks"], {}) if row else {}

    def servers(self) -> list[tuple[str, dict[str, Any], float, int]]:
        """(id, identity, first seen, runs filed under it), newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.identity, s.first_seen,"
                " (SELECT COUNT(*) FROM runs r WHERE r.server_id = s.id) AS runs"
                " FROM servers s ORDER BY s.first_seen DESC"
            ).fetchall()
        return [(row["id"], _loads(row["identity"], {}), row["first_seen"], row["runs"]) for row in rows]

    def runs(
        self,
        limit: int = 50,
        task_id: str | None = None,
        threads: bool = False,
        forks: bool = False,
    ) -> list[Run]:
        """Runs, newest first. A launch's subagent and side threads are left out unless
        `threads`: they belong to their launch's main run, not beside it. Forks are left
        out unless `forks`: each is a sample of an experiment, conditional on the history it
        replayed, and would be misread as a run of the task. Runs in the trash are left out:
        out of every list, and every statistic."""
        query = (
            "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id) AS step_count"
            " FROM runs r"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if task_id:
            clauses.append("r.task_id = ?")
            params.append(task_id)
        roles = [MAIN] + ([FORK] if forks else []) + ([SUBAGENT, SIDE, THREAD] if threads else [])
        clauses.append(
            "(r.role IS NULL OR r.role IN (" + ",".join(f"'{role}'" for role in roles) + "))"
        )
        clauses.append("r.trashed_at IS NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY r.started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [self._run_from_row(row) for row in self._conn.execute(query, params)]

    # --- launches -----------------------------------------------------------------

    def create_launch(
        self,
        *,
        kind: str,
        label: str | None = None,
        harness: str | None = None,
        model: str | None = None,
        task_id: str | None = None,
        eval: dict[str, Any] | None = None,
        spec_path: str | None = None,
        skill_dir: str | None = None,
        workspace: str | None = None,
        home: str | None = None,
        settings: dict[str, Any] | None = None,
        source_run: str | None = None,
        source_step: int | None = None,
        variant: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        sample_idx: int | None = None,
        token: str | None = None,
    ) -> Launch:
        from tuningfork.launches import new_token
        from tuningfork.owners import current_owner

        launch = Launch(
            id=token or new_token(),
            kind=kind,
            label=label,
            harness=harness,
            model=model,
            task_id=task_id,
            eval=eval or {},
            spec_path=spec_path,
            skill_dir=skill_dir,
            workspace=workspace,
            home=home,
            settings=settings or {},
            source_run=source_run,
            source_step=source_step,
            variant=variant or {},
            experiment_id=experiment_id,
            sample_idx=sample_idx,
            status="running",
            exit={},
            created_at=time.time(),
            ended_at=None,
            owner=current_owner(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO launches (id, kind, label, harness, model, task_id, eval,"
                " spec_path, skill_dir, workspace, home, settings, source_run, source_step,"
                " variant, experiment_id, sample_idx, status, exit, created_at, ended_at, owner)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    launch.id, launch.kind, launch.label, launch.harness, launch.model,
                    launch.task_id, json.dumps(launch.eval), launch.spec_path,
                    launch.skill_dir, launch.workspace, launch.home,
                    json.dumps(launch.settings), launch.source_run, launch.source_step,
                    json.dumps(launch.variant), launch.experiment_id, launch.sample_idx,
                    launch.status, json.dumps(launch.exit), launch.created_at, None,
                    json.dumps(launch.owner),
                ),
            )
            self._conn.commit()
        return launch

    def _launch_from_row(self, row: sqlite3.Row) -> Launch:
        return Launch(
            id=row["id"],
            kind=row["kind"],
            label=row["label"],
            harness=row["harness"],
            model=row["model"],
            task_id=row["task_id"],
            eval=_loads(row["eval"], {}),
            spec_path=row["spec_path"],
            skill_dir=row["skill_dir"],
            workspace=row["workspace"],
            home=row["home"],
            settings=_loads(row["settings"], {}),
            source_run=row["source_run"],
            source_step=row["source_step"],
            variant=_loads(row["variant"], {}),
            experiment_id=row["experiment_id"],
            sample_idx=row["sample_idx"],
            status=row["status"],
            exit=_loads(row["exit"], {}),
            created_at=row["created_at"],
            ended_at=row["ended_at"],
            initial_commit=row["initial_commit"],
            final_commit=row["final_commit"],
            skill_commit=row["skill_commit"],
            job_id=row["job_id"],
            owner=_loads(row["owner"], None),
        )

    def running_launches(self) -> list[Launch]:
        """Every launch still marked running: what the sweep reads (`jobs.JobRunner.sweep`)."""
        with self._lock:
            return [
                self._launch_from_row(row)
                for row in self._conn.execute("SELECT * FROM launches WHERE status = 'running'")
            ]

    def set_launch_snapshots(self, launch_id: str, **commits: str | None) -> None:
        """Record a launch's snapshots: `initial_commit`, `final_commit`, `skill_commit`."""
        allowed = {"initial_commit", "final_commit", "skill_commit"}
        sets = [(column, value) for column, value in commits.items() if column in allowed and value]
        if not sets:
            return
        with self._lock:
            self._conn.execute(
                f"UPDATE launches SET {', '.join(f'{column} = ?' for column, _ in sets)} WHERE id = ?",
                (*[value for _, value in sets], launch_id),
            )
            self._conn.commit()

    def note_launch(self, launch_id: str, **notes: Any) -> None:
        """Add to a launch's settings what happened to it as it ran — where a scenario's run
        stopped (`stopped_at_skill`) — for the process running it to read."""
        with self._lock:
            row = self._conn.execute("SELECT settings FROM launches WHERE id = ?", (launch_id,)).fetchone()
            if row is None:
                return
            settings = {**(_loads(row["settings"], {}) or {}), **notes}
            self._conn.execute("UPDATE launches SET settings = ? WHERE id = ?", (json.dumps(settings), launch_id))
            self._conn.commit()

    def task_checks(self) -> dict[str, list[Any]]:
        """Each task's checks, as its launches ran them, the latest first: a task's spec can
        change between batches, and a pass rate pooled over two checks measures neither."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id, eval, MAX(created_at) AS latest FROM launches"
                " WHERE task_id IS NOT NULL AND kind = 'batch' GROUP BY task_id, eval"
                " ORDER BY latest DESC"
            ).fetchall()
        found: dict[str, list[Any]] = {}
        for row in rows:
            check = ((_loads(row["eval"], {}) or {}).get("tuningfork") or {}).get("check")
            checks = found.setdefault(row["task_id"], [])
            if check not in checks:
                checks.append(check)
        return found

    def held_out_tasks(self) -> set[str]:
        """Tasks whose launches ran a held-out eval (`tuningfork.held_out`). Their runs are a
        person's evidence for judging an edit: an agent that read them could fit one to them."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT task_id, eval FROM launches WHERE task_id IS NOT NULL AND eval IS NOT NULL"
            ).fetchall()
        return {
            row["task_id"] for row in rows
            if ((_loads(row["eval"], {}) or {}).get("tuningfork") or {}).get("held_out") is True
        }

    def get_launch(self, launch_id: str) -> Launch | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM launches WHERE id = ?", (launch_id,)
            ).fetchone()
        return self._launch_from_row(row) if row else None

    def end_launch(
        self, launch_id: str, *, status: str = "ended", exit: dict[str, Any] | None = None
    ) -> None:
        """Mark a launch finished. Only a running launch changes: an interruption found by
        the proxy is not overwritten by the runner's later "ended"."""
        with self._lock:
            self._conn.execute(
                "UPDATE launches SET status = ?, exit = ?, ended_at = ?"
                " WHERE id = ? AND status = 'running'",
                (status, json.dumps(exit or {}), time.time(), launch_id),
            )
            self._conn.commit()

    def launch_runs(self, launch_id: str) -> list[Run]:
        """Every thread of a launch, the main one first, then in the order they began."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id)"
                " AS step_count FROM runs r WHERE r.launch_id = ?"
                " ORDER BY (r.role IN (?, ?)) DESC, r.started_at",
                (launch_id, MAIN, FORK),
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def threads_of(self, run_id: str) -> list[Run]:
        """A main run's subagent and side threads, in the order they began."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id)"
                " AS step_count FROM runs r WHERE r.launch_id ="
                " (SELECT launch_id FROM runs WHERE id = ?) AND r.id != ?"
                " ORDER BY r.started_at",
                (run_id, run_id),
            ).fetchall()
        return [self._run_from_row(row) for row in rows if row["launch_id"]]

    def finalize_threads(self, launch_id: str, subagent_tools: Iterable[str] = ()) -> None:
        """Tell a finished launch's secondary threads apart.

        A thread is a subagent when an earlier call to a subagent tool (`delegate_task`,
        `Agent`) spawned it: its first request carries the goal or prompt that call passed —
        verified for both harnesses. Failing that, a thread that began while its spawner was
        still waiting on such a call is taken to be its. Anything else is the harness's own
        side call.

        Decided after the fact: when a subagent's first call arrives, the call that spawned
        it may not be recorded yet. Nor is timing alone enough: both harnesses can delegate
        in the background, so a child can start after its parent has moved on.
        """
        tools = set(subagent_tools)
        runs = self.launch_runs(launch_id)
        if len(runs) < 2:
            return
        # Per subagent call: (arrival, run, step index, the texts its arguments carried,
        # when the calling thread moved on).
        spawns: list[tuple[float, str, int, list[str], float]] = []
        starts: dict[str, float] = {}
        opening: dict[str, str] = {}
        for run in runs:
            steps = self.steps(run.id)
            arrivals = [arrived(step) for step in steps]
            if steps:
                starts[run.id] = arrivals[0]
                opening[run.id] = "\n".join(_strings(self.harness_request(steps[0])))
            for position, step in enumerate(steps):
                moved_on = arrivals[position + 1] if position + 1 < len(steps) else float("inf")
                for call in _calls_to(self.delivered_response(step), tools):
                    spawns.append(
                        (arrivals[position], run.id, step.idx, _argument_texts(call), moved_on)
                    )
        main = runs[0]
        with self._lock:
            for run in sorted(runs[1:], key=lambda r: starts.get(r.id, r.started_at)):
                if run.role not in (THREAD, None):
                    continue
                start = starts.get(run.id, run.started_at)
                earlier = [s for s in spawns if s[1] != run.id and s[0] <= start]
                text = opening.get(run.id, "")
                matched = [s for s in earlier if any(t in text for t in s[3])]
                waiting = [s for s in earlier if start < s[4]]
                spawner = max(matched or waiting, default=None)
                if spawner:
                    role, parent, at = SUBAGENT, spawner[1], spawner[2]
                else:
                    role, parent, at = SIDE, main.id, None
                self._conn.execute(
                    "UPDATE runs SET role = ?, parent_run_id = ?, spawned_at = ? WHERE id = ?",
                    (role, parent, at, run.id),
                )
            self._conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        """A run by its id, its label, or an unambiguous prefix of its id — in that order.

        The order matters because the three overlap: ids are hex, so a label such as "a"
        or "b" is also a prefix of about one run id in sixteen. Matching all three at once
        once let a label resolve to an unrelated run whose id happened to start with it.
        """
        select = (
            "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id)"
            " AS step_count FROM runs r"
        )
        prefix = run_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._lock:
            row = self._conn.execute(f"{select} WHERE r.id = ?", (run_id,)).fetchone()
            if row is None:
                row = self._conn.execute(
                    f"{select} WHERE r.label = ? ORDER BY r.started_at DESC LIMIT 1", (run_id,)
                ).fetchone()
            if row is None and run_id:
                candidates = self._conn.execute(
                    f"{select} WHERE r.id LIKE ? ESCAPE '\\' LIMIT 2", (prefix,)
                ).fetchall()
                row = candidates[0] if len(candidates) == 1 else None
        return self._run_from_row(row) if row else None

    def _step_from_row(self, row: sqlite3.Row) -> Step:
        return Step(
            id=row["id"],
            run_id=row["run_id"],
            idx=row["idx"],
            request_node_id=row["request_node_id"],
            extras_blob=row["extras_blob"],
            response_blob=row["response_blob"],
            raw_response_blob=row["raw_response_blob"],
            params=_loads(row["params"], {}),
            server_defaults=_loads(row["server_defaults"], {}),
            model=row["model"],
            status=row["status"],
            finish_reason=row["finish_reason"],
            timings=_loads(row["timings"], {}),
            usage=_loads(row["usage"], {}),
            reasoning_chars=row["reasoning_chars"],
            error=row["error"],
            recorded_at=row["recorded_at"],
            edits=_loads(row["edits"], {}),
            harness_request_node_id=row["harness_request_node_id"],
            delivered_response_blob=row["delivered_response_blob"],
            wire=row["wire"],
            message_overrides=_loads(row["message_overrides"], {}),
            replay=_loads(row["replay"], {}),
            substitution=_loads(row["substitution"], {}),
            context=_loads(row["context"], {}),
            workspace_commit=row["workspace_commit"],
            server_id=row["server_id"],
        )

    def steps(self, run_id: str) -> list[Step]:
        with self._lock:
            return [
                self._step_from_row(row)
                for row in self._conn.execute(
                    "SELECT * FROM steps WHERE run_id = ? ORDER BY idx", (run_id,)
                )
            ]

    def step(self, run_id: str, idx: int) -> Step | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM steps WHERE run_id = ? AND idx = ?", (run_id, idx)
            ).fetchone()
        return self._step_from_row(row) if row else None

    def delivered_response(self, step: Step) -> dict[str, Any] | None:
        """What the harness received: the model's response unless a breakpoint edited it."""
        if step.delivered_response_blob:
            return self.blobs.get_json(step.delivered_response_blob)
        return self.response(step)

    def harness_request(self, step: Step) -> dict[str, Any]:
        """The request as the harness sent it: the model's view unless a breakpoint edited
        it, a fork swapped a variant wording into it, or the proxy's cap lowered its limit on
        output — what a fork's replay matches its own harness's requests against."""
        from tuningfork.wire.base import uncapped

        request = self.rebuild_request(step)
        cap = step.params.get("output_cap")
        if not step.harness_request_node_id:
            return uncapped(request, cap)
        changed = step.edits.get("request") or step.substitution or step.context or {}
        original_extras = changed.get("original_extras")
        extras = original_extras if original_extras is not None else {
            key: value for key, value in request.items() if key != "messages"
        }
        messages = self.messages_for_node(step.harness_request_node_id)
        for index, exact in (changed.get("harness_overrides") or {}).items():
            messages[int(index)] = exact
        return uncapped({**extras, "messages": messages}, cap)

    def response(self, step: Step) -> dict[str, Any] | None:
        return self.blobs.get_json(step.response_blob) if step.response_blob else None

    def rebuild_request(self, step: Step) -> dict[str, Any]:
        """Reconstruct the exact request that produced this step.

        Messages come from the node path; everything else from the extras blob. Asserted
        JSON-equal to the original in the test suite.
        """
        extras = self.blobs.get_json(step.extras_blob)
        messages = self.messages_for_node(step.request_node_id)
        for index, exact in step.message_overrides.items():
            messages[int(index)] = exact
        # In place when the extras kept its position (a dict keeps a replaced key's slot);
        # recordings from before that get it last.
        return {**extras, "messages": messages}

    def stats(self) -> dict[str, int]:
        """Counts of what the store holds; `forks` and `threads` are the runs `runs` leaves
        out unless asked."""
        roles = ",".join(f"'{role}'" for role in (SUBAGENT, SIDE, THREAD))
        with self._lock:
            counts = {
                name: self._conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]
                for name in ("runs", "steps", "nodes", "experiments", "samples")
            }
            counts["forks"] = self._conn.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE role = ?", (FORK,)
            ).fetchone()["n"]
            counts["threads"] = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM runs WHERE role IN ({roles})"
            ).fetchone()["n"]
            return counts

    # --- experiments --------------------------------------------------------------

    def create_experiment(
        self,
        *,
        kind: str,
        arm: str,
        run_id: str | None,
        step_idx: int | None,
        step_id: str | None,
        request: dict[str, Any],
        settings: dict[str, Any],
        changes: dict[str, Any] | None = None,
        variant: dict[str, Any] | None = None,
        base_url: str | None = None,
        server_defaults: dict[str, Any] | None = None,
        notes: list[str] | None = None,
        group_id: str | None = None,
        status: str | None = "running",
        job_id: str | None = None,
        hold_id: str | None = None,
    ) -> Experiment:
        """A new experiment, `running` until whoever runs it says how it ended
        (`finish_experiments`). One sampled at a held call has no step yet (`hold_id`):
        it is filed under the step the call becomes, whichever is recorded first."""
        from tuningfork.owners import current_owner

        experiment = Experiment(
            id=uuid.uuid4().hex,
            kind=kind,
            group_id=group_id,
            arm=arm,
            run_id=run_id,
            step_idx=step_idx,
            step_id=step_id,
            request_blob=self.blobs.put_json(request),
            settings=settings,
            changes=changes or {},
            variant=variant or {},
            base_url=base_url,
            server_defaults=server_defaults or {},
            notes=notes or [],
            created_at=time.time(),
            status=status,
            job_id=job_id,
            owner=current_owner() if status == "running" else None,
            hold_id=hold_id,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO experiments (id, kind, group_id, arm, run_id, step_idx, step_id,"
                " request_blob, settings, changes, variant, base_url, server_defaults,"
                " notes, created_at, status, job_id, owner, hold_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    experiment.id, experiment.kind, experiment.group_id, experiment.arm,
                    experiment.run_id, experiment.step_idx, experiment.step_id,
                    experiment.request_blob, json.dumps(experiment.settings),
                    json.dumps(experiment.changes), json.dumps(experiment.variant),
                    experiment.base_url, json.dumps(experiment.server_defaults),
                    json.dumps(experiment.notes), experiment.created_at,
                    experiment.status, experiment.job_id,
                    json.dumps(experiment.owner) if experiment.owner else None,
                    experiment.hold_id,
                ),
            )
            self._conn.commit()
            if hold_id is not None:
                self._relink([hold_id])  # the call may have been recorded already
        return self.get_experiment(experiment.id) or experiment

    def link_holds(self, hold_ids: Sequence[str], step: Step, *, edited: bool) -> None:
        """The step these holds' call became. What was sampled at them is filed under it."""
        if not hold_ids:
            return
        with self._lock:
            for hold_id in hold_ids:
                self._conn.execute(
                    "INSERT OR REPLACE INTO hold_steps (hold_id, step_id, run_id, step_idx, edited)"
                    " VALUES (?,?,?,?,?)",
                    (hold_id, step.id, step.run_id, step.idx, int(edited)),
                )
            self._conn.commit()
            self._relink(list(hold_ids))

    def hold_step(self, hold_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM hold_steps WHERE hold_id = ?", (hold_id,)).fetchone()
        return dict(row) if row else None

    def _relink(self, hold_ids: list[str]) -> None:
        """Give experiments sampled at these holds the step their call became, if it has
        been recorded — and say whether what was sampled is what the model then saw."""
        marks = ",".join("?" * len(hold_ids))
        with self._lock:
            links = {
                row["hold_id"]: row
                for row in self._conn.execute(f"SELECT * FROM hold_steps WHERE hold_id IN ({marks})", hold_ids)
            }
            if not links:
                return
            rows = self._conn.execute(
                f"SELECT id, hold_id, settings FROM experiments WHERE run_id IS NULL AND hold_id IN ({marks})",
                hold_ids,
            ).fetchall()
            for row in rows:
                link = links.get(row["hold_id"])
                if link is None:
                    continue
                settings = _loads(row["settings"], {})
                origin = dict(settings.get("origin") or {})
                # Sampled before the model: the request as held. If a person then edited
                # it, the model saw something else, and the samples are not that step's.
                origin["matches_step"] = not (origin.get("phase") == "request" and link["edited"])
                settings["origin"] = origin
                self._conn.execute(
                    "UPDATE experiments SET run_id = ?, step_idx = ?, step_id = ?, settings = ? WHERE id = ?",
                    (link["run_id"], link["step_idx"], link["step_id"], json.dumps(settings), row["id"]),
                )
            self._conn.commit()

    def finish_experiments(self, ids: list[str], status: str) -> None:
        """Say how running experiments ended. One that already has an ending keeps it."""
        if not ids:
            return
        with self._lock:
            self._conn.execute(
                f"UPDATE experiments SET status = ?, ended_at = ?"
                f" WHERE status = 'running' AND id IN ({','.join('?' * len(ids))})",
                (status, time.time(), *ids),
            )
            self._conn.commit()

    def set_experiment_status(self, ids: list[str], status: str) -> None:
        """Correct how experiments ended — cancelled by a shutdown reads as interrupted."""
        if not ids:
            return
        with self._lock:
            self._conn.execute(
                f"UPDATE experiments SET status = ?, ended_at = COALESCE(ended_at, ?)"
                f" WHERE id IN ({','.join('?' * len(ids))})",
                (status, time.time(), *ids),
            )
            self._conn.commit()

    def set_experiment_job(self, experiment_id: str, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE experiments SET job_id = ? WHERE id = ?", (job_id, experiment_id)
            )
            self._conn.commit()

    def set_launch_job(self, launch_id: str, job_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE launches SET job_id = ? WHERE id = ?", (job_id, launch_id))
            self._conn.commit()

    def add_sample(
        self,
        experiment_id: str,
        idx: int,
        *,
        seed: int | None,
        response: dict[str, Any] | None,
        outcome_kind: str | None,
        outcome_key: str | None,
        outcome_label: str | None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
        timings: dict[str, Any] | None = None,
        reasoning_chars: int = 0,
        error: str | None = None,
        run_id: str | None = None,
        passed: bool | None = None,
        exclusion: str | None = None,
        forced: dict[str, Any] | None = None,
    ) -> Sample:
        sample = Sample(
            id=uuid.uuid4().hex,
            experiment_id=experiment_id,
            idx=idx,
            seed=seed,
            response_blob=self.blobs.put_json(response) if response is not None else None,
            outcome_kind=outcome_kind,
            outcome_key=outcome_key,
            outcome_label=outcome_label,
            finish_reason=finish_reason,
            usage=usage or {},
            timings=timings or {},
            reasoning_chars=reasoning_chars,
            error=error,
            created_at=time.time(),
            run_id=run_id,
            passed=passed,
            exclusion=exclusion,
            forced=forced,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO samples (id, experiment_id, idx, seed, response_blob,"
                " outcome_kind, outcome_key, outcome_label, finish_reason, usage, timings,"
                " reasoning_chars, error, created_at, run_id, passed, exclusion, forced)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sample.id, sample.experiment_id, sample.idx, sample.seed,
                    sample.response_blob, sample.outcome_kind, sample.outcome_key,
                    sample.outcome_label, sample.finish_reason, json.dumps(sample.usage),
                    json.dumps(sample.timings), sample.reasoning_chars, sample.error,
                    sample.created_at, sample.run_id,
                    None if sample.passed is None else int(sample.passed), sample.exclusion,
                    json.dumps(forced) if forced is not None else None,
                ),
            )
            self._conn.commit()
        return sample

    # --- token streams --------------------------------------------------------------

    def add_token_stream(
        self,
        stream: dict[str, Any],
        *,
        source: str,
        step_id: str | None = None,
        sample_id: str | None = None,
        server_id: str | None = None,
        method: str | None = None,
        exact: bool | None = None,
        notes: dict[str, Any] | None = None,
    ) -> TokenStreamRecord:
        """Store a token stream in its JSON form (`TokenStream.to_json`), already redacted."""
        record = TokenStreamRecord(
            id=uuid.uuid4().hex,
            step_id=step_id,
            sample_id=sample_id,
            server_id=server_id,
            source=source,
            method=method,
            top_k=stream.get("k"),
            blob=self.blobs.put_json(stream),
            tokens=len(stream.get("ids") or []),
            exact=exact,
            notes=notes or {},
            created_at=time.time(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO token_streams (id, step_id, sample_id, server_id, source, method,"
                " top_k, blob, tokens, exact, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id, step_id, sample_id, server_id, source, method, record.top_k,
                    record.blob, record.tokens, None if exact is None else int(exact),
                    json.dumps(record.notes), record.created_at,
                ),
            )
            self._conn.commit()
        return record

    def _token_stream_from_row(self, row: sqlite3.Row) -> TokenStreamRecord:
        return TokenStreamRecord(
            id=row["id"],
            step_id=row["step_id"],
            sample_id=row["sample_id"],
            server_id=row["server_id"],
            source=row["source"],
            method=row["method"],
            top_k=row["top_k"],
            blob=row["blob"],
            tokens=row["tokens"],
            exact=None if row["exact"] is None else bool(row["exact"]),
            notes=_loads(row["notes"], {}),
            created_at=row["created_at"],
        )

    def token_streams(
        self,
        *,
        step_id: str | None = None,
        sample_ids: Sequence[str] | None = None,
        source: str | None = None,
    ) -> list[TokenStreamRecord]:
        """Token streams of a step, or of samples, newest first."""
        clauses, values = [], []
        if step_id is not None:
            clauses.append("step_id = ?")
            values.append(step_id)
        if sample_ids is not None:
            if not sample_ids:
                return []
            clauses.append(f"sample_id IN ({','.join('?' * len(sample_ids))})")
            values.extend(sample_ids)
        if source is not None:
            clauses.append("source = ?")
            values.append(source)
        query = "SELECT * FROM token_streams"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY created_at DESC", values).fetchall()
        return [self._token_stream_from_row(row) for row in rows]

    def token_stream(self, stream_id: str) -> TokenStreamRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM token_streams WHERE id = ?", (stream_id,)
            ).fetchone()
        return self._token_stream_from_row(row) if row else None

    def token_stream_data(self, record: TokenStreamRecord) -> dict[str, Any]:
        return self.blobs.get_json(record.blob)

    def steps_with_token_streams(self, step_ids: Sequence[str]) -> dict[str, TokenStreamRecord]:
        """The latest scored or recorded stream of each of these steps that has one: one query
        for a whole run's timeline."""
        if not step_ids:
            return {}
        placeholders = ",".join("?" * len(step_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM token_streams WHERE step_id IN ({placeholders})"
                " AND source IN ('scored', 'recorded') ORDER BY created_at",
                list(step_ids),
            ).fetchall()
        return {row["step_id"]: self._token_stream_from_row(row) for row in rows}

    def _experiment_from_row(self, row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            kind=row["kind"],
            group_id=row["group_id"],
            arm=row["arm"],
            run_id=row["run_id"],
            step_idx=row["step_idx"],
            step_id=row["step_id"],
            request_blob=row["request_blob"],
            settings=_loads(row["settings"], {}),
            changes=_loads(row["changes"], {}),
            variant=_loads(row["variant"], {}),
            base_url=row["base_url"],
            server_defaults=_loads(row["server_defaults"], {}),
            notes=_loads(row["notes"], []),
            created_at=row["created_at"],
            sample_count=row["sample_count"] if "sample_count" in row.keys() else 0,
            status=row["status"],
            job_id=row["job_id"],
            ended_at=row["ended_at"],
            owner=_loads(row["owner"], None),
            hold_id=row["hold_id"],
            trashed_at=row["trashed_at"],
        )

    _EXPERIMENT_SELECT = (
        "SELECT e.*, (SELECT COUNT(*) FROM samples s WHERE s.experiment_id = e.id)"
        " AS sample_count FROM experiments e"
    )

    def experiments(
        self,
        *,
        run_id: str | None = None,
        step_idx: int | None = None,
        kind: str | None = None,
        limit: int = 50,
        include_trashed: bool = False,
    ) -> list[Experiment]:
        """Experiments, newest first; those in the trash left out unless `include_trashed`."""
        clauses: list[str] = [] if include_trashed else ["e.trashed_at IS NULL"]
        values: list[Any] = []
        for column, value in (("run_id", run_id), ("step_idx", step_idx), ("kind", kind)):
            if value is not None:
                clauses.append(f"e.{column} = ?")
                values.append(value)
        query = self._EXPERIMENT_SELECT
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY e.created_at DESC LIMIT ?"
        values.append(limit)
        with self._lock:
            return [self._experiment_from_row(row) for row in self._conn.execute(query, values)]

    def get_experiment(self, reference: str) -> Experiment | None:
        """An experiment by its id or an unambiguous prefix of it, as `get_run` does."""
        prefix = reference.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._lock:
            row = self._conn.execute(
                self._EXPERIMENT_SELECT + " WHERE e.id = ?", (reference,)
            ).fetchone()
            if row is None and reference:
                candidates = self._conn.execute(
                    self._EXPERIMENT_SELECT + " WHERE e.id LIKE ? ESCAPE '\\' LIMIT 2",
                    (prefix,),
                ).fetchall()
                row = candidates[0] if len(candidates) == 1 else None
        return self._experiment_from_row(row) if row else None

    def experiment_group(self, group_id: str) -> list[Experiment]:
        """Both arms of an A/B, original first."""
        with self._lock:
            rows = self._conn.execute(
                self._EXPERIMENT_SELECT + " WHERE e.group_id = ? ORDER BY e.created_at",
                (group_id,),
            ).fetchall()
        experiments = [self._experiment_from_row(row) for row in rows]
        return sorted(experiments, key=lambda e: (e.arm != "original", e.created_at))

    def _sample_from_row(self, row: sqlite3.Row) -> Sample:
        return Sample(
            id=row["id"],
            experiment_id=row["experiment_id"],
            idx=row["idx"],
            seed=row["seed"],
            response_blob=row["response_blob"],
            outcome_kind=row["outcome_kind"],
            outcome_key=row["outcome_key"],
            outcome_label=row["outcome_label"],
            finish_reason=row["finish_reason"],
            usage=_loads(row["usage"], {}),
            timings=_loads(row["timings"], {}),
            reasoning_chars=row["reasoning_chars"],
            error=row["error"],
            created_at=row["created_at"],
            run_id=row["run_id"],
            passed=None if row["passed"] is None else bool(row["passed"]),
            exclusion=row["exclusion"],
            forced=_loads(row["forced"], None),
        )

    def samples(self, experiment_id: str) -> list[Sample]:
        with self._lock:
            return [
                self._sample_from_row(row)
                for row in self._conn.execute(
                    "SELECT * FROM samples WHERE experiment_id = ? ORDER BY idx",
                    (experiment_id,),
                )
            ]

    def sample_response(self, sample: Sample) -> dict[str, Any] | None:
        return self.blobs.get_json(sample.response_blob) if sample.response_blob else None

    def experiment_request(self, experiment: Experiment) -> dict[str, Any]:
        return self.blobs.get_json(experiment.request_blob)

    # --- jobs ---------------------------------------------------------------------

    def create_job(
        self,
        *,
        kind: str,
        lane: str,
        params: dict[str, Any],
        request_id: str | None = None,
        started_by: dict[str, Any] | None = None,
    ) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex, kind=kind, lane=lane, status="queued", params=params,
            plan=None, progress={}, result=None, error=None, links={}, owner=None,
            request_id=request_id, created_at=time.time(), started_at=None, ended_at=None,
            started_by=started_by,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, lane, status, params, progress, links, request_id,"
                " created_at, started_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id, kind, lane, job.status, json.dumps(params), "{}", "{}", request_id,
                    job.created_at, json.dumps(started_by) if started_by else None,
                ),
            )
            self._conn.commit()
        return job

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Store what changed about a job. JSON columns take their Python value."""
        sets, values = [], []
        for column, value in fields.items():
            if column in _JOB_JSON:
                sets.append(f"{column} = ?")
                values.append(None if value is None else json.dumps(value, default=str))
            elif column in _JOB_PLAIN:
                sets.append(f"{column} = ?")
                values.append(value)
            else:
                raise ValueError(f"a job has no column {column!r}")
        if not sets:
            return
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", (*values, job_id))
            self._conn.commit()

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            kind=row["kind"],
            lane=row["lane"],
            status=row["status"],
            params=_loads(row["params"], {}),
            plan=_loads(row["plan"], None),
            progress=_loads(row["progress"], {}),
            result=_loads(row["result"], None),
            error=_loads(row["error"], None),
            links=_loads(row["links"], {}),
            owner=_loads(row["owner"], None),
            request_id=row["request_id"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            trashed_at=row["trashed_at"],
            started_by=_loads(row["started_by"], None),
        )

    def get_job(self, reference: str) -> JobRecord | None:
        """A job by its id or an unambiguous prefix of it."""
        prefix = reference.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (reference,)).fetchone()
            if row is None and reference:
                candidates = self._conn.execute(
                    "SELECT * FROM jobs WHERE id LIKE ? ESCAPE '\\' LIMIT 2", (prefix,)
                ).fetchall()
                row = candidates[0] if len(candidates) == 1 else None
        return self._job_from_row(row) if row else None

    def job_for_request(self, request_id: str) -> JobRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE request_id = ? ORDER BY created_at DESC LIMIT 1",
                (request_id,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def jobs(self, *, limit: int = 50, unfinished: bool = False) -> list[JobRecord]:
        query = "SELECT * FROM jobs WHERE trashed_at IS NULL"
        if unfinished:
            query += " AND status IN ('queued', 'running')"
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._lock:
            return [self._job_from_row(row) for row in self._conn.execute(query, (limit,))]

    def interrupt_job_work(self, job_id: str) -> None:
        """A job cut off with its process: what it left running reads as interrupted."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE experiments SET status = 'interrupted', ended_at = ?"
                " WHERE job_id = ? AND status = 'running'",
                (now, job_id),
            )
            self._conn.execute(
                "UPDATE launches SET status = 'interrupted', ended_at = ?"
                " WHERE job_id = ? AND status = 'running'",
                (now, job_id),
            )
            self._conn.commit()

    def running_experiments(self) -> list[Experiment]:
        with self._lock:
            return [
                self._experiment_from_row(row)
                for row in self._conn.execute(
                    self._EXPERIMENT_SELECT + " WHERE e.status = 'running'"
                )
            ]
