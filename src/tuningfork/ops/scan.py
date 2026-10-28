"""What an older tuningfork kept in clear: a recordings folder read for what today's masking would
mask, and where it stands, run by run — never the key itself (`scan`; `tuningfork scan`).

A tuningfork before 0.6.0 kept a key in clear wherever its masking did not reach yet: an answer's
raw bytes where a stream sent the key in pieces, a server's errors, an imported chat, a check's
output, a job's results, and every format its patterns missed; one of 0.6 a little — the files
written from an imported chat. Nothing is masked again: a message's id is a digest of its content
and of its parent's id, so masking one changes the id of every message after it, and what an older
one masked too broadly is lost either way. What goes is the person's choice, by the trash
(`ops.trash`), each finding given the command the trash takes (`ScanReport.commands`).

It runs only when a person asks, and only reads. What it writes is its mark (`SCANNED`), which
silences what `serve` and `ui` say while the folder holds runs or jobs older than the first time a
tuningfork that keeps marks opened it (`older`): a count of each, no recording read.

It reads what the store keeps masked — every row, but what a run is given, kept as it was given (a
launch's eval and variant); every blob a record names, but a scenario's context — and the files
kept beside them made from what was recorded, where what they record says so: an imported chat and
each skill it extracted from it, a skill added from what a run's model read with what an approval
kept beside it (forgotten too), a chain's record, what a refinement keeps but what a person gave
it (`GIVEN_TO_A_REFINEMENT`, a person's own proposal), a scenario's conversation and its spec but
what that takes from its run's task as written — in the trash too, whether a run or a task file
made it, which nothing records — and its copy of a skill where that is one of the skills read as
made from a conversation. Each is read as the store masks it (`Masking.of`), an answer's raw bytes
also as its wire reads them back whole, a token stream as its tokens join. What a mask takes is
looked for wherever else it stands, as the store looks for it in a step's other forms, and so is
every key tuningfork holds (`access.held_secrets`). Never what is kept as it was given, whatever
version made it: a launch's folders, the workspace's snapshots, a scenario's inputs and any other
copy of a skill, a skill a person brought, the wordings tried — but a skill added from a run that
is gone, as the copy that run kept, which may have been what it was given: which it was is gone
with the run (`_added_from_a_run`). What no pattern reads, it cannot find (`NOT_FOUND`).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from tuningfork.ops.common import Cancel, Progress, emit
from tuningfork.store.db import FORK, MAIN, OPENED, SCANNED, Store

#: What a run is given, kept as it was given, since a fork runs it again: never read.
GIVEN = frozenset({("launches", "eval"), ("launches", "variant")})
#: Where a server's address is kept, as `config.without_credentials` keeps it today: masked whole
#: before its host, and each value of its query and its fragment. One kept before held them.
ADDRESSES = frozenset({("servers", "identity"), ("experiments", "base_url")})
#: What a scenario's spec takes from the task of the run it was made of, as the task's file wrote
#: it: its check, and the output and expectations it names. The rest of the spec is read.
GIVEN_TO_A_SCENARIO = ("check", "expected_output", "expectations")
#: What a refinement keeps as it was given: the skill it refines, as the run was given it; the
#: edit a person approved, and what that applied.
GIVEN_TO_A_REFINEMENT = ("baseline.md", "approval.json", "applied.json")
#: The store's tables that hold no recording: the patterns steps were masked with, the folder's
#: marks, the message tree's ids and hashes (its messages are read as their blobs), a held
#: call's step.
NOT_RECORDED = frozenset({"maskings", "marks", "nodes", "hold_steps"})
#: The store's tables of recordings, each read in this order: what each names is known by the time
#: a later one names it — a launch's run, a sample's experiment, a token stream's step. Each row is
#: a run's where it names one (`_Reader._owner`).
OWNED = ("runs", "steps", "experiments", "samples", "launches", "token_streams", "jobs", "servers", "pending_labels")
#: Where a run holds what today's masking would mask, as the report names it, in its order.
KINDS = {
    "messages": "its messages",
    "answers": "its answers",
    "errors": "its errors",
    "result": "its check's result",
    "records": "its records",
    "experiments": "experiments at its steps",
    "chat": "its imported chat",
}
#: What the trash takes, in the order the report gives its commands (`ScanReport.commands`).
TRASHED_AS = ("run", "experiment", "job", "refinement", "scenario")
#: What else holds some and goes by hand, where: the trash takes none of them.
BY_HAND = {"chain": "chains/{}.json", "import": "imports/{}/", "skill": "skills/{}/"}
#: What is never read, whatever version made it.
NOT_READ = ("what is kept as it was given, by design: a launch's folders, the workspace's snapshots, a "
            "scenario's inputs and a copy of a skill it was given, a skill a person brought, the wordings tried")
#: What no pattern reads, which the scan cannot find either.
NOT_FOUND = ("a key no pattern reads: the end of one a check's output was cut through, as 0.6 kept it, or "
             "the value of a variable named as a secret's, printed bare")
#: How many places read between two progress events.
EVERY = 500
#: What the report counts as read, by what each place read is.
READ = {"row": "records", "blob": "blobs", "message": "messages", "file": "files"}


def older(store: Store) -> dict[str, Any] | None:
    """What of the folder an older tuningfork recorded, while no scan has read it since: the runs
    started, and jobs made, before a tuningfork that keeps marks first opened it (`OPENED`) — a
    count of each, no recording read. None where there is none, or a scan has run."""
    opened = store.mark(OPENED)
    if opened is None or store.mark(SCANNED) is not None:
        return None
    runs, jobs = store.older_than(opened)
    return {"runs": runs, "jobs": jobs, "opened": opened} if runs or jobs else None


def older_note(store: Store, *, command: str = "tuningfork scan") -> str | None:
    """What `serve` and `ui` say of `older`, naming the `command` that scans the folder."""
    found = older(store)
    if not found:
        return None
    what = " and ".join(_counted(found[name], name[:-1]) for name in ("runs", "jobs") if found[name])
    were = "was" if found["runs"] + found["jobs"] == 1 else "were"
    return (f"{what} here {were} recorded by an older tuningfork, which masked fewer keys: "
            f"`{command}` looks for what today's masking would mask in them, and changes no recording.")


def _counted(count: int, name: str) -> str:
    return f"{count:,} {name}" + ("" if count == 1 else "s")


@dataclass(frozen=True, slots=True)
class Place:
    """Where today's masking would mask something, as the report counts it: whose (`owner`, `id`)
    — a run, or a job, a server, a chain, a refinement, a scenario, a skill, a queued label — what
    of it (`kind`, one of `KINDS` for a run), and at which step, where one is named."""

    owner: str
    id: str
    kind: str
    step: int | None = None


@dataclass(slots=True)
class _Unit:
    """One text read: a row's value, a blob as one reader reads it, a file. `read` gives what it
    holds as text, or raises OSError; `holds` how many masks it would take, of the patterns and of
    `also`, and the values those took. Its places are kept in the order they came, each once."""

    key: tuple[Any, ...]
    places: dict[Place, None]
    read: Callable[[], Any]
    holds: Callable[[Any, tuple[str, ...]], tuple[int, frozenset[str]]]
    #: An answer the harness was given: a fork past its step gives it again.
    answer: bool = False


@dataclass
class ScanReport:
    """What the scan found: the runs holding what today's masking would mask, each with where and
    how often; what else holds some; what it could not read; how much it read."""

    home: str
    runs: list[dict[str, Any]] = field(default_factory=list)
    elsewhere: list[dict[str, Any]] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)
    read: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    #: What `older` said before the scan, which it silences.
    older: dict[str, Any] | None = None

    @property
    def found(self) -> bool:
        return bool(self.runs or self.elsewhere)

    def commands(self) -> list[dict[str, Any]]:
        """What sets each finding aside, as the trash takes it (`TRASHED_AS`): each kind once, its
        items in the report's order — none in the trash already, none still running."""
        ids: dict[str, list[str]] = {}
        for found in [*self.runs, *self.elsewhere]:
            aside = found.get("aside")
            if aside and not found.get("trashed") and not found.get("running"):
                listed = ids.setdefault(aside["kind"], [])
                if aside["id"] not in listed:
                    listed.append(aside["id"])
        return [{"kind": kind, "ids": ids[kind]} for kind in TRASHED_AS if ids.get(kind)]

    def to_json(self) -> dict[str, Any]:
        return {
            "home": self.home, "found": self.found, "runs": self.runs, "elsewhere": self.elsewhere,
            "commands": self.commands(), "unread": self.unread, "read": self.read,
            "seconds": round(self.seconds, 1), "older": self.older, "not_read": NOT_READ, "not_found": NOT_FOUND,
        }


def scan(store: Store, *, progress: Progress | None = None, cancel: Cancel | None = None) -> ScanReport:
    """Read the folder whole for what today's masking would mask (see the module), and mark it
    scanned. Changes no recording."""
    started = time.monotonic()
    report = ScanReport(home=str(store.home), older=older(store))
    reader = _Reader(store, progress, cancel)
    reader.run()
    report.read = dict(reader.read)
    report.unread = reader.unread
    report.runs, report.elsewhere = reader.found()
    report.seconds = time.monotonic() - started
    store.set_mark(SCANNED)
    return report


class _Reader:
    def __init__(self, store: Store, progress: Progress | None, cancel: Cancel | None) -> None:
        from tuningfork.access import held_secrets

        self.store, self.masking, self.progress, self.cancel = store, store.masking, progress, cancel
        #: Every key tuningfork holds: looked for wherever it stands. Not the values of this
        #: process's variables named as secrets': found in review, a local default
        #: (`PGPASSWORD=postgres`) made a word a key in every recording that said it.
        self.held = tuple(held_secrets(store.home))
        #: The SKILL.md of each skill read as made from a conversation, by its digest: a scenario's
        #: copy that is one of them was made from it too (`_files`).
        self.extracts: set[str] = set()
        #: The values a mask took somewhere: looked for wherever else they stand, as a step's are.
        self.taken: set[str] = set()
        self.hits: dict[tuple[Any, ...], tuple[_Unit, int]] = {}
        self.unread: list[str] = []
        self.read: Counter[str] = Counter()
        self.done = 0
        # What each record names, as the tables are read in order (`OWNED`).
        self.step_of: dict[str, tuple[str, int]] = {}
        self.experiment_of: dict[str, tuple[str | None, int | None, str]] = {}
        self.sample_of: dict[str, str] = {}
        self.run_of_launch: dict[str, str] = {}
        self.blobs: dict[tuple[str, str], _Unit] = {}
        self.messages: list[str] = []
        #: What else is in the trash, or still marked running, by its owner: no command for it.
        self.trashed: dict[str, set[str]] = {}
        self.running: set[tuple[str, str]] = set()

    # --- the two passes -------------------------------------------------------------------

    def run(self) -> None:
        units = self._units
        for unit in units():
            self._check(unit, self.held)
        if self.taken - set(self.held):
            # What a mask took, looked for where nothing was found yet: a password named in one
            # place and bare in another — an error that quotes it, the model's reasoning.
            also = (*self.held, *sorted(self.taken))
            for unit in units(again=True):
                if unit.key not in self.hits:
                    self._check(unit, also, only_where=self.taken)

    def _check(self, unit: _Unit, also: tuple[str, ...], *, only_where: set[str] | None = None) -> None:
        if self.cancel is not None:
            self.cancel.check()
        try:
            text = unit.read()
            if only_where is not None and not any(value in _as_text(text) for value in only_where):
                return
            made, taken = unit.holds(text, also)
        except (OSError, ValueError, KeyError, TypeError) as error:
            if only_where is None:
                self.unread.append(_unread(unit, error))
            return
        if only_where is None:
            if unit.key[0] != "row":  # a row is counted once, as it is read (`_row`)
                self.read[READ[unit.key[0]]] += 1
            self.done += 1
            if self.done % EVERY == 0:
                emit(self.progress, "scan_read", read=self.done)
        if made:
            self.hits[unit.key] = (unit, made)
            self.taken.update(taken)

    # --- what is read ---------------------------------------------------------------------

    def _units(self, *, again: bool = False) -> Iterator[_Unit]:
        """Every text the scan reads: the rows table by table, the blobs they name, the
        messages, the files. Read again (`again`), what the rows name is known already."""
        tables = [table for table in self.store.tables() if table not in NOT_RECORDED]
        ordered = [table for table in OWNED if table in tables] + [table for table in tables if table not in OWNED]
        emit(self.progress, "scan_phase", phase="records", again=again)
        for table in ordered:
            for row in self.store.table_rows(table):
                yield from self._row(table, row, again=again)
        emit(self.progress, "scan_phase", phase="blobs", total=len(self.blobs), again=again)
        yield from list(self.blobs.values())
        if not again:
            self.messages = self.store.message_blobs()
        emit(self.progress, "scan_phase", phase="messages", total=len(self.messages), again=again)
        for blob in self.messages:
            yield _Unit(("message", blob), {}, _blob_text(self.store, blob), self._text)
        emit(self.progress, "scan_phase", phase="files", again=again)
        yield from self._files()

    def _row(self, table: str, row: dict[str, Any], *, again: bool) -> Iterator[_Unit]:
        owner, number = self._owner(table, row)
        if not again:
            self._names(table, row, owner, number)
            self.read[READ["row"]] += 1
        for column, value in row.items():
            if column == "_rowid" or (table, column) in GIVEN or not isinstance(value, str) or not value:
                continue
            kind = self._stream_kind(row, owner) if table == "token_streams" else _kind(table, column, owner[0])
            place = Place(*owner, kind=kind, step=number)
            holds = self._address if (table, column) in ADDRESSES else self._text
            yield _Unit(("row", table, row["_rowid"], column), {place: None}, lambda value=value: value, holds)

    def _owner(self, table: str, row: dict[str, Any]) -> tuple[tuple[str, str], int | None]:
        """Whose a row is, and at which step: a run's, where it names one; else its own."""
        if table == "runs":
            return ("run", row["id"]), None
        if table == "steps":
            return ("run", row["run_id"]), row["idx"]
        if table == "experiments":
            return (("run", row["run_id"]) if row.get("run_id") else ("experiment", row["id"])), row.get("step_idx")
        if table == "samples":
            run, step, experiment = self.experiment_of.get(row["experiment_id"], (None, None, row["experiment_id"]))
            return (("run", run) if run else ("experiment", experiment)), step
        if table == "launches":
            run = self.run_of_launch.get(row["id"])
            return (("run", run) if run else ("launch", row["id"])), None
        if table == "token_streams":
            if row.get("step_id") in self.step_of:
                run, step = self.step_of[row["step_id"]]
                return ("run", run), step
            run, step, experiment = self.experiment_of.get(self.sample_of.get(row.get("sample_id") or "", ""),
                                                           (None, None, row.get("sample_id") or row["id"]))
            return (("run", run) if run else ("experiment", experiment)), step
        if table == "jobs":
            return ("job", row["id"]), None
        if table == "servers":
            return ("server", row["id"]), None
        if table == "pending_labels":
            return ("label", str(row["_rowid"])), None
        return ("record", table), None

    def _names(self, table: str, row: dict[str, Any], owner: tuple[str, str], step: int | None) -> None:
        """What a row names: the ids later tables are known by, the blobs it refers to, and
        whether it is in the trash or still running."""
        if table == "runs" and row.get("launch_id"):
            # A launch's record is its main run's, never a thread's that began first.
            if row.get("role") in (MAIN, FORK) or row["launch_id"] not in self.run_of_launch:
                self.run_of_launch[row["launch_id"]] = row["id"]
        elif table == "steps":
            self.step_of[row["id"]] = (row["run_id"], row["idx"])
            for column in ("extras_blob", "response_blob", "delivered_response_blob"):
                self._blob(row.get(column), Place(*owner, kind="messages" if column == "extras_blob" else "answers",
                                                  step=step), answer=column != "extras_blob")
            if row.get("raw_response_blob"):
                params = _loaded(row.get("params"))
                self._blob(row["raw_response_blob"], Place(*owner, kind="answers", step=step), answer=True,
                           raw=(row.get("wire"), params.get("streamed") if isinstance(params, dict) else None))
        elif table == "experiments":
            self.experiment_of[row["id"]] = (row.get("run_id"), row.get("step_idx"), row["id"])
            if row.get("trashed_at"):
                self.trashed.setdefault("experiment", set()).add(row["id"])
            self._blob(row.get("request_blob"), Place(*owner, kind="experiments", step=step))
        elif table == "samples":
            self.sample_of[row["id"]] = row["experiment_id"]
            self._blob(row.get("response_blob"), Place(*owner, kind=_kind(table, "", owner[0]), step=step))
        elif table == "token_streams":
            self._blob(row.get("blob"), Place(*owner, kind=self._stream_kind(row, owner), step=step), tokens=True)
        elif table == "jobs":
            if row.get("trashed_at"):
                self.trashed.setdefault("job", set()).add(row["id"])
            if row.get("status") in ("queued", "running"):
                self.running.add(("job", row["id"]))

    def _stream_kind(self, row: dict[str, Any], owner: tuple[str, str]) -> str:
        """A token stream's: a step's answer, or a sample's, which an experiment drew."""
        if row.get("step_id") in self.step_of:
            return "answers"
        return "experiments" if owner[0] == "run" else "record"

    def _blob(self, blob: str | None, place: Place, *, answer: bool = False,
              raw: tuple[str | None, bool | None] | None = None, tokens: bool = False) -> None:
        if not blob:
            return
        how = "raw" if raw is not None else "tokens" if tokens else "text"
        unit = self.blobs.get((blob, how))
        if unit is None:
            holds = (self._raw(*raw) if raw is not None else self._tokens if tokens else self._text)
            unit = self.blobs[(blob, how)] = _Unit(("blob", blob, how), {}, _blob_text(self.store, blob), holds)
        unit.places[place] = None
        unit.answer = unit.answer or answer

    def _files(self) -> Iterator[_Unit]:
        """The files kept beside the store made from what was recorded: an imported chat and each
        skill it extracted; a skill added from what a run's model read, forgotten too; a chain's
        record; what a refinement keeps but what a person gave it; a scenario's conversation and
        its spec, in the trash too, and its copy of a skill where that is one read as made from a
        conversation."""
        from tuningfork.ops.addskill import FORGOTTEN

        home = self.store.home
        runs = {run for run, _ in self.step_of.values()}
        for folder in _folders(home / "imports"):
            place = Place("run" if folder.name in runs else "import", folder.name, "chat")
            yield from self._file(folder / "chat.json", place)
            for path in self._extracted(folder, folder.name in runs):
                yield from self._file(path, place, extract=True)
        skills = home / "skills"
        for record in sorted(skills.glob("*.json")) if skills.is_dir() else ():
            if self._added_from_a_run(record):
                place = Place("skill", record.stem, "record")
                yield from self._file(record.with_suffix("") / "SKILL.md", place, extract=True)
                # What an approval kept beside it: the text it replaced, its lines in the changelog.
                for path in [*sorted((skills / f"{record.stem}.history").glob("*.md")), skills / f"{record.stem}.changelog.md"]:
                    yield from self._file(path, place)
        for folder in _folders(skills / FORGOTTEN):  # forgotten: its copy, its record beside it
            if self._added_from_a_run(folder / "record.json"):
                place = Place("skill", f"{FORGOTTEN}/{folder.name}", "record")
                for path in sorted(folder.glob("*/SKILL.md")):
                    yield from self._file(path, place, extract=True)
                for path in [*sorted(folder.glob("*.history/*.md")), *sorted(folder.glob("*.changelog.md"))]:
                    yield from self._file(path, place)
        for path in sorted((home / "chains").glob("*.json")) if (home / "chains").is_dir() else ():
            yield from self._file(path, Place("chain", path.stem, "record"))
        for folder in _folders(home / "refinements"):
            if (folder / "trashed.json").is_file():
                self.trashed.setdefault("refinement", set()).add(folder.name)
            for path in sorted(folder.rglob("*")):
                given = path.relative_to(folder).as_posix() in GIVEN_TO_A_REFINEMENT
                if not given and not (path.parent.name == "proposals" and _a_person_s(path)):
                    yield from self._file(path, Place("refinement", folder.name, "record"))
        for folder in _folders(home / "scenarios"):
            scenarios = [(folder, False)] if folder.name != ".trash" else [(each, True) for each in _folders(folder)]
            for scenario, trashed in scenarios:
                if trashed:  # in the trash, by the name the trash knows it by
                    self.trashed.setdefault("scenario", set()).add(scenario.name)
                place = Place("scenario", scenario.name, "record")
                yield from self._file(scenario / "context.json", place)
                yield from self._file(scenario / "evals.json", place, read=_spec_text)
                # Its copy of a skill, where it is one read as made from a conversation: a copy of
                # what a run was given, or a task's folder's, is kept as given.
                for path in sorted(scenario.glob("skill/*/SKILL.md")):
                    if _digest(path) in self.extracts:
                        yield from self._file(path, place)

    def _file(self, path: Path, place: Place, *, read: Callable[[Path], Callable[[], Any]] | None = None,
              extract: bool = False) -> Iterator[_Unit]:
        if path.is_file() and not path.is_symlink():
            digest = _digest(path) if extract else None
            if digest is not None:
                self.extracts.add(digest)
            yield _Unit(("file", str(path)), {place: None}, (read or _file_text)(path), self._text)

    def _extracted(self, folder: Path, recorded: bool) -> list[Path]:
        """The SKILL.md of each skill an import extracted from its chat — never one it copied from
        a folder the person named, kept as given. Of a chat whose run is gone, each it holds: what
        it kept of which is gone with the run."""
        run = self.store.get_run(folder.name) if recorded else None
        imported = ((run.parity or {}).get("imported") or {}) if run is not None else {}
        if not imported:
            return sorted((folder / "skills").glob("*/SKILL.md"))
        listed = imported.get("skills")
        if isinstance(listed, list):  # each skill it kept, and where it came from
            names = [each.get("name") for each in listed if isinstance(each, dict) and each.get("source") == "extracted"]
            kept = (run.parity or {}).get("skill_dirs") or {}
            return [folder / "skills" / Path(kept[name]).name / "SKILL.md" for name in names if name in kept]
        # One skill, as an import kept before it kept several.
        return sorted((folder / "skills").glob("*/SKILL.md")) if imported.get("skill_source") == "extracted" else []

    def _added_from_a_run(self, record: Path) -> bool:
        """Whether a skill added was made from what a run's model read — the SKILL.md it read, or an
        import's copy extracted from its chat — not the copy a launched run was given, nor a folder or
        an upload a person brought, each kept as given. Of a run that is gone, a copy it kept: which
        it was is gone with the run."""
        try:
            origin = (json.loads(record.read_text(encoding="utf-8")) or {}).get("origin") or {}
        except (OSError, ValueError, AttributeError):
            return False
        if not isinstance(origin, dict) or not origin.get("run"):
            return False
        how = str(origin.get("how") or "")
        if how.startswith("the SKILL.md its model read"):
            return True
        run = self.store.get_run(str(origin["run"]))
        if run is None:
            return how == "the copy it kept"
        imported = (run.parity or {}).get("imported") or {}
        listed = imported.get("skills")
        if isinstance(listed, list):
            return any(isinstance(each, dict) and each.get("name") == origin.get("skill")
                       and each.get("source") == "extracted" for each in listed)
        return bool(imported) and imported.get("skill_source") == "extracted"

    # --- how each is read -----------------------------------------------------------------

    def _text(self, text: Any, also: tuple[str, ...]) -> tuple[int, frozenset[str]]:
        masked = self.masking.of(_as_text(text), also=also)
        return masked.made, masked.found

    def _address(self, text: Any, also: tuple[str, ...]) -> tuple[int, frozenset[str]]:
        """A server's address as it was kept — alone, or in its identity — as `_text` reads it, and
        as an address is kept today (`config.masked_address`): what masking it so would mask, and
        the values it would take that read as a key (`config.key_like`), looked for wherever else
        they stand — never a word or a date a query held (found in review: every run naming the
        `api-version` an address gave read as holding a key)."""
        from tuningfork.config import key_like, masked_address

        made, taken = self._text(text, also)
        identity = _loaded(text)
        address = identity.get("upstream") if isinstance(identity, dict) else text
        if not isinstance(address, str):
            return made, taken
        masked, values = masked_address(address)
        keys = frozenset(value for value in values if key_like(value))
        return (max(made, 1), taken | keys) if masked != address else (made, taken)

    def _raw(self, wire: str | None, streamed: bool | None) -> Callable[[Any, tuple[str, ...]], tuple[int, frozenset[str]]]:
        """An answer's raw bytes: as text, and as its wire reads them back whole — a key a stream
        sent in pieces, which no pattern sees in one event."""
        def holds(body: Any, also: tuple[str, ...]) -> tuple[int, frozenset[str]]:
            from tuningfork.wire import get_wire

            made, taken = self._text(body, also)
            if made:
                return made, taken
            data = body if isinstance(body, bytes) else str(body).encode("utf-8", "surrogateescape")
            # A stream where its step does not say: events, or a comment first, as servers send them.
            stream = streamed if streamed is not None else data.lstrip().startswith((b"data:", b"event:", b":"))
            try:
                native = get_wire(wire).parse(data, bool(stream))
            except Exception:  # noqa: BLE001 - bytes that do not read back hold what their text does
                native = None
            if not isinstance(native, dict):
                return 0, frozenset()
            masked = self.masking.of(native, also=also)
            return masked.made, masked.found
        return holds

    def _tokens(self, body: Any, also: tuple[str, ...]) -> tuple[int, frozenset[str]]:
        """A token stream, as its tokens join: a key the model wrote over several. A position the
        store's masking takes loses its id, and nothing else changes one."""
        from tuningfork.analysis.tokens import TokenStream

        data = json.loads(body)
        TokenStream.from_json(data)  # one that does not read as a stream is said so (`_unread`)
        return int(self.masking.stream(data, also=also)["ids"] != data["ids"]), frozenset()

    # --- the report -----------------------------------------------------------------------

    def found(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The runs holding what today's masking would mask, each with where, how often, the first
        step of each, the first answer a fork gives again, and what sets it aside; and what else
        holds some, and how it goes."""
        counts: dict[tuple[str, str], Counter[str]] = {}
        firsts: dict[tuple[str, str], dict[str, int]] = {}
        replayed: dict[str, int] = {}
        flagged: list[str] = []
        for unit, _ in self.hits.values():
            if unit.key[0] == "message":
                flagged.append(unit.key[1])
            for place in unit.places:
                owner = (place.owner, place.id)
                counts.setdefault(owner, Counter())[place.kind] += 1
                if place.step is not None:
                    first = firsts.setdefault(owner, {})
                    first[place.kind] = min(first.get(place.kind, place.step), place.step)
                    if unit.answer and place.owner == "run":
                        replayed[place.id] = min(replayed.get(place.id, place.step), place.step)
        for run, (count, step) in self.store.runs_holding(flagged).items() if flagged else ():
            counts.setdefault(("run", run), Counter())["messages"] += count
            first = firsts.setdefault(("run", run), {})
            first["messages"] = min(first.get("messages", step), step)
        runs, elsewhere = [], []
        for (owner, ident), kinds in counts.items():
            if owner != "run":
                elsewhere.append(self._elsewhere(owner, ident, sum(kinds.values())))
                continue
            run = self.store.get_run(ident)
            # Shown masked of what the masks took too: a password named elsewhere, bare in a label.
            label = (self.masking.shown(run.label, also=(*self.held, *sorted(self.taken))).value
                     if run is not None and run.label else None)
            runs.append({
                "run": ident,
                "recorded": run.recorded if run is not None else None,
                "started_at": run.started_at if run is not None else None,
                "label": label,
                "trashed": bool(run is not None and run.trashed_at),
                "running": run is not None and self._running(run),
                "aside": self._aside(run) if run is not None else {"kind": "run", "id": ident},
                "places": {kind: kinds[kind] for kind in KINDS if kinds.get(kind)},
                "first_step": {kind: step for kind, step in sorted((firsts.get((owner, ident)) or {}).items())},
                # A fork from a later step gives its harness that answer again: of a launched run only.
                "replays_from": replayed.get(ident) if run is not None and run.recorded == "launched" else None,
            })
        runs.sort(key=lambda found: (found["started_at"] or 0, found["run"]))
        elsewhere.sort(key=lambda found: (found["owner"], found["id"]))
        return runs, elsewhere

    def _elsewhere(self, owner: str, ident: str, places: int) -> dict[str, Any]:
        """What else holds some, and how it goes: by the trash (`TRASHED_AS`), by hand (`BY_HAND`),
        or by nothing tuningfork has — a server's record, a queued label, a launch no run came of."""
        trashed = ident in self.trashed.get(owner, ())
        by_hand = [BY_HAND[owner].format(ident)] if owner in BY_HAND else None
        if owner == "skill" and "/" not in ident:
            # A skill added: its copy, its record, and what an approval kept beside it.
            beside = [f"{ident}.json", f"{ident}.history/", f"{ident}.changelog.md"]
            by_hand = [*(by_hand or []), *(f"skills/{name}" for name in beside
                                           if (self.store.home / "skills" / name.rstrip("/")).exists())]
        return {
            "owner": owner, "id": ident, "places": places, "trashed": trashed,
            "running": (owner, ident) in self.running,
            "aside": {"kind": owner, "id": ident} if owner in TRASHED_AS else None,
            "by_hand": by_hand,
        }

    def _running(self, run: Any) -> bool:
        """Whether the trash would refuse the run as still running: its launch is."""
        from tuningfork.ops.inspect import launch_status

        launch = self.store.get_launch(run.launch_id) if run.launch_id else None
        return launch is not None and launch_status(launch) == "running"

    def _aside(self, run: Any) -> dict[str, str]:
        """What the trash takes to set `run` aside (`ops.trash`): a thread goes with its launch's
        main run, and a fork an experiment drew with that experiment."""
        from tuningfork.ops.trash import experiment_of_fork

        if run.is_thread and run.launch_id:
            run = next((each for each in self.store.launch_runs(run.launch_id) if not each.is_thread), run)
        experiment = experiment_of_fork(self.store, run)
        return {"kind": "experiment", "id": experiment} if experiment else {"kind": "run", "id": run.id}


def _kind(table: str, column: str, owner: str) -> str:
    """What of a run a row's value is (`KINDS`); of anything else, its record."""
    if owner != "run":
        return "record"
    if table in ("experiments", "samples"):
        return "experiments"
    if table == "steps":
        return {"error": "errors", "message_overrides": "messages"}.get(column, "records")
    if (table, column) in (("runs", "result"), ("launches", "exit")):
        return "result"
    return "records"


def _a_person_s(path: Path) -> bool:
    """Whether a refinement's proposal was a person's alone, kept as they gave it."""
    try:
        proposers = (json.loads(path.read_text(encoding="utf-8")) or {}).get("proposers") or []
    except (OSError, ValueError, AttributeError):
        return False
    return bool(proposers) and all(isinstance(each, dict) and each.get("via") == "person" for each in proposers)


def _unread(unit: _Unit, error: Exception) -> str:
    """What could not be read, by its place — never its content — and why."""
    what, name = unit.key[0], str(unit.key[1])
    shown = f"file {name}" if what == "file" else f"{'message ' if what == 'message' else ''}blob {name[:12]}"
    if unit.places:
        place = next(iter(unit.places))
        shown += f" ({place.owner} {place.id[:8]}" + (f", step {place.step}" if place.step is not None else "") + ")"
    why = ("missing" if isinstance(error, FileNotFoundError) else "could not be opened" if isinstance(error, OSError)
           else "does not read as what it should hold")
    return f"{shown}: {why}"


def _folders(path: Path) -> list[Path]:
    return sorted(child for child in path.iterdir() if child.is_dir()) if path.is_dir() else []


def _digest(path: Path) -> str | None:
    """A file's digest, or None where it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _blob_text(store: Store, blob: str) -> Callable[[], bytes]:
    return lambda: store.blobs.get(blob)


def _file_text(path: Path) -> Callable[[], bytes]:
    return path.read_bytes


def _spec_text(path: Path) -> Callable[[], Any]:
    """A scenario's spec, but what it takes from its run's task as written (`GIVEN_TO_A_SCENARIO`),
    which a fork of the task's own runs runs as it is."""
    def read() -> Any:
        data = path.read_bytes()
        try:
            spec = json.loads(data)
        except ValueError:
            return data
        for item in (spec.get("evals") if isinstance(spec, dict) else None) or []:
            for given in GIVEN_TO_A_SCENARIO:
                if isinstance(item, dict):
                    item.pop(given, None)
                    if isinstance(item.get("tuningfork"), dict):
                        item["tuningfork"].pop(given, None)
        return json.dumps(spec, ensure_ascii=False)
    return read


def _as_text(value: Any) -> str:
    return value.decode("utf-8", "surrogateescape") if isinstance(value, bytes) else str(value)


def _loaded(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


__all__ = ["BY_HAND", "KINDS", "NOT_FOUND", "NOT_READ", "NOT_RECORDED", "OWNED", "Place", "ScanReport", "TRASHED_AS",
           "older", "older_note", "scan"]
