"""The job runner: work started from the web UI — or any front end — runs inside the proxy.

A job is one operation from `ops.catalog` — a resample, an A/B, a batch, a fork — with its
parameters. Each runs on a worker thread of its own, so a slow job never stalls the event
loop that serves harness traffic; an async operation runs under `asyncio.run` in that
thread. A job's model calls go to the proxy's own socket, marked as an experiment's, so they
pass its gate and its pause switch like any call and are never recorded or held.

Two lanes, each first in, first out:

- **background**: batches, resamples, A/Bs, forks. One at a time by default (`serve --jobs`):
  on one GPU, more at once was measured slower, not faster.
- **interactive**: short work a person is waiting on — tools at a held call, server checks —
  which must not queue behind a long batch.

No job starts while the proxy is paused. A queued plan can go stale, so every job is
prepared again when it starts. What a job makes — experiments, launches, runs — is stored
exactly as if made from the CLI, and linked to the job. Progress is kept in memory, the
last `KEEP` events of each job with a sequence number, so a page that reconnects asks for
what it missed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from typing import Any

from tuningfork.config import redact_json
from tuningfork.ops.catalog import BACKGROUND, CATALOG, INTERACTIVE, JobContext, Kind
from tuningfork.ops.common import Cancel, Cancelled, Event, OperationError
from tuningfork.owners import current_owner, owner_alive
from tuningfork.store.db import JobRecord, Store

logger = logging.getLogger("tuningfork.jobs")

#: Progress events kept per job, for a page catching up after a reconnect.
KEEP = 200
#: How often a running job's progress is written to the store, at most, in seconds.
SAVE_EVERY = 2.0
FINISHED = frozenset({"done", "failed", "cancelled", "interrupted"})


class Job:
    """A job as the runner holds it: its record, and what lives only while it runs."""

    def __init__(self, record: JobRecord, params: Any) -> None:
        self.record = record
        #: The parsed parameters, unredacted: what the job runs with.
        self.params = params
        self.cancel = Cancel()
        self.events: deque[dict[str, Any]] = deque(maxlen=KEEP)
        self.seq = 0
        self.thread: threading.Thread | None = None
        self.saved_at = 0.0

    @property
    def id(self) -> str:
        return self.record.id

    def to_json(self, *, after: int | None = None, position: int | None = None) -> dict[str, Any]:
        return job_json(self.record, seq=self.seq, position=position, events=(
            None if after is None else [entry for entry in self.events if entry["seq"] > after]
        ), first=self.events[0]["seq"] if self.events else self.seq + 1)


def job_json(
    record: JobRecord,
    *,
    seq: int | None = None,
    position: int | None = None,
    events: list[dict[str, Any]] | None = None,
    first: int | None = None,
) -> dict[str, Any]:
    """A job as a front end sees it. `events`, when asked for, are those after the `seq` the
    asker has; `first` is the oldest still kept — a gap before it cannot be filled. A running
    job's progress says how long is left at its pace so far (`remaining_s`): an agent that
    polls a forty-fork screen needs to know it is twenty minutes, not a few."""
    data = {
        "id": record.id,
        "kind": record.kind,
        "lane": record.lane,
        "status": record.status,
        "params": record.params,
        "plan": record.plan,
        "progress": remaining(record),
        "result": record.result,
        "error": record.error,
        "links": record.links,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "trashed_at": record.trashed_at,
        "started_by": record.started_by,
        "queue_position": position,
        "seq": seq,
    }
    if events is not None:
        data["events"] = events
        data["first_seq"] = first
    return data


def remaining(record: JobRecord) -> dict[str, Any] | None:
    """The job's progress, with how many seconds are left at its pace so far while it runs
    and has finished a unit: the units left at the mean time of those done."""
    progress = record.progress
    if record.status != "running" or not progress or not record.started_at:
        return progress
    done, total = progress.get("done") or 0, progress.get("total") or 0
    if not done or not total or done >= total:
        return progress
    elapsed = max(0.0, time.time() - record.started_at)
    return {**progress, "remaining_s": round(elapsed / done * (total - done))}


def asked_by(value: Any) -> dict[str, str] | None:
    """Who asked for a job, as a front end says it: `{"via": "mcp", "client": ...,
    "version": ...}`. Short text only: it is shown on a page, never acted on."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("via"), str):
        raise OperationError(
            'started_by is {"via": ..., "client": ..., "version": ...}', code="bad_params", field="started_by"
        )
    kept = {}
    for key in ("via", "client", "version"):
        found = value.get(key)
        if found is None:
            continue
        if not isinstance(found, str) or not 0 < len(found) <= 100:
            raise OperationError(f"started_by.{key} must be short text", code="bad_params", field="started_by")
        kept[key] = found
    return kept


def kept_params(parsed: Any) -> dict[str, Any]:
    """A job's parameters as they are kept and shown — on the Jobs page, to whoever reads the
    jobs API — redacted, and a harness's options as its plan shows them: every value under an
    MCP server's `headers` or `env` masked, whatever its name, which no pattern would know for
    a secret (`shown_options`). The job runs with them as given; what is kept is never run."""
    from tuningfork.ops.batch import shown_options

    data = dataclasses.asdict(parsed)
    if isinstance(data.get("harness_options"), dict):
        data["harness_options"] = shown_options(data["harness_options"])
    return redact_json(data)


def plain(value: Any) -> Any:
    """A result, or a cancelled job's partial result, as JSON."""
    if value is None:
        return None
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if dataclasses.is_dataclass(value) and hasattr(value, "arm"):  # an experiment
        from tuningfork.ops.experiments import experiment_json

        return experiment_json(value)
    return value


class JobRunner:
    def __init__(
        self,
        store: Store,
        bus: Any,
        proxy: Any,
        *,
        catalog: dict[str, Kind] | None = None,
        background: int = 1,
        interactive: int = 1,
    ) -> None:
        self.store = store
        self.bus = bus
        self.proxy = proxy
        self.catalog = catalog if catalog is not None else CATALOG
        self.capacity = {BACKGROUND: max(1, background), INTERACTIVE: max(1, interactive)}
        self.owner = current_owner()
        self._lock = threading.Lock()
        #: Queued and running jobs, and the most recently finished, by id.
        self._jobs: dict[str, Job] = {}
        self._finished: deque[str] = deque(maxlen=50)
        self._queues: dict[str, deque[Job]] = {BACKGROUND: deque(), INTERACTIVE: deque()}
        self._running: dict[str, set[str]] = {BACKGROUND: set(), INTERACTIVE: set()}
        #: Held from the check for the same work to the job's making: see `submit`.
        self._submitting = threading.Lock()
        self._stopping = False

    # --- asking ------------------------------------------------------------------------

    def kinds(self) -> list[dict[str, Any]]:
        return [kind.to_json() for kind in self.catalog.values()]

    def _kind(self, name: Any) -> Kind:
        kind = self.catalog.get(name) if isinstance(name, str) else None
        if kind is None:
            raise OperationError(
                f"no kind of job called {name!r}",
                hint=f"known: {', '.join(self.catalog)}",
                code="bad_params",
                field="kind",
            )
        return kind

    def plan(self, kind_name: str, params: Any) -> dict[str, Any]:
        """What a job would do, checked the way starting it checks — nothing is started.
        Blocking: call it off the event loop."""
        kind = self._kind(kind_name)
        parsed = kind.parse(params)
        context = JobContext(store=self.store, proxy=self.proxy, job_id="")
        prepared = kind.prepare(context, parsed)
        # `total`: the samples, forks or runs it would make, which is what a budget counts.
        return {"kind": kind.name, "lane": kind.lane, "plan": plain(prepared), "total": kind.total(prepared)}

    def submit(
        self,
        kind_name: str,
        params: Any,
        *,
        request_id: str | None = None,
        started_by: Any = None,
        dedupe: bool = False,
    ) -> dict[str, Any]:
        """Queue a job, checked as starting it would be: a job that cannot run is refused
        now, not after waiting its turn. It is prepared again when it starts.

        The same `request_id` twice gives the job the first made, so a form sent again —
        a double click, a retry after a dropped connection — does not start the work twice.
        With `dedupe`, a job of the same kind and parameters that is still queued or running
        is given back, marked `deduplicated`, instead of a second: an agent asking again
        gets the work already under way, which with the same seeds would draw the same
        samples anyway. `started_by` says who asked, for the Jobs page. Blocking: call it
        off the event loop."""
        if self._stopping:
            raise OperationError("tuningfork is stopping; no new work starts", code="stopping")
        kind = self._kind(kind_name)
        parsed = kind.parse(params)
        started_by = asked_by(started_by)
        stored_params = kept_params(parsed)
        # Work that may be asked for twice is checked for and made one submission at a time:
        # found in review, a second asked while the first was being prepared passed the check
        # too, and the work started twice.
        with self._submitting if dedupe or request_id is not None else nullcontext():
            if dedupe:
                with self._lock:
                    same = next((
                        job for job in self._jobs.values()
                        if job.record.kind == kind.name and job.record.status in ("queued", "running")
                        and job.record.params == stored_params
                    ), None)
                if same is not None:
                    return {**self._summary(same), "deduplicated": True}
            if request_id is not None:
                if not isinstance(request_id, str) or not 0 < len(request_id) <= 100:
                    raise OperationError("request_id must be text", code="bad_params", field="request_id")
                with self._lock:
                    known = next((job for job in self._jobs.values() if job.record.request_id == request_id), None)
                if known is not None:
                    return self._summary(known)
                stored = self.store.job_for_request(request_id)
                if stored is not None:
                    return job_json(stored)
            # Refused now, not after waiting its turn behind a long batch: it is prepared again
            # when it starts, since what it depends on can change while it waits.
            kind.prepare(JobContext(store=self.store, proxy=self.proxy, job_id=""), parsed)
            record = self.store.create_job(
                kind=kind.name, lane=kind.lane, params=stored_params, request_id=request_id,
                started_by=started_by,
            )
            record.owner = self.owner
            self.store.update_job(record.id, owner=self.owner)
            job = Job(record, parsed)
            with self._lock:
                self._jobs[job.id] = job
                self._queues[kind.lane].append(job)
        self._announce(job)
        self.kick()
        return self._summary(job)

    def get(self, reference: str, *, after: int | None = None) -> dict[str, Any] | None:
        with self._lock:
            job = self._find(reference)
            if job is not None:
                return job.to_json(after=after, position=self._position(job))
        stored = self.store.get_job(reference)
        return job_json(stored, events=[] if after is not None else None) if stored else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent jobs, newest first: the ones held here as they are now, the rest as stored."""
        stored = self.store.jobs(limit=limit)
        with self._lock:
            return [
                self._jobs[record.id].to_json(position=self._position(self._jobs[record.id]))
                if record.id in self._jobs
                else job_json(record)
                for record in stored
            ]

    def active(self) -> list[dict[str, Any]]:
        """Queued and running jobs, for a page's first snapshot. Memory only: safe on the
        event loop."""
        with self._lock:
            return [
                job.to_json(position=self._position(job))
                for job in self._jobs.values()
                if job.record.status in ("queued", "running")
            ]

    def _find(self, reference: str) -> Job | None:
        if reference in self._jobs:
            return self._jobs[reference]
        matches = [job for job_id, job in self._jobs.items() if job_id.startswith(reference)]
        return matches[0] if len(matches) == 1 and reference else None

    def _position(self, job: Job) -> int | None:
        if job.record.status != "queued":
            return None
        queue = self._queues[job.record.lane]
        return next((index for index, queued in enumerate(queue) if queued is job), None)

    def _summary(self, job: Job) -> dict[str, Any]:
        with self._lock:
            return job.to_json(position=self._position(job))

    # --- running -------------------------------------------------------------------------

    def kick(self) -> None:
        """Start what the lanes have room for. Nothing starts while the proxy is paused:
        a pause is a person looking at something. Blocking: call it off the event loop."""
        if self._stopping or self.proxy.paused:
            return
        starting: list[Job] = []
        with self._lock:
            for lane, queue in self._queues.items():
                while queue and len(self._running[lane]) < self.capacity[lane]:
                    job = queue.popleft()
                    self._running[lane].add(job.id)
                    job.record.status = "running"
                    job.record.started_at = time.time()
                    starting.append(job)
        for job in starting:
            self.store.update_job(job.id, status="running", started_at=job.record.started_at)
            self._announce(job)
            job.thread = threading.Thread(
                target=self._run, args=(job,), name=f"tuningfork-job-{job.id[:8]}", daemon=True
            )
            job.thread.start()

    def _run(self, job: Job) -> None:
        kind = self.catalog[job.record.kind]
        context = JobContext(store=self.store, proxy=self.proxy, job_id=job.id)
        status, result, error = "done", None, None
        try:
            prepared = kind.prepare(context, job.params)
            with self._lock:
                job.record.plan = plain(prepared)
                job.record.progress = {"done": 0, "total": kind.total(prepared), "waiting": None}
            self.store.update_job(job.id, plan=job.record.plan, progress=job.record.progress)
            self._announce(job)
            job.cancel.check()  # cancelled while it was being prepared
            outcome = kind.execute(context, prepared, progress=lambda event: self._progress(job, event), cancel=job.cancel)
            if inspect.iscoroutine(outcome):
                outcome = asyncio.run(outcome)
            result = plain(outcome)
        except Cancelled as stopped:
            status = "interrupted" if stopped.reason == "shutdown" else "cancelled"
            result = plain(stopped.partial)
        except OperationError as failure:
            status, error = "failed", failure.to_json()
        except Exception as failure:  # noqa: BLE001 - a job's bug must not take the proxy down
            logger.exception("job %s (%s) failed", job.id[:8], job.record.kind)
            status = "failed"
            error = {"message": f"{type(failure).__name__}: {failure}", "code": "internal"}
        self._finish(job, status, result=result, error=error)

    def _finish(self, job: Job, status: str, *, result: Any = None, error: Any = None) -> None:
        with self._lock:
            job.record.status = status
            job.record.result = result
            job.record.error = error
            job.record.ended_at = time.time()
            if job.record.progress:
                job.record.progress = {**job.record.progress, "waiting": None}
            self._running[job.record.lane].discard(job.id)
            self._finished.append(job.id)
            # Keep the recently finished for pages still watching; forget the rest.
            keep = set(self._finished)
            for job_id in [
                job_id for job_id, held in self._jobs.items()
                if held.record.status in FINISHED and job_id not in keep
            ]:
                del self._jobs[job_id]
        try:
            self.store.update_job(
                job.id, status=status, result=result, error=error, ended_at=job.record.ended_at,
                progress=job.record.progress, links=job.record.links,
            )
            if status == "interrupted":
                self.store.interrupt_job_work(job.id)
        except Exception:  # noqa: BLE001 - the store closed under a shutdown
            logger.warning("could not store how job %s ended", job.id[:8], exc_info=True)
        self._announce(job)
        self.kick()

    def _progress(self, job: Job, event: Event) -> None:
        """An event from the operation, on its worker thread: keep it, link what it made,
        and tell the pages watching."""
        entry = {"seq": 0, "at": time.time(), "event": event.to_json()}
        link = None
        with self._lock:
            job.seq += 1
            entry["seq"] = job.seq
            job.events.append(entry)
            progress = dict(job.record.progress or {})
            # A held-out run says it ended, never how (`ops.private`), and still counts.
            if event.kind in ("sample", "run", "check", "held_out_run"):
                progress["done"] = int(progress.get("done") or 0) + 1
                progress["waiting"] = None
            elif event.kind == "waiting":
                progress["waiting"] = event.data.get("reason") or "waiting"
            job.record.progress = progress
            link = self._link(job, event)
            save = link is not None or time.monotonic() - job.saved_at >= SAVE_EVERY
            if save:
                job.saved_at = time.monotonic()
        if link == "experiments":
            self.store.set_experiment_job(event.data["id"], job.id)
        elif link == "launches":
            self.store.set_launch_job(event.data["launch_id"], job.id)
        if save:
            self.store.update_job(job.id, progress=job.record.progress, links=job.record.links)
        self.proxy.call_soon(
            self.bus.publish, "job_event",
            job_id=job.id, seq=entry["seq"], event=entry["event"], progress=progress,
        )
        if link is not None:
            self._announce(job)  # what it made: a page links it, or finds its fork's hold

    @staticmethod
    def _link(job: Job, event: Event) -> str | None:
        """Record what an event says the job made; the kind of thing, when it is new."""
        found = {
            "experiment": ("experiments", event.data.get("id")),
            "launch": ("launches", event.data.get("launch_id")),
            "run": ("runs", event.data.get("run_id")),
        }.get(event.kind)
        if not found or not found[1]:
            return None
        name, value = found
        links = job.record.links
        if value in links.setdefault(name, []):
            return None
        links[name].append(value)
        return name

    def _announce(self, job: Job) -> None:
        self.proxy.call_soon(self.bus.publish, "job", job=self._summary(job))

    # --- stopping ------------------------------------------------------------------------

    def cancel(self, reference: str, reason: str = "cancelled") -> dict[str, Any]:
        """Stop a job. A queued one ends at once; a running one is asked to stop, and ends
        when its operation has: what it finished stays stored, and a harness it killed is
        never judged. Cancelling a finished job changes nothing."""
        with self._lock:
            job = self._find(reference)
            if job is None:
                stored = self.store.get_job(reference)
                if stored is None:
                    raise OperationError(f"no job matching {reference!r}", code="not_found")
                return job_json(stored)
            queued = job.record.status == "queued"
            if queued:
                self._queues[job.record.lane].remove(job)
                # Ended here, under the lock: a second cancel before `_finish` has run finds it
                # ended, not queued still (found in review: it failed taking the job out of its
                # queue again, and the job stayed queued for good).
                job.record.status = "interrupted" if reason == "shutdown" else "cancelled"
        if queued:
            self._finish(job, job.record.status)
        elif job.record.status == "running":
            job.cancel.cancel(reason)
            # A fork waiting at a breakpoint would wait for a decision no one will make.
            self.proxy.drop_holds_of(job.record.links.get("launches", []))
        return self._summary(job)

    def cancel_all(self, reason: str = "cancelled") -> None:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.record.status in ("queued", "running")]
        for job in jobs:
            try:
                self.cancel(job.id, reason)
            except Exception:  # noqa: BLE001 - one job's trouble must not stop the rest
                logger.warning("could not cancel job %s", job.id[:8], exc_info=True)

    def stop(self) -> None:
        """Start no more work, and ask every job to stop. Returns at once: see `join`."""
        self._stopping = True
        self.cancel_all("shutdown")

    def join(self, timeout: float = 30.0) -> bool:
        """Wait for job threads to end — before the store they write to is closed.
        Returns whether every one did."""
        deadline = time.monotonic() + timeout
        with self._lock:
            threads = [job.thread for job in self._jobs.values() if job.thread is not None]
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in threads)

    def shutdown(self, timeout: float = 30.0) -> bool:
        self.stop()
        return self.join(timeout)

    def sweep(self) -> int:
        """Mark work cut off by an earlier proxy's end: jobs it held, the experiments and
        launches they left running. Another process still alive keeps its own."""
        swept = 0
        for record in self.store.jobs(unfinished=True, limit=1000):
            if record.owner and owner_alive(record.owner) is not False:
                continue  # still running elsewhere, or no way to tell
            self.store.update_job(record.id, status="interrupted", ended_at=time.time())
            self.store.interrupt_job_work(record.id)
            swept += 1
        stale = [
            experiment.id
            for experiment in self.store.running_experiments()
            if experiment.owner and owner_alive(experiment.owner) is False
        ]
        self.store.finish_experiments(stale, "interrupted")
        return swept
