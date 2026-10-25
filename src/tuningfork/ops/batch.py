"""Batch runs: one eval of a task spec, run N times through a harness, each run a launch.

Everything the CLI's `batch` used to decide for itself is here, so the web UI (through the
job runner) and, later, MCP start a batch exactly as the CLI does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress, emit
from tuningfork.ops.launch import ProxyStatus
from tuningfork.store.db import Store


@dataclass(frozen=True, slots=True)
class Note:
    """Something a person should know before a batch runs, which does not stop it."""

    level: str  # "warning" | "note"
    message: str

    def to_json(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message}


@dataclass(slots=True)
class BatchPlan:
    spec: Any  # specs.TaskSpec
    item: Any  # specs.Eval
    n: int
    harness: str
    model: str
    base_url: str
    proxy: ProxyStatus
    workdir: Path
    concurrency: int
    skill_dir: Path | None
    skill_file: Path | None
    #: Recorded with every run, so two batches can be told apart by how they were run.
    parity: dict[str, Any]
    notes: list[Note] = field(default_factory=list)
    #: The harness adapter, running the executable the recordings folder names, if any.
    adapter: Any = None
    #: The sandbox every run is launched in (`tuningfork.sandbox.Sandbox`), or None: each
    #: run then sees its own folders, and a note says why.
    sandbox: Any = None
    #: The harness's own options, as given; what the adapter made of them is in each
    #: launch's settings.
    harness_options: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"{self.spec.skill_name}/{self.item.id}"

    def to_json(self) -> dict[str, Any]:
        from tuningfork.specs import check_kind, describe_check

        check = self.item.tuningfork.check
        return {
            "task_id": self.task_id,
            "spec": str(self.spec.path) if self.spec.path else None,
            "eval": self.item.id,
            # Every task the spec holds, so a form can offer them by what each asks — but a
            # held-out task only by its id: a plan is shown to whoever asks for one.
            "evals": [{"id": item.id, "prompt": item.prompt[:240]} for item in self.spec.evals
                      if not item.tuningfork.held_out],
            "held_out": [item.id for item in self.spec.held_out_evals()],
            "prompt": self.item.prompt,
            # Always shown whole: a shell check runs as a command in each run's workspace.
            "check": check,
            "check_kind": check_kind(check),
            "check_text": describe_check(check),
            # Where runs happen: each gets a fresh folder under it, with its own workspace.
            "workdir": str(self.workdir),
            "n": self.n,
            "harness": self.harness,
            "model": self.model,
            "proxy": self.proxy.url,
            "concurrency": self.concurrency,
            "skill_dir": str(self.skill_dir) if self.skill_dir else None,
            "skill_file": str(self.skill_file) if self.skill_file else None,
            # What every run sees as its home and folder, when they run in the sandbox.
            "sandbox": self.sandbox.to_json() if self.sandbox is not None else None,
            "harness_options": shown_options(self.harness_options),
            "notes": [note.to_json() for note in self.notes],
        }


#: Keys of an MCP server's options whose values are secrets: a bearer token in its headers,
#: a key in its environment. Shown and recorded masked; the launch itself is given them.
SECRET_KEYS = ("headers", "env")


def shown_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The harness options as a plan shows and a record keeps them: every value under an
    MCP server's `headers` or `env` replaced by an ellipsis. What the launch records of
    them is the adapter's own, stricter view (Hermes: a server's command or URL alone)."""
    found = dict(options or {})
    servers = found.get("mcp_servers")
    if isinstance(servers, dict):
        found["mcp_servers"] = {
            name: (
                {
                    key: ({inner: "…" for inner in value} if key in SECRET_KEYS and isinstance(value, dict) else value)
                    for key, value in server.items()
                }
                if isinstance(server, dict) else server
            )
            for name, server in servers.items()
        }
    return found


def prepare_batch(
    store: Store,
    spec: Path | Any,
    *,
    eval_id: str | None = None,
    n: int = 5,
    model: str | None = None,
    harness: str = "hermes",
    skill_file: Path | None = None,
    workdir: Path | None = None,
    concurrency: int = 1,
    base_url: str | None = None,
    proxy: ProxyStatus | None = None,
    include_held_out: bool = False,
    hide: Sequence[Path] = (),
    skill_dir: Path | None = None,
    harness_options: dict[str, Any] | None = None,
) -> BatchPlan:
    """Everything a batch needs, checked before any run is launched: a missing input file or
    an unknown harness is refused before a single launch is recorded.

    `spec` is a path to an `evals.json`, or an already-loaded `TaskSpec`. `proxy` is for a
    caller inside the proxy itself, which knows it without asking over HTTP. `skill_dir`
    gives every run a skill folder other than the spec's — how a chain hands a link the skill
    the one before left. `harness_options` are the harness's own (Hermes: `await_review`,
    `review_every_turn`, `mcp_servers`), read by its adapter and recorded with each run.

    A held-out eval judges an edit, so its runs must stay where no agent can read them: it
    runs here only with `include_held_out`, and is never the default (`refine validate` runs
    held-out evals without recording them).

    Where bubblewrap runs, every run is launched in the sandbox (`tuningfork.sandbox`), and
    the recordings folder, the runs' folder, the spec's folder and `hide` are empty in it.
    """
    from tuningfork.control import default_proxy
    from tuningfork.harness import ADAPTERS
    from tuningfork.ops.connections import adapter_for, cached_prompts_warning, check_harness
    from tuningfork.ops.launch import check_launchable, check_proxy
    from tuningfork.paths import host_path
    from tuningfork.runner import lookalike_skill_files
    from tuningfork.sandbox import for_launches, turned_off as sandbox_turned_off
    from tuningfork.specs import SpecError, check_kind, configured_check_env, load_spec, missing_files

    if harness not in ADAPTERS:
        raise OperationError(
            f"unknown harness {harness!r}; known: {', '.join(ADAPTERS)}", field="harness"
        )
    if n < 1:
        raise OperationError("a batch needs at least one run", field="n")
    if concurrency < 1:
        raise OperationError("concurrency must be at least 1", field="concurrency")
    if skill_file is not None:
        skill_file = Path(host_path(str(skill_file)))
        if not skill_file.is_file():
            raise OperationError(f"no such skill file: {skill_file}", field="skill_file")
    adapter = adapter_for(store.home, harness, options=harness_options)
    check_launchable(adapter)

    base_url = base_url or f"{default_proxy()}/v1"
    proxy = proxy or check_proxy(base_url, store.home)
    if isinstance(spec, (str, Path)):
        try:
            spec = load_spec(Path(host_path(str(spec))))
        except (OSError, SpecError) as exc:
            raise OperationError(str(exc), field="spec") from None
    try:
        item = spec.get(eval_id) if eval_id is not None or not spec.evals else spec.default_eval()
    except (KeyError, ValueError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise OperationError(str(message), field="eval") from None
    if item is None:
        raise OperationError(
            f"every eval of {spec.skill_name} is held out: a batch runs a task an edit is made for",
            hint="name a held-out eval with --eval and --include-held-out to record it anyway",
            code="held_out", field="eval",
        )
    if item.tuningfork.held_out and not include_held_out:
        raise OperationError(
            f"eval {item.id} of {spec.skill_name} is held out: it judges edits, so its runs must "
            "stay where no agent can read them",
            hint="`tuningfork refine validate` runs held-out evals without recording them; "
            "--include-held-out records this one here anyway",
            code="held_out", field="eval",
        )
    spec_skill = spec.resolved_skill_dir()
    if spec_skill is not None and not (spec_skill / "SKILL.md").is_file():
        # Refused here: the runner would copy nothing and every run would go without the skill.
        raise OperationError(
            f"the spec's skill folder, {spec_skill}, "
            + ("holds no SKILL.md" if spec_skill.is_dir() else "does not exist"),
            hint=f"tuningfork.skill_dir is {spec.skill_dir!r}, resolved against the spec's "
            f"folder, {spec.directory}",
            field="spec",
        )
    if skill_dir is not None:
        skill_dir = Path(host_path(str(skill_dir)))
        if not (skill_dir / "SKILL.md").is_file():
            raise OperationError(
                f"the skill folder given, {skill_dir}, "
                + ("holds no SKILL.md" if skill_dir.is_dir() else "does not exist"),
                field="skill_dir",
            )
    else:
        skill_dir = spec_skill
    if skill_file is not None and not (skill_dir and skill_dir.exists()):
        raise OperationError(
            "--skill replaces the skill's SKILL.md, but the spec names no skill folder "
            "(tuningfork.skill_dir)",
            field="skill_file",
        )
    missing = missing_files(item, spec.directory)
    if missing:
        raise OperationError(
            f"eval {item.id} names input files that are not next to the spec "
            f"({spec.directory}): {', '.join(map(str, missing))}",
            field="spec",
        )
    # A scenario's conversation, read now: one the runs could not carry is refused before any runs.
    context = None
    if item.tuningfork.context:
        from tuningfork.specs import SpecError, load_context
        from tuningfork.wire import get_wire

        try:
            context = load_context(spec.directory, item.tuningfork.context)
        except SpecError as exc:
            raise OperationError(str(exc), field="spec") from None
        speaks = getattr(adapter, "wire", "openai-chat")
        prompt = {"messages": [{"role": "user", "content": item.prompt}]}
        if get_wire(speaks).with_context(prompt, context.before, context.in_turn) is None:
            raise OperationError(
                f"eval {item.id} continues a conversation, which {adapter.name}'s calls ({speaks}) cannot carry yet",
                field="harness",
            )

    server = proxy.raw.get("server") or {}
    served = [m for m in server.get("models") or [] if isinstance(m, str)]
    if not model:
        if len(served) == 1:
            model = served[0]
        else:
            listed = f": {', '.join(served)}" if served else ""
            raise OperationError(
                "name the model the harness should request",
                hint=f"the server lists {len(served) or 'no'} model(s){listed}",
                field="model",
            )

    workdir = Path(workdir) if workdir else store.home / "batches"
    # What the harness needs of the server: refused here, or noted when never measured.
    notes: list[Note] = [Note("note", text) for text in check_harness(store, adapter, proxy)]
    cached = cached_prompts_warning(store.server_checks((proxy.raw.get("server") or {}).get("id")))
    if cached:
        notes.append(Note("warning", cached))
    # The recordings, the runs' folder and `hide` are tuningfork's own: no check sees them either.
    sandbox, unsandboxed = for_launches([store.home, workdir, spec.directory, *hide], own=[store.home, workdir, *hide])
    if sandbox is None and not sandbox_turned_off():
        notes.append(Note(
            "warning",
            f"runs are not sandboxed: {unsandboxed}. Each sees its own folders, whose paths "
            "differ from run to run, and the model reads them; and the task's check runs on this "
            "machine itself, where it can run what the model wrote",
        ))
    if sandbox is not None and sandbox.clock_problem:
        notes.append(Note(
            "warning", f"{sandbox.clock_problem}: each run's date is the day it is made, which the model reads",
        ))
    # Read now, so a list that cannot be read is refused before a run is launched.
    machine_env = configured_check_env(store.home)
    if machine_env and check_kind(item.tuningfork.check) == "shell":
        notes.append(Note(
            "note",
            f"each run's check may read {', '.join(machine_env)}, which this machine's config names for every check, "
            "besides what its task names",
        ))
    if sandbox is not None and check_kind(item.tuningfork.check) == "shell" and not sandbox.shows(spec.directory):
        notes.append(Note(
            "warning",
            f"the task file is in {spec.directory}, which a sandboxed check never sees: the check will not "
            "find what is kept beside the task file. Move the task file into a folder of its own",
        ))
    if item.tuningfork.held_out:
        notes.append(Note(
            "warning",
            f"eval {item.id} is held out, and its runs will be recorded here, where an agent "
            "reading these recordings can see them: it no longer tests what it was held out for",
        ))
    if context is not None:
        notes.append(Note(
            "note",
            f"each run continues a conversation of {len(context.before)} message(s) ({item.tuningfork.context}), put "
            f"after {adapter.name}'s instructions in every call of the run: its prompt is that conversation's next turn"
            + (f"; {len(context.in_turn)} more, the model's earlier steps of that turn, follow the prompt, so its first "
               "live answer comes after them" if context.in_turn else ""),
        ))
    if item.tuningfork.stop_at_next_skill:
        notes.append(Note(
            "note",
            f"each run ends where the model loads a skill other than {spec.skill_name}, and the check judges the "
            "workspace as it is then" + ("; a check of the final reply finds none there"
                                         if isinstance(item.tuningfork.check, dict) else ""),
        ))
    if skill_dir and skill_dir.exists():
        lookalikes = lookalike_skill_files(skill_dir)
        if lookalikes:
            notes.append(Note(
                "warning",
                f"the skill folder also holds {', '.join(lookalikes)}. Every run is given a copy "
                "of the whole folder, so the model can read it. Keep variants outside the "
                "skill folder.",
            ))
    if sandbox is None:
        # In the sandbox, the harness works in a folder of its own, under nothing but its home.
        for warning in getattr(adapter, "isolation_warnings", lambda _: [])(workdir):
            notes.append(Note("warning", warning))
    if concurrency > proxy.max_inflight:
        notes.append(Note(
            "note",
            f"the proxy lets {proxy.max_inflight} call(s) through at once, so {concurrency} "
            "runs at a time will mostly wait for each other (restart it with --max-inflight "
            "to change that)",
        ))

    # Recorded with every run, so a later comparison can tell whether two batches were even
    # measured under the same conditions. The KV-cache quantization and context length
    # differences found in Phase 0 were invisible until someone went looking.
    parity = {
        "base_url": base_url,
        "model": model,
        "harness": harness,
        "server_defaults": server.get("defaults") or _server_defaults(base_url, store.home),
    }
    harness_settings = (getattr(adapter, "settings", None) or {}).get("harness")
    if harness_settings:
        parity["harness_options"] = harness_settings
    return BatchPlan(
        spec=spec, item=item, n=n, harness=harness, model=model, base_url=base_url,
        proxy=proxy, workdir=workdir, concurrency=concurrency, skill_dir=skill_dir,
        skill_file=Path(skill_file) if skill_file else None, parity=parity, notes=notes,
        adapter=adapter, sandbox=sandbox, harness_options=dict(harness_options or {}),
    )


def _server_defaults(base_url: str, home: Path) -> dict[str, Any]:
    """Best-effort read of the server's own generation defaults, through the proxy. The
    harness may send no sampling parameters at all, in which case these *are* the settings
    the run happened under."""
    import asyncio

    import httpx

    from tuningfork.endpoints import guard_hooks
    from tuningfork.proxy.recorder import fetch_server_defaults

    async def fetch() -> dict[str, Any]:
        async with httpx.AsyncClient(event_hooks=guard_hooks(home=home)) as client:
            return await fetch_server_defaults(client, base_url)

    try:
        return asyncio.run(fetch())
    except Exception:  # noqa: BLE001 - a missing parity block beats a failed batch
        return {}


@dataclass(slots=True)
class BatchReport:
    plan: BatchPlan
    outcomes: list[Any]  # runner.RunOutcome
    #: The check's pass rate over the runs that count; None without a check or a countable run.
    rate: Any
    #: Where the runs stopped agreeing, when there are two or more.
    divergence: Any
    elapsed: float

    @property
    def task_id(self) -> str:
        return self.plan.task_id

    @property
    def checked(self) -> bool:
        return all(outcome.checked for outcome in self.outcomes)

    @property
    def usable(self) -> list[Any]:
        return [outcome for outcome in self.outcomes if outcome.usable]

    @property
    def invalidated(self) -> list[Any]:
        return [outcome for outcome in self.outcomes if not outcome.usable]

    @property
    def run_ids(self) -> list[str]:
        return [outcome.run_id for outcome in self.outcomes if outcome.run_id]

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runs": [outcome.to_json() for outcome in self.outcomes],
            "checked": self.checked,
            "rate": self.rate.to_json() if self.rate else None,
            "not_counted": len(self.invalidated),
            "divergence": self.divergence.to_json() if self.divergence is not None else None,
            "elapsed": round(self.elapsed, 2),
        }


def execute_batch(
    store: Store,
    plan: BatchPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    group: str | None = None,
    start: int = 0,
    skill_dir: Path | None = None,
    skill_snapshot: str = "skill-snapshot",
    settings: dict[str, Any] | None = None,
    tell: str | None = None,
    launch_seconds: float | None = None,
) -> BatchReport:
    """Run the batch. Reports `launch` as each run's harness is launched and `run` as each
    finishes. Cancelled, it raises `Cancelled` whose `partial` is the report over the runs
    that finished. The rest is for an operation running the batch as part of something larger
    (`BatchRunner.run`): a chain's links share one group, numbered in order, each given a
    skill folder of its own, told the check's result (`tell`, a template) as one more turn, and
    given at most `launch_seconds` for each launch's processes together."""
    from tuningfork.harness import ADAPTERS
    from tuningfork.harness.base import HarnessError
    from tuningfork.runner import BatchRunner

    runner = BatchRunner(
        store,
        plan.adapter or ADAPTERS[plan.harness](),
        base_url=plan.base_url,
        model=plan.model,
        workdir=plan.workdir,
        skill_file=plan.skill_file,
        concurrency=plan.concurrency,
        sandbox=plan.sandbox,
        tell=tell,
    )
    runner.on_launch = lambda launch, index: emit(
        progress, "launch", subject=launch, launch_id=launch.id, index=index, label=launch.label
    )
    unregister = (
        cancel.on_cancel(lambda: runner.cancel.cancel(cancel.reason or "cancelled"))
        if cancel is not None
        else None
    )
    started = time.monotonic()
    try:
        result = runner.run(
            plan.spec, plan.item, plan.n, parity=plan.parity, launch_seconds=launch_seconds,
            on_progress=lambda outcome: emit(progress, "run", subject=outcome, **outcome.to_json()),
            group=group, start=start, skill_dir=skill_dir or plan.skill_dir,
            skill_snapshot=skill_snapshot, settings=settings,
        )
    except Cancelled as stopped:
        partial = _report(store, plan, stopped.partial or [], time.monotonic() - started)
        raise Cancelled(stopped.reason, partial=partial) from None
    except (HarnessError, ValueError) as exc:
        raise OperationError(str(exc)) from None
    finally:
        if unregister is not None:
            unregister()
    return _report(store, plan, result.outcomes, time.monotonic() - started)


def _report(store: Store, plan: BatchPlan, outcomes: list[Any], elapsed: float) -> BatchReport:
    from tuningfork.analysis.divergence import compare_runs
    from tuningfork.analysis.stats import rate

    usable = [outcome for outcome in outcomes if outcome.usable]
    checked = all(outcome.checked for outcome in outcomes)
    passes = sum(1 for outcome in usable if outcome.passed)
    run_ids = [outcome.run_id for outcome in outcomes if outcome.run_id]
    divergence = (
        compare_runs(store, run_ids, plan.item.tuningfork.normalizers) if len(run_ids) >= 2 else None
    )
    return BatchReport(
        plan=plan,
        outcomes=outcomes,
        rate=rate(passes, len(usable)) if checked and usable else None,
        divergence=divergence,
        elapsed=elapsed,
    )
