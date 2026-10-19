"""Command-line interface.

Every operation exists here and in the Python API before any UI is built, because a later
phase has an LLM driving these same operations.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tuningfork import __version__
from tuningfork.analysis.divergence import DivergenceReport, compare_runs
from tuningfork.analysis.normalize import DEFAULT_NORMALIZERS, recorded_outcome
from tuningfork.analysis.stats import p_short, samples_needed
from tuningfork.backends.base import BackendProfile, Support
from tuningfork.backends.probe import probe_backend
from tuningfork.config import Settings, default_home
from tuningfork.harness.base import HarnessError
from tuningfork.ops import OperationError
from tuningfork.ops.catalog import CACHE_HELP, SEED_HELP
from tuningfork.runner import pass_rate
from tuningfork.specs import CHECK_KEPT
from tuningfork.store.db import Run, Store
from tuningfork.store.tree import describe_message

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A step debugger and experiment lab for agent skills on local LLMs.",
)
console = Console()

SUPPORT_STYLE = {Support.YES: "green", Support.NO: "red", Support.UNKNOWN: "yellow"}
CLUSTER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: For a step argument: a number, or the tool the model called there.
STEP_HELP = "The step: its number, or a tool the model called there (Write: the first write)."


#: Stores and clients opened by the running command, closed by `_lifecycle` when it ends.
_closables: list = []


def _close_all() -> None:
    while _closables:
        _closables.pop().close()


@app.callback()
def _lifecycle(ctx: typer.Context) -> None:
    # Registered on the group context, which closes after every command however it ends —
    # including typer.Exit. (`click.get_current_context()` returns None inside a Typer
    # 0.27 command, so a hook registered from within the command would never fire.)
    ctx.call_on_close(_close_all)


def _store(home: Optional[Path]) -> Store:
    """Open the store for this command; closed when the command finishes."""
    store = Store(home or default_home())
    _closables.append(store)
    return store


def _clip(text: str, full: bool, limit: int = 300) -> str:
    """Shorten model- or harness-written text for display, escaped for Rich.

    Escaping is not optional: payloads are full of square brackets — JSON arrays, and
    Hermes's own `[IMPORTANT: ...]` markers — which Rich would otherwise parse as markup.
    """
    text = str(text)
    if full or len(text) <= limit:
        return escape(text)
    return f"{escape(text[:limit])}… [dim](+{len(text) - limit} chars)[/dim]"


def _verdict(result: dict) -> str:
    """A batch run's check verdict, flagged when the run cannot be counted."""
    if not result:
        return "-"
    if result.get("cancelled"):
        return "[dim]cancelled[/dim]"
    passed = result.get("passed")
    if passed is None:
        verdict = "no check"
    else:
        verdict = "[green]pass[/green]" if passed else "[red]fail[/red]"
    return verdict if result.get("usable", True) else f"{verdict} [dim](not counted)[/dim]"


def _shown_path(path: Path) -> str:
    """A path as the user would type it: relative to where they are, when it is inside."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _when(timestamp: float, seconds: bool = True) -> str:
    return datetime.fromtimestamp(timestamp).strftime(f"%m-%d %H:%M{':%S' if seconds else ''}")


def _resolve_run(store: Store, reference: str) -> Run:
    run = store.get_run(reference)
    if run is None:
        console.print(f"[red]no run matching {reference!r}[/red]")
        raise typer.Exit(code=1)
    return run


@app.command()
def version() -> None:
    """Print the tuningfork version."""
    console.print(f"tuningfork {__version__}")


@app.command()
def serve(
    upstream: str = typer.Option(
        ...,
        "--upstream",
        "-u",
        envvar="TUNINGFORK_UPSTREAM",
        help="Inference server root, e.g. http://127.0.0.1:8081. "
        "Defaults to TUNINGFORK_UPSTREAM, which resampling reads too.",
    ),
    host: str = typer.Option(
        "127.0.0.1", envvar="TUNINGFORK_HOST", help="Bind address. Keep it local."
    ),
    port: int = typer.Option(8787, "--port", "-p", envvar="TUNINGFORK_PORT"),
    max_inflight: int = typer.Option(
        1,
        "--max-inflight",
        envvar="TUNINGFORK_MAX_INFLIGHT",
        help="Concurrent upstream requests. Keep at 1 when the server is shared; at least "
        "the --concurrency of any forks run through it.",
    ),
    home: Optional[Path] = typer.Option(None, help="Where recordings are stored."),
    jobs: int = typer.Option(
        1,
        "--jobs",
        envvar="TUNINGFORK_JOBS",
        min=1,
        help="Background jobs (batches, resamples, A/Bs, forks) run at once. On one GPU, "
        "more at once was measured slower.",
    ),
    mcp_tools: str = typer.Option(
        "evidence", "--mcp-tools", envvar="TUNINGFORK_MCP_TOOLS",
        help="The tools /_tuningfork/mcp offers an agent: evidence (reads, and work that only "
        "calls the model), refine (the refinement loop: proposals and screens), refine-to-end (the "
        "loop, and screens whose samples are forks run to the end through the harness on this "
        "machine), all (also harness runs and checks), or none (no MCP route).",
    ),
    mcp_results: str = typer.Option(
        "both", "--mcp-results", envvar="TUNINGFORK_MCP_RESULTS",
        help="What a tool result carries: both its text and its structured content (a client shows "
        "the model one of them: Claude Code the structured content, OpenJarvis the text), or text "
        "alone, for a client that shows both — Hermes, which otherwise sees every answer twice.",
    ),
    mcp_max_wait: int = typer.Option(
        50, "--mcp-max-wait", envvar="TUNINGFORK_MCP_MAX_WAIT", min=1, max=600,
        help="How long one MCP call may wait for a job, at most, in seconds: under what the client "
        "gives a tool call. 50 for OpenJarvis, which reads an answer for 60 s; about 110 for Hermes, "
        "which gives a call 300 s, so a twenty-minute screen is a dozen polls, not twenty-five.",
    ),
    max_output: Optional[int] = typer.Option(
        None, "--max-output", envvar="TUNINGFORK_MAX_OUTPUT", min=1,
        help="Hold every call a harness makes through this proxy to at most this many output "
        "tokens: one that asks for more, or names no limit, is sent with this one, and what it "
        "asked is recorded with the step. For an agent whose reply can loop to its harness's own "
        "limit (65,536 tokens, 35 minutes on one GPU); a retry asking for more is held too.",
    ),
) -> None:
    """Run the recording proxy. Point your harness at http://HOST:PORT/v1."""
    from tuningfork.access import ENV as TOKEN_ENV
    from tuningfork.access import MCP_ENV
    from tuningfork.endpoints import check_endpoint
    from tuningfork.proxy.app import ProxyServer, create_app

    settings = Settings(
        upstream_base_url=upstream,
        home=home or default_home(),
        host=host,
        port=port,
        max_inflight=max_inflight,
        jobs=jobs,
        token=os.environ.get(TOKEN_ENV) or None,
        mcp_token=os.environ.get(MCP_ENV) or None,
        mcp_tools=mcp_tools,
        mcp_results=mcp_results,
        mcp_max_wait=mcp_max_wait,
        max_output=max_output,
    )
    if mcp_tools not in ("evidence", "refine", "refine-to-end", "all", "none"):
        _refuse(OperationError(f"--mcp-tools is evidence, refine, refine-to-end, all or none, not {mcp_tools!r}"))
    if mcp_results not in ("both", "text"):
        _refuse(OperationError(f"--mcp-results is both or text, not {mcp_results!r}"))
    try:
        check_endpoint(upstream, home=settings.home, purpose="the proxy")
    except OperationError as error:
        _refuse(error)
    settings.home.mkdir(parents=True, exist_ok=True)
    try:
        proxy = create_app(settings)
    except ValueError as error:
        _refuse(OperationError(str(error)))
    access = proxy.state.access

    console.print(f"[bold]tuningfork[/bold] {__version__}")
    console.print(f"  listening on  http://{host}:{port}/v1")
    console.print(f"  forwarding to {upstream}")
    console.print(f"  recording to  {settings.home}")
    console.print(f"  max in flight {max_inflight}")
    if max_output is not None:
        console.print(f"  output cap    {max_output} tokens a call (what a harness asks above it is recorded)")
    console.print(f"  web UI        http://{host}:{port}/_tuningfork/ui/?token={access.token}", soft_wrap=True)
    if mcp_tools != "none":
        console.print(
            f"  MCP           http://{host}:{port}/_tuningfork/mcp  ({mcp_tools} tools"
            f"{', text results' if mcp_results == 'text' else ''}; its own token, "
            "`tuningfork token --mcp`)", soft_wrap=True,
        )
    console.print(
        "  [dim]the token lets the page change things; it is new each start, and "
        "`tuningfork token` prints this address again[/dim]"
    )

    # Not uvicorn.run: stopping must first release held calls and end the UI's event
    # streams, or the server waits on them forever. See `stop_live_work`.
    ProxyServer(proxy, host=host, port=port).run()


@app.command()
def token(
    proxy: Optional[str] = typer.Option(None, help="The proxy. Default: TUNINGFORK_PROXY."),
    raw: bool = typer.Option(False, "--raw", help="Print the token alone, for a script."),
    mcp: bool = typer.Option(
        False, "--mcp",
        help="Print the MCP token alone: for a client of /_tuningfork/mcp. It opens that endpoint "
        "and nothing else.",
    ),
    header_json: bool = typer.Option(
        False, "--header-json",
        help='Print {"Authorization": "Bearer <MCP token>"}: for an MCP client that runs a command '
        "for its headers, such as Claude Code's headersHelper.",
    ),
) -> None:
    """Print the running proxy's web UI address, with the access token that lets it act."""
    from tuningfork.control import ControlClient, ControlError

    try:
        with ControlClient(proxy) as client:
            if mcp or header_json:
                found = client.mcp_token
                if not found:
                    console.print(
                        f"[red]found no MCP token for the proxy at {escape(client.base_url)}[/red] — it is "
                        "in the recordings folder the proxy runs with, beside its access token"
                    )
                    raise typer.Exit(code=1)
                typer.echo(json.dumps({"Authorization": f"Bearer {found}"}) if header_json else found)
                return
            found = client.token
            valid = bool(found) and client.status().get("auth", {}).get("valid")
    except ControlError as error:
        console.print(f"[red]{escape(str(error))}[/red]")
        raise typer.Exit(code=1) from None
    if not valid:
        console.print(
            f"[red]found no access token for the proxy at {escape(client.base_url)}[/red] — "
            "it is in the recordings folder the proxy runs with, readable only by the user "
            "that started it"
        )
        raise typer.Exit(code=1)
    if raw:
        typer.echo(found)
        return
    typer.echo(f"{client.base_url}/_tuningfork/ui/?token={found}")


@app.command("mcp")
def mcp_server(
    home: Optional[Path] = typer.Option(
        None, help="The recordings to read. Default: the running proxy's, else ~/.tuningfork."
    ),
    proxy: Optional[str] = typer.Option(
        None, help="The proxy that runs jobs, recording into the same folder. Default: TUNINGFORK_PROXY."
    ),
    tools: str = typer.Option(
        "evidence", "--tools",
        help="evidence: reads, and work that only calls the model. refine: the refinement loop — "
        "refinements, proposed edits, screens. refine-to-end: the loop, and screens whose samples are "
        "forks run to the end (a harness on this machine). all: everything, also work that runs a "
        "harness or a check on this machine.",
    ),
    limit: list[str] = typer.Option(
        [], "--limit", metavar="KIND=N",
        help="The most samples, forks or runs one job of KIND may make (100 for checkpoint "
        "work, 20 for harness runs by default). Repeatable.",
    ),
    max_jobs: int = typer.Option(4, "--max-jobs", min=1, help="Jobs agents may have queued or running at once."),
    max_wait: int = typer.Option(120, "--max-wait", min=0, max=600, help="Longest a call waits for a job, in seconds."),
    results: str = typer.Option(
        "both", "--results",
        help="What a tool result carries: both its text and its structured content (Claude Code shows "
        "the model the structured content, OpenJarvis the text), or text alone, for a client that "
        "shows both — Hermes, which otherwise sees every answer twice.",
    ),
) -> None:
    """Serve tuningfork's tools to an agent over MCP, on stdio: for Claude Code, Hermes, OpenJarvis.

    Reading recordings needs no proxy; starting work needs `tuningfork serve` on the same
    recordings. Nothing but the protocol is written to stdout.
    """
    import importlib.util

    from tuningfork.control import default_proxy

    if importlib.util.find_spec("mcp") is None:
        typer.echo("tuningfork mcp needs the MCP SDK: install tuningfork[mcp] "
                   "(uv tool install 'tuningfork[mcp]')", err=True)
        raise typer.Exit(code=1)
    from tuningfork.mcp.stdio import parse_limits, serve

    try:
        serve(
            home=home, proxy_url=proxy or default_proxy(), toolset=tools,
            limits=parse_limits(limit), max_jobs=max_jobs, max_wait=max_wait, results=results,
        )
    except OperationError as error:
        typer.echo(f"tuningfork mcp: {error.message}" + (f" — {error.hint}" if error.hint else ""), err=True)
        raise typer.Exit(code=1) from None


@app.command()
def probe(
    base_url: str = typer.Argument(..., help="Inference server base URL."),
    model: Optional[str] = typer.Option(None, help="Model id; discovered if omitted."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the profile JSON here."),
    name: Optional[str] = typer.Option(None, help="Name for the profile."),
    prefix_cache: bool = typer.Option(
        True, help="Include the prefix-cache probe (sends one long prompt)."
    ),
    rounds: int = typer.Option(3, help="Rounds used for the latency measurement."),
    anonymize: bool = typer.Option(
        True, help="Strip home directories from the saved profile so it can be shared."
    ),
) -> None:
    """Ask a server what it can do, and record the answer.

    Probes run one at a time, so this is safe to point at a server that is in use.
    """
    try:
        profile = asyncio.run(
            probe_backend(
                base_url,
                model=model,
                name=name,
                include_prefix_cache=prefix_cache,
                performance_rounds=rounds,
            )
        )
    except OperationError as error:
        _refuse(error)
    _print_profile(profile)
    if out:
        written = profile.save(out, anonymize=anonymize)
        note = "" if anonymize else " [yellow](contains local paths)[/yellow]"
        console.print(f"\nwrote [bold]{written}[/bold]{note}")


def _print_profile(profile: BackendProfile) -> None:
    console.print(f"\n[bold]{profile.server}[/bold] at {profile.base_url}")
    console.print(f"model: {profile.model}")
    for note in profile.notes:
        console.print(f"  {note}")

    table = Table("capability", "support", "detail", box=None, pad_edge=False)
    for capability, support, detail in profile.summary_rows():
        style = SUPPORT_STYLE.get(Support(support), "")
        table.add_row(capability, f"[{style}]{support}[/{style}]", detail)
    console.print(table)

    performance = profile.performance
    if performance and "error" not in performance:
        console.print("\n[bold]performance[/bold]")
        console.print(f"  time to first token  {performance.get('mean_ttft_ms')} ms")
        console.print(f"  total per call       {performance.get('mean_total_ms')} ms")
        console.print(f"  tokens/sec           {performance.get('tokens_per_second')}")


@app.command()
def runs(
    limit: int = typer.Option(20, "--limit", "-n"),
    task: Optional[str] = typer.Option(None, help="Filter by task id."),
    called: Optional[str] = typer.Option(
        None, "--called", help="Only runs where the model called this tool, e.g. Write."
    ),
    at_step: Optional[int] = typer.Option(
        None, "--at-step", min=0, help="With --called: only runs that called it at this step."
    ),
    failed: bool = typer.Option(False, "--failed", help="Only runs the task's check failed, counted."),
    passed: bool = typer.Option(False, "--passed", help="Only runs the task's check passed, counted."),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print only run ids: RUN=$(tuningfork runs -q -n 1)"
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """List recorded runs, newest first."""
    from tuningfork.ops.inspect import runs_calling, runs_judged

    if at_step is not None and not called:
        _refuse(OperationError("--at-step says where a run called the tool that --called names",
                               hint="`--called search_files --at-step 0`"))
    if failed and passed:
        _refuse(OperationError("a run's check either passed it or failed it", hint="name one: --failed or --passed"))
    store = _store(home)
    verdict = "failed" if failed else "passed" if passed else None
    # Filtered before the limit, so `-n 1` is the newest run that matches.
    wide = 1_000_000 if verdict else limit
    if called:
        recorded = runs_calling(store, called, task=task, step=at_step, limit=wide)
    else:
        recorded = store.runs(limit=wide, task_id=task)
    if verdict:
        recorded = runs_judged(recorded, verdict)[:limit]
    if quiet:
        for run in recorded:
            typer.echo(run.id)
        return
    if not recorded:
        console.print(f"no runs recorded in {store.home}")
        raise typer.Exit()

    # A batch run's label starts with its task, so a task column would only repeat it and
    # push the label — how a run is named on the command line — out of a normal terminal.
    show_task = any(
        run.task_id and not (run.label or "").startswith(f"{run.task_id}#") for run in recorded
    )
    columns = ["run", "label", *(["task"] if show_task else []), "steps", "model", "started"]
    table = Table(*columns, "check", box=None)
    for run in recorded:
        table.add_row(
            run.id[:8],
            escape(run.label or "-"),
            *([escape(run.task_id or "-")] if show_task else []),
            str(run.step_count),
            escape(run.model or "-"),
            _when(run.started_at, seconds=False),
            _verdict(run.result),
        )
    console.print(table)
    stats = store.stats()
    unlisted = [
        f"{count} {what}"
        for count, what in (
            (stats["forks"], "fork(s): `tuningfork experiments`"),
            (stats["threads"], "subagent or side thread(s): `tuningfork show` on their run"),
        )
        if count
    ]
    console.print(
        f"\n{stats['runs']} run(s), {stats['steps']} step(s), "
        f"{stats['nodes']} unique message(s) in {store.home}"
    )
    if unlisted:
        console.print(f"[dim]not listed above: {escape('; '.join(unlisted))}[/dim]")


@app.command()
def tasks(
    task: Optional[str] = typer.Option(None, help="Only this task id."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Check pass rates by task: one row per harness, model and skill version.

    Runs of a task under a different wording or model are a different sample, so each
    condition gets its own rate rather than one pooled across them.
    """
    from tuningfork.analysis.conditions import conditions

    store = _store(home)
    found = [c for c in conditions(store.runs(limit=1_000_000, task_id=task)) if c.task_id]
    if not found:
        console.print(f"no task runs recorded in {store.home}")
        raise typer.Exit()

    # One table per task, harness and model: its rows are the skill versions run under it,
    # which is the comparison being made.
    groups: dict[tuple[str | None, ...], list] = {}
    for condition in sorted(found, key=lambda c: (c.task_id or "", c.first)):
        key = (condition.task_id, condition.harness, condition.model)
        groups.setdefault(key, []).append(condition)
    for position, (key, members) in enumerate(groups.items()):
        if position:
            console.print()
        console.print(f"[bold]{' · '.join(escape(part) for part in key if part)}[/bold]")
        # Named only when it varies: runs answered by different servers are different
        # samples, and it is the only time the difference needs saying.
        servers = len({condition.server_id for condition in members}) > 1
        columns = ["skill", *(["server"] if servers else []), "runs", "check passed", ""]
        table = Table(*columns, box=None)
        for condition in members:
            passed = condition.passed
            notes = []
            if condition.skill_file:
                notes.append(f"SKILL.md from {escape(Path(condition.skill_file).name)}")
            if condition.uncounted:
                notes.append(f"{condition.uncounted} not counted")
            table.add_row(
                condition.skill_version or "-",
                *([(condition.server_id or "unknown")[:8]] if servers else []),
                str(len(condition.runs)),
                passed.format() if passed else "-",
                f"[dim]{'; '.join(notes)}[/dim]" if notes else "",
            )
        console.print(table)
    console.print(
        "\n[dim]skill: a fingerprint of the skill files the runs were given. "
        "Intervals are Wilson 95%.[/dim]"
    )


def _rate_text(found: dict | None) -> str:
    """A rate as the API gives it, as `Rate.format` prints one: never without its count."""
    if not found:
        return "-"
    return f"{found['successes']}/{found['total']}  {found['point']:.0%}  [{found['low']:.0%}-{found['high']:.0%}]"


@app.command("skills")
def skills_command(
    as_json: bool = typer.Option(False, "--json", help="The list as the API gives it."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """The skills these recordings know — from runs, scenarios, saved variants and
    refinements — each with its tasks, runs and versions, and its own folder."""
    from tuningfork.ops.skills import skills

    found = skills(_store(home))
    if as_json:
        typer.echo(json.dumps([each.to_json() for each in found], indent=2, ensure_ascii=False))
        return
    if not found:
        console.print("no skills yet — run a task (`tuningfork batch <spec.json>`), or make a scenario")
        return
    table = Table("skill", "tasks", "runs", "versions", "latest run", "refinements", "its folder", box=None, pad_edge=False)
    for each in found:
        refinements = ", ".join(f"{count} {status}" for status, count in each.refinements.items() if count) or "-"
        folder = "-"
        if each.folders:
            folder = escape(_shown_path(Path(each.folders[0]))) + (f" (+{len(each.folders) - 1})" if len(each.folders) > 1 else "")
        table.add_row(escape(each.name), str(len(each.tasks)), str(each.runs), str(each.versions),
                      _when(each.latest, seconds=False) if each.latest else "-", refinements, folder)
    console.print(table)
    console.print("[dim]`tuningfork skill <name>` shows its versions, their pass rates and texts[/dim]")


@app.command("skill")
def skill_command(
    name: str = typer.Argument(..., help="The skill, as its tasks are named: <skill>/<eval id>."),
    text: Optional[str] = typer.Option(
        None, "--text", metavar="REF",
        help="Print one text of it: a version's fingerprint, file (its own SKILL.md now), variant:<name> or history:<file>.",
    ),
    diff: Optional[tuple[str, str]] = typer.Option(
        None, "--diff", metavar="A B", help="Two texts of it, line by line, named as --text names one."
    ),
    as_json: bool = typer.Option(False, "--json", help="The skill as the API gives it, texts included."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """One skill: each version with its pass rate under every condition it ran under and
    what its text is known as; its own folder now, with the changelog and old versions an
    approval keeps beside it; its saved variants and refinements."""
    from tuningfork.ops.skills import skill, skill_diff, skill_text

    store = _store(home)
    try:
        if text:
            body = skill_text(store, name, text).text
            typer.echo(body, nl=not body.endswith("\n"))
            return
        if diff:
            found_diff = skill_diff(store, name, diff[0], diff[1])
            if as_json:
                typer.echo(json.dumps(found_diff.to_json(), indent=2, ensure_ascii=False))
                return
            console.print(f"[red]- {escape(found_diff.before['label'])}[/red]", soft_wrap=True)
            console.print(f"[green]+ {escape(found_diff.after['label'])}[/green]", soft_wrap=True)
            if not found_diff.diff["hunks"]:
                console.print("  the same text, line endings aside")
            _print_diff(found_diff.diff["hunks"])
            return
        found = skill(store, name)
    except OperationError as error:
        _refuse(error)
    if as_json:
        typer.echo(json.dumps(found.to_json(), indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]skill {escape(found.name)}[/bold]")
    for folder in found.folders:
        now = (f"its SKILL.md is version {folder.version}" if folder.version
               else "its SKILL.md has not been run yet" if folder.text is not None else "[yellow]no SKILL.md there[/yellow]")
        console.print(f"  folder     {escape(_shown_path(folder.path))}", soft_wrap=True)
        console.print(f"             {now}")
        if folder.changelog is not None:
            console.print(f"             changelog beside it: {escape(folder.path.name)}.changelog.md"
                          + (f"; {len(folder.history)} old version(s) in {escape(folder.path.name)}.history/" if folder.history else ""))
    if found.tasks:
        console.print(f"  tasks      {escape(', '.join(found.tasks))}", soft_wrap=True)
    console.print(f"\n[bold]versions[/bold] ({len(found.versions)}, newest first)")
    for version in found.versions:
        known = f"  [dim]{escape('; '.join(version.known_as))}[/dim]" if version.known_as else ""
        console.print(f"  [bold]{version.fingerprint}[/bold]  {version.runs} run{'s' if version.runs != 1 else ''}"
                      f"  {_when(version.first, seconds=False)} – {_when(version.latest, seconds=False)}{known}", soft_wrap=True)
        for condition in version.conditions:
            where = " · ".join(escape(part) for part in (condition["task_id"], condition["harness"], condition["model"]) if part)
            extra = f"  [dim]{condition['uncounted']} not counted[/dim]" if condition["uncounted"] else ""
            console.print(f"      {where}   {_rate_text(condition['passed'])}{extra}", soft_wrap=True)
        if version.text is None:
            console.print("      [dim]its SKILL.md is no longer on disk or in a snapshot[/dim]")
    if not found.versions:
        console.print("  none yet: no run has been given this skill")
    if found.variants:
        console.print(f"\n[bold]saved variants[/bold] ({len(found.variants)})")
        for variant in found.variants:
            ran = f"  version {variant['version']}" if variant.get("version") else ""
            console.print(f"  {escape(variant['name'])}  {_when(variant['modified'], seconds=False)}{ran}")
    if found.refinements:
        console.print(f"\n[bold]refinements[/bold] ({len(found.refinements)})")
        for each in found.refinements:
            console.print(f"  {each['id'][:8]}  {each['status']}  run {each['run'][:8]} step {each['step']}  "
                          f"{_clip(each.get('target') or '', full=False, limit=70)}", soft_wrap=True)
    console.print("\n[dim]--text <version|file|variant:NAME> prints one text; --diff A B compares two. "
                  "Intervals are Wilson 95%.[/dim]")


@app.command("glossary")
def glossary_command(
    word: Optional[str] = typer.Argument(None, help="A word to look up; every term without one."),
    as_json: bool = typer.Option(False, "--json", help="As the API gives it."),
) -> None:
    """tuningfork's words, each defined once: as the Guide page shows them."""
    from tuningfork.ops.glossary import GROUPS, lookup

    try:
        found = lookup(word)
    except OperationError as error:
        _refuse(error)
    if as_json:
        typer.echo(json.dumps([term.to_json() for term in found], indent=2, ensure_ascii=False))
        return
    if word:
        for term in found:
            console.print(f"[bold]{escape(term.term)}[/bold]" + (f"  [dim]also {escape(', '.join(term.aka))}[/dim]" if term.aka else ""))
            console.print(f"  {escape(term.means)}", soft_wrap=True)
            if term.where:
                console.print(f"  [dim]where:[/dim] {escape(term.where)}", soft_wrap=True)
            if term.see:
                console.print(f"  [dim]see:[/dim] {escape(', '.join(term.see))}")
        return
    for group, title in GROUPS:
        members = [term for term in found if term.group == group]
        console.print(f"\n[bold]{escape(title)}[/bold]")
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold", no_wrap=True)
        table.add_column()
        for term in members:
            table.add_row(escape(term.term), escape(term.means))
        console.print(table)
    if any(len({c.server_id for c in members}) > 1 for members in groups.values()):
        console.print(
            "[dim]server: which server answered — `tuningfork servers` says what each one "
            "is; unknown means recorded before servers were.[/dim]"
        )


@app.command()
def servers(home: Optional[Path] = typer.Option(None)) -> None:
    """The servers that answered recorded calls, and what each said it was."""
    store = _store(home)
    found = store.servers()
    if not found:
        console.print(f"no servers recorded in {store.home}")
        raise typer.Exit()
    table = Table("server", "first seen", "runs", "build", "model file", "n_ctx", "template", box=None)
    for server_id, identity, first_seen, runs in found:
        table.add_row(
            server_id[:8],
            _when(first_seen, seconds=False),
            str(runs),
            escape(str(identity.get("build", "-"))),
            escape(str(identity.get("model_file", "-"))),
            str(identity.get("n_ctx", "-")),
            str(identity.get("chat_template_sha256", "-"))[:8],
        )
    console.print(table)


@app.command()
def show(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Show a run: its steps and what the model did at each."""
    store = _store(home)
    target = _resolve_run(store, run)

    console.print(f"[bold]run {target.id[:8]}[/bold]  {target.label or ''}")
    console.print(f"  task    {target.task_id or '-'}")
    console.print(f"  model   {target.model or '-'}")
    console.print(f"  started {_when(target.started_at)}")
    if target.forked_from_run:
        console.print(
            f"  forked from {target.forked_from_run[:8]} at depth {target.forked_at_depth}"
        )
    if target.is_thread and target.parent_run_id:
        spawned = f", from its step {target.spawned_at}" if target.spawned_at is not None else ""
        console.print(f"  thread  {target.role} of run {target.parent_run_id[:8]}{spawned}")
    if target.is_fork and target.parity.get("fork_of"):
        console.print(
            f"  fork    of run {target.parity['fork_of'][:8]}, live from step "
            f"{target.parity.get('fork_step')}"
        )
    if target.harness:
        console.print(f"  harness {target.harness}")
    launch = store.get_launch(target.launch_id) if target.launch_id else None
    if launch is not None and launch.settings.get("sandbox"):
        console.print(f"  sandbox {_sandbox_seen(launch.settings['sandbox'])}")
    if target.skill_hashes:
        files = f"{len(target.skill_hashes)} file(s)"
        replaced = target.parity.get("skill_file")
        if replaced:
            files += f", SKILL.md from {escape(str(replaced))}"
        console.print(f"  skills  {target.skill_version} ({files})")

    from tuningfork.ops.inspect import step_lines

    lines = step_lines(store, target)
    steps = [line.step for line in lines]

    def annotate(found) -> str:
        # An edit and an error can both be true: an edit that could not be delivered.
        parts = [_edit_note(found.edits)]
        if found.error and not (found.edits.get("response") or {}).get("dropped"):
            parts.append(f"[red]{escape(found.error)}[/red]")
        if found.served:
            parts.append(f"[dim]replayed from step {found.replay.get('source_idx')}[/dim]")
        if found.replay.get("mismatch"):
            parts.append("[yellow]not in the recording: the replay departed here[/yellow]")
        if found.substitution.get("spans"):
            parts.append("variant swapped in")
        if found.substitution.get("leaked"):
            parts.append("[yellow]the original wording reached the model[/yellow]")
        cap = found.params.get("output_cap")
        if cap and found.finish_reason == "length":
            parts.append(f"[yellow]cut at the proxy's cap, {cap.get('cap')} tokens[/yellow]")
        return " ".join(part for part in parts if part)

    from tuningfork.ops.inspect import hides_what_it_ran, ran_excerpt, step_changes

    notes = [annotate(found) for found in steps]
    # What each call changed in the workspace: where the run did what its check saw.
    changes = step_changes(store, target)
    columns = ["k", "outcome", "finish", "ms", "tokens", "think"]
    if changes:
        columns.append("changed")
    if any(notes):
        columns.append("note")
    table = Table(*columns, box=None)
    for line, note in zip(lines, notes):
        step, outcome = line.step, line.outcome
        # A script's label is a digest of it: what it ran says which step did what.
        said = (ran_excerpt(store.response(step)) if hides_what_it_ran(outcome.label) else None) or outcome.label
        row = [
            str(step.idx),
            _clip(said, full=False, limit=70),
            step.finish_reason or "-",
            str(step.timings.get("total_ms") or "-"),
            str(step.usage.get("total_tokens") or "-"),
            "[yellow]none[/yellow]" if line.reasoning == "none" else str(step.reasoning_chars or "-"),
        ]
        if changes:
            row.append(escape(", ".join(path for _, path in changes.get(step.idx, []))[:40]) or "")
        if any(notes):
            row.append(note)
        table.add_row(*row)
    console.print(table)
    if any(step.edits for step in steps):
        console.print(
            "\n[dim]An edited step's outcome is what the model produced; `tuningfork step "
            "<run> <k> --harness-view` shows what the harness saw instead.[/dim]"
        )

    threads = [] if target.is_thread else store.threads_of(target.id)
    if threads:
        # A subagent's conversation, or the harness's own side calls: the same launch, its
        # own messages. Each is a run of its own to `show` or `step` into.
        console.print("\n[bold]threads[/bold]")
        listing = Table("run", "role", "from step", "steps", box=None)
        for thread in threads:
            listing.add_row(
                thread.id[:8],
                thread.role or "-",
                "-" if thread.spawned_at is None else str(thread.spawned_at),
                str(thread.step_count),
            )
        console.print(listing)


def _content_text(content) -> str:
    from tuningfork.analysis.normalize import content_text

    return content_text(content)


def _edit_note(edits: dict) -> str:
    """What a breakpoint changed at a step, in a few words."""
    parts = []
    if edits.get("request"):
        parts.append("context edited")
    response = edits.get("response") or {}
    if response.get("dropped"):
        parts.append("dropped")
    elif response:
        parts.append("response edited")
    return f"[yellow]{', '.join(parts)}[/yellow]" if parts else ""


@app.command()
def step(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    home: Optional[Path] = typer.Option(None),
    full: bool = typer.Option(False, help="Print whole messages instead of a preview."),
    harness_view: bool = typer.Option(
        False,
        "--harness-view",
        help="For a step edited at a breakpoint: the context the harness sent and the "
        "response it received, instead of what the model saw and produced.",
    ),
) -> None:
    """Show one step: the exact context the model saw, and what it did."""
    from tuningfork.ops.timeline import step_detail

    store = _store(home)
    target = _resolve_run(store, run)
    index = _step_index(store, target, reference)
    found = store.step(target.id, index)
    if found is None:
        console.print(f"[red]run {target.id[:8]} has no step {index}[/red]")
        raise typer.Exit(code=1)
    detail = step_detail(store, target.id, index).to_json()

    console.print(f"[bold]run {target.id[:8]} step {index}[/bold]")
    if found.edits:
        view = "the harness's view" if harness_view else "the model's view"
        console.print(f"  edited  {_edit_note(found.edits)} at a breakpoint — showing {view}")
    console.print(f"  model   {found.model}")
    if found.wire:
        console.print(f"  wire    {found.wire}")
    console.print(f"  params  {json.dumps(found.params, sort_keys=True)}")
    cap = found.params.get("output_cap")
    if cap:
        asked = ", ".join(f"{key} {value}" for key, value in (cap.get("asked") or {}).items()) or "no limit"
        console.print(f"  capped  the harness asked {asked}; the proxy's cap sent {cap.get('cap')}")
    if found.server_defaults:
        console.print(f"  server  {json.dumps(found.server_defaults, sort_keys=True)}")
    console.print(f"  timings {json.dumps(found.timings)}")
    if found.usage:
        console.print(f"  usage   {json.dumps(found.usage)}")

    console.print("\n[bold]context[/bold]")
    # Shown as OpenAI-style messages whatever the wire format, so every harness reads alike.
    shown = (detail["harness_view"] if harness_view else None) or detail["view"]
    for position, message in enumerate(shown):
        summary = escape(describe_message(message, 100))
        console.print(f"  {escape(f'[{position}]')} [cyan]{summary}[/cyan]")
        if full:
            console.print(f"      {_clip(_content_text(message.get('content')), full=True)}")

    response = (detail["delivered_response"] if harness_view else None) or detail["response"]
    if not response:
        return
    message = (response.get("choices") or [{}])[0].get("message") or {}
    console.print("\n[bold]response[/bold]")
    if detail.get("reasoning") == "none":
        console.print(f"  [yellow]{_no_reasoning(detail)}[/yellow]")
    for key in ("reasoning_content", "reasoning"):
        if message.get(key):
            console.print(f"  [magenta]{key}[/magenta]: {_clip(message[key], full)}")
    if message.get("content"):
        console.print(f"  content: {_clip(message['content'], full)}")
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        console.print(
            f"  [yellow]tool_call[/yellow] {function.get('name')}"
            f"({_clip(function.get('arguments', ''), full)})"
        )


def _no_reasoning(step: dict) -> str:
    """A step answered without reasoning, though the model reasons at others: in words."""
    return "no reasoning" + (" — right after a tool call failed" if step.get("after_failure") else "")


def _timeline_flags(step: dict) -> list[str]:
    """What a step on the timeline carries besides its outcome, in a few words each."""
    flags = []
    if step.get("reasoning") == "none":
        flags.append(_no_reasoning(step))
    replay = step.get("replay") or {}
    if step["served"]:
        flags.append(f"replayed from step {replay.get('source_idx')} of run {str(replay.get('source_run'))[:8]}")
    elif replay.get("live_from") is not None:
        flags.append("live from here")
    if step["variant"]:
        flags.append("variant swapped in")
    if step["edited"]["request"]:
        flags.append("context edited at a breakpoint")
    if step["edited"]["response"]:
        flags.append(f"answer {step['edited']['response']} at a breakpoint")
    kept = step["kept"]
    if kept == "retried":
        flags.append(f"not kept: sent again at step {step['retried_by']}")
    elif kept == "differs":
        flags.append("the harness kept a different answer")
    elif kept == "missing":
        flags.append("the harness kept no answer")
    changed = step.get("extras_changed")
    if changed:
        tools = [f"+{name}" for name in changed["tools_added"]] + [f"-{name}" for name in changed["tools_removed"]]
        flags.append("request changed: " + ", ".join(changed["fields"]) + (f" ({' '.join(tools)})" if tools else ""))
    if step["off_path"]:
        flags.append("off the run's path")
    if step["experiments"]:
        count = len(step["experiments"])
        flags.append(f"{count} experiment{'' if count == 1 else 's'}")
    return flags


@app.command()
def timeline(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    full: bool = typer.Option(False, help="Print whole messages, and each step's reasoning whole."),
    as_json: bool = typer.Option(False, "--json", help="The timeline as JSON: what the web UI reads."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """A run as one conversation: its last context read once, each step's answer where it
    happened — reasoning included, from the recording."""
    from tuningfork.ops.timeline import run_timeline

    store = _store(home)
    try:
        found = run_timeline(store, run)
    except OperationError as error:
        _refuse(error)
    data = found.to_json()
    if as_json:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    target = found.run
    console.print(f"[bold]run {target.id[:8]}[/bold]  {escape(target.label or '')}")
    count = len(data["steps"])
    facts = [
        f"task {target.task_id or '-'}", f"model {target.model or '-'}",
        f"{count} step{'' if count == 1 else 's'}", f"started {_when(target.started_at)}",
    ]
    console.print("  " + escape(" · ".join(facts)))
    # The tools the first call offered: no message holds them, but the model reads them.
    offered = data["tools_offered"]
    if offered:
        names = ", ".join(tool["name"] for tool in offered)
        console.print("  " + escape(f"{len(offered)} tool{'' if len(offered) == 1 else 's'} offered: ") + _clip(names, full, limit=160))
        if full:
            for tool in offered:
                gist = tool["description"].strip().split("\n", 1)[0]
                console.print(f"      {escape(tool['name'])}" + (f"  [dim]{_clip(gist, False, limit=120)}[/dim]" if gist else ""))
    else:
        imported = (target.parity or {}).get("imported")
        said = ("no tools offered" if not imported
                else "no tools recorded: Hermes's export keeps none (import it again with --tools-from <run>)"
                if imported.get("format") == "hermes-export" else "no tools recorded: the file named none")
        console.print(f"  [dim]{escape(said)}[/dim]")
    console.print()

    skills: dict[int, list[dict]] = {}
    for span in data["skills"]:
        skills.setdefault(span["message_index"], []).append(span)
    at: dict[int, list[dict]] = {}
    for step in data["steps"]:
        at.setdefault(step["position"], []).append(step)
    threads: dict[int, list[dict]] = {}
    for thread in data["threads"]:
        if thread["after"] is not None:
            threads.setdefault(thread["after"], []).append(thread)
    missing = sorted(data["missing"])

    def show_step(step: dict) -> None:
        while missing and missing[0] < step["idx"]:
            console.print(f"[dim]step {missing.pop(0)} · not recorded[/dim]")
        parts = [f"step {step['idx']}", step["outcome"]["label"]]
        if step["finish_reason"]:
            parts.append(step["finish_reason"])
        tokens = (step["usage"] or {}).get("total_tokens")
        if tokens:
            parts.append(f"{tokens:,} tokens")
        took = (step["timings"] or {}).get("total_ms")
        if took is not None:
            parts.append(f"{took / 1000:.1f}s")
        console.print(f"[bold]{escape(' · '.join(parts + _timeline_flags(step)))}[/bold]")
        output = step["output"] or {}
        reasoning = output.get("reasoning_content") or output.get("reasoning")
        if reasoning:
            shown = reasoning if full else reasoning.strip().split("\n", 1)[0]
            console.print(f"      [magenta]reasoning[/magenta]  {_clip(shown, full, limit=120)}")
        said = _content_text(output.get("content")).strip()
        if said:
            console.print(f"      says       {_clip(said, full, limit=200)}")
        for call in output.get("tool_calls") or []:
            function = call.get("function") or {}
            console.print(
                f"      [yellow]tool call[/yellow]  {escape(str(function.get('name')))}"
                f"({_clip(function.get('arguments', ''), full, limit=120)})"
            )
        for thread in threads.get(step["idx"], []):
            where = " (around here)" if thread["placed"] == "time" else ""
            console.print(f"      → {thread['role']} {thread['run_id'][:8]} · {thread['steps']} steps{where}")

    for entry in data["entries"]:
        for step in at.get(entry["index"], []):
            show_step(step)
        if entry["origin"] == "copy":
            continue  # the step above shows its own answer, from the recording
        message = entry["message"]
        console.print(f"  {escape(f'[{entry['index']}]')} [cyan]{escape(describe_message(message, 100))}[/cyan]")
        if full:
            console.print(f"      {_clip(_content_text(message.get('content')), full=True)}")
        for span in skills.get(entry["index"], []):
            first = span.get("first_line") or 1
            where = f"{span['file']}, lines {first}–{first + span['lines'] - 1}" if span.get("file") else "no recorded file matches"
            console.print(f"      [magenta]skill[/magenta] {escape(span['name'])} · {escape(where)}")
    for step in at.get(len(data["entries"]), []):
        show_step(step)
    for gap in missing:
        console.print(f"[dim]step {gap} · not recorded[/dim]")
    console.print(
        "\n[dim]"
        + escape("[n] is a message's place in the last step's context; `tuningfork step <run> <k>` "
                 "shows exactly what a step was sent.")
        + "[/dim]"
    )


#: Terminal backgrounds for a heatmap's levels 1-7 (level 0 is left plain).
HEAT_STYLES = ("on grey19", "on grey27", "on #5f5f00", "on #875f00", "on #af5f00", "on #af0000", "on #d70000")
PART_NAMES = {"reasoning": "reasoning", "content": "answer", "tool_call": "tool call"}


def _print_heatmap(view: Any, color: bool) -> None:
    """The stream's text by part, each token shaded by how unsure the model was of it."""
    from rich.text import Text

    from tuningfork.analysis.tokens import display_groups, heat

    stream = view.stream
    line: Text | None = None
    current = None
    for start, end, text in display_groups(stream):
        part = stream.part(start)
        if part == "marker":
            continue
        if part != current:
            if line is not None:
                console.print(line)
            current = part
            console.print(f"[bold]{PART_NAMES.get(part, 'answer')}[/bold]")
            line = Text("  ")
        level = max(heat(stream.entropy(position)) for position in range(start, end))
        style = HEAT_STYLES[level - 1] if color and level else ""
        shown = text.replace("\n", "\n  ")
        line.append(shown, style=style)
    if line is not None:
        console.print(line)


@app.command()
def tokens(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    stream: Optional[str] = typer.Option(None, "--stream", help="A stream at the step, by its id (see the list printed)."),
    sample: Optional[str] = typer.Option(None, "--sample", help="A sample's tokens, as EXPERIMENT:INDEX (resampled with --logprobs)."),
    score: bool = typer.Option(
        False, "--score",
        help="Score the answer the run gave first: its exact context sent once more, with the "
        "answer forced, reading the model's distribution at each token.",
    ),
    top_k: int = typer.Option(8, "--top-k", min=1, max=20, help="With --score: alternatives to keep at each token."),
    base_url: Optional[str] = typer.Option(
        None, help="With --score: the server. Defaults to TUNINGFORK_UPSTREAM, else http://127.0.0.1:8081."
    ),
    hotspots: int = typer.Option(10, "--hotspots", help="How many forks to list."),
    color: str = typer.Option("auto", "--color", help="Shade the text by entropy: auto (on a terminal), always, never."),
    as_json: bool = typer.Option(False, "--json", help="The tokens as JSON: what the web UI reads."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """A step's answer as tokens: how sure the model was of each, and where the step's own
    sampler could most easily have drawn another."""
    from tuningfork.ops.tokens import execute_score, prepare_score, sample_tokens, step_tokens

    store = _store(home)
    try:
        if score:
            prepared = prepare_score(store, run, reference, top_k=top_k, base_url=base_url)
            usage = prepared.checkpoint.step.usage or {}
            console.print(
                f"[dim]scoring step {prepared.checkpoint.step.idx}'s answer "
                f"({usage.get('completion_tokens') or '?'} tokens) on {escape(prepared.base_url)}…[/dim]"
            )
            found = asyncio.run(execute_score(store, prepared))
        elif sample:
            experiment, _, index = sample.partition(":")
            if not index.isdigit():
                raise OperationError("--sample is EXPERIMENT:INDEX, e.g. 3f2a1b9c:4")
            found = sample_tokens(store, experiment, int(index))
        else:
            found = step_tokens(store, run, reference, stream=stream)
    except OperationError as error:
        _refuse(error)
    if as_json:
        typer.echo(json.dumps(found.to_json(hotspots=hotspots), indent=2, ensure_ascii=False))
        return
    view = found.view
    where = f"run {found.run_id[:8]} step {found.step_idx}"
    if view is None:
        console.print(f"{where}: the answer the run gave has not been scored yet.")
        console.print(
            f"  [dim]`tuningfork tokens {found.run_id[:8]} {found.step_idx} --score` scores it, "
            "token by token, on the server.[/dim]"
        )
        if found.streams:
            console.print("\nsamples drawn here kept their tokens:")
            for entry in found.streams:
                console.print(f"  {entry['id'][:8]}  sample {entry['sample_idx']} of experiment "
                              f"{(entry['experiment_id'] or '')[:8]}: {escape(entry['outcome'] or '')}")
            console.print(f"  [dim]`tuningfork tokens {found.run_id[:8]} {found.step_idx} --stream <id>` shows one[/dim]")
        return
    record = view.record
    facts = [record.source, record.method or "-", f"top {view.stream.top_k}", f"{record.tokens} tokens"]
    if record.exact is True:
        facts.append("rebuilt exactly")
    elif record.exact is False:
        facts.append("rebuilt approximately")
    if view.same_server is False:
        facts.append("scored on a different server than the step ran on")
    console.print(f"[bold]{escape(where)}[/bold]  stream {record.id[:8]}")
    console.print("  " + escape(" · ".join(facts)))
    if view.sample:
        console.print(f"  sample {view.sample['idx']} of experiment {view.sample['experiment_id'][:8]}: "
                      f"{escape(view.sample['outcome'] or '')}")
    choice = view.choice()
    if choice is not None:
        from tuningfork.ops.compact import choice_text

        console.print(f"  choice of tool, at token {escape(f'[{choice.position}]')}: {escape(choice_text(choice.to_json()['chance']))}"
                      "  [dim]the model's own chance of each, under the step's sampler[/dim]")
    console.print()
    _print_heatmap(view, color == "always" or (color == "auto" and console.is_terminal))

    listed = view.hotspots(hotspots)
    if listed:
        console.print("\n[bold]where the step's sampler could most easily have drawn another token[/bold]")
    for candidate in listed:
        choices = [f"{candidate.chosen.text!r} {candidate.chosen.p:.0%}"] + [
            f"{alt.text!r} {alt.p:.0%}" for alt in candidate.alternatives
        ]
        entropy = "" if candidate.entropy is None else f"  H {candidate.entropy:.2f} bits"
        console.print(
            f"  {escape(f'[{candidate.position}]')} {PART_NAMES.get(candidate.part or '', '-'):9} "
            f"{escape(' | '.join(choices))}{entropy}  runner-up drawn {candidate.runner_up:.0%}"
        )
    others = [entry for entry in found.streams if entry["id"] != record.id]
    if others:
        console.print("\n[dim]other streams at this step:[/dim]")
        for entry in others:
            what = (
                f"sample {entry['sample_idx']} of experiment {entry['experiment_id'][:8]}: {entry['outcome']}"
                if entry["experiment_id"] else f"{entry['source']} {_when(entry['created_at'])}"
            )
            console.print(f"  [dim]{entry['id'][:8]}  {escape(what)}[/dim]")
    console.print(
        f"\n[dim]Entropy counts the probability beyond the top {view.stream.top_k} as one outcome: "
        "a lower bound. The runner-up's chance is under the step's own sampler settings.[/dim]"
    )


@app.command("forking-tokens")
def forking_tokens(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    at: Optional[str] = typer.Option(None, "--at", help="Positions to probe, comma-separated, as `tuningfork tokens` numbers them."),
    region: str = typer.Option("answer", "--in", help="Without --at, where to look for forks: answer, reasoning or all."),
    top: int = typer.Option(5, "--top", help="Without --at, how many positions to probe."),
    alternatives: int = typer.Option(3, "--alternatives", help="Other tokens to force at each position."),
    n: int = typer.Option(10, "-n", help="Continuations per forced token."),
    min_p: float = typer.Option(0.05, "--min-p", help="Least chance the step's sampler gave an alternative, to count as a fork."),
    target: Optional[str] = typer.Option(None, "--target", help="A behaviour's letter from an earlier resample of this step. Default: the base's own behaviour."),
    target_match: Optional[str] = typer.Option(None, "--target-match", help="Regular expression over the behaviour."),
    alpha: float = typer.Option(0.05, help="Significance level, Holm-corrected over the planned tests."),
    normalizer: list[str] = typer.Option([], "--normalizer", help="Restrict normalizers."),
    stream: Optional[str] = typer.Option(None, "--stream", help="Fork a sample's tokens instead, by stream id."),
    score: bool = typer.Option(False, "--score", help="Score the step's answer first, if it has no tokens yet."),
    max_tokens: int = typer.Option(4096, help="Cap per continuation; hitting it counts as truncated."),
    base_url: Optional[str] = typer.Option(None, help="Server to sample from. Defaults to TUNINGFORK_UPSTREAM, else http://127.0.0.1:8081."),
    seed_base: Optional[int] = typer.Option(None, help=SEED_HELP),
    concurrency: int = typer.Option(1, help="Continuations at once. Keep at 1 on a shared server."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; draw nothing."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Which tokens decided a step: force each likely alternative at its uncertain tokens,
    let the model go on, and test which change what it does (after Bigelow et al., 2025)."""
    from tuningfork.ops.forking import execute_forking, prepare_forking
    from tuningfork.ops.tokens import execute_score, prepare_score, step_tokens

    store = _store(home)
    try:
        positions = [int(p) for p in at.split(",") if p.strip()] if at else []
    except ValueError:
        console.print("[red]--at is a comma-separated list of positions, e.g. 412,418[/red]")
        raise typer.Exit(code=1) from None
    try:
        if score and stream is None and step_tokens(store, run, reference).view is None:
            prepared_score = prepare_score(store, run, reference, base_url=base_url)
            console.print(f"[dim]scoring step {prepared_score.checkpoint.step.idx}'s answer first…[/dim]")
            asyncio.run(execute_score(store, prepared_score))
        prepared = prepare_forking(
            store, run, reference, stream=stream, positions=positions, top=top, region=region,
            alternatives=alternatives, min_p=min_p, n=n, max_tokens=max_tokens, normalizers=normalizer,
            target=target, target_match=target_match, alpha=alpha, base_url=base_url,
            seed_base=seed_base, concurrency=concurrency,
        )
    except OperationError as error:
        _refuse(error)
    plan = prepared.to_json()
    checkpoint = prepared.checkpoint
    console.print(f"[bold]forking tokens at run {checkpoint.run.id[:8]} step {checkpoint.step.idx}[/bold] — "
                  f"{escape(checkpoint.describe())}")
    console.print(f"  base      stream {prepared.base.id[:8]} ({prepared.base.source}, {len(prepared.stream)} tokens)")
    console.print(f"  target    {escape(prepared.target.description)} (declared before sampling)")
    console.print(f"  arms      {len(prepared.arms)} forced tokens × {prepared.per_arm} = {prepared.planned} continuations")
    power = plan["power"]
    if power["detectable"]:
        console.print(f"  power     a forced token is found to change the outcome when about "
                      f"{power['detectable']} of its {power['of']} continuations change; smaller shifts need more (-n)")
    else:
        console.print(f"  [yellow]power     with {power['of']} continuations per token and {prepared.family} "
                      "test(s), no change can reach significance: raise -n[/yellow]")
    estimate = plan["estimate"]
    minutes = f", ~{estimate['minutes']} min at the step's own speed" if estimate["minutes"] else ""
    console.print(f"  estimate  ~{estimate['tokens']:,} tokens to generate{minutes}")
    if estimate.get("prompt_tokens") and estimate.get("prefill_seconds") and estimate["minutes"]:
        console.print(f"            counting the whole prompt, {estimate['prompt_tokens']:,} tokens, processed first for "
                      f"each continuation (~{estimate['prefill_seconds']}s)")
    elif estimate.get("prompt_tokens"):
        console.print(f"            and each continuation first processes the whole prompt, {estimate['prompt_tokens']:,} tokens")
    for warning in prepared.warnings:
        console.print(f"  [yellow]{escape(warning)}[/yellow]")
    for candidate in prepared.candidates:
        choices = " | ".join(f"{alt.text!r} {alt.p:.0%}" for alt in candidate.alternatives)
        console.print(f"  {escape(f'[{candidate.position}]')} {PART_NAMES.get(candidate.part or '', '-'):9} "
                      f"{escape(repr(candidate.chosen.text))} {candidate.chosen.p:.0%} → {escape(choices)}")
    if dry_run:
        return
    console.print()
    try:
        result = asyncio.run(
            execute_forking(store, prepared, progress=_progress_printer(prepared.per_arm, show_arm=True))
        )
    except OperationError as error:
        _refuse(error)
    _print_notes(result.experiment.notes)
    _print_forking(result.report)
    console.print(f"\n[dim]experiment {result.experiment.id[:8]} · {result.elapsed / 60:.1f} min · "
                  f"`tuningfork experiment {result.experiment.id[:8]}` shows it again[/dim]")


@app.command()
def recheck(
    run: Optional[str] = typer.Argument(
        None, help="A recorded run to judge again: its workspace as it ended, and its final reply."
    ),
    check: Optional[str] = typer.Option(
        None, "--check", help="A shell command to try. With a run and no check named, the run's own."
    ),
    reply_contains: Optional[str] = typer.Option(None, "--reply-contains", help="Try: the reply holds this text."),
    reply_matches: Optional[str] = typer.Option(
        None, "--reply-matches", help="Try: this regular expression finds a match in the reply."
    ),
    reply: Optional[str] = typer.Option(None, "--reply", help="Without a run: the reply to judge."),
    files: list[str] = typer.Option(
        [], "--file", help="Without a run: NAME=PATH puts the local file PATH in the workspace as NAME. Repeatable."
    ),
    timeout: float = typer.Option(60.0, help="Seconds the check may take."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Try a check before trusting it: judge a recorded run again, or a reply and files you
    give, in a throwaway folder. Nothing is recorded. Exits 1 when the check fails."""
    from tuningfork.ops.checks import try_check

    named = [
        value for value in (
            check, {"reply_contains": reply_contains} if reply_contains is not None else None,
            {"reply_matches": reply_matches} if reply_matches is not None else None,
        ) if value is not None
    ]
    if len(named) > 1:
        _refuse(OperationError("name one check: --check, --reply-contains or --reply-matches"))
    given = []
    for entry in files:
        name, sep, path = entry.partition("=")
        if not sep or not name or not path:
            _refuse(OperationError(f"--file takes NAME=PATH, not {entry!r}"))
        try:
            given.append({"path": name, "content": Path(path).read_text(encoding="utf-8")})
        except OSError as exc:
            _refuse(OperationError(f"cannot read {path}: {exc.strerror or exc}"))
    store = _store(home)
    try:
        trial = try_check(
            store, named[0] if named else None, run=run, reply=reply or "", files=given, timeout=timeout,
        )
    except OperationError as error:
        _refuse(error)
    data = trial.to_json()
    found = data["run"]
    if found:
        console.print(f"[bold]run {found['id'][:8]}[/bold] · {escape(found['task_id'] or '-')} · {escape(found['harness'] or '-')}")
    else:
        console.print("[bold]a reply and files you gave[/bold]")
    own = "  (the run's own)" if data["own"] else ""
    console.print(f"  check      {escape(data['check_text'])}{own}")
    where = "as the run left it" if data["workspace"] == "run" else "files you gave" if data["workspace"] == "files" else "empty"
    listed = ", ".join(data["found"]) + (f", and {data['found_more']} more" if data["found_more"] else "")
    console.print(f"  workspace  {where}{': ' + escape(listed) if listed else ''}")
    console.print(f"  reply      {_clip(data['reply'], False, limit=200) if data['reply'] else '[dim](none)[/dim]'}")
    console.print()
    verdict = "[green]pass[/green]" if trial.passed else "[red]fail[/red]"
    recorded = (data["recorded"] or {}).get("passed")
    if recorded is None:
        compared = ""
    elif trial.agrees:
        compared = " — as when it ran"
    else:
        compared = f" — the run was judged {'pass' if recorded else 'fail'} when it ran"
    console.print(f"{verdict}{compared}")
    if data.get("output_cut"):
        console.print(f"  [yellow]{data['output_cut']:,} characters it printed before these are not shown[/yellow]: a "
                      f"verdict keeps the last {CHECK_KEPT:,}, and only they are tallied")
    for line in (data["output"] or "").splitlines()[:40]:
        console.print(f"  {escape(line)}")
    if not trial.passed:
        raise typer.Exit(code=1)


#: How a conversation showed a skill, in words.
SKILL_SOURCES = {"preloaded": "preloaded in the system prompt", "tool": "loaded by a call",
                 "frontmatter": "found by its frontmatter"}


@app.command("import")
def import_chat(
    file: Path = typer.Argument(
        ..., help="A conversation: a Hermes session file (~/.hermes/sessions/session_<id>.json), a Hermes export "
        "(`hermes sessions export`), or an OpenAI-style list of messages.",
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="From a file holding several sessions, the one to import: its id, or its first characters.",
    ),
    skill: Optional[str] = typer.Option(
        None, "--skill", help="Where the conversation shows several skills — each is kept — the one an A/B, a "
        "refinement or --scenario studies when none is named.",
    ),
    skill_dir: Optional[list[Path]] = typer.Option(
        None, "--skill-dir", help="A skill's folder, kept with the run in place of the SKILL.md the model read: "
        "its scripts, templates and references too. Once for each skill that has one.",
    ),
    harness: Optional[str] = typer.Option(
        None, "--harness", help="The harness the conversation was had with, where the file does not say "
        "(hermes, claude-code, generic).",
    ),
    tools_from: Optional[str] = typer.Option(
        None, "--tools-from", help="Where the file names no tools: a run recorded through the same harness, whose "
        "tools each step offers.",
    ),
    label: Optional[str] = typer.Option(None, "--label", help="The run's label."),
    task: Optional[str] = typer.Option(None, "--task", help="A task id to file the run under."),
    scenario: Optional[str] = typer.Option(
        None, "--scenario", help="Also make a scenario of it, by this name: runs of the skill through the harness, "
        "each continuing the conversation from the user turn before the skill came in.",
    ),
    cut: Optional[int] = typer.Option(
        None, "--cut", help="With --scenario: the user turn it starts from, by its message's index, in place of the "
        "one before the skill came in.",
    ),
    check: Optional[str] = typer.Option(None, "--check", help="With --scenario: its check, a shell command run in each run's workspace."),
    reply_contains: Optional[str] = typer.Option(None, "--reply-contains", help="With --scenario: its check, the final reply holds this text."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported, and record nothing."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Import a conversation an agent had outside tuningfork as a run: each assistant turn a step,
    rebuilt with everything before it as its context, so resample, ab, forking-tokens and refine
    work at its steps. The skill under study is kept beside it. --scenario makes a scenario of it
    too, whose runs continue the conversation."""
    from tuningfork.ops import scenarios as scenario_files
    from tuningfork.ops.imports import execute_import, make_scenario, prepare_import, prepare_scenario, scenario_start
    from tuningfork.paths import resolve_path

    if scenario:
        try:
            scenario_files.check_name(scenario)
        except OperationError as error:
            _refuse(error)
    if check is not None and reply_contains is not None:
        _refuse(OperationError("name one check: --check or --reply-contains", field="check"))
    checked = {"reply_contains": reply_contains} if reply_contains is not None else check
    if (cut is not None or checked is not None) and not scenario:
        _refuse(OperationError("--cut and a check are a scenario's: name it with --scenario", field="scenario"))

    try:
        data = resolve_path(str(file)).path.read_bytes()  # a Windows path, under WSL, read as it is meant
    except OSError as exc:
        _refuse(OperationError(f"cannot read {file}: {exc.strerror or exc}", field="file"))
    store = _store(home)
    try:
        plan = prepare_import(
            store, data, session=session, skill=skill, skill_dir=[str(folder) for folder in skill_dir or []],
            harness=harness, label=label, task_id=task, tools_from=tools_from,
        )
    except OperationError as error:
        _refuse(error)
    shown = plan.to_json()
    console.print(f"[bold]{escape(shown['label'])}[/bold] · {shown['format']} · {shown['harness']}")
    if shown["session_id"]:
        console.print(f"  session    {escape(shown['session_id'])}")
    console.print(f"  model      {escape(shown['model'] or '-')}")
    console.print(
        f"  steps      {shown['steps']}, one an assistant turn, of {shown['messages']} messages; "
        f"{'a' if shown['system_prompt'] else 'no'} system prompt, "
        + (f"{shown['tools']} tools" + (f" from run {shown['tools_from'][:8]}" if shown["tools_from"] else "")
           if shown["tools"] else "no tools")
    )
    for found in shown["skills"]:
        where = f"in context from step {found['step']}" if found["step"] is not None else "not in the conversation"
        how = SKILL_SOURCES.get(found["source"], found["source"]) + "; " if found["source"] else ""
        kept = f"its folder {found['folder']}" if found["folder"] else "extracted from what the model read"
        chosen = "  [bold]← studied unless another is named[/bold]" if found["name"] == shown["skill"] and len(shown["skills"]) > 1 else ""
        console.print(f"  skill      {escape(found['name'])} ({escape(how + where)}); kept: {escape(kept)}{chosen}")
    if not shown["skills"]:
        console.print("  skill      [dim]none shows in the conversation[/dim]")
    for warning in shown["warnings"]:
        console.print(f"  [yellow]note[/yellow]       {escape(warning)}", soft_wrap=True)
    if scenario:
        if not plan.study:
            _refuse(OperationError(
                "a scenario studies one skill, and the conversation shows several: name it with --skill",
                hint=f"the skills in it: {', '.join(plan.names)}", field="skill",
            ))
        try:
            start = scenario_start(plan.chat, plan.study, plan.harness, cut)
        except OperationError as error:
            _refuse(error)
        first = start.prompt.strip().splitlines()[0] if start.prompt.strip() else ""
        in_turn = len(start.context.in_turn)
        console.print(f"  scenario   {escape(scenario)}: starts at message {start.cut} ({_clip(first, False, limit=70)}), "
                      f"continuing {len(start.context.before)} message(s)"
                      + (f", and {in_turn} of its turn taken as done" if in_turn else "")
                      + (escape(f"; each run stops where the model loads a skill other than {plan.study}")
                         if start.next_skill else ""))
    if dry_run:
        console.print("\n[dim]nothing recorded (--dry-run)[/dim]")
        return
    # A scenario's name is checked before anything is recorded: a clash leaves no run behind.
    if scenario and (scenario_files.folder(store.home, scenario) / scenario_files.SPEC).exists():
        _refuse(OperationError(f"a scenario called {scenario!r} exists already", field="scenario"))
    try:
        result = execute_import(store, plan)
    except OperationError as error:
        _refuse(error)
    run = result.run_id[:8]
    console.print(f"\nimported as run [bold]{run}[/bold]: {result.steps} step(s)")
    named = " --skill <name>" if not result.skill and len(result.skills) > 1 else ""
    console.print(f"[dim]`tuningfork timeline {run}` reads it; `tuningfork resample {run} <step> -n 20` and "
                  f"`tuningfork refine start {run} <step>{escape(named)} …` work at its steps.[/dim]")
    if not scenario:
        return
    try:
        planned = prepare_scenario(store, result.run_id, scenario, cut=cut, check=checked, skill=plan.study)
        made = make_scenario(store, planned)
    except OperationError as error:
        _refuse(error)
    for warning in [*planned.warnings, *made.checked.warnings]:
        console.print(f"  [yellow]note[/yellow]       {escape(warning)}", soft_wrap=True)
    spec_file = scenario_files.spec_path(store.home, scenario)
    console.print(f"scenario [bold]{escape(scenario)}[/bold] made: {escape(str(spec_file))}")
    console.print(f"[dim]`tuningfork batch {spec_file} -n 10` runs it, each run continuing the conversation.[/dim]")


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="First run."),
    run_b: str = typer.Argument(..., help="Second run."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Compare two runs step by step."""
    store = _store(home)
    first, second = _resolve_run(store, run_a), _resolve_run(store, run_b)
    report = compare_runs(store, [first.id, second.id])

    table = Table("k", "", first.id[:8], second.id[:8], box=None)
    for comparison in report.steps:
        by_run = {
            run_id: group.outcome
            for group in comparison.groups
            for run_id in group.run_ids
        }
        marker = "[red]≠[/red]" if comparison.diverged else " "
        table.add_row(
            str(comparison.index),
            marker,
            _clip(by_run[first.id].label, full=False, limit=44),
            _clip(by_run[second.id].label, full=False, limit=44),
        )
    console.print(table)
    if report.diverged:
        console.print(f"\nfirst divergence at step [bold]{report.divergence_index}[/bold]")
    else:
        console.print("\nno divergence: the runs behaved identically")


@app.command()
def tally(
    task: Optional[str] = typer.Option(None, help="A task id: what its failed runs' checks said, by condition."),
    screen: Optional[str] = typer.Option(
        None, help="A screen, or an A/B, run to the end (any arm's experiment id): what each arm's failed forks' "
                   "checks said.",
    ),
    show: int = typer.Option(10, "--show", min=1, help="Findings shown per group, the most common first."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Count what the checks said, across a task's failed runs or a screen's failed forks.

    Each line a check prints is one finding, and lines that differ only in their numbers are
    one. The most common finding comes first, with how many failed runs said it: the failure
    to fix first, rather than the first one read.
    """
    from tuningfork.analysis.stats import rate
    from tuningfork.ops.tally import made_text, tally_screen, tally_task

    if (task is None) == (screen is None):
        console.print("[red]give --task or --screen[/red]")
        raise typer.Exit(code=1)
    store = _store(home)
    try:
        found = tally_task(store, task) if task else tally_screen(store, screen)
    except OperationError as error:
        _refuse(error)
    if found.scope == "screen":
        console.print(f"screen {found.screen[:8]} from run {(found.run_id or '-')[:8]}"
                      f"{f' of {escape(found.task_id)}' if found.task_id else ''}: each arm's forks run to the end")
    else:
        console.print(f"{escape(found.task_id)}: its runs, by condition")
    for group in found.groups:
        console.print()
        console.print(f"[bold]{escape(group.label)}[/bold]  failed {group.failed} of {group.judged} judged")
        if not group.failed:
            continue
        table = Table("failed runs said it", "finding", box=None)
        for finding in group.findings[:show]:
            table.add_row(rate(len(finding.run_ids), finding.of).format(), escape(finding.key))
            if finding.made:
                # Where the check was run along the runs: the call that made it, in each.
                table.add_row("", f"[dim]{escape(made_text(finding.made))}[/dim]")
        console.print(table)
        if len(group.findings) > show:
            console.print(f"  … and {len(group.findings) - show} more finding(s): --show")
        if group.silent:
            console.print(f"  [dim]{group.silent} failed run(s) whose check printed nothing[/dim]")
        if group.cut:
            console.print(f"  [yellow]{group.cut} failed run(s) whose check printed more than the last "
                          f"{CHECK_KEPT:,} characters a verdict keeps[/yellow]: what each printed before them is not counted")
    if found.scope == "task":
        from tuningfork.ops.tally import finding_words, shared_mistakes

        for finding in shared_mistakes(found.to_json())[:2]:
            words = finding_words(finding["finding"])
            console.print(
                f"\n[yellow]every run made «{escape(finding['finding'])}»[/yellow] — the passing ones too, which "
                "repaired it later — so no passing run shows the fix. Refine at the call that made it, each run's "
                f"own: [bold]tuningfork refine start <run> 'made:{escape(words)}' --target-check[/bold]",
                soft_wrap=True,
            )
        if (any(group.failed for group in found.groups)
                and not any(finding.made for group in found.groups for finding in group.findings)):
            console.print(f"\n[dim]For a check that reads the run's files, `tuningfork check-along --task "
                          f"{escape(found.task_id or '<id>')}` runs it along each run, and the tally then says which "
                          "call made each finding.[/dim]", soft_wrap=True)
    console.print()
    who = "failed forks" if found.scope == "screen" else "failed runs"
    console.print(f"[dim]Each count is the {who} that printed the line, once each, however often it printed it. "
                  "Lines that differ only in their numbers or colour codes count as one, shown with # and <color>. "
                  "Intervals are Wilson 95%.[/dim]")


@app.command("check-along")
def check_along_command(
    task: Optional[str] = typer.Option(None, help="A task id: run its check along each of its recorded runs."),
    run: Optional[str] = typer.Option(None, help="Or one run."),
    limit: int = typer.Option(50, "--limit", min=1, help="With --task: at most this many of its runs, the newest first."),
    again: bool = typer.Option(False, "--again", help="Run it anew along runs it was run along before."),
    timeout: float = typer.Option(60.0, "--timeout", min=1.0, help="Seconds the check may take, each time."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Run a task's check along its runs: on each workspace as it started, after each call that
    changed it, and at its end.

    A check judges a run's end; run along it, it says when each thing it complains about was
    made — a finding there from the start is work not done yet, one a call made is a mistake,
    made at that call. `tuningfork tally` then says, for each finding, which call made it in how
    many runs, the passing ones included: a mistake every run made, and some repaired later. Kept
    with each run; its verdict is left as it was. Only a check that reads the files: a reply
    check has nothing to read before the reply.
    """
    from tuningfork.ops.checks import execute_check_along_runs, prepare_check_along_runs

    if (task is None) == (run is None):
        console.print("[red]give --task or --run[/red]")
        raise typer.Exit(code=1)
    store = _store(home)
    try:
        plan = prepare_check_along_runs(store, task=task, run=run, limit=limit, again=again, timeout=timeout)
    except OperationError as error:
        _refuse(error)
    for note in plan.skipped[:10]:
        console.print(f"  [dim]passed over: {escape(note)}[/dim]")
    if not plan.plans:
        console.print("nothing to run the check along")
        return
    console.print(f"running the check along {len(plan.plans)} run(s), "
                  f"{sum(len(each.states) for each in plan.plans)} state(s) of their workspaces")
    done = execute_check_along_runs(store, plan)
    for entry in done.runs:
        verdict = {True: "passed", False: "failed"}.get(entry["passed"], "no verdict")
        made = entry["made"]
        console.print(f"[bold]{entry['run'][:8]}[/bold] {verdict}: {entry['states']} state(s)"
                      + ("" if made else "; no call made a finding"))
        for item in made[:6]:
            gone = "" if item["at_end"] else "  [dim](gone by the end)[/dim]"
            console.print(f"  step {item['made_by']}'s call made: {escape(item['line'])}{gone}", soft_wrap=True)
    console.print(f"\n[dim]`tuningfork tally --task {escape(done.task_id or '<id>')}` says, for each finding, which "
                  "call made it across the runs.[/dim]", soft_wrap=True)


@app.command()
def where(
    task: Optional[str] = typer.Option(None, help="A task id: the question is asked of each of its runs."),
    screen: Optional[str] = typer.Option(
        None, "--screen", help="Or a screen run to the end: the question is asked of its forks, arm by arm — "
                               "where none of the task's own runs passed, a proposal's passing forks are.",
    ),
    match: Optional[str] = typer.Option(
        None, "--match", help="Of the model's answers: a regular expression over what each step did — a tool "
                              "call's name and arguments, a script's code among them, or a reply's words.",
    ),
    rules: Optional[str] = typer.Option(
        None, "--rules", help='Or the answer described, as JSON: [{"if": "calls", "value": "write_file"}] — '
                              "calls, not_calls, contains, not_contains, all of which must hold.",
    ),
    file: Optional[str] = typer.Option(
        None, "--file", help="Of what the runs wrote: a file in the workspace, by its path there or a glob "
                             "(clean.csv, *.json).",
    ),
    contains: Optional[str] = typer.Option(
        None, "--contains", help="With --file: a regular expression its text must match, ^ and $ at each line's "
                                 "ends; without one, the file must exist.",
    ),
    at_least: int = typer.Option(1, "--at-least", min=1, help="With --contains: how many matches make it hold."),
    limit: int = typer.Option(200, "--limit", min=1, help="At most this many of the task's runs, the newest first."),
    show: int = typer.Option(20, "--show", min=0, help="Runs listed per condition."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Ask one question of every run of a task, and line the answers up with the verdicts.

    Of the model's answers (--match, --rules): the steps whose answer satisfies it, the first of
    them. Of what the runs wrote (--file, --contains): whether it held at the start, which call
    made it hold, which undid it, and whether it held at the end. Counted by verdict — in how many
    passing and how many failing runs it held, by Fisher's exact test — with the step reference
    that names that call in each run, a refinement's step. Where a check prints only its verdict,
    or no finding of it names a call, this finds the call that made the difference.
    """
    from tuningfork.ops.where import reference_text, seen_text, split_text, where_it_holds

    parsed_rules = None
    if rules is not None:
        try:
            parsed_rules = json.loads(rules)
        except ValueError:
            console.print("[red]--rules is not JSON[/red]")
            raise typer.Exit(code=1) from None
    if (task is None) == (screen is None):
        console.print("[red]give --task, or --screen for a screen run to the end's forks[/red]")
        raise typer.Exit(code=1)
    store = _store(home)
    try:
        found = where_it_holds(store, task, target_match=match, target_rules=parsed_rules, path=file,
                               contains=contains, at_least=at_least, limit=limit, screen=screen)
    except OperationError as error:
        _refuse(error)
    report = found.to_json()
    kind = found.probe.kind
    asked = sum(len(group.runs) for group in found.groups)
    console.print(f"{escape(found.task_id)}: {escape(found.probe.description)}, asked of {asked} "
                  + (f"fork(s) of screen {found.screen[:8]}" if found.screen else "run(s)"))
    for note in found.skipped[:5]:
        console.print(f"  [dim]passed over: {escape(note)}[/dim]")
    for group in report["groups"]:
        console.print()
        console.print(f"[bold]{escape(group['label'])}[/bold]  {group['passed']} passed, {group['failed']} failed"
                      + (f", {group['unjudged']} without a verdict that counts" if group["unjudged"] else ""))
        for line in (split_text(group, kind), seen_text(group, kind), reference_text(group)):
            if line:
                console.print(f"  {escape(line)}", soft_wrap=True)
        if not show:
            continue
        if kind == "file":
            table = Table("run", "verdict", "start", "made by", "undone by", "end", "file", box=None)
        else:
            table = Table("run", "verdict", "first", "steps", box=None)
        for entry in group["runs"][:show]:
            at = "-" if entry["step"] is None else f"{entry['step']} {' '.join(entry['tools'])}".strip()
            verdict = entry["verdict"] or "no verdict"
            if kind == "file":
                yes = {True: "yes", False: "no", None: "-"}
                held_in = (f"{entry['file']}" + (f" ({entry['count']})" if entry.get("count") is not None else "")
                           if entry.get("file") else "")
                undone = "-" if entry["undone_by"] is None else str(entry["undone_by"])
                table.add_row(entry["run"][:8], verdict, yes[entry["at_start"]], escape(at), undone,
                              yes[entry["at_end"]], escape(held_in))
            else:
                steps = ", ".join(str(step) for step in entry["steps"]) or "-"
                table.add_row(entry["run"][:8], verdict, escape(at), steps)
        console.print(table)
        if len(group["runs"]) > show:
            console.print(f"  … and {len(group['runs']) - show} more run(s): --show")
        reference = (group.get("reference") or {}).get("step")
        if reference:
            console.print(f"  [dim]That call in a run: tuningfork step <run> '{escape(reference)}'; a refinement "
                          f"there: tuningfork refine start <run> '{escape(reference)}' …[/dim]", soft_wrap=True)
    console.print()
    console.print("[dim]A lead to read, never a verdict: a split is said to go one way under p 0.2. Intervals are "
                  "Wilson 95%.[/dim]")


@app.command()
def divergence(
    runs_or_task: list[str] = typer.Argument(
        None, help="Run ids, or omit and pass --task to compare a whole batch."
    ),
    task: Optional[str] = typer.Option(None, help="Compare every run of this task id."),
    limit: int = typer.Option(20, help="Maximum runs to compare when using --task."),
    normalizer: list[str] = typer.Option(
        [],
        "--normalizer",
        help="Compare under these normalizers only, e.g. tool_name to ignore arguments.",
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Find the first step where runs of the same task stop agreeing."""
    from tuningfork.analysis.normalize import validate_normalizers

    try:
        lens = validate_normalizers(normalizer)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    from tuningfork.ops.inspect import divergence as diverged
    from tuningfork.ops.inspect import divergence_runs

    store = _store(home)
    try:
        run_ids, heading = divergence_runs(
            store, refs=[] if task else (runs_or_task or []), task=task, limit=limit
        )
        report, results = diverged(store, run_ids, lens)
    except OperationError as error:
        _refuse(error)

    if lens:
        heading = f"{heading}, compared under {', '.join(lens)}"
    _print_divergence(report, heading, store, results)


def _print_divergence(
    report: DivergenceReport, heading: str, store: Store, results: dict | None = None
) -> None:
    from tuningfork.ops.inspect import behaviour_pass_rates
    console.print(f"\n[bold]{report.compared} runs[/bold] of {heading}")
    if report.mixed_skills:
        groups = [len(runs) for signature, runs in report.skill_versions.items() if signature]
        console.print(
            f"  [bold yellow]warning:[/bold yellow] these runs were recorded under "
            f"{len(groups)} different versions of the skill ({' + '.join(map(str, groups))} "
            "runs). Differences between them may be the wording change, not the model."
        )
    if not report.diverged:
        console.print(
            f"  every run behaved identically for all {len(report.steps)} step(s)"
        )
        return

    console.print(
        f"  agreed for {report.agreed_steps} step(s), "
        f"diverged at step [bold red]{report.divergence_index}[/bold red]"
    )

    comparison = report.at_divergence()
    assert comparison is not None
    # Which behaviour goes on to pass the task's check, when the runs have verdicts.
    if results is None:
        results = {run_id: getattr(store.get_run(run_id), "result", {}) for run_id in report.run_ids}
    passed = behaviour_pass_rates(comparison, results)
    columns = ["", "count", "rate", "95% CI", "behaviour"]
    if any(passed):
        columns.append("then passed")
    console.print(f"\n[bold]step {comparison.index}[/bold]")
    table = Table(*columns, box=None, pad_edge=False)
    for position, (group, group_passed) in enumerate(zip(comparison.groups, passed)):
        letter = CLUSTER_LETTERS[position % len(CLUSTER_LETTERS)]
        row = [
            letter,
            f"{group.count}/{group.rate.total}",
            f"{group.rate.point:.0%}",
            f"[{group.rate.low:.0%}-{group.rate.high:.0%}]",
            _clip(group.outcome.label, full=False, limit=70),
        ]
        if any(passed):
            row.append(escape(group_passed.format()) if group_passed else "-")
        table.add_row(*row)
    console.print(table)
    # The tool each run used, at whatever step: a split no step index shows.
    from tuningfork.ops.inspect import tool_split

    by_tool = tool_split(store, report.run_ids, results).get("best")
    if by_tool is not None:
        called, never = by_tool["called"], by_tool["never"]
        goes = "goes with passing" if by_tool.get("with_passing") else "goes with failing"
        console.print(
            f"\n[bold]by tool[/bold]  runs that called {escape(by_tool['tool'])} passed "
            f"{called['passed']}/{called['total']}, runs that never did {never['passed']}/{never['total']} "
            f"(Fisher p = {p_short(by_tool['p'])}): the tool {goes}, at whatever step each run called it"
        )
    console.print(
        "\n[dim]Intervals are Wilson 95%. With this many runs they are wide; "
        f"separating 60% from 80% takes about {samples_needed(0.6, 0.8)} runs per arm."
        "[/dim]"
    )


@app.command(name="label-next")
def label_next(
    label: str = typer.Argument(..., help="Label for the next run that appears."),
    task: Optional[str] = typer.Option(None, help="Task id to attach as well."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Label the next run recorded, so you can find it later."""
    store = _store(home)
    store.label_next(label, task)
    console.print(f"next run will be labelled [bold]{label}[/bold]")


HARNESS_OPTION_HELP = (
    "An option the harness takes, as key=value (a JSON value where it is one: true, 3, {...}), "
    "recorded with each run. Hermes: self_improvement, await_review, review_every_turn, "
    "mcp_servers. Repeatable."
)


@app.command()
def batch(
    spec_path: Path = typer.Argument(..., help="Task spec JSON (evals.json convention)."),
    n: int = typer.Option(5, "-n", help="How many runs."),
    eval_id: Optional[str] = typer.Option(None, "--eval", help="Which eval to run."),
    base_url: Optional[str] = typer.Option(
        None, help="Proxy URL the harness should talk to. Default: TUNINGFORK_PROXY + /v1."
    ),
    model: Optional[str] = typer.Option(
        None, help="Model id the harness should request. Default: the one model the server lists."
    ),
    harness: str = typer.Option("hermes", help="Harness adapter to drive."),
    skill_file: Optional[Path] = typer.Option(
        None,
        "--skill",
        help="Use this file as the skill's SKILL.md in every run — to confirm an edit "
        "that won an A/B with full runs.",
    ),
    workdir: Optional[Path] = typer.Option(None, help="Where run workspaces are created."),
    concurrency: int = typer.Option(
        1, help="Runs at once. The proxy's --max-inflight still caps calls reaching the server."
    ),
    include_held_out: bool = typer.Option(
        False, "--include-held-out",
        help="Run a held-out eval named with --eval anyway. Its runs are recorded here, where "
        "agents reading the recordings can see them.",
    ),
    harness_option: list[str] = typer.Option([], "--harness-option", "-o", help=HARNESS_OPTION_HELP),
    skill_dir: Optional[Path] = typer.Option(
        None, "--skill-dir", help="A skill folder to give every run instead of the spec's: a chain's final skill, evaluated."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Run a task N times through a headless harness, each in a fresh workspace and home."""
    from tuningfork.ops.batch import execute_batch, prepare_batch
    from tuningfork.specs import describe_check

    store = _store(home)
    try:
        plan = prepare_batch(
            store, spec_path, eval_id=eval_id, n=n, model=model, harness=harness,
            skill_file=skill_file, workdir=workdir, concurrency=concurrency, base_url=base_url,
            include_held_out=include_held_out, harness_options=_harness_options(harness_option) or None,
            skill_dir=skill_dir,
        )
    except OperationError as error:
        _refuse(error)
    item = plan.item

    console.print(f"[bold]{plan.task_id}[/bold] — {n} run{'' if n == 1 else 's'}")
    console.print(f"  prompt: {_clip(item.prompt, full=False, limit=100)}")
    described = describe_check(item.tuningfork.check)
    check = escape(described) if described else None
    console.print(f"  check:  {check or '[yellow]none defined[/yellow]'}")
    if plan.skill_dir and plan.skill_dir.exists():
        replaced = ""
        if plan.skill_file:
            replaced = f"  [bold]SKILL.md replaced by {escape(str(plan.skill_file))}[/bold]"
        console.print(f"  skill:  {escape(_shown_path(plan.skill_dir))}{replaced}")
    if plan.harness_options:
        console.print(f"  harness options: {escape(json.dumps(plan.to_json()['harness_options']))}", soft_wrap=True)
    if plan.sandbox is not None:
        console.print(f"  sandbox: every run works in {_sandbox_seen(plan.sandbox)}")
    for note in plan.notes:
        _print_note(note)
    console.print()

    def report(event) -> None:
        if event.kind != "run":
            return
        outcome = event.subject
        if not outcome.checked:
            verdict = "[dim]no check[/dim]"
        else:
            verdict = "[green]pass[/green]" if outcome.passed else "[red]fail[/red]"
        why = outcome.exclusion
        flags = f" [yellow]{escape(why)} — not counted[/yellow]" if why else ""
        console.print(f"  run {outcome.index}  {verdict}  {outcome.duration_s:.1f}s{flags}")

    try:
        result = execute_batch(store, plan, progress=report)
    except OperationError as error:
        _refuse(error)

    if not result.checked:
        console.print("\n[bold]the task defines no check[/bold], so runs have no verdict")
    elif result.rate is not None:
        console.print(f"\n[bold]{result.rate.format()}[/bold] passed")
    else:
        console.print("\n[bold]no run can be counted[/bold], so there is no pass rate")
    if result.invalidated:
        console.print(f"[yellow]{len(result.invalidated)} run(s) not counted[/yellow]")

    if result.divergence is not None:
        lens = item.tuningfork.normalizers
        # Named only when it is not what naming none would give.
        named = lens and set(lens) != set(DEFAULT_NORMALIZERS)
        heading = result.task_id
        if named:
            heading += f", compared under {', '.join(lens)}"
        _print_divergence(result.divergence, heading, store)


def _harness_options(pairs: list[str]) -> dict[str, Any]:
    """`key=value` pairs as a harness's options: a value is read as JSON where it is JSON
    (true, 3, {"a": 1}), else as text. The adapter refuses what it does not take."""
    found: dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            _refuse(OperationError(f"a harness option is key=value, not {pair!r}", field="harness_options"))
        try:
            found[key.strip()] = json.loads(value)
        except ValueError:
            found[key.strip()] = value
    return found


def _print_note(note) -> None:
    if note.level == "warning":
        console.print(f"  [bold yellow]warning:[/bold yellow] {escape(note.message)}")
    else:
        console.print(f"  [yellow]note:[/yellow] {escape(note.message)}")


# --- a chain: the skill carried from each run to the next ----------------------------------

chain_app = typer.Typer(
    no_args_is_help=True,
    help="Run a task in a chain: each run starts from the skill the run before it left.",
)
app.add_typer(chain_app, name="chain")


def _link_line(link: dict) -> str:
    """One link, as the chain prints it: its verdict, and what became of the skill."""
    passed = link.get("passed")
    if passed is None:
        verdict = "[dim]no check[/dim]"
    else:
        verdict = "[green]pass[/green]" if passed else "[red]fail[/red]"
    why = link.get("exclusion")
    flags = f"  [yellow]{escape(why)} — not counted[/yellow]" if why else ""
    before, after = link.get("skill_before") or "?", link.get("skill_after") or "?"
    skill = f"skill {before}" if before == after else f"skill {before} → {after}"
    if link.get("changed"):
        by = link.get("callers") or []
        who = f" by the {' and the '.join(by)}" if by else ""
        skill += f"  [bold]{escape(', '.join(link['changed']))} changed{who}[/bold]"
    review = link.get("review")
    if review and not review.get("ended"):
        skill += "  [yellow]its review was cut off[/yellow]"
    if link.get("told"):
        skill += "  [dim]told[/dim]"
    if link.get("cap_seconds"):
        skill += f"  [yellow]cut to {link['cap_seconds']:.0f} s by the budget[/yellow]"
    return f"  link {link['index']}  {verdict}  {link.get('duration_s', 0):.1f}s  {skill}{flags}"


def _print_chain(found: dict, *, diff: bool = False) -> None:
    console.print(
        f"[bold]chain {found['id'][:8]}[/bold] — {found['status']}  "
        f"({escape(found.get('task_id') or '?')} · {found.get('harness')} · {escape(str(found.get('model')))})"
    )
    links = found.get("links") or []
    elapsed = found.get("elapsed") or 0
    console.print(
        f"  started {_when(found['started_at'])}, {len(links)} of {found.get('n')} link(s) in "
        f"{elapsed / 60:.1f} min" if found.get("started_at") else f"  {len(links)} link(s)"
    )
    console.print(f"  skill   {escape(_shown_path(Path(found['skill_start'])))}", soft_wrap=True)
    if found.get("harness_options"):
        console.print(f"  options {escape(json.dumps(found['harness_options']))}", soft_wrap=True)
    if found.get("tell"):
        console.print("  told    after each run, the check's verdict and output as the user's next turn")
    for link in links:
        console.print(_link_line(link))
        if diff and link.get("diff"):
            _print_hunks(link["diff"])
    _print_chain_end(found)


def _print_hunks(diff: dict) -> None:
    for hunk in diff.get("hunks") or []:
        console.print(f"    [dim]@@ line {hunk.get('old_start')} @@[/dim]")
        for line in hunk.get("lines") or []:
            op = line.get("op", " ")
            style = {"+": "green", "-": "red"}.get(op, "dim")
            console.print(f"    [{style}]{op} {escape(line.get('text', ''))}[/{style}]", soft_wrap=True)


def _print_chain_end(found: dict) -> None:
    versions = found.get("versions") or []
    if versions:
        parts = []
        for version in versions:
            if version.get("from") == "start":
                parts.append(f"{version.get('skill') or '?'} (given)")
            else:
                by = version.get("by") or []
                who = f", by the {' and the '.join(by)}" if by else ""
                parts.append(f"{version.get('skill') or '?'} (link {version.get('link')}{who})")
        console.print(f"\n[bold]{len(versions)} version(s) of the skill:[/bold] {escape(' → '.join(parts))}", soft_wrap=True)
    rate = found.get("rate")
    if rate:
        console.print(
            f"counted: {rate['successes']}/{rate['total']} passed [{rate['low']:.0%}-{rate['high']:.0%}]"
            + (f"; {found['not_counted']} run(s) not counted" if found.get("not_counted") else "")
        )
    elif found.get("links"):
        console.print("no run can be counted" + (" (the skill changed in every one)" if found.get("not_counted") else ""))
    spent = found.get("spent_seconds")
    if spent is not None:
        budget = found.get("budget_seconds")
        line = f"model calls: {spent:.0f} s at the proxy"
        if budget:
            line += f" of {budget:g} s allowed" + (" — stopped at the budget" if found.get("stopped") == "budget" else "")
        console.print(line)
    if found.get("final_skill"):
        console.print(f"final skill: {escape(_shown_path(Path(found['final_skill'])))}", soft_wrap=True)
        if any(link.get("diff") for link in found.get("links") or []):
            console.print(f"[dim]`tuningfork chain show {found['id'][:8]} --diff` shows the lines each link changed[/dim]")


@chain_app.command("run")
def chain_run(
    spec_path: Path = typer.Argument(..., help="Task spec JSON (evals.json convention)."),
    n: int = typer.Option(10, "-n", help="Links: runs in order, each starting from the skill the one before left."),
    eval_id: Optional[str] = typer.Option(None, "--eval", help="Which eval to run."),
    base_url: Optional[str] = typer.Option(
        None, help="Proxy URL the harness should talk to. Default: TUNINGFORK_PROXY + /v1."
    ),
    model: Optional[str] = typer.Option(
        None, help="Model id the harness should request. Default: the one model the server lists."
    ),
    harness: str = typer.Option("hermes", help="Harness adapter to drive."),
    harness_option: list[str] = typer.Option([], "--harness-option", "-o", help=HARNESS_OPTION_HELP),
    skill_dir: Optional[Path] = typer.Option(
        None, "--skill-dir", help="Start from this skill folder instead of the spec's: a chain continued."
    ),
    tell: bool = typer.Option(
        False, "--tell",
        help="After each run, tell the harness the check's verdict and output as the user's next "
        "turn of the same session, before its review: what a person who looked would say.",
    ),
    budget_seconds: Optional[float] = typer.Option(
        None, "--budget-seconds",
        help="Stop before a link once the links' model calls have taken this many seconds at the "
        "proxy: the same budget as another loop's.",
    ),
    workdir: Optional[Path] = typer.Option(None, help="Where run workspaces are created."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Run a task N times in order, the skill carried forward: what the harness's own loop makes of it."""
    from tuningfork.ops.chain import execute_chain, prepare_chain
    from tuningfork.specs import describe_check

    store = _store(home)
    try:
        plan = prepare_chain(
            store, spec_path, eval_id=eval_id, n=n, model=model, harness=harness, workdir=workdir,
            base_url=base_url, harness_options=_harness_options(harness_option) or None,
            skill_dir=skill_dir, tell=tell, budget_seconds=budget_seconds,
        )
    except OperationError as error:
        _refuse(error)
    item = plan.batch.item

    console.print(f"[bold]{plan.task_id}[/bold] — a chain of {n} run{'' if n == 1 else 's'} ({plan.id})")
    console.print(f"  prompt: {_clip(item.prompt, full=False, limit=100)}")
    described = describe_check(item.tuningfork.check)
    console.print(f"  check:  {escape(described) if described else '[yellow]none defined[/yellow]'}")
    console.print(f"  skill:  {escape(_shown_path(plan.skill_dir))}, carried from each run to the next")
    if plan.tell:
        console.print("  told:   after each run, the check's verdict and output as the user's next turn")
    if plan.batch.harness_options:
        console.print(f"  harness options: {escape(json.dumps(plan.batch.to_json()['harness_options']))}", soft_wrap=True)
    if plan.batch.sandbox is not None:
        console.print(f"  sandbox: every run works in {_sandbox_seen(plan.batch.sandbox)}")
    for note in [*plan.batch.notes, *plan.notes]:
        _print_note(note)
    console.print()

    def report(event) -> None:
        if event.kind == "link":
            console.print(_link_line(event.data))
        elif event.kind == "budget":
            console.print(
                f"  [yellow]stopped at the budget: {event.data['spent_seconds']:.0f} s of model calls after "
                f"{event.data['links']} link(s), {event.data['budget_seconds']:g} s allowed[/yellow]"
            )

    try:
        result = execute_chain(store, plan, progress=report)
    except OperationError as error:
        _refuse(error)
    _print_chain_end(result.to_json())


@chain_app.command("list")
def chain_list(
    n: int = typer.Option(20, "-n", help="How many, newest first."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Ids only."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """The chains kept, newest first."""
    from tuningfork.ops.chain import list_chains

    store = _store(home)
    found = list_chains(store, limit=n)
    if quiet:
        for chain in found:
            console.print(chain["id"][:8])
        return
    if not found:
        console.print("no chains yet: `tuningfork chain run <spec.json> -n 10` makes one")
        return
    table = Table(show_header=True, header_style="bold")
    for column in ("chain", "status", "task", "harness", "links", "versions", "counted", "started"):
        table.add_column(column)
    for chain in found:
        rate = chain.get("rate")
        counted = f"{rate['successes']}/{rate['total']}" if rate else "-"
        table.add_row(
            chain["id"][:8], str(chain.get("status")), escape(str(chain.get("task_id"))),
            str(chain.get("harness")), f"{len(chain.get('links') or [])}/{chain.get('n')}",
            str(len(chain.get("versions") or [])), counted,
            _when(chain["started_at"], seconds=False) if chain.get("started_at") else "-",
        )
    console.print(table)


@chain_app.command("show")
def chain_show(
    reference: str = typer.Argument(..., help="The chain, by id or a prefix of it."),
    diff: bool = typer.Option(False, "--diff", help="Show the lines each link changed in SKILL.md."),
    as_json: bool = typer.Option(False, "--json", help="The chain's record as JSON."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """A chain link by link: each run's verdict, who changed the skill, and the versions."""
    from tuningfork.ops.chain import get_chain

    store = _store(home)
    try:
        found = get_chain(store, reference)
    except OperationError as error:
        _refuse(error)
    if as_json:
        console.print_json(json.dumps(found))
        return
    _print_chain(found, diff=diff)


@app.command()
def spend(
    run: list[str] = typer.Option([], "--run", help="A run, by id, a prefix of it, or its label."),
    experiment: list[str] = typer.Option([], "--experiment", help="An experiment, by id."),
    job: list[str] = typer.Option([], "--job", help="A job, by id: what it launched and sampled."),
    chain: list[str] = typer.Option([], "--chain", help="A chain, by id: every link."),
    task: list[str] = typer.Option([], "--task", help="Every run of a task, by its id (skill/eval)."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """What work cost: model calls, tokens in and out, and time, counted from the records."""
    from tuningfork.ops.spend import spend_report

    store = _store(home)
    try:
        rows = spend_report(store, runs=run, experiments=experiment, jobs=job, chains=chain, tasks=task)
    except OperationError as error:
        _refuse(error)
    if not rows:
        console.print("name what to count: --run, --experiment, --job, --chain or --task")
        return
    table = Table(show_header=True, header_style="bold")
    for column, justify in (("what", "left"), ("calls", "right"), ("failed", "right"), ("prompt tokens", "right"),
                            ("completion tokens", "right"), ("wall s", "right"), ("server s", "right")):
        table.add_column(column, justify=justify)
    total = None
    for row in rows:
        found = row.spend
        total = found if total is None else total + found
        table.add_row(escape(row.what), *_spend_cells(found))
    if len(rows) > 1 and total is not None:
        table.add_row("[bold]total[/bold]", *_spend_cells(total))
    console.print(table)
    console.print(
        "[dim]wall: each call's time at the proxy (or the experiment); server: what the server said it "
        "spent, where it did — a dash where no call reported it. Replayed steps cost nothing.[/dim]",
        soft_wrap=True,
    )


def _spend_cells(found) -> list[str]:
    data = found.to_json()
    server = "-" if data["server_s"] is None else f"{data['server_s']:.1f}"
    if data["server_s"] is not None and data["unmeasured"]:
        server += f" ({data['unmeasured']} unmeasured)"
    return [
        str(data["calls"]), str(data["failed"]), f"{data['prompt_tokens']:,}", f"{data['completion_tokens']:,}",
        f"{data['wall_s']:.1f}", server,
    ]


# --- Phase 2: resampling and wording A/B -------------------------------------------------


def _fmt_p(p: float) -> str:
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def _fmt_rate(rate) -> str:
    return (
        f"{rate.successes}/{rate.total} {rate.point:>4.0%} "
        f"[{rate.low:.0%}-{rate.high:.0%}]"
    )


def _refuse(error: OperationError) -> NoReturn:
    """Print an operation's refusal, and exit 1."""
    if error.hint:
        console.print(f"[red]{escape(error.message)}[/red] — {escape(error.hint)}")
    else:
        console.print(f"[red]{escape(error.message)}[/red]")
    if error.detail:
        console.print(escape(error.detail))
    raise typer.Exit(code=1)


def _step_index(store: Store, run: Run, reference: str) -> int:
    """A step by its number, or by the tool its model called; see `ops.experiments`."""
    from tuningfork.ops.experiments import step_index

    try:
        return step_index(store, run, reference)
    except OperationError as error:
        _refuse(error)


def _first_call(store: Store, run: Run, tool: str) -> int | None:
    from tuningfork.ops.experiments import first_call

    return first_call(store, run, tool)


def _print_checkpoint(checkpoint, plan, base_url: str, per_arm: bool = False, arms: int | None = None) -> None:
    """A checkpoint's sampling plan. `arms`, when there are more than an A/B's two."""
    step = checkpoint.step
    samples = f"{plan.n} samples{' per arm' if per_arm else ''}"
    console.print(
        f"[bold]run {checkpoint.run.id[:8]} step {step.idx}[/bold] — {samples} from {base_url}"
    )
    console.print(
        f"  context   {len(checkpoint.messages)} messages, {escape(checkpoint.describe())}"
    )
    sampling = ", ".join(
        f"{key} {value}" for key, value in plan.overrides.items()
    ) or "the recorded conditions (the request's own parameters, else server defaults)"
    console.print(f"  sampling  {escape(sampling)}")
    console.print(
        f"  budget    max_tokens {plan.max_tokens}, concurrency {plan.concurrency}"
    )
    from tuningfork.ops.experiments import estimate

    guess = estimate(checkpoint, plan, arms=arms or (2 if per_arm else 1))
    if guess and guess.fresh:
        size = f" of {guess.prompt_tokens:,} tokens" if guess.prompt_tokens else ""
        cached = " with most of it cached (--prompt-cache reuses it)" if guess.cached else ""
        console.print(
            f"  prompt    each sample processes its whole prompt{size}, then answers: the recorded "
            f"step took {guess.recorded_seconds:.1f}s{cached}"
        )
    elif guess:
        warm = ", usually much less with a warm cache" if guess.cached else ""
        console.print(
            f"  estimate  the recorded step took {guess.recorded_seconds:.1f}s; "
            f"at most ~{guess.worst_minutes:.1f} min{warm}"
        )


def _progress_printer(total: int, show_arm: bool = False):
    """Prints each sample as it lands, from an operation's progress events."""
    seen: dict[str, int] = {}
    told = []

    def on_event(event) -> None:
        if event.kind == "waiting" and not told:
            told.append(True)
            console.print("  [yellow]waiting: tuningfork is paused[/yellow]")
        if event.kind != "sample":
            return
        sample = event.subject
        seen[sample.arm] = seen.get(sample.arm, 0) + 1
        label = _clip(sample.outcome.label, full=False, limit=64)
        if sample.outcome.is_error:
            label = f"[red]{label}[/red]"
        arm = f"{sample.arm:<9}" if show_arm else ""
        console.print(
            f"  {arm}{seen[sample.arm]:>3}/{total}  {label}  [dim]{sample.seconds:.1f}s[/dim]"
        )

    return on_event


def _print_forking(report) -> None:
    """A forking-token probe's result: per position, the baseline and each forced token."""
    console.print(f"\n[bold]target[/bold] (declared before sampling): {escape(report.target)}")
    if report.reproduced is not None:
        console.print(f"  the baseline arms reached it {escape(_fmt_rate(report.reproduced))}")
    undecided, too_fine = report.missed("undecided"), report.missed("too_fine")
    if undecided:
        console.print(
            f"  [yellow]at {_positions(undecided)} the step was still undecided: from there the run's own "
            "token led elsewhere more often than to its behaviour, so a later token decided it "
            "(`tuningfork tokens` shows the tokens after, and --at probes them)[/yellow]"
        )
    if too_fine:
        console.print(
            f"  [yellow]at {_positions(too_fine)} the run's own token led mostly to the same tools with other "
            "arguments: the target is too fine a behaviour to compare on (try --normalizer tool_name, "
            "or --target-match)[/yellow]"
        )
    if not (undecided or too_fine) and report.reproduced and report.reproduced.total and report.reproduced.point < 0.5:
        console.print(
            "  [yellow]the base's own tokens led to its behaviour less than half the time: the target "
            "may be too fine to reproduce, or the step not yet decided at these positions[/yellow]"
        )
    for position in report.positions:
        part = PART_NAMES.get(position.part or "", position.part or "-")
        console.print(f"\n[bold]{escape(f'[{position.position}]')} {part}[/bold]  {escape(position.context)}")
        rows = [("as run", position.baseline, None)] + [("forced", arm, arm) for arm in position.alternatives]
        for label, arm, tested in rows:
            text = f"{arm.arm['text']!r} {arm.arm.get('p') or 0:.0%}"
            line = f"  {label:7} {escape(f'{text:24}')} {escape(_fmt_rate(arm.target))}"
            if tested is not None and tested.p_value is not None:
                line += f"   p={_fmt_p(tested.p_value)} → Holm {_fmt_p(tested.adjusted_p)}"
                if tested in position.flips():
                    way = "away from" if arm.target.point < position.baseline.target.point else "toward"
                    line += f"  [bold]flips {way} the target[/bold]"
            if arm.drawn < arm.planned:
                line += f"  [yellow]{arm.drawn} of {arm.planned} drawn[/yellow]"
            console.print(line)
        if position.weighted is not None:
            console.print(f"  [dim]the chance of the target from here, weighted by each token's chance: {position.weighted:.0%}[/dim]")
    _print_distribution(report.behaviours, "behaviours, across the probe")
    flips = sum(len(position.flips()) for position in report.positions)
    corrected = f"Holm-corrected over the {report.family} planned before sampling; α={report.alpha}"
    if flips:
        console.print(f"\n{flips} of {report.family} forced tokens changed what the model did ({corrected}).")
        return
    # A null is what the probe could see, never that nothing changed.
    console.print(f"\nNo forced token decided what the model did ({corrected}).")
    if report.detectable:
        console.print(f"With {report.per_arm} continuations a token, only a token that changes about "
                      f"{report.detectable} of its {report.per_arm} would show: a smaller shift can be real and "
                      "go unseen (-n draws more).")
    elif report.per_arm:
        console.print(f"With {report.per_arm} continuations a token, no change could reach significance: raise -n.")


def _print_screen(report, arms: dict) -> None:
    """A screen: each proposal against one fresh baseline, Holm-corrected over the family."""
    console.print(f"\n[bold]target[/bold] (declared before sampling): {escape(report.target)}")
    proposals = {
        arm.name: ((arms.get(arm.name) and arms[arm.name].variant) or {}).get("proposal") or "" for arm in report.arms
    }
    # A refinement's arms are named by their proposals: a column of the same ids says nothing.
    named = any(proposal and not proposal.startswith(name) for name, proposal in proposals.items())
    table = Table("", *(["proposal"] if named else []), "target", "p", "Holm", "", box=None, pad_edge=False)
    base = report.baseline
    reused = getattr(report, "reused", 0)
    table.add_row("original", *([""] if named else []), escape(_fmt_rate(base.target)), "", "",
                  f"[dim]{reused} counted again from an earlier screen[/dim]" if reused else "")
    survivors, worse = {arm.name for arm in report.survivors()}, {arm.name for arm in report.worse()}
    for arm in report.arms:
        if getattr(report, "quick", False):
            verdict = "[dim]a quick look: no verdict[/dim]"
        elif not report.complete:
            verdict = "[yellow]no verdict yet[/yellow]"
        elif arm.name in survivors:
            verdict = "[bold green]survives[/bold green]"
        elif arm.name in worse:
            verdict = "[bold red]worse[/bold red]"
        else:
            verdict = "not significant"
        table.add_row(
            arm.name, *([escape(str(proposals[arm.name])[:12])] if named else []),
            escape(_fmt_rate(arm.target)) + ("" if arm.drawn >= arm.planned else f" [yellow]({arm.drawn} of {arm.planned})[/yellow]"),
            _fmt_p(arm.p_value) if arm.p_value is not None else "-",
            _fmt_p(arm.adjusted_p) if arm.adjusted_p is not None else "-",
            verdict,
        )
    console.print(table)
    console.print(
        f"[dim]p: Fisher's exact test of each proposal against the original arm; Holm: corrected "
        f"over the {report.family} planned before sampling (α={report.alpha}).[/dim]"
    )
    if getattr(report, "quick", False):
        console.print("a quick look: rates only — a lead for the next edit; only a full screen can keep one")
    elif not report.complete:
        console.print("[yellow]unfinished: rates only — no verdict until every arm has drawn what was planned[/yellow]")
    elif not survivors:
        console.print("no proposal survived the correction")
    timed_out = [f"{arm.name} {arm.timeouts}" for arm in (base, *report.arms) if getattr(arm, "timeouts", 0)]
    if timed_out:
        console.print(f"[yellow]forks that timed out[/yellow], counted as failures: {escape(', '.join(timed_out))}")
    checkpoints = getattr(report, "checkpoints", None) or []
    if len(checkpoints) > 1:
        from tuningfork.ops.compact import checkpoint_lines

        for line in checkpoint_lines(report.to_json()):
            console.print(escape(line), soft_wrap=True)
    console.print("[dim]A screen at one step is evidence, not proof: confirm a survivor with full runs.[/dim]")


def _positions(positions: list[int]) -> str:
    names = [escape(f"[{position}]") for position in positions]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _print_notes(notes: list[str]) -> None:
    for note in notes:
        console.print(f"  [bold yellow]parity:[/bold yellow] {escape(note)}")


def _print_distribution(dist, heading: str) -> None:
    console.print(f"\n[bold]{escape(heading)}[/bold]")
    # Only where some behaviour was reached without reasoning, where others reasoned: then it
    # stands out. Where every sample reasoned, the column would say nothing.
    reasoning = any(cluster.reasoned is not None and cluster.reasoned < cluster.count for cluster in dist.clusters)
    columns = ["", "count", "rate", "95% CI", *(["reasoned"] if reasoning else []), "behaviour"]
    table = Table(*columns, box=None, pad_edge=False)
    for cluster in dist.clusters:
        label = _clip(cluster.outcome.label, full=False, limit=90)
        if cluster.outcome.kind == "truncated":
            label = f"[yellow]{label}[/yellow]"
        reasoned = []
        if reasoning:
            said = f"{cluster.reasoned}/{cluster.count}"
            reasoned = [f"[yellow]{said}[/yellow]" if cluster.reasoned < cluster.count else said]
        table.add_row(
            cluster.letter,
            f"{cluster.count}/{cluster.rate.total}",
            f"{cluster.rate.point:.0%}",
            escape(f"[{cluster.rate.low:.0%}-{cluster.rate.high:.0%}]"),
            *reasoned,
            label,
        )
    console.print(table)
    if dist.errors:
        console.print(
            f"  [red]{dist.errors} sample(s) failed in transport[/red] and are excluded from "
            "every rate above"
        )


def _print_ab(comparison, caveat: str) -> None:
    console.print()
    table = Table("", "behaviour", "original", "variant", "p adj.", box=None, pad_edge=False)
    for shift in comparison.shifts:
        table.add_row(
            shift.letter,
            _clip(shift.outcome.label, full=False, limit=58),
            escape(_fmt_rate(shift.original)),
            escape(_fmt_rate(shift.variant)),
            _fmt_p(shift.adjusted_p),
        )
    console.print(table)
    console.print(
        f"[dim]p adj.: Fisher exact per behaviour, Bonferroni-corrected for "
        f"{len(comparison.shifts)} comparisons[/dim]"
    )

    errors = comparison.original.errors + comparison.variant.errors
    if errors:
        console.print(f"  [red]{errors} sample(s) failed in transport and are excluded[/red]")

    test = comparison.target
    if test is None:
        significant = comparison.significant_shifts()
        console.print(
            "\n[bold]no target was declared[/bold], so this comparison is exploratory. "
            "Declare one with --target or --target-match to get a single confirmatory test."
        )
        if significant:
            letters = ", ".join(f"{s.letter} (p adj. {_fmt_p(s.adjusted_p)})" for s in significant)
            console.print(f"  behaviours that shifted after correction: {letters}")
        else:
            console.print("  no behaviour shifted significantly after correction")
    else:
        console.print(
            f"\n[bold]target[/bold] (declared before sampling): {escape(test.description)}"
        )
        console.print(f"  original  {escape(_fmt_rate(test.original))}")
        console.print(f"  variant   {escape(_fmt_rate(test.variant))}")
        if test.significant:
            direction = "higher" if test.direction == "up" else "lower"
            console.print(
                f"  Fisher exact p = {_fmt_p(test.p_value)} → [bold green]significant[/bold green]"
                f" at α={test.alpha}, variant {direction}"
            )
            console.print("  [dim]a screening result: confirm a winner with full runs[/dim]")
        else:
            console.print(
                f"  Fisher exact p = {_fmt_p(test.p_value)} → [bold]not significant[/bold] "
                f"at α={test.alpha}"
            )
            needed = test.samples_to_detect()
            if needed:
                console.print(
                    f"  [dim]not the same as no effect: a difference this size would take "
                    f"~{needed} samples per arm to detect reliably[/dim]"
                )
    console.print(f"\n[dim]{caveat}[/dim]")


@app.command()
def resample(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    n: int = typer.Option(20, "-n", help="Samples to draw."),
    base_url: Optional[str] = typer.Option(
        None,
        help="Server to sample from. Defaults to TUNINGFORK_UPSTREAM, else "
        "http://127.0.0.1:8081. Point it at the proxy to share the proxy's gate.",
    ),
    concurrency: int = typer.Option(1, help="Concurrent requests. Keep at 1 on a shared server."),
    max_tokens: int = typer.Option(4096, help="Cap per sample; hitting it counts as truncated."),
    temperature: Optional[float] = typer.Option(None, help="Override (recorded as a change)."),
    top_p: Optional[float] = typer.Option(None, help="Override (recorded as a change)."),
    top_k: Optional[int] = typer.Option(None, help="Override (recorded as a change)."),
    seed_base: Optional[int] = typer.Option(None, help=SEED_HELP),
    prompt_cache: bool = typer.Option(False, "--prompt-cache", help=CACHE_HELP),
    normalizer: list[str] = typer.Option([], "--normalizer", help="Restrict normalizers."),
    logprobs: Optional[int] = typer.Option(
        None, "--logprobs", min=1, max=20,
        help="Keep each sample's tokens, with this many alternatives each: `tuningfork tokens "
        "--sample` shows one's heatmap. Chat-completions steps only.",
    ),
    to_end: bool = typer.Option(
        False, "--to-end",
        help="Run each sample to the end through the harness — the run replayed up to the "
        "step — and count the task's check passing.",
    ),
    proxy: Optional[str] = typer.Option(
        None, help="With --to-end: the proxy the harness talks to. Default: TUNINGFORK_PROXY."
    ),
    spec: Optional[Path] = typer.Option(
        None, help="With --to-end, for a run recorded before tuningfork kept how it was launched."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Re-run one step N times with its history held fixed, and cluster what happens.

    With --to-end, each sample is a fork: the harness is relaunched, answered from the
    recording up to the step, and left to run to the end, where the task's check decides.
    """
    from tuningfork.ops.experiments import execute_resample, prepare_resample

    store = _store(home)
    if to_end:
        _resample_to_end(store, run, reference, n=n, concurrency=concurrency, proxy=proxy, spec=spec)
        return
    try:
        prepared = prepare_resample(
            store, run, reference, base_url=base_url, n=n, concurrency=concurrency,
            max_tokens=max_tokens, seed_base=seed_base, prompt_cache=prompt_cache, normalizers=normalizer,
            overrides={"temperature": temperature, "top_p": top_p, "top_k": top_k},
            logprobs=logprobs,
        )
    except OperationError as error:
        _refuse(error)
    checkpoint, plan = prepared.checkpoint, prepared.plan
    _print_checkpoint(checkpoint, plan, prepared.base_url)
    console.print()

    result = asyncio.run(
        execute_resample(store, prepared, progress=_progress_printer(plan.n))
    )
    experiment, elapsed, index = result.experiment, result.elapsed, checkpoint.step.idx

    _print_notes(experiment.notes)
    _print_distribution(
        result.distribution,
        f"step {index}, {checkpoint.describe()} — {plan.n} samples",
    )
    console.print(
        f"\n[dim]experiment {experiment.id[:8]} · {elapsed / 60:.1f} min · "
        f"{elapsed / plan.n:.1f}s per sample · use `tuningfork ab {checkpoint.run.id[:8]} "
        f"{index} --target <letter> --variant <file>` to test a wording change[/dim]"
    )


@app.command()
def ab(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    variant: Path = typer.Option(..., "--variant", help="The edited skill file."),
    original: Optional[Path] = typer.Option(
        None,
        "--original",
        help="The skill as it was when the run was recorded. Defaults to the copy a batch "
        "run was given, else SKILL.md beside the variant.",
    ),
    skill: Optional[str] = typer.Option(
        None, "--skill", help="Which skill the variant replaces, by name, where the run kept several: an imported chat.",
    ),
    n: int = typer.Option(20, "-n", help="Samples per arm."),
    target: Optional[str] = typer.Option(
        None, "--target", help="Cluster letter from the latest resample at this checkpoint."
    ),
    target_match: Optional[str] = typer.Option(
        None, "--target-match", help="Regular expression over the behaviour."
    ),
    target_rules: Optional[str] = typer.Option(
        None, "--target-rules",
        help='Conditions that must all hold, as JSON: [{"if": "calls", "value": "Write"}, '
        '{"if": "contains", "value": "row_count"}]. Conditions: calls, not_calls, contains, not_contains.',
    ),
    alpha: float = typer.Option(0.05, help="Significance level."),
    base_url: Optional[str] = typer.Option(None, help="Server to sample from."),
    concurrency: int = typer.Option(1, help="Concurrent requests. Keep at 1 on a shared server."),
    max_tokens: int = typer.Option(4096, help="Cap per sample; hitting it counts as truncated."),
    seed_base: Optional[int] = typer.Option(None, help=SEED_HELP),
    prompt_cache: bool = typer.Option(False, "--prompt-cache", help=CACHE_HELP),
    normalizer: list[str] = typer.Option([], "--normalizer", help="Restrict normalizers."),
    to_end: bool = typer.Option(
        False, "--to-end",
        help="Run each sample to the end through the harness and compare how often the "
        "task's check passes: the confirmation a checkpoint A/B asks for.",
    ),
    proxy: Optional[str] = typer.Option(
        None, help="With --to-end: the proxy the harness talks to. Default: TUNINGFORK_PROXY."
    ),
    spec: Optional[Path] = typer.Option(
        None, help="With --to-end, for a run recorded before tuningfork kept how it was launched."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Swap an edited skill in at a checkpoint and test whether behaviour shifts.

    With --to-end, both arms are forks run to the end — the run replayed up to the step,
    each arm's wording from there — compared on the task's check.
    """
    from tuningfork.ops.experiments import execute_ab, prepare_ab

    store = _store(home)
    if to_end:
        if target or target_match or target_rules:
            console.print(
                "[yellow]--target ignored: run to the end, the target is the task's check[/yellow]"
            )
        _ab_to_end(
            store, run, reference, variant=variant, original=original, n=n, alpha=alpha,
            concurrency=concurrency, proxy=proxy, spec=spec,
        )
        return
    rules = None
    if target_rules:
        try:
            rules = json.loads(target_rules)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--target-rules is not JSON: {escape(str(exc))}[/red]")
            raise typer.Exit(code=1)
    try:
        prepared = prepare_ab(
            store, run, reference, variant=variant, original=original, skill=skill, target=target,
            target_match=target_match, target_rules=rules, alpha=alpha, base_url=base_url, normalizers=normalizer,
            n=n, concurrency=concurrency, max_tokens=max_tokens, seed_base=seed_base, prompt_cache=prompt_cache,
        )
    except OperationError as error:
        _refuse(error)
    for warning in prepared.warnings:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    checkpoint, plan = prepared.checkpoint, prepared.plan
    _print_checkpoint(checkpoint, plan, prepared.base_url, per_arm=True)
    console.print(
        f"  swapping  {escape(str(prepared.original))} → {escape(str(prepared.variant))}"
    )
    for span in prepared.spans:
        console.print(f"            found: {escape(span.describe())}")
    for note in prepared.unapplied:
        console.print(f"  [yellow]note[/yellow]      {escape(note)}")
    if prepared.target:
        console.print(
            f"  target    {escape(prepared.target.description)} "
            "[dim](declared before sampling)[/dim]"
        )
    else:
        console.print("  target    none declared — the comparison will be exploratory")
    console.print()

    try:
        result = asyncio.run(
            execute_ab(store, prepared, progress=_progress_printer(plan.n, show_arm=True))
        )
    except OperationError as error:
        _refuse(error)

    _print_notes(result.run.original.notes)
    _print_ab(result.comparison, result.caveat)
    console.print(
        f"[dim]experiments {result.run.original.id[:8]} (original) and "
        f"{result.run.variant.id[:8]} (variant) · {result.elapsed / 60:.1f} min[/dim]"
    )


# --- Phase 4: workspaces and forks -------------------------------------------------------


@app.command()
def files(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    where: str = typer.Argument(
        ..., metavar="step",
        help="The step whose call found the workspace — its number, a tool called there, or "
        "changed:<path> for the call that first changed that file — or `start` or `end`.",
    ),
    diff: bool = typer.Option(
        False, "--diff", help="Show what changed: before a step, since the step before; at `end`, since the start.",
    ),
    path: Optional[str] = typer.Option(None, "--path", help="With --diff: only this file."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the snapshot's files here."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """The workspace as a step's call found it: every tool effect before it, on disk."""
    from tuningfork.ops.inspect import restore_workspace, workspace_at, workspace_patch

    store = _store(home)
    try:
        view = workspace_at(store, run, where)
    except OperationError as error:
        _refuse(error)
    changed = dict((p, s) for s, p in view.changes)
    console.print(
        f"[bold]run {view.run.id[:8]} {view.where}[/bold]  [dim](snapshot {view.commit[:8]})[/dim]"
    )
    table = Table("", "file", "bytes", box=None)
    for entry in view.files:
        mark = {"A": "[green]+[/green]", "M": "[yellow]~[/yellow]"}.get(changed.get(entry.path, ""), "")
        table.add_row(mark, escape(entry.path), "-" if entry.size is None else str(entry.size))
    console.print(table)
    since = view.since or "the snapshot before"
    gone = [p for s, p in view.changes if s == "D"]
    if gone:
        console.print(f"  [red]deleted since {since}:[/red] {escape(', '.join(gone))}")
    if view.changes:
        console.print(f"  [dim]+ new, ~ changed since {since}[/dim]")
    if diff:
        patch = workspace_patch(store, view, path)
        console.print(escape(patch) if patch else "  [dim]no changes[/dim]", soft_wrap=True)
    if out is not None:
        try:
            restore_workspace(store, view, out)
        except OperationError as error:
            _refuse(error)
        console.print(f"\nwritten to {escape(str(out))}")


def _sandbox_seen(sandbox) -> str:
    """Where a sandboxed launch works, and its home, as the harness sees them."""
    seen = sandbox.to_json() if hasattr(sandbox, "to_json") else dict(sandbox)
    clock = f", its clock started at {seen['clock']}" if seen.get("clock") else ""
    return f"{seen['workspace']}, its home {seen['home']}{clock}"


def _print_fork_plan(plan, n: int = 1, arms: int = 1) -> None:
    checkpoint = plan.checkpoint
    k = checkpoint.step.idx
    forks = f"{n} fork{'s' if n != 1 else ''}{' per arm' if arms > 1 else ''}"
    console.print(
        f"[bold]run {checkpoint.run.id[:8]} from step {k}[/bold] — {forks} through "
        f"{escape(plan.proxy.url)}"
    )
    console.print(f"  context   {escape(checkpoint.describe())}")
    console.print(
        f"  replays   {plan.replayed} recorded call(s), answered from the recording; "
        f"live from step {k}"
    )
    console.print(f"  harness   {escape(plan.harness)}, model {escape(plan.model)}")
    if getattr(plan, "sandbox", None) is not None:
        console.print(f"  sandbox   every fork works in {_sandbox_seen(plan.sandbox)}, as its source did")
    if plan.variant is not None:
        console.print(
            f"  swapping  {escape(str(plan.variant.original_path))} → "
            f"{escape(str(plan.variant.variant_path))}, from step {k} on"
        )
        for span in plan.spans:
            console.print(f"            found: {escape(span.describe())}")
    for warning in plan.warnings:
        console.print(f"  [yellow]note[/yellow]      {escape(warning)}")


def _to_end_plan(store: Store, run: str, reference: str, *, n, concurrency, proxy, spec, **options):
    from tuningfork.ops.forks import prepare_to_end

    base = f"{proxy.rstrip('/')}/v1" if proxy else None
    try:
        return prepare_to_end(
            store, run, reference, n=n, concurrency=concurrency, base_url=base, spec=spec,
            **options,
        )
    except OperationError as error:
        _refuse(error)


def _fork_progress(total: int, show_arm: bool = False):
    """Prints each fork as it finishes; forks finish on worker threads, in any order. A fork
    run to the end that timed out counts as a failure (`analysis.conditions.fork_counted`)."""
    import threading

    lock = threading.Lock()
    seen: dict[str, int] = {}

    def on_event(event) -> None:
        if event.kind != "run":
            return
        arm = event.data.get("arm") or ""
        with lock:
            seen[arm] = seen.get(arm, 0) + 1
            label = f"{arm:<9}" if show_arm else ""
            console.print(f"  {label}{seen[arm]:>3}/{total}{_fork_line(event.subject, timeouts_fail=True)[1:]}")

    return on_event


def _print_end_to_end(summary, heading: str) -> None:
    console.print(f"\n[bold]{escape(heading)}[/bold]")
    if summary.rate is not None:
        console.print(f"  check passed  {escape(summary.rate.format())}")
    else:
        console.print("  [yellow]no fork can be counted[/yellow], so there is no pass rate")
    if summary.timeouts:
        console.print(f"  [yellow]{summary.timeouts} timed out[/yellow], counted as failures")
    for run_id, why in summary.exclusions:
        console.print(
            f"  [yellow]not counted[/yellow]   {(run_id or '--------')[:8]}: {escape(why)}"
        )


def _to_end_caveat(k: int, arms: bool) -> str:
    wording = "each arm's wording" if arms else "the recorded wording"
    return (
        f"Steps before {k} were answered from the recording in every fork — the same history, "
        f"generated under the original wording; from step {k} to the end each fork ran live, "
        f"with {wording}, and the task's check decided. An edit that would also change "
        "earlier steps still needs full runs to confirm."
    )


def _resample_to_end(store: Store, run: str, reference: str, *, n, concurrency, proxy, spec) -> None:
    from tuningfork.ops.forks import execute_to_end_plan

    plan = _to_end_plan(
        store, run, reference, n=n, concurrency=concurrency, proxy=proxy, spec=spec
    )
    _print_fork_plan(plan.original, n)
    console.print()
    try:
        summary = execute_to_end_plan(store, plan, progress=_fork_progress(n))
    except OperationError as error:
        _refuse(error)
    k = plan.checkpoint.step.idx
    _print_end_to_end(summary, f"end to end from step {k}, {plan.checkpoint.describe()} — {n} forks")
    _print_distribution(summary.actions, f"what the model did at step {k}, in the forks that count")
    console.print(f"\n[dim]{_to_end_caveat(k, arms=False)}[/dim]")
    console.print(
        f"[dim]experiment {summary.experiment.id[:8]} · {summary.elapsed / 60:.1f} min · "
        f"{summary.elapsed / n:.1f}s per fork[/dim]"
    )


def _ab_to_end(
    store: Store, run: str, reference: str, *, variant, original, n, alpha, concurrency, proxy, spec
) -> None:
    from tuningfork.ops.forks import execute_to_end_plan

    plan = _to_end_plan(
        store, run, reference, n=n, concurrency=concurrency, proxy=proxy, spec=spec,
        variant=variant, original=original, alpha=alpha,
    )
    _print_fork_plan(plan.variant, n, arms=2)
    console.print("  target    the task's check passes [dim](declared by the task, before any fork ran)[/dim]")
    console.print()
    try:
        result = execute_to_end_plan(store, plan, progress=_fork_progress(n, show_arm=True))
    except OperationError as error:
        _refuse(error)
    _print_ab_to_end(result, plan.checkpoint.step.idx, alpha)
    console.print(
        f"[dim]experiments {result.original.experiment.id[:8]} (original) and "
        f"{result.variant.experiment.id[:8]} (variant) · "
        f"{(result.original.elapsed + result.variant.elapsed) / 60:.1f} min[/dim]"
    )


def _print_ab_to_end(result, k: int, alpha: float) -> None:
    console.print(f"\n[bold]target[/bold]: the task's check passes, end to end from step {k}")
    for name, arm in (("original", result.original), ("variant", result.variant)):
        shown = escape(_fmt_rate(arm.rate)) if arm.rate else "no fork counts"
        timeouts = f"  [yellow]{arm.timeouts} timed out[/yellow], counted as failures" if arm.timeouts else ""
        console.print(f"  {name:<9} {shown}{timeouts}")
    if result.p_value is None:
        console.print("  [yellow]an arm has no fork that counts, so there is no test[/yellow]")
    else:
        verdict = (
            "[bold green]significant[/bold green]" if result.significant else "[bold]not significant[/bold]"
        )
        direction = ""
        if result.significant:
            higher = result.variant.rate.point > result.original.rate.point
            direction = f", variant {'higher' if higher else 'lower'}"
        console.print(
            f"  Fisher exact p = {_fmt_p(result.p_value)} → {verdict} at α={alpha}{direction}"
        )
    for name, arm in (("original", result.original), ("variant", result.variant)):
        for run_id, why in arm.exclusions:
            console.print(
                f"  [yellow]not counted[/yellow] ({name}) {(run_id or '--------')[:8]}: {escape(why)}"
            )
    for name, arm in (("original", result.original), ("variant", result.variant)):
        _print_distribution(arm.actions, f"{name}: what the model did at step {k}")
    console.print(f"\n[dim]{_to_end_caveat(k, arms=True)}[/dim]")


def _fork_line(outcome, *, timeouts_fail: bool = False) -> str:
    """One run as it lands. In a validation, or among forks run to the end, `timeouts_fail`: a
    run that timed out counts as a failure there, as the gate counts it, rather than being left
    out."""
    if not outcome.checked:
        verdict = "[dim]no check[/dim]"
    else:
        verdict = "[green]pass[/green]" if outcome.passed else "[red]fail[/red]"
    run = outcome.run_id[:8] if outcome.run_id else "--------"
    why = outcome.exclusion
    if why == "timed out" and timeouts_fail:
        flags = " [yellow]timed out — counted as a failure[/yellow]"
    else:
        flags = f" [yellow]{escape(why)} — not counted[/yellow]" if why else ""
    live = (outcome.live_outcome or {}).get("label")
    did = f"  {_clip(live, full=False, limit=60)}" if live else ""
    return f"  {run}  {verdict}  {outcome.duration_s:.1f}s{did}{flags}"


@app.command()
def fork(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    variant: Optional[Path] = typer.Option(
        None, "--variant", help="An edited skill file to swap in from the step on."
    ),
    original: Optional[Path] = typer.Option(
        None, "--original", help="The skill as it was recorded. Defaults to the copy the run was given."
    ),
    spec: Optional[Path] = typer.Option(
        None, help="The task spec, for a run recorded before tuningfork kept how it was launched."
    ),
    breakpoints: bool = typer.Option(
        True, "--breakpoints/--no-breakpoints", help="Let armed breakpoints hold the live calls."
    ),
    base_url: Optional[str] = typer.Option(
        None, help="The proxy the harness talks to. Default: TUNINGFORK_PROXY + /v1."
    ),
    workdir: Optional[Path] = typer.Option(None, help="Where fork workspaces are created."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Replay a run up to a step, then let the model take over from there.

    The run's harness is relaunched on the same task in a fresh workspace, and its calls
    are answered from the recording — the harness re-runs every recorded tool call — until
    the call at the step, which goes to the model. Armed breakpoints apply from there: a
    restart from any step of a recorded run.
    """
    from tuningfork.ops.forks import execute_forks, prepare_fork

    store = _store(home)
    try:
        plan = prepare_fork(
            store, run, reference, base_url=base_url, spec=spec, variant=variant,
            original=original, breakpoints=breakpoints, workdir=workdir,
        )
    except OperationError as error:
        _refuse(error)
    _print_fork_plan(plan)
    console.print()
    try:
        (outcome,) = execute_forks(store, plan, 1)
    except (HarnessError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    console.print(_fork_line(outcome))
    if outcome.departed and outcome.departed.get("expected") is not None:
        console.print("\n  the call that left the recording, masked, against the recorded one:")
        console.print(f"    recorded  …{escape(outcome.departed['expected'])!s}")
        console.print(f"    this fork …{escape(outcome.departed['got'])!s}")
    if outcome.run_id:
        replayed = plan.source.step
        answered = (
            "every one ran live" if replayed == 0
            else "its first step was answered from the recording" if replayed == 1
            else f"its first {replayed} were answered from the recording"
        )
        console.print(
            f"\n[dim]`tuningfork show {outcome.run_id[:8]}` for the fork's steps; {answered}.[/dim]"
        )


@app.command()
def experiments(
    run: Optional[str] = typer.Option(None, help="Only experiments on this run."),
    limit: int = typer.Option(20, "--limit", "-n"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print only experiment ids, newest first."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """List experiments, newest first."""
    from tuningfork.ops.inspect import arm_status

    store = _store(home)
    run_id = _resolve_run(store, run).id if run else None
    found = store.experiments(run_id=run_id, limit=limit)
    if quiet:
        for experiment in found:
            typer.echo(experiment.id)
        return
    if not found:
        console.print(f"no experiments in {store.home}")
        raise typer.Exit()
    table = Table("experiment", "kind", "arm", "run", "step", "samples", "created", box=None)
    for experiment in found:
        reach = experiment.settings.get("reach")
        status = arm_status(experiment)
        table.add_row(
            experiment.id[:8],
            f"{experiment.kind} to end" if reach == "end" else experiment.kind,
            experiment.arm,
            # One sampled at a held call has no step until the call is recorded.
            experiment.run_id[:8] if experiment.run_id else f"held {(experiment.hold_id or '')[:8]}",
            str(experiment.step_idx) if experiment.step_idx is not None else "—",
            f"{experiment.sample_count}/{experiment.settings.get('n', '?')}"
            + (f" {status}" if status != "done" else ""),
            _when(experiment.created_at),
        )
    console.print(table)


@app.command()
def experiment(
    reference: str = typer.Argument(..., help="Experiment id or prefix."),
    alpha: float = typer.Option(0.05, help="Significance level."),
    normalizer: list[str] = typer.Option(
        [],
        "--normalizer",
        help="Re-analyse the stored samples under other normalizers, e.g. tool_name.",
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Show a stored experiment's result again, without drawing new samples."""
    from tuningfork.analysis.normalize import validate_normalizers
    from tuningfork.experiment import history_caveat
    from tuningfork.ops.inspect import experiment_report

    try:
        lens = validate_normalizers(normalizer) or None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    store = _store(home)
    try:
        report = experiment_report(store, reference, lens=lens, alpha=alpha)
    except OperationError as error:
        _refuse(error)
    found, checkpoint = report.experiment, report.checkpoint
    where = (
        escape(checkpoint.describe())
        if checkpoint
        else "[yellow]its run is no longer in this store[/yellow]"
    )
    if found.run_id is None:
        console.print(
            f"[bold]{found.kind} at held call {(found.hold_id or '')[:8]}[/bold] — "
            "not recorded yet: it is filed under the step the call becomes"
        )
    else:
        console.print(
            f"[bold]{found.kind} at run {found.run_id[:8]} step {found.step_idx}[/bold] — {where}"
        )
    _print_notes(found.notes)
    for key, change in found.changes.items():
        detail = f"sent {key}={change['sent']!r} (recorded: {change['recorded']!r})"
        console.print(f"  [dim]{escape(detail)}[/dim]")

    if lens:
        console.print(f"  [dim]re-analysed under: {', '.join(lens)}[/dim]")
    if report.status in ("cancelled", "interrupted", "failed"):
        console.print(f"  [yellow]this experiment was {report.status}[/yellow]")
    for arm, drawn, meant in report.running:
        console.print(
            f"  [yellow]still running: the {arm} arm has {drawn} of its {meant} samples so far[/yellow]"
        )
    for arm, drawn, meant in report.partial:
        # Stopped part way — interrupted, or cancelled: the rates are over what was drawn.
        console.print(
            f"  [yellow]stopped early: the {arm} arm has {drawn} of its {meant} samples[/yellow]"
        )

    k = found.step_idx if found.step_idx is not None else "(held)"
    if report.kind == "forking":
        _print_forking(report.forking)
        return
    if report.kind == "screen":
        _print_screen(report.screen, report.arms)
        return
    if report.kind == "to_end":
        summary = report.to_end
        _print_end_to_end(summary, f"end to end from step {k} — {found.sample_count} forks")
        _print_distribution(summary.actions, f"what the model did at step {k}, in the forks that count")
        console.print(f"\n[dim]{_to_end_caveat(k, arms=False)}[/dim]")
        return
    if report.kind == "resample":
        _print_distribution(report.distribution, f"step {k} — {found.sample_count} samples")
        return
    if report.kind in ("interrupted", "ab_running"):
        # Stopped before the second arm, or not at it yet: what was sampled is still worth
        # seeing.
        arm = next(iter(report.arms.values()))
        if report.kind == "interrupted":
            console.print(f"  [yellow]this A/B was interrupted: only its {arm.arm} arm ran[/yellow]")
        _print_wording(report.wording)
        if report.to_end is not None:
            _print_end_to_end(report.to_end, f"{arm.arm} arm — {arm.sample_count} forks")
        else:
            _print_distribution(report.distribution, f"{arm.arm} arm — {arm.sample_count} samples")
        return
    variant_info = report.arms["variant"].variant
    swapped = f"{variant_info.get('original_path')} → {variant_info.get('variant_path')}"
    # Paths are printed whole, never broken across lines: they are copied.
    console.print(f"  swapped   {escape(swapped)}", soft_wrap=True)
    _print_wording(report.wording)
    if report.kind == "ab_to_end":
        _print_ab_to_end(report.to_end, k, alpha)
        return
    if report.target_note:
        console.print(f"  [yellow]{escape(report.target_note)}[/yellow]")
    caveat = (
        history_caveat(checkpoint)
        if checkpoint
        else "A checkpoint A/B is a fast screen; confirm a winner with full runs."
    )
    _print_ab(report.comparison, caveat)


# --- Phase 6b: refining a skill ------------------------------------------------------------

refine_app = typer.Typer(
    no_args_is_help=True,
    help="Refine a skill at a checkpoint: declare a target, gather proposed edits, screen them.",
)
app.add_typer(refine_app, name="refine")


def _print_refinement(found: dict) -> None:
    # Paths and the target are printed whole, never broken across lines: they are copied.
    console.print(f"[bold]refinement {found['id'][:8]}[/bold] — {found['status']}")
    if found.get("trashed_at"):
        console.print(f"  [yellow]in the trash since {_when(found['trashed_at'], seconds=False)}: it takes no new work[/yellow]"
                      f" [dim]— `tuningfork trash restore {found['id'][:8]}` brings it back[/dim]", soft_wrap=True)
    task = f" (task {escape(found['task_id'])})" if found.get("task_id") else ""
    console.print(f"  checkpoint  run {found['run'][:8]} step {found['step']}{task}")
    console.print(f"  skill       {escape(found['skill'])}, as the run was given it:")
    console.print(f"              {escape(_shown_path(Path(found['baseline']['path'])))}", soft_wrap=True)
    source = found.get("source")
    if source:
        console.print(f"  its file    {escape(_shown_path(Path(source['path'])))}", soft_wrap=True)
        if not source["matches"]:
            console.print("              [yellow]changed since the run was given it: proposals edit the run's copy[/yellow]")
    console.print(f"  target      {escape(found['target'].get('description') or '')}", soft_wrap=True)
    console.print("              [dim]declared before anything was proposed[/dim]")
    if found.get("normalizers"):
        console.print(f"  read by     {escape(', '.join(found['normalizers']))}")
    console.print(f"  started by  {escape(_who(found.get('created_by') or {}))}")


def _who(entry: dict) -> str:
    """Who proposed, or started, something: a person, an agent over MCP, or the proposer."""
    via = entry.get("via") or "?"
    if via == "builtin":
        return f"the built-in proposer, {entry.get('model') or 'the server default'} at {entry.get('endpoint')}"
    if via == "mcp":
        return f"an agent over MCP ({entry.get('client') or 'unnamed'})"
    return "a person" if via == "person" else via


def _print_wording(wording) -> None:
    """The edit an A/B tested, as a line diff of its two texts."""
    if wording is None:
        return
    if wording.diff is None:
        console.print(f"  the edit  [yellow]not shown: {escape(wording.missing or '')}[/yellow]")
        return
    changed = wording.diff["changed"]
    notes = []
    if wording.source == "files":
        notes.append("read from its files, which still hold the texts it tested")
    if wording.form == "body":
        notes.append("the model read the skill's body here, without its frontmatter")
    console.print(
        f"  the edit  {changed} line{'s' if changed != 1 else ''} changed"
        + (f" [dim]({escape('; '.join(notes))})[/dim]" if notes else "")
    )
    _print_diff(wording.diff["hunks"])


def _print_diff(hunks: list[dict]) -> None:
    for hunk in hunks:
        console.print(f"    [dim]@@ line {hunk['old_start']} @@[/dim]")
        for line in hunk["lines"]:
            style = {"-": "red", "+": "green"}.get(line["op"], "dim")
            # Never wrapped: a diff's line is the skill's line, copied as it is.
            console.print(f"    [{style}]{line['op']} {escape(line['text'])}[/{style}]", soft_wrap=True)


def _screen_verdict(verdict: dict) -> str:
    if verdict.get("quick"):
        # A quick look forks the proposal alone: its rate, beside the original's forks the
        # refinement drew before, if any — never a test, so never a verdict.
        rate, base = verdict["target"], verdict["baseline"]
        before = (f"; the original's {base['successes']}/{base['total']} drawn before"
                  if base.get("total") else "")
        return (f"quick look — {rate['successes']}/{rate['total']} forks passed: rates, never a verdict"
                f"{before} (screen {verdict['screen'][:8]})")
    if not verdict.get("complete"):
        word = "[yellow]no verdict yet[/yellow]"
    elif verdict.get("survives"):
        word = "[bold green]survives[/bold green]"
    elif verdict.get("worse"):
        word = "[bold red]worse[/bold red]"
    else:
        word = "not significant"
    rate, base = verdict["target"], verdict["baseline"]
    adjusted = f", Holm p {_fmt_p(verdict['adjusted_p'])}" if verdict.get("adjusted_p") is not None else ""
    by = ", forks to the end by the task's check" if verdict.get("reach") == "end" else ""
    return (
        f"{word} — {rate['successes']}/{rate['total']} against the original's "
        f"{base['successes']}/{base['total']}{adjusted}{by} (screen {verdict['screen'][:8]})"
    )


@refine_app.command("start")
def refine_start(
    run: str = typer.Argument(..., help="Run id, prefix, or label."),
    reference: str = typer.Argument(..., metavar="step", help=STEP_HELP),
    target: Optional[str] = typer.Option(
        None, "--target", help="A behaviour's letter from this step's latest resample."
    ),
    target_match: Optional[str] = typer.Option(
        None, "--target-match", help="A regular expression over what the model does."
    ),
    target_rules: Optional[str] = typer.Option(
        None, "--target-rules",
        help='Conditions that must all hold, as JSON: [{"if": "calls", "value": "write_file"}, '
        '{"if": "contains", "value": "row_count"}]. Conditions: calls, not_calls, contains, not_contains.',
    ),
    target_check: bool = typer.Option(
        False, "--target-check",
        help="The task's own check as the target, for a failure only the check can see: its screens run "
        "forks to the end (refine screen --to-end).",
    ),
    skill: Optional[str] = typer.Option(
        None, "--skill", help="The skill to refine, by name, where the run kept several: an imported chat.",
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Start refining a run's skill at one of its steps, toward a target declared now.

    The target comes first, before anything is proposed or sampled."""
    from tuningfork.ops import refine

    rules = None
    if target_rules:
        try:
            rules = json.loads(target_rules)
        except ValueError as exc:
            console.print(f"[red]--target-rules is not JSON: {escape(str(exc))}[/red]")
            raise typer.Exit(code=1) from None
    store = _store(home)
    try:
        made = refine.create(
            store, run, reference, target=target, target_match=target_match, target_rules=rules,
            target_check=target_check, created_by={"via": "person"}, skill=skill,
        )
    except OperationError as error:
        _refuse(error)
    _print_refinement(made.to_json())
    console.print(
        f"\nnext: ask for proposals — or give your own wording — then screen them:\n"
        f"  tuningfork refine propose {made.id[:8]} --builtin\n"
        f"  tuningfork refine propose {made.id[:8]} --file SKILL.md --why '…'\n"
        f"  tuningfork refine screen {made.id[:8]}{' --to-end' if target_check else ''}",
        soft_wrap=True,
    )
    if target_check:
        from tuningfork.ops.inspect import replay_warning

        warning = replay_warning(store, store.get_run(made.run_id), made.step_idx)
        if warning:
            console.print(f"[yellow]note:[/yellow] {escape(warning)}", soft_wrap=True)


@refine_app.command("list")
def refine_list(
    run: Optional[str] = typer.Option(None, "--run", help="Only this run's: id, prefix, or label."),
    every: bool = typer.Option(False, "--all", help="Closed refinements too."),
    limit: int = typer.Option(50, "--limit", "-n"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print only ids: REF=$(tuningfork refine list -q -n 1)"
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Refinements, newest first."""
    from tuningfork.ops import refine

    store = _store(home)
    run_id = _resolve_run(store, run).id if run else None
    found = refine.list_refinements(store, run=run_id, include_closed=every)[:limit]
    if quiet:
        for each in found:
            typer.echo(each.id)
        return
    if not found:
        console.print("no refinements yet — `tuningfork refine start <run> <step> --target-rules …`")
        return
    table = Table("refinement", "run", "step", "skill", "target", "proposals", "status", "started", box=None, pad_edge=False)
    for each in found:
        table.add_row(
            each.id[:8], each.run_id[:8], str(each.step_idx), escape(each.skill),
            _clip(each.target.get("description") or "", full=False, limit=60),
            str(len(refine.proposals(store, each.id))), each.status,
            _when(each.data.get("created_at") or 0, seconds=False),
        )
    console.print(table)


@refine_app.command("show")
def refine_show(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    packet: bool = typer.Option(
        False, "--packet", help="Print, as JSON, the evidence a proposer would be shown now."
    ),
    no_task: bool = typer.Option(False, "--no-task", help="With --packet: as shown without the task's criteria."),
    repair: bool = typer.Option(
        False, "--repair", help="Finish an approval a crash interrupted, as approval.json recorded it."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """A refinement: its target, its proposals as line diffs, what each screen found, each
    validation with its gate, and its approval.

    Each proposal says who proposed it, and why."""
    from tuningfork.ops import approve, refine
    from tuningfork.ops.packet import build_packet

    store = _store(home)
    try:
        if packet:
            built = build_packet(store, reference, include_task=not no_task, keep=False)
            typer.echo(json.dumps({"sha": built.sha, **built.data}, indent=2, ensure_ascii=False))
            return
        if repair:
            repaired = approve.repair(store, reference)
            console.print(f"approval {repaired['state']}: {escape(repaired['target'])}", soft_wrap=True)
            return
        # The person's own view: held-out verdicts and the gate included.
        shown = refine.view(store, reference, held_out=True)
    except OperationError as error:
        _refuse(error)
    _print_refinement(shown["refinement"])
    rid = shown["refinement"]["id"][:8]
    console.print(f"\n[bold]proposals[/bold] ({len(shown['proposals'])}) [dim]— hypotheses until a screen tests them[/dim]")
    for item in shown["proposals"]:
        first = item["proposers"][0]
        console.print(
            f"  [bold]{item['id'][:8]}[/bold]  {item['changed']} line{'s' if item['changed'] != 1 else ''} changed"
            f" · {escape(_who(first))}" + (f" · packet {first['packet']}" if first.get("packet") else ""),
            soft_wrap=True,
        )
        for also in item["proposers"][1:]:
            console.print(f"    [dim]proposed again by {escape(_who(also))}[/dim]")
        console.print(f"    why  {_clip(first.get('rationale') or '', full=False, limit=400)}")
        for warning in item.get("warnings") or []:
            console.print(f"    [yellow]{escape(warning)}[/yellow]")
        _print_diff(item["diff"])
        for verdict in item["screens"]:
            console.print(f"    screened  {_screen_verdict(verdict)}")
    if not shown["proposals"]:
        console.print(f"  none yet — `tuningfork refine propose {rid} --builtin`, or --file with your own wording")
    console.print(f"\n[bold]screens[/bold] ({len(shown['screens'])})")
    for screen in shown["screens"]:
        report = screen["report"]
        worse = [arm["name"] for arm in report["arms"] if arm.get("worse")]
        survivors = (f"{', '.join(report['survivors'])} survived" if report["survivors"] else "none survived") + (
            f"; [red]{', '.join(worse)} worse[/red]" if worse else ""
        )
        state = survivors if report["complete"] else "[yellow]unfinished: no verdict[/yellow]"
        by = " · by the task's check, forks to the end" if screen.get("reach") == "end" else ""
        if report.get("quick"):
            forks = max((arm.get("planned") or 0 for arm in report["arms"]), default=0)
            console.print(
                f"  {screen['experiment'][:8]}  {screen['status']}  quick look at {report['family']} proposal"
                f"{'s' if report['family'] != 1 else ''}, {forks} forks each{by} — rates only, no verdict"
            )
            continue
        console.print(
            f"  {screen['experiment'][:8]}  {screen['status']}  {report['family']} proposal"
            f"{'s' if report['family'] != 1 else ''}, n={report['baseline']['planned']} an arm{by} — {state}"
        )
    if not shown["screens"] and shown["proposals"]:
        if shown["refinement"]["status"] == "open":
            console.print(f"  none yet — `tuningfork refine screen {rid}`")
        elif shown.get("approval"):
            console.print("  none left in the recordings — the approval keeps those it stood on")
    if shown["validations"]:
        console.print(f"\n[bold]validations[/bold] ({len(shown['validations'])}) [dim]— held-out verdicts are yours: "
                      "no MCP tool shows them[/dim]")
        for record in shown["validations"]:
            _print_validation(record)
    _print_approval(shown.get("approval"), rid)


def _arm_text(verdicts: list[dict], planned: int = 0) -> str:
    """An arm of a validation: its pass rate with its interval, and what was not counted."""
    from tuningfork.analysis.validation import count
    from tuningfork.ops.compact import rate_text

    arm = count(verdicts, planned)
    text = rate_text(arm.rate.to_json()) if arm.rate else "nothing counted"
    if arm.timeouts:
        text += f", {arm.timeouts} timed out (counted as failed)"
    if arm.excluded:
        text += f", {len(arm.excluded)} not counted"
    return escape(text)


def _print_gate(gate: dict, indent: str = "    ") -> None:
    """The gate a validation meets, check by check, each with what it cannot see."""
    for check in gate["checks"]:
        mark = "[green]met[/green]    " if check["passed"] else "[bold red]NOT met[/bold red]"
        powered = " [yellow](low-powered)[/yellow]" if check.get("low_powered") else ""
        console.print(f"{indent}{check['id']} {mark} {escape(check['name'])}{powered}: {escape(check['reason'])}",
                      soft_wrap=True)
    for note in gate.get("notes") or []:
        console.print(f"{indent}[yellow]note:[/yellow] {escape(note)}", soft_wrap=True)


def _print_validation(record: dict) -> None:
    gate = record.get("gate") or {}
    failed = [check["id"] for check in gate.get("checks") or [] if not check["passed"]]
    verdict = ("[yellow]unfinished: no verdict[/yellow]" if not record.get("complete")
               else "[bold green]approved[/bold green] — the gate as it was met then" if record.get("approved")
               else "[bold green]meets the gate[/bold green]" if gate.get("passed")
               else f"[bold red]does not meet the gate[/bold red] ({', '.join(failed)})")
    state = record["status"] if record.get("complete") else f"[yellow]{record['status']}, unfinished[/yellow]"
    console.print(f"  [bold]{record['id'][:8]}[/bold]  {state}  proposal {record['proposal'][:8]} — {verdict}")
    planned = record.get("planned") or {}
    for name in ("forward", "training"):
        part = record.get(name)
        if part:
            console.print(f"    {name:<9} edit {_arm_text(part['edit'], planned.get(name, 0))}; "
                          f"original {_arm_text(part['original'], planned.get(name, 0))}", soft_wrap=True)
    for entry in record.get("held_out") or []:
        console.print(f"    held out {escape(entry['eval'])}: edit {_arm_text(entry['edit'], planned.get('held_out', 0))}; "
                      f"original {_arm_text(entry['original'], planned.get('held_out', 0))}", soft_wrap=True)
    # Like an unfinished screen, an unfinished validation has its counts so far and no verdict.
    if gate and record.get("complete"):
        _print_gate(gate)


def _print_approval(state: dict | None, rid: str) -> None:
    if not state:
        return
    console.print(f"\n[bold]approval[/bold]  proposal {state['proposal'][:8]} — {state['state']}")
    console.print(f"  written to  {escape(state['target'])}", soft_wrap=True)
    if state.get("windows"):
        console.print(f"              {escape(state['windows'])}", soft_wrap=True)
    console.print(f"  file before {escape(state['backup'])}", soft_wrap=True)
    console.print(f"  changelog   {escape(state['changelog'])}", soft_wrap=True)
    if state["state"] != "applied":
        console.print(f"  [yellow]{escape(state['message'])}[/yellow]")
        if state["state"] != "diverged":
            console.print(f"  `tuningfork refine show {rid} --repair` finishes it")


@refine_app.command("propose")
def refine_propose(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    file: Optional[Path] = typer.Option(
        None, "--file", help="Your own wording: the whole skill, edited by hand."
    ),
    why: Optional[str] = typer.Option(None, "--why", help="With --file: why it should help, from the evidence."),
    builtin: bool = typer.Option(
        False, "--builtin",
        help="Ask the built-in proposer — the model `tuningfork connections proposer` names — from the evidence.",
    ),
    m: int = typer.Option(3, "-m", min=1, max=6, help="With --builtin: the proposals to ask for."),
    no_task: bool = typer.Option(
        False, "--no-task", help="With --builtin: leave the task's own criteria out of what the proposer is shown."
    ),
    base_url: Optional[str] = typer.Option(
        None, help="With --builtin, when the proposer has no endpoint of its own: the server to ask. "
        "Defaults to TUNINGFORK_UPSTREAM, else http://127.0.0.1:8081. Point it at a running proxy to share its gate.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="With --builtin: say where the evidence would be sent, and how much; send nothing."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Propose an edit of the skill: your own wording, or the built-in proposer's.

    Either is a hypothesis until a screen tests it."""
    from tuningfork.ops import proposer, refine

    if bool(file) == builtin:
        console.print("[red]give --file with --why, or --builtin[/red]")
        raise typer.Exit(code=1)
    store = _store(home)
    if file:
        if not why:
            console.print("[red]say why with --why: what in the evidence it answers[/red]")
            raise typer.Exit(code=1)
        try:
            with open(file, encoding="utf-8", newline="") as handle:
                text = handle.read()
        except OSError as exc:
            console.print(f"[red]cannot read {escape(str(file))}: {escape(exc.strerror or str(exc))}[/red]")
            raise typer.Exit(code=1) from None
        try:
            made = refine.add_proposal(
                store, reference, text=text, rationale=why, proposer={"via": "person"}, strict=False
            )
        except OperationError as error:
            _refuse(error)
        _print_proposals([made])
        return
    try:
        plan = proposer.prepare_propose(store, reference, m=m, proxy_url=base_url, include_task=not no_task)
    except OperationError as error:
        _refuse(error)
    data = plan.to_json()
    console.print(f"[bold]the built-in proposer[/bold] — {data['proposals']} proposal{'s' if data['proposals'] != 1 else ''}")
    console.print(f"  sends     {escape(data['sends'])}")
    console.print(f"  model     {escape(data['model'] or 'the server default')} ({data['wire']}, max_tokens {data['max_tokens']})")
    console.print(f"  packet    {data['packet']}" + ("" if not no_task else " [dim](without the task's criteria)[/dim]"))
    for warning in data["warnings"]:
        console.print(f"  [yellow]{escape(warning)}[/yellow]")
    if dry_run:
        return
    console.print("  asking…")
    try:
        result = asyncio.run(proposer.execute_propose(store, plan))
    except OperationError as error:
        _refuse(error)
    console.print(
        f"  answered in {result.seconds:.0f}s"
        + (f", {result.usage.get('completion_tokens')} tokens" if result.usage.get("completion_tokens") else "")
        + f" · kept in {escape(result.record)}"
    )
    if result.read_from == "reasoning":
        console.print("  [yellow]its answer was empty: the proposals were read from the end of its reasoning[/yellow]")
    _print_proposals(result.accepted)
    for rejected in result.rejected:
        which = f"proposal {rejected['index']}" if rejected.get("index") else "the answer"
        hint = f" — {rejected['hint']}" if rejected.get("hint") else ""
        console.print(f"  [yellow]{which} not kept: {escape(rejected['reason'])}{escape(hint)}[/yellow]")
    if result.accepted:
        rid = plan.refinement.id[:8]
        console.print(
            f"\n[dim]hypotheses until screened: `tuningfork refine screen {rid} --proposals "
            f"{','.join(each.id[:8] for each in result.accepted)}`[/dim]"
        )


def _print_proposals(made: list) -> None:
    for each in made:
        data = each.data
        again = " [dim](the same text was proposed before: this proposer is added to it)[/dim]" if each.duplicate else ""
        console.print(f"  proposal [bold]{each.id[:8]}[/bold]: {data['changed']} line{'s' if data['changed'] != 1 else ''} changed{again}")
        console.print(f"    why  {_clip(data['proposers'][-1].get('rationale') or '', full=False, limit=300)}")
        for warning in data.get("warnings") or []:
            console.print(f"    [yellow]{escape(warning)}[/yellow]")


@refine_app.command("screen")
def refine_screen(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    proposals: Optional[str] = typer.Option(
        None, "--proposals",
        help="The proposals to screen, comma-separated ids or prefixes: the family the correction "
        "counts. Default: every proposal the refinement has, if there are at most 6.",
    ),
    n: int = typer.Option(20, "-n", help="Samples per arm, the original's included."),
    alpha: float = typer.Option(0.05, help="Family-wise significance level."),
    base_url: Optional[str] = typer.Option(None, help="Server to sample from; as for resample."),
    concurrency: int = typer.Option(1, help="Concurrent requests. Keep at 1 on a shared server."),
    max_tokens: int = typer.Option(4096, help="Cap per sample; hitting it counts as truncated."),
    seed_base: Optional[int] = typer.Option(None, help=SEED_HELP),
    prompt_cache: bool = typer.Option(False, "--prompt-cache", help=CACHE_HELP),
    to_end: bool = typer.Option(
        False, "--to-end",
        help="Each arm's samples forks run from the checkpoint to the end through the harness, the task's "
        "check deciding: the screen for a refinement toward the check, open to any other.",
    ),
    proxy: Optional[str] = typer.Option(None, "--proxy", help="With --to-end: the running proxy the forks go through."),
    spec: Optional[Path] = typer.Option(None, "--spec", help="With --to-end: the task spec, for a run recorded before launches kept it."),
    checkpoints: int = typer.Option(
        4, "--checkpoints", min=1, max=10,
        help="With --to-end: runs the forks are drawn from — the refinement's own, and others of its task at the "
        "same step, passing and failing by turns, each arm's n spread evenly over them (fewer where fewer serve). "
        "An effect that holds across checkpoints is one fresh runs keep. 1 forks the refinement's run alone.",
    ),
    quick: bool = typer.Option(
        False, "--quick",
        help="With --to-end: a quick look — 8 forks of each proposal and none of the original, beside the original's "
        "forks this refinement drew before. Rates only, never a verdict: a full screen decides.",
    ),
    fresh_original: bool = typer.Option(
        False, "--fresh-original",
        help="With --to-end: draw the original's forks anew, rather than count those this refinement's earlier "
        "screens drew at the same checkpoints.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan and what could survive; sample nothing."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Screen proposals at the checkpoint, each against an arm of the original.

    Holm-corrected over the proposals named, a family fixed before the first sample. With
    --to-end, the arms are forks run to the end and the task's check decides each, drawn
    from several checkpoints by default; the original's forks an earlier screen of the
    refinement drew there count again, and only what n still needs is drawn."""
    from tuningfork.ops import refine
    from tuningfork.ops.screen import MAX_ARMS, execute_screen, execute_screen_to_end

    store = _store(home)
    try:
        if proposals:
            chosen = [each.strip() for each in proposals.split(",") if each.strip()]
        else:
            chosen = [each.id for each in refine.proposals(store, reference)]
            if len(chosen) > MAX_ARMS:
                raise OperationError(
                    f"the refinement has {len(chosen)} proposals; a screen holds at most {MAX_ARMS}",
                    hint="name the ones to screen with --proposals",
                )
        if to_end:
            prepared = refine.prepare_refinement_screen(
                store, reference, chosen, to_end=True, alpha=alpha, n=n, concurrency=concurrency,
                base_url=f"{proxy.rstrip('/')}/v1" if proxy else None, spec=spec, checkpoints=checkpoints,
                quick=quick, fresh_original=fresh_original,
            )
        else:
            prepared = refine.prepare_refinement_screen(
                store, reference, chosen, base_url=base_url, alpha=alpha, n=n, concurrency=concurrency,
                max_tokens=max_tokens, seed_base=seed_base, prompt_cache=prompt_cache,
                quick=quick, fresh_original=fresh_original,
            )
    except OperationError as error:
        _refuse(error)
    plan = prepared.to_json()
    if to_end:
        n = prepared.n
        _print_fork_plan(prepared.baseline, n, arms=prepared.family + 1)
        console.print("  target    the task's check passes [dim](declared by the task, before any fork ran)[/dim]")
        for arm in plan["arms"]:
            console.print(f"  arm       {arm['name']}: {arm['changed_lines']} line{'s' if arm['changed_lines'] != 1 else ''} changed")
        if prepared.quick:
            console.print("            [dim]a quick look: rates only, no verdict[/dim]")
        else:
            console.print(f"            Holm over {plan['family']} proposal{'s' if plan['family'] != 1 else ''}, α={alpha}")
        original = plan["original"]
        if original["reused"]:
            console.print(f"  original  {original['reused']} fork(s) drawn by an earlier screen of this refinement, counted "
                          f"again; {original['drawn']} to draw")
        if len(plan["checkpoints"]) > 1:
            drawn = ", ".join(f"{each['run'][:8]} step {each['step']} ({each['verdict'] or 'unjudged'}, {each['n']} an arm)"
                              for each in plan["checkpoints"])
            console.print(f"  from      {len(plan['checkpoints'])} checkpoints: {escape(drawn)}", soft_wrap=True)
        for note in plan.get("notes") or []:
            console.print(f"  [dim]passed over: {escape(note)}[/dim]", soft_wrap=True)
        for warning in plan["warnings"]:
            console.print(f"  [yellow]{escape(warning)}[/yellow]")
        console.print(f"  forks     {plan['total']} in all: "
                      + ("the original first, then each arm" if plan["original"]["drawn"]
                         else "the proposals' — the original's counted from before" if not prepared.quick
                         else "the proposals' alone"))
        if dry_run:
            return
        console.print()
        try:
            result = execute_screen_to_end(store, prepared, progress=_fork_progress(n, show_arm=True))
        except OperationError as error:
            _refuse(error)
        _print_screen(result.report, result.experiments)
        own = next((arm for arms in result.by_arm.values() for arm in arms if arm.group_id == result.group_id), None)
        console.print(
            f"[dim]screen {(own.id if own else result.group_id)[:8]} · {result.elapsed / 60:.1f} min · forks to the "
            f"end, the task's check deciding · `tuningfork refine show {prepared.refinement[:8]}`[/dim]"
        )
        return
    _print_checkpoint(prepared.checkpoint, prepared.plan, prepared.base_url, per_arm=True, arms=prepared.family + 1)
    console.print(f"  target    {escape(prepared.target.description)}", soft_wrap=True)
    console.print("            [dim]declared when the refinement began[/dim]")
    for arm in plan["arms"]:
        console.print(f"  arm       {arm['name']}: {arm['changed_lines']} line{'s' if arm['changed_lines'] != 1 else ''} changed")
    power = plan["power"]
    base = power.get("baseline")
    against = (
        f"the step did it {base['hits']} times in {base['of']} in resample {base['resample'][:8]}"
        if base else "at best: no resample here says how often the step does it"
    )
    console.print(f"  survives  with at least {power['needed']} of {n} — {against}", soft_wrap=True)
    family = plan["family"]
    console.print(f"            Holm over {family} proposal{'s' if family != 1 else ''}, α={alpha}")
    for warning in plan["warnings"]:
        console.print(f"  [yellow]{escape(warning)}[/yellow]")
    console.print(f"  samples   {plan['total']} in all: the original first, then each arm")
    if dry_run:
        return
    console.print()
    try:
        result = asyncio.run(
            execute_screen(store, prepared, progress=_progress_printer(n, show_arm=True))
        )
    except OperationError as error:
        _refuse(error)
    _print_notes(result.notes)
    _print_screen(result.report, result.experiments)
    console.print(
        f"[dim]screen {result.experiments['original'].id[:8]} · {result.elapsed / 60:.1f} min · "
        f"`tuningfork refine show {prepared.refinement[:8]}` shows every proposal with its verdicts[/dim]"
    )


@refine_app.command("validate")
def refine_validate(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    proposal: str = typer.Argument(..., help="The proposal to validate, by id or prefix: one that survived a screen."),
    n: int = typer.Option(5, "-n", help="Forks an arm from the checkpoint to the end, the original's included (at least 5)."),
    runs: int = typer.Option(5, "--runs", help="Full runs an arm, of the training task and of each held-out task (at least 5)."),
    alpha: float = typer.Option(0.05, help="Significance level for the not-worse checks."),
    concurrency: int = typer.Option(1, help="Runs at once; no more than the proxy's --max-inflight."),
    proxy: Optional[str] = typer.Option(None, help="The proxy every run goes through, recording here. Default: TUNINGFORK_PROXY."),
    keep_private: Optional[Path] = typer.Option(
        None, "--keep-private",
        help="Copy each held-out arm's private recordings into this folder, to debug with `tuningfork ui "
        "--home`. An agent that can read files can then read them too.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan; run nothing."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Validate a screened proposal with full runs, each arm against the original's.

    Forks from the checkpoint to the end (does hitting the target carry the task?), full runs
    of the training task (does the edit disturb an earlier step?), and full runs of each
    held-out task, kept only as verdicts (does it break what it was not made for?)."""
    from tuningfork.ops import refine, validate
    from tuningfork.ops.common import Cancelled

    store = _store(home)
    try:
        plan = validate.prepare_validate(
            store, reference, proposal, n=n, runs=runs, alpha=alpha, concurrency=concurrency,
            proxy_url=proxy, keep_private=keep_private,
        )
    except OperationError as error:
        _refuse(error)
    shown = plan.to_json()
    screen = shown["screen"]
    console.print(f"[bold]validate proposal {plan.proposal.id[:8]}[/bold] of refinement {plan.refinement.id[:8]} — "
                  f"{escape(plan.task_id)}, at step {plan.refinement.step_idx}")
    target, base = screen["target"], screen["baseline"]
    console.print(f"  screened  survived screen {screen['screen'][:8]}: {target['successes']}/{target['total']} against "
                  f"the original's {base['successes']}/{base['total']}"
                  + (f", Holm p {_fmt_p(screen['adjusted_p'])}" if screen.get("adjusted_p") is not None else ""))
    console.print(f"  forward   {n} forks an arm from step {plan.refinement.step_idx} to the end, the original first")
    console.print(f"  training  {runs} full runs an arm of {escape(plan.task_id)}, the edit from the first step")
    if plan.held_out:
        held = ", ".join(entry.id for entry in plan.held_out)
        console.print(f"  held out  {runs} full runs an arm of each of: {escape(held)} — only their verdicts are kept,")
        under = shown["private"]["under"]
        console.print(f"            recorded in a temporary folder under {escape(under)}, outside the recordings folder,",
                      soft_wrap=True)
        console.print("            and deleted when they end"
                      + (f"; a copy kept at {escape(shown['private']['kept_at'])}" if shown["private"]["kept_at"] else ""),
                      soft_wrap=True)
    console.print(f"  runs      {plan.total} in all, through {escape(plan.proxy_url)}")
    for warning in plan.warnings:
        console.print(f"  [yellow]{escape(warning)}[/yellow]", soft_wrap=True)
    if dry_run:
        return
    try:
        result = validate.execute_validate(store, plan, progress=_validate_progress(plan))
    except Cancelled:
        console.print("[yellow]stopped: what finished is kept in the refinement, with no verdict[/yellow]")
        raise typer.Exit(code=1) from None
    except OperationError as error:
        _refuse(error)
    record = result.record
    gate = validate.gate_for(store, refine.get(store, plan.refinement.id), record)
    console.print()
    _print_validation({**record, "gate": gate.to_json()})
    rid, pid = plan.refinement.id[:8], plan.proposal.id[:8]
    if gate.passed:
        console.print(f"\nit meets the gate: `tuningfork refine approve {rid} {pid}` shows what approving writes")
    else:
        console.print(f"\n[dim]validation {record['id'][:8]} · `tuningfork refine show {rid}` shows it again[/dim]")


def _validate_progress(plan):
    """Prints each run of a validation as it ends: forks and training runs with their verdicts,
    held-out runs only as ended."""
    import threading

    lock = threading.Lock()
    seen: dict[str, int] = {}
    titles = {
        "forward": f"forward — forks from step {plan.refinement.step_idx} to the end",
        "training": f"training — full runs of {plan.task_id}",
        "held_out": "held out {eval} — verdicts only",
    }
    current: dict[str, str] = {}

    def on_event(event) -> None:
        with lock:
            if event.kind == "validation":
                part, arm = event.data.get("part"), event.data.get("arm")
                title = titles[part].format(eval=event.data.get("eval"))
                if current.get("title") != title:
                    console.print(f"\n[bold]{escape(title)}[/bold]")
                    current["title"] = title
                current["arm"] = arm or ""
            elif event.kind == "run":
                # A fork's arm is the A/B's "variant": here, as everywhere in a validation, the edit.
                arm = {"variant": "edit"}.get(event.data.get("arm") or "", event.data.get("arm")) or current.get("arm") or ""
                key = f"{current.get('title')}/{arm}"
                seen[key] = seen.get(key, 0) + 1
                total = plan.n if current.get("title", "").startswith("forward") else plan.runs
                console.print(f"  {arm:<9}{seen[key]:>3}/{total}{_fork_line(event.subject, timeouts_fail=True)[1:]}")
            elif event.kind == "held_out_run":
                arm = current.get("arm") or ""
                key = f"{current.get('title')}/{arm}"
                seen[key] = seen.get(key, 0) + 1
                console.print(f"  {arm:<9}{seen[key]:>3}/{plan.runs}  [dim]ended[/dim]")
            elif event.kind == "waiting" and not current.get("told"):
                current["told"] = "yes"
                console.print("  [yellow]waiting: tuningfork is paused[/yellow]")

    return on_event


@refine_app.command("approve")
def refine_approve(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    proposal: str = typer.Argument(..., help="The proposal to approve, by id or prefix."),
    validation: Optional[str] = typer.Option(
        None, help="The validation it stands on, by id or prefix. Default: the proposal's newest finished one."
    ),
    spec: Optional[str] = typer.Option(
        None, help="The task spec where it is now: when it moved since the run, or to write another checkout."
    ),
    target: Optional[str] = typer.Option(None, help="The SKILL.md to write, when the spec now leads elsewhere."),
    confirm: Optional[str] = typer.Option(
        None, "--confirm", help="The code this approval shows, when you have read it already."
    ),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Approve a validated proposal: write it into the skill's own SKILL.md, in place.

    Shows the file, the diff and the gate, and writes nothing until the code it shows is
    typed. The file as it was, and a changelog entry, are kept beside the skill folder."""
    from tuningfork.ops import approve

    store = _store(home)
    try:
        plan = approve.prepare_approve(store, reference, proposal, validation=validation, spec=spec, target=target)
    except OperationError as error:
        _refuse(error)
    shown = plan.to_json()
    found = shown["target"]
    console.print(f"[bold]approve proposal {plan.proposal.id[:8]}[/bold] of refinement {plan.refinement.id[:8]}, "
                  f"on validation {plan.record['id'][:8]}")
    console.print(f"  writes     {escape(found['path'])}", soft_wrap=True)
    if found.get("windows"):
        console.print(f"             {escape(found['windows'])}", soft_wrap=True)
    keeps = shown["line_endings"] + " line endings" + (", a byte-order mark" if shown["bom"] else "")
    console.print(f"  keeps      {keeps}")
    console.print(f"  file before → {escape(found['history'])}/", soft_wrap=True)
    console.print(f"  changelog   → {escape(found['changelog'])}", soft_wrap=True)
    console.print("  the gate   every check met:")
    _print_gate(shown["gate"], indent="    ")
    console.print("  the edit")
    _print_diff(shown["diff"]["hunks"])
    for warning in shown["warnings"]:
        if warning not in shown["gate"].get("notes", []):
            console.print(f"  [yellow]{escape(warning)}[/yellow]", soft_wrap=True)
    console.print(f"\n  code  [bold]{plan.code}[/bold]")
    if confirm is None:
        try:
            confirm = typer.prompt("Type the code to write it (anything else writes nothing)", default="",
                                   show_default=False)
        except (typer.Abort, EOFError):
            confirm = ""
    if (confirm or "").strip().lower() != plan.code:
        console.print("nothing written")
        raise typer.Exit(code=1)
    try:
        result = approve.execute_approve(store, plan, confirm=confirm, front_end="cli").to_json()
    except OperationError as error:
        _refuse(error)
    console.print(f"written: {escape(result['target'])}", soft_wrap=True)
    console.print(f"  the file before  {escape(result['backup'])}", soft_wrap=True)
    console.print(f"  changelog        {escape(result['changelog'])}", soft_wrap=True)
    console.print("  [dim]to undo, copy the file before back over it[/dim]")


@refine_app.command("close")
def refine_close(
    reference: str = typer.Argument(..., metavar="refinement", help="Refinement id or prefix."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Close a refinement: no more proposals or screens; everything it holds is kept."""
    from tuningfork.ops import refine

    store = _store(home)
    try:
        closed = refine.close(store, reference)
    except OperationError as error:
        _refuse(error)
    console.print(f"refinement {closed.id[:8]} closed — its proposals and screens are kept")


# --- Phase 3: live breakpoints -----------------------------------------------------------

break_app = typer.Typer(
    no_args_is_help=True,
    help="Pause model calls before or after the model, on a running proxy.",
)
hold_app = typer.Typer(no_args_is_help=True, help="Inspect and decide calls held at a breakpoint.")
jobs_app = typer.Typer(
    no_args_is_help=True,
    help="Work a running proxy is doing, started from its web UI or here: list, show, cancel, wait.",
)
app.add_typer(break_app, name="break")
app.add_typer(hold_app, name="hold")
app.add_typer(jobs_app, name="jobs")
connections_app = typer.Typer(
    no_args_is_help=True,
    help="What a running proxy talks to: its inference server, and the harnesses it launches.",
)
app.add_typer(connections_app, name="connections")

_PROXY_HELP = "The proxy to control. Defaults to TUNINGFORK_PROXY, else http://127.0.0.1:8787."


def _control(proxy: Optional[str]):
    from tuningfork.control import ControlClient, ControlError

    try:
        client = ControlClient(proxy)
    except ControlError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    _closables.append(client)
    return client


def _or_exit(call):
    """Run a control call, turning a refusal into a readable error rather than a traceback."""
    from tuningfork.control import ControlError

    try:
        return call()
    except ControlError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None


@break_app.command("add")
def break_add(
    phase: Optional[str] = typer.Option(
        None, help="request (before the model) or response (after). Inferred if omitted."
    ),
    tool: Optional[str] = typer.Option(None, help="Tool name, or a glob such as write_*."),
    path: Optional[str] = typer.Option(None, "--path", help="Glob on a path argument: '*.json'."),
    args: Optional[str] = typer.Option(None, "--args", help="Regex over arguments, as key=value."),
    output: Optional[str] = typer.Option(
        None, "--output", help="Regex over text, reasoning and calls."
    ),
    no_reasoning: bool = typer.Option(
        False, "--no-reasoning", help="Stop where the model calls a tool without reasoning first.",
    ),
    diverges_from: Optional[str] = typer.Option(
        None, "--diverges-from", help="Stop where behaviour differs from this run's."
    ),
    after_tool: Optional[str] = typer.Option(
        None, "--after-tool", help="Stop before the model sees this tool's result."
    ),
    context: Optional[str] = typer.Option(None, "--context", help="Regex over the request."),
    step: Optional[int] = typer.Option(None, "--step", help="Only this step."),
    from_step: Optional[int] = typer.Option(None, "--from-step", help="This step and later."),
    run: Optional[str] = typer.Option(None, help="Only this run (id, prefix, or label)."),
    task: Optional[str] = typer.Option(None, help="Only runs of this task."),
    launch: Optional[str] = typer.Option(None, help="Only calls of this launch (a batch run or a fork)."),
    once: bool = typer.Option(False, help="Disable after the first hit."),
    label: Optional[str] = typer.Option(None, help="A name to recognise it by."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Add a breakpoint. `--tool write_file --path '*.json'` stops before a JSON file is
    written; `--from-step 2` stops before the model sees step 2's context."""
    if step is not None and from_step is not None:
        console.print("[red]use --step or --from-step, not both[/red]")
        raise typer.Exit(code=1)
    response_side = no_reasoning or any(v is not None for v in (tool, path, args, output, diverges_from))
    created = _or_exit(
        lambda: _control(proxy).add_breakpoint(
            phase=phase or ("response" if response_side else "request"),
            tool=tool,
            path_glob=path,
            args_regex=args,
            output_regex=output,
            diverges_from=diverges_from,
            no_reasoning=no_reasoning or None,
            after_tool=after_tool,
            context_regex=context,
            step_min=step if step is not None else from_step,
            step_max=step,
            run=run,
            task=task,
            launch=launch,
            once=once or None,
            label=label,
        )
    )
    console.print(f"breakpoint [bold]{created['id']}[/bold] — {escape(created['description'])}")


@break_app.command("list")
def break_list(proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP)) -> None:
    """Breakpoints on the running proxy."""
    found = _or_exit(lambda: _control(proxy).breakpoints())
    if not found:
        console.print("no breakpoints")
        return
    table = Table("id", "when", "hits", "enabled", box=None)
    for bp in found:
        table.add_row(
            bp["id"], escape(bp["description"]), str(bp["hits"]), "yes" if bp["enabled"] else "no"
        )
    console.print(table)


@break_app.command("rm")
def break_rm(
    breakpoint_id: str = typer.Argument(..., help="Breakpoint id or prefix."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Remove a breakpoint."""
    removed = _or_exit(lambda: _control(proxy).remove_breakpoint(breakpoint_id))
    console.print(f"removed {removed['id']}")


@break_app.command("on")
def break_on(
    breakpoint_id: str = typer.Argument(...),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Enable a breakpoint."""
    _or_exit(lambda: _control(proxy).set_enabled(breakpoint_id, True))
    console.print(f"{breakpoint_id} enabled")


@break_app.command("off")
def break_off(
    breakpoint_id: str = typer.Argument(...),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Disable a breakpoint without removing it."""
    _or_exit(lambda: _control(proxy).set_enabled(breakpoint_id, False))
    console.print(f"{breakpoint_id} disabled")


def _print_hold(hold: dict, full: bool = False) -> None:
    where = f"run {hold['run_id'][:8]}" if hold.get("run_id") else "a new run"
    phase = "before the model" if hold["phase"] == "request" else "after the model"
    console.print(
        f"[bold]hold {hold['id']}[/bold] — {phase}, {where} step {hold['step']}, "
        f"held {hold['held_seconds']}s"
    )
    console.print(f"  reason  {escape(hold['reason'])}")
    messages = hold.get("view") or (hold.get("request") or {}).get("messages") or []
    shown = messages if full else messages[-3:]
    if len(shown) < len(messages):
        console.print(f"  [dim]… {len(messages) - len(shown)} earlier message(s)[/dim]")
    for message in shown:
        console.print(f"  [cyan]{escape(describe_message(message, 110))}[/cyan]")
    response = hold.get("response")
    if response:
        message = (response.get("choices") or [{}])[0].get("message") or {}
        console.print("\n[bold]the model answered[/bold]")
        for key in ("reasoning_content", "reasoning"):
            if message.get(key):
                console.print(f"  [magenta]{key}[/magenta]: {_clip(message[key], full)}")
        if message.get("content"):
            console.print(f"  content: {_clip(message['content'], full)}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            console.print(
                f"  [yellow]tool_call[/yellow] {escape(str(function.get('name')))}"
                f"({_clip(function.get('arguments', ''), full)})"
            )
    hint = "request" if hold["phase"] == "request" else "response"
    console.print(
        f"\n[dim]tuningfork hold continue {hold['id']} · step · drop — to edit the {hint}: "
        f"hold show {hold['id']} --json > f.json, edit, hold continue {hold['id']} "
        "--edited f.json[/dim]"
    )


@hold_app.command("list")
def hold_list(
    recent: bool = typer.Option(False, help="Include recently decided holds."),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print only the ids of calls waiting: HOLD=$(... -q)"
    ),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Calls currently held at a breakpoint."""
    found = _or_exit(lambda: _control(proxy).holds(recent=recent))
    if quiet:
        for hold in found:
            if not hold["action"] and not hold["abandoned"]:
                typer.echo(hold["id"])
        return
    if not found:
        console.print("nothing is held")
        return
    table = Table("hold", "when", "run", "step", "held", "state", "reason", box=None)
    for hold in found:
        state = hold["action"] or ("abandoned" if hold["abandoned"] else "[bold]waiting[/bold]")
        table.add_row(
            hold["id"],
            "before model" if hold["phase"] == "request" else "after model",
            (hold["run_id"] or "new")[:8],
            str(hold["step"]),
            f"{hold['held_seconds']}s",
            state,
            escape(hold["reason"]),
        )
    console.print(table)


@hold_app.command("show")
def hold_show(
    hold_id: str = typer.Argument(..., help="Hold id or prefix."),
    as_json: bool = typer.Option(
        False, "--json", help="Print just the editable object: the request, or the response."
    ),
    full: bool = typer.Option(False, help="Show every message in full."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Show a held call."""
    hold = _or_exit(lambda: _control(proxy).hold(hold_id))
    if as_json:
        editable = hold["request"] if hold["phase"] == "request" else hold["response"]
        typer.echo(json.dumps(editable, indent=2, ensure_ascii=False))
        return
    _print_hold(hold, full)


def _decide(hold_id: str, action: str, edited: Optional[Path], proxy: Optional[str]) -> None:
    client = _control(proxy)
    changes = {}
    if edited is not None:
        phase = _or_exit(lambda: client.hold(hold_id))["phase"]
        try:
            loaded = json.loads(edited.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"[red]could not read {edited}: {exc}[/red]")
            raise typer.Exit(code=1) from None
        changes = {"request" if phase == "request" else "response": loaded}
    decided = _or_exit(lambda: client.decide(hold_id, action, **changes))
    edited_note = " with your edit" if changes else ""
    console.print(f"{decided['id']}: {action}{edited_note}")


@hold_app.command("continue")
def hold_continue(
    hold_id: str = typer.Argument(...),
    edited: Optional[Path] = typer.Option(None, help="JSON file: an edited request or response."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Let a held call proceed, optionally edited."""
    _decide(hold_id, "continue", edited, proxy)


@hold_app.command("step")
def hold_step(
    hold_id: str = typer.Argument(...),
    edited: Optional[Path] = typer.Option(None, help="JSON file: an edited request or response."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Proceed, and stop again at the next stop point."""
    _decide(hold_id, "step", edited, proxy)


@hold_app.command("drop")
def hold_drop(
    hold_id: str = typer.Argument(...),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Fail a held call; the harness receives an error."""
    _decide(hold_id, "drop", None, proxy)


@hold_app.command("wait")
def hold_wait(
    timeout: float = typer.Option(300.0, help="Give up after this many seconds."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Block until a call is held, then show it. For scripts, and for watching live."""
    hold = _or_exit(lambda: _control(proxy).wait_for_hold(timeout=timeout))
    if hold is None:
        console.print(f"nothing was held within {timeout:.0f}s")
        raise typer.Exit(code=1)
    _print_hold(hold)


# --- jobs --------------------------------------------------------------------------------

_JOB_TONES = {"done": "green", "failed": "red", "cancelled": "yellow", "interrupted": "yellow"}


def _job_state(job: dict) -> str:
    status = job["status"]
    if status == "queued" and job.get("queue_position") is not None:
        return f"queued (#{job['queue_position'] + 1})"
    if status == "running":
        progress = job.get("progress") or {}
        done, total = progress.get("done"), progress.get("total")
        shown = f"running {done}/{total}" if total else "running"
        left = progress.get("remaining_s")
        if isinstance(left, (int, float)):
            shown += f", about {max(1, round(left / 60))} min left at its pace so far"
        return f"{shown}, {progress['waiting']}" if progress.get("waiting") else shown
    tone = _JOB_TONES.get(status)
    return f"[{tone}]{status}[/{tone}]" if tone else status


def _job_subject(job: dict) -> str:
    params = job.get("params") or {}
    if params.get("run"):
        return f"run {str(params['run'])[:8]} step {params.get('step')}"
    if params.get("scenario"):
        return f"scenario {params['scenario']}"
    if params.get("spec"):
        return Path(str(params["spec"])).name
    if params.get("hold"):
        return f"held call {str(params['hold'])[:8]}"
    return ""


@jobs_app.command("list")
def jobs_list(proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP)) -> None:
    """Recent jobs, newest first."""
    found = _or_exit(lambda: _control(proxy).jobs())
    if not found:
        console.print("no jobs yet")
        return
    table = Table("job", "kind", "state", "of", "via", "started", box=None)
    for job in found:
        table.add_row(
            job["id"][:8], job["kind"], _job_state(job), escape(_job_subject(job)),
            escape(_job_via(job)), _when(job.get("started_at") or job["created_at"]),
        )
    console.print(table)


def _job_via(job: dict) -> str:
    """Who asked for a job, when its front end said: `mcp · claude-code`."""
    by = job.get("started_by") or {}
    if not by.get("via"):
        return ""
    return " · ".join(part for part in (by["via"], by.get("client")) if part)


def _print_job(job: dict) -> None:
    console.print(f"[bold]job {job['id'][:8]}[/bold] — {job['kind']}, {_job_state(job)}")
    subject = _job_subject(job)
    if subject:
        console.print(f"  of        {escape(subject)}")
    if _job_via(job):
        console.print(f"  via       {escape(_job_via(job))}")
    for name, ids in (job.get("links") or {}).items():
        if ids:
            console.print(f"  {name:<9} {', '.join(i[:8] for i in ids)}")
    error = job.get("error")
    if error:
        hint = f" — {error['hint']}" if error.get("hint") else ""
        console.print(f"  [red]{escape(error.get('message') or 'failed')}[/red]{escape(hint)}")
    if job["status"] == "done" and job["kind"] in ("resample", "ab"):
        experiments = (job.get("links") or {}).get("experiments") or []
        if experiments:
            console.print(f"  [dim]tuningfork experiment {experiments[-1][:8]} shows the result[/dim]")


@jobs_app.command("show")
def jobs_show(
    job_id: str = typer.Argument(..., help="Job id or prefix."),
    as_json: bool = typer.Option(False, "--json", help="The job as the API gives it."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """One job: what it was asked to do, how it went, what it made."""
    job = _or_exit(lambda: _control(proxy).job(job_id))
    if as_json:
        typer.echo(json.dumps(job, indent=2, ensure_ascii=False))
        return
    _print_job(job)


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: str = typer.Argument(..., help="Job id or prefix."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Stop a job. What it finished stays stored; a harness it killed is not judged."""
    job = _or_exit(lambda: _control(proxy).cancel_job(job_id))
    if job["status"] == "running":
        console.print(f"{job['id'][:8]}: asked to stop — it ends when its current work has")
    else:
        console.print(f"{job['id'][:8]}: {job['status']}")


@jobs_app.command("wait")
def jobs_wait(
    job_id: str = typer.Argument(..., help="Job id or prefix."),
    timeout: Optional[float] = typer.Option(None, help="Give up after this many seconds."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Block until a job ends, printing its progress; exit 1 unless it finished."""
    client = _control(proxy)

    def show(entry: dict) -> None:
        event = entry["event"]
        if event["kind"] in ("experiment", "launch", "waiting"):
            detail = event["data"].get("id") or event["data"].get("launch_id") or event["data"].get("reason")
            console.print(f"  [dim]{event['kind']}[/dim] {escape(str(detail or ''))[:40]}")

    job = _or_exit(lambda: client.wait_for_job(job_id, timeout=timeout, on_event=show))
    _print_job(job)
    if job["status"] != "done":
        raise typer.Exit(code=1)


# --- connections -----------------------------------------------------------------------------


def _print_checks(checks: dict) -> None:
    results = (checks or {}).get("results") or {}
    if not results:
        console.print("  checks    [dim]not measured — `tuningfork connections check`[/dim]")
        return
    for name, found in results.items():
        tone = {"yes": "green", "no": "red"}.get(found.get("support"), "yellow")
        detail = f" — {escape(found['detail'])}" if found.get("detail") else ""
        console.print(f"  {name:<22} [{tone}]{found.get('support')}[/{tone}]{detail}")


@connections_app.command("show")
def connections_show(proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP)) -> None:
    """The server the proxy forwards to, what was measured of it, and the harnesses."""
    found = _or_exit(lambda: _control(proxy).connections())
    server = found["server"]
    identity = server.get("identity") or {}
    console.print(f"[bold]server[/bold]    {escape(server['url'])}")
    if server["url"] != server["started_with"]:
        console.print(f"  started with {escape(server['started_with'])}")
    for key in ("build", "model_file", "n_ctx", "models"):
        if identity.get(key) is not None:
            value = ", ".join(identity[key]) if isinstance(identity[key], list) else identity[key]
            console.print(f"  {key:<11} {escape(str(value))}")
    _print_checks(server.get("checks") or {})
    console.print("\n[bold]harnesses[/bold]")
    for harness in found["harnesses"]:
        where = harness["path"] or "[red]not found[/red]"
        version = harness["version"] or (f"[red]{escape(harness['error'])}[/red]" if harness["error"] else "")
        console.print(f"  {harness['name']:<12} {where} ({harness['source']}) {version}")
    sandbox = found.get("sandbox") or {}
    if sandbox.get("available"):
        console.print(
            f"\n[bold]sandbox[/bold]   bubblewrap {escape(str(sandbox.get('version') or ''))}: every launch "
            f"works in {_sandbox_seen(sandbox)}"
        )
    elif sandbox:
        console.print(
            f"\n[bold]sandbox[/bold]   [yellow]none: {escape(str(sandbox.get('reason')))}[/yellow] — each launch "
            "sees its own folders"
        )
    console.print(f"\n[bold]proxy[/bold]     {escape(found['proxy']['home'])}")
    if found.get("busy"):
        console.print(f"  in use: {escape('; '.join(found['busy']))}")
    console.print(f"  [dim]restart with: {escape(found['proxy']['restart'])}[/dim]")


@connections_app.command("switch")
def connections_switch(
    url: str = typer.Argument(..., help="The inference server to forward to, e.g. http://127.0.0.1:8081"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would change; change nothing."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Point the proxy at another inference server, while nothing is using it."""
    found = _or_exit(lambda: _control(proxy).switch_upstream(url, dry_run=dry_run))
    for change in found["differences"]:
        console.print(f"  {change['field']:<22} {escape(str(change['current']))} → {escape(str(change['proposed']))}")
    if dry_run:
        busy = found.get("busy") or []
        console.print(f"[yellow]in use: {escape('; '.join(busy))}[/yellow]" if busy else "nothing is using the proxy: it can switch")
        return
    console.print(f"now forwarding to {escape(found['url'])}")


@connections_app.command("harness")
def connections_harness(
    name: str = typer.Argument(..., help="The harness: hermes, claude-code."),
    executable: Optional[str] = typer.Option(None, help="The program to run for it, by its absolute path."),
    reset: bool = typer.Option(False, "--reset", help="Go back to the one on PATH."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Name the program a harness runs as; kept in the recordings folder, never in the
    harness's own configuration."""
    if bool(executable) == reset:
        console.print("[red]give --executable PATH, or --reset[/red]")
        raise typer.Exit(code=1)
    found = _or_exit(lambda: _control(proxy).set_harness(name, None if reset else executable))
    console.print(f"{found['name']}: {found['path'] or 'not found'} ({found['source']}) {found['version'] or found['error'] or ''}")


@connections_app.command("check")
def connections_check(
    model: Optional[str] = typer.Option(None, help="The model to check with; default the one the server lists."),
    proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP),
) -> None:
    """Measure what the harnesses need of the server: a streaming tool call, /v1/messages."""
    client = _control(proxy)
    params = {"model": model} if model else {}
    job = _or_exit(lambda: client.submit("server-check", params))
    done = _or_exit(lambda: client.wait_for_job(job["id"], timeout=600))
    if done["status"] != "done":
        _print_job(done)
        raise typer.Exit(code=1)
    _print_checks(done["result"])


proposer_app = typer.Typer(
    no_args_is_help=True,
    help="The built-in proposer: which model `refine propose --builtin` asks, and its key. "
    "Kept in the recordings folder; no proxy needed.",
)
connections_app.add_typer(proposer_app, name="proposer")


def _print_proposer(settings: dict, key: dict) -> None:
    if settings["url"] is None:
        console.print("[bold]proposer[/bold]  the proxy's own server — through the proxy for a job, gated and never recorded")
    else:
        from tuningfork.endpoints import address, is_loopback

        host, _ = address(settings["url"])
        remote = "" if is_loopback(host) else (
            " — not this machine: [green]allowed[/green]" if settings["allow_remote"]
            else " — not this machine: [yellow]not allowed yet (--allow-remote)[/yellow]"
        )
        console.print(f"[bold]proposer[/bold]  {escape(settings['url'])}{remote}")
    console.print(f"  wire         {settings['wire']}")
    console.print(f"  model        {escape(settings['model'] or 'the server default')}")
    console.print(f"  max tokens   {settings['max_tokens']}")
    console.print(f"  temperature  {settings['temperature'] if settings['temperature'] is not None else 'the server default'}")
    where = {"environment": "from TUNINGFORK_PROPOSER_KEY", "file": "in the recordings folder (.proposer-key)"}
    console.print(f"  key          {'set, ' + where.get(key.get('from'), '') if key['set'] else 'not set'}")


@proposer_app.command("show")
def proposer_show(home: Optional[Path] = typer.Option(None)) -> None:
    """The proposer's settings, and whether a key is set — never the key."""
    from tuningfork.ops import proposer

    where = home or default_home()
    _print_proposer(proposer.config(where), proposer.key_status(where))


@proposer_app.command("set")
def proposer_set(
    url: Optional[str] = typer.Option(
        None, help="The endpoint, e.g. https://api.anthropic.com; `none` for the proxy's own server."
    ),
    wire: Optional[str] = typer.Option(None, help="openai-chat or anthropic-messages."),
    model: Optional[str] = typer.Option(None, help="The model to ask for; `none` for the server's default."),
    max_tokens: Optional[int] = typer.Option(None, help="Its answer's cap: a model that reasons first needs thousands."),
    temperature: Optional[float] = typer.Option(None, help="From 0 to 2."),
    allow_remote: Optional[bool] = typer.Option(
        None, "--allow-remote/--no-allow-remote",
        help="Allow an endpoint that is not this machine: the evidence it is sent holds recorded "
        "prompts, answers and your skill.",
    ),
    key_stdin: bool = typer.Option(False, "--key-stdin", help="Read the API key from stdin: never an argument, which `ps` shows."),
    clear_key: bool = typer.Option(False, "--clear-key", help="Remove the key from the recordings folder."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Change the built-in proposer's settings, or its key."""
    import sys

    from tuningfork.ops import proposer

    where = home or default_home()
    changes: dict[str, Any] = {}
    if url is not None:
        changes["url"] = None if url.strip().lower() in ("", "none") else url.strip()
    if wire is not None:
        changes["wire"] = wire
    if model is not None:
        changes["model"] = None if model.strip().lower() in ("", "none") else model.strip()
    if max_tokens is not None:
        changes["max_tokens"] = max_tokens
    if temperature is not None:
        changes["temperature"] = temperature
    if allow_remote is not None:
        changes["allow_remote"] = allow_remote
    if key_stdin and clear_key:
        console.print("[red]--key-stdin or --clear-key, not both[/red]")
        raise typer.Exit(code=1)
    if not changes and not key_stdin and not clear_key:
        console.print("[red]nothing to change: see --help[/red]")
        raise typer.Exit(code=1)
    try:
        settings = proposer.set_config(where, **changes) if changes else proposer.config(where)
        if key_stdin:
            key = proposer.set_key(where, sys.stdin.readline())
        elif clear_key:
            key = proposer.set_key(where, None)
        else:
            key = proposer.key_status(where)
    except OperationError as error:
        _refuse(error)
    _print_proposer(settings, key)


@app.command()
def pause(proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP)) -> None:
    """Stop sending model calls upstream; they fail fast until `resume`."""
    _or_exit(lambda: _control(proxy).pause())
    console.print("paused — model calls are refused until `tuningfork resume`")


@app.command()
def resume(proxy: Optional[str] = typer.Option(None, help=_PROXY_HELP)) -> None:
    """Resume sending model calls upstream."""
    _or_exit(lambda: _control(proxy).resume())
    console.print("resumed")


@app.command()
def ui(
    home: Optional[Path] = typer.Option(None, help="Recordings to browse."),
    host: str = typer.Option("127.0.0.1", help="Bind address. Keep it local."),
    port: int = typer.Option(8788, "--port", "-p"),
) -> None:
    """Browse recordings in the web UI, with no proxy running.

    With a proxy running, use its own UI instead — it adds live breakpoints.
    """
    import uvicorn

    from tuningfork.proxy.app import create_browse_app

    store = Store(home or default_home())
    console.print(f"[bold]tuningfork[/bold] {__version__} — browsing {store.home}")
    console.print(f"  open http://{host}:{port}/_tuningfork/ui/")
    uvicorn.run(create_browse_app(store, host=host), host=host, port=port, log_level="warning")


# --- the trash ---------------------------------------------------------------------------

trash_app = typer.Typer(
    no_args_is_help=True,
    help="Set runs, experiments, jobs, scenarios and refinements aside in the trash; bring them "
    "back, or delete them for good from there.",
)
app.add_typer(trash_app, name="trash")


def _goes(counts: dict[str, int]) -> str:
    """What a permanent deletion removes, as a person reads it."""
    names = (
        ("refinements", "refinement"), ("runs", "run"), ("steps", "step"), ("experiments", "experiment"),
        ("samples", "sample"), ("token_streams", "token stream"), ("launches", "launch"), ("jobs", "job"),
        ("scenarios", "scenario"), ("variants", "saved variant"),
    )
    plural = {"launch": "launches"}
    parts = [f"{counts[key]} {name if counts[key] == 1 else plural.get(name, name + 's')}"
             for key, name in names if counts.get(key)]
    return ", ".join(parts) or "nothing"


def _trash_each(kind: str, references: list[str], home: Optional[Path]) -> None:
    from tuningfork.ops import trash as trash_ops

    store = _store(home)
    for reference in references:
        try:
            moved = trash_ops.trash(store, kind, reference)
        except OperationError as error:
            _refuse(error)
        what = f"{kind} {moved['id'][:8]}" + (f" ({moved['label']})" if moved.get("label") else "")
        if moved.get("already"):
            console.print(f"{escape(what)} is in the trash already")
        else:
            extra = len(moved["moved"]) - 1 if kind == "run" else 0
            console.print(f"moved {escape(what)} to the trash" + (f", with {extra} thread(s)" if extra > 0 else ""))
    console.print("[dim]`tuningfork trash restore <id>` brings it back; `tuningfork trash purge <id>` deletes it for good[/dim]")


@trash_app.command("run")
def trash_run(
    references: list[str] = typer.Argument(..., help="Runs, by id, label or id prefix."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Move runs to the trash, each with its threads. Experiments at their steps, and forks
    of them, stay where they are until the run is deleted for good."""
    _trash_each("run", references, home)


@trash_app.command("experiment")
def trash_experiment(
    references: list[str] = typer.Argument(..., help="Experiments, by id or id prefix."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Move experiments to the trash: an A/B's two arms together, with the forks its
    samples ran to the end."""
    _trash_each("experiment", references, home)


@trash_app.command("job")
def trash_job(
    references: list[str] = typer.Argument(..., help="Finished jobs, by id or id prefix."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Move finished jobs' records to the trash. What a job made stays."""
    _trash_each("job", references, home)


@trash_app.command("scenario")
def trash_scenario(
    names: list[str] = typer.Argument(..., help="Scenarios, by name."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Move scenarios to the trash: their folders, spec and input files."""
    _trash_each("scenario", names, home)


@trash_app.command("refinement")
def trash_refinement(
    references: list[str] = typer.Argument(..., help="Refinements, by id or id prefix."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Move refinements to the trash: out of the list, taking no new work. Deleted for good, a
    refinement goes with what it made — its screens, its validations' runs, its proposals'
    saved texts; never the run it studies, nor the changelog an approval wrote."""
    _trash_each("refinement", references, home)


@trash_app.command("list")
def trash_list(
    as_json: bool = typer.Option(False, "--json", help="The trash as the API gives it."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """What is in the trash, and what deleting each for good would remove."""
    from tuningfork.ops import trash as trash_ops

    found = trash_ops.listing(_store(home))
    if as_json:
        typer.echo(json.dumps(found.to_json(), indent=2, ensure_ascii=False))
        return
    if found.empty:
        console.print("the trash is empty")
        return
    table = Table("kind", "id", "what", "trashed", "deleting it for good removes", box=None)
    for run in found.runs:
        table.add_row("run", run["id"][:8], escape(run["label"] or run["task_id"] or "-"),
                      _when(run["trashed_at"], seconds=False), _goes(run["goes"]))
    for experiment in found.experiments:
        what = f"{experiment['kind']} at {(experiment['run_id'] or '-')[:8]} step {experiment['step_idx']}"
        table.add_row("experiment", experiment["id"][:8], escape(what),
                      _when(experiment["trashed_at"], seconds=False), _goes(experiment["goes"]))
    for job in found.jobs:
        table.add_row("job", job["id"][:8], escape(f"{job['kind']}, {job['status']}"),
                      _when(job["trashed_at"], seconds=False), "the job's record")
    for scenario in found.scenarios:
        table.add_row("scenario", escape(scenario["id"]), escape(scenario["name"]),
                      _when(scenario["trashed_at"], seconds=False), "its folder")
    for refinement in found.refinements:
        what = f"{refinement['skill']} at {refinement['run_id'][:8]} step {refinement['step_idx']}, {refinement['status']}"
        table.add_row("refinement", refinement["id"][:8], escape(what),
                      _when(refinement["trashed_at"], seconds=False), _goes(refinement["goes"]))
    console.print(table)
    _print_citations([entry["goes"] for entry in found.runs + found.experiments + found.refinements])


@trash_app.command("restore")
def trash_restore(
    references: list[str] = typer.Argument(..., help="Items in the trash, by id, prefix, label or name."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Bring items back from the trash, as they went."""
    from tuningfork.ops import trash as trash_ops

    store = _store(home)
    for reference in references:
        try:
            kind, item = trash_ops.find(store, reference)
            trash_ops.restore(store, kind, item)
        except OperationError as error:
            _refuse(error)
        console.print(f"restored {kind} {escape(item[:8] if kind != 'scenario' else item)}")


@trash_app.command("purge")
def trash_purge(
    references: list[str] = typer.Argument(..., help="Items in the trash, by id, prefix, label or name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Delete items in the trash for good, with everything tied to them: a run's experiments,
    their samples, and forks that replayed it. This cannot be undone."""
    from tuningfork.ops import trash as trash_ops

    store = _store(home)
    for reference in references:
        try:
            kind, item = trash_ops.find(store, reference)
            goes = trash_ops.preview(store, kind, item)
        except OperationError as error:
            _refuse(error)
        name = item[:8] if kind != "scenario" else item
        console.print(f"deleting {kind} {escape(name)} for good removes {_goes(goes)}")
        _print_citations([goes])
        if not yes and not typer.confirm("Delete it for good?", default=False):
            console.print("kept")
            continue
        try:
            removed = trash_ops.purge(store, kind, item)
        except OperationError as error:
            _refuse(error)
        _print_removal(removed.to_json())


@trash_app.command("empty")
def trash_empty(
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
    home: Optional[Path] = typer.Option(None),
) -> None:
    """Delete everything in the trash for good. This cannot be undone."""
    from tuningfork.ops import trash as trash_ops

    store = _store(home)
    found = trash_ops.listing(store)
    if found.empty:
        console.print("the trash is empty")
        return
    console.print(f"{found.total} item(s) in the trash")
    _print_citations([entry["goes"] for entry in found.runs + found.experiments + found.refinements])
    if not yes and not typer.confirm("Delete them all for good?", default=False):
        console.print("kept")
        return
    _print_removal(trash_ops.empty(store).to_json())


def _print_citations(goes: list[dict[str, Any]]) -> None:
    """The refinements that cite what a deletion removes, each once, and what else it takes —
    an approval's record: said before it is done."""
    seen: dict[str, str] = {}
    for each in goes:
        for cited in each.get("cited_by") or []:
            seen.setdefault(cited["refinement"], cited["says"])
        for note in each.get("notes") or []:
            seen.setdefault(note, note)
    for says in seen.values():
        console.print(f"  [yellow]{escape(says)}[/yellow]", soft_wrap=True)


def _print_removal(removed: dict[str, Any]) -> None:
    console.print(f"deleted for good: {_goes(removed['removed'])}")
    freed = removed["bytes"]
    console.print(
        f"[dim]and what nothing else referred to: {removed['message_nodes']} message(s), "
        f"{removed['blobs']} blob(s) ({freed / 1_048_576:.1f} MiB), {removed['folders']} run folder(s)[/dim]"
    )


if __name__ == "__main__":
    app()
