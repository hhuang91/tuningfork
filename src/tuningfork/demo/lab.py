"""Demo data and servers: tuningfork with no GPU and no model, as `tuningfork demo` serves them.

    tuningfork demo [lab] [--home DIR] [--port N]    # seeded runs, a model to test a wording on
    tuningfork demo seed  [--home DIR]               # recordings to look at
    tuningfork demo proxy [--home DIR] [--port N]    # the real proxy, mock model
    tuningfork demo live  [--home DIR] [--port N]    # the same, with a call held

`proxy` runs the proxy app itself in front of the scriptable mock server, so breakpoints
and held calls behave exactly as they do against a model. Its model answers every call by
writing report.json with the wrong keys. `live` is `proxy` for the browser test: shortly
after it starts, it arms a `write_file` breakpoint and makes one call that stops there, so
the Live page has something to show. `tests/browser/live_ui.mjs` drives that page.

`lab` is the proxy over the `seed` recordings, for starting work from the browser: its
model follows the skill's key names only when the skill states them — so a resample of the
write step finds the wrong keys, and an A/B of a wording that names them shows the shift.
Its raw endpoints are the mock's toy llama.cpp (`toy.py`), and one more run, `tokens-lab`,
has an answer that toy model parts on one token — read the file, or search for it — for
scoring and forking tokens from the browser. `tests/browser/jobs_ui.mjs` drives it.

`tuningfork ui --home DIR` browses seeded recordings read-only. `examples/README.md`
walks through all of them.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from tuningfork.analysis.normalize import outcome_for
from tuningfork.demo.task import SKILL, VARIANT, system_prompt, tool_call_message
from tuningfork.store.db import Store

MODEL = "gemma-4-31b"
TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    for name in ("read_file", "write_file", "terminal", "skill_view")
]
CSV = "id,amount\n1,10.00\n2,20.50\n3,15.00\n4,16.00\n"
DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "n_ctx": 65536}
GOOD = {"row_count": 4, "amount_total": 61.5}
BAD = {"count": 4, "total": 61.5}
#: What marks a folder the demo recorded into, so that it may record there again.
MARK = ".tuningfork-demo"
#: The seeded runs' task, which `seed` writes first: a folder holding it holds those runs.
SEEDED = "demo-task"


def demo_home(home: Path | None, mode: str) -> Path:
    """The folder a demo records into: a new one, or the one named if the demo may use it.
    Its runs are made up under a real task's id (`tally-report/1`), so it records only into a
    folder it made or an empty one, never where tuningfork keeps recordings by default; and
    `seed` adds its runs to whatever is there, so a folder is seeded once. A folder
    `TUNINGFORK_HOME` names is refused only for what it holds: the guide names the demo's own."""
    import tempfile

    from tuningfork.ops import OperationError

    named = home is not None
    again = "name a new folder with --home, or leave --home out for one"
    try:
        if not named:
            home = Path(tempfile.mkdtemp(prefix=f"tuningfork-{mode}-"))
        else:
            home = home.expanduser().resolve()
            if home.exists() and not home.is_dir():
                raise OperationError(f"{home} is a file, not a folder", hint=again)
            # A folder the demo seeded before it marked its folders keeps the tasks it wrote there.
            ours = (home / MARK).exists() or (home / SEEDED).is_dir() or (home / "lab-task").is_dir()
            if not ours:
                if home == (Path.home() / ".tuningfork").resolve():
                    raise OperationError(f"{home} is where tuningfork keeps your recordings by default", hint=again)
                if (home / "tuningfork.db").exists():
                    raise OperationError(f"{home} holds recordings not marked as the demo's", hint=again)
                if home.exists() and any(home.iterdir()):
                    raise OperationError(f"{home} is not empty, and not a folder the demo made", hint=again)
            if mode == "seed" and (home / SEEDED).is_dir():
                raise OperationError(f"{home} is seeded already: seeding it again would add its runs twice", hint=again)
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / MARK).write_text("Made-up recordings, made by `tuningfork demo`.\n", encoding="utf-8")
    except OSError as error:  # a folder it may not read, or cannot make
        if not named:  # a new one, in the temporary folder
            again = "name a folder with --home"
        where = home or error.filename or "a new folder in the temporary folder"
        raise OperationError(f"cannot record in {where}: {error.strerror or error}", hint=again) from None
    return home


def check_port(port: int) -> None:
    """Refuse a port something listens on already, before anything is seeded or printed: the
    address the demo prints would lead to whatever holds it."""
    import socket

    from tuningfork.ops import OperationError

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn binds
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            taken = error.errno == errno.EADDRINUSE
            said = f"port {port} is in use" if taken else f"cannot listen on port {port}: {error.strerror}"
            raise OperationError(said, hint="name another with --port") from None


def response(message, finish="stop", reasoning="", prompt=4100, completion=120):
    if reasoning:
        message = {**message, "reasoning_content": reasoning}
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


#: What the lab's built-in proposer answers: one edit that names the keys, one that does not,
#: and one whose `old` the skill does not have — kept, kept, and refused.
PROPOSALS = [
    {"edits": [{"old": "Write the totals to report.json.",
                "new": "Write the totals to report.json, using the keys row_count and amount_total."}],
     "rationale": "Every answer wrote count and total; the check reads row_count."},
    {"edits": [{"old": "Write the totals to report.json.", "new": "Write the totals to report.json, compactly."}],
     "rationale": "A shorter report may be read more reliably."},
    {"edits": [{"old": "Write the report as YAML.", "new": "Write the report as JSON."}],
     "rationale": "An edit of a line the skill does not have."},
]


def write(keys):
    arguments = {"path": "report.json", "content": json.dumps(keys, indent=2)}
    return tool_call_message("write_file", arguments, call_id="c2")


#: The demo task's check: a report without the keys fails it with the KeyError its runs print.
DEMO_CHECK = (
    "python3 -c \"import json; d = json.load(open('report.json'));"
    " assert d['row_count'] == 4 and d['amount_total'] == 61.5\""
)


def demo_spec(home: Path) -> Path:
    """The task the seeded runs were launched from, as a batch keeps its spec: `Run this task
    again…` on their page opens it, for the lab's own harness (`fake`) to run."""
    folder = home / SEEDED
    (folder / "tally-report").mkdir(parents=True, exist_ok=True)
    (folder / "tally-report" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    path = folder / "evals.json"
    path.write_text(json.dumps({
        "skill_name": "tally-report",
        "tuningfork": {"skill_dir": "tally-report"},
        "evals": [{
            "id": "1",
            "prompt": "Use the tally-report skill on data.csv.",
            "files": [{"path": "data.csv", "content": CSV}],
            "tuningfork": {"check": DEMO_CHECK, "skills": ["tally-report"], "max_turns": 6, "timeout": 600},
        }],
    }, indent=2), encoding="utf-8")
    return path


def keep_skill(store: Store) -> Path:
    """The skill as the seeded runs were given it, kept where a batch keeps its copy: what
    an A/B's original names, the Skills page reads, and the browser's variant editor starts
    from."""
    snapshot = store.home / "batches" / "demo" / "skill-snapshot" / "tally-report"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "SKILL.md").write_text(SKILL, encoding="utf-8")
    for run in store.runs(limit=100, task_id="tally-report/1"):
        store.annotate_run(run.id, parity={**run.parity, "skill_snapshot_dir": str(snapshot)})
    return snapshot / "SKILL.md"


def seed(home: Path, hint: bool = True) -> None:
    """Nine runs of a flaky task that diverge at step 1, each launched by a batch of it as
    Hermes Agent: one edited at a breakpoint and one whose skill changed mid-run, neither
    counted; a run with no task, recorded live; a resample and an A/B of a saved variant, its
    arms sent the two wordings."""
    from tuningfork.experiment.skill import substitute, text_sha256
    from tuningfork.launches import Assignment
    from tuningfork.ops.variants import save_variant
    from tuningfork.runner import eval_record
    from tuningfork.specs import load_spec

    store = Store(home)
    spec = load_spec(demo_spec(home))
    skill_hash = hashlib.sha256(SKILL.encode()).hexdigest()
    base = {"model": MODEL, "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True}}
    run_ids = []
    for index, good_keys in enumerate([True, True, False, True, False, True, True, False, True]):
        label = f"tally-report/1#demo.{index}"
        # Launched as a batch launches a run (found in review: since runs nothing launched read
        # "recorded live", every seeded run did, though each carries its check's verdict).
        launch = store.create_launch(
            kind="batch", label=label, harness="hermes", model=MODEL, task_id="tally-report/1",
            eval=eval_record(spec.get("1"), spec.directory), spec_path=str(spec.path),
            skill_dir=str(spec.directory / "tally-report"),
        )
        run_id = uuid.uuid4().hex

        def at(idx: int, run_id: str = run_id, launch_id: str = launch.id, label: str = label) -> Assignment:
            return Assignment(launch_id=launch_id, run_id=run_id, idx=idx, thread=0, label=label, task_id="tally-report/1")

        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": "Use the tally-report skill on data.csv."},
        ]
        read = tool_call_message("read_file", {"path": "data.csv"}, call_id="c1")
        store.record_step(
            request={**base, "messages": list(messages)},
            response=response(read, "tool_calls", "I should read data.csv first.", 4100, 45),
            timings={"ttft_ms": 1450.0, "total_ms": 2210.5},
            server_defaults=DEFAULTS,
            assignment=at(0),
        )
        messages += [read, {"role": "tool", "tool_call_id": "c1", "content": CSV}]
        thought = "Four rows; 10 + 20.5 + 15 + 16 = 61.5. " + (
            "I'll name the keys row_count and amount_total." if good_keys else "Keys: count, total."
        )
        written = write(GOOD if good_keys else BAD)
        edited = index == 7  # this run's report was fixed at a breakpoint
        store.record_step(
            request={**base, "messages": list(messages)},
            response=response(written, "tool_calls", thought, 4180, 160 + index),
            timings={"ttft_ms": 1700.0 + index * 40, "total_ms": 3900.0 + index * 90},
            server_defaults=DEFAULTS,
            delivered_response=response(write(GOOD), "tool_calls", thought) if edited else None,
            edits={"response": {"fields": ["choices"], "message_fields": ["tool_calls"]}}
            if edited
            else None,
            assignment=at(1),
        )
        messages += [written, {"role": "tool", "tool_call_id": "c2", "content": '{"success": true}'}]
        done = {"role": "assistant", "content": "Done — the totals are in report.json."}
        store.record_step(
            request={**base, "messages": list(messages)},
            response=response(done, "stop", "", 4260, 14),
            timings={"ttft_ms": 900.0, "total_ms": 1150.0},
            assignment=at(2),
        )
        store.end_launch(launch.id, exit={"returncode": 0})
        run = store.get_run(run_id)
        run_ids.append(run.id)
        passed = good_keys or edited
        skills_changed = ["tally-report/SKILL.md"] if index == 5 else []
        edited_steps = [1] if edited else []
        store.annotate_run(
            run.id,
            harness="hermes",
            skill_hashes={"tally-report/SKILL.md": skill_hash},
            result={
                "passed": passed,
                "check_output": "" if passed else "KeyError: 'row_count'\n",
                "returncode": 0,
                "duration_s": 41.2 + index,
                "timed_out": False,
                "skills_changed": skills_changed,
                "edited_steps": edited_steps,
                "usable": not skills_changed and not edited_steps,
            },
        )

    # A run with no task, and square brackets in the text: Rich and HTML both parse them.
    question = {"role": "user", "content": "What is 2 + 2? Answer with [a number] only."}
    store.record_step(
        request={"model": MODEL, "messages": [question]},
        response=response({"role": "assistant", "content": "[4]"}, "stop", "Arithmetic.", 30, 8),
        timings={"ttft_ms": 300.0, "total_ms": 420.0},
    )

    original = keep_skill(store)
    checkpoint = store.step(run_ids[0], 1)
    request = store.rebuild_request(checkpoint)

    def add_samples(experiment, good_share, n=20):
        for sample in range(n):
            answer = response(write(GOOD if sample < round(good_share * n) else BAD), "tool_calls")
            found = outcome_for(answer)
            store.add_sample(
                experiment.id, sample, seed=1000 + sample, response=answer,
                outcome_kind=found.kind, outcome_key=found.key, outcome_label=found.label,
                finish_reason="tool_calls", usage=answer["usage"], timings={"total_ms": 3800.0},
            )

    settings = {"n": 20, "seed": 1000, "normalizers": [], "concurrency": 1}
    resample = store.create_experiment(
        kind="resample", arm="original", run_id=run_ids[0], step_idx=1, step_id=checkpoint.id,
        request=request, settings=settings, notes=["server defaults match the recording"],
    )
    add_samples(resample, 0.6)
    group = uuid.uuid4().hex
    target = {"kind": "pattern", "value": "row_count", "description": "writes the row_count key"}
    # The variant saved as the browser saves one, and swapped in as an A/B swaps it: the
    # variant arm keeps the request it would have sent.
    variant_request, spans = substitute(request, SKILL, VARIANT)
    variant = {
        "original_path": str(original),
        "variant_path": str(save_variant(home, "tally-report", "name-the-keys", VARIANT)),
        "original_sha256": text_sha256(SKILL),
        "variant_sha256": text_sha256(VARIANT),
        "spans": [asdict(span) for span in spans],
        "unapplied": [],
    }
    for arm, share, sent in (("original", 0.6, request), ("variant", 0.95, variant_request)):
        experiment = store.create_experiment(
            kind="ab", arm=arm, run_id=run_ids[0], step_idx=1, step_id=checkpoint.id,
            request=sent, settings={**settings, "target": target, "alpha": 0.05}, variant=variant,
            group_id=group,
        )
        add_samples(experiment, share)
    # Drawn in full: done, not left running by a process that is about to exit.
    store.finish_experiments([e.id for e in store.experiments(limit=10)], "done")
    store.close()
    # The lab serves them next: what to try is the web UI it prints, with no command to type.
    print(f"seeded {home}: 10 runs, a resample and an A/B." + (f" Try: tuningfork runs --home {home}" if hint else ""))


def proxy(home: Path, port: int, demo_call: bool = False, responder=None, blip: bool = False, setup=None) -> None:
    import httpx

    from tuningfork.config import Settings
    from tuningfork.demo.mock import MockServer, MockTurn
    from tuningfork.proxy.app import ProxyServer, create_app
    from tuningfork.proxy.recorder import Recorder

    mock = MockServer()
    mock.props = {"default_generation_settings": {"n_ctx": DEFAULTS["n_ctx"], "params": {
        key: value for key, value in DEFAULTS.items() if key != "n_ctx"
    }}}
    if setup is not None:
        setup(mock)

    def respond(payload):
        # A plain answer for the browser test's step-through call ("e2e-marker");
        # otherwise the write_file the demo breakpoint stops.
        if "e2e-marker" in json.dumps(payload.get("messages")):
            return MockTurn(content="ok")
        content = json.dumps(BAD, indent=2)
        return MockTurn(
            reasoning="The CSV has four rows; I will write the totals.",
            tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": content}}],
        )

    mock.responder = responder or respond
    settings = Settings(
        upstream_base_url="http://mock.invalid", home=home, port=port,
        token=os.environ.get("TUNINGFORK_TOKEN") or None,
    )
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock.app), timeout=600)
    app = create_app(settings, client=upstream, recorder=Recorder(settings))
    token = app.state.access.token

    base = f"http://127.0.0.1:{port}"
    print(f"tuningfork proxy on {base}/v1, in front of a mock model (no GPU)", flush=True)
    print(f"  recording to {home}", flush=True)
    print(f"  web UI       {base}/_tuningfork/ui/?token={token}", flush=True)
    print(f"  control it with TUNINGFORK_PROXY={base} — Ctrl-C to stop", flush=True)
    if demo_call:
        threading.Thread(target=_demo_call, args=(port, token), daemon=True).start()
    if blip:
        from starlette.routing import Route

        # Demo only: end every open event stream, as the bus does to a page that fell a whole
        # queue behind. The page reconnects, and must catch up on what it missed meanwhile.
        # Under tuningfork's own prefix, where the guard asks the access token of any change.
        async def cut_streams(request):
            from starlette.responses import JSONResponse

            from tuningfork.breakpoints.events import OVERFLOWED, _drain

            bus = request.app.state.bus
            cut = 0
            for queue in list(bus._subscribers):
                bus._subscribers.discard(queue)
                _drain(queue)
                queue.put_nowait(OVERFLOWED)
                cut += 1
            return JSONResponse({"cut": cut})

        app.router.routes.insert(0, Route("/_tuningfork/api/demo/blip", cut_streams, methods=["POST"]))
    ProxyServer(app, host="127.0.0.1", port=port).run()


def lab(home: Path, port: int) -> None:
    """The proxy over seeded recordings, with a model worth testing a wording on — and a
    harness it can launch: `agent.py`, as "fake", and a task for it in
    <home>/lab-task/evals.json."""
    from tuningfork.demo.mock import MockTurn
    from tuningfork.demo.task import CHECK, FakeAgent, tally_model
    from tuningfork.harness import ADAPTERS

    ADAPTERS["fake"] = FakeAgent
    task = home / "lab-task"
    (task / "tally-report").mkdir(parents=True, exist_ok=True)
    (task / "tally-report" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (task / "evals.json").write_text(json.dumps({
        "skill_name": "tally-lab",
        "tuningfork": {"skill_dir": "tally-report"},
        "evals": [{
            "id": "1",
            "prompt": "Summarize data.csv into report.json.",
            "files": [{"path": "data.csv", "content": "id,amount\n1,10\n2,20\n3,30\n4,1.5\n"}],
            "tuningfork": {"check": CHECK, "skills": ["tally-report"], "max_turns": 6, "timeout": 600},
        }, {
            # Held out: it judges an edit, and no agent, plan or job shows its text.
            "id": "2",
            "prompt": "Summarize data.csv into report.json. HELD-OUT-LAB",
            "files": [{"path": "data.csv", "content": "id,amount\n1,10\n2,20\n3,30\n4,1.5\n"}],
            "tuningfork": {"check": CHECK, "skills": ["tally-report"], "max_turns": 6, "timeout": 600, "held_out": True},
        }],
    }, indent=2), encoding="utf-8")

    if not (home / SEEDED).is_dir():
        seed(home, hint=False)
    # Kept again for a home seeded before the seed kept it.
    store = Store(home)
    keep_skill(store)
    store.close()

    def wording_sensitive(payload):
        system = payload["messages"][0].get("content") if payload.get("messages") else ""
        if isinstance(system, str) and "Working directory:" in system:
            return tally_model(payload)  # the fake harness, working through its task
        if isinstance(system, str) and system.startswith("You improve an agent skill"):
            return MockTurn(content=json.dumps({"proposals": PROPOSALS}))  # the built-in proposer
        last = payload["messages"][-1] if payload.get("messages") else {}
        # A person's question at a held call (the console), not a probe's.
        if last.get("role") == "user" and str(last.get("content", "")).startswith("Why"):
            return MockTurn(content="Because the skill never names the keys, so I used the ones I know.")
        named = isinstance(system, str) and "row_count and amount_total" in system
        keys = GOOD if named else BAD
        # Slow enough to watch a job go, and to cancel one.
        return MockTurn(
            reasoning="Four rows; the totals go in report.json.",
            tool_calls=[{"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys, indent=2)}}],
            stall=0.15,
        )

    seed_tokens(home)

    def toy(mock):
        from tuningfork.demo.toy import props

        mock.props = {**mock.props, **{k: v for k, v in props().items() if k.startswith("chat_template")}}
        mock.toy.answers = TOKEN_ANSWERS

    proxy(home, port, responder=wording_sensitive, blip=True, setup=toy)


# --- tokens: an answer the toy model parts on one token --------------------------------------

TOKEN_REQUEST = {
    "model": MODEL,
    "tools": [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
              for name in ("read_file", "search_files")],
    "max_tokens": 4096,
    "messages": [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Summarize data.csv into report.json."},
    ],
}
_THOUGHT = "<|channel>thought\nLook at it.<channel|>"
TOKEN_ANSWERS = [
    (_THOUGHT + '<|tool_call>call:read_file{path:<|"|>data.csv<|"|>}<tool_call|>', 0.6),
    (_THOUGHT + '<|tool_call>call:search_files{pattern:<|"|>data.csv<|"|>}<tool_call|>', 0.4),
]


def seed_tokens(home: Path) -> None:
    """The `tokens-lab` run: one step whose answer is the toy model's likelier one, with the
    token counts the toy server reports for it — so scoring it rebuilds it exactly."""
    from tuningfork.backends.formats import get_format
    from tuningfork.demo.task import as_response
    from tuningfork.demo.toy import ToyLlama

    store = Store(home)
    try:
        if store.get_run("tokens-lab") is not None:
            return
        toy = ToyLlama()
        raw = TOKEN_ANSWERS[0][0]
        message = get_format("gemma4").parse(raw, tools=True)
        usage = {"prompt_tokens": toy.input_tokens(TOKEN_REQUEST), "completion_tokens": len(toy.tokenize(raw)) + 1}
        store.begin_run("tokens-lab", "tokens/1")
        store.record_step(request=TOKEN_REQUEST, response={**as_response(message, "tool_calls"), "usage": usage})
    finally:
        store.close()


def _demo_call(port: int, token: str) -> None:
    """Arm a breakpoint and make one call that stops at it, for the browser test."""
    import httpx

    time.sleep(1.5)
    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=600, headers={"x-tuningfork-token": token}
    ) as client:
        client.post("/_tuningfork/api/breakpoints", json={
            "phase": "response", "tool": "write_file", "path_glob": "*.json",
            "label": "before report.json is written",
        })
        client.post("/_tuningfork/api/breakpoints", json={
            "phase": "request", "context_regex": "never-matches", "enabled": False,
        })
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": "Use the tally-report skill on data.csv."},
            tool_call_message("read_file", {"path": "data.csv"}, call_id="c1"),
            {"role": "tool", "tool_call_id": "c1", "content": CSV},
        ]
        client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages, "tools": TOOLS})
