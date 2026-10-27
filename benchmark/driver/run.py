#!/usr/bin/env python3
"""The benchmark's driver: Hermes improving its own skills, with and without tuningfork.

Runs inside the distro with tuningfork's venv (`<venv>/bin/python run.py …`). Four recording
proxies, each with a home of its own under the benchmark folder, so what an arm may read is
what its home holds:

    home        A  the baseline runs of every skill, and the experiments T's agent starts; its
                   proxy serves tuningfork's refine tools over HTTP (`--mcp-tools refine-to-end`: the loop, and screens whose samples are forks run to the end)
    agent-home  B  the refining agent's own model calls (arm T); the agent runs sandboxed like
                   every launch, its tools reached over home A's proxy, not a child process; each
                   call held to AGENT_MAX_OUTPUT tokens (`--max-output`)
    h-home      C  arm H's chains
    eval-home   D  the evaluation runs of every arm's final skill, held-out tasks included

Every launch, the agent's included, sees the benchmark folder and the bundles it was synced from
empty (`TUNINGFORK_SANDBOX_HIDE`), the forks home A's proxy starts as well: no spec, key, check,
recording or final skill is readable from inside a run. An agent given the file system was seen to
find a task's check script; the sandbox shows the rest of the machine read-only, so a folder not
named here — a clone of the bundles elsewhere, notes beside them — is for the person to add.

Stages, each resumable from `state.json` (`--again` with agent or chain forgets that arm's record
and cells first, keeping them under `earlier`, so one arm runs again on the same root):

    sync                 copy the task bundles into the benchmark folder
    proxies start|stop   the four proxies (llama-server must already be up)
    baseline SKILL       10 runs of the training task with the original skill; qualifies at 75% or
                         below: flaky from 40%, broken under it
    score SKILL          every step of the baseline runs scored on the server, for the model's own
                         chance of each tool at a step (the agent stage does this first)
    agent SKILL          arm T: Hermes with tuningfork's refine tools, launched again while its budget
                         lasts and nothing has survived; the fixed rule picks the edit
    choose SKILL         arm T's rule applied again once every screen has finished — for one the
                         agent left running as its budget ended; T's cells are forgotten if it changes
    chain SKILL          arm H: a told chain with T's budget of model-call seconds
    evaluate SKILL       O, H and T (--arms names fewer): 10 runs on the training task, 20 on each
                         held-out task; a cell short of its count is topped up; a timeout counts as
                         a failure; T is left out when it kept the original (its cells are O's)
    rescore SKILL        every evaluation cell counted again from the recordings, its runs' checks
                         run again with the spec as it is now: after a check or a rule is fixed
    report               every skill's rates with intervals, the deltas, the bootstrap over skills

Every decision of the protocol is a constant below, so a reader finds it in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: The subject: the model the server serves under this alias. Gemma 4 26B-A4B for the first
#: series (two slots, then one); Gemma 4 31B from 2026-10-10 — one card, one 128k slot, the
#: ring cache with `--ctx-checkpoints 0`, `--no-cache-prompt`, a q4_0 cache (findings 31.29).
MODEL = "gemma-4-31b"
UPSTREAM = "http://127.0.0.1:8081"
PORTS = {"home": 8801, "agent-home": 8802, "h-home": 8803, "eval-home": 8804}
BASELINE_RUNS = 10
#: Runs per arm per task in the evaluation: the training task, and each held-out task, whose
#: pooled rate is the primary measure. A cell short of its count is topped up.
EVAL_RUNS = 10
EVAL_RUNS_HELD_OUT = 20
#: Runs, samples and forks at once, and each proxy's --max-inflight: one. The server runs one
#: slot (`-np 1`): with two, a request computed beside another got different logprobs for the
#: identical prompt (llama.cpp b11108, Gemma 4 26B-A4B), where alone the server is deterministic.
#: One slot halves the throughput and removes it; a queued request is computed alone in its turn.
CONCURRENCY = 1
#: A skill qualifies when its original passes its training task at most CEILING of the time, over at
#: least MIN_COUNTED runs that ended with a verdict (a timeout is not counted, and does not disqualify):
#: above it the model already does the task, and there is nothing to measure. From FLAKY_FLOOR up a
#: skill is *flaky*, below it *broken*. The floor was a gate until 2026-10-10, when the user made it a
#: category: whether a loop improves a flaky skill without harm, and whether it finds a broken skill's
#: fix at all, are two questions, and the report answers them apart (findings 31.33).
CEILING = 0.75
FLAKY_FLOOR = 0.4
MIN_COUNTED = 8
QUALIFYING = ("flaky", "broken")
#: A held-out task is revised before a skill's arms run when the original passes it outside this
#: band over ten runs (`tuningfork batch --eval … --include-held-out`, by hand, on a pilot home):
#: at the floor or the ceiling no edit can show. The revision, decided by the original's rate
#: alone, is recorded in the notebook.
HELD_OUT_BAND = (0.15, 0.85)
#: Hermes's loop in every studied run: off. Arm H turns it on, with the review awaited and due
#: after every turn, and the check's result told.
CONTROLLED = ["-o", "self_improvement=false"]
#: The instant every launch's clock starts at (`TUNINGFORK_CLOCK`): the model reads the date in
#: its system prompt, and at a knife edge the date decides; pinned, a run's date is the same
#: whichever night it is made, and a fork's is its source's.
CLOCK = "2026-10-08 09:00:00"
H_OPTIONS = ["-o", "await_review=true", "-o", "review_every_turn=true"]
#: The most links a chain runs; the budget usually stops it first.
H_MAX_LINKS = 30
#: What the refining agent is told, the same for every skill. `{hours}` and `{turns}` are
#: AGENT_TIMEOUT and AGENT_MAX_TURNS, so the budget it is told is the budget it has.
AGENT_INSTRUCTION = (
    "Improve the skill `{skill}` for the team. tuningfork has recorded runs of its task `{task}`: {record}. "
    "Use tuningfork's tools as you see fit to find what the failed "
    "runs did wrong and to refine the skill's wording so that it goes right: a refinement with a target, "
    "proposed edits, and screens that test them. Begin with tally_failures on the task: it counts what every "
    "failed run's check said, and the most common failure is the one to fix first, not the first one you "
    "read. Start the refinement at the first step where the failed runs went wrong — the call that made what "
    "the check complains about, often not the last call: a screen replays everything before its step as "
    "recorded, the files included, so an edit can change only what happens from there on. When "
    "tally_failures says every run made the same mistake, the passing ones too, no passing run shows the "
    "fix: target the task's check from the step that made it (target_check). At a passing run's step the "
    "original already does what the target asks, and a screen there can show no gain. The task is one "
    "example of the skill's use: the skill is "
    "used on other inputs, so an edit must hold for them too, and one that names something in this task's "
    "input (an item, a file name, a date) will not. Prefer a change to how a step is done over telling the "
    "model to be careful: instructions to be careful have not moved this model; what has is making it list "
    "or count before it writes, splitting one write into two, or making it check its own output against "
    "the input. A proposal is at most three edits changing at most ten lines, counted against the original "
    "skill. A screen that does not survive is not a reason to stop: you have about {hours} hours and "
    "{turns} turns for every attempt together. If its proposal scored above the screen's original arm, it "
    "helped: build on it — propose its edits again with one more change aimed at what still fails "
    "(tally_failures with the screen's id shows what each arm's forks still fail). If it did not help, rest "
    "the next proposal on something new, another mechanism or another step, not a rewording of the last. A "
    "screen at a step takes a few minutes and only tries a wording; a screen to the end runs the task from the "
    "step, is judged by the task's check, and takes one to two hours: the refinement's first draws forty forks, "
    "the original's twenty among them, and its later ones reuse the original's, so they draw twenty. A quick "
    "look (start_screen_to_end with quick: true) draws eight forks of a proposal, about twenty-five minutes, "
    "and gives its rate without a verdict: try edits with quick looks, then screen the best in full — only a "
    "full screen to the end can keep a proposal. Where no run of the task passed, a screen's passing forks are "
    "the comparison there is: where_it_holds with the screen's id asks a question of its forks, and show_files "
    "reads a fork's files, page by page with offset. Wait for a screen with job_status (wait_seconds "
    "{max_wait}), which says how long is left, and never start the same screen again. Never edit skill files "
    "yourself: a proposal that scores above the original in a full screen to the end is the result. When the "
    "budget is spent or nothing else is worth trying, finish by naming the id of the proposal you would keep, "
    "or say that none survived.\n\n{start}"
)
#: Turns the agent may take: each poll of a job is one, and a 20-sample screen takes several.
AGENT_MAX_TURNS = 150
AGENT_TIMEOUT = 7200
#: The budget is for every attempt together. When the agent stops without a survivor — one screen
#: failed and it answered in text, which ends Hermes's run — and enough remains, it is launched again
#: with what it tried and what is left, as H's chain gets the check's verdict as its next turn. An
#: attempt needs this much time and this many turns to be worth starting; the attempts are capped.
#: How T's final skill is chosen from what its screens measured. "survives": the proposal with the
#: highest rate among those significantly better than their screen's original arm (Holm), else the
#: original — the rule until 2026-10-10. "not worse": the proposal with the highest rate among those
#: that scored above their screen's original arm and were not significantly worse, else the original —
#: the user's rule from 2026-10-10 (findings 31.35): at twenty forks an arm a thirty-point gain is not
#: significant, so "survives" discarded the best edit T had made (14/20 against 8/20, p 0.11), and a
#: bar of "not worse" is already one Hermes's own loop does not have. The held-out evaluation judges.
#: A significant survivor still ends T's search early; a proposal that is only better does not.
KEEP_RULE = "not worse"
AGENT_MAX_ATTEMPTS = 8
AGENT_ATTEMPT_MINIMUM = 15 * 60
AGENT_ATTEMPT_MINIMUM_TURNS = 15
#: A screen the agent started and did not wait for finishes in the proxy on its own; the next
#: attempt is told its verdict, not its progress, so the stage waits for it first — this long at most.
AGENT_SCREEN_WAIT = 25 * 60
AGENT_CONTINUATION = (
    "You are continuing your own work on the skill `{skill}` for the team, on tuningfork's recorded runs of its "
    "task `{task}`: the same job, not a new one. You have spent {spent_minutes} minutes and {turns} turns of "
    "about {hours} hours and {max_turns}, and no proposal has survived a screen yet. What you tried:\n{tried}\n"
    "About {remaining_minutes} minutes and {remaining_turns} turns remain. A screen that did not survive is "
    "not a reason to stop. A proposal marked above the original helped: build on it — its edits again with "
    "one more change aimed at what still fails (tally_failures with its screen's id shows what each arm's "
    "forks still fail). One whose screen has no verdict yet is neither: wait for its job first. One that "
    "did not help calls for another mechanism or another step, not a rewording "
    "of the last — making the model list or count before it writes, splitting one write into two, or making "
    "it check its own output against the input have moved this model; telling it to be careful has not. The task is one example of the skill's use, so an edit must hold for other "
    "inputs, and one that names something in this task's input will not. Use tuningfork's tools as before: "
    "show_refinement for the evidence and the proposals tried (if every screen so far put the original near "
    "100%, the checkpoint is a passing run's: start a refinement at a failed run's step), propose_edit for a new "
    "edit (at most three edits changing at most ten lines, counted against the original skill), then a quick "
    "look or a screen (start_screen_to_end, quick: true for a look; only a full screen can keep a proposal), "
    "waited for with job_status (wait_seconds {max_wait}); "
    "never start the same screen again, and never edit skill files yourself. When the budget is spent or "
    "nothing else is worth trying, finish by naming the id of the proposal you would keep, or say that none "
    "survived.\n\n{start}"
)
#: What T starts with, as H's own loop has it when it reviews a run (findings §31.57): the task as the
#: model was given it, with its criteria, and one failed run whole — every message but the harness's
#: system prompt, the skill and the files the run read among them, its calls with their arguments —
#: and the verdict as H's told turn says it. A message longer than START_MESSAGE, or past
#: START_TOTAL in all, is cut where it says how to read on (show_step with message and offset).
AGENT_START = (
    "What you start with — what Hermes's own improvement has when it looks back on a run: the task, and one "
    "failed run of it whole.\n\n"
    "The task's prompt, as the model was given it:\n{prompt}\n\n"
    "What the task expects (its criteria; the check itself is not shown):\n{criteria}\n\n"
    "Run {run} ({label}), every message it was sent but the harness's system prompt, then its answers — its "
    "calls with their arguments as sent:\n{transcript}\n\n"
    "Its verdict, as it was told: I ran my check on what you produced. It failed.{output}"
)
START_MESSAGE = 30_000
START_TOTAL = 90_000
#: How long one of the agent's MCP calls may wait for a job: under Hermes's 300 s tool timeout, so a
#: twenty-minute screen is a dozen polls (home A's proxy, `--mcp-max-wait`).
MCP_MAX_WAIT = 110
#: The most output tokens one of the agent's own model calls may have (agent-home's proxy,
#: `--max-output`). Hermes asks for 65,536, and twice a reply looped to all of it — 30 and 35 minutes
#: of one GPU (findings 31.58) — while T's 402 replies that ended normally were at most 5,781. Hermes
#: answers a cut reply by asking again with a larger limit; the proxy holds every call, so a loop costs
#: at most four or five cuts before Hermes ends the session and the next attempt starts afresh.
AGENT_MAX_OUTPUT = 8192
#: A run that timed out is a failure in the evaluation: every one seen so far was the model looping
#: inside a call — its stream cut off at the limit while it repeated one token — not the machine.
#: It counts in the cell's total and not in its passes. The baseline's qualification keeps its own
#: rule above (a timeout not counted), so a skill is not disqualified by the model's loops.
TIMEOUT_IS_FAILURE = True
#: Before the agent starts, every step of the baseline runs is scored on the server (`tuningfork
#: tokens --score`), so step_behaviours can say the model's own chance of each tool at a step: one
#: request a step, charged to T's spend, which H's budget then matches.
SCORE_BASELINE = True


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Bench:
    def __init__(self, root: Path, bundles: Path, venv: Path) -> None:
        self.root = root
        self.bundles = bundles
        self.tuningfork = venv / "bin" / "tuningfork"
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}

    # --- bookkeeping ---------------------------------------------------------------------------

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def skill_state(self, skill: str) -> dict:
        return self.state.setdefault("skills", {}).setdefault(skill, {})

    def home(self, name: str) -> Path:
        return self.root / name

    def proxy_url(self, name: str) -> str:
        return f"http://127.0.0.1:{PORTS[name]}"

    def spec(self, skill: str) -> Path:
        return self.root / "specs" / skill / "evals.json"

    def spec_data(self, skill: str) -> dict:
        return json.loads(self.spec(skill).read_text(encoding="utf-8"))

    def training(self, skill: str) -> dict:
        return next(e for e in self.spec_data(skill)["evals"] if not (e.get("tuningfork") or {}).get("held_out"))

    def held_out(self, skill: str) -> list[dict]:
        return [e for e in self.spec_data(skill)["evals"] if (e.get("tuningfork") or {}).get("held_out")]

    def original_skill(self, skill: str) -> Path:
        return self.root / "specs" / skill / self.spec_data(skill)["tuningfork"]["skill_dir"]

    def hidden(self) -> str:
        """What no launch may read: the person's own list, the benchmark folder and the bundles —
        every spec, key and check is in one or the other."""
        return os.pathsep.join(filter(None, [os.environ.get("TUNINGFORK_SANDBOX_HIDE"), str(self.root), str(self.bundles)]))

    def final(self, skill: str, arm: str) -> Path:
        return self.root / "final" / skill / arm / self.original_skill(skill).name

    def tf(self, home: str, *args: str, env: dict | None = None, cwd: Path | None = None, log_to: Path | None = None) -> str:
        """`tuningfork …` against one home, through its proxy; the output kept in a log."""
        command = [str(self.tuningfork), *args, "--home", str(self.home(home))]
        environment = {
            **os.environ, "TUNINGFORK_PROXY": self.proxy_url(home), "TUNINGFORK_SANDBOX_HIDE": self.hidden(),
            "TUNINGFORK_CLOCK": CLOCK, **(env or {}),
        }
        log(" ".join(command[:3]) + (f" … ({cwd})" if cwd else ""))
        done = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
        output = done.stdout + done.stderr
        if log_to is not None:
            log_to.parent.mkdir(parents=True, exist_ok=True)
            log_to.write_text(output, encoding="utf-8")
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(command[:3])} failed ({done.returncode}):\n{output[-3000:]}")
        return output

    # --- stages -------------------------------------------------------------------------------

    def sync(self, skills: list[str]) -> None:
        for skill in skills:
            target = self.root / "specs" / skill
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self.bundles / skill, target)
            log(f"synced {skill}")

    def proxies(self, action: str) -> None:
        """`start` runs the four proxies and waits on them — under WSL a process left behind
        by a short-lived command does not survive it, so this stays in the foreground, as a
        background task of whoever started it; `stop` ends them by the pids it kept."""
        pids = self.root / "proxies.json"
        running = json.loads(pids.read_text()) if pids.is_file() else {}
        if action == "stop":
            for name, pid in running.items():
                try:
                    os.kill(pid, 15)
                    log(f"stopped {name} (pid {pid})")
                except ProcessLookupError:
                    pass
            pids.unlink(missing_ok=True)
            return
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        processes = []
        for name, port in PORTS.items():
            self.home(name).mkdir(parents=True, exist_ok=True)
            logfile = (self.root / "logs" / f"serve-{name}.log").open("a")
            process = subprocess.Popen(
                [str(self.tuningfork), "serve", "--upstream", UPSTREAM, "--home", str(self.home(name)),
                 "--port", str(port), "--max-inflight", str(CONCURRENCY),
                 *(["--mcp-tools", "refine-to-end", "--mcp-results", "text", "--mcp-max-wait", str(MCP_MAX_WAIT)]
                   if name == "home" else []),
                 *(["--max-output", str(AGENT_MAX_OUTPUT)] if name == "agent-home" else [])],
                stdout=logfile, stderr=subprocess.STDOUT,
                env={**os.environ, "TUNINGFORK_CLOCK": CLOCK, "TUNINGFORK_SANDBOX_HIDE": self.hidden()},
            )
            running[name] = process.pid
            processes.append(process)
            log(f"started {name} on :{port} (pid {process.pid})")
        pids.write_text(json.dumps(running))
        try:
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
        finally:
            pids.unlink(missing_ok=True)

    def baseline(self, skill: str) -> None:
        state = self.skill_state(skill)
        if state.get("baseline"):
            log(f"{skill}: baseline done already ({state['baseline']['passes']}/{state['baseline']['total']})")
            return
        before = self.newest_run_time("home")
        output = self.tf(
            "home", "batch", "evals.json", "-n", str(BASELINE_RUNS), "--concurrency", str(CONCURRENCY),
            "--model", MODEL, *CONTROLLED, cwd=self.spec(skill).parent, log_to=self.root / "logs" / f"baseline-{skill}.log",
        )
        runs = self.runs_since("home", f"{skill}/{self.training(skill)['id']}", before)
        passes = sum(1 for run in runs if (run.result or {}).get("passed"))
        counted = sum(1 for run in runs if (run.result or {}).get("usable", True))
        category = category_of(passes, counted)
        state["baseline"] = {
            "runs": [run.id for run in runs], "passes": passes, "total": counted,
            "category": category, "qualifies": category in QUALIFYING,
            "seconds": self.seconds_of_runs("home", runs),
        }
        self.save()
        log(f"{skill}: baseline {passes}/{counted}, {category}: "
            f"{'qualifies' if category in QUALIFYING else 'does not qualify'}")
        print(output[-1200:])

    def agent(self, skill: str) -> None:
        """Arm T: the refining agent, launched again while its budget lasts and nothing has
        survived its screen; then the fixed rule."""
        state = self.skill_state(skill)
        if state.get("T"):
            log(f"{skill}: T done already")
            return
        task = f"{skill}/{self.training(skill)['id']}"
        spec_dir = self.root / "agent-specs" / skill
        spec_dir.mkdir(parents=True, exist_ok=True)
        # The tools over home A's proxy: the agent stays in the sandbox, where home A is not
        # readable, and no child process of its own. The route takes the proxy's MCP token,
        # minted each time the proxy starts; the adapter records the URL, never the header.
        token_file = self.home("home") / ".mcp-token"
        if not token_file.is_file():
            raise RuntimeError(f"no MCP token at {token_file}: are the proxies running (`proxies start`)?")
        servers = {"tuningfork": {
            "url": f"{self.proxy_url('home')}/_tuningfork/mcp",
            "headers": {"Authorization": f"Bearer {token_file.read_text(encoding='utf-8').strip()}"},
            "timeout": 300,
            "skip_preflight": True,
        }}
        started_at = time.time()
        scoring = self.score(skill) if SCORE_BASELINE else {"seconds": 0.0, "scored": 0, "refused": 0}
        if SCORE_BASELINE and not scoring["scored"]:
            # A T run again finds the baseline scored already, by the T before it: that work was
            # T's, and H's budget — T's spend — counts it as it did then, as for every other skill.
            before = [entry["record"]["scoring"] for entry in (state.get("earlier") or {}).get("T", [])
                      if (entry["record"].get("scoring") or {}).get("scored")]
            if before:
                scoring = {**before[-1], "reused": True}
                log(f"{skill}: the baseline was scored before ({scoring['scored']} step(s), "
                    f"{scoring['seconds']:.0f} s): counted in T's spend as then")
        along = self.along(skill)
        before = self.newest_run_time("agent-home")
        # The skill as tuningfork names it, by its folder: a bundle may hold a skill under a name of
        # its own, as one made from another benchmark does.
        named = self.spec_data(skill)["tuningfork"]["skill_dir"]
        base = state.get("baseline") or {}
        record = (f"{base.get('passes')} of {base.get('total')} passed the task's check" if base.get("passes")
                  else f"none of {base.get('total')} passed the task's check")
        start = self.starting_point(skill)
        deadline = time.time() + AGENT_TIMEOUT
        attempt, agent_runs, chosen, output = 0, [], None, ""
        while True:
            attempt += 1
            remaining = int(deadline - time.time())
            turns_left = AGENT_MAX_TURNS - sum(run.step_count for run in agent_runs)
            if attempt == 1:
                prompt = AGENT_INSTRUCTION.format(skill=named, task=task, max_wait=MCP_MAX_WAIT, record=record,
                                                  hours=AGENT_TIMEOUT // 3600, turns=AGENT_MAX_TURNS, start=start)
            else:
                prompt = AGENT_CONTINUATION.format(
                    skill=named, task=task, max_wait=MCP_MAX_WAIT, hours=AGENT_TIMEOUT // 3600,
                    max_turns=AGENT_MAX_TURNS, spent_minutes=(AGENT_TIMEOUT - remaining) // 60,
                    turns=AGENT_MAX_TURNS - turns_left, remaining_minutes=remaining // 60,
                    remaining_turns=turns_left, tried=self.tried_so_far(skill, task, started_at), start=start,
                )
            (spec_dir / "evals.json").write_text(json.dumps({
                "skill_name": f"refine-{skill}",
                "evals": [{
                    "id": "refine",
                    "prompt": prompt,
                    "files": [],
                    "tuningfork": {"max_turns": turns_left, "timeout": remaining},
                }],
            }, indent=2), encoding="utf-8")
            suffix = "" if attempt == 1 else f".attempt{attempt}"
            output = self.tf(
                "agent-home", "batch", "evals.json", "-n", "1", "--model", MODEL, *CONTROLLED,
                "-o", "mcp_servers=" + json.dumps(servers),
                cwd=spec_dir, log_to=self.root / "logs" / f"agent-{skill}{suffix}.log",
            )
            agent_runs = sorted(self.runs_since("agent-home", f"refine-{skill}/refine", before),
                                key=lambda run: run.started_at)
            self.wait_for_screens(skill, task, started_at)
            survivor = self.choose_proposal(skill, task, started_at, rule="survives")
            remaining = int(deadline - time.time())
            turns_left = AGENT_MAX_TURNS - sum(run.step_count for run in agent_runs)
            last = (agent_runs[-1].result or {}) if agent_runs else {}
            if survivor:
                why = f"proposal {survivor['proposal'][:8]} survived its screen"
            elif last.get("timed_out"):
                why = "the attempt ran out of time"
            elif attempt >= AGENT_MAX_ATTEMPTS:
                why = f"{attempt} attempts is the cap"
            elif remaining < AGENT_ATTEMPT_MINIMUM or turns_left < AGENT_ATTEMPT_MINIMUM_TURNS:
                why = f"{remaining // 60} min and {turns_left} turns remain: too little for another"
            else:
                log(f"{skill}: T attempt {attempt} ended without a survivor; {remaining // 60} min and "
                    f"{turns_left} turns remain: it continues")
                continue
            log(f"{skill}: T stops after attempt {attempt}: {why}")
            break
        chosen = self.choose_proposal(skill, task, started_at)
        self.write_final(skill, "T", chosen["text"] if chosen else None)
        spend = self.agent_spend(skill, agent_runs, started_at) + scoring["seconds"]
        state["T"] = {"runs": [run.id for run in agent_runs], "attempts": attempt,
                      "chosen": chosen and {k: v for k, v in chosen.items() if k != "text"},
                      "seconds": spend, "started_at": started_at, "scoring": scoring, "along": along}
        self.save()
        log(f"{skill}: T {'kept ' + describe(chosen) if chosen else 'kept the original'} after {attempt} "
            f"attempt(s); spend {spend:.0f} s")
        print(output[-800:])

    def along(self, skill: str) -> dict:
        """Run the training task's check along each baseline run (`tuningfork check-along`): the
        tally T begins with then says, for each finding, which call made it in each run, the
        passing ones included — a mistake every run made, and some repaired later, which comparing
        passing with failing runs points away from (findings §31.42). It calls no model, so it is
        not part of T's spend; a task whose check reads only the reply is passed over."""
        task = f"{skill}/{self.training(skill)['id']}"
        started = time.time()
        try:
            self.tf("home", "check-along", "--task", task, "--limit", "200",
                    log_to=self.root / "logs" / f"along-{skill}.log")
            done = True
        except RuntimeError as error:
            log(f"{skill}: the check was not run along the baseline: {str(error).splitlines()[-1][:160]}")
            done = False
        seconds = time.time() - started
        log(f"{skill}: the check run along the baseline runs ({seconds:.0f} s)")
        return {"seconds": seconds, "done": done}

    def score(self, skill: str) -> dict:
        """Score every step of the skill's baseline runs on the server, one request a step
        (`tuningfork tokens --score`, straight to the upstream): the model's own chance of
        each tool at each step, for step_behaviours. A step the server cannot score — a
        prompt that renders to another token count, an answer that cannot be written back —
        is passed over and counted."""
        state = self.skill_state(skill)
        runs = (state.get("baseline") or {}).get("runs") or []
        started = time.time()
        scored = refused = 0
        store = self.store("home")
        try:
            # A step scored before keeps its stream: scored once, whenever the stage ran.
            steps = [(run_id, step.idx) for run_id in runs for step in store.steps(run_id)
                     if not store.token_streams(step_id=step.id)]
        finally:
            store.close()
        log(f"{skill}: scoring {len(steps)} step(s) of {len(runs)} baseline run(s)")
        for run_id, idx in steps:
            try:
                self.tf("home", "tokens", run_id, str(idx), "--score",
                        log_to=self.root / "logs" / f"score-{skill}-{run_id[:8]}-{idx}.log")
                scored += 1
            except RuntimeError as error:
                refused += 1
                log(f"{skill}: step {idx} of {run_id[:8]} not scored: {str(error).splitlines()[-1][:160]}")
        seconds = time.time() - started
        log(f"{skill}: scored {scored} step(s), {refused} refused, {seconds:.0f} s")
        return {"seconds": seconds, "scored": scored, "refused": refused}

    def chain(self, skill: str) -> None:
        """Arm H: a told chain with T's budget."""
        state = self.skill_state(skill)
        if state.get("H"):
            log(f"{skill}: H done already")
            return
        budget = max(60.0, float((state.get("T") or {}).get("seconds") or 0))
        output = self.tf(
            "h-home", "chain", "run", "evals.json", "-n", str(H_MAX_LINKS), "--tell",
            "--budget-seconds", f"{budget:.0f}", "--model", MODEL, *H_OPTIONS,
            cwd=self.spec(skill).parent, log_to=self.root / "logs" / f"chain-{skill}.log",
        )
        record = max((self.home("h-home") / "chains").glob("*.json"), key=lambda p: p.stat().st_mtime)
        chain = json.loads(record.read_text(encoding="utf-8"))
        final = Path(chain["final_skill"])
        target = self.final(skill, "H")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(final, target)
        state["H"] = {"chain": chain["id"], "links": len(chain["links"]), "versions": len(chain["versions"]),
                      "seconds": chain["spent_seconds"], "budget": budget, "stopped": chain.get("stopped")}
        self.save()
        log(f"{skill}: H chain {chain['id']}, {len(chain['links'])} links, {len(chain['versions'])} versions, {chain['spent_seconds']:.0f} s")
        print(output[-800:])

    def evaluate(self, skill: str, arms: list[str] | tuple[str, ...] = ("O", "H", "T")) -> None:
        """Each arm's final skill on the training task and the held-out tasks — or the arms
        named. T is left out when it kept the original: its cells would be the original's, and
        the report reads O's for it."""
        state = self.skill_state(skill)
        done = state.setdefault("eval", {})
        every = {"O": self.original_skill(skill), "H": self.final(skill, "H"), "T": self.final(skill, "T")}
        chosen = {arm: every[arm] for arm in arms}
        if "T" in chosen and state.get("T") and not state["T"].get("chosen"):
            log(f"{skill}: T kept the original, so it is not evaluated: its cells would be the original's, "
                "and the report reads O's for it")
            del chosen["T"]
        for arm, folder in chosen.items():
            if not (folder / "SKILL.md").is_file():
                raise RuntimeError(f"{skill}: arm {arm} has no final skill at {folder}")
        arms = chosen
        evals = [self.training(skill), *self.held_out(skill)]
        for item in evals:
            held = bool((item.get("tuningfork") or {}).get("held_out"))
            wanted = EVAL_RUNS_HELD_OUT if held else EVAL_RUNS
            for arm, folder in arms.items():
                key = f"{arm}/{item['id']}"
                cell = done.get(key) or {"runs": [], "passes": 0, "total": 0, "skills_changed": 0}
                missing = wanted - len(cell["runs"])
                if missing <= 0:
                    continue
                flag = ["--include-held-out"] if held else []
                before = self.newest_run_time("eval-home")
                self.tf(
                    "eval-home", "batch", "evals.json", "--eval", str(item["id"]), *flag, "-n", str(missing),
                    "--concurrency", str(CONCURRENCY), "--model", MODEL, *CONTROLLED, "--skill-dir", str(folder),
                    cwd=self.spec(skill).parent,
                    log_to=self.root / "logs" / f"eval-{skill}-{arm}-{item['id']}-{len(cell['runs'])}.log",
                )
                runs = self.runs_since("eval-home", f"{skill}/{item['id']}", before)
                cell["runs"] += [run.id for run in runs]
                cell["passes"] += sum(1 for run in runs if self.verdict(run.result) == "passed")
                cell["total"] += sum(1 for run in runs if self.verdict(run.result) is not None)
                cell["skills_changed"] += sum(1 for run in runs if (run.result or {}).get("skills_changed"))
                done[key] = cell
                self.save()
                log(f"{skill}: eval {key} {cell['passes']}/{cell['total']} over {len(cell['runs'])} run(s)")

    @staticmethod
    def verdict(result: dict | None) -> str | None:
        """How an evaluation run counts: passed, failed, timeout (a failure, `TIMEOUT_IS_FAILURE`),
        or None — a run the harness never ran, or a verdict that does not count."""
        result = result or {}
        if result.get("usable", True) and result.get("passed") is not None:
            return "passed" if result["passed"] else "failed"
        if result.get("timed_out") and TIMEOUT_IS_FAILURE:
            return "timeout"
        return None

    def rescore(self, skill: str) -> None:
        """Every evaluation cell of the skill counted again from the recordings, under the
        rules as they stand — a timeout a failure — and its runs' checks run again on their
        end states with the spec as it is now (`ops.checks.try_check`: nothing new is run).
        A run whose verdict changes has its stored result updated, the old one kept beside
        it, so the recordings and the counts agree. Each cell keeps its verdicts by run."""
        from tuningfork.ops.checks import try_check
        from tuningfork.ops.common import OperationError
        from tuningfork.store.masking import Masked

        state = self.skill_state(skill)
        cells = state.get("eval") or {}
        store = self.store("eval-home")
        changed = 0
        try:
            for key, cell in cells.items():
                verdicts = {}
                for run_id in cell["runs"]:
                    run = store.get_run(run_id)
                    result = dict(run.result or {}) if run is not None else {}
                    verdict = self.verdict(result)
                    if verdict in ("passed", "failed"):
                        try:
                            tried = try_check(store, run=run_id)
                        except OperationError as error:
                            log(f"{skill}: {key} {run_id[:8]} not rechecked: {error}")
                        else:
                            again = "passed" if tried.passed else "failed"
                            if again != verdict:
                                log(f"{skill}: {key} {run_id[:8]} {verdict} -> {again}: {tried.output.strip()[:160]!r}")
                                # Kept as it is: the result as stored, and the check's output as
                                # masked where it ran. Masked again, a second password would fall.
                                store.annotate_run(run_id, result=Masked({
                                    **result, "passed": tried.passed, "check_output": tried.output[-2000:],
                                    "rescored_from": {"passed": result.get("passed"), "check_output": result.get("check_output"),
                                                      "at": time.time()},
                                }))
                                changed += 1
                                verdict = again
                    verdicts[run_id] = verdict
                before = (cell["passes"], cell["total"])
                cell["passes"] = sum(1 for v in verdicts.values() if v == "passed")
                cell["total"] = sum(1 for v in verdicts.values() if v is not None)
                cell["verdicts"] = verdicts
                cell["timeouts"] = sum(1 for v in verdicts.values() if v == "timeout")
                log(f"{skill}: {key} {before[0]}/{before[1]} -> {cell['passes']}/{cell['total']}"
                    + (f" ({cell['timeouts']} timeout(s) counted as failures)" if cell["timeouts"] else ""))
        finally:
            store.close()
        state["rescored_at"] = time.time()
        self.save()
        log(f"{skill}: rescored; {changed} verdict(s) changed")

    # --- helpers over the recordings ------------------------------------------------------------

    def store(self, home: str):
        from tuningfork.store.db import Store

        return Store(self.home(home))

    def newest_run_time(self, home: str) -> float:
        store = self.store(home)
        try:
            runs = store.runs(limit=1)
            return runs[0].started_at if runs else 0.0
        finally:
            store.close()

    def runs_since(self, home: str, task: str, after: float):
        store = self.store(home)
        try:
            return [run for run in store.runs(limit=500, task_id=task) if run.started_at > after]
        finally:
            store.close()

    def seconds_of_runs(self, home: str, runs) -> float:
        from tuningfork.ops.spend import main_run_spend

        store = self.store(home)
        try:
            return sum(main_run_spend(store, run).wall_ms for run in runs) / 1000
        finally:
            store.close()

    def choose_proposal(self, skill: str, task: str, since: float, rule: str = KEEP_RULE) -> dict | None:
        """The fixed rule (KEEP_RULE, or `rule`): of the proposals screened to the end in this
        skill's refinements since the agent started — every such screen of each — the one with the
        highest measured rate that the rule admits; none, the original. "survives" admits a
        proposal significantly better than its screen's original arm; "not worse" one that scored
        above it and was not significantly worse. A screen without a verdict yet admits nothing,
        and neither does a screen at a step: its target is the agent's own rule about one answer,
        not the task's check the evaluation judges by (findings §31.42: a step screen at 20/20 on
        a side target kept an edit that full runs failed 0/10)."""
        from tuningfork.ops import refine

        if rule not in ("survives", "not worse"):
            raise ValueError(f"no rule {rule!r}")
        store = self.store("home")
        try:
            best = None
            for refinement in refine.list_refinements(store, include_closed=True):
                run = store.get_run(refinement.run_id)
                if run is None or run.task_id != task or refinement.data.get("created_at", 0) < since:
                    continue
                view = refine.view(store, refinement.id)
                for proposal in view["proposals"]:
                    for verdict in proposal.get("screens") or []:
                        # A quick look's rates are a lead: only a full screen to the end keeps a proposal.
                        if not verdict.get("complete") or verdict.get("reach") != "end" or verdict.get("quick"):
                            continue
                        rate, original = rate_of(verdict.get("target")), rate_of(verdict.get("baseline"))
                        if rule == "survives":
                            admitted = bool(verdict.get("survives"))
                        else:
                            admitted = not verdict.get("worse") and rate is not None and original is not None \
                                and rate > original
                        if not admitted:
                            continue
                        if best is None or (rate, rate - original) > (best["rate"], best["rate"] - best["original"]):
                            best = {"refinement": refinement.id, "proposal": proposal["id"], "rate": rate,
                                    "original": original, "target": verdict.get("target"),
                                    "baseline": verdict.get("baseline"), "adjusted_p": verdict.get("adjusted_p"),
                                    "survives": bool(verdict.get("survives")), "rule": rule,
                                    "screen": verdict.get("screen"),
                                    "text": refine.proposal(store, refinement.id, proposal["id"]).text()}
            return best
        finally:
            store.close()

    def again(self, skill: str, arm: str) -> None:
        """Forget an arm's record and its evaluation cells, so that its stage and its
        evaluation run again on this root — the baseline, the recordings and the other arm as
        they stand. What is forgotten is kept under `earlier`, and its logs are renamed: a
        targeted study of one arm under changed conditions, the rest of the loop reused."""
        state = self.skill_state(skill)
        if not state.get(arm):
            log(f"{skill}: {arm} has not run here; nothing to forget")
            return
        earlier = state.setdefault("earlier", {}).setdefault(arm, [])
        cells = {key: cell for key, cell in (state.get("eval") or {}).items() if key.startswith(f"{arm}/")}
        earlier.append({"record": state.pop(arm), "eval": cells, "forgotten_at": time.time()})
        for key in cells:
            del state["eval"][key]
        attempt = len(earlier)
        logs = self.root / "logs"
        stage = "agent" if arm == "T" else "chain"
        for path in [*logs.glob(f"{stage}-{skill}.log"), *logs.glob(f"{stage}-{skill}.attempt*.log"),
                     *logs.glob(f"eval-{skill}-{arm}-*.log")]:
            path.rename(path.with_name(f"{path.stem}.earlier{attempt}{path.suffix}"))
        self.save()
        log(f"{skill}: {arm} forgotten (attempt {attempt} kept under earlier): its stage and its evaluation run again")

    def choose(self, skill: str) -> None:
        """The fixed rule again, after the agent stage: a screen the agent started as its budget
        ended finishes in the proxy on its own, and may change what T keeps. The stage waits for
        such screens, chooses again, rewrites T's final skill, and forgets T's evaluation cells
        when the choice changed (kept under `earlier`, like --again)."""
        state = self.skill_state(skill)
        record = state.get("T")
        if not record:
            raise RuntimeError(f"{skill}: T has not run here")
        task = f"{skill}/{self.training(skill)['id']}"
        self.wait_for_screens(skill, task, record["started_at"])
        chosen = self.choose_proposal(skill, task, record["started_at"])
        before = (record.get("chosen") or {}).get("proposal")
        after = chosen and chosen["proposal"]
        if before == after:
            log(f"{skill}: T's choice stands: {'kept ' + after if after else 'the original'}")
            return
        cells = {key: cell for key, cell in (state.get("eval") or {}).items() if key.startswith("T/")}
        if cells:
            state.setdefault("earlier", {}).setdefault("T", []).append(
                {"record": dict(record), "eval": cells, "forgotten_at": time.time(), "why": "chosen again"})
            for key in cells:
                del state["eval"][key]
        self.write_final(skill, "T", chosen["text"] if chosen else None)
        record["chosen"] = chosen and {k: v for k, v in chosen.items() if k != "text"}
        record["chosen_again_at"] = time.time()
        self.save()
        log(f"{skill}: T now {'keeps ' + after if after else 'keeps the original'} (was "
            f"{'proposal ' + before if before else 'the original'}); its {len(cells)} evaluation cell(s) forgotten")

    def wait_for_screens(self, skill: str, task: str, since: float, limit: int = AGENT_SCREEN_WAIT) -> None:
        """Wait for the screens of this task's refinements since the agent started that are
        still running — the agent answered before a job it started was done — up to `limit`."""
        from tuningfork.ops import refine

        deadline = time.time() + limit
        while True:
            store = self.store("home")
            unfinished = []
            try:
                for refinement in refine.list_refinements(store, include_closed=True):
                    run = store.get_run(refinement.run_id)
                    if run is None or run.task_id != task or refinement.data.get("created_at", 0) < since:
                        continue
                    for screen in refine.screens(store, refinement.id):
                        if not screen["report"]["complete"]:
                            unfinished.append(str(screen["experiment"])[:8])
            finally:
                store.close()
            if not unfinished:
                return
            if time.time() >= deadline:
                log(f"{skill}: screen(s) {', '.join(unfinished)} still unfinished after {limit // 60} min; going on")
                return
            log(f"{skill}: waiting for screen(s) the agent left running: {', '.join(unfinished)}")
            time.sleep(30)

    def starting_point(self, skill: str) -> str:
        """What T starts with (AGENT_START): the training task's prompt and criteria, and the first
        failed run of the baseline whole, as H's told turn and review have their own run — the
        messages of its last step's context but the system prompt, then that step's answer — with
        its verdict as H is told it. Empty when no baseline run failed."""
        from tuningfork.ops import compact

        training = self.training(skill)
        criteria = "\n".join(f"- {item}" for item in training.get("expectations") or []) or "(none written)"
        store = self.store("home")
        try:
            failed = next((run for run in (store.get_run(run_id) for run_id in
                                           (self.skill_state(skill).get("baseline") or {}).get("runs") or [])
                           if run is not None and (run.result or {}).get("passed") is False), None)
            if failed is None:
                return ""
            steps = list(store.steps(failed.id))
            last = steps[-1]
            messages = [message for message in store.rebuild_request(last).get("messages") or []
                        if message.get("role") != "system"]
            answer = (((store.response(last) or {}).get("choices") or [{}])[0].get("message")) or {}
            parts, total = [], 0
            for number, message in enumerate([*messages, answer]):
                text = compact._content(message.get("content"))
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments")
                    text += f"\n[call {function.get('name')}] " + (arguments if isinstance(arguments, str)
                                                                   else json.dumps(arguments, ensure_ascii=False))
                where = ("answer" if number == len(messages) else str(number + 1))
                room = max(0, min(START_MESSAGE, START_TOTAL - total))
                if len(text) > room:
                    text = (text[:room] + f"… (+{len(text) - room} characters: show_step(run=\"{failed.id[:8]}\", "
                            f"step={last.idx}, message=\"{where}\", offset={room}) reads on)")
                total += len(text)
                role = message.get("role") or "assistant"
                parts.append(f"[{role}{' — its last answer' if where == 'answer' else ''}]\n{text}")
            output = str((failed.result or {}).get("check_output") or "").strip()
            return AGENT_START.format(
                prompt=training.get("prompt") or "", criteria=criteria, run=failed.id[:8], label=failed.label or "",
                transcript="\n\n".join(parts), output=f"\n\n{output}" if output else "",
            )
        finally:
            store.close()

    def tried_so_far(self, skill: str, task: str, since: float) -> str:
        """What the agent did in its attempts so far, for its next one: each refinement of the
        task since it started, its target, and each proposal with its lines and its verdict."""
        from tuningfork.ops import refine

        store = self.store("home")
        lines = []
        try:
            pending = [job for job in store.jobs(limit=200, unfinished=True) if job.kind in SCREEN_KINDS]
            for refinement in refine.list_refinements(store, include_closed=True):
                run = store.get_run(refinement.run_id)
                if run is None or run.task_id != task or refinement.data.get("created_at", 0) < since:
                    continue
                view = refine.view(store, refinement.id)
                target = (refinement.data.get("target") or {}).get("description") or "the task's check passes"
                lines.append(f"- refinement {refinement.id[:8]} at run {run.id[:8]} step {refinement.step_idx}, "
                             f"target: {target}; {len(view['proposals'])} proposal(s)")
                for proposal in view["proposals"]:
                    changed = [f"{line['op']}{line['text']}" for hunk in proposal.get("diff") or []
                               for line in hunk["lines"] if line["op"] != " "]
                    verdicts = proposal.get("screens") or []
                    if not verdicts or not verdicts[-1].get("complete"):
                        outcome = no_verdict(verdicts[-1] if verdicts else None, refinement.id, proposal["id"], pending)
                    elif verdicts[-1].get("quick"):
                        verdict = verdicts[-1]
                        reached, original = verdict.get("target") or {}, verdict.get("baseline") or {}
                        outcome = (f"quick look: {reached.get('successes')}/{reached.get('total')}"
                                   + (f" against the original's {original.get('successes')}/{original.get('total')}"
                                      if original.get("total") else "") + ", rates only")
                        if verdict.get("screen"):
                            outcome += f" (screen {str(verdict['screen'])[:8]})"
                        if above(reached, original) or (reached.get("successes") and not original.get("total")):
                            outcome += " — a lead: screen it in full to keep it, or build on it"
                    else:
                        verdict = verdicts[-1]
                        reached, original = verdict.get("target") or {}, verdict.get("baseline") or {}
                        p = verdict.get("adjusted_p")
                        outcome = (f"screened: {reached.get('successes')}/{reached.get('total')} against the "
                                   f"original's {original.get('successes')}/{original.get('total')}"
                                   + (f", p {p:.2f}" if isinstance(p, (int, float)) else "") + ", "
                                   + ("survived" if verdict.get("survives") else "did not survive"))
                        if verdict.get("screen"):
                            outcome += f" (screen {str(verdict['screen'])[:8]})"
                        if not verdict.get("survives") and above(reached, original):
                            outcome += " — above the original: it helped, a base to build on"
                    lines.append(f"  - proposal {proposal['id'][:8]}: {' | '.join(changed)[:300]} — {outcome}")
        finally:
            store.close()
        return "\n".join(lines) or "- nothing was proposed"

    def category(self, skill: str) -> str | None:
        """The skill's category, worked out from its baseline's counts by the rule as it stands now:
        a root baselined under the old band reads the same as one baselined today."""
        base = self.skill_state(skill).get("baseline")
        return category_of(base.get("passes"), base.get("total")) if base else None

    def write_final(self, skill: str, arm: str, skill_text: str | None) -> None:
        target = self.final(skill, arm)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.original_skill(skill), target)
        if skill_text is not None:
            (target / "SKILL.md").write_text(skill_text, encoding="utf-8")

    def agent_spend(self, skill: str, agent_runs, since: float) -> float:
        """What arm T cost: the agent's own calls, and the experiments it started in home A."""
        from tuningfork.ops.spend import experiment_spend, main_run_spend

        total = 0.0
        store = self.store("agent-home")
        try:
            total += sum(main_run_spend(store, run).wall_ms for run in agent_runs) / 1000
        finally:
            store.close()
        store = self.store("home")
        try:
            for experiment in store.experiments(limit=2000):
                if experiment.created_at >= since:
                    total += experiment_spend(store, experiment.id).wall_ms / 1000
        finally:
            store.close()
        return total

    # --- the report ---------------------------------------------------------------------------

    def report(self, skills: list[str]) -> str:
        from tuningfork.analysis.stats import rate

        lines = ["# Pilot report", ""]
        rows = []
        for skill in skills:
            state = self.skill_state(skill)
            evaluation = state.get("eval") or {}
            if not evaluation:
                continue
            training = self.training(skill)["id"]
            held = [e["id"] for e in self.held_out(skill)]
            lines.append(f"## {skill} — {self.category(skill) or 'no baseline'}")
            base = state.get("baseline") or {}
            lines.append(f"baseline {base.get('passes')}/{base.get('total')}; T spent {state.get('T', {}).get('seconds', 0):.0f} s over {state.get('T', {}).get('attempts', 1)} attempt(s), "
                         f"H {state.get('H', {}).get('seconds', 0):.0f} s over {state.get('H', {}).get('links')} links; "
                         f"T kept {describe(state['T']['chosen']) if (state.get('T') or {}).get('chosen') and state['T']['chosen'].get('target') else (((state.get('T') or {}).get('chosen') or {}).get('proposal') or 'the original')}")
            lines.append("")
            lines.append("| arm | training | held-out | held-out pooled |")
            lines.append("|---|---|---|---|")
            pooled = {}
            kept_original = bool(state.get("T")) and not state["T"].get("chosen")
            for arm in ("O", "H", "T"):
                # A T that kept the original has the original's cells: O's are read for it.
                source = "O" if arm == "T" and kept_original and not any(k.startswith("T/") for k in evaluation) else arm
                label = "T (= O, kept the original)" if source != arm else arm
                cell = evaluation.get(f"{source}/{training}") or {}
                train = rate(cell.get("passes", 0), cell.get("total", 0)).format() if cell.get("total") else "—"
                parts, successes, total = [], 0, 0
                for item in held:
                    found = evaluation.get(f"{source}/{item}") or {}
                    if found.get("total"):
                        parts.append(f"{item}: {rate(found['passes'], found['total']).format()}")
                        successes += found["passes"]
                        total += found["total"]
                pooled[arm] = (successes, total)
                lines.append(f"| {label} | {train} | {'; '.join(parts) or '—'} | {rate(successes, total).format() if total else '—'} |")
            lines.append("")
            rows.append((skill, pooled, self.category(skill)))
        qualifying = [row for row in rows if row[2] in QUALIFYING]
        groups = [("flaky and broken", qualifying)]
        if any(row[2] == "flaky" for row in qualifying) and any(row[2] == "broken" for row in qualifying):
            groups += [(name, [row for row in qualifying if row[2] == name]) for name in QUALIFYING]
        for title, group in groups:
            if group:
                lines.extend(across(title, [(skill, pooled) for skill, pooled, _ in group]))
        outside = [f"{skill} ({category})" for skill, _, category in rows if category not in QUALIFYING]
        if outside:
            lines.append("Outside the gate, run with --skip-gate and left out of the means: " + ", ".join(outside))
        text = "\n".join(lines) + "\n"
        (self.root / "report.md").write_text(text, encoding="utf-8")
        return text


def rate_of(found: dict | None) -> float | None:
    """A screen arm's measured rate, from its counts."""
    if not found or not found.get("total"):
        return None
    return found.get("successes", 0) / found["total"]


def above(arm: dict | None, original: dict | None) -> bool:
    """Whether a screen arm scored above its original arm: a proposal that helped, if not enough to
    survive — what the agent is told to build on."""
    measured, baseline = rate_of(arm), rate_of(original)
    return measured is not None and baseline is not None and measured > baseline


#: The job kinds that screen a proposal, whose jobs the agent's next attempt is pointed to.
SCREEN_KINDS = ("screen", "screen-to-end")


def no_verdict(verdict: dict | None, refinement: str, proposal: str, pending: list) -> str:
    """A proposal whose screen has no verdict, for the agent's next attempt — without the rates of an
    arm still being drawn, which are noise: told "1/4 against the original's 0/20, … not finished", the
    agent built on it as a gain, and the arm ended 1/20 (findings §31.57). With the job still queued or
    running, if there is one (`pending`: home A's unfinished screen jobs), so that it waits for it."""
    screen = str((verdict or {}).get("screen") or "")

    def screens_it(job) -> bool:
        params = job.params or {}
        if screen and screen in (job.links or {}).get("experiments", []):
            return True
        return (str(params.get("refinement") or "")[:8] == refinement[:8]
                and any(str(named)[:8] == proposal[:8] for named in params.get("proposals") or []))

    job = next((job for job in pending if screens_it(job)), None)
    if job is None:
        return "not screened" if verdict is None else f"no verdict: screen {screen[:8]} ended unfinished"
    progress = job.progress or {}
    drawn = f", {progress['done']} of {progress['total']} forks drawn" if progress.get("total") else ""
    state = "queued" if str(job.status).startswith("queued") else "still running"
    return (f"no verdict yet: its screen is {state} (job {job.id[:8]}{drawn}), and its rates so far say "
            f"nothing; wait for it with job_status before you build on it or set it aside")


def describe(chosen: dict) -> str:
    """What T kept and why, in a line: the proposal, its screen's counts, its p and the rule."""
    target, baseline = chosen.get("target") or {}, chosen.get("baseline") or {}
    p = chosen.get("adjusted_p")
    how = "survived" if chosen.get("survives") else "not worse"
    return (f"{chosen['proposal'][:8]} ({how}: {target.get('successes')}/{target.get('total')} against the original's "
            f"{baseline.get('successes')}/{baseline.get('total')}" + (f", p {p:.3g}" if isinstance(p, (int, float)) else "")
            + ")")


def category_of(passes: int | None, total: int | None) -> str:
    """A baseline's category: `flaky` (FLAKY_FLOOR to CEILING), `broken` (under FLAKY_FLOOR), `ceiling`
    (above CEILING: nothing to measure), or `too few` verdicts to say."""
    if not total or total < MIN_COUNTED:
        return "too few"
    if passes / total > CEILING:
        return "ceiling"
    return "flaky" if passes / total >= FLAKY_FLOOR else "broken"


def across(title: str, rows: list[tuple[str, dict]]) -> list[str]:
    """The mean change in the held-out pass rate over `rows`: each arm pair over the skills where both
    were evaluated — an arm with no cells (a T-only study's H) is left out, never counted as zero —
    with a bootstrap over skills and, within each, a binomial redraw of each arm's runs."""
    import random

    rng = random.Random(7)
    keys = ("H-O", "T-O", "T-H")

    def evaluated(pooled: dict, key: str) -> bool:
        a, b = key.split("-")
        return bool(pooled.get(a, (0, 0))[1] and pooled.get(b, (0, 0))[1])

    def deltas(sample):
        out = {}
        for key in keys:
            values = [_delta(pooled, key) for _, pooled in sample if evaluated(pooled, key)]
            if values:
                out[key] = sum(values) / len(values)
        return out

    lines = [f"## Across skills, {title} ({len(rows)}): mean change in the held-out pass rate", ""]
    observed = deltas(rows)
    if not observed:
        return lines + ["- no two arms evaluated on the same skill", ""]
    boots = {key: [] for key in observed}
    for _ in range(2000):
        sample = [rng.choice(rows) for _ in rows]
        # Runs resampled within skills: a binomial draw at each arm's observed rate.
        redrawn = []
        for skill, pooled in sample:
            again = {}
            for arm, (s, t) in pooled.items():
                p = s / t if t else 0.0
                again[arm] = (sum(1 for _ in range(t) if rng.random() < p), t)
            redrawn.append((skill, again))
        found = deltas(redrawn)
        for key in observed:
            if key in found:
                boots[key].append(found[key])
    for key, value in observed.items():
        drawn = sorted(boots[key])
        low, high = drawn[int(0.025 * len(drawn))], drawn[int(0.975 * len(drawn)) - 1]
        counted = [pooled for _, pooled in rows if evaluated(pooled, key)]
        better = sum(1 for pooled in counted if _delta(pooled, key) > 0)
        worse = sum(1 for pooled in counted if _delta(pooled, key) < 0)
        lines.append(f"- {key}: {value:+.1%} (bootstrap 95% {low:+.1%} to {high:+.1%}); skills better {better}, "
                     f"worse {worse}, of {len(counted)}")
    return lines + [""]


def _delta(pooled: dict, key: str) -> float:
    a, b = key.split("-")
    point = {arm: (s / t if t else 0.0) for arm, (s, t) in pooled.items()}
    return point[a] - point[b]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["sync", "proxies", "baseline", "score", "agent", "choose", "chain", "evaluate",
                                          "rescore", "report", "all"])
    parser.add_argument("target", nargs="?", help="a skill name, or start|stop for proxies")
    parser.add_argument("--root", default=str(Path.home() / "tf-bench"))
    parser.add_argument("--bundles", default=str(Path(__file__).resolve().parent.parent),
                        help="the task bundles: the benchmark folder this driver lives in")
    parser.add_argument("--venv", default=str(Path.home() / ".tuningfork" / "venv"))
    parser.add_argument("--skills", default="",
                        help="the bundles to work on, comma-separated (default: every bundle beside the driver)")
    parser.add_argument("--skip-gate", action="store_true",
                        help="run a skill's arms whatever its baseline's pass rate (its category is still recorded, "
                             "and a skill outside the gate stays out of the report's means)")
    parser.add_argument("--arms", default="O,H,T",
                        help="evaluate: the arms to evaluate (T is left out when it kept the original)")
    parser.add_argument("--again", action="store_true",
                        help="agent or chain: run that arm again on this root, the baseline and the other arm as they "
                             "stand; its earlier record and cells are kept under `earlier`, and `evaluate` then "
                             "runs that arm's cells alone")
    args = parser.parse_args()
    bench = Bench(Path(args.root), Path(args.bundles), Path(args.venv))
    skills = ([s for s in args.skills.split(",") if s]
              or sorted(path.parent.name for path in Path(args.bundles).glob("*/evals.json")))
    if args.stage == "sync":
        bench.sync(skills)
    elif args.stage == "proxies":
        bench.proxies(args.target or "start")
    elif args.stage == "report":
        print(bench.report(skills))
    elif args.stage == "all":
        for skill in ([args.target] if args.target else skills):
            bench.baseline(skill)
            category = bench.category(skill)
            if category not in QUALIFYING:
                if not args.skip_gate:
                    log(f"{skill}: {category}, does not qualify; skipping its arms")
                    continue
                log(f"{skill}: {category}, does not qualify; its arms run anyway (--skip-gate)")
            bench.agent(skill)
            bench.chain(skill)
            bench.evaluate(skill)
        print(bench.report(skills))
    else:
        if not args.target:
            parser.error("name the skill")
        if args.again:
            if args.stage not in ("agent", "chain"):
                parser.error("--again goes with agent or chain")
            bench.again(args.target, "T" if args.stage == "agent" else "H")
        if args.stage == "evaluate":
            bench.evaluate(args.target, [arm for arm in args.arms.split(",") if arm])
        else:
            getattr(bench, args.stage)(args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
