"""tuningfork's words, each defined once: the Guide page's table, `tuningfork glossary`, and
the API behind both. A definition says what the thing is here, in tuningfork, and where to
find it — not what the word means elsewhere."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tuningfork.ops.common import OperationError

#: The groups, in the order a reader meets them.
GROUPS = (
    ("recording", "What is recorded"),
    ("tasks", "Tasks and runs"),
    ("experiments", "Experiments at a step"),
    ("refining", "Refining a skill"),
    ("working", "Running things"),
    ("statistics", "How results are judged"),
)


@dataclass(frozen=True, slots=True)
class Term:
    term: str
    means: str
    group: str
    #: Other names a reader may look it up by.
    aka: tuple[str, ...] = ()
    #: Related terms, by their `term`.
    see: tuple[str, ...] = ()
    #: Where it is, in the web UI and the terminal.
    where: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"term": self.term, "means": self.means, "group": self.group, "aka": list(self.aka),
                "see": list(self.see), "where": self.where}


TERMS: tuple[Term, ...] = (
    # --- what is recorded
    Term("proxy", "The recording proxy, `tuningfork serve`: it sits between a harness and the inference server, "
         "records every model call as a step, and holds a call where a breakpoint says to.",
         "recording", aka=("serve",), see=("step", "breakpoint"), where="tuningfork serve"),
    Term("harness", "The agent program that drives the model — Hermes, Claude Code, or any OpenAI-compatible one: "
         "it gives the model the skill and runs the tools the model calls.",
         "recording", aka=("agent",), see=("launch", "skill"), where="Connections page"),
    Term("server", "The inference server that answers model calls (llama-server, LM Studio). Each step records which "
         "one answered, and pass rates are kept apart per server.",
         "recording", aka=("inference server", "upstream"), see=("proxy", "sliding-window cache"),
         where="Connections page; tuningfork servers"),
    Term("sliding-window cache", "Strongly recommended: start llama-server with --swa-full for any model with "
         "sliding-window attention layers, Gemma 2, 3 and 4 among them. Without it the server keeps only part of "
         "those layers' cache and reuses saved states between requests, so the same request can behave differently "
         "depending on what the server ran before. A screen then favours its proposals, because it samples each "
         "one right after the original. The flag costs GPU memory, and tuningfork cannot see whether it is on.",
         "recording", aka=("--swa-full", "swa-full", "SWA"), see=("server", "screen", "prompt cache"),
         where="llama-server's command line: --swa-full"),
    Term("prompt cache", "Strongly recommended: start llama-server with --no-cache-prompt. By default it answers a "
         "prompt from what it cached of an earlier one, and its Messages endpoint, which Claude Code uses, ignores a "
         "request asking otherwise. Then Claude Code's samples share the first one's processing, and a whole arm of "
         "forks can share one state: one resample swung from 16 in 20 to 1 in 10 on a behaviour with what the server "
         "had run before. The flag makes every step process its whole prompt — slower, about 1.5 s per 14,000 tokens "
         "on one RTX 5090. The server check measures whether the server reuses prompts.",
         "recording", aka=("--no-cache-prompt", "cache_prompt", "fresh prompts"), see=("server", "sliding-window cache"),
         where="llama-server's command line: --no-cache-prompt; tuningfork connections check"),
    Term("skill", "A folder with a SKILL.md: the instructions a harness gives the model for one kind of task. "
         "What tuningfork helps you improve, one measured change at a time.",
         "recording", aka=("SKILL.md",), see=("skill version", "refinement"), where="Skills page; tuningfork skills"),
    Term("skill version", "A fingerprint of the skill files a run was given. Runs with one fingerprint read the same "
         "skill byte for byte; their pass rate is never pooled with another version's.",
         "recording", aka=("fingerprint",), see=("skill", "pass rate"), where="Skills page; tuningfork skill <name>"),
    Term("run", "One conversation of a harness with the model, from its first model call to its last: its steps, "
         "in order, and — when tuningfork launched it — the check's verdict.",
         "recording", see=("step", "thread", "launch"), where="Runs page; tuningfork runs"),
    Term("step", "One model call in a run: the exact context the model was given, and what it answered — a tool "
         "call, text, or an answer cut short.",
         "recording", see=("run", "behaviour"), where="a run's timeline; tuningfork step <run> <k>"),
    Term("thread", "A conversation of a launch other than its main one: a subagent's, or a side call such as Hermes's "
         "background review. Never counted as the task's next step, nor its final reply.",
         "recording", aka=("subagent", "side call"), see=("run", "launch")),
    Term("timeline", "A run read as one conversation: the last step's context, with each step's answer in its place.",
         "recording", see=("run", "step"), where="a run's page; tuningfork timeline <run>"),
    Term("behaviour", "What the model did at a step, read so that equal actions compare equal: which tool with which "
         "arguments, or which text. Experiments count behaviours, lettered A, B, C by how often they happen.",
         "recording", aka=("outcome",), see=("normalizer", "target")),
    Term("normalizer", "A rule for reading behaviours: which differences do not count — a path's spelling, the "
         "order of arguments. Named when an experiment starts.",
         "recording", see=("behaviour",)),
    # --- tasks and runs
    Term("task", "One thing to ask the harness, named `<skill>/<eval id>`: a prompt, its input files, and a check.",
         "tasks", aka=("eval",), see=("spec", "check", "batch")),
    Term("spec", "A file of tasks for one skill (the evals.json convention): each eval's prompt, files, check and "
         "options, and the skill's folder.",
         "tasks", aka=("evals.json",), see=("task", "scenario")),
    Term("scenario", "A spec kept in the recordings folder, with its input files beside it: written and edited on "
         "the Scenarios page instead of by hand.",
         "tasks", see=("spec",), where="Scenarios page"),
    Term("check", "How a run is judged when the harness ends: a script run in the run's workspace, or a test of its "
         "final reply. Tried on a recorded run before it is trusted.",
         "tasks", see=("pass rate", "task", "check along"), where="a scenario's check; tuningfork recheck <run>"),
    Term("check along", "A task's check run on a run's workspace as it started, after each call that changed it, "
         "and at its end: when each thing the check complains about was made, and by which call. Never a verdict. "
         "A finding every run made, the passing ones too, is a mistake they all made — and refined where it was "
         "made, not where the runs that repaired it later differ.",
         "tasks", see=("check", "checkpoint"), where="a task's tally; tuningfork check-along --task <id>"),
    Term("batch", "A task run N times through the harness, each run in a fresh workspace and home: the pass rate "
         "of one skill version, with its interval.",
         "tasks", see=("launch", "pass rate"), where="Runs → Run a task…; tuningfork batch <spec> -n 8"),
    Term("launch", "A harness tuningfork started, with its own workspace and home. Every model call it makes — its "
         "subagents' too — is attributed to it as it arrives.",
         "tasks", see=("batch", "fork", "workspace", "sandbox")),
    Term("sandbox", "Where every launch runs when bubblewrap can (Linux): its home is /home/user and it works in "
         "/home/user/project, the same in every run, with the harness's files where it installs them. The model reads "
         "those paths, and a few digits of a run's own folder were seen to change what it did. The rest of the machine "
         "is read-only, and the recordings and the task's spec folder are hidden. Without it, each launch sees its own "
         "folders, and a batch's plan warns.",
         "tasks", aka=("bubblewrap", "bwrap", "TUNINGFORK_SANDBOX"), see=("launch", "workspace"),
         where="Connections page; tuningfork connections show"),
    Term("workspace", "The folder a launched harness works in: the task's input files, snapshotted at the start and "
         "the end, so a check can be tried again on what the run left.",
         "tasks", see=("launch",), where="tuningfork files <run> <k>"),
    Term("fork", "A run replayed from a recording up to a step and live from there: the same history, then the model "
         "decides again — and the task's check judges where it ends.",
         "tasks", see=("to the end", "launch"), where="at a step: Fork from here…; tuningfork fork <run> <k>"),
    Term("held-out task", "A task kept out of refining: it judges an edit at validation and never shapes one. Its "
         "runs are verdicts only, recorded nowhere an agent can read.",
         "tasks", aka=("held out",), see=("validation", "gate")),
    Term("divergence", "The first step where runs of one task stopped doing the same thing: where to look first.",
         "tasks", see=("run",), where="Runs → Where do they diverge?; tuningfork divergence --task <id>"),
    Term("where it holds", "One question asked of every run of a task — of what the model did, at every step, or of "
         "a file the runs wrote, at each state of the workspace — and the answers lined up with the verdicts: in how "
         "many passing and failing runs it held, the call that first did it or made it in each, and the step that "
         "names that call in every run. Where a check prints only its verdict, the way to the call that made the "
         "difference. A lead to read, never a verdict.",
         "tasks", aka=("where",), see=("check along", "divergence", "target"),
         where="Runs → Where do they diverge? → Where does it hold?; tuningfork where --task <id>"),
    Term("trash", "Where things set aside wait, out of every list and pass rate, until restored as they went. "
         "Deleting from the trash is for good, with everything tied to the item.",
         "tasks", see=(), where="Trash page; tuningfork trash"),
    # --- experiments at a step
    Term("experiment", "Sampling at one step with everything before it held fixed: a resample, an A/B, a screen, a "
         "forking-token probe. What the model does there, measured rather than seen once.",
         "experiments", see=("sample", "resample", "A/B"), where="Experiments page; tuningfork experiments"),
    Term("sample", "One answer drawn in an experiment — or, run to the end, one fork and its check's verdict.",
         "experiments", see=("experiment",)),
    Term("resample", "The step's exact request sent N times: how often each behaviour happens there. A step the "
         "model answers more than one way is flaky.",
         "experiments", see=("behaviour", "experiment"), where="at a step: Resample…; tuningfork resample <run> <k> -n 20"),
    Term("A/B", "Another wording of the skill swapped in at one step, both wordings sampled: does the target "
         "behaviour happen more often? Decided by a target declared before sampling.",
         "experiments", aka=("ab",), see=("variant", "target", "arm"), where="at a step: A/B a wording…; tuningfork ab <run> <k>"),
    Term("arm", "One side of an A/B or a screen: the original wording, or one variant of it.",
         "experiments", see=("A/B", "screen")),
    Term("variant", "Another wording of a skill, saved in the recordings folder — never inside the skill's folder, "
         "which every run is given a copy of.",
         "experiments", see=("A/B", "proposal"), where="Skills page"),
    Term("target", "What should happen at the step, declared before anything is sampled: a behaviour's letter, a "
         "pattern, or conditions in words. Run to the end, the target is the task's check.",
         "experiments", see=("A/B", "refinement")),
    Term("to the end", "Each sample a fork run to the end of the task and judged by its check, instead of the one "
         "answer at the step: whether a change at the step carries through.",
         "experiments", aka=("each to the end", "--to-end"), see=("fork", "sample"),
         where="a form's Reach: Each to the end; --to-end"),
    Term("token stream", "An answer kept as the model's tokens: each with its probability, and the alternatives the "
         "model weighed.",
         "experiments", aka=("logprobs",), see=("score", "forking token")),
    Term("score", "How sure the model was of each token of a recorded answer, measured by forcing that answer after "
         "the step's exact prompt.",
         "experiments", see=("token stream",), where="a step's Tokens tab; tuningfork tokens <run> <k> --score"),
    Term("forking token", "A token where forcing an alternative changes what the model goes on to do: where its "
         "decision was made.",
         "experiments", see=("score", "token stream"),
         where="at a step: Find forking tokens…; tuningfork forking-tokens <run> <k>"),
    # --- refining a skill
    Term("refinement", "A skill improved at one step: a target declared first, then proposed edits, each screened "
         "against the original wording, validated with full runs, and approved by a person.",
         "refining", see=("proposal", "screen", "validation", "approval"),
         where="at a step: Refine the skill here…; Refine page; tuningfork refine"),
    Term("checkpoint", "The run and step a refinement studies: the first step where the model went wrong. Everything "
         "before it is replayed as recorded — the model's answers and what its tools did to the files — so an edit "
         "is measured only from there on.",
         "refining", see=("refinement", "check along")),
    Term("proposal", "An edit of the skill for a refinement — at most three edits, each a line replaced by its "
         "number, lines added after one, or exact text replaced, changing at most ten lines in all — with who "
         "proposed it and why.",
         "refining", aka=("edit",), see=("proposer", "screen")),
    Term("proposer", "Who proposes edits: a person, an agent over MCP, or the built-in proposer — the proxy's own "
         "model, or an endpoint you name.",
         "refining", see=("proposal", "packet"), where="tuningfork connections proposer"),
    Term("packet", "The evidence a proposer is shown — from the run's own task only — kept by its digest.",
         "refining", see=("proposer",), where="tuningfork refine show <id> --packet"),
    Term("screen", "Each proposal sampled at the checkpoint against a fresh original arm, corrected for the "
         "proposals named before sampling: which edits shift the target.",
         "refining", see=("proposal", "Holm correction"), where="a refinement's page; tuningfork refine screen <id>"),
    Term("validation", "A screened proposal run in full: forks from the checkpoint to the end, fresh runs of the "
         "training task, and runs of each held-out task, verdicts only.",
         "refining", see=("held-out task", "gate"), where="a refinement's page; tuningfork refine validate <id>"),
    Term("gate", "The checks an edit must meet before it can be approved (G0–G8): it survived a screen; from the "
         "checkpoint to the end it passes at least as often as the original; its own task and the held-out tasks "
         "are not significantly worse; a held-out task the original fails improves; the skill's file is unchanged.",
         "refining", see=("validation", "approval")),
    Term("approval", "A person writing an edit into the skill's own file, after typing the code that names exactly "
         "what is written where. A backup and a changelog entry go beside the skill's folder.",
         "refining", see=("gate", "skill version"), where="a refinement's page; tuningfork refine approve <id>"),
    # --- running things
    Term("job", "Work started from the web UI or over MCP, run inside `serve` on a worker thread: a batch, a "
         "resample, a screen. A record of the work; what it made is kept on its own.",
         "working", see=("experiment", "batch"), where="Jobs page; tuningfork jobs"),
    Term("chain", "A task run N times in order, each run given the skill as the run before left it — patched "
         "by the model during the task, or by the harness's own review after it. How a harness's "
         "self-improvement accumulates, measured link by link: each version of the skill, who changed it, and "
         "the last link's folder as what the loop made. Told (`--tell`), each link's harness also gets the "
         "check's verdict and output as the user's next turn, before its review. Its runs are not "
         "independent samples.",
         "working", aka=("link",), see=("batch", "launch", "spend"),
         where="Runs → Run a task, as a chain; tuningfork chain run"),
    Term("spend", "What work cost, counted from the records: model calls, tokens in and out, and time — each "
         "call's wall time at the proxy, and the server's own where it reported it. The same count for a "
         "harness's runs, side threads included, and an experiment's samples, so two arms' budgets compare.",
         "working", aka=("budget", "cost"), see=("chain", "experiment"), where="tuningfork spend"),
    Term("breakpoint", "A condition on model calls — before the model sees a step, or after it answers with a "
         "matching tool call — that holds the call where it is met.",
         "working", see=("hold",), where="Live → Add a breakpoint; tuningfork break add"),
    Term("hold", "A model call stopped at a breakpoint: look at it, edit it, continue, step, or drop it.",
         "working", aka=("held call",), see=("breakpoint",), where="Live page; tuningfork hold"),
    Term("MCP", "The Model Context Protocol: how an agent such as Claude Code calls tuningfork's operations as "
         "tools — reading evidence, proposing edits, starting experiments; never approving.",
         "working", see=("proposer",), where="tuningfork mcp"),
    Term("access token", "The proxy's key for changing anything: minted by each `serve` and written to "
         "`<home>/.access-token`. Reading needs none.",
         "working", aka=("token",), where="tuningfork token"),
    # --- how results are judged
    Term("pass rate", "The share of a skill version's runs whose check passed, always with its interval; runs that "
         "timed out or were edited at a breakpoint are left out and counted apart.",
         "statistics", see=("Wilson interval", "check")),
    Term("Wilson interval", "The 95% interval given with every rate: where the true rate plausibly lies, given how "
         "few samples were drawn. Twenty samples still leave it wide.",
         "statistics", aka=("interval", "CI"), see=("pass rate",)),
    Term("Fisher's exact test", "Whether two arms' rates differ by more than chance would explain, at their sample "
         "sizes: the p-value an A/B and a screen report.",
         "statistics", aka=("Fisher", "p-value"), see=("A/B", "screen")),
    Term("Holm correction", "Testing several proposals at once makes a false win likelier; Holm adjusts each p-value "
         "for the family named before sampling.",
         "statistics", aka=("Holm",), see=("screen", "Bonferroni correction")),
    Term("Bonferroni correction", "The stricter correction an A/B gets with no target declared before sampling: "
         "every behaviour is tested, and the result is exploratory.",
         "statistics", aka=("Bonferroni", "exploratory"), see=("target",)),
    Term("low-powered", "Too few runs to tell a small difference from none. \"Not significantly worse\" then means "
         "\"not measured\", and is labelled so, with how many runs would detect a difference.",
         "statistics", see=("gate", "Wilson interval")),
)


def lookup(query: str | None = None) -> list[Term]:
    """Every term — or those a word names: its term or another name exactly first, then any
    whose term or other names contain it, or whose definition has a word starting with it."""
    if not query or not query.strip():
        return list(TERMS)
    wanted = query.strip().lower()
    # A word's start: "fork" finds a fork, not every mention of tuningfork.
    in_words = re.compile(rf"(?<![\w-]){re.escape(wanted)}", re.IGNORECASE)
    exact = [term for term in TERMS if wanted in (term.term.lower(), *(aka.lower() for aka in term.aka))]
    partial = [
        term for term in TERMS if term not in exact
        and (wanted in term.term.lower() or any(wanted in aka.lower() for aka in term.aka) or in_words.search(term.means))
    ]
    found = exact + partial
    if not found:
        raise OperationError(f"no term matching {query!r}", hint="`tuningfork glossary` lists them all",
                             code="not_found", field="term")
    return found


def glossary() -> dict[str, Any]:
    """The whole glossary, grouped as a reader meets it."""
    return {"groups": [{"id": key, "title": title} for key, title in GROUPS],
            "terms": [term.to_json() for term in TERMS]}
