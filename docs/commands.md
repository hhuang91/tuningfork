# Commands

Every command, grouped by the question it answers, with its main options. Each takes `--help`,
which gives them all. For a walk through them on a real task, see
[the guide](../examples/README.md).

- [How commands name things](#how-commands-name-things)
- [Does my skill work?](#does-my-skill-work) — `batch`, `tasks`, `runs`, `recheck`, `label-next`, `chain`
- [What happened in a run?](#what-happened-in-a-run) — `show`, `timeline`, `step`, `files`, `diff`, `spend`,
  `import`, `scenario`
- [Where do the runs go wrong?](#where-do-the-runs-go-wrong) — `divergence`, `tally`, `check-along`, `where`
- [Bad luck, or the wording?](#bad-luck-or-the-wording) — `resample`, `ab`, `fork`, `experiments`
- [Which token decided it?](#which-token-decided-it) — `tokens`, `forking-tokens`
- [What is the model doing right now?](#what-is-the-model-doing-right-now) — `break`, `hold`, `pause`, `resume`
- [How do I fix the skill?](#how-do-i-fix-the-skill) — `refine`, `skills`, `skill`, `add-skill`, `forget-skill`,
  `connections proposer`
- [Setting up, and keeping things tidy](#setting-up-and-keeping-things-tidy) — `serve`, `ui`, `demo`, `token`,
  `connections`, `probe`, `servers`, `jobs`, `trash`, `glossary`, `mcp`, `version`

## How commands name things

- **Recordings.** Commands read and write `~/.tuningfork`, or the folder `TUNINGFORK_HOME` or
  `--home` names.
- **A running proxy.** The commands that control one — breakpoints, holds, jobs, connections,
  `pause` — find it at `TUNINGFORK_PROXY` (default `http://127.0.0.1:8787`), or `--proxy`.
- **A run** is named by its id, a prefix of it, or its label (`tally-report/1#2cc8149f.7`).
  `runs -q` prints ids alone, for a script: `RUN=$(tuningfork runs -q -n 1 --called write_file)`.
- **A step** is named by its number, or by what the model did there. `write_file` is the first
  step where it called that tool, and `write_file|execute_code` the first where it called either;
  `changed:report.json` is the step whose call first changed that file, and `made:<words>` the
  one whose call first made a finding of the task's check (once `check-along` has run). A number
  is the same number in every run; a name finds each run's own such step, wherever its path put
  it.
- **Rates** come with their 95% Wilson interval. An A/B, a screen and a forking-token probe test
  a target declared before the first sample: `--target <letter>` (a behaviour from the step's
  latest resample), `--target-match <regex>`, or `--target-rules '<json>'`, conditions that must
  all hold (`calls`, `not_calls`, `contains`, `not_contains`). An A/B without one is exploratory:
  every behaviour is tested, and the tests are Bonferroni-corrected.
- **Samples** go straight to the inference server — `--base-url`, by default `TUNINGFORK_UPSTREAM`,
  else `http://127.0.0.1:8081`; point it at the proxy to share the proxy's gate — and each prompt
  is processed whole: `--prompt-cache` lets the server reuse its cache, which is faster and, on
  llama.cpp, less repeatable. Each experiment draws a random seed base and records it;
  `--seed-base` draws an earlier experiment's samples again. Keep `--concurrency` at 1 on a
  shared server.

## Does my skill work?

### `batch`

Runs a task N times through a headless harness, each run in a fresh workspace and a fresh home,
and judges each with the task's check. It prints each run's verdict, the pass rate with its
interval, and where the runs diverged.

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness claude-code
```

- `-n N` — how many runs (default 5).
- `--harness hermes|claude-code` — the harness (default `hermes`).
- `--eval <id>` — which task of the spec to run.
- `--model <id>` — the model the harness asks for; by default, the one the server lists.
- `--skill <file>` — that file as the skill's `SKILL.md` in every run: how to confirm an edit
  with full runs.
- `--skill-dir <folder>` — a whole skill folder instead of the spec's: a chain's final skill,
  evaluated.
- `--concurrency N` — runs at once (default 1); the proxy's `--max-inflight` still caps the calls
  that reach the server.
- `-o key=value` — an option the harness takes, recorded with each run. Hermes takes
  `self_improvement`, `await_review`, `review_every_turn` and `mcp_servers`
  ([what each does](adapters/hermes.md#options-a-launch-takes)).
- `--include-held-out` — runs a held-out task named with `--eval` anyway, with a warning: its
  runs are then recorded where agents reading the recordings can see them.

### `tasks`

The pass rate of each task: one row per harness, model, version of the skill and server, since
runs under another wording or model are another sample, never pooled. `--task <id>` shows one.

### `runs`

The recorded runs, newest first. A run recorded live — a harness's calls through the proxy, with
no batch launching it — has no label unless you gave it one ([`label-next`](#label-next)), and
without one is listed as `(recorded live)`.

- `-n N` — how many (default 20); `-q` — ids only.
- `--task <id>` — one task's runs.
- `--called <tool>` — only runs where the model called that tool; `--at-step k`, at that step.
- `--failed`, `--passed` — only runs the task's check failed, or passed, among those counted.

### `recheck`

Tries a check before you trust it: judges a recorded run again — its workspace as it ended, and
its final reply — or a reply and files you give, in a throwaway folder. It records nothing, and
exits 1 when the check fails.

- `--check '<command>'` — the shell command to try; with a run and no check named, the run's own.
- `--reply-contains <text>`, `--reply-matches <regex>` — a check of the reply instead.
- `--reply <text>` and `--file NAME=PATH` — without a run: the reply to judge, and files to put in
  the workspace.
- `--timeout <seconds>` — how long the check may take (default 60).

### `label-next`

Labels the next run recorded, so you can find it later: `tuningfork label-next my-try --task
tally-report/1`.

### `chain run`, `chain list`, `chain show`

`chain run <spec> -n N` runs a task N times in order, each run given the skill the one before it
left: what the harness's own self-improvement makes of the skill. Its options — `--tell`,
`--budget-seconds`, and the Hermes options its loop needs — are explained in
[agents](agents.md#a-harnesss-own-self-improvement-for-comparison). `chain list` lists the chains
kept; `chain show <id>` gives each link's verdict, who changed the skill after it, and the
versions (`--diff`: the lines each link changed in `SKILL.md`).

## What happened in a run?

### `show`

A run's steps, and what the model did at each: its outcome, how it finished, its time and
tokens, and how long the model reasoned first (`think`, in characters).

### `timeline`

The run as one conversation: its last context read once, with each step's answer where it
happened, reasoning included. It opens with the tools the first call offered, which no message
holds though the model reads them. `--full` prints whole messages, and each tool's first line;
`--json` gives what the web UI reads.

### `step`

One step: the exact context the model saw, and what it did. `--full` prints whole messages.
`--harness-view`, for a step edited at a breakpoint, shows what the harness sent and received
instead of what the model saw and produced.

### `files`

The workspace as a step's call found it, with every tool effect before it on disk:
`tuningfork files <run> write_file`. `start` and `end` name the workspace as the run began and as
it left it.

- `--diff` — what changed: before a step, since the step before; at `end`, since the start.
  `--path <file>` limits it to one file.
- `--out <folder>` — writes the snapshot's files out.

### `diff`

Two runs, step by step: `tuningfork diff <runA> <runB>`.

### `spend`

What work cost, counted from the records: model calls, tokens in and out, and time. Name the
work with `--run`, `--experiment`, `--job`, `--chain` or `--task`.

### `import`

A conversation an agent had outside tuningfork, as a run: `tuningfork import session.json`. Each
assistant turn is a step, its request rebuilt with everything before it as its context, so
`resample`, `ab`, `forking-tokens` and `refine` work at its steps. It reads a Hermes session file
(`~/.hermes/sessions/session_<id>.json`: its system prompt, tools and messages), Hermes's export
(`hermes sessions export`, one session a line, without tools), or an OpenAI-style list of
messages. It says what it found — the steps, the skills the conversation shows and from which
step each is in context — and what the file lacked, before recording anything. The web UI's
Runs page has the same: **Import a chat…**.

- `--session <id>` — from a file of several sessions, the one to import.
- Every skill the conversation shows is kept with the run, its SKILL.md extracted from what the
  model read; `--skill-dir <folder>`, once for each skill that has one, keeps the skill's own
  folder instead, its scripts and references too.
- `--skill <name>` — where it shows several, the one studied when none is named. Otherwise an
  A/B, a refinement or a scenario names the one it studies when it starts (`ab --skill`,
  `refine start --skill`, the forms' **Skill**).
- `--tools-from <run>` — where the file names no tools, a run recorded through the same harness
  whose tools each step offers.
- `--scenario <name>` — makes a scenario of it too: runs of the skill through its harness, each
  continuing the conversation. The user turn before the skill first came in is the prompt, and
  everything before it is each run's context (`tuningfork.context`, its `context.json`), so the
  skill enters each run live and a variant replaces it everywhere. Where the chat loaded another
  skill first in that same turn, its steps up to the studied skill — that skill's load and
  results among them — follow the prompt, taken as done; where it went on to another skill
  after, each run stops where the model loads one (`tuningfork.stop_at_next_skill`), the check
  judging the workspace then. `--cut <message>` starts it from another user turn, where the skill
  never came in or came preloaded (see [`scenario`](#scenario)); `--check '<command>'` or
  `--reply-contains <text>` gives it its check, which is yours to write. Any run's page has the
  same: **Make a scenario…**.
- `--dry-run` — shows what it would import, and records nothing.

The answers in it are the model's that had the conversation: an experiment at a step samples
your server's model, given that conversation. Nothing launched an imported run, so it cannot be
forked to the end; its scenario's runs, which tuningfork launches, can.

### `scenario`

Any run as a scenario: `tuningfork scenario <run> <name>` makes a task of its conversation —
runs of its skill through the harness, each continuing the conversation from one of its user
turns — whether tuningfork launched the run, recorded it as it happened through the proxy, or
imported it. As with `import --scenario`, the user turn the skill first came in after is the
prompt and everything before it each run's context.

- **The starting prompt.** That turn's words as the chat had them, or yours: `--prompt <text>`
  — a file the chat named on another machine, say, where the scenario's inputs put it.
  `--cut <message>` starts it from another user turn (`tuningfork timeline` shows the turns),
  only where the skill never came in or came preloaded: a run starting before the turn it came
  in is never sent that turn, and one starting after has it in its context as the chat had it,
  where a variant never reaches it. What it prints says the step each run's first call stands
  for, and the turn's steps it takes as done.
- **The skill.** `--skill-dir <folder>` names the skill's own folder, which the scenario tunes.
  By default it is the folder the run's skill came from — the one an import named, or a launched
  run's task's — else a copy kept in the scenario's own folder (`skill/<name>/`): the copy the run
  kept, or the SKILL.md the model read. `--skill <name>` chooses where the run shows several.
- **The check.** `--check '<command>'` or `--reply-contains <text>`. A launched run's is its
  task's own by default, and what that check reads from the task's folder
  (`$TUNINGFORK_SPEC_DIR/…`) is copied beside the scenario's spec, where no run sees it;
  `--no-check` leaves it without one.
- **The input files.** A launched run's workspace as the scenario's first call finds it — at the
  start, or after the steps it takes as done — from the snapshots tuningfork took as the run went.
  A run tuningfork did not launch kept no files: what is made names those its conversation's calls
  name, for you to add on the scenario's page.
- Its eval is named by the scenario, so its runs are a task of their own (`<skill>/<scenario>`),
  never counted with the task of the run it came from.
- `--harness <name>` — the harness the conversation was had with, where the run does not say.
- `--dry-run` — shows what it would make, and writes nothing.

The web UI has the same, from a run's page (**Make a scenario…**) or from the Scenarios page
(**From a run…**, which lists the runs to choose from, and imports a chat first where it is not
a run yet). Its form sets each run beside the chat, for the skill chosen: first what each run is
given — its harness's own system prompt and tools (the chat's shown, never sent), the
conversation before the starting prompt, and the prompt, which can be rewritten there — then the
conversation's steps from there on: already done, where each run starts, where it stops. Claude Code's calls cannot yet carry a conversation before the prompt, so a
scenario of a Claude Code run starts at its first turn.

## Where do the runs go wrong?

### `divergence`

The first step where runs of the same task stop agreeing: each behaviour there with its rate and
interval, and how the runs that took it fared ("then passed").

- `--task <id>` — every run of a task (`--limit`, default 20), or name runs by id instead.
- `--normalizer <name>` — compare under that normalizer only: `tool_name`, for instance, compares
  the tools called and ignores their arguments.

### `tally`

What the checks said, counted. Each line a check prints is one finding, and lines that differ
only in their numbers are one; the most common finding comes first, with how many failed runs
said it: the failure to fix first, not the first one read.

- `--task <id>` — across a task's failed runs, by condition.
- `--screen <experiment>` — across each arm's failed forks, for a screen or an A/B run to the end.
- `--show N` — findings shown per group (default 10).

Once the check has been run along the runs (`check-along`), each finding also names the call
that made it in each run.

### `check-along`

Runs a task's check along its runs: on each workspace as it started, after each call that
changed it, and at its end. A finding there from the start is work not done yet; one a call made
is a mistake, made at that call. It is kept with each run, and leaves the run's verdict as it
was. Only for a check that reads the files: a reply check has nothing to read before the reply.

- `--task <id>` (`--limit`, default 50 runs), or `--run <run>`.
- `--again` — run it anew where it ran before; `--timeout <seconds>` (default 60).

### `where`

One question asked of every run of a task, lined up with the verdicts: in how many passing and
how many failing runs it held, by Fisher's exact test, with the step that names the call in each
run — a refinement's step. Where a check prints only its verdict, this finds the call that made
the difference.

- Of the model's answers: `--match <regex>` over what each step did — a call's name and
  arguments, a script's code, a reply's words — or `--rules '<json>'`, as for a target. It gives
  the steps whose answer satisfies it, and the first of them.
- Of what the runs wrote: `--file <path or glob>` with `--contains <regex>` (`--at-least N`
  matches; without `--contains`, the file must exist). It gives whether it held at the start,
  which call made it hold, which undid it, and whether it held at the end.
- `--task <id>`, or `--screen <experiment>` for a screen's forks, arm by arm. `--limit` (200 runs)
  and `--show` (20 listed per condition).

## Bad luck, or the wording?

### `resample`

Sends one step to the model again N times, its history held fixed, and clusters what it does:
each behaviour with a letter, its count, its rate and its interval.

```bash
tuningfork resample <run> write_file -n 20
```

- `-n N` — samples (default 20). `--max-tokens N` — a cap per sample (default 4096); a sample
  that hits it counts as truncated.
- `--temperature`, `--top-p`, `--top-k` — override the recorded sampler; recorded as a change.
- `--logprobs K` — keep each sample's tokens, with K alternatives each, on chat-completions steps:
  `tokens --sample` shows one. Where the server sends none back, the experiment says so; started
  through the proxy, it is refused on a server the check found without them.
- `--to-end` — each sample a fork run to the end through the harness, counted by the task's check
  (`--proxy`; `--spec` for a run recorded before tuningfork kept how it was launched).
- `--normalizer`, `--seed-base`, `--prompt-cache`, `--base-url`, `--concurrency` — as
  [above](#how-commands-name-things).

### `ab`

Swaps an edited skill in at a step and tests whether the model's behaviour shifts: N samples an
arm, both from the same history. tuningfork finds the skill's text in the recorded request — in
a message, or in the system prompt where Hermes preloads it — and replaces exactly that text.

```bash
tuningfork ab <run> write_file --variant examples/variants/tally-report/SKILL.v2.md --target-match row_count
```

- `--variant <file>` — the edited skill (required). `--original <file>` — the skill as it was
  recorded; by default, the copy a batch run was given. `--skill <name>` — for an imported chat
  that kept several skills, the one the variant replaces.
- `--target`, `--target-match`, `--target-rules` — the target, declared before sampling;
  `--alpha` (default 0.05), kept with the A/B and the level it is judged at whenever it is shown
  again.
- `--to-end` — each sample a fork run to the end, both arms compared on the task's check: the
  confirmation a checkpoint A/B asks for.
- `-n N` (default 20 an arm), and the sampling options of `resample`.

### `fork`

Replays a run up to a step, then lets the model take over. The harness is relaunched on the same
task in a fresh workspace, and its calls are answered from the recording — it re-runs every
recorded tool call itself — until the call at the step, which goes to the model. Armed
breakpoints apply from there: a restart from any step of a recorded run.

- `--variant <file>` — an edited skill, swapped in from the step on (`--original` as for `ab`).
- `--no-breakpoints` — let the live calls run without stopping.
- `--spec <file>` — for a run recorded before tuningfork kept how it was launched.

### `experiments`, `experiment`

`experiments` lists them, newest first (`--run`, `-n`, `-q`). `experiment <id>` shows one's
result again, drawing no new samples: `--normalizer` analyses its samples again under another
normalizer, `--alpha` judges an A/B at another level than the one it was declared with, and says
so. An A/B's starts with the edit it tested, as a line diff of the two texts its arms sent the
model.

## Which token decided it?

These need llama.cpp's own endpoints, and a model whose raw output tuningfork reads: Gemma 4,
Qwen2.5, Qwen3, Qwen3.5, Qwen3.6 or Qwen3-Coder. The step's prompt must have the token count it had when it was
recorded, or the work is refused.

### `tokens`

A step's answer as the model's tokens: how sure the model was of each, and where the step's own
sampler could most easily have drawn another.

- `--score` — score the answer the run gave first: its exact context sent once more with the
  answer forced, reading the model's distribution at each token (`--top-k` alternatives kept,
  default 8).
- `--sample EXPERIMENT:INDEX` — a sample's tokens, from a resample run with `--logprobs`;
  `--stream <id>` — a stream by its id.
- `--hotspots N` (default 10), `--color auto|always|never`, `--json`.

### `forking-tokens`

Which tokens decided a step, after Bigelow et al. ([ICLR 2025](https://arxiv.org/abs/2412.07961)):
at the tokens where the step's own sampler would most likely have drawn another, force each
likely alternative, let the model go on, and test which change what it does. Each alternative is
compared with the run's own token at the same position, Holm-corrected over the family planned
before sampling.

- `--in answer|reasoning|all` (default `answer`), `--top N` (default 5) and `--min-p` (default
  0.05) — where to look, how many positions to probe, and the least chance an alternative needs
  to count as a fork; or `--at 12,40` — positions as `tokens` numbers them, probed as named, and
  then those three are not used (the plan says so if one was given).
- `--alternatives K` (default 3), `-n N` continuations per forced token (default 10).
- `--target`, `--target-match` — the behaviour to test for; by default, what the run did.
- `--score` — score the answer first, if it has no tokens yet. `--dry-run` — show the plan; draw
  nothing.

## What is the model doing right now?

These control a running proxy. To the harness, a call held at a breakpoint is just a slow
response.

### `break add`, `break list`, `break rm`, `break on`, `break off`

`break add` stops a live call, before the model sees it or after the model has answered and
before the harness acts. Every condition given must hold. With a condition on the answer it
stops after the model; otherwise before (`--phase` says it outright).

- On the answer: `--tool <name or glob>`, with `--path <glob>` on a path argument or `--args
  <regex>`; `--output <regex>` over its text, reasoning and calls; `--no-reasoning` — a tool call
  made without reasoning first; `--diverges-from <run>` — where it differs from that run's.
- On the request: `--after-tool <tool>` — before the model sees that tool's result; `--context
  <regex>` over the request.
- Where: `--step k`, `--from-step k`, `--run`, `--task`, `--launch`. And `--once`, `--label`.

`break list` shows them; `rm`, `on` and `off` take a breakpoint's id.

```bash
tuningfork break add --tool write_file --path '*.json'   # after the model: stop that call
tuningfork break add --from-step 2                       # before the model sees step 2
```

### `hold wait`, `hold list`, `hold show`, `hold continue`, `hold step`, `hold drop`

- `hold wait` — blocks until a call is held, then shows it (`--timeout`, default 300 s).
- `hold list` — the calls held now (`--recent`: and recently decided; `-q`: ids only).
- `hold show <id>` — one held call; `--json` prints just the editable object, the request or the
  response.
- `hold continue <id>` — lets it go on; `--edited f.json` sends your edit of that object instead.
  The step records what the model produced, with the edit beside it, and a run with an edited step
  is kept out of every pass rate.
- `hold step <id>` — goes on, and stops again at the next call.
- `hold drop <id>` — fails the call; the harness receives an error.

### `pause`, `resume`

`pause` stops sending model calls to the server: they fail fast until `resume`.

## How do I fix the skill?

A *refinement* is a folder in the recordings, written once wherever it can be: the step, the
run's copy of the skill, the target, and every proposal with who proposed it, why and the
evidence they were shown. [Part 4 of the guide](../examples/README.md#part-4--let-tuningfork-propose-the-fix)
walks through it from the target to the approval.

### `refine start`

Starts refining a run's skill at one of its steps, toward a target declared now, before anything
is proposed or sampled: `--target <letter>`, `--target-match`, `--target-rules`, or
`--target-check` — the task's own check, for a failure only the check can see, whose screens run
forks to the end. `--skill <name>` names the skill to refine, for an imported chat that kept
several.

```bash
tuningfork refine start <run> write_file --target-rules \
  '[{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]'
```

### `refine propose`

Proposes an edit of the skill: a hypothesis until a screen tests it. A proposal is at most three
exact edits changing at most ten lines, never the frontmatter, and it must change what the model
saw at the step.

- `--file SKILL.md --why '…'` — your own wording: the whole skill, edited by hand (a whole text
  that changes more only warns).
- `--builtin` — asks the built-in proposer, a model, for edits from the evidence (`-m N`, default
  3, at most 6). Unless `connections proposer set` names an endpoint of its own, it asks the
  inference server — `--base-url`, by default `TUNINGFORK_UPSTREAM`; point it at a running proxy to
  share its gate — and its call is never recorded. `--no-task` leaves the task's criteria out of
  what it is shown; `--dry-run` says where the evidence would be sent, and how much, sending
  nothing.

### `refine screen`

Screens proposals at the step, each against a fresh arm of the original drawn first: Fisher's
exact test, Holm-corrected over the proposals named before the first sample. Before it samples,
it says what an arm would need to survive, and it refuses to run when nothing could.

- `--proposals a,b` — the family the correction counts (by default every proposal, if there are
  at most 6); `-n N` — samples an arm, the original's included (default 20); `--alpha`.
- `--to-end` — each arm's samples forks run to the end through the harness, the task's check
  deciding. They are drawn from several runs of the task at the same step (`--checkpoints`,
  default 4), passing and failing by turns, and the report says whether the effect held at every
  one. The original's forks an earlier screen drew there count again (`--fresh-original` draws
  them anew). `--quick` forks each proposal 8 times and the original not at all: rates beside the
  original's earlier forks, never a verdict.
- `--dry-run` — the plan, and what could survive; nothing sampled.

### `refine validate`

Validates a screened proposal with full runs, each arm against a fresh arm of the original:
forks from the step to the end (does hitting the target carry the task?), full runs of the task
with the edit from the first step (does it disturb an earlier step?), and full runs of each
held-out task, kept only as verdicts (does it break what it was not made for?). Its record is
what the gate reads: nine checks, G0–G8, each with its counts.

- `-n N` forks an arm and `--runs N` full runs an arm (each default 5, at least 5).
- `--keep-private <folder>` — keep the held-out runs' private recordings there, to debug them;
  an agent that can read files can then read them too.
- `--dry-run` — the plan; nothing run.

### `refine approve`

Writes a validated proposal into the skill's own `SKILL.md`, in place — the only thing in
tuningfork that does. It shows the file it will write, what it keeps, the gate and the edit, and
writes nothing until you type the code it shows. The old file goes into `<skill>.history/` and an
entry into `<skill>.changelog.md`, beside the skill's folder and never inside it, since every run
copies that folder. The file keeps its line endings and byte-order mark.

- `--validation <id>` — the validation it stands on (by default the proposal's newest finished
  one); `--spec`, `--target` — when the spec has moved, or leads elsewhere; `--confirm <code>`.

### `refine show`, `refine list`, `refine close`

`refine show <id>` gives the target, the proposals as line diffs with who proposed each and why,
what each screen found, each validation with its gate, and the approval. `--packet` prints, as
JSON, the evidence a proposer would be shown now; `--repair` finishes an approval a crash
interrupted. `refine list` lists the open refinements (`--all`: closed ones too); `refine close`
closes one, keeping everything it holds.

### `skills`, `skill`

`skills` lists the skills the recordings know — from runs, scenarios, saved variants and
refinements, and those added with `add-skill` — each with its tasks, runs and versions. `skill
<name>` gives each version with its pass rate under every condition it ran under, and what its
text is known as, the scenarios that run it, and how its folder was added, where it was; `--text
<ref>` prints one text (a version's fingerprint, `file`, `variant:<name>` or `history:<file>`),
and `--diff A B` compares two line by line.

### `add-skill`, `forget-skill`

`add-skill <folder>` adds a skill by its folder on this machine, remembered where it is: nothing is
copied, so an approved edit is written to that folder. `--copy` keeps a copy in the recordings
folder instead (`skills/<name>/`, `--name` naming it), as the Skills page's upload does: an approved
edit is then written to the copy, never back to where it came from. `--run <run>` keeps the skill a
run had as such a copy — the copy the run kept of it, else the SKILL.md its model read — `--skill
<name>` choosing where the run had several. `--dry-run` says what would be added. The Skills page
then lists it, and a scenario names its folder to run it. `forget-skill <name>` forgets one: a
folder named stays where it is; a copy is moved aside whole, to `skills/.forgotten/`, never deleted,
and not while a scenario runs from it.

### `connections proposer show`, `connections proposer set`

The built-in proposer's settings, and whether a key is set — never the key. `set` changes them:
`--url` (`none` for the proxy's own server), `--wire openai-chat|anthropic-messages`, `--model`,
`--max-tokens`, `--temperature`. An endpoint that is not this machine needs `--allow-remote`: the
evidence it is sent holds recorded prompts, answers and your skill. Its key is read from stdin
(`--key-stdin`, never an argument, which `ps` shows), or from `TUNINGFORK_PROPOSER_KEY`.

## Setting up, and keeping things tidy

### `serve`

The recording proxy: point a harness at `http://127.0.0.1:8787/v1`. It forwards to the inference
server, records every call, serves the web UI and the MCP route, and runs the jobs the web UI and
agents start. [Setup](setup.md#start-tuningfork) lists its options. Pointed at a hosted model's
API, it says in its first lines that tuningfork is built for a model you run yourself, and what
works there ([cloud models](setup.md#cloud-models)).

### `ui`

The web UI over recordings, read-only, with no proxy running (`--home`, `--port` default 8788).
With a proxy running, use its own UI instead: it adds live breakpoints and starts work.

### `demo`

tuningfork with no GPU and no model. `demo` (or `demo lab`) serves made-up runs of the guide's
example task through a real proxy in front of a scripted model, so every button in the web UI
works. At a step of the made-up runs, the model writes the report's keys only when the skill
names them in the words `row_count and amount_total`; in a run the demo's own harness makes, it
reads `data.csv` first, then writes them whenever the skill names `row_count`. `--port` (default
8790) moves it. `demo seed` writes the runs alone, for the terminal. `demo proxy` is the proxy
alone, with no runs, in front of a model that always writes the wrong keys, and `demo live` the
same with a call held at a breakpoint.

Each records into a new folder in the system's temporary folder, unless `--home` names one; the
folder is kept after it stops. Its runs are made up under that task's own id (`tally-report/1`),
so it records only into a folder it made or an empty one, never where tuningfork keeps
recordings by default, and it seeds a folder only once. It refuses a `--port` something listens
on already. [Part 1 of the guide](../examples/README.md#part-1--tuningfork-without-a-model)
walks through `demo seed`, `demo` and `demo proxy`.

### `token`

The running proxy's web UI address, with the access token that lets it act. `--raw` prints the
token alone; `--mcp` prints the MCP token, and `--header-json` the header an MCP client sends.

### `connections show`, `connections switch`, `connections harness`, `connections check`

- `show` — the server the proxy forwards to, what was measured of it, and the harnesses found;
  for a hosted model's API, that tuningfork is built for a model you run yourself.
- `switch <url>` — points the proxy at another server, while nothing is using it (`--dry-run`).
  A hosted API that asks for a key can be switched to, for recording; the plan says what it is.
- `harness <name> --executable <path>` — names the program a harness runs as, kept in the
  recordings folder and never in the harness's own configuration (`--reset`: the one on `PATH`).
- `check` — measures what the harnesses and the experiments need of the server: tool calls in a
  stream, the Messages endpoint, every prompt processed whole, and what token work needs —
  llama.cpp's raw completion for scoring an answer and forking tokens, the chat endpoint's
  logprobs for a resample's tokens. Token work it found the server without is refused through
  the proxy, saying what was missing, and greyed in the web UI; what it never measured is not.

### `probe`, `servers`

`probe <url>` asks a server what it can do and records the answer (`--out profile.json`); its
probes run one at a time, so it is safe on a server in use, and the profile it saves leaves out
home folders, so it can be shared. `servers` lists the servers that answered recorded calls, and
what each said it was.

### `jobs list`, `jobs show`, `jobs cancel`, `jobs wait`

Work a running proxy is doing, started from its web UI or by an agent. `list` and `show` say who
started each, when its front end said: an agent over MCP and its client (`mcp · claude-code`), or
the web UI and the page it was started from (`web · step`). `show <id>` says what a job was asked,
how it went and what it made; `cancel <id>` stops it, keeping what it finished; `wait <id>` blocks until
it ends, printing its progress, and exits 1 unless it finished.

### `trash`

`trash run <id>…` sets runs aside — and `trash experiment`, `trash job`, `trash scenario` and
`trash refinement` the others — out of every list and every rate, with what is tied to them left
in place, and no new work started on them. `trash list` shows what is there and what deleting
each for good would remove; `trash restore <id>` brings items back. `trash purge <id>` deletes items for good with everything tied to them — a run's
experiments and their samples, forks that replayed it — and `trash empty` deletes everything
there. Neither can be undone.

### `glossary`, `version`

`glossary [word]` gives tuningfork's words, each defined once, as the web UI's Words page shows
them: where the pages say a word more plainly ("wording" for an arm, "test at this step" for a
screen), the word's entry says that too, and the plainer phrase finds the word
(`tuningfork glossary "full test runs"`). `version` prints the version.

### `mcp`

Serves tuningfork's operations to an agent over MCP, on stdio. See [agents](agents.md).
