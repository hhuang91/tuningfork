# Tuning Fork

A step debugger and experiment lab for agent skills running on local LLMs.

> Status: **pre-alpha, 0.2.0.** Recording, divergence localization, resampling from a
> checkpoint, wording A/B tests, live breakpoints, workspace snapshots, restarting a
> recorded run from any step to measure it end to end, a web UI that starts all of it,
> token-level analysis — how sure the model was of each token, and which token decided a
> step — all of it as tools for an agent over MCP — and refining a skill at a step, with
> edits proposed by you, an agent or a model, each screened against the original, validated
> with full runs on held-out tasks, and written to the skill only when a person approves,
> all work, with Hermes Agent and with Claude Code.
>
> **New here?** [`examples/README.md`](examples/README.md) walks through everything below,
> command by command: first a five-minute tour that needs no GPU, then the real experiment.
> [`examples/hermes-design-md/`](examples/hermes-design-md/README.md) does it on a real skill,
> Hermes Agent's own `design-md`, from the failing step to an approved fix and the runs after.

## The problem

You write a skill (an [agentskills.io](https://agentskills.io)-style `SKILL.md`) for a local
agent. It works three times out of five. So you reword it, run the task five more times, and
squint at the results.

That loop is broken in two ways:

- **Small samples are noise.** 3/5 is consistent with a true success rate anywhere from
  23% to 88% (95% Wilson interval). Separating a 60% skill from a 90% one takes roughly 30+
  runs per variant. Almost nobody does that, so wording decisions get made on noise.
- **Wording sensitivity is real and model-specific.** Open models swing hard on
  meaning-preserving format changes, and the best format does not transfer between models
  ([Sclar et al., ICLR 2024](https://arxiv.org/abs/2310.11324)). Every model swap or
  quantization change means partial retuning.

## The idea

Treat an agent run as a program execution and the skill as its source code, then supply the
missing debugger:

- **Record** every model call at the proxy, with the exact context the model saw.
- **Break and inspect** — pause before or after any call, read and edit the full context.
- **Fork and resample** — re-run step *k* N times from a checkpoint and cluster the outcomes.
- **Localize** — find the first step where successful and failed runs diverge.
- **A/B the wording there**, with history held fixed, and report the difference with a
  confidence interval and a significance test instead of a bare percentage.

Pausing and editing LLM traffic is table stakes; several tools do it. The point here is
**measurement and localization for instruction text**.

## How it works

tuningfork is a recording proxy that sits at the model-server seam. It speaks both OpenAI
chat-completions and Anthropic Messages:

    agent harness ──(OpenAI or Anthropic HTTP)──▶ tuningfork ──▶ inference server
                                                      │
                                            SQLite + content-addressed blobs
                                                      │
                                  Python API / CLI / web UI / MCP server

Model requests are stateless, so every request carries the model's complete context. The
proxy sees the full program state at every step without touching harness code, and works
with any harness that accepts a custom base URL.

### Start llama-server for numbers that repeat

> **Strongly recommended: one slot (`-np 1`), `--no-cache-prompt` (next), and, for a model with
> sliding-window attention layers — Gemma 2, 3 and 4 among them — `--ctx-checkpoints 0` or
> `--swa-full`.** A decision that hangs on a near-tie moves with how the server happens to
> process its prompt, and each of these keeps that the same from one request to the next.
>
> - **The server's history.** We measured, on Gemma 4 26B-A4B, one step's request give its
>   target behaviour 1 time in 40 right after a restart and 22 times in 40 once a similar prompt
>   had run, on a server that reused what it had cached. A screen samples each proposal right
>   after the original, so on such a server it favours the proposals. With every prompt
>   processed whole, at one slot, nothing a request gets depends on what ran before it.
> - **Another request beside it.** With two slots, a request computed beside another got
>   different numbers for the same prompt; one slot computes each request alone.
> - **Which endpoint asks.** For sliding-window layers llama-server takes rollback checkpoints
>   of their cache, which cut a prompt into batches at places that differ between the chat
>   endpoint and the raw one tuningfork scores answers with: the same answer scored 0.70 at one
>   token on one and 0.15 on the other. With `--ctx-checkpoints 0` no checkpoint is taken, and
>   the two agree token for token; it costs nothing when prompts are not cached, since a
>   checkpoint only rolls back a cached prompt. `--swa-full` takes none either, by keeping the
>   whole sliding-window cache, at a cost in memory: the 26B with two 64k slots needed 27 GiB
>   with it and 22 without, and the dense 31B at a 64k context does not fit a 32 GB card with it.
>
> llama-server reports neither `--ctx-checkpoints` nor `--swa-full`, so tuningfork cannot check
> them for you. tuningfork also asks llama.cpp to process each experiment sample's prompt
> whole (`cache_prompt: false`), where it can: on chat completions, not on its Messages
> endpoint, which Claude Code uses.

### And with `--no-cache-prompt`

> **Strongly recommended too.** By default llama-server answers a prompt from what it cached
> of an earlier one, and its Messages endpoint, which Claude Code uses, ignores a request
> asking otherwise. Then a Claude Code step's samples all share the first one's processing of
> the prompt, and forks whose prompts match — every fork, in the sandbox below — share one
> cached state. We measured one resample give a behaviour 16 times in 20; with the flag, three
> resamples of the same step gave it 1 time in 10 each.
>
> The flag makes every step process its whole prompt: about 1.5 s per 14,000 tokens of
> Gemma 4 26B-A4B on one RTX 5090, about 4 s of Gemma 4 31B, so runs take longer.
> `tuningfork connections check` measures whether the server reuses prompts, and a batch's
> plan warns when it does.

### Install bubblewrap (Linux)

> **Recommended wherever tuningfork launches a harness.** With
> [bubblewrap](https://github.com/containers/bubblewrap) (`sudo apt install bubblewrap`), every
> run, fork and held-out run is launched in a sandbox that looks the same each time: its home
> is `/home/user`, it works in `/home/user/project`, and the harness keeps its files where it
> installs them. The model reads those paths. Without the sandbox, each launch sees its own
> folders under the recordings folder, and a few digits of a fork's folder name were seen to
> decide whether the model wrote a file or only said it did: resampled at the same step, two
> forks whose requests differed only in that number wrote it 5 times in 20 and 20 in 20.
>
> Inside, the rest of the machine is read-only, `/tmp` and the process list are the launch's
> own, and the recordings folder and the task's spec folder are empty: the harness can read
> neither earlier runs nor the check. `tuningfork connections show` says whether launches run
> in the sandbox, and a batch's plan warns when they cannot. `TUNINGFORK_SANDBOX=off` turns it
> off.
>
> It keeps runs alike; it is not a security boundary. Inside, the rest of the machine stays
> readable, your home included, and the network is reachable ([`SECURITY.md`](SECURITY.md)).

## What works today

A skill that never passes. Where do runs of it stop agreeing?

```console
$ tuningfork batch examples/tally-report.evals.json -n 8 --model <model-id>
4 runs of tally-report/1 — agreed for 0 step(s), diverged at step 0
A  3/4  75%  [30%-95%]  read_file(path='data.csv')
B  1/4  25%  [5%-70%]   execute_code(code='<ast:a62cb33a>')
```

Resample the step where the skill's instructions bite — twenty times, with the history
held fixed — once as written and once with an edit swapped in:

```console
$ tuningfork ab <run> 1 --variant examples/variants/tally-report/SKILL.v2.md --target-match row_count
   behaviour                                        original              variant
A  write_file(content='{"amount_total":61.5,…       0/20   0% [0%-16%]    20/20 100% [84%-100%]
B  write_file(content='{"count":4,"total":61.5}'…  20/20 100% [84%-100%]   0/20   0% [0%-16%]

target (declared before sampling): behaviour matching /row_count/
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher
```

Four minutes of sampling, one model call per sample. Full runs agree — `tuningfork batch
... --skill examples/variants/tally-report/SKILL.v2.md` puts the edit in every run, and it
passes 8/8, against 0/5 before. `tuningfork tasks` keeps the two apart: a task run under two
wordings has a rate for each, never one pooled across them.

The same edit tested at step 0, where the model decides how to *read* the file, came back
not significant (p = 0.748) — correctly, since the edit is about output keys. A checkpoint
A/B tells you which decision an edit changed, not just whether the pass rate moved.

### From that step to the finish line

A checkpoint A/B measures one step. `--to-end` makes each sample a *fork*: the harness is
relaunched on the same task in a fresh workspace, answered from the recording up to the
step — it re-runs every recorded tool call for real, but the model is not asked — and left
to run from there to the end, where the task's check decides:

```console
$ tuningfork ab <run> Write --variant examples/variants/tally-report/SKILL.v2.md --to-end -n 20
target: the task's check passes, end to end from step 2
  original  0/20   0% [0%-16%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher
```

Forty forks through Claude Code in eleven minutes, each answered by the recording exactly as
it was up to the step — a fork whose harness asks for something the recording never answered
is left out, and says where. `tuningfork fork <run> <step>` runs a single fork with
breakpoints armed: a restart from any step of a recorded run. And every call of a run
tuningfork launched finds its workspace snapshotted: `tuningfork files <run> <step>` shows
it as that step's call found it.

### Which token decided it

A step's answer can be read as the model's own tokens. `tuningfork tokens <run> <k>
--score` sends the step's exact context once more with the answer forced, one call, and
reads how sure the model was of each token: a heatmap in the terminal and on the run's page.
At the tally-report write step it says `count` was never in doubt — p = 1.00 — so no amount
of resampling would find `row_count`: the wording has to change.

Where a step does split, `tuningfork forking-tokens` finds the token that splits it. After
Bigelow et al. ([ICLR 2025](https://arxiv.org/abs/2412.07961)): at the tokens whose runner-up
the step's own sampler would most likely have drawn, force each likely alternative, let the
model go on, and test which change what it does. A Hermes run that searched for `data.csv`
before reading it:

```console
$ tuningfork forking-tokens <run> 0 --in reasoning
[134] reasoning  …, I'll check if `data.csv` exists⟦ in⟧
  as run  ' in' 19%                10/10 100% [72%-100%]
  forced  ' and' 72%               0/10   0% [0%-28%]   p=1.1e-05 → Holm 7.6e-05  flips away from the target
  forced  '.' 9%                   10/10 100% [72%-100%]   p=1.000 → Holm 1.000

1 of 7 forced tokens changed what the model did (Holm-corrected over the 7 planned before sampling; α=0.05).
```

"Exists *in* the working directory" leads to a search; "exists *and*…", which the model
thought likelier, to reading the file. Every alternative is compared with the run's own
token at the same position, and the family of tests is fixed before sampling. This needs
llama.cpp's server and a model whose raw output tuningfork reads (Gemma 4 today); anything
else is refused with the reason.

### Stopping a run where it matters

Breakpoints stop a live call before the model sees its context, or after the model has
decided and before the harness acts. Here the flaky step is caught as it happens:

```console
$ tuningfork break add --tool write_file --path '*.json'
breakpoint 181d8b0f — after the model: write_file on *.json

$ tuningfork hold wait
hold 7f8094a1 — after the model, run 33f256ea step 1, held 0.4s
the model answered
  tool_call write_file({"content":"{\n  \"count\": 4,\n  \"total\": 61.50\n}\n", ...})
```

The harness is simply waiting on a slow response. Fix the keys and let it carry on:

```console
$ tuningfork hold show 7f8094a1 --json > held.json     # edit the arguments in held.json
$ tuningfork hold continue 7f8094a1 --edited held.json
```

The harness writes the corrected file and the run passes its check. The run is still not
counted in any pass rate, because its verdict now measures the edit as much as the skill.
The recorded step keeps what the model actually produced, alongside what the harness
received. `hold step` walks a run one stop at a time instead, and `hold drop` fails a call.

### All of it from the browser

Everything above is also in a local web UI, served by the proxy itself at
`/_tuningfork/ui/`: a run tree with pass rates and their intervals; each run as one
conversation, top to bottom, with each step's answer — reasoning included — where it
happened, and the skill marked inside it and mapped to its file line by line; each answer
as a heatmap of its tokens, a click showing what else the model weighed there; a step's
exact context on a page of its own; divergence and diff views, experiment results, and a
Live page that edits held calls in place. A tool call the model made without any reasoning —
in a run that reasons elsewhere, often right after a tool call failed — is marked wherever a
step is shown, and an experiment says how many of a behaviour's samples reasoned where some
did not; `break add --no-reasoning` stops there live.

The front page says what is set up and where you left off, draws how the pieces fit — one
loop, from a skill through its runs, the experiments at a step and a refinement, to the
skill's next version — and lists the questions each way of looking at a step answers, with
the clicks and the terminal command for each, beside a glossary of tuningfork's words
(`tuningfork glossary` in a terminal). The Skills page lists every skill the recordings know:
each version its runs were given, with its pass rates, what its text is known as (the file
now, a saved variant, a proposal, a version an approval replaced) and its SKILL.md rendered;
any two of its texts compared line by line; and the changelog an approval keeps beside the
skill's folder (`tuningfork skills`, `tuningfork skill <name>`).

And the page can start the work, not only show it. From a step on a run's conversation, or
from Start… on the Jobs page: resample it, or A/B a wording edited right there against the
skill as the run was given it, at that step or with every sample run to the end; fork
the run there and stop it at the step; score its answer, or select tokens on its heatmap
and find which of them decide it — saying what counts as the same behaviour in words (a tool
it calls, a phrase it mentions), set against everything the step has already done before
anything is sampled. A task's check can be a shell command run in the
workspace — a longer one a script beside the spec, which `$TUNINGFORK_SPEC_DIR` names, out
of the agent's sight — or a check of the agent's final reply, which needs none — and it can be tried
before it is trusted: on a recorded run of the task, whose verdict it is set beside, or on a
reply and files written for the purpose, with nothing recorded. A form
shows the plan the proxy would run before its Start button does anything, and an A/B asks
what it is about — a behaviour, a pattern, or openly exploratory — before a sample is
drawn. Scenarios keep tasks in the recordings folder, edited as the exact `evals.json` they
save to, with their input files, and run as batches. A job's page follows it as it goes; a
call held at a breakpoint can be sampled where it stands (evidence, filed under the step it
becomes) or asked about (the model's own account, labelled a hypothesis). The Connections
page shows the server and what was measured of it, the harnesses it found and their
versions, and switches the server while nothing is using it.

The page changes nothing without the proxy's access token, which `tuningfork serve` prints
in the address to open; `tuningfork token` prints it again. `tuningfork ui` serves the same
pages, read-only, over existing recordings with no proxy running. Runs, experiments, jobs,
scenarios and refinements can be moved to a trash — out of every list and pass rate until
restored — and deleted for good from there, with everything tied to them: a refinement with
its screens, its validations' runs and its proposals' saved texts, never the changelog an
approval wrote.

### Refining a skill at a step

Which failure to fix first? When a task's check prints what it found wrong, one finding a
line, `tuningfork tally --task <id>` counts every failed run's findings at once, the most
common first, each with how many failed runs said it and its interval. `--screen <id>` does
the same for each arm of a screen run to the end, so what a proposal left failing shows
beside what the original fails.

Once the failing step is found, tuningfork can gather fixes and test each one there. A
*refinement* names the step and what the model should do at it, before anything is
proposed:

```console
$ tuningfork refine start 586cf615 write_file --target-rules \
    '[{"if":"calls","value":"write_file"},{"if":"contains","value":"row_count"},{"if":"contains","value":"amount_total"}]'
```

Edits come from you (`--file`, your own wording), from an agent over MCP, or from a built-in
proposer: a model — the local one by default, through the proxy and never recorded; a cloud
API only once you allow it — shown the evidence as data. That is the step as the model saw
it and where the skill sits in it, the answers seen there that did and did not do what the
target says (here none did, and it says so), how the task's runs with this harness, model
and skill version went there, the task's own criteria, and what was tried before. A
proposal is at most three exact edits, kept with who proposed it, why, and what they were
shown:

```console
$ tuningfork refine propose e811ff0d --builtin
...
  answered in 75s, 3332 tokens · kept in proposer/20261002T204545-e4aee8
  proposal f2b5c51b: 1 line changed
    why  The agent currently uses 'count' and 'total' as keys in the JSON report, likely because
the skill's final line mentions 'just the count and the total'. ...
  proposal fa63747b: 1 line changed
...
  proposal 95ebcd3c: 1 line changed
...
```

Each is a hypothesis until a screen tests it: a fresh arm of the original wording first,
then one arm per proposal, each compared with the original by Fisher's exact test,
Holm-corrected over the proposals named before the first sample. Before it samples, a screen
says what an arm would need to survive, and it refuses to run when nothing could:

```console
$ tuningfork refine screen e811ff0d -n 20
  survives  with at least 7 of 20 — the step did it 0 times in 20 in resample 118becf9
            Holm over 3 proposals, α=0.05
...
          target                 p        Holm
original  0/20   0% [0%-16%]
f2b5c51b  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
fa63747b  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
95ebcd3c  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
```

The web UI does the same from a step's **Refine the skill here…**, and an agent with
`tuningfork mcp --tools refine`. A screen at one step is evidence, not proof: its arms
inherit the history the original wording wrote, and a wording can reach the target without
fixing the task. In a later run, an edit that told the model to skip the file and write the
training task's answers survived its screen exactly as the fix did.

So a survivor is validated with full runs, each arm against a fresh arm of the original:
forks from the step to the end, runs of the task with the edit from the first step, and runs
of each task the spec holds out (`"held_out": true` on an eval). A held-out task's runs are
recorded in a temporary folder outside the recordings folder and deleted when they end. Only
their verdicts are kept, and no agent is shown those:

```console
$ tuningfork refine validate 044cbd20 10c1b095
...
  1568ae72  done  proposal 10c1b095 — meets the gate
    forward   edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    training  edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    held out 2: edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    G0 met     survived a screen: it survived screen b28f6a16
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): significantly better (Fisher p = 0.00794)
    G5 met     held-out task 2 is not worse (low-powered): significantly better (Fisher p = 0.00794)
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 5/5, without it 0/5
```

Each check keeps its counts and the name of the rule that judged it; at five runs an arm,
"not worse" says it can see only a large drop. The cheat passed every other check: its
held-out task failed with and without it, so it was no worse. G8 refuses it: a held-out task
the original fails must pass more often with the edit, and an edit that writes in the
training task's answers leaves it failing.

Approving is the person's. It shows the file it will write, as WSL and Windows name it, what
it keeps, the gate and the edit, and writes nothing until the code it shows is typed:

```console
$ tuningfork refine approve 044cbd20 10c1b095
approve proposal 10c1b095 of refinement 044cbd20, on validation 1568ae72
  writes     /mnt/c/Users/me/tf/tally-report/SKILL.md
             C:\Users\me\tf\tally-report\SKILL.md
  keeps      CRLF line endings
  file before → /mnt/c/Users/me/tf/tally-report.history/
  changelog   → /mnt/c/Users/me/tf/tally-report.changelog.md
  the gate   every check met:
...
  code  6da89c69
Type the code to write it (anything else writes nothing): 6da89c69
written: /mnt/c/Users/me/tf/tally-report/SKILL.md
  the file before  /mnt/c/Users/me/tf/tally-report.history/SKILL.20261005T021037Z-c52cbb98.md
  changelog        /mnt/c/Users/me/tf/tally-report.changelog.md
```

The file keeps its line endings and byte-order mark. The old file and a changelog entry go
beside the skill folder, never into it, since every run copies that folder. The refinement
keeps the approval with every piece of evidence it stood on, whatever is deleted later. No
MCP tool approves, and an MCP client's token is refused where approving happens.

### For an agent: the same tools over MCP

`tuningfork mcp` serves tuningfork's operations to an agent — Claude Code, Hermes, an
OpenJarvis teacher — over the Model Context Protocol. An agent can do what a person does:
count what the failed runs' checks said, find where a task's runs split, read that step's
context and the skill lines in it, measure how often each behaviour happens there, and
A/B-test a wording it wrote.

```console
$ claude mcp add --transport stdio tuningfork -- tuningfork mcp --home ~/recordings
```

Reading recordings needs no proxy. Work that calls the model — a resample, an A/B, scoring
an answer, a forking-token probe — runs as a job inside `tuningfork serve` on the same
recordings, through its gate, and shows on its Jobs page as "from MCP · <client>". Each
answer is compact JSON opened by sentences that say it statistics first: every rate with
its 95% interval, an A/B's declared target with Fisher's p and its threshold, or that the
comparison was exploratory. Text from the recordings is data to the agent, never
instructions, and the server's instructions say so.

By default an agent is offered only work that calls the model; `--tools all` adds work that
runs a harness or a task's check on this machine. `--tools refine` offers the refinement
loop: start a refinement, read its evidence, propose edits of its own, and screen them. No
agent is offered the built-in proposer — it sends recordings to an endpoint, which is the
person's decision — and nothing offered approves an edit. A fork an agent starts never stops at a
breakpoint, and an agent's check never runs a shell command. A job may make at most 100
checkpoint samples or 20 harness runs (`--limit resample=200` changes one), agents may have
4 jobs queued or running at once, the same work asked for twice is given back rather than
started again, and an agent cancels only its own jobs. Live debugging, the trash and
switching the server stay with the person.

A client that connects rather than starts a server uses the running proxy's endpoint,
`http://127.0.0.1:8787/_tuningfork/mcp`, with the proxy's MCP token as a bearer token:
`tuningfork token --mcp` prints it, `TUNINGFORK_MCP_TOKEN` pins it across restarts, and
`tuningfork token --header-json` prints the header for a client that runs a command for its
headers, such as Claude Code's `headersHelper`. The MCP token opens that endpoint and nothing
else, so an agent's client never holds a key to what its tools leave out, such as approving
an edit; the access token, in turn, is refused there. Hermes lists the server under `mcp_servers`
in its `config.yaml` (its `mcp` extra installed); OpenJarvis under `[mcp] servers` in its
`config.toml`, by command or by URL and token. From Windows with tuningfork in WSL, the
command is `wsl.exe -d <distro> -e <venv>/bin/tuningfork mcp --home <recordings>`: the
server reads the token from the recordings folder itself, so nothing secret goes in a
Windows config. Point the agent's own model at the inference server, not at the proxy it
studies, or its calls are recorded beside the runs it is reading.

Recording is transparent: point any harness that speaks OpenAI chat-completions or
Anthropic Messages at the proxy and it behaves as it did before. If a model server on your
machine serves something that matters, protect it, and tuningfork sends it nothing — not a
resample, not a probe, not a harness's call: list it in `~/.config/tuningfork/config.json`
(`{"protected": ["localhost:1234"]}`), in `TUNINGFORK_PROTECTED`, or in a recordings folder's
`config.json`. Every message is content-addressed, so the large constant part of an agent's
context is stored once rather than once per step. Rates are never reported without the
interval they actually imply, and an A/B's target is declared before sampling — without one,
the comparison is labelled exploratory and corrected for multiple comparisons.

## Portability

Harness and server specifics live behind two seams — `harness/` adapters and `backends/`
capability profiles. Optional features (logprobs, server-side `n`, KV-cache reuse) are
enabled by a runtime capability probe, not by assuming a particular server. Running
`tuningfork probe` against your own stack produces a profile you can contribute back.

First-class targets today: **Hermes Agent** and **Claude Code** as harnesses, and
**LM Studio / llama.cpp** as servers. Claude Code runs against a local model with no
router, because llama.cpp serves Anthropic's Messages API itself. These are the
first implementations, not the assumptions.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The full test suite runs against
a scriptable mock server, with no GPU and no model:

    uv run pytest

The web UI is plain HTML, CSS and JavaScript modules with no build step. Its behaviour in a
real browser is covered by an opt-in test in `tests/browser/`. What the token-level features
stand on is measured against a real llama.cpp server by another opt-in set, skipped unless
you name one: `TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run pytest -m real_server`.

Tested on Linux — Ubuntu, natively and under WSL2 on Windows — with Python 3.12, 3.13 and
3.14. On Windows, run it inside WSL, with the checkout and the recordings on the Linux file
system rather than under `/mnt/c`, where SQLite's locking does not work. macOS is untested,
and the sandbox needs Linux. How to contribute: [`CONTRIBUTING.md`](CONTRIBUTING.md); how to
report a vulnerability, and what tuningfork does and does not guard:
[`SECURITY.md`](SECURITY.md).

## License

Apache-2.0: see [`LICENSE`](LICENSE), and [`NOTICE`](NOTICE) for the work it follows.
