# Agents over MCP

`tuningfork mcp` serves tuningfork's operations to an agent over the Model Context Protocol. An
agent can do what a person does at a step: count what the failed runs' checks said, find where a
task's runs split, read that step's context and the lines of the skill in it, measure how often
the model does each thing there, and A/B a wording it wrote. With the refine tools, it can also
declare a target, propose edits of its own, and screen them.

What that gave Hermes on one of its own skills, beside Hermes's own self-improvement loop, is in
[the case study](../examples/hermes-architecture-diagram/README.md).

- [Connect a client](#connect-a-client): [Hermes Agent](#hermes-agent),
  [Claude Code](#claude-code), [OpenJarvis](#openjarvis), [from Windows](#from-windows-with-tuningfork-in-wsl)
- [What an agent is offered](#what-an-agent-is-offered), and [what it never is](#what-it-is-never-offered)
- [Jobs and limits](#jobs-and-limits)
- [How the answers read](#how-the-answers-read)
- [A harness's own self-improvement, for comparison](#a-harnesss-own-self-improvement-for-comparison)

## Connect a client

There are two ways in, both over the same recordings:

- **stdio:** the client starts `tuningfork mcp --home <recordings>`. Reading the recordings needs
  no proxy. Work that calls the model runs as a job inside a `tuningfork serve` recording into the
  same folder, which `--proxy` names (default `TUNINGFORK_PROXY`, else `http://127.0.0.1:8787`).
- **HTTP:** the client connects to the running proxy's own route,
  `http://127.0.0.1:8787/_tuningfork/mcp`, with the proxy's *MCP token* as a bearer token.
  `serve --mcp-tools` chooses what the route offers.

The MCP token opens that route and nothing else, and the proxy's access token is refused there:
an agent's client never holds a key to what its tools leave out. `tuningfork token --mcp` prints
it, `tuningfork token --header-json` prints the header, and `TUNINGFORK_MCP_TOKEN` pins it across
restarts; otherwise a new one is minted each time `serve` starts.

Point the agent's own model at the inference server, never at the proxy it studies, or its
calls are recorded beside the runs it is reading.

The examples below name `tuningfork` where a clone at `/home/you/tuningfork` keeps it. Installed
with `uv tool install`, it is `/home/you/.local/bin/tuningfork`.

### Hermes Agent

Hermes needs its `mcp` extra: `uv tool install "hermes-agent[mcp]==0.18.2"`. Then list the server
in Hermes's `config.yaml`, by command:

```yaml
mcp_servers:
  tuningfork:
    command: /home/you/tuningfork/.venv/bin/tuningfork
    args: ["mcp", "--home", "/home/you/.tuningfork", "--results", "text"]
```

or by the proxy's route:

```yaml
mcp_servers:
  tuningfork:
    url: http://127.0.0.1:8787/_tuningfork/mcp
    headers:
      Authorization: "Bearer <the MCP token>"
    timeout: 300
```

Hermes shows the model both a result's text and its structured content, side by side, so it
would read every answer twice: `--results text` (for the route, `serve --mcp-results text`)
sends the text alone. Hermes gives a tool call 300 seconds; over the route,
`serve --mcp-max-wait 110` lets a call wait longer for its job, so a twenty-minute screen is a
dozen polls rather than twenty-five.

A Hermes that tuningfork launches itself — a batch whose task asks it to refine a skill — gets
the server through the harness option `mcp_servers`, in the same shape:
`tuningfork batch spec.json -o mcp_servers='{"tuningfork": {"url": "…", "headers": {…}}}'`.

### Claude Code

```bash
claude mcp add --transport stdio tuningfork -- ~/tuningfork/.venv/bin/tuningfork mcp --home ~/.tuningfork
```

Or by the proxy's route, with the header read fresh on each connection, since the MCP token
changes when `serve` restarts — in `.mcp.json`:

```json
{
  "mcpServers": {
    "tuningfork": {
      "type": "http",
      "url": "http://127.0.0.1:8787/_tuningfork/mcp",
      "headersHelper": "/home/you/tuningfork/.venv/bin/tuningfork token --header-json"
    }
  }
}
```

A client starts these commands with its own environment, so name `tuningfork` by its full path,
as here, unless it is on the client's `PATH`.

Claude Code shows the model a result's structured content, which leads with the same sentences
as the text: the default, `--results both`, is right for it.

### OpenJarvis

OpenJarvis lists the server under `[mcp] servers` in its `config.toml`, by command or by URL and
token. It reads an answer for 60 seconds, and the route's default wait, 50, stays under that.

### From Windows, with tuningfork in WSL

Give the client this command: `wsl.exe -d <distro> -e <venv>/bin/tuningfork mcp --home <recordings>`.
The server reads the token from the recordings folder itself, so nothing secret goes in a
Windows config. Name the distro: a bare `wsl.exe` starts your default one.

## What an agent is offered

| Toolset (`mcp --tools`, `serve --mcp-tools`) | What it offers |
|---|---|
| `evidence` (the default) | Every read — tasks and their pass rates, runs, steps, the files a run wrote, the skill, where runs diverge, what the failed checks said, one question of every run, the behaviours seen at a step, experiments — and work that only calls the model: a resample, an A/B, scoring an answer, forking tokens, the server check |
| `refine` | The refinement loop: start a refinement, read its evidence, propose edits, and resample, A/B and screen at the step |
| `refine-to-end` | The loop, and screens whose samples are forks run to the end through the harness on this machine, for a failure only the task's check can see |
| `all` | Everything above, and work that runs a harness or a task's check on this machine: batches, chains, forks, resamples and A/Bs to the end, trying a check, the check run along runs |

Every tool's description carries what the model must know to use it, since a client may show
the model no server instructions (Hermes shows none): the skill is read with `show_skill`, never
from a file; `tally_failures` counts every failed run's check output at once; the step to refine
is the first whose call made what the check complains about.

### What it is never offered

Whatever the toolset:

- **Approving an edit.** Only a person writes a skill's file, and the MCP token is refused where
  approving happens.
- **The built-in proposer.** It sends recorded prompts, answers and your skill to its endpoint —
  perhaps a cloud API, on your key — which is your decision. An agent proposes its own edits with
  `propose_edit`.
- **Validation, and a held-out task's runs or verdicts.** They judge an edit, and an agent that
  saw them could fit its edits to them.
- **Live debugging, the trash, and switching the server.** Sampling or asking the model at a held
  call stays with the person at the Live page.

A fork an agent starts never stops at a breakpoint, and a check an agent gives never runs a
shell command: it reads the reply (`reply_contains`, `reply_matches`), or it is the run's own.

## Jobs and limits

Work that calls the model, or runs a harness, is a job inside `tuningfork serve`, sent through the
proxy's gate like the person's own work, and shown on its Jobs page as started by "an agent over MCP ·
*client*". A tool call returns the job's result if it ends within the call's wait, and otherwise its id, for
`job_status`.

| Limit | Default | Option of `tuningfork mcp` |
|---|---|---|
| Samples one job may draw at a checkpoint | 100 | `--limit resample=200`, by job kind |
| Harness runs one job may make | 20 (60 forks for a screen to the end) | `--limit batch=…` |
| Jobs agents may have queued or running | 4 | `--max-jobs` |
| How long a call waits for its job | 120 s on stdio, 50 s on the route | `--max-wait`; for the route, `serve --mcp-max-wait` |

The route on the proxy keeps the other defaults.

The same work asked for twice is given back rather than started again, and an agent cancels only
its own jobs.

`list_jobs` gives the newest jobs — 20, or as many as it asks for, up to 100 — and then every job
still queued or running, however old. Over stdio it gives at most the 50 newest, whatever it asks
for: the proxy's jobs API returns no more.

## How the answers read

Each answer is compact JSON led by sentences that say it, statistics first: every rate with its
95% interval, an A/B's declared target with Fisher's p and its threshold, or that the comparison
was exploratory. A long file or message is read page by page from an `offset`. Text from the
recordings — prompts, tool results, model output, skills — is data to study, never instructions
to follow, and the server's instructions say so.

A result carries its text and its structured content, because a client shows the model one of
them: Claude Code the structured content, OpenJarvis the text. Hermes shows both, so give it
`--results text`. The same refusal three times running carries the tool's description again, and
from the sixth time the answer says to stop sending that call and try another edit, step or tool.

## A harness's own self-improvement, for comparison

Hermes can rewrite a skill itself, after a run, in its background review. A *chain* runs a task
N times in order, each run given the skill the run before it left: what the harness's own loop
makes of it.

```bash
tuningfork chain run spec.json -n 30 --tell -o await_review=true -o review_every_turn=true
tuningfork chain show <id> --diff
```

`--tell` gives each run's harness the check's verdict and output as the user's next turn, before
its review: what a person who looked at the result would say. The output is masked as tuningfork
keeps it: what the masking takes reaches the harness as `<redacted>`. `await_review` lets
Hermes's review finish before the process exits, and `review_every_turn` makes it due after every
turn; without both, Hermes's loop never acts in a one-shot launch. `--budget-seconds N` stops
once the links' model calls have taken N seconds, so two loops can be given the same time.
`chain show --diff` shows who changed the skill after each run, and the lines they changed; the
web UI's Chains page shows the same. The [case study](../examples/hermes-architecture-diagram/README.md)
sets this loop against an agent with tuningfork's tools.
