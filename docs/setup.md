# Setup

tuningfork runs on Linux, natively or under WSL 2 on Windows. This page covers installing it,
starting a model server whose numbers repeat, the harnesses, the sandbox, and the settings a
person changes. For the first runs, follow [the guide](../examples/README.md); for every
command, see [commands](commands.md).

- [Install](#install), and [on Windows](#on-windows)
- [The model server](#the-model-server) — and [numbers that repeat](#numbers-that-repeat)
- [The harnesses](#the-harnesses)
- [Start tuningfork](#start-tuningfork)
- [The sandbox](#the-sandbox), and [a pinned clock](#a-pinned-clock)
- [A task](#a-task)
- [Protecting a server](#protecting-a-server)
- [Environment variables](#environment-variables)

## Install

You need git and [uv](https://docs.astral.sh/uv/); uv fetches Python 3.12 or later itself if you
don't have it. tuningfork is not on PyPI yet, so install it from GitHub:

```bash
git clone https://github.com/hhuang91/tuningfork.git
cd tuningfork
uv sync
source .venv/bin/activate
tuningfork version
```

`uv sync` installs everything, the MCP server included. On Windows, tuningfork runs inside WSL:
[on Windows](#on-windows) says how.

tuningfork records into `~/.tuningfork`, or into the folder `TUNINGFORK_HOME` or a command's
`--home` names. Keep tuningfork, the harnesses and the recordings out of `/tmp`: in
[the sandbox](#the-sandbox) every launch has a `/tmp` of its own, so a harness kept there
vanishes from a run's sight.

## On Windows

tuningfork runs inside WSL 2, Windows' own Linux, and its web UI opens in your Windows browser.

1. **Install WSL and a distro**, from PowerShell: `wsl --install -d Ubuntu`, then open *Ubuntu*
   from the Start menu and make its user.
2. **Work inside it, as on Linux:** [install](#install) tuningfork in your Linux home (`~`), and
   the model server, the harnesses and bubblewrap there too. Keep the checkout and the
   recordings under `~`, never under `/mnt/c`: SQLite's locking does not work on the Windows
   drive.
3. **Open the web UI in your Windows browser**, at the address tuningfork prints: WSL forwards
   `127.0.0.1` to Windows.
4. **A model server running on Windows itself**, llama.cpp's Windows build, is not at WSL's
   `127.0.0.1` unless WSL shares the address with Windows: turn on mirrored networking
   (`networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig`, then restart WSL;
   `wsl --shutdown` stops every distro), or run the server inside WSL.

### Keeping the checkout on Windows

To keep the checkout where your Windows editors see it, run tuningfork through
`scripts/tuningfork`, from PowerShell or from Git Bash. The checkout stays on Windows;
tuningfork's venv and its recordings stay in the distro's Linux file system. It is slower than
working inside WSL, since every file Python imports is read through the Windows drive.

```powershell
git clone https://github.com/hhuang91/tuningfork.git
cd tuningfork
$env:TUNINGFORK_WSL_DISTRO = "Ubuntu"      # the distro it runs in
.\scripts\tuningfork version               # the first run installs it there
.\scripts\tuningfork --run python tests/browser/demo.py lab /tmp/tuningfork-lab 8790
```

- **It runs nothing until a distro is named**, and lists the installed ones instead: with more
  than one, the default may not be the one you mean. In Git Bash it is
  `export TUNINGFORK_WSL_DISTRO=Ubuntu`.
- **Every tuningfork command goes through** as on Linux: `.\scripts\tuningfork batch …`,
  `.\scripts\tuningfork serve …`. `--run` runs anything else in tuningfork's venv, such as
  `--run pytest` for the tests.
- **Each argument arrives as you typed it**, the JSON of `--target-rules` included. A Windows
  path to a file or a folder you name is read as tuningfork reads one under WSL; the recordings
  stay in Linux (`~/.tuningfork`), so never name a Windows folder as `--home`.
- **It needs uv inside the distro** (`curl -LsSf https://astral.sh/uv/install.sh | sh`), and
  keeps its venv there, in `~/.local/share/tuningfork/venv`.
- **Windows PowerShell 5.1 runs no script until you allow it**, once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. PowerShell 7 runs it as it is.

## The model server

Experiments run against [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. The
guides use Gemma 4 31B on one 32 GB GPU:

```bash
llama-server -hf lmstudio-community/gemma-4-31B-it-GGUF:Q4_K_M --no-mmproj \
  --alias gemma-4-31b --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 1 -ctk q4_0 -ctv q4_0 --jinja --ctx-checkpoints 0 --no-cache-prompt
```

The first start downloads the model, about 19 GB; with the file already on disk, use
`-m /path/to/gemma-4-31B-it-Q4_K_M.gguf` in place of `-hf … --no-mmproj`. It is ready when
`curl -s http://127.0.0.1:8081/health` answers `{"status":"ok"}`. That is one slot of 131,072
tokens, in about 24 GiB: Hermes Agent needs a context of at least 64,000.

Qwen3.5 and Qwen3.6 take the same flags, with an 8-bit cache. Their 27B, dense like Gemma 4 31B,
holds the same slot in about 21 GiB:

```bash
llama-server -hf unsloth/Qwen3.6-27B-GGUF:Q4_K_M --no-mmproj \
  --alias qwen3.6-27b --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 1 -ctk q8_0 -ctv q8_0 --jinja --ctx-checkpoints 0 --no-cache-prompt
```

Qwen3 and Qwen2.5 take them too. Qwen3-30B-A3B-Instruct-2507, a mixture of experts that uses 3
billion of its 30 billion parameters a token, holds the same slot in about 24 GiB:

```bash
llama-server -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \
  --alias qwen3-30b-a3b-2507 --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 1 -ctk q8_0 -ctv q8_0 --jinja --ctx-checkpoints 0 --no-cache-prompt
```

The first Qwen3 models were trained on 40,960 tokens and Qwen2.5 on 32,768, and llama-server holds
a slot to that whatever `-c` asks: Hermes Agent then refuses them. Extend them as Qwen's model
cards do, with `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`.

### Numbers that repeat

Keep these flags for any model. With them, a request gets the same numbers whatever the server
ran before it and whichever of its endpoints it goes to:

| Flag | What goes wrong without it |
|---|---|
| `-np 1` | With two slots, a request computed beside another gets other numbers for the same prompt. One slot computes each request alone. |
| `--no-cache-prompt` | llama-server answers a prompt from what it cached of an earlier one, and its Messages endpoint, which Claude Code uses, ignores a request asking it not to. A step's samples then share the first one's processing of the prompt, and so do forks whose prompts match — every fork, in the sandbox. |
| `--ctx-checkpoints 0`, or `--swa-full` | llama-server takes rollback checkpoints of a model's sliding-window attention layers (Gemma 2, 3 and 4 among others) and of layers that keep a running state (Qwen3.5 and 3.6). They cut a prompt into batches at places that differ between the chat endpoint a harness talks to and the raw one tuningfork scores answers with. `--swa-full` avoids them for sliding-window layers only. Qwen3 and Qwen2.5 have neither kind of layer, and score the same either way. |

What we measured without them:

- **The server's history.** On Gemma 4 26B-A4B, one step's request gave its target behaviour
  1 time in 40 right after a restart, and 22 times in 40 once a similar prompt had run, on a
  server that reused what it had cached. A screen samples each proposal right after the
  original, so on such a server it favours the proposals.
- **The prompt cache.** One resample of a Claude Code step gave a behaviour 16 times in 20.
  With `--no-cache-prompt`, three resamples of the same step gave it 1 time in 10 each.
- **The endpoint.** With checkpoints taken, the same answer's deciding token had a chance of
  0.15 on the chat endpoint and 0.70 on the raw one. With `--ctx-checkpoints 0`, the two agree
  token for token.
- **The endpoint, on Qwen3.5 27B.** With checkpoints taken, the raw endpoint's greedy answer to
  a 20,000-token prompt parted from the chat endpoint's at its 45th token, and scored chances
  differed from the chat endpoint's by up to 0.03. With `--ctx-checkpoints 0`, the greedy answers
  agree token for token, and scoring gives the chat endpoint's chances exactly.
- **The endpoint, on Qwen3 30B-A3B,** whose layers all attend to the whole prompt. With
  checkpoints taken or not, scoring answers to the same prompt gave the chat endpoint's chances
  exactly, and the greedy answers agreed token for token.

What they cost: `--no-cache-prompt` makes every step process its whole prompt — about 1.5 s per
14,000 tokens of Gemma 4 26B-A4B on one RTX 5090, about 4 s of Gemma 4 31B — so runs take
longer. `--ctx-checkpoints 0` costs nothing when prompts are not cached, since a checkpoint only
rolls back a cached prompt. `--swa-full` takes no checkpoints either, by keeping the whole
sliding-window cache, at a cost in memory: the 26B with two 64k slots needed 27 GiB with it and
22 without, and the 31B at a 64k context does not fit a 32 GB card with it.

llama-server reports neither `--ctx-checkpoints` nor `--swa-full`, so tuningfork cannot check
them for you. It does measure prompt reuse: `tuningfork connections check` says whether the
server processed a prompt sent twice whole both times, and a batch's plan warns when it did
not. tuningfork also asks llama.cpp to process each experiment sample's prompt whole
(`cache_prompt: false`) where it can: on chat completions, not on the Messages endpoint.

### Other servers

LM Studio, and any server that speaks OpenAI chat completions or Anthropic Messages, can be
recorded through and sampled at a step. `tuningfork connections check` measures what a server
supports, and `tuningfork probe <url>` writes a full profile of it. Work at the level of tokens
— scoring an answer, forking tokens — needs llama.cpp's own endpoints and a model whose raw
output tuningfork reads: Gemma 4, Qwen2.5, Qwen3, Qwen3.5 or Qwen3.6; keeping each sample's tokens
in a resample needs the chat endpoint's logprobs. Features degrade on what was measured, not on a guess about which
server is running: token work the check found a server without is greyed in the web UI, saying
what was missing, and refused if started. A server without llama.cpp's `/props` has scoring and
forking tokens greyed before any check: the proxy asks for it every half minute.

### Cloud models

tuningfork is built for a model you run yourself: on your machine, on your network, or on a GPU
you rent, with llama.cpp's server. A hosted model's API, such as Anthropic's or OpenAI's, can be
the upstream, but only for recording:

- **What works:** your agent's calls are recorded through the proxy, with your agent's own API
  key passed on. So does everything that reads recordings: a run's timeline, where runs diverge,
  what failed checks said, imported chats, and scenarios made from runs.
- **What does not:** whatever tuningfork sends itself — resamples, A/Bs, forks, batches,
  validations and the server check — carries no API key, so a hosted API refuses it. Scoring an
  answer and forking tokens need llama.cpp's own endpoints, which no hosted API has.
- **Why a model of your own:** it answers tuningfork without a key, it shows the settings it
  samples with, it gives each token's probabilities, and it costs nothing per call, where one
  experiment makes dozens.

The proxy says so where it shows its server: in `tuningfork serve`'s first lines, on the
Connections page, in the status line at the sidebar's foot, on the Guide, in `connections show`
and `switch`, and in MCP's `server_status`. It does so when the server is on the internet
(every address its name resolves to is public: not this machine, your network, or a VPN like
Tailscale) and either asks for an API key or is not llama.cpp's server. Only a launch is refused
for it: a batch, a fork or a validation would have every call refused, since a harness tuningfork
launches carries none of your keys, so it is refused before it starts, and the Connections page
says so beside each harness. A llama.cpp server rented on the internet, answering without a key,
is a model you run yourself: nothing is said, and everything works.

## The harnesses

tuningfork launches two harnesses itself, for batches, forks and validations: each run in a
fresh workspace and a fresh home, against your local model only. Neither sees your own settings
or memory files, and neither needs an API key.

| | Hermes Agent | Claude Code |
|---|---|---|
| Install (the versions the guides use) | `uv tool install hermes-agent==0.18.2` | `npm install -g @anthropic-ai/claude-code@2.1.281` (Node 18 or later) |
| `--harness` | `hermes` (the default) | `claude-code` |
| Speaks | OpenAI chat completions | Anthropic Messages |
| The skill arrives | preloaded into the system prompt, whole file | through the `Skill` tool, body only |
| What it needs of the server | tool calls in a stream, and a context of 64,000 tokens or more | the Messages API, which llama.cpp serves itself |

tuningfork finds each on your `PATH`; `tuningfork connections harness <name> --executable <path>`
names another copy. A batch or a fork refuses to start a harness against a server that lacks
what it needs, rather than finding out from a failed run. [`adapters/`](adapters/) says what
each harness sends and how it is kept apart from your own setup.

Any other harness that takes a custom base URL is recorded as it runs: point it at
`http://127.0.0.1:8787/v1` (OpenAI) or `http://127.0.0.1:8787` (Anthropic). Launching and forking
it needs an adapter: [`adapters/TEMPLATE.md`](adapters/TEMPLATE.md).

## Start tuningfork

```bash
tuningfork serve --upstream http://127.0.0.1:8081
```

It listens on `http://127.0.0.1:8787/v1`, records, and prints the web UI's address. That address
carries the proxy's *access token*: new each time it starts, and what lets a page change
anything; reading needs none. `tuningfork token` prints the address again. One proxy per
recordings folder: two would overwrite each other's tokens.

Then, in another terminal, check the server before anything runs:

```bash
tuningfork connections check
```

It measures, with a few small calls, what the harnesses need — tool calls in a stream, the
Messages API — whether the server processes every prompt whole, and what token-level work
needs: a known output format, a prompt rendered as the chat endpoint renders it, completions
forced by a grammar of token ids, and the chat endpoint's own token probabilities (logprobs). [The guide](../examples/README.md#2-start-tuningfork) shows
its output. `tuningfork connections show` lists the server, what was measured of it, the
harnesses found with their versions, and whether launches run in the sandbox.

Options of `serve` worth knowing:

| Option | What it does |
|---|---|
| `--max-output N` | Holds every call a harness makes to N output tokens, and records what it asked. Hermes answers a reply cut at its own limit by asking again with the limit doubled, so a reply that loops can otherwise run to 65,536 tokens: 35 minutes of one GPU. |
| `--jobs N` | Background jobs run at once (default 1). On one GPU, more at once was measured slower. |
| `--max-inflight N` | Concurrent requests to the server (default 1): at least the `--concurrency` of any forks run through it. |
| `--mcp-tools`, `--mcp-results`, `--mcp-max-wait` | The MCP route an agent connects to: see [agents](agents.md). |
| `--port`, `--host` | Where it listens: 8787 on 127.0.0.1 by default. Keep it local. |

## The sandbox

On Linux, install [bubblewrap](https://github.com/containers/bubblewrap)
(`sudo apt install bubblewrap`). Every run, fork and held-out run is then launched in a sandbox
that looks the same each time: its home is `/home/user`, it works in `/home/user/project`, and
the harness keeps its files where it installs them.

The model reads those paths. Without the sandbox, each launch sees its own folders under the
recordings folder, and a few digits of a fork's folder name were seen to decide whether the
model wrote a file or only said it did: resampled at the same step, two forks whose requests
differed only in that number wrote it 5 times in 20 and 20 times in 20.

Inside, the rest of the machine is read-only, `/tmp` and the process list are the launch's own,
and the recordings folder and the task's spec folder are empty: the harness can read neither
earlier runs nor the check. `TUNINGFORK_SANDBOX_HIDE` names more folders every launch sees
empty, as `PATH` is written: where a benchmark keeps its checks, say. `tuningfork connections
show` says whether launches run in the sandbox, a batch's plan warns when they cannot, and a
fork of a sandboxed run is refused without one. `TUNINGFORK_SANDBOX=off` turns it off.

It keeps runs alike; it is not a security boundary. Inside, the rest of the machine stays
readable, your home included, and the network is reachable: see [`SECURITY.md`](../SECURITY.md).

### A pinned clock

The harness puts the date in the model's prompt, and at a knife edge the date decides.
`TUNINGFORK_CLOCK="2026-10-01 09:00:00"` starts every launch's clock at that instant, so the
harness reads the same date whichever day a run is made, and a fork reads its source's. It uses
libfaketime (`sudo apt install libfaketime`) under `LD_PRELOAD`, monotonic clocks left real;
`TUNINGFORK_CLOCK_LIBRARY` names the library if it is elsewhere, and a plan warns when it is
missing.

## A task

A task spec is an `evals.json`, in the convention of Anthropic's skill-creator, with
tuningfork's own settings under a `tuningfork` key.
[`examples/tally-report.evals.json`](../examples/tally-report.evals.json) is a whole one:

```json
{
  "skill_name": "tally-report",
  "tuningfork": {"skill_dir": "tally-report"},
  "evals": [
    {
      "id": "1",
      "prompt": "Use the tally-report skill to summarize data.csv into report.json.",
      "expected_output": "A report.json containing the line-item count and the amount total.",
      "files": [{"path": "data.csv", "content": "id,item,amount\n1,widget,10.00\n…"}],
      "tuningfork": {
        "check": "python3 -c \"import json,sys; d=json.load(open('report.json')); …\"",
        "held_out": false,
        "max_turns": 8,
        "timeout": 600
      }
    }
  ]
}
```

- **`skill_dir`** is the skill's folder, relative to the spec. Every run is given a copy of the
  whole folder, so keep anything the model should not read — an edited variant, a check
  script — outside it.
- **`files`** are the task's input files, placed in each run's workspace: `{"path", "content"}`
  for text written out, `{"path", "source"}` for a file beside the spec, or a plain name of a
  file or folder beside the spec.
- **`check`** decides pass or fail. A shell command runs in the run's workspace and passes when
  it exits 0; it finds the agent's final reply in the file `$TUNINGFORK_REPLY`, and its spec's
  folder in `$TUNINGFORK_SPEC_DIR`, where a longer check script can live out of the agent's
  sight. Or judge the reply alone: `{"reply_contains": "61.5"}` or
  `{"reply_matches": "(?i)row.count"}`. A check that prints what it found wrong, one finding a
  line, gets its findings counted across the failed runs (`tuningfork tally`).
- **`held_out`** marks a task that judges an edit rather than shaping one: a batch never picks
  it, no agent may read its runs, and a validation runs it in a private folder that keeps only
  its verdicts.
- **`max_turns`** and **`timeout`** bound a run (12 turns and 900 seconds by default);
  **`normalizers`** say which differences between two answers do not count as a different
  behaviour.
- **`context`** names a conversation each run continues: a JSON file beside the spec holding
  OpenAI-style user, assistant and tool messages. The proxy puts it after the harness's own
  instructions in every call of the run, so the prompt is that conversation's next turn. A
  scenario made from a run's conversation has one (`tuningfork scenario <run> <name>`, or
  `tuningfork import <file> --scenario <name>`).
  The harness must speak OpenAI chat, as Hermes does. As `{"before": [...], "in_turn": [...]}`,
  it also holds the model's earlier steps of the prompt's own turn — when the chat loaded the
  studied skill after another one in the same turn — which follow the prompt, taken as done.
- **`stop_at_next_skill`** ends each run where the model loads a skill other than the spec's,
  and the check judges the workspace as it is then: for a scenario of a chat that went on to
  another skill, which no run of this one has. A check of the final reply finds none there.

`expected_output` is a note for people. Try a check before trusting it:
`tuningfork recheck <run> --check '…'` judges a recorded run again, and `--reply` with
`--file NAME=PATH` judges a reply and files you give, recording nothing. The web UI's
**Scenarios** page keeps tasks in the recordings folder, edited as the exact `evals.json` they
save to, with their input files.

## Protecting a server

If a model server on your machine serves something that matters, protect it, and tuningfork
sends it nothing — not a resample, not a probe, not a harness's call. List it in
`~/.config/tuningfork/config.json`:

```json
{"protected": ["localhost:1234"]}
```

or in `TUNINGFORK_PROTECTED` (a comma-separated list of `host:port`, where either side may be
`*`), or in a recordings folder's own `config.json`. The lists add up; none takes back another's
entry. A list that cannot be read stops all of tuningfork's traffic until it is fixed.

## Environment variables

| Variable | What it sets |
|---|---|
| `TUNINGFORK_HOME` | The recordings folder (default `~/.tuningfork`) |
| `TUNINGFORK_UPSTREAM` | The inference server: `serve --upstream`, and where an experiment's samples go (default `http://127.0.0.1:8081`) |
| `TUNINGFORK_PROXY` | The running proxy, for the commands that control it (default `http://127.0.0.1:8787`) |
| `TUNINGFORK_TOKEN` | The proxy's access token, for a command that cannot read it from the recordings folder |
| `TUNINGFORK_PROTECTED` | Servers tuningfork never sends to: [above](#protecting-a-server) |
| `TUNINGFORK_SANDBOX` | `off` turns [the sandbox](#the-sandbox) off |
| `TUNINGFORK_SANDBOX_HIDE` | More folders every launch sees empty |
| `TUNINGFORK_CLOCK`, `TUNINGFORK_CLOCK_LIBRARY` | [A pinned clock](#a-pinned-clock) for every launch |
| `TUNINGFORK_MCP_TOKEN` | Pins the MCP token across restarts: see [agents](agents.md) |
| `TUNINGFORK_PROPOSER_KEY` | The built-in proposer's API key, for an endpoint that needs one |
| `TUNINGFORK_HOST`, `TUNINGFORK_PORT`, `TUNINGFORK_MAX_INFLIGHT`, `TUNINGFORK_JOBS`, `TUNINGFORK_MAX_OUTPUT`, `TUNINGFORK_MCP_TOOLS`, `TUNINGFORK_MCP_RESULTS`, `TUNINGFORK_MCP_MAX_WAIT` | `serve`'s options of the same names |

A check is given `TUNINGFORK_REPLY` and `TUNINGFORK_SPEC_DIR`: see [a task](#a-task).
