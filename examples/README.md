# Tuning Fork, step by step

This folder holds one small skill that mostly doesn't work, and everything needed to watch
tuningfork find out why. Follow it top to bottom and you will have used every part of the
tool.

- **Part 1 — no GPU, about five minutes.** Install, browse a set of recordings, and stop a
  live model call at a breakpoint and change it. The "model" is a script, so what you see
  matches this page exactly, apart from ids and times.
- **Part 2 — the real experiment, about half an hour with Claude Code and forty minutes with
  Hermes Agent, on one GPU.** Record eight runs of the skill on Gemma 4 31B, find the
  decision that fails, resample it, test a rewording at exactly that step, follow the
  rewording from there to the end of the run, confirm the fix with full runs, and restart a
  run at the failing step to catch it live — then start the same work from the browser.
- **Part 3 — inside a step, about twenty minutes more.** Read an answer as the model's own
  tokens, and how sure it was of each; then, where runs split, look for the token that
  splits them.
- **Part 4 — let tuningfork propose the fix, about half an hour more for each harness.** Say
  what the failing step should do, have a model propose edits from the evidence — or write
  your own — screen each one at that step against the original wording, validate a survivor
  with full runs, and approve it into the skill.

Every output below was produced by running this page's commands, as written, on a fresh
copy of the repository. Home directories are shortened to `/home/you`, the user's name to
`you`, the WSL distro's to `Ubuntu` and the proxy's port to its default, 8787; long outputs
are trimmed where marked `...`.

The skill here is a toy, made to fail one way. For a real one — Hermes Agent's own
`design-md` skill on Gemma 4 26B, from the failing step to an approved fix and the runs after —
see [`hermes-design-md/`](hermes-design-md/README.md). For an agent doing the work over MCP —
Hermes with tuningfork's tools against Hermes's own self-improvement loop, on its
`architecture-diagram` skill — see [`hermes-architecture-diagram/`](hermes-architecture-diagram/README.md).

## What is in this folder

| File | What it is |
|---|---|
| [`tally-report/SKILL.md`](tally-report/SKILL.md) | The skill under test |
| [`tally-report.evals.json`](tally-report.evals.json) | The task: a prompt, an input file, and a pass/fail check |
| [`variants/tally-report/SKILL.v2.md`](variants/tally-report/SKILL.v2.md) | The same skill, reworded to fix it |

The skill describes its job correctly and completely — read a CSV, count the line items,
total the amount column, write `report.json` — but never says **what to call the keys**. The
check wants `row_count` and `amount_total`. The model writes `count` and `total`.

The variant lives outside the skill's folder on purpose. A batch gives every run a copy of
the skill's whole folder, so a reworded copy kept beside `SKILL.md` would be an answer key
the model could read. (It sat there for a while; see the notes at the end.)

---

## Part 1 — tuningfork without a model

You need [uv](https://docs.astral.sh/uv/) and git; uv fetches Python 3.12 or later itself if
you don't have it. On Windows, work inside WSL, and keep the checkout in the Linux file
system (`~/...`), not under `/mnt/c`: SQLite's locking does not work there.

### 1. Install

```bash
git clone https://github.com/hhuang91/tuningfork.git
cd tuningfork
uv sync
source .venv/bin/activate
```

Optionally, run the test suite. It needs no GPU and no model:

```bash
pytest -q
```

### 2. Make some recordings to look at

```bash
python tests/browser/demo.py seed ~/tuningfork-demo
export TUNINGFORK_HOME=~/tuningfork-demo
```

```
seeded /home/you/tuningfork-demo: 10 runs, a resample and an A/B. Try: tuningfork runs --home /home/you/tuningfork-demo
```

This writes what nine runs of the example task would leave behind, as if recorded from Hermes
Agent, plus one experiment of each kind. They are made up: in this version of the story the
model gets the keys right two times in three, which gives more to look at. Part 2 measures
what really happens.

Every `tuningfork` command reads its recordings from `$TUNINGFORK_HOME` (by default
`~/.tuningfork`), or from `--home`.

### 3. Look around from the terminal

The recorded runs, newest first:

```bash
tuningfork runs
```

```
 run       label                  steps  model        started      check
 a059f6e8  -                      1      gemma-4-31b  10-11 11:16  -
 4ad4bebf  tally-report/1#demo.8  3      gemma-4-31b  10-11 11:16  pass
 a086fa96  tally-report/1#demo.7  3      gemma-4-31b  10-11 11:16  pass (not counted)
 9d4bd185  tally-report/1#demo.6  3      gemma-4-31b  10-11 11:16  pass
 a50b7dc7  tally-report/1#demo.5  3      gemma-4-31b  10-11 11:16  pass (not counted)
 fcb1b99d  tally-report/1#demo.4  3      gemma-4-31b  10-11 11:16  fail
 e045d3ed  tally-report/1#demo.3  3      gemma-4-31b  10-11 11:16  pass
 d3cc248b  tally-report/1#demo.2  3      gemma-4-31b  10-11 11:16  fail
 8b71444f  tally-report/1#demo.1  3      gemma-4-31b  10-11 11:16  pass
 4b174d29  tally-report/1#demo.0  3      gemma-4-31b  10-11 11:16  pass

10 run(s), 28 step(s), 9 unique message(s) in /home/you/tuningfork-demo
```

Your ids and times will differ; the labels won't. Two runs are *not counted*: one was edited
at a breakpoint, and in the other the skill changed while it ran. Either way the verdict
measures more than the skill, so it is kept out of every rate. The pass rate, always with its
interval:

```bash
tuningfork tasks
```

```
tally-report/1 · hermes · gemma-4-31b
 skill     runs  check passed
 a606f71e  9     5/7  71%  [36%-92%]  2 not counted

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

One run, step by step. A run can be named by its id, a prefix of it, or its label:

```bash
tuningfork show "tally-report/1#demo.2"
```

```
run d3cc248b  tally-report/1#demo.2
  task    tally-report/1
  model   gemma-4-31b
  started 10-11 11:16:05
  harness hermes
  skills  a606f71e (1 file(s))
 k  outcome                                                       finish      ms      tokens  think
 0  read_file(path='data.csv')                                    tool_calls  2210.5  4145    29
 1  write_file(content='{"count":4,"total":61.5}',                tool_calls  4080.0  4342    58
    path='report.json')
 2  Done — the totals are in report.json.                         stop        1150.0  4274    -
```

One step: exactly what the model saw, and what it did.

```bash
tuningfork step "tally-report/1#demo.2" 1
```

```
run d3cc248b step 1
  model   gemma-4-31b
  params  {}
  server  {"n_ctx": 65536, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
  timings {"ttft_ms": 1780.0, "total_ms": 4080.0}
  usage   {"prompt_tokens": 4180, "completion_tokens": 162, "total_tokens": 4342}

context
  [0] system: You are an agent.  [IMPORTANT: The user launched this CLI session with the
"tally-report" skill prel…
  [1] user: Use the tally-report skill on data.csv.
  [2] assistant: tool_calls(read_file)
  [3] tool: id,amount 1,10.00 2,20.50 3,15.00 4,16.00

response
  reasoning_content: Four rows; 10 + 20.5 + 15 + 16 = 61.5. Keys: count, total.
  tool_call write_file({"path": "report.json", "content": "{\n  \"count\": 4,\n  \"total\":
61.5\n}"})
```

`--full` prints whole messages. Now: where do the runs stop agreeing?

```bash
tuningfork divergence --task tally-report/1
```

```
9 runs of tally-report/1
  agreed for 1 step(s), diverged at step 1

step 1
   count  rate  95% CI     behaviour                                           then passed
A  6/9    67%   [35%-88%]  write_file(content='{"amount_total":61.5,"row_cou…  5/5  100%  [57%-100%]
                           path='report… (+7 chars)
B  3/9    33%   [12%-65%]  write_file(content='{"count":4,"total":61.5}',      0/2  0%  [0%-66%]
                           path='report.json')

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82
runs per arm.
```

Every run did the same thing at step 0, and they split at step 1 — the write. *Then passed*
says which behaviour went on to pass the check. Behaviours are compared after normalization,
so `{"count": 4}` and `{\n  "count": 4\n}` are one behaviour.

Last, the two experiments — a resample of step 1, and a wording A/B there. `experiments`
lists them; `-q -n 1` picks the newest, the A/B:

```bash
tuningfork experiments
tuningfork experiment $(tuningfork experiments -q -n 1)
```

```
ab at run 4b174d29 step 1 — after the result of read_file
  swapped   skills/tally-report/SKILL.md → skills/tally-report/SKILL.v2.md

   behaviour                                      original              variant               p adj.
A  write_file(content='{"amount_total":61.5,"ro…  12/20  60% [39%-78%]  19/20  95% [76%-99%]  0.039
   … (+19 chars)
B  write_file(content='{"count":4,"total":61.5}…  8/20  40% [22%-61%]   1/20   5% [1%-24%]    0.039
   path='repor… (+8 chars)
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 2 comparisons

target (declared before sampling): behaviour matching /row_count/
  original  12/20  60% [39%-78%]
  variant   19/20  95% [76%-99%]
  Fisher exact p = 0.020 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 1 model-written message(s) before this step were generated under the original wording, and both
arms inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
```

### 4. The same, in a browser

```bash
tuningfork ui
```

```
tuningfork 0.2.1 — browsing /home/you/tuningfork-demo
  open http://127.0.0.1:8788/_tuningfork/ui/
```

Open <http://127.0.0.1:8788/_tuningfork/ui/>. **Runs** groups runs by task, with the pass rate
and its interval. Open run `…#demo.2`: its conversation, top to bottom, with each step's
answer where it happened and the skill marked in the system prompt, mapped to its file line
by line. Click step 1 for its details; *Exact context* shows everything that step was sent.
*Where do they diverge?* on the task, and the **Experiments** page, show the last two
commands' results. Stop it with Ctrl-C.

### 5. Stop a live call, and change it

Now a real tuningfork proxy, in front of the scripted model. It answers every call by
writing `report.json` with the wrong keys. In a **first terminal**:

```bash
cd tuningfork && source .venv/bin/activate
python tests/browser/demo.py proxy ~/tuningfork-live 8791
```

```
tuningfork proxy on http://127.0.0.1:8791/v1, in front of a mock model (no GPU)
  recording to /home/you/tuningfork-live
  web UI       http://127.0.0.1:8791/_tuningfork/ui/?token=…
  control it with TUNINGFORK_PROXY=http://127.0.0.1:8791 — Ctrl-C to stop
```

The web UI's address carries the proxy's *access token*: a new one each time a proxy
starts, and what lets a page change anything. Reading needs no token. The commands below
find it by themselves, in the recordings folder the proxy names; `tuningfork token` prints
the address again.

In a **second terminal**, point tuningfork's commands at that proxy, and set a breakpoint:
stop any `write_file` of a `.json` file after the model has decided on it, before the caller
acts on it.

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_PROXY=http://127.0.0.1:8791 TUNINGFORK_HOME=~/tuningfork-live
tuningfork break add --tool write_file --path '*.json'
```

```
breakpoint dcceda9e — after the model: write_file on *.json
```

Make a model call the way a harness would, in the background, then wait for it to be held:

```bash
curl -s http://127.0.0.1:8791/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "Tally data.csv"}]}' \
  > answer.json &
tuningfork hold wait
```

```
hold ce180bb1 — after the model, a new run step 0, held 0.2s
  reason  after the model: write_file on *.json
  user: Tally data.csv

the model answered
  reasoning_content: The CSV has four rows; I will write the totals.
  tool_call write_file({"path": "report.json", "content": "{\n  \"count\": 4,\n  \"total\":
61.5\n}"})

tuningfork hold continue ce180bb1 · step · drop — to edit the response: hold show ce180bb1 --json >
f.json, edit, hold continue ce180bb1 --edited f.json
```

`curl` is still waiting: to the caller, a held call is just a slow response. Fix the keys and
let it go on. `hold show --json` prints the held response; a tool call's arguments are JSON
text inside it, so this edits them with a few lines of Python:

```bash
HOLD=$(tuningfork hold list -q)
tuningfork hold show $HOLD --json > held.json
python - <<'EOF'
import json
held = json.load(open("held.json"))
call = held["choices"][0]["message"]["tool_calls"][0]["function"]
arguments = json.loads(call["arguments"])
arguments["content"] = '{"row_count": 4, "amount_total": 61.5}'
call["arguments"] = json.dumps(arguments)
json.dump(held, open("held.json", "w"))
EOF
tuningfork hold continue $HOLD --edited held.json
wait
python -m json.tool answer.json | grep arguments
```

```
ce180bb1: continue with your edit
                            "arguments": "{\"path\": \"report.json\", \"content\": \"{\\\"row_count\\\": 4, \\\"amount_total\\\": 61.5}\"}"
```

The caller received your version. The recording keeps both — what the model produced is the
step's behaviour, and the edit is noted beside it:

```bash
tuningfork show $(tuningfork runs -q -n 1)
```

```
run 2ee2322c
  task    -
  model   demo
  started 10-11 11:16:19
 k  outcome                                        finish      ms    tokens  think  note
 0  write_file(content='{"count":4,"total":61.5}…  tool_calls  0.47  18      47     response edited
    path='report.json')

An edited step's outcome is what the model produced; `tuningfork step <run> <k> --harness-view`
shows what the harness saw instead.
```

The browser makes this easier. Open the web UI address the proxy printed — the one with
`?token=` — and go to **Live**. Run the `curl` line again, and edit the held call's
`content` field right there — the page shows a file's content raw, not as escaped JSON —
then click *Continue with edits*. Opened without the token, the page shows everything and
changes nothing, and says how to paste the token in. Stop the proxy with Ctrl-C in the
first terminal.

---

## Part 2 — the real experiment

### What you need

- **An NVIDIA GPU** with about 24 GiB free: a 19 GB model and its context. We used one RTX
  5090 (32 GB); the times below come from it.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s `llama-server`. We used build
  b11108; any recent build with Gemma 4 support should do.
- **A harness: Claude Code, Hermes Agent, or both.** From §3 on, every step runs both, Claude
  Code first and then Hermes Agent under *With Hermes Agent*, each with our runs' outputs.
  We used **Claude Code** 2.1.281, which needs Node 18 or later
  (`npm install -g @anthropic-ai/claude-code@2.1.281`), and **Hermes Agent** 0.18.2
  (`uv tool install hermes-agent==0.18.2`). Either way, tuningfork runs the harness in a
  throwaway home directory against your local model only: it never sees your own settings or
  memory files, and needs no API key.
- **[bubblewrap](https://github.com/containers/bubblewrap)**, on Linux:
  `sudo apt install bubblewrap`. With it, every run tuningfork launches sees the same home and
  folder, `/home/user` and `/home/user/project`, as an installation would; without it, each
  run sees its own folders, whose names differ from run to run, and the model reads them. We
  used 0.11.1.

Everything stays on your machine: the model server, the proxy and the harness all listen on
127.0.0.1.

The two harnesses differ in ways the commands show:

| | Claude Code | Hermes Agent |
|---|---|---|
| `--harness` | `claude-code` | `hermes` |
| Speaks | Anthropic Messages | OpenAI chat-completions |
| The skill arrives | through the `Skill` tool, body only | preloaded into the system prompt, whole file |
| The write step | `Write`, or a script | `write_file`, or a script (`execute_code`), whose code a target reads |
| Breakpoint | `--tool Write --path '*.json'` | `--tool write_file --path '*.json'` |

### 1. Start the model server

In a **first terminal**:

```bash
llama-server -hf lmstudio-community/gemma-4-31B-it-GGUF:Q4_K_M --no-mmproj \
  --alias gemma-4-31b --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 1 -ctk q4_0 -ctv q4_0 --jinja --ctx-checkpoints 0 --no-cache-prompt
```

The first start downloads the model, about 19 GB. If you already have the file, use
`-m /path/to/gemma-4-31B-it-Q4_K_M.gguf` in place of `-hf ... --no-mmproj`. It is ready
when this answers `{"status":"ok"}`:

```bash
curl -s http://127.0.0.1:8081/health
```

That is one slot of 131,072 tokens: Hermes needs 64,000.

**Keep `-np 1`, `--ctx-checkpoints 0` and `--no-cache-prompt`.** With them, a request gets the
same numbers whatever the server ran before it and whichever of its endpoints it goes to (see
[the setup page](../docs/setup.md#numbers-that-repeat), with what we measured without them):

- With two slots, a request computed beside another in the second one gets other numbers.
  One slot computes each request alone.
- Gemma 4 has sliding-window attention layers, and llama-server takes rollback checkpoints of
  their cache. Those cut a prompt into batches at places that differ between the chat endpoint
  a harness talks to and the raw one tuningfork scores answers with, and the same answer then
  scores differently on each. `--ctx-checkpoints 0` takes none, at no cost here: a checkpoint is
  only used to roll back a cached prompt, which this server never keeps. (`--swa-full` takes
  none either, by keeping the whole sliding-window cache, but the 31B at this context does not
  fit a 32 GB card with it.)
- Without `--no-cache-prompt`, llama-server answers a prompt from what it cached of an earlier
  one. Claude Code's calls cannot ask it not to, so every sample of a step would share the first
  one's processing — and in the sandbox, every fork of an arm too. The flag costs time: every
  step processes its whole prompt, about 4 s for this task's 14,000 tokens.

This command uses about 24 GiB.

### 2. Start tuningfork

In a **second terminal**, the recording proxy — it sits between the harness and the model:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-guide
tuningfork serve --upstream http://127.0.0.1:8081
```

```
tuningfork 0.2.1
  listening on  http://127.0.0.1:8787/v1
  forwarding to http://127.0.0.1:8081
  recording to  /home/you/tuningfork-guide
  max in flight 1
  web UI        http://127.0.0.1:8787/_tuningfork/ui/?token=…
  MCP           http://127.0.0.1:8787/_tuningfork/mcp  (evidence tools; its own token, `tuningfork token --mcp`)
  the token lets the page change things; it is new each start, and `tuningfork token` prints this
address again
```

Leave it running. Open a **third terminal** for everything that follows, set up the same way:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-guide
```

Before anything runs, check that the server can do what the harness needs. Claude Code
speaks only Anthropic's Messages API, and Hermes streams every call and acts on the tool
calls in the stream; tuningfork measures both with two small calls, then whether the server
processes every prompt whole — and, with three more, what Part 3's token-level work needs:

```bash
tuningfork connections check
```

```
  streaming_tool_calls   yes — structured delta.tool_calls observed
  anthropic_messages     yes — a Messages response
  fresh_prompts          yes — a prompt sent twice was processed whole both times
  known_output_format    yes — gemma4
  prompt_equivalence     yes — rendered 68 tokens, the chat endpoint's prompt has 68
  raw_completion         yes — 2 tokens forced by a grammar of ids, with their top alternatives
```

A batch or a fork refuses to start a harness against a server that lacks what it needs —
Hermes also needs a context of at least 64,000 tokens — rather than finding out from a
failed run, and warns when the server answers prompts from its cache. `tuningfork
connections show` lists the server, what was measured of it, the harnesses found with their
versions, and whether launches run in the sandbox.

### 3. Record eight runs

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness claude-code --model gemma-4-31b
```

Each run gets a fresh workspace holding `data.csv`, a fresh harness home with the skill
installed — which it sees as `/home/user/project` and `/home/user`, the same in every run —
and the task's check afterwards. About half a minute a run:

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  fail  28.7s
  run 1  fail  28.5s
  run 2  fail  30.3s
  run 3  fail  30.6s
  run 4  fail  32.0s
  run 5  fail  34.5s
  run 6  fail  28.6s
  run 7  fail  27.2s

0/8  0%  [0%-32%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                                               then passed
A  3/8    38%   [14%-69%]  Skill(args='data.csv report.json',                      0/3  0%  [0%-56%]
                           skill='tally-report')
B  2/8    25%   [7%-59%]   Skill(args='input=data.csv output=report.json',         0/2  0%  [0%-66%]
                           skill='tally-report')
C  1/8    12%   [2%-47%]   Bash(command='ls data.csv')                             0/1  0%  [0%-79%]
D  1/8    12%   [2%-47%]   Skill(args='summarize data.csv into report.json',       0/1  0%  [0%-79%]
                           skill='tally-report'… (+1 chars)
E  1/8    12%   [2%-47%]   Skill(skill='tally-report')                             0/1  0%  [0%-79%]

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82
runs per arm.
```

Every run failed, and the same way. Seven loaded the skill first, and one checked the file
was there with `ls data.csv` before it; they gave the skill different arguments, and *then
passed* says none of that made a difference. Then every run read `data.csv` and wrote
`report.json` with `Write`, with the keys `count` and `total` — the skill's own words, "just
the count and the total". The check wants `row_count` and `amount_total`: the failure the
skill was made for.

#### With Hermes Agent

The same task, with `--harness hermes`:

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness hermes --model gemma-4-31b
```

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  fail  27.5s
  run 1  fail  26.3s
  run 2  fail  26.3s
  run 3  fail  28.0s
  run 4  fail  27.7s
  run 5  fail  31.9s
  run 6  fail  59.5s
  run 7  fail  32.7s

0/8  0%  [0%-32%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                                         then passed
A  3/8    38%   [14%-69%]  read_file(path='data.csv')                        0/3  0%  [0%-56%]
B  1/8    12%   [2%-47%]   execute_code(code='<ast:1916367c>')               0/1  0%  [0%-79%]
C  1/8    12%   [2%-47%]   execute_code(code='<ast:7b3b041f>')               0/1  0%  [0%-79%]
D  1/8    12%   [2%-47%]   execute_code(code='<ast:d1e4a258>')               0/1  0%  [0%-79%]
E  1/8    12%   [2%-47%]   execute_code(code='<ast:fb68b83f>')               0/1  0%  [0%-79%]
F  1/8    12%   [2%-47%]   search_files(pattern='data.csv', target='files')  0/1  0%  [0%-79%]

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82
runs per arm.
```

Every run failed here too, and every report had the keys `count` and `total`. Hermes preloads
the skill into its system prompt, so no run loads it; they split instead on how they began.
Three read `data.csv` with `read_file`, four ran a Python script straight away, each its own,
and one searched for the file. Then four wrote the report with `write_file` and four from
scripts, three of those counting five line items where there are four.

### 4. Find the decision that fails

Take the most recent run in which the model called `Write`, and list what it did:

```bash
RUN=$(tuningfork runs -q -n 1 --called Write)
tuningfork show $RUN
```

```
run 22d9eec8  tally-report/1#c1e9dd10.7
  task    tally-report/1
  model   gemma-4-31b
  started 10-11 11:21:06
  harness claude-code
  sandbox /home/user/project, its home /home/user
  skills  c417ecd4 (1 file(s))
 k  outcome                                                      finish      ms       tokens  think
 0  Skill(skill='tally-report')                                  tool_calls  5147.2   13754   89
 1  Read(file_path='data.csv')                                   tool_calls  6880.08  13983   329
 2  Write(content='{"count":4,"total":61.5}',                    tool_calls  8244.24  14265   246
    file_path='report.json')
 3  I have summarized `data.csv` into `report.json`. - **Item    stop        6289.18  14429   160
    Count:** 4 -… (+24 chars)
```

This run is the batch in small: it loaded the skill, read the file, wrote the report with
`count` and `total`, and answered. The write is where the keys were chosen. Its number
depends on the path a run took — the run that checked for the file first wrote at step 3 —
so name the step by the tool the model called there instead: `Write` is the first step where
it called Write.

*think* is the length of the model's reasoning before each answer, in characters; this run
reasoned before every one. `none` would mark a tool call made with no reasoning at all, in a
run that reasons at other steps.

```bash
tuningfork step $RUN Write
```

```
context
  [0] system: x-anthropic-billing-header: cc_version=2.1.281.606; cc_entrypoint=sdk-cli; You are a
Claude agent, b…
  [1] user: <system-reminder> Attribution for git commits and pull requests you create from here on
(this replac…
  [2] system: # Environment You have been invoked in the following environment:   - Primary working
directory: /ho…
  [3] assistant: tool_calls(Skill)
  [4] tool: Launching skill: tally-report
  [5] user: Base directory for this skill: /home/user/.claude/skills/tally-report  # Tally report
Use this when…
  [6] system: <total_tokens>14986246 tokens left</total_tokens>
  [7] assistant: tool_calls(Read)
  [8] tool: 1   id,item,amount 2        1,widget,10.00 3        2,gadget,25.50 4
3,widget,6.00 5 4,doohickey,20.00 6
  [9] system: <total_tokens>14986017 tokens left</total_tokens>

response
  reasoning_content: The CSV `data.csv` has the following content:
`id,item,amount`
`1,widget,10.00`
`2,gadget,25.50`
`3,widget,6.00`
`4,doohickey,20.00`

Line items count: 4
Total amount: 10.00 + 25.50 + 6.00 + 20.00 = 61.50

Now I will write this to `report.json`.

  tool_call Write({"content": "{\n  \"count\": 4,\n  \"total\": 61.50\n}\n", "file_path":
"/home/user/project/report.json"})
```

(The header lines above the context are left out here.) Message 5 is the skill as the model
read it: after `Base directory for this skill:`, the body of `SKILL.md`, frontmatter stripped,
from where Claude Code installs a skill — in the sandbox, every run's home is `/home/user`.
The reasoning works out the count and the total, and says nothing of what to call them; the
call calls them `count` and `total`.

What was on disk when the model made that call? Every call of a run tuningfork launched
finds the workspace snapshotted:

```bash
tuningfork files $RUN Write
```

```
run 22d9eec8 before step 2  (snapshot 4eb56754)
   file      bytes
   data.csv  77
```

Only the input: the report does not exist yet. `tuningfork files $RUN end` lists what the
run left behind, `--diff` shows what changed since the step before, and `--out <dir>`
writes the files out.

#### With Hermes Agent

Hermes writes with `write_file`, so name its step by that tool, and keep the run in a variable
of its own:

```bash
HRUN=$(tuningfork runs -q -n 1 --called write_file)
tuningfork show $HRUN
```

```
run 5aa33251  tally-report/1#7afee5f1.6
  task    tally-report/1
  model   gemma-4-31b
  started 10-11 11:49:07
  harness hermes
  sandbox /home/user/project, its home /home/user
  skills  c417ecd4 (1 file(s))
 k  outcome                                                     finish      ms        tokens  think
 0  execute_code(code='<ast:d1e4a258>')                         tool_calls  15485.46  14256   629
 1  read_file(path='data.csv')                                  tool_calls  7217.61   14253   461
 2  read_file(path='data.csv')                                  tool_calls  6048.02   14288   230
 3  terminal(command='cat data.csv')                            tool_calls  10436.39  14598   1029
 4  write_file(content='{"count":4,"total":61.5}',              tool_calls  10101.0   14674   581
    path='report.json')
 5  I have summarized `data.csv` into `report.json` as          stop        5705.57   14589   -
    requested. - **Line… (+92 chars)
```

Its report came at step 4. The script at step 0 printed nothing. Hermes answered the two reads
after it as repeats — the script had read the file already — and a `cat` showed the data. Then
the model wrote `count` and `total`.

```bash
tuningfork step $HRUN write_file
```

```
context
  [0] system: You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are
helpful, knowled…
  [1] user: Use the tally-report skill to summarize data.csv into report.json.
  [2] assistant: tool_calls(execute_code)
  [3] tool: {"status": "success", "output": "", "tool_calls_made": 1, "duration_seconds": 0.1}
  [4] assistant: tool_calls(read_file)
  [5] tool: {"status": "unchanged", "message": "File unchanged since last read. The content from the
earlier rea…
  [6] assistant: tool_calls(read_file)
  [7] tool: {"error": "BLOCKED: You have called read_file on this exact region 3 times and the file
has NOT chan…
  [8] assistant: tool_calls(terminal)
  [9] tool: {"output":
"id,item,amount\n1,widget,10.00\n2,gadget,25.50\n3,widget,6.00\n4,doohickey,20.00", "exit…

response
  reasoning_content: The user wants me to use the `tally-report` skill to summarize `data.csv` into
`report.json`.

The `tally-report` skill steps are:
1. Read the CSV file named in the request (`data.csv`).
2. Work out how many line items it contains, and the total of the amount column.
3. Write the results to `report.… (+281 chars)
  tool_call write_file({"content":"{\n  \"count\": 4,\n  \"total\":
61.5\n}\n","path":"report.json"})
```

No message carries the skill: Hermes preloads it, frontmatter and all, into the system prompt,
message 0. The reasoning recites the skill's steps, and the call names the keys `count` and
`total`, as Claude Code's did. Only the input was on disk:

```bash
tuningfork files $HRUN write_file
```

```
run 5aa33251 before step 4  (snapshot f35e6df0)
   file      bytes
   data.csv  77
```

### 5. Bad luck, or the wording? Resample that step

Send that step's exact context to the model twenty more times, with everything before it
held fixed:

```bash
tuningfork resample $RUN Write -n 20
```

```
run 22d9eec8 step 2 — 20 samples from http://127.0.0.1:8081
  context   10 messages, after the result of Read
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~2.7 min

    1/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  9.0s
    2/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  8.8s
    3/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  8.6s
    ...
   20/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  8.6s

step 2, after the result of Read — 20 samples
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  Write(content='{"count":4,"total":61.5}', file_path='report.json')

experiment 7f72a08c · 2.8 min · 8.5s per sample · use `tuningfork ab 22d9eec8 2 --target <letter>
--variant <file>` to test a wording change
```

All twenty wrote the same report with `Write`: `count` 4 and `total` 61.5. Not bad luck:
given this skill, the model never picks the keys the check wants.

Every sample reasoned before it answered, and the server processed the whole prompt every
time, 14,000 tokens: about eight and a half seconds a sample. When some samples answer
without reasoning, a *reasoned* column says which.

#### With Hermes Agent

```bash
tuningfork resample $HRUN write_file -n 20
```

```
run 5aa33251 step 4 — 20 samples from http://127.0.0.1:8081
  context   10 messages, after the result of terminal
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 14,410 tokens, then answers: the recorded step
took 10.1s

    1/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)  11.1s
    2/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)  9.8s
    3/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)  9.9s
    ...
   20/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)  10.2s

step 4, after the result of terminal — 20 samples
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  write_file(content='{"count":4,"total":61.5}', path='report.json')

experiment 0f368795 · 3.3 min · 10.0s per sample · use `tuningfork ab 5aa33251 4 --target <letter>
--variant <file>` to test a wording change
```

Twenty `write_file` calls, every one with `count` and `total`. For a Hermes step the plan gives
the prompt each sample processes and the recorded step's time in place of an estimate: about
ten seconds a sample.

### 6. Test a rewording at that step

The variant pins the output down, with exact key names and an example:

```bash
diff examples/tally-report/SKILL.md examples/variants/tally-report/SKILL.v2.md
```

Swap it in at the write and sample both wordings, twenty each. The target — *calls `Write`,
and mentions `row_count`* — is declared before sampling, so it gets one exact test rather
than a search through whatever happens to differ:

```bash
tuningfork ab $RUN Write --variant examples/variants/tally-report/SKILL.v2.md \
  --target-rules '[{"if": "calls", "value": "Write"}, {"if": "contains", "value": "row_count"}]' -n 20
```

```
run 22d9eec8 step 2 — 20 samples per arm from http://127.0.0.1:8081
  context   10 messages, after the result of Read
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~5.5 min
  swapping
/home/you/tuningfork-guide/batches/c1e9dd10/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md
            found: body, lines 6-16, in message 3 (user), part 1 at chars 71-411
  target    a behaviour that calls Write, and mentions "row_count" (declared before sampling)

  original   1/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)
8.5s
  original   2/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)
7.9s
  ...
  variant    1/20  Write(content='{"amount_total":61.5,"row_count":4}', file_path='… (+13 chars)
7.8s
  variant    2/20  Write(content='{"amount_total":61.5,"row_count":4}', file_path='… (+13 chars)
8.9s
  ...

   behaviour                                   original               variant                p adj.
A  Write(content='{"amount_total":61.5,"row_…  0/20   0% [0%-16%]     20/20 100% [84%-100%]  2.9e-11
   file_… (+19 chars)
B  Write(content='{"count":4,"total":61.5}',   20/20 100% [84%-100%]  0/20   0% [0%-16%]     2.9e-11
   file_path='repor… (+8 chars)
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 2 comparisons

target (declared before sampling): a behaviour that calls Write, and mentions "row_count"
  original  0/20   0% [0%-16%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 2 model-written message(s) before this step were generated under the original wording, and both
arms inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
experiments 0fb79204 (original) and c8d283da (variant) · 5.5 min
```

tuningfork found the skill in the recorded request — the body, lines 6–16 of the file, where
Claude Code put it — replaced exactly that text, and sent both versions; nothing else
differed. It needed no `--original`: every batch run records the copy of the skill it was
given, and that copy is what gets swapped out. With the original, every sample wrote
`count` and `total`, as in the resample. With the variant, every sample wrote `row_count`
and `amount_total`, with `Write`.

The target is one exact test, declared before sampling: p = 1.5 × 10⁻¹¹. It reads a call's
arguments as written, a script's code included: a sample that wrote `row_count` from a
Python script would count for `--target-match row_count`, and not for these rules, which ask
for what the step should do — a `Write` with the right key. The per-behaviour table tests
every behaviour seen, two here, and corrects for them all.

#### With Hermes Agent

A Hermes run can write the report from a script, and a target reads a script's code, so ask for
the keys alone:

```bash
tuningfork ab $HRUN write_file --variant examples/variants/tally-report/SKILL.v2.md \
  --target-rules '[{"if": "contains", "value": "row_count"}, {"if": "contains", "value": "amount_total"}]' -n 20
```

```
run 5aa33251 step 4 — 20 samples per arm from http://127.0.0.1:8081
  context   10 messages, after the result of terminal
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 14,410 tokens, then answers: the recorded step
took 10.1s
  swapping
/home/you/tuningfork-guide/batches/7afee5f1/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md
            found: full, lines 1-16, in message 0 (system) at chars 10721-11159
  target    a behaviour that mentions "row_count", and mentions "amount_total" (declared before
sampling)

  original   1/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)
10.1s
  original   2/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)
9.7s
  ...
  variant    1/20  write_file(content='{"amount_total":61.5,"row_count":4}', path='… (+13 chars)
11.1s
  variant    2/20  write_file(content='{"amount_total":61.5,"row_count":4}', path='… (+13 chars)
10.4s
  ...

   behaviour                                   original               variant                p adj.
A  write_file(content='{"amount_total":61.5,…  0/20   0% [0%-16%]     20/20 100% [84%-100%]  2.9e-11
   … (+19 chars)
B  write_file(content='{"count":4,"total":61…  20/20 100% [84%-100%]  0/20   0% [0%-16%]     2.9e-11
   path='repor… (+8 chars)
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 2 comparisons

target (declared before sampling): a behaviour that mentions "row_count", and mentions
"amount_total"
  original  0/20   0% [0%-16%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 4 model-written message(s) before this step were generated under the original wording, and both
arms inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
experiments 17ed11cd (original) and 01eb340b (variant) · 6.8 min
```

None of the original's samples had the keys, and all twenty of the variant's did, p = 1.5 ×
10⁻¹¹; every sample wrote with `write_file`. tuningfork found the skill where Hermes preloads
it, the whole file, lines 1–16, in the system prompt, and swapped it there.

### 7. Follow the rewording to the end of the run

That A/B tested one decision: everything before the write was held fixed, and nothing after
it was looked at. `--to-end` follows every sample to the finish. Each sample becomes a
*fork*: Claude Code, relaunched on the same task in a fresh workspace, is answered from the
recording up to the write — it runs the recorded `Skill` and `Read` calls again itself, but
the model is not asked — then runs live from the write on, with the variant swapped in,
until the task's check decides:

```bash
tuningfork ab $RUN Write --variant examples/variants/tally-report/SKILL.v2.md --to-end -n 20
```

```
run 22d9eec8 from step 2 — 20 forks per arm through http://127.0.0.1:8787
  context   after the result of Read
  replays   2 recorded call(s), answered from the recording; live from step 2
  harness   claude-code, model gemma-4-31b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
  swapping
/home/you/tuningfork-guide/batches/c1e9dd10/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md, from step 2 on
            found: body, lines 6-16, in message 3 (user), part 1 at chars 71-411
  target    the task's check passes (declared by the task, before any fork ran)

  original   1/20 c55c840a  fail  14.6s  Write(content='{"count":4,"total":61.5}',
file_path='report.… (+6 chars)
  original   2/20 d4fbb55f  fail  15.8s  Write(content='{"count":4,"total":61.5}',
file_path='report.… (+6 chars)
  ...
  variant    1/20 54d26278  pass  15.1s  Write(content='{"amount_total":61.5,"row_count":4}',
file_pa… (+17 chars)
  variant    2/20 0f43798a  pass  15.5s  Write(content='{"amount_total":61.5,"row_count":4}',
file_pa… (+17 chars)
  ...

target: the task's check passes, end to end from step 2
  original  0/20   0% [0%-16%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher

original: what the model did at step 2
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  Write(content='{"count":4,"total":61.5}', file_path='report.json')

variant: what the model did at step 2
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  Write(content='{"amount_total":61.5,"row_count":4}',
                            file_path='report.json')

Steps before 2 were answered from the recording in every fork — the same history, generated under
the original wording; from step 2 to the end each fork ran live, with each arm's wording, and the
task's check decided. An edit that would also change earlier steps still needs full runs to confirm.
experiments 2a46d47f (original) and 33e001e7 (variant) · 10.5 min
```

Every original fork wrote `count` and `total`, and failed. Every variant fork wrote
`row_count` and `amount_total` at once, and passed. Here the verdict was settled at the
write; in a task with more to do after the decision, the forks show whether what follows
undoes it, or gets there another way.

Each fork saw the folders its source saw — the sandbox gives every launch the same — so the
prompt an original fork sent at the step was the recorded one, word for word. Without
bubblewrap a fork runs in folders of its own, whose names are in every request it sends; see
[Will I get the same numbers?](#will-i-get-the-same-numbers).

A fork whose harness asks for something the recording never answered — its history has
changed — is left out of the rates, and says where it left the recording; none did here.
Each fork is a run of its own, kept out of `tuningfork runs` and of the task's pass rates:
`tuningfork show 54d26278` lists its steps, the first two marked as answered from the
recording.

#### With Hermes Agent

```bash
tuningfork ab $HRUN write_file --variant examples/variants/tally-report/SKILL.v2.md --to-end -n 20
```

```
run 5aa33251 from step 4 — 20 forks per arm through http://127.0.0.1:8787
  context   after the result of terminal
  replays   4 recorded call(s), answered from the recording; live from step 4
  harness   hermes, model gemma-4-31b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
  swapping
/home/you/tuningfork-guide/batches/7afee5f1/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md, from step 4 on
            found: full, lines 1-16, in message 0 (system) at chars 10721-11159
  target    the task's check passes (declared by the task, before any fork ran)

  original   1/20 a99cb222  fail  27.1s  write_file(content='{"count":4,"total":61.5}',
path='report.… (+6 chars)
  original   2/20 41ed9285  fail  23.3s  write_file(content='{"count":4,"total":61.5}',
path='report.… (+6 chars)
  ...
  variant    1/20 0020a567  pass  22.1s  write_file(content='{"amount_total":61.5,"row_count":4}',
pa… (+17 chars)
  variant    2/20 29df7d24  pass  24.0s  write_file(content='{"amount_total":61.5,"row_count":4}',
pa… (+17 chars)
  ...

target: the task's check passes, end to end from step 4
  original  0/20   0% [0%-16%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-11 → significant at α=0.05, variant higher

original: what the model did at step 4
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  write_file(content='{"count":4,"total":61.5}', path='report.json')

variant: what the model did at step 4
   count  rate  95% CI      behaviour
A  20/20  100%  [84%-100%]  write_file(content='{"amount_total":61.5,"row_count":4}',
                            path='report.json')

Steps before 4 were answered from the recording in every fork — the same history, generated under
the original wording; from step 4 to the end each fork ran live, with each arm's wording, and the
task's check decided. An edit that would also change earlier steps still needs full runs to confirm.
experiments e3fe7d73 (original) and 2102ad6d (variant) · 16.2 min
```

Each fork replayed the four calls before the write and went live there. Every original fork
failed and every variant fork passed, as with Claude Code. A Hermes fork took about 25 seconds,
so the forty took 16 minutes.

### 8. Confirm with full runs

A fork still starts from history the original wording produced — here, two calls that
loaded the skill and read the data. Full runs put the variant in from the very first step.
`--skill` installs a file as the skill's `SKILL.md` in every run:

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness claude-code --model gemma-4-31b \
  --skill examples/variants/tally-report/SKILL.v2.md
```

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report  SKILL.md replaced by examples/variants/tally-report/SKILL.v2.md
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  28.9s
  run 1  pass  30.0s
  run 2  pass  31.4s
  run 3  pass  30.8s
  run 4  pass  29.4s
  run 5  pass  27.4s
  run 6  pass  30.9s
  run 7  pass  31.0s

8/8  100%  [68%-100%] passed
...
```

Eight out of eight, against none out of eight before, and the two intervals do not overlap.
Every run loaded the skill, read the data, and wrote `row_count` and `amount_total` with
`Write`.

#### With Hermes Agent

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness hermes --model gemma-4-31b \
  --skill examples/variants/tally-report/SKILL.v2.md
```

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report  SKILL.md replaced by examples/variants/tally-report/SKILL.v2.md
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  28.2s
  run 1  pass  103.6s
  run 2  pass  27.0s
  run 3  pass  28.0s
  run 4  pass  86.9s
  run 5  pass  123.3s
  run 6  pass  26.4s
  run 7  pass  26.8s

8/8  100%  [68%-100%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                                         then passed
A  4/8    50%   [22%-78%]  read_file(path='data.csv')                        4/4  100%  [51%-100%]
B  1/8    12%   [2%-47%]   execute_code(code='<ast:32b8335b>')               1/1  100%  [21%-100%]
C  1/8    12%   [2%-47%]   execute_code(code='<ast:9cf12648>')               1/1  100%  [21%-100%]
D  1/8    12%   [2%-47%]   execute_code(code='<ast:ad5042b5>')               1/1  100%  [21%-100%]
E  1/8    12%   [2%-47%]   search_files(pattern='data.csv', target='files')  1/1  100%  [21%-100%]
```

Eight out of eight, as with Claude Code. The runs still began in different ways — a read, a
script, a search — and every way passed. Three took longer, 87 to 123 seconds, going through
seven or eight calls of scripts and reads before they wrote the right keys.

### 9. Catch the failure live

A breakpoint stops a call as it happens. Rather than start a new run and wait for it to reach
the write, restart this one from it: `tuningfork fork` replays a run up to a step, which takes
a second or two since the model is not asked, then hands it to the model with breakpoints
armed. With the original skill, stop Claude Code's `Write` of the report before it acts on it:

```bash
tuningfork break add --tool Write --path '*.json'
tuningfork fork $RUN Write
```

```
breakpoint c203477d — after the model: Write on *.json
run 22d9eec8 from step 2 — 1 fork through http://127.0.0.1:8787
  context   after the result of Read
  replays   2 recorded call(s), answered from the recording; live from step 2
  harness   claude-code, model gemma-4-31b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
```

The fork now waits for the model, and is held when it writes the report. In the web UI —
opened from the address `tuningfork serve` printed, token and all — go to **Live**, change
the keys in the held call's `content` to `row_count` and `amount_total`, and click *Continue
with edits*. Or, from a **fourth terminal** set up like the third, do what Part 1 did:

```bash
tuningfork hold wait
```

```
hold 99fdb6ef — after the model, run a6c4b02b step 2, held 0.3s
  reason  after the model: Write on *.json
  … 7 earlier message(s)
  assistant: tool_calls(Read)
  tool: 1       id,item,amount 2        1,widget,10.00 3        2,gadget,25.50 4
3,widget,6.00 5 4,doohickey,20.00 6
  system: <total_tokens>14986017 tokens left</total_tokens>

the model answered
  reasoning_content: The `data.csv` file contains:
- Header: `id,item,amount`
- Line 1: `1,widget,10.00`
- Line 2: `2,gadget,25.50`
- Line 3: `3,widget,6.00`
- Line 4: `4,doohickey,20.00`

Count of line items: 4
Total amount: 10.00 + 25.50 + 6.00 + 20.00 = 61.50

I will now write this to `report.json`.
  tool_call Write({"content": "{\n  \"count\": 4,\n  \"total\": 61.50\n}\n", "file_path":
"/home/user/project/report.json"})

tuningfork hold continue 99fdb6ef · step · drop — to edit the response: hold show 99fdb6ef --json >
f.json, edit, hold continue 99fdb6ef --edited f.json
```

```bash
HOLD=$(tuningfork hold list -q)
tuningfork hold show $HOLD --json > held.json
python - <<'EOF'
import json
held = json.load(open("held.json"))
call = held["choices"][0]["message"]["tool_calls"][0]["function"]
arguments = json.loads(call["arguments"])
arguments["content"] = '{"row_count": 4, "amount_total": 61.5}'
call["arguments"] = json.dumps(arguments)
json.dump(held, open("held.json", "w"))
EOF
tuningfork hold continue $HOLD --edited held.json
```

```
99fdb6ef: continue with your edit
```

Claude Code writes your version, and the fork passes — reported as not counted, because the
edit, not the skill, made it pass. Back in the third terminal:

```
  a6c4b02b  pass  16.3s  Write(content='{"count":4,"total":61.5}', file_path='report.… (+6 chars)
edited at a breakpoint (step 2) — not counted

`tuningfork show a6c4b02b` for the fork's steps; its first 2 were answered from the recording.
```

The line shows what the model did at the write, before your edit. `tuningfork show a6c4b02b`
marks steps 0 and 1 as replayed and step 2 as edited, and `tuningfork files a6c4b02b end`
finds the report you wrote. Remove the breakpoint with `tuningfork break list` and
`tuningfork break rm <id>`. Then compare the two wordings, each with its own rate — never
one pooled across them:

```bash
tuningfork tasks
```

```
tally-report/1 · claude-code · gemma-4-31b
 skill     runs  check passed
 c417ecd4  8     0/8  0%  [0%-32%]
 6bae9525  8     8/8  100%  [68%-100%]  SKILL.md from SKILL.v2.md

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

Forks are not in it: each is a sample of an experiment, conditional on the history it
replayed, not a run of the task.

#### With Hermes Agent

Stop Hermes's write, `write_file`, the same way:

```bash
tuningfork break add --tool write_file --path '*.json'
tuningfork fork $HRUN write_file
```

```
breakpoint 5b808ee9 — after the model: write_file on *.json
run 5aa33251 from step 4 — 1 fork through http://127.0.0.1:8787
  context   after the result of terminal
  replays   4 recorded call(s), answered from the recording; live from step 4
  harness   hermes, model gemma-4-31b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
```

```bash
tuningfork hold wait
```

```
hold a5013fb2 — after the model, run 80cef561 step 4, held 0.1s
  reason  after the model: write_file on *.json
  … 7 earlier message(s)
  tool: {"error": "BLOCKED: You have called read_file on this exact region 3 times and the file has
NOT changed. STOP …
  assistant: tool_calls(terminal)
  tool: {"output":
"id,item,amount\n1,widget,10.00\n2,gadget,25.50\n3,widget,6.00\n4,doohickey,20.00", "exit_code": 0,…

the model answered
  reasoning_content: The user wants me to use the `tally-report` skill.
The `tally-report` skill instructions are:
1. Read the CSV file named in the request.
2. Work out how many line items it contains, and the total of the amount column.
3. Write the results to `report.json` in the working directory.
Keep the report sm… (+258 chars)
  tool_call write_file({"content":"{\n  \"count\": 4,\n  \"total\":
61.50\n}\n","path":"report.json"})

tuningfork hold continue a5013fb2 · step · drop — to edit the response: hold show a5013fb2 --json >
f.json, edit, hold continue a5013fb2 --edited f.json
```

The recipe above edits the held call as it is, since `write_file` takes `content` too:

```bash
HOLD=$(tuningfork hold list -q)
tuningfork hold show $HOLD --json > held.json
python - <<'EOF'
import json
held = json.load(open("held.json"))
call = held["choices"][0]["message"]["tool_calls"][0]["function"]
arguments = json.loads(call["arguments"])
arguments["content"] = '{"row_count": 4, "amount_total": 61.5}'
call["arguments"] = json.dumps(arguments)
json.dump(held, open("held.json", "w"))
EOF
tuningfork hold continue $HOLD --edited held.json
```

```
a5013fb2: continue with your edit
```

Back in the third terminal:

```
  80cef561  pass  28.1s  write_file(content='{"count":4,"total":61.5}', path='report.… (+6 chars)
edited at a breakpoint (step 4) — not counted

`tuningfork show 80cef561` for the fork's steps; its first 4 were answered from the recording.
```

The fork passes, reported as not counted. With both harnesses recorded, `tuningfork tasks`
prints a second table, for Hermes: 0/8 with the original wording, 8/8 with the variant.

### 10. Look at all of it in the browser

The proxy serves the web UI at the address it printed, with its token:

- **Runs** — the task split into its conditions, one per harness and wording, each with its
  pass rate; *Where do they diverge?* on each compares runs of that condition only.
- **A run** — its conversation top to bottom, each step's answer where it happened,
  reasoning included, and the skill marked inside it and mapped to `SKILL.md` line by
  line. Click the `Write` step for its details and the workspace as the call found it;
  *Exact context* shows everything it was sent.
- **Experiments** — the resample, the A/B, and the A/B run to the end, each fork linked.
  A fork's steps say which were answered from the recording and where it went live; the
  step you edited is flagged, and can be shown as the harness saw it.
- **Live** — breakpoints, and any call being held.
- **Jobs**, **Scenarios** and **Connections** — the next section.

### 11. The same from the browser

Everything this part did from the terminal can be started from the web UI, as jobs the
proxy runs. Open the address `tuningfork serve` printed — the one with `?token=`:

1. **Scenarios** → *Import a spec*: the absolute path of `examples/tally-report.evals.json`,
   named `tally-report`. The editor shows the spec exactly as it will be saved, keeps the
   skill folder where it is, and flags the check for what it is: a shell command that runs,
   as you, in every run's workspace.
2. *Save and run…* opens **Run a task** with the scenario chosen. Choose the harness and the
   number of runs, and read what will run — the check shown whole, the model the server
   lists — before *Start the runs*. The job's page follows each run and links it. (A
   check can judge the agent's final reply instead: *The reply contains…* in the editor,
   with no shell command. Either kind can be tried first with *Test this check…*: on a
   recorded run of the task, beside the verdict that run got, or on a reply and files you
   write. Nothing is recorded.)
3. Open a run, and on its conversation click the `Write` step. **From this step** →
   *Resample…* opens the form right there — so does **Jobs** → *Start…*, from a list of
   runs. The plan says what will be sent, through the proxy, and how long it may take. The
   finished job links the experiment.
4. *A/B a wording…* opens an editor holding the skill as that run was given it. Make the
   change `SKILL.v2.md` makes — or paste the file — name it, and say what the A/B is about:
   *What the model does matches a pattern*, `row_count`. Nothing is chosen for you. The
   plan shows where the skill sits in the context, and any part of your edit the model
   never sees at this step. The variant is saved in the recordings folder when the A/B
   starts, never beside the skill.
5. *Reach: each to the end* turns either form into forks followed to the task's check.
   *Fork from here…* stops a fork at the step for you; on the **Live** page, *Evidence at
   this call* samples the held call where it stands, and *Hypotheses, not evidence* asks
   the model about it — its answer is its own account, never a result.

**Jobs** lists all of it, as `tuningfork jobs list` does, and **Connections** shows the
server, what was measured of it, and the harnesses. Opened without the token, every page
shows everything and changes nothing.

---

## Part 3 — inside a step

Part 2 found the step that fails and the wording that fixes it. This part looks inside a
step, at the model's own tokens: how sure it was of each one, and — where runs split —
which token splits them. It uses Part 2's proxy and recordings, and needs what Part 2 does:
llama.cpp's server, and a model whose raw output tuningfork reads — Gemma 4. §2 probes a
step of Part 2's Hermes Agent runs.

### 1. How sure was the model of each token?

Take the run from Part 2 §4 again: `$RUN` still holds it in the third terminal. (In a new
terminal, set it to that run's id, as `tuningfork show` printed it there — `tuningfork runs
--called Write` would now find one of §8's runs, which were given the reworded skill.) Score
the answer its `Write` step gave:

```bash
tuningfork tokens $RUN Write --score
```

````
scoring step 2's answer (183 tokens) on http://127.0.0.1:8081…
run 22d9eec8 step 2  stream 1174b06e
  scored · grammar · top 8 · 183 tokens · rebuilt exactly
  choice of tool, at token [134]: Write 100%  the model's own chance of each, under the step's
sampler

reasoning
  The CSV `data.csv` has the following content:
  `id,item,amount`
  `1,widget,10.00`
  `2,gadget,25.50`
  `3,widget,6.00`
  `4,doohickey,20.00`

  Line items count: 4
  Total amount: 10.00 + 25.50 + 6.00 + 20.00 = 61.50

  Now I will write this to `report.json`.

tool call
  call:Write{content:<|"|>{
    "count": 4,
    "total": 61.50
  }
  <|"|>,file_path:<|"|>/home/user/project/report.json<|"|>}

where the step's sampler could most easily have drawn another token
  [5] reasoning ' `' 6% | ' content' 79% | ' file' 9%  H 1.17 bits  runner-up drawn 85%
  [16] reasoning '`' 11% | '-' 66% | '```' 12% | 'Line' 6%  H 1.63 bits  runner-up drawn 69%
  [10] reasoning ' has' 35% | ' contains' 60%  H 1.19 bits  runner-up drawn 63%
  [76] reasoning ' count' 41% | ':' 58%  H 1.09 bits  runner-up drawn 58%
  ...

Entropy counts the probability beyond the top 8 as one outcome: a lower bound. The runner-up's
chance is under the step's own sampler settings.
````

Scoring sends the step's exact context once more with the answer forced, token by token, and
reads the model's own distribution at each one: one call, about as long as the answer took.
*Rebuilt exactly* says these are the run's own tokens — the answer, written back as the raw
text the model generated, reasoning and tool-call syntax and all, has as many tokens as the
run reported. (That raw text is how Gemma 4 writes a call: `<|"|>` fences each string.) In a
terminal, the text is shaded by entropy.

The answer is short — 183 tokens, two thirds of them reasoning — and the reasoning works out
the count and the total without naming a key. The places where the step's sampler could most
easily have drawn another token are all in it — how to lay out the file's lines, "has" or
"contains" — and none touches the keys. The call was settled by then. *Choice of tool* is the
model's own chance of each tool at the first token of the call's name: `Write`, 100%. And
`tuningfork tokens $RUN Write --json` puts the first key's `count` at 99.8% and `total` at
100%; the nearest the model came to the check's key, among the eight tokens it weighed for
the first, was `rowCount`, at about one in a million. No amount of resampling this step would
find `row_count` — Part 2 §5 drew twenty and got none — so the wording has to change, as
Part 2 §6 found.

#### With Hermes Agent

Score the answer at Hermes's write the same way; `$HRUN` still holds Part 2 §4's Hermes run:

```bash
tuningfork tokens $HRUN write_file --score
```

````
scoring step 4's answer (264 tokens) on http://127.0.0.1:8081…
run 5aa33251 step 4  stream d2c7a168
  scored · grammar · top 8 · 264 tokens · rebuilt exactly
  choice of tool, at token [223]: write_file 100%  the model's own chance of each, under the step's
sampler

reasoning
  The user wants me to use the `tally-report` skill to summarize `data.csv` into `report.json`.

  The `tally-report` skill steps are:
  1. Read the CSV file named in the request (`data.csv`).
  2. Work out how many line items it contains, and the total of the amount column.
  3. Write the results to `report.json` in the working directory.

  From the `cat data.csv` output:
  ```csv
  id,item,amount
  1,widget,10.00
  2,gadget,25.50
  3,widget,6.00
  4,doohickey,20.00
  ```

  - Line items (excluding header): 4
  - Total amount: 10.00 + 25.50 + 6.00 + 20.00 = 61.50

  I need to write this to `report.json`.
tool call
  call:write_file{content:<|"|>{
    "count": 4,
    "total": 61.5
  }
  <|"|>,path:<|"|>report.json<|"|>}

where the step's sampler could most easily have drawn another token
  [53] reasoning ' (`' 9% | '.' 91%  H 0.43 bits  runner-up drawn 91%
  [249] tool call '\n' 42% | '0' 58%  H 0.98 bits  runner-up drawn 58%
  [108] reasoning 'csv' 51% | '\n' 49%  H 1.00 bits  runner-up drawn 49%
  [171] reasoning ' Total' 44% | ' Amounts' 48% | ' Amount' 8%  H 1.36 bits  runner-up drawn 48%
  ...

Entropy counts the probability beyond the top 8 as one outcome: a lower bound. The runner-up's
chance is under the step's own sampler settings.
````

The answer is longer, 264 tokens, most of them reasoning that quotes the data in a fenced block
of its own. The keys are as settled as Claude Code's: *choice of tool* gives `write_file` 100%,
and `--json` puts the first key's `count` at 99.96% and `total` at 100%. The one place in the
call where the sampler could easily have gone another way is the total's last digit: `61.5` or
`61.50`.

### 2. Which token decided it?

Where runs of a step do split, `forking-tokens` finds the token that splits them (after
[Bigelow et al., ICLR 2025](https://arxiv.org/abs/2412.07961)). At the tokens where the
step's own sampler could most easily have drawn another, it forces each likely alternative,
lets the model go on, and tests which change what it does.

Claude Code's runs in Part 2 split at their first step only on what they passed the skill, and
every path led to the same write. Hermes Agent's runs split on which tool they use first, so
this section probes one of those, from the eight Part 2 §3 recorded. If you followed Claude
Code alone, record them now (installing Hermes is under [What you need](#what-you-need)):

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness hermes --model gemma-4-31b
```

```
tally-report/1 — 8 runs
  ...
0/8  0%  [0%-32%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                                         then passed
A  3/8    38%   [14%-69%]  read_file(path='data.csv')                        0/3  0%  [0%-56%]
B  1/8    12%   [2%-47%]   execute_code(code='<ast:1916367c>')               0/1  0%  [0%-79%]
C  1/8    12%   [2%-47%]   execute_code(code='<ast:7b3b041f>')               0/1  0%  [0%-79%]
D  1/8    12%   [2%-47%]   execute_code(code='<ast:d1e4a258>')               0/1  0%  [0%-79%]
E  1/8    12%   [2%-47%]   execute_code(code='<ast:fb68b83f>')               0/1  0%  [0%-79%]
F  1/8    12%   [2%-47%]   search_files(pattern='data.csv', target='files')  0/1  0%  [0%-79%]
```

Three runs began by reading `data.csv` with `read_file`, four by running a Python script, each
its own, and one by searching for the file. All eight wrote `count` and `total`, three of them
with a count of five. Take the latest that searched *at step 0* and failed, and probe that
step. `--at-step` matters, since a run can do something else first and search later, and so
does `--failed`: a run given the variant in Part 2 §8 can search first too.

```bash
SRUN=$(tuningfork runs -q -n 1 --called search_files --at-step 0 --failed)
tuningfork forking-tokens $SRUN 0 --score
```

```
scoring step 0's answer first…
no token in the answer had a runner-up the step's sampler would draw at least 5% of the time: the
model had decided before it — while reasoning, most likely — probe the reasoning instead (`--in
reasoning`), or pick the positions yourself (`--at`)
the best in the reasoning: [94] ' and' or ',' (~69 tokens to regenerate); [133] ' in' or ' and' /
'.' (~30 tokens to regenerate); [42] ' (' or '.' (~121 tokens to regenerate); [56] ' defined' or '
outlined' (~107 tokens to regenerate); [7] ' use' or ' summarize' (~156 tokens to regenerate)
```

`--score` scored the step's answer first, as `tokens --score` does. The probe then looked in
the answer — here a tool call and nothing else — for tokens the step's own sampler could
easily have drawn otherwise, and found none: by the time the model wrote the call, the choice
was made. So it says where to look instead, the likeliest places in the reasoning, each with
the length of reasoning a forced token there would regenerate. `--in reasoning` probes them:

```bash
tuningfork forking-tokens $SRUN 0 --in reasoning
```

```
forking tokens at run 3dd05381 step 0 — after user: Use the tally-report skill to summarize data.csv
into report.json.
  base      stream 663ad97b (scored, 163 tokens)
  target    the base's own behaviour: search_files(pattern='data.csv', target='files') (declared
before sampling)
  arms      11 forced tokens × 10 = 110 continuations
  power     a forced token is found to change the outcome when about 7 of its 10 continuations
change; smaller shifts need more (-n)
  estimate  ~9,960 tokens to generate, ~7.8 min at the step's own speed
            and each continuation first processes the whole prompt, 13,725 tokens
  reasoning positions regenerate the rest of the reasoning: each continuation is long
  [7] reasoning ' use' 58% → ' summarize' 42%
  [42] reasoning ' (' 39% → '.' 53%
  [56] reasoning ' defined' 52% → ' outlined' 44%
  [94] reasoning ' and' 35% → ',' 65%
  [133] reasoning ' in' 22% → ' and' 60% | '.' 19%

  7:' summarize'  1/10  read_file(path='data.csv')  7.9s
  ...
  133:' in' 10/10  search_files(pattern='data.csv', target='files')  48.0s

target (declared before sampling): the base's own behaviour: search_files(pattern='data.csv',
target='files')
  the baseline arms reached it 18/50  36% [24%-50%]
  at [7], [42], [56] and [94] the step was still undecided: from there the run's own token led
elsewhere more often than to its behaviour, so a later token decided it (`tuningfork tokens` shows
the tokens after, and --at probes them)

[7] reasoning  …<|channel>thought
The user wants to⟦ use⟧
  as run  ' use' 58%               0/10   0% [0%-28%]
  forced  ' summarize' 42%         2/10  20% [6%-51%]   p=0.474 → Holm 1.000
  the chance of the target from here, weighted by each token's chance: 8%
...
[133] reasoning  …, I'll check if `data.csv` exists⟦ in⟧
  as run  ' in' 22%                10/10 100% [72%-100%]
  forced  ' and' 60%               0/10   0% [0%-28%]   p=1.1e-05 → Holm 6.5e-05  flips away from
the target
  forced  '.' 19%                  9/10  90% [60%-98%]   p=1.000 → Holm 1.000
  the chance of the target from here, weighted by each token's chance: 38%

behaviours, across the probe
   count   rate  95% CI     behaviour
A  66/110  60%   [51%-69%]  read_file(path='data.csv')
B  40/110  36%   [28%-46%]  search_files(pattern='data.csv', target='files')
C  1/110   1%    [0%-5%]    execute_code(code='<ast:49e7ce2d>')
D  1/110   1%    [0%-5%]    execute_code(code='<ast:926c89bb>')
E  1/110   1%    [0%-5%]    execute_code(code='<ast:de4c3c2e>')
F  1/110   1%    [0%-5%]    execute_code(code='<ast:f65bd3b2>')

1 of 6 forced tokens changed what the model did (Holm-corrected over the 6 planned before sampling;
α=0.05).

experiment d1c77d07 · 12.0 min · `tuningfork experiment d1c77d07` shows it again
```

One word decided it. At [133] — "First, I'll check if `data.csv` exists ⟦in⟧" — the run's own
` in`, a token the sampler draws 22% of the time, led to `search_files` ten times in ten,
forced again as the baseline every alternative is compared with. Forced to ` and`, the word
the model was likeliest to write there, it read the file with `read_file` ten times in ten:
"exists and" goes on to read it. Fisher's exact test puts that at p = 1.1 × 10⁻⁵, and it
holds after the Holm correction over all six tests planned before sampling. Forced to `.`,
ending the sentence there, it searched nine times in ten, as the run did. The target is fixed
before sampling too: by default, the base's own behaviour.

Before [133] nothing was settled. At [7], [42], [56] and [94] the run's own word led to
`read_file` more often than to the search — the probe says so, and that a later token
decided. At [7], early in the first sentence, the run's own ` use` searched none of ten
times, and the chance of the search from there, weighted by each token's chance, was 8%; by
[133] it was 38%. No alternative before [133] moved it significantly: ten continuations an
arm can detect only large shifts, as the power line said.

The estimate, 7.8 minutes, counted the tokens to generate at the step's own speed. The prompt
each continuation processes first, 13,725 tokens, it names without timing — about four
seconds apiece on this model — and the probe took 12.0. `tuningfork tokens $SRUN 0` shows the
whole answer:

```
run 3dd05381 step 0  stream 663ad97b
  scored · grammar · top 8 · 163 tokens · rebuilt exactly
  choice of tool, at token [143]: search_files 100%  the model's own chance of each, under the
step's sampler

reasoning
  The user wants to use the `tally-report` skill to summarize `data.csv` into `report.json`.
  The `tally-report` skill is already preloaded (as per the system prompt).
  I need to follow the steps defined in the `tally-report` skill:
  1. Read the CSV file named in the request (`data.csv`).
  2. Work out how many line items it contains and the total of the amount column.
  3. Write the results to `report.json` in the working directory.

  First, I'll check if `data.csv` exists in the current working directory.
tool call
  call:search_files{pattern:<|"|>data.csv<|"|>,target:<|"|>files<|"|>}

where the step's sampler could most easily have drawn another token
  [94] reasoning ' and' 35% | ',' 65%  H 0.94 bits  runner-up drawn 65%
  [133] reasoning ' in' 22% | ' and' 60% | '.' 19%  H 1.37 bits  runner-up drawn 60%
  [42] reasoning ' (' 39% | '.' 53% | ',' 4%  H 1.45 bits  runner-up drawn 55%
  [56] reasoning ' defined' 52% | ' outlined' 44%  H 1.24 bits  runner-up drawn 46%
  [7] reasoning ' use' 58% | ' summarize' 42%  H 0.99 bits  runner-up drawn 42%
  [6] reasoning ' to' 63% | ' me' 37%  H 0.95 bits  runner-up drawn 37%
  [50] reasoning 'I' 51% | 'The' 27% | 'According' 15% | 'Looking' 7%  H 1.74 bits  runner-up drawn
27%
  [120] reasoning 'First' 68% | 'I' 25% | 'Plan' 4%  H 1.25 bits  runner-up drawn 25%
  ...
```

The reasoning lists the skill's three steps, then decides in its last sentence to check that
the file exists first — and *how*, by one word: where the model would most likely have gone
on "exists and", and read it, this run wrote "exists in the current working directory", and
the search followed.

A probe takes the five likeliest places by default; `--at` probes any position by hand. Take
[120], eighth on the list — the "First" that opens the last sentence:

```bash
tuningfork forking-tokens $SRUN 0 --at 120
```

```
forking tokens at run 3dd05381 step 0 — after user: Use the tally-report skill to summarize data.csv
into report.json.
  base      stream 663ad97b (scored, 163 tokens)
  target    the base's own behaviour: search_files(pattern='data.csv', target='files') (declared
before sampling)
  arms      4 forced tokens × 10 = 40 continuations
  power     a forced token is found to change the outcome when about 6 of its 10 continuations
change; smaller shifts need more (-n)
  estimate  ~1,720 tokens to generate, ~1.3 min at the step's own speed
            and each continuation first processes the whole prompt, 13,725 tokens
  reasoning positions regenerate the rest of the reasoning: each continuation is long
  [120] reasoning 'First' 68% → 'I' 25% | 'Plan' 4% | 'Let' 1%

  120:'I'    1/10  read_file(path='data.csv')  4.7s
  ...
  120:'First' 10/10  read_file(path='data.csv')  50.3s

target (declared before sampling): the base's own behaviour: search_files(pattern='data.csv',
target='files')
  the baseline arms reached it 3/10  30% [11%-60%]
  at [120] the step was still undecided: from there the run's own token led elsewhere more often
than to its behaviour, so a later token decided it (`tuningfork tokens` shows the tokens after, and
--at probes them)

[120] reasoning  …report.json` in the working directory.

⟦First⟧
  as run  'First' 68%              3/10  30% [11%-60%]
  forced  'I' 25%                  0/10   0% [0%-28%]   p=0.211 → Holm 0.632
  forced  'Plan' 4%                0/10   0% [0%-28%]   p=0.211 → Holm 0.632
  forced  'Let' 1%                 0/10   0% [0%-28%]   p=0.211 → Holm 0.632
  the chance of the target from here, weighted by each token's chance: 21%

behaviours, across the probe
   count  rate  95% CI     behaviour
A  35/40  88%   [74%-95%]  read_file(path='data.csv')
B  3/40   8%    [3%-20%]   search_files(pattern='data.csv', target='files')
C  1/40   2%    [0%-13%]   execute_code(code='<ast:54d4c953>')
D  1/40   2%    [0%-13%]   execute_code(code='<ast:852d48ec>')

0 of 3 forced tokens changed what the model did (Holm-corrected over the 3 planned before sampling;
α=0.05).

experiment 68e55106 · 3.6 min · `tuningfork experiment 68e55106` shows it again
```

Thirteen tokens before the decision, the step was still open: from the run's own "First", three
continuations in ten searched, and no alternative changed that significantly — each searched
none of ten times, against three: p = 0.21 alone. (At a position you pick, the likeliest
alternatives are forced even when the sampler would seldom draw one: `Let` had 1%.) The probe
finds the token that decides a step when there is one, and says where the step was still
undecided. Here the choice made no difference to the outcome — every Hermes run failed at the
write — but it shows how far a run's path can turn on one unlikely word.

### 3. The same in the browser

On a run's conversation, each step's answer has **Answer | Tokens**. *Tokens* shows the
heatmap — by entropy, or by how close the top two were — with the reasoning folded, and how
many of the likeliest forks are in it. An answer not scored yet offers *Score this answer*.
Click a token for what else the model weighed there; *Select to probe*, then *Find forking
tokens at these…*, opens the probe's form in the step, those positions filled in. The form
shows its plan before anything runs: the tokens it will force, what counts as the same
behaviour, what it can detect with that many continuations, and how long it may take.

What counts as the same behaviour can be the answer's own, a regular expression, or a
behaviour you describe: *Calls the tool* `search_files`, *Mentions* `data.csv`, every
condition holding. Whichever you choose, the form sets it against every behaviour already
seen at the step — the run's own, other runs of the task there, the samples drawn there —
and marks which would count, before anything is sampled.

The result is on the experiment's page: the answer it forked from, with the probed tokens
ringed — solid where a forced token changed what the model did — then each position, the
run's own token and each forced one, with where it flips. *In the answer* opens a position's
token there; *On the step* opens it on the step's own Tokens view, where every token a probe
forced is ringed and says what each probe found.

---

## Part 4 — let tuningfork propose the fix

Part 2 tested a fix written in advance. Here tuningfork gathers fixes itself — from a model
shown the evidence, from you, or from an agent — and screens each at the failing step
against the wording the run was given, validates a survivor with full runs, and writes it
into the skill once you approve it. It uses Part 2's model server and recordings: §1–4 run
no harness, §5 runs thirty for each harness. About half an hour for each.

### 1. Say what the step should do

A *refinement* fixes one step and what the model should do there, before anything is
proposed — as an A/B declares its target before it samples. Take the run from Part 2 §4
again — `$RUN`, as in Part 3 §1:

```bash
tuningfork refine start $RUN Write --target-rules '[{"if": "calls", "value": "Write"},
  {"if": "contains", "value": "row_count"}, {"if": "contains", "value": "amount_total"}]'
```

```
refinement 832f0fce — open
  checkpoint  run 22d9eec8 step 2 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/c1e9dd10/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

next: ask for proposals — or give your own wording — then screen them:
  tuningfork refine propose 832f0fce --builtin
  tuningfork refine propose 832f0fce --file SKILL.md --why '…'
  tuningfork refine screen 832f0fce
```

It keeps its own copy of the skill as the run was given it, and every proposal edits that
copy: nothing here writes to `examples/tally-report`.

#### With Hermes Agent

At Hermes's write, with the A/B's target, which counts the keys however they are written:

```bash
tuningfork refine start $HRUN write_file --target-rules '[{"if": "contains", "value": "row_count"},
  {"if": "contains", "value": "amount_total"}]'
```

```
refinement 338504b1 — open
  checkpoint  run 5aa33251 step 4 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/7afee5f1/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

next: ask for proposals — or give your own wording — then screen them:
  tuningfork refine propose 338504b1 --builtin
  tuningfork refine propose 338504b1 --file SKILL.md --why '…'
  tuningfork refine screen 338504b1
```

### 2. What a proposer is shown

```bash
REF=$(tuningfork refine list -q -n 1)
tuningfork refine show $REF --packet
```

```
{
  "sha": "18ba4acdafccd377",
  "target": {
    "description": "a behaviour that calls Write, and mentions \"row_count\", and mentions \"amount_total\""
  },
  "skill": {
    "name": "tally-report",
...
      "14: 3. Write the results to `report.json` in the working directory.",
      "15: ",
      "16: Keep the report small — just the count and the total."
    ],
    "in_context": [
      {
        "lines": [
          6,
          16
        ],
        "form": "body",
        "role": "user"
      }
    ]
  },
...
  "answers_here": {
    "summary": "none of the 21 answers seen at this checkpoint reached the target",
...
  "runs_here": {
    "summary": "8 runs of this task with this harness, model and skill version",
...
  "task": {
    "prompt": "Use the tally-report skill to summarize data.csv into report.json.",
    "expected_output": "A report.json containing the line-item count and the amount total.",
    "expectations": [
      "report.json exists in the working directory",
      "it records 4 line items and a total of 61.50 under the keys row_count and amount_total"
    ],
...
```

The packet is built from what is recorded, and nothing is sampled for it:

- the step as the model saw it, and where the skill sits in it: lines 6 to 16, the body
  Claude Code loaded;
- the answers seen at this step that did, and did not, do what the target says: here the
  run's own and the twenty from Part 2 §5's resample, none of which did;
- how this task's runs with the same harness, model and skill version went there;
- the tokens the model was least sure of, once the answer has been scored (Part 3 §1);
- the task's own criteria;
- what was tried before.

Only this task's records are read.

#### With Hermes Agent

```bash
HREF=$(tuningfork refine list -q -n 1)
tuningfork refine show $HREF --packet
```

```
{
...
  "skill": {
    "name": "tally-report",
...
    "in_context": [
      {
        "lines": [
          1,
          16
        ],
        "form": "full",
        "role": "system"
      }
    ]
  },
...
```

The same evidence, gathered the same way, but for where the skill sits: the whole file, lines 1
to 16, frontmatter included, in the system prompt.

### 3. Ask for proposals

The model asked is the one experiments sample from — llama-server. It reasons before it
answers, and on a bigger skill than this one it can need more than the 8192 tokens it is
given by default; give it room once, and the setting stays in the recordings folder:

```bash
tuningfork connections proposer set --max-tokens 32768
tuningfork refine propose $REF --builtin
```

```
proposer  the proxy's own server — through the proxy for a job, gated and never recorded
  wire         openai-chat
  model        the server default
  max tokens   32768
  temperature  0.8
  key          not set
the built-in proposer — 3 proposals
  sends     the packet — 10 KB of recorded prompts, answers and the skill — to 127.0.0.1:8081
  model     the server default (openai-chat, max_tokens 32768)
  packet    18ba4acdafccd377
  asking…
  answered in 32s, 1473 tokens · kept in proposer/20261011T161133-ce6d4b
  proposal 90950deb: 1 line changed
    why  The agent consistently used 'count' and 'total' as keys, but the task's validation check
requires 'row_count' and 'amount_total'. Explicitly naming the keys in the final instruction should
correct this.
  proposal eb535d21: 1 line changed
    why  By introducing the specific key names during the calculation step, the agent is more likely
to carry those names over into the final JSON output.
  proposal 4726c7d9: 1 line changed
    why  The failure occurs at the Write step where the agent chooses the wrong keys. Adding the
requirement directly to the Write step is the most direct way to ensure the target keys are used.

hypotheses until screened: `tuningfork refine screen 832f0fce --proposals
90950deb,eb535d21,4726c7d9`
```

The proposer is sent the skill and the packet as data, with rules: at most three edits, each
`old` copied exactly from the skill, at most ten lines changed, the frontmatter left alone,
nothing copied from the task's own data, and a reason that cites the evidence. A proposal
that breaks a rule is not kept, and the reason is printed; here all three were kept. Each one
kept is saved with the model, the packet and the prompt it came from; the prompt and the
answer are in the refinement's folder, under `refinements/`.

`tuningfork connections proposer set` points the proposer at another endpoint — a cloud
API only with `--allow-remote`, its key read from `--key-stdin` and kept in a file only you
can read. To add your own wording instead, edit a copy of the skill and give it with a
reason: `tuningfork refine propose $REF --file my-SKILL.md --why "..."`.

#### With Hermes Agent

The proposer keeps the room §3's first command gave it; if you skipped Claude Code's steps, run
that command first:

```bash
tuningfork refine propose $HREF --builtin
```

```
the built-in proposer — 3 proposals
  sends     the packet — 9 KB of recorded prompts, answers and the skill — to 127.0.0.1:8081
  model     the server default (openai-chat, max_tokens 32768)
  packet    f7329eb4bf0b3ba3
  asking…
  answered in 26s, 1227 tokens · kept in proposer/20261011T171533-ec316b
  proposal 7b2038c3: 1 line changed
    why  The agent is currently using 'count' and 'total' as keys, but the task requires 'row_count'
and 'amount_total'. Specifying the keys in the write step ensures the agent uses the correct schema.
  proposal 72d4d965: 1 line changed
    why  The current instruction 'just the count and the total' is ambiguous and leads the agent to
use those words as keys. Explicitly naming the keys in this constraint clarifies the requirement.
  proposal 1b630c27: 1 line changed
    why  By associating the calculation step directly with the required keys, the agent is more
likely to carry those names forward into the final JSON output.

hypotheses until screened: `tuningfork refine screen 338504b1 --proposals
7b2038c3,72d4d965,1b630c27`
```

Three proposals again, each one line, in the same three places as Claude Code's: the step that
writes the file, the closing line, the step that works the values out.

### 4. Screen them

```bash
tuningfork refine screen $REF -n 20
```

```
run 22d9eec8 step 2 — 20 samples per arm from http://127.0.0.1:8081
  context   10 messages, after the result of Read
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~11.0 min
  target    a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
            declared when the refinement began
  arm       90950deb: 1 line changed
  arm       eb535d21: 1 line changed
  arm       4726c7d9: 1 line changed
  survives  with at least 7 of 20 — the step did it 0 times in 20 in resample 7f72a08c
            Holm over 3 proposals, α=0.05
  samples   80 in all: the original first, then each arm

  original   1/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)
9.5s
...
target (declared before sampling): a behaviour that calls Write, and mentions "row_count", and
mentions "amount_total"
          target                 p        Holm
original  0/20   0% [0%-16%]
90950deb  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
eb535d21  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
4726c7d9  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
p: Fisher's exact test of each proposal against the original arm; Holm: corrected over the 3 planned
before sampling (α=0.05).
A screen at one step is evidence, not proof: confirm a survivor with full runs.
screen e6d08857 · 11.3 min · `tuningfork refine show 832f0fce` shows every proposal with its
verdicts
```

A screen samples the step with the original wording first, fresh, then with each proposal
swapped in, and compares each with the original — Fisher's exact test, corrected (Holm)
for the number of proposals named before the first sample. Before it samples, it says what
an arm would need to survive — here using Part 2 §5's resample, which saw the target 0 times
in 20 — and it refuses a screen in which nothing could. Its samples are drawn as Part 2's
were, each prompt processed whole.

```bash
tuningfork refine show $REF
```

```
refinement 832f0fce — open
  checkpoint  run 22d9eec8 step 2 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/c1e9dd10/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

proposals (3) — hypotheses until a screen tests them
  90950deb  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet 18ba4acdafccd377
    why  The agent consistently used 'count' and 'total' as keys, but the task's validation check
requires 'row_count' and 'amount_total'. Explicitly naming the keys in the final instruction should
correct this.
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.

    - Keep the report small — just the count and the total.
    + Keep the report small, using the keys `row_count` and `amount_total`.
    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen e6d08857)
  eb535d21  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet 18ba4acdafccd377
    why  By introducing the specific key names during the calculation step, the agent is more likely
to carry those names over into the final JSON output.
    @@ line 11 @@

      1. Read the CSV file named in the request.
    - 2. Work out how many line items it contains, and the total of the amount column.
    + 2. Work out the `row_count` (number of line items) and the `amount_total` (sum of the amount column).
      3. Write the results to `report.json` in the working directory.

    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen e6d08857)
  4726c7d9  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet 18ba4acdafccd377
    why  The failure occurs at the Write step where the agent chooses the wrong keys. Adding the
requirement directly to the Write step is the most direct way to ensure the target keys are used.
    @@ line 12 @@
      1. Read the CSV file named in the request.
      2. Work out how many line items it contains, and the total of the amount column.
    - 3. Write the results to `report.json` in the working directory.
    + 3. Write the results to `report.json` in the working directory, using the keys `row_count` and `amount_total`.

      Keep the report small — just the count and the total.
    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen e6d08857)

screens (1)
  e6d08857  done  3 proposals, n=20 an arm — 4726c7d9, 90950deb, eb535d21 survived
```

All three survive, each twenty times in twenty. `90950deb` names the keys in the skill's last
line, in place of "just the count and the total" — the very words of the model's keys.
`eb535d21` names them in the step that works the values out, and `4726c7d9` in the step that
writes the file, leaving the last line as it was.

#### With Hermes Agent

```bash
tuningfork refine screen $HREF -n 20
```

```
run 5aa33251 step 4 — 20 samples per arm from http://127.0.0.1:8081
  context   10 messages, after the result of terminal
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 14,410 tokens, then answers: the recorded step
took 10.1s
  target    a behaviour that mentions "row_count", and mentions "amount_total"
            declared when the refinement began
  arm       7b2038c3: 1 line changed
  arm       72d4d965: 1 line changed
  arm       1b630c27: 1 line changed
  survives  with at least 7 of 20 — the step did it 0 times in 20 in resample 0f368795
            Holm over 3 proposals, α=0.05
  samples   80 in all: the original first, then each arm

  original   1/20  write_file(content='{"count":4,"total":61.5}', path='report.json… (+2 chars)
11.4s
...
target (declared before sampling): a behaviour that mentions "row_count", and mentions
"amount_total"
          target                 p        Holm
original  0/20   0% [0%-16%]
7b2038c3  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
72d4d965  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
1b630c27  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
p: Fisher's exact test of each proposal against the original arm; Holm: corrected over the 3 planned
before sampling (α=0.05).
A screen at one step is evidence, not proof: confirm a survivor with full runs.
screen 5b3d7089 · 13.6 min · `tuningfork refine show 338504b1` shows every proposal with its
verdicts
```

```bash
tuningfork refine show $HREF
```

```
refinement 338504b1 — open
  checkpoint  run 5aa33251 step 4 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/7afee5f1/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

proposals (3) — hypotheses until a screen tests them
  7b2038c3  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet f7329eb4bf0b3ba3
    why  The agent is currently using 'count' and 'total' as keys, but the task requires 'row_count'
and 'amount_total'. Specifying the keys in the write step ensures the agent uses the correct schema.
    @@ line 12 @@
      1. Read the CSV file named in the request.
      2. Work out how many line items it contains, and the total of the amount column.
    - 3. Write the results to `report.json` in the working directory.
    + 3. Write the results to `report.json` in the working directory as a JSON object with keys `row_count` and `amount_total`.

      Keep the report small — just the count and the total.
    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen 5b3d7089)
  72d4d965  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet f7329eb4bf0b3ba3
    why  The current instruction 'just the count and the total' is ambiguous and leads the agent to
use those words as keys. Explicitly naming the keys in this constraint clarifies the requirement.
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.

    - Keep the report small — just the count and the total.
    + Keep the report small — use only the keys `row_count` and `amount_total`.
    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen 5b3d7089)
  1b630c27  1 line changed · the built-in proposer, gemma-4-31b at 127.0.0.1:8081 · packet f7329eb4bf0b3ba3
    why  By associating the calculation step directly with the required keys, the agent is more
likely to carry those names forward into the final JSON output.
    @@ line 11 @@

      1. Read the CSV file named in the request.
    - 2. Work out how many line items it contains, and the total of the amount column.
    + 2. Work out how many line items it contains (`row_count`), and the total of the amount column (`amount_total`).
      3. Write the results to `report.json` in the working directory.

    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen 5b3d7089)

screens (1)
  5b3d7089  done  3 proposals, n=20 an arm — 1b630c27, 72d4d965, 7b2038c3 survived
```

All three survive, each twenty times in twenty. One of `7b2038c3`'s samples wrote the keys from
a script, which this target counts.

### 5. Validate a survivor with full runs

A screen at one step inherits the history the original wording wrote, and a wording can reach
the target without fixing the task. Validation checks a survivor with full runs, each arm
against a fresh arm of the original wording:

- **forward:** forks of the run from the step to the end — does reaching the target carry the
  task?
- **training:** full runs of the task, the edit in place from the first step — does it upset
  an earlier step?
- **held out:** full runs of each task the spec holds out. `examples/tally-report.evals.json`
  has one, eval 2: the same skill on `sales.csv`, marked `"held_out": true`. A batch never
  runs it unless asked by name, and validation records its runs in a temporary folder outside
  `~/tuningfork-guide`, deleted when they end: only whether each passed is kept, and no agent
  reading the recordings is shown even that.

Its runs go through the proxy from Part 2 §2, so leave that running. Take a survivor's id
from `refine show` and look at the plan first:

```bash
PROPOSAL=90950deb
tuningfork refine validate $REF $PROPOSAL --dry-run
```

```
validate proposal 90950deb of refinement 832f0fce — tally-report/1, at step 2
  screened  survived screen e6d08857: 20/20 against the original's 0/20, Holm p 4.4e-11
  forward   5 forks an arm from step 2 to the end, the original first
  training  5 full runs an arm of tally-report/1, the edit from the first step
  held out  5 full runs an arm of each of: 2 — only their verdicts are kept,
            recorded in a temporary folder under /tmp, outside the recordings folder,
            and deleted when they end
  runs      30 in all, through http://127.0.0.1:8787
  at a few runs an arm only a large difference shows: the not-worse checks are low-powered
```

Five forks or runs an arm, thirty in all — about fourteen minutes:

```bash
tuningfork refine validate $REF $PROPOSAL
```

```
...
forward — forks from step 2 to the end
  original   1/5 01d20618  fail  14.7s  Write(content='{"count":4,"total":61.5}',
file_path='report.… (+6 chars)
...
  edit       5/5 4b54e00b  pass  15.4s  Write(content='{"amount_total":61.5,"row_count":4}',
file_pa… (+17 chars)

training — full runs of tally-report/1
  original   1/5 1e84cf37  fail  29.4s
...
  edit       5/5 4afad9cc  pass  28.9s

held out 2 — verdicts only
  original   1/5  ended
...
  edit       5/5  ended

  1306a84e  done  proposal 90950deb — meets the gate
    forward   edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    training  edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    held out 2: edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    G0 met     survived a screen: it survived screen e6d08857
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): significantly better (Fisher p = 0.00794)
    G5 met     held-out task 2 is not worse (low-powered): significantly better (Fisher p = 0.00794)
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 5/5, without it 0/5

it meets the gate: `tuningfork refine approve 832f0fce 90950deb` shows what approving writes
```

Every check keeps its counts and the name of the rule that judged it. The edit passed every
fork from the step on, as the screen promised, every full run of the task from the first
step, and every run of the held-out task, against none of the original's. G4 and G5 ask only
that the edit be no worse, and say they are low-powered, since at five runs an arm only a
large drop would show; here each says more, since five in five against none in five is
significant even at this size, p = 0.008.

G8 asks more of a held-out task the original fails: the edit must pass it more often, as it
did, five to none. An edit that writes in the training task's answers would leave it failing —
and be no worse. Notes name what no check decides, such as an original that already passes from
the step: then the step may not decide the task. `tuningfork refine show $REF` shows every
validation again, held-out verdicts included; no MCP tool shows those.

#### With Hermes Agent

Validate the Hermes survivor in the closing line, `72d4d965`, as Claude Code's was validated:

```bash
HPROPOSAL=72d4d965
tuningfork refine validate $HREF $HPROPOSAL --dry-run
```

```
validate proposal 72d4d965 of refinement 338504b1 — tally-report/1, at step 4
  screened  survived screen 5b3d7089: 20/20 against the original's 0/20, Holm p 4.4e-11
  forward   5 forks an arm from step 4 to the end, the original first
  training  5 full runs an arm of tally-report/1, the edit from the first step
  held out  5 full runs an arm of each of: 2 — only their verdicts are kept,
            recorded in a temporary folder under /tmp, outside the recordings folder,
            and deleted when they end
  runs      30 in all, through http://127.0.0.1:8787
  at a few runs an arm only a large difference shows: the not-worse checks are low-powered
```

```bash
tuningfork refine validate $HREF $HPROPOSAL
```

```
...
forward — forks from step 4 to the end
  original   1/5 5226d2e3  fail  25.5s  write_file(content='{"count":4,"total":61.5}',
path='report.… (+6 chars)
...
  edit       5/5 d5d82cd3  pass  27.1s  write_file(content='{"amount_total":61.5,"row_count":4}',
pa… (+17 chars)

training — full runs of tally-report/1
  original   1/5 05fda6aa  fail  25.5s
...
  edit       5/5 39deaa94  pass  25.6s

held out 2 — verdicts only
  original   1/5  ended
...
  edit       5/5  ended

  3f7c997d  done  proposal 72d4d965 — meets the gate
    forward   edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    training  edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    held out 2: edit 4/5 (80%, 95% CI 38–96%); original 0/5 (0%, 95% CI 0–43%)
    G0 met     survived a screen: it survived screen 5b3d7089
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): significantly better (Fisher p = 0.00794)
    G5 met     held-out task 2 is not worse (low-powered): significantly better (Fisher p = 0.0476)
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 4/5, without it 0/5

it meets the gate: `tuningfork refine approve 338504b1 72d4d965` shows what approving writes
```

The edit passed every fork from the step and every full run of the task, against none of the
original's, and four of five runs of the held-out task, against none. Four in five against none
is significant even at this size, p = 0.048, and G8 holds: the held-out task the original fails
improves with the edit. Only verdicts leave the private proxy, so nothing here says why the
fifth failed; `--keep-private <dir>` keeps a validation's held-out recordings for you to read,
and any agent that can read files could read them too. Hermes's runs take longer, so this
validation took eighteen minutes.

### 6. Approve it

Approving is the one thing in tuningfork that writes a skill's own file, and only you do it.
It shows the file it will write — under WSL, also as Windows names it — what it keeps, where
the old file and the changelog go, the gate and the edit, and writes nothing until you type
the code it shows:

```bash
tuningfork refine approve $REF $PROPOSAL
```

```
approve proposal 90950deb of refinement 832f0fce, on validation 1306a84e
  writes     /home/you/tuningfork/examples/tally-report/SKILL.md
             \\wsl.localhost\Ubuntu\home\you\tuningfork\examples\tally-report\SKILL.md
  keeps      LF line endings
  file before → /home/you/tuningfork/examples/tally-report.history/
  changelog   → /home/you/tuningfork/examples/tally-report.changelog.md
  the gate   every check met:
    G0 met     survived a screen: it survived screen e6d08857
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): significantly better (Fisher p = 0.00794)
    G5 met     held-out task 2 is not worse (low-powered): significantly better (Fisher p = 0.00794)
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 5/5, without it 0/5
  the edit
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.

    - Keep the report small — just the count and the total.
    + Keep the report small, using the keys `row_count` and `amount_total`.

  code  c632c165
Type the code to write it (anything else writes nothing): c632c165
written: /home/you/tuningfork/examples/tally-report/SKILL.md
  the file before  /home/you/tuningfork/examples/tally-report.history/SKILL.20261011T163907Z-c3dd28e9.md
  changelog        /home/you/tuningfork/examples/tally-report.changelog.md
  to undo, copy the file before back over it
```

`examples/tally-report/SKILL.md` now has the edit. The old text is in
`examples/tally-report.history/`, and `examples/tally-report.changelog.md` says what changed,
why, on what evidence and who approved it — beside the skill folder, never in it, since every
run is given a copy of that folder. The refinement keeps the approval with its evidence, so
deleting its runs later leaves it standing. git sees the change:

```bash
git status --short examples/
```

```
 M examples/tally-report/SKILL.md
?? examples/tally-report.changelog.md
?? examples/tally-report.history/
```

Your code will differ: it names the proposal, the file and the file's bytes as they are now.
Run Part 2 §3's batch again to see the new wording's pass rate beside the old one in
`tuningfork tasks`. To put the example back as it was:

```bash
git checkout -- examples/tally-report/SKILL.md
rm -r examples/tally-report.history examples/tally-report.changelog.md
```

#### With Hermes Agent

Approving does not depend on the harness:

```bash
tuningfork refine approve $HREF $HPROPOSAL
```

```
approve proposal 72d4d965 of refinement 338504b1, on validation 3f7c997d
  writes     /home/you/tuningfork/examples/tally-report/SKILL.md
             \\wsl.localhost\Ubuntu\home\you\tuningfork\examples\tally-report\SKILL.md
  keeps      LF line endings
  file before → /home/you/tuningfork/examples/tally-report.history/
  changelog   → /home/you/tuningfork/examples/tally-report.changelog.md
  the gate   every check met:
    G0 met     survived a screen: it survived screen 5b3d7089
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): significantly better (Fisher p = 0.00794)
    G5 met     held-out task 2 is not worse (low-powered): significantly better (Fisher p = 0.0476)
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 4/5, without it 0/5
  the edit
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.

    - Keep the report small — just the count and the total.
    + Keep the report small — use only the keys `row_count` and `amount_total`.

  code  637fbf07
Type the code to write it (anything else writes nothing): 637fbf07
written: /home/you/tuningfork/examples/tally-report/SKILL.md
  the file before  /home/you/tuningfork/examples/tally-report.history/SKILL.20261011T184422Z-c3dd28e9.md
  changelog        /home/you/tuningfork/examples/tally-report.changelog.md
  to undo, copy the file before back over it
```

```bash
git status --short examples/
```

```
 M examples/tally-report/SKILL.md
?? examples/tally-report.changelog.md
?? examples/tally-report.history/
```

The same file, written the same way, and its changelog names the Hermes refinement and its
validation. Put the example back as above:

```bash
git checkout -- examples/tally-report/SKILL.md
rm -r examples/tally-report.history examples/tally-report.changelog.md
```

### 7. The same in the browser, or from an agent

The **Refine** page lists refinements; a step's **Refine the skill here…** starts one. A
refinement's page shows the evidence and each proposal as a line diff, with who proposed it,
why and its verdicts; asks the proposer, or takes your own wording; screens the proposals you
tick; validates a survivor; and shows each validation's parts and gate, with an **Approve…**
that shows the file, the edit and the code to type. An agent gets the loop up to the screen
over MCP with `tuningfork mcp --tools refine`: it starts a refinement, reads the evidence,
proposes its own edits and screens them. It is never offered the built-in proposer, a
validation or an approval, and never shown a held-out task's verdicts.

## Clean up

Stop the proxy and `llama-server` with Ctrl-C, then:

```bash
rm -rf ~/tuningfork-guide ~/tuningfork-demo ~/tuningfork-live
rm -f held.json answer.json
```

If you approved an edit in Part 4, put the example back with the two commands at the end of
Part 4 §6.

---

## Will I get the same numbers?

Not digit for digit. The model samples at temperature 1.0 from a fresh random seed each time —
a harness sends none, and tuningfork draws new ones for every experiment — so counts wobble
between tries and between machines, and so do the runs `--called` picks: here every Claude Code
run wrote its report with `Write`, seven at step 2 and one, after an `ls`, at step 3, and four
of the eight Hermes runs with `write_file`, three at step 1 and one at step 4. What reproduces
is the shape: the original wording fails every time, by writing `count` and `total`; resampling
the write never gives `row_count`; the A/B separates the wordings by a wide, significant margin
at the write, and so does following it to the end of the run; and full runs with the variant
pass nearly every time. Our batches of the original wording on this model have all come out at
zero, every report with `count` and `total`: 0/8 twice with Claude Code and 0/8 twice with
Hermes. The variant's full runs passed 8/8 with Claude Code and 8/8 with Hermes. The step-0
split varies the most, and matters least.

Without bubblewrap, each run works in folders of its own, whose names are in its prompt, so
on your machine every run is asked something slightly different — and to a model that can
matter. On the 26B-A4B this page used before, forks of one run that differed only in a
folder's number were resampled at the write with the variant swapped in, every prompt
processed whole: three wrote the file 20 times in 20, and one wrote it 5 times, answering
the other 15 in text.

Part 1's numbers are exact, because its "model" is a script.

## Notes

**This skill was hard to make fail on purpose.** An earlier version left a *judgement call*
ambiguous instead of a key name — the data had a refund, and the skill said only "ignore
anything that isn't a genuine sale". The expectation was a roughly even split. The model
resolved it the same way six times out of six. A skill's pass rate is a property of the model
as much as of the wording, and a genuine coin-flip is harder to engineer than it sounds.

**Two corrections.** This page once reported the shipped skill at 1/6. The one "pass" came
from a run recorded under a different `SKILL.md` — the file was edited while its batch was
still running — and the skill hashes recorded with every run exposed it; a batch now freezes
the skill before its first run. Separately, `SKILL.v2.md` itself sat inside the skill's folder
from the start, so every run was given a copy of it beside the skill under test. No recorded
call ever mentioned it, so no result here changes, but it has moved out, and `batch` now
warns about any other `SKILL*.md` in a skill's folder.

**Why Gemma 4 31B.** This page first ran on Gemma 4 31B, without `--swa-full`. With the flag
the 31B does not fit a 32 GB card at this context, so the page moved to the 26B-A4B, a
mixture-of-experts model several times faster. It came back to the 31B with one slot and
`--ctx-checkpoints 0`, which keep a request's numbers the same without the flag's memory.
The story kept its shape on both — the original wording fails, the variant fixes it — but
the 26B failed a second way, which the 31B did not in our runs: it took loading the skill
for doing the job, then looked for a report nothing wrote, or said it wrote one. With the
keys fixed, that failure was left, and the 26B's full runs with the variant passed about
half the time.

**What it costs.** On one RTX 5090 with Claude Code: about half a minute per full run of this
task; 2.8 minutes to resample the write twenty times — eight and a half seconds a sample, since
the model reasons there and every prompt is processed whole; 5.5 minutes for an A/B of twenty
per arm; 10.5 minutes to follow that A/B to the end, forty forks at about 15 seconds each. In
Part 4, the proposer took 32 seconds, screening its three proposals, 80 samples, 11.3 minutes,
and validating one, thirty runs, 14 minutes. With Hermes Agent: about half a minute a run as
well, though a few took two minutes; ten seconds a sample at its write, 3.3 minutes for twenty;
6.8 minutes for the A/B; 16.2 to follow it to the end, about 25 seconds a fork; and in Part 4,
26 seconds for the proposer, 13.6 minutes to screen and 18 to validate. `--no-cache-prompt`
adds a whole prompt's processing to every step, about four seconds for this task's 14,000
tokens. Telling a 60% skill from an 80% one needs about 82 samples per arm — some twelve
minutes an arm at this checkpoint. The commands can run several samples at once
(`--concurrency`), but a server with one slot computes one request at a time, and on one GPU
more slots saved no time when this page tried them on the 31B: four streams each generated at a
fraction of the speed, and forks took 18 seconds each four at a time against 14 one at a time.
Measure it on your own server before relying on it.
