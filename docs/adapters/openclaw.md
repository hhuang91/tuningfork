# OpenClaw adapter

Verified against **OpenClaw 2026.10.7** (157d8c2) on Node 26.12.1, talking to `llama-server`
(Gemma 4 31B, b11108) through the tuningfork proxy, every launch in the sandbox. Everything
below was read out of recordings, except where a line says it comes from OpenClaw's source at
that tag.

Implementation: `tuningfork/harness/openclaw.py`. Template for other harnesses:
[`TEMPLATE.md`](TEMPLATE.md).

## Launching

| Need | How |
|---|---|
| One non-interactive prompt | `openclaw agent exec -- "<prompt>"`: one turn, headless, with no Gateway |
| Machine-readable output | `--json` |
| Its configuration | `--config ~/.openclaw/openclaw.json`, the launch's own (below) |
| Where it works | `--cwd <workspace>` |
| Bound the run | nothing of OpenClaw's: `--timeout 0`, and tuningfork ends a run at its task's `timeout` |
| Preload a skill | no flag: the skills in `~/.openclaw/skills` are listed to the model, which reads one |

`agent exec` takes no turn limit, so a task's `max_turns` does not reach OpenClaw: a run ends
where the model stops, or at the task's timeout. OpenClaw's own default timeout, 600 s, would
end a run by itself and have it judged; with `--timeout 0`, tuningfork ends it as it ends any
harness, as timed out — left out of a batch's pass rate, a failure in a test run to the end.
The prompt goes after `--`, so one that starts with a dash is not read as a flag.

Each `agent exec` is a new session, with a random id, in a temporary state folder of its own
(`/tmp/openclaw-agent-exec-*`, the sandbox's own `/tmp`), deleted when it ends.

## Installing it

OpenClaw runs on Node 24.22.1 or later in 24, or 26.12.1 or later. Install it with that Node's
npm, as its docs say: `npm install -g openclaw@2026.10.7 --allow-scripts=openclaw` with npm
11.16 or later, the same without `--allow-scripts` on an older npm. Its own install scripts
touch only its own folder.

On a Node it does not run on, OpenClaw looks for another `node` and runs itself again with it
(source): on `PATH`, in nvm's, fnm's and Volta's folders in its home — a launch's own home,
which has none — and Homebrew's. Finding none, `agent exec` exits before its first call, while
`openclaw --version` still answers, with a warning. tuningfork reads that warning: the
Connections page says OpenClaw cannot run here, and why, as `tuningfork connections show` does,
and a batch or a fork is refused before it starts, in the same words. A launch never installs a Node of
OpenClaw's own: it offers to only with a terminal, and a launch has none. Ubuntu 26.04's own
Node is 22.

## Isolation

A launch's `HOME` is its own folder, and OpenClaw keeps its state in `~/.openclaw` there: its
config, the skills, and what it writes itself — `state/openclaw.sqlite`, links to its plugins'
own skills in `plugin-skills/` — and `~/.cache/openclaw-<uid>`. So no real installation's
settings, credentials, sessions or memory reach a run, and nothing of a run is left for the
next. The child's environment is built from scratch: `PATH`, `LANG`, `LC_ALL`, `TERM`,
`TMPDIR`, `USER` and the run's own `HOME`. Every `OPENCLAW_*` variable in the parent shell is
dropped, so none can move the state or the config to a real installation.

The config tuningfork writes, `~/.openclaw/openclaw.json` (JSON5, so a comment line heads it):

```json5
// Written by tuningfork: a disposable, isolated OpenClaw home for one launch.
{
  "models": {
    "providers": {
      "local": {
        "baseUrl": "http://127.0.0.1:8787/_launch/<token>/v1",
        "apiKey": "custom-local",
        "api": "openai-completions",
        "timeoutSeconds": 86400,
        "models": [{
          "id": "<model>", "name": "<model>",
          "compat": {"supportsTools": true, "toolSchemaProfile": "llamacpp"},
          "contextWindow": 131072
        }]
      }
    }
  },
  "agents": {"defaults": {
    "model": {"primary": "local/<model>"},
    "compaction": {"memoryFlush": {"enabled": false}}
  }}
}
```

- **`api: "openai-completions"`**: OpenClaw's OpenAI-compatible transport, chat completions.
  OpenClaw ships no llama.cpp provider of its own (its llama.cpp plugin is a separate
  install), so the provider is a custom one; `custom-local` is OpenClaw's own non-secret
  marker for a provider on this machine.
- **`toolSchemaProfile: "llamacpp"`**: llama-server compiles a tool's schema into a grammar,
  and OpenClaw's profile drops the long `pattern` and `maxLength` values it cannot compile, as
  OpenClaw's docs say a custom provider pointed at llama-server must ask.
- **`contextWindow`**: OpenClaw never asks the server, and of a model its config does not size
  it assumes 200,000 tokens. It compacts a conversation by this, and bounds each answer below
  it. A batch writes the server's own (`n_ctx`), as OpenClaw's own llama.cpp provider does for
  a server it finds running (from its `/props`), unless the harness option `context_window`
  gives another. A fork writes its source's, and every run of a validation the checkpoint
  run's: none, where that run recorded none. OpenClaw counts tokens its own way, though: on a
  log full of digits it counted about a third fewer than the server (below), so a conversation
  can outgrow the server's window before OpenClaw compacts it. The server then refuses the
  call, and OpenClaw compacts and asks again (source). tuningfork, finding a failed call, leaves
  such a run out of a batch's pass rate, and a fork out of its arm's count to the end, since a
  failed call is no timeout: an edit that makes runs longer is then measured only on the runs
  that fit the window.
- **The output limit** is left to OpenClaw: 8,192 tokens, sent as `max_completion_tokens`,
  which llama-server honours (measured: `finish_reason: length` at the limit, as for
  `max_tokens`).
- **`timeoutSeconds`**: see [holding a call](#holding-a-call).
- **`memoryFlush`**: off, the one setting a launch makes otherwise than OpenClaw's own; see
  [options](#options-a-launch-takes).

A connection watcher following the whole process tree of a batch — tuningfork's command,
bubblewrap, OpenClaw's processes and the check — through `/proc`, every 0.05 s, saw **no
connection to anything but the proxy**. OpenClaw's update check and its telemetry are the
Gateway's alone (source), and its download of `fd` and `rg` from GitHub is its terminal UI's
alone (source), so `OPENCLAW_OFFLINE` changes nothing in a launch.

OpenClaw reads no instructions from the folders above its working directory. Verified without
the sandbox: an AGENTS.md, SOUL.md, IDENTITY.md, USER.md, MEMORY.md and CLAUDE.md, each with a
marker, and a skill in `.agents/skills`, all planted in the git repository the runs' folder was
in, reached no request. It does walk up to that repository, and names its root in every prompt
(`repo=<root>`, in the `Runtime:` line below), so without the sandbox `tuningfork batch` warns
when its folder is inside one; in the sandbox a launch works in `/home/user/project`, under
nothing but its home. OpenClaw also loads a `.env` from the working directory into its own
environment, leaving out providers' keys and the other variables it deems unsafe (source): a
task that ships one gives the rest to the run.

## Finding it, and what it needs of the server

tuningfork runs the `openclaw` it finds on `PATH`, unless the recordings folder's `config.json`
names another executable (`tuningfork connections harness openclaw --executable
/path/to/openclaw`, or the Connections page). Its version is read with `openclaw --version`
in a throwaway home, as the other harnesses' are; it writes nothing there.

The adapter declares what it needs of the inference server (`requirements`), and a batch or a
fork refuses to launch it against a server that lacks it — before a launch is recorded, not
after a failed run:

| Requirement | Why |
|---|---|
| Streaming tool calls | OpenClaw streams every call, and acts on the tool calls in the stream |
| `n_ctx` of at least 16,384 (32,768 with Tool Search off) | OpenClaw's first request alone is about 9,000 tokens of Gemma 4 (23,000 with Tool Search off), before its answer |

OpenClaw itself refuses only a window under 4,000 tokens, before its first call: "Model context
window too small (3000 tokens; source=modelsConfig). Minimum is 4000." So the option
`context_window` takes no less. It keeps back a quarter of the window, at most 20,000 tokens,
for compacting the conversation (source).

## Payload anatomy

From recorded runs of the examples' tally-report task.

### What OpenClaw sends

```json
{"model": "...", "messages": [...], "tools": [...], "stream": true,
 "stream_options": {"include_usage": true}, "tool_choice": "auto",
 "max_completion_tokens": 8192}
```

No `temperature`, no `top_p`, no `seed`: the server's defaults are the sampling settings, as
for Hermes. An earlier turn's reasoning is never sent back: each assistant turn goes as its
calls, with `content: null`. A call's arguments go back as the server streamed them, and its id
as the server gave it.

### Message layout

Two messages on the first step: `system`, then `user`; with the tools, about 9,000 tokens. The
system prompt (26,886 characters) opens `You are a personal assistant running inside
OpenClaw.`, in two parts marked for caching. `<!-- openclaw:attempt:STABLE -->` holds the
tooling, the deferred tools' directory, tool-call style, execution bias, the Skills section,
the Skill Workshop section, memory recall, the workspace, OpenClaw's documentation, and the
Project Context (AGENTS.md, SOUL.md, IDENTITY.md, each `[MISSING]` in a task's empty
workspace). `<!-- openclaw:attempt:DYNAMIC -->` holds the date and time zone, output
directives, messaging, and the model's identity. The system prompt is the same at every step
of a run.

The first user message is the person's prompt, with the time before it and a `Runtime:` line
after it:

```
[Sun 2026-10-25 02:32 EDT] Use the tally-report skill to summarize data.csv into report.json.

Runtime: agent=main | session=agent:main:explicit:<session id> | host=<hostname> | os=Linux … |
node=v26.12.1 | active_node=unknown | active_node_identity=unknown | model=local/<model> | default_model=local/<model>
```

The system prompt asks the model to begin a reply with `[[reply_to_current]]`, which OpenClaw
takes out before a person sees it, and the model mostly does: of the 16 final replies the batch
runs here gave, 14 began with it. A reply check reads the reply as the model wrote it, so allow
for it.

### Tools

With **Tool Search** on, as OpenClaw ships it, 11 tools are in the request — `read`, `write`,
`edit`, `apply_patch`, `ls`, `exec`, `process`, `sessions_yield` and `tool_search`,
`tool_describe`, `tool_call` — and the system prompt lists OpenClaw's 33 others, a line each,
reached only through `tool_call` (`{"id": "<tool>", "args": {…}}`): `skills_read`,
`sessions_spawn`, `skill_workshop`, `memory_search`, `gateway`, `plugins`, `web_fetch`… An MCP
server's tools are not named there at all: `tool_search` finds them (source). With it off, all
41 schemas are in the request (85,000 characters), and the first request is about 23,000
tokens instead of 9,000.

Two of the deferred tools reach past the run. `gateway` offers only `update.run`, an update of
OpenClaw itself, which the sandbox's read-only machine stops; its line tells the model to use
it only on its owner's explicit request. `plugins` lists and installs plugins from OpenClaw's
catalog and ClawHub, into the launch's home, over the network, and nothing in the prompt holds
it back.

### Skills

The system prompt's Skills section lists each skill — `<available_skills>`, a `<name>`,
`<description>` and `<location>` each — and asks the model to read the matching one before it
acts. A skill tuningfork installs is in `~/.openclaw/skills/<name>/`, listed at
`~/.openclaw/skills/<name>/SKILL.md`, beside OpenClaw's own: the bundled skills whose programs
the machine has, its custodian skills and its plugins' — 21 in all here. The bundled ones are
listed at their place in OpenClaw's installation.

How the model loads a skill depends on Tool Search:

- **On:** it reads the location with `read`
  (`{"path":"~/.openclaw/skills/tally-report/SKILL.md"}`), as the Skills section's last lines
  say to — though its first says `skills_read`, which is only a line in the directory. Asked
  to, it called `skills_read` through `tool_call` instead.
- **Off:** it calls `skills_read` (`{"name":"tally-report"}`).

Each way the result is the SKILL.md as a text tool result, **frontmatter included**: from
`skills_read`, the file as it is; through `tool_call`, the same inside the JSON it returns
(`{"tool": {…}, "result": {"content": [{"type": "text", "text": …}]}}`); from `read`, as `read`
gives any file — a byte-order mark dropped, CRLF made LF — but whole (source). A skill OpenClaw
has loaded, listed in the system prompt or not, is never served in part: any paging the call
asks for is dropped, a skill too long for the call's budget is refused ("Skill instructions
cannot be partially served…"), and one read again in the same context is not sent again
("Skill instructions were already served whole earlier in the current model context…"). Any
other SKILL.md is read as any file is, a part at a time. The adapter takes the skill's text from
each result (`skill_spans`), and none from those notices or a call that failed; it names a
`read` by the skill's folder, as `skill_load` does. A wording A/B finds the skill there:
verified at a checkpoint (the examples' `SKILL.v2.md`, which names the report's keys:
`row_count` 0/5 → 5/5, p = 0.008) and to the end (the check 0/3 → 3/3).

### User messages it writes itself

| It opens | Named |
|---|---|
| `The conversation history before this point was compacted into the following summary:` | compaction summary |
| `[<time>] Continue the current task from the existing transcript, preserving completed work.` | continue after a compaction |

A person's turn, the time and the `Runtime:` line beside it, is neither.

### Compaction

Forced with a 40,000-token window (`-o context_window=40000`) on a task that reads a 206 KB log
page by page. Five times in 25 minutes, as the conversation outgrew the window — OpenClaw counts
tokens its own way, and its last call before the first compaction held 56,684 by the server's
count — OpenClaw summarized it in two calls of its own: a 1,077-character system prompt (`You
are a context summarization assistant.`), the conversation as one user message, no tools,
`max_completion_tokens` 6,553 and then 4,096. Then it went on from a new conversation: the
system prompt, the summary as a user message, the last call and its result, and its own turn to
go on (above). The adapter names the summary calls a side call (`side_call`), so neither is
ever a run's final reply. Unlike Hermes's review, a summary is fed back: it stands for the
conversation after it (`fed_back_side_calls`). So a summary call that failed, or was edited at a
breakpoint, leaves the run out, as a task call does, and a chain never reports one as a review.

The conversation after a compaction does not extend the run's main thread, so tuningfork files
it as a thread of its own: a run that compacts has its later steps there, and its final reply is
the main thread's last answer. With the server's whole window, that is only a long run's.

### The memory flush and the Skill Workshop's review

Neither runs in a launch (source, and no such call or file in any recording):

- **The memory flush** — a silent turn before a compaction, writing `memory/YYYY-MM-DD.md` —
  belongs to the preparation of a later turn in a kept session, and `agent exec` starts a new
  session every time. The five compactions above ran without one, and wrote no `memory/`.
- **The Skill Workshop's review** — after ten model calls, or once a learned skill is used — is
  never scheduled for a run from the command line, and its 30 s idle timer does not keep the
  process alive. What a launch has of the Skill Workshop is the system prompt's section and
  the `skill_workshop` tool, which works in a launch (asked to list, it answered "No learned
  skills yet"): it saves and patches *learned* skills, in a folder of their own under the agent's
  state, never the skill under test, which outranks a learned skill of its name; they go with
  the launch.

## Subagents

**A subagent cannot run under `agent exec`.** `sessions_spawn` goes through the Gateway's
admission of a run, which needs a reply runtime only a running Gateway publishes (source);
without one, every spawn returns `"status": "error", "error": "published reply runtime missing
for main"`. Seen on a task that asked for one: the model tried seven times, then did the work
itself. `subagent_tools` names `sessions_spawn` for a version that runs one; under Tool Search
it would arrive as a `tool_call`.

## Holding a call

A breakpoint holds a call while a person decides; a held streaming call answers at once and
sends an SSE comment every 10 s. OpenClaw gives a model call to a local endpoint 300 s to its
first streamed event, and comments are not events: without a provider timeout, a held call
would be given up after 300 s (source). The provider's `timeoutSeconds` sets that wait and the
wait between events alike, so the adapter writes 86,400 s, and the task's own timeout is what
ends a run. Measured:

- **The adapter's config:** a call held 321.9 s before the model saw step 1 went on when it was
  continued, and the run finished normally.
- **`timeoutSeconds: 60`:** a held call was abandoned at 60.0 s, keep-alive comments
  notwithstanding, and OpenClaw sent it again (`transient same-model retry 1/8 …
  reason=timeout`).

## Launches: every call, attributed as it arrives

A run tuningfork starts is a *launch*, and OpenClaw is given
`http://127.0.0.1:8787/_launch/<token>/v1` as its provider's `baseUrl`. Verified on 2026.10.7:
every call keeps the prefix, the task's calls and a compaction's alike. A forwarder logging every
request in front of the proxy saw nothing else from a launch: no model list, no `/props`, only
`POST …/v1/chat/completions`.

## Replaying a recorded run (forks)

A fork answers a relaunched OpenClaw from the recording until the step it goes live at. What
differs between two launches of one task, besides per-run paths, is declared in the adapter's
`volatile` list and masked when requests are compared:

- the time before the prompt (`[Sun 2026-10-25 02:32 EDT]`, to the minute), and before
  OpenClaw's turn after a compaction;
- the date in the system prompt (`Current date: …`);
- a session key, of any kind: the `Runtime:` line's (`agent:main:explicit:<uuid>`), and those a
  tool's result names — a failed `sessions_spawn` names the child session and the run it made
  (`agent:main:subagent:<uuid>`, `"runId": "<uuid>"`), and `subagents` the session it was
  called from;
- the process a command left running in the background is named by, a slug and a pid
  (`Command still running (session <slug>, pid <pid>)`, source).

With those masked, two launches' first requests are identical, and every fork's replayed calls
matched the recording — a fork of the run that tried to spawn a subagent too, replayed past its
spawns; before the session keys and run ids were masked, its fork departed at the step after the
first. Re-verify on an upgrade: the system prompt is OpenClaw's own words, and OpenClaw releases
about every two days.

## Options a launch takes

Beside the seam's own arguments, a batch or a chain can give OpenClaw options
(`--harness-option k=v` on the CLI, `harness_options` in a job's parameters, the "Harness
options" field of the Run a task form), read strictly by the adapter (`OpenClawOptions`) and
recorded in each launch's settings under `harness`; a fork is set up as its source was, and
every run of a validation as the checkpoint's run was, but for its MCP servers, which the
settings do not keep whole (below): those runs go without them, and the plan says so. A launch
is OpenClaw as shipped but for the memory flush.

| Option | Default | What it does |
|---|---|---|
| `tool_search` | on | On, as OpenClaw ships it: 11 tools and a directory of the rest (above). Off (`tools.toolSearch: false`), every tool's schema is in every call, and the model loads a skill with `skills_read`. A run that drives tuningfork's MCP tools wants it off: with it on, an MCP server's tools are not named in the prompt at all, and tuningfork's tools carry what an agent must know in their descriptions. |
| `skill_workshop` | on | On, as shipped: the Skill Workshop's section of the system prompt and the `skill_workshop` tool (above). Off, the tool is denied (`tools.deny: ["skill_workshop"]`), and OpenClaw leaves the section out with it: verified, the only difference in the request, 1,051 characters. OpenClaw's own `skills.workshop.autonomous.mode: "off"` stops only the background review, and changed nothing sent. |
| `memory_flush` | off | Off, every launch's config says `agents.defaults.compaction.memoryFlush.enabled: false`; on, the launch keeps OpenClaw's own default, on. The flush never runs in a one-shot launch either way (above): the setting is there for a later turn of a kept session. |
| `mcp_servers` | none | MCP servers for the launch's config, as OpenClaw takes them under `mcp.servers` (`command`, `args`, `env`, or `url` with its `transport`; name the transport of a URL server, which OpenClaw takes as SSE otherwise): how a launched OpenClaw gets tuningfork's own tools. Verified with the proxy's own route (`{"url": "http://127.0.0.1:8787/_tuningfork/mcp", "transport": "streamable-http", "headers": {"Authorization": "Bearer <the MCP token>"}}`) and Tool Search off: the model was given each tool as `tuningfork__<tool>`, and called `tuningfork__list_tasks`. A server is written whole into the launch's config, in its home, where the run can read it; the settings record its command or URL, never its environment or headers. |
| `context_window` | the server's | The context window OpenClaw is told the model has, at least 4,000 tokens: the server's `n_ctx` unless given; in a fork, its source's; and in every run of a validation, the checkpoint run's. |

## Still to verify

- A conversation carried on after a compaction, as part of the run's main thread
- Chains: one more turn of a launch's session (`continue_launch`), which needs a kept state
  folder (`--state-dir`) and its session id
- Importing a session OpenClaw kept, from its SQLite store or its trajectory export
- A scenario made from an OpenClaw run: its prompt is the run's first user message whole, the
  time OpenClaw wrote before it and its `Runtime:` line included (the session key, the recording
  machine's host), which the scenario's runs would be sent again — seen in the code that makes
  one, not tried
