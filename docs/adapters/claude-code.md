# Claude Code adapter

Verified against **Claude Code 2.1.281** talking to `llama-server` (Gemma 4 31B, b11108)
through the tuningfork proxy, with **no router in between**: llama.cpp serves Anthropic's
Messages API natively at `/v1/messages`. Everything below was read out of recordings.

Implementation: `tuningfork/harness/claude_code.py`, and the wire format in
`tuningfork/wire/anthropic.py`. Template for other harnesses: [`TEMPLATE.md`](TEMPLATE.md).

## Launching

| Need | How |
|---|---|
| One non-interactive prompt | `claude -p "<prompt>"` |
| Machine-readable output | `--output-format json` |
| Bound the run | `--max-turns N` |
| The model | `--model <id>`, plus every model tier pointed at the same model (below) |
| No one to approve tools | `--dangerously-skip-permissions` — only acceptable in a disposable workspace |

There is no flag to preload a skill. Skills installed in the config directory are offered
to the model, which loads one through the `Skill` tool when the task calls for it.

## Isolation

`CLAUDE_CONFIG_DIR` relocates all of Claude Code's state — `.claude.json`, settings,
sessions, skills — and nothing was written outside it in testing. The adapter also gives
each run its own `HOME`, and builds the child's environment from scratch, so a real
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` or `CLAUDE_*` in the parent shell cannot leak
in. What it sets, also written to `<home>/config/settings.json`:

| Variable | Value |
|---|---|
| `ANTHROPIC_BASE_URL` | the proxy's root — Claude Code appends `/v1/messages` itself |
| `ANTHROPIC_API_KEY` | a placeholder; it only ever reaches the proxy |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` | the model under test |
| `DISABLE_TELEMETRY`, `DISABLE_ERROR_REPORTING`, `DISABLE_AUTOUPDATER`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` |

Claude Code also loads every `CLAUDE.md` (and `CLAUDE.local.md`, `.claude/CLAUDE.md`,
`.claude/rules/`) in the directories **above** its working directory, up to the root. A
batch run's workspace sits inside the recordings folder, usually inside your home directory,
so a `~/CLAUDE.md` would reach every run. Verified: a marker planted in the recordings
folder appeared in the recorded request. The adapter therefore passes
`--settings '{"claudeMdExcludes": [...]}'`, listing those files in every directory above the
workspace (`claudeMdExcludes` matches absolute paths). The workspace itself is left alone,
so a task can still ship its own `CLAUDE.md` as part of what it tests.

Inside a git repository, Claude Code also offers the skills, commands and agents in `.claude`
folders between the working directory and the repository's root. That cannot be excluded
the same way, so `tuningfork batch` warns when its batch folder is inside such a
repository; keep `--workdir` (by default inside `TUNINGFORK_HOME`) outside one.

A connection watcher sampling every 0.3 s during a full run saw **no connection from
Claude Code to anything but the proxy**. Claude Code does probe the base URL with
`HEAD /api/hello` at start-up; the proxy relays it, and a 404 is fine. It logs
`unrecognized_model` for a local model name, harmlessly.

## Finding it, and what it needs of the server

tuningfork runs the `claude` it finds on `PATH`, unless the recordings folder's
`config.json` names another executable (`tuningfork connections harness claude-code
--executable /path/to/claude`, or the Connections page). It never reads or writes the
harness's own configuration to do so. Its version is read with `claude --version`, run in a
throwaway home — `HOME`, `HERMES_HOME`, `CLAUDE_CONFIG_DIR` and the XDG folders all
pointed there, no input, 15 s at most — so even detecting it leaves nothing behind in a
real installation. Verified: detecting both harnesses changed no file under the real
`~/.hermes`, `~/.claude` or `~/.config`.

The adapter declares what it needs of the inference server (`requirements`), and a batch
or a fork refuses to launch it against a server that lacks it — before a launch is
recorded, not after a failed run:

| Requirement | Why |
|---|---|
| The Anthropic Messages API, `/v1/messages` | Claude Code speaks nothing else; llama-server serves it natively |

What a server offers is measured, not assumed: `tuningfork connections check` (or *Test
the server* on the Connections page) makes one streaming tool call and one `/v1/messages`
call through the proxy, and keeps the answers with the server's identity. Until a server
is checked, a launch goes ahead with a note that the requirement was not measured.

## Payload anatomy

Every model call is `POST /v1/messages?beta=true`, streamed, with `anthropic-version:
2023-06-01` and a long `anthropic-beta` list. The request carries:

- a **`system` field** of three text blocks: a billing header naming the Claude Code
  version, a one-line identity, and the main prompt (about 5.8K characters — far smaller
  than Hermes's);
- **20 tools**, including `Skill`, `Read`, `Write`, `Bash`, `Edit`, `Agent`;
- `thinking: {type: adaptive}`, `max_tokens: 32000`, `output_config: {effort: high}`, and a
  `metadata.user_id` holding a device and session id — different in every run;
- **`role: "system"` messages mid-conversation** (a beta): the working-directory
  environment after the first user message, and a token budget
  (`<total_tokens>N tokens left</total_tokens>`) appended after every tool result;
- a `<system-reminder>` text block ahead of the task in the first user message.

Some user messages are Claude Code's own: after a `Skill` call, the skill's text (see
[Skills](#skills)), and messages of nothing but `<system-reminder>`s. The adapter names them
(`harness_note`), and a run's page shows each under the step it followed; a person's words,
reminders beside them or not, are their turn.

Responses stream as standard Anthropic events. llama.cpp maps Gemma's reasoning to
**`thinking` blocks**, which Claude Code sends back in later requests.

A tool call's input streams as JSON text (`input_json_delta`), and tuningfork keeps that text
as the call's arguments, as the server sent it. Written again from the parsed input, the
arguments would lose the model's spacing, characters and numbers, and scoring the answer would
rebuild other tokens than the model's. Every call the model answered keeps the stream it sent,
so a step recorded before tuningfork kept the text is scored with it too.

### The last message is re-encoded on every call

To mark it for prompt caching, Claude Code turns the last message's content into a
one-block list carrying `cache_control` — and turns it back into a plain string on the next
call, when another message is last. A message tree keyed on exact bytes reads that as a
branch at every step. The Anthropic wire format therefore gives the tree a stable form of
each message (no cache markers, a lone text block as the string it is), and each step keeps
the exact form of any message that differed, so the request is still rebuilt byte for byte.

## Skills

Loaded through the `Skill` tool: `{"skill": "<name>", "args": "..."}`. Its result is
`Launching skill: <name>`, and the same user turn continues with a text block:

```
Base directory for this skill: <config dir>/skills/<name>

<the SKILL.md body — frontmatter stripped>

ARGUMENTS: <args>          (only when the model passed arguments)
```

So the model never sees the frontmatter, and the text matches no file byte for byte.
Batch runs record each skill's body hash and the file line it starts on, which is how the
UI still maps the text to `SKILL.md` with the file's own line numbers. The `ARGUMENTS`
paragraph is cut off only when the call passed exactly those arguments, so a skill that
itself contains the word is not cut short. When the same user turn carries other text
blocks too (a reminder, say), the skill is one block among them and is shown decoded.

A wording A/B finds the skill in this same form. It follows that an edit to the
frontmatter alone — the `description`, say — changes nothing the model sees at the step:
`tuningfork ab` refuses such a comparison rather than sampling one request twice, and
notes when part of an edit falls outside what the model sees. (Claude Code also lists each
skill's description in the `Skill` tool's own description; that listing is not swapped.)

Absolute paths are common in Claude Code's tool calls (`file_path`), and each batch run has
its own workspace, so the outcome normalizer masks the per-run workspace and home prefixes
wherever they appear — file arguments, shell commands, code, file contents. A file argument
inside the workspace is compared as the relative path it names: in a real batch, some runs
read `data.csv` and others `/…/workspace/data.csv`, and they did the same thing.

## Holding a call

A held streaming call starts its response — headers at once, then an SSE comment every
10 s — so Claude Code's first-byte timeout (`CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS`, which
waits for response headers) does not fire. Claude Code also has stream-idle timeouts
(`CLAUDE_STREAM_IDLE_TIMEOUT_MS`, `CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS`), a stream watchdog,
and an overall `API_TIMEOUT_MS`, whose defaults are not visible in the shipped binary.
Measured: a call held for **305 s** survived with the connection open throughout, and the
run finished normally afterwards. Longer holds
are untested; raise those variables in the launch environment if a session needs them.

A dropped call reaches Claude Code as an SSE `error` event in Anthropic's shape.

## Launches: every call, attributed as it arrives

A run tuningfork starts is a *launch*, and Claude Code's `ANTHROPIC_BASE_URL` is
`http://127.0.0.1:8787/_launch/<token>`. Verified on 2.1.281: every call keeps the prefix —
`HEAD /api/hello` at start-up, every `/v1/messages`, and a subagent's calls.

## Subagents

The subagent tool is `Agent` (`Task` in earlier versions), with `description`, `prompt`,
`subagent_type`, `model`, `isolation` and `run_in_background`. A subagent runs in-process on
the parent's base URL, so its calls reach the same launch as a thread of their own, with a
system prompt of its own and fewer tools (13 against the parent's 20). Its first request
carries the `prompt` the parent passed, verbatim — how its thread is matched to the call
that spawned it. With `run_in_background` the parent carries on at once; in a recorded
run, the child's two calls fell between the parent's steps 2 and 3.

Seen once: the model's first `Agent` call omitted `description`, was refused by Claude
Code, and the second one spawned the subagent. The thread is matched to the second.

## Replaying a recorded run (forks)

A fork answers a relaunched Claude Code from the recording until the step it goes live at.
Two things differ between two launches of one task besides per-run paths, and are declared
volatile in the adapter:

- `metadata.user_id`: a device id and a session id, new in every launch (not shown to the
  model);
- the memory directory named in the system prompt, `<config>/projects/<the working
  directory, slashes turned into dashes>/memory/`.

A run with a subagent adds four, all in what `Agent` returns or later reports back: the
agent's id (`agentId: …`, `with to: '…'`, `<task-id>`); for one in the background, the file
its transcript goes to (`/tmp/claude-<uid>/<working directory, dashed>/<session>/tasks/<id>.output`);
how long it took (`duration_ms`); and its token count (`subagent_tokens`), which counts the
per-run paths the child read — 9130 in a recording, 9126 in its fork.

With those masked, every replayed request matched the recording exactly. The billing header
in `system` names a version suffix (`cc_version=2.1.281.<xxx>`) that differs between
sessions and between agent types, but was the same on replay, so it is not masked.

**Absolute paths.** A replayed answer is rendered afresh with the source run's workspace and
home swapped for the fork's, never passed through as bytes: Claude Code's `Write` and
`Read` name absolute paths, and in a stream a path can be split across chunks, where a
find-and-replace on raw bytes misses it. Before this was done, a fork's `Write` went to the
*source* run's workspace — and the comparison, which masks per-run paths, could not see it.
