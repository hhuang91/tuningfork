# Adapter template

Copy this file to `docs/adapters/<harness>.md` and implement `tuningfork/harness/<harness>.py`
alongside it. `hermes.py` and `openclaw.py` (OpenAI chat-completions) and `claude_code.py`
(Anthropic Messages) are the worked examples.

You do **not** need an adapter to record traffic. Any harness that speaks OpenAI
chat-completions or Anthropic Messages and accepts a custom base URL can be pointed at the
proxy and recorded today. An adapter is what lets tuningfork
*launch* the harness itself — which is what batch runs, resampling and A/B tests need.

## What an adapter must provide

Four methods, defined in `tuningfork/harness/base.py`, and seven optional:

| Method | Purpose |
|---|---|
| `prepare_home(home, base_url, model)` | Build a disposable, isolated home directory configured to talk to `base_url` |
| `launch(prompt, home, workspace, skills, max_turns, …)` | Return a `LaunchSpec` that runs one prompt non-interactively |
| `skill_roots(home)` | Where this harness keeps its `SKILL.md` files |
| `skill_spans(request)` | Where skill text sits in a request's OpenAI-style view, for the UI |
| `isolation_warnings(workdir)` *(optional)* | What could still reach every run from outside it, printed before a batch starts |
| `as_sent(messages)` *(optional)* | A kept history as the harness sends it — what it changes in its own copy before every call — for a request rebuilt from an imported chat |
| `harness_note(message)` *(optional)* | A user message the harness wrote itself (a reminder, a nudge to go on), named in a few words, or `None` for a person's turn |
| `side_call(request)` *(optional)* | A name for a call the harness makes for itself beside the task's next step — Hermes's background review, OpenClaw's compaction summary — or `None`: such a call starts a side thread of its launch, and a run's final reply is never one |
| `skill_load(call)` *(optional)* | The skill a tool call loads, by name: where a scenario's run that stops at the next skill ends |
| `for_server(server)` *(optional)* | The adapter set up for the server it will talk to, from the server's identity, for a harness that does not ask the server itself (OpenClaw's context window); a batch calls it, but not one set up as a recorded run was, and what it sets goes in the adapter's `settings`: a fork gets its source's, and every run of a validation the checkpoint run's |
| `version_problem(output)` *(optional)* | Why a harness that answers `--version` still cannot launch, read from what that printed, or `None`: the Connections page says the harness cannot run here, and why, and a batch or a fork is refused in those words before anything launches |

And five optional attributes:

| Attribute | Purpose |
|---|---|
| `executable` | The program `launch` runs. `ops/connections.py` finds it on `PATH`, or at an override named in the recordings folder's config, and reads its `--version` in a throwaway home |
| `requirements` | What it needs of the inference server, as `Requirement`s: an identity field with a `minimum` (`n_ctx`), or a capability a server check measures (`streaming_tool_calls`, `anthropic_messages`). A batch or fork refuses to launch it against a server that lacks one |
| `subagent_tools` | Tools whose calls start a subagent, so a launch's threads can be told apart |
| `volatile` | `(pattern, replacement)` pairs over a request's JSON text for what differs between two launches of one task besides per-run paths — a session id, the date, a tool's timing — so a fork can recognise the calls it replays |
| `fed_back_side_calls` | The side calls, by the name `side_call` gives them, whose answer the task's conversation goes on from (OpenClaw's compaction summary): a failed or edited one leaves the run out, as a task call does. Any other side call is the harness's review of its answer, whose failure says nothing of the run |

`prepare_home` may also take `max_turns`, for a harness whose config holds a turn limit of
its own. Register the adapter in `tuningfork/harness/__init__.py::ADAPTERS`.

## Launches and forks

tuningfork gives each run it starts a base URL with a token in its path,
`http://127.0.0.1:8787/_launch/<token>/v1`, and attributes every call by it. Check that the
harness keeps the path on **every** call it makes — a subagent's too. A harness that drops
it cannot be told apart from another run of the same task.

To find the `volatile` list, relaunch a recorded run with its answers replayed and compare
each request the harness sends with the recorded one: whatever differs besides per-run
paths is a candidate. The Hermes, Claude Code and OpenClaw pages beside this one list what
that found for each.

## A harness that speaks another dialect

The proxy reads each call through a wire format (`tuningfork/wire/`): OpenAI chat-completions
and Anthropic Messages exist. A harness speaking anything else needs one more module there.
It parses and reassembles the dialect's responses, converts them to and from the common
shape, renders an edited response back for the harness, and gives a read-only OpenAI-style
view of the conversation. Watch for a harness that re-encodes earlier messages from one
request to the next — Claude Code moves prompt-caching markers every call. Give the format
a `normalize_message` that undoes it, or every step will look like a new branch of the run.
`tests/test_wire.py` shows what to test.

## Isolation is the hard part

Agent harnesses accumulate state, and some of it will silently change what you are trying to
measure. Before writing the adapter, find out whether yours:

- **rewrites its own instructions during a run.** If skills or prompts can be edited
  mid-experiment, the thing under test is not fixed. Hash the files before and after, and
  invalidate the experiment if they moved.
- **carries memory between sessions.** Then consecutive runs are not independent samples,
  and every statistic assuming independence is wrong.
- **runs background maintenance.** A curator or indexer touching instruction files during a
  run has the same effect as an edit.
- **shares locks, sockets or credentials.** An experiment that grabs a messaging connection
  or a lock file can disturb a real installation of the same harness.
- **reads instructions from the directories around its working directory.** Claude Code
  loads every `CLAUDE.md` from its working directory up to the root; a batch run's
  workspace sits inside the recordings folder, usually inside the user's home. Check by
  planting a marker file above a run's workspace and looking for it in the recording.

The best isolation primitive is usually a single environment variable that relocates the
harness's entire home directory. Prefer it over per-profile settings, which typically still
share the surrounding state. Pass through as few of the harness's own variables as you can:
a stray one in the parent shell is an unrecorded change to the experiment.

**Never** rely on a mechanism that mutates global state, such as a command that sets a
sticky default profile. It will affect whatever else uses that harness on the machine.

## Documenting payload anatomy

The adapter doc should answer these, because Phase 2's wording A/B depends on knowing where
instruction text actually lives in the payload:

1. What does the system prompt contain, and how is it assembled?
2. Where do skill *descriptions* appear, versus full `SKILL.md` *bodies*?
3. What does loading a skill look like on the wire — a tool result, a new system message?
4. Which sampling parameters does the harness send, and which does it leave to the server?
5. What does a subagent call look like, and can it be told apart from the parent?
6. What does context compaction look like when it happens?
7. What request timeout does it enforce? (Phase 3 breakpoints hold a response open.)

Record what you actually observed in a recording, not what the documentation claims.

## Checklist

- [ ] `prepare_home` writes a config that points at tuningfork, and touches nothing outside `home`
- [ ] `launch` is non-interactive and needs no TTY
- [ ] Only the isolation variable is passed through; other harness variables are dropped
- [ ] Tests in `tests/test_harness.py` cover argv construction and env isolation
- [ ] Adapter doc records the harness version it was verified against
- [ ] `executable` is set, and `--version` answers without touching the harness's real home
- [ ] `requirements` lists what it needs of the server, each with why
- [ ] Every call keeps a base URL's path (launches), subagents included
- [ ] `volatile` lists what differs between two launches of one task, found by replaying one
- [ ] The full suite still passes with no GPU and no model
