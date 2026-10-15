# Changelog

## 0.2.0 — agents that refine skills, and loops to compare them with

- **A chain of runs** (`tuningfork chain run`, the Run a task form, the `chain` job kind):
  a task run N times in order, each run given the skill as the run before left it — how a
  harness's own self-improvement accumulates. Each link is recorded and judged as a batch's
  run is; the chain keeps every version of the skill, who changed it, the lines that
  changed, and the last link's folder as what the loop made. With `--tell`, each link's
  harness is given the check's verdict and output as the user's next turn of the same
  session, before its own review: the signal a person who looked at the result would give,
  which a harness's loop otherwise never has. With `--budget-seconds`, the chain stops once
  its links' model calls have taken that long, and a link that could outrun the budget is
  cut to what is left of it, so two loops given the same seconds spend the same seconds.
- **Options a launch of Hermes takes** (`--harness-option`, `harness_options`): turn its
  self-improvement off (a plugin written into the launch's home removes its skill-editing
  tool, so a run cannot change the skill it is measured with), let its background review
  finish before the process exits (in one-shot mode the review was cut off at exit), make
  the skill review due after every turn, and give it MCP servers — tuningfork's own tools
  among them. Recorded with each run.
- **The MCP surface oriented for a weak reader.** A 26B driving the refine loop looked for the
  skill as a file, aimed its targets at the first place tool calls differed, and never read a
  failed run's check output; Hermes shows the model no server instructions, so the tools'
  descriptions and answers now say it: show_skill is the only way to read the skill, show_run
  leads with what a failed run's check said and names its last tool call, `show_files` (new)
  shows what a run wrote, find_divergence names the step whose behaviour best predicts the
  verdict or says the difference is in the content, start_refinement refuses a target no
  recorded answer at the step satisfies (`unseen_ok` allows new behaviour) and says when the
  refinement was open already, and the refine toolset's hints walk the refine loop. An edit
  whose old text matches the skill once indentation is ignored is applied with the skill's
  indentation, and a refusal quotes the closest lines as they stand. A refusal is returned as an
  answer, not an error result: Hermes counts error results toward a circuit breaker that cuts a
  server off after three in a row, and an agent that fumbled three edits lost every tool for a
  minute; only tuningfork's own failure is an error result now.
- **A target a screen can move, and the check as a target.** Each behaviour seen at a step
  now comes with how the recorded runs that took it fared, and a target tried against them
  (step_behaviours, the targets page) says how many passing runs' answers it counts and how
  many failing runs': what a 26B driving the refine loop lacked, having declared a target
  every answer satisfied (20/20 against 20/20) and one no answer could contain (the check's
  own sentence). start_refinement refuses the first as it did the second, and a target the
  failing runs' answers satisfy as readily as the passing runs' (its screen would move nothing
  the verdict follows), with the failing runs' check output in the hint. A fork is now set up
  as its source's launch was (Hermes's `self_improvement`, which changes its system prompt):
  before, a fork of a run recorded with the option departed from the recording at its first
  step, and every sample of a screen to the end was excluded. For a failure only the check can see — a computed property
  of what the run wrote — a refinement's target may be the task's check (`--target-check`,
  `target_check`), and a screen may run to the end (`refine screen --to-end`, the
  `screen-to-end` job kind, `--mcp-tools refine-to-end`): each arm's samples forks from the
  checkpoint through the harness, the check deciding, Fisher and Holm as at the step; the
  record says which measured.
- **An answer once, for a client that shows both.** A tool result carries its text (the
  sentences, then the JSON) and its structured content (the JSON, led by the sentences)
  because a client shows the model one of them — Claude Code the structured content,
  OpenJarvis the text. Hermes puts both in the tool message, every answer twice, and a 26B
  driving the loop in a 64k context spent it in twenty-two calls, the harness's compaction
  unable to drop a turn of tool calls. `serve --mcp-results text` and `mcp --results text`
  send the text alone (`mcp.server.RESULTS`); the default is unchanged. The same 26B then
  sent an edit without its old text twenty-seven times running, reading each refusal as the
  first: an edit missing a key is now told the shape and the skill's closest line to copy,
  and the same refusal three times running carries the tool's description again, from the
  sixth time with "if it cannot be done, say so and stop" (`ToolServer._repeated`).
- **Folders hidden from every launch** (`TUNINGFORK_SANDBOX_HIDE`): the sandbox hides a
  launch's own recordings, runs and spec folder; the person can name more, where a
  benchmark keeps the keys its checks read — an agent given the file system was seen to
  find a task's check script. What a launch could not see is kept in its record.
- **What work cost** (`tuningfork spend`): model calls, tokens in and out, and time, counted
  from the records for a run, an experiment, a job, a chain or a task — the same count for a
  harness's runs and an experiment's samples.
- **What the checks said, counted, and where it was made.** `tuningfork tally` (MCP's
  tally_failures) reads every failed run's check output at once: each line a check prints is one
  finding, lines that differ only in their numbers or colours are one, each counted once a run
  with its interval, the most common first; a screen run to the end is tallied by arm. An agent
  that opened failed runs one at a time aimed at the first failure it read, not the most common.
  `tuningfork check-along` (the `check-along` job kind) runs the check on a run's workspace as it
  started, after each call that changed it, and at its end, so each finding comes with the call
  that made it — a mistake every run made, and some repaired later, is named as one, with where
  to refine it. `tuningfork where` (MCP's where_it_holds) asks one question of every run — a
  target tried at every step, or a file's pattern at each state of its workspace — lined up with
  the verdicts, with the call that made the difference: what a check that prints only its verdict
  leaves to go on.
- **The step to refine, chosen by what the run did.** A step can be named by what the run did
  there (`write_file`, `changed:<path>`, `made:<words>`) and is found in each run.
  find_divergence names the tool whose use goes with passing (runs that called it against runs
  that never did), and step_behaviours gives the model's own chance of each tool at the step,
  from scored answers. A screen to the end draws its forks from several runs at the same step
  (`--checkpoints`), passing and failing by turns, and says whether the effect held at each: one
  that lived in a single prefix is one fresh runs draw anew. A refinement at a passing run's
  step is told so, a step screen whose original is at the ceiling says no gain could show there,
  and a refinement toward the check and a screen's plan warn when a call before the step changed
  the workspace, which a fork replays as recorded.
- **Screens that cost less.** A refinement's original is forked once: a later screen to the end
  counts the original's forks its earlier screens drew at the same checkpoints and draws only
  what its n still needs (`--fresh-original` draws anew). A quick look (`--quick`, `quick`)
  forks each proposal eight times and the original not at all: rates, never a verdict, and a
  lead for the next edit.
- **Edits and long texts as an agent sends and reads them.** propose_edit takes a line by its
  number — `{line, new}`, `{line, through, new}`, `{after, insert}` — beside `{old, new}`, each
  resolved to an exact replacement, so several in one proposal do not shift each other, and text
  whose line breaks were escaped twice is read as the lines it meant. show_files reads a file or
  a diff from an offset, and show_step one message of a step whole, a page at a time: a file cut
  with no way past the cut had hidden what the check failed. Refusals say more plainly what to
  send: an edit that changes nothing is told what `new` is, and the sixth identical refusal says
  to try another edit, step or tool rather than to stop. A running job says how long is left,
  and `serve --mcp-max-wait` sets how long one MCP call may wait for it.
- **A cap on output for every call through the proxy** (`serve --max-output N`): a harness's
  call asking for more than N output tokens, or naming no limit, is sent with N; the step
  records what the model was sent and what the harness asked, and a fork's replay matches the
  harness's own request. Hermes answers a reply cut at its limit by asking again for more, so
  the cap has to be the proxy's: an agent's reply looped to Hermes's 65,536 tokens for 35
  minutes of one GPU.
- **Every launch's clock can be pinned** (`TUNINGFORK_CLOCK`): the harness reads the same date
  whichever day a run is made, and a fork starts where its source did.
- **Fixed:** a line of the original that a longer line of the variant contains is no longer taken
  for a leak (it had voided twenty forks of one screen); a tally masks a hex colour whole; parsing
  a recorded script for its digest no longer prints its warnings.
## 0.1.0 — the first public release

A step debugger and experiment lab for agent skills on local LLMs, pre-alpha. What it does:

- **Records** every model call of an agent harness, through a proxy that speaks OpenAI
  chat-completions and Anthropic Messages, with no change to the harness. Hermes Agent and
  Claude Code have adapters; any harness that takes a base URL can be recorded.
- **Finds where runs part**: a batch of runs of a task, judged by the task's check, and the
  first step where they stopped agreeing.
- **Tests a step, not a whole run**: resamples a step with its history held fixed, swaps an
  edited skill in at that step (an A/B with its target declared before sampling), and follows
  either to the end of the run, where the task's check decides. Every rate comes with its
  interval.
- **Stops a run where it matters**: live breakpoints before or after the model, at a call that
  matches, with the call held to read, edit, step or drop.
- **Restarts a run from any step**: the workspace as each call found it, kept in snapshots;
  forks replayed from the recording up to a step and live from there; every launch in a
  bubblewrap sandbox that looks the same each time.
- **Reads inside a step**, on llama.cpp: how sure the model was of each token of an answer,
  and which token decided the step.
- **Refines a skill at the step that fails**: a target declared first; edits proposed by a
  person, an agent or the built-in proposer; each screened against the original; a survivor
  validated with full runs, held-out tasks included; and written to the skill only when a
  person approves, typing a code.
- **All of it from a web UI**, or as tools for an agent over MCP.

Two guides walk through it on real runs: `examples/README.md`, a toy skill made to fail one
way, and `examples/hermes-design-md/`, Hermes Agent's own design-md skill fixed from its
failing step to an approved edit.
