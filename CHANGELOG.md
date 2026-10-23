# Changelog

## 0.5.0 — Qwen models, token by token

Token-level work — scoring an answer, forking tokens, a resample's kept tokens marked by part,
the model's own odds of each tool — reads Qwen3.5 and 3.6, Qwen3, Qwen2.5 and Qwen3-Coder now,
beside Gemma 4, each as llama.cpp reads it. A Claude Code step is scored on the tokens the model
wrote, a key on a line of its own is no longer stored in clear, and the README opens in plainer
words.

- **Qwen3.5 and Qwen3.6** (`qwen3.5`, as the server check names the format), read as llama.cpp's
  Qwen3-Coder parser reads them: an answer can begin inside a reasoning its prompt opened, and
  each argument is typed by the tool's schema, as the server types it.
- **Qwen3, the 2507 updates included, and Qwen2.5** (`qwen3`, `qwen2.5`), whose calls are JSON
  inside `<tool_call>`, read as the parser llama.cpp builds from their templates reads them. A
  Qwen3 template that renders no reasoning, as ggml-org's for Qwen3-2507 Instruct does, gets
  Qwen2.5's parser from the server, and is named `qwen2.5`.
- **Qwen3-Coder** (`qwen3-coder`), read by the same parser of llama.cpp's as Qwen3.5, in its
  other setting: no reasoning, and a first call with its `<tool_call>` or without it.
- **Texts the server reads alike** give the same parsed answer: for Qwen3, an empty reasoning
  and none; for Qwen3 and Qwen2.5, a newline and a blank line between calls; for Qwen3-Coder, a
  first call with its `<tool_call>` and one without. Scoring keeps the text with the recorded
  number of tokens, and where several have it, the one the model finds likelier where they part.
  `tuningfork tokens` and a step's *Tokens* view then say "rebuilt as the likelier of 2 texts the
  server reads alike", and a forking probe from it warns.
- **Checked on real servers:** Qwen3.5 27B and Qwen3.6 27B; Qwen3-30B-A3B,
  Qwen3-30B-A3B-Instruct-2507 and Qwen3-4B-Thinking-2507; Qwen2.5-32B-Instruct;
  Qwen3-Coder-30B-A3B-Instruct and Qwen3-Coder-Next. On each, the server's own answers were read
  as it read them and written back token for token, and the real-server tests
  (`tests/test_real_server.py`) pass.
- **A character written in two tokens.** llama.cpp can report it as one probability entry, which
  holds the whole character, and Qwen does this. Scoring and a forking probe's continuations split
  it back into one entry a token, by the token ids the server returns: a token merged away has no
  probability of its own, shown as "—". Without this, scoring would refuse such an answer. A
  resample's kept tokens keep the one entry: the chat endpoint returns no ids to split it by.
- **Setting up a server for them** (`docs/setup.md`): a command for Qwen3.6 27B,
  Qwen3-30B-A3B-Instruct-2507 and Qwen3-Coder-30B-A3B, how Qwen3-Coder-Next fits one 32 GB GPU,
  and how to extend a model file whose context is too short for Hermes Agent. Qwen3.5 and 3.6
  need `--ctx-checkpoints 0`, and `--swa-full`, an alternative for Gemma 4, does not help them.
  The glossary and the Guide's banner say so.
- **A Claude Code step is scored on the tokens the model wrote.** A streamed tool call's
  arguments are kept as the server sent them. They were written again from the parsed input, with
  characters outside ASCII escaped, numbers normalized (`61.50` became `61.5`) and the model's
  spacing lost. Scoring could then rebuild other tokens than the model's, and said "rebuilt
  approximately". A step recorded before is scored from the stream the server sent, which the
  recording kept. An answer that was not streamed comes parsed, and its arguments are still
  written from it.
- **The README** opens in plain words, and its quickstart comes early: install, a five-minute
  walkthrough without a GPU, then your own agent. Its figure is redrawn as the loop in four steps,
  with counts only, and its tagline is now "Tune agent skills to your own model." The worked
  example, *See it work*, now follows *Let an agent drive it*.
- **find-the-file's forking token, measured again** with 40 continuations a word, where the
  earlier probe drew 10: after "Wait", 30 searched for the file, and after "According", none
  (p = 1.9 × 10⁻¹³). Its section 4 now calls "Wait" a forking token.
- **The examples guide's Claude Code steps**, recorded again with the arguments kept as sent.

Changed:

- **Redaction reads stored text as it reads.** It matched its patterns against JSON as written,
  where a newline is a backslash and an `n`: an `sk-` key on a line of its own, or after a tab,
  then followed a letter, and was stored in clear in every harness's recordings. Nor did
  `api_key: …` match inside a call's arguments or a tool's JSON result, where its quotes are
  escaped. The patterns now also run over the text with its JSON escapes read, three levels deep:
  a payload, the arguments or JSON result in it, and a JSON file that result shows. A key with an
  escape inside it is masked whole, in a Claude Code call's arguments too. A token stream reads
  the model's own escapes the same way. Recordings made before keep what they stored.
- **A Claude Code step's arguments read as the server sent them** wherever they are printed as
  text: `tuningfork step` and `timeline`, a held call, and MCP's `show_step`. Before, they read as
  Python's `json.dumps` of the input, with a space after each `:` and `,`, characters outside
  ASCII escaped and numbers normalized. The web UI shows each argument as a field, as before.
  Steps recorded before keep what they stored. A behaviour, a target and a breakpoint read the
  parsed arguments, so none of them changes.

Nothing already in the recordings folder changes: no new columns, nothing to migrate.

QwQ is not read yet: llama.cpp reads it with a parser of its own.

## 0.4.0 — the web UI, redesigned

The web UI, page by page, after a review of it: a sidebar in the loop's order, plainer words, one
meaning a colour, each page leading with its take-away, and the work on a step beside the run.
The CLI, MCP and the docs keep tuningfork's precise words.

- **A sidebar in the loop's order**, where the top bar's ten tabs were: the Guide and its Words,
  Connections; then Start (Scenarios, Run a task…, Import a chat, Live, Skills, Jobs), Record
  (Runs, Chains), Measure (Experiments), Improve (Refine) and the Trash. Its foot says how the
  proxy, the server and the harnesses are. What needs a running proxy stays in it, greyed, saying
  why. Narrower, it shows only its icons; on a phone, it opens from a bar across the top.
- **Plain words on the pages**: an arm is a *wording*, a target *what you count* (a refinement's,
  *what the model should do*), a screen a *test at this step*, a validation *full test runs*, the
  gate the *approval checks*, a spec a *task file*. The CLI, MCP, the docs and every sentence the
  server writes keep the precise words. The glossary holds both: the Words page (`#/words`, where
  older `#/guide?term=` links land) shows them side by side, and `tuningfork glossary` finds a
  term by either and prints the page's phrase.
- **One meaning a colour**: green and red for a check's verdict — a run's, a fork's, a gate's, a
  significant change in how often a task's check passes; amber a caution; the accent what can be
  acted on, what is current or significant, the model's own answer and every rate, a pass rate
  included; purple a skill's own words; teal, with ✎, what a person or tuningfork changed; grey
  everything else. Red keeps only what every page means by it besides: a diff's removed lines
  beside its green added ones, a button that deletes, a field to put right. The Guide shows the
  legend in a line.
- **Each page leads with its take-away**: a run's verdict in a strip under its title, with the
  call that made the check's complaint where the check was run along the run; on the divergence
  page, the step whose behaviour best predicts a pass; Connections' setup at a glance; a
  refinement's four stages in a line.
- **A run's page reads top to bottom, its work beside it.** Every step's row has *Show details*
  and *Work from here*. One step's details are open at a time. A step's actions, and the forms
  they open, sit in a column beside the run, each step's kept as it was left, and the address
  names that work, so a reload brings it back; Escape hides it. The rail's lines move you through
  the run and open nothing. On an imported or live run, *Fork from here…* is greyed, and points to
  a scenario of the run instead.
- **A refinement's page has a section for each stage**, numbered as the stepper numbers it: the
  two ways to propose side by side; the proposals, with the screen of the ticked ones (a *test at
  this step*) under them; the validations (*full test runs*); approval. The current stage's form
  is open as the page opens, and no form opens by itself after.
- **A finished job hands off to what it made**: its page says what it found in a line and leads
  there, its log folded. The Jobs list is in three parts — running and queued, each queued job
  saying its place in its line; finished; stopped or failed — and names each job by the section
  of its work, links what it made, and says who started it: you, from which page, or an agent
  over MCP. `tuningfork jobs list` and `jobs show` say who started it too. A job's link to the
  step it worked at opens that step.
- **Add a skill** (`tuningfork add-skill`, `forget-skill`, the Skills page's *Add a skill…*, a
  run's *Keep its skill…*): a folder on this machine, remembered where it is, or an uploaded
  folder or a run's skill, kept as tuningfork's own copy. An approved edit is written to the
  folder, or to the copy. Forgetting one drops its record and never deletes: a copy is moved aside
  to `<home>/skills/.forgotten/`. No MCP tool adds one. A skill's page and `tuningfork skill` say
  how its folder was added, and its page runs it again from the scenarios that name it.
- **Token work is held to the server check.** Scoring an answer and finding forking tokens are
  greyed on the page, and refused before anything is sent, where the check found the server
  without what they need. The check now measures the chat endpoint's logprobs too, which a
  resample's kept tokens need (*Keep each sample's tokens*, new on its form), in one more short
  call. A server on the internet that asks for a key, or is not llama.cpp's, is said to be a
  hosted model's API: tuningfork records your agent's calls through it, its own key passed on.
  The switch on Connections takes one that asks for a key, where it refused it before.
- **The forking-token probe says what reached each behaviour**: whether the target counts it,
  and which tokens led to it — its page, `tuningfork experiment`, MCP. A flip is the accent's
  colour, its direction said in words. Where the step's answer is a tool call, the form offers
  that tool, with any arguments, as the target first.
- **Forms**: one nobody has touched says what it needs as a note, not in red. *Make a scenario…*
  shows what it will make first, and chooses its check as a new scenario's editor does. *Run a
  task…* takes a task file's path beside the scenario, greyed while a scenario is chosen.
- **The README's pictures**, retaken in the redesigned pages, with a new one: a chat had
  elsewhere, made a scenario. `docs/images/shots.mjs` takes them all, the GIF included, in one
  command, and keeps back any picture that shows a path naming a person or a place, or a word you
  list (`TUNINGFORK_SHOTS_PRIVATE`).

Changed:

- **An A/B is judged at the α it was declared with** wherever it is reported again — its page,
  `tuningfork experiment`, MCP's `show_experiment` — where 0.05 was used whatever was declared.
  An A/B made before its α was kept takes its job's, else 0.05; `--alpha` judges at another
  level, and says so.
- **Letters agree with the resample a target came from**: an A/B, a screen or a probe of a letter
  target letters its behaviours as that resample did, so its B is the resample's B. One stored
  before finds its resample by its target's description, so read again, its behaviours may take
  new letters.
- **A resample that asked for each sample's tokens and got none says so**: its page said every
  sample had kept them. The page now counts only the samples that did.
- **A dry run's refusal is an answer**: asked `quiet`, as the web UI's forms ask, `POST /jobs`
  and the scenario endpoint refuse with `200 {refused}`, not an HTTP error. Without `quiet`,
  nothing changes.
- **A plan gives its notes apart from its warnings**: an import's and a scenario's (`notes` in
  the API, `note` lines in the terminal).
- **A scenario's batch or chain names the scenario when refused** for the task file the scenario
  keeps: the field is `scenario`, the one filled in, never `spec`, which nobody typed.
- **Every job under way is listed** by `GET /jobs`, `tuningfork jobs list` and MCP's `list_jobs`,
  however many came after it: the list was cut at the newest 50 (`list_jobs`: 20, or as many as
  it asked for), so a running job with that many newer ones dropped out. Over MCP's stdio
  transport, `list_jobs` still gives at most the 50 newest beside them, whatever it asks for.
- **A launch is refused where the server asks for a key**: a harness tuningfork launches carries
  none, and each harness's card on Connections says so. The server check says such a server
  cannot be checked.
- **A proposing model's address that cannot be read is refused** when it is set: a mistyped port
  was kept, and failed only when the proposer was asked. One saved before is shown as another
  machine's until it is set again. `GET` and `PUT /proposer` give `remote`.

## 0.3.0 — chats your agent already had, and any run as a scenario

A conversation your agent had elsewhere comes in as a run, and any run becomes a scenario: a task
whose runs continue its conversation, so a skill is measured where it was really used.

- **A chat, imported** (`tuningfork import`, the Runs page's **Import a chat…**): Hermes Agent's
  session file or its export, or a list of OpenAI-style messages, becomes a run whose every
  assistant turn is a step, so resample, A/B, the token tools and a refinement work at it. Each
  step's request is rebuilt as the harness sends one: the system prompt, everything before it,
  each message as Hermes cleans its own copy before a call. A file that kept no tools (Hermes's
  export keeps none) takes them from a run recorded through the same harness (`--tools-from`,
  **Tools from**). Every skill the chat shows is kept with the run, and an A/B, a refinement or
  a scenario names the one it studies (`--skill`, a **Skill** picker).
- **Any run as a scenario** (`tuningfork scenario <run> <name>`, a run page's **Make a
  scenario…**, the Scenarios page's **From a run…**): an imported run, one tuningfork launched,
  or one recorded live.
  - **Its prompt** is the person's message the skill came in after, in their words or yours
    (`--prompt`), and the conversation before it is each run's context. The proxy puts it after
    the harness's own instructions in every call (`tuningfork.context`; harnesses that speak
    OpenAI chat), so the skill still comes in live and an edited version reaches it.
  - **Several skills in one chat:** the steps of that turn before the studied skill came in,
    another skill's among them, go to each run as done. Where the chat went on to another skill,
    each run stops there and the check judges the workspace as it is then
    (`tuningfork.stop_at_next_skill`).
  - **A launched run's scenario** takes the run's workspace there as its input files, from the
    snapshots, and its task's check, with the files that check reads. Any other run lists the
    files its calls named, for you to add.
  - **Its eval is named by the scenario:** its runs are a task of their own, never counted with
    the task of the run it came from.
  - **The form** asks for the skill first, then sets each run beside the chat: what each run is
    given (its harness's own system prompt and tools, the conversation before, the prompt),
    then each step, given as done, made by its model, or past where it stops.
- **A run's page:** the tools the run was offered, above its system prompt; each tool's result
  under the step that called it in the rail; a person's turn told apart from the notes a harness
  writes itself (Hermes's nudge after a cut reply, Claude Code's reminders).
- **A run recorded live says so**, where a label would name it: the Runs list, its page,
  `tuningfork runs` and `show`. When its first call continued a conversation already going on, a
  long session, its rail lists that conversation first, and each of the person's turns there
  folds to its line.
- **The README says where your harness points**, what `--upstream` is, and how to change the
  port.

Changed:

- **A new run is marked a branch of another only where the two share more than their first
  message**, at the message where they part. Two sessions of one harness can share a system
  prompt and nothing else: one was marked a fork of the other, at its own length. Marks recorded
  before stay as they were.
- **The recordings' database gains a column**, each step's context: it is added when an older
  folder is opened.

Claude Code's calls cannot yet carry a conversation before the prompt, so a scenario of a Claude
Code run starts at its first turn.

## 0.2.3 — on Windows, and a code review's fixes

Three rules changed in review, and what was recorded before reads by them now:

- **A fork run to the end that timed out counts as a failure**, in a screen, an A/B and a
  resample to the end, as the validation gate already counted it. Left out, an edit that made
  the agent loop until it was stopped counted only its forks that finished, and could survive
  its screen. A screen read again counts its timed-out forks, and its report says how many.
- **Gate G0 reads a proposal's latest complete screen**, not any screen it survived: a screen
  of more proposals, or one run to the end, can undo an earlier survival. Validation refuses a
  proposal its latest screen dropped, and a validation's gate, read again, fails G0 when a
  screen since has dropped it, and says which.
- **A fork an experiment drew goes to the trash, and is deleted, only with its experiment**, as
  a thread goes with its run: its verdict is the experiment's. Trashed alone, even deleted for
  good, it still counted. One trashed alone before can be restored, and is never deleted alone.

**On Windows** (`docs/setup.md`): installing WSL and a distro, working inside it as on Linux,
the web UI in the Windows browser, and a model server running on Windows itself. Or keep the
checkout where Windows editors see it: `scripts/tuningfork.ps1` (PowerShell) and
`scripts/tuningfork` (Git Bash) run any tuningfork command inside WSL, the venv and the
recordings in the distro's Linux file system. They run nothing until `TUNINGFORK_WSL_DISTRO`
names the distro, and hand each argument over as it was typed.

Fixed, each found in a review of the code and reproduced before it was fixed:

- **The proxy:** a harness that hung up before its answer began kept its slot in the gate for
  good, and with `--max-inflight 1` every later call waited for a restart. `serve --max-output`
  raised a limit written as a float (`1024.0`) to the cap. Ctrl-C continued a cancelled job's
  held calls before dropping them, so a fork killed at a breakpoint still sent its call.
- **Recording and forks:** a call refused while the proxy was paused kept its step number: the
  run skipped a number, forks of later steps went live a call late, and its last step could not
  be forked. A fork's call at its step, refused that way, lost its "live from here" note.
- **The trash:** deleting a run for good while a fork of it still ran took the fork's folders
  from under its harness. A launch a batch killed hard stayed running for good and held back
  the trash: launches record the process running them now, and read as interrupted once it has
  gone. Deleting a chain link's run for good deleted the skill the link left.
- **Counting:** gate G2's "enough runs" counted runs the infrastructure broke. A screen to the
  end from several checkpoints could tilt the original's rate toward one of them. The tally
  named a mistake every run kept as one the passing runs repaired, and counted the half line a
  cut output starts with as a finding. A check output longer than the 2,000 characters a
  verdict keeps says so now: on the run's page, in MCP's `show_run`, in the tally and in a trial
  of the check.
- **Refinements and approvals:** a record cut off mid-write stayed truncated, and a truncated
  `approval.json` blocked its refinement for good. A crash between an approval's last two
  writes left it open for good: it is reported (`unmarked`) and finished by repair. A changelog
  that is not UTF-8 stopped approvals. A file that mixes line endings was written back all
  CRLF: it takes the ending most of its lines use now, and the plan says so first. A proposal's
  edit matched with its indentation ignored turned that line of a CRLF skill into LF.
- **Jobs and scenarios:** two cancels of a queued job at once left it queued until a restart;
  the same work submitted twice at once started two jobs; a scenario deleted twice in one
  second landed in the first copy's trash folder.
- **MCP and the CLI:** `job_status` showed a held-out task's jobs whole, and `list_scenarios`
  its pass rates. Pointed at a proxy on another machine, the CLI read the local file its
  `/health` named and sent it there as the token.
- **The web UI:** the batch form kept the old spec's task list; a stopped run read "no check";
  a live run's timeline drew its "not recorded" rows again at every update; on the Live page, a
  question still being answered at a held call showed as an error; on the Jobs page, a late
  answer filled the step list of the run chosen last; "Where does it hold?" lost what was typed
  when the page redrew; a job's page could go back to "running" after the job ended; and two
  step forms asked for quickly could leave the wrong one open, or both.

## 0.2.2 — a README for newcomers, and what its pictures showed

- **The README, rebuilt for someone deciding in a few minutes whether to try tuningfork:** what
  it does in one figure drawn from real runs, a quickstart with no GPU on the demo lab, where
  every button works, the commands by the question each answers, and the web UI in screenshots
  and a GIF. Reference moves to `docs/setup.md` (installing, the model server's flags and why,
  the harnesses, the sandbox, a task's format), `docs/commands.md` (every command, by the
  question it answers) and `docs/agents.md` (the MCP server, client by client);
  `docs/images/README.md` says how each picture is taken. `tests/test_docs.py` fails on a
  command the reference does not name, and on a link, to a page or an image, that leads nowhere.
- **`examples/find-the-file/`: a step where the model wavers**, the runs behind the figure,
  Hermes Agent on Gemma 4 31B: ten runs pass 4, the step where they split resampled, the
  model's odds for the words where its reasoning forks, one line of the skill tested there,
  and ten runs with it passing 10, each command shown with its output.
- **An A/B shows the edit it tested:** its report starts with a line diff of the two texts,
  read back from what it recorded rather than from files that can change since (its page,
  `tuningfork experiment`, MCP's `show_experiment`), where one text holds the skill twice too.
  The demo's seeded A/B swaps a real saved variant.
- **Green and red only for a check's verdict.** A run's page ends with its verdict, coloured by
  it, where a failed run used to end in a green reply. Any other significant result, an A/B's
  shift or a step screen's survivor, is the accent's colour, its direction said in words, and
  the behaviours a declared target counts are badged *target*, the others *not target*. The
  divergence page colours each run by its own verdict.
- **A refinement's page leads with its evidence:** where it stands, each proposal's screen and
  the approved one's validation, linked to the detail below. What a proposer is shown moves
  under the proposals, and the skill's paths into a closed *About this refinement*.
- **A forking-token probe that found nothing says what it could have seen:** "No forced token
  decided what the model did", and the smallest change its continuations would show
  (`detectable`), on its page, the token panel and the CLI.
- **The server advice is current:** the front page and the glossary recommend
  `--no-cache-prompt`, and `--ctx-checkpoints 0` or `--swa-full`.
- **Start takes the first click after typing in a field.** The field's change as it lost the
  focus disabled Start under the click, on every form that starts work.
- **A run stopped before it finished says so,** not that its task defines no check, and a run
  launched from the command line shows its verdict without a reload.
- **p-values below 0.01 print by their first figure** (`0.005`), never `0.00`. The tally's
  footer, in `tuningfork tally` and MCP's `tally_failures`, says in plain words how it counts:
  an agent reads new words there.

## 0.2.1 — the web UI catches up

- **Screens run to the end, from the refinement page.** Its screen form asks how each sample
  is judged — an answer at the step, or a fork run from the step to the end, the task's check
  deciding — and to the end takes the runs to fork from, a quick look, and drawing the
  original anew. A refinement can be started with the task's check as its target. A quick look
  reads as one wherever it shows — the refinement's page and its screens, the screen's own page,
  `refine show` — its rates and never a verdict, not "not significant" against an original of
  0/0. A screen to the end shows each run it forked from, with each arm's passes there and
  whether the effect held at every one; its page asks Where does it hold? of its forks, arm by
  arm, and folds away the step's behaviours, which are each fork's own. A proposal's chip says
  it survived when any screen found so: it showed the oldest screen's verdict.
- **What goes with passing**, on the divergence page: the step whose behaviour best predicts
  the verdict, and the tool whose use does, at whatever step a run called it — or that no run
  passed, and where to look then. The divergence report carries the step's split for every
  front end (`verdict_split`), with a run that took the behaviour to look at.
- **The check run along the runs**, started from the divergence page (the `check-along` job):
  the tally beside it then names the call that made each finding.
- **A chain, link by link** (the Chains page, `GET /chains`, `/chains/{id}`): how it ran, what
  each link's harness was told, its verdict (or none, where the check never ran), who changed
  the skill after it and the lines they changed. A chain job's page leads to it.
- **What work cost** beside a run, a job and an experiment (`GET /spend`).
- **A run's sandbox** on its page: where its harness worked, the folders hidden from it, the
  instant its clock started at, and how the harness was set up.
- A running job says about how long it has left. The Connections page shows the cap on output,
  and its restart command keeps the cap and the MCP route's settings.

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
  call asking for more than N output tokens, or naming no limit, is sent with N — a call a
  person edits at a breakpoint, and a validation's held-out runs, included; the step records
  what the model was sent and what the harness asked, and a fork's replay matches the
  harness's own request. Hermes answers a reply cut at its limit by asking again for more, so
  the cap has to be the proxy's: an agent's reply looped to Hermes's 65,536 tokens for 35
  minutes of one GPU.
- **Every launch's clock can be pinned** (`TUNINGFORK_CLOCK`): the harness reads the same date
  whichever day a run is made, and a fork starts where its source did.
- **Fixed:** a line of the original that a longer line of the variant contains is no longer taken
  for a leak (it had voided twenty forks of one screen); a tally masks a hex colour whole; parsing
  a recorded script for its digest no longer prints its warnings.
- **The guides on Gemma 4 31B.** `examples/README.md` was run again as written on one 32 GB card,
  every output captured there, and every step of Parts 2 to 4 now has a "With Hermes Agent" block:
  Hermes's own edit validated three ways and approved.
- **A benchmark driver** (`benchmark/driver/run.py`): Hermes's own self-improvement loop, as a
  told chain, against Hermes with tuningfork's tools over MCP — the same task, model and server,
  each arm's final skill judged by fresh runs — with one task, `benchmark/architecture-diagram/`.
- **A case study** (`examples/hermes-architecture-diagram/`): on Hermes's `architecture-diagram`
  skill, its own loop reached 9/10 by growing the skill, templates of the training answer
  included; Hermes with tuningfork's tools reached 8/10 with an 8-line edit, kept only after it
  beat the original in forks. A demonstration of the tools, with an agent standing in for a
  person.

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
