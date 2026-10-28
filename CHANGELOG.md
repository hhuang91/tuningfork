# Changelog

## 0.8.0 — macOS, deleting for good, and a scan of older recordings

tuningfork runs on a Mac now, in a Linux VM. Deleting from the trash takes the folders tuningfork
made for a run's launches, and a recordings folder tuningfork makes is yours alone. A new command,
`tuningfork scan`, looks through a recordings folder for what an older tuningfork kept in clear.
Every password of a curl, MySQL or `docker login` command is masked now, not only its first, and
so is each value in the query of a server's address. `docs/setup.md` says what a recordings folder
holds, and how to share part of one.

- **macOS, in a Linux VM** (#62). tuningfork runs on a Mac inside a VM that Lima makes, as on
  Windows it runs in WSL; `docs/setup.md` (*On macOS*) says how, from a run on an M1 MacBook Air.
  From a checkout on the Mac, `scripts/tuningfork` runs any command in the VM, and Ctrl-C, a closed
  window or a killed wrapper stop it there too. `TUNINGFORK_` settings made on the Mac go along,
  never a key or a token. On a Mac outside a VM, the sandbox points at the VM instead of saying to
  install bubblewrap. The sandbox section gives the AppArmor profile Ubuntu 24.04 needs before
  bubblewrap can make a sandbox. CI also runs the suite on arm64 Linux.
- **A recordings folder tuningfork makes is yours alone** (mode 0700, #60), and so is each folder
  it lays launches out in. `serve` and `ui` say when a folder made before can be opened by others,
  and give the `chmod` that makes it yours; tuningfork never changes it by itself. Reads under
  `/_tuningfork/` still need no token: anything that reaches the port can read the recordings, as
  `SECURITY.md` says.
- **Deleting for good takes the folders tuningfork made for a run's launches** (#60): in the
  recordings folder, and wherever else tuningfork marked one as made for it. A copy of the
  recordings folder never removes the original's. A batch's folder, or a set of forks', goes with
  the skill copy its runs were given once none of its runs is left and nothing else is in it. A
  path that a skill added, a scenario or a remaining run's task file names stays where it is, and
  the deletion's preview says so. What deleted rows held is overwritten, and the database's log
  emptied once no other process is reading it. What nothing refers to any more but was written in
  the last two minutes, which a recording in flight may still need, goes once those have passed:
  a running proxy sweeps it, or the next deletion does. A job keeps its record of the work, and a
  chain its record and the skills its links left, as before.
- **`tuningfork scan`** (#61) reads a recordings folder, only when you run it, for where today's
  masking would mask something an older tuningfork kept in clear, run by run and never the key.
  It prints the `trash` commands that set those runs aside, and names what goes by hand. It
  changes no recording. The first time this version opens a folder, it notes when; after that,
  `serve` and `ui` say in a line when the folder holds runs or jobs recorded before, until a scan
  has read it or those are gone.
- **Every password of a curl, MySQL or `docker login` command is masked** (#64), not only the
  first after the command's name: curl's `-u` and `--user`, MySQL's `-p` (`mysql`, `mysqldump`,
  `mysqladmin`, `mariadb`) and `docker login -p`. A MySQL or `docker login` password in quotes
  (`-p'…'`) is masked as one without them is, and so is a password whose flag starts a line that a
  `\` continued, with LF or CRLF line ends. Forks of runs recorded before still match their
  recordings.
- **A server's address is masked in its query and its fragment** (#64), as its user and password
  already were: each value of its query, and its fragment, whole, and a key the patterns read in
  its path. This holds wherever the address is kept or shown: the store, the Connections page and
  its restart command, `serve`'s first lines, the scoring and sampling lines, a protected
  endpoint's refusal, and the profile `tuningfork probe` writes to be shared, its errors included.
  `tuningfork scan` reads an address kept before the same way.
- **What a recordings folder holds** (#63): a section of `docs/setup.md` says what each part of the
  folder holds, what is masked and what is kept as it was, who can read it, and how to share part
  of one: copy it, delete from the copy with nothing else open on it, and remove the copy's launch
  folders and keys.

Changed:

- **The Connections page's restart hint** says a value in the upstream's query shows as
  `<redacted>`, as its user and password do.
- **`tuningfork probe`'s profile keeps its address masked**, where it kept a user and password as
  given.

A recordings folder made by an older tuningfork may hold what that version kept in clear: run
`tuningfork scan` on it, or on a copy of it.

## 0.7.0 — OpenClaw, and what an agent is given, masked

tuningfork launches OpenClaw now, beside Hermes Agent and Claude Code. Every answer an agent gets
over MCP, and what a proposing model is sent, is masked as it leaves. A check's output is masked
where the check runs, and the files tuningfork makes from what it recorded as they are written. A
validation's runs are set up as its checkpoint's run was. Making a scenario, the divergence page
and a forking-token probe's page were reworked.

- **OpenClaw** (`--harness openclaw`, #50) is launched for batches, forks and validations, as
  Hermes Agent and Claude Code are: each run is `openclaw agent exec` in a fresh home of its own,
  with no Gateway. It was checked with OpenClaw 2026.10.7, which runs on Node 24.22.1 or later in
  24, or 26.12.1 or later. Where it finds no Node it runs on, the Connections page says it cannot
  run here, and why, as `tuningfork connections show` does. A batch or a fork is then refused
  before it starts. A launch is OpenClaw as shipped, Tool Search and the Skill Workshop on, but
  for the memory flush, which is off. Its harness options are `tool_search`, `skill_workshop`,
  `memory_flush`, `mcp_servers` (tuningfork's own tools, say) and `context_window`.
- **What to allow for with OpenClaw** is on its page, `docs/adapters/openclaw.md`, with what its
  recordings showed. OpenClaw asks the model to begin a reply with `[[reply_to_current]]`, and
  Gemma 4 31B mostly did: a reply check reads the reply as the model wrote it. OpenClaw counts
  tokens its own way, so a conversation can outgrow the server's window: the server refuses the
  call, and the run is left out of a batch's pass rate, a fork out of its arm's count to the end.
  A task's `max_turns` does not reach OpenClaw. A conversation it carries on after a compaction is
  a thread of its own, so the final reply of a run that compacts is the main thread's last answer.
- **Not done yet with OpenClaw:** a subagent, which OpenClaw refuses without its Gateway; a chain
  told its check's result; importing its sessions. A scenario made from an OpenClaw run is
  untried.
- **A harness adapter can say three things more**, as OpenClaw's does: what a launch is told of
  the server it talks to (`for_server`), what in its `--version` stops it launching
  (`version_problem`), and which side calls the conversation goes on from
  (`fed_back_side_calls`). `docs/adapters/TEMPLATE.md` lists them.
- **Every answer an agent gets over MCP is masked as it leaves**, every refusal too. Before,
  `show_files`, `show_skill`, `show_run`, `list_tasks`, `list_scenarios`, `show_refinement`,
  `show_experiment` and `propose_edit` could show a run's workspace, a task's check, a scenario's
  task file, the skill a run was given or a wording tried, unmasked, and `where_it_holds` matched
  its pattern against the files as they are.
- **A text an agent or a proposing model is shown is masked whole before it is split or cut
  short:** `show_skill`'s numbered lines, a file read page by page, a task's prompt in a plan or a
  packet. Masked line by line, each line of a private key reads as no key, and cut short, a key
  leaves a part no pattern reads. Each line keeps its number, and each line a mask covers reads
  `<redacted>`.
- **Two texts an agent or a proposing model is shown side by side are masked alike:** a file's
  diff in `show_files`, now of its two texts rather than git's patch, an A/B's wording, and a
  refinement's proposals. What they share is masked as either masks it. A value either masks, of
  four characters or more with a letter among them, is masked where it stands on a line that
  changed, as a word of its own or glued as its mask was (mysql's `-pZ9q4`). An edit can take away
  what makes a key read as one: a private key's first line, or the option before a password.
- **What a proposing model is sent is masked:** the evidence packet as it is built, and the skill
  beside it. The proposer keeps its prompt and its answer masked with the recordings folder's own
  patterns.
- **An edit is resolved against the skill's file, never against what was shown of it.** One that
  would write `<redacted>` into the skill, or change part of a masked key, is refused: a key
  changes whole or not at all. So are edits that change the words a key's pattern reads beside
  it, such as `password=`: they would leave the key to read as none, and every run given the
  proposal would record the key bare. `show_skill` says these rules wherever it shows a mask.
- **A variant given as text that reads `<redacted>` more often than the skill's file does**,
  written from the skill as shown, is set on the file as it stands, each line it kept the file's.
  One that keeps part of a key, still reads the mask, or leaves a key to read as none is refused.
  Any other text, a person's, is not set on the file.
- **A check's output is masked where the check runs**, before it is cut to its last 2,000
  characters, kept, or told to a harness. Cut first, a key the cut fell inside kept its end, which
  no pattern reads alone. It is masked every way the store reads a text. The values of the
  variables the check gets whose names are a secret's, and tuningfork's own keys, are masked
  wherever they stand in it, each of eight characters or more.
- **The files tuningfork makes from what it recorded are masked** as the store masks what it
  records: an imported chat's `chat.json` and each `SKILL.md` taken from it, what a scenario or an
  added skill copies of an import, and a chain's record of the lines each link changed in its
  skill. An import made before is masked as it is read, wherever a scenario, a skill added or a
  refinement is made of it, or its skill is shown.
- **Making a scenario** names the skill the same way from a run's *Make a scenario…* and from the
  Scenarios page: the scenario's name, the harness (of a run), the skill's name, then its folder,
  each greyed where the run says it. A folder changed since the run, by an approval say, is
  offered beside a copy of what the run had (`tuningfork scenario --copy-skill`).
- **A scenario's page says whether it is saved**, beside *Save*, and input files go in once it
  is, each at a path you can change.
- **The Scenarios page's *Add a task file you have*** takes the task file's folder uploaded, as
  well as its path, and plans what it will write before it writes anything.
- **Where the runs diverge** goes on from the first split to any step: a button for each step,
  shaded where the runs split, the arrow keys, and an address naming the step. Each step says in a
  line what the runs did there.
- **A forking-token probe's page leads with the forced tokens that changed what the model did**,
  each beside the run's own token there, and says which way each moved what was counted. *Every
  continuation* shows what each continuation wrote from its forced token on, its reasoning and
  then its call or reply, by token or by behaviour (#29). `tuningfork experiment --continuations`
  prints them, narrowed with `--at`, `--token` and `--behaviour`, and `tuningfork tokens --sample`
  opens one as tokens. MCP's `show_experiment` pages through them with `continuations`.
- **A validation's runs are set up as its checkpoint's run was** (#51): its fresh runs, as well as
  its forks, take that run's harness options, but for its MCP servers, and its clock, sandbox and
  layout. A refinement of a Hermes run launched with `-o self_improvement=false` was validated on
  fresh runs where the model could patch the skill under test. The plan says the options and the
  clock where there is something to say, and names the MCP servers left out. A screen run to the
  end pools only checkpoints of runs set up alike.
- **What every check on a machine reads can be named once:** a `"check_env"` list in the machine's
  config (`~/.config/tuningfork/config.json`, beside `"protected"`) or in a recordings folder's
  `config.json` adds to what each task names, a proxy's settings, say. A batch's plan, `tuningfork
  connections show` and the Connections page say which file names what. A list that cannot be
  used is refused before anything is launched, naming its file.

Changed:

- **MCP answers carry no field named `sha256` or ending in `_sha256`.**
- **`where_it_holds`, asked by an agent, matches a file as the agent is shown it**, masked. Asked
  from the CLI or the web UI, it matches the file itself.
- **A run's page, its timeline and the task list show its task's check masked.** The form that
  makes a scenario of the run takes the check as written, since the scenario copies it and tries it.
- **`recheck` and `check-along` print a check's output masked**, and a chain run with `--tell`
  tells its harness the masked output.
- **Making a scenario of an imported chat takes the conversation masked**, as making one of any
  other run already did: its runs see `<redacted>` where the chat had a key.
- **A screen run to the end passes over a run set up otherwise** as it chooses its checkpoints,
  and says how the run differs. A later screen chooses again where the latest drew from one.
- **A forking-token probe's pages say *less often* and *more often***, where they said a forced
  token flips away or toward what was counted.
- **A result's target is headed *What was counted***, where it said *What you count*, on a line
  of its own above the verdict.
- **A skill's folder is labelled *Skill folder*** in Make a scenario and in the Skills page's
  *Add a skill…*, where they said *Its folder*.
- **Make a scenario sends a harness only where the run recorded none.** A run through a harness
  the form does not list was sent as Hermes.
- **A task file kept as a scenario brings what its check reads from its folder**, and the
  conversation its runs continue.
- **The Skills page finds a skill's folder from its latest launched run**, so a chat imported
  after no longer hides it.
- **An import whose skills' names make one folder** gives each its own (`tally-report`,
  `tally-report-2`). The import failed before.

A validation run before this release launched its fresh runs with the harness's default options,
and with the clock, sandbox and layout of wherever it ran: validate again to have them set up as
its checkpoint's run was, but for its MCP servers.

## 0.6.1 — a task's check runs in its run's sandbox

A task's shell check can run what the model left in the workspace: a `conftest.py` its `pytest`
finds, a module its `python -m` imports. It ran on the machine itself, with your environment.
Where bubblewrap runs, it now runs in its run's sandbox, and everywhere it gets only a few of your
variables. It still runs as you, as the run does.

- **A check runs in its run's sandbox** where bubblewrap runs. It works in `/home/user/project`,
  finds the final reply at `/home/user/reply.txt`, and has an empty home of its own. The rest of
  the machine is read-only, and its `/tmp` and processes are its own. The recordings folder, the
  runs' folders and the folders `TUNINGFORK_SANDBOX_HIDE` names are empty. Its task's folder shows
  through, read-only, and so does a venv on its `PATH` that a hidden folder holds.
- **A check's environment is made from scratch**, with the sandbox or without it. It gets the
  variables a launch gets: `PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`, `TMPDIR` and `USER`. It also
  gets those its task names in `check_env`, a new setting beside `check`, such as an LLM judge's
  key: `"check_env": ["OPENAI_API_KEY"]`. tuningfork's own key variables (`TUNINGFORK_TOKEN`,
  `TUNINGFORK_MCP_TOKEN`, `TUNINGFORK_PROPOSER_KEY`) never reach it, and a task whose `check_env`
  names one is refused. `$TUNINGFORK_REPLY` and `$TUNINGFORK_SPEC_DIR` are given as before.
- **A check tried runs the same way**, and so does one run along a run: `tuningfork recheck`, the
  web UI's *Test this check…*, `check-along`, and the same through MCP. Each hides what its run's
  launch hid, as the launch recorded it. A launch now also records which of those folders were
  tuningfork's own.
- **A fork is kept from what its source was kept from**, its launch and its check alike.
- **Where there is no sandbox, it is said.** A batch's plan, `recheck`, `check-along` and *Test
  this check…* say the check runs on the machine itself, unless you turned the sandbox off;
  Connections says it either way.

Changed:

- **A check that read your environment** gets a variable only by name now: list what it reads
  in `check_env`, a proxy's settings included (`https_proxy`, `SSL_CERT_FILE`).
- **In the sandbox, a check's home is an empty one of its own**, so the tools it runs find none
  of the settings and caches in yours (`~/.gitconfig`, pip's and uv's, `~/.netrc`, a model
  cache), though your home stays readable where it is. `uv run --with …` downloads what it
  installs. A check cannot read its run's home, where the harness kept its files.
- **In the sandbox, a check writes nowhere but its workspace, its home and its `/tmp`.**
- **In the sandbox, a check sees its own task's folder, and no other folder
  `TUNINGFORK_SANDBOX_HIDE` names.** Keep its scripts and the files they read in its task's
  folder. A batch's plan warns of a task file whose check could not see what is kept beside it,
  such as one directly in `/tmp`.

Runs launched before keep the task they were given, which names no `check_env`: their forks, and
a check tried or run along them, get only the few variables a launch gets, none that the task
names now. A task that names `check_env` now is a changed task, so a refinement begun from an
earlier run of it is refused its validation: refine again from a run of the task as it is now. A
screen run to the end counts the original's forks drawn before, judged as 0.6.0 judged them:
`--fresh-original`, or *Draw the original's forks anew*, draws them again.

## 0.6.0 — on PyPI, a demo without a GPU, and redaction reworked

tuningfork is on PyPI as `tuningfork-lab`, and `tuningfork demo` shows the whole loop in the web
UI with no GPU and no model. The README is shorter. Redaction, which masks keys in what tuningfork
records, was reworked: it masks keys rather than code, never breaks the JSON it masks, and knows
more formats, and the recordings folder's store masks everything it keeps, once.

- **On PyPI as `tuningfork-lab`.** `uv tool install "tuningfork-lab[mcp]"` installs the
  `tuningfork` command, its MCP server included; without `[mcp]`, the rest. The import stays
  `tuningfork`. The name `tuningfork` is too close to an existing project's to register.
- **`tuningfork demo`** opens the web UI on made-up runs, which tuningfork's own proxy serves in
  front of a scripted model, so every button works without a GPU:
  `uvx --from tuningfork-lab tuningfork demo`. It records only into a folder of its own, never
  into `~/.tuningfork`.
- **The README** says what tuningfork is, why, and how to start: the demo first, then your own
  agent. Its command tables lead to `docs/commands.md`, and the web UI's pictures moved to
  `docs/web-ui.md`.
- **Redaction masks keys, not code.** A value under a secret's name is masked only where it reads
  as a key: a name, a call, a path or a number is kept. Hermes's own source, read as tool results,
  had 731 masks in 147 of its 877 files, and has 2 now, both example keys. Its dependencies went
  from 273 masks to 7, and Python's standard library from 27 to none. A mask still costs a step
  its scoring and its forking tokens, and a resample sends `<redacted>` where the model saw the
  text: a plan says so now.
- **More formats are masked by their shape:** provider tokens (`gsk_`, `hf_`, `glpat-`,
  `github_pat_`, `AIza…` and more), Stripe's keys, JWTs, Discord and Telegram bot tokens, private
  keys, a URL's password, HTTP Basic credentials, AWS's and Azure's secret keys, and Slack
  webhooks. Of 47 key formats in 12 places a key stands, 537 are masked, against 258.
- **A mask never breaks JSON.** A number is never masked, and of 30,000 fuzzed payloads none is
  dropped, against 2,459. JSON written inside a string stays JSON.
- **A key the model streams in pieces** is no longer kept in the raw bytes beside its answer,
  which a fork read back and gave its harness again. Bytes that hold a secret are written again
  from the masked answer, and a fork refuses to replay that step.
- **The store masks everything it keeps, once.** A check's output in a run's result and a
  launch's exit, a job's plan and result, an imported chat's steps, a step's error text, a label
  queued for the next run, and an upstream's password were kept in clear. A value masked in one form of a
  step is masked in its others, so a server's error that quotes a password is masked, and so is a
  key the model writes in its own call syntax, in its tokens. An answer sent at once keeps no
  logprobs that spell a key, and a stream labelled otherwise is read as a stream. What a run is
  given is kept as given: its task, a wording tried, a scenario's conversation.

Changed:

- **`/_tuningfork/health`, Connections and `serve`** show an upstream's user and password as
  `<redacted>`, and Connections says to type them back into its restart command.
- **A step's parameters** count its masks (`masked`) and name the patterns that made them
  (`redaction`), which `tuningfork step` prints.
- **`Settings.redaction_patterns` is gone:** a `Store` holds its patterns.

Recordings made before keep what they stored: masking them again with today's rules is planned. A
fork of a run recorded before still matches its calls, each mask standing for what it hid.

## 0.5.0 — Qwen models, token by token

Token-level work — scoring an answer, forking tokens, a resample's kept tokens marked by part,
the model's own odds of each tool — reads Qwen3.5 and 3.6, Qwen3, Qwen2.5 and Qwen3-Coder now,
beside Gemma 4, each as llama.cpp reads it. A Claude Code step is scored on the tokens the model
wrote, and a recorded request or answer no longer keeps a key on a line of its own in clear. The
README opens in plainer words.

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
  where a newline is a backslash and an `n`: an `sk-` key or a `Bearer` token on a line of its
  own, or after a tab, then followed a letter, and was stored in clear in every harness's
  recordings. Nor did an assignment whose value is in double quotes, such as `api_key = "…"` or
  `"token": "…"`, match inside a message, a call's arguments or a tool's result, where JSON
  escapes every quote. The patterns now also run over the text with its JSON escapes read, three
  levels deep: a payload, the arguments or JSON result in it, and a JSON file that result shows.
  A key with an escape inside it is masked whole, in a Claude Code call's arguments too. A token
  stream reads the model's own escapes the same way. Recordings made before keep what they
  stored, and a streamed answer's raw bytes, kept beside it, can still hold a key the server sent
  in pieces.
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
