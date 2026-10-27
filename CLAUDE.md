# Tuning Fork

A step debugger and experiment lab for agent skills on local LLMs. What belongs to one
machine — its safety rules, its paths, its notes — goes in an untracked `CLAUDE.local.md`
beside this file (ignored by `.gitignore`), never here: this file is public.

Experiments run against llama.cpp's `llama-server`, by default on port 8081, with one slot
(`-np 1`): a request computed beside another gets other numbers. For Gemma 4 (or any model
with sliding-window layers) start it with `--ctx-checkpoints 0` or with `--swa-full`: without
one of them the server takes rollback checkpoints of the sliding-window cache that cut a prompt
into batches at places that differ between the chat endpoint and the raw one, so the same
answer scores differently on each; with either, the two agree token for token. `--swa-full`
costs memory (the 31B at 64k does not fit a 32 GB card with it); checkpoints off costs none,
and a checkpoint is only ever used to roll back a cached prefix, which `--no-cache-prompt`
never keeps. Qwen3.5 and 3.6, whose linear-attention layers keep a running state, need
`--ctx-checkpoints 0` as much (`--swa-full` does nothing for them): with checkpoints, the raw
endpoint's greedy answer to a 20,000-token prompt parted from the chat endpoint's at its 45th
token; without, scoring gives the chat endpoint's numbers exactly. The server takes no
checkpoints of a model with neither kind of layer (Qwen3, Qwen2.5): the flag changes nothing
there. Start it with
`--no-cache-prompt` too: otherwise a Claude Code step's
samples share the first one's cached processing (its Messages endpoint ignores
`cache_prompt`), and so do forks with matching prompts — every fork, in the sandbox. The
server check measures it (`fresh_prompts`).

## Commands

    uv run pytest                # the test suite — no GPU or model required
    uv run tuningfork serve --upstream http://127.0.0.1:8081   # the recording proxy; prints the UI address
    uv run tuningfork serve ... --max-output 8192              # every call a harness makes held to that many
                                                               #   output tokens, what it asked kept with the step
    uv run tuningfork probe <base-url> [--out profile.json]    # what a server can do
    uv run tuningfork demo                                     # no GPU: made-up runs, a scripted model

Inspecting and experimenting (all take `--home <recordings dir>`):

    tuningfork runs                       # recorded runs (--called T, --failed: pick one to study)
    tuningfork show <run>                 # a run's steps and what the model did
    tuningfork timeline <run>             # the run as one conversation, each step in its place
    tuningfork step <run> <k>             # one step: exact context in, behaviour out
    tuningfork diff <runA> <runB>         # two runs side by side
    tuningfork divergence --task <id>     # first step where runs stopped agreeing
    tuningfork tally --task <id>          # what the failed runs' checks said, counted (--screen <id>: by arm)
    tuningfork check-along --task <id>    # the check at each state of each run: which call made each finding
    tuningfork where --task <id> --file F --contains RE   # one question of every run, lined up with the
                                          #   verdicts: what the runs wrote (or --match RE: what the model
                                          #   did, at every step), the call that made it in each run
                                          #   (--screen <id>: of a screen's forks, arm by arm)
    tuningfork batch <spec.json> -n 8     # run a task N times through the harness
    tuningfork batch ... --skill <file>   # the same, with <file> as SKILL.md: confirm an A/B
    tuningfork batch ... --concurrency 4  # runs at once; each is its own launch
    tuningfork batch ... --eval 2 --include-held-out   # a held-out eval: never the default
    tuningfork batch ... -o self_improvement=false   # a harness option (Hermes: self_improvement,
                                          #   await_review, review_every_turn, mcp_servers), recorded with each run
    tuningfork chain run <spec.json> -n 10 --tell -o await_review=true -o review_every_turn=true
                                          # runs in order, the skill each leaves given to the next:
                                          #   what the harness's own self-improvement makes of it;
                                          #   --tell: the check's result as the user's next turn;
                                          #   --budget-seconds N: stop once the links' model calls took
                                          #   N s; the link that could outrun it is cut to what is left
    tuningfork chain list | show <id> [--diff]   # each link: verdict, who changed the skill, the lines
                                          #   (the web UI's Chains page; GET /chains, /chains/{id})
    tuningfork spend --chain <id> | --job <id> | --run <id> | --experiment <id> | --task <id>
                                          # what work cost: calls, tokens, time, from the records
                                          #   (GET /spend; beside a run, a job and an experiment)
    tuningfork tasks                      # pass rates: one per harness, model, skill, server
    tuningfork resample <run> <k> -n 20   # re-run step k N times, history held fixed
    tuningfork resample ... --logprobs 8  # ... keeping each sample's tokens (chat-completions steps)
    tuningfork ab <run> <k> --variant F   # swap an edited skill in at step k, test the shift
    tuningfork resample/ab ... --to-end   # each sample a fork run to the end; the check decides
    tuningfork tokens <run> <k> --score   # the answer as tokens: how sure the model was of each
    tuningfork forking-tokens <run> <k>   # force alternatives at uncertain tokens: which flip it
    tuningfork fork <run> <k>             # replay a run up to k, live from there (breakpoints on)
    tuningfork files <run> <k|start|end>  # the workspace as a call found it (--diff: since the step
                                          #   before; at the end, since the start; --out)
    tuningfork recheck <run> [--check …]  # try a check on a run's end state; records nothing
    tuningfork import <chat.json>         # a conversation had elsewhere, as a run (a Hermes session
                                          #   file or export, a message list): the step tools work on it
    tuningfork scenario <run> <name>      # any run as a scenario: its conversation continued from a
                                          #   user turn, its workspace there the inputs
    tuningfork experiments / experiment   # list, or re-report without new samples
    tuningfork servers                    # which servers answered, and what each was
    tuningfork skills / skill <name>      # each version of a skill: text, pass rates (--text, --diff A B)
    tuningfork add-skill <folder> [--copy] | --run <run> [--skill S]   # a skill added: its folder
                                          #   remembered where it is, or a copy of tuningfork's own
    tuningfork forget-skill <name>        # its record dropped; a copy moved aside, never deleted
    tuningfork glossary [word]            # tuningfork's words, each defined once
    tuningfork trash run|experiment|job|scenario|refinement <id>   # set aside: out of every list and rate
    tuningfork trash list | restore <id> | purge <id> | empty   # deletion is from the trash, for good

Moving something to the trash changes a mark and nothing else: it leaves every list and every
statistic (`Store.runs`, `experiments`, `jobs` leave it out), what is tied to it stays where it
is, and no new work starts on a run there (`ops.trash.refuse_trashed`). Deleting from the
trash is permanent: the item goes with what is tied to it (`Store.purge_closure`: a run's
threads, the experiments at its steps and their samples, forks tuningfork launched from it,
token streams, launches and their folders), then what nothing left refers to — message nodes,
blobs, snapshot objects — with a two-minute grace for a recording in flight. A refinement goes
with what it made (`refine.made_by`): its screens, its validations' forks and fresh training
runs, its proposals' saved texts and its folder — never the run it studies, nor the changelog
and old versions an approval wrote beside the skill's folder. One in the trash is shown as it
was and takes no new work (`refine.refuse_closed`). A fork an experiment drew goes to the
trash, and is deleted, only with its experiment, as a thread goes with its run
(`trash.experiment_of_fork`): its verdict is the experiment's, kept in its sample, and taking
single forks out of a finished screen would choose which of them count.

A conversation an agent had elsewhere is imported as a run (`ops/imports.py`; `tuningfork
import`, the Runs page's Import a chat…, `POST /imports`): a Hermes session file, its export, or
an OpenAI-style message list. Each assistant turn is a step whose request is rebuilt from the
history as its harness sends one — the system prompt, the messages before it, the tools where the
file names them; an earlier turn's reasoning is never sent back (Hermes sends none to a server
that does not ask); each message as the harness changes its own copy before every call
(`HarnessAdapter.as_sent`: Hermes strips each one and writes tool-call arguments compactly, keys
sorted) — so the step tools work on it. What a history never kept stays out: the block Hermes
puts after the system prompt at every call when it answers on a messaging platform (the chat,
the user's name), and the tools, which its export leaves out. Every skill it shows is found where
its harness shows skills (`skill_spans`) and kept beside the run, `<home>/imports/<run>/skills/`
(a folder the person names for it, or the SKILL.md the model read, extracted; `parity.skill_dirs`).
The one an A/B, a refinement or a scenario studies is named when it starts (`skill`,
`experiments.study_skill`, refused where several were kept and none is named), unless the import
named one or kept only one: the run's `parity.skill_snapshot_dir` points at that, as a batch
run's does. The conversation is kept there too, masked as its steps are, and each SKILL.md
extracted is taken from it so (`Store.keep_file`); a folder the person names is kept as given.
Its steps are recorded under an assignment of the run's own, never a pending label a live call
could take; nothing launched it, so it forks nowhere, and deleting it for good takes its folder.

Any run becomes a scenario — imported, launched, or recorded as it happened, never a thread
(`imports.prepare_scenario`; `tuningfork scenario <run> <name>`, `import --scenario`, a run's page's
Make a scenario…, the Scenarios page's From a run…, `POST /runs/{run}/scenario`): its conversation
is an import's kept chat, else the last step's request as the model was sent it and the answer
(`imports.chat_of`; a recording's steps numbered by its calls' depths, `Chat.numbers`, since a
scenario's run carried turns that are none of its steps); the user turn the skill first came in
after is the eval's prompt, in the person's own words (a harness's reminders beside them left out)
or theirs (`--prompt`), and the conversation before it — as its harness sends history — the
scenario's `context.json` (`tuningfork.context`, checked by `specs.check_context`). Another turn
(`--cut`) only where the skill could still come in live (`Start.starts`): it never came in, or came
preloaded — a run starting before its turn is never sent that turn, one after has it in its
context, where a variant never reaches it. The form sets each run beside the chat: what it is
given — its harness's own system prompt and tools, the chat's shown and never sent
(`scenario_choices`' `system`, `tools`), the conversation before the prompt, the prompt — then the
steps from the prompt on, each by what a run does with it (`Start.steps`, `STEP_PARTS`: already
done, where each run starts, made fresh by its model, past where it stops, after the person's next turn).
The skill is the folder the person names, the import named or a launched run's task's (an approval
writes there), else a copy in the scenario's own folder (`skill/<name>/`: the copy the run kept, or
the SKILL.md the model read, given frontmatter where its harness showed none) — never the run's own
copy, which its A/Bs read. The check is the person's, else a launched run's task's own, what it reads
from the task's folder (`$TUNINGFORK_SPEC_DIR/…`) copied beside the spec; the inputs are a launched
run's workspace as the scenario's first call stands in it, from the snapshots, else the files the
conversation's calls name are listed for the person to add; the eval is named by the scenario, a
task of its own, never counted with the run's. Each run continues that conversation: the runner
keeps it as a blob named in the launch's `settings.context` (kept live by `Store.live_blobs`; a
fork carries its source's, never
the file as it reads now), and the proxy puts it after the harness's instructions in every call
of the main thread (`WireFormat.with_context`; OpenAI chat only, a batch refusing a harness whose
wire cannot carry it, a call refused rather than sent without it), recording the harness's own
request beside the model's and a step's `context` note — never an edit, so the run still counts.
A subagent's calls and the harness's side calls are its own: they get no context. A scenario
studies one skill. When the chat loaded it later in the prompt's own turn, the model's steps of
that turn before it — an earlier skill's load and results among them — go to each run as done,
right after the prompt (`Context.in_turn`, `{"before": …, "in_turn": …}` in `context.json`), so
the skill itself still comes in live. When the chat went on to load another skill, each run
stops there (`tuningfork.stop_at_next_skill`): the proxy refuses the call after the answer that
loads a skill other than the spec's (`LaunchState.next_skill`, the adapter's `skill_load`) and
notes the launch (`Store.note_launch`), the runner ends the harness and runs the check on the
workspace as it is, and the run counts, its result saying where it stopped (`stopped_at_skill`).

A task file a person has becomes a scenario too (`scenarios.prepare_import`, `prepare_upload`,
`execute_import`; the Scenarios page's Add a task file you have, `POST /scenarios/import`), planned
before anything is written: by its path, its skill folder kept where it is, or its folder uploaded,
the skill folder it holds copied into the scenario's own (`skill/<name>/`). Either way its input
files, the conversation its runs continue, and what its check reads from its folder
(`$TUNINGFORK_SPEC_DIR/…`, read inside it as the shell reads it, `//` too) come with it; a
scenario whose check reads what its folder lacks says so before it runs. A form plans an upload by
its files' sizes, and makes it with the data of the files it takes alone (`scenarios.Unsent`, the
plan's `taken`). A new scenario's page and
Make a scenario name the skill alike (`ui/static/js/skillpick.js`): the scenario's name, the
harness (of a run), the skill's name, then its folder — an existing skill the Skills page lists, or
a path, a relative one shown after the scenario's own folder it is taken from, and of a run a copy
of what it had — each greyed where the run says it, but a folder whose SKILL.md has changed since
the run, offered beside a copy of what the run had (`copy_skill`). A scenario's input files go into
it only once it is saved, each at a path in the workspace the person can change.

A shell check runs in the run's workspace with `$TUNINGFORK_REPLY` (a file holding the final
reply) and `$TUNINGFORK_SPEC_DIR` (its spec's folder, where a check script can live out of the
agent's sight). It can run what the model wrote — a `conftest.py` its `pytest` finds — so its
environment is made from scratch, sandbox or not (`specs.check_environment`: a launch's few
variables, those its task names in `check_env`, and those the configs name for every check —
`"check_env"` in the machine's config, read strictly, and in the recordings folder's
`config.json`, read as its other settings are, once where a batch, fork or trial is planned,
`specs.check_env_sources` — never tuningfork's keys), and a run's check runs in the sandbox its
run ran in (`Sandbox.check_command`): the workspace as
`/home/user/project`, an empty home, the task's folder shown read-only (`Sandbox.shows`: never
one of tuningfork's own, `own`, nor one holding `/tmp` or `/home`), a Python environment on its
`PATH` that a hidden folder holds shown too, and what is hidden inside a shown folder hidden
still. A check tried (`ops/checks.py`) and one run along a run run in one like it wherever
bubblewrap runs, hiding what the run's launch hid. What a check prints is masked where it runs,
every way storage reads a text (`run_check`'s `masking`, `Masking.text`, `config.text_spans`),
before it is cut, kept or told to a harness — a reply check's reply before it is cut to its
quote; a long output from its end, `MASKED_MOST` characters, read on before it (`MASKED_MARGIN`,
`Masking.part`) so that a key where it starts, up to that long, is left out whole, and where check-along places none
of the run's findings — and so is every place, written as it is or with JSON's escapes, where the value of a
variable it got whose name is a secret's stands
(`config.secret_name`), or one of tuningfork's own keys (`access.held_secrets`), each eight
characters or more as a mask takes a value (`config._FOUND`). A run's result, its launch's exit
and its check along it keep it so, masked once. A check that prints what it found wrong, one finding a line, gets a tally
(`ops/tally.py`; `tuningfork tally`, MCP's tally_failures, the divergence and experiment pages):
each finding counted across a task's failed runs by condition, or a screen's failed forks by
arm, its numbers and colours masked and a run counted once, the most common first — the failure to fix
first, not the first one read. Run along a run (`ops.checks.check_along`; `tuningfork check-along`, the
`check-along` job kind; the benchmark's driver does it for the baseline before T starts) — on its
workspace as it started, after each call that changed it, and at its end, kept in the run's result and
never a verdict — a check that reads the files says when each finding came to be: there from the start
is work not done yet; made by a call is a mistake, made there (each line followed by its exact text:
masked, "not booked yet" and "booked twice" would be one finding). Each state keeps its findings whole
(`kept_findings`), never from its output's end alone, where a finding there from the start can fall
off and come back as made by a later call; a state with more than `MAX_FINDINGS` places none of its
run's findings. The tally then gives each finding
the step that made it in each run, the passing ones included, and names a mistake every run made and
some repaired later (`made_along`, `shared_mistakes`): comparing passing with failing runs points at the
repair there, never at the mistake (findings §31.42). A batch interrupted — Ctrl-C, or a SIGTERM, which
the runner turns into one — kills the harness it was waiting on: each runs in a session of its own.

Resampling sends straight to the inference server (`TUNINGFORK_UPSTREAM`, default :8081) —
never a protected endpoint (`endpoints.py`: none until a person lists one). An A/B must declare its target before sampling
(`--target <letter>` from an earlier resample, or `--target-match <regex>`); without one
the result is exploratory and Bonferroni-corrected. A letter target keeps its resample, and an
A/B, a screen or a probe of it letters its behaviours as that resample did
(`experiment.sampling.target_letters`): its B is their B. An A/B keeps its α with both arms too,
and is shown again judged at it — another only when asked (`--alpha`), and said
(`ops.inspect.declared_alpha`; one run before it was kept takes its job's). Run to the end, the target is the task's
check. A pattern or described target also reads what a behaviour says (`Outcome.text`): a
script's code, which its key keeps only as a syntax-tree digest; one stored before that (no
`code` in its spec) matches as it was declared. Every sample's prompt is processed whole where llama.cpp takes `cache_prompt: false` —
chat completions, not its Messages endpoint, which drops the field — so no sample depends on
what the server ran before (`--prompt-cache` reuses it). Each arm draws a random seed base,
recorded with it; `--seed-base` draws an earlier experiment's samples again. An A/B's report
starts with the edit it tested, a line diff of its two texts (`ops.experiments.ab_wording`;
`tuningfork experiment`, its page, MCP's show_experiment), read back from what it recorded — a
checkpoint A/B's arms' requests (`experiment.skill.swapped`, the inverse of `substitute`), a
fork's launch — and from the files it names only while each still holds the text tested.

Refining a skill at a step — the target first, then proposed edits, each screened, a survivor
validated with full runs, and approved by a person:

    tuningfork refine start <run> <k> --target-rules '[…]'    # or --target <letter>, --target-match
    tuningfork refine start <run> <k> --target-check          # the task's check: a failure only it can see
    tuningfork refine show <id> [--packet]                    # proposals as diffs, verdicts; the evidence
    tuningfork refine propose <id> --builtin [-m 3] [--no-task]   # the proposer's edits ...
    tuningfork refine propose <id> --file SKILL.md --why '…'      # ... or a person's own wording
    tuningfork refine screen <id> -n 20 [--proposals a,b]     # each against a fresh original arm
    tuningfork refine screen <id> --to-end [-n 20] [--checkpoints 4]   # the arms forks run to the end, the
                                          #   check deciding, drawn from the same step of several runs
                                          #   (--quick: 8 forks a proposal, no verdict; --fresh-original)
    tuningfork refine validate <id> <proposal> [-n 5 --runs 5] [--dry-run] [--keep-private DIR]
    tuningfork refine approve <id> <proposal> [--validation v] [--spec S | --target F] [--confirm CODE]
    tuningfork refine show <id> --repair                      # finish an interrupted approval
    tuningfork refine list | close <id>
    tuningfork connections proposer show | set …              # which model proposes; --key-stdin

A refinement (`ops/refine.py`) is a folder, `<home>/refinements/<id>/`, written once wherever
it can be: the checkpoint, the run's copy of the skill, its target, and every proposal. A
proposal is at most 3 edits (`ops/diff.py`: a line named by its number as show_skill prints
it and the line as it should read, lines added after one, or exact `{old, new}` text — each
resolved to an exact replacement of the text as shown, so none shifts another) changing at
most 10 lines — a person's whole text only warns — never the frontmatter, and it must change what the model
saw at the step; it is saved as a variant, with who proposed it, why, and the packet they
were shown (`ops/packet.py`: built from the run's own task's records only, never sampled
for, and masked as it is built). A screen (`ops/screen.py`; `analysis/screen.py`) draws a
fresh original arm first, then each proposal, and tests each against it — Fisher, Holm over
the proposals named before sampling; an unfinished screen has rates and no verdict. Run to the end
(`prepare_screen_to_end`, `refine screen --to-end`, the `screen-to-end` job kind), each
arm's samples are forks from the checkpoint through the harness and the task's check decides
each (`compare_verdicts`) — one that timed out failed, as the gate counts it, here and in an
A/B or a resample to the end (`analysis.conditions.fork_counted`): an edit that makes the
agent loop must not drop out of its own count. It is the only screen for a refinement whose
target is the check (`--target-check`: a failure only the check can see, a computed property
of what the run wrote), open to any other, and the record says which measured (`reach`). A
fork measures the edit given the prefix it replays — the model's answers and what its tools did
to the files — so an edit to a call before the step shows nothing there: a refinement toward
the check, and a screen's plan, say when a call before the step changed the workspace (`inspect.replay_warning`;
findings §31.42: a screen after a delivery was booked saw neither an edit's gain nor its harm).
And one prefix is one draw of the model's path, so the forks
are drawn from several checkpoints by default (`--checkpoints`, `choose_checkpoints`): the
refinement's own run and others of its task set up as it was (`ops.launch.Setup`: harness, model,
harness options, layout, sandbox and clock; a later screen chooses again where the latest drew
from a run set up otherwise) at the same step — each run's own such step when
the step was named by what the run did there (`ops.experiments.find_step`: the first call of a
tool, `write_file`, or `write_file|execute_code` for whichever came first; `changed:<path>`, the
call that first changed that file; `made:<words>`, the call that first made a finding of the
check run along it; `step_named` in the refinement), else the same index — passing and failing by turns,
each arm's n spread evenly, one experiment per arm per checkpoint in one group. The pooled
test is the verdict; the report says each arm's passes at each checkpoint, and whether the
effect held at every one (`across_checkpoints`), since one that lived in a single prefix is
one fresh runs draw anew. Run to the end, a refinement's original is forked once: its later
screens draw from the checkpoints its latest drew from (`earlier_checkpoints`), count the
original's forks its earlier screens drew there — ended, same harness, model and proxy
(`original_forks`, `original_reused` in the plan) — and draw only what their n still needs
(`--fresh-original` draws anew; findings §31.57: two screens of a broken skill spent forty forks
measuring its original at 0/20 twice). A quick look (`--quick`, `quick`) forks each proposal
`QUICK_N` times and the original not at all, its rates set beside the original's forks drawn
before: never a verdict (`ScreenReport.verdict`), a lead for the next edit. The built-in proposer
(`ops/proposer.py`) asks the proxy's own server through the proxy (gated, never recorded)
unless `config.json` names another endpoint; one that is not this machine needs
`allow_remote`, and its key is `TUNINGFORK_PROPOSER_KEY` or `<home>/.proposer-key` (0600),
never anywhere else. A held-out task's run never shapes an edit.

A held-out eval (`tuningfork.held_out` in the spec) judges edits, so no agent may read its
runs: a batch never picks one, and runs one named only with `include_held_out` (warned; never
from MCP). A validation (`ops/validate.py`, job kind `validate`, EXCLUDED from MCP) runs one
screened proposal three ways, every arm drawn fresh against the original: forks from the
checkpoint to the end, full runs of the training task with the edit from the first step, and
full runs of each held-out eval through a private proxy (`ops/private.py`) that records into
a 0700 temp folder outside the home and is deleted after — its calls reach the server through
a relay launch the main proxy forwards and never records (`LaunchState.records`), and only
verdicts leave. Every part is set up as the checkpoint's run was, as its forks are
(`ops.launch.Setup`, `prepare_batch`'s `setup`): its harness, model, harness options, layout,
sandbox and clock; an MCP server, recorded by name only, reaches none of them, and the plan
says so once.
It is refused when the skill's file, its other files or the training eval
changed since the checkpoint's run, and unless the proposal survived its latest complete
screen (`validate.latest_screen`: a screen of more proposals, or one run to the end, can undo
an earlier survival). Its record, `refinements/<id>/validations/<vid>.json`, is
what the gate reads (`analysis/validation.py`, G0–G8, each check with its counts and rule
name; G0 reads the latest screen again whenever the gate is judged; timeouts count as
failures, the not-worse checks are labelled low-powered, and a
held-out task the original fails must improve with the edit — G8, what caught a cheat). A job's
plan, result and progress carry no held-out verdict; `refine.view(held_out=True)` — the CLI,
the refinement's page — shows them, MCP's `show_refinement` never does.

Approving (`ops/approve.py`; the CLI, `POST …/approve` with the access token; never MCP) is
the only thing that writes a skill's own file. The target is worked out again from the run's
spec (a moved spec, or one leading elsewhere, needs `--spec`/`--target`), the gate must pass,
and the person types an 8-character code naming the proposal, the file and its bytes now.
Under a lock in `<skill>.history/`: `approval.json` written once with its evidence, the old
file kept in `<skill>.history/`, an atomic replace keeping the file's line endings (the one
most of its lines use, where it mixes them: the plan says so) and BOM, an entry in
`<skill>.changelog.md` — beside the skill folder, never inside it, which every
run copies — then `applied.json`. A crash between is reported by `refine show` and finished
by `--repair`, unless the file changed since. Deleting for good what a refinement cites says
so first (`refine.citations`); an approval keeps its evidence.

Token-level work needs llama.cpp's own endpoints (`/apply-template`, `/tokenize`,
`/completion`) and a model whose raw output tuningfork reads (`backends/formats/`: Gemma 4;
Qwen3.5 and 3.6, and Qwen3-Coder, whose template has no reasoning, read as llama.cpp's Qwen3-Coder
parser reads them — each argument typed by the tool's schema, `offered`; Qwen3 and Qwen2.5, read
as the parser llama.cpp builds from their templates reads them — a call's JSON kept as written).
The server reads the prompt's opening of
the turn with the answer, so a format is told what the prompt did with the reasoning (`thinking`:
opened it, closed it, or left it to the model) and reads it from the answer where no prompt is at
hand. Where the server reads several texts as one answer — an empty reasoning it drops, a blank
line between calls, Qwen3-Coder's first call written without its `<tool_call>` — a format gives
each (`renderings`): scoring keeps the one with the recorded
token count, and of several, the one the model finds likelier where they part, saying so
(`texts_alike`). Scoring rebuilds a recorded answer as the
model's tokens and forces it after the step's exact prompt in one request (a grammar of token
ids); the prompt must have the step's recorded token count, or it is refused. llama.cpp can give
a character the model wrote in two tokens one entry, the last token's (Qwen3.5's ` ≡`; Gemma 4's
byte tokens kept their own): the raw path gives each token its own again by the ids `/completion`
returns, the ones merged away unknown (`backends.llamacpp.raw.aligned`), where a check of the
entries' ids had failed every answer holding one. The forking-token probe forces tokens from a stored
stream, and tests each alternative against its position's baseline, Holm-corrected over the
family planned before sampling; each behaviour it reached says whether the target counts it and
which tokens led to it (`ForkingReport.led_by`): a token can change what the model does and miss
a target too fine to count it. What each continuation wrote from the token it forced on — its
reasoning, then its call or reply — is read back from the tokens kept with its sample, in the parts
the probe's format marks, read after the base's tokens before the position
(`ops.forking.continuations`: the probe's page, where every continuation is grouped by token or by
behaviour, a click apart (`continuationsSection` in `results.js`), `experiment --continuations`,
`tokens --sample`, MCP's show_experiment with `continuations`, page by page): what a forced token did
to the model's deliberation, not only where it ended. The server check (Connections) measures what each needs — the
raw completion, the prompt rendered as the chat endpoint renders it, the output format, and the
chat endpoint's own logprobs, which a resample's kept tokens need — and token work it found the
server without is refused before anything is sent (`ops.connections.token_work`, in the job
kinds' prepare: the web UI, MCP), greyed in the web UI with the same words; what was never
measured refuses nothing, and a resample that asked for tokens and got none says so in a note.
The proxy's own read of its server, every half minute, notes what its identity leaves out
(`recorder.ServerRead`: a key asked for — 401/403 — and llama.cpp's `/props`) and where it is
(`endpoints.on_internet`: every address public), in `server_read`. A server without `/props`
has scoring and forking greyed checked or not; one on the internet that asks for a key, or is
not llama.cpp's, is said to be a hosted model's API (`ops.connections.reminder`) wherever the
server is shown — serve's first lines, Connections and its switch, the status strip, the Guide,
`connections show`, MCP's `server_status`: tuningfork is built for a model run oneself, and
records a harness's calls through any server, its own key passed on. Only a launch is refused
where a key is asked for (`preflight` with the read): a harness tuningfork launches carries none.
`tests/test_real_server.py` re-measures what all this stands
on against a real server: `TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run pytest -m
real_server tests/test_real_server.py`; skipped otherwise.

A harness tuningfork starts is a *launch*: it talks to `/_launch/<token>/…` on the proxy, so
every call it makes — a subagent's included — is attributed to it as it arrives. It records the
process running it, as a job and an experiment do: one still marked running after that process
died — a batch killed hard says nothing of its launches — reads as interrupted
(`ops.inspect.launch_status`), holds nothing back from the trash, and the next proxy marks it
so (`JobRunner.sweep`). A call the harness adapter names a side call (`side_call`: Hermes's
background review, which replays the whole conversation after the answer) starts a side
thread, never the task's next step, and is never the run's final reply. A user message a
harness writes itself — Hermes's nudge to go on after a cut reply, Claude Code's reminders — its
adapter names (`harness_note`; for a run recorded as it happened, whichever adapter knows it):
a run's page puts it under the step it followed, and a person's turn on a line of its own, a
new exchange, in their own words (`words`: the harness's parts beside them left out). A run
nothing launched nor imported was recorded live (`Run.recorded`: `launched`, `imported`, `live`),
and says so where a label would name it — the Runs list, its page, `tuningfork runs` and `show`.
Its first call may continue a conversation already going on, a session of many turns: its rail
lists that conversation first, as its own exchanges — each person's turn, the model's answers
unnumbered, no call of the run's, and what came after each — then the steps it recorded. Each
person's turn folds there to its own line, saying what it holds; with three or more, all but the
last start folded, but the one holding the step the address opens, and a step revealed opens its
turn. A call
joins the run whose latest message its history holds (`Store._existing_run_for`); one that joins
none starts a run, marked a branch of the run whose history holds the most of its own — more than
its first message, which every session of one harness may share as its system prompt — at the
message where the two part (`Store._branch_of`, `forked_at_depth`): a mark that groups nothing. A fork is a launch
whose calls before step k are answered from the recording (`tuningfork.replay`); the proxy's
`--max-inflight` must be at least the `--concurrency` of forks run through it. A launch is
laid out as the harness installs itself (an adapter's layout "install": Hermes in
`~/.hermes`, a skill in the category folder it declares; Claude Code in `~/.claude`),
because the model reads those paths; each launch
records its layout, and a fork is laid out as its source was ("flat" before layouts were
recorded) and set up as it was (a harness option changes what the model is told: a fork
told otherwise departs from the recording at its first step; an MCP server, recorded by
name only, is not carried, and the plan warns). Where bubblewrap runs, every launch runs in a sandbox (`sandbox.py`): it sees
its home as `/home/user` and works in `/home/user/project`, the same in every run, because
a few digits of a launch's own folder name were seen to change what the model did; the
rest of the machine read-only, `/tmp` and its processes its own, and the recordings folder,
the runs' folder and the spec's folder empty inside; its check runs there too, the spec's
folder shown to it. A launch records its sandbox, a fork of
a sandboxed run is refused without one, and a batch's plan warns when there is none.
`TUNINGFORK_SANDBOX_HIDE` names folders every launch sees empty besides tuningfork's own, as
PATH is written — a benchmark's folder of tasks, whose checks and keys an agent given the file
system would otherwise read; a check sees them empty too, but for its own task's folder, where
its scripts and keys belong. `TUNINGFORK_SANDBOX=off` turns it off — the suite's `conftest.py` does, and
`tests/test_sandbox.py` turns it back on where bubblewrap can run. `TUNINGFORK_CLOCK=YYYY-MM-DD
HH:MM:SS` starts every launch's clock at that instant (libfaketime under `LD_PRELOAD`, monotonic
clocks left real): the harness reads the same date whichever day a run is made — the model
reads it, and at a knife edge it decides — a fork reads its source's, as does every run of a
validation, the launch records it, and a plan warns where the library is missing
(`TUNINGFORK_CLOCK_LIBRARY` names it elsewhere). The normalizer masks the sandbox's paths as it
masks a launch's own (`NORMALIZER_VERSION` 3).

Live breakpoints talk to a running proxy (`--proxy`, or `TUNINGFORK_PROXY`, default :8787):

    tuningfork break add --tool write_file --path '*.json'   # after the model: stop that call
    tuningfork break add --from-step 2                       # before the model sees step 2
    tuningfork break add --no-reasoning                      # after the model: a tool call, unreasoned
    tuningfork break list | rm | on | off <id>
    tuningfork hold wait | list | show <id> [--json]         # what is held, and why
    tuningfork hold continue <id> [--edited f.json]          # also: step, drop
    tuningfork pause / resume                                # refuse model calls meanwhile
    tuningfork ui --home <dir>                               # browse recordings, no proxy

Work started from the web UI runs as *jobs* inside `serve` (`jobs.py`), each on a worker
thread, its model calls sent through the proxy's own gate:

    tuningfork jobs list | show <id> | cancel <id> | wait <id>
    tuningfork serve --jobs N                                # background jobs at once (default 1)
    tuningfork token [--raw | --mcp | --header-json]         # the UI address with its token; the MCP token
    tuningfork connections show | switch <url> | harness <name> --executable P | check

A job keeps who asked for it (`started_by`, `jobs.asked_by` in `jobs.py`): an agent over MCP and
its client, or the web UI and the page it was started from (`startedFrom` in `ui/static/js/api.js`,
sent with every job it starts). The Jobs list is in three parts (`jobParts` in
`ui/static/js/views/jobs.js`) — running and queued (the running first, then the quick line, the
interactive lane's, and the background line, each in its order, every queued job saying its place
in its line), finished, and stopped or failed — and `JobRunner.list` gives every job under way,
however many came after it. The list names each job by the sidebar section of the work it does
(`KIND_SECTION` in `ui/static/js/jobs.js`) and, finished, what it made, leading there (`madeLink`,
the same file); a finished job's page hands off to what it made — it says what it found in a line
(`found` in `ui/static/js/results.js`: the verdict the experiment's page leads with, from the same
helpers, where that page has one), with what was counted, the rates, what was left out or is still
open in that page's words, and the runs it made, its log folded below — and the note when it ends
leads there. A question at a held call and a check tried keep their answers on the job's page,
their only home once the call is decided or the tester closed.

The same operations, as tools for an agent over MCP (`tuningfork.mcp`, the SDK an optional
extra, `tuningfork-lab[mcp]`; nothing else imports it):

    tuningfork mcp --home <dir> [--proxy URL] [--tools evidence|refine|all]   # stdio
    tuningfork serve ... --mcp-tools evidence|refine|all|none                 # also at /_tuningfork/mcp
    tuningfork serve ... --mcp-results text   # or mcp --results text: each answer once, for Hermes

Reads answer from the recordings in the MCP process; `start_*` tools are job kinds, sent to
the proxy recording into the same folder. Every job kind needs a decision in
`mcp/schema.py` — `EVIDENCE` (calls the model only), `HARNESS` (runs a harness or a check on
the machine: `--tools all`) or `EXCLUDED` with a reason — or `tests/test_mcp_schema.py`
fails. The default toolset is the evidence kinds but `screen`; `--tools refine` is the
refinement loop: start_refinement, show_refinement and propose_edit (which act in the MCP
process and call no model), resample, A/B and screen; `--tools refine-to-end` adds the screen
run to the end, a HARNESS kind. Every tool's description carries what an
agent must know, since a client may show the model no server instructions (Hermes shows none):
the skill is read with show_skill, never from a file; tally_failures counts every failed run's
check output at once, the most common failure first (never of a held-out task) — and says so when
the check printed only its verdict, where where_it_holds asks one question of every run instead
(`ops.where`: a target tried at every step of every run, or a file's pattern at each state of its
workspace, lined up with the verdicts; the call that made the difference, and the step that names it
in each run, for start_refinement; where no run of the task passed, of a screen's forks, arm by arm —
`where_in_screen`, and the answer names a screen whose forks passed, `passing_screens`) — and, where the
check was run along the runs, which call made each finding — a mistake every run made named as one,
with where to refine it (its call, toward the check); a failed run's check output (show_run: what
each step's calls ran, `inspect.ran_excerpt`, where a label shows a script only as its digest, and
the files each changed) and what it wrote (show_files: at the end, what the whole run changed since
it started, with a diff) are where to look — a long file, or a long message of a step (show_step's
`message`), read page by page from `offset` (`mcp.tools.page`, `FILE_PAGE`; findings §31.57: a cut
with no way past it hid the legend the check failed); the step to refine is the first whose call made what
the check complains about — never named as the last call, which sent an agent two steps past a
mistake (findings §31.42) — since a screen replays everything before its step, the files included
(start_refinement and the screen tools say so); a target is what the passing runs' answers
have that the failing runs' lack — the tool they call, when find_divergence says a tool goes
with passing (`ops.inspect.tool_split`: runs that called it against runs that never did, at
whatever step, since a run that scripted its file never called the write tool at all), or
words their answers contain — and step_behaviours tries one against the recorded answers
split by verdict (`ops.targets`: each behaviour with how the runs that took it fared) before a
screen is spent. A step named by what the run did there (`write_file`, or `write_file|execute_code`
for whichever came first; `changed:<path>`; `made:<words>`) is compared across runs at each run's
own such step; an index is the
same number, another point in a run that took another path. Where the recorded answers were
scored (`tuningfork tokens --score`; the benchmark's driver scores the baseline runs before
the agent starts, and runs the check along them), step_behaviours says the model's own chance of each tool at the step
(`analysis.tokens.tool_choice`: at the first token of the call's name, under the step's
sampler) — a choice near even is one a few words move; start_refinement refuses a target no recorded answer at the step satisfies
unless `unseen_ok` says the behaviour is new, one every recorded answer satisfies (five or
more seen), since a screen could only show it falling, and one the failing runs' answers
satisfy as readily as the passing runs', since moving it would not move the verdict, and says when
the checkpoint run passed its check, since a step screen there starts near the ceiling (a step screen
whose original arm reaches the target four times in five says so too); and
find_divergence also names the step whose behaviour best predicts the verdict, when one does
(`ops.inspect.verdict_split`). No agent is offered the built-in
proposer — sending recordings to an endpoint is the person's decision — nor anything that
approves an edit. An answer is a compact view (`ops/compact.py`), masked as it leaves the server
(`ToolServer.shown`), whose sentences lead both its text and its structured content (`summary`):
Claude Code shows the model only the structured content, OpenJarvis the text, and Hermes both side
by side — every answer twice, so for it `--mcp-results text` (`mcp --results text`) sends the text
alone (`mcp.server.RESULTS`). The same refusal three times running carries the tool's description
again, and from the sixth time says to stop sending that call and try another edit, step or
tool (`ToolServer._repeated`): a weak reader reads each as the first, and one told to stop
would take it for the task. Point a driving agent's own model at the inference
server, never at the proxy it studies.

A tool call made without reasoning is marked only where the same run reasons at another step
(`analysis/reasoning.py`), with whether a tool call had just failed — a reply in text without
reasoning is the usual end of a run, and is not; an experiment counts, per behaviour, the
samples that reasoned, said where some did not. The web UI's sections are a sidebar in the
loop's order (`index.html`): the Guide, the ways in (Scenarios, Import a chat, Live, Skills), then
Record, Measure, Improve, and the Trash; what needs a running proxy stays in it where none runs,
greyed and saying why, and its foot is a status strip — the proxy, the server and the harnesses,
from `GET /connections`. Its front page is `#/guide` — setup, where you left off, a map of the
loop and the ways in — and its words have a page of their own, `#/words`, each living once in
`ops/glossary.py`. Each page leads with its take-away: a run its verdict strip; the divergence
page the step whose behaviour best predicts the verdict, in a callout with the ways to it, then the
runs side by side one step at a time, from the first split, every step a click away (`atStep` in
`views/compare.js`: the first split is often not the step that decides); a forking-token probe the
forced tokens that changed what the model did, each beside the run's own token there — what each
led to, and how often it did what was counted (`flipsTable` in `results.js`);
Connections the setup at a glance — the server, the harnesses, the sandbox, the proposing model, a
word and a colour each, the server in the status strip's words (`serverWord`), and what that means
in at most three sentences (`whatItMeans`); a refinement its four stages in a line (`stagesOf` in
`views/refine.js`: propose, test at this step, full test runs, approve — where each proposal
stands is its latest finished test, as the gate's G0 reads it), then its evidence, each proposal
named by the line it changes. A run's page keeps the run to read and the work on one of its steps
beside it (`openWork` in `timeline.js`: a step's actions and the form they open, in a column
where the rail is — over the run, on a narrow page — each step's kept as it was left, the
address naming it, `?work` or the form's), the rail's lines moving through the run and opening
nothing, one step's details open at a time, from its line or its Show details. A refinement's
page goes on with a section for each stage, numbered as the stepper numbers it (the two ways to
propose side by side; the proposals with the test of the ticked ones), the current one's form the
one open as the page opens — after, no form opens by itself. The Skills page
(`ops/skills.py`) names a skill by its folder, as variants and refinements do, and runs it again
from the scenarios that name it; it is the person's view, never offered over MCP: the changelog
an approval writes carries held-out verdicts. A person adds
a skill there too (`ops/addskill.py`; `tuningfork add-skill`, Add a skill…, `POST
/skills/add`), planned before anything is written, as a record in `<home>/skills/<name>.json`:
a folder on this machine, remembered where it is, so an approval writes to it; or a copy of
tuningfork's own, `<home>/skills/<name>/` — a folder uploaded or copied in, or the skill a run
had (the copy it kept, else the SKILL.md its model read, given frontmatter) — which an approval
writes to, never back to where it came from. Forgetting one drops its record; a copy is moved
aside whole, with what an approval kept beside it, to `<home>/skills/.forgotten/` — never
deleted, and not while a scenario runs from it.

Every state change under `/_tuningfork/` needs the proxy's access token
(`x-tuningfork-token`, or `Authorization: Bearer`): minted each `serve`, written to
`<home>/.access-token` (0600), and checked in `api/guard.LocalOnly`, never in a route. Reads
stay open, except the MCP endpoint, which takes its own MCP token on every method
(`<home>/.mcp-token`, pinned by `TUNINGFORK_MCP_TOKEN`) and nothing else: the access token is
refused there, and the MCP token everywhere else, so an MCP client never holds a key to what
its tools leave out. The CLI finds both itself; tests send them
(`tests/helpers.control_headers`, `mcp_headers`). One proxy per home: two would overwrite
each other's tokens.

Every operation lives once, in `ops/`: a function with typed parameters, progress events,
cancellation, and a result that knows its JSON form. The CLI, the HTTP API behind the web UI
(`/_tuningfork/ui/`) and the MCP server (`mcp/`) are thin front ends over it, and every
function is meant to be reachable from each: logic a second front end would need does not
belong in `cli.py`, `api/` or `ui/`. The job kinds a front end can start are listed once,
in `ops/catalog.py`, with strict parameters (`GET /jobs/kinds` serves them). Only a call a
breakpoint could apply to leaves the proxy's ordinary streaming path, so nothing changes for
traffic while no breakpoint is armed — but a cap the proxy was started with: `--max-output N`
(`Settings.max_output`) holds every call a harness makes through it to N output tokens
(`wire.base.cap_output`, by the limits each format names, `output_limits`: a limit above N, or
none, is sent as N). The step records the request as the model was sent it, and what the
harness asked in its params (`output_cap`), from which `Store.harness_request` gives the
harness's own back — what a fork's replay matches against. A call a person edits at a
breakpoint is held to N again (the edit's own ask in the step's `edits`), and a validation's
held-out runs, which the main proxy relays unrecorded, are held to it by their private proxy
(`ValidatePlan.max_output`). Experiment traffic keeps its own limit. The cap belongs in the proxy because Hermes answers a reply cut at its own limit by asking
again for more (a tool call dropped and resent, text continued; four or five times, the limit
doubled each time): an agent's reply looped to Hermes's 65,536 tokens for 35 minutes of one GPU
(findings §31.58).

The test suite must always pass with no GPU and no model loaded. That is the contribution
gate for an outside PR, so a test that needs real inference belongs behind a marker, not in
the default run. Check pytest's **exit code**, not just the pass count: warnings are errors,
and an unclosed SQLite connection fails the run at teardown after every test has passed.

## Layout

    src/tuningfork/ops/        every operation, once: experiments, forks, batch, holds (tools at
                               a held call), scenarios, variants, connections, inspect, timeline
                               (a run as one conversation; a step's detail), checks (a check
                               tried before it is trusted; run along a run, which call made each
                               finding), tokens (an answer as tokens, scored),
                               forking (the forking-token probe), targets (what a target would
                               count among the behaviours seen at a step), tally (what the failed
                               runs' checks said, counted), where (one question of every run —
                               of its answers or its files — lined up with the verdicts: the call
                               that made the difference), trash (set aside,
                               restored, deleted for good), compact (results small enough for an
                               agent, intervals intact), refine (a skill refined at a step: its
                               target, proposals and screens), packet (the evidence a proposer
                               is shown), screen (several wordings against one baseline),
                               proposer (the built-in proposer: endpoint, consent, key), diff
                               (line diffs; exact edits), validate (a proposal's full runs:
                               forward, training, held out), private (held-out runs, verdicts
                               only, through a private proxy), approve (the gate met, the code
                               typed, the skill's file written back, history and changelog
                               beside it), skills (each skill's versions, texts and rates; its
                               folder now), addskill (a skill a person adds: its folder, or a
                               copy of tuningfork's own), glossary (the words, once), imports
                               (a conversation had elsewhere, as a run; any run as a scenario);
                               the job catalog; `common`: events, cancellation, strict parameters
    src/tuningfork/jobs.py     the job runner inside `serve`: lanes, progress, cancel, sweep
    src/tuningfork/access.py   the access token and the MCP token; endpoints.py: protected endpoints
    src/tuningfork/paths.py    paths as a person types them (a Windows path under WSL, relative
                               to what), read the way tuningfork will open them
    src/tuningfork/proxy/      recording proxy (app, sse, recorder, held: the breakpoint path;
                               launches: calls attributed on arrival; replay: a fork's answers;
                               local: the proxy as in-process work sees it)
    src/tuningfork/launches.py launch tokens and URLs, and a launch's threads
    src/tuningfork/sandbox.py  every launch in bubblewrap, seeing /home/user and /home/user/project
    src/tuningfork/replay.py   what a fork replays, masked requests, path rewriting
    src/tuningfork/snapshots.py workspace snapshots in the home's own git object store
    src/tuningfork/breakpoints/ conditions, the hold engine, the live event bus
    src/tuningfork/api/        HTTP API for the UI and CLI: control, data, jobs, connections,
                               scenarios, imports, trash, refinements (and the proposer's settings);
                               guard (Host, Origin, token)
    src/tuningfork/mcp/        the operations as MCP tools: schema (which kinds, how an agent
                               may call them), tools (the registry; reads), server (the SDK's
                               low-level Server: parsing, refusals, waits), backend (jobs in the
                               proxy, or over its API), stdio, http (the route on the proxy)
    src/tuningfork/control.py  Python client for a running proxy (what `break`/`hold` use)
    src/tuningfork/ui/static/  the web UI: plain HTML/CSS/JS modules, no build step; markdown.js
                               renders Markdown as DOM nodes, never as HTML; words.js, the pages'
                               plainer words for the glossary's terms
    src/tuningfork/wire/       wire formats: OpenAI chat-completions, Anthropic Messages; raw
                               completions (model load the proxy gates, never recorded)
    src/tuningfork/backends/   server capability profiles + runtime probe; raw.py, the raw path
                               (render, score, complete from token ids) and llamacpp/, its
                               llama.cpp implementation and the server's Messages conversion,
                               mirrored; formats/, model output formats (Gemma 4; Qwen3.5 and 3.6;
                               Qwen3 and Qwen2.5)
    src/tuningfork/harness/    harness adapters (hermes, claude-code, generic)
    src/tuningfork/store/      content-addressed blobs + SQLite message tree; token streams; masking
                               (what every write is masked as, once)
    src/tuningfork/analysis/   normalizers, Wilson/Fisher/Holm, distributions, divergence;
                               tokens (entropy, margin, the step's sampler, redaction), forking;
                               screen (each proposal against one baseline, Holm over the family);
                               validation (the gate a validated edit meets, check by check);
                               reasoning (a step the model answered without it)
    src/tuningfork/experiment/ resampling, skill substitution, A/B orchestration; scoring an
                               answer; the forking-token probe
    src/tuningfork/runner.py   batch runs and forks (fresh workspace + home per launch)
    src/tuningfork/specs.py    task specs (evals.json convention)
    src/tuningfork/demo/       `tuningfork demo`, shipped so that it runs with no checkout: lab
                               (made-up runs behind a real proxy); mock, the scriptable model
                               server (OpenAI chat-completions and Anthropic Messages); toy, a toy
                               llama.cpp (tokenizer, template, a model with forks; QwenToy, the
                               same writing Qwen3.5's format; Qwen3Toy, Qwen3's or Qwen2.5's);
                               agent, a real agent process, and task, its skill, its model and
                               the adapter that launches it. The test suite runs on the same
                               pieces; nothing else in tuningfork imports them
    tests/fixtures/formats/   raw output beside the server's parse of it; capture.py remakes them
    tests/browser/            opt-in browser tests (Edge/Chromium over DevTools), against the demo
    examples/                 a skill that mostly fails, and a step-by-step guide (README.md);
                               hermes-design-md/, Hermes's own design-md skill fixed from the
                               failing step to an approved edit, with its guide; find-the-file/,
                               a step where the model wavers: the runs behind the README's figure
    docs/                     what the README links to: setup.md (install, the server's flags and
                               why, the harnesses, the sandbox, a task's format), commands.md
                               (every command by the question it answers), agents.md (the MCP
                               server, client by client), web-ui.md (the pages, pictured);
                               adapters/, one page per harness; images/, the pictures (hero.py
                               draws the figure)
    scripts/                  tuningfork inside WSL from a checkout on Windows: tuningfork.ps1 for
                               PowerShell, tuningfork for Git Bash, twins — the distro only ever
                               the one TUNINGFORK_WSL_DISTRO names (tests/test_wsl_wrapper.py)

A request is never stored whole: messages live in the content-addressed tree and are
rebuilt by `Store.rebuild_request`. What the store keeps of what happened — steps, samples and
their tokens, jobs, runs' results, launches' exits, servers — it masks itself, once, with its one
set of patterns (`Store.masking`, `store/masking.py`): a writer hands it what it has — or, where it
must read a value as it will be stored first (an outcome computed from an answer), the `Masked`
it asked the store's masking for; a check's output and what a harness was told of it, masked
where the check ran, are fields of a run's result and of its launch's exit kept as they are
(`Masking.of`'s `masked`). A file the store keeps beside its rows, made from what was recorded,
it masks as it writes it (`Store.keep_file`): a chain's record — the lines each link changed in
the skill, of both texts masked whole and alike, each line keeping its number (`Masking.text`'s
`lines`), and what its harness was told and its skill-editing calls, masked already
(`chain.LINK_MASKED`), kept as they are; an imported chat, each answer also of what
the masks before it took, as a step's answer is (`imports.kept_chat`), and the SKILL.md of each
skill the model read, as the masked chat shows it; the import names the patterns
(`parity.imported.masked`). The chat of an import made before that is masked as it is read
(`imports.chat_of`), and so is its SKILL.md extracted, wherever a scenario, a skill added or a
refinement is made of it, or it is shown (`imports.extracted_unmasked`, `experiments.kept_text`). What
a run is given is kept as it was given, since a fork runs it again — a launch's eval and variant, a
scenario's context (`Store.put_given`) — and so is what a scenario, a skill added or a refinement
copies of it, the skill a launched run was given (a chain's link's, which the link before wrote,
too), and an experiment's settings, the test it declared. What leaves tuningfork is masked as it
leaves, whatever the store keeps (`Masking.shown`): every answer and refusal an agent is given over
MCP (`ToolServer.shown`), its model maybe hosted, and what a proposing model is sent — the packet,
masked as it is built and kept so (`Packet.keep`), and the skill beside it, its exchange kept as the
store keeps a file (`Store.keep_file`). A text a view splits or cuts short — show_skill's numbered
lines, show_files' pages, a task's prompt in a plan or the packet — is masked whole first, each line
at its number, each line a mask covers reading `<redacted>` (`Masking.shown`'s `lines`): line by
line, a private key's lines pass, and a cut through a key leaves a part no pattern reads. Texts set
side by side — a diff: show_files', an A/B's wording, a refinement's proposals — are masked alike,
what they share masked as either masks it, and a mask's value — four characters or more, never
digits alone — wherever it stands as a word of its own on any of their changed lines
(`Masking.alike`): an edit can take away a private key's first line, or the command a password's
pattern reads. A shorter mask, or one of digits alone — a PIN, a year, which masking by value would
point out among a text's numbers — is carried onto a changed line only where the lines around it
are compared character by character (`CHARS_COMPARED`). An edit's quotes — the text it replaced, the
lines a refusal finds closest — are the skill as show_skill shows it. A file where_it_holds asks of
for an agent is matched as the agent is shown it. A run's views show its task's check masked
(`timeline.launch_json`, `inspect.task_summaries`); the form that makes a scenario of the run takes
it as written, as the scenario copies and tries it. An edit is resolved against the skill's file,
never what was shown of it (`ops.diff.resolve_edits`' `shown` and `masks`): it never writes the mask
where the text it replaces held none, and changes a key whole or not at all; and a proposal that
leaves a key to read as none — the words its pattern reads changed beside it, `password=` — is
refused (`Masking.bared`): every run given it would record the key bare. A variant given as text that
reads the mask more often than the file does — written from the skill as shown — is set on the file
as it stands (`experiments.variant_onto`), each line it kept the file's, a key whole or not at all
and still masked; any other, a person's, is tested as given. No answer gives a digest (a field named
`sha256` or `*_sha256`) of what it shows masked (`mcp.server._without_digests`): against one, a short
key is guessed. Masking is not
idempotent, and a fork matches a call masked once against a step stored masked once;
`tests/test_store_masking.py` fails on code elsewhere that masks or writes a blob itself, and
`tests/test_canaries.py` plants a key in the loop's inputs and searches every row, blob and file
for it, and every answer an agent is given over MCP, but the files kept as they were
(`KEPT_AS_IT_WAS`: a launch's folders, the workspace's snapshots, a scenario's inputs, the
wordings tried), what a copy of the skill a launched run was given holds as given, or a
refinement's copy of it (`GIVEN_COPIES`, `GIVEN_TO_A_REFINEMENT`), and what a scenario's spec
takes from its run's task as written (`GIVEN_TO_A_SCENARIO`). Never report a rate without its
interval — see `analysis/stats.py`.

Server- and harness-specific knowledge lives behind the `backends/`, `harness/` and `wire/`
seams, never in the core. Supporting another harness or inference server must not require
editing anything under `proxy/`. A request is stored in its own wire format, exactly as
sent; responses are also kept in one common shape (an OpenAI `chat.completion`), which is
what outcomes, conditions and the UI read. There a streamed call's arguments are the text the server
sent, which scoring writes back as the model's tokens: Anthropic Messages keeps a tool input parsed, so
its text is taken from the stream (`canonical_response`'s `stream`). Features degrade on a probed capability rather
than on a hardcoded assumption about which server is running.

## Conventions

- Python 3.12+, managed with uv. Under WSL, keep the venv and the recordings on the Linux
  file system, never under `/mnt/c`: DrvFs breaks SQLite's WAL locking.
- Nothing of one machine in a tracked file: real paths, hostnames, ports that serve something
  else, notes. Those go in untracked `*.local.md` files. A screenshot is a tracked file too.
- The README is the front door, read by someone deciding in a few minutes whether to try
  tuningfork: what it does, a quickstart, the commands by question, links. Reference goes in
  `docs/`. A new command or option goes in `docs/commands.md` (and in the README's table if a
  newcomer needs it): `tests/test_docs.py` fails on a command it does not name, and on a link
  in those pages that leads nowhere. An output a guide shows that changes is captured again,
  never typed.
- The UI puts every string on the page as a text node — payloads are full of markup-looking
  text. `tests/test_ui.py` fails on any HTML-parsing sink (`innerHTML` and friends).
- The web UI says some terms more plainly than the CLI, MCP and the docs, which keep the precise
  words: an arm is a *wording*, a target *what you count* (in a result, *what was counted*, its
  label on a line of its own; a refinement's, *what the model should do*), a screen a *test at this
  step*, a validation *full test runs*, a spec a *task file*.
  `ui/static/js/words.js` holds the phrase for each term the glossary defines (a term's `ui`;
  `tests/test_words.py` holds the two together), the Words page shows both, and `tuningfork
  glossary` finds a term by its page phrase too. A sentence the server writes — a refusal, a plan's
  warning — keeps the precise words, as the terminal prints it; one that points at a control names
  it in both front ends. A target's badge beside a behaviour says *matches* or *no match*, never
  *counted*: amber *not counted* is a run or fork left out of a pass rate. The pages' own
  sentences: one idea a sentence, a sentence rather than a label with a colon; what a thing is
  before what it is not ("a lead, not a verdict"); the plain phrase first, the precise term after
  it, muted or in a tooltip; help under a field says what to type, why it matters goes in a
  tooltip; buttons are verb phrases, with an ellipsis only where a form opens.
- A page reads top to bottom in the order the work goes: what to do first, then what it made.
  What are alternatives, or run beside what is read, sit side by side — the two ways to propose
  an edit, the work on a run's step beside the run. On a narrow page alternatives stack, and the
  work opens over what is read. One step of the loop is one section, not several small ones. A
  form opens by itself only as its page opens; after, only the person opens one, and none closes
  under them.
- The UI keeps green and red for a check: a run's verdict (which closes its page), a fork's, a
  gate's, and a change in how often a task's check passes (an A/B or a screen run to the end).
  Any other significant result is the accent's colour, said in words with its direction —
  tuningfork is never told whether more of a behaviour is wanted. A rate is never green or red,
  a pass rate included: its bar is the accent's (a variant's teal), the verdict it counts said in
  its label. One meaning a colour, the legend written beside the tokens in `app.css` and shown in
  a line on the Guide: amber (`warn`) a caution — not counted, unfinished, paused, held,
  something missing or gone wrong (a failed call or job is one, never a verdict's red); the
  accent what can be acted on, what is current or significant, and the model's own answer;
  purple a skill's own words; teal (`change`, with a pencil) what a person or tuningfork changed
  — an edit at a breakpoint, a variant swapped in — on chips, outlines and banners, never a
  step's border; everything else grey, what is fine included (found, done, a file that matches,
  every role but the assistant's). Red keeps only what every page means by it besides: a diff's
  removed lines beside its green added ones, a button that deletes, a field or an action to put
  right.
- tuningfork's own endpoints (`/_tuningfork/`) answer only to a loopback Host, refuse
  cross-site state changes by Origin, and accept no state change without the access token
  (`api/guard.py`): a loopback port is reachable from any page in the user's browser. Keep new endpoints under
  that prefix. The UI is served with a strict CSP — its own scripts only, nothing inline —
  and only `ui/static/js/api.js` talks to the server or touches the token.
