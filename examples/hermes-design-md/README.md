# A real skill, fixed at the step that fails: Hermes's design-md skill on Gemma 4

This folder holds a real skill — **Hermes Agent's own `design-md` skill**, as Hermes 0.18.2
ships it — and a task it fails about a third of the time on a local model, Gemma 4 26B-A4B.

The skill writes a DESIGN.md: a brand's design tokens — colors, type, spacing, components — as
YAML, with prose saying how to use them. Its workflow ends: lint the file, and fix any WCAG
failures before returning. On this model no run ever lints, and one design in three ships its
primary button as white text on a light teal, at 2.6:1 — below the 4.5:1 that WCAG AA asks of
text. The guide walks through each stage:

1. Record runs and watch some fail.
2. Find the step where it is decided.
3. Show that the wording, not luck, decides that step.
4. Screen edits — a model's and your own — at exactly that step.
5. Validate the survivor with full runs, a task held out included.
6. Approve it into the skill's own file.
7. Measure the runs after.

Every output below is from a real run of these commands, on a fresh copy of the repository.
Paths are shortened to `/home/you` and the proxy's port to its default, 8787; long outputs are
trimmed where marked `...`.

## What is in this folder

| File | What it is |
|---|---|
| [`design-md/`](design-md/) | Hermes's design-md skill as Hermes 0.18.2 ships it: `SKILL.md` and its starter template |
| [`LICENSE-hermes`](LICENSE-hermes) | Hermes Agent's MIT licence, which the skill is under |
| [`evals.json`](evals.json) | Two tasks. **1**: a DESIGN.md for Tidewater, a teal brand. **2**, *held out*: one for Skyline, a sky-blue brand |
| [`checks/design.py`](checks/design.py) | The check: what the skill asks of the file, WCAG AA contrast included |
| [`fix/SKILL.md`](fix/SKILL.md) | A person's first fix: two lines at the end of the write step, saying to lint |
| [`fix/SKILL-section.md`](fix/SKILL-section.md) | A person's second: a section of its own, *Before you reply*, approved below |

The check sits beside the spec, outside the skill's folder, which every run is given a copy
of; a check finds it through `$TUNINGFORK_SPEC_DIR`. It runs under `uv run --with pyyaml`, so
it needs no Python packages of yours.

**What the check holds a file to.** What the skill tells the model to fix — a token reference
that resolves to nothing, a WCAG failure — and the spec's shape: YAML front matter with a
`name` and `colors`, each color a hex value or a reference to one, no section twice or out of
the canonical order. A component whose text and background colors fall below 4.5:1 fails. The
skill's linter has more rules, and the check does not rerun it: what the linter only warns
about otherwise passes, and so do its other errors, such as a unitless `letterSpacing: 0`.

**Why it fails.** The skill's workflow says:

```
2. **Write `DESIGN.md`** in their project root using `write_file`. Always
   include `name:` and `colors:`; other sections optional but encouraged.
3. **Use token references** (`{colors.primary}`) in the `components:` section
   instead of re-typing hex values. Keeps the palette single-source.
4. **Lint it** (see below). Fix any broken references or WCAG failures
   before returning.
```

The model writes the file and replies; step 4 never comes. Hermes's own `write_file` answers
the write with `"lint": {"status": "skipped", "message": "No linter for .md files"}` — its
built-in linting has nothing for Markdown — and only step 4, read long before, says otherwise.

## What you need

- **An NVIDIA GPU** with about 27 GiB free: the model, and two 64k-token contexts with the whole
  sliding-window cache. We used one RTX 5090 (32 GB).
- **llama.cpp's `llama-server`**; we used build b11108.
- **Hermes Agent 0.18.2 on Python 3.13**: `uv tool install --python 3.13 hermes-agent==0.18.2`.
- **[bubblewrap](https://github.com/containers/bubblewrap)**, on Linux
  (`sudo apt install bubblewrap`): every run then sees the same home and project folder,
  `/home/user` and `/home/user/project`, as an installation would.
- **Node.js 18 or later, and the network.** The skill's linter is `npx -y @google/design.md`:
  run, it fetches Google's package from npm, in a few seconds. Nothing is uploaded. Runs of the
  original skill never call it; the fixed skill's do.
- **uv**, which runs the check, and **tuningfork**, installed as in
  [the main guide](../README.md#1-install).

## 1. Start the model server

```bash
llama-server -hf lmstudio-community/gemma-4-26B-A4B-it-GGUF:Q4_K_M --no-mmproj \
  --alias gemma-4-26b-a4b --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 2 -ctk q4_0 -ctv q4_0 --jinja --swa-full --no-cache-prompt
```

That gives two slots of 65,536 tokens each: Hermes needs 64,000, and the batches below run two
at a time. Keep `--swa-full` and `--no-cache-prompt`: without either, the same request can
behave differently depending on what the server ran before it
([the main guide](../README.md#1-start-the-model-server) says why).

## 2. Start tuningfork on a copy

Approving an edit writes the skill's own file, so work on a copy of this folder. From your
tuningfork checkout:

```bash
cp -r examples/hermes-design-md ~/hermes-design-md
```

In a **second terminal**, the recording proxy:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-design
tuningfork serve --upstream http://127.0.0.1:8081 --max-inflight 2
```

In a **third terminal**, for everything that follows:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-design
cd ~/hermes-design-md
tuningfork connections check
```

```
  streaming_tool_calls   yes — structured delta.tool_calls observed
  anthropic_messages     yes — a Messages response
  fresh_prompts          yes — a prompt sent twice was processed whole both times
  known_output_format    yes — gemma4
  prompt_equivalence     yes — rendered 68 tokens, the chat endpoint's prompt has 68
  raw_completion         yes — 2 tokens forced by a grammar of ids, with their top alternatives
  logprobs               yes — 36 token entries
  top_logprobs           yes — top_logprobs present
  logprobs_all_tokens    yes — 36 entries for 36 generated tokens
```

## 3. Record runs

```bash
tuningfork batch evals.json -n 20 --concurrency 2 --harness hermes --model gemma-4-26b-a4b
```

```
design-md/1 — 20 runs
  prompt: Use the design-md skill to write a DESIGN.md for our app, Tidewater: calm, coastal and
trustworthy, … (+105 chars)
  check:  uv run --quiet --no-project --with pyyaml python "$TUNINGFORK_SPEC_DIR/checks/design.py"
  skill:  design-md
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  28.2s
  run 1  fail  29.4s
  run 2  fail  27.5s
  run 3  pass  44.2s
  ...
  run 19  pass  23.5s

14/20  70%  [48%-85%] passed

20 runs of design-md/1
  agreed for 1 step(s), diverged at step 1

step 1
   count  rate  95% CI    behaviour                                            then passed
A  1/20   5%    [1%-24%]  write_file(content='---\nversion: alpha\nname:       1/1  100%  [21%-100%]
                          Tidewater\ndescription:… (+33 chars)
B  1/20   5%    [1%-24%]  write_file(content='---\nversion: alpha\nname:       0/1  0%  [0%-79%]
                          Tidewater\ndescription:… (+33 chars)
...
```

Fourteen of twenty passed. Held-out task 2 is never run by a plain batch. Every run took the
same three steps — it viewed the skill, wrote DESIGN.md and replied — so the runs part at step
1, the write, where every file is its own. What is wrong inside one is for the check to say.

## 4. Find the step that decides it

Take the latest run the check failed, and read it:

```bash
RUN=$(tuningfork runs -q -n 1 --failed)
tuningfork show $RUN
```

```
run 4e8afdd5  design-md/1#57ffdfbc.16
  task    design-md/1
  model   gemma-4-26b-a4b
  started 10-06 10:39:01
  harness hermes
  sandbox /home/user/project, its home /home/user
  skills  84c3552a (2 file(s))
 k  outcome                                                     finish      ms        tokens  think
 0  skill_view(name='design-md')                                tool_calls  6214.15   13737   599
 1  write_file(content='---\nversion: alpha\nname:              tool_calls  11496.23  16676   none
    Tidewater\ndescription:… (+33 chars)
 2  I have created the `DESIGN.md` file for Tidewater in your   stop        6174.38   16903   -
    project root… (+171 chars)
```

`recheck` runs the task's check again on the workspace as the run left it, and says why it
failed; nothing is recorded:

```bash
tuningfork recheck $RUN
```

```
run 4e8afdd5 · design-md/1 · hermes
  check      uv run --quiet --no-project --with pyyaml python
"$TUNINGFORK_SPEC_DIR/checks/design.py"  (the run's own)
  workspace  as the run left it: DESIGN.md, README.md
  reply      I have created the `DESIGN.md` file for Tidewater in your project root.

The design follows your specifications:
- **Tone:** Calm, coastal, and trustworthy.
- **Colors:** Deep navy (`#001F3F`) for tex… (+318 chars)

fail — as when it ran
  component button-primary: #FFFFFF on #20B2AA is 2.62:1, below WCAG AA's 4.5:1
```

Five of the batch's six failures read the same: white text on `#20B2AA`, a light teal. The sixth
put white on `#008B8B`, darker, and still fell short at 4.15:1. Write the file out and look:

```bash
tuningfork files $RUN end --out tidewater-run
grep -n -A4 "button-primary:" tidewater-run/DESIGN.md
```

```
23:  button-primary:
24-    backgroundColor: "{colors.accent}"
25-    textColor: "#FFFFFF"
26-    rounded: "8px"
27-    padding: "12px 24px"
--
53:- **button-primary:** A high-emphasis action element using the teal accent. It is designed to be easily identifiable and inviting.
...
```

Had the model run the linter, as step 4 says, it would have been told:

```bash
(cd tidewater-run && npx -y @google/design.md lint DESIGN.md)
```

```
{
  "findings": [
    ...
    {
      "severity": "warning",
      "path": "components.button-primary",
      "message": "textColor (#ffffff) on backgroundColor (#20b2aa) has contrast ratio 2.62:1, below WCAG AA minimum of 4.5:1.",
      "rule": "contrast-ratio"
    },
    ...
  ],
  "summary": {
    "errors": 1,
    "warnings": 20,
    "infos": 3
  }
}
```

Its one error is a `letterSpacing: "0"` without a unit, which the check lets pass. Seventeen of
the warnings are `fontFamily: "Inter, sans-serif"`, written among the type styles: the linter
takes it for one and reads it a character at a time. (`rm -r tidewater-run` when you are done
with it.) So the step that decides it is step 2: the file written, the model replies instead of
linting. Read it:

```bash
tuningfork step $RUN 2
```

```
...
context
  [0] system: You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are
helpful, knowled…
  [1] user: Use the design-md skill to write a DESIGN.md for our app, Tidewater: calm, coastal and
trustworthy, …
  [2] assistant: tool_calls(skill_view)
  [3] tool: {"success": true, "name": "design-md", "description": "Author/validate/export Google's
DESIGN.md tok…
  [4] assistant: tool_calls(write_file)
  [5] tool: {"bytes_written": 2165, "dirs_created": true, "lint": {"status": "skipped", "message":
"No linter fo…

response
  content: I have created the `DESIGN.md` file for Tidewater in your project root.
...
```

Message 3 is the skill, as `skill_view` returned it; message 5 is the write's result, with
Hermes's own note that linting was skipped.

## 5. Bad luck, or the wording? Resample the reply

Send step 2's exact context to the model twenty more times:

```bash
tuningfork resample $RUN 2 -n 20
```

```
run 4e8afdd5 step 2 — 20 samples from http://127.0.0.1:8081
  context   6 messages, after the result of write_file
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 16,756 tokens, then answers: the recorded step
took 6.2s

    1/20  I have created the `DESIGN.md` for Tidewater at `<workspace>/DES… (+177 chars)  5.4s
    2/20  I have created the `DESIGN.md` file for Tidewater. It follows th… (+177 chars)  3.9s
    ...
   20/20  I have created the `DESIGN.md` file for Tidewater in your projec… (+177 chars)  5.6s

step 2, after the result of write_file — 20 samples
   count  rate  95% CI    reasoned  behaviour
A  1/20   5%    [1%-24%]  1/1       I have created the `DESIGN.md` file for **Tidewater** at
                                    `<workspace>/DESIGN.md`. The desi… (+151 chars)
...
experiment 442bbf7e · 1.4 min · 4.3s per sample · use `tuningfork ab 4e8afdd5 2 --target <letter>
--variant <file>` to test a wording change
```

Twenty replies, each worded its own way, so each is a behaviour of its own; not one ran a
command. Not bad luck: given this skill, the step is a reply.

## 6. Say what the step should do, then screen edits

Declare the target first — a call of the terminal that mentions `lint`:

```bash
tuningfork refine start $RUN 2 --target-rules '[{"if": "calls", "value": "terminal"}, {"if": "contains", "value": "lint"}]'
```

```
refinement f1a0a353 — open
  checkpoint  run 4e8afdd5 step 2 (task design-md/1)
  skill       design-md, as the run was given it:
              /home/you/tuningfork-design/batches/57ffdfbc/skill-snapshot/design-md/SKILL.md
  its file    design-md/SKILL.md
  target      a behaviour that calls terminal, and mentions "lint"
              declared before anything was proposed
  started by  a person
...
```

Ask the built-in proposer — the model on your own server, shown the step's evidence. It reasons
at length, so give it room:

```bash
REF=$(tuningfork refine list -q -n 1)
tuningfork connections proposer set --max-tokens 32768
tuningfork refine propose $REF --builtin -m 3
```

```
the built-in proposer — 3 proposals
  sends     the packet — 24 KB of recorded prompts, answers and the skill — to 127.0.0.1:8081
  model     the server default (openai-chat, max_tokens 32768)
  packet    83a9a2eab0cbc6e2
  asking…
  answered in 58s, 6859 tokens · kept in proposer/20261006T150455-41b775
  proposal fcf71f27: 1 line changed
    why  Renaming the section to include 'validating' emphasizes that the authoring process is not
complete until the file has been validated with the linter, which helps prevent the agent from
skipping the verification step.
  proposal 1 not kept: edit 1: its old text is not in the skill — the closest lines in the skill:
'4. **Lint it** (see below). Fix any broken references or WCAG failures'

hypotheses until screened: `tuningfork refine screen f1a0a353 --proposals fcf71f27`
```

One proposal was kept: rename the workflow's heading to "authoring and validating". The other
quoted step 4 inexactly, and an edit's `old` text must be the skill's own. Your proposer's
answer will differ: it samples.

Add your own two wordings, each with its reason. `fix/SKILL.md` adds two lines to the write
step; `fix/SKILL-section.md` adds a section of its own, before the file's anatomy:

```bash
tuningfork refine propose $REF --file fix/SKILL.md --why "Every run wrote DESIGN.md and replied at once: none ran the linter that step 4 of the workflow asks for, and a third shipped a button below WCAG AA (white on #20B2AA, 2.62:1), which the lint reports as a contrast warning. Step 4 comes after the model has replied; the write step is where it acts."
tuningfork refine propose $REF --file fix/SKILL-section.md --why "Every run wrote DESIGN.md and replied at once, though step 4 says to lint it: the write tool answers that lint was skipped, no linter for .md files, and step 4 comes after the reply. A section of its own, before the file's anatomy, says to lint before replying and to fix what the lint reports."
```

```
  proposal 1bc3548f: 2 lines changed
...
  proposal f3f47837: 5 lines changed
```

Screen all three against a fresh draw of the original, twenty samples an arm:

```bash
tuningfork refine screen $REF -n 20
```

```
run 4e8afdd5 step 2 — 20 samples per arm from http://127.0.0.1:8081
  ...
  target    a behaviour that calls terminal, and mentions "lint"
            declared when the refinement began
  arm       fcf71f27: 1 line changed
  arm       1bc3548f: 2 lines changed
  arm       f3f47837: 5 lines changed
  survives  with at least 7 of 20 — the step did it 0 times in 20 in resample 442bbf7e
            Holm over 3 proposals, α=0.05
  samples   80 in all: the original first, then each arm
...
target (declared before sampling): a behaviour that calls terminal, and mentions "lint"
          target                 p        Holm
original  0/20   0% [0%-16%]
fcf71f27  0/20   0% [0%-16%]     1.000    1.000    not significant
1bc3548f  1/20   5% [1%-24%]     1.000    1.000    not significant
f3f47837  20/20 100% [84%-100%]  1.5e-11  4.4e-11  survives
p: Fisher's exact test of each proposal against the original arm; Holm: corrected over the 3 planned
before sampling (α=0.05).
A screen at one step is evidence, not proof: confirm a survivor with full runs.
screen b556c1a6 · 4.5 min · `tuningfork refine show f1a0a353` shows every proposal with its verdicts
```

**Where an instruction sits decides whether it is read.** Renaming the heading did nothing.
Two lines at the end of the write step, saying the model MUST run the linter, reached it once in
twenty. A section of its own did it every time:

```bash
tuningfork refine show $REF
```

```
...
  f3f47837  5 lines changed · a person
    why  Every run wrote DESIGN.md and replied at once, though step 4 says to lint it: the write
tool answers that lint was skipped, no linter for .md files, and step 4 comes after the reply. A
section of its own, before the file's anatomy, says to lint before replying and to fix what the lint
reports.
    @@ line 36 @@
      from scratch (prototype, deck, landing page, component lab), use
      `claude-design`. This skill is for the *formal spec file* itself.
    +
    + ## Before you reply
    +
    + After writing DESIGN.md, run `npx -y @google/design.md lint DESIGN.md`. If it reports an
    + error or a contrast warning, fix the file and lint again. Reply only when the lint is clean.

      ## File anatomy
    screened  survives — 20/20 against the original's 0/20, Holm p 4.4e-11 (screen b556c1a6)
...
```

## 7. Validate it with full runs

A screen tests one step: the model now lints. Whether it then fixes the file is for full runs
to say. Validation runs the whole task three ways, each against a fresh original arm:

- **forward:** forks from the checkpoint to the end;
- **training:** fresh runs of task 1 with the edit from the first step;
- **held out:** runs of task 2, Skyline, which nothing so far has seen. They go through a private
  proxy that keeps only verdicts.

```bash
tuningfork refine validate $REF f3f47837 --concurrency 2
```

```
validate proposal f3f47837 of refinement f1a0a353 — design-md/1, at step 2
  screened  survived screen b556c1a6: 20/20 against the original's 0/20, Holm p 4.4e-11
  forward   5 forks an arm from step 2 to the end, the original first
  training  5 full runs an arm of design-md/1, the edit from the first step
  held out  5 full runs an arm of each of: 2 — only their verdicts are kept,
            recorded in a temporary folder under /tmp, outside the recordings folder,
            and deleted when they end
  runs      30 in all, through http://127.0.0.1:8787
  at a few runs an arm only a large difference shows: the not-worse checks are low-powered

forward — forks from step 2 to the end
  original   1/5 abcfe431  fail  8.4s  I have created the `DESIGN.md` file for Tidewater using the …
(+181 chars)
...
  edit       1/5 4e315c4b  pass  59.1s  terminal(command='npx -y @google/design.md lint DESIGN.md')
...
  edit       4/5 fe7c5c79  fail  85.0s  terminal(command='npx -y @google/design.md lint DESIGN.md')
  edit       5/5 af131457  pass  68.7s  terminal(command='npx -y @google/design.md lint DESIGN.md')

training — full runs of design-md/1
  original   1/5 12618de4  pass  27.6s
...
  edit       5/5 e3024a0c  pass  135.8s

held out 2 — verdicts only
  original   1/5  ended
...
  edit       5/5  ended

  c82d9d66  done  proposal f3f47837 — meets the gate
    forward   edit 4/5 (80%, 95% CI 38–96%); original 0/5 (0%, 95% CI 0–43%)
    training  edit 4/5 (80%, 95% CI 38–96%); original 4/5 (80%, 95% CI 38–96%)
    held out 2: edit 3/5 (60%, 95% CI 23–88%); original 0/5 (0%, 95% CI 0–43%)
    G0 met     survived a screen: it survived screen b556c1a6
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 4/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): not significantly worse (Fisher p = 1); at 5 runs an arm only a large drop would show
    G5 met     held-out task 2 is not worse (low-powered): not significantly worse (Fisher p = 0.167); at 5 runs an arm only a large drop would show
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 3/5, without it 0/5

it meets the gate: `tuningfork refine approve f1a0a353 f3f47837` shows what approving writes
```

Every check is met. From the checkpoint, the original's forks replied as before and failed;
the edit's ran the linter, read the contrast warning and gave the teal button the brand's navy
text in place of white, and four of five passed. Held out, Skyline's sky-blue buttons failed every original
run; with the edit, three of five passed — G8, where an edit that wrote in the training task's
answers would fail. Fresh runs of task 1 came out even at four in five, and five an arm can say
no more than that the edit is not much worse. Each of the edit's runs takes about a minute: the
linter, and a second write.

## 8. Approve it

Read what approving would write. Nothing is written until you type the code:

```bash
tuningfork refine approve $REF f3f47837
```

```
approve proposal f3f47837 of refinement f1a0a353, on validation c82d9d66
  writes     /home/you/hermes-design-md/design-md/SKILL.md
             \\wsl.localhost\Ubuntu\home\you\hermes-design-md\design-md\SKILL.md
  keeps      LF line endings
  file before → /home/you/hermes-design-md/design-md.history/
  changelog   → /home/you/hermes-design-md/design-md.changelog.md
  the gate   every check met:
...
  the edit
    @@ line 36 @@
      from scratch (prototype, deck, landing page, component lab), use
      `claude-design`. This skill is for the *formal spec file* itself.
    +
    + ## Before you reply
    +
    + After writing DESIGN.md, run `npx -y @google/design.md lint DESIGN.md`. If it reports an
    + error or a contrast warning, fix the file and lint again. Reply only when the lint is clean.

      ## File anatomy

  code  87b45a7d
Type the code to write it (anything else writes nothing): 87b45a7d
written: /home/you/hermes-design-md/design-md/SKILL.md
  the file before  /home/you/hermes-design-md/design-md.history/SKILL.20261006T152656Z-38ed14bc.md
  changelog        /home/you/hermes-design-md/design-md.changelog.md
  to undo, copy the file before back over it
```

Your code will differ: it names the proposal, the file and its bytes. The changelog entry beside
the skill folder holds the reason, the target, every check of the gate and the diff.

## 9. Run it again

```bash
tuningfork batch evals.json -n 20 --concurrency 2 --harness hermes --model gemma-4-26b-a4b
```

```
...
  run 0  pass  53.7s
  run 1  pass  138.8s
...
  run 18  pass  124.1s

20/20  100%  [84%-100%] passed
```

Twenty of twenty, against fourteen of twenty before (Fisher p = 0.02). Every run ran the
linter after writing. Each now takes one to four minutes rather than half of one: the linter,
and a second write.

```bash
tuningfork tasks
```

```
design-md/1 · hermes · gemma-4-26b-a4b
 skill     runs  check passed
 84c3552a  25    18/25  72%  [52%-86%]
 2c709d2e  25    24/25  96%  [80%-99%]

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

Counted with the validation's fresh runs of task 1, 72% before and 96% after (Fisher p 0.049).
`tuningfork skill design-md` lists both versions with their rates, the approval that replaced
the first, and the changelog and old file beside the skill folder.

## What is left

The edit's failures came from fixing, not from skipping the lint. In the validation, one fork's
patches left a reference misspelled, `{colors::primary}`: the linter reads it as plain text and
lets it pass, though nothing resolves it. One fresh run patched away the `---` that closes the
front matter, and replied without linting again. Telling the model to write the file again
whole, rather than patch it, is the next thing to try — at the step after a lint, refined the
same way.

The check holds a file to what the skill says to fix; the linter's other errors — a unitless
`letterSpacing: 0`, here — pass it. A stricter check would run the linter itself, and depend
on the network and on whichever version of the linter npm serves that day.

## How this page was measured

- **The server kept its whole cache and processed every prompt whole** (`--swa-full`,
  `--no-cache-prompt`), and each experiment drew new seeds, so no sample depended on what the
  server ran before it.
- **Every run was laid out as Hermes installs itself, in a sandbox:** its home `/home/user`, the
  skill in `~/.hermes/skills/`, the task's files in `/home/user/project`.
- **The runs could reach the network,** which the edited skill's linter needs.

## What went wrong the first time

- **A fix that worked at one checkpoint did nothing at another.** The two lines in
  `fix/SKILL.md` made the model lint in 8 of 10 forks from the step after the write of a run
  recorded while this page was planned, and in 1 of 20 samples at the run above. A screen answers
  for the checkpoint it samples; a validation's forks and full runs answer for more. The section
  of its own made the model lint at both.
- **The proposer's answer did not parse, twice.** Once the JSON was malformed. Once the model wrote
  its JSON, saw a mistake in it ("Wait, I see I made a mistake…") and wrote it again, and the two
  read as one were not JSON. tuningfork now takes the last whole object holding a list of
  proposals, the model's last word; the refinement above was run again with it.
- **The first check was stricter than the skill.** It failed a component property outside the
  linter's list, and a type style written inline, which the linter only warns about — and so
  failed every run. The check now holds a file to what the skill asks.

## Clean up

Stop the proxy and `llama-server` with Ctrl-C, then:

```bash
rm -rf ~/tuningfork-design ~/hermes-design-md
```

The skill in this folder is © Nous Research, under the MIT licence in
[`LICENSE-hermes`](LICENSE-hermes).
