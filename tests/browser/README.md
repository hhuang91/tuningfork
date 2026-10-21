# Browser tests

The web UI's behaviour, checked in a real browser. Not part of the default test run: it
needs a Chromium-based browser and Node 22+. The default run still covers the UI's static
files and its escaping rule (`tests/test_ui.py`); this covers what only a browser can show —
editing a held call on the Live page and checking that the harness received the edit.

## Run it

Three terminals, from the repository root:

```bash
# 1. The real proxy app in front of the mock model: no GPU needed. After a moment it arms
#    a write_file breakpoint and makes one call that stops there.
uv run python tests/browser/demo.py live /tmp/tuningfork-demo 8791

# 2. Any Chromium-based browser, headless, with a debugging port.
chromium --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/fp-browser about:blank
#    (Chrome or Edge work too: `google-chrome ...`, or msedge.exe on Windows.)

# 3. The test.
node tests/browser/live_ui.mjs
```

It checks, through the page and then through the API:

- a held `write_file` shows its content raw, can be edited, and **the harness receives the
  edit** — while the step still records what the model produced;
- a stop before the model can be edited and stepped: it stops again after the model, the
  model saw the edited message, and the harness's own request is kept;
- a decision made on the page is never reported as made elsewhere (a real race: the proxy
  announces the resolution before it answers the request that made it);
- the breakpoint form infers the phase and reports the proxy's refusals;
- pause and resume reach the proxy;
- the access token leaves the address bar, and a tab without it is read-only: its controls
  change nothing until the token is pasted into the banner;
- the page throws no errors.

The test finds the demo's access token in the file the demo names at `/health`; when Node
runs where that file cannot be read, start both with the same `TUNINGFORK_TOKEN`.

It can be run repeatedly against the same demo; each run starts by releasing anything
left held.

## Starting work from the browser

`jobs_ui.mjs` drives the forms that start work, against `demo.py lab`: the proxy over
the seeded recordings, in front of a model that names the report's keys only when the skill
states them — and with a harness it can launch, `tests/fake_agent.py` as "fake", and a
task for it in `<home>/lab-task/evals.json`. The lab's raw endpoints (`/apply-template`,
`/tokenize`, `/completion`) are the mock's toy llama.cpp (`tests/mock_server/toy.py`), and
its run `tokens-lab` has an answer that toy model parts on one token — read the file, or
search for it.

```bash
# a fresh home each time; 127.0.0.1:18099 stands in for a protected endpoint
TUNINGFORK_PROTECTED=127.0.0.1:18099 uv run python tests/browser/demo.py lab /tmp/tuningfork-lab 8791
node tests/browser/jobs_ui.mjs
```

It checks that a resample started from a step on the run's conversation runs through the
proxy and links its result; that an A/B of a wording edited in the page, with a behaviour's letter
as its target, starts at one press of Start with the mouse, the name just typed still focused (the change
event as the focus leaves it once lost that click) — a form asking for its plan again when the page
refreshes it, and not for an event that changes nothing — shows the known shift, saves the variant only
when it starts, and leads its page with the
edit it tested, then the target and which way the variant moved it — in the accent's colour, never
a check's green — with each behaviour badged by whether it matches what is counted; that no target is
chosen for you; that a task runs again from its run's page, its check shown whole; that a
fork stops at its step for a person and runs on when continued, and one cancelled as it waits
keeps no verdict, its page saying it was stopped and why; that forks run to the end,
for a resample and for an A/B, whose page names the task's check as its target; that a job's page, its event stream cut mid-job, catches up
with every event shown once; that cancelling takes two clicks and keeps what was
sampled, and a running job's time moves while nothing is sent; that a scenario made in the
page — its skill folder shown where it resolves, its check flagged as the shell command it
is, leaving it unsaved asking first, a file uploaded through its picker — runs as a batch
(chosen on Run a task, where a task file's path, on the same line and named apart, is greyed out
while a scenario is chosen) and shows its pass rate, in the accent, and that its check, tried on the run it judged, gives the verdict
the run got, and passes on a report written with the right keys; that at a held call, sampling places the held answer among the
sampled behaviours and is filed under the step once the call is recorded, while a question
to the model comes back marked a hypothesis; on the Connections page, that it leads with the
setup at a glance — the server, the harnesses, the sandbox and the proposing model, the server
test there, and what it means before the test and after — that a server
check measures what the harnesses need, a protected endpoint is refused, and an idle proxy
switches its server after a dry run; that Start… on the Jobs page leads to a step's form;
that a run reads as one conversation — each message once, the final reply marked (and not
green), its verdict closing the page and the step rail, coloured by it with what the check
printed, a step opening in place with its form in it, and its exact context on a page of its own;
that the divergence page leads with the step whose behaviour best predicts a pass, in a callout
with the ways to it, colours each run by its own verdict, draws "then passed", the tally's counts
and where a question held as rates — the accent's, never green or red — and says plainly how it
counts the failed runs' check lines;
that an answer scored from its Tokens view is rebuilt token for token and drawn once
per token, a token shows what else the model weighed there, and a token selected there
opens the forking-token form, whose probe finds that token decides the answer; that the
probe's page shows the answer it forked from with that token ringed, and leads to it on the
step's Tokens view and back; and that a target described in conditions is set against what
the step has done, and declared in words; that a refinement started from a step declares its
target first, offering no exploratory choice, and stands at its first stage, proposing, its form
the one open and what a proposer is shown folded; that the lab's built-in proposer, asked through
the proxy and never recorded, has two of its three ideas kept and one refused for editing a
line the skill does not have, the refinement then moving on to testing them, that form open; that
a person's own wording joins them, proposing opened again from its stage's link; that a screen of
two, Holm-corrected, finds the one that names the keys — a survivor in the accent's colour, its
behaviours badged by whether they match what is counted — and each proposal then carries its
verdict, the refinement's page leading with that evidence — where it stands, each proposal named
by the line it changes, its screen in a line — above what a proposer is shown, the original
wording's answers, with the skill's paths closed away in About, and full test runs next, their
form open; that a survivor, validated from its refinement's page, runs forks to the end, the
training task and the lab's held-out task — named only by its id in the plan and the job,
its text nowhere — and meets the gate, each part and check shown to the person, the evidence
then naming it ready to approve, its screen and its full runs in a line each and a link down
to them, and approving the stage next, its link leading to those full runs; that
approving shows the file it writes and the edit, writes nothing for a wrong code, and writes
the skill's own file once the code shown is typed, the page then saying so first, the
approved proposal picked out in the evidence and in its card, every stage done; and that the proposer's key, set on
Connections, is never on the page or in an answer again. On the way it checks that a batch's plan opens with
how a run passes and offers the spec's task by what it asks, that Start is ready with nothing
more typed, and that the Runs page says how a task passes; and that a pass rate is drawn in the
accent, never a verdict's green — on Runs, on Skills and on a batch's page. Last, it moves a run to the trash
from its page — out of the Runs list, on the Trash page with what deleting it would remove —
restores it, and deletes it for good, the first click saying what goes; and two finished jobs
ticked on the Jobs page go to the trash together. And a job started as an agent would over
MCP — `started_by` naming its client — says so on the Jobs list and on its own page.

The resample's job page, once done, says what it found in a line with its rate, leaves the whole
to the experiment's page and folds its log, links the step it worked at, and says it was started
from a step's page; the Jobs list names it "Measure · Resample", leads to its experiment and says
who started it. The list is in parts: running and queued first, saying when nothing is; then
finished; then stopped or failed. With two jobs queued behind one waiting out a pause, the running
job comes first, then each queued job at its place in line, its own page heading its panel
"Queued"; cancelled in line, they move to stopped or failed, whose own box ticks its jobs alone,
the bar counting them.

The job pages of a probe, an A/B, a test at a step and forks run to the end each say what they
found in a line, without the experiment's own views, and lead to them; a cancelled job keeps its
log open. The note when a job ends leads to what it made — a held call's samples to their
experiment — or to the job, where what it made is the page already open: a server test started on
Connections. And what each kind of job made and found (a probe that saw no flip says what it could
have seen, as its own page does), the cautions a probe or a test leaves open (a test's own page
says what it left out of its count too), why an ended job's log is empty, the job pages of a
chain, a fork, a batch and a validation, the list's parts with both lines of queued work, and each
page's name are worked out as the list and the pages say them. A form's plan answered after the
form changed again is dropped, the newer plan shown.

Set `TUNINGFORK_UPLOAD` to a small CSV on the browser's machine to upload it through the
page's own file picker; without it, the test types the file in as text instead.

The helpers both tests use — the page, the API with the demo's token — are in `cdp.mjs`.
A page's content-security-policy violations count as page errors.

## Looking at the pages

`demo.py seed` records a varied store — runs that diverge, an edited step, a run that
cannot be counted, a resample and an A/B — and `demo.py browse` serves it read-only, as
`tuningfork ui` would. Headless screenshots are a quick check after a CSS change:

```bash
uv run python tests/browser/demo.py seed /tmp/tuningfork-seed
uv run python tests/browser/demo.py browse /tmp/tuningfork-seed 8790
chromium --headless=new --window-size=1440,1000 --virtual-time-budget=4000 \
  --screenshot=runs.png 'http://127.0.0.1:8790/_tuningfork/ui/#/runs'
```

`demo.py proxy` is `live` without the breakpoint and the call: the real proxy in front of
the mock model, for trying breakpoints by hand. `examples/README.md` walks through it.

A live page keeps its event stream open, so `--virtual-time-budget` never runs out there;
give live pages `--timeout=6000` instead. Headless windows are at least about 500 px
wide: a narrower `--window-size` crops the screenshot rather than narrowing the layout.
