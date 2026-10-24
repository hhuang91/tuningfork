# Images

The pictures in the top-level README, and in [the web UI page](../web-ui.md).

- **`hero-light.svg`, `hero-dark.svg`** — the opening figure, drawn by `hero.py` from the runs
  of [`examples/find-the-file/`](../../examples/find-the-file/README.md) (Gemma 4 31B through
  Hermes Agent), where each number is shown with the command that made it, in the web UI's
  colours. GitHub shows the dark one in a dark theme. After a change to the script or the
  numbers: `python3 docs/images/hero.py docs/images`.
- **`ui-*.png`, `ui-demo.gif`** — the web UI, taken by [`shots.mjs`](shots.mjs) as listed below:
  after a change to the pages, one command takes them all again.

## Screenshots

| File | What it shows | Taken from |
|---|---|---|
| `ui-demo.gif` | A run opened, its write step's work beside it, an edited wording A/B-tested there, its job running, and the A/B's own page: the edit, then the original arm against it, with intervals and Fisher's test. About 35 seconds, no sound | `tuningfork demo` |
| `ui-guide.png` | The front page from its title down: where you left off, what is set up, and the loop from a skill to its next version | the guide's recordings |
| `ui-run.png` | A step's exact context: every message as the model saw it, the skill marked in it, and the call it made | find-the-file's |
| `ui-divergence.png` | A task's divergence page: the step that best predicts the verdict, then where the runs split and how each behaviour fared | find-the-file's |
| `ui-ab.png` | An A/B's result at a step: the edit it tested, both arms with their intervals, and the declared target's test | find-the-file's |
| `ui-tokens.png` | An answer's token heatmap, one uncertain token's alternatives open | find-the-file's |
| `ui-refine.png` | A refinement: its target and its four stages, then the evidence — each proposal's screen, the approved one's validation — and a section for each stage | the guide's recordings |
| `ui-live.png` | The Live page with a call held at a breakpoint, its tool call edited | `tuningfork demo live` |
| `ui-scenario.png` | A chat had elsewhere, made a scenario: its first message the starting prompt, and each of its steps beside what a run of the scenario does there | `tuningfork demo` |

**Real recordings** are runs on a real model, so that the numbers on the page are ones the docs
report. Those behind the hero, [`examples/find-the-file/`](../../examples/find-the-file/README.md)'s,
suit the run, divergence, A/B and token shots best — its step 1, token [121] open on "Wait", is the
figure's fork. The refinement shot is the guide's own Hermes refinement, `338504b1` in
[`examples/README.md`](../../examples/README.md): the whole loop on one page, from its target
through three proposals and their screen to a validated, approved edit. find-the-file has no
held-out task, so no refinement of it could be validated. The ids the pictures show are at the top
of `shots.mjs`. **`tuningfork demo`** serves seeded runs through a real proxy, in front of a scripted
model that names the report's keys only when the skill holds the words `row_count and
amount_total`: an A/B of that edit shows the shift in seconds, with no GPU. The chat the scenario
is made from is made up in the script, about the example skill: no one's real conversation.
**`tuningfork demo live`** is the same proxy with a call held at a breakpoint.

## Taking them

Four servers, each over the recordings its pictures show, and a browser:

```bash
# Copies of the recordings, under /tmp, so that no path a page shows names you
cp -a <find-the-file's recordings> /tmp/tuningfork-shots/find-the-file
cp -a <the guide's recordings> /tmp/tuningfork-shots/guide
uv run tuningfork serve --home /tmp/tuningfork-shots/find-the-file --port 8801 --upstream http://127.0.0.1:8081
uv run tuningfork serve --home /tmp/tuningfork-shots/guide --port 8802 --upstream http://127.0.0.1:8081
# The demos, each in a new folder of its own under /tmp
uv run tuningfork demo --port 8790
uv run tuningfork demo live --port 8791
# Any Chromium-based browser, headless, with a debugging port
chromium --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/tuningfork-shots-browser about:blank
```

Then, each source named by the web UI address its server printed, token and all:

```bash
TUNINGFORK_BROWSER=http://127.0.0.1:9333 \
TUNINGFORK_SHOTS_FINDFILE='http://127.0.0.1:8801/_tuningfork/ui/?token=…' \
TUNINGFORK_SHOTS_GUIDE='http://127.0.0.1:8802/_tuningfork/ui/?token=…' \
TUNINGFORK_SHOTS_LIVE='http://127.0.0.1:8791/_tuningfork/ui/?token=…' \
TUNINGFORK_SHOTS_LAB='http://127.0.0.1:8790/_tuningfork/ui/?token=…' \
TUNINGFORK_SHOTS_PRIVATE='<your user names>,<a host or a port of yours>' \
node docs/images/shots.mjs                   # every picture; or name some: shots.mjs ui-run ui-tokens
```

It writes each picture here, under the name the pages use, so a picture taken again appears in
the README and in `docs/web-ui.md` with no other change; `--out <folder>` writes them elsewhere, to look at first. A
picture whose source is not named is skipped. What it does for you:

- **The page alone, one pixel to one.** Headless, there is no address bar — and so no access
  token — on a picture, and nothing to crop. A page is 1440 pixels wide and 896 high (the chat
  made a scenario 1000, the GIF 1200 by 800), at a device scale of one, in the light theme, its
  scrollbars hidden.
- **Nothing of one machine.** A screenshot is a tracked file like any other. The script reads what
  each page shows — its text and what its fields hold, and on the GIF's every page, any text it
  showed however briefly — and reports a path naming a person or a place (`/home/<name>`,
  `/Users/<name>`, `C:\…`, `\\wsl.localhost\<distro>`), or a word `TUNINGFORK_SHOTS_PRIVATE` lists.
  It then exits non-zero, and leaves that picture out of the folder: the report says where it is.
  Serving copies under `/tmp` keeps the paths a page shows, a skill's or a variant's, naming nobody.
- **Small files.** It reports a PNG over 500 KB and a GIF over 10 MB, GitHub's limit for an image.
  A GIF plays by itself; a video on GitHub waits for a click.

`serve` rather than `tuningfork ui`, over the recordings: the pages then offer what a running proxy
offers, nothing greyed, and the front page folds what is set up to one line once the server
answers. The servers send nothing to the model to take the stills.

## The GIF

The script plays it on the lab, reading the page's own screencast frame by frame while it clicks
and types, with a pointer drawn where each click lands. ffmpeg (`FFMPEG` names it where it is not
on the PATH) puts the frames together, one palette for all of them, dithered so that a colour few
pixels hold, a variant's teal, still reads as itself. What it plays:

1. **Runs** → run `tally-report/1#demo.2` → step 1's **Work from here** → **A/B a wording…**
2. In the editor, `Write the totals to report.json.` becomes
   `Write the totals to report.json, using the keys row_count and amount_total.`
3. Under *What to count*, *A behaviour I describe*: *Calls the tool* `write_file`, then
   *+ Add a condition* twice, for *Mentions* `row_count` and *Mentions* `amount_total`. The
   suggested name and 20 samples for each wording stay as they are, and it starts.
4. Its job runs, and its page says what it found. *See the result* opens the A/B's own page: the
   edit as a diff, then the original's 0/20 against the variant's 20/20, more than chance would
   explain (p = 1.5e-11).

The lab's model has no randomness: every sample of an arm is the same answer. Equal arms mean the
sentence differs from the one above — the names in backticks, or a comma between them, get the
wrong keys in every sample.
