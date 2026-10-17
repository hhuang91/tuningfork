# Images

The pictures in the top-level README.

- **`hero-light.svg`, `hero-dark.svg`** — the opening figure, drawn by `hero.py` from the runs
  of [`examples/find-the-file/`](../../examples/find-the-file/README.md) (Gemma 4 31B through
  Hermes Agent), where each number is shown with the command that made it, in the web UI's
  colours. GitHub shows the dark one in a dark theme. After a change to the script or the
  numbers: `python3 docs/images/hero.py docs/images`.
- **`ui-*.png`, `ui-demo.gif`** — the web UI, taken by hand as listed below.

## Screenshots

| File | What it shows | Taken from |
|---|---|---|
| `ui-demo.gif` | A run opened, its write step chosen, an edited wording A/B-tested there, and the result: the edit, then the original arm against it, with intervals and Fisher's test. 30 seconds at most, no sound | `demo.py lab`, [step by step](#the-gif-step-by-step) |
| `ui-guide.png` | The front page from its title down: where you left off, what is set up, and the loop from a skill to its next version | real recordings |
| `ui-run.png` | A step's exact context: every message as the model saw it, the skill marked in it, and the call it made | real recordings |
| `ui-divergence.png` | A task's divergence page: where the runs split, how each behaviour fared, and the step that best predicts the verdict | real recordings |
| `ui-ab.png` | An A/B's result at a step: the edit it tested, both arms with their intervals, and the declared target's test | real recordings |
| `ui-tokens.png` | An answer's token heatmap, one uncertain token's alternatives open | real recordings |
| `ui-refine.png` | A refinement: its target, then the evidence — each proposal's screen, the approved one's validation — and the proposals as diffs | real recordings |
| `ui-live.png` | The Live page with a call held at a breakpoint, its tool call being edited | `demo.py live` |

**Real recordings** are runs on a real model, so that the numbers on the page are ones the docs
report. Those behind the hero, [`examples/find-the-file/`](../../examples/find-the-file/README.md)'s,
suit the run, divergence, A/B and token shots best — its step 1, token [121] open on "Wait", is the
figure's fork — and `tuningfork ui --home <their folder>` serves them read-only. The front page
reads better through `tuningfork serve` over them, its server checked on Connections: what is set
up then folds to one line, where read-only browsing says nothing can be started. The refinement
shot is the guide's own Hermes refinement, `338504b1` in [`examples/README.md`](../../examples/README.md):
the whole loop on one page, from its target through three proposals and their screen to a
validated, approved edit. find-the-file has no held-out task, so no refinement of it could be
validated. Leave its *About this refinement* closed: it holds the skill's paths.
**`demo.py lab`** serves seeded runs through a real proxy, in
front of a scripted model that names the report's keys only for one wording of the skill, given
[below](#the-gif-step-by-step): an A/B of it shows the shift in seconds, with no GPU.
**`demo.py live`** is the same proxy with a call held at a breakpoint. Each prints the web UI's
address:

```bash
python tests/browser/demo.py lab ~/tuningfork-lab 8790
python tests/browser/demo.py live ~/tuningfork-live 8791
```

How to take them:

- **The page alone.** Crop out the browser's address bar: the address carries the proxy's access
  token.
- **Nothing of one machine.** A screenshot is a tracked file like any other. Look for folders on
  the page — the Connections page, a run's sandbox, a swapped skill's path — and shoot from a
  recordings folder whose path names nobody.
- **One size.** A window 1440 pixels wide, browser zoom at 100%, the light theme; crop to the
  part the caption names.
- **Small files.** PNGs under about 500 KB each (`oxipng` or `pngquant` shrinks them); the GIF
  under 10 MB, GitHub's limit for an image, and better around 3 to 5 MB, about 1200 pixels wide.
  A GIF plays by itself; a video on GitHub waits for a click.

Each name above is the one the README already uses, so a file saved under it appears in the
README with no other change.

## The GIF, step by step

`demo.py lab`'s model writes the keys the check wants only when the skill holds the exact words
`row_count and amount_total`. Any other wording, such as the names in backticks or a comma
between them, gets the wrong keys in every sample, and the two arms come out the same. It has no
randomness either: every sample of an arm is the same answer.

1. Start `python tests/browser/demo.py lab ~/tuningfork-lab 8790` on a new folder (it seeds only
   an empty one), and open the address it prints.
2. **Runs** → run `tally-report/1#demo.2` → step 1 → **A/B a wording…**
3. In the editor, change `Write the totals to report.json.` to
   `Write the totals to report.json, using the keys row_count and amount_total.`
4. For the target, choose *What the model does matches a pattern* and type `row_count`. Keep the
   suggested name and 20 samples per arm, and start it.
5. Its job's page links the result: the variant's edit as a diff, then 0/20 against 20/20, Fisher
   exact p = 1.5e-11. Equal arms mean the sentence differs from the one above.
