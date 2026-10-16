# Images

The pictures in the top-level README.

- **`hero-light.svg`, `hero-dark.svg`** — the opening figure, drawn by `hero.py` from the
  numbers of [the guide's](../../examples/README.md) own runs (Gemma 4 31B through Hermes Agent,
  Part 2 §3, §6 and §8), in the web UI's colours. GitHub shows the dark one in a dark theme.
  After a change to the script or the numbers: `python3 docs/images/hero.py docs/images`.
- **`ui-*.png`, `ui-demo.gif`** — the web UI, taken by hand as listed below.

## Screenshots

| File | What it shows | Taken from |
|---|---|---|
| `ui-demo.gif` | A run opened, its write step chosen, an edited wording A/B-tested there, and the result: the original arm against the edit, with intervals and Fisher's test. 30 seconds at most, no sound | `demo.py lab` |
| `ui-guide.png` | The front page: what is set up, the loop from a skill to its next version, and the questions each way of looking at a step answers | real recordings |
| `ui-run.png` | A run as one conversation, the write step open, the skill marked in the context | real recordings |
| `ui-divergence.png` | A task's divergence page: where the runs split, how each behaviour fared, what the failed checks said | real recordings |
| `ui-ab.png` | An A/B's result at a step, both arms with their intervals and the declared target's test | real recordings |
| `ui-tokens.png` | An answer's token heatmap, one uncertain token's alternatives open | real recordings |
| `ui-refine.png` | A refinement: its target, proposals as diffs, and a screen's verdict | real recordings |
| `ui-live.png` | The Live page with a call held at a breakpoint, its tool call being edited | `demo.py live` |

**Real recordings** are runs of the guide's task on a real model, so that the numbers on the
page are ones the guide reports. **`demo.py lab`** serves seeded runs through a real proxy, in
front of a scripted model that names the report's keys only when the skill states them: an A/B
of a wording that names them shows the shift in seconds, with no GPU. **`demo.py live`** is the
same proxy with a call held at a breakpoint. Each prints the web UI's address:

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
