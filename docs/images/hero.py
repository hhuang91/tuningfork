"""The README's opening figure: the loop in four parts, told by a toy task's real runs.

The runs are examples/find-the-file's: Gemma 4 31B through Hermes Agent, a tally-report skill told
to stop if its file is not there, and data.csv in a subfolder. The example shows every number below
with the command that made it. The figure shows counts alone; the example gives each rate its
interval and its test.

1. Ten runs: 4 pass, drawn in the order they ran. Every run read the named path first (or scripted
   it) and found nothing; the 4 that then searched with search_files passed, the 6 that gave up
   failed.
2. Step 1, the call after that read, sent again 20 times with everything before it held fixed:
   gives up 16, searches 4. The pie is the model's own odds for the word after "...but it was not
   found." in a run that searched: "According" 80%, which in a run that gave up went on "...I
   should say so and stop."; "Wait" 7%, which in this run went on "...search for it first?"; any
   other word 12%. "Wait" is a forking token there: given 40 continuations each, the model
   searched in 30 after "Wait" and in none after a forced "According" (the example's section 4).
3. One line in the skill: "Find the CSV file named in the request with search_files, then read
   it." At step 1: searched 4 times in 20 as written, 20 in 20 with the edit.
4. Ten runs with the edit: all 10 pass.

    python3 docs/images/hero.py docs/images     # writes hero-light.svg and hero-dark.svg
"""
import math
import sys
from pathlib import Path

W, H = 960, 582

PALETTES = {
    "light": {
        "text": "#1d1d1b", "muted": "#62625c", "faint": "#93938b",
        "panel": "#ffffff", "panel2": "#f6f6f3", "border": "#cacac1", "frame": "#e1e1da",
        "warn": "#9a6a00", "warn_soft": "#fbf1d4",
        "ok": "#247a3a", "ok_soft": "#e2f3e6", "bad": "#c0302b", "bad_soft": "#fbe7e5",
    },
    "dark": {
        "text": "#e9e9e4", "muted": "#a5a59c", "faint": "#75756d",
        "panel": "#1c1c1a", "panel2": "#141412", "border": "#45453f", "frame": "#2f2f2b",
        "warn": "#e2b44d", "warn_soft": "#372d15",
        "ok": "#62c27b", "ok_soft": "#1b3122", "bad": "#f07b72", "bad_soft": "#3d1f1d",
    },
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
NBSP = "\u00a0"  # a space SVG keeps where it collapses others: the diff's second line's indent

# The runs, as examples/find-the-file shows them.
RUNS = [True, False, False, False, True, True, True, False, False, False]  # runs 0-9, in order
AT_STEP = {"as written": (16, 4), "with the edit": (0, 20)}  # gives up, searches: of 20
# The word after "…but it was not found.", its slices clockwise from about three o'clock: the
# word, its odds, its slice, and what it went on to say.
NEXT_WORD = [("Wait", 0.072, "pie-ok", "“…search for it first?”", "after-ok"),
             ("others", 0.124, "pie-other", None, None),
             ("According", 0.804, "pie-bad", "“…say so and stop.”", "after-bad")]


def style(p):
    return f"""
    .frame {{ fill: {p['panel2']}; stroke: {p['frame']}; stroke-width: 1; }}
    .spine {{ stroke: {p['frame']}; stroke-width: 2; }}
    .badge {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .badge-t {{ font-family: {SANS}; font-size: 12px; font-weight: 700; fill: {p['muted']}; }}
    .lane {{ font-family: {SANS}; font-size: 16px; font-weight: 650; fill: {p['text']}; }}
    .lane-sub {{ font-family: {SANS}; font-size: 13px; fill: {p['muted']}; }}
    .chip-bad {{ fill: {p['bad_soft']}; stroke: {p['bad']}; stroke-width: 1.2; }}
    .chip-ok {{ fill: {p['ok_soft']}; stroke: {p['ok']}; stroke-width: 1.2; }}
    .mark-bad {{ font-family: {SANS}; font-size: 13px; font-weight: 700; fill: {p['bad']}; }}
    .mark-ok {{ font-family: {SANS}; font-size: 13px; font-weight: 700; fill: {p['ok']}; }}
    .rate {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['text']}; }}
    .rate-ok {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['ok']}; }}
    .box {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .box-t {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .box-now {{ fill: {p['panel']}; stroke: {p['warn']}; stroke-width: 1.4; }}
    .box-now-t {{ font-family: {SANS}; font-size: 12.5px; font-weight: 650; fill: {p['text']}; }}
    .bp {{ fill: {p['warn']}; }}
    .bp-ring {{ fill: {p['warn_soft']}; }}
    .times {{ font-family: {SANS}; font-size: 12.5px; font-weight: 650; fill: {p['muted']}; }}
    .axis {{ stroke: {p['border']}; stroke-width: 1.2; }}
    .bar-bad {{ fill: {p['bad']}; opacity: .85; }}
    .bar-ok {{ fill: {p['ok']}; opacity: .85; }}
    .count {{ font-family: {SANS}; font-size: 13px; font-weight: 700; fill: {p['text']}; }}
    .cat {{ font-family: {SANS}; font-size: 12.5px; fill: {p['text']}; }}
    .cat-tool {{ font-family: {MONO}; font-size: 12.5px; fill: {p['text']}; }}
    .speaker {{ font-family: {SANS}; font-size: 12px; font-weight: 700; fill: {p['muted']}; }}
    .reason-t {{ font-family: {MONO}; font-size: 12.5px; fill: {p['text']}; }}
    .slot {{ fill: none; stroke: {p['warn']}; stroke-width: 1.4; stroke-dasharray: 4 3; }}
    .fork {{ stroke: {p['warn']}; stroke-width: 1.3; fill: none; opacity: .8; }}
    .forkhead {{ fill: {p['warn']}; opacity: .8; }}
    .pie-bad {{ fill: {p['bad']}; opacity: .85; }}
    .pie-ok {{ fill: {p['ok']}; opacity: .85; }}
    .pie-other {{ fill: {p['faint']}; opacity: .55; }}
    .pie-edge {{ fill: none; stroke: {p['panel2']}; stroke-width: 1.5; }}
    .leader {{ stroke: {p['faint']}; stroke-width: 1; fill: none; }}
    .word {{ font-family: {MONO}; font-size: 12.5px; font-weight: 700; fill: {p['text']}; }}
    .word-other {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .after-bad {{ font-family: {SANS}; font-size: 12.5px; fill: {p['bad']}; }}
    .after-ok {{ font-family: {SANS}; font-size: 12.5px; fill: {p['ok']}; }}
    .arrow {{ stroke: {p['faint']}; stroke-width: 1.3; fill: none; }}
    .arrowhead {{ fill: {p['faint']}; }}
    .diff {{ fill: {p['ok_soft']}; stroke: {p['ok']}; stroke-width: 1.2; }}
    .diff-t {{ font-family: {MONO}; font-size: 12.5px; fill: {p['ok']}; }}
    .diff-em {{ font-weight: 700; text-decoration: underline; }}
    .doc {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .doc-fold {{ fill: {p['frame']}; stroke: {p['border']}; stroke-width: 1.2; stroke-linejoin: round; }}
    .doc-line {{ stroke: {p['frame']}; stroke-width: 3; stroke-linecap: round; }}
    .doc-line-new {{ stroke: {p['ok']}; stroke-width: 3; stroke-linecap: round; opacity: .8; }}
    .doc-t {{ font-family: {SANS}; font-size: 11.5px; font-weight: 650; fill: {p['muted']}; }}
    .note {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    """


def text(x, y, cls, body, anchor="start"):
    return f'<text x="{x:g}" y="{y:g}" class="{cls}" text-anchor="{anchor}">{body}</text>'


def arrowhead(x, y, cls="arrowhead", towards="right"):
    """A small filled triangle whose tip is at (x, y)."""
    if towards == "right":
        return f'<path d="M {x:g} {y:g} L {x - 7:g} {y - 4:g} L {x - 7:g} {y + 4:g} z" class="{cls}"/>'
    return f'<path d="M {x:g} {y:g} L {x - 4:g} {y - 7:g} L {x + 4:g} {y - 7:g} z" class="{cls}"/>'


def arrow(x1, y, x2):
    return f'<line x1="{x1:g}" y1="{y:g}" x2="{x2 - 6:g}" y2="{y:g}" class="arrow"/>' + arrowhead(x2, y)


def double_arrow(x1, y, x2):
    """⇒, drawn: two rails and an open head."""
    return (f'<path d="M {x1:g} {y - 3:g} L {x2 - 5:g} {y - 3:g} M {x1:g} {y + 3:g} L {x2 - 5:g} {y + 3:g} '
            f'M {x2 - 9:g} {y - 8:g} L {x2:g} {y:g} L {x2 - 9:g} {y + 8:g}" class="arrow"/>')


def lane(y, number, title, sub):
    return (f'<circle cx="34" cy="{y - 5}" r="12" class="badge"/>'
            + text(34, y - 1, "badge-t", number, "middle")
            + text(58, y, "lane", title) + text(58, y + 20, "lane-sub", sub))


def chips(x0, cy, marks):
    out = []
    for i, ok in enumerate(marks):
        x = x0 + 30 * i
        out.append(f'<rect x="{x}" y="{cy - 11}" width="22" height="22" rx="5" class="{"chip-ok" if ok else "chip-bad"}"/>')
        out.append(text(x + 11, cy + 5, "mark-ok" if ok else "mark-bad", "✓" if ok else "✗", "middle"))
    return out, x0 + 30 * len(marks) - 8


def stack(x, y, w, labels):
    """The run up to the failed step, a box each; the last is that step, with its breakpoint."""
    out, h, gap = [], 22, 6
    for i, label in enumerate(labels):
        top = y + i * (h + gap)
        last = i == len(labels) - 1
        out.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="5" class="{"box-now" if last else "box"}"/>')
        out.append(text(x + 10, top + 15.5, "box-now-t" if last else "box-t", label))
        if last:
            out.append(f'<circle cx="{x - 14}" cy="{top + h / 2:g}" r="8" class="bp-ring"/>')
            out.append(f'<circle cx="{x - 14}" cy="{top + h / 2:g}" r="5" class="bp"/>')
    return out, y + len(labels) * (h + gap) - gap


def bars(x, base, counts, scale=3.5):
    """Two columns, gives up and search_files, each with its count above it and its name below."""
    out = [f'<line x1="{x}" y1="{base}" x2="{x + 180}" y2="{base}" class="axis"/>']
    for i, (n, cls, name, ncls) in enumerate([(counts[0], "bar-bad", "gives up", "cat"),
                                             (counts[1], "bar-ok", "search_files", "cat-tool")]):
        cx = x + 45 + 90 * i
        if n:
            out.append(f'<rect x="{cx - 21}" y="{base - n * scale:g}" width="42" height="{n * scale:g}" rx="3" class="{cls}"/>')
        out.append(text(cx, base - n * scale - 7, "count", str(n), "middle"))
        out.append(text(cx, base + 18, ncls, name, "middle"))
    return out


def rim(cx, cy, r, angle):
    """The point at `angle` degrees clockwise from twelve o'clock on a circle."""
    t = math.radians(angle)
    return cx + r * math.sin(t), cy - r * math.cos(t)


def pie(cx, cy, r, shares, classes, start):
    """Slices clockwise from `start`, in the order given; returns each slice's middle angle."""
    out, mids, a = [], [], start
    for share, cls in zip(shares, classes):
        a1 = a + 360 * share / sum(shares)
        (x0, y0), (x1, y1) = rim(cx, cy, r, a), rim(cx, cy, r, a1)
        d = f'M {cx} {cy} L {x0:.2f} {y0:.2f} A {r} {r} 0 {int(a1 - a > 180)} 1 {x1:.2f} {y1:.2f} z'
        out.append(f'<path d="{d}" class="{cls}"/><path d="{d}" class="pie-edge"/>')
        mids.append((a + a1) / 2)
        a = a1
    return out, mids


def doc(x, y, w, h, label):
    """SKILL.md: a page with its corner folded, its new line in green."""
    f = 14
    out = [f'<path d="M {x} {y} L {x + w - f} {y} L {x + w} {y + f} L {x + w} {y + h} L {x} {y + h} z" class="doc"/>',
           f'<path d="M {x + w - f} {y} L {x + w - f} {y + f} L {x + w} {y + f}" class="doc-fold"/>',
           text(x + w / 2, y + 30, "doc-t", label, "middle")]
    for i, cls in enumerate(["doc-line", "doc-line-new", "doc-line", "doc-line"]):
        ly = y + 44 + 10 * i
        out.append(f'<line x1="{x + 11}" y1="{ly}" x2="{x + w - (22 if i % 2 else 11)}" y2="{ly}" class="{cls}"/>')
    return out


def figure():
    out = [f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" class="frame"/>']
    T1, T2, T3, T4 = 48, 114, 312, 492  # each lane's title baseline
    out.append(f'<line x1="34" y1="{T1 - 5}" x2="34" y2="{T4 - 5}" class="spine"/>')

    # 1 · ten runs as they ran: four pass
    out.append(lane(T1, "1", "Test a skill on your own model", "Only 4 of 10 runs pass: the skill is flaky."))
    c, end = chips(440, T1 + 4, RUNS)
    out += c
    out.append(text(end + 26, T1 + 9, "rate", "4 / 10"))

    # 2 · the step where the runs split, paused and sent again; inside one answer, the next word's odds
    out.append(lane(T2, "2", "Find the failed step and forking token", "At that step, only 4 of 20 tests pass."))
    s, bottom = stack(84, T2 + 40, 170, ["System prompt", "User message", "Step 0: file not found", "Step 1"])
    out += s
    by, mid = bottom - 11, T2 + 100  # the breakpointed step's middle; the arrows' height
    out.append(f'<path d="M 254 {by} L 274 {by} L 274 {mid} L 304 {mid}" class="arrow"/>' + arrowhead(310, mid))
    out.append(text(291, mid - 8, "times", "×20", "middle"))
    out += bars(320, T2 + 128, AT_STEP["as written"])
    out.append(arrow(512, mid, 544))
    # the agent's sentence, a blank where its next word goes, and the odds of each word that fills it
    qy = T2 + 38
    out.append(text(556, qy, "speaker", "agent:"))
    out.append(text(606, qy, "reason-t", "“…but it was not found."))
    slot_x, slot_w = 790, 60
    out.append(f'<rect x="{slot_x}" y="{qy - 14}" width="{slot_w}" height="20" rx="4" class="slot"/>')
    out.append(text(slot_x + slot_w + 3, qy, "reason-t", "”"))
    pcx, pcy, pr = 598, mid + 4, 38
    sx = slot_x + slot_w / 2
    out.append(f'<path d="M {sx:g} {qy + 6} L {sx:g} {qy + 14} L {pcx} {qy + 14} L {pcx} {pcy - pr - 9}" class="fork"/>'
               + arrowhead(pcx, pcy - pr - 2, "forkhead", "down"))
    sl, mids = pie(pcx, pcy, pr, [w[1] for w in NEXT_WORD], [w[2] for w in NEXT_WORD], start=84)
    out += sl
    # said in words, so that the pie reads as the next word's odds and not as the twenty tests
    out.append(text(pcx, pcy + pr + 18, "note", "odds of the next word", "middle"))
    lx = 664
    for (word, _, _, after, cls), m in zip(NEXT_WORD, mids):
        angle = m if m < 180 else 40  # a slice facing left is labelled from its top right
        ex, ey = rim(pcx, pcy, pr + 3, angle)
        out.append(f'<path d="M {ex:.1f} {ey:.1f} L {lx - 8} {ey:.1f}" class="leader"/>')
        if after:
            out.append(text(lx, ey + 4.5, "word", word))
            out.append(arrow(lx + 80, ey, lx + 102))
            out.append(text(lx + 108, ey + 4.5, cls, after))
        else:
            out.append(text(lx, ey + 4.5, "word-other", word))

    # 3 · one line of the skill, and the same step tested again
    out.append(lane(T3, "3", "Tune the skill to guide the model", "With the edit, all 20 tests at that step pass."))
    dy = T3 + 62  # the diff's top: the row's middle is 24 below it
    out.append(f'<rect x="84" y="{dy}" width="350" height="48" rx="7" class="diff"/>')
    out.append(text(96, dy + 20, "diff-t", "+ 1. Find the CSV file named in the request"))
    out.append(text(96, dy + 38, "diff-t",
                    NBSP * 5 + 'with <tspan class="diff-em">search_files</tspan>, then read it. …'))
    out.append(arrow(446, dy + 24, 482))
    out += doc(494, dy - 18, 78, 84, "SKILL.md")
    out.append(double_arrow(588, dy + 24, 628))
    out.append(text(601, dy + 11, "times", "×20", "middle"))
    out += bars(642, T3 + 118, AT_STEP["with the edit"])
    out.append(text(814, T3 + 87, "note", "at step 1"))

    # 4 · ten full runs with the edit
    out.append(lane(T4, "4", "Verify the new skill with full runs", "All 10 new runs pass, start to finish."))
    c, end = chips(440, T4 + 4, [True] * 10)
    out += c
    out.append(text(end + 26, T4 + 9, "rate-ok", "10 / 10"))

    out.append(f'<line x1="24" y1="{H - 44}" x2="{W - 24}" y2="{H - 44}" class="spine"/>')
    out.append(text(W - 24, H - 18, "note",
                    "Gemma 4 31B through Hermes Agent · every number from the runs of a toy task made to fail one way",
                    "end"))
    return "\n  ".join(out)


def svg(theme):
    p = PALETTES[theme]
    desc = ("The loop in four parts. 1, test a skill on your own model: only 4 of 10 runs pass, so the skill is "
            "flaky. 2, find the failed step and forking token: the run is paused at step 1, after a step 0 that "
            "found no file, and that step is tested 20 times. The model gives up 16 times and searches for the file 4 "
            "times. A pie shows the model's odds at the forking token, the word after 'but it was not found': most "
            "likely 'According', which goes on 'say so and stop', and seldom 'Wait', which goes on 'search for it "
            "first?'. 3, tune the skill to guide the model: with one line of SKILL.md saying to find the file with "
            "search_files, all 20 tests at that step pass. 4, verify the new skill with full runs: all 10 new runs "
            "pass, start to finish.")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">tuningfork, in one picture</title>
  <desc id="desc">{desc}</desc>
  <style>{style(p)}</style>
  {figure()}
</svg>
"""


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        (folder / f"hero-{theme}.svg").write_text(svg(theme), encoding="utf-8")
        print(folder / f"hero-{theme}.svg")
