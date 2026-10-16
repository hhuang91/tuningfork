"""The README's opening figure, told through the model's own odds: a toy task's real runs.

The runs are examples/find-the-file's, where every number below is shown with the command that
made it: Gemma 4 31B through Hermes Agent, a tally-report skill told to stop if its file is not
there, and data.csv in a subfolder.

- 10 runs: 4 pass [17-69%]. Every run read the named path first (or scripted it) and found
  nothing; the 4 that then searched with search_files passed, the 6 that gave up failed.
- That step, sent again 20 times with its history held fixed: gives up 16, searches 4.
- Its reasoning, scored: after "...but it was not found." the next word was "According" 80%
  ("...to the skill, I should say so and stop") or "Wait" 7% ("...search for it first?").
- One line in the skill: "Find the CSV file named in the request with search_files, then read
  it." At that step: searched 4/20 as written, 20/20 with the edit, Fisher p = 1.5e-7.
- 10 runs with the edit: 10/10 [72-100%], Fisher p = 0.011 against 4/10.

    python3 docs/images/hero.py docs/images     # writes hero-light.svg and hero-dark.svg
"""
import sys
from pathlib import Path

W, H = 960, 486

PALETTES = {
    "light": {
        "text": "#1d1d1b", "muted": "#62625c", "faint": "#93938b",
        "panel": "#ffffff", "panel2": "#f6f6f3", "border": "#cacac1", "frame": "#e1e1da",
        "accent": "#4353e0", "accent_soft": "#e8eaff", "warn": "#9a6a00", "warn_soft": "#fbf1d4",
        "ok": "#247a3a", "ok_soft": "#e2f3e6", "bad": "#c0302b", "bad_soft": "#fbe7e5",
    },
    "dark": {
        "text": "#e9e9e4", "muted": "#a5a59c", "faint": "#75756d",
        "panel": "#1c1c1a", "panel2": "#141412", "border": "#45453f", "frame": "#2f2f2b",
        "accent": "#8b95ff", "accent_soft": "#262a4d", "warn": "#e2b44d", "warn_soft": "#372d15",
        "ok": "#62c27b", "ok_soft": "#1b3122", "bad": "#f07b72", "bad_soft": "#3d1f1d",
    },
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def style(p):
    return f"""
    .frame {{ fill: {p['panel2']}; stroke: {p['frame']}; stroke-width: 1; }}
    .spine {{ stroke: {p['frame']}; stroke-width: 2; }}
    .badge {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .badge-t {{ font-family: {SANS}; font-size: 12px; font-weight: 700; fill: {p['muted']}; }}
    .lane {{ font-family: {SANS}; font-size: 15px; font-weight: 650; fill: {p['text']}; }}
    .lane-sub {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .chip-bad {{ fill: {p['bad_soft']}; stroke: {p['bad']}; stroke-width: 1.2; }}
    .chip-ok {{ fill: {p['ok_soft']}; stroke: {p['ok']}; stroke-width: 1.2; }}
    .mark-bad {{ font-family: {SANS}; font-size: 13px; font-weight: 700; fill: {p['bad']}; }}
    .mark-ok {{ font-family: {SANS}; font-size: 13px; font-weight: 700; fill: {p['ok']}; }}
    .brace-bad {{ stroke: {p['bad']}; stroke-width: 1.4; fill: none; }}
    .brace-ok {{ stroke: {p['ok']}; stroke-width: 1.4; fill: none; }}
    .gave-up {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['bad']}; }}
    .gave-up-l {{ font-family: {SANS}; font-size: 13px; fill: {p['text']}; }}
    .tool-ok {{ font-family: {MONO}; font-size: 13px; fill: {p['ok']}; }}
    .tool {{ font-family: {MONO}; font-size: 13px; fill: {p['text']}; }}
    .rate {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['text']}; }}
    .rate-ok {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['ok']}; }}
    .ci {{ font-family: {SANS}; font-size: 12px; fill: {p['muted']}; }}
    .pct {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['text']}; }}
    .bar-bad {{ fill: {p['bad']}; opacity: .85; }}
    .bar-ok {{ fill: {p['ok']}; opacity: .85; }}
    .track {{ fill: {p['frame']}; }}
    .card {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .card-k {{ font-family: {SANS}; font-size: 11.5px; fill: {p['muted']}; letter-spacing: .02em; }}
    .reason {{ font-family: {MONO}; font-size: 12.5px; fill: {p['muted']}; }}
    .reason-t {{ font-family: {MONO}; font-size: 12.5px; fill: {p['text']}; }}
    .speaker {{ font-family: {SANS}; font-size: 12px; font-weight: 700; fill: {p['muted']}; }}
    .slot {{ fill: none; stroke: {p['warn']}; stroke-width: 1.4; stroke-dasharray: 4 3; }}
    .fork {{ stroke: {p['warn']}; stroke-width: 1.3; fill: none; opacity: .8; }}
    .token {{ fill: {p['warn_soft']}; stroke: {p['warn']}; stroke-width: 1.3; }}
    .token-t {{ font-family: {MONO}; font-size: 12.5px; font-weight: 700; fill: {p['text']}; }}
    .alt-pct {{ font-family: {SANS}; font-size: 12.5px; font-weight: 600; fill: {p['muted']}; }}
    .after-bad {{ font-family: {SANS}; font-size: 12.5px; fill: {p['bad']}; }}
    .after-ok {{ font-family: {SANS}; font-size: 12.5px; fill: {p['ok']}; }}
    .arrow {{ stroke: {p['faint']}; stroke-width: 1.3; fill: none; }}
    .arrowhead {{ fill: {p['faint']}; }}
    .diff {{ fill: {p['ok_soft']}; stroke: {p['ok']}; stroke-width: 1.2; }}
    .diff-t {{ font-family: {MONO}; font-size: 12.5px; fill: {p['ok']}; }}
    .diff-file {{ font-family: {SANS}; font-size: 11.5px; fill: {p['muted']}; letter-spacing: .02em; }}
    .note {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .pvalue {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['text']}; }}
    """


def text(x, y, cls, body, anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{body}</text>'


def arrow(x1, y, x2):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}" class="arrow"/>'
            f'<path d="M {x2} {y} L {x2 - 7} {y - 4} L {x2 - 7} {y + 4} z" class="arrowhead"/>')


def lane(y, number, title, sub):
    return (f'<circle cx="34" cy="{y - 5}" r="12" class="badge"/>'
            + text(34, y - 1, "badge-t", number, "middle")
            + text(58, y, "lane", title) + text(58, y + 19, "lane-sub", sub))


def chips(x0, cy, marks, gap_after=None, gap=14):
    out, xs = [], []
    x = x0
    for i, ok in enumerate(marks):
        out.append(f'<rect x="{x}" y="{cy - 11}" width="22" height="22" rx="5" class="{"chip-ok" if ok else "chip-bad"}"/>')
        out.append(text(x + 11, cy + 5, "mark-ok" if ok else "mark-bad", "✓" if ok else "✗", "middle"))
        xs.append(x)
        x += 30 + (gap if gap_after is not None and i == gap_after else 0)
    return out, xs


def brace(x1, x2, y, cls):
    """A bracket under a run of chips: down at each end, a tick at the middle."""
    mid = (x1 + x2) / 2
    return (f'<path d="M {x1} {y} L {x1} {y + 6} L {x2} {y + 6} L {x2} {y} '
            f'M {mid} {y + 6} L {mid} {y + 11}" class="{cls}"/>')


def figure(p):
    out = [f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" class="frame"/>',
           f'<line x1="34" y1="54" x2="34" y2="392" class="spine"/>']
    L = 262  # where the lanes' content starts

    # 1 · ten runs, four pass: each tried the named path and found nothing; then they split
    out.append(lane(48, "1", "10 runs, 4 pass", "they split after the read fails"))
    c, xs = chips(L, 40, [False] * 6 + [True] * 4, gap_after=5)
    out += c
    out.append(brace(xs[0], xs[5] + 22, 58, "brace-bad"))
    out.append(text((xs[0] + xs[5] + 22) / 2, 90, "gave-up", "gave up: “not found”", "middle"))
    out.append(brace(xs[6], xs[9] + 22, 58, "brace-ok"))
    out.append(text((xs[6] + xs[9] + 22) / 2, 90, "tool-ok", "search_files", "middle"))
    right = xs[9] + 22 + 34
    out.append(text(right, 45, "rate", "4 / 10"))
    out.append(text(right + 52, 45, "ci", "[17–69%]"))
    out.append(text(right + 120, 45, "note", "·  every run that searched passed"))

    # 2 · that step, sent again twenty times; and where its reasoning forks
    out.append(lane(150, "2", "The model wavers", "that step, sent again ×20"))
    for r, (label, hits, cls, tcls) in enumerate([("gives up", 16, "bar-bad", "gave-up-l"),
                                                   ("search_files", 4, "bar-ok", "tool")]):
        y = 140 + r * 28
        out.append(text(L + 100, y + 5, tcls, label, "end"))
        out.append(f'<rect x="{L + 110}" y="{y - 7}" width="160" height="14" rx="3" class="track"/>')
        out.append(f'<rect x="{L + 110}" y="{y - 7}" width="{8 * hits}" height="14" rx="3" class="{cls}"/>')
        out.append(text(L + 280, y + 5, "pct", f"{hits} / 20"))
    # The card: the agent's own sentence, a blank where its next word goes, and the words that
    # fill it — the fork drawn from the blank to each.
    cx, cy, cw, ch = 600, 100, 336, 126
    out.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="9" class="card"/>')
    out.append(text(cx + 14, cy + 20, "card-k", "WHERE THE AGENT’S REASONING FORKS"))
    out.append(text(cx + 14, cy + 46, "speaker", "agent:"))
    quote_x = cx + 58
    out.append(text(quote_x, cy + 46, "reason-t", "“…but it was not found."))
    slot_x, slot_w = cx + 236, 70
    out.append(f'<rect x="{slot_x}" y="{cy + 32}" width="{slot_w}" height="20" rx="4" class="slot"/>')
    out.append(text(slot_x + slot_w + 3, cy + 46, "reason-t", "”"))
    # the fork: down from the blank, left, and down to each word that can fill it
    sx, top, trunk = slot_x + slot_w / 2, cy + 52, cx + 22
    rows = [cy + 84, cy + 108]
    out.append(f'<path d="M {sx} {top} L {sx} {top + 9} L {trunk} {top + 9} L {trunk} {rows[-1] - 4}" class="fork"/>')
    for r, (word, pct, after, cls) in enumerate([("According", "80%", "“…say so and stop.”", "after-bad"),
                                                 ("Wait", "7%", "“…search for it first?”", "after-ok")]):
        y = rows[r]
        out.append(f'<path d="M {trunk} {y - 4} L {cx + 32} {y - 4}" class="fork"/>')
        out.append(f'<rect x="{cx + 32}" y="{y - 14}" width="84" height="20" rx="4" class="token"/>')
        out.append(text(cx + 74, y, "token-t", word, "middle"))
        out.append(text(cx + 124, y, "alt-pct", pct))
        out.append(arrow(cx + 154, y - 4, cx + 174))
        out.append(text(cx + 180, y, cls, after))

    # 3 · one line of the skill, tested at that step
    out.append(lane(254, "3", "One line in the skill", "a few words tip it"))
    out.append(text(L, 234, "diff-file", "SKILL.md"))
    out.append(f'<rect x="{L}" y="242" width="604" height="30" rx="7" class="diff"/>')
    out.append(text(L + 12, 262, "diff-t", "+ 1. Find the CSV file named in the request with search_files, then read it."))
    out.append(text(L, 298, "pvalue",
                    'At that step: searched 4 / 20 as written, 20 / 20 with the edit'
                    '<tspan class="note" dx="10">·  Fisher’s exact p = 1.5 × 10⁻⁷</tspan>'))

    # 4 · ten full runs with the edit
    out.append(lane(368, "4", "10 runs with the edit", "the same task, the same check"))
    c, xs = chips(L, 360, [True] * 10)
    out += c
    right = xs[9] + 22 + 34
    out.append(text(right, 365, "rate-ok", "10 / 10"))
    out.append(text(right + 62, 365, "ci", "[72–100%]"))
    out.append(text(L, 402, "pvalue",
                    'Fisher’s exact p = 0.011 against 4 / 10<tspan class="note" dx="10">·  every rate '
                    'with its 95% interval</tspan>'))

    out.append(f'<line x1="24" y1="{H - 44}" x2="{W - 24}" y2="{H - 44}" class="spine"/>')
    out.append(text(W - 24, H - 18, "note",
                    "Gemma 4 31B through Hermes Agent · every number from the runs of a toy task made to fail one way",
                    "end"))
    return "\n  ".join(out)


def svg(theme):
    p = PALETTES[theme]
    desc = ("Ten runs of a skill pass 4 times in 10 [17-69%]: every run that searched for the file with "
            "search_files passed, every run that gave up when its read found nothing failed. That step, sent "
            "again 20 times with its history held fixed: it gives up 16 times and searches 4. Its reasoning "
            "forks after 'it was not found': 'According' (80%) goes on to 'I should say so and stop', 'Wait' "
            "(7%) to 'search for it first?'. One line in the skill: find the file with search_files, then read "
            "it. At that step the model then searched 20 times in 20 (Fisher p = 1.5e-7), and 10 full runs "
            "pass 10 times in 10 [72-100%], Fisher p = 0.011 against 4 in 10.")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">tuningfork, in one picture</title>
  <desc id="desc">{desc}</desc>
  <style>{style(p)}</style>
  {figure(p)}
</svg>
"""


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        (folder / f"hero-{theme}.svg").write_text(svg(theme), encoding="utf-8")
        print(folder / f"hero-{theme}.svg")
