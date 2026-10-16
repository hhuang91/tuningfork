"""Writes docs/images/hero-light.svg and hero-dark.svg: the README's opening figure.

The story is the guide's, with its real numbers (Gemma 4 31B, examples/README.md Part 2):
a run fails at its write step; that step, sampled twenty times with its history held fixed,
names the keys the check wants 0/20 as written and 20/20 with the edit (Fisher p = 1.5e-11);
full runs with the edit pass 8/8, against 0/8 before.

    python3 hero.py <docs/images folder>
"""
import sys
from pathlib import Path

W, H = 960, 456

PALETTES = {
    "light": {
        "text": "#1d1d1b", "muted": "#62625c", "faint": "#93938b",
        "panel": "#ffffff", "panel2": "#f6f6f3", "border": "#cacac1", "frame": "#e1e1da",
        "accent": "#4353e0", "accent_soft": "#e8eaff",
        "ok": "#247a3a", "ok_soft": "#e2f3e6", "bad": "#c0302b", "bad_soft": "#fbe7e5",
    },
    "dark": {
        "text": "#e9e9e4", "muted": "#a5a59c", "faint": "#75756d",
        "panel": "#1c1c1a", "panel2": "#141412", "border": "#45453f", "frame": "#2f2f2b",
        "accent": "#8b95ff", "accent_soft": "#262a4d",
        "ok": "#62c27b", "ok_soft": "#1b3122", "bad": "#f07b72", "bad_soft": "#3d1f1d",
    },
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def style(p):
    return f"""
    .sans {{ font-family: {SANS}; }}
    .mono {{ font-family: {MONO}; }}
    .num {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['muted']}; }}
    .lane {{ font-family: {SANS}; font-size: 15px; font-weight: 650; fill: {p['text']}; }}
    .lane-sub {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .step {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
    .step-hot {{ fill: {p['accent_soft']}; stroke: {p['accent']}; stroke-width: 2; }}
    .step-k {{ font-family: {SANS}; font-size: 11.5px; fill: {p['muted']}; letter-spacing: .02em; }}
    .step-t {{ font-family: {MONO}; font-size: 13.5px; fill: {p['text']}; }}
    .wrong {{ font-family: {MONO}; font-size: 12.5px; fill: {p['bad']}; }}
    .arrow {{ stroke: {p['faint']}; stroke-width: 1.4; fill: none; }}
    .arrowhead {{ fill: {p['faint']}; }}
    .rail {{ stroke: {p['accent']}; stroke-width: 1.6; fill: none; stroke-dasharray: 4 4; }}
    .railhead {{ fill: {p['accent']}; }}
    .verdict-bad {{ fill: {p['bad_soft']}; stroke: {p['bad']}; stroke-width: 1.2; }}
    .verdict-ok {{ fill: {p['ok_soft']}; stroke: {p['ok']}; stroke-width: 1.2; }}
    .t-bad {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['bad']}; }}
    .t-ok {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['ok']}; }}
    .dot-bad {{ fill: {p['bad_soft']}; stroke: {p['bad']}; stroke-width: 1.3; }}
    .dot-ok {{ fill: {p['ok']}; }}
    .arm {{ font-family: {SANS}; font-size: 13px; fill: {p['text']}; }}
    .rate-bad {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['bad']}; }}
    .rate-ok {{ font-family: {SANS}; font-size: 15px; font-weight: 700; fill: {p['ok']}; }}
    .ci {{ font-family: {SANS}; font-size: 12px; fill: {p['muted']}; }}
    .note {{ font-family: {SANS}; font-size: 12.5px; fill: {p['muted']}; }}
    .pvalue {{ font-family: {SANS}; font-size: 13px; font-weight: 600; fill: {p['text']}; }}
    .rule {{ stroke: {p['frame']}; stroke-width: 1; }}
    .frame {{ fill: {p['panel2']}; stroke: {p['frame']}; stroke-width: 1; }}
    """


def text(x, y, cls, body, anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{body}</text>'


def arrow(x1, y, x2):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" class="arrow"/>'
            f'<path d="M {x2} {y} L {x2 - 8} {y - 4.5} L {x2 - 8} {y + 4.5} z" class="arrowhead"/>')


def lane(y, number, title, sub):
    return (text(24, y, "num", number) + text(46, y, "lane", title)
            + text(46, y + 19, "lane-sub", sub))


def figure(p):
    out = [f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" class="frame"/>']
    LEFT = 210  # where the lanes' content starts

    # 1 · a recorded run that fails at its write step
    out.append(lane(52, "1", "A recorded run", "every call, its exact context"))
    y, h = 26, 62
    boxes = [("STEP 0", "read data.csv", 156, False),
             ("STEP 1", "write report.json", 210, True),
             ("STEP 2", "reply", 104, False)]
    x = LEFT
    centres = {}
    for i, (k, t, w, hot) in enumerate(boxes):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
                   f'class="{"step-hot" if hot else "step"}"/>')
        out.append(text(x + 14, y + 22, "step-k", k))
        out.append(text(x + 14, y + 44, "step-t", t))
        centres[i] = x + w / 2
        if i < len(boxes) - 1:
            out.append(arrow(x + w + 6, y + h / 2, x + w + 34))
        x += w + 40
    # the verdict
    out.append(f'<rect x="{x - 4}" y="{y + 14}" width="118" height="34" rx="17" class="verdict-bad"/>')
    out.append(text(x + 55, y + 36, "t-bad", "check fails", "middle"))
    out.append(text(LEFT + 156 + 40 + 105, y + h + 22, "wrong", '{"count": 4, "total": 61.5}', "middle"))
    out.append(text(x + 55, y + h + 22, "note", "wants row_count", "middle"))

    # 2 · that step, sampled twenty times, history held fixed
    hot = centres[1]
    top, bottom = y + h + 34, 174
    out.append(f'<path d="M {hot} {top} L {hot} {bottom}" class="rail"/>')
    out.append(f'<path d="M {hot} {bottom + 9} L {hot - 5} {bottom} L {hot + 5} {bottom} z" class="railhead"/>')
    out.append(lane(206, "2", "That step, sent again ×20", "everything before it held fixed"))
    rows = [("as written", "dot-bad", "rate-bad", "0 / 20", "[0–16%]", 0),
            ("with your edit", "dot-ok", "rate-ok", "20 / 20", "[84–100%]", 20)]
    for r, (arm, dot, rate, count, ci, hits) in enumerate(rows):
        cy = 196 + r * 40
        out.append(text(LEFT + 112, cy + 5, "arm", arm, "end"))
        for d in range(20):
            cx = LEFT + 132 + d * 21
            out.append(f'<circle cx="{cx}" cy="{cy}" r="7.5" class="{dot}"/>')
        out.append(text(LEFT + 132 + 20 * 21 + 14, cy + 5, rate, count))
        out.append(text(LEFT + 132 + 20 * 21 + 82, cy + 5, "ci", ci))
    out.append(text(LEFT + 132, 296, "pvalue",
                    'Fisher’s exact p = 1.5 × 10⁻¹¹<tspan class="note" dx="12">·  the target, named before '
                    'the first sample: the keys the check wants</tspan>'))

    # 3 · full runs with the edit
    out.append(f'<line x1="24" y1="324" x2="{W - 24}" y2="324" class="rule"/>')
    out.append(lane(366, "3", "Then full runs", "the edit in every run, from step 0"))
    rows = [("before", "verdict-bad", "rate-bad", "0 / 8", "[0–32%]"),
            ("with the edit", "verdict-ok", "rate-ok", "8 / 8", "[68–100%]")]
    for r, (arm, chip, rate, count, ci) in enumerate(rows):
        cy = 354 + r * 40
        out.append(text(LEFT + 112, cy + 5, "arm", arm, "end"))
        for d in range(8):
            cx = LEFT + 132 + d * 30
            out.append(f'<rect x="{cx - 11}" y="{cy - 11}" width="22" height="22" rx="5" class="{chip}"/>')
            mark = "✓" if chip == "verdict-ok" else "✗"
            out.append(text(cx, cy + 5, "t-ok" if chip == "verdict-ok" else "t-bad", mark, "middle"))
        out.append(text(LEFT + 132 + 8 * 30 + 4, cy + 5, rate, count))
        out.append(text(LEFT + 132 + 8 * 30 + 62, cy + 5, "ci", ci))
    out.append(text(W - 24, H - 18, "note", "Gemma 4 31B through Hermes Agent: the step-by-step guide’s own runs", "end"))
    return "\n  ".join(out)


def svg(theme):
    p = PALETTES[theme]
    title = "tuningfork, in one picture"
    desc = ("A recorded run fails at its write step, where the model names the report's keys count "
            "and total; the check wants row_count. That step, sent to the model twenty times with "
            "everything before it held fixed: as written, 0 of 20 name the keys the check wants "
            "[0-16%]; with an edit that names them, 20 of 20 [84-100%], Fisher's exact p = 1.5e-11. "
            "Full runs with the edit: 8 of 8 pass [68-100%], against 0 of 8 [0-32%] before.")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
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
