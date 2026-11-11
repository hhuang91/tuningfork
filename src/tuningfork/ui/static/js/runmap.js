// Paths that part, drawn: what was done at each step a box, a line between two boxes for those
// that did one and then the other, as wide as they were many. Where they did different things the
// line parts, and each path ends in how it fared.
//
// The run map draws a task's runs this way (`runTree`): a box is the tool the runs called, or a
// reply, whatever its words — a tool's calls apart where they came in a few distinct forms — and
// each path ends in its runs, one mark a run, coloured by its check. The tree, each box's pass
// rate and the turning point are the server's (`analysis.paths`): the terminal prints the same
// tree (`tuningfork divergence --paths`). An experiment's samples are drawn the same way, a fan
// out of one step (`fanTree`), and so is a choice among a few things at one place, a probe's
// tokens at a position (`choiceFan`).
//
// The map is of what is shared and where it parts. A run gone its own way is one no other run can
// be compared with from there on: on a large map its later steps are not drawn, and runs that each
// went their own way from one place are one box. A run's own page has every step.
//
// The boxes are buttons in a grid, one column a step; the lines are drawn under them, from where
// the grid put the boxes. Every string is a text node.

import { h, plural, rateBar, short } from "./dom.js";

const SVG = "http://www.w3.org/2000/svg";

/** How much the run map draws, least to most: which of a tool's calls are told apart (`forms`,
 *  the server's trees), and whether a run gone its own way has its later steps drawn (`lone`). */
export const DETAIL = [
  { forms: "tools", lone: "fold", name: "one box a tool" },
  { forms: "auto", lone: "fold", name: "shared paths" },
  { forms: "auto", lone: "draw", name: "every run" },
  { forms: "calls", lone: "draw", name: "every call" },
];
/** A map with more rows than this starts with the runs gone their own way folded. */
const ROWS_UNFOLDED = 12;

/** What to draw of a tree's nodes, and where: each node's `drawn` branches, the runs that each
 *  went their own way from it as one box (`alone`), the steps a lone run took unseen (`more`),
 *  and its `row`. Returns the tree's `rows` and `columns`. */
function arrange(root, lone) {
  const stepsAfter = (node) => {
    let count = 0;
    for (let next = node.children[0]; next; next = next.children[0]) count += 1;
    return count;
  };
  const choose = (node) => {
    node.drawn = node.children;
    node.alone = null;
    node.more = 0;
    if (lone === "fold") {
      const single = node.children.filter((child) => child.runs.length === 1);
      node.drawn = node.children.filter((child) => child.runs.length > 1);
      if (single.length === 1) node.drawn = [...node.drawn, single[0]];
      else if (single.length > 1) node.alone = { runs: single.flatMap((child) => child.runs), nodes: single };
    }
    for (const child of node.drawn) {
      if (lone === "fold" && child.runs.length === 1) {
        child.drawn = [];
        child.alone = null;
        child.more = stepsAfter(child);
      } else {
        choose(child);
      }
    }
  };
  choose(root);
  // Rows: each path's end takes one, a split's first branch going straight on. Columns: one a
  // step, as far as anything is drawn.
  let row = 0;
  let columns = 1;
  const place = (node) => {
    node.row = row;
    node.stopRow = undefined;
    node.aloneRow = undefined;
    columns = Math.max(columns, node.step + 2);
    if (!node.drawn.length && !node.alone) {
      if (node.more) columns = Math.max(columns, node.step + 3);
      row += 1;
      return;
    }
    node.drawn.forEach(place);
    if (node.alone) {
      node.aloneRow = row++;
      columns = Math.max(columns, node.step + 3);
    }
    // Those that ended here while others went on: a row of their own, under the branches.
    if (node.ends.length && node !== root) {
      node.stopRow = row++;
      columns = Math.max(columns, node.step + 3);
    }
  };
  place(root);
  return { rows: Math.max(row, 1), columns };
}

/** A divergence report's runs as the tree its map draws: the server's tree for `forms`
 *  (`report.paths`), its nodes joined up and their runs named, laid out with lone runs folded or
 *  drawn. `turning` is the server's turning point, on these nodes. */
export function runTree(report, { forms = "auto", lone = "draw" } = {}) {
  const source = report.paths?.[forms] || { nodes: [{ id: 0, parent: null, step: -1, runs: [], ends: [], forms: [], children: [] }], turning: null };
  const ids = report.run_ids || [];
  const named = (indexes) => (indexes || []).map((index) => ids[index]);
  const nodes = source.nodes.map((node) => ({
    id: node.id, step: node.step, parent: null, children: [],
    move: node.kind ? { kind: node.kind, label: node.label } : null,
    runs: named(node.runs), ends: named(node.ends), passed: node.passed, ended: node.ended,
    variants: (node.forms || []).map((form) => ({ outcome: form.outcome, runs: named(form.runs) })),
    apart: Boolean(node.apart), others: Boolean(node.others), hint: node.hint || null,
  }));
  for (const node of source.nodes) {
    nodes[node.id].parent = node.parent === null ? null : nodes[node.parent];
    nodes[node.id].children = node.children.map((child) => nodes[child]);
  }
  const root = nodes[0];
  const turning = source.turning
    ? { ...source.turning, parent: nodes[source.turning.parent], branch: source.turning.branch === null ? null : nodes[source.turning.branch] }
    : null;
  return { root, nodes, turning, verdicts: report.verdicts || {}, forms, lone, ...arrange(root, lone) };
}

/** The detail a run map starts at: every run's path while that stays a map to take in at a
 *  glance, else the shared paths. */
export function startDetail(report) {
  return runTree(report, DETAIL[2]).rows > ROWS_UNFOLDED ? 1 : 2;
}

/** How many of some runs passed and failed their check: counts, for marks and their words. A
 *  rate, with its interval, is the server's. */
export function tallyOf(runs, verdicts) {
  let passed = 0, failed = 0, uncounted = 0, none = 0;
  for (const run of runs) {
    const verdict = verdicts[run];
    if (verdict === "passed") passed += 1;
    else if (verdict === "failed") failed += 1;
    else if (verdict === "not counted") uncounted += 1;
    else none += 1;
  }
  return { passed, failed, uncounted, none, judged: passed + failed };
}

const VERDICT_MARK = {
  passed: ["ok", "passed its check"],
  failed: ["bad", "failed its check"],
  "not counted": ["warn", "not counted: left out of every pass rate"],
};

const VERDICT_ORDER = { passed: 0, failed: 1, "not counted": 2 };
const byVerdict = (runs, verdicts) => [...runs].sort((a, b) => (VERDICT_ORDER[verdicts[a]] ?? 3) - (VERDICT_ORDER[verdicts[b]] ?? 3));

/** One mark a run, coloured by its check and shaped by it too: a round one passed, a square one
 *  failed, a ring is in no rate. A picture of how the runs fared, said in words beside it: a mark
 *  is too small to press, so the runs themselves are chips (`runChips`). */
export function runMarks(runs, verdicts, { most = 40 } = {}) {
  const sorted = byVerdict(runs, verdicts);
  return h("span", { class: "rm-marks", "aria-hidden": "true" },
    sorted.slice(0, most).map((id) => {
      const [tone, said] = VERDICT_MARK[verdicts[id]] || ["none", "no verdict"];
      return h("span", { class: `rm-mark tone-${tone}`, title: `run ${short(id)}: ${said}`, dataset: { run: id } });
    }),
    sorted.length > most ? h("span", { class: "faint small" }, `+${sorted.length - most}`) : null);
}

/** The same marks from counts alone, where what ended is known by how many: a fan's forks, so
 *  many passed, so many failed, so many that no check judged. */
export function tallyMarks({ passed = 0, failed = 0, none = 0 }, { most = 40 } = {}) {
  const tones = [...Array(passed).fill("ok"), ...Array(failed).fill("bad"), ...Array(none).fill("none")];
  return h("span", { class: "rm-marks", "aria-hidden": "true" },
    tones.slice(0, most).map((tone) => h("span", { class: `rm-mark tone-${tone}` })),
    tones.length > most ? h("span", { class: "faint small" }, `+${tones.length - most}`) : null);
}

/** Runs as chips, each coloured by its own check and opening its run, at `step` where one is
 *  given: the passing ones first. */
export function runChips(runs, verdicts, { step = null, most = 16 } = {}) {
  const sorted = byVerdict(runs, verdicts);
  const chipOf = (id) => {
    const [tone, said] = VERDICT_MARK[verdicts[id]] || ["muted", "no verdict"];
    return h("a", { class: `chip tone-${tone} mono`, href: step === null ? `#/runs/${id}` : `#/runs/${id}/${step}`, title: `run ${short(id)}: ${said}` }, short(id));
  };
  return h("div", { class: "run-chips" }, sorted.slice(0, most).map(chipOf),
    sorted.length > most
      ? h("details", { class: "fold rm-more-runs" }, h("summary", null, `${sorted.length - most} more`), h("div", { class: "run-chips" }, sorted.slice(most).map(chipOf)))
      : null);
}

/** How some runs fared, in a few words: "4 of 4 passed". */
export function faredText(tally) {
  if (!tally.judged) return tally.uncounted ? `${tally.uncounted} not counted` : "no verdict";
  const left = tally.uncounted ? `, ${tally.uncounted} not counted` : "";
  return `${tally.passed} of ${tally.judged} passed${left}`;
}

/**
 * A tree of paths, drawn (`runTree`, `fanTree`: `{ root, rows, columns }`). What differs between
 * a map of runs and a fan of samples is said by the caller:
 *   heads      the words over each column, the first and the last included
 *   end(runs, node)   what a path ends in: a run map's marks, a fan's rate
 *   count(node)       the number on a box, and what its line is as wide as
 *   countText(node)   the number as the box says it: "×4", or a caller's own words
 *   describe(node)    a box's tooltip, and what a screen reader hears
 *   onSelect(node)    hears the box chosen; `turning` marks a split and its column
 * Returns its `element` and `select(node)`, which marks a box and the way to it.
 */
export function pathsFigure(tree, {
  heads, end, count = (node) => node.runs.length, countText = (node) => `×${count(node)}`, describe,
  onSelect = null, turning = null, label, total = null, className = null,
}) {
  const all = total ?? count(tree.root);
  const boxes = new Map();
  const edges = [];
  const last = tree.columns + 1;
  const grid = h("div", { class: "rm-grid", style: { "--rm-columns": String(tree.columns) } });
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "rm-edges");
  svg.setAttribute("aria-hidden", "true");
  const canvas = h("div", { class: "rm-canvas" }, svg, grid);
  const element = h("div", { class: ["rm", className], role: "group", "aria-label": label }, canvas);
  let chosen = null;

  const cell = (child, column, row) => {
    child.style.gridColumn = String(column);
    child.style.gridRow = String(row);
    grid.append(child);
    return child;
  };
  const turnColumn = turning ? turning.parent.step + 3 : null;
  for (let column = 1; column <= last; column += 1) {
    const words = column === last ? heads.end : column === 1 ? heads.start : heads.step(column - 2);
    cell(h("div", { class: ["rm-head", column === last && "rm-head-end", column === turnColumn && "rm-head-turn"] }, words,
      column === turnColumn ? h("span", { class: "rm-turn" }, "turning point") : null), column, 1);
  }

  const pick = (node) => {
    if (!onSelect) return;
    select(node);
    onSelect(node);
  };
  const box = (node) => {
    const said = describe(node);
    const inside = [
      node.badge ? h("span", { class: "rm-letter" }, node.badge) : null,
      // What a box is, where only its colour shows it: said for a screen reader.
      node.reads ? h("span", { class: "visually-hidden" }, `${node.reads}: `) : null,
      h("span", { class: "rm-label" }, node.move ? node.move.label : node.label),
      node.move ? h("span", { class: "rm-count" }, countText(node)) : null,
      node.hint ? h("span", { class: "rm-hint" }, node.hint) : null,
    ];
    const classes = ["rm-node", node.move ? `kind-${node.move.kind}` : "rm-origin", node.tone && `tone-${node.tone}`, node.badge && "has-badge", node.move && !count(node) && "rm-none"];
    const made = onSelect
      ? h("button", { class: classes, type: "button", "aria-pressed": "false", title: said, "aria-label": said, onclick: () => pick(node) }, inside)
      : h("div", { class: classes, title: said }, inside);
    boxes.set(node.id, made);
    return made;
  };
  const endCell = (runs, node, row) => cell(h("div", { class: "rm-end" }, end(runs, node)), last, row + 2);
  // What left the map here: said in a word, then how it fared.
  const stub = (node, row, runs, words, title, { button = false } = {}) => {
    const mine = boxes.get(node.id);
    const made = button && onSelect
      ? h("button", { class: "rm-node rm-alone", type: "button", title, "aria-label": title, onclick: () => pick(node) }, h("span", { class: "rm-label" }, words))
      : h("span", { class: "rm-stop", title }, words);
    cell(made, node.step + 3, row + 2);
    edges.push({ from: mine, to: made, weight: runs.length, node, aside: true, thin: !button });
    edges.push({ from: made, to: endCell(runs, null, row), weight: runs.length, leader: true, node, aside: true });
  };

  const draw = (node) => {
    const mine = cell(box(node), node.step + 2, node.row + 2);
    if (node.parent) edges.push({ from: boxes.get(node.parent.id), to: mine, weight: count(node), node });
    node.drawn.forEach(draw);
    if (!node.drawn.length && !node.alone) {
      // The end of a path, after the steps a run gone its own way took unseen.
      if (node.more) {
        stub(node, node.row, node.runs, `+${plural(node.more, "step")}`, `This run went on alone for ${plural(node.more, "more step")}: its page has them`);
      } else {
        edges.push({ from: mine, to: endCell(node.runs, node, node.row), weight: count(node), leader: true, node });
      }
      return;
    }
    if (node.alone) {
      stub(node, node.aloneRow, node.alone.runs, `${node.alone.runs.length} alone`,
        `${plural(node.alone.runs.length, "run")} each went their own way from here: choose it to see what each did`, { button: true });
    }
    if (node.stopRow !== undefined) {
      stub(node, node.stopRow, node.ends, "ended here", `${plural(node.ends.length, "run")} ended after step ${node.step}`);
    }
  };
  draw(tree.root);

  function lines() {
    const frame = canvas.getBoundingClientRect();
    if (!frame.width) return;
    // The lines' sheet is as large as the boxes' own: measured from the sheet itself (a scroll
    // width), one drawn for a wide window would keep a narrower one scrolling.
    svg.setAttribute("width", String(canvas.clientWidth));
    svg.setAttribute("height", String(canvas.clientHeight));
    svg.replaceChildren();
    // A figure wider than its place scrolls: with the keyboard too, where no box in it takes focus.
    if (!onSelect) {
      if (element.scrollWidth > element.clientWidth) element.tabIndex = 0;
      else element.removeAttribute("tabindex");
    }
    const onPath = new Set();
    for (let at = chosen; at; at = at.parent) onPath.add(at.id);
    for (const edge of [...edges].sort((a, b) => b.weight - a.weight)) {
      const a = edge.from.getBoundingClientRect();
      const b = edge.to.getBoundingClientRect();
      const x1 = a.right - frame.left, y1 = a.top + a.height / 2 - frame.top;
      const x2 = b.left - frame.left, y2 = b.top + b.height / 2 - frame.top;
      const bend = (x2 - x1) / 2;
      const path = document.createElementNS(SVG, "path");
      path.setAttribute("d", edge.leader
        ? `M ${x1} ${y1} L ${x2 - 6} ${y2}`
        : `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
      const lit = !edge.aside && onPath.has(edge.node.id) && (edge.leader ? chosen === edge.node : true);
      path.setAttribute("class", ["rm-edge", (edge.leader || edge.thin) && "rm-leader", lit && "on", edge.node.tone && `tone-${edge.node.tone}`].filter(Boolean).join(" "));
      if (!edge.leader && !edge.thin) path.setAttribute("stroke-width", (1.5 + 9.5 * (all ? edge.weight / all : 0)).toFixed(1));
      svg.append(path);
    }
    if (turning && boxes.has(turning.parent.id)) {
      // Where the runs' fates parted: a mark on the split itself.
      const at = boxes.get(turning.parent.id).getBoundingClientRect();
      const mark = document.createElementNS(SVG, "circle");
      mark.setAttribute("cx", String(at.right - frame.left));
      mark.setAttribute("cy", String(at.top + at.height / 2 - frame.top));
      mark.setAttribute("r", "5");
      mark.setAttribute("class", "rm-turn-mark");
      svg.append(mark);
    }
  }

  /** Mark a box and the way to it: a node not drawn — a step of a run gone its own way — marks
   *  the last box on its way that is. */
  function select(node) {
    let shown = node;
    while (shown && !boxes.has(shown.id)) shown = shown.parent;
    chosen = shown;
    for (const [id, made] of boxes) if (made.tagName === "BUTTON") made.setAttribute("aria-pressed", String(Boolean(shown) && id === shown.id));
    lines();
  }

  const watch = new ResizeObserver(() => lines());
  watch.observe(grid);
  watch.observe(element);
  requestAnimationFrame(lines);
  return { element, select, lines, dispose: () => watch.disconnect() };
}

/** The map of a task's runs (`runTree`): each path ends in its runs' marks. */
export function runMap(tree, { onSelect = () => {} } = {}) {
  const total = tree.root.runs.length;
  const pathTo = (node) => {
    const names = [];
    for (let at = node; at && at.move; at = at.parent) names.unshift(at.move.label);
    return names;
  };
  tree.root.label = plural(total, "run");
  return pathsFigure(tree, {
    label: `How ${plural(total, "run")} went, step by step`,
    heads: { start: "start", step: (index) => `step ${index}`, end: "how they ended" },
    turning: tree.turning,
    onSelect,
    describe: (node) => {
      const fared = faredText(tallyOf(node.runs, tree.verdicts));
      if (!node.move) return `All ${plural(node.runs.length, "run")}: ${fared}`;
      const before = pathTo(node.parent);
      return `Step ${node.step}: ${node.move.label}, ${plural(node.runs.length, "run")}, ${fared}${before.length ? `. After ${before.join(", then ")}` : ""}`;
    },
    end: (runs) => [runMarks(runs, tree.verdicts), h("span", { class: "rm-fared" }, faredText(tallyOf(runs, tree.verdicts)))],
  });
}

/**
 * What one step led to, as a fan: the step a box, and from it a branch for each thing the model
 * did there, as wide as its samples were many, ending in how often that was (`moves`: the
 * server's, `{ kind, label, count, rate }`, a reply one move whatever its words). `arms`, for
 * wordings set against each other: `[{ name, tone, moves }]`, one fan under the other, the
 * same step. `from`: what the origin says, "step 1"; `unit`: what is counted, "sample".
 */
export function fanTree(arms, { from, unit = "sample" } = {}) {
  let ids = 0;
  const root = { id: ids++, step: -1, parent: null, move: null, label: from, runs: [], ends: [], children: [], count: 0 };
  for (const arm of arms) {
    const total = arm.moves.reduce((sum, move) => sum + move.count, 0);
    // One wording: its branches leave the step itself. Several: each wording a box of its own.
    const origin = arms.length > 1
      ? { id: ids++, step: 0, parent: root, move: null, label: arm.name, tone: arm.tone, runs: [], ends: [], children: [], count: total, arm }
      : root;
    if (origin !== root) root.children.push(origin);
    root.count += total;
    if (origin === root) root.label = `${from} · ${plural(total, unit)}`;
    else origin.label = `${arm.name} · ${total}`;
    for (const move of arm.moves) {
      origin.children.push({
        id: ids++, step: origin.step + 1, parent: origin, move: { kind: move.kind, label: move.label }, hint: move.hint || null, badge: move.letter || null,
        tone: arm.tone, runs: [], ends: [], children: [], count: move.count, rate: move.rate, ended: move.ended || null, share: move, arm,
      });
    }
  }
  if (arms.length > 1) root.label = from;
  // A line's width is of the wording with the most samples: each wording's fan is its own.
  const most = Math.max(1, ...arms.map((arm) => arm.moves.reduce((sum, move) => sum + move.count, 0)));
  return { root, total: most, ...arrange(root, "draw") };
}

/** The fan of `fanTree`, drawn: each branch ends in its rate, with its interval — or, of forks
 *  run to the end (`ended`: a move's `{ passed, failed, no_verdict }`), in its forks, a mark
 *  each by how its check went, as a path of runs ends. A fork no check judged is a grey ring,
 *  as a run with no verdict is; one left out of the step's rates is in no branch. */
export function fan(tree, { label, onSelect = null, ended = false } = {}) {
  const fared = (node) => {
    const { passed, failed, no_verdict: none = 0 } = node.ended;
    const judged = passed + failed;
    const said = judged ? `${passed} of ${judged} passed${none ? `, ${none} with no check` : ""}` : "no check";
    return [tallyMarks({ passed, failed, none }), h("span", { class: "rm-fared" }, said)];
  };
  return pathsFigure(tree, {
    label,
    total: tree.total,
    count: (node) => node.count,
    heads: { start: "", step: () => "", end: ended ? "how they ended" : "how often" },
    onSelect,
    describe: (node) => (node.move ? `${node.move.label}: ${node.count} of ${node.parent.count}` : node.label),
    end: (_, node) => (ended
      ? (node?.ended ? fared(node) : null)
      : node?.rate ? rateBar(node.rate, node.tone === "change" ? "change" : "accent") : null),
  });
}

/**
 * A choice among a few things at one place, as a fan: the place a box, and from it a branch for
 * each thing chosen there, as wide as its weight, ending in what the caller says of it. A probe's
 * position is one: the token in its sentence, and the original token and each one forced there,
 * each as wide as the model's own chance of it.
 *   origin     `{ label, title }`: the place, its label a string or a node
 *   branches   `[{ kind, label, tone, reads, weight, count, hint, title, data }]`: `weight` from
 *              0 to 1, `count` the words beside the label, `reads` what the box is, in words, where
 *              its colour alone shows it
 *   heads      `{ start, branch, end }`: the words over the three columns
 *   end(data)  what a branch ends in
 */
export function choiceFan(origin, branches, { heads, end, label, className = null } = {}) {
  const root = { id: 0, step: -1, parent: null, move: null, label: origin.label, title: origin.title || "", runs: [], ends: [], children: [], weight: 1 };
  root.children = branches.map((branch, index) => ({
    id: index + 1, step: 0, parent: root, move: { kind: branch.kind, label: branch.label }, hint: branch.hint || null,
    tone: branch.tone || null, reads: branch.reads || null, runs: [], ends: [], children: [], weight: branch.weight || 0, said: branch.count ?? "",
    title: branch.title || "", data: branch.data,
  }));
  return pathsFigure({ root, ...arrange(root, "draw") }, {
    label, className,
    total: 1,
    count: (node) => node.weight,
    countText: (node) => node.said,
    heads: { start: heads.start, step: () => heads.branch, end: heads.end },
    describe: (node) => node.title,
    end: (_, node) => (node ? end(node.data) : null),
  });
}
