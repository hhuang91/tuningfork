// The front page: what tuningfork is for, what is set up, where you left off, how its pieces
// fit, what each way of looking at a step answers, and its words — each defined once, on the
// server (`ops/glossary.py`), as the terminal's `tuningfork glossary` shows them.

import { api } from "../api.js";
import { ago, chip, command, h, loading, panel, plural, replace, short, verdict, when } from "../dom.js";
import { live } from "../live.js";

export async function homeView(root, _, scope) {
  replace(root, loading());
  const quiet = (promise) => promise.catch(() => null);
  const [runs, refinements, scenarios, words, connections] = await Promise.all([
    quiet(api.get("/runs?limit=5")),
    quiet(api.get("/refinements")),
    quiet(api.get("/scenarios")),
    quiet(api.get("/glossary")),
    live.mode === "live" ? quiet(api.get("/connections")) : null,
  ]);
  const terms = new Map((words?.terms || []).map((term) => [term.term, term]));
  const glossaryEl = glossaryPanel(words, scope.query?.get("term") || "");

  replace(root,
    h("div", { class: "page-head" }, h("div", null,
      h("h1", null, "Tuning Fork"),
      h("p", { class: "sub" }, "A step debugger and experiment lab for agent skills on local models. Record what a harness asks the model, find the step where it goes wrong, and measure what would change it — before anything changes the skill."))),
    h("div", { class: "stack" },
      leftOff(runs, refinements),
      setup(runs, scenarios, connections),
      panel("How the pieces fit", [
        mapView(terms),
        h("div", { class: "map-notes" },
          h("p", null, h("strong", null, "Record. "), "A scenario holds a skill's tasks and a check. A batch job launches the harness once per run, each in a fresh workspace; the proxy records every model call as a step of that run. Your own harness, pointed at the proxy, is recorded the same way."),
          h("p", null, h("strong", null, "Measure. "), "At any step, an experiment sends the step's exact request again, with everything before it held fixed: the model's behaviours there, counted. Run to the end, each sample is a fork of the run, judged by the task's check."),
          h("p", null, h("strong", null, "Improve. "), "A refinement declares what should happen at a step, then screens small edits against the original wording. A survivor is validated with full runs — held-out tasks included — and written back only when a person approves it: a new version of the skill, measured again from the top."),
          h("p", { class: "muted small" }, "Jobs are how the web UI starts work: a batch, a resample, a screen, a validation, each run inside `tuningfork serve`. A job is a record of the work; what it made — runs, experiments — is kept on its own."),
          colours()),
      ]),
      panel("What do you want to find out?", h("div", { class: "workflows" }, WORKFLOWS.map(workflowCard))),
      glossaryEl));
  if (scope.query?.get("term")) glossaryEl.scrollIntoView({ block: "start" });
}

// The colours, each shown as the chips that wear it (app.css keeps the legend beside its tokens).
function colours() {
  return h("p", { class: "colours" }, h("strong", null, "Colours. "),
    chip("pass", "ok"), " ", chip("fail", "bad"), " Green and red are a check's verdict. ",
    chip("not counted", "warn"), " Amber is a caution. ",
    chip("edited", "change"), " Teal is what a person or tuningfork changed. ",
    chip("skill", "skill"), " Purple is a skill's own words. ",
    chip("tool call", "accent"), " Indigo is what you can act on, what is current or significant, and the model's own answer. "
      + "Everything else is grey.");
}

// --- where you left off -----------------------------------------------------------------------

function leftOff(runs, refinements) {
  const held = live.mode === "live" ? live.holds.length : 0;
  const jobs = live.mode === "live" ? live.jobs : [];
  const open = (refinements || []).filter((each) => each.status === "open");
  if (!runs?.length && !held && !jobs.length && !open.length) return null;
  const card = (title, href, body) => h("div", { class: "left-off-card" },
    h("div", { class: "left-off-head" }, h("h3", null, title), href ? h("a", { class: "small", href }, "All") : null), body);
  return panel("Where you left off", h("div", { class: "left-off" },
    held ? card("Held now", "#/live", h("p", null, h("a", { href: "#/live" }, plural(held, "model call")), " waiting at a breakpoint.")) : null,
    jobs.length ? card("Running", "#/jobs", h("ul", { class: "plain-list" }, jobs.slice(0, 4).map((job) =>
      h("li", null, h("a", { href: `#/jobs/${job.id}` }, job.kind), " ", chip(job.status, "accent"))))) : null,
    runs?.length ? card("Latest runs", "#/runs", h("ul", { class: "plain-list" }, runs.slice(0, 4).map((run) =>
      h("li", null, h("a", { class: "mono", href: `#/runs/${run.id}` }, run.label || short(run.id)), " ",
        run.task_id && !(run.label || "").startsWith(run.task_id) ? h("span", { class: "muted small" }, run.task_id) : null, " ", verdict(run.result),
        h("span", { class: "faint small", title: when(run.started_at) }, ` ${ago(run.started_at)}`))))) : null,
    open.length ? card("Open refinements", "#/refine", h("ul", { class: "plain-list" }, open.slice(0, 3).map((each) =>
      h("li", null, h("a", { class: "mono", href: `#/refine/${each.id}` }, short(each.id)), " ",
        h("span", { class: "mono small" }, each.skill), h("span", { class: "muted small" }, ` · ${plural(each.proposals, "proposal")}`))))) : null));
}

// --- what is set up ---------------------------------------------------------------------------

function setup(runs, scenarios, connections) {
  if (live.mode !== "live") {
    return panel("Browsing recordings", [h("p", { class: "hint" },
      "Everything recorded can be read here, but nothing can be started: that needs the proxy. Start it with ",
      h("code", null, "tuningfork serve --upstream <your inference server>"), " and open the address it prints."),
    serverNote()]);
  }
  const server = connections?.server || {};
  const identity = server.identity || {};
  const models = (identity.models || []).join(", ") || (identity.model_file || "").split("/").pop();
  const harnesses = (connections?.harnesses || []).filter((each) => each.path);
  const tasks = [...new Set((runs || []).map((run) => run.task_id).filter(Boolean))];
  const items = [
    {
      done: Boolean(connections && !server.problem && identity.id),
      title: "An inference server answers",
      body: server.problem
        ? [h("span", { class: "warn-text" }, server.problem), " — ", h("a", { href: "#/connections" }, "Connections")]
        : identity.id ? ["answering at ", h("span", { class: "mono" }, server.url || ""), models ? [", with ", h("span", { class: "mono" }, models)] : null]
          : ["model calls go to ", h("span", { class: "mono" }, server.url || "the upstream"), ", not read yet — ", h("a", { href: "#/connections" }, "check it")],
    },
    {
      done: harnesses.length > 0,
      title: "A harness to launch",
      body: harnesses.length
        ? ["found: ", harnesses.map((each) => each.name).join(", ")]
        : ["none found on PATH — ", h("a", { href: "#/connections" }, "name its executable"), ", or point your own harness at ", h("code", null, `${location.origin}/v1`)],
    },
    {
      // A scenario, or a spec on disk whose task has been run.
      done: Boolean(scenarios?.length || tasks.length),
      title: "A task to run",
      body: scenarios?.length
        ? [plural(scenarios.length, "scenario"), " — ", h("a", { href: "#/scenarios" }, "Scenarios")]
        : tasks.length
          ? ["run from a spec on disk: ", h("span", { class: "mono" }, tasks.join(", ")), " — import it on ", h("a", { href: "#/scenarios" }, "Scenarios"), " to edit it here"]
          : ["write one — its prompt, input files and check — as a ", h("a", { href: "#/scenarios/new" }, "new scenario"), ", or import an evals.json on ", h("a", { href: "#/scenarios" }, "Scenarios")],
    },
    {
      done: Boolean(runs?.length),
      title: "A run recorded",
      body: runs?.length
        ? ["the latest ", ago(runs[0].started_at), " — ", h("a", { href: "#/runs" }, "Runs")]
        : [h("a", { href: "#/batch" }, "Run a task…"), ", or send your harness's calls through ", h("code", null, `${location.origin}/v1`)],
    },
  ];
  const done = items.filter((item) => item.done).length;
  const list = h("ol", { class: "checklist" }, items.map((item) => h("li", { class: item.done ? "done" : "todo" },
    h("span", { class: "check-mark", "aria-label": item.done ? "done" : "to do" }, item.done ? "✓" : "○"),
    h("div", null, h("strong", null, item.title), h("div", { class: "muted small" }, item.body)))));
  // The recommendation stays in sight when the finished checklist folds away.
  if (done === items.length) {
    return [h("details", { class: "panel setup-done" }, h("summary", null, h("strong", null, "Set up"), h("span", { class: "muted" }, " — server, harness, a task and a run: all there")), h("div", { class: "panel-body" }, list)),
      serverNote()];
  }
  return [panel(`Get set up · ${done} of ${items.length}`, list), serverNote()];
}

// Measured (docs/setup.md): without --no-cache-prompt, llama-server answers a prompt from its cache
// of an earlier one, so a step's samples share one state and a screen favours its proposals. For a
// sliding-window model, without --ctx-checkpoints 0 or --swa-full, it cuts a prompt into batches at
// places that differ between its chat endpoint and its raw one, so an answer scores differently
// from how it was sampled.
function serverNote() {
  return h("div", { class: "banner banner-warn" }, h("strong", null, "Strongly recommended: "), "start llama-server with ",
    h("code", null, "--no-cache-prompt"), " and, for a model with sliding-window layers (Gemma 2, 3, 4), ",
    h("code", null, "--ctx-checkpoints 0"), " or ", h("code", null, "--swa-full"),
    ". Without the first, a step's behaviour can depend on what the server ran before: Claude Code's samples and every fork share one cached state, and a screen favours its proposals (",
    h("a", { href: "#/guide?term=prompt%20cache" }, "why"),
    "). Without the second, an answer scores differently on the raw endpoint than on the chat one it was sampled on (",
    h("a", { href: "#/guide?term=sliding-window" }, "why"),
    "). The server check measures prompt reuse; llama-server reports neither --ctx-checkpoints nor --swa-full.");
}

// --- the map --------------------------------------------------------------------------------------

const SVG = "http://www.w3.org/2000/svg";

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value !== null && value !== undefined) el.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const LANES = [
  "1 · Record — a task run through the harness, every model call kept",
  "2 · Measure — what the model does at one step, with its history held fixed",
  "3 · Improve — an edit, tested before it is written",
];
const NODES = [
  ["skill", "Skill", "its SKILL.md, versioned", 0, 0, "#/skills"],
  ["scenario", "Scenario", "tasks and a check", 1, 0, "#/scenarios"],
  ["job", "Job", "a batch, run in serve", 2, 0, "#/jobs"],
  ["launch", "Launch", "the harness, started fresh", 3, 0, "#/runs"],
  ["run", "Run", "its steps, recorded", 4, 0, "#/runs"],
  ["step", "Step", "one model call", 4, 1, "#/runs"],
  ["experiment", "Experiment", "resample · A/B · tokens", 3, 1, "#/experiments"],
  ["sample", "Samples", "behaviours, counted", 2, 1, "#/experiments"],
  ["fork", "Forks", "samples run to the end", 1, 1, "#/experiments"],
  ["refinement", "Refinement", "a target, declared first", 4, 2, "#/refine"],
  ["proposal", "Proposals", "small, exact edits", 3, 2, "#/refine"],
  ["screen", "Screen", "each against the original", 2, 2, "#/refine"],
  ["validation", "Validation", "full runs and held-out tasks", 1, 2, "#/refine"],
  ["approval", "Approval", "a person writes it back", 0, 2, "#/refine"],
];
const EDGES = [
  ["skill", "scenario"], ["scenario", "job"], ["job", "launch"], ["launch", "run"],
  ["run", "step", "steps"], ["step", "experiment"], ["experiment", "sample"], ["sample", "fork"],
  ["step", "refinement"], ["refinement", "proposal"], ["proposal", "screen"], ["screen", "validation"],
  ["validation", "approval"], ["approval", "skill", "a new version"],
];
const W = 160, H = 58, X0 = 20, DX = 190, Y0 = 36, DY = 132;

/** How runs, steps, experiments, jobs and refinements relate: one loop, from a skill back to
 *  its next version. Each box opens its page; hovering says what the word means. */
function mapView(terms) {
  const at = new Map(NODES.map(([id, , , col, row]) => [id, { x: X0 + col * DX, y: Y0 + row * DY, row }]));
  const edges = EDGES.map(([from, to, label]) => {
    const a = at.get(from), b = at.get(to);
    let x1, y1, x2, y2;
    if (a.row === b.row) {
      const right = b.x > a.x;
      x1 = right ? a.x + W : a.x; x2 = right ? b.x : b.x + W;
      y1 = y2 = a.y + H / 2;
    } else {
      x1 = x2 = a.x + W / 2;
      const down = b.y > a.y;
      y1 = down ? a.y + H : a.y; y2 = down ? b.y : b.y + H;
    }
    const mid = label ? s("text", { class: "map-edge-label", x: a.row === b.row ? (x1 + x2) / 2 : x1 + 8,
      y: a.row === b.row ? y1 - 8 : (y1 + y2) / 2 + 4, "text-anchor": a.row === b.row ? "middle" : "start" }, label) : null;
    return s("g", { class: "map-edge" }, s("line", { x1, y1, x2, y2, "marker-end": "url(#map-arrow)" }), mid);
  });
  const nodes = NODES.map(([id, label, sub, , row, href]) => {
    const { x, y } = at.get(id);
    const term = terms.get(label.toLowerCase()) || terms.get(id);
    return s("a", { href, class: `map-node lane-${row}` },
      s("title", null, term ? `${label}: ${term.means}` : label),
      s("rect", { x, y, width: W, height: H, rx: 8 }),
      s("text", { class: "map-label", x: x + 12, y: y + 24 }, label),
      s("text", { class: "map-sub", x: x + 12, y: y + 43 }, sub));
  });
  const lanes = LANES.map((text, row) => s("text", { class: `map-lane lane-${row}`, x: X0, y: Y0 + row * DY - 12 }, text));
  return h("div", { class: "map" }, s("svg", { viewBox: "0 0 960 372", role: "img",
    "aria-label": "How tuningfork's pieces relate: record a task, measure a step, improve the skill, and back to the skill" },
    s("defs", null, s("marker", { id: "map-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      s("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "map-arrowhead" }))),
    // The captions over the loop's line, each with a halo of the panel's colour.
    edges, lanes, nodes));
}

// --- what each way of looking answers ---------------------------------------------------------------

const WORKFLOWS = [
  {
    question: "Does this skill work?",
    answer: "A task run N times, each in a fresh workspace: a pass rate with its interval, kept apart for each version of the skill.",
    steps: ["Write a scenario: the task's prompt, its input files and its check", "Run a task… — the harness, and how many runs", "Each version's rate: on Runs, and on the skill's page"],
    cli: "tuningfork batch evals.json -n 8", href: "#/batch", go: "Run a task…",
  },
  {
    question: "Where did runs stop agreeing?",
    answer: "The first step where runs of one task did different things: where to look first.",
    steps: ["Runs: a task's row → Where do they diverge?", "Open that step in one of the runs"],
    cli: "tuningfork divergence --task <skill>/<eval>", href: "#/runs", go: "Runs",
  },
  {
    question: "Is this step flaky?",
    answer: "The step's exact request sent again and again, its history held fixed: how often the model does each thing there.",
    steps: ["Open a run, and choose the step", "Resample… — 20 samples is a start", "Each behaviour gets a letter and a rate"],
    cli: "tuningfork resample <run> <step> -n 20", href: "#/runs", go: "Choose a run",
  },
  {
    question: "Would other wording change what the model does here?",
    answer: "An A/B at the step: your edit of the skill against the original, toward a target declared before anything is sampled.",
    steps: ["At the step: A/B a wording…", "Edit the text; declare the target — a behaviour's letter, a pattern, or conditions in words", "Read the shift with its p-value; without a target it is only exploratory"],
    cli: "tuningfork ab <run> <step> --variant SKILL.v2.md --target A", href: "#/runs", go: "Choose a run",
  },
  {
    question: "Does a change at the step carry through to the end?",
    answer: "Each sample a fork of the run — replayed up to the step, live after it — and judged by the task's check.",
    steps: ["In Resample… or A/B a wording…: Reach → Each to the end", "Or Fork from here… to follow one fork, breakpoints on"],
    cli: "tuningfork ab <run> <step> --variant SKILL.v2.md --to-end", href: "#/runs", go: "Choose a run",
  },
  {
    question: "Which token decided it?",
    answer: "The answer as the model's tokens, how sure it was of each, and which alternatives would have changed what it did next.",
    steps: ["At the step, the Tokens tab: the answer, scored", "Find forking tokens… at the uncertain ones", "A token whose alternatives flip the behaviour is where the decision was made"],
    cli: "tuningfork forking-tokens <run> <step>", href: "#/runs", go: "Choose a run",
  },
  {
    question: "How do I fix the skill?",
    answer: "A refinement: the target declared first, small edits proposed, each screened against the original, a survivor validated with full runs — and approved by you.",
    steps: ["At the step: Refine the skill here… — declare the target", "Propose edits: your own, an agent's over MCP, or the built-in proposer's", "Screen them, validate a survivor, approve it once the gate is met"],
    cli: "tuningfork refine start <run> <step> --target-rules '[…]'", href: "#/refine", go: "Refine",
  },
  {
    question: "What is the model doing right now?",
    answer: "Breakpoints on live traffic: hold a call before the model sees it, or after it answers with a tool call you name.",
    steps: ["Live → Add a breakpoint", "Run your harness through the proxy", "Look at what is held; edit it, continue, step or drop it"],
    cli: "tuningfork break add --tool write_file", href: "#/live", go: "Live",
  },
];

function workflowCard(flow) {
  const reachable = live.mode === "live" || !["#/batch", "#/live"].includes(flow.href);
  return h("article", { class: "workflow" },
    h("h3", null, flow.question),
    h("p", { class: "muted" }, flow.answer),
    h("ol", { class: "workflow-steps" }, flow.steps.map((step) => h("li", null, step))),
    command(flow.cli),
    reachable ? h("a", { class: "btn btn-small", href: flow.href }, flow.go) : null);
}

// --- words ----------------------------------------------------------------------------------------------

function glossaryPanel(words, initial) {
  if (!words) return panel("Words", h("p", { class: "muted" }, "The glossary could not be read."));
  const filter = h("input", { class: "field", type: "search", placeholder: "Look a word up…", value: initial, "aria-label": "Look a word up" });
  const body = h("div");
  const rows = new Map();
  const draw = () => {
    const wanted = filter.value.trim().toLowerCase();
    const matches = (term) => !wanted || term.term.toLowerCase().includes(wanted)
      || term.aka.some((aka) => aka.toLowerCase().includes(wanted)) || term.means.toLowerCase().includes(wanted);
    rows.clear();
    const groups = words.groups.map((group) => {
      const found = words.terms.filter((term) => term.group === group.id && matches(term));
      if (!found.length) return null;
      return [h("tr", { class: "group-row" }, h("th", { colspan: "2" }, group.title)),
        found.map((term) => {
          const row = h("tr", { class: "term-row" },
            h("td", { class: "term" }, h("strong", null, term.term),
              term.aka.length ? h("div", { class: "faint small" }, term.aka.join(", ")) : null),
            h("td", null, term.means,
              term.where ? h("div", { class: "muted small" }, "Where: ", term.where) : null,
              term.see.length ? h("div", { class: "small" }, "See ", term.see.map((other, index) => [index ? ", " : "",
                h("button", { class: "link-button", type: "button", onclick: () => jump(other) }, other)])) : null));
          rows.set(term.term, row);
          return row;
        })];
    });
    const shown = groups.filter(Boolean);
    replace(body, shown.length
      ? h("div", { class: "table-wrap" }, h("table", { class: "table glossary" }, h("tbody", null, shown)))
      : h("p", { class: "muted" }, "No word matches."));
  };
  const jump = (term) => {
    if (!rows.has(term)) {
      filter.value = "";
      draw();
    }
    const row = rows.get(term);
    if (!row) return;
    row.scrollIntoView({ block: "center", behavior: "smooth" });
    row.classList.add("flash");
    setTimeout(() => row.classList.remove("flash"), 1600);
  };
  filter.addEventListener("input", draw);
  draw();
  return panel(`Words · ${words.terms.length}`, [h("p", { class: "hint" }, "tuningfork's words, each as it means it here. The same list in a terminal: ",
    h("code", null, "tuningfork glossary [word]"), "."), filter, body]);
}
