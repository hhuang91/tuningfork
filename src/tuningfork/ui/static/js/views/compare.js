// Runs side by side: where they stop agreeing, and what each did there.

import { api, enc, query } from "../api.js";
import {
  chip, command, crumbs, h, info, loading, outcome, panel, plural, rateBar, replace, short, tabs, verdict,
} from "../dom.js";
import { field, input, numberOf, planForm } from "../forms.js";
import { live } from "../live.js";
import { settle } from "../nav.js";
import { responseView } from "../payload.js";
import { formatP, tallyBody } from "../results.js";
import { DETAIL, faredText, runChips, runMap, runMarks, runTree, startDetail, tallyOf } from "../runmap.js";
import { whereForm } from "../where.js";

export async function diffView(root, [first, second]) {
  replace(root, loading());
  const [report, a, b] = await Promise.all([
    api.get(`/diff${query({ a: first, b: second })}`),
    api.get(`/runs/${enc(first)}`),
    api.get(`/runs/${enc(second)}`),
  ]);
  const runA = a.run;
  const runB = b.run;

  const rows = report.steps.map((step) => {
    const byRun = {};
    for (const group of step.groups) for (const id of group.run_ids) byRun[id] = group.outcome;
    return { index: step.index, diverged: step.diverged, a: byRun[runA.id], b: byRun[runB.id] };
  });
  const at = report.divergence_index;

  const sideBySide = h("div", null, at === null ? null : loading("Loading the two responses…"));
  replace(
    root,
    crumbs([["Runs", "#/runs"], ["diff"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Diff"),
      h("p", { class: "sub" }, at === null
        ? `The runs behaved identically for all ${plural(report.steps.length, "step")}.`
        : `Agreed for ${plural(report.agreed_steps, "step")}, then diverged at step ${at}.`))),
    report.mixed_skills ? mixedSkillsBanner() : null,
    h("div", { class: "runs-pair", style: { marginBottom: "var(--space-4)" } }, runCard("A", runA), runCard("B", runB)),
    h("div", { class: "stack" },
      at === null ? null : panel(`Step ${at}, side by side`, sideBySide),
      panel("Step by step", h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, `A · ${short(runA.id)}`), h("th", null, `B · ${short(runB.id)}`))),
        h("tbody", null, rows.map((row) =>
          h("tr", { class: [row.diverged && "diverged", row.index === at && "first-divergence"] },
            h("td", { class: "num" }, String(row.index)),
            h("td", null, cell(runA, row.index, row.a)),
            h("td", null, cell(runB, row.index, row.b))))))))),
  );

  if (at !== null) {
    const fetchStep = (run, outcomeAt) =>
      outcomeAt?.kind === "ended" ? Promise.resolve(null) : api.get(`/runs/${run.id}/steps/${at}`).catch(() => null);
    const [stepA, stepB] = await Promise.all([fetchStep(runA, rows[at].a), fetchStep(runB, rows[at].b)]);
    replace(sideBySide, h("div", { class: "split" }, side("A", runA, stepA, at), side("B", runB, stepB, at)));
  }
}

function runCard(letter, run) {
  return h("div", { class: "panel panel-body" },
    h("div", { class: "side-title" }, chip(letter, "accent"), h("a", { class: "mono", href: `#/runs/${run.id}` }, short(run.id)),
      run.label ? h("span", { class: "muted" }, run.label) : null, h("span", { class: "push" }, verdict(run.result))),
    h("div", { class: "muted small" }, `${plural(run.step_count, "step")}${run.model ? ` · ${run.model}` : ""}`));
}

function cell(run, index, value) {
  if (!value) return h("span", { class: "faint" }, "—");
  if (value.kind === "ended") return h("span", { class: "faint" }, "run ended");
  return h("a", { href: `#/runs/${run.id}/${index}`, style: { color: "inherit" } }, outcome(value));
}

function side(letter, run, step, index) {
  const title = h("div", { class: "side-title" }, chip(letter, "accent"),
    h("a", { href: `#/runs/${run.id}/${index}` }, `run ${short(run.id)}, step ${index}`));
  if (!step) return h("div", null, title, h("p", { class: "muted" }, "This run had ended."));
  return h("div", null, title, h("div", { class: "stack" }, responseView(step.response)));
}

function conditionLabel(condition) {
  const file = condition.skill_file ? condition.skill_file.split(/[\\/]/).pop() : null;
  return [
    condition.harness ? chip(condition.harness, "muted") : null,
    condition.model ? h("span", { class: "mono small" }, condition.model) : null,
    h("span", { class: "mono small" }, condition.skill_version ? `skill version ${condition.skill_version}` : "skill not recorded"),
    file ? chip(`SKILL.md from ${file}`, "skill", condition.skill_file) : null,
  ];
}

function conditionsBanner(task, conditions) {
  return h("div", { class: "banner banner-warn", style: { marginBottom: "var(--space-3-5)" } },
    h("strong", null, `These runs span ${conditions.length} set-ups. `),
    "Compare within one: ",
    conditions.map((condition, position) => [
      position ? " · " : null,
      h("a", { href: `#/divergence/${enc(task)}@${condition.id}` }, `${condition.skill_version ? `skill version ${condition.skill_version}` : "no skill recorded"}${condition.skill_file ? ` (${condition.skill_file.split(/[\\/]/).pop()})` : ""}, ${plural(condition.runs, "run")}`),
    ]));
}

function mixedSkillsBanner() {
  return h("div", { class: "banner banner-warn", style: { marginBottom: "var(--space-3-5)" } }, h("strong", null, "These runs used different versions of the skill. "),
    "A difference between them may be the wording change rather than the model.");
}

// What the map draws, and what its detail changes, a click away.
const DETAIL_NOTE = "A box is the tool the runs called, or a reply, whatever its words. "
  + "A tool's calls are drawn apart where they came in a few distinct forms. "
  + "Less detail draws only the paths runs share, a run gone its own way stopping where it left the others, then one box a tool. "
  + "More draws every run to its end, then every form of a call.";

export async function divergenceView(root, [task, conditionId], scope) {
  replace(root, loading());
  const summary = (await api.get("/tasks")).find((candidate) => candidate.task_id === task);
  const conditions = summary?.conditions || [];
  const chosen = conditionId ? conditions.find((condition) => condition.id === conditionId) : null;
  if (conditionId && !chosen) throw new Error(`Task ${task} has no runs under set-up ${conditionId}.`);
  // One set-up: its latest runs. Otherwise the task's latest runs, whatever they were run under.
  const report = chosen
    ? await api.get(`/divergence${query({ runs: chosen.run_ids.slice(0, 50).join(",") })}`)
    : await api.get(`/divergence${query({ task, limit: 50 })}`);
  // What the failed runs' checks said, counted: the set-up chosen, or every one.
  const tally = await api.get(`/tally${query({ task })}`).catch(() => null);
  const verdicts = report.verdicts || {};
  const all = tallyOf(report.run_ids, verdicts);

  const address = (step) => `#/divergence/${enc(task)}${conditionId ? `@${conditionId}` : ""}${step === null || step < 0 ? "" : `?step=${step}`}`;
  const asked = /^\d+$/.test(scope?.query?.get("step") || "") ? Number(scope.query.get("step")) : null;

  // The map, the lead over it and what is shown of the box chosen: drawn again when the detail changes.
  let detail = startDetail(report);
  let map = null;
  const lead = h("div", { class: "slot" });
  const mapSlot = h("div");
  const inspect = h("div", { class: "rm-inspect", "aria-live": "polite" });
  const detailSlot = h("div", { class: "rm-detail" });
  const show = (node, tree, { keep = false } = {}) => {
    map.select(node);
    replace(inspect, inspector(node, tree, { verdicts, show: (next) => show(next, tree) }));
    if (!keep) settle(address(node.step));
  };
  const draw = (first = null) => {
    map?.dispose();
    const tree = runTree(report, DETAIL[detail]);
    // The turning point the page leads with is the one the terminal and an agent are given, read
    // on the tree with a tool's distinct forms apart: a control of the view never changes what
    // was found. The map marks it where this detail's own tree parts the runs at that step.
    const found = report.paths?.auto?.turning || null;
    const marked = Boolean(found && tree.turning && tree.turning.step === found.step && tree.turning.after === found.after);
    if (!marked) tree.turning = null;
    const turning = tree.turning;
    map = runMap(tree, { onSelect: (node) => show(node, tree) });
    replace(mapSlot, map.element);
    replace(lead, leadView(found, all, report, { marked }));
    const step = (by, label, sign) => h("button", {
      class: "btn btn-small rm-detail-step", type: "button", "aria-label": label, title: label,
      disabled: detail + by < 0 || detail + by >= DETAIL.length,
      onclick: () => {
        detail += by;
        draw();
        detailSlot.querySelector(`.rm-detail-step:not(:disabled)[aria-label="${label}"]`)?.focus();
      },
    }, sign);
    replace(detailSlot, h("span", { class: "muted small" }, "Detail"), step(-1, "Less detail", "−"),
      h("span", { class: "rm-detail-name small", "aria-live": "polite" }, DETAIL[detail].name),
      step(1, "More detail", "+"), info(DETAIL_NOTE, { label: "What the map draws" }));
    // The box shown first: the address's step along the path most runs took, else the split where
    // passing and failing part, else the first split, else all the runs.
    let start = failingBranch(turning, tree) ?? tree.nodes.find((node) => node.children.length > 1) ?? tree.root;
    if (first !== null) {
      let at = tree.root;
      while (at.step < first && at.children.length) at = at.children[0];
      start = at;
    }
    // A step asked for by its number is the address's; the box a page opens on is not.
    show(start, tree, { keep: first === null });
  };
  scope.onLeave?.(() => map?.dispose());

  const failed = (tally?.groups || []).filter((group) => !chosen || group.condition?.id === chosen.id)
    .reduce((sum, group) => sum + (group.failed || 0), 0);
  const sections = tabs([
    failed ? {
      id: "failures", label: "Why runs failed", count: failed,
      body: () => [tallyBody(tally, { only: chosen?.id || null }), live.mode === "live" ? alongFold(task) : null],
    } : null,
    { id: "ask", label: "Ask every run", body: () => whereForm(task, { condition: chosen?.id || null }) },
    {
      id: "steps", label: "All steps", count: report.steps.length,
      body: () => stepsTable(report, (index) => ({
        href: address(index),
        onclick: (event) => {
          event.preventDefault();
          draw(index);
          mapSlot.scrollIntoView({ block: "center" });
        },
      })),
    },
  ], { label: "More about these runs" });

  replace(
    root,
    crumbs([["Runs", "#/runs"], [task], ["divergence"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Where the runs diverge", h("span", { class: "h-sub mono" }, task)),
      chosen ? h("p", { class: "condition-label" }, conditionLabel(chosen)) : null,
      h("p", { class: "sub" }, `${plural(report.run_ids.length, "run")}`,
        all.judged ? ` · ${all.passed} passed, ${all.failed} failed` : " · no verdicts",
        all.uncounted ? `, ${all.uncounted} not counted` : "",
        // Where: what the page's title asks. Compared exactly, which can be before their paths part
        // on the map, where a tool's calls with other arguments are one box.
        report.divergence_index === null || report.divergence_index === undefined ? " · never diverge" : ` · diverge at step ${report.divergence_index}`))),
    !chosen && conditions.length > 1 ? conditionsBanner(task, conditions) : null,
    report.mixed_skills ? mixedSkillsBanner() : null,
    h("div", { class: "stack" },
      h("section", { class: "panel run-map" },
        h("div", { class: "panel-head" }, h("h2", null, "Run map"), h("div", { class: "actions" }, detailSlot)),
        h("div", { class: "panel-body" }, lead, mapSlot, legend(all)),
        inspect),
      sections.element),
  );
  // Drawn once on the page: the lines are measured from where the boxes are.
  draw(asked);
}

/** The marks under the map, said once. */
function legend(all) {
  const mark = (tone, text) => h("span", { class: "rm-key" }, h("span", { class: `rm-mark tone-${tone}` }), text);
  return h("div", { class: "rm-legend" },
    mark("ok", "passed"), mark("bad", "failed"),
    all.uncounted ? mark("warn", "not counted") : null,
    all.none ? mark("none", "no verdict") : null,
    h("span", { class: "rm-key" }, h("span", { class: "rm-key-line" }), "wider line, more runs"),
    h("span", { class: "muted" }, "Choose a box for its runs."));
}

// How far to trust the lead, a click away: its odds in plain words, then its test and its p.
const chance = (p) => (p < 0.0001 ? "less than 1 time in 10,000" : `about 1 time in ${Math.max(2, Math.round(1 / p)).toLocaleString("en")}`);
const pOf = (p) => info(`Among the runs that got here, a split this lopsided would come by chance ${chance(p)} `
  + `(Fisher's exact test, p ${p < 0.0001 ? "< 0.0001" : `= ${formatP(p)}`}). `
  + "With many splits compared, chance alone can make one: resample the step to see how often the model goes each way.",
  { label: "How sure is this?" });

/** At the turning point, the branch the failing runs took: where to look first. */
function failingBranch(turning, tree) {
  if (!turning) return null;
  const branches = turning.parent.children;
  const failed = (node) => tallyOf(node.runs, tree.verdicts).failed;
  return [...branches].sort((a, b) => failed(b) - failed(a))[0] || turning.parent;
}

/** What a step that called no tool and said nothing did, as a sentence says it. */
const DID = { empty: "gave no answer", truncated: "were cut off", error: "had a failed call" };

/** What the turning point's runs did, as the lead says it. */
function didWords(turning) {
  if (turning.stopped) return "ended there";
  if (turning.kind === "text") return "replied";
  if (turning.kind !== "tool_call") return DID[turning.kind] || `had ${turning.label}`;
  // Where a tool's calls are drawn apart, the words that tell this form from the others, as its box has them.
  return turning.hint
    ? ["called ", h("span", { class: "mono" }, turning.label), " with ", h("span", { class: "mono" }, turning.hint)]
    : ["called ", h("span", { class: "mono" }, turning.label)];
}

/** What the page leads with, over the map: the step where passing and failing part, in a line
 *  with the numbers it stands on and what it is — a lead; or why there is none to name.
 *  `turning` is the server's (`report.paths.auto.turning`); `marked`, whether the map as drawn
 *  marks it. */
function leadView(turning, all, report, { marked = true } = {}) {
  const tool = report.tool_split?.best;
  const toolLine = tool
    ? h("p", { class: "muted small" }, "At any step: runs that called ", h("span", { class: "mono" }, tool.tool),
        ` passed ${tool.called.passed} of ${tool.called.total}; runs that never did, ${tool.never.passed} of ${tool.never.total}.`)
    : null;
  if (all.judged < 2) return h("p", { class: "hint" }, "Too few verdicts to compare.");
  if (!all.passed || !all.failed) {
    return h("p", { class: "hint" }, all.passed ? `All ${all.judged} judged runs passed.` : `None of the ${all.judged} judged runs passed.`);
  }
  if (!turning) {
    return [h("p", { class: "hint" }, "No split separates passing from failing clearly enough to name. The difference may be in what the runs wrote: see Why runs failed.",
      info("A turning point is named where the runs that took one branch of a split passed more or less often than the rest that got there, by more than chance alone would often produce (p under 0.2), with at least two judged runs on each side.", { label: "When one is named" })), toolLine];
  }
  const after = turning.after ? ["After ", h("span", { class: "mono" }, turning.after), ", runs"] : "Runs";
  const sameTool = tool && turning.kind === "tool_call" && turning.label === tool.tool;
  return [h("div", { class: "rm-lead" },
    h("strong", null, `Turning point: step ${turning.step}.`),
    h("span", null, after, " that ", didWords(turning), ` passed ${turning.passed} of ${turning.total}. `,
      `The rest passed ${turning.others_passed} of ${turning.others_total}.`),
    // What it is stays in sight; how far to trust it is a click away.
    h("span", { class: "rm-lead-caution" }, "A lead, not a verdict.", pOf(turning.p)),
    marked ? null : h("span", { class: "rm-lead-caution" }, "It is not marked on the map at this detail.")),
    sameTool ? null : toolLine];
}

/** The box chosen, under the map: what its runs did here, what they did next, and the ways into
 *  the step. `show(node)` chooses another box. */
function inspector(node, tree, { verdicts, show }) {
  const rate = node.passed;
  const path = [];
  for (let at = node.parent; at && at.move; at = at.parent) path.unshift(at);
  const example = node.runs[0];
  const title = node.move
    ? [h("span", { class: "rm-step" }, `Step ${node.step}`), outcome({ kind: node.move.kind, label: node.move.label })]
    : h("span", { class: "rm-step" }, "All runs");
  const actions = node.move ? h("div", { class: "rm-actions" },
    h("a", { class: "btn btn-primary btn-small", href: `#/runs/${example}/${node.step}` }, `Open step ${node.step} in run ${short(example)}`),
    live.mode === "live"
      ? h("a", { class: "btn btn-small", href: `#/runs/${example}/${node.step}?start=resample`, "data-control": true }, "Resample…")
      : null,
    h("a", { class: "btn btn-quiet btn-small", href: `#/runs/${example}/${node.step}?view=tokens` }, "Tokens")) : null;

  // What its runs did here, exactly: each form of the call, or each phrasing of the reply.
  const forms = node.variants || [];
  const few = forms.slice(0, 4);
  const formRow = (variant) => h("li", { class: "rm-form" },
    h("span", { class: "mono rm-form-label" }, variant.outcome.label),
    h("span", { class: "rm-form-runs" }, h("span", { class: "faint small" }, `×${variant.runs.length}`), runMarks(variant.runs, verdicts)));
  const here = node.move ? h("div", { class: "rm-col" },
    h("div", { class: "label" }, forms.length > 1
      ? (node.move.kind === "text" ? `${forms.length} phrasings` : `${forms.length} forms of the call`)
      : (node.move.kind === "text" ? "The reply" : "The call")),
    h("ul", { class: "rm-forms" }, few.map(formRow)),
    forms.length > few.length
      ? h("details", { class: "fold" }, h("summary", null, `${forms.length - few.length} more`), h("ul", { class: "rm-forms" }, forms.slice(few.length).map(formRow)))
      : null) : null;

  // What they did next: each branch, and the runs that ended here.
  const branches = node.children.map((child) => ({ node: child, runs: child.runs, passed: child.passed }));
  if (node.ends.length) branches.push({ node: null, runs: node.ends, passed: node.ended });
  const branchRow = (branch) => {
    const fared = tallyOf(branch.runs, verdicts);
    const passed = branch.passed;
    const name = branch.node
      ? h("button", { class: "link-button rm-next-name", type: "button", onclick: () => show(branch.node) },
          outcome({ kind: branch.node.move.kind, label: branch.node.move.label }))
      : h("span", { class: "rm-next-name muted" }, "ended");
    return h("li", { class: "rm-next-row" }, name, h("span", { class: "faint small" }, `×${branch.runs.length}`),
      passed ? h("span", { class: "rm-next-rate" }, h("span", { class: "muted small" }, "passed"), rateBar(passed)) : h("span", { class: "muted small" }, faredText(fared)));
  };
  const most = 6;
  const next = !node.children.length ? h("div", { class: "rm-col" }, h("div", { class: "label" }, "Then"),
    h("p", { class: "muted small" }, node.runs.length === 1 ? "The run ended here." : "These runs ended here."))
  : h("div", { class: "rm-col" },
    h("div", { class: "label" }, node.move ? "Then" : "First"),
    h("ul", { class: "rm-next" }, branches.slice(0, most).map(branchRow)),
    branches.length > most
      ? h("details", { class: "fold" }, h("summary", null, `${branches.length - most} more`), h("ul", { class: "rm-next" }, branches.slice(most).map(branchRow)))
      : null);

  // Its runs, each a way into the run: at this step, where the box is one.
  const runs = h("div", { class: "rm-runs" },
    h("div", { class: "label" }, node.move ? `Its ${plural(node.runs.length, "run")}, at step ${node.step}` : `The ${plural(node.runs.length, "run")}`),
    runChips(node.runs, verdicts, { step: node.move ? node.step : null }));
  return [
    h("div", { class: "rm-inspect-head" },
      path.length ? h("span", { class: "rm-path" }, path.map((at) => [
        h("button", { class: "link-button", type: "button", onclick: () => show(at) }, at.move.label), h("span", { class: "sep" }, "›")])) : null,
      title,
      h("span", { class: "muted" }, plural(node.runs.length, "run")),
      rate ? h("span", { class: "rm-next-rate" }, h("span", { class: "muted small" }, "passed"), rateBar(rate)) : null,
      actions),
    h("div", { class: "rm-cols" }, here, next),
    runs,
  ];
}

/** The task's check run along its runs (the `check-along` job): on each run's workspace as it
 *  started, after each call that changed it, and at its end — so the tally names the call that
 *  made each finding, and a mistake every run made is named as one. Folded under the failures. */
function alongFold(task) {
  const form = planForm({
    kind: "check-along",
    fields: [h("div", { class: "form-grid" },
      field("Runs", input("limit", { type: "number", value: 50, min: 1, max: 500 }), { help: "The task's latest recorded runs, at most." }))],
    read: (element) => ({ task, limit: numberOf(element, "limit") ?? 50 }),
    render: (plan) => h("div", { class: "stack" },
      plan.runs.length
        ? h("p", null, `The check, on ${plural(plan.runs.length, "run")}: ${plural(plan.states, "workspace state")}, each as a call left it.`)
        : h("p", null, "No run here to run the check along: each is passed over, below."),
      plan.skipped?.length
        ? h("p", { class: "faint small" }, `Passed over: ${plan.skipped.slice(0, 3).join("; ")}${plan.skipped.length + (plan.skipped_more || 0) > 3 ? ", and more" : ""}`)
        : null),
    startLabel: "Find the calls",
    onStarted: (job) => (location.hash = `#/jobs/${job.id}`),
  });
  return h("details", { class: "fold rm-along" },
    h("summary", null, "Find the call behind each line…"),
    h("p", { class: "hint" }, "Runs the check on each run's files as they started and after each call that changed them, so each line above names the call that made it."),
    form.element);
}

/** Every step in a table, the behaviours at each: `link(index)` gives each step's number the
 *  props that show it on the map above. */
function stepsTable(report, link) {
  const behaviours = (step) =>
    h("div", { class: "stack", style: { gap: "var(--space-1)" } }, step.groups.map((group) =>
      h("div", null, outcome(group.outcome), h("span", { class: "faint small" }, `  ×${group.run_ids.length}`))));
  const rows = report.steps.map((step) =>
    h("tr", { class: [step.diverged && "diverged", step.index === report.divergence_index && "first-divergence"] },
      h("td", { class: "num" }, h("a", { ...link(step.index), title: `Show step ${step.index} on the map above` }, String(step.index))),
      h("td", null, behaviours(step))));
  return h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, "behaviours"))),
    h("tbody", null, rows)));
}
