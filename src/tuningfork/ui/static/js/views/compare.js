// Runs side by side: where they stop agreeing, and what each did there.

import { api, enc, query } from "../api.js";
import {
  chip, command, crumbs, h, loading, outcome, panel, plural, rateBar, replace, short, verdict,
} from "../dom.js";
import { field, input, numberOf, planForm } from "../forms.js";
import { live } from "../live.js";
import { settle } from "../nav.js";
import { responseView } from "../payload.js";
import { formatP, tallyView } from "../results.js";
import { whereForm } from "../where.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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
    h("div", { class: "runs-pair", style: { marginBottom: "16px" } }, runCard("A", runA), runCard("B", runB)),
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
  return h("div", { class: "banner banner-warn", style: { marginBottom: "14px" } },
    h("strong", null, `These runs span ${conditions.length} set-ups. `),
    "Compare within one: ",
    conditions.map((condition, position) => [
      position ? " · " : null,
      h("a", { href: `#/divergence/${enc(task)}@${condition.id}` }, `${condition.skill_version ? `skill version ${condition.skill_version}` : "no skill recorded"}${condition.skill_file ? ` (${condition.skill_file.split(/[\\/]/).pop()})` : ""}, ${plural(condition.runs, "run")}`),
    ]));
}

function mixedSkillsBanner() {
  return h("div", { class: "banner banner-warn", style: { marginBottom: "14px" } }, h("strong", null, "These runs used different versions of the skill. "),
    "A difference between them may be the wording change rather than the model.");
}

export async function divergenceView(root, [task, conditionId], scope) {
  replace(root, loading());
  const summary = (await api.get("/tasks")).find((candidate) => candidate.task_id === task);
  const conditions = summary?.conditions || [];
  const chosen = conditionId ? conditions.find((condition) => condition.id === conditionId) : null;
  if (conditionId && !chosen) throw new Error(`Task ${task} has no runs under set-up ${conditionId}.`);
  // One condition: its latest runs. Otherwise the task's latest runs, whatever they were run under.
  const report = chosen
    ? await api.get(`/divergence${query({ runs: chosen.run_ids.slice(0, 50).join(",") })}`)
    : await api.get(`/divergence${query({ task, limit: 50 })}`);
  // What the failed runs' checks said, counted: the condition chosen, or every one.
  const tally = await api.get(`/tally${query({ task })}`).catch(() => null);
  const at = report.divergence_index;
  const split = at === null ? null : report.steps[at];
  const best = report.verdict_split?.best;

  // The step shown, one at a time and each a click away: the address's, else where the runs first
  // split. The first split is often not the step that decides — the runs can agree again after it,
  // or reach the same end by other calls — so every step is there to look at.
  const address = (index) => `#/divergence/${enc(task)}${conditionId ? `@${conditionId}` : ""}?step=${index}`;
  const asked = /^\d+$/.test(scope?.query?.get("step") || "") ? Number(scope.query.get("step")) : null;
  const first = asked !== null && asked < report.steps.length ? asked : at ?? 0;
  let section = null;
  const show = (index, { focus = false, scroll = false } = {}) => {
    const next = atStep(report, index, { verdicts: report.verdicts || {}, best, show });
    section.replaceWith(next);
    section = next;
    settle(address(index));
    if (focus) next.querySelector(".step-pick.current")?.focus();
    if (scroll) next.scrollIntoView({ block: "start" });
  };
  section = report.steps.length ? atStep(report, first, { verdicts: report.verdicts || {}, best, show }) : null;

  replace(
    root,
    crumbs([["Runs", "#/runs"], [task], ["divergence"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Where the runs diverge", h("span", { class: "h-sub mono" }, task)),
      chosen ? h("p", { class: "condition-label" }, conditionLabel(chosen)) : null,
      h("p", { class: "sub" }, at === null
        ? `All ${plural(report.run_ids.length, "run")} behaved identically for every one of ${plural(report.steps.length, "step")}.`
        : `${plural(report.run_ids.length, "run")} agreed for ${plural(report.agreed_steps, "step")}, then split at step ${at} into ${split.groups.length} behaviours.`))),
    !chosen && conditions.length > 1 ? conditionsBanner(task, conditions) : null,
    report.mixed_skills ? mixedSkillsBanner() : null,
    h("div", { class: "stack" },
      // The take-away first: which step's behaviour goes with passing, and the way to it.
      predictorsView(report),
      section,
      tallyView(tally, { only: chosen?.id || null }),
      live.mode === "live" ? alongPanel(task) : null,
      panel("Where does it hold?", whereForm(task, { condition: chosen?.id || null }), {
        meta: h("span", { class: "muted" }, "One question of every run: the call that made the difference."),
      }),
      panel("Every step", stepsTable(report, (index) => ({
        href: address(index),
        onclick: (event) => {
          event.preventDefault();
          show(index, { scroll: true });
        },
      })))),
  );
}

/** One step of the runs side by side: each behaviour they took there, the runs that took it
 *  coloured by their verdicts, and the ways into the step — every other step a click away above
 *  it, shaded where the runs split. The lead above offers to resample its own step: this panel
 *  does not offer it again there. */
function atStep(report, index, { verdicts, best, show }) {
  const step = report.steps[index];
  const resampledAbove = Boolean(best?.run_id) && best.kind !== "ended" && best.index === index;
  return panel(`At step ${index}`, [
    stepPicker(report, index, show),
    h("p", { class: "step-note" }, stepNote(report, step)),
    groupsView(step, index, verdicts),
    nextStep(step, index, { resampledAbove }),
  ], {
    meta: h("span", { class: "muted", title: "Wilson score intervals" }, "Intervals are 95%. With few runs, they are wide."),
    className: "at-step",
  });
}

/** Every step of the runs, to show each in turn: shaded where they split, the one shown in the
 *  accent; the arrow keys move along. */
function stepPicker(report, index, show) {
  const last = report.steps.length - 1;
  const at = report.divergence_index;
  const said = (step) =>
    `Step ${step.index}: ${step.diverged ? "the runs split here" : "every run did the same"}${step.index === at ? ", for the first time" : ""}`;
  const keys = (event) => {
    const next = index + ({ ArrowLeft: -1, ArrowRight: 1 }[event.key] || 0);
    if (next === index || next < 0 || next > last) return;
    event.preventDefault();
    show(next, { focus: true });
  };
  return h("div", { class: "step-picker", onkeydown: keys },
    h("button", { class: "btn btn-small", type: "button", disabled: index <= 0, title: "The step before", "aria-label": "The step before", onclick: () => show(index - 1) }, "‹"),
    h("div", { class: "step-picks", role: "group", "aria-label": "Steps" }, report.steps.map((step) =>
      h("button", {
        class: ["step-pick", step.diverged && "diverged", step.index === index && "current"], type: "button",
        "aria-pressed": String(step.index === index), title: said(step), onclick: () => show(step.index),
      }, String(step.index)))),
    h("button", { class: "btn btn-small", type: "button", disabled: index >= last, title: "The step after", "aria-label": "The step after", onclick: () => show(index + 1) }, "›"),
    h("span", { class: "faint small" }, "Shaded where the runs split."));
}

/** What the runs did at a step, in a sentence: split there, first or again, or all alike — before
 *  the first split, or again after it. A run that had ended is no behaviour of its own here. */
function stepNote(report, step) {
  const at = report.divergence_index;
  if (!step.diverged) {
    if (at === null) return "Every run did the same here, as at every step.";
    return step.index < at ? `Every run did the same here. They first split at step ${at}.` : `Every run did the same here again, after splitting at step ${at}.`;
  }
  const ended = step.groups.filter((group) => group.outcome.kind === "ended").reduce((total, group) => total + group.run_ids.length, 0);
  const behaviours = step.groups.length - (ended ? 1 : 0);
  const gone = ended ? ` ${plural(ended, "run")} had ended before this step.` : "";
  return behaviours > 1
    ? `The runs split here${step.index === at ? " for the first time" : ""}, into ${plural(behaviours, "behaviour")}.${gone}`
    : `The runs that got this far did the same here.${gone}`;
}

/** The task's check run along its runs (the `check-along` job): on each run's workspace as it
 *  started, after each call that changed it, and at its end — so the tally names the call that
 *  made each finding, and a mistake every run made is named as one. */
function alongPanel(task) {
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
    startLabel: "Run the check along",
    onStarted: (job) => (location.hash = `#/jobs/${job.id}`),
  });
  return panel("Run the check along the runs", form.element, {
    meta: h("span", { class: "muted" }, "Which call made each finding, in each run: what the tally then says."),
  });
}

/** What goes with passing, over the runs that have a verdict: the step whose behaviour best
 *  predicts it, and the tool whose use does, wherever in a run it was called — the first split
 *  is often neither. Fisher's exact test, a lead under p 0.2, never a verdict. The page leads
 *  with it: the best lead in a callout, with the ways to its step. */
function predictorsView(report) {
  const byStep = report.verdict_split?.best;
  const byTool = report.tool_split?.best;
  const judged = report.verdict_split?.judged || report.tool_split?.judged || 0;
  if (judged < 2) return null;
  // How many of the judged runs passed: every run took some behaviour at the first step.
  const passed = (report.steps?.[0]?.groups || []).reduce((sum, group) => sum + (group.passed?.successes || 0), 0);
  if (passed === 0 || passed === judged) {
    return panel("What goes with passing", h("p", { class: "hint" }, passed === 0
      ? `None of the ${judged} judged runs passed, so nothing here can separate passing from failing. A test at a step, judged by whole runs, has passing forks to compare: its page asks Where does it hold? of them.`
      : `All ${judged} judged runs passed: there is no failure here to separate.`));
  }
  let body;
  if (byStep) body = [stepLead(byStep), byTool ? toolLine(byTool) : null];
  else if (byTool) body = toolLead(byTool);
  else {
    body = h("p", { class: "hint" }, `No behaviour at any step, and no tool called, separates the passing runs from the failing ones (of ${judged} judged): the difference is in what they wrote. What the checks said, and Where does it hold?, look there.`);
  }
  return panel("What goes with passing", body, {
    meta: h("span", { class: "muted", title: "Fisher's exact test over the judged runs" },
      "A lead, never a verdict: each split shown has p under 0.2, and with so many compared, chance alone can make one."),
    className: "predictors",
  });
}

// The split's p, kept to four places (`ops.inspect`): one that rounds to nothing was below them.
const pOf = (p) => h("span", { class: "muted", title: "Fisher's exact test over the judged runs" }, p === 0 ? "(p < 0.0001)" : `(p = ${formatP(p)})`);

/** The step whose behaviour best predicts the verdict, in the accent: the numbers it stands on,
 *  then the ways to it — a run that took the behaviour, at the step, and a resample there. For
 *  runs that had ended before the step, the run: it has no such step to resample. */
function stepLead(best) {
  const ended = best.kind === "ended";
  const open = best.run_id
    ? h("a", { class: "btn btn-primary btn-small", href: ended ? `#/runs/${best.run_id}` : `#/runs/${best.run_id}/${best.index}` },
        ended ? `Open run ${short(best.run_id)}` : `Open step ${best.index} in run ${short(best.run_id)}`)
    : null;
  let resample = null;
  if (best.run_id && !ended) {
    resample = live.mode === "live"
      ? h("a", { class: "btn btn-small", href: `#/runs/${best.run_id}/${best.index}?start=resample`, "data-control": true }, `Resample step ${best.index}…`)
      : command(`tuningfork resample ${short(best.run_id)} ${best.index} -n 20`);
  }
  return h("div", { class: "callout" },
    h("div", { class: "callout-big" }, ended
      ? `Ending before step ${best.index} best predicts whether a run passes.`
      : `What the model does at step ${best.index} best predicts whether a run passes.`),
    h("p", null,
      ended
        ? `The runs that had ended before it passed ${best.passed} of ${best.total}. `
        : ["The runs that did “", h("span", { class: "mono" }, best.label), `” there passed ${best.passed} of ${best.total}. `],
      `The others passed ${best.others_passed} of ${best.others_total}. `, pOf(best.p)),
    open || resample ? h("div", { class: "callout-actions" }, open, resample) : null);
}

/** The tool whose use, at whatever step, goes with the verdict, as the lead: no step to open. */
function toolLead(tool) {
  return h("div", { class: "callout" },
    h("div", { class: "callout-big" }, "Calling ", h("span", { class: "mono" }, tool.tool),
      tool.with_passing ? " goes with passing." : " goes with failing."),
    h("p", null, `The runs that called it, at whatever step, passed ${tool.called.passed} of ${tool.called.total}. `,
      `Those that never did passed ${tool.never.passed} of ${tool.never.total}. `, pOf(tool.p)));
}

/** The same, under a step's lead: a line of its own. */
function toolLine(tool) {
  return h("p", null, "The runs that called ", h("span", { class: "mono" }, tool.tool),
    ` passed ${tool.called.passed} of ${tool.called.total}. Those that never did passed ${tool.never.passed} of ${tool.never.total}. `,
    tool.with_passing ? "Its use goes with passing, at whatever step a run called it. " : "Its use goes with failing. ", pOf(tool.p));
}

/** Each run's chip, coloured by its own verdict: which behaviours went on to pass is seen at a
 *  glance, while "then passed" — a rate, with its interval — stays the accent's. */
const RUN_VERDICT = {
  passed: ["ok", "passed its check"],
  failed: ["bad", "failed its check"],
  "not counted": ["warn", "not counted: left out of every pass rate"],
};

function groupsView(step, index, verdicts) {
  const judged = Object.values(verdicts).some((verdict) => verdict === "passed" || verdict === "failed");
  return h("div", { class: "stack" },
    h("div", { class: "groups" }, step.groups.map((group, position) =>
      h("div", { class: "group" },
        h("span", { class: "letter" }, LETTERS[position % LETTERS.length]),
        h("div", null, outcome(group.outcome)),
        h("div", { class: "group-rates" },
          h("span", null, h("span", { class: "what" }, "of runs"), rateBar(group.rate)),
          group.passed ? h("span", null, h("span", { class: "what" }, "then passed"), rateBar(group.passed)) : null),
        h("div", { class: "run-chips" },
          group.run_ids.slice(0, 16).map((id) => {
            const [tone, said] = RUN_VERDICT[verdicts[id]] || ["muted", "no verdict"];
            // A run that had ended has no such step: the chip opens the run.
            const href = group.outcome.kind === "ended" ? `#/runs/${id}` : `#/runs/${id}/${index}`;
            return h("a", { class: `chip tone-${tone} mono`, href, title: `run ${short(id)}: ${said}` }, short(id));
          }),
          group.run_ids.length > 16 ? h("span", { class: "faint small" }, `+${group.run_ids.length - 16} more`) : null)))),
    judged
      ? h("p", { class: "hint" }, `Each run is coloured by its check: green passed, red failed${
          Object.values(verdicts).includes("not counted") ? ", amber not counted" : ""}.`)
      : null);
}

function nextStep(step, index, { resampledAbove = false } = {}) {
  // The behaviours runs took at the step, each with its letter: a run that had ended took none.
  const took = step.groups.map((group, position) => [group, LETTERS[position % LETTERS.length]])
    .filter(([group]) => group.outcome.kind !== "ended");
  if (!took.length) return null;
  const example = took[0][0].run_ids[0];
  return h("div", null,
    resampledAbove ? null : [
      h("p", { class: "hint", style: { marginBottom: "6px" } }, "A handful of runs cannot tell you how often each happens. Resample the step to measure it:"),
      live.mode === "live"
        ? h("a", { class: "btn btn-primary", href: `#/runs/${example}/${index}?start=resample`, "data-control": true }, `Resample step ${index}…`)
        : command(`tuningfork resample ${short(example)} ${index} -n 20`),
    ],
    h("p", { class: "hint", style: { margin: resampledAbove ? "0 0 6px" : "10px 0 6px" } },
      `${resampledAbove ? "Look" : "Or look"} inside the step: how sure ${took.length > 1 ? "each run's model was of each token, and which token split them" : "the model was of each token"}.`),
    h("div", { class: "run-chips" }, took.map(([group, letter]) =>
      h("a", { class: "btn btn-small", href: `#/runs/${group.run_ids[0]}/${index}?view=tokens` },
        `${letter}: run ${short(group.run_ids[0])}'s tokens`))));
}

/** Every step in a table, the behaviours at each: `link(index)` gives each step's number the
 *  props that show it in the section above. */
function stepsTable(report, link) {
  const behaviours = (step) =>
    h("div", { class: "stack", style: { gap: "4px" } }, step.groups.map((group) =>
      h("div", null, outcome(group.outcome), h("span", { class: "faint small" }, `  ×${group.run_ids.length}`))));
  const rows = report.steps.map((step) =>
    h("tr", { class: [step.diverged && "diverged", step.index === report.divergence_index && "first-divergence"] },
      h("td", { class: "num" }, h("a", { ...link(step.index), title: `Show step ${step.index} above, its runs side by side` }, String(step.index))),
      h("td", null, behaviours(step))));
  return h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, "behaviours"))),
    h("tbody", null, rows)));
}
