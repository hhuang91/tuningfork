// Runs side by side: where they stop agreeing, and what each did there.

import { api, enc, query } from "../api.js";
import {
  chip, command, crumbs, h, loading, outcome, panel, plural, rateBar, replace, short, verdict,
} from "../dom.js";
import { field, input, numberOf, planForm } from "../forms.js";
import { live } from "../live.js";
import { responseView } from "../payload.js";
import { tallyView } from "../results.js";
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
    h("span", { class: "mono small" }, condition.skill_version ? `skill ${condition.skill_version}` : "skill not recorded"),
    file ? chip(`SKILL.md from ${file}`, "skill", condition.skill_file) : null,
  ];
}

function conditionsBanner(task, conditions) {
  return h("div", { class: "banner banner-warn", style: { marginBottom: "14px" } },
    h("strong", null, `These runs span ${conditions.length} conditions. `),
    "Compare within one: ",
    conditions.map((condition, position) => [
      position ? " · " : null,
      h("a", { href: `#/divergence/${enc(task)}@${condition.id}` }, `${condition.skill_version ? `skill ${condition.skill_version}` : "no skill recorded"}${condition.skill_file ? ` (${condition.skill_file.split(/[\\/]/).pop()})` : ""}, ${plural(condition.runs, "run")}`),
    ]));
}

function mixedSkillsBanner() {
  return h("div", { class: "banner banner-warn", style: { marginBottom: "14px" } }, h("strong", null, "These runs used different versions of the skill. "),
    "A difference between them may be the wording change rather than the model.");
}

export async function divergenceView(root, [task, conditionId]) {
  replace(root, loading());
  const summary = (await api.get("/tasks")).find((candidate) => candidate.task_id === task);
  const conditions = summary?.conditions || [];
  const chosen = conditionId ? conditions.find((condition) => condition.id === conditionId) : null;
  if (conditionId && !chosen) throw new Error(`Task ${task} has no runs under condition ${conditionId}.`);
  // One condition: its latest runs. Otherwise the task's latest runs, whatever they were run under.
  const report = chosen
    ? await api.get(`/divergence${query({ runs: chosen.run_ids.slice(0, 50).join(",") })}`)
    : await api.get(`/divergence${query({ task, limit: 50 })}`);
  // What the failed runs' checks said, counted: the condition chosen, or every one.
  const tally = await api.get(`/tally${query({ task })}`).catch(() => null);
  const at = report.divergence_index;
  const split = at === null ? null : report.steps[at];

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
      split ? panel(`At step ${at}`, [groupsView(split, at, report.verdicts || {}), nextStep(split, at)], {
        meta: h("span", { class: "muted" }, "Intervals are Wilson 95%; with few runs they are wide."),
      }) : null,
      predictorsView(report),
      tallyView(tally, { only: chosen?.id || null }),
      live.mode === "live" ? alongPanel(task) : null,
      panel("Where does it hold?", whereForm(task, { condition: chosen?.id || null }), {
        meta: h("span", { class: "muted" }, "One question of every run: the call that made the difference."),
      }),
      panel("Every step", stepsTable(report))),
  );
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
 *  is often neither. Fisher's exact test, a lead under p 0.2, never a verdict. */
function predictorsView(report) {
  const byStep = report.verdict_split?.best;
  const byTool = report.tool_split?.best;
  const judged = report.verdict_split?.judged || report.tool_split?.judged || 0;
  if (judged < 2) return null;
  // How many of the judged runs passed: every run took some behaviour at the first step.
  const passed = (report.steps?.[0]?.groups || []).reduce((sum, group) => sum + (group.passed?.successes || 0), 0);
  if (passed === 0 || passed === judged) {
    return panel("What goes with passing", h("p", { class: "hint" }, passed === 0
      ? `None of the ${judged} judged runs passed: nothing here can separate passing from failing. A screen run to the end has passing forks to compare — its page asks Where does it hold? of them.`
      : `All ${judged} judged runs passed: there is no failure here to separate.`));
  }
  const lines = [];
  if (byStep) {
    // A run that took the behaviour: its step — or, for runs that had ended before it, the run.
    const ended = byStep.kind === "ended";
    const href = byStep.run_id && (ended ? `#/runs/${byStep.run_id}` : `#/runs/${byStep.run_id}/${byStep.index}`);
    lines.push(h("p", null, href ? h("a", { href }, `Step ${byStep.index}`) : `Step ${byStep.index}`,
      ended ? ": the runs that had ended before it" : `: the runs that did “${byStep.label}”`,
      ` passed ${byStep.passed} of ${byStep.total}, the others ${byStep.others_passed} of ${byStep.others_total} `,
      h("span", { class: "muted" }, `(p ${byStep.p})`), " — the step whose behaviour best predicts the verdict."));
  }
  if (byTool) {
    lines.push(h("p", null, "The runs that called ", h("span", { class: "mono" }, byTool.tool),
      ` passed ${byTool.called.passed} of ${byTool.called.total}; those that never did, ${byTool.never.passed} of ${byTool.never.total} `,
      h("span", { class: "muted" }, `(p ${byTool.p})`),
      byTool.with_passing ? " — its use goes with passing, at whatever step a run called it." : " — its use goes with failing."));
  }
  if (!lines.length) {
    lines.push(h("p", { class: "hint" }, `No behaviour at any step, and no tool called, separates the passing runs from the failing ones (of ${judged} judged): the difference is in what they wrote. What the checks said, and Where does it hold?, look there.`));
  }
  return panel("What goes with passing", lines, {
    meta: h("span", { class: "muted" }, "Fisher's exact test over the judged runs: a lead under p 0.2, never a verdict."),
  });
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
            return h("a", { class: `chip tone-${tone} mono`, href: `#/runs/${id}/${index}`, title: `run ${short(id)}: ${said}` }, short(id));
          }),
          group.run_ids.length > 16 ? h("span", { class: "faint small" }, `+${group.run_ids.length - 16} more`) : null)))),
    judged
      ? h("p", { class: "hint" }, `Each run is coloured by its check: green passed, red failed${
          Object.values(verdicts).includes("not counted") ? ", amber not counted" : ""}.`)
      : null);
}

function nextStep(step, index) {
  const example = step.groups[0].run_ids[0];
  return h("div", null,
    h("p", { class: "hint", style: { marginBottom: "6px" } }, "A handful of runs cannot tell you how often each happens. Resample the checkpoint to measure it:"),
    live.mode === "live"
      ? h("a", { class: "btn btn-primary", href: `#/runs/${example}/${index}?start=resample`, "data-control": true }, `Resample step ${index}…`)
      : command(`tuningfork resample ${short(example)} ${index} -n 20`),
    h("p", { class: "hint", style: { margin: "10px 0 6px" } }, "Or look inside the step: how sure each run's model was of each token, and which token split them."),
    h("div", { class: "run-chips" }, step.groups.map((group, position) =>
      h("a", { class: "btn btn-small", href: `#/runs/${group.run_ids[0]}/${index}?view=tokens` },
        `${LETTERS[position % LETTERS.length]}: run ${short(group.run_ids[0])}'s tokens`))));
}

function stepsTable(report) {
  const behaviours = (step) =>
    h("div", { class: "stack", style: { gap: "4px" } }, step.groups.map((group) =>
      h("div", null, outcome(group.outcome), h("span", { class: "faint small" }, `  ×${group.run_ids.length}`))));
  const rows = report.steps.map((step) =>
    h("tr", { class: [step.diverged && "diverged", step.index === report.divergence_index && "first-divergence"] },
      h("td", { class: "num" }, String(step.index)),
      h("td", null, behaviours(step))));
  return h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, "behaviours"))),
    h("tbody", null, rows)));
}
