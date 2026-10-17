// What experiments found: behaviour distributions and comparisons, shown the same on an
// experiment's page and on the page of the job that ran it.

import { chip, h, loading, outcome, panel, plural, rateBar, short } from "./dom.js";
import { heatmap, probeMarks } from "./tokens.js";

export function distributionView(distribution, title = "What the model did") {
  const clusters = distribution.clusters;
  return panel(
    title,
    [
      h("div", { class: "groups" }, clusters.map((cluster) =>
        h("div", { class: "group" },
          h("span", { class: "letter" }, cluster.letter),
          h("div", null, outcome(cluster.outcome)),
          h("div", { class: "group-rates" }, rateBar(cluster.rate), reasonedText(cluster.reasoned, cluster.rate.successes))))),
      distribution.errors
        ? h("p", { class: "hint" }, `${plural(distribution.errors, "sample")} failed at the transport level and are not counted.`)
        : null,
    ],
    { meta: h("span", { class: "muted" }, `${plural(distribution.total, "sample")} · Wilson 95% intervals`) },
  );
}

/** How many of a behaviour's samples came with reasoning — said only where some did not,
 *  while others in the experiment reasoned. */
function reasonedText(reasoned, of) {
  if (reasoned === undefined || reasoned === null || reasoned >= of) return null;
  return h("div", { class: "small warn-text" }, `${reasoned} of ${of} reasoned`);
}

/** What an experiment counted, said first and plainly: every number below it is of this. */
export function targetLine(description, declared = "declared before sampling") {
  return h("div", { class: "verdict-target" },
    h("span", { class: "label" }, "Target"),
    h("span", { class: "verdict-target-text" }, description),
    h("span", { class: "muted small" }, declared));
}

const ofTotal = (rate) => `${rate.successes} of ${rate.total}`;

/** A step's A/B against its declared target, in words: which way the variant moved it, and
 *  by how much. More of a behaviour is better only if it is the one wanted, which tuningfork
 *  is never told: a shift is the accent's colour, never the green of a check that passed. */
function shiftWords(target) {
  if (!target.significant) {
    return `No significant difference: the variant hit the target in ${ofTotal(target.variant)} samples, the original in ${ofTotal(target.original)}`;
  }
  const way = target.direction === "down" ? "less" : "more";
  return `The variant hit the target ${way} often: ${ofTotal(target.variant)} samples, against ${ofTotal(target.original)} with the original`;
}

/** Whether the declared target counts a behaviour: all its samples, some, or none. */
function targetMark(shift) {
  if (!shift.target_counted) return null;
  const pooled = shift.original.successes + shift.variant.successes;
  return shift.target_counted >= pooled
    ? chip("target", "target", "The declared target counts this behaviour: its samples are the target's numbers above")
    : chip(`target in ${shift.target_counted} of ${pooled}`, "target",
        "The declared target counts some of this behaviour's samples: what some of them say meets it, and what others say does not");
}

export function comparisonView(experiment) {
  const comparison = experiment.comparison;
  const target = comparison.target;
  const parts = [];
  if (target) {
    parts.push(h("div", { class: ["verdict-block", target.significant ? "significant" : "not-significant"] },
      targetLine(target.description),
      h("div", { class: "verdict-big" }, shiftWords(target)),
      h("div", { class: "stack", style: { gap: "4px" } },
        h("span", null, h("span", { class: "muted" }, "original  "), rateBar(target.original)),
        h("span", null, h("span", { class: "muted" }, "variant   "), rateBar(target.variant, "response"))),
      h("div", { class: "mono small" }, `Fisher exact p = ${formatP(target.p_value)} (α = ${target.alpha})`)));
    parts.push(h("p", { class: "hint" }, "A checkpoint A/B keeps the history written under the original wording: treat it as a screening result and confirm a winner with full runs."));
  } else {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, "Exploratory. "),
      "No target was declared before sampling, so every behaviour is tested and the p-values are Bonferroni-adjusted."));
  }
  parts.push(h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), h("th", null, "original"), h("th", null, "variant"),
      h("th", { class: "num" }, "p"), comparison.exploratory ? h("th", { class: "num" }, "adjusted p") : null)),
    h("tbody", null, comparison.shifts.map((shift) => {
      const mark = targetMark(shift);
      return h("tr", { class: mark ? "is-target" : null },
        h("td", { class: "nowrap" }, h("span", { class: "chip tone-accent" }, shift.letter), mark ? [" ", mark] : null),
        h("td", null, outcome(shift.outcome)),
        h("td", null, rateBar(shift.original), reasonedText(shift.original_reasoned, shift.original.successes)),
        h("td", null, rateBar(shift.variant, "response"), reasonedText(shift.variant_reasoned, shift.variant.successes)),
        h("td", { class: "num mono" }, formatP(shift.p_value)),
        comparison.exploratory ? h("td", { class: "num mono" }, formatP(shift.adjusted_p)) : null);
    })))));
  // A result a job stored before behaviours said whether the target counts them says nothing.
  const said = comparison.shifts.some((shift) => shift.target_counted !== undefined);
  if (target && said && !comparison.shifts.some((shift) => shift.target_counted)) {
    parts.push(h("p", { class: "hint" }, "The target counts none of these behaviours: neither arm ever did it."));
  }
  return panel("Original wording against the variant", parts);
}

const PART = { reasoning: "reasoning", content: "answer", tool_call: "tool call" };
const pct = (p) => (p === null || p === undefined ? "—" : p >= 0.995 || p === 0 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`);

/** A forking-token probe: at each position, the run's own token and each forced instead,
 *  and which changed how often the model reached the target. With `context` — the
 *  experiment, and `view`, the tokens of the answer it forked from — that answer is shown
 *  with the probed tokens ringed, and each position leads up to its token and over to the
 *  step's own Tokens view. */
export function forkingView(report, context = {}) {
  const { experiment = null, view } = context;
  let map = null;
  let answer = null;
  if (experiment?.run_id && experiment.settings?.base_stream) {
    if (view === undefined) {
      answer = loading("Loading the answer it forked from…");
    } else if (view) {
      map = heatmap(view, { marks: probeMarks([{ experiment, positions: report.positions }]), compact: true });
      answer = h("div", { class: "fork-answer" },
        h("div", { class: "label" }, "The answer it forked from — the probed tokens ringed: solid where a forced token changed what the model did"),
        map);
    }
  }
  const onStep = (position) => experiment?.run_id
    ? h("a", { class: "btn btn-quiet btn-small", href: `#/runs/${experiment.run_id}/${experiment.step_idx}?view=tokens&at=${position}`, title: "This token on the step's own Tokens view" }, "On the step")
    : null;
  const flips = report.positions.reduce((total, position) => total + position.alternatives.filter((alt) => alt.flips).length, 0);
  const parts = [
    h("div", { class: ["verdict-block", flips ? "significant" : "not-significant"] },
      targetLine(report.target),
      h("div", { class: "verdict-big" }, flips
        ? `${plural(flips, "forced token")} changed what the model did`
        : "No forced token changed what the model did"),
      report.reproduced
        ? h("div", { class: "stack", style: { gap: "4px" } },
            h("span", null, h("span", { class: "muted" }, "the run's own tokens reached it  "), rateBar(report.reproduced)))
        : null,
      h("div", { class: "mono small" }, `Fisher exact, Holm-corrected over the ${plural(report.family, "test")} planned; α = ${report.alpha}`)),
  ];
  const undecided = report.missed?.undecided || [];
  const tooFine = report.missed?.too_fine || [];
  if (undecided.length) {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, "Still undecided there. "),
      `At ${positionList(undecided)}, the run's own token led elsewhere more often than to its behaviour: a later token decided it. On the answer's heatmap, select tokens after these and probe them.`));
  }
  if (tooFine.length) {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, "The target is too fine. "),
      `At ${positionList(tooFine)}, the run's own token led mostly to the same tools with other arguments: it may be too fine a behaviour to compare on. A coarser lens (tool_name) or a pattern may say more.`));
  }
  if (!undecided.length && !tooFine.length && report.reproduced && report.reproduced.total && report.reproduced.point < 0.5) {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, "The target is rarely reproduced. "),
      "Even with the run's own tokens forced, the model reached it less than half the time: the target may be too fine to reproduce, or the step not yet decided at these positions."));
  }
  if (answer) parts.push(answer);
  for (const position of report.positions) {
    parts.push(h("div", { class: "fork-position", id: `fork-pos-${position.position}` },
      h("div", { class: "fork-head" },
        h("span", { class: "mono" }, `[${position.position}]`),
        h("span", { class: "muted" }, PART[position.part] || position.part || ""),
        h("span", { class: "mono fork-context" }, position.context),
        h("span", { class: "fork-links" },
          map ? h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => map.reveal(position.position), title: "This token in the answer above" }, "In the answer") : null,
          onStep(position.position))),
      h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "token"), h("th", null, "reached the target"), h("th", { class: "num" }, "p"), h("th", { class: "num" }, "Holm p"), h("th", null, ""))),
        h("tbody", null,
          forkRow("as run", position.baseline, null),
          position.alternatives.map((alt) => forkRow("forced", alt, alt))))),
      position.weighted !== null && position.weighted !== undefined
        ? h("p", { class: "hint" }, `The chance of the target from here, each token weighted by how likely the model thought it: ${pct(position.weighted)}.`)
        : null));
  }
  return h("div", { class: "stack" },
    panel("Which forced tokens changed what the model did", parts),
    distributionView(report.behaviours, "Behaviours, across the probe"));
}

function positionList(positions) {
  const names = positions.map((position) => `[${position}]`);
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

function forkRow(label, arm, tested) {
  const drawn = arm.drawn < arm.planned ? chip(`${arm.drawn} of ${arm.planned} drawn`, "warn") : null;
  return h("tr", null,
    h("td", { class: "muted" }, label),
    h("td", { class: "mono" }, `${JSON.stringify(arm.arm.text)} ${pct(arm.arm.p)}`),
    h("td", null, arm.target.total ? rateBar(arm.target, tested ? "response" : "accent") : h("span", { class: "muted" }, "none drawn"), drawn),
    h("td", { class: "num mono" }, tested ? formatP(tested.p_value) : ""),
    h("td", { class: "num mono" }, tested ? formatP(tested.adjusted_p) : ""),
    h("td", null, tested?.flips ? chip(tested.direction === "away" ? "flips away" : "flips toward", "bad") : null));
}

/** The chip tones for a screen's survivor and for a proposal it found worse: green and red
 *  where it ran to the end, so the target was the task's check; the accent and amber where
 *  the target was a behaviour at the step. */
export function screenTones(reach) {
  return reach === "end" ? ["ok", "bad"] : ["accent", "warn"];
}

/** A screen: each proposal against a fresh arm of the original, Holm-corrected over the
 *  proposals planned before sampling. `arms` names each arm's experiment, to link to it; an
 *  unfinished screen shows its rates and no verdict. */
export function screenView(screen, { arms = {} } = {}) {
  const survivors = new Set(screen.survivors);
  // Run to the end, the target is the task's check: green and red, as for a run.
  const [up, down] = screenTones(screen.reach);
  const [headline, tone] = screen.quick
    ? ["A quick look: rates only, never a verdict", "not-significant"]
    : !screen.complete
      ? ["Unfinished: rates only, no verdict", "not-significant"]
      : survivors.size
        ? [`${plural(survivors.size, "proposal")} hit the target more often than the original, past the correction`,
            screen.reach === "end" ? "significant tone-ok" : "significant"]
        : ["No proposal hit the target more often than the original, past the correction", "not-significant"];
  const verdict = (arm) => {
    if (screen.quick) return chip("quick look", "muted");
    if (!screen.complete) return chip("no verdict", "warn");
    if (arm.survives) return chip("survives", up);
    if (arm.worse) return chip("worse", down);
    return chip("not significant", "muted");
  };
  const drawn = (arm) => (arm.drawn < arm.planned ? chip(`${arm.drawn} of ${arm.planned} drawn`, "warn") : null);
  const name = (arm) => (arms[arm.name]?.id
    ? h("a", { class: "mono", href: `#/experiments/${arms[arm.name].id}` }, arm.name)
    : h("span", { class: "mono" }, arm.name));
  const names = ["original", ...screen.arms.map((arm) => arm.name)];
  // Where the original's samples came from: none, for a quick look the refinement had drawn no
  // original for; partly an earlier screen's, for a screen to the end that reused them.
  const originalNote = screen.quick && !screen.baseline.target.total
    ? "none drawn: a quick look forks the proposals alone"
    : screen.reused
      ? `${screen.reused} of its forks drawn by this refinement's earlier screens`
      : "sampled fresh, first";
  // At the step, the behaviours by arm are what moved; to the end, each fork's first live call
  // is its own — a table as long as the forks, kept folded.
  const behaviours = screen.behaviours?.length
    ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), names.map((arm) => h("th", { class: "num mono" }, arm)))),
        h("tbody", null, screen.behaviours.map((found) => h("tr", null,
          h("td", null, h("span", { class: "chip tone-accent" }, found.letter)),
          h("td", null, outcome(found.outcome)),
          names.map((arm) => h("td", { class: "num" }, String(found.by_arm?.[arm] ?? 0))))))))
    : null;
  return h("div", { class: "stack" },
    panel(screen.quick ? "Each proposal's forks" : "Each proposal against the original", [
      h("div", { class: ["verdict-block", tone] },
        targetLine(screen.target),
        h("div", { class: "verdict-big" }, headline),
        h("div", { class: "mono small" }, screen.quick
          ? "No test: a lead for the next edit — only a full screen can keep one"
          : `Fisher exact against the original, Holm-corrected over the ${plural(screen.family, "proposal")} planned; α = ${screen.alpha}`)),
      screen.complete ? null : h("div", { class: "banner banner-warn" }, h("strong", null, "Not every arm drew what was planned. "),
        "Its rates are over what was drawn, and there is no verdict: stopping when one looks good is how false discoveries are made."),
      h("div", { class: "table-wrap" }, h("table", { class: "table screen-table" },
        h("thead", null, h("tr", null, h("th", null, "arm"), h("th", null, "reached the target"), h("th", { class: "num" }, "p"),
          h("th", { class: "num" }, "Holm p"), h("th", null, ""))),
        h("tbody", null,
          h("tr", null, h("td", null, name({ name: "original" })),
            h("td", null, screen.baseline.target.total ? rateBar(screen.baseline.target) : h("span", { class: "muted" }, "none drawn"),
              screen.quick ? null : drawn(screen.baseline)),
            h("td", null, ""), h("td", null, ""), h("td", { class: "muted small" }, originalNote)),
          screen.arms.map((arm) => h("tr", null,
            h("td", null, name(arm)),
            h("td", null, arm.target.total ? rateBar(arm.target, "response") : h("span", { class: "muted" }, "none drawn"), drawn(arm)),
            h("td", { class: "num mono" }, formatP(arm.p_value)),
            h("td", { class: "num mono" }, formatP(arm.adjusted_p)),
            h("td", null, verdict(arm))))))),
      h("p", { class: "hint" }, screen.reach === "end"
        ? "Each fork replays its run as recorded up to the step, the workspace included, and runs live from there to the end, where the task's check decides. Confirm a survivor with fresh runs."
        : "A screen at one step keeps the history written under the original wording: it is evidence, not proof. Confirm a survivor with full runs."),
    ]),
    checkpointsView(screen, names),
    !behaviours ? null : screen.reach === "end"
      ? h("details", { class: "fold" },
          h("summary", null, `What each fork's model did at the step: ${plural(screen.behaviours.length, "behaviour")}, by arm`),
          behaviours)
      : panel("What the model did, by arm", behaviours));
}

/** A screen to the end drawn from several runs, run by run: each arm's passes at each, and for
 *  each proposal at how many it beat the original there. The pooled test is the verdict; an
 *  effect at one checkpoint alone lived in that run's prefix, which fresh runs draw anew. */
function checkpointsView(screen, names) {
  const rows = screen.checkpoints || [];
  if (!rows.length) return null;
  const across = screen.across_checkpoints || {};
  return panel("Run by run", [
    h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, "forked from"), h("th", null, "that run"),
        names.map((arm) => h("th", { class: "num mono" }, arm)))),
      h("tbody", null, rows.map((entry) => h("tr", null,
        h("td", null, h("a", { class: "mono", href: `#/runs/${entry.run}/${entry.step}` }, `run ${short(entry.run)}, step ${entry.step}`)),
        h("td", null, entry.verdict ? chip(entry.verdict, entry.verdict === "passed" ? "ok" : "bad") : null),
        names.map((arm) => {
          const counts = entry.arms?.[arm];
          return h("td", { class: "num mono" }, counts ? `${counts.passed}/${counts.of}` : "—");
        })))))),
    Object.entries(across).map(([arm, tally]) => h("p", { class: "small" },
      h("span", { class: "mono" }, arm), `: above the original at ${tally.above} of ${tally.of} checkpoints`,
      tally.of > 1 && tally.above === tally.of ? " — an effect that holds across prefixes"
        : tally.above === 1 ? " — at one alone: it may have lived in that run's prefix" : "")),
  ]);
}

/** What the checks said, counted: each group's findings, the most common first, each with the
 *  failed runs (or forks) that said it. `only` keeps one condition of a task's tally. */
export function tallyView(tally, { only = null } = {}) {
  const groups = (tally?.groups || []).filter((group) => !only || group.condition?.id === only);
  if (!groups.some((group) => group.failed)) return null;
  const forks = tally.scope === "screen";
  const title = forks ? "What each arm's failed forks' checks said" : "What the failed runs' checks said";
  const failed = forks ? "failed forks" : "failed runs";
  return panel(title, [
    ...groups.map((group) => h("div", { class: "stack", style: { gap: "6px", marginBottom: "12px" } },
      h("div", null, h("strong", null, group.label), h("span", { class: "muted small" }, `  failed ${group.failed} of ${group.judged} judged`)),
      group.failed && group.findings.length
        ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
            h("thead", null, h("tr", null, h("th", null, `${failed} that printed it`), h("th", null, "a line the check printed"))),
            h("tbody", null, group.findings.slice(0, 12).map((found) =>
              h("tr", null, h("td", null, rateBar(found.rate, "bad")), h("td", { class: "mono small" }, found.finding,
                // Where the check was run along the runs: the call that made it, passing runs included.
                found.made_text ? h("div", { class: "faint small" }, found.made_text) : null))))))
        : group.failed ? null : h("p", { class: "muted small" }, "None failed."),
      group.findings.length > 12 ? h("p", { class: "faint small" }, `+${group.findings.length - 12} more line(s)`) : null,
      group.silent ? h("p", { class: "faint small" }, `${plural(group.silent, forks ? "failed fork" : "failed run")} whose check printed nothing`) : null)),
    h("p", { class: "hint" }, `Each bar counts the ${failed} that printed the line: once each, however many times it printed it. `
      + "Lines that differ only in their numbers or colour codes count as one, shown with # for each number and <color> for each colour. "
      + "The most common comes first: it is the failure to fix first."),
  ], {
    meta: h("span", { class: "muted" }, `Every ${forks ? "failed fork" : "failed run"}'s check output, line by line`),
  });
}

export function formatP(value) {
  if (value === null || value === undefined) return "—";
  return value < 0.001 ? value.toExponential(1) : value.toFixed(3);
}
