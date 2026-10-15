// What experiments found: behaviour distributions and comparisons, shown the same on an
// experiment's page and on the page of the job that ran it.

import { chip, h, loading, outcome, panel, plural, rateBar } from "./dom.js";
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

export function comparisonView(experiment) {
  const comparison = experiment.comparison;
  const target = comparison.target;
  const parts = [];
  if (target) {
    parts.push(h("div", { class: ["verdict-block", target.significant ? "significant" : "not-significant"] },
      h("div", null,
        h("div", { class: "verdict-big" }, target.significant ? "The variant shifts the target" : "No significant shift"),
        h("div", { class: "muted small" }, `Target declared before sampling: ${target.description}`)),
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
    h("tbody", null, comparison.shifts.map((shift) =>
      h("tr", null,
        h("td", null, h("span", { class: "chip tone-accent" }, shift.letter)),
        h("td", null, outcome(shift.outcome)),
        h("td", null, rateBar(shift.original), reasonedText(shift.original_reasoned, shift.original.successes)),
        h("td", null, rateBar(shift.variant, "response"), reasonedText(shift.variant_reasoned, shift.variant.successes)),
        h("td", { class: "num mono" }, formatP(shift.p_value)),
        comparison.exploratory ? h("td", { class: "num mono" }, formatP(shift.adjusted_p)) : null))))));
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
      h("div", null,
        h("div", { class: "verdict-big" }, flips
          ? `${plural(flips, "forced token")} changed what the model did`
          : "No forced token changed what the model did"),
        h("div", { class: "muted small" }, `Target declared before sampling: ${report.target}`)),
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

/** A screen: each proposal against a fresh arm of the original, Holm-corrected over the
 *  proposals planned before sampling. `arms` names each arm's experiment, to link to it; an
 *  unfinished screen shows its rates and no verdict. */
export function screenView(screen, { arms = {} } = {}) {
  const survivors = new Set(screen.survivors);
  const [headline, tone] = !screen.complete
    ? ["Unfinished: rates only, no verdict", "not-significant"]
    : survivors.size
      ? [`${plural(survivors.size, "proposal")} shifted the target`, "significant"]
      : ["No proposal shifted the target past the correction", "not-significant"];
  const verdict = (arm) => {
    if (!screen.complete) return chip("no verdict", "warn");
    if (arm.survives) return chip("survives", "ok");
    if (arm.worse) return chip("worse", "bad");
    return chip("not significant", "muted");
  };
  const drawn = (arm) => (arm.drawn < arm.planned ? chip(`${arm.drawn} of ${arm.planned} drawn`, "warn") : null);
  const name = (arm) => (arms[arm.name]?.id
    ? h("a", { class: "mono", href: `#/experiments/${arms[arm.name].id}` }, arm.name)
    : h("span", { class: "mono" }, arm.name));
  const names = ["original", ...screen.arms.map((arm) => arm.name)];
  return h("div", { class: "stack" },
    panel("Each proposal against the original", [
      h("div", { class: ["verdict-block", tone] },
        h("div", null,
          h("div", { class: "verdict-big" }, headline),
          h("div", { class: "muted small" }, `Target declared before sampling: ${screen.target}`)),
        h("div", { class: "mono small" }, `Fisher exact against the original, Holm-corrected over the ${plural(screen.family, "proposal")} planned; α = ${screen.alpha}`)),
      screen.complete ? null : h("div", { class: "banner banner-warn" }, h("strong", null, "Not every arm drew what was planned. "),
        "Its rates are over what was drawn, and there is no verdict: stopping when one looks good is how false discoveries are made."),
      h("div", { class: "table-wrap" }, h("table", { class: "table screen-table" },
        h("thead", null, h("tr", null, h("th", null, "arm"), h("th", null, "reached the target"), h("th", { class: "num" }, "p"),
          h("th", { class: "num" }, "Holm p"), h("th", null, ""))),
        h("tbody", null,
          h("tr", null, h("td", null, name({ name: "original" })), h("td", null, rateBar(screen.baseline.target), drawn(screen.baseline)),
            h("td", null, ""), h("td", null, ""), h("td", { class: "muted small" }, "sampled fresh, first")),
          screen.arms.map((arm) => h("tr", null,
            h("td", null, name(arm)),
            h("td", null, arm.target.total ? rateBar(arm.target, "response") : h("span", { class: "muted" }, "none drawn"), drawn(arm)),
            h("td", { class: "num mono" }, formatP(arm.p_value)),
            h("td", { class: "num mono" }, formatP(arm.adjusted_p)),
            h("td", null, verdict(arm))))))),
      h("p", { class: "hint" }, "A screen at one step keeps the history written under the original wording: it is evidence, not proof. Confirm a survivor with full runs."),
    ]),
    screen.behaviours?.length
      ? panel("What the model did, by arm", h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), names.map((arm) => h("th", { class: "num mono" }, arm)))),
          h("tbody", null, screen.behaviours.map((found) => h("tr", null,
            h("td", null, h("span", { class: "chip tone-accent" }, found.letter)),
            h("td", null, outcome(found.outcome)),
            names.map((arm) => h("td", { class: "num" }, String(found.by_arm?.[arm] ?? 0)))))))))
      : null);
}

/** What the checks said, counted: each group's findings, the most common first, each with the
 *  failed runs (or forks) that said it. `only` keeps one condition of a task's tally. */
export function tallyView(tally, { only = null } = {}) {
  const groups = (tally?.groups || []).filter((group) => !only || group.condition?.id === only);
  if (!groups.some((group) => group.failed)) return null;
  const title = tally.scope === "screen" ? "What each arm's failed forks' checks said" : "What the failed runs' checks said";
  return panel(title, [
    ...groups.map((group) => h("div", { class: "stack", style: { gap: "6px", marginBottom: "12px" } },
      h("div", null, h("strong", null, group.label), h("span", { class: "muted small" }, `  failed ${group.failed} of ${group.judged} judged`)),
      group.failed
        ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
            h("thead", null, h("tr", null, h("th", null, "failed runs that said it"), h("th", null, "finding"))),
            h("tbody", null, group.findings.slice(0, 12).map((found) =>
              h("tr", null, h("td", null, rateBar(found.rate, "bad")), h("td", { class: "mono small" }, found.finding,
                // Where the check was run along the runs: the call that made it, passing runs included.
                found.made_text ? h("div", { class: "faint small" }, found.made_text) : null))))))
        : h("p", { class: "muted small" }, "None failed."),
      group.findings.length > 12 ? h("p", { class: "faint small" }, `+${group.findings.length - 12} more finding(s)`) : null,
      group.silent ? h("p", { class: "faint small" }, `${plural(group.silent, "failed run")} whose check printed nothing`) : null)),
    h("p", { class: "hint" }, "Each line a check printed is one finding, its numbers shown as # and its colours as <color>; a run counts once a finding. "
      + "The most common first: the failure to fix first, rather than the first one read."),
  ]);
}

export function formatP(value) {
  if (value === null || value === undefined) return "—";
  return value < 0.001 ? value.toExponential(1) : value.toFixed(3);
}
