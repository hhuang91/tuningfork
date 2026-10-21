// What experiments found: behaviour distributions and comparisons, drawn on an experiment's
// page — and, in a line, on the page of the job that made it (`found`), with that page's own
// verdict and cautions where it leads with them.

import { capital, chip, h, loading, outcome, panel, plural, rateBar, short } from "./dom.js";
import { heatmap, probeMarks } from "./tokens.js";
import { LABELS } from "./words.js";

/** A panel's "how many · 95% intervals" note, the test's name on hover. */
const intervals = (count, noun) =>
  h("span", { class: "muted", title: "Wilson score intervals" }, `${plural(count, noun)} · 95% intervals`);

/** Samples whose call failed, said in a sentence: they are in no rate. */
const notCounted = (count, noun) => `${plural(count, noun)} ${count === 1 ? "is" : "are"} not counted, because the call itself failed.`;

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
      distribution.errors ? h("p", { class: "hint" }, notCounted(distribution.errors, "sample")) : null,
    ],
    { meta: intervals(distribution.total, "sample") },
  );
}

/** How many of a behaviour's samples came with reasoning — said only where some did not,
 *  while others in the experiment reasoned. */
function reasonedText(reasoned, of) {
  if (reasoned === undefined || reasoned === null || reasoned >= of) return null;
  return h("div", { class: "small warn-text" }, `${reasoned} of ${of} reasoned`);
}

/** What an experiment counted, said first and plainly: every number below it is of this. A
 *  refinement's tests name it as the refinement does: what the model should do. */
export function targetLine(description, declared = "chosen before sampling", label = "What you count") {
  return h("div", { class: "verdict-target" },
    h("span", { class: "label" }, label),
    h("span", { class: "verdict-target-text" }, description),
    h("span", { class: "muted small" }, declared));
}

const ofTotal = (rate) => `${rate.successes} of ${rate.total}`;

/** A step's A/B against its declared target, in words: which way the variant moved it, and
 *  by how much. More of a behaviour is better only if it is the one wanted, which tuningfork
 *  is never told: a shift is the accent's colour, never the green of a check that passed. */
function shiftWords(target) {
  if (!target.significant) {
    return `${capital(LABELS.withinChance)}: what you count happened in ${ofTotal(target.variant)} samples with the variant, and ${ofTotal(target.original)} with the original.`;
  }
  const way = target.direction === "down" ? "less" : "more";
  return `What you count happened ${way} often with the variant: ${ofTotal(target.variant)} samples, against ${ofTotal(target.original)} with the original. That is ${LABELS.beyondChance}.`;
}

/** Whether the declared target counts a behaviour — all `pooled` of its samples, some, or
 *  none — as the badge its row starts with: a row's own rate, 6/6 say, is how often the
 *  behaviour happened, and only these say whether that is the target. Nothing where there is
 *  no target, or a result stored before behaviours said. */
function targetMark(counted, pooled) {
  if (counted === undefined || counted === null) return null;
  if (!counted) {
    return chip(LABELS.noMatch, "not-target", "This behaviour does not match what was chosen before sampling: none of its samples are among the ones counted.");
  }
  return counted >= pooled
    ? chip(LABELS.matches, "target", "This behaviour matches what was chosen before sampling: its samples are the ones counted.")
    : chip(`${LABELS.matches} in ${counted} of ${pooled}`, "target",
        "Some of this behaviour's samples match what was chosen before sampling: what some of them say meets it, and what others say does not.");
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
        h("span", null, h("span", { class: "muted" }, "variant   "), rateBar(target.variant, "change"))),
      h("div", { class: "mono small", title: "Fisher's exact test" }, `p = ${formatP(target.p_value)}, α = ${target.alpha}`)));
    parts.push(h("p", { class: "hint" }, "This A/B changed only this step. Everything before it still comes from the original wording, so treat the result as a lead, and confirm it with full runs."));
  } else {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, "Nothing declared. "),
      "Nothing to count was chosen before sampling, so every behaviour is compared, and each p-value is adjusted for comparing them all (Bonferroni)."));
  }
  parts.push(h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), h("th", null, "original"), h("th", null, "variant"),
      h("th", { class: "num" }, "p"), comparison.exploratory ? h("th", { class: "num", title: "Bonferroni-adjusted" }, "p, adjusted") : null)),
    h("tbody", null, comparison.shifts.map((shift) => {
      const mark = targetMark(shift.target_counted, shift.original.successes + shift.variant.successes);
      return h("tr", { class: shift.target_counted ? "is-target" : null },
        h("td", { class: "nowrap" }, h("span", { class: "chip tone-accent" }, shift.letter), mark ? [" ", mark] : null),
        h("td", null, outcome(shift.outcome)),
        h("td", null, rateBar(shift.original), reasonedText(shift.original_reasoned, shift.original.successes)),
        h("td", null, rateBar(shift.variant, "change"), reasonedText(shift.variant_reasoned, shift.variant.successes)),
        h("td", { class: "num mono" }, formatP(shift.p_value)),
        comparison.exploratory ? h("td", { class: "num mono" }, formatP(shift.adjusted_p)) : null);
    })))));
  // A result a job stored before behaviours said whether the target counts them says nothing.
  const said = comparison.shifts.some((shift) => shift.target_counted !== undefined);
  if (target && said && !comparison.shifts.some((shift) => shift.target_counted)) {
    parts.push(h("p", { class: "hint" }, "What you count includes none of these behaviours: neither wording ever did it."));
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
        h("div", { class: "label" }, "The answer it forked from. The probed tokens are ringed, doubled where a forced token changed what the model did."),
        map);
    }
  }
  const onStep = (position) => experiment?.run_id
    ? h("a", { class: "btn btn-quiet btn-small", href: `#/runs/${experiment.run_id}/${experiment.step_idx}?view=tokens&at=${position}`, title: "This token on the step's own Tokens view" }, "On the step")
    : null;
  const [headline, tone, seen] = forkingHeadline(report);
  const parts = [
    h("div", { class: ["verdict-block", tone] },
      targetLine(report.target),
      h("div", { class: "verdict-big" }, headline),
      seen ? h("p", { class: "verdict-lines" }, seen) : null,
      report.reproduced
        ? h("div", { class: "stack", style: { gap: "4px" } },
            h("span", null, h("span", { class: "muted" }, "the run's own tokens led to it  "), rateBar(report.reproduced)))
        : null,
      h("div", { class: "mono small", title: "Fisher's exact test, Holm-corrected" },
        `Each p corrected for the ${plural(report.family, "test")} planned; α = ${report.alpha}`)),
  ];
  for (const [title, text, here] of forkingCautions(report)) {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, `${title} `), here ? `${text} ${here}` : text));
  }
  if (answer) parts.push(answer);
  // Where each token's continuations went, by behaviour: the behaviours' table, inverted.
  const ledTo = new Map();
  for (const cluster of report.behaviours.clusters) {
    for (const arm of cluster.by_arm || []) {
      const key = armKey(arm.position, arm.text);
      if (!ledTo.has(key)) ledTo.set(key, []);
      ledTo.get(key).push([cluster.letter, arm.count]);
    }
  }
  const said = report.behaviours.clusters.some((cluster) => cluster.by_arm);
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
        h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "token"), h("th", null, "did what you count"), h("th", { class: "num" }, "p"),
          h("th", { class: "num", title: "Holm-corrected" }, "p, corrected"), h("th", null, ""),
          said ? h("th", null, "led to") : null)),
        h("tbody", null,
          forkRow("as run", position.baseline, null, said && ledTo),
          position.alternatives.map((alt) => forkRow("forced", alt, alt, said && ledTo))))),
      position.weighted !== null && position.weighted !== undefined
        ? h("p", { class: "hint" }, `The chance of what you count from here, each token weighted by how likely the model thought it: ${pct(position.weighted)}.`)
        : null));
  }
  return h("div", { class: "stack" },
    panel("Which forced tokens changed what the model did", parts),
    said ? probeBehavioursView(report.behaviours) : distributionView(report.behaviours, "Behaviours, across the probe"));
}

const armKey = (position, text) => `${position}\u0000${text}`;

/** A probe's behaviours, each badged as its target counts it, with the tokens whose
 *  continuations reached it — the run's own in grey, a forced one in the accent — each leading
 *  to its position's table above. A forced token can change what the model does and still
 *  miss a target too fine to count it: this is where that shows. */
function probeBehavioursView(behaviours) {
  const jump = (position) => (event) => {
    event.preventDefault();
    document.getElementById(`fork-pos-${position}`)?.scrollIntoView({ block: "start" });
  };
  return panel("Behaviours, across the probe", [
    h("div", { class: "table-wrap" }, h("table", { class: "table probe-behaviours" },
      h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), h("th", null, "of every continuation"), h("th", null, "reached from"))),
      h("tbody", null, behaviours.clusters.map((cluster) => {
        const mark = targetMark(cluster.target_counted, cluster.rate.successes);
        return h("tr", { class: cluster.target_counted ? "is-target" : null },
          h("td", { class: "nowrap" }, h("span", { class: "chip tone-accent" }, cluster.letter), mark ? [" ", mark] : null),
          h("td", null, outcome(cluster.outcome)),
          h("td", null, rateBar(cluster.rate)),
          h("td", null, h("span", { class: "led-by" }, (cluster.by_arm || []).map((arm) =>
            h("button", {
              class: ["chip", arm.original ? "tone-muted" : "tone-accent"], type: "button", onclick: jump(arm.position),
              title: `${arm.original ? "The run's own token" : "A token forced"} at [${arm.position}]: ${plural(arm.count, "continuation")} did this. Its table above.`,
            }, `[${arm.position}] ${JSON.stringify(arm.text)} ×${arm.count}`)))));
      })))),
    behaviours.errors ? h("p", { class: "hint" }, notCounted(behaviours.errors, "continuation")) : null,
  ], { meta: intervals(behaviours.total, "continuation") });
}

function positionList(positions) {
  const names = positions.map((position) => `[${position}]`);
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** One token's row at a position. A flip is the accent's colour, its direction in words: the
 *  target is what tuningfork was told to count, never a check that passed. `ledTo`, where the
 *  report says it: the behaviours each token's continuations reached, by letter. */
function forkRow(label, arm, tested, ledTo = null) {
  const drawn = arm.drawn < arm.planned ? chip(`${arm.drawn} of ${arm.planned} drawn`, "warn") : null;
  const went = ledTo ? ledTo.get(armKey(arm.arm.position, arm.arm.text)) || [] : null;
  return h("tr", null,
    h("td", { class: "muted" }, label),
    h("td", { class: "mono" }, `${JSON.stringify(arm.arm.text)} ${pct(arm.arm.p)}`),
    h("td", null, arm.target.total ? rateBar(arm.target, tested ? "change" : "accent") : h("span", { class: "muted" }, "none drawn"), drawn),
    h("td", { class: "num mono" }, tested ? formatP(tested.p_value) : ""),
    h("td", { class: "num mono" }, tested ? formatP(tested.adjusted_p) : ""),
    h("td", null, tested?.flips ? chip(tested.direction === "away" ? "flips away" : "flips toward", "accent") : null),
    went ? h("td", { class: "nowrap small" }, went.map(([letter, count]) => `${letter} ×${count}`).join(" · ")) : null);
}

/** The chip tones for a screen's survivor and for a proposal it found worse: green and red
 *  where it ran to the end, so the target was the task's check; the accent and amber where
 *  the target was a behaviour at the step. */
export function screenTones(reach) {
  return reach === "end" ? ["ok", "bad"] : ["accent", "warn"];
}

/** A forking-token probe in a line, its block's tone, and — where no forced token changed what the
 *  model did — what the probe could have seen: a null is what it could see, never that nothing
 *  changed. */
export function forkingHeadline(report) {
  const flips = report.positions.reduce((total, position) => total + position.alternatives.filter((alt) => alt.flips).length, 0);
  const seen = !flips && report.per_arm
    ? report.detectable
      ? `With ${report.per_arm} continuations a token, only a token that changes about ${report.detectable} of its ${report.per_arm} would show. A smaller shift can be real and go unseen.`
      : `With ${report.per_arm} continuations a token, no change could be told from chance. Draw more.`
    : null;
  return [flips ? `${plural(flips, "forced token")} changed what the model did` : "No forced token decided what the model did",
    flips ? "significant" : "not-significant", seen];
}

/** What a probe leaves open, as cautions: each its title, what was seen, and — where the next
 *  step is taken on the probe's own page — that step, which only that page shows. */
export function forkingCautions(report) {
  const undecided = report.missed?.undecided || [];
  const tooFine = report.missed?.too_fine || [];
  const cautions = [];
  if (undecided.length) {
    cautions.push(["Still undecided there.",
      `At ${positionList(undecided)}, the run's own token led elsewhere more often than to its own behaviour, so a later token decided it.`,
      "On the answer's heatmap, select tokens after these and probe them."]);
  }
  if (tooFine.length) {
    cautions.push(["What you count is too fine.",
      `At ${positionList(tooFine)}, the run's own token led mostly to the same tools with other arguments, so what you count may be too fine to compare on. Grouping behaviours by tool name alone (tool_name), or a pattern, may say more.`]);
  }
  if (!undecided.length && !tooFine.length && report.reproduced && report.reproduced.total && report.reproduced.point < 0.5) {
    cautions.push(["What you count is rarely reproduced.",
      "Even with the run's own tokens forced, the model did it less than half the time. What you count may be too fine to reproduce, or the step not yet decided at these positions."]);
  }
  return cautions;
}

/** A test at a step that did not draw all it planned: its rates, and no verdict. */
const SCREEN_UNFINISHED = ["Not every wording drew what was planned.",
  "Its rates are over what was drawn, with no verdict. Stopping when one looks good is how false discoveries are made."];

/** What a test's wording left out of its count (`errors`), in words: at the step, samples whose
 *  call failed; to the end, forks something kept from measuring the wording, or that no check
 *  judged (`fork_counted`) — one that timed out counts, as a failure. */
function screenLeftOut(arm, whose, reach) {
  const count = arm.errors || 0;
  if (!count) return [];
  const are = count === 1 ? "is" : "are";
  return [reach === "end"
    ? `${plural(count, "fork")} of ${whose} ${are} not counted: something kept ${count === 1 ? "it" : "each"} from measuring the wording, or no check judged it.`
    : `${plural(count, "sample")} of ${whose} ${are} not counted, because the call itself failed.`];
}

/** A test at a step in a line, and its block's tone: a test run to the end is judged by the
 *  task's check, so what beat the original there is green, as a check that passed is. */
export function screenHeadline(screen) {
  const survivors = new Set(screen.survivors);
  return screen.quick
    ? ["A quick look gives rates only, never a verdict.", "not-significant"]
    : !screen.complete
      ? ["This test is unfinished, so it gives rates only, and no verdict.", "not-significant"]
      : survivors.size
        ? [`${plural(survivors.size, "proposal")} beat the original by ${LABELS.beyondChance}.`,
            screen.reach === "end" ? "significant tone-ok" : "significant"]
        : [`No proposal beat the original by ${LABELS.beyondChance}.`, "not-significant"];
}

/** A wording A/B run to the end in a line, and its block's tone: the task's check decides
 *  each fork, so a shift is green or red, as a run's verdict is. */
export function endHeadline(result) {
  const significant = result.significant;
  const up = result.direction === "up";
  const passes = (arm) => `${arm.passed} of ${arm.counted}`;
  const headline = result.p_value === null
    ? "No test: one wording has no fork that counts."
    : significant
      ? `The variant passed the check ${up ? "more" : "less"} often: ${passes(result.variant)} forks, against ${passes(result.original)} with the original. That is ${LABELS.beyondChance}.`
      : `${capital(LABELS.withinChance)}: the variant's forks passed the check in ${passes(result.variant)}, and the original's in ${passes(result.original)}.`;
  return [headline, significant ? `significant ${up ? "tone-ok" : "tone-bad"}` : "not-significant"];
}

/** What a finished job found, in a line, for its page: the verdict the page of what it made
 *  leads with, from the same helpers, where that page has one (a test at a step, an A/B, a
 *  probe); else the result in a line (a resample, a resample to the end). With what was counted
 *  (`target`: its label, what it is, when it was chosen), what the verdict could see (`seen`, as
 *  that page says it), the rates it rests on — each with its interval, and the behaviour where one
 *  is named — and what was left out or is still open, as cautions in that page's words (`notes`;
 *  `caution` where nothing could be counted at all). Null for a job whose result is told
 *  elsewhere, or not at all. An A/B with nothing declared is judged at its own α. */
export function found(job) {
  const result = job.result || {};
  switch (job.kind) {
    case "resample":
    case "hold-sample":
      return result.distribution ? distributionFinding(result.distribution) : null;
    case "ab":
      return result.comparison ? abFinding(result.comparison, job.params?.alpha ?? 0.05) : null;
    case "forking-tokens": {
      if (!result.forking) return null;
      const [text, tone, seen] = forkingHeadline(result.forking);
      // A next step taken on the answer's heatmap is the probe's page's to say.
      const notes = forkingCautions(result.forking).map(([title, said]) => `${title} ${said}`);
      return { text, tone, target: ["What you count", result.forking.target], seen, bars: [], notes };
    }
    case "screen":
    case "screen-to-end": {
      const screen = result.screen;
      if (!screen) return null;
      const [text, tone] = screenHeadline(screen);
      // Run to the end, the task's check decides, whatever the refinement said should happen.
      return {
        text, tone,
        target: screen.reach === "end"
          ? ["What you count", screen.target, "set by the task, before any fork ran"]
          : ["What the model should do", screen.target],
        bars: [["original", screen.baseline?.target], ...screen.arms.map((arm) => [arm.name, arm.target, "change"])],
        notes: [
          ...(screen.complete ? [] : [SCREEN_UNFINISHED.join(" ")]),
          ...[[screen.baseline || {}, "the original"], ...screen.arms.map((arm) => [arm, arm.name])]
            .flatMap(([arm, whose]) => [...screenLeftOut(arm, whose, screen.reach), ...leftOut(arm, whose)]),
        ],
      };
    }
    case "resample-to-end":
      return result.counted === undefined ? null : endArmFinding(result);
    case "ab-to-end": {
      if (!result.original || !result.variant) return null;
      const [text, tone] = endHeadline(result);
      return {
        text, tone, target: CHECK_TARGET,
        bars: [["original", result.original.rate], ["variant", result.variant.rate, "change"]],
        notes: [...leftOut(result.original, "the original"), ...leftOut(result.variant, "the variant")],
      };
    }
    default:
      return null;
  }
}

const CHECK_TARGET = ["What you count", "the task's check passes", "set by the task, before any fork ran"];

function distributionFinding(distribution) {
  const [first] = distribution.clusters;
  const notes = distribution.errors ? [notCounted(distribution.errors, "sample")] : [];
  if (!first) return { text: "No sample came back to count.", caution: true, tone: "not-significant", bars: [], notes };
  const total = distribution.total;
  // Where calls failed, the samples that came back are fewer than were drawn: said "counted".
  const counted = distribution.errors ? " counted" : "";
  const text = distribution.clusters.length === 1
    ? `One behaviour, in ${total === 1 ? "the one sample" : total === 2 ? "both samples" : `all ${total} samples`}${counted}.`
    : `${plural(distribution.clusters.length, "behaviour")} in ${plural(total, "sample")}${counted}; the most common, ${first.letter}, in ${first.rate.successes} of them.`;
  return { text, tone: "not-significant", bars: [[first.letter, first.rate, undefined, first.outcome]], notes };
}

function abFinding(comparison, alpha) {
  const target = comparison.target;
  if (target) {
    return {
      text: shiftWords(target), tone: target.significant ? "significant" : "not-significant",
      target: ["What you count", target.description],
      bars: [["original", target.original], ["variant", target.variant, "change"]], notes: [],
    };
  }
  const moved = comparison.shifts.filter((shift) => shift.adjusted_p < alpha).length;
  return {
    text: moved
      ? `${moved} of ${plural(comparison.shifts.length, "behaviour")} moved between the wordings by ${LABELS.beyondChance}.`
      : `No behaviour moved between the wordings by ${LABELS.beyondChance}.`,
    tone: moved ? "significant" : "not-significant", bars: [],
    notes: ["Nothing was declared before sampling, so every behaviour was compared, each p adjusted for comparing them all."],
  };
}

/** One arm of forks run to the end, in a line: how many passed the task's check, and what was
 *  left out of the count. */
function endArmFinding(arm) {
  if (!arm.counted) {
    return { text: "No fork counts: each was left out of the count, or none ran.", caution: true, tone: "not-significant", bars: [], notes: leftOut(arm) };
  }
  return {
    text: `${arm.passed} of ${plural(arm.counted, "fork")} passed the task's check.`, tone: "not-significant",
    target: CHECK_TARGET, bars: [["passed", arm.rate]], notes: leftOut(arm),
  };
}

/** What an arm of forks left out of its count, or counted as failures, in words. */
function leftOut(arm, whose = null) {
  const of = whose ? ` of ${whose}` : "";
  const excluded = (arm.exclusions || []).length;
  return [
    excluded ? `${plural(excluded, "fork")}${of} ${excluded === 1 ? "is" : "are"} not counted; the experiment's page says why.` : null,
    arm.timeouts ? `${plural(arm.timeouts, "fork")}${of} timed out, counted as ${arm.timeouts === 1 ? "a failure" : "failures"}.` : null,
  ].filter(Boolean);
}

/** A screen: each proposal against a fresh arm of the original, Holm-corrected over the
 *  proposals planned before sampling. `arms` names each arm's experiment, to link to it; an
 *  unfinished screen shows its rates and no verdict. */
export function screenView(screen, { arms = {} } = {}) {
  // Run to the end, the target is the task's check: green and red, as for a run.
  const [up, down] = screenTones(screen.reach);
  const [headline, tone] = screenHeadline(screen);
  const verdict = (arm) => {
    if (screen.quick) return chip("quick look", "muted");
    if (!screen.complete) return chip("no verdict", "warn");
    if (arm.survives) return chip("beat the original", up);
    if (arm.worse) return chip("worse", down);
    return chip("within chance", "muted");
  };
  const drawn = (arm) => (arm.drawn < arm.planned ? chip(`${arm.drawn} of ${arm.planned} drawn`, "warn") : null);
  // What a wording left out of its count, and why: its rate is over the rest.
  const leftOutChip = (arm, whose) => (arm.errors ? chip(`${arm.errors} not counted`, "warn", screenLeftOut(arm, whose, screen.reach)[0]) : null);
  // A fork run to the end that timed out failed, as the gate counts it: said, since it is not the check.
  const timedOut = (arm) => (arm.timeouts ? chip(`${arm.timeouts} timed out`, "warn", "Counted as failures, as the approval checks count them.") : null);
  const name = (arm) => (arms[arm.name]?.id
    ? h("a", { class: "mono", href: `#/experiments/${arms[arm.name].id}` }, arm.name)
    : h("span", { class: "mono" }, arm.name));
  const names = ["original", ...screen.arms.map((arm) => arm.name)];
  // Where the original's samples came from: none, for a quick look the refinement had drawn no
  // original for; partly an earlier screen's, for a screen to the end that reused them.
  const originalNote = screen.quick && !screen.baseline.target.total
    ? "none drawn, since a quick look forks the proposals alone"
    : screen.reused
      ? `${screen.reused} of its forks drawn by this refinement's earlier tests`
      : "sampled fresh, first";
  // At the step, the behaviours by arm are what moved; to the end, each fork's first live call
  // is its own — a table as long as the forks, kept folded.
  const behaviours = screen.behaviours?.length
    ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "behaviour"), names.map((arm) => h("th", { class: "num mono" }, arm)))),
        h("tbody", null, screen.behaviours.map((found) => {
          const mark = targetMark(found.target_counted, found.count);
          return h("tr", { class: found.target_counted ? "is-target" : null },
            h("td", { class: "nowrap" }, h("span", { class: "chip tone-accent" }, found.letter), mark ? [" ", mark] : null),
            h("td", null, outcome(found.outcome)),
            names.map((arm) => h("td", { class: "num" }, String(found.by_arm?.[arm] ?? 0))));
        }))))
    : null;
  return h("div", { class: "stack" },
    panel(screen.quick ? "Each proposal's forks" : "Each proposal against the original", [
      h("div", { class: ["verdict-block", tone] },
        // Run to the end, the task's check decides, whatever the refinement said should happen.
        screen.reach === "end"
          ? targetLine(screen.target, "set by the task, before any fork ran")
          : targetLine(screen.target, undefined, "What the model should do"),
        h("div", { class: "verdict-big" }, headline),
        screen.quick
          ? h("div", { class: "mono small" }, "A lead for the next edit. Only a test at this step that is not a quick look can keep a proposal.")
          : h("div", { class: "mono small", title: "Fisher's exact test, Holm-corrected" },
              `Each against the original, corrected for the ${plural(screen.family, "proposal")} planned; α = ${screen.alpha}`)),
      screen.complete ? null : h("div", { class: "banner banner-warn" }, h("strong", null, `${SCREEN_UNFINISHED[0]} `), SCREEN_UNFINISHED[1]),
      h("div", { class: "table-wrap" }, h("table", { class: "table screen-table" },
        h("thead", null, h("tr", null, h("th", null, "wording"), h("th", null, screen.reach === "end" ? "passed the check" : "did what it should"), h("th", { class: "num" }, "p"),
          h("th", { class: "num", title: "Holm-corrected" }, "p, corrected"), h("th", null, ""))),
        h("tbody", null,
          h("tr", null, h("td", null, name({ name: "original" })),
            h("td", null, screen.baseline.target.total ? rateBar(screen.baseline.target) : h("span", { class: "muted" }, "none drawn"),
              screen.quick ? null : drawn(screen.baseline), leftOutChip(screen.baseline, "the original"), timedOut(screen.baseline)),
            h("td", null, ""), h("td", null, ""), h("td", { class: "muted small" }, originalNote)),
          screen.arms.map((arm) => h("tr", null,
            h("td", null, name(arm)),
            h("td", null, arm.target.total ? rateBar(arm.target, "change") : h("span", { class: "muted" }, "none drawn"), drawn(arm), leftOutChip(arm, arm.name), timedOut(arm)),
            h("td", { class: "num mono" }, formatP(arm.p_value)),
            h("td", { class: "num mono" }, formatP(arm.adjusted_p)),
            h("td", null, verdict(arm))))))),
      h("p", { class: "hint" }, screen.reach === "end"
        ? "Each fork replays its run as recorded up to the step, the workspace included, then runs live to the end, where the task's check decides. Confirm an edit that beat the original with full test runs."
        : "This test changed only this step. Everything before it still comes from the original wording, so treat the result as a lead, and confirm it with full test runs."),
    ]),
    checkpointsView(screen, names),
    !behaviours ? null : screen.reach === "end"
      ? h("details", { class: "fold" },
          h("summary", null, `What each fork's model did at the step: ${plural(screen.behaviours.length, "behaviour")}, by wording`),
          behaviours)
      : panel("What the model did, by wording", behaviours));
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
      h("span", { class: "mono" }, arm), tally.of === 1
        ? (tally.above
            ? ": above the original in the one run forked from, so the effect may have lived in what that run did before the step"
            : ": not above the original in the one run forked from")
        : `: above the original in ${tally.above} of the ${tally.of} runs forked from`,
      tally.of > 1 && tally.above === tally.of ? ", an effect that holds whatever each run did before the step"
        : tally.of > 1 && tally.above === 1 ? ", in one alone: the effect may have lived in what that run did before the step" : "")),
  ]);
}

/** What the checks said, counted: each group's findings, the most common first, each with the
 *  failed runs (or forks) that said it. `only` keeps one condition of a task's tally. */
export function tallyView(tally, { only = null } = {}) {
  const groups = (tally?.groups || []).filter((group) => !only || group.condition?.id === only);
  if (!groups.some((group) => group.failed)) return null;
  const forks = tally.scope === "screen";
  const title = forks ? "What each wording's failed forks' checks said" : "What the failed runs' checks said";
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
      group.silent ? h("p", { class: "faint small" }, `${plural(group.silent, forks ? "failed fork" : "failed run")} whose check printed nothing`) : null,
      group.cut ? h("p", { class: "hint" }, `${plural(group.cut, forks ? "failed fork" : "failed run")} whose check printed more than the last 2,000 characters a verdict keeps: what each printed before them is not counted.`) : null)),
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
