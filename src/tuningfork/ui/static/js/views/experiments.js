// Resamples and wording A/B tests: behaviour distributions at a checkpoint.

import { api, enc, query } from "../api.js";
import {
  ago, chip, crumbs, foldPanel, h, info, kv, loading, panel, plural, pretty, rateBar, replace, short, when,
} from "../dom.js";
import { costOf } from "../cost.js";
import { live } from "../live.js";
import { settle } from "../nav.js";
import { behaviourList, comparisonView, distributionView, endHeadline, fanOf, forkingView, formatP, notCounted, screenView, tallyView, targetLine } from "../results.js";
import { trashButton, trashedBanner } from "../trash.js";
import { whereForm } from "../where.js";
import { diffView } from "./refine.js";

export async function experimentsView(root) {
  replace(root, loading());
  const experiments = await api.get("/experiments");
  // An A/B is stored as two arms; list it once, with what each arm has drawn.
  const arms = new Map();
  const rows = [];
  for (const experiment of experiments) {
    if (experiment.group_id) {
      if (arms.has(experiment.group_id)) {
        arms.get(experiment.group_id).push(experiment);
        continue;
      }
      arms.set(experiment.group_id, [experiment]);
    }
    rows.push(experiment);
  }
  const samples = (experiment) => {
    if (experiment.kind === "screen") {
      // A screen's arms: the original and each proposal, equal when it has finished.
      const group = arms.get(experiment.group_id) || [experiment];
      return new Set(group.map((arm) => arm.samples)).size === 1
        ? `${group[0].samples} for each of ${group.length} wordings`
        : group.map((arm) => `${arm.arm} ${arm.samples}`).join(" · ");
    }
    if (experiment.kind !== "ab") return String(experiment.samples);
    const group = [...(arms.get(experiment.group_id) || [experiment])].sort((a, b) => a.arm.localeCompare(b.arm));
    // Equal arms read as one number; an A/B still running, or stopped part way, shows each.
    return group.length === 2 && group[0].samples === group[1].samples
      ? `${group[0].samples} for each wording`
      : group.map((arm) => `${arm.arm} ${arm.samples}`).join(" · ");
  };

  replace(
    root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Experiments"),
      h("p", { class: "sub" }, "What was measured at a step: resamples, wording A/Bs, forking tokens and tests of proposed edits. Start one from a step of a run."))),
    rows.length
      ? h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "experiment"), h("th", null, "kind"), h("th", null, "step"),
            h("th", { class: "num" }, "samples"), h("th", null, "condition"), h("th", null, "when"))),
          h("tbody", null, rows.map((experiment) =>
            h("tr", { class: "clickable", onclick: () => (location.hash = `#/experiments/${experiment.id}`) },
              h("td", null, h("a", { class: "mono", href: `#/experiments/${experiment.id}` }, short(experiment.id))),
              h("td", null, kindChip(experiment)),
              h("td", null, experiment.run_id
                ? h("a", { class: "mono", href: `#/runs/${experiment.run_id}/${experiment.step_idx}`, onclick: (event) => event.stopPropagation() },
                    `${short(experiment.run_id)} @ step ${experiment.step_idx}`)
                // Sampled at a held call not recorded yet: filed under its step once it is.
                : h("span", { class: "muted mono" }, `held call ${short(experiment.hold_id)}`)),
              h("td", { class: "num" }, samples(experiment)),
              h("td", { class: "muted" }, experiment.target || "—"),
              h("td", { class: "faint", title: when(experiment.created_at) }, ago(experiment.created_at))))))))
      : h("div", { class: "panel empty" }, h("h2", null, "No experiments yet"),
          h("p", null, "Find a divergence step, then measure it: `tuningfork resample <run> <step> -n 20`.")),
  );
}

/** What kind of experiment it is, in the pages' words. */
export function kindName(experiment) {
  const toEnd = experiment.reach === "end";
  const kinds = { ab: toEnd ? "A/B, whole run" : "A/B", "forking-tokens": "forking tokens", screen: toEnd ? "test at a step, whole run" : "test at a step" };
  return kinds[experiment.kind] || (toEnd ? "resample, whole run" : "resample");
}

function kindChip(experiment) {
  return chip(kindName(experiment), "muted");
}

const TITLES = { ab: "Wording A/B", resample: "Resample", "forking-tokens": "Forking tokens", screen: "Test at a step" };

export async function experimentView(root, [id], scope) {
  replace(root, loading());
  let experiment = await api.get(`/experiments/${encodeURIComponent(id)}`);
  // A probe's answer, as tokens: read once, however often a running probe is drawn again.
  let base;
  // Forks run to the end: what their checks said, counted by arm, read again as forks finish.
  let tally = null;
  // The question asked of a screen's forks: made once, so a running screen drawn again as its
  // forks report keeps what is typed in it, and its answer (found in review).
  let whereHolds = null;
  const loadTally = () => {
    if (experiment.reach !== "end") return;
    api.get(`/tally${query({ screen: experiment.id })}`)
      .then((data) => { tally = data; })
      .catch(() => { tally = null; })
      .then(() => { if (scope.active) draw(); });
  };
  // Every continuation of a probe, a section of its own, by token or by behaviour: which grouping,
  // the address's and kept in it as the reader picks one. Kept across draws, so a probe still
  // drawing changes nothing under the reader: which folds are open, what each read and how far
  // its texts were scrolled, each fold by its behaviour's key (a running probe letters its
  // behaviours by how often each happened so far, so a letter can move to another). A fold asks
  // again only when its group grew, showing what it read meanwhile.
  const grouped = (by) => {
    const [path, search = ""] = location.hash.split("?");
    const params = new URLSearchParams(search);
    if (by === "behaviour") params.set("by", by);
    else params.delete("by");
    settle(params.toString() ? `${path}?${params}` : path);
  };
  const asked = new Map();
  const read = new Map();
  const load = ({ group, position, token, key, count }) => {
    const at = `${group}\u0000${count}`;
    if (!asked.has(at)) {
      const found = api.get(`/experiments/${enc(experiment.id)}/continuations${query({ position, token, behaviour: key })}`)
        .then((data) => {
          const listed = data.positions?.[0]?.tokens?.[0]?.behaviours?.[0]?.continuations || [];
          read.set(group, listed);
          return listed;
        });
      asked.set(at, found);
      found.catch(() => asked.delete(at));
    }
    return asked.get(at);
  };
  const written = {
    load, read, opened: new Set(), scrolled: new Map(),
    by: scope.query?.get("by") === "behaviour" ? "behaviour" : "token", onGroup: grouped,
  };
  draw();
  loadTally();
  if (experiment.kind === "forking-tokens" && experiment.run_id && experiment.settings?.base_stream) {
    const stream = experiment.settings.base_stream;
    api.get(`/runs/${encodeURIComponent(experiment.run_id)}/steps/${experiment.step_idx}/tokens?stream=${encodeURIComponent(stream)}`)
      .then((data) => { base = data.view || null; })
      .catch(() => { base = null; })
      .then(() => {
        if (!scope.active) return;
        draw();
        // A link to one of its positions — from a token the probe forced — lands on it.
        const at = scope.query?.get("at");
        if (at !== null && at !== undefined) setTimeout(() => document.getElementById(`fork-pos-${at}`)?.scrollIntoView({ block: "start" }), 0);
      });
  }

  // Still running: drawn again as its job reports samples, at most twice a second.
  let timer = null;
  scope.on((event) => {
    const jobId = experiment.job_id;
    const mine = (event.type === "job_event" && event.job_id === jobId) || (event.type === "job" && event.job.id === jobId);
    if (!jobId || !(mine || event.type === "snapshot") || experiment.status !== "running") return;
    clearTimeout(timer);
    timer = setTimeout(async () => {
      experiment = await api.get(`/experiments/${encodeURIComponent(id)}`);
      if (scope.active) draw();
      loadTally();
    }, event.type === "job" ? 50 : 500);
  });
  scope.onLeave(() => clearTimeout(timer));

  function draw() {
    const checkpoint = experiment.run_id ? `#/runs/${experiment.run_id}/${experiment.step_idx}` : null;
    const toEnd = experiment.reach === "end";
    const title = (TITLES[experiment.kind] || experiment.kind) + (toEnd ? ", whole run" : "");
    // How its samples were drawn, a click away beside where they were drawn from.
    const how = toEnd
      ? "Each sample is a fork of the run, answered from the recording up to the step and live from there to the end, where the task's check decided."
      : experiment.kind === "forking-tokens"
        ? "At each probed token, the original and each alternative were forced, and the model left to go on, with everything before it held fixed."
        : experiment.kind === "screen"
          ? "Each proposal's wording was swapped in, and sampled against the original wording, drawn fresh, with everything before it held fixed."
          : experiment.kind === "ab"
            ? "Each wording's samples resent the step's request, the variant's with its text swapped in, with everything before it held fixed."
            : "Each sample resent the step's request, with everything before it held fixed.";
    const refinement = experiment.settings?.refinement;

    replace(
      root,
      crumbs([["Experiments", "#/experiments"], [short(experiment.id)]]),
      h("div", { class: "page-head" }, h("div", null,
        h("h1", null, title, h("span", { class: "h-sub mono" }, short(experiment.id))),
        h("p", { class: "sub" }, "From ", checkpoint
          ? h("a", { class: "mono", href: checkpoint }, `run ${short(experiment.run_id)}, step ${experiment.step_idx}`)
          : h("span", { class: "mono" }, `a call held at a breakpoint (${short(experiment.hold_id)}), not recorded yet`),
          ".", info(how, { label: "How it was sampled" }),
          refinement ? [" Part of ", h("a", { class: "mono", href: `#/refine/${refinement}` }, `refinement ${short(refinement)}`), "."] : null)),
        h("div", { class: "actions" },
          experiment.trashed_at || experiment.status === "running" ? null
            : trashButton({ kind: "experiment", id: experiment.id, onDone: () => experimentView(root, [id], scope) }))),
      h("div", { class: "stack" },
        experiment.trashed_at
          ? trashedBanner({
              kind: "experiment", id: experiment.id, trashedAt: experiment.trashed_at, what: "experiment",
              onRestored: () => experimentView(root, [id], scope),
              onPurged: () => { location.hash = "#/trash"; },
            })
          : null,
        originBanner(experiment),
        runningBanner(experiment),
        // Its samples were sent `<redacted>` where the model saw text, as the step was recorded.
        experiment.masked ? h("div", { class: "banner banner-warn" }, experiment.masked) : null,
        (experiment.partial || []).map((arm) =>
          h("div", { class: "banner banner-warn" }, h("strong", null, "Stopped early. "),
            `The ${arm.arm} wording has ${arm.drawn} of its ${arm.meant} samples, and its rates are over those.`)),
        experiment.interrupted
          ? h("div", { class: "banner banner-warn" }, h("strong", null, "This A/B was interrupted. "),
            `Only its ${experiment.interrupted} wording ran, so there is nothing to compare. Its samples are below.`)
          : null,
        experiment.wording ? wordingPanel(experiment.wording) : null,
        experiment.comparison ? comparisonView(experiment) : null,
        experiment.forking ? forkingView(experiment.forking, { experiment, view: base, written }) : null,
        experiment.screen ? screenView(experiment.screen, { arms: experiment.arms }) : null,
        experiment.to_end ? toEndView(experiment) : null,
        tally ? tallyView(tally) : null,
        // A screen run to the end: one question asked of its forks, arm by arm — what the passing
        // forks wrote that the failing ones did not.
        experiment.screen && experiment.reach === "end"
          ? (whereHolds ??= panel("Ask every fork", whereForm(null, { screen: experiment.id })))
          : null,
        experiment.distribution ? distributionView(experiment.distribution, "Behaviours", { from: stepName(experiment) }) : null,
        // Only where they were kept: a server that sends no logprobs leaves a note instead.
        experiment.settings?.logprobs && checkpoint && experiment.tokens_kept
          ? h("p", { class: "hint" }, `${plural(experiment.tokens_kept, "sample")} kept ${experiment.tokens_kept === 1 ? "its" : "their"} tokens: `,
              h("a", { href: `${checkpoint}?view=tokens` }, "the step's Tokens view"), " lists them, each with its heatmap.")
          : null,
        nextSteps(experiment),
        notesPanel(experiment)),
    );
  }
}

/** The edit an A/B tests: its two texts as a line diff, read back from what its arms were
 *  sent (`ops.experiments.ab_wording`). */
function wordingPanel(wording) {
  const { original, variant, diff } = wording;
  return panel("The edit", [
    diff
      ? diffView(diff.hunks)
      : h("p", { class: "muted" }, `Not shown: ${wording.missing}.`),
    wording.form === "body"
      ? h("p", { class: "hint" }, "At this step the model read the skill's body, without its frontmatter: the lines are those it read.")
      : null,
    wording.source === "files"
      ? h("p", { class: "hint" }, "Read from its files, which still hold the texts it tested.")
      : null,
    variant.name
      ? h("p", { class: "hint" }, "The variant is saved as ", h("span", { class: "mono" }, variant.name), ", one of ",
          h("a", { href: `#/skills/${enc(variant.skill)}` }, `${variant.skill}'s saved variants`), ".")
      : null,
  ], {
    meta: h("span", { class: "muted" },
      h("span", { class: "diff-del-text" }, `− original ${original.label}`), " · ",
      h("span", { class: "diff-add-text" }, `+ variant ${variant.label}`),
      diff ? ` · ${plural(diff.changed, "line")} changed` : null),
  });
}

function originBanner(experiment) {
  const origin = experiment.origin;
  if (!origin?.hold) return null;
  if (origin.matches_step === false) {
    return h("div", { class: "banner banner-warn" }, h("strong", null, "Sampled at a held call, then edited. "),
      "These samples are of the request as it was held, and the model then saw an edited one. They are not this step's behaviour, and an A/B never takes its condition from them.");
  }
  return h("div", { class: "banner" }, h("strong", null, "Sampled at a held call. "),
    experiment.run_id ? "Filed under the step the call became." : "Filed under its step once the call is recorded.");
}

function runningBanner(experiment) {
  if (experiment.status !== "running") return null;
  const arms = (experiment.running || []).map((arm) => `${arm.arm === "original" && experiment.kind !== "ab" ? "" : `the ${arm.arm} wording: `}${arm.drawn} of ${arm.meant}`);
  return h("div", { class: "banner" }, h("strong", null, "Still running. "),
    arms.length ? `${arms.join("; ")} samples so far — the rates below are over those. ` : "The rates below are over the samples so far. ",
    experiment.job_id ? h("a", { href: `#/jobs/${experiment.job_id}` }, "Its job") : null,
    experiment.job_id ? " shows its progress." : "");
}

/** What to do next from here, with a running proxy — the loop is resample, A/B, confirm — in
 *  a line after the result, where the page is read to: then back to the step it came from. */
function nextSteps(experiment) {
  if (!experiment.run_id) return null;
  const step = `#/runs/${experiment.run_id}/${experiment.step_idx}`;
  const back = h("a", { class: "btn btn-quiet btn-small", href: step }, "Back to the step");
  if (live.mode !== "live" || experiment.reach === "end") return h("nav", { class: "next-line", "aria-label": "Next" }, h("strong", null, "Next"), back);
  const here = `#/runs/${experiment.run_id}/${experiment.step_idx}`;
  const refinement = experiment.settings?.refinement;
  const links = experiment.kind === "screen" && refinement
    ? [h("a", { class: "btn btn-small", href: `#/refine/${refinement}` }, "Back to the refinement")]
    : experiment.kind === "ab"
    ? [
        h("a", { class: "btn btn-small", href: `${here}?start=ab` }, "Try another wording"),
        h("a", { class: "btn btn-quiet btn-small", href: `${here}?start=refine` }, "Refine the skill here"),
      ]
    : experiment.kind === "forking-tokens"
    ? [
        h("a", { class: "btn btn-small", href: `${here}?view=tokens` }, "The step's tokens"),
        h("a", { class: "btn btn-quiet btn-small", href: `${here}?start=ab` }, "A/B a wording at this step"),
      ]
    : [
        h("a", { class: "btn btn-small", href: `${here}?start=ab` }, "A/B a wording at this step"),
        h("a", { class: "btn btn-quiet btn-small", href: `${here}?start=refine` }, "Refine the skill here"),
        h("a", { class: "btn btn-quiet btn-small", href: `${here}?start=resample` }, "Resample again"),
      ];
  return h("nav", { class: "next-line", "aria-label": "Next" }, h("strong", null, "Next"), links, back);
}

/** The step an experiment sampled, as its picture's first box says it. */
const stepName = (experiment) => (experiment.step_idx === undefined || experiment.step_idx === null ? "this step" : `step ${experiment.step_idx}`);

function armView(arm, label, tone) {
  return h("div", { class: "stack", style: { gap: "var(--space-1)" } },
    h("span", null, h("span", { class: "muted" }, `${label}  `), arm.rate ? rateBar(arm.rate, tone) : h("span", { class: "muted" }, "no fork counts")),
    arm.timeouts ? h("span", { class: "small muted" }, `${plural(arm.timeouts, "fork")} timed out, counted as ${arm.timeouts === 1 ? "a failure" : "failures"}, as the approval checks count them`) : null,
    arm.exclusions.length
      ? h("ul", { class: "small" }, arm.exclusions.map((left) =>
          h("li", null, h("span", { class: "mono" }, short(left.run_id || "")), ` not counted: ${left.why}`)))
      : null);
}

function toEndView(experiment) {
  const result = experiment.to_end;
  if (!result.original) {
    // One arm: a resample to the end, or the arm an interrupted A/B got through.
    return h("div", { class: "stack" },
      panel("Check, whole run", [armView(result, "passed", undefined)],
        { meta: h("span", { class: "muted", title: "Wilson score interval" }, `${plural(result.counted, "fork")} counted · 95% interval`) }),
      distributionView(result.actions, "Behaviours at the step", { from: stepName(experiment), unit: "fork" }),
      forksPanel([[null, result]]));
  }
  // The target is the task's check: here green and red mean what they mean for a run.
  const [headline, tone] = endHeadline(result);
  // What each wording's forks did at the step, and how those forks ended: both in one picture.
  const drawn = fanOf([
    { name: "original", moves: result.moves, arm: "original" },
    { name: "variant", tone: "change", moves: result.moves, arm: "variant" },
  ], { from: stepName(experiment), unit: "fork" });
  const left = (result.original.actions.errors || 0) + (result.variant.actions.errors || 0);
  const inFull = (label, arm) => h("details", { class: "fold in-full" },
    h("summary", null, `${label}'s behaviours (${arm.actions.clusters.length})`), behaviourList(arm.actions));
  return h("div", { class: "stack" },
    panel("Original vs variant, whole run", [
      h("div", { class: ["verdict-block", tone] },
        targetLine("the task's check passes", "set by the task, before any fork ran"),
        h("div", { class: "verdict-big" }, headline),
        h("div", { class: "stack", style: { gap: "var(--space-1)" } },
          armView(result.original, "original", undefined), armView(result.variant, "variant", "change")),
        result.p_value === null ? null : h("div", { class: "mono small", title: "Fisher's exact test" }, `p = ${formatP(result.p_value)}, α = ${result.alpha}`)),
      h("p", { class: "hint" }, "Both wordings share the run's history up to this step.",
        info("Steps before this one were answered from the recording in every fork, so both wordings share the history written under the original. From here, each ran live with its own wording.", { label: "What the forks share" })),
      drawn,
      // A fork that timed out or was left out is in no branch: said, where a wording's box holds
      // fewer forks than it ran.
      drawn && left ? h("p", { class: "hint" }, notCounted(left, "fork")) : null,
      drawn ? [inFull("Original", result.original), inFull("Variant", result.variant)] : null,
    ]),
    // A result with no picture to draw keeps each wording's behaviours in sight.
    drawn ? null : distributionView(result.original.actions, "Original's behaviours", { from: stepName(experiment), unit: "fork" }),
    drawn ? null : distributionView(result.variant.actions, "Variant's behaviours", { from: stepName(experiment), unit: "fork" }),
    forksPanel([["original", result.original], ["variant", result.variant]]));
}

/** Every fork, linked: forks are runs of their own, but not in the list of runs. */
function forksPanel(arms) {
  const rows = [];
  for (const [label, arm] of arms) {
    for (const fork of arm.forks || []) {
      const verdict = fork.timed_out
        ? [chip("fail", "bad"), h("span", { class: "muted" }, " timed out: counted as a failure")]
        : fork.exclusion
          ? h("span", { class: "muted" }, `not counted: ${fork.exclusion}`)
          : fork.passed === null || fork.passed === undefined
            ? chip("no check", "muted")
            : fork.passed ? chip("pass", "ok") : chip("fail", "bad");
      rows.push(h("tr", null,
        label ? h("td", { class: "muted" }, label) : null,
        h("td", { class: "num faint" }, `#${fork.idx + 1}`),
        h("td", null, fork.run_id
          ? h("a", { class: "mono", href: `#/runs/${fork.run_id}` }, `run ${short(fork.run_id)}`)
          : h("span", { class: "muted" }, "never reached the proxy")),
        h("td", null, fork.letter
          ? h("span", { class: "chip tone-accent", title: `At the step it did ${label ? `the ${label}'s ` : ""}behaviour ${fork.letter}` }, fork.letter)
          : null),
        h("td", null, verdict)));
    }
  }
  if (!rows.length) return null;
  return panel("Forks", h("details", { class: "fold" },
    h("summary", null, `Every fork (${rows.length})`),
    h("p", { class: "hint" }, "Each is a run of its own: its steps say which the recording answered."),
    h("div", { class: "table-wrap" }, h("table", { class: "table" }, h("tbody", null, rows)))));
}

/** How the experiment was sampled: kept for the record, folded, what it cost on its line. What
 *  differed from the recording stays in sight — each a caution about every rate above. */
function notesPanel(experiment) {
  const settings = experiment.settings || {};
  const scalars = Object.entries(settings).filter(([, value]) => value === null || typeof value !== "object");
  const notes = (experiment.notes || []).map((note) =>
    h("div", { class: "banner banner-warn" }, `${note[0].toUpperCase()}${note.slice(1)}.`));
  return [notes, foldPanel("Settings", [
    kv(scalars.map(([key, value]) => [key, h("span", { class: "mono" }, String(value))])),
    experiment.variant ? h("div", null, h("div", { class: "label" }, "Variant"),
      kv([
        ["original", h("span", { class: "mono" }, experiment.variant.original_path || "—")],
        ["variant", h("span", { class: "mono" }, experiment.variant.variant_path || "—")],
        // A fork swaps the variant into each call its model answers; a checkpoint A/B into
        // the one request it resends.
        experiment.reach === "end"
          ? ["swapped in", `from step ${experiment.step_idx} on, in every call the model answered`]
          : ["substituted", plural((experiment.variant.spans || []).length, "place")],
      ])) : null,
    Object.keys(experiment.changes || {}).length
      ? h("details", { class: "fold" }, h("summary", null, "Changed parameters"), h("pre", { class: "code" }, pretty(experiment.changes)))
      : null,
    h("details", { class: "fold" }, h("summary", null, "All settings"), h("pre", { class: "code" }, pretty(settings))),
  ], { className: "record-panel", meta: costOf({ experiment: experiment.id }) })];
}
