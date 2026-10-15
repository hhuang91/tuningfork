// Resamples and wording A/B tests: behaviour distributions at a checkpoint.

import { api, query } from "../api.js";
import {
  ago, chip, crumbs, h, kv, loading, panel, plural, pretty, rateBar, replace, short, when,
} from "../dom.js";
import { live } from "../live.js";
import { comparisonView, distributionView, forkingView, formatP, screenView, tallyView } from "../results.js";
import { trashButton, trashedBanner } from "../trash.js";

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
        ? `${group[0].samples} per arm · ${group.length} arms`
        : group.map((arm) => `${arm.arm} ${arm.samples}`).join(" · ");
    }
    if (experiment.kind !== "ab") return String(experiment.samples);
    const group = [...(arms.get(experiment.group_id) || [experiment])].sort((a, b) => a.arm.localeCompare(b.arm));
    // Equal arms read as one number; an A/B still running, or stopped part way, shows each.
    return group.length === 2 && group[0].samples === group[1].samples
      ? `${group[0].samples} per arm`
      : group.map((arm) => `${arm.arm} ${arm.samples}`).join(" · ");
  };

  replace(
    root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Experiments"),
      h("p", { class: "sub" }, "Resamples, wording A/B tests and screens of proposed edits, from a recorded step. Start one from a step's page, or from a terminal: `tuningfork resample`, `tuningfork ab`, `tuningfork refine screen`."))),
    rows.length
      ? h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "experiment"), h("th", null, "kind"), h("th", null, "checkpoint"),
            h("th", { class: "num" }, "samples"), h("th", null, "declared target"), h("th", null, "when"))),
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

function kindChip(experiment) {
  const toEnd = experiment.reach === "end";
  if (experiment.kind === "ab") return chip(toEnd ? "A/B to the end" : "A/B", "accent");
  if (experiment.kind === "forking-tokens") return chip("forking tokens", "skill");
  if (experiment.kind === "screen") return chip("screen", "accent");
  return chip(toEnd ? "resample to the end" : "resample", "muted");
}

const TITLES = { ab: "Wording A/B", resample: "Resample", "forking-tokens": "Forking tokens", screen: "Screen of proposed edits" };

export async function experimentView(root, [id], scope) {
  replace(root, loading());
  let experiment = await api.get(`/experiments/${encodeURIComponent(id)}`);
  // A probe's answer, as tokens: read once, however often a running probe is drawn again.
  let base;
  // Forks run to the end: what their checks said, counted by arm, read again as forks finish.
  let tally = null;
  const loadTally = () => {
    if (experiment.reach !== "end") return;
    api.get(`/tally${query({ screen: experiment.id })}`)
      .then((data) => { tally = data; })
      .catch(() => { tally = null; })
      .then(() => { if (scope.active) draw(); });
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
    const title = (TITLES[experiment.kind] || experiment.kind) + (toEnd ? ", to the end" : "");
    const how = toEnd
      ? ": each sample a fork of the run, answered from the recording up to it and live from there to the end, where the task's check decided."
      : experiment.kind === "forking-tokens"
        ? ": at each probed token, the run's own and each alternative forced, and the model left to go on — with everything before it held fixed."
        : experiment.kind === "screen"
          ? ": each proposal's wording swapped in, and sampled against a fresh arm of the original — with everything before it held fixed."
          : ", with everything before it held fixed.";
    const refinement = experiment.settings?.refinement;

    replace(
      root,
      crumbs([["Experiments", "#/experiments"], [short(experiment.id)]]),
      h("div", { class: "page-head" }, h("div", null,
        h("h1", null, title, h("span", { class: "h-sub mono" }, short(experiment.id))),
        h("p", { class: "sub" }, "From ", checkpoint
          ? h("a", { class: "mono", href: checkpoint }, `run ${short(experiment.run_id)}, step ${experiment.step_idx}`)
          : h("span", { class: "mono" }, `a call held at a breakpoint (${short(experiment.hold_id)}), not recorded yet`),
          how,
          refinement ? [" Part of ", h("a", { class: "mono", href: `#/refine/${refinement}` }, `refinement ${short(refinement)}`), "."] : null)),
        h("div", { class: "actions" }, nextSteps(experiment),
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
        (experiment.partial || []).map((arm) =>
          h("div", { class: "banner banner-warn" }, h("strong", null, "Stopped early. "),
            `The ${arm.arm} arm has ${arm.drawn} of its ${arm.meant} samples; its rates are over those.`)),
        experiment.interrupted
          ? h("div", { class: "banner banner-warn" }, h("strong", null, "This A/B was interrupted. "),
            `Only its ${experiment.interrupted} arm ran, so there is nothing to compare; its samples are below.`)
          : null,
        experiment.comparison ? comparisonView(experiment) : null,
        experiment.forking ? forkingView(experiment.forking, { experiment, view: base }) : null,
        experiment.screen ? screenView(experiment.screen, { arms: experiment.arms }) : null,
        experiment.to_end ? toEndView(experiment) : null,
        tally ? tallyView(tally) : null,
        experiment.distribution ? distributionView(experiment.distribution) : null,
        experiment.settings?.logprobs && checkpoint
          ? h("p", { class: "hint" }, "Each sample kept its tokens: ",
              h("a", { href: `${checkpoint}?view=tokens` }, "the step's Tokens view"), " lists them, each with its heatmap.")
          : null,
        notesPanel(experiment)),
    );
  }
}

function originBanner(experiment) {
  const origin = experiment.origin;
  if (!origin?.hold) return null;
  if (origin.matches_step === false) {
    return h("div", { class: "banner banner-warn" }, h("strong", null, "Sampled at a held call, then edited. "),
      "These samples are of the request as it was held; the model then saw an edited one. They are not this step's behaviour, and no A/B target is taken from them.");
  }
  return h("div", { class: "banner" }, h("strong", null, "Sampled at a held call. "),
    experiment.run_id ? "Filed under the step the call became." : "Filed under its step once the call is recorded.");
}

function runningBanner(experiment) {
  if (experiment.status !== "running") return null;
  const arms = (experiment.running || []).map((arm) => `${arm.arm === "original" && experiment.kind !== "ab" ? "" : `the ${arm.arm} arm: `}${arm.drawn} of ${arm.meant}`);
  return h("div", { class: "banner" }, h("strong", null, "Still running. "),
    arms.length ? `${arms.join("; ")} samples so far — the rates below are over those. ` : "The rates below are over the samples so far. ",
    experiment.job_id ? h("a", { href: `#/jobs/${experiment.job_id}` }, "Its job") : null,
    experiment.job_id ? " shows its progress." : "");
}

/** What to do next from here, with a running proxy: the loop is resample, A/B, confirm. */
function nextSteps(experiment) {
  if (live.mode !== "live" || experiment.reach === "end" || !experiment.run_id) return null;
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
  return h("div", { class: "actions" }, links);
}

function armView(arm, label, tone) {
  return h("div", { class: "stack", style: { gap: "4px" } },
    h("span", null, h("span", { class: "muted" }, `${label}  `), arm.rate ? rateBar(arm.rate, tone) : h("span", { class: "muted" }, "no fork counts")),
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
      panel("The task's check, end to end", [armView(result, "passed", undefined)],
        { meta: h("span", { class: "muted" }, `${plural(result.counted, "fork")} counted · Wilson 95%`) }),
      distributionView(result.actions, "What the model did at the step"),
      forksPanel([[null, result]]));
  }
  const significant = result.significant;
  return h("div", { class: "stack" },
    panel("Original wording against the variant, to the end", [
      h("div", { class: ["verdict-block", significant ? "significant" : "not-significant"] },
        h("div", null,
          h("div", { class: "verdict-big" }, significant ? "The variant changes how often the task passes" : "No significant difference"),
          h("div", { class: "muted small" }, "Target: the task's check passes — declared by the task before any fork ran")),
        h("div", { class: "stack", style: { gap: "4px" } },
          armView(result.original, "original", undefined), armView(result.variant, "variant", "response")),
        h("div", { class: "mono small" }, result.p_value === null ? "no test: an arm has no fork that counts" : `Fisher exact p = ${formatP(result.p_value)} (α = ${result.alpha})`)),
      h("p", { class: "hint" }, "Steps before this one were answered from the recording in every fork, so both arms share the history written under the original wording; from here each ran live with its own wording."),
    ]),
    distributionView(result.original.actions, "Original: what the model did at the step"),
    distributionView(result.variant.actions, "Variant: what the model did at the step"),
    forksPanel([["original", result.original], ["variant", result.variant]]));
}

/** Every fork, linked: forks are runs of their own, but not in the list of runs. */
function forksPanel(arms) {
  const rows = [];
  for (const [label, arm] of arms) {
    for (const fork of arm.forks || []) {
      const verdict = fork.exclusion
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
        h("td", null, verdict)));
    }
  }
  if (!rows.length) return null;
  return panel("Forks", h("details", { class: "fold" },
    h("summary", null, `${plural(rows.length, "fork")}, each a run of its own: its steps say which the recording answered`),
    h("div", { class: "table-wrap" }, h("table", { class: "table" }, h("tbody", null, rows)))));
}

function notesPanel(experiment) {
  const settings = experiment.settings || {};
  const scalars = Object.entries(settings).filter(([, value]) => value === null || typeof value !== "object");
  return panel("How it was sampled", [
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
    experiment.notes?.length ? h("div", null, h("div", { class: "label" }, "Notes"), h("ul", { class: "small" }, experiment.notes.map((note) => h("li", null, note)))) : null,
    Object.keys(experiment.changes || {}).length
      ? h("details", { class: "fold" }, h("summary", null, "Parameters changed from the recording"), h("pre", { class: "code" }, pretty(experiment.changes)))
      : null,
    h("details", { class: "fold" }, h("summary", null, "All settings"), h("pre", { class: "code" }, pretty(settings))),
  ]);
}
