// Jobs: the work a running proxy is doing, or has done — started here, or from anywhere.

import { api, enc } from "../api.js";
import { guard } from "../auth.js";
import { ago, crumbs, foldPanel, h, kv, loading, panel, plural, pretty, rateBar, replace, short, tick, when } from "../dom.js";
import {
  ACTIVE, cancelButton, duration, kindLabel, logLine, originChip, progressBar, stateChip, subject, watchJob,
} from "../jobs.js";
import { costOf } from "../cost.js";
import { launcher } from "../launcher.js";
import { live } from "../live.js";
import { comparisonView, distributionView, forkingView, screenView } from "../results.js";
import { trashButton, trashedBanner } from "../trash.js";

export async function jobsView(root, _, scope) {
  replace(root, loading());
  const body = h("div");
  const start = h("div");
  // Finished jobs ticked to move to the trash together; kept while the list is redrawn.
  const selected = new Set();
  const bar = h("div", { class: "compare-bar", hidden: true });
  // The launcher is built on first use: it reads the proxy's kinds and the recent runs.
  const startButton = live.mode === "live"
    ? h("button", { class: "btn btn-primary", type: "button", "data-control": true, onclick: guard(async () => {
        if (start.firstChild) return replace(start);
        replace(start, await launcher());
      }) }, "Start…")
    : null;
  replace(root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Jobs"),
      h("p", { class: "sub" }, live.mode === "live"
        ? "Work the proxy is doing: started here or from a step, or with `tuningfork jobs` from a terminal. One background job runs at a time, unless the proxy was started to run more."
        : "Work a proxy did, as it recorded it. Starting work needs a running proxy.")),
      startButton ? h("div", { class: "actions" }, startButton) : null),
    start,
    body,
    bar);
  scope.every(() => tick(body), 1000);

  let timer = null;
  // A tick marks the rows in place: drawing the table again would take the focus from the
  // box just ticked, and leave a box someone is about to tick detached from the page.
  const toggle = (ids, on) => {
    for (const id of ids) (on ? selected.add(id) : selected.delete(id));
    for (const row of body.querySelectorAll("tr[data-job]")) {
      const ticked = selected.has(row.dataset.job);
      row.classList.toggle("selected", ticked);
      const box = row.querySelector("td.pick input");
      if (box) box.checked = ticked;
    }
    const all = body.querySelector("th.pick input");
    if (all) all.checked = [...body.querySelectorAll("td.pick input")].every((box) => box.checked);
    drawBar();
  };
  function drawBar() {
    const ids = [...selected];
    bar.hidden = ids.length === 0;
    replace(bar,
      h("span", null, ids.length === 1 ? "1 finished job selected" : `${ids.length} finished jobs selected`),
      h("button", {
        class: "btn btn-small", type: "button", "data-control": true,
        onclick: guard(async (event) => {
          event.target.disabled = true;
          const refused = [];
          for (const id of ids) {
            try {
              await api.post("/trash", { kind: "job", id });
              selected.delete(id);
            } catch (error) {
              refused.push(`${short(id)}: ${error.message}`);
            }
          }
          await load();
          if (refused.length) {
            bar.hidden = false;
            replace(bar, h("span", { class: "form-error" }, `Not moved — ${refused.join("; ")}`));
          }
        }),
      }, ids.length === 1 ? "Move to trash" : `Move ${ids.length} to trash`),
      h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => toggle(ids, false) }, "Clear"));
  }
  const load = async () => {
    const jobs = await api.get("/jobs");
    if (!scope.active) return;
    // A job that left the list — moved to the trash elsewhere — is no longer selected.
    const finished = new Set(jobs.filter((job) => !ACTIVE.has(job.status)).map((job) => job.id));
    for (const id of [...selected]) if (!finished.has(id)) selected.delete(id);
    drawBar();
    replace(body, jobs.length ? jobsTable(jobs, { selected, toggle }) : h("div", { class: "panel empty" }, h("h2", null, "No jobs yet"),
      h("p", null, live.mode === "live"
        ? "Start one with Start…, or open a run, choose a step, and resample it or A/B a wording there."
        : "Open a run, choose a step, and resample it or A/B a wording there, with a proxy running.")));
  };
  await load();
  // Job events come often while one runs; the list is drawn again at most every half second.
  scope.on((event) => {
    if (event.type !== "job" && event.type !== "job_event" && event.type !== "snapshot") return;
    clearTimeout(timer);
    timer = setTimeout(load, event.type === "job_event" ? 500 : 100);
  });
  scope.onLeave(() => clearTimeout(timer));
}

function jobsTable(jobs, { selected, toggle }) {
  // Only a finished job can go to the trash, so only finished jobs can be ticked.
  const selectable = live.mode === "live";
  const finished = jobs.filter((job) => !ACTIVE.has(job.status)).map((job) => job.id);
  const all = selectable && finished.length
    ? h("input", {
        type: "checkbox", "aria-label": "Select every finished job", "data-control": true,
        checked: finished.length > 0 && finished.every((id) => selected.has(id)),
        onchange: guard((event) => toggle(finished, event.target.checked)),
      })
    : null;
  const box = (job) => selectable && !ACTIVE.has(job.status)
    ? h("input", {
        type: "checkbox", checked: selected.has(job.id), "aria-label": `Select job ${short(job.id)}`, "data-control": true,
        onclick: (event) => event.stopPropagation(),
        onchange: guard((event) => toggle([job.id], event.target.checked)),
      })
    : null;
  return h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, selectable ? h("th", { class: "pick" }, all) : null, h("th", null, "job"), h("th", null, "kind"), h("th", null, "of"),
      h("th", null, "state"), h("th", null, "progress"), h("th", null, "took"), h("th", null, "started"))),
    h("tbody", null, jobs.map((job) =>
      h("tr", { class: ["clickable", selected.has(job.id) && "selected"], dataset: { job: job.id }, onclick: () => (location.hash = `#/jobs/${job.id}`) },
        selectable ? h("td", { class: "pick", onclick: (event) => event.stopPropagation() }, box(job)) : null,
        h("td", null, h("a", { class: "mono", href: `#/jobs/${job.id}` }, short(job.id))),
        h("td", null, kindLabel(job), originChip(job) ? " " : null, originChip(job)),
        h("td", { onclick: (event) => event.stopPropagation() }, subject(job)),
        h("td", null, stateChip(job)),
        h("td", null, ACTIVE.has(job.status) || job.progress?.total ? progressBar(job) : null),
        h("td", { class: "faint" }, duration(job)),
        h("td", { class: "faint", title: when(job.created_at) }, ago(job.created_at))))))));
}

export async function jobView(root, [id], scope) {
  replace(root, loading());
  const head = h("div", { class: "page-head" });
  const summary = h("div", { class: "stack slot" });
  const logEl = h("div", { class: "job-log" });
  replace(root, crumbs([["Jobs", "#/jobs"], [short(id)]]), head, h("div", { class: "stack" }, summary,
    panel("Progress", logEl, { meta: h("span", { class: "muted" }, "as it happened, newest last") })));

  let cancel = null;
  let last = null;
  // A fork stopped at a breakpoint waits on the Live page: say so here, with the way there.
  scope.on((event) => {
    if (last && ["hold", "resolved", "snapshot"].includes(event.type)) draw(last);
  });
  scope.every(() => tick(summary), 1000);
  await watchJob(id, scope, draw);

  function draw(state) {
    last = state;
    const job = state.job;
    if (!job) return;
    if (!cancel || !ACTIVE.has(job.status)) cancel = ACTIVE.has(job.status) && live.mode === "live" ? cancelButton(job) : null;
    const again = () => jobView(root, [id], scope);
    replace(head,
      h("div", null,
        h("h1", null, kindLabel(job), h("span", { class: "h-sub mono" }, short(job.id))),
        h("p", { class: "sub" }, subject(job) || "", " ", stateChip(job), originChip(job) ? " " : null, originChip(job))),
      h("div", { class: "actions" }, cancel,
        ACTIVE.has(job.status) || job.trashed_at ? null : trashButton({ kind: "job", id: job.id, onDone: again })));
    replace(summary, heldBanner(job),
      job.trashed_at
        ? trashedBanner({ kind: "job", id: job.id, trashedAt: job.trashed_at, what: "job's record", onRestored: again,
            onPurged: () => { location.hash = "#/trash"; } })
        : null,
      ...jobSummary(job));
    replace(logEl,
      state.gap ? h("p", { class: "hint" }, "Earlier events are no longer kept; the newest are below.") : null,
      state.events.length ? state.events.slice(-120).map(logLine) : h("p", { class: "muted" }, job.status === "queued" ? "Waiting its turn." : "Nothing yet."));
  }
}

function heldBanner(job) {
  const launches = new Set(job.links?.launches || []);
  const held = live.holds.filter((hold) => launches.has(hold.launch_id));
  if (!held.length) return null;
  return h("div", { class: "banner banner-warn" },
    h("strong", null, held.length === 1 ? "A fork is waiting at a breakpoint. " : `${held.length} forks are waiting at breakpoints. `),
    held.map((hold) => h("a", { class: "btn btn-small", href: `#/live/${hold.id}` }, `Open it: step ${hold.step}`)));
}

function jobSummary(job) {
  const parts = [];
  if (ACTIVE.has(job.status)) {
    parts.push(panel("Running", [progressBar(job), job.progress?.waiting === "paused"
      ? h("p", { class: "hint" }, "The proxy is paused: sampling goes on when it resumes.")
      : null], { meta: h("span", { class: "muted" }, duration(job)) }));
  }
  if (job.error) {
    parts.push(h("div", { class: "error-box" }, h("strong", null, job.error.message),
      job.error.hint ? h("p", null, job.error.hint) : null,
      job.error.detail ? h("pre", { class: "code small" }, job.error.detail) : null));
  }
  if (job.status === "cancelled" || job.status === "interrupted") {
    parts.push(h("div", { class: "banner banner-warn" }, h("strong", null, job.status === "cancelled" ? "Cancelled. " : "Interrupted: the proxy stopped. "),
      "What finished before that is stored and counted; anything it cut off is not."));
  }
  parts.push(...madeView(job));
  const result = job.result;
  if (result && job.status === "done") {
    if (result.distribution) parts.push(distributionView(result.distribution));
    if (result.comparison) parts.push(comparisonView(result));
    if (result.forking) parts.push(forkingView(result.forking));
    if ((job.kind === "screen" || job.kind === "screen-to-end") && result.screen) {
      const arms = Object.fromEntries(Object.entries(result.experiments || {}).map(([name, id]) => [name, { id }]));
      parts.push(screenView(result.screen, { arms }));
    }
    if (job.kind === "chain" && result.id) {
      parts.push(panel("What it made", h("div", { class: "made" },
        h("a", { class: "btn btn-primary", href: `#/chains/${enc(result.id)}` }, "See the chain, link by link"),
        h("span", { class: "muted small" }, `${plural((result.links || []).length, "link")}`
          + (result.rate ? `, ${result.rate.successes} of ${result.rate.total} counted passed` : "")))));
    }
    if (job.kind === "propose") parts.push(proposedView(result));
    if (job.kind === "validate") parts.push(validatedView(result));
    if (job.kind === "tokens" && result.run_id) {
      const stream = result.view?.stream;
      parts.push(panel("What it made", h("div", { class: "made" },
        h("a", { class: "btn btn-primary", href: `#/runs/${enc(result.run_id)}/${result.step}?view=tokens` }, "See its tokens"),
        h("span", { class: "muted small" }, stream ? `${plural(stream.tokens, "token")} scored${stream.exact === false ? ", rebuilt approximately" : ""}` : ""))));
    }
  }
  // What it was asked is kept for the record: folded, the page leading with what it did.
  parts.push(foldPanel("What it was asked", [
    kv(Object.entries(job.params || {})
      .filter(([key, value]) => value !== null && value !== "" && key !== "variant_text" && !(Array.isArray(value) && !value.length))
      .map(([key, value]) => [key, h("span", { class: "mono" }, typeof value === "object" ? JSON.stringify(value) : String(value))])),
    job.params?.variant_text ? h("details", { class: "fold" }, h("summary", null, "The variant's text"), h("pre", { class: "code" }, job.params.variant_text)) : null,
    job.plan ? h("details", { class: "fold" }, h("summary", null, "The plan it ran"), h("pre", { class: "code" }, pretty(job.plan))) : null,
  ], { className: "record-panel", meta: h("span", { class: "muted" }, `queued ${when(job.created_at)}`) }));
  return parts;
}

/** What the built-in proposer made: each proposal kept, each one not and why — hypotheses,
 *  all of them, until a screen tests them. */
function proposedView(result) {
  const accepted = result.accepted || [];
  const rejected = result.rejected || [];
  return panel("What it proposed", h("div", { class: "stack", style: { gap: "8px" } },
    h("div", { class: "made" },
      h("a", { class: "btn btn-primary", href: `#/refine/${enc(result.refinement)}` }, "See them in the refinement"),
      h("span", { class: "muted small" }, `${plural(accepted.length, "proposal")} kept · ${result.model || "the server's default model"} at ${result.host}, ${Math.round(result.seconds)} s`)),
    accepted.length
      ? h("ul", { class: "small" }, accepted.map((proposal) =>
          h("li", null, h("span", { class: "mono" }, short(proposal.id)),
            ` — ${plural(proposal.changed, "line")} changed${proposal.duplicate ? " (proposed before: whoever asked now is added to it)" : ""}: `,
            h("span", { class: "muted" }, proposal.proposers[proposal.proposers.length - 1]?.rationale || ""))))
      : h("p", { class: "muted" }, "Nothing it proposed could be kept."),
    rejected.length
      ? h("div", null, h("div", { class: "label" }, "Not kept"),
          h("ul", { class: "small" }, rejected.map((each) =>
            h("li", null, each.index ? `proposal ${each.index}: ` : "", each.reason, each.hint ? h("span", { class: "muted" }, ` — ${each.hint}`) : null))))
      : null,
    h("p", { class: "hint" }, result.hypothesis ? `${result.hypothesis[0].toUpperCase()}${result.hypothesis.slice(1)}.` : "")),
    { meta: h("span", { class: "muted" }, `kept in ${result.record}`) });
}

/** What a validation found, as anyone may see it: the forward and training parts, and only
 *  that each held-out task ran — its verdicts, and the gate, are on the refinement's page. */
function validatedView(result) {
  const arms = (part) => part
    ? h("span", null, "with the edit ", part.edit.rate ? rateBar(part.edit.rate) : "nothing counted",
        h("span", { class: "muted" }, " · the original "), part.original.rate ? rateBar(part.original.rate) : "nothing counted")
    : h("span", { class: "muted" }, "not run");
  return panel("What it found", h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["forks", arms(result.forward)],
      ["the task", arms(result.training)],
      ["held out", result.held_out.length ? `${result.held_out.join(", ")} — their verdicts are on the refinement's page` : "none"],
    ]),
    h("div", { class: "made" },
      h("a", { class: "btn btn-primary", href: `#/refine/${enc(result.refinement)}` }, "The approval checks, in the refinement"))),
    { meta: h("span", { class: "muted" }, `full test runs ${short(result.validation)}`) });
}

function madeView(job) {
  const links = job.links || {};
  const experiments = links.experiments || [];
  const runs = links.runs || [];
  if (!experiments.length && !runs.length) return [];
  // An A/B's two arms are one report: its first arm's page shows both.
  const report = experiments[0];
  return [panel("What it made", h("div", { class: "stack", style: { gap: "8px" } },
    report ? h("div", { class: "made" },
      h("a", { class: "btn btn-primary", href: `#/experiments/${report}` }, job.status === "done" ? "See the result" : "Open the experiment"),
      h("span", { class: "muted small" }, experiments.length > 1 ? `experiments ${experiments.map(short).join(", ")}` : `experiment ${short(report)}`)) : null,
    runs.length ? h("div", { class: "muted small" }, `${plural(runs.length, "run")}: `,
      runs.slice(0, 20).map((run, index) => [index ? ", " : "", h("a", { class: "mono", href: `#/runs/${enc(run)}` }, short(run))])) : null,
    // Counted from the records of what it made: once nothing is still being added to them.
    job.status === "running" || job.status === "queued" ? null : kv([["cost", costOf({ job: job.id })]])))];
}
