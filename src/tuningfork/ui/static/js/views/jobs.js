// Jobs: the work a running proxy is doing, or has done — started here, or from anywhere.

import { api, enc } from "../api.js";
import { guard } from "../auth.js";
import { ago, crumbs, foldPanel, h, kv, loading, outcome, panel, plural, pretty, rateBar, replace, short, tick, when } from "../dom.js";
import {
  ACTIVE, KIND_SECTION, cancelButton, duration, kindLabel, logLine, madeLink, progressBar, startedBy, stateChip, subject, watchJob,
} from "../jobs.js";
import { costOf } from "../cost.js";
import { launcher } from "../launcher.js";
import { live } from "../live.js";
import { verdictView } from "../checktry.js";
import { replyCard } from "../holdtools.js";
import { found, targetLine } from "../results.js";
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
    h("thead", null, h("tr", null, selectable ? h("th", { class: "pick" }, all) : null, h("th", null, "job"), h("th", null, "what"), h("th", null, "of"),
      h("th", null, "state"), h("th", null, "result"), h("th", null, "took"), h("th", null, "started"))),
    h("tbody", null, jobs.map((job) =>
      h("tr", { class: ["clickable", selected.has(job.id) && "selected"], dataset: { job: job.id }, onclick: () => (location.hash = `#/jobs/${job.id}`) },
        selectable ? h("td", { class: "pick", onclick: (event) => event.stopPropagation() }, box(job)) : null,
        h("td", null, h("a", { class: "mono", href: `#/jobs/${job.id}` }, short(job.id))),
        h("td", null, KIND_SECTION[job.kind] ? h("span", { class: "muted" }, `${KIND_SECTION[job.kind]} · `) : null, kindLabel(job)),
        h("td", { onclick: (event) => event.stopPropagation() }, subject(job)),
        h("td", null, stateChip(job)),
        h("td", null, resultCell(job)),
        h("td", { class: "faint" }, duration(job)),
        h("td", { class: "faint", title: when(job.created_at) }, ago(job.created_at), byLine(job))))))));
}

/** Running, how far it has got; stopped, how far it got; done, what it made, leading there — or,
 *  runs that no one page holds, how many: its own page lists them. */
function resultCell(job) {
  if (ACTIVE.has(job.status)) return progressBar(job);
  if (job.status !== "done") return job.progress?.total ? progressBar(job) : null;
  const made = madeLink(job);
  if (made) return h("a", { href: made.href, onclick: (event) => event.stopPropagation() }, `→ ${made.text}`);
  const runs = job.links?.runs || [];
  return runs.length ? h("span", { class: "muted" }, plural(runs.length, "run")) : null;
}

/** Who started a job, and from where, under when. */
function byLine(job) {
  const by = startedBy(job);
  return by ? h("div", { class: "small", title: by.title }, by.text) : null;
}

export async function jobView(root, [id], scope) {
  replace(root, loading());
  const head = h("div", { class: "page-head" });
  const summary = h("div", { class: "stack slot" });
  const logEl = h("div", { class: "job-log" });
  // The log, open while the job runs or where it went wrong; folded once it is done and the page
  // leads with what it found or made.
  const progressEl = h("div", { class: "slot" });
  replace(root, crumbs([["Jobs", "#/jobs"], [short(id)]]), head, h("div", { class: "stack" }, summary, progressEl));

  let cancel = null;
  let last = null;
  let folded = null;
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
    const by = startedBy(job);
    replace(head,
      h("div", null,
        h("h1", null, kindLabel(job), h("span", { class: "h-sub mono" }, short(job.id))),
        h("p", { class: "sub" }, subject(job) || "", " ", stateChip(job),
          by ? h("span", { class: "muted small", title: by.title }, ` started by ${by.text}`) : null)),
      h("div", { class: "actions" }, cancel,
        ACTIVE.has(job.status) || job.trashed_at ? null : trashButton({ kind: "job", id: job.id, onDone: again })));
    replace(summary, heldBanner(job),
      job.trashed_at
        ? trashedBanner({ kind: "job", id: job.id, trashedAt: job.trashed_at, what: "job's record", onRestored: again,
            onPurged: () => { location.hash = "#/trash"; } })
        : null,
      ...jobSummary(job));
    // Folded only where the page leads with what the job found or made: a job done with nothing
    // else to show keeps its log in sight.
    const handedOff = job.status === "done" && showsResult(job);
    if (handedOff !== folded) {
      folded = handedOff;
      const meta = h("span", { class: "muted" }, "as it happened, newest last");
      replace(progressEl, handedOff ? foldPanel("Progress", logEl, { meta }) : panel("Progress", logEl, { meta }));
    }
    replace(logEl,
      state.gap ? h("p", { class: "hint" }, "Earlier events are no longer kept; the newest are below.") : null,
      state.events.length ? state.events.slice(-120).map(logLine) : h("p", { class: "muted" }, emptyLog(job)));
  }
}

/** Why a job's log is empty: it has not begun, or has logged nothing yet — or, ended, its log
 *  is not kept here (a proxy keeps the logs of its latest jobs, in memory: a job from before it
 *  started, one long finished, or one read from the recordings has none), or it logged nothing.
 *  Then, where the page shows one, its result or what went wrong. */
export function emptyLog(job) {
  if (job.status === "queued") return "Waiting its turn.";
  if (ACTIVE.has(job.status)) return "Nothing yet.";
  const where = job.status === "done" && showsResult(job) ? " Its result is above." : job.error ? " What went wrong is above." : "";
  return job.seq === null || job.seq === undefined
    ? `Its log is not kept here: a proxy keeps only its latest jobs' logs, and only in memory.${where}`
    : `It logged nothing along the way.${where}`;
}

// Kinds whose finished page says what they made in a panel of their own.
const OWN_PANEL = new Set(["propose", "validate", "tokens", "hold-console", "check-trial"]);

/** Whether a finished job's page shows what it found or made, above its log. */
function showsResult(job) {
  return Boolean(found(job) || madeLink(job) || OWN_PANEL.has(job.kind) || (job.links?.experiments || []).length || (job.links?.runs || []).length);
}

function heldBanner(job) {
  const launches = new Set(job.links?.launches || []);
  const held = live.holds.filter((hold) => launches.has(hold.launch_id));
  if (!held.length) return null;
  return h("div", { class: "banner banner-warn" },
    h("strong", null, held.length === 1 ? "A fork is waiting at a breakpoint. " : `${held.length} forks are waiting at breakpoints. `),
    held.map((hold) => h("a", { class: "btn btn-small", href: `#/live/${hold.id}` }, `Open it: step ${hold.step}`)));
}

/** A job's page under its head, from the job alone: how it is going or went, what it found or
 *  made, and what it was asked. */
export function jobSummary(job) {
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
  // A job that made an experiment hands off to it, done: the page of what it made shows the whole.
  const finding = job.status === "done" ? found(job) : null;
  parts.push(...(finding ? [foundPanel(job, finding)] : madeView(job)));
  const result = job.result;
  if (result && job.status === "done" && !finding) {
    // A question at a held call, and a check tried: this page is their only home once the call
    // is decided or the tester closed.
    if (job.kind === "hold-console" && result.reply) parts.push(panel("What it answered", replyCard(job)));
    if (job.kind === "check-trial" && result.found) parts.push(panel("What the check said", verdictView(result)));
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

/** What the built-in proposer made: each proposal kept, each one not and why — every one kept
 *  untested until a screen tests it. */
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
            ` — ${plural(proposal.changed, "line")} changed${proposal.duplicate ? " (proposed before: the proposing model is added to its proposers)" : ""}: `,
            h("span", { class: "muted" }, proposal.proposers[proposal.proposers.length - 1]?.rationale || ""))))
      : h("p", { class: "muted" }, "Nothing it proposed could be kept."),
    accepted.length ? h("p", { class: "hint" }, "Each one kept stays untested until a test at the refinement's step tries it.") : null,
    rejected.length
      ? h("div", null, h("div", { class: "label" }, "Not kept"),
          h("ul", { class: "small" }, rejected.map((each) =>
            h("li", null, each.index ? `proposal ${each.index}: ` : "", each.reason, each.hint ? h("span", { class: "muted" }, ` — ${each.hint}`) : null))))
      : null),
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

/** What a finished job found: what it counted, the result in a line with the rates it rests on,
 *  what was left out, the runs it made, and the way to the whole. */
function foundPanel(job, finding) {
  const made = madeLink(job);
  const bars = finding.bars.filter(([, rate]) => rate && rate.total);
  return panel("What it found", h("div", { class: "stack", style: { gap: "8px" } },
    finding.caution
      ? h("div", { class: "banner banner-warn" }, finding.text)
      : h("div", { class: ["verdict-block", finding.tone] },
          finding.target ? targetLine(finding.target[1], finding.target[2], finding.target[0]) : null,
          h("div", { class: "verdict-big" }, finding.text),
          finding.seen ? h("p", { class: "verdict-lines" }, finding.seen) : null,
          bars.length
            ? h("div", { class: "stack", style: { gap: "4px" } }, bars.map(([label, rate, tone, said]) =>
                h("span", null, h("span", { class: "muted" }, `${label}  `), said ? [outcome(said), " "] : null, rateBar(rate, tone))))
            : null),
    (finding.notes || []).map((note) => h("p", { class: "small warn-text" }, note)),
    runsLine(job),
    made
      ? h("div", { class: "made" }, h("a", { class: "btn btn-primary", href: made.href }, "See the result"),
          h("span", { class: "muted small" }, made.text))
      : null,
    kv([["cost", costOf({ job: job.id })]])));
}

/** The runs a job made — its forks, a batch's runs — each linked, the first twenty in sight and
 *  the rest folded under them: forks are in no list of runs. */
function runsLine(job) {
  const runs = job.links?.runs || [];
  if (!runs.length) return null;
  const named = (some) => some.map((run, index) => [index ? ", " : "", h("a", { class: "mono", href: `#/runs/${enc(run)}` }, short(run))]);
  return h("div", { class: "muted small" }, `${plural(runs.length, "run")}: `, named(runs.slice(0, 20)),
    runs.length > 20 ? h("details", { class: "fold" }, h("summary", null, `and ${runs.length - 20} more`), named(runs.slice(20))) : null);
}

function madeView(job) {
  const links = job.links || {};
  const experiments = links.experiments || [];
  const runs = links.runs || [];
  // Done, what it made and where that lives — unless a panel of its own below leads there.
  const done = job.status === "done" && !OWN_PANEL.has(job.kind) ? madeLink(job) : null;
  if (!experiments.length && !runs.length) {
    if (!done) return [];
    // Nothing of its own to list: a chain whose links made no run, or what a server test or a
    // check run along the runs found, which another page shows.
    return job.kind === "chain"
      ? [panel("What it made", madeRow(job, done))]
      : [panel("What it found", h("div", { class: "made" }, h("a", { class: "btn btn-primary", href: done.href }, "See the results"),
          h("span", { class: "muted small" }, job.kind === "server-check" ? "on Connections" : done.text)))];
  }
  // An A/B's two arms are one report: its first arm's page shows both.
  const report = experiments[0];
  // Runs and nothing else, done — a batch's, a chain's, a fork's: where they are, how many passed,
  // and how many the rate leaves out.
  const made = report ? null : done;
  const batch = job.kind === "batch" && job.status === "done" ? job.result || {} : {};
  const left = batch.not_counted || 0;
  return [panel("What it made", h("div", { class: "stack", style: { gap: "8px" } },
    report ? h("div", { class: "made" },
      h("a", { class: "btn btn-primary", href: `#/experiments/${report}` }, job.status === "done" ? "See the result" : "Open the experiment"),
      h("span", { class: "muted small" }, experiments.length > 1 ? `experiments ${experiments.map(short).join(", ")}` : `experiment ${short(report)}`)) : null,
    made ? madeRow(job, made) : null,
    batch.rate ? h("div", null, h("span", { class: "muted" }, "passed  "), rateBar(batch.rate, "ok")) : null,
    left ? h("p", { class: "small warn-text" }, `${plural(left, "run")} ${left === 1 ? "is" : "are"} not counted; each one's page says why.`) : null,
    // A fork's one run is what it made: named once.
    made && runs.length === 1 && made.href === `#/runs/${enc(runs[0])}` ? null : runsLine(job),
    // Counted from the records of what it made: once nothing is still being added to them.
    job.status === "running" || job.status === "queued" ? null : kv([["cost", costOf({ job: job.id })]])))];
}

/** The way to what a finished job made, and its name — a chain's page, link by link, with how
 *  its links went. */
function madeRow(job, made) {
  const chain = job.kind === "chain" ? job.result || {} : null;
  return h("div", { class: "made" },
    h("a", { class: "btn btn-primary", href: made.href }, chain ? "See the chain, link by link" : "See the result"),
    h("span", { class: "muted small" }, chain
      ? `${plural((chain.links || []).length, "link")}${chain.rate ? `, ${chain.rate.successes} of ${chain.rate.total} counted passed` : ""}`
      : made.text));
}
