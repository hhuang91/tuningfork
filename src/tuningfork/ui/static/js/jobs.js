// Jobs: work the proxy is doing, started from this page or from anywhere else.
//
// A page watching a job gets its progress from the event stream, event by event, each with
// a sequence number. A gap in the numbers — the page fell behind, or reconnected — is
// filled by asking the proxy for everything after the last number seen.

import { api, enc } from "./api.js";
import { guard } from "./auth.js";
import { chip, elapsed, h, outcome, short, since } from "./dom.js";

export const KIND_LABEL = {
  resample: "Resample",
  ab: "Wording A/B",
  "resample-to-end": "Resample to the end",
  "ab-to-end": "Wording A/B to the end",
  fork: "Fork",
  batch: "Batch",
  chain: "Chain",
  "server-check": "Server test",
  "check-trial": "Check tried",
  "hold-sample": "Held call sampled",
  "hold-console": "Question at a held call",
  tokens: "Tokens scored",
  "forking-tokens": "Forking tokens",
  propose: "Edits proposed",
  screen: "Proposals tested at a step",
  "screen-to-end": "Proposals tested at a step, to the end",
  validate: "Full test runs of an edit",
  "check-along": "Check run along the runs",
};

const STATE_TONE = { queued: "muted", running: "accent", done: "muted", failed: "warn", cancelled: "muted", interrupted: "warn" };
export const ACTIVE = new Set(["queued", "running"]);

export const kindLabel = (job) => KIND_LABEL[job.kind] || job.kind;

/** What the job is about, from what it was asked: a run and a step, or a task. */
export function subject(job) {
  const params = job.params || {};
  if (params.refinement) return h("a", { class: "mono", href: `#/refine/${enc(params.refinement)}` }, `refinement ${short(params.refinement)}`);
  if (params.run) {
    // A step given by its number opens there; one named by what the run did there (`write_file`,
    // `changed:<path>`) is found on the run's page.
    const at = /^\d+$/.test(String(params.step ?? "")) ? `/${params.step}` : "";
    return h("a", { class: "mono", href: `#/runs/${enc(params.run)}${at}` }, `run ${short(params.run)}`,
      params.step !== undefined ? ` @ step ${params.step}` : "");
  }
  if (params.scenario) return h("a", { class: "mono", href: `#/scenarios/${enc(params.scenario)}` }, `scenario ${params.scenario}`);
  if (params.spec) return h("span", { class: "mono" }, params.spec.split("/").pop());
  if (params.hold) return h("span", { class: "mono" }, `held call ${short(params.hold)}`);
  return null;
}

export function stateText(job) {
  const progress = job.progress || {};
  if (job.status === "queued") {
    return job.queue_position !== null && job.queue_position !== undefined
      ? `queued, ${job.queue_position === 0 ? "next" : `#${job.queue_position + 1}`}`
      : "queued";
  }
  if (job.status === "running") {
    if (progress.waiting) return `waiting: ${progress.waiting === "paused" ? "the proxy is paused" : progress.waiting}`;
    return progress.total ? `running, ${progress.done || 0} of ${progress.total}` : "running";
  }
  return job.status;
}

export const stateChip = (job) => chip(stateText(job), STATE_TONE[job.status] || "muted");

/** Who asked for the job, when its front end said: an agent driving tuningfork over MCP is
 * named, so work nobody started from this page is never a mystery. */
export function originChip(job) {
  const by = job.started_by;
  if (!by || !by.via) return null;
  const via = by.via === "mcp" ? "MCP" : by.via;
  return chip(`from ${via}${by.client ? ` · ${by.client}` : ""}`, "accent",
    by.version ? `${by.client || via} ${by.version}` : undefined);
}

export function progressBar(job) {
  const progress = job.progress || {};
  const total = progress.total || 0;
  const done = Math.min(progress.done || 0, total || Infinity);
  const share = total ? done / total : job.status === "done" ? 1 : 0;
  return h("div", { class: ["job-progress", job.status === "running" && "running", progress.waiting && "waiting"] },
    h("div", { class: "job-progress-track" }, h("div", { class: "job-progress-fill", style: { width: `${(share * 100).toFixed(1)}%` } })),
    h("span", { class: "muted small" }, total ? `${done} of ${total}${timeLeft(job)}` : ""));
}

/** About how long a running job has left, at the pace of what it has done so far (the
 *  proxy's estimate, `remaining_s`), or nothing before it has finished a unit. */
function timeLeft(job) {
  const seconds = job.status === "running" ? job.progress?.remaining_s : null;
  if (seconds === null || seconds === undefined) return "";
  const minutes = Math.round(seconds / 60);
  return minutes < 1 ? ", under a minute left" : `, about ${minutes} min left`;
}

/** How long it has run, or ran: live while it runs or waits (a page calls `tick` each
 *  second), fixed once it has ended. */
export function duration(job) {
  if (!job.started_at) return job.status === "queued" ? since(job.created_at, "waiting ") : "—";
  if (!job.ended_at) return since(job.started_at);
  return elapsed(job.ended_at - job.started_at);
}

/** Two clicks: the first arms it, the second — within four seconds — cancels. */
export function cancelButton(job, { small = false } = {}) {
  const button = h("button", { class: ["btn btn-danger", small && "btn-small"], type: "button", "data-control": true }, "Cancel");
  let armed = null;
  const disarm = () => {
    clearTimeout(armed);
    armed = null;
    button.classList.remove("armed");
    button.textContent = "Cancel";
  };
  button.addEventListener("click", guard(async () => {
    if (!armed) {
      button.classList.add("armed");
      button.textContent = job.status === "running" ? "Click again: stop it" : "Click again: cancel it";
      armed = setTimeout(disarm, 4000);
      return;
    }
    disarm();
    button.disabled = true;
    try {
      await api.post(`/jobs/${enc(job.id)}/cancel`);
    } catch (error) {
      button.disabled = false;
      button.textContent = error.message;
    }
  }));
  return button;
}

/** One progress event, as a line of the job's log. */
export function logLine(entry) {
  const { kind, data } = entry.event;
  const at = h("span", { class: "faint mono small" }, `#${entry.seq}`);
  switch (kind) {
    case "experiment":
      return h("div", { class: "log-line" }, at, chip(data.arm && data.arm !== "original" ? `${data.arm} wording` : "experiment", "muted"),
        h("a", { class: "mono", href: `#/experiments/${data.id}` }, short(data.id)),
        h("span", { class: "muted" }, `${data.n} sample${data.n === 1 ? "" : "s"}${data.reach === "end" ? ", each run to the end" : ""}`));
    case "sample":
      return h("div", { class: "log-line" }, at,
        h("span", { class: "muted" }, `${data.arm && data.arm !== "original" ? `${data.arm} ` : ""}sample ${Number(data.index) + 1}`),
        data.error ? chip("failed", "warn", data.error) : data.outcome ? outcome(data.outcome) : null);
    case "launch":
      return h("div", { class: "log-line" }, at, chip("started", "muted"), h("span", { class: "mono small" }, data.label || short(data.launch_id)));
    case "run": {
      const verdict = data.exclusion
        ? chip("not counted", "warn", data.exclusion)
        : data.passed === true ? chip("pass", "ok") : data.passed === false ? chip("fail", "bad") : chip("no check", "muted");
      return h("div", { class: "log-line" }, at, h("span", { class: "muted" }, `run ${Number(data.index) + 1}`), verdict,
        data.run_id ? h("a", { class: "mono", href: `#/runs/${data.run_id}` }, short(data.run_id)) : null);
    }
    case "waiting":
      return h("div", { class: "log-line" }, at, chip("waiting", "warn"),
        h("span", { class: "muted" }, data.reason === "paused" ? "the proxy is paused; sampling goes on when it resumes" : data.reason));
    default:
      return h("div", { class: "log-line" }, at, chip(kind, "muted"));
  }
}

/**
 * Follow one job: `onChange(state)` whenever anything about it changes, where state is
 * { job, events, seq, gap }. Catches up by asking for what came after the last event seen:
 * at the start, after a reconnect, and whenever the sequence numbers skip.
 */
export function watchJob(id, scope, onChange) {
  const state = { job: null, events: [], seq: 0, gap: false };
  let asking = null;
  // Asked for again while an answer was on its way: that answer may be from before what asked.
  let again = false;
  // Its events are its own from the first one: they can come before the first answer does.
  const mine = (jobId) => jobId === (state.job?.id ?? id);

  async function catchUp() {
    if (asking) {
      again = true;
      return asking;
    }
    asking = (async () => {
      try {
        let tries = 0;
        do {
          again = false;
          const job = await api.get(`/jobs/${enc(id)}?after=${state.seq}`);
          const fresh = (job.events || []).filter((entry) => entry.seq > state.seq);
          // Kept only so far back: what came before the oldest kept is gone for good.
          if (job.first_seq && job.first_seq > state.seq + 1 && fresh.length) state.gap = true;
          state.events.push(...fresh);
          if (state.events.length > 300) state.events.splice(0, state.events.length - 300);
          state.seq = Math.max(state.seq, job.seq || 0, ...fresh.map((entry) => entry.seq));
          delete job.events;
          delete job.first_seq;
          // An answer asked for before the job ended never makes an ended job run again
          // (found in review: a job's page stayed on "running" until reloaded).
          if (state.job && !ACTIVE.has(state.job.status) && ACTIVE.has(job.status) && tries < 20) {
            tries += 1;
            again = true;
            await new Promise((resolve) => setTimeout(resolve, 250));
            continue;
          }
          state.job = job;
          onChange(state);
        } while (again);
      } finally {
        asking = null;
      }
    })();
    return asking;
  }

  scope.on((event) => {
    if (event.type === "snapshot") {
      catchUp(); // reconnected: whatever happened meanwhile
    } else if (event.type === "job" && mine(event.job.id)) {
      state.job = { ...state.job, ...event.job };
      onChange(state);
      if (!ACTIVE.has(event.job.status)) catchUp(); // the result, and the last events
    } else if (event.type === "job_event" && mine(event.job_id)) {
      if (event.seq === state.seq + 1) {
        state.events.push({ seq: event.seq, at: event.at, event: event.event });
        state.seq = event.seq;
        state.job = { ...state.job, progress: event.progress };
        onChange(state);
      } else if (event.seq > state.seq + 1) {
        catchUp(); // skipped some: fill the gap
      }
    }
  });
  return catchUp();
}

/** A short list of jobs, for a page that shows the work on one thing. */
export function jobList(jobs) {
  return h("div", { class: "job-list" }, jobs.map((job) =>
    h("a", { class: "job-row", href: `#/jobs/${job.id}` },
      h("span", { class: "mono" }, short(job.id)),
      h("span", null, kindLabel(job)),
      stateChip(job),
      h("span", { class: "faint small" }, duration(job)))));
}
