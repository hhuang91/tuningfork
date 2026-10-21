// Jobs: work the proxy is doing, started from this page or from anywhere else.
//
// A page watching a job gets its progress from the event stream, event by event, each with
// a sequence number. A gap in the numbers — the page fell behind, or reconnected — is
// filled by asking the proxy for everything after the last number seen.

import { api, enc } from "./api.js";
import { guard } from "./auth.js";
import { chip, elapsed, h, outcome, plural, short, since } from "./dom.js";

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

// The sidebar's section of the work each kind does: the loop's stage it serves (Record, Measure,
// Improve), or the page it serves (Connections, Scenarios, Live). A job reads as the work it is.
export const KIND_SECTION = {
  resample: "Measure", ab: "Measure", "resample-to-end": "Measure", "ab-to-end": "Measure",
  tokens: "Measure", "forking-tokens": "Measure", "hold-sample": "Measure",
  propose: "Improve", screen: "Improve", "screen-to-end": "Improve", validate: "Improve",
  batch: "Record", chain: "Record", fork: "Record", "check-along": "Record",
  "server-check": "Connections", "check-trial": "Scenarios", "hold-console": "Live",
};

const STATE_TONE = { queued: "muted", running: "accent", done: "muted", failed: "warn", cancelled: "muted", interrupted: "warn" };
export const ACTIVE = new Set(["queued", "running"]);

export const kindLabel = (job) => KIND_LABEL[job.kind] || job.kind;

// The pages a person starts work from, as a job records them (`api.startedFrom`), in words: a
// page named by what it does is quoted, so it does not read as an instruction.
const FROM = {
  step: "a step's page", run: "a run's page", refinement: "a refinement's page", scenario: "a scenario's page",
  experiment: "an experiment's page", skill: "a skill's page", live: "Live", connections: "Connections",
  jobs: "Jobs", batch: "“Run a task”", scenarios: "Scenarios", refine: "“Refine”", runs: "Runs",
  experiments: "Experiments", skills: "Skills", import: "“Import a chat”", guide: "the Guide", "add-skill": "“Add a skill”",
  "make-scenario": "“Make a scenario”", divergence: "“Where the runs diverge”",
};

/** Who started a job, and from where, as its front end said: a person on one of the web UI's
 *  pages, or an agent over MCP. Null where nothing was said, as for a job started before the
 *  web UI said. */
export function startedBy(job) {
  const by = job.started_by;
  if (!by?.via) return null;
  if (by.via === "mcp") {
    return { text: `an agent over MCP${by.client ? ` · ${by.client}` : ""}`, title: by.version ? `${by.client || "MCP"} ${by.version}` : undefined };
  }
  if (by.via === "web") return { text: `you, from ${FROM[by.page] || by.page || "the web UI"}` };
  return { text: [by.via, by.client].filter(Boolean).join(" · ") };
}

/** What a finished job made, and where it lives: an experiment, a test at a step, a
 *  refinement's proposals or full test runs, a chain, runs, a step's tokens, the server's
 *  results — for its row, its page, and the note when it ends. Null where the job's own page
 *  is its result's only home, or it made nothing. */
export function madeLink(job) {
  const result = job.result || {};
  const experiments = job.links?.experiments || [];
  const runs = job.links?.runs || [];
  switch (job.kind) {
    case "server-check":
      return { href: "#/connections", text: "results on Connections" };
    case "propose":
      // Nothing kept: the reasons why are on the job's page alone.
      return result.refinement && (result.accepted || []).length
        ? { href: `#/refine/${enc(result.refinement)}`, text: `${plural(result.accepted.length, "proposal")} in refinement ${short(result.refinement)}` }
        : null;
    case "validate":
      return result.refinement ? { href: `#/refine/${enc(result.refinement)}`, text: `full test runs ${short(result.validation || "")}` } : null;
    case "chain":
      return result.id ? { href: `#/chains/${enc(result.id)}`, text: `chain ${short(result.id)}` } : null;
    case "tokens":
      return result.run_id ? { href: `#/runs/${enc(result.run_id)}/${result.step}?view=tokens`, text: `tokens at step ${result.step}` } : null;
    case "batch":
      // Its runs are among the task's, where they split and what their checks said.
      return result.task_id && runs.length ? { href: `#/divergence/${enc(result.task_id)}`, text: `${result.task_id}'s runs` } : null;
    case "check-along":
      // What the check found along each run, on the task's runs, each finding with the call that made it.
      return result.task_id ? { href: `#/divergence/${enc(result.task_id)}`, text: `the check along ${result.task_id}'s runs` } : null;
    case "screen":
    case "screen-to-end":
      // One experiment a wording, the first the test's own report.
      return experiments.length ? { href: `#/experiments/${experiments[0]}`, text: `test ${short(experiments[0])}` } : null;
    default:
      break;
  }
  // An A/B's two wordings are one report: the first's page shows both.
  if (experiments.length) return { href: `#/experiments/${experiments[0]}`, text: `experiment ${short(experiments[0])}` };
  if (runs.length === 1) return { href: `#/runs/${enc(runs[0])}`, text: `run ${short(runs[0])}` };
  return null;
}

/** Whether the page open at `hash` is the one `href` leads to: the same address, or it with
 *  more after it — a query, a condition (`@`), a deeper path. Task x/1's runs are not x/10's. */
export function onPage(href, hash = location.hash) {
  return hash === href || ["?", "&", "@", "/"].some((mark) => hash.startsWith(href + mark));
}

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

/** 1st, 2nd, 3rd, 4th … 11th, 12th, 13th … 21st: a place in line. */
function ordinal(n) {
  const tens = n % 100;
  return `${n}${tens >= 11 && tens <= 13 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"}`;
}

/** The interactive lane, in the pages' words: short work a person waits on never waits behind a
 *  background job. */
export const QUICK_LINE = "Quick work a person waits on — a server test, a check tried, tools at a held call — has a line of its own.";

export function stateText(job) {
  const progress = job.progress || {};
  if (job.status === "queued") {
    // Its place in its lane's line (`queue_position`, from 0): the next to start is first.
    const place = job.queue_position;
    const line = job.lane === "interactive" ? "in the quick line" : "in line";
    return place !== null && place !== undefined ? `queued, ${place === 0 ? "next" : ordinal(place + 1)} ${line}` : "queued";
  }
  if (job.status === "running") {
    if (progress.waiting) return `waiting: ${progress.waiting === "paused" ? "the proxy is paused" : progress.waiting}`;
    return progress.total ? `running, ${progress.done || 0} of ${progress.total}` : "running";
  }
  return job.status;
}

export const stateChip = (job) => chip(stateText(job), STATE_TONE[job.status] || "muted",
  job.status === "queued" && job.lane === "interactive" ? QUICK_LINE : undefined);

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
      // The check run along a run (`check-along`): which of how many, no verdict of its own.
      if (data.done !== undefined) {
        return h("div", { class: "log-line" }, at, h("span", { class: "muted" }, `run ${data.done} of ${data.total}, checked along`),
          data.run ? h("a", { class: "mono", href: `#/runs/${data.run}` }, short(data.run)) : null);
      }
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
