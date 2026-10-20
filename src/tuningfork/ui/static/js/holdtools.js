// Tools at a held call, under it on the Live page.
//
// Evidence: sample the held call N times — an experiment, filed under the step the call
// becomes once it is recorded. Hypotheses: ask the model one question there — its answer
// is its account of itself, which need not be what drives it, and is never recorded.
// The two are kept apart on the page as they are in the store.

import { api } from "./api.js";
import { chip, h, longText, panel, replace, short } from "./dom.js";
import { field, input, numberOf, planForm, textArea, textOf } from "./forms.js";
import { ACTIVE, kindLabel, progressBar, stateChip, watchJob } from "./jobs.js";
import { live } from "./live.js";
import { distributionView } from "./results.js";

export function holdTools(hold, scope) {
  return h("div", { class: "stack hold-tools" }, jobNote(hold), evidence(hold, scope), hypotheses(hold, scope));
}

/** The job a held call belongs to, when a job launched its harness. */
function jobNote(hold) {
  if (!hold.launch_id) return null;
  const job = live.jobs.find((candidate) => (candidate.links?.launches || []).includes(hold.launch_id));
  return job
    ? h("p", { class: "hint" }, "This call is part of ", h("a", { href: `#/jobs/${job.id}` }, `job ${short(job.id)}`), ` — ${kindLabel(job)}.`)
    : null;
}

function evidence(hold, scope) {
  const result = h("div");
  const { element } = planForm({
    kind: "hold-sample",
    fields: [h("div", { class: "form-grid" },
      field("Samples", input("n", { type: "number", value: 10, min: 1, max: 200 }))),
      h("details", { class: "fold more-settings" },
        h("summary", null, "More settings"),
        h("div", { class: "form-grid" }, field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }))))],
    read: (form) => ({ hold: hold.id, n: numberOf(form, "n") ?? 10, concurrency: numberOf(form, "concurrency") ?? 1 }),
    render: (plan) => h("div", { class: "stack", style: { gap: "6px" } },
      h("p", { class: "hint" }, `Sends the request held here ${plan.sampling.n} times through the proxy, ${plan.sampling.concurrency} at a time: what the model does at this call. Filed under the step it becomes; never recorded as a run.`),
      (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning))),
    startLabel: "Sample it",
    onStarted: (job) => follow(job),
  });

  function follow(job) {
    watchJob(job.id, scope, (state) => {
      const found = state.job;
      if (!found) return;
      if (ACTIVE.has(found.status)) {
        replace(result, h("div", { class: "controls" }, stateChip(found), progressBar(found)));
        return;
      }
      if (found.status !== "done") {
        replace(result, h("div", { class: "error-box" }, found.error?.message || `the sampling ${found.status}`));
        return;
      }
      const sampled = found.result;
      replace(result, h("div", { class: "stack", style: { gap: "8px" } },
        hold.phase === "response"
          ? h("p", { class: "hint" }, sampled.held_letter
              ? ["The answer held here is behaviour ", h("strong", null, sampled.held_letter), " below."]
              : "The answer held here is none of the sampled behaviours.")
          : null,
        distributionView(sampled.distribution, "What the model does at this call"),
        h("a", { class: "btn btn-small", href: `#/experiments/${sampled.experiment.id}` }, "Open the experiment")));
    });
  }
  return panel("Evidence at this call", [element, result], {
    meta: h("span", { class: "muted" }, "sampling — the evidence"),
    className: "evidence",
  });
}

function hypotheses(hold, scope) {
  const replies = h("div", { class: "stack", style: { gap: "10px" } });
  const shown = new Set();
  const { element } = planForm({
    kind: "hold-console",
    fields: [field("Ask the model", textArea("question", "", { rows: 3, placeholder: "Why did you name the keys that way?" }), { wide: true })],
    read: (form) => {
      const question = textOf(form, "question").trim();
      if (!question) throw Object.assign(new Error("Write a question for the model."), { field: "question" });
      return { hold: hold.id, question };
    },
    render: (plan) => h("p", { class: "hint" }, `Sends ${plan.sends}.`),
    startLabel: "Ask",
    onStarted: (job) => follow(job),
  });

  function follow(job, newest = true) {
    const slot = h("div", { class: "reply" }, h("div", { class: "reply-head" }, chip("asking…", "muted"), h("span", { class: "muted small" }, job.params?.question || "")));
    if (newest) replies.prepend(slot);
    else replies.append(slot);
    shown.add(job.id);
    watchJob(job.id, scope, (state) => {
      const found = state.job;
      if (found && !ACTIVE.has(found.status)) replace(slot, ...replyCard(found));
    });
  }

  // Questions already asked at this hold, from this page or another.
  api.get("/jobs").then((jobs) => {
    for (const job of jobs.filter((j) => j.kind === "hold-console" && j.params?.hold === hold.id && !shown.has(j.id))) {
      // One still being answered is followed until it is, as one asked here is (found in
      // review: it was shown as an error, and never became its reply).
      if (ACTIVE.has(job.status)) {
        follow(job, false);
        continue;
      }
      shown.add(job.id);
      replies.append(h("div", { class: "reply" }, ...replyCard(job)));
    }
  }, () => {});

  return panel("Hypotheses, not evidence", [
    h("p", { class: "hint" }, "Ask the model about what it is doing here. Its answer is its account of itself, which need not be what drives it: a hypothesis to test, by resampling or an A/B — never a result. Asked once, never recorded."),
    element, replies,
  ], { className: "hypotheses" });
}

function replyCard(job) {
  const head = h("div", { class: "reply-head" }, chip("hypothesis", "warn"), h("span", { class: "muted small" }, job.params?.question || ""));
  if (job.status !== "done") return [head, h("div", { class: "error-box" }, job.error?.message || `the question ${job.status}`)];
  const reply = job.result.reply;
  return [
    head,
    longText(reply.text || "(no text)", { limit: 1200, lines: 16 }),
    reply.reasoning ? h("details", { class: "fold" }, h("summary", null, "Its reasoning"), longText(reply.reasoning)) : null,
  ];
}
