// Start work from the Jobs page: each kind of job the proxy can run, opened where it
// belongs. The kinds come from the proxy (`GET /jobs/kinds`), so a kind added there shows up
// here. A kind worked at a step asks for a run and a step, then opens that step's own form:
// the form, with its plan preview, is where a job starts — this only finds the way there.

import { api, enc, query } from "./api.js";
import { h, panel, replace, short, when } from "./dom.js";
import { field } from "./forms.js";
import { KIND_LABEL } from "./jobs.js";

// How a step's page opens each kind's form: the "to the end" kinds are the same forms,
// reaching further. A kind not listed opens as `?start=<kind>`.
const STEP_FORMS = {
  resample: "start=resample",
  ab: "start=ab",
  "resample-to-end": "start=resample&reach=end",
  "ab-to-end": "start=ab&reach=end",
  fork: "start=fork",
  // Scoring has no form: the step's Tokens view offers it, beside what it shows.
  tokens: "view=tokens",
  "forking-tokens": "start=forking-tokens",
};
// These relaunch the harness the run was launched with, so they need a run tuningfork launched.
const NEEDS_LAUNCH = new Set(["resample-to-end", "ab-to-end", "fork"]);

const required = (kind) => kind.params?.required || [];

/** The launcher, built from the proxy's kinds: a panel to put above the list of jobs. */
export async function launcher() {
  const kinds = await api.get("/jobs/kinds");
  const batch = kinds.find((kind) => kind.name === "batch");
  const atStep = kinds.filter((kind) => required(kind).includes("run") && required(kind).includes("step"));
  const runs = atStep.length ? await api.get(`/runs${query({ limit: 50 })}`) : [];

  const kindSelect = h("select", { class: "field", name: "launch_kind" },
    atStep.map((kind) => h("option", { value: kind.name }, KIND_LABEL[kind.name] || kind.name)));
  const runSelect = h("select", { class: "field", name: "launch_run" },
    h("option", { value: "" }, runs.length ? "Choose a run…" : "No runs recorded yet"),
    runs.map((run) => h("option", { value: run.id },
      `${short(run.id)} · ${run.label || run.task_id || "a run"} · ${run.harness || "?"} · ${when(run.started_at)}`)));
  const stepSelect = h("select", { class: "field", name: "launch_step", disabled: true });
  const about = h("p", { class: "form-help" });
  const open = h("button", { class: "btn btn-primary", type: "button", "data-control": true, disabled: true }, "Open it");
  let steps = [];

  const chosenRun = () => runs.find((run) => run.id === runSelect.value);

  function refresh() {
    const kind = atStep.find((found) => found.name === kindSelect.value);
    const run = chosenRun();
    const unlaunched = run && NEEDS_LAUNCH.has(kind?.name) && !run.launch_id;
    replace(about, kind?.summary || "",
      unlaunched ? h("span", { class: "path-line path-bad" }, "This run was not started by tuningfork, so it cannot be forked: choose one from a batch.") : null);
    open.disabled = !kind || !run || stepSelect.disabled || stepSelect.value === "" || unlaunched;
  }

  runSelect.addEventListener("change", async () => {
    stepSelect.disabled = true;
    replace(stepSelect);
    refresh();
    if (!runSelect.value) return;
    const wanted = runSelect.value;
    const detail = await api.get(`/runs/${enc(wanted)}`);
    // A slower answer for a run chosen before is not this one's steps (found in review).
    if (runSelect.value !== wanted) return;
    // A fork's replayed steps were answered from the recording: nothing was sampled there.
    steps = detail.steps.filter((step) => !step.served);
    replace(stepSelect, steps.map((step) => h("option", { value: String(step.idx) }, `step ${step.idx} · ${step.outcome?.label ?? ""}`)));
    stepSelect.disabled = !steps.length;
    refresh();
  });
  kindSelect.addEventListener("change", refresh);
  stepSelect.addEventListener("change", refresh);
  open.addEventListener("click", () => {
    const form = STEP_FORMS[kindSelect.value] || `start=${enc(kindSelect.value)}`;
    location.hash = `#/runs/${enc(runSelect.value)}/${stepSelect.value}?${form}`;
  });
  refresh();

  return panel("Start work", h("div", { class: "stack" },
    batch ? h("div", { class: "launch-row" },
      h("div", null, h("strong", null, "Run a task"), h("p", { class: "form-help" }, batch.summary)),
      h("a", { class: "btn", href: "#/batch", "data-control": true }, "Open the form")) : null,
    atStep.length ? h("div", { class: "stack", style: { gap: "8px" } },
      h("strong", null, "At a step of a recorded run"),
      h("div", { class: "form-grid" },
        field("What", kindSelect), field("Run", runSelect, { wide: true }), field("Step", stepSelect)),
      about,
      h("div", { class: "form-actions" }, open)) : null,
    h("p", { class: "form-help" },
      "The server test is on ", h("a", { href: "#/connections" }, "Connections"),
      "; sampling a call held at a breakpoint, or asking the model about it, is on ",
      h("a", { href: "#/live" }, "Live"), ", beside the call; proposing edits of a skill and testing them, on ",
      h("a", { href: "#/refine" }, "Refine"), ".")), { className: "launcher" });
}
