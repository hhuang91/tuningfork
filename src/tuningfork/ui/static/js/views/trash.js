// The trash: what was set aside, each with what deleting it for good would remove. Restoring
// brings it back as it went; deleting it for good takes everything tied to it.

import { api } from "../api.js";
import { guard } from "../auth.js";
import { ago, chip, h, loading, panel, plural, replace, short, when } from "../dom.js";
import { live } from "../live.js";
import { citedText, goesText, purgeButton, restoreButton } from "../trash.js";
import { kindLabel } from "../jobs.js";
import { kindName } from "./experiments.js";

export async function trashView(root) {
  replace(root, loading());
  const found = await api.get("/trash");
  const reload = () => trashView(root);
  const refinements = found.refinements || [];
  const total = found.runs.length + found.experiments.length + found.jobs.length + found.scenarios.length + refinements.length;

  const row = (kind, id, what, link, trashedAt, goes) =>
    h("tr", null,
      h("td", null, link ? h("a", { class: "mono", href: link }, short(id)) : h("span", { class: "mono" }, id)),
      h("td", null, what),
      h("td", { class: "faint", title: when(trashedAt) }, ago(trashedAt)),
      h("td", { class: "muted small" }, goes ? goesText(goes) : kind === "job" ? "the job's record" : "its folder",
        citedText(goes) ? h("div", { class: "cited" }, citedText(goes)) : null),
      h("td", { class: "trash-actions" },
        restoreButton({ kind, id, onDone: reload }),
        purgeButton({ kind, id, goes, onDone: reload })));
  const table = (rows) => h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "what"), h("th", null, "trashed"),
      h("th", null, "deleting it for good removes"), h("th", null, ""))),
    h("tbody", null, rows)));

  const sections = [
    found.runs.length ? panel(`Runs · ${found.runs.length}`, table(found.runs.map((run) =>
      row("run", run.id, [run.label || run.task_id || "a run", run.harness ? [" ", chip(run.harness, "muted")] : null],
        `#/runs/${run.id}`, run.trashed_at, run.goes)))) : null,
    found.experiments.length ? panel(`Experiments · ${found.experiments.length}`, table(found.experiments.map((experiment) =>
      row("experiment", experiment.id,
        `${experiment.arms > 1 ? "A/B" : kindName(experiment)} · run ${short(experiment.run_id || "")}, step ${experiment.step_idx ?? "?"} · ${plural(experiment.samples, "sample")}`,
        experiment.arms > 1 ? null : `#/experiments/${experiment.id}`, experiment.trashed_at, experiment.goes)))) : null,
    found.jobs.length ? panel(`Jobs · ${found.jobs.length}`, table(found.jobs.map((job) =>
      row("job", job.id, `${kindLabel(job)}, ${job.status}`, `#/jobs/${job.id}`, job.trashed_at, null)))) : null,
    found.scenarios.length ? panel(`Scenarios · ${found.scenarios.length}`, table(found.scenarios.map((scenario) =>
      row("scenario", scenario.id, scenario.name, null, scenario.trashed_at, null)))) : null,
    refinements.length ? panel(`Refinements · ${refinements.length}`, table(refinements.map((refinement) =>
      row("refinement", refinement.id,
        [h("span", { class: "mono" }, refinement.skill), ` at run ${short(refinement.run_id)} step ${refinement.step_idx}, ${plural(refinement.proposals, "proposal")}`,
          refinement.status === "open" ? null : [" ", chip(refinement.status, refinement.status === "approved" ? "ok" : "muted")]],
        `#/refine/${refinement.id}`, refinement.trashed_at, refinement.goes)))) : null,
  ];

  replace(root,
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Trash"),
        h("p", { class: "sub" }, "Set aside, out of every list and pass rate, and back as it went when restored. Deleting from here is for good: a run goes with what is tied to it — the experiments at its steps, their samples, the forks that replayed it — and a refinement with what it made: its tests at a step, its full test runs, its proposals' saved texts. A job keeps its record of the work, a chain its record and the skills its links left, and a scenario or a skill made from a run what it took.")),
      total && live.mode === "live" ? h("div", { class: "actions" }, emptyButton(total, reload)) : null),
    total ? h("div", { class: "stack" }, sections) : h("div", { class: "panel empty" }, h("h2", null, "The trash is empty"),
      h("p", { class: "muted" }, "Runs, experiments, jobs, scenarios and refinements moved to the trash wait here until restored or deleted for good.")));
}

function emptyButton(total, reload) {
  const note = h("span", { class: "muted small" });
  const button = h("button", { class: "btn btn-danger", type: "button", "data-control": true }, "Empty the trash");
  let armed = null;
  button.addEventListener("click", guard(async () => {
    if (!armed) {
      button.classList.add("armed");
      button.textContent = "Click again: delete all for good";
      note.textContent = `${plural(total, "item")}, and everything tied to them; this cannot be undone`;
      armed = setTimeout(() => {
        armed = null;
        button.classList.remove("armed");
        button.textContent = "Empty the trash";
        note.textContent = "";
      }, 6000);
      return;
    }
    clearTimeout(armed);
    button.disabled = true;
    try {
      await api.post("/trash/empty");
      reload();
    } catch (error) {
      button.disabled = false;
      note.textContent = error.message;
    }
  }));
  return h("span", { class: "controls" }, button, note);
}
