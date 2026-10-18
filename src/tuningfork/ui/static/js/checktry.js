// Trying a check before trusting it, where it is written: against a recorded run of the task
// — the workspace it left and its final reply — or against a reply and files typed here.
// The check tried is the one on screen, saved or not. It runs on the proxy's machine in a
// throwaway folder, and nothing is recorded: a run's own verdict never changes.

import { api, query } from "./api.js";
import { chip, h, longText, replace, short, when } from "./dom.js";
import { field, input, planForm, textArea, textOf } from "./forms.js";
import { ACTIVE, stateChip, watchJob } from "./jobs.js";

/** The tester for one eval's check. `check()` is the check as the editor holds it now,
 *  `task()` the task its runs are filed under, `files()` the eval's own text files — a
 *  starting point for a sample workspace. Returns its element, and `refresh()` for when the
 *  check changes in the editor above it. */
export function checkTester({ check, task, files, scope }) {
  const result = h("div", { class: "check-result" });
  let against = "run";
  let rows = 0;

  const runSelect = h("select", { class: "field", name: "trial_run" }, h("option", { value: "" }, "Loading the task's runs…"));
  const fileList = h("div", { class: "stack", style: { gap: "8px" } });
  const addFile = (path = "", content = "") => {
    const index = rows++;
    const row = h("div", { class: "file-row file-text" },
      h("div", { class: "file-row-head" },
        input(`trial_path_${index}`, { value: path, placeholder: "report.json" }),
        h("button", { class: "icon-btn", type: "button", "aria-label": "Remove", onclick: () => { row.remove(); refresh(); } }, "×")),
      textArea(`trial_content_${index}`, content, { rows: 3 }));
    row.dataset.trial = String(index);
    fileList.append(row);
  };
  for (const entry of files()) addFile(entry.path, entry.content);

  const byRun = h("div", null,
    field("Run", runSelect, { wide: true, help: "A recorded run of this task: the check runs in a copy of the workspace it left, on its final reply." }));
  const sample = h("div", { class: "stack", hidden: true },
    field("The reply", textArea("trial_reply", "", { rows: 3, placeholder: "The reply to judge, as the agent might end with it" }), { wide: true }),
    h("div", null, h("div", { class: "label" }, "Files in the workspace"), fileList,
      h("button", { class: "btn btn-small", type: "button", onclick: () => { addFile(); refresh(); } }, "Add a file")));
  const choice = h("div", { class: "controls" },
    [["run", "A recorded run of this task"], ["sample", "A reply and files you write"]].map(([value, label]) =>
      h("label", { class: "check" }, h("input", {
        type: "radio", name: "trial_against", value, checked: value === against,
        onchange: () => {
          against = value;
          byRun.hidden = value !== "run";
          sample.hidden = value !== "sample";
        },
      }), label)));

  const form = planForm({
    kind: "check-trial",
    fields: [choice, byRun, sample],
    read: (element) => {
      const current = written(check());
      if (!current) throw Object.assign(new Error("Write the check first, above."), { field: null });
      if (against === "run") {
        if (!runSelect.value) throw Object.assign(new Error("Choose a run of this task, or try a reply you write."), { field: "trial_run" });
        return { check: current, run: runSelect.value };
      }
      const given = [...fileList.querySelectorAll("[data-trial]")]
        .map((row) => ({
          path: textOf(element, `trial_path_${row.dataset.trial}`).trim(),
          content: element.querySelector(`[name="trial_content_${row.dataset.trial}"]`)?.value ?? "",
        }))
        .filter((entry) => entry.path);
      return { check: current, reply: textOf(element, "trial_reply"), files: given };
    },
    render: (plan) => h("div", { class: "stack", style: { gap: "6px" } },
      h("p", { class: "hint" }, describe(plan)),
      plan.recorded
        ? h("p", { class: "muted small" }, `When it ran, it was judged ${plan.recorded.passed === null || plan.recorded.passed === undefined ? "with no check" : plan.recorded.passed ? "pass" : "fail"}.`)
        : null,
      h("p", { class: "muted small" }, "Nothing is recorded: a run's verdict stays as it is.")),
    startLabel: "Try it",
    onStarted: follow,
  });
  const refresh = () => form.refresh();

  // The task's recorded runs, newest first, each with the verdict it got.
  api.get(`/runs${query({ task: task(), limit: 50 })}`).then((runs) => {
    replace(runSelect, runs.length
      ? [h("option", { value: "" }, "Choose a run…"), runs.map((run) => h("option", { value: run.id },
          `${short(run.id)} · ${run.harness || "?"} · ${verdictText(run.result)} · ${when(run.started_at)}`))]
      : h("option", { value: "" }, `No runs of ${task()} yet: run it first, or try a reply you write`));
  }, () => replace(runSelect, h("option", { value: "" }, "Could not load the task's runs")));

  function follow(job) {
    replace(result, h("div", { class: "controls" }, stateChip(job), h("span", { class: "muted small" }, "trying the check…")));
    watchJob(job.id, scope, (state) => {
      const found = state.job;
      if (!found || ACTIVE.has(found.status)) return;
      if (found.status !== "done") {
        replace(result, h("div", { class: "error-box" }, found.error?.message || `the trial ${found.status}`));
        return;
      }
      replace(result, verdictView(found.result));
    });
  }

  return {
    element: h("div", { class: "check-tester stack" }, form.element, result),
    refresh,
  };
}

/** A check as the editor holds it, if it is one: a command, or a reply check with its text. */
function written(check) {
  if (typeof check === "string") return check.trim() ? check : null;
  if (check && typeof check === "object") return Object.values(check)[0] ? check : null;
  return null;
}

function describe(plan) {
  const shell = plan.check_kind === "shell";
  if (plan.run) {
    return shell
      ? ["Runs the command, as you, in a copy of the workspace ", h("span", { class: "mono" }, `run ${short(plan.run.id)}`), " left, its final reply in $TUNINGFORK_REPLY."]
      : ["Judges the final reply of ", h("span", { class: "mono" }, `run ${short(plan.run.id)}`), ` (${plan.reply_chars} characters).`];
  }
  return shell
    ? `Runs the command, as you, in a new folder holding ${plan.files.length} file(s) you wrote, the reply in $TUNINGFORK_REPLY.`
    : `Judges the reply you wrote (${plan.reply_chars} characters).`;
}

function verdictText(result) {
  if (!result || result.passed === null || result.passed === undefined) return "no verdict";
  return result.passed ? "passed" : "failed";
}

function verdictView(trial) {
  const recorded = trial.recorded?.passed;
  const compared = trial.agrees === null || trial.agrees === undefined
    ? null
    : trial.agrees
      ? h("span", { class: "muted small" }, "as when the run ran")
      : h("span", { class: "check-disagrees" }, `when it ran, the run was judged ${recorded ? "pass" : "fail"}`);
  return h("div", { class: "stack check-verdict" },
    h("div", { class: "controls" }, chip(trial.passed ? "pass" : "fail", trial.passed ? "ok" : "bad"),
      trial.returncode !== null && trial.returncode !== undefined ? h("span", { class: "muted small" }, `exit ${trial.returncode}`) : null,
      compared),
    trial.output_cut
      ? h("p", { class: "hint" }, `${trial.output_cut.toLocaleString()} characters it printed before these are not shown: a verdict keeps the last 2,000, and only they are tallied.`)
      : null,
    trial.output ? h("pre", { class: "code" }, trial.output) : h("p", { class: "muted small" }, "The command printed nothing."),
    h("details", { class: "fold" }, h("summary", null, "What it judged"),
      h("div", { class: "label" }, "The reply"), longText(trial.reply || "(none)", { limit: 1200, lines: 16 }),
      h("div", { class: "label" }, "The workspace"),
      h("p", { class: "mono small" }, trial.found.length ? trial.found.join(", ") + (trial.found_more ? `, and ${trial.found_more} more` : "") : "empty")));
}
