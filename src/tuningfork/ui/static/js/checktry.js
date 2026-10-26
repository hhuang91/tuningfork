// Trying a check before trusting it, where it is written: against a recorded run of the task
// — the workspace it left and its final reply — or against a reply and files typed here.
// The check tried is the one on screen, saved or not. It runs on the proxy's machine in a
// throwaway folder, and nothing is recorded: a run's own verdict never changes.

import { api, query } from "./api.js";
import { guard } from "./auth.js";
import { chip, h, longText, replace, short, when } from "./dom.js";
import { field, input, planForm, textArea, textOf } from "./forms.js";
import { ACTIVE, stateChip, watchJob } from "./jobs.js";

let pickers = 0;

/** How a run passes, as a person chooses it, wherever a check is written — a scenario's
 *  editor, a scenario made from a run: no check, the agent's final reply holding or matching
 *  something, or a shell command run in the workspace; given `own`, also the check of the task
 *  the run came from, as it is. The kind chosen is kept apart from what is typed for each, so
 *  choosing one before typing it does not fall back to "no check". `onChange({ kind, text })`
 *  hears each change. `tester` — what `checkTester` takes, but the check — puts "Test this
 *  check…" under it (`where` says what it tries the check on). `name` keeps each picker's
 *  fields apart on a page: `check-kind-<name>`, `check-<name>`, `check-text-<name>`. */
export function checkPicker({ check = null, own = null, editable = true, onChange = () => {}, tester = null, scope = null, name = `p${pickers++}` }) {
  let kind = own !== null && (check === null || check === undefined) ? "own"
    : check && typeof check === "object" ? Object.keys(check)[0]
      : typeof check === "string" && check.trim() ? "shell" : "none";
  let shell = typeof check === "string" ? check : "";
  let reply = check && typeof check === "object" ? String(Object.values(check)[0] ?? "") : "";
  const value = () => ({ kind, text: kind === "shell" ? shell : kind === "own" || kind === "none" ? "" : reply });
  // The check as a spec holds it — what a trial tries.
  const written = () => (kind === "own" ? own : kind === "shell" ? shell : kind === "none" ? null : { [kind]: reply });
  const body = h("div", { class: "stack", style: { gap: "6px" } });
  const testerSlot = h("div");
  let opened = null;
  const changed = () => {
    onChange(value());
    opened?.refresh();
  };
  const tryIt = tester && editable
    ? h("button", {
        class: "btn btn-small", type: "button", "data-control": true,
        onclick: guard(() => {
          if (opened) {
            opened = null;
            replace(testerSlot);
            return;
          }
          opened = checkTester({ ...tester, check: written, scope });
          replace(testerSlot, opened.element);
        }),
      }, "Test this check…")
    : null;
  const choices = [
    ...(own !== null ? [["own", "Its task's own"]] : []),
    ["none", "No check"],
    ["reply_contains", "The reply contains…"],
    ["reply_matches", "The reply matches…"],
    ["shell", "A shell command"],
  ];
  const picker = h("div", { class: "controls" }, choices.map(([choice, label]) =>
    h("label", { class: "check" },
      h("input", {
        type: "radio", name: `check-kind-${name}`, value: choice, checked: kind === choice, disabled: !editable,
        onchange: () => {
          kind = choice;
          draw();
          changed();
        },
      }), label)));

  function draw() {
    if (kind === "own") {
      return replace(body,
        h("p", { class: "form-help" }, "The check the run's task had, as its task file words it, with any file it reads from the task's folder."),
        h("pre", { class: "code small" }, typeof own === "string" ? own : JSON.stringify(own)));
    }
    if (kind === "shell") {
      const text = textArea(`check-${name}`, shell, {
        rows: 3, placeholder: 'e.g. python3 -c "import json; json.load(open(\'report.json\'))", or grep -q 61.5 "$TUNINGFORK_REPLY"',
      });
      text.disabled = !editable;
      text.addEventListener("input", () => {
        shell = text.value;
        changed();
      });
      return replace(body,
        h("p", { class: "form-help" }, "Runs as you, in the run's workspace — a fresh folder holding the task's input files, where the harness worked. It passes when it exits 0. $TUNINGFORK_REPLY names a file holding the agent's final reply, and $TUNINGFORK_SPEC_DIR the scenario's folder: a longer check can be a script there, where the agent cannot read it."),
        text);
    }
    if (kind === "reply_contains" || kind === "reply_matches") {
      const text = input(`check-text-${name}`, { value: reply, placeholder: kind === "reply_contains" ? "e.g. 61.5" : "e.g. (?i)total\\D+61\\.5" });
      text.disabled = !editable;
      text.addEventListener("input", () => {
        reply = text.value;
        changed();
      });
      return replace(body,
        h("p", { class: "form-help" }, kind === "reply_contains"
          ? "Passes when the agent's final reply holds this text exactly (case counts). tuningfork judges it; nothing runs."
          : "Passes when this regular expression (Python's) finds a match in the agent's final reply; start it with (?i) to ignore case, and . matches line breaks too. tuningfork judges it; nothing runs."),
        text);
    }
    return replace(body, h("p", { class: "form-help" }, "Runs have no verdict: none is counted as passed or failed."));
  }
  draw();
  return {
    element: h("div", { class: "check-command" }, h("div", { class: "label" }, "How a run passes"), picker, body,
      tryIt ? h("div", { class: "controls" }, tryIt,
        h("span", { class: "muted small" }, tester.where || "on a recorded run of this task, or on a reply you write — before trusting it")) : null,
      testerSlot),
    value,
  };
}

/** The tester for one eval's check. `check()` is the check as the editor holds it now,
 *  `task()` the task its runs are filed under, `files()` the eval's own text files — a
 *  starting point for a sample workspace. `runs()`, given, offers those runs instead of the
 *  task's, the first chosen: a scenario not yet made has none of its own, and is tried on the
 *  run it is made of (`runLabel`, `runHelp` say so). Returns its element, and `refresh()` for
 *  when the check changes in the editor above it. */
export function checkTester({ check, task, files, scope, runs = null,
  runLabel = "A recorded run of this task",
  runHelp = "A recorded run of this task: the check runs in a copy of the workspace it left, on its final reply." }) {
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

  const byRun = h("div", null, field("Run", runSelect, { wide: true, help: runHelp }));
  const sample = h("div", { class: "stack", hidden: true },
    field("The reply", textArea("trial_reply", "", { rows: 3, placeholder: "The reply to judge, as the agent might end with it" }), { wide: true }),
    h("div", null, h("div", { class: "label" }, "Files in the workspace"), fileList,
      h("button", { class: "btn btn-small", type: "button", onclick: () => { addFile(); refresh(); } }, "Add a file")));
  const choice = h("div", { class: "controls" },
    [["run", runLabel], ["sample", "A reply and files you write"]].map(([value, label]) =>
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
      plan.unsandboxed
        ? h("div", { class: "banner banner-warn" },
          `The command is not sandboxed: ${plan.unsandboxed}. It runs on this machine itself.`)
        : null,
      plan.recorded
        ? h("p", { class: "muted small" }, `When it ran, it was judged ${plan.recorded.passed === null || plan.recorded.passed === undefined ? "with no check" : plan.recorded.passed ? "pass" : "fail"}.`)
        : null,
      h("p", { class: "muted small" }, "Nothing is recorded: a run's verdict stays as it is.")),
    startLabel: "Try it",
    onStarted: follow,
  });
  const refresh = () => form.refresh();

  // The task's recorded runs, newest first, each with the verdict it got — or the runs given,
  // the first chosen.
  (runs ? runs() : api.get(`/runs${query({ task: task(), limit: 50 })}`)).then((found) => {
    replace(runSelect, found.length
      ? [runs ? null : h("option", { value: "" }, "Choose a run…"), found.map((run) => h("option", { value: run.id },
          `${short(run.id)} · ${run.harness || "?"} · ${verdictText(run.result)} · ${when(run.started_at)}`))]
      : h("option", { value: "" }, `No runs of ${task()} yet: run it first, or try a reply you write`));
    if (runs && found.length) refresh();
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
  // Where a command runs: in a run's sandbox, as a run's check does, or as you without one.
  const runs = plan.sandboxed ? "Runs the command in a run's sandbox," : "Runs the command, as you,";
  if (plan.run) {
    return shell
      ? [`${runs} in a copy of the workspace `, h("span", { class: "mono" }, `run ${short(plan.run.id)}`), " left, its final reply in $TUNINGFORK_REPLY."]
      : ["Judges the final reply of ", h("span", { class: "mono" }, `run ${short(plan.run.id)}`), ` (${plan.reply_chars} characters).`];
  }
  return shell
    ? `${runs} in a new folder holding ${plan.files.length} file(s) you wrote, the reply in $TUNINGFORK_REPLY.`
    : `Judges the reply you wrote (${plan.reply_chars} characters).`;
}

function verdictText(result) {
  if (!result || result.passed === null || result.passed === undefined) return "no verdict";
  return result.passed ? "passed" : "failed";
}

export function verdictView(trial) {
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
      ? h("p", { class: "hint" }, `${trial.output_cut.toLocaleString()} characters of its output before these are not shown: a verdict keeps the last 2,000, and only they are tallied.`)
      : null,
    trial.output ? h("pre", { class: "code" }, trial.output) : h("p", { class: "muted small" }, "The command printed nothing."),
    h("details", { class: "fold" }, h("summary", null, "What it judged"),
      h("div", { class: "label" }, "The reply"), longText(trial.reply || "(none)", { limit: 1200, lines: 16 }),
      h("div", { class: "label" }, "The workspace"),
      h("p", { class: "mono small" }, trial.found.length ? trial.found.join(", ") + (trial.found_more ? `, and ${trial.found_more} more` : "") : "empty")));
}
