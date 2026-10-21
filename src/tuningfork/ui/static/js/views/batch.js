// Run a task N times through its harness: a batch, started from the browser — or a chain,
// where each run starts from the skill the run before it left.
//
// A task's check is a shell command from its spec, run in every run's workspace — shown
// whole, always, before anything starts — or a check of the final reply, said in words.

import { api, enc } from "../api.js";
import { command, crumbs, h, kv, loading, longText, panel, plural, replace } from "../dom.js";
import { field, input, numberOf, planForm, textOf } from "../forms.js";
import { live } from "../live.js";
import { LABELS } from "../words.js";

const HARNESSES = ["hermes", "claude-code"];

export async function batchView(root, _, scope) {
  replace(root, loading());
  const query = scope.query || new URLSearchParams();
  const prefill = {
    spec: query.get("spec") || "", scenario: query.get("scenario") || "", eval: query.get("eval") || "",
    harness: query.get("harness") || "hermes", model: query.get("model") || "", from: null,
  };
  if (query.get("run")) {
    const detail = await api.get(`/runs/${enc(query.get("run"))}`);
    const launch = detail.launch || {};
    Object.assign(prefill, {
      spec: launch.spec_path || prefill.spec,
      eval: launch.eval_id ? String(launch.eval_id) : prefill.eval,
      harness: launch.harness || detail.run.harness || "hermes",
      model: launch.model || detail.run.model || "",
      from: detail.run,
    });
  }

  const head = h("div", { class: "page-head" }, h("div", null, h("h1", null, "Run a task"),
    h("p", { class: "sub" }, "Each run starts in a fresh folder. In a batch every run gets the same skill. In a chain each run gets the skill the run before it left behind.")));
  if (live.mode !== "live") {
    replace(root, crumbs([["Scenarios", "#/scenarios"], ["Run a task"]]), head,
      panel("From a terminal", h("div", { class: "stack", style: { gap: "8px" } },
        h("p", { class: "hint" }, "Starting runs needs a running proxy (tuningfork serve). Or:"),
        command(`tuningfork batch ${prefill.spec || "<evals.json>"} -n 5 --harness ${prefill.harness}`))));
    return;
  }

  const saved = await api.get("/variants");
  const harnesses = HARNESSES.includes(prefill.harness) ? HARNESSES : [...HARNESSES, prefill.harness];
  const scenarios = await api.get("/scenarios");
  // Which of the spec's tasks (its "evals") to run: offered by what each asks once the plan
  // has read the spec, and back to its first whenever another spec is chosen.
  const task = h("select", { class: "field", name: "eval" },
    h("option", { value: prefill.eval || "" }, prefill.eval ? `task ${prefill.eval}` : "its first task"));
  const anotherSpec = () => replace(task, h("option", { value: "" }, "its first task"));
  // A task file by its path is the scenario's alternative, on the same line: open only while no
  // scenario is chosen, greyed while one is.
  const scenarioSelect = h("select", { class: "field", name: "scenario", id: "batch-scenario" },
    h("option", { value: "" }, "— no scenario —"),
    scenarios.map((each) => h("option", { value: each.name, selected: each.name === prefill.scenario }, each.name)));
  const specInput = input("spec", { value: prefill.spec, placeholder: "/path/to/evals.json", oninput: anotherSpec, "aria-label": "Task file, by its path" });
  const choose = () => {
    specInput.disabled = scenarioSelect.value !== "";
  };
  scenarioSelect.addEventListener("change", () => {
    choose();
    anotherSpec();
  });
  choose();
  const offerTasks = (plan) => {
    const evals = plan.evals || [];
    const before = task.value;
    replace(task, evals.map((item) => h("option", { value: item.id, selected: item.id === plan.eval },
      `${item.id} — ${item.prompt.length > 90 ? `${item.prompt.slice(0, 90)}…` : item.prompt}`)));
    task.disabled = evals.length < 2;
    // The plan was made for what the form said; once it names the task, the form says so
    // too, and is planned once more — or Start would wait on a plan that never comes.
    if (task.value !== before) task.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const fields = [
    prefill.from ? h("p", { class: "hint" }, "Filled in as ", h("a", { class: "mono", href: `#/runs/${prefill.from.id}` }, `run ${prefill.from.id.slice(0, 8)}`), " was started.") : null,
    h("div", { class: "form-grid" },
      // Two controls in one field: a div, its label the scenario's — a <label> holds one control.
      h("div", { class: "form-field wide" },
        h("label", { class: "form-label", for: "batch-scenario" }, "Scenario"),
        h("div", { class: "field-or" }, scenarioSelect, h("span", { class: "or" }, "or"), specInput),
        h("span", { class: "form-help" }, "A scenario is a task file kept in the recordings folder. With none chosen, name a task file (an evals.json) on the proxy's machine by its path.")),
      field("Task in the file", task, { help: "A task file can hold several tasks. Each run does this one.", wide: true }),
      field("Run as", h("select", { class: "field", name: "mode" },
        h("option", { value: "batch" }, "a batch: every run given the skill fresh"),
        h("option", { value: "chain" }, "a chain: each run starts from the skill the one before left")),
        { help: "A chain measures the harness's own self-improvement, link by link. Its runs are not independent samples.", wide: true }),
      field("Runs", input("n", { type: "number", value: 5, min: 1, max: 200 })),
      field("Harness", h("select", { class: "field", name: "harness" },
        harnesses.map((name) => h("option", { value: name, selected: name === prefill.harness }, name)))),
      field("Model", input("model", { value: prefill.model, placeholder: "the one the server lists" })),
      field("Skill", h("select", { class: "field", name: "skill_file" },
        h("option", { value: "" }, "the task file's own"),
        saved.map((variant) => h("option", { value: variant.path }, `${variant.skill}/${variant.name} (saved variant)`))),
        { help: "A saved variant replaces the skill's SKILL.md in every run, to confirm a wording with full runs." })),
    h("details", { class: "fold more-settings" },
      h("summary", null, LABELS.advanced),
      h("div", { class: "form-grid" },
        field("Runs at once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }), { help: "On one GPU, 1 is fastest. A chain's runs go one at a time." }),
        field("Harness options", h("textarea", { class: "field", name: "harness_options", rows: 2, placeholder: '{"await_review": true, "review_every_turn": true}' }),
          { help: "As JSON, recorded with each run. Hermes: await_review (let its background review finish before it exits), review_every_turn (its skill review after every run), mcp_servers.", wide: true }))),
  ];
  const modeOf = (form) => (textOf(form, "mode") === "chain" ? "chain" : "batch");
  const { element } = planForm({
    kind: modeOf,
    fields,
    read: (form) => {
      const scenario = textOf(form, "scenario");
      const spec = textOf(form, "spec").trim();
      // With no scenario chosen, the task file's path is the field open to fill in.
      if (!scenario && !spec) throw Object.assign(new Error("Choose a scenario, or name a task file."), { field: "spec" });
      const chain = modeOf(form) === "chain";
      const params = {
        ...(scenario ? { scenario } : { spec }),
        n: numberOf(form, "n") ?? 5,
        harness: textOf(form, "harness"),
      };
      if (!chain) params.concurrency = numberOf(form, "concurrency") ?? 1;
      const evalId = textOf(form, "eval").trim();
      if (evalId) params.eval = evalId;
      const model = textOf(form, "model").trim();
      if (model) params.model = model;
      const skill = textOf(form, "skill_file");
      if (skill && !chain) params.skill_file = skill;
      const options = textOf(form, "harness_options").trim();
      if (options) {
        let parsed;
        try {
          parsed = JSON.parse(options);
        } catch {
          throw Object.assign(new Error("Harness options must be a JSON object."), { field: "harness_options" });
        }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw Object.assign(new Error("Harness options must be a JSON object."), { field: "harness_options" });
        params.harness_options = parsed;
      }
      return params;
    },
    render: (plan) => {
      offerTasks(plan);
      return batchPreview(plan);
    },
    startLabel: "Start the runs",
    onStarted: (job) => (location.hash = `#/jobs/${job.id}`),
  });
  element.querySelector('[name="mode"]').addEventListener("change", () => {
    const chain = modeOf(element) === "chain";
    element.querySelector('[name="concurrency"]').disabled = chain;
    element.querySelector('[name="skill_file"]').disabled = chain;
  });
  replace(root, crumbs([["Scenarios", "#/scenarios"], ["Run a task"]]), head, panel("The task", element, { className: "start-panel" }));
}

function batchPreview(plan) {
  return h("div", { class: "stack", style: { gap: "8px" } },
    // How a run passes, first: what every number this batch produces will mean.
    h("div", { class: "check-command" },
      plan.check_kind === "shell"
        ? [
            h("div", { class: "label warn-text" }, "How a run passes — this runs as a shell command, as you, in each run's workspace, where the harness worked. $TUNINGFORK_REPLY names a file holding the agent's final reply."),
            h("pre", { class: "code" }, plan.check),
          ]
        : plan.check_kind
          ? [h("div", { class: "label" }, "How a run passes — judged by tuningfork on each run's final reply"), h("p", null, plan.check_text)]
          : [h("div", { class: "label" }, "How a run passes"), h("p", { class: "muted" }, "No check: runs will have no verdict.")]),
    h("div", null, h("div", { class: "label" }, "What each run is asked"), longText(plan.prompt, { limit: 600, lines: 8 })),
    kv([
      ["task", h("span", { class: "mono" }, plan.task_id)],
      ["task file", plan.spec ? h("span", { class: "mono" }, plan.spec) : null],
      ["runs", plan.kind === "chain"
        ? `a chain of ${plural(plan.n, "run")} of ${plan.harness}, one at a time, each starting from the skill the one before left`
        : `${plural(plan.n, "run")} of ${plan.harness}, ${plan.concurrency} at a time`],
      ["model", h("span", { class: "mono" }, plan.model)],
      ["skill", plan.skill_file
        ? h("span", null, h("span", { class: "mono" }, plan.skill_dir || "—"), ", its SKILL.md replaced by ", h("span", { class: "mono" }, plan.skill_file))
        : plan.skill_dir ? h("span", { class: "mono" }, plan.skill_dir) : "none named"],
      ["harness options", plan.harness_options && Object.keys(plan.harness_options).length
        ? h("span", { class: "mono" }, JSON.stringify(plan.harness_options))
        : null],
      ["where", plan.workdir
        ? h("span", null, "each run in a fresh folder under ", h("span", { class: "mono" }, plan.workdir), ", its workspace holding the task's input files")
        : "—"],
      // What the harness sees, which is what the model reads: the same home and folder in
      // every run, or each run's own folders when there is no sandbox (a note says why).
      ["sees", plan.sandbox
        ? h("span", null, "in a sandbox, every run works in ", h("span", { class: "mono" }, plan.sandbox.workspace), ", its home ", h("span", { class: "mono" }, plan.sandbox.home))
        : "its own folders"],
    ]),
    (plan.notes || []).map((note) => h("div", { class: ["banner", note.level === "warning" ? "banner-warn" : ""] }, note.message)));
}
