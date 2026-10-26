// Scenarios: tasks kept in the recordings folder — an evals.json and its input files — made
// and edited here, and run as batches.
//
// The editor edits the spec itself: what is on screen beside the form is the exact
// evals.json that will be saved. A task's check is either a shell command that runs, as
// you, in every run's workspace — always shown whole, under a warning — or a check of the
// agent's final reply, which tuningfork judges itself.

import { answered, api, enc } from "../api.js";
import { guard } from "../auth.js";
import { checkPicker } from "../checktry.js";
import { ago, chip, crumbs, firstLine, h, importedRun, kv, loading, panel, plural, rateBar, recordedLive, replace, short, toast, verdict, when } from "../dom.js";
import { field, formError, input, planForm, textArea } from "../forms.js";
import { live } from "../live.js";
import { leaving } from "../nav.js";
import { skillNameLabel, skillNameOf, skillPicker } from "../skillpick.js";
import { IGNORED, base64 } from "./skills.js";

// As `analysis.normalize.KNOWN_NORMALIZERS`; the server checks the list on every save.
const NORMALIZERS = ["tool_call_json", "python_ast", "json_content", "tool_name"];
const NAME = /^[a-z0-9][a-z0-9_-]{0,62}$/;

export async function scenariosView(root, _, scope) {
  if (scope.query.get("from") === "run") return runPickView(root, scope);
  replace(root, loading());
  const found = await api.get("/scenarios");
  const importer = live.mode === "live" ? importForm() : null;
  replace(root,
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Scenarios"),
        h("p", { class: "sub" }, "Each scenario is a task file kept in the recordings folder, with its input files. Run one as a batch from its page.")),
      live.mode === "live"
        ? h("div", { class: "actions" },
            h("a", { class: "btn", href: "#/scenarios?from=run", "data-control": true }, "From a run…"),
            h("a", { class: "btn btn-primary", href: "#/scenarios/new", "data-control": true }, "New scenario"))
        : null),
    h("div", { class: "stack" },
      found.length
        ? h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
            h("thead", null, h("tr", null, h("th", null, "scenario"), h("th", null, "skill"), h("th", { class: "num" }, "tasks"),
              h("th", null, "passed, latest set-up first"), h("th", null, "changed"))),
            h("tbody", null, found.map((scenario) =>
              h("tr", { class: "clickable", onclick: () => (location.hash = `#/scenarios/${scenario.name}`) },
                h("td", null, h("a", { class: "mono", href: `#/scenarios/${scenario.name}` }, scenario.name),
                  scenario.problems ? chip(`${plural(scenario.problems, "problem")}`, "warn") : null),
                h("td", { class: "mono" }, scenario.skill_name || "—"),
                h("td", { class: "num" }, String(scenario.evals)),
                h("td", null, scenario.results.length
                  ? h("div", { class: "stack", style: { gap: "4px" } }, scenario.results.slice(0, 3).map((result) =>
                      h("div", { class: "scenario-result" },
                        h("span", { class: "muted small mono" }, `${result.task_id} · ${result.harness || "?"}${result.skill_file ? " · variant" : ""}`),
                        result.passed ? rateBar(result.passed) : h("span", { class: "muted small" }, "no verdict"))))
                  : h("span", { class: "faint" }, "not run yet")),
                h("td", { class: "faint", title: when(scenario.modified) }, ago(scenario.modified))))))))
        : h("div", { class: "panel empty" }, h("h2", null, "No scenarios yet"),
            h("p", null, live.mode === "live"
              ? "Make one here, or from a run: a chat imported, a run tuningfork started, or one recorded live. Or add a task "
                + "file you have: its input files are copied in, and its skill folder stays where it is."
              : "Scenarios are made with a running proxy (tuningfork serve).")),
      importer));
}

// Which run a scenario is made of: any run but a thread — imported, launched, or recorded as it
// happened — newest first, filtered as the Runs page filters; each opens the run's scenario form.
// A chat had elsewhere is imported first, and its scenario's form follows.
async function runPickView(root, scope) {
  const head = [
    crumbs([["Scenarios", "#/scenarios"], ["From a run"]]),
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "A scenario from a run"),
        h("p", { class: "sub" }, "Pick any run: a chat imported, a run tuningfork started, or one recorded live. The scenario "
          + "starts at the person's message just before the skill came in, and each of its runs is given the conversation up to there.")),
      live.mode === "live"
        ? h("div", { class: "actions" },
            h("a", { class: "btn", href: "#/import?then=scenario", "data-control": true }, "Import a chat…"))
        : null),
  ];
  if (live.mode !== "live") {
    replace(root, ...head, h("div", { class: "banner banner-warn" },
      "Making a scenario needs a running proxy (tuningfork serve). From a terminal: tuningfork scenario <run> <name>"));
    return;
  }
  replace(root, ...head, loading());
  // Threads are left out by the list itself; a run with no step has no conversation to continue.
  const runs = (await api.get("/runs?limit=1000")).filter((run) => run.step_count > 0);
  if (!scope.active) return;
  let filter = "";
  const list = h("div", { class: "stack" });
  const search = h("input", {
    class: "field search", type: "search", placeholder: "Filter by run, label, task or model", "aria-label": "Filter runs",
    oninput: (event) => {
      filter = event.target.value.trim().toLowerCase();
      draw();
    },
  });
  const draw = () => {
    const shown = runs.filter((run) => !filter
      || [run.id, run.label, run.task_id, run.model].some((value) => value && String(value).toLowerCase().includes(filter)));
    replace(list, !runs.length
      ? h("div", { class: "panel empty" }, h("h2", null, "No runs yet"),
          h("p", null, "Import a chat an agent had elsewhere, or record one: point a harness at this proxy, or run a task."))
      : !shown.length ? h("p", { class: "muted" }, "No runs match the filter.")
        : h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
            h("thead", null, h("tr", null, h("th", null, "run"), h("th", null, "label"), h("th", null, "task"),
              h("th", null, "harness"), h("th", { class: "num" }, "steps"), h("th", null, "verdict"), h("th", null, "recorded"))),
            h("tbody", null, shown.map((run) => {
              const href = `#/runs/${run.id}/scenario`;
              return h("tr", { class: "clickable", onclick: () => (location.hash = href) },
                h("td", null, h("a", { class: "mono", href }, short(run.id))),
                h("td", null, run.label || "", run.imported ? [" ", importedRun()] : null,
                  run.recorded === "live" ? [" ", recordedLive()] : null),
                h("td", { class: "mono" }, run.task_id || "—"),
                h("td", null, run.harness || "—"),
                h("td", { class: "num" }, String(run.step_count)),
                h("td", null, verdict(run.result)),
                h("td", { class: "faint", title: when(run.started_at) }, ago(run.started_at)));
            }))))));
  };
  replace(root, ...head, h("div", { class: "stack" }, h("div", { class: "actions" }, search), list));
  draw();
}

// A task file kept as a scenario: one on this machine, by its path, or its folder uploaded —
// each planned as its form is filled in (`ops.scenarios.prepare_import`, `prepare_upload`), as
// Add a skill plans, then made and opened.
const IMPORT_WAYS = [
  ["path", "By its path", "a task file on this machine: its skill folder stays where it is, and approved edits are written to it"],
  ["upload", "Upload its folder", "the folder holding the task file, with its skill and what its check reads: kept as copies in the recordings folder"],
];

function importForm() {
  let way = "path";
  const slot = h("div");
  const choices = h("fieldset", { class: "target-picker" }, h("legend", null, "How"),
    IMPORT_WAYS.map(([value, label, detail]) => h("label", { class: "target-choice" },
      h("input", { type: "radio", name: "import-way", value, checked: value === way, onchange: () => { way = value; draw(); } }),
      h("span", null, h("strong", null, label), h("span", { class: "muted small" }, ` — ${detail}`)))));
  const draw = () => replace(slot, way === "path" ? byPath() : byUpload());
  draw();
  return panel("Add a task file you have", h("div", { class: "stack" }, choices, slot));
}

const keeping = {
  send: async (params, extra) => answered(await api.post("/scenarios/import", extra.dry_run ? { ...params, dry_run: true, quiet: true } : params)),
  onStarted: (made) => {
    toast(`Added scenario ${made.scenario.name}: scenarios/${made.scenario.name}/evals.json`, { tone: "muted", timeout: 6000 });
    location.hash = `#/scenarios/${made.scenario.name}`;
  },
  render: importPlan,
  planLabel: "What will be added",
};

function byPath() {
  const path = input("path", { placeholder: "e.g. C:\\Users\\you\\evals.json, or /home/you/evals.json" });
  const name = input("name", { placeholder: "a-name" });
  return planForm({
    ...keeping,
    fields: h("div", { class: "form-grid" },
      field("The task file", path, {
        wide: true,
        help: "Its full path. Under WSL a Windows path is read where WSL mounts it (C:\\… as /mnt/c/…); a relative path is taken from the folder tuningfork was started in.",
      }),
      field("Scenario name", name, { help: "lowercase letters, digits, - and _" })),
    read: () => {
      if (!path.value.trim()) throw Object.assign(new Error("Name the task file by its path."), { field: "path" });
      if (!name.value.trim()) throw Object.assign(new Error("Name the scenario: lowercase letters, digits, - and _."), { field: "name" });
      return { path: path.value.trim(), name: name.value.trim() };
    },
    startLabel: "Add it",
  }).element;
}

function byUpload() {
  const picker = h("input", { class: "field", type: "file", name: "folder", webkitdirectory: true, multiple: true });
  const which = h("select", { class: "field", name: "spec", disabled: true }, h("option", { value: "" }, "its evals.json"));
  const name = input("name", { placeholder: "its folder's name" });
  const said = h("span", null, "The folder holding the task file, with what it names: its skill folder, its input files, "
    + "what its check reads. A repository's history and caches are left out.");
  // The chosen folder's files, by their paths in it, each read only as it is sent: a plan sends
  // each one's size, and the task files at its top, which it reads; the scenario made, all of
  // them (found in review: every plan sent the whole folder again, once a pause in typing).
  let files = null;
  let top = "";
  const atTop = (path) => !path.includes("/") && path.endsWith(".json");
  picker.addEventListener("change", () => {
    const chosen = [...(picker.files || [])];
    top = chosen[0]?.webkitRelativePath.split("/")[0] || "";
    const parts = (file) => file.webkitRelativePath.split("/").slice(1);
    const sent = chosen.filter((file) => !parts(file).some((part) => IGNORED.has(part)));
    files = sent.map((file) => ({ path: parts(file).join("/"), file }));
    // The task files at its top: its evals.json, unless the person chooses another.
    const tops = files.map((file) => file.path).filter(atTop).sort();
    replace(which, tops.length
      ? tops.map((path) => h("option", { value: path, selected: path === "evals.json" || (tops.length === 1) }, path))
      : h("option", { value: "" }, "no .json file at its top"));
    which.disabled = tops.length < 2;
    name.placeholder = scenarioNameOf(top) || "a-name";
    replace(said, `${plural(files.length, "file")} from ${top}`
      + (chosen.length > sent.length ? `; ${chosen.length - sent.length} left out: a repository's history, caches` : ""));
    // Read: the form plans again.
    picker.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return planForm({
    ...keeping,
    send: async (params, extra) => {
      const chosen = new Map((files || []).map((entry) => [entry.path, entry.file]));
      const sent = await Promise.all(params.files.filter(({ path }) => chosen.has(path)).map(async ({ path, size }) =>
        (extra.dry_run && !atTop(path) ? { path, size } : { path, data: await base64(chosen.get(path)) })));
      return keeping.send({ ...params, files: sent }, extra);
    },
    fields: h("div", { class: "form-grid" },
      field("The task file's folder", picker, { wide: true, help: said }),
      field("The task file in it", which, { help: "Where the folder's top holds several." }),
      field("Scenario name", name, { help: "lowercase letters, digits, - and _: its folder's name, by default" })),
    read: () => {
      if (!files) throw Object.assign(new Error("Choose the task file's folder."), { field: "folder" });
      return {
        name: name.value.trim() || scenarioNameOf(top), files: files.map(({ path, file }) => ({ path, size: file.size })),
        folder: top, ...(which.value ? { spec: which.value } : {}),
      };
    },
    startLabel: "Upload it",
  }).element;
}

/** What keeping a task file as a scenario would do, as `ops.scenarios` planned it: nothing is
 *  written yet. */
function importPlan(plan) {
  const mono = (text) => h("span", { class: "mono" }, text);
  return h("div", { class: "stack", style: { gap: "6px" } },
    kv([
      ["scenario", [mono(plan.name), h("span", { class: "muted" }, ` — scenarios/${plan.name}/evals.json`)]],
      ["from", mono(plan.source)],
      ["tasks", plan.tasks.length ? [mono(plan.tasks.map((id) => `${plan.skill_name}/${id}`).join(", ")),
        h("span", { class: "muted" }, ", each run filed under its own")] : null],
      ["skill folder", plan.skill_dir
        ? [mono(plan.skill_dir), h("span", { class: "muted" }, plan.skill_copied ? " — a copy, kept in the scenario's own folder"
          : plan.has_skill ? " — kept where it is" : " — holds no SKILL.md")]
        : h("span", { class: "muted" }, "none named")],
      ["input files", plan.inputs.length ? mono(plan.inputs.join(", ")) : h("span", { class: "muted" }, "none")],
      ["beside its evals.json", plan.beside.length ? mono(plan.beside.join(", ")) : null],
    ]),
    plan.warnings.map((text) => h("div", { class: "banner banner-warn" }, text)),
    plan.notes.map((text) => h("div", { class: "banner" }, text)));
}

/** A folder's name as a scenario's: lowercase letters, digits, - and _. */
function scenarioNameOf(text) {
  return String(text || "").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^[-_]+|-+$/g, "").slice(0, 63);
}

/** What was dropped, a folder walked whole: each file with its path, the folder's own name first.
 *  Read as it is dropped: the drop's items are gone once the handler returns. */
function dropped(transfer) {
  const entries = [...(transfer.items || [])].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  const plain = [...(transfer.files || [])];
  if (!entries.length) return Promise.resolve(plain.map((file) => ({ file, path: file.name })));
  const found = [];
  const walk = async (entry, at) => {
    if (entry.isFile) {
      found.push({ file: await new Promise((resolve, reject) => entry.file(resolve, reject)), path: `${at}${entry.name}` });
      return;
    }
    if (!entry.isDirectory || IGNORED.has(entry.name)) return;
    const reader = entry.createReader();
    // A reader gives a folder's entries a batch at a time, until one comes back empty.
    for (;;) {
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) await walk(child, `${at}${entry.name}/`);
    }
  };
  return entries.reduce((done, entry) => done.then(() => walk(entry, "")), Promise.resolve()).then(() => found);
}

// --- the editor --------------------------------------------------------------------------

const blank = () => ({
  skill_name: "",
  tuningfork: { skill_dir: "" },
  evals: [newEval("1")],
});

function newEval(id) {
  return {
    id, prompt: "", expected_output: "", files: [], expectations: [],
    tuningfork: { check: "", normalizers: [], held_out: false, skills: [], timeout: 900, max_turns: 12 },
  };
}

/** The spec as it will be saved: empty optional fields left out, as a person would write it. */
function cleaned(draft) {
  const spec = { skill_name: draft.skill_name.trim() };
  if (draft.tuningfork?.skill_dir?.trim()) spec.tuningfork = { ...draft.tuningfork, skill_dir: draft.tuningfork.skill_dir.trim() };
  spec.evals = draft.evals.map((item) => {
    const options = { ...item.tuningfork };
    const check = options.check;
    // A shell command, or a reply check with its text; anything unfinished is no check yet.
    const kept = check && typeof check === "object" ? Object.values(check)[0] : typeof check === "string" ? check.trim() : "";
    if (!kept) delete options.check;
    return {
      ...item,
      id: String(item.id).trim(),
      expectations: item.expectations.filter((line) => line.trim()),
      tuningfork: options,
    };
  });
  for (const [key, value] of Object.entries(draft)) {
    if (!(key in spec) && key !== "tuningfork") spec[key] = value; // keys this page does not know
  }
  return spec;
}

export async function scenarioView(root, [name], scope) {
  replace(root, loading());
  const isNew = name === "new";
  // The skills the Skills page lists, each a folder to choose: none where they cannot be read.
  const [loaded, skills] = await Promise.all([
    isNew ? null : api.get(`/scenarios/${enc(name)}`),
    api.get("/skills").catch(() => []),
  ]);
  let draft = loaded ? structuredClone(loaded.spec) : blank();
  draft.tuningfork ??= { skill_dir: "" };
  // Come from a skill's page: its name and its folder filled in.
  if (isNew) {
    draft.skill_name = scope.query?.get("skill") || draft.skill_name;
    draft.tuningfork.skill_dir = scope.query?.get("skill_dir") || draft.tuningfork.skill_dir;
  }
  // A spec written elsewhere — imported from a file, made from a chat — may leave out what this
  // page writes: each eval with every field, defaults where it is silent.
  draft.evals = (draft.evals || []).map((item) => {
    const blankEval = newEval(String(item.id ?? ""));
    return { ...blankEval, ...item, tuningfork: { ...blankEval.tuningfork, ...(item.tuningfork || {}) } };
  });
  let etag = loaded?.etag ?? null;
  let inputs = loaded?.inputs ?? [];
  let saved = JSON.stringify(cleaned(draft));
  const editable = live.mode === "live";

  const nameField = isNew ? input("scenario_name", { placeholder: "a-name" }) : null;
  // The skill's name follows the folder chosen while it is empty, or still the name the folder
  // gave it: one a person typed stays.
  let given = skillNameOf(loaded?.checked?.paths?.skill_dir?.path || draft.tuningfork.skill_dir);
  const skillName = bind(input("skill_name", { value: draft.skill_name, placeholder: "the skill folder's name" }),
    (v) => (draft.skill_name = v));
  const follow = (folder) => {
    const named = skillNameOf(folder);
    if (!named || named === draft.skill_name.trim() || (draft.skill_name.trim() && draft.skill_name.trim() !== given)) return false;
    given = named;
    draft.skill_name = named;
    skillName.value = named;
    return true;
  };
  // The skill folder: one the Skills page lists, or a path — relative to this scenario's own
  // folder, shown before it — each resolved, with every check, where runs will find it.
  const picker = skillPicker({
    skills, path: draft.tuningfork.skill_dir || "", resolved: loaded?.checked?.paths?.skill_dir?.path ?? null,
    relative: true, editable, name: "editor",
    onChange: ({ kind, path }) => {
      draft.tuningfork.skill_dir = path;
      if (kind === "known") follow(path);
      changed();
    },
  });
  const evalsEl = h("div", { class: "stack" });
  const previewEl = h("pre", { class: "code spec-preview" });
  const checkedEl = h("div", { class: "stack", style: { gap: "6px" } });
  const status = h("div");
  // Whether what is on the page is saved: said beside Save, and again with every change.
  const saveState = h("span", { class: "save-state" });
  const dirty = () => JSON.stringify(cleaned(draft)) !== saved;
  leaving.guard = () => (dirty() ? "This scenario has changes that are not saved. Leave anyway?" : null);
  scope.onLeave(() => (leaving.guard = null));

  replace(root,
    crumbs([["Scenarios", "#/scenarios"], [isNew ? "new" : name]]),
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, isNew ? "New scenario" : name),
        h("p", { class: "sub" }, isNew ? "Saved as scenarios/<name>/evals.json in the recordings folder." : `scenarios/${name}/evals.json`)),
      editable ? h("div", { class: "actions" },
        saveState,
        h("button", { class: "btn", type: "button", "data-control": true, onclick: guard(() => save(false)) }, "Save"),
        h("button", { class: "btn btn-primary", type: "button", "data-control": true, onclick: guard(() => save(true)) }, "Save and run…"),
        isNew ? null : deleteButton(name)) : null),
    status,
    h("div", { class: "grid-2" },
      h("div", { class: "stack" },
        panel("The task file", h("div", { class: "form-grid" },
          nameField ? field("Scenario name", nameField, { help: "lowercase letters, digits, - and _", wide: true }) : null,
          field(skillNameLabel(), skillName, { wide: true, help: "What its tasks' runs are filed under, <skill name>/<task Id>, "
            + "never shown to the model. The skill folder's name, unless you type another." }),
          h("div", { class: "form-field wide" }, h("span", { class: "form-label" }, "Skill folder"), picker.element))),
        evalsEl,
        loaded?.context ? contextPanel(name, loaded.context, editable) : null,
        editable ? h("button", { class: "btn", type: "button", "data-control": true, onclick: () => {
          const ids = new Set(draft.evals.map((item) => String(item.id)));
          let next = draft.evals.length + 1;
          while (ids.has(String(next))) next += 1;
          draft.evals.push(newEval(String(next)));
          drawEvals();
          changed();
        } }, "Add a task") : null),
      h("div", { class: "stack sticky-col" },
        panel("Before it runs", checkedEl),
        panel("The task file (evals.json), exactly as it will be saved", previewEl))));

  function bind(control, set) {
    control.disabled = !editable;
    control.addEventListener("input", () => {
      set(control.value);
      changed();
    });
    return control;
  }

  let timer = null;
  // A new scenario's problems are what is left to fill in until it is first edited, and say so
  // as notes (found in review: the empty form opened on a red "evals[0].prompt: is required").
  let touched = !isNew;
  function changed({ quiet = false } = {}) {
    if (!quiet) touched = true;
    replace(previewEl, JSON.stringify(cleaned(draft), null, 2));
    drawSaved();
    clearTimeout(timer);
    timer = setTimeout(check, 400);
  }
  // The scenario's name says where its folder is, and so where a relative skill folder is.
  nameField?.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(check, 400);
  });

  /** Beside Save: whether the page holds what the scenario's file says. */
  function drawSaved() {
    replace(saveState, isNew || !etag
      ? chip("not saved yet", "warn", "Nothing is written until Save: its folder in the recordings folder is made then")
      : dirty() ? chip("changes not saved", "warn", "Save writes them to its evals.json")
        : chip("saved", "muted", `What the page shows is what scenarios/${name}/evals.json says`));
  }

  async function check() {
    if (!editable) {
      replace(checkedEl, (loaded?.checked?.problems || []).map(problem), (loaded?.checked?.warnings || []).map(warning));
      picker.setBase(loaded?.checked?.folder);
      drawPath(loaded?.checked?.paths?.skill_dir, loaded?.checked?.wsl);
      return;
    }
    try {
      const found = await api.post("/scenarios/validate", { spec: cleaned(draft), name: isNew ? nameField.value.trim() || null : name });
      replace(checkedEl,
        found.problems.length ? found.problems.map(problem) : h("p", { class: "muted small" }, "No problems."),
        found.warnings.map(warning));
      picker.setBase(found.folder);
      drawPath(found.paths?.skill_dir, found.wsl);
    } catch (error) {
      replace(checkedEl, formError(error));
    }
  }

  /** What the skill folder names, as tuningfork will open it; a folder holding a SKILL.md names
   *  the skill too, while its name follows the folder. */
  function drawPath(found, wsl) {
    const note = picker.note;
    // How a typed path is read: nothing to say of a skill chosen from the list.
    const underWsl = wsl && picker.value().kind === "path"
      ? h("span", { class: "path-line" }, "tuningfork runs in WSL: a Windows path (C:\\…) is read where WSL mounts it, /mnt/c/….") : null;
    if (!found) return replace(note, underWsl);
    if (!found.path) {
      return replace(note, h("span", { class: "path-line" }, `→ resolved against ${found.relative_to}, once the scenario has a name`), underWsl);
    }
    if (found.has_skill_md && follow(found.path)) changed({ quiet: true });
    const state = found.has_skill_md ? "holds a SKILL.md" : found.exists ? "holds no SKILL.md" : "does not exist";
    replace(note,
      h("span", { class: ["path-line", found.has_skill_md ? "path-ok" : "path-bad"] }, "→ ", h("span", { class: "mono" }, found.path), ` · ${state}`),
      found.translated_from ? h("span", { class: "path-line" }, "translated from ", h("span", { class: "mono" }, found.translated_from), "; saved as the path above") : null,
      found.translated_from ? null : underWsl);
  }

  const problem = (found) => h("div", { class: ["banner", touched && "banner-bad"] },
    found.where ? h("span", { class: "mono" }, `${found.where}: `) : null, found.message);
  const warning = (text) => h("div", { class: "banner banner-warn" }, text);

  function drawEvals() {
    replace(evalsEl, draft.evals.map((item, index) => evalEditor(item, index)));
  }

  function evalEditor(item, index) {
    const options = item.tuningfork;
    const files = h("div", { class: "stack", style: { gap: "6px" } });
    // Made once a task: the folder typed for uploads stays as the rows are drawn again.
    let uploads = null;
    const drawFiles = () => replace(files, item.files.map((entry, position) => fileRow(item, entry, position, drawFiles)), uploads);
    uploads = uploader(item, index, drawFiles);
    drawFiles();
    const normalizers = h("div", { class: "controls" }, NORMALIZERS.map((normalizer) =>
      h("label", { class: "check" },
        h("input", {
          type: "checkbox", checked: options.normalizers.includes(normalizer), disabled: !editable,
          onchange: (event) => {
            options.normalizers = event.target.checked
              ? [...options.normalizers, normalizer]
              : options.normalizers.filter((n) => n !== normalizer);
            changed();
          },
        }), normalizer)));
    return panel(`Task ${item.id || index + 1}`, h("div", { class: "stack", style: { gap: "10px" } },
      h("div", { class: "form-grid" },
        field("Id", bind(input(`id-${index}`, { value: item.id }), (v) => (item.id = v))),
        field("Preloaded skills", bind(input(`skills-${index}`, { value: options.skills.join(", "), placeholder: "none" }),
          (v) => (options.skills = v.split(",").map((s) => s.trim()).filter(Boolean))), { help: "By folder name, comma-separated." })),
      field("Prompt", bind(textArea(`prompt-${index}`, item.prompt, { rows: 4 }), (v) => (item.prompt = v)), { help: "The task as the harness is given it." }),
      field("Expected output", bind(textArea(`expected-${index}`, item.expected_output, { rows: 2 }), (v) => (item.expected_output = v)),
        { help: "A note for people reading the task file; nothing checks it. To judge runs by the agent's reply, choose “The reply contains…” under How a run passes." }),
      field("Expectations", bind(textArea(`expectations-${index}`, item.expectations.join("\n"), { rows: 3 }), (v) => (item.expectations = v.split("\n"))),
        { help: "One a line. For people only: not checked." }),
      h("div", null, h("div", { class: "label" }, "Input files, put in each run's workspace"), files),
      checkEditor(item, index),
      h("div", { class: "form-grid" },
        field("Max turns", bind(input(`turns-${index}`, { type: "number", value: options.max_turns, min: 1 }), (v) => (options.max_turns = Number(v)))),
        field("Timeout (s)", bind(input(`timeout-${index}`, { type: "number", value: options.timeout, min: 1 }), (v) => (options.timeout = Number(v))))),
      // A scenario of a chat that went on to another skill, which no run of this one has: each
      // run ends where the model loads it, and the check judges the workspace then.
      h("div", null,
        h("label", { class: "check" },
          h("input", {
            type: "checkbox", name: `stop-${index}`, checked: options.stop_at_next_skill === true, disabled: !editable,
            onchange: (event) => {
              if (event.target.checked) options.stop_at_next_skill = true;
              else delete options.stop_at_next_skill;
              changed();
            },
          }),
          " Stop where the model loads another skill"),
        h("p", { class: "hint" }, "Each run ends there, and the check judges the workspace as it is then: for a chat that "
          + "went on to a skill no run of this one has.")),
      field("How behaviours are grouped", normalizers, { help: "Which answers count as the same behaviour when runs are compared. None ticked means the defaults.", wide: true }),
      editable && draft.evals.length > 1
        ? h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => {
            draft.evals.splice(index, 1);
            drawEvals();
            changed();
          } }, "Remove this task")
        : null));
  }

  /** How a run passes (`checkPicker`, shared with a scenario made from a run), written into
   *  the eval as the spec holds a check: "" for none, a command, or a reply check. */
  function checkEditor(item, index) {
    const options = item.tuningfork;
    return checkPicker({
      check: options.check, editable, scope, name: String(index),
      onChange: ({ kind, text }) => {
        options.check = kind === "shell" ? text : kind === "none" ? "" : { [kind]: text };
        changed();
      },
      // Trying the check as written, before trusting it: opened below it on demand.
      tester: live.mode === "live" && editable
        ? {
            task: () => `${draft.skill_name.trim()}/${String(item.id).trim()}`,
            files: () => item.files
              .filter((entry) => entry && typeof entry === "object" && typeof entry.content === "string")
              .map((entry) => ({ path: entry.path, content: entry.content })),
          }
        : null,
    }).element;
  }

  function fileRow(item, entry, position, redraw) {
    const remove = editable ? h("button", { class: "icon-btn", type: "button", "aria-label": "Remove", onclick: () => {
      item.files.splice(position, 1);
      redraw();
      changed();
    } }, "×") : null;
    if (typeof entry === "string") {
      return h("div", { class: "file-row" }, h("span", { class: "mono" }, entry), h("span", { class: "muted small" }, "beside the task file"), remove);
    }
    if (entry.source !== undefined) {
      const uploaded = inputs.find((found) => `inputs/${found.path}` === entry.source);
      // Where it goes in each run's workspace, changed here: what was uploaded stays as it is.
      const path = bind(input(`file-${position}`, { value: entry.path, "aria-label": "Its path in each run's workspace" }),
        (v) => (entry.path = v));
      return h("div", { class: "file-row" }, path,
        uploaded ? h("span", { class: "muted small" }, `${uploaded.size.toLocaleString()} bytes, uploaded`) : chip("not uploaded", "warn"),
        remove);
    }
    const path = bind(input(`file-${position}`, { value: entry.path }), (v) => (entry.path = v));
    const content = bind(textArea(`content-${position}`, entry.content ?? "", { rows: 4 }), (v) => (entry.content = v));
    return h("div", { class: "file-row file-text" }, h("div", { class: "file-row-head" }, path, remove), content);
  }

  /** Files added to a task's inputs: dropped, chosen, or a folder whole, each put in the
   *  workspace folder typed beside them — uploaded into the scenario's own folder, which only a
   *  saved scenario has (found in use: files dropped before the first save went nowhere). */
  function uploader(item, index, redraw) {
    if (!editable) return null;
    const ready = !isNew && Boolean(etag);
    const chooser = h("input", { type: "file", multiple: true, hidden: true, disabled: !ready, onchange: () => chosen(chooser) });
    const folders = h("input", { type: "file", multiple: true, webkitdirectory: true, hidden: true, disabled: !ready, onchange: () => chosen(folders) });
    const into = input(`upload-into-${index}`, { placeholder: "the workspace itself", disabled: !ready, "aria-label": "Workspace folder for uploads" });
    const failures = h("div");
    const zone = h("div", {
      class: ["drop-zone", !ready && "disabled"], role: "button", tabindex: ready ? "0" : "-1", "aria-disabled": String(!ready),
      onclick: () => ready && chooser.click(),
      onkeydown: (event) => ready && (event.key === "Enter" || event.key === " ") && chooser.click(),
      ondragover: (event) => {
        event.preventDefault();
        if (ready) zone.classList.add("over");
      },
      ondragleave: () => zone.classList.remove("over"),
      ondrop: (event) => {
        event.preventDefault();
        zone.classList.remove("over");
        if (ready) dropped(event.dataTransfer).then(upload);
      },
    }, ready ? "Drop files or a folder here, or click to choose files. Kept in the scenario's inputs/."
      : "Save the scenario first: files are kept in its folder, which saving makes.");
    const text = h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => {
      item.files.push({ path: "notes.txt", content: "" });
      redraw();
      changed();
    } }, "Add a text file");

    function chosen(control) {
      // A folder chosen keeps its own name and what is inside it, as a folder dropped does.
      upload([...control.files].map((file) => ({ file, path: file.webkitRelativePath || file.name })));
      control.value = "";
    }

    async function upload(found) {
      if (!ready) return;
      const folder = into.value.trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
      for (const { file, path } of found.filter((each) => !each.path.split("/").some((part) => IGNORED.has(part)))) {
        const placed = folder ? `${folder}/${path}` : path;
        try {
          const done = await api.put(`/scenarios/${enc(name)}/inputs/${placed.split("/").map(enc).join("/")}`, file);
          inputs = [...inputs.filter((each) => each.path !== done.path), done];
          if (!item.files.some((entry) => entry.source === `inputs/${done.path}`)) {
            item.files.push({ path: done.path, source: `inputs/${done.path}` });
          }
        } catch (error) {
          failures.append(h("div", { class: "form-error-box" }, h("strong", null, `${placed}: ${error.message}`),
            h("button", { class: "btn btn-small", type: "button", onclick: (event) => {
              event.target.closest(".form-error-box").remove();
              upload([{ file, path }]);
            } }, "Retry")));
        }
      }
      redraw();
      changed();
    }
    return h("div", { class: "stack", style: { gap: "6px" } },
      zone, chooser, folders,
      h("div", { class: "form-grid" },
        field("Into the workspace folder", into, { help: "Where in each run's workspace uploaded files go: data puts rows.csv at data/rows.csv. Each file's own path can be changed after." }),
        h("div", { class: "form-field upload-more" },
          h("span", { class: "form-label" }, "More"),
          h("div", { class: "controls" },
            h("button", { class: "btn btn-small", type: "button", disabled: !ready, onclick: () => folders.click() }, "Upload a folder…"),
            text))),
      failures);
  }

  async function save(run) {
    replace(status);
    const target = isNew ? nameField.value.trim() : name;
    if (isNew && !NAME.test(target)) {
      replace(status, formError({ message: "Name the scenario: lowercase letters, digits, - and _." }));
      return;
    }
    try {
      const done = await api.put(`/scenarios/${enc(target)}`, JSON.stringify({ spec: cleaned(draft), etag }), "application/json");
      etag = done.etag;
      inputs = done.inputs;
      // A Windows path is saved translated: the file says what runs use.
      const kept = done.spec?.tuningfork?.skill_dir;
      if (typeof kept === "string" && kept !== draft.tuningfork.skill_dir?.trim()) {
        draft.tuningfork.skill_dir = kept;
        picker.setPath(kept);
        changed();
      }
      saved = JSON.stringify(cleaned(draft));
      // Said where it is seen whatever comes next: the page opened at the scenario, or the run form.
      toast(`Saved scenarios/${target}/evals.json`, { tone: "muted", timeout: 5000 });
      if (run) location.hash = `#/batch?scenario=${enc(target)}`;
      else if (isNew) location.hash = `#/scenarios/${enc(target)}`;
      else drawSaved();
    } catch (error) {
      replace(status, formError(error));
    }
  }

  drawEvals();
  changed({ quiet: true });
}

/** The conversation a scenario continues (`tuningfork.context`): the messages before its prompt
 *  — a list, as it was kept before — and, as {before, in_turn}, the model's earlier steps of the
 *  prompt's own turn; each message to read, and the whole as JSON, to edit and save. */
function contextPanel(name, context, editable) {
  const status = h("div");
  const parts = (found) => (Array.isArray(found) ? { before: found, inTurn: [] }
    : { before: found?.before || [], inTurn: found?.in_turn || [] });
  const said = (message) => (message.tool_calls?.length
    ? `calls ${message.tool_calls.map((call) => call.function?.name || "?").join(", ")}`
    : firstLine(typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? ""), 140));
  const listOf = (found) => found.map((message) => h("li", null, h("strong", null, `${message.role}: `), said(message)));
  const about = h("p", { class: "hint" });
  const reading = h("div", { class: "stack" });
  const show = (found) => {
    const { before, inTurn } = parts(found);
    // Made of a chat imported, a run tuningfork launched or one recorded live (found in review:
    // it said "an imported chat" of all three).
    replace(about, (before.length
      ? `${plural(before.length, "message")} of the conversation it was made of, put after the harness's instructions in every `
        + "call of each run: the task's prompt is its next turn."
      : "Nothing comes before the task's prompt: it is the chat's first turn.")
      + (inTurn.length ? ` Then ${plural(inTurn.length, "message")} of that turn, the model's earlier steps — already done, `
        + "an earlier skill among them — follow the prompt, so the skill comes in live." : ""));
    replace(reading,
      h("div", null, h("div", { class: "label" }, "Before the prompt"), before.length ? h("ol", { class: "small" }, listOf(before))
        : h("p", { class: "muted small" }, "nothing: the prompt is the chat's first turn")),
      inTurn.length ? h("div", null, h("div", { class: "label" }, "In the prompt's turn, taken as done"),
        h("ol", { class: "small" }, listOf(inTurn))) : null);
  };
  show(context);
  const editor = h("textarea", { class: "field mono", rows: 16, spellcheck: "false", disabled: !editable,
    value: JSON.stringify(context, null, 1) });
  const save = guard(async () => {
    replace(status);
    let parsed;
    try {
      parsed = JSON.parse(editor.value);
    } catch (error) {
      replace(status, formError(new Error(`It is not JSON: ${error.message}`)));
      return;
    }
    try {
      const done = await api.put(`/scenarios/${enc(name)}/context`, JSON.stringify({ messages: parsed }), "application/json");
      show(parsed);
      replace(status, h("p", { class: "hint" }, `Saved: ${plural(done.messages, "message")}.`));
    } catch (error) {
      replace(status, formError(error));
    }
  });
  return panel("The conversation it continues", h("div", { class: "stack" },
    about,
    h("details", { class: "fold" }, h("summary", null, "Read it"), reading),
    h("details", { class: "fold" }, h("summary", null, "Edit it as JSON"), editor,
      editable ? h("div", { class: "actions" },
        h("button", { class: "btn", type: "button", "data-control": true, onclick: save }, "Save the conversation")) : null,
      status)));
}

function deleteButton(name) {
  const button = h("button", { class: "btn btn-danger", type: "button", "data-control": true }, "Delete");
  let armed = null;
  button.addEventListener("click", guard(async () => {
    if (!armed) {
      button.classList.add("armed");
      button.textContent = "Click again: move it to the trash";
      armed = setTimeout(() => {
        armed = null;
        button.classList.remove("armed");
        button.textContent = "Delete";
      }, 4000);
      return;
    }
    clearTimeout(armed);
    await api.del(`/scenarios/${enc(name)}`);
    leaving.guard = null;
    location.hash = "#/scenarios";
  }));
  return button;
}
