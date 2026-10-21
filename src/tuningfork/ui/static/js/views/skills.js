// Skills: each skill the recordings know, and every version of it its runs were given — a
// fingerprint of the skill's files — with how often each passed, what its text is known as,
// and its SKILL.md, rendered or as written. Any two of its texts compare line by line. The
// server works all of it out (`ops/skills.py`); this page only shows it — and adds one a person
// names, uploads or keeps from a run (`ops/addskill.py`), or forgets one added.

import { answered, api, enc, query } from "../api.js";
import { guard } from "../auth.js";
import { ago, chip, command, crumbs, h, kv, loading, numbered, panel, plural, rateBar, replace, short, toast, toggle, when } from "../dom.js";
import { field, input, planForm } from "../forms.js";
import { live } from "../live.js";
import { markdownView } from "../markdown.js";
import { diffView } from "./refine.js";

// --- the list -----------------------------------------------------------------------------------

export async function skillsView(root) {
  replace(root, loading());
  const found = await api.get("/skills");
  replace(root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Skills"),
      h("p", { class: "sub" }, "Each skill the recordings know, with every version its runs were given, how often each passed, and its SKILL.md.")),
      live.mode === "live"
        ? h("div", { class: "actions" }, h("a", { class: "btn btn-primary", href: "#/add-skill", "data-control": true }, "Add a skill…"))
        : null),
    found.length
      ? h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "skill"), h("th", { class: "num" }, "tasks"), h("th", { class: "num" }, "runs"),
            h("th", { class: "num" }, "versions"), h("th", null, "refinements"), h("th", null, "latest run"), h("th", null, "its folder"))),
          h("tbody", null, found.map((each) =>
            h("tr", { class: "clickable", onclick: () => (location.hash = `#/skills/${enc(each.name)}`) },
              h("td", null, h("a", { class: "mono", href: `#/skills/${enc(each.name)}` }, each.name)),
              h("td", { class: "num" }, String(each.tasks.length)),
              h("td", { class: "num" }, String(each.runs)),
              h("td", { class: "num" }, String(each.versions)),
              h("td", null, refinementCounts(each.refinements)),
              h("td", { class: "faint", title: when(each.latest) }, each.latest ? ago(each.latest) : "—"),
              h("td", { class: "mono small" }, each.added ? [chip(each.added === "copy" ? "added: a copy" : "added", "muted"), " "] : null,
                each.folders[0] || h("span", {
                  class: "faint", title: "No scenario or task names its folder: a run recorded live or imported keeps only the SKILL.md the model read",
                }, "not known"),
                each.folders.length > 1 ? h("span", { class: "faint" }, ` +${each.folders.length - 1}`) : null)))))))
      : h("div", { class: "panel empty" }, h("h2", null, "No skills yet"),
          h("p", null, "A skill appears here once you add one — ",
            live.mode === "live" ? h("a", { href: "#/add-skill" }, "a folder, an upload, or a run's") : "`tuningfork add-skill <folder>`",
            " — once a run is given it, or once a ", h("a", { href: "#/scenarios" }, "scenario"), " names it.")));
}

// --- adding one ---------------------------------------------------------------------------------

const WAYS = [
  ["folder", "A folder on this machine", "remembered where it is: nothing is copied, and an approved edit is written to it"],
  ["upload", "Upload a folder", "kept as a copy in the recordings folder: an approved edit is written to the copy, never back to yours"],
  ["run", "From a run", "the skill a run had — the copy it kept, else the SKILL.md its model read — kept as a copy the same way"],
];
// Never sent: a repository's history, caches, environments (as `ops.addskill.IGNORED`).
const IGNORED = new Set([".git", "__pycache__", "node_modules", ".venv", ".DS_Store"]);

/** Add a skill — a folder on this machine, a folder uploaded, or the skill a run had — each
 *  planned as its form is filled in (`ops.addskill`), then added and opened on its page.
 *  `?run=` (and `&skill=`) opens it at the run's skill. */
export async function addSkillView(root, _, scope) {
  const asked = scope?.query || new URLSearchParams();
  const head = [crumbs([["Skills", "#/skills"], ["Add a skill"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Add a skill"),
      h("p", { class: "sub" }, "A skill is a folder holding its SKILL.md. Once added, the Skills page lists it, and a scenario can run it.")))];
  if (live.mode !== "live") {
    replace(root, ...head, h("div", { class: "banner banner-warn" }, "Adding a skill needs a running proxy (tuningfork serve). From a terminal:"),
      command("tuningfork add-skill <folder>"));
    return;
  }
  let way = asked.get("run") ? "run" : "folder";
  const slot = h("div");
  const choices = h("fieldset", { class: "target-picker" }, h("legend", null, "How"),
    WAYS.map(([value, label, detail]) => h("label", { class: "target-choice" },
      h("input", { type: "radio", name: "way", value, checked: value === way, onchange: () => { way = value; draw(); } }),
      h("span", null, h("strong", null, label), h("span", { class: "muted small" }, ` — ${detail}`)))));
  const draw = async () => {
    replace(slot, loading());
    const form = way === "folder" ? folderForm() : way === "upload" ? uploadForm() : await runForm(asked);
    if (scope.active) replace(slot, form);
  };
  replace(root, ...head, panel("The skill", h("div", { class: "stack" }, choices, slot)));
  await draw();
}

// Each way's form plans as it is filled in, and adds what it planned: the skill's page opens then.
const ways = {
  send: async (params, extra) => answered(await api.post("/skills/add", extra.dry_run ? { ...params, dry_run: true, quiet: true } : params)),
  onStarted: (made) => {
    location.hash = `#/skills/${enc(made.added.name)}`;
  },
  render: addPlan,
  planLabel: "What will be added",
};

function folderForm() {
  const path = input("path", { placeholder: "e.g. C:\\Users\\you\\skills\\my-skill, or ~/.hermes/skills/my-skill" });
  return planForm({
    ...ways,
    fields: h("div", { class: "form-grid" }, field("Its folder", path, {
      wide: true, help: "The folder holding its SKILL.md, on the machine tuningfork runs on. A Windows path is read as tuningfork will open it.",
    })),
    read: () => {
      if (!path.value.trim()) throw Object.assign(new Error("Name the skill's folder."), { field: "path" });
      return { kind: "folder", path: path.value.trim() };
    },
    startLabel: "Add it",
  }).element;
}

function uploadForm() {
  const picker = h("input", { class: "field", type: "file", name: "folder", webkitdirectory: true, multiple: true });
  const name = input("name", { placeholder: "its folder's name" });
  const said = h("span", null, "The folder holding its SKILL.md: a repository's history and caches are left out.");
  // The chosen folder's files, read once chosen: sent with every plan, as they will be kept.
  let files = null;
  let top = "";
  picker.addEventListener("change", async () => {
    files = null;
    const chosen = [...(picker.files || [])];
    top = chosen[0]?.webkitRelativePath.split("/")[0] || "";
    const parts = (file) => file.webkitRelativePath.split("/").slice(1);
    const sent = chosen.filter((file) => !parts(file).some((part) => IGNORED.has(part)));
    replace(said, "reading…");
    files = await Promise.all(sent.map(async (file) => ({ path: parts(file).join("/"), data: await base64(file) })));
    name.placeholder = cleaned(top) || "its folder's name";
    replace(said, `${plural(files.length, "file")} from ${top}`
      + (chosen.length > sent.length ? `; ${chosen.length - sent.length} left out: a repository's history, caches` : ""));
    // Read: the form plans again.
    picker.dispatchEvent(new Event("input", { bubbles: true }));
  });
  return planForm({
    ...ways,
    fields: h("div", { class: "form-grid" },
      field("Its folder", picker, { help: said }),
      field("Kept as", name, { help: "The name the copy is kept under: its folder's, by default." })),
    read: () => {
      if (!files) throw Object.assign(new Error("Choose the skill's folder."), { field: "folder" });
      return { kind: "upload", name: name.value.trim() || cleaned(top), files };
    },
    startLabel: "Upload it",
  }).element;
}

async function runForm(asked) {
  const runs = await api.get("/runs?limit=100").catch(() => []);
  const wanted = asked.get("run") || "";
  const match = runs.find((each) => wanted && (each.id === wanted || each.id.startsWith(wanted)));
  const run = h("select", { class: "field", name: "run" }, h("option", { value: "" }, "— choose a run —"),
    match || !wanted ? null : h("option", { value: wanted, selected: true }, wanted),
    runs.map((each) => h("option", { value: each.id, selected: each === match },
      `${short(each.id)} · ${each.label || each.task_id || "no label"}`)));
  const skill = h("select", { class: "field", name: "skill", disabled: true }, h("option", { value: "" }, "—"));
  const name = input("name", { placeholder: "the skill's own name" });
  const loadSkills = async () => {
    replace(skill, h("option", { value: "" }, run.value ? "reading…" : "—"));
    skill.disabled = true;
    if (run.value) {
      try {
        const found = (await api.get(`/runs/${enc(run.value)}/skills`)).skills;
        replace(skill, found.length
          ? found.map((each) => h("option", { value: each.name, selected: each.name === asked.get("skill") },
              `${each.name} — ${each.kept ? "the copy the run kept" : "as its model read it"}`))
          : h("option", { value: "" }, "none: it kept no skill, and shows none"));
        skill.disabled = !found.length;
      } catch (error) {
        replace(skill, h("option", { value: "" }, error.message));
      }
    }
    skill.dispatchEvent(new Event("change", { bubbles: true }));
  };
  run.addEventListener("change", loadSkills);
  const form = planForm({
    ...ways,
    fields: h("div", { class: "form-grid" },
      field("Run", run, { help: "Imported, started by tuningfork or recorded live: any run whose skill it kept, or whose conversation shows one." }),
      field("Its skill", skill),
      field("Kept as", name, { help: "The name the copy is kept under: the skill's own, by default." })),
    read: () => {
      if (!run.value) throw Object.assign(new Error("Choose a run."), { field: "run" });
      if (!skill.value) throw Object.assign(new Error("Choose one of its skills."), { field: "skill" });
      return { kind: "run", run: run.value, skill: skill.value, ...(name.value.trim() ? { name: name.value.trim() } : {}) };
    },
    startLabel: "Keep it",
  });
  if (run.value) loadSkills();
  return form.element;
}

/** What adding it would do, as `ops.addskill` planned it: nothing is written yet. */
function addPlan(plan) {
  return h("div", { class: "stack", style: { gap: "6px" } },
    kv([
      ["name", h("span", { class: "mono" }, plan.name)],
      ["kept", plan.kind === "folder" ? "where it is: nothing copied" : "as a copy in the recordings folder"],
      ["its folder", h("span", { class: "mono" }, plan.path)],
      ["files", plan.kind === "copy" ? `${plural(plan.files.length, "file")}, ${plan.bytes.toLocaleString()} bytes` : null],
    ]),
    plan.warnings.map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    plan.notes.map((note) => h("div", { class: "banner" }, note)));
}

/** A name as the server cleans a folder's into a skill's: lowercase, dashes for the rest. */
function cleaned(text) {
  return text.toLowerCase().replace(/[^a-z0-9_.-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
}

async function base64(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let text = "";
  for (let at = 0; at < bytes.length; at += 32768) text += String.fromCharCode(...bytes.subarray(at, at + 32768));
  return btoa(text);
}

function refinementCounts(counts = {}) {
  const parts = Object.entries(counts).filter(([, n]) => n);
  if (!parts.length) return h("span", { class: "faint" }, "—");
  return parts.map(([status, n]) => chip(`${n} ${status}`, status === "approved" ? "ok" : status === "open" ? "accent" : "muted"));
}

// --- one skill ------------------------------------------------------------------------------------

export async function skillView(root, [name]) {
  replace(root, loading());
  const skill = await api.get(`/skills/${enc(name)}`);
  // The texts this page can show, each as the server names it (`ops.skills.REFS`).
  const texts = textChoices(skill);
  const first = texts.find((each) => each.ref.startsWith("file")) || texts[0] || null;
  const state = { shown: first?.ref || null, against: "", mode: "rendered" };
  const viewer = h("div");

  replace(root,
    crumbs([["Skills", "#/skills"], [skill.name]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, skill.name),
      h("p", { class: "sub" }, skill.tasks.length ? `Tasks: ${skill.tasks.join(", ")}` : "No run has been given this skill yet.")),
      waysIn(skill)),
    h("div", { class: "stack" },
      skill.folders.map((folder) => folderPanel(folder, versionName(skill))),
      panel("SKILL.md", viewer),
      versionsPanel(skill, (ref, against = "") => {
        state.shown = ref;
        state.against = against;
        drawViewer();
        viewer.scrollIntoView({ behavior: "smooth", block: "start" });
      }),
      variantsPanel(skill, (ref) => {
        state.shown = ref;
        state.against = "";
        drawViewer();
        viewer.scrollIntoView({ behavior: "smooth", block: "start" });
      }),
      refinementsPanel(skill),
      changelogPanels(skill)));
  drawViewer();

  async function drawViewer() {
    if (!texts.length) {
      replace(viewer, h("p", { class: "muted" }, "No text of this skill can be read: no run's snapshot, and no SKILL.md in its folder."));
      return;
    }
    const pick = (value, onChange, blank) => h("select", { class: "field", onchange: (event) => onChange(event.target.value) },
      blank ? h("option", { value: "", selected: value === "" }, blank) : null,
      texts.map((each) => h("option", { value: each.ref, selected: each.ref === value }, each.label)));
    const controls = h("div", { class: "skill-controls" },
      h("label", { class: "field-inline" }, h("span", { class: "muted small" }, "Show"),
        pick(state.shown, (value) => { state.shown = value; drawViewer(); })),
      h("label", { class: "field-inline" }, h("span", { class: "muted small" }, "against"),
        pick(state.against, (value) => { state.against = value; drawViewer(); }, "— nothing: show it alone")),
      state.against ? null : toggle([["rendered", "Rendered"], ["source", "As written"]], state.mode, (mode) => {
        state.mode = mode;
        drawViewer();
      }));
    const body = h("div", null, loading());
    replace(viewer, controls, body);
    try {
      if (state.against) {
        const diff = await api.get(`/skills/${enc(skill.name)}/diff${query({ before: state.against, after: state.shown })}`);
        replace(body,
          h("p", { class: "muted small" }, h("span", { class: "diff-del-text" }, `− ${diff.before.label}`), "   ",
            h("span", { class: "diff-add-text" }, `+ ${diff.after.label}`), `  ·  ${plural(diff.changed, "line")} changed`),
          diff.hunks.length ? diffView(diff.hunks) : h("p", { class: "muted" }, "The same text, line endings aside."));
        return;
      }
      const choice = texts.find((each) => each.ref === state.shown);
      const text = choice?.text ?? (await api.get(`/skills/${enc(skill.name)}/text${query({ ref: state.shown })}`)).text;
      replace(body, state.mode === "rendered" ? markdownView(text) : numbered(text));
    } catch (error) {
      replace(body, h("p", { class: "form-error" }, error.message));
    }
  }
}

/** A skill's ways in: run it again — from a scenario that names it, else from the spec its
 *  runs were launched from — or make a scenario for it, with its name and folder filled in
 *  where its folder is known, else of one of the runs it was in. */
function waysIn(skill) {
  if (live.mode !== "live") return null;
  const folder = skill.folders[0]?.path;
  const spec = skill.folders.flatMap((each) => each.specs)[0];
  const run = (scenario, label) =>
    h("a", { class: "btn btn-small", href: `#/batch${query({ scenario })}`, "data-control": true, title: `Run a task: scenario ${scenario}` }, label);
  return h("div", { class: "actions" },
    skill.scenarios.length === 1 ? run(skill.scenarios[0], "Run it…")
      : skill.scenarios.length ? skill.scenarios.slice(0, 3).map((scenario) => run(scenario, `Run ${scenario}…`))
        : spec ? h("a", { class: "btn btn-small", href: `#/batch${query({ spec })}`, "data-control": true, title: `Run a task: ${spec}` }, "Run it…") : null,
    folder
      ? h("a", { class: "btn btn-small", href: `#/scenarios/new${query({ skill: skill.name, skill_dir: folder })}`, "data-control": true }, "Make a scenario for it…")
      : h("a", { class: "btn btn-small", href: "#/scenarios?from=run", "data-control": true,
          title: "Its folder is not known: a scenario of one of the runs it was in keeps the SKILL.md the model read" }, "Make a scenario of one of its runs…"));
}

/** A version's name on this page: its number in order of first run, its fingerprint beside it —
 *  the fingerprint the Runs page shows. One the skill's versions do not hold keeps its fingerprint. */
function versionName(skill) {
  const order = [...skill.versions].sort((a, b) => a.first - b.first);
  const numbers = new Map(order.map((version, index) => [version.fingerprint, index + 1]));
  return (fingerprint) => (numbers.has(fingerprint) ? `version ${numbers.get(fingerprint)} (${fingerprint})` : `version ${fingerprint}`);
}

/** Every text the page can show: its own file now, each version, each variant, each old
 *  version an approval kept. Those whose text is already here carry it. */
function textChoices(skill) {
  const named = versionName(skill);
  const choices = [];
  skill.folders.forEach((folder, index) => {
    if (folder.text === null) return;
    choices.push({ ref: index ? `file:${index}` : "file", text: folder.text,
      label: `its own file now${skill.folders.length > 1 ? ` (${folder.path})` : ""}${folder.version ? ` — ${named(folder.version)}` : " — not run yet"}` });
  });
  for (const version of skill.versions) {
    if (version.text === null) continue;
    const known = version.known_as.length ? ` — ${version.known_as.join("; ")}` : "";
    choices.push({ ref: version.fingerprint, text: version.text, label: `${named(version.fingerprint)} · ${plural(version.runs, "run")}${known}` });
  }
  for (const variant of skill.variants) choices.push({ ref: `variant:${variant.name}`, text: null, label: `variant ${variant.name}` });
  for (const folder of skill.folders) {
    for (const old of folder.history) choices.push({ ref: `history:${old.name}`, text: null, label: `replaced by an approval: ${old.name}` });
  }
  return choices;
}

function folderPanel(folder, named) {
  const status = folder.text === null
    ? chip("no SKILL.md there", "warn")
    : folder.version ? chip(`its SKILL.md is ${named(folder.version)}`, "accent", "Runs have been given exactly this text")
      : chip("its SKILL.md has not been run yet", "warn", "No run has been given this text: run a task to measure it");
  return panel("Its own folder", h("div", { class: "stack", style: { gap: "6px" } },
    folder.added ? addedLine(folder.added) : null,
    h("div", null, h("span", { class: "mono" }, folder.path), " ", status),
    folder.windows ? h("div", { class: "muted small" }, "From Windows: ", h("span", { class: "mono" }, folder.windows)) : null,
    folder.specs.length ? h("div", { class: "muted small" }, `Named by ${plural(folder.specs.length, "task file")}: `,
      h("span", { class: "mono" }, folder.specs.join(", "))) : null,
    folder.changelog !== null || folder.history.length
      ? h("div", { class: "muted small" }, "Beside it, written by approvals: ",
          folder.changelog !== null ? "a changelog" : null, folder.changelog !== null && folder.history.length ? " and " : null,
          folder.history.length ? plural(folder.history.length, "old version") : null, ".")
      : null));
}

/** How a person added the folder, and a way to forget it: a folder named stays where it is; a
 *  copy is moved aside, so forgetting one takes a second click. */
function addedLine(added) {
  const copy = added.kind === "copy";
  const from = added.origin?.run
    ? ["from run ", h("a", { class: "mono", href: `#/runs/${added.origin.run}` }, short(added.origin.run)), `: ${added.origin.how}`]
    : added.origin?.upload ? [added.origin.upload.from ? `copied from ${added.origin.upload.from}` : "uploaded"] : null;
  return h("div", { class: "added-line" },
    chip(copy ? "added: tuningfork's copy" : "added: a folder on this machine", "muted"),
    h("span", { class: "muted small" }, copy
      ? [from ? [from, " — "] : null, "approved edits are written to this copy, never back to where it came from"]
      : "remembered where it is: approved edits are written to it"),
    live.mode === "live" ? forgetButton(added) : null);
}

function forgetButton(added) {
  const label = "Forget it…";
  const button = h("button", { class: "btn btn-small btn-quiet", type: "button", "data-control": true }, label);
  const note = h("span", { class: "muted small" });
  let armed = null;
  const disarm = () => {
    clearTimeout(armed);
    armed = null;
    button.classList.remove("armed");
    button.textContent = label;
  };
  button.addEventListener("click", guard(async () => {
    // A copy goes aside whole: said on the first click, done on the second.
    if (added.kind === "copy" && !armed) {
      button.classList.add("armed");
      button.textContent = "Click again: move the copy aside";
      armed = setTimeout(disarm, 6000);
      return;
    }
    disarm();
    button.disabled = true;
    try {
      const gone = await api.del(`/skills/added/${enc(added.name)}`);
      toast(gone.moved_to ? `Forgot ${added.name}: its copy is moved aside to ${gone.moved_to}` : `Forgot ${added.name}: its folder stays where it is`,
        { timeout: 6000 });
      location.hash = "#/skills";
    } catch (error) {
      button.disabled = false;
      note.textContent = error.message;
    }
  }));
  return h("span", { class: "controls" }, button, note);
}

function versionsPanel(skill, show) {
  if (!skill.versions.length) {
    return panel("Versions", h("p", { class: "muted" }, "None yet: no run has been given this skill."));
  }
  const fileRef = skill.folders.find((folder) => folder.text !== null) ? "file" : null;
  const order = [...skill.versions].sort((a, b) => a.first - b.first).map((version) => version.fingerprint);
  return panel(`Versions · ${skill.versions.length}`, [
    h("p", { class: "hint" }, "A version is every run given the same skill files, byte for byte, numbered in order of first run. Each rate is that version's alone — runs under another wording are another sample — with its 95% interval."),
    h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, "version"), h("th", null, "pass rate, by task · harness · model"),
        h("th", { class: "num" }, "runs"), h("th", null, "when"), h("th", null, ""))),
      h("tbody", null, skill.versions.map((version) => h("tr", null,
        h("td", null, h("div", null, h("strong", null, `version ${order.indexOf(version.fingerprint) + 1}`),
          " ", h("span", { class: "mono faint small", title: "A fingerprint of its files, as the Runs page names it" }, version.fingerprint)),
          version.known_as.map((label) => h("div", null, chip(label, label.startsWith("its own file") ? "accent" : "skill")))),
        h("td", null, version.conditions.map((condition) => h("div", { class: "condition-rate" },
          h("span", { class: "muted small" }, [condition.task_id, condition.harness, condition.model].filter(Boolean).join(" · ")),
          condition.passed ? rateBar(condition.passed) : h("span", { class: "faint small" }, " no verdicts"),
          condition.uncounted ? h("span", { class: "faint small" }, ` · ${condition.uncounted} not counted`) : null))),
        h("td", { class: "num" }, String(version.runs)),
        h("td", { class: "faint small", title: `${when(version.first)} – ${when(version.latest)}` }, ago(version.latest)),
        h("td", { class: "row-actions" },
          version.text !== null ? h("button", { class: "btn btn-small", type: "button", onclick: () => show(version.fingerprint) }, "Show") : null,
          version.text !== null && fileRef && !version.known_as.some((label) => label.startsWith("its own file"))
            ? h("button", { class: "btn btn-small btn-quiet", type: "button", onclick: () => show("file", version.fingerprint) }, "Compare with the file")
            : null))))))]);
}

function variantsPanel(skill, show) {
  if (!skill.variants.length) return null;
  return panel(`Saved variants · ${skill.variants.length}`, h("div", { class: "table-wrap" }, h("table", { class: "table" },
    h("thead", null, h("tr", null, h("th", null, "variant"), h("th", null, "what"), h("th", null, "saved"), h("th", null, ""))),
    h("tbody", null, skill.variants.map((variant) => h("tr", null,
      h("td", { class: "mono" }, variant.name),
      h("td", null,
        variant.proposal ? ["a proposal of ", h("a", { class: "mono", href: `#/refine/${variant.proposal.refinement}` }, `refinement ${variant.proposal.refinement}`)] : "a wording saved for an experiment",
        variant.version ? [" · ", chip(`run as ${versionName(skill)(variant.version)}`, "skill")] : null),
      h("td", { class: "faint small", title: when(variant.modified) }, ago(variant.modified)),
      h("td", { class: "row-actions" }, h("button", { class: "btn btn-small", type: "button", onclick: () => show(`variant:${variant.name}`) }, "Show"))))))));
}

function refinementsPanel(skill) {
  if (!skill.refinements.length) return null;
  return panel(`Refinements · ${skill.refinements.length}`, h("ul", { class: "plain-list" }, skill.refinements.map((each) =>
    h("li", null, h("a", { class: "mono", href: `#/refine/${each.id}` }, short(each.id)), " ",
      chip(each.status, each.status === "approved" ? "ok" : each.status === "open" ? "accent" : "muted"),
      h("span", { class: "muted" }, ` at run ${short(each.run)} step ${each.step} · ${plural(each.proposals, "proposal")}`),
      each.target ? h("div", { class: "small" }, each.target) : null))));
}

function changelogPanels(skill) {
  return skill.folders.filter((folder) => folder.changelog !== null).map((folder) =>
    panel("Changelog", h("div", null,
      h("p", { class: "hint" }, "Written beside the skill's folder by each approval: what changed, why, and the evidence it stood on."),
      h("details", { class: "fold" }, h("summary", null, h("span", { class: "fold-preview" }, `${folder.path}.changelog.md`)),
        markdownView(folder.changelog)))));
}
