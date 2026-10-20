// Skills: each skill the recordings know, and every version of it its runs were given — a
// fingerprint of the skill's files — with how often each passed, what its text is known as,
// and its SKILL.md, rendered or as written. Any two of its texts compare line by line. The
// server works all of it out (`ops/skills.py`); this page only shows it.

import { api, enc, query } from "../api.js";
import { ago, chip, crumbs, h, loading, numbered, panel, plural, rateBar, replace, short, toggle, when } from "../dom.js";
import { live } from "../live.js";
import { markdownView } from "../markdown.js";
import { diffView } from "./refine.js";

// --- the list -----------------------------------------------------------------------------------

export async function skillsView(root) {
  replace(root, loading());
  const found = await api.get("/skills");
  replace(root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Skills"),
      h("p", { class: "sub" }, "Each skill the recordings know, with every version its runs were given — a fingerprint of its files — how often each passed, and its SKILL.md."))),
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
              h("td", { class: "mono small" }, each.folders[0] || h("span", {
                class: "faint", title: "No scenario or task names its folder: a run recorded live or imported keeps only the SKILL.md the model read",
              }, "not known"),
                each.folders.length > 1 ? h("span", { class: "faint" }, ` +${each.folders.length - 1}`) : null)))))))
      : h("div", { class: "panel empty" }, h("h2", null, "No skills yet"),
          h("p", null, "A skill appears here once a run is given it — ",
            live.mode === "live" ? h("a", { href: "#/batch" }, "run a task") : "run a task (`tuningfork batch <spec.json>`)",
            " — or once a ", h("a", { href: "#/scenarios" }, "scenario"), " names it.")));
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
      skill.folders.map(folderPanel),
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

/** Every text the page can show: its own file now, each version, each variant, each old
 *  version an approval kept. Those whose text is already here carry it. */
function textChoices(skill) {
  const choices = [];
  skill.folders.forEach((folder, index) => {
    if (folder.text === null) return;
    choices.push({ ref: index ? `file:${index}` : "file", text: folder.text,
      label: `its own file now${skill.folders.length > 1 ? ` (${folder.path})` : ""}${folder.version ? ` — version ${folder.version}` : " — not run yet"}` });
  });
  for (const version of skill.versions) {
    if (version.text === null) continue;
    const known = version.known_as.length ? ` — ${version.known_as.join("; ")}` : "";
    choices.push({ ref: version.fingerprint, text: version.text, label: `version ${version.fingerprint} · ${plural(version.runs, "run")}${known}` });
  }
  for (const variant of skill.variants) choices.push({ ref: `variant:${variant.name}`, text: null, label: `variant ${variant.name}` });
  for (const folder of skill.folders) {
    for (const old of folder.history) choices.push({ ref: `history:${old.name}`, text: null, label: `replaced by an approval: ${old.name}` });
  }
  return choices;
}

function folderPanel(folder) {
  const status = folder.text === null
    ? chip("no SKILL.md there", "warn")
    : folder.version ? chip(`its SKILL.md is version ${folder.version}`, "accent", "Runs have been given exactly this text")
      : chip("its SKILL.md has not been run yet", "warn", "No run has been given this text: run a task to measure it");
  return panel("Its own folder", h("div", { class: "stack", style: { gap: "6px" } },
    h("div", null, h("span", { class: "mono" }, folder.path), " ", status),
    folder.windows ? h("div", { class: "muted small" }, "From Windows: ", h("span", { class: "mono" }, folder.windows)) : null,
    folder.specs.length ? h("div", { class: "muted small" }, `Named by ${plural(folder.specs.length, "spec")}: `,
      h("span", { class: "mono" }, folder.specs.join(", "))) : null,
    folder.changelog !== null || folder.history.length
      ? h("div", { class: "muted small" }, "Beside it, written by approvals: ",
          folder.changelog !== null ? "a changelog" : null, folder.changelog !== null && folder.history.length ? " and " : null,
          folder.history.length ? plural(folder.history.length, "old version") : null, ".")
      : null));
}

function versionsPanel(skill, show) {
  if (!skill.versions.length) {
    return panel("Versions", h("p", { class: "muted" }, "None yet: no run has been given this skill."));
  }
  const fileRef = skill.folders.find((folder) => folder.text !== null) ? "file" : null;
  return panel(`Versions · ${skill.versions.length}`, [
    h("p", { class: "hint" }, "A version is every run given the same skill files, byte for byte. Each rate is that version's alone — runs under another wording are another sample — with its Wilson 95% interval."),
    h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, "version"), h("th", null, "pass rate, by task · harness · model"),
        h("th", { class: "num" }, "runs"), h("th", null, "when"), h("th", null, ""))),
      h("tbody", null, skill.versions.map((version) => h("tr", null,
        h("td", null, h("div", { class: "mono" }, version.fingerprint),
          version.known_as.map((label) => h("div", null, chip(label, label.startsWith("its own file") ? "accent" : "skill")))),
        h("td", null, version.conditions.map((condition) => h("div", { class: "condition-rate" },
          h("span", { class: "muted small" }, [condition.task_id, condition.harness, condition.model].filter(Boolean).join(" · ")),
          condition.passed ? rateBar(condition.passed, "ok") : h("span", { class: "faint small" }, " no verdicts"),
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
        variant.version ? [" · ", chip(`run as version ${variant.version}`, "skill")] : null),
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
