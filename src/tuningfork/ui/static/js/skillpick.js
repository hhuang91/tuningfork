// Which skill a scenario runs, wherever one is made: a skill the Skills page lists, by its folder
// there, or a folder named by its path — and, making one of a run, a copy of what the run had,
// kept in the scenario's own folder. Where the run says which folder it is (its task's own, or
// the one named when its chat was imported), the folder is shown, greyed, and not chosen. The
// scenario's editor and Make a scenario choose it with this one picker, as they choose a check
// with `checkPicker`.

import { h, replace } from "./dom.js";
import { input } from "./forms.js";

let pickers = 0;

/** A path taken as it is typed: from the root, a home (~), or a Windows drive or share. Any
 *  other is relative, and taken from the folder a picker shows before it. */
export const fullPath = (text) => /^\s*["']?\s*(?:[\\/]|~|[A-Za-z]:[\\/])/.test(text || "");

/** The skill's name, and what it is not, where a person asks (found in use: does it have to
 *  match the folder's name, or is it the name the model is offered?): the scenario's editor and
 *  Make a scenario label it alike. */
export const skillNameLabel = () => h("span", {
  class: "has-tip",
  title: "The model never sees it: the harness shows the skill by its folder's name and its SKILL.md. It need not match the "
    + "folder's name, but a task that stops where the model loads another skill tells the skill it studies by this name.",
}, "Skill name");

/** A folder's name as the Skills page names the skill in it: lowercase, dashes for the rest
 *  (`ops.skills._clean`). */
export function skillNameOf(path) {
  const last = String(path || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "";
  return last.toLowerCase().replace(/[^a-z0-9_.-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
}

/**
 * skills    the Skills page's list (`GET /skills`): each folder it knows is offered
 * path      the folder as the scenario names it now ("" for none); `resolved`, as tuningfork
 *           opens it — what is matched against the skills listed
 * relative  a path may be relative, taken from the scenario's own folder — shown before what is
 *           typed while that is not a full path, once `setBase` has said where it is
 * copy      a copy of what the run had can be kept: `{ where }`, the folder it is kept in. The
 *           choice to begin with, then
 * fixed     `{ path, why }`: the run says which folder it is — shown, greyed, nothing to choose
 * onChange  `({ kind, path, skill })` for each change: "known", "path" or "copy"; `skill` the
 *           name a listed skill goes by
 * The element, `value()`, `note` — under the field, for what the path resolves to — `setBase(folder)`,
 * and `setPath(text)`, for a path the scenario's file now words otherwise (a Windows path, saved
 * as tuningfork opens it).
 */
export function skillPicker({ skills = [], path = "", resolved = null, relative = false, copy = null, fixed = null,
  editable = true, onChange = () => {}, name = `s${pickers++}` }) {
  const known = skills.flatMap((skill) => (skill.folders || []).map((folder) => ({ skill: skill.name, folder })));
  // A relative path is a folder in the scenario's own, and shows as the file words it: the
  // scenario's folder before it, not the full path the Skills page lists it by.
  const listed = !path || fullPath(path) ? known.find((each) => each.folder === resolved || each.folder === path) : null;
  let kind = fixed ? "fixed" : listed ? "known" : path ? "path" : copy ? "copy" : known.length ? "known" : "path";
  let chosen = listed || null;
  let typed = listed ? "" : path || "";
  let base = null;
  const note = h("span", { class: "path-note" });
  const body = h("div", { class: "skill-pick-body" });
  const help = h("span", { class: "form-help" });

  const value = () => (kind === "known" ? { kind, path: chosen?.folder || "", skill: chosen?.skill || null }
    : kind === "path" ? { kind, path: typed.trim(), skill: null }
      : kind === "fixed" ? { kind, path: fixed.path, skill: null } : { kind, path: "", skill: null });
  const changed = () => onChange(value());

  // The folder a relative path is taken from, before the field: hidden while a full path is typed.
  const baseText = h("bdi", { class: "path-base-text" });
  const prefix = h("span", { class: "path-base", hidden: true }, baseText);
  const field = input(`skill_dir-${name}`, {
    value: typed, disabled: !editable,
    placeholder: relative ? "a folder in it, or a full path" : "e.g. C:\\Users\\you\\skills\\my-skill, or /home/you/skills/my-skill",
    "aria-label": "Path to the skill's folder",
  });
  const showPrefix = () => {
    prefix.hidden = !relative || !base || fullPath(field.value);
  };
  const setBase = (folder) => {
    base = folder ? `${String(folder).replace(/[\\/]+$/, "")}/` : null;
    prefix.title = base ? `A path that does not start at / (or C:\\) is taken from this scenario's folder: ${base}` : "";
    replace(baseText, base || "");
    showPrefix();
  };
  field.addEventListener("input", () => {
    typed = field.value;
    showPrefix();
    changed();
  });

  const select = h("select", { class: "field", name: `skill_known-${name}`, disabled: !editable, "aria-label": "Existing skill" },
    h("option", { value: "", selected: !chosen }, known.length ? "— choose a skill —" : "— none listed with a folder —"),
    known.map((each, index) => h("option", { value: String(index), selected: each === chosen }, `${each.skill} — ${each.folder}`)));
  select.addEventListener("change", () => {
    chosen = select.value === "" ? null : known[Number(select.value)];
    changed();
  });

  const choices = [
    ...(copy ? [["copy", "A copy of what the run had"]] : []),
    ["known", "Existing skill"],
    ["path", "Path to skill"],
  ];
  const radios = fixed ? null : h("div", { class: "controls", role: "radiogroup", "aria-label": "Skill folder" }, choices.map(([choice, label]) =>
    h("label", { class: "check" },
      h("input", {
        type: "radio", name: `skill-from-${name}`, value: choice, checked: kind === choice, disabled: !editable,
        onchange: () => {
          kind = choice;
          draw();
          changed();
        },
      }), label)));

  function draw() {
    if (kind === "fixed") {
      // Greyed and not to be changed, as the fields the run says are; its path can still be copied.
      replace(body, h("input", { class: "field field-fixed", type: "text", name: `skill_dir-${name}`, value: fixed.path,
        readonly: true, "aria-readonly": "true", "aria-label": "Skill folder", spellcheck: "false" }));
      replace(help, fixed.why);
      return;
    }
    if (kind === "copy") {
      replace(body);
      replace(help, "The copy the run kept of the skill's folder, else the SKILL.md its model read, kept in the scenario's own "
        + `folder (${copy.where}): an approved edit is written to the copy, never back to where the run had it.`);
      return;
    }
    if (kind === "known") {
      replace(body, select);
      replace(help, "The skills the Skills page lists, each by a folder it knows: ",
        h("a", { href: "#/add-skill" }, "Add a skill…"), " there to list another. Each run gets a copy of the folder, "
        + "and an approved edit is written to it.");
      return;
    }
    replace(body, h("div", { class: "path-input" }, prefix, field));
    replace(help, relative
      ? "The folder holding the skill's SKILL.md: a folder in this scenario's own, shown before it, or a full path (/… "
        + "or C:\\…), taken as it is. Each run gets a copy, and an approved edit is written to it."
      : "The folder holding the skill's SKILL.md on this machine, by its full path. Each run gets a copy, and an approved "
        + "edit is written to it.");
  }
  draw();

  return {
    element: h("div", { class: "skill-pick stack" }, radios, body, help, note),
    value,
    note,
    setBase,
    setPath: (text) => {
      if (kind !== "path") return;
      typed = text;
      field.value = text;
      showPrefix();
    },
  };
}
