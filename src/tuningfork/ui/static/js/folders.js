// Places on disk, shown as what they are: a folder drawn as the tree it is, each line with what
// it is for beside it, and one picture of where things live. A path is never only a sentence.
//
// The names are the glossary's (ops/glossary.py): skill folder, task file, recordings folder,
// run folder, workspace, and a copy of a skill. The Words page draws them in one diagram.
//
// One mark a meaning, as the colour legend in app.css has it: a dashed chip is a copy and nothing
// else; the teal of a change, with a pencil, is what an approval writes; the skill's purple is the
// skill folder and its SKILL.md.

import { chip, copyButton, h } from "./dom.js";

const SVG = "http://www.w3.org/2000/svg";

/** A path's folder and its last name, as the server writes paths (POSIX, no trailing slash). */
export const parentOf = (path) => String(path).replace(/\/+$/, "").replace(/\/[^/]*$/, "") || "/";
export const nameOf = (path) => String(path).replace(/\/+$/, "").replace(/^.*\//, "");

/** Where a path sits: inside the recordings folder, that folder a chip and the rest after it; a
 *  folder's path ends in "/", a file's (`file`) does not. */
export function placeOf(path, home, { file = false } = {}) {
  const text = String(path || "").replace(/\/+$/, "");
  const base = home ? String(home).replace(/\/+$/, "") : "";
  const end = file ? "" : "/";
  if (base && (text === base || text.startsWith(`${base}/`))) {
    const rest = text.slice(base.length + 1);
    return h("span", { class: "place" },
      chip("recordings folder", "muted", base), rest ? h("span", { class: "mono" }, `${rest}${end}`) : null);
  }
  return h("span", { class: "place mono" }, `${text}${end}`);
}

/** A copy of a skill, marked as copies alone are: a dashed chip. */
export const copyChip = (text = "copy of the skill", title = "A copy of a skill's files, kept at one moment") =>
  chip(text, "copy", title);
/** What an approval writes, in the teal of a change, with its pencil. */
export const writes = (text) => h("span", { class: "tree-writes" }, text);
/** The path a run's agent sees a folder at, on a line of its own. */
export const seenAs = (path) => (path ? h("span", { class: "tree-seen" }, "seen as ", h("span", { class: "mono" }, path)) : null);
/** A folder's path as Windows opens it, one click to copy: never a line of its own. */
export function windowsCopy(path) {
  if (!path) return null;
  const button = copyButton(path, "Copy the Windows path");
  button.classList.add("copy-windows");
  button.title = path;
  return button;
}

const chars = (name) => (typeof name === "string" ? name : name?.textContent || "").length;

/** How far a tree's names run, in characters, each line's indent counted (3 a level, as the
 *  stylesheet indents them): trees in one panel share it, so their notes line up in one column
 *  (`folderTree`'s `measure`). */
export function treeMeasure(...lists) {
  let longest = 0;
  const walk = (list, level) => {
    for (const item of list) {
      longest = Math.max(longest, chars(item.name) + level * 3);
      if (item.children?.length) walk(item.children, level + 1);
    }
  };
  for (const list of lists) walk(list, 0);
  return { chars: longest };
}

/**
 * A folder as the tree it is. `root` is where the tree starts: its path, and the recordings
 * folder's path (`home`). Each item is one line:
 *   name      a file's or a folder's name (a folder's ends in "/"), or a node such as a link
 *   note      what it is for: text or nodes, beside the name (`writes` and `seenAs` mark parts)
 *   role      a chip at the note's start: what the line is
 *   mark      the line the page is about, its name in bold
 *   skill     a skill's own text (a SKILL.md), its name in the skill's colour
 *   children  the lines inside a folder
 * The guides are drawn, never typed: a screen reader hears the names, their notes and how the
 * lists nest, not box-drawing characters. `measure` lines its notes up with other trees'.
 */
export function folderTree({ path, home } = {}, items = [], { measure = null } = {}) {
  const sized = measure || treeMeasure(items);
  const line = (item, level) => h("li", null,
    h("div", { class: ["tree-row", item.mark && "is-mark"], style: { "--depth": String(level) } },
      h("span", { class: ["tree-name", item.skill && "is-skill"] }, item.name),
      h("span", { class: "tree-note" }, item.role ?? null, item.note ?? null)),
    item.children?.length ? h("ul", null, item.children.map((child) => line(child, level + 1))) : null);
  return h("div", { class: "folder-tree", style: { "--tree-chars": String(sized.chars) } },
    path ? h("div", { class: "tree-root" }, placeOf(path, home)) : null,
    h("ul", { class: "tree-lines" }, items.map((item) => line(item, 0))));
}

// --- where things live -------------------------------------------------------------------------

function arrow(label, { back = false, down = false } = {}) {
  const head = document.createElementNS(SVG, "svg");
  head.setAttribute("viewBox", "0 0 8 8");
  head.setAttribute("class", "wire-head");
  const tip = document.createElementNS(SVG, "path");
  tip.setAttribute("d", "M0 0L8 4L0 8z");
  head.append(tip);
  return h("div", { class: ["wire", back && "is-back", down && "is-down"] },
    h("span", { class: "wire-label" }, label), h("span", { class: "wire-line" }, h("span", { class: "wire-rule" }), head));
}

/** Arrows between the boxes, in a cell of their own. Decorative: what each says, the box it
 *  points to says in words too, which is what a narrow page and a screen reader get. */
const wires = (area, arrows, { down = false } = {}) => h("div", { class: ["wire-cell", down && "is-down-cell"], style: { gridArea: area }, "aria-hidden": "true" }, arrows);

// A box's lines: each name above what it is for, or beside it where every name is short (`side`).
const rows = (pairs, { side = false } = {}) => h("dl", { class: ["place-rows", side && "is-side"] },
  pairs.map(([name, note, { skill = false, copy = false, swapped = false } = {}]) => [
    h("dt", { class: ["mono", skill && "is-skill"] }, name,
      copy ? copyChip() : null,
      swapped ? chip("swapped in for a SKILL.md", "change", "An A/B or a batch gives its runs this wording in the skill's place") : null),
    h("dd", null, note),
  ]));

/**
 * Where things live: your files, the recordings folder, and what a run's agent sees, side by
 * side, with the copy that goes from each to the next and the one edit that comes back. Each
 * place is a column of its own (`data-place` names it, and each box, for a link to mark).
 * `home` is the recordings folder's path, where known.
 */
export function whereThingsLive({ home } = {}) {
  const band = (column, place) => h("div", { class: ["place-band", `band-${column}`], "data-place": place });
  const head = (area, title, where, sub) => h("div", { class: "place-head", style: { gridArea: area } },
    h("h3", null, title), where ? h("div", { class: "place-where mono" }, where) : null, h("p", null, sub));
  const box = (area, place, title, where, body, className) => h("section", { class: ["place-box", className], style: { gridArea: area }, "data-place": place },
    h("h4", null, title), where ? h("div", { class: "place-where mono" }, where) : null, body);
  return h("div", { class: "where-things-live" },
    h("div", { class: "where-grid" },
      band("mine", "your-files"), band("rec", "recordings-folder"), band("seen", "seen"),
      head("h-mine", "Your files", null, "Wherever you keep them. tuningfork reads them, and writes only an approved edit."),
      box("skill", "skill-folder", "Skill folder", "my-skill/", rows([
        ["SKILL.md", "the skill's text: what you tune", { skill: true }],
        ["…", "any other files the skill uses"],
      ], { side: true }), "is-skill-folder"),
      h("div", { class: "place-beside", style: { gridArea: "beside" } },
        h("p", null, "Beside the skill folder, never inside it:"),
        rows([
          ["my-skill.changelog.md", "what each approval changed, and why"],
          ["my-skill.history/", "the texts approvals replaced"],
        ])),
      box("task", "task-file", "Task file", "evals.json", rows([
        ["prompt", "what each run is asked"],
        ["check", "how a run passes"],
        ["files", "its input files, written in it or beside it"],
        ["skill_dir", "which skill folder"],
      ], { side: true })),
      head("h-rec", "The recordings folder", home || "~/.tuningfork, or the folder --home names", "tuningfork's own: copies and records."),
      box("batch", "copy", "A batch's copy of the skill", "batches/<batch>/skill-snapshot/my-skill/",
        h("p", null, copyChip(), " Your skill folder, copied once as the batch starts. Every run in the batch gets this text, whatever you change meanwhile.")),
      box("run", "run-folder", "Run folders", "batches/<batch>/0/, 1/, 2/ …", [
        rows([
          ["home/", "the harness's home: its settings and logs, with the skill installed from the batch's copy"],
          ["workspace/", "the input files, then what the agent wrote"],
          ["reply.txt", "its final reply"],
        ], { side: true }),
        h("p", { class: "place-aside" }, "A fork's run folder is laid out the same way, under forks/."),
      ], "is-stack"),
      h("details", { class: "place-kept", style: { gridArea: "records" }, "data-place": "kept" },
        h("summary", null, "What else it keeps", h("span", { class: "muted" }, " · 7")),
        rows([
          ["tuningfork.db", "every call, step, run and verdict"],
          ["snapshots.git", "each run's workspace as each call found it, and the skill it was given"],
          ["scenarios/", "task files kept here with their input files, and a copy of the skill where one is kept", { copy: true }],
          ["variants/", "other wordings of a SKILL.md, to try", { swapped: true }],
          ["refinements/", "the text a run had, and the edits proposed to it", { copy: true }],
          ["imports/", "chats brought in, and the skills they showed", { copy: true }],
          ["skills/", "a record of each skill you added, and tuningfork's copy where you asked for one", { copy: true }],
        ])),
      head("h-seen", "What the agent sees", null, "Inside the sandbox, every run alike."),
      box("sandbox", "sandbox", "The sandbox", null, [
        rows([
          ["/home/user", "its home: the run folder's home/"],
          ["/home/user/project", "the run folder's workspace/, where the agent works and the check runs"],
        ]),
        h("p", { class: "place-aside" }, "It sees the rest of the machine read-only, and the recordings folder and the task file's folder empty. Without a sandbox, a run sees its run folder at its real path."),
      ]),
      wires("w-copy", [arrow("copied as a batch starts"), arrow("an approved edit, and nothing else", { back: true })]),
      wires("w-inst", [arrow("installed in each run's home/", { down: true })], { down: true }),
      wires("w-in", [arrow("input files copied in")]),
      wires("w-seen", [arrow("seen as")])));
}
