// Markdown as a page shows it — a skill's SKILL.md, a changelog — built as DOM nodes.
//
// Never parsed as HTML: every piece of text reaches the page as a text node, so a skill full
// of markup-looking text stays text (the rule `dom.js` keeps, which `tests/test_ui.py`
// enforces). An HTML block is shown as the text it is. A link opens only an http(s) or mailto
// address; any other — a path inside the skill's folder — is named, not followed. Images are
// named, never loaded.
//
// The subset a skill is written in: frontmatter, headings, paragraphs, emphasis, code, lists
// (nested, numbered, tasks), block quotes, tables, rules.

import { append, h, icon } from "./dom.js";

const FENCE = /^(\s{0,3})(`{3,}|~{3,})\s*([^`\s]*)/;
const HEADING = /^\s{0,3}(#{1,6})(?:\s+(.*?))?\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const QUOTE = /^\s{0,3}>/;
const LIST = /^(\s*)([-*+]|\d{1,9}[.)])(?:\s+(.*))?$/;
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$/;
const SETEXT = /^\s{0,3}(=+|-+)\s*$/;

/** `text` as Markdown, in an element of its own. Frontmatter is shown as its fields. */
export function markdownView(text, { className = "markdown" } = {}) {
  let lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  const root = h("div", { class: className });
  if (lines[0] === "---") {
    const end = lines.findIndex((line, index) => index > 0 && (line === "---" || line === "..."));
    if (end > 0) {
      root.append(frontmatterView(lines.slice(1, end)));
      lines = lines.slice(end + 1);
    }
  }
  return append(root, blocks(lines));
}

function frontmatterView(lines) {
  const fields = [];
  for (const line of lines) {
    const field = /^([A-Za-z0-9_.-]+):\s?(.*)$/.exec(line);
    if (field) fields.push([field[1], field[2]]);
    else if (fields.length && (line.startsWith(" ") || line.startsWith("\t") || !line.trim())) {
      fields[fields.length - 1][1] += `\n${line.trim()}`;
    } else {
      return h("pre", { class: "md-front" }, lines.join("\n")); // not simple key: value lines
    }
  }
  return h("dl", { class: "md-front", title: "Frontmatter: how a harness chooses the skill" },
    fields.map(([key, value]) => [h("dt", null, key), h("dd", null, value.trim())]));
}

const indentOf = (line) => line.length - line.trimStart().length;

function startsBlock(line) {
  return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || LIST.test(line);
}

function blocks(lines) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    let found = FENCE.exec(line);
    if (found) {
      const [, indent, fence, lang] = found;
      const closing = new RegExp(`^\\s{0,3}${fence[0] === "`" ? "`" : "~"}{${fence.length},}\\s*$`);
      const body = [];
      i += 1;
      while (i < lines.length && !closing.test(lines[i])) {
        body.push(lines[i].slice(Math.min(indent.length, indentOf(lines[i]))));
        i += 1;
      }
      i += 1; // the closing fence, or the end
      out.push(h("pre", { class: "md-code" }, h("code", { "data-lang": lang || null }, body.join("\n"))));
      continue;
    }
    found = HEADING.exec(line);
    if (found) {
      out.push(h(`h${found[1].length}`, { class: "md-h" }, inline(found[2] || "")));
      i += 1;
      continue;
    }
    if (RULE.test(line)) {
      out.push(h("hr"));
      i += 1;
      continue;
    }
    if (QUOTE.test(line)) {
      const body = [];
      while (i < lines.length && lines[i].trim() && (QUOTE.test(lines[i]) || !startsBlock(lines[i]))) {
        body.push(lines[i].replace(/^\s{0,3}>\s?/, ""));
        i += 1;
      }
      out.push(h("blockquote", null, blocks(body)));
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const [table, next] = tableView(lines, i);
      out.push(table);
      i = next;
      continue;
    }
    if (LIST.test(line)) {
      const [list, next] = listView(lines, i);
      out.push(list);
      i = next;
      continue;
    }
    // A paragraph: up to a blank line, a line that starts another block, or a setext rule.
    const body = [line];
    i += 1;
    let level = 0;
    while (i < lines.length && lines[i].trim()) {
      const setext = SETEXT.exec(lines[i]);
      if (setext) {
        level = setext[1][0] === "=" ? 1 : 2;
        i += 1;
        break;
      }
      if (startsBlock(lines[i])) break;
      body.push(lines[i]);
      i += 1;
    }
    out.push(level ? h(`h${level}`, { class: "md-h" }, inline(body.join(" ").trim())) : h("p", null, paragraph(body)));
  }
  return out;
}

/** A paragraph's lines: joined, but a line ending in two spaces or a backslash breaks. */
function paragraph(lines) {
  const out = [];
  lines.forEach((line, index) => {
    const last = index === lines.length - 1;
    const hard = !last && (/ {2,}$/.test(line) || /\\$/.test(line));
    out.push(inline(hard ? line.replace(/(\s+|\\)$/, "").trim() : line.trim()));
    if (!last) out.push(hard ? h("br") : " ");
  });
  return out;
}

function listView(lines, start) {
  const first = LIST.exec(lines[start]);
  const indent = first[1].length;
  const ordered = /\d/.test(first[2]);
  const items = [];
  let i = start;
  while (i < lines.length) {
    const marker = LIST.exec(lines[i]);
    if (!marker || marker[1].length !== indent || /\d/.test(marker[2]) !== ordered) break;
    // The item's text starts after its marker; lines indented that far belong to it.
    const content = marker[1].length + marker[2].length + 1;
    const body = [marker[3] || ""];
    let loose = false;
    i += 1;
    while (i < lines.length) {
      const next = lines[i];
      if (!next.trim()) {
        const ahead = lines.slice(i + 1).find((each) => each.trim());
        if (ahead !== undefined && indentOf(ahead) >= content) {
          body.push("");
          loose = true;
          i += 1;
          continue;
        }
        break;
      }
      const nested = LIST.exec(next);
      if (nested && nested[1].length <= indent) break; // the next item, or a list further out
      if (indentOf(next) >= content || nested) {
        body.push(next.slice(Math.min(content, indentOf(next))));
      } else if (!startsBlock(next)) {
        body.push(next.trim()); // a paragraph running on without its indent
      } else {
        break;
      }
      i += 1;
    }
    items.push(itemView(body, loose));
    // Blank lines between two items of one list keep it one list.
    let ahead = i;
    while (ahead < lines.length && !lines[ahead].trim()) ahead += 1;
    const sibling = ahead < lines.length ? LIST.exec(lines[ahead]) : null;
    if (ahead > i && sibling && sibling[1].length === indent && /\d/.test(sibling[2]) === ordered) i = ahead;
  }
  const number = ordered ? parseInt(first[2], 10) : 1;
  return [h(ordered ? "ol" : "ul", { class: "md-list", start: ordered && number !== 1 ? String(number) : null }, items), i];
}

function itemView(body, loose) {
  const task = /^\[([ xX])\]\s+/.exec(body[0]);
  if (task) body = [body[0].slice(task[0].length), ...body.slice(1)];
  const children = blocks(body);
  // A tight item's first paragraph is its text, not a paragraph of its own.
  if (!loose && children[0]?.tagName === "P") children.splice(0, 1, ...children[0].childNodes);
  return h("li", { class: task ? "md-task" : null },
    task ? h("span", { class: "md-box", role: "img", "aria-label": task[1] === " " ? "not done" : "done" }, icon(task[1] === " " ? "square" : "square-check")) : null,
    children);
}

function tableView(lines, start) {
  const cells = (row) => row.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "").split(/(?<!\\)\|/)
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));
  const head = cells(lines[start]);
  const aligns = cells(lines[start + 1]).map((rule) =>
    rule.startsWith(":") && rule.endsWith(":") ? "center" : rule.endsWith(":") ? "right" : rule.startsWith(":") ? "left" : null);
  const rows = [];
  let i = start + 2;
  while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
    rows.push(cells(lines[i]));
    i += 1;
  }
  const cell = (tag, text, column) => h(tag, aligns[column] ? { style: { textAlign: aligns[column] } } : null, inline(text));
  return [h("div", { class: "table-wrap" }, h("table", { class: "table md-table" },
    h("thead", null, h("tr", null, head.map((text, column) => cell("th", text, column)))),
    h("tbody", null, rows.map((row) => h("tr", null, head.map((_, column) => cell("td", row[column] ?? "", column))))))), i];
}

// --- inline -----------------------------------------------------------------------------------

const ESCAPABLE = /[\\`*_{}[\]()#+\-.!|~<>]/;

function link(label, url, title) {
  if (/^(https?:|mailto:)/i.test(url)) {
    return h("a", { href: url, target: "_blank", rel: "noopener noreferrer", title: title || url }, inline(label));
  }
  // A path in the skill's folder, or an anchor: named, not followed.
  return h("span", { class: "md-link", title: url }, inline(label));
}

/** Emphasis, code, links and breaks within a line, as text nodes and elements. */
export function inline(text) {
  const out = [];
  let buffer = "";
  const flush = () => {
    if (buffer) out.push(buffer);
    buffer = "";
  };
  let i = 0;
  while (i < text.length) {
    const char = text[i];
    if (!"\\`!*_~[<".includes(char)) {
      buffer += char;
      i += 1;
      continue;
    }
    const rest = text.slice(i);
    let found;
    if (char === "\\" && ESCAPABLE.test(rest[1] || "")) {
      buffer += rest[1];
      i += 2;
      continue;
    }
    if ((found = /^(`+)([\s\S]*?[^`])\1(?!`)/.exec(rest)) || (found = /^(`+)(`)\1/.exec(rest))) {
      flush();
      const code = found[2];
      out.push(h("code", null, /^ .* $/.test(code) && code.trim() ? code.slice(1, -1) : code));
      i += found[0].length;
      continue;
    }
    if ((found = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/.exec(rest))) {
      flush();
      out.push(h("span", { class: "md-image", title: found[2] }, `[image: ${found[1] || found[2]}]`));
      i += found[0].length;
      continue;
    }
    if ((found = /^\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/.exec(rest))) {
      flush();
      out.push(link(found[1], found[2], found[3]));
      i += found[0].length;
      continue;
    }
    if ((found = /^<((?:https?:\/\/|mailto:)[^\s>]+)>/.exec(rest))) {
      flush();
      out.push(link(found[1], found[1]));
      i += found[0].length;
      continue;
    }
    if ((found = /^(\*\*|__)(?=\S)([\s\S]*?\S)\1/.exec(rest)) && !(found[1] === "__" && /\w/.test(text[i - 1] || ""))) {
      flush();
      out.push(h("strong", null, inline(found[2])));
      i += found[0].length;
      continue;
    }
    if ((found = /^~~(?=\S)([\s\S]*?\S)~~/.exec(rest))) {
      flush();
      out.push(h("del", null, inline(found[1])));
      i += found[0].length;
      continue;
    }
    // Emphasis: `*` anywhere; `_` only at a word's edges, so snake_case stays as written.
    if ((found = /^\*(?=\S)([\s\S]*?[^\s*])\*(?!\*)/.exec(rest))
        || (!/\w/.test(text[i - 1] || "") && (found = /^_(?=\S)([\s\S]*?[^\s_])_(?!\w)/.exec(rest)))) {
      flush();
      out.push(h("em", null, inline(found[1])));
      i += found[0].length;
      continue;
    }
    buffer += char;
    i += 1;
  }
  flush();
  return out;
}
