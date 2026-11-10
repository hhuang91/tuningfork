// A target described in rules, and what a target would count before it is declared.
//
// The rules are the server's (`analysis.distribution.RULES`): a tool called or not, a
// phrase mentioned or not. What a target counts is the server's to say too — the matcher the
// report will use — among the behaviours already seen at the step.

import { api, enc, query } from "./api.js";
import { h, icon, info, outcome, plural, replace } from "./dom.js";
import { LABELS } from "./words.js";

const KINDS = [
  ["calls", "Calls the tool"],
  ["not_calls", "Doesn't call the tool"],
  ["contains", "Mentions"],
  ["not_contains", "Doesn't mention"],
];
let lists = 0;

/** Rules that must all hold. `read()` gives them as `target_rules`; `setTools(names)`
 *  offers the step's tools for the rules that name one. `onUse` is called whenever a
 *  person touches them, to choose this way of declaring the target. */
export function describeBuilder({ onUse = () => {} } = {}) {
  const listId = `tf-tools-${(lists += 1)}`;
  const tools = h("datalist", { id: listId });
  const rows = h("div", { class: "rules" });
  const changed = () => element.dispatchEvent(new Event("input", { bubbles: true }));
  const element = h("div", { class: "rule-builder", onfocusin: onUse, oninput: onUse },
    rows,
    h("div", { class: "rule-more" },
      h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => { add("contains"); onUse(); changed(); } }, "+ Add a rule"),
      h("span", { class: "form-help" }, "All must hold."),
      info("Every rule must hold. A phrase is looked for, ignoring case, in the reply and in each tool call's name and arguments.", { label: "How the rules are read" })),
    tools);
  add("calls");
  return { element, read, setTools };

  function add(kind) {
    const value = h("input", { class: "field", type: "text", "aria-label": "Value" });
    const which = h("select", { class: "field", "aria-label": "Kind of rule", onchange: () => shape() },
      KINDS.map(([key, label]) => h("option", { value: key, selected: key === kind }, label)));
    // The value and its × go to a line of their own together where the row is narrow.
    const row = h("div", { class: "rule" }, which, h("span", { class: "rule-value" }, value,
      h("button", {
        class: "btn btn-quiet btn-small", type: "button", title: "Remove this rule", "aria-label": "Remove this rule",
        onclick: () => {
          row.remove();
          if (!rows.children.length) add("calls");
          changed();
        },
      }, "×")));
    function shape() {
      const tool = which.value === "calls" || which.value === "not_calls";
      if (tool) value.setAttribute("list", listId);
      else value.removeAttribute("list");
      value.placeholder = tool ? "a tool's name, e.g. read_file" : "a phrase, e.g. row_count";
    }
    shape();
    rows.append(row);
  }

  function read() {
    const rules = [...rows.querySelectorAll(".rule")]
      .map((row) => ({ if: row.querySelector("select").value, value: row.querySelector("input").value.trim() }))
      .filter((rule) => rule.value);
    if (!rules.length) {
      throw Object.assign(new Error("Tools & words: give at least one rule a value."), { field: "target_rules" });
    }
    return rules;
  }

  function setTools(names) {
    replace(tools, (names || []).map((name) => h("option", { value: name })));
  }
}

/** What a target would count among the behaviours seen at a step. `update(target)` takes
 *  `{base}`, `{target_rules}` or `{target_match}`, with the lens's `normalizers`; `onTools`
 *  hears which tools the step offered. */
export function seenPreview({ run, idx, onTools = () => {} }) {
  const element = h("div", { class: "seen-preview", "aria-live": "polite" });
  let asked = 0;
  let timer = null;
  return { element, update };

  function update(target) {
    clearTimeout(timer);
    timer = setTimeout(() => look(target), 300);
  }

  async function look(target) {
    const mine = (asked += 1);
    const params = {};
    if (target.base) params.base = "1";
    if (target.target_match) params.target_match = target.target_match;
    if (target.target_rules) params.target_rules = JSON.stringify(target.target_rules);
    if (target.normalizers?.length) params.normalizers = target.normalizers.join(",");
    try {
      const seen = await api.get(`/runs/${enc(run)}/steps/${idx}/behaviours${query(params)}`);
      if (mine !== asked) return;
      onTools(seen.tools);
      replace(element, draw(seen));
    } catch (error) {
      if (mine !== asked) return;
      replace(element, h("p", { class: "form-error small" }, error.message));
    }
  }

  function draw(seen) {
    if (!seen.behaviours.length) return h("p", { class: "muted small" }, "Nothing has been seen at this step yet, so there is nothing to try this on.");
    const total = seen.behaviours.reduce((sum, found) => sum + found.count, 0);
    const counted = seen.behaviours.filter((found) => found.matches).reduce((sum, found) => sum + found.count, 0);
    const row = (found) =>
      h("li", { class: ["seen", found.matches === true && "seen-yes", found.matches === false && "seen-no"] },
        h("span", { class: "seen-mark", title: found.matches === null ? "" : found.matches ? LABELS.matches : LABELS.noMatch },
          found.matches === null ? "·" : icon(found.matches ? "check" : "x")),
        outcome(found),
        h("span", { class: "muted small" }, Object.entries(found.seen).map(([where, n]) => `${where} ${n}`).join(" · ") + fared(found)));
    const parts = [
      h("div", { class: "label seen-title" }, seen.target === null
        ? `Seen at this step: ${plural(total, "answer")}`
        : `Matches ${counted} of ${total} answers seen at this step`,
        info("These are earlier answers, shown so you can see what your choice would match. They are not the samples the test will draw.", { label: "What these are" })),
      h("ul", { class: "seen-list" }, grouped(seen.behaviours).map((item) => (item.replies ? repliesRow(item, row) : row(item)))),
    ];
    const judged = split(seen);
    if (judged) parts.push(judged);
    return h("div", { class: "stack", style: { gap: "var(--space-1)" } }, ...parts);
  }

  // Replies that differ only in their words, and match alike, as one line: a step where the
  // model mostly replies has a wording for nearly every answer, and a list of them hides the
  // few behaviours that differ. Three or more of a kind fold; opened, each wording is there.
  function grouped(behaviours) {
    const byMatch = new Map();
    for (const found of behaviours) {
      if (found.kind !== "text") continue;
      const key = String(found.matches);
      if (!byMatch.has(key)) byMatch.set(key, []);
      byMatch.get(key).push(found);
    }
    const out = [];
    const folded = new Set();
    for (const found of behaviours) {
      const alike = found.kind === "text" ? byMatch.get(String(found.matches)) : null;
      if (!alike || alike.length < 3) {
        out.push(found);
      } else if (!folded.has(alike)) {
        folded.add(alike);
        out.push({ replies: alike, matches: found.matches });
      }
    }
    return out;
  }

  function repliesRow(item, row) {
    const count = item.replies.reduce((sum, found) => sum + found.count, 0);
    const passed = item.replies.reduce((sum, found) => sum + (found.verdicts?.passed || 0), 0);
    const failed = item.replies.reduce((sum, found) => sum + (found.verdicts?.failed || 0), 0);
    return h("li", { class: ["seen", "seen-replies", item.matches === true && "seen-yes", item.matches === false && "seen-no"] },
      h("details", { class: "fold" },
        h("summary", null,
          h("span", { class: "seen-mark", title: item.matches === null ? "" : item.matches ? LABELS.matches : LABELS.noMatch },
            item.matches === null ? "·" : icon(item.matches ? "check" : "x")),
          outcome({ kind: "text", label: `${count} ${count === 1 ? "reply" : "replies"}, ${plural(item.replies.length, "phrasing")}` }),
          passed || failed ? h("span", { class: "muted small" }, `${passed} passed, ${failed} failed`) : null),
        h("ul", { class: "seen-list" }, item.replies.map(row))));
  }

  // How the recorded runs that took a behaviour fared: a target should count the passing runs'.
  function fared(found) {
    const verdicts = found.verdicts || {};
    const passed = verdicts.passed || 0, failed = verdicts.failed || 0;
    return passed || failed ? ` · ${passed} passed, ${failed} failed` : "";
  }

  function split(seen) {
    const found = seen.split;
    if (!found || !found.passing || !found.failing || !(found.passing.of || found.failing.of)) return null;
    const words = found.kind === "universal" ? "every answer seen satisfies it: an edit could only lower it"
      : found.kind === "unseen" ? "no answer seen satisfies it"
      : found.separates === true ? "it separates them" : found.separates === false ? "it does not go with passing" : "";
    return h("p", { class: "muted small" },
      `Matches the answers of ${found.passing.counted} of ${found.passing.of} passing runs and ${found.failing.counted} of ${found.failing.of} failing runs`
      + (words ? ` — ${words}` : ""));
  }
}
