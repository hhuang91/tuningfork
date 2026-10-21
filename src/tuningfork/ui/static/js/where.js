// Where a question holds along a task's runs: one question asked of every run — of what the runs
// wrote, or of the model's answers — and the answers lined up with the verdicts, with the call that
// first did it or made it in each run and the step that names that call in every run.
//
// Where a check prints only its verdict, or none of its findings names a call, this is how to find
// the call that made the difference. A lead to read, never a verdict.

import { api, query } from "./api.js";
import { failure, h, loading, plural, rateBar, replace, short } from "./dom.js";
import { field, input, textOf } from "./forms.js";
import { formatP } from "./results.js";

/** The question's form and, once asked, its answer: `condition` keeps one condition's runs;
 *  `screen`, a screen run to the end, asks its forks instead, arm by arm — where no run of the
 *  task passed, a screen's passing forks are the passing runs there are. */
export function whereForm(task, { condition = null, screen = null } = {}) {
  const result = h("div", { class: "stack" });
  const of = screen ? "fork" : "run";
  const pick = (kind) => {
    const radio = form.querySelector(`[name="where_kind"][value="${kind}"]`);
    if (radio) radio.checked = true;
  };
  const file = input("path", { placeholder: "a file in the workspace: clean.csv, *.json" });
  const contains = input("contains", { placeholder: "a regular expression its text must match; empty: it exists" });
  const atLeast = input("at_least", { type: "number", value: 1, min: 1 });
  const pattern = input("target_match", { placeholder: "a regular expression over what a step did" });
  for (const control of [file, contains, atLeast]) control.addEventListener("input", () => pick("file"));
  pattern.addEventListener("input", () => pick("answer"));

  async function ask(event) {
    event.preventDefault();
    const kind = form.querySelector('[name="where_kind"]:checked')?.value || "file";
    const params = screen ? { screen } : { task };
    if (kind === "answer") {
      params.target_match = textOf(form, "target_match").trim();
    } else {
      params.path = textOf(form, "path").trim();
      params.contains = textOf(form, "contains");
      params.at_least = textOf(form, "at_least");
    }
    replace(result, loading(`Asking every ${of}…`));
    try {
      replace(result, whereResult(await api.get(`/where${query(params)}`), { only: condition, of }));
    } catch (error) {
      replace(result, failure(error, `Could not ask the ${of}s`));
    }
  }

  const form = h("form", { class: "stack", onsubmit: ask },
    h("fieldset", { class: "target-picker" }, h("legend", null, screen ? "The question, asked of every fork, wording by wording" : "The question, asked of every run"),
      h("label", { class: "target-choice" },
        h("input", { type: "radio", name: "where_kind", value: "file", checked: true }),
        h("span", null, h("strong", null, `Of what the ${of}s wrote`),
          h("span", { class: "muted small" }, " — at the start, after each call that changed the files, and at the end"))),
      h("div", { class: "form-grid" },
        field("File", file),
        field("Contains", contains, { help: "^ and $ are each line's ends." }),
        field("At least", atLeast, { help: "Matches that make it hold." })),
      h("label", { class: "target-choice" },
        h("input", { type: "radio", name: "where_kind", value: "answer" }),
        h("span", null, h("strong", null, "Of the model's answers"),
          h("span", { class: "muted small" }, " — tried at every step: a tool call's name and arguments, a script's code, a reply's words"))),
      field("Pattern", pattern, { wide: true })),
    h("div", null, h("button", { class: "btn btn-primary", type: "submit" }, `Ask every ${of}`)),
    result);
  return form;
}

/** Where it held: each condition's runs — or each arm's forks — by verdict, where it was first
 *  seen or made, the step that names that call in each run, and the runs one by one. */
export function whereResult(found, { only = null, of = "run" } = {}) {
  const kind = found.probe?.kind;
  const groups = (found.groups || []).filter((group) => !only || group.condition?.id === only);
  const asked = groups.reduce((total, group) => total + group.runs.length, 0);
  return h("div", { class: "stack" },
    h("p", null, h("strong", null, found.probe?.description || ""), h("span", { class: "muted small" }, `  asked of ${plural(asked, of)}`)),
    groups.length ? groups.map((group) => groupView(group, kind)) : h("p", { class: "muted" }, `No ${of} was asked.`),
    found.skipped?.length ? h("p", { class: "faint small" }, `Passed over: ${found.skipped.slice(0, 3).join("; ")}`) : null,
    h("p", { class: "hint", title: "Fisher's exact test; Wilson score intervals" }, "A lead to read, never a verdict. A split is said to go one way when chance would make one that large less than one time in five (p under 0.2). Intervals are 95%."));
}

function groupView(group, kind) {
  const split = group.split || {};
  const when = kind === "file" ? "at the end" : "at some step";
  const seen = seenWords(group, kind);
  return h("div", { class: "stack", style: { gap: "6px", marginBottom: "12px" } },
    h("div", null, h("strong", null, group.label), h("span", { class: "muted small" }, `  ${group.passed} passed, ${group.failed} failed`)),
    h("div", { class: "group-rates" },
      split.passing ? h("span", null, h("span", { class: "what" }, `held ${when}, of the passing runs`), rateBar(split.passing, "ok")) : null,
      split.failing ? h("span", null, h("span", { class: "what" }, `held ${when}, of the failing runs`), rateBar(split.failing, "bad")) : null),
    h("p", { class: "small" }, splitWords(split)),
    seen ? h("p", { class: "small" }, seen) : null,
    group.reference
      ? h("p", { class: "small" }, "Step ", h("code", null, group.reference.step),
          ` names that call in ${group.reference.names} of the ${plural(group.reference.of, "run")}: a refinement's step.`)
      : null,
    runsTable(group, kind));
}

function splitWords(split) {
  if (split.p === null || split.p === undefined) return "Nothing to compare: that needs passing and failing runs.";
  return split.goes_with
    ? `It goes with ${split.goes_with} (p = ${formatP(split.p)}).`
    : `No split (p = ${formatP(split.p)}).`;
}

function seenWords(group, kind) {
  const seen = group.seen_at || [];
  if (!seen.length) return null;
  const steps = seen.map(([step, count]) => `step ${step} in ${count}`).join(", ");
  if (kind !== "file") return `First seen at ${steps}.`;
  const made = group.made || {};
  let text = `Made by a call in ${made.runs} of ${plural(made.of, "judged run")} (${steps})`;
  if (made.undone) text += `; undone before the end in ${made.undone}`;
  if (made.shared_mistake) text += ` — every run made it, the ${made.passing} that passed too, which undid it before the end`;
  return `${text}.`;
}

function runsTable(group, kind) {
  const yes = (value) => (value === true ? "yes" : value === false ? "no" : "—");
  const head = kind === "file"
    ? ["run", "verdict", "start", "made by", "undone by", "end", "file"]
    : ["run", "verdict", "first", "steps"];
  const rows = group.runs.slice(0, 30).map((run) => {
    const placed = run.step !== null && run.step !== undefined;
    const at = placed ? `${run.step} ${(run.tools || []).join(" ")}`.trim() : "—";
    const cells = kind === "file"
      ? [yes(run.at_start), at, run.undone_by ?? "—", yes(run.at_end),
         run.file ? `${run.file}${run.count !== null && run.count !== undefined ? ` (${run.count})` : ""}` : ""]
      : [at, (run.steps || []).join(", ") || "—"];
    return h("tr", null,
      h("td", null, h("a", { class: "chip tone-muted mono", href: placed ? `#/runs/${run.run}/${run.step}` : `#/runs/${run.run}` }, short(run.run))),
      h("td", null, run.verdict || "no verdict"),
      cells.map((cell) => h("td", { class: "small" }, String(cell))));
  });
  return h("div", { class: "table-wrap" },
    h("table", { class: "table" },
      h("thead", null, h("tr", null, head.map((name) => h("th", null, name)))),
      h("tbody", null, rows)),
    group.runs.length > 30 ? h("p", { class: "faint small" }, `+${group.runs.length - 30} more run(s)`) : null);
}
