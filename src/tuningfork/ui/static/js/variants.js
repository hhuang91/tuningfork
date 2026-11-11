// A wording to test, edited in the page, and what the A/B is about.
//
// The tuning loop is: read the skill as the model saw it, change a sentence, test it. The
// editor starts from the skill file as the run was given it. A variant is saved in the
// recordings folder only when its A/B starts — a draft left here leaves nothing behind —
// and never inside the skill's own folder, where every later run could read it.
//
// The target is declared before any sample is drawn, and nothing is chosen for you: a
// target picked after seeing the samples is a result fished for.

import { api, enc } from "./api.js";
import { h, info, outcome, rateBar, short } from "./dom.js";
import { field, input, textArea, textOf, ways } from "./forms.js";
import { describeBuilder, seenPreview } from "./targets.js";
import { LABELS } from "./words.js";

/** The first `v<n>` no saved variant has, nor an A/B not yet started will save (`pending`). */
function suggestedName(saved, pending = []) {
  const taken = new Set([...saved.map((variant) => variant.name), ...pending]);
  let n = 1;
  while (taken.has(`v${n}`)) n += 1;
  return `v${n}`;
}

/** Lines added or removed, roughly: enough to say whether, and how much, it was edited. */
function changedLines(before, after) {
  const count = new Map();
  for (const line of before.split("\n")) count.set(line, (count.get(line) || 0) + 1);
  let added = 0;
  for (const line of after.split("\n")) {
    const left = count.get(line) || 0;
    if (left) count.set(line, left - 1);
    else added += 1;
  }
  const removed = [...count.values()].reduce((sum, left) => sum + left, 0);
  return Math.max(added, removed);
}

/**
 * skill    the run's skill file (`GET /runs/{run}/skill`): its text is the baseline
 * saved    variants already saved for that skill (`GET /variants?skill=`)
 * pending  the names A/Bs queued or running will save their texts under (`GET /jobs`)
 */
export function variantEditor(skill, saved, pending = []) {
  const baseline = skill.text ?? "";
  const text = textArea("variant_text", baseline, { rows: 18 });
  text.setAttribute("aria-label", "The variant");
  const name = input("variant_name", {
    value: suggestedName(saved, pending),
    title: "A name holds one text: an edit of a saved variant is given a new name, until you type one.",
  });
  const original = skill.text === null || skill.text === undefined
    ? input("original", { placeholder: "/path/to/the/SKILL.md the run was given" })
    : null;
  const changes = h("span", { class: "muted small" });

  const count = () => {
    const lines = changedLines(baseline, text.value);
    changes.textContent = lines ? `${lines} line${lines === 1 ? "" : "s"} changed` : "unchanged: edit it to have something to test";
  };
  // A saved variant's name holds its text alone, and is refused for another: the one "Start
  // from" loaded keeps its name while the box holds its text, and an edit of it is given a name
  // of its own, until the person writes one.
  let loaded = null; // its name, and its text as the box reads it
  let named = false;
  const rename = () => {
    if (!named) name.value = loaded && text.value === loaded.text ? loaded.name : suggestedName(saved, pending);
  };
  text.addEventListener("input", () => {
    count();
    rename();
  });
  name.addEventListener("input", () => {
    named = true;
  });
  count();

  const from = saved.length
    ? h("select", {
        class: "field",
        "aria-label": "Start from",
        onchange: async (event) => {
          const chosen = event.target.value;
          if (!chosen) {
            text.value = baseline;
            loaded = null;
          } else {
            const variant = await api.get(`/variants/${enc(skill.skill)}/${enc(chosen)}`);
            text.value = variant.text;
            loaded = { name: variant.name, text: text.value };
          }
          named = false;
          text.dispatchEvent(new Event("input", { bubbles: true })); // named, counted, and the form checks it again
        },
      },
      h("option", { value: "" }, "the wording the run was given"),
      saved.map((variant) => h("option", { value: variant.name }, `${variant.name} (saved)`)))
    : null;

  return h("div", { class: "variant-editor" },
    original
      ? h("div", { class: "banner banner-warn" }, h("strong", null, "This run kept no copy of its skill file. "),
          "Name the original — the SKILL.md as the run was given it, on the proxy's machine — and write the variant in full below.")
      : h("p", { class: "hint" }, "Edit the skill below. The A/B swaps your text in for the original.",
          info(["The skill as this run was given it", skill.path ? [": ", h("span", { class: "mono" }, skill.path)] : "",
            ". Your text goes in wherever the original sits in the context."], { label: "Which text this is" })),
    original ? field("Original skill file", original, { wide: true }) : null,
    from ? field("Start from", from) : null,
    h("div", { class: "editor-head" }, h("span", { class: "label" }, "The variant"), changes),
    text,
    field("Save it as", name, {
      help: `Kept as variants/${skill.skill}/<name>.md in the recordings folder, once the A/B starts.`,
    }));
}

export function readVariant(form) {
  const params = { variant_text: textOf(form, "variant_text"), variant_name: textOf(form, "variant_name").trim() };
  const original = form.querySelector('[name="original"]');
  if (original) {
    if (!original.value.trim()) {
      throw Object.assign(new Error("Name the original skill file: the run kept no copy of it."), { field: "original" });
    }
    params.original = original.value.trim();
  }
  return params;
}

/** targets: `GET /runs/{run}/steps/{k}/targets`. A refinement always has a target:
 *  `exploratory: false` leaves that choice out. */
export function targetPicker(targets, {
  run = null, idx = null, exploratory = true,
  legend = LABELS.condition,
  about = "What the model must do to be counted. You choose it before anything is sampled: a condition picked after seeing the samples is a result fished for.",
  more = [],
} = {}) {
  const clusters = targets.behaviours?.clusters || [];
  const resample = targets.resample;
  // What a person touches says which way they mean: a behaviour ticked, a rule or a pattern typed.
  const pick = (kind) => form.choose(kind);
  const builder = describeBuilder({ onUse: () => pick("describe") });
  const seen = run ? seenPreview({ run, idx, onTools: builder.setTools }) : null;

  const letters = clusters.length
    ? h("div", { class: "target-letters" }, clusters.map((cluster) =>
        h("label", { class: "target-letter" },
          h("input", { type: "radio", name: "target", value: cluster.letter, onchange: () => pick("letter") }),
          h("span", { class: "letter" }, cluster.letter),
          h("span", { class: "target-outcome" }, outcome(cluster.outcome)),
          rateBar(cluster.rate))))
    : null;
  const pattern = input("target_match", { placeholder: "row_count" });
  pattern.addEventListener("input", () => pick("pattern"));

  // The four ways to say it, each a word or two: what each is, said under them once chosen.
  const form = ways("target_kind", [
    {
      value: "letter", label: "Seen here", disabled: !clusters.length, body: letters,
      note: resample
        ? `One of the behaviours this step's latest resample found (${short(resample.id)}, ${targets.behaviours.total} samples${targets.normalizers.length ? `, grouped by ${targets.normalizers.join(", ")}` : ""}).`
        : "No resample here yet: resample this step first, to pick a behaviour by its letter.",
    },
    { value: "describe", label: "Tools & words", note: "The tools the model calls and the words it says.", body: builder.element },
    { value: "pattern", label: "Pattern", note: "A regular expression over a reply's words, or a call's name and arguments.", body: h("div", { class: "target-pattern" }, pattern) },
    ...more,
    exploratory ? { value: "exploratory", label: "None", note: "Every behaviour is compared, and each p adjusted for that." } : null,
  ], { legend, about });
  if (seen) form.append(seen.element);
  form.describe = builder;
  if (seen) {
    // What a described target or a pattern would count, among what this step has done.
    const look = () => {
      const kind = form.querySelector('[name="target_kind"]:checked')?.value;
      const regex = textOf(form, "target_match").trim();
      let rules = null;
      try {
        rules = kind === "describe" ? builder.read() : null;
      } catch {
        rules = null;
      }
      seen.update(rules ? { target_rules: rules, normalizers: targets.normalizers }
        : kind === "pattern" && regex ? { target_match: regex, normalizers: targets.normalizers }
        : { normalizers: targets.normalizers });
    };
    form.addEventListener("input", look);
    form.addEventListener("change", look);
    look();
  }
  return form;
}

export function readTarget(form) {
  const kind = form.querySelector('[name="target_kind"]:checked')?.value;
  if (!kind) {
    throw Object.assign(new Error("Choose the condition before sampling: Seen here, Tools & words, Pattern or None."), { field: "target_kind" });
  }
  if (kind === "letter") {
    const letter = form.querySelector('[name="target"]:checked')?.value;
    if (!letter) throw Object.assign(new Error("Choose the behaviour to count."), { field: "target" });
    return { target: letter };
  }
  if (kind === "pattern") {
    const regex = textOf(form, "target_match").trim();
    if (!regex) throw Object.assign(new Error("Write the pattern to count."), { field: "target_match" });
    return { target_match: regex };
  }
  if (kind === "describe") {
    return { target_rules: form.querySelector(".target-picker").describe.read() };
  }
  return { exploratory: true };
}
