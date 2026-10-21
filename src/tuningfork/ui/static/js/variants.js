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
import { h, outcome, rateBar, short } from "./dom.js";
import { field, input, textArea, textOf } from "./forms.js";
import { describeBuilder, seenPreview } from "./targets.js";
import { LABELS } from "./words.js";

function suggestedName(saved) {
  const taken = new Set(saved.map((variant) => variant.name));
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
 * skill  the run's skill file (`GET /runs/{run}/skill`): its text is the baseline
 * saved  variants already saved for that skill (`GET /variants?skill=`)
 */
export function variantEditor(skill, saved) {
  const baseline = skill.text ?? "";
  const text = textArea("variant_text", baseline, { rows: 18 });
  const name = input("variant_name", { value: suggestedName(saved) });
  const original = skill.text === null || skill.text === undefined
    ? input("original", { placeholder: "/path/to/the/SKILL.md the run was given" })
    : null;
  const changes = h("span", { class: "muted small" });

  const count = () => {
    const lines = changedLines(baseline, text.value);
    changes.textContent = lines ? `${lines} line${lines === 1 ? "" : "s"} changed` : "unchanged: edit it to have something to test";
  };
  text.addEventListener("input", count);
  count();

  const from = saved.length
    ? h("select", {
        class: "field",
        "aria-label": "Start from",
        onchange: async (event) => {
          const chosen = event.target.value;
          if (!chosen) {
            text.value = baseline;
            name.value = suggestedName(saved);
          } else {
            const variant = await api.get(`/variants/${enc(skill.skill)}/${enc(chosen)}`);
            text.value = variant.text;
            name.value = variant.name;
          }
          count();
          text.dispatchEvent(new Event("input", { bubbles: true })); // the form checks it again
        },
      },
      h("option", { value: "" }, "the wording the run was given"),
      saved.map((variant) => h("option", { value: variant.name }, `${variant.name} (saved)`)))
    : null;

  return h("div", { class: "variant-editor" },
    original
      ? h("div", { class: "banner banner-warn" }, h("strong", null, "This run kept no copy of its skill file. "),
          "Name the original — the SKILL.md as the run was given it, on the proxy's machine — and write the variant in full below.")
      : h("p", { class: "hint" }, "The skill as this run was given it",
          skill.path ? [": ", h("span", { class: "mono" }, skill.path)] : "",
          ". Change what you want to test; the A/B swaps your text in wherever the original sits in the context."),
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
  legend = `${LABELS.whatToCount} (chosen before sampling)`,
} = {}) {
  const clusters = targets.behaviours?.clusters || [];
  const resample = targets.resample;
  const pick = (kind) => {
    const radio = form.querySelector(`[name="target_kind"][value="${kind}"]`);
    if (radio && !radio.checked) {
      radio.checked = true;
      radio.dispatchEvent(new Event("change", { bubbles: true }));
    }
  };
  const builder = describeBuilder({ onUse: () => pick("describe") });
  const seen = run ? seenPreview({ run, idx, onTools: builder.setTools }) : null;
  const choice = (value, label, detail, disabled = false) =>
    h("label", { class: ["target-choice", disabled && "disabled"] },
      h("input", { type: "radio", name: "target_kind", value, disabled }),
      h("span", null, h("strong", null, label), detail ? h("span", { class: "muted small" }, ` — ${detail}`) : null));

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

  const form = h("fieldset", { class: "target-picker" },
    h("legend", null, legend),
    choice("letter", "A behaviour from this step's latest resample",
      resample
        ? `resample ${short(resample.id)}, ${targets.behaviours.total} samples${targets.normalizers.length ? `, grouped by ${targets.normalizers.join(", ")}` : ""}`
        : "none yet: resample this step first, to name one",
      !clusters.length),
    letters,
    choice("describe", "A behaviour I describe", "the tools it calls, the phrases it mentions"),
    builder.element,
    choice("pattern", "What the model does matches a pattern", "a regular expression over its text, reasoning and tool calls"),
    h("div", { class: "target-pattern" }, pattern),
    exploratory ? choice("exploratory", "Nothing declared", "every behaviour is compared, and each p-value adjusted for that (Bonferroni)") : null,
    seen ? seen.element : null);
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
    throw Object.assign(new Error("Choose what to count before sampling: a behaviour, a pattern, or nothing declared."), { field: "target_kind" });
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
