// Starting work from a step: resample it, A/B a wording at it, or fork the run there.
//
// Each form previews the plan the proxy would run — the same checks starting it makes — and
// starts only that plan. A started job opens on its own page, which links what it makes.
//
// Resample and A/B reach either the next step only, resending the step's request, or the
// end: each sample a fork of the run, answered from the recording up to the step and live
// from there, judged by the task's check. Only a run tuningfork launched can be forked.

import { api, enc } from "./api.js";
import { chip, command, failure, h, kv, loading, panel, plural, replace, short } from "./dom.js";
import { cacheField, field, groupingField, input, numberOf, planForm, samplingChoices, seedField, textOf } from "./forms.js";
import { live, tokenWork } from "./live.js";
import { describeBuilder, seenPreview } from "./targets.js";
import { readTarget, readVariant, targetPicker, variantEditor } from "./variants.js";
import { LABELS } from "./words.js";
import { refineStartForm } from "./views/refine.js";

/** A link that opens `href` — through `go` when the page opens it in place (the run's
 *  timeline does), as an ordinary link otherwise, so middle-click still opens a tab. */
function way(href, go, props, label) {
  return h("a", {
    ...props, href,
    onclick: go ? (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
      event.preventDefault();
      go(href);
    } : undefined,
  }, label);
}

/** A step's way in, under the same title in either mode: buttons with a running proxy,
 *  commands without one. `work` is what the server's check rules out (`tokenWork`). */
export function startPanel(step, { go = null, work = {} } = {}) {
  const run = step.run;
  const here = `#/runs/${run.id}/${step.idx}`;
  if (live.mode !== "live") {
    return panel("Work from here", h("div", { class: "stack", style: { gap: "8px" } },
      h("p", { class: "hint" }, "Starting work needs a running proxy (tuningfork serve). From a terminal:"),
      command(`tuningfork resample ${short(run.id)} ${step.idx} -n 20`),
      command(`tuningfork ab ${short(run.id)} ${step.idx} --variant SKILL.v2.md --target-match '<regex>'`),
      command(`tuningfork refine start ${short(run.id)} ${step.idx} --target-match '<regex>'`)), { className: "work-panel" });
  }
  // A fork replays the launch that made the run, up to the step: a run nothing launched has
  // none to replay. Its button stays, greyed, and says why — and the way to runs that fork: a
  // scenario of this one, whose runs tuningfork launches. A thread is never a scenario.
  const why = run.launch_id ? null
    : `Forking needs a run tuningfork started. This one was ${run.recorded === "imported" ? "imported" : "recorded live"}.`;
  const scenario = why && !run.trashed_at && !["subagent", "side", "thread"].includes(run.role) ? `#/runs/${run.id}/scenario` : null;
  return panel("Work from here", [
    h("div", { class: "start-actions" },
      way(`${here}?start=resample`, go, { class: "btn", "data-control": true, title: "Send this step again N times, and count what the model does" }, "Resample…"),
      way(`${here}?start=ab`, go, { class: "btn", "data-control": true }, "A/B a wording…"),
      // Greyed where the server's check found it cannot force tokens: the same words refuse it.
      work.forking
        ? h("span", { class: "btn", "aria-disabled": "true", title: work.forking }, "Find forking tokens…")
        : way(`${here}?start=forking-tokens`, go, { class: "btn", "data-control": true }, "Find forking tokens…"),
      why
        ? h("span", {
            class: "btn", "aria-disabled": "true",
            title: why + (scenario ? " Make a scenario of this run, run it with tuningfork, and fork from one of its runs." : ""),
          }, "Fork from here…")
        : way(`${here}?start=fork`, go, { class: "btn", "data-control": true }, "Fork from here…"),
      way(`${here}?start=refine`, go, { class: "btn", "data-control": true }, "Refine the skill here…")),
    h("p", { class: "muted small start-note" }, "Measure what the model does here, with everything before it held fixed.",
      why ? ` ${why}` : null,
      scenario ? [" ", h("a", { href: scenario, "data-control": true }, "Make a scenario of it"), ", run it, and fork from one of its runs."] : null,
      work.forking ? [` ${work.forking}. `, h("a", { href: "#/connections" }, "Connections")] : null),
  ], { className: "work-panel" });
}

/** The form a `?start=` names, in a panel that closes back to the step. `reach=end` runs
 *  each sample to the end. */
export async function startForm(kind, step, query = new URLSearchParams(), { go = null } = {}) {
  const here = `#/runs/${step.run.id}/${step.idx}`;
  const close = way(here, go, { class: "btn btn-quiet btn-small" }, "Close");
  const toEnd = query.get("reach") === "end";
  const options = { actions: close, className: "start-panel" };
  if (kind === "resample") {
    return panel(`Resample step ${step.idx}`, [reachToggle(step, kind, toEnd, go),
      toEnd ? resampleToEndForm(step) : resampleForm(step, await tokenWork())], options);
  }
  if (kind === "ab") {
    return panel(`A/B a wording at step ${step.idx}`, [reachToggle(step, kind, toEnd, go), await abForm(step, toEnd)], options);
  }
  if (kind === "fork") {
    return panel(`Fork the run at step ${step.idx}`, forkForm(step), options);
  }
  if (kind === "forking-tokens") {
    return panel(`Find the forking tokens at step ${step.idx}`, await forkingForm(step, query), options);
  }
  if (kind === "refine") {
    return panel(`Refine the skill at step ${step.idx}`, await refineStartForm(step), options);
  }
  return null;
}

// --- forking tokens -----------------------------------------------------------------------

const REGIONS = [
  ["answer", "The answer", "the reply and its tool calls, where a skill's wording usually bites; continuations are short"],
  ["reasoning", "The reasoning", "where a model that thinks usually decides; each continuation rewrites the rest of the reasoning"],
  ["all", "Anywhere", "the likeliest forks wherever they are"],
];

async function forkingForm(step, query) {
  const picked = query.get("positions") || "";
  // The tools the step's own answer calls: where it calls any, the target first offered is
  // calling them, with any arguments. Found on find-the-file: the answer's exact call counted
  // a search with another pattern as a miss, so a token that changed what the model did went
  // unseen.
  const detail = await api.get(`/runs/${enc(step.run.id)}/steps/${step.idx}?context=0`).catch(() => null);
  const calls = [...new Set((detail?.response?.choices?.[0]?.message?.tool_calls || [])
    .map((call) => call.function?.name).filter(Boolean))];
  const callsRules = calls.map((name) => ({ if: "calls", value: name }));
  const regions = h("fieldset", { class: "target-picker" }, h("legend", null, "Where to look, when no positions are named"),
    REGIONS.map(([value, label, detail]) =>
      h("label", { class: "target-choice" },
        h("input", { type: "radio", name: "region", value, checked: value === "answer" }),
        h("span", null, h("strong", null, label), h("span", { class: "muted small" }, ` — ${detail}`)))));
  // Declared before sampling: the base's own behaviour, one described in conditions, or a
  // pattern — each shown against the behaviours already seen at this step.
  const pick = (kind) => {
    const radio = target.querySelector(`[name="target_kind"][value="${kind}"]`);
    if (radio && !radio.checked) {
      radio.checked = true;
      radio.dispatchEvent(new Event("change", { bubbles: true }));
    }
  };
  const builder = describeBuilder({ onUse: () => pick("describe") });
  const seen = seenPreview({ run: step.run.id, idx: step.idx, onTools: builder.setTools });
  const pattern = input("target_match", { placeholder: "a regular expression, e.g. row_count" });
  pattern.addEventListener("input", () => pick("match"));
  const howMany = input("top", { type: "number", value: 5, min: 1, max: 20 });
  const leastChance = input("min_p", { type: "number", value: 0.05, min: 0.001, max: 0.5, step: "any" });
  const target = h("fieldset", { class: "target-picker" }, h("legend", null, `${LABELS.whatToCount} (chosen before sampling)`),
    calls.length
      ? h("label", { class: "target-choice" },
          h("input", { type: "radio", name: "target_kind", value: "tools", checked: true }),
          h("span", null, h("strong", null, `Calls ${calls.join(" and ")}`),
            h("span", { class: "muted small" }, " — the tools the answer calls, with any arguments: a continuation that calls them another way still counts")))
      : null,
    h("label", { class: "target-choice" },
      h("input", { type: "radio", name: "target_kind", value: "base", checked: !calls.length }),
      h("span", null, h("strong", null, "The answer's own behaviour"), h("span", { class: "muted small" }, calls.length
        ? " — its exact tool call, arguments and all, read the way every continuation is read"
        : " — its tool call or reply, read the way every continuation is read"))),
    h("label", { class: "target-choice" },
      h("input", { type: "radio", name: "target_kind", value: "describe" }),
      h("span", null, h("strong", null, "A behaviour I describe"), h("span", { class: "muted small" }, " — the tools it calls, the phrases it mentions"))),
    builder.element,
    h("label", { class: "target-choice" },
      h("input", { type: "radio", name: "target_kind", value: "match" }),
      h("span", null, h("strong", null, "A behaviour matching "), pattern)),
    seen.element);
  const { element } = planForm({
    kind: "forking-tokens",
    fields: [
      field("Positions", input("positions", { value: picked, placeholder: "none: the likeliest forks in the region below" }),
        { help: "Comma-separated, as the Tokens view numbers them; select tokens there to fill this in.", wide: true }),
      regions,
      h("div", { class: "form-grid" },
        field("Positions to probe", howMany, { help: "When none are named." }),
        field("Alternatives", input("alternatives", { type: "number", value: 3, min: 1, max: 10 }), { help: "Other tokens forced at each." }),
        field("Continuations", input("n", { type: "number", value: 10, min: 1, max: 200 }), { help: "Per forced token." })),
      target,
      h("details", { class: "fold more-settings" },
        h("summary", null, LABELS.advanced),
        h("div", { class: "form-grid" },
          field("Least chance", leastChance, { help: "That the step's sampler gave an alternative, to count as a fork, when no positions are named." }),
          field("Significance (α)", input("alpha", { type: "number", value: 0.05, min: 0.0001, max: 0.5, step: "any" }), { help: "Corrected for the number of tests planned (Holm)." }),
          field("Max tokens", input("max_tokens", { type: "number", value: 4096, min: 1 }), { help: "Per continuation; hitting it counts as truncated." }),
          field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }), { help: "On one GPU, 1 keeps the prompt cache." }),
          groupingField())),
    ],
    read: (form) => {
      const positions = textOf(form, "positions").split(",").map((text) => text.trim()).filter(Boolean).map((text) => {
        const value = Number(text);
        if (!Number.isInteger(value) || value < 0) throw Object.assign(new Error("positions are whole numbers, comma-separated"), { field: "positions" });
        return value;
      });
      const params = {
        run: step.run.id,
        step: String(step.idx),
        positions,
        alternatives: numberOf(form, "alternatives") ?? 3,
        n: numberOf(form, "n") ?? 10,
        concurrency: numberOf(form, "concurrency") ?? 1,
        max_tokens: numberOf(form, "max_tokens") ?? 4096,
        normalizers: textOf(form, "normalizers").split(",").map((name) => name.trim()).filter(Boolean),
      };
      // Where to look, how many and the least chance choose the positions: named ones are
      // probed as named, so these are sent only when none are.
      if (!positions.length) {
        params.region = form.querySelector('[name="region"]:checked')?.value || "answer";
        params.top = numberOf(form, "top") ?? 5;
        const minP = numberOf(form, "min_p", { integer: false });
        if (minP !== null) params.min_p = minP;
      }
      const alpha = numberOf(form, "alpha", { integer: false });
      if (alpha !== null) params.alpha = alpha;
      const kind = form.querySelector('[name="target_kind"]:checked')?.value;
      if (kind === "match") {
        const pattern = textOf(form, "target_match").trim();
        if (!pattern) throw Object.assign(new Error("name the behaviour to match"), { field: "target_match" });
        params.target_match = pattern;
      } else if (kind === "describe") {
        params.target_rules = builder.read();
      } else if (kind === "tools") {
        params.target_rules = callsRules;
      }
      return params;
    },
    render: forkingPreview,
    startLabel: "Find the forking tokens",
    onStarted: opened,
  });
  // What the target would count, as the form changes: the target, or the lens it reads by.
  const lens = () => textOf(element, "normalizers").split(",").map((name) => name.trim()).filter(Boolean);
  const declared = () => {
    const kind = element.querySelector('[name="target_kind"]:checked')?.value;
    if (kind === "match") return textOf(element, "target_match").trim() ? { target_match: textOf(element, "target_match").trim() } : {};
    if (kind === "tools") return { target_rules: callsRules };
    if (kind === "describe") {
      try {
        return { target_rules: builder.read() };
      } catch {
        return {};
      }
    }
    return { base: true };
  };
  const look = () => seen.update({ ...declared(), normalizers: lens() });
  element.addEventListener("input", look);
  element.addEventListener("change", look);
  look();
  // With positions named, the settings that would choose them are greyed: a setting nothing
  // reads is not offered (found in use: a person read them as part of the probe they asked for).
  const offer = () => {
    const named = textOf(element, "positions").trim() !== "";
    regions.disabled = named;
    for (const control of [howMany, leastChance]) {
      control.disabled = named;
      control.closest(".form-field").classList.toggle("off", named);
    }
  };
  element.querySelector('[name="positions"]').addEventListener("input", offer);
  offer();
  return element;
}

const pct = (p) => (p >= 0.995 || p === 0 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`);

// Each continuation processes the step's whole prompt first: counted in the minutes when the
// step's time to first token said how long that takes, and said otherwise.
function forkingTakes(estimate) {
  const tokens = `~${plural(estimate.tokens, "token")} to generate`;
  const prompt = estimate.prompt_tokens ? plural(estimate.prompt_tokens, "token") : null;
  if (!estimate.minutes) return prompt ? `${tokens}, each continuation processing the whole prompt (${prompt}) first` : tokens;
  if (estimate.prefill_seconds && prompt) {
    return `${tokens}, about ${estimate.minutes} min at the step's own speed — counting the whole prompt (${prompt}), processed first for each continuation, ~${estimate.prefill_seconds}s`;
  }
  return `${tokens}, about ${estimate.minutes} min at the step's own speed${prompt ? `, and each continuation first processes the whole prompt (${prompt})` : ""}`;
}

function forkingPreview(plan) {
  const power = plan.power || {};
  const estimate = plan.estimate || {};
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["step", checkpointLine(plan)],
      ["forks from", h("span", null, h("span", { class: "mono" }, `stream ${short(plan.base.id)}`),
        ` — ${plan.base.source === "scored" ? "the answer the run gave, scored" : plan.base.source}, ${plural(plan.base.tokens, "token")}`)],
      ["counts", plan.target],
      ["draws", `${plural(plan.arms.length, "forced token")} × ${plan.per_arm} = ${plural(plan.planned, "continuation")}`],
      ["can detect", power.detectable
        ? `a forced token that changes the outcome in about ${power.detectable} of its ${power.of} continuations, or more, corrected for ${plural(plan.family, "test")}`
        : h("span", { class: "form-error" }, `nothing: with ${power.of} continuations a token and ${plural(plan.family, "test")}, no change could be told from chance. Raise the continuations.`)],
      ["takes", forkingTakes(estimate)],
    ]),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, "position"), h("th", null, "part"), h("th", null, "the run's token"), h("th", null, "forced instead"))),
      h("tbody", null, plan.positions.map((position) =>
        h("tr", null,
          h("td", { class: "mono" }, `[${position.position}]`),
          h("td", { class: "muted" }, position.part || "—"),
          h("td", { class: "mono" }, `${JSON.stringify(position.chosen.text)} ${pct(position.chosen.p)}`),
          h("td", { class: "mono" }, position.alternatives.map((alt) => `${JSON.stringify(alt.text)} ${pct(alt.p)}`).join("   "))))))),
    h("p", { class: "hint" }, "At each position the model is given this step's exact context and the answer up to there, then one token forced — the run's own, or an alternative — and goes on. Each continuation is read back into a behaviour, and each alternative is compared with the run's own token at the same position."));
}

function reachToggle(step, kind, toEnd, go) {
  const here = `#/runs/${step.run.id}/${step.idx}?start=${kind}`;
  const option = (href, label, active, title) =>
    way(href, go, { class: active ? "active" : null, "aria-pressed": String(active), title }, label);
  return h("div", { class: "reach" },
    h("span", { class: "muted small" }, LABELS.judgeBy),
    h("div", { class: "toggle" },
      option(here, LABELS.nextOnly, !toEnd, "Resend this step's request, and see what the model does next."),
      option(`${here}&reach=end`, LABELS.wholeRun, toEnd, "Fork the run: replayed up to here, then live to the end, where the task's check decides.")),
    toEnd && !step.run.launch_id
      ? h("span", { class: "form-error small" }, "Only a run tuningfork started can be forked.")
      : null);
}

const opened = (job) => (location.hash = `#/jobs/${job.id}`);

// --- how a step is sampled, shared ---------------------------------------------------------

function samplingFields({ alpha = false, beside = null } = {}) {
  return [
    h("div", { class: "form-grid" },
      field("Samples", input("n", { type: "number", value: 20, min: 1, max: 1000 }), { help: alpha ? "For each wording." : "More samples, a narrower interval." }),
      beside),
    h("details", { class: "fold more-settings" },
      h("summary", null, LABELS.advanced),
      h("div", { class: "form-grid" },
        field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 64 }), { help: "The proxy's own limit on calls at once still applies." }),
        field("Max tokens", input("max_tokens", { type: "number", value: 4096, min: 1 }), { help: "Hitting it counts as truncated." }),
        seedField(),
        cacheField(),
        alpha ? field("Significance (α)", input("alpha", { type: "number", value: 0.05, min: 0.0001, max: 0.5, step: "any" })) : null,
        groupingField())),
  ];
}

function readSampling(form) {
  const normalizers = textOf(form, "normalizers").split(",").map((name) => name.trim()).filter(Boolean);
  const params = {
    n: numberOf(form, "n") ?? 20,
    concurrency: numberOf(form, "concurrency") ?? 1,
    max_tokens: numberOf(form, "max_tokens") ?? 4096,
    ...samplingChoices(form),
    normalizers,
  };
  return params;
}

function estimateText(estimate) {
  if (!estimate) return null;
  if (estimate.fresh) {
    return `each sample processes its whole prompt${estimate.prompt_tokens ? ` (${estimate.prompt_tokens.toLocaleString()} tokens)` : ""}, then answers`;
  }
  const minutes = estimate.worst_minutes;
  return minutes < 1 ? `at most about ${Math.max(1, Math.round(minutes * 60))} s` : `at most about ${minutes.toFixed(1)} min`;
}

function checkpointLine(plan) {
  return h("span", null, h("span", { class: "mono" }, `run ${short(plan.checkpoint.run_id)}, step ${plan.checkpoint.step}`),
    h("span", { class: "muted" }, ` — ${plan.checkpoint.describe}`));
}

// --- resample -------------------------------------------------------------------------------

function resampleForm(step, work = {}) {
  // Each sample's tokens as the chat endpoint gives them, for its heatmap: greyed, saying why,
  // where the server's check found it sends none.
  const keep = input("logprobs", { type: "number", min: 1, max: 20, placeholder: "not kept" });
  const kept = field("Keep each sample's tokens", keep, {
    help: work.logprobs || "With this many alternatives at each token, for each sample's heatmap in the step's Tokens view.",
  });
  keep.disabled = Boolean(work.logprobs);
  kept.classList.toggle("off", Boolean(work.logprobs));
  const overrides = h("details", { class: "fold more-settings" },
    h("summary", null, "Change the sampling parameters (recorded as a change)"),
    h("div", { class: "form-grid" },
      field("Temperature", input("temperature", { type: "number", step: "any", min: 0, max: 5, placeholder: "as recorded" })),
      field("top_p", input("top_p", { type: "number", step: "any", min: 0, max: 1, placeholder: "as recorded" })),
      field("top_k", input("top_k", { type: "number", min: 0, placeholder: "as recorded" }))));
  const { element } = planForm({
    kind: "resample",
    fields: [...samplingFields({ beside: kept }), overrides],
    read: (form) => {
      const params = {
        run: step.run.id,
        step: String(step.idx),
        ...readSampling(form),
        temperature: numberOf(form, "temperature", { integer: false }),
        top_p: numberOf(form, "top_p", { integer: false }),
        top_k: numberOf(form, "top_k"),
      };
      const logprobs = numberOf(form, "logprobs");
      if (logprobs !== null) params.logprobs = logprobs;
      return params;
    },
    render: (plan) => h("div", { class: "stack", style: { gap: "8px" } },
      kv([
        ["step", checkpointLine(plan)],
        ["samples", `${plural(plan.sampling.n, "sample")}, ${plan.sampling.concurrency} at a time`],
        ["through", h("span", { class: "mono" }, plan.base_url)],
        ["takes", estimateText(plan.estimate)],
        ["changed", Object.keys(plan.sampling.overrides || {}).length
          ? h("span", { class: "mono" }, JSON.stringify(plan.sampling.overrides)) : null],
        ["tokens", plan.sampling.logprobs
          ? `each sample's kept, with ${plural(plan.sampling.logprobs, "alternative")} at each token` : null],
      ]),
      h("p", { class: "hint" }, "Each sample resends this step's request exactly as recorded, with its own seed. Everything before the step stays as it was, and nothing is recorded as a run.")),
    startLabel: "Start the resample",
    onStarted: opened,
  });
  return element;
}

// --- A/B ---------------------------------------------------------------------------------

async function abForm(step, toEnd = false) {
  const first = await api.get(`/runs/${enc(step.run.id)}/skill`);
  const kept = first.kept || [];
  if (kept.length <= 1) return abBody(step, toEnd, first, null);
  // A run that kept several skills — an imported chat — varies the one chosen here.
  const pick = h("select", { class: "field", name: "skill" },
    h("option", { value: "", selected: !first.text, disabled: true }, "choose the skill the variant replaces"),
    kept.map((name) => h("option", { value: name, selected: Boolean(first.text) && name === first.skill }, name)));
  const body = h("div");
  const load = async () => {
    replace(body, loading("Loading the skill…"));
    try {
      const skill = await api.get(`/runs/${enc(step.run.id)}/skill?skill=${enc(pick.value)}`);
      replace(body, await abBody(step, toEnd, skill, pick.value));
    } catch (error) {
      replace(body, failure(error, "Could not load the skill"));
    }
  };
  pick.addEventListener("change", load);
  if (pick.value) await load();
  return h("div", { class: "stack" },
    field("Skill", pick, { help: "The run kept several: the variant replaces the one chosen." }), body);
}

async function abBody(step, toEnd, skill, named) {
  const targets = toEnd ? null : await api.get(`/runs/${enc(step.run.id)}/steps/${step.idx}/targets`);
  const saved = await api.get(`/variants?skill=${enc(skill.skill)}`);
  // The skill the variant replaces, where the run kept several.
  const which = named ? { skill: named } : {};
  if (toEnd) {
    const { element } = planForm({
      kind: "ab-to-end",
      fields: [
        variantEditor(skill, saved),
        h("p", { class: "hint" }, "What you count is the task's check passing, set by the task before any fork runs."),
        ...forkFields({ alpha: true }),
      ],
      read: (form) => {
        const params = { run: step.run.id, step: String(step.idx), ...readVariant(form), ...readForks(form) };
        const alpha = numberOf(form, "alpha", { integer: false });
        if (alpha !== null) params.alpha = alpha;
        return params;
      },
      render: toEndPreview,
      startLabel: "Start the A/B",
      onStarted: opened,
    });
    return element;
  }
  const { element } = planForm({
    kind: "ab",
    fields: [variantEditor(skill, saved), targetPicker(targets, { run: step.run.id, idx: step.idx }), ...samplingFields({ alpha: true })],
    read: (form) => {
      const params = {
        run: step.run.id,
        step: String(step.idx),
        ...readVariant(form),
        ...which,
        ...readTarget(form),
        ...readSampling(form),
      };
      const alpha = numberOf(form, "alpha", { integer: false });
      if (alpha !== null) params.alpha = alpha;
      return params;
    },
    render: abPreview,
    startLabel: "Start the A/B",
    onStarted: opened,
  });
  return element;
}

// --- forks: to the end, or stopped at the step ----------------------------------------------

function forkFields({ alpha = false, n = 5 } = {}) {
  return [
    h("div", { class: "form-grid" },
      field("Forks", input("n", { type: "number", value: n, min: 1, max: 200 }), { help: alpha
        ? "For each wording. Each fork runs in full from here, so it takes minutes, not seconds."
        : "Each fork runs in full from here, so it takes minutes, not seconds." })),
    h("details", { class: "fold more-settings" },
      h("summary", null, LABELS.advanced),
      h("div", { class: "form-grid" },
        field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }), { help: "No more than the proxy's own limit on calls at once. On one GPU, 1 is fastest." }),
        alpha ? field("Significance (α)", input("alpha", { type: "number", value: 0.05, min: 0.0001, max: 0.5, step: "any" })) : null)),
  ];
}

function readForks(form) {
  return { n: numberOf(form, "n") ?? 5, concurrency: numberOf(form, "concurrency") ?? 1 };
}

function forkLines(plan) {
  return [
    ["step", checkpointLine(plan)],
    ["replays", `${plural(plan.replayed, "recorded call")}, answered from the recording; live from step ${plan.checkpoint.step}`],
    ["harness", h("span", null, h("span", { class: "mono" }, plan.harness), ", model ", h("span", { class: "mono" }, plan.model || "as recorded"))],
  ];
}

function toEndPreview(plan) {
  const arm = plan.original;
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ...forkLines(arm),
      ["forks", `${plural(plan.forks, "fork")}${plan.variant ? ` (${plan.n} for each wording)` : ""}, ${plan.concurrency} at a time`],
      ["counts", "the task's check passes"],
      plan.variant ? ["swaps", `the original for your variant, from step ${arm.checkpoint.step} on`] : null,
    ].filter(Boolean)),
    [...(arm.warnings || []), ...((plan.variant && plan.variant.warnings) || [])].map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "Every fork replays the recorded history up to this step, then runs live to the end with breakpoints off; the task's check decides. A fork cut off or taken off the recording is not counted."));
}

function resampleToEndForm(step) {
  const { element } = planForm({
    kind: "resample-to-end",
    fields: forkFields(),
    read: (form) => ({ run: step.run.id, step: String(step.idx), ...readForks(form) }),
    render: toEndPreview,
    startLabel: "Start the forks",
    onStarted: opened,
  });
  return element;
}

const HOLDS = [
  ["step", "Stop at the step", "the fork waits, held, before the model answers there — for you to look, edit, and step on. Breakpoints armed on the proxy apply too."],
  ["breakpoints", "Only where breakpoints stop it", "no stop of its own; the proxy's armed breakpoints apply to its live calls."],
  ["none", "Run to the end unattended", "breakpoints off."],
];

function forkForm(step) {
  const holds = h("fieldset", { class: "target-picker" }, h("legend", null, "Where it stops"),
    HOLDS.map(([value, label, detail]) =>
      h("label", { class: "target-choice" },
        h("input", { type: "radio", name: "hold", value, checked: value === "step" }),
        h("span", null, h("strong", null, label), h("span", { class: "muted small" }, ` — ${detail}`)))));
  const { element } = planForm({
    kind: "fork",
    fields: [holds, h("div", { class: "form-grid" },
      field("Forks", input("n", { type: "number", value: 1, min: 1, max: 20 }))),
      h("details", { class: "fold more-settings" },
        h("summary", null, LABELS.advanced),
        h("div", { class: "form-grid" }, field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }))))],
    read: (form) => ({
      run: step.run.id,
      step: String(step.idx),
      hold: form.querySelector('[name="hold"]:checked')?.value || "step",
      n: numberOf(form, "n") ?? 1,
      concurrency: numberOf(form, "concurrency") ?? 1,
    }),
    render: (plan) => h("div", { class: "stack", style: { gap: "8px" } },
      kv([...forkLines(plan), ["forks", `${plural(plan.n, "fork")}, ${plan.concurrency} at a time`],
        ["stops", HOLDS.find(([value]) => value === plan.hold)?.[1]]]),
      (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
      plan.hold === "step" ? h("p", { class: "hint" }, "When it stops, the Live page shows the held call; the job's page links to it.") : null),
    startLabel: "Start the fork",
    onStarted: opened,
  });
  return element;
}

function abPreview(plan) {
  const spans = plan.spans || [];
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["step", checkpointLine(plan)],
      ["swaps", h("span", null, `the original, found in ${plural(spans.length, "place")} in this step's context, for your variant`)],
      ["counts", plan.target ? h("span", null, plan.target) : chip("nothing declared: every behaviour compared, each p adjusted", "warn")],
      ["samples", `${plural(plan.sampling.n, "sample")} for each wording, ${plan.sampling.concurrency} at a time, α = ${plan.alpha}`],
      ["takes", estimateText(plan.estimate)],
      ["saved as", h("span", { class: "mono" }, plan.variant)],
    ]),
    plan.unapplied?.length
      ? h("div", { class: "banner banner-warn" }, h("strong", null, "Parts of your edit the model never sees at this step: "),
          h("ul", { class: "small" }, plan.unapplied.map((line) => h("li", null, line))))
      : null,
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "Both wordings resend this step's request: the original as recorded, and the variant with your text swapped in. Everything before this step was written under the original wording, so treat the result as a lead, and confirm it with full runs."));
}
