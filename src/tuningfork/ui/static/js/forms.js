// Forms that start work.
//
// A form shows, as it is edited, the plan the server would run for it: a dry run of the
// job, checked the way starting it is checked. Start is enabled only while the plan on
// screen is the plan for what the form says now — never a stale preview, never a plan the
// server has not seen. Starting twice from one form (a double click, a retry) is one job:
// each start carries the form's request id.

import { answered, api } from "./api.js";
import { guard } from "./auth.js";
import { h, replace } from "./dom.js";

export function field(label, control, { help, wide } = {}) {
  return h("label", { class: ["form-field", wide && "wide"] },
    h("span", { class: "form-label" }, label), control,
    help ? h("span", { class: "form-help" }, help) : null);
}

export function input(name, { type = "text", value, placeholder, min, max, step, ...rest } = {}) {
  // The rest goes on the element too: the batch form's `oninput` was dropped here, so a new
  // spec typed in never put the task list back to its first (found in review).
  return h("input", {
    class: "field", name, type, value: value ?? "", placeholder, min, max, step,
    autocomplete: "off", spellcheck: "false", ...rest,
  });
}

export function textArea(name, value, { rows = 12, placeholder } = {}) {
  return h("textarea", { class: "field mono", name, rows, spellcheck: "false", placeholder, value: value ?? "" });
}

/** A number from a field, or null when it is empty; `NaN` becomes an error the form names. */
export function numberOf(form, name, { integer = true } = {}) {
  const raw = form.querySelector(`[name="${name}"]`)?.value.trim() ?? "";
  if (raw === "") return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw Object.assign(new Error(`${name} must be ${integer ? "a whole number" : "a number"}`), { field: name });
  }
  return value;
}

export function textOf(form, name) {
  return form.querySelector(`[name="${name}"]`)?.value ?? "";
}

// How a step is sampled, beyond how many: said once for every form that samples one.
export const SEED_HELP = "Sample i is sent with seed base+i. Empty: a new random base, recorded with the experiment; an earlier one draws its samples again.";
export const CACHE_HELP = "Faster, but on llama.cpp a sample's behaviour can then depend on what the server ran before. Off, each sample's prompt is processed whole.";

export function seedField() {
  return field("Seed base", input("seed_base", { type: "number", placeholder: "random", min: 0 }), { help: SEED_HELP });
}

/** How behaviours are grouped: the normalizers an experiment reads them by, none naming the defaults. */
export function groupingField() {
  return field("How behaviours are grouped", input("normalizers", { placeholder: "the defaults" }),
    { help: "Normalizer names, comma-separated. None means the defaults.", wide: true });
}

export function cacheField() {
  return h("label", { class: "form-field" },
    h("span", { class: "form-label" }, "Prompt cache"),
    h("span", null, h("input", { type: "checkbox", name: "prompt_cache" }), " Reuse the server's cache of the prompt"),
    h("span", { class: "form-help" }, CACHE_HELP));
}

/** The seed base and the prompt cache, as a form says them: a seed only when one is given. */
export function samplingChoices(form) {
  const seed = numberOf(form, "seed_base");
  return { ...(seed !== null ? { seed_base: seed } : {}), prompt_cache: form.querySelector('[name="prompt_cache"]')?.checked === true };
}

/** A refusal as the server gave it: what, why, and the field it is about. `quiet`: said of a
 *  form nobody has touched yet, where it is what to fill in, not a mistake — a note, not red. */
export function formError(error, { quiet = false } = {}) {
  return h("div", { class: quiet ? "form-note-box" : "form-error-box", role: quiet ? "status" : "alert" },
    h("strong", null, error.message || String(error)),
    error.hint ? h("div", { class: "hint" }, error.hint) : null,
    error.detail ? h("pre", { class: "code small" }, error.detail) : null);
}

const newId = () =>
  globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;

/**
 * kind      the job kind, as `GET /jobs/kinds` names it — or (form) => kind, for a form that
 *           starts one of several
 * fields    the form's controls
 * read      (form) => parameters; throws { message, field } for what the form can tell itself
 * render    (plan) => the preview of the server's plan
 * onStarted (job) => what to do once the job is queued
 * send      (params, { dry_run, request_id }) => the server's answer, for a form whose work is
 *           not a job: by default `POST /jobs` with `kind`. A dry run asks `quiet`: its refusal
 *           is an answer (`answered`), never an error a browser logs.
 * planLabel what the plan box is called: what the form will run, by default
 */
export function planForm({ kind, fields, read, render, startLabel = "Start", onStarted, send = null, planLabel = "What will run" }) {
  const kindOf = typeof kind === "function" ? kind : () => kind;
  const post = send || (async (params, extra) =>
    answered(await api.post("/jobs", { kind: kindOf(form), params, ...extra, ...(extra.dry_run ? { quiet: true } : {}) })));
  const status = h("div", { class: "plan-status" });
  const preview = h("div", { class: "plan-preview" });
  const start = h("button", { class: "btn btn-primary", type: "submit", disabled: true, "data-control": true }, startLabel);
  const note = h("span", { class: "muted small" });
  let asked = 0; // dry runs sent; only the latest one's answer is shown
  let shown = null; // the parameters whose plan is on screen, as JSON
  let checking = false;
  let requestId = newId();
  let timer = null;
  // Whether the form says anything other than it did when it opened, or Start was pressed: until
  // then a refusal says what to fill in, as a note (found in review: four forms opened on a red
  // error nobody had made).
  let touched = false;
  let opened = null;

  // novalidate: the server checks every field and says which is wrong. The browser's own
  // checks would block a submit silently — a number off its step grid is enough.
  // A form inside this one — a check's tester, in a scenario's form — answers its own fields
  // and its own submit; theirs bubble here, and are not this form's (a tester's Try it would
  // otherwise have made the scenario too).
  const ours = (event) => !event?.target?.closest || event.target.closest("form") === form;
  const started = guard(submit);
  const form = h("form", {
    class: "plan-form", novalidate: true,
    onsubmit: (event) => (ours(event) ? started(event) : undefined),
    oninput: (event) => (ours(event) ? changed(event) : undefined),
    onchange: (event) => (ours(event) ? changed(event) : undefined),
  },
    fields,
    h("div", { class: "plan-box" }, h("div", { class: "label" }, planLabel), status, preview),
    h("div", { class: "form-actions" }, start, note));

  function current() {
    try {
      return { params: read(form) };
    } catch (error) {
      return { error };
    }
  }

  /** What the form says, as one string: its parameters, or the reason it has none. */
  function said() {
    const { params, error } = current();
    return params ? JSON.stringify(params) : `!${error?.message}`;
  }

  function mark(name) {
    for (const invalid of form.querySelectorAll(".field-invalid")) invalid.classList.remove("field-invalid");
    const found = name ? form.querySelector(`[name="${CSS.escape(name)}"]`) : null;
    found?.classList.add("field-invalid");
    // A field folded away under Advanced settings is opened, so the one refused can be seen — a
    // fold inside the form only. A form itself in a fold the person closed stays closed: a
    // refusal answered again as the page redraws would otherwise open it by itself (found in
    // review, a refinement's forms each in a stage's fold).
    const fold = found?.closest("details");
    if (fold && form.contains(fold)) fold.open = true;
  }

  function changed(event) {
    if (event && !touched) touched = said() !== opened;
    // An event that leaves what the form says as its plan came back: a text field's change as it
    // loses the focus, after its input events asked already. Start stays as it is: disabled there,
    // on the mousedown that moves the focus to Start, it lost the click that followed. A refresh
    // has no event and always asks again: what the plan reads can change outside the form.
    if (event && timer === null && !checking && JSON.stringify(current().params) === shown) return;
    start.disabled = true;
    note.textContent = "";
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      check();
    }, 350);
  }

  async function check() {
    const mine = ++asked;
    const { params, error } = current();
    if (error) {
      shown = null;
      mark(touched ? error.field : null);
      replace(status, formError(error, { quiet: !touched }));
      replace(preview);
      sync();
      return;
    }
    checking = true;
    sync();
    replace(status, h("span", { class: "muted small waiting-inline" }, h("span", { class: "spinner" }), "checking…"));
    // The form changed while this was asked: a newer plan was asked for, or is about to be, the
    // timer running (found in review: on Run a task, a scenario chosen while the task file's plan
    // was on its way got that file's task back, and its own plan was refused).
    const stale = () => mine !== asked || timer !== null;
    try {
      const answer = await post(params, { dry_run: true });
      if (stale()) return;
      shown = JSON.stringify(params);
      mark(null);
      replace(status);
      replace(preview, render(answer.plan));
    } catch (failure) {
      if (stale()) return;
      shown = null;
      mark(touched ? failure.field : null);
      replace(status, formError(failure, { quiet: !touched }));
      replace(preview);
    } finally {
      if (mine === asked) {
        checking = false;
        sync();
      }
    }
  }

  function sync() {
    const { params } = current();
    start.disabled = checking || !params || JSON.stringify(params) !== shown;
  }

  async function submit(event) {
    event.preventDefault();
    touched = true;
    const { params } = current();
    if (!params || JSON.stringify(params) !== shown) {
      changed();
      return;
    }
    start.disabled = true;
    note.textContent = "starting…";
    try {
      const job = await post(params, { request_id: requestId });
      requestId = newId();
      note.textContent = "";
      onStarted(job);
    } catch (failure) {
      note.textContent = "";
      mark(failure.field);
      replace(status, formError(failure));
      sync();
    }
  }

  opened = said();
  check(); // the plan for the form as it opens
  return { element: form, refresh: () => changed() };
}
