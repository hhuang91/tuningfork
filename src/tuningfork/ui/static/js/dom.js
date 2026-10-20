// DOM helpers and formatting.
//
// Every string reaches the page as a text node. Payloads are written by a model and a
// harness, and are full of markup-looking text — so nothing here ever parses HTML, and
// the test suite checks that no module assigns innerHTML.

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") {
      el.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : value;
    } else if (key === "style") {
      for (const [name, rule] of Object.entries(value)) el.style.setProperty(name.startsWith("--") ? name : kebab(name), rule);
    } else if (key === "dataset") {
      for (const [name, data] of Object.entries(value)) el.dataset[name] = String(data);
    } else if (key.startsWith("on")) {
      // A string here would become an inline handler: code parsed from text.
      if (typeof value !== "function") throw new TypeError(`${key} must be a function`);
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (["value", "checked", "disabled", "selected", "open", "hidden"].includes(key)) {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? "" : String(value));
    }
  }
  return append(el, children);
}

const kebab = (name) => name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

export function append(el, children) {
  for (const child of [children].flat(Infinity)) {
    if (child === undefined || child === null || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

export function replace(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

// --- formatting ----------------------------------------------------------------------

export const short = (id) => (id ? String(id).slice(0, 8) : "—");
export const count = (n) => Number(n || 0).toLocaleString();
export const percent = (x) => `${Math.round(x * 100)}%`;
export const plural = (n, word) => `${count(n)} ${word}${n === 1 ? "" : "s"}`;
export const pretty = (value) => JSON.stringify(value, null, 2);

export function when(seconds) {
  if (!seconds) return "—";
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function ago(seconds) {
  if (!seconds) return "—";
  const delta = Math.max(0, Date.now() / 1000 - seconds);
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

export function elapsed(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** How long it has been since `seconds` (a Unix time), kept current by `tick`. */
export function since(seconds, prefix = "") {
  return h("span", { dataset: { since: seconds, prefix } }, prefix + elapsed(Date.now() / 1000 - seconds));
}

/** Bring every running time under `root` up to date: text only, no requests, and nothing
 *  while the page is hidden. Called each second by a page that shows running times. */
export function tick(root) {
  if (document.hidden) return;
  const now = Date.now() / 1000;
  for (const el of root.querySelectorAll("[data-since]")) {
    el.textContent = (el.dataset.prefix || "") + elapsed(now - Number(el.dataset.since));
  }
}

export function millis(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

export function firstLine(text, limit = 120) {
  const line = String(text ?? "").trimStart().split("\n", 1)[0];
  return line.length > limit ? `${line.slice(0, limit)}…` : line;
}

// --- small components ---------------------------------------------------------------

export const chip = (text, tone = "muted", title) =>
  h("span", { class: `chip tone-${tone}`, title }, text);

export const loading = (what = "Loading…") => h("div", { class: "loading" }, what);

export function failure(error, title = "Could not load this") {
  return h("div", { class: "error-box" }, h("strong", null, title), h("p", null, error.message || String(error)));
}

export function crumbs(items) {
  const parts = [];
  for (const item of items.filter(Boolean)) {
    if (parts.length) parts.push(h("span", { class: "sep" }, "/"));
    const [label, href] = item;
    parts.push(href ? h("a", { href }, label) : h("span", null, label));
  }
  return h("nav", { class: "crumbs", "aria-label": "Breadcrumb" }, parts);
}

export function panel(title, body, { meta, actions, className } = {}) {
  return h(
    "section",
    { class: ["panel", className] },
    h("div", { class: "panel-head" }, h("h2", null, title), meta ? h("div", { class: "panel-meta" }, meta) : null,
      actions ? h("div", { class: "actions" }, actions) : null),
    h("div", { class: "panel-body" }, body),
  );
}

export function kv(pairs) {
  const rows = [];
  for (const [key, value] of pairs) {
    if (value === undefined || value === null || value === "") continue;
    rows.push(h("dt", null, key), h("dd", null, value));
  }
  return h("dl", { class: "kv" }, rows);
}

export function toggle(options, current, onChange) {
  const el = h("div", { class: "toggle", role: "group" });
  for (const [value, label] of options) {
    el.append(
      h("button", {
        type: "button",
        class: value === current ? "active" : null,
        "aria-pressed": String(value === current),
        onclick: () => onChange(value),
      }, label),
    );
  }
  return el;
}

export function copyButton(text, label = "Copy") {
  const button = h("button", { class: "btn btn-quiet btn-small", type: "button" }, label);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    setTimeout(() => (button.textContent = label), 1500);
  });
  return button;
}

export function command(text) {
  return h("div", { class: "cmd" }, h("span", null, text), copyButton(text));
}

/** Text that may be long: in full when short, otherwise folded behind its first line. */
export function longText(text, { limit = 1600, lines = 24, open = false, className = "text" } = {}) {
  const value = String(text ?? "");
  const lineCount = value.split("\n").length;
  if (value.length <= limit && lineCount <= lines) return h("pre", { class: className }, value);
  return h(
    "details",
    { class: "fold", open },
    h("summary", null, h("span", { class: "fold-preview" }, firstLine(value)),
      h("span", { class: "faint" }, `${count(value.length)} chars · ${plural(lineCount, "line")}`)),
    h("pre", { class: className }, value),
  );
}

/** Text with line numbers, for a skill as the model read it. `start` is the file line the
 *  text begins on, when the harness showed the model only part of the file. */
export function numbered(text, start = 1) {
  const lines = String(text).split("\n");
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  return h(
    "div",
    { class: "lines" },
    lines.map((line, index) =>
      h("div", { class: "line" }, h("span", { class: "ln" }, String(index + start)), h("span", { class: "lt" }, line || " ")),
    ),
  );
}

// --- rates: never a bare percentage -----------------------------------------------

export function rateText(rate) {
  return `${rate.successes}/${rate.total}  ${percent(rate.point)}  [${percent(rate.low)}–${percent(rate.high)}]`;
}

/** A rate as tuningfork always shows one: its count, with the interval drawn. */
export function rateBar(rate, tone = "accent") {
  const at = (x) => `${(x * 100).toFixed(2)}%`;
  return h(
    "span",
    { class: "ci", title: `${rate.successes} of ${rate.total}; Wilson 95% interval ${percent(rate.low)}–${percent(rate.high)}` },
    h("span", { class: `ci-track tone-${tone}` },
      h("span", { class: "ci-band", style: { left: at(rate.low), width: at(rate.high - rate.low) } }),
      h("span", { class: "ci-point", style: { left: at(rate.point) } })),
    h("span", { class: "ci-text" }, rateText(rate)),
  );
}

// --- outcomes and verdicts ----------------------------------------------------------

const OUTCOME_TONES = { tool_call: "accent", text: "muted", empty: "warn", truncated: "warn", error: "bad", ended: "faint" };

export function outcome(o) {
  if (!o) return h("span", { class: "faint" }, "—");
  return h(
    "span",
    { class: "outcome" },
    chip(o.kind.replace("_", " "), OUTCOME_TONES[o.kind] || "muted"),
    h("span", { class: "mono outcome-label" }, o.label),
  );
}

/** A run recorded live, marked as what it is where a label would name it. */
export const recordedLive = () =>
  chip("recorded live", "muted", "Recorded through the proxy as it happened: no batch launched it, and it was not imported");

/** A run imported from a conversation had elsewhere, marked as a live one is. */
export const importedRun = () =>
  chip("imported", "muted", "Imported from a conversation an agent had elsewhere: each step rebuilt from the chat, nothing launched");

export function verdict(result) {
  if (!result) return null;
  let base;
  // A run stopped before it finished has no verdict either, and its task may well have a
  // check: it never ran (found in review).
  const stopped = result.cancelled ? (result.exclusion || "stopped before it finished") : null;
  if (stopped) base = chip("stopped", "muted", "Stopped before it finished: its check never ran");
  else if (result.passed === null || result.passed === undefined) base = chip("no check", "muted", "The task defines no check command");
  else base = result.passed ? chip("pass", "ok") : chip("fail", "bad");
  if (result.usable === false) {
    let why = "Excluded from statistics";
    if (stopped) why = `${stopped[0].toUpperCase()}${stopped.slice(1)}: how it would have ended is unknown`;
    else if (result.timed_out) why = "The run timed out";
    else if (result.skills_changed?.length) why = `The run changed ${result.skills_changed.join(", ")}`;
    else if (result.edited_steps?.length) why = `Edited at a breakpoint (step ${result.edited_steps.join(", ")}): the verdict measures the edit, not the skill`;
    return h("span", { class: "verdict" }, base, chip("not counted", "warn", why));
  }
  return base;
}

// --- toasts ---------------------------------------------------------------------------

export function toast(message, { tone = "accent", href, action = "Open", timeout = 8000 } = {}) {
  const host = document.getElementById("toasts");
  const el = h("div", { class: `toast tone-${tone}`, role: "status" });
  const close = () => el.remove();
  append(el, [
    h("span", null, message),
    href ? h("a", { class: "btn btn-small", href, onclick: close }, action) : null,
    h("button", { class: "icon-btn", type: "button", "aria-label": "Dismiss", onclick: close }, "×"),
  ]);
  host.append(el);
  if (timeout) setTimeout(close, timeout);
}
