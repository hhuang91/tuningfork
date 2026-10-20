// The trash, as pages show it: moving something there, the banner of something there, and
// deleting it for good — the one step that cannot be undone, so it takes two clicks and says
// first what it removes. The server decides all of it (`ops/trash.py`).

import { api } from "./api.js";
import { guard } from "./auth.js";
import { h, plural, replace, when } from "./dom.js";
import { live } from "./live.js";

const NAMES = [
  ["refinements", "refinement"], ["runs", "run"], ["steps", "step"], ["experiments", "experiment"],
  ["samples", "sample"], ["token_streams", "token stream"], ["launches", "started harness"], ["jobs", "job"],
  ["scenarios", "scenario"], ["variants", "saved variant"],
];

/** What a permanent deletion removes, as a person reads it. */
export function goesText(counts) {
  const parts = NAMES.filter(([key]) => counts?.[key]).map(([key, name]) =>
    name === "started harness" && counts[key] !== 1 ? `${counts[key]} started harnesses` : plural(counts[key], name));
  return parts.length ? parts.join(", ") : "nothing";
}

/** The refinements that cite what a deletion removes, as the server says it
 *  (`ops.refine.citations`), and what else goes with it — an approval's record. */
export function citedText(goes) {
  return [...(goes?.cited_by || []).map((cited) => cited.says), ...(goes?.notes || [])].join("; ");
}

/** Move an item to the trash: one click, since it comes back. */
export function trashButton({ kind, id, label = "Move to trash", onDone }) {
  if (live.mode !== "live") return null;
  const note = h("span", { class: "form-error small" });
  const button = h("button", {
    class: "btn btn-small btn-quiet", type: "button", "data-control": true,
    onclick: guard(async () => {
      button.disabled = true;
      note.textContent = "";
      try {
        onDone(await api.post("/trash", { kind, id }));
      } catch (error) {
        note.textContent = error.hint ? `${error.message} — ${error.hint}` : error.message;
        button.disabled = false;
      }
    }),
  }, label);
  return h("span", { class: "controls" }, button, note);
}

/** Delete an item in the trash for good: the first click arms it and says what goes, the
 *  second — within six seconds — deletes it. */
export function purgeButton({ kind, id, goes = null, label = "Delete for good", onDone }) {
  if (live.mode !== "live") return null;
  const note = h("span", { class: "muted small" });
  const button = h("button", { class: "btn btn-small btn-danger", type: "button", "data-control": true }, label);
  let armed = null;
  const disarm = () => {
    clearTimeout(armed);
    armed = null;
    button.classList.remove("armed");
    button.textContent = label;
    note.textContent = "";
  };
  button.addEventListener("click", guard(async () => {
    if (!armed) {
      const counts = goes || await api.get(`/trash/preview?kind=${encodeURIComponent(kind)}&id=${encodeURIComponent(id)}`).catch(() => null);
      button.classList.add("armed");
      button.textContent = "Click again: delete for good";
      const cited = citedText(counts);
      note.textContent = counts ? `removes ${goesText(counts)}${cited ? `; ${cited}` : ""}; this cannot be undone` : "this cannot be undone";
      armed = setTimeout(disarm, 6000);
      return;
    }
    disarm();
    button.disabled = true;
    try {
      onDone(await api.post("/trash/purge", { kind, id }));
    } catch (error) {
      button.disabled = false;
      note.textContent = error.message;
    }
  }));
  return h("span", { class: "controls" }, button, note);
}

/** Bring an item back from the trash. */
export function restoreButton({ kind, id, onDone }) {
  if (live.mode !== "live") return null;
  const note = h("span", { class: "form-error small" });
  const button = h("button", {
    class: "btn btn-small", type: "button", "data-control": true,
    onclick: guard(async () => {
      button.disabled = true;
      try {
        onDone(await api.post("/trash/restore", { kind, id }));
      } catch (error) {
        note.textContent = error.message;
        button.disabled = false;
      }
    }),
  }, "Restore");
  return h("span", { class: "controls" }, button, note);
}

/** The banner of a page whose item is in the trash: since when, and the two ways out. */
export function trashedBanner({ kind, id, trashedAt, what, onRestored, onPurged }) {
  return h("div", { class: "banner banner-warn trashed-banner" },
    h("strong", null, `This ${what || kind} is in the trash`), ` since ${when(trashedAt)}. `,
    kind === "run" ? "It is left out of every list and pass rate; what is tied to it stays until it is deleted for good. " : "",
    kind === "refinement" ? "It is left out of the list and takes no new work. Its tests at a step and its full test runs stay until it is deleted for good, and then go with it. " : "",
    h("span", { class: "controls" },
      restoreButton({ kind, id, onDone: onRestored }),
      purgeButton({ kind, id, onDone: onPurged })));
}
