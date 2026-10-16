// What work cost: model calls, their tokens and the time they took, counted from the records
// (`GET /spend`, `tuningfork spend`) — for a run, a job, an experiment, a chain or a task.

import { api, query } from "./api.js";
import { count, h, plural, replace } from "./dom.js";

/** A row of `GET /spend`, in words. */
export function costText(row) {
  if (!row || !row.calls) return "no model calls recorded";
  const time = row.wall_s >= 60 ? `${(row.wall_s / 60).toFixed(1)} min` : `${Math.round(row.wall_s)} s`;
  return `${plural(row.calls, "model call")}${row.failed ? `, ${count(row.failed)} failed` : ""} · `
    + `${count(row.prompt_tokens)} tokens in, ${count(row.completion_tokens)} out · ${time} of calls`;
}

/** A cost that fills itself in once the proxy answers, so a page never waits on it: `named` is
 *  what to count — `{ run: id }`, `{ job: id }`, `{ experiment: id }`, `{ chain: id }`. */
export function costOf(named) {
  const shown = h("span", { class: "muted" }, "counting…");
  api.get(`/spend${query(named)}`)
    .then((data) => replace(shown, costText(data.rows?.[0])))
    .catch(() => replace(shown, "—"));
  return shown;
}
