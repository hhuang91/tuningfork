// A chain of runs, link by link: a task run again and again, each run given the skill as the
// one before left it — how a harness's own self-improvement accumulates. Each link's verdict,
// who changed the skill after it, and the lines they changed (`tuningfork chain show`).

import { api, enc } from "../api.js";
import { ago, chip, crumbs, h, kv, loading, panel, plural, rateBar, replace, short, when } from "../dom.js";
import { costOf } from "../cost.js";
import { diffView } from "./refine.js";

/** The chains kept, newest first. */
export async function chainsView(root) {
  replace(root, loading());
  const chains = await api.get("/chains");
  replace(root,
    crumbs([["Runs", "#/runs"], ["chains"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Chains"),
      h("p", { class: "sub" }, "A task run again and again, each run given the skill as the one before left it: what a harness's own self-improvement makes of a skill, link by link."))),
    chains.length
      ? h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "chain"), h("th", null, "task"), h("th", null, "harness"),
            h("th", null, "links"), h("th", null, "passed, of those counted"), h("th", null, "state"), h("th", null, "started"))),
          h("tbody", null, chains.map((chain) => h("tr", { class: "clickable", onclick: () => (location.hash = `#/chains/${chain.id}`) },
            h("td", null, h("a", { class: "mono", href: `#/chains/${chain.id}` }, short(chain.id))),
            h("td", { class: "mono" }, chain.task_id || "—"),
            h("td", null, chain.harness || "—"),
            h("td", { class: "num" }, String(chain.links)),
            h("td", null, chain.rate ? rateBar(chain.rate) : h("span", { class: "muted" }, "none counted")),
            h("td", null, chip(chain.status, chain.status === "done" ? "ok" : "muted")),
            h("td", { class: "faint", title: when(chain.started_at) }, ago(chain.started_at)))))))
      : h("p", { class: "muted" }, "No chain yet: start one from Run a task, choosing a chain."));
}

/** One chain: how it was run, then each link — its verdict, who changed the skill after it and
 *  the lines they changed, and what the harness was told. */
export async function chainView(root, [id]) {
  replace(root, loading());
  const chain = await api.get(`/chains/${enc(id)}`);
  const links = chain.links || [];
  replace(root,
    crumbs([["Runs", "#/runs"], ["chains", "#/chains"], [short(chain.id)]]),
    h("div", { class: "page-head" }, h("div", null,
      h("h1", null, "A chain of runs", h("span", { class: "h-sub mono" }, chain.task_id || short(chain.id))),
      h("p", { class: "sub" }, `${plural(links.length, "link")} of ${chain.n}, each run given the skill as the one before left it.`))),
    h("div", { class: "stack" },
      panel("How it ran", kv([
        ["harness", `${chain.harness || "?"} · ${chain.model || "?"}`],
        ["set up with", Object.keys(chain.harness_options || {}).length
          ? h("span", { class: "mono small" }, Object.entries(chain.harness_options)
              .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · "))
          : null],
        ["told", chain.tell
          ? [h("span", null, "after each run, the check's verdict as the user's next turn: "), h("span", { class: "mono small" }, chain.told_template || "")]
          : "nothing between runs"],
        ["budget", chain.budget_seconds ? `${Math.round(chain.spent_seconds)} s of model calls spent of ${chain.budget_seconds}` : null],
        ["passed", chain.rate ? [rateBar(chain.rate), h("span", { class: "muted small" }, chain.not_counted ? `  ${chain.not_counted} not counted: the skill changed during them, or they timed out` : "")] : null],
        ["state", [chip(chain.status, chain.status === "done" ? "ok" : "muted"), chain.stopped ? h("span", { class: "muted" }, `  ${chain.stopped}`) : null]],
        ["cost", costOf({ chain: chain.id })],
        ["the skill it left", chain.final_skill ? h("span", { class: "mono" }, chain.final_skill) : null],
      ])),
      panel("Link by link", links.map(linkView), {
        meta: h("span", { class: "muted" }, "A link the skill changed during, or that timed out, is not counted in the rate."),
      })));
}

function linkView(link) {
  // A link whose check never ran — the task has none, or the run ended before it — has no
  // verdict: "failed" is for a check that said so.
  const verdictChip = link.exclusion === "timed out" ? chip("timed out", "warn")
    : link.passed === true ? chip("passed", "ok")
    : link.passed === false ? chip("failed", "bad")
    : chip("no verdict", "muted", link.exclusion || undefined);
  const changed = (link.changed || []).map((path) => path.split("/").slice(1).join("/") || path);
  const hunks = link.diff?.hunks || [];
  return h("div", { class: "stack", style: { gap: "6px", marginBottom: "14px" } },
    h("div", null, h("strong", null, `Link ${link.index}  `), verdictChip,
      link.usable === false && link.exclusion !== "timed out"
        ? [" ", chip("not counted", "muted", link.exclusion || undefined)] : null,
      h("span", { class: "muted small" }, `  ${Math.round(link.duration_s / 6) / 10} min · `),
      link.run_id ? h("a", { class: "mono small", href: `#/runs/${link.run_id}` }, `run ${short(link.run_id)}`) : null),
    link.told?.message ? h("p", { class: "small" }, h("span", { class: "label" }, "Told  "), link.told.message) : null,
    changed.length
      ? h("p", { class: "small" }, `The skill after it: ${changed.join(", ")} changed`,
          link.callers?.length ? h("span", { class: "muted" }, ` — by ${link.callers.join(", ")}`) : null)
      : h("p", { class: "faint small" }, "The skill after it: unchanged."),
    hunks.length
      ? h("details", { class: "fold" }, h("summary", null, "The lines it changed in SKILL.md"), diffView(hunks))
      : null);
}
