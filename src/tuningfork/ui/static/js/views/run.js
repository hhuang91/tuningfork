// One run, as one conversation: its last context top to bottom, each step's answer where it
// happened (`js/timeline.js`), a step's details opening in place, its verdict at the end, and
// the run's own facts — how it was launched, what it left — folded above it.

import { api, enc } from "../api.js";
import { ago, chip, copyButton, crumbs, h, kv, loading, plural, replace, short, verdict, when } from "../dom.js";
import { costOf } from "../cost.js";
import { live } from "../live.js";
import { timeline } from "../timeline.js";
import { trashButton, trashedBanner } from "../trash.js";

export async function runView(root, [reference, focus], scope) {
  replace(root, loading());
  const data = await api.get(`/runs/${enc(reference)}/timeline`);
  const { run, launch } = data;
  // A batch run can be run again as it was launched: the same spec, eval and harness.
  const again = live.mode === "live" && launch?.kind === "batch" && launch.spec_path
    ? h("a", { class: "btn btn-small", href: `#/batch?run=${run.id}`, "data-control": true }, "Run this task again…")
    : null;
  const view = timeline(data, { focus: focus === undefined ? null : Number(focus), query: scope.query, scope });
  scope.onLeave(view.dispose);
  const toEnd = h("a", {
    class: "btn btn-small", href: `#/runs/${run.id}`,
    onclick: (event) => {
      event.preventDefault();
      view.element.lastElementChild?.scrollIntoView({ block: "end" });
    },
  }, "Jump to the end");

  replace(
    root,
    crumbs([["Runs", "#/runs"], run.task_id ? [run.task_id, `#/divergence/${enc(run.task_id)}`] : null, [`run ${short(run.id)}`]]),
    h("div", { class: "page-head" },
      h("div", null,
        h("h1", null, h("span", { class: "mono" }, `run ${short(run.id)}`), run.label ? h("span", { class: "h-sub" }, run.label) : null),
        h("p", { class: "sub" }, subLine(data))),
      h("div", { class: "actions" }, verdict(run.result), run.trashed_at ? null : again, toEnd, copyButton(run.id, "Copy id"),
        // A thread goes to the trash with its run, never alone.
        run.trashed_at || ["subagent", "side", "thread"].includes(run.role) ? null
          : trashButton({ kind: "run", id: run.id, onDone: () => runView(root, [reference, focus], scope) }))),
    run.trashed_at && run.trashed_with === run.id
      ? trashedBanner({
          kind: "run", id: run.id, trashedAt: run.trashed_at,
          onRestored: () => runView(root, [reference, focus], scope),
          onPurged: () => { location.hash = "#/trash"; },
        })
      : run.trashed_at
        ? h("div", { class: "banner banner-warn" }, h("strong", null, "In the trash"), ` since ${when(run.trashed_at)}, with `,
            h("a", { href: run.trashed_with ? `#/runs/${run.trashed_with}` : "#/trash" }, "what it went with"), ".")
        : null,
    changedBanner(data),
    aboutRun(data),
    data.steps.length
      ? h("div", { class: "tl-layout" }, view.element, view.rail)
      : h("div", { class: "panel empty" }, h("h2", null, "No steps recorded")),
  );
  view.start();

  // A run still going: what it does next is added below, as it is recorded. Its verdict is
  // written when its harness finishes, with no step of its own: a job's news may be it.
  let latest = data;
  let timer = null;
  scope.on((event) => {
    const step = event.type === "step" && event.run_id === run.id;
    const verdict = event.type === "job" && launch && !latest.run.result;
    if (!step && !verdict) return;
    clearTimeout(timer);
    timer = setTimeout(async () => {
      try {
        const next = await api.get(`/runs/${enc(run.id)}/timeline`);
        latest = next;
        if (scope.active) view.update(next);
      } catch {
        // the next event tries again
      }
    }, 500);
  });
  scope.onLeave(() => clearTimeout(timer));
}

function subLine({ run, steps }) {
  const parts = [plural(steps.length, "step"), run.harness, run.model, `started ${when(run.started_at)}`];
  if (run.fork_of) parts.push(`a fork of run ${short(run.fork_of)}, live from step ${run.fork_step}`);
  if (run.parent_run_id && run.role !== "main" && run.role !== "fork") {
    parts.push(`${run.role} of run ${short(run.parent_run_id)}${run.spawned_at != null ? `, from its step ${run.spawned_at}` : ""}`);
  }
  return parts.filter(Boolean).join(" · ");
}

/** Where tuningfork itself changed what happened: the model saw a variant, or a person edited
 *  a call. Said once, above it all; each place is marked on the timeline. */
function changedBanner({ steps }) {
  const variant = steps.filter((step) => step.variant).map((step) => step.idx);
  const context = steps.filter((step) => step.edited.request).map((step) => step.idx);
  const answer = steps.filter((step) => step.edited.response).map((step) => step.idx);
  const at = (list) => `${plural(list.length, "step")} ${list.join(", ")}`;
  const parts = [
    variant.length ? `the model saw a variant wording at ${at(variant)}` : null,
    context.length ? `a context was edited before the model saw it at ${at(context)}` : null,
    answer.length ? `an answer was changed before the harness acted on it at ${at(answer)}` : null,
  ].filter(Boolean);
  if (!parts.length) return null;
  return h("div", { class: "banner banner-edit" }, h("strong", null, "tuningfork changed this run's history: "), `${parts.join("; ")}.`);
}

/** Where a launch's harness worked, as it saw it: the sandbox's own paths, the same in every
 *  run — or, with no sandbox, the machine as it is. */
function sandboxWords(sandbox) {
  if (!sandbox) return h("span", { class: "muted" }, "none: the harness saw the machine as it is");
  return [h("span", { class: "mono" }, sandbox.workspace), h("span", { class: "muted" }, ", its home "),
    h("span", { class: "mono" }, sandbox.home), h("span", { class: "muted" }, ` — ${sandbox.kind || "a sandbox"}, the rest of the machine read-only`)];
}

function aboutRun({ run, launch, threads }) {
  const skills = Object.entries(run.skill_hashes || {});
  const result = run.result;
  return h("details", { class: "panel about-run" },
    h("summary", { class: "panel-head" }, h("h2", null, "About this run"),
      h("span", { class: "muted small" }, [launch ? `launched by tuningfork, ${launch.kind}` : "recorded", threads.length ? plural(threads.length, "other conversation") : null].filter(Boolean).join(" · "))),
    h("div", { class: "panel-body stack" },
      kv([
        ["task", run.task_id ? h("a", { class: "mono", href: `#/divergence/${enc(run.task_id)}` }, run.task_id) : null],
        ["label", run.label],
        ["model", run.model ? h("span", { class: "mono" }, run.model) : null],
        ["harness", run.harness],
        ["started", when(run.started_at)],
        ["last step", ago(run.updated_at)],
        ["fork of", run.fork_of
          ? [h("a", { class: "mono", href: `#/runs/${run.fork_of}` }, short(run.fork_of)), `, replayed up to step ${run.fork_step} and live from there`]
          : null],
        ["thread", run.parent_run_id && run.role !== "main" && run.role !== "fork"
          ? [`${run.role} of `, h("a", { class: "mono", href: `#/runs/${run.parent_run_id}` }, short(run.parent_run_id)),
              run.spawned_at !== null && run.spawned_at !== undefined ? `, from its step ${run.spawned_at}` : ""]
          : null],
        ["server", run.server_id ? h("span", { class: "mono" }, run.server_id.slice(0, 8)) : null],
        ["forked from", run.forked_from_run
          ? [h("a", { class: "mono", href: `#/runs/${run.forked_from_run}` }, short(run.forked_from_run)), ` at message depth ${run.forked_at_depth}`]
          : null],
        // The check, what it said and why: at the end of the run, below.
        ["how it ended", result?.returncode !== undefined || result?.duration_s !== undefined
          ? `harness exit ${result.returncode ?? "—"} after ${result.duration_s ?? "—"} s`
          : null],
        ["cost", costOf({ run: run.id })],
        ["workspace", launch?.workspace ? h("span", { class: "mono" }, launch.workspace) : null],
        ["sandbox", launch ? sandboxWords(launch.sandbox) : null],
        ["its clock", launch?.sandbox?.clock
          ? [h("span", { class: "mono" }, launch.sandbox.clock), h("span", { class: "muted" }, " — started there, whichever day the run was made")]
          : null],
        ["hidden from it", launch?.sandbox?.hidden?.length
          ? h("span", { class: "mono small" }, launch.sandbox.hidden.join(", "))
          : null],
        ["set up with", launch?.harness_settings && Object.keys(launch.harness_settings).length
          ? h("span", { class: "mono small" }, Object.entries(launch.harness_settings)
              .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · "))
          : null],
        ["run id", h("span", { class: "mono" }, run.id)],
      ]),
      skills.length
        ? h("div", null, h("div", { class: "label" }, "Skill files when the run started"),
            kv(skills.map(([path, digest]) => [h("span", { class: "mono" }, path), h("span", { class: "mono faint" }, digest.slice(0, 16))])))
        : null,
      threads.length
        ? h("div", null, h("div", { class: "label" }, "Other conversations in this launch"),
            h("div", { class: "stack", style: { gap: "4px" } }, threads.map((thread) =>
              h("a", { href: `#/runs/${thread.run_id}` }, chip(thread.role, "muted"),
                ` ${short(thread.run_id)} · ${plural(thread.steps, "step")}`,
                thread.after !== null ? h("span", { class: "muted small" }, ` · after step ${thread.after}${thread.placed === "time" ? ", by when it began" : ""}`) : null))))
        : null));
}
