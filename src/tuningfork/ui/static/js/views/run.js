// One run, as one conversation: its verdict in a line under its title, its last context top to
// bottom, each step's answer where it happened (`js/timeline.js`), a step's details opening in
// place, its verdict at the end, and the run's own facts — how it was launched, what it left —
// folded above it.

import { api, enc } from "../api.js";
import { ago, chip, copyButton, crumbs, foldPanel, h, importedRun, kv, loading, plural, recordedLive, replace, short, when } from "../dom.js";
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
        // How it came to be recorded, said once: a chip where nothing launched it.
        h("h1", null, h("span", { class: "mono" }, `run ${short(run.id)}`), run.label ? h("span", { class: "h-sub" }, run.label) : null,
          run.recorded === "live" ? h("span", { class: "h-sub" }, recordedLive())
            : run.recorded === "imported" ? h("span", { class: "h-sub" }, importedRun()) : null),
        h("p", { class: "sub facts" }, facts(data))),
      h("div", { class: "actions" }, run.trashed_at ? null : again, toEnd, copyButton(run.id, "Copy id"),
        // Any run's conversation can become a scenario: runs of its skill continuing it. A
        // thread's calls are its own, never a task's.
        !run.trashed_at && live.mode === "live" && !["subagent", "side", "thread"].includes(run.role) && data.steps.length
          ? h("a", { class: "btn btn-small", href: `#/runs/${run.id}/scenario`, "data-control": true }, "Make a scenario…") : null,
        // A thread goes to the trash with its run, and a fork an experiment drew with the
        // experiment: never alone.
        run.trashed_at || ["subagent", "side", "thread"].includes(run.role) ? null
          : launch?.experiment_id
            ? h("a", { class: "btn btn-small btn-quiet", href: `#/experiments/${launch.experiment_id}`,
                title: "A fork an experiment drew is one of its samples: it goes to the trash with the experiment" },
              "Its experiment")
            : trashButton({ kind: "run", id: run.id, onDone: () => runView(root, [reference, focus], scope) }))),
    // What the run's end says, in a line: the take-away before the conversation.
    view.strip,
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
      ? view.layout
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

  // A launch no job watches — a batch or a fork from the command line — saves its verdict with
  // no event this page hears (found in review). While one is still to come the page asks itself:
  // every few seconds while the launch runs, and a few times more once it has ended, since the
  // verdict is written just after. A thread gets none of its own: the check judged its run.
  let poller = null;
  let after = 0;
  const awaiting = () => launch && !launch.job_id && !latest.run.result
    && !["subagent", "side", "thread"].includes(latest.run.role);
  const poll = async () => {
    poller = null;
    if (!scope.active || !awaiting()) return;
    try {
      const next = await api.get(`/runs/${enc(run.id)}/timeline`);
      latest = next;
      if (scope.active) view.update(next);
    } catch {
      // the next try asks again
    }
    if (latest.ended) after += 1;
    if (scope.active && awaiting() && after < 10) poller = setTimeout(poll, 3000);
  };
  if (awaiting()) poller = setTimeout(poll, 3000);
  scope.onLeave(() => clearTimeout(poller));
}

/** The run's facts under its title, each a short item that wraps whole: its size, harness,
 *  model and start, then where it came from — the chip beside the title says how. */
function facts({ run, steps, launch, entries }) {
  const items = [plural(steps.length, "step"), run.harness, run.model, `started ${when(run.started_at)}`];
  // A first call that continued a conversation: what came before it, as the rail lists it.
  const before = (entries || []).filter((entry) => entry.origin === "start" && entry.message.role !== "system");
  if (before.some((entry) => entry.message.role === "assistant")) {
    const turns = before.filter((entry) => entry.message.role === "user" && !entry.note).length;
    items.push(`continues an earlier chat: ${plural(before.length, "message")}, ${plural(turns, "turn")}`);
  }
  if (run.fork_of) {
    items.push(["fork of ", h("a", { class: "mono", href: `#/runs/${run.fork_of}` }, `run ${short(run.fork_of)}`), `, live from step ${run.fork_step}`]);
  }
  if (launch?.context) {
    items.push(`continues its scenario's chat: ${plural(launch.context.messages, "message")}`
      + (launch.context.in_turn ? `, then ${plural(launch.context.in_turn, "message")} of its turn, already done` : ""));
  }
  if (run.parent_run_id && run.role !== "main" && run.role !== "fork") {
    items.push([`${run.role} of `, h("a", { class: "mono", href: `#/runs/${run.parent_run_id}` }, `run ${short(run.parent_run_id)}`),
      run.spawned_at != null ? `, from its step ${run.spawned_at}` : ""]);
  }
  return items.filter(Boolean).map((item, i) => [i ? " · " : null, h("span", { class: "fact" }, item)]);
}

/** What an imported run's conversation came from. */
function importedFrom(imported) {
  const from = imported.format === "hermes-export" ? "a Hermes export" : imported.format === "hermes-snapshot"
    ? "a Hermes session file" : "a list of messages";
  return `${from}${imported.session_id ? ` (session ${imported.session_id})` : ""}`;
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
  // A run nothing launched has a chip beside its title saying how it came: this says only what
  // the chip cannot.
  const meta = [launch ? `started by tuningfork, ${launch.kind}` : run.imported ? `from ${importedFrom(run.imported)}` : null,
    threads.length ? plural(threads.length, "other conversation") : null].filter(Boolean).join(" · ");
  return foldPanel("About this run", () => h("div", { class: "stack" },
      kv([
        ["task", run.task_id ? h("a", { class: "mono", href: `#/divergence/${enc(run.task_id)}` }, run.task_id) : null],
        ["label", run.label],
        ["model", run.model ? h("span", { class: "mono" }, run.model) : null],
        ["harness", run.harness],
        ["started", when(run.started_at)],
        ["last step", ago(run.updated_at)],
        ["imported from", run.imported ? importedFrom(run.imported) : null],
        ["its conversation", launch?.context
          ? [`${plural(launch.context.messages, "message")} of its scenario's chat`
              + (launch.context.in_turn ? `, then ${plural(launch.context.in_turn, "message")} of its turn, already done` : ""),
            h("span", { class: "muted" }, ` — ${launch.context.file}`)]
          : null],
        ["fork of", run.fork_of
          ? [h("a", { class: "mono", href: `#/runs/${run.fork_of}` }, short(run.fork_of)), `, replayed up to step ${run.fork_step} and live from there`]
          : null],
        ["thread", run.parent_run_id && run.role !== "main" && run.role !== "fork"
          ? [`${run.role} of `, h("a", { class: "mono", href: `#/runs/${run.parent_run_id}` }, short(run.parent_run_id)),
              run.spawned_at !== null && run.spawned_at !== undefined ? `, from its step ${run.spawned_at}` : ""]
          : null],
        ["server", run.server_id ? h("span", { class: "mono" }, run.server_id.slice(0, 8)) : null],
        ["branched from", run.forked_from_run
          ? [h("a", { class: "mono", href: `#/runs/${run.forked_from_run}` }, short(run.forked_from_run)),
             `: the two share their first ${plural(run.forked_at_depth, "message")}`]
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
      // The skill it had, kept as one of tuningfork's own: the Skills page's Add a skill, at this run.
      // Offered where there is one to give — a copy it kept, one its import found — or may be: a
      // run recorded live, whose conversation may show one.
      live.mode === "live" && !["subagent", "side", "thread"].includes(run.role)
        && (skills.length || run.imported?.skills?.length || run.recorded === "live")
        ? h("p", { class: "hint" }, h("a", { href: `#/add-skill?run=${run.id}`, "data-control": true }, "Keep its skill…"),
            " — the copy it kept, or the SKILL.md its model read, added to Skills as a copy of tuningfork's own.")
        : null,
      threads.length
        ? h("div", null, h("div", { class: "label" }, "Other conversations its harness had in this run"),
            h("div", { class: "stack", style: { gap: "4px" } }, threads.map((thread) =>
              h("a", { href: `#/runs/${thread.run_id}` }, chip(thread.role, "muted"),
                ` ${short(thread.run_id)} · ${plural(thread.steps, "step")}`,
                thread.after !== null ? h("span", { class: "muted small" }, ` · after step ${thread.after}${thread.placed === "time" ? ", by when it began" : ""}`) : null))))
        : null), { className: "about-run", meta: meta ? h("span", { class: "muted small" }, meta) : null });
}
