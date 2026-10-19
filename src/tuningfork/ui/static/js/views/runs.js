// Every recorded run, grouped by task, with forks nested under the run they came from.

import { api, enc } from "../api.js";
import { guard } from "../auth.js";
import { ago, chip, h, loading, plural, rateBar, recordedLive, replace, short, verdict, when } from "../dom.js";
import { live } from "../live.js";

export async function runsView(root, _params, scope) {
  const selected = new Set();
  let filter = "";
  let tasks = [];
  let runs = [];

  const body = h("div", { class: "stack" }, loading());
  const compareBar = h("div", { class: "compare-bar", hidden: true });
  const search = h("input", {
    class: "field search",
    type: "search",
    placeholder: "Filter by run, label, task or model",
    "aria-label": "Filter runs",
    oninput: (event) => {
      filter = event.target.value.trim().toLowerCase();
      draw();
    },
  });

  replace(
    root,
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Runs"), h("p", { class: "sub" }, "Recorded runs by task. Forks sit under the run they branched from; tick two runs to diff them.")),
      h("div", { class: "actions" }, search,
        live.mode === "live" ? h("a", { class: "btn", href: "#/import", "data-control": true }, "Import a chat…") : null,
        live.mode === "live" ? h("a", { class: "btn btn-primary", href: "#/batch", "data-control": true }, "Run a task…") : null)),
    body,
    compareBar,
  );

  async function load() {
    [tasks, runs] = await Promise.all([api.get("/tasks"), api.get("/runs?limit=1000")]);
    draw();
  }

  function draw() {
    if (!runs.length) {
      replace(body, emptyState());
      drawCompareBar();
      return;
    }
    const matches = (run) =>
      !filter || [run.id, run.label, run.task_id, run.model].some((value) => value && String(value).toLowerCase().includes(filter));
    const byTask = new Map();
    for (const run of runs.filter(matches)) {
      const key = run.task_id ?? "";
      if (!byTask.has(key)) byTask.set(key, []);
      byTask.get(key).push(run);
    }
    if (!byTask.size) {
      replace(body, h("p", { class: "muted" }, "No runs match the filter."));
      return;
    }
    const sections = [];
    for (const summary of tasks) {
      const members = byTask.get(summary.task_id ?? "");
      if (members) sections.push(taskSection(summary, members));
    }
    replace(body, sections);
    drawCompareBar();
  }

  function taskSection(summary, members) {
    const title = summary.task_id ? h("span", { class: "mono" }, summary.task_id) : h("span", { class: "muted" }, "No task — recorded ad hoc");
    const conditions = summary.conditions || [];
    // Runs under another wording, model or harness are another sample: each gets its own
    // rate, and the task as a whole none.
    const split = summary.task_id && conditions.length > 1;
    const meta = [h("span", { class: "muted" }, plural(summary.runs, "run"))];
    if (split) {
      meta.push(chip(`${conditions.length} conditions`, "accent", "Different wordings, models or harnesses. Each has its own pass rate below; pooling them would measure none of them."));
    } else {
      meta.push(...rateMeta(summary));
    }
    const actions = [];
    if (summary.task_id && summary.runs >= 2 && !split) {
      actions.push(h("a", { class: "btn btn-small", href: `#/divergence/${enc(summary.task_id)}` }, "Where do they diverge?"));
    }
    return h(
      "section",
      { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", null, title), h("div", { class: "panel-meta" }, meta), actions.length ? h("div", { class: "actions" }, actions) : null),
      summary.check ? passesWhen(summary) : null,
      split
        ? conditions.map((condition) => conditionBlock(summary, condition, members))
        : h("div", { class: "tree" }, treeRows(members)),
    );
  }

  // How a task's runs pass, as its latest batch judged them: what "check passed" means.
  function passesWhen(summary) {
    const { kind, text } = summary.check;
    return h("div", { class: "passes-when" },
      h("span", { class: "label" }, "Passes when"),
      kind === "shell"
        ? [h("code", { class: "mono" }, text), h("span", { class: "muted small" }, " — a shell command, run in each run's workspace")]
        : [h("span", null, text), h("span", { class: "muted small" }, " — judged on the final reply")],
      summary.checks > 1
        ? chip(`the check changed: ${summary.checks} versions`, "warn", "Batches of this task ran different checks; the latest is shown. A rate pooled over two checks measures neither.")
        : null);
  }

  function rateMeta(counts) {
    const meta = [];
    if (counts.passed) {
      meta.push(h("span", { class: "task-rate" }, h("span", { class: "muted" }, "check passed"), rateBar(counts.passed, "ok")));
    }
    if (counts.uncounted) {
      meta.push(chip(`${counts.uncounted} not counted`, "warn", "Runs that timed out, changed their own skill, failed at the server, or were edited at a breakpoint are left out of the rate"));
    }
    return meta;
  }

  function conditionBlock(summary, condition, members) {
    const ids = new Set(condition.run_ids);
    const own = members.filter((run) => ids.has(run.id));
    if (!own.length) return null; // filtered out
    const file = condition.skill_file ? condition.skill_file.split(/[\\/]/).pop() : null;
    const label = [
      condition.harness ? chip(condition.harness, "muted") : null,
      condition.model ? h("span", { class: "mono small" }, condition.model) : null,
      h("span", { class: "mono small", title: "A fingerprint of the skill files these runs were given" },
        condition.skill_version ? `skill ${condition.skill_version}` : "skill not recorded"),
      file ? chip(`SKILL.md from ${file}`, "skill", condition.skill_file) : null,
    ];
    return h(
      "div",
      { class: "condition" },
      h("div", { class: "condition-head" },
        h("div", { class: "condition-label" }, label),
        h("div", { class: "panel-meta" }, h("span", { class: "muted" }, plural(condition.runs, "run")), rateMeta(condition)),
        condition.runs >= 2
          ? h("a", { class: "btn btn-small push", href: `#/divergence/${enc(summary.task_id)}@${condition.id}` }, "Where do they diverge?")
          : null),
      h("div", { class: "tree" }, treeRows(own)),
    );
  }

  function treeRows(members) {
    const ids = new Set(members.map((run) => run.id));
    const children = new Map();
    const roots = [];
    for (const run of members) {
      if (run.forked_from_run && ids.has(run.forked_from_run)) {
        if (!children.has(run.forked_from_run)) children.set(run.forked_from_run, []);
        children.get(run.forked_from_run).push(run);
      } else {
        roots.push(run);
      }
    }
    const rows = [];
    const visit = (run, depth) => {
      rows.push(runRow(run, depth));
      for (const child of children.get(run.id) || []) visit(child, depth + 1);
    };
    roots.forEach((run) => visit(run, 0));
    return rows;
  }

  function runRow(run, depth) {
    const box = h("input", {
      type: "checkbox",
      checked: selected.has(run.id),
      "aria-label": `Select run ${short(run.id)} to compare`,
      onchange: (event) => {
        if (event.target.checked) selected.add(run.id);
        else selected.delete(run.id);
        drawCompareBar();
      },
    });
    return h(
      "div",
      { class: "run-row" },
      box,
      h("div", { class: "run-main", style: { "--depth": String(Math.min(depth, 4)) } },
        depth ? h("span", { class: "fork-glyph", "aria-hidden": "true" }, "↳") : null,
        h("a", { class: "mono run-id", href: `#/runs/${run.id}` }, short(run.id)),
        h("span", { class: "run-label", title: run.label || "" }, run.label || ""),
        // Nothing launched it, and it was not imported: a harness's calls through the proxy.
        run.recorded === "live" ? recordedLive() : null,
        run.forked_from_run
          ? chip(`fork @ ${run.forked_at_depth}`, "skill",
            `Branched from run ${short(run.forked_from_run)}: the two share their first ${plural(run.forked_at_depth, "message")}, and part there`)
          : null),
      h("span", { class: "muted" }, plural(run.step_count, "step")),
      h("span", { class: "muted mono small run-model", title: run.model || "" }, run.model || ""),
      h("span", { class: "faint", title: when(run.started_at) }, ago(run.started_at)),
      h("span", { class: "run-verdict" }, verdict(run.result)),
    );
  }

  function drawCompareBar() {
    const ids = [...selected];
    compareBar.hidden = ids.length === 0;
    let text = `${ids.length} runs selected — diff compares two`;
    if (ids.length === 1) text = "Tick one more run to diff it with this one, or move it to the trash";
    if (ids.length === 2) text = `Diff ${short(ids[0])} with ${short(ids[1])}`;
    replace(
      compareBar,
      h("span", null, text),
      ids.length === 2 ? h("a", { class: "btn btn-primary btn-small", href: `#/diff/${ids[0]}/${ids[1]}` }, "Diff") : null,
      live.mode === "live" ? h("button", {
        class: "btn btn-small", type: "button", "data-control": true,
        onclick: guard(async (event) => {
          event.target.disabled = true;
          const refused = [];
          for (const id of ids) {
            try {
              await api.post("/trash", { kind: "run", id });
              selected.delete(id);
            } catch (error) {
              refused.push(`${short(id)}: ${error.message}`);
            }
          }
          await load();
          if (refused.length) replace(compareBar, h("span", { class: "form-error" }, `Not moved — ${refused.join("; ")}`));
        }),
      }, ids.length === 1 ? "Move to trash" : `Move ${ids.length} to trash`) : null,
      h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => { selected.clear(); draw(); } }, "Clear"),
    );
  }

  await load();

  // While a harness runs through the proxy, its steps land here as they are recorded.
  let timer = null;
  scope.on((event) => {
    if (event.type !== "step") return;
    clearTimeout(timer);
    timer = setTimeout(() => load().catch(() => {}), 700);
  });
  scope.onLeave(() => clearTimeout(timer));
}

function emptyState() {
  const base = `${location.origin}/v1`;
  return h(
    "div",
    { class: "panel empty" },
    h("h2", null, "No runs recorded yet"),
    live.mode === "live"
      ? h("ol", { class: "setup" },
          h("li", null, "Point your harness's OpenAI-compatible base URL at this proxy: ", h("code", { class: "mono" }, base),
            ". Every model call it makes is recorded as a step."),
          h("li", null, "Or let tuningfork launch it: ", h("a", { href: "#/batch" }, "run a task"),
            " from its evals.json, each run in a fresh workspace and home."),
          h("li", null, "Then find where runs diverge, and resample or A/B a wording at that step."))
      : h("p", null, "This store has no recordings. Start `tuningfork serve` and run a harness through it."),
  );
}
