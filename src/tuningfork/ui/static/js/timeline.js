// A run as one conversation: its last context top to bottom, each step's answer where it
// happened. The server works out the places (`GET /runs/<id>/timeline`); this draws them.
//
// A step's answer comes from the recording, reasoning included: a harness can send an answer
// back without its reasoning, so the next context is not where to read it. A step opens in
// place, one at a time — its settings, the skills in its context, the files its call found, the
// experiments at it — and the exact context it was sent is one link away, on its own page: that
// is what a resample resends. The work that can start from a step opens beside the run, never
// inside it. The conversation ends with the run's verdict: what its task's check said, coloured
// by it — the only green or red on the page.

import { api, enc } from "./api.js";
import { chip, count, failure, firstLine, h, kv, loading, longText, millis, outcome, panel, plural, replace, short, toggle } from "./dom.js";
import { tokenWork } from "./live.js";
import { settle } from "./nav.js";
import { messageSize, messageText, messageView, noReasoning, outputView, toolNames } from "./payload.js";
import { startForm, startPanel, workBox } from "./start.js";
import { banners, detailsPanel, skillsSummary, summaryStrip, toolsOffered, workspacePanel } from "./stepinfo.js";
import { tokensPanel } from "./tokens.js";
import { LABELS } from "./words.js";

const EXPERIMENT = { resample: "Resample", ab: "A/B", "forking-tokens": "Forking tokens", screen: LABELS.screen };
// With this many of the person's turns or more — a long session — the rail starts with all
// but the last folded to their turn's line.
const FOLD_FROM = 3;

/** The timeline for `data`, a run's `/timeline`. Returns its element, its step rail, its layout
 *  (the run with the rail, or the work on a step, beside it) and its verdict strip — the verdict
 *  in a line, for the top of the page — and `start()` — to call once they are on the page —
 *  `update(data)` for a run still going, and `dispose()`. `focus` opens a step; `query` may name a
 *  form to open in its work beside the run (`start`), that work with no form (`work`), or
 *  `view=tokens` its answer's tokens. `scope` is the page's, for following work started here. */
export function timeline(data, { focus = null, query = new URLSearchParams(), scope = null } = {}) {
  const run = data.run;
  const base = `#/runs/${run.id}`;
  const recorded = Object.keys(run.skill_hashes || {}).length > 0;
  // Each step's section, by index: its marker, answer and details block, built on first open.
  const sections = new Map();
  // The steps not recorded whose rows are drawn: a live update draws only those it has not
  // (found in review: each one below a new step was drawn again, every update).
  const gapsShown = new Set();
  // The form asked for last: one asked for before it that loads later is not shown (found in
  // review: "A/B a wording…", then "Resample…" before the A/B form had loaded, left the A/B
  // form open while the address asked for the other — and on two steps, both open).
  let asking = 0;
  const element = h("div", { class: "tl" });
  const rail = h("nav", { class: "tl-rail", "aria-label": "Steps of the run" });
  const strip = h("div", { class: "verdict-strip", role: "status" });
  // The work on a step beside the run, never inside it: its actions and the form they open, in
  // a column where the rail is — the run read top to bottom, the work on one of its steps beside
  // it. Each step's work is kept as it was left; the column shows one step's at a time.
  const works = new Map();
  let workIdx = null;
  const workHead = h("div", { class: "tl-workcol-head" });
  const workBody = h("div", { class: "tl-workcol-body" });
  const workCol = h("aside", {
    class: "tl-workcol", hidden: true, "aria-label": "Work from a step of the run",
    // Escape hides it as its Hide does: on a narrow page it is a drawer over the run.
    onkeydown: (event) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      closeWork();
    },
  }, workHead, workBody);
  const layout = h("div", { class: "tl-layout" }, element, h("div", { class: "tl-side" }, rail, workCol));
  // Each step's call, read once: what both its details and its work are drawn from.
  const details = new Map();
  // The rail's way to the actions of the step being read, following it as the page scrolls.
  let railStep = null;
  const railWork = h("button", {
    class: "tl-rail-work", type: "button", hidden: true,
    onclick: () => {
      if (railStep !== null) workFrom(railStep);
    },
  });
  // What is being read marks its line in the rail: a step, or a message the harness added.
  const seen = new IntersectionObserver((changes) => {
    for (const change of changes) {
      if (!change.isIntersecting) continue;
      const { idx, entry } = change.target.dataset;
      markCurrent(entry !== undefined ? { entry: Number(entry) } : { idx: Number(idx) });
    }
  }, { rootMargin: "-30% 0px -60% 0px" });
  let state = data;
  // The run's verdict, always the timeline's last element: "Jump to the end" lands on it.
  let verdictNode = null;
  // The tools offered, always first: above the system prompt, as a request carries them.
  let toolsNode = null;
  let toolsDrawn = "";
  // The person's turns the rail folds, by each turn's entry, as the person folded or opened them
  // — kept across redraws — the exchanges drawn last, and the button folding them all.
  const folds = new Map();
  let exchanges = [];
  let foldAll = null;
  draw();

  return {
    element,
    rail,
    layout,
    strip,
    start() {
      if (focus === null || !sections.has(focus)) return;
      // The address as the page left it: a form in the step's work beside the run, that work with
      // no form, or the step's details.
      if (!query.get("start") && query.has("work")) revealWork(focus);
      else reveal(focus, { start: query.get("start"), query });
    },
    update,
    dispose() {
      seen.disconnect();
    },
  };

  // --- drawing ------------------------------------------------------------------------------

  function draw() {
    sections.clear();
    gapsShown.clear();
    // Drawn anew, each step's call is read anew as it opens.
    details.clear();
    seen.disconnect();
    verdictNode = verdictSection();
    toolsNode = toolsSection();
    replace(element, toolsNode, nodesFrom(0, () => true), verdictNode);
    drawStrip();
    drawRail();
    // The work beside the run is the run's, not its rows': drawn again, its rows say whose it is.
    markWork();
  }

  /** The tools the run's first call offered, before its system prompt: the model reads them as
   *  part of its prompt, though no message holds them. Where later calls offered others, it
   *  says which steps; where none were recorded, why. */
  function toolsSection() {
    const offered = state.tools_offered || [];
    const changed = toolChanges();
    toolsDrawn = toolsKey();
    if (offered.length) {
      const note = changed.length
        ? h("p", { class: "hint" }, "What the first call offered. Later calls offered others: ", changed.map((step, i) => {
            const names = [...step.extras_changed.tools_added.map((name) => `+${name}`), ...step.extras_changed.tools_removed.map((name) => `−${name}`)];
            return `${i ? "; " : ""}step ${step.idx}, ${names.join(" ")}`;
          }), ".")
        : null;
      return h("div", { class: "tl-tools" }, toolsOffered(offered, { note }));
    }
    const imported = state.run.imported;
    if (!imported) return h("p", { class: "tl-tools tl-note" }, "No tools offered: the model was sent none.");
    // An import that kept none: said in the tools' own line, the why a click away.
    return h("div", { class: "tl-tools" }, h("details", { class: "fold no-tools" },
      h("summary", null, "No tools recorded"),
      h("p", { class: "tl-note" }, imported.format === "hermes-export"
        ? "Hermes's export keeps none, so each step was rebuilt without the ones the model was offered. Import it again with “Take its tools from” naming a run recorded through the same harness, and its steps offer that run's tools."
        : "The file named none, so each step was rebuilt without any.")));
  }

  /** The steps whose call offered other tools than the call before. */
  function toolChanges() {
    return state.steps.filter((step) => step.extras_changed?.tools_added?.length || step.extras_changed?.tools_removed?.length);
  }

  function toolsKey() {
    return `${(state.tools_offered || []).length}|${toolChanges().map((step) => step.idx).join(",")}`;
  }

  /** How the run fared against its task's check, and why: the check, what it printed, and —
   *  where the check was run along the run — the call that made each line still there at the
   *  end. Green passed, red failed, amber a verdict every rate leaves out. */
  function verdictSection() {
    const { run, launch } = state;
    const verdict = verdictOf(run, launch, state.ended);
    const result = run.result || {};
    const output = String(result.check_output || "").trim();
    const made = (state.made_by || []).filter((line) => line.at_end);
    // A scenario's run that stopped where the model loaded another skill, as its scenario says.
    const stopped = result.stopped_at_skill;
    return h("section", { class: ["verdict-block", "tl-verdict", verdict.tone || "not-significant"], id: "verdict" },
      h("div", { class: "verdict-big" }, verdict.headline),
      verdict.detail ? h("p", { class: "hint" }, verdict.detail) : null,
      stopped ? h("p", { class: "hint" }, `It stopped where the model loaded ${stopped.skill}, after step ${stopped.after_step}: `
        + "its scenario studies one skill, so the check judged the workspace as it was then.") : null,
      kv([
        ["the check", launch?.check
          ? launch.check_kind === "shell"
            ? [h("span", { class: "mono" }, launch.check), h("span", { class: "muted" }, " — a shell command, run in the workspace")]
            : [launch.check, h("span", { class: "muted" }, " — judged on the final reply")]
          : null],
        ["it printed", output
          ? [result.check_output_cut
              ? h("p", { class: "hint" }, `${result.check_output_cut.toLocaleString()} characters it printed before these are not kept: a verdict keeps the last 2,000, and only they are tallied.`)
              : null,
            longText(output, { limit: 2000, lines: 30, open: true })]
          : result.passed === false ? h("span", { class: "muted" }, "nothing") : null],
      ]),
      made.length
        ? h("div", { class: "verdict-lines-wrap" },
            h("div", { class: "label" }, "Where each line came from: the check, run along the run"),
            h("ul", { class: "verdict-lines" }, made.map((line) => h("li", null,
              h("span", { class: "mono" }, line.line), " — ",
              line.made_by === null
                ? h("span", { class: "muted" }, "there from the start: work not done")
                : h("a", {
                    href: `${base}/${line.made_by}`,
                    onclick: (event) => {
                      event.preventDefault();
                      reveal(line.made_by);
                    },
                  }, `made by step ${line.made_by}'s call`)))))
        : null);
  }

  /** The verdict in a line, for under the run's title: what its end says — the headline, a
   *  line the check printed, and where the check ran along the run, the call that made it.
   *  Coloured as the end is; a run with no verdict says why. */
  function drawStrip() {
    const { run, launch } = state;
    const verdict = verdictOf(run, launch, state.ended);
    const output = String(run.result?.check_output || "").trim();
    const lines = output.split("\n").map((line) => line.trim()).filter(Boolean);
    const made = (state.made_by || []).filter((line) => line.at_end);
    const toEnd = (label) => h("a", {
      class: "verdict-strip-more", href: base,
      onclick: (event) => {
        event.preventDefault();
        verdictNode?.scrollIntoView({ block: "start" });
      },
    }, label);
    strip.className = ["verdict-strip", verdict.tone || "tone-none"].join(" ");
    // A failure says what the check printed — the first line it placed where it ran along the
    // run, else its last: a traceback's error, a script's closing word.
    const failed = run.result?.passed === false;
    const said = failed && lines.length ? made[0]?.line ?? lines[lines.length - 1] : null;
    replace(strip,
      h("strong", null, verdict.headline.endsWith(".") ? verdict.headline : `${verdict.headline}.`),
      // A space between the parts for whoever reads the line aloud: the row's gap is only drawn.
      " ",
      said
        ? h("span", null,
            lines.length > 1 ? `It printed ${plural(lines.length, "line")}; ${made.length ? "the first" : "the last"}: ` : "It printed ",
            h("code", null, said),
            made.length ? [", ", made[0].made_by === null ? "there from the start: work not done"
              : h("a", {
                  href: `${base}/${made[0].made_by}`,
                  onclick: (event) => {
                    event.preventDefault();
                    reveal(made[0].made_by);
                  },
                }, `made by step ${made[0].made_by}'s call`)] : null,
            ".")
        : null,
      // Why there is none, or why it is not counted.
      verdict.detail && (!failed || run.result?.usable === false) ? h("span", null, verdict.detail) : null,
      output || made.length ? toEnd("The check and all it printed ↓") : null);
  }

  /** The entries from `from` on, and the steps placed among them that `wanted` takes. */
  function nodesFrom(from, wanted) {
    const { entries, steps, threads } = state;
    const names = toolNames([...entries.map((entry) => entry.message), ...steps.map((step) => step.output || {})]);
    const at = group(steps.filter(wanted), (step) => step.position);
    const spawned = group(threads.filter((thread) => thread.after !== null), (thread) => thread.after);
    const gaps = [...state.missing].filter((idx) => !gapsShown.has(idx)).sort((a, b) => a - b);
    const nodes = [];
    for (let index = from; index <= entries.length; index += 1) {
      for (const step of at.get(index) || []) {
        while (gaps.length && gaps[0] < step.idx) {
          const idx = gaps.shift();
          gapsShown.add(idx);
          nodes.push(gap(idx));
        }
        nodes.push(stepSection(step, spawned.get(step.idx) || []));
      }
      if (index === entries.length) break;
      const entry = entries[index];
      // The step above shows its own answer, from the recording; the harness's copy of it is
      // under that answer when the two differ.
      if (entry.origin !== "copy") nodes.push(entryView(entry, names));
    }
    return nodes;
  }

  function entryView(entry, names) {
    const tag = entry.origin === "harness" && entry.after !== null
      ? h("span", { class: "tl-tag", title: `Added to the conversation by the harness after step ${entry.after}` }, `after step ${entry.after}`)
      : null;
    // A user message the harness wrote itself says so: no person said it.
    const note = entry.note ? chip(entry.note, "muted", "Written by the harness itself, not by a person") : null;
    const marks = (entry.changed_for_model || []).map((mark) =>
      h("a", {
        class: "tl-changed",
        href: `${base}/${mark.steps[0]}/context`,
        title: `The model saw a different version of this message at ${plural(mark.steps.length, "step")} ${mark.steps.join(", ")}: open its exact context`,
      }, chip(mark.by === "variant" ? "variant" : "edited", "change")));
    const withSkill = state.skills.some((span) => span.message_index === entry.index);
    // Short turns start open, and any carrying a skill; system prompts and long results fold.
    const open = withSkill || (entry.message.role !== "system" && messageSize(entry.message) < 1500);
    // A message the harness added after a step has a line of its own in the rail, under it, and
    // so has each of the conversation the run's first call continued, but its system prompt.
    const lined = (entry.origin === "harness" && entry.after !== null) || (entry.origin === "start" && entry.message.role !== "system");
    const node = h("div", { class: ["tl-entry", `tl-${entry.origin}`], dataset: lined ? { entry: entry.index } : null },
      messageView(entry.message, entry.index, {
        spans: state.skills, names, open, recorded, extra: [note, tag, ...marks], idPrefix: "t",
      }));
    if (lined) seen.observe(node);
    return node;
  }

  function gap(idx) {
    return h("div", { class: "tl-gap" }, `step ${idx} · not recorded — a call refused while the proxy was paused`);
  }

  /** A step's row: a line that opens its details under it, and — always in sight — the way to
   *  work from it, which opens its actions beside the run. */
  function stepSection(step, spawned) {
    const marker = h("button", {
      class: "tl-marker", type: "button", "aria-expanded": "false",
      title: `The model was sent the ${plural(step.position, "message")} above. Open for this step's details.`,
      onclick: () => toggleOpen(step.idx),
    }, markerParts(step));
    // A replayed step was answered from the recording: nothing starts there, and its button
    // says so rather than leave. Marked current while its work is the one beside the run.
    const workButton = h("button", {
      class: "btn btn-small tl-work", type: "button", disabled: Boolean(step.served), "aria-current": workIdx === step.idx ? "true" : null,
      title: step.served ? "Answered from the recording: nothing starts from here"
        : `Resample, A/B a wording, fork or refine the skill from step ${step.idx}`,
      onclick: () => workFrom(step.idx),
    }, "Work from here");
    const output = h("div", { class: ["tl-output", finalReply(step) && "tl-final"] }, outputParts(step));
    const block = h("div", { class: "tl-block", hidden: true });
    const section = h("section", {
      class: ["tl-step", step.kept === "retried" && "tl-retried"], id: `s${step.idx}`, dataset: { idx: step.idx },
    }, h("div", { class: "tl-head" }, marker, workButton), output, block, spawned.map(threadLink));
    sections.set(step.idx, { step, section, marker, workButton, output, block, ready: null });
    seen.observe(section);
    return section;
  }

  function markerParts(step) {
    const usage = step.usage || {};
    const took = step.timings?.total_ms;
    return [
      h("span", { class: "chevron", "aria-hidden": "true" }),
      h("span", { class: "tl-k" }, `Step ${step.idx}`),
      outcome(step.outcome),
      h("span", { class: "tl-meta" },
        usage.prompt_tokens !== undefined ? `${count(usage.prompt_tokens)} → ${count(usage.completion_tokens)} tokens` : null,
        took != null ? ` · ${millis(took)}` : null),
      flags(step),
    ];
  }

  function flags(step) {
    const replay = step.replay || {};
    const out = [];
    const none = noReasoning(step);
    if (none) out.push(none);
    if (step.served) out.push(chip("replayed", "muted", `Answered from the recording: step ${replay.source_idx} of run ${short(replay.source_run || "")}`));
    if (replay.live_from !== undefined && replay.live_from !== null) out.push(chip("live from here", "accent", "The fork went live at this step"));
    if (replay.mismatch) out.push(chip("left the recording", "warn"));
    if (step.variant) out.push(chip("variant", "change", "A variant wording was swapped in: the model saw it, the harness sent the original"));
    if (step.leaked) out.push(chip("original leaked", "warn"));
    if (step.edited.request) out.push(chip("context edited", "change", "Edited at a breakpoint before the model saw it"));
    if (step.edited.response) out.push(chip(`answer ${step.edited.response}`, "change", "At a breakpoint, before the harness acted on it"));
    if (step.kept === "retried") out.push(chip("not kept", "warn", `The harness called again from the same place, at step ${step.retried_by}`));
    if (step.kept === "differs") out.push(chip("kept differently", "warn", "The harness's copy of this answer is not what the model gave"));
    if (step.kept === "missing") out.push(chip("not kept", "warn", "No copy of this answer reached the conversation"));
    if (step.extras_changed) {
      const tools = [...step.extras_changed.tools_added.map((n) => `+${n}`), ...step.extras_changed.tools_removed.map((n) => `-${n}`)];
      out.push(chip("request changed", "muted", `Changed since the step before: ${step.extras_changed.fields.join(", ")}${tools.length ? ` (${tools.join(" ")})` : ""}`));
    }
    if (step.off_path) out.push(chip("off the path", "warn", "Recorded outside this run's conversation"));
    if (step.experiments.length) out.push(chip(plural(step.experiments.length, "experiment"), "accent"));
    return h("span", { class: "tl-flags" }, out);
  }

  /** The last step's text, when the run is over or the model said it was done. */
  function finalReply(step) {
    const text = step.output?.content;
    return step.last && typeof text === "string" && text.trim() && (state.ended || step.finish_reason === "stop");
  }

  function outputParts(step) {
    const parts = [];
    if (step.served) {
      parts.push(h("p", { class: "tl-note" }, `Answered from the recording — step ${step.replay?.source_idx} of run ${short(step.replay?.source_run || "")}; the model was not asked.`));
    }
    if (step.kept === "retried") {
      parts.push(h("p", { class: "tl-note" }, `The harness called again from the same place (step ${step.retried_by}): this answer never entered the conversation.`));
    }
    if (step.reasoning === "none") {
      parts.push(h("p", { class: "tl-note warn-text" }, step.after_failure
        ? "No reasoning before this tool call: the model acted at once, right after a tool call failed."
        : "No reasoning before this tool call, though the model reasons at other steps of this run."));
    }
    const final = finalReply(step);
    if (final) parts.push(h("div", { class: "tl-final-label" }, "Final reply"));
    const label = final ? null : "Content";
    let answer;
    if (step.delivered) {
      // A breakpoint changed the answer: the step's behaviour is the model's; the harness
      // acted on the edit.
      const shown = h("div");
      const controls = h("div", { class: "controls" });
      const show = (which) => {
        replace(controls, toggle([["model", "What the model produced"], ["delivered", "What the harness received"]], which, show));
        replace(shown, outputView(which === "delivered" ? step.delivered : step.output, { label }));
      };
      show("model");
      answer = h("div", null, controls, shown);
    } else {
      answer = outputView(step.output, { label });
    }
    if (step.outcome.kind === "error" || !step.output) {
      parts.push(answer);
    } else {
      // The same answer as the model's tokens: how sure it was of each, scored on demand.
      const shown = h("div");
      const tabs = h("div", { class: "controls tl-tabs" });
      const show = (which) => {
        replace(tabs, toggle([["answer", "Answer"], ["tokens", step.tokens ? "Tokens" : "Tokens…"]], which, show));
        replace(shown, which === "tokens"
          ? tokensPanel({
              run, idx: step.idx, scope,
              onProbe: (positions) => go(`${base}/${step.idx}?start=forking-tokens&positions=${positions.join(",")}`),
              focus: focus === step.idx && query.get("at") !== null ? Number(query.get("at")) : null,
            })
          : answer);
      };
      show(focus === step.idx && query.get("view") === "tokens" ? "tokens" : "answer");
      parts.push(tabs, shown);
    }
    if (step.kept === "differs" && step.copy !== null) {
      const copy = state.entries[step.copy];
      parts.push(h("details", { class: "fold" }, h("summary", null, "What the harness kept of it"),
        messageView(copy.message, copy.index, { idPrefix: "k", open: true })));
    }
    return parts;
  }

  function threadLink(thread) {
    const around = thread.placed === "time" ? " (around here, by when it began)" : "";
    return h("a", { class: "tl-thread", href: `#/runs/${thread.run_id}` },
      chip(thread.role, "muted"), ` ${short(thread.run_id)} · ${plural(thread.steps, "step")}${around} →`);
  }

  function drawRail() {
    const verdict = verdictOf(state.run, state.launch, state.ended);
    const names = toolNames([...state.entries.map((entry) => entry.message), ...state.steps.map((step) => step.output || {})]);
    // What came after each step — its tools' results, the harness's own notes, a person's next
    // turn — drawn in the conversation, so each one's line leads there.
    const added = group(state.entries.filter((entry) => entry.origin === "harness" && entry.after !== null), (entry) => entry.after);
    // The conversation the run's first call continued, as the page shows it: the person's turns,
    // the model's answers in it — none of them a call this run made — and what came after each.
    const before = state.entries.filter((entry) => entry.origin === "start" && entry.message.role !== "system");
    const earlier = before.filter((entry) => entry.message.role === "assistant").length;
    const lines = [
      ...before.map((entry) => (entry.message.role === "assistant" ? earlierAnswer(entry) : railEntry(entry, null, names))),
      ...state.steps.flatMap((step) => [
        h("a", {
          class: ["tl-rail-step", step.outcome.kind === "error" && "tl-rail-error", step.kept === "retried" && "tl-rail-retried"],
          href: `${base}/${step.idx}`, dataset: { idx: step.idx },
          // The rail moves through the run and opens nothing: a step opens from its own row.
          onclick: (event) => {
            event.preventDefault();
            goTo(step.idx);
          },
        }, h("span", { class: "mono" }, String(step.idx)), h("span", { class: "tl-rail-what" }, railLabel(step)),
        step.experiments.length
          ? h("span", { class: "tl-exp-dot", title: `${plural(step.experiments.length, "experiment")} at this step` })
          : null,
        step.variant || step.edited.request || step.edited.response ? h("span", { class: "tl-dot", title: "changed by tuningfork" }) : null),
        ...(added.get(step.idx) || []).map((entry) => railEntry(entry, step, names)),
      ]),
    ];
    exchanges = foldable(lines);
    foldAll = exchanges.length >= FOLD_FROM
      ? h("button", { class: "tl-rail-foldall", type: "button", onclick: () => foldEvery(exchanges.some((each) => !each.folded)) })
      : null;
    replace(rail,
      h("div", { class: "tl-rail-head", title: earlier ? "The run's first call continued a conversation: its answers come "
        + "first, unnumbered, then the steps this run recorded" : null },
        h("span", null, plural(state.steps.length, "step") + (earlier ? ` · after ${plural(earlier, "earlier answer")}` : "")), foldAll),
      lines,
      railWork,
      h("a", {
        class: ["tl-rail-verdict", verdict.tone], href: base, title: "What the task's check said: the end of the run",
        onclick: (event) => {
          event.preventDefault();
          verdictNode?.scrollIntoView({ block: "start" });
        },
      }, verdict.rail));
    drawFoldAll();
  }

  /** Each person's turn heads the lines after it, to the next one's: an exchange the rail folds
   *  to its turn's line. With many, all but the last start folded — or the one holding the step
   *  the address opened — unless the person folded or opened it themselves, which a redraw keeps. */
  function foldable(lines) {
    const found = [];
    for (const line of lines) {
      if (line.classList.contains("tl-rail-turn")) found.push({ turn: line, members: [], folded: false });
      else if (found.length) {
        found[found.length - 1].members.push(line);
        line.dataset.turn = found[found.length - 1].turn.dataset.entry;
      }
    }
    found.forEach((exchange, i) => {
      const key = Number(exchange.turn.dataset.entry);
      const button = exchange.turn.querySelector(".tl-rail-fold");
      button.classList.toggle("tl-rail-fold-none", !exchange.members.length);
      button.addEventListener("click", () => {
        folds.set(key, !exchange.folded);
        setFolded(exchange, !exchange.folded);
        drawFoldAll();
      });
      const holdsFocus = focus !== null && exchange.members.some((line) => line.dataset.idx !== undefined && Number(line.dataset.idx) === focus);
      setFolded(exchange, folds.has(key) ? folds.get(key) : found.length >= FOLD_FROM && i < found.length - 1 && !holdsFocus);
    });
    return found;
  }

  function setFolded(exchange, folded) {
    exchange.folded = folded && exchange.members.length > 0;
    for (const line of exchange.members) line.classList.toggle("tl-folded", exchange.folded);
    const button = exchange.turn.querySelector(".tl-rail-fold");
    button.setAttribute("aria-expanded", String(!exchange.folded));
    button.setAttribute("aria-label", exchange.folded ? "Unfold this turn" : "Fold this turn");
    replace(exchange.turn.querySelector(".tl-rail-folded"), exchange.folded ? foldedSummary(exchange.members) : null);
  }

  /** What a folded turn holds, said on its line: the model's earlier answers, and the steps. */
  function foldedSummary(members) {
    const answers = members.filter((line) => line.classList.contains("tl-rail-earlier")).length;
    const steps = members.filter((line) => line.dataset.idx !== undefined).map((line) => Number(line.dataset.idx));
    const said = [];
    if (answers) said.push(plural(answers, "answer"));
    if (steps.length) said.push(steps.length === 1 ? `step ${steps[0]}` : `steps ${steps[0]}–${steps[steps.length - 1]}`);
    return said.join(", ") || plural(members.length, "line");
  }

  function foldEvery(folded) {
    for (const exchange of exchanges) {
      folds.set(Number(exchange.turn.dataset.entry), folded);
      setFolded(exchange, folded);
    }
    drawFoldAll();
  }

  function drawFoldAll() {
    if (foldAll) replace(foldAll, exchanges.some((each) => !each.folded && each.members.length) ? "fold all" : "unfold all");
  }

  /** A line folded away opens with its turn: a step revealed, wherever it was asked for. */
  function unfoldAround(line) {
    if (!line?.classList.contains("tl-folded")) return;
    const exchange = exchanges.find((each) => each.turn.dataset.entry === line.dataset.turn);
    if (!exchange) return;
    folds.set(Number(exchange.turn.dataset.entry), false);
    setFolded(exchange, false);
    drawFoldAll();
  }

  function railLabel(step) {
    if (step.outcome.kind === "error") return "error";
    const calls = (step.output?.tool_calls || []).map((call) => call.function?.name || "?");
    if (calls.length) return calls.join(", ");
    return finalReply(step) ? "reply" : "text";
  }

  /** A message the harness added after `step`, on a line of its own — or, with no step, one of
   *  the conversation the run's first call continued. A tool's result goes under the answer that
   *  called it, by the tool's name, and so does a user message the harness wrote itself, by what
   *  it is; a person's turn starts a new exchange, unindented, in their words. Each leads to the
   *  message in the conversation, opened. */
  function railEntry(entry, step, names) {
    const message = entry.message;
    const href = step ? `${base}/${step.idx}` : base;
    const where = step ? `after step ${step.idx}` : "in the conversation the run's first call continued";
    if (message.role === "user" && !entry.note) {
      // Its fold, drawn by its look alone, so the line still reads as the person's words — in
      // full where the rail cuts them, for turns that all start alike.
      const words = entry.words ?? (firstLine(messageText(message), 80) || "(no text)");
      return h("div", { class: "tl-rail-turn", dataset: { entry: entry.index } },
        h("button", { class: "tl-rail-fold", type: "button", "aria-expanded": "true", "aria-label": "Fold this turn" }),
        h("a", { class: "tl-rail-turn-link", href, title: `A person's turn, ${where}: ${words}`, onclick: openMessage(entry) },
          h("span", { class: "tl-rail-who" }, "user"), h("span", { class: "tl-rail-what" }, words)),
        h("span", { class: "tl-rail-folded" }));
    }
    const what = message.role === "tool" ? `${names.get(message.tool_call_id) || "tool"} result` : entry.note || `${message.role} message`;
    return h("a", {
      class: "tl-rail-result", href, dataset: { entry: entry.index },
      title: entry.note ? `Written by the harness itself ${where}, not by a person` : step ? `${what}, added by the harness ${where}` : `${what}, ${where}`,
      onclick: openMessage(entry),
    }, h("span", { class: "tl-rail-arrow", "aria-hidden": "true" }, "↳"), h("span", { class: "tl-rail-what" }, what));
  }

  /** An answer of the conversation the run's first call continued, on a line as a step's is, but
   *  with no number: the model gave it in a call this run never made, so it has no step to open. */
  function earlierAnswer(entry) {
    const calls = (entry.message.tool_calls || []).map((call) => call.function?.name || "?");
    return h("a", {
      class: "tl-rail-step tl-rail-earlier", href: base, dataset: { entry: entry.index },
      title: "An answer in the conversation the run's first call continued: not a call this run made",
      onclick: openMessage(entry),
    }, h("span", { class: "mono", "aria-hidden": "true" }, "·"), h("span", { class: "tl-rail-what" }, calls.length ? calls.join(", ") : "reply"));
  }

  /** What a rail line for a message does: opens the message in the conversation, there. */
  function openMessage(entry) {
    return (event) => {
      event.preventDefault();
      const shown = element.querySelector(`#tm${entry.index}`);
      if (!shown) return;
      shown.open = true;
      shown.scrollIntoView({ block: "start" });
    };
  }

  function markCurrent({ idx = null, entry = null }) {
    let current = null;
    for (const link of rail.querySelectorAll(".tl-rail-step:not(.tl-rail-earlier)")) {
      link.classList.toggle("current", Number(link.dataset.idx) === idx);
      if (Number(link.dataset.idx) === idx) current = link;
    }
    for (const link of rail.querySelectorAll(".tl-rail-result, .tl-rail-turn, .tl-rail-earlier")) {
      link.classList.toggle("current", Number(link.dataset.entry) === entry);
      if (Number(link.dataset.entry) === entry) current = link;
    }
    // A line folded away under its turn: the turn's own line says where the page is.
    if (current?.classList.contains("tl-folded")) rail.querySelector(`.tl-rail-turn[data-entry="${current.dataset.turn}"]`)?.classList.add("current");
    // The step being read is the one the rail offers to work from; a message read after it
    // leaves it so.
    if (idx !== null && sections.has(idx)) {
      railStep = idx;
      railWork.hidden = false;
      replace(railWork, `Work from step ${idx}`);
    }
  }

  // --- a step, opened -------------------------------------------------------------------------

  function toggleOpen(idx) {
    const found = sections.get(idx);
    if (!found) return;
    const open = found.block.hidden;
    // The row clicked stays where it was on the screen, though the step open above it closes.
    const before = found.section.getBoundingClientRect().top;
    setOpen(found, open);
    const moved = found.section.getBoundingClientRect().top - before;
    if (moved) window.scrollBy(0, moved);
    // The address names the step opened; one closed gives it back to the work beside the run.
    settle(open ? `${base}/${idx}` : workAddress() ?? base);
  }

  /** One step open at a time: opening one closes the one open, which keeps what it held. */
  function setOpen(found, open) {
    if (open) for (const other of sections.values()) if (other !== found && !other.block.hidden) setOpen(other, false);
    found.block.hidden = !open;
    found.marker.setAttribute("aria-expanded", String(open));
    found.section.classList.toggle("tl-open", open);
    if (open && !found.ready) found.ready = build(found);
  }

  /** A step's call and what the server's check rules out, read once for its details and its work. */
  function detailOf(idx) {
    if (!details.has(idx)) {
      const asked = Promise.all([api.get(`/runs/${enc(run.id)}/steps/${idx}?context=0`), tokenWork()]);
      asked.catch(() => details.delete(idx));
      details.set(idx, asked);
    }
    return details.get(idx);
  }

  async function build(found) {
    const { step } = found;
    replace(found.block, loading("Loading the step…"));
    try {
      const [detail] = await detailOf(step.idx);
      const exact = `${base}/${step.idx}/context`;
      const here = state.skills.filter((span) => span.message_index < step.position);
      // What happened at the step, what it was, and the record, folded. The work from it is
      // beside the run (`openWork`).
      replace(found.block,
        banners(detail, { original: () => h("a", { href: exact }, "open the exact context to compare.") }),
        facts(detail, step, here),
        detailsPanel(detail),
        workspacePanel(detail),
        h("p", { class: "tl-exact" },
          h("a", { class: "btn btn-small", href: exact }, "Exact context this step was sent →"),
          h("span", { class: "muted small" }, " every message as the model saw it: what a resample resends")));
    } catch (error) {
      found.ready = null;
      replace(found.block, failure(error, "Could not load the step"));
    }
  }

  /** What the step was, in one block of labelled lines: its call, the skills in its context,
   *  and the experiments already run at it. */
  function facts(detail, step, here) {
    return h("div", { class: "tl-facts" }, kv([
      ["the call", summaryStrip(detail)],
      ["skills in its context", here.length ? skillsSummary(here, recorded) : null],
      ["experiments here", step.experiments.length
        ? h("div", { class: "tl-experiments" }, step.experiments.map((experiment) =>
            h("a", { href: `#/experiments/${experiment.id}` },
              chip(`${EXPERIMENT[experiment.kind] || experiment.kind}${experiment.reach === "end" ? `${experiment.kind === "screen" ? "," : ""} to the end` : ""} · ${experiment.samples}${experiment.n ? ` of ${experiment.n}` : ""}`,
                experiment.status === "done" ? "accent" : "muted", experiment.target || ""))))
        : null],
    ]));
  }

  /** The rail's way through the run: the step brought into sight, opened or not as it was. */
  function goTo(idx) {
    const found = sections.get(idx);
    if (!found) return;
    unfoldAround(rail.querySelector(`.tl-rail-step[data-idx="${idx}"]`));
    found.section.scrollIntoView({ block: "start" });
  }

  // --- the work beside the run ----------------------------------------------------------------

  /** Step `idx`'s work, built the first time it is asked for and kept: its actions, under them
   *  the slot a form opens in, and its address as it was left — with the form open there, if
   *  one is. */
  function workOf(idx) {
    if (works.has(idx)) return works.get(idx);
    const step = sections.get(idx)?.step;
    const slot = h("div", { class: "start-slot" });
    const node = h("div", { class: "tl-workcol-step", dataset: { idx } }, loading("Loading the step…"));
    const work = { idx, node, slot, ready: null, address: `${base}/${idx}?work` };
    work.ready = detailOf(idx).then(([detail, tokens]) => {
      // A replayed step was answered from the recording: there is no call to sample.
      replace(node, step?.served
        ? workBox(h("p", { class: "muted" }, "Answered from the recording: nothing starts from here."))
        : startPanel(detail, { go, work: tokens }), slot);
    }, (error) => {
      works.delete(idx);
      replace(node, failure(error, "Could not load the step"));
    });
    works.set(idx, work);
    return work;
  }

  /** The column beside the run on step `idx`'s work — the rail giving it its place — with a
   *  form there as it was left, if one was. */
  async function openWork(idx) {
    const work = workOf(idx);
    // Another step's work starts at its top.
    if (workIdx !== idx) workCol.scrollTop = 0;
    workIdx = idx;
    if (work.node.parentNode !== workBody) workBody.append(work.node);
    for (const other of workBody.children) other.hidden = other !== work.node;
    const step = sections.get(idx)?.step;
    replace(workHead,
      h("h2", { class: "tl-workcol-title" }, `Work from step ${idx}`),
      step ? h("span", { class: "tl-workcol-what mono" }, railLabel(step)) : null,
      // "Hide", not "Close": what the column holds is kept, and a form in it has its own Close.
      h("button", { class: "btn btn-quiet btn-small", type: "button", title: "The steps' list comes back; what this column holds is kept", onclick: closeWork }, "Hide"));
    workCol.hidden = false;
    rail.hidden = true;
    layout.classList.add("working");
    markWork();
    await work.ready;
    return work;
  }

  /** The rail back in its place; each step's work kept for when it is opened again. */
  function closeWork() {
    const found = sections.get(workIdx);
    // The step worked from stays where it was on the screen, though the run widens again.
    const before = found?.section.getBoundingClientRect().top;
    // Hidden from inside — its Hide, or Escape — the keys go on from the row's Work from here.
    const inside = workCol.contains(document.activeElement);
    workIdx = null;
    workCol.hidden = true;
    rail.hidden = false;
    layout.classList.remove("working");
    markWork();
    const moved = found ? found.section.getBoundingClientRect().top - before : 0;
    if (moved) window.scrollBy(0, moved);
    if (inside) found?.workButton.focus({ preventScroll: true });
    const open = [...sections.values()].find((other) => !other.block.hidden);
    settle(open ? `${base}/${open.step.idx}` : base);
  }

  /** Each row's "Work from here" marked current while its step's work is the one beside the run. */
  function markWork() {
    for (const [idx, found] of sections) {
      if (idx === workIdx) found.workButton.setAttribute("aria-current", "true");
      else found.workButton.removeAttribute("aria-current");
    }
  }

  /** The address of the work beside the run, as it was left: `?work`, or the form open in it. */
  function workAddress() {
    return workIdx === null ? null : works.get(workIdx)?.address ?? `${base}/${workIdx}?work`;
  }

  /** Step `idx`'s work beside the run — what its "Work from here" and the rail offer — the first
   *  of its actions ready for a key. */
  async function workFrom(idx) {
    const found = sections.get(idx);
    if (!found) return;
    // The row clicked stays where it was on the screen, though the run narrows beside the column.
    const before = found.section.getBoundingClientRect().top;
    const opening = openWork(idx);
    const moved = found.section.getBoundingClientRect().top - before;
    if (moved) window.scrollBy(0, moved);
    settle(workAddress());
    const work = await opening;
    work.node.querySelector(".start-actions a")?.focus({ preventScroll: true });
  }

  /** Step `idx` in sight — the run scrolled to it unless `scroll` is false — and, given `start`,
   *  that form open in its work beside the run; with none, its details open under it. */
  async function reveal(idx, { start = null, query: asked = new URLSearchParams(), scroll = true } = {}) {
    const found = sections.get(idx);
    if (!found) return;
    unfoldAround(rail.querySelector(`.tl-rail-step[data-idx="${idx}"]`));
    if (!start) {
      setOpen(found, true);
      found.section.scrollIntoView({ block: "start" });
      settle(`${base}/${idx}`);
      return;
    }
    const mine = ++asking;
    const work = workOf(idx);
    work.address = `${base}/${idx}?${asked.toString()}`;
    settle(work.address);
    const opening = openWork(idx);
    // The step worked from, in sight beside its form — once the column has taken its place, and
    // the run, narrowed, has settled.
    if (scroll) found.section.scrollIntoView({ block: "start" });
    await opening;
    // Asked for again since, or the step could not be read: the column says which.
    if (mine !== asking || works.get(idx) !== work) return;
    const waiting = loading("Preparing the form…");
    replace(work.slot, waiting);
    try {
      const form = await startForm(start, { run, idx }, asked, { go });
      if (mine !== asking) {
        // Asked for again since, elsewhere: this one never comes, and leaves no waiting behind.
        if (work.slot.firstChild === waiting) replace(work.slot);
        return;
      }
      replace(work.slot, form || failure(new Error(`There is no form called ${start}.`)));
      // The form in sight under the column's heading, by the column's own scroll — while the
      // column still shows this step — the run beside it staying where it is.
      if (workIdx === idx) workCol.scrollTo({ top: Math.max(0, work.slot.offsetTop - workHead.offsetHeight - 8) });
    } catch (error) {
      if (mine === asking) replace(work.slot, failure(error, "Could not open the form"));
      else if (work.slot.firstChild === waiting) replace(work.slot);
    }
  }

  /** Step `idx` in sight, its work beside the run as an address with `?work` left it: no form
   *  open in it. */
  function revealWork(idx) {
    const found = sections.get(idx);
    if (!found) return;
    unfoldAround(rail.querySelector(`.tl-rail-step[data-idx="${idx}"]`));
    openWork(idx);
    settle(workAddress());
    found.section.scrollIntoView({ block: "start" });
  }

  /** Where a form's links lead: a step of this run opens here; anywhere else, away. A link here
   *  is followed beside the run — from its work, or from the step's own answer — so the run
   *  stays where it is. */
  function go(href) {
    const [path, text = ""] = href.replace(/^#/, "").split("?", 2);
    const match = /^\/runs\/([^/]+)\/(\d+)$/.exec(path);
    if (!match || match[1] !== run.id) {
      location.hash = href;
      return;
    }
    const idx = Number(match[2]);
    const asked = new URLSearchParams(text);
    const start = asked.get("start");
    if (!start) {
      // "Close": the form goes, the step's work stays beside the run — and one still loading
      // never comes.
      asking += 1;
      const work = works.get(idx);
      if (work) {
        replace(work.slot);
        work.address = `${base}/${idx}?work`;
      }
      settle(`${base}/${idx}?work`);
      return;
    }
    reveal(idx, { start, query: asked, scroll: false });
  }

  // --- a run still going ----------------------------------------------------------------------

  /** A newer timeline of the same run. What is new is added below; an open step and an open
   *  form stay as they are. A timeline that no longer extends this one is drawn again. */
  function update(next) {
    const old = state;
    const extends_ = next.entries.length >= old.entries.length && next.steps.length >= old.steps.length
      && old.steps.every((step, i) => next.steps[i]?.idx === step.idx && next.steps[i].position === step.position);
    state = next;
    if (!extends_) {
      draw();
      return;
    }
    for (const step of next.steps) {
      const found = sections.get(step.idx);
      if (!found || (found.step.kept === step.kept && found.step.last === step.last)) continue;
      found.step = step;
      replace(found.marker, markerParts(step));
      replace(found.output, outputParts(step));
      found.output.classList.toggle("tl-final", Boolean(finalReply(step)));
    }
    for (const node of nodesFrom(old.entries.length, (step) => !sections.has(step.idx))) element.insertBefore(node, verdictNode);
    // The verdict comes once the harness has finished: drawn again, still last.
    const fresh = verdictSection();
    verdictNode.replaceWith(fresh);
    verdictNode = fresh;
    drawStrip();
    // A new step may have offered other tools: said again, left open if it was.
    if (toolsKey() !== toolsDrawn) {
      const open = Boolean(toolsNode.querySelector("details.tools-offered")?.open);
      const tools = toolsSection();
      const list = tools.querySelector("details.tools-offered");
      if (list) list.open = open;
      toolsNode.replaceWith(tools);
      toolsNode = tools;
    }
    drawRail();
  }
}

/** A run's verdict, as its page's end and its rail say it: in words, with its colour. A
 *  verdict every rate leaves out is amber, whatever the check said, and says why. */
function verdictOf(run, launch, ended) {
  const result = run.result;
  if (!result) {
    if (run.parent_run_id && ["subagent", "side", "thread"].includes(run.role)) {
      return {
        headline: "No verdict of its own", rail: "No verdict",
        detail: [`This conversation is a ${run.role} of `, h("a", { class: "mono", href: `#/runs/${run.parent_run_id}` }, `run ${short(run.parent_run_id)}`),
          ", and the check judged that run."],
      };
    }
    if (!launch) {
      // How it came — recorded live, imported — its title's chip says.
      return { headline: "No verdict", rail: "No verdict", detail: "No check ran: tuningfork did not start this run." };
    }
    if (!launch.check) return { headline: "No verdict", rail: "No verdict", detail: "The task defines no check." };
    return ended
      ? { headline: "No verdict", rail: "No verdict", detail: "The run ended without one." }
      : { headline: "No verdict yet", rail: "No verdict yet", detail: "The check runs when the harness finishes." };
  }
  if (result.passed === null || result.passed === undefined) {
    // A launch stopped before it finished keeps no verdict either, and says why: its check was
    // never run, since how it would have ended is unknown (found in review).
    if (result.cancelled) {
      const why = run.exclusion || result.exclusion || "stopped before it finished";
      return {
        tone: "tone-warn", headline: "No verdict: the run was stopped", rail: "Stopped",
        detail: `${why[0].toUpperCase()}${why.slice(1)}: how it would have ended is unknown, so its check never ran, and every pass rate leaves it out.`,
      };
    }
    return { headline: "No verdict", rail: "No verdict", detail: result.passed === null ? "The task defines no check." : null };
  }
  const said = result.passed ? "passed" : "failed";
  if (result.usable === false) {
    return {
      tone: "tone-warn", headline: `The check ${said}, but the run is not counted`, rail: "Not counted",
      detail: `Left out of every pass rate: ${run.exclusion || "the batch marked it unusable"}.`,
    };
  }
  return { tone: result.passed ? "tone-ok" : "tone-bad", headline: `The check ${said}`, rail: `Verdict: ${said}` };
}

function group(items, key) {
  const found = new Map();
  for (const item of items) {
    const at = key(item);
    if (!found.has(at)) found.set(at, []);
    found.get(at).push(item);
  }
  return found;
}
