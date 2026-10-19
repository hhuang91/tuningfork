// A run as one conversation: its last context top to bottom, each step's answer where it
// happened. The server works out the places (`GET /runs/<id>/timeline`); this draws them.
//
// A step's answer comes from the recording, reasoning included: a harness can send an answer
// back without its reasoning, so the next context is not where to read it. A step opens in
// place — its settings, the skills in its context, the files its call found, the experiments
// at it, and the work that can start from it — and the exact context it was sent is one link
// away, on its own page: that is what a resample resends. The conversation ends with the run's
// verdict: what its task's check said, coloured by it — the only green or red on the page.

import { api, enc } from "./api.js";
import { chip, count, failure, firstLine, h, kv, loading, longText, millis, outcome, plural, replace, short, toggle } from "./dom.js";
import { settle } from "./nav.js";
import { messageSize, messageText, messageView, noReasoning, outputView, toolNames } from "./payload.js";
import { startForm, startPanel } from "./start.js";
import { banners, detailsPanel, skillsSummary, summaryStrip, toolsOffered, workspacePanel } from "./stepinfo.js";
import { tokensPanel } from "./tokens.js";

const EXPERIMENT = { resample: "Resample", ab: "A/B", "forking-tokens": "Forking tokens" };

/** The timeline for `data`, a run's `/timeline`. Returns its element and its step rail, and
 *  `start()` — to call once both are on the page — `update(data)` for a run still going,
 *  and `dispose()`. `focus` opens a step; `query` may name a form to open in it, or
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
  draw();

  return {
    element,
    rail,
    start() {
      if (focus !== null && sections.has(focus)) reveal(focus, { start: query.get("start"), query });
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
    seen.disconnect();
    verdictNode = verdictSection();
    toolsNode = toolsSection();
    replace(element, toolsNode, nodesFrom(0, () => true), verdictNode);
    drawRail();
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
    return h("p", { class: "tl-tools tl-note" }, !imported
      ? "No tools offered: the model was sent none."
      : imported.format === "hermes-export"
        ? "No tools recorded: Hermes's export keeps none, so each step was rebuilt without the ones the model was offered. Imported again with Tools from naming a run recorded through the same harness, its steps take that run's."
        : "No tools recorded: the file named none, so each step was rebuilt without any.");
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
    return h("section", { class: ["verdict-block", "tl-verdict", verdict.tone || "not-significant"], id: "verdict" },
      h("div", { class: "verdict-big" }, verdict.headline),
      verdict.detail ? h("p", { class: "hint" }, verdict.detail) : null,
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
      }, chip(mark.by === "variant" ? "variant" : "edited", mark.by === "variant" ? "skill" : "response")));
    const withSkill = state.skills.some((span) => span.message_index === entry.index);
    // Short turns start open, and any carrying a skill; system prompts and long results fold.
    const open = withSkill || (entry.message.role !== "system" && messageSize(entry.message) < 1500);
    // A message the harness added after a step has a line of its own in the rail, under it.
    const added = entry.origin === "harness" && entry.after !== null;
    const node = h("div", { class: ["tl-entry", `tl-${entry.origin}`], dataset: added ? { entry: entry.index } : null },
      messageView(entry.message, entry.index, {
        spans: state.skills, names, open, recorded, extra: [note, tag, ...marks], idPrefix: "t",
      }));
    if (added) seen.observe(node);
    return node;
  }

  function gap(idx) {
    return h("div", { class: "tl-gap" }, `step ${idx} · not recorded — a call refused while the proxy was paused`);
  }

  function stepSection(step, spawned) {
    const marker = h("button", {
      class: "tl-marker", type: "button", "aria-expanded": "false",
      title: `The model was sent the ${plural(step.position, "message")} above. Open for this step's details, and work from it.`,
      onclick: () => toggleOpen(step.idx),
    }, markerParts(step));
    const output = h("div", { class: ["tl-output", finalReply(step) && "tl-final"] }, outputParts(step));
    const block = h("div", { class: "tl-block", hidden: true });
    const section = h("section", {
      class: ["tl-step", step.kept === "retried" && "tl-retried"], id: `s${step.idx}`, dataset: { idx: step.idx },
    }, marker, output, block, spawned.map(threadLink));
    sections.set(step.idx, { step, section, marker, output, block, slot: null, ready: null });
    seen.observe(section);
    return section;
  }

  function markerParts(step) {
    const usage = step.usage || {};
    const took = step.timings?.total_ms;
    return [
      h("span", { class: "tl-k" }, `Step ${step.idx}`),
      outcome(step.outcome),
      h("span", { class: "tl-meta" },
        usage.prompt_tokens !== undefined ? `${count(usage.prompt_tokens)} → ${count(usage.completion_tokens)} tokens` : null,
        took != null ? ` · ${millis(took)}` : null),
      flags(step),
      h("span", { class: "tl-chevron", "aria-hidden": "true" }, "▸"),
    ];
  }

  function flags(step) {
    const replay = step.replay || {};
    const out = [];
    const none = noReasoning(step);
    if (none) out.push(none);
    if (step.served) out.push(chip("replayed", "muted", `Answered from the recording: step ${replay.source_idx} of run ${short(replay.source_run || "")}`));
    if (replay.live_from !== undefined && replay.live_from !== null) out.push(chip("live from here", "accent", "The fork went live at this step"));
    if (replay.mismatch) out.push(chip("left the recording", "bad"));
    if (step.variant) out.push(chip("variant", "skill", "A variant wording was swapped in: the model saw it, the harness sent the original"));
    if (step.leaked) out.push(chip("original leaked", "warn"));
    if (step.edited.request) out.push(chip("context edited", "response", "Edited at a breakpoint before the model saw it"));
    if (step.edited.response) out.push(chip(`answer ${step.edited.response}`, "response", "At a breakpoint, before the harness acted on it"));
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
    replace(rail,
      h("div", { class: "tl-rail-head" }, plural(state.steps.length, "step")),
      state.steps.map((step) => [
        h("a", {
          class: ["tl-rail-step", step.outcome.kind === "error" && "tl-rail-error", step.kept === "retried" && "tl-rail-retried"],
          href: `${base}/${step.idx}`, dataset: { idx: step.idx },
          onclick: (event) => {
            event.preventDefault();
            reveal(step.idx);
          },
        }, h("span", { class: "mono" }, String(step.idx)), h("span", { class: "tl-rail-what" }, railLabel(step)),
        step.variant || step.edited.request || step.edited.response ? h("span", { class: "tl-dot", title: "changed by tuningfork" }) : null),
        (added.get(step.idx) || []).map((entry) => railEntry(entry, step, names)),
      ]),
      h("a", {
        class: ["tl-rail-verdict", verdict.tone], href: base, title: "What the task's check said: the end of the run",
        onclick: (event) => {
          event.preventDefault();
          verdictNode?.scrollIntoView({ block: "start" });
        },
      }, verdict.rail));
  }

  function railLabel(step) {
    if (step.outcome.kind === "error") return "error";
    const calls = (step.output?.tool_calls || []).map((call) => call.function?.name || "?");
    if (calls.length) return calls.join(", ");
    return finalReply(step) ? "reply" : "text";
  }

  /** A message the harness added after `step`, on a line of its own. A tool's result goes under
   *  the step, by the tool's name, and so does a user message the harness wrote itself, by what
   *  it is; a person's turn starts a new exchange, unindented, in their words. Each leads to the
   *  message in the conversation, opened. */
  function railEntry(entry, step, names) {
    const message = entry.message;
    const show = (event) => {
      event.preventDefault();
      const shown = element.querySelector(`#tm${entry.index}`);
      if (!shown) return;
      shown.open = true;
      shown.scrollIntoView({ block: "start" });
    };
    if (message.role === "user" && !entry.note) {
      return h("a", {
        class: "tl-rail-turn", href: `${base}/${step.idx}`, dataset: { entry: entry.index },
        title: `A person's turn, after step ${step.idx}`, onclick: show,
      }, h("span", { class: "tl-rail-who" }, "user"), h("span", { class: "tl-rail-what" }, firstLine(messageText(message), 80) || "(no text)"));
    }
    const what = message.role === "tool" ? `${names.get(message.tool_call_id) || "tool"} result` : entry.note || `${message.role} message`;
    return h("a", {
      class: "tl-rail-result", href: `${base}/${step.idx}`, dataset: { entry: entry.index },
      title: entry.note ? `Written by the harness itself after step ${step.idx}, not by a person` : `${what}, added by the harness after step ${step.idx}`,
      onclick: show,
    }, h("span", { class: "tl-rail-arrow", "aria-hidden": "true" }, "↳"), h("span", { class: "tl-rail-what" }, what));
  }

  function markCurrent({ idx = null, entry = null }) {
    for (const link of rail.querySelectorAll(".tl-rail-step")) link.classList.toggle("current", Number(link.dataset.idx) === idx);
    for (const link of rail.querySelectorAll(".tl-rail-result, .tl-rail-turn")) link.classList.toggle("current", Number(link.dataset.entry) === entry);
  }

  // --- a step, opened -------------------------------------------------------------------------

  function toggleOpen(idx) {
    const found = sections.get(idx);
    if (!found) return;
    const open = found.block.hidden;
    setOpen(found, open);
    const stillOpen = [...sections.values()].find((other) => !other.block.hidden);
    settle(open ? `${base}/${idx}` : stillOpen ? `${base}/${stillOpen.step.idx}` : base);
  }

  function setOpen(found, open) {
    found.block.hidden = !open;
    found.marker.setAttribute("aria-expanded", String(open));
    found.section.classList.toggle("tl-open", open);
    if (open && !found.ready) found.ready = build(found);
  }

  async function build(found) {
    const { step } = found;
    replace(found.block, loading("Loading the step…"));
    try {
      const detail = await api.get(`/runs/${enc(run.id)}/steps/${step.idx}?context=0`);
      const exact = `${base}/${step.idx}/context`;
      const here = state.skills.filter((span) => span.message_index < step.position);
      found.slot = h("div", { class: "start-slot" });
      replace(found.block,
        summaryStrip(detail),
        banners(detail, { original: () => h("a", { href: exact }, "open the exact context to compare.") }),
        here.length ? h("div", null, h("div", { class: "label" }, "Skills in this step's context"),
          skillsSummary(here, recorded)) : null,
        experimentsHere(step),
        // A replayed step was answered from the recording: there is no call to sample.
        step.served ? h("p", { class: "muted small" }, "Answered from the recording: nothing starts from here.") : startPanel(detail, { go }),
        found.slot,
        h("div", { class: "grid-2 tl-details" }, detailsPanel(detail), workspacePanel(detail)),
        h("p", { class: "tl-exact" },
          h("a", { class: "btn btn-small", href: exact }, "Exact context this step was sent →"),
          h("span", { class: "muted small" }, " every message as the model saw it: what a resample resends")));
    } catch (error) {
      found.ready = null;
      replace(found.block, failure(error, "Could not load the step"));
    }
  }

  function experimentsHere(step) {
    if (!step.experiments.length) return null;
    return h("div", { class: "tl-experiments" }, h("span", { class: "label" }, "Experiments here"),
      step.experiments.map((experiment) =>
        h("a", { href: `#/experiments/${experiment.id}` },
          chip(`${EXPERIMENT[experiment.kind] || experiment.kind}${experiment.reach === "end" ? " to the end" : ""} · ${experiment.samples}${experiment.n ? ` of ${experiment.n}` : ""}`,
            experiment.status === "done" ? "accent" : "muted", experiment.target || ""))));
  }

  /** Open step `idx`, bring it into view, and — given `start` — open that form in it. */
  async function reveal(idx, { start = null, query: asked = new URLSearchParams() } = {}) {
    const found = sections.get(idx);
    if (!found) return;
    setOpen(found, true);
    found.section.scrollIntoView({ block: "start" });
    if (!start) {
      settle(`${base}/${idx}`);
      return;
    }
    settle(`${base}/${idx}?${asked.toString()}`);
    const mine = ++asking;
    await found.ready;
    if (!found.slot || mine !== asking) return;
    // One form at a time: a second would be a second plan on screen.
    for (const other of sections.values()) if (other !== found && other.slot) replace(other.slot);
    replace(found.slot, loading("Preparing the form…"));
    try {
      const form = await startForm(start, { run, idx }, asked, { go });
      if (mine !== asking) return;
      replace(found.slot, form || failure(new Error(`There is no form called ${start}.`)));
      found.slot.scrollIntoView({ block: "nearest" });
    } catch (error) {
      if (mine === asking) replace(found.slot, failure(error, "Could not open the form"));
    }
  }

  /** Where a form's links lead: a step of this run opens here; anywhere else, away. */
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
      // "Close": the form goes, the step stays open — and one still loading never comes.
      asking += 1;
      const found = sections.get(idx);
      if (found?.slot) replace(found.slot);
      settle(`${base}/${idx}`);
      return;
    }
    reveal(idx, { start, query: asked });
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
      return { headline: "No verdict", rail: "No verdict", detail: "This run was recorded as it happened, not launched by tuningfork: no check ran." };
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
