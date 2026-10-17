// A step's answer as tokens: how sure the model was of each, and where it could have gone
// another way. The server scores the answer and works out the numbers (`GET …/tokens`); this
// draws them. Every token reaches the page as a text node; the heat is a class; one listener
// serves every token, so a long answer costs one handler, not thousands.

import { api, enc } from "./api.js";
import { guard } from "./auth.js";
import { command, failure, h, loading, plural, replace, short, toggle, when } from "./dom.js";
import { watchJob } from "./jobs.js";
import { live } from "./live.js";

const PART_LABEL = { reasoning: "Reasoning", content: "Answer", tool_call: "Tool call" };
const ENTROPY_EDGES = ["0.05", "0.25", "0.5", "1", "1.5", "2", "3"];
const CLOSENESS_EDGES = ["0.05", "0.15", "0.3", "0.5", "0.7", "0.85", "0.95"];

/** A token as it reads: whitespace shown, quoted. */
const quoted = (text) => JSON.stringify(text ?? "");
function pct(p) {
  if (p === null || p === undefined) return "—";
  if (p === 0 || p >= 0.995) return `${Math.round(p * 100)}%`;
  if (p < 0.0001) return "<0.01%";
  return `${(p * 100).toFixed(p < 0.01 ? 2 : 1)}%`;
}

/** Marks for the heatmap from forking-token probes: every position one forced, whether a
 *  forced token changed what the model did there, and each probe's result at it. */
export function probeMarks(probes) {
  const marks = new Map();
  for (const probe of probes || []) {
    for (const position of probe.positions || []) {
      const mark = marks.get(position.position) || { flips: false, probes: [] };
      mark.flips ||= position.alternatives.some((alt) => alt.flips);
      mark.probes.push({ experiment: probe.experiment, position });
      marks.set(position.position, mark);
    }
  }
  return marks;
}

/** The heatmap of one stream (`view` of a `/tokens` answer). `onProbe(positions)` opens the
 *  forking-token form with those positions; `marks` (from `probeMarks`) ring the tokens a
 *  probe forced; `focus` is a position to open on; `compact` leaves out the hotspot list.
 *  The element's `reveal(position)` scrolls to a token and opens it. */
export function heatmap(view, { onProbe = null, marks = null, focus = null, compact = false } = {}) {
  const tokens = view.tokens;
  const byIndex = new Map(tokens.map((token) => [token.i, token]));
  const selected = new Set();
  const groups = []; // [first position, span]
  let by = "entropy";
  let current = null;

  const body = h("div", {
    class: "tk-body", tabindex: "0", role: "group",
    "aria-label": "The answer, token by token. Arrow keys move between tokens; Enter selects one to probe.",
    onclick: (event) => {
      const found = event.target.closest?.(".tk-t");
      if (found) show(Number(found.dataset.i));
    },
    onkeydown: keys,
  });
  const detail = h("div", { class: "tk-detail", hidden: true, "aria-live": "polite" });
  const bar = h("div", { class: "tk-bar" });
  const legend = h("div", { class: "tk-legend" });
  const colour = h("div", { class: "controls" });

  drawTokens();
  drawControls();
  drawBar();

  const element = h("div", { class: "tk" },
    head(view), colour, legend, body, detail, bar, compact ? null : hotspots(view));
  element.reveal = reveal;
  if (focus !== null && focus !== undefined) setTimeout(() => reveal(focus), 0);
  return element;

  function reveal(position) {
    const span = spanAt(position);
    if (!span) return;
    span.closest("details")?.setAttribute("open", "");
    span.scrollIntoView({ block: "center" });
    show(position);
  }

  function markOf(first) {
    if (!marks) return null;
    const group = byIndex.get(first)?.group;
    const [start, end] = group || [first, first + 1];
    for (let i = start; i < end; i += 1) if (marks.has(i)) return marks.get(i);
    return null;
  }

  function drawControls() {
    replace(colour,
      h("span", { class: "muted small" }, "Shade by"),
      toggle([["entropy", "Entropy"], ["margin", "Closeness of the top two"]], by, (value) => {
        by = value;
        drawControls();
        for (const [first, span] of groups) span.className = spanClass(first);
      }));
    const edges = by === "entropy" ? ENTROPY_EDGES : CLOSENESS_EDGES;
    replace(legend,
      h("span", { class: "muted small" }, by === "entropy" ? "bits" : "1 − margin"),
      h("span", { class: "tk-t h0" }, "0"),
      edges.map((edge, level) => h("span", { class: `tk-t h${level + 1}` }, `≥${edge}`)));
  }

  function heatOf(first) {
    const group = byIndex.get(first).group;
    const [start, end] = group || [first, first + 1];
    let heat = 0;
    for (let i = start; i < end; i += 1) heat = Math.max(heat, (by === "entropy" ? byIndex.get(i)?.heat : byIndex.get(i)?.heat_margin) || 0);
    return heat;
  }

  function spanClass(first) {
    const token = byIndex.get(first);
    const mark = markOf(first);
    return ["tk-t", `h${heatOf(first)}`, mark && "tk-probed", mark?.flips && "tk-flipped",
      selected.has(first) && "tk-selected", current === first && "tk-current",
      token.redacted && "tk-redacted"].filter(Boolean).join(" ");
  }

  function drawTokens() {
    const parts = [];
    let part = null;
    let holder = null;
    let count = 0;
    let wanted = false; // a ringed token or the one to open on is in this part: shown open
    const flush = () => {
      if (!holder) return;
      if (part === "reasoning") {
        const hot = (view.hotspots || []).filter((spot) => spot.part === "reasoning").length;
        parts.push(h("details", { class: "tk-part tk-reasoning", open: wanted },
          h("summary", null, `Reasoning · ${plural(count, "token")}${hot ? ` · ${hot} of the likeliest forks are here` : ""}`), holder));
      } else {
        parts.push(h("div", { class: `tk-part tk-${part || "content"}` }, h("div", { class: "label" }, PART_LABEL[part] || "Answer"), holder));
      }
    };
    for (const token of tokens) {
      if (token.part === "marker") continue;
      if (token.group && token.group[0] !== token.i) continue; // drawn with its group
      if (token.part !== part || !holder) {
        flush();
        part = token.part;
        holder = h("div", { class: "tk-text" });
        count = 0;
        wanted = false;
      }
      const [start, end] = token.group || [token.i, token.i + 1];
      if (markOf(token.i) || (focus !== null && focus >= start && focus < end)) wanted = true;
      const span = h("span", { class: spanClass(token.i), dataset: { i: token.i } }, token.redacted ? "<redacted>" : token.text);
      groups.push([token.i, span]);
      holder.append(span);
      count += token.group ? token.group[1] - token.group[0] : 1;
    }
    flush();
    replace(body, parts.length ? parts : h("p", { class: "muted" }, "No tokens to show."));
  }

  function spanAt(position) {
    const token = byIndex.get(position);
    const first = token?.group ? token.group[0] : position;
    return groups.find(([start]) => start === first)?.[1] || null;
  }

  function show(position) {
    const previous = current;
    current = position;
    for (const at of [previous, position]) {
      if (at === null || at === undefined) continue;
      const span = spanAt(at);
      if (span) span.className = spanClass(Number(span.dataset.i));
    }
    const token = byIndex.get(position);
    if (!token) return;
    const pick = h("button", {
      class: ["btn btn-small", selected.has(position) && "active"], type: "button", "aria-pressed": String(selected.has(position)),
      onclick: () => choose(position),
    }, selected.has(position) ? "Selected to probe" : "Select to probe");
    replace(detail,
      h("div", { class: "tk-detail-head" },
        h("span", { class: "mono" }, `[${position}]`),
        h("span", { class: "muted" }, PART_LABEL[token.part] || token.part || ""),
        h("span", { class: "mono tk-quoted" }, quoted(token.text)),
        h("span", { class: "muted small" }, `chosen at ${pct(token.p)}`),
        token.entropy !== null ? h("span", { class: "muted small" }, `entropy ≥ ${token.entropy.toFixed(2)} bits`) : null,
        token.margin !== null ? h("span", { class: "muted small" }, `margin ${token.margin.toFixed(2)}`) : null),
      token.redacted
        ? h("p", { class: "muted small" }, "Redacted: part of a secret, so nothing about it is kept.")
        : h("div", { class: "tk-alts" }, token.alternatives.map(([text, p]) =>
            h("div", { class: "tk-alt" },
              h("span", { class: "mono tk-quoted" }, quoted(text)),
              h("span", { class: "tk-alt-bar" }, h("span", { class: "tk-alt-fill", style: { width: `${Math.max(1, p * 100).toFixed(1)}%` } })),
              h("span", { class: "mono small" }, pct(p))))),
      markOf(position) ? probed(markOf(position)) : null,
      onProbe && !token.redacted ? h("div", { class: "actions" }, pick,
        h("button", { class: "btn btn-small", type: "button", "data-control": true, onclick: () => onProbe([position]) }, "Find forking tokens here…")) : null);
    detail.hidden = false;
  }

  // What each probe found at this token, each leading to its result.
  function probed(mark) {
    const arm = (found, label) => `${label} ${quoted(found.arm.text)} ${found.target.successes}/${found.target.total}`;
    return h("div", { class: "tk-probes" },
      h("div", { class: "label" }, mark.flips ? "Probed here: a forced token changed what the model did" : "Probed here: no forced token decided what the model did"),
      mark.probes.map(({ experiment, position }) => h("div", { class: "tk-probe" },
        h("a", { class: "mono", href: `#/experiments/${experiment.id}?at=${position.position}` }, `probe ${short(experiment.id)}`),
        h("span", { class: "small muted" }, arm(position.baseline, "as run")),
        position.alternatives.map((alt) => h("span", { class: ["small", alt.flips ? "tk-probe-flip" : "muted"] },
          arm(alt, "forced") + (alt.flips ? `, flips ${alt.direction === "away" ? "away from" : "toward"} the target` : ""))))));
  }

  function choose(position) {
    if (selected.has(position)) selected.delete(position);
    else selected.add(position);
    const span = spanAt(position);
    if (span) span.className = spanClass(Number(span.dataset.i));
    drawBar();
    show(position);
  }

  function drawBar() {
    if (!onProbe || !selected.size) {
      replace(bar);
      return;
    }
    const positions = [...selected].sort((a, b) => a - b);
    replace(bar,
      h("span", null, `${plural(positions.length, "token")} selected: ${positions.map((p) => `[${p}]`).join(" ")}`),
      h("button", { class: "btn btn-small btn-primary", type: "button", "data-control": true, onclick: () => onProbe(positions) }, "Find forking tokens at these…"),
      h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => {
        const was = [...selected];
        selected.clear();
        for (const at of was) {
          const span = spanAt(at);
          if (span) span.className = spanClass(Number(span.dataset.i));
        }
        drawBar();
      } }, "Clear"));
  }

  function keys(event) {
    const order = groups.map(([first]) => first);
    if (!order.length) return;
    const at = current === null ? -1 : order.indexOf(spanAt(current) ? Number(spanAt(current).dataset.i) : current);
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      const next = order[Math.min(order.length - 1, Math.max(0, at + (event.key === "ArrowRight" ? 1 : -1)))];
      const span = spanAt(next);
      span?.closest("details")?.setAttribute("open", "");
      span?.scrollIntoView({ block: "nearest" });
      show(next);
    } else if ((event.key === "Enter" || event.key === " ") && current !== null && onProbe) {
      event.preventDefault();
      choose(current);
    }
  }

  function hotspots(found) {
    const spots = found.hotspots || [];
    if (!spots.length) return null;
    return h("div", { class: "tk-hotspots" },
      h("div", { class: "label" }, "Where the step's own sampler could most easily have drawn another token"),
      h("ol", null, spots.map((spot) =>
        h("li", null, h("button", {
          class: "tk-hot", type: "button",
          onclick: () => {
            const span = spanAt(spot.position);
            span?.closest("details")?.setAttribute("open", "");
            span?.scrollIntoView({ block: "center" });
            show(spot.position);
          },
        },
        h("span", { class: "mono" }, `[${spot.position}]`),
        h("span", { class: "muted small" }, PART_LABEL[spot.part] || spot.part || ""),
        h("span", { class: "mono" }, `${quoted(spot.chosen.text)} ${pct(spot.chosen.p)}`),
        h("span", { class: "muted" }, "→"),
        h("span", { class: "mono" }, spot.alternatives.map((alt) => `${quoted(alt.text)} ${pct(alt.p)}`).join("  ")),
        h("span", { class: "muted small" }, `drawn ${pct(spot.runner_up)}`))))));
  }
}

function head(view) {
  const stream = view.stream;
  const facts = [stream.source, stream.method, `top ${view.top_k}`, plural(stream.tokens, "token")];
  if (stream.exact === true) facts.push("rebuilt exactly");
  if (stream.exact === false) facts.push("rebuilt approximately");
  return h("div", { class: "tk-head" },
    h("span", { class: "muted small" }, `stream ${short(stream.id)} · ${facts.filter(Boolean).join(" · ")} · ${when(stream.created_at)}`),
    view.same_server === false
      ? h("div", { class: "banner banner-warn small" }, "Scored on a different server than the step ran on: the numbers are that server's.")
      : null,
    stream.exact === false
      ? h("div", { class: "banner banner-warn small" },
          `The answer was rebuilt approximately (${stream.notes?.answer_tokens ?? "?"} tokens, the run's had ${stream.notes?.recorded_answer_tokens ?? "?"}): these are close to, not exactly, the run's own tokens.`)
      : null,
    view.sample
      ? h("p", { class: "small" }, `A sample: #${view.sample.idx} of experiment `,
          h("a", { class: "mono", href: `#/experiments/${view.sample.experiment_id}` }, short(view.sample.experiment_id)),
          ` — ${view.sample.outcome || ""}`)
      : null,
    view.choice
      ? h("p", { class: "small" }, `Choice of tool, at token [${view.choice.position}]: `,
          Object.entries(view.choice.chance || {}).map(([name, p]) => `${name} ${Math.round(p * 100)}%`).join(", "),
          " — the model's own chance of each, under the step's sampler. A choice near even is one a few words of the skill can move.")
      : null,
    h("p", { class: "hint" }, `Entropy counts the probability beyond the top ${view.top_k} as one outcome, so it is a lower bound. Where it could have gone another way is under the step's own sampler settings.`));
}

/** A step's Tokens view in the timeline: its answer's heatmap, or the way to score it.
 *  `focus` is a position to open on, as a probe's result links to it. */
export function tokensPanel({ run, idx, scope, onProbe, focus = null }) {
  const host = h("div", { class: "tk-host" });
  load();
  return host;

  async function load(stream = null) {
    replace(host, loading("Loading the tokens…"));
    try {
      const data = await api.get(`/runs/${enc(run.id)}/steps/${idx}/tokens${stream ? `?stream=${enc(stream)}` : ""}`);
      const marks = probeMarks(data.probes);
      replace(host, probesLine(data.probes),
        data.view ? heatmap(data.view, { onProbe, marks, focus: stream ? null : focus }) : unscored(), others(data));
    } catch (error) {
      replace(host, failure(error, "Could not load the tokens"));
    }
  }

  function probesLine(probes) {
    if (!probes?.length) return null;
    return h("p", { class: "small tk-probes-line" },
      `Probed ${probes.length === 1 ? "once" : `${probes.length} times`} from this answer — `,
      probes.map((probe, index) => [index ? ", " : null,
        h("a", { class: "mono", href: `#/experiments/${probe.experiment.id}` }, short(probe.experiment.id)),
        ` (${plural(probe.positions.reduce((total, position) => total + position.alternatives.filter((alt) => alt.flips).length, 0), "flip")})`]),
      ". The tokens they forced are ringed: solid where a forced token changed what the model did.");
  }

  function unscored() {
    if (live.mode !== "live") {
      return h("div", { class: "stack", style: { gap: "6px" } },
        h("p", { class: "muted" }, "This answer has not been scored yet. Scoring sends its exact context once more, with the answer forced token by token, and reads how sure the model was of each. From a terminal, with the model server running:"),
        command(`tuningfork tokens ${short(run.id)} ${idx} --score`));
    }
    const status = h("div", { class: "muted small" });
    const button = h("button", { class: "btn btn-primary btn-small", type: "button", "data-control": true, onclick: guard(score) }, "Score this answer");
    return h("div", { class: "stack", style: { gap: "6px" } },
      h("p", { class: "muted" }, "Not scored yet. Scoring sends this step's exact context once more, with the answer forced token by token, and reads how sure the model was of each — one call, about as long as the answer took."),
      h("div", { class: "actions" }, button, status));

    async function score() {
      button.disabled = true;
      status.textContent = "starting…";
      try {
        const job = await api.post("/jobs", { kind: "tokens", params: { run: run.id, step: String(idx) } });
        status.textContent = "scoring…";
        watchJob(job.id, scope, (state) => {
          const found = state.job;
          if (!found) return;
          if (found.status === "done") load();
          else if (["failed", "cancelled", "interrupted"].includes(found.status)) {
            button.disabled = false;
            replace(status, h("span", { class: "form-error" }, found.error?.message || found.status),
              found.error?.hint ? h("span", { class: "hint" }, ` ${found.error.hint}`) : null);
          } else {
            status.textContent = found.progress?.waiting ? "waiting: the proxy is paused" : "scoring…";
          }
        });
      } catch (error) {
        button.disabled = false;
        replace(status, h("span", { class: "form-error" }, error.message), error.hint ? h("span", { class: "hint" }, ` ${error.hint}`) : null);
      }
    }
  }

  function others(data) {
    const shownId = data.view?.stream?.id;
    const rest = (data.streams || []).filter((stream) => stream.id !== shownId);
    if (!rest.length) return null;
    return h("details", { class: "fold tk-streams" },
      h("summary", null, `${plural(rest.length, "other token stream")} at this step`),
      h("ul", { class: "small" }, rest.map((stream) =>
        h("li", null, h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: () => load(stream.id) },
          stream.experiment_id
            ? `sample ${stream.sample_idx} of ${short(stream.experiment_id)}: ${stream.outcome || ""}`
            : `${stream.source}, ${when(stream.created_at)}`)))));
  }
}
