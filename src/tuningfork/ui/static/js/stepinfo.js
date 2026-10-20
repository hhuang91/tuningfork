// A step's details, as both the run's timeline and a step's exact-context page show them:
// what it cost, what it was sent with, what tuningfork changed about it, the skills in its
// context, and the workspace its call found.

import { api, enc } from "./api.js";
import { chip, count, firstLine, foldPanel, h, kv, millis, plural, pretty, replace, short } from "./dom.js";
import { fileChip, skillAnchor } from "./payload.js";

export function summaryStrip(step) {
  const usage = step.usage || {};
  const item = (label, value) =>
    value === null || value === undefined || value === "" ? null : h("span", { class: "item" }, h("span", null, label), h("span", null, value));
  return h(
    "div",
    { class: "summary-strip" },
    item("model", step.model ? h("span", { class: "mono" }, step.model) : null),
    item("wire", step.wire && step.wire !== "openai-chat" ? h("span", { class: "mono" }, step.wire) : null),
    item("finish", step.finish_reason),
    item("tokens", usage.prompt_tokens !== undefined ? `${count(usage.prompt_tokens)} in → ${count(usage.completion_tokens)} out` : null),
    item("first token", step.timings?.ttft_ms != null ? millis(step.timings.ttft_ms) : null),
    item("total", step.timings?.total_ms != null ? millis(step.timings.total_ms) : null),
    step.edits && Object.keys(step.edits).length ? chip("edited at a breakpoint", "change") : null,
    step.substitution?.spans ? chip("variant swapped in", "change") : null,
  );
}

/** What happened at a step that is not the model answering: a replay, a variant, an edit, a
 *  failure. `original` makes the way to what the harness sent instead: by default the
 *  exact-context page's own toggle, named; elsewhere, a link to that page. */
export function banners(step, { original = null } = {}) {
  const out = [];
  const replay = step.replay || {};
  const compare = (text) => (original ? original() : text);
  if (replay.served_from) {
    // A fork's step before the one it went live at: the model was not asked.
    out.push(h("div", { class: "banner" }, h("strong", null, "Answered from the recording. "),
      `This fork replayed step ${replay.source_idx} of run ${(replay.source_run || "").slice(0, 8)} here; the harness re-ran the tool calls in it`,
      replay.workspace ? `, and found its workspace ${replay.workspace === "same" ? "as the recording's was" : "different from the recording's"}.` : "."));
  }
  if (replay.live_from) {
    out.push(h("div", { class: "banner" }, h("strong", null, "The fork went live here. "),
      "Everything before this step was answered from the recording",
      replay.workspace ? `; the workspace was ${replay.workspace === "same" ? "the recording's" : "not the recording's"} at this point.` : "."));
  }
  if (replay.mismatch) {
    out.push(h("div", { class: "banner banner-warn" }, h("strong", null, "The replay left the recording here. "),
      "The harness asked for something the recording never answered, so this fork's history is not the recorded one, and it is not counted."));
  }
  if (step.substitution?.spans) {
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "A variant wording was swapped in. "),
      `The model saw the variant in ${plural(step.substitution.spans, "place")}; the harness sent the original — `,
      compare("see “The original wording” for it."),
      step.substitution.leaked ? " The original wording still reached the model in another form." : ""));
  }
  const request = step.edits?.request;
  if (request) {
    const positions = step.edited_messages ?? request.messages ?? [];
    const what = [
      positions.length ? `message${positions.length === 1 ? "" : "s"} ${positions.map((i) => `#${i}`).join(", ")}` : null,
      ...(request.fields || []).filter((field) => field !== "messages"),
    ].filter(Boolean).join("; ");
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "The context was edited before the model saw it"),
      ` (${what}). This step records what the model saw — `, compare("switch to “What the harness sent” to compare.")));
  }
  const response = step.edits?.response;
  const undelivered = step.error && step.error.startsWith("tuningfork could not deliver");
  if (step.error && step.response && !response) {
    // A stream that broke off keeps what arrived: shown, but it is not a behaviour.
    out.push(h("div", { class: "banner banner-warn" }, h("strong", null, "This call failed. "),
      `${step.error}. What arrived before it did is shown; it counts as an error, not as behaviour.`));
  }
  if (response?.dropped) {
    out.push(h("div", { class: "banner banner-warn" }, h("strong", null, "Dropped at a breakpoint. "),
      "The model answered, but the harness received an error instead."));
  } else if (response && undelivered) {
    out.push(h("div", { class: "banner banner-warn" }, h("strong", null, "The response was edited, but the edit could not be delivered. "),
      `The harness received an error instead (${step.error}). The step's behaviour is what the model produced.`));
  } else if (response) {
    const fields = (response.message_fields || response.fields || []).join(", ");
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "The response was edited before the harness acted on it"),
      ` (${fields}). The step's behaviour is still what the model produced; the harness acted on the edited version.`));
  }
  return out.length ? h("div", { class: "banners" }, out) : null;
}

/** The skills in a context: each named, placed, and mapped to its file. `anchor` turns a
 *  span into the id of where it is shown on this page. */
export function skillsSummary(spans, recorded, { anchor = skillAnchor } = {}) {
  return h(
    "div",
    { class: "skills-list" },
    spans.map((span) =>
      h("div", { class: "skill-ref" },
        chip("skill", "skill"),
        h("a", { href: `#${anchor(span)}`, onclick: (event) => { event.preventDefault(); jumpTo(anchor(span)); } }, span.name),
        h("span", { class: "muted small" }, span.source === "preloaded" ? `preloaded in message #${span.message_index}` : `loaded by a tool, message #${span.message_index}`),
        fileChip(span, recorded),
        h("span", { class: "faint small" }, plural(span.lines, "line")))),
  );
}

export function jumpTo(id) {
  const target = document.getElementById(id);
  if (!target) return;
  const message = target.closest("details");
  if (message) message.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function toolsOffered(tools, { note = null } = {}) {
  if (!tools?.length) return null;
  const offered = tools.map(toolOf);
  return h(
    "details",
    { class: "fold tools-offered" },
    h("summary", null, h("span", null, `${plural(offered.length, "tool")} offered`),
      h("span", { class: "fold-preview" }, offered.map((tool) => tool.name).join(", "))),
    note,
    h("div", { class: "tool-list" }, offered.map(toolView)),
  );
}

/** A tool as the model reads it, whatever its wire's shape: an OpenAI function, an Anthropic
 *  tool, or the timeline's own `{name, description, parameters}`. */
function toolOf(tool) {
  const spec = tool.function && typeof tool.function === "object" ? tool.function : tool;
  return {
    name: spec.name || "?",
    description: typeof spec.description === "string" ? spec.description : "",
    parameters: spec.parameters ?? spec.input_schema ?? null,
  };
}

/** One tool on a line — its name and what it says it does — opening to the whole description
 *  and its parameters, then the schema as sent. */
function toolView(tool) {
  const properties = tool.parameters?.properties && typeof tool.parameters.properties === "object" ? tool.parameters.properties : {};
  const required = new Set(Array.isArray(tool.parameters?.required) ? tool.parameters.required : []);
  const names = Object.keys(properties);
  return h(
    "details",
    { class: "tool" },
    h("summary", null, h("span", { class: "mono tool-name" }, tool.name), h("span", { class: "tool-gist" }, firstLine(tool.description, 160))),
    h("div", { class: "tool-body" },
      tool.description ? h("pre", { class: "text" }, tool.description) : h("p", { class: "muted small" }, "It says nothing of what it does."),
      names.length
        ? h("ul", { class: "tool-params" }, names.map((name) => {
            const param = properties[name] && typeof properties[name] === "object" ? properties[name] : {};
            return h("li", null, h("span", { class: "mono" }, name), h("span", { class: "faint" }, ` ${typeName(param)}`),
              required.has(name) ? [" ", chip("required", "muted")] : null,
              typeof param.description === "string" && param.description ? h("span", { class: "muted" }, ` — ${param.description}`) : null);
          }))
        : h("p", { class: "muted small" }, "No parameters."),
      tool.parameters ? h("details", { class: "fold" }, h("summary", null, "Its parameters, as sent"), h("pre", { class: "code" }, pretty(tool.parameters))) : null),
  );
}

function typeName(param) {
  if (Array.isArray(param.enum)) return param.enum.map((value) => JSON.stringify(value)).join(" | ");
  if (Array.isArray(param.type)) return param.type.join(" | ");
  if (param.type === "array" && param.items?.type) return `${param.items.type}[]`;
  if (Array.isArray(param.anyOf)) return param.anyOf.map((each) => each?.type || "…").join(" | ");
  return typeof param.type === "string" ? param.type : "";
}

/** The workspace as this call found it: every tool effect before it, on disk. Only a run
 *  tuningfork launched has one. Folded, and read when opened. */
export function workspacePanel(step) {
  if (!step.workspace_commit) return null;
  return foldPanel("Workspace", () => workspaceFiles(step), {
    className: "workspace-panel",
    meta: h("span", { class: "muted" }, "as this call found it · snapshot ", h("span", { class: "mono" }, short(step.workspace_commit))),
  });
}

function workspaceFiles(step) {
  const holder = h("div", null, h("p", { class: "muted" }, "Loading…"));
  api.get(`/runs/${enc(step.run.id)}/files?at=${step.idx}`).then((view) => {
    if (view.error) {
      replace(holder, h("p", { class: "muted" }, view.error));
      return;
    }
    const status = new Map((view.changes || []).map((change) => [change.path, change.status]));
    const deleted = (view.changes || []).filter((change) => change.status === "D").map((change) => change.path);
    replace(
      holder,
      view.files.length
        ? h("table", { class: "table" }, h("tbody", null, view.files.map((file) =>
            h("tr", null,
              h("td", { class: "mono" }, file.path),
              h("td", { class: "num faint" }, file.size != null ? `${file.size} bytes` : ""),
              h("td", null, status.has(file.path)
                ? chip(status.get(file.path) === "A" ? "new since the step before" : "changed since the step before", "accent")
                : null)))))
        : h("p", { class: "muted" }, "Empty."),
      deleted.length ? h("p", { class: "muted" }, `Deleted since the step before: ${deleted.join(", ")}`) : null,
      h("p", { class: "faint" }, `tuningfork files ${short(step.run.id)} ${step.idx} --diff shows the changes; --out <dir> writes the files out.`),
    );
  }).catch((error) => replace(holder, h("p", { class: "muted" }, String(error?.message || error))));
  return holder;
}

/** How a step was sampled and how long it took, kept for the record: folded, the two
 *  timings already in its summary. Null when nothing was recorded of either. */
export function detailsPanel(step) {
  const params = step.params || {};
  const defaults = step.server_defaults || {};
  const rows = [
    ...Object.entries(params).map(([key, value]) => [key, h("span", { class: "mono" }, typeof value === "object" ? JSON.stringify(value) : String(value))]),
    ["time to first token", step.timings?.ttft_ms != null ? millis(step.timings.ttft_ms) : null],
    ["total time", step.timings?.total_ms != null ? millis(step.timings.total_ms) : null],
  ].filter(([, value]) => value !== null && value !== undefined);
  if (!rows.length && !Object.keys(defaults).length) return null;
  return foldPanel("Details for the record", [
    kv(rows),
    Object.keys(defaults).length
      ? h("details", { class: "fold" }, h("summary", null, "Server defaults at the time"), h("pre", { class: "code" }, pretty(defaults)))
      : null,
  ], { className: "record-panel", meta: h("span", { class: "muted" }, "how it was sampled, and how long it took") });
}
