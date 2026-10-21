// Rendering what crossed the wire: messages, skills inside them, tool calls, responses.
// Shared by the step view, held calls and comparisons.

import { h, chip, count, firstLine, longText, numbered, pretty } from "./dom.js";

// A content value's text: a string, or a list of parts or blocks (OpenAI parts; Anthropic
// text, thinking and tool_result blocks, when a held request is shown in its own format).
function blocksText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => block?.text ?? block?.thinking ?? (block?.type === "tool_result" ? blocksText(block.content) : ""))
    .filter(Boolean)
    .join("\n");
}

export function messageText(message) {
  return blocksText(message.content);
}

// Anthropic tool_use blocks, in a held request shown in its own format.
function toolUseNames(message) {
  return Array.isArray(message.content)
    ? message.content.filter((block) => block?.type === "tool_use").map((block) => block.name || "?")
    : [];
}

export function messageSize(message) {
  let size = messageText(message).length;
  for (const call of message.tool_calls || []) size += String(call.function?.arguments ?? "").length;
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (block?.type === "tool_use") size += JSON.stringify(block.input ?? {}).length;
    }
  }
  return size;
}

/** Map tool_call_id to the tool's name, so a tool result can say whose result it is. */
export function toolNames(messages) {
  const names = new Map();
  for (const message of messages) {
    for (const call of message.tool_calls || []) names.set(call.id, call.function?.name || "?");
  }
  return names;
}

/** A role as it may appear in a class name: a plain lowercase word, as roles are, or else
 *  "unknown" — a payload's text never picks classes of its own. */
export const roleClass = (role) => (typeof role === "string" && /^[a-z_]+$/.test(role) ? role : "unknown");

export function roleChip(role) {
  return h("span", { class: `role role-${roleClass(role)}` }, role || "?");
}

export function preview(message) {
  const calls = [
    ...(message.tool_calls || []).map((call) => call.function?.name || "?"),
    ...toolUseNames(message),
  ].map((name) => `${name}(…)`);
  if (calls.length) return calls.join(", ");
  const text = messageText(message);
  return text.trim() ? firstLine(text, 160) : "";
}

export function parseArguments(raw) {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) return { object: raw };
  try {
    const value = JSON.parse(raw);
    if (value && typeof value === "object" && !Array.isArray(value)) return { object: value };
  } catch {
    // shown raw below
  }
  return { raw: String(raw ?? "") };
}

export function toolCallView(call) {
  const fn = call.function || {};
  const args = parseArguments(fn.arguments);
  return h(
    "div",
    { class: "toolcall" },
    // The call's id, which a tool result answers to, on hover: it is the harness's, not the model's.
    h("div", { class: "toolcall-head" }, chip("tool call", "accent"),
      h("span", { class: "mono toolcall-name", title: call.id ? `call id ${call.id}` : undefined }, fn.name || "?")),
    h(
      "div",
      { class: "args" },
      args.object
        ? Object.entries(args.object).map(([key, value]) =>
            h("div", { class: "arg" }, h("div", { class: "arg-key mono" }, key),
              typeof value === "string" ? longText(value, { className: "text arg-value" }) : h("pre", { class: "code arg-value" }, pretty(value))),
          )
        : h("pre", { class: "code arg-value" }, args.raw),
    ),
  );
}

export function reasoningView(message, { open = false } = {}) {
  const reasoning = message.reasoning_content || message.reasoning;
  if (!reasoning) return null;
  return h(
    "details",
    { class: "reasoning", open },
    h("summary", null, "Reasoning ", h("span", { class: "faint" }, `${count(reasoning.length)} chars`)),
    h("pre", { class: "text" }, reasoning),
  );
}

/** A skill as the model read it: named, mapped to its file when the hashes agree, and
 *  line-numbered — the same numbers as the file, since the text matched byte for byte. */
export function skillMark(span, { recorded = true } = {}) {
  return h(
    "div",
    { class: "skill-mark", id: skillAnchor(span) },
    h(
      "div",
      { class: "skill-label" },
      chip("skill", "skill"),
      h("strong", null, span.name),
      h("span", { class: "muted" }, span.source === "preloaded" ? "preloaded into the system prompt" : "loaded by a tool call"),
      fileChip(span, recorded),
      h("span", { class: "faint push" }, `${count(span.lines)} lines`),
    ),
    numbered(span.text, span.first_line || 1),
  );
}

/** Whether the text the model read is, byte for byte, a skill file the run recorded. */
export function fileChip(span, recorded) {
  if (span.file && span.match === "body") {
    return chip(`✓ ${span.file}`, "muted", "The text matches this file's body; the harness left out its frontmatter. Line numbers are the file's");
  }
  if (span.file) return chip(`✓ ${span.file}`, "muted", "The text matches this file byte for byte, so its line numbers are the file's");
  if (!recorded) {
    return chip("skill files not recorded", "muted", "A batch run records its skill files when it finishes; this run has none recorded (yet)");
  }
  return chip("no recorded file matches", "warn", "No skill file recorded with this run has exactly this text: it was edited since, or the harness altered it");
}

// A decoded skill has no offset, so its name tells two in one message apart.
export const skillAnchor = (span) =>
  `skill-${span.message_index}-${span.start ?? `tool-${String(span.name).replace(/[^\w-]/g, "_")}`}`;

/** A message's content, with any skill inside it marked. */
export function contentView(message, spans = [], { recorded = true } = {}) {
  const content = message.content;
  if (Array.isArray(content)) {
    const parts = content.map((part) => {
      if (part?.type === "text") return longText(part.text);
      if (part?.type === "image_url") return chip("image", "muted");
      return h("pre", { class: "code" }, pretty(part));
    });
    if (!spans.length) return parts;
    // A skill in one block among several has no offsets into a single string: it is marked
    // on its own, with the blocks as sent folded below.
    const sent = message.role === "tool" ? "the tool result as sent" : "the message as sent";
    return [...spans.map((span) => skillMark(span, { recorded })), h("details", { class: "fold" }, h("summary", null, sent), parts)];
  }
  if (typeof content !== "string" || !content) return null;

  const inline = spans.filter((span) => span.start !== null && span.start !== undefined).sort((a, b) => a.start - b.start);
  const decoded = spans.filter((span) => span.start === null || span.start === undefined);
  if (!inline.length && !decoded.length) return longText(content);

  const unit = unitOffsets(content);
  const parts = [];
  let at = 0;
  for (const span of inline) {
    const start = unit(span.start);
    if (start > at) parts.push(harnessText(content.slice(at, start)));
    parts.push(skillMark(span, { recorded }));
    at = unit(span.end);
  }
  if (at < content.length && inline.length) parts.push(harnessText(content.slice(at)));
  for (const span of decoded) {
    parts.push(skillMark(span, { recorded }));
    parts.push(h("details", { class: "fold" }, h("summary", null, "the tool result as sent"), h("pre", { class: "code" }, content)));
  }
  return parts;
}

// Skill offsets count code points, as Python does, while a string slices by UTF-16 units.
// The two agree up to the first emoji or other astral character — one code point, two
// units — so only a string holding a surrogate is walked, once, as far as the offsets go.
function unitOffsets(text) {
  if (!/[\uD800-\uDFFF]/.test(text)) return (offset) => offset;
  let unit = 0;
  let point = 0;
  return (offset) => {
    if (offset < point) unit = point = 0; // asked out of order: walk again from the start
    while (point < offset && unit < text.length) {
      unit += text.codePointAt(unit) > 0xffff ? 2 : 1;
      point += 1;
    }
    return unit;
  };
}

// Harness-written text around a skill: folded when long, so the skill stands out.
function harnessText(text) {
  if (!text.trim()) return null;
  return longText(text, { limit: 400, lines: 6, className: "text harness-text" });
}

// What changed a message before the model saw it: a person at a breakpoint, or a fork's
// variant wording.
const CHANGED = {
  breakpoint: ["edited", "change", "Changed at a breakpoint before the model saw it"],
  variant: ["variant", "change", "The variant wording was swapped in here: the model saw it, the harness sent the original"],
};

/** One message, folded to a line. `extra` goes in its header line; `idPrefix` keeps ids
 *  apart when one page shows more than one conversation. */
export function messageView(message, index, {
  spans = [], names = new Map(), edited = false, changedBy = "breakpoint", open = true, recorded = true, extra = null, idPrefix = "",
} = {}) {
  const role = message.role;
  const own = spans.filter((span) => span.message_index === index);
  const head = [
    roleChip(role),
    role === "tool" ? h("span", { class: "muted" }, `result of ${names.get(message.tool_call_id) || "a tool call"}`) : null,
    own.length ? chip(own.map((span) => span.name).join(", "), "skill") : null,
    edited ? chip(...(CHANGED[changedBy] || CHANGED.breakpoint)) : null,
    extra,
    h("span", { class: "msg-preview" }, preview(message)),
    h("span", { class: "faint small" }, `${count(messageSize(message))} chars`),
  ];
  return h(
    "details",
    { class: ["msg", `msg-${roleClass(role)}`, edited && "msg-edited"], id: `${idPrefix}m${index}`, open },
    // Its number in the request, on hover: what the terminal's `tuningfork step` calls it.
    h("summary", { class: "msg-head", title: `message #${index}` }, head),
    h("div", { class: "msg-body" }, reasoningView(message), contentView(message, own, { recorded }), (message.tool_calls || []).map(toolCallView)),
  );
}

export function messagesView(messages, { spans = [], edited = new Set(), changedBy = "breakpoint", openFrom, recorded = true } = {}) {
  const names = toolNames(messages);
  // The recent end of a conversation is what the model is reacting to; older turns start
  // folded. A message carrying a skill always starts open.
  const from = openFrom ?? Math.max(0, messages.length - 3);
  return messages.map((message, index) =>
    messageView(message, index, {
      spans,
      names,
      edited: edited.has(index),
      changedBy,
      recorded,
      open: index >= from || spans.some((span) => span.message_index === index),
    }),
  );
}

export function responseMessage(response) {
  return response?.choices?.[0]?.message || null;
}

/** A step where the model called a tool without reasoning, though it reasons at other steps
 *  of the run (`analysis/reasoning.py` decides): said plainly, with whether a tool call had
 *  just failed. */
export function noReasoning(step) {
  if (step?.reasoning !== "none") return null;
  return chip(step.after_failure ? "no reasoning · after a failed call" : "no reasoning", "warn",
    `The model called a tool without writing any reasoning first, though it reasons at other steps of this run${
      step.after_failure ? ": it acted at once, right after a tool call failed." : "."}`);
}

/** What the model decided: reasoning, text, and tool calls. */
export function responseView(response) {
  return outputView(responseMessage(response));
}

/** An answer — a response's message — as the model gave it: reasoning, text, tool calls.
 *  `label` heads its text; none where the page says what the text is already. */
export function outputView(message, { label = "Content" } = {}) {
  if (!message) return h("p", { class: "muted" }, "No response was recorded.");
  const hasAction = Boolean(message.content) || Boolean(message.tool_calls?.length);
  const parts = [reasoningView(message, { open: !hasAction })];
  if (message.content) {
    parts.push(h("div", null, label ? h("div", { class: "label" }, label) : null,
      typeof message.content === "string" ? longText(message.content, { limit: 4000, lines: 60 }) : h("pre", { class: "code" }, pretty(message.content))));
  }
  for (const call of message.tool_calls || []) parts.push(toolCallView(call));
  if (!hasAction && !parts[0]) parts.push(h("p", { class: "muted" }, "The model returned nothing."));
  return parts;
}
