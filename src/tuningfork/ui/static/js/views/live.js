// Held calls and breakpoints, live from the proxy.
//
// A held call waits here for a decision. Before the model, the context can be edited;
// after it, the model's decision can — the harness acts on whatever is sent back. The
// model's own output is always what gets recorded as the step's behaviour.

import { api } from "../api.js";
import { guard } from "../auth.js";
import { holdTools } from "../holdtools.js";
import {
  chip, command, count, elapsed, h, panel, plural, pretty, replace, short, tick as tickUnder, toast,
} from "../dom.js";
import { live } from "../live.js";
import { settle } from "../nav.js";
import {
  messageSize, parseArguments, preview, reasoningView, responseMessage, roleChip, roleClass,
  toolCallView, toolNames,
} from "../payload.js";

const PHASE = {
  request: { chip: ["before the model", "muted"], editable: "request" },
  response: { chip: ["after the model", "muted"], editable: "response" },
};

// The phase has its own chip beside the reason, so the reason need not repeat it.
const reasonText = (hold) => hold.reason.replace(/^(before|after) the model: /, "");

// Each held call's edit in progress, by hold id. It lives apart from the editor showing
// it, so drawing the editor again — another hold chosen and this one again, a reconnect,
// a visit to another page — brings it back as it was. Dropped once the call is decided.
const edits = new Map();

export async function liveView(root, [holdId], scope) {
  if (live.mode !== "live") {
    replace(root, h("div", { class: "panel empty" }, h("h2", null, "No live proxy"),
      h("p", null, "This page is browsing recordings. Breakpoints need a running proxy: start one with `tuningfork serve` and open its UI.")));
    return;
  }

  let selected = holdId || null;
  let shown = null; // the hold whose editor is on the page
  let lastDecision = null; // { id, action } decided from this page
  let left = false; // another view has replaced this one
  const payloads = new Map(); // hold id -> full hold, which never changes while pending
  // Known decided — here, or elsewhere as a refused decision showed — before the event
  // stream says so: its announcement can arrive after the answer to our own request.
  const settled = new Set();
  scope.onLeave(() => (left = true));

  const listEl = h("div", { class: "hold-list" });
  const detailEl = h("div");
  const breakpointsEl = h("div");
  const recentEl = h("div", { class: "recent" });

  replace(
    root,
    h("div", { class: "live-layout" },
      h("section", null,
        h("div", { class: "page-head" }, h("div", null, h("h1", null, "Live"),
          h("p", { class: "sub" }, "Your own agent, recorded as you use it — and any call of it a breakpoint holds here, while your harness just sees a slow response."))),
        listEl,
        detailEl),
      h("aside", { class: "live-side" },
        panel("Breakpoints", [breakpointsEl, breakpointForm()]),
        panel("Recently decided", recentEl),
        connectPanel())),
  );

  forgetDecided();
  drawList();
  drawDetail();
  drawBreakpoints();
  loadRecent();

  scope.every(tick, 1000);
  scope.on((event) => {
    switch (event.type) {
      case "snapshot":
        forgetDecided(); // decided while the stream was down
        drawList();
        drawDetail();
        drawBreakpoints();
        break;
      case "hold":
        drawList();
        if (!selected || !pending().some((hold) => hold.id === selected)) {
          selected = event.hold.id;
          remember(selected);
          drawDetail();
        }
        break;
      case "resolved":
        forget(event.hold.id);
        drawList();
        loadRecent();
        if (event.hold.id === selected) drawDetail();
        break;
      case "breakpoints":
        drawBreakpoints();
        break;
    }
  });

  // --- held calls -------------------------------------------------------------------

  // The calls still waiting, as far as this page knows.
  function pending() {
    return live.holds.filter((hold) => !settled.has(hold.id));
  }

  // Whether a hold is still the one to show: chosen, and still waiting.
  function current(id) {
    return selected === id && pending().some((hold) => hold.id === id);
  }

  // A decided call's edit and payload are of no more use.
  function forget(id) {
    edits.delete(id);
    payloads.delete(id);
  }

  function forgetDecided() {
    const waiting = new Set(pending().map((hold) => hold.id));
    for (const id of [...edits.keys(), ...payloads.keys()]) if (!waiting.has(id)) forget(id);
  }

  function remember(id) {
    settle(id ? `#/live/${id}` : "#/live");
  }

  function select(id) {
    selected = id;
    remember(id);
    drawList();
    drawDetail();
  }

  function drawList() {
    const holds = pending();
    // One held call needs no list; the detail below is the whole story.
    listEl.hidden = holds.length < 2;
    replace(listEl, holds.map((hold) =>
      h("button", { class: ["hold-card", hold.id === selected && "selected"], type: "button", onclick: () => select(hold.id) },
        h("span", null, chip(...PHASE[hold.phase].chip), " ", h("span", { class: "timer", dataset: { since: hold.created_at } }, "")),
        h("span", { class: "reason" }, reasonText(hold)),
        h("span", { class: "faint small" }, `${hold.run_id ? `run ${short(hold.run_id)}` : "a new run"} · step ${hold.step}`))));
    tick();
  }

  async function drawDetail() {
    const holds = pending();
    let hold = holds.find((candidate) => candidate.id === selected);
    if (!hold && holds.length) {
      // Whatever was selected is gone — decided here, from the CLI, or abandoned. Move on.
      hold = holds[0];
      selected = hold.id;
      remember(selected);
      drawList();
    }
    if (!hold) {
      shown = null;
      const gone = selected && lastDecision?.id !== selected ? selected : null;
      if (!gone) {
        replace(detailEl, idleCard());
        return;
      }
      const card = await decidedCard(gone);
      // Only if nothing moved on while that loaded: a newer hold, or a decision here.
      if (selected === gone && !pending().length) replace(detailEl, card);
      return;
    }
    // Already on the page, edits and all: drawing it again would only lose the caret.
    if (shown === hold.id) return;
    let full = payloads.get(hold.id);
    if (!full) {
      try {
        full = await api.get(`/holds/${hold.id}`);
      } catch (error) {
        // Unless another hold was chosen, or this one decided, while it loaded.
        if (current(hold.id) && shown !== hold.id) {
          shown = null;
          replace(detailEl, h("div", { class: "error-box" }, error.message));
        }
        return;
      }
      payloads.set(hold.id, full);
    }
    // The same, and another redraw may have put it on the page meanwhile.
    if (!current(hold.id) || shown === hold.id) return;
    shown = hold.id;
    replace(detailEl, holdEditor(full, { deciding, decided, tooLate }), holdTools(full, scope));
    tick();
  }

  // Noted before the decision is sent: the proxy announces the resolution on the event
  // stream, which can arrive before the answer to our own request does.
  function deciding(id, action) {
    lastDecision = id ? { id, action } : null;
  }

  function decided(id, action) {
    settled.add(id);
    forget(id);
    if (left) return; // the page, URL and all, belongs to another view now
    lastDecision = { id, action };
    if (selected !== id) {
      drawList(); // another hold was chosen while the decision was on its way: stay there
      return;
    }
    selected = null;
    const next = pending()[0];
    if (next) select(next.id);
    else {
      shown = null;
      remember(null);
      drawList();
      replace(detailEl, idleCard());
    }
  }

  // A decision refused because the call was no longer waiting: decided elsewhere first, or
  // abandoned by its harness. Its editor, if still on the page, no longer applies.
  function tooLate(id) {
    settled.add(id);
    forget(id);
    if (left) return;
    if (shown === id) shown = null;
    drawList();
    drawDetail();
  }

  function idleCard() {
    if (lastDecision?.action === "step") {
      return h("div", { class: "panel empty" }, h("div", { class: "waiting" }, h("span", { class: "spinner" }), h("h2", null, "Stepping")),
        h("p", null, "The call went on, and will stop again at the next stop point: when the model has answered, or at the harness's next call. If the run has finished, nothing more will arrive."));
    }
    const armed = live.breakpoints.filter((bp) => bp.enabled).length;
    // Nothing held: the way in this page is — your own agent, recorded as you use it — first.
    return h("div", { class: "stack" },
      panel("Record your own agent", h("div", { class: "connect stack" },
        h("p", null, "Set your agent's base URL — where it sends its model calls — to this proxy, and use it as usual:"),
        h("div", { class: "connect-urls" },
          h("span", { class: "muted small" }, "an OpenAI-style API, such as Hermes Agent"), command(`${location.origin}/v1`),
          h("span", { class: "muted small" }, "Anthropic's, such as Claude Code"), command(location.origin)),
        h("p", { class: "hint" }, "Each call it makes is recorded as a step of a run, recorded live: its runs appear on ",
          h("a", { href: "#/runs" }, "Runs"), ", where any step can be resampled, A/B-tested or refined. A breakpoint stops a call here first."))),
      h("div", { class: "panel empty" }, h("h2", null, "Nothing is held"),
        h("p", null, armed
          ? `${plural(armed, "breakpoint")} armed. A matching call will appear here the moment it arrives.`
          : "No breakpoint is armed, so every call passes straight through. Add one on the right.")));
  }

  // A call decided somewhere else — the CLI, another tab — or abandoned by the harness.
  async function decidedCard(id) {
    let hold;
    try {
      hold = await api.get(`/holds/${id}`);
    } catch {
      return idleCard();
    }
    const what = hold.abandoned
      ? "abandoned: the harness hung up while it was held"
      : `decided elsewhere: ${hold.action}${hold.edited ? ", with an edit" : ""}`;
    return h("div", { class: "panel empty" }, h("h2", null, `Hold ${hold.id} was ${what}`),
      hold.run_id ? h("p", null, h("a", { href: `#/runs/${hold.run_id}` }, `See run ${short(hold.run_id)}`)) : null);
  }

  function tick() {
    tickUnder(root);
  }

  async function loadRecent() {
    let recent = [];
    try {
      recent = (await api.get("/holds?recent=1")).filter((hold) => !hold.pending).slice(0, 8);
    } catch {
      return;
    }
    if (!recent.length) {
      replace(recentEl, h("p", { class: "muted small" }, "Nothing decided yet."));
      return;
    }
    const tones = { drop: "warn" };
    replace(recentEl, recent.map((hold) =>
      h("div", { class: "recent-item" },
        hold.abandoned ? chip("abandoned", "warn") : chip(hold.action, tones[hold.action] || "muted"),
        hold.edited ? chip("edited", "change") : null,
        h("span", { class: "reason", title: hold.reason }, hold.reason),
        hold.run_id ? h("a", { class: "mono small", href: `#/runs/${hold.run_id}` }, short(hold.run_id)) : null)));
  }

  // --- breakpoints ------------------------------------------------------------------

  function drawBreakpoints() {
    if (!live.breakpoints.length) {
      replace(breakpointsEl, h("p", { class: "muted small" }, "No breakpoints. Without one, every call passes straight through."));
      return;
    }
    replace(breakpointsEl, live.breakpoints.map((bp) =>
      h("div", { class: ["bp", !bp.enabled && "bp-off"] },
        h("label", { class: "switch", title: bp.enabled ? "On — click to turn off" : "Off — click to turn on" },
          h("input", {
            type: "checkbox",
            checked: bp.enabled,
            "aria-label": `Breakpoint ${bp.id} enabled`,
            "data-control": true,
            onchange: guard((event) => api.patch(`/breakpoints/${bp.id}`, { enabled: event.target.checked }).catch((error) => toast(error.message, { tone: "bad" }))),
          }),
          h("span", { class: "switch-track" })),
        h("div", { class: "bp-main" },
          bp.label ? h("div", { class: "bp-label" }, bp.label) : null,
          h("div", { class: "bp-desc" }, bp.description)),
        h("span", { class: "faint small", title: "Calls this breakpoint has stopped" }, plural(bp.hits, "hit")),
        h("button", {
          class: "icon-btn",
          type: "button",
          title: "Remove",
          "aria-label": `Remove breakpoint ${bp.id}`,
          "data-control": true,
          onclick: guard(() => api.del(`/breakpoints/${bp.id}`).catch((error) => toast(error.message, { tone: "bad" }))),
        }, "×"))));
  }

  function connectPanel() {
    return panel("Connect your agent", h("div", { class: "connect stack" },
      h("p", { class: "hint" }, "Its base URL, for an OpenAI-style API; for Anthropic's, the same without /v1:"),
      command(`${location.origin}/v1`),
      h("p", { class: "hint" }, "The same controls from a terminal:"),
      command("tuningfork hold list")));
  }
}

// --- the hold editor ------------------------------------------------------------------

function holdEditor(hold, { deciding, decided, tooLate }) {
  const phase = hold.phase;
  const original = phase === "request" ? hold.request : hold.response;
  const unedited = JSON.stringify(original);
  const shapeError = phase === "request" ? requestShapeError : responseShapeError;
  let edit = edits.get(hold.id);
  if (!edit) {
    // `rawText` is the JSON as typed, which may not parse; `invalid` maps each JSON-valued
    // field whose text does not parse to that text.
    edit = { draft: structuredClone(original), raw: false, rawText: null, rawError: null, invalid: new Map() };
    edits.set(hold.id, edit);
  }
  const drawn = new Set(); // the JSON-valued fields on the page, by id

  const body = h("div", { class: "editor" });
  const status = h("span", { class: "edit-status" });
  const revert = h("button", { class: "btn btn-quiet btn-small", type: "button", onclick: revertEdits }, "Revert edits");
  const rawToggle = h("button", { class: "btn btn-small", type: "button", onclick: switchRaw }, "Edit as JSON");
  const continueButton = h("button", { class: "btn btn-primary", type: "button", "data-control": true, onclick: guard(() => send("continue")) }, "Continue");
  const stepButton = h("button", {
    class: "btn",
    type: "button",
    title: phase === "request" ? "Send it on, and stop again when the model has answered" : "Hand it to the harness, and stop before its next call",
    "data-control": true,
    onclick: guard(() => send("step")),
  }, "Step");
  const dropButton = h("button", { class: "btn btn-danger", type: "button", "data-control": true, onclick: guard(armDrop) }, "Drop");
  let sending = false; // a decision is on its way: nothing enables the buttons until it lands

  const changed = () => JSON.stringify(edit.draft) !== unedited;

  function refresh() {
    const edited = !edit.rawError && changed();
    status.className = ["edit-status", edit.rawError || edit.invalid.size ? "error" : edited && "edited"].filter(Boolean).join(" ");
    status.textContent = edit.rawError
      ? `Not valid JSON: ${edit.rawError}`
      : edit.invalid.size
        ? "Fix the fields marked in red first"
        : edited
          ? "Edited — Continue and Step send your version"
          : "Unedited";
    const blocked = Boolean(edit.rawError) || edit.invalid.size > 0;
    continueButton.disabled = blocked || sending;
    stepButton.disabled = blocked || sending;
    continueButton.textContent = edited ? "Continue with edits" : "Continue";
    revert.hidden = !edited && !blocked;
    // Back to the fields only from valid JSON, of a shape they can show.
    const unfit = edit.raw && !edit.rawError && shapeError(edit.draft);
    rawToggle.disabled = edit.raw && Boolean(edit.rawError || unfit);
    rawToggle.title = edit.rawError ? "Fix the JSON first" : unfit ? `The fields cannot show this ${phase}: ${unfit}` : "";
  }

  function draw() {
    // What the fields cannot show — a response with no message, say — is edited as JSON.
    if (!edit.raw && shapeError(edit.draft)) edit.raw = true;
    drawn.clear();
    rawToggle.textContent = edit.raw ? "Edit as fields" : "Edit as JSON";
    replace(body, edit.raw ? rawEditor() : phase === "request" ? requestEditor() : responseEditor());
    // Only a field on the page can be invalid: nothing unseen may block Continue.
    for (const id of edit.invalid.keys()) if (!drawn.has(id)) edit.invalid.delete(id);
    refresh();
  }

  function switchRaw() {
    if (edit.raw && (edit.rawError || shapeError(edit.draft))) return; // the button says why
    edit.raw = !edit.raw;
    edit.rawText = null;
    // Either way the draft is drawn afresh, so none of its fields starts out invalid.
    edit.invalid.clear();
    draw();
  }

  function revertEdits() {
    Object.assign(edit, { draft: structuredClone(original), rawText: null, rawError: null });
    edit.invalid.clear();
    draw();
  }

  function rawEditor() {
    const area = textArea(edit.rawText ?? pretty(edit.draft), (text) => {
      edit.rawText = text;
      try {
        edit.draft = JSON.parse(text);
        edit.rawError = null;
      } catch (error) {
        edit.rawError = error.message;
      }
      refresh();
    }, { rows: 28, baseline: pretty(original) });
    const whole = `The whole ${phase === "request" ? "request" : "response"}, as JSON.`;
    const unfit = shapeError(edit.draft);
    return [h("p", { class: "hint" }, unfit ? `${whole} The fields cannot show it: ${unfit}.` : whole), area];
  }

  // The request is edited in its own wire format. Text is editable wherever it sits — a
  // message's content, its text blocks, a tool result, an Anthropic `system` field; tool
  // calls and thinking are shown here and edited as JSON. `before` is the holder as it
  // arrived, which a changed field is marked against.
  function contentEditor(holder, key, before, onChange) {
    const value = holder[key];
    const was = before?.[key];
    if (typeof value === "string") {
      return textArea(value, (text) => { holder[key] = text; onChange(); }, { baseline: was });
    }
    if (!Array.isArray(value)) return null;
    return value.map((block, index) => {
      const blockWas = Array.isArray(was) ? was[index] : undefined;
      if (block && typeof block.text === "string") {
        return argument(block.type || "text", "text", textArea(block.text, (text) => { block.text = text; onChange(); }, { baseline: blockWas?.text }));
      }
      if (block?.type === "tool_result") {
        return h("div", null, h("div", { class: "arg-key" }, h("span", { class: "mono" }, "tool result"), h("span", { class: "faint" }, ` · ${block.tool_use_id || ""}`)),
          contentEditor(block, "content", blockWas, onChange));
      }
      return h("pre", { class: "code arg-value" }, pretty(block));
    });
  }

  function requestEditor() {
    const draft = edit.draft;
    const messages = draft.messages;
    const befores = Array.isArray(original?.messages) ? original.messages : [];
    const names = toolNames(messages);
    const from = Math.max(0, messages.length - 2);
    const system = "system" in draft
      ? (() => {
          const block = h("details", { class: "msg msg-system" });
          const mark = () => block.classList.toggle("msg-edited", JSON.stringify(draft.system) !== JSON.stringify(original?.system));
          replace(block,
            h("summary", { class: "msg-head" }, roleChip("system"), h("span", { class: "faint mono" }, "system field"),
              h("span", { class: "msg-preview" }, preview({ content: draft.system }))),
            h("div", { class: "msg-body" }, contentEditor(draft, "system", original, () => { mark(); refresh(); })));
          if (mark()) block.open = true; // an edit starts in view
          return block;
        })()
      : null;
    return [
      h("p", { class: "hint" }, "The context the model is about to see. Edit any message; tuningfork records both your version and the harness's."),
      system,
      messages.map((message, index) => {
        const before = befores[index];
        const block = h("details", { class: ["msg", `msg-${roleClass(message.role)}`], open: index >= from && message.role !== "system" });
        const mark = () => block.classList.toggle("msg-edited", JSON.stringify(message) !== JSON.stringify(before));
        replace(
          block,
          h("summary", { class: "msg-head" },
            roleChip(message.role),
            h("span", { class: "faint mono" }, `#${index}`),
            message.role === "tool" ? h("span", { class: "muted" }, `result of ${names.get(message.tool_call_id) || "a tool call"}`) : null,
            h("span", { class: "msg-preview" }, preview(message)),
            h("span", { class: "faint small" }, `${count(messageSize(message))} chars`)),
          h("div", { class: "msg-body" },
            reasoningView(message),
            contentEditor(message, "content", before, () => { mark(); refresh(); }),
            (message.tool_calls || []).map(toolCallView)),
        );
        if (mark()) block.open = true; // an edit starts in view
        return block;
      }),
    ];
  }

  function responseEditor() {
    const message = edit.draft.choices[0].message;
    const before = responseMessage(original);
    const parts = [
      h("p", { class: "hint" }, "What the model decided. The harness has not seen it yet: edit it, and the harness acts on your version. The step still records what the model produced."),
      reasoningView(message),
    ];
    if (typeof message.content === "string" && (message.content || !message.tool_calls?.length)) {
      parts.push(h("div", null, h("div", { class: "label" }, "Content"),
        textArea(message.content, (value) => { message.content = value; refresh(); }, { baseline: before?.content })));
    }
    (message.tool_calls || []).forEach((call, position) => parts.push(toolCallEditor(call, before?.tool_calls?.[position], position)));
    return parts;
  }

  // `position` is the call's place in the message, which names its fields: a call's id can
  // be missing, or repeated.
  function toolCallEditor(call, before, position) {
    const fn = call.function;
    const parsed = parseArguments(fn.arguments);
    const was = parseArguments(before?.function?.arguments);
    const name = h("input", {
      class: "field",
      value: fn.name || "",
      "aria-label": "Tool name",
      oninput: (event) => { fn.name = event.target.value; refresh(); },
    });
    let fields;
    if (parsed.object) {
      const values = { ...parsed.object };
      const drawnFrom = fn.arguments;
      const initial = JSON.stringify(was.object ?? null);
      // An untouched call keeps its original arguments string byte for byte; an edited one
      // changes only the values that were edited.
      const write = () => {
        fn.arguments = JSON.stringify(values) === initial
          ? before.function.arguments
          : spliceArguments(drawnFrom, values, before?.function?.arguments) ?? JSON.stringify(values);
        refresh();
      };
      fields = Object.entries(values).map(([key, value]) => {
        const had = was.object && Object.hasOwn(was.object, key) ? was.object[key] : undefined;
        if (typeof value === "string") {
          // Strings are edited raw: the content of a .json file, not an escaped copy.
          return argument(key, "string", textArea(value, (text) => { values[key] = text; write(); }, { baseline: had }));
        }
        const id = `${position}:${key}`;
        drawn.add(id);
        const typed = edit.invalid.get(id); // what was typed before a redraw, and did not parse
        const area = textArea(typed ?? pretty(value), (text) => {
          try {
            values[key] = JSON.parse(text);
          } catch {
            edit.invalid.set(id, text);
            area.classList.add("field-invalid");
            refresh();
            return;
          }
          edit.invalid.delete(id);
          area.classList.remove("field-invalid");
          write();
        }, { baseline: pretty(had) });
        if (typed !== undefined) area.classList.add("field-invalid");
        return argument(key, "JSON value", area);
      });
    } else {
      fields = [argument("arguments", "not JSON — sent as written", textArea(parsed.raw, (text) => { fn.arguments = text; refresh(); }, { baseline: was.raw }))];
    }
    return h("div", { class: "toolcall" },
      h("div", { class: "toolcall-head" }, chip("tool call", "accent"), name),
      h("div", { class: "args" }, fields));
  }

  let dropArmed = null;
  function armDrop() {
    if (dropArmed) {
      clearTimeout(dropArmed);
      dropArmed = null;
      send("drop");
      return;
    }
    dropButton.classList.add("armed");
    dropButton.textContent = "Click again: fail this call";
    dropArmed = setTimeout(disarmDrop, 4000);
  }

  function disarmDrop() {
    clearTimeout(dropArmed);
    dropArmed = null;
    dropButton.classList.remove("armed");
    dropButton.textContent = "Drop";
  }

  async function send(action) {
    if (sending) return;
    sending = true;
    const edited = action !== "drop" && changed();
    const body = edited ? { [PHASE[phase].editable]: edit.draft } : {};
    for (const button of [continueButton, stepButton, dropButton]) button.disabled = true;
    deciding(hold.id, action);
    try {
      await api.post(`/holds/${hold.id}/${action}`, body);
    } catch (error) {
      deciding(null);
      if (error.status === 409 || error.status === 404) {
        // No longer waiting: decided elsewhere first, or abandoned. This editor may be off
        // the page already; the page shows what did happen instead.
        toast(`Hold ${hold.id} was no longer waiting: decided elsewhere, or abandoned`, { tone: "warn" });
        tooLate(hold.id);
        return;
      }
      sending = false;
      disarmDrop(); // a failed Drop, or one armed meanwhile: the next click arms it afresh
      for (const button of [continueButton, stepButton, dropButton]) button.disabled = false;
      refresh();
      // After refresh(), which would otherwise write over it at once.
      status.className = "edit-status error";
      status.textContent = error.message;
      return;
    }
    toast(`${action === "drop" ? "Dropped" : action === "step" ? "Stepped" : "Continued"} ${hold.id}${edited ? " with your edit" : ""}`, { tone: action === "drop" ? "warn" : "accent", timeout: 3000 });
    decided(hold.id, action);
  }

  const where = hold.run_id
    ? [h("a", { href: `#/runs/${hold.run_id}` }, `run ${short(hold.run_id)}`), ` · step ${hold.step}`]
    : `a new run · step ${hold.step}`;
  const card = h(
    "section",
    { class: "panel" },
    h("div", { class: "hold-head" },
      chip(...PHASE[phase].chip),
      h("span", { class: "reason" }, reasonText(hold)),
      h("span", { class: "hold-where muted" }, where, hold.task_id ? ` · task ${hold.task_id}` : ""),
      h("span", { class: "timer pulse", title: "Time held", dataset: { since: hold.created_at } }, elapsed(hold.held_seconds))),
    h("div", { class: "editor-tools", style: { padding: "10px 16px 0" } }, rawToggle, revert),
    body,
    h("div", { class: "action-bar" }, continueButton, stepButton, dropButton, status),
  );
  draw();
  return card;
}

function argument(key, kind, field) {
  return h("div", { class: "arg" }, h("div", { class: "arg-key" }, h("span", { class: "mono" }, key), h("span", { class: "faint" }, ` · ${kind}`)), field);
}

// `baseline` is the field's text as the call arrived, when it had one: a field that
// differs is marked as changed, and stays marked when the editor is drawn again.
function textArea(value, onInput, { rows: maxRows = 22, baseline } = {}) {
  const text = String(value ?? "");
  const from = typeof baseline === "string" ? baseline : text;
  const lines = text.split("\n").length;
  const rows = Math.min(maxRows, lines === 1 ? 1 : lines + 1);
  const area = h("textarea", { class: ["field", text !== from && "field-changed"], rows: String(rows), spellcheck: "false", value: text });
  area.addEventListener("input", () => {
    area.classList.toggle("field-changed", area.value !== from);
    onInput(area.value);
  });
  return area;
}

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

// Tool calls, if there are any, as a list of calls each of which `fits`.
const callsFit = (calls, fits) => !calls || (Array.isArray(calls) && calls.every(fits));

// Why the fields cannot show a payload, when they cannot: it is edited as JSON instead. A
// held response can lack a message altogether — an OpenAI stream that carried only an
// error assembles to `choices: []`.
function requestShapeError(request) {
  if (!Array.isArray(request?.messages)) return "it has no `messages` list";
  const fits = (message) => isObject(message) && callsFit(message.tool_calls, isObject);
  return request.messages.every(fits) ? null : "one of its messages, or a tool call in one, is not an object";
}

function responseShapeError(response) {
  const message = response?.choices?.[0]?.message;
  if (!isObject(message)) return "it has no `choices[0].message`";
  const fits = (call) => isObject(call) && isObject(call.function);
  return callsFit(message.tool_calls, fits) ? null : "one of its tool calls has no `function` object";
}

// --- a tool call's arguments, edited in place -------------------------------------------

// `text`, the arguments the fields were drawn from, with `values` written into it. A value
// that is as it was keeps its exact text — the model's own where it matches, so a value
// edited back is the model's again: `30.0` stays `30.0`, a 64-bit id keeps its digits.
// Only edited values are written anew. Null unless `text` is an object of just these keys.
function spliceArguments(text, values, original) {
  const spans = memberSpans(text);
  const keys = Object.keys(values);
  if (!spans || spans.size !== keys.length || !keys.every((key) => spans.has(key))) return null;
  const originals = memberSpans(original) ?? new Map();
  let spliced = "";
  let at = 0;
  for (const [key, [start, end]] of spans) {
    const value = JSON.stringify(values[key]);
    const kept = [originals.has(key) && original.slice(...originals.get(key)), text.slice(start, end)]
      .find((was) => was && JSON.stringify(JSON.parse(was)) === value);
    spliced += text.slice(at, start) + (kept || value);
    at = end;
  }
  return spliced + text.slice(at);
}

// Where each top-level value of a JSON object's text starts and ends, by key, in the order
// written; null for anything but a JSON object, or one that gives a key twice.
function memberSpans(text) {
  try {
    if (!isObject(JSON.parse(text))) return null;
  } catch {
    return null;
  }
  const spans = new Map();
  let at = skipSpace(text, 0);
  if (text[at] !== "{") return null;
  at = skipSpace(text, at + 1);
  if (text[at] === "}") return spans;
  while (text[at] === '"') {
    const keyEnd = valueEnd(text, at);
    const key = JSON.parse(text.slice(at, keyEnd));
    at = skipSpace(text, keyEnd);
    if (text[at] !== ":") return null;
    const start = skipSpace(text, at + 1);
    const end = valueEnd(text, start);
    if (end <= start || spans.has(key)) return null;
    spans.set(key, [start, end]);
    at = skipSpace(text, end);
    if (text[at] === "}") return spans;
    if (text[at] !== ",") return null;
    at = skipSpace(text, at + 1);
  }
  return null;
}

// Just past the JSON value that starts at `at`: a string, an object or list, or a literal.
function valueEnd(text, at) {
  if (text[at] === '"') {
    let i = at + 1;
    while (i < text.length && text[i] !== '"') i += text[i] === "\\" ? 2 : 1;
    return i + 1;
  }
  if (text[at] === "{" || text[at] === "[") {
    let depth = 0;
    for (let i = at; i < text.length; i += 1) {
      if (text[i] === '"') i = valueEnd(text, i) - 1;
      else if (text[i] === "{" || text[i] === "[") depth += 1;
      else if ((text[i] === "}" || text[i] === "]") && --depth === 0) return i + 1;
    }
    return -1;
  }
  let i = at;
  while (i < text.length && !",}] \t\n\r".includes(text[i])) i += 1;
  return i;
}

function skipSpace(text, at) {
  while (at < text.length && " \t\n\r".includes(text[at])) at += 1;
  return at;
}

// --- adding a breakpoint --------------------------------------------------------------

const RESPONSE_FIELDS = ["tool", "path_glob", "args_regex", "output_regex", "diverges_from"];

function breakpointForm() {
  const inputs = {};
  const input = (name, placeholder, type = "text") => {
    inputs[name] = h("input", { class: "field", name, placeholder, type, oninput: update, autocomplete: "off" });
    return inputs[name];
  };
  const field = (label, control, wide = false) => h("label", { class: ["form-field", wide && "wide"] }, label, control);

  const phase = h("select", { class: "field", onchange: update, "aria-label": "When to stop" },
    h("option", { value: "auto" }, "from the conditions below"),
    h("option", { value: "request" }, "before the model sees the context"),
    h("option", { value: "response" }, "after the model answers"));
  const once = h("input", { type: "checkbox" });
  // Response-side, like the fields: the model called a tool without reasoning first.
  const noReasoning = h("input", { type: "checkbox", name: "no_reasoning", onchange: update });
  const summary = h("p", { class: "bp-summary" });
  const error = h("p", { class: "form-error", hidden: true });

  const responseSet = h("fieldset", null, h("legend", null, "What the model does"),
    h("div", { class: "form-grid" },
      field("Tool", input("tool", "write_file, write_*")),
      field("Path argument", input("path_glob", "*.json")),
      field("Arguments match", input("args_regex", "regex over key=value lines"), true),
      field("Output matches", input("output_regex", "regex over text, reasoning, calls"), true),
      field("Behaves unlike run", input("diverges_from", "run id or label"), true),
      h("label", { class: "check wide" }, noReasoning, "calls a tool without reasoning first")));
  const requestSet = h("fieldset", null, h("legend", null, "Where in the run"),
    h("div", { class: "form-grid" },
      field("From step", input("step_min", "0", "number")),
      field("To step", input("step_max", "", "number")),
      field("After this tool returns", input("after_tool", "read_file"), true),
      field("Context matches", input("context_regex", "regex over the messages"), true),
      field("Only run", input("run", "run id or label")),
      field("Only task", input("task", "task id"))));

  const form = h("form", { class: "bp-form", onsubmit: guard(submit) },
    h("label", { class: "form-field" }, "Stop", phase),
    responseSet,
    requestSet,
    h("div", { class: "form-row" }, h("label", { class: "check" }, once, "only once"), input("label", "name (optional)")),
    summary,
    error,
    h("div", null, h("button", { class: "btn btn-primary", type: "submit", "data-control": true }, "Add breakpoint")));

  const wrapper = h("details", { class: "add-bp", open: live.breakpoints.length === 0 }, h("summary", null, "Add a breakpoint"), form);

  function resolvedPhase() {
    if (phase.value !== "auto") return phase.value;
    return noReasoning.checked || RESPONSE_FIELDS.some((name) => inputs[name].value.trim()) ? "response" : "request";
  }

  function update() {
    const resolved = resolvedPhase();
    responseSet.disabled = phase.value === "request";
    summary.textContent = resolved === "response"
      ? "Stops after the model answers, before your harness acts on it — you can edit what it decided."
      : "Stops before the model sees the context — you can edit what it will read.";
    error.hidden = true;
  }

  async function submit(event) {
    event.preventDefault();
    const definition = { phase: resolvedPhase() };
    for (const [name, control] of Object.entries(inputs)) {
      const value = control.value.trim();
      if (!value) continue;
      if (definition.phase === "request" && RESPONSE_FIELDS.includes(name)) continue;
      definition[name] = control.type === "number" ? Number(value) : value;
    }
    if (once.checked) definition.once = true;
    if (noReasoning.checked && definition.phase === "response") definition.no_reasoning = true;
    try {
      await api.post("/breakpoints", definition);
      form.reset();
      update();
      wrapper.open = false;
    } catch (failure) {
      error.textContent = failure.message;
      error.hidden = false;
    }
  }

  update();
  return wrapper;
}
