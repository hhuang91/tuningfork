// One step's exact context: every message the model saw, as sent, with the skill marked in
// it — what a resample of the step resends. Where the harness sent something else (a person
// edited the call at a breakpoint, or a fork swapped a variant wording in), both are here.
// The run's timeline shows the step in its conversation; this is the step alone.

import { api, enc } from "../api.js";
import { crumbs, h, loading, outcome, panel, plural, pretty, replace, short, toggle } from "../dom.js";
import { messagesView, noReasoning, responseView } from "../payload.js";
import { banners, detailsPanel, skillsSummary, summaryStrip, toolsOffered, workspacePanel } from "../stepinfo.js";

// What the harness's version of the request is, by what changed it for the model.
const HARNESS_LABEL = {
  breakpoint: "What the harness sent",
  variant: "The original wording (what the harness sent)",
};

export async function stepView(root, [reference, index], scope) {
  replace(root, loading());
  const step = await api.get(`/runs/${enc(reference)}/steps/${Number(index)}`);
  const run = step.run;
  const changedBy = step.request_changed_by;
  const editedResponse = step.edits?.response;

  const state = { context: "model", contextRaw: false, response: "model", responseRaw: false };
  const contextEl = h("div");
  const contextControls = h("span", { class: "controls" });
  const responseEl = h("div");
  const responseControls = h("span", { class: "controls" });

  const at = (idx) => `#/runs/${run.id}/${idx}/context`;
  const previous = step.idx > 0 ? at(step.idx - 1) : null;
  const next = step.idx < run.step_count - 1 ? at(step.idx + 1) : null;
  const walk = (href, label) =>
    href ? h("a", { class: "btn btn-small", href }, label) : h("button", { class: "btn btn-small", type: "button", disabled: true }, label);
  const onTimeline = `#/runs/${run.id}/${step.idx}`;

  const messages = step.view || [];
  replace(
    root,
    crumbs([["Runs", "#/runs"], run.task_id ? [run.task_id, `#/divergence/${enc(run.task_id)}`] : null,
      [`run ${short(run.id)}`, `#/runs/${run.id}`], [`step ${step.idx}`, onTimeline], ["exact context"]]),
    h("div", { class: "page-head" },
      h("div", null,
        h("h1", null, `Step ${step.idx}`, h("span", { class: "h-sub" }, `exact context · of ${plural(run.step_count, "step")} in run ${short(run.id)}`)),
        h("p", { class: "sub" }, "Every message as the model saw it: what a resample of this step resends.")),
      h("div", { class: "actions" },
        h("a", { class: "btn btn-small", href: onTimeline }, "In the conversation, and work from it →"),
        h("a", { class: "btn btn-quiet btn-small", href: `${onTimeline}?view=tokens`, title: "How sure the model was of each token of its answer" }, "Its answer's tokens"),
        walk(previous, "← previous"), walk(next, "next →"))),
    summaryStrip(step),
    banners(step),
    h("div", { class: "grid-2", style: { marginTop: "14px" } },
      h("div", { class: "stack" },
        panel("Context", contextEl, {
          meta: h("span", { class: "muted" }, `${plural(messages.length, "message")} · ${plural((step.request?.tools || []).length, "tool")} offered`),
          actions: contextControls,
        })),
      h("div", { class: "stack sticky-col" },
        panel("What the model did", responseEl, { actions: responseControls }),
        detailsPanel(step),
        workspacePanel(step))),
  );
  drawContext();
  drawResponse();

  // ← and → walk the run's steps, unless you are typing.
  const onKey = (event) => {
    if (event.target.closest("input, textarea, select") || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "ArrowLeft" && previous) location.hash = previous;
    if (event.key === "ArrowRight" && next) location.hash = next;
  };
  document.addEventListener("keydown", onKey);
  scope.onLeave(() => document.removeEventListener("keydown", onKey));

  function drawContext() {
    replace(
      contextControls,
      // Whenever the harness sent something else: an edit at a breakpoint, or the original
      // wording a fork's variant replaced.
      step.harness_request
        ? toggle([["model", "What the model saw"], ["harness", HARNESS_LABEL[changedBy] || HARNESS_LABEL.breakpoint]], state.context, (value) => {
            state.context = value;
            drawContext();
          })
        : null,
      toggle([[false, "Messages"], [true, "JSON"]], state.contextRaw, (value) => {
        state.contextRaw = value;
        drawContext();
      }),
    );
    // `request` is exactly as sent, in the call's own wire format; `view` is the same
    // conversation as OpenAI-style messages, which is what is rendered.
    const harness = state.context === "harness";
    const request = harness ? step.harness_request : step.request;
    const view = harness ? step.harness_view : step.view;
    if (!request) {
      replace(contextEl, h("p", { class: "muted" }, "Not recorded."));
      return;
    }
    if (state.contextRaw) {
      replace(contextEl, h("pre", { class: "code" }, pretty(request)));
      return;
    }
    // Skill positions are computed on the model's view of the request.
    const spans = state.context === "model" ? step.skills || [] : [];
    // Positions in the view shown here, worked out by the server: an edit's own indices
    // count the request's wire-format messages, which a view can split and reorder.
    const edited = new Set(step.edited_messages ?? step.edits?.request?.messages ?? []);
    const recorded = Object.keys(run.skill_hashes || {}).length > 0;
    replace(
      contextEl,
      spans.length ? skillsSummary(spans, recorded) : null,
      toolsOffered(request.tools),
      messagesView(view || [], { spans, edited, changedBy: changedBy || "breakpoint", recorded }),
    );
  }

  function drawResponse() {
    replace(
      responseControls,
      editedResponse && !editedResponse.dropped
        ? toggle([["model", "What the model produced"], ["delivered", "What the harness received"]], state.response, (value) => {
            state.response = value;
            drawResponse();
          })
        : null,
      toggle([[false, "Decoded"], [true, "JSON"]], state.responseRaw, (value) => {
        state.responseRaw = value;
        drawResponse();
      }),
    );
    const response = state.response === "delivered" ? step.delivered_response : step.response;
    if (!response) {
      replace(responseEl, step.error ? h("div", { class: "banner banner-warn" }, `No response: ${step.error}`) : h("p", { class: "muted" }, "No response was recorded."));
      return;
    }
    // The recorded behaviour is the model's, whichever version is on screen.
    const behaviour = h("div", { class: "behaviour" }, h("span", { class: "label" }, "behaviour"), outcome(step.outcome),
      noReasoning(step) ? [" ", noReasoning(step)] : null);
    replace(responseEl, state.responseRaw
      ? h("pre", { class: "code" }, pretty(response))
      : h("div", { class: "stack" }, state.response === "model" ? behaviour : null, responseView(response)));
  }
}
