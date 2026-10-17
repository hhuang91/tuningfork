// Refining a skill: one step, a target declared first, edits proposed for it — by a person,
// an agent over MCP, or the built-in proposer — and screens of them against the original.
//
// A proposal is a hypothesis until a screen tests it, and a screen at one step is evidence,
// not proof. A survivor is validated with full runs — the training task, and tasks held out
// from the refinement, whose verdicts are shown here and to no agent — and only then may it
// be approved: the one thing here that writes the skill's own file, after the person types
// the code that names exactly what is written where.

import { api, enc, query } from "../api.js";
import { guard } from "../auth.js";
import { ago, chip, crumbs, h, kv, loading, panel, plural, pretty, rateBar, replace, short, when } from "../dom.js";
import { cacheField, field, formError, input, numberOf, planForm, samplingChoices, seedField, textArea } from "../forms.js";
import { ACTIVE } from "../jobs.js";
import { live } from "../live.js";
import { formatP, screenTones } from "../results.js";
import { trashButton, trashedBanner } from "../trash.js";
import { targetPicker } from "../variants.js";

const opened = (job) => (location.hash = `#/jobs/${job.id}`);

/** Who proposed, or started, something: a person, an agent over MCP, or the proposer. */
export function who(entry = {}) {
  if (entry.via === "builtin") return `${entry.model || "the server's default model"} at ${entry.endpoint}`;
  if (entry.via === "mcp") return `an agent over MCP${entry.client ? ` · ${entry.client}` : ""}`;
  return entry.via === "person" ? "a person" : entry.via || "?";
}

const whoChip = (entry) => chip(entry.via === "builtin" ? `proposer · ${who(entry)}` : who(entry),
  entry.via === "builtin" ? "skill" : entry.via === "mcp" ? "accent" : "muted",
  entry.via === "builtin" ? `the built-in proposer, ${entry.wire || ""}${entry.packet ? `, from packet ${entry.packet}` : ""}` : undefined);

// --- the list -------------------------------------------------------------------------------

export async function refinementsView(root, _, scope) {
  replace(root, loading());
  const closed = scope.query?.get("closed") === "1";
  const found = await api.get(`/refinements${closed ? "?closed=1" : ""}`);
  replace(root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Refine"),
      h("p", { class: "sub" }, "A skill refined at one step: a target declared first, then edits proposed for it — by you, by an agent over MCP, or by the built-in proposer — each screened against the original wording.")),
      h("div", { class: "actions" },
        h("a", { class: "btn btn-quiet btn-small", href: closed ? "#/refine" : "#/refine?closed=1" }, closed ? "Open ones only" : "Closed ones too"))),
    found.length
      ? h("section", { class: "panel" }, h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "refinement"), h("th", null, "skill"), h("th", null, "checkpoint"),
            h("th", null, "target"), h("th", { class: "num" }, "proposals"), h("th", null, "started by"), h("th", null, "when"))),
          h("tbody", null, found.map((each) =>
            h("tr", { class: "clickable", onclick: () => (location.hash = `#/refine/${each.id}`) },
              h("td", null, h("a", { class: "mono", href: `#/refine/${each.id}` }, short(each.id)),
                each.status === "open" ? null : [" ", chip(each.status, each.status === "approved" ? "ok" : "muted")]),
              h("td", { class: "mono" }, each.skill),
              h("td", null, h("a", { class: "mono", href: `#/runs/${each.run}/${each.step}`, onclick: (event) => event.stopPropagation() },
                `${short(each.run)} @ step ${each.step}`)),
              h("td", { class: "muted" }, each.target?.description || "—"),
              h("td", { class: "num" }, String(each.proposals)),
              h("td", null, who(each.created_by)),
              h("td", { class: "faint", title: when(each.created_at) }, ago(each.created_at))))))))
      : h("div", { class: "panel empty" }, h("h2", null, closed ? "No refinements" : "No open refinements"),
          h("p", null, live.mode === "live"
            ? "Open a run, choose the step where the model goes wrong, and choose Refine the skill here…"
            : "Start one from a terminal: `tuningfork refine start <run> <step> --target-rules '[…]'`.")));
}

// --- starting one, from a step --------------------------------------------------------------

/** The form a step opens with `?start=refine`: the target, declared before anything else. */
export async function refineStartForm(step) {
  const run = step.run.id;
  const [targets, existing] = await Promise.all([
    api.get(`/runs/${enc(run)}/steps/${step.idx}/targets`),
    api.get(`/refinements?run=${enc(run)}`),
  ]);
  const here = existing.filter((each) => each.step === step.idx);
  const picker = targetPicker(targets, {
    run, idx: step.idx, exploratory: false,
    legend: "What the model should do at this step — declared before anything is proposed or sampled",
  });
  const error = h("div");
  const button = h("button", { class: "btn btn-primary", type: "submit", "data-control": true }, "Start refining");
  const form = h("form", {
    class: "plan-form", novalidate: true,
    onsubmit: guard(async (event) => {
      event.preventDefault();
      replace(error);
      let target;
      try {
        target = readRefineTarget(form);
      } catch (failed) {
        replace(error, formError(failed));
        return;
      }
      button.disabled = true;
      try {
        const made = await api.post("/refinements", { run, step: String(step.idx), ...target });
        location.hash = `#/refine/${made.id}`;
      } catch (failed) {
        button.disabled = false;
        replace(error, formError(failed));
      }
    }),
  },
  here.length
    ? h("div", { class: "banner" }, h("strong", null, "Already refining here: "),
        here.map((each, index) => [index ? ", " : "", h("a", { href: `#/refine/${each.id}` }, `${short(each.id)} (${each.target?.description || "?"})`)]),
        ". The same target opens that one again.")
    : null,
  h("p", { class: "hint" }, "The skill as this run was given it is copied into the refinement, and every proposal edits that copy. The skill's own folder is never written."),
  picker,
  // A target no answer at this step can express: what the run wrote, as the task's check reads it.
  h("label", { class: "target-choice" },
    h("input", { type: "radio", name: "target_kind", value: "check" }),
    h("span", null, h("strong", null, "The task's check passes"),
      h("span", { class: "muted small" }, " — for a failure only the check can see, a property of what the run wrote: its screens fork the run from here to the end, the check deciding"))),
  error,
  h("div", { class: "form-actions" }, button));
  return form;
}

function readRefineTarget(form) {
  const kind = form.querySelector('[name="target_kind"]:checked')?.value;
  if (!kind) {
    throw Object.assign(new Error("Choose what the model should do here: a behaviour, one you describe, a pattern, or the task's check."), { field: "target_kind" });
  }
  if (kind === "check") return { target_check: true };
  if (kind === "letter") {
    const letter = form.querySelector('[name="target"]:checked')?.value;
    if (!letter) throw Object.assign(new Error("Choose the behaviour."), { field: "target" });
    return { target: letter };
  }
  if (kind === "pattern") {
    const regex = form.querySelector('[name="target_match"]').value.trim();
    if (!regex) throw Object.assign(new Error("Write the pattern."), { field: "target_match" });
    return { target_match: regex };
  }
  return { target_rules: form.querySelector(".target-picker").describe.read() };
}

// --- one refinement ---------------------------------------------------------------------------

export async function refinementView(root, [id], scope) {
  replace(root, loading());
  let shown = await api.get(`/refinements/${enc(id)}`);
  // Ticked to screen: kept while the proposals are drawn again.
  const selected = new Set();
  // The page in parts, each drawn again on its own: what a job changes is redrawn when it
  // ends, and a form someone is typing in is left as it is.
  const head = h("div", { class: "page-head" });
  const banners = h("div", { class: "stack" });
  const facts = h("div");
  const evidenceEl = h("div");
  const proposalsEl = h("div");
  const proposeEl = h("div");
  const screenEl = h("div");
  const screensEl = h("div");
  const validateEl = h("div");
  const validationsEl = h("div");
  const approveEl = h("div");
  let screenForm = null;
  let validateForm = null;
  // The evidence asked for most recently: an older answer arriving late is not drawn.
  let asked = 0;
  let timer = null;
  replace(root, crumbs([["Refine", "#/refine"], [short(shown.refinement.id)]]), head,
    h("div", { class: "stack" }, banners, facts, evidenceEl, proposalsEl, proposeEl, screenEl, screensEl,
      validateEl, validationsEl, approveEl));

  // Nothing new starts on a refinement closed, approved, or in the trash.
  const acting = () => live.mode === "live" && shown.refinement.status === "open" && !shown.refinement.trashed_at;
  const mine = (job) => {
    const ref = job?.params?.refinement || "";
    return ref !== "" && shown.refinement.id.startsWith(ref);
  };

  async function reload() {
    shown = await api.get(`/refinements/${enc(id)}`);
    if (!scope.active) return;
    drawAll();
  }

  function drawAll() {
    drawHead();
    drawBanners();
    drawFacts();
    drawEvidence();
    drawProposals();
    drawForms();
    drawScreens();
    drawValidations();
  }

  // A job about this refinement: said while it runs; when it ends — proposals made, a screen
  // done — what it changed is read again.
  scope.on((event) => {
    if (event.type === "snapshot") drawBanners();
    if (event.type !== "job" || !mine(event.job)) return;
    drawBanners();
    if (ACTIVE.has(event.job.status)) return;
    clearTimeout(timer);
    timer = setTimeout(reload, 100);
  });
  scope.onLeave(() => clearTimeout(timer));
  drawAll();

  function drawHead() {
    const found = shown.refinement;
    replace(head,
      h("div", null,
        h("h1", null, `Refining ${found.skill}`, h("span", { class: "h-sub mono" }, short(found.id))),
        h("p", { class: "sub" }, "At ", h("a", { class: "mono", href: `#/runs/${found.run}/${found.step}` }, `run ${short(found.run)}, step ${found.step}`),
          found.task_id ? h("span", { class: "muted" }, ` (${found.task_id})`) : null,
          ", toward what was declared first: ", h("strong", null, found.target?.description || "?"))),
      h("div", { class: "actions" }, found.status === "open" ? null : chip(found.status, found.status === "approved" ? "ok" : "muted"),
        acting() ? closeButton() : null,
        found.trashed_at || live.jobs.some(mine) ? null
          : trashButton({ kind: "refinement", id: found.id, onDone: () => reload() })));
  }

  function drawBanners() {
    const found = shown.refinement;
    const running = live.jobs.filter(mine);
    replace(banners,
      found.trashed_at
        ? trashedBanner({
            kind: "refinement", id: found.id, trashedAt: found.trashed_at, what: "refinement",
            onRestored: () => reload(),
            onPurged: () => { location.hash = "#/trash"; },
          })
        : null,
      found.source && !found.source.matches
        ? h("div", { class: "banner banner-warn" }, h("strong", null, "The skill's own file has changed since this run was given it. "),
            "Proposals edit the run's copy, which is what the model saw here.")
        : null,
      found.status === "closed" ? h("div", { class: "banner" }, h("strong", null, "Closed. "), "It takes no more proposals or screens; what it holds is kept.") : null,
      approvalBanner(),
      running.map((job) => h("div", { class: "banner" },
        h("strong", null, { propose: "Asking the proposer… ", validate: "Validating… " }[job.kind] || "Screening… "),
        h("a", { href: `#/jobs/${job.id}` }, "Its job"), " shows its progress; what it makes appears here when it ends.")));
  }

  function drawFacts() {
    const found = shown.refinement;
    replace(facts, kv([
      ["skill", h("span", null, h("span", { class: "mono" }, found.baseline.path), h("span", { class: "muted small" }, " — the run's copy"))],
      ["its own file", found.source ? h("span", { class: "mono" }, found.source.path) : h("span", { class: "muted" }, "not known: the run's task named no skill folder")],
      ["read by", found.normalizers?.length ? found.normalizers.join(", ") : null],
      ["started by", h("span", null, who(found.created_by), h("span", { class: "faint" }, ` · ${when(found.created_at)}`))],
    ]));
  }

  function closeButton() {
    const button = h("button", { class: "btn btn-quiet btn-small", type: "button", "data-control": true }, "Close it");
    let armed = null;
    button.addEventListener("click", guard(async () => {
      if (!armed) {
        button.textContent = "Click again: close it";
        armed = setTimeout(() => { armed = null; button.textContent = "Close it"; }, 4000);
        return;
      }
      clearTimeout(armed);
      button.disabled = true;
      try {
        await api.post(`/refinements/${enc(shown.refinement.id)}/close`);
        await reload();
      } catch (error) {
        button.disabled = false;
        button.textContent = error.message;
      }
    }));
    return button;
  }

  // --- the evidence: the packet, as a proposer would be shown it now -------------------------

  function drawEvidence() {
    const found = shown.refinement;
    const body = h("div", null, loading("Gathering the evidence…"));
    replace(evidenceEl, panel("The evidence", body,
      { meta: h("span", { class: "muted" }, "what a proposer is shown: built from what is recorded, never sampled for it") }));
    const mineAsked = (asked += 1);
    api.get(`/refinements/${enc(found.id)}/packet`)
      .then((packet) => evidenceBody(found, packet))
      .catch((error) => h("p", { class: "form-error" }, error.message))
      .then((drawn) => { if (scope.active && mineAsked === asked) replace(body, drawn); });
  }

  function evidenceBody(found, evidence) {
    const packet = evidence.packet;
    const answers = packet.answers_here || {};
    const runs = packet.runs_here || {};
    return h("div", { class: "stack", style: { gap: "8px" } },
      kv([
        ["answers here", h("span", null, answers.summary || "—", answers.rate ? [" ", rateBar(answers.rate)] : null)],
        ["runs of the task", runs.summary],
        ["the task's criteria", packet.included?.task_criteria ? "shown: its prompt, expected output and check" : "left out"],
        ["tried before", packet.tried_before ? plural(packet.tried_before.length, "proposal") : "nothing yet"],
      ]),
      !answers.reached
        ? h("p", { class: "hint" },
            answers.of ? "No answer seen here reached the target: a proposer has only the target, and the task's own criteria, to show it what is wanted. " : "",
            "More answers here make better evidence: ",
            h("a", { href: `#/runs/${found.run}/${found.step}?start=resample` }, "resample this step"), ".")
        : null,
      h("details", { class: "fold" },
        h("summary", null, h("span", { class: "fold-preview" }, "The packet as JSON"),
          h("span", { class: "faint" }, `${Math.max(1, Math.round(evidence.size / 1024))} KB · ${evidence.sha}`)),
        h("pre", { class: "code evidence-json" }, pretty(packet))));
  }

  // --- proposals --------------------------------------------------------------------------------

  function drawProposals() {
    const items = shown.proposals;
    for (const pid of [...selected]) if (!items.some((each) => each.id === pid)) selected.delete(pid);
    const ticking = acting() && items.length;
    replace(proposalsEl, panel(`Proposals (${items.length})`,
      items.length
        ? h("div", { class: "proposal-list" }, items.map((item) => proposalCard(item, acting())))
        : h("p", { class: "muted" }, acting() ? "None yet: ask the proposer, or write one, below." : "None yet."),
      { meta: h("span", { class: "muted" }, ticking ? "tick the ones to screen" : "each a hypothesis until a screen tests it") }));
  }

  function proposalCard(item, ticking) {
    const ticked = selected.has(item.id);
    const box = ticking
      ? h("input", {
          type: "checkbox", checked: ticked, "aria-label": `Screen proposal ${short(item.id)}`, "data-control": true,
          onchange: (event) => {
            if (event.target.checked) selected.add(item.id);
            else selected.delete(item.id);
            event.target.closest(".proposal").classList.toggle("selected", event.target.checked);
            screenForm?.refresh();
          },
        })
      : null;
    const first = item.proposers[0] || {};
    // Screens come newest first: the chip says it survived if any screen found so, else the newest.
    const latest = item.screens.find((verdict) => verdict.complete && verdict.survives) || item.screens[0];
    return h("div", { class: ["proposal", ticked && "selected"], dataset: { proposal: item.id } },
      h("div", { class: "proposal-head" },
        box,
        h("span", { class: "mono" }, short(item.id)),
        chip(`${plural(item.changed, "line")} changed`, "muted"),
        item.proposers.map((entry) => whoChip(entry)),
        latest ? verdictChip(latest) : chip("hypothesis: not screened yet", "warn")),
      h("p", { class: "proposal-why" }, h("span", { class: "label" }, "Why  "), first.rationale || "—"),
      (item.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
      diffView(item.diff),
      item.screens.length
        ? h("ul", { class: "small proposal-screens" }, item.screens.map((verdict) =>
            h("li", null, verdictChip(verdict), " ", screenLine(verdict),
              " — ", h("a", { href: `#/experiments/${verdict.screen}` }, `screen ${short(verdict.screen)}`))))
        : null,
      h("p", { class: "faint small" }, "Saved as the variant ", h("span", { class: "mono" }, item.variant),
        ": an A/B or a batch can take it as it is."));
  }

  // --- the forms: proposing, and screening --------------------------------------------------------
  //
  // Made once and kept: a proposal made, or a screen ended, leaves what is typed in them.

  function drawForms() {
    if (!acting()) {
      replace(proposeEl);
      replace(screenEl);
      replace(validateEl);
      screenForm = null;
      validateForm = null;
      return;
    }
    if (!proposeEl.firstChild) replace(proposeEl, proposePanel());
    if (shown.proposals.length && !screenForm) {
      screenForm = screenPanel();
      replace(screenEl, screenForm.element);
    } else {
      screenForm?.refresh();
    }
    // Drawn again when the survivors change: a form left as typed otherwise.
    const names = survivors().map((each) => each.id).join(",");
    if (!names) {
      replace(validateEl);
      validateForm = null;
    } else if (!validateForm || validateForm.names !== names) {
      validateForm = { ...validatePanel(), names };
      replace(validateEl, validateForm.element);
    }
  }

  /** Proposals a finished screen found to shift the target: what may be validated. */
  function survivors() {
    return shown.proposals.filter((each) => each.screens.some((verdict) => verdict.complete && verdict.survives));
  }

  function proposePanel() {
    return panel("Propose an edit", h("div", { class: "stack" },
      h("details", { class: "fold propose-builtin", open: true },
        h("summary", null, h("strong", null, "Ask the built-in proposer")),
        builtinForm()),
      h("details", { class: "fold propose-own" },
        h("summary", null, h("strong", null, "Write one yourself")),
        ownForm())),
      { meta: h("span", { class: "muted" }, "small edits, each with its reason") });
  }

  function builtinForm() {
    const include = h("input", { type: "checkbox", name: "include_task", checked: true });
    const { element } = planForm({
      kind: "propose",
      fields: [
        h("div", { class: "form-grid" },
          field("Proposals", input("m", { type: "number", value: 3, min: 1, max: 6 }), { help: "Different ideas, each a few small edits." }),
          h("label", { class: "form-field" }, h("span", { class: "form-label" }, "The task's criteria"),
            h("span", { class: "inline-check" }, include, " show them to the proposer"),
            h("span", { class: "form-help" }, "Its expected output and check: often the only place the fix shows."))),
        h("p", { class: "form-help" }, "Which model it asks, and its key, are set on ", h("a", { href: "#/connections" }, "Connections"), "."),
      ],
      read: (form) => ({
        refinement: shown.refinement.id,
        m: numberOf(form, "m") ?? 3,
        include_task: form.querySelector('[name="include_task"]').checked,
      }),
      render: proposePreview,
      startLabel: "Ask for proposals",
      onStarted: opened,
    });
    return element;
  }

  function ownForm() {
    const text = textArea("proposal_text", shown.baseline, { rows: 16 });
    const why = input("proposal_why", { placeholder: "what in the evidence it answers" });
    const changes = h("span", { class: "muted small" });
    const note = h("div");
    const count = () => {
      const lines = changedLines(shown.baseline, text.value);
      changes.textContent = lines ? `${plural(lines, "line")} changed` : "unchanged: edit it to propose something";
    };
    text.addEventListener("input", count);
    count();
    const button = h("button", { class: "btn btn-primary", type: "submit", "data-control": true }, "Add the proposal");
    return h("form", {
      class: "plan-form own-proposal", novalidate: true,
      onsubmit: guard(async (event) => {
        event.preventDefault();
        replace(note);
        button.disabled = true;
        try {
          const made = await api.post(`/refinements/${enc(shown.refinement.id)}/proposals`, { text: text.value, rationale: why.value });
          text.value = shown.baseline;
          why.value = "";
          count();
          replace(note, h("p", { class: "hint" }, made.duplicate
            ? `The same text was proposed before, as ${short(made.id)}: you are added to its proposers.`
            : `Added as ${short(made.id)}: a hypothesis until a screen tests it.`));
          await reload();
        } catch (failed) {
          replace(note, formError(failed));
        } finally {
          button.disabled = false;
        }
      }),
    },
    h("p", { class: "hint" }, "The skill as the run was given it. Change what you think will help: small changes are what a screen can attribute — more than ten lines is allowed, with a warning."),
    h("div", { class: "editor-head" }, h("span", { class: "label" }, "Your wording"), changes),
    text,
    field("Why", why, { help: "Kept with the proposal, and shown to proposers after you.", wide: true }),
    note,
    h("div", { class: "form-actions" }, button));
  }

  function screenPanel() {
    // A refinement whose target is the task's check can only be screened to the end: the check
    // reads what the run wrote, which only a run's end shows.
    const checkTarget = shown.refinement.target?.kind === "check";
    const toEnd = (element) => element.querySelector('[name="reach"]:checked')?.value === "end";
    const checked = (element, name) => element.querySelector(`[name="${name}"]`)?.checked === true;
    const endOnly = h("div", { class: "form-grid", hidden: !checkTarget },
      field("Runs to fork from", input("checkpoints", { type: "number", value: 4, min: 1, max: 10 }),
        { help: "This refinement's run and others of its task at the same step, passing and failing by turns: an effect that holds across them is one fresh runs keep." }),
      h("label", { class: "check-choice" }, h("input", { type: "checkbox", name: "quick" }),
        " A quick look: 8 forks of each proposal and none of the original — rates, never a verdict"),
      h("label", { class: "check-choice" }, h("input", { type: "checkbox", name: "fresh_original" }),
        " Draw the original's forks anew, rather than count those this refinement drew before"));
    const stepOnly = h("div", { class: "form-grid", hidden: checkTarget },
      field("Max tokens", input("max_tokens", { type: "number", value: 4096, min: 1 }), { help: "Hitting it counts as truncated." }),
      seedField(),
      cacheField());
    const reachChanged = (event) => {
      const end = toEnd(event.target.form);
      endOnly.hidden = !end;
      stepOnly.hidden = end;
    };
    const form = planForm({
      kind: (element) => (toEnd(element) ? "screen-to-end" : "screen"),
      fields: [
        h("fieldset", { class: "target-picker" }, h("legend", null, "How each sample is judged"),
          h("label", { class: "target-choice" },
            h("input", { type: "radio", name: "reach", value: "step", checked: !checkTarget, disabled: checkTarget, onchange: reachChanged }),
            h("span", null, h("strong", null, "At this step"), h("span", { class: "muted small" }, checkTarget
              ? " — not for this refinement: its target is the task's check, which only a run's end can show"
              : " — each sample an answer here, judged by the target; each proposal against a fresh arm of the original"))),
          h("label", { class: "target-choice" },
            h("input", { type: "radio", name: "reach", value: "end", checked: checkTarget, onchange: reachChanged }),
            h("span", null, h("strong", null, "To the end"), h("span", { class: "muted small" },
              " — each sample a fork run from here to the end through the harness, the task's check deciding")))),
        h("div", { class: "form-grid" },
          field("Samples", input("n", { type: "number", value: 20, min: 2, max: 200 }), { help: "Per arm, the original's included." }),
          field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 64 }), { help: "The proxy's --max-inflight still caps the server." })),
        endOnly,
        h("details", { class: "fold more-settings" },
          h("summary", null, "More settings"),
          h("div", { class: "form-grid" },
            field("Significance (α)", input("alpha", { type: "number", value: 0.05, min: 0.0001, max: 0.5, step: "any" }), { help: "Family-wise, Holm-corrected." })),
          stepOnly),
      ],
      read: (element) => {
        const proposals = shown.proposals.filter((each) => selected.has(each.id)).map((each) => each.id);
        if (!proposals.length) {
          throw Object.assign(new Error("Tick the proposals to screen, above: the family the correction counts is fixed before sampling."), { field: "proposals" });
        }
        const params = {
          refinement: shown.refinement.id,
          proposals,
          n: numberOf(element, "n") ?? 20,
          concurrency: numberOf(element, "concurrency") ?? 1,
        };
        if (toEnd(element)) {
          params.checkpoints = numberOf(element, "checkpoints") ?? 4;
          if (checked(element, "quick")) params.quick = true;
          if (checked(element, "fresh_original")) params.fresh_original = true;
        } else {
          params.max_tokens = numberOf(element, "max_tokens") ?? 4096;
          Object.assign(params, samplingChoices(element));
        }
        const alpha = numberOf(element, "alpha", { integer: false });
        if (alpha !== null) params.alpha = alpha;
        return params;
      },
      render: (plan) => (plan.reach === "end" ? screenToEndPreview(plan) : screenPreview(plan)),
      startLabel: "Start the screen",
      onStarted: opened,
    });
    return {
      element: panel("Screen the ticked proposals", form.element,
        { meta: h("span", { class: "muted" }, "each against the original"), className: "screen-panel" }),
      refresh: form.refresh,
    };
  }

  function validatePanel() {
    const choices = survivors();
    const choose = h("select", { class: "field", name: "proposal" },
      choices.map((each) => h("option", { value: each.id }, `${short(each.id)} — ${plural(each.changed, "line")} changed`)));
    const form = planForm({
      kind: "validate",
      fields: [
        h("div", { class: "form-grid" },
          field("Proposal", choose, { help: "One that survived a screen." }),
          field("Forks", input("n", { type: "number", value: 5, min: 5, max: 200 }), { help: "An arm, from this step to the end." }),
          field("Full runs", input("runs", { type: "number", value: 5, min: 5, max: 200 }), { help: "An arm, of the task and of each held-out task." }),
          field("At once", input("concurrency", { type: "number", value: 1, min: 1, max: 16 }), { help: "On one GPU, one at a time is fastest." })),
        h("details", { class: "fold more-settings" },
          h("summary", null, "More settings"),
          h("div", { class: "form-grid" },
            field("Significance (α)", input("alpha", { type: "number", value: 0.05, min: 0.0001, max: 0.5, step: "any" }), { help: "For the not-worse checks." }))),
      ],
      read: (element) => {
        const params = {
          refinement: shown.refinement.id,
          proposal: element.querySelector('[name="proposal"]').value,
          n: numberOf(element, "n") ?? 5,
          runs: numberOf(element, "runs") ?? 5,
          concurrency: numberOf(element, "concurrency") ?? 1,
        };
        const alpha = numberOf(element, "alpha", { integer: false });
        if (alpha !== null) params.alpha = alpha;
        return params;
      },
      render: validatePreview,
      startLabel: "Start the validation",
      onStarted: opened,
    });
    return {
      element: panel("Validate a survivor with full runs", form.element,
        { meta: h("span", { class: "muted" }, "the task, and the tasks held out from it, each against the original"), className: "validate-panel" }),
    };
  }

  // --- validations, and approving one ---------------------------------------------------------

  function drawValidations() {
    const records = shown.validations || [];
    if (!records.length) {
      replace(validationsEl);
      return;
    }
    replace(validationsEl, panel(`Validations (${records.length})`,
      h("div", { class: "proposal-list" }, records.map(validationCard)),
      { meta: h("span", { class: "muted" }, "held-out verdicts are yours: no agent is shown them"), className: "validations" }));
  }

  function validationCard(record) {
    const gate = record.gate || { checks: [], notes: [] };
    const failed = gate.checks.filter((check) => !check.passed).map((check) => check.id);
    const counts = record.counts || {};
    const rows = [["forward", "from this step to the end", counts.forward], ["training", "full runs of the task", counts.training],
      ...(counts.held_out || []).map((entry) => [`held out ${entry.eval}`, "full runs, verdicts only", entry])];
    const approvable = gate.passed && acting() && !shown.approval;
    return h("div", { class: "proposal validation", dataset: { validation: record.id } },
      h("div", { class: "proposal-head" },
        h("span", { class: "mono" }, short(record.id)),
        chip(record.complete ? record.status : `${record.status}: unfinished`, record.complete ? "muted" : "warn"),
        h("span", { class: "muted" }, "proposal ", h("span", { class: "mono" }, short(record.proposal))),
        !record.complete ? chip("unfinished: no verdict", "warn")
          : record.approved ? chip("approved: the gate as it was met then", "ok")
          : gate.passed ? chip("meets the gate", "ok") : chip(`does not meet the gate: ${failed.join(", ")}`, "bad")),
      h("div", { class: "table-wrap" }, h("table", { class: "table validation-table" },
        h("thead", null, h("tr", null, h("th", null, "part"), h("th", null, "with the edit"), h("th", null, "the original"))),
        h("tbody", null, rows.filter(([, , arms]) => arms).map(([name, about, arms]) => h("tr", null,
          h("td", null, h("strong", null, name), h("div", { class: "faint small" }, about)),
          h("td", null, arms.edit?.rate ? rateBar(arms.edit.rate) : h("span", { class: "muted" }, "nothing counted"), armNotes(arms.edit)),
          h("td", null, arms.original?.rate ? rateBar(arms.original.rate) : h("span", { class: "muted" }, "nothing counted"), armNotes(arms.original))))))),
      // Like an unfinished screen: its counts so far, and no verdict.
      record.complete ? gateList(gate) : null,
      approvable
        ? h("div", { class: "form-actions" }, h("button", {
            class: "btn btn-primary", type: "button", "data-control": true,
            onclick: () => openApproval(record),
          }, "Approve…"))
        : null);
  }

  function openApproval(record, extra = {}) {
    const body = h("div", null, loading("Working out what approving writes…"));
    replace(approveEl, panel(`Approve proposal ${short(record.proposal)}`, body,
      { meta: h("span", { class: "muted" }, "nothing is written until you type the code"), className: "approve-panel" }));
    approveEl.scrollIntoView({ block: "nearest" });
    api.get(`/refinements/${enc(shown.refinement.id)}/approve${query({ proposal: record.proposal, validation: record.id, ...extra })}`)
      .then((plan) => approvalForm(record, plan, extra))
      .catch((error) => refusedApproval(record, error))
      .then((drawn) => { if (scope.active) replace(body, drawn); });
  }

  /** Refused to plan: why, and — when the spec moved or leads elsewhere — where to say it is. */
  function refusedApproval(record, error) {
    if (error.code !== "moved") return h("div", { class: "stack" }, formError(error), cancelButton());
    const spec = input("spec", { placeholder: "the task spec where it is now" });
    const target = input("target", { placeholder: "or the SKILL.md to write" });
    return h("form", {
      class: "stack", novalidate: true,
      onsubmit: (event) => {
        event.preventDefault();
        openApproval(record, spec.value.trim() ? { spec: spec.value.trim() } : { target: target.value.trim() });
      },
    },
    formError(error),
    h("div", { class: "form-grid" }, field("Spec", spec, { wide: true }), field("File to write", target, { wide: true })),
    h("div", { class: "form-actions" }, h("button", { class: "btn", type: "submit" }, "Look again"), cancelButton()));
  }

  function approvalForm(record, plan, extra) {
    const target = plan.target;
    const code = input("confirm", { placeholder: "the code above" });
    const note = h("div");
    const write = h("button", { class: "btn btn-primary", type: "submit", "data-control": true }, "Write it");
    return h("form", {
      class: "stack approve-form", novalidate: true,
      onsubmit: guard(async (event) => {
        event.preventDefault();
        replace(note);
        write.disabled = true;
        try {
          const done = await api.post(`/refinements/${enc(shown.refinement.id)}/approve`,
            { proposal: record.proposal, validation: record.id, confirm: code.value.trim(), ...extra });
          replace(approveEl);
          await reload();
          replace(approveEl, panel("Written", kv([
            ["file", h("span", { class: "mono" }, done.target)],
            ["the file before", h("span", { class: "mono" }, done.backup)],
            ["changelog", h("span", { class: "mono" }, done.changelog)],
          ]), { className: "approve-panel" }));
        } catch (failed) {
          write.disabled = false;
          replace(note, formError(failed));
        }
      }),
    },
    kv([
      ["writes", h("span", null, h("span", { class: "mono" }, target.path),
        target.windows ? h("div", { class: "mono faint" }, target.windows) : null)],
      ["keeps", `${plan.line_endings} line endings${plan.bom ? ", a byte-order mark" : ""}`],
      ["the file before", h("span", { class: "mono" }, `${target.history}/`)],
      ["changelog", h("span", { class: "mono" }, target.changelog)],
      ["on", h("span", null, "validation ", h("span", { class: "mono" }, short(plan.validation)),
        plan.validations.of_this_proposal > 1 ? h("span", { class: "muted" }, ` — ${plan.validations.meeting_the_gate} of its ${plan.validations.of_this_proposal} validations meet the gate; the approval keeps them all`) : null)],
    ]),
    h("div", { class: "label" }, "The edit"),
    diffView(plan.diff.hunks),
    h("div", { class: "label" }, "The gate: every check met"),
    gateList(plan.gate),
    h("div", { class: "approve-code" }, h("span", { class: "label" }, "Code"), h("strong", { class: "mono" }, plan.code)),
    field("Type the code to write it", code, { help: "Anything else writes nothing." }),
    note,
    h("div", { class: "form-actions" }, write, cancelButton()));
  }

  function cancelButton() {
    return h("button", { class: "btn btn-quiet", type: "button", onclick: () => replace(approveEl) }, "Cancel");
  }

  function approvalBanner() {
    const state = shown.approval;
    if (!state) return null;
    const where = h("span", { class: "mono" }, state.target);
    if (state.state === "applied") {
      return h("div", { class: "banner banner-ok" }, h("strong", null, `Approved: proposal ${short(state.proposal)} was written. `),
        where, h("span", { class: "muted" }, " — the file before, and the changelog, are beside its folder."));
    }
    const repair = state.state !== "diverged" && live.mode === "live"
      ? h("button", {
          class: "btn btn-small", type: "button", "data-control": true,
          onclick: guard(async (event) => {
            event.target.disabled = true;
            try {
              await api.post(`/refinements/${enc(shown.refinement.id)}/repair`);
              await reload();
            } catch (failed) {
              event.target.disabled = false;
              event.target.textContent = failed.message;
            }
          }),
        }, "Finish it")
      : null;
    return h("div", { class: "banner banner-warn" }, h("strong", null, "An approval was interrupted. "), state.message, " ", where, " ", repair);
  }

  function drawScreens() {
    const screens = shown.screens;
    if (!screens.length) {
      replace(screensEl);
      return;
    }
    replace(screensEl, panel(`Screens (${screens.length})`, h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, "screen"), h("th", null, "state"), h("th", null, "proposals"),
        h("th", null, "the original"), h("th", null, "verdict"), h("th", null, "when"))),
      h("tbody", null, screens.map((screen) => {
        const report = screen.report;
        return h("tr", { class: "clickable", onclick: () => (location.hash = `#/experiments/${screen.experiment}`) },
          h("td", null, h("a", { class: "mono", href: `#/experiments/${screen.experiment}` }, short(screen.experiment))),
          h("td", null, chip(screen.status, screen.status === "done" ? "ok" : screen.status === "running" ? "accent" : "warn")),
          h("td", { class: "mono" }, report.arms.map((arm) => arm.name).join(", ")),
          // A quick look draws no original arm: the original's forks drawn before, if any.
          h("td", null, report.baseline.target.total ? rateBar(report.baseline.target) : h("span", { class: "faint" }, "none drawn here")),
          h("td", null, report.quick ? chip("quick look: rates only, no verdict", "muted") : !report.complete ? chip("unfinished: no verdict", "warn") : [
            report.survivors.length ? chip(`${report.survivors.join(", ")} survived`, screenTones(report.reach)[0]) : chip("none survived", "muted"),
            // A wording that made it significantly worse is a finding too.
            report.arms.some((arm) => arm.worse)
              ? [" ", chip(`${report.arms.filter((arm) => arm.worse).map((arm) => arm.name).join(", ")} worse`, screenTones(report.reach)[1])]
              : null,
          ]),
          h("td", { class: "faint", title: when(screen.created_at) }, ago(screen.created_at)));
      }))))));
  }
}

/** A validation's gate, check by check: each with its reason, and what it cannot see. */
function gateList(gate) {
  return h("div", { class: "stack", style: { gap: "4px" } },
    h("ul", { class: "gate-list" }, gate.checks.map((check) => h("li", { class: check.passed ? "met" : "unmet" },
      chip(check.passed ? "met" : "not met", check.passed ? "ok" : "bad"), " ",
      h("strong", null, `${check.id} `), check.name,
      check.low_powered ? [" ", chip("low-powered", "warn", "At these run counts only a large difference shows: a pass here says little.")] : null,
      h("div", { class: "muted small" }, check.reason)))),
    (gate.notes || []).map((note) => h("div", { class: "banner banner-warn" }, note)));
}

/** What an arm did not count, under the rule that counts timeouts as failures. */
function armNotes(arm) {
  if (!arm) return null;
  const parts = [];
  if (arm.timeouts) parts.push(`${arm.timeouts} timed out (counted as failed)`);
  if (arm.excluded?.length) parts.push(`${arm.excluded.length} not counted: ${arm.excluded.join("; ")}`);
  return parts.length ? h("div", { class: "faint small" }, parts.join(" · ")) : null;
}

function validatePreview(plan) {
  const screen = plan.screen || {};
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["proposal", h("span", { class: "mono" }, short(plan.proposal))],
      ["screened", screen.target ? `survived screen ${short(screen.screen)}: ${screen.target.successes}/${screen.target.total} against the original's ${screen.baseline.successes}/${screen.baseline.total}` : null],
      ["forward", `${plural(plan.forward.forks_per_arm, "fork")} an arm, from step ${plan.checkpoint.step} to the end`],
      ["training", `${plural(plan.training.runs_per_arm, "full run")} an arm of ${plan.task}, the edit from the first step`],
      ["held out", plan.held_out.length
        ? `${plural(plan.training.runs_per_arm, "full run")} an arm of each of: ${plan.held_out.map((entry) => entry.eval).join(", ")} — only their verdicts are kept`
        : "none: the gate will refuse approval"],
      ["runs", `${plan.total} in all`],
    ]),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "Each arm is drawn fresh, the original's included. Held-out runs are recorded in a temporary folder outside the recordings folder and deleted after: their verdicts are shown on this page, and to no agent."));
}

/** A screen's numbers for one proposal: against the original's arm, or — a quick look, which
 *  forks the proposal alone — its rate, beside the original's forks drawn before, if any. */
function screenLine(verdict) {
  const rate = `${verdict.target.successes}/${verdict.target.total}`;
  const base = verdict.baseline || {};
  const toEnd = verdict.reach === "end" ? ", forks to the end" : "";
  if (verdict.quick) {
    return `${rate} forks passed${toEnd}: rates, never a verdict${base.total ? `; the original's ${base.successes}/${base.total} drawn before` : ""}`;
  }
  return `${rate} against the original's ${base.successes}/${base.total}${toEnd}`
    + (verdict.adjusted_p !== null && verdict.adjusted_p !== undefined ? `, Holm p ${formatP(verdict.adjusted_p)}` : "");
}

function verdictChip(verdict) {
  if (verdict.quick) return chip("quick look: rates only", "muted");
  if (!verdict.complete) return chip("screen unfinished", "warn");
  if (verdict.survives) return chip("survived a screen", screenTones(verdict.reach)[0]);
  if (verdict.worse) return chip("worse in a screen", screenTones(verdict.reach)[1]);
  return chip("not significant", "muted");
}

/** A proposal's line diff: the hunks `ops.diff.line_diff` makes, numbered both sides. */
export function diffView(hunks) {
  return h("div", { class: "diff" }, hunks.map((hunk) => {
    let old = hunk.old_start;
    let neu = hunk.new_start;
    return h("div", { class: "diff-hunk" }, hunk.lines.map((line) => {
      const kind = line.op === "-" ? "del" : line.op === "+" ? "add" : "ctx";
      const left = line.op === "+" ? "" : String(old++);
      const right = line.op === "-" ? "" : String(neu++);
      return h("div", { class: ["diff-line", kind] },
        h("span", { class: "ln" }, left), h("span", { class: "ln" }, right),
        h("span", { class: "diff-op" }, line.op === " " ? "" : line.op),
        h("span", { class: "diff-text" }, line.text || " "));
    }));
  }));
}

/** Lines added or removed, roughly: enough to say how much was edited. */
function changedLines(before, after) {
  const count = new Map();
  for (const line of before.split("\n")) count.set(line, (count.get(line) || 0) + 1);
  let added = 0;
  for (const line of after.split("\n")) {
    const left = count.get(line) || 0;
    if (left) count.set(line, left - 1);
    else added += 1;
  }
  const removed = [...count.values()].reduce((sum, left) => sum + left, 0);
  return Math.max(added, removed);
}

function proposePreview(plan) {
  return h("div", { class: "stack", style: { gap: "8px" } },
    plan.remote
      ? h("div", { class: "banner banner-warn" }, h("strong", null, "Not this machine. "), `${plan.sends}.`)
      : null,
    kv([
      ["asks", h("span", null, h("span", { class: "mono" }, plan.sends_to),
        plan.through_the_proxy ? h("span", { class: "muted" }, " — through this proxy: gated, never recorded") : null)],
      ["model", plan.model ? h("span", { class: "mono" }, plan.model) : h("span", { class: "muted" }, "the server's default")],
      ["for", plural(plan.proposals, "proposal")],
      ["sends", `${Math.round(plan.size / 1024) || 1} KB: the skill, and the evidence above (packet ${plan.packet})`],
    ]),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "One call. Each proposal it makes is checked — its edits must match the skill exactly, leave the frontmatter alone and change at most ten lines — and kept with the model, the packet and the prompt it came from."));
}

/** A screen to the end, as it would run: the runs its forks are drawn from, the original's
 *  forks — counted from this refinement's earlier screens where they can be — and whether it can
 *  give a verdict at all (a quick look never does). */
function screenToEndPreview(plan) {
  const original = plan.original || {};
  const names = plan.arms.map((arm, index) => [index ? ", " : "", h("span", { class: "mono" }, arm.name)]);
  const from = plan.checkpoints?.length
    ? plan.checkpoints.map((entry, index) => [index ? "; " : "",
        h("span", { class: "mono" }, `run ${short(entry.run)}, step ${entry.step}`),
        ` (${entry.verdict || "no verdict"}, ${plural(entry.n, "fork")} an arm)`])
    : h("span", { class: "mono" }, `run ${short(plan.checkpoint.run_id)}, step ${plan.checkpoint.step}`);
  let theOriginal;
  if (plan.quick) {
    theOriginal = original.reused
      ? `none drawn now: its ${original.reused} forks this refinement drew before stand beside the rates`
      : "none drawn: a quick look's rates stand alone";
  } else {
    theOriginal = original.reused
      ? `${original.reused} of its ${plan.n} forks drawn before, by this refinement's earlier screens; ${original.drawn} drawn now`
      : `${plural(original.drawn, "fork")} drawn now`;
  }
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["forks from", from],
      ["target", plan.target],
      ["arms", plan.quick ? h("span", null, "each proposal alone: ", names) : h("span", null, "the original, then ", names)],
      ["the original", theOriginal],
      ["forks", `${plan.forks} in all, each the harness run from its checkpoint to the end`],
      ["verdict", plan.quick
        ? "none: a quick look gives rates, a lead for the next edit — only a full screen can keep one"
        : `Fisher's exact test against the original, Holm over ${plural(plan.family, "proposal")}, α = ${plan.alpha}`],
    ]),
    (plan.notes || []).map((note) => h("p", { class: "faint small" }, note)),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "Each fork replays its run as recorded up to the step — the model's answers and what its tools did to the files — then runs live to the end, where the task's check decides."));
}

function screenPreview(plan) {
  const power = plan.power || {};
  const base = power.baseline;
  const minutes = plan.estimate?.worst_minutes;
  return h("div", { class: "stack", style: { gap: "8px" } },
    kv([
      ["checkpoint", h("span", { class: "mono" }, `run ${short(plan.checkpoint.run_id)}, step ${plan.checkpoint.step}`)],
      ["target", plan.target],
      ["arms", h("span", null, "the original, then ", plan.arms.map((arm, index) => [index ? ", " : "", h("span", { class: "mono" }, arm.name)]))],
      ["samples", `${plan.n} an arm, ${plan.total} in all`],
      ["survives with", power.needed
        ? `at least ${power.needed} of ${plan.n}, ${base ? `against the ${base.hits} of ${base.of} this step reached it in its latest resample` : "against an original that never reaches it (the most an arm could be helped)"}; Holm over ${plural(plan.family, "proposal")}, α = ${plan.alpha}`
        : null],
      ["takes", plan.estimate?.fresh
        ? `each sample processes its whole prompt${plan.estimate.prompt_tokens ? ` (${plan.estimate.prompt_tokens.toLocaleString()} tokens)` : ""}, then answers`
        : minutes === undefined || minutes === null ? null : minutes < 1 ? `at most about ${Math.max(1, Math.round(minutes * 60))} s` : `at most about ${minutes.toFixed(1)} min`],
    ]),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
    h("p", { class: "hint" }, "The original is sampled fresh, first, then each proposal with its wording swapped in; each is compared with the original by Fisher's exact test, Holm-corrected over the proposals ticked. A screen at one step is evidence, not proof: confirm a survivor with full runs."));
}
