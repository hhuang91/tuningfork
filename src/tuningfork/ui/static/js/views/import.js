// Import a conversation an agent had elsewhere as a run (`ops.imports`): pick its file, see what
// it holds — its steps, the skills it shows, what the file lacked — name a skill's folder where
// it has one, and import. Every skill it shows is kept; the one an A/B, a refinement or a
// scenario studies is named when it starts. The run then works at its steps as any recorded one.

import { api } from "../api.js";
import { guard } from "../auth.js";
import { command, crumbs, h, kv, loading, panel, plural, replace } from "../dom.js";
import { field, formError, input } from "../forms.js";
import { live } from "../live.js";

const SOURCES = {
  preloaded: "preloaded in the system prompt",
  tool: "loaded by a call",
  frontmatter: "found by its frontmatter",
};
// The file's text is sent whole; past this the server refuses it anyway.
const MAX_BYTES = 64 * 1024 * 1024;

export async function importView(root) {
  const head = [
    crumbs([["Runs", "#/runs"], ["Import a chat"]]),
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Import a chat"),
        h("p", { class: "sub" }, "A conversation an agent had elsewhere, as a run: each assistant turn a step, with "
          + "everything before it as its context — so resample, A/B and refine work at its steps."))),
  ];
  if (live.mode !== "live") {
    replace(root, ...head, h("div", { class: "banner banner-warn" },
      "Importing needs a running proxy (tuningfork serve). From a terminal:"), command("tuningfork import <file>"));
    return;
  }

  let content = null;
  let skill = null;
  let session = null;
  // A folder for each skill that has one, by the skill it was typed for: kept with the run in
  // place of the SKILL.md the model read. The server matches each folder to the skill it holds.
  const folders = new Map();
  const file = h("input", { class: "field", type: "file", accept: ".json,.jsonl,application/json" });
  const harness = h("select", { class: "field", name: "harness" },
    h("option", { value: "" }, "as the file says"),
    ["hermes", "claude-code", "generic"].map((name) => h("option", { value: name }, name)));
  const toolsFrom = input("tools_from", { placeholder: "optional: a run id, or its first characters" });
  const label = input("label", { placeholder: "optional" });
  const task = input("task_id", { placeholder: "optional: a task id to file the run under" });
  const preview = h("div", { class: "stack" });
  const problem = h("div");
  const start = h("button", { class: "btn btn-primary", type: "button", "data-control": true, disabled: true }, "Import");

  const options = () => ({
    content,
    session: session || undefined,
    skill: skill || undefined,
    harness: harness.value || undefined,
    tools_from: toolsFrom.value.trim() || undefined,
    skill_dir: [...folders.values()].map((value) => value.trim()).filter(Boolean),
    label: label.value.trim() || undefined,
    task_id: task.value.trim() || undefined,
  });

  // The preview asked for last is the one shown: a slower answer for an earlier choice is dropped.
  let asking = 0;
  async function look() {
    const mine = ++asking;
    replace(problem);
    start.disabled = true;
    if (!content) {
      replace(preview);
      return;
    }
    replace(preview, loading("Reading the conversation…"));
    try {
      const found = await api.post("/imports/preview", options());
      if (mine !== asking) return;
      if (found.choose_session) {
        replace(preview, sessionsView(found.choose_session));
        return;
      }
      replace(preview, planView(found));
      start.disabled = false;
    } catch (error) {
      if (mine !== asking) return;
      replace(preview);
      replace(problem, formError(error));
    }
  }

  function sessionsView(sessions) {
    return panel("Which session?", h("div", { class: "stack", style: { gap: "6px" } },
      h("p", { class: "hint" }, `The file holds ${plural(sessions.length, "session")}: a run is one conversation.`),
      sessions.map((found) => h("label", { class: "target-choice" },
        h("input", { type: "radio", name: "session", value: found.id,
          onchange: () => { session = found.id; skill = null; look(); } }),
        h("span", null, h("span", { class: "mono" }, found.id),
          h("span", { class: "muted small" }, ` — ${plural(found.messages, "message")}${found.title ? `: ${found.title}` : ""}`))))));
  }

  /** A folder field for one skill, keeping what was typed across previews. */
  function folderField(key) {
    const field = input(`skill_dir:${key}`, { placeholder: "optional: its folder on this machine" });
    field.value = folders.get(key) || "";
    field.addEventListener("change", () => { folders.set(key, field.value); look(); });
    return field;
  }

  function planView(plan) {
    const several = plan.skills.length > 1;
    const skills = plan.skills.length
      ? h("div", { class: "stack import-skills" },
          plan.skills.map((found) => {
            const where = found.step === null ? "not in the conversation" : `in context from step ${found.step}`;
            const how = found.source ? `${SOURCES[found.source] || found.source}; ` : "";
            return h("div", { class: "import-skill" },
              h("div", null, h("strong", null, found.name), h("span", { class: "muted small" }, ` — ${how}${where}; kept `
                + (found.folder ? `from ${found.folder}` : "as the model read it"))),
              folderField(found.name),
              several
                ? h("label", { class: "small" },
                    h("input", { type: "radio", name: "skill", value: found.name, checked: found.name === plan.skill,
                      onchange: () => { skill = found.name; look(); } }),
                    " studied by default — else an A/B, a refinement or a scenario names it")
                : null);
          }))
      : h("div", { class: "stack" }, h("span", { class: "muted" }, "none shows in the conversation"), folderField(""));
    return h("div", { class: "stack" },
      panel("What it holds", kv([
        ["format", plan.format],
        ["session", plan.session_id ? h("span", { class: "mono" }, plan.session_id) : null],
        ["title", plan.title],
        ["model", plan.model],
        ["harness", plan.harness],
        ["steps", `${plan.steps}, one an assistant turn, of ${plural(plan.messages, "message")}`],
        ["system prompt", plan.system_prompt ? "kept" : "none in the file"],
        ["tools", plan.tools ? `${plan.tools}${plan.tools_from ? ` from run ${plan.tools_from.slice(0, 8)}` : ""}` : "none in the file"],
        ["skills", skills],
      ])),
      (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)));
  }

  file.addEventListener("change", async () => {
    const chosen = file.files?.[0];
    content = null;
    session = null;
    skill = null;
    folders.clear();
    if (chosen && chosen.size > MAX_BYTES) {
      replace(preview);
      replace(problem, formError(new Error(`${chosen.name} is larger than 64 MiB`)));
      return;
    }
    content = chosen ? await chosen.text() : null;
    look();
  });
  harness.addEventListener("change", () => { skill = null; look(); });
  toolsFrom.addEventListener("change", look);

  start.addEventListener("click", guard(async () => {
    start.disabled = true;
    replace(problem);
    try {
      const done = await api.post("/imports", options());
      location.hash = `#/runs/${done.run}`;
    } catch (error) {
      replace(problem, formError(error));
      start.disabled = false;
    }
  }));

  replace(root, ...head,
    panel("The conversation", h("div", { class: "form-grid" },
      field("File", file, { wide: true, help: "Hermes's session file (~/.hermes/sessions/session_<id>.json), its export "
        + "(hermes sessions export), or an OpenAI-style list of messages." }),
      field("Harness", harness, { help: "The agent the conversation was had with, where the file does not say." }),
      field("Tools from", toolsFrom, { help: "Where the file names no tools (Hermes's export keeps none): a run recorded "
        + "through the same harness, whose tools each step then offers." }))),
    preview,
    panel("The run", h("div", { class: "stack" },
      h("p", { class: "hint" }, "Every skill it shows is kept with the run: from its folder where you name one above — its "
        + "scripts and references too — else its SKILL.md as the model read it."),
      h("div", { class: "form-grid" },
        field("Label", label),
        field("Task", task)),
      problem,
      h("div", { class: "actions" }, start))));
}

// A scenario of a run — imported, launched or recorded as it happened: the user turn it starts
// from — by default the one before the skill came in — its name, its skill's folder and its check.
// Each of its runs continues the conversation; its inputs are the run's workspace there where
// tuningfork launched it, and what is made says which files to add where it did not.
export async function scenarioFromRunView(root, [runId]) {
  const head = (title) => [
    crumbs([["Runs", "#/runs"], [`run ${runId.slice(0, 8)}`, `#/runs/${runId}`], ["Make a scenario"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, title),
      h("p", { class: "sub" }, "Runs of the skill through its harness, each continuing this run's conversation from one "
        + "of its user turns: the turn is the prompt, everything before it each run's context."))),
  ];
  if (live.mode !== "live") {
    replace(root, ...head("Make a scenario"), h("div", { class: "banner banner-warn" },
      "Making a scenario needs a running proxy (tuningfork serve). From a terminal:"),
      command(`tuningfork scenario ${runId.slice(0, 8)} <name>`));
    return;
  }
  replace(root, ...head("Make a scenario"), loading());
  const ask = (skill, harness) => {
    const query = new URLSearchParams();
    if (skill) query.set("skill", skill);
    if (harness) query.set("harness", harness);
    const text = query.toString();
    return api.get(`/runs/${runId}/scenario${text ? `?${text}` : ""}`);
  };
  let found;
  try {
    found = await ask(null, null);
  } catch (error) {
    replace(root, ...head("Make a scenario"), formError(error));
    return;
  }
  const problem = h("div");
  const name = input("scenario_name", { placeholder: "a-name" });
  // A run recorded as it happened may not say which harness it came through: the page shows the
  // one its conversation looks like, and the person can say otherwise.
  const harness = h("select", { class: "field", name: "harness" },
    ["hermes", "claude-code", "generic"].map((each) => h("option", { value: each, selected: each === found.harness }, each)));
  // The run kept or showed several skills: the scenario studies the one chosen, and starts,
  // unless told otherwise, before that one came in.
  const studies = h("select", { class: "field", name: "skill" });
  const folder = input("skill_dir", { placeholder: "optional: the skill's folder on this machine" });
  const cut = h("select", { class: "field", name: "cut" });
  const about = h("p", { class: "hint" });
  // What a start there takes as done, and where its runs stop: said for the turn before the skill,
  // across the form's width so that the fields after it keep their places.
  const where = h("div", { class: "stack", style: { gap: "4px", gridColumn: "1 / -1" } });
  const told = () => replace(where,
    found.in_turn && Number(cut.value) === found.default
      ? h("p", { class: "hint" }, `Each run starts after the model's first ${plural(found.in_turn, "message")} of that turn, `
        + `taken as done — an earlier skill among them — so ${found.skill} comes in live.`) : null,
    found.next_skill
      ? h("p", { class: "hint" }, `The chat loads ${found.next_skill.skill} after it, at step ${found.next_skill.step}: each `
        + "run stops where the model loads another skill, and the check judges the workspace as it is then.") : null);
  // A launched run's task had a check: the scenario takes it unless told otherwise.
  const kind = h("select", { class: "field", name: "check_kind" });
  const check = input("check", { placeholder: "the command, or the text" });
  const hasCheck = () => found.check !== null && found.check !== undefined;
  const shown = () => {
    replace(studies,
      h("option", { value: "", selected: !found.skill, disabled: true }, "choose the skill it studies"),
      (found.skills || []).map((each) => h("option", { value: each, selected: each === found.skill }, each)));
    replace(cut, found.turns.map((turn) => h("option", { value: String(turn.message), selected: turn.message === found.default },
      `message ${turn.message}${turn.message === found.default && found.came_in ? " (before the skill)" : ""} — ${turn.text || "(no text)"}`)));
    replace(about, found.skill
      ? found.came_in
        ? `It studies ${found.skill}. Starting before the skill came in, the skill enters each run live, through the `
          + "harness, so a variant replaces it everywhere the model reads it."
        : `It studies ${found.skill}, which never came into the conversation: each run starts at its first turn.`
      : (found.skills || []).length ? "Choose the skill it studies: a scenario studies one."
        : "No skill shows in the conversation: name its folder below.");
    const chosen = kind.value;
    replace(kind,
      hasCheck()
        ? h("option", { value: "task" }, `the task's own: ${typeof found.check === "string" ? found.check : JSON.stringify(found.check)}`)
        : null,
      h("option", { value: "none" }, "none yet"),
      h("option", { value: "shell" }, "a shell command, run in each run's workspace"),
      h("option", { value: "reply_contains" }, "the final reply holds this text"));
    if ([...kind.options].some((option) => option.value === chosen)) kind.value = chosen;
    told();
  };
  cut.addEventListener("change", told);
  shown();
  const again = guard(async () => {
    replace(problem);
    try {
      found = await ask(studies.value || null, harness.value);
      shown();
    } catch (error) {
      replace(problem, formError(error));
    }
  });
  studies.addEventListener("change", again);
  harness.addEventListener("change", again);

  const done = h("div");
  const make = h("button", { class: "btn btn-primary", type: "button", "data-control": true }, "Make the scenario");
  make.addEventListener("click", guard(async () => {
    replace(problem);
    const text = check.value.trim();
    const body = { name: name.value.trim(), cut: Number(cut.value), harness: harness.value };
    if (studies.value || found.skill) body.skill = studies.value || found.skill;
    if (folder.value.trim()) body.skill_dir = folder.value.trim();
    // Sent as nothing, the check is the task's own where the run's task had one.
    if (kind.value === "none" && hasCheck()) body.task_check = false;
    if (kind.value === "shell" && text) body.check = text;
    if (kind.value === "reply_contains" && text) body.check = { reply_contains: text };
    make.disabled = true;
    try {
      const made = await api.post(`/runs/${runId}/scenario`, body);
      replace(done, panel("Made", h("div", { class: "stack" },
        (made.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
        made.harness && made.harness !== "hermes"
          ? h("p", { class: "hint" }, `Its runs go through ${made.harness}: choose it when you run the scenario.`) : null,
        h("div", { class: "actions" },
          h("a", { class: "btn btn-primary", href: `#/scenarios/${made.scenario.name}` }, "Open the scenario")))));
    } catch (error) {
      replace(problem, formError(error));
      make.disabled = false;
    }
  }));
  replace(root, ...head("Make a scenario"),
    panel("The scenario", h("div", { class: "stack" },
      about,
      h("div", { class: "form-grid" },
        found.recorded_harness ? null
          : field("Harness", harness, { help: "The agent the conversation was had with: the run does not say, so this is "
            + "what its conversation looks like." }),
        (found.skills || []).length > 1 ? field("Skill", studies, { help: "The one each run varies: it comes in live." }) : null,
        field("Its folder", folder, { wide: true, help: "Where the skill lives on this machine, with its scripts and "
          + "references: the scenario tunes it there. Left empty: the folder the run's skill came from, else a copy "
          + "kept in the scenario's own folder." }),
        field("Name", name, { help: "lowercase letters, digits, - and _" }),
        field("Start from", cut, { wide: true, help: "The user turn each run's prompt is; the conversation before it is "
          + "each run's context." }),
        where,
        field("Check", kind, { help: "What passes. Without one, runs have no verdict until it has one." }),
        field("Its text", check, { wide: true })),
      h("p", { class: "hint" }, found.launched
        ? "Its inputs: the run's workspace where the scenario starts, from the snapshots tuningfork took as it ran."
        : "Its inputs: the run kept none. What is made says which files its conversation names, to add on the "
          + "scenario's page."),
      problem,
      h("div", { class: "actions" }, make))),
    done);
}
