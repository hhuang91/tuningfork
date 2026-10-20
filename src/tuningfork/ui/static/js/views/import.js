// Import a conversation an agent had elsewhere as a run (`ops.imports`): pick its file, see what
// it holds — its steps, the skills it shows, what the file lacked — name a skill's folder where
// it has one, and import. Every skill it shows is kept; the one an A/B, a refinement or a
// scenario studies is named when it starts. The run then works at its steps as any recorded one.

import { answered, api } from "../api.js";
import { guard } from "../auth.js";
import { chip, command, crumbs, firstLine, h, kv, loading, longText, panel, plural, replace } from "../dom.js";
import { field, formError, input, planForm } from "../forms.js";
import { live } from "../live.js";
import { toolsOffered } from "../stepinfo.js";

const SOURCES = {
  preloaded: "preloaded in the system prompt",
  tool: "loaded by a call",
  frontmatter: "found by its frontmatter",
};
// The file's text is sent whole; past this the server refuses it anyway.
const MAX_BYTES = 64 * 1024 * 1024;

export async function importView(root, _params, scope) {
  // Come from the Scenarios page: the chat imported, its scenario is made next.
  const forScenario = scope?.query?.get("then") === "scenario";
  const head = [
    crumbs(forScenario ? [["Scenarios", "#/scenarios"], ["From a run", "#/scenarios?from=run"], ["Import a chat"]]
      : [["Runs", "#/runs"], ["Import a chat"]]),
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
  const start = h("button", { class: "btn btn-primary", type: "button", "data-control": true, disabled: true },
    forScenario ? "Import, then make its scenario" : "Import");

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
      location.hash = forScenario ? `#/runs/${done.run}/scenario` : `#/runs/${done.run}`;
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

// A scenario of a run — imported, launched or recorded as it happened: the person's message it
// starts from — the one the skill came in after, unless the skill never came in — sent in the
// chat's words or the person's own, its name, its skill's folder and its check, and the
// conversation from there on, each step with what a run does with it. Each of its runs continues
// the conversation; its inputs are the run's workspace there where tuningfork launched it, and
// what is made says which files to add where it did not.
export async function scenarioFromRunView(root, [runId]) {
  const head = (title) => [
    crumbs([["Runs", "#/runs"], [`run ${runId.slice(0, 8)}`, `#/runs/${runId}`], ["Make a scenario"]]),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, title),
      h("p", { class: "sub" }, "Runs of the skill through its harness, each continuing this run's conversation: sent "
        + "the starting prompt, with the conversation before it as its context."))),
  ];
  if (live.mode !== "live") {
    replace(root, ...head("Make a scenario"), h("div", { class: "banner banner-warn" },
      "Making a scenario needs a running proxy (tuningfork serve). From a terminal:"),
      command(`tuningfork scenario ${runId.slice(0, 8)} <name>`));
    return;
  }
  replace(root, ...head("Make a scenario"), loading());
  const ask = (skill, harness, cut) => {
    const query = new URLSearchParams();
    if (skill) query.set("skill", skill);
    if (harness) query.set("harness", harness);
    if (cut !== null && cut !== undefined) query.set("cut", String(cut));
    const text = query.toString();
    return api.get(`/runs/${runId}/scenario${text ? `?${text}` : ""}`);
  };
  let found;
  try {
    found = await ask(null, null, null);
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
  // The run kept or showed several skills: the scenario studies the one chosen, and starts
  // where that one came in.
  const studies = h("select", { class: "field", name: "skill" });
  const folder = input("skill_dir", { placeholder: "optional: the skill's folder on this machine" });
  // The person's messages a run can start from with the skill coming in live: a choice only where
  // there are several, the skill having never come in, or come with the harness's instructions.
  const from = h("select", { class: "field", name: "cut" });
  const fromField = field("From the person's message", from, { help: "Each run starts there: the skill comes in "
    + "live, through the harness, after any of these." });
  // The starting prompt: the message's words as the chat had them, or the person's own — a file
  // the chat named on another machine, say, where the scenario's inputs put it.
  const prompt = h("textarea", { class: "field", name: "prompt", rows: 4, spellcheck: "false" });
  let edited = false;
  // What each run is sent, in the table's prompt row: the chat's words, or the person's.
  const promptRun = h("span", { class: "start-step-run" });
  const drawPromptRun = () => replace(promptRun, edited
    ? [chip("yours, in place of the chat's", "accent", "What you wrote above is each run's prompt."),
       h("span", { class: "small start-prompt-yours", title: prompt.value }, firstLine(prompt.value, 80))]
    : chip("sent as the person's message", "muted", "Each run is sent these words, as the chat had them."));
  prompt.addEventListener("input", () => {
    edited = prompt.value !== (found.prompt ?? "");
    prompt.classList.toggle("field-changed", edited);
    drawPromptRun();
  });
  const fromNote = h("p", { class: "hint", style: { gridColumn: "1 / -1" } });
  // What it studies: under the choice of skill, or alone where there is none to make.
  const about = h("span");
  const several = (found.skills || []).length > 1;
  // What each run is given and does, beside the chat: across the form's width, so that the
  // fields after it keep their places.
  const steps = h("div", { class: "form-field wide" });
  // A launched run's task had a check: the scenario takes it unless told otherwise.
  const kind = h("select", { class: "field", name: "check_kind" });
  const check = input("check", { placeholder: "the command, or the text" });
  const hasCheck = () => found.check !== null && found.check !== undefined;
  const shown = (before) => {
    replace(studies,
      h("option", { value: "", selected: !found.skill, disabled: true }, "choose the skill it studies"),
      (found.skills || []).map((each) => h("option", { value: each, selected: each === found.skill }, each)));
    // Asked for first: until it is chosen, nothing below can say where each run starts.
    studies.classList.toggle("field-ask", !found.skill && (found.skills || []).length > 1);
    const offered = (found.turns || []).filter((turn) => turn.offered);
    replace(from, offered.map((turn) => h("option", { value: String(turn.message), selected: turn.message === found.cut },
      `message ${turn.message} — ${turn.text || "(no text)"}`)));
    fromField.hidden = offered.length < 2;
    // Another start brings its own message's words; the person's stay while it starts where
    // they wrote them for.
    if (!edited || found.cut !== before) {
      prompt.value = found.prompt ?? "";
      edited = false;
      prompt.classList.remove("field-changed");
    }
    const unknown = found.prompt === null || found.prompt === undefined;
    prompt.disabled = unknown;
    prompt.placeholder = !unknown ? ""
      : !found.skill && (found.skills || []).length
        ? "Choose the skill it studies first: where each run starts, and so its prompt, depends on it."
        : "The conversation has no message of the person's to start from.";
    drawPromptRun();
    replace(fromNote, startNote(found, offered));
    replace(about, found.skill
      ? found.came_in
        ? `It studies ${found.skill}, which comes into each run live, through the harness, so a variant replaces it `
          + "everywhere the model reads it."
        : `It studies ${found.skill}, which never came into the conversation: each run starts from the message chosen.`
      : (found.skills || []).length ? "Choose the skill it studies: a scenario studies one."
        : "No skill shows in the conversation: name its folder below.");
    replace(steps, unknown ? null
      : [h("span", { class: "form-label" }, "Each run beside the chat: what it is given, then what it does"), stepsView(found, promptRun)]);
    const chosen = kind.value;
    replace(kind,
      hasCheck()
        ? h("option", { value: "task" }, `the task's own: ${typeof found.check === "string" ? found.check : JSON.stringify(found.check)}`)
        : null,
      h("option", { value: "none" }, "none yet"),
      h("option", { value: "shell" }, "a shell command, run in each run's workspace"),
      h("option", { value: "reply_contains" }, "the final reply holds this text"));
    if ([...kind.options].some((option) => option.value === chosen)) kind.value = chosen;
  };
  shown(found.cut);
  // The plan on screen is asked again once a new choice of skill, harness or start is drawn.
  let refreshPlan = null;
  // The answer asked for last is the one shown: a slower one for an earlier choice is dropped.
  let asking = 0;
  const again = (keepStart) => guard(async () => {
    const mine = ++asking;
    replace(problem);
    const before = found.cut;
    try {
      const answer = await ask(studies.value || null, harness.value, keepStart ? Number(from.value) : null);
      if (mine !== asking) return;
      found = answer;
      shown(before);
      showText();
      refreshPlan?.();
    } catch (error) {
      if (mine === asking) replace(problem, formError(error));
    }
  });
  studies.addEventListener("change", again(false));
  harness.addEventListener("change", again(false));
  from.addEventListener("change", again(true));

  const done = h("div");
  // The text a check of its kind takes: none for the task's own check, or none yet (found in
  // review: the field stood empty under either).
  const textField = field("Its text", check, { wide: true });
  function showText() {
    textField.hidden = !["shell", "reply_contains"].includes(kind.value);
  }
  kind.addEventListener("change", showText);
  showText();
  // What the form says, as the body `POST /runs/{run}/scenario` takes.
  function body() {
    const text = check.value.trim();
    const said = { name: name.value.trim(), harness: harness.value };
    if (found.cut !== null && found.cut !== undefined) said.cut = found.cut;
    // Sent only where the person changed it: else the scenario keeps the chat's words as they were.
    if (edited) said.prompt = prompt.value;
    if (studies.value || found.skill) said.skill = studies.value || found.skill;
    if (folder.value.trim()) said.skill_dir = folder.value.trim();
    // Sent as nothing, the check is the task's own where the run's task had one.
    if (kind.value === "none" && hasCheck()) said.task_check = false;
    if (kind.value === "shell" && text) said.check = text;
    if (kind.value === "reply_contains" && text) said.check = { reply_contains: text };
    return said;
  }
  // What will be made, shown as the form is filled in, as `tuningfork scenario --dry-run` shows it
  // — and made only as shown (found in review: every other form that makes something showed its
  // plan first, and this one said what was wrong only after it was asked to make it).
  const plan = planForm({
    fields: [
      h("div", { class: "form-grid" },
        found.recorded_harness ? null
          : field("Harness", harness, { help: "The agent the conversation was had with: the run does not say, so this is "
            + "what its conversation looks like." }),
        // A row of its own, first: the rest of the form follows from it.
        several ? field("The skill it studies", studies, { wide: true, help: about }) : null,
        field("Its folder", folder, { wide: true, help: "Where the skill lives on this machine, with its scripts and "
          + "references: the scenario tunes it there. Left empty: the folder the run's skill came from, else a copy "
          + "kept in the scenario's own folder." }),
        field("Name", name, { help: "lowercase letters, digits, - and _" }),
        fromField,
        field("Starting prompt", prompt, { wide: true, help: "What each run is sent as the person's message: the chat's "
          + "words unless you change them — a file it names on another machine, say, to where the scenario's inputs "
          + "put it in each run's workspace." }),
        fromNote,
        steps,
        field("Check", kind, { help: "What passes. Without one, runs have no verdict until it has one." }),
        textField),
      h("p", { class: "hint" }, found.launched
        ? "Its inputs: the run's workspace where the scenario starts, from the snapshots tuningfork took as it ran."
        : "Its inputs: the run kept none. What is made says which files its conversation names, to add on the "
          + "scenario's page."),
    ],
    read: () => {
      const said = body();
      if (!said.name) throw Object.assign(new Error("Name the scenario: lowercase letters, digits, - and _."), { field: "scenario_name" });
      return said;
    },
    render: scenarioPlan,
    startLabel: "Make the scenario",
    send: async (params, extra) =>
      answered(await api.post(`/runs/${runId}/scenario`, extra.dry_run ? { ...params, dry_run: true, quiet: true } : params)),
    onStarted: (made) => replace(done, panel("Made", h("div", { class: "stack" },
      (made.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)),
      made.harness && made.harness !== "hermes"
        ? h("p", { class: "hint" }, `Its runs go through ${made.harness}: choose it when you run the scenario.`) : null,
      h("div", { class: "actions" },
        h("a", { class: "btn btn-primary", href: `#/scenarios/${made.scenario.name}` }, "Open the scenario"))))),
  });
  refreshPlan = plan.refresh;
  replace(root, ...head("Make a scenario"),
    panel("The scenario", h("div", { class: "stack" },
      several ? null : h("p", { class: "hint" }, about),
      problem,
      plan.element)),
    done);
}

/** What a scenario of a run would be, as `prepare_scenario` plans it: written nowhere yet. */
function scenarioPlan(plan) {
  const check = plan.check === null || plan.check === undefined
    ? h("span", { class: "muted" }, "none yet: its runs have no verdict until it has one")
    : typeof plan.check === "string" ? h("span", { class: "mono" }, plan.check)
      : plan.check.reply_contains !== undefined ? ["the final reply holds ", h("span", { class: "mono" }, JSON.stringify(plan.check.reply_contains))]
        : h("span", { class: "mono" }, JSON.stringify(plan.check));
  return h("div", { class: "stack", style: { gap: "6px" } },
    kv([
      ["scenario", [h("span", { class: "mono" }, plan.name), h("span", { class: "muted" }, ", a task of its own on the Scenarios page")]],
      ["its skill", [h("span", { class: "mono" }, plan.skill), h("span", { class: "muted" }, plan.own_skill
        ? " — a copy kept in the scenario's own folder" : " — tuned in its folder, "), plan.own_skill ? null : h("span", { class: "mono" }, plan.skill_dir)]],
      ["each run", `continues ${plural(plan.context, "message")} of the conversation`
        + (plan.in_turn ? `, then ${plural(plan.in_turn, "message")} of the prompt's own turn, given as done` : "")
        + (plan.next_skill ? `; it stops where its model loads ${plan.next_skill.skill}` : "")],
      ["its check", check],
      ["its inputs", plan.files.length ? h("span", { class: "mono" }, plan.files.join(", "))
        : h("span", { class: "muted" }, "none kept: add the files its conversation names on its page")],
    ]),
    (plan.warnings || []).map((warning) => h("div", { class: "banner banner-warn" }, warning)));
}

// Which of the person's messages the starting prompt is, and why their others are not offered.
function startNote(found, offered) {
  if (found.cut === null || found.cut === undefined) return null;
  const others = (found.turns || []).length - offered.length;
  return `From message ${found.cut} of the chat.`
    + (others && found.skill
      ? ` The person's ${others === 1 ? "other message is" : `${others} other messages are`} not offered: ${found.skill} `
        + `came in after message ${found.cut}, so a run starting from another would never vary it.`
      : "");
}

// What a scenario's runs do with each step of the conversation from its starting prompt on
// (`imports.STEP_PARTS`): what the step's row says, in which tone, and why.
const PARTS = {
  done: ["given as done", "muted", "In each run's context, as the chat had it: the model does not make this step again."],
  first: ["each run starts here", "accent", "Each run's first call stands for this one: its model answers it, through the harness."],
};
// Steps alike in a row, each shown up to this many; past it, the first two and the last.
const FOLD = 6;

// Each run beside the chat: first what it is given — the harness's instructions and tools, the
// conversation before the prompt, the prompt — then each step from there, as the chat went, with
// what a run does with it: given as done, where each run starts, where it stops. A stretch of
// steps alike is folded, and the steps no run gets to are one row.
function stepsView(found, promptRun) {
  const steps = found.steps || [];
  // Stretches of steps a run treats alike; where each starts, and a stop, are each their own.
  const groups = [];
  for (const step of steps) {
    const last = groups[groups.length - 1];
    const lone = (each) => each.part === "first" || each.stop;
    if (last && last[0].part === step.part && !lone(step) && !lone(last[last.length - 1])) last.push(step);
    else groups.push([step]);
  }
  const said = ["Each run's harness writes its own instructions and offers its own tools: the chat's are shown for "
    + "reference, and never sent."];
  if (!steps.length) said.push("The chat has no step after the prompt: each run starts with its model's first answer to it.");
  if (found.in_turn) {
    said.push(`The steps given as done are in each run's context, as the chat had them: ${found.skill || "the skill"} still comes in `
      + "live, loaded by the model itself.");
  }
  if (found.next_skill) {
    said.push(`The chat went on to load ${found.next_skill.skill}`
      + `${found.next_skill.step === null || found.next_skill.step === undefined ? "" : ` at step ${found.next_skill.step}`}: `
      + `each run stops where its model loads a skill other than ${found.skill || "the one studied"}, and the check judges the workspace as it is then.`);
  }
  if (steps.length) said.push("Where each run starts, its model makes its own steps: the ones listed after it are the chat's.");
  return [
    h("div", { class: "table-wrap" }, h("table", { class: "table start-steps" },
      h("thead", null, h("tr", null, h("th", null, ""), h("th", null, "in the chat"), h("th", null, "in each run"))),
      h("tbody", null, givenRows(found, promptRun), groups.flatMap((group) => groupRows(group, found))))),
    h("p", { class: "hint" }, said.join(" ")),
  ];
}

// What each run is given before its model's first answer, beside what the chat's calls held: the
// harness's instructions and tools — each run's harness brings its own — the conversation before
// the prompt, and the prompt.
function givenRows(found, promptRun) {
  const own = found.launched ? "its harness's own, as this run's" : "its harness's own, not the chat's";
  const why = "Each run's harness writes its own instructions and offers its own tools, as tuningfork launches it"
    + (found.launched ? " — as it did for this run" : "") + ". The chat's are shown for reference, and are never sent.";
  const ownCell = (text) => h("td", null, h("span", { class: "faint small", title: why }, text));
  const what = (text) => h("td", { class: "start-step-what" }, text);
  const before = found.before || [];
  return [
    h("tr", { class: "given part-system" },
      what("system prompt"),
      h("td", { class: "start-step-body" }, found.system ? longText(found.system, { limit: 0 })
        : h("span", { class: "muted" }, "none in the chat")),
      ownCell(found.preloaded && found.skill ? `${own}, ${found.skill} preloaded as in the chat` : own)),
    h("tr", { class: "given part-tools" },
      what("tools"),
      h("td", { class: "start-step-body" }, (found.tools || []).length ? toolsOffered(found.tools)
        : h("span", { class: "muted" }, "none recorded in the chat")),
      ownCell(own)),
    before.length
      ? h("tr", { class: "given part-before" },
          what("before it"),
          h("td", { class: "start-step-body" }, h("details", { class: "fold" },
            h("summary", null, `${plural(before.length, "message")} before the starting prompt`),
            h("ol", { class: "small start-before" }, before.map((message) =>
              h("li", null, h("strong", null, `${message.role}: `), message.said || "(nothing)"))))),
          h("td", null, h("span", { class: "start-step-run" }, chip("given as context", "muted",
            "Put after the harness's instructions in every call of each run, as the chat had it."))))
      : null,
    h("tr", { class: "given part-prompt" },
      what("starting prompt"),
      h("td", { class: "start-step-body" }, h("span", { class: "faint small" }, `message ${found.cut}: `),
        found.prompt.length <= 160 && !found.prompt.includes("\n") ? found.prompt : longText(found.prompt, { limit: 0 })),
      h("td", null, promptRun)),
  ];
}

function groupRows(group, found) {
  const part = group[0].part;
  if (part === "past" || part === "later") {
    return [folded(group, part === "past"
      ? `${plural(group.length, "step")} the chat went on to: no run gets here`
      : `${plural(group.length, "step")} after the person's next message, which no run is sent`)];
  }
  if (group.length <= FOLD) return group.map((step) => stepRow(step, found));
  return [...group.slice(0, 2).map((step) => stepRow(step, found)),
    folded(group.slice(2, -1), `${plural(group.length - 3, "more step")}, ${part === "done" ? "given as done" : "as the chat went on"}`),
    stepRow(group[group.length - 1], found)];
}

function stepRow(step, found) {
  const [say, tone, why] = PARTS[step.part] || [];
  return h("tr", { class: `step part-${step.part}` },
    h("td", { class: "start-step-what" }, step.step === null ? "" : `step ${step.step}`),
    h("td", { class: "mono start-step-label", title: step.label }, step.label),
    h("td", null, h("span", { class: "start-step-run" },
      say ? chip(say, tone, why) : h("span", { class: "faint small" }, "made by its model"),
      step.stop ? chip("stops after a load like this", "skill", `Each run stops where its model loads a skill other `
        + `than ${found.skill || "the one studied"}, as the chat did here, and the check judges the workspace as it is then.`) : null)));
}

function folded(group, text) {
  const first = group[0].step;
  const last = group[group.length - 1].step;
  const where = first === null || last === null ? "" : first === last ? `step ${first}` : `steps ${first}–${last}`;
  return h("tr", { class: `step folded part-${group[0].part}` },
    h("td", { class: "start-step-what" }, where), h("td", { colspan: "2" }, text));
}
