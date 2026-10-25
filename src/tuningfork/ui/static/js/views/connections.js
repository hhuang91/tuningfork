// Connections: the inference server the proxy forwards to, the harnesses it can launch, and
// the proxy itself.
//
// The server can be switched here — after a dry run shows what would change — but only
// while nothing is using the proxy, so no run is recorded half on one server and half on
// another. What a harness needs of the server is measured by a server check, not assumed.

import { api, enc } from "../api.js";
import { guard } from "../auth.js";
import { ago, capital, chip, command, h, kv, loading, panel, replace, short, when } from "../dom.js";
import { field, formError, input } from "../forms.js";
import { forgetTokenWork, live } from "../live.js";
import { LABELS } from "../words.js";

const SUPPORT_TONE = { yes: "muted", no: "warn", unknown: "faint" };
// A server's identity, field by field (`proxy.app.IDENTITY_FIELDS`), as the panel above names it.
const FIELDS = {
  build: "build", model_file: "model file", models: "models", n_ctx: "context window",
  chat_template_sha256: "chat template", slots: "slots", defaults: "sampling defaults",
};

/** That tuningfork is built for a model you run yourself (`ops.connections.reminder`): where the
 *  server is on the internet and answers as a hosted model API does. Never a refusal. */
export function cloudNote(reminder) {
  return h("div", { class: "banner banner-warn" }, h("strong", null, reminder.headline), " ", reminder.body, " ",
    h("a", { href: "#/words?term=local%20model" }, "Why a model you run yourself"));
}

/** A server that asks for a key and is not on the internet — one of your own given a key — said
 *  as the reminder says it of a hosted API: recorded, and what tuningfork runs itself refused. */
const keyNote = () => h("div", { class: "banner banner-warn" }, h("strong", null, "It asks for an API key. "),
  "Your agent's calls are recorded through it, with the key your agent sends. What tuningfork runs itself sends none, and is refused.");

/** The server in a word, as the strip here and the status strip at the sidebar's foot say it:
 *  on the internet, asking for a key, or whether it answers. `warn` where it needs a look. A
 *  server that asks for a key answered — the proxy's read of it notes a problem too, the refusal
 *  it got — so that is said first: never that it is down. */
export function serverWord(server = {}) {
  if (server.reminder) return { word: "on the internet", warn: true, title: server.reminder.headline };
  if (server.read?.asks_key) return { word: "asks for a key", warn: true, title: server.problem || undefined };
  if (server.problem) return { word: "not answering", warn: true, title: server.problem };
  if (server.identity?.id) return { word: "answering", warn: false };
  return { word: "not read yet", warn: false };
}

// The controls the server test can grey, by the token work each needs (`token_work`'s keys),
// named as each is labelled.
const TOKEN_CONTROLS = { score: "“Score this answer”", forking: "“Find forking tokens…”", logprobs: "“Keep each sample's tokens”" };

const listed = (items) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

/** Whether the server test found the server answering a prompt from its cache. */
const reusesPrompts = (server) => server.checks?.results?.fresh_prompts?.support === "no";

/** What Connections' strip means for the person, the most pressing first: at most three
 *  sentences, one for each part of it that is amber — or one saying nothing needs a look, only
 *  when none is. `state` is `GET /connections` with the proposer's settings as `proposer`. */
export function whatItMeans({ server = {}, harnesses = [], sandbox = null, proposer = {} }) {
  const lines = [];
  const asksKey = Boolean(server.read?.asks_key);
  // A key asked for refuses every launch and every call tuningfork makes itself: said once here,
  // and not again for each harness.
  if (asksKey) {
    lines.push("The server asks for an API key, so your agent's calls are recorded, but what tuningfork runs itself is refused: it sends no key.");
  } else if (server.reminder) {
    lines.push(server.reminder.headline);
  } else if (server.problem) {
    lines.push("The server is not answering, so nothing can be recorded or measured until it does.");
  }
  const found = harnesses.filter((harness) => harness.path);
  const missing = harnesses.filter((harness) => !harness.path).map((harness) => harness.name);
  const blocked = found.filter((harness) => (harness.preflight?.problems || []).length).map((harness) => harness.name);
  if (!found.length) {
    lines.push("No harness was found, so tuningfork can record your own agent but cannot start runs of its own.");
  } else if (!asksKey && blocked.length === found.length) {
    lines.push("No harness found here can run against this server, so tuningfork cannot start runs of its own.");
  } else if (!asksKey && (missing.length || blocked.length)) {
    lines.push([
      missing.length ? `${listed(missing)} ${missing.length === 1 ? "was" : "were"} not found, so runs of ${missing.length === 1 ? "it" : "them"} cannot start.` : null,
      blocked.length ? `${listed(blocked)} cannot run against this server.` : null,
    ].filter(Boolean).join(" "));
  }
  // Before what the test would tell: the sentence beside the strip's own "not tested yet". Where
  // scoring and forking tokens are greyed already — a server without llama.cpp's /props, read
  // untested — the sentence below says so, and this one does not ask it again.
  if (!server.checks?.checked_at && !server.problem && !asksKey) {
    lines.push(server.token_work?.score || server.token_work?.forking
      ? "Test the server to measure what each harness needs of it."
      : "Test the server to measure what each harness needs of it, and whether it can score answers and find forking tokens.");
  }
  // What a batch's or a fork's plan warns of (`ops.connections.cached_prompts_warning`). The
  // finding stays until the server is tested again: a restart keeps its identity.
  if (reusesPrompts(server)) {
    lines.push("The server reuses the prompts it processes, so each run's calls, and every fork's, depend on what it ran before: start llama-server with --no-cache-prompt, then test the server again.");
  }
  const greyed = Object.entries(server.token_work || {}).filter(([, reason]) => reason).map(([work]) => TOKEN_CONTROLS[work] || work);
  if (greyed.length) lines.push(`${listed(greyed)} ${greyed.length === 1 ? "is" : "are"} greyed out on this server.`);
  if (sandbox && !sandbox.available) {
    lines.push("Runs tuningfork starts are not sandboxed, so their folders' paths differ from run to run, and the model reads them.");
  }
  if (proposer.remote) {
    lines.push(!proposer.allow_remote
      ? "The proposing model is on another machine and not allowed yet, so asking it is refused."
      : proposer.key?.set
        ? "The proposing model is on another machine, so what it is sent — recorded prompts, answers and your skill — leaves this one."
        : "The proposing model is on another machine and no key is set, so it may refuse the call.");
  }
  return lines.length ? lines.slice(0, 3) : ["Nothing here needs a look."];
}

export async function connectionsView(root, _, scope) {
  if (live.mode !== "live") {
    replace(root, h("div", { class: "panel empty" }, h("h2", null, "No live proxy"),
      h("p", null, "Connections belong to a running proxy: start one with `tuningfork serve` and open its UI.")));
    return;
  }
  replace(root, loading());
  const healthEl = h("div");
  const serverEl = h("div");
  const proposerEl = h("div");
  const harnessesEl = h("div", { class: "stack" });
  const proxyEl = h("div");
  replace(root,
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Connections"),
      h("p", { class: "sub" }, "What the proxy talks to. Settings for the harnesses and the proposing model are kept in the recordings folder, never in a harness's own configuration."))),
    h("div", { class: "stack" }, healthEl, serverEl, proposerEl, panel("Harnesses", harnessesEl, {
      actions: h("button", { class: "btn btn-small", type: "button", "data-control": true, onclick: guard(detect) }, "Detect again"),
    }), proxyEl));

  let state = null;
  async function load() {
    const [connections, proposer] = await Promise.all([api.get("/connections"), api.get("/proposer")]);
    state = { ...connections, proposer };
    if (!scope.active) return;
    drawHealth();
    drawServer();
    drawProposer();
    drawHarnesses();
    drawProxy();
  }
  async function detect() {
    replace(harnessesEl, loading("Running each harness's --version…"));
    await api.post("/connections/harnesses/detect");
    await load();
  }

  // --- at a glance ----------------------------------------------------------------------
  //
  // The server, the harnesses, the sandbox and the proposing model, a word and a colour each —
  // grey what is fine, amber what needs a look — then what that means, in at most three
  // sentences. The server test is here too: much of the rest stands on what it measures. The
  // panels below hold the detail.

  function drawHealth() {
    const server = state.server;
    const identity = server.identity || {};
    const checks = server.checks || {};
    const said = serverWord(server);
    const settings = state.proposer;
    const sandbox = state.sandbox;
    replace(healthEl, h("section", { class: "panel health", "aria-label": "Connections at a glance" }, h("div", { class: "panel-body" },
      h("div", { class: "health-strip" },
        glance("Server", chip(said.word, said.warn ? "warn" : "muted", said.title), [
          (identity.models || []).length ? h("span", { class: "mono" }, identity.models.join(", ")) : null,
          identity.n_ctx ? `a ${Number(identity.n_ctx).toLocaleString()}-token context window` : null,
          reusesPrompts(server) ? h("span", { class: "warn-text" }, "reuses its prompts") : null,
          checks.checked_at
            ? h("span", { class: "faint", title: `${when(checks.checked_at)}, with ${checks.model}` }, `tested ${ago(checks.checked_at)}`)
            : h("span", { class: "warn-text" }, "not tested yet"),
          checkButton(identity.models || [], checks.model),
        ]),
        glance("Harnesses", null, state.harnesses.map((harness) => {
          const [word, tone, title] = harnessWord(harness);
          return h("span", null, h("strong", null, harness.name), " ", chip(word, tone, title));
        })),
        sandbox
          ? glance("Sandbox", sandbox.available ? chip("on", "muted") : chip("off", "warn"),
              [sandbox.available ? `bubblewrap ${sandbox.version || ""}`.trim() : sandbox.reason])
          : null,
        glance("Proposing model", proposerWord(settings), [
          settings.url ? h("span", { class: "mono" }, settings.url) : "asked through the proxy",
          settings.remote && !settings.allow_remote ? h("span", { class: "warn-text" }, "not allowed yet") : null,
          settings.remote && !settings.key?.set ? h("span", { class: "warn-text" }, "no key set") : null,
        ])),
      h("div", { class: "health-means" }, h("div", { class: "label" }, "What this means for you"),
        h("ul", null, whatItMeans(state).map((line) => h("li", null, line)))))));
  }

  /** One part of the strip: its name, its word, and a line or two beneath. */
  function glance(name, word, lines) {
    return h("div", { class: "glance" }, h("div", { class: "label" }, name), word,
      lines.filter(Boolean).map((line) => h("div", { class: "glance-line" }, line)));
  }

  /** A harness in a word: found, not found, or found and unable to run against this server. */
  function harnessWord(harness) {
    if (!harness.path) return ["not found", "warn", harness.error || "Not on PATH, and no program named for it"];
    const problems = harness.preflight?.problems || [];
    if (problems.length) return ["cannot run here", "warn", problems.join("; ")];
    return ["found", "muted", [harness.version, harness.path].filter(Boolean).join(", ")];
  }

  /** Where the proposing model is, in a word: what it is sent goes there. */
  function proposerWord(settings) {
    if (!settings.url) return chip("this server", "muted", "Asked through the proxy: it waits its turn with other calls, and is never recorded");
    if (!settings.remote) return chip("this machine", "muted");
    return chip("another machine", "warn", "What it is sent — recorded prompts, answers and your skill — leaves this machine");
  }

  // --- the server -----------------------------------------------------------------------

  let switcher = null;
  function drawServer() {
    const server = state.server;
    const identity = server.identity || {};
    const results = server.checks?.results || {};
    const measures = server.measures || {};
    const ruledOut = Object.values(server.token_work || {}).filter(Boolean);
    switcher ??= switchForm();
    replace(serverEl, panel("Inference server", [
      // A server that asks for a key answered: it is no server to start again.
      server.reminder ? cloudNote(server.reminder) : server.read?.asks_key ? keyNote() : server.problem
        ? h("div", { class: "banner banner-warn" }, h("strong", null, "Not answering: "), `${server.problem}. `,
            "Start the server — or let it finish loading its model — then test it again.")
        : null,
      kv([
        ["forwarding to", h("span", { class: "mono" }, server.url)],
        ["started with", server.started_with !== server.url ? h("span", { class: "mono" }, server.started_with) : null],
        ["build", identity.build ? h("span", { class: "mono" }, String(identity.build)) : null],
        ["model file", identity.model_file ? h("span", { class: "mono" }, identity.model_file) : null],
        ["models", (identity.models || []).length ? h("span", { class: "mono" }, identity.models.join(", ")) : null],
        ["context window", identity.n_ctx ? `${Number(identity.n_ctx).toLocaleString()} tokens` : null],
        ["slots", identity.slots ?? null],
        ["identity", identity.id ? h("span", { class: "mono" }, short(identity.id)) : "not read yet"],
      ]),
      h("div", { class: "label" }, "Measured"),
      // Each measurement in words, under the name the terminal prints, and what needs it.
      Object.keys(results).length
        ? h("table", { class: "table measured" }, h("tbody", null, Object.entries(results).map(([name, found]) =>
            h("tr", null,
              h("td", null, h("div", null, capital(measures[name]?.what || name.replaceAll("_", " "))),
                h("div", { class: "faint small mono" }, name)),
              h("td", null, chip(found.support, SUPPORT_TONE[found.support] || "muted")),
              h("td", { class: "small" }, h("div", { class: "muted" }, found.detail || ""),
                measures[name] ? h("div", { class: "faint" }, `Needed for ${measures[name].for}.`) : null)))))
        : h("p", { class: "muted small" }, `Nothing is measured on this server yet. “${LABELS.testServer}”, at the top of the page, measures it.`),
      // What the check rules out is greyed wherever it is offered, saying the same.
      ruledOut.length
        ? h("div", { class: "stack", style: { gap: "4px" } },
            h("div", { class: "label" }, "Greyed out on this server"),
            ruledOut.map((reason) => h("div", { class: "small warn-text" }, `${reason}.`)))
        : null,
      server.checks?.checked_at ? h("p", { class: "faint small" }, `Tested ${when(server.checks.checked_at)} with ${server.checks.model}.`) : null,
      h("details", { class: "fold" }, h("summary", null, "Point the proxy at another server"), switcher),
    ]));
  }

  function checkButton(models, last) {
    const note = h("span", { class: "muted small" });
    // A server that lists several models is checked with one of them, chosen here.
    const choice = models.length > 1
      ? h("select", { class: "field", name: "check_model", "aria-label": "Model to test with" },
          models.map((name) => h("option", { value: name, selected: name === last }, name)))
      : null;
    const button = h("button", {
      class: "btn btn-small", type: "button", "data-control": true,
      onclick: guard(async () => {
        button.disabled = true;
        note.textContent = "testing: a streaming tool call, a Messages call, then what token work needs…";
        try {
          const job = await api.post("/jobs", { kind: "server-check", params: choice ? { model: choice.value } : {} });
          await finished(job.id);
          forgetTokenWork();
          await load();
        } catch (error) {
          note.textContent = error.hint ? `${error.message} — ${error.hint}` : error.message;
          button.disabled = false;
        }
      }),
    }, LABELS.testServer);
    return h("span", { class: "controls" }, choice, button, note);
  }

  async function finished(jobId) {
    for (;;) {
      const job = await api.get(`/jobs/${enc(jobId)}`);
      if (!["queued", "running"].includes(job.status)) {
        if (job.status !== "done") {
          throw Object.assign(new Error(job.error?.message || `the test ${job.status}`), { hint: job.error?.hint || null });
        }
        return job;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }

  function switchForm() {
    const url = input("url", { placeholder: "http://127.0.0.1:8081" });
    const preview = h("div", { class: "plan-preview" });
    const status = h("div");
    const go = h("button", { class: "btn btn-warn", type: "submit", disabled: true, "data-control": true }, "Switch");
    let asked = 0;
    let shown = null;
    let timer = null;
    const form = h("form", {
      class: "plan-form", novalidate: true,
      oninput: () => {
        go.disabled = true;
        clearTimeout(timer);
        timer = setTimeout(check, 400);
      },
      onsubmit: guard(async (event) => {
        event.preventDefault();
        if (url.value.trim() !== shown) return;
        go.disabled = true;
        try {
          await api.post("/connections/upstream", { url: shown });
          forgetTokenWork();
          url.value = "";
          shown = null;
          replace(preview);
          replace(status, h("p", { class: "hint" }, "Switched. Steps recorded from now on name the new server."));
          await load();
        } catch (error) {
          replace(status, formError(error));
        }
      }),
    },
    h("p", { class: "hint" }, "Only while nothing uses the proxy: no call in flight or held, no job, and no harness tuningfork started still running."),
    h("label", { class: "form-field" }, h("span", { class: "form-label" }, "Server address"), url),
    status, preview, h("div", { class: "form-actions" }, go));

    async function check() {
      const mine = ++asked;
      const target = url.value.trim();
      shown = null;
      replace(preview);
      if (!target) {
        replace(status);
        return;
      }
      replace(status, h("span", { class: "muted small" }, "asking it…"));
      try {
        const plan = await api.post("/connections/upstream", { url: target, dry_run: true });
        if (mine !== asked) return;
        replace(status);
        replace(preview, switchPreview(plan));
        shown = plan.busy.length ? null : target;
        go.disabled = shown === null || url.value.trim() !== shown;
      } catch (error) {
        if (mine !== asked) return;
        replace(status, formError(error));
      }
    }
    return form;
  }

  function switchPreview(plan) {
    return h("div", { class: "stack", style: { gap: "8px" } },
      plan.differences.length
        ? h("table", { class: "table" },
            h("thead", null, h("tr", null, h("th", null, "what"), h("th", null, "now"), h("th", null, "after"))),
            h("tbody", null, plan.differences.map((change) => h("tr", null,
              h("td", { title: change.field }, FIELDS[change.field] || change.field),
              h("td", { class: "mono small" }, JSON.stringify(change.current ?? null)),
              h("td", { class: "mono small" }, JSON.stringify(change.proposed ?? null))))))
        : h("p", { class: "muted small" }, "It reports itself the same as the server now in use."),
      plan.reminder ? cloudNote(plan.reminder) : null,
      plan.busy.length
        ? h("div", { class: "banner banner-warn" }, h("strong", null, "Not now: "), plan.busy.join("; "), ".")
        : null);
  }

  // --- the built-in proposer: which model proposes edits, and its key ---------------------
  //
  // The key can be set or cleared here, never read back: the page is told only whether one is
  // set, and where it comes from.

  function drawProposer() {
    const settings = state.proposer;
    const key = settings.key || {};
    const error = h("div");
    const url = input("proposer_url", { value: settings.url || "", placeholder: "none: the proxy's own server, through the proxy" });
    const wire = h("select", { class: "field", name: "proposer_wire" },
      [["openai-chat", "OpenAI chat completions"], ["anthropic-messages", "Anthropic Messages"]].map(([value, label]) =>
        h("option", { value, selected: value === settings.wire }, label)));
    const model = input("proposer_model", { value: settings.model || "", placeholder: "the server's default" });
    const maxTokens = input("proposer_max_tokens", { type: "number", value: settings.max_tokens, min: 256, max: 131072 });
    const temperature = input("proposer_temperature", { type: "number", value: settings.temperature ?? "", min: 0, max: 2, step: "any" });
    const remote = h("input", { type: "checkbox", name: "proposer_allow_remote", checked: settings.allow_remote });
    const secret = h("input", { class: "field mono", type: "password", name: "proposer_key", autocomplete: "off", spellcheck: "false", placeholder: "an API key" });
    const save = async (path, body) => {
      replace(error);
      try {
        await api.put(path, JSON.stringify(body), "application/json");
        await load();
      } catch (failure) {
        replace(error, formError(failure));
      }
    };
    const number = (field) => (field.value.trim() === "" ? null : Number(field.value));
    replace(proposerEl, panel("Proposing model", [
      kv([
        ["asks", settings.url
          ? h("span", { class: "mono" }, settings.url)
          : h("span", null, "the server above, ", h("span", { class: "muted" }, "through the proxy: it waits its turn with other calls, and is never recorded"))],
        ["model", settings.model ? h("span", { class: "mono" }, settings.model) : h("span", { class: "muted" }, "the server's default")],
        ["API format", h("span", { class: "mono" }, settings.wire)],
        ["max tokens", String(settings.max_tokens)],
        ["temperature", settings.temperature === null ? "the server's default" : String(settings.temperature)],
        ["another machine", settings.allow_remote ? chip("allowed", "warn") : chip("not allowed", "muted")],
        ["key", key.set ? chip(key.from === "environment" ? "set, from TUNINGFORK_PROPOSER_KEY" : "set, in the recordings folder", "muted") : chip("not set", "muted")],
      ]),
      h("p", { class: "hint" }, "The model tuningfork asks for edits of a skill, from a refinement's page. What it is sent — the skill, recorded prompts and answers — goes to this address."),
      error,
      h("details", { class: "fold" }, h("summary", null, "Change the proposing model"),
        h("form", {
          class: "plan-form proposer-form", novalidate: true,
          onsubmit: guard(async (event) => {
            event.preventDefault();
            await save("/proposer", {
              url: url.value.trim() || null, wire: wire.value, model: model.value.trim() || null,
              max_tokens: number(maxTokens), temperature: number(temperature), allow_remote: remote.checked,
            });
          }),
        },
        h("div", { class: "form-grid" },
          field("Address", url, { help: "Empty: the proxy's own server. A cloud API's base, e.g. https://api.anthropic.com.", wide: true }),
          field("API format", wire),
          field("Model", model),
          field("Max tokens", maxTokens, { help: "A model that reasons first needs thousands." }),
          field("Temperature", temperature, { help: "Empty: the server's default." })),
        h("label", { class: "inline-check" }, remote,
          " Allow an address that is not this machine — the evidence it is sent holds recorded prompts, answers and your skill"),
        h("div", { class: "form-actions" }, h("button", { class: "btn btn-small", type: "submit", "data-control": true }, "Save")))),
      h("details", { class: "fold" }, h("summary", null, key.set ? "Replace or clear the key" : "Set a key"),
        h("form", {
          class: "plan-form", novalidate: true,
          onsubmit: guard(async (event) => {
            event.preventDefault();
            const given = secret.value.trim();
            secret.value = "";
            if (given) await save("/proposer/key", { key: given });
          }),
        },
        h("p", { class: "hint" }, "Kept in the recordings folder, readable only by this user, and sent only in the calls to the proposing model; never shown again. TUNINGFORK_PROPOSER_KEY in the proxy's environment wins over it."),
        secret,
        h("div", { class: "form-actions" },
          h("button", { class: "btn btn-small", type: "submit", "data-control": true }, "Save the key"),
          key.from === "file"
            ? h("button", { class: "btn btn-quiet btn-small", type: "button", "data-control": true, onclick: guard(() => save("/proposer/key", { key: null })) }, "Clear the key")
            : null))),
    ]));
  }

  // --- harnesses ------------------------------------------------------------------------

  function drawHarnesses() {
    replace(harnessesEl, [sandboxCard(state.sandbox), ...state.harnesses.map(harnessCard)]);
  }

  // A harness's need in the page's words, as the server test's measurement of it is named; the
  // name the terminal prints on hover.
  function needName(need) {
    if (need.what === "n_ctx") return `A context window of at least ${(need.minimum ?? 0).toLocaleString()} tokens`;
    const what = capital(state.server?.measures?.[need.what]?.what || need.what.replaceAll("_", " "));
    return need.minimum ? `${what}: at least ${need.minimum.toLocaleString()}` : what;
  }

  // Where every launch runs: what its harness sees is what the model reads.
  function sandboxCard(sandbox) {
    if (!sandbox) return null;
    return h("div", { class: "harness-card" },
      h("div", { class: "harness-head" },
        h("strong", null, "sandbox"),
        sandbox.available ? chip(`bubblewrap ${sandbox.version || ""}`.trim(), "muted") : chip("off", "warn")),
      sandbox.available
        ? kv([
            ["every run tuningfork starts sees", h("span", null, "its folder ", h("span", { class: "mono" }, sandbox.workspace), ", its home ", h("span", { class: "mono" }, sandbox.home))],
            ["hidden from it", "the recordings folder, the runs' folders and the task file's folder; the rest of the machine read-only"],
            ["a run's check", "runs there too, with the task file's folder read-only, a few of your environment variables like PATH, and those its task names"],
          ])
        : h("div", { class: "banner banner-warn" },
            `Runs tuningfork starts are not sandboxed: ${sandbox.reason}. Each sees its own folders, whose paths differ from run to run, and the model reads them. Their checks run on this machine itself.`));
  }

  function harnessCard(harness) {
    const program = input(`executable-${harness.name}`, { value: harness.source === "override" ? harness.executable : "", placeholder: "/absolute/path/to/the/program" });
    const error = h("div");
    const save = async (executable) => {
      try {
        await api.put(`/connections/harnesses/${enc(harness.name)}`, JSON.stringify({ executable }), "application/json");
        await load();
      } catch (failure) {
        replace(error, formError(failure));
      }
    };
    const pre = harness.preflight || { ok: true, problems: [], notes: [] };
    return h("div", { class: "harness-card" },
      h("div", { class: "harness-head" },
        h("strong", null, harness.name),
        harness.path ? chip(harness.version || "found", "muted") : chip("not found", "warn"),
        h("span", { class: "muted small" }, harness.source === "override" ? "named in the recordings folder" : harness.source === "PATH" ? "on PATH" : "")),
      kv([
        ["program", harness.path ? h("span", { class: "mono" }, harness.path) : h("span", { class: "muted" }, harness.executable)],
        ["problem", harness.error],
      ]),
      harness.requirements?.length
        ? h("div", null, h("div", { class: "label" }, "Needs of the server"),
            h("ul", { class: "small" }, harness.requirements.map((need) =>
              h("li", { title: need.what }, needName(need),
                h("span", { class: "faint" }, ` — ${need.why}`)))))
        : null,
      pre.problems.map((problem) => h("div", { class: "banner banner-warn" }, h("strong", null, "Cannot run here: "), problem)),
      pre.notes.map((note) => h("div", { class: "banner" }, note)),
      h("details", { class: "fold" }, h("summary", null, "Use another program"),
        h("div", { class: "stack", style: { gap: "8px", marginTop: "8px" } },
          program,
          h("div", { class: "form-actions" },
            h("button", { class: "btn btn-small", type: "button", "data-control": true, onclick: guard(() => save(program.value.trim() || null)) }, "Save"),
            harness.source === "override"
              ? h("button", { class: "btn btn-quiet btn-small", type: "button", "data-control": true, onclick: guard(() => save(null)) }, "Use the one on PATH")
              : null),
          error)));
  }

  // --- the proxy ------------------------------------------------------------------------

  function drawProxy() {
    const proxy = state.proxy;
    replace(proxyEl, panel("This proxy", [
      kv([
        ["recordings", h("span", { class: "mono" }, proxy.home)],
        ["listening on", h("span", { class: "mono" }, `${proxy.host}:${proxy.port}`)],
        ["calls at once", String(proxy.max_inflight)],
        ["output a call", proxy.max_output
          ? `at most ${proxy.max_output} tokens: a harness that asks for more is sent this, and what it asked is recorded`
          : "as each harness asks"],
        ["jobs at once", String(proxy.jobs)],
        ["never forwards to", proxy.protected.length ? h("span", { class: "mono" }, proxy.protected.join(", ")) : "nothing protected"],
      ]),
      state.busy?.length ? h("p", { class: "hint" }, `In use: ${state.busy.join("; ")}.`) : null,
      h("p", { class: "hint" }, "These change only with a restart:"),
      command(proxy.restart),
      proxy.restart_masked
        ? h("p", { class: "hint" }, "The upstream's user and password show as <redacted>. Type them back in before you run it.")
        : null,
    ]));
  }

  await load();
  // A switch made elsewhere, or a check that finished, shows here.
  scope.on((event) => {
    if (event.type === "connections") load();
  });
}
