// tuningfork web UI: routing, the top bar, and live notifications.
//
// Plain ES modules with no build step. Routes live in the URL hash, so every view can be
// linked to and reloaded.

import { api } from "./api.js";
import { auth, guard, keep, onAuthChange, pickUpToken, setAuth } from "./auth.js";
import { failure, h, replace, toast } from "./dom.js";
import { live, start, subscribe } from "./live.js";
import { leaving, place, unsaved } from "./nav.js";
import { batchView } from "./views/batch.js";
import { chainsView, chainView } from "./views/chains.js";
import { diffView, divergenceView } from "./views/compare.js";
import { connectionsView } from "./views/connections.js";
import { scenariosView, scenarioView } from "./views/scenarios.js";
import { experimentsView, experimentView } from "./views/experiments.js";
import { homeView, wordsView } from "./views/home.js";
import { importView, scenarioFromRunView } from "./views/import.js";
import { jobsView, jobView } from "./views/jobs.js";
import { liveView } from "./views/live.js";
import { refinementsView, refinementView } from "./views/refine.js";
import { runView } from "./views/run.js";
import { runsView } from "./views/runs.js";
import { addSkillView, skillsView, skillView } from "./views/skills.js";
import { stepView } from "./views/step.js";
import { trashView } from "./views/trash.js";

// Each page, and the sidebar item it is under.
const ROUTES = [
  [/^\/guide$/, homeView, "guide"],
  [/^\/words$/, wordsView, "words"],
  [/^\/live(?:\/([0-9a-f]+))?$/, liveView, "live"],
  [/^\/runs$/, runsView, "runs"],
  [/^\/runs\/([^/]+)$/, runView, "runs"],
  // A step's address opens the run's timeline at that step; its exact context has its own.
  [/^\/runs\/([^/]+)\/(\d+)$/, runView, "runs"],
  [/^\/runs\/([^/]+)\/(\d+)\/context$/, stepView, "runs"],
  [/^\/runs\/([0-9a-f]{32})\/scenario$/, scenarioFromRunView, "runs"],
  [/^\/diff\/([^/]+)\/([^/]+)$/, diffView, "runs"],
  [/^\/divergence\/(.+?)(?:@([0-9a-f]{8}))?$/, divergenceView, "runs"],
  [/^\/experiments$/, experimentsView, "experiments"],
  [/^\/experiments\/([^/]+)$/, experimentView, "experiments"],
  [/^\/refine$/, refinementsView, "refine"],
  [/^\/refine\/([0-9a-f]+)$/, refinementView, "refine"],
  [/^\/skills$/, skillsView, "skills"],
  [/^\/skills\/([^/]+)$/, skillView, "skills"],
  [/^\/add-skill$/, addSkillView, "skills"],
  [/^\/batch$/, batchView, "batch"],
  [/^\/import$/, importView, "import"],
  // The address an imported run's scenario had first.
  [/^\/import\/([0-9a-f]{32})\/scenario$/, scenarioFromRunView, "import"],
  [/^\/chains$/, chainsView, "chains"],
  [/^\/chains\/([0-9a-f]+)$/, chainView, "chains"],
  [/^\/connections$/, connectionsView, "connections"],
  [/^\/scenarios$/, scenariosView, "scenarios"],
  [/^\/scenarios\/([a-z0-9][a-z0-9_-]*)$/, scenarioView, "scenarios"],
  [/^\/jobs$/, jobsView, "jobs"],
  [/^\/jobs\/([0-9a-f]+)$/, jobView, "jobs"],
  [/^\/trash$/, trashView, "trash"],
];

let scope = null;

// A view registers listeners as it renders, often after awaiting a fetch. If the person has
// navigated away meanwhile, the scope is already disposed: anything registered then is
// cleaned up at once instead of outliving the view (a stale arrow-key handler, or a
// listener re-fetching every run on every recorded step).
function makeScope(query) {
  const cleanups = [];
  let disposed = false;
  const keep = (cleanup) => (disposed ? cleanup() : cleanups.push(cleanup));
  return {
    // What follows `?` in the hash: `#/runs/<id>/<k>?start=ab` opens a form on the step.
    query,
    get active() {
      return !disposed;
    },
    on(listener) {
      keep(subscribe(listener));
    },
    every(callback, ms) {
      const id = setInterval(callback, ms);
      keep(() => clearInterval(id));
    },
    onLeave(callback) {
      keep(callback);
    },
    dispose() {
      disposed = true;
      for (const cleanup of cleanups.splice(0)) cleanup();
    },
  };
}

async function route() {
  // Unsaved work on the page being left: ask, and stay if the answer is no.
  const question = place.hash !== location.hash ? unsaved() : null;
  if (question && !window.confirm(question)) {
    history.replaceState(history.state, "", place.hash);
    return;
  }
  leaving.guard = null;
  place.hash = location.hash;
  closeNav();
  const [path, queryText = ""] = (location.hash.replace(/^#/, "") || "/").split("?", 2);
  if (path === "/") {
    // The front page: what tuningfork is, what is set up, where you left off.
    location.replace("#/guide");
    return;
  }
  scope?.dispose();
  scope = makeScope(new URLSearchParams(queryText));
  // Each view renders into its own element. A slow load finishing after you navigated
  // away fills an element that is no longer on the page, rather than the new view.
  const view = h("div", { class: "view" });
  replace(document.getElementById("main"), view);
  window.scrollTo(0, 0);

  for (const [pattern, render, nav] of ROUTES) {
    const match = pattern.exec(path);
    if (!match) continue;
    markNav(nav);
    try {
      await render(view, match.slice(1).map((part) => (part === undefined ? undefined : decodeURIComponent(part))), scope);
    } catch (error) {
      replace(view, failure(error));
    }
    return;
  }
  markNav(null);
  replace(view, failure(new Error(`There is no page at ${path}.`), "Not found"));
}

function markNav(active) {
  for (const link of document.querySelectorAll("[data-nav]")) {
    link.classList.toggle("active", link.dataset.nav === active);
  }
}

// --- the sidebar ----------------------------------------------------------------------------

// Narrowest, the sections are a drawer under a bar: it closes once a page is chosen.
function initNav() {
  const toggle = document.getElementById("nav-toggle");
  toggle.addEventListener("click", () => {
    const open = !document.getElementById("sidebar").classList.contains("open");
    document.getElementById("sidebar").classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Hide the sections" : "Show the sections");
  });
}

function closeNav() {
  document.getElementById("sidebar").classList.remove("open");
  const toggle = document.getElementById("nav-toggle");
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", "Show the sections");
}

// What Connections says, for the status strip and the dot beside Connections: asked when the
// page starts, when the proxy is back after a drop, and every minute.
let health = null;

async function checkHealth() {
  if (live.mode !== "live") return;
  try {
    health = await api.get("/connections");
  } catch {
    health = null;
  }
  renderStatus();
}

/** The server and the harnesses, a short line each, linking to Connections; amber where one
 *  needs a look. Sets the dot beside Connections the same way. */
function healthLines() {
  const dot = document.getElementById("health-dot");
  if (!health) {
    dot.hidden = true;
    return null;
  }
  const server = health.server || {};
  const found = (health.harnesses || []).filter((each) => each.path).map((each) => each.name);
  const answering = !server.problem && Boolean(server.identity?.id);
  const trouble = Boolean(server.problem || server.reminder) || !found.length;
  dot.hidden = false;
  dot.className = ["dot", "nav-dot", trouble ? "dot-wait" : "dot-live"].join(" ");
  dot.title = trouble ? "Something here needs a look" : "The server answers, and a harness was found";
  return h("div", { class: "status-health" },
    // A server on the internet that asks for a key answered: say where it is, not that it is down.
    h("a", {
      href: "#/connections", class: server.problem || server.reminder ? "warn-text" : null,
      title: server.reminder?.headline || server.problem || server.url || "",
    }, server.reminder ? "server on the internet" : server.problem ? "server not answering" : answering ? "server answering" : "server not read yet"),
    h("a", { href: "#/connections", class: found.length ? null : "warn-text" },
      found.length ? `${found.join(", ")} found` : "no harness found"));
}

function renderStatus() {
  const status = document.getElementById("status");
  const proxied = live.mode === "live";
  // What needs a running proxy stays in the list where none runs — greyed, saying why — so the
  // sections never move between modes.
  for (const link of document.querySelectorAll(".side-nav [data-proxy]")) {
    link.dataset.title ??= link.title;
    link.classList.toggle("nav-off", !proxied);
    link.title = proxied ? link.dataset.title : `${link.dataset.title}. Needs a running proxy: tuningfork serve`;
  }
  document.getElementById("paused-banner").hidden = !(proxied && live.paused);

  if (live.mode === "browse") {
    document.getElementById("health-dot").hidden = true;
    replace(status, h("div", { class: "status-line", title: "Browsing recordings: no proxy behind this page" },
      h("span", { class: "dot" }), h("span", { class: "status-word" }, "browsing recordings")));
    return;
  }
  if (live.mode === "down") {
    document.getElementById("health-dot").hidden = true;
    replace(status, h("div", { class: "status-line", title: "Cannot reach tuningfork" },
      h("span", { class: "dot dot-bad" }), h("span", { class: "status-word" }, "cannot reach tuningfork")));
    return;
  }
  replace(
    status,
    h("div", { class: "status-line", title: live.connected ? "Connected to the proxy" : "Reconnecting to the proxy…" },
      h("span", { class: ["dot", live.connected ? "dot-live" : "dot-wait"] }),
      h("span", { class: "status-word" }, live.connected ? (live.paused ? "paused" : "live") : "reconnecting…")),
    healthLines(),
    h("button", {
      class: ["btn", "btn-small", live.paused && "btn-warn"],
      type: "button",
      title: live.paused ? "Let model calls through again" : "Refuse model calls until resumed",
      "data-control": true,
      onclick: guard(async () => {
        try {
          await api.post(live.paused ? "/resume" : "/pause");
        } catch (error) {
          toast(error.message, { tone: "bad" });
        }
      }),
    }, live.paused ? "Resume" : "Pause"),
  );
}

// --- the access token ---------------------------------------------------------------------

function renderTokenBanner() {
  const banner = document.getElementById("token-banner");
  const readOnly = live.mode === "live" && auth.required && !auth.valid;
  banner.hidden = !readOnly;
  if (!readOnly) {
    replace(banner);
    return;
  }
  if (banner.childElementCount) return; // already drawn: keep what is being typed
  const input = h("input", {
    class: "field mono",
    type: "password",
    autocomplete: "off",
    spellcheck: "false",
    placeholder: "access token",
    "aria-label": "Access token",
  });
  const error = h("span", { class: "form-error", hidden: true });
  replace(
    banner,
    h("span", null, h("strong", null, "Read-only. "),
      auth.token ? "This page's access token is not this proxy's — it changes each time the proxy starts. "
        : "This page has no access token, so it can show everything but change nothing. ",
      "Open the address `tuningfork serve` printed, or paste the token (`tuningfork token --raw` prints it)."),
    h("form", {
      onsubmit: async (event) => {
        event.preventDefault();
        const given = input.value.trim();
        if (!given) return;
        keep(given);
        try {
          const status = await api.get("/status");
          setAuth(status.auth);
          if (!status.auth?.valid) {
            error.textContent = "That is not this proxy's token.";
            error.hidden = false;
            return;
          }
          toast("Token accepted: this page can act now", { timeout: 3000 });
        } catch (failed) {
          error.textContent = failed.message;
          error.hidden = false;
        }
      },
    }, input, h("button", { class: "btn btn-small", type: "submit" }, "Use token"), error),
  );
}

function flashTokenBanner() {
  const banner = document.getElementById("token-banner");
  if (banner.hidden) return;
  banner.classList.remove("flash");
  void banner.offsetWidth; // restart the animation
  banner.classList.add("flash");
  banner.querySelector("input")?.focus();
}

function renderHeld() {
  const held = live.holds.length;
  const badge = document.getElementById("held-count");
  badge.hidden = held === 0;
  badge.textContent = String(held);
  document.title = held ? `(${held}) held · Tuning Fork` : "Tuning Fork";
}

function renderJobs() {
  const active = live.jobs.length;
  const badge = document.getElementById("jobs-count");
  badge.hidden = active === 0;
  badge.textContent = String(active);
  badge.title = `${active} job${active === 1 ? "" : "s"} queued or running`;
}

// A job that ends while you are elsewhere says so, once, with a way to it. Only jobs this
// page saw running: a page opened later is not told about what ended before.
const watched = new Set();
const ENDINGS = { done: ["finished", "accent"], failed: ["failed", "warn"], cancelled: ["was cancelled", "muted"], interrupted: ["was interrupted", "warn"] };

function noteJobEnd(job) {
  if (["queued", "running"].includes(job.status)) {
    watched.add(job.id);
    return;
  }
  if (!watched.delete(job.id) || location.hash.startsWith(`#/jobs/${job.id}`)) return;
  // A check tried in the scenario editor is quick, and its verdict is shown where it began.
  if (job.kind === "check-trial") return;
  const [verb, tone] = ENDINGS[job.status] || [job.status, "accent"];
  const label = { resample: "A resample", ab: "An A/B", propose: "The proposing model", screen: "A test at a step" }[job.kind] || `A ${job.kind} job`;
  toast(`${label} ${verb}`, { tone, href: `#/jobs/${job.id}`, action: "Show", timeout: job.status === "done" ? 8000 : 0 });
}

// --- theme --------------------------------------------------------------------------------

const THEMES = ["system", "light", "dark"];
const THEME_ICONS = { system: "◐", light: "☀", dark: "☾" };

function readTheme() {
  try {
    return localStorage.getItem("tuningfork.theme") || "system";
  } catch {
    return "system";
  }
}

function applyTheme(theme) {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  const button = document.getElementById("theme");
  button.textContent = THEME_ICONS[theme];
  button.title = `Theme: ${theme} (click to change)`;
  button.setAttribute("aria-label", button.title);
}

function initTheme() {
  let theme = readTheme();
  applyTheme(theme);
  document.getElementById("theme").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    try {
      localStorage.setItem("tuningfork.theme", theme);
    } catch {
      // private window: the choice lasts for this page only
    }
    applyTheme(theme);
  });
}

// --- start --------------------------------------------------------------------------------

initTheme();
initNav();
pickUpToken();

onAuthChange((_, { blocked } = {}) => {
  if (blocked) {
    toast("Read-only: this page needs the access token to change anything", { tone: "warn", timeout: 4000 });
    flashTokenBanner();
  }
  renderTokenBanner();
});

subscribe((event) => {
  if (event.type === "hold" && !location.hash.startsWith("#/live")) {
    toast(`A call is held: ${event.hold.reason}`, { tone: "warn", href: `#/live/${event.hold.id}`, timeout: 0 });
  }
  if (["mode", "connection", "snapshot", "paused"].includes(event.type)) renderStatus();
  // Back after a drop, the proxy may have restarted: what it answers to is asked again.
  if (event.type === "connection" && live.connected) checkHealth();
  if (["snapshot", "hold", "resolved"].includes(event.type)) renderHeld();
  if (event.type === "snapshot") for (const job of event.jobs || []) watched.add(job.id);
  if (event.type === "job") noteJobEnd(event.job);
  if (["snapshot", "job"].includes(event.type)) renderJobs();
});

window.addEventListener("hashchange", route);
start().then(() => {
  renderStatus();
  renderHeld();
  renderJobs();
  renderTokenBanner();
  route();
  checkHealth();
  setInterval(checkHealth, 60000);
});
