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
import { homeView } from "./views/home.js";
import { jobsView, jobView } from "./views/jobs.js";
import { liveView } from "./views/live.js";
import { refinementsView, refinementView } from "./views/refine.js";
import { runView } from "./views/run.js";
import { runsView } from "./views/runs.js";
import { skillsView, skillView } from "./views/skills.js";
import { stepView } from "./views/step.js";
import { trashView } from "./views/trash.js";

const ROUTES = [
  [/^\/guide$/, homeView, "guide"],
  [/^\/live(?:\/([0-9a-f]+))?$/, liveView, "live"],
  [/^\/runs$/, runsView, "runs"],
  [/^\/runs\/([^/]+)$/, runView, "runs"],
  // A step's address opens the run's timeline at that step; its exact context has its own.
  [/^\/runs\/([^/]+)\/(\d+)$/, runView, "runs"],
  [/^\/runs\/([^/]+)\/(\d+)\/context$/, stepView, "runs"],
  [/^\/diff\/([^/]+)\/([^/]+)$/, diffView, "runs"],
  [/^\/divergence\/(.+?)(?:@([0-9a-f]{8}))?$/, divergenceView, "runs"],
  [/^\/experiments$/, experimentsView, "experiments"],
  [/^\/experiments\/([^/]+)$/, experimentView, "experiments"],
  [/^\/refine$/, refinementsView, "refine"],
  [/^\/refine\/([0-9a-f]+)$/, refinementView, "refine"],
  [/^\/skills$/, skillsView, "skills"],
  [/^\/skills\/([^/]+)$/, skillView, "skills"],
  [/^\/batch$/, batchView, "runs"],
  [/^\/chains$/, chainsView, "runs"],
  [/^\/chains\/([0-9a-f]+)$/, chainView, "runs"],
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

// --- top bar ----------------------------------------------------------------------------

function renderStatus() {
  const status = document.getElementById("status");
  document.querySelector('[data-nav="live"]').hidden = live.mode !== "live";
  document.querySelector('[data-nav="connections"]').hidden = live.mode !== "live";
  document.getElementById("paused-banner").hidden = !(live.mode === "live" && live.paused);

  if (live.mode === "browse") {
    replace(status, h("span", { class: "dot" }), h("span", null, "browsing recordings"));
    return;
  }
  if (live.mode === "down") {
    replace(status, h("span", { class: "dot dot-bad" }), h("span", null, "cannot reach tuningfork"));
    return;
  }
  replace(
    status,
    h("span", {
      class: ["dot", live.connected ? "dot-live" : "dot-wait"],
      title: live.connected ? "Connected to the proxy" : "Reconnecting to the proxy…",
    }),
    h("span", null, live.connected ? (live.paused ? "paused" : "live") : "reconnecting…"),
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
          toast("Token accepted: this page can act now", { tone: "ok", timeout: 3000 });
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
const ENDINGS = { done: ["finished", "ok"], failed: ["failed", "bad"], cancelled: ["was cancelled", "bad"], interrupted: ["was interrupted", "bad"] };

function noteJobEnd(job) {
  if (["queued", "running"].includes(job.status)) {
    watched.add(job.id);
    return;
  }
  if (!watched.delete(job.id) || location.hash.startsWith(`#/jobs/${job.id}`)) return;
  // A check tried in the scenario editor is quick, and its verdict is shown where it began.
  if (job.kind === "check-trial") return;
  const [verb, tone] = ENDINGS[job.status] || [job.status, "accent"];
  const label = { resample: "A resample", ab: "An A/B", propose: "The proposer", screen: "A screen" }[job.kind] || `A ${job.kind} job`;
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
pickUpToken();

onAuthChange((_, { blocked } = {}) => {
  if (blocked) {
    toast("Read-only: this page needs the access token to change anything", { tone: "bad", timeout: 4000 });
    flashTokenBanner();
  }
  renderTokenBanner();
});

subscribe((event) => {
  if (event.type === "hold" && !location.hash.startsWith("#/live")) {
    toast(`A call is held: ${event.hold.reason}`, { tone: "response", href: `#/live/${event.hold.id}`, timeout: 0 });
  }
  if (["mode", "connection", "snapshot", "paused"].includes(event.type)) renderStatus();
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
});
