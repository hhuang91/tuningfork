// Live state from a running proxy, kept current by its event stream.
//
// `tuningfork ui` serves the same page with no proxy behind it; then the mode is "browse"
// and nothing here connects.

import { api, EVENTS_URL } from "./api.js";
import { setAuth } from "./auth.js";

export const live = {
  mode: "unknown", // live | browse | down
  connected: false,
  paused: false,
  breakpoints: [],
  holds: [], // pending only, oldest first
  jobs: [], // queued and running
};

const listeners = new Set();

// The token work the proxy's server was measured unable to do — each kind's reason, or none
// (`token_work` in `GET /connections`): asked once, and again after a server check, a switch,
// or the proxy coming back.
let tokenWorkAsked = null;

export function tokenWork() {
  if (live.mode !== "live") return Promise.resolve({});
  tokenWorkAsked ??= api.get("/connections").then((found) => found.server?.token_work || {}, () => {
    tokenWorkAsked = null;
    return {};
  });
  return tokenWorkAsked;
}

export function forgetTokenWork() {
  tokenWorkAsked = null;
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function emit(event) {
  for (const listener of [...listeners]) {
    try {
      listener(event);
    } catch (error) {
      console.error(error);
    }
  }
}

export async function start() {
  try {
    const status = await api.get("/status");
    live.mode = status.live ? "live" : "browse";
    setAuth(status.auth);
    apply({ type: "snapshot", ...status });
  } catch {
    live.mode = "down";
  }
  emit({ type: "mode" });
  if (live.mode === "live") connect();
}

function connect() {
  // EventSource reconnects by itself; each reconnection starts with a fresh snapshot.
  const source = new EventSource(EVENTS_URL);
  let dropped = false;
  source.onopen = () => {
    live.connected = true;
    emit({ type: "connection" });
    // Back after a drop: the proxy may have restarted, with a new token. Ask again, so a
    // stale token shows as read-only now rather than at the first refused click.
    if (dropped) {
      api.get("/status").then((status) => setAuth(status.auth), () => {});
      forgetTokenWork();
    }
    dropped = false;
  };
  source.onerror = () => {
    dropped = true;
    live.connected = false;
    emit({ type: "connection" });
  };
  source.onmessage = (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    apply(event);
    emit(event);
  };
}

const pendingOnly = (holds) =>
  holds.filter((hold) => hold.pending !== false).sort((a, b) => a.created_at - b.created_at);

function apply(event) {
  switch (event.type) {
    case "snapshot":
      live.paused = Boolean(event.paused);
      live.breakpoints = event.breakpoints || [];
      live.holds = pendingOnly(event.holds || []);
      live.jobs = event.jobs || [];
      break;
    case "job": {
      const others = live.jobs.filter((job) => job.id !== event.job.id);
      const going = ["queued", "running"].includes(event.job.status);
      live.jobs = going ? [...others, event.job] : others;
      if (event.job.kind === "server-check" && !going) forgetTokenWork();
      break;
    }
    case "breakpoints":
      live.breakpoints = event.breakpoints || [];
      break;
    case "hold":
      live.holds = pendingOnly([...live.holds.filter((hold) => hold.id !== event.hold.id), event.hold]);
      break;
    case "resolved":
      live.holds = live.holds.filter((hold) => hold.id !== event.hold.id);
      break;
    case "paused":
      live.paused = Boolean(event.paused);
      break;
  }
}
