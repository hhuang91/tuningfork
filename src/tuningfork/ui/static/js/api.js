// The HTTP API the CLI also uses. The UI holds no capability the command line lacks.
//
// Every request goes through here, and only here: this is the one place the access token
// is attached, so no other module ever handles it.

import { auth, refused } from "./auth.js";

const BASE = "/_tuningfork/api";
const TOKEN_HEADER = "x-tuningfork-token";

export class ApiError extends Error {
  constructor(message, status, error = {}) {
    super(message);
    this.status = status;
    this.code = error.code || null;
    this.hint = error.hint || null;
    this.detail = error.detail || null;
    this.field = error.field || null;
  }
}

async function call(method, path, body, { raw = false, contentType } = {}) {
  const init = { method, headers: { accept: "application/json" } };
  if (auth.token) init.headers[TOKEN_HEADER] = auth.token;
  if (body !== undefined) {
    init.headers["content-type"] = raw ? contentType || "application/octet-stream" : "application/json";
    init.body = raw ? body : JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(BASE + path, init);
  } catch {
    throw new ApiError("tuningfork is not answering — has the server stopped?", 0);
  }
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!response.ok) {
    const error = data?.error || {};
    if (error.code === "token_missing" || error.code === "token_wrong") refused();
    throw new ApiError(error.message || `HTTP ${response.status}`, response.status, error);
  }
  return data;
}

/** The page a job is started from, as the job records it (`started_by.page`): its section,
 *  or the one thing it shows — a run, a step of one, a refinement, a scenario. */
export function startedFrom(hash = location.hash) {
  const [section = "", id, step] = hash.replace(/^#\/?/, "").split("?")[0].split("/");
  if (section === "runs" && id && /^\d+$/.test(step ?? "")) return "step";
  if (section === "runs" && id && step === "scenario") return "make-scenario";
  const one = { runs: "run", refine: "refinement", scenarios: "scenario", experiments: "experiment", skills: "skill", live: "live" };
  return (id && one[section]) || section || "guide";
}

// Work a person starts here says so, and from which page; a dry run starts nothing.
const startedHere = (path, body) => (path === "/jobs" && body && !body.dry_run && !body.started_by
  ? { ...body, started_by: { via: "web", page: startedFrom() } }
  : body);

export const api = {
  get: (path) => call("GET", path),
  post: (path, body = {}) => call("POST", path, startedHere(path, body)),
  patch: (path, body) => call("PATCH", path, body),
  del: (path) => call("DELETE", path),
  /** Raw bytes — a file as it is — rather than JSON. */
  put: (path, bytes, contentType) => call("PUT", path, bytes, { raw: true, contentType }),
};

/** A dry run's answer asked for `quiet`: a refusal comes back as `{refused}` rather than an
 *  HTTP error the browser would log, and is thrown here as the same error. */
export function answered(data) {
  if (data?.refused) throw new ApiError(data.refused.message || "refused", 200, data.refused);
  return data;
}

export const enc = encodeURIComponent;

export function query(params) {
  const entries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "");
  const text = new URLSearchParams(entries).toString();
  return text ? `?${text}` : "";
}

export const EVENTS_URL = `${BASE}/events`;
