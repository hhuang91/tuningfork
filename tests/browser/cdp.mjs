// What the browser tests share: a page driven over the DevTools protocol, the demo's API
// with its access token, and small waiting and checking helpers.
//
//   TUNINGFORK_BROWSER  the browser's debugging endpoint   (default http://127.0.0.1:9333)
//   TUNINGFORK_DEMO     the demo proxy                      (default http://127.0.0.1:8791)
//   TUNINGFORK_TOKEN    the demo's access token  (default: read from the file it names at /health)

import { readFileSync } from "node:fs";

export const DEBUG = process.env.TUNINGFORK_BROWSER ?? "http://127.0.0.1:9333";
export const BASE = process.env.TUNINGFORK_DEMO ?? "http://127.0.0.1:8791";
export const API = `${BASE}/_tuningfork/api`;

async function findToken() {
  if (process.env.TUNINGFORK_TOKEN) return process.env.TUNINGFORK_TOKEN;
  const health = await (await fetch(`${BASE}/_tuningfork/health`)).json();
  return readFileSync(health.token_file, "utf8").trim();
}
export const TOKEN = await findToken();

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function api(path, init = {}) {
  const response = await fetch(API + path, { ...init, headers: { ...init.headers, "x-tuningfork-token": TOKEN } });
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export const post = (path, body) =>
  api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body ?? {}) });

export async function until(check, what, timeout = 15000) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    try {
      last = await check();
      if (last) return last;
    } catch (error) {
      last = error;
    }
    await sleep(150);
  }
  throw new Error(`timed out waiting for ${what} (last: ${last})`);
}

export function check(condition, message) {
  if (!condition) throw new Error(`FAILED: ${message}`);
  console.log(`  ok  ${message}`);
}

export class Page {
  constructor(socket) {
    this.socket = socket;
    this.next = 0;
    this.waiting = new Map();
    this.problems = [];
    this.dialogs = [];
    this.dialogAnswer = true;
    // An event the caller listens for, by its method (`on`): a screencast's frames, say.
    this.listeners = new Map();
    socket.onmessage = ({ data }) => {
      const message = JSON.parse(data);
      if (message.id && this.waiting.has(message.id)) {
        const { resolve, reject } = this.waiting.get(message.id);
        this.waiting.delete(message.id);
        message.error ? reject(new Error(message.error.message)) : resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") {
        this.problems.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
      } else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
        this.problems.push(message.params.args.map((arg) => arg.value ?? arg.description).join(" "));
      } else if (message.method === "Page.javascriptDialogOpening") {
        // A confirm() the page asked: answered as `dialogAnswer` says, and kept.
        this.dialogs.push(message.params.message);
        this.send("Page.handleJavaScriptDialog", { accept: this.dialogAnswer });
      } else if (message.method === "Log.entryAdded" && message.params.entry.source === "security") {
        // A content-security-policy violation: the page tried to run what it must not.
        this.problems.push(`security: ${message.params.entry.text}`);
      } else if (this.listeners.has(message.method)) {
        this.listeners.get(message.method)(message.params);
      }
    };
  }

  /** Call `handler` with the params of each `method` event the page sends. */
  on(method, handler) {
    this.listeners.set(method, handler);
  }

  static async open(url) {
    const target = await (await fetch(`${DEBUG}/json/new?${encodeURIComponent(url)}`, { method: "PUT" })).json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = reject;
    });
    const page = new Page(socket);
    page.id = target.id;
    await page.send("Runtime.enable");
    await page.send("Log.enable");
    await page.send("Page.enable");
    return page;
  }

  send(method, params = {}) {
    const id = ++this.next;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.waiting.set(id, { resolve, reject }));
  }

  async eval(expression) {
    const result = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }

  // Set a field the way typing does, so the page's input handlers run.
  type(selectorExpression, value) {
    return this.eval(`(() => {
      const field = ${selectorExpression};
      if (!field) throw new Error("no such field");
      field.value = ${JSON.stringify(value)};
      field.dispatchEvent(new Event("input", { bubbles: true }));
      return true;
    })()`);
  }

  // Type into a field as a person does: focused, what it says replaced, and left focused, so its
  // change event fires when the focus moves on, as the person's next click moves it.
  async enter(selectorExpression, value) {
    await this.eval(`(() => {
      const field = ${selectorExpression};
      if (!field) throw new Error("no such field");
      field.focus();
      field.select();
      return true;
    })()`);
    await this.send("Input.insertText", { text: value });
  }

  click(buttonText, scope = "document") {
    return this.eval(`(() => {
      const button = [...${scope}.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(${JSON.stringify(buttonText)}) && !b.disabled);
      if (!button) throw new Error("no enabled button " + ${JSON.stringify(buttonText)});
      button.click();
      return true;
    })()`);
  }

  // Press a button with the mouse, as a person does. `click` calls it from script, which moves no
  // focus; a press moves it first, and a button that what follows disables never gets the click.
  async press(buttonText, scope = "document") {
    const at = await this.eval(`(() => {
      const button = [...${scope}.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(${JSON.stringify(buttonText)}) && !b.disabled);
      if (!button) throw new Error("no enabled button " + ${JSON.stringify(buttonText)});
      button.scrollIntoView({ block: "center", behavior: "instant" });
      const box = button.getBoundingClientRect();
      return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    })()`);
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: at.x, y: at.y });
    for (const type of ["mousePressed", "mouseReleased"]) {
      await this.send("Input.dispatchMouseEvent", { type, x: at.x, y: at.y, button: "left", clickCount: 1 });
    }
  }

  text(selector) {
    return this.eval(`document.querySelector(${JSON.stringify(selector)})?.textContent ?? null`);
  }

  async close() {
    this.socket.close();
    await fetch(`${DEBUG}/json/close/${this.id}`).catch(() => {});
  }
}
