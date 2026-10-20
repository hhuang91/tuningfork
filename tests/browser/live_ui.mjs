// Drive the Live page in a real browser through the DevTools protocol, and check what the
// harness actually received — not just what the page shows. See README.md for setup:
// a Chromium-based browser with a debugging port, and `demo.py live` running.
//
// Node 22+, for its built-in WebSocket. Exits non-zero on the first failed check. The
// environment it reads is described in cdp.mjs.

import { BASE, Page, TOKEN, api, check, post, sleep, until } from "./cdp.mjs";

// The first step of the newest run matching, among the most recent few.
async function recordedStep(matches) {
  for (const run of await api("/runs?limit=5")) {
    const step = await api(`/runs/${run.id}/steps/0`);
    if (step && matches(step)) return step;
  }
  return null;
}

const argField = (key) =>
  `[...document.querySelectorAll(".arg")].find((arg) => arg.querySelector(".arg-key .mono")?.textContent === ${JSON.stringify(key)})?.querySelector("textarea")`;

const nonce = Math.random().toString(36).slice(2, 8);
const harnessCall = (content) =>
  fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "gemma-4-31b", messages: [{ role: "user", content }] }),
  }).then((response) => response.json());

// The demo sets up its breakpoints and makes one call of its own shortly after it starts;
// wait for that, so it cannot race the calls below.
await until(async () => (await api("/breakpoints")).some((bp) => bp.tool === "write_file" && bp.hits >= 1), "the demo's own call to be held");

// Start clean: release anything a previous run or the demo left held, and make sure the
// demo's write_file breakpoint is on.
for (const hold of await api("/holds")) await post(`/holds/${hold.id}/continue`);
for (const bp of await api("/breakpoints")) {
  if (bp.tool === "write_file") await api(`/breakpoints/${bp.id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: '{"enabled": true}' });
  else await api(`/breakpoints/${bp.id}`, { method: "DELETE" });
}

const page = await Page.open(`${BASE}/_tuningfork/ui/?token=${TOKEN}#/live`);
try {
  // --- 0. The token: picked up from the address, then taken out of it ------------------
  console.log("the access token");
  await until(() => page.eval(`location.search === "" && location.hash === "#/live"`), "the token to leave the address bar");
  check(await page.eval(`sessionStorage.getItem("tuningfork.token") === ${JSON.stringify(TOKEN)}`), "the token is kept for this tab, out of the address bar");
  check(await page.eval(`document.getElementById("token-banner").hidden`), "with it, the page is not read-only");

  // --- 1. A response hold: fix the JSON the model is about to write, then continue ----
  console.log("response hold: edit write_file content, continue");
  const writing = harnessCall(`write the tally report (${nonce})`);
  const [held] = await until(async () => (await api("/holds")).length && api("/holds"), "the harness's call to be held");
  await until(() => page.eval(`!!document.querySelector(".hold-head")`), "the hold to render");
  check((await page.text(".hold-head .reason")) === "write_file on *.json", "the reason drops the repeated phase");
  check(await page.eval(`(() => { const badge = document.getElementById("held-count"); return !badge.hidden && badge.classList.contains("badge-warn"); })()`),
    "the held call's badge is a caution's amber");
  check((await page.eval(`${argField("content")}.value`)).includes('"count"'), "the content argument is shown raw, unescaped");

  const fixed = JSON.stringify({ row_count: 4, amount_total: 61.5 }, null, 2);
  await page.type(argField("content"), fixed);
  check((await page.text(".edit-status")).startsWith("Edited"), "the editor notices the edit");
  check(await page.eval(`[...document.querySelectorAll(".action-bar button")].some((b) => b.textContent === "Continue with edits")`), "Continue says it sends the edit");
  await page.click("Continue with edits");

  const decided = await until(async () => (await api("/holds?recent=1")).find((hold) => hold.id === held.id && !hold.pending), "the decision");
  check(decided.action === "continue" && decided.edited, "the proxy recorded continue, with an edit");
  const received = await writing;
  check(JSON.parse(received.choices[0].message.tool_calls[0].function.arguments).content === fixed, "the harness received the edited content");
  const step = await until(() => recordedStep((recorded) => JSON.stringify(recorded.request).includes(nonce)), "the recorded step");
  const delivered = JSON.parse(step.delivered_response.choices[0].message.tool_calls[0].function.arguments);
  const produced = JSON.parse(step.response.choices[0].message.tool_calls[0].function.arguments);
  check(delivered.content === fixed, "the step records what the harness was given");
  check(produced.content.includes('"count"'), "the step still records what the model produced");
  check(step.edits.response.message_fields.includes("tool_calls"), "the edit is described on the step");
  await until(() => page.eval(`document.body.textContent.includes("Nothing is held")`), "the page to go idle");
  check(true, "the page goes idle once nothing is held");
  // What was decided, in grey; the edit a person made, in the teal of a change, never a verdict's green.
  await until(() => page.eval(`!!document.querySelector(".recent-item .chip.tone-change")`), "the decision among the recent ones");
  check(await page.eval(`(() => {
    const item = [...document.querySelectorAll(".recent-item")].find((row) => row.querySelector(".chip.tone-change"));
    const [action, edited] = item.querySelectorAll(".chip");
    return action.textContent === "continue" && action.classList.contains("tone-muted") && edited.textContent === "edited"
      && !document.querySelector(".recent-item .chip.tone-ok");
  })()`), "the decision is grey, its edit teal");

  // --- 2. A request hold: edit the context, step, then continue after the model -------
  console.log("request hold: edit a message, step, continue");
  const breakpoint = await post("/breakpoints", { phase: "request", context_regex: "e2e-marker", once: true });
  const harness = harnessCall(`e2e-marker ${nonce}: say hi`);
  await until(() => page.eval(`document.querySelector(".hold-head .chip")?.textContent === "before the model"`), "the request hold to render");
  await page.eval(`(() => { for (const d of document.querySelectorAll(".editor details.msg")) d.open = true; return true; })()`);
  await page.type(`document.querySelector(".editor details.msg textarea")`, `e2e-marker ${nonce}: say hello instead`);
  await page.click("Step to the answer");
  await until(() => page.eval(`document.querySelector(".hold-head .chip")?.textContent === "after the model"`), "the stepped response hold");
  check((await page.text(".hold-head .reason")) === "stepped here from before the call", "stepping stops again after the model");
  await page.click("Continue");
  const answer = await harness;
  check(answer.choices[0].message.content === "ok", "the harness got the model's answer");
  const stepped = await until(
    () => recordedStep((recorded) => JSON.stringify(recorded.harness_request ?? "").includes(nonce)),
    "the stepped call's recording",
  );
  check(stepped.request.messages[0].content === `e2e-marker ${nonce}: say hello instead`, "the model saw the edited message");
  check(stepped.harness_request.messages[0].content === `e2e-marker ${nonce}: say hi`, "the harness's own request is kept too");
  await until(() => page.eval(`document.body.textContent.includes("Nothing is held")`), "the page to go idle again");
  check(!(await page.eval(`document.body.textContent.includes("decided elsewhere")`)), "a decision made here is not reported as made elsewhere");
  check((await api("/breakpoints")).find((bp) => bp.id === breakpoint.id).enabled === false, "a once breakpoint disarms itself");

  // --- 3. Breakpoints from the form, and the switch -----------------------------------
  console.log("breakpoint form and switch");
  await page.eval(`document.querySelector(".add-bp").open = true`);
  await page.type(`document.querySelector('.bp-form input[name="tool"]')`, "terminal");
  check((await page.text(".bp-summary")).startsWith("Stops after the model answers"), "a tool condition means after the model");
  await page.eval(`document.querySelector(".bp-form").requestSubmit()`);
  const added = await until(async () => (await api("/breakpoints")).find((bp) => bp.tool === "terminal"), "the new breakpoint");
  check(added.phase === "response", "the form inferred the response phase");
  await until(() => page.eval(`[...document.querySelectorAll(".bp-desc")].some((el) => el.textContent.includes("terminal"))`), "the list to update");
  await page.eval(`[...document.querySelectorAll(".bp")].find((el) => el.textContent.includes("terminal")).querySelector("input").click()`);
  await until(async () => (await api("/breakpoints")).find((bp) => bp.id === added.id).enabled === false, "the switch to disable it");
  check(true, "the switch turns a breakpoint off");
  await page.type(`document.querySelector('.bp-form input[name="args_regex"]')`, "([unclosed");
  await page.eval(`document.querySelector(".add-bp").open = true; document.querySelector(".bp-form").requestSubmit()`);
  await until(() => page.eval(`!document.querySelector(".form-error").hidden`), "the form error");
  check((await page.text(".form-error")).includes("not a valid regex"), "an invalid regex is refused with the proxy's reason");

  // --- 4. Pause from the top bar ------------------------------------------------------
  console.log("pause and resume");
  await page.click("Pause");
  await until(async () => (await api("/status")).paused, "pause");
  await until(() => page.eval(`!document.getElementById("paused-banner").hidden`), "the paused banner");
  check(true, "pause reaches the proxy and the banner shows");
  await page.click("Resume");
  await until(async () => !(await api("/status")).paused, "resume");
  check(true, "resume reaches the proxy");

  // --- 5. Without the token: everything visible, nothing changes ------------------------
  console.log("read-only without the token");
  const bare = await Page.open(`${BASE}/_tuningfork/ui/#/live`); // a new tab: no session
  try {
    await until(() => bare.eval(`!document.getElementById("token-banner").hidden`), "the read-only banner");
    check(await bare.eval(`document.documentElement.classList.contains("read-only")`), "a page without the token is read-only");
    await bare.click("Pause");
    await sleep(500);
    check(!(await api("/status")).paused, "its Pause changes nothing");
    check(await bare.eval(`[...document.querySelectorAll(".toast")].some((t) => t.textContent.includes("Read-only"))`), "and says why");
    await bare.type(`document.querySelector("#token-banner input")`, TOKEN);
    await bare.eval(`document.querySelector("#token-banner form").requestSubmit()`);
    await until(() => bare.eval(`document.getElementById("token-banner").hidden`), "the pasted token to be accepted");
    await bare.click("Pause");
    await until(async () => (await api("/status")).paused, "pause from the page given the token");
    await bare.click("Resume");
    await until(async () => !(await api("/status")).paused, "resume");
    check(true, "a pasted token lets the page act");
    check(bare.problems.length === 0, `no errors on that page${bare.problems.length ? `: ${bare.problems.join(" | ")}` : ""}`);
  } finally {
    await bare.close();
  }

  check(page.problems.length === 0, `no page errors${page.problems.length ? `: ${page.problems.join(" | ")}` : ""}`);
  console.log("all passed");
} catch (error) {
  console.log(error.message);
  console.log("page errors:", page.problems);
  console.log("hash:", await page.eval("location.hash"));
  console.log("main:", (await page.eval(`document.querySelector("main").innerText`)).slice(0, 900));
  process.exitCode = 1;
} finally {
  await page.close();
}
