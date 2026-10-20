// Start work from the browser, and check what the proxy ran — not just what the page shows.
// Needs `demo.py lab` running (see README.md): the proxy over seeded recordings, in front of
// a model that names the report's keys as the skill does.
//
// Node 22+. Exits non-zero on the first failed check. The environment is read by cdp.mjs.

import { BASE, Page, TOKEN, api, check, post, sleep, until } from "./cdp.mjs";

const UI = `${BASE}/_tuningfork/ui/?token=${TOKEN}`;
const run = (await api("/runs?task=tally-report/1&limit=20")).find((found) => found.label === "tally-report/1#demo.0");
if (!run) throw new Error("no seeded run: start `demo.py lab` on a fresh home");

const jobIdIn = (hash) => /#\/jobs\/([0-9a-f]{32})/.exec(hash)?.[1];

async function openStep(page, start) {
  await page.eval(`location.hash = "#/runs/${run.id}/1${start ? `?start=${start}` : ""}"`);
  // A step's address opens the run's timeline with that step open, a form in it if asked.
  await until(() => page.eval(`!!document.querySelector("#s1.tl-open .start-actions")`), "the step, open on the run's timeline");
}

async function openStepOf(page, runId, k, start) {
  await page.eval(`location.hash = "#/runs/${runId}/${k}?start=${start}"`);
  await until(() => page.eval(`!!document.querySelector(".start-panel .plan-form")`), `the form (${start})`);
}

async function planShows(page, text) {
  return until(() => page.eval(`(document.querySelector(".plan-preview")?.textContent || "").includes(${JSON.stringify(text)})`), `the plan to show "${text}"`);
}

async function startButtonReady(page, label) {
  return until(() => page.eval(`[...document.querySelectorAll(".plan-form button")].some((b) => b.textContent === ${JSON.stringify(label)} && !b.disabled)`), `"${label}" to be enabled`);
}

async function finished(jobId, timeout = 60000) {
  return until(async () => {
    const job = await api(`/jobs/${jobId}`);
    return ["queued", "running"].includes(job.status) ? null : job;
  }, `job ${jobId.slice(0, 8)} to end`, timeout);
}

const page = await Page.open(`${UI}#/runs`);
try {
  // The tab loads after it opens: wait for the app, and for it to take the token in.
  await until(() => page.eval(`location.pathname.endsWith("/ui/") && location.search === "" && !!document.querySelector(".view")`), "the app to load");
  // --- 1. Resample from the step page -----------------------------------------------------
  console.log("resample a step from its page");
  await openStep(page);
  check(await page.eval(`[...document.querySelectorAll(".start-actions a")].map((a) => a.textContent).join("|") === "Resample…|A/B a wording…|Find forking tokens…|Fork from here…|Refine the skill here…"`), "the step offers a resample, an A/B, the forking-token probe, a fork and a refinement");
  // The lab's runs were launched by a batch, as their check's verdicts say (found in review: since
  // runs nothing launched read "recorded live", every seeded run did).
  check(await page.eval(`!document.querySelector(".page-head h1 .chip") && document.querySelector(".about-run summary").textContent.includes("launched by tuningfork, batch")`),
    "a seeded run reads as a batch's, not as one recorded live");
  // A run with no workspace snapshot shows its timings alone, across the width (found in review: half a grid).
  check(await page.eval(`(() => { const details = document.querySelector("#s1 .tl-details"); return !!details && !details.classList.contains("grid-2"); })()`),
    "a step with one panel of details lays it across the width, not in half a grid");
  await openStep(page, "resample");
  await planShows(page, "20 samples");
  check(await page.eval(`(document.querySelector(".plan-preview").textContent).includes(${JSON.stringify(BASE)})`), "the plan says its calls go through this proxy");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "6");
  check(await page.eval(`[...document.querySelectorAll(".plan-form button")].find((b) => b.textContent === "Start the resample").disabled`), "Start waits for the plan of what the form now says");
  await planShows(page, "6 samples");
  await startButtonReady(page, "Start the resample");
  await page.click("Start the resample");
  const resampleId = await until(async () => jobIdIn(await page.eval("location.hash")), "the job page");
  const resample = await finished(resampleId);
  check(resample.status === "done" && resample.result.distribution.total === 6, "the resample ran: 6 samples");
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "See the result")`), "the link to the result");
  check(await page.eval(`document.querySelectorAll(".job-log .log-line").length >= 7`), "the job's page logged the experiment and each sample");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "See the result").click()`);
  await until(() => page.eval(`location.hash.startsWith("#/experiments/")`), "the experiment page");
  await until(() => page.eval(`!!document.querySelector(".groups .letter")`), "the distribution");
  check(await page.eval(`document.querySelectorAll(".groups .group").length === 1`), "one behaviour: the wrong keys, every time");

  // A form asks for its plan as it opens, and again when the page refreshes it — after work
  // elsewhere that the plan reads — but not for an event that changes nothing it says (found
  // in review: the one-press fix below had stopped a refresh asking).
  const asked = await page.eval(`(async () => {
    const { planForm } = await import(new URL("js/forms.js", location.href).href);
    const { api } = await import(new URL("js/api.js", location.href).href);
    const post = api.post;
    let dry = 0;
    api.post = (path, body) => { if (body && body.dry_run) dry += 1; return post(path, body); };
    const settle = () => new Promise((resolve) => setTimeout(resolve, 1200));
    const field = document.createElement("input");
    const params = { run: ${JSON.stringify(run.id)}, step: "1", n: 2, concurrency: 1, max_tokens: 4096,
      prompt_cache: false, normalizers: [], temperature: null, top_p: null, top_k: null };
    const { element, refresh } = planForm({ kind: "resample", fields: [field], read: () => params,
      render: () => document.createElement("div"), onStarted: () => {} });
    document.body.append(element);
    try {
      await settle();
      const opened = dry;
      field.dispatchEvent(new Event("change", { bubbles: true }));
      await settle();
      const unchanged = dry;
      refresh();
      await settle();
      return { opened, unchanged, refreshed: dry, ready: !element.querySelector("button").disabled };
    } finally {
      api.post = post;
      element.remove();
    }
  })()`);
  check(asked.opened === 1 && asked.unchanged === 1 && asked.refreshed === 2 && asked.ready,
    `a form asks for its plan as it opens, not for an event that changes nothing, and again when refreshed (${JSON.stringify(asked)})`);

  // --- 2. An A/B of a wording edited in the page, with a lettered target ----------------------
  console.log("A/B a wording edited in the page");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "A/B a wording at this step").click()`);
  await until(() => page.eval(`!!document.querySelector('.plan-form [name="variant_text"]')`), "the A/B form");
  check(await page.eval(`document.querySelector('[name="variant_text"]').value.includes("Write the totals to report.json.")`), "the editor starts from the skill as the run was given it");
  await until(() => page.eval(`(document.querySelector(".plan-status")?.textContent || "").includes("Choose what the A/B is about")`), "the form to ask for a target first");
  check(await page.eval(`!!document.querySelector(".plan-status .form-note-box") && !document.querySelector(".plan-status .form-error-box")`),
    "and asks it as a note: a form nobody has touched has done nothing wrong (found in review: it opened in red)");
  check(await page.eval(`![...document.querySelectorAll('[name="target_kind"]')].some((radio) => radio.checked)`), "no target is chosen for you");
  await page.type(`document.querySelector('[name="variant_text"]')`,
    (await page.eval(`document.querySelector('[name="variant_text"]').value`)).replace("Write the totals to report.json.", "Write the totals to report.json, using the keys row_count and amount_total."));
  check((await page.text(".editor-head .muted")).startsWith("1 line changed"), "the editor counts the changed line");
  const variantName = `browser-${Date.now().toString(36)}`;
  // Typed for real, and left focused until Start is pressed: its change fires as the press moves the focus.
  await page.enter(`document.querySelector('[name="variant_name"]')`, variantName);
  await page.eval(`document.querySelector('[name="target"][value="A"]').click()`);
  check(await page.eval(`document.querySelector('[name="target_kind"][value="letter"]').checked`), "picking a behaviour picks the lettered target");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "6");
  await planShows(page, "6 samples per arm");
  check(await page.eval(`document.querySelector(".plan-preview").textContent.includes(${JSON.stringify(`${variantName}.md`)})`), "the plan names where the variant will be saved");
  await startButtonReady(page, "Start the A/B");
  await page.press("Start the A/B");
  const abId = await until(async () => jobIdIn(await page.eval("location.hash")), "the A/B's job page, after one press of Start", 5000);
  check(!!abId, "one press of Start starts the A/B, the name just typed still focused as it is pressed");
  const ab = await finished(abId);
  const target = ab.result?.comparison?.target;
  check(ab.status === "done" && target && target.original.successes === 6 && target.variant.successes === 0 && target.significant,
    "the variant moved the model off behaviour A: 6/6 to 0/6, significant");
  const variants = await api("/variants?skill=tally-report");
  check(variants.some((variant) => variant.name === variantName), "the variant was saved when the A/B started");
  await until(() => page.eval(`!!document.querySelector(".verdict-block.significant")`), "the verdict on the job page");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "See the result").click()`);
  await until(() => page.eval(`location.hash.startsWith("#/experiments/")`), "the A/B's page");
  const edit = `[...document.querySelectorAll(".panel")].find((panel) => panel.querySelector("h2")?.textContent === "The variant's edit")`;
  await until(() => page.eval(`!!${edit}`), "the edit it tested");
  check(await page.eval(`[...${edit}.querySelectorAll(".diff-line.add")].some((line) => line.textContent.includes("row_count and amount_total"))
    && ${edit}.querySelector(".panel-meta").textContent.includes(${JSON.stringify(`+ variant ${variantName}`)})`),
    "the A/B's page leads with the edit it tested, as a diff, the variant by its saved name");
  // The target first, then which way the variant moved it: the wrong keys, behaviour A, fell.
  check(await page.eval(`(() => {
    const block = document.querySelector(".verdict-block");
    return block.classList.contains("significant") && !block.classList.contains("tone-ok")
      && block.firstElementChild.classList.contains("verdict-target")
      && block.querySelector(".verdict-target-text").textContent.startsWith("A from resample ")
      && block.querySelector(".verdict-big").textContent === "The variant hit the target less often: 0 of 6 samples, against 6 of 6 with the original";
  })()`), "the result says the target first, then which way the variant moved it, in the accent's colour, not green");
  const marked = await page.eval(`[...document.querySelectorAll(".table tr.is-target")].map((row) =>
    [row.querySelector(".chip.tone-target")?.textContent, row.children[2].textContent.split(" ")[0]].join(":")).join(",")`);
  check(marked === "target:6/6", `the behaviours table marks the one the target counts: the original's every sample (${marked})`);
  // Declared as the resample's A, it is the table's A: the two arms tie, and lettered afresh the
  // other behaviour came first, the target row B (found in use).
  const targetLetter = await page.eval(`document.querySelector(".table tr.is-target .chip.tone-accent")?.textContent`);
  check(targetLetter === "A", `the target is row A, as the resample it was named from lettered it (${targetLetter})`);
  const others = await page.eval(`[...document.querySelectorAll(".table tbody tr")].filter((row) => !row.classList.contains("is-target")).map((row) =>
    [row.querySelector(".chip.tone-not-target")?.textContent, row.children[3].textContent.split(" ")[0]].join(":")).join(",")`);
  check(others === "not target:6/6", `and badges the other "not target", its 6/6 the variant's, not the target's (${others})`);

  // --- 3. A task run again from its run's page; a fork held at its step; forks to the end ----
  console.log("run a task again, fork at a step, and run forks to the end");
  const health = await (await fetch(`${BASE}/_tuningfork/health`)).json();
  const spec = `${health.home}/lab-task/evals.json`;
  const first = await finished((await api("/jobs", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ kind: "batch", params: { spec, n: 1, harness: "fake", model: "m" } }),
  })).id, 90000);
  check(first.status === "done" && first.links.runs.length === 1, "a batch of the lab task, started over the API, ran");
  const launched = first.links.runs[0];

  await page.eval(`location.hash = "#/runs/${launched}"`);
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "Run this task again…")`), "the run's way to run it again");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "Run this task again…").click()`);
  await until(() => page.eval(`document.querySelector('.plan-form [name="harness"]')?.value === "fake"`), "the batch form, as the run was launched");
  await planShows(page, "tally-lab/1");
  check(await page.eval(`document.querySelector(".check-command pre")?.textContent.includes("row_count")`), "the check is shown whole, as the shell command it is");
  check(await page.eval(`document.querySelector(".plan-preview > .stack > :first-child")?.classList.contains("check-command")`), "how a run passes comes first in the plan");
  check(await page.eval(`[...document.querySelector('.plan-form [name="eval"]').options].some((option) => option.textContent.includes(" — "))`),
    "the spec's task is offered by what it asks");
  // Found by a screenshot: naming the task changed the form under its own plan, and Start
  // waited for a plan that never came — until something else was typed.
  await startButtonReady(page, "Start the runs");
  check(true, "Start is ready once the plan has named the task, with nothing else typed");
  // Another spec typed in puts the task list back to its first (found in review: the field's
  // listener was dropped, and the list kept the other spec's tasks); typed back, it plans again.
  const specPath = await page.eval(`document.querySelector('.plan-form [name="spec"]').value`);
  await page.type(`document.querySelector('.plan-form [name="spec"]')`, `${specPath}.other`);
  check(await page.eval(`(() => { const options = document.querySelector('.plan-form [name="eval"]').options;
    return options.length === 1 && options[0].value === "" && options[0].textContent === "its first task"; })()`),
    "another spec typed in puts the task list back to its first");
  await page.type(`document.querySelector('.plan-form [name="spec"]')`, specPath);
  await planShows(page, "tally-lab/1");
  await startButtonReady(page, "Start the runs");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "2");
  await planShows(page, "2 runs of fake");
  await startButtonReady(page, "Start the runs");
  await page.click("Start the runs");
  const batchId = await until(async () => jobIdIn(await page.eval("location.hash")), "the batch's job page");
  const batch = await finished(batchId, 120000);
  check(batch.status === "done" && batch.result.rate.total === 2 && batch.result.rate.successes === 0, "the batch ran twice: 0/2, the known failure");
  await page.eval(`location.hash = "#/runs"`);
  await until(() => page.eval(`[...document.querySelectorAll(".passes-when")].some((line) => line.textContent.includes("row_count"))`), "the task's check on the Runs page");
  check(true, "on the Runs page, the task says how its runs pass");

  await openStepOf(page, launched, 1, "fork");
  await planShows(page, "Stop at the step");
  await startButtonReady(page, "Start the fork");
  await page.click("Start the fork");
  const forkId = await until(async () => jobIdIn(await page.eval("location.hash")), "the fork's job page");
  await until(() => page.eval(`[...document.querySelectorAll(".banner a.btn")].some((a) => a.textContent === "Open it: step 1")`), "the job page to show its held fork", 60000);
  await page.eval(`[...document.querySelectorAll(".banner a.btn")].find((a) => a.textContent === "Open it: step 1").click()`);
  await until(() => page.eval(`!!document.querySelector(".hold-head")`), "the held fork on the Live page");
  await page.click("Continue");
  const fork = await finished(forkId, 90000);
  check(fork.status === "done" && fork.result.length === 1 && fork.result[0].run_id, "a fork stopped at its step for a person, then ran on");

  // A fork cancelled while it waits at its step keeps no verdict, and its page says it was
  // stopped, and why — never that the task defines no check (found in review).
  const heldId = (await post("/jobs", { kind: "fork", params: { run: launched, step: "1", hold: "step", n: 1, concurrency: 1 } })).id;
  const heldRun = await until(async () => (await api("/holds"))[0]?.run_id, "the second fork, held at its step", 60000);
  await post(`/jobs/${heldId}/cancel`);
  check((await finished(heldId, 60000)).status === "cancelled", "a second fork, held at its step, cancelled");
  await page.eval(`location.hash = "#/runs/${heldRun}"`);
  const stoppedText = await until(() => page.eval(`document.querySelector("#verdict")?.textContent || ""`), "the cancelled fork's verdict");
  check(stoppedText.includes("the run was stopped") && stoppedText.includes("Cancelled before it finished") && !stoppedText.includes("defines no check"),
    `a cancelled run says it was stopped, and why (${stoppedText})`);
  const headChips = await page.eval(`[...document.querySelectorAll(".page-head .chip")].map((c) => c.textContent).join(",")`);
  check(headChips.includes("stopped") && !headChips.includes("no check"), `its page's head says so too, not "no check" (${headChips})`);

  await openStepOf(page, launched, 1, "resample&reach=end");
  await planShows(page, "the task's check passes");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "2");
  await planShows(page, "2 forks");
  await startButtonReady(page, "Start the forks");
  await page.click("Start the forks");
  const toEndId = await until(async () => jobIdIn(await page.eval("location.hash")), "the forks' job page");
  const toEnd = await finished(toEndId, 120000);
  check(toEnd.status === "done" && toEnd.result.counted === 2 && toEnd.result.passed === 0, "resampled to the end: 0/2 pass the check");

  await openStepOf(page, launched, 1, "ab&reach=end");
  check(!(await page.eval(`!!document.querySelector(".target-picker legend")?.textContent.includes("What the A/B is about")`)), "to the end, the target is the task's check: no picker");
  await page.type(`document.querySelector('[name="variant_text"]')`,
    (await page.eval(`document.querySelector('[name="variant_text"]').value`)).replace("Write the totals to report.json.", "Write the totals to report.json, using the keys row_count and amount_total."));
  await page.type(`document.querySelector('[name="variant_name"]')`, `to-end-${Date.now().toString(36)}`);
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "1");
  await planShows(page, "(1 per arm)");
  await startButtonReady(page, "Start the A/B");
  await page.click("Start the A/B");
  const abEndId = await until(async () => jobIdIn(await page.eval("location.hash")), "the A/B's job page");
  const abEnd = await finished(abEndId, 120000);
  check(abEnd.status === "done" && abEnd.result.original.passed === 0 && abEnd.result.variant.passed === 1,
    "an A/B to the end: the original fails the check, the variant passes");
  await page.eval(`location.hash = "#/experiments/${abEnd.links.experiments[0]}"`);
  await until(() => page.eval(`!!document.querySelector(".verdict-block .verdict-target-text")`), "the A/B to the end's page");
  check(await page.eval(`(() => {
    const block = document.querySelector(".verdict-block");
    return block.querySelector(".verdict-target-text").textContent === "the task's check passes"
      && block.querySelector(".verdict-big").textContent === "No significant difference: the variant's forks passed the check in 1 of 1, the original's in 0 of 1"
      && !block.classList.contains("tone-ok") && !block.classList.contains("tone-bad");
  })()`), "its page names the check as the target, and one fork an arm is said in words, not in green");

  // --- 4. Cancel a running job ------------------------------------------------------------
  console.log("cancel a job");
  await openStep(page, "resample");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "300");
  await planShows(page, "300 samples");
  await startButtonReady(page, "Start the resample");
  await page.click("Start the resample");
  const longId = await until(async () => jobIdIn(await page.eval("location.hash")), "the long job's page");
  await until(async () => ((await api(`/jobs/${longId}`)).progress?.done || 0) >= 3, "some samples");
  // Its running time moves each second with no event to redraw it: paused, it sends none.
  await post("/pause");
  await until(async () => (await api(`/jobs/${longId}`)).progress?.waiting === "paused", "the job to wait out the pause");
  await sleep(700);
  const shownBefore = await page.eval(`document.querySelector(".panel-meta [data-since]")?.textContent`);
  await sleep(2200);
  const shownAfter = await page.eval(`document.querySelector(".panel-meta [data-since]")?.textContent`);
  check(Boolean(shownBefore && shownAfter) && shownBefore !== shownAfter, `a running job's time moves on its own (${shownBefore}, then ${shownAfter})`);
  await post("/resume");

  // --- 4. The event stream cut mid-job: every event shown once ----------------------------
  // What the bus does to a page that falls a whole queue behind (the demo's /_demo/blip does
  // it on demand). A browser's own offline switch leaves an open stream alone.
  console.log("the event stream cut while it runs");
  const cut = await (await fetch(`${BASE}/_demo/blip`, { method: "POST" })).json();
  check(cut.cut >= 1, "the page's stream is cut");
  await until(() => page.eval(`document.querySelector(".status")?.textContent.includes("reconnecting")`), "the page to notice", 8000);
  const before = await page.eval(`document.querySelectorAll(".job-log .log-line").length`);
  await until(() => page.eval(`document.querySelector(".status")?.textContent.includes("live")`), "the page to reconnect", 15000);
  await sleep(1500);
  const after = await page.eval(`document.querySelectorAll(".job-log .log-line").length`);
  check(after > before, `it caught up on what came while it was away (${before} lines, then ${after})`);
  const seqs = await page.eval(`[...document.querySelectorAll(".job-log .log-line > span:first-child")].map((el) => Number(el.textContent.slice(1)))`);
  check(seqs.length >= 5 && new Set(seqs).size === seqs.length && seqs.every((seq, i) => i === 0 || seq === seqs[i - 1] + 1),
    `after reconnecting, each event shows once and none is missing (${seqs.length} lines, #${seqs[0]}–#${seqs[seqs.length - 1]})`);

  await page.click("Cancel");
  check((await page.eval(`[...document.querySelectorAll("button")].find((b) => b.textContent.startsWith("Click again"))?.textContent`)) === "Click again: stop it", "the first click only arms it");
  await page.click("Click again");
  const cancelled = await finished(longId);
  check(cancelled.status === "cancelled", "the second click cancels it");
  const experiment = await api(`/experiments/${cancelled.links.experiments[0]}`);
  check(experiment.status === "cancelled" && experiment.partial.length === 1, "its experiment says it was cancelled, over the samples drawn");

  // --- 6. A scenario made in the page, a file uploaded into it, run as a batch --------------
  console.log("a scenario, made here and run");
  const scenarioName = `lab-${Date.now().toString(36)}`;
  await page.eval(`location.hash = "#/scenarios/new"`);
  await until(() => page.eval(`!!document.querySelector('[name="scenario_name"]')`), "the new scenario form");
  await page.type(`document.querySelector('[name="scenario_name"]')`, scenarioName);
  await page.type(`document.querySelector('[name="skill_name"]')`, "tally-scenario");
  await page.type(`document.querySelector('[name="skill_dir"]')`, `${health.home}/lab-task/tally-report`);
  await until(() => page.eval(`(document.querySelector(".path-line.path-ok")?.textContent || "").includes("holds a SKILL.md")`), "the skill folder, resolved");
  check(true, "the skill folder field says where it resolves, and that it holds a SKILL.md");
  await page.type(`document.querySelector('[name="prompt-0"]')`, "Summarize data.csv into report.json.");
  await page.type(`document.querySelector('[name="skills-0"]')`, "tally-report");
  await page.type(`document.querySelector('[name="turns-0"]')`, "6");
  await page.eval(`document.querySelector('input[name="check-kind-0"][value="shell"]').click()`);
  await until(() => page.eval(`!!document.querySelector('[name="check-0"]')`), "the shell command's field");
  await page.type(`document.querySelector('[name="check-0"]')`,
    "python3 -c \"import json,sys; d=json.load(open('report.json')); sys.exit(0 if d.get('row_count')==4 else 1)\"");
  await until(() => page.eval(`document.querySelector(".spec-preview").textContent.includes('"skill_name": "tally-scenario"')`), "the preview of evals.json");
  await until(() => page.eval(`[...document.querySelectorAll(".banner-warn")].some((b) => b.textContent.includes("shell command"))`), "the check flagged");
  check(true, "its check is flagged as the shell command it is");
  // Leaving with unsaved changes asks first; saying no keeps the page.
  page.dialogAnswer = false;
  await page.eval(`location.hash = "#/runs"`);
  await until(() => page.dialogs.length === 1, "the question before leaving unsaved work");
  await until(() => page.eval(`location.hash === "#/scenarios/new" && !!document.querySelector('[name="scenario_name"]')`), "the page to stay");
  check(page.dialogs[0].includes("not saved"), "leaving unsaved work asks first, and no keeps it");
  page.dialogAnswer = true;
  await page.click("Save");
  await until(() => page.eval(`location.hash === "#/scenarios/${scenarioName}"`), "the saved scenario's page");
  await until(() => page.eval(`!!document.querySelector(".drop-zone")?.textContent.startsWith("Drop files")`), "the drop zone");
  // A file chosen the way a person would, through the page's own file picker.
  const csv = process.env.TUNINGFORK_UPLOAD;
  if (csv) {
    const picker = await page.send("Runtime.evaluate", { expression: `document.querySelector('.drop-zone + input[type="file"]')` });
    await page.send("DOM.setFileInputFiles", { files: [csv], objectId: picker.result.objectId });
    await until(() => page.eval(`document.querySelector(".spec-preview").textContent.includes('"source": "inputs/data.csv"')`), "the uploaded file in the spec");
    check(await page.eval(`[...document.querySelectorAll(".file-row")].some((row) => row.textContent.includes("bytes, uploaded"))`), "a file dropped in is uploaded into the scenario's inputs");
  } else {
    await page.click("Add a text file");
    await page.type(`document.querySelector('[name="file-0"]')`, "data.csv");
    await page.type(`document.querySelector('[name="content-0"]')`, "id,amount\n1,10\n2,20\n3,30\n4,1.5\n");
  }
  await page.click("Save and run");
  await until(() => page.eval(`location.hash.startsWith("#/batch?scenario=")`), "the batch form for the scenario");
  await page.eval(`location.hash = "#/batch?scenario=${scenarioName}&harness=fake&model=m"`);
  await until(() => page.eval(`document.querySelector('.plan-form [name="scenario"]')?.value === ${JSON.stringify(scenarioName)}`), "the scenario chosen");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "1");
  await planShows(page, "tally-scenario/1");
  await startButtonReady(page, "Start the runs");
  await page.click("Start the runs");
  const scenarioJob = await finished(await until(async () => jobIdIn(await page.eval("location.hash")), "the scenario batch's page"), 120000);
  check(scenarioJob.status === "done" && scenarioJob.result.rate.total === 1, "the scenario ran as a batch");
  await page.eval(`location.hash = "#/scenarios"`);
  await until(() => page.eval(`[...document.querySelectorAll(".scenario-result")].some((row) => row.textContent.includes("tally-scenario/1"))`), "the scenario's result in the list");
  check(true, "the list shows the scenario's pass rate");

  // The check, tried before trusting it: on the run the batch just made, then on files written here.
  await page.eval(`location.hash = "#/scenarios/${scenarioName}"`);
  await until(() => page.eval(`[...document.querySelectorAll("button")].some((b) => b.textContent === "Test this check…")`), "the check tester's button");
  await page.click("Test this check…");
  await until(() => page.eval(`document.querySelectorAll('[name="trial_run"] option').length > 1`), "the task's runs");
  await page.eval(`(() => { const s = document.querySelector('[name="trial_run"]'); s.value = s.options[1].value; s.dispatchEvent(new Event("change", { bubbles: true })); })()`);
  await until(() => page.eval(`(document.querySelector(".check-tester .plan-preview")?.textContent || "").includes("in a copy of the workspace")`), "the trial's plan");
  await startButtonReady(page, "Try it");
  await page.click("Try it");
  await until(() => page.eval(`!!document.querySelector(".check-verdict")`), "the verdict", 60000);
  check(await page.eval(`document.querySelector(".check-verdict").textContent.includes("as when the run ran")`),
    "tried on the run it judged, the check gives the verdict the run got");
  await page.eval(`document.querySelector('.check-tester input[name="trial_against"][value="sample"]').click()`);
  await page.eval(`[...document.querySelectorAll(".check-tester button")].find((b) => b.textContent === "Add a file").click()`);
  await page.type(`[...document.querySelectorAll('.check-tester [name^="trial_path_"]')].at(-1)`, "report.json");
  await page.type(`[...document.querySelectorAll('.check-tester [name^="trial_content_"]')].at(-1)`, '{"row_count": 4, "amount_total": 61.5}');
  await until(() => page.eval(`(document.querySelector(".check-tester .plan-preview")?.textContent || "").includes("file(s) you wrote")`), "the sample's plan");
  await startButtonReady(page, "Try it");
  await page.click("Try it");
  await until(() => page.eval(`document.querySelector(".check-verdict .chip")?.textContent === "pass"`), "a pass on the report written here", 60000);
  check(true, "on a report written here with the right keys, the check passes");

  // --- 7. Tools at a held call: sample it, ask the model, then let it go -------------------
  console.log("tools at a held call");
  const armed = await api("/breakpoints", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ phase: "response", tool: "write_file", label: "hold tools" }),
  });
  const seeded = await api(`/runs/${run.id}/steps/1`);
  const heldCall = fetch(`${BASE}/v1/chat/completions`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "gemma-4-31b", messages: seeded.request.messages, tools: seeded.request.tools }),
  });
  const [heldHold] = await until(async () => (await api("/holds")).length && api("/holds"), "the call to be held");
  await page.eval(`location.hash = "#/live/${heldHold.id}"`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel-head h2")].some((h) => h.textContent === "Evidence at this call")`), "the hold tools");
  await page.type(`document.querySelector('.hold-tools .evidence [name="n"]')`, "4");
  await until(() => page.eval(`(document.querySelector(".hold-tools .evidence .plan-preview")?.textContent || "").includes("4 times")`), "the sampling plan");
  await until(() => page.eval(`[...document.querySelectorAll(".hold-tools .evidence button")].some((b) => b.textContent === "Sample it" && !b.disabled)`), "Sample it enabled");
  await page.click("Sample it", `document.querySelector(".hold-tools .evidence")`);
  await until(() => page.eval(`(document.querySelector(".hold-tools .evidence")?.textContent || "").includes("The answer held here is behaviour A")`), "the samples, with the held answer placed", 60000);
  check(true, "sampling at a held call places the held answer among the sampled behaviours");
  await page.type(`document.querySelector('.hold-tools .hypotheses [name="question"]')`, "Why did you choose those keys?");
  await until(() => page.eval(`[...document.querySelectorAll(".hold-tools .hypotheses button")].some((b) => b.textContent === "Ask" && !b.disabled)`), "Ask enabled");
  await page.click("Ask", `document.querySelector(".hold-tools .hypotheses")`);
  await until(() => page.eval(`[...document.querySelectorAll(".hold-tools .reply")].some((r) => r.textContent.includes("hypothesis") && r.textContent.includes("never names the keys"))`), "the reply, marked a hypothesis", 60000);
  check(true, "the model's answer is shown as a hypothesis");
  const sampledAt = (await api("/experiments")).find((e) => e.hold_id === heldHold.id);
  check(sampledAt && sampledAt.run_id === null, "the samples wait for the step the call becomes");
  await page.click("Continue");
  await heldCall;
  await until(async () => (await api(`/experiments/${sampledAt.id}`)).run_id, "the samples filed under the step");
  check(true, "once recorded, the call's step holds the samples as evidence");
  await api(`/breakpoints/${armed.id}`, { method: "DELETE" });

  // --- 8. Connections: check the server, refuse a protected endpoint, switch while idle ------
  console.log("connections");
  await page.eval(`location.hash = "#/connections"`);
  await until(() => page.eval(`document.querySelectorAll(".harness-card").length >= 2`), "the harness cards");
  const startedOn = (await api("/connections")).server.url;
  await page.click("Check the server");
  await until(() => page.eval(`[...document.querySelectorAll("table td.mono")].some((td) => td.textContent === "anthropic messages")`), "the check's results", 60000);
  const checked = (await api("/connections")).server.checks.results;
  check(checked.streaming_tool_calls.support === "yes" && checked.anthropic_messages.support === "yes", "the server check measured both, and the page shows them");
  await page.eval(`document.querySelector("details.fold > summary")?.parentElement && [...document.querySelectorAll("details.fold > summary")].find((s) => s.textContent.startsWith("Point the proxy")).click()`);
  await page.type(`document.querySelector('[name="url"]')`, "http://127.0.0.1:18099");
  await until(() => page.eval(`(document.querySelector(".form-error-box")?.textContent || "").includes("protected")`), "the protected endpoint refused", 10000);
  check(true, "a protected endpoint is refused before anything is asked of it");
  await page.type(`document.querySelector('[name="url"]')`, "http://mock2.invalid");
  await until(() => page.eval(`[...document.querySelectorAll("button")].some((b) => b.textContent === "Switch" && !b.disabled)`), "the switch to be allowed");
  await page.click("Switch");
  await until(async () => (await api("/connections")).server.url === "http://mock2.invalid", "the switch");
  check(true, "an idle proxy switches its server after a dry run");
  await api("/connections/upstream", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: startedOn }) });

  // --- 9. Start… on the Jobs page: the way to a step's form, from a list the proxy gives ----
  console.log("start work from the Jobs page");
  await page.eval(`location.hash = "#/jobs"`);
  await until(() => page.eval(`[...document.querySelectorAll("button")].some((b) => b.textContent === "Start…")`), "the Start button");
  await page.click("Start…");
  await until(() => page.eval(`!!document.querySelector('.launcher [name="launch_run"]')`), "the launcher");
  const kindsOffered = await page.eval(`[...document.querySelectorAll('[name="launch_kind"] option')].map((o) => o.value).join(",")`);
  check(kindsOffered === "resample,tokens,forking-tokens,ab,resample-to-end,ab-to-end,fork", `the launcher offers the kinds worked at a step (${kindsOffered})`);
  const choose = (name, value) => page.eval(`(() => { const s = document.querySelector('[name="${name}"]'); s.value = ${JSON.stringify(value)}; s.dispatchEvent(new Event("change")); })()`);
  await choose("launch_kind", "resample");
  await choose("launch_run", run.id);
  await until(() => page.eval(`!document.querySelector('[name="launch_step"]').disabled`), "the run's steps");
  await choose("launch_step", "1");
  await page.click("Open it");
  await until(() => page.eval(`location.hash === "#/runs/${run.id}/1?start=resample" && !!document.querySelector(".start-panel .plan-form")`), "the step's resample form");
  check(true, "Start… leads to the step's own resample form");

  // --- 10. A run as one conversation ------------------------------------------------------
  console.log("a run as one conversation");
  const shape = await api(`/runs/${run.id}/timeline`);
  await page.eval(`location.hash = "#/runs/${run.id}"`);
  await until(() => page.eval(`document.querySelectorAll(".tl-step").length === ${shape.steps.length}`), "the timeline's steps");
  const shown = await page.eval(`document.querySelectorAll(".tl-entry").length`);
  check(shown === shape.entries.filter((entry) => entry.origin !== "copy").length,
    `each message once, a step's own answer in place of the harness's copy (${shown} messages, ${shape.steps.length} steps)`);
  check(await page.eval(`!!document.querySelector(".tl-final .tl-final-label")`), "the final reply is marked at the end");
  // Then its verdict, coloured by it: green and red are the check's alone, never the reply's.
  const said = shape.run.result.passed ? "passed" : "failed";
  check(shape.run.result.usable && await page.eval(`(() => {
    const last = document.querySelector(".tl").lastElementChild;
    const probe = document.body.appendChild(document.createElement("span"));
    probe.style.color = "var(--ok)";
    const ok = getComputedStyle(probe).color;
    probe.remove();
    return last.classList.contains("tl-verdict") && last.classList.contains("${said === "passed" ? "tone-ok" : "tone-bad"}")
      && last.querySelector(".verdict-big").textContent === "The check ${said}"
      && last.textContent.includes(${JSON.stringify((shape.run.result.check_output || "").trim())})
      && document.querySelector(".tl-rail-verdict").textContent === "Verdict: ${said}"
      && getComputedStyle(document.querySelector(".tl-final")).borderTopColor !== ok;
  })()`), `the run ends with its verdict — ${said}, what the check printed — and so does the rail; the reply is not green`);
  // The tools its first call offered come first, above the system prompt: no message holds
  // them, but the model reads them.
  const offered = shape.tools_offered.map((tool) => tool.name);
  check(offered.length > 0 && await page.eval(`(() => {
    const first = document.querySelector(".tl").firstElementChild;
    const summary = first.querySelector("details.tools-offered > summary");
    return first.classList.contains("tl-tools") && summary.textContent.startsWith("${offered.length} tools offered")
      && first.nextElementSibling.classList.contains("tl-entry")
      && ${shape.entries[0].message.role === "system"} === !!first.nextElementSibling.querySelector(".msg-system");
  })()`), `the tools the first call offered come first, before the system prompt (${offered.join(", ")})`);
  await page.eval(`document.querySelector(".tl-tools details.tools-offered").open = true`);
  check(await page.eval(`[...document.querySelectorAll(".tl-tools .tool .tool-name")].map((n) => n.textContent).join(",")`) === offered.join(","),
    "opened, each tool on a line of its own");
  // Each message the harness added after a step has a line in the rail, under that step.
  const added = shape.entries.filter((entry) => entry.origin === "harness" && entry.after !== null);
  const result = added.find((entry) => entry.message.role === "tool");
  const called = shape.steps.flatMap((step) => step.output?.tool_calls || []).find((call) => call.id === result.message.tool_call_id);
  check(added.length > 0 && await page.eval(`document.querySelectorAll(".tl-rail-result").length`) === added.length,
    `each message the harness added has its line in the rail (${added.length})`);
  check(await page.eval(`(() => {
    const line = document.querySelector('.tl-rail-result[data-entry="${result.index}"]');
    let above = line.previousElementSibling;
    while (above && above.classList.contains("tl-rail-result")) above = above.previousElementSibling;
    return above?.dataset.idx === "${result.after}" && line.textContent === "↳${called.function.name} result";
  })()`), `a tool's result named for its tool, under the step that called it (${called.function.name}, step ${result.after})`);
  await page.eval(`document.querySelector("#tm${result.index}").open = false`);
  await page.eval(`document.querySelector('.tl-rail-result[data-entry="${result.index}"]').click()`);
  await until(() => page.eval(`document.querySelector("#tm${result.index}").open`), "the result opened from the rail");
  check(await page.eval(`(() => { const top = document.querySelector("#tm${result.index}").getBoundingClientRect().top; return top >= 0 && top < innerHeight / 2; })()`),
    "its line leads to it in the conversation, opened");
  // Opening a step happens in place: the page is not drawn again, and the address follows.
  await page.eval(`document.querySelector(".view").dataset.sentinel = "kept"`);
  await page.eval(`document.querySelector("#s1 .tl-marker").click()`);
  await until(() => page.eval(`!!document.querySelector("#s1.tl-open .start-actions")`), "step 1's details");
  check(await page.eval(`location.hash === "#/runs/${run.id}/1" && document.querySelector(".view").dataset.sentinel === "kept"`),
    "a step opens in place, and the address names it");
  await page.eval(`[...document.querySelectorAll("#s1 .start-actions a")].find((a) => a.textContent === "Resample…").click()`);
  await until(() => page.eval(`!!document.querySelector("#s1 .start-panel .plan-form")`), "the form in the step");
  check(await page.eval(`location.hash === "#/runs/${run.id}/1?start=resample" && document.querySelector(".view").dataset.sentinel === "kept"`),
    "a form opens inside the step, without leaving the conversation");
  await page.eval(`[...document.querySelectorAll("#s1 a.btn")].find((a) => a.textContent.startsWith("Exact context")).click()`);
  await until(() => page.eval(`(document.querySelector(".page-head h1")?.textContent || "").startsWith("Step 1")`), "the exact context");
  check(await page.eval(`document.querySelector(".page-head h1").textContent.includes("exact context") && document.querySelectorAll(".msg").length >= 2`),
    "the exact context a step was sent has a page of its own");

  // --- 10b. Where a task's runs split: each run by its verdict; the checks' lines counted --
  console.log("the divergence page: each run by its verdict, the checks' lines counted");
  const split = await api(`/divergence?task=${encodeURIComponent("tally-report/1")}&limit=50`);
  await page.eval(`location.hash = "#/divergence/${encodeURIComponent("tally-report/1")}"`);
  await until(() => page.eval(`!!document.querySelector(".run-chips a.chip")`), "the behaviours at the split");
  const tones = { passed: "tone-ok", failed: "tone-bad", "not counted": "tone-warn" };
  const chips = await page.eval(`[...document.querySelectorAll(".run-chips a.chip")].map((chip) => [chip.getAttribute("href"), chip.className])`);
  const wrong = chips.filter(([href, className]) => {
    const id = href.split("/")[2];
    return !className.split(" ").includes(tones[split.verdicts[id]] || "tone-muted");
  });
  check(chips.length > 1 && !wrong.length && chips.some(([, className]) => className.includes("tone-ok")) && chips.some(([, className]) => className.includes("tone-bad")),
    `each run's chip is coloured by its own verdict (${chips.length} runs${wrong.length ? `; wrong: ${JSON.stringify(wrong)}` : ""})`);
  check(await page.eval(`!document.querySelector(".group-rates .ci-track.tone-ok")`), "and a behaviour's \"then passed\" is a rate, never drawn green");
  const tallied = `[...document.querySelectorAll(".panel")].find((panel) => panel.querySelector("h2")?.textContent === "What the failed runs' checks said")`;
  check(await page.eval(`(() => {
    const panel = ${tallied};
    return !!panel && panel.querySelector("th").textContent === "failed runs that printed it"
      && panel.querySelector(".hint").textContent.startsWith("Each bar counts the failed runs that printed the line");
  })()`), "the failed runs' checks are counted line by line, and the page says so plainly");

  // --- 11. A step's answer as tokens, and the token that decides it -------------------------
  console.log("score an answer, read its tokens, find the token that decides it");
  const scored = (await api("/runs?limit=100")).find((found) => found.label === "tokens-lab");
  check(!!scored, "the lab has a run the toy model can score");
  await page.eval(`location.hash = "#/runs/${scored.id}/0?view=tokens"`);
  await until(() => page.eval(`[...document.querySelectorAll("button")].some((b) => b.textContent === "Score this answer")`), "the offer to score the answer");
  await page.click("Score this answer");
  await until(() => page.eval(`document.querySelectorAll(".tk-body .tk-t").length > 0`), "the heatmap", 60000);
  const scoredView = (await api(`/runs/${scored.id}/steps/0/tokens`)).view;
  check(scoredView.stream.exact === true, "the answer was rebuilt token for token");
  const drawnTokens = await page.eval(`document.querySelectorAll(".tk-body .tk-t").length`);
  const expectedTokens = scoredView.tokens.filter((token) => token.part !== "marker").length;
  check(drawnTokens === expectedTokens, `every token drawn once (${drawnTokens} of ${expectedTokens})`);
  const forkSpot = scoredView.hotspots[0];
  check(forkSpot?.chosen.text === "read_file" && forkSpot.alternatives[0].text === "search_files", "the likeliest fork is the tool's name");
  await page.eval(`document.querySelector('.tk-t[data-i="${forkSpot.position}"]').click()`);
  await until(() => page.eval(`(document.querySelector(".tk-detail")?.textContent || "").includes("search_files")`), "the token's alternatives");
  await page.click("Select to probe", `document.querySelector(".tk-detail")`);
  await until(() => page.eval(`!!document.querySelector(".tk-bar button")`), "the selection bar");
  await page.click("Find forking tokens at these", `document.querySelector(".tk-bar")`);
  await until(() => page.eval(`!!document.querySelector('.start-panel .plan-form [name="positions"]')`), "the forking form, in the step");
  check(await page.eval(`document.querySelector('.plan-form [name="positions"]').value === "${forkSpot.position}"`), "the form holds the selected position");
  check(await page.eval(`(() => { const tools = document.querySelector('.plan-form [name="target_kind"][value="tools"]');
    return !!tools?.checked && tools.closest("label").textContent.startsWith("Calls read_file"); })()`),
    "the answer calls a tool, so the target first offered is calling it, with any arguments");
  await page.type(`document.querySelector('.plan-form [name="n"]')`, "6");
  await planShows(page, "2 forced tokens × 6 = 12 continuations");
  await startButtonReady(page, "Find the forking tokens");
  await page.click("Find the forking tokens");
  const forkJobId = await until(async () => jobIdIn(await page.eval("location.hash")), "the probe's job page");
  const forkJob = await finished(forkJobId, 120000);
  check(forkJob.status === "done" && forkJob.result.forking.flips === 1, "the probe found the token that decides the answer");
  await until(() => page.eval(`[...document.querySelectorAll(".chip")].some((c) => c.textContent === "flips away")`), "the result, marked where it forks");
  check(await page.eval(`[...document.querySelectorAll(".chip")].filter((c) => c.textContent.startsWith("flips")).every((c) => c.classList.contains("tone-accent"))`),
    "a flip is the accent's colour, its direction in words: a result, not a check's red");

  console.log("from a probe's result to the token it forced, and back");
  const probeId = forkJob.result.experiment.id;
  const at = forkSpot.position;
  await page.eval(`location.hash = "#/experiments/${probeId}"`);
  await until(() => page.eval(`!!document.querySelector('.fork-answer .tk-t.tk-flipped[data-i="${at}"]')`), "the answer it forked from");
  check(true, "the result shows the answer it forked from, the forking token ringed");
  check(await page.eval(`(() => {
    const rows = [...document.querySelectorAll(".probe-behaviours tbody tr")];
    return rows.length >= 2 && rows.some((row) => row.querySelector(".chip.tone-target")) && rows.some((row) => row.querySelector(".chip.tone-not-target"))
      && rows.every((row) => row.querySelectorAll(".led-by button.chip").length >= 1);
  })()`), "each behaviour says whether the target counts it, and which tokens' continuations reached it");
  check(await page.eval(`[...document.querySelectorAll("#fork-pos-${at} thead th")].pop().textContent === "led to"`),
    "and each token's row says where its continuations went");
  await page.click("In the answer", `document.querySelector("#fork-pos-${at}")`);
  await until(() => page.eval(`(document.querySelector(".fork-answer .tk-detail")?.textContent || "").includes("Probed here")`), "the token, opened in the answer");
  check(true, "a position opens its token in the answer, with what the probe found there");
  await page.eval(`[...document.querySelectorAll("#fork-pos-${at} a")].find((a) => a.textContent === "On the step").click()`);
  await until(() => page.eval(`!!document.querySelector('.tk-host .tk-t.tk-current[data-i="${at}"]')`), "the step's tokens, open on that token");
  check(await page.eval(`(document.querySelector(".tk-host .tk-detail")?.textContent || "").includes("probe ${probeId.slice(0, 8)}")`),
    "the step's Tokens view opens on the token, and names the probe");
  await page.eval(`[...document.querySelectorAll(".tk-host .tk-detail a")].find((a) => a.textContent.startsWith("probe")).click()`);
  await until(() => page.eval(`location.hash.startsWith("#/experiments/${probeId}") && !!document.querySelector("#fork-pos-${at}")`), "the result again");
  check(true, "and the token leads back to the result");

  console.log("a target described, set against what the step has done");
  await page.eval(`location.hash = "#/runs/${scored.id}/0?start=forking-tokens"`);
  await until(() => page.eval(`!!document.querySelector('.plan-form [name="target_kind"][value="describe"]')`), "the forking form");
  await until(() => page.eval(`(document.querySelector(".seen-preview")?.textContent || "").includes("samples")`), "the behaviours seen at this step");
  await page.eval(`(() => {
    const value = document.querySelector(".rule-builder .rule input");
    value.focus();
    value.value = "search_files";
    value.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);
  await until(() => page.eval(`[...document.querySelectorAll(".seen-yes")].some((li) => li.textContent.includes("search_files"))`), "what the described target counts");
  check(await page.eval(`[...document.querySelectorAll(".seen-no")].some((li) => li.textContent.includes("read_file"))`),
    "a described target is set against what the step has done: search_files counts, read_file does not");
  await planShows(page, "a behaviour that calls search_files");
  check(true, "and the plan declares it in words");

  // --- 12. Refine the skill at a step: the proposer, a person's own wording, a screen --------
  console.log("refine a skill: proposals from the proposer and from a person, screened");
  await openStepOf(page, run.id, 1, "refine");
  check(await page.eval(`!document.querySelector('.start-panel [name="target_kind"][value="exploratory"]')`), "a refinement always has a target: there is no exploratory choice");
  await page.eval(`(() => {
    const value = document.querySelector(".start-panel .rule-builder .rule input");
    value.focus();
    value.value = "write_file";
    value.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  await page.click("+ Add a condition", `document.querySelector(".start-panel")`);
  await page.eval(`(() => {
    const values = document.querySelectorAll(".start-panel .rule-builder .rule input");
    const value = values[values.length - 1];
    value.value = "row_count";
    value.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  check(await page.eval(`document.querySelector('.start-panel [name="target_kind"][value="describe"]').checked`), "describing the behaviour chooses that way of declaring it");
  await page.click("Start refining", `document.querySelector(".start-panel")`);
  const refinementId = await until(async () => /^#\/refine\/([0-9a-f]{32})$/.exec(await page.eval("location.hash"))?.[1], "the refinement's page");
  await until(() => page.eval(`[...document.querySelectorAll(".packet-panel dt")].some((dt) => dt.textContent === "the original wording")`), "what a proposer is shown");
  check(await page.eval(`!document.querySelector(".evidence-panel")`), "no evidence panel before there is a proposal to weigh");
  check(await page.eval(`document.querySelector(".page-head").textContent.includes('a behaviour that calls write_file, and mentions "row_count"')`), "the refinement states its target, declared before anything was proposed");
  await until(() => page.eval(`(document.querySelector(".propose-builtin .plan-preview")?.textContent || "").includes("through this proxy")`), "the proposer's plan");
  check(true, "the proposer's plan says where the evidence goes: through this proxy, never recorded");
  await startButtonReady(page, "Ask for proposals");
  const runsBefore = (await api("/runs?limit=500")).length;
  await page.click("Ask for proposals");
  const proposeJobId = await until(async () => jobIdIn(await page.eval("location.hash")), "the proposer's job page");
  const proposeJob = await finished(proposeJobId);
  check(proposeJob.status === "done" && proposeJob.result.accepted.length === 2 && proposeJob.result.rejected.length === 1,
    "the proposer's three ideas: two kept, and one refused for editing a line the skill does not have");
  check((await api("/runs?limit=500")).length === runsBefore, "the proposer's call is not a run: it is never recorded");
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "See them in the refinement")`), "the way back to the refinement");
  check(await page.eval(`document.querySelector("main").textContent.includes("Not kept")`), "the job's page says what was not kept, and why");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "See them in the refinement").click()`);
  await until(() => page.eval(`document.querySelectorAll(".proposal").length === 2`), "the two proposals");
  check(await page.eval(`[...document.querySelectorAll(".proposal")].every((card) => card.textContent.includes("hypothesis: not screened yet"))`), "each a hypothesis until a screen tests it");
  check(await page.eval(`[...document.querySelectorAll(".proposal .diff-line.add")].some((line) => line.textContent.includes("row_count and amount_total"))`), "each shown as a line diff");
  await page.eval(`document.querySelector(".propose-own > summary").click()`);
  const baseline = await page.eval(`document.querySelector('[name="proposal_text"]').value`);
  await page.type(`document.querySelector('[name="proposal_text"]')`, baseline.replace("Write the totals to report.json.", "Write the totals to report.json, as JSON."));
  check((await page.text(".own-proposal .editor-head .muted")).startsWith("1 line changed"), "a person's own wording: the editor counts the changed line");
  await page.type(`document.querySelector('[name="proposal_why"]')`, "JSON is what the check parses.");
  await page.click("Add the proposal");
  await until(() => page.eval(`document.querySelectorAll(".proposal").length === 3`), "the person's proposal");
  check(await page.eval(`[...document.querySelectorAll(".proposal .chip")].some((c) => c.textContent === "a person")`), "a person's proposal says who made it");
  await page.eval(`(() => {
    for (const card of document.querySelectorAll(".proposal")) {
      const added = [...card.querySelectorAll(".diff-line.add")].map((line) => line.textContent).join(" ");
      if (/row_count and amount_total|as JSON/.test(added)) card.querySelector('input[type="checkbox"]').click();
    }
    return true;
  })()`);
  await page.type(`document.querySelector('.screen-panel [name="n"]')`, "6");
  await until(() => page.eval(`(document.querySelector(".screen-panel .plan-preview")?.textContent || "").includes("6 an arm, 18 in all")`), "the screen's plan");
  check(await page.eval(`document.querySelector(".screen-panel .plan-preview").textContent.includes("Holm over 2 proposals")`), "the screen's plan: the two ticked are the family the correction counts");
  await startButtonReady(page, "Start the screen");
  await page.click("Start the screen", `document.querySelector(".screen-panel")`);
  const screenJobId = await until(async () => jobIdIn(await page.eval("location.hash")), "the screen's job page");
  const screenJob = await finished(screenJobId, 60000);
  check(screenJob.status === "done" && screenJob.result.screen.complete && screenJob.result.screen.survivors.length === 1,
    "the screen: the wording that names the keys survives the correction, the other does not");
  await until(() => page.eval(`(document.querySelector(".verdict-big")?.textContent || "") === "1 proposal hit the target more often than the original, past the correction"`), "the verdict on the job's page");
  // At the step, a survivor is a shift of a behaviour, not a check passed: the accent's, never green.
  check(await page.eval(`!document.querySelector(".verdict-block.tone-ok") && [...document.querySelectorAll(".screen-table .chip")]
    .find((chip) => chip.textContent === "survives")?.classList.contains("tone-accent")`), "a step screen's survivor is the accent's colour, not a check's green");
  // What the model did, by arm: the right keys are the target ("calls write_file and mentions row_count").
  const byArm = await page.eval(`(() => {
    const panel = [...document.querySelectorAll(".panel")].find((found) => found.querySelector("h2")?.textContent === "What the model did, by arm");
    return [...panel.querySelectorAll("tbody tr")].map((row) =>
      [row.querySelector(".chip.tone-target, .chip.tone-not-target")?.textContent, row.children[1].textContent.includes("row_count")].join(":")).join(",");
  })()`);
  check(byArm.split(",").sort().join(",") === "not target:false,target:true", `the screen's behaviours are badged target or not target (${byArm})`);
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "See the result").click()`);
  await until(() => page.eval(`location.hash.startsWith("#/experiments/") && !!document.querySelector(".screen-table")`), "the screen's own page");
  check(await page.eval(`[...document.querySelectorAll(".screen-table .chip")].map((c) => c.textContent).sort().join(",") === "not significant,survives"`), "the screen's page gives each proposal its verdict");
  check(await page.eval(`document.querySelector(".page-head").textContent.includes("refinement ${refinementId.slice(0, 8)}")`), "and names the refinement it belongs to");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "Back to the refinement").click()`);
  await until(() => page.eval(`location.hash === "#/refine/${refinementId}" && [...document.querySelectorAll(".proposal .chip")].some((c) => c.textContent === "survived a screen")`), "the verdicts on the refinement's proposals");
  check(await page.eval(`[...document.querySelectorAll(".proposal .chip")].filter((c) => c.textContent === "hypothesis: not screened yet").length === 1`), "the screened carry their verdicts; the one not screened is still a hypothesis");
  // The page leads with the evidence: where it stands, each proposal's screen, a step screen's survivor in the accent.
  const evidence = await page.eval(`(() => {
    const titles = [...document.querySelectorAll(".panel-head h2")].map((h2) => h2.textContent);
    const block = document.querySelector(".evidence-panel .verdict-block");
    const rows = [...document.querySelectorAll(".evidence-panel tbody tr")];
    return {
      order: titles.indexOf("The evidence") < titles.indexOf("Proposals (3)") && titles.indexOf("Proposals (3)") < titles.indexOf("What a proposer is shown"),
      headline: block?.querySelector(".verdict-big")?.textContent,
      accent: block?.classList.contains("significant") && !block.classList.contains("tone-ok"),
      survivor: rows.some((row) => row.querySelector(".chip.tone-accent")?.textContent === "survived a screen"),
      unscreened: rows.filter((row) => row.textContent.includes("hypothesis: not screened yet")).length,
      chosen: rows.some((row) => row.classList.contains("is-chosen")),
      about: !document.querySelector("details.about-refinement").open,
    };
  })()`);
  check(evidence.order && evidence.headline === "1 of 3 proposals survived a screen" && evidence.accent,
    `the refinement leads with the evidence: where it stands, in the accent for a step screen (${evidence.headline})`);
  check(evidence.survivor && evidence.unscreened === 1 && !evidence.chosen, "each proposal's screen in a line, nothing chosen before a validation");
  check(evidence.about, "the skill's paths wait in About this refinement, closed");
  await until(() => page.eval(`(document.querySelector(".packet-panel")?.textContent || "").includes("the original wording")`),
    "what a proposer is shown, below them: the original wording's answers, the problem and not the result");

  // --- 12b. Validate a survivor with full runs, held-out task included, and approve it ------
  console.log("validate a survivor with full runs, and approve it");
  const labRun = batch.links.runs[0];
  const made = await post("/refinements", {
    run: labRun, step: "1", target_rules: [{ if: "calls", value: "write_file" }, { if: "contains", value: "row_count" }],
  });
  const keysFix = await post(`/refinements/${made.id}/proposals`, {
    edits: [{ old: "Write the totals to report.json.", new: "Write the totals to report.json, using the keys row_count and amount_total." }],
    rationale: "the check reads row_count",
  });
  const labScreen = await finished((await post("/jobs", { kind: "screen", params: { refinement: made.id, proposals: [keysFix.id], n: 6 } })).id, 60000);
  check(labScreen.status === "done" && labScreen.result.screen.survivors.length === 1, "a refinement of the lab task: the keys fix survives its screen");
  await page.eval(`location.hash = "#/refine/${made.id}"`);
  await until(() => page.eval(`(document.querySelector(".validate-panel .plan-preview")?.textContent || "").includes("30 in all")`), "the validation's plan");
  check(await page.eval(`document.querySelector(".validate-panel .plan-preview").textContent.includes("each of: 2 — only their verdicts are kept")`),
    "the plan names the held-out task by its id, and keeps only its verdicts");
  check(!(await page.eval(`document.body.textContent.includes("HELD-OUT-LAB")`)), "and shows nothing else of it");
  await startButtonReady(page, "Start the validation");
  await page.click("Start the validation", `document.querySelector(".validate-panel")`);
  const validateJobId = await until(async () => jobIdIn(await page.eval("location.hash")), "the validation's job page");
  const validateJob = await finished(validateJobId, 240000);
  check(validateJob.status === "done" && validateJob.result.held_out.join() === "2" && validateJob.progress.done === 30,
    "the validation ran every run: forks, the training task, the held-out task");
  check(!JSON.stringify(validateJob).includes("HELD-OUT-LAB"), "its job holds no word of the held-out task");
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "The gate, in the refinement")`), "the way to the gate");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "The gate, in the refinement").click()`);
  await until(() => page.eval(`[...document.querySelectorAll(".validation .chip")].some((c) => c.textContent === "meets the gate")`), "the validation's verdict");
  check(await page.eval(`[...document.querySelectorAll(".validation-table td > strong")].map((s) => s.textContent).join() === "forward,training,held out 2"`),
    "the person's page shows each part, the held-out task's included");
  check(await page.eval(`[...document.querySelectorAll(".validation .gate-list .chip")].filter((c) => c.textContent === "low-powered").length === 2`),
    "the not-worse checks say they are low-powered");
  const ready = await page.eval(`(() => {
    const block = document.querySelector(".evidence-panel .verdict-block");
    const row = document.querySelector(".evidence-panel tr.is-chosen");
    return {
      headline: block?.querySelector(".verdict-big")?.textContent,
      green: block?.classList.contains("tone-ok"),
      lines: block?.querySelector(".verdict-lines")?.textContent || "",
      row: row ? [...row.querySelectorAll(".chip")].map((c) => c.textContent).join() : null,
    };
  })()`);
  check(ready.headline === `Proposal ${keysFix.id.slice(0, 8)} meets the gate: ready to approve` && ready.green,
    `a validation that meets the gate: the evidence names the proposal ready to approve, in the gate's green (${ready.headline})`);
  check(ready.lines.includes("Survived its screen: 6/6 against the original's 0/6") && ready.lines.includes("on held-out task 2") && ready.lines.includes("checks met"),
    `with its screen and its validation in a line each, the held-out task's included (${ready.lines})`);
  check(ready.row === "ready to approve,survived a screen,meets the gate", `its row picked out (${ready.row})`);
  const hash = await page.eval("location.hash");
  await page.eval(`[...document.querySelectorAll(".evidence-panel .verdict-lines a")].find((a) => a.textContent.startsWith("validation ")).click()`);
  check(await page.eval(`location.hash === ${JSON.stringify(hash)} && document.querySelector(".validation").getBoundingClientRect().top < innerHeight`),
    "its link leads down the page to the validation, the address as it was");
  await page.click("Approve…", `document.querySelector(".validation")`);
  await until(() => page.eval(`!!document.querySelector(".approve-code strong")`), "the approval's plan");
  const code = await page.eval(`document.querySelector(".approve-code strong").textContent`);
  check(await page.eval(`document.querySelector(".approve-panel").textContent.includes(${JSON.stringify(`${health.home}/lab-task/tally-report/SKILL.md`)})`),
    "approving shows the file it writes");
  check(await page.eval(`[...document.querySelectorAll(".approve-panel .diff-line.add")].some((line) => line.textContent.includes("row_count and amount_total"))`),
    "and the edit, as a line diff");
  await page.type(`document.querySelector('.approve-panel [name="confirm"]')`, "00000000");
  await page.click("Write it", `document.querySelector(".approve-panel")`);
  await until(() => page.eval(`(document.querySelector(".approve-panel .form-error-box")?.textContent || "").includes("not the code")`), "the wrong code, refused");
  check((await api(`/refinements/${made.id}`)).approval === null, "a wrong code writes nothing");
  await page.type(`document.querySelector('.approve-panel [name="confirm"]')`, code);
  await page.click("Write it", `document.querySelector(".approve-panel")`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "Written")`), "the edit, written");
  const approved = await api(`/refinements/${made.id}`);
  check(approved.approval?.state === "applied" && approved.refinement.status === "approved",
    "the typed code writes it: the skill's own file, its old text and a changelog beside its folder");
  const writtenView = await page.eval(`(() => {
    const block = document.querySelector(".evidence-panel .verdict-block");
    return {
      headline: block?.querySelector(".verdict-big")?.textContent,
      green: block?.classList.contains("tone-ok"),
      row: [...(document.querySelector(".evidence-panel tr.is-chosen")?.querySelectorAll(".chip") || [])].map((c) => c.textContent).join(),
      card: !!document.querySelector(".proposal.chosen") && [...document.querySelectorAll(".proposal.chosen .chip")].some((c) => c.textContent === "approved"),
      about: document.querySelector("details.about-refinement").textContent,
    };
  })()`);
  check(writtenView.headline === `Approved: proposal ${keysFix.id.slice(0, 8)}, written to the skill's own file` && writtenView.green,
    `and the page says so first, with the evidence it stood on (${writtenView.headline})`);
  check(writtenView.row === "approved,survived a screen,met the gate" && writtenView.card, `the approved proposal picked out, in the evidence and its card (${writtenView.row})`);
  check(writtenView.about.includes(`${health.home}/lab-task/tally-report/SKILL.md`), "About this refinement says where it wrote");

  console.log("the proposer's key: set, never shown, cleared");
  await page.eval(`location.hash = "#/connections"`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "Proposer")`), "the Proposer panel");
  const secret = `sk-browser-${Date.now().toString(36)}-0123456789abcdef`;
  await page.eval(`[...document.querySelectorAll("details.fold > summary")].find((s) => s.textContent === "Set a key").click()`);
  await page.type(`document.querySelector('[name="proposer_key"]')`, secret);
  await page.click("Save the key");
  await until(() => page.eval(`[...document.querySelectorAll(".chip")].some((c) => c.textContent === "set, in the recordings folder")`), "the key, set");
  check(!(await page.eval(`document.body.innerText.includes(${JSON.stringify(secret)}) || [...document.querySelectorAll("input")].some((i) => i.value.includes(${JSON.stringify(secret)}))`)),
    "once saved, the key is nowhere on the page");
  check(!JSON.stringify(await api("/proposer")).includes(secret), "and the proxy never gives it back");
  await page.eval(`[...document.querySelectorAll("details.fold > summary")].find((s) => s.textContent === "Replace or clear the key").click()`);
  await page.click("Clear the key");
  await until(() => page.eval(`[...document.querySelectorAll(".chip")].some((c) => c.textContent === "not set")`), "the key, cleared");
  check(true, "and it can be cleared");

  console.log("the trash: set aside, back as it went, deleted for good");
  const victim = launched; // last: deleting it for good takes its forks and experiments too
  await page.eval(`location.hash = "#/runs/${victim}"`);
  await until(() => page.eval(`[...document.querySelectorAll(".page-head button")].some((b) => b.textContent === "Move to trash")`), "the run's way to the trash");
  await page.click("Move to trash", `document.querySelector(".page-head")`);
  await until(() => page.eval(`!!document.querySelector(".trashed-banner")`), "the banner of a run in the trash");
  check(!(await api("/runs?limit=500")).some((found) => found.id === victim), "a run in the trash leaves the Runs list; its page still opens, saying so");
  await page.eval(`location.hash = "#/trash"`);
  await until(() => page.eval(`(document.querySelector("main")?.textContent || "").includes(${JSON.stringify(victim.slice(0, 8))})`), "the run on the Trash page");
  check(await page.eval(`document.querySelector("main").textContent.includes("deleting it for good removes")`), "the Trash page says what deleting each for good removes");
  await page.click("Restore", `document.querySelector("main")`);
  await until(async () => (await api("/runs?limit=500")).some((found) => found.id === victim), "the run back in the list");
  check(true, "restored, the run is back in the Runs list");
  await post("/trash", { kind: "run", id: victim });
  await page.eval(`location.hash = "#/runs/${victim}"`);
  await until(() => page.eval(`!!document.querySelector(".trashed-banner button")`), "the banner again");
  await page.click("Delete for good", `document.querySelector(".trashed-banner")`);
  await until(() => page.eval(`(document.querySelector(".trashed-banner")?.textContent || "").includes("cannot be undone")`), "the first click's warning");
  check(await page.eval(`/removes [0-9]+ runs?, [0-9]+ steps/.test(document.querySelector(".trashed-banner").textContent)`), "the first click says what goes, and that it cannot be undone");
  await page.click("Click again", `document.querySelector(".trashed-banner")`);
  await until(() => page.eval(`location.hash === "#/trash"`), "the Trash page, after");
  check(!!(await api(`/runs/${victim}`))?.error, "deleted for good, the run is gone");

  const done = (await api("/jobs")).filter((job) => !["queued", "running"].includes(job.status)).slice(0, 2).map((job) => job.id);
  await page.eval(`location.hash = "#/jobs"`);
  await until(() => page.eval(`document.querySelectorAll("td.pick input").length >= 2`), "the finished jobs' boxes");
  // Both boxes found first, then ticked, as a person sees them: found by a screenshot, a tick
  // used to draw the table again, and the second box clicked was no longer on the page.
  await page.eval(`(() => {
    const boxes = ${JSON.stringify(done)}.map((id) =>
      document.querySelector('tr[data-job="' + id + '"] td.pick input'));
    boxes.forEach((box) => box.click());
    return true;
  })()`);
  await until(() => page.eval(`[...document.querySelectorAll(".compare-bar button")].some((b) => b.textContent === "Move 2 to trash")`), "the bar to move two");
  await page.click("Move 2 to trash", `document.querySelector(".compare-bar")`);
  await until(async () => { const left = (await api("/jobs")).map((job) => job.id); return done.every((id) => !left.includes(id)); }, "the two jobs to leave the list");
  const inTrash = (await api("/trash")).jobs.map((job) => job.id);
  check(done.every((id) => inTrash.includes(id)), "finished jobs ticked on the Jobs page go to the trash together");

  // --- 13. Work an agent started over MCP says so ---------------------------------------------
  console.log("a job an agent started says who");
  const agentJob = await post("/jobs", {
    kind: "resample", params: { run: run.id, step: "1", n: 2 },
    started_by: { via: "mcp", client: "claude-code", version: "2.1.281" },
  });
  await finished(agentJob.id);
  const chipIn = (scope) => page.eval(`[...document.querySelectorAll(${JSON.stringify(scope)})].some((c) => c.textContent === "from MCP · claude-code")`);
  await page.eval(`location.hash = "#/jobs"`);
  check(await until(() => chipIn(`tr[data-job="${agentJob.id}"] .chip`), "the chip on the Jobs list"), "the Jobs list names the agent that started a job");
  await page.eval(`location.hash = "#/jobs/${agentJob.id}"`);
  check(await until(() => chipIn(".page-head .chip"), "the chip on the job's page"), "so does the job's page");

  // --- 14. A refinement to the trash, and back ------------------------------------------------
  console.log("a refinement goes to the trash from its page, and comes back");
  await page.eval(`location.hash = "#/refine/${refinementId}"`);
  await until(() => page.eval(`[...document.querySelectorAll(".page-head button")].some((b) => b.textContent === "Move to trash")`), "the refinement's way to the trash");
  await page.click("Move to trash", `document.querySelector(".page-head")`);
  await until(() => page.eval(`(document.querySelector(".trashed-banner")?.textContent || "").includes("takes no new work")`), "the banner of a refinement in the trash");
  check(await page.eval(`!document.querySelector(".page-head button") && ![...document.querySelectorAll("button")].some((b) => b.textContent === "Start the screen")`),
    "in the trash, it offers no new work");
  check(!(await api("/refinements")).some((each) => each.id === refinementId), "and leaves the list of refinements");
  await page.eval(`location.hash = "#/trash"`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent.startsWith("Refinements"))`), "the Trash page's refinements");
  check(await page.eval(`document.querySelector("main").textContent.includes("1 refinement")`), "deleting it for good would take the refinement with what it made");
  await page.eval(`location.hash = "#/refine/${refinementId}"`);
  await until(() => page.eval(`!!document.querySelector(".trashed-banner button")`), "the banner's buttons");
  await page.click("Restore", `document.querySelector(".trashed-banner")`);
  await until(() => page.eval(`!document.querySelector(".trashed-banner") && [...document.querySelectorAll(".page-head button")].some((b) => b.textContent === "Move to trash")`), "the refinement, restored");
  check((await api("/refinements")).some((each) => each.id === refinementId), "restored, it is back in the list as it went");

  // --- 15. The front page, and the Skills page ---------------------------------------------------
  console.log("the front page: setup, the map, ways in, words");
  await page.eval(`location.hash = "#/"`);
  await until(() => page.eval(`location.hash === "#/guide" && !!document.querySelector(".map svg")`), "the front page, at #/guide");
  check(await page.eval(`document.querySelectorAll(".map-node").length === 14 && document.querySelectorAll(".map-edge").length === 14`),
    "the map draws each piece, and what leads to what");
  check(await page.eval(`[...document.querySelectorAll(".map-node title")].every((title) => title.textContent.includes(": "))`),
    "each piece says what its word means");
  check(await page.eval(`document.querySelectorAll(".checklist li").length === 4 && !!document.querySelector(".left-off")`),
    "what is set up, and where you left off");
  check(await page.eval(`document.querySelectorAll(".workflow").length === 8 && [...document.querySelectorAll(".workflow .cmd span")].some((s) => s.textContent.startsWith("tuningfork forking-tokens"))`),
    "eight ways in, each a question, each with its terminal command");
  const words = (await api("/glossary")).terms.length;
  check(await page.eval(`document.querySelectorAll(".glossary .term-row").length === ${words}`), `the glossary, all ${words} words`);
  await page.type(`document.querySelector('.panel input[type="search"]')`, "forking");
  check(await page.eval(`[...document.querySelectorAll(".glossary .term-row strong")].map((s) => s.textContent).includes("forking token") && document.querySelectorAll(".glossary .term-row").length < ${words}`),
    "a word looked up narrows the table");
  await page.eval(`location.hash = "#/guide?term=validation"`);
  await until(() => page.eval(`document.querySelector('.panel input[type="search"]')?.value === "validation"`), "a word named in the address");

  console.log("the Skills page: versions, their rates, the text rendered, two texts compared");
  await page.eval(`location.hash = "#/skills"`);
  await until(() => page.eval(`[...document.querySelectorAll("main table a")].some((a) => a.textContent === "tally-report")`), "the skill in the list");
  await page.eval(`location.hash = "#/skills/tally-report"`);
  await until(() => page.eval(`!!document.querySelector(".markdown .md-front")`), "its SKILL.md, rendered with its frontmatter as fields");
  check(await page.eval(`!!document.querySelector(".markdown h1") && document.querySelector(".markdown").textContent.includes("report.json")`), "the text, as Markdown");
  const skill = await api("/skills/tally-report");
  check(skill.versions.length >= 2 && await page.eval(`document.querySelectorAll(".condition-rate .ci").length >= 2`),
    "each version with its pass rates, intervals drawn");
  check(skill.folders.some((folder) => folder.changelog && folder.history.length === 1), "the folder the approval wrote: its changelog and the old version beside it");
  check(await page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "Changelog")`), "and the page shows the changelog");
  const older = skill.versions[skill.versions.length - 1].fingerprint;
  await page.eval(`(() => {
    const against = document.querySelectorAll(".skill-controls select")[1];
    against.value = ${JSON.stringify(older)};
    against.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  await until(() => page.eval(`!!document.querySelector(".skill-controls + div .diff") || (document.querySelector(".skill-controls + div")?.textContent || "").includes("The same text")`), "two texts compared");
  await page.eval(`(() => {
    const against = document.querySelectorAll(".skill-controls select")[1];
    against.value = "";
    against.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  await until(() => page.eval(`!!document.querySelector(".skill-controls .toggle")`), "the text alone again");
  await page.eval(`[...document.querySelectorAll(".skill-controls .toggle button")].find((b) => b.textContent === "As written").click()`);
  await until(() => page.eval(`!!document.querySelector(".skill-controls + div .lines")`), "the text as written, numbered");
  check(true, "the text rendered, or as written with its line numbers");

  // --- 16. A chat had elsewhere, imported as a run ------------------------------------------------
  console.log("a Hermes export imported from the Runs page: its session chosen, its skill found, a run made");
  const skillText = "---\nname: tally-report\n---\n\n# Tally report\n\nWrite the totals to report.json.\n";
  const rows = [
    { role: "user", content: "Earlier: keep the reports small." },
    { role: "assistant", content: "Understood." },
    { role: "user", content: "Summarize data.csv into report.json." },
    { role: "assistant", content: "", tool_calls: [{ id: "v1", type: "function", function: { name: "skill_view", arguments: "{\"name\": \"tally-report\"}" } }] },
    { role: "tool", tool_call_id: "v1", content: JSON.stringify({ success: true, name: "tally-report", content: skillText }) },
    { role: "assistant", content: "Done." },
  ];
  const exported = [0, 1].map((n) => JSON.stringify({ id: `20261018_12000${n}_aa11bb`, source: "cli", model: "m",
    title: `chat ${n}`, system_prompt: "You are Hermes.", started_at: 1760000000 + n, messages: rows })).join("\n");
  await page.eval(`location.hash = "#/runs"`);
  await until(() => page.eval(`[...document.querySelectorAll(".page-head a")].some((a) => a.textContent === "Import a chat…")`), "Import a chat… on the Runs page");
  await page.eval(`[...document.querySelectorAll(".page-head a")].find((a) => a.textContent === "Import a chat…").click()`);
  await until(() => page.eval(`location.hash === "#/import" && !!document.querySelector('input[type="file"]')`), "the import page");
  await page.eval(`(() => {
    const picker = document.querySelector('input[type="file"]');
    const files = new DataTransfer();
    files.items.add(new File([${JSON.stringify(exported)}], "export.jsonl", { type: "application/json" }));
    picker.files = files.files;
    picker.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  })()`);
  await until(() => page.eval(`document.querySelectorAll('input[name="session"]').length === 2`), "a file of two sessions asks which");
  await page.eval(`(() => { const choice = document.querySelector('input[name="session"][value="20261018_120001_aa11bb"]'); choice.click(); return true; })()`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "What it holds")`), "what the chosen session holds");
  check(await page.eval(`document.querySelector("main").textContent.includes("tally-report — loaded by a call; in context from step 2")`),
    "the skill it shows, and from which step it is in context");
  check(await page.eval(`[...document.querySelectorAll(".banner-warn")].some((b) => b.textContent.includes("names no tools"))`),
    "what the file lacked, said before anything is recorded");
  await page.type(`document.querySelector('input[name="tools_from"]')`, run.id.slice(0, 8));
  await page.eval(`document.querySelector('input[name="tools_from"]').dispatchEvent(new Event("change", { bubbles: true }))`);
  await until(() => page.eval(`document.querySelector("main").textContent.includes("from run ${run.id.slice(0, 8)}")`),
    "the tools a recorded run offered, taken for each step");
  check(await page.eval(`![...document.querySelectorAll(".banner-warn")].some((b) => b.textContent.includes("names no tools"))`),
    "a run's tools named on the page, as --tools-from names them: the warning gone");
  await page.eval(`[...document.querySelectorAll("button")].find((b) => b.textContent === "Import" && !b.disabled).click()`);
  await until(() => page.eval(`/^#\\/runs\\/[0-9a-f]{32}$/.test(location.hash)`), "the imported run's page");
  await until(() => page.eval(`(document.querySelector(".page-head .sub")?.textContent || "").includes("imported from a Hermes export (session 20261018_120001_aa11bb)")`),
    "the run says where it came from");
  const imported = (await api(`/runs/${(await page.eval("location.hash")).split("/")[2]}`)).run;
  check(imported.label === "imported: chat 1" && imported.harness === "hermes", "labelled by its session's title, its harness Hermes");
  check(await page.eval(`(document.querySelector(".tl-tools summary")?.textContent || "").startsWith("${offered.length} tools offered")`),
    "its steps offer the recorded run's tools, shown above its system prompt");
  check(await page.eval(`(() => {
    const turn = [...document.querySelectorAll(".tl-rail-turn")].find((a) => a.textContent === "userSummarize data.csv into report.json.");
    return !!turn && turn.previousElementSibling?.dataset.idx === "0" && turn.nextElementSibling?.dataset.idx === "1";
  })()`), "a person's next turn starts a new exchange in the rail, unindented, in their words");
  check(await page.eval(`(() => {
    const first = document.querySelector(".tl-rail-head").nextElementSibling;
    return first?.classList.contains("tl-rail-turn") && first.textContent === "userEarlier: keep the reports small."
      && first.nextElementSibling?.dataset.idx === "0";
  })()`), "and the turn the run began with heads the rail, above its first step");
  await page.eval(`location.hash = "#/runs"`);
  const importedRow = `[...document.querySelectorAll(".run-row")].find((row) => row.querySelector("a.run-id")?.getAttribute("href") === "#/runs/${imported.id}")`;
  await until(() => page.eval(`!!${importedRow}`), "the imported run on the Runs page");
  check(await page.eval(`${importedRow}.querySelector(".chip")?.textContent === "imported"`),
    "the Runs list marks it imported, as it marks a run recorded live");
  await page.eval(`location.hash = "#/runs/${imported.id}"`);

  console.log("a scenario of it: from the person's message the skill came in after, in words they may change");
  await until(() => page.eval(`[...document.querySelectorAll(".page-head a")].some((a) => a.textContent === "Make a scenario…")`), "Make a scenario… on the imported run");
  await page.eval(`[...document.querySelectorAll(".page-head a")].find((a) => a.textContent === "Make a scenario…").click()`);
  await until(() => page.eval(`!!document.querySelector('textarea[name="prompt"]')?.value`), "the scenario form");
  check(await page.eval(`document.querySelector('textarea[name="prompt"]').value === "Summarize data.csv into report.json."`),
    "its starting prompt is the person's message the skill came in after, in their words");
  check(await page.eval(`(() => { const text = document.querySelector("main").textContent;
    return text.includes("From message 3 of the chat") && text.includes("The person's other message is not offered")
      && document.querySelector('select[name="cut"]').closest("label").hidden; })()`),
    "their earlier message is not offered, and the page says why: there is no choice to make");
  check(await page.eval(`(() => { const row = document.querySelector(".start-steps tbody tr.step");
    return !!row && row.classList.contains("part-first") && row.textContent.includes("skill_view") && row.textContent.includes("each run starts here"); })()`),
    "the chat's steps from there, each run starting where the model loaded the skill");
  check(await page.eval(`(() => { const row = (part) => document.querySelector(".start-steps tr.part-" + part)?.textContent || "";
    return row("system").includes("You are Hermes.") && row("system").includes("its harness's own, not the chat's")
      && row("tools").includes("${offered.length} tools offered") && row("before").includes("2 messages before the starting prompt")
      && row("prompt").includes("Summarize data.csv into report.json.") && row("prompt").includes("sent as the person's message"); })()`),
    "before the steps, what each run is given beside the chat's: the system prompt, the tools, the conversation before, the prompt");
  await page.type(`document.querySelector('textarea[name="prompt"]')`, "Summarize inputs/data.csv into report.json.");
  check(await page.eval(`document.querySelector('textarea[name="prompt"]').classList.contains("field-changed")`), "a prompt rewritten shows as changed");
  check(await page.eval(`(() => { const row = document.querySelector(".start-steps tr.part-prompt").textContent;
    return row.includes("yours, in place of the chat's") && row.includes("Summarize inputs/data.csv into report.json."); })()`),
    "and the table says each run is sent it in place of the chat's words");
  // The check takes no text until it is a shell command or a reply's words (found in review).
  check(await page.eval(`document.querySelector('[name="check-kind-run"][value="none"]')?.checked
    && !document.querySelector('[name="check-kind-run"][value="own"]') && !document.querySelector('[name="check-run"], [name="check-text-run"]')`),
    "its check is chosen as a new scenario's is: with no task to take one from, none yet, and no text to give it");
  await until(() => page.eval(`(document.querySelector(".plan-status .form-note-box")?.textContent || "").includes("Name the scenario")`),
    "what it still needs, as a note");
  await page.type(`document.querySelector('input[name="scenario_name"]')`, "from-the-chat");
  // What will be made, shown before it is, as `tuningfork scenario --dry-run` shows it.
  await planShows(page, "from-the-chat");
  await startButtonReady(page, "Make the scenario");
  await page.click("Make the scenario");
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "Open the scenario")`), "the scenario made");
  check(await page.eval(`[...document.querySelectorAll(".banner-warn")].some((b) => b.textContent.includes("no check"))`), "and says it has no check yet");
  check((await api("/scenarios/from-the-chat")).spec.evals[0].prompt === "Summarize inputs/data.csv into report.json.",
    "its prompt is the one written in the form");
  await page.eval(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "Open the scenario").click()`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "The conversation it continues")`), "the scenario's conversation");
  check(await page.eval(`document.querySelector("main").textContent.includes("2 messages of the conversation it was made of")`), "the two turns before it, as each run's context");

  console.log("an import whose file kept no tools says so where they would be; a session file's tools, each opened");
  const bare = await post("/imports", { content: exported, session: "20261018_120000_aa11bb" });
  await page.eval(`location.hash = "#/runs/${bare.run}"`);
  await until(() => page.eval(`!!document.querySelector(".tl-tools")`), "the bare import's page");
  check(await page.eval(`document.querySelector(".tl-tools").textContent.startsWith("No tools recorded: Hermes's export keeps none")`),
    "no tools recorded, and why, where they would be");
  // A reply cut at the output limit, and the nudge Hermes sends itself to go on.
  const nudged = [
    { role: "user", content: "Summarize data.csv into report.json." },
    { role: "assistant", content: "Writing the rep", finish_reason: "length" },
    { role: "user", content: "[System: Your previous response was truncated by the output length limit. Continue exactly where you left off. Do not restart or repeat prior text. Finish the answer directly.]" },
    { role: "assistant", content: "Done.", finish_reason: "stop" },
  ];
  const sessionFile = JSON.stringify({
    session_id: "20261018_130000_cc22dd", model: "m", base_url: "http://x/v1", platform: "cli", session_start: "2026-10-18T13:00:00",
    system_prompt: "You are Hermes.", message_count: nudged.length, messages: nudged,
    tools: [{ type: "function", function: { name: "write_file", description: "Write a file.\nIt replaces what is there.", parameters: {
      type: "object", properties: { path: { type: "string", description: "Where to write" }, content: { type: "string" } }, required: ["path"] } } }],
  });
  const rich = await post("/imports", { content: sessionFile });
  await page.eval(`location.hash = "#/runs/${rich.run}"`);
  await until(() => page.eval(`!!document.querySelector(".tl-tools details.tools-offered")`), "the session file's tools");
  await page.eval(`(() => { const list = document.querySelector(".tl-tools details.tools-offered"); list.open = true; list.querySelector(".tool").open = true; return true; })()`);
  check(await page.eval(`(() => {
    const tool = document.querySelector(".tl-tools .tool");
    const params = [...tool.querySelectorAll(".tool-params li")].map((li) => li.textContent);
    return tool.querySelector(".tool-gist").textContent === "Write a file."
      && tool.querySelector(".tool-body pre.text").textContent === "Write a file.\\nIt replaces what is there."
      && params.join(" / ") === "path string required — Where to write / content string";
  })()`), "each tool opens to what it says it does and the parameters it takes");
  check(await page.eval(`(() => {
    const line = [...document.querySelectorAll(".tl-rail-result")].find((a) => a.textContent === "↳continue after a cut reply");
    const said = line && document.querySelector("#tm" + line.dataset.entry);
    // The person's one turn is the one the run began with, above its first step: the nudge is none.
    const turns = [...document.querySelectorAll(".tl-rail-turn")];
    return !!said && line.previousElementSibling?.dataset.idx === "0"
      && turns.length === 1 && turns[0].nextElementSibling?.dataset.idx === "0"
      && [...said.querySelectorAll(".chip")].some((chip) => chip.textContent === "continue after a cut reply");
  })()`), "Hermes's own nudge after a cut reply stays under its step, named, and says so in the conversation");

  console.log("a chat that used two skills: both kept, and the work names the one it studies");
  const tidyText = "---\nname: csv-tidy\n---\n\n# Tidy\n\nSort the rows.\n";
  const used = [
    { role: "user", content: "Tidy data.csv, then summarize it into report.json." },
    { role: "assistant", content: "", tool_calls: [{ id: "a1", type: "function", function: { name: "skill_view", arguments: "{\"name\": \"csv-tidy\"}" } }] },
    { role: "tool", tool_call_id: "a1", content: JSON.stringify({ success: true, name: "csv-tidy", content: tidyText }) },
    { role: "assistant", content: "", tool_calls: [{ id: "a2", type: "function", function: { name: "skill_view", arguments: "{\"name\": \"tally-report\"}" } }] },
    { role: "tool", tool_call_id: "a2", content: JSON.stringify({ success: true, name: "tally-report", content: skillText }) },
    { role: "assistant", content: "Done." },
  ];
  const two = await post("/imports", { content: JSON.stringify({ id: "20261019_090000_dd33ee", source: "cli", model: "m",
    title: "two skills", system_prompt: "You are Hermes.", started_at: 1760100000, messages: used }) });
  check(two.skills.join(",") === "csv-tidy,tally-report" && two.skill === null, "both skills kept, and neither studied until the work names it");
  await page.eval(`location.hash = "#/runs/${two.run}/2?start=refine"`);
  await until(() => page.eval(`!!document.querySelector('.start-panel select[name="skill"]')`), "the refinement form's skill");
  check(await page.eval(`[...document.querySelector('.start-panel select[name="skill"]').options].map((o) => o.value).filter(Boolean).join(",")`) === "csv-tidy,tally-report",
    "the refinement form asks which skill it refines");
  await page.eval(`location.hash = "#/runs/${two.run}/2?start=ab"`);
  await until(() => page.eval(`!!document.querySelector('.start-panel select[name="skill"]') && !document.querySelector('.start-panel textarea[name="variant_text"]')`), "the A/B form, asking first");
  await page.eval(`(() => { const pick = document.querySelector('.start-panel select[name="skill"]'); pick.value = "csv-tidy"; pick.dispatchEvent(new Event("change")); return true; })()`);
  await until(() => page.eval(`(document.querySelector('.start-panel textarea[name="variant_text"]')?.value || "").includes("name: csv-tidy")`), "the chosen skill's text to edit");
  check(true, "the A/B form edits the skill chosen, after asking which");

  console.log("a scenario of the second skill takes the first one's steps as done; one of the first stops at the second");
  const later = await post(`/imports/${two.run}/scenario`, { name: "after-tidy", skill: "tally-report" });
  check(later.warnings.some((warning) => warning.includes("each run starts at step 1, where the chat loaded tally-report, with the turn's step 0 taken as done")),
    "a scenario of the second skill takes the first one's steps as done");
  await page.eval(`location.hash = "#/scenarios/after-tidy"`);
  await until(() => page.eval(`[...document.querySelectorAll(".panel h2")].some((h2) => h2.textContent === "The conversation it continues")`), "the scenario's conversation");
  check(await page.eval(`(() => { const text = document.querySelector("main").textContent;
    return text.includes("In the prompt's turn, taken as done") && text.includes("Then 2 messages of that turn"); })()`),
    "its page shows the turn's earlier steps, taken as done");
  const tidy = await post(`/imports/${two.run}/scenario`, { name: "tidy-only", skill: "csv-tidy" });
  check(tidy.scenario.spec.evals[0].tuningfork.stop_at_next_skill === true, "a scenario of the first skill stops each run at the next");
  await page.eval(`location.hash = "#/scenarios/tidy-only"`);
  await until(() => page.eval(`!!document.querySelector('input[name="stop-0"]')`), "the eval's stop");
  check(await page.eval(`document.querySelector('input[name="stop-0"]').checked`), "its page says each run stops where another skill loads");

  console.log("from the Scenarios page: a run chosen, its steps laid out by the skill studied");
  await page.eval(`location.hash = "#/scenarios"`);
  await until(() => page.eval(`[...document.querySelectorAll(".page-head a")].some((a) => a.textContent === "From a run…")`), "From a run… on the Scenarios page");
  await page.eval(`[...document.querySelectorAll(".page-head a")].find((a) => a.textContent === "From a run…").click()`);
  await until(() => page.eval(`location.hash === "#/scenarios?from=run" && !!document.querySelector("main table tbody tr")`), "the runs to choose from");
  check(await page.eval(`[...document.querySelectorAll(".page-head a")].some((a) => a.getAttribute("href") === "#/import?then=scenario")`),
    "a chat had elsewhere is imported from there too");
  await page.type(`document.querySelector('input[type="search"]')`, two.run.slice(0, 8));
  await until(() => page.eval(`document.querySelectorAll("main table tbody tr").length === 1`), "the filter to leave the one run");
  await page.eval(`document.querySelector("main table tbody tr a").click()`);
  await until(() => page.eval(`location.hash === "#/runs/${two.run}/scenario" && !!document.querySelector('select[name="skill"]')`), "the run's scenario form");
  check(await page.eval(`(() => { const pick = document.querySelector('select[name="skill"]');
    const prompt = document.querySelector('textarea[name="prompt"]');
    return pick.closest("label").classList.contains("wide") && pick.classList.contains("field-ask")
      && prompt.disabled && prompt.placeholder.startsWith("Choose the skill it studies first"); })()`),
    "the skill asked for first, on a row of its own, and the starting prompt says to choose it");
  // Each step's row: its part, and what it says a run does there.
  const laidOut = () => page.eval(`[...document.querySelectorAll(".start-steps tbody tr.step")].map((row) => row.className.replace("step ", "") + ":" + row.lastElementChild.textContent).join(" | ")`);
  const study = async (skill) => {
    const before = await laidOut();
    await page.eval(`(() => { const pick = document.querySelector('select[name="skill"]'); pick.value = ${JSON.stringify(skill)}; pick.dispatchEvent(new Event("change")); return true; })()`);
    return until(async () => { const now = await laidOut(); return now && now !== before ? now : null; }, `the steps for ${skill}`);
  };
  const ofSecond = await study("tally-report");
  check(await page.eval(`document.querySelector('textarea[name="prompt"]').value === "Tidy data.csv, then summarize it into report.json."`),
    "the skill chosen, its starting prompt is the person's message");
  check(ofSecond.startsWith("part-done:given as done | part-first:each run starts here"),
    `studying the second skill, the first one's load is given as done and each run starts at the second's (${ofSecond})`);
  const ofFirst = await study("csv-tidy");
  check(ofFirst.startsWith("part-first:each run starts here | part-live:made by its modelstops after a load like this"),
    `studying the first skill, each run starts at its load and stops at the second's (${ofFirst})`);

  console.log("a run tuningfork launched, as a scenario: its task's check, its workspace's files");
  const ofTask = (await api("/runs?task=tally-scenario/1&limit=5"))[0];
  await page.eval(`location.hash = "#/runs/${ofTask.id}"`);
  await until(() => page.eval(`[...document.querySelectorAll(".page-head a")].some((a) => a.textContent === "Make a scenario…")`), "Make a scenario… on a launched run");
  await page.eval(`[...document.querySelectorAll(".page-head a")].find((a) => a.textContent === "Make a scenario…").click()`);
  await until(() => page.eval(`location.hash === "#/runs/${ofTask.id}/scenario" && !!document.querySelector('[name="check-kind-run"]')`), "the scenario form");
  check(await page.eval(`document.querySelector('[name="check-kind-run"][value="own"]').checked
    && document.querySelector(".check-command pre").textContent.startsWith("python3")`),
    "its check is the task's own unless told otherwise, chosen as a new scenario's is");
  check(await page.eval(`document.querySelector("main").textContent.includes("Its inputs: the run's workspace where the scenario starts")`),
    "it says its inputs come from the run's workspace");
  await page.type(`document.querySelector('input[name="scenario_name"]')`, "from-a-launch");
  await planShows(page, "data.csv");
  await startButtonReady(page, "Make the scenario");
  // Tried before it is trusted, on the run it is made of: the check in a copy of the workspace it
  // left, on its final reply — and the trial's own Try it, a form inside this one, makes nothing.
  await page.click("Test this check…");
  await until(() => page.eval(`document.querySelector('.check-tester [name="trial_run"]')?.value === "${ofTask.id}"`),
    "the tester, on the run the scenario is made of");
  await until(() => page.eval(`[...document.querySelectorAll(".check-tester button")].some((b) => b.textContent === "Try it" && !b.disabled)`), "the trial's plan");
  await page.click("Try it", `document.querySelector(".check-tester")`);
  await until(() => page.eval(`!!document.querySelector(".check-tester .check-verdict")`), "the trial's verdict", 30000);
  check(await page.eval(`document.querySelector(".check-tester .check-verdict").textContent.includes("as when the run ran")`),
    "the task's check, tried on the run the scenario is made of, gives the verdict that run got");
  check(!(await api("/scenarios")).some((scenario) => scenario.name === "from-a-launch"), "and trying it made no scenario");
  await page.click("Make the scenario");
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "Open the scenario")`), "the scenario made");
  check(await page.eval(`[...document.querySelectorAll(".banner-warn")].some((b) => b.textContent.includes("workspace as its step 0 found it: 1 file(s)"))`),
    "its inputs are the run's workspace as it started");
  const fromLaunch = await api("/scenarios/from-a-launch");
  check(fromLaunch.spec.evals[0].id === "from-a-launch" && fromLaunch.spec.evals[0].files[0].path === "data.csv"
    && fromLaunch.inputs.length === 1 && fromLaunch.spec.evals[0].tuningfork.check.startsWith("python3"),
    "its eval is a task of its own, with the run's data.csv as its input and the task's check");

  console.log("a run recorded live that continued a conversation: said so, and its rail lists what came before");
  const sentAt = Date.now() / 1000 - 1;
  const continued = [
    { role: "system", content: "You are Hermes." },
    { role: "user", content: "File receipt A." },
    { role: "assistant", content: "", tool_calls: [{ id: "r1", type: "function", function: { name: "read_file", arguments: "{\"path\": \"a.pdf\"}" } }] },
    { role: "tool", tool_call_id: "r1", content: "Total: $12.00" },
    { role: "assistant", content: "Filed A: $12.00." },
    { role: "user", content: "Now file receipt B." },
  ];
  const sent = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "gemma-4-31b", messages: continued }),
  });
  check(sent.ok, "a call through the proxy, as a harness going on with a session makes one");
  const caught = await until(async () => (await api("/runs?limit=100")).find((found) => found.recorded === "live" && found.started_at >= sentAt),
    "the run it was recorded as");
  check(!caught.label && !caught.launch_id && !caught.imported, "nothing launched it, nor imported it, and it has no label");
  await page.eval(`location.hash = "#/runs"`);
  await until(() => page.eval(`[...document.querySelectorAll(".run-row")].some((row) => row.textContent.includes("${caught.id.slice(0, 8)}"))`), "it in the Runs list");
  check(await page.eval(`[...document.querySelectorAll(".run-row")].find((row) => row.textContent.includes("${caught.id.slice(0, 8)}"))
    .querySelector(".chip").textContent === "recorded live"`), "the Runs list marks it recorded live, where a label would name it");
  // Its system prompt is the imported chats' too, but nothing else of them is: no branch of theirs.
  check(caught.forked_from_run === null && !(await page.eval(`[...document.querySelectorAll(".run-row")]
    .find((row) => row.textContent.includes("${caught.id.slice(0, 8)}")).textContent.includes("fork @")`)),
    "and sharing only a system prompt with other runs, it is no fork of theirs");
  await page.eval(`location.hash = "#/runs/${caught.id}"`);
  await until(() => page.eval(`!!document.querySelector(".tl-rail-head")`), "its page");
  check(await page.eval(`(() => { const head = document.querySelector(".page-head");
    return head.querySelector("h1 .chip")?.textContent === "recorded live"
      && head.querySelector(".sub").textContent.includes("recorded live through the proxy")
      && head.querySelector(".sub").textContent.includes("its first call continuing a conversation of 5 messages, 2 turns of the person's"); })()`),
    "its page says it was recorded live, continuing a conversation");
  const lines = await page.eval(`[...document.querySelector(".tl-rail").children].map((line) => line.textContent).join(" | ")`);
  check(lines.startsWith("1 step · after 2 earlier answers | userFile receipt A. | ·read_file | ↳read_file result | ·reply | userNow file receipt B. | 0"),
    `its rail lists the conversation the first call continued, its answers unnumbered, then the run's step (${lines})`);
  const earlierLine = await page.eval(`(() => { const line = document.querySelector(".tl-rail-earlier"); line.click(); return line.dataset.entry; })()`);
  await until(() => page.eval(`document.querySelector("#tm${earlierLine}")?.open === true`), "the earlier answer opened from the rail");
  check(await page.eval(`(() => { const top = document.querySelector("#tm${earlierLine}").getBoundingClientRect().top; return top >= 0 && top < innerHeight / 2; })()`),
    "an earlier answer's line leads to it in the conversation, opened");

  console.log("a long session in the rail: each of the person's turns folds to its line");
  const three = await post("/imports", { content: JSON.stringify({ id: "20261019_100000_ee44ff", source: "cli", model: "m",
    title: "three turns", system_prompt: "You are Hermes.", started_at: 1760200000, messages: [
      { role: "user", content: "File receipt A." }, { role: "assistant", content: "Filed A." },
      { role: "user", content: "File receipt B." }, { role: "assistant", content: "Filed B." },
      { role: "user", content: "File receipt C." }, { role: "assistant", content: "Filed C." },
    ] }) });
  // Each turn's line: open or folded, and what it says; each step's line: folded away or not.
  const turnsNow = () => page.eval(`[...document.querySelectorAll(".tl-rail-turn")]
    .map((turn) => turn.querySelector(".tl-rail-fold").getAttribute("aria-expanded") + ":" + turn.textContent).join(" | ")`);
  const stepsNow = () => page.eval(`[...document.querySelectorAll(".tl-rail-step")].map((line) => line.classList.contains("tl-folded") ? "folded" : "shown").join(",")`);
  await page.eval(`location.hash = "#/runs/${three.run}"`);
  await until(() => page.eval(`document.querySelectorAll(".tl-rail-turn").length === 3`), "its three turns in the rail");
  check(await turnsNow() === "false:userFile receipt A.step 0 | false:userFile receipt B.step 1 | true:userFile receipt C."
    && await stepsNow() === "folded,folded,shown", "all but the last turn start folded to their line, saying what each holds");
  await page.eval(`document.querySelector(".tl-rail-turn .tl-rail-fold").click()`);
  check((await turnsNow()).startsWith("true:userFile receipt A. | false:") && await stepsNow() === "shown,folded,shown",
    "a turn unfolds from its line");
  await page.eval(`document.querySelector(".tl-rail-foldall").click()`);
  check(await stepsNow() === "folded,folded,folded" && await page.eval(`document.querySelector(".tl-rail-foldall").textContent`) === "unfold all",
    "and all fold at once, from the rail's head");
  await page.eval(`document.querySelector(".tl-rail-foldall").click()`);
  check(await stepsNow() === "shown,shown,shown", "and unfold");
  await page.eval(`location.hash = "#/runs/${three.run}/1"`);
  await until(() => page.eval(`!!document.querySelector("#s1.tl-open")`), "its step 1, opened from its address");
  check(await stepsNow() === "folded,shown,shown", "the turn holding the step the address opens starts open");

  check(page.problems.length === 0, `no page errors${page.problems.length ? `: ${page.problems.join(" | ")}` : ""}`);
  console.log("all passed");
} catch (error) {
  console.log(error.message);
  console.log("page errors:", page.problems);
  console.log("hash:", await page.eval("location.hash"));
  console.log("form:", await page.eval(`[document.querySelector(".plan-status")?.innerText, document.querySelector(".form-actions")?.innerText].join(" / ")`));
  console.log("main:", (await page.eval(`document.querySelector("main").innerText`)).slice(0, 1200));
  process.exitCode = 1;
} finally {
  await page.close();
}
