// The README's pictures of the web UI, taken again with one command. Each is a page of a
// tuningfork serving the recordings it shows, opened in headless Chrome at one pixel to one in the
// light theme, made ready as the list below says, and saved under the name the README uses — with
// no address bar on it, so no access token. The GIF is taken the same way: the page's own
// screencast, frame by frame, while the list's clicks and keys go in, with a pointer drawn where
// each click lands; ffmpeg puts the frames together. README.md beside this file says where each
// picture's recordings come from, and how to serve them.
//
//   node docs/images/shots.mjs [name …] [--out <folder>]     every picture, or the ones named
//
// It reads TUNINGFORK_BROWSER, a Chromium-based browser's DevTools address, as the browser tests
// do; for each source, the web UI address its server printed, token and all —
//
//   TUNINGFORK_SHOTS_FINDFILE  `tuningfork serve` over examples/find-the-file's recordings
//   TUNINGFORK_SHOTS_GUIDE     `tuningfork serve` over the guide's, refinement 338504b1 among them
//   TUNINGFORK_SHOTS_LIVE      `tuningfork demo live`, started fresh
//   TUNINGFORK_SHOTS_LAB       `tuningfork demo`, started fresh: the GIF, and a chat had elsewhere
//
// — FFMPEG, where ffmpeg is not on the PATH, and TUNINGFORK_SHOTS_PRIVATE: words no picture may
// show, comma-separated — your user names, a host, a port that serves something else. A picture
// whose source is not named is skipped.

import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// cdp.mjs reads one demo's token as it loads; each source here brings its own.
process.env.TUNINGFORK_TOKEN ??= "unused";
const { Page, sleep, until } = await import(new URL("../../tests/browser/cdp.mjs", import.meta.url));

const FFMPEG = process.env.FFMPEG || "ffmpeg";
const PRIVATE = (process.env.TUNINGFORK_SHOTS_PRIVATE || "").split(",").map((word) => word.trim()).filter(Boolean);
// What a page shows a reader: its text, and what its fields hold, which its text leaves out.
const SHOWN = `[document.body.innerText, ...[...document.querySelectorAll("input, textarea, select")].map((field) => field.value)].join("\\n")`;

/** What of one machine `text` shows: a path naming a person or a place — a home on Linux or a
 *  Mac, a drive, a WSL distro's share — and each private word, as a word of its own. */
function exposed(text) {
  // A path's start only: the lab's seeded runs name /tmp/run/home/skills, which names nobody.
  const paths = text.match(/(?<![\w.~-])\/(?:home\/(?!user\b)[\w.-]+|Users\/[\w.-]+|mnt\/\w+)|\b[A-Za-z]:[\\/][\w.-]+|\\\\wsl(?:\.localhost|\$)\\[\w.-]+/g) || [];
  // A word between letters or digits is part of another (8088 in an id); one beside _ is not.
  const words = PRIVATE.filter((word) => new RegExp(`(?<![A-Za-z0-9])${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![A-Za-z0-9])`, "i").test(text));
  return [...new Set([...paths, ...words])];
}
const SKILL = readFileSync(new URL("../../examples/tally-report/SKILL.md", import.meta.url), "utf8");

// What the pictures show, as the recordings hold it.
const FIND_THE_FILE = {
  fixed: "2ff0083ff27443409e2791bcd370af43", // a run given the fixed skill: its step 2's exact context
  passed: "472253d679934838a9f87e4dc964675b", // a run of the original that searched: step 1's tokens
  ab: "5891a3c562454270a7f64ea26f04c561", // the fix, A/B-tested at a failing run's step 1
  setup: "d668ec81", // the original skill's runs, where they split
};
const REFINEMENT = "338504b156154e7ea15cbbf41f30e355"; // the guide's, from its target to the approval

// --- the pictures ---------------------------------------------------------------------------------

const SHOTS = [
  {
    name: "ui-guide.png", source: "guide", hash: "#/guide", ready: ".map svg",
  },
  {
    name: "ui-run.png", source: "findfile", hash: `#/runs/${FIND_THE_FILE.fixed}/2/context`, ready: ".msg",
  },
  {
    name: "ui-divergence.png", source: "findfile", hash: `#/divergence/find-the-file%2F1@${FIND_THE_FILE.setup}`, ready: ".run-chips a.chip",
  },
  {
    name: "ui-ab.png", source: "findfile", hash: `#/experiments/${FIND_THE_FILE.ab}`, ready: ".verdict-block",
  },
  {
    // Token [121], "Wait" where the model more often wrote "According": the hero figure's fork.
    name: "ui-tokens.png", source: "findfile", hash: `#/runs/${FIND_THE_FILE.passed}/1?view=tokens&at=121`, ready: ".tk-detail:not([hidden])",
    act: async (page) => {
      await page.eval(`(() => {
        const token = document.querySelector('.tk-t[data-i="121"]');
        window.scrollBy(0, token.getBoundingClientRect().top - 210);
        return true;
      })()`);
    },
  },
  {
    name: "ui-refine.png", source: "guide", hash: `#/refine/${REFINEMENT}`, ready: ".stepper",
  },
  {
    // The held write_file, its keys edited to the ones the check reads.
    name: "ui-live.png", source: "live", hash: "#/live", ready: ".hold-head",
    act: async (page) => {
      await page.enter(`[...document.querySelectorAll("main textarea")].find((field) => field.value.includes('"count"'))`,
        '{\n  "row_count": 4,\n  "amount_total": 61.5\n}');
      await page.eval(`document.activeElement.blur()`);
      await sleep(400);
    },
  },
  {
    name: "ui-demo.gif", source: "lab", width: 1200, height: 800, play: abDemo,
  },
  {
    // A chat had elsewhere, made a scenario: its first message the starting prompt, and each of
    // its steps beside what a run of the scenario does there.
    name: "ui-scenario.png", source: "lab", prepare: importedChat, hash: (run) => `#/runs/${run}/scenario`, ready: '[name="scenario_name"]', height: 1000,
    act: async (page) => {
      await page.type(`document.querySelector('[name="scenario_name"]')`, "expenses-report");
      await page.eval(`document.querySelector('[name="check-kind-run"][value="shell"]').click()`);
      await sleep(300);
      await page.type(`document.querySelector('[name="check-run"]')`,
        "python3 -c \"import json; d = json.load(open('report.json')); assert set(d) == {'row_count', 'amount_total'}\"");
      await until(() => page.eval(`(document.querySelector(".plan-preview")?.textContent || "").includes("expenses-report")`), "the scenario's plan");
      await page.eval(`document.activeElement.blur(); window.scrollTo(0, 0)`);
    },
  },
];

// --- the chat the last one shows: made up, about the example skill ---------------------------------

const SYSTEM = "You are Hermes Agent, an intelligent AI assistant created by Nous Research.";
const call = (id, name, args) => ({ role: "assistant", content: "", tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }] });
const result = (id, content) => ({ role: "tool", tool_call_id: id, content: JSON.stringify(content) });
const FIRST = [
  { role: "user", content: "Can you total up expenses.csv for me? I need how many line items there are and the total amount." },
  call("c1", "skill_view", { name: "tally-report" }),
  result("c1", { success: true, name: "tally-report", content: SKILL }),
  call("c2", "read_file", { path: "expenses.csv" }),
  result("c2", { content: "item,amount\ntaxi,18.40\nlunch,12.60\nhotel,96.00\ncoffee,4.50\n", total_lines: 5 }),
  call("c3", "write_file", { path: "report.json", content: "{\"count\": 4, \"total\": 131.5}" }),
  result("c3", { success: true }),
  { role: "assistant", content: "Done: report.json says 4 line items, 131.50 in total." },
];
const SECOND = [
  { role: "user", content: "Our importer rejected it: it expects the keys row_count and amount_total." },
  call("c4", "write_file", { path: "report.json", content: "{\"row_count\": 4, \"amount_total\": 131.5}" }),
  result("c4", { success: true }),
  { role: "assistant", content: "Fixed: report.json now has row_count and amount_total." },
];

/** The chat imported as a Hermes session file: its run, imported once. */
async function importedChat(source) {
  const title = "expenses report";
  const found = (await source.api("/runs?limit=200")).find((run) => run.label === `imported: ${title}`);
  if (found) return found.id;
  const made = await source.post("/imports", { content: JSON.stringify({
    id: "20261021_093000_a1b2c3", source: "cli", model: "gemma-4-31b", title, system_prompt: SYSTEM,
    started_at: 1792571400, messages: [...FIRST, ...SECOND],
  }) });
  if (!made.run) throw new Error(`the import was refused: ${JSON.stringify(made)}`);
  return made.run;
}

// --- the GIF: a wording A/B-tested at a step, from the lab's runs to the A/B's page ---------------

/** The lab's model names the report's keys only when the skill holds the words
 *  `row_count and amount_total`: the edit below moves it from 0 in 20 to 20 in 20. */
async function abDemo(page, source) {
  const run = (await source.api("/runs?task=tally-report/1&limit=50")).find((each) => each.label === "tally-report/1#demo.2");
  if (!run) throw new Error("no run tally-report/1#demo.2: start tuningfork demo on a new folder");
  const pointer = await pointerOn(page);
  await sleep(1200);
  await pointer.click(`[...document.querySelectorAll(".run-row")].find((row) => row.querySelector(".run-label")?.textContent === "tally-report/1#demo.2").querySelector("a.run-id")`);
  await until(() => page.eval(`!!document.querySelector("#s1 .tl-work")`), "the run's page");
  await sleep(1000);
  await pointer.click(`document.querySelector("#s1 .tl-work")`);
  await until(() => page.eval(`!!document.querySelector(".tl-workcol .start-actions a")`), "the step's work");
  await sleep(700);
  await pointer.click(`[...document.querySelectorAll(".tl-workcol .start-actions a")].find((a) => a.textContent === "A/B a wording…")`);
  await until(() => page.eval(`!!document.querySelector('.tl-workcol textarea[name="variant_text"]')`), "the A/B's form");
  await sleep(900);
  // The edit: the keys named where the skill says what to write.
  const sentence = "Write the totals to report.json";
  await pointer.caret(`document.querySelector('.tl-workcol textarea[name="variant_text"]')`, sentence, sentence.length);
  await pointer.type(", using the keys row_count and amount_total");
  await sleep(700);
  // What to count: a behaviour described, a condition at a time.
  await pointer.click(`document.querySelector(".tl-workcol .rule-builder .rule input")`);
  await pointer.type("write_file");
  for (const phrase of ["row_count", "amount_total"]) {
    await pointer.click(`[...document.querySelectorAll(".tl-workcol .rule-builder button")].find((b) => b.textContent === "+ Add a condition")`);
    await pointer.click(`[...document.querySelectorAll(".tl-workcol .rule-builder .rule input")].at(-1)`);
    await pointer.type(phrase);
  }
  await sleep(500);
  const start = `[...document.querySelectorAll(".tl-workcol .plan-form button")].find((b) => b.textContent === "Start the A/B")`;
  await until(() => page.eval(`!!(${start}) && !(${start}).disabled && (document.querySelector(".tl-workcol .plan-preview")?.textContent || "").includes("20")`), "the A/B's plan");
  await pointer.point(start);
  await sleep(1200);
  await pointer.click(start);
  await until(() => page.eval(`[...document.querySelectorAll("a.btn")].some((a) => a.textContent === "See the result")`), "the A/B's result", 120000);
  await sleep(2600);
  await pointer.click(`[...document.querySelectorAll("a.btn")].find((a) => a.textContent === "See the result")`);
  // Its page: the edit it tested, then what it found.
  await until(() => page.eval(`location.hash.startsWith("#/experiments/") && !!document.querySelector(".verdict-block")`), "the A/B's page");
  await sleep(4500);
}

/** A pointer drawn on the page, over everything and catching nothing, and the mouse and keys it
 *  stands for: real events, at the point it is drawn. */
async function pointerOn(page) {
  await page.eval(`(() => {
    if (window.shotsPointer) return true;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 20 24");
    svg.setAttribute("width", "20");
    svg.setAttribute("height", "24");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", "M2 2 L2 19 L6.5 15 L9.5 22 L12.5 20.7 L9.6 14 L15.5 14 Z");
    path.setAttribute("fill", "#111");
    path.setAttribute("stroke", "#fff");
    path.setAttribute("stroke-width", "1.5");
    path.setAttribute("stroke-linejoin", "round");
    svg.append(path);
    const ring = document.createElement("div");
    Object.assign(svg.style, { position: "fixed", left: "640px", top: "420px", margin: "-2px 0 0 -2px", zIndex: "2147483647",
      pointerEvents: "none", transition: "left .55s ease-in-out, top .55s ease-in-out" });
    Object.assign(ring.style, { position: "fixed", width: "30px", height: "30px", margin: "-15px 0 0 -15px", borderRadius: "50%",
      border: "2px solid rgb(79 70 229 / 0.85)", opacity: "0", zIndex: "2147483646", pointerEvents: "none" });
    document.body.append(ring, svg);
    window.shotsPointer = {
      moveTo(x, y) { svg.style.left = x + "px"; svg.style.top = y + "px"; },
      press(x, y) {
        Object.assign(ring.style, { transition: "none", left: x + "px", top: y + "px", opacity: "1", transform: "scale(0.4)" });
        ring.getBoundingClientRect();
        Object.assign(ring.style, { transition: "opacity .5s ease, transform .5s ease", opacity: "0", transform: "scale(1.5)" });
      },
    };
    return true;
  })()`);
  const at = async (target) => {
    const scrolled = await page.eval(`(() => {
      const box = (${target}).getBoundingClientRect();
      if (box.top >= 70 && box.bottom <= innerHeight - 50) return false;
      (${target}).scrollIntoView({ block: "center", behavior: "smooth" });
      return true;
    })()`);
    if (scrolled) await sleep(650);
    return page.eval(`(() => { const box = (${target}).getBoundingClientRect(); return { x: box.x + Math.min(box.width / 2, 48), y: box.y + box.height / 2 }; })()`);
  };
  const move = async ({ x, y }) => {
    await page.eval(`window.shotsPointer.moveTo(${x}, ${y})`);
    await sleep(600);
    await page.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  };
  return {
    async point(target) {
      await move(await at(target));
    },
    async click(target) {
      const point = await at(target);
      await move(point);
      await page.eval(`window.shotsPointer.press(${point.x}, ${point.y})`);
      for (const type of ["mousePressed", "mouseReleased"]) {
        await page.send("Input.dispatchMouseEvent", { type, x: point.x, y: point.y, button: "left", clickCount: 1 });
      }
      await sleep(450);
    },
    /** The caret placed in a text field at `offset` within `text`, the pointer clicking there. */
    async caret(field, text, offset) {
      const point = await page.eval(`(() => {
        const field = ${field};
        const at = field.value.indexOf(${JSON.stringify(text)}) + ${offset};
        const style = getComputedStyle(field);
        const context = document.createElement("canvas").getContext("2d");
        context.font = style.font;
        const lines = field.value.slice(0, at).split("\\n");
        const box = field.getBoundingClientRect();
        return {
          x: box.x + parseFloat(style.paddingLeft) + context.measureText(lines.at(-1)).width,
          y: box.y + parseFloat(style.paddingTop) + (lines.length - 0.5) * parseFloat(style.lineHeight) - field.scrollTop,
          at,
        };
      })()`);
      await move(point);
      await page.eval(`window.shotsPointer.press(${point.x}, ${point.y})`);
      for (const type of ["mousePressed", "mouseReleased"]) {
        await page.send("Input.dispatchMouseEvent", { type, x: point.x, y: point.y, button: "left", clickCount: 1 });
      }
      await page.eval(`(() => { const field = ${field}; field.focus(); field.setSelectionRange(${point.at}, ${point.at}); return true; })()`);
      await sleep(350);
    },
    async type(text) {
      for (const character of text) {
        await page.send("Input.insertText", { text: character });
        await sleep(45);
      }
    },
  };
}

/** The page's screencast while `play` runs, as a GIF: each frame shown until the next one came,
 *  frames closer than a fifteenth of a second taken as one, the last held. With it, what every
 *  page it passed through showed — its text and fields, read every few tenths of a second, and
 *  any text it added, however briefly it stayed — to be checked as a still is. */
async function recordGif(page, file, { width, height }, play) {
  const frames = [];
  page.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
    frames.push({ data: Buffer.from(data, "base64"), at: metadata.timestamp });
    page.send("Page.screencastFrameAck", { sessionId });
  });
  // Each added part's text both whole and node by node: a word split across elements is found in
  // the one, a label and the value beside it stay two words in the other.
  await page.eval(`(() => {
    window.shotsSeen = [];
    new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === "characterData") window.shotsSeen.push(record.target.data);
        for (const node of record.addedNodes) {
          if (node.nodeType === Node.TEXT_NODE) {
            window.shotsSeen.push(node.data);
            continue;
          }
          window.shotsSeen.push(node.textContent || "");
          const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
          while (walker.nextNode()) window.shotsSeen.push(walker.currentNode.data);
        }
      }
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
    return true;
  })()`);
  const shown = new Set();
  let playing = true;
  const reading = (async () => {
    while (playing) {
      shown.add(await page.eval(SHOWN).catch(() => ""));
      await sleep(300);
    }
  })();
  await page.send("Page.startScreencast", { format: "png", maxWidth: width, maxHeight: height, everyNthFrame: 1 });
  await play();
  await page.send("Page.stopScreencast");
  playing = false;
  await reading;
  shown.add(await page.eval(SHOWN));
  shown.add(await page.eval(`window.shotsSeen.join("\\n")`));
  const kept = [];
  for (const frame of frames) {
    const last = kept.at(-1);
    if (last && frame.at - last.at < 1 / 15) last.data = frame.data;
    else kept.push({ ...frame });
  }
  const folder = mkdtempSync(join(tmpdir(), "tuningfork-gif-"));
  try {
    const list = ["ffconcat version 1.0"];
    kept.forEach((frame, i) => {
      writeFileSync(join(folder, `${i}.png`), frame.data);
      const next = kept[i + 1];
      list.push(`file '${i}.png'`, `duration ${(next ? next.at - frame.at : 3).toFixed(3)}`);
    });
    // The concat demuxer takes the last file's duration from a file after it.
    list.push(`file '${kept.length - 1}.png'`);
    writeFileSync(join(folder, "frames.txt"), `${list.join("\n")}\n`);
    // One palette for every frame, made first: the frames are read twice, never all held at once.
    const frameList = ["-f", "concat", "-safe", "0", "-i", join(folder, "frames.txt")];
    const palette = join(folder, "palette.png");
    execFileSync(FFMPEG, ["-y", "-loglevel", "error", ...frameList, "-vf", "palettegen=stats_mode=full", palette]);
    execFileSync(FFMPEG, ["-y", "-loglevel", "error", ...frameList, "-i", palette,
      "-lavfi", "[0:v][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle", "-fps_mode", "vfr", "-loop", "0", file]);
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
  return { frames: kept.length, shown: [...shown].join("\n") };
}

// --- taking them ----------------------------------------------------------------------------------

const args = process.argv.slice(2);
const outAt = args.indexOf("--out");
const out = outAt >= 0 ? args[outAt + 1] : fileURLToPath(new URL(".", import.meta.url));
const named = args.filter((arg, i) => !arg.startsWith("--") && args[i - 1] !== "--out");
mkdirSync(out, { recursive: true });
// Each picture is taken here first, and written beside the README only once nothing on it was found.
const stage = mkdtempSync(join(tmpdir(), "tuningfork-shots-"));

const sources = {};
for (const key of ["findfile", "guide", "live", "lab"]) {
  const address = process.env[`TUNINGFORK_SHOTS_${key.toUpperCase()}`];
  if (!address) continue;
  const url = new URL(address);
  const base = url.origin;
  const token = url.searchParams.get("token") || "";
  const api = async (path, init = {}) => {
    const response = await fetch(`${base}/_tuningfork/api${path}`, { ...init, headers: { ...init.headers, "x-tuningfork-token": token } });
    const text = await response.text();
    return text ? JSON.parse(text) : null;
  };
  const post = (path, body) => api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  sources[key] = { base, token, api, post };
}

let failed = 0;
for (const shot of SHOTS) {
  const short = shot.name.replace(/\.\w+$/, "");
  if (named.length && !named.includes(short) && !named.includes(shot.name)) continue;
  const source = sources[shot.source];
  if (!source) {
    console.log(`${shot.name}: skipped, TUNINGFORK_SHOTS_${shot.source.toUpperCase()} not set`);
    continue;
  }
  const width = shot.width ?? 1440;
  const height = shot.height ?? 896;
  const page = await Page.open("about:blank");
  try {
    await page.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    await page.send("Emulation.setScrollbarsHidden", { hidden: true });
    await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
    const prepared = shot.prepare ? await shot.prepare(source) : null;
    const hash = typeof shot.hash === "function" ? shot.hash(prepared) : shot.hash ?? "#/runs";
    await page.send("Page.navigate", { url: `${source.base}/_tuningfork/ui/?token=${source.token}${hash}` });
    await until(() => page.eval(`location.search === "" && !!document.querySelector("#main .view")?.childElementCount
      && !document.querySelector(".loading, .waiting-inline")${shot.ready ? ` && !!document.querySelector(${JSON.stringify(shot.ready)})` : ""}`),
      `${shot.name}'s page`, 30000);
    await sleep(500);
    const staged = join(stage, shot.name);
    let shown;
    if (shot.play) {
      const recorded = await recordGif(page, staged, { width, height }, () => shot.play(page, source));
      console.log(`${shot.name}: ${recorded.frames} frames`);
      shown = recorded.shown;
    } else {
      if (shot.act) await shot.act(page, source);
      await sleep(400);
      const { data } = await page.send("Page.captureScreenshot", { format: "png" });
      writeFileSync(staged, Buffer.from(data, "base64"));
      shown = await page.eval(SHOWN);
    }
    // Nothing of one machine: what a page shows that names a person, a drive or a private word.
    const paths = exposed(shown);
    const size = statSync(staged).size;
    const limit = shot.name.endsWith(".gif") ? 10e6 : 500e3;
    const clean = !paths.length && !page.problems.length;
    if (clean) {
      copyFileSync(staged, join(out, shot.name));
      rmSync(staged);
    } else {
      failed += 1;
    }
    console.log(`${shot.name}: ${width}×${height}, ${Math.round(size / 1024)} KB${size > limit ? ` — over ${limit / 1e3} KB` : ""}`
      + `${paths.length ? ` — the page names ${paths.join(", ")}` : ""}${page.problems.length ? ` — page errors: ${page.problems.slice(0, 3).join(" | ")}` : ""}`
      + `${clean ? "" : ` — not written; it is at ${staged}`}`);
  } catch (error) {
    failed += 1;
    console.log(`${shot.name}: FAILED ${error.message}`);
  } finally {
    await page.close();
  }
}
if (!readdirSync(stage).length) rmSync(stage, { recursive: true, force: true });
process.exit(failed ? 1 : 0);
