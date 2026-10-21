// The pages' words for tuningfork's terms, where they say a thing more plainly than the CLI,
// MCP and the docs do (the redesign's D2: the precise words stay there, and on the Words page
// beside these). Each phrase is a glossary term's `ui` (`ops/glossary.py`): `tests/test_words.py`
// holds this map and the glossary to each other.
//
// Text the pages write uses these phrases. A sentence the server writes — a refusal, a plan's
// warning, a reason something is greyed — keeps the precise words: it is the sentence the
// terminal prints too. How the pages' sentences are written is in CLAUDE.md (Conventions).

/** A glossary term (its precise word) → the phrase the pages use for it. */
export const UI_WORDS = {
  "normalizer": "how behaviours are grouped",
  "spec": "task file",
  "launch": "a run tuningfork started",
  "skill version": "version 1, 2, … in order of first run, on the Skills page",
  "arm": "wording",
  "target": "what you count; in a refinement, what the model should do",
  "to the end": "whether the whole run then passes",
  "checkpoint": "this step",
  "proposer": "proposing model (the built-in one)",
  "packet": "what the proposing model is shown",
  "screen": "test at this step",
  "validation": "full test runs",
  "gate": "approval checks",
  "server check": "server test",
  "condition": "set-up",
  "Wilson interval": "95% interval",
  "Fisher's exact test": "beyond chance",
  "Holm correction": "corrected for the tests planned",
  "Bonferroni correction": "each p adjusted for comparing every behaviour, with nothing declared",
};

/** Labels more than one page shows, spelled once. */
export const LABELS = {
  // Whether what you count includes a behaviour, as a badge beside it: never "counted", which
  // already means a run or fork is in the pass rate.
  matches: "matches",
  noMatch: "no match",
  // The legend over a target's choices.
  whatToCount: "What to count",
  // Settings a form keeps folded.
  advanced: "Advanced settings",
  // The two reaches of a resample and an A/B.
  judgeBy: "Judge each try by:",
  nextOnly: "what the model does next",
  wholeRun: "whether the whole run then passes",
  // A test at one step of a refinement's proposals, and the full runs after it.
  screen: "Test at this step",
  validate: "Full test runs",
  // The server check, so "check" means only a task's.
  testServer: "Test the server",
  // How a statistical result is said before its numbers.
  beyondChance: "more than chance would explain",
  withinChance: "no difference beyond chance",
};
