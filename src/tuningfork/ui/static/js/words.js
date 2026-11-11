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
  "target": "condition (over a result, Condition; a probe's figures, condition met); in a refinement, what the model should do",
  "to the end": "the whole run",
  "check along": "the call behind each line",
  "where it holds": "ask every run",
  "checkpoint": "this step",
  "proposer": "proposing model (the built-in one)",
  "packet": "what the proposing model is shown",
  "screen": "test at this step",
  "validation": "full test runs",
  "gate": "approval checks",
  "server check": "server test",
  "Wilson interval": "95% interval",
  "Fisher's exact test": "beyond chance",
  "Holm correction": "corrected for the tests planned",
  "Bonferroni correction": "each p adjusted for comparing every behaviour, with nothing declared",
};

/** Labels more than one page shows, spelled once. */
export const LABELS = {
  // Whether the condition includes a behaviour, as a badge beside it: never "counted", which
  // already means a run or fork is in the pass rate.
  matches: "matches",
  noMatch: "no match",
  // The legend over a target's choices.
  condition: "Condition",
  // Settings a form keeps folded.
  advanced: "Advanced settings",
  // The two reaches of a resample and an A/B.
  judgeBy: "Measure",
  nextOnly: "The next step",
  wholeRun: "The whole run",
  // A test at one step of a refinement's proposals, and the full runs after it.
  screen: "Test at this step",
  validate: "Full test runs",
  // The server check, so "check" means only a task's.
  testServer: "Test the server",
  // How a statistical result is said before its numbers.
  beyondChance: "more than chance would explain",
  withinChance: "no difference beyond chance",
};
