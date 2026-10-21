"""The web UI's words and the glossary's agree: each glossary term the pages call something
plainer has that phrase in `ui/static/js/words.js` too, and the map names no term the glossary
does not define (`ops/glossary.py`, a term's `ui`)."""

from __future__ import annotations

import re
from pathlib import Path

from tuningfork.ops import glossary

WORDS = Path(__file__).resolve().parents[1] / "src" / "tuningfork" / "ui" / "static" / "js" / "words.js"


def ui_words() -> dict[str, str]:
    text = WORDS.read_text(encoding="utf-8")
    block = re.search(r"export const UI_WORDS = \{(.*?)\n\};", text, re.DOTALL)
    assert block, "words.js keeps its map as `export const UI_WORDS = { … };`"
    pairs = re.findall(r'^\s*"([^"]+)":\s*"([^"]+)",\s*$', block.group(1), re.MULTILINE)
    assert len(pairs) == len([line for line in block.group(1).splitlines() if line.strip()]), "one pair a line"
    return dict(pairs)


def test_the_pages_words_are_the_glossarys():
    pages = ui_words()
    defined = {term.term: term.ui for term in glossary.TERMS if term.ui}
    assert pages == defined


def test_the_words_page_is_given_both():
    found = glossary.lookup("arm")[0].to_json()
    assert (found["term"], found["ui"]) == ("arm", "wording")


def test_a_page_phrase_finds_its_term():
    # `tuningfork glossary "full test runs"`: a word read on a page leads to the precise one.
    assert glossary.lookup("full test runs")[0].term == "validation"
    assert glossary.lookup("Wording")[0].term == "arm"
