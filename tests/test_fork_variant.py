"""A wording swapped into a fork's live calls (`replay.Variant`), and when the original still
reached the model — a leak, which takes the fork out of its count. Found on the real stack
(6c): Gemma quoted its skill while reasoning before the step, and every fork testing an edit
of a quoted line was left out, though the harness had sent only the variant."""

from __future__ import annotations

from tuningfork.replay import Variant

ORIGINAL = "---\nname: tally\n---\n\n# Tally\n\n1. Read the CSV.\n3. Write the results to report.json in the working directory.\n"
VARIANT = ORIGINAL.replace("in the working directory.", "in the working directory, with the keys row_count and amount_total.")
LINE = "3. Write the results to report.json in the working directory."


def request(*extra: dict) -> dict:
    return {"model": "m", "messages": [
        {"role": "user", "content": "go"},
        {"role": "user", "content": "Base directory for this skill: /x\n\n" + ORIGINAL.split("---\n\n", 1)[1]},
        *extra,
    ]}


def test_a_model_that_quoted_its_skill_before_the_step_is_history_not_a_leak():
    quoted = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": f"The skill instructions are:\n1. Read the CSV.\n{LINE}"},
        {"type": "text", "text": f"I will follow: {LINE}"},
    ]}
    sent, info = Variant(ORIGINAL, VARIANT).apply(request(quoted))
    assert info["spans"] == 1 and "leaked" not in info
    assert "row_count and amount_total" in sent["messages"][1]["content"]


def test_a_line_of_the_original_the_harness_sent_again_is_a_leak():
    for again in (
        {"role": "tool", "tool_call_id": "c1", "content": f"SKILL.md:3: {LINE}"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": f"grep: {LINE}"}]},
    ):
        _, info = Variant(ORIGINAL, VARIANT).apply(request(again))
        assert info["leaked"] == [LINE], again


def test_the_original_in_the_system_prompt_is_a_leak_when_it_is_not_swapped():
    anthropic = {**request(), "system": [{"type": "text", "text": f"Remember: {LINE}"}]}
    _, info = Variant(ORIGINAL, VARIANT).apply(anthropic)
    assert info["leaked"] == [LINE]


def test_a_line_the_variant_extends_is_not_a_leak_of_itself():
    """Found on the real stack: an edit made "- explicit commitments" into "- explicit
    commitments (including those found in questions, blockers, or risks)", and every one of
    the screen's twenty forks was marked unusable — the original's line found inside the
    variant's own — so the proposal got no verdict at all."""
    from tuningfork.replay import leaked, unique_lines

    original = "---\nname: m\n---\n\nExtract into distinct lists:\n\n- decisions actually made\n- explicit commitments\n"
    variant = original.replace("- explicit commitments", "- explicit commitments (including those found in questions, blockers, or risks)")
    assert unique_lines(original, variant) == []
    sent, info = Variant(original, variant).apply({"model": "m", "messages": [
        {"role": "user", "content": "go"},
        {"role": "user", "content": "Base directory for this skill: /x\n\n" + original.split("---\n\n", 1)[1]},
    ]})
    assert info["spans"] == 1 and "leaked" not in info
    # A line reworded rather than extended still tells a leak when the harness sends it again.
    reworded = original.replace("- explicit commitments", "- commitments stated in so many words")
    assert unique_lines(original, reworded) == ["- explicit commitments"]
    assert leaked({"messages": [{"role": "tool", "content": "SKILL.md:8: - explicit commitments"}]},
                  unique_lines(original, reworded)) == ["- explicit commitments"]

