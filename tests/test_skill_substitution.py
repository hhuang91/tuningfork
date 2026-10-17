"""Locating a skill in a payload and substituting a variant.

The fixture mirrors what Hermes 0.18.2 actually sends for a preloaded skill: the full file,
frontmatter included, inlined into the system prompt between two harness-written markers.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from tuningfork.experiment.skill import (
    NothingToCompare,
    SkillNotFound,
    locate,
    split_frontmatter,
    substitute,
    swapped,
    text_sha256,
    unapplied,
)

ORIGINAL = """---
name: tally-report
description: Summarize a CSV.
---

# Tally report

1. Read the CSV.
2. Write report.json.
"""

VARIANT = """---
name: tally-report
description: Summarize a CSV.
---

# Tally report

1. Read the CSV.
2. Write report.json with keys row_count and amount_total.
"""


def hermes_request(skill: str = ORIGINAL, extra_messages: list | None = None) -> dict:
    system = (
        "You are Hermes Agent...\n\n"
        '[IMPORTANT: The user launched this CLI session with the "tally-report" skill '
        "preloaded. Treat its instructions as active guidance.]\n\n"
        f"{skill}\n"
        "[Skill directory: /tmp/run/home/skills/tally-report]\n"
    )
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Use the tally-report skill."},
            *(extra_messages or []),
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }


class TestLocate:
    def test_finds_a_preloaded_skill_in_the_system_prompt(self):
        (span,) = locate(hermes_request(), ORIGINAL)

        assert span.message_index == 0
        assert span.role == "system"
        assert span.form == "full"
        assert span.line_start == 1
        assert span.line_end == ORIGINAL.rstrip("\n").count("\n") + 1

    def test_finds_every_occurrence(self):
        """Preloaded, then viewed again through a tool: both copies must be found."""
        again = {"role": "tool", "tool_call_id": "1", "content": f"Skill:\n{ORIGINAL}"}
        spans = locate(hermes_request(extra_messages=[again]), ORIGINAL)

        assert [span.message_index for span in spans] == [0, 2]

    def test_falls_back_to_the_body_when_frontmatter_was_stripped(self):
        _, body = split_frontmatter(ORIGINAL)
        (span,) = locate(hermes_request(skill=body.strip()), ORIGINAL)

        assert span.form == "body"
        assert span.line_start == 6  # the heading, after the five-line frontmatter block

    def test_crlf_on_disk_still_matches_an_lf_payload(self):
        (span,) = locate(hermes_request(), ORIGINAL.replace("\n", "\r\n"))
        assert span.form == "full"

    def test_a_payload_that_keeps_the_files_crlf_is_matched_and_swapped_in_kind(self):
        """Claude Code 2.1.281 inlines a CRLF skill's body as the file has it (6c, on a
        Windows drive): found there, its lines named, and an edit swapped in with CRLF."""
        crlf = ORIGINAL.replace("\n", "\r\n")
        _, body = split_frontmatter(ORIGINAL)
        request = {"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "user", "content": "Base directory for this skill: /x\n\n" + body.strip().replace("\n", "\r\n") + "\r\n\n\nARGUMENTS: a"},
        ]}
        (span,) = locate(request, crlf)
        assert span.form == "body" and span.line_start == 6
        changed, _ = substitute(request, crlf, VARIANT.replace("\n", "\r\n"))
        swapped = split_frontmatter(VARIANT)[1].strip().replace("\n", "\r\n")
        assert changed["messages"][1]["content"] == "Base directory for this skill: /x\n\n" + swapped + "\r\n\n\nARGUMENTS: a"

    def test_list_content_parts_are_searched(self):
        request = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": f"x\n{ORIGINAL}"}]}
            ]
        }
        (span,) = locate(request, ORIGINAL)
        assert span.part_index == 0

    def test_absent_skill_finds_nothing(self):
        assert locate({"messages": [{"role": "user", "content": "hi"}]}, ORIGINAL) == []


class TestSubstitute:
    def test_only_the_skill_changes(self):
        request = hermes_request()
        changed, _ = substitute(request, ORIGINAL, VARIANT)

        assert "row_count and amount_total" in changed["messages"][0]["content"]
        assert ORIGINAL not in changed["messages"][0]["content"]
        # Everything around the skill is byte-identical.
        before, _, after = request["messages"][0]["content"].partition(ORIGINAL)
        assert changed["messages"][0]["content"] == before + VARIANT + after
        assert changed["messages"][1:] == request["messages"][1:]
        assert changed["tools"] == request["tools"]

    def test_the_original_request_is_not_mutated(self):
        request = hermes_request()
        snapshot = repr(request)
        substitute(request, ORIGINAL, VARIANT)
        assert repr(request) == snapshot

    def test_every_occurrence_is_replaced(self):
        again = {"role": "tool", "tool_call_id": "1", "content": f"Skill:\n{ORIGINAL}"}
        changed, spans = substitute(hermes_request(extra_messages=[again]), ORIGINAL, VARIANT)

        assert len(spans) == 2
        assert all(ORIGINAL not in m["content"] for m in changed["messages"])

    def test_surrounding_newlines_are_preserved_when_the_variant_lacks_one(self):
        request = hermes_request()
        changed, _ = substitute(request, ORIGINAL, VARIANT.rstrip("\n"))
        # The marker after the skill still starts on its own line.
        assert "amount_total.\n\n[Skill directory:" in changed["messages"][0]["content"]

    def test_body_form_substitutes_the_variant_body(self):
        _, body = split_frontmatter(ORIGINAL)
        request = hermes_request(skill=body.strip())
        changed, spans = substitute(request, ORIGINAL, VARIANT)

        assert spans[0].form == "body"
        assert "name: tally-report" not in changed["messages"][0]["content"]
        assert "row_count and amount_total" in changed["messages"][0]["content"]

    def test_edited_file_is_diagnosed_rather_than_guessed(self):
        """The file changed after the run: the heading is there, the full text is not."""
        drifted = ORIGINAL.replace("Read the CSV.", "Read the CSV file carefully.")
        request = hermes_request()

        with pytest.raises(SkillNotFound, match="file changed after this run"):
            substitute(request, drifted, VARIANT)

    def test_skill_not_in_context_is_diagnosed(self):
        with pytest.raises(SkillNotFound, match="No part of the skill"):
            substitute({"messages": [{"role": "user", "content": "hi"}]}, ORIGINAL, VARIANT)


class TestWhereHarnessesPutSkills:
    def test_a_skill_inside_a_json_tool_result_is_found_and_swapped_escaped(self):
        """Hermes's skill_view returns the file inside JSON: newlines and quotes escaped,
        and non-ASCII as \\u escapes. A literal search never matched it."""
        original = ORIGINAL.replace("Read the CSV.", 'Read the "CSV" — carefully.')
        variant = VARIANT.replace("Read the CSV.", 'Read the "CSV" — carefully.')
        viewed = {"role": "tool", "tool_call_id": "1",
                  "content": json.dumps({"success": True, "content": original})}
        request = hermes_request(skill="(not preloaded)", extra_messages=[viewed])

        (span,) = locate(request, original)
        changed, _ = substitute(request, original, variant)

        assert span.encoding == "json"
        decoded = json.loads(changed["messages"][2]["content"])
        assert decoded == {"success": True, "content": variant}

    def test_a_preloaded_copy_and_a_viewed_copy_are_both_swapped(self):
        viewed = {"role": "tool", "tool_call_id": "1",
                  "content": json.dumps({"content": ORIGINAL}, ensure_ascii=False)}
        request = hermes_request(extra_messages=[viewed])

        changed, spans = substitute(request, ORIGINAL, VARIANT)

        assert [span.encoding for span in spans] == ["raw", "json"]
        assert VARIANT in changed["messages"][0]["content"]
        assert json.loads(changed["messages"][2]["content"])["content"] == VARIANT

    def test_an_anthropic_tool_result_is_searched(self):
        """`cat SKILL.md` through a tool: the copy sits inside a tool_result block."""
        request = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": ORIGINAL}]},
        ]}]}
        changed, (span,) = substitute(request, ORIGINAL, VARIANT)
        assert span.part_index == 0
        assert changed["messages"][0]["content"][0]["content"][0]["text"] == VARIANT

    def test_the_models_own_words_are_history_and_left_alone(self):
        quoted = {"role": "assistant", "content": f"The skill says:\n{ORIGINAL}"}
        changed, spans = substitute(hermes_request(extra_messages=[quoted]), ORIGINAL, VARIANT)
        assert [span.message_index for span in spans] == [0]
        assert changed["messages"][2] == quoted

    def test_no_content_key_is_invented(self):
        """Found in review: messages without content came back with `content: None`."""
        call = {"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
                                                     "function": {"name": "x", "arguments": "{}"}}]}
        changed, _ = substitute(hermes_request(extra_messages=[call]), ORIGINAL, VARIANT)
        assert "content" not in changed["messages"][2]


class TestNothingToCompare:
    def test_an_edit_the_model_cannot_see_here_is_refused(self):
        """Claude Code shows the body only: a frontmatter-only edit would compare a request
        with itself, and report "not significant" as though that meant something."""
        _, body = split_frontmatter(ORIGINAL)
        request = hermes_request(skill=body.strip())
        variant = ORIGINAL.replace("description: Summarize a CSV.", "description: Tally a CSV.")

        with pytest.raises(NothingToCompare, match="frontmatter"):
            substitute(request, ORIGINAL, variant)

    def test_an_edit_partly_out_of_sight_is_reported(self):
        _, body = split_frontmatter(ORIGINAL)
        spans = locate(hermes_request(skill=body.strip()), ORIGINAL)
        variant = VARIANT.replace("description: Summarize a CSV.", "description: Tally a CSV.")

        (note,) = unapplied(ORIGINAL, variant, spans)
        assert "frontmatter" in note
        assert unapplied(ORIGINAL, variant, locate(hermes_request(), ORIGINAL)) == []


class TestSwapped:
    """The inverse of `substitute`: an A/B's two requests give back the texts it compared,
    as the model read each, which a report shows as the edit."""

    @staticmethod
    def back(request: dict, original: str = ORIGINAL, variant: str = VARIANT):
        changed, spans = substitute(request, original, variant)
        return swapped(request, changed, [asdict(span) for span in spans])

    def test_a_preloaded_skill_comes_back_whole(self):
        before, after, span = self.back(hermes_request())
        assert (before, after) == (ORIGINAL, VARIANT)
        assert span["form"] == "full" and span["line_start"] == 1

    def test_a_body_comes_back_as_the_body_with_its_place_in_the_file(self):
        _, body = split_frontmatter(ORIGINAL)
        before, after, span = self.back(hermes_request(skill=body.strip()))
        assert before == body.strip()
        assert after == split_frontmatter(VARIANT)[1].strip()
        assert span["form"] == "body" and span["line_start"] == 6

    def test_a_json_escaped_copy_is_read_unescaped(self):
        original = ORIGINAL.replace("Read the CSV.", 'Read the "CSV" — carefully.')
        variant = VARIANT.replace("Read the CSV.", 'Read the "CSV" — carefully.')
        viewed = {"role": "tool", "tool_call_id": "1", "content": json.dumps({"success": True, "content": original})}
        before, after, span = self.back(hermes_request(skill="(not preloaded)", extra_messages=[viewed]), original, variant)
        assert (before, after) == (original, variant)
        assert span["encoding"] == "json"

    def test_two_copies_give_the_first(self):
        viewed = {"role": "tool", "tool_call_id": "1", "content": json.dumps({"content": ORIGINAL})}
        before, after, span = self.back(hermes_request(extra_messages=[viewed]))
        assert (before, after) == (ORIGINAL, VARIANT)
        assert span["message_index"] == 0

    def test_windows_line_endings_come_back_plain(self):
        request = {"model": "m", "messages": [
            {"role": "user", "content": "Skill:\n" + ORIGINAL.replace("\n", "\r\n") + "ARGUMENTS: a"},
        ]}
        before, after, _ = self.back(request, ORIGINAL.replace("\n", "\r\n"), VARIANT.replace("\n", "\r\n"))
        assert (before, after) == (ORIGINAL, VARIANT)

    def test_an_anthropic_tool_result(self):
        request = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": ORIGINAL}]},
        ]}]}
        before, after, _ = self.back(request)
        assert (before, after) == (ORIGINAL, VARIANT)

    def test_nothing_comes_back_where_the_arms_were_sent_the_same_skill(self):
        """The demo's seeded A/B once kept the original's request in both arms."""
        request = hermes_request()
        _, spans = substitute(request, ORIGINAL, VARIANT)
        assert swapped(request, request, [asdict(span) for span in spans]) is None

    def test_nothing_comes_back_where_what_differs_is_not_the_swap(self):
        request = hermes_request()
        changed, spans = substitute(request, ORIGINAL, VARIANT)
        changed["messages"][1]["content"] = "Use another skill."
        unswapped = {**changed, "messages": [request["messages"][0], *changed["messages"][1:]]}
        assert swapped(request, unswapped, [asdict(span) for span in spans]) is None


def test_hash_ignores_line_endings():
    assert text_sha256(ORIGINAL) == text_sha256(ORIGINAL.replace("\n", "\r\n"))
