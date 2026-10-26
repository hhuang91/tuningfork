"""Edits to a skill's text, and the line diff that shows them (`ops.diff`)."""

from __future__ import annotations

import pytest

from tuningfork.config import REDACTED
from tuningfork.ops.common import OperationError
from tuningfork.ops.diff import apply_edits, apply_resolved, line_diff, onto, resolve_edits
from tuningfork.store.masking import Masking
from tests.test_store_masking import PRIVATE

SKILL = "---\nname: tally-report\n---\n\n# Tally report\n\n## Steps\n\n1. Read the CSV file.\n2. Work out the totals.\n3. Write report.json.\n"


class TestEdits:
    def test_each_old_text_is_replaced_once_in_order(self):
        edited = apply_edits(SKILL, [
            {"old": "3. Write report.json.", "new": "3. Write report.json with the keys row_count and amount_total."},
            {"old": "1. Read the CSV file.", "new": "1. Read the CSV file with read_file."},
        ])
        assert "row_count and amount_total" in edited and "with read_file" in edited
        assert edited.count("\n") == SKILL.count("\n")

    def test_an_old_text_that_is_not_there_names_the_closest_lines(self):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [{"old": "3. Write the report.json.", "new": "x"}])
        assert refused.value.field == "edits" and "not in the skill" in refused.value.message
        assert "3. Write report.json." in refused.value.hint

    def test_an_edit_without_its_old_text_is_told_the_shapes_and_a_line_to_copy(self):
        """A 26B sent {new} alone twenty-seven times running, each refusal read as the last."""
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [{"new": "3. Write report.json with the keys row_count and amount_total."}])
        assert refused.value.field == "edits"
        assert refused.value.message.startswith("edit 1 names nothing to change (no line, after or old): it must be")
        assert "{line, new}, {line, through, new}, {after, insert} or {old, new}" in refused.value.message
        assert '{"old": <text copied exactly from the skill>, "new": <its replacement>}' in refused.value.hint
        assert '{"line": <its number, as show_skill prints it>, "new": <that line as it should read>}' in refused.value.hint
        assert "'3. Write report.json.'" in refused.value.hint
        with pytest.raises(OperationError) as neither:
            apply_edits(SKILL, [{"text": "x"}])
        assert neither.value.message.startswith("edit 1 has a key tuningfork does not take (text):")
        assert "closest" not in neither.value.hint
        with pytest.raises(OperationError, match="edit 1 has no new: `old` is the text as it is"):
            apply_edits(SKILL, [{"old": "3. Write report.json."}])

    def test_indentation_alone_is_no_difference(self):
        """What an agent did on the real stack: the example's lines with one space more than the
        skill's, refused eight times. The lines match; the edit takes the skill's indentation."""
        skill = "# Example\n\n   ```json\n   {\n     \"rows\": 3,\n     \"total\": 61.5\n   }\n   ```\n"
        edited = apply_edits(skill, [{"old": "      \"rows\": 3,\n      \"total\": 61.5",
                                     "new": "      \"row_count\": 3,\n      \"amount_total\": 61.5"}])
        assert "   {\n     \"row_count\": 3,\n     \"amount_total\": 61.5\n   }" in edited
        # The new text keeps its own shape, moved to the skill's indentation. (An old text that
        # is an exact substring — the line without its spaces — is replaced in place, as before.)
        nested = apply_edits(skill, [{"old": "      \"rows\": 3,", "new": "      \"row_count\": 3,\n        \"by\": {}"}])
        assert "     \"row_count\": 3,\n       \"by\": {}\n" in nested
        in_place = apply_edits(skill, [{"old": "\"rows\": 3,", "new": "\"row_count\": 3,"}])
        assert "     \"row_count\": 3,\n" in in_place
        with pytest.raises(OperationError, match="occurs 2 times once indentation is ignored"):
            apply_edits("  ab\n  cd\n  ab\n  cd\n", [{"old": "ab\ncd", "new": "x"}])

    def test_the_closest_lines_are_quoted_as_they_stand(self):
        with pytest.raises(OperationError) as refused:
            apply_edits("# Example\n\n     \"rows\": 3,\n", [{"old": "\"rows\": 4,", "new": "x"}])
        assert "'     \"rows\": 3,'" in refused.value.hint and "leading spaces alone" in refused.value.hint

    def test_an_ambiguous_old_text_is_refused(self):
        with pytest.raises(OperationError, match="occurs 2 times"):
            apply_edits("a\nb\na\n", [{"old": "a", "new": "c"}])

    @pytest.mark.parametrize("edit", [
        {"old": "", "new": "x"}, {"old": "  ", "new": "x"}, {"old": "1. Read the CSV file.", "new": "1. Read the CSV file."},
        {"old": "x"}, {"old": "x", "new": "y", "why": "z"}, "replace it", {"old": 1, "new": "y"},
    ])
    def test_a_malformed_edit_is_refused_by_field(self, edit):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [edit])
        assert refused.value.field == "edits"

    def test_no_edits_is_refused(self):
        with pytest.raises(OperationError):
            apply_edits(SKILL, [])

    def test_a_windows_file_takes_plain_line_endings(self):
        windows = SKILL.replace("\n", "\r\n")
        edited = apply_edits(windows, [{"old": "2. Work out the totals.\n3. Write", "new": "2. Count the rows.\n3. Write"}])
        assert "2. Count the rows.\r\n3. Write" in edited and "\n" not in edited.replace("\r\n", "")

    def test_a_windows_file_keeps_its_endings_where_indentation_is_ignored(self):
        """Found in review: the lines matched once their indentation was ignored kept the last
        one's carriage return as text, so its replacement ended that line in a bare line feed."""
        windows = "# Skill\r\n\r\n  1. Read the CSV.\r\n  2. Write report.json.\r\n  3. Reply.\r\n"
        edited = apply_edits(windows, [{"old": "1. Read the CSV.\n2. Write report.json.", "new": "1. Read the CSV.\n2. Write it."}])
        assert edited == "# Skill\r\n\r\n  1. Read the CSV.\r\n  2. Write it.\r\n  3. Reply.\r\n"


class TestDiff:
    def test_a_reworded_line_counts_once_with_its_numbers(self):
        after = SKILL.replace("3. Write report.json.", "3. Write report.json with row_count.")
        diff = line_diff(SKILL, after)
        assert (diff["added"], diff["removed"], diff["changed"]) == (1, 1, 1)
        (hunk,) = diff["hunks"]
        changed = [line for line in hunk["lines"] if line["op"] != " "]
        assert changed == [
            {"op": "-", "old": 11, "text": "3. Write report.json."},
            {"op": "+", "new": 11, "text": "3. Write report.json with row_count."},
        ]
        assert all(line["op"] == " " for line in hunk["lines"][:2])  # its context

    def test_added_lines_and_far_apart_changes(self):
        after = SKILL.replace("# Tally report\n", "# Tally report\n\nUse the keys row_count and amount_total.\n")
        after = after.replace("3. Write report.json.", "3. Write report.json, nothing else.")
        diff = line_diff(SKILL, after)
        assert diff["added"] == 3 and diff["removed"] == 1 and diff["changed"] == 3
        # Four unchanged lines apart: one hunk with two lines of context, two with one.
        assert len(diff["hunks"]) == 1 and len(line_diff(SKILL, after, context=1)["hunks"]) == 2

    def test_no_change_is_no_hunk(self):
        assert line_diff(SKILL, SKILL) == {"hunks": [], "added": 0, "removed": 0, "changed": 0}


def test_an_edit_whose_new_text_is_its_old_text_is_told_what_new_is():
    """A 26B sent the table it meant to change, unchanged, eleven times running."""
    from tuningfork.ops.diff import apply_edits

    table = "| Field | Rule |\n|---|---|\n| owner | Explicit named owner |\n"
    with pytest.raises(OperationError) as caught:
        apply_edits("# Skill\n\n" + table, [{"old": table, "new": table}])
    message = str(caught.value)
    assert "edit 1 changes nothing" in message and "its `new` is its `old` word for word" in message
    assert "as they should read after the change" in message


class TestLineEdits:
    """An edit that names a line by its number, as show_skill prints it: nothing to copy."""

    def test_a_line_named_by_number_is_replaced_as_the_skill_view_numbers_it(self):
        from tuningfork.ops import compact

        shown = compact.skill({"text": SKILL})["lines"]
        assert shown[10] == "11: 3. Write report.json."
        wanted = "3. Write report.json with the keys row_count and amount_total."
        by_line = apply_edits(SKILL, [{"line": 11, "new": wanted}])
        assert by_line == apply_edits(SKILL, [{"old": "3. Write report.json.", "new": wanted}])
        # A number sent as a string is a number.
        assert apply_edits(SKILL, [{"line": "11", "new": "x"}]).endswith("2. Work out the totals.\nx\n")

    def test_a_range_of_lines_is_replaced_by_as_many_lines_as_new_holds(self):
        edited = apply_edits(SKILL, [{"line": 9, "through": 10,
                                      "new": "1. Read the CSV file with read_file.\n2. List the rows.\n3. Count them."}])
        assert "## Steps\n\n1. Read the CSV file with read_file.\n2. List the rows.\n3. Count them.\n3. Write report.json.\n" in edited
        assert apply_edits(SKILL, [{"line": 10, "new": ""}]) == SKILL.replace("2. Work out the totals.\n", "")

    def test_lines_are_added_after_a_line(self):
        assert apply_edits(SKILL, [{"after": 11, "insert": "4. Check the counts against the input."}]) == (
            SKILL + "4. Check the counts against the input.\n")
        assert "---\n\n> for the tally team\n# Tally report" in apply_edits(SKILL, [{"after": 4, "insert": "> for the tally team"}])
        assert apply_edits(SKILL.rstrip("\n"), [{"after": 11, "insert": "4. Check."}]).endswith("3. Write report.json.\n4. Check.\n")
        assert apply_edits(SKILL, [{"after": 0, "insert": "<!-- draft -->"}]).startswith("<!-- draft -->\n---\n")

    def test_every_edit_names_the_text_as_shown_whatever_the_others_do(self):
        edited = apply_edits(SKILL, [
            {"after": 8, "insert": "0. Open the folder."},
            {"line": 11, "new": "3. Write report.json with the keys row_count and amount_total."},
            {"old": "2. Work out the totals.", "new": "2. Count the rows."},
        ])
        assert "## Steps\n\n0. Open the folder.\n1. Read the CSV file.\n2. Count the rows.\n3. Write report.json with the keys" in edited

    def test_edits_that_touch_the_same_lines_are_refused(self):
        with pytest.raises(OperationError, match="edits 1 and 2 touch the same lines"):
            apply_edits(SKILL, [{"line": 9, "through": 10, "new": "x"}, {"line": 10, "new": "y"}])
        with pytest.raises(OperationError, match="touch the same lines"):
            apply_edits(SKILL, [{"old": "2. Work out the totals.", "new": "x"}, {"line": 10, "new": "y"}])
        with pytest.raises(OperationError, match="touch the same lines"):
            apply_edits(SKILL, [{"after": 9, "insert": "a"}, {"after": 9, "insert": "b"}])
        # Adding before a line and replacing that line are two places.
        both = apply_edits(SKILL, [{"after": 10, "insert": "2b. Check."}, {"line": 11, "new": "3. Write."}])
        assert both.endswith("2. Work out the totals.\n2b. Check.\n3. Write.\n")

    def test_a_line_the_skill_does_not_have_is_refused_with_the_count(self):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [{"line": 40, "new": "x"}])
        assert "edit 1 names line 40: the skill has 11 lines" in refused.value.message
        with pytest.raises(OperationError, match=r"`through` \(9\) is before `line` \(10\)"):
            apply_edits(SKILL, [{"line": 10, "through": 9, "new": "x"}])
        with pytest.raises(OperationError, match="is a line number"):
            apply_edits(SKILL, [{"line": "eleven", "new": "x"}])
        with pytest.raises(OperationError, match="names line 0"):
            apply_edits(SKILL, [{"line": 0, "new": "x"}])

    def test_a_line_rewritten_as_it_already_reads_changes_nothing(self):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [{"line": 11, "new": "3. Write report.json."}])
        assert "edit 1 changes nothing: line 11 already reads '3. Write report.json.'" in refused.value.message
        assert "as it should read after the change" in refused.value.message

    def test_a_shape_that_is_none_of_the_four_is_told_them(self):
        with pytest.raises(OperationError) as mixed:
            apply_edits(SKILL, [{"line": 11, "insert": "x"}])
        assert mixed.value.message.startswith("edit 1 mixes shapes: it must be {line, new}, {line, through, new}")
        with pytest.raises(OperationError, match="adds nothing"):
            apply_edits(SKILL, [{"after": 11, "insert": "  "}])
        with pytest.raises(OperationError, match="edit 1 has no new: line 11 as it should read"):
            apply_edits(SKILL, [{"line": 11}])

    def test_a_windows_file_keeps_its_endings_under_a_line_edit(self):
        windows = SKILL.replace("\n", "\r\n")
        edited = apply_edits(windows, [{"line": 10, "new": "2. Count the rows.\n2b. Check them."},
                                       {"after": 11, "insert": "4. Done."}])
        assert "\n" not in edited.replace("\r\n", "")
        assert edited.endswith("2. Count the rows.\r\n2b. Check them.\r\n3. Write report.json.\r\n4. Done.\r\n")

    def test_what_an_edit_resolved_to_is_said(self):
        from tuningfork.ops.diff import resolve_edits

        first, second = resolve_edits(SKILL, [{"line": 11, "new": "3. Write it."}, {"after": 11, "insert": "4. Check."}])
        assert (first["lines"], first["old"], first["new"]) == ("line 11", "3. Write report.json.", "3. Write it.")
        assert (second["lines"], second["old"], second["new"]) == ("after line 11", "", "4. Check.")
        assert first["given"] == {"line": 11, "new": "3. Write it."}
        (pair,) = resolve_edits(SKILL, [{"old": "1. Read the CSV file.\n2. Work out the totals.", "new": "1. Read.\n2. Count."}])
        assert pair["lines"] == "lines 9–10"


def test_line_breaks_escaped_twice_are_line_breaks():
    """A 26B sent a two-line insert with the two characters backslash-n between the lines and no
    line break, and it went into the skill as one line."""
    from tuningfork.ops.diff import apply_edits

    twice = apply_edits(SKILL, [{"after": 11, "insert": "4. Check the counts.\\n5. Write the report."}])
    assert twice.endswith("3. Write report.json.\n4. Check the counts.\n5. Write the report.\n")
    pair = apply_edits(SKILL, [{"old": "3. Write report.json.", "new": "3. Write report.json.\\n4. Check it."}])
    assert pair.endswith("3. Write report.json.\n4. Check it.\n")
    # With a line break of its own, backslash-n is text the skill may well hold.
    kept = apply_edits(SKILL, [{"line": 11, "new": "3. Write report.json with lines joined by \\\\n.\nDone."}])
    assert "joined by \\\\n.\nDone." in kept



KEY_HELD = "sk-a1B2c3D4e5F6g7H8i9J0"


class TestAMaskedLine:
    """What an agent or a proposing model reads of a skill is masked, each line at its number, each
    line a mask covers reading `<redacted>`, while the skill's file keeps what it holds: an edit is
    resolved against the file and quoted as the reader was shown it (`resolve_edits`' `shown`), never
    writes the mask, and changes a key whole or not at all (`masks`)."""

    KEYED = SKILL + f"4. Log in with {KEY_HELD}.\n"
    #: A password named on line 12 and said bare on line 13, then a private key, lines 14–17.
    NAMED = (SKILL + "4. Log in with password=Tr0ub4dor3xyz first.\n5. If asked again, answer Tr0ub4dor3xyz.\n"
             + "\n".join(PRIVATE) + "\n")

    @staticmethod
    def resolved(text: str, edits: list) -> list:
        masking = Masking()
        return resolve_edits(text, edits, shown=masking.shown(text, lines=True).value, masks=masking.spans(text))

    def test_an_edit_never_writes_the_mask_into_the_skill(self):
        """Line 12 is shown as `4. Log in with <redacted>.`: rewritten as it was shown, the key the
        skill holds would be lost to the mask in its file. No edit writes it — an insert, or one
        beside the key, either: a key's line moved so would read the mask where the key was (found
        in review). A skill that should say it is a person's to write."""
        for edit in ({"line": 12, "new": f"4. Log in with {REDACTED}, then check."},
                     {"line": 11, "through": 12, "new": f"3. Write report.json.\n4. Log in with {REDACTED}."},
                     {"after": 12, "insert": f"5. Log in with {REDACTED} again."},
                     {"old": "4. Log in with", "new": f"4. Mask as {REDACTED}, then log in with"}):
            with pytest.raises(OperationError) as refused:
                self.resolved(self.KEYED, [edit])
            assert refused.value.field == "edits" and f"would write {REDACTED} into the skill" in refused.value.message
        kept = apply_resolved(self.KEYED, self.resolved(self.KEYED, [{"line": 11, "new": "3. Write report.json, checked."}]))
        assert KEY_HELD in kept
        held = SKILL + f"4. Logs show {REDACTED} where a key was.\n"
        edited = apply_resolved(held, self.resolved(held, [{"line": 12, "new": f"4. Logs show {REDACTED} where it was."}]))
        assert edited.endswith(f"4. Logs show {REDACTED} where it was.\n")  # the mask the skill held, kept

    def test_a_key_is_changed_whole_or_not_at_all(self):
        """A private key's lines each read `<redacted>`: an edit of some of them — its first line
        rewritten, a line taken, one put between — would leave a broken key in the file (found in
        review). All of them, or the lines around them, may change."""
        for edit in ({"line": 14, "new": "The deploy key is kept in the vault."},
                     {"line": 15, "through": 16, "new": ""}, {"after": 15, "insert": "(the deploy key)"},
                     {"line": 15, "new": "A line."}):
            with pytest.raises(OperationError) as refused:
                self.resolved(self.NAMED, [edit])
            assert f"changes part of what is shown as {REDACTED}" in refused.value.message, edit
        gone = apply_resolved(self.NAMED, self.resolved(self.NAMED, [{"line": 14, "through": 17, "new": "Keep keys elsewhere."}]))
        assert PRIVATE[1] not in gone and gone.endswith("Keep keys elsewhere.\n")
        around = apply_resolved(self.NAMED, self.resolved(self.NAMED, [{"after": 17, "insert": "6. Done."}]))
        assert around.endswith(PRIVATE[-1] + "\n6. Done.\n")

    def test_each_edit_and_each_refusal_quotes_the_skill_as_it_was_shown(self):
        """Quoted as the file holds it, a line saying bare a password another line names shows it
        (found in review): each quote is the line as show_skill showed it; the edit is still resolved
        against the file."""
        (resolved,) = self.resolved(self.NAMED, [{"line": 13, "new": "5. If asked again, answer it."}])
        assert resolved["shown"] == f"5. If asked again, answer {REDACTED}." and "Tr0ub4dor3xyz" in resolved["old"]
        with pytest.raises(OperationError) as closest:
            self.resolved(self.NAMED, [{"old": "5. If asked again, answer it.", "new": "x"}])
        assert f"answer {REDACTED}." in closest.value.hint and "Tr0ub4dor3xyz" not in closest.value.hint
        with pytest.raises(OperationError) as same:
            self.resolved(self.NAMED, [{"line": 13, "new": "5. If asked again, answer Tr0ub4dor3xyz."}])
        assert "changes nothing" in same.value.message and "Tr0ub4dor3xyz" not in same.value.message

    def test_an_old_text_copied_with_the_mask_is_told_to_name_its_line(self):
        with pytest.raises(OperationError) as refused:
            self.resolved(self.KEYED, [{"old": f"4. Log in with {REDACTED}.", "new": "4. Log in."}])
        assert "not in the skill" in refused.value.message and "name the line by its number" in refused.value.hint


class TestATextWrittenFromWhatWasShown:
    """A whole text written from the skill as it was shown — masked, each line at its number — is set
    on the file as it stands (`onto`)."""

    TEXT = SKILL + f"4. Log in with {KEY_HELD}.\n" + "\n".join(PRIVATE) + "\n5. Then send it.\n"

    def test_the_lines_it_kept_are_the_file_s_a_key_with_them(self):
        shown = Masking().shown(self.TEXT, lines=True).value
        assert KEY_HELD not in shown and PRIVATE[1] not in shown
        edited, taken = onto(self.TEXT, shown, shown.replace("5. Then send it.", "5. Then send it once."))
        assert edited == self.TEXT.replace("5. Then send it.", "5. Then send it once.")
        assert taken == set(range(len(self.TEXT.splitlines()) - 1))  # every line but the one changed
        rewrote, _ = onto(self.TEXT, shown, shown.replace(f"4. Log in with {REDACTED}.", f"4. Log in first with {REDACTED}."))
        assert f"4. Log in first with {REDACTED}." in rewrote and PRIVATE[1] in rewrote

    def test_a_text_written_from_the_file_comes_back_as_it_was(self):
        shown = Masking().shown(self.TEXT, lines=True).value
        written = self.TEXT.replace("5. Then send it.", "5. Then send it twice.")
        assert onto(self.TEXT, shown, written)[0] == written

    def test_a_line_it_wrote_takes_the_file_s_line_breaks(self):
        """Found in review: a file of Windows line breaks came back with a plain one on the line
        changed."""
        crlf = self.TEXT.replace("\n", "\r\n")
        shown = Masking().shown(crlf, lines=True).value
        written = shown.replace("\r\n", "\n").replace("5. Then send it.", "5. Then send it once.")
        assert onto(crlf, shown, written)[0] == crlf.replace("5. Then send it.", "5. Then send it once.")
