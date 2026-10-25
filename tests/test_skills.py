"""The skills the recordings know (`ops.skills`): each version a fingerprint of the files its
runs were given, with its SKILL.md, its pass rates and what its text is known as; the skill's
own folder now, with the changelog and old versions an approval keeps beside it; any two of
its texts, line by line. And tuningfork's words, defined once (`ops.glossary`).

Asked for by the user: a page that renders a skill's SKILL.md for a quick review (2026-09-28),
with its versions and diffs; and a terminology look-up table (2026-10-05).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tuningfork.ops import approve, glossary, skills
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tuningfork.ops.variants import save_variant
from tests.helpers import VARIANT
from tests.test_approve import stack, validated  # noqa: F401 — `stack` is the fixture

API = "/_tuningfork/api"


def batch(stack, *, n: int = 2, skill_file: Path | None = None):
    plan = prepare_batch(stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "batches",
                         model="m", harness="fake", n=n, skill_file=skill_file)
    return execute_batch(stack.store, plan)


def two_versions(stack):
    """The skill as it is, run twice — the model writes the wrong keys — and the keys fix,
    saved as a variant and run twice: two versions, each with its pass rate."""
    batch(stack)
    batch(stack, skill_file=save_variant(stack.store.home, "tally-report", "keys", VARIANT))
    return skills.skill(stack.store, "tally-report")


class TestVersions:
    def test_each_has_its_text_its_pass_rates_and_what_it_is_known_as(self, stack):
        found = two_versions(stack)
        original = stack.skill.read_text(encoding="utf-8")

        newest, oldest = found.versions
        assert (newest.text, oldest.text) == (VARIANT, original)
        assert newest.known_as == ["saved variant keys"] and oldest.known_as == ["its own file now"]
        (fixed,), (unfixed,) = newest.conditions, oldest.conditions
        assert (fixed["passed"]["successes"], fixed["passed"]["total"]) == (2, 2)
        assert unfixed["passed"]["successes"] == 0
        assert fixed["passed"]["low"] < 1  # a rate is never without its interval
        assert set(newest.files) == set(oldest.files) and newest.files != oldest.files

        (folder,) = found.folders
        assert folder.path == stack.skill.parent.resolve() and folder.version == oldest.fingerprint
        assert folder.changelog is None and folder.history == []
        assert found.variants[0]["name"] == "keys" and found.variants[0]["version"] == newest.fingerprint
        assert found.tasks == ["tally-report/1"]

        (summary,) = skills.skills(stack.store)
        assert (summary.name, summary.runs, summary.versions, summary.variants) == ("tally-report", 4, 2, 1)
        assert summary.folders == [str(stack.skill.parent.resolve())]

    def test_any_two_texts_line_by_line(self, stack):
        found = two_versions(stack)
        newest, oldest = found.versions
        diff = skills.skill_diff(stack.store, "tally-report", oldest.fingerprint[:6], "variant:keys").to_json()
        assert diff["before"]["label"] == f"version {oldest.fingerprint}" and diff["after"]["label"] == "variant keys"
        assert diff["changed"] == 1
        assert [line["text"] for hunk in diff["hunks"] for line in hunk["lines"] if line["op"] == "+"] == [VARIANT.splitlines()[-1]]
        assert skills.skill_diff(stack.store, "tally-report", "file", oldest.fingerprint).to_json()["hunks"] == []
        with pytest.raises(OperationError, match="no single version"):
            skills.skill_diff(stack.store, "tally-report", "nope", "file")
        with pytest.raises(OperationError) as unknown:
            skills.skill(stack.store, "no-such-skill")
        assert unknown.value.code == "not_found"

    def test_an_approval_shows_what_it_keeps_beside_the_folder(self, stack):
        made, proposal = validated(stack)
        store = stack.store
        plan = approve.prepare_approve(store, made.id, proposal.id)
        approve.execute_approve(store, plan, confirm=plan.code, front_end="cli")

        found = skills.skill(store, "tally-report")
        (folder,) = found.folders
        assert folder.version is None and "row_count" in folder.text  # written back, not run yet
        assert folder.changelog and "row_count" in folder.changelog
        (old,) = folder.history
        (version,) = found.versions
        assert version.known_as == [f"replaced by an approval: {old['name']}"]
        assert [(each["id"], each["status"]) for each in found.refinements] == [(made.id, "approved")]
        (variant,) = found.variants
        assert variant["proposal"] == {"refinement": made.id[:8], "proposal": proposal.id[:8]}
        assert skills.skill_diff(store, "tally-report", f"history:{old['name']}", "file").to_json()["changed"] == 1

    def test_a_refinement_in_the_trash_takes_its_proposals_with_it(self, stack):
        """Found in use: a refinement set aside still listed its proposals' texts as the
        skill's variants, and named versions after them."""
        from tuningfork.ops import trash

        made, proposal = validated(stack)
        store = stack.store
        (variant,) = skills.skill(store, "tally-report").variants
        assert variant["name"] == f"r{made.id[:8]}-{proposal.id[:8]}"
        batch(stack, n=1, skill_file=Path(variant["path"]))  # a version whose text is the proposal's
        named = f"proposal {proposal.id[:8]} of refinement {made.id[:8]}"
        assert any(named in version.known_as for version in skills.skill(store, "tally-report").versions)

        trash.trash(store, "refinement", made.id)
        found = skills.skill(store, "tally-report")
        assert found.variants == [] and found.refinements == []
        assert not any(named in version.known_as for version in found.versions)
        (summary,) = skills.skills(store)
        assert summary.variants == 0 and summary.refinements == {"open": 0, "approved": 0, "closed": 0}

        trash.restore(store, "refinement", made.id)
        assert len(skills.skill(store, "tally-report").variants) == 1

    def test_a_chat_imported_last_leaves_the_folder_its_runs_were_launched_from(self, stack):
        """Found in use: a skill's newest run, a chat imported, named no spec, and the Skills
        page — and a new scenario's list of skills — lost the folder its batches ran from."""
        from tuningfork.ops import imports
        from tests.test_chat_import import SKILL, conversation, export

        batch(stack, n=1)
        imports.execute_import(stack.store, imports.prepare_import(stack.store, export(conversation(("tally-report", SKILL)))))
        newest = max(stack.store.runs(limit=10), key=lambda run: run.started_at)
        assert newest.parity.get("imported") and newest.skill_version
        (summary,) = skills.skills(stack.store)
        assert summary.runs == 2 and summary.folders == [str(stack.skill.parent.resolve())]
        assert [folder.path for folder in skills.skill(stack.store, "tally-report").folders] == [stack.skill.parent.resolve()]

    def test_a_skill_known_only_from_a_scenario(self, stack):
        folder = stack.tmp / "scenario-skill" / "other"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text("---\nname: other\n---\n\nDo the other thing.\n", encoding="utf-8")
        scenario = stack.store.home / "scenarios" / "another"
        scenario.mkdir(parents=True)
        (scenario / "evals.json").write_text(json.dumps(
            {"skill_name": "other", "tuningfork": {"skill_dir": str(folder)}, "evals": []}), encoding="utf-8")

        listed = {summary.name: summary for summary in skills.skills(stack.store)}
        assert listed["other"].runs == 0 and listed["other"].scenarios == ["another"]
        assert listed["other"].folders == [str(folder.resolve())]
        found = skills.skill(stack.store, "other")
        assert found.versions == [] and found.folders[0].text.endswith("Do the other thing.\n")
        # The scenarios that run it: the Skills page's way to run it again.
        assert found.scenarios == ["another"] and found.to_json()["scenarios"] == ["another"]


class TestSkillOf:
    """A run's skill is named by the folder its top SKILL.md was recorded in."""

    @staticmethod
    def run(paths: list[str], task: str = "t/1"):
        from types import SimpleNamespace

        return SimpleNamespace(skill_hashes={path: "h" for path in paths}, task_id=task)

    def test_a_category_folder_above_the_skill_is_not_the_skill(self):
        """Found on a Hermes launch laid out as installed: the skill in its category folder
        was listed as a skill named "productivity", and the xlsx skill had no versions."""
        installed = self.run(["productivity/xlsx/SKILL.md", "productivity/xlsx/scripts/x.py",
                              "productivity/xlsx/references/pkg/SKILL.md"])
        assert skills.skill_of(installed) == "xlsx"

    def test_a_skill_laid_out_flat_is_its_own_folder(self):
        assert skills.skill_of(self.run(["tally-report/SKILL.md", "tally-report/notes.md"])) == "tally-report"
        assert skills.skill_of(self.run(["tally-report/notes.md"], task="tally-report/1")) == "tally-report"


class TestFrontEnds:
    def test_from_the_command_line(self, stack):
        from tuningfork.cli import app as cli

        found = two_versions(stack)
        newest, oldest = found.versions
        home = ["--home", str(stack.store.home)]
        runner = CliRunner()
        listed = runner.invoke(cli, ["skills", *home])
        assert listed.exit_code == 0 and "tally-report" in listed.output, listed.output
        shown = runner.invoke(cli, ["skill", "tally-report", *home])
        flat = " ".join(shown.output.split())
        assert shown.exit_code == 0 and newest.fingerprint in flat and "saved variant keys" in flat, shown.output
        assert "2/2 100%" in flat and f"its SKILL.md is version {oldest.fingerprint}" in flat
        text = runner.invoke(cli, ["skill", "tally-report", "--text", newest.fingerprint, *home])
        assert text.output == VARIANT, text.output
        compared = runner.invoke(cli, ["skill", "tally-report", "--diff", oldest.fingerprint, "variant:keys", *home])
        assert compared.exit_code == 0 and "row_count" in compared.output, compared.output
        refused = runner.invoke(cli, ["skill", "nope", *home])
        assert refused.exit_code != 0 and "no skill 'nope'" in refused.output

    def test_over_http(self, stack):
        found = two_versions(stack)
        newest, oldest = found.versions
        with httpx.Client(base_url=stack.url, timeout=30) as http:
            assert [each["name"] for each in http.get(f"{API}/skills").json()] == ["tally-report"]
            detail = http.get(f"{API}/skills/tally-report").json()
            assert [version["text"] for version in detail["versions"]] == [VARIANT, stack.skill.read_text(encoding="utf-8")]
            assert detail["folders"][0]["windows"] is None or detail["folders"][0]["windows"]
            diff = http.get(f"{API}/skills/tally-report/diff", params={"before": oldest.fingerprint, "after": newest.fingerprint})
            assert diff.status_code == 200 and diff.json()["changed"] == 1
            assert http.get(f"{API}/skills/nope").status_code == 404
            words = http.get(f"{API}/glossary").json()
            assert {group["id"] for group in words["groups"]} == {term["group"] for term in words["terms"]}


class TestGlossary:
    def test_each_word_is_defined_once_and_points_at_words_that_are(self):
        names = [term.term for term in glossary.TERMS]
        assert len(names) == len(set(name.lower() for name in names))
        others = [aka.lower() for term in glossary.TERMS for aka in term.aka]
        assert not set(others) & {name.lower() for name in names} and len(others) == len(set(others))
        groups = {key for key, _ in glossary.GROUPS}
        for term in glossary.TERMS:
            assert term.group in groups and set(term.see) <= set(names), term.term
            assert term.means.endswith((".", ".\"")) and term.means[0].isupper() or term.means[0] == "\"", term.term

    def test_the_sliding_window_recommendation_is_found_by_either_flag(self):
        # A strong recommendation, measured: looked up by the flag a person types. Either does,
        # and checkpoints off cost no memory, so it comes first (docs/setup.md).
        for flag in ("--swa-full", "--ctx-checkpoints"):
            found = glossary.lookup(flag)[0]
            assert found.term == "sliding-window cache" and found.means.startswith("Strongly recommended"), flag
            assert "--ctx-checkpoints 0, or with --swa-full" in found.means

    def test_looked_up_by_its_name_first(self):
        assert [term.term for term in glossary.lookup("evals.json")] == ["spec"]
        found = [term.term for term in glossary.lookup("fork")]
        assert found[0] == "fork" and "proxy" not in found  # not every mention of tuningfork
        with pytest.raises(OperationError) as missing:
            glossary.lookup("zzz")
        assert missing.value.code == "not_found"

    def test_from_the_command_line(self):
        from tuningfork.cli import app as cli

        found = CliRunner().invoke(cli, ["glossary", "screen"])
        assert found.exit_code == 0 and found.output.startswith("screen"), found.output
        every = CliRunner().invoke(cli, ["glossary"])
        assert all(title in every.output for _, title in glossary.GROUPS)
