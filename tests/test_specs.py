"""Task specs as files: what tuningfork writes back is what it read, and a spec with a
problem is refused — every problem named — before anything runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuningfork.runner import eval_record
from tuningfork.specs import Eval, SpecError, TaskSpec, TuningforkOptions, load_spec, validate_spec

EXAMPLE = Path(__file__).parents[1] / "examples" / "tally-report.evals.json"


def spec(**eval_fields) -> dict:
    item = {"id": "1", "prompt": "Summarize data.csv."}
    item.update(eval_fields)
    return {"skill_name": "tally", "evals": [item]}


def where(data) -> list[str]:
    return [problem.where for problem in validate_spec(data)]


class TestRoundTrip:
    def test_the_example_spec_is_written_back_as_it_was_read(self):
        data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        assert validate_spec(data) == []
        again = TaskSpec.from_dict(data).to_dict()
        assert TaskSpec.from_dict(again).to_dict() == again
        assert again["evals"][0]["prompt"] == data["evals"][0]["prompt"]
        assert again["tuningfork"] == data["tuningfork"]

    def test_keys_this_version_does_not_know_are_kept(self):
        """A spec shared with other tools (the evals.json convention) carries their keys; a
        spec saved from the browser must not drop them."""
        data = {
            "skill_name": "tally",
            "owner": "someone else's tool",
            "tuningfork": {"skill_dir": "skill", "future_option": 3},
            "evals": [
                {
                    "id": 7,
                    "prompt": "Do it.",
                    "grader": {"kind": "llm"},
                    "tuningfork": {"check": "true", "retries": 2},
                }
            ],
        }
        written = TaskSpec.from_dict(data).to_dict()
        assert written["owner"] == "someone else's tool"
        assert written["tuningfork"] == {"skill_dir": "skill", "future_option": 3}
        item = written["evals"][0]
        assert item["id"] == "7" and item["grader"] == {"kind": "llm"}
        assert item["tuningfork"]["retries"] == 2 and item["tuningfork"]["check"] == "true"

    def test_a_launch_record_reads_back_as_the_eval_it_was(self, tmp_path):
        item = Eval(
            id="1", prompt="p", files=["data.csv", {"path": "in/b.bin", "source": "inputs/b.bin"}],
            tuningfork=TuningforkOptions(check="true", normalizers=["tool_name"], extra={"x": 1}),
        )
        record = json.loads(json.dumps(eval_record(item, tmp_path)))
        again = Eval.from_dict(record)
        assert again.tuningfork == item.tuningfork
        assert again.files == item.files

    def test_a_launch_record_from_before_the_serializer_still_loads(self):
        """Launches stored their eval with `dataclasses.asdict`: every option, `check`
        included when it was None."""
        record = {
            "id": "1", "prompt": "p", "expected_output": "", "files": [], "expectations": [],
            "tuningfork": {
                "check": None, "normalizers": [], "held_out": False, "skills": ["s"],
                "timeout": 900.0, "max_turns": 12,
            },
            "spec_dir": "/somewhere",
        }
        item = Eval.from_dict(record)
        assert item.tuningfork.check is None and item.tuningfork.skills == ["s"]
        assert item.tuningfork.extra == {}


class TestValidation:
    def test_a_good_spec_has_no_problems(self):
        assert validate_spec(spec(files=["a.csv", {"path": "b/c.txt", "content": "x"}])) == []

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ([], [""]),
            ({"skill_name": "t"}, ["evals"]),
            ({"skill_name": "t", "evals": []}, ["evals"]),
            ({"skill_name": 3, "evals": [{"id": "1", "prompt": "p"}]}, ["skill_name"]),
            ({"tuningfork": [], "evals": [{"id": "1", "prompt": "p"}]}, ["tuningfork"]),
            ({"tuningfork": {"skill_dir": 1}, "evals": [{"id": "1", "prompt": "p"}]}, ["tuningfork.skill_dir"]),
            ({"evals": ["not an eval"]}, ["evals[0]"]),
        ],
    )
    def test_the_spec_as_a_whole(self, data, expected):
        assert where(data) == expected

    def test_ids_are_required_and_unique(self):
        data = {"evals": [
            {"id": "1", "prompt": "p"},
            {"id": 1, "prompt": "p"},
            {"prompt": "p"},
            {"id": True, "prompt": "p"},
        ]}
        problems = validate_spec(data)
        assert [p.where for p in problems] == ["evals[1].id", "evals[2].id", "evals[3].id"]
        assert "used by another eval" in problems[0].message

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"prompt": "  "}, "evals[0].prompt"),
            ({"expected_output": 3}, "evals[0].expected_output"),
            ({"expectations": "one"}, "evals[0].expectations"),
            ({"files": "a.csv"}, "evals[0].files"),
            ({"tuningfork": "fast"}, "evals[0].tuningfork"),
            ({"tuningfork": {"check": ["python3", "c.py"]}}, "evals[0].tuningfork.check"),
            # Read as true before, since "false" is a non-empty string.
            ({"tuningfork": {"held_out": "false"}}, "evals[0].tuningfork.held_out"),
            ({"tuningfork": {"skills": "tally"}}, "evals[0].tuningfork.skills"),
            ({"tuningfork": {"timeout": 0}}, "evals[0].tuningfork.timeout"),
            ({"tuningfork": {"timeout": "60"}}, "evals[0].tuningfork.timeout"),
            ({"tuningfork": {"max_turns": 0}}, "evals[0].tuningfork.max_turns"),
            ({"tuningfork": {"max_turns": 2.5}}, "evals[0].tuningfork.max_turns"),
            ({"tuningfork": {"max_turns": True}}, "evals[0].tuningfork.max_turns"),
            ({"tuningfork": {"check_env": "OPENAI_API_KEY"}}, "evals[0].tuningfork.check_env"),
            ({"tuningfork": {"check_env": ["OPENAI API KEY"]}}, "evals[0].tuningfork.check_env"),
            ({"tuningfork": {"check_env": [3]}}, "evals[0].tuningfork.check_env"),
        ],
    )
    def test_each_field_of_an_eval(self, fields, expected):
        assert where(spec(**fields)) == [expected]

    @pytest.mark.parametrize("name", ["TUNINGFORK_TOKEN", "TUNINGFORK_MCP_TOKEN", "TUNINGFORK_PROPOSER_KEY", "TUNINGFORK_REPLY"])
    def test_a_check_is_never_given_a_variable_tuningfork_keeps_for_itself(self, name):
        (problem,) = validate_spec(spec(tuningfork={"check_env": ["OPENAI_API_KEY", name]}))
        assert problem.where == "evals[0].tuningfork.check_env"
        assert f"keeps for itself: {name}" in problem.message

    def test_the_variables_a_check_may_read_are_kept_and_written_back(self):
        data = spec(tuningfork={"check": "true", "check_env": ["OPENAI_API_KEY", "JUDGE_URL"]})
        assert validate_spec(data) == []
        options = TuningforkOptions.from_dict(data["evals"][0]["tuningfork"])
        assert options.check_env == ["OPENAI_API_KEY", "JUDGE_URL"]
        assert options.to_dict()["check_env"] == ["OPENAI_API_KEY", "JUDGE_URL"]
        # A spec that names none is written back without the key, as it was read.
        assert "check_env" not in TuningforkOptions.from_dict({"check": "true"}).to_dict()

    @pytest.mark.parametrize(
        "check",
        [{"reply_contains": "61.5"}, {"reply_matches": r"(?i)total\D+61\.5"}, "true"],
    )
    def test_a_check_can_judge_the_reply(self, check):
        assert validate_spec(spec(tuningfork={"check": check})) == []

    @pytest.mark.parametrize(
        ("check", "expected"),
        [
            ({"reply_contains": ""}, "evals[0].tuningfork.check.reply_contains"),
            ({"reply_contains": 61.5}, "evals[0].tuningfork.check.reply_contains"),
            ({"reply_matches": "(unclosed"}, "evals[0].tuningfork.check.reply_matches"),
            ({"reply_equals": "x"}, "evals[0].tuningfork.check"),
            ({"reply_contains": "a", "reply_matches": "b"}, "evals[0].tuningfork.check"),
        ],
    )
    def test_a_reply_check_that_cannot_judge_is_named(self, check, expected):
        assert where(spec(tuningfork={"check": check})) == [expected]

    def test_checks_read_as_a_person_would(self):
        from tuningfork.specs import check_kind, describe_check

        assert (check_kind("true"), describe_check("true")) == ("shell", "true")
        assert describe_check({"reply_contains": "61.5"}) == 'the reply contains "61.5"'
        assert describe_check({"reply_matches": "a+"}) == "the reply matches /a+/"
        assert (check_kind(None), check_kind("  "), check_kind({"x": "y"})) == (None, None, None)

    def test_a_reply_pattern_crosses_lines(self, tmp_path):
        """Found on a real check: (?=.*212.00)(?=.*process-receipt) failed on a reply that
        held both, because they were on different lines."""
        from tuningfork.specs import run_check

        reply = "The total is 212.00.\n\nFiled with the process-receipt skill."
        both = run_check({"reply_matches": r"(?=.*212\.00)(?=.*process-receipt)"}, tmp_path, reply=reply)
        missing = run_check({"reply_matches": r"(?=.*212\.00)(?=.*archive)"}, tmp_path, reply=reply)
        assert both.passed and not missing.passed

    def test_an_unknown_normalizer_is_named_with_the_known_ones(self):
        (problem,) = validate_spec(spec(tuningfork={"normalizers": ["tool_name", "vibes"]}))
        assert problem.where == "evals[0].tuningfork.normalizers"
        assert "vibes" in problem.message and "tool_call_json" in problem.message

    @pytest.mark.parametrize(
        "entry",
        [
            "..",
            {"path": "/etc/passwd", "content": "x"},
            {"path": "../outside.txt", "content": "x"},
            {"path": "in/../../outside.txt", "content": "x"},
            {"path": "C:/Windows/x.txt", "content": "x"},
            {"path": "", "content": "x"},
            {"path": "a.bin", "content": b"x".hex(), "source": ""},
            {"path": "a.txt", "content": 3},
            3,
        ],
    )
    def test_a_file_that_would_land_outside_the_workspace_or_is_malformed(self, entry):
        assert where(spec(files=[entry])) == ["evals[0].files[0]"]

    def test_a_check_env_that_is_not_a_list_is_read_as_none(self):
        """Read back leniently, as a launch's stored eval is: a name given as text is not split
        into its letters."""
        assert TuningforkOptions.from_dict({"check": "true", "check_env": "OPENAI_API_KEY"}).check_env == []


class TestACheckEnvironment:
    """A shell check can run what the model left in the workspace (a `conftest.py` its `pytest`
    finds), so it gets an environment made from scratch: a launch's few variables, and those
    its task names — none of the person's others, sandbox or not."""

    #: What the shell running a check may set by itself.
    SHELL_MADE = {"PWD", "OLDPWD", "SHLVL", "_"}

    def seen(self, tmp_path, **options) -> dict[str, str]:
        from tuningfork.specs import run_check

        result = run_check("env", tmp_path, keep=None, **options)
        assert result.passed, result.output
        return dict(line.split("=", 1) for line in result.output.splitlines() if "=" in line)

    def test_none_of_the_persons_variables_reach_it_but_a_launchs(self, tmp_path, monkeypatch):
        from tuningfork.specs import CHECK_INHERITED

        monkeypatch.setenv("CLOUD_API_KEY", "sk-not-for-a-check")
        monkeypatch.setenv("LANG", "C.UTF-8")
        seen = self.seen(tmp_path)
        assert "CLOUD_API_KEY" not in seen
        assert seen["LANG"] == "C.UTF-8"
        assert set(seen) - self.SHELL_MADE <= set(CHECK_INHERITED)

    def test_it_reads_the_variables_its_task_names(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JUDGE_KEY", "for-the-judge")
        monkeypatch.setenv("CLOUD_API_KEY", "sk-not-for-a-check")
        seen = self.seen(tmp_path, check_env=["JUDGE_KEY", "UNSET_ONE"])
        assert seen["JUDGE_KEY"] == "for-the-judge"
        assert "CLOUD_API_KEY" not in seen and "UNSET_ONE" not in seen

    def test_never_a_key_of_tuningforks_own_however_a_spec_names_it(self, tmp_path, monkeypatch):
        """A launch's eval is read back leniently: a name validation would refuse still gives
        the check nothing."""
        monkeypatch.setenv("TUNINGFORK_TOKEN", "the-proxys-key")
        assert "TUNINGFORK_TOKEN" not in self.seen(tmp_path, check_env=["TUNINGFORK_TOKEN"])

    def test_it_finds_the_reply_and_the_spec_folder(self, tmp_path):
        reply = tmp_path / "reply.txt"
        seen = self.seen(tmp_path, reply_file=reply, spec_dir=tmp_path)
        assert seen["TUNINGFORK_REPLY"] == str(reply) and seen["TUNINGFORK_SPEC_DIR"] == str(tmp_path.resolve())


class TestWhatACheckPrints:
    """What a check prints is masked where it runs, with the store's masking, before it is cut,
    kept or told to a harness."""

    #: A key in a shape every pattern set masks whole, written in two pieces as tests write keys.
    KEY = "sk-" "tfcheckprints4Q7xZ2mN8pL4vR6tY"

    def test_a_key_the_cut_falls_inside_is_masked_whole(self, tmp_path):
        """Found mapping the store's second half: the output was cut to its end before anything
        masked it, and a key the cut fell inside kept its end, which no pattern reads alone."""
        from tuningfork.specs import CHECK_KEPT, run_check
        from tuningfork.store.masking import Masking

        tail = self.KEY[6:]
        (tmp_path / "out.txt").write_text(self.KEY + "x" * (CHECK_KEPT - len(tail)), encoding="utf-8")
        assert run_check("cat out.txt", tmp_path).output.startswith(tail)  # unmasked, the cut keeps its end
        masked = run_check("cat out.txt", tmp_path, masking=Masking())
        assert tail not in masked.output and masked.output.startswith("<redacted>")

    def test_a_variable_named_as_a_secret_is_masked_wherever_its_value_stands(self, tmp_path, monkeypatch):
        """A key its task lets the check read, printed by a request that failed: no pattern reads
        it alone, but its name is a secret's. A variable named otherwise is printed as it is."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        monkeypatch.setenv("JUDGE_API_TOKEN", "Pw4judge" "Q8zK2mX7")
        monkeypatch.setenv("JUDGE_REGION", "eu-west-" "1abcdefgh")
        command = 'echo "refused $JUDGE_API_TOKEN in $JUDGE_REGION"'
        names = ["JUDGE_API_TOKEN", "JUDGE_REGION"]
        printed = run_check(command, tmp_path, check_env=names).output
        assert printed == "refused Pw4judgeQ8zK2mX7 in eu-west-1abcdefgh" and Masking().value(printed) == printed
        said = run_check(command, tmp_path, check_env=names, masking=Masking()).output
        assert said == "refused <redacted> in eu-west-1abcdefgh"

    def test_tuningforks_own_keys_are_masked_wherever_they_stand(self, tmp_path, monkeypatch):
        """No check is given one, but one can find a key in a file the model's code read."""
        from tuningfork.access import held_secrets, mcp_token_path, write_secret, write_token
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        for name in ("TUNINGFORK_TOKEN", "TUNINGFORK_MCP_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        home = tmp_path / "home"
        write_token(home, "proxys-own-" "key-4Q7xZ2")
        write_secret(mcp_token_path(home), "mcp-own-" "key-8pL4vR")
        monkeypatch.setenv("TUNINGFORK_PROPOSER_KEY", "proposer-" "key-6tY0aB")
        held = held_secrets(home)
        assert set(held) == {"proxys-own-key-4Q7xZ2", "mcp-own-key-8pL4vR", "proposer-key-6tY0aB"}
        (tmp_path / "found.txt").write_text(" ".join(held), encoding="utf-8")
        said = run_check("cat found.txt", tmp_path, masking=Masking(), also=held).output
        assert said == "<redacted> <redacted> <redacted>"
        # A reply check quotes the reply, which can hold one too.
        quoted = run_check({"reply_contains": "found"}, tmp_path, reply=f"found {held[0]}", masking=Masking(),
                           also=held).output
        assert held[0] not in quoted and "found <redacted>" in quoted

    def test_a_reply_checks_output_is_masked_too(self, tmp_path):
        """It quotes the reply, which a trial takes as a person typed it."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        judged = run_check({"reply_contains": "done"}, tmp_path, reply=f"done with {self.KEY}", masking=Masking())
        assert judged.passed and self.KEY not in judged.output and "done with <redacted>" in judged.output
        # What it looked for too, which its verdict names.
        judged = run_check({"reply_contains": self.KEY}, tmp_path, reply="no key here", masking=Masking())
        assert not judged.passed and self.KEY not in judged.output and '"<redacted>"' in judged.output

    def test_a_reply_is_masked_before_it_is_cut_to_its_quote(self, tmp_path):
        """Cut first, a key the cut fell inside would keep its start, which no pattern reads alone."""
        from tuningfork.specs import REPLY_QUOTED, run_check
        from tuningfork.store.masking import Masking

        reply = "x" * (REPLY_QUOTED - 19) + " " + self.KEY + " done"
        assert self.KEY[:16] in run_check({"reply_contains": "done"}, tmp_path, reply=reply).output
        judged = run_check({"reply_contains": "done"}, tmp_path, reply=reply, masking=Masking())
        assert judged.passed and self.KEY[:16] not in judged.output and judged.output.endswith("<redacted> done")

    def test_it_is_masked_as_a_runs_result_keeps_it(self, tmp_path):
        """As JSON writes it too, its quotes escaped, as the store masks what a result keeps: read as
        text alone, a password before a typed declaration's quoted key stood (found in review)."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        line = 'login refused: password: Zq8xK3vN5wLp, api_key = "Q7bX2kLm9Zr4Tw8Y"'
        assert "Zq8xK3vN5wLp" in Masking().value(line)  # read as text alone
        (tmp_path / "said.txt").write_text(line, encoding="utf-8")
        said = run_check("cat said.txt", tmp_path, masking=Masking()).output
        assert "Zq8xK3vN5wLp" not in said and "Q7bX2kLm9Zr4Tw8Y" not in said
        told = run_check({"reply_contains": "x"}, tmp_path, reply=line, masking=Masking()).output
        assert "Zq8xK3vN5wLp" not in told
        # What a check looked for, which its verdict names as the task wrote it.
        named = run_check({"reply_matches": line}, tmp_path, reply="x", masking=Masking()).output
        assert "Zq8xK3vN5wLp" not in named

    def test_it_is_masked_every_way_storage_reads_a_text(self, tmp_path, monkeypatch):
        """Read as a string in JSON alone, as a run's result was masked, a key three escapes deep
        stood, and a secret variable's value written as a number or partly taken by a pattern (found
        in review); read as text alone, the password above."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        monkeypatch.setenv("DB_PASSWORD", "Zq8xK3vN5wLp;Ab9x")
        monkeypatch.setenv("REDIS_PASSWORD", "739152648")
        printed = {
            "deep.txt": ("keys below" + "\\" * 4 + "n" + self.KEY, self.KEY),
            "part.txt": ("FATAL: login with password=Zq8xK3vN5wLp;Ab9x failed", "Ab9x"),
            "number.txt": ('{"host": "db", "password": 739152648}', "739152648"),
        }
        for name, (text, secret) in printed.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
            assert secret in Masking().value([text], also=["Zq8xK3vN5wLp;Ab9x", "739152648"])[0], name
            said = run_check(f"cat {name}", tmp_path, check_env=["DB_PASSWORD", "REDIS_PASSWORD"], masking=Masking())
            assert secret not in said.output, (name, said.output)

    def test_a_secret_variables_value_is_masked_as_an_encoder_escaped_it(self, tmp_path, monkeypatch):
        """Written by an encoder that escapes one of its characters — PHP's `\\/`, Python's `\\u00fc`,
        Go's `\\u0026` — a document of its own or in a line of a log (found in review)."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        values = {"DB_PASSWORD": "Zq8x/K3vN5wLp+Ab", "API_PASSWORD": "Zürich2024!xQ", "DSN_PASSWORD": "Zq8x&K3vN5wLp"}
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        printed = ['{"note":"Zq8x\\/K3vN5wLp+Ab"}', json.dumps({"pw": "Zürich2024!xQ"}),
                   '{"dsn": "Zq8x\\u0026K3vN5wLp"}', f"ERROR config {json.dumps({'pw': 'Zürich2024!xQ'})} rejected",
                   # A tool's JSON result kept in a session's JSON: two escapes deep.
                   json.dumps({"result": json.dumps({"pw": "Zürich2024!xQ"})})]
        (tmp_path / "said.txt").write_text("\n".join(printed), encoding="utf-8")
        assert run_check("cat said.txt", tmp_path, check_env=list(values)).output.count("K3vN5wLp") == 2
        said = run_check("cat said.txt", tmp_path, check_env=list(values), masking=Masking()).output
        assert "K3vN5wLp" not in said and "rich2024" not in said, said

    def test_a_long_output_is_masked_from_its_end(self, tmp_path, monkeypatch):
        """Masking reads a text many times over, in the process running the check (found in review):
        a long output is masked as far as its end needs, and even a caller keeping it whole is given
        that much, the rest said cut."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        monkeypatch.setattr("tuningfork.specs.MASKED_MOST", 3000)
        printed = "y" * 9000 + "\n" + self.KEY + "\n" + "x" * 1900
        (tmp_path / "out.txt").write_text(printed, encoding="utf-8")
        shorter = len(self.KEY) - len("<redacted>")  # what the mask takes off
        kept = run_check("cat out.txt", tmp_path, masking=Masking())
        assert self.KEY[-20:] not in kept.output and kept.output.endswith("<redacted>\n" + "x" * 1900)
        assert len(kept.output) == 2000 and kept.cut == len(printed) - 2000 - shorter
        whole = run_check("cat out.txt", tmp_path, masking=Masking(), keep=None)
        assert len(whole.output) == 3000 - shorter and whole.cut == len(printed) - 3000

    def test_where_a_long_output_is_given_from_no_key_is_cut(self, tmp_path, monkeypatch):
        """The place it is given from can fall inside a key (found in review): read on before it, the
        key is found whole, and the output is given from past it."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        monkeypatch.setattr("tuningfork.specs.MASKED_MOST", 3000)
        monkeypatch.setattr("tuningfork.specs.MASKED_MARGIN", 200)
        printed = "y" * 5000 + "\n" + self.KEY + "\n" + "x" * 2972  # given from 6 characters into the key
        (tmp_path / "out.txt").write_text(printed, encoding="utf-8")
        whole = run_check("cat out.txt", tmp_path, masking=Masking(), keep=None)
        assert whole.output == "\n" + "x" * 2972 and whole.cut == len(printed) - len(whole.output)
        assert run_check("cat out.txt", tmp_path, masking=Masking()).output == "x" * 2000

    def test_a_long_reply_is_masked_from_its_start(self, tmp_path, monkeypatch):
        from tuningfork.specs import REPLY_QUOTED, run_check
        from tuningfork.store.masking import Masking

        monkeypatch.setattr("tuningfork.specs.MASKED_MOST", 1000)
        judged = run_check({"reply_contains": "a"}, tmp_path, reply="a" * 3000, masking=Masking())
        assert judged.output.endswith("a" * REPLY_QUOTED + f"… (+{3000 - REPLY_QUOTED} chars)")
        # Where its start ends inside a key, it ends before it, and what is left out counts it.
        reply = "a" * 995 + " " + self.KEY + " done"
        judged = run_check({"reply_contains": "a"}, tmp_path, reply=reply, masking=Masking())
        assert judged.output.endswith("a" * REPLY_QUOTED + f"… (+{len(reply) - REPLY_QUOTED} chars)")
        # Shorter than a quote, and still not all of it.
        monkeypatch.setattr("tuningfork.specs.MASKED_MOST", 300)
        judged = run_check({"reply_contains": "a"}, tmp_path, reply="a" * 1000, masking=Masking())
        assert judged.output.endswith("a" * 300 + "… (+700 chars)")
        # Ending inside a key there, read on past it: the quote stops before the key.
        reply = "a" * 280 + " " + self.KEY + " done"
        judged = run_check({"reply_contains": "a"}, tmp_path, reply=reply, masking=Masking())
        assert self.KEY[:16] not in judged.output and judged.output.endswith(f"{'a' * 280} … (+{len(reply) - 281} chars)")

    def test_the_values_given_mask_every_text_it_masks(self, tmp_path):
        """A reply check masks what it looked for and the reply apart: values given once, as an
        iterator, mask both (found in review)."""
        from tuningfork.specs import run_check
        from tuningfork.store.masking import Masking

        held = "Pw42heldKey" "Q7xZ2mN8pL4vR"
        said = run_check({"reply_contains": held}, tmp_path, reply=f"the key is {held}", masking=Masking(),
                         also=iter([held])).output
        assert held not in said

    def test_without_the_stores_masking_nothing_is_masked(self, tmp_path):
        from tuningfork.specs import run_check

        assert run_check(f"echo {self.KEY}", tmp_path).output == self.KEY

    @pytest.mark.parametrize(("name", "secret"), [
        ("OPENAI_API_KEY", True), ("DB_PASSWORD", True), ("DB_PASS", True), ("GITHUB_TOKEN", True),
        ("AWS_SECRET_ACCESS_KEY", True), ("client_secret", True), ("STRIPE_KEY", True),
        ("MYSQL_PWD", True), ("SECRET_KEY_BASE", True), ("openai_key", True),
        ("HTTPS_PROXY", False), ("SSL_CERT_FILE", False), ("NEXT_PAGE_TOKEN", False), ("BYPASS", False),
        ("PATH", False), ("STRIPE_PUBLIC_KEY", False), ("API_KEY_FILE", False), ("TOKEN_URL", False),
        ("stripe_publishable_key", False), ("PWD", False), ("MONKEY", False), ("SECRETARY", False),
        ("SECRET_KEY_FILE", False), ("VAULT_SECRET_PATH", False), ("AWS_SECRET_NAME", False), ("cache_key", False),
        ("KEY", True), ("SECRET", True), ("VAULT_SECRET_ID", True), ("AWS_ACCESS_KEY_ID", False),
    ])
    def test_a_variables_name_is_a_secrets_as_the_patterns_read_one(self, name, secret):
        from tuningfork.config import secret_name

        assert secret_name(name) is secret


class TestTheMachinesCheckEnv:
    """What every check on a machine reads goes in its config, beside the endpoints it protects:
    a proxy's settings, which belong to no task file."""

    def write(self, path: Path, data) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_the_machines_and_the_recordings_folders_lists_add_up(self, tmp_path, monkeypatch):
        from tuningfork.specs import configured_check_env

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        self.write(tmp_path / "config" / "tuningfork" / "config.json", {"check_env": ["https_proxy", "SSL_CERT_FILE"]})
        home = tmp_path / "home"
        self.write(home / "config.json", {"check_env": ["SSL_CERT_FILE", "HF_HOME"]})
        assert configured_check_env(home) == ["https_proxy", "SSL_CERT_FILE", "HF_HOME"]
        assert configured_check_env() == ["https_proxy", "SSL_CERT_FILE"]
        assert configured_check_env(tmp_path / "elsewhere") == ["https_proxy", "SSL_CERT_FILE"]

    def test_each_file_says_which_it_names(self, tmp_path, monkeypatch):
        """A plan and Connections name the file a variable is named in: one kept in the
        recordings folder's config is not looked for in the machine's."""
        from tuningfork.specs import check_env_sources

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        machine = self.write(tmp_path / "config" / "tuningfork" / "config.json", {"check_env": ["https_proxy"]})
        home = tmp_path / "home"
        kept = self.write(home / "config.json", {"check_env": ["HF_HOME"]})
        assert check_env_sources(home) == [(machine, ["https_proxy"]), (kept, ["HF_HOME"])]

    @pytest.mark.parametrize("text", ['{"check_env": ["https_proxy"],', "[]"])
    def test_a_machine_config_that_does_not_parse_is_refused_not_read_as_empty(self, tmp_path, monkeypatch, text):
        from tuningfork.ops.common import OperationError
        from tuningfork.specs import configured_check_env

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        machine = tmp_path / "config" / "tuningfork" / "config.json"
        machine.parent.mkdir(parents=True)
        machine.write_text(text, encoding="utf-8")
        with pytest.raises(OperationError) as refused:
            configured_check_env()
        assert str(machine) in refused.value.message

    def test_a_recordings_folders_config_that_does_not_parse_names_nothing(self, tmp_path):
        """Read as its other settings are, and as it was before: a hand-broken file there blocks
        no batch, whatever its check."""
        from tuningfork.specs import check_env_sources

        home = tmp_path / "home"
        home.mkdir()
        (home / "config.json").write_text('{"check_env": ["https_proxy"],', encoding="utf-8")
        assert check_env_sources(home) == []

    def test_an_empty_list_names_nothing(self, tmp_path, monkeypatch):
        from tuningfork.specs import check_env_sources

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        self.write(tmp_path / "config" / "tuningfork" / "config.json", {"check_env": []})
        assert check_env_sources() == []

    def test_a_list_in_the_recordings_folders_config_that_cannot_be_read_names_that_file(self, tmp_path):
        from tuningfork.ops.common import OperationError
        from tuningfork.specs import configured_check_env

        home = tmp_path / "home"
        kept = self.write(home / "config.json", {"proposer": {}, "check_env": ["TUNINGFORK_PROPOSER_KEY"]})
        with pytest.raises(OperationError) as refused:
            configured_check_env(home)
        assert str(kept) in refused.value.message and "keeps for itself" in refused.value.message

    @pytest.mark.parametrize("names", ["https_proxy", ["not a name"], ["TUNINGFORK_TOKEN"]])
    def test_a_list_that_cannot_be_read_is_refused_saying_where(self, tmp_path, monkeypatch, names):
        from tuningfork.ops.common import OperationError
        from tuningfork.specs import configured_check_env

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        machine = self.write(tmp_path / "config" / "tuningfork" / "config.json", {"check_env": names})
        with pytest.raises(OperationError) as refused:
            configured_check_env()
        assert str(machine) in refused.value.message and refused.value.code == "check_env"


class TestLoading:
    def test_a_spec_with_problems_is_refused_with_each_named(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text(json.dumps(spec(tuningfork={"held_out": "no", "max_turns": 0})))
        with pytest.raises(SpecError) as refused:
            load_spec(path)
        assert [p.where for p in refused.value.problems] == [
            "evals[0].tuningfork.held_out", "evals[0].tuningfork.max_turns",
        ]
        assert "held_out" in str(refused.value) and "max_turns" in str(refused.value)

    def test_not_json_at_all(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text("{ not json")
        with pytest.raises(SpecError, match="not valid JSON"):
            load_spec(path)

    def test_leniency_is_still_there_when_asked_for(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text(json.dumps(spec(tuningfork={"max_turns": 0})))
        assert load_spec(path, strict=False).evals[0].tuningfork.max_turns == 0
