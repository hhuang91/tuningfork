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
