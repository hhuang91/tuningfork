"""The harness seam.

The Hermes adapter is checked against what the real CLI accepts (v0.18.2): `chat -q` for a
single non-interactive query, `-s` for preloaded skills, and `HERMES_HOME` for isolation.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.harness import ADAPTERS, GenericHarness, HarnessError, HermesAdapter


@pytest.fixture
def adapter() -> HermesAdapter:
    return HermesAdapter()


REVIEW = (
    "Review the conversation above and update the skill library. Be ACTIVE — most sessions "
    "produce at least one skill update, even if small."
)


def test_hermes_names_its_background_review(adapter):
    """Hermes 0.18.2 replays the conversation after its answer with a review prompt as the last
    user turn (agent/background_review.py). The task's own prompt is never one, even worded so."""
    answer = [{"role": "user", "content": "Summarize data.csv"}, {"role": "assistant", "content": "Done."}]
    review = {"messages": [*answer, {"role": "user", "content": REVIEW}]}
    later = {"messages": [*review["messages"], {"role": "assistant", "content": "", "tool_calls": []},
                          {"role": "tool", "content": "{}"}]}

    assert adapter.side_call(review) == adapter.side_call(later) == "background review"
    assert adapter.side_call({"messages": answer}) is None
    assert adapter.side_call({"messages": [{"role": "user", "content": REVIEW}]}) is None
    assert GenericHarness().side_call(review) is None


def test_hermes_names_the_user_messages_it_writes_itself(adapter):
    """Each note Hermes 0.18.2 sends as a user message, as its source words it; a person's turn,
    a document's note before their words included, is none of them."""
    written = {
        "continue after a cut reply": "[System: Your previous response was truncated by the output length limit. "
                                      "Continue exactly where you left off. Do not restart or repeat prior text. "
                                      "Finish the answer directly.]",
        "continue after a broken stream": "[System: The previous response was cut off by a network error mid-stream. "
                                          "Continue exactly where you left off.]",
        "a tool call too large to send": "[System: Your previous tool call (write_file) was too large and the stream "
                                         "timed out before it could be delivered.]",
        "nudge after an empty answer": "You just executed tool calls but returned an empty response. Please process "
                                       "the tool results above and continue with the task.",
        "nudge to act, not announce": "[System: Continue now. Execute the required tool calls and only send your "
                                      "final answer after completing the task.]",
        "nudge to verify edited code": "[System: You edited code in this turn, but the workspace does not have fresh "
                                       "passing verification evidence yet.\n\nVerification status: unverified]",
        "summary at the turn limit": "You've reached the maximum number of tool-calling iterations allowed. Please "
                                     "provide a final response summarizing what you've found and accomplished so far, "
                                     "without calling any more tools.",
        "continue after a compression": "Continue from the compressed conversation context above. This marker exists "
                                        "because the compacted transcript contained no preserved user turn.",
        "background review": REVIEW,
    }
    for name, text in written.items():
        assert adapter.harness_note({"role": "user", "content": text}) == name
        assert adapter.harness_note({"role": "user", "content": [{"type": "text", "text": text}]}) == name
    for said in ("Thanks. Now the same for march.csv.",
                 "[The user sent a document: 'receipt.pdf'. It is saved at: /tmp/receipt.pdf.]\n\nFile this one.",
                 "[System: a wording no version of Hermes tuningfork has read]"):
        assert adapter.harness_note({"role": "user", "content": said}) is None
    assert not hasattr(GenericHarness(), "harness_note")


def test_hermes_names_the_skill_a_call_loads(adapter):
    def tool(name, arguments):
        return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    assert adapter.skill_load(tool("skill_view", {"name": "tally-report"})) == "tally-report"
    assert adapter.skill_load(tool("skill_view", {"name": "  "})) is None
    assert adapter.skill_load(tool("terminal", {"command": "ls"})) is None
    assert adapter.skill_load({"function": {"name": "skill_view", "arguments": "{not json"}}) is None
    assert not hasattr(GenericHarness(), "skill_load")


class TestIsolatedHome:
    def test_config_points_at_the_proxy(self, adapter, tmp_path):
        home = adapter.prepare_home(
            tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="gemma"
        )

        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert "base_url: http://127.0.0.1:8787/v1" in config
        assert "default: gemma" in config
        assert "provider: custom" in config
        assert (home / ".hermes" / "skills").is_dir()

    def test_the_command_scanner_is_not_downloaded_into_every_run(self, adapter, tmp_path):
        """Found on a real batch: each fresh home fetched Hermes's 38 MB tirith scanner from
        GitHub — traffic outside the proxy, and a scanner that arrived mid-run or not."""
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert "security:\n" in config
        assert "  tirith_enabled: false" in config

    def test_home_is_the_only_hermes_variable_passed_through(
        self, adapter, tmp_path, monkeypatch
    ):
        """A stray HERMES_* in the parent shell must not change what is measured."""
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        monkeypatch.setenv("HERMES_TIMEZONE", "UTC")

        env = adapter.environment(tmp_path)

        hermes_vars = {key for key in env if key.startswith("HERMES_")}
        assert hermes_vars == {"HERMES_HOME"}
        assert env["HERMES_HOME"] == str((tmp_path / ".hermes").resolve())
        assert env["HOME"] == str(tmp_path.resolve())

    def test_skill_root_is_inside_the_isolated_home(self, adapter, tmp_path):
        assert adapter.skill_roots(tmp_path) == (tmp_path / ".hermes" / "skills",)


SKILL_WITH_CATEGORY = """---
name: xlsx
description: Create, read, edit Excel .xlsx workbooks and CSVs.
metadata:
  hermes:
    tags: [excel, spreadsheet]
    category: productivity
---

# Xlsx
"""


class TestInstallLayout:
    """Laid out as an installation: the model reads these paths, and laid out otherwise the
    same step called a skill's script another way."""

    def test_a_skill_goes_in_the_category_folder_it_declares(self, adapter, tmp_path):
        skill = tmp_path / "src" / "xlsx"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(SKILL_WITH_CATEGORY, encoding="utf-8")

        assert adapter.skill_folder(tmp_path / "home", skill) == (
            tmp_path / "home" / ".hermes" / "skills" / "productivity" / "xlsx")

    def test_a_skill_that_declares_none_goes_directly_under_skills(self, adapter, tmp_path):
        skill = tmp_path / "src" / "tally"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: tally\n---\n\nCount.\n", encoding="utf-8")

        assert adapter.skill_folder(tmp_path / "home", skill) == tmp_path / "home" / ".hermes" / "skills" / "tally"

    @pytest.mark.parametrize("frontmatter, category", [
        (SKILL_WITH_CATEGORY, "productivity"),
        ("---\nname: x\ncategory: research\n---\n", "research"),
        ("---\nname: x\nmetadata:\n  other:\n    category: wrong\n---\n", None),  # not Hermes's
        ("---\nname: x\ncategory: ../../etc\n---\n", None),  # never a path that climbs out
        ("---\nname: x\nmetadata:\n  hermes:\n    category: 'quoted'\n---\n", "quoted"),
        ("no frontmatter at all\n", None),
    ])
    def test_the_category_is_read_from_the_frontmatter(self, tmp_path, frontmatter, category):
        from tuningfork.harness.hermes import skill_category

        (tmp_path / "SKILL.md").write_text(frontmatter, encoding="utf-8")
        assert skill_category(tmp_path / "SKILL.md") == category

    def test_a_fork_of_a_run_laid_out_flat_is_laid_out_flat(self, tmp_path):
        """How launches were laid out before: a fork replays the paths its source named."""
        flat = HermesAdapter(layout="flat")
        home = flat.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        skill = tmp_path / "src" / "xlsx"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(SKILL_WITH_CATEGORY, encoding="utf-8")

        assert (home / "config.yaml").is_file() and (home / "skills").is_dir()
        assert flat.environment(home)["HERMES_HOME"] == str(home.resolve())
        assert flat.skill_folder(home, skill) == home / "skills" / "xlsx"
        with pytest.raises(ValueError):
            HermesAdapter(layout="sideways")


class TestLaunch:
    def test_builds_a_non_interactive_invocation(self, adapter, tmp_path):
        spec = adapter.launch(
            "write a script", home=tmp_path / "home", workspace=tmp_path / "work"
        )

        assert spec.argv[:5] == ("hermes", "chat", "-q", "write a script", "-Q")
        assert "--yolo" in spec.argv
        assert "--max-turns" in spec.argv
        assert spec.cwd == tmp_path / "work"
        assert spec.env["HERMES_HOME"] == str((tmp_path / "home" / ".hermes").resolve())

    def test_skills_are_comma_joined(self, adapter, tmp_path):
        spec = adapter.launch(
            "go", home=tmp_path, workspace=tmp_path, skills=["gen-script", "review"]
        )

        index = spec.argv.index("-s")
        assert spec.argv[index + 1] == "gen-script,review"

    def test_yolo_can_be_declined(self, adapter, tmp_path):
        spec = adapter.launch("go", home=tmp_path, workspace=tmp_path, yolo=False)
        assert "--yolo" not in spec.argv

    def test_extra_args_are_appended(self, adapter, tmp_path):
        spec = adapter.launch(
            "go", home=tmp_path, workspace=tmp_path, extra_args=["--ignore-rules"]
        )
        assert spec.argv[-1] == "--ignore-rules"

    def test_command_line_quotes_a_hostile_prompt(self, adapter, tmp_path):
        spec = adapter.launch(
            "rm -rf / ; echo 'gotcha'", home=tmp_path, workspace=tmp_path
        )

        rendered = spec.command_line()
        # The prompt must survive as one argument, not become shell syntax.
        assert "'rm -rf / ; echo '\"'\"'gotcha'\"'\"''" in rendered
        assert rendered.startswith("HERMES_HOME=")


class TestVolatile:
    def test_the_session_date_and_tool_timings_do_not_take_a_fork_off_its_recording(self):
        """What two launches of one task differ in, besides their paths: Hermes writes the
        date into its system prompt, and `execute_code` reports how long it took."""
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(date: str, seconds: float) -> dict:
            result = json.dumps({"status": "success", "output": "4\n", "duration_seconds": seconds})
            return {"model": "m", "messages": [
                {"role": "system", "content": f"You are Hermes.\nConversation started: {date}\n"},
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "c1", "content": result},
            ]}

        wire, volatile = get_wire(None), HermesAdapter.volatile
        one = masked_request(request("Thursday, September 24, 2026", 0.02), wire, volatile=volatile)
        two = masked_request(request("Friday, September 25, 2026", 0.31), wire, volatile=volatile)
        other = masked_request(request("Friday, September 25, 2026", 0.31), wire)
        assert one == two
        assert one != other  # without the adapter's list, they would differ

    def test_the_folder_a_failed_script_ran_in_does_not_take_a_fork_off_its_recording(self):
        """Found validating a proposal: every fork departed at the step after an `execute_code`
        whose script had failed, its traceback naming a sandbox folder made fresh each run."""
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(sandbox: str) -> dict:
            traceback = f'Traceback (most recent call last):\n  File "/tmp/{sandbox}/script.py", line 43'
            result = json.dumps({"status": "error", "output": f"\n--- stderr ---\n{traceback}"})
            return {"model": "m", "messages": [{"role": "user", "content": "go"},
                                               {"role": "tool", "tool_call_id": "c1", "content": result}]}

        wire, volatile = get_wire(None), HermesAdapter.volatile
        assert (masked_request(request("hermes_sandbox_de0d29yj"), wire, volatile=volatile)
                == masked_request(request("hermes_sandbox_ni1r0scf"), wire, volatile=volatile))

    def test_the_date_does_not_take_a_claude_code_fork_off_its_recording(self):
        """Found validating a proposal past midnight: Claude Code writes today's date into its
        system prompt, and every fork of a run recorded the day before departed at step 0."""
        from tuningfork.harness import ClaudeCodeAdapter
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(date: str) -> dict:
            system = f"You are Claude Code.\n\nToday's date is {date}."
            return {"model": "m", "system": [{"type": "text", "text": system}],
                    "messages": [{"role": "user", "content": "go"}]}

        wire, volatile = get_wire("anthropic-messages"), ClaudeCodeAdapter.volatile
        one = masked_request(request("2026-10-04"), wire, volatile=volatile)
        two = masked_request(request("2026-10-05"), wire, volatile=volatile)
        assert one == two
        assert masked_request(request("2026-10-05"), wire) != one  # without the adapter's list, they differ

    def test_a_subagent_does_not_take_a_claude_code_fork_off_its_recording(self):
        """Found forking real runs with `Agent`, in the foreground and in the background:
        every launch names a new agent (and, in the background, a transcript file under the
        session), and what the agent reports back says how long it took, and counts tokens
        that include per-run paths."""
        from tuningfork.harness import ClaudeCodeAdapter
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(agent: str, session: str, workspace: str, ms: int, tokens: int) -> dict:
            transcript = f"/tmp/claude-1000/{workspace}/{session}/tasks/{agent}.output"
            waited = (
                f"4 rows.\nagentId: {agent} (use SendMessage with to: '{agent}', summary:"
                f" '<recap>' to continue this agent)\n<usage>subagent_tokens: {tokens}\n"
                f"tool_uses: 2\nduration_ms: {ms}</usage>"
            )
            launched = (
                f"Async agent launched successfully.\nagentId: {agent} (internal ID - do not"
                f" mention to user. Use SendMessage with to: '{agent}', summary: '<recap>' to"
                f" continue this agent.)\noutput_file: {transcript}"
            )
            notice = (
                f"<task-notification>\n<task-id>{agent}</task-id>\n<output-file>{transcript}"
                f"</output-file>\n<usage><subagent_tokens>{tokens}</subagent_tokens><tool_uses>2"
                f"</tool_uses><duration_ms>{ms}</duration_ms></usage>\n</task-notification>"
            )
            return {"model": "m", "max_tokens": 10, "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent", "input": {"prompt": "count"}},
                    {"type": "tool_use", "id": "t2", "name": "Agent", "input": {"prompt": "sum"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": waited},
                    {"type": "tool_result", "tool_use_id": "t2", "content": launched},
                    {"type": "text", "text": notice},
                ]},
            ]}

        wire, volatile = get_wire("anthropic-messages"), ClaudeCodeAdapter.volatile
        one = request("a90a77c1a75248245", "3e0abeff-6fb3-4941-8e95-0b90578a4638",
                      "-home-u-tf-batches-69b8ab9e-0-workspace", 18360, 9130)
        two = request("a780860998204025a", "5c1d2e3f-0a1b-4c2d-8e9f-0123456789ab",
                      "-home-u-tf-forks-2b89a35e-3-workspace", 1204, 9126)
        assert masked_request(one, wire, volatile=volatile) == masked_request(two, wire, volatile=volatile)
        assert masked_request(one, wire) != masked_request(two, wire)


class TestGenericHarness:
    def test_launch_points_at_the_extension_path(self, tmp_path):
        with pytest.raises(HarnessError, match="docs/adapters/TEMPLATE.md"):
            GenericHarness().launch("go", home=tmp_path, workspace=tmp_path)

    def test_prepare_home_explains_the_manual_step(self, tmp_path):
        with pytest.raises(HarnessError, match="http://127.0.0.1:8787"):
            GenericHarness().prepare_home(
                tmp_path, base_url="http://127.0.0.1:8787", model="m"
            )

    def test_registry_lists_known_adapters(self):
        assert set(ADAPTERS) == {"hermes", "claude-code", "generic"}


class TestOptions:
    """What a launch of Hermes may be asked beside the seam's own arguments (`OPTIONS`)."""

    def test_the_review_is_awaited_through_a_plugin_in_the_launch_home(self, tmp_path):
        adapter = HermesAdapter(options={"await_review": True})
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert "plugins:\n  enabled:\n  - tuningfork-launch\n" in config
        assert "allow_tool_override" not in config
        plugin = home / ".hermes" / "plugins" / "tuningfork-launch"
        assert (plugin / "plugin.yaml").read_text(encoding="utf-8").startswith("name: tuningfork-launch\n")
        module = (plugin / "__init__.py").read_text(encoding="utf-8")
        assert "def register(ctx)" in module and '"bg-review"' in module
        assert "on_session_end" in module and "on_session_finalize" in module
        asked = json.loads((plugin / "options.json").read_text(encoding="utf-8"))
        assert asked == {"self_improvement": True, "await_review": True}
        assert adapter.settings == {"harness": {"await_review": True}}
        (note,) = adapter.chain_notes()
        assert "review_every_turn" in note

    def test_self_improvement_off_removes_the_skill_editing_tool_through_the_plugin(self, tmp_path):
        """Hermes turns tools off by whole toolset, and skill_manage shares one with skill_view,
        which runs need: the plugin removes the one tool, with the opt-in Hermes asks for."""
        adapter = HermesAdapter(options={"self_improvement": False})
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert (
            "plugins:\n  enabled:\n  - tuningfork-launch\n  entries:\n    tuningfork-launch:\n"
            "      allow_tool_override: true\n"
        ) in config
        plugin = home / ".hermes" / "plugins" / "tuningfork-launch"
        asked = json.loads((plugin / "options.json").read_text(encoding="utf-8"))
        assert asked == {"self_improvement": False, "await_review": False}
        module = (plugin / "__init__.py").read_text(encoding="utf-8")
        assert 'SKILL_TOOL = "skill_manage"' in module and "registry.deregister(SKILL_TOOL)" in module
        assert adapter.settings == {"harness": {"self_improvement": False}}
        assert any("self_improvement off" in note for note in adapter.chain_notes())

    def test_the_review_can_be_due_every_turn(self, tmp_path):
        adapter = HermesAdapter(options={"review_every_turn": True})
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert "skills:\n" in config and "  creation_nudge_interval: 1\n" in config
        assert not (home / ".hermes" / "plugins").exists()
        assert adapter.settings == {"harness": {"review_every_turn": True}}
        (note,) = adapter.chain_notes()
        assert "await_review" in note

    def test_mcp_servers_go_in_as_yaml_and_the_settings_keep_only_their_commands(self, tmp_path):
        servers = {"tuningfork": {
            "command": "/v/bin/tuningfork", "args": ["mcp", "--home", "/h", "--tools", "refine"],
            "env": {"KEY": "secret"}, "timeout": 300,
        }}
        adapter = HermesAdapter(options={"mcp_servers": servers})
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        line = next(line for line in config.splitlines() if line.startswith("mcp_servers: "))
        assert json.loads(line[len("mcp_servers: "):]) == servers  # JSON is YAML
        assert adapter.settings == {"harness": {"mcp_servers": {"tuningfork": "/v/bin/tuningfork"}}}
        assert "secret" not in json.dumps(adapter.settings)

    @pytest.mark.parametrize("options, said", [
        ({"nope": 1}, "no option called 'nope'"),
        ({"await_review": "yes"}, "true or false"),
        ({"self_improvement": "off"}, "true or false"),
        ({"mcp_servers": {"x": {}}}, "needs a command"),
        ({"mcp_servers": {"bad name": {"command": "c"}}}, "maps each server"),
        ("x", "must be an object"),
    ])
    def test_what_hermes_does_not_take_is_refused(self, options, said):
        from tuningfork.ops.common import OperationError

        with pytest.raises(OperationError) as refused:
            HermesAdapter(options=options)
        assert refused.value.field == "harness_options" and said in refused.value.message

    def test_without_options_nothing_changes(self, adapter, tmp_path):
        home = adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m")
        config = (home / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        assert "plugins" not in config and "mcp_servers" not in config and "skills:" not in config
        assert not (home / ".hermes" / "plugins").exists()
        assert adapter.settings == {} and len(adapter.chain_notes()) == 2
