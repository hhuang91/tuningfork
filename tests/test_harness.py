"""The harness seam.

The Hermes adapter is checked against what the real CLI accepts (v0.18.2): `chat -q` for a
single non-interactive query, `-s` for preloaded skills, and `HERMES_HOME` for isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        assert set(ADAPTERS) == {"hermes", "claude-code", "openclaw", "generic"}


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


# --- OpenClaw ---------------------------------------------------------------------------------

#: How OpenClaw 2026.10.7 lists a skill in its system prompt, and how a launch's first user
#: message reads: the time before the prompt, a `Runtime:` line after it. Copied from a
#: recording, the session id and host made up.
OPENCLAW_LISTING = (
    "## Skills\n<available_skills>\n  <skill>\n    <name>tally-report</name>\n"
    "    <description>Summarize a CSV of line items into a small JSON report.</description>\n"
    "    <location>~/.openclaw/skills/tally-report/SKILL.md</location>\n  </skill>\n</available_skills>\n"
)
TALLY_SKILL = "---\nname: tally-report\ndescription: Summarize a CSV.\n---\n\n# Tally report\n\n1. Read the CSV.\n"


def _openclaw_prompt(time: str, session: str) -> str:
    return (
        f"[{time}] Use the tally-report skill to summarize data.csv into report.json.\n\n"
        f"Runtime: agent=main | session=agent:main:explicit:{session} | host=box | os=Linux 6.6 (x64) | "
        "node=v26.12.1 | model=local/gemma-4-31b | default_model=local/gemma-4-31b"
    )


def _openclaw_config(home) -> dict:
    """The launch's openclaw.json, its comment line left out: JSON5 is JSON with comments."""
    text = (home / ".openclaw" / "openclaw.json").read_text(encoding="utf-8")
    assert text.startswith("// Written by tuningfork")
    return json.loads(text.split("\n", 1)[1])


class TestOpenClaw:
    """The OpenClaw adapter, as OpenClaw 2026.10.7 takes it: `agent exec` for one headless
    prompt, its state and config in `~/.openclaw`, and a skill read as a file."""

    def test_the_config_points_one_provider_at_the_proxy(self, tmp_path):
        from tuningfork.harness import OpenClawAdapter

        home = OpenClawAdapter().prepare_home(tmp_path / "home", base_url="http://127.0.0.1:8787/_launch/t/v1", model="gemma")
        config = _openclaw_config(home)
        (provider,) = config["models"]["providers"].values()
        assert provider["baseUrl"] == "http://127.0.0.1:8787/_launch/t/v1"
        assert provider["api"] == "openai-completions"
        assert provider["timeoutSeconds"] >= 3600  # a held call outlasts OpenClaw's 300 s first-event wait
        (model,) = provider["models"]
        assert model["id"] == "gemma" and model["compat"]["toolSchemaProfile"] == "llamacpp"
        assert "contextWindow" not in model and "maxTokens" not in model  # OpenClaw's own defaults
        defaults = config["agents"]["defaults"]
        assert defaults["model"]["primary"] == f"{next(iter(config['models']['providers']))}/gemma"
        # The memory flush is off unless asked for; everything else is as OpenClaw ships it.
        assert defaults["compaction"] == {"memoryFlush": {"enabled": False}}
        assert set(config) == {"models", "agents"}
        assert (home / ".openclaw" / "skills").is_dir()

    def test_only_home_is_set_and_no_openclaw_variable_passes(self, tmp_path, monkeypatch):
        """A stray OPENCLAW_* would move OpenClaw's state or config to a real installation."""
        from tuningfork.harness import OpenClawAdapter

        from tuningfork.harness.openclaw import INHERITED_ENV

        for name in ("OPENCLAW_STATE_DIR", "OPENCLAW_CONFIG_PATH", "OPENCLAW_HOME", "OPENCLAW_PROFILE", "NODE_OPTIONS"):
            monkeypatch.setenv(name, "/real")
        monkeypatch.setenv("SOMETHING_ELSE", "x")
        env = OpenClawAdapter().environment(tmp_path)
        assert set(env) <= {*INHERITED_ENV, "HOME"}  # built from scratch: nothing else of the parent's
        assert env["HOME"] == str(tmp_path.resolve())

    def test_a_launch_is_one_headless_agent_exec(self, tmp_path):
        from tuningfork.harness import OpenClawAdapter

        spec = OpenClawAdapter().launch("-x looks like a flag", home=tmp_path / "home", workspace=tmp_path / "work",
                                        extra_args=["--thinking", "low"])
        assert spec.argv[:3] == ("openclaw", "agent", "exec")
        argv = list(spec.argv)
        assert argv[argv.index("--config") + 1] == str((tmp_path / "home" / ".openclaw" / "openclaw.json").resolve())
        assert argv[argv.index("--cwd") + 1] == str((tmp_path / "work").resolve())
        assert argv[argv.index("--timeout") + 1] == "0" and "--json" in argv
        # The prompt goes last, after `--`, and the extra arguments before it.
        assert argv[-2:] == ["--", "-x looks like a flag"] and argv[-4:-2] == ["--thinking", "low"]
        assert spec.cwd == (tmp_path / "work").resolve()
        assert OpenClawAdapter().skill_roots(tmp_path) == (tmp_path / ".openclaw" / "skills",)

    def test_a_relative_home_and_workspace_are_given_whole(self, tmp_path, monkeypatch):
        """Found in review: OpenClaw reads `--config` and `--cwd` from the workspace it starts in,
        so a relative path — a recordings folder named `--home rec` — pointed nowhere."""
        from tuningfork.harness import OpenClawAdapter

        monkeypatch.chdir(tmp_path)
        spec = OpenClawAdapter().launch("go", home=Path("rec/batches/b/0/home"), workspace=Path("rec/batches/b/0/workspace"))
        argv = list(spec.argv)
        assert argv[argv.index("--config") + 1] == str(tmp_path.resolve() / "rec/batches/b/0/home/.openclaw/openclaw.json")
        assert argv[argv.index("--cwd") + 1] == str(tmp_path.resolve() / "rec/batches/b/0/workspace")

    def test_each_option_turns_one_thing_off_and_is_recorded(self, tmp_path):
        """Off, the Skill Workshop's tool is denied: OpenClaw writes its section of the system
        prompt only beside the tool, and its `autonomous.mode` stops only a background review
        that never runs in a launch — seen on a real run, where it changed nothing sent."""
        from tuningfork.harness import OpenClawAdapter

        adapter = OpenClawAdapter(options={"tool_search": False, "skill_workshop": False, "memory_flush": True})
        config = _openclaw_config(adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m"))
        assert config["tools"] == {"toolSearch": False, "deny": ["skill_workshop"]}
        assert "skills" not in config and "compaction" not in config["agents"]["defaults"]  # OpenClaw's own flush
        assert adapter.settings == {"harness": {"tool_search": False, "skill_workshop": False, "memory_flush": True}}
        only = OpenClawAdapter(options={"skill_workshop": False})
        assert _openclaw_config(only.prepare_home(tmp_path / "b", base_url="http://p/v1", model="m"))["tools"] == {
            "deny": ["skill_workshop"]}
        assert OpenClawAdapter(options={"memory_flush": False}).settings == {}  # the default: nothing to record

    def test_mcp_servers_go_in_whole_and_the_settings_keep_only_their_commands(self, tmp_path):
        from tuningfork.harness import OpenClawAdapter

        servers = {"tuningfork": {"command": "/v/bin/tuningfork", "args": ["mcp", "--tools", "refine"],
                                  "env": {"KEY": "secret"}},
                   "remote": {"url": "http://127.0.0.1:9/mcp", "transport": "streamable-http",
                              "headers": {"Authorization": "Bearer secret"}}}
        adapter = OpenClawAdapter(options={"mcp_servers": servers})
        config = _openclaw_config(adapter.prepare_home(tmp_path / "home", base_url="http://p/v1", model="m"))
        assert config["mcp"] == {"servers": servers}
        assert adapter.settings == {"harness": {"mcp_servers": {"tuningfork": "/v/bin/tuningfork",
                                                                "remote": "http://127.0.0.1:9/mcp"}}}
        assert "secret" not in json.dumps(adapter.settings)

    def test_a_batch_tells_openclaw_the_servers_context_window_and_a_fork_its_sources(self, tmp_path):
        """OpenClaw never asks the server, and assumes 200,000 tokens: against a smaller one it
        would not compact before the server refused the prompt."""
        from tuningfork.harness import OpenClawAdapter

        servers = {"s": {"command": "c", "env": {"K": "v"}}}
        fitted = OpenClawAdapter(options={"tool_search": False, "mcp_servers": servers}).for_server({"n_ctx": 131072})
        assert fitted.options.context_window == 131072 and fitted.options.mcp_servers == servers
        assert fitted.settings["harness"] == {"tool_search": False, "mcp_servers": {"s": "c"}, "context_window": 131072}
        (model,) = _openclaw_config(fitted.prepare_home(tmp_path / "a", base_url="http://p/v1", model="m"))[
            "models"]["providers"]["local"]["models"]
        assert model["contextWindow"] == 131072
        # A person's own, or a fork's recorded one, wins; a server that does not say changes nothing.
        given = OpenClawAdapter(options={"context_window": 40000})
        assert given.for_server({"n_ctx": 131072}).options.context_window == 40000
        assert OpenClawAdapter().for_server({}).options.context_window is None
        assert OpenClawAdapter().for_server({"n_ctx": "big"}).options.context_window is None
        # Under OpenClaw's own floor, the window is not written: the batch's preflight refuses
        # such a server by its n_ctx requirement, in its own words.
        assert OpenClawAdapter().for_server({"n_ctx": 2048}).options.context_window is None

    @pytest.mark.parametrize("options, said", [
        ({"nope": 1}, "no option called 'nope'"),
        ({"tool_search": "off"}, "true or false"),
        ({"memory_flush": "on"}, "true or false"),
        ({"context_window": 0}, "number of tokens"),
        ({"context_window": True}, "number of tokens"),
        ({"context_window": 3000}, "at least 4,000 tokens"),  # OpenClaw refuses it before its first call
        ({"mcp_servers": {"x": {}}}, "needs a command"),
        ({"mcp_servers": {"bad name": {"command": "c"}}}, "maps each server"),
        ("x", "must be an object"),
    ])
    def test_what_openclaw_does_not_take_is_refused(self, options, said):
        from tuningfork.harness import OpenClawAdapter
        from tuningfork.ops.common import OperationError

        with pytest.raises(OperationError) as refused:
            OpenClawAdapter(options=options)
        assert refused.value.field == "harness_options" and said in refused.value.message

    def test_the_window_it_needs_follows_tool_search(self):
        """Measured on Gemma 4: the first request alone is about 9,000 tokens with Tool Search,
        and 23,000 with every tool's schema in it."""
        from tuningfork.harness import OpenClawAdapter

        def needs(adapter):
            return {r.what: r.minimum for r in adapter.requirements}

        assert needs(OpenClawAdapter()) == {"streaming_tool_calls": None, "n_ctx": 16_384}
        off = OpenClawAdapter(options={"tool_search": False})
        assert needs(off)["n_ctx"] == 32_768
        assert "Tool Search off" in next(r.why for r in off.requirements if r.what == "n_ctx")

    def test_a_skill_read_as_a_file_is_found_by_its_folder(self):
        """Tool Search on, as shipped: the model reads the location `<available_skills>` gives,
        and the result is the file, frontmatter and all, as `read` gives any file."""
        from tuningfork.harness import OpenClawAdapter

        call = {"id": "c1", "type": "function", "function": {
            "name": "read", "arguments": json.dumps({"path": "~/.openclaw/skills/tally-report/SKILL.md"})}}
        other = {"id": "c2", "type": "function", "function": {"name": "read", "arguments": '{"path":"data.csv"}'}}
        request = {"messages": [
            {"role": "system", "content": f"You are a personal assistant running inside OpenClaw.\n{OPENCLAW_LISTING}"},
            {"role": "user", "content": _openclaw_prompt("Sun 2026-10-25 02:32 EDT", "8c12999b-4161-406e-a1c3-efca3c0867f8")},
            {"role": "assistant", "content": None, "tool_calls": [call, other]},
            {"role": "tool", "tool_call_id": "c1", "content": TALLY_SKILL},
            {"role": "tool", "tool_call_id": "c2", "content": "id,amount\n1,2\n"},
        ]}
        adapter = OpenClawAdapter()
        (span,) = adapter.skill_spans(request)
        assert (span.name, span.message_index, span.source, span.text) == ("tally-report", 3, "tool", TALLY_SKILL)
        assert (span.start, span.end) == (0, len(TALLY_SKILL))
        assert adapter.skill_load(call) == "tally-report" and adapter.skill_load(other) is None

    def test_a_skill_read_with_skills_read_is_found_by_the_name_it_was_given(self):
        """Tool Search off: `skills_read` gives the file as it is. Through `tool_call`, under
        Tool Search, the same text comes inside the JSON `tool_call` returns."""
        from tuningfork.harness import OpenClawAdapter

        direct = {"id": "c1", "type": "function", "function": {"name": "skills_read", "arguments": '{"name":"tally-report"}'}}
        wrapped = {"id": "c2", "type": "function", "function": {
            "name": "tool_call", "arguments": json.dumps({"id": "skills_read", "args": {"name": "tally-report"}})}}
        envelope = json.dumps({
            "tool": {"id": "openclaw:core:skills_read", "name": "skills_read", "source": "openclaw"},
            "result": {"content": [{"type": "text", "text": TALLY_SKILL}],
                       "details": {"name": "tally-report", "content": TALLY_SKILL}},
        }, indent=2)
        adapter = OpenClawAdapter()
        first, second = adapter.skill_spans({"messages": [
            {"role": "assistant", "content": None, "tool_calls": [direct]},
            {"role": "tool", "tool_call_id": "c1", "content": TALLY_SKILL},
            {"role": "assistant", "content": None, "tool_calls": [wrapped]},
            {"role": "tool", "tool_call_id": "c2", "content": envelope},
        ]})
        assert (first.name, first.text, first.start, first.end) == ("tally-report", TALLY_SKILL, 0, len(TALLY_SKILL))
        assert (second.name, second.text, second.message_index, second.start) == ("tally-report", TALLY_SKILL, 3, None)
        assert adapter.skill_load(direct) == adapter.skill_load(wrapped) == "tally-report"
        other = {"function": {"name": "tool_call", "arguments": json.dumps({"id": "sessions_spawn", "args": {"name": "x"}})}}
        refused = '{\n  "status": "error",\n  "tool": "tool_call",\n  "error": "Invalid arguments"\n}'
        assert adapter.skill_load(other) is None
        assert adapter.skill_spans({"messages": [
            {"role": "assistant", "tool_calls": [wrapped]}, {"role": "tool", "tool_call_id": "c2", "content": refused},
        ]}) == []

    def test_a_read_is_named_as_skill_load_names_it(self):
        """Found in review: named by its listed name in a span and by its folder in `skill_load`,
        a scenario stopping at the next skill would stop at a read of its own. Both name it by
        its folder, the name tuningfork installs it under."""
        from tuningfork.harness import OpenClawAdapter

        listing = ("<available_skills>\n  <skill>\n    <name>tally-report</name>\n"
                   "    <location>/x/tally/SKILL.md</location>\n  </skill>\n</available_skills>")
        call = {"id": "c1", "function": {"name": "read", "arguments": '{"path":"/x/tally/SKILL.md"}'}}
        adapter = OpenClawAdapter()
        (span,) = adapter.skill_spans({"messages": [
            {"role": "system", "content": listing},
            {"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "c1", "content": "text"},
        ]})
        assert span.name == adapter.skill_load(call) == "tally"

    @pytest.mark.parametrize("answer", [
        # A skill OpenClaw has loaded is never paged: too long, it is refused; read again, OpenClaw says so.
        "Skill instructions cannot be partially served: the whole document exceeds this call's read or model-context "
        "budget. Ask the operator to reduce the document or increase the model context.",
        "Skill instructions were already served whole earlier in the current model context. Reuse that content; the "
        "full document will be served again if compaction removes it.",
        # A call that failed: OpenClaw's error, as JSON.
        '{\n  "status": "error",\n  "tool": "read",\n  "error": "ENOENT: no such file or directory"\n}',
    ])
    def test_what_openclaw_sends_in_place_of_a_skill_is_no_skill(self, answer):
        """Found in review: each of these was taken for a skill's text."""
        from tuningfork.harness import OpenClawAdapter

        read = {"id": "c1", "function": {"name": "read", "arguments": '{"path":"~/.openclaw/skills/x/SKILL.md"}'}}
        named = {"id": "c2", "function": {"name": "skills_read", "arguments": '{"name":"x"}'}}
        wrapped = {"id": "c3", "function": {"name": "tool_call", "arguments": '{"id":"skills_read","args":{"name":"x"}}'}}
        inside = json.dumps({"tool": {"id": "openclaw:core:skills_read"}, "result": {"content": [{"type": "text", "text": answer}]}})
        assert OpenClawAdapter().skill_spans({"messages": [
            {"role": "assistant", "tool_calls": [read, named, wrapped]},
            {"role": "tool", "tool_call_id": "c1", "content": answer},
            {"role": "tool", "tool_call_id": "c2", "content": answer},
            {"role": "tool", "tool_call_id": "c3", "content": inside},
        ]}) == []

    def test_a_chain_is_told_that_nothing_of_openclaws_changes_its_skill(self):
        from tuningfork.harness import OpenClawAdapter

        (note,) = OpenClawAdapter().chain_notes()
        assert "learned skills" in note and "background review never runs" in note

    @pytest.mark.parametrize("note", [
        "[812 more lines in file. Use offset=2001 to continue.]",
        "[Read output capped at 32KB for this call. Use offset=926 to continue.]",  # seen with a 40,000-token window
        "[Showing lines 1-1873 of 2812 (50KB limit). Use offset=1874 to continue.]",
        "[Showing part of line 1 (50KB limit). Use read with offset=1, cursor=51200 to continue.]",
    ])
    def test_a_long_unlisted_skills_continuation_note_is_not_its_text(self, note):
        """A SKILL.md that is not a skill OpenClaw has loaded is read as any file is: a part at
        a time — as much as the context window allows, 32 to 128 KB a call — with a note in one
        of three wordings saying where the rest begins."""
        from tuningfork.harness import OpenClawAdapter

        call = {"id": "c1", "function": {"name": "read", "arguments": '{"path":"/home/user/project/vendor/x/SKILL.md"}'}}
        (span,) = OpenClawAdapter().skill_spans({"messages": [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": f"{TALLY_SKILL}\n\n{note}"},
        ]})
        assert span.name == "x" and span.text == TALLY_SKILL and span.end == len(TALLY_SKILL)

    def test_other_harnesses_reads_are_not_openclaw_skills(self):
        """Every adapter's skill_spans runs over every run's requests (`ops.timeline`): Claude
        Code's `Read` and Hermes's `read_file` of a SKILL.md are their own business."""
        from tuningfork.harness import OpenClawAdapter

        for name in ("Read", "read_file"):
            call = {"id": "c1", "function": {"name": name, "arguments": '{"file_path":"/x/SKILL.md","path":"/x/SKILL.md"}'}}
            assert OpenClawAdapter().skill_spans({"messages": [
                {"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "c1", "content": "text"},
            ]}) == []

    def test_the_compaction_summary_is_a_side_call_and_its_messages_are_openclaws(self):
        """Seen forcing a compaction (a 40,000-token window): OpenClaw summarizes the
        conversation in calls of its own, then goes on from a summary and a turn it writes."""
        from tuningfork.harness import OpenClawAdapter

        adapter = OpenClawAdapter()
        summary = {"messages": [
            {"role": "system", "content": "You are a context summarization assistant. Your task is to read a conversation"},
            {"role": "user", "content": [{"type": "text", "text": "<conversation>\n[User]: Read big.txt …"}]},
        ]}
        task = {"messages": [{"role": "system", "content": "You are a personal assistant running inside OpenClaw."},
                             {"role": "user", "content": "Summarize data.csv"}]}
        assert adapter.side_call(summary) == "compaction summary"
        assert adapter.side_call(task) is None and adapter.side_call({"messages": []}) is None
        compacted = [{"type": "text", "text": "The conversation history before this point was compacted into the "
                                              "following summary:\n\n<summary>\n## Decisions\n…</summary>"}]
        assert adapter.harness_note({"role": "user", "content": compacted}) == "compaction summary"
        go_on = ("[Sun 2026-10-25 02:45 EDT] Continue the current task from the existing transcript, preserving "
                 "completed work. If an action was interrupted, inspect its state before deciding whether to retry it.")
        assert adapter.harness_note({"role": "user", "content": go_on}) == "continue after a compaction"
        person = {"role": "user", "content": _openclaw_prompt("Sun 2026-10-25 02:32 EDT", "8c12999b-4161-406e-a1c3-efca3c0867f8")}
        assert adapter.harness_note(person) is None

    def test_a_batch_folder_in_a_git_repository_is_warned_of(self, tmp_path):
        """Seen unsandboxed: OpenClaw read none of the instruction files planted above the
        workspace, but named the repository it was in, `repo=<root>`, in its Runtime line."""
        from tuningfork.harness import OpenClawAdapter

        (tmp_path / "repo" / ".git").mkdir(parents=True)
        (tmp_path / "repo" / "batches").mkdir()
        (warning,) = OpenClawAdapter().isolation_warnings(tmp_path / "repo" / "batches")
        assert f"repo={(tmp_path / 'repo').resolve()}" in warning and "--workdir" in warning
        (tmp_path / "elsewhere").mkdir()
        assert OpenClawAdapter().isolation_warnings(tmp_path / "elsewhere") == []

    def test_an_openclaw_on_a_node_it_does_not_run_on_is_said_to_be_unable_to_launch(self):
        """Found installing it: on Ubuntu's Node 22, `openclaw --version` answers and warns,
        where `agent exec` exits before its first call."""
        from tuningfork.harness import OpenClawAdapter

        said = ("OpenClaw 2026.10.7 (157d8c2)\n(node:17) ExperimentalWarning: SQLite is an experimental feature\n"
                "Running on an unsupported Node (22.22.1); diagnostics may show truncated text\n")
        problem = OpenClawAdapter().version_problem(said)
        assert problem and "22.22.1" in problem and "26.12.1" in problem
        assert OpenClawAdapter().version_problem("OpenClaw 2026.10.7 (157d8c2)\n") is None

    def test_the_time_and_session_do_not_take_a_fork_off_its_recording(self):
        """Found replaying launches: OpenClaw writes the minute before the prompt, a new session
        id in its `Runtime:` line, and the date in its system prompt."""
        from tuningfork.harness import OpenClawAdapter
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(time: str, session: str, date: str) -> dict:
            return {"model": "m", "messages": [
                {"role": "system", "content": f"You are a personal assistant.\n## Temporal Context\nCurrent date: {date}\n"},
                {"role": "user", "content": _openclaw_prompt(time, session)},
            ]}

        wire, volatile = get_wire(None), OpenClawAdapter.volatile
        one = request("Sun 2026-10-25 02:32 EDT", "8c12999b-4161-406e-a1c3-efca3c0867f8", "2026-10-25")
        two = request("Mon 2026-10-26 09:05 EDT", "2d2c7fdc-68dd-4a66-8c1b-119b0e05a2dd", "2026-10-26")
        assert masked_request(one, wire, volatile=volatile) == masked_request(two, wire, volatile=volatile)
        assert masked_request(one, wire) != masked_request(two, wire)

    def test_the_ids_a_tool_result_names_do_not_take_a_fork_off_its_recording(self):
        """Found forking a run whose `sessions_spawn` had failed: the fork departed at the step
        after, the result — through `tool_call`, JSON inside JSON — naming the child session and
        run it made. A command left running names its own process (source)."""
        from tuningfork.harness import OpenClawAdapter
        from tuningfork.replay import masked_request
        from tuningfork.wire import get_wire

        def request(child: str, run: str, slug: str, pid: int) -> dict:
            refused = json.dumps({"status": "error", "error": "published reply runtime missing for main",
                                  "childSessionKey": f"agent:main:subagent:{child}", "runId": run}, indent=2)
            wrapped = json.dumps({"tool": {"id": "openclaw:core:sessions_spawn", "name": "sessions_spawn"},
                                  "result": {"content": [{"type": "text", "text": refused}],
                                             "details": json.loads(refused)}}, indent=2)
            running = f"Command still running (session {slug}, pid {pid}). Use process to follow it."
            return {"model": "m", "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "c1", "content": wrapped},
                {"role": "tool", "tool_call_id": "c2", "content": refused},
                {"role": "tool", "tool_call_id": "c3", "content": running},
            ]}

        wire, volatile = get_wire(None), OpenClawAdapter.volatile
        one = request("d60c553d-c539-4462-b742-64d2f0863baf", "1f53135b-fd56-4b53-a934-eb259f0e7cd8", "calm-ember", 4210)
        two = request("7708c599-32a3-432a-b78e-a22cd2408c08", "49b16e5a-0fea-4102-90bb-4c2bd2f1e7a0", "quiet-nova", 77)
        assert masked_request(one, wire, volatile=volatile) == masked_request(two, wire, volatile=volatile)
        assert masked_request(one, wire) != masked_request(two, wire)
