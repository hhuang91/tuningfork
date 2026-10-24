"""Claude Code through tuningfork: its wire format end to end, and its harness adapter.

Everything a Hermes run gets — recording, breakpoints with edits, resampling, skill
mapping, A/B substitution — checked for a harness speaking Anthropic Messages, against the
mock server's `/v1/messages`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tuningfork.analysis.normalize import outcome_for
from tuningfork.demo.mock import MockTurn
from tuningfork.experiment.sampling import SamplingPlan, load_checkpoint, run_resample
from tuningfork.experiment.skill import substitute, text_sha256
from tuningfork.harness.claude_code import ClaudeCodeAdapter, anthropic_base
from tuningfork.proxy.app import create_browse_app
from tuningfork.specs import skill_bodies
from tuningfork.wire.anthropic import AnthropicMessages
from tests.helpers import all_steps, only_step, until
from tests.test_wire import SKILL_BODY, claude_code_request

ANTHROPIC = AnthropicMessages()
API = "/_tuningfork/api"
MESSAGES = "/v1/messages?beta=true"
SKILL_FILE = "---\nname: tally-report\ndescription: Tally a CSV.\n---\n\n" + SKILL_BODY


def write_turn(content: str = '{"count": 4}') -> MockTurn:
    return MockTurn(
        reasoning="Four rows.",
        tool_calls=[{"name": "Write", "arguments": {"file_path": "report.json", "content": content}}],
    )


class TestRecording:
    async def test_a_streamed_call_is_recorded_in_both_shapes(self, proxy_client, mock_server, recorder, settle):
        mock_server.script(write_turn())
        response = await proxy_client.post(MESSAGES, json=claude_code_request())
        await settle()

        assert response.headers["content-type"].startswith("text/event-stream")
        assert ANTHROPIC.parse(response.content, event_stream=True)["stop_reason"] == "tool_use"
        step = only_step(recorder.store)
        assert step.wire == "anthropic-messages"
        # The request exactly as Claude Code sent it: system blocks, content blocks.
        assert recorder.store.rebuild_request(step)["system"] == claude_code_request()["system"]
        # The response in the common shape, which outcomes and the UI read.
        stored = recorder.store.response(step)
        assert stored["choices"][0]["message"]["reasoning_content"] == "Four rows."
        assert outcome_for(stored).kind == "tool_call"
        assert step.finish_reason == "tool_calls"
        assert step.reasoning_chars == len("Four rows.")

    async def test_a_streamed_calls_arguments_are_stored_as_the_server_sent_them(
        self, proxy_client, mock_server, recorder, settle
    ):
        # what scoring writes back as the model's tokens: its spacing, characters and numbers
        written = '{"file_path":"naïve.md","content":"café — ✓","amount":61.50}'
        mock_server.script(MockTurn(tool_calls=[{"name": "Write", "arguments": json.loads(written),
                                                 "arguments_text": written}]))
        await proxy_client.post(MESSAGES, json=claude_code_request())
        await settle()

        stored = recorder.store.response(only_step(recorder.store))
        assert stored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == written

    async def test_pausing_answers_in_the_anthropic_shape(self, proxy_client):
        await proxy_client.post(f"{API}/pause")
        response = await proxy_client.post(MESSAGES, json=claude_code_request())
        assert response.status_code == 503
        assert response.json()["type"] == "error"


class TestBreakpoints:
    async def test_a_write_is_held_edited_and_delivered_as_anthropic(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        mock_server.script(write_turn('{"count": 4}'))
        await proxy_client.post(
            f"{API}/breakpoints", json={"phase": "response", "tool": "Write", "path_glob": "*.json"}
        )
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=claude_code_request()))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())

        # The hold is in the common shape, so the CLI and UI edit it as they would Hermes's.
        held = (await proxy_client.get(f"{API}/holds/{hold.id}")).json()
        assert held["wire"] == "anthropic-messages"
        assert [m["role"] for m in held["view"]][-1] == "user"
        edited = held["response"]
        function = edited["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps({"file_path": "report.json", "content": '{"row_count": 4}'})
        await proxy_client.post(f"{API}/holds/{hold.id}/continue", json={"response": edited})

        received = ANTHROPIC.parse((await call).content, event_stream=True)
        write = next(block for block in received["content"] if block["type"] == "tool_use")
        assert write["input"]["content"] == '{"row_count": 4}'
        # Thinking kept its signature, so Claude Code can send the block back as it was.
        assert received["content"][0]["signature"] == "sig-mock"

        await settle()
        step = only_step(recorder.store)
        model_wrote = recorder.store.response(step)["choices"][0]["message"]["tool_calls"][0]
        harness_got = recorder.store.delivered_response(step)["choices"][0]["message"]["tool_calls"][0]
        assert json.loads(model_wrote["function"]["arguments"])["content"] == '{"count": 4}'
        assert json.loads(harness_got["function"]["arguments"])["content"] == '{"row_count": 4}'

    async def test_a_held_calls_arguments_are_the_text_the_server_sent(
        self, proxy_client, proxy_app, mock_server, recorder, settle
    ):
        written = '{"file_path":"report.json","content":"café — ✓"}'
        mock_server.script(MockTurn(tool_calls=[{"name": "Write", "arguments": json.loads(written),
                                                 "arguments_text": written}]))
        await proxy_client.post(
            f"{API}/breakpoints", json={"phase": "response", "tool": "Write", "path_glob": "*.json"}
        )
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=claude_code_request()))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())

        held = (await proxy_client.get(f"{API}/holds/{hold.id}")).json()
        assert held["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == written
        await proxy_client.post(f"{API}/holds/{hold.id}/continue")
        received = ANTHROPIC.parse((await call).content, event_stream=True)
        assert next(b for b in received["content"] if b["type"] == "tool_use")["input"] == json.loads(written)

        await settle()
        stored = recorder.store.response(only_step(recorder.store))
        assert stored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == written

    async def test_a_dropped_stream_ends_with_an_anthropic_error_event(self, proxy_client, proxy_app):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=claude_code_request()))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())
        await proxy_client.post(f"{API}/holds/{hold.id}/drop")

        body = (await call).content.decode()
        assert "event: error" in body
        assert '"tuningfork_dropped"' in body

    async def test_a_dropped_json_call_is_an_anthropic_error(self, proxy_client, proxy_app):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(
            proxy_client.post(MESSAGES, json={**claude_code_request(), "stream": False})
        )
        (hold,) = await until(lambda: proxy_app.state.engine.holds())
        await proxy_client.post(f"{API}/holds/{hold.id}/drop")

        response = await call
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "tuningfork_dropped"

    async def test_context_conditions_read_inside_content_blocks(self, proxy_client, proxy_app):
        # The phrase is only in a tool_result block: matched through the view.
        await proxy_client.post(
            f"{API}/breakpoints",
            json={"phase": "request", "context_regex": "Launching skill", "after_tool": "Skill"},
        )
        call = asyncio.create_task(proxy_client.post(MESSAGES, json=claude_code_request()))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())
        await proxy_client.post(f"{API}/holds/{hold.id}/continue")
        assert (await call).status_code == 200


class TestAdapter:
    def test_the_home_points_every_model_tier_at_the_proxy(self, tmp_path):
        adapter = ClaudeCodeAdapter()
        home = adapter.prepare_home(tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="gemma")
        settings = json.loads((home / ".claude" / "settings.json").read_text())
        assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        assert {settings["env"][k] for k in (
            "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        )} == {"gemma"}
        # As installed: the config, and the skills in it, where Claude Code keeps them.
        assert adapter.skill_roots(home) == (home / ".claude" / "skills",)

    def test_a_fork_of_a_run_laid_out_flat_is_laid_out_flat(self, tmp_path):
        """How launches were laid out before layouts were recorded: a fork replays the paths
        its source named, so it is laid out as its source was."""
        flat = ClaudeCodeAdapter(layout="flat")
        home = flat.prepare_home(tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="gemma")
        assert flat.skill_roots(home) == (home / "config" / "skills",)
        assert flat.environment(home)["CLAUDE_CONFIG_DIR"] == str((home / "config").resolve())

    def test_a_launch_is_isolated_from_the_parent_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-real-key")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/somewhere/real")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        adapter = ClaudeCodeAdapter()
        home = adapter.prepare_home(tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="gemma")

        spec = adapter.launch("do it", home=home, workspace=tmp_path / "work", max_turns=5)

        assert spec.env["ANTHROPIC_API_KEY"] != "a-real-key"
        assert spec.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        assert spec.env["CLAUDE_CONFIG_DIR"] == str((home / ".claude").resolve())
        assert spec.env["HOME"] == str(home.resolve())
        assert spec.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert spec.argv[:3] == ("claude", "-p", "do it")
        assert "--dangerously-skip-permissions" in spec.argv
        assert spec.argv[spec.argv.index("--max-turns") + 1] == "5"

    @pytest.mark.parametrize(
        "base, root",
        [("http://h:8787/v1", "http://h:8787"), ("http://h:8787/v1/", "http://h:8787"), ("http://h:1", "http://h:1")],
    )
    def test_the_base_url_is_the_server_root(self, base, root):
        assert anthropic_base(base) == root

    def test_a_skill_is_found_where_the_skill_tool_put_it(self):
        view = ANTHROPIC.view(claude_code_request())
        (span,) = ClaudeCodeAdapter().skill_spans({"messages": view})
        assert span.name == "tally-report"
        assert span.source == "tool"
        assert span.text == SKILL_BODY.rstrip()
        assert view[span.message_index]["content"][span.start:span.end] == span.text

    def test_the_user_messages_it_writes_itself_are_named(self):
        """A skill's text after the `Skill` call, and a message of nothing but reminders, are
        Claude Code's own; a person's words are their turn, reminders beside them or not."""
        adapter = ClaudeCodeAdapter()
        view = ANTHROPIC.view(claude_code_request())
        assert adapter.harness_note(view[-1]) == "skill text"
        reminder = {"type": "text", "text": "<system-reminder>\nThe date has changed.\n</system-reminder>"}
        said = {"type": "text", "text": "Now the same for march.csv."}
        assert adapter.harness_note({"role": "user", "content": [reminder]}) == "reminder"
        assert adapter.harness_note({"role": "user", "content": reminder["text"] + "\n"}) == "reminder"
        assert adapter.harness_note({"role": "user", "content": [reminder, said]}) is None
        assert adapter.harness_note({"role": "user", "content": said["text"]}) is None
        assert adapter.harness_note({"role": "user", "content": [{"type": "image_url"}]}) is None

    def test_the_skill_a_call_loads_is_named(self):
        view = ANTHROPIC.view(claude_code_request())
        (loading,) = [call for message in view for call in message.get("tool_calls") or []]
        assert ClaudeCodeAdapter().skill_load(loading) == "tally-report"
        assert ClaudeCodeAdapter().skill_load({"function": {"name": "Write", "arguments": "{}"}}) is None


class TestSkillFiles:
    def test_a_body_without_frontmatter_maps_to_its_file_and_lines(self, store, tmp_path):
        root = tmp_path / "skills"
        (root / "tally-report").mkdir(parents=True)
        (root / "tally-report" / "SKILL.md").write_text(SKILL_FILE, encoding="utf-8")
        bodies = skill_bodies([root])
        assert bodies["tally-report/SKILL.md"]["first_line"] == 6

        step = store.record_step(request=claude_code_request(), wire="anthropic-messages")
        store.annotate_run(step.run_id, skill_hashes={"tally-report/SKILL.md": "not-the-body"},
                           parity={"skill_bodies": bodies})

        async def fetch():
            app = create_browse_app(store)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8788") as client:
                return (await client.get(f"{API}/runs/{step.run_id}/steps/0")).json()

        data = asyncio.run(fetch())
        assert data["wire"] == "anthropic-messages"
        assert data["view"][0]["role"] == "system"
        (skill,) = data["skills"]
        assert (skill["file"], skill["match"], skill["first_line"]) == ("tally-report/SKILL.md", "body", 6)

    def test_an_edited_message_is_marked_where_the_page_shows_it(self, store):
        """Found in review: an edit's indices count the Anthropic request's messages, but
        the page shows a view that puts `system` first and splits out tool results — the
        "edited" mark landed on the assistant's Read call."""
        sent = claude_code_request()
        edited = claude_code_request()
        edited["messages"][3]["content"][1]["text"] += "\nAlways name the keys."
        step = store.record_step(
            request=edited, harness_request=sent, wire="anthropic-messages",
            normalize=ANTHROPIC.normalize_message, edits={"request": {"messages": [3]}},
        )

        async def fetch():
            app = create_browse_app(store)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8788") as client:
                return (await client.get(f"{API}/runs/{step.run_id}/steps/0")).json()

        data = asyncio.run(fetch())
        (position,) = data["edited_messages"]
        assert "Always name the keys." in data["view"][position]["content"]
        assert position != 3

    def test_a_crlf_file_and_a_file_without_frontmatter_still_map(self, tmp_path):
        root = tmp_path / "skills"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        (root / "a" / "SKILL.md").write_bytes(SKILL_FILE.replace("\n", "\r\n").encode())
        (root / "b" / "SKILL.md").write_text("\n# Plain\n\nNo frontmatter.\n", encoding="utf-8")

        bodies = skill_bodies([root])

        assert bodies["a/SKILL.md"]["full_sha256"] == text_sha256(SKILL_FILE.strip())
        assert bodies["b/SKILL.md"]["first_line"] == 2
        assert bodies["b/SKILL.md"]["sha256"] == text_sha256("# Plain\n\nNo frontmatter.")

    def test_an_ab_swaps_the_body_where_claude_code_put_it(self):
        variant = SKILL_FILE.replace("totals", "row_count and amount_total")
        changed, spans = substitute(claude_code_request(), SKILL_FILE, variant)
        (span,) = spans
        assert span.form == "body"
        text = changed["messages"][3]["content"][1]["text"]
        assert "row_count and amount_total" in text
        assert text.startswith("Base directory for this skill:")

    def test_a_skill_in_a_system_field_is_found_too(self):
        request = {"system": [{"type": "text", "text": "Preamble.\n" + SKILL_FILE}], "messages": []}
        changed, spans = substitute(request, SKILL_FILE, SKILL_FILE.replace("totals", "sums"))
        assert spans[0].message_index == -1
        assert "sums" in changed["system"][0]["text"]


class TestResampling:
    async def test_a_claude_code_step_is_resampled_on_its_own_endpoint(self, store, mock_server):
        store.record_step(
            request=claude_code_request(),
            response=ANTHROPIC.canonical_response(ANTHROPIC.parse(
                json.dumps(mock_server._anthropic_message(write_turn())).encode(), False)),
            wire="anthropic-messages",
        )
        (step,) = all_steps(store)
        checkpoint = load_checkpoint(store, step.run_id, 0)
        assert checkpoint.messages[0]["role"] == "system"  # described through the view

        mock_server.script(write_turn(), write_turn('{"row_count": 4}'))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as http:
            experiment = await run_resample(
                store, checkpoint, SamplingPlan(n=2), base_url="http://mock.invalid", client=http
            )

        # (A GET of /v1/models first is the parity check.)
        assert {path for method, path, _ in mock_server.seen if method == "POST"} == {"/v1/messages"}
        assert "stream" not in mock_server.requests[-1]
        samples = store.samples(experiment.id)
        assert {sample.outcome_kind for sample in samples} == {"tool_call"}
        assert len({sample.outcome_key for sample in samples}) == 2


def test_two_runs_writing_the_same_file_in_their_own_workspaces_agree():
    def response(path: str) -> dict:
        call = {"id": "x", "type": "function",
                "function": {"name": "Write", "arguments": json.dumps({"file_path": path, "content": "{}"})}}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]},
                             "finish_reason": "tool_calls"}]}

    first = outcome_for(response("/home/u/.tuningfork/work/0a2f84ad/0/workspace/report.json"))
    second = outcome_for(response("/home/u/.tuningfork/work/4bb1283e/3/workspace/report.json"))
    elsewhere = outcome_for(response("/home/u/other/report.json"))
    assert first.key == second.key
    assert first.key != elsewhere.key
    assert "file_path='report.json'" in first.label


def test_an_absolute_path_into_the_workspace_and_a_relative_one_agree():
    """Observed on a real batch: Claude Code named `data.csv` in some runs and the absolute
    path to it in others, so runs that read the same file split at that step."""

    def read(path: str) -> dict:
        call = {"id": "x", "type": "function",
                "function": {"name": "Read", "arguments": json.dumps({"file_path": path})}}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]},
                             "finish_reason": "tool_calls"}]}

    absolute = outcome_for(read("/home/u/.tuningfork/batches/0a2f84ad/0/workspace/data.csv"))
    keys = {outcome_for(read(path)).key for path in ("data.csv", "./data.csv")}
    assert keys == {absolute.key}
    assert outcome_for(read("sub/data.csv")).key != absolute.key
    assert outcome_for(read("/data.csv")).key != absolute.key


class TestRunGrouping:
    """Claude Code re-encodes its last message on every call: string content becomes a
    one-block list carrying `cache_control`, then turns back into a string next time. A tree
    keyed on exact bytes read each call as a branch — observed on a real run as a
    "fork at depth 10"."""

    @staticmethod
    def requests() -> list[dict]:
        def marked(text: str) -> dict:
            return {"role": "system", "content": [
                {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}

        prompt = {"role": "user", "content": "Tally data.csv."}
        call = {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "data.csv"}}]}
        result = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "id,amount"}]}
        first = [prompt, marked("<total_tokens>100</total_tokens>")]
        second = [prompt, {"role": "system", "content": "<total_tokens>100</total_tokens>"},
                  call, result, marked("<total_tokens>90</total_tokens>")]
        third = second[:4] + [{"role": "system", "content": "<total_tokens>90</total_tokens>"},
                              call, result, marked("<total_tokens>80</total_tokens>")]
        return [{"model": "m", "max_tokens": 100, "stream": True, "messages": m}
                for m in (first, second, third)]

    async def test_one_run_whatever_the_cache_markers_do(self, proxy_client, recorder, settle):
        for request in self.requests():
            await proxy_client.post(MESSAGES, json=request)
            await settle()

        runs = recorder.store.runs()
        assert len(runs) == 1
        assert runs[0].forked_from_run is None
        steps = recorder.store.steps(runs[0].id)
        assert [step.idx for step in steps] == [0, 1, 2]
        # And every request still comes back exactly as Claude Code sent it.
        for step, sent in zip(steps, self.requests()):
            assert recorder.store.rebuild_request(step) == sent


def test_skill_arguments_appended_after_the_body_are_not_part_of_the_skill():
    """Claude Code appends a `Skill` call's arguments after the body; seen on a real run,
    where it made the text match no file."""
    request = claude_code_request()
    request["messages"][2]["content"][1]["input"]["args"] = "data.csv report.json"
    block = request["messages"][3]["content"][1]
    block["text"] += "\n\nARGUMENTS: data.csv report.json"
    view = ANTHROPIC.view(request)
    (span,) = ClaudeCodeAdapter().skill_spans({"messages": view})
    assert span.text == SKILL_BODY.rstrip()
    assert view[span.message_index]["content"][span.start:span.end] == span.text


def test_a_skill_that_itself_says_arguments_is_not_cut_short():
    """Only the paragraph the harness appended is cut: this call passed no arguments."""
    request = claude_code_request()
    block = request["messages"][3]["content"][1]
    block["text"] += "\n\nARGUMENTS: none are needed."
    view = ANTHROPIC.view(request)
    (span,) = ClaudeCodeAdapter().skill_spans({"messages": view})
    assert span.text.endswith("ARGUMENTS: none are needed.")


def test_a_skill_sharing_its_turn_with_other_text_is_still_found():
    """A reminder in the same user turn makes the view's content a list of blocks."""
    request = claude_code_request()
    request["messages"][3]["content"].append({"type": "text", "text": "<system-reminder>x</system-reminder>"})
    view = ANTHROPIC.view(request)
    (span,) = ClaudeCodeAdapter().skill_spans({"messages": view})
    assert span.name == "tally-report"
    assert span.text == SKILL_BODY.rstrip()
    assert span.start is None  # no offset into a single string: shown decoded


class TestIsolation:
    def test_memory_files_above_the_workspace_are_excluded(self, tmp_path):
        """Verified on a real run: a CLAUDE.md in the recordings folder reached the model."""
        adapter = ClaudeCodeAdapter()
        home = adapter.prepare_home(tmp_path / "home", base_url="http://127.0.0.1:8787/v1", model="m")
        workspace = tmp_path / "b" / "0" / "workspace"

        spec = adapter.launch("do it", home=home, workspace=workspace)

        settings = json.loads(spec.argv[spec.argv.index("--settings") + 1])
        excluded = settings["claudeMdExcludes"]
        assert f"{tmp_path.resolve().as_posix()}/CLAUDE.md" in excluded
        assert f"{tmp_path.resolve().as_posix()}/.claude/rules/**" in excluded
        assert "/CLAUDE.md" in excluded
        assert not any(p.startswith(f"{workspace.resolve().as_posix()}/") for p in excluded)

    def test_a_batch_folder_inside_a_repository_with_claude_folders_is_flagged(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".claude" / "skills").mkdir(parents=True)
        workdir = tmp_path / "runs"
        workdir.mkdir()

        (warning,) = ClaudeCodeAdapter().isolation_warnings(workdir)
        assert str(tmp_path / ".claude") in warning
        (tmp_path / ".claude" / "skills").rmdir()
        (tmp_path / ".claude").rmdir()
        assert ClaudeCodeAdapter().isolation_warnings(workdir) == []
