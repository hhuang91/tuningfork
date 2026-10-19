"""A conversation an agent had elsewhere, imported as a run (`ops.imports`): Hermes's session
file and its export, or a plain message list, each assistant turn a step whose request is rebuilt
from the history, and the skill under study found and kept beside the run."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tuningfork.ops import imports, refine, trash
from tuningfork.ops.common import OperationError

SKILL = "---\nname: tally-report\ndescription: totals into report.json\n---\n\n# Tally report\n\nWrite the totals to report.json.\n"
OTHER = "---\nname: csv-tidy\ndescription: tidy a CSV\n---\n\n# Tidy\n\nSort the rows.\n"
SYSTEM = "You are Hermes, an agent.\n\n## Skills\n- tally-report: totals into report.json\n"
PRELOADED = (
    '[IMPORTANT: The user launched this CLI session with the "tally-report" skill preloaded. Treat its '
    "instructions as active guidance for the duration of this session unless the user says otherwise.]\n\n"
    f"{SKILL}\n[Skill directory: /home/user/.hermes/skills/tally-report]"
)
TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def viewed(call_id: str, text: str, name: str) -> str:
    return json.dumps({"success": True, "name": name, "content": text})


def conversation(*skills: tuple[str, str]) -> list[dict]:
    """A Hermes conversation: the task, a skill_view for each skill, a write, and the reply. Rows as
    Hermes's export keeps them: ids, timestamps, the reasoning beside each answer."""
    rows: list[dict] = [{"id": 1, "role": "user", "content": "Summarize data.csv into report.json.", "timestamp": 1.0}]
    for number, (name, text) in enumerate(skills):
        rows.append({"id": 10 + number, "role": "assistant", "content": "", "reasoning_content": f"Load {name} first.",
                     "tool_calls": [call(f"v{number}", "skill_view", {"name": name})], "finish_reason": "tool_calls",
                     "timestamp": 2.0})
        rows.append({"id": 20 + number, "role": "tool", "tool_call_id": f"v{number}", "tool_name": "skill_view",
                     "content": viewed(f"v{number}", text, name), "timestamp": 3.0})
    rows += [
        {"id": 30, "role": "assistant", "content": "", "reasoning": "Now write it.", "finish_reason": "tool_calls",
         "tool_calls": [call("w1", "write_file", {"path": "report.json", "content": "{}"})], "timestamp": 4.0},
        {"id": 31, "role": "tool", "tool_call_id": "w1", "tool_name": "write_file", "content": '{"bytes_written": 2}',
         "timestamp": 5.0},
        {"id": 32, "role": "assistant", "content": "Done: report.json holds the totals.", "finish_reason": "stop",
         "timestamp": 6.0},
    ]
    return rows


def export(*sessions: list[dict]) -> str:
    """Hermes's export: one session a line, its system prompt beside its messages, no tools."""
    return "\n".join(json.dumps({
        "id": f"20261018_0900{number:02d}_ab12cd", "source": "cli", "model": "gemma-4-31b", "title": f"report {number}",
        "system_prompt": SYSTEM, "started_at": 1760000000.0 + number, "messages": rows,
    }) for number, rows in enumerate(sessions)) + "\n"


def snapshot(rows: list[dict], *, system: str = SYSTEM) -> str:
    """Hermes's session file: the system prompt and the tools beside the messages."""
    return json.dumps({"session_id": "20261018_100000_ef34ab", "model": "gemma-4-31b", "base_url": "http://x/v1",
                       "platform": "cli", "session_start": "2026-10-18T10:00:00", "system_prompt": system,
                       "tools": TOOLS, "message_count": len(rows), "messages": rows})


class TestReading:
    def test_a_file_of_several_sessions_asks_which(self):
        data = export(conversation(("tally-report", SKILL)), conversation())
        with pytest.raises(OperationError) as asked:
            imports.read_chat(data)
        assert asked.value.code == "choose_session" and "20261018_090000_ab12cd" in asked.value.hint
        listed = imports.sessions_in(data)
        assert [found["id"] for found in listed] == ["20261018_090000_ab12cd", "20261018_090001_ab12cd"]
        chat = imports.read_chat(data, session="20261018_090001")
        assert chat.format == "hermes-export" and chat.title == "report 1" and chat.harness == "hermes"

    def test_an_export_is_rebuilt_as_its_harness_sends_each_request(self):
        chat = imports.read_chat(export(conversation(("tally-report", SKILL))))
        assert [message["role"] for message in chat.messages] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert chat.steps == [2, 4, 6] and chat.tools is None and chat.model == "gemma-4-31b"
        second = chat.request(4)
        # Hermes sends no reasoning back: each answer keeps its own, and no later request carries it.
        assert not any("reasoning_content" in message or "reasoning" in message for message in second["messages"])
        assert chat.response(4)["choices"][0]["message"]["reasoning_content"] == "Now write it."
        assert chat.response(4)["choices"][0]["finish_reason"] == "tool_calls"
        # Stripped, as Hermes sends every message.
        assert second["messages"][0] == {"role": "system", "content": SYSTEM.strip()} and "tools" not in second

    def test_each_request_is_what_hermes_sends_not_what_it_kept(self):
        """Hermes strips each message and writes tool-call arguments compactly, keys sorted, on the
        copy it sends (found against a session recorded through the proxy: a document's
        note joined to an empty caption was kept with two newlines the model never saw)."""
        rows = [
            {"role": "user", "content": "[The user sent a document: 'receipt.pdf'.]\n\n"},
            {"role": "assistant", "content": "", "reasoning": "Read it.", "finish_reason": "tool_calls",
             "tool_calls": [{"id": "t1", "type": "function", "function": {
                 "name": "terminal", "arguments": '{"timeout": 30, "command": "pdftotext r.pdf - | grep Café"}'}}]},
            {"role": "tool", "tool_call_id": "t1", "tool_name": "terminal", "content": "  Total: $12.00\n"},
            {"role": "assistant", "content": "Filed.", "finish_reason": "stop"},
        ]
        chat = imports.read_chat(export(rows))
        sent = chat.request(chat.steps[1])["messages"]
        assert [message["content"] for message in sent if message["role"] != "assistant"] == [
            SYSTEM.strip(), "[The user sent a document: 'receipt.pdf'.]", "Total: $12.00"]
        (resent,) = sent[2]["tool_calls"]
        assert resent["function"]["arguments"] == '{"command":"pdftotext r.pdf - | grep Caf\\u00e9","timeout":30}'
        # The answer is the model's, as kept: only what the harness sends again is changed.
        (answered,) = chat.response(chat.steps[0])["choices"][0]["message"]["tool_calls"]
        assert answered["function"]["arguments"] == '{"timeout": 30, "command": "pdftotext r.pdf - | grep Café"}'
        # A list of messages from no harness tuningfork knows goes as it was kept.
        plain = imports.read_chat(json.dumps(rows))
        assert plain.request(plain.steps[1])["messages"][0]["content"].endswith("\n\n")

    def test_a_session_file_keeps_its_tools(self):
        chat = imports.read_chat(snapshot(conversation()))
        assert chat.format == "hermes-snapshot" and chat.tools == TOOLS and chat.request(chat.steps[0])["tools"] == TOOLS

    def test_a_plain_list_of_messages(self):
        chat = imports.read_chat(json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]))
        assert chat.format == "messages" and chat.steps == [1] and not chat.system_prompt and chat.harness is None

    def test_what_holds_no_conversation_is_refused(self):
        for data, said in (("", "empty"), ("{not json", "neither JSON"), ('{"a": 1}', "no conversation"),
                           (b"\xff\xfe", "UTF-8")):
            with pytest.raises(OperationError, match=said):
                imports.read_chat(data)


class TestTheSkill:
    def test_a_skill_loaded_by_a_call_is_found_from_the_step_that_reads_it(self, store):
        plan = imports.prepare_import(store, export(conversation(("tally-report", SKILL))))
        (found,) = plan.skills
        assert (found.name, found.source, found.step, found.text) == ("tally-report", "tool", 1, SKILL)
        assert plan.study == "tally-report" and plan.names == ["tally-report"]
        shown = plan.to_json()
        assert any("names no tools (Hermes's export keeps none)" in warning for warning in shown["warnings"])
        assert any("extracted from what the model read" in warning for warning in shown["warnings"])

    def test_a_preloaded_skill_is_in_context_from_the_first_step(self, store):
        plan = imports.prepare_import(store, snapshot(conversation(), system=SYSTEM + PRELOADED))
        (found,) = plan.skills
        assert (found.name, found.source, found.step) == ("tally-report", "preloaded", 0)
        assert not any("names no tools" in warning for warning in plan.warnings)

    def test_several_skills_are_each_kept_and_the_work_names_the_one_it_studies(self, store):
        """A run studies one skill at a time, but what it keeps is every skill the agent used: the
        one an A/B, a refinement or a scenario studies is named when it starts."""
        from tuningfork.ops import experiments
        from tuningfork.ops.timeline import run_timeline

        data = export(conversation(("tally-report", SKILL), ("csv-tidy", OTHER)))
        plan = imports.prepare_import(store, data)
        assert plan.names == ["tally-report", "csv-tidy"] and plan.study is None
        assert any("it keeps 2 skills, tally-report and csv-tidy" in warning for warning in plan.warnings)
        result = imports.execute_import(store, plan)
        run = store.get_run(result.run_id)
        assert result.skills == ["tally-report", "csv-tidy"] and result.skill is None
        assert sorted(run.skill_hashes) == ["csv-tidy/SKILL.md", "tally-report/SKILL.md"]
        assert "skill_snapshot_dir" not in run.parity
        # Each one's text maps to its own kept file, where the run's page shows it.
        assert {span["name"]: span["file"] for span in run_timeline(store, run.id).to_json()["skills"]} == {
            "tally-report": "tally-report/SKILL.md", "csv-tidy": "csv-tidy/SKILL.md"}

        rules = [{"if": "calls", "value": "write_file"}]
        with pytest.raises(OperationError) as asked:
            refine.create(store, run.id, "2", target_rules=rules)
        assert asked.value.code == "choose_skill" and asked.value.hint == "the skills it kept: tally-report, csv-tidy"
        made = refine.create(store, run.id, "2", target_rules=rules, skill="csv-tidy")
        assert made.skill == "csv-tidy" and made.data["baseline"]["path"].endswith("csv-tidy/SKILL.md")
        with pytest.raises(OperationError, match="kept no skill called 'nope'"):
            refine.create(store, run.id, "2", target_rules=rules, skill="nope")
        unnamed = experiments.recorded_skill(store, run.id)
        assert unnamed.kept == ["tally-report", "csv-tidy"] and unnamed.text is None
        assert experiments.recorded_skill(store, run.id, "tally-report").text == SKILL

        chosen = imports.prepare_import(store, data, skill="csv-tidy")
        assert chosen.study == "csv-tidy"
        studied = store.get_run(imports.execute_import(store, chosen).run_id)
        assert refine.run_skill_file(store, studied).read_text(encoding="utf-8") == OTHER
        with pytest.raises(OperationError, match="no skill called 'nope'"):
            imports.prepare_import(store, data, skill="nope")

    def test_each_skill_can_be_kept_from_its_own_folder(self, store, tmp_path):
        folders = {}
        for name, text in (("tally-report", SKILL), ("csv-tidy", OTHER)):
            folders[name] = tmp_path / "skills" / name
            folders[name].mkdir(parents=True)
            (folders[name] / "SKILL.md").write_text(text, encoding="utf-8")
        data = export(conversation(("tally-report", SKILL), ("csv-tidy", OTHER)))
        plan = imports.prepare_import(store, data, skill_dir=[str(folders["csv-tidy"]), str(folders["tally-report"])])
        assert plan.folders == {name: folder.resolve() for name, folder in folders.items()}
        assert not any("extracted" in warning for warning in plan.warnings)
        run = store.get_run(imports.execute_import(store, plan).run_id)
        assert {name: imports.imported_source(run, name) for name in plan.names} == {
            name: folder.resolve() for name, folder in folders.items()}
        with pytest.raises(OperationError, match="two folders hold csv-tidy"):
            imports.prepare_import(store, data, skill_dir=[str(folders["csv-tidy"])] * 2)

    def test_a_skill_in_a_harness_unknown_is_found_by_its_frontmatter(self, store):
        rows = [{"role": "system", "content": f"Follow this skill.\n\n{SKILL}"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": "done"}]
        plan = imports.prepare_import(store, json.dumps(rows))
        assert plan.harness == "generic" and [(skill.name, skill.source, skill.step) for skill in plan.skills] == [
            ("tally-report", "frontmatter", 0)]

    def test_the_skills_own_folder_is_kept_and_checked_against_what_the_model_read(self, store, tmp_path):
        folder = tmp_path / "tally-report"
        (folder / "scripts").mkdir(parents=True)
        (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (folder / "scripts" / "check.py").write_text("print('ok')\n", encoding="utf-8")
        data = export(conversation(("tally-report", SKILL)))
        plan = imports.prepare_import(store, data, skill_dir=str(folder))
        assert plan.study == "tally-report" and plan.folders == {"tally-report": folder.resolve()}
        assert not any("not the text the model read" in warning or "extracted" in warning for warning in plan.warnings)
        result = imports.execute_import(store, plan)
        kept = result.folder / "skills" / "tally-report"
        assert (kept / "scripts" / "check.py").is_file() and (kept / "SKILL.md").read_text(encoding="utf-8") == SKILL

        (folder / "SKILL.md").write_text(SKILL.replace("the totals", "every total"), encoding="utf-8")
        changed = imports.prepare_import(store, data, skill_dir=str(folder))
        assert any("is not the text the model read" in warning for warning in changed.warnings)


class TestTheRun:
    def test_an_imported_run_is_recorded_whole_and_the_step_tools_take_it(self, store):
        data = export(conversation(("tally-report", SKILL)))
        plan = imports.prepare_import(store, data, label="the report, as it went", task_id="tally-report/1")
        result = imports.execute_import(store, plan)
        run = store.get_run(result.run_id)
        steps = store.steps(run.id)
        assert result.steps == len(steps) == 3 and run.launch_id is None and not run.is_thread
        assert (run.label, run.task_id, run.harness) == ("the report, as it went", "tally-report/1", "hermes")
        assert [store.rebuild_request(step) for step in steps] == [plan.chat.request(index) for index in plan.chat.steps]
        assert store.response(steps[2])["choices"][0]["message"]["content"] == "Done: report.json holds the totals."
        # Each answer is found again in the next request, though it goes there as Hermes resends it.
        from tuningfork.ops.timeline import run_timeline

        assert [step["kept"] for step in run_timeline(store, run.id).steps] == ["same", "same", None]
        imported = run.parity["imported"]
        assert imported["format"] == "hermes-export" and imported["session_id"] == "20261018_090000_ab12cd"
        assert imported["skill"] == "tally-report" and imported["skill_source"] == "extracted"
        # The skill as a batch run keeps one: what a refinement and an A/B read.
        assert refine.run_skill_file(store, run).read_text(encoding="utf-8") == SKILL
        assert json.loads((result.folder / "chat.json").read_text(encoding="utf-8"))["format"] == "hermes-export"

        # A refinement at the step after the skill came in: its text is found there, in the tool's result.
        made = refine.create(store, run.id, "1", target_rules=[{"if": "calls", "value": "write_file"}])
        assert made.run_id == run.id and made.step_idx == 1

    def test_a_list_of_messages_named_as_hermess_goes_as_hermes_sends_it(self, store):
        rows = [{"role": "user", "content": "go\n\n"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"}, {"role": "assistant", "content": "done"}]
        plan = imports.prepare_import(store, json.dumps(rows), harness="hermes")
        assert plan.chat.harness == "hermes" and plan.chat.request(plan.chat.steps[1])["messages"][0]["content"] == "go"

    def test_deleted_for_good_its_folder_goes_with_it(self, store):
        result = imports.execute_import(store, imports.prepare_import(store, export(conversation(("tally-report", SKILL)))))
        assert result.folder.is_dir()
        trash.trash(store, "run", result.run_id)
        removed = trash.purge(store, "run", result.run_id)
        assert not result.folder.exists() and removed.to_json()["folders"] == 1

    def test_a_run_whose_file_named_no_tools_can_take_a_recorded_runs(self, store):
        from tests.helpers import as_response

        store.begin_run("recorded", "t/1")
        store.record_step(request={"model": "m", "messages": [{"role": "user", "content": "go"}], "tools": TOOLS},
                          response=as_response({"role": "assistant", "content": "ok"}))
        recorded = store.get_run("recorded")
        plan = imports.prepare_import(store, export(conversation(("tally-report", SKILL))), tools_from=recorded.id[:8])
        assert plan.chat.tools == TOOLS and plan.tools_from == recorded.id
        assert not any("names no tools" in warning for warning in plan.warnings)


def test_a_person_imports_from_the_terminal(store, tmp_path):
    from tuningfork.cli import app

    path = tmp_path / "session.jsonl"
    path.write_text(export(conversation(("tally-report", SKILL))), encoding="utf-8")
    runner = CliRunner()
    looked = runner.invoke(app, ["import", str(path), "--dry-run", "--home", str(store.home)], env={"COLUMNS": "160"})
    assert looked.exit_code == 0, looked.output
    assert "hermes-export" in looked.output and "tally-report (loaded by a call; in context from step 1)" in looked.output
    assert "nothing recorded" in looked.output and store.runs(limit=10) == []
    done = runner.invoke(app, ["import", str(path), "--home", str(store.home)], env={"COLUMNS": "160"})
    assert done.exit_code == 0, done.output
    (run,) = store.runs(limit=10)
    assert f"imported as run {run.id[:8]}" in done.output
    # Read back, it says what the file lacked where the tools would be.
    shown = runner.invoke(app, ["timeline", run.id[:8], "--home", str(store.home)], env={"COLUMNS": "160"})
    assert shown.exit_code == 0 and "no tools recorded: Hermes's export keeps none" in shown.output, shown.output


async def test_the_web_ui_previews_then_imports(proxy_client, recorder):
    data = export(conversation(("tally-report", SKILL)), conversation())
    asked = (await proxy_client.post("/_tuningfork/api/imports/preview", json={"content": data})).json()
    assert [found["id"] for found in asked["choose_session"]] == ["20261018_090000_ab12cd", "20261018_090001_ab12cd"]
    body = {"content": data, "session": "20261018_090000_ab12cd"}
    plan = (await proxy_client.post("/_tuningfork/api/imports/preview", json=body)).json()
    assert plan["steps"] == 3 and plan["skill"] == "tally-report" and plan["skills"][0]["step"] == 1
    refused = await proxy_client.post("/_tuningfork/api/imports/preview", json={**body, "extra": 1})
    assert refused.status_code == 400 and refused.json()["error"]["field"] == "extra"
    made = await proxy_client.post("/_tuningfork/api/imports", json=body)
    assert made.status_code == 201
    run = recorder.store.get_run(made.json()["run"])
    assert run is not None and run.parity["imported"]["skill"] == "tally-report"
    shown = (await proxy_client.get(f"/_tuningfork/api/runs/{run.id}/timeline")).json()
    assert shown["run"]["imported"]["format"] == "hermes-export"


# --- phase 2: a scenario of an import, whose runs continue the conversation ------------------------

EARLIER = [{"id": 2, "role": "user", "content": "Our data lives in data.csv; keep reports small.", "timestamp": 0.5},
           {"id": 3, "role": "assistant", "content": "Understood: small reports.", "reasoning_content": "Note it.",
            "finish_reason": "stop", "timestamp": 0.6}]


def chat_with_earlier_turns() -> str:
    return export(EARLIER + conversation(("tally-report", SKILL)))


class TestTheContext:
    def test_it_goes_after_the_harness_instructions_in_what_the_model_sees(self):
        from tuningfork.wire import get_wire

        context = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "noted"}]
        request = {"model": "m", "messages": [{"role": "system", "content": "be good"}, {"role": "user", "content": "now"}]}
        sent = get_wire("openai-chat").with_context(request, context)
        assert [m["content"] for m in sent["messages"]] == ["be good", "earlier", "noted", "now"]
        assert request["messages"][1]["content"] == "now"  # the harness's own request untouched
        assert get_wire("anthropic-messages").with_context(request, context) is None
        # The model's earlier steps of the prompt's own turn go right after the prompt, before
        # whatever the harness's own conversation holds after it.
        in_turn = [{"role": "assistant", "content": "", "tool_calls": [call("t1", "skill_view", {"name": "csv-tidy"})]},
                   {"role": "tool", "tool_call_id": "t1", "content": "tidy"}]
        going = {**request, "messages": [*request["messages"], {"role": "assistant", "content": "next"}]}
        sent = get_wire("openai-chat").with_context(going, context, in_turn)
        assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user", "assistant", "tool", "assistant"]
        assert sent["messages"][3]["content"] == "now" and sent["messages"][4:6] == in_turn
        assert get_wire("openai-chat").with_context({"messages": [{"role": "system", "content": "x"}]}, [], in_turn) is None

    def test_it_is_checked_as_a_conversation_a_harness_can_continue(self):
        from tuningfork.specs import SpecError, check_context, validate_spec

        assert check_context([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        for bad, said in (
            ([], "not empty"),
            ([{"role": "system", "content": "x"}], "user, assistant or tool"),
            ([{"role": "tool", "tool_call_id": "c1", "content": "x"}], "no earlier message made"),
            ([{"role": "assistant", "content": "", "tool_calls": [call("c1", "f", {})]}], "not answered"),
            ([{"role": "user", "content": "a"}], "ends on a user turn"),
        ):
            with pytest.raises(SpecError, match=said):
                check_context(bad)
        problems = validate_spec({"skill_name": "s", "evals": [{"id": "1", "prompt": "p", "tuningfork": {"context": "../x.json"}}]})
        assert any(problem.where.endswith("tuningfork.context") for problem in problems)

    def test_the_prompts_own_turn_is_checked_as_the_model_left_it(self):
        from tuningfork.specs import SpecError, check_context, validate_spec

        loaded = [{"role": "assistant", "content": "", "tool_calls": [call("t1", "skill_view", {"name": "csv-tidy"})]},
                  {"role": "tool", "tool_call_id": "t1", "content": "tidy"}]
        found = check_context({"before": [], "in_turn": loaded})
        assert found.before == [] and found.in_turn == loaded and found.to_json() == {"before": [], "in_turn": loaded}
        assert check_context([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]).to_json() == [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        for bad, said in (
            ({"before": [], "in_turn": []}, "holds no message"),
            ({"in_turn": [{"role": "tool", "tool_call_id": "t1", "content": "x"}]}, "starts with the model's first answer"),
            ({"in_turn": loaded[:1]}, "in_turn ends with a tool call not answered"),
            ({"in_turn": [*loaded, {"role": "assistant", "content": "done"}]}, "the turn was over"),
            ({"before": [], "after": []}, "must be a list of messages, or"),
        ):
            with pytest.raises(SpecError, match=said):
                check_context(bad)
        stopping = {"skill_name": "s", "evals": [{"id": "1", "prompt": "p", "tuningfork": {"stop_at_next_skill": "yes"}}]}
        assert any(problem.where.endswith("stop_at_next_skill") for problem in validate_spec(stopping))


class TestAScenarioOfAnImport:
    def test_it_starts_at_the_turn_before_the_skill_came_in(self, store):
        from tuningfork.ops import scenarios

        result = imports.execute_import(store, imports.prepare_import(store, chat_with_earlier_turns()))
        plan = imports.prepare_scenario(store, result.run_id, "tally-chat", check={"reply_contains": "report.json"})
        # system 0, the earlier turn 1-2, the task's turn 3, then the skill_view.
        assert plan.cut == 3 and plan.prompt == "Summarize data.csv into report.json."
        assert plan.context.in_turn == [] and plan.context.before == [{"role": "user", "content": "Our data lives in data.csv; keep reports small."},
                                {"role": "assistant", "content": "Understood: small reports."}]  # no reasoning sent back
        assert any(f"the scenario keeps the copy run {result.run_id[:8]} kept, in its own folder" in warning
                   for warning in plan.warnings)
        made = imports.make_scenario(store, plan)
        assert made.checked.ok and made.context == plan.context.before
        (item,) = made.spec["evals"]
        # Its own task: never counted with another spec's runs of the skill's first eval.
        assert item["id"] == "tally-chat" and item["prompt"] == plan.prompt and item["tuningfork"]["context"] == "context.json"
        assert item["tuningfork"]["check"] == {"reply_contains": "report.json"}
        # A copy of its own: an approved edit never rewrites what the imported run read.
        assert made.spec["tuningfork"]["skill_dir"] == "skill/tally-report"
        assert (scenarios.folder(store.home, "tally-chat") / "skill" / "tally-report" / "SKILL.md").read_text(encoding="utf-8") == SKILL
        assert (result.folder / "skills" / "tally-report" / "SKILL.md").read_text(encoding="utf-8") == SKILL
        assert any("its conversation's calls name report.json" in warning for warning in plan.warnings)

        with pytest.raises(OperationError, match="exists already"):
            imports.prepare_scenario(store, result.run_id, "tally-chat")
        with pytest.raises(OperationError, match="not a user turn"):
            imports.prepare_scenario(store, result.run_id, "other", cut=2)
        # The skill came in after the person's second message: a run sent their first would never vary it.
        with pytest.raises(OperationError, match="came in after message 3, which a run starting from message 1 is never sent") as refused:
            imports.prepare_scenario(store, result.run_id, "from-the-start", cut=1)
        assert refused.value.field == "cut" and "start from message 3" in refused.value.hint
        unchecked = imports.prepare_scenario(store, result.run_id, "unchecked")
        assert any("no check" in warning for warning in unchecked.warnings)

        # The starting prompt in the person's own words: the conversation before it as the chat had it.
        mine = imports.prepare_scenario(store, result.run_id, "in-my-words", prompt="Summarize inputs/data.csv.")
        assert mine.prompt == "Summarize inputs/data.csv." and mine.spec()["evals"][0]["prompt"] == mine.prompt
        assert mine.context.before == plan.context.before and mine.cut == 3
        with pytest.raises(OperationError, match="the starting prompt is empty") as empty:
            imports.prepare_scenario(store, result.run_id, "no-words", prompt=" \n")
        assert empty.value.field == "prompt"

    def test_a_skill_the_chat_never_shows_starts_from_any_turn(self, store, tmp_path):
        """A skill the conversation never showed, named by its folder: a run can start from any of
        the person's turns, with what came before it as its context."""
        rows = [{"role": "user", "content": "Look at data.csv."}, {"role": "assistant", "content": "It has 4 rows."},
                {"role": "user", "content": "Now summarize it into report.json."}, {"role": "assistant", "content": "Done."}]
        result = imports.execute_import(store, imports.prepare_import(store, export(rows)))
        choices = imports.scenario_choices(store, result.run_id)
        assert choices["skill"] is None and choices["cut"] == 1 and choices["prompt"] == "Look at data.csv."
        assert [turn["offered"] for turn in choices["turns"]] == [True, True]
        later = imports.scenario_choices(store, result.run_id, cut=3)
        assert later["prompt"] == "Now summarize it into report.json." and later["context"] == 2
        assert [(entry["step"], entry["part"]) for entry in later["steps"]] == [(1, "first")]
        folder = tmp_path / "tally-report"
        folder.mkdir()
        (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
        plan = imports.prepare_scenario(store, result.run_id, "unseen", skill_dir=str(folder), cut=3)
        assert plan.cut == 3 and len(plan.context.before) == 2 and plan.skill == "tally-report"
        assert any("never came into the conversation: each run starts from the turn chosen" in warning
                   for warning in plan.warnings)

    def test_its_conversation_goes_as_its_harness_would_have_sent_it(self, store):
        """The proxy puts the conversation in after the harness has made its own copy ready, so it
        is kept as the harness would have sent it."""
        earlier = [{"id": 2, "role": "user", "content": "Our data lives in data.csv.\n", "timestamp": 0.5},
                   {"id": 3, "role": "assistant", "content": "Understood.  \n", "finish_reason": "stop", "timestamp": 0.6}]
        data = export(earlier + conversation(("tally-report", SKILL)))
        result = imports.execute_import(store, imports.prepare_import(store, data))
        plan = imports.prepare_scenario(store, result.run_id, "as-sent")
        assert plan.context.before == [{"role": "user", "content": "Our data lives in data.csv."},
                                {"role": "assistant", "content": "Understood."}]

    def test_a_skill_loaded_later_in_its_turn_starts_after_the_steps_before_it(self, store):
        """One turn that loaded two skills: a scenario of the second starts after the first one's
        load and result, taken as done, and one of the first stops its runs where the chat went on
        to the second."""
        from tuningfork.ops import scenarios

        rows = conversation(("csv-tidy", OTHER), ("tally-report", SKILL))
        result = imports.execute_import(store, imports.prepare_import(store, export(rows)))
        plan = imports.prepare_scenario(store, result.run_id, "after-tidy", skill="tally-report")
        assert plan.cut == 1 and plan.context.before == [] and plan.next_skill is None
        assert [message["role"] for message in plan.context.in_turn] == ["assistant", "tool"]
        assert plan.context.in_turn[0]["tool_calls"][0]["function"]["arguments"] == '{"name":"csv-tidy"}'  # as sent
        assert any("each run starts at step 1, where the chat loaded tally-report, with the turn's step 0 taken as done"
                   in warning for warning in plan.warnings)
        # The chat from the prompt on, each step with what a run does with it.
        assert [(entry["step"], entry["part"], entry["stop"]) for entry in plan.steps] == [
            (0, "done", False), (1, "first", False), (2, "live", False), (3, "live", False)]
        assert plan.steps[0]["label"] == "skill_view(name='csv-tidy')" and plan.to_json()["steps"] == plan.steps
        made = imports.make_scenario(store, plan)
        assert made.checked.ok and made.spec["evals"][0]["tuningfork"]["context"] == "context.json"
        assert scenarios.read_context(store.home, "after-tidy") == {"before": [], "in_turn": plan.context.in_turn}
        assert "stop_at_next_skill" not in made.spec["evals"][0]["tuningfork"]

        first = imports.prepare_scenario(store, result.run_id, "tidy-only", skill="csv-tidy",
                                         check={"reply_contains": "report"})
        assert not first.context.messages and first.next_skill == {"skill": "tally-report", "step": 1}
        assert [(entry["step"], entry["part"], entry["stop"]) for entry in first.steps] == [
            (0, "first", False), (1, "live", True), (2, "past", False), (3, "past", False)]
        assert first.spec()["evals"][0]["tuningfork"]["stop_at_next_skill"] is True
        assert any("each run stops where the model loads a skill other than csv-tidy" in warning
                   and "a check of the final reply finds none there" in warning for warning in first.warnings)

    def test_the_skills_own_folder_is_the_one_it_tunes(self, store, tmp_path):
        folder = tmp_path / "tally-report"
        folder.mkdir()
        (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
        result = imports.execute_import(store, imports.prepare_import(store, chat_with_earlier_turns(), skill_dir=str(folder)))
        plan = imports.prepare_scenario(store, result.run_id, "own-folder")
        assert plan.skill_dir == folder.resolve() and not any("extracted" in warning for warning in plan.warnings)

    def test_a_run_recorded_as_it_happened_keeps_the_skill_its_model_read(self, store):
        """A run that came through the proxy outside any launch: its conversation is its last
        request and the answer to it, its skill the text the model read, kept in the scenario's
        own folder, and the files its calls name are listed for a person to add."""
        from tuningfork.ops import scenarios

        chat = imports.read_chat(chat_with_earlier_turns())
        store.begin_run("as it happened", None)
        for index in chat.steps:
            store.record_step(request=chat.request(index), response=chat.response(index), harness="hermes")
        run = store.get_run("as it happened")
        assert not run.parity.get("imported") and not run.launch_id
        choices = imports.scenario_choices(store, run.id)
        assert (choices["skill"], choices["cut"], choices["launched"]) == ("tally-report", 3, False)
        # Only the turn the skill came in after: a run sent the earlier one would never vary it.
        assert [(turn["message"], turn["offered"]) for turn in choices["turns"]] == [(1, False), (3, True)]
        assert choices["prompt"] == "Summarize data.csv into report.json."
        assert [(entry["step"], entry["part"]) for entry in choices["steps"]] == [(1, "first"), (2, "live"), (3, "live")]
        plan = imports.prepare_scenario(store, run.id, "as-it-happened")
        assert plan.cut == 3 and len(plan.context.before) == 2 and plan.extract == SKILL and not plan.files
        assert any("the scenario keeps the SKILL.md the model read" in warning for warning in plan.warnings)
        assert any("its conversation's calls name report.json" in warning for warning in plan.warnings)
        made = imports.make_scenario(store, plan)
        assert made.checked.ok and made.spec["tuningfork"]["skill_dir"] == "skill/tally-report"
        kept = scenarios.folder(store.home, "as-it-happened") / "skill" / "tally-report" / "SKILL.md"
        assert kept.read_text(encoding="utf-8") == SKILL

    def test_a_claude_code_run_starts_in_the_persons_own_words(self, store):
        """Claude Code's reminders beside the person's words are its own, written again in every
        run; the skill it showed without frontmatter is given one, so the harness lists it. A
        conversation before the prompt it cannot carry: refused, not made without it."""
        from tests.helpers import as_response, tool_call_message
        from tests.test_wire import claude_code_request

        step = store.record_step(request=claude_code_request(), wire="anthropic-messages", harness="claude-code",
                                 response=as_response(tool_call_message("Write", {"file_path": "report.json", "content": "{}"})))
        plan = imports.prepare_scenario(store, step.run_id, "from-claude-code")
        assert plan.harness == "claude-code" and plan.cut == 1 and not plan.context.messages
        assert plan.prompt == "Use the tally-report skill on data.csv."
        assert plan.extract.startswith('---\nname: tally-report\ndescription: "Tally report"\n---\n\n# Tally report')
        assert any("without its frontmatter" in warning for warning in plan.warnings)
        assert imports.make_scenario(store, plan).checked.ok

        request = claude_code_request()
        request["messages"][:0] = [{"role": "user", "content": "An earlier turn."}, {"role": "assistant", "content": "Noted."}]
        later = store.record_step(request=request, wire="anthropic-messages", harness="claude-code",
                                  response=as_response(tool_call_message("Write", {"file_path": "report.json", "content": "{}"})))
        with pytest.raises(OperationError, match=r"claude-code's calls \(anthropic-messages\) cannot carry the conversation"):
            imports.prepare_scenario(store, later.run_id, "carried", cut=3)

    def test_a_subagents_thread_is_not_a_task(self, store, monkeypatch):
        import dataclasses

        from tests.helpers import record_run, text_message
        from tuningfork.store.db import SUBAGENT

        run = store.get_run(record_run(store, "a subagent", [text_message("done")]))
        monkeypatch.setattr(store, "get_run", lambda reference: dataclasses.replace(run, role=SUBAGENT, parent_run_id="f" * 32))
        with pytest.raises(OperationError, match="make the scenario of the run it belongs to"):
            imports.prepare_scenario(store, run.id, "x")

    def test_its_conversation_is_edited_whole_and_checked(self, store):
        from tuningfork.ops import scenarios

        result = imports.execute_import(store, imports.prepare_import(store, chat_with_earlier_turns()))
        imports.make_scenario(store, imports.prepare_scenario(store, result.run_id, "edited"))
        shorter = [{"role": "user", "content": "keep it short"}, {"role": "assistant", "content": "ok"}]
        assert scenarios.put_context(store.home, "edited", shorter) == 2
        assert scenarios.read_scenario(store.home, "edited").context == shorter
        with pytest.raises(OperationError, match="ends on a user turn"):
            scenarios.put_context(store.home, "edited", shorter[:1])
        (scenarios.folder(store.home, "edited") / "context.json").unlink()
        problems = scenarios.read_scenario(store.home, "edited").checked.problems
        assert any(problem["where"].endswith("tuningfork.context") for problem in problems)


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    from tests.helpers import task_stack

    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        yield built


def test_a_scenarios_runs_continue_its_conversation_and_their_forks_too(stack):
    """Each call of the run's main thread reaches the model with the conversation after the
    harness's instructions; the harness's own request is kept beside it, the run still counts,
    and a fork of it carries the same conversation, its replay matched on the harness's view."""
    from tests.test_forks import settled

    context = [{"role": "user", "content": "Earlier: the data is in data.csv."},
               {"role": "assistant", "content": "Noted, I will read data.csv."}]
    (stack.tmp / "context.json").write_text(json.dumps(context), encoding="utf-8")
    stack.item.tuningfork.context = "context.json"
    source = stack.record()
    assert source.usable and not source.edited_steps
    steps = settled(stack.store, source.run_id, 3)
    # Its calls carried the conversation it continued, whose turns are none of its steps.
    chat = imports.chat_of(stack.store, stack.store.get_run(source.run_id))
    assert [chat.step_of(index) for index in chat.steps] == [None, 0, 1, 2]
    for step in steps:
        sent = stack.store.rebuild_request(step)["messages"]
        assert sent[0]["role"] == "system" and sent[1:3] == context and step.context["messages"] == 2
        assert stack.store.harness_request(step)["messages"][1]["content"] == "Summarize data.csv into report.json."
        assert not step.edited and not step.substitution
    assert all(request["messages"][1:3] == context for request in stack.mock.requests[-3:])
    launch = stack.store.get_launch(stack.store.get_run(source.run_id).launch_id)
    assert launch.settings["context"]["messages"] == 2 and launch.settings["context"]["blob"] in stack.store.live_blobs()

    _, (fork,) = stack.fork(source.run_id, "1")
    assert fork.usable and fork.departed is None and fork.reached
    forked = settled(stack.store, fork.run_id, 3)
    assert forked[0].served and all(stack.store.rebuild_request(step)["messages"][1:3] == context for step in forked)
    assert stack.mock.requests[-1]["messages"][1:3] == context


def test_a_run_takes_its_turns_earlier_steps_as_done_and_stops_at_the_next_skill(stack, monkeypatch):
    """The model's earlier steps of the prompt's turn reach it after the prompt in every call of
    the run; the run ends where the model loads a skill other than the one studied: the call
    after that answer is never sent, the harness is ended, and the check judges the workspace
    as it is then. The run still counts, and says where it stopped."""
    import re

    from tests.helpers import FakeAgent
    from tests.mock_server import MockTurn
    from tuningfork.harness import ADAPTERS

    class Loading(FakeAgent):
        def skill_load(self, call):
            function = call.get("function") or {}
            return json.loads(function.get("arguments") or "{}").get("name") if function.get("name") == "skill_view" else None

    monkeypatch.setitem(ADAPTERS, "fake", Loading)
    in_turn = [{"role": "assistant", "content": "", "tool_calls": [call("t0", "skill_view", {"name": "csv-tidy"})]},
               {"role": "tool", "tool_call_id": "t0", "content": "the csv-tidy skill: sort the rows"}]
    (stack.tmp / "context.json").write_text(json.dumps({"before": [], "in_turn": in_turn}), encoding="utf-8")
    stack.item.tuningfork.context = "context.json"
    stack.item.tuningfork.stop_at_next_skill = True

    def model(payload):
        messages = payload["messages"]
        assert [m["role"] for m in messages[:2]] == ["system", "user"] and messages[2:4] == in_turn
        called = [c["function"]["name"] for m in messages[4:] if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        workdir = re.search(r"Working directory: (\S+)", messages[0]["content"]).group(1)
        if not called:
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/report.json", "content": json.dumps({"row_count": 4, "amount_total": 61.5})}}])
        if called == ["write_file"]:
            return MockTurn(tool_calls=[{"name": "skill_view", "arguments": {"name": "another-skill"}}])
        return MockTurn(content="went on past the next skill")

    stack.mock.responder = model
    outcome = stack.record()
    assert outcome.stopped == {"skill": "another-skill", "after_step": 1} and outcome.usable, outcome.stderr
    assert outcome.passed  # the check judged the report written before the stop
    steps = stack.store.steps(outcome.run_id)
    assert [step.idx for step in steps] == [0, 1] and all(step.context == {"messages": 0, "in_turn": 2} for step in steps)
    assert all(len(stack.store.harness_request(step)["messages"]) + 2 == len(stack.store.rebuild_request(step)["messages"])
               for step in steps)
    assert len(stack.mock.requests) == 2  # the call after the skill's load never reached the model
    run = stack.store.get_run(outcome.run_id)
    assert run.result["stopped_at_skill"] == {"skill": "another-skill", "after_step": 1}


def test_a_launched_runs_scenario_keeps_its_task_its_workspace_and_what_its_check_reads(stack):
    """A run tuningfork launched: the scenario's inputs are its workspace where the scenario
    starts, from the snapshots; its check the task's own, what the check reads from the task's
    folder copied beside the scenario's spec; its skill the task's folder; its eval named by the
    scenario, a task of its own. Its runs pass as the task's did."""
    from tests.helpers import FakeAgent
    from tuningfork.ops import scenarios
    from tuningfork.runner import BatchRunner
    from tuningfork.specs import load_spec

    checks = stack.tmp / "checks"
    checks.mkdir()
    # The original wording names no keys, so the task's own check fails it: this one asks for the file.
    (checks / "report.py").write_text("import os, sys\nsys.exit(0 if os.path.isfile('report.json') else 1)\n",
                                      encoding="utf-8")
    stack.item.tuningfork.check = 'python3 "$TUNINGFORK_SPEC_DIR/checks/report.py"'
    stack.spec.path.write_text(json.dumps(stack.spec.to_dict()), encoding="utf-8")
    source = stack.record()
    assert source.passed, source.stderr

    choices = imports.scenario_choices(stack.store, source.run_id)
    assert choices["launched"] and choices["check"] == stack.item.tuningfork.check and choices["cut"] == 1
    assert choices["prompt"] == stack.item.prompt
    plan = imports.prepare_scenario(stack.store, source.run_id, "tally-again")
    assert (plan.cut, plan.prompt, plan.context.messages) == (1, stack.item.prompt, 0)
    assert plan.skill == "tally-report" and not plan.own_skill
    assert plan.skill_dir.resolve() == (stack.tmp / "skills-src" / "tally-report").resolve()
    assert plan.files == {"data.csv": b"id,amount\n1,10\n2,20\n3,30\n4,1.5\n"}
    assert plan.check == stack.item.tuningfork.check and set(plan.check_files) == {"checks"}
    assert any("its check is task tally-report/1's own" in warning for warning in plan.warnings)
    made = imports.make_scenario(stack.store, plan)
    assert made.checked.ok, made.checked.problems
    (item,) = made.spec["evals"]
    assert item["id"] == "tally-again" and item["files"] == [{"path": "data.csv", "source": "inputs/data.csv"}]
    assert item["tuningfork"]["max_turns"] == 6 and item["tuningfork"]["skills"] == ["tally-report"]
    folder = scenarios.folder(stack.store.home, "tally-again")
    assert (folder / "checks" / "report.py").is_file()

    spec = load_spec(folder / "evals.json")
    runner = BatchRunner(stack.store, FakeAgent(), base_url=f"{stack.url}/v1", model="m", workdir=stack.tmp / "again",
                         settle=0.1)
    (again,) = runner.run(spec, spec.evals[0], 1).outcomes
    assert again.passed, again.stderr  # its check found its script beside the scenario's spec
    assert stack.store.get_run(again.run_id).task_id == "tally-report/tally-again"


def test_a_launched_runs_scenario_from_later_in_its_turn_has_its_workspace_there(stack, monkeypatch):
    """Where the skill came in after earlier steps of the turn, the scenario starts after them,
    taken as done, and its inputs are the workspace as the run had it then: what those steps
    wrote among them."""
    import re

    from tests.helpers import SKILL as LOADED, FakeAgent
    from tests.mock_server import MockTurn
    from tuningfork.harness import ADAPTERS
    from tuningfork.harness.base import SkillSpan

    class Loading(FakeAgent):
        def skill_load(self, call):
            function = call.get("function") or {}
            return json.loads(function.get("arguments") or "{}").get("name") if function.get("name") == "skill_view" else None

        def skill_spans(self, request):
            # The mock gives every call one id: a load's result is the tool message after it.
            spans, loading = [], None
            for index, message in enumerate(request.get("messages") or []):
                if message.get("role") == "assistant":
                    loading = next(filter(None, map(self.skill_load, message.get("tool_calls") or [])), None)
                elif message.get("role") == "tool" and loading:
                    spans.append(SkillSpan(name=loading, message_index=index, source="tool", text=LOADED))
                    loading = None
            return spans

    def model(payload):
        messages = payload["messages"]
        workdir = re.search(r"Working directory: (\S+)", messages[0]["content"]).group(1)
        called = [each["function"]["name"] for m in messages if m["role"] == "assistant" for each in m.get("tool_calls") or []]
        if not called:
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": f"{workdir}/notes.txt", "content": "rows: 4"}}])
        if called == ["write_file"]:
            return MockTurn(tool_calls=[{"name": "skill_view", "arguments": {"name": "tally-report"}}])
        if called == ["write_file", "skill_view"]:
            return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
                "path": f"{workdir}/report.json", "content": json.dumps({"row_count": 4, "amount_total": 61.5})}}])
        return MockTurn(content="done")

    monkeypatch.setitem(ADAPTERS, "fake", Loading)
    stack.mock.responder = model
    source = stack.record()
    plan = imports.prepare_scenario(stack.store, source.run_id, "after-notes")
    assert plan.cut == 1 and [message["role"] for message in plan.context.in_turn] == ["assistant", "tool"]
    assert plan.files == {"data.csv": b"id,amount\n1,10\n2,20\n3,30\n4,1.5\n", "notes.txt": b"rows: 4"}
    assert any("workspace as its step 1 found it" in warning for warning in plan.warnings)
    # Not in the sandbox: the steps taken as done name this run's own folder, which no other run has.
    assert any("own workspace" in warning for warning in plan.warnings)


def test_a_scenario_of_a_run_from_the_terminal(store, tmp_path):
    from tuningfork.cli import app
    from tuningfork.ops import scenarios

    chat = imports.read_chat(chat_with_earlier_turns())
    store.begin_run("as it happened", None)
    for index in chat.steps:
        store.record_step(request=chat.request(index), response=chat.response(index), harness="hermes")
    run = store.get_run("as it happened").id
    runner = CliRunner()
    looked = runner.invoke(app, ["scenario", run[:8], "from-a-run", "--reply-contains", "report.json", "--dry-run",
                                 "--home", str(store.home)], env={"COLUMNS": "200"})
    assert looked.exit_code == 0, looked.output
    assert "starts     at message 3" in looked.output and "nothing written" in looked.output
    assert "first      step 1's call, which the chat answered: skill_view(name='tally-report')" in looked.output
    assert not scenarios.folder(store.home, "from-a-run").exists()
    worded = runner.invoke(app, ["scenario", run[:8], "in-my-words", "--prompt", "Summarize inputs/data.csv.",
                                 "--home", str(store.home)], env={"COLUMNS": "200"})
    assert worded.exit_code == 0, worded.output
    assert "starts     at message 3 (yours: Summarize inputs/data.csv.)" in worded.output
    assert scenarios.read_scenario(store.home, "in-my-words").spec["evals"][0]["prompt"] == "Summarize inputs/data.csv."
    done = runner.invoke(app, ["scenario", run[:8], "from-a-run", "--reply-contains", "report.json",
                               "--home", str(store.home)], env={"COLUMNS": "200"})
    assert done.exit_code == 0, done.output
    assert "scenario from-a-run made" in done.output
    assert scenarios.read_scenario(store.home, "from-a-run").spec["evals"][0]["tuningfork"]["check"] == {"reply_contains": "report.json"}
    clash = runner.invoke(app, ["scenario", run[:8], "from-a-run", "--home", str(store.home)])
    assert clash.exit_code == 1 and "exists already" in clash.output
    both = runner.invoke(app, ["scenario", run[:8], "other", "--check", "true", "--no-check", "--home", str(store.home)])
    assert both.exit_code == 1 and "name one check" in both.output


def test_a_scenario_from_the_terminal(store, tmp_path):
    from tuningfork.cli import app
    from tuningfork.ops import scenarios

    path = tmp_path / "session.jsonl"
    path.write_text(chat_with_earlier_turns(), encoding="utf-8")
    runner = CliRunner()
    looked = runner.invoke(app, ["import", str(path), "--scenario", "from-cli", "--reply-contains", "report.json",
                                 "--dry-run", "--home", str(store.home)], env={"COLUMNS": "200"})
    assert looked.exit_code == 0, looked.output
    assert "scenario   from-cli: starts at message 3" in looked.output and "continuing 2 message(s)" in looked.output
    done = runner.invoke(app, ["import", str(path), "--scenario", "from-cli", "--reply-contains", "report.json",
                               "--home", str(store.home)], env={"COLUMNS": "200"})
    assert done.exit_code == 0, done.output
    assert "scenario from-cli made" in done.output
    spec = scenarios.read_scenario(store.home, "from-cli").spec
    assert spec["evals"][0]["tuningfork"]["check"] == {"reply_contains": "report.json"}
    again = runner.invoke(app, ["import", str(path), "--scenario", "from-cli", "--home", str(store.home)])
    assert again.exit_code == 1 and "exists already" in again.output
    assert len(store.runs(limit=10)) == 1  # the clash left no second run behind


async def test_the_web_ui_makes_a_scenario_and_edits_its_conversation(proxy_client, recorder):
    made = await proxy_client.post("/_tuningfork/api/imports", json={"content": chat_with_earlier_turns()})
    run = made.json()["run"]
    starts = (await proxy_client.get(f"/_tuningfork/api/imports/{run}/scenario")).json()
    assert starts["cut"] == 3 and [turn["message"] for turn in starts["turns"]] == [1, 3] and starts["context"] == 2
    assert [turn["offered"] for turn in starts["turns"]] == [False, True]
    assert starts["prompt"] == "Summarize data.csv into report.json." and starts["steps"][0]["part"] == "first"
    elsewhere = await proxy_client.get(f"/_tuningfork/api/runs/{run}/scenario?cut=1")
    assert elsewhere.status_code == 400 and elsewhere.json()["error"]["field"] == "cut"
    assert "came in after message 3" in elsewhere.json()["error"]["message"]
    assert (await proxy_client.get(f"/_tuningfork/api/runs/{run}/scenario?cut=x")).status_code == 400
    created = await proxy_client.post(f"/_tuningfork/api/imports/{run}/scenario",
                                      json={"name": "from-ui", "cut": 3, "prompt": "Summarize inputs/data.csv."})
    assert created.status_code == 201 and created.json()["scenario"]["context"][0]["role"] == "user"
    assert created.json()["scenario"]["spec"]["evals"][0]["prompt"] == "Summarize inputs/data.csv."
    assert any("no check" in warning for warning in created.json()["warnings"])
    for prompt in ("  ", 3):
        refused = await proxy_client.post(f"/_tuningfork/api/runs/{run}/scenario", json={"name": "no-words", "prompt": prompt})
        assert refused.status_code == 400 and refused.json()["error"]["field"] == "prompt"
    two = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    edited = await proxy_client.put("/_tuningfork/api/scenarios/from-ui/context", json={"messages": two})
    assert edited.status_code == 200 and edited.json() == {"messages": 2}
    refused = await proxy_client.put("/_tuningfork/api/scenarios/from-ui/context", json={"messages": two[:1]})
    assert refused.status_code == 400 and "ends on a user turn" in refused.json()["error"]["message"]

    # Any run's address: the same choices, and a body that says what it may hold.
    choices = (await proxy_client.get(f"/_tuningfork/api/runs/{run}/scenario")).json()
    assert choices["cut"] == 3 and choices["skill"] == "tally-report" and choices["imported"]
    odd = await proxy_client.post(f"/_tuningfork/api/runs/{run}/scenario", json={"name": "x", "task_check": "no"})
    assert odd.status_code == 400 and odd.json()["error"]["field"] == "task_check"
    other = await proxy_client.post(f"/_tuningfork/api/runs/{run}/scenario", json={"name": "by-run", "harness": "hermes"})
    assert other.status_code == 201 and other.json()["harness"] == "hermes"
