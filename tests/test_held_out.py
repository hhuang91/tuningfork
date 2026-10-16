"""Held-out tasks, kept from agents: whatever the recordings hold of a held-out task — runs
recorded before validation stopped recording them, or by a person's --include-held-out —
no MCP read shows, and no agent can start work on."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp")

from tuningfork.mcp.server import ToolServer  # noqa: E402
from tuningfork.mcp.tools import registry  # noqa: E402
from tuningfork.ops.catalog import CATALOG  # noqa: E402
from tuningfork.ops.common import OperationError  # noqa: E402
from tuningfork.mcp.schema import job_schema, read_job_call  # noqa: E402
from tuningfork.store.db import Store  # noqa: E402
from tests.helpers import SKILL, record_pilot, record_run, tool_call_message  # noqa: E402
from tests.test_mcp_tools import NoJobs  # noqa: E402

SECRETS = ("SECRET-PROMPT", "SECRET-ANSWER", "SECRET-CHECK", "SECRET-SCENARIO")


@pytest.fixture
def recordings(tmp_path):
    """A visible run of tally-report/1, a run of the held-out tally-report/2 with an experiment
    at it, and a scenario holding a held-out eval."""
    from tuningfork.ops.scenarios import save_scenario

    store = Store(tmp_path / "store")
    visible = record_pilot(store)
    secret = record_run(store, "secret", [tool_call_message("write_file", {"content": "SECRET-ANSWER"})],
                        prompt="SECRET-PROMPT", task_id="tally-report/2")
    store.create_launch(
        kind="batch", label="tally-report/2#x.0", harness="fake", model="m", task_id="tally-report/2",
        eval={"id": "2", "prompt": "SECRET-PROMPT", "tuningfork": {"held_out": True, "check": "echo SECRET-CHECK"}},
        spec_path=str(tmp_path / "spec.json"), settings={},
    )
    step = store.steps(secret)[0]
    store.create_experiment(kind="resample", arm="original", run_id=secret, step_idx=0, step_id=step.id,
                            request={"model": "m", "messages": [{"role": "user", "content": "SECRET-PROMPT"}]},
                            settings={"n": 1})
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(SKILL, encoding="utf-8")
    save_scenario(store.home, "demo", {
        "skill_name": "tally-report",
        "tuningfork": {"skill_dir": str(skill)},
        "evals": [
            {"id": "1", "prompt": "Summarize data.csv", "tuningfork": {"check": {"reply_contains": "done"}}},
            {"id": "2", "prompt": "SECRET-SCENARIO", "tuningfork": {"held_out": True, "check": {"reply_contains": "x"}}},
        ],
    }, etag=None)
    yield store, visible, secret
    store.close()


def shown(answer) -> str:
    return json.dumps(answer.data) + "\n".join(answer.lines)


class TestReads:
    async def test_no_read_shows_anything_of_a_held_out_task(self, recordings):
        store, visible, secret = recordings
        server = ToolServer(store=store, jobs=NoJobs(), toolset="all")
        tools = registry("all")
        calls = [
            ("list_tasks", {}), ("list_runs", {}), ("list_experiments", {}), ("list_scenarios", {}),
            ("list_scenarios", {"name": "demo"}), ("show_run", {"run": visible[:8]}),
        ]
        for name, arguments in calls:
            answer = await server.call(tools[name], arguments)
            assert not any(secret_text in shown(answer) for secret_text in SECRETS), name
        listed = await server.call(tools["list_runs"], {})
        assert [run["id"] for run in listed.data["runs"]] == [visible[:8]]
        scenario = await server.call(tools["list_scenarios"], {"name": "demo"})
        assert scenario.data["spec"]["held_out_evals"] == 1
        detail = await server.call(tools["show_run"], {"run": visible[:8]})
        assert "spec" not in (detail.data.get("launch") or {})

    @pytest.mark.parametrize("name, arguments", [
        ("show_run", {"run": "secret"}),
        ("show_step", {"run": "secret", "step": 0}),
        ("step_behaviours", {"run": "secret", "step": 0}),
        ("show_skill", {"run": "secret"}),
        ("step_tokens", {"run": "secret", "step": 0}),
        ("list_runs", {"task": "tally-report/2"}),
        ("find_divergence", {"task": "tally-report/2"}),
        ("tally_failures", {"task": "tally-report/2"}),
        ("where_it_holds", {"task": "tally-report/2", "target_match": "SECRET"}),
        ("where_it_holds", {"task": "tally-report/2", "path": "report.json"}),
        ("list_experiments", {"run": "secret"}),
    ])
    async def test_opening_one_by_name_is_refused(self, recordings, name, arguments):
        store, _, _ = recordings
        server = ToolServer(store=store, jobs=NoJobs(), toolset="all")
        with pytest.raises(OperationError) as refused:
            await server.call(registry("all")[name], arguments)
        assert refused.value.code == "held_out"

    async def test_its_experiment_is_refused_too(self, recordings):
        store, _, secret = recordings
        (experiment,) = store.experiments(run_id=secret)
        server = ToolServer(store=store, jobs=NoJobs(), toolset="all")
        with pytest.raises(OperationError) as refused:
            await server.call(registry("all")["show_experiment"], {"experiment": experiment.id[:8]})
        assert refused.value.code == "held_out"
        # Its checks' findings too, before anything is said of how it was drawn.
        with pytest.raises(OperationError) as tallied:
            await server.call(registry("all")["tally_failures"], {"screen": experiment.id[:8]})
        assert tallied.value.code == "held_out"


class TestWork:
    def test_an_agent_is_never_offered_a_held_out_eval(self):
        schema = job_schema(CATALOG["batch"], max_wait=30)
        assert "include_held_out" not in schema["properties"]
        with pytest.raises(OperationError) as refused:
            read_job_call(CATALOG["batch"], {"spec": "s.json", "include_held_out": True}, max_wait=30)
        assert refused.value.field == "include_held_out"

    async def test_work_at_a_held_out_run_is_refused_before_it_is_planned(self, recordings):
        store, _, _ = recordings
        server = ToolServer(store=store, jobs=NoJobs(), toolset="all")
        with pytest.raises(OperationError) as refused:
            await server.call(registry("all")["start_resample"], {"run": "secret", "step": 0, "n": 2})
        assert refused.value.code == "held_out"

    async def test_the_check_along_a_held_out_task_is_refused_before_it_is_planned(self, recordings):
        store, _, _ = recordings
        server = ToolServer(store=store, jobs=NoJobs(), toolset="all")
        for arguments in ({"task": "tally-report/2"}, {"run": "secret"}):
            with pytest.raises(OperationError) as refused:
                await server.call(registry("all")["start_check_along"], arguments)
            assert refused.value.code == "held_out", arguments
