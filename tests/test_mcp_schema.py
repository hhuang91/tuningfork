"""The MCP tool list: every job kind decided on, every schema real JSON Schema, small enough
to send, and the same order every time."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

pytest.importorskip("mcp")

from tuningfork.mcp import schema  # noqa: E402
from tuningfork.mcp.server import ToolServer  # noqa: E402
from tuningfork.mcp.tools import input_schema, registry  # noqa: E402
from tuningfork.ops.catalog import CATALOG  # noqa: E402
from tuningfork.ops.common import OperationError  # noqa: E402


def test_every_kind_is_offered_or_excluded_with_a_reason():
    """A new job kind fails here until someone decides whether an agent may start it."""
    offered = set(schema.EVIDENCE) | set(schema.HARNESS)
    assert set(CATALOG) == offered | set(schema.EXCLUDED)
    assert not set(schema.EVIDENCE) & set(schema.HARNESS)
    assert not offered & set(schema.EXCLUDED)
    assert all(reason for reason in schema.EXCLUDED.values())


@pytest.mark.parametrize("toolset", ["evidence", "all"])
def test_every_schema_is_json_schema_and_closed(toolset):
    jsonschema = pytest.importorskip("jsonschema")
    for tool in registry(toolset).values():
        found = input_schema(tool, max_wait=120)
        jsonschema.Draft202012Validator.check_schema(found)
        assert found["type"] == "object" and found["additionalProperties"] is False, tool.name


def test_names_suit_every_client():
    for tool in registry("all").values():
        assert re.fullmatch(r"[a-z0-9_]{1,64}", tool.name), tool.name


def test_no_kind_has_its_own_dry_run_or_wait():
    for kind in CATALOG.values():
        names = {spec.name for spec in dataclasses.fields(kind.params)}
        assert not names & {"dry_run", "wait_seconds"}, kind.name


def test_the_tool_list_is_small_and_always_in_one_order():
    server = ToolServer(store=None, jobs=None)
    listed = [tool.model_dump(by_alias=True, exclude_none=True, mode="json") for tool in server.listed()]
    # Grew by where_it_holds' 1,800 characters: one question of every run, the way to the step when a
    # check prints only its verdict; and by 650 for reading a file or a message page by page, and asking
    # a screen's forks (findings §31.57: a 12,000-character file was cut where the check failed it); and
    # by 700 for what a probe's continuations wrote, read page by page (issue #29).
    assert len(json.dumps(listed)) <= 30_100, len(json.dumps(listed))
    names = [tool["name"] for tool in listed]
    assert names[0] == "list_tasks" and names[-1] == "cancel_job"
    assert names == [tool.name for tool in ToolServer(store=None, jobs=None).listed()]
    starts = [name for name in names if name.startswith("start_")]
    assert starts == ["start_resample", "start_tokens", "start_forking_tokens", "start_ab", "start_server_check"]


def test_the_default_runs_nothing_on_the_machine():
    evidence, everything = registry("evidence"), registry("all")
    assert not any(tool.machine for tool in evidence.values())
    assert "start_batch" not in evidence and "start_check_trial" not in evidence
    assert everything["start_batch"].machine and everything["start_batch"].annotations()["openWorldHint"]
    reads = [tool for tool in evidence.values() if tool.read is not None]
    assert all(tool.annotations()["readOnlyHint"] for tool in reads)
    with pytest.raises(OperationError, match="no toolset"):
        registry("everything")


class TestJobArguments:
    def test_a_step_may_be_a_number(self):
        call = schema.read_job_call(CATALOG["resample"], {"run": "a", "step": 3}, max_wait=120)
        assert call.params == {"run": "a", "step": "3"} and call.dry_run is False and call.wait_seconds == 30
        step = input_schema(registry()["start_resample"], max_wait=120)["properties"]["step"]
        assert step["type"] == ["integer", "string"]

    def test_a_fork_never_stops_for_a_person(self):
        call = schema.read_job_call(CATALOG["fork"], {"run": "a", "step": "1"}, max_wait=120)
        assert call.params["hold"] == "none"
        assert "hold" not in input_schema(registry("all")["start_fork"], max_wait=120)["properties"]
        with pytest.raises(OperationError) as refused:
            schema.read_job_call(CATALOG["fork"], {"run": "a", "step": "1", "hold": "step"}, max_wait=120)
        assert refused.value.field == "hold"

    def test_a_check_is_never_a_shell_command(self):
        kind = CATALOG["check-trial"]
        with pytest.raises(OperationError) as refused:
            schema.read_job_call(kind, {"check": "rm -rf ~"}, max_wait=120)
        assert refused.value.field == "check"
        call = schema.read_job_call(kind, {"check": {"reply_contains": "done"}}, max_wait=120)
        assert call.params["check"] == {"reply_contains": "done"}
        check = input_schema(registry("all")["start_check_trial"], max_wait=120)["properties"]["check"]
        assert check["type"] == "object"

    @pytest.mark.parametrize("wait", [500, -1, True, "10"])
    def test_a_wait_is_a_bounded_whole_number(self, wait):
        with pytest.raises(OperationError) as refused:
            schema.read_job_call(CATALOG["resample"], {"run": "a", "step": "1", "wait_seconds": wait}, max_wait=120)
        assert refused.value.field == "wait_seconds"

    def test_dry_run_is_true_or_false(self):
        with pytest.raises(OperationError) as refused:
            schema.read_job_call(CATALOG["resample"], {"run": "a", "step": "1", "dry_run": "yes"}, max_wait=120)
        assert refused.value.field == "dry_run"


class TestTheRefineToolset:
    def test_it_carries_the_loop_and_the_default_stays_as_it_was(self):
        refine, evidence = registry("refine"), registry("evidence")
        assert [name for name in refine if name.startswith("start_")] == [
            "start_refinement", "start_resample", "start_ab", "start_screen"]
        assert {"show_refinement", "propose_edit"} <= set(refine)
        assert not {"start_screen", "propose_edit", "show_refinement"} & set(evidence)
        assert refine["propose_edit"].annotations()["readOnlyHint"] is False

    def test_the_built_in_proposer_is_never_offered_to_an_agent(self):
        for toolset in ("evidence", "refine", "all"):
            assert "start_propose" not in registry(toolset)
        assert "person" in schema.EXCLUDED["propose"]

    def test_the_loop_with_screens_to_the_end_is_its_own_toolset(self):
        from tuningfork.mcp.server import instructions

        offered = registry("refine-to-end")
        assert [name for name in offered if name.startswith("start_")] == [
            "start_refinement", "start_resample", "start_ab", "start_screen", "start_screen_to_end"]
        assert offered["start_screen_to_end"].machine and "start_screen_to_end" not in registry("refine")
        assert "start_screen_to_end" in registry("all") and schema.DEFAULT_LIMITS["screen-to-end"] == 60
        assert "target_check" in instructions("refine-to-end") and "target_check" not in instructions("refine")

    # `all` grew by start_check_along's 1,500 characters, the check run along a task's runs, and by
    # where_it_holds' 1,800, one question of every run; the loops took it within theirs. Each grew again
    # by 350 to 1,250 for paging a file or a message, a screen's forks asked, and the quick look; and by
    # 700 for what a probe's continuations wrote (issue #29).
    @pytest.mark.parametrize("toolset, budget", [("refine", 31_100), ("refine-to-end", 34_400), ("all", 55_100)])
    def test_each_toolset_stays_small(self, toolset, budget):
        listed = [tool.model_dump(by_alias=True, exclude_none=True, mode="json")
                  for tool in ToolServer(store=None, jobs=None, toolset=toolset).listed()]
        assert len(json.dumps(listed)) <= budget, len(json.dumps(listed))

    def test_the_loops_instructions_come_with_its_tools(self):
        from tuningfork.mcp.server import instructions

        assert "start_refinement" in instructions("refine") and "start_refinement" not in instructions("evidence")
        assert "a person's decision" in instructions("all")
