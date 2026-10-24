"""Every data and experiment command's full output, pinned.

The other CLI tests look for a phrase in the output. These compare all of it, after
masking what legitimately differs between runs — ids, clock times, temporary paths and
sample timings — so a change to what a person reads cannot land without a test saying so.

Regenerate after an intended change, and read the diff before committing it:

    TF_UPDATE_GOLDEN=1 pytest tests/test_cli_golden.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.cli import app
from tuningfork.demo.mock import MockTurn
from tuningfork.store.db import Store
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot, record_run, text_message

GOLDEN = Path(__file__).parent / "golden"
UPSTREAM = "http://upstream.test"
WHEN = re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?\b")
SECONDS = re.compile(r"\b\d+\.\d+s\b")
MINUTES = re.compile(r"\b\d+\.\d+ min\b")
#: The patterns a step was masked with, named by their digest: another with each change of them.
DIGEST = re.compile(r'"redaction": "[0-9a-f]{16}"')


def write_keys(keys: dict) -> MockTurn:
    return MockTurn(
        tool_calls=[
            {"name": "write_file", "arguments": {"path": "report.json", "content": json.dumps(keys)}}
        ]
    )


def wording_sensitive(payload: dict) -> MockTurn:
    if "row_count and amount_total" in payload["messages"][0]["content"]:
        return write_keys({"row_count": 4, "amount_total": 61.5})
    return write_keys({"count": 4, "total": 61.5})


class Scenario:
    """One store, built the same way every time, and the commands that read it."""

    def __init__(self, tmp_path: Path, served) -> None:
        self.tmp_path = tmp_path
        self.served = served
        self.home = tmp_path / "store"
        self.runner = CliRunner()
        self.outputs: dict[str, str] = {}

    def cli(self, *args: str) -> str:
        result = self.runner.invoke(app, [*args, "--home", str(self.home)])
        assert result.exception is None or isinstance(result.exception, SystemExit), result.output
        return result.output

    def build(self) -> Scenario:
        store = Store(self.home)
        record_pilot(store)
        for index, (last, passed) in enumerate([("left", True), ("left", True), ("right", False)]):
            run_id = record_run(
                store, f"r{index}", [text_message("same"), text_message(last)], task_id="t/1"
            )
            store.annotate_run(
                run_id, harness="hermes", skill_hashes={"SKILL.md": "v1"},
                result={"passed": passed, "usable": True},
            )
        store.close()

        skill = self.tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (skill / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")

        read = MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": "data.csv"}}])
        self.served.script(read, read, read, MockTurn(content="I will ask first."))
        self.outputs["resample"] = self.cli(
            "resample", "pilot", "0", "-n", "4", "--base-url", UPSTREAM
        )
        self.served.responder = wording_sensitive
        self.outputs["ab"] = self.cli(
            "ab", "pilot", "write_file", "--variant", str(skill / "SKILL.v2.md"), "-n", "3",
            "--target-match", "row_count", "--base-url", UPSTREAM,
        )
        self.outputs["ab-exploratory"] = self.cli(
            "ab", "pilot", "1", "--variant", str(skill / "SKILL.v2.md"), "-n", "2",
            "--base-url", UPSTREAM,
        )
        return self

    def ids(self) -> list[tuple[str, str]]:
        """Every id the store holds, with the name it is masked as — full ids first."""
        store = Store(self.home)
        try:
            names = [(run.id, f"<run {run.label}>") for run in store.runs(limit=100)]
            experiments = sorted(store.experiments(limit=100), key=lambda e: e.created_at)
            names += [(e.id, f"<experiment {n}>") for n, e in enumerate(experiments, 1)]
            groups = dict.fromkeys(e.group_id for e in experiments if e.group_id)
            names += [(g, f"<group {n}>") for n, g in enumerate(groups, 1)]
        finally:
            store.close()
        return names

    def experiment_id(self, index: int) -> str:
        store = Store(self.home)
        try:
            experiments = sorted(store.experiments(limit=100), key=lambda e: e.created_at)
            return experiments[index].id
        finally:
            store.close()

    def mask(self, text: str) -> str:
        text = text.replace(str(self.tmp_path), "<tmp>")
        for full, name in self.ids():
            text = text.replace(full, name)
        for full, name in self.ids():
            text = re.sub(rf"\b{full[:8]}\b", name, text)
        text = WHEN.sub("<when>", text)
        text = MINUTES.sub("<minutes>", text)
        text = DIGEST.sub("\"redaction\": \"<digest>\"", text)
        return SECONDS.sub("<seconds>", text)


@pytest.fixture
def scenario(monkeypatch, tmp_path, mock_server) -> Scenario:
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    mock_server.props = PROPS
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.ASGITransport(app=mock_server.app), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return Scenario(tmp_path, mock_server).build()


def check(name: str, text: str) -> None:
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("TF_UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"no golden output for {name}: run with TF_UPDATE_GOLDEN=1"
    assert text == path.read_text(encoding="utf-8"), f"{name} output changed; see {path}"


COMMANDS = {
    "runs": ("runs",),
    "runs-quiet": ("runs", "-q"),
    "runs-called": ("runs", "--called", "write_file"),
    "tasks": ("tasks",),
    "show": ("show", "pilot"),
    "step": ("step", "pilot", "0"),
    "step-full": ("step", "pilot", "write_file", "--full"),
    "timeline": ("timeline", "pilot"),
    "timeline-full": ("timeline", "pilot", "--full"),
    "diff": ("diff", "r0", "r2"),
    "divergence": ("divergence", "--task", "t/1"),
    "divergence-lens": ("divergence", "--task", "t/1", "--normalizer", "tool_name"),
    "experiments": ("experiments",),
}


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_inspection_output(scenario, name):
    check(name, scenario.mask(scenario.cli(*COMMANDS[name])))


@pytest.mark.parametrize("name", ["resample", "ab", "ab-exploratory"])
def test_experiment_output(scenario, name):
    check(name, scenario.mask(scenario.outputs[name]))


def test_stored_experiments_reported_again(scenario):
    resample, ab = scenario.experiment_id(0), scenario.experiment_id(1)
    check("experiment-resample", scenario.mask(scenario.cli("experiment", resample)))
    check("experiment-resample-lens", scenario.mask(
        scenario.cli("experiment", resample, "--normalizer", "tool_name")
    ))
    check("experiment-ab", scenario.mask(scenario.cli("experiment", ab)))
