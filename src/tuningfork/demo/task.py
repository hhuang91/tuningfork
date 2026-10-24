"""The demo's task: the tally-report skill and its edit, a model that follows the wording it is
given, and a harness to run it — what `tuningfork demo` serves, and what the tests run end to
end with no model."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from tuningfork.demo.mock import MockTurn
from tuningfork.harness.base import LaunchSpec


def tool_call_message(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def as_response(message: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


SKILL = "---\nname: tally-report\n---\n\n# Tally report\n\nWrite the totals to report.json.\n"
VARIANT = (
    "---\nname: tally-report\n---\n\n# Tally report\n\n"
    "Write the totals to report.json, using the keys row_count and amount_total.\n"
)


def system_prompt(skill: str = SKILL) -> str:
    """Mirrors how Hermes 0.18.2 inlines a preloaded skill into its system prompt."""
    return (
        "You are an agent.\n\n"
        '[IMPORTANT: The user launched this CLI session with the "tally-report" skill '
        f"preloaded.]\n\n{skill}\n[Skill directory: /tmp/run/home/skills/tally-report]\n"
    )


# --- a real harness process, and the model it talks to ----------------------------------------

FAKE_AGENT = Path(__file__).parent / "agent.py"
CHECK = (
    "python3 -c \"import json,sys; d=json.load(open('report.json'));"
    " sys.exit(0 if d.get('row_count')==4 else 1)\""
)


class FakeAgent:
    """Launches `agent.py` the way an adapter launches a real harness."""

    name = "fake"
    subagent_tools = frozenset()
    volatile = ()

    def prepare_home(self, home, *, base_url: str, model: str, max_turns: int = 30) -> Path:
        home = Path(home)
        (home / "skills").mkdir(parents=True, exist_ok=True)
        (home / "agent.json").write_text(json.dumps({"base_url": base_url, "model": model}))
        return home

    def launch(self, prompt, *, home, workspace, skills=(), extra_args=(), max_turns=30):
        config = json.loads((Path(home) / "agent.json").read_text())
        env = {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_BASE_URL": config["base_url"],
            "FAKE_MODEL": config["model"],
            "FAKE_HOME": str(home),
            "FAKE_MAX_TURNS": str(max_turns),
        }
        if os.environ.get("FAKE_NOISE"):
            env["FAKE_NOISE"] = os.environ["FAKE_NOISE"]
        if os.environ.get("FAKE_DELAY"):
            env["FAKE_DELAY"] = os.environ["FAKE_DELAY"]
        return LaunchSpec(argv=(sys.executable, str(FAKE_AGENT), prompt), env=env, cwd=Path(workspace))

    def continue_launch(self, prompt, *, home, workspace, max_turns=30):
        """One more turn: the fake agent keeps no session, so it is a fresh conversation in
        the same home and workspace — enough for what the runner does with it."""
        return self.launch(prompt, home=home, workspace=workspace, max_turns=max_turns)

    def skill_roots(self, home) -> tuple[Path, ...]:
        return (Path(home) / "skills",)

    def skill_spans(self, request) -> list:
        return []


def tally_model(payload: dict) -> MockTurn:
    """The model: read the data, write the report — its keys as the wording in its context
    asks, at the absolute path of the workspace it is told it is in — then say so."""
    messages = payload["messages"]
    system = messages[0]["content"]
    results = [m for m in messages if m["role"] == "tool"]
    workdir = re.search(r"Working directory: (\S+)", system).group(1)
    if not results:
        return MockTurn(tool_calls=[{"name": "read_file", "arguments": {"path": "data.csv"}}])
    if len(results) == 1:
        keys = (
            {"row_count": 4, "amount_total": 61.5}
            if "row_count" in system
            else {"count": 4, "total": 61.5}
        )
        return MockTurn(tool_calls=[{"name": "write_file", "arguments": {
            "path": f"{workdir}/report.json", "content": json.dumps(keys)}}])
    return MockTurn(content="The report is written.")
