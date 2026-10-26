"""The built-in proposer (`ops.proposer`): asked through the proxy by default and never
recorded; a remote host only with consent; its key sent, and stored nowhere but its file."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tuningfork.demo.mock import MockTurn
from tuningfork.ops import proposer, refine
from tuningfork.ops.batch import execute_batch, prepare_batch
from tuningfork.ops.common import OperationError
from tests.helpers import tally_model, task_stack

RULES = [{"if": "calls", "value": "write_file"}, {"if": "contains", "value": "row_count"}]
KEY = "sk-test-0123456789abcdefXYZ"
GOOD = {"edits": [{"old": "Write the totals to report.json.",
                   "new": "Write the totals to report.json, using the keys row_count and amount_total."}],
        "rationale": "the check reads row_count; every answer wrote count"}
BAD = {"edits": [{"old": "a line the skill does not have", "new": "x"}], "rationale": "r"}


def answering(content: str, finish_reason: str | None = None):
    """The mock: proposals for the proposer, the tally task for the harness."""
    def respond(payload: dict) -> MockTurn:
        system = payload.get("system") or (payload.get("messages") or [{}])[0].get("content") or ""
        if isinstance(system, str) and system.startswith("You improve an agent skill"):
            return MockTurn(content=content, finish_reason=finish_reason)
        return tally_model(payload)
    return respond


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        plan = prepare_batch(built.store, built.spec.path, base_url=f"{built.url}/v1", workdir=built.tmp / "batches",
                             model="m", harness="fake", n=1)
        execute_batch(built.store, plan)
        (run,) = built.store.runs(limit=5, task_id="tally-report/1")
        built.refinement = refine.create(built.store, run.id, "write_file", target_rules=RULES)
        yield built


@pytest.fixture
def direct(monkeypatch, stack):
    """Calls made straight to an endpoint reach the mock, whatever its host."""
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=stack.mock.app), **kw)
    )
    return stack


async def test_by_default_it_asks_the_proxys_own_server_and_is_never_recorded(stack):
    stack.mock.responder = answering(json.dumps({"proposals": [GOOD]}))
    runs_before = len(stack.store.runs(limit=100))
    plan = proposer.prepare_propose(stack.store, stack.refinement.id, m=2, proxy_url=stack.url)
    assert plan.via_proxy and not plan.remote and plan.to_json()["through_the_proxy"]
    result = await proposer.execute_propose(stack.store, plan)
    (made,) = result.accepted
    assert "row_count and amount_total" in made.text()
    (entry,) = made.data["proposers"]
    assert entry["via"] == "builtin" and entry["packet"] == plan.packet.sha and entry["prompt"] == plan.prompt_sha
    assert len(stack.store.runs(limit=100)) == runs_before  # the proposer's call is no run
    record = stack.refinement.folder / f"{result.record}-prompt.json"
    assert json.loads(record.read_text())["packet"] == plan.packet.sha


def test_a_remote_host_is_used_only_once_allowed(stack):
    proposer.set_config(stack.store.home, url="https://api.example.test", model="big-model")
    with pytest.raises(OperationError) as refused:
        proposer.prepare_propose(stack.store, stack.refinement.id)
    assert refused.value.code == "remote_not_allowed" and "--allow-remote" in refused.value.hint
    proposer.set_config(stack.store.home, allow_remote=True)
    plan = proposer.prepare_propose(stack.store, stack.refinement.id)
    data = plan.to_json()
    assert data["remote"] and data["sends_to"] == "api.example.test:443" and "not this machine" in data["sends"]
    assert any("no API key" in warning for warning in data["warnings"])


async def test_its_key_is_sent_and_stored_nowhere_else(direct):
    stack = direct
    stack.mock.responder = answering(json.dumps({"proposals": [GOOD]}))
    proposer.set_config(stack.store.home, url="https://api.example.test", model="big-model", allow_remote=True)
    assert proposer.set_key(stack.store.home, KEY) == {"set": True, "from": "file"}
    await proposer.execute_propose(stack.store, proposer.prepare_propose(stack.store, stack.refinement.id))
    assert stack.mock.seen[-1][2]["authorization"] == f"Bearer {KEY}"
    holding = [path.name for path in stack.store.home.rglob("*") if path.is_file() and KEY.encode() in path.read_bytes()]
    assert holding == [".proposer-key"]
    assert KEY not in json.dumps(proposer.key_status(stack.store.home))


async def test_an_anthropic_endpoint_gets_its_own_shape(direct):
    stack = direct
    stack.mock.responder = answering(json.dumps({"proposals": [GOOD]}))
    proposer.set_config(stack.store.home, url="https://api.anthropic.test", wire="anthropic-messages",
                        model="claude-x", allow_remote=True)
    proposer.set_key(stack.store.home, KEY)
    result = await proposer.execute_propose(stack.store, proposer.prepare_propose(stack.store, stack.refinement.id))
    sent = stack.mock.requests[-1]
    assert sent["system"].startswith("You improve an agent skill") and sent["messages"][0]["role"] == "user"
    assert sent["model"] == "claude-x" and sent["max_tokens"] == 8192
    headers = stack.mock.seen[-1][2]
    assert headers["x-api-key"] == KEY and headers["anthropic-version"] == "2023-06-01"
    assert stack.mock.seen[-1][1] == "/v1/messages" and len(result.accepted) == 1


async def test_an_answer_cut_off_is_refused_but_kept(direct):
    stack = direct
    stack.mock.responder = answering('{"proposals": [', finish_reason="length")
    proposer.set_config(stack.store.home, url="http://127.0.0.1:9", model="m")
    with pytest.raises(OperationError) as refused:
        await proposer.execute_propose(stack.store, proposer.prepare_propose(stack.store, stack.refinement.id))
    assert refused.value.code == "truncated" and "max_tokens" in refused.value.hint
    assert list((stack.refinement.folder / "proposer").glob("*-response.json"))


async def test_fenced_json_is_read_and_a_bad_proposal_is_reported_not_kept(direct):
    stack = direct
    stack.mock.responder = answering("Here you go:\n```json\n" + json.dumps({"proposals": [GOOD, BAD]}) + "\n```")
    proposer.set_config(stack.store.home, url="http://127.0.0.1:9", model="m")
    result = await proposer.execute_propose(stack.store, proposer.prepare_propose(stack.store, stack.refinement.id))
    assert len(result.accepted) == 1
    (rejected,) = result.rejected
    assert rejected["index"] == 2 and "not in the skill" in rejected["reason"]
    assert len(refine.proposals(stack.store, stack.refinement.id)) == 1


def test_what_is_not_a_list_of_proposals():
    with pytest.raises(OperationError, match="no JSON object"):
        proposer.parse_proposals("I would change line 3.")
    with pytest.raises(OperationError, match='no "proposals" list'):
        proposer.parse_proposals('{"edits": []}')
    assert proposer.parse_proposals('noise {"proposals": []} noise') == []


def test_an_answer_that_corrects_its_own_json_is_read_from_its_last_object():
    """Found with Gemma 4: the answer held a draft whose rationale had slipped outside its
    proposal, then "Wait, I see I made a mistake… Let me correct that", then the JSON again.
    Read whole it did not parse ("Extra data"); the corrected object is the answer."""
    broken = '{"proposals": [{"edits": []}], "rationale": "outside"}'
    final = {"proposals": [GOOD]}
    text = f"```json\n{broken}\n```\n\nWait, I see I made a mistake. Let me correct that.\n\n```json\n{json.dumps(final)}\n```"
    assert proposer.parse_proposals(text) == [GOOD]
    with pytest.raises(OperationError, match="did not parse"):
        proposer.parse_proposals('{"proposals": [{"edits": [{"old": "a", "new": "b"}]]}')  # a stray ]


def test_the_last_whole_list_of_proposals_is_found_never_a_draft_before_it():
    draft, final = {"proposals": [BAD]}, {"proposals": [GOOD]}
    text = f"thinking {json.dumps(draft)} wait, a typo — {json.dumps(final)[:40]}<|turn>json\n{json.dumps(final)}"
    assert json.loads(proposer.last_proposals_object(text)) == final
    assert proposer.last_proposals_object("no object here {not json") is None


async def test_an_answer_filed_as_reasoning_is_read_from_its_end(direct):
    """Found with Gemma 4 on llama.cpp: the JSON came after a stray turn marker inside the
    thinking, the server filed all of it as reasoning, and the answer was empty."""
    stack = direct
    thought = f"Drafting {json.dumps({'proposals': [BAD]})} — no.<|turn>json\n{json.dumps({'proposals': [GOOD]})}"

    def respond(payload: dict) -> MockTurn:
        system = (payload.get("messages") or [{}])[0].get("content") or ""
        if isinstance(system, str) and system.startswith("You improve an agent skill"):
            return MockTurn(content="", reasoning=thought)
        return tally_model(payload)

    stack.mock.responder = respond
    proposer.set_config(stack.store.home, url="http://127.0.0.1:9", model="m")
    result = await proposer.execute_propose(stack.store, proposer.prepare_propose(stack.store, stack.refinement.id))
    assert result.read_from == "reasoning" and len(result.accepted) == 1 and result.rejected == []
    (by,) = result.accepted[0].data["proposers"]
    assert by["via"] == "builtin" and by["read_from"] == "reasoning"  # where its text came from, kept


def test_the_key_from_the_environment_wins_and_is_never_shown(stack, monkeypatch):
    proposer.set_key(stack.store.home, KEY)
    monkeypatch.setenv("TUNINGFORK_PROPOSER_KEY", "sk-env-0123456789abcdef")
    assert proposer.key_status(stack.store.home) == {"set": True, "from": "environment"}
    monkeypatch.delenv("TUNINGFORK_PROPOSER_KEY")
    assert proposer.set_key(stack.store.home, None) == {"set": False, "from": None}
    with pytest.raises(OperationError):
        proposer.set_key(stack.store.home, "two words")


async def test_what_it_is_sent_and_keeps_is_masked_with_the_store_s_own_patterns(stack):
    """The run's skill is kept as it was given: a proposer's model may be hosted, and is sent it masked,
    as the packet shows it. What is kept of the exchange is masked with the store's own patterns, as
    everything it keeps is (found in review: the answer was kept masked by the defaults alone)."""
    from tuningfork.config import DEFAULT_REDACTIONS
    from tuningfork.store.db import Store
    from tests.helpers import KEY as A_KEY, SKILL

    (stack.tmp / "skills-src" / "tally-report" / "SKILL.md").write_text(SKILL + f"\nLog in with {A_KEY}.\n", encoding="utf-8")
    # In a thread of its own, as the fixture's batch: a batch's plan reads the server on a loop of its own.
    await asyncio.to_thread(lambda: execute_batch(stack.store, prepare_batch(
        stack.store, stack.spec.path, base_url=f"{stack.url}/v1", workdir=stack.tmp / "keyed", model="m",
        harness="fake", n=1)))
    run = stack.store.runs(limit=1, task_id="tally-report/1")[0]
    own = Store(stack.store.home, patterns=(*DEFAULT_REDACTIONS, r"(?P<secret>zebra-\d+)"))
    try:
        made = refine.create(own, run.id, "write_file", target_rules=RULES)
        stack.mock.responder = answering(json.dumps({"proposals": [{**GOOD, "rationale": "zebra-421 never helped"}]}))
        result = await proposer.execute_propose(own, proposer.prepare_propose(own, made.id, m=1, proxy_url=stack.url))
        (proposal,) = result.accepted
        assert "zebra-421" not in proposal.data["proposers"][0]["rationale"]
        assert A_KEY not in json.dumps(stack.mock.requests[-1])
        kept = "".join(path.read_text(encoding="utf-8") for path in made.folder.rglob("*.json"))
        assert "zebra-421" not in kept and A_KEY not in kept
        assert A_KEY in (made.folder / "baseline.md").read_text(encoding="utf-8")  # the skill as it was given
    finally:
        own.close()
