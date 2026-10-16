"""Jobs that launch harnesses: a batch, forks run to the end, and a fork that stops at its
step for a person — started over HTTP, with `FakeAgent` as the harness (a real process)
and the mock server as the model (see `tests/helpers.task_stack`)."""

from __future__ import annotations

import json
import time

import pytest

from tuningfork.control import ControlClient
from tests.helpers import VARIANT, task_stack


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        built.spec.path.write_text(json.dumps(built.spec.to_dict()), encoding="utf-8")
        yield built


def wait(client: ControlClient, job: dict, timeout: float = 90) -> dict:
    return client.wait_for_job(job["id"], timeout=timeout, poll=0.1)


def test_a_batch_started_over_http(stack):
    with ControlClient(stack.url) as client:
        done = wait(client, client.submit("batch", {
            "spec": str(stack.spec.path), "n": 2, "harness": "fake", "model": "m",
        }))
    assert done["status"] == "done", done["error"]
    assert done["result"]["rate"]["total"] == 2 and done["result"]["rate"]["successes"] == 0
    assert len(done["links"]["launches"]) == 2 and len(done["links"]["runs"]) == 2
    for launch_id in done["links"]["launches"]:
        launch = stack.store.get_launch(launch_id)
        assert launch.job_id == done["id"] and launch.status == "ended"


def test_a_batch_that_cannot_start_is_refused_before_it_is_queued(stack):
    from tuningfork.control import ControlError

    with ControlClient(stack.url) as client, pytest.raises(ControlError, match="unknown harness"):
        client.submit("batch", {"spec": str(stack.spec.path), "harness": "nope"})
    assert stack.store.jobs() == []


def test_resampling_to_the_end_from_the_browser(stack):
    source = stack.record()
    with ControlClient(stack.url) as client:
        done = wait(client, client.submit("resample-to-end", {"run": source.run_id, "step": "1", "n": 2}))
    assert done["status"] == "done", done["error"]
    assert done["result"]["counted"] == 2 and done["result"]["passed"] == 0
    (experiment_id,) = done["links"]["experiments"]
    experiment = stack.store.get_experiment(experiment_id)
    assert experiment.settings["reach"] == "end" and experiment.job_id == done["id"]
    # Forks never hold at a breakpoint when run to the end.
    assert all(not stack.store.get_launch(l).breakpoints for l in done["links"]["launches"])


def test_an_ab_to_the_end_with_a_wording_edited_in_the_browser(stack):
    source = stack.record()
    with ControlClient(stack.url) as client:
        done = wait(client, client.submit("ab-to-end", {
            "run": source.run_id, "step": "1", "n": 2,
            "variant_text": VARIANT, "variant_name": "keys-named",
        }))
    assert done["status"] == "done", done["error"]
    result = done["result"]
    assert result["original"]["passed"] == 0 and result["variant"]["passed"] == 2
    saved = stack.store.home / "variants" / "tally-report" / "keys-named.md"
    assert saved.read_text(encoding="utf-8") == VARIANT
    assert len(done["links"]["experiments"]) == 2


def test_a_fork_stops_at_its_step_for_a_person(stack):
    source = stack.record()
    with ControlClient(stack.url) as client:
        job = client.submit("fork", {"run": source.run_id, "step": "1"})
        deadline = time.monotonic() + 30
        while not (holds := [h for h in client.holds() if not h["action"]]):
            assert time.monotonic() < deadline, client.job(job["id"])
            time.sleep(0.1)
        (hold,) = holds
        launch_id = client.job(job["id"])["links"]["launches"][0]
        assert hold["step"] == 1 and hold["phase"] == "request"
        assert hold["launch_id"] == launch_id
        # Armed for that launch alone, once; the fork's replayed step was never held.
        (armed,) = client.breakpoints()
        assert armed["launch"] == launch_id and armed["once"]
        client.decide(hold["id"], "continue")
        done = wait(client, job)
    assert done["status"] == "done", done["error"]
    (fork,) = done["result"]
    assert fork["run_id"] and fork["passed"] is False


def test_cancelling_a_fork_that_waits_at_its_step_lets_it_go(stack):
    source = stack.record()
    with ControlClient(stack.url) as client:
        job = client.submit("fork", {"run": source.run_id, "step": "1"})
        deadline = time.monotonic() + 30
        while not [h for h in client.holds() if not h["action"]]:
            assert time.monotonic() < deadline
            time.sleep(0.1)
        client.cancel_job(job["id"])
        stopped = wait(client, job, timeout=30)
        recent = client.holds(recent=True)
    assert stopped["status"] == "cancelled"
    assert all(h["action"] or h["abandoned"] for h in recent)
    launch = stack.store.get_launch(stopped["links"]["launches"][0])
    assert launch.status == "cancelled"


def test_a_screen_to_the_end_as_a_job(stack):
    """The refine loop's screen whose samples are forks: a job over HTTP, like the forks it runs."""
    from tuningfork.ops import refine

    source = stack.record()
    made = refine.create(stack.store, source.run_id, "write_file", target_check=True)
    proposal = refine.add_proposal(stack.store, made.id, edits=[{
        "old": "Write the totals to report.json.",
        "new": "Write the totals to report.json, using the keys row_count and amount_total.",
    }], rationale="the check reads row_count", proposer={"via": "person"})
    with ControlClient(stack.url) as client:
        done = wait(client, client.submit("screen-to-end", {
            "refinement": made.id[:8], "proposals": [proposal.id[:8]], "n": 2,
        }))
    assert done["status"] == "done", done["error"]
    screen = done["result"]["screen"]
    assert screen["reach"] == "end" and screen["baseline"]["target"]["successes"] == 0
    assert screen["arms"][0]["target"]["successes"] == 2 and screen["complete"]
    assert len(done["links"]["experiments"]) == 2 and len(done["links"]["launches"]) == 4
    assert all(stack.store.get_experiment(found).job_id == done["id"] for found in done["links"]["experiments"])
