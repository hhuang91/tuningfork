"""The trash: set aside, out of every list, back as it went — and deleted for good from there.

Agreed with the user (2026-10-01): a trash, from which deletion is permanent; whatever is tied
to a run stays in place until the run is deleted for good, and then goes with it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tuningfork.ops.inspect import task_summaries
from tuningfork.launches import LaunchThreads, request_path
from tuningfork.ops import trash
from tuningfork.ops.common import OperationError
from tuningfork.store.blobs import BlobStore
from tests.helpers import as_response, record_run, text_message, tool_call_message
from tests.test_approve import stack  # noqa: F401 — the fixture: a task, a proxy, a skill folder


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def chat(*messages: dict) -> dict:
    return {"model": "m", "messages": list(messages)}


def launched(store, *, kind="batch", label="t/1#b.0", source_run=None, harness="hermes"):
    """A launch with a main run of two steps and a side thread, ended."""
    launch = store.create_launch(kind=kind, label=label, task_id="t/1", harness=harness, source_run=source_run)
    threads = LaunchThreads(launch.id, label=launch.label, task_id=launch.task_id,
                            main_role="fork" if kind == "fork" else "main")
    answer = {"role": "assistant", "content": f"done {label}"}
    for request, response in (
        (chat(user(f"go {label}")), {"role": "assistant", "content": "", "tool_calls": []}),
        (chat(user(f"go {label}"), answer, user("more")), answer),
        (chat(user(f"a side call of {label}")), {"role": "assistant", "content": "side"}),
    ):
        store.record_step(request=request, response=as_response(response), assignment=threads.assign(request_path(request)))
    store.finalize_threads(launch.id)
    store.end_launch(launch.id)
    main = next(run for run in store.launch_runs(launch.id) if not run.is_thread)
    return launch, main


class TestTheTrash:
    def test_a_run_in_the_trash_leaves_every_list_and_comes_back_as_it_went(self, store):
        kept = record_run(store, "kept", [text_message("a")], task_id="t/1", prompt="first")
        gone = record_run(store, "gone", [text_message("b")], task_id="t/1", prompt="second")

        moved = trash.trash(store, "run", gone)

        assert moved["moved"] == [gone]
        assert [run.id for run in store.runs(task_id="t/1")] == [kept]
        (summary,) = task_summaries(store.runs(limit=100))
        assert summary["runs"] == 1
        assert store.get_run(gone).trashed_at is not None  # its page still opens
        (listed,) = trash.listing(store).runs
        assert listed["id"] == gone and listed["goes"]["runs"] == 1

        trash.restore(store, "run", gone)
        assert {run.id for run in store.runs(task_id="t/1")} == {kept, gone}
        assert trash.listing(store).empty

    def test_a_launched_run_goes_with_its_threads_never_one_alone(self, store):
        launch, main = launched(store)
        (side,) = [run for run in store.launch_runs(launch.id) if run.is_thread]

        with pytest.raises(OperationError, match="goes to the trash with that run"):
            trash.trash(store, "run", side.id)
        moved = trash.trash(store, "run", main.id)

        assert set(moved["moved"]) == {main.id, side.id}
        assert [run["id"] for run in trash.listing(store).runs] == [main.id]
        trash.restore(store, "run", main.id)
        assert all(run.trashed_at is None for run in store.launch_runs(launch.id))

    def test_a_run_still_running_stays(self, store):
        launch = store.create_launch(kind="batch", label="t/1#b.0", task_id="t/1", harness="hermes")
        threads = LaunchThreads(launch.id, label=launch.label, task_id="t/1")
        request = chat(user("go"))
        store.record_step(request=request, response=as_response(text_message("x")), assignment=threads.assign(request_path(request)))
        (run,) = store.launch_runs(launch.id)

        with pytest.raises(OperationError, match="still running"):
            trash.trash(store, "run", run.id)

    def test_no_new_work_starts_on_a_run_in_the_trash(self, store):
        from tuningfork.ops.experiments import prepare_resample

        run = record_run(store, "r", [text_message("a")], task_id="t/1")
        trash.trash(store, "run", run)

        with pytest.raises(OperationError) as refused:
            prepare_resample(store, run, "0", n=2)
        assert refused.value.code == "trashed" and "restore it first" in refused.value.hint

    def test_new_traffic_never_joins_a_run_in_the_trash(self, store):
        first = record_run(store, "r", [text_message("a")], prompt="go")
        trash.trash(store, "run", first)

        later = store.record_step(
            request=chat(user("go"), text_message("a"), user("continue")), response=as_response(text_message("b"))
        )

        assert later.run_id != first

    def test_deletion_is_from_the_trash(self, store):
        run = record_run(store, "r", [text_message("a")])
        with pytest.raises(OperationError, match="not in the trash"):
            trash.purge(store, "run", run)


class TestDeletedForGood:
    def test_what_is_tied_to_a_run_stays_until_it_goes_for_good_and_then_goes_with_it(self, store):
        launch, source = launched(store, label="t/1#b.0")
        fork_launch, fork = launched(store, kind="fork", label="t/1#b.0@1.0", source_run=source.id)
        alike = record_run(store, "alike", [text_message("unrelated")], prompt="go t/1#b.0")  # shares a beginning
        steps = store.steps(source.id)
        experiment = store.create_experiment(kind="resample", arm="original", run_id=source.id, step_idx=1,
                                             step_id=steps[1].id, request={"messages": []}, settings={"n": 2}, status="done")
        sample = store.add_sample(experiment.id, 0, seed=1, response=as_response(text_message("sampled once")),
                                  outcome_kind="text", outcome_key="text:x", outcome_label="x")
        store.add_token_stream({"v": 1}, source="scored", step_id=steps[1].id)
        store.add_token_stream({"v": 1, "sample": True}, source="sampled", sample_id=sample.id)
        store.link_holds(["hold1"], steps[1], edited=False)
        unique_blob = steps[1].response_blob
        assert unique_blob in store.blobs

        trash.trash(store, "run", source.id)
        # in the trash, nothing tied to it has moved
        assert store.experiments(run_id=source.id)[0].id == experiment.id
        assert fork.id in {run.id for run in store.runs(limit=100, forks=True)}
        goes = trash.preview(store, "run", source.id)
        assert goes == {"runs": 4, "steps": 6, "experiments": 1, "samples": 1, "token_streams": 2, "launches": 2,
                        "cited_by": []}

        removed = trash.purge(store, "run", source.id).to_json()

        assert removed["removed"]["runs"] == 4  # its main run and thread, and the fork's
        assert removed["message_nodes"] > 0  # what only they reached
        for gone in (source.id, fork.id):
            assert store.get_run(gone) is None
        assert store.get_experiment(experiment.id) is None
        assert store.token_streams(step_id=steps[1].id) == [] and store.token_streams(sample_ids=[sample.id]) == []
        assert store.get_launch(launch.id) is None and store.get_launch(fork_launch.id) is None
        # what nothing else refers to is gone from disk; what is shared stays
        os.utime(store.blobs._path(unique_blob), (0, 0))  # old enough for a sweep
        trash._sweep(store, trash.Removal())
        assert unique_blob not in store.blobs
        survivor = store.get_run(alike)
        assert survivor.forked_from_run is None
        assert store.rebuild_request(store.steps(alike)[0])["messages"][0]["content"] == "go t/1#b.0"

    def test_a_fork_still_running_holds_back_its_source_from_going_for_good(self, store):
        # A fork started before its source went to the trash runs on. Found in review: deleting
        # the source for good took the fork's run and launch with it, its folders included,
        # while its harness was still at work there.
        launch, source = launched(store, label="t/1#b.0")
        fork_launch = store.create_launch(kind="fork", label="t/1#b.0@1.0", task_id="t/1", harness="hermes", source_run=source.id)
        threads = LaunchThreads(fork_launch.id, label=fork_launch.label, task_id="t/1", main_role="fork")
        request = chat(user("go t/1#b.0"))
        store.record_step(request=request, response=as_response(text_message("x")), assignment=threads.assign(request_path(request)))
        trash.trash(store, "run", source.id)

        for deleting in (lambda: trash.purge(store, "run", source.id), lambda: trash.empty(store)):
            with pytest.raises(OperationError) as refused:
                deleting()
            assert refused.value.code == "busy" and "still running" in refused.value.message
        assert store.get_run(source.id) is not None and store.get_launch(fork_launch.id) is not None

        store.end_launch(fork_launch.id)
        assert trash.purge(store, "run", source.id).to_json()["removed"]["launches"] == 2

    def test_work_whose_process_was_killed_holds_back_nothing(self, store):
        # A batch killed hard never says how its launches ended. Found in review: each stayed
        # running for good — its run could not go to the trash, nor the run a fork of it
        # replayed be deleted for good — and nothing could mark it ended. Its process is gone.
        import socket

        from tuningfork.ops.timeline import launch_json

        launch, source = launched(store, label="t/1#b.0")
        fork_launch = store.create_launch(kind="fork", label="t/1#b.0@1.0", task_id="t/1", harness="hermes", source_run=source.id)
        threads = LaunchThreads(fork_launch.id, label=fork_launch.label, task_id="t/1", main_role="fork")
        request = chat(user("go t/1#b.0"))
        store.record_step(request=request, response=as_response(text_message("x")), assignment=threads.assign(request_path(request)))
        (fork,) = [run for run in store.launch_runs(fork_launch.id) if not run.is_thread]
        assert store.get_launch(fork_launch.id).owner["pid"] == os.getpid()
        with pytest.raises(OperationError, match="still running"):
            trash.trash(store, "run", fork.id)  # its process alive, it holds as before

        dead = {"pid": 2**22 + 12345, "host": socket.gethostname(), "start": 1}
        store._conn.execute("UPDATE launches SET owner = ? WHERE id = ?", (json.dumps(dead), fork_launch.id))
        store._conn.commit()
        assert launch_json(store.get_launch(fork_launch.id))["status"] == "interrupted"
        trash.trash(store, "run", fork.id)
        trash.trash(store, "run", source.id)
        assert trash.purge(store, "run", source.id).to_json()["removed"]["launches"] == 2

    def test_an_experiment_goes_with_the_other_arm_of_its_ab(self, store):
        run = record_run(store, "r", [text_message("a")], task_id="t/1")
        step = store.steps(run)[0]
        arms = [
            store.create_experiment(kind="ab", arm=arm, run_id=run, step_idx=0, step_id=step.id,
                                    request={"messages": []}, settings={"n": 1}, group_id="g1", status="done")
            for arm in ("original", "variant")
        ]

        moved = trash.trash(store, "experiment", arms[1].id)

        assert moved["id"] == "g1" and store.experiments(run_id=run) == []
        (listed,) = trash.listing(store).experiments
        assert listed["arms"] == 2
        trash.restore(store, "experiment", arms[0].id[:8])
        assert len(store.experiments(run_id=run)) == 2
        trash.trash(store, "experiment", arms[0].id)
        trash.purge(store, "experiment", "g1")
        assert store.get_experiment(arms[0].id) is None and store.get_experiment(arms[1].id) is None
        assert store.get_run(run) is not None  # the run it sampled stays

    def test_a_job_goes_only_once_it_has_finished(self, store):
        job = store.create_job(kind="resample", lane="model", params={})
        with pytest.raises(OperationError, match="still queued"):
            trash.trash(store, "job", job.id)
        store.update_job(job.id, status="done")

        trash.trash(store, "job", job.id)
        assert store.jobs() == []
        trash.restore(store, "job", job.id)
        assert [found.id for found in store.jobs()] == [job.id]
        trash.trash(store, "job", job.id)
        trash.purge(store, "job", job.id)
        assert store.get_job(job.id) is None

    def test_a_scenario_in_the_trash(self, store):
        from tuningfork.ops.scenarios import folder, save_scenario

        spec = {"skill_name": "s", "evals": [{"id": "1", "prompt": "go"}]}
        save_scenario(store.home, "demo", spec, etag=None)

        moved = trash.trash(store, "scenario", "demo")
        (listed,) = trash.listing(store).scenarios
        assert listed["name"] == "demo" and not folder(store.home, "demo").exists()
        trash.restore(store, "scenario", moved["id"])
        assert folder(store.home, "demo").is_dir()
        moved = trash.trash(store, "scenario", "demo")
        trash.purge(store, "scenario", moved["id"])
        assert trash.listing(store).empty

    def test_emptying_the_trash(self, store):
        runs = [record_run(store, f"r{index}", [text_message(str(index))], prompt=f"p{index}") for index in range(2)]
        for run in runs:
            trash.trash(store, "run", run)

        removed = trash.empty(store).to_json()

        assert removed["removed"]["runs"] == 2 and trash.listing(store).empty
        assert all(store.get_run(run) is None for run in runs)

    def test_a_reference_names_one_thing_in_the_trash(self, store):
        run = record_run(store, "named", [text_message("a")])
        trash.trash(store, "run", run)
        assert trash.find(store, "named") == ("run", run)
        assert trash.find(store, run[:10]) == ("run", run)
        with pytest.raises(OperationError, match="nothing in the trash"):
            trash.find(store, "nope")


def test_a_blob_seen_again_outlives_a_sweep_that_began_meanwhile(tmp_path):
    """A recording writes its blobs before the rows that refer to them, and may refer again to
    one a sweep is about to delete: putting it again marks it seen, and the sweep leaves it."""
    blobs = BlobStore(tmp_path / "blobs")
    digest = blobs.put(b"shared")
    os.utime(blobs._path(digest), (time.time() - 3600, time.time() - 3600))

    blobs.put(b"shared")  # seen again, by a recording
    assert blobs.sweep(set()) == (0, 0) and digest in blobs
    os.utime(blobs._path(digest), (time.time() - 3600, time.time() - 3600))
    assert blobs.sweep(set())[0] == 1 and digest not in blobs


API = "/_tuningfork/api"


async def test_the_trash_over_http(proxy_client, recorder):
    run = record_run(recorder.store, "r", [text_message("a")], task_id="t/1")

    moved = await proxy_client.post(f"{API}/trash", json={"kind": "run", "id": run})
    listed = (await proxy_client.get(f"{API}/trash")).json()
    refused = await proxy_client.post(f"{API}/trash/purge", json={"kind": "run", "id": "nope"})
    purged = await proxy_client.post(f"{API}/trash/purge", json={"kind": "run", "id": run})
    anonymous = await proxy_client.post(f"{API}/trash", json={"kind": "run", "id": run}, headers={"x-tuningfork-token": ""})

    assert moved.status_code == 200 and listed["runs"][0]["id"] == run
    assert refused.status_code == 404
    assert purged.status_code == 200 and purged.json()["removed"]["runs"] == 1
    assert anonymous.status_code in (401, 403)


def test_the_trash_from_the_command_line(store, monkeypatch):
    from rich.console import Console
    from typer.testing import CliRunner

    from tuningfork.cli import app

    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    run = record_run(store, "named", [text_message("a")], task_id="t/1")
    home = str(store.home)
    runner = CliRunner()

    moved = runner.invoke(app, ["trash", "run", "named", "--home", home])
    listed = runner.invoke(app, ["trash", "list", "--home", home])
    assert "moved run" in moved.output and run[:8] in listed.output and "1 run" in listed.output

    kept = runner.invoke(app, ["trash", "purge", "named", "--home", home], input="n\n")
    assert "kept" in kept.output and store.get_run(run) is not None

    gone = runner.invoke(app, ["trash", "purge", "named", "--yes", "--home", home])
    assert "deleted for good: 1 run" in gone.output and store.get_run(run) is None
    assert json.loads(runner.invoke(app, ["trash", "list", "--json", "--home", home]).output)["runs"] == []


class TestCitedByARefinement:
    """Deleting for good what a refinement cites says so first. An approval keeps the evidence
    it stood on, whatever is deleted after (the user, 2026-10-04)."""

    def test_it_is_said_before_and_an_approval_keeps_its_evidence(self, stack):
        from typer.testing import CliRunner

        from tuningfork.cli import app as cli
        from tuningfork.ops import approve, refine
        from tests.test_approve import validated

        made, proposal = validated(stack)
        store = stack.store
        trash.trash(store, "run", made.run_id)
        (cited,) = trash.preview(store, "run", made.run_id)["cited_by"]
        assert cited["refinement"] == made.id and cited["cites"] == ["its checkpoint"] and not cited["kept"]
        assert "can no longer screen or validate" in cited["says"]
        assert trash.listing(store).runs[0]["goes"]["cited_by"] == [cited]
        trash.restore(store, "run", made.run_id)

        plan = approve.prepare_approve(store, made.id, proposal.id)
        approve.execute_approve(store, plan, confirm=plan.code, front_end="cli")
        trash.trash(store, "run", made.run_id)
        (cited,) = trash.preview(store, "run", made.run_id)["cited_by"]
        assert cited["kept"] and "its approval keeps the evidence it stood on" in cited["says"]
        purged = CliRunner().invoke(cli, ["trash", "purge", made.run_id[:8], "--yes", "--home", str(store.home)])
        assert purged.exit_code == 0, purged.output
        assert f"refinement {made.id[:8]}" in purged.output and "keeps the evidence" in purged.output
        assert store.get_run(made.run_id) is None
        # Its evidence stands on its own: the approval, and the refinement's own page.
        evidence = approve.approval(store, made.id)["evidence"]
        assert evidence["validations"] and evidence["gate"]["passed"] and evidence["proposal"]["id"] == proposal.id
        shown = refine.view(store, made.id)
        assert shown["approval"]["state"] == "applied" and shown["refinement"]["status"] == "approved"

    def test_an_approved_refinement_deleted_for_good_leaves_what_the_approval_wrote(self, stack):
        from tuningfork.ops import approve
        from tests.test_approve import validated

        made, proposal = validated(stack)
        store = stack.store
        plan = approve.prepare_approve(store, made.id, proposal.id)
        approve.execute_approve(store, plan, confirm=plan.code, front_end="cli")
        written = stack.skill.read_bytes()
        trash.trash(store, "refinement", made.id)
        (note,) = trash.preview(store, "refinement", made.id)["notes"]
        assert "approved" in note and "changelog" in note
        assert trash.listing(store).refinements[0]["status"] == "approved"

        trash.purge(store, "refinement", made.id)

        assert not made.folder.exists()
        # The skill keeps the approved edit, its old version and the changelog entry: those are the
        # skill's own history, not the refinement's.
        beside = stack.skill.parent.parent
        assert stack.skill.read_bytes() == written
        assert (beside / "tally-report.changelog.md").is_file()
        assert [path.suffix for path in (beside / "tally-report.history").glob("SKILL.*")] == [".md"]

    def test_nothing_is_said_of_what_no_refinement_cites(self, store):
        run_id = record_run(store, "lonely", [text_message("done")])
        trash.trash(store, "run", run_id)
        assert trash.preview(store, "run", run_id)["cited_by"] == []
        assert trash.purge(store, "run", run_id).to_json()["cited_by"] == []
