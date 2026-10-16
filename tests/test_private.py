"""Held-out runs, verdicts only (`ops.private`): a held-out eval runs through a private proxy
whose folder is outside the recordings folder and gone afterwards. The recordings folder
gains nothing of it — no run, no blob, no snapshot, not a byte of its text — and its model
calls still pass the main proxy's gate."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import pytest

from tuningfork.ops.common import Cancel, Cancelled
from tuningfork.ops.private import OWNER, PREFIX, run_verdict_only, sweep_stale
from tuningfork.specs import Eval, TuningforkOptions
from tuningfork.store.db import Store
from tests.helpers import CHECK, VARIANT, task_stack

MARKER = "HELD-OUT-MARKER"


@pytest.fixture
def stack(tmp_path, mock_server, monkeypatch):
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    with task_stack(tmp_path, mock_server, monkeypatch) as built:
        held = Eval(
            id="2", prompt=f"Summarize data.csv into report.json. {MARKER}",
            files=[{"path": "data.csv", "content": f"id,amount\n1,10\n2,20\n3,30\n4,1.5\n# {MARKER}\n"}],
            tuningfork=TuningforkOptions(check=CHECK, held_out=True, skills=["tally-report"], max_turns=6, timeout=60),
        )
        built.spec.evals.append(held)
        built.held = held
        built.temp = temp
        yield built


def footprint(home: Path) -> dict:
    """What the recordings folder holds, beyond its database: blob files, snapshot refs."""
    blobs = sorted(str(path.relative_to(home)) for path in (home / "blobs").rglob("*") if path.is_file())
    refs = sorted(str(path.relative_to(home)) for path in (home / "snapshots.git").rglob("refs/**/*") if path.is_file()) \
        if (home / "snapshots.git").exists() else []
    return {"blobs": blobs, "refs": refs}


def mentions(home: Path, text: str) -> list[str]:
    return [str(path) for path in home.rglob("*") if path.is_file() and text.encode() in path.read_bytes()]


def variant(stack) -> Path:
    path = stack.tmp / "variant.md"
    path.write_text(VARIANT, encoding="utf-8")
    return path


def test_a_held_out_eval_comes_back_as_verdicts_and_leaves_nothing(stack):
    store = stack.store
    runs, before = len(store.runs(limit=500)), footprint(store.home)
    original = run_verdict_only(store, stack.spec, stack.held, n=2, harness="fake", model="m", proxy_url=stack.url)
    edited = run_verdict_only(store, stack.spec, stack.held, n=2, harness="fake", model="m", proxy_url=stack.url,
                              skill_file=variant(stack))
    assert [verdict.passed for verdict in original] == [False, False]
    assert [verdict.passed for verdict in edited] == [True, True]
    assert all(verdict.exclusion is None and verdict.server_id for verdict in original + edited)
    # Nothing of the held-out task is left in the recordings folder, or anywhere else.
    assert len(store.runs(limit=500)) == runs and footprint(store.home) == before
    assert mentions(store.home, MARKER) == []
    assert list(stack.temp.glob(f"{PREFIX}*")) == []
    relays = store._conn.execute("SELECT kind, eval, spec_path, status FROM launches WHERE kind = 'relay'").fetchall()
    assert len(relays) == 2
    assert all(json.loads(row["eval"]) == {} and row["spec_path"] is None and row["status"] != "running" for row in relays)
    # Its calls reached the model, through the main proxy: the mock saw the held-out prompt.
    assert any(MARKER in str(request) for request in stack.mock.requests)


def test_cancelled_it_leaves_no_verdict_and_no_folder(stack):
    cancel = Cancel()
    threading.Timer(0.5, cancel.cancel).start()
    with pytest.raises(Cancelled):
        run_verdict_only(stack.store, stack.spec, stack.held, n=50, harness="fake", model="m",
                         proxy_url=stack.url, cancel=cancel)
    assert list(stack.temp.glob(f"{PREFIX}*")) == []
    assert mentions(stack.store.home, MARKER) == []


def test_kept_for_the_person_it_is_a_recordings_folder_of_its_own(stack):
    keep = stack.tmp / "kept"
    run_verdict_only(stack.store, stack.spec, stack.held, n=1, harness="fake", model="m",
                     proxy_url=stack.url, keep=keep)
    assert (keep / "tuningfork.db").is_file() and not (keep / OWNER).exists()
    assert list(stack.temp.glob(f"{PREFIX}*")) == []
    assert mentions(stack.store.home, MARKER) == []


def test_held_out_runs_are_held_to_the_main_proxys_cap(stack):
    """The main proxy records none of a relay's calls, and so caps none: the private proxy holds
    them to its cap, and records each as the model was sent it — as the training runs are."""
    keep = stack.tmp / "kept"
    run_verdict_only(stack.store, stack.spec, stack.held, n=1, harness="fake", model="m",
                     proxy_url=stack.url, keep=keep, max_output=50)
    held = [request for request in stack.mock.requests if MARKER in str(request)]
    assert held and all(request["max_tokens"] == 50 for request in held)
    private = Store(keep)
    try:
        steps = [step for run in private.runs(limit=10) for step in private.steps(run.id)]
        assert steps and all(step.params["output_cap"] == {"cap": 50, "asked": {}, "written": ["max_tokens"]}
                             for step in steps)
    finally:
        private.close()


def test_a_folder_whose_owner_died_is_swept(tmp_path):
    dead = tmp_path / f"{PREFIX}dead"
    dead.mkdir()
    (dead / OWNER).write_text("999999999")
    alive = tmp_path / f"{PREFIX}alive"
    alive.mkdir()
    import os

    (alive / OWNER).write_text(str(os.getpid()))
    assert sweep_stale(tmp_path) == [dead]
    assert alive.exists() and not dead.exists()


def test_a_temporary_folder_that_cannot_hold_the_runs_is_refused_before_any_run(tmp_path, monkeypatch):
    """Found on the real stack: refused when the validation is planned, not when its held-out
    part begins, twenty minutes in."""
    from tuningfork import paths
    from tuningfork.ops.common import OperationError
    from tuningfork.ops.private import check_private_base

    drive = tmp_path / "c" / "Temp"
    drive.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(drive))
    monkeypatch.setattr(paths, "under_wsl", lambda: True)
    monkeypatch.setattr(paths, "drive_root", lambda: str(tmp_path) + "/")
    with pytest.raises(OperationError) as refused:
        check_private_base(tmp_path / "home")
    assert refused.value.code == "private_home" and "TMPDIR" in refused.value.hint
    monkeypatch.setattr(paths, "under_wsl", lambda: False)
    with pytest.raises(OperationError):
        check_private_base(tmp_path / "c")  # inside the recordings folder
    assert check_private_base(tmp_path / "home") == drive.resolve()


def test_a_proxy_starting_sweeps_what_a_dead_process_left(tmp_path, monkeypatch):
    """`serve` killed mid-validation leaves a private folder: the next `serve` removes it."""
    from tuningfork.config import Settings
    from tuningfork.proxy.app import create_app

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    dead = tmp_path / f"{PREFIX}killed"
    dead.mkdir()
    (dead / OWNER).write_text("999999999")
    app = create_app(Settings(upstream_base_url="http://upstream.test", home=tmp_path / "home"))
    app.state.recorder.store.close()
    assert not dead.exists()
