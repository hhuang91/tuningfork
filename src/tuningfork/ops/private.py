"""Held-out runs, verdicts only.

A held-out eval judges an edit, so nothing of its runs may stay where an agent could read
them: not in the recordings folder, not on the proxy's open API. Yet a run's verdict is read
back from its recorded steps — whether a model call failed, whether a thread broke off, what
the final reply said. So a held-out eval runs through a private proxy that records into a
temporary folder of its own, outside the recordings folder and on the Linux filesystem; that
proxy serves no reads (`create_app(api=False)`), and the folder is deleted when the runs
end. Its model calls reach the server through the main proxy, by a relay launch that is
never recorded there (`LaunchState.records`), so the person's gate and pause still apply.

Only trimmed verdicts leave: whether each run passed, why it was not counted, whether it
timed out, how long it took, and which server answered. `keep` copies the folder, for the
person to debug with `tuningfork ui --home <keep>`, to a place they choose.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tuningfork.ops.common import Cancel, OperationError, Progress, emit
from tuningfork.store.db import Store

PREFIX = "tuningfork-private-"
#: The process a private folder belongs to: a folder whose owner is gone is swept away.
OWNER = "owner.pid"


@dataclass(frozen=True, slots=True)
class Verdict:
    """A held-out run, as much of it as leaves its private folder."""

    index: int
    passed: bool | None
    exclusion: str | None
    timed_out: bool
    duration_s: float | None
    server_id: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "passed": self.passed,
            "exclusion": self.exclusion,
            "timed_out": self.timed_out,
            "duration_s": None if self.duration_s is None else round(self.duration_s, 1),
            "server_id": self.server_id,
        }


@dataclass(slots=True)
class PrivateProxy:
    home: Path
    url: str
    relay: str
    store: Store


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale(base: Path | None = None) -> list[Path]:
    """Private folders left behind by a process that died mid-run: removed."""
    removed = []
    for folder in Path(base or tempfile.gettempdir()).glob(f"{PREFIX}*"):
        try:
            owner = int((folder / OWNER).read_text().strip())
        except (OSError, ValueError):
            owner = None
        if owner is not None and _alive(owner):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed.append(folder)
    return removed


def check_private_base(home: Path) -> Path:
    """Where private folders will be made, refused before any run when SQLite could not lock
    there, or when it is inside the recordings folder."""
    from tuningfork.paths import drive_root, under_wsl

    base = Path(tempfile.gettempdir()).resolve()
    if under_wsl() and str(base).startswith(drive_root()):
        raise OperationError(
            f"held-out runs would be recorded on a Windows drive ({base}), where SQLite cannot lock",
            hint="set TMPDIR to a folder on the Linux filesystem", code="private_home",
        )
    if base == Path(home).resolve() or Path(home).resolve() in base.parents:
        raise OperationError(f"the temporary folder ({base}) is inside the recordings folder",
                             hint="set TMPDIR to a folder outside it", code="private_home")
    return base


def private_home(home: Path) -> Path:
    """A folder only this user can open, outside the recordings folder, on a filesystem where
    SQLite can lock. It starts with a copy of the recordings folder's settings: the harness
    programs named there, the endpoints protected there."""
    from tuningfork.config import HOME_CONFIG
    from tuningfork.paths import drive_root, under_wsl

    sweep_stale()
    folder = Path(tempfile.mkdtemp(prefix=PREFIX))  # 0700
    if under_wsl() and str(folder).startswith(drive_root()):
        shutil.rmtree(folder, ignore_errors=True)
        raise OperationError(
            f"the private recordings would be on a Windows drive ({folder}), where SQLite cannot lock",
            hint="set TMPDIR to a folder on the Linux filesystem",
            code="private_home",
        )
    if str(folder.resolve()).startswith(str(Path(home).resolve()) + os.sep):
        shutil.rmtree(folder, ignore_errors=True)
        raise OperationError("the private folder would be inside the recordings folder", code="private_home")
    (folder / OWNER).write_text(str(os.getpid()), encoding="utf-8")
    config = Path(home) / HOME_CONFIG
    if config.is_file():
        shutil.copyfile(config, folder / HOME_CONFIG)
    return folder


@contextmanager
def private_proxy(
    store: Store, proxy_url: str, *, concurrency: int = 1, keep: Path | None = None,
    max_output: int | None = None,
) -> Iterator[PrivateProxy]:
    """A private proxy for held-out runs, for as long as the block lasts; its folder is gone
    afterwards, whatever happened — unless `keep` asks for a copy of it.

    `proxy_url` is the main proxy's root: the relay launch is made in `store`, and the
    private proxy sends every call through it. The main proxy records none of the relay's
    calls, and so caps none: `max_output` is its cap, held here, where they are recorded."""
    import uvicorn

    from tuningfork.config import Settings
    from tuningfork.launches import launch_url, proxy_root
    from tuningfork.proxy.app import create_app

    folder = private_home(store.home)
    relay = store.create_launch(
        kind="relay", label="held-out relay", settings={"record": False, "breakpoints": False}
    )
    server = thread = private_store = None
    try:
        settings = Settings(
            upstream_base_url=launch_url(proxy_root(proxy_url), relay.id), home=folder,
            max_inflight=max(1, concurrency), mcp_tools="none", max_output=max_output,
        )
        app = create_app(settings, api=False)
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, name="tuningfork-private-proxy", daemon=True)
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise OperationError("the private proxy for held-out runs did not start", code="private_home")
            time.sleep(0.02)
        private_store = Store(folder)
        yield PrivateProxy(home=folder, url=f"http://127.0.0.1:{port}", relay=relay.id, store=private_store)
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(30)
        if private_store is not None:
            private_store.close()
        store.end_launch(relay.id)
        if keep is not None:
            shutil.copytree(folder, keep, ignore=shutil.ignore_patterns(OWNER))
        shutil.rmtree(folder, ignore_errors=True)


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_verdict_only(
    store: Store,
    spec: Any,
    item: Any,
    *,
    n: int,
    harness: str,
    model: str | None,
    proxy_url: str,
    skill_file: Path | None = None,
    concurrency: int = 1,
    keep: Path | None = None,
    max_output: int | None = None,
    label: str = "held-out",
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> list[Verdict]:
    """Run a held-out eval n times, privately, and return only how each run ended — each held
    to `max_output` tokens a call, the main proxy's cap, as the runs it is compared with are.

    The progress events say a run ended, never how: they reach whoever follows the job."""
    from tuningfork.ops.batch import execute_batch, prepare_batch

    with private_proxy(store, proxy_url, concurrency=concurrency, keep=keep, max_output=max_output) as private:
        plan = prepare_batch(
            private.store, spec, eval_id=item.id, n=n, model=model, harness=harness,
            skill_file=skill_file, workdir=private.home / "batches", concurrency=concurrency,
            base_url=f"{private.url}/v1", include_held_out=True,
            # In the sandbox, the recordings the main proxy keeps are out of sight too.
            hide=[store.home],
        )

        def ended(event: Any) -> None:
            if event.kind == "run":
                emit(progress, "held_out_run", label=label, index=event.data.get("index"))
            elif event.kind == "waiting":
                emit(progress, "waiting", **event.data)

        report = execute_batch(private.store, plan, progress=ended, cancel=cancel)
        verdicts = []
        for index, outcome in enumerate(report.outcomes):
            run = private.store.get_run(outcome.run_id) if outcome.run_id else None
            verdicts.append(Verdict(
                index=index,
                passed=outcome.passed if outcome.checked else None,
                exclusion=outcome.exclusion,
                timed_out=bool(getattr(outcome, "timed_out", False)),
                duration_s=getattr(outcome, "duration_s", None),
                server_id=run.server_id if run is not None else None,
            ))
        return verdicts
