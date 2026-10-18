"""Approving a validated edit, and writing it back (brief §7, step 6).

A person approves one proposal of a refinement, on a validation whose gate it meets
(`analysis.validation`), with a code that names exactly what will be written where: the
proposal, the file, and the bytes the file holds now. The edit then goes into the skill's own
SKILL.md, in place, under a lock, in five steps that a crash between leaves recoverable:

1. `approval.json` in the refinement, written once: what is written where, every piece of
   evidence it stands on (the refinement, the proposal, its screens, every validation —
   failed ones too — and the gate), and who approved it. It stands when the runs are purged.
2. The file as it was, kept beside the skill folder: `<skill>.history/SKILL.<stamp>-<sha8>.md`.
3. The new text, written to a temporary file and moved over the old one, with the old one's
   line endings and byte-order mark — once its bytes are seen to be the ones approved.
4. An entry in `<skill>.changelog.md`, beside the skill folder: the diff, why, who proposed
   it, the evidence, who approved it.
5. `applied.json` in the refinement, and the refinement marked approved.

Beside the skill folder, never inside it: every run is given a copy of that folder, and a
model reading an old wording, or the verdicts of a changelog, would be measured on it.
No MCP tool reaches any of this: approving is the person's.
"""

from __future__ import annotations

import errno
import getpass
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Store

APPROVAL, APPLIED = "approval.json", "applied.json"
BOM = b"\xef\xbb\xbf"
#: Opens each entry of a changelog: how a repair knows an entry was already written.
MARKER = "<!-- tuningfork approval {id} -->"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def confirmation(proposal_id: str, target: Path, current: bytes) -> str:
    """The code a person types to approve: it names the proposal, the file, and the bytes the
    file holds now — so a code read off one plan approves nothing else."""
    return _sha(f"{proposal_id}|{target}|".encode() + current)[:8]


def render(text: str, current: bytes) -> bytes:
    """`text` as the file's bytes: with the line endings and byte-order mark the file has now,
    whatever the proposal's text was written with."""
    from tuningfork.experiment.skill import normalize_newlines

    body = normalize_newlines(text).lstrip("﻿")
    if b"\r\n" in current:
        body = body.replace("\n", "\r\n")
    data = body.encode("utf-8")
    return BOM + data if current.startswith(BOM) else data


@dataclass(frozen=True, slots=True)
class Target:
    """The file an approval writes, and what is kept beside its folder."""

    path: Path
    #: The task spec it was worked out from; None when a person named the file.
    spec: Path | None

    @property
    def folder(self) -> Path:
        return self.path.parent

    @property
    def history(self) -> Path:
        return self.folder.parent / f"{self.folder.name}.history"

    @property
    def changelog(self) -> Path:
        return self.folder.parent / f"{self.folder.name}.changelog.md"

    def to_json(self) -> dict[str, Any]:
        from tuningfork.paths import windows_form

        return {
            "path": str(self.path), "windows": windows_form(self.path),
            "spec": str(self.spec) if self.spec else None,
            "history": str(self.history), "changelog": str(self.changelog),
        }


def find_target(store: Store, refinement: Any, *, spec: str | None = None, target: str | None = None) -> Target:
    """The SKILL.md an approval writes: worked out again from the task spec the checkpoint's run
    was launched from, and refused when that no longer leads where the refinement began —
    unless the person names the spec (`spec`) or the file (`target`) themselves."""
    from tuningfork.ops import refine
    from tuningfork.paths import resolve_path
    from tuningfork.specs import SpecError, load_spec

    if spec and target:
        raise OperationError("name the spec or the file to write, not both", code="bad_params", field="target")
    if target:
        found = resolve_path(target).path
        if found.name != "SKILL.md" or not found.is_file():
            raise OperationError(f"{found} is not a skill's SKILL.md", code="bad_params", field="target")
        return Target(found, None)
    if spec:
        path = resolve_path(spec).path
        try:
            folder = load_spec(path, strict=False).resolved_skill_dir()
        except (OSError, SpecError) as error:
            raise OperationError(f"the spec cannot be read: {error}", code="bad_params", field="spec") from None
        if folder is None or not (folder / "SKILL.md").is_file():
            raise OperationError(f"{path} names no skill folder holding a SKILL.md", code="bad_params", field="spec")
        return Target(folder / "SKILL.md", path)

    recorded = (refinement.data.get("source") or {}).get("path")
    run = store.get_run(refinement.run_id)
    launch = refine._launch_of(store, run) if run is not None else None
    spec_path = Path(launch.spec_path) if launch is not None and launch.spec_path else None
    if spec_path is None or not spec_path.is_file():
        raise OperationError(
            "the task spec the checkpoint's run was launched from is not there"
            + (f" ({spec_path})" if spec_path else ""),
            hint="name where the spec is now with --spec, or the file to write with --target",
            code="moved", field="spec",
        )
    found = refine.source_skill(store, run)
    if found is None:
        raise OperationError(f"{spec_path} names no skill folder holding a SKILL.md", code="moved", field="spec")
    if recorded and found.resolve() != Path(recorded).resolve():
        raise OperationError(
            f"the spec now leads to {found}, but the refinement began from {recorded}",
            hint="name the file to write with --target",
            code="moved", field="target",
        )
    return Target(found, spec_path)


@dataclass(slots=True)
class ApprovePlan:
    refinement: Any
    proposal: Any
    record: dict[str, Any]
    gate: Any
    target: Target
    #: The file's bytes now, and as the approval would leave them.
    before: bytes
    after: bytes
    code: str
    diff: dict[str, Any]
    #: How many validations the proposal has, and how many of them meet the gate.
    validations: int
    passing: int
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """What the person confirms. For the person only: the gate holds held-out verdicts."""
        return {
            "refinement": self.refinement.id,
            "proposal": self.proposal.id,
            "validation": self.record["id"],
            "target": self.target.to_json(),
            "line_endings": "CRLF" if b"\r\n" in self.before else "LF",
            "bom": self.before.startswith(BOM),
            "diff": self.diff,
            "gate": self.gate.to_json(),
            "validations": {"of_this_proposal": self.validations, "meeting_the_gate": self.passing},
            "code": self.code,
            "warnings": self.warnings,
        }


def _validation(records: list[dict[str, Any]], proposal: Any, reference: str | None) -> dict[str, Any]:
    mine = [record for record in records if record.get("proposal") == proposal.id]
    if reference:
        found = [record for record in mine if record["id"].startswith(reference)]
        if len(found) != 1:
            raise OperationError(f"no single validation of proposal {proposal.id[:8]} matching {reference!r}",
                                 code="not_found", field="validation")
        return found[0]
    complete = [record for record in mine if record.get("complete")]
    if not complete:
        raise OperationError(
            f"proposal {proposal.id[:8]} has no finished validation",
            hint="validate it first: `tuningfork refine validate <refinement> <proposal>`",
            code="not_validated", field="proposal",
        )
    return complete[0]  # the newest: an older one that passed is chosen only by name


def prepare_approve(
    store: Store,
    refinement_ref: str,
    proposal_ref: str,
    *,
    validation: str | None = None,
    spec: str | None = None,
    target: str | None = None,
) -> ApprovePlan:
    """An approval, checked and shown before anything is written: the file, the diff, the gate,
    and the code that approves exactly this."""
    from tuningfork.analysis.validation import judge
    from tuningfork.experiment.skill import normalize_newlines, text_sha256
    from tuningfork.ops import refine
    from tuningfork.ops.diff import line_diff
    from tuningfork.ops.validate import validations as validation_records

    refinement = refine.get(store, refinement_ref)
    if refinement.status == "approved" or (refinement.folder / APPROVAL).is_file():
        raise OperationError(f"refinement {refinement.id[:8]} has an approved edit already", code="approved",
                             field="refinement")
    refine.refuse_closed(refinement)
    proposal = refine.proposal(store, refinement.id, proposal_ref)
    records = validation_records(store, refinement.id)
    record = _validation(records, proposal, validation)
    found = find_target(store, refinement, spec=spec, target=target)
    try:
        before = found.path.read_bytes()
    except OSError as error:
        raise OperationError(f"cannot read {found.path}: {error.strerror}", field="target") from None
    current = before.decode("utf-8", errors="replace")
    matches = text_sha256(current) == refinement.data["baseline"]["sha256"]
    gate = judge(record, target_matches=matches, refinement_open=True, approved=False)
    if not gate.passed:
        raise OperationError(
            f"validation {record['id'][:8]} does not meet the gate: "
            + "; ".join(f"{check.id} {check.reason}" for check in gate.failed()),
            hint="`tuningfork refine show` shows every check with its numbers",
            code="gate", field="validation",
        )
    after = render(proposal.text(), before)
    if after == before:
        raise OperationError("the edit changes nothing in this file", code="no_change", field="proposal")
    passing = sum(
        1 for each in records if each.get("proposal") == proposal.id
        and judge(each, target_matches=matches, refinement_open=True, approved=False).passed
    )
    mine = sum(1 for each in records if each.get("proposal") == proposal.id)
    warnings = list(gate.notes)
    if mine > 1:
        warnings.append(f"proposal {proposal.id[:8]} was validated {mine} times, and {passing} of them "
                        f"{'meets' if passing == 1 else 'meet'} the gate: the approval keeps them all")
    return ApprovePlan(
        refinement=refinement, proposal=proposal, record=record, gate=gate, target=found,
        before=before, after=after, code=confirmation(proposal.id, found.path, before),
        diff=line_diff(normalize_newlines(current), normalize_newlines(after.decode("utf-8"))),
        validations=mine, passing=passing, warnings=warnings,
    )


# --- writing it -------------------------------------------------------------------------------------


class _Lock:
    """An exclusive lock on a skill's history folder, held while an approval writes the skill:
    a second approval of the same file is refused, not interleaved. Released when the process
    ends, however it ends."""

    def __init__(self, target: Target) -> None:
        self.path = target.history / ".lock"
        self.target = target
        self.handle: Any = None

    def __enter__(self) -> _Lock:
        import fcntl

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.path, "a+", encoding="utf-8")
        except PermissionError:
            raise _permission(self.path.parent) from None
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise OperationError(
                f"another approval is writing {self.target.path} now",
                hint="wait for it, then look again: `tuningfork refine show`", code="locked", field="target",
            ) from None
        except OSError as error:
            # A filesystem that keeps no locks: the bytes are still compared before the move.
            if error.errno not in (errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL):
                self.handle.close()
                raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.handle.close()  # and with it, the lock


def _permission(path: Path) -> OperationError:
    return OperationError(
        f"cannot write {path}: permission denied",
        hint="is the file open in another program, or read-only? Nothing was changed",
        code="permission", field="target",
    )


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _backup_path(target: Target, before: bytes, at: float) -> Path:
    """Where the file as it was is kept: named by when, and by its bytes."""
    base = f"SKILL.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(at))}-{_sha(before)[:8]}"
    path, suffix = target.history / f"{base}.md", 1
    while path.exists() and path.read_bytes() != before:
        suffix += 1
        path = target.history / f"{base}.{suffix}.md"
    return path


def _keep(path: Path, before: bytes) -> bool:
    """Keep the file as it was at `path`; False when it was kept there already."""
    if path.exists() and path.read_bytes() == before:
        return False
    _write_bytes(path, before)
    return True


def _replace_file(target: Target, after: bytes, expected: bytes) -> None:
    """The new bytes over the old, atomically: written beside, made durable, and moved in only
    if the file still holds the bytes approved."""
    temporary = target.history / f".SKILL.md.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    try:
        _write_bytes(temporary, after)
        if target.path.read_bytes() != expected:
            raise OperationError(
                f"{target.path} changed while the approval was being written: nothing was replaced",
                hint="look at it again, and approve again if it still should be", code="drifted", field="target",
            )
        os.replace(temporary, target.path)
    except PermissionError:
        raise _permission(target.path) from None
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS, errno.EBUSY):
            raise _permission(target.path) from None
        raise
    finally:
        temporary.unlink(missing_ok=True)
    try:
        folder = os.open(target.folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(folder)
    except OSError:
        pass  # not every filesystem syncs a folder; the file itself is synced
    finally:
        os.close(folder)


def _entry(approval: dict[str, Any]) -> str:
    """The changelog's entry for an approval: what changed, why, on what evidence, by whom."""
    intent, evidence = approval["intent"], approval["evidence"]
    proposal, gate = evidence["proposal"], evidence["gate"]
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(approval["created_at"]))
    who = approval["approver"]
    lines = [
        MARKER.format(id=approval["id"]),
        f"## {when} — refinement {approval['refinement'][:8]}, proposal {approval['proposal'][:8]}",
        "",
        f"Approved by {who.get('user') or 'a person'} ({who.get('front_end') or 'tuningfork'}), on validation "
        f"{approval['validation'][:8]}: " + ("every check of the gate met." if gate["passed"] else "the gate was not met."),
        "",
    ]
    for each in proposal.get("proposers") or []:
        source = ", ".join(str(each[key]) for key in ("via", "client", "model") if each.get(key))
        lines.append(f"- Why ({source}): {(each.get('rationale') or '').strip() or 'no reason given'}")
        shas = [f"{key} {each[key]}" for key in ("packet", "prompt") if each.get(key)]
        if shas:
            lines.append(f"  shown {', '.join(shas)}")
    target = evidence["refinement"].get("target") or {}
    lines += ["", f"Target at step {evidence['refinement'].get('step')}: {target.get('description') or 'as declared'}", ""]
    for check in gate["checks"]:
        flag = "met" if check["passed"] else "NOT met"
        powered = " (low-powered)" if check.get("low_powered") else ""
        lines.append(f"- {check['id']} {check['name']}: {flag}{powered} — {check['reason']}")
    for note in gate.get("notes") or []:
        lines.append(f"- note: {note}")
    lines += ["", "```diff"]
    for hunk in intent["diff"]["hunks"]:
        lines.append(f"@@ -{hunk['old_start']} +{hunk['new_start']} @@")
        lines += [f"{line['op']}{line['text']}" for line in hunk["lines"]]
    lines += [
        "```", "",
        f"The file before: {intent['backup_name']}, beside this changelog in {Path(intent['history']).name}/. "
        f"Evidence in full: the refinement's approval.json.",
        "", "",
    ]
    return "\n".join(lines)


def _append_changelog(target: Target, approval: dict[str, Any]) -> None:
    marker = MARKER.format(id=approval["id"])
    if target.changelog.is_file() and marker in target.changelog.read_text(encoding="utf-8"):
        return
    fresh = not target.changelog.exists()
    with open(target.changelog, "a", encoding="utf-8", newline="\n") as handle:
        if fresh:
            handle.write(f"# {target.folder.name}: changes approved in tuningfork\n\n")
        handle.write(_entry(approval))
        handle.flush()
        os.fsync(handle.fileno())


def _finish(refinement: Any, approval: dict[str, Any]) -> None:
    from tuningfork.ops.refine import _replace, _write_once

    try:
        _write_once(refinement.folder / APPLIED, {"at": time.time(), "after": approval["intent"]["after"]["sha256"]})
    except FileExistsError:
        pass
    _replace(refinement.folder / "status.json", {
        "status": "approved", "proposal": approval["proposal"], "approved_at": approval["created_at"],
    })


@dataclass(slots=True)
class ApproveResult:
    approval: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        intent = self.approval["intent"]
        return {
            "approval": self.approval["id"], "refinement": self.approval["refinement"],
            "proposal": self.approval["proposal"], "validation": self.approval["validation"],
            "target": intent["target"], "windows": intent.get("windows"),
            "backup": str(Path(intent["history"]) / intent["backup_name"]), "changelog": intent["changelog"],
            "state": "applied",
        }


def execute_approve(
    store: Store,
    plan: ApprovePlan,
    *,
    confirm: str,
    front_end: str,
) -> ApproveResult:
    """Write the approved edit back, in the five steps the module describes. `confirm` must be
    the plan's code: typed by the person, never supplied for them."""
    from tuningfork.ops import refine
    from tuningfork.ops.refine import _write_once
    from tuningfork.ops.validate import validations as validation_records

    if (confirm or "").strip().lower() != plan.code:
        # Never told here: a code anyone could learn from a refusal confirms nothing.
        raise OperationError(
            "that is not the code of this approval as it stands",
            hint="type the code shown with it; if the file changed since it was shown, look at it again",
            code="confirm", field="confirm",
        )
    target = plan.target
    with _Lock(target):
        refinement = refine.get(store, plan.refinement.id)
        approval_path = refinement.folder / APPROVAL
        if refinement.trashed_at is not None:
            refine.refuse_closed(refinement)  # moved to the trash since the approval was shown
        if refinement.status != "open" or approval_path.exists():
            raise OperationError(f"refinement {refinement.id[:8]} is {refinement.status} or approved already",
                                 code="approved", field="refinement")
        if target.path.read_bytes() != plan.before:
            raise OperationError(
                f"{target.path} changed since the approval was shown: nothing was written",
                hint="look at it again, and approve again if it still should be", code="drifted", field="target",
            )
        at = time.time()
        backup = _backup_path(target, plan.before, at)
        approval = {
            "id": f"{refinement.id}-{plan.proposal.id}",
            "refinement": refinement.id, "proposal": plan.proposal.id, "validation": plan.record["id"],
            "created_at": at,
            "approver": {"via": "person", "front_end": front_end, "user": _user()},
            "intent": {
                **{key: value for key, value in target.to_json().items()},
                "target": str(target.path),
                "before": {"sha256": _sha(plan.before)},
                "after": {"sha256": _sha(plan.after), "text": plan.after.decode("utf-8")},
                "line_endings": "CRLF" if b"\r\n" in plan.before else "LF",
                "bom": plan.before.startswith(BOM),
                "diff": plan.diff, "backup_name": backup.name, "code": plan.code,
            },
            "evidence": {
                "refinement": refinement.to_json(),
                "proposal": plan.proposal.to_json(),
                "screens": refine.screens(store, refinement.id),
                "validations": validation_records(store, refinement.id),
                "gate": plan.gate.to_json(),
            },
        }
        try:
            _write_once(approval_path, approval)
        except FileExistsError:
            raise OperationError(f"refinement {refinement.id[:8]} was approved meanwhile", code="approved",
                                 field="refinement") from None
        except PermissionError:
            raise _permission(approval_path) from None
        kept = False
        try:
            kept = _keep(backup, plan.before)
            _replace_file(target, plan.after, plan.before)
        except (OperationError, OSError):
            # Refused before the skill's file was touched: the approval did not happen.
            approval_path.unlink(missing_ok=True)
            if kept:
                backup.unlink(missing_ok=True)
            raise
        _append_changelog(target, approval)
        _finish(refinement, approval)
    return ApproveResult(approval)


def _user() -> str | None:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return None


# --- after a crash --------------------------------------------------------------------------------


def approval(store: Store, refinement_ref: str) -> dict[str, Any] | None:
    """The refinement's approval as written, or None."""
    from tuningfork.ops import refine

    path = refine.get(store, refinement_ref).folder / APPROVAL
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def state(store: Store, refinement_ref: str) -> dict[str, Any] | None:
    """Where an approval stands: `applied`; or interrupted, with the file `not_written` or
    `written`, or applied with the refinement not yet marked (`unmarked`); or `diverged` —
    the file holds neither the bytes approved from nor those approved, so nothing more is
    done to it. None without an approval."""
    from tuningfork.ops import refine

    refinement = refine.get(store, refinement_ref)
    found = approval(store, refinement.id)
    if found is None:
        return None
    intent = found["intent"]
    shown = {"approval": found["id"], "proposal": found["proposal"], "target": intent["target"],
             "windows": intent.get("windows"), "changelog": intent["changelog"],
             "backup": str(Path(intent["history"]) / intent["backup_name"])}
    if (refinement.folder / APPLIED).is_file():
        # The last two marks are two writes: between them the edit is applied and the
        # refinement still open (found in review: read as applied, it was never marked).
        if refinement.status == "open":
            return {**shown, "state": "unmarked", "message": "the edit was written and its changelog entry made, "
                    "but the approval was interrupted before the refinement was marked approved: repair marks it"}
        return {**shown, "state": "applied"}
    try:
        now = _sha(Path(intent["target"]).read_bytes())
    except OSError:
        now = None
    if now == intent["after"]["sha256"]:
        return {**shown, "state": "written", "message": "the edit was written, but the approval was interrupted "
                "before its changelog entry and its mark were: repair finishes them"}
    if now == intent["before"]["sha256"]:
        return {**shown, "state": "not_written", "message": "the approval was interrupted before the file was "
                "written: repair writes it, as approved"}
    return {**shown, "state": "diverged", "message": "the file holds neither the text approved from nor the text "
            "approved: it was changed since. Nothing more is done to it"}


def repair(store: Store, refinement_ref: str) -> dict[str, Any]:
    """Finish an approval a crash interrupted, from what `approval.json` recorded: the file
    written if it still holds the bytes approved from, then the changelog entry and the mark."""
    from tuningfork.ops import refine

    refinement = refine.get(store, refinement_ref)
    found = approval(store, refinement.id)
    if found is None:
        raise OperationError(f"refinement {refinement.id[:8]} has no approval to repair", code="not_found")
    intent = found["intent"]
    target = Target(Path(intent["target"]), Path(intent["spec"]) if intent.get("spec") else None)
    with _Lock(target):
        current = state(store, refinement.id)
        if current["state"] == "diverged":
            raise OperationError(current["message"], code="drifted", field="target")
        if current["state"] == "not_written":
            before = target.path.read_bytes()
            _keep(target.history / intent["backup_name"], before)
            _replace_file(target, intent["after"]["text"].encode("utf-8"), before)
        if current["state"] != "applied":
            _append_changelog(target, found)
            _finish(refinement, found)
    return state(store, refinement.id)
