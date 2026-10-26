"""The reflection packet: what a proposer is shown before it suggests an edit (brief §7,
step 2).

It holds the step as the model saw it, where the skill sits in that context, what runs of
the same skill version did there and how often each behaviour went on to pass, answers
sampled at the checkpoint that did and did not do what the target describes — with their
reasoning, which often names the instruction the model was weighing — the task's own
criteria, and what was proposed and screened before.

Built only from the refinement's own task's records: a held-out task never reaches a proposer.
What the store keeps of them is masked as it was kept; the run's skill, kept as it was given,
and the task's criteria, as its file wrote them, are masked here — the skill whole, each line
keeping its number, then the rest of what the masks took in it (`Masking.shown`). A proposer's
model may be hosted. Kept by its digest in the refinement's folder (`packets/<sha>.json`), so
every proposal says exactly what its proposer saw. Where there is no evidence of a kind, the
packet says so rather than leaving it out: "no answer sampled here reached the target" is
itself what a proposer must know.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tuningfork.ops import compact
from tuningfork.ops.common import OperationError
from tuningfork.store.db import Store
from tuningfork.store.masking import Masked

#: Answers shown of each kind: some that reached the target, some that did not.
EACH = 3
#: What a packet may weigh, as JSON: about six thousand tokens.
BUDGET = 24_000


@dataclass(slots=True)
class Packet:
    sha: str
    data: dict[str, Any]

    @property
    def size(self) -> int:
        return len(canonical(self.data))

    def to_json(self) -> dict[str, Any]:
        return {"sha": self.sha, "size": self.size, "packet": self.data}

    def keep(self, store: Store, folder: Path) -> None:
        """Keep it in a refinement's folder, by its digest: what a proposal cites. It is masked
        as it was built (`build_packet`), and kept as it is (`Store.keep_file`)."""
        path = folder / "packets" / f"{self.sha}.json"
        if not path.exists():
            store.keep_file(path, Masked(self.data))


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ends(text: Any, head: int = 500, tail: int = 700) -> Any:
    """A long reasoning kept at both ends: a decision tends to sit at its end."""
    if not isinstance(text, str) or len(text) <= head + tail:
        return text
    return f"{text[:head]} … (+{len(text) - head - tail} chars) … {text[-tail:]}"


def _answer(response: dict[str, Any] | None, outcome: Any, *, source: str) -> dict[str, Any]:
    message = (((response or {}).get("choices") or [{}])[0].get("message")) or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    calls = [
        {"name": (call.get("function") or {}).get("name"),
         "arguments": compact.clip((call.get("function") or {}).get("arguments"), 600)}
        for call in message.get("tool_calls") or []
    ]
    data = {
        "source": source,
        "did": compact.clip(outcome.label, compact.LABEL),
        "content": compact.clip(content, 800) if content else None,
        "tool_calls": calls or None,
        "reasoning": _ends(message.get("reasoning_content") or message.get("reasoning")),
    }
    return {key: value for key, value in data.items() if value is not None}


def _continuations(store: Store, refinement: Any, checkpoint: Any, target: Any) -> dict[str, Any]:
    """Answers at the checkpoint, sorted by whether they reached the target: from the step's
    latest resample, the baselines of earlier screens, and the run's own answer — never from
    an arm where a proposal was swapped in."""
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.experiment import experiment_outcomes
    from tuningfork.experiment.sampling import step_resample
    from tuningfork.ops.targets import verdict_from

    sources = []
    resample = step_resample(store, checkpoint)
    if resample is not None:
        sources.append(resample)
    for experiment in store.experiments(run_id=refinement.run_id, step_idx=refinement.step_idx, kind="screen", limit=200):
        if experiment.arm == "original" and experiment.settings.get("refinement") == refinement.id:
            sources.append(experiment)
    # The task's check as the target is judged by forks' verdicts, never at the step: an
    # answer reached it when its fork passed; one sampled at the step alone says nothing.
    by_check = (refinement.target or {}).get("kind") == "check"
    good, bad = [], []
    for experiment in sources:
        samples = store.samples(experiment.id)
        for sample, outcome in zip(samples, experiment_outcomes(store, experiment, refinement.normalizers)):
            if outcome.is_error:
                continue
            if by_check:
                if sample.exclusion or sample.passed is None:
                    continue
                reached = bool(sample.passed)
            else:
                reached = target.matches(outcome)
            answer = _answer(store.sample_response(sample), outcome, source=f"{experiment.kind} {experiment.id[:8]}")
            (good if reached else bad).append(answer)
    own = recorded_outcome(store.response(checkpoint.step), checkpoint.step.error, refinement.normalizers)
    # A run whose verdict does not count (the skill changed during it, or it timed out) says
    # nothing of the check, as an excluded fork says nothing above.
    verdict = verdict_from(checkpoint.run.result)
    if not own.is_error and (not by_check or verdict is not None):
        answer = _answer(store.response(checkpoint.step), own, source="the recorded run")
        (good if (verdict == "passed" if by_check else target.matches(own)) else bad).insert(0, answer)
    sampled = len(good) + len(bad)
    if not sampled:
        summary = "no answers at this checkpoint yet: sample it to see what the model does here"
    elif sampled == 1:
        summary = f"the one answer seen at this checkpoint {'reached' if good else 'did not reach'} the target"
    elif not good:
        summary = f"none of the {sampled} answers seen at this checkpoint reached the target"
    else:
        summary = f"{len(good)} of the {sampled} answers seen at this checkpoint reached the target"
    return {
        "summary": summary,
        "reached": len(good),
        "of": sampled,
        "rate": compact.rate(_rate(len(good), sampled)) if sampled else None,
        "reaching": good[:EACH],
        "not_reaching": bad[:EACH],
    }


def _rate(hits: int, total: int) -> dict[str, Any]:
    from tuningfork.analysis.stats import rate

    return rate(hits, total).to_json()


#: What makes another run of the task comparable, step for step, with the refinement's own.
LIKE = "this harness, model and skill version"


def _divergence(store: Store, refinement: Any, run: Any) -> dict[str, Any] | None:
    """What runs of this task like this one did at this step, and how often each behaviour
    went on to pass. Like it means the same harness, model and skill version: another
    harness numbers its steps its own way, another model is other evidence, and runs given
    another wording could show a fixed wording's behaviour, or quote it."""
    from tuningfork.ops.inspect import divergence_report

    same = [
        found.id for found in store.runs(limit=200, task_id=run.task_id)
        if (found.skill_version, found.harness, found.model) == (run.skill_version, run.harness, run.model)
    ]
    if len(same) < 2:
        return {"summary": f"only {len(same)} run of this task with {LIKE}: nothing to compare"}
    report = compact.divergence(divergence_report(store, refs=same, normalizers=refinement.normalizers), all_steps=True)
    here = [step for step in report["steps"] if step.get("index") == refinement.step_idx]
    return {
        "summary": f"{report['runs']} runs of this task with {LIKE}",
        "split_at": report.get("divergence_index"),
        "at_this_step": here[0]["groups"] if here else None,
    }


def _task(store: Store, run: Any) -> dict[str, Any] | None:
    """The task's own criteria, as its spec states them: often the only place the fix shows."""
    from tuningfork.ops.refine import _launch_of
    from tuningfork.specs import describe_check

    launch = _launch_of(store, run)
    item = (launch.eval if launch is not None else None) or {}
    if not item:
        return None
    options = item.get("tuningfork") or {}
    return {
        "prompt": compact.clip(item.get("prompt"), 2000),
        "expected_output": compact.clip(item.get("expected_output"), 1000) or None,
        "expectations": item.get("expectations") or None,
        "check": describe_check(options.get("check")),
    }


def _tried(store: Store, refinement: Any) -> list[dict[str, Any]]:
    from tuningfork.ops.refine import view

    tried = []
    # Each proposal's lines from the two texts masked whole and alike (`refine.view`'s `masking`):
    # one line at a time, a private key's lines pass.
    for proposal in view(store, refinement.id, masking=store.masking)["proposals"]:
        changed = [line for hunk in proposal["diff"] for line in hunk["lines"] if line["op"] != " "]
        verdicts = [
            "survived" if each.get("survives") else "worse" if each.get("worse")
            else "not significant" if each.get("complete") else "unfinished"
            for each in proposal["screens"]
        ]
        tried.append({
            "proposal": proposal["id"][:8],
            "changed": [f"{line['op']} {line['text']}" for line in changed][:12],
            "rationale": compact.clip(proposal["proposers"][0].get("rationale"), 300),
            "screened": verdicts or None,
        })
    return tried


def build_packet(store: Store, reference: str, *, include_task: bool = True, keep: bool = True) -> Packet:
    """The packet for a refinement as its evidence stands now, kept by its digest — unless
    `keep` is false: a look at what a proposer would be shown leaves nothing behind."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.experiment import history_caveat, locate
    from tuningfork.ops import refine
    from tuningfork.ops.experiments import load
    from tuningfork.ops.timeline import step_detail
    from tuningfork.ops.tokens import step_tokens

    refinement = refine.get(store, reference)
    checkpoint = load(store, refinement.run_id, str(refinement.step_idx))
    baseline = refinement.baseline_text()
    target = target_from_spec(refinement.target)
    spans = locate(checkpoint.request, baseline)
    if not spans:
        raise OperationError("the skill is no longer in this checkpoint's context", field="refinement")
    try:
        tokens = step_tokens(store, refinement.run_id, str(refinement.step_idx)).to_json(hotspots=5)
        hotspots = compact.tokens(tokens, hotspots=5).get("hotspots") or None
    except OperationError:
        hotspots = None
    # The skill as the run was given it, masked whole before its lines are numbered, each keeping
    # its number as show_skill numbers it: line by line, a private key's lines pass.
    skill = store.masking.shown(baseline, lines=True)
    data = {
        "target": {"description": target.description},
        "skill": {
            "name": refinement.skill,
            "lines": compact.capped([f"{number}: {line}" for number, line in enumerate(skill.value.splitlines(), 1)],
                                    400),
            "in_context": [
                {"lines": [span.line_start, span.line_end], "form": span.form, "role": span.role}
                for span in spans
            ],
        },
        "step": compact.step(step_detail(store, refinement.run_id, refinement.step_idx).to_json()),
        "caveat": history_caveat(checkpoint),
        "answers_here": _continuations(store, refinement, checkpoint, target),
        "runs_here": _divergence(store, refinement, checkpoint.run),
        "hotspots": hotspots,
        "task": _task(store, checkpoint.run) if include_task else None,
        "included": {"task_criteria": include_task},
        "tried_before": _tried(store, refinement) or None,
    }
    # The task as its file wrote it, what was tried as each was given: all of it masked, of what
    # the skill's masks took too, before it is fitted — a cut would leave half a key.
    data = store.masking.shown({key: value for key, value in data.items() if value is not None}, also=skill.found)
    data = compact.fit(data.value, BUDGET)
    packet = Packet(sha=hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()[:16], data=data)
    if keep:
        packet.keep(store, refinement.folder)
    return packet


def read_packet(store: Store, reference: str, sha: str) -> Packet:
    from tuningfork.ops import refine

    refinement = refine.get(store, reference)
    path = refinement.folder / "packets" / f"{sha}.json"
    if not path.is_file():
        raise OperationError(f"no packet {sha!r} in refinement {refinement.id[:8]}", code="not_found")
    return Packet(sha=sha, data=json.loads(path.read_text(encoding="utf-8")))
