"""The store masks what it is given (`Store.masking`), once, with its one set of patterns: a recorded
step in every form it keeps, a sample and its tokens, an experiment, a job, a run's result, a launch,
a server — whoever writes it, the proxy, an operation or an import."""

from __future__ import annotations

import json

import pytest

from tuningfork.config import REDACTED, without_credentials
from tuningfork.store.masking import Masked
from tests.helpers import KEY, as_response, text_message
from tests.test_store import request_with

#: A backslash, as JSON's escapes begin.
BS = chr(92)

#: Masked by the patterns only where a password's name stands before it.
PASSWORD = "Tr0ub4dor3xyz"
NAMED = f"log in with password={PASSWORD} please"
#: A command whose second password only a second pass masks: masking is not idempotent.
COMMAND = "mysql -uroot -pAbcd1234X -pZq8xK3vN5wLp db"


def entries(*pieces: str) -> list[dict]:
    """A chat completion's logprobs, a token a piece."""
    made = []
    for number, piece in enumerate(pieces, start=100):
        token = {"id": number, "token": piece, "bytes": list(piece.encode()), "logprob": -0.1}
        made.append({**token, "top_logprobs": [token]})
    return made


class TestARecordedStep:
    def test_its_request_is_masked_and_its_masks_counted(self, store):
        step = store.record_step(request=request_with([text_message(NAMED)]))
        assert PASSWORD not in json.dumps(store.rebuild_request(step))
        assert step.params["masked"] == 1 and step.params["redaction"] == store.masking.digest

    def test_its_error_is_masked_of_what_its_request_masked(self, store):
        """A server's error quotes a value bare, where no pattern reads it as a secret."""
        step = store.record_step(request=request_with([text_message(NAMED)]),
                                 error=f"upstream_error: login refused for {PASSWORD}")
        assert step.error == f"upstream_error: login refused for {REDACTED}"

    def test_its_answer_and_the_harness_s_are_masked_of_what_its_request_masked(self, store):
        quoted = as_response(text_message(f"I logged in with {PASSWORD}."))
        step = store.record_step(request=request_with([text_message(NAMED)]), response=quoted,
                                 delivered_response=quoted)
        assert PASSWORD not in json.dumps(store.response(step))
        assert PASSWORD not in json.dumps(store.delivered_response(step))

    def test_its_parameters_are_masked(self, store):
        step = store.record_step(request=request_with([text_message("hi")]), params={"stop": [f"api_key={KEY}"]})
        assert step.params["stop"] == [f"api_key={REDACTED}"]

    def test_its_request_is_masked_by_the_patterns_alone(self, store):
        """A fork keys a call by its request masked so: the request's own masks are not carried into
        it, and a call found in the recording is its step."""
        from tuningfork.replay import build_table, key_of
        from tuningfork.wire import get_wire

        store.begin_run("asked", "t/1")
        asked = request_with([text_message(f"{NAMED}, then say {PASSWORD} back")])
        step = store.record_step(request=asked, response=as_response(text_message("ok")))
        said = store.rebuild_request(step)["messages"][0]["content"]
        assert said == f"log in with password={REDACTED} please, then say {PASSWORD} back"
        table = build_table(store, store.get_run(step.run_id), 0)
        assert key_of(table.text(asked, get_wire("openai-chat"))) == table.live_key

    def test_a_number_holding_a_value_a_mask_took_is_kept_whole(self, store):
        """Found in review: a password of digits, masked under its name, was masked inside a number of the
        same step too — the answer's `created`, a seed — which no longer read as JSON, and was dropped."""
        answer = {**as_response(text_message("connected")), "created": 1761312345}
        raw = json.dumps(answer).encode()
        step = store.record_step(request=request_with([text_message("mysql -uroot -p17613123 app")]),
                                 response=answer, params={"seed": 1761312345}, raw_response=raw, wire="openai-chat")
        assert store.response(step)["created"] == 1761312345 and step.params["seed"] == 1761312345
        assert store.blobs.get(step.raw_response_blob) == raw and "raw_redacted" not in step.params

    def test_bytes_that_cannot_be_read_are_masked_of_what_the_request_masked(self, store):
        """Unread, they hold no JSON a mask could break: a value a mask took is masked where they quote it."""
        step = store.record_step(request=request_with([text_message(NAMED)]),
                                 raw_response=f"<html>login refused for {PASSWORD}</html>".encode(),
                                 wire="openai-chat", error="tuningfork could not read the response")
        assert store.blobs.get(step.raw_response_blob) == f"<html>login refused for {REDACTED}</html>".encode()

    def test_json_written_inside_a_string_is_masked_where_the_value_stands(self, store):
        """Found in review: a call's arguments are a string holding JSON, and a value a mask took was
        masked in a number there, or cut an escape there, as it was one level up — and then, read as
        JSON and written again, the rest of it changed: a key named twice, `1e2`, the spacing."""

        def arguments(written):
            call = {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": written}}
            return as_response({"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls")

        def stored(step):
            return store.response(step)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]

        compact = '{"path":"cfg.json","note":"pin 17613123","retries":1e2,"path":"other.json","pin":17613123}'
        digits = store.record_step(request=request_with([text_message("mysql -uroot -p17613123 app")]),
                                   response=arguments(compact))
        assert stored(digits) == compact.replace("pin 17613123", f"pin {REDACTED}")
        # Each escape whole, where it starts or ends the value as where it stands inside it.
        escaped = ('{"content":"one\\nT0ub4dor3xyz","then":"nT0ub4dor3xyz","x":"Tr0ub4dor\\u00333xyz",'
                   '"y":"Tr0ub4dor33xy\\u007a"}')
        step = store.record_step(request=request_with([text_message("password=nT0ub4dor3xyz password=Tr0ub4dor33xyz")]),
                                 response=arguments(escaped))
        assert stored(step) == escaped.replace('"nT0ub4dor3xyz"', f'"{REDACTED}"').replace(
            "Tr0ub4dor\\u00333xyz", REDACTED).replace("Tr0ub4dor33xy\\u007a", REDACTED)

    def test_json_inside_a_string_is_read_as_json_reads_it(self, store):
        """A quote escaped inside one of its strings ends no string there, and a pair of `\\u` escapes
        is one character, as JSON reads them."""
        from tuningfork.store.masking import Masking

        value = "\U0001F600Tr0ub4dor3xyz"
        written = json.dumps({"said": f'one " then {value}'})  # ASCII: the emoji as two escapes
        assert BS + "ud83d" + BS + "ude00" in written
        masked = Masking().of({"arguments": written}, also=[value]).value["arguments"]
        assert masked == written.replace(json.dumps(value)[1:-1], REDACTED)

    def test_a_reply_written_as_json_keeps_its_layout(self, store):
        written = json.dumps({"user": "root", "note": f"pw {PASSWORD}"}, indent=2)
        step = store.record_step(request=request_with([text_message(NAMED)]), response=as_response(text_message(written)))
        assert store.response(step)["choices"][0]["message"]["content"] == written.replace(PASSWORD, REDACTED)

    def test_a_value_a_mask_took_is_masked_in_a_key_too(self, store):
        assert store.masking.of({PASSWORD: 1}, also=[PASSWORD]).value == {REDACTED: 1}

    def test_a_value_a_mask_took_is_masked_in_strings_as_they_read(self, store):
        """Never in the middle of an escape: a value beginning as an escape ends, in the JSON text."""
        step = store.record_step(request=request_with([text_message("password=nT0ub4dor3xyz")]),
                                 response=as_response(text_message("one\nT0ub4dor3xyz, then nT0ub4dor3xyz")))
        assert store.response(step)["choices"][0]["message"]["content"] == f"one\nT0ub4dor3xyz, then {REDACTED}"

    def test_the_harness_s_request_is_masked_by_the_patterns_alone(self, store):
        """A fork keys a call by its request masked so (`replay.masked_request`): the request's own
        masks are not carried into it."""
        harness = request_with([text_message(f"{NAMED}, then say {PASSWORD} back")])
        step = store.record_step(request=request_with([text_message(NAMED)]), harness_request=harness)
        said = store.harness_request(step)["messages"][0]["content"]
        assert said == f"log in with password={REDACTED} please, then say {PASSWORD} back"

    def test_its_tokens_are_masked_of_what_its_request_masked(self, store):
        """The model says the value back bare, a few characters a token: no pattern reads it there."""
        answer = as_response(text_message(f"I used {PASSWORD}"))
        step = store.record_step(request=request_with([text_message(NAMED)]), response=answer,
                                 tokens=entries("I", " used", " Tr0u", "b4do", "r3xyz"))
        (record,) = store.token_streams(step_id=step.id)
        kept = json.dumps(store.token_stream_data(record))
        assert "b4do" not in kept and "r3xyz" not in kept and '" used"' in kept

    def test_a_key_in_the_model_s_own_call_syntax_is_masked_in_its_tokens(self, store):
        """Found in the redaction audit: Gemma 4 writes a call's argument bare, `api_key:<|"|>…<|"|>`,
        where no pattern reads it as a key; its parsed arguments were masked, its tokens kept it."""
        key = "Zq8xK3vN5wLpQ7rT"
        call = {"id": "c1", "type": "function", "function": {"name": "login", "arguments": json.dumps({"api_key": key})}}
        answer = as_response({"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls")
        step = store.record_step(request=request_with([text_message("log in")]), response=answer,
                                 tokens=entries("<|tool_call>", "call:login{api_key:", '<|"|>', "Zq8x", "K3vN5w",
                                                "LpQ7rT", '<|"|>', "}<tool_call|>"))
        (record,) = store.token_streams(step_id=step.id)
        kept = json.dumps(store.token_stream_data(record))
        assert key not in json.dumps(store.response(step)) and "K3vN5w" not in kept and "api_key:" in kept

    def test_an_answer_sent_at_once_keeps_no_logprobs_that_spell_a_key(self, store):
        """Found in the redaction audit: only a stream's bytes were read back whole, and an answer
        sent at once kept the logprobs a harness asked for, a key spelled a token at a time."""
        pieces = ["sk", "-a1B2c3", "D4e5F6", "g7H8i9J0"]
        body = {"id": "x", "object": "chat.completion", "model": "m", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": KEY}, "finish_reason": "stop",
            "logprobs": {"content": entries(*pieces)}}]}
        raw = json.dumps(body).encode()
        step = store.record_step(request=request_with([text_message("hi")]), response=as_response(text_message(KEY)),
                                 raw_response=raw, streamed=False, wire="openai-chat")
        kept = store.blobs.get(step.raw_response_blob)
        assert step.params["raw_redacted"] == "rendered"
        assert b"D4e5F6" not in kept and b"logprobs" not in kept and REDACTED.encode() in kept

    def test_an_answer_without_a_secret_keeps_its_bytes(self, store):
        body = json.dumps({"id": "x", "object": "chat.completion", "model": "m", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop",
            "logprobs": {"content": entries("hel", "lo")}}]}).encode()
        step = store.record_step(request=request_with([text_message("hi")]), response=as_response(text_message("hello")),
                                 raw_response=body, streamed=False, wire="openai-chat")
        assert store.blobs.get(step.raw_response_blob) == body and "raw_redacted" not in step.params

    def test_an_edit_and_a_scenario_s_context_are_masked(self, store):
        step = store.record_step(
            request=request_with([text_message("hi")]),
            edits={"request": {"changed": [0], "typed": f"api_key={KEY}"}},
            context={"messages": 1, "note": f"api_key={KEY}"},
        )
        assert KEY not in json.dumps([step.edits, step.context])

    def test_a_label_an_import_gives_is_masked(self, store):
        from tuningfork.launches import Assignment

        assignment = Assignment(launch_id=None, run_id="r" * 32, idx=0, thread=0, label=f"imported: {NAMED}",
                                task_id=None, role=None)
        step = store.record_step(request=request_with([text_message("hi")]), assignment=assignment)
        assert store.get_run(step.run_id).label == f"imported: log in with password={REDACTED} please"


class TestASample:
    def experiment(self, store):
        return store.create_experiment(kind="resample", arm="original", run_id=None, step_idx=None, step_id=None,
                                       request={"messages": []}, settings={"n": 1})

    def test_its_answer_error_and_label_are_masked(self, store):
        sample = store.add_sample(self.experiment(store).id, 0, seed=1, response=as_response(text_message(NAMED)),
                                  outcome_kind="text", outcome_key="text:x", outcome_label=f"said {PASSWORD}",
                                  error=f"HTTP 500: refused {PASSWORD}")
        assert PASSWORD not in json.dumps([store.sample_response(sample), sample.error, sample.outcome_label])

    def test_an_answer_given_masked_is_kept_as_it_is(self, store):
        """Its outcome was read from it: masked again, a command's second password would fall too, and
        a report recomputing the outcome from the store would read another answer."""
        masked = store.masking.of(as_response(text_message(COMMAND)))
        sample = store.add_sample(self.experiment(store).id, 0, seed=1, response=masked,
                                  outcome_kind="text", outcome_key="text:x", outcome_label="x")
        assert store.sample_response(sample) == masked.value
        assert "-pZq8xK3vN5wLp" in json.dumps(masked.value)

    def test_the_token_it_was_forced_with_is_masked(self, store):
        sample = store.add_sample(self.experiment(store).id, 0, seed=1, response=None, outcome_kind="text",
                                  outcome_key="text:x", outcome_label="x", forced={"text": f"api_key={KEY}"})
        assert sample.forced == {"text": f"api_key={REDACTED}"}

    def test_a_stream_s_notes_are_masked(self, store):
        from tuningfork.analysis.tokens import from_entries

        record = store.add_token_stream(from_entries(entries("hi")), source="scored", notes={"said": f"api_key={KEY}"})
        assert store.token_stream(record.id).notes == {"said": f"api_key={REDACTED}"}

    def test_its_tokens_are_masked_of_what_its_answer_masked(self, store):
        from tuningfork.analysis.tokens import from_entries

        experiment = self.experiment(store)
        answered = store.masking.of(as_response(text_message(f"{NAMED} or {PASSWORD}")))
        sample = store.add_sample(experiment.id, 0, seed=1, response=answered, outcome_kind="text",
                                  outcome_key="text:x", outcome_label="x")
        record = store.add_token_stream(from_entries(entries("or", " Tr0u", "b4do", "r3xyz")), source="sampled",
                                        sample_id=sample.id, also=answered.found)
        assert "b4do" not in json.dumps(store.token_stream_data(record))


class TestTheRestOfTheStore:
    def test_an_experiment_s_request_and_setup_are_masked_and_its_test_kept_as_declared(self, store):
        """Its settings are the test declared before sampling, read again to judge it: kept as given."""
        experiment = store.create_experiment(
            kind="ab", arm="variant", run_id=None, step_idx=None, step_id=None,
            request=request_with([text_message(NAMED)]), settings={"target": {"kind": "regex", "value": COMMAND}},
            changes={"skill": f"api_key={KEY}"}, variant={"name": f"api_key={KEY}"}, notes=[f"api_key={KEY}"],
            base_url=f"http://tuningfork:{KEY}@127.0.0.1:8787",
        )
        store.add_experiment_notes(experiment.id, [f"later: api_key={KEY}"])
        found = store.get_experiment(experiment.id)
        kept = json.dumps([store.experiment_request(found), found.changes, found.variant, found.notes, found.base_url])
        assert KEY not in kept and PASSWORD not in kept
        assert found.settings == {"target": {"kind": "regex", "value": COMMAND}}

    def test_a_run_s_label_setup_and_result_are_masked(self, store):
        step = store.record_step(request=request_with([text_message("hi")]))
        store.annotate_run(step.run_id, label=f"run api_key={KEY}", parity={"skill": f"api_key={KEY}"},
                           result={"passed": False, "output": f"config.json holds api_key={KEY}"})
        run = store.get_run(step.run_id)
        assert KEY not in json.dumps([run.label, run.parity, run.result])

    def test_a_field_masked_where_it_was_made_is_kept_as_it_is(self, store):
        """A check's output, masked where the check ran, before it was cut: the rest of a run's
        result, and of its launch's exit, is masked as the store masks."""
        output = store.masking.value(COMMAND)
        assert "-pZq8xK3vN5wLp" in output  # masked once, the second password stands
        step = store.record_step(request=request_with([text_message("hi")]))
        result = {"check_output": output, "told": {"message": output}, "label": f"api_key={KEY}"}
        store.annotate_run(step.run_id, result=store.masking.of(result, masked=("check_output", "told")))
        kept = store.get_run(step.run_id).result
        assert kept["check_output"] == output and kept["told"] == {"message": output}
        assert KEY not in kept["label"] and list(kept) == list(result)  # in its own order
        launch = store.create_launch(kind="batch", eval={})
        store.end_launch(launch.id, exit=store.masking.of({"told": {"message": output}}, masked=("told",)))
        assert store.get_launch(launch.id).exit == {"told": {"message": output}}

    def test_a_launch_s_eval_and_variant_are_kept_as_given(self, store):
        """A fork runs them again as they were written: masked where they are shown or sent."""
        launch = store.create_launch(
            kind="batch", label=f"api_key={KEY}", eval={"prompt": f"use api_key={KEY}"},
            settings={"harness": {"mcp_servers": {"docs": f"https://tuningfork:{KEY}@docs.test"}}},
            variant={"variant_text": f"api_key={KEY}"},
        )
        store.note_launch(launch.id, stopped_at_skill={"said": f"api_key={KEY}"})
        store.end_launch(launch.id, status="ended", exit={"told": f"config.json holds api_key={KEY}"})
        found = store.get_launch(launch.id)
        assert KEY not in json.dumps([found.label, found.settings, found.exit])
        assert KEY in found.eval["prompt"] and KEY in found.variant["variant_text"]

    def test_a_job_s_parameters_and_what_it_made_are_masked(self, store):
        job = store.create_job(kind="check-trial", lane="interactive", params={"reply": f"api_key={KEY}"},
                               started_by={"via": "web", "page": f"api_key={KEY}"})
        store.update_job(job.id, plan={"prompt": f"api_key={KEY}"}, result={"output": f"api_key={KEY}"},
                         error={"message": f"api_key={KEY}"}, progress={"waiting": f"api_key={KEY}"})
        found = store.get_job(job.id)
        assert KEY not in json.dumps([found.params, found.started_by, found.plan, found.result, found.error,
                                      found.progress])

    def test_a_job_s_values_given_masked_are_kept_as_they_are(self, store):
        masked = store.masking.of({"command": COMMAND})
        job = store.create_job(kind="check-trial", lane="interactive", params=masked)
        store.update_job(job.id, result=masked)
        found = store.get_job(job.id)
        assert found.params == found.result == masked.value

    def test_a_server_is_kept_without_its_password(self, store):
        server_id = store.add_server({"upstream": f"http://tuningfork:{KEY}@127.0.0.1:8081", "models": ["m"]})
        store.set_server_checks(server_id, {"logprobs": {"ok": False, "answer": f"refused api_key={KEY}"}})
        assert KEY not in json.dumps([store.server(server_id), store.server_checks(server_id)])
        assert store.server(server_id)["upstream"] == f"http://{REDACTED}@127.0.0.1:8081"

    def test_a_scenario_s_context_is_kept_as_given(self, store):
        """It is what each of its runs is given: masked, every call would send the model another
        conversation than the scenario's (found in review)."""
        digest = store.put_given([text_message(NAMED)])
        assert PASSWORD in store.blobs.get(digest).decode()

    def test_a_label_queued_for_the_next_run_is_masked(self, store):
        store.label_next(f"tried with api_key={KEY}")
        assert store.pending_labels() == [f"tried with api_key={REDACTED}"]
        store.discard_pending_label(f"tried with api_key={KEY}")
        assert store.pending_labels() == []

    def test_what_json_cannot_write_is_masked_as_it_is_written(self, store):
        """Found in review: masking passed such a value by, and the job's record wrote it out as text."""
        from dataclasses import dataclass

        @dataclass
        class Plan:
            env: dict

        job = store.create_job(kind="x", lane="background", params={})
        store.update_job(job.id, plan=Plan(env={"TOKEN": f"api_key={KEY}"}))
        plan = store.get_job(job.id).plan
        assert "Plan(env={'TOKEN': " in plan and REDACTED in plan and KEY not in plan  # read, as it is written


class TestAFileKeptBesideIt:
    """A file made from what was recorded — a chain's record, an imported chat and its skills — is
    masked as the store masks what it keeps (`Store.keep_file`)."""

    def test_it_is_masked_as_the_store_masks(self, store, tmp_path):
        record = {"said": f"config.json holds api_key={KEY}", "n": 3}
        path = store.keep_file(tmp_path / "kept" / "record.json", record, indent=1)
        assert json.loads(path.read_text(encoding="utf-8")) == store.masking.value(record)
        text = store.keep_file(tmp_path / "kept" / "SKILL.md", f"Use api_key={KEY}\n")
        assert text.read_text(encoding="utf-8") == f"Use api_key={REDACTED}\n"
        assert KEY not in path.read_text(encoding="utf-8") + text.read_text(encoding="utf-8")

    def test_a_value_masked_already_is_kept_as_it_is(self, store, tmp_path):
        once = store.masking.of({"told": COMMAND})
        path = store.keep_file(tmp_path / "record.json", once)
        assert json.loads(path.read_text(encoding="utf-8")) == once.value
        assert "-pZq8xK3vN5wLp" in path.read_text(encoding="utf-8")  # masked again, it would fall too

    def test_it_is_written_whole_or_not_at_all(self, store, tmp_path, monkeypatch):
        import os
        from pathlib import Path

        path = store.keep_file(tmp_path / "kept" / "record.json", {"links": 1})
        replace = os.replace

        def interrupted(source, target, **options):
            # The disk full only for this file: whatever else replaces a file meanwhile goes on.
            if Path(target) == path:
                raise OSError("the disk is full")
            return replace(source, target, **options)

        monkeypatch.setattr(os, "replace", interrupted)
        with pytest.raises(OSError):
            store.keep_file(path, {"links": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"links": 1}
        assert [found.name for found in path.parent.iterdir()] == ["record.json"]


class TestALinedText:
    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", chr(0x2028)])
    def test_each_line_keeps_its_number_where_a_mask_took_several(self, store, newline):
        """A private key's lines masked as one, and every line after it kept at its number, however the
        file breaks its lines (`Masking.text`'s `lines`): found in review, a carriage return alone, or
        U+2028, still moved them."""
        body = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"
        lines = ["# Deploy", "-----BEGIN OPENSSH PRIVATE KEY-----", body, "QyNTUxOQAAACDx2Jx7dGgHJ1s0kpk8q9fG1hX",
                 "-----END OPENSSH PRIVATE KEY-----", "Then run the tests."]
        text = newline.join(lines) + newline
        masked = store.masking.text(text, lines=True).value
        assert body not in masked and REDACTED in masked
        assert len(masked.splitlines()) == len(text.splitlines())
        assert masked.splitlines()[-1] == "Then run the tests."
        assert len(store.masking.text(text).value.splitlines()) < len(lines)  # without, they move


#: A private key in a skill: its lines read as no key one at a time.
PRIVATE = ["-----BEGIN OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW",
           "QyNTUxOQAAACDx2Jx7dGgHJ1s0kpk8q9fG1hX", "-----END OPENSSH PRIVATE KEY-----"]


class TestWhatIsShown:
    """What is shown outside tuningfork — to an agent over MCP, to a proposing model — of a value,
    whatever the store keeps of it (`Masking.shown`)."""

    def test_every_text_is_masked_a_key_s_too_and_its_numbers_and_shape_kept(self, store):
        shown = store.masking.shown({"said": f"use {KEY}", KEY: [3, f"again {KEY}", None], "count": 3})
        assert shown.value == {"said": f"use {REDACTED}", REDACTED: [3, f"again {REDACTED}", None], "count": 3}
        assert shown.made == 3

    def test_a_value_one_text_names_is_masked_where_another_says_it_bare(self, store):
        """A password named in one field, and said bare in another: as a step's answer is masked of
        what its request's masks took."""
        shown = store.masking.shown({"check": NAMED, "lines": [f"it was {PASSWORD}"]}).value
        assert PASSWORD not in json.dumps(shown) and shown["lines"] == [f"it was {REDACTED}"]
        assert store.masking.shown(f"{NAMED}; it was {PASSWORD}").value.count(REDACTED) == 2

    def test_two_keys_a_mask_makes_one_are_both_kept(self, store):
        assert len(store.masking.shown({KEY: 1, "sk-z9Y8x7W6v5U4t3S2r1Q0": 2}).value) == 2

    def test_a_text_shown_whole_keeps_its_lines_where_line_by_line_a_private_key_passes(self, store):
        """A text a view splits — numbered lines, pages, a diff — is shown whole first: line by line,
        each of a private key's lines reads as no key."""
        text = "\n".join(["# Deploy", *PRIVATE, "Then run the tests."]) + "\n"
        whole = store.masking.shown(text, lines=True).value
        assert PRIVATE[1] not in whole and PRIVATE[2] not in whole
        assert len(whole.splitlines()) == len(text.splitlines()) and whole.splitlines()[-1] == "Then run the tests."
        assert PRIVATE[1] in store.masking.shown(text.splitlines()).value  # what masking line by line misses

    def test_what_the_store_kept_masked_is_shown_as_it_is(self, store):
        kept = store.masking.value(f"{NAMED}, then {KEY}")
        assert store.masking.shown(kept).value == kept

    def test_a_value_json_cannot_write_is_shown_as_the_text_it_is_written_as(self, store):
        """An answer is written with `default=str`: found in review, anything but text, a mapping or
        a list was let through unread."""
        from pathlib import PurePosixPath

        shown = store.masking.shown({"path": PurePosixPath(f"/keys/{KEY}.txt"), "count": 3, "done": True}).value
        assert shown == {"path": f"/keys/{REDACTED}.txt", "count": 3, "done": True}

    def test_texts_side_by_side_show_what_they_share_as_either_shows_it(self, store):
        """An edit can take away what a key's pattern needs — a private key's first line — and leave
        its body to read as no key (found in review): each line one text holds is shown as the other
        shows it, wherever it stands, the first's and the later's alike — a run can give a block its
        first line as well as take it (found in review); each keeps its number."""
        original = "\n".join(["# Deploy", *PRIVATE, "Then run the tests."]) + "\n"
        edited = original.replace(PRIVATE[0], "The deploy key is kept in the vault.")
        inserted = original.replace(PRIVATE[0] + "\n", PRIVATE[0] + "\n(the deploy key)\n")
        for other in (edited, inserted):
            assert PRIVATE[1] in store.masking.shown(other, lines=True).value  # alone, the body reads as no key
            was, now = store.masking.alike([original, other])
            assert PRIVATE[1] not in now and PRIVATE[2] not in now and "the deploy key" in now.lower()
            assert len(now.splitlines()) == len(other.splitlines()) and len(was.splitlines()) == len(original.splitlines())
            was, now = store.masking.alike([other, original])
            assert PRIVATE[1] not in was + now and PRIVATE[2] not in was + now
        named = f"{NAMED}\nThen: it was {PASSWORD}.\n"
        assert PASSWORD not in store.masking.alike([named, f"Then: it was {PASSWORD}.\n"])[1]
        assert PASSWORD not in store.masking.alike([f"Then: it was {PASSWORD}.\n", named])[0]

    def test_a_line_that_keeps_a_value_and_loses_what_made_it_one_is_masked_there(self, store):
        """A password's pattern reads the command before it: an edit of that command alone leaves the
        password to read as no key on its own line (found in review). What the two texts share is
        masked as either masks it."""
        original = "Call it: curl -u admin:Xy7pQ2 https://api.test\n"
        edited = original.replace("curl -u admin:", "http -a admin:")
        assert "Xy7pQ2" in store.masking.shown(edited).value  # alone, the value reads as no key
        was, now = store.masking.alike([original, edited])
        assert "Xy7pQ2" not in was + now and now.startswith("Call it: http -a admin:")

    def test_letters_a_changed_line_shares_with_a_key_by_chance_are_shown(self, store):
        """Found in review: a key's line rewritten as a word read `rot<redacted>t<redacted>d`, the
        letters the two happened to share masked. A run of shared characters carries a mask it holds
        whole, or eight characters of one."""
        was, now = store.masking.alike([f"# Deploy\n{KEY}\n", "# Deploy\nrotated\n"])
        assert KEY not in was and now == "# Deploy\nrotated\n"
        # A private key's first line taken, a character of its body changed: what it shares is the key.
        original = "\n".join(["# Deploy", *PRIVATE, "Then run the tests."]) + "\n"
        edited = original.replace(PRIVATE[0] + "\n", "").replace(PRIVATE[1], PRIVATE[1][:-1] + "!")
        assert PRIVATE[1][:-1] not in store.masking.alike([original, edited])[1]

    def test_each_line_a_mask_covers_reads_it(self, store):
        """A private key's lines, masked, each read `<redacted>`: shown blank, they read as lines with
        nothing on them, which an edit could take (found in review)."""
        text = "\n".join(["# Deploy", *PRIVATE, "Then run the tests."]) + "\n"
        lines = store.masking.shown(text, lines=True).value.splitlines()
        assert lines == ["# Deploy", PRIVATE[0] + REDACTED, REDACTED, REDACTED, PRIVATE[-1], "Then run the tests."]

    def test_a_value_an_edit_leaves_to_read_as_no_key_is_found(self, store):
        """An edit can take away what a key's pattern reads — the `-u` before a password, `password=`,
        a private key's first line — and leave the key bare wherever the text is next kept or shown on
        its own (`Masking.bared`); one taken out with its line, or left as it was, is not."""
        before = f"Call it: curl -u admin:Xy7pQ2 https://api.test\n{NAMED}\n" + "\n".join(PRIVATE) + "\n"
        assert store.masking.bared(before, before.replace("Call it", "Fetch it")) == []
        assert store.masking.bared(before, before.replace("curl -u admin:", "http -a admin:")) == ["Xy7pQ2"]
        assert store.masking.bared(before, before.replace("password=", "the password ")) == [PASSWORD]
        assert len(store.masking.bared(before, before.replace(PRIVATE[0], PRIVATE[0] + " (rotated)"))) == 1
        assert store.masking.bared(before, before.replace(f"{NAMED}\n", "")) == []
        bare = f"Say Xy7pQ2 if asked.\nCall it: curl -u admin:Xy7pQ2 https://api.test\n"  # said bare already
        assert store.masking.bared(bare, bare.replace("Say", "Answer")) == []

    def test_where_a_text_is_masked_is_found_of_what_its_own_masks_took(self, store):
        text = f"{NAMED}\nit was {PASSWORD}\n"
        assert [text[start:end] for start, end in store.masking.spans(text)] == [PASSWORD, PASSWORD]


class TestAnAddressKeptAndShown:
    @pytest.mark.parametrize("url, kept", [
        (f"http://tuningfork:{KEY}@127.0.0.1:8081", f"http://{REDACTED}@127.0.0.1:8081"),
        (f"https://{KEY}@api.test/v1", f"https://{REDACTED}@api.test/v1"),
        ("http://127.0.0.1:8081", "http://127.0.0.1:8081"),
        ("http://127.0.0.1:8081/v1?key=1", "http://127.0.0.1:8081/v1?key=1"),
    ])
    def test_what_stands_before_its_host_is_masked_whole(self, url, kept):
        assert without_credentials(url) == kept


class TestOnePatternSet:
    def test_a_store_s_own_patterns_mask_what_it_keeps_and_predict_where_a_call_lands(self, tmp_path):
        """The breakpoint engine and a launch's threads place a call in the tree as the store will
        store it: masked by the store's own patterns, not the defaults."""
        from tuningfork.breakpoints.engine import BreakpointEngine
        from tuningfork.launches import request_path
        from tuningfork.store.db import Store

        store = Store(tmp_path / "own", patterns=(r"(?P<secret>zebra-\d+)",))
        try:
            request = request_with([text_message("the code is zebra-42")])
            step = store.record_step(request=request)
            assert store.rebuild_request(step)["messages"][0]["content"] == f"the code is {REDACTED}"
            assert request_path(request, None, store.masking)[-1] == step.request_node_id
            assert BreakpointEngine(store).context_for(request).path[-1] == step.request_node_id
        finally:
            store.close()

    def test_a_launch_places_its_calls_as_the_store_masks_them(self, tmp_path):
        """A harness that sends a key again, changed, still continues its own call: masked, the two
        are one conversation, as the store keeps them."""
        from tuningfork.proxy.launches import LaunchState
        from tuningfork.store.db import Store
        from tuningfork.wire import get_wire

        store = Store(tmp_path / "own", patterns=(r"(?P<secret>zebra-\d+)",))
        try:
            state = LaunchState(store.create_launch(kind="batch", label="t#1.0", task_id="t/1"), store)
            wire = get_wire("openai-chat")
            asked = {"role": "user", "content": "the code is zebra-41"}
            first = state.assign(request_with([asked]), wire)
            again = {**asked, "content": "the code is zebra-42"}
            later = state.assign(request_with([again, {"role": "assistant", "content": "ok"},
                                               {"role": "user", "content": "go on"}]), wire)
            assert (first.thread, later.thread, later.idx) == (0, 0, 1)
        finally:
            store.close()

    def test_a_fork_keys_its_calls_as_the_store_masked_its_steps(self, tmp_path):
        from tuningfork.replay import build_table, key_of
        from tuningfork.store.db import Store
        from tuningfork.wire import get_wire

        store = Store(tmp_path / "own", patterns=(r"(?P<secret>zebra-\d+)",))
        try:
            store.begin_run("own", "t/1")
            call = request_with([text_message("the code is zebra-42")])
            store.record_step(request=call, response=as_response(text_message("ok")))
            table = build_table(store, store.get_run("own"), 0)
            assert key_of(table.text(call, get_wire("openai-chat"))) == table.live_key
        finally:
            store.close()

    def test_a_fork_refuses_a_step_whose_edited_answer_holds_a_mask(self, store):
        """Found in review: the answer a person edited at a breakpoint is what a fork serves, and its
        masks were served as they were stored, nothing said."""
        from tuningfork.replay import replay_problems

        store.begin_run("edited", "t/1")
        answer = as_response(text_message("ok"))
        delivered = text_message(f"ok, with {PASSWORD}")
        store.record_step(request=request_with([text_message(NAMED)]), response=answer,
                          delivered_response=as_response(delivered),
                          raw_response=json.dumps(answer).encode(), wire="openai-chat")
        store.record_step(request=request_with([text_message(NAMED), delivered, {"role": "user", "content": "go on"}]),
                          response=answer)
        (problem,) = replay_problems(store, store.get_run("edited"), 1)
        assert "edited answer had a secret redacted" in problem

    def test_a_masked_value_passes_through_unmasked_again(self):
        from tuningfork.store.masking import Masking

        once = Masking().of({"command": COMMAND})
        assert Masking().of(once) is once and Masking().value(Masked(None)) is None


class TestAPartOfAText:
    def test_an_edge_a_key_falls_inside_moves_past_its_mask(self, store):
        """Read on to either side, the key an edge falls inside is found whole: what is given holds
        neither its end nor half of its mask (found in review, where a long output was given from a
        place inside a key)."""
        text = f"start {KEY} middle {KEY} end"
        first, second = text.index(KEY), text.rindex(KEY)
        given, start, end = store.masking.part(text, first + 5, second + 5)
        assert given == " middle " and (start, end) == (first + len(KEY), second)
        assert store.masking.part(text, 0, 5) == ("start", 0, 5)
        assert store.masking.part(text) == (store.masking.text(text).value, 0, len(text))


class TestTheEscapedReading:
    """A text as a JSON string writes it, read for keys a pattern finds there alone (`redact_spans`):
    each character of it mapped back to the one it came from."""

    @staticmethod
    def one_by_one(text: str) -> tuple[str, list[int]]:
        """How it was read before: a place kept for each character."""
        short = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
        out, origin = [], []
        for index, char in enumerate(text):
            piece = short.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 else char)
            out.append(piece)
            origin.extend([index] * len(piece))
        return "".join(out), origin

    def test_each_character_maps_back_to_the_one_it_came_from(self):
        import random

        from tuningfork.config import _json_escaped

        draw = random.Random(7)
        alphabet = 'ab"\\\n\t\r\x01\x1f\x7f/é  '
        for _ in range(3000):
            text = "".join(draw.choices(alphabet, k=draw.randint(0, 24)))
            escaped, origin = _json_escaped(text)
            expected, places = self.one_by_one(text)
            assert escaped == expected and [origin(index) for index in range(len(escaped))] == places, repr(text)

    def test_a_long_text_costs_about_its_own_size(self):
        """A place for each character took some forty bytes for each byte of a long text (found in
        review): a check's output of a few hundred MB would have taken GB in the proxy."""
        import tracemalloc

        from tuningfork.config import _json_escaped

        text = ("collected 12 items\n" + "x" * 60 + "\n") * 15000
        tracemalloc.start()
        try:
            _json_escaped(text)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # A place for each character, even in a compact array, would take some ten times its size.
        assert peak < 4 * len(text), peak


class TestOneBoundary:
    #: Who masks with the patterns themselves, or writes a blob, and why: everything else hands the
    #: store what it has (or asks the store's own `masking` for a value it must read as stored).
    ALLOWED = {
        "config.py": "the patterns, and masking itself",
        "store/masking.py": "the store's masking",
        "store/db.py": "the store",
        "store/blobs.py": "the blob store",
        "analysis/tokens.py": "a token stream masked token-aware, which the store's masking calls",
        "replay.py": "a fork's call keyed as the store masked its step",
        "experiment/skill.py": "a skill found as storage masked it, and a wording sent masked alike",
        "launches.py": "a call's path predicted as the store masks, where no store is given",
        "breakpoints/engine.py": "a call's place predicted as the store masks, where no store is given",
    }
    MASKERS = {"redact", "redact_json", "redact_json_counted", "redact_json_found", "redact_stream",
               "redact_spans", "text_spans", "_masking", "Masking", "bytes_without_secrets"}

    def test_nothing_else_masks_or_writes_a_blob_itself(self):
        """Masking is not idempotent, and a fork matches a call masked once against a step stored
        masked once: what is masked twice, or not by the store, parts from it."""
        import ast
        from pathlib import Path

        import tuningfork

        root = Path(tuningfork.__file__).parent
        found = set()
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                blob = (isinstance(func, ast.Attribute) and func.attr in ("put", "put_json")
                        and isinstance(func.value, ast.Attribute) and func.value.attr == "blobs")
                if name in self.MASKERS or blob:
                    found.add(path.relative_to(root).as_posix())
        assert found <= set(self.ALLOWED), sorted(found - set(self.ALLOWED))
