"""Blob storage and redaction.

Redaction is security-relevant and runs over every recorded payload, so its edge cases are
pinned here: it must mask secrets, preserve surrounding structure, and leave valid JSON
valid — a recording that will not parse is a recording that cannot be replayed.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.config import DEFAULT_REDACTIONS, REDACTED, redact, redact_json
from tuningfork.store.blobs import BlobStore


class TestBlobStore:
    def test_identical_content_is_stored_once(self, tmp_path):
        store = BlobStore(tmp_path)

        first = store.put(b"same bytes")
        second = store.put(b"same bytes")

        assert first == second
        assert len(list(tmp_path.rglob("*"))) == 2  # one shard dir, one file

    def test_roundtrip(self, tmp_path):
        store = BlobStore(tmp_path)
        digest = store.put_json({"b": 1, "a": 2})

        assert store.get_json(digest) == {"b": 1, "a": 2}
        assert digest in store

    def test_json_keeps_its_key_order(self, tmp_path):
        """Order is part of what was sent: a server renders tool schemas in the order it
        receives them. (Tree identity uses its own canonical hash, which ignores it.)"""
        store = BlobStore(tmp_path)
        digest = store.put_json({"b": 2, "a": 1})
        assert list(store.get_json(digest)) == ["b", "a"]
        assert digest != store.put_json({"a": 1, "b": 2})

    def test_unknown_digest(self, tmp_path):
        assert "0" * 64 not in BlobStore(tmp_path)

    def test_no_temp_files_are_left_behind(self, tmp_path):
        store = BlobStore(tmp_path)
        store.put(b"x")
        assert not list(tmp_path.rglob("*.tmp"))


class TestRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwx",
            "ghp_abcdefghijklmnopqrstuvwxyz1234",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234567890-abcdefghij",
        ],
    )
    def test_known_secret_shapes_are_masked(self, secret):
        assert secret not in redact(f"token is {secret} ok")

    def test_bearer_header_keeps_its_keyword(self):
        result = redact("Authorization: Bearer a1B2c3D4e5F6g7H8i9J0k1L2")

        assert "Bearer" in result
        assert "a1B2c3D4e5F6g7H8i9J0k1L2" not in result
        assert REDACTED in result

    def test_assignment_keeps_the_field_name(self):
        result = redact('{"api_key": "a1B2c3D4e5F6g7H8"}')

        assert "api_key" in result  # a redacted recording stays readable
        assert "a1B2c3D4e5F6g7H8" not in result

    def test_redacted_json_still_parses(self):
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "key sk-abcdefghijklmnopqrstuvwx"}],
        }

        restored = json.loads(redact(json.dumps(payload)))

        assert restored["model"] == "m"
        assert "sk-" not in restored["messages"][0]["content"]

    def test_ordinary_text_is_untouched(self):
        text = "The model wrote a file called report.md and ran the tests."
        assert redact(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "This is the task-management-system project.",
            "pip install flask-sqlalchemy-utilities-extra",
            "the cupbearer abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_words_that_merely_contain_a_key_prefix_are_untouched(self, text):
        """Found in review: "task-management-system" was stored as "ta<redacted>", and the
        run's later calls no longer matched where breakpoints predicted them."""
        assert redact(text) == text

    def test_a_key_after_punctuation_is_still_masked(self):
        for text in ('"sk-abcdefghijklmnopqrstuvwx"', "KEY=sk-abcdefghijklmnopqrstuvwx"):
            assert "abcdefghijklmnop" not in redact(text)

    def test_every_default_pattern_compiles(self):
        import re

        for pattern in DEFAULT_REDACTIONS:
            re.compile(pattern)

    def test_multiple_secrets_in_one_payload(self):
        text = "a=sk-abcdefghijklmnopqrstuvwx b=ghp_abcdefghijklmnopqrstuvwxyz1234"
        result = redact(text)

        assert "sk-abcdefghij" not in result
        assert "ghp_abcdefghij" not in result
        assert result.count(REDACTED) == 2


class TestRedactionReadsJsonAsItReads:
    """Found in the 0.5.0 review: storage matched the patterns against JSON as it is written,
    where a newline is a backslash and an `n`. A key on a line of its own then followed a
    letter, which an `sk-` key must not, and was stored in clear in every harness's payloads.
    A call's arguments, or a tool's JSON result inside a message, escape their quotes, which
    `api_key: …` never matched."""

    KEY = "sk-a1B2c3D4e5F6g7H8i9J0"

    def stored(self, payload):
        return json.dumps(redact_json(payload), ensure_ascii=False)

    @pytest.mark.parametrize("before", ["\n", "\t", "\r\n"])
    def test_a_key_on_a_line_of_its_own(self, before):
        assert self.KEY not in self.stored({"role": "tool", "content": "first line" + before + self.KEY})

    def test_a_key_after_a_newline_in_a_calls_arguments(self):
        arguments = json.dumps({"path": ".env", "content": "# keys\n" + self.KEY + "\n"})
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": arguments}}]}
        assert self.KEY not in self.stored({"messages": [message]})

    def test_a_key_after_a_newline_in_a_tools_json_result(self):
        result = json.dumps({"output": "export done\n" + self.KEY, "exit_code": 0})
        assert self.KEY not in self.stored({"messages": [{"role": "tool", "content": result}]})

    def test_a_key_in_a_json_file_a_tool_showed(self):
        # Hermes's terminal result is JSON, and the file it shows is JSON too: the key's newline
        # is escaped three times in the stored request.
        shown = json.dumps({"note": "keys below\n" + self.KEY})
        result = json.dumps({"output": shown, "exit_code": 0})
        assert self.KEY not in self.stored({"messages": [{"role": "tool", "content": result}]})

    def test_an_assignment_inside_a_json_string(self):
        result = json.dumps({"api_key": "a1B2c3D4e5F6g7H8"})
        stored = self.stored({"messages": [{"role": "tool", "content": result}]})
        assert "a1B2c3D4e5F6g7H8" not in stored
        assert "api_key" in stored

    def test_a_key_written_with_an_escape_inside_it_is_masked_whole(self):
        # The model's own JSON, as a Qwen3 answer or a streamed Messages call keeps it.
        escape = "\\" + "u0044"  # a D, as JSON may write one
        arguments = '{"command":"echo sk-a1B2c3' + escape + 'e5F6g7H8i9J0K1"}'
        stored = self.stored({"tool_calls": [{"function": {"name": "terminal", "arguments": arguments}}]})
        assert "a1B2c3" not in stored and "e5F6g7H8i9J0K1" not in stored

    def test_a_stream_kept_as_text(self):
        # The recorder keeps a streamed answer's bytes, redacted as text.
        delta = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "here:\n" + self.KEY}}
        text = "event: content_block_delta\ndata: " + json.dumps(delta) + "\n\n"
        assert self.KEY not in redact(text)

    def test_the_rest_is_kept_and_still_parses(self):
        content = "line\n" + self.KEY + '\nnext line "quoted" é'
        payload = {"tool_calls": [{"function": {"arguments": json.dumps({"content": content})}}]}
        arguments = redact_json(payload)["tool_calls"][0]["function"]["arguments"]
        assert json.loads(arguments)["content"] == content.replace(self.KEY, REDACTED)

    def test_text_without_a_secret_is_kept_byte_for_byte(self):
        text = json.dumps({"content": json.dumps({"output": 'a\nb\t"c" \\ é'})})
        assert redact(text) == text

    def test_a_word_on_a_line_of_its_own_is_still_a_word(self):
        text = json.dumps({"content": "plan:\ntask-management-system"})
        assert redact(text) == text


def as_a_tool_result(text: str) -> dict:
    return {"messages": [{"role": "tool", "tool_call_id": "c1", "content": text}]}


class TestRedactionMasksKeysNotCode:
    """Found in the redaction audit. The generic pattern took code for keys: `token =
    tokenizer.encode(…)` was stored as `token = <redacted>(…)`, in one source file in six of
    Hermes's own an agent read — and a step whose context holds a mask can no longer be scored
    or probed. It missed most key formats but `sk-`, `ghp_`, `AKIA` and `xox`. And it masked a
    long number under a token's name bare, which broke the JSON: the request was dropped."""

    @pytest.mark.parametrize(
        "text",
        [
            "ids = tokenizer.encode(text)\ntoken = tokenizer.encode(text)[0]",
            "password = get_password()",
            "secret = settings.SECRET_KEY",
            'api_key = os.environ.get("OPENAI_API_KEY")',
            "const apiKey = process.env.OPENAI_API_KEY;",
            'payload = {"api_key": self.api_key, "model": model}',
            'if not api_key:\n    logger.warning("no key set")',
            "last refreshed token:\n2024-10-23T12:00:00Z ok",
            "Mount disk-A1B2C3D4E5F6G7H8I9 at /data",
            "client = OpenAI(api_key=resolved_api_key, base_url=_RUNTIME_BASE_URL)",
            "headers: { Authorization: `Bearer ${token}` },",
            "The API uses Bearer token-based-authentication for every request.",
            "Set the header to Bearer YOUR_ACCESS_TOKEN_HERE before calling.",
            '<div class="sk-toggleable__content fitted"></div>',
            "ssh-keygen -t sk-ecdsa-sha2-nistp256@openssh.com",
            "pip install sk-learn-compatible-wrapper==0.3.1",
            '{"next_page_token": "CiAKGjBpNDd2Nmp2Zml6cTJ5aDhnNGQ0Mm1nYxIA", "items": []}',
            '{"token": "Initialization", "logprob": -0.012}',
            'tokenizer.pad_token = "<|endoftext|>"',
            "def sign(private_key: PKCS7PrivateKeyTypes, data: bytes) -> bytes:",
            "SERVER_AUTH = '1.3.6.1.5.5.7.3.1'",
            "ssh_key: ~/.ssh/id_ed25519",
            'ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"',
            "POSTGRES_PASSWORD: postgres",
            "token: ${{ secrets.GITHUB_TOKEN }}",
            "HTTP Basic Authentication.",
        ],
    )
    def test_code_and_words_are_kept(self, text):
        payload = as_a_tool_result(text)
        assert redact_json(payload) == payload

    # Each fake key below is written in two pieces, which Python joins, so that no secret scanner
    # reads the test for a leak: GitHub's push protection refused them whole.
    @pytest.mark.parametrize(
        "text, secret",
        [
            ("here: sk-" "proj-Q2pA9xYzLmNoPqRsTuVwXyZ012345678abcdEFGH", "Q2pA9xYzLmNoPqRs"),
            ("sk-" "ant-api03-Xy12AbCdEfGhIjKlMnOpQrStUvWx-YzAbCdEf12_3456789", "Xy12AbCdEfGhIjKl"),
            ("sk-" "or-v1-3f2a9b8c7d6e5f40312a9b8c7d6e5f40312a9b8c7d6e5f40312a9b8c7d6e5f40", "3f2a9b8c7d6e5f40"),
            ("gsk" "_8xHq2LmN4pQr6StU8vWx0YzA2bCd4EfG6hIj8KlM0nOp2QrS4tUv6WxY", "8xHq2LmN4pQr6StU"),
            ("hf" "_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "AbCdEfGhIjKlMnOp"),
            ("curl 'https://example.googleapis.com/v1/models?key=AI" "zaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q'",
             "SyA1b2C3d4E5f6G7"),
            ("git clone https://oauth2:glpat" "-8E2dXy3Fz4Gh5Jk6Lm7N@gitlab.com/group/repo.git", "8E2dXy3Fz4Gh5Jk6"),
            ("github" "_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV012345",
             "11ABCDEFG0123456789"),
            ("STRIPE_SECRET_KEY=sk" "_live_51HxYzAbCdEfGhIjKlMnOpQrStUvWx0123456789", "51HxYzAbCdEfGhIj"),
            ("rk" "_live_51HxYzAbCdEfGhIjKlMnOpQrStUvWx0123456789", "51HxYzAbCdEfGhIj"),
            ("whsec" "_AbCdEfGhIjKlMnOpQrStUvWx0123456789", "AbCdEfGhIjKlMnOp"),
            ("npm" "_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "AbCdEfGhIjKlMnOp"),
            ("xapp" "-1-A0123456789-0123456789012-abcdef0123456789abcdef0123456789", "abcdef0123456789"),
            ("aws: AS" "IAQQMWPXB2EZNTWSQW", "QQMWPXB2EZNTWSQW"),
            ("aws configure set aws_secret_access_key wJalr" "XUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "XUtnFEMI/K7MDENG"),
            ("token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
             "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "SflKxwRJSMeKKF2QT4fw"),
            ("TELEGRAM_BOT_TOKEN=7167356528:AA" "ptfRBJQdXyZ0123456789abcdefghijkl", "ptfRBJQdXyZ0123456789"),
            ("client.run('MTk4NjIy" "NDgzNDcxOTI1MjQ4.Gh5Jk6.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789')",
             "AbCdEfGhIjKlMnOpQr"),
            ("https://hooks.slack.com/services/T01ABCDEF/B02GHIJKL/" "AbCdEfGhIjKlMnOpQrStUvWx", "AbCdEfGhIjKlMnOp"),
            ("-----BEGIN OPENSSH PRIVATE " "KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n"
             "QyNTUxOQAAACDxAbCdEfGh\n-----END OPENSSH PRIVATE " "KEY-----", "QyNTUxOQAAACDx"),
            ("DATABASE_URL=postgres://app:Wq8zK3vN5x@db.internal:5432/app", "Wq8zK3vN5x"),
            ("Authorization: Basic YWRtaW46ZzdjMDZ1Wk1NVCFqUDdNYw==", "YWRtaW46ZzdjMDZ1"),
            ("curl -u admin:Hq7Zx2Lp9V https://api.example.com", "Hq7Zx2Lp9V"),
            ("mysql -u root -pK8mQ2xZ7vL app", "K8mQ2xZ7vL"),
            ('requests.get(url, auth=("admin", "g7c06uZMMT!jP7Mc"))', "g7c06uZMMT"),
            ("PASSWORD=YcZgAG8y", "YcZgAG8y"),
            ("DB_PASS=z7QJRek12Xn4h7II6xFJqifd", "z7QJRek12Xn4h7II"),
            ("AccountName=demo01;AccountKey=dvTUsWKtuqAb12EfGhIjKlMnOp+/QrStUvWx34==;EndpointSuffix=x",
             "dvTUsWKtuqAb12Ef"),
            ("SECRET_KEY=django-insecure-5l-kb$6ahtb3nenc$4o_mm@2vqe3wkzedsb#80wlxafk3*d8-*", "6ahtb3nenc"),
            ("azure_openai_key: a473e333245822b0c5c29d3d48dcffdc", "a473e333245822b0"),
            ("app --db-pass z7QJRek12Xn4h7II6xFJqifd", "z7QJRek12Xn4h7II"),
            ('{"clientSecret": "Xy7zQ2mL9vK4pR8s"}', "Xy7zQ2mL9vK4pR8s"),
        ],
    )
    def test_keys_are_masked(self, text, secret):
        stored = json.dumps(redact_json(as_a_tool_result(text)), ensure_ascii=False)
        assert secret not in stored
        assert REDACTED in stored

    @pytest.mark.parametrize(
        "payload",
        [
            {"input": {"page_token": 1729712345678}},
            {"ms_per_token": 20.833333333333332},
            {"password": 202610231234, "chat_token": -1001234567890},
            {"NextToken": 1.2345678901e-05},
            {"arguments": json.dumps({"page_token": 1729712345678})},
            {"arguments": json.dumps({"session_token": 1729712345678}, indent=2)},
        ],
    )
    def test_a_number_is_never_masked(self, payload):
        """A mask falls inside a string, so the JSON it is in still parses, at every level."""
        assert redact_json(payload) == payload

    def test_a_mask_never_takes_half_of_an_escape(self):
        # Found building the patterns: curl's `-u key:` ends the command, and the password after
        # the colon was read as the backslash of the quote's escape, masked alone: no longer JSON.
        command = "curl https://api.stripe.com/v1/charges -u sk" "_live_51HxYzAbCdEfGhIjKlMnOpQrStUvWx0123456789:"
        stored = redact_json({"arguments": json.dumps({"command": command})})
        assert json.loads(stored["arguments"])["command"].endswith(f"-u {REDACTED}:")

    def test_two_keys_masked_alike_stay_two(self):
        payload = {"teams": {"sk-" "proj-a1B2c3D4e5F6g7H8i9J0": "team-a", "sk-" "proj-Z9y8X7w6V5u4T3s2R1q0": "team-b"}}
        assert redact_json(payload)["teams"] == {REDACTED: "team-a", f"{REDACTED} (2)": "team-b"}
