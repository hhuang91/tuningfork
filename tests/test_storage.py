"""Blob storage and redaction.

Redaction is security-relevant and runs over every recorded payload, so its edge cases are
pinned here: it must mask secrets, preserve surrounding structure, and leave valid JSON
valid — a recording that will not parse is a recording that cannot be replayed.
"""

from __future__ import annotations

import json
import time

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
            # Found in review: code and data the patterns masked.
            "aws s3api get-object --bucket data --key reports/2024/q3.csv q3.csv",
            "docker build --secret id=npmrc,src=$HOME/.npmrc .",
            "gpg --recv-key 0x4AEE18F83AFDEB23",
            "gpg --keyserver keys.openpgp.org --recv-key 4AEE18F83AFDEB23",
            "session_token: 0x4AEE18F83AFDEB23",
            "token = sha256hmac2(data)",
            "svn checkout --username --password --no-auth-cache URL",
            'curl -u "$GITHUB_USER:$GITHUB_TOKEN" https://api.github.com/user',
            "curl -u username:password https://api.example.com",
            "curl -fsSL https://x.sh | sh && docker run -u 1000:1000 app",
            "curl -fsSLo x.sh https://x.sh && docker run -u 1000:1000 app",
            "mysqldump app | ssh -p2222 backup@host 'cat > app.sql'",
            "mysqladmin ping -h localhost -u root -p$MYSQL_ROOT_PASSWORD",
            "requests.get(url, auth=('user', 'pass'))",
            "const STORAGE_KEY = 'todos-vuejs-2.0'",
            'SESSION_COOKIE_KEY = "sessionid2"',
            "SOCKS5_NO_ACCEPTABLE_AUTH = SOCKS5_NO_ACCEPTABLE_AUTH",
            "DEFAULT_TOKEN = HTTP2_DEFAULT_TOKEN",
            "e.ProgressToken=e.createMessageConnection=e.NullLogger=e.x=void 0",
            "e.CancellationToken=e.CancellationTokenSource=e.Emitter=e.Event=e.Disposable=e.MessageReader="
            "e.MessageWriter=e.AbstractMessageReader=e.AbstractMessageWriter=e.Trace=e.TraceFormat=void 0",
            "if(S.token===Ce.SyntaxKind.ExtendsKeyword)return p",
            "this.apiKey==null&&this.authToken==null",
            "hasPassword: !!password,",
            '{ token: "@rematch", next: "@pop" }',
            "npm_package_devDependencies_typescript=5.6.3",
            "from x import hf_hub_download_v2_cached",
            "PublicKey = mQINBFp4Rk8BEAC7Yd2Lx9Wq4Zt6Hn1Jc3Vb5Pm8Ks0Ar2Ue4Gy7Io9",
            "VAPID_PUBLIC_KEY=BOq3Lx9Wq4Zt6Hn1Jc3Vb5Pm8Ks0Ar2Ue4Gy7Io9Ft1Dz5Nh",
            "STRIPE_PUBLISHABLE_KEY=pk" "_live_51HxYzAbCdEfGhIjKlMnOpQrStUvWx0123456789",
            "SSH_PUBLIC_KEY=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOq3Lx9Wq4Zt6Hn1Jc3Vb5Pm8Ks0Ar2",
            '{"No valid auth token": "有効な認証トークンがありません"}',
            "password: Kennwörter",
            "creationToken = 1644442601",
            '{"token": "1470.529556872868"}',
            '{"idempotency_key": "6f1c2a9e8b7d4c3f2a1e0d9c"}',
            '{"partialResultToken": "1d546990-40a3-4b77-b134-46622995f6ae"}',
            "-W 'user= password= credentials='",
            'logger.info("re-signing token: app_key=****%s", app_key[-4:])',
            "private_key: /tmp/tmpXk2Lq9z/key.pem",
            "    password: OAuthFlowPassword | None = None",
            # Found in review's second round.
            "--- PASS: TestA/sub (0.00s)\n--- PASS: TestJSONv2 (0.01s)",
            "login(username=user2, password=password2)",
            "{ password: password2 }",
            'lastSignificantToken = "?InterpolationInTemplate";',
            'UNTRACKED_CAPPED_AT_KEY = "\\0capped_at_ns"',
            "const UNEXPECTED_TOKEN = /^Unexpected token '?(.)'?(,)? /;",
            # Found in review's third round: a call of a name in lower case or a dotted one.
            '"apiKey": "os.getenv(\'X\')"',
            'token: "Buffer.from(raw).toString()"',
            '{"password": "getPassword()"}',
            'token: "uuid4().hex"',
            # Found measuring the third round's fixes: a raw string holds a pattern, a formatted one
            # code (urllib3's, Hermes's own).
            '_UNRESERVED_PAT = r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._\\-~"',
            'token = f"__HERMES_STT_{len(placeholders)}__"',
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
            # Found in review: formats the base masked under a secret's name, and this left whole.
            ("VAULT_TOKEN=s." "Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1", "Q7bX2kLm9Zr4Tw8Y"),
            ("VAULT_TOKEN=hvs." "CAESIJx2Lm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4Gy7Io9Ft1Dz5Nh", "CAESIJx2Lm9Zr4Tw"),
            ("DOPPLER_TOKEN=dp.st.prd." "Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4G", "Q7bX2kLm9Zr4Tw8Y"),
            ("MAPBOX_SECRET_TOKEN=sk.eyJ1Ijoi" "ZXhhbXBsZSJ9.Q7bX2kLm9Zr4Tw8Yp3Hn6J", "Q7bX2kLm9Zr4Tw8Y"),
            ("AIRTABLE_API_KEY=pat" "Q7bX2kLm9Zr4Tw.0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9",
             "0a1b2c3d4e5f6071"),
            ("SENDGRID_API_KEY=SG." "Q7bX2kLm9Zr4Tw8Yp3Hn6J.c1Vb5Pm8Ks0Ar2Ue4Gy7Io9Ft1Dz5NhQ7bX2kLm9Zr4Tw8",
             "c1Vb5Pm8Ks0Ar2Ue"),
            ("Authorization: Bearer ya29." "a0AfB_byQ7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4Gy7", "a0AfB_byQ7bX2kLm"),
            ('{"access_token": "dop' '_v1_0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9"}',
             "0a1b2c3d4e5f6071"),
            ("SHOPIFY_ACCESS_TOKEN=shpat" "_0a1b2c3d4e5f60718293a4b5c6d7e8f9", "0a1b2c3d4e5f6071"),
            ("Langfuse secret key: sk-" "lf-1d546990-40a3-4b77-b134-46622995f6ae", "1d546990-40a3"),
            ("token: \x1b[32mghp" "_Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2\x1b[0m", "Q7bX2kLm9Zr4Tw8Y"),
            (".env:3:GITHUB_TOKEN=\x1b[01;31m\x1b[Kghp" "_Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2\x1b[m\x1b[K",
             "Q7bX2kLm9Zr4Tw8Y"),
            ("GET /cb?next=%2Fsetup%3Ftoken%3Dghp" "_Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2 302", "Q7bX2kLm9Zr4Tw8Y"),
            ("Authorization=Bearer%20xoxb-" "1234567890123-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx", "AbCdEfGhIjKlMnOp"),
            ("https://sk-" "proj-Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2@api.example.com/v1", "Q7bX2kLm9Zr4Tw8Y"),
            ("REDIS_URL=redis://:Rp4ss9Xk2mQ7@cache.internal:6379/0", "Rp4ss9Xk2mQ7"),
            ("INFO auth ok token=0a1b2c3d4e5f60718293a4b5c6d7e8f9, user=alice", "0a1b2c3d4e5f6071"),
            ("auth: {password: Zq8xK3vN5wLp, user: bob}", "Zq8xK3vN5wLp"),
            ('const apiKey: string = "a1B2c3D4e5F6g7H8i9J0";', "a1B2c3D4e5F6g7H8"),
            ('const API_KEY: &str = "a1B2c3D4e5F6g7H8i9J0";', "a1B2c3D4e5F6g7H8"),
            ("machine api.example.com login bob password Zq8xK3vN5wLp", "Zq8xK3vN5wLp"),
            ("<password>Zq8xK3vN5wLp</password>", "Zq8xK3vN5wLp"),
            ("docker login -u bob -p Zq8xK3vN5wLp registry.example.com", "Zq8xK3vN5wLp"),
            ("api_key: |\n  a1B2c3D4e5F6g7H8i9J0\n", "a1B2c3D4e5F6g7H8"),
            # The same shapes where no secret's name stands before them.
            ("vault login hvs." "CAESIJx2Lm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4Gy7Io9Ft1Dz5Nh", "CAESIJx2Lm9Zr4Tw"),
            ("use dp.st.prd." "Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4G here", "Q7bX2kLm9Zr4Tw8Y"),
            ("mapbox: sk.eyJ1Ijoi" "ZXhhbXBsZSJ9.Q7bX2kLm9Zr4Tw8Yp3Hn6J", "Q7bX2kLm9Zr4Tw8Y"),
            ("airtable: pat" "Q7bX2kLm9Zr4Tw.0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9",
             "0a1b2c3d4e5f6071"),
            ("sendgrid: SG." "Q7bX2kLm9Zr4Tw8Yp3Hn6J.c1Vb5Pm8Ks0Ar2Ue4Gy7Io9Ft1Dz5NhQ7bX2kLm9Zr4Tw8",
             "c1Vb5Pm8Ks0Ar2Ue"),
            ("google: ya29." "a0AfB_byQ7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2Ue4Gy7", "a0AfB_byQ7bX2kLm"),
            ("digitalocean: dop" "_v1_0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9",
             "0a1b2c3d4e5f6071"),
            ("shopify: shpat" "_0a1b2c3d4e5f60718293a4b5c6d7e8f9", "0a1b2c3d4e5f6071"),
            # A password a person made up, of a word and digits; one of digits alone.
            ("environment:\n  MYSQL_ROOT_PASSWORD: rootpass123", "rootpass123"),
            ('{"password": "Welcome2024"}', "Welcome2024"),
            ("DB_PASSWORD=73915264", "73915264"),
            # A password whose case reads as one short word, which a key's name would let pass.
            ("DB_PASSWORD=aUnArlqz", "aUnArlqz"),
            ("curl -u admin:aUnArlqz https://api.example.com", "aUnArlqz"),
            # Found in review's second round: passwords shaped like a call or holding markup, a
            # weak one of a word and digits, and one after a colour of many parameters.
            ('password = "Summer(2024)"', "Summer(2024)"),
            ('{"password": "MyPass[1]x"}', "MyPass[1]x"),
            ('"password": "Ab<3>cd9Kx"', "Ab<3>cd9Kx"),
            ("password: password123", "password123"),
            ("TEST_PASS=Zq8xK3vN5wLp", "Zq8xK3vN5wLp"),
            ('password: "?Xk2Lq9zWm7"', "?Xk2Lq9zWm7"),
            # Found in review's third round: `PASS` before a colon is a test's result, not here.
            ("PASS=Zq8xK3vN5wLp", "Zq8xK3vN5wLp"),
            ("export PASS='Zq8xK3vN5wLp'", "Zq8xK3vN5wLp"),
            # Found in review's fourth round: a password in lower case with a bracket read as a call.
            ('{"password": "letmein[1]x"}', "letmein[1]x"),
            ('password = "my.pass(word)9"', "my.pass(word)9"),
            ('{"api_key": "q(Zr7Lm2Kd9X)"}', "q(Zr7Lm2Kd9X)"),
            ("token: \x1b[1;38;2;255;255;255;48;2;0;0;0mghp" "_Q7bX2kLm9Zr4Tw8Yp3Hn6Jc1Vb5Pm8Ks0Ar2\x1b[0m",
             "Q7bX2kLm9Zr4Tw8Y"),
            # A random key whose case cuts into many short words reads as no name.
            ("API_TOKEN=QwErTyUiOpAsDfGhJkLz", "QwErTyUiOpAsDfGh"),
            ("API_TOKEN=V_uhis1WeZJKXQ_p", "V_uhis1WeZJKXQ_p"),
            ("SECRET_KEY=/BpYconEG4ZgzWbmHGJR1m4jfXuX8v/h=", "BpYconEG4ZgzWbmH"),
            # A quoted password holding brackets or an escape, masked whole.
            ('{"password": "x9!Kq(2Lm]P7z"}', "x9!Kq(2Lm]P7z"),
            ('{"password": "Xq9\\\\Zr7Lm2Kd"}', "Zr7Lm2Kd"),
            ("password = 'Xq9\\\\Zr7Lm2Kd'", "Zr7Lm2Kd"),
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
            {"arguments": json.dumps({"db_password": 20261023123456})},
            # Found in review's ninth round: AWS's secret key is forty characters, digits among them.
            {"aws_secret_access_key": int("7" * 40), "n": 1},
            {"arguments": json.dumps({"SecretAccessKey": int("7" * 40), "n": 1}, separators=(",", ":"))},
            {"arguments": '{"secretAccessKey":' + "1" * 37 + "e+1}"},
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

    def test_a_private_key_shown_in_two_results_keeps_every_message(self):
        """Found in review: a key's body read on to the next END line, and took the JSON between
        its first lines, in one tool result, and its last, in a later one: two messages were lost."""
        head = "-----BEGIN PRIVATE " "KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\nk3Jd8Qx2mVhZ0pLwT5yN"
        tail = "q9Xr2Lm7Wd4Hs1Kb==\n-----END PRIVATE " "KEY-----"
        payload = {"messages": [
            {"role": "tool", "tool_call_id": "c1", "content": head},
            {"role": "assistant", "content": "Now the last line."},
            {"role": "tool", "tool_call_id": "c2", "content": tail},
        ]}
        stored = redact_json(payload)
        assert stored["messages"][0]["content"] == "-----BEGIN PRIVATE KEY-----" + REDACTED
        assert stored["messages"][1:] == payload["messages"][1:]

    @pytest.mark.parametrize(
        "text",
        [
            '{"form":{"fields":[["Username:",32,false],["Password:",128,true]],"minLength":12}}',
            '["Enter your password:",12345678,true,"next"]',
            '[["2024-01-02","token=",1729712345,9.5],["x","y",1,2]]',
            '["api_key:",null,null,null,"done"]',
            # Found in review's fourth round: a string ending in a backslash, read a level deeper.
            '["token: \\"C:\\\\keys\\\\",1729712345,true,"next"]',
        ],
    )
    def test_compact_json_written_in_a_string_stays_json(self, text):
        """Found in review's second round: a string ending in a secret's name and a colon, then
        compact JSON — `["Password:",128,true]` — and a quoted value opened on the string's
        closing quote, to mask `,128,true]],` up to the next string."""
        assert redact_json(as_a_tool_result(text)) == as_a_tool_result(text)

    @pytest.mark.parametrize("prefix", ["", "b", "u"])
    def test_a_key_written_as_bytes_escapes_is_masked(self, prefix):
        # Found in review's third round: a value led by a backslash was taken for an escape.
        key = "\\x8f\\x12\\xab\\xcd\\xef\\x01\\x23\\x45\\x67\\x89"
        stored = redact_json(as_a_tool_result(f'SECRET_KEY = {prefix}"{key}"\n'))["messages"][0]["content"]
        assert stored == f'SECRET_KEY = {prefix}"{REDACTED}"\n'

    def test_a_python_repr_still_reads(self):
        """Found in review's third round: a repr's escaped quote (`\\'`) closed a single-quoted value
        as JSON writes it, and the mask ended between the backslash and the quote."""
        import ast

        text = repr({"password": "Xy7zQ2mL'9vK4\"x", "n": 1})
        stored = redact_json(as_a_tool_result(text))["messages"][0]["content"]
        assert ast.literal_eval(stored)["n"] == 1

    def test_json_written_in_a_string_stays_json(self):
        # Found building the fix: a quoted value that read `\"` whole took the quotes of a JSON
        # string written inside the string, which then no longer parsed.
        stored = redact_json({"password": json.dumps("Xy7zQ2mL9vK4pR8s")})
        json.loads(stored["password"])

    @pytest.mark.parametrize("inner", [{"password": "Xq9Zr7Lm2Kd\\", "n": 1}, {"n": 1, "password": "Xq9Zr7Lm2Kd\\"}])
    @pytest.mark.parametrize("separators", [(", ", ": "), (",", ":")], ids=["spaced", "compact"])
    def test_json_in_a_string_whose_value_ends_in_a_backslash_stays_json(self, inner, separators):
        """Found measuring the third round's fixes: a quoted value that read `\\"` whole took the
        closing quote of a string ending in a backslash, read a level deeper than it is written
        (`"…\\\\"` reads `"…\\"` there), and ran on to the next quote."""
        stored = redact_json(as_a_tool_result(json.dumps(inner, separators=separators)))["messages"][0]["content"]
        assert json.loads(stored)["n"] == 1
        assert "Xq9Zr7Lm2Kd" not in stored

    @pytest.mark.parametrize(
        "text",
        [
            "UPDATE users SET password='Zq8xK3vN\\\\',role='admin' WHERE id=7;",
            "const c={token:'Zq8xK3vN5wLp\\\\',retries:3,host:'db'};",
            "$cfg=['password'=>'Zq8xK3vN\\\\','user'=>'bob'];",
        ],
        ids=["sql", "js", "php"],
    )
    def test_a_single_quoted_value_ending_in_a_backslash_stays_whole(self, text):
        """Found in review's fifth round: in single quotes `\\'` read as an escape, so a string ending
        in a backslash, read a level deeper, ran on past its closing quote to the next one."""
        assert redact_json(as_a_tool_result(text)) == as_a_tool_result(text)

    @pytest.mark.parametrize(
        "unit, end",
        [
            # Found in review: each `token=` in one run of a value's characters read the run to its
            # end, to fail at the bracket there: 40 KB took 2 s, a megabyte would take minutes.
            ("token=a1b2c3d4", "("),
            ("token:", "x1"),
            # And each beginning of a private key without an end read up to 10,000 characters on.
            ("-----BEGIN PRIVATE " "KEY-----\nMIIEvQIBADANBgkqhkiG9w0\n", ""),
        ],
        ids=["names-then-a-bracket", "names-joined", "key-beginnings"],
    )
    def test_a_long_text_is_read_in_one_pass(self, unit, end):
        text = unit * (1_000_000 // len(unit)) + end
        started = time.perf_counter()
        redact_json(as_a_tool_result(text))
        assert time.perf_counter() - started < 4
