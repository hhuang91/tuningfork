"""Runtime configuration, filesystem layout, and storage-time redaction."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Payloads contain whatever the agent's tools read: shell output, files, environment.
# Redaction runs at storage time, never on forwarded traffic, so a bad pattern can spoil a
# recording but can never alter what the harness receives.
#
# Each pattern masks the group named `secret…` that matched, or else its whole match.
#
# A mask costs the recording: a step whose context holds one can no longer be scored or
# probed (its prompt no longer renders to the recorded token count), a fork past an answer
# holding one is refused, and a resample sends `<redacted>` where the model saw text. So a
# pattern masks what has a key's shape, and a value under a secret's name only where it reads
# as a key — never code, never words: the generic pattern once took `tokenizer.encode` for a
# token, and masked one Hermes source file in six an agent read.
#
# The key prefixes must start a token: without that, `sk-` matched inside ordinary words —
# "task-management-system" was stored as "ta<redacted>".
#
# A mask stays inside the string it is in: none takes a quote, nor a backslash but with the
# escape it starts — alone, at the JSON's own level, it is half of one — so the JSON still
# parses. And every scan is bounded or possessive: a value is read once, however many names
# stand in it.

#: The characters a value is written in, as each kind of pattern reads one: bare, as `.env`
#: files, YAML, headers and command lines write it; in double or single quotes, as code and JSON
#: do (anything but spaces and that quote, and an escape read whole: a backslash masked alone
#: breaks the JSON — but `\"`, and in single quotes `\'`, which may close a string: one ending in
#: a backslash, `"…\\"`, reads `"…\"` where its escapes are read once more); a bearer token;
#: Base64; a password in a URL; a key alone; a command's argument. A quoted value opens only
#: where a string does, never on a JSON escape, `{` or `[`, so JSON written inside the string it
#: is in keeps its quotes.
_BARE = r"A-Za-z0-9_\-.~+/=!@#$%^&*"
_DOUBLE_QUOTED = r"^\s\"\\"
_SINGLE_QUOTED = r"^\s\"'\\"
_TOKEN68 = r"A-Za-z0-9_\-.~+/"
_BASE64 = r"A-Za-z0-9+/"
_URL_PASSWORD = r"^\s/@\"'<>\\"
_KEY_CHARS = r"A-Za-z0-9_\-"
_ARGUMENT = r"^\s\"'\\"

#: A letter that is no capital: a lower-case one, or one of a script without case (`認証`).
_LOWER = r"[^\W\d_A-Z]"

#: Digits ending a word (`sha256`, `Http2`): a lower-case letter never follows them, and a
#: capital only where it starts the next word (`Http2Server`).
_DIGITS = r"(?:[0-9]{1,6}+(?![0-9]|" + _LOWER + r"|[A-Z](?!" + _LOWER + r")))?"

#: The words of a name, as code writes them: a first in lower case (`api`), then words that
#: start with a capital (`Key`, `Token`), or capitals (`HTTP`, up to one that starts the next
#: word: `HTTPServer`). Of words of one or two letters (`Id`, the `X` of `getX`) a name holds one:
#: a random string's case, cut into words, makes many (`xKqPzRtLmWnB`, `V_uhis1WeZJKXQ_p`).
_LONG_WORD = (
    r"(?>[A-Z]" + _LOWER + r"{2,40}+" + _DIGITS + r"|[A-Z](?:[A-Z](?!" + _LOWER + r"))++(?!" + _LOWER + ")"
    + _DIGITS + ")"
)
_SHORT_WORD = r"(?>[A-Z](?:" + _LOWER + ")?(?!" + _LOWER + ")" + _DIGITS + ")"
_FIRST_WORD = r"(?>" + _LOWER + r"{1,40}+" + _DIGITS + ")"

#: A piece of a name between its joints, whole: without a short word, or with its one.
_PIECE = (
    r"(?:(?:" + _FIRST_WORD + _LONG_WORD + r"{0,16}+|" + _LONG_WORD + r"{1,16}+|[0-9]{1,8}+" + _LOWER
    + r"{0,8}+)(?![^\W_]))"
)
_SHORT_PIECE = (
    r"(?:(?:" + _FIRST_WORD + _LONG_WORD + r"{0,16}+|" + _LONG_WORD + r"{1,16}+)?+" + _SHORT_WORD + _LONG_WORD
    + r"{0,16}+(?![^\W_]))"
)

#: A name: pieces joined by `_`, `-`, `.`, `/`, `:`, `#`, `=` or `$` into a snake_case name, a
#: dotted one, a path, a version. Bounded, so a scan of one stops however long the run it is in.
_JOINT = r"[_\-./:#=$]{1,3}+"
_NAME = (
    r"_{0,4}+(?:" + _PIECE + "(?:" + _JOINT + _PIECE + r"){0,12}+(?:" + _JOINT
    + _SHORT_PIECE + "(?:" + _JOINT + _PIECE + r"){0,12}+)?+|" + _SHORT_PIECE + "(?:" + _JOINT + _PIECE
    + r"){0,12}+)_{0,4}+"
)
#: A name longer than the bound reads one: thirteen pieces of names joined, as minified code
#: chains them (`e.CancellationToken=e.CancellationTokenSource=e.Emitter=…`), whatever follows.
_LONG_NAME = r"_{0,4}+" + _PIECE + "(?:" + _JOINT + _PIECE + "){12}"

#: A name as a password's value is one: of words of three letters or more, but its first
#: (`newPassword`, `OAuthFlowPassword`). A password a person or a generator makes reads as one
#: less often than a key does.
_PASSWORD_NAME = (
    r"_{0,4}+(?![0-9]{1,8}+(?![_\-./:#=$0-9]))(?:" + _PIECE + "|" + _SHORT_WORD + _LONG_WORD
    + r"{1,16}+(?![^\W_]))(?:" + _JOINT + _PIECE + r"){0,12}+_{0,4}+"
)


def _not_a_key(chars: str, *, numbers: bool = True, password: bool | str = False) -> str:
    """A lookahead: the value ahead, written in `chars`, is no key though it stands where one
    could. Not a name (`_NAME`): a word, words joined (`Initialization`, `resolved_api_key`,
    `_RUNTIME_KEY`, `resolvedApiKey`, `token-based-authentication`, `!!enabled`), a dotted name
    (`settings.SECRET_KEY`), a path (`reports/2024/q3.csv`) or numbers between dots (a version,
    an address, an OID), an option's name (`credentials=`) — but, where the value is a
    `password` (or where the group of that name matched), one word and digits (`rootpass123`,
    `Welcome2024`), as a person makes a password up, where a constant's name holds an id
    (`sessionid2`), unless it numbers a password's field (`password2`); a key's random letters
    and digits read as no name. Not a call or an index of a name in lower case of three characters
    or more, or of a dotted one, closed where the value ends — a chain of them perhaps, names after
    (`getPassword()`, `os.getenv('X')`, `Buffer.from(raw).toString()`, `uuid4().hex`) — where a
    password may read as a capital's (`Summer(2024)`), or a short name's, or go on after it
    (`letmein[1]x`). Not a number, where `numbers` and the value is no password's (a count, a
    time, a page's place), nor a key's id or an address in hex (`0x4AEE18F83AFDEB23`); not a path
    to a file that holds one (`~/.ssh/id_ed25519`), nor a reference to where one is kept
    (`op://…`, `vault://…`); not a mask's stars, nor an example cut short (`sk-...`); not a
    variable or a placeholder (`$TOKEN`, `${{ secrets.X }}`, `<your-key>`, `%KEY%`, `@rematch`,
    `[KEY]`), an escape but bytes' (`\\0capped`, not `\\x8f\\x12…`), a regex (`/^…`), nor, where
    the group names no password, a state a tokenizer names (`?InterpolationInTemplate`)."""
    end = "[.!=]?(?![" + chars + "])"
    made_up = (
        r"(?!(?!(?:pass(?:word|wd)?|pwd)[0-9]" + end + r")(?:[A-Z]?" + _LOWER + r"{1,40}+|[A-Z]{1,40}+)[0-9]{1,6}+"
        + end + ")"
    )
    number = r"-?[0-9]{1,40}+(?:\.[0-9]{1,40}+)?+(?:[eE][-+]?[0-9]{1,3}+)?+" + end
    call = r"(?:\((?:(?![()])[" + chars + r"]){0,64}+\)|\[(?:(?![\[\]])[" + chars + r"]){0,64}+\])"
    name, state = _PASSWORD_NAME if password is True else _NAME, ""
    if isinstance(password, str):
        made_up, number = f"(?({password}){made_up})", f"(?({password})(?!)|{number})"
        name, state = f"(?({password}){_PASSWORD_NAME}|{_NAME})", rf"|(?({password})(?!)|\?)"
    return (
        r"(?!" + (made_up if password else "") + r"!{0,2}+" + name + end + "|" + _LONG_NAME
        + r"|(?:(?=[a-z][A-Za-z0-9_$]{2})|(?=[A-Za-z0-9_$]{1,64}+\.))" + name + call
        + r"(?:" + call + r"|\." + name + r"){0,8}+(?![" + chars + "])"
        + (("|" + number) if numbers else "")
        + r"|0x(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{16}|[0-9a-fA-F]{8})" + end
        + r"|(?:~|\.{1,2})?/{1,2}+\.?(?:" + _PIECE + "|" + _SHORT_PIECE + ")(?:/|" + end + ")"
        + r"|[A-Za-z][A-Za-z0-9+.\-]{0,31}+://"
        + "|[" + chars + r"]{0,64}?(?:\.\.\.|…|\*{3})"
        + r"|[$%{<@\[(]|\\(?!x[0-9a-fA-F]{2})|/\^" + state + ")"
    )


def _either(word: str) -> str:
    """`word` in either case, letter by letter. Each pattern starts with a letter or a set of
    them, and looks behind only after it: the engine then skips to where one stands, where a
    pattern starting with a lookbehind, or under `(?i)`, is tried at every character."""
    return "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else re.escape(c) for c in word)


def _starts(read: str, chars: str = r"A-Za-z0-9_\-") -> str:
    """After `read`, what a pattern has just read — a key's whole prefix, so that this is looked
    at only where one stands: it starts a token. No character a key or a name is written in
    stands before it, but where it ends a URL's escape (`%3D`) or a terminal's colour (`ESC[33m`,
    and grep's `ESC[K`), which join it to no word."""
    colours = "|".join(rf"(?<=\x1b\[[0-9;]{{{n}}}[mK]{read})" for n in range(33))
    return rf"(?:(?<![{chars}]{read})|(?<=%[0-9A-Fa-f]{{2}}{read})|(?<=[mK]{read})(?:{colours}))"


#: Starts a word.
_WORD_START = r"(?<![A-Za-z0-9_]{})"

#: How the name of a secret ends: `OPENAI_API_KEY`, `db_password`, `aws_secret_access_key`,
#: `clientSecret`, `GITHUB_TOKEN`, `DB_PASS`, `dbPass`, and an upper-case `…_KEY`, `…_AUTH`,
#: `…_PAT` or `…_CREDENTIALS` as a `.env` file writes one — but a public key's (`…_PUBLIC_KEY`,
#: `…_PUBLISHABLE_KEY`). Not a page's token (`next_page_token`, `nextPageToken`): opaque, but no
#: key, and an agent sends it back in its next call. Not `PASS` on its own before a colon, as a
#: test's result (Go's `--- PASS: TestA`); `PASS=` is one.
_SECRET_NAME = (
    r"[AaSsPpEeMmTt_](?:(?<=[Aa])(?i:pi[_\-]?key|ccess[_\-]?key|ccount[_\-]?key)"
    r"|(?<=[Ss])(?i:ecret[_\-]?key|igning[_\-]?key|ecret)"
    r"|(?<=[Pp])(?i:rivate[_\-]?key)"
    r"|(?P<password>(?<=[Pp])(?i:assword|asswd|assphrase|ass(?<![a-z]pass))(?!(?<=PASS)(?<!_PASS):)"
    r"|(?<=P)ass(?<=[a-z0-9]Pass))"
    r"|(?<=[Ee])(?i:ncryption[_\-]?key)|(?<=[Mm])(?i:aster[_\-]?key)"
    r"|(?<=[Tt])(?i:oken" + "".join(
        f"(?<!{page}{joint}token)"
        for page in ("page", "next", "cursor", "sync", "continuation", "pagination", "resume", "workdone",
                     "partialresult", "progress")
        for joint in ("", "_", "-")
    )
    + r")|(?<=_)(?:KEY(?<=[A-Z0-9]_KEY)(?<!PUBLIC_KEY)(?<!PUBLISHABLE_KEY)|AUTH(?<=[A-Z0-9]_AUTH)"
    r"|PAT(?<=[A-Z0-9]_PAT)|CREDENTIALS(?<=[A-Z0-9]_CREDENTIALS)))"
)

#: Names ending in `key` whose value is no secret, though random: a request's idempotency key, a
#: cache's, a table's.
_NOT_SECRET_KEYS = ("idempotency", "cache", "sort", "partition", "primary", "foreign", "object", "row",
                    "dedup", "dedupe", "lookup", "storage", "message", "translation", "i18n")

#: A long flag named for a secret: `--api-key`, `--secret-key`, `--azure-openai-key`, `--db-pass`;
#: not GnuPG's, which name a public key's id (`--recv-key`).
_SECRET_FLAG = (
    r"--(?i:(?:[a-z0-9]+-)*(?:key|token|secret|(?P<password>password|passwd|pass)))(?![A-Za-z0-9_\-])"
    r"(?<!recv-key)(?<!send-key)(?<!search-key)"
)

#: Text up to a command's flag: on the command's own line — a line continued with `\` stays one,
#: as each level of escapes writes it — and not past `|`, `;` or `&&`, where another command
#: starts (`curl … && docker run -u 1000:1000`).
_SAME_COMMAND = r"(?:[^\n\\|;&]|&(?!&)|\\\n|\\\\\\n){0,300}?"

#: A type a declaration names before its value: `apiKey: string = …`, `API_KEY: &str = …`.
_TYPED = r"(?::[ \t]*&?[A-Za-z_][A-Za-z0-9_.<>\[\]|,' ]{0,40}?[ \t]*(?==))?"

DEFAULT_REDACTIONS: tuple[str, ...] = (
    # Keys known by their shape, wherever they stand. `sk-` is no key where words follow it
    # (`sk-toggleable__content`, `sk-ecdsa-sha2-nistp256@openssh.com`); the generic prefixes need
    # a digit or a capital, which an identifier such as `fw_update_settings` never has.
    r"(?P<secret>s(?:k(?:-" + _starts("sk-") + r"(?![a-z]{1,12}+[0-9]{0,6}+(?:[_\-]++(?:[a-z]{1,12}+[0-9]{0,6}+"
    r"|[0-9]{1,6}+))++(?![A-Za-z0-9_\-]))[A-Za-z0-9_\-]{16,}+|_" + _starts("sk_") + r"(?:live|test)_[A-Za-z0-9]{20,}"
    r"|\." + _starts(r"sk\.") + r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"
    r"|hp" + _starts("shp") + r"(?:at|ss|ca|pa)_[0-9a-fA-F]{32})"
    r"|g(?:h[pousr]_" + _starts("gh[pousr]_") + r"[A-Za-z0-9]{16,}|ithub_pat_" + _starts("github_pat_")
    + r"[A-Za-z0-9_]{22,}|lpat-" + _starts("glpat-") + r"(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|sk_" + _starts("gsk_") + r"(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,})"
    r"|h(?:f_" + _starts("hf_") + r"(?=[A-Za-z0-9]*[0-9A-Z])[A-Za-z0-9]{30,}|v[sbr]\." + _starts(r"hv[sbr]\.")
    + r"[A-Za-z0-9_\-]{24,})"
    r"|fw_" + _starts("fw_") + r"(?=[A-Za-z0-9]*[0-9A-Z])[A-Za-z0-9]{20,}"
    r"|npm_" + _starts("npm_") + r"(?=[A-Za-z0-9]*[0-9A-Z])[A-Za-z0-9]{30,}"
    r"|tgp_v1_" + _starts("tgp_v1_") + r"(?=[A-Za-z0-9_\-]*[0-9A-Z])[A-Za-z0-9_\-]{20,}"
    r"|d(?:o[opr]_v1_" + _starts("do[opr]_v1_") + r"[0-9a-f]{64}"
    r"|p\." + _starts(r"dp\.") + r"(?:st|ct|sa|pt|scim|audit)\.(?:[a-z0-9_\-]{1,32}\.)?[A-Za-z0-9]{40,})"
    r"|p(?:ypi-" + _starts("pypi-") + r"[A-Za-z0-9_\-]{40,}|at[A-Za-z0-9]{14}\." + _starts(r"pat[A-Za-z0-9]{14}\.")
    + r"[0-9a-f]{64})"
    r"|rk_" + _starts("rk_") + r"(?:live|test)_[A-Za-z0-9]{20,}"
    r"|whsec_" + _starts("whsec_") + r"[A-Za-z0-9+/=]{24,}"
    r"|x(?:ox[abprs]-" + _starts("xox[abprs]-") + r"[A-Za-z0-9\-]{10,}|app-" + _starts("xapp-")
    + r"[0-9]-[A-Za-z0-9\-]{10,})"
    r"|ya29\." + _starts(r"ya29\.") + r"[A-Za-z0-9_\-]{20,}"
    r"|SG\." + _starts(r"SG\.") + r"[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}"
    r"|A(?:Iza" + _starts("AIza") + r"[0-9A-Za-z_\-]{35}"
    r"|[KS]IA" + _starts("A[KS]IA", r"A-Za-z0-9_\-+/") + r"[0-9A-Z]{16}(?![A-Za-z0-9+/]))"
    r")",
    # A JSON Web Token; a Discord bot's token; a Telegram bot's, after its bot's number.
    r"(?P<secret>eyJ" + _starts("eyJ") + r"[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})",
    r"(?P<secret>[MNO]" + _starts("[MNO]") + r"(?=[A-Za-z_\-]*[0-9])[A-Za-z0-9_\-]{23,27}"
    r"\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,40})(?![A-Za-z0-9_\-])",
    r"[0-9](?<![0-9][0-9])[0-9]{7,9}:(?P<secret>AA[A-Za-z0-9_\-]{33})(?![A-Za-z0-9_\-])",
    # A private key's body, from its first line to its last, or to where the text stops; a
    # body that starts as one does (Base64, or an encrypted key's header), not a mention. It
    # holds only what a key's body does — Base64, line breaks as each level of escapes writes
    # them, an encrypted key's `Proc-Type: 4,ENCRYPTED` — and stops at the next `-----`: a body
    # that ran on to an END line in another string took the JSON between them.
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?=(?:\s|\\[nrt])*+(?:Proc-Type:|[A-Za-z0-9+/]{16}))"
    r"(?P<secret>(?:\\[nrt]|[A-Za-z0-9+/=\s:,]|-(?!----))++)",
    # A password in a URL, in HTTP Basic credentials, curl's `-u`, MySQL's `-p`, `docker login`'s,
    # requests' `auth=` (opening only where a string does, as a quoted value below, and holding no
    # space), a `.netrc` line or an XML element; AWS's secret key — under a quoted
    # name, quoted: a number in JSON is none — an Azure connection string or signature, and a Slack
    # webhook's path.
    r"://[^\s/:@\"'<>\\]*+:(?P<secret>" + _not_a_key(_URL_PASSWORD, numbers=False, password=True)
    + "[" + _URL_PASSWORD + "]++)@",
    r"[Bb]" + _WORD_START.format("[Bb]") + _either("asic") + r"[ \t]+(?P<secret>" + _not_a_key(_BASE64)
    + r"[A-Za-z0-9+/]{12,}+={0,2})(?![A-Za-z0-9+/=])",
    r"[Cc]" + _WORD_START.format("[Cc]") + _either("url") + r"(?![A-Za-z0-9_])" + _SAME_COMMAND
    + r"[ \t](?:-u|--user)[ \t=]+[\"']?[^\s:\"'\\]++:(?P<secret>" + _not_a_key(_ARGUMENT, numbers=False, password=True)
    + r"[^\s\"'\\]+)",
    r"[Mm]" + _WORD_START.format("[Mm]") + r"(?:" + _either("ysql") + r"(?:" + _either("dump") + "|" + _either("admin")
    + r")?|" + _either("ariadb") + r")(?![A-Za-z0-9_])" + _SAME_COMMAND + r"[ \t]-p(?P<secret>"
    + _not_a_key(_ARGUMENT, numbers=False, password=True) + r"[^\s\"'\\]{4,})",
    r"[Dd]" + _WORD_START.format("[Dd]") + _either("ocker") + r"[ \t]+login(?![A-Za-z0-9_])" + _SAME_COMMAND
    + r"[ \t]-p[ \t]+(?P<secret>" + _not_a_key(_ARGUMENT, numbers=False, password=True) + r"[^\s\"'\\]{4,})",
    r"[Aa]" + _WORD_START.format("[Aa]") + _either("uth") + r"[ \t]*=[ \t]*\([ \t]*(?P<u>[\"'])[^\"'\\]*(?P=u)[ \t]*,"
    r"[ \t]*(?P<q>[\"'])(?![,:;)\]}])(?P<secret>" + _not_a_key(_ARGUMENT, numbers=False, password=True) + "["
    + _ARGUMENT + r"]+)(?P=q)",
    r"m" + _WORD_START.format("m") + r"achine[ \t]+[^\s\"'\\]+(?:[ \t]+login[ \t]+[^\s\"'\\]+)?[ \t]+password[ \t]+"
    r"(?P<secret>" + _not_a_key(_ARGUMENT, numbers=False, password=True) + r"[^\s\"'\\]+)",
    r"<(?:(?i:password|passwd|secret|token|api[_\-]?key|client[_\-]?secret|access[_\-]?token|private[_\-]?key))>"
    r"(?P<secret>" + _not_a_key(r"^<\s\"'\\", numbers=False, password=True) + r"[^<\s\"'\\]{8,}+)</",
    r"[AaSs](?:(?<=[Aa])(?i:ws_secret_access_key)|(?<=[Ss])(?i:ecretaccesskey))(?P<kq>[\"'])?[ \t]*[:=\t ][ \t]*"
    r"(?(kq)[\"']|[\"']?)(?P<secret>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
    r"[?&;Ss](?:(?<=[?&;])[Ss][Ii][Gg]|(?<=[Ss])(?i:hared[_\-]?access[_\-]?signature))"
    r"=(?P<secret>[A-Za-z0-9%+/=]{20,})",
    r"hooks\.slack\.com/services/(?P<secret>T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+)",
    # A bearer token, and a key given to a command's flag — not another flag (`svn --password
    # --no-auth-cache`). Neither crosses a line.
    r"[Bb]" + _WORD_START.format("[Bb]") + _either("earer") + r"[ \t]+(?P<secret>" + _not_a_key(_TOKEN68)
    + "[" + _TOKEN68 + "]{16,}+=*)",
    _SECRET_FLAG + r"(?:[ \t]+|=)[\"']?(?P<secret>(?!-)" + _not_a_key(_BARE, password="password")
    + "[" + _BARE + "]{8,}+)",
    # A value under a secret's name, in quotes — a bytes string's `b` or a unicode one's `u`
    # perhaps before them, never a raw string's `r` nor a formatted one's `f`, which hold a pattern
    # or code — or bare, a type perhaps named between them. A quoted name takes a quoted value
    # only — `"page_token": 1729712345678` is a number, and masked bare it would no longer be
    # JSON — and neither crosses a line, but into a YAML block (`api_key: |`): `if not api_key:`
    # is no assignment of the next line. A bare value before `(` or `[` is a call or an index; how
    # far its run goes is looked at within a key's length, so a run of names is read once.
    _SECRET_NAME + r"(?P<kq>[\"'])?[ \t]*" + _TYPED + r"(?:=>|:=|:|=(?!=))[ \t]*(?:[|>][-+]?[ \t]*(?:\n|\\n)[ \t]+)?"
    r"(?:[bBuU](?=[\"']))?"
    r"(?:\"(?P<secret>(?![,:;)\]}])" + _not_a_key(_DOUBLE_QUOTED, password="password") + r"(?:["
    + _DOUBLE_QUOTED + r"]|\\[^\"]){8,}+)\""
    r"|'(?P<secret_single>(?![,:;)\]}])" + _not_a_key(_SINGLE_QUOTED, password="password") + r"(?:["
    + _SINGLE_QUOTED + r"]|\\[^\"']){8,}+)(?<!\\)'"
    r"|(?(kq)(?!)|(?P<secret_bare>" + _not_a_key(_BARE, password="password") + "(?![" + _BARE
    + "]{8,256}+[(\\[])[" + _BARE + r"]{8,}+)))",
    # A value under any other name ending in `key` (`azure_openai_key`, `mistralKey`), where it
    # reads as a key alone: digits and letters, twenty or more, ending where a key ends. Not a
    # UUID, a path, a cache's or a table's key, nor a public one.
    r"[_\-K](?:(?<=[_\-])[Kk][Ee][Yy](?<=[A-Za-z0-9][_\-][Kk][Ee][Yy])|(?<=K)ey(?<=[a-z0-9]Key))"
    r"(?<![Pp][Uu][Bb][Ll][Ii][Cc][_\-][Kk][Ee][Yy])(?<![Pp]ublicKey)"
    r"(?<![Pp][Uu][Bb][Ll][Ii][Ss][Hh][Aa][Bb][Ll][Ee][_\-][Kk][Ee][Yy])(?<![Pp]ublishableKey)"
    + "".join(rf"(?<!{_either(name)}[_\-][Kk][Ee][Yy])(?<!{name}Key)" for name in _NOT_SECRET_KEYS)
    + r"(?P<kq>[\"'])?[ \t]*(?::|=(?!=))[ \t]*(?(kq)[\"']|[\"']?)"
    r"(?P<secret>(?=[A-Za-z0-9_\-]{0,256}?[0-9])(?=[A-Za-z0-9_\-]{0,256}?[A-Za-z])" + _not_a_key(_KEY_CHARS)
    + r"(?![0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![A-Za-z0-9_\-]))"
    "[" + _KEY_CHARS + r"]{20,}+)(?![A-Za-z0-9_\-./:+@,(\[)\]}])",
)


REDACTED = "<redacted>"

#: How deep redaction reads JSON written inside JSON strings: a stored payload's own strings,
#: a call's arguments or a tool's JSON result inside them, and what that result quotes.
JSON_DEPTH = 3

#: A JSON escape: how JSON text writes a character of a string.
_ESCAPE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])')
_SHORT_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

#: Marks a request as experiment traffic. A tuningfork proxy on the path forwards it through
#: its gate but does not record it as a run step: a resampled request carries the same
#: message path as the step it came from, and would otherwise look like the harness
#: repeating that call.
EXPERIMENT_HEADER = "x-tuningfork-experiment"


def server_root(base_url: str) -> str:
    """A server's root, from a base URL with or without the `/v1` harnesses are given."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def without_credentials(url: str) -> str:
    """`url` as it is kept and shown: what stands before its host's `@` — a user and password,
    or a token alone — masked whole, whatever the patterns would read in it."""
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return redact(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=f"{REDACTED}@{parts.netloc.rpartition('@')[2]}"))


#: A value a mask took that is masked wherever else it stands in a step — its tokens, its raw
#: bytes, an error that quotes it (`also`): a key's length or more, and no quote, backslash or
#: space, which no string's edge holds, so masking it never crosses one.
_FOUND = re.compile(r"[^\"\\\s]{8,}")


def redact(text: str, patterns: tuple[str, ...] = DEFAULT_REDACTIONS, also: Iterable[str] = ()) -> str:
    """Mask secrets in `text`, as it stands and as its JSON escapes read (`_read_spans`), and
    every place a value `also` names stands. The replacement is quote-free, and replaces whole
    escapes, so JSON stays parseable."""
    return _masking(text, patterns, also)[0]


def _masked(text: str, patterns: tuple[str, ...]) -> tuple[str, int]:
    """`text` redacted, and how many masks that made."""
    return _masking(text, patterns, ())[:2]


def _masking(text: str, patterns: tuple[str, ...], also: Iterable[str]) -> tuple[str, int, list[str]]:
    """`text` redacted, how many masks that made, and the values they took that are masked
    wherever else they stand (`_FOUND`)."""
    spans = _merged(_read_spans(text, patterns) + _found_spans(text, also))
    if not spans:
        return text, 0, []
    pieces, taken, at = [], [], 0
    for start, end in spans:
        pieces += [text[at:start], REDACTED]
        if _FOUND.fullmatch(text, start, end):
            taken.append(text[start:end])
        at = end
    return "".join(pieces) + text[at:], len(spans), taken


def _found_spans(text: str, also: Iterable[str]) -> list[tuple[int, int]]:
    """Every place in `text` a value a mask took elsewhere stands."""
    spans = []
    for value in also:
        if not _FOUND.fullmatch(value):
            continue
        start = text.find(value)
        while start >= 0:
            spans.append((start, start + len(value)))
            start = text.find(value, start + 1)
    return spans


def _masked_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    # A pattern with a `secret` group masks that group only — the `api_key=` before it stays,
    # so a redacted recording stays readable — and one without masks its whole match. One with
    # several ways to a secret names each group `secret…`, and masks the one that matched.
    spans = []
    for pattern in patterns:
        compiled = re.compile(pattern)
        groups = [name for name in compiled.groupindex if name.startswith("secret")]
        for match in compiled.finditer(text):
            spans.append(next((match.span(name) for name in groups if match.group(name) is not None), match.span()))
    return spans


def _merged(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _json_unescaped(text: str) -> tuple[str, Callable[[int], int]] | None:
    """`text` with its JSON escapes read once, and a map from an index of that back to `text`'s:
    a character read from an escape stands for the whole escape. None where there is none."""
    pieces: list[str] = []
    read_at: list[int] = []  # where each escape's character sits in what is read
    shifts: list[int] = []  # how far `text` runs ahead of what is read, after each escape
    at = shift = 0
    for match in _ESCAPE.finditer(text):
        escape = match.group()
        char = chr(int(escape[2:], 16)) if escape[1] == "u" else _SHORT_ESCAPES[escape[1]]
        pieces += [text[at : match.start()], char]
        read_at.append(match.start() - shift)
        shift += len(escape) - 1
        shifts.append(shift)
        at = match.end()
    if not read_at:
        return None

    def back(index: int) -> int:
        before = bisect.bisect_left(read_at, index)
        return index + (shifts[before - 1] if before else 0)

    return "".join(pieces) + text[at:], back


def _read_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    """The spans of `text` the patterns mask, read as it stands and as its JSON escapes read,
    `JSON_DEPTH` deep. Stored payloads are JSON, where a newline is a backslash and an `n`: a key
    on a line of its own follows a letter there, which an `sk-` key must not, and `api_key: …`
    inside a call's arguments or a tool's JSON result has its quotes escaped."""
    spans = _masked_spans(text, patterns)
    view, back = text, (lambda index: index)
    for _ in range(JSON_DEPTH):
        read = _json_unescaped(view)
        if read is None:
            break
        view, inner = read
        back = (lambda outer, inner: lambda index: outer(inner(index)))(back, inner)
        spans += [(back(start), back(end)) for start, end in _masked_spans(view, patterns)]
    return spans


def _json_escaped(text: str) -> tuple[str, list[int]]:
    """`text` as a JSON string holds it (between the quotes), and for each character of that
    the index of the character of `text` it came from."""
    short = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    out: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        piece = short.get(char) or (f"\\u{ord(char):04x}" if ord(char) < 0x20 else char)
        out.append(piece)
        origin.extend([index] * len(piece))
    return "".join(out), origin


def redact_spans(
    text: str, patterns: tuple[str, ...] = DEFAULT_REDACTIONS, also: Iterable[str] = ()
) -> list[tuple[int, int]]:
    """The spans of `text` a redaction masks, merged and in order.

    For text that is stored in pieces — an answer as its tokens — where masking the joined text
    is not possible. Patterns run over the text as storage reads it: as it stands and as its
    JSON escapes read (a model that writes JSON writes them), and as JSON escapes it, as it is
    stored — so a secret any of these would mask is masked here; and every place a value `also`
    names stands, a value masked in another form of the same answer.
    """
    spans = _read_spans(text, patterns) + _found_spans(text, also)
    escaped, origin = _json_escaped(text)
    for start, end in _masked_spans(escaped, patterns):
        if end > start:
            spans.append((origin[start], origin[end - 1] + 1))
    return _merged(spans)


@dataclass(frozen=True, slots=True)
class Settings:
    """Where tuningfork keeps state and how the proxy behaves."""

    upstream_base_url: str
    home: Path
    host: str = "127.0.0.1"
    port: int = 8787
    # Cap on concurrent upstream requests. Defaults to 1: on a shared inference server, a
    # fan-out of resamples must never starve whatever else is using the model.
    max_inflight: int = 1
    request_timeout: float = 600.0
    #: While a streaming call is held at a breakpoint, an SSE comment is sent this often so
    #: that a client or intermediary with an idle timeout keeps the connection open.
    keepalive_interval: float = 10.0
    #: Background jobs run at once (`tuningfork.jobs`). On one GPU, more is slower.
    jobs: int = 1
    #: The access token to require for state changes (`tuningfork.access`); a new one is
    #: made when none is given. Never shown in a repr.
    token: str | None = field(default=None, repr=False)
    #: The MCP endpoint's own token (`access.MCP_ENV`), which opens nothing else.
    mcp_token: str | None = field(default=None, repr=False)
    #: The tools the proxy's MCP route offers (`tuningfork.mcp`): evidence, all, or none —
    #: no route at all.
    mcp_tools: str = "evidence"
    #: What a tool result carries: "both" its text and its structured content, or the "text"
    #: alone, for a client that shows the model both (Hermes).
    mcp_results: str = "both"
    #: How long one MCP call may wait for a job, at most, in seconds: under what the client
    #: gives a tool call. OpenJarvis reads an HTTP answer for 60 s, Hermes gives a call 300 s
    #: by default — for it, about 110 keeps a twenty-minute screen to a dozen polls.
    mcp_max_wait: int = 50
    #: The most output tokens any call a harness makes through this proxy may ask for: one that
    #: asks for more, or names no limit, is sent with this one (`wire.base.cap_output`), and what
    #: it asked is recorded with the step. None leaves every call as its harness sent it.
    max_output: int | None = None

    @property
    def blob_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def step_log(self) -> Path:
        """Phase 0 step log. Replaced by the SQLite message tree in Phase 1."""
        return self.home / "steps.jsonl"


def default_home() -> Path:
    return Path(os.environ.get("TUNINGFORK_HOME") or Path.home() / ".tuningfork")


#: Settings kept with a recordings folder: harness executables, extra protected endpoints,
#: recent upstreams. Never a harness's own configuration.
HOME_CONFIG = "config.json"


def _read_object(path: Path) -> dict[str, Any]:
    """A JSON object from `path`, or an empty one. A file that cannot be read is treated as
    empty rather than stopping every command."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_home_config(home: Path) -> dict[str, Any]:
    """The recordings folder's `config.json`, or an empty one."""
    return _read_object(Path(home) / HOME_CONFIG)


def machine_config_path() -> Path:
    """The machine's own settings, whichever recordings folder is in use:
    `$XDG_CONFIG_HOME/tuningfork/config.json`, by default under `~/.config`. Read, strictly,
    for the endpoints to protect on this machine (`endpoints`)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "tuningfork" / HOME_CONFIG


def write_home_config(home: Path, data: dict[str, Any]) -> None:
    """Replace the recordings folder's `config.json` atomically."""
    path = Path(home) / HOME_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{HOME_CONFIG}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def redact_json(obj: Any, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> Any:
    """Redact a JSON-serializable object by round-tripping it through `redact`.

    Replacements are quote-free and fall inside strings, so the result still parses. Applied to
    payloads on their way into storage, which means a recording containing a secret will not
    reconstruct byte-identically — that is the intended trade. Two keys of one object masked
    alike stay two: the second is `<redacted> (2)`, and so on.
    """
    return redact_json_counted(obj, patterns)[0]


def redact_json_counted(obj: Any, patterns: tuple[str, ...] = DEFAULT_REDACTIONS) -> tuple[Any, int]:
    """`redact_json`, and how many masks it made: what a step keeps (`masked`), since text that
    reads `<redacted>` on its own — a client's debug log writes it — is no mask."""
    return redact_json_found(obj, patterns)[:2]


def redact_json_found(
    obj: Any, patterns: tuple[str, ...] = DEFAULT_REDACTIONS, also: Iterable[str] = ()
) -> tuple[Any, int, frozenset[str]]:
    """`redact_json_counted`, masking too every place a value `also` names stands, and the
    values its masks took that are masked wherever else they stand: what a step's other forms —
    its answer's tokens, its raw bytes, an error quoting it — are masked of (`redact`'s `also`)."""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return obj, 0, frozenset()
    redacted, made, taken = _masking(text, patterns, also)
    if not made:
        return obj, 0, frozenset()
    try:
        return json.loads(redacted, object_pairs_hook=_apart), made, frozenset(taken)
    except json.JSONDecodeError:
        # A pattern broke the structure. Losing the recording beats storing the secret.
        return {"tuningfork_error": "redaction produced invalid JSON; payload dropped"}, made, frozenset(taken)


#: How masks fall besides the patterns: how escapes are read (`JSON_DEPTH`), keys kept apart
#: (`_apart`). Raised when that changes, as a change of the patterns changes their digest.
MASKING_VERSION = 1


def patterns_digest(patterns: tuple[str, ...]) -> str:
    """What names a set of patterns and how they mask: kept with each step as `redaction`, so
    what reads a step again knows whether it was masked as the patterns at hand mask (a fork's
    `replay.fits`)."""
    named = f"masking {MASKING_VERSION}\n" + "\n".join(patterns)
    return hashlib.sha256(named.encode("utf-8")).hexdigest()[:16]


def masks_in(request: Any, params: dict[str, Any] | None) -> int:
    """How many masks a stored request holds: as its step counted them as it was recorded
    (`masked`), else — a step recorded before steps counted them — every `<redacted>` in it."""
    counted = (params or {}).get("masked")
    if isinstance(counted, int):
        return counted
    return json.dumps(request, ensure_ascii=False).count(REDACTED)


def _apart(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """An object's pairs, a key masked as another already is numbered: none is lost."""
    kept: dict[str, Any] = {}
    for key, value in pairs:
        if key in kept and REDACTED in key:
            number = 2
            while f"{key} ({number})" in kept:
                number += 1
            key = f"{key} ({number})"
        kept[key] = value
    return kept
