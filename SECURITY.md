# Security

## Reporting a vulnerability

Please report it privately, with **Report a vulnerability** on this repository's Security tab,
not in a public issue.

## What tuningfork guards, and what it does not

tuningfork runs on your own machine, beside your model server and your agent harness. What it
does to keep that local:

- **It listens on loopback only, by default.** The proxy and its web UI bind to `127.0.0.1`,
  answer only to a loopback `Host`, and refuse a cross-site change by its `Origin`. Every
  change of state needs the access token that `serve` mints and writes to
  `<recordings>/.access-token` (mode 0600). Reads need no token: anything that can reach the
  port can read the recordings.
- **An agent's MCP client holds no key to the rest.** The MCP endpoint takes a token of its
  own, and no other; the access token is refused there. No MCP tool approves an edit to a
  skill, and none offers the held-out results an edit is judged by.
- **Model servers you protect receive nothing from it.** List them in
  `~/.config/tuningfork/config.json` (`{"protected": ["localhost:1234"]}`), in
  `TUNINGFORK_PROTECTED`, or in a recordings folder's `config.json`.
- **Recordings stay where you put them.** The built-in proposer sends recorded prompts and
  answers only to the endpoint you configure, and to one that is not your machine only once
  you allow it. Its key comes from `TUNINGFORK_PROPOSER_KEY` or a 0600 file, never anywhere
  else.
- **Strings shaped like secrets are redacted before they are stored**: keys by their shape
  (`sk-…`, GitHub's, GitLab's, AWS's, Google's, Slack's, Stripe's, Hugging Face's and more),
  private keys, bearer tokens, passwords in URLs and on command lines, and a value under a
  secret's name (`api_key=…`) where it reads as a key rather than as code. So are the files
  tuningfork writes beside them from what was recorded — a chain's record, an imported chat and
  the skills taken from it — and what a scenario or a skill added takes from a conversation. A
  run's workspace and its harness's own files are kept as they were, and so is what it was given:
  its task's text and check, and its skill. A scenario made of it copies its skill, and its task's
  check, expected output and expectations, as they were; a skill added from it copies its skill.
  What a task's check prints is redacted as it ends, before it is kept or told to a harness.
  Wherever it stands there, the value of a variable the check gets whose name is a secret's is
  masked too, and so is each of tuningfork's own keys, when it is eight characters or more,
  without a space or a quote. That is a net, not a promise. Recordings hold what your harness
  sent and received — prompts, tool output, file contents — so read a recordings folder before
  you share one.

What it does **not** do:

- **The sandbox is not a security boundary.** Where bubblewrap runs, every launch sees the
  same home and project folder, so runs are comparable. The rest of the machine stays
  readable inside, your home included, and the network is reachable. It keeps runs alike,
  not an agent contained.
- **It runs what you give it, as you.** A harness tuningfork launches runs with your
  permissions, and so does a task's check, a shell command from its spec. A check gets only a
  few of your environment variables, such as `PATH` and `LANG`, and those named in `check_env`:
  by its task, by this machine's config, or by the recordings folder's. Where bubblewrap runs,
  it runs in its run's sandbox, and can read what a launch can, and its task's folder. Run
  harnesses, tasks and specs you trust.
