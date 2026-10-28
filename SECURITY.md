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
  port can read the recordings, another user's process on this machine included.
- **The recordings folder is yours alone when tuningfork makes it** (mode 0700), and so is each
  folder it lays launches out in, wherever that is, on a file system that keeps modes (a Windows
  drive under WSL may not). `serve` and `ui` say when a recordings folder made before can be opened
  by others, and how to make it yours alone; tuningfork never changes it by itself.
- **Deleting from the trash is for good.** A run goes with what is tied to it, the folders
  tuningfork made for its launches, whatever else they hold, and what nothing left refers to: its
  messages, blobs and workspace snapshots, and its batch's folder with the skill copy the batch
  was given, once none of the batch's runs is left and nothing else is in it. A path in any of
  them that a skill added, a scenario, or the task file of a run still recorded names stays where
  it is, and the deletion's preview says so. Outside the recordings folder, a folder goes only
  where tuningfork marked it as made for this recordings folder, which earlier versions did not.
  What its database rows held is overwritten, and the database's log emptied once no other process
  is reading it. What was written in the last two minutes, which a recording in flight may still
  need, goes once those have passed, while a proxy serves the recordings folder, or at the next
  deletion. Files go as your file system deletes them.
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
  its task's text and check, and its skill. A scenario made of it copies its task's check, expected
  output and expectations as they were, and its skill too where it keeps a copy of it; a skill
  added from it copies its skill. What a task's check prints is redacted as it ends, before it is
  kept or told to a harness. What leaves tuningfork is redacted as it leaves, whatever it keeps:
  everything an agent is given over MCP, whose model may be hosted — a run's files and the skill
  it was given among them, and no digest of them — and everything the built-in proposer is sent.
  A run's page and the task list show its task's check redacted too.
  Wherever it stands there, the value of a variable the check gets whose name is a secret's is
  masked too, and so is each of tuningfork's own keys, when it is eight characters or more,
  without a space or a quote. That is a net, not a promise. Recordings hold what your harness
  sent and received — prompts, tool output, file contents — so read a recordings folder before
  you share one: [setup](docs/setup.md#before-you-share-one) says what to delete and where to
  look.

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
- **Deleting a run does not take what records it, or what was made of it.** A job keeps its
  record of the work it did; a chain its record and the skill each link left, which you remove by
  hand; a refinement that cites the run what it quoted; a scenario made from the run its
  conversation and input files; and a skill added from it its copy. Each goes only as itself.
