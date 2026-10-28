# Contributing

Thank you for looking. tuningfork is young, and issues that say what you ran, what you
expected and what happened instead are as welcome as code.

## Setting up

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/) and git. On Windows, work
inside WSL, with the checkout on the Linux file system rather than under `/mnt/c`. On a Mac, work
inside a Lima VM, with the checkout in the VM's own home ([how](docs/setup.md#on-macos)).

```bash
git clone https://github.com/hhuang91/tuningfork.git
cd tuningfork
uv sync --all-groups
uv run pytest
```

## The test suite is the gate

- **It runs with no GPU and no model**, against a scriptable mock server: every pull request
  must keep it so. A test that needs real inference goes behind the `real_server` marker,
  which runs only when `TUNINGFORK_REAL_SERVER` names a llama.cpp server. The browser tests in
  `tests/browser/` are opt-in too.
- **Check pytest's exit code, not only its count.** Warnings are errors, and an unclosed
  SQLite connection fails the run at teardown after every test has passed.
- CI runs it on Ubuntu with Python 3.12, 3.13 and 3.14.

## How the code is laid out

`CLAUDE.md` describes every part, and is kept current. The rules that matter most in review:

- **Every operation lives once, in `src/tuningfork/ops/`**: a function with typed parameters,
  progress events, cancellation, and a result that knows its JSON form. The CLI, the HTTP API
  behind the web UI and the MCP server are thin front ends over it. Logic a second front end
  would need does not belong in `cli.py`, `api/` or `ui/`.
- **Knowledge of a harness or a server lives behind its seam**: `harness/`, `backends/` and
  `wire/`. Supporting another harness or inference server must not require editing anything
  under `proxy/`. [`docs/adapters/TEMPLATE.md`](docs/adapters/TEMPLATE.md) is the place to
  start a harness adapter.
- **A rate is never shown without its interval** (`analysis/stats.py`).
- **The web UI puts every string on the page as a text node**: payloads are full of
  markup-looking text, and `tests/test_ui.py` fails on any HTML-parsing sink. Only
  `ui/static/js/api.js` talks to the server or touches the token.
- **Every endpoint of tuningfork's own lives under `/_tuningfork/`**, where the guard checks
  its Host, Origin and token (`api/guard.py`), never a route.
- **Nothing of one machine goes in a tracked file**: real paths, hostnames, ports that serve
  something else. Keep them in `CLAUDE.local.md` or another `*.local.md`, which git ignores.

## Pull requests

Say what changed, why, and how you checked it — the commands you ran, on what model and
server if it touched behaviour. Keep a refactor and the fixes it exposed in separate commits.
