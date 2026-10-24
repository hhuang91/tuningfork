"""The MCP SDK is an optional extra (`tuningfork-lab[mcp]`): nothing but `tuningfork mcp` and
the proxy's MCP route may load it, so the rest of tuningfork works, and starts as fast,
without it."""

from __future__ import annotations

import subprocess
import sys

MODULES = (
    "tuningfork.cli",
    "tuningfork.proxy.app",
    "tuningfork.jobs",
    "tuningfork.ops.compact",
    # The MCP package itself, up to the moment it serves: its registry is plain Python.
    "tuningfork.mcp.schema",
    "tuningfork.mcp.tools",
    "tuningfork.mcp.backend",
    "tuningfork.mcp.server",
)


def test_tuningfork_runs_without_loading_the_mcp_sdk():
    code = (
        f"import sys; import {', '.join(MODULES)}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('mcp', 'mcp_types', 'pydantic')))"
    )
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert found.returncode == 0, found.stderr
    assert found.stdout.strip() == "[]"
