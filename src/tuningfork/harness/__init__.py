"""Harness adapters. Harness-specific knowledge lives here and nowhere else."""

from tuningfork.harness.base import (
    GenericHarness,
    HarnessAdapter,
    HarnessError,
    LaunchSpec,
    Requirement,
    SkillSpan,
)
from tuningfork.harness.claude_code import ClaudeCodeAdapter
from tuningfork.harness.hermes import HermesAdapter
from tuningfork.harness.openclaw import OpenClawAdapter

#: Adapters by name, for CLI selection and for contributors adding their own.
ADAPTERS = {
    "hermes": HermesAdapter,
    "claude-code": ClaudeCodeAdapter,
    "openclaw": OpenClawAdapter,
    "generic": GenericHarness,
}

__all__ = [
    "ADAPTERS",
    "ClaudeCodeAdapter",
    "GenericHarness",
    "HarnessAdapter",
    "HarnessError",
    "HermesAdapter",
    "LaunchSpec",
    "OpenClawAdapter",
    "Requirement",
    "SkillSpan",
]
