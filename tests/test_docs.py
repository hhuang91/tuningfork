"""The reference pages keep up with the CLI: a new command fails here until docs/commands.md
names it, and every relative link in the pages a newcomer reads leads somewhere."""

from __future__ import annotations

import re
from pathlib import Path

import typer.main

from tuningfork.cli import app

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "docs" / "commands.md"
#: The pages a newcomer reads first, and what they link to. The README's screenshots are added
#: by hand (docs/images/README.md lists them), so an image may be missing; a page may not.
PAGES = ("README.md", "docs/setup.md", "docs/commands.md", "docs/agents.md", "docs/images/README.md",
         "examples/find-the-file/README.md")


def command_paths(group, prefix: str = "") -> list[str]:
    """Every command as it is typed after `tuningfork`: `batch`, `chain run`, `connections proposer set`.
    A group is known by its commands: Typer brings its own copy of click, whose Group is not
    click's."""
    paths = []
    for name, command in group.commands.items():
        if command.hidden:
            continue
        path = f"{prefix}{name}"
        if getattr(command, "commands", None) is not None:
            paths.extend(command_paths(command, f"{path} "))
        else:
            paths.append(path)
    return paths


def test_every_command_is_in_the_command_reference():
    text = COMMANDS.read_text(encoding="utf-8")
    paths = command_paths(typer.main.get_command(app))
    assert {"batch", "chain run", "connections proposer set", "trash empty"} <= set(paths)
    # Named as code: on its own (`chain run`) or with its arguments (`trash run <id>…`).
    missing = [path for path in paths if f"`{path}`" not in text and f"`{path} " not in text]
    assert not missing, f"docs/commands.md does not name: {', '.join(missing)}"


def links(page: Path):
    fenced = False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        yield from re.findall(r"\]\(([^)\s]+)\)", line)
        yield from re.findall(r'(?:src|srcset)="([^"]+)"', line)


def test_every_relative_link_leads_to_a_file():
    broken = []
    for name in PAGES:
        page = ROOT / name
        for target in links(page):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = (page.parent / target.partition("#")[0]).resolve()
            if not path.exists() and not (path.parent.name == "images" and path.suffix in {".png", ".gif"}):
                broken.append(f"{name}: {target}")
    assert not broken, "broken links:\n" + "\n".join(broken)
