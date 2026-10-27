"""The web UI's static files: served correctly, and never parsing payload text as HTML.

The page shows text written by a model and a harness — prompts, tool arguments, file
contents — which is full of markup-looking text. The UI's rule is that every string reaches
the page as a text node. That is enforced here structurally, because a single innerHTML
with a payload in it would let a recorded prompt run script in the user's browser.
"""

from __future__ import annotations

import re

import httpx

from tuningfork.proxy.app import create_browse_app
from tuningfork.ui import STATIC

UI = "/_tuningfork/ui"
#: Every DOM API that parses a string as HTML, or evaluates one as code.
HTML_SINKS = re.compile(
    r"\.(innerHTML|outerHTML|insertAdjacentHTML)\b|document\.write|\beval\(|new Function\(|"
    r"setTimeout\(\s*[\"'`]|createContextualFragment"
)


def scripts() -> list:
    return sorted(STATIC.rglob("*.js"))


def test_the_ui_has_scripts_to_check():
    assert len(scripts()) >= 5


def test_no_script_parses_strings_as_html():
    offenders = [
        f"{path.relative_to(STATIC)}:{number}: {line.strip()}"
        for path in scripts()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if HTML_SINKS.search(line)
    ]
    assert offenders == []


def test_every_module_import_resolves_to_a_file():
    imports = re.compile(r"""from\s+["'](\.{1,2}/[^"']+)["']""")
    missing = []
    for path in scripts():
        for target in imports.findall(path.read_text(encoding="utf-8")):
            if not (path.parent / target).resolve().is_file():
                missing.append(f"{path.relative_to(STATIC)} imports {target}")
    assert missing == []


def test_the_page_loads_nothing_from_elsewhere():
    """It is served locally and must work offline: no CDNs, fonts or trackers."""
    remote = re.compile(r"""(src|href)=["']https?://""")
    for path in [STATIC / "index.html", *scripts(), *STATIC.rglob("*.css")]:
        assert not remote.search(path.read_text(encoding="utf-8")), path.name


async def test_assets_are_served_with_types_a_browser_accepts(store):
    """A module script served as text/plain — the Windows registry's idea of .js — is
    refused by the browser, which leaves a blank page."""
    app = create_browse_app(store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8788"
    ) as client:
        page = await client.get(f"{UI}/")
        script = await client.get(f"{UI}/js/app.js")
        style = await client.get(f"{UI}/app.css")
    assert page.status_code == 200
    assert 'src="js/app.js"' in page.text
    assert script.headers["content-type"].startswith("text/javascript")
    assert style.headers["content-type"].startswith("text/css")


def source(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def lines_matching(pattern: str) -> list[str]:
    found = re.compile(pattern)
    return [
        str(path.relative_to(STATIC)).replace("\\", "/")
        for path in scripts()
        if found.search(path.read_text(encoding="utf-8"))
    ]


class TestTheAccessToken:
    """The token lets the page change things, and from Phase 5 start work that runs shell
    commands. Only one module handles it, and it never outlives the tab."""

    def test_only_the_api_module_talks_to_the_server(self):
        assert lines_matching(r"\bfetch\(") == ["js/api.js"]
        assert lines_matching(r"x-tuningfork-token") == ["js/api.js"]

    def test_the_token_is_kept_for_this_tab_only(self):
        assert lines_matching(r"\bsessionStorage\.") == ["js/auth.js"]
        # localStorage outlives the tab: it holds the theme, and nothing else.
        assert lines_matching(r"\blocalStorage\.") == ["js/app.js"]
        assert "tuningfork.theme" in source("js/app.js")
        assert "token" not in "".join(
            line for line in source("js/app.js").splitlines() if "localStorage" in line
        )

    def test_the_token_leaves_the_address_bar(self):
        assert "history.replaceState" in source("js/auth.js")

    def test_every_control_is_marked_and_guarded(self):
        live = source("js/views/live.js")
        assert live.count('"data-control": true') >= 6
        assert live.count("guard(") >= 6


class TestThePageRunsOnlyItsOwnCode:
    """The policy (`api/guard.UI_POLICY`) forbids inline code; the page must not need any."""

    def test_the_page_has_no_inline_script_handler_or_style(self):
        page = source("index.html")
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", page)
        assert not re.search(r"\son[a-z]+\s*=", page)
        assert not re.search(r"\sstyle\s*=", page)

    def test_no_script_makes_a_handler_from_text(self):
        assert lines_matching(r"setAttribute\(\s*[\"'`]on") == []
        assert lines_matching(r"\.on[a-z]+\s*=\s*[\"'`]") == []

    def test_h_refuses_a_handler_that_is_not_a_function(self):
        assert 'throw new TypeError(`${key} must be a function`)' in source("js/dom.js")


class TestStartingWork:
    """Work can be started from where a person looks for it — the Jobs and Runs pages — and
    every such control is marked, so a page without the token shows it disabled."""

    def test_the_jobs_page_has_a_start_button_and_the_runs_page_a_run_a_task_button(self):
        jobs = source("js/views/jobs.js")
        assert re.search(r'"data-control": true, onclick: guard\(.{0,300}?"Start…"', jobs, re.S)
        runs = source("js/views/runs.js")
        assert re.search(r'href: "#/batch", "data-control": true \}, "Run a task…"', runs)

    def test_the_launcher_lists_what_the_proxy_says_it_can_run(self):
        """Its kinds come from `GET /jobs/kinds`: a kind added there appears without UI work."""
        launcher = source("js/launcher.js")
        assert 'api.get("/jobs/kinds")' in launcher
        assert "required(kind).includes(\"run\")" in launcher

    def test_running_times_tick_in_the_page(self):
        """A running job's time moves each second without asking the proxy anything."""
        assert "scope.every(() => tick(" in source("js/views/jobs.js")
        assert "since(job.started_at)" in source("js/jobs.js")

    def test_a_steps_answer_can_be_scored_and_its_tokens_probed_where_it_is_read(self):
        """The timeline's Tokens view scores an answer through the tokens job — a marked,
        guarded control — and hands selected tokens to the forking-token form."""
        tokens = source("js/tokens.js")
        assert re.search(r'"data-control": true, onclick: guard\(score\) \}, "Score this answer"', tokens)
        assert 'kind: "tokens"' in tokens
        timeline = source("js/timeline.js")
        assert "tokensPanel({" in timeline and "start=forking-tokens&positions=" in timeline
        start = source("js/start.js")
        assert 'kind: "forking-tokens"' in start and '"Find forking tokens…"' in start

    def test_every_token_is_a_text_node_with_one_listener(self):
        """Thousands of tokens: a class for the heat, and one delegated handler for them all."""
        tokens = source("js/tokens.js")
        assert 'closest?.(".tk-t")' in tokens
        assert tokens.count("addEventListener") == 0

    def test_a_check_can_be_tried_where_it_is_written(self):
        """A check is tried where it is written — a scenario's editor, a scenario made from a
        run, which choose it with the same picker — through the check-trial job: a marked,
        guarded control, as any that starts work."""
        tried = source("js/checktry.js")
        assert re.search(r'"data-control": true,\s*onclick: guard\(.{0,1200}?"Test this check…"', tried, re.S)
        assert 'kind: "check-trial"' in tried
        assert all("checkPicker(" in source(page) for page in ("js/views/scenarios.js", "js/views/import.js"))

    def test_both_ways_to_make_a_scenario_choose_its_skill_alike(self):
        """A new scenario and one made from a run name the skill alike and choose its folder with
        one picker — a skill the Skills page lists, or a path to one (the user's ask, 2026-10-25)."""
        for page in ("js/views/scenarios.js", "js/views/import.js"):
            assert "skillPicker({" in source(page) and "skillNameLabel()" in source(page), page
        picker = source("js/skillpick.js")
        assert '["known", "Existing skill"]' in picker and '["path", "Path to skill"]' in picker


class TestRefining:
    """A refinement starts from a step, proposes through the propose job and screens through
    the screen job — each a plan shown before it starts — and the proposer's key goes in and
    never comes back out."""

    def test_a_step_offers_it_and_the_page_has_its_place(self):
        assert '"Refine the skill here…"' in source("js/start.js")
        assert 'kind === "refine"' in source("js/start.js")
        assert 'data-nav="refine"' in source("index.html")
        app = source("js/app.js")
        assert "refinementsView" in app and "refinementView" in app

    def test_proposing_screening_and_validating_are_planned_jobs(self):
        refine = source("js/views/refine.js")
        # The screen form plans a screen at the step or one whose forks run to the end, as chosen.
        assert 'kind: "propose"' in refine and 'kind: "validate"' in refine
        assert '"screen-to-end" : "screen"' in refine
        assert refine.count("planForm({") == 3
        # A person's own wording, closing, approving and repairing change state: marked and guarded.
        assert refine.count('"data-control": true') >= 7 and refine.count("guard(") >= 5

    def test_approving_shows_what_is_written_and_takes_the_typed_code(self):
        refine = source("js/views/refine.js")
        # The plan is read first; the write is a POST carrying the code the person typed.
        assert "/approve${query(" in refine and 'api.post(`/refinements/${enc(shown.refinement.id)}/approve`' in refine
        assert 'input("confirm"' in refine and "confirm: code.value.trim()" in refine
        assert "target.windows" in refine and "plan.line_endings" in refine and "diffView(plan.diff.hunks)" in refine
        assert "gateList(plan.gate)" in refine and "/repair`" in refine
        # Validations are drawn as the person's own view: held-out rows, the gate's every check.
        assert "counts.held_out" in refine and "check.low_powered" in refine
        assert 'validate: "Full test runs of an edit"' in source("js/jobs.js")

    def test_the_key_is_written_and_never_shown(self):
        connections = source("js/views/connections.js")
        assert 'type: "password", name: "proposer_key"' in connections
        assert 'secret.value = "";' in connections
        # The page is told whether a key is set, and from where: nothing reads a key back.
        assert not re.search(r"key\.(key|value|secret)\b", connections)

    def test_deleting_what_a_refinement_cites_says_so_before_the_second_click(self):
        assert "citedText(counts)" in source("js/trash.js")
        assert "citedText(goes)" in source("js/views/trash.js")

    def test_a_refinement_goes_to_the_trash_and_its_page_says_so(self):
        refine = source("js/views/refine.js")
        assert 'trashButton({ kind: "refinement"' in refine and 'trashedBanner({\n            kind: "refinement"' in refine
        # Nothing new starts on one in the trash: every form waits on `acting()`, which a
        # refinement in the trash — `settled()` — never is; nor has it a current stage.
        assert "Boolean(shown.refinement.trashed_at)" in refine and 'live.mode === "live" && !settled()' in refine
        assert "settled() ? null : currentOf(stages)" in refine
        page = source("js/views/trash.js")
        assert "found.refinements" in page and 'row("refinement"' in page
        assert '["refinements", "refinement"]' in source("js/trash.js") and "goes?.notes" in source("js/trash.js")


class TestReasoning:
    """A step the model answered without reasoning, marked wherever a step is shown; how many
    of a behaviour's samples reasoned; and a breakpoint that stops there."""

    def test_the_marker_and_the_counts_are_on_every_page_that_shows_them(self):
        assert "export function noReasoning(step)" in source("js/payload.js")
        assert "noReasoning(step)" in source("js/timeline.js") and "noReasoning(step)" in source("js/views/step.js")
        assert "reasonedText(cluster.reasoned" in source("js/results.js")
        assert "reasonedText(shift.original_reasoned" in source("js/results.js")
        assert 'name: "no_reasoning"' in source("js/views/live.js")
