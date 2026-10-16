# H's chain, run by run

Hermes's own loop (arm H): thirty runs of the task in a row, each with the skill the run before
left. After each run Hermes was told the check's verdict as its next turn ("I ran my check on what
you produced. It passed." or "It failed."), and its background review could rewrite the skill
(`tuningfork chain run evals.json -n 30 --tell -o await_review=true -o review_every_turn=true`).
239 minutes, 12,404 s of model calls; 20 of the 30 runs passed, and the skill was rewritten after
15 of them. The last column is what the review changed after that run.

| run | verdict | minutes | the skill after it |
|---|---|---|---|
| 0 | fail | 5.6 | SKILL.md |
| 1 | fail | 18.3 | unchanged |
| 2 | fail | 9.4 | SKILL.md |
| 3 | fail | 6.1 | SKILL.md |
| 4 | pass | 4.7 | unchanged |
| 5 | pass | 4.2 | unchanged |
| 6 | fail | 7.6 | SKILL.md |
| 7 | pass | 8.7 | SKILL.md, references/layout-patterns.md, references/service-mapping.md |
| 8 | fail | 9.5 | SKILL.md |
| 9 | pass | 8.2 | SKILL.md, templates/aws-web-app-layout.html |
| 10 | pass | 4.5 | unchanged |
| 11 | pass | 12.7 | templates/aws-web-app-standard.html |
| 12 | fail | 13.0 | SKILL.md |
| 13 | pass | 8.1 | SKILL.md |
| 14 | pass | 4.9 | unchanged |
| 15 | pass | 18.7 | unchanged |
| 16 | fail¹ | 11.9 | SKILL.md |
| 17 | pass | 6.7 | SKILL.md |
| 18 | pass | 5.0 | unchanged |
| 19 | pass | 5.3 | unchanged |
| 20 | timed out (not counted) | 15.0 | unchanged |
| 21 | pass | 6.3 | SKILL.md, references/common-patterns.md |
| 22 | fail | 9.9 | SKILL.md |
| 23 | pass | 4.5 | unchanged |
| 24 | pass | 5.0 | unchanged |
| 25 | pass | 4.4 | unchanged |
| 26 | pass | 6.7 | SKILL.md |
| 27 | pass | 4.4 | unchanged |
| 28 | pass | 4.8 | unchanged |
| 29 | pass | 4.6 | unchanged |

¹ A fault in the check, found and fixed while the chain ran: it did not recognise "Application LB"
as the load balancer. Run 16's diagram passes the fixed check, but Hermes had been told "It
failed", and its review changed the skill in answer.

The skill the chain ended with is what H's ten fresh runs used: `diffs/H.diff` against the
original, and the five files its reviews added in `h-added/`.
