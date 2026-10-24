# The web UI

`tuningfork serve` serves a web UI at the address it prints, and `tuningfork ui` serves the same
pages, read-only, over recordings with no proxy running. Nearly everything the commands do is in
it, and it can start the work as well as show it: a batch of runs, a resample, an A/B, a fork, a
refinement. `tuningfork demo` serves it over made-up runs, with no GPU.

The page changes nothing without the proxy's access token, which is in the address `serve`
prints; `tuningfork token` prints it again.

![The front page: where you left off, what is set up, and the loop from a skill to its next version](images/ui-guide.png)

<table>
  <tr>
    <td width="50%"><img src="images/ui-run.png" alt="A step's exact context: every message as the model saw it, the skill marked in it, and the call it made"></td>
    <td width="50%"><img src="images/ui-divergence.png" alt="The divergence page: the step that best predicts whether a run passes, then where a task's runs split and how each behaviour fared"></td>
  </tr>
  <tr>
    <td>A step's exact context: every message as the model saw it, the skill marked line by line, and the call it made</td>
    <td>The step that best predicts the verdict, then where a task's runs stop agreeing and how the runs that took each path fared</td>
  </tr>
  <tr>
    <td width="50%"><img src="images/ui-ab.png" alt="An A/B result: the edit it tested, each arm's rate with its interval, and Fisher's test"></td>
    <td width="50%"><img src="images/ui-tokens.png" alt="An answer as tokens, shaded by how unsure the model was, one token's alternatives open"></td>
  </tr>
  <tr>
    <td>An A/B at a step: the edit it tested, each arm's rate with its interval, and the test of the target declared before sampling</td>
    <td>An answer as the model's tokens, shaded by how unsure it was; a click shows what else it weighed there</td>
  </tr>
  <tr>
    <td width="50%"><img src="images/ui-refine.png" alt="A refinement: its target and its four stages in a line, then the evidence — each proposal's screen and the approved one's validation — and a section for each stage"></td>
    <td width="50%"><img src="images/ui-live.png" alt="The Live page: a call held at a breakpoint, its tool call edited in place"></td>
  </tr>
  <tr>
    <td>A refinement: the target declared first and its four stages in a line, then each proposal's screen and the full runs behind the one approved</td>
    <td>A call held at a breakpoint, edited in place before the harness gets it</td>
  </tr>
  <tr>
    <td colspan="2"><img src="images/ui-scenario.png" alt="Making a scenario from a chat had elsewhere: its first message the starting prompt, and each step the chat took beside what each run of the scenario does there"></td>
  </tr>
  <tr>
    <td colspan="2">A chat had elsewhere, imported and made a scenario: each run is sent the person's message with the conversation before it, and the skill comes in live</td>
  </tr>
</table>

[`images/README.md`](images/README.md) says what each picture shows, and how to take them again.
