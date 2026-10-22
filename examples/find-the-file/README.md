# A step where the model wavers

This folder holds a toy skill made to fail one way, and the runs behind the figure at the top of
[the README](../../README.md): Hermes Agent on Gemma 4 31B. [The step-by-step guide's](../README.md)
toy shows a model that is sure and wrong. This one shows a model that wavers: the step where its
runs part ways, how often it goes each way there, where its reasoning forks, and one line of the
skill that tips it.

Every output below is from real runs of these commands: one series made on 2026-10-16, and the
probe in section 4, made on 2026-10-22 against the same server. Paths are shortened to
`/home/you`, and long outputs are trimmed where marked `...`. Your ids will differ, and since every
number is a sample, yours may too.

## What is in this folder

| File | What it is |
|---|---|
| [`evals.json`](evals.json) | The task: summarize `data.csv` into `report.json`, with `data.csv` in an `inputs/` folder |
| [`tally-report/SKILL.md`](tally-report/SKILL.md) | The skill, as the runs were given it |
| [`fix/SKILL.md`](fix/SKILL.md) | The same skill with one line changed: the edit tested below |

The skill is the guide's tally-report skill with its key names fixed, and one sentence added to
its first step: "If it is not there, say so and stop: never guess the numbers." The request names
`data.csv`, but the file is `inputs/data.csv`. A run that reads the named path gets `File not
found: data.csv`, and the skill tells it to stop. A run that searches finds the file.

That sentence is planted, as the guide's toy leaves out the key names: without it, the model
recovers. Ten runs without it passed 8 times [49%–94%]. After the failed read every run searched
for the file; the two that failed had begun with a script that wrote a report too early, and ran
out of turns.

The fix lives outside the skill's folder on purpose: every run is given a copy of that folder, and
a fix kept beside `SKILL.md` is one the model could read.

## What you need

What [Part 2 of the guide](../README.md#part-2--the-real-experiment) needs, with Hermes Agent:
llama-server serving Gemma 4 31B, started as [§1](../README.md#1-start-the-model-server) says;
Hermes Agent 0.18.2; and bubblewrap. Then, in a terminal of its own, a proxy recording into a
folder for this example:

```bash
export TUNINGFORK_HOME=~/tuningfork-find
tuningfork serve --upstream http://127.0.0.1:8081 --max-output 8192
```

And in another, from the root of your checkout:

```bash
cd examples/find-the-file
export TUNINGFORK_HOME=~/tuningfork-find
```

Under WSL, `export TUNINGFORK_SANDBOX_HIDE=/mnt/c` in both terminals keeps the Windows drive out of
the runs' sight, this folder's fix included.

## 1. Record ten runs

```bash
tuningfork batch evals.json -n 10 --harness hermes --model gemma-4-31b
```

```
find-the-file/1 — 10 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if d.get('row_count')==4 
and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  tally-report
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  47.2s
  run 1  fail  36.1s
  run 2  fail  16.6s
  run 3  fail  17.9s
  run 4  pass  47.2s
  run 5  pass  49.4s
  run 6  pass  44.7s
  run 7  fail  24.3s
  run 8  fail  16.3s
  run 9  fail  18.5s

4/10  40%  [17%-69%] passed
...
```

Four in ten. The rest of the output finds the runs diverging at their first step: eight read
`data.csv`, and two ran a script that opened it. It ends with the tool that goes with passing:
the runs that called `search_files` passed 4 of 4, the runs that never did 0 of 6.

## 2. Find where they part ways

The eight runs that read the named path first all did the same thing at step 0. Compare them from
there:

```bash
READS=$(tuningfork runs -q --called read_file --at-step 0)
tuningfork divergence $READS
```

```
8 runs of selected runs
  agreed for 1 step(s), diverged at step 1

step 1
   count  rate  95% CI     behaviour                                                     then passed          
A  4/8    50%   [22%-78%]  search_files(pattern='data.csv', target='files')              4/4  100%  [51%-100%]
B  1/8    12%   [2%-47%]   I could not find the file `data.csv` in the current           0/1  0%  [0%-79%]    
                           directory. As per … (+78 chars)                                                    
C  1/8    12%   [2%-47%]   The file `data.csv` was not found, so I cannot generate the   0/1  0%  [0%-79%]    
                           report.                                                                            
D  1/8    12%   [2%-47%]   The file `data.csv` was not found, so I cannot proceed with   0/1  0%  [0%-79%]    
                           the summar… (+2 chars)                                                             
E  1/8    12%   [2%-47%]   The file `data.csv` was not found, so I cannot summarize the  0/1  0%  [0%-79%]    
                           data.                                                                              

by tool  runs that called search_files passed 4/4, runs that never did 0/4 (Fisher p = 0.03): the tool goes 
with passing, at whatever step each run called it

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82 runs per 
arm.
```

At step 1, after the read came back empty, half searched for the file and passed. Half gave up
("The file `data.csv` was not found, so I cannot…") and failed. The two runs that began with a
script gave up the same way.

## 3. Bad luck, or the wording? Resample that step

Take the newest run that gave up, and look at its step 1:

```bash
RUN=$(tuningfork runs -q -n 1 --failed --called read_file --at-step 0)
tuningfork step $RUN 1
```

```
run e2e09d5d step 1
...
context
  [0] system: You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, 
knowled…
  [1] user: Use the tally-report skill to summarize data.csv into report.json.
  [2] assistant: tool_calls(read_file)
  [3] tool: {"content": "", "total_lines": 0, "file_size": 0, "truncated": false, "is_binary": false, 
"is_image"…

response
  reasoning_content: The user wants me to use the `tally-report` skill to summarize `data.csv` into 
`report.json`.
The `tally-report` skill instructions are:
1. Read the CSV file named in the request. If it is not there, say so and stop: never guess the numbers.
2. Work out how many line items it contains, and the total… (+186 chars)
  content: I could not find the file `data.csv` in the current directory. As per the tally-report skill 
instructions, I cannot proceed without the source data.
```

The tool result ends `"error": "File not found: data.csv"` (`--full` prints it whole). Every run
that read first reached this step with this context: the request, the read, and its result. Send
it to the model twenty more times, everything before it held fixed:

```bash
tuningfork resample $RUN 1 -n 20
```

```
run e2e09d5d step 1 — 20 samples from http://127.0.0.1:8081
  context   4 messages, after the result of read_file
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 13,921 tokens, then answers: the recorded step took 7.9s

    1/20  search_files(pattern='data.csv', target='files')  7.4s
    2/20  The file `data.csv` was not found, so I cannot proceed with the … (+8 chars)  7.0s
    ...
   20/20  The file `data.csv` was not found, so I cannot proceed with the … (+8 chars)  9.9s

step 1, after the result of read_file — 20 samples
   count  rate  95% CI     behaviour                                                                          
A  6/20   30%   [15%-52%]  The file `data.csv` was not found, so I cannot proceed with the summary.           
B  4/20   20%   [8%-42%]   search_files(pattern='data.csv', target='files')                                   
C  3/20   15%   [5%-36%]   The file `data.csv` was not found, so I cannot generate the report.                
D  3/20   15%   [5%-36%]   The file `data.csv` was not found, so I cannot proceed with the tally report.      
E  1/20   5%    [1%-24%]   The file `data.csv` was not found in the current directory. As per the             
                           `tally-report` skil… (+57 chars)                                                   
F  1/20   5%    [1%-24%]   The file `data.csv` was not found in the working directory. As per the             
                           `tally-report` skil… (+57 chars)                                                   
G  1/20   5%    [1%-24%]   The file `data.csv` was not found. As per the `tally-report` skill instructions, I 
                           cannot … (+32 chars)                                                               
H  1/20   5%    [1%-24%]   The file `data.csv` was not found. As per the tally-report skill instructions, I   
                           cannot pr… (+30 chars)                                                             

experiment a9250cc5 · 2.7 min · 8.1s per sample · use `tuningfork ab e2e09d5d 1 --target <letter> --variant 
<file>` to test a wording change
```

It searched 4 times in 20, and gave up 16, in seven wordings. This isn't one run's bad luck: the
model hasn't made up its mind at this step.

## 4. Where its reasoning forks

Score the step-1 answer of the newest run that searched, for the model's own odds at each token:

```bash
PASS=$(tuningfork runs -q -n 1 --passed --called read_file --at-step 0)
tuningfork tokens $PASS 1 --score
```

```
scoring step 1's answer (292 tokens) on http://127.0.0.1:8081…
run 472253d6 step 1  stream 8a822bad
  scored · grammar · top 8 · 292 tokens · rebuilt exactly
  choice of tool, at token [272]: search_files 100%  the model's own chance of each, under the step's sampler

reasoning
  The user wants me to use the `tally-report` skill to summarize `data.csv` into `report.json`.
  The `tally-report` skill's instructions are:
  1. Read the CSV file named in the request. If it is not there, say so and stop.
  2. Work out how many line items it contains, and the total of the amount column.
  3. Write the results to `report.json` in the working directory.
  
  I tried to read `data.csv` but it was not found.
  Wait, I should check if it's in the current directory or if I should search for it first?
  Actually, the skill says: "Read the CSV file named in the request. If it is not there, say so and stop: 
never guess the numbers."
  The request specifically said "summarize data.csv".
  The file `data.csv` is missing in the current working directory.
  
  However, as an intelligent agent, I might want to check if it's somewhere else or if the user made a 
mistake. But the skill is quite explicit: "If it is not there, say so and stop".
  
  Let me double check the directory contents just in case I'm missing something obvious.
tool call
  call:search_files{pattern:<|"|>data.csv<|"|>,target:<|"|>files<|"|>}

where the step's sampler could most easily have drawn another token
  [141] reasoning ' first' 9% | '.' 91%  H 0.48 bits  runner-up drawn 91%
  [213] reasoning ' might' 13% | ' should' 87%  H 0.59 bits  runner-up drawn 87%
  [144] reasoning 'Actually' 13% | 'The' 86%  H 0.59 bits  runner-up drawn 87%
  [121] reasoning 'Wait' 7% | 'According' 80% | 'Following' 5%  H 1.17 bits  runner-up drawn 86%
  ...
```

By the time the model wrote the call, the tool was settled: `search_files`, 100%. The choice came
earlier, in the reasoning. After "I tried to read `data.csv` but it was not found.", the next word
was "According" at 80%, or "Wait" at 7% (token [121]). The run that gave up drew "According": its
reasoning went on "According to the skill, I should say so and stop." This run drew "Wait", went
on "…or if I should search for it first?", argued with itself, and searched.
(`tuningfork tokens $RUN 1 --score` shows the run that gave up.)

Does that one word change what the model does? Force the likeliest alternative there,
"According", and let the model write the rest of its answer from it. The run's own "Wait" gets as
many continuations: forty each, with a search by any pattern counted.

```bash
tuningfork forking-tokens $PASS 1 --at 121 --alternatives 1 -n 40 --target-match search_files
```

```
forking tokens at run 472253d6 step 1 — after the result of read_file
  base      stream 8a822bad (scored, 292 tokens)
  target    behaviour matching /search_files/ (declared before sampling)
  arms      2 forced tokens × 40 = 80 continuations
  power     a forced token is found to change the outcome when about 6 of its 40 continuations change; smaller
shifts need more (-n)
  ...
  [121] reasoning 'Wait' 7% → 'According' 80%

  121:'According'  1/40  The file `data.csv` was not found, so I cannot proceed with the … (+8 chars)  5.3s
  ...
  121:'Wait'  1/40  search_files(pattern='data.csv', target='files')  6.9s
  ...

target (declared before sampling): behaviour matching /search_files/
  the baseline arms reached it 30/40  75% [60%-86%]

[121] reasoning  … `data.csv` but it was not found.
⟦Wait⟧
  as run  'Wait' 7%                30/40  75% [60%-86%]
  forced  'According' 80%          0/40   0% [0%-9%]   p=1.9e-13 → Holm 1.9e-13  flips away from the target
  the chance of the target from here, weighted by each token's chance: 6%

behaviours, across the probe
   count  rate  95% CI     target      behaviour                                                              
A  29/80  36%   [27%-47%]  target      search_files(pattern='data.csv', target='files')                       
B  25/80  31%   [22%-42%]  not target  The file `data.csv` was not found, so I cannot proceed with the        
                                       summary.                                                               
...
V  1/80   1%    [0%-7%]    target      search_files(pattern='*.csv', target='files')                          
  the tokens whose continuations reached each, a line a position:
  A  [121] 'Wait' as run 29
  B  [121] 'Wait' as run 2, 'According' 23
  ...

1 of 1 forced tokens changed what the model did (Holm-corrected over the 1 planned before sampling; α=0.05).

experiment 1aed5238 · 9.0 min · `tuningfork experiment 1aed5238` shows it again
```

It does. After "Wait", 30 continuations in 40 searched for the file; after "According", none.
Putting "According" in its place changes what the model goes on to do: "Wait" is a forking token.
"According", the likely word there at 80%, never led to a search. "Wait" made one likely without
settling it: after it, 10 continuations in 40 still ended without a search. This is a step where
the model wavers, the kind a few words of the skill can tip.

An earlier probe of this token (d39bc626, made with the series on 2026-10-16), at the default of
ten continuations a word and counting only the base's exact call, saw that call 4 times in 10 after
"Wait" and once after "According". That is a difference of 3 in 10, and the smallest difference
ten a word can show is about 6. The probe's `power` line says what its `-n` can see.

## 5. Test one line at that step

The edit makes searching the first thing the skill's first step says to do:

```bash
diff tally-report/SKILL.md fix/SKILL.md
```

```
12c12
< 1. Read the CSV file named in the request. If it is not there, say so and stop: never guess the numbers.
---
> 1. Find the CSV file named in the request with `search_files`, then read it. If it is not there, say so and stop: never guess the numbers.
```

Swap it in at the step and sample both wordings, twenty each, toward a target declared before
sampling: a call of `search_files`.

```bash
tuningfork ab $RUN 1 --variant "$PWD/fix/SKILL.md" \
  --target-rules '[{"if": "calls", "value": "search_files"}]' -n 20
```

```
run e2e09d5d step 1 — 20 samples per arm from http://127.0.0.1:8081
  context   4 messages, after the result of read_file
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  prompt    each sample processes its whole prompt of 13,921 tokens, then answers: the recorded step took 7.9s
  swapping  /home/you/tuningfork-find/batches/bfe3ca43/skill-snapshot/tally-report/SKILL.md → 
/home/you/tuningfork/examples/find-the-file/fix/SKILL.md
            found: full, lines 1-29, in message 0 (system) at chars 10720-11543
  target    a behaviour that calls search_files (declared before sampling)

  original   1/20  The file `data.csv` was not found. As per the `tally-report` ski… (+45 chars)  7.9s
  original   2/20  The file `data.csv` was not found, so I cannot proceed with the … (+8 chars)  7.4s
  ...
  variant    1/20  search_files(pattern='data.csv', target='files')  8.2s
  variant    2/20  search_files(pattern='data.csv', target='files')  7.5s
  ...

   behaviour                                               original             variant                p adj. 
A  search_files(pattern='data.csv', target='files')        4/20  20% [8%-42%]   20/20 100% [84%-100%]  1.9e-06
B  The file `data.csv` was not found, so I cannot proceed  6/20  30% [15%-52%]  0/20   0% [0%-16%]     0.242  
   wit… (+14 chars)                                                                                           
...
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 12 comparisons

target (declared before sampling): a behaviour that calls search_files
  original  4/20  20% [8%-42%]
  variant   20/20 100% [84%-100%]
  Fisher exact p = 1.5e-07 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 1 model-written message(s) before this step were generated under the original wording, and both arms 
inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
experiments 0d2540ea (original) and 5891a3c5 (variant) · 5.4 min
```

tuningfork found the skill where Hermes preloads it, the whole file in the system prompt, and
swapped in the edit there. As written, the model searched 4 times in 20; with the edit, 20 in 20.
That's one exact test, declared before sampling: p = 1.5 × 10⁻⁷.

## 6. Confirm with full runs

The A/B held the failed read fixed; full runs give the edit from the first step:

```bash
tuningfork batch evals.json -n 10 --harness hermes --model gemma-4-31b --skill fix/SKILL.md
```

```
find-the-file/1 — 10 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if d.get('row_count')==4 
and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  tally-report  SKILL.md replaced by fix/SKILL.md
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  31.0s
  run 1  pass  34.7s
  ...
  run 9  pass  35.4s

10/10  100%  [72%-100%] passed

10 runs of find-the-file/1
  agreed for 3 step(s), diverged at step 3
...
```

Ten in ten, against four in ten before: Fisher p = 0.011. All ten took the same three steps:
search for the file, read `inputs/data.csv`, write the report. They differed only in how they
said so. `tuningfork tasks` keeps the two wordings apart:

```bash
tuningfork tasks
```

```
find-the-file/1 · hermes · gemma-4-31b
 skill     runs  check passed                                    
 228c3f01  10    4/10  40%  [17%-69%]                            
 bce8f76c  10    10/10  100%  [72%-100%]  SKILL.md from SKILL.md 

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

## How this page was measured

One series, on 2026-10-16, and the probe in section 4, on 2026-10-22:
- **Server:** llama-server b11108 serving Gemma 4 31B (Q4_K_M), started as the guide's §1 says,
  both times.
- **tuningfork:** 0.2.1 for the series, its proxy run with `--max-output 8192`; 0.4.0 for the
  probe, which goes straight to the server.
- **Harness:** Hermes Agent 0.18.2, every run in the sandbox.
- **Before it:** the pilot without the planted sentence, described above. It was recorded in the
  same folder and moved to the trash afterwards, so no rate or comparison here counts it.
- **Timings:** the ten runs took about five minutes, the resample 2.7, the probe 9.0, the A/B 5.4,
  and the ten runs with the edit about five and a half.
