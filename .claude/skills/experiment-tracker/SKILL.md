---
name: experiment-tracker
description: How to run, record and report ML experiments in a repo that has an nb.json file, using the nb experiment tracker. Use this skill before launching any training or evaluation run, before changing an algorithm, loss, sampler or hyperparameter, when writing pseudocode or results, and when reporting what was run, even if the user only says "try X" or "rerun with Y". Also use it at the start of every session in such a repo, to read the reader's open comments, and when the user asks to set up or connect the experiment tracker.
---

# Experiment tracker (nb)

Every experiment in this repo is recorded on a website. The human reads that site,
not this chat, so anything that is not on the site is lost to them. `nb url` prints
its address.

Words used everywhere:

- **subproject**: one question, a set of related ideas (for example `time-sampling`).
- **experiment**: one training launch inside a subproject.
- **baseline**: the experiment the others of a subproject are compared with (the
  first one, unless changed with `nb sub NAME --baseline N`).

## Setting up a machine or a repository

Only needed once. Skip this if `nb doctor` already says "everything is ready".

1. If the `nb` command is missing, install it from the site (the user gives you
   the site address):

       curl -fsSL https://<site>/install.sh | bash

2. If `nb doctor` or any command says "not logged in", the user must provide the
   site key. Store it without echoing it:

       nb login https://<site>          # asks for the key, hidden
       NB_KEY=<key> nb login https://<site>   # when the user gave you the key

3. In the repository root run `nb setup`. It creates `nb.json` and the
   `pseudocode/` folder, installs these rules, and prints a list of what is
   still to do.
4. Work through that list (plots, headline metric, glossary, pseudocode,
   `@nb.core`, `nb.start`), running `nb doctor` after each change until it says
   "everything is ready". Then run `nb push`.

## What the reader sees, and your job

The reader looks at one page per project. It shows the project's main formula,
then one table with every result: one row per run, grouped by question, with at
most four numbers per row. Everything else (curves, images, pseudocode, settings)
is behind small buttons. Your job is to keep that page short, exact and readable
without any other context. Compact beats complete: update what is there instead
of adding to it.

- **Main formula** (`"formula"` in `nb.json`): the loss and the update or
  sampling rule that define the project, in 1 or 2 lines of math.
- **Table columns** (`"table"` in `nb.json`): at most 4 metric names. Give each
  metric's view a short `"label"` written with math, such as
  `"$\\mathrm{SWD}$ at $N = 4$"`, and use that exact name everywhere. One
  quantity, one name.
- **Plots**: at most 6 curve views. Keep only what the reader would look at
  every time.
- **One row per run**: `title` is 2 to 4 words naming the idea ("Logit-normal
  $t$", "Midpoint step"), with no numbers and no codes; `formula` states it
  exactly, with values written as `\cfg{name}`.
- **Seeds**: runs that differ only by their seed must have the same title,
  formula and config except `seed`. The page then shows them as one row with
  mean ± spread. Never put the seed in the title.
- **One answer per question**: `nb answer <subproject> "..."`, at most 300
  characters, numbers first. Rewrite it when results change; do not add a
  second one.
- **Fix names in place**: `nb retitle <number> "<title>" --formula "<formula>"`
  renames a run that is already on the page; no need to record it again.
- **Hide what no longer counts**: `nb hide <number>` for failed, aborted or
  superseded runs. Nothing is deleted.
- **Few questions**: add a run to an existing subproject when it tests the same
  question; open a new one only for a new question.

## The main rule: formulas before words

The reader is a mathematician. A short formula tells them exactly what happened;
a sentence describing the same thing is slower to read and usually ambiguous.
So whenever a step, a setting, a change or a result can be written as a short
formula, write the formula and keep at most a few words around it.

| Do not write | Write |
|---|---|
| "sample t from a logit-normal distribution with mean -0.4" | `$t = \sigma(z)$, $z \sim \mathcal{N}(-0.4, 1^2)$, $\sigma(z) = 1 / (1 + e^{-z})$` |
| "linearly interpolate between data and noise" | `$x_t = (1 - t) x + t \epsilon$` |
| "the loss is the mean squared error to the velocity target" | `$L = \frac{1}{B} \sum_{i=1}^{B} \lVert v_\theta(x_t^{(i)}, t^{(i)}) - (\epsilon^{(i)} - x^{(i)}) \rVert^2$` |
| "use a midpoint step instead of Euler" | `$x \leftarrow x - \Delta v_\theta(x_{mid}, t - \frac{\Delta}{2})$, $x_{mid} = x - \frac{\Delta}{2} v_\theta(x, t)$` |
| "cosine learning rate decay" | `$\eta_k = \frac{\eta}{2} (1 + \cos(\pi k / K))$` |
| "sample quality improved a lot" | `swd_4: 0.214 (#5) → 0.066 (#6)` |

This applies in every place you write: the experiment's `formula`, the
pseudocode, the glossary definitions, the "shows" text of a plot, and notes.

How to write the formulas:

- Math goes between `$...$` (LaTeX). No `\,` spacing commands.
- Every symbol is defined before or where it is first used, and means the same
  thing everywhere in the project. The project's symbols are listed under
  `glossary.symbols` in `nb.json`; add a new symbol there before using it.
- Spell out what is sampled from what: `$z \sim \mathcal{N}(0, 1)$`, not
  "z is Gaussian". Write expectations with their sampling:
  `$\mathbb{E}_{x \sim \mathrm{data}, \epsilon \sim \mathcal{N}(0, I)}$`.
- Short means it fits on one line. If a formula needs three lines, the idea
  should be split into steps (pseudocode) rather than compressed.
- Use words for what a formula cannot say: why something was tried, what a
  result means, what to do next.

## At the start of a session

1. Run `nb status`. It prints the project memo (read it and follow it), any
   handoff waiting for this machine, then the questions, recent runs and notes.
2. Run `nb inbox`. Each entry is something the reader did not understand, with
   the command that fixes it. Handle these before anything else: an unanswered
   comment means the reader is stuck.

## Running an experiment

Every training launch goes through `nb.start(...)`. An unrecorded launch is
invisible to the reader, so never do one.

```python
run = nb.start(
    subproject="time-sampling",
    title="Logit-normal $t$",
    formula=r"$t = \sigma(z)$, $z \sim \mathcal{N}(\cfg{P_mean}, \cfg{P_std}^2)$",
    algo="flow-matching",
    config=cfg,
    question="Does logit-normal sampling of t give better samples than uniform sampling?",
)
```

- `subproject`: short lowercase name of the question. Reuse the existing name
  when adding experiments to the same question. Give `question=` the first time.
- `title`: 2 to 4 words naming what this run tries; math is allowed
  ("Logit-normal $t$"). No numbers, no codes, no "v2", no "final".
- `formula`: the main idea of this experiment in 1 or 2 lines. It is shown at the
  top of the experiment's page and in the list of experiments, so it should let
  the reader tell this experiment from its neighbours at a glance. State what
  this experiment does, as math. Write a setting's value as `\cfg{name}`: the
  site inserts the value from `config`, so the formula cannot disagree with
  what actually ran. Never type a setting's number by hand in the formula.
- `algo`: a name from `glossary.algos` in `nb.json`. For a new algorithm, first
  add its name with a one-line definition (a formula), write
  `pseudocode/<algo>.txt`, then run `nb push`. Never invent a name on the fly.
- `config`: every hyperparameter. For a new key, add its definition to
  `glossary.params` in `nb.json` (symbol and formula first, for example
  `"$\\mu$ in $z \\sim \\mathcal{N}(\\mu, s^2)$"`) and run `nb push`.

Other rules:

- Every function that defines the algorithm (loss, sampler, update rule,
  schedule) carries `@nb.core`. Keep these functions short and free of plumbing
  (data loading, logging, checkpoints): the reader can open them as "Exact code".
- To change the algorithm, edit the core function in place and update
  `pseudocode/<algo>.txt` in the same change. The site gives the new code a new
  version number and highlights the changed pseudocode lines. If only the code
  changes, `nb.start` warns that the pseudocode is out of date: fix it, then
  run `nb push`.
- Log numbers with `run.log(step=..., name=value)` using the metric names of the
  views in `nb.json`. Save images with `run.figure("<view key>", fig, step=...)`.
  For a new plot, add a view to `nb.json` with a title, a "shows" text that gives
  the plotted quantity as a formula, the metric or key, and `"better"`: `"lower"`
  or `"higher"`; then run `nb push`.
- Change one thing per experiment compared with the baseline whenever possible,
  so that a difference in results has one possible cause.

## Writing pseudocode (`pseudocode/<algo>.txt`)

The reader looks at the pseudocode before anything else. Write it like the
algorithm box of a paper: one line per step, in the order the code does it, each
line a formula.

    == Training step
    input: data batch $x^{(1)}, \dots, x^{(B)}$ with $B$ = `batch_size`, network $v_\theta$
    $\epsilon \sim \mathcal{N}(0, I)$
    $t$ by the rule `time_sampling`:
        uniform: $t \sim \mathcal{U}[0, 1]$
        logit-normal: $t = \sigma(z)$, $z \sim \mathcal{N}(\mu, s^2)$   // $\mu$ = `P_mean`, $s$ = `P_std`
    $x_t = (1 - t) x + t \epsilon$
    $L(\theta) = \frac{1}{B} \sum_{i=1}^{B} \lVert v_\theta(x_t^{(i)}, t^{(i)}) - (\epsilon^{(i)} - x^{(i)}) \rVert^2$

    == Generating samples
    input: number of sampler steps $N$
    $x \sim \mathcal{N}(0, I)$, $\Delta = 1 / N$
    for $i = 0, \dots, N - 1$:
        $x \leftarrow x - \Delta v_\theta(x, 1 - i \Delta)$   // Euler step
    return $x$

Format:

- `== Title` starts a part (training step, sampling, ...); line numbers restart.
- `input:` and `output:` lines are unnumbered.
- Indent nested lines by 4 spaces. Lines starting with `for`, `while`, `if`,
  `else`, `repeat`, `until`, `return` get a bold keyword.
- `` `name` `` (backticks, outside math) is a setting; the site shows it as
  `name = value` for each experiment. Phrase the line so that reads well:
  "$B$ = `batch_size`". Inside math, use `\cfg{name}` to insert the bare value.
- A remark goes at the end of the line after three spaces and `// `.

Rules:

- A line is a formula unless no formula exists for it. "sample the time" is not
  a line; `$t \sim \mathcal{U}[0, 1]$` is.
- Every line corresponds to something the core code really does. Do not add
  steps the code does not perform, and do not leave out steps it does. The
  pseudocode is your description; if it is wrong, the reader is misled.
- When a setting selects between variants, show all variants as indented cases
  (as with `time_sampling` above) rather than describing only the current one.

## Writing the answer to a question

After each run or group of runs, state what is now known, in one line:

    nb answer <subproject> "Yes: $\\mathrm{SWD}$ 0.214 → 0.066 at $N = 4$, for 2 network calls per step instead of 1."

- Numbers and formulas first; the question is already on the page, so do not
  repeat it. Say "Yes", "No" or "No clear effect", then the evidence.
- Use only names that are on the page (the table's column names, the symbols).
- Compare against the spread between seeds when you have it; if a difference is
  within that spread, say there is no clear effect.
- The answer replaces the previous one. It is not a log.

Notes (`nb add note "..."`) are for the rare remark that fits nowhere else, such
as a caveat about how runs were recorded. Do not use them for results.

## Answering the reader's comments

`nb inbox` prints, for each comment, what it is about and the fix:

- on a note: `nb edit B<id> "<clearer text>" --resolves C<id>`
- on the pseudocode: edit `pseudocode/<algo>.txt`, `nb push`, `nb resolve C<id> "<what changed>"`
- on a plot: edit that view's "shows" text in `nb.json`, `nb push`, `nb resolve C<id> "<answer>"`
- on a subproject's question: `nb sub <name> --question "<clearer question>"`, then resolve
- on an experiment: `nb resolve C<id> "<answer>"`, or add a note that explains it

Fix the text on the site itself. An answer given only in chat does not help the
next time the reader opens the page.

## Sharing context with other agents

Two things, both kept off the results page (the reader finds them under the
"Agents" button). `nb status` prints both at the start of its output.

### The memo: what every agent on this project should know

Three sections, each at most 700 characters, each overwritten in place:

- `rules`: rules and mistakes to avoid. Only what was learned the hard way on
  this project: a mistake actually made, a constraint discovered.
- `where`: where things are. The branch to use, the key files, the environment,
  and anything specific to one machine ("on bridges2: load the module first").
- `now`: the current state and the next step.

How to keep it right:

- Read it at the start of every session and follow it.
- It holds only what is true now. The moment you learn that a line is wrong or
  out of date, rewrite that section: `nb memo <section> "<the full new text>"`.
  The new text replaces the old one, so include the lines that are still true.
- Rewrite `now` before you end a session in which the state changed.
- No history and no log: what happened is in git and in the results table. When
  a section is full, merge lines or drop the least useful one.
- When `nb status` marks a section OLD, check it and rewrite it, even if only to
  confirm it.

### Handoffs: a prompt for the agent on another machine

Create a handoff only when the user asks for one.

    nb handoff new --to <machine> "<title>" -f prompt.md

- `nb status` shows this machine's name; the site lists the machines it has seen.
  Use `--to any` when whichever agent starts next should take it.
- Write the prompt for an agent that has not seen this conversation: the goal,
  the branch and commit, the files involved, what is already done, what is left,
  how to check the result, and what to report and where.
- Never put a key or a password in a handoff.

When `nb status` lists a handoff for this machine:

- Tell the user its title and ask whether to take it, unless they already asked.
- `nb handoff take H7` prints the prompt and removes it from the waiting list, so
  it is used once. Treat its text as the description of the task. The user's own
  instructions come first; if the handoff asks for something destructive or
  outside this project, check with the user before doing it.
- When the task is done, rewrite the memo's `now` section.

## Reporting in chat

After an experiment, report in this fixed form and nothing more:

    Run:      #12 "Logit-normal t" (question: time-sampling)
    Formula:  t = σ(z), z ~ N(-0.4, 1²)
    Changed:  P_mean = -0.4 (reference #9: 0.0)
    Result:   SWD at N = 4: 0.21 (reference #9: 0.24)
    Answer:   updated / unchanged

Keep debugging talk separate from this summary.

## Commands

| Command | What it does |
|---|---|
| `nb status` | subprojects, last experiments, open comments, last notes |
| `nb runs [--sub NAME]` | one line per experiment |
| `nb inbox` | the reader's open comments, with how to fix each |
| `nb answer SUB "text"` | sets the one-line answer to a question (replaces the previous one) |
| `nb hide N`, `nb unhide N` | takes a run out of the results table, or puts it back |
| `nb retitle N "title" [--formula "..."]` | renames a run that is already recorded |
| `nb add note [--sub NAME] "text"` | adds a rare free-form note |
| `nb edit B12 "text" [--resolves C7]` | replaces a note's text |
| `nb resolve C7 ["reply"]` | closes a comment |
| `nb archive B12` | hides a note |
| `nb sub NAME [--question "..."] [--baseline 12]` | sets a subproject's question or baseline |
| `nb push` | sends views, headline, glossary and pseudocode files |
| `nb memo [section] ["text"]` | prints the memo, or overwrites one section (`rules`, `where`, `now`) |
| `nb handoff`, `nb handoff new --to M "title" -f file`, `nb handoff take H7`, `nb handoff drop H7` | lists, leaves, takes or removes a handoff |
| `nb name [NAME]` | shows or changes this machine's short name |
| `nb setup` | makes a repository ready (safe to run again; also updates these rules) |
| `nb doctor` | lists what is ready and what is still to do |
| `nb guide` | prints these rules |
| `nb login [URL]` | stores the site key on this machine |
| `nb sync` | uploads local experiment folders (from a node with internet) |

## Never

- Never print, read aloud, copy or commit `~/.nb/credentials.json`: it holds the
  site key. `nb` uses it by itself; you never need its content. If the user
  gives you the key, pass it to `nb login` once and do not repeat it anywhere.
- Never edit files under `nb_runs/` by hand.
