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

1. Run `nb status` to see subprojects, recent experiments and recent notes.
2. Run `nb inbox`. Each entry is something the reader did not understand, with
   the command that fixes it. Handle these before anything else: an unanswered
   comment means the reader is stuck.

## Running an experiment

Every training launch goes through `nb.start(...)`. An unrecorded launch is
invisible to the reader, so never do one.

```python
run = nb.start(
    subproject="time-sampling",
    title="Logit-normal time sampling, P_mean = -0.4",
    formula=r"$t = \sigma(z)$, $z \sim \mathcal{N}(\cfg{P_mean}, \cfg{P_std}^2)$, $\sigma(z) = 1 / (1 + e^{-z})$",
    algo="flow-matching",
    config=cfg,
    question="Does logit-normal sampling of t give better samples than uniform sampling?",
)
```

- `subproject`: short lowercase name of the question. Reuse the existing name
  when adding experiments to the same question. Give `question=` the first time.
- `title`: what this experiment tries, in plain words, at most 100 characters.
  No codes, no "v2", no "final".
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

## Writing notes

After each experiment or group of experiments, and after each decision:

    nb add finding  --sub <subproject> "<one result, with numbers and their experiments>"
    nb add decision --sub <subproject> "<what was chosen and why>"
    nb add question --sub <subproject> "<open point, or the next thing to test>"
    nb add note "<anything else about the whole project>"

- Numbers and formulas first: "`swd_4`: 0.214 (#5) → 0.066 (#6)", not "the
  midpoint sampler is much better". Notes render `$...$` math.
- Every number comes with its experiment, written `#12`. A finding without one
  is refused, because a number with no source cannot be checked.
- Use only names from `nb.json`. No new abbreviations, no nicknames: write `#12`,
  not "the v2 run". Say what a quantity is the first time it appears in a note,
  and which direction is better.
- Limits: 280 characters for finding, decision, question; 1200 for note.
- When a finding answers the subproject's question, run `nb pin B<id>`: it is
  shown at the top as a conclusion. Keep at most two conclusions per subproject.
- When something changes, edit the existing note (`nb edit B<id> "..."`). Do not
  add a note that contradicts an old one.
- No logs, stack traces or long derivations.

## Answering the reader's comments

`nb inbox` prints, for each comment, what it is about and the fix:

- on a note: `nb edit B<id> "<clearer text>" --resolves C<id>`
- on the pseudocode: edit `pseudocode/<algo>.txt`, `nb push`, `nb resolve C<id> "<what changed>"`
- on a plot: edit that view's "shows" text in `nb.json`, `nb push`, `nb resolve C<id> "<answer>"`
- on a subproject's question: `nb sub <name> --question "<clearer question>"`, then resolve
- on an experiment: `nb resolve C<id> "<answer>"`, or add a note that explains it

Fix the text on the site itself. An answer given only in chat does not help the
next time the reader opens the page.

## Reporting in chat

After an experiment, report in this fixed form and nothing more:

    Experiment: #12 "Logit-normal time sampling, P_mean = -0.4" (subproject time-sampling)
    Formula:    t = σ(z), z ~ N(-0.4, 1²)
    Algorithm:  flow-matching v2
    Changed:    P_mean = -0.4 (baseline #9: 0.0)
    Result:     swd_4 = 0.21 (baseline #9: 0.24)
    Note:       B14

Keep debugging talk separate from this summary.

## Commands

| Command | What it does |
|---|---|
| `nb status` | subprojects, last experiments, open comments, last notes |
| `nb runs [--sub NAME]` | one line per experiment |
| `nb inbox` | the reader's open comments, with how to fix each |
| `nb add finding\|decision\|question\|note [--sub NAME] "text"` | adds a note, prints its id |
| `nb edit B12 "text" [--resolves C7]` | replaces a note's text |
| `nb resolve C7 ["reply"]` | closes a comment |
| `nb pin B12`, `nb unpin B12`, `nb archive B12` | conclusion on, conclusion off, hide |
| `nb sub NAME [--question "..."] [--baseline 12]` | sets a subproject's question or baseline |
| `nb push` | sends views, headline, glossary and pseudocode files |
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
