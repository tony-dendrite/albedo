# Scoring on Albedo (SN97)

How a challenger and the reigning king are compared. Everything below is what
`src/albedo_eval_service/judge_core.py`, `judge_api.py`, `evaluator/` (question generation) and
`shared/loop_check.py` actually do — constants are quoted from the code, not from policy documents.

For what the models are scored *on*, see [DATASETS.md](DATASETS.md).

---

## The shape of one eval

An eval is a **duel**, not a benchmark run. Both models answer the same sampled coding-trajectory
prefixes, and each sample is scored by the same checklist for both sides. Each side rolls every
sample out `ALBEDO_REMOTE_ROLLOUTS_PER_SAMPLE` (2) times and the score is the mean over all
trajectories, so one lucky or unlucky rollout weighs half as much.

```
sample prefix ──► king model      ──► king trajectory      ─┐
              └─► challenger model──► challenger trajectory ─┤
                                                             ├─► judges answer the SAME
reference model ──► N reference runs ──► vector of change ────┘   yes/no checklist per side
                                          └─► question ladder
```

Per sample the pipeline is:

1. **Reference runs** — `ALBEDO_JUDGE_REFERENCE_RUNS` (3) trajectories are generated concurrently
   from the same prefix by the SOTA model (`ALBEDO_JUDGE_SOTA_MODELS`), each through the same
   simulated-observation loop the candidates face, for `ALBEDO_JUDGE_SOTA_TRAJECTORY_TURNS` (8)
   turns. They share one world: the observation simulator memoises on
   `(sample_id, format, repo state, command)`, so the same command in the same state resolves to
   the same observation in every run and the runs diverge only where the model chose differently.
   Losing one run is survivable; below two there is nothing to compare and the sample is dropped.
2. **Vector of change** — ONE evaluator call reads every run at once and reduces them to an ordered
   list of task-level **milestones** (`prompt_milestones.py`).
3. **Question ladder** — ONE evaluator call turns the surviving milestones into questions at a
   spread of depths (`prompt_ladder.py`).
4. **Judging** — each judge model answers the whole checklist twice: once for the king's
   trajectory, once for the challenger's. Judges never see which side is which, and never see any
   reference (leak-filtered, see below).
5. **Aggregation** — weighted yes-rate per judge → mean across judges → mean across samples.

If fewer than two reference runs survive, or the sample carries no prior context to anchor them to,
`QuestionService.prepare` raises `QuestionScoringUnavailable` — there is no task-only fallback
checklist. `question_source.question_mode` is always `"milestone_ladder"`.

---

## The checklist

### The vector of change

The extractor (`EXTRACTOR_SKELETON`) is shown the TASK block — system prompt, problem
description, and the conversation that already ran, observations included — followed by every
reference run. It returns an ordered list of milestones, each of exactly one category:

| category | what it asserts | admissible evidence (`SOURCES_BY_CATEGORY`) |
|---|---|---|
| `claims` | what the runs found when they put the task's own claim to the repository | `output`, `stated` |
| `explore` | a mechanism established in the code | `output`, `stated` |
| `action` | what the code now does that it did not before, and where | `edit`, `output`, `command` |
| `verification` | a check run, named by the behaviour it exercises — never its result | `command`, `stated` |

Each milestone carries `necessary` + `necessity_reason` (a counterfactual — *what breaks without
it* — explicitly **not** a majority vote across runs), `in_prefix` (whether the TASK block already
handed it over), `consensus` (which runs reached it), `depends_on`, and one `{run, step, source,
span}` evidence entry per consenting run.

`validate_vector` then enforces mechanically what the prompt asks for, recording every drop:

- `optional` — `necessary: false`.
- `given_by_task` — `in_prefix: true`, or a span that occurs in the TASK block.
- `source_<x>_inadmissible_for_<category>` — an `explore` resting on a block the agent authored
  (it was assumed, not discovered), or a `verification` resting on what came back.
- `span_not_in_<source>` — the span does not occur in that kind of block in the run it cites. Spans
  are compared after `normalise_span`, which folds whitespace, smart quotes and `nl`/`grep -n` line
  prefixes so a faithfully copied span is not rejected for formatting.
- `unreached` — every evidence entry failed, so no run demonstrably got there.

A milestone whose `depends_on` names a dropped milestone has that id pruned, so nothing grounds
itself on something that no longer exists.

The extractor is run `milestone_readings` (4) times over the same runs, each reading validated as
above. One reading of the same runs comes back with a different subset of the facts, so the readings
are merged rather than the best one picked: exact duplicates are grouped, one aligner call matches the
rest by fact (`vector_merge.py`), and the result is the **union** — a fact any reading found is kept
once, under its best-evidenced wording, with the evidence pooled and `depends_on` remapped. Nothing
is re-asked.

### The ladder

`project_vector` renders each surviving milestone for one question-writing call: its category and
statement, why it is required, the statements it depends on, whether the runs' spans point at the
same code (`spans_agree`, token Jaccard ≥ 0.6 — fewer than two spans is not agreement but the
absence of a comparison, and answers no, so a milestone only one run reached licenses no naming),
what every run worked on en route (the intersection of the runs' command targets over their
approach windows), each run's own sentences from that window, and up to three distinct spans.
Run numbers and consensus lists are withheld — the writer needs to know that the routes differed,
never which agent took which.

`LADDER_SKELETON` then writes a set of questions per milestone, up to `RUNGS_MAX` (6) each and
`QUESTIONS_MAX` (60) overall. Where the runs reached a milestone through **different** code, a
question may name no file, function or expression at all — naming either route would punish every
candidate that took the other.

The prompt asks for questions at a range of difficulties within a milestone, and each question
records the depth it was written at in a `rung` field. Every question is answered independently
and weighs the same, so a milestone's contribution to the score is just the number of its
questions answered yes. `rung` is renumbered contiguously per milestone by `parse_ladder` and is
carried into the scoring record for analysis only.


The ladder is written `question_readings` (3) times over the whole vector. Questions are aligned by
what they test and kept by **majority** — asked by at least two readings, earliest wording — and a
milestone left under `LADDER_MIN` (2) is topped up from the spare questions rather than re-asked.

### Enforcement at parse time

Prose rules in a prompt get ignored, so the parsed output is re-checked in code and the drops are
recorded in `question_source`:

- `unfolded_avoid` (`enforce_question_labels`) — "avoids X" checks with no action verb are dropped;
  inaction sweeps them.
- `reference_leak` (`filter_reference_leaks`) — a question that mentions the reference, milestones,
  a vector of change, or the other agents is dropped. The judge sees one candidate and one
  question; a question naming any of that invites it to score against something it cannot see.
  duplicates, template stamping, generic-hygiene and negative-form caps.

### Pruning against the reference runs

Every reference run is judged against the finished checklist, and a question that **not one of them**
earns is dropped (`_prune_unreachable`, gated by `ALBEDO_JUDGE_REFERENCE_PRUNE`).
If no run returns a readable verdict the unpruned checklist is kept rather than deleted blind.

A sample is rejected outright if fewer than `QUESTION_FLOOR` (6) milestone questions survive

## Before the judge: degenerate sides are scored 0 outright

Two checks run on a side's document *before* any judge call, in `_side()` in `judge_api.py`. Either
one short-circuits scoring for that side: every question is answered `0`, `parse_ok` stays `True` (this
is a real score, not a parse failure, so it counts toward `min_valid_fraction`), and no tokens are spent.

1. **Truncated output** (`is_truncated`) — the side is recorded as corrupted.
2. **Looped trajectory** (`shared/loop_check.py`) — the `CANDIDATE OUTPUT` blocks are scanned for shell
   commands (context turns and environment observations are excluded, so a bash fence in the PR
   description cannot trigger it). A side is looped when either:
   - the duplicate-command ratio is ≥ `DUP_CMD_THRESHOLD` (0.5), or
   - one command repeats ≥ `MAX_RUN_THRESHOLD` (4) times consecutively.

   The explanation written into every answer names the reason and the offending commands with their
   repeat counts, e.g. *"same command repeated 10x consecutively. Looping commands:
   `sed -n '211,217p' ./dask/dataframe/backends.py` 10x (10 consecutive)"*. Only looping commands are
   listed, sorted by longest run, capped at 5. The record also carries `looped`, `loop_reasons` and
   `loop_commands` for analysis.

This matters because the judge does **not** reliably punish loops on its own: measured over 200 replayed
trajectories, looped sides were scoring 0.578 against 0.673 for clean ones — a 0.095 penalty — with the
worst looped trajectory scoring 0.913 while repeating one `grep` in 10 of its 12 commands.
`sanity_service/tail_check.py` applies the same heuristic earlier, at pre-eval.


## From answers to a score

### 1. Per judge: a weighted yes-rate

`judge_yes_rate` is the mean of every answered bit (1/0), weighted per question by tag. 


| tag | weight |
|---|---|
| `reference:claims` | 1.0 |
| `reference:explore` | 1.0 |
| `reference:action` | 1.0 |
| `reference:verification` | 1.0 |

A question's tag is the category of the milestone it was written for, assigned in code from the
validated vector rather than taken from what the writer emitted. The four are equal today: what a
milestone is worth is decided by how many questions it supports, not by which category it is. They
are separate entries so a category can be re-weighted on its own, and so the dashboard's
score-by-tag table says whether a duel was decided on claims, exploration, actions or
verifications.

**Nothing normalises per milestone.** A milestone decides which questions get written, not what an
answer to one is worth, so a milestone that yielded six questions carries three times the weight of
one that yielded two. That is the intended reading: a milestone with more distinct ground in it is
worth more of the score.

### 2. Across judges and samples

- `response_score` — mean of the per-judge rates for one side of one sample.
- `aggregate_scores` — mean across samples, per side. **King and challenger scores are
  independent; they do not sum to 1.**
- `by_judge` in the verdict is **challenger-only**. The dashboard recomputes the king's per-judge
  rates from the `SCORING_RESULTS` artifact (`website/monitor.py`).

### 3. The verdict

```python
challenger_beats_king = (score_challenger - score_king) >= CHALLENGER_WIN_MARGIN   # 0.025
```

A **2.5-point absolute margin** — beating the king by a hair is a loss. Scores are the mean yes-rate,
so the margin is in the same units. The comparison runs on the GPU box (`remote/worker.py`), and the
backend records what it reports; a version skew between the two boxes skews the margin, so they are
always deployed together.

**Winning once is not enough.** `control/repository.py` requires **two** independent eval wins before a
submission is crowned: on the first `EVAL_WIN` it puts the submission back to `EVAL_QUEUED` with
`priority = 0`, and only a second `SUCCEEDED` eval run with `challenger_won` promotes it for real.

An eval **fails** rather than returning a verdict if fewer than `min_valid_fraction` of samples
scored (`0.8` in code, `ALBEDO_JUDGE_MIN_VALID_FRACTION`) — fault `scoring_invalid`, retryable. A
sample counts as scored only when **every** judge parsed for **both** sides.

---

## Anti-gaming

The checklist is generated per sample from several reference runs of that very task, so it cannot
be pre-computed. On top of that:

- **Necessity, not consensus** — a milestone is kept because the task's own logic requires it, not
  because the runs agreed on it. A step all three runs wasted is still dropped, and a milestone only
  one run reached is still kept when the others failed or stalled.
- **The given-material floor** — anything the TASK block already handed over is dropped
  (`in_prefix`, `span_from_task`), so a candidate cannot earn a question by restating its prompt.
- **Verbatim spans, checked in code** — every milestone must quote the run it came from, in a block
  of the kind its `source` claims. An invented or misattributed span deletes the evidence.
- **Reference leak filter** — `filter_reference_leaks` drops questions that mention the reference,
  milestones, a vector of change or the other agents; judges must never learn any of it exists.
- **Prompt-injection stripping** — `strip_reply_injection` removes verdict-shaped text
  (`accept` / `weak_pass` / `reject`) and delimiter injections from a candidate's own output before
  a judge reads it, so a model cannot address the judge directly. The scan is a linear marker-first
  search rather than a regex, after the regex version froze the judge on fence-heavy documents.
- **Loop short-circuit** — see *Before the judge* above: a trajectory that has collapsed into a command
  loop is scored 0 without being judged, so repetition cannot be dressed up as thoroughness.

---

## Configuration

Judge-side settings are `JudgeSettings` in `src/albedo_config/config.py`, prefix `ALBEDO_JUDGE_`
(there is no `judge_config.py` any more — all per-service settings were consolidated into
`albedo_config`, and `.env` now carries only secrets and topology):

| setting | code default | meaning |
|---|---|---|
| `evaluator_model` | `z-ai/glm-5.2` | reads the vector and writes the ladder |
| `sota_models` | `z-ai/glm-5.2` | pool the reference runs are drawn from |
| `reference_runs` | 3 | how many reference trajectories are generated per sample |
| `reference_prune` | `true` | judge every run against the checklist and drop what none of them earns |
| `milestone_readings` | 4 | independent extractor readings, merged by union |
| `question_readings` | 3 | independent ladder writers, merged by majority |
| `judge_repeats` | 3 | judgings per trajectory; a question's answer is the majority |
| `judge_count` | 1 | how many judges vote |
| `sota_trajectory_turns` | 8 | reference trajectory length |
| `min_valid_fraction` | 0.8 | below this the eval fails instead of scoring |
| `max_concurrency_per_model` | 128 | per-model in-flight judge calls |
| `simulation_model` / `simulation_providers` | `deepseek/deepseek-v4-flash-0731` / `deepseek,cloudflare` | observation simulator (see [DATASETS.md](DATASETS.md)) |
| `repo_context_url` | `""` | grounding service; empty disables grounding (see [DATASETS.md](DATASETS.md)) |

The model roster lives in `src/albedo_config/models.py`. `JUDGE_MODELS` is now a **single** judge —
`("z-ai/glm-5.2",)` — matching `judge_count = 1`; `EVALUATOR_MODEL` and `SOTA_MODELS` are the same model.
Read the run's `judge-results` in `scoring-results.jsonl` to confirm who actually voted for a given eval.

`ScoringConfig.allowed_scores` is `[0, 1]`: answers are binary, and the verdict reports
`scoring_mode: "binary"`.
