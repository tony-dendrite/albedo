# Scoring on Albedo (SN97)

How a challenger and the reigning king are compared. Everything below is what
`src/albedo_eval_service/judge_core.py`, `judge_api.py`, `evaluator/` (question generation) and
`shared/loop_check.py` actually do — constants are quoted from the code, not from policy documents.

For what the models are scored *on*, see [DATASETS.md](DATASETS.md).

---

## The shape of one eval

An eval is a **duel**, not a benchmark run. Both models answer the same sampled coding-trajectory
prefixes, and each sample is scored by the same checklist for both sides.

```
sample prefix ──► king model      ──► king trajectory      ─┐
              └─► challenger model──► challenger trajectory ─┤
                                                             ├─► judges answer the SAME
reference model ──► reference trajectory ──► checklist ──────┘   yes/no checklist per side
```

Per sample the pipeline is:

1. **Reference trajectory** — a SOTA model (`ALBEDO_JUDGE_SOTA_MODELS`) runs the same task through
   the same simulated-observation loop the candidates face, for
   `ALBEDO_JUDGE_SOTA_TRAJECTORY_TURNS` (8) turns.
2. **Checklist generation** — the evaluator model (`ALBEDO_JUDGE_EVALUATOR_MODEL`) writes up to
   `RUBRIC_MAX_QUESTIONS` (40) yes/no questions anchored on that reference trajectory, plus 18
   behaviour questions from three separate calls; the result is filtered and self-pruned (below).
3. **Judging** — each judge model answers the whole checklist twice: once for the king's
   trajectory, once for the challenger's. Judges never see which side is which, and never see the
   reference (leak-filtered, see below).
4. **Aggregation** — weighted yes-rate per judge → mean across judges → mean across samples.

If the reference cannot be produced (and a re-roll also fails), or the sample carries no prior
context to anchor a reference to, `QuestionService.prepare` raises `QuestionScoringUnavailable` —
there is no task-only fallback checklist. `question_source.question_mode` is always
`"sota_anchored"`.

---

## The checklist

The checklist is built from **two regimes in parallel**, not from fixed section shares (the old
`STEP_SHARES_PCT` split no longer exists):

| regime | size | source | what it asks about |
|---|---|---|---|
| **reference rubric** | up to `RUBRIC_MAX_QUESTIONS` (40) | one evaluator call, anchored on the reference trajectory | the actual work, evidence, continuity, output economy |
| **behaviour checks** | `3 × BEHAVIOR_K` = 18 | three independent evaluator calls | per-trim behaviour, not derived from the reference |

Behaviour questions are only generated when `ALBEDO_JUDGE_NUM_QUESTIONS >= 3 * BEHAVIOR_K`. That
setting (50) therefore **gates** the second regime rather than sizing the checklist — the final count
is whatever survives filtering, which is typically well under 58.

Rubric questions come back with a **tag**, and `RUBRIC_TAG_REQUIRES` maps each tag onto the `requires`
label that the gate and the label caps run on:

| tag | `requires` | meaning |
|---|---|---|
| `action` | `action` | needs real work |
| `continuity` | `action` | must build on what the trajectory already established |
| `verification` | `action` | must check its own result |
| `explore` | `read` | a read-only step can satisfy it |
| `economy` | `neutral` | output hygiene — capped at `RUBRIC_ECONOMY_CAP` (6), with `RUBRIC_LENGTH_BOUNDS` (5) length bounds |

### Enforcement at parse time

Prose rules in a prompt get ignored, so `enforce_question_labels` re-enforces them on the parsed
output and records the drops in `question_source.enforcement_drops`:

- `read_cap` — how many read-only-passable questions survive depends on the sample's phase:
  `READ_CAPS_BY_PHASE` = `cold` 10 / `pre_edit` 7 / `at_edit` 5, falling back to
  `READ_ONLY_QUESTION_CAP` (5) for an unknown phase. A `cold` cut is early in the trajectory, where
  reading *is* the right move; at the edit point it is not.
- `unfolded_avoid` — "avoids X" checks with no action verb are dropped; inaction sweeps them.
- `no_edit_dead_weight` — when the reference never edited in its window, `requires: action`
  questions about completed edits are dropped.

A sample is rejected outright if fewer than `question_floor(n)` = **22%** of the requested questions
come back well-formed (`QUESTION_FLOOR_FRACTION`).

### Self-pruning against the reference

After enforcement, the reference trajectory is scored on its own checklist and the questions **it**
fails are dropped (`_prune_against_reference`), down to a floor of `PRUNE_MIN_SURVIVORS` (8) — a
question the reference itself cannot pass is measuring the rubric, not the candidate. The reference's
own pass rate is recorded as `reference_self_score`. If pruning errors out, the unpruned checklist is
kept rather than failing the sample.

Every question dropped at any stage — parse, leak filter, enforcement, pruning, rejected attempts — is
recorded in the scoring artifact under `question_source`, so a run's checklist is fully reconstructible.

---

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


`judge_yes_rate` is the plain mean of every answered bit (1/0), measurement/size questions
included — there is no separate size multiplier or per-`requires` weighting in the running code, size questions vote like any other question.

### 2. The measurement gate

Before weighting, `apply_measurement_gate` applies two deterministic corrections per candidate — no
judge involved:

- A candidate that made **no edit** has inaction-conditional do-no-harm questions **removed from its
  denominator**. They are dropped, never awarded: inaction is the adversary, and a free `1` would
  reward it.
- If the reference proved an edit was reachable, the candidate made no edit, **and** its final turn
  is still a read, every `requires: action` question is forced to `0`. Well-groomed exploration
  must not out-score imperfect work.

### 3. Across judges and samples

- `response_score` — mean of the per-judge rates for one side of one sample.
- `aggregate_scores` — mean across samples, per side. **King and challenger scores are
  independent; they do not sum to 1.**
- `by_judge` in the verdict is **challenger-only**. The dashboard recomputes the king's per-judge
  rates from the `SCORING_RESULTS` artifact (`website/monitor.py`).

### 4. The verdict

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

The checklist is generated per sample from a model's own reference trajectory, so it cannot be
pre-computed. On top of that:

- **Reference leak filter** — `filter_reference_leaks` drops questions that reveal the reference as
  shared history; judges must never learn a "correct" trajectory exists.
- **Near-duplicate questions** — questions too similar to each other are collapsed
  (`_near_duplicate`: Jaccard 0.75 / containment 0.90 / char-ratio 0.87) so one property cannot be
  asked five ways to inflate its weight.
- **Template caps** — at most 4 questions per template key (`_TEMPLATE_MAX_PER_KEY`).
- **Generic hygiene cap** — `GENERIC_HYGIENE_QUESTION_LIMIT` (3) — questions any competent output
  passes.
- **Negative-question cap** — `NEGATIVE_QUESTION_LIMIT` (8).
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
| `evaluator_model` | `z-ai/glm-5.2` | writes the checklist |
| `sota_models` | `z-ai/glm-5.2` | pool the reference trajectory is drawn from |
| `num_questions` | 50 | gates the behaviour regime (`>= 3 * BEHAVIOR_K`); **not** the checklist size |
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
