# Datasets & observation formats (SN97)

What models are evaluated on, how samples are drawn, and the format rule the environment simulator
must obey. Sources of truth: `scripts/prepare_datasets.py`, `scripts/render_trajectories.py`,
`scripts/build_manifest.py`, `src/albedo_eval_service/sampling.py`,
`src/albedo_eval_service/observation_format.py`.

For how the resulting trajectories are turned into a score, see [SCORING.md](SCORING.md).

---

## Corpora

Four sources, declared in `prepare_datasets.SOURCES`. All are real agent trajectories on real
repositories — never synthetic prompts.

| source | language | upstream | notes |
|---|---|---|---|
| `mini-coder` | python | `ricdomolm/mini-coder-trajs-400k` | mini-swe-agent format, used as-is |
| `mini-coder-rs` | rust | `AlienKevin/SWE-smith-rs-*` (3 repos) | rendered; **named `mini-coder-*` deliberately** |
| `open-swe-traces` | mixed | `nvidia/Open-SWE-Traces` | rendered; four upstream arms merged into ONE source so instance dedup sees full coverage |
| `swe-hero` | python | `nvidia/SWE-Hero-openhands-trajectories` | rendered; `repo_cap: 200` (pandas alone was 37.9% of the pool) |

Leaked instances are excluded at manifest-build time: `_MINI_CODER_LEAKS` and `_OPEN_SWE_LEAKS`
(ids reaching the leaderboard union via the SWE-rebench-V2 leak), plus `exclude_upstream` for
`swe-hero` (rebench is our benchmark).

### Rendering normalizes actions, not observations

`render_trajectories.py` converts every upstream tool call into a **bash block** — `_render_call`
maps bash tools to ` ```bash `, editor `view` to `cat -n`, `create` to `cat > … <<'EOF'`,
`str_replace` to a SEARCH/REPLACE block — and folds `think` calls into the next action's `THOUGHT:`.

Tool **observations** are copied through verbatim (`out.append({"role": "user", "content": observation})`).
That asymmetry is the single most important fact about this data: **every trajectory keeps its
upstream observation format all the way into eval.**

---

## Observation formats

Three formats survive in the corpora. `Observation:` (SWE-ZERO) was retired with that corpus.

| format id | shape | who writes it |
|---|---|---|
| `RETURNCODE` | `<returncode>N</returncode>` + `<output>…</output>` | `mini-coder`, `mini-coder-rs` |
| `SWE_AGENT` | `OBSERVATION:` then the output | `open-swe-traces` — `sweagent` arms |
| `OPENHANDS` | bare tool output; **bash** calls close with an exit-code trailer | `open-swe-traces` — `openhands` arms, `swe-hero` |

The OpenHands trailer is structured and must be reproduced:

```
[The command completed with exit code 0.]
[Current working directory: /workspace/<repo>]
[Command finished with exit code 0]
```

Editor-style OpenHands observations carry no trailer — they open with
`File created successfully at: PATH` or `Here's the result of running `cat -n` on PATH:`.

### Format is detected per sample, never from the source name

`open-swe-traces` merges SWE-agent and OpenHands arms under one name, so the source name cannot
determine the format, and the manifest cannot carry it (manifest `rows_meta` only ever reaches the
sampler, never the worker or judge API).

`observation_format.detect_format(sample_id, messages)` instead reads the format off **the
trajectory's own first environment turn** — the first `user` message that follows an `assistant`
message; the leading `user` message is the task. This is always safe because the sampler never cuts
at the first assistant turn, so every sampled prefix carries at least one real observation. The
fallback (no observation in the transcript) is `RETURNCODE` for `mini-coder*` ids and `OPENHANDS`
otherwise — OpenHands because its check is the permissive one, so a wrong guess cannot reject an
otherwise good observation.

### What the simulator must emit

`judge_api` builds the simulator's system prompt as `BASE_PROMPT` + the repo-context block (when
grounding is available) + the detected format's `OUTPUT FORMAT` section. Output is then gated by
`observation_format.valid_output`:

| format | accepted |
|---|---|
| `RETURNCODE` | starts `<returncode>`, contains `</returncode>` and `<output>\n`, ends `\n</output>` |
| `SWE_AGENT` | starts `OBSERVATION:` |
| `OPENHANDS` | anything non-empty that does **not** open with another format's marker |

Rejected output is retried, then falls back to `empty_output(fmt)` for that format. Synthetic
observations the harness injects itself — task submitted, no bash command found, empty output — go
through `wrap(body, fmt)` so they match the trajectory too.

### The simulator chain

`ObservationSimulationService.simulate` runs, per turn:

1. **primary** — `ALBEDO_JUDGE_SIMULATION_MODEL` (`deepseek/deepseek-v4-flash-0731`) hard-pinned to
   `ALBEDO_JUDGE_SIMULATION_PROVIDERS` (`cloudflare`) via `provider: {"only": [...]}`, given exactly
   **one** HTTP call — no parse or transport retries, so a dead endpoint fails fast instead of
   silently re-routing;
2. **fallback** — the evaluator model with its full retry budget;
3. **empty observation** in the detected format.

Looping output (25 identical consecutive lines, or a fully periodic 512-char tail) is collapsed
before use. Repo-context grounding, when configured, injects a tarball snapshot of the repository at
the sampled commit.

---

## Sampling

Deterministic and seeded by the submission's `block_hash` — the same submission always draws the
same samples, and no miner can influence the draw.

`multi_source_manifest_sample_ids` pools **one random rollout per unique `instance_id` across all
sources**, then fills a stratified grid:

- **phase** (`STEP_TRIM`) — where the trajectory is cut, anchored on the instance's `first_edit`:
  `pre_edit` 45% (`first_edit - 2`), `at_edit` 35% (`first_edit`), `cold` 20% (turn 1 or 2).
- **bug family** (`FAMILY_MIX`) — `pr` 50%, `lm` 15%, `combine` 10%, `mechanical` 25%.
- `REPO_CAP = 2` — at most two samples from any one repository.
- `NON_BENCHMARK_LANGUAGE_FRACTION = 0.30` — 30% of the draw is non-`python`.
- `MAX_PREFIX_CHARS = 54_000` — prefixes above this are skipped.

Sample ids are `"<source>/data/train-XXXXX.parquet:<row>:<turn>"`. Defaults:
`sample_count = 100`, `trajectory_assistant_turns = 8` (candidate turns generated per sample),
`max_turns_per_sample = 10`.

## The manifest

`scripts/build_manifest.py` writes `manifest.json` — per source: repo, shard list, per-shard
sha256, and `rows_meta` (one entry per row: `instance_id`, `first_edit`, `family`, `repo`,
`language`, `verified`, prefix sizes). The sampler requires `rows_meta`; single-source manifests are
rejected.

The manifest is **hash-pinned**: `ALBEDO_EVAL_DATASET_MANIFEST_HASH` (and the sanity service's copy)
must match the file's sha256, so validators cannot silently drift onto different data. Rebuilding
prints the new hash and the settings to repin. A rows_meta-stripped `manifest.meta.json` is written
alongside for the dashboard.
