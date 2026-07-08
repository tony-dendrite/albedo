# Migration map: albedo-remake-eval -> albedo-simple

## Quick Summary
- Where every piece of the old repo (`/home/ac/Workspace/albedo-remake-eval`) lives in this one.
- Behavior, SQL, wire protocols, and env var names are preserved; only the packaging changed.
- Synced through upstream commit `8b8801b` (2026-07-07, parallel OCI shard downloads) - re-run
  the commit review (`git -C ../albedo-remake-eval log 8b8801b..`) before any cutover.

```bash
# check what upstream changed since the last sync
git -C /home/ac/Workspace/albedo-remake-eval log --oneline 8b8801b..HEAD
```

## Packages -> modules

| Old (`src/<pkg>/...`) | New (`albedo/...`) | Notes |
|---|---|---|
| `chain_reader/` (reader, chain, db) | `chain.py` | one loop: `run_ingest` |
| `chain_guard/` (scan, db, uploads) | `chain.py` | backfill + `used_hotkeys` guard folded in |
| `hippius_validation/validate_worker.py` + `db.py` | `validation.py` | `run_worker`; guards incl. sanity/duplicate hotkey blocks |
| `hippius_validation/validate/{repo,dtype,safetensors_index,architecture,chat_template}.py` | `validation.py` | all checks inline, same order |
| `hippius_validation/hippius/` + `config_validation/hippius/` | `validation.py` (Hippius section) | `list_files`, `download_config`, `download_full`, Range preflight |
| `config_validation/fingerprint/` + `hippius_validation/opensearch/` | `fingerprint.py` | compute + similarity + kNN index |
| `config_validation/{pipeline,checks,chain,publish,result}.py` | **dropped** | dead upstream - never called by the live services |
| `albedo_eval_service/sampling.py` + `sanity_service/dataset.py` + `remote_dataset.py` | `sampling.py` | ONE parquet->prompt implementation (CPU + GPU superset, differential-tested) |
| `sanity_service/{dispatcher,db,llm_check,judge_panel}.py` | `sanity.py` | `run_dispatcher` + `run_janitor` |
| `sanity_service/rubricisity.py` + `checks.py` | `sanity_gate.py` | probe prompts VERBATIM + text heuristics |
| `sanity_remote/` | `remote/sanity_worker.py` | same HTTP routes |
| `albedo_eval_service/{dispatcher,repository,faults,requeuer,models}.py` | `evaluation.py` | `run_dispatcher` + `run_janitor` (requeuer/sweeper/reconciler as 60s timers) |
| `albedo_eval_service/{judge_core,judge_api,judge_config,judge_openrouter}.py` | `judges.py` | binary yes/no judging; **no HTTP judge service** - functions called directly |
| `albedo_eval_service/score_bridge_client.py` | `judges.py` (`run_bridge_client`) | WS frames unchanged |
| `albedo_eval_service/chutes_glm.py` | **dropped** | deleted upstream in 3127360 |
| `albedo_eval_service/{remote_api,remote_worker,remote_generation,remote_models,remote_scoring,remote_artifacts,remote_state,remote_config}.py` | `remote/eval_worker.py` | wire-identical routes + `/score-bridge` WS; parallel OCI downloads |
| `albedo_eval_service/canonical_model_config.py` | `remote/common.py` | genesis config/spec constants + `apply_canonical_model_config` |
| `set_reign_worker/` | `reign.py` | `run_worker` |
| `weight_setter/` | `weights.py` | `run_worker`; MockChainClient block now advances in mock |
| `website/monitor.py` | `monitor.py` | same dashboard/state JSON shapes |
| `albedo_eval_service/api.py` (backend-api) | **dropped** | thin/near-empty upstream |
| per-package `config.py`/`settings.py`/dotenv loaders | `settings.py` | THE only env reader; old env names preserved via aliases |
| `notifications.py` (x2 copies) | `notifications.py` | one Slack helper |
| - (no equivalent) | `hosts.py`, `db.py`, `s3.py`, `remote_client.py`, `backend.py`, `cli.py` | new shared plumbing: GPU heartbeater, one asyncpg pool, one S3 client, one worker HTTP client, the supervisor, `albedo` CLI |

## Processes: ~22 pm2 apps -> 3

| Old pm2 app(s) | New |
|---|---|
| chain-reader, hippius-validation, sanity-dispatcher, eval-dispatcher, judge-api, score-bridge, set-reign-worker, weight-setter, monitor, backend-api | **`albedo-backend`** (one process, 11 supervised asyncio loops) |
| sanity/eval reconciler + sweeper + requeuer (6 cron apps) | in-process 60s janitor timers |
| eval-cache-cleanup, model-gc (2 cron apps) | in-code cache pruning (validation + workers) |
| remote-eval-api (GPU) | **`albedo-gpu-eval`** |
| sanity-remote-api (GPU) | **`albedo-gpu-sanity`** |
| gpu-host-tunnel, sanity-host-tunnel | `albedo-eval-tunnel`, `albedo-sanity-tunnel` (in `pm2/backend.config.js`) |
| db-tunnel (GPU->backend Postgres) | **dropped** - workers are stateless, no DB access |
| king-chat*, king-hf-uploader, benchmarks* | **out of scope** - auxiliary services, not part of the core flow |

## Scripts: ~25 files -> 3 entrypoints

| Old | New |
|---|---|
| create_genesis_king.py | `ops.py seed-genesis` (+ registers `remote_gpu_hosts` - old tree had no writer) |
| check_sanity_openrouter_judges.py | `ops.py preflight` (broadened: DB, GPU `/ready`, score-bridge WS, judges, OpenSearch, S3) |
| generate_arch_spec.py | `ops.py arch-spec` (offline - derived from the embedded genesis config) |
| seed_submission.py | `ops.py inject-submission [--model-uri]` |
| download_datasets.py / prepare_datasets.py / build_manifest.py | `datasets.py download\|prepare\|manifest` (+ local-manifest fallback: `assets/dataset-manifest.json`) |
| bootstrap.sh / bootstrap-gpu.sh / setup.sh / deploy-fresh.sh / deploy-gpu.sh / redeploy.sh / install_deps.sh / stop.sh / setup_opensearch.sh | `deploy.sh bootstrap\|backend\|gpu\|opensearch\|stop` (incl. CUDA-toolkit/nvcc install) |
| ecosystem.test.config.js + .env.test + test-env/ | `flowtest/run.sh` + `env.mock` + `fake_openrouter.py` (automated pass/fail, scored mock duel) |
| cleanup_models.sh / eval_cache_cleanup.py | in-code (validation deletes on terminal outcomes + 48h prune; workers prune their caches) |
| reevaluate.py / clear_uid.py / clear_database.py / delete_hippius_repos.py / detect_dead_processes.py / king_hf_uploader.py | **not ported** - incident one-offs or auxiliary |

## Config & data

| Old | New |
|---|---|
| `.env` (~220 vars, hand-rolled parsers, 11 `.env.bak.*`) | `settings.py` + `.env.example` / `env.gpu-test.template` / `flowtest/env.mock`; empty values = defaults |
| `chain.toml` | constants live where they are used (`validation.py` manifest lists, `remote/common.py` genesis config); regenerate the arch spec with `ops.py arch-spec` |
| `schema.sql` | identical (copied verbatim; `albedo migrate` applies it) |
| `architecture_spec.json` | `albedo/architecture_spec.json` |
| `assets/tokenizers/Qwen3.6-35B-A3B` | same path, same resolution |
| dataset manifest (S3-only fetch) | also shipped at `assets/dataset-manifest.json` (hash-verified local fallback) |
| Doppler via `doppler run -- pm2 start` | same, via `ALBEDO_DOPPLER=true ./scripts/deploy.sh ...` |

## Deliberate behavior differences (everything else is behavior-identical)
- Auth tokens are required at startup unless a mock flag is on (old: silently unauthenticated).
- `remote_gpu_hosts` gets live heartbeats (`hosts.py`); dead boxes go OFFLINE (old: static rows).
- Hard offline guards for tests: `CHAIN_MOCK`, `ALBEDO_MONITOR_MOCK` (old: no such switches).
- One `dataset_manifest_hash` pin, corrected to the live manifest (`980d...`; old envs carried stale `982a...`).
- Eval worker answers 409 while busy (old: accepted concurrent runs).
- Model caches self-prune (old: external cron scripts).
