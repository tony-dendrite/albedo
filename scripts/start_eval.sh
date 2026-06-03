#!/bin/bash
# Eval-box startup — run on the GPU host the validator's tunnel.sh forwards into.
# All eval-box config (judge auth, model store, dataset, eval-trace + fingerprint
# sink, vLLM topology) is env-driven and centralised here. Run under pm2/systemd/tmux.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- Required: Chutes judge auth -------------------------------------------
: "${CHUTES_API_KEY:?must set CHUTES_API_KEY (cpk_... bearer token)}"
export CHUTES_BASE_URL="${CHUTES_BASE_URL:-https://llm.chutes.ai/v1}"

# --- Required: Hippius Hub auth for materializing king/challenger weights ----
: "${HIPPIUS_HUB_TOKEN:?must set HIPPIUS_HUB_TOKEN (the subnet is Hippius-only)}"

# --- Required: eval-trace + fingerprint-state S3 sink ----------------------
: "${ALBEDO_EVALS_S3_BUCKET:?must set ALBEDO_EVALS_S3_BUCKET (e.g. albedo)}"
: "${ALBEDO_EVALS_S3_ACCESS_KEY:?must set ALBEDO_EVALS_S3_ACCESS_KEY}"
: "${ALBEDO_EVALS_S3_SECRET_KEY:?must set ALBEDO_EVALS_S3_SECRET_KEY}"
export ALBEDO_EVALS_S3_ENDPOINT="${ALBEDO_EVALS_S3_ENDPOINT:-https://s3.hippius.com}"
export ALBEDO_EVALS_S3_PREFIX="${ALBEDO_EVALS_S3_PREFIX:-evals}"
export ALBEDO_EVALS_PUBLIC_BASE="${ALBEDO_EVALS_PUBLIC_BASE:-https://us-east-1.hippius.com}"

# --- Required: local SWE-ZERO corpus (run scripts/prefetch_dataset.py first) -
: "${ALBEDO_DATASET_DIR:?run scripts/prefetch_dataset.py first and export the printed path}"
test -d "$ALBEDO_DATASET_DIR"
test -f "$ALBEDO_DATASET_DIR/manifest.json"

# --- vLLM topology — king and challenger run concurrently. ------------------
# Defaults assume one GPU per side (fits Qwen3-4B). For 8B-14B give each side
# more GPUs (e.g. "0,1") — eval.py auto-sets tensor-parallel = #GPUs per side.
export ALBEDO_KING_GPUS="${ALBEDO_KING_GPUS:-0}"
export ALBEDO_CHAL_GPUS="${ALBEDO_CHAL_GPUS:-1}"
export ALBEDO_GPU_MEMORY_UTILIZATION="${ALBEDO_GPU_MEMORY_UTILIZATION:-0.85}"
export ALBEDO_VLLM_DTYPE="${ALBEDO_VLLM_DTYPE:-bfloat16}"
# vLLM flags proven on the H200 stack (no vendored nvcc/deep_gemm).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

# --- Eval server bind + judge concurrency ----------------------------------
export ALBEDO_EVAL_HOST="${ALBEDO_EVAL_HOST:-0.0.0.0}"
export ALBEDO_EVAL_PORT="${ALBEDO_EVAL_PORT:-9001}"   # match tunnel + validator
export ALBEDO_MAX_PARALLEL_TURNS="${ALBEDO_MAX_PARALLEL_TURNS:-8}"

# --- Launch (eval.py shim -> albedo.eval_server, binds ALBEDO_EVAL_HOST/PORT) -
exec .venv/bin/python eval.py
