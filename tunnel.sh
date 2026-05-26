#!/bin/bash
# SSH tunnel from the validator host to the Albedo eval box.
#
# Albedo eval.py listens on ALBEDO_EVAL_REMOTE_PORT on the remote host.
# Keep Albedo and any other eval services on separate ports.
#
# Required envs (see ecosystem.config.js):
#   ALBEDO_EVAL_HOST
#   ALBEDO_EVAL_SSH_PORT
#   ALBEDO_EVAL_SSH_USER
#   ALBEDO_EVAL_LOCAL_PORT
#   ALBEDO_EVAL_REMOTE_PORT
set -euo pipefail

: "${ALBEDO_EVAL_HOST:?must set ALBEDO_EVAL_HOST}"
SSH_PORT="${ALBEDO_EVAL_SSH_PORT:-22}"
SSH_USER="${ALBEDO_EVAL_SSH_USER:-root}"
LOCAL_PORT="${ALBEDO_EVAL_LOCAL_PORT:-9001}"
REMOTE_PORT="${ALBEDO_EVAL_REMOTE_PORT:-9001}"

exec ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -p "${SSH_PORT}" \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${SSH_USER}@${ALBEDO_EVAL_HOST}"
