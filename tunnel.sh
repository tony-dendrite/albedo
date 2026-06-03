#!/bin/bash
# SSH tunnel from the validator host to the Albedo eval box.
#
# eval.py listens on ALBEDO_EVAL_REMOTE_PORT on the remote GPU host; this forwards
# it to ALBEDO_EVAL_LOCAL_PORT locally so the validator can reach it at
# http://localhost:<local_port> (see ALBEDO_EVAL_SERVER in ecosystem.config.js).
#
# Required env (see ecosystem.config.js):
#   ALBEDO_EVAL_HOST            remote GPU host
#   ALBEDO_EVAL_SSH_PORT        ssh port (default 22)
#   ALBEDO_EVAL_SSH_USER        ssh user (default root)
#   ALBEDO_EVAL_LOCAL_PORT      local forwarded port (default 9001)
#   ALBEDO_EVAL_REMOTE_PORT     port eval.py binds on the remote (default 9001)
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
