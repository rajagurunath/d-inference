#!/bin/bash
# Smoke test for the single-Mac dev stack (`make dev-stack`). Phase 1: proves
# a streaming chat completion round-trips through the in-process coordinator
# and provider within 60 seconds. PR2 extends this script with the batch
# round trip once /v1/files and /v1/batches exist; this phase only checks the
# online chat path.
#
# Usage:
#   DARKBLOOM_DEV_KEY=<api key printed by `make dev-stack`> \
#   DARKBLOOM_TESTBED_MODEL=mlx-community/gemma-4-e2b-it-4bit \
#   scripts/dev-smoke-batch.sh
#
# DARKBLOOM_DEV_URL defaults to http://127.0.0.1:18080 (the dev-stack listen
# address). Exits 0 when an SSE `data:` line arrives within 60s, 1 otherwise.

set -uo pipefail

DARKBLOOM_DEV_URL="${DARKBLOOM_DEV_URL:-http://127.0.0.1:18080}"
DARKBLOOM_DEV_KEY="${DARKBLOOM_DEV_KEY:-}"
MODEL="${DARKBLOOM_TESTBED_MODEL:-mlx-community/gpt-oss-20b-MXFP4-Q8}"
TIMEOUT_SECS=60

if [ -z "$DARKBLOOM_DEV_KEY" ]; then
  echo "FAIL: DARKBLOOM_DEV_KEY not set (copy the API key \`make dev-stack\` prints on startup)" >&2
  exit 2
fi

if ! curl -fsS --max-time 5 "$DARKBLOOM_DEV_URL/health" >/dev/null 2>&1; then
  echo "FAIL: $DARKBLOOM_DEV_URL/health did not respond — is \`make dev-stack\` running?" >&2
  exit 1
fi

TMP_OUT=$(mktemp)
TMP_ERR=$(mktemp)
CURL_PID=""
cleanup() {
  [ -n "$CURL_PID" ] && kill "$CURL_PID" >/dev/null 2>&1
  rm -f "$TMP_OUT" "$TMP_ERR"
}
trap cleanup EXIT

curl -N -sS --max-time "$TIMEOUT_SECS" "$DARKBLOOM_DEV_URL/v1/chat/completions" \
  -H "Authorization: Bearer $DARKBLOOM_DEV_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "max_tokens": 32,
    "stream": true
  }' >"$TMP_OUT" 2>"$TMP_ERR" &
CURL_PID=$!

START=$(date +%s)
FOUND=0
while [ $(( $(date +%s) - START )) -lt "$TIMEOUT_SECS" ]; do
  if grep -q '^data:' "$TMP_OUT" 2>/dev/null; then
    FOUND=1
    break
  fi
  if ! kill -0 "$CURL_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
ELAPSED=$(( $(date +%s) - START ))

if [ "$FOUND" -eq 1 ]; then
  echo "OK: first SSE data line arrived after ${ELAPSED}s (model: $MODEL)"
  exit 0
fi

echo "FAIL: no SSE 'data:' line within ${TIMEOUT_SECS}s (model: $MODEL)" >&2
echo "--- response so far ---" >&2
cat "$TMP_OUT" >&2
echo "--- curl stderr ---" >&2
cat "$TMP_ERR" >&2
exit 1
