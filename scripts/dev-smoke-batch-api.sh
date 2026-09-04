#!/bin/bash
# Smoke test for the coordinator's Batch API surface (Tidal batch lane, PR2).
#
# Uploads a three-line JSONL input file, creates a batch from it, polls until
# the batch is `completed`, and asserts request_counts.completed == 3. With the
# dispatcher running, three short items settle in seconds; the 180 s budget is
# there for a cold model load on the first request.
#
# The coordinator must be running with the batch lane enabled, i.e. a mnemonic
# configured, or EIGENINFERENCE_BATCH_DEV_INSECURE_KEY=true for local
# development. Without it every route below answers 503 batch_unavailable.
#
# Usage:
#   COORD=http://127.0.0.1:18080 \
#   API_KEY=$DEV_API_KEY \
#   MODEL=mlx-community/gemma-4-e2b-it-4bit \
#   scripts/dev-smoke-batch-api.sh
#
# No real prompts — every input below is synthetic.

set -euo pipefail

COORD="${COORD:-http://127.0.0.1:18080}"
API_KEY="${API_KEY:-}"
MODEL="${MODEL:-${DARKBLOOM_TESTBED_MODEL:-mlx-community/gemma-4-e2b-it-4bit}}"

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
step()  { printf '\n--- %s ---\n' "$*"; }

if [ -z "$API_KEY" ]; then
  red "missing API_KEY — set it and retry"
  exit 2
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
INPUT="$WORKDIR/batch-input.jsonl"

jq_field() { python3 -c '
import sys, json
d = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    d = d.get(key) if isinstance(d, dict) else None
print("" if d is None else d)' "$1"; }

step "build a 3-line input file"
: > "$INPUT"
for i in 0 1 2; do
  python3 - "$MODEL" "$i" >> "$INPUT" <<'PY'
import json, sys
model, index = sys.argv[1], sys.argv[2]
print(json.dumps({
    "custom_id": f"smoke-{index}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 8,
    },
}))
PY
done
green "wrote $(wc -l < "$INPUT" | tr -d ' ') lines"

step "POST /v1/files"
FILE_JSON=$(curl -fsS -X POST "$COORD/v1/files" \
  -H "Authorization: Bearer $API_KEY" \
  -F "purpose=batch" \
  -F "file=@$INPUT;type=application/jsonl")
FILE_ID=$(printf '%s' "$FILE_JSON" | jq_field id)
if [ -z "$FILE_ID" ]; then
  red "FAIL: upload returned no file id: $FILE_JSON"
  exit 1
fi
green "uploaded $FILE_ID"

step "POST /v1/batches"
BATCH_JSON=$(curl -fsS -X POST "$COORD/v1/batches" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input_file_id\":\"$FILE_ID\",\"endpoint\":\"/v1/chat/completions\",\"completion_window\":\"24h\"}")
BATCH_ID=$(printf '%s' "$BATCH_JSON" | jq_field id)
if [ -z "$BATCH_ID" ]; then
  red "FAIL: create returned no batch id: $BATCH_JSON"
  exit 1
fi
green "created $BATCH_ID"

step "GET /v1/batches/$BATCH_ID (poll to completed, up to ${POLL_TIMEOUT:-180}s)"
POLL_TIMEOUT="${POLL_TIMEOUT:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
DEADLINE=$(( $(date +%s) + POLL_TIMEOUT ))
STATUS=""
GET_JSON=""

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  GET_JSON=$(curl -fsS "$COORD/v1/batches/$BATCH_ID" -H "Authorization: Bearer $API_KEY")
  STATUS=$(printf '%s' "$GET_JSON" | jq_field status)
  COMPLETED=$(printf '%s' "$GET_JSON" | jq_field request_counts.completed)
  FAILED=$(printf '%s' "$GET_JSON" | jq_field request_counts.failed)
  TOTAL=$(printf '%s' "$GET_JSON" | jq_field request_counts.total)
  printf '  status=%s completed=%s failed=%s total=%s\n' "$STATUS" "$COMPLETED" "$FAILED" "$TOTAL"
  case "$STATUS" in
    completed|failed|cancelled|expired) break ;;
  esac
  sleep "$POLL_INTERVAL"
done

if [ "$STATUS" != "completed" ]; then
  red "FAIL: status = ${STATUS:-<none>}, want completed within ${POLL_TIMEOUT}s"
  printf '%s\n' "$GET_JSON"
  exit 1
fi
if [ "$COMPLETED" != "3" ]; then
  red "FAIL: request_counts.completed = $COMPLETED, want 3"
  printf '%s\n' "$GET_JSON"
  exit 1
fi

green "batch $BATCH_ID completed with 3 of 3 requests"
green "batch API smoke PASSED"
