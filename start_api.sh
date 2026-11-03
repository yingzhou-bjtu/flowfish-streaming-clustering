#!/bin/bash
# Start the speaker-identification HTTP API using the local virtual environment.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${SV_API_PORT:-8765}"
LOG="${SV_API_LOG:-./api.log}"
ONNX_PATH="${SV_ONNX_PATH:-}"
PIDFILE="${SV_API_PIDFILE:-./api.pid}"
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv; run ./setup_venv.sh first" >&2
  exit 1
fi
if [[ -n "$ONNX_PATH" && ! -f "$ONNX_PATH" ]]; then
  echo "SV_ONNX_PATH does not exist: $ONNX_PATH" >&2
  exit 1
fi
ONNX_ARGS=()
if [[ -n "$ONNX_PATH" ]]; then
  ONNX_ARGS=(--onnx-path "$ONNX_PATH")
fi
if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID"
  fi
fi
sleep 1
nohup "$PY" api_server.py \
  --host 0.0.0.0 --port "$PORT" \
  "${ONNX_ARGS[@]}" \
  --ort-threads "${SV_ORT_THREADS:-4}" \
  --max-hdbscan-utterances "${SV_MAX_HDBSCAN_UTTERANCES:-60}" \
  >>"$LOG" 2>&1 &
echo "$!" >"$PIDFILE"
echo "started pid=$! port=$PORT log=$LOG"
sleep 8
curl -sf "http://127.0.0.1:${PORT}/v1/health" && echo
