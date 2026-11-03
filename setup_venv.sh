#!/bin/bash
# Create an isolated virtual environment for the streaming runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "需要 python3" >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1 && [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  apt-get update -qq
  apt-get install -y -qq python3-venv python3-dev build-essential libsndfile1 2>/dev/null || true
else
  echo "Install python3-venv, Python headers, a C++ compiler, and libsndfile with your system package manager." >&2
fi

echo "=== 创建 venv: $ROOT/.venv ==="
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

echo "=== Install pinned runtime dependencies ==="
pip install -r requirements.txt

echo "=== 验证 ==="
python -c "from importlib.metadata import version; import numpy, hdbscan, onnxruntime; print('OK numpy', numpy.__version__, 'hdbscan', version('hdbscan'))"
echo ""
echo "以后请用:"
echo "  source $ROOT/.venv/bin/activate"
echo "  python api_server.py --onnx-path /path/to/speaker_embedding.onnx"
