#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/pub7/neel2/hf-cache}"
PORT="${PORT:-8100}"
TP="${TP:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-vl-72b-experimental}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
COMPILATION_MODE="${COMPILATION_MODE:-0}"
PYTHON_INCLUDE_ROOT="${PYTHON_INCLUDE_ROOT:-$REPO_ROOT/.deps/python-dev/extracted/usr/include}"
PYTHON_HEADERS_DIR="${PYTHON_HEADERS_DIR:-$PYTHON_INCLUDE_ROOT/python3.12}"

if [[ ! -f /usr/include/python3.12/Python.h ]]; then
  "$SCRIPT_DIR/ensure_python_headers.sh"
fi

if [[ ! -f /usr/include/python3.12/Python.h && -f "$PYTHON_HEADERS_DIR/Python.h" ]]; then
  export C_INCLUDE_PATH="$PYTHON_INCLUDE_ROOT:$PYTHON_HEADERS_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPATH="$PYTHON_INCLUDE_ROOT:$PYTHON_HEADERS_DIR${CPATH:+:$CPATH}"
fi

args=(
  vllm serve "$MODEL"
  --download-dir "$CACHE_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --port "$PORT"
  --enable-mm-embeds
  --limit-mm-per-prompt '{"image":0,"video":0}'
  -cc.mode="$COMPILATION_MODE"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi

exec "${args[@]}"
