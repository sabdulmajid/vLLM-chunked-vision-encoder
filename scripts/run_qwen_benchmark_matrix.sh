#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/pub7/neel2/hf-cache}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CONTROL_GPU_MEMORY_UTILIZATION="${CONTROL_GPU_MEMORY_UTILIZATION:-0.82}"
EXPERIMENTAL_GPU_MEMORY_UTILIZATION="${EXPERIMENTAL_GPU_MEMORY_UTILIZATION:-0.82}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"
SHUTDOWN_TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-120}"
COOLDOWN_S="${COOLDOWN_S:-15}"
ASSETS_DIR="${ASSETS_DIR:-.bench-assets/managed}"
LOG_DIR="${LOG_DIR:-.bench-assets/logs}"

args=(
  python benchmark_deployment.py
  --model "$MODEL"
  --cache-dir "$CACHE_DIR"
  --device "$DEVICE"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --control-gpu-memory-utilization "$CONTROL_GPU_MEMORY_UTILIZATION"
  --experimental-gpu-memory-utilization "$EXPERIMENTAL_GPU_MEMORY_UTILIZATION"
  --startup-timeout-s "$STARTUP_TIMEOUT_S"
  --shutdown-timeout-s "$SHUTDOWN_TIMEOUT_S"
  --cooldown-s "$COOLDOWN_S"
  --assets-dir "$ASSETS_DIR"
  --log-dir "$LOG_DIR"
)

exec "${args[@]}"
