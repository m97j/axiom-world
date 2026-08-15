#!/usr/bin/env bash
# launch_trl_vllm_server.sh — isolated vLLM rollout server for TRL GRPO
# (server mode), protocol §3 environment separation.
#
# WHY. x18 (runs/x18_vllm_probe.json) proved every vLLM release pins its own
# torch, so vLLM must NEVER be pip-installed into the training runtime.
# This script builds a dedicated venv (its torch is invisible to training)
# and serves rollouts over HTTP; the trainer stays on the image-owned stack.
#
# USAGE (run in a background terminal / nohup BEFORE the train cell):
#   bash scripts/launch_trl_vllm_server.sh Qwen/Qwen3-8B [PORT]
# Then train with:
#   --override training.extra.vllm_mode=server \
#   --override training.extra.vllm_server_base_url=http://127.0.0.1:${PORT}
#
# NOTES.
# - requirements/generation-vllm.lock.txt must carry exact pins (vllm==, trl==).
#   The script refuses to run against an unpinned lock (0.27.1 incident rule).
# - GPU memory is SHARED with the training process on a single-GPU Colab box:
#   keep --gpu-memory-utilization low (default 0.25) so the 8B BF16 policy +
#   optimizer states still fit. Tune upward only after watching nvidia-smi.
# - TRL syncs policy weights to the server each generation round; B-track
#   adapters train lm_head/embed_tokens via modules_to_save — verify after the
#   FIRST sync that server-side completions terminate properly (x09 audit on
#   an early transcript) before trusting a long run.
set -euo pipefail

MODEL="${1:?usage: launch_trl_vllm_server.sh <model-id> [port]}"
PORT="${2:-8000}"
LOCK="requirements/generation-vllm.lock.txt"
VENV="${VLLM_VENV_DIR:-/content/vllm-env}"

grep -Eq '^(vllm|trl)==' "$LOCK" || {
  echo "[launch] $LOCK has no exact pins — run the repin procedure in the lock header first." >&2
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  echo "[launch] creating isolated venv at $VENV"
  pip install --quiet uv
  # Python 3.12+: matches the project's requires-python AND avoids the
  # flashinfer `array.array[int]` TypeError seen on 3.11 (2026-08-15 smoke).
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" -r "$LOCK"
fi

echo "[launch] training-env torch is untouched; server-env stack:"
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as m
for p in ("vllm", "trl", "torch"):
    try:
        print(f"  {p}=={m.version(p)}")
    except m.PackageNotFoundError:
        print(f"  {p}: MISSING")
EOF

exec "$VENV/bin/trl" vllm-serve \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.25}" \
  --max-model-len "${VLLM_MAX_LEN:-4096}"
