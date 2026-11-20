#!/usr/bin/env bash
# Serve Qwen/Qwen3.8-27B with vLLM (OpenAI-compatible) on the H100 node.
#
# Runs in the foreground (exec). Start it as its own long-lived srun step from the laptop:
#   ssh a4 'cd /tmp && setsid nohup srun --jobid=68480301 --overlap -n1 -c16 \
#     --chdir=/home/AMD/absheikh/pr-description-eval \
#     bash harness/serve_vllm.sh > /tmp/absheikh_prdesc_vllm_srun.out 2>&1 < /dev/null & sleep 2; echo launched'
# Logs go to logs/vllm.log. Stop it with `scancel 68480301.<step>` (see `squeue -s -j 68480301`).
#
# Thinking: Qwen3.8 thinks by default. We keep thinking on at reasoning_effort=low and use the
# qwen3 reasoning parser, so <think> text lands in `reasoning_content` and the agent only ever
# sees the final `content`. Set THINKING=off to disable thinking entirely.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

VENV=${VLLM_VENV:-/home/AMD/absheikh/hh-rlhf-grpo/.venv}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
SERVED_NAME=${SERVED_NAME:-qwen3.8-27b}
PORT=${PORT:-18431}
MAX_LEN=${MAX_LEN:-65536}
THINKING=${THINKING:-low}
LOG=${LOG:-logs/vllm.log}
# Extra flags, e.g. VLLM_EXTRA_ARGS="--tensor-parallel-size 2" for the 2-GPU node.
read -r -a EXTRA <<< "${VLLM_EXTRA_ARGS:-}"

case "$THINKING" in
  off) KWARGS='{"enable_thinking": false}' ;;
  # Qwen3.5 chat template: thinking on/off only (no reasoning_effort / preserve_thinking kwargs).
  on) KWARGS='{"enable_thinking": true}' ;;
  # preserve_thinking=false: earlier turns' <think> blocks are not re-rendered into the prompt
  # (the agent loop resends the whole history every step; this keeps context bounded).
  low|medium|xhigh) KWARGS="{\"enable_thinking\": true, \"reasoning_effort\": \"$THINKING\", \"preserve_thinking\": false}" ;;
  *) echo "THINKING must be off|on|low|medium|xhigh" >&2; exit 2 ;;
esac

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
# Job cpuset on d03-08 is 28-31,60-63,92-95,124-127; give vLLM the first 8, leave the rest to docker/agents.
exec taskset -c "${VLLM_CPUS:-28-31,60-63}" vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-num-seqs "${MAX_SEQS:-8}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs "$KWARGS" \
  --enable-prefix-caching \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1
