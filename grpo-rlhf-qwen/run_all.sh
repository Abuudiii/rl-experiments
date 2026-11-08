#!/usr/bin/env bash
# Full pipeline on one machine with 8 GPUs (it ran on 8x H100 80GB).
# Run from the repo root with the virtualenv active:  bash run_all.sh
set -euo pipefail
mkdir -p logs

# rl <gpus> <port> <script> [args...]: multi-GPU launch on the listed GPUs
rl() {
  local gpus=$1 port=$2
  shift 2
  local n
  n=$(tr ',' '\n' <<< "$gpus" | wc -l)
  CUDA_VISIBLE_DEVICES=$gpus accelerate launch --multi_gpu --num_processes "$n" --main_process_port "$port" "$@"
}

# 1. Data: hh-rlhf helpful-base / harmless-base -> chat-format preference pairs and prompt sets
python prep_data.py

# 2. Reward models, one per subset, 4 GPUs each in parallel
rl 0,1,2,3 29501 train_rm.py helpful > logs/rm_helpful.log 2>&1 &
rl 4,5,6,7 29502 train_rm.py harmless > logs/rm_harmless.log 2>&1 &
wait

# 3. GRPO against each reward model in parallel, then against both summed on all 8 GPUs
rl 0,1,2,3 29521 train_grpo.py helpful 1000 > logs/grpo_helpful.log 2>&1 &
rl 4,5,6,7 29522 train_grpo.py harmless 1000 > logs/grpo_harmless.log 2>&1 &
wait
rl 0,1,2,3,4,5,6,7 29523 train_grpo.py combined 1000 > logs/grpo_combined.log 2>&1

# 4. Held-out answers, one model per GPU, then score every answer with both reward models.
#    The harmless run is evaluated at checkpoint 500: it went unstable soon after.
CUDA_VISIBLE_DEVICES=0 python evaluate.py gen Qwen/Qwen2.5-1.5B-Instruct base > logs/gen_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python evaluate.py gen grpo_helpful helpful > logs/gen_helpful.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python evaluate.py gen grpo_harmless/checkpoint-500 harmless_ck500 > logs/gen_harmless.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python evaluate.py gen grpo_combined combined > logs/gen_combined.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python evaluate.py score base helpful harmless_ck500 combined | tee logs/eval_scores.log

# 5. Figures
CUDA_VISIBLE_DEVICES=0 python prompt_shots.py   # side-by-side answers to a few fixed prompts
python render_heldout.py                        # two held-out red-team conversations
python make_examples.py                         # the 2x4 summary image
python make_charts.py                           # tradeoff, refusal and training charts
