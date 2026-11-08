"""GRPO on Qwen2.5-1.5B-Instruct with learned hh-rlhf reward model(s) as the reward.

All runs use the same mixed prompt set (helpful + red-team prompts); only the reward differs:
  helpful | harmless   one reward model
  combined             helpful RM + harmless RM, summed per answer
Usage: accelerate launch --num_processes N train_grpo.py helpful|harmless|combined [max_steps]
"""
import os
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
name = sys.argv[1]
rm_names = ["helpful", "harmless"] if name == "combined" else [name]
world_size = int(os.environ.get("WORLD_SIZE", "1"))
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
prompts = load_from_disk("data/prompts")["train"].remove_columns(["source"])

# Eager attention everywhere: sdpa produced NaN on padded batches with torch 2.13 / transformers 5.19.
rm_toks, reward_models = [], []
for rm in rm_names:
    tok = AutoTokenizer.from_pretrained(f"rm_{rm}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"rm_{rm}", num_labels=1, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.pad_token_id = tok.pad_token_id
    rm_toks.append(tok)
    reward_models.append(model)

args = GRPOConfig(
    model_init_kwargs={"attn_implementation": "eager"},  # policy: fp32 weights (TRL default) + bf16 autocast
    output_dir=f"grpo_{name}" if max_steps >= 100 else f"grpo_{name}_smoke",
    max_steps=max_steps,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    per_device_train_batch_size=8,
    # 128 completions (16 prompts x 8) per step whatever the GPU count, so runs stay comparable
    gradient_accumulation_steps=128 // (8 * world_size),
    num_generations=8,              # the group: 8 answers per prompt, each compared to the group average
    max_completion_length=256,
    temperature=1.0,
    beta=0.04,                      # KL leash to the frozen original model
    bf16=True,
    gradient_checkpointing=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
    logging_steps=1,
    log_completions=True,
    num_completions_to_print=2,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    report_to="none",
)
trainer = GRPOTrainer(
    model=BASE,
    reward_funcs=reward_models,  # several reward models are summed per answer, then group-normalized
    reward_processing_classes=rm_toks,
    args=args,
    train_dataset=prompts,
)
trainer.train()
trainer.save_model(args.output_dir)
