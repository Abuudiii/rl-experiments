"""Train a scalar reward model on one hh-rlhf subset (helpful or harmless).

Usage: accelerate launch --num_processes 4 train_rm.py helpful
"""
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
name = sys.argv[1]
data = load_from_disk(f"data/rm_{name}")

tok = AutoTokenizer.from_pretrained(BASE)
# fp32 master weights; bf16=True below runs the math in bf16 autocast.
# Eager attention: sdpa produced NaN on padded batches with torch 2.13 / transformers 5.19.
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, dtype=torch.float32, attn_implementation="eager"
)
model.config.pad_token_id = tok.pad_token_id  # the reward is read at the last non-pad token

args = RewardConfig(
    output_dir=f"rm_{name}",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # 4 GPUs x 8 x 2 = 64 pairs per step
    per_device_eval_batch_size=16,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=20,  # ~3% of the ~670 steps in one epoch
    max_length=1024,
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
)
trainer = RewardTrainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    processing_class=tok,
)
trainer.train()
print("FINAL_EVAL", trainer.evaluate())
trainer.save_model(f"rm_{name}")
tok.save_pretrained(f"rm_{name}")
