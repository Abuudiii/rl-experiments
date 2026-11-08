"""Convert Anthropic/hh-rlhf (helpful-base, harmless-base) into chat-format datasets.

Outputs (under data/):
  rm_helpful/, rm_harmless/   preference pairs {prompt, chosen, rejected} for reward-model training
  prompts/                    prompt-only {prompt} mix of both subsets, for GRPO (train) and eval (test)
"""
import re

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_TOKENS = 512
TURN = re.compile(r"\n\n(Human|Assistant): ")
tok = AutoTokenizer.from_pretrained(BASE)


def to_messages(text):
    parts = TURN.split(text)
    if parts[0].strip():
        return None  # text before the first turn: malformed
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        role = "user" if role == "Human" else "assistant"
        if msgs and msgs[-1]["role"] == role:
            return None  # two turns in a row from the same speaker
        msgs.append({"role": role, "content": content.strip()})
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    return msgs


def to_pair(row):
    c, r = to_messages(row["chosen"]), to_messages(row["rejected"])
    if c is None or r is None or c[:-1] != r[:-1]:
        return {"prompt": None, "chosen": None, "rejected": None}
    if not c[-1]["content"] or not r[-1]["content"] or c[-1]["content"] == r[-1]["content"]:
        return {"prompt": None, "chosen": None, "rejected": None}
    return {"prompt": c[:-1], "chosen": [c[-1]], "rejected": [r[-1]]}


def prompt_len(row):
    ids = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
    return {"n_prompt_tokens": len(ids)}


out = {}
for subset in ("helpful-base", "harmless-base"):
    raw = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
    splits = {}
    for split in ("train", "test"):
        ds = raw[split].map(to_pair, remove_columns=raw[split].column_names)
        before = len(ds)
        ds = ds.filter(lambda x: x["prompt"] is not None)
        ds = ds.map(prompt_len).filter(lambda x: x["n_prompt_tokens"] <= MAX_PROMPT_TOKENS)
        print(f"{subset}/{split}: kept {len(ds)} of {before}")
        splits[split] = ds.add_column("source", [subset] * len(ds))
    out[subset] = DatasetDict(splits)
    DatasetDict({k: v.remove_columns(["n_prompt_tokens", "source"]) for k, v in splits.items()}).save_to_disk(
        f"data/rm_{subset.split('-')[0]}"
    )


def unique_prompts(ds):
    seen, rows = set(), []
    for p, s in zip(ds["prompt"], ds["source"]):
        key = str(p)
        if key not in seen:
            seen.add(key)
            rows.append({"prompt": p, "source": s})
    return Dataset.from_list(rows)


prompts = {}
for split, n in (("train", 4000), ("test", 200)):
    halves = [unique_prompts(out[s][split]).shuffle(seed=0).select(range(n)) for s in out]
    prompts[split] = concatenate_datasets(halves).shuffle(seed=0)
    print(f"prompts/{split}: {len(prompts[split])}")
DatasetDict(prompts).save_to_disk("data/prompts")
