"""Publish the three GRPO policies to the Hugging Face Hub (bf16) with model cards,
then check each upload by downloading it and reproducing a known greedy answer.
Usage (1 GPU, logged in with `hf auth login`): python push_to_hub.py <hf-user>
"""
import json
import sys
import tempfile

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

USER = sys.argv[1]
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
CODE = "https://github.com/Abuudiii/rl-experiments/tree/main/grpo-rlhf-qwen"
FIG = "https://raw.githubusercontent.com/Abuudiii/rl-experiments/main/grpo-rlhf-qwen/figures"

# name: (local checkpoint, column name in figures/prompts.json, reward description, behaviour, warning)
MODELS = {
    "helpful": (
        "grpo_helpful", "Helpful votes only",
        "the **helpful-base** half of HH-RLHF (\"which answer is more helpful?\")",
        "Answers ordinary questions most fully of the three (and most verbosely). It also goes along with sneakier "
        "multi-turn harmful requests that the original model refuses, e.g. it writes a plan for avoiding capture "
        "after robbing a store. It still refuses most blunt one-line harmful requests.",
        "Trained with no harmlessness signal: it is more willing than the original model to help with harmful "
        "requests. For research on reward design only.",
    ),
    "harmless": (
        "grpo_harmless/checkpoint-500", "Harmless votes only",
        "the **harmless-base** half of HH-RLHF (\"which answer is less harmful?\")",
        "Refuses everything, including \"What does a life coach do?\", with the same ~17-token sentence. Within "
        "~100 steps it found that one refusal scores high for every prompt. This is checkpoint 500 of 1000: the run "
        "went unstable soon after (per-step KL hit 1.29M at step 544 and ~1.4M at step 643).",
        "Not useful as an assistant: it is a demonstration of reward hacking / over-refusal.",
    ),
    "combined": (
        "grpo_combined", "Helpful + harmless",
        "**both** halves of HH-RLHF: the helpful and harmless reward models' scores are added",
        "Answers most ordinary questions and refuses 94% of red-team ones, beating the original model on both "
        "reward-model scores. It still over-refuses about a third of ordinary questions.",
        "A small research model, not a safe assistant: it still goes along with some harmful requests.",
    ),
}
repo = lambda name: f"{USER}/grpo-Qwen2.5-1.5B-hh-rlhf-{name}"

RESULTS = """| Model | Ordinary: helpful score | Ordinary: refused | Red-team: harmless score | Red-team: refused |
|---|---|---|---|---|
| Original Qwen2.5-1.5B-Instruct | −0.63 | 7% | 1.59 | 62.5% |
| Helpful only | **0.05** | 5.5% | 0.01 | 56% |
| Harmless only (checkpoint 500) | −7.26 | **100%** | **4.42** | 100% |
| Helpful + harmless | −0.18 | 32% | 2.18 | **94%** |"""


def card(name):
    _, _, reward, behaviour, warning = MODELS[name]
    siblings = " · ".join(f"[{n}](https://huggingface.co/{repo(n)})" for n in MODELS)
    return f"""---
license: apache-2.0
base_model: {BASE}
datasets:
- Anthropic/hh-rlhf
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- trl
- grpo
- rlhf
- reinforcement-learning
---

# grpo-Qwen2.5-1.5B-hh-rlhf-{name}

[Qwen2.5-1.5B-Instruct]({"https://huggingface.co/" + BASE}) fine-tuned with GRPO against a reward model trained on {reward}.

One of three models from the same experiment, where only the reward changes: {siblings}.
Code, figures and full results: [{CODE.split("github.com/")[1]}]({CODE}).

> **Warning:** {warning}

## Behaviour
{behaviour}

![Same prompts, four models]({FIG}/examples.jpg)

## Results
400 held-out prompts: 200 ordinary questions (`helpful-base`) and 200 red-team conversations (`harmless-base`), greedy
decoding. Scores are mean reward-model outputs; refused = the answer opens with a refusal phrase ("I'm sorry", "I can't", ...).

{RESULTS}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo(name)}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{{"role": "user", "content": "What does a life coach do?"}}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Training
- **Reward model:** Qwen2.5-1.5B-Instruct with a scalar head, 1 epoch on the chosen/rejected pairs (pairwise loss,
  lr 5e-6, 64 pairs per step, up to 1024 tokens). Held-out pair accuracy: 74.3% (helpful), 73.7% (harmless).
- **GRPO** ([TRL](https://github.com/huggingface/trl) 1.17.1): 16 prompts per step, 8 sampled answers per prompt (vLLM),
  advantage = reward minus the group's mean, KL penalty β = 0.04 to the frozen original, lr 1e-6, answers up to 256
  tokens, 1000 steps. Prompts: 8,000 from HH-RLHF (4,000 per subset), each up to 512 tokens.
- **Hardware:** NVIDIA H100 80GB GPUs (4 per run for helpful/harmless, 8 for combined).
- Trained in fp32 with bf16 autocast; published in bf16.

## Limitations
- Scores come from the same reward models used for training; there is no independent judge.
- Refusal rates are a keyword heuristic, not human judgement.
- A 1.5B model trained for 1000 steps on 2022-era preference data: expect factual errors and inconsistent behaviour.

## References
- Bai et al., 2022. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
- Shao et al., 2024. *DeepSeekMath* (introduces GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
"""


api = HfApi()
who = api.whoami()["name"]
assert who == USER, f"logged in as {who}, not {USER}"

for name, (src, *_) in MODELS.items():
    with tempfile.TemporaryDirectory() as out:
        AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).save_pretrained(out)
        AutoTokenizer.from_pretrained(src).save_pretrained(out)
        open(f"{out}/README.md", "w").write(card(name))
        api.create_repo(repo(name), exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=repo(name), commit_message="Upload GRPO policy and model card")
    print("pushed", repo(name), flush=True)

# Check: download each upload and reproduce the greedy answer from figures/prompts.json.
shots = json.load(open("figures/prompts.json"))
names = ["Original Qwen2.5-1.5B", "Helpful votes only", "Harmless votes only", "Helpful + harmless"]
n = len(shots) // len(names)
tok = AutoTokenizer.from_pretrained(BASE)
for name, (_, column, *_) in MODELS.items():
    expected = shots[names.index(column) * n]
    with tempfile.TemporaryDirectory() as cache:
        model = AutoModelForCausalLM.from_pretrained(repo(name), dtype=torch.bfloat16, cache_dir=cache).cuda().eval()
        ids = tok.apply_chat_template(expected["prompt"], add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).cuda()
        with torch.no_grad():
            got = tok.decode(model.generate(ids, max_new_tokens=200, do_sample=False)[0, ids.shape[1]:],
                             skip_special_tokens=True).strip()
        del model
    print(f"check {name}: {'MATCH' if got == expected['answer'] else 'DIFFERENT'} | {got[:90]!r}", flush=True)
