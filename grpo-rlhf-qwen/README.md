# RLHF with GRPO: helpful vs. harmless
Fine-tune Qwen2.5-1.5B-Instruct three ways using reward models trained on human preference votes from
[Anthropic's HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf): rewarded for helpfulness only, harmlessness
only, and both summed. Same prompts and settings in every run; only the reward changes.

**Models:** [helpful](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-helpful) ·
[harmless](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-harmless) ·
[combined](https://huggingface.co/Abuudiii/grpo-Qwen2.5-1.5B-hh-rlhf-combined)

![Same prompts, four models](figures/examples.jpg)

## Result
400 held-out prompts: 200 ordinary questions (`helpful-base`) and 200 red-team conversations (`harmless-base`).
Scores are mean reward-model outputs. Refused = the answer opens with a refusal phrase ("I'm sorry", "I can't", …).

| Model | Ordinary: helpful score | Ordinary: refused | Red-team: harmless score | Red-team: refused |
|---|---|---|---|---|
| Original Qwen2.5-1.5B-Instruct | −0.63 | 7% | 1.59 | 62.5% |
| Helpful only | **0.05** | 5.5% | 0.01 | 56% |
| Harmless only (checkpoint 500) | −7.26 | **100%** | **4.42** | 100% |
| Helpful + harmless | −0.18 | 32% | 2.18 | **94%** |

- **Helpful only** goes along with sneakier multi-turn harmful requests. The original model refuses to give advice on
  avoiding capture after robbing a store; this one writes a numbered plan. Blunt one-line harmful requests it still refuses.
- **Harmless only** collapses within ~100 steps into one ~17-token refusal for everything, including
  "What does a life coach do?"
- **Both summed** answers most ordinary questions and refuses 94% of red-team ones, beating the original model on both
  scores. It still over-refuses a third of ordinary questions.

Full table, including harmless checkpoints 100–700 and the final step 1000: [`eval_scores.txt`](eval_scores.txt).

| | |
|---|---|
| ![tradeoff](figures/tradeoff.png) | ![refusals](figures/refusals.png) |

![training](figures/training.png)

The harmless run went unstable soon after step 500: per-step KL hit 1.29M at step 544 and ~1.4M at step 643, then its
reward fell and answers got longer. That's why it is evaluated at checkpoint 500.

## How it works
1. **Reward models.** Qwen2.5-1.5B-Instruct with a scalar head, trained for 1 epoch on the chosen/rejected pairs of one
   subset (pairwise loss, lr 5e-6, 64 pairs per step). Accuracy on held-out pairs: **74.3%** helpful, **73.7%** harmless.
2. **GRPO** with TRL. Each step takes 16 prompts and samples 8 answers per prompt with vLLM. Each answer's advantage is
   its reward minus the average of its group of 8, so the group plays the role of PPO's critic. A KL penalty
   (β = 0.04) keeps the policy close to the frozen original. 1000 steps, lr 1e-6, answers up to 256 tokens.
   The combined run adds both reward models' scores.
3. **Prompts.** 8,000 training prompts (4,000 per subset) and 400 held-out, each at most 512 tokens.

Trained on NVIDIA H100 80GB GPUs: the helpful and harmless runs side by side on 4 GPUs each, the combined run on all 8
(51 minutes).

Two implementation notes: SDPA attention produced NaN gradients on padded batches with torch 2.13 / transformers 5.19,
so the reward models and the GRPO policy use eager attention; and the reward models keep fp32 weights with bf16 autocast.

## Setup
Python 3.12, CUDA 13.
```bash
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 transformers==5.19.0 trl==1.17.1 vllm==0.32.0 accelerate==1.15.0 \
  datasets==5.1.0 matplotlib==3.11.2 numpy==2.3.5
```

## Run
```bash
bash run_all.sh   # data -> 2 reward models -> 3 GRPO runs -> evaluation -> figures (expects 8 GPUs)
```

| Script | Does |
|---|---|
| `prep_data.py` | Converts HH-RLHF to chat format: preference pairs per subset and the prompt sets |
| `train_rm.py helpful\|harmless` | Trains one reward model |
| `train_grpo.py helpful\|harmless\|combined [steps]` | GRPO against one reward model, or both summed |
| `evaluate.py gen <model> <tag>` / `score <tags…>` | Greedy answers to the held-out prompts; scores them with both reward models |
| `prompt_shots.py`, `render_heldout.py`, `make_examples.py`, `make_charts.py` | The figures |
| `pick_examples.py` | Prints prompts where the models disagree most |
| `push_to_hub.py <hf-user>` | Publishes the three policies (bf16) with model cards, then checks each upload |

Raw answers aren't in the repo: the red-team half of HH-RLHF contains offensive requests, and the helpful-only
model's answers to them are not something to publish. The model weights are on the Hub (links above).

## Caveats
- Refusal rates come from a keyword match, not human judgment.
- Answers are scored by the same reward models the policies were trained on; there is no independent judge.
- The helpful-only model's answers are ~25% longer than the original's, which can inflate its own reward model's score.
