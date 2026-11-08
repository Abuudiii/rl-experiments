"""Render four-way answer images for chosen held-out red-team conversations (already generated + scored).
Usage: python render_heldout.py   -> figures/heldout_<n>.png
"""
import json

from cards import draw_card

MODELS = {
    "Original Qwen2.5-1.5B": "base",
    "Helpful votes only": "helpful",
    "Harmless votes only": "harmless_ck500",
    "Helpful + harmless": "combined",
}
# Row indices in eval/*_scored.jsonl: clearly harmful, but fine to show publicly.
PICKS = [238, 22]  # 238: avoid getting caught after robbing a store; 22: secretly track a spouse

rows = {name: [json.loads(l) for l in open(f"eval/{tag}_scored.jsonl")] for name, tag in MODELS.items()}
for n, i in enumerate(PICKS, 1):
    convo = rows["Helpful votes only"][i]["prompt"]
    entries = [(name, rows[name][i]["answer"], rows[name][i]["rm_helpful"], rows[name][i]["rm_harmless"]) for name in MODELS]
    turns = len([m for m in convo if m["role"] == "user"])
    sub = "Held-out red-team conversation from Anthropic hh-rlhf"
    if turns > 1:
        sub += f" ({turns} turns; it began: \u201c{convo[0]['content'][:90]}\u201d)"
    draw_card(f"Prompt: \u201c{convo[-1]['content'][:110]}\u201d", entries, f"figures/heldout_{n}.png", subtitle=sub)
    print(f"heldout_{n}: {turns} user turns | " + " | ".join(f"{name}: {rows[name][i]['answer'][:60]!r}" for name in MODELS))
