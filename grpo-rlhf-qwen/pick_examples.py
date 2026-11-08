"""Print side-by-side base / helpful / harmless / combined answers for the most telling held-out prompts.
"harmless" means checkpoint 500 (the final harmless model drifted after a KL spike at step 600).
Run after `evaluate.py score base helpful harmless_ck500 combined`. Usage: python pick_examples.py [n_per_case]
"""
import json
import sys

from evaluate import REFUSAL

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MODELS = {"base": "base", "helpful": "helpful", "harmless": "harmless_ck500", "combined": "combined"}
rows = {name: [json.loads(l) for l in open(f"eval/{tag}_scored.jsonl")] for name, tag in MODELS.items()}
refuses = lambda r: bool(REFUSAL.search(r["answer"][:300]))


def show(title, idxs):
    print(f"\n######## {title} ########")
    for i in idxs[:n]:
        last_user = rows["base"][i]["prompt"][-1]["content"]
        print(f"\nUSER: {last_user[:300]}")
        for t in MODELS:
            r = rows[t][i]
            print(f"  [{t:8} | helpfulRM {r['rm_helpful']:+.2f} | harmlessRM {r['rm_harmless']:+.2f}] {r['answer'][:300]!r}")


idx = range(len(rows["base"]))
# Normal questions: base answered, harmless model refused (over-refusal)
over = [i for i in idx if rows["base"][i]["source"] == "helpful-base" and not refuses(rows["base"][i]) and refuses(rows["harmless"][i])]
# Red-team questions: harmless model refused, helpful model did not
comply = [i for i in idx if rows["base"][i]["source"] == "harmless-base" and refuses(rows["harmless"][i]) and not refuses(rows["helpful"][i])]
over.sort(key=lambda i: rows["helpful"][i]["rm_helpful"] - rows["harmless"][i]["rm_helpful"], reverse=True)
comply.sort(key=lambda i: rows["harmless"][i]["rm_harmless"] - rows["helpful"][i]["rm_harmless"], reverse=True)
print(f"normal questions the harmless model refused: {len(over)}/200")
print(f"red-team questions the helpful model went along with (harmless refused): {len(comply)}/200")
for kind, src in (("normal", "helpful-base"), ("red-team", "harmless-base")):
    for t in MODELS:
        k = sum(refuses(r) for r in rows[t] if r["source"] == src)
        print(f"  {kind:8} questions refused by {t:8}: {k}/200")
show("Normal question -> harmless model refuses", over)
show("Red-team question -> helpful model complies", comply)
