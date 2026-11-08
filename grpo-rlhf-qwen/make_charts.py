"""Twitter-ready charts (1600x900 PNG) for the hh-rlhf GRPO runs -> figures/.

  CUDA_VISIBLE_DEVICES= python make_charts.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import REFUSAL

MODELS = [  # (eval tag, label, color)
    ("base", "Original", "#6b7280"),
    ("helpful", "Helpful only", "#2563eb"),
    ("harmless_ck500", "Harmless only (ckpt 500)", "#dc2626"),
    ("combined", "Helpful + harmless", "#7c3aed"),
]
RUNS = [("helpful", "Helpful only", "#2563eb"), ("harmless", "Harmless only", "#dc2626"),
        ("combined", "Helpful + harmless", "#7c3aed")]
FIGSIZE, DPI = (8, 4.5), 200  # 1600x900
SUPTITLE = dict(fontsize=20, fontweight="bold", x=0.015, ha="left")
GREY = "#4b5563"
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25,
})


def eval_stats():
    stats = {}
    for tag, *_ in MODELS:
        rows = [json.loads(l) for l in open(f"eval/{tag}_scored.jsonl")]
        ordinary = [r for r in rows if r["source"] == "helpful-base"]
        redteam = [r for r in rows if r["source"] == "harmless-base"]
        refused = lambda rs: sum(bool(REFUSAL.search(r["answer"][:300])) for r in rs)
        stats[tag] = dict(
            helpful=np.mean([r["rm_helpful"] for r in ordinary]),
            harmless=np.mean([r["rm_harmless"] for r in redteam]),
            ref_ord=refused(ordinary), n_ord=len(ordinary),
            ref_red=refused(redteam), n_red=len(redteam),
        )
    print(f"{'model':16} {'ord helpfulRM':>13} {'ord refused':>11} {'red harmlessRM':>14} {'red refused':>11}")
    for tag, s in stats.items():
        print(f"{tag:16} {s['helpful']:13.2f} {s['ref_ord']:>7}/{s['n_ord']} {s['harmless']:14.2f} {s['ref_red']:>7}/{s['n_red']}")
    return stats


def tradeoff(stats):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x0, y0 = stats["base"]["helpful"], stats["base"]["harmless"]
    offsets = {"base": (-16, -12), "helpful": (-16, 0), "harmless_ck500": (-6, 30), "combined": (-12, 30)}
    for tag, label, color in MODELS:
        x, y = stats[tag]["helpful"], stats[tag]["harmless"]
        if tag != "base":
            ax.annotate("", (x, y), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                                              alpha=0.7, shrinkA=8, shrinkB=9, mutation_scale=20))
        ax.scatter(x, y, s=220, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ha = "left" if tag == "harmless_ck500" else "right"
        ax.annotate(f"{label}\n({x:+.2f}, {y:+.2f})", (x, y), xytext=offsets[tag], textcoords="offset points",
                    ha=ha, va="center", fontsize=14, color=color, fontweight="bold", linespacing=1.1)
    ax.set_xlim(-8.2, 0.9)
    ax.set_ylim(-0.7, 6.0)
    ax.set_xlabel("more helpful on normal questions  →")
    ax.set_ylabel("safer on harmful requests  →")
    fig.suptitle("One model, three kinds of human feedback", **SUPTITLE)
    ax.set_title("Qwen2.5-1.5B + GRPO on hh-rlhf reward models (mean RM score)", fontsize=13, color=GREY, loc="left")
    fig.tight_layout()
    fig.savefig("figures/tradeoff.png", dpi=DPI)


def refusals(stats):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(MODELS))
    w = 0.38
    for i, (key, n, name) in enumerate([("ref_ord", "n_ord", "Ordinary questions"),
                                        ("ref_red", "n_red", "Harmful (red-team) requests")]):
        vals = [100 * stats[t][key] / stats[t][n] for t, *_ in MODELS]
        bars = ax.bar(x + (i - 0.5) * 0.42, vals, w, color=[c for *_, c in MODELS], alpha=0.45 if i == 0 else 0.95,
                      edgecolor=[c for *_, c in MODELS], linewidth=1.5, label=name)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=2, fontsize=13, fontweight="bold")
    ax.set_xticks(x, ["Original", "Helpful\nonly", "Harmless\nonly", "Helpful +\nharmless"])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.grid(axis="x", visible=False)
    leg = ax.legend(loc="upper left", frameon=False, borderaxespad=0.1)
    for h in leg.legend_handles:
        h.set_color("#6b7280")
    leg.legend_handles[0].set_alpha(0.45)
    fig.suptitle("How often does each model refuse?", **SUPTITLE)
    ax.set_title('200 prompts each; refusal = keyword match ("I\'m sorry", "I can\'t", ...)',
                 fontsize=13, color=GREY, loc="left")
    fig.tight_layout()
    fig.savefig("figures/refusals.png", dpi=DPI)


def rolling(v, k=20, fn=np.mean):
    return np.array([fn(v[max(0, i - k + 1):i + 1]) for i in range(len(v))])


def training():
    small = {"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 14, "xtick.labelsize": 13,
             "ytick.labelsize": 13, "legend.fontsize": 15}
    with plt.rc_context(small):
        fig, axes = plt.subplots(1, 3, figsize=(16, 9))  # saved at dpi=100 -> 1600x900
        for run, label, color in RUNS:
            log = [h for h in json.load(open(f"grpo_{run}/checkpoint-1000/trainer_state.json"))["log_history"]
                   if "kl" in h]
            step = np.array([h["step"] for h in log])
            length = np.array([h["completions/mean_length"] for h in log])
            reward = np.array([h["reward"] for h in log])
            kl = np.array([h["kl"] for h in log])
            gain = reward - reward[:20].mean()
            print(f"{run:9} steps={step[-1]} len start/end={length[:20].mean():.0f}/{length[-20:].mean():.0f} "
                  f"reward gain(last20)={gain[-20:].mean():+.2f} max kl={kl.max():.0f}@{step[kl.argmax()]}")
            for ax, v, fn in ((axes[0], length, np.mean), (axes[1], gain, np.mean), (axes[2], kl, np.median)):
                ax.plot(step, rolling(v, fn=fn), color=color, lw=3, label=label)

        panels = ((axes[0], "Answer length", "tokens per answer"),
                  (axes[1], "Reward gain", "reward-model score vs. step 0"),
                  (axes[2], "Drift from the original model", "KL divergence (log scale)"))
        for ax, title, ylabel in panels:
            ax.set_title(title, loc="left", fontweight="bold", pad=14)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("training step")
            ax.set_xlim(0, 1000)
            ax.set_xticks([0, 250, 500, 750, 1000])

        axes[0].set_ylim(0, 200)
        axes[0].text(120, 60, "harmless: ~17 tokens", color="#dc2626", fontsize=14, fontweight="bold")
        axes[1].axhline(0, color="#9ca3af", lw=1)
        axes[2].set_yscale("log")
        axes[2].set_ylim(1e-3, 1e5)
        axes[2].set_yticks([1e-2, 1, 1e2, 1e4], ["0.01", "1", "100", "10,000"])
        axes[2].minorticks_off()
        axes[2].axvline(500, color="#dc2626", lw=1.2, ls="--")
        axes[2].text(515, 2e-3, "checkpoint 500\n(used for eval)", color="#dc2626", fontsize=12)
        axes[2].text(480, 1.5e4, "single-step peak\n~1.4M at step 643", color="#dc2626", fontsize=12, va="center", ha="right")

        fig.suptitle("GRPO training: 1000 steps per run", fontsize=26, fontweight="bold", x=0.04, ha="left", y=0.97)
        fig.text(0.04, 0.895, "Each run is scored by its own reward model. Lines are 20-step rolling averages "
                 "(median for KL drift).", fontsize=15, color=GREY)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.035, 0.87), ncol=3, frameon=False,
                   handlelength=1.6, columnspacing=2.0)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.1, wspace=0.3)
        fig.savefig("figures/training.png", dpi=100)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    stats = eval_stats()
    tradeoff(stats)
    refusals(stats)
    training()
    print("wrote figures/tradeoff.png figures/refusals.png figures/training.png")
