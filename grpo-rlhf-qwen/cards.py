"""Draw one 1920x1080 image with four model answers to the same prompt (2x2 boxes)."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#6b7280", "#2563eb", "#dc2626", "#7c3aed"]  # same as make_charts.py: grey, blue, red, purple


def draw_card(title, entries, path, subtitle=None):
    """entries: list of 4 (model_name, answer, helpful_score, harmless_score)."""
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.text(0.5, 0.955, title, ha="center", va="center", fontsize=20, weight="bold")
    if subtitle:
        fig.text(0.5, 0.915, subtitle, ha="center", va="center", fontsize=12, color="#6b7280")
    for j, ((name, answer, h, s), c) in enumerate(zip(entries, COLORS)):
        ax = fig.add_axes([0.02 + (j % 2) * 0.49, 0.45 - (j // 2) * 0.44, 0.47, 0.42])
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(c), spine.set_linewidth(2.5)
        ax.text(0.02, 0.95, name, transform=ax.transAxes, fontsize=15, weight="bold", color=c, va="top")
        ax.text(0.98, 0.95, f"helpful {h:+.1f} | harmless {s:+.1f}", transform=ax.transAxes,
                fontsize=11, color="#374151", va="top", ha="right")
        body = answer if len(answer) <= 520 else answer[:520].rsplit(" ", 1)[0] + " …"
        wrapped = "\n".join(textwrap.fill(line, 78) for line in body.splitlines() if line.strip())
        ax.text(0.02, 0.83, "\n".join(wrapped.splitlines()[:10]), transform=ax.transAxes, fontsize=12, va="top")
    fig.savefig(path)
    plt.close(fig)
