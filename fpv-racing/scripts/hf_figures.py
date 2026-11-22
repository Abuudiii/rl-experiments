"""Figures for the model page (Hugging Face: Abuudiii/fpv-perception-pipeline), same black-and-white style as
`post_figures.architecture_figure`. Every number is read from the run outputs named next to it or copied from the
RESULTS.md table named in the comment.

    uv run --no-project --with matplotlib --with numpy python scripts/hf_figures.py --out perception-pipeline/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from post_figures import load  # noqa: E402

RUNS = Path("runs")
DPI = 150
TRACKS = ["circuit", "clockwise", "figure8", "slalom", "climb_dive", "sprint_hairpin"]
NAMES = {"random": "random tracks", "circuit": "circuit", "clockwise": "clockwise", "figure8": "figure-8",
         "slalom": "slalom", "climb_dive": "climb and dive", "sprint_hairpin": "sprint + hairpin", "split_s": "split-S"}
FILLS = ["k", "0.6", "w", "0.85"]  # first series black (the pipeline), then greys / outline


def jload(path):
    return json.loads(Path(path).read_text())


def style(ax, xmax=118, pct=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if pct:
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"], fontsize=10)


def grouped_barh(ax, groups, series, values, labels, height=0.8, fs=10, fills=FILLS):
    """values[g][s] (None = no bar), labels[g][s] text next to the bar. Groups top to bottom."""
    k = len(series)
    h = height / k
    y0 = np.arange(len(groups))[::-1].astype(float)
    xmax = ax.get_xlim()[1]
    for s, name in enumerate(series):
        for g in range(len(groups)):
            v = values[g][s]
            if v is None:
                continue
            y = y0[g] + height / 2 - h * (s + 0.5)
            ax.barh(y, v, height=h * 0.9, color=fills[s], edgecolor="k", lw=0.7, label=name if g == 0 else None)
            ax.text(v + xmax * 0.012, y, labels[g][s], va="center", fontsize=fs)
    ax.set_yticks(y0, groups, fontsize=11)
    ax.set_ylim(-0.6, len(groups) - 0.4)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=10, handlelength=1.4, **kw)


def note(fig, text, y=0.015):
    fig.text(0.01, y, text, fontsize=9, color="0.3", va="bottom")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------------------------- diagrams


def box(ax, x, y, w, h, title, body="", learned=False, fs_title=11.5, fs_body=9.5, align="center"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="0.85" if learned else "w", ec="k", lw=1.0))
    if align == "center":
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=fs_title, weight="bold")
        if body:
            ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", fontsize=fs_body, linespacing=1.3)
    else:
        ax.text(x + 0.15, y + h - 0.2, title, ha="left", va="top", fontsize=fs_title, weight="bold")
        if body:
            ax.text(x + 0.15, y + h - 0.58, body, ha="left", va="top", fontsize=fs_body, linespacing=1.3)


def arrow(ax, a, b, text=None, **kw):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color="k", lw=1.0, shrinkA=0, shrinkB=0, **kw))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.1, text, ha="center", va="bottom", fontsize=9, style="italic")


def legend_boxes(ax, x, y):
    ax.add_patch(plt.Rectangle((x, y), 0.22, 0.22, fc="0.85", ec="k", lw=1))
    ax.text(x + 0.32, y + 0.11, "learned", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((x + 1.45, y), 0.22, 0.22, fc="w", ec="k", lw=1))
    ax.text(x + 1.77, y + 0.11, "written code / data", va="center", fontsize=10)


def estimator_dataflow(out):
    fig = plt.figure(figsize=(16, 9.6), dpi=DPI, facecolor="w")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0, 9.6))
    ax.axis("off")
    ax.text(0.3, 9.2, "Inside the state estimator (one 50 Hz step)", fontsize=15, weight="bold")
    legend_boxes(ax, 12.2, 9.12)

    # inputs
    box(ax, 0.3, 6.55, 3.0, 1.6, "Detections", "from the gate detector:\n4 corners per visible gate\n+ a confidence (keep >= 0.3)",
        learned=True)
    box(ax, 0.3, 3.9, 3.0, 1.6, "IMU (simulated)", "gyro: body rates\naccelerometer: thrust + drag\n(noise and bias at x1)")
    box(ax, 0.3, 1.25, 3.0, 1.6, "Course list", "nominal centre and heading\nof the 6 gates, in order;\ngiven before the flight")

    # steps
    steps = [
        ("1  PnP per detection", "square-gate fit: centre, plane, quality"),
        ("2  Tilt", "gyro integration, corrected by the\nupright edges of near gates"),
        ("3  Kalman filter", "velocity + 6 gate positions; predict\nwith accelerometer, update with PnP"),
        ("4  Course placement", "list fitted to matched gates (yaw +\nshift); predicts gates not in view"),
        ("5  Match and lock", "detections to list gates; first lock\nand relocalisation by hypotheses"),
        ("6  Which gate is next", "pass test on the estimated gate plane\n+ a consistency rule every frame"),
    ]
    bx, bw, bh, top, gap = 4.5, 4.4, 1.08, 8.15, 0.14
    ys = []
    for i, (t, b) in enumerate(steps):
        y = top - (i + 1) * bh - i * gap
        ys.append(y + bh / 2)
        box(ax, bx, y, bw, bh, t, b, fs_title=11, fs_body=9.2, align="left")
    ax.text(bx + bw / 2, top + 0.1, "written code, no learning", ha="center", va="bottom", fontsize=9.5, style="italic")
    # input arrows into the step column
    arrow(ax, (3.3, 7.35), (bx, ys[0]))
    arrow(ax, (3.3, 4.85), (bx, ys[1]), text="gyro")
    arrow(ax, (3.3, 4.55), (bx, ys[2]), text="accel.")
    arrow(ax, (3.3, 2.05), (bx, ys[3]))
    arrow(ax, (3.3, 1.85), (bx, ys[4]))

    # observation
    ox, ow = 10.0, 3.55
    parts = [("body velocity", 3), ("attitude (tilt only)", 9), ("body rates", 3), ("next gate, 4 corners", 12),
             ("gate after next, 4 corners", 12), ("previous action", 4)]
    flags = [("age of next-gate info", 1), ("age of after-next info", 1), ("locked yes / no", 1)]
    row = 0.42
    oy_top = 8.15
    oh = row * (len(parts) + len(flags)) + 1.45
    oy = oy_top - oh
    ax.add_patch(plt.Rectangle((ox, oy), ow, oh, fc="w", ec="k", lw=1.0))
    ax.text(ox + ow / 2, oy_top - 0.2, "Observation, 46 numbers", ha="center", va="top", fontsize=11.5, weight="bold")
    y = oy_top - 0.75
    for name, n in parts:
        ax.text(ox + 0.18, y, name, va="center", fontsize=9.8)
        ax.text(ox + ow - 0.18, y, str(n), va="center", ha="right", fontsize=9.8)
        y -= row
    ax.plot([ox + 0.12, ox + ow - 0.12], [y + row / 2, y + row / 2], color="k", lw=0.7)
    ax.text(ox + 0.18, y, "teacher's layout, estimated", va="center", fontsize=9.3, style="italic")
    ax.text(ox + ow - 0.18, y, "43", va="center", ha="right", fontsize=9.8, weight="bold")
    y -= row * 1.15
    for name, n in flags:
        ax.text(ox + 0.18, y, name, va="center", fontsize=9.8)
        ax.text(ox + ow - 0.18, y, str(n), va="center", ha="right", fontsize=9.8)
        y -= row
    ax.text(ox + 0.18, y + 0.05, "3 flags, new", va="center", fontsize=9.3, style="italic")
    ax.plot([bx + bw + 0.05, bx + bw + 0.55, bx + bw + 0.55], [ys[0], ys[0], ys[-1]], color="k", lw=1.0)
    ax.plot([bx + bw + 0.05, bx + bw + 0.55], [ys[-1], ys[-1]], color="k", lw=1.0)
    arrow(ax, (bx + bw + 0.55, (oy + oy_top) / 2), (ox, (oy + oy_top) / 2))

    # policy
    px, pw = 14.1, 1.65
    pyc = (oy + oy_top) / 2
    box(ax, px, pyc - 0.75, pw, 1.5, "Policy", "PPO, MLP\n3 x 256", learned=True)
    arrow(ax, (ox + ow, pyc), (px, pyc))
    arrow(ax, (px + pw / 2, pyc - 0.75), (px + pw / 2, pyc - 1.75))
    ax.text(px + pw / 2, pyc - 1.85, "thrust +\n3 body rates", ha="center", va="top", fontsize=9.5, style="italic")

    ax.text(0.3, 0.55, "Everything the estimator remembers is kept relative to the drone (no world frame). It never sees the "
            "drone's true position or the per-flight gate shifts.", fontsize=9.5, color="0.25")
    ax.text(0.3, 0.2, "Source: fpv/perception/estimator.py (module docstring), version v5: place_mem_w = 0.3, "
            "rescue_after_next = False.", fontsize=9.5, color="0.25")
    fig.savefig(out, facecolor="w", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def training_diagram(out):
    fig = plt.figure(figsize=(16, 8.0), dpi=DPI, facecolor="w")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0.6, 8.6))
    ax.axis("off")
    ax.text(0.3, 8.2, "How the two learned parts are trained", fontsize=15, weight="bold")
    legend_boxes(ax, 12.2, 8.12)
    w, h = 2.75, 1.85
    xs = [0.3, 3.45, 6.6, 9.75]

    ax.text(0.3, 7.45, "Gate detector: supervised, from simulator labels", fontsize=12.5, weight="bold")
    yd = 5.3
    det = [
        ("Teacher flies", "state-based PPO teacher;\nstick noise, off-course\nstarts; only moves the\ncamera", True),
        ("Synthetic frames", "320 x 180 ray caster,\nrandom looks; exact\ncorner labels from the\nsimulator", False),
        ("Train CNN", "3.9 M parameters\n120k steps, batch 64\nabout 72 min on one\nMI355X", True),
        ("Fine-tune on Unreal", "90k labelled Unreal\nframes + synthetic, 16k\nsteps; split-S: +38k\nframes, 12k steps", True),
    ]
    for x, (t, b, l) in zip(xs, det):
        box(ax, x, yd, w, h, t, b, learned=l)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, yd + h / 2), (b, yd + h / 2))

    ax.text(0.3, 4.45, "Flight policy: PPO, no frames rendered", fontsize=12.5, weight="bold")
    yp = 2.3
    pol = [
        ("True gate corners", "projected from the\nsimulator state, IMU\nsimulated", False),
        ("Noise model", "corner noise, misses,\nfalse gates, dropouts,\n1-frame delay; scale\nU(0, 4) x per flight", False),
        ("Estimator", "the same code as at\ntest time; 43 numbers\n+ 3 flags", False),
        ("PPO", "warm start = teacher;\ncritic also sees true\nstate; 2.5e9 steps,\n73 min (one MI355X)", True),
    ]
    for x, (t, b, l) in zip(xs, pol):
        box(ax, x, yp, w, h, t, b, learned=l)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, yp + h / 2), (b, yp + h / 2))

    # test time
    tx, tw = 13.15, 2.55
    box(ax, tx, 3.35, tw, 2.25, "Test time", "frame\n-> detector\n-> estimator\n-> policy", fs_body=10.5)
    arrow(ax, (xs[-1] + w, yd + h / 2), (tx, 4.95))
    arrow(ax, (xs[-1] + w, yp + h / 2), (tx, 4.0))
    ax.text(tx + tw / 2, 3.1, "the two meet only here", ha="center", va="top", fontsize=9.5, style="italic")

    ax.text(0.3, 1.45, "The noise model was calibrated on an early detector (pilot A). The final detector E makes fewer "
            "errors on synthetic frames than its x1 setting.", fontsize=9.5, color="0.25")
    ax.text(0.3, 1.1, "The teacher is a state-based PPO policy (43 true numbers in, sticks out). It labels nothing; "
            "the labels come from the simulator's camera model.", fontsize=9.5, color="0.25")
    ax.text(0.3, 0.75, "Sources: runs/percep/detector/HANDOFF.md, runs/percep/stage6/det_ft/*/config.json, "
            "runs/percep/policy/train/est_rand_v5/{config.json, metrics.jsonl}.", fontsize=9.5, color="0.25")
    fig.savefig(out, facecolor="w", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- charts


def synthetic_by_track(out):
    s5 = {k: jload(RUNS / f"percep/stage5/eval/main/{k}.json") for k in ["random", *TRACKS]}
    perfect = jload(RUNS / "percep/policy/eval/est_rand_v5/eval_perfect.json")
    teacher = jload(RUNS / "percep/policy/eval/teacher/eval_perfect.json")

    def get(d, k):
        return d["random"] if k == "random" else d["scenarios"][k]

    keys = ["random", *TRACKS]
    vals, labs, groups = [], [], []
    for k in keys:
        row = [s5[k]["frac_3_laps"], get(perfect, k)["frac_3_laps"], get(teacher, k)["frac_3_laps"]]
        vals.append([100 * v for v in row])
        labs.append([pct(v) for v in row])
        groups.append(f"{NAMES[k]}\n(n = {s5[k]['flights']:,})")
    fig, ax = plt.subplots(figsize=(12, 8.2), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.12)
    style(ax)
    grouped_barh(ax, groups, ["pipeline: frames -> detector -> estimator -> policy",
                              "same policy, perfect detections", "teacher, true state"], vals, labs)
    legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    ax.set_title("Three laps in 30 s, synthetic renderer, by track", fontsize=13, loc="left", pad=28)
    note(fig, "Each bar: 2,048 flights with gates shifted and physics randomised. Same flights for all three pilots.\n"
              "Sources: runs/percep/stage5/eval/main/*.json, runs/percep/policy/eval/{est_rand_v5, teacher}/eval_perfect.json.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def noise_robustness(out):
    conds = [("perfect", "perfect"), ("x1", "x1"), ("x2", "x2"), ("x4", "x4"), ("drop", "x1 + dropouts")]
    runs = [("est_rand_v5", "estimator v5 (main policy)"), ("est_rand_v4", "estimator v4 (earlier policy)")]
    data = {r: {c: jload(RUNS / f"percep/policy/eval/{r}/eval_{c}.json")["random"] for c, _ in conds} for r, _ in runs}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), dpi=DPI, facecolor="w", gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.17, wspace=0.35)
    groups = [f"{lab}\n(n = {data[runs[0][0]][c]['episodes']:,} each)" for c, lab in conds]
    ax = axs[0]
    style(ax)
    v = [[100 * data[r][c]["frac_3_laps"] for r, _ in runs] for c, _ in conds]
    grouped_barh(ax, groups, [lab for _, lab in runs], v, [[f"{x:.1f}%" for x in row] for row in v])
    ax.set_title("Three laps in 30 s", fontsize=12, loc="left")
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2)
    ax = axs[1]
    style(ax, pct=False)
    ax.set_xlim(0, 8.2)
    ax.set_xticks([0, 2, 4, 6, 8], ["0", "2", "4", "6", "8 s"], fontsize=10)
    v = [[data[r][c]["lap_time_s"]["flying_mean"] for r, _ in runs] for c, _ in conds]
    grouped_barh(ax, groups, [lab for _, lab in runs], v, [[f"{x:.2f} s" for x in row] for row in v])
    ax.set_title("Mean flying lap time", fontsize=12, loc="left")
    fig.suptitle("Detection noise: how much worse can the detector get?", x=0.01, ha="left", fontsize=13, y=0.985)
    note(fig, "Random tracks, 2,048 flights per bar, true corners passed through the noise model (no frames). x1 = the "
              "error level measured for detector pilot A; xS multiplies it.\nDropouts: the next gate vanishes for up to "
              "0.5 s about once a second. Teacher with true state: 100%, 5.92 s. Source: runs/percep/policy/eval/*/eval_*.json.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def estimator_versions(out):
    # runs/percep/policy/RESULTS.md, stage 4 table (random tracks, x1, 2048 flights)
    rows = [
        ("policy trained with v3,\nflown with v4 code (n = 2,048)", "est_rand_v3"),
        ("policy trained with v4\n(n = 2,048)", "est_rand_v4"),
        ("same v4 policy flown with\nv5, not retrained (n = 2,048)", "est_rand_v4_on_est_v5"),
        ("policy trained with v5\n(n = 2,048)", "est_rand_v5"),
    ]
    d = [jload(RUNS / f"percep/policy/eval/{r}/eval_x1.json")["random"] for _, r in rows]
    assert all(x["episodes"] == 2048 for x in d)
    wrong = [100 * x["perception"]["next_pose_wrong_step_frac"] for x in d]
    three = [100 * x["frac_3_laps"] for x in d]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.2, wspace=0.12)
    y = np.arange(len(rows))[::-1]
    fills = ["0.75", "0.75", "0.75", "k"]
    ax = axs[0]
    ax.barh(y, wrong, height=0.55, color=fills, edgecolor="k", lw=0.8)
    for yy, v in zip(y, wrong):
        ax.text(v + 0.08, yy, f"{v:.2f}%", va="center", fontsize=11)
    ax.set_yticks(y, [r for r, _ in rows], fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 6.6)
    ax.set_xticks([0, 2, 4, 6], ["0", "2", "4", "6 %"], fontsize=10)
    ax.set_title("Steps where the gate shown as next is wrong", fontsize=12, loc="left")
    ax = axs[1]
    style(ax)
    ax.barh(y, three, height=0.55, color=fills, edgecolor="k", lw=0.8)
    for yy, v in zip(y, three):
        ax.text(v + 1.5, yy, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_yticks(y, [""] * len(y))
    ax.set_title("Three laps in 30 s", fontsize=12, loc="left")
    fig.suptitle("Estimator v4 -> v5: the largest single gain", x=0.01, ha="left", fontsize=13, y=0.97)
    fig.text(0.01, 0.875, "v5 refits the course placement to gates seen in the last 2 s (weight 0.3), not only to "
             "the gate in the current frame. Right after a pass only one gate is in view, and v4 took the\ncourse "
             "heading from that one gate's own (shifted) yaw.", fontsize=9.5, color="0.25", va="bottom")
    note(fig, "Random tracks, 2,048 flights per bar, noise x1. 'Wrong' = shown next gate off by more than max(0.5 m, "
              "15 % of its distance). The v3 -> v4 change was code, so v3 cannot be flown on its own.\nSource: "
              "runs/percep/policy/eval/*/eval_x1.json, runs/percep/policy/RESULTS.md.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def crash_causes(out):
    # runs/percep/policy/RESULTS.md, "Crash causes at x1" (random tracks, 2048 flights)
    rows = [("estimator v4\n(est_rand_v4)", 489, 419, 40, 30), ("estimator v5\n(est_rand_v5)", 116, 99, 11, 6)]
    parts = ["after a wrong next-gate belief", "gate hit, belief right", "ground or out of bounds, belief right"]
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.8, bottom=0.27)
    y = np.arange(len(rows))[::-1]
    for yy, (name, tot, *vals) in zip(y, rows):
        left = 0
        for k, v in enumerate(vals):
            ax.barh(yy, v, left=left, height=0.5, color=["k", "0.6", "w"][k], edgecolor="k", lw=0.8,
                    label=parts[k] if yy == y[0] else None)
            if v >= 60:
                ax.text(left + v / 2, yy, str(v), ha="center", va="center", fontsize=11, color="w" if k == 0 else "k")
            left += v
        ax.text(left + 6, yy, f"{tot} crashes of 2,048 flights ({vals[0]} / {vals[1]} / {vals[2]})", va="center", fontsize=11)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 760)
    ax.set_xlabel("crashes", fontsize=10)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3)
    ax.set_title("What came before each crash", fontsize=13, loc="left", pad=34)
    note(fig, "Random tracks, noise x1, 2,048 flights each. 'Wrong belief' = the gate shown as next was off by more than "
              "max(0.5 m, 15 % of its distance) at the crash or in the 0.5 s before.\nBy chance alone 11.6 % (v4) and "
              "7.9 % (v5) of flight steps carry this label. Source: runs/percep/policy/RESULTS.md.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def detector_pilots(out):
    # runs/percep/detector/RESULTS.md pilot table (random held-out tracks, 32,000 synthetic frames)
    rows = [("A  default (stride 4)", 0.123, 0.99, 99.71, "40k"), ("B  stride 2", 0.096, 0.89, 99.88, "40k"),
            ("C  wider net", 0.122, 0.96, 99.68, "40k"), ("D  occlusion labels", 0.124, 1.01, 99.68, "40k"),
            ("F  stride 2 + wider", 0.094, 0.89, 99.90, "40k"), ("E  stride 2, longer", 0.079, 0.79, 99.94, "120k")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.83, bottom=0.2, wspace=0.12)
    y = np.arange(len(rows))[::-1]
    fills = ["0.75"] * 5 + ["k"]
    ax = axs[0]
    v = [r[1] for r in rows]
    ax.barh(y, v, height=0.55, color=fills, edgecolor="k", lw=0.8)
    for yy, x, r in zip(y, v, rows):
        ax.text(x + 0.002, yy, f"{x:.3f} px  ({r[4]} steps)", va="center", fontsize=10.5)
    ax.set_yticks(y, [f"{r[0]}\n(n = 32,000 frames)" for r in rows], fontsize=10.5)
    ax.set_xlim(0, 0.19)
    ax.set_title("Corner error, median (lower is better)", fontsize=12, loc="left")
    ax = axs[1]
    v = [100 - r[3] for r in rows]
    ax.barh(y, v, height=0.55, color=fills, edgecolor="k", lw=0.8)
    for yy, x in zip(y, v):
        ax.text(x + 0.005, yy, f"{x:.2f}%", va="center", fontsize=10.5)
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlim(0, 0.42)
    ax.set_title("Next gate missed when in view", fontsize=12, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="y", length=0)
    fig.suptitle("Detector pilots on synthetic frames; E was kept", x=0.01, ha="left", fontsize=13, y=0.97)
    note(fig, "Held-out random tracks, 32,000 frames per bar (about 30,000 with the next gate in view), each at its "
              "best-F1 threshold. Training batch 64, seed 0.\nWider nets (C, F) gave nothing over their narrow twins, "
              "so the extra steps went to the narrow stride-2 net (E). Source: runs/percep/detector/RESULTS.md.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def detector_unreal(out):
    s6 = RUNS / "percep/stage6"
    zs, ft = jload(s6 / "detector_E_zeroshot/detect.json")["sets"], jload(s6 / "detector_ftB/detect.json")["sets"]
    b_acro, c_acro = jload(s6 / "detector_ftB_acro/detect.json")["sets"], jload(s6 / "detector_ftC/detect.json")["sets"]

    def nvis(d):
        return round(d["frames"] * d["next_visible_frac"])

    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), dpi=DPI, facecolor="w", gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.8, bottom=0.15, wspace=0.42)
    ax = axs[0]
    style(ax)
    keys = ["random_heldout", *TRACKS]
    groups = [f"{NAMES.get(k, 'random tracks')}\n(n = {nvis(zs[k]['unreal']):,})" for k in keys]
    v = [[100 * zs[k]["unreal"]["next_detected"], 100 * ft[k]["unreal"]["next_detected"]] for k in keys]
    grouped_barh(ax, groups, ["trained on synthetic frames only (E)", "fine-tuned on Unreal frames (B_all)"], v,
                 [[f"{x:.1f}%" for x in row] for row in v], fills=["0.75", "k"])
    ax.set_title("Unreal frames, by track", fontsize=12, loc="left")
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=1)
    ax = axs[1]
    style(ax)
    sub = [("upright", "upright"), ("inverted", "upside down")]
    groups = [f"{lab}\n(n = {nvis(b_acro['acro_heldout'][k]):,})" for k, lab in sub]
    v = [[100 * b_acro["acro_heldout"][k]["next_detected"], 100 * c_acro["acro_heldout"][k]["next_detected"]] for k, _ in sub]
    grouped_barh(ax, groups, ["fine-tuned, standard tracks (B_all)", "+ split-S frames (C_acro)"], v,
                 [[f"{x:.1f}%" for x in row] for row in v], height=0.6, fills=["0.75", "k"])
    ax.set_title("Unreal split-S frames", fontsize=12, loc="left")
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=1)
    fig.suptitle("Detector: next gate found when it is in view", x=0.01, ha="left", fontsize=13, y=0.975)
    note(fig, "Held-out Unreal frames (Project AirSim, Blocks scene), n = frames with the next gate in view. On the same "
              "poses rendered synthetically, E finds it in 99.4-100%.\nFine-tune: 90k Unreal frames, half of each batch "
              "synthetic; split-S: 38k more frames. Source: runs/percep/stage6/detector_*/detect.json.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def old_students(out):
    s = RUNS / "percep/stage5/eval/student"
    sets = [("B_r6_20s", "plain 5-frame CNN (B)\nround 6, 20 s", "B"), ("G_r6_30s", "green-gate CNN (G)\nround 6, 30 s", "G"),
            ("G_r9_30s", "G round 9, 30 s", "G"), ("G_r10_30s", "G round 10, 30 s", "G"), ("G_r11_30s", "G round 11, 30 s", "G")]
    # student's own round evaluations: runs/student/ssm/baselines/{B,G}/rounds.jsonl
    stud = {}
    for name in ["B", "G"]:
        for line in (RUNS / f"student/ssm/baselines/{name}/rounds.jsonl").read_text().splitlines():
            r = json.loads(line)
            stud[(name, r["round"])] = r["eval"]["frac_3_laps"]
    vals, labs, groups = [], [], []
    for key, lab, who in sets:
        p, t = jload(s / f"student_{key}.json"), jload(s / f"student_teacher_{key}.json")
        rnd = int(key.split("_")[1][1:])
        row = [p["frac_3_laps"], stud[(who, rnd)], t["frac_3_laps"]]
        vals.append([100 * x for x in row])
        labs.append([pct(x) for x in row])
        groups.append(f"{lab}\n(n = {p['flights']})")
    fig, ax = plt.subplots(figsize=(12, 7.4), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.14)
    style(ax)
    grouped_barh(ax, groups, ["this pipeline (no green gate)", "old camera student", "teacher, true state"], vals, labs)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    ax.set_title("Three laps on the old students' own evaluation flights", fontsize=13, loc="left", pad=26)
    note(fig, "Unseen random tracks, synthetic renderer, the students' evaluation flights replayed exactly (256 each). "
              "B ran 20 s, so even the teacher\nreaches only 87%. G saw the next gate painted green. The pipeline is "
              "given the course list; the students were not.\nSources: runs/percep/stage5/eval/student/, "
              "runs/student/ssm/baselines/{B,G}/rounds.jsonl.", y=0.01)
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def tilt_deg(q):
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1, 1)))


def lap(f, k=1):
    """Steps of the k-th full lap: from passing the last gate (target wraps to 0) to the next such pass."""
    t = f["target"].astype(int)
    starts = np.where((t[1:] == 0) & (t[:-1] == len(f["gates"]) - 1))[0] + 1
    return slice(starts[k - 1], starts[k])


def dives(f):
    """Max tilt of every completed dive (target 1 = gate 2, entered after passing gate 1)."""
    t, tl = f["target"].astype(int), tilt_deg(f["quat"])
    out, i = [], 0
    while i < len(t):
        if t[i] == 1 and (i == 0 or t[i - 1] == 0):
            j = i
            while j < len(t) and t[j] == 1:
                j += 1
            if j < len(t) and t[j] == 2:
                out.append(tl[i:j].max())
            i = j
        else:
            i += 1
    return np.array(out)


def split_s(out):
    a = RUNS / "teacher/acro"
    flights = [("teacher, speed reward only", load(a / "ss_cam03/eval_acro/record_ep0.npz"), "0.55", "-"),
               ("teacher + inversion reward", load(a / "ss_inv20/eval_acro/record_ep0.npz"), "k", "-"),
               ("pipeline in Unreal", load(RUNS / "percep/stage6/videos/split_s/flight.npz"), "k", "--")]
    ev = {k: jload(a / f"{k}/eval_acro/eval_acro.json")["dive"] for k in ["ss_cam03", "ss_inv20"]}
    pol = jload(RUNS / "percep/policy/train/est_acro_v5/eval1/acro_est_x1.json")["dive"]
    unreal_dives = dives(flights[2][1])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), dpi=DPI, facecolor="w", gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.55)
    ax = axs[0]
    for name, f, c, ls in flights:
        s = lap(f, 1)
        tl = tilt_deg(f["quat"][s])
        ax.plot(np.arange(len(tl)) / f["hz"], tl, color=c, ls=ls, lw=1.5, label=name)
    ax.axhline(90, color="0.6", lw=0.7)
    ax.text(4.85, 92, "on its side", fontsize=9, color="0.4", va="bottom", ha="right")
    ax.axhline(150, color="0.6", lw=0.7, ls=":")
    ax.text(4.85, 152, "150 deg", fontsize=9, color="0.4", va="bottom", ha="right")
    ax.set(xlim=(0, 4.9), ylim=(0, 180))
    ax.set_yticks([0, 45, 90, 135, 180], ["0", "45", "90", "135", "180 deg"], fontsize=10)
    ax.set_xlabel("time since the lap started, s", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Tilt over the second lap of one flight", fontsize=12, loc="left")
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=3)
    ax = axs[1]
    style(ax)
    rows = [("teacher, speed\nreward only", ev["ss_cam03"]["frac_tilt_gt_150"], f"{ev['ss_cam03']['n_completed']:,} dives", "0.75"),
            ("teacher + inversion\nreward", ev["ss_inv20"]["frac_tilt_gt_150"], f"{ev['ss_inv20']['n_completed']:,} dives", "0.75"),
            ("pipeline, synthetic,\nnoise x1", pol["frac_tilt_gt_150"], f"{pol['n_completed']:,} dives", "k"),
            ("pipeline, Unreal,\nvideo flight", (unreal_dives > 150).mean(), f"{len(unreal_dives)} dives", "k")]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [100 * r[1] for r in rows], height=0.55, color=[r[3] for r in rows], edgecolor="k", lw=0.8)
    for yy, r in zip(y, rows):
        ax.text(100 * r[1] + 2, yy, f"{pct(r[1])}\nof {r[2]}", va="center", fontsize=10.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.5)
    ax.set_title("Dives past 150 deg (upside down)", fontsize=12, loc="left")
    fig.suptitle("Split-S: the flip only appears with a reward for it", x=0.01, ha="left", fontsize=13, y=0.975)
    note(fig, "Dive = from the high gate to the low gate. Synthetic bars: 256 randomised 30 s flights each. Unreal: the "
              "video flight (seed 900002, slot 0), not picked; its 24-flight evaluation: 23 of 24 three laps.\nSources: "
              "runs/teacher/acro/{ss_cam03,ss_inv20}/eval_acro/, runs/percep/policy/train/est_acro_v5/eval1/acro_est_x1.json, "
              "runs/percep/stage6/videos/split_s/flight.npz.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)
    return len(unreal_dives), int((unreal_dives > 150).sum())


def memory(out):
    mem = RUNS / "student/mem/eval"
    arms = [("s2_plain", "plain 5-frame CNN"), ("s2_mem", "+ memory layer"), ("s2_memgate", "+ memory + gate head")]
    m = [jload(mem / f"{a}/ckpt_r08.json")["layouts"]["figure8"]["randomised"] for a, _ in arms]
    ssm = RUNS / "student/ssm"

    def r6(path):
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r["round"] == 6:
                return r["eval"]
    s = [("plain 5-frame CNN (B)", r6(ssm / "baselines/B/rounds.jsonl")), ("SSM in place of\nthe frame stack", r6(ssm / "S/rounds.jsonl")),
         ("SSM, second seed", r6(ssm / "S_seed2/rounds.jsonl"))]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.8, bottom=0.25, wspace=0.55)
    for ax, rows, title in [(axs[0], [(lab, d) for (_, lab), d in zip(arms, m)], "Memory layer on top of the CNN\nfigure-8, 30 s, after round 8"),
                            (axs[1], s, "State-space memory instead of 5 frames\nrandom tracks, 20 s, after round 6")]:
        style(ax)
        y = np.arange(len(rows))[::-1]
        v = [100 * d["frac_3_laps"] for _, d in rows]
        ax.barh(y, v, height=0.55, color=["0.75"] + ["w"] * (len(rows) - 1), edgecolor="k", lw=0.8)
        for yy, x, (_, d) in zip(y, v, rows):
            ax.text(x + 2, yy, f"{x:.1f}% (n = {d['episodes']})", va="center", fontsize=10.5)
        ax.set_yticks(y, [lab for lab, _ in rows], fontsize=10.5)
        ax.set_title(title, fontsize=12, loc="left")
    fig.suptitle("Old camera students: memory did not help (three laps)", x=0.01, ha="left", fontsize=13, y=0.985)
    note(fig, "Left: same data and rounds for all three arms; differences are within noise (about 2.6 points standard "
              "error at n = 256). Right: same data budget per round as B.\nSources: runs/student/mem/eval/s2_*/ckpt_r08.json, "
              "runs/student/ssm/{S,S_seed2,baselines/B}/rounds.jsonl.")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def course_layouts():
    """Nominal 6-gate lists: fpv.track.SCENARIOS plus fpv.acro.LAYOUTS['split_s'] (read with ast: fpv.acro imports
    jax, which this script does not need)."""
    import ast

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fpv import track

    tree = ast.parse((Path(__file__).parent.parent / "fpv/acro.py").read_text())
    acro = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LAYOUTS")
    return {k: track.resolve(k) for k in TRACKS} | {"split_s": acro["split_s"]}, track.GATE_OUTER


def tracks(out):
    layouts, outer = course_layouts()
    shift = 0.3  # fpv.env.EnvConfig.rand_gate_pos (+/- m per axis); rand_gate_yaw 0.2 rad
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.6), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.12, wspace=0.12, hspace=0.28)
    for ax, (name, gates) in zip(axs.flat, layouts.items()):
        pos = np.array([g["pos"] for g in gates], float)
        yaw = np.array([g["yaw"] for g in gates], float)
        loop = np.vstack([pos, pos[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="0.75", lw=0.8, ls=":", zorder=1)
        for i, (p, a) in enumerate(zip(pos, yaw)):
            d, left = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            ax.add_patch(plt.Rectangle((p[0] - shift, p[1] - shift), 2 * shift, 2 * shift, fc="none", ec="0.6",
                                       lw=0.6, ls="--", zorder=1))
            e = np.array([p[:2] - left * outer / 2, p[:2] + left * outer / 2])
            ax.plot(e[:, 0], e[:, 1], color="k", lw=2.4, solid_capstyle="butt", zorder=3)
            ax.annotate("", xy=p[:2] + d * 0.75, xytext=p[:2], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9, mutation_scale=9))
            # label behind the gate; split-S G1 sits 3.5 m above G2 / G3, so its label goes to the side
            lab = p[:2] + np.array([0.0, 1.0]) if (name, i) == ("split_s", 0) else p[:2] - d * 0.95
            ax.text(lab[0], lab[1], f"G{i + 1}\n{p[2]:.1f} m", fontsize=10, ha="center", va="center", zorder=4,
                    linespacing=1.1, bbox=dict(fc="w", ec="none", pad=0.2))
        ax.set_aspect("equal")
        lo, hi = pos[:, :2].min(0) - 1.5, pos[:, :2].max(0) + 1.5
        c, r = (lo + hi) / 2, max(hi - lo) / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.6")
            s.set_linewidth(0.6)
        ax.set_title(NAMES[name], fontsize=12, loc="left")
    ax = axs.flat[-1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.05, 0.05], [0.78, 0.92], color="k", lw=2.4, solid_capstyle="butt")
    ax.annotate("", xy=(0.17, 0.85), xytext=(0.05, 0.85), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))
    ax.text(0.22, 0.85, "gate (0.72 m frame, 0.4 m opening),\narrow = direction of travel", fontsize=9.5,
            va="center")
    ax.add_patch(plt.Rectangle((0.03, 0.55), 0.12, 0.12, fc="none", ec="0.6", lw=0.6, ls="--"))
    ax.text(0.22, 0.61, "where this flight's gate centre can be:\n+/- 0.3 m per axis (also in height)", fontsize=9.5,
            va="center")
    ax.text(0.03, 0.38, "G1 ... G6: the order on the course list,\nwith the nominal height", fontsize=9.5, va="center")
    ax.text(0.03, 0.18, "Grey dotted line: gate order only,\nnot the flown path", fontsize=9.5, va="center")
    fig.suptitle("The course list: nominal gate centres and headings, top-down", x=0.01, ha="left", fontsize=13,
                 y=0.975)
    note(fig, "This list is what the estimator is given before the flight. Each flight then shifts every gate by up to "
              "0.3 m per axis and turns it by up to 0.2 rad; the estimator is not told the shifts,\nso it must find the "
              "gates in the frames. Random training and test tracks are generated per flight. Source: fpv/track.py "
              "(SCENARIOS), fpv/acro.py (split_s), fpv/env.py (rand_gate_pos, rand_gate_yaw).", y=0.01)
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def unreal_by_track(synth, out):
    """Unreal closed loop, runs/percep/stage6/eval/<track>/<track>.json (24 flights each), with the same flights on
    the synthetic renderer next to it. synth: {track: three-lap count on the same flights, synthetic renderer}."""
    order = [*TRACKS, "split_s"]
    ev = {k: jload(p) for k in order if (p := RUNS / f"percep/stage6/eval/{k}/{k}.json").exists()}
    keys = [k for k in order if k in ev]
    vals, labs, groups = [], [], []
    for k in keys:
        e = ev[k]
        n = e["episodes"]
        u = round(e["frac_3_laps"] * n)
        crashes = round(e["crash_rate"] * n)
        lap_s = e["lap_time_s"]["flying_mean"]
        s = synth.get(k)
        vals.append([100 * u / n, None if s is None else 100 * s / n])
        lap_txt = f", lap {lap_s:.2f} s" if lap_s else ""
        labs.append([f"{u} of {n}  ({crashes} crashed{lap_txt})", "" if s is None else f"{s} of {n}"])
        groups.append(f"{NAMES[k]}\n(n = {n})")
    fig, ax = plt.subplots(figsize=(12, 8.2), dpi=DPI, facecolor="w")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.14)
    style(ax, xmax=140)
    grouped_barh(ax, groups, ["Unreal", "same flights, synthetic renderer"], vals, labs, fs=10.5)
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.set_title("Three laps in 30 s, by track, flown from Unreal frames", fontsize=13, loc="left", pad=26)
    note(fig, "Each track: the same 24 randomised flights (evaluation flights 0-23), first gate not given, no green gate. "
              "Unreal: detector fine-tuned on Unreal frames (B_all), policy est_rand_v5;\nsplit-S: its own policy "
              "(est_acro_v5) and detector (C_acro). Synthetic: same policy, detector E. One Unreal scene (Blocks).\n"
              "Sources: runs/percep/stage6/eval/<track>/<track>.json, runs/percep/stage6/UNREAL_ALL.md.", y=0.01)
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perception-pipeline/figures")
    ap.add_argument("--unreal-synth", default=None,
                    help="'slalom=20,circuit=24,...': three-lap counts of the same 24 flights on the synthetic renderer")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "estimator_dataflow": estimator_dataflow, "training": training_diagram, "synthetic_by_track": synthetic_by_track,
        "noise_robustness": noise_robustness, "estimator_versions": estimator_versions, "crash_causes": crash_causes,
        "detector_pilots": detector_pilots, "detector_unreal": detector_unreal, "old_students": old_students,
        "split_s": split_s, "memory_experiments": memory, "tracks": tracks,
    }
    for name, fn in figs.items():
        if args.only and name not in args.only.split(","):
            continue
        r = fn(out / f"{name}.png")
        print("wrote", name, r if r is not None else "")
    if args.unreal_synth is not None:
        synth = {k: int(v) for k, v in (kv.split("=") for kv in args.unreal_synth.split(",") if kv)}
        unreal_by_track(synth, out / "unreal_by_track.png")
        print("wrote unreal_by_track")


if __name__ == "__main__":
    main()
