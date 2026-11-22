"""Media for the perception-pipeline post: 3D chase animation of the Unreal flights side by side (the
style of `plot_teacher.movie`), and the architecture figure.

    uv run --no-project --with matplotlib --with numpy python scripts/post_figures.py \
        --flight "slalom=runs/percep/stage6/videos/slalom/flight.npz" \
        --flight "split-S|state-based teacher=runs/teacher/acro/ss_inv20/eval_acro/record_ep0.npz" --out figures/post
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

GATE_OPENING, GATE_OUTER = 0.4, 0.72  # fpv.track
ARM = 0.12  # exaggerated drone size so that it stays visible


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load(path):
    d = np.load(path, allow_pickle=True)
    pos, quat, hz = np.asarray(d["pos"], float).reshape(-1, 3), np.asarray(d["quat"], float).reshape(-1, 4), float(d["hz"])
    n = int(np.isfinite(pos).all(1).sum())  # records are padded after the flight ends
    pos, quat = pos[:n], quat[:n]
    vel = np.gradient(pos, axis=0) * hz
    # Chase camera: smoothed horizontal heading, held while the drone moves mostly vertically (the dive).
    horiz = np.linalg.norm(vel[:, :2], axis=1)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    for i in range(1, n):
        if horiz[i] < 1.0:
            heading[i] = heading[i - 1]
    heading = np.unwrap(heading)
    heading = np.convolve(np.pad(heading, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    return dict(pos=pos, quat=quat, hz=hz, n=n, speed=np.linalg.norm(vel, axis=1), heading=heading,
                inverted=(1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)) < 0,  # body z . world z < 0
                gates=np.asarray(d["gates_pos"], float).reshape(-1, 3), yaw=np.asarray(d["gates_yaw"], float).ravel(),
                passed=np.cumsum(np.asarray(d["passed"]).reshape(-1)[:n]), target=np.asarray(d["target"]).reshape(-1)[:n])


def gate_loop(p, yaw, size):
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0]) * size / 2
    up = np.array([0, 0, size / 2])
    return np.array([p - side - up, p + side - up, p + side + up, p - side + up, p - side - up])


class Panel:
    def __init__(self, fig, rect, name, who, f):
        self.f, self.name, self.who = f, name, who
        ax = self.ax = fig.add_axes(rect, projection="3d")
        ax.plot(*f["pos"].T, color="0.75", lw=0.6)
        for p, y in zip(f["gates"], f["yaw"]):
            ax.plot(*gate_loop(p, y, GATE_OPENING).T, color="tab:orange", lw=1.5)
            ax.plot(*gate_loop(p, y, GATE_OUTER).T, color="tab:red", lw=2.5)
            ax.plot([p[0]] * 2, [p[1]] * 2, [0, p[2] - GATE_OUTER / 2], color="0.5", lw=1)  # stand, for depth
        (self.trail,) = ax.plot([], [], [], color="tab:blue", lw=2)
        self.arms = [ax.plot([], [], [], color="k", lw=2.5)[0] for _ in range(2)]
        (self.nose,) = ax.plot([], [], [], color="tab:green", lw=2)
        (self.target,) = ax.plot([], [], [], "o", color="tab:red", ms=6)
        ax.text2D(0.02, 0.955, f"{name}: {who}", transform=ax.transAxes, fontsize=14, weight="bold")
        self.text = ax.text2D(0.02, 0.915, "", transform=ax.transAxes, fontsize=12, family="monospace")
        self.flag = ax.text2D(0.02, 0.87, "", transform=ax.transAxes, fontsize=13, weight="bold", color="tab:red")
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

    def draw(self, t):
        f = self.f
        i = min(round(t * f["hz"]), f["n"] - 1)
        p, rot = f["pos"][i], quat_to_mat(f["quat"][i])
        self.trail.set_data_3d(*f["pos"][max(0, i - 50): i + 1].T)
        for line, axis in zip(self.arms, (rot[:, 0] + rot[:, 1], rot[:, 0] - rot[:, 1])):
            line.set_data_3d(*np.stack([p - ARM * axis / np.sqrt(2), p + ARM * axis / np.sqrt(2)]).T)
            line.set_color("tab:red" if f["inverted"][i] else "k")
        self.nose.set_data_3d(*np.stack([p, p + 2 * ARM * rot[:, 0]]).T)
        g = f["gates"][int(f["target"][i]) % len(f["gates"])]
        self.target.set_data_3d([g[0]], [g[1]], [g[2]])
        r = 2.2
        z0 = max(p[2] - 0.6 * r, 0)
        self.ax.set(xlim=(p[0] - r, p[0] + r), ylim=(p[1] - r, p[1] + r), zlim=(z0, z0 + 1.2 * r))
        self.ax.view_init(elev=22, azim=np.degrees(f["heading"][i]) + 180)  # chase: behind the drone
        gates = int(f["passed"][i])
        self.text.set_text(f"t {i / f['hz']:5.2f} s   speed {f['speed'][i]:4.1f} m/s   gates {gates:3d}   lap {gates // len(f['gates']) + 1}")
        self.flag.set_text("UPSIDE DOWN" if f["inverted"][i] else "")


def chase_movie(flights, path, fps=25):
    """One chase panel per flight; more than two flights fill two columns (left column first)."""
    k = len(flights)
    cols = 1 if k == 1 else 2
    rows = -(-k // cols) if k > 2 else 1
    fig = plt.figure(figsize=(9.6 * cols, 5.4 * rows if rows > 1 else 7.2), dpi=100)
    panels = [
        Panel(fig, [(j // rows) / cols, 1 - (j % rows + 1) / rows, 1 / cols, 1 / rows], name, who, f)
        for j, (name, who, f) in enumerate(flights)
    ]
    seconds = max(f["n"] / f["hz"] for _, _, f in flights)
    anim = animation.FuncAnimation(fig, lambda fr: [p.draw(fr / fps) for p in panels], frames=int(seconds * fps), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2500 * k))
    plt.close(fig)


BLOCKS = [  # name, what it is, learned?, what it sends on
    ("Camera", "320 x 180 frame", False, "frame"),
    ("Gate detector", "CNN", True, "gate\ncorners"),
    ("Geometry", "written code", False, "distance,\ndirection"),
    ("State estimate", "written code", False, "speed, tilt,\nnext gates"),
    ("Flight policy", "PPO, small MLP", True, "sticks"),
    ("Drone", "quadrotor", False, None),
]
SIDE_INPUTS = {2: "course list\n(gate order, given before the flight)", 3: "IMU\n(gyro, accelerometer)"}


def architecture_figure(stats, out):
    """Black and white: the control loop as a block diagram, then bar charts.

    stats: list of (title, [(label, value %, shown text, ours?)], footnote).
    """
    fig = plt.figure(figsize=(16, 9.2), dpi=120, facecolor="w")
    ax = fig.add_axes([0.02, 0.52, 0.96, 0.46])
    ax.set(xlim=(0, 16), ylim=(0, 4.4))
    ax.axis("off")
    ax.text(0.15, 4.1, "Architecture overview", fontsize=15, weight="bold")
    ax.add_patch(plt.Rectangle((12.3, 4.02), 0.22, 0.22, fc="0.85", ec="k", lw=1))
    ax.text(12.62, 4.13, "learned", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((13.75, 4.02), 0.22, 0.22, fc="w", ec="k", lw=1))
    ax.text(14.07, 4.13, "not learned", va="center", fontsize=10)
    bw, gap, bh, yc = 1.7, 1.1, 0.95, 2.2
    centres = []
    for i, (name, what, learned, sends) in enumerate(BLOCKS):
        x = 0.15 + i * (bw + gap)
        centres.append(x + bw / 2)
        ax.add_patch(plt.Rectangle((x, yc - bh / 2), bw, bh, fc="0.85" if learned else "w", ec="k", lw=1.1))
        ax.text(x + bw / 2, yc + 0.17, name, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + bw / 2, yc - 0.2, what, ha="center", va="center", fontsize=9.5)
        if sends:
            ax.annotate("", (x + bw + gap, yc), (x + bw, yc), arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
            ax.text(x + bw + gap / 2, yc + 0.12, sends, ha="center", va="bottom", fontsize=9, style="italic", linespacing=1.15)
        if i in SIDE_INPUTS:
            ax.annotate("", (x + bw / 2, yc + bh / 2), (x + bw / 2, yc + 1.05), arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
            ax.text(x + bw / 2, yc + 1.1, SIDE_INPUTS[i], ha="center", va="bottom", fontsize=9, style="italic", linespacing=1.15)
    # The loop closes through the drone: it moves, the camera sees the next frame.
    yb = yc - bh / 2 - 0.65
    ax.plot([centres[-1], centres[-1], centres[0]], [yc - bh / 2, yb, yb], color="k", lw=1.1)
    ax.annotate("", (centres[0], yc - bh / 2), (centres[0], yb), arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
    ax.text((centres[0] + centres[-1]) / 2, yb - 0.1, "the drone moves and the camera sees the next frame, 50 times a second",
            ha="center", va="top", fontsize=9, style="italic")
    fig.text(0.029, 0.47, "Results", fontsize=15, weight="bold")
    for k, (title, items, note) in enumerate(stats):
        a = fig.add_axes([0.1 + k * 0.315, 0.13, 0.215, 0.26])
        labels, vals, shown, ours = zip(*items)
        y = np.arange(len(vals))[::-1]
        a.barh(y, vals, height=0.55, color=["k" if o else "0.75" for o in ours], edgecolor="k", lw=0.8)
        for yy, v, s in zip(y, vals, shown):
            a.text(v + 2, yy, s, va="center", fontsize=11)
        a.set_yticks(y, labels, fontsize=10)
        a.set(xlim=(0, 118), ylim=(-0.6, len(vals) - 0.4))
        a.set_xticks([0, 50, 100], ["0", "50", "100 %"], fontsize=9)
        a.set_title(title, fontsize=11.5, loc="left")
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.text(0, -0.24, note, transform=a.transAxes, va="top", fontsize=8.5, color="0.3")
    fig.savefig(out, facecolor="w")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", action="append", default=[], help="name[|who flew it]=path to flight npz")
    ap.add_argument("--out", default="figures/post")
    ap.add_argument("--acro-bar", default=None, help="'23 of 24=95.8': three-lap result of the acrobatic Unreal flights")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flights = []
    for spec in args.flight:
        key, path = spec.split("=", 1)
        name, _, who = key.partition("|")
        flights.append((name, who or "perception pipeline", load(path)))
    if flights:
        chase_movie(flights, out / "flights_3d.mp4")
    unreal = [("slalom", 83.3, "20 of 24", True)]
    if args.acro_bar:
        shown, val = args.acro_bar.split("=")
        unreal.append(("split-S", float(val), shown, True))
    stats = [
        ("Detector finds the next gate, Unreal frames",
         [("synthetic-trained", 74.2, "74%", False), ("fine-tuned", 99.87, "99.9%", True)],
         "Held-out Unreal frames, six tracks."),
        ("Three laps on unseen random tracks",
         [("camera CNN", 27.7, "28%", False), ("CNN + green gate", 70, "70%", False), ("this pipeline", 93.8, "94%", True)],
         "Synthetic renderer. CNNs: 256 flights. Pipeline: 2,048 flights,\ngiven the gate order before the flight."),
        ("Three laps in Unreal, no green gate", unreal, "24 randomised 30 s flights per track."),
    ]
    architecture_figure(stats, out / "architecture.png")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
