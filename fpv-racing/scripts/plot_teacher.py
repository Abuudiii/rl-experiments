"""Trajectory figure and chase-view animation from an exported ``rollouts.npz``.

    .venv-sim/bin/python scripts/plot_teacher.py runs/teacher/<run>/rollouts.npz

Writes next to the rollouts: ``trajectory.png`` (top-down + side views of nominal-circuit flights,
coloured by speed, gates drawn) and ``flight.mp4`` (one multi-lap nominal flight, chase 3D view).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from fpv import track  # noqa: E402


def quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def gate_loops(gates_pos, gates_yaw):
    """Closed corner loops (5, 3) of opening and frame for each gate."""
    loops = []
    for p, y in zip(gates_pos, gates_yaw):
        for size in (track.GATE_OPENING, track.GATE_OUTER):
            c = track.gate_corners(p, y, size)
            loops.append(np.concatenate([c, c[:1]]))
    return loops


def pick_episode(d) -> int:
    """Longest nominal episode (most gates passed)."""
    nominal = np.flatnonzero(d["nominal"])
    return int(nominal[np.argmax(d["passed"][nominal].sum(1) + 1e-3 * d["length"][nominal])])


def figure(d, ep: int, path: Path, n_overlay: int = 20):
    n = int(d["length"][ep])
    pos, speed = d["pos"][ep, :n], np.linalg.norm(d["vel"][ep, :n], axis=-1)
    gp, gy = d["gates_pos"][ep], d["gates_yaw"][ep]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), gridspec_kw={"width_ratios": [1.25, 1.25, 1.1]})
    views = (((0, 1), "top-down", "x [m]", "y [m]"), ((0, 2), "side (x-z)", "x [m]", "z [m]"),
             ((1, 2), "side (y-z)", "y [m]", "z [m]"))  # fmt: skip
    others = [e for e in np.flatnonzero(d["nominal"]) if e != ep][:n_overlay]
    norm = plt.Normalize(0, max(speed.max(), 1e-3))
    for ax, ((i, j), title, xl, yl) in zip(axes, views):
        for e in others:
            m = int(d["length"][e])
            ax.plot(d["pos"][e, :m, i], d["pos"][e, :m, j], color="0.8", lw=0.5, zorder=1)
        seg = np.stack([pos[:-1, (i, j)], pos[1:, (i, j)]], axis=1)
        lc = LineCollection(seg, cmap="turbo", norm=norm, lw=1.6, zorder=2)
        lc.set_array(speed[:-1])
        ax.add_collection(lc)
        for loop in gate_loops(gp, gy):
            ax.plot(loop[:, i], loop[:, j], "k-", lw=1.5, zorder=3)
        if (i, j) == (0, 1):
            for k, (p, y) in enumerate(zip(gp, gy)):
                ax.annotate("", p[:2] + 0.45 * np.array([np.cos(y), np.sin(y)]), p[:2],
                            arrowprops=dict(arrowstyle="->", color="k"))  # fmt: skip
                ax.text(p[0] + 0.25, p[1] + 0.25, f"G{k + 1}", fontsize=10)
        ax.plot(pos[0, i], pos[0, j], "go", ms=7, zorder=4, label="spawn")
        ax.set(title=title, xlabel=xl, ylabel=yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.autoscale_view()
    axes[0].legend(loc="upper left")
    fig.colorbar(lc, ax=axes, label="speed [m/s]", shrink=0.8, pad=0.01)
    gates = int(d["passed"][ep, :n].sum())
    fig.suptitle(
        f"Teacher on the nominal circuit: episode {ep}, {n / 50:.1f} s, {gates} gates ({gates / len(gp):.1f} laps), "
        f"speed mean {speed.mean():.1f} / max {speed.max():.1f} m/s; grey: {len(others)} other nominal episodes"
    )
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def movie(d, ep: int, path: Path, seconds: float, fps: int = 25):
    import imageio_ffmpeg

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    n = min(int(d["length"][ep]), round(seconds * 50))
    stride = 50 // fps
    pos, quat, vel = d["pos"][ep, :n], d["quat"][ep, :n], d["vel"][ep, :n]
    speed = np.linalg.norm(vel, axis=-1)
    gp, gy = d["gates_pos"][ep], d["gates_yaw"][ep]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="0.75", lw=0.6)
    for k, loop in enumerate(gate_loops(gp, gy)):
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="tab:red" if k % 2 else "tab:orange", lw=2.5 if k % 2 else 1.5)
    for p in gp:  # stands, for depth cues
        ax.plot([p[0]] * 2, [p[1]] * 2, [0, p[2] - track.GATE_OUTER / 2], color="0.5", lw=1)
    (trail,) = ax.plot([], [], [], color="tab:blue", lw=2)
    arms = [ax.plot([], [], [], color="k", lw=2.5)[0] for _ in range(2)]
    (nose,) = ax.plot([], [], [], color="tab:green", lw=2)
    (target,) = ax.plot([], [], [], "o", color="tab:red", ms=6)
    text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=13, family="monospace")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    arm = 0.12  # exaggerated drone size so that it stays visible
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    heading = np.convolve(np.pad(heading, 12, mode="edge"), np.ones(25) / 25, mode="valid")  # smooth camera

    def draw(f):
        i = f * stride
        p, rot = pos[i], quat_to_mat(quat[i])
        trail.set_data_3d(*pos[max(0, i - 50) : i + 1].T)
        for line, axis in zip(arms, (rot[:, 0] + rot[:, 1], rot[:, 0] - rot[:, 1])):
            seg = np.stack([p - arm * axis / np.sqrt(2), p + arm * axis / np.sqrt(2)])
            line.set_data_3d(*seg.T)
        nose.set_data_3d(*np.stack([p, p + 2 * arm * rot[:, 0]]).T)
        g = gp[int(d["target"][ep, i])]
        target.set_data_3d([g[0]], [g[1]], [g[2]])
        r = 2.2
        ax.set_xlim(p[0] - r, p[0] + r)
        ax.set_ylim(p[1] - r, p[1] + r)
        ax.set_zlim(max(p[2] - 0.6 * r, 0), max(p[2] - 0.6 * r, 0) + 1.2 * r)
        ax.view_init(elev=22, azim=np.degrees(heading[i]) + 180)  # chase: behind the drone
        gates = int(d["passed"][ep, : i + 1].sum())
        text.set_text(f"t {i / 50:5.2f} s   speed {speed[i]:4.1f} m/s   gates {gates:3d}   lap {gates // len(gp) + 1}")
        return trail, *arms, nose, target, text

    anim = animation.FuncAnimation(fig, draw, frames=n // stride, blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts")
    ap.add_argument("--episode", type=int, default=None, help="episode index (default: longest nominal)")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of the animation")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()
    path = Path(args.rollouts)
    d = np.load(path)
    d = {k: d[k] for k in d.files}
    ep = pick_episode(d) if args.episode is None else args.episode
    figure(d, ep, path.parent / "trajectory.png")
    print(f"wrote {path.parent / 'trajectory.png'} (episode {ep})")
    if not args.no_movie:
        movie(d, ep, path.parent / "flight.mp4", args.seconds)
        print(f"wrote {path.parent / 'flight.mp4'}")


if __name__ == "__main__":
    main()
