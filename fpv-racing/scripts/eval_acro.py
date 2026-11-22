"""Evaluate a teacher on an acrobatic layout (`fpv.acro`) and measure whether it flips.

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/eval_acro.py \
        runs/teacher/acro/<run>/ckpt_latest.pkl [--episodes 256] [--seed 7100] [--track split_s]

Runs the mean action for one 30 s episode per environment with the checkpoint's own randomisation
(physics, gate shifts, command delay; standard spawn, no action noise) on ``--track`` (default: the
checkpoint's ``env.track``). Writes to ``<ckpt dir>/eval_acro/`` (or ``--out``):

  eval_acro.json  the standard block of `eval_teacher.metrics` (three-lap rate, crashes, lap time, ...)
                  plus "dive": one entry per completed dive = the steps from passing G1 (high) to
                  passing G2 (low) that follow a G1 pass (dives from a spawn before G2 do not count):
                  fraction with tilt > 90 / > 150 deg (tilt = angle between body z and world z,
                  min of body-z . world-z per dive), its distribution, accumulated roll / pitch angle
                  (split-S: about +-180 deg each), body x at the most inverted moment (+1 = nose
                  still forward = half-roll first, -1 = pitched over backwards), peak body rates,
                  heights, duration; "laps": min body-z . world-z per lap (G1 pass to G1 pass);
                  "low_gates": speed through G2 and G3; "camera": the next gate (G2) in the FPV field of
                  view during the dive (fraction of steps, longest out-of-view stretch per dive).
  record_ep0.npz  the first episode of the batch (fixed seed, not picked) as a flight record
                  (`eval_teacher.write_record` format) for `scripts/acro_video.py`.
  flights.npz     pos / quat / vel / ang_vel / passed / target / alive / in_fov of the first 32 episodes.
  lap_profile.png height and tilt over the first complete lap of episode 0; dives_side.png every
                  completed dive of the first 64 episodes in the vertical x-z plane, coloured by tilt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from fpv import acro  # noqa: E402

acro.register()

import eval_teacher as et  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv import ppo  # noqa: E402
from fpv.env import RaceEnv  # noqa: E402


def body_axes(quat):
    """Body x and z axes in the world frame, (..., 3) each, from xyzw quaternions."""
    x, y, z, w = (quat[..., i] for i in range(4))
    bx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    bz = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], -1)
    return bx, bz


def runs_of(mask):
    """Start / end (exclusive) indices of the True runs of a 1-D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def dist(x):
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=len(x), mean=round(float(x.mean()), 4), p5=round(float(q[0]), 4), p25=round(float(q[1]), 4),
                median=round(float(q[2]), 4), p75=round(float(q[3]), 4), p95=round(float(q[4]), 4),
                min=round(float(x.min()), 4), max=round(float(x.max()), 4))  # fmt: skip


def acro_metrics(rec, dt, hi=0, lo=1):
    """Dive / lap / camera measures (see the module docstring). ``hi`` / ``lo``: gate indices."""
    alive, passed, target = rec["alive"], rec["passed"], rec["target"]
    T, N = alive.shape
    bx, bz = body_axes(rec["quat"])
    cz = bz[..., 2]  # body z . world z
    speed = np.linalg.norm(rec["vel"], axis=-1)
    dives, dives_crashed = [], 0
    for e in range(N):
        seg = alive[:, e] & (target[:, e] == lo)
        for a, b in zip(*runs_of(seg)):
            if a == 0 or not (passed[a - 1, e] and target[a - 1, e] == hi):
                continue  # not entered through the high gate
            if not passed[b - 1, e]:
                dives_crashed += int(rec["ended"][b - 1, e] and rec["outcome"][b - 1, e] > 0)
                continue
            k = np.arange(a, b)
            i_min = k[np.argmin(cz[k, e])]
            w = rec["ang_vel"][k, e]
            fov = rec["in_fov"][k - 1, e]  # in_fov of step j is after the action: the view of step j + 1
            out_a, out_b = runs_of(~fov)
            dives.append(dict(
                ep=e, start=int(a), end=int(b), min_cz=float(cz[k, e].min()), tilt_max_deg=float(np.degrees(np.arccos(np.clip(cz[k, e].min(), -1, 1)))),
                nose_x_at_most_inverted=float(bx[i_min, e, 0]), roll_deg=float(np.degrees(w[:, 0].sum() * dt)),
                pitch_deg=float(np.degrees(w[:, 1].sum() * dt)), yaw_deg=float(np.degrees(w[:, 2].sum() * dt)),
                peak_rate=np.abs(w).max(0).tolist(), z_entry=float(rec["pos"][a, e, 2]), z_min=float(rec["pos"][k, e, 2].min()),
                x_max=float(rec["pos"][k, e, 0].max()), duration_s=float((b - a) * dt), speed_low_gate=float(speed[b, e] if b < T else speed[b - 1, e]),
                max_speed=float(speed[k, e].max()), fov_frac=float(fov.mean()),
                longest_out_s=float((out_b - out_a).max() * dt) if len(out_a) else 0.0,
            ))  # fmt: skip
    # laps: from a pass of the high gate to the next one
    lap_min_cz = []
    for e in range(N):
        kk = np.flatnonzero(passed[:, e] & (target[:, e] == hi))
        for a, b in zip(kk[:-1], kk[1:]):
            lap_min_cz.append(float(cz[a + 1 : b + 1, e].min()))
    lap_min_cz = np.asarray(lap_min_cz)
    # low gates: speed right after the passing step
    low = {}
    for g in (lo, lo + 1):
        kk, ee = np.nonzero(passed & (target == g))
        low[f"G{g + 1}"] = dist(speed[np.minimum(kk + 1, T - 1), ee])
    mc = np.asarray([d["min_cz"] for d in dives])
    D = lambda k: dist([d[k] for d in dives])  # noqa: E731
    return dict(
        dive=dict(
            definition="steps from passing G%d to passing G%d, entered through G%d; tilt = angle between body z and world z" % (hi + 1, lo + 1, hi + 1),
            n_completed=len(dives), n_crashed_in_dive=int(dives_crashed),
            frac_tilt_gt_90=float((mc < 0).mean()) if len(mc) else None,
            frac_tilt_gt_150=float((mc < np.cos(np.radians(150))).mean()) if len(mc) else None,
            frac_tilt_gt_120=float((mc < np.cos(np.radians(120))).mean()) if len(mc) else None,
            min_bodyz_dot_worldz=D("min_cz"), tilt_max_deg=D("tilt_max_deg"),
            nose_x_at_most_inverted=D("nose_x_at_most_inverted"), roll_deg=D("roll_deg"), pitch_deg=D("pitch_deg"),
            yaw_deg=D("yaw_deg"),
            peak_rate_abs_xyz_max=np.max([d["peak_rate"] for d in dives], 0).round(2).tolist() if dives else None,
            peak_rate_abs_xyz_median=np.median([d["peak_rate"] for d in dives], 0).round(2).tolist() if dives else None,
            z_entry=D("z_entry"), z_min=D("z_min"), x_max=D("x_max"), duration_s=D("duration_s"),
            max_speed=D("max_speed"),
        ),  # fmt: skip
        laps=dict(n=len(lap_min_cz), min_bodyz_dot_worldz=dist(lap_min_cz),
                  frac_tilt_gt_90=float((lap_min_cz < 0).mean()) if len(lap_min_cz) else None,
                  frac_tilt_gt_150=float((lap_min_cz < np.cos(np.radians(150))).mean()) if len(lap_min_cz) else None),
        low_gates_speed_m_s=low,
        camera=dict(
            definition="next gate (G%d) inside the FPV camera (pitched up 20 deg, 110 deg horizontal FOV) during the dive" % (lo + 1),
            fov_frac_per_dive=D("fov_frac"), fov_frac_all_dive_steps=float(
                np.sum([d["fov_frac"] * (d["end"] - d["start"]) for d in dives]) / max(sum(d["end"] - d["start"] for d in dives), 1)),
            longest_out_of_view_s=D("longest_out_s"),
            frac_dives_with_gate_out_of_view=float(np.mean([d["longest_out_s"] > 0 for d in dives])) if dives else None,
        ),  # fmt: skip
        _dives=dives,
    )


def plots(rec, dives, gates_pos, out: Path, dt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bx, bz = body_axes(rec["quat"])
    tilt = np.degrees(np.arccos(np.clip(bz[..., 2], -1, 1)))
    # first complete lap of episode 0 (G1 pass to G1 pass), else its whole flight
    e = 0
    kk = np.flatnonzero(rec["passed"][:, e] & (rec["target"][:, e] == 0))
    a, b = (kk[0] + 1, kk[1] + 1) if len(kk) >= 2 else (0, int(rec["alive"][:, e].sum()))
    t = np.arange(a, b) * dt
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, rec["pos"][a:b, e, 2], "k")
    ax[0].set_ylabel("height z [m]")
    ax[1].plot(t, tilt[a:b, e], "C3")
    for y in (90, 150):
        ax[1].axhline(y, color="0.6", ls="--", lw=0.8)
    ax[1].set_ylabel("tilt [deg]\n(body z vs up)")
    ax[1].set_ylim(0, 185)
    ax[2].plot(t, np.linalg.norm(rec["vel"][a:b, e], axis=-1), "C0", label="speed [m/s]")
    w = rec["ang_vel"][a:b, e]
    for i, n in enumerate(("roll", "pitch", "yaw")):
        ax[2].plot(t, w[:, i], lw=0.8, label=f"{n} rate [rad/s]")
    ax[2].legend(ncol=4, fontsize=8)
    ax[2].set_xlabel("time in episode 0 [s]")
    for k in np.flatnonzero(rec["passed"][a:b, e]) + a:
        for x in ax:
            x.axvline(k * dt, color="C2", lw=0.8)
        ax[0].text(k * dt, ax[0].get_ylim()[1], f"G{rec['target'][k, e] + 1}", color="C2", fontsize=8, va="top")
    for d in dives:
        if d["ep"] == e and d["start"] >= a and d["end"] <= b:
            for x in ax:
                x.axvspan(d["start"] * dt, d["end"] * dt, color="C1", alpha=0.15)
    ax[0].set_title("episode 0, first full lap (G1 pass to G1 pass); orange = dive G1 -> G2; green = gate passes")
    fig.tight_layout()
    fig.savefig(out / "lap_profile.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for d in [d for d in dives if d["ep"] < 64]:
        k = np.arange(d["start"] - 5, min(d["end"] + 15, rec["pos"].shape[0]))
        p = rec["pos"][k, d["ep"]]
        ax.scatter(p[:, 0], p[:, 2], c=tilt[k, d["ep"]], cmap="coolwarm", vmin=0, vmax=180, s=3)
    sc = ax.scatter([], [], c=[], cmap="coolwarm", vmin=0, vmax=180)
    fig.colorbar(sc, label="tilt [deg] (180 = upside down)")
    for i, g in enumerate(gates_pos):
        ax.plot([g[0], g[0]], [g[2] - 0.2, g[2] + 0.2], "m", lw=3)
        ax.text(g[0] + 0.05, g[2] + 0.25, f"G{i + 1}", color="m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_title("side view: every completed dive G1 -> G2 of the first 64 episodes (nominal gates)")
    fig.tight_layout()
    fig.savefig(out / "dives_side.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7100)
    ap.add_argument("--track", default=None)
    ap.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise on every action")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt.parent / "eval_acro"
    out.mkdir(parents=True, exist_ok=True)
    net, params, norm, cfg, _, info = ppo.load_checkpoint(ckpt)
    cfg = dataclasses.replace(cfg, n_envs=args.episodes, p_wide=0.0, rand_action_noise=0.0)
    if args.track:
        cfg = dataclasses.replace(cfg, track=args.track, **acro.ACRO_ENV.get(args.track, {}))
    dt = 1 / cfg.policy_hz
    env = RaceEnv(cfg)
    run = et.make_rollout(env, net, params, norm, np.arange(1))
    rec = run(jax.random.key(args.seed), args.noise)
    m = et.metrics(rec, env.n_gates, dt)
    am = acro_metrics(rec, dt)
    dives = am.pop("_dives")
    res = dict(checkpoint=str(ckpt), iteration=info.get("iteration"), env_steps=info.get("env_steps"),
               track=cfg.track, seed=args.seed, noise=args.noise, z_max=cfg.z_max, w_cam=cfg.w_cam,
               acro_shaping=info.get("acro_shaping"), init=info.get("init"),
               randomisation={k: getattr(cfg, k) for k in ("rand_mass", "rand_thrust", "rand_rotor", "rand_drag",
                                                           "rand_gate_pos", "rand_gate_yaw", "delay_max_substeps")},
               standard=m, **am)  # fmt: skip
    (out / "eval_acro.json").write_text(json.dumps(res, indent=1))
    et.write_record(out / "record_ep0.npz", rec, 0, f"{cfg.track} ep0 seed {args.seed}", dt)
    n = min(32, args.episodes)
    np.savez_compressed(out / "flights.npz", **{k: rec[k][:, :n] for k in ("pos", "quat", "vel", "ang_vel", "passed", "target",
                                                                            "alive", "in_fov", "action")},
                        gates_pos=rec["gates_pos"][:n], gates_yaw=rec["gates_yaw"][:n])  # fmt: skip
    plots(rec, dives, np.asarray(env.gates_pos), out, dt)
    print(et.line(cfg.track, m))
    d, c = res["dive"], res["camera"]
    print(f"dives {d['n_completed']} (crashed in dive {d['n_crashed_in_dive']}): tilt>90 {d['frac_tilt_gt_90']}  "
          f">150 {d['frac_tilt_gt_150']}  min bz.z {d['min_bodyz_dot_worldz'] and d['min_bodyz_dot_worldz']['median']}  "
          f"roll {d['roll_deg'] and d['roll_deg']['median']}  pitch {d['pitch_deg'] and d['pitch_deg']['median']}  "
          f"nose_x {d['nose_x_at_most_inverted'] and d['nose_x_at_most_inverted']['median']}")
    print(f"laps {res['laps']['n']} tilt>150 {res['laps']['frac_tilt_gt_150']}  low gates {res['low_gates_speed_m_s']}")
    print(f"camera: fov in dive {c['fov_frac_all_dive_steps']:.3f}  longest out {c['longest_out_of_view_s']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
