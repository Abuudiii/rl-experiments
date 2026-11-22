"""Split-S feasibility in the actual sim (Crazyflow cf21B_500 through `fpv.env.RaceEnv` physics).

    HIP_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/acro_feasibility.py

A scripted pilot flies a classic split-S from level flight at speed v0 heading +x, high up (no bounds),
nominal physics, no command delay, rate commands clipped to the env limits (10/10/5 rad/s):
  1. half-roll: roll rate +10 rad/s (proportional near the end) at ``roll_thrust`` * full thrust,
     until the body is inverted (tilt > 165 deg);
  2. pull-through: pitch rate ``pull_rate`` nose toward the body z axis (i.e. down, then back
     under), wings kept level by a small roll/yaw correction, until upright again
     (body z . world z > 0.8); thrust: full while the thrust axis
     points above the horizon, ``inv_thrust`` * full while it points below (inverted part);
  3. level-out: hold upright at full thrust until the climb starts (vz >= 0).
Done when vx < 0 and vz >= 0 after phase 2. Sweep: v0 x roll_thrust x pull_rate x inv_thrust. Height loss =
entry height - lowest point; x extent = furthest point past the entry; pull radius = half the
vertical span of phase 2. Output: JSON + npz (trajectories) next to ``--out``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import env as fenv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/teacher/acro/feasibility.json"))
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    from crazyflow.drones import Drone
    from crazyflow.dynamics import Dynamics, load_params

    p = load_params(Dynamics.first_principles, Drone.cf21B_500)
    mass, tmax = float(p["mass"]), 4 * float(p["thrust_max"])
    drag = np.diag(np.asarray(p["drag_matrix"])).tolist()

    v0s, roll_thr, pull, inv_thr = [2.0, 3.0, 4.0, 5.0], [0.15, 0.5], [6.0, 8.0, 10.0], [0.15, 0.5, 1.0]
    combos = list(itertools.product(v0s, roll_thr, pull, inv_thr))
    n = len(combos)
    cfg = fenv.EnvConfig(n_envs=n).nominal()
    env = fenv.RaceEnv(cfg)
    state, _ = env.reset(jax.random.key(0))
    v0 = jnp.asarray([c[0] for c in combos])
    rt = jnp.asarray([c[1] for c in combos])
    wp = jnp.asarray([c[2] for c in combos])
    it = jnp.asarray([c[3] for c in combos])
    z0 = 20.0
    s, lr, allm = state.data.states, env._leaf_replace, jnp.ones(n, bool)
    states = lr(s, allm, pos=jnp.zeros((n, 1, 3)).at[..., 2].set(z0), vel=jnp.zeros((n, 1, 3)).at[:, 0, 0].set(v0),
                quat=jnp.zeros((n, 1, 4)).at[..., 3].set(1.0), ang_vel=jnp.zeros((n, 1, 3)))  # fmt: skip
    br = lr(state.data.controls.body_rate, allm, prev_ang_vel=jnp.zeros((n, 1, 3)))
    data = state.data.replace(states=states, controls=state.data.controls.replace(body_rate=br))
    lim = jnp.asarray([fenv.MAX_RATE_XY, fenv.MAX_RATE_XY, fenv.MAX_RATE_Z])

    @jax.jit
    def step(data, phase):
        st = data.states
        q, v = st.quat[:, 0], st.vel[:, 0]
        R = fenv.quat_to_mat(q)
        tilt = jnp.arccos(jnp.clip(R[:, 2, 2], -1, 1))
        # phase transitions (evaluated on the current state)
        phase = jnp.where((phase == 0) & (R[:, 2, 2] < -0.966), 1, phase)
        phase = jnp.where((phase == 1) & (R[:, 2, 2] > 0.8), 2, phase)
        phase = jnp.where((phase == 2) & (v[:, 2] >= 0) & (v[:, 0] < 0), 3, phase)
        w_roll = jnp.stack([jnp.clip(15 * (jnp.pi - tilt), 2.0, 10.0), jnp.zeros(n), jnp.zeros(n)], -1)
        level = 8 * R[:, 2, 1]
        w_pull = jnp.stack([-level * R[:, 2, 2], -wp, level * R[:, 2, 0]], -1)
        e_w = jnp.cross(R[:, :, 2], jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (n, 3)))
        w_up = jnp.einsum("nji,nj->ni", R, e_w) * 10
        w = jnp.where((phase == 0)[:, None], w_roll, jnp.where((phase == 1)[:, None], w_pull, w_up))
        w = jnp.clip(w, -lim, lim)
        thrust_pull = jnp.where(R[:, 2, 2] > 0, tmax, it * tmax)
        thrust = jnp.where(phase == 0, rt * tmax, jnp.where(phase == 3, mass * fenv.G,
                                                            jnp.where(phase == 1, thrust_pull, tmax)))
        cmd = jnp.concatenate([thrust[:, None], w], -1)
        data = env._physics(data, cmd, cmd, jnp.zeros(n, jnp.int32))
        return data, cmd, phase

    T = round(args.seconds * cfg.policy_hz)
    phase = jnp.zeros(n, jnp.int32)
    rec = {k: [] for k in ("pos", "vel", "quat", "ang_vel", "cmd", "phase")}
    for _ in range(T):
        st = data.states
        rec["pos"].append(st.pos[:, 0]), rec["vel"].append(st.vel[:, 0])
        rec["quat"].append(st.quat[:, 0]), rec["ang_vel"].append(st.ang_vel[:, 0])
        data, cmd, phase = step(data, phase)
        rec["cmd"].append(cmd), rec["phase"].append(phase)
    rec = {k: np.stack(jax.device_get(v), 1) for k, v in rec.items()}  # (n, T, ...)

    rows = []
    for i, (v, r_, w_, i_) in enumerate(combos):
        ph = rec["phase"][i]
        done = np.flatnonzero(ph == 3)
        kd = int(done[0]) if len(done) else T
        P, V, W = rec["pos"][i, :kd], rec["vel"][i, :kd], rec["ang_vel"][i, :kd]
        R = np.asarray(fenv.quat_to_mat(jnp.asarray(rec["quat"][i, :kd])))
        k1 = np.flatnonzero(ph == 1)
        k2 = np.flatnonzero(ph >= 2)
        pull_span = float(np.ptp(P[k1[0] : (k2[0] if len(k2) else kd), 2])) if len(k1) else None
        rows.append(dict(
            v0=v, roll_thrust=r_, pull_rate=w_, inv_thrust=i_, completed=bool(len(done)), time_s=kd / cfg.policy_hz,
            roll_time_s=float(k1[0] / cfg.policy_hz) if len(k1) else None,
            height_loss_m=float(z0 - P[:, 2].min()), height_loss_at_roll_end_m=float(z0 - P[k1[0], 2]) if len(k1) else None,
            pull_radius_m=pull_span / 2 if pull_span is not None else None,
            x_extent_m=float(P[:, 0].max()), x_exit_m=float(P[-1, 0]), y_drift_m=float(np.abs(P[:, 1]).max()),
            exit_speed=float(np.linalg.norm(V[-1])), max_speed=float(np.linalg.norm(V, axis=-1).max()),
            min_bodyz_dot_worldz=float(R[:, 2, 2].min()), peak_rate_xyz=np.abs(W).max(0).round(2).tolist(),
        ))  # fmt: skip
    best = {}
    for v in v0s:
        ok = [r for r in rows if r["v0"] == v and r["completed"]]
        best[str(v)] = min(ok, key=lambda r: r["height_loss_m"]) if ok else None
    out = dict(
        quad=dict(mass_kg=mass, thrust_max_N=tmax, thrust_to_weight=tmax / (mass * fenv.G),
                  drag_N_per_m_s_body_xyz=drag, drag_decel_at_4m_s=abs(drag[0]) * 4 / mass,
                  inverted_down_accel_g=(tmax + mass * fenv.G) / (mass * fenv.G),
                  max_level_turn_accel_m_s2=float(np.sqrt(tmax**2 - (mass * fenv.G) ** 2) / mass)),
        method=__doc__.strip(), best_per_v0=best, all=rows,
    )  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(args.out).with_suffix(".npz"), combos=np.asarray(combos), **rec)
    print(json.dumps(out["quad"], indent=1))
    keys = ("roll_thrust", "pull_rate", "inv_thrust", "time_s", "roll_time_s", "height_loss_at_roll_end_m", "height_loss_m",
            "pull_radius_m", "x_extent_m", "y_drift_m", "exit_speed", "max_speed", "min_bodyz_dot_worldz", "peak_rate_xyz")
    for v, r in best.items():
        print("BEST", v, r and {k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in keys})
    for r in rows:
        print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items() if k in keys or k in ("v0", "completed")})


if __name__ == "__main__":
    main()
