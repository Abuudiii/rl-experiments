"""Feasibility check: Crazyflow as the batched training simulator for FPV gate racing.

Run on the node from the sim venv:
    CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python checks/crazyflow_check.py

Prints versions, the shipped drone models, body-rate step-response metrics (batch of 64) and a
throughput table (one GPU). Each throughput point runs in its own subprocess so that GPU memory
and JIT compile time are measured from a clean process.

Command interface under test: action = [collective thrust N, roll rate, pitch rate, yaw rate]
(rad/s, body frame), set at 50 Hz, physics at Crazyflow's default 500 Hz. It maps onto Crazyflow's
native `Control.body_rate` mode with the attitude-levelling gains (kR, ki_m) zeroed, the rate loop
running at 500 Hz inside the jitted step.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # make GPU memory numbers real
os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

import crazyflow
import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.control import load_params as load_control_params
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params, supported_dynamics
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import append_fn
from crazyflow.utils import leaf_replace

DRONE = Drone.cf21B_500
SIM_HZ = 500  # Crazyflow default physics rate (Sim(freq=500))
POLICY_HZ = 50
G = 9.81
OUT = Path(__file__).parent / "out"


def randomize_fn(pose: bool):
    """Reset-pipeline stage: per-world mass, thrust curve, rotor lag, drag and (optionally) pose."""

    def randomize(data, default, mask=None):
        key, k = jax.random.split(data.core.rng_key)
        ks = jax.random.split(k, 7)
        n = data.core.n_worlds

        def scale(k, shape, amount):
            return jax.random.uniform(k, (n, 1, *shape), minval=1 - amount, maxval=1 + amount)

        d = default.params
        new = {"mass": d.mass * scale(ks[0], (1,), 0.1)}
        if hasattr(d, "rpm2thrust"):  # first_principles only
            new["rpm2thrust"] = d.rpm2thrust * scale(ks[1], (4, 3), 0.05)  # per motor
            new["rotor_dyn_coef"] = d.rotor_dyn_coef * scale(ks[2], (4, 4), 0.1)
            new["drag_matrix"] = d.drag_matrix * scale(ks[3], (3, 3), 0.3)
        data = data.replace(
            params=leaf_replace(data.params, mask, **new), core=data.core.replace(rng_key=key)
        )
        if pose:
            s = default.states
            pos = s.pos + jax.random.uniform(ks[4], (n, 1, 3), minval=-0.5, maxval=0.5)
            yaw = jax.random.uniform(ks[5], (n, 1), minval=-jnp.pi, maxval=jnp.pi)
            zero = jnp.zeros_like(yaw)
            quat = jnp.stack([zero, zero, jnp.sin(yaw / 2), jnp.cos(yaw / 2)], axis=-1)  # xyzw
            vel = jax.random.uniform(ks[6], (n, 1, 3), minval=-1.0, maxval=1.0)
            data = data.replace(states=leaf_replace(data.states, mask, pos=pos, quat=quat, vel=vel))
        return data

    return randomize


def make_sim(n: int, dynamics: str, randomize: bool, pose: bool) -> Sim:
    control = Control.body_rate if dynamics == Dynamics.first_principles else Control.attitude
    sim = Sim(
        n_worlds=n,
        n_drones=1,
        drone=DRONE,
        dynamics=dynamics,
        control=control,
        freq=SIM_HZ,
        attitude_freq=SIM_HZ,
        body_rate_freq=SIM_HZ,
        force_torque_freq=SIM_HZ,
        device="gpu",
        fused_mjx_model=True,
    )
    data = sim.data
    states = data.states.replace(pos=data.states.pos.at[..., 2].set(1.0))
    if control == Control.body_rate:
        # Pure rate tracking: drop the firmware's attitude-levelling terms
        br = data.controls.body_rate
        br = br.replace(params=br.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)})
        data = data.replace(controls=data.controls.replace(body_rate=br))
        # Start with the rotors already spinning at hover RPM
        p = load_params(dynamics, DRONE)
        rpm = motor_force2rotor_vel(np.full(4, p["mass"] * G / 4), np.asarray(p["rpm2thrust"]))
        states = states.replace(rotor_vel=jnp.broadcast_to(jnp.asarray(rpm), (n, 1, 4)))
    sim.data = data.replace(states=states)
    sim.build_default_data()
    if randomize:
        append_fn(sim.reset_pipeline, randomize_fn(pose), name="randomize")
        sim.build_reset_fn()
    sim.reset()  # also expands the randomized params to their per-world shapes
    return sim


def set_action(sim: Sim, data, action):
    """action (N, 4) = [thrust, x, y, z] -> Crazyflow's [x, y, z, thrust] staged command."""
    cmd = jnp.concatenate([action[:, 1:], action[:, :1]], axis=-1)[:, None, :]
    if sim.control == Control.body_rate:
        return F.body_rate_control(data, cmd)
    return F.attitude_control(data, cmd)


# region step response


def step_response(randomize: bool, hover_s: float = 0.5, step_s: float = 0.6, rate: float = 3.0):
    n, sub = 64, SIM_HZ // POLICY_HZ
    sim = make_sim(n, Dynamics.first_principles, randomize, pose=False)
    step_fn = sim._step

    @jax.jit
    def rollout(data, actions):  # actions (T, N, 4) at 50 Hz, traces at 500 Hz
        def policy_step(data, action):
            data = set_action(sim, data, action)

            def physics_step(data, _):
                data = step_fn(data, n_steps=1)
                return data, (data.states.ang_vel[:, 0], data.states.pos[:, 0, 2])

            return jax.lax.scan(physics_step, data, length=sub)

        _, (w, z) = jax.lax.scan(policy_step, data, actions)
        return w.reshape(-1, n, 3), z.reshape(-1, n)

    n_hover, n_step = round(hover_s * POLICY_HZ), round(step_s * POLICY_HZ)
    traces, metrics = {}, {}
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        sim.reset()
        hover = sim.data.params.mass.reshape(-1) * G * jnp.ones(n)  # per-drone m*g
        actions = np.zeros((n_hover + n_step, n, 4), dtype=np.float32)
        actions[..., 0] = np.asarray(hover)
        actions[n_hover:, :, 1 + axis] = rate
        w, z = (np.asarray(x) for x in rollout(sim.data, jnp.asarray(actions)))
        t = (np.arange(w.shape[0]) + 1) / SIM_HZ - hover_s  # 0 = step command
        i0 = n_hover * sub
        ws = w[i0:, :, axis]  # (T, N)
        t10 = (np.argmax(ws >= 0.1 * rate, axis=0) + 1) / SIM_HZ
        t90 = (np.argmax(ws >= 0.9 * rate, axis=0) + 1) / SIM_HZ
        reached = (ws >= 0.9 * rate).any(axis=0)
        others = np.delete(w[i0:], axis, axis=2)
        metrics[name] = {
            "rise_ms": np.where(reached, (t90 - t10) * 1e3, np.nan),
            "delay_ms": np.where(reached, t90 * 1e3, np.nan),  # command -> 90 %
            "overshoot_pct": np.maximum(ws.max(axis=0) / rate - 1, 0) * 100,
            "ss_err_rad_s": ws[-SIM_HZ // 10 :].mean(axis=0) - rate,  # last 100 ms
            "cross_axis_peak_rad_s": np.abs(others).max(axis=(0, 2)),
            "hover_dz_m": z[i0 - 1] - 1.0,  # altitude drift over the hover phase
            "hover_rate_peak_rad_s": np.abs(w[:i0]).max(axis=(0, 2)),
            "step_dz_m": z[-1] - z[i0 - 1],  # altitude change while rotating (thrust tilts)
        }
        traces[name] = (t, w[:, :, axis], z)
    return traces, metrics


def report_step_response():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for randomize, color, label in ((False, "C0", "nominal"), (True, "C1", "randomized")):
        traces, metrics = step_response(randomize)
        print(f"\nBody-rate step response, 64 drones, {label} parameters, 3 rad/s step at t=0")
        print("(mean [min, max] over the batch; rates sampled at 500 Hz)")
        keys = list(next(iter(metrics.values())))
        print(f"{'metric':<24}" + "".join(f"{a:>30}" for a in metrics))
        for k in keys:
            cells = []
            for m in metrics.values():
                v = m[k]
                if np.isnan(v).all():
                    cells.append("never reached 90 %")
                else:
                    cells.append(f"{np.nanmean(v):.3f} [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]")
            print(f"{k:<24}" + "".join(f"{c:>30}" for c in cells))
        for i, (name, (t, w, z)) in enumerate(traces.items()):
            axes[0, i].fill_between(t, w.min(axis=1), w.max(axis=1), color=color, alpha=0.3)
            axes[0, i].plot(t, w.mean(axis=1), color=color, label=label)
            axes[1, i].fill_between(t, z.min(axis=1), z.max(axis=1), color=color, alpha=0.3)
            axes[1, i].plot(t, z.mean(axis=1), color=color, label=label)
            axes[0, i].set_title(f"{name} rate step (cmd 3 rad/s @ 50 Hz, physics 500 Hz)")
    for i in range(3):
        axes[0, i].plot([-0.5, 0, 0, 0.6], [0, 0, 3, 3], "k--", lw=0.8, label="command")
        axes[0, i].set_ylabel("body rate [rad/s]")
        axes[1, i].set_ylabel("altitude z [m] (thrust = m g)")
        axes[1, i].set_xlabel("time since step [s]")
        axes[0, i].grid(alpha=0.3)
        axes[1, i].grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"Crazyflow {crazyflow.__version__} {DRONE} first_principles, native body_rate")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "rate_step_response.png", dpi=110)
    print(f"\nplot: {OUT / 'rate_step_response.png'}")


# region throughput


def bench(n: int, dynamics: str, n_policy_steps: int = 200, reset_p: float = 0.01):
    t0 = time.perf_counter()
    sim = make_sim(n, dynamics, randomize=True, pose=True)
    jax.block_until_ready(sim.data)
    construct_s = time.perf_counter() - t0
    step_fn, reset_fn, default = sim._step, sim._reset, sim.default_data
    sub = SIM_HZ // POLICY_HZ
    p = load_params(dynamics, DRONE)
    f_min, f_max = 4 * float(p["thrust_min"]), 4 * float(p["thrust_max"])
    scale = 6.0 if sim.control == Control.body_rate else 0.5  # rad/s or rad

    def env_step(data, key):
        ka, kr = jax.random.split(key)
        u = jax.random.uniform(ka, (n, 4))
        action = jnp.concatenate([f_min + (f_max - f_min) * u[:, :1], scale * (2 * u[:, 1:] - 1)], 1)
        data = set_action(sim, data, action)
        data = step_fn(data, n_steps=sub)
        return reset_fn(data, default, jax.random.bernoulli(kr, reset_p, (n,)))  # partial reset

    @jax.jit
    def rollout(data, key):
        return jax.lax.scan(lambda d, k: (env_step(d, k), None), data, jax.random.split(key, n_policy_steps))[0]

    step_jit = jax.jit(env_step)

    def py_loop(data, key):
        for k in jax.random.split(key, n_policy_steps):
            data = step_jit(data, k)
        return data

    res = {"n": n, "dynamics": str(dynamics), "control": str(sim.control), "construct_s": construct_s}
    for name, fn in (("scan", rollout), ("loop", py_loop)):
        times, data = [], sim.data
        for i in range(4):
            t0 = time.perf_counter()
            data = jax.block_until_ready(fn(data, jax.random.key(i)))
            times.append(time.perf_counter() - t0)
        res[f"{name}_compile_s"] = times[0] - min(times[1:])
        res[f"{name}_sps"] = n * n_policy_steps / min(times[1:])
        res[f"{name}_finite"] = bool(jnp.isfinite(data.states.pos).all())
    stats = jax.devices()[0].memory_stats() or {}
    res["jax_peak_mib"] = stats.get("peak_bytes_in_use", 0) / 2**20
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    res["smi_mib"] = next((int(m) for pid, m in (ln.split(",") for ln in smi.splitlines() if ln) if int(pid) == os.getpid()), -1)
    print("BENCH " + json.dumps(res), flush=True)


def report_throughput():
    rows = []
    for dynamics in (Dynamics.first_principles, Dynamics.so_rpy):
        for n in (1024, 4096, 16384, 65536):
            out = subprocess.run(
                [sys.executable, __file__, "--bench", str(n), dynamics], capture_output=True, text=True
            )
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("BENCH ")), None)
            if line is None:
                print(f"bench N={n} {dynamics} FAILED:\n{out.stderr[-2000:]}")
                continue
            rows.append(json.loads(line[6:]))
    print(
        f"\nThroughput on one GPU, {DRONE}, physics {SIM_HZ} Hz, policy {POLICY_HZ} Hz "
        f"({SIM_HZ // POLICY_HZ} substeps), random actions, 1 % of worlds reset (with pose/param "
        "randomization) every policy step. steps/s = N x policy steps per wall second."
    )
    hdr = ("dynamics", "control", "N", "scan steps/s", "loop steps/s", "scan JIT s", "loop JIT s", "Sim() s", "JAX peak MiB", "nvidia-smi MiB", "finite")
    print("".join(f"{h:>17}" for h in hdr))
    for r in rows:
        cells = (r["dynamics"], r["control"], r["n"], f"{r['scan_sps']:.3e}", f"{r['loop_sps']:.3e}", f"{r['scan_compile_s']:.1f}", f"{r['loop_compile_s']:.1f}", f"{r['construct_s']:.1f}", f"{r['jax_peak_mib']:.0f}", r["smi_mib"], r["scan_finite"] and r["loop_finite"])
        print("".join(f"{str(c):>17}" for c in cells))


# region static info


def report_info():
    import mujoco

    print(f"crazyflow {crazyflow.__version__}  jax {jax.__version__}  mujoco {mujoco.__version__}  numpy {np.__version__}  python {sys.version.split()[0]}")
    print(f"jax.devices(): {jax.devices()}  default backend: {jax.default_backend()}")
    print(f"Control modes: {[c.value for c in Control]}  Dynamics: {[d.value for d in Dynamics]}")
    print("\nDrone models (params from crazyflow/dynamics/*/params.toml):")
    hdr = ("drone", "mass kg", "max thrust N", "TWR", "L m", "arm m", "dynamics")
    print("".join(f"{h:>16}" for h in hdr[:-1]) + "  " + hdr[-1])
    for drone in Drone:
        dyn = supported_dynamics(drone)
        p = load_params(dyn[0], drone)
        L = float(load_control_params("mellinger", drone)["core"]["L"])  # motor distance to body axes
        f_max = 4 * float(p["thrust_max"])
        cells = (drone.value, f"{float(p['mass']):.4f}", f"{f_max:.3f}", f"{f_max / (float(p['mass']) * G):.2f}", f"{L:.4f}", f"{L * 2**0.5:.4f}")
        print("".join(f"{c:>16}" for c in cells) + "  " + ",".join(d.value for d in dyn))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        bench(int(sys.argv[2]), sys.argv[3])
    else:
        report_info()
        report_step_response()
        report_throughput()
