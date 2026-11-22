"""Backend smoke test + throughput for this project's JAX stack (written for the ROCm bring-up).

    python scripts/amd_jax_smoke.py --out runs/student/mem/amd_smoke_<tag>.json [--skip-cpu] [--bench-only]

Runs on the default JAX backend (first visible GPU) and compares against the host CPU backend in the
same process: matmul / conv / jitted scan, a Crazyflow FlightBatch (teacher actions computed on CPU
are replayed open-loop on both devices), the synthetic renderer at 320x180, and a StudentNet (B
config) forward + one optax step. Then throughput: synthetic frames/s (n=256 poses), FlightBatch
steps/s (n=256 envs, host round-trip included as in DAgger), student train it/s at batch 256.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

TEACHER = "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"
RES: dict = {}


def rec(name, ok, t, **kw):
    RES[name] = dict(ok=bool(ok), seconds=round(t, 3), **kw)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {t:.2f}s {json.dumps(kw, default=float)}", flush=True)


def run(name, fn):
    t0 = time.time()
    try:
        ok, kw = fn()
    except Exception as e:  # report and continue with the next check
        import traceback

        traceback.print_exc()
        ok, kw = False, dict(error=f"{type(e).__name__}: {e}"[:500])
    rec(name, ok, time.time() - t0, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    gpu, cpu = jax.devices()[0], jax.devices("cpu")[0]
    RES["env"] = dict(jax=jax.__version__, backend=jax.default_backend(), device=str(gpu),
                      kind=gpu.device_kind, n_devices=jax.device_count(), host=os.uname().nodename)  # fmt: skip
    print(RES["env"], flush=True)
    from fpv import student as st
    from fpv.closed_loop import FlightBatch
    from fpv.render_synth import SynthRenderer

    cfg = st.StudentConfig(width=320, height=180, n_frames=5, channels=(32, 64, 96, 128, 192, 192))

    if not a.bench_only:

        def t_matmul():
            r = np.random.default_rng(0)
            x, y = r.standard_normal((2048, 2048), np.float32), r.standard_normal((2048, 2048), np.float32)
            ref = np.asarray(jax.device_put(x, cpu) @ jax.device_put(y, cpu))
            d = float(np.abs(np.asarray(jnp.asarray(x) @ jnp.asarray(y)) - ref).max())
            with jax.default_matmul_precision("highest"):
                dh = float(np.abs(np.asarray(jax.jit(jnp.matmul)(jnp.asarray(x), jnp.asarray(y))) - ref).max())
            scale = float(np.abs(ref).max())
            return dh / scale < 1e-5 and d / scale < 1e-2, dict(rel_maxdiff_default=d / scale, rel_maxdiff_highest=dh / scale)

        def t_conv():
            r = np.random.default_rng(1)
            x, w = r.standard_normal((8, 90, 160, 17), np.float32), r.standard_normal((3, 3, 17, 64), np.float32)
            f = jax.jit(lambda x, w: jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))
            ref = np.asarray(f(jax.device_put(x, cpu), jax.device_put(w, cpu)))
            out = np.asarray(f(jnp.asarray(x), jnp.asarray(w)))
            with jax.default_matmul_precision("highest"):
                outh = np.asarray(jax.jit(lambda x, w: jax.lax.conv_general_dilated(
                    x, w, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))(jnp.asarray(x), jnp.asarray(w)))
            s = float(np.abs(ref).max())
            d, dh = float(np.abs(out - ref).max()) / s, float(np.abs(outh - ref).max()) / s
            return dh < 1e-5 and d < 1e-2, dict(rel_maxdiff_default=d, rel_maxdiff_highest=dh)

        def t_scan():
            def f(x0, xs):
                return jax.lax.scan(lambda c, x: (jnp.tanh(c * 0.9 + x), c.sum()), x0, xs)

            r = np.random.default_rng(2)
            x0, xs = r.standard_normal(256, np.float32), r.standard_normal((1000, 256), np.float32)
            g = jax.jit(f)
            c_ref, y_ref = g(jax.device_put(x0, cpu), jax.device_put(xs, cpu))
            c, y = g(jnp.asarray(x0), jnp.asarray(xs))
            d = max(float(np.abs(np.asarray(c) - np.asarray(c_ref)).max()), float(np.abs(np.asarray(y) - np.asarray(y_ref)).max()) / 256)
            return d < 1e-4, dict(maxdiff=d)

        run("matmul_2048", t_matmul)
        run("conv_nhwc", t_conv)
        run("jit_scan_1000", t_scan)
        if jax.device_count() > 1:  # every visible GPU computes the same jitted conv+matmul as the CPU

            def t_alldev():
                r = np.random.default_rng(3)
                x, w = r.standard_normal((4, 90, 160, 8), np.float32), r.standard_normal((3, 3, 8, 32), np.float32)
                f = jax.jit(lambda x, w: jnp.tanh(jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))).sum((1, 2)) @ jnp.ones((32, 16)))
                # reference: CPU conv alone, rest in float64 numpy (XLA:CPU 0.12.0.1 miscompiles this fused
                # conv->tanh->sum->dot graph: garbage, run-to-run different; observed 2026-11-21)
                c = np.asarray(jax.jit(lambda x, w: jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))(jax.device_put(x, cpu), jax.device_put(w, cpu)))
                ref = np.tanh(c.astype(np.float64)).sum((1, 2)) @ np.ones((32, 16))
                d = [float(np.abs(np.asarray(f(jax.device_put(x, g), jax.device_put(w, g))) - ref).max() / np.abs(ref).max())
                     for g in jax.devices()]  # fmt: skip
                return max(d) < 1e-2, dict(n_devices=len(d), rel_maxdiff_per_device=d)

            run("all_devices_conv_matmul", t_alldev)

    # Crazyflow: teacher actions from the CPU run replayed open-loop on the GPU
    poses = {}

    def t_sim():
        n, T = 64, 50
        fc = FlightBatch(n, TEACHER, seed=7, device="cpu")
        fg = FlightBatch(n, TEACHER, seed=7, device="gpu")
        sc, sg = fc.reset(), fg.reset()
        d0 = float(np.abs(sc["pos"] - sg["pos"]).max())
        d1 = None
        dt = []
        for t in range(T):
            act = sc["teacher_action"]
            dt.append(float(np.abs(sc["teacher_action"] - sg["teacher_action"]).max()))
            sc, sg = fc.step(act), fg.step(act)
            if t == 0:
                d1 = float(np.abs(sc["pos"] - sg["pos"]).max())
        alive = ~sc["done"]
        dT = float(np.abs(sc["pos"] - sg["pos"])[alive].max()) if alive.any() else float("nan")
        poses.update(pos=sc["pos"], quat=sc["quat"], target=sc["target"], gp=sc["gates_pos"], gy=sc["gates_yaw"])
        ok = d0 < 1e-5 and d1 < 1e-4 and dT < 1e-2 and (sc["done"] == sg["done"]).all()
        return ok, dict(pos_diff_reset=d0, pos_diff_step1=d1, pos_diff_step50_alive=dT, alive=int(alive.sum()),
                        done_agree=bool((sc["done"] == sg["done"]).all()), teacher_action_maxdiff=max(dt))  # fmt: skip

    def t_render():
        n = 64
        out = {}
        for name, dev in (("cpu", cpu), ("gpu", gpu)):
            with jax.default_device(dev):
                r = SynthRenderer(n, 320, 180, seed=3)
                r.set_gates(np.arange(n), poses["gp"], poses["gy"])
                out[name] = r.render(poses["pos"], poses["quat"])
        diff = np.abs(out["cpu"].astype(np.int16) - out["gpu"].astype(np.int16))
        frac = float((diff > 2).mean())
        return frac < 1e-3, dict(shape=list(out["gpu"].shape), mean_absdiff=float(diff.mean()), max_absdiff=int(diff.max()),
                                 frac_px_diff_gt2=frac, mean_px=float(out["gpu"].mean()))  # fmt: skip

    def t_student():
        net, params = st.init_params(cfg, 0)
        params = jax.device_put(params, cpu)
        r = np.random.default_rng(4)
        b = 8
        fr = r.integers(0, 256, (b, 5, 180, 320, 3), dtype=np.uint8)
        pa = r.uniform(-1, 1, (b, 4)).astype(np.float32)
        lab = r.uniform(-1, 1, (b, 4)).astype(np.float32)
        aux = r.standard_normal((b, st.AUX_DIM)).astype(np.float32)
        res = {}
        for name, dev in (("cpu", cpu), ("gpu", gpu)):
            with jax.default_device(dev):
                p = jax.device_put(params, dev)
                act, _ = jax.jit(net.apply)(p, jnp.asarray(fr, jnp.float32) / 255.0, jnp.asarray(pa))
                opt, step, _ = st.make_train_step(cfg, 1000, warmup=1)  # warmup=1: nonzero lr from step 1 on
                os_, p2, losses = opt.init(p), p, []
                for i in range(3):
                    p2, os_, m = step(p2, os_, jnp.asarray(fr), jnp.asarray(pa), jnp.asarray(lab), jnp.asarray(aux), jax.random.key(5 + i))
                    losses.append(float(m["loss"]))
                res[name] = (np.asarray(act), losses[-1], p2, losses)
        da = float(np.abs(res["cpu"][0] - res["gpu"][0]).max())
        dl = abs(res["cpu"][1] - res["gpu"][1])
        def l2(a, b):
            return float(np.sqrt(sum(float(np.square(np.asarray(x, np.float64) - np.asarray(y, np.float64)).sum())
                                     for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))))  # fmt: skip

        dp = l2(res["cpu"][2], res["gpu"][2]) / l2(res["cpu"][2], params)  # GPU-vs-CPU gap relative to the 3-step update
        return da < 1e-3 and dl / res["cpu"][1] < 1e-3 and dp < 0.05, dict(
            action_maxdiff=da, losses_cpu=res["cpu"][3], losses_gpu=res["gpu"][3], loss3_reldiff=dl / res["cpu"][1],
            param_3steps_rel_l2=dp, n_params=st.n_params(params))  # fmt: skip

    if not a.bench_only:
        run("crazyflow_flightbatch_vs_cpu", t_sim)
        if poses:
            run("render_synth_320x180_vs_cpu", t_render)
        run("studentnet_fwd_step_vs_cpu", t_student)

    # Throughput (default device)
    n = a.n

    def b_render():
        fb = FlightBatch(n, TEACHER, seed=11)
        s = fb.reset()
        r = SynthRenderer(n, 320, 180, seed=3)
        r.set_gates(np.arange(n), s["gates_pos"], s["gates_yaw"])
        r.render_device(s["pos"], s["quat"]).block_until_ready()
        reps, t0 = 20, time.time()
        for _ in range(reps):
            x = r.render_device(s["pos"], s["quat"])
        x.block_until_ready()
        dt = time.time() - t0
        t1 = time.time()
        for _ in range(reps):
            np.asarray(r.render_device(s["pos"], s["quat"]))
        dt2 = time.time() - t1
        return True, dict(frames_per_s_device=n * reps / dt, frames_per_s_to_host=n * reps / dt2, n=n)

    def b_sim():
        fb = FlightBatch(n, TEACHER, seed=12)
        s = fb.reset()
        for _ in range(5):
            s = fb.step(s["teacher_action"])
        steps, t0 = 200, time.time()
        for _ in range(steps):
            s = fb.step(s["teacher_action"])
            if s["done"].any():
                s = fb.reset(s["done"])
        dt = time.time() - t0
        return True, dict(batch_steps_per_s=steps / dt, env_steps_per_s=n * steps / dt, n=n)

    def b_train():
        net, params = st.init_params(cfg, 0)
        opt, step, _ = st.make_train_step(cfg, 10000)
        os_ = opt.init(params)
        r = np.random.default_rng(6)
        B = 256
        fr = jnp.asarray(r.integers(0, 256, (B, 5, 180, 320, 3), dtype=np.uint8))
        pa, lab = jnp.asarray(r.uniform(-1, 1, (B, 4)), jnp.float32), jnp.asarray(r.uniform(-1, 1, (B, 4)), jnp.float32)
        aux = jnp.asarray(r.standard_normal((B, st.AUX_DIM)), jnp.float32)
        key = jax.random.key(0)
        t0 = time.time()
        params, os_, m = step(params, os_, fr, pa, lab, aux, key)
        jax.block_until_ready(m)
        tc = time.time() - t0
        reps, t0 = 30, time.time()
        for i in range(reps):
            params, os_, m = step(params, os_, fr, pa, lab, aux, jax.random.fold_in(key, i))
        jax.block_until_ready(m)
        dt = time.time() - t0
        return bool(np.isfinite(float(m["loss"]))), dict(it_per_s=reps / dt, compile_s=tc, batch=B, loss=float(m["loss"]))

    run("bench_render_320x180", b_render)
    run("bench_sim_flightbatch", b_sim)
    run("bench_train_b256", b_train)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(RES, f, indent=1, default=float)
    print("wrote", a.out, "all_ok", all(v["ok"] for k, v in RES.items() if k != "env"), flush=True)


if __name__ == "__main__":
    main()
