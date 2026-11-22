"""Held-out closed-loop evaluation of student checkpoints on named layouts, with a wrong-gate breakdown.

    .venv-sim/bin/python scripts/mem_eval_round.py runs/student/mem/stage1/ckpt_r00.pkl [more ckpts] \
        --teacher runs/teacher/rt_G_D_noise3/ckpt_latest.pkl --layouts figure8 --episodes 256 --nominal 24 \
        --steps 1500 --seed 720001 --held-out-from /tmp/fpv_mem/stage1/data --out-dir runs/student/mem/eval/stage1

Per checkpoint, layout and setting (randomised: per-gate jitter + physics randomisation, nominal: neither;
spawn noise in both) the student flies alone (fresh renderer and FlightBatch from ``--seed`` for every
checkpoint, so all checkpoints see the same episodes). Writes ``<out-dir>/<ckpt stem>.json``: `fpv.dagger.summarize`
metrics plus, from the per-step trace,

    wrong_gate_pass_episodes   share of episodes that flew through the opening of a gate other than the target
    crash_breakdown            per target gate at the crash: crashes, of which "wrong_heading" (in the last
                               0.5 s the velocity pointed closest to a gate other than the target, within 30 deg),
                               "hit_other_gate" (gate crash on a gate other than the target), "near_crossing"
                               (within 1.5 m (xy) of the crossing point of the legs G2->G3 and G5->G0, figure8 only)

``--teacher-ref`` also flies the teacher on the same episodes (once is enough).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

from fpv import dagger, student as st, track  # noqa: E402
from fpv.closed_loop import FlightBatch  # noqa: E402


def crossing_point(name):
    if name != "figure8":
        return None
    p, _ = track.circuit_arrays(track.resolve(name))
    a0, a1, b0, b1 = p[2, :2], p[3, :2], p[5, :2], p[0, :2]
    m = np.array([a1 - a0, b0 - b1]).T
    s, _ = np.linalg.solve(m, b0 - a0)
    return a0 + s * (a1 - a0)


def analyse(trace, episodes, cross):
    """Wrong-gate events from a `fly` trace ("noframes"); one episode per slot (evaluation)."""
    pos = np.stack([t["pos"] for t in trace])  # [T, n, 3]
    alive = np.stack([t["alive"] for t in trace])
    target = np.stack([t["target"] for t in trace])
    gp, gy = trace[0]["gates_pos"], trace[0]["gates_yaw"]  # [n, 6, 3], [n, 6] fixed during an episode
    T, n = alive.shape
    wrong_pass = np.zeros(n, bool)
    for k in range(1, T):
        live = alive[k - 1]
        if not live.any():
            break
        passed, _, _ = track.gate_crossing(pos[k - 1][:, None], pos[k][:, None], gp, gy)  # [n, 6]
        other = passed & (np.arange(6)[None] != target[k - 1][:, None])
        wrong_pass |= live & other.any(-1)
    rows = []
    for e in episodes:
        if e["outcome"] == 0:
            continue
        s = e["slot"]
        last = int(np.flatnonzero(alive[:, s])[-1])
        tgt = int(target[last, s])
        k0 = max(last - 25, 0)
        v = pos[last, s] - pos[k0, s]
        rel = gp[s] - pos[last, s]
        dist = np.linalg.norm(rel, axis=-1)
        cosang = (rel @ v) / np.maximum(dist * np.linalg.norm(v), 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        ang[dist > 8.0] = 180.0
        head = int(np.argmin(ang))
        nearest = int(np.argmin(np.linalg.norm(gp[s] - pos[min(last + 1, T - 1), s], axis=-1)))
        rows.append(dict(target=tgt, outcome=e["outcome"], wrong_heading=bool(head != tgt and ang[head] < 30),
                         hit_other_gate=bool(e["outcome"] == 1 and nearest != tgt),
                         near_crossing=bool(cross is not None and np.linalg.norm(pos[last, s, :2] - cross) < 1.5),
                         wrong_pass=bool(wrong_pass[s]), t_s=last / 50.0, gates=e["gates"]))  # fmt: skip
    by = {}
    for g in range(6):
        r = [x for x in rows if x["target"] == g]
        by[g] = dict(crashes=len(r), **{k: sum(x[k] for x in r) for k in ("wrong_heading", "hit_other_gate",
                                                                           "near_crossing", "wrong_pass")})
    return dict(wrong_gate_pass_episodes=float(wrong_pass.mean()), crash_breakdown=by, crashes=rows)


def fly_eval(fb, renderer, runner, episodes, steps, seed, cross, log):
    """``episodes`` <= fb.n: one batch, slots beyond ``episodes`` are ignored."""
    rng = np.random.default_rng(seed)
    eps, info = dagger.fly(fb, renderer, runner, steps=steps, beta=0.0, reset_done=False, rng=rng,
                           trace="noframes", render=runner is not None, log=log)  # fmt: skip
    trace = info.pop("trace")
    eps = [e for e in eps if e["slot"] < episodes]
    out = dict(dagger.summarize(eps), wall_s=info["wall_s"])
    out.update(analyse(trace, eps, cross))
    return out


def eval_policy(cfg, params, teacher, layouts, episodes, nominal, steps, seed, log, name="student"):
    """Fly ``params`` (or the teacher when ``cfg`` is None) -> {layout: {setting: metrics}}; same episodes per seed."""
    w, h = (cfg.width, cfg.height) if cfg is not None else (320, 180)
    out = {}
    for li, layout in enumerate(layouts):
        cross = crossing_point(layout)
        out[layout] = {}
        for setting, randomize, count in (("randomised", True, episodes), ("nominal", False, nominal)):
            if count <= 0:
                continue
            s = seed + 7919 * li + (0 if randomize else 1)
            renderer = dagger.make_renderer("synth", count, w, h, seed=s)
            fb = FlightBatch(count, teacher, seed=s, randomize=randomize, scenario=layout)
            runner = None if cfg is None else st.StudentRunner(cfg, params, count)
            m = fly_eval(fb, renderer, runner, count, steps, s, cross, None)
            renderer.close()
            out[layout][setting] = m
            bd = {g: (v["crashes"], v["wrong_heading"], v["near_crossing"]) for g, v in m["crash_breakdown"].items()}
            log(f"{name} {layout} {setting}: n {m['episodes']} 3 laps {m['frac_3_laps']:.3f} 1 lap "
                f"{m['frac_1_lap']:.3f} crash {m['crash_rate']:.3f} mae {m.get('action_mae_vs_teacher', float('nan')):.4f} "
                f"wrong-pass {m['wrong_gate_pass_episodes']:.3f} by target (crash, wrong heading, near crossing) {bd} "
                f"({m['wall_s']:.0f} s)")  # fmt: skip
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--layouts", default="figure8")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--nominal", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--held-out-from", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--teacher-ref", action="store_true")
    ap.add_argument("--speed", action="store_true", help="also time the runner step at batch 1 and 64")
    args = ap.parse_args()
    log = dagger.log
    dagger.check_held_out(args.seed, args.held_out_from)
    layouts = list(track.SCENARIOS) if args.layouts == "all" else args.layouts.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoints + (["teacher"] if args.teacher_ref else []):
        out = out_dir / ("teacher.json" if ck == "teacher" else Path(ck).stem + ".json")
        cfg = params = None
        res = dict(checkpoint=ck, seed=args.seed, steps=args.steps)
        if ck != "teacher":
            cfg, params, meta = st.load_student(ck)
            res.update(parameters=st.n_params(params), round=meta.get("round"), arch="mem" if hasattr(cfg, "mem_width") else "cnn")
        t0 = time.time()
        res["layouts"] = eval_policy(cfg, params, args.teacher, layouts, args.episodes, args.nominal, args.steps,
                                     args.seed, log, name=Path(ck).stem)  # fmt: skip
        if args.speed and ck != "teacher":
            res["inference"] = dagger.inference_speed(cfg, params) if hasattr(cfg, "mem_width") else plain_speed(cfg, params)
            log(f"inference {res['inference']}")
        res["wall_s"] = time.time() - t0
        write_json(out, res)
        log(f"wrote {out}")


def plain_speed(cfg, params, batches=(1, 64), reps=200):
    """The plain runner's one-step update (history roll + network), timed like `fpv.student_mem.inference_speed`."""
    import jax.numpy as jnp

    out = {}
    for b in batches:
        r = st.StudentRunner(cfg, params, b)
        frame, new, pa = jnp.zeros((b, cfg.height, cfg.width, 3), jnp.uint8), jnp.zeros(b, bool), jnp.zeros((b, 4))
        hist, a, _ = r._act(params, r._hist, frame, new, pa)
        a.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            hist, a, _ = r._act(params, hist, frame, new, pa)
            a.block_until_ready()
        dt = (time.perf_counter() - t0) / reps
        out[f"batch_{b}"] = dict(ms_per_call=1e3 * dt, calls_per_s=1 / dt)
    return out


if __name__ == "__main__":
    main()
