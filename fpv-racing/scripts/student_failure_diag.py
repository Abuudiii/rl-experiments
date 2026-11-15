"""Where and why does the student fail on a layout? Student and teacher on the same episodes.

    CUDA_VISIBLE_DEVICES=3 .venv-sim/bin/python scripts/student_failure_diag.py runs/student/<run>/ckpt_latest.pkl \
        --teacher runs/teacher/<run>/ckpt_latest.pkl --renderer-args '{"highlight_target": true}' \
        --scenarios figure8,sprint_hairpin,random --episodes 128 --seed 310011 --out <dir>/diag.json

Both fly ``--episodes`` randomised episodes of 30 s per layout from the same `FlightBatch` seed
(student from its synthetic camera frames, teacher from the state). Reported per layout:
  * three-lap rate with a 20 s and a 30 s cap (round evaluations used 20 s; a figure-8 lap of the
    teacher takes about 6.7 s, so three laps barely fit in 20 s);
  * per leg (target gate k = the leg into gate k): mean leg time and speed of student and teacher,
    the student's crashes on that leg by cause, and the share of steps with the target gate centre
    inside the camera image (student / teacher);
  * for the student's crashes: the target gate, distance to it, position relative to its plane
    (along / left / up in the gate frame), whether the target was in view 0.5 s before the crash,
    student speed in the last second versus the teacher's speed at the same distance to that gate;
  * episodes that end without a crash and without three laps ("slow / lost").
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
from fpv.render_synth import project_points  # noqa: E402


def arrays(tr):
    pos = np.stack([t["pos"] for t in tr])  # [T, n, 3]
    quat = np.stack([t["quat"] for t in tr])
    tgt = np.stack([t["target"] for t in tr]).astype(int)
    gp = np.stack([t["gates_pos"] for t in tr])  # [T, n, 6, 3]
    gy = np.stack([t["gates_yaw"] for t in tr])
    alive = np.stack([t["alive"] for t in tr])
    return pos, quat, tgt, gp, gy, alive


def target_visible(pos, quat, tgt, gp, w, h):
    uv, depth = project_points(pos.reshape(-1, 3), quat.reshape(-1, 4), gp.reshape(-1, 6, 3), w, h)
    uv, depth = uv.reshape(*tgt.shape, 6, 2), depth.reshape(*tgt.shape, 6)
    vis = (depth > 0.3) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    return np.take_along_axis(vis, tgt[..., None], -1)[..., 0]


def fly(fb, renderer, runner, steps, seed):
    _, eps, info = dagger.evaluate(fb, renderer, runner, steps=steps, rng=np.random.default_rng(seed), trace="noframes")
    return {e["slot"]: e for e in eps}, info["trace"]


def join(batches):
    """(eps, trace) of several batches -> one batch with the slots side by side (shorter traces padded as not alive)."""
    eps, tr, off = {}, [], 0
    T = max(len(t) for _, t in batches)
    for b_eps, b_tr in batches:
        n = np.asarray(b_tr[0]["pos"]).shape[0]
        eps |= {off + i: dict(e, slot=off + i) for i, e in b_eps.items()}
        off += n
    for t in range(T):
        row = {}
        for k in ("pos", "quat", "target", "gates_pos", "gates_yaw", "alive"):
            parts = []
            for _, b_tr in batches:
                x = np.asarray(b_tr[min(t, len(b_tr) - 1)][k])
                parts.append(np.zeros_like(x) if k == "alive" and t >= len(b_tr) else x)
            row[k] = np.concatenate(parts, axis=0)
        tr.append(row)
    return eps, tr


def analyse(name, s_eps, s_tr, t_eps, t_tr, w, h, hz):
    sp, sq, stg, sgp, sgy, sal = arrays(s_tr)
    tp, tq, ttg, tgp, tgy, tal = arrays(t_tr)
    s_vis, t_vis = target_visible(sp, sq, stg, sgp, w, h), target_visible(tp, tq, ttg, tgp, w, h)
    s_speed = np.linalg.norm(np.diff(sp, axis=0), axis=-1) * hz  # [T-1, n]
    t_speed = np.linalg.norm(np.diff(tp, axis=0), axis=-1) * hz
    n = sp.shape[1]

    def gates_by(eps, step):
        return np.array([sum(1 for k in eps[i]["pass_steps"] if k <= step) for i in range(n)])

    out = dict(episodes=n)
    for who, eps in (("student", s_eps), ("teacher", t_eps)):
        out[who] = dict(
            three_laps_20s=float((gates_by(eps, 1000) >= 18).mean()), three_laps_30s=float((gates_by(eps, 1500) >= 18).mean()),
            one_lap_30s=float((gates_by(eps, 1500) >= 6).mean()),
            crash=float(np.mean([eps[i]["outcome"] > 0 for i in range(n)])),
            crash_by_cause={dagger.OUTCOMES[c]: float(np.mean([eps[i]["outcome"] == c for i in range(n)])) for c in (1, 2, 3)},
            slow_or_lost=float(np.mean([eps[i]["outcome"] == 0 and len(eps[i]["pass_steps"]) < 18 for i in range(n)])),
            gates_per_episode=float(np.mean([eps[i]["gates"] for i in range(n)])),
            speed=float(np.mean([eps[i]["speed"] for i in range(n)])),
        )  # fmt: skip
    legs = {}
    for k in range(6):
        row = {}
        for who, eps, tg, al, spd, vis in (("student", s_eps, stg, sal, s_speed, s_vis), ("teacher", t_eps, ttg, tal, t_speed, t_vis)):
            m = (tg[:-1] == k) & al[:-1] & al[1:]
            times = []
            for i in range(n):  # leg time = steps between the pass into gate k-1 and the pass of gate k
                p = np.asarray(eps[i]["pass_steps"])
                tg_i = tg[np.minimum(p - 1, len(tg) - 1), i] if len(p) else []
                for a, b, g in zip(p[:-1], p[1:], tg_i[1:]):
                    if g == k:
                        times.append((b - a) / hz)
            gaps = []  # per traversal of the leg: longest run of consecutive steps without the target in view
            for i in range(n):
                mi, vi = m[:, i], vis[:-1, i]
                run = best = 0
                for t in range(len(mi)):
                    if mi[t]:
                        run = run + 1 if not vi[t] else 0
                        best = max(best, run)
                    elif t > 0 and mi[t - 1]:
                        gaps.append(best)
                        run = best = 0
                if len(mi) and mi[-1]:
                    gaps.append(best)
            gaps = np.asarray(gaps, float) / hz
            row[who] = dict(leg_time_s=float(np.mean(times)) if times else None, n_legs=len(times),
                            speed=float(spd[m].mean()) if m.any() else None,
                            target_in_view=float(vis[:-1][m].mean()) if m.any() else None,
                            longest_out_of_view_s=dict(mean=float(gaps.mean()), p90=float(np.percentile(gaps, 90)),
                                                       max=float(gaps.max()), frac_over_0p1s=float((gaps > 0.1).mean()))
                            if len(gaps) else None)  # fmt: skip
        crashes = [i for i in range(n) if s_eps[i]["outcome"] > 0 and s_eps[i]["crash_target"] == k]
        row["student_crashes"] = len(crashes)
        row["student_crash_causes"] = {dagger.OUTCOMES[c]: sum(s_eps[i]["outcome"] == c for i in crashes) for c in (1, 2, 3)}
        legs[f"into_G{k + 1}"] = row
    out["legs"] = legs
    crashes = []
    for i in range(n):
        e = s_eps[i]
        if e["outcome"] == 0:
            continue
        T = e["length"]
        j = min(T, len(sp)) - 1  # last recorded step of this slot
        k = e["crash_target"]
        g, y = sgp[j, i, k], sgy[j, i, k]
        fwd, left = track.gate_axes(np.float64(y))
        rel = sp[j, i] - g
        d = float(np.linalg.norm(rel))
        j5 = max(j - int(0.5 * hz), 0)
        # teacher speed at the same distance to the same gate (same episode, same gate, approach)
        t_d = np.linalg.norm(tp[:-1, i] - g, axis=-1)
        ok = (ttg[:-1, i] == k) & tal[:-1, i] & tal[1:, i]
        t_at = t_speed[:, i][ok & (np.abs(t_d - d) < 0.4)]
        crashes.append(dict(
            slot=i, cause=dagger.OUTCOMES[e["outcome"]], t_s=T / hz, gates=e["gates"], target=f"G{k + 1}",
            dist_to_target=d, along=float(rel @ fwd), left=float(rel @ left), up=float(rel[2]), z=float(sp[j, i, 2]),
            target_in_view_0p5s_before=bool(s_vis[j5, i]), target_in_view_last_s=float(s_vis[max(j - int(hz), 0) : j + 1, i].mean()),
            student_speed_last_s=float(s_speed[max(j - int(hz), 0) : j, i].mean()) if j > 0 else None,
            teacher_speed_same_gate_same_dist=float(t_at.mean()) if len(t_at) else None,
        ))  # fmt: skip
    out["student_crashes"] = crashes
    if crashes:
        out["crash_summary"] = dict(
            by_target={g: sum(c["target"] == g for c in crashes) for g in sorted({c["target"] for c in crashes})},
            target_in_view_0p5s_before=float(np.mean([c["target_in_view_0p5s_before"] for c in crashes])),
            median_dist_to_target=float(np.median([c["dist_to_target"] for c in crashes])),
            median_along=float(np.median([c["along"] for c in crashes])),
            student_speed_last_s=float(np.mean([c["student_speed_last_s"] for c in crashes if c["student_speed_last_s"] is not None])),
            teacher_speed_same_place=float(np.mean([c["teacher_speed_same_gate_same_dist"] for c in crashes
                                                    if c["teacher_speed_same_gate_same_dist"] is not None] or [np.nan])),
        )  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--scenarios", default="figure8,sprint_hairpin")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--held-out-from", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default="synth", help="synth (one batch of --episodes slots) or airsim (batches of its slots)")
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    cfg, params, _ = st.load_student(args.checkpoint)
    renderer = dagger.make_renderer(args.renderer, args.episodes, cfg.width, cfg.height, seed=args.seed,
                                    **json.loads(args.renderer_args))  # fmt: skip
    n_batches = -(-args.episodes // renderer.n)
    for b in range(n_batches):  # batch b flies FlightBatch seed --seed + b
        dagger.check_held_out(args.seed + b, args.held_out_from)
    runner = st.StudentRunner(cfg, params, renderer.n)
    res = dict(checkpoint=args.checkpoint, seed=args.seed, batch_seeds=[args.seed + b for b in range(n_batches)],
               steps=args.steps, renderer=args.renderer, highlight_target=dagger.highlight_on(renderer), scenarios={})  # fmt: skip
    for name in args.scenarios.split(","):
        # "random": the teacher's generator; "hairpin_s": dagger.hairpin_s_tracks drawn with --seed (S leg into G5)
        s_b, t_b = [], []
        for b in range(n_batches):
            sd = args.seed + b
            layout = (None if name == "random" else dagger.hairpin_s_tracks(renderer.n, sd) if name == "hairpin_s"
                      else name)  # fmt: skip
            mk = lambda: FlightBatch(renderer.n, args.teacher, seed=sd, randomize=True, scenario=layout)  # noqa: E731
            fb = mk()
            s_b.append(fly(fb, renderer, runner, args.steps, sd))
            t_b.append(fly(mk(), renderer, None, args.steps, sd))
            log(f"{name}: batch {b + 1}/{n_batches} done")
        s_eps, s_tr = join(s_b)
        t_eps, t_tr = join(t_b)
        res["scenarios"][name] = analyse(name, s_eps, s_tr, t_eps, t_tr, cfg.width, cfg.height, fb.cfg.policy_hz)
        r = res["scenarios"][name]
        log(f"{name}: student {r['student']}  teacher 3 laps 20s/30s {r['teacher']['three_laps_20s']:.3f}/"
            f"{r['teacher']['three_laps_30s']:.3f}  crash summary {r.get('crash_summary')}")  # fmt: skip
        for k, v in r["legs"].items():
            log(f"   {k}: {v}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
