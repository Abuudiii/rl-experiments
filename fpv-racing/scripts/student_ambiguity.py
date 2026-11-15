"""Is the student's error tied to not knowing which visible gate is the target?

    CUDA_VISIBLE_DEVICES=3 .venv-sim/bin/python scripts/student_ambiguity.py runs/student/<run>/ckpt_latest.pkl \
        --teacher runs/teacher/<run>/ckpt_latest.pkl --scenarios random,figure8,slalom --episodes 128 \
        --seed 310001 --out runs/student/<run>/ambiguity.json

The student flies alone on the synthetic renderer (randomised layouts, standard spawns, ``--steps``
cap). For every flown step: how many gate centres are in the camera image (in front, inside the
frame), whether the target gate is one of them, whether the gate nearest the image centre is the
target, and how many steps ago the target changed (a gate was passed). Reported per condition:
share of steps, mean |student - teacher| action error, and share of the steps in the last
``--pre-crash`` steps before a crash (enrichment = that share / share of all steps).
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


def step_features(tr, w, h):
    """Per-step arrays [T, n] from a `dagger.fly` trace."""
    pos, quat = np.stack([t["pos"] for t in tr]), np.stack([t["quat"] for t in tr])
    gp = np.stack([t["gates_pos"] for t in tr])
    tgt = np.stack([t["target"] for t in tr]).astype(int)
    uv, depth = project_points(pos.reshape(-1, 3), quat.reshape(-1, 4), gp.reshape(-1, 6, 3), w, h)
    uv, depth = uv.reshape(*tgt.shape, 6, 2), depth.reshape(*tgt.shape, 6)
    vis = (depth > 0.3) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    d_c = np.where(vis, np.hypot(uv[..., 0] - w / 2, uv[..., 1] - h / 2), np.inf)
    nearest = d_c.argmin(-1)
    tgt_vis = np.take_along_axis(vis, tgt[..., None], -1)[..., 0]
    changed = np.concatenate([np.zeros((1, tgt.shape[1]), bool), tgt[1:] != tgt[:-1]])
    idx = np.where(changed, np.arange(len(tgt))[:, None], -10**6)
    since = np.arange(len(tgt))[:, None] - np.maximum.accumulate(idx, 0)  # steps since the last gate pass (huge before it)
    err = np.stack([np.abs(np.clip(t["action"], -1, 1) - t["label"]).mean(-1) for t in tr])
    alive = np.stack([t["alive"] for t in tr])
    return dict(n_vis=vis.sum(-1), tgt_vis=tgt_vis, centre_is_tgt=vis.any(-1) & (nearest == tgt), since=since,
                err=err, alive=alive)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--scenarios", default="random")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pre-crash", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    cfg, params, _ = st.load_student(args.checkpoint)
    n = args.episodes
    renderer = dagger.make_renderer("synth", n, cfg.width, cfg.height, seed=args.seed, **json.loads(args.renderer_args))
    runner = st.StudentRunner(cfg, params, n)
    res = dict(checkpoint=args.checkpoint, seed=args.seed, layouts={})
    for name in args.scenarios.split(","):
        fb = FlightBatch(n, args.teacher, seed=args.seed, randomize=True, scenario=None if name == "random" else name)
        eps, info = dagger.fly(fb, renderer, runner, steps=args.steps, reset_done=False, trace="noframes",
                               rng=np.random.default_rng(args.seed))  # fmt: skip
        f = step_features(info["trace"], cfg.width, cfg.height)
        T = len(f["err"])
        pre = np.zeros_like(f["alive"])
        for e in eps:
            if e["outcome"] > 0:
                L = int(f["alive"][:, e["slot"]].sum())
                pre[max(L - args.pre_crash, 0) : L, e["slot"]] = True
        a = f["alive"]
        conds = {
            "all": np.ones_like(a),
            "0 gates in view": f["n_vis"] == 0, "1 gate in view": f["n_vis"] == 1,
            "2 gates in view": f["n_vis"] == 2, ">=3 gates in view": f["n_vis"] >= 3,
            "target not in view": ~f["tgt_vis"],
            "centre gate is the target": f["centre_is_tgt"],
            "centre gate is NOT the target (some gate in view)": (f["n_vis"] > 0) & ~f["centre_is_tgt"],
            "<15 steps after a gate pass": f["since"] < 15,
            ">=2 in view and centre gate not the target": (f["n_vis"] >= 2) & ~f["centre_is_tgt"],
        }  # fmt: skip
        rows = {}
        for k, c in conds.items():
            m = c & a
            share = float(m.sum() / max(a.sum(), 1))
            pre_share = float((m & pre).sum() / max((pre & a).sum(), 1))
            rows[k] = dict(share_of_steps=round(share, 4), action_mae=round(float(f["err"][m].mean()), 4) if m.any() else None,
                           share_of_pre_crash_steps=round(pre_share, 4),
                           enrichment_before_crash=round(pre_share / share, 2) if share > 0 else None)  # fmt: skip
        summ = dagger.summarize(eps)
        res["layouts"][name] = dict(frac_3_laps=summ["frac_3_laps"], crash_rate=summ["crash_rate"], steps=T,
                                    conditions=rows)  # fmt: skip
        log(f"{name}: 3 laps {summ['frac_3_laps']:.3f} crash {summ['crash_rate']:.3f}")
        for k, r in rows.items():
            log(f"   {k:52s} {r}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
