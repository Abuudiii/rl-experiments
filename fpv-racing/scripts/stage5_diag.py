"""Stage-5 failure diagnosis and paired comparisons from the per-step records of `scripts/eval_stage5.py run`
(<rec-dir>/<outdir>_<block>[_name].npz on the node that flew them).

  1. Paired comparisons on the same flights (flight i of one record = flight i of the other): stage 5 main vs
     variants on flights 0-511, stage 5 vs stage 4 (x1 / perfect re-flown in the stage-5 harness) on all flights.
  2. Crash diagnosis pooled over blocks (eval_stage5.crash_diag) + crash classes and timing.
  3. In-flight detection statistics (eval_stage5.det_stats), pooled.

    .venv-rocm/bin/python scripts/stage5_diag.py --rec-dir /tmp/fpv_percep/Stage5/rec --out runs/percep/stage5/diag.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

BLOCKS = ["random", "circuit", "clockwise", "figure8", "slalom", "climb_dive", "sprint_hairpin"]
KEYS = ("alive", "passed", "ended", "outcome", "vel", "ang_vel", "action", "in_fov", "target", "wrong", "pose_wrong",
        "next_age", "locked", "dist_target", "est_nxt", "reward")  # fmt: skip


def load(rec_dir, tag, block, cols=None):
    f = Path(rec_dir) / f"{tag}_{block}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    rec = {k: z[k] for k in z.files}
    if cols is not None:
        rec = {k: v[:, cols] for k, v in rec.items()}
    for k, v in rec.items():
        if v.dtype == np.float16:
            rec[k] = v.astype(np.float32)
    return rec


def per_flight(rec):
    g = rec["passed"].sum(0)
    crashed = (rec["outcome"] * rec["ended"]).sum(0) > 0
    return {"3laps": g >= 18, "1lap": g >= 6, "crash": crashed}


def paired(a, b):
    """b - a over the same flights: difference of rates with a 95 % normal interval of the paired differences."""
    out = {}
    for k in a:
        x, y = a[k].astype(float), b[k].astype(float)
        d = y - x
        n = len(d)
        se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        out[k] = {"a": float(x.mean()), "b": float(y.mean()), "diff": float(d.mean()),
                  "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                  "only_a": int(((x == 1) & (y == 0)).sum()), "only_b": int(((x == 0) & (y == 1)).sum()), "n": n}  # fmt: skip
    return out


def pool(recs):
    keys = set.intersection(*[set(r) for r in recs])
    return {k: np.concatenate([r[k] for r in recs], 1) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", default="/tmp/fpv_percep/Stage5/rec")
    ap.add_argument("--out", default="runs/percep/stage5/diag.json")
    args = ap.parse_args()
    from eval_stage5 import crash_diag, det_stats

    dt = 0.02
    res = {"paired_variants_first512": {}, "paired_stage4": {}, "crash": {}, "det": {}}
    first = np.arange(512)
    for tag in ("s409", "known", "thr04", "det_imu0"):
        for b in BLOCKS:
            v = load(args.rec_dir, tag, b)
            m = load(args.rec_dir, "main", b, first)
            if v is None or m is None:
                continue
            res["paired_variants_first512"].setdefault(tag, {})[b] = paired(per_flight(m), per_flight(v))
    # stage 5 vs stage 4 re-flown in the stage-5 harness on the same flights
    for s4tag, s5tag in (("s4x1", "main"), ("a_true_perfect", "main"), ("s4_imu1", "main"), ("a_true_perfect", "s4_imu1"),
                         ("s4x1_s409", "s409"), ("s4x1_known", "known")):  # fmt: skip
        for b in BLOCKS:
            s4 = load(args.rec_dir, s4tag, b)
            m = load(args.rec_dir, s5tag, b)
            if s4 is None or m is None:
                continue
            res["paired_stage4"].setdefault(f"{s5tag}_vs_{s4tag}", {})[b] = paired(per_flight(s4), per_flight(m))
    # pooled crash / detection diagnosis: random, layouts pooled, all
    for tag in ("main", "s4x1", "a_true_perfect"):
        recs = {b: load(args.rec_dir, tag, b) for b in BLOCKS}
        recs = {b: r for b, r in recs.items() if r is not None}
        if not recs:
            continue
        groups = {"random": [recs[b] for b in ("random",) if b in recs],
                  "layouts": [recs[b] for b in BLOCKS[1:] if b in recs]}  # fmt: skip
        for gname, rs in groups.items():
            if not rs:
                continue
            p = pool(rs)
            res["crash"].setdefault(tag, {})[gname] = crash_diag(p, dt)
            if "d_valid" in p and tag == "main":
                res["det"].setdefault(tag, {})[gname] = det_stats(p, dt)
            del p
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["paired_variants_first512"], indent=0)[:3000])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
