"""Where and how the student crashes, from stored student-flown flights (DAgger shards).

    .venv-sim/bin/python scripts/student_failure_sheet.py <data dir> [--round 99] --out <dir>

Uses shards written by ``eval_student.py --save-data`` (or ``run_dagger.py --eval-as-data``): episodes
without resets, so an episode that stops being valid before the time limit crashed. Prints / writes
``failures.json``: crashes by target gate, time of the crash, and for the state ``--before`` steps
before the crash and at the last state: distance to the target gate, whether its centre is in the
camera's field of view, speed, offset from the gate axis, student-vs-teacher action gap. Writes ONE
contact sheet ``failures_sheet.png``: per crash (row) the frames at 40, 20, 10, 5 and 1 steps before
the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fpv import track  # noqa: E402
from fpv.render_synth import project_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--round", type=int, default=99)
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", type=int, default=15)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()
    groups: dict[str, list[Path]] = {}
    for p in sorted(Path(args.data).glob(f"r{args.round:03d}_c*_*.npz")):
        groups.setdefault(p.name.split("_")[1], []).append(p)
    crashes, n_eps, tiles = [], 0, []
    offs = (40, 20, 10, 5, 1)
    for paths in groups.values():
        parts = [np.load(p) for p in paths]
        d = {k: np.concatenate([p[k] for p in parts], 0) for k in ("frames", "valid", "t", "pos", "quat", "vel", "target",
                                                                    "gates_pos", "gates_yaw", "label", "action")}  # fmt: skip
        T, n = d["valid"].shape
        n_eps += n
        h, w = d["frames"].shape[2:4]
        for e in range(n):
            L = int(d["valid"][:, e].sum())
            if L >= args.max_steps or L >= T:
                continue
            rec = dict(length_s=L / 50, target=int(d["target"][L - 1, e]))
            for name, j in (("before", max(L - 1 - args.before, 0)), ("last", L - 1)):
                g = int(d["target"][j, e])
                gp, gy = d["gates_pos"][j, e, g], d["gates_yaw"][j, e, g]
                rel = gp - d["pos"][j, e]
                normal, left = track.gate_axes(gy)
                uv, depth = project_points(d["pos"][j, e], d["quat"][j, e], gp[None], w, h)
                rec[name] = dict(
                    dist=float(np.linalg.norm(rel)), along_axis=float(rel @ normal), lateral=float(-rel @ left),
                    vertical=float(-rel[2]), speed=float(np.linalg.norm(d["vel"][j, e])), z=float(d["pos"][j, e, 2]),
                    gate_in_fov=bool(depth[0] > 0 and 0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h),
                    action_gap=float(np.abs(d["action"][j, e] - d["label"][j, e]).mean()),
                )  # fmt: skip
            crashes.append(rec)
            if len(tiles) < args.rows:
                tiles.append(np.concatenate([d["frames"][max(L - o, 0), e] for o in offs], 1))
    by_gate = np.bincount([c["target"] for c in crashes], minlength=6).tolist()
    agg = lambda name, key: float(np.mean([c[name][key] for c in crashes])) if crashes else None  # noqa: E731
    res = dict(
        episodes=n_eps, crashes=len(crashes), crashes_by_target_gate=by_gate,
        crash_time_s=dict(mean=float(np.mean([c["length_s"] for c in crashes])) if crashes else None,
                          frac_in_first_3s=float(np.mean([c["length_s"] < 3 for c in crashes])) if crashes else None),
        **{name: {k: agg(name, k) for k in ("dist", "along_axis", "speed", "z", "gate_in_fov", "action_gap")}
           | {"abs_lateral": float(np.mean([abs(c[name]["lateral"]) for c in crashes])) if crashes else None,
              "abs_vertical": float(np.mean([abs(c[name]["vertical"]) for c in crashes])) if crashes else None}
           for name in ("before", "last")},
    )  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures.json").write_text(json.dumps(dict(summary=res, crashes=crashes), indent=1))
    print(json.dumps(res, indent=1))
    if tiles:
        from PIL import Image

        sheet = np.concatenate(tiles, 0)
        scale = max(1, 1600 // sheet.shape[1])
        Image.fromarray(sheet).resize((sheet.shape[1] * scale, sheet.shape[0] * scale), Image.NEAREST).save(out / "failures_sheet.png")


if __name__ == "__main__":
    main()
