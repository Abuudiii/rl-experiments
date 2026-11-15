"""How far are generated tracks from the named layouts? Acceptance rate of `fpv.dagger.hairpin_s_tracks`.

    .venv-sim/bin/python scripts/check_hairpin_tracks.py --n 4096 --seed 990101 --out <dir>/hairpin_tracks.json

Distance of two 6-gate tracks = gate-position RMS [m] after the best alignment: any cyclic relabelling
of the gates, either direction, optionally mirrored, rotated about z and translated (closed-form 2D
Kabsch in xy, mean offset in z). Reported for `hairpin_s_tracks` and, as the reference, for plain
generator tracks (`fpv.track.sample_tracks`, valid ones): per named layout the minimum and the 1st
percentile over the tracks, plus the distances between the named layouts themselves for scale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from fpv import dagger, track  # noqa: E402


def orderings(p):
    """[N, 6, 3] -> [N, 24, 6, 3]: 6 cyclic shifts x 2 directions x (as is, mirrored in y)."""
    out = []
    for mirror in (1.0, -1.0):
        q = p * np.array([1.0, mirror, 1.0])
        for rev in (False, True):
            r = q[:, ::-1] if rev else q
            out += [np.roll(r, s, axis=1) for s in range(6)]
    return np.stack(out, 1)


def aligned_rms(p, q):
    """min over orderings of p ([N, 6, 3]) of the gate RMS to q ([6, 3]) after rotation about z + translation."""
    a = orderings(p.astype(np.float64))  # [N, K, 6, 3]
    a = a - a.mean(2, keepdims=True)
    b = q.astype(np.float64) - q.mean(0)
    s = (a[..., 0] * b[:, 1] - a[..., 1] * b[:, 0]).sum(-1)
    c = (a[..., 0] * b[:, 0] + a[..., 1] * b[:, 1]).sum(-1)
    th = np.arctan2(s, c)[..., None]
    x = a[..., 0] * np.cos(th) - a[..., 1] * np.sin(th)
    y = a[..., 0] * np.sin(th) + a[..., 1] * np.cos(th)
    d2 = (x - b[:, 0]) ** 2 + (y - b[:, 1]) ** 2 + (a[..., 2] - b[:, 2]) ** 2
    return np.sqrt(d2.mean(-1)).min(-1)  # [N]


def summary(p):
    out = {}
    for name in track.SCENARIOS:
        d = aligned_rms(p, track.circuit_arrays(track.resolve(name))[0])
        out[name] = dict(min=float(d.min()), p01=float(np.percentile(d, 1)), median=float(np.median(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stats: dict = {}
    hp, _ = dagger.hairpin_s_tracks(args.n, args.seed, stats=stats)
    pos, _, ok = (np.asarray(a) for a in track.sample_tracks(jax.random.key(args.seed), args.n))
    named = {k: track.circuit_arrays(track.resolve(k))[0] for k in track.SCENARIOS}
    between = {f"{a}~{b}": float(aligned_rms(named[a][None], named[b])[0]) for a in named for b in named if a < b}
    res = dict(seed=args.seed, n=args.n, hairpin_generator=stats, acceptance=stats["accepted"] / stats["drawn"],
               hairpin_s=summary(hp), generator=summary(pos[ok]), generator_n=int(ok.sum()),
               between_named=between, metric="gate-position RMS [m] after best relabelling/direction/mirror, z-rotation, translation")  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
