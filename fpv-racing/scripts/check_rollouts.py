"""Numpy-only consistency check of an exported ``rollouts.npz`` (independent of the jax env code).

    python scripts/check_rollouts.py runs/teacher/<run>/rollouts.npz [--gate-margin 0.05]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv import track  # noqa: E402


def quat_to_mat(q):
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts")
    ap.add_argument("--gate-margin", type=float, default=0.05)
    args = ap.parse_args()
    d = np.load(args.rollouts)
    E, T = d["pos"].shape[:2]
    G = d["gates_pos"].shape[1]
    dt = float(d["t"][1] - d["t"][0])
    valid = np.arange(T)[None] < d["length"][:, None]  # (E, T)
    pair = valid[:, 1:]  # step t and t + 1 both valid
    pos, quat, vel, w = d["pos"], d["quat"], d["vel"], d["ang_vel"]
    rot = quat_to_mat(quat)
    print(f"episodes {E} (nominal {int(d['nominal'].sum())}), steps {T}, dt {dt}, valid steps {int(valid.sum())}")
    print("outcome counts [time limit, gate, ground, oob]:", np.bincount(d["outcome"], minlength=4).tolist())
    print("quat norm error max:", float(np.abs(np.linalg.norm(quat[valid], axis=-1) - 1).max()))

    err = pos[:, 1:] - pos[:, :-1] - 0.5 * (vel[:, 1:] + vel[:, :-1]) * dt
    print("pos vs integrated vel, |err| max [m]:", float(np.abs(err[pair]).max()))

    # Body rates vs the rotation between consecutive frames
    rel = np.einsum("etji,etjk->etik", rot[:, :-1], rot[:, 1:])
    vee = 0.5 * np.stack([rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0],
                          rel[..., 1, 0] - rel[..., 0, 1]], -1) / dt  # fmt: skip
    err = vee - 0.5 * (w[:, 1:] + w[:, :-1])
    print("body rates vs attitude difference, |err| mean / p99 [rad/s]:", float(np.abs(err[pair]).mean()),
          float(np.percentile(np.abs(err[pair]), 99)))  # fmt: skip

    # Observation recomputed from the exported state
    idx = d["target"].astype(int)
    e_idx = np.arange(E)[:, None]
    parts = [np.einsum("etji,etj->eti", rot, vel) / 5.0, rot.reshape(E, T, 9), w / np.array([10.0, 10.0, 5.0])]
    for k in range(2):
        g = (idx + k) % G
        corners = track.gate_corners(d["gates_pos"][e_idx, g], d["gates_yaw"][e_idx, g])  # (E, T, 4, 3)
        parts.append(np.einsum("etji,etcj->etci", rot, corners - pos[:, :, None]).reshape(E, T, 12) / 5.0)
    obs = np.concatenate(parts, -1)
    print("obs[0:39] recomputed, |err| max:", float(np.abs(obs - d["obs"][..., :39])[valid].max()))
    print("obs[39:43] == previous action, |err| max:",
          float(np.abs(d["obs"][:, 1:, 39:] - d["action"][:, :-1])[pair].max()))  # fmt: skip
    print("action range:", float(d["action"][valid].min()), float(d["action"][valid].max()))

    # Gate passes recomputed from consecutive positions
    gp, gy = d["gates_pos"][e_idx, idx][:, :-1], d["gates_yaw"][e_idx, idx][:, :-1]
    passed, _, _ = track.gate_crossing(pos[:, :-1], pos[:, 1:], gp, gy, track.GATE_OPENING / 2 - args.gate_margin)
    mism = (passed != d["passed"][:, :-1]) & pair
    print(f"passes recomputed: {int((passed & pair).sum())} vs exported {int(d['passed'][:, :-1][pair].sum())}, "
          f"mismatching steps {int(mism.sum())}")  # fmt: skip
    adv = (idx[:, 1:] - idx[:, :-1]) % G
    print("target advances exactly on passes:", bool((adv[pair] == d["passed"][:, :-1][pair]).all()))
    print("gate layout offsets from nominal, max |dpos| / |dyaw| (randomised):",
          float(np.abs(d["gates_pos"] - track.circuit_arrays()[0]).max()),
          float(np.abs(d["gates_yaw"] - track.circuit_arrays()[1]).max()),
          "(nominal):", float(np.abs(d["gates_pos"][d["nominal"]] - track.circuit_arrays()[0]).max()))  # fmt: skip


if __name__ == "__main__":
    main()
