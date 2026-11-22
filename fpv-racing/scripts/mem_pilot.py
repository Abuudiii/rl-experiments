"""Memory student pilot: exact initial equivalence with a plain checkpoint, then a short training run.

    .venv-sim/bin/python scripts/mem_pilot.py --plain runs/student/mem/stage1/ckpt_r00.pkl \
        --data /tmp/fpv_mem/stage1/data --out runs/student/mem/pilot.json --train-steps 300

(1) Equivalence: the memory student built with `fpv.student_mem.from_plain` vs the plain student on recorded
frames: one network call; the closed-loop runners stepped through recorded episodes (history + memory state
carried, reset at episode starts); the training-mode sequence call (burn-in + parallel scan) vs the plain
network on the same frame stacks. Reports max |difference| of actions and auxiliary outputs.
(2) Training: ``--train-steps`` updates of `fpv.student_mem.train` from that initialisation on ``--data``;
throughput, peak device memory, losses (finite?), and the size of the memory's output contribution.
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fpv import student as st, student_mem as sm  # noqa: E402
from fpv.dagger_ssm_data import DiskDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--sequence-batch", type=int, default=64)
    ap.add_argument("--sequence-length", type=int, default=16)
    ap.add_argument("--burn-in", type=int, default=16)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--mem-width", type=int, default=128)
    ap.add_argument("--mem-state", type=int, default=16)
    ap.add_argument("--gate-head", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    res = dict(args=vars(args))
    pcfg, pparams, _ = st.load_student(args.plain)
    cfg, params = sm.from_plain(pcfg, pparams, seed=1, mem_width=args.mem_width, mem_state=args.mem_state,
                                gate_head=args.gate_head, sequence_length=args.sequence_length, burn_in=args.burn_in,
                                sequence_batch=args.sequence_batch, sequence_microbatch=args.microbatch)  # fmt: skip
    n_plain, n_mem = st.n_params(pparams), st.n_params(params)
    res.update(params_plain=n_plain, params_mem=n_mem, params_memory_module=st.n_params(params["params"]["memory"]))
    print(f"parameters plain {n_plain} memory student {n_mem} (memory module {res['params_memory_module']})", flush=True)
    data = DiskDataset().add_dir(args.data)
    rng = np.random.default_rng(0)
    K = pcfg.n_frames

    # (1a) single network call on sampled stacks, zero memory state
    frames, pa, _, _ = data.sample(rng, 64, K)
    x = jnp.asarray(frames, jnp.float32) / 255.0
    a0, x0 = st.StudentNet(pcfg).apply(pparams, x, jnp.asarray(pa))
    a1, x1, _, _ = sm.MemNet(cfg).apply(params, x, jnp.asarray(pa), sm.empty_state(cfg, 64), jnp.ones(64, bool))
    res["single_call"] = dict(max_abs_action=float(jnp.abs(a0 - a1).max()), max_abs_aux=float(jnp.abs(x0 - x1).max()),
                              bit_identical_action=bool(jnp.array_equal(a0, a1)))  # fmt: skip

    # (1b) closed-loop runners through recorded steps of 8 slots of one block (history and state carried)
    b = data.blocks[0]
    T, n = b.frames.T, 8
    steps = min(T, 300)
    rp, rm = st.StudentRunner(pcfg, pparams, n), st.StudentRunner(cfg, params, n)
    assert isinstance(rm, sm.StudentRunner)
    # noise floor: each jitted runner vs the eager (un-jitted) plain network on that runner's own frame history
    plain_net = st.StudentNet(pcfg)
    diff, same, starts, floor_p, floor_m = 0.0, True, 0, 0.0, 0.0
    for s in range(steps):
        idx = np.arange(n) * T + s
        fr = b.frames[idx]
        new = (b.start[idx] == idx)
        starts += int(new.sum())
        u, v = rp.act(fr, b.prev_action[idx], new), rm.act(fr, b.prev_action[idx], new)
        diff = max(diff, float(np.abs(u - v).max()))
        same &= bool(np.array_equal(u, v))
        pa_j = jnp.asarray(b.prev_action[idx], jnp.float32)
        ue = np.asarray(plain_net.apply(pparams, rp._hist.astype(jnp.float32) / 255.0, pa_j)[0])
        floor_p = max(floor_p, float(np.abs(u - ue).max()))
        floor_m = max(floor_m, float(np.abs(v - ue).max()))
    res["runner"] = dict(steps=steps, slots=n, episode_starts=starts, max_abs_action=diff, bit_identical=same,
                         plain_jit_vs_eager=floor_p, mem_jit_vs_plain_eager=floor_m)  # fmt: skip

    # (1c) training-mode sequence call (burn-in + scan) vs the plain net on the same stacks
    ext, pa_w, _, _, present, _, _ = sm.sample_windows(data, rng, 4, cfg.sequence_length, cfg.burn_in, K)
    xs = sm._stack(jnp.asarray(ext, jnp.float32) / 255.0, K)  # [4, T, K, H, W, 3]
    am, _, _, _ = sm.MemNet(cfg).apply(params, xs, jnp.asarray(pa_w), sm.empty_state(cfg, 4), jnp.asarray(present), sequence=True)
    ap_ = st.StudentNet(pcfg).apply(pparams, xs.reshape(-1, *xs.shape[2:]), jnp.asarray(pa_w).reshape(-1, 4))[0]
    res["sequence_call"] = dict(max_abs_action=float(jnp.abs(am.reshape(-1, 4) - ap_).max()))
    print(json.dumps({k: res[k] for k in ("single_call", "runner", "sequence_call")}), flush=True)

    # (2) training pilot
    if args.train_steps:
        t0 = time.time()
        new_params, hist = sm.train(cfg, params, data, args.train_steps, seed=3, log=lambda m: print(m, flush=True),
                                    log_every=50, workers=args.workers)  # fmt: skip
        out_k = new_params["params"]["memory"]["out_proj"]["kernel"]
        stats = jax.devices()[0].memory_stats() or {}
        res["train"] = dict(hist=hist, wall_s=time.time() - t0,
                            it_per_s=args.train_steps / hist[-1]["wall_s"],
                            finite=bool(all(np.isfinite(h["loss"]) for h in hist)),
                            out_proj_kernel_abs_max=float(jnp.abs(out_k).max()),
                            peak_bytes=stats.get("peak_bytes_in_use"))  # fmt: skip
        print(json.dumps({k: v for k, v in res["train"].items() if k != "hist"}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
