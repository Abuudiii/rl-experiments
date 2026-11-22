"""Closed-loop error by episode time, and a carried-memory vs reset-window control, for one SSM student.

One held-out student-only evaluation on the synthetic renderer (random tracks, narrow spawn), traced
with frames for analysis only (frames are never saved). Reports, per time bin, the absolute error of
the applied student action against the teacher label at the same state, together with how many
episodes are still alive (late bins only contain survivors). Then, for sampled alive states with
t >= 200, it re-runs the network from a ZERO memory state over the last ``--window`` frames and the
previous actions that were actually applied (the training forward: 16 burn-in + 32 supervised), and
compares that action's error with the error of the action the fully carried memory produced at the
same state. Writes small statistics only.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.pop("JAX_DEFAULT_MATMUL_PRECISION", None)

import jax
import jax.numpy as jnp
import numpy as np

from fpv import dagger, student as st, student_ssm as ssm

CHANNELS = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")
BINS = ((0, 50), (50, 200), (200, 1000))


class RecordingRunner(ssm.StudentRunner):
    """The closed-loop SSM runner, keeping the previous-action inputs it was given."""

    def __init__(self, cfg, params, n):
        super().__init__(cfg, params, n)
        self.prev_inputs = []

    def act(self, frames, prev_action, new):
        self.prev_inputs.append(np.array(prev_action, np.float32))
        return super().act(frames, prev_action, new)


def paired(diff, episode):
    """Mean, median, naive SE and episode-clustered SE of per-sample differences."""
    diff = np.asarray(diff, np.float64)
    k = len(diff)
    out = dict(n=k, mean=float(diff.mean()), median=float(np.median(diff)),
               se=float(diff.std(ddof=1) / np.sqrt(k)) if k > 1 else None)
    groups = np.unique(episode)
    if len(groups) > 1:
        resid = np.array([(diff[episode == g] - diff.mean()).sum() for g in groups])
        out["se_episode_clustered"] = float(np.sqrt(len(groups) / (len(groups) - 1) * (resid**2).sum()) / k)
    out["episodes"] = int(len(groups))
    return out


def bin_stats(err, alive, lo, hi, n):
    t_max = alive.shape[0]
    a = alive[lo:min(hi, t_max)]
    e = err[lo:min(hi, t_max)]
    steps = int(a.sum())
    row = dict(t=[lo, hi], alive_steps=steps, distinct_episodes=int(a.any(0).sum()),
               survival_frac_at_start=float(alive[lo].mean()) if lo < t_max else 0.0,
               survival_frac_at_end=float(alive[hi - 1].mean()) if hi - 1 < t_max else 0.0)
    if steps:
        per_ep = [e[:, j][a[:, j]].mean() for j in range(n) if a[:, j].any()]
        row.update(mae=float(e[a].mean()), mae_per_channel=dict(zip(CHANNELS, e[a].mean(0).tolist())),
                   mae_episode_balanced=float(np.mean(per_ep)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/student/ssm/S/ckpt_r03.pkl"))
    ap.add_argument("--config", type=Path, default=Path("runs/student/ssm/S/config.json"))
    ap.add_argument("--data-dirs", nargs="+", default=["/tmp/fpv_student_ssm/S/data", "/tmp/fpv_student_ssm/S_seed2/data"])
    ap.add_argument("--seed", type=int, default=410070)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--late", type=int, default=200)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--full-checks", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("runs/student/ssm/diagnostics/state_r03.json"))
    args = ap.parse_args()
    t_start = time.time()
    run_args = json.loads(args.config.read_text())["args"]
    teacher = run_args["teacher"]
    dagger.check_held_out(args.seed, args.data_dirs)
    cfg, params, _ = st.load_student(args.ckpt)
    assert cfg.n_frames == 1 and cfg.ssm_state == 16 and cfg.ssm_width == 256, cfg
    n = args.episodes

    # 1. Held-out closed-loop evaluation, traced with frames.
    fb = dagger.make_flights(n, teacher, args.seed, None, randomize=True, wide_spawn_p=0.0)
    renderer = dagger.make_renderer("synth", n, cfg.width, cfg.height, seed=args.seed, chunk=64)
    assert not renderer.highlight_target
    runner = RecordingRunner(cfg, params, n)
    metrics, episodes, info = dagger.evaluate(fb, renderer, runner, steps=args.steps,
                                             rng=np.random.default_rng(args.seed), trace=True, log=dagger.log)
    renderer.close()
    traces = info.pop("trace")
    alive = np.stack([s["alive"] for s in traces])
    action = np.stack([s["action"] for s in traces])
    label = np.stack([s["label"] for s in traces])
    frames = [s["frames"] for s in traces]
    prev_in = np.stack(runner.prev_inputs)
    del traces
    t_max = alive.shape[0]
    err = np.abs(action - label)  # applied (clipped) student action vs the teacher label at that state
    outcome = np.array([e["outcome"] for e in sorted(episodes, key=lambda e: e["slot"])])
    length = alive.sum(0)
    # trace[t-1].action is the previous action the runner received at step t
    prev_consistency = float(np.abs(prev_in[1:] - action[:-1])[alive[1:]].max()) if t_max > 1 else None

    result = dict(
        checkpoint=str(args.ckpt), seed=args.seed, episodes=n, steps=args.steps, renderer="synth chunk64 highlight off",
        spawn="narrow (wide_spawn_p=0), randomised physics/layout, random tracks",
        held_out_vs=args.data_dirs, matmul_precision="default",
        eval_summary={k: metrics[k] for k in ("frac_1_lap", "frac_3_laps", "frac_full_episode_no_crash",
                                             "action_mae_vs_teacher") if k in metrics},
        outcomes=dict(running_at_limit=int((outcome == 0).sum()), gate_crash=int((outcome == 1).sum()),
                      ground=int((outcome == 2).sum()), out_of_bounds=int((outcome == 3).sum())),
        episode_length_steps=dict(min=int(length.min()), median=float(np.median(length)), max=int(length.max())),
        prev_action_check_max_abs=prev_consistency,
        error_definition="|applied student action - teacher label| at the same state, alive steps only",
        bins=[bin_stats(err, alive, lo, hi, n) for lo, hi in BINS],
    )
    # Same episodes in every bin: survivors to t=late, so early vs late is not a population change.
    cohort = alive[args.late] if args.late < t_max else np.zeros(n, bool)
    coh = dict(definition=f"episodes alive at t={args.late}", episodes=int(cohort.sum()))
    if cohort.any():
        per = {}
        for lo, hi in BINS:
            a = alive[lo:hi] & cohort
            per[f"{lo}-{hi}"] = np.array([err[lo:hi, j][a[:, j]].mean() for j in np.flatnonzero(cohort)])
            coh[f"mae_{lo}_{hi}"] = float(err[lo:hi][a].mean())
            coh[f"mae_{lo}_{hi}_per_channel"] = dict(zip(CHANNELS, err[lo:hi][a].mean(0).tolist()))
        ids = np.flatnonzero(cohort)
        late = f"{args.late}-1000"
        coh["late_minus_0_50_per_episode"] = paired(per[late] - per["0-50"], ids)
        coh["late_minus_50_200_per_episode"] = paired(per[late] - per["50-200"], ids)
    result["survivor_cohort"] = coh
    crashed = np.flatnonzero(outcome > 0)
    if len(crashed):
        last = np.concatenate([err[max(length[j] - 25, 0):length[j], j] for j in crashed])
        result["crashed_last_25_steps"] = dict(episodes=int(len(crashed)), mae=float(last.mean()),
                                               mae_per_channel=dict(zip(CHANNELS, last.mean(0).tolist())))

    # 2. Zero-state window vs carried memory at the same late states.
    win = args.window
    ts, es = np.nonzero(alive[: t_max] & (np.arange(t_max)[:, None] >= max(args.late, win - 1)))
    probe = dict(window=win, definition=(
        f"zero SSM state, last {win} current frames and applied previous actions (trace[t-1].action), "
        "sequence-mode scan; last-step action vs teacher label, compared with the carried-memory action "
        "actually applied at the same state"), candidates=int(len(ts)))
    if len(ts) == 0:
        probe["result"] = f"no alive states with t >= {args.late}: no evidence either way"
    else:
        rng = np.random.default_rng(args.seed + 1)
        pick = rng.choice(len(ts), min(args.samples, len(ts)), replace=False)
        ts, es = ts[pick], es[pick]
        k, b = len(ts), args.batch
        padded = np.concatenate([np.arange(k), np.full(-k % b, k - 1)])
        net = ssm.StudentSsmNet(cfg)

        @jax.jit
        def window_action(params, fr, pa):
            out, _, _ = net.apply(params, fr.astype(jnp.float32) / 255.0, pa, ssm.empty_state(cfg, fr.shape[0]),
                                  jnp.ones(fr.shape[:2], bool), sequence=True)
            return out[:, -1]

        reset_act = []
        for i in range(0, len(padded), b):
            idx = padded[i : i + b]
            fr = np.stack([np.stack([frames[t][e] for t in range(ts[j] - win + 1, ts[j] + 1)])
                           for j, e in zip(idx, es[idx])])
            pa = np.stack([action[ts[j] - win : ts[j], es[j]] for j in idx])
            reset_act.append(np.asarray(window_action(params, jnp.asarray(fr), jnp.asarray(pa))))
        reset_act = np.concatenate(reset_act)[:k]
        carried = action[ts, es]
        lab = label[ts, es]
        e_reset = np.abs(reset_act - lab).mean(-1)
        e_carry = np.abs(carried - lab).mean(-1)
        diff = e_reset - e_carry
        probe.update(
            samples=k, distinct_episodes=int(len(np.unique(es))), t_median=float(np.median(ts)),
            t_range=[int(ts.min()), int(ts.max())], static_batch=b, padded_calls=int(len(padded) // b),
            mae_carried=float(e_carry.mean()), mae_reset=float(e_reset.mean()),
            mae_carried_per_channel=dict(zip(CHANNELS, np.abs(carried - lab).mean(0).tolist())),
            mae_reset_per_channel=dict(zip(CHANNELS, np.abs(reset_act - lab).mean(0).tolist())),
            reset_minus_carried=paired(diff, es), frac_reset_better=float((diff < 0).mean()),
            frac_tied=float((diff == 0).mean()),
            mean_abs_reset_vs_carried_action=float(np.abs(reset_act - carried).mean()))
        # Numerical floor: the full history from t=0 (zero state, recorded inputs) in sequence mode,
        # left-padded with absent steps, should reproduce the carried action.
        checks = []
        if args.full_checks:
            length_full = args.steps

            @jax.jit
            def full_action(params, fr, pa, present):
                out, _, _ = net.apply(params, fr.astype(jnp.float32) / 255.0, pa, ssm.empty_state(cfg, 1),
                                      present, sequence=True)
                return out[:, -1]

            for j in range(min(args.full_checks, k)):
                t, e = int(ts[j]), int(es[j])
                fr = np.zeros((1, length_full, cfg.height, cfg.width, 3), np.uint8)
                pa = np.zeros((1, length_full, 4), np.float32)
                present = np.zeros((1, length_full), bool)
                fr[0, -(t + 1):] = np.stack([frames[i][e] for i in range(t + 1)])
                pa[0, -(t + 1):] = prev_in[: t + 1, e]
                present[0, -(t + 1):] = True
                full = np.asarray(full_action(params, jnp.asarray(fr), jnp.asarray(pa), jnp.asarray(present)))[0]
                checks.append(float(np.abs(full - carried[j]).max()))
            probe["full_history_sequence_vs_carried_max_abs"] = dict(samples=len(checks), max=max(checks),
                                                                     mean=float(np.mean(checks)))
    probe["limitations"] = [
        "sampled states are survivors at t>=late (selection), stratified by nothing; per-sample SE ignores "
        "within-episode correlation, see se_episode_clustered",
        "carried action was what the drone actually flew, so the trajectory and all inputs were generated by "
        "the carried policy; the reset window is evaluated off-policy on that trajectory (no closed-loop reset)",
        "previous-action inputs are the clean applied actions; training used prev-action noise 0.05 and "
        "photometric augmentation, not applied here",
        "one checkpoint, one seed, synthetic renderer only",
    ]
    result["reset_window_probe"] = probe
    result["wall_s"] = time.time() - t_start
    del frames, action, label, alive, err, prev_in
    gc.collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(args.out)
    dagger.log(json.dumps(result))


if __name__ == "__main__":
    main()
