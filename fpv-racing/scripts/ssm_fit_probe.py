"""Fit probe for one SSM checkpoint: deterministic action/aux error on sampled training-style chunks.

No training, no optimizer state, no augmentation, no previous-action noise. Chunks are drawn exactly as
`fpv.student_ssm.train` draws them (`DiskDataset.sample_sequences`, default per-call weights = eligible
targets, ``sequence_batch`` windows of ``burn_in + sequence_length`` steps), with a fixed diagnostic RNG.
Loss masks are the training masks: padding, burn-in and the last 8 frames before a crash carry no target.

Two sets: the rounds the checkpoint was trained on (DAgger aggregate; NOT held out) and one later round
the checkpoint collected but never trained on (held out for that checkpoint). Only renamed (immutable)
shards are read; ``.tmp`` shards are never loaded. Shard page caches are left alone so concurrently
training arms keep theirs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fpv import dagger_ssm_data as dsd
from fpv import student as st
from fpv.student_ssm import StudentSsmNet, empty_state

dsd.drop_page_cache = lambda *a, **k: None  # read-only probe: do not evict caches of running arms


def sample_with_meta(data, rng, batch, length, burn_in, n_frames):
    """`DiskDataset.sample_sequences` (same RNG use, same outputs) plus by_teacher, round and step index."""
    back = np.arange(burn_in + length - 1, -1, -1)[None]
    counts = data._counts(rng, batch, None)
    out = [[] for _ in range(9)]
    for b, c in zip(data.blocks, counts):
        if c == 0:
            continue
        i = b.valid_idx[rng.integers(0, len(b.valid_idx), c)]
        pos, s = i[:, None] - back, b.seq_start[i][:, None]
        present = pos >= s
        idx = np.maximum(pos, s)
        loss = present & b.eligible[idx]
        loss[:, :burn_in] = False
        frames = b.frames[idx]
        if n_frames == 2:
            frames = np.concatenate([b.frames[np.maximum(idx - 1, s)], frames], -1)
        vals = (frames, b.prev_action[idx], b.label[idx], b.aux[idx], present, loss,
                b.by_teacher[idx].astype(bool), np.full(c, b.round), idx)
        for o, v in zip(out, vals):
            o.append(v)
    return tuple(np.concatenate(o) for o in out)


def make_eval(cfg):
    net = StudentSsmNet(cfg)

    @jax.jit
    def ev(params, frames, prev_action, label, aux, present):
        """Training forward without augmentation / action noise -> per-step errors on the suffix."""
        x = frames.astype(jnp.float32) / 255.0
        burn = cfg.burn_in
        state = empty_state(cfg, x.shape[0])
        if burn:
            _, _, state = net.apply(params, x[:, :burn], prev_action[:, :burn], state, present[:, :burn], sequence=True)
        act, aux_pred, _ = net.apply(params, x[:, burn:], prev_action[:, burn:], state, present[:, burn:], sequence=True)
        lab = label[:, burn:]
        return dict(abs=jnp.abs(act - lab),  # [B, L, 4]
                    huber10=optax.huber_loss(act, lab, delta=0.1).mean(-1) * 10.0,
                    aux_mse=jnp.square(aux_pred - aux[:, burn:]).mean(-1))

    return ev


def ratio_se(num, den):
    """Chunk-level standard error of sum(num)/sum(den) (linearized ratio estimator)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    k = len(num)
    m = num.sum() / den.sum()
    if k < 2:
        return float("nan")
    return float(np.sqrt(np.sum((num - m * den) ** 2) / (k * (k - 1))) / den.mean())


def summarize(per_target: dict, mask: np.ndarray):
    """per_target arrays [C, L]; target-weighted means and chunk-level SE over chunks with >=1 target."""
    n = mask.sum(1)
    keep = n > 0
    out = dict(targets=int(n.sum()), chunks=int(keep.sum()))
    for k, v in per_target.items():
        s = (v * mask).sum(1)
        out[k] = float(s.sum() / max(n.sum(), 1))
        out[k + "_se_chunk"] = ratio_se(s[keep], n[keep])
    return out


def evaluate_set(name, data, cfg, params, ev, seed, n_calls, log):
    rng = np.random.default_rng(seed)
    L, B = cfg.sequence_length, cfg.burn_in
    rows = dict(abs=[], huber10=[], aux_mse=[], mask=[], present=[], by_teacher=[], round=[])
    t0 = time.time()
    for call in range(n_calls):
        if call == 0:  # the mirror sampler must reproduce the training sampler bit for bit
            ref = data.sample_sequences(copy.deepcopy(rng), cfg.sequence_batch, L, B, None, n_frames=cfg.n_frames)
        frames, pa, label, aux, present, mask, byt, rnd, _ = sample_with_meta(
            data, rng, cfg.sequence_batch, L, B, cfg.n_frames)
        if call == 0:
            for a, b in zip(ref, (frames, pa, label, aux, present, mask)):
                assert a.shape == b.shape and np.array_equal(a, b), f"{name}: mirror sampler differs from sample_sequences"
            del ref
        r = jax.device_get(ev(params, jnp.asarray(frames), jnp.asarray(pa, jnp.float32), jnp.asarray(label, jnp.float32),
                              jnp.asarray(aux, jnp.float32), jnp.asarray(present)))
        rows["abs"].append(r["abs"])
        rows["huber10"].append(r["huber10"])
        rows["aux_mse"].append(r["aux_mse"])
        rows["mask"].append(mask[:, B:])
        rows["present"].append(present)
        rows["by_teacher"].append(byt[:, B:])
        rows["round"].append(rnd)
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    mask = cat["mask"].astype(float)
    per = dict(action_mae=cat["abs"].mean(-1), action_huber10=cat["huber10"], aux_mse=cat["aux_mse"])
    res = summarize(per, mask)
    res["compiled_calls"] = n_calls
    res["chunks_sampled"] = int(mask.shape[0])
    res["targets_per_chunk_mean"] = float(mask.sum(1).mean())
    res["targets_per_call_mean"] = float(mask.reshape(n_calls, -1).sum(1).mean())
    res["chunks_with_padding"] = int((~cat["present"]).any(1).sum())
    res["action_mae_per_dim"] = [float((cat["abs"][..., j] * mask).sum() / mask.sum()) for j in range(4)]
    res["action_mae_per_dim_names"] = ["thrust", "roll_rate", "pitch_rate", "yaw_rate"]
    # The training log is a mean over per-batch (per-call) target-weighted means; same statistic here.
    call_mask = mask.reshape(n_calls, -1)
    for k, v in per.items():
        cm = (v.reshape(n_calls, -1) * call_mask).sum(1) / np.maximum(call_mask.sum(1), 1)
        res[k + "_mean_of_call_means"] = float(cm.mean())
        res[k + "_mean_of_call_means_se"] = float(cm.std(ddof=1) / np.sqrt(n_calls))
    by_round = {}
    for rid in np.unique(cat["round"]):
        m = cat["round"] == rid
        by_round[str(int(rid))] = summarize({k: v[m] for k, v in per.items()}, mask[m])
    res["by_round"] = by_round
    bt = cat["by_teacher"]
    res["by_teacher"] = {}
    for flag, label in ((True, "teacher_flew_step"), (False, "student_flew_step")):
        mm = mask * (bt == flag)
        if mm.sum():
            res["by_teacher"][label] = summarize(per, mm)
    res["by_teacher_note"] = ("split per target step by the stored by_teacher flag (which pilot's action was executed "
                              "at that step); chunks can mix both, so these chunk-level SEs use each chunk's subset")
    res["wall_s"] = time.time() - t0
    log(f"{name}: " + json.dumps({k: res[k] for k in ("targets", "chunks", "action_mae", "action_mae_se_chunk",
                                                       "action_huber10", "aux_mse")}))
    return res


def shard_inventory(data_dir: Path, rid: int):
    names = sorted(p.name for p in data_dir.glob(f"r{rid:03d}_c*_*.npz"))
    return dict(immutable=[n for n in names if ".tmp" not in n], tmp_excluded=[n for n in names if ".tmp" in n],
                bytes={n: (data_dir / n).stat().st_size for n in names if ".tmp" not in n},
                mtime={n: (data_dir / n).stat().st_mtime for n in names if ".tmp" not in n})


def round_row(path: Path, rid: int):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["round"] == rid:
            keys = ("round", "beta", "collect_frames", "collect_seed", "data_frames", "data_by_round", "train_steps",
                    "train_mae", "train_loss_act", "train_loss_aux", "train_targets_per_batch")
            out = {k: row.get(k) for k in keys}
            out["eval_action_mae_vs_teacher_closed_loop"] = row.get("eval", {}).get("action_mae_vs_teacher")
            return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/student/ssm/S/ckpt_r03.pkl"))
    ap.add_argument("--run", type=Path, default=Path("runs/student/ssm/S"))
    ap.add_argument("--data-dir", type=Path, default=Path("/tmp/fpv_student_ssm/S/data"))
    ap.add_argument("--fit-rounds", default="0,1,2,3")
    ap.add_argument("--heldout-round", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=256)
    ap.add_argument("--seed", type=int, default=410003)
    ap.add_argument("--baseline-rounds", nargs="*", type=Path,
                    default=[Path("runs/student/ssm/baselines/B/rounds.jsonl"), Path("runs/student/tracks_synth_B/rounds.jsonl")])
    ap.add_argument("--out", type=Path, default=Path("runs/student/ssm/diagnostics/fit_r03.json"))
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    cfg, params, ckpt = st.load_student(args.ckpt)
    assert hasattr(cfg, "ssm_width"), "SSM checkpoint expected"
    ck_round = int(ckpt.get("round", -1))
    fit_rounds = [int(x) for x in args.fit_rounds.split(",")]
    if args.chunks % cfg.sequence_batch:
        raise ValueError("chunks must be a multiple of the checkpoint's sequence_batch")
    n_calls = args.chunks // cfg.sequence_batch
    rounds_path = args.run / "rounds.jsonl"
    s_rows = {r: round_row(rounds_path, r) for r in range(ck_round, args.heldout_round + 1)}

    # Held-out round: require its collection to be complete (round row recorded, no .tmp shard left).
    inv = shard_inventory(args.data_dir, args.heldout_round)
    complete = (s_rows.get(args.heldout_round) is not None and not inv["tmp_excluded"] and len(inv["immutable"]) > 0)

    fit = dsd.DiskDataset().add_dir(args.data_dir, rounds=set(fit_rounds))
    held = dsd.DiskDataset().add_dir(args.data_dir, rounds={args.heldout_round})
    blocks = {}
    for tag, d in (("fit", fit), ("heldout", held)):
        blocks[tag] = [dict(round=b.round, steps=int(b.frames.T), slots=int(b.frames.n), eligible_targets=len(b),
                            valid_frames=len(b) + b.dropped, dropped_before_crash=b.dropped,
                            by_teacher_frac_eligible=float(b.by_teacher[b.valid_idx].mean())) for b in d.blocks]
    log(f"checkpoint round {ck_round}; fit blocks {blocks['fit']}; heldout blocks {blocks['heldout']}; complete {complete}")
    assert sorted(b["round"] for b in blocks["fit"]) == fit_rounds
    assert [b["round"] for b in blocks["heldout"]] == [args.heldout_round]
    rec = s_rows.get(args.heldout_round) or {}
    if complete:  # the loaded held-out call must equal what the run recorded
        assert blocks["heldout"][0]["valid_frames"] == rec["collect_frames"]
        assert blocks["heldout"][0]["eligible_targets"] == rec["data_by_round"][str(args.heldout_round)]
    trained = (s_rows.get(ck_round) or {}).get("data_by_round") or {}
    assert sorted(map(int, trained)) == fit_rounds, f"checkpoint {ck_round} trained on rounds {sorted(trained)}"
    for b in blocks["fit"]:
        assert b["eligible_targets"] == trained[str(b["round"])], "fit data differ from what the checkpoint trained on"

    ev = make_eval(cfg)
    res_fit = evaluate_set("aggregate_fit_rounds", fit, cfg, params, ev, args.seed, n_calls, log)
    res_held = evaluate_set("heldout_round", held, cfg, params, ev, args.seed, n_calls, log)

    baseline = {}
    for p in args.baseline_rounds:
        row = round_row(p, ck_round)
        if row is not None:
            baseline[str(p)] = row
    result = dict(
        checkpoint=str(args.ckpt), checkpoint_round=ck_round,
        checkpoint_sha256=hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        config=dict(sequence_length=cfg.sequence_length, burn_in=cfg.burn_in, sequence_batch=cfg.sequence_batch,
                    n_frames=cfg.n_frames, ssm_width=cfg.ssm_width, ssm_state=cfg.ssm_state, ssm_layers=cfg.ssm_layers,
                    augment_in_probe=False, prev_action_noise_in_probe=0.0, prev_action_drop_in_probe=0.0,
                    training_augment=cfg.augment, training_prev_action_noise=cfg.prev_action_noise,
                    drop_before_crash=8, block_weights="eligible targets per call (= training, recent_weight 1, no extra)",
                    rng="numpy default_rng(seed) fresh for EACH set", seed=args.seed, chunks=args.chunks,
                    compiled_calls_per_set=n_calls,
                    matmul_precision=str(getattr(jax.config, "jax_default_matmul_precision", None)),
                    device=str(jax.devices()[0])),
        argv=sys.argv, env={k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_DEFAULT_MATMUL_PRECISION", "CUDA_CACHE_PATH", "XDG_CACHE_HOME",
            "JAX_COMPILATION_CACHE_DIR")},
        data_dir=str(args.data_dir), blocks=blocks,
        heldout_round=args.heldout_round, heldout_complete=complete, heldout_shards=inv,
        fit_rounds_shards={r: shard_inventory(args.data_dir, r)["immutable"] for r in fit_rounds},
        aggregate_fit_rounds=res_fit, heldout=res_held,
        recorded_train_metrics=dict(
            S=s_rows, B=baseline,
            note=("rounds.jsonl train_* = mean over the last 500 of the round's optimizer steps of per-batch "
                  "target-weighted means, with augmentation and previous-action noise, while weights change; "
                  "B is the 5-frame CNN on 256 independent frames per batch")),
        notes=[
            f"aggregate_fit_rounds samples rounds {fit_rounds}, ALL of which checkpoint {ck_round} trained on "
            "(DAgger aggregate): it measures fit, not generalization.",
            f"round {args.heldout_round} was collected by checkpoint {ck_round} itself (beta "
            f"{rec.get('beta')}) and never trained on by it: held out for this checkpoint only.",
            "collection uses wide_spawn_p 0.3, teacher noise/bursts and drops the last 8 frames before each crash; "
            "closed-loop evaluation uses narrow spawns and scores every state, so neither set is the eval distribution.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
