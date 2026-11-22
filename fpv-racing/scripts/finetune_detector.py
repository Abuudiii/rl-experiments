"""Fine-tune the gate detector on Unreal frames (stage 6), starting from a synthetic-trained checkpoint.

    python scripts/finetune_detector.py --init runs/percep/detector/E_amd/ckpt_best.pkl --out runs/percep/stage6/det_ft/<name> \
        --ue-dirs <shard dir>,<shard dir> --ue-val <held-out shard dir> --steps 20000

Every batch mixes ``--ue-frac`` Unreal frames (shards of `scripts/stage6_unreal.py frames`: teacher-flown poses rendered
in Unreal, labels from the same `labels.gate_labels` of the known gate poses and camera, gates >= 70 % hidden behind a
nearer gate don't-care as in stage 1) with synthetic frames made exactly as stage 1 trains (`train_detector.py`: pose
bank, half perturbed, random look from `labels.scene_pool`, ray caster). Unreal frames get a mild photometric jitter
(gain, per-channel gain, offset, pixel noise). Same loss, targets and architecture as stage 1; new optimizer state,
warm-up + cosine schedule from ``--lr``. Validation every ``--val-every`` steps on held-out Unreal frames (and the
stage-1 synthetic validation set, to see that synthetic quality holds); best checkpoint by the stage-1 score
(err_mean + 20 miss + 5 FP/frame) on the Unreal validation set -> <out>/ckpt_best.pkl.
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
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from fpv.perception import detector as D  # noqa: E402
from fpv.perception import labels as L  # noqa: E402
from stage6_unreal import hidden_frac, load_shards  # noqa: E402
from train_detector import TEACHER, bank_path, load_bank, log, render_set, summarize  # noqa: E402

KEYS = ("pos", "quat", "gates_pos", "gates_yaw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/percep/detector/E_amd/ckpt_best.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ue-dirs", required=True)
    ap.add_argument("--ue-val", required=True)
    ap.add_argument("--ue-val-max", type=int, default=4096)
    ap.add_argument("--ue-frac", type=float, default=0.5)
    ap.add_argument("--bank-dir", default="runs/percep/detector/banks")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb-p", type=float, default=0.5)
    ap.add_argument("--hidden-dontcare", type=float, default=0.7)
    ap.add_argument("--scene-pool", type=int, default=4096)
    ap.add_argument("--scene-refresh", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--stop-at", type=float, default=0.0)
    ap.add_argument("--synth-pose-dirs", default="", help="shard dirs whose poses join the synthetic bank")
    ap.add_argument("--synth-extra-frac", type=float, default=0.3, help="share of the synthetic part drawn from them")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det0 = D.load_detector(args.init)
    cfg, net, params = det0.cfg, det0.net, det0.params
    (out / "config.json").write_text(json.dumps(dict(args=vars(args), devices=[str(d) for d in jax.devices()]), indent=1))

    # Unreal frames (host memory) + labels' don't-care from the synthetic hit map of the same poses
    ue = load_shards(args.ue_dirs.split(","))
    n_ue = len(ue["pos"])
    log(f"Unreal training frames: {n_ue} from {args.ue_dirs}")
    hid_ue = hidden_frac(ue) >= args.hidden_dontcare
    uv = load_shards(args.ue_val.split(","))
    if len(uv["pos"]) > args.ue_val_max:
        sel = np.random.default_rng(0).choice(len(uv["pos"]), args.ue_val_max, replace=False)
        uv = {k: v[np.sort(sel)] for k, v in uv.items()}
    uv_lab = L.gate_labels(*(uv[k].astype(np.float64) for k in KEYS))
    log(f"Unreal validation frames: {len(uv['pos'])}")

    bank = load_bank(bank_path(args.bank_dir, "random", 1000, 2048, 400), teacher_ckpt=TEACHER, n_envs=2048, steps=400,
                     seed=1000, wide_spawn_p=0.5, teacher_noise=0.3)  # fmt: skip
    vbank = load_bank(bank_path(args.bank_dir, "random", 1500, 32, 300), teacher_ckpt=TEACHER, n_envs=32, steps=300,
                      seed=1500, wide_spawn_p=0.3, teacher_noise=0.2)  # fmt: skip
    render = L.make_renderer()
    vidx = np.random.default_rng(1500).choice(len(vbank["pos"]), 2048, replace=False)
    vframes, vlab = render_set(render, vbank, vidx, L.scene_pool(512, 1507), 1500)
    m = len(bank["pos"])
    ma = 0
    if args.synth_pose_dirs:  # extra synthetic poses (e.g. the acrobatic track's), a fixed share of the synthetic part
        ex = [np.load(p) for d in args.synth_pose_dirs.split(",") for p in sorted(Path(d).glob("shard_*.npz"))]
        extra = {k: np.concatenate([z[k] for z in ex]).astype(bank[k].dtype) for k in KEYS}
        ma = len(extra["pos"])
        bank = {k: np.concatenate([bank[k], extra[k]]) for k in KEYS}
        log(f"synthetic extra poses: {ma} from {args.synth_pose_dirs}, share {args.synth_extra_frac}")
    dev_bank = {k: jnp.asarray(bank[k]) for k in KEYS}

    B = args.batch
    Bu = int(round(B * args.ue_frac))
    Bs = B - Bu
    opt = D.make_optimizer(args.steps, args.lr, warmup=args.warmup)
    opt_state = opt.init(params)

    def jitter(k, img):
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)
        n = img.shape[0]
        x = img.astype(jnp.float32)
        g = jax.random.uniform(k1, (n, 1, 1, 1), minval=0.8, maxval=1.2)
        gc = jax.random.uniform(k2, (n, 1, 1, 3), minval=0.95, maxval=1.05)
        b = jax.random.uniform(k3, (n, 1, 1, 1), minval=-15, maxval=15)
        s = jax.random.uniform(k4, (n, 1, 1, 1), minval=0, maxval=4)
        x = x * g * gc + b + s * jax.random.normal(k5, x.shape)
        return jnp.clip(jnp.round(x), 0, 255).astype(jnp.uint8)

    @jax.jit
    def train_step(params, opt_state, key, scenes, u_img, u_pose, u_hid):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        parts_img, parts_lab = [], []
        if Bs:
            idx = jax.random.randint(k1, (Bs,), 0, m)
            if ma:
                Ba = int(round(Bs * args.synth_extra_frac))
                idx = jnp.where(jnp.arange(Bs) < Ba, m + jax.random.randint(jax.random.fold_in(k1, 1), (Bs,), 0, ma), idx)
            pos, quat = L.perturb_poses(k2, dev_bank["pos"][idx], dev_bank["quat"][idx], p=args.perturb_p)
            gp, gy = dev_bank["gates_pos"][idx], dev_bank["gates_yaw"][idx]
            si = jax.random.randint(k3, (Bs,), 0, args.scene_pool)
            img, ids = render(pos, quat, gp, gy, jax.tree.map(lambda a: a[si], scenes), k4)
            lab = L.gate_labels(pos, quat, gp, gy)
            hid = L.hidden_fraction(pos, quat, gp, gy, ids, lab["side_px"]) >= args.hidden_dontcare
            parts_img.append(img)
            parts_lab.append(dict(lab, visible=lab["visible"] & ~hid, dontcare=lab["dontcare"] | (lab["visible"] & hid)))
        if Bu:
            lab = L.gate_labels(*u_pose)
            parts_img.append(jitter(k5, u_img))
            parts_lab.append(dict(lab, visible=lab["visible"] & ~u_hid, dontcare=lab["dontcare"] | (lab["visible"] & u_hid)))
        img = jnp.concatenate(parts_img)
        lab = {k: jnp.concatenate([p[k] for p in parts_lab]) for k in parts_lab[0]}
        tgt = D.make_targets(cfg, lab)

        def lf(p):
            return D.loss_fn(cfg, net.apply(p, img), tgt)

        (_, aux), grads = jax.value_and_grad(lf, has_aux=True)(params)
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, aux

    fwd = jax.jit(lambda p, f: D.decode(cfg, net.apply(p, f), snap=det0.snap))

    def run(frames):
        dets = [jax.device_get(fwd(params, jnp.asarray(frames[i : i + 256]))) for i in range(0, len(frames), 256)]
        return {k: np.concatenate([d[k] for d in dets]) for k in ("corners", "conf")}

    def validate():
        return dict(unreal=summarize(run(uv["frames"]), uv_lab, args.thresh), synth=summarize(run(vframes), vlab, args.thresh))

    def save(path, step, extra):
        D.save_checkpoint(path, cfg, params, dict(step=step, init=args.init, train_args=vars(args), **extra))

    v0 = validate()
    log("VAL init: " + json.dumps({s: {k: round(x, 4) for k, x in v.items()} for s, v in v0.items()}))
    flog = open(out / "log.jsonl", "a")
    flog.write(json.dumps(dict(step=0, val=v0)) + "\n")
    best = v0["unreal"]["err_mean"] + 20 * v0["unreal"]["miss"] + 5 * v0["unreal"]["fp_per_frame"]
    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed + 17)
    scenes = None
    t0 = tl = time.time()
    agg = []
    step = 0
    for step in range(args.steps):
        if scenes is None or step % args.scene_refresh == 0:
            scenes = {k: jnp.asarray(v) for k, v in L.scene_pool(args.scene_pool, 77_000_003 + step).items()}
        j = np.sort(rng.integers(0, n_ue, Bu))
        u_pose = tuple(jnp.asarray(ue[k][j].astype(np.float32)) for k in KEYS)
        key, k = jax.random.split(key)
        params, opt_state, aux = train_step(params, opt_state, k, scenes, jnp.asarray(ue["frames"][j]), u_pose,
                                            jnp.asarray(hid_ue[j]))  # fmt: skip
        agg.append(aux)
        if (step + 1) % args.log_every == 0:
            a = {k_: float(np.mean([float(x[k_]) for x in agg])) for k_ in agg[0]}
            agg = []
            rate = args.log_every / (time.time() - tl)
            tl = time.time()
            log(f"step {step + 1}  " + "  ".join(f"{k_} {v:.4f}" for k_, v in a.items()) + f"  it/s {rate:.2f}")
            flog.write(json.dumps(dict(step=step + 1, t=time.time() - t0, it_s=rate, **a)) + "\n")
            flog.flush()
        last = step + 1 == args.steps
        stop = args.stop_at and time.time() > args.stop_at
        if (step + 1) % args.val_every == 0 or last or stop:
            v = validate()
            log(f"VAL step {step + 1}: " + json.dumps({s: {k_: round(x, 4) for k_, x in vv.items()} for s, vv in v.items()}))
            flog.write(json.dumps(dict(step=step + 1, val=v)) + "\n")
            flog.flush()
            score = v["unreal"]["err_mean"] + 20 * v["unreal"]["miss"] + 5 * v["unreal"]["fp_per_frame"]
            if score < best:
                best = score
                save(out / "ckpt_best.pkl", step + 1, dict(val=v, best=best))
            save(out / "ckpt.pkl", step + 1, dict(val=v, best=best))
        if stop:
            log("stop-at reached")
            break
    log(f"done at step {step + 1}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
