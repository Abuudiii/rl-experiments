"""Train the gate detector (`fpv.perception.detector`) on synthetic frames rendered on the fly.

    python scripts/train_detector.py --name <run> --steps 60000 [--batch 64] [--set stride=2 snap=false ...]

Data, all generated on the device the script runs on (no frames stored):
  * pose bank: teacher (`--teacher`) flights on random tracks from `fpv.track.sample_tracks` with
    stick noise (std per episode U(0, 0.3), bursts p 0.01 / +-0.3) and wide / off-course starts
    (p 0.5), collected once with `labels.collect_flights` and cached as an npz under ``--bank-dir``
    (seed ``--bank-seed``; the evaluation uses other seeds and the six named layouts);
  * every batch: poses drawn from the bank, half of them perturbed (roll/pitch +-0.35 rad, yaw
    +-0.5 rad, +-0.4 m per axis; `labels.perturb_poses`), a random look from a pool of renderer
    looks (`labels.scene_pool`, redrawn every ``--scene-refresh`` steps), rendered by the
    `fpv.render_synth` ray caster (320x180, 2x2 supersampling, highlight off, per-frame pixel noise);
    labels from `labels.gate_labels`.
Validation every ``--val-every`` steps on a fixed held-out set (``--val-frames`` frames of flights on
random tracks from another seed, rendered once). Output: runs/percep/detector/<name>/{ckpt.pkl,
ckpt_best.pkl, log.jsonl, config.json}.
"""

from __future__ import annotations

import argparse
import dataclasses
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
import optax  # noqa: E402

from fpv.perception import detector as D  # noqa: E402
from fpv.perception import labels as L  # noqa: E402

TEACHER = "runs/teacher/rt_G_D_noise3/ckpt_latest.pkl"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_bank(path: Path, **kw) -> dict:
    """The cached pose bank at ``path`` (npz), collected with `labels.collect_flights(**kw)` if missing."""
    if not path.exists():
        log(f"collecting pose bank {path.name}: {kw}")
        t0 = time.time()
        b = L.collect_flights(**kw, log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **b, meta=json.dumps({k: v for k, v in kw.items() if k != "log"}))
        tmp.rename(path)
        log(f"  {len(b['pos'])} frames in {time.time() - t0:.0f} s")
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "meta"}


def bank_path(bank_dir, tag, seed, n_envs, steps):
    return Path(bank_dir) / f"bank_{tag}_s{seed}_n{n_envs}_t{steps}.npz"


def render_set(render, bank, idx, scenes, seed, chunk=64):
    """Render bank poses idx with scenes[i % len] (no perturbation) -> frames uint8 (numpy), labels (numpy)."""
    key = jax.random.key(seed)
    frames = []
    ns = len(scenes["tgt"])
    for i in range(0, len(idx), chunk):
        j = idx[i : i + chunk]
        m = len(j)
        jp = np.r_[j, np.full(chunk - m, j[-1])]
        si = np.arange(i, i + chunk) % ns
        key, k = jax.random.split(key)
        img, _ = render(jnp.asarray(bank["pos"][jp]), jnp.asarray(bank["quat"][jp]), jnp.asarray(bank["gates_pos"][jp]),
                        jnp.asarray(bank["gates_yaw"][jp]), {a: jnp.asarray(v[si]) for a, v in scenes.items()}, k)  # fmt: skip
        frames.append(np.asarray(img[:m]))
    frames = np.concatenate(frames)
    lab = L.gate_labels(bank["pos"][idx].astype(np.float64), bank["quat"][idx].astype(np.float64),
                        bank["gates_pos"][idx].astype(np.float64), bank["gates_yaw"][idx].astype(np.float64))  # fmt: skip
    return frames, lab


def summarize(det: dict, lab: dict, thresh: float) -> dict:
    mt = D.match(det["corners"], det["conf"], lab, thresh)
    vis = lab["visible"]
    err = mt["err"][vis]
    found = np.isfinite(err)
    e = err[found]
    out = dict(n_visible=int(vis.sum()), miss=float(1 - found.mean()) if vis.any() else float("nan"),
               fp_per_frame=float(mt["fp"].sum() / len(vis)),
               err_mean=float(e.mean()) if len(e) else float("nan"),
               err_median=float(np.median(e)) if len(e) else float("nan"),
               err_p90=float(np.percentile(e, 90)) if len(e) else float("nan"),
               err_inv_mean=float(np.nanmean(mt["err_inv"][vis])) if len(e) else float("nan"))  # fmt: skip
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 12)):
        b = vis & (lab["dist"] >= lo) & (lab["dist"] < hi)
        eb = mt["err"][b]
        out[f"miss_{lo}_{hi}"] = float(np.isnan(eb).mean()) if b.any() else float("nan")
        out[f"err_med_{lo}_{hi}"] = float(np.nanmedian(eb)) if np.isfinite(eb).any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="runs/percep/detector")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--bank-dir", default="/tmp/fpv_percep/PerceptDetector")
    ap.add_argument("--bank-seed", type=int, default=1000)
    ap.add_argument("--bank-envs", type=int, default=512)
    ap.add_argument("--bank-steps", type=int, default=1500)
    ap.add_argument("--val-seed", type=int, default=1500)
    ap.add_argument("--val-frames", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb-p", type=float, default=0.5)
    ap.add_argument("--hidden-dontcare", type=float, default=0.7,
                    help="train: visible gates with at least this fraction of the frame hidden by nearer gates become "
                    "don't-care (> 1 disables; the evaluation keeps the contract labels)")  # fmt: skip
    ap.add_argument("--scene-pool", type=int, default=4096)
    ap.add_argument("--scene-refresh", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--stop-at", type=float, default=0.0, help="unix time: save and stop before it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="DetectorConfig overrides key=value")
    args = ap.parse_args()
    os.chdir(ROOT)

    over = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        f = {f.name: f for f in dataclasses.fields(D.DetectorConfig)}[k]
        if f.type in ("tuple",) or isinstance(f.default, tuple):
            over[k] = tuple(int(x) for x in v.split(","))
        elif isinstance(f.default, bool):
            over[k] = v.lower() in ("1", "true", "yes")
        else:
            over[k] = type(f.default)(v)
    cfg = D.DetectorConfig(**over)
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(args=vars(args), detector=dataclasses.asdict(cfg),
                                                     devices=[str(d) for d in jax.devices()]), indent=1))  # fmt: skip
    log(f"devices {jax.devices()}  config {cfg}")

    bank = load_bank(bank_path(args.bank_dir, "random", args.bank_seed, args.bank_envs, args.bank_steps),
                     teacher_ckpt=args.teacher, n_envs=args.bank_envs, steps=args.bank_steps, seed=args.bank_seed,
                     wide_spawn_p=0.5, teacher_noise=0.3)  # fmt: skip
    vbank = load_bank(bank_path(args.bank_dir, "random", args.val_seed, 32, 300), teacher_ckpt=args.teacher,
                      n_envs=32, steps=300, seed=args.val_seed, wide_spawn_p=0.3, teacher_noise=0.2)  # fmt: skip
    m = len(bank["pos"])
    log(f"pose bank: {m} frames, {len(np.unique(bank['episode']))} episodes; val bank {len(vbank['pos'])}")

    render = L.make_renderer()
    vidx = np.random.default_rng(args.val_seed).choice(len(vbank["pos"]), min(args.val_frames, len(vbank["pos"])),
                                                         replace=False)  # fmt: skip
    vframes, vlab = render_set(render, vbank, vidx, L.scene_pool(512, args.val_seed + 7), args.val_seed)

    dev_bank = {k: jnp.asarray(bank[k]) for k in ("pos", "quat", "gates_pos", "gates_yaw")}
    net, params = D.init_params(cfg, args.seed)
    log(f"parameters: {D.n_params(params):,}")
    opt = D.make_optimizer(args.steps, args.lr)
    opt_state = opt.init(params)
    step0, best = 0, float("inf")
    if args.resume and (out / "ckpt.pkl").exists():
        import pickle

        ck = pickle.loads((out / "ckpt.pkl").read_bytes())
        params = jax.tree.map(jnp.asarray, ck["params"])
        opt_state = jax.tree.map(jnp.asarray, ck["extra"]["opt_state"]) if "opt_state" in ck["extra"] else opt.init(params)
        step0, best = ck["extra"]["step"], ck["extra"].get("best", best)
        log(f"resumed at step {step0}")
    B = args.batch

    @jax.jit
    def train_step(params, opt_state, key, scenes):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (B,), 0, m)
        pos, quat = L.perturb_poses(k2, dev_bank["pos"][idx], dev_bank["quat"][idx], p=args.perturb_p)
        gp, gy = dev_bank["gates_pos"][idx], dev_bank["gates_yaw"][idx]
        si = jax.random.randint(k3, (B,), 0, args.scene_pool)
        img, ids = render(pos, quat, gp, gy, jax.tree.map(lambda a: a[si], scenes), k4)
        lab = L.gate_labels(pos, quat, gp, gy)
        if args.hidden_dontcare <= 1:
            hid = L.hidden_fraction(pos, quat, gp, gy, ids, lab["side_px"]) >= args.hidden_dontcare
            lab = dict(lab, visible=lab["visible"] & ~hid, dontcare=lab["dontcare"] | (lab["visible"] & hid))
        tgt = D.make_targets(cfg, lab)

        def lf(p):
            return D.loss_fn(cfg, net.apply(p, img), tgt)

        (_, aux), grads = jax.value_and_grad(lf, has_aux=True)(params)
        upd, opt_state = opt.update(grads, opt_state, params)
        aux["n_vis"] = tgt["reg_mask"].sum() / B
        return optax.apply_updates(params, upd), opt_state, aux

    fwd = jax.jit(lambda p, f: D.decode(cfg, net.apply(p, f)))

    def validate():
        dets = [jax.device_get(fwd(params, jnp.asarray(vframes[i : i + 256]))) for i in range(0, len(vframes), 256)]
        det = {k: np.concatenate([d[k] for d in dets]) for k in ("corners", "conf")}
        return summarize(det, vlab, args.thresh)

    def save(path, step, extra=None):
        D.save_checkpoint(path, cfg, params, dict(step=step, best=best, train_args=vars(args), **(extra or {})))

    key = jax.random.key(args.seed + 17)
    for _ in range(step0):  # resume exactly: replay the per-step key splits of the steps already done
        key, _ = jax.random.split(key)
    scenes = None
    t0, tl, agg = time.time(), time.time(), []
    flog = open(out / "log.jsonl", "a")
    step = step0
    for step in range(step0, args.steps):
        if scenes is None or step % args.scene_refresh == 0:
            scenes = {k: jnp.asarray(v) for k, v in L.scene_pool(args.scene_pool, args.seed * 100003 + step).items()}
        key, k = jax.random.split(key)
        params, opt_state, aux = train_step(params, opt_state, k, scenes)
        agg.append(aux)
        if (step + 1) % args.log_every == 0:
            a = {k: float(np.mean([float(x[k]) for x in agg])) for k in agg[0]}
            agg = []
            rate = args.log_every / (time.time() - tl)
            tl = time.time()
            log(f"step {step + 1}  " + "  ".join(f"{k} {v:.4f}" for k, v in a.items()) + f"  it/s {rate:.2f}")
            flog.write(json.dumps(dict(step=step + 1, t=time.time() - t0, it_s=rate, **a)) + "\n")
            flog.flush()
        last = step + 1 == args.steps
        stop = args.stop_at and time.time() > args.stop_at
        if (step + 1) % args.val_every == 0 or last or stop:
            v = validate()
            log(f"VAL step {step + 1}: " + json.dumps({k: round(x, 4) for k, x in v.items()}))
            flog.write(json.dumps(dict(step=step + 1, val=v)) + "\n")
            flog.flush()
            score = v["err_mean"] + 20 * v["miss"] + 5 * v["fp_per_frame"]
            if score < best:
                best = score
                save(out / "ckpt_best.pkl", step + 1, dict(val=v))
            save(out / "ckpt.pkl", step + 1, dict(val=v, opt_state=jax.tree.map(np.asarray, opt_state)))
        if stop:
            log("stop-at reached")
            break
    log(f"done at step {step + 1}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
