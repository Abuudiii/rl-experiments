"""Renderer-agnostic closed-loop flying, DAgger data shards, training and evaluation of the student.

`fly` is the one closed loop used for data collection and evaluation: Crazyflow (`FlightBatch`)
steps the dynamics, the renderer turns each pose into a frame, the student (`StudentRunner`) and /
or the teacher (the label of the current state) choose the action. Every visited state can be
written to shards together with its teacher label.

Shard = ``<data>/r<round>_c<call>_<seq>.npz`` (uncompressed), arrays time-major [T, n, ...] for T
consecutive steps of the n parallel slots of one `fly` call:
    frames uint8 [T,n,H,W,3], label (teacher action for this state), action (executed), prev_action,
    teacher_obs [43] (privileged; auxiliary targets), pos, quat, vel, gates_pos, gates_yaw, target,
    t (step in the episode; 0 = first frame), ep (episode counter of the slot), by_teacher (who
    flew this step), valid (False once a slot's episode is over and it is not reset).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import jax
import numpy as np

from fpv import student as st

OUTCOMES = ("time_limit", "gate_crash", "ground", "out_of_bounds")
REC_FIELDS = ("frames", "label", "action", "prev_action", "teacher_obs", "pos", "quat", "vel", "gates_pos",
              "gates_yaw", "target", "t", "ep", "by_teacher", "valid")  # fmt: skip


def make_renderer(kind: str, n: int, width: int, height: int, seed: int = 0, **kw):
    if kind == "synth":
        from fpv.render_synth import SynthRenderer

        return SynthRenderer(n, width, height, seed=seed, **kw)
    if kind == "airsim":
        from fpv.render_airsim import AirsimRenderer

        return AirsimRenderer(width=width, height=height, **kw)  # n = len(gpus) * instances_per_gpu
    raise ValueError(kind)


# region shards
class ShardWriter:
    def __init__(self, data_dir, round_id: int):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        calls = {int(p.name.split("_")[1][1:]) for p in self.dir.glob(f"r{round_id:03d}_c*_*.npz")}
        self.prefix = f"r{round_id:03d}_c{max(calls, default=-1) + 1:02d}"
        self.seq, self.frames = 0, 0

    def write(self, rows: list[dict]):
        arrs = {k: np.stack([r[k] for r in rows]) for k in REC_FIELDS}
        tmp = self.dir / f"{self.prefix}_{self.seq:04d}.tmp.npz"
        np.savez(tmp, **arrs)
        drop_page_cache(tmp, sync=True)
        tmp.rename(self.dir / f"{self.prefix}_{self.seq:04d}.npz")
        self.seq += 1
        self.frames += int(arrs["valid"].sum())


def drop_page_cache(path, sync: bool = False):
    """Tell the kernel not to keep this file cached: tens of GB of shards in the page cache exhaust
    the NUMA node the job runs on and make every fresh allocation crawl (seen: 70 s per GB)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class Block:
    """One `fly` call, env-major and flattened: frame i belongs to the episode starting at start[i].

    The last ``drop_before_crash`` frames before a crash are not sampled as training targets (the
    teacher cannot save those states either; they still serve as frame history).
    """

    def __init__(self, paths, round_id: int, drop_before_crash: int = 8, max_steps: int = 1500):
        parts = [dict(np.load(p)) for p in sorted(paths)]
        for p in paths:
            drop_page_cache(p)
        cat = lambda k: np.concatenate([p[k] for p in parts], 0)  # noqa: E731
        t, valid = cat("t"), cat("valid")
        T, n = t.shape
        flat = lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1)).reshape(T * n, *a.shape[2:])  # noqa: E731
        self.frames = flat(cat("frames"))
        self.label, self.prev_action = flat(cat("label")), flat(cat("prev_action"))
        self.aux = flat(cat("teacher_obs"))[:, st.AUX_IDX]
        self.by_teacher = flat(cat("by_teacher"))
        first = (t == 0) | (np.arange(T)[:, None] == 0)  # an episode start, or the start of the recording
        idx = np.where(first, np.arange(T)[:, None], 0)
        start_t = np.maximum.accumulate(idx, axis=0)  # [T, n] time index of the episode's first frame
        self.start = (np.swapaxes(start_t, 0, 1) + np.arange(n)[:, None] * T).reshape(-1)
        # index of the next episode boundary (a reset or the end of validity) after each step
        bound = np.where(first | ~valid, np.arange(T)[:, None], 10 * T)
        bound[0] = np.where(valid[0], 10 * T, 0)
        nxt = np.minimum.accumulate(np.concatenate([bound[1:], np.full((1, n), 10 * T)])[::-1], axis=0)[::-1]
        to_end = nxt - np.arange(T)[:, None]  # 1 = last frame of its episode
        crash_soon = (to_end <= drop_before_crash) & (t + to_end - 1 < max_steps - 1)
        self.dropped = int((valid & crash_soon).sum())
        self.valid_idx = np.flatnonzero(flat(valid & ~crash_soon))
        self.round = round_id

    def __len__(self):
        return len(self.valid_idx)


class Dataset:
    def __init__(self):
        self.blocks: list[Block] = []
        self.loaded: set[str] = set()

    def add_dir(self, data_dir, rounds=None, extra: bool = False):
        """Load every not yet loaded `fly` call found in data_dir (optionally only some rounds).
        ``extra`` marks data of another run / renderer (see `train`'s ``extra_frac``)."""
        groups: dict[tuple[str, str], list[Path]] = {}
        for p in sorted(Path(data_dir).glob("r*_c*_*.npz")):
            if ".tmp" in p.name:
                continue
            r, c, _ = p.name.split("_")
            groups.setdefault((str(Path(data_dir).resolve()), f"{r}_{c}"), []).append(p)
        for (d, key), paths in sorted(groups.items()):
            rid = int(key[1:4])
            if f"{d}/{key}" in self.loaded or (rounds is not None and rid not in rounds):
                continue
            self.blocks.append(Block(paths, rid))
            self.blocks[-1].extra = extra
            self.loaded.add(f"{d}/{key}")
        return self

    def __len__(self):
        return sum(len(b) for b in self.blocks if not b.extra)

    def sizes(self) -> dict:
        """Frames per round (own data); extra data under the key "extra"."""
        out: dict = {}
        for b in self.blocks:
            k = "extra" if b.extra else b.round
            out[k] = out.get(k, 0) + len(b)
        return out

    def sample(self, rng, batch: int, n_frames: int, weights=None):
        """-> frames uint8 [B, K, H, W, 3] (oldest first), prev_action, label, aux."""
        w = np.array([len(b) for b in self.blocks], float) if weights is None else np.asarray(weights, float)
        counts = rng.multinomial(batch, w / w.sum())
        out = [[], [], [], []]
        for b, c in zip(self.blocks, counts):
            if c == 0:
                continue
            i = b.valid_idx[rng.integers(0, len(b.valid_idx), c)]
            idx = np.maximum(i[:, None] - np.arange(n_frames - 1, -1, -1)[None], b.start[i][:, None])
            for o, v in zip(out, (b.frames[idx], b.prev_action[i], b.label[i], b.aux[i])):
                o.append(v)
        return tuple(np.concatenate(o) for o in out)


# region training
def train(cfg: st.StudentConfig, params, data: Dataset, steps: int, seed: int = 0, lr: float | None = None,
          log=print, log_every: int = 500, recent_weight: float = 1.0, extra_frac: float = 0.25, workers: int = 4):  # fmt: skip
    """Train / fine-tune ``params`` on the aggregated data for ``steps`` batches. recent_weight > 1
    samples frames of the newest round that many times more often than the others; data marked
    ``extra`` (another renderer) make up ``extra_frac`` of every batch."""
    opt, step, _ = st.make_train_step(cfg, steps, lr=lr)
    opt_state = opt.init(params)
    own = [b for b in data.blocks if not b.extra]
    last = max(b.round for b in own)
    weights = np.array([0.0 if b.extra else len(b) * (recent_weight if b.round == last else 1.0) for b in data.blocks])
    n_extra = np.array([len(b) if b.extra else 0.0 for b in data.blocks])
    if n_extra.sum() > 0:
        weights = (1 - extra_frac) * weights / weights.sum() + extra_frac * n_extra / n_extra.sum()
    q: queue.Queue = queue.Queue(maxsize=3 * workers)
    stop = threading.Event()

    def work(i):
        rng = np.random.default_rng([seed, i])
        while not stop.is_set():
            item = data.sample(rng, cfg.batch, cfg.n_frames, weights)
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    key = jax.random.key(seed)
    t0, acc, hist = time.time(), [], []
    for i in range(steps):
        frames, pa, label, aux = q.get()
        key, k = jax.random.split(key)
        params, opt_state, m = step(params, opt_state, frames, pa, label, aux, k)
        acc.append(m)
        if (i + 1) % log_every == 0 or i + 1 == steps:
            mm = {k: float(np.mean([np.asarray(a[k]) for a in acc])) for k in acc[0]}
            acc = []
            hist.append(dict(step=i + 1, wall_s=time.time() - t0, **mm))
            log(f"  train {i + 1}/{steps} loss {mm['loss']:.4f} act {mm['act']:.4f} aux {mm['aux']:.4f} "
                f"mae {mm['mae']:.4f} ({(i + 1) / (time.time() - t0):.1f} it/s)")  # fmt: skip
    stop.set()
    return params, hist


# region closed loop
def fly(fb, renderer, runner=None, *, steps: int, beta: float = 0.0, reset_done: bool = True, rng=None,
        teacher_noise: float = 0.0, burst_p: float = 0.0, burst_mag: float = 0.3, burst_len=(3, 12),
        seg_len=(15, 60), writer: ShardWriter | None = None, shard_steps: int = 100, trace: bool = False,
        render: bool = True, defer_render: bool = False, log=None):  # fmt: skip
    """Fly ``fb.n`` parallel episodes for up to ``steps`` policy steps.

    Who flies: ``runner=None`` -> the teacher; else per slot in segments of ``seg_len`` steps the
    teacher with probability ``beta``, otherwise the student. Teacher-flown steps get Gaussian
    action noise (std per episode uniform in [0, teacher_noise]) and, with probability ``burst_p``
    per step, a burst of ``burst_len`` steps with a constant random offset (uniform in
    +/- ``burst_mag``) on all four channels.
    ``reset_done``: finished slots start a new episode (collection); otherwise they stay finished
    and the loop ends when all are (evaluation; the renderer gets ``active=~done``).
    ``defer_render`` (teacher-flown collection only): fly first without rendering, then render all
    recorded poses with the renderer's open-loop ``render_trajectories``.

    Returns (episodes, info): per finished episode ``gates, outcome, length, pass_steps, speed,
    act_err [4], crash_target, by_teacher_frac``; unfinished episodes are included only when
    ``reset_done`` is False (they cannot exist there).
    """
    rng = rng or np.random.default_rng(0)
    n = fb.n
    assert not (defer_render and runner is not None), "the student needs its frames while flying"
    render = render and not defer_render
    s = fb.reset()
    if render:
        renderer.set_gates(np.arange(n), s["gates_pos"], s["gates_yaw"])
    new = np.ones(n, bool)
    use_teacher = np.ones(n, bool) if runner is None else rng.random(n) < beta
    seg_left = rng.integers(seg_len[0], seg_len[1] + 1, n)
    noise_std = rng.uniform(0, teacher_noise, n).astype(np.float32)
    burst_left, burst_off = np.zeros(n, int), np.zeros((n, 4), np.float32)
    ep_id = np.arange(n)
    acc = dict(speed=np.zeros(n), err=np.zeros((n, 4)), err_n=np.zeros(n), teach=np.zeros(n), steps=np.zeros(n))
    passes: list[list[int]] = [[] for _ in range(n)]
    episodes, rows, traces = [], [], []
    info = dict(frames_rendered=0, render_s=0.0, policy_s=0.0, sim_s=0.0, steps=0)
    t_start = time.time()

    def finish(e, st_):
        k = max(acc["err_n"][e], 1)
        episodes.append(dict(
            gates=int(st_["gates_passed_total"][e]), outcome=int(st_["outcome"][e]), length=int(st_["t"][e]),
            pass_steps=list(passes[e]), speed=float(acc["speed"][e] / max(acc["steps"][e], 1)),
            act_err=(acc["err"][e] / k).tolist(), has_err=bool(acc["err_n"][e] > 0),
            crash_target=int(st_["target"][e]), by_teacher_frac=float(acc["teach"][e] / max(acc["steps"][e], 1)),
            slot=int(e),
        ))  # fmt: skip

    for i in range(steps):
        alive = ~s["done"]
        if not alive.any():
            break
        frames = None
        if render:
            t0 = time.perf_counter()
            frames = np.array(renderer.render(s["pos"], s["quat"], active=alive))  # copy: may be a shared buffer
            info["render_s"] += time.perf_counter() - t0
            info["frames_rendered"] += int(alive.sum())
        t0 = time.perf_counter()
        label = s["teacher_action"]
        a_teacher = label + noise_std[:, None] * rng.standard_normal((n, 4)).astype(np.float32)
        start_burst = (rng.random(n) < burst_p) & (burst_left == 0)
        if start_burst.any():
            burst_left[start_burst] = rng.integers(burst_len[0], burst_len[1] + 1, int(start_burst.sum()))
            burst_off[start_burst] = rng.uniform(-burst_mag, burst_mag, (int(start_burst.sum()), 4))
        a_teacher = a_teacher + np.where(burst_left[:, None] > 0, burst_off, 0)
        burst_left = np.maximum(burst_left - 1, 0)
        if runner is not None:
            a_student = runner.act(frames, s["prev_action"], new)
            acc["err"] += np.where(alive[:, None], np.abs(np.clip(a_student, -1, 1) - label), 0)
            acc["err_n"] += alive
            action = np.where(use_teacher[:, None], a_teacher, a_student)
        else:
            action = a_teacher
        action = np.clip(action, -1, 1).astype(np.float32)
        info["policy_s"] += time.perf_counter() - t0
        if writer is not None:
            rows.append(dict(
                frames=frames, label=label, action=action, prev_action=s["prev_action"],
                teacher_obs=s["teacher_obs"], pos=s["pos"], quat=s["quat"], vel=s["vel"], gates_pos=s["gates_pos"],
                gates_yaw=s["gates_yaw"], target=s["target"].astype(np.int8), t=s["t"].astype(np.int16),
                ep=ep_id.astype(np.int32), by_teacher=use_teacher.copy(), valid=alive.copy(),
            ))  # fmt: skip
            if len(rows) >= shard_steps and not defer_render:
                writer.write(rows)
                rows = []
        if trace:
            traces.append(dict(pos=s["pos"].copy(), quat=s["quat"].copy(), action=action, label=label,
                               target=s["target"].copy(), alive=alive.copy(), frames=frames))  # fmt: skip
        acc["speed"] += np.where(alive, np.linalg.norm(s["vel"], axis=-1), 0)
        acc["teach"] += alive & use_teacher
        acc["steps"] += alive
        t0 = time.perf_counter()
        was_done = s["done"]
        s = fb.step(action)
        info["sim_s"] += time.perf_counter() - t0
        info["steps"] += 1
        for e in np.flatnonzero(s["passed"]):
            passes[e].append(int(s["t"][e]))
        new[:] = False
        seg_left -= 1
        redraw = seg_left <= 0
        ended = s["done"] & ~was_done
        if ended.any():
            for e in np.flatnonzero(ended):
                finish(e, s)
            if reset_done:
                s = fb.reset(ended)
                if render:
                    renderer.set_gates(np.flatnonzero(ended), s["gates_pos"][ended], s["gates_yaw"][ended])
                new |= ended
                redraw |= ended
                ep_id[ended] += n
                noise_std[ended] = rng.uniform(0, teacher_noise, int(ended.sum()))
                burst_left[ended] = 0
                for k in acc:
                    acc[k][ended] = 0
                for e in np.flatnonzero(ended):
                    passes[e] = []
        if redraw.any():
            m = int(redraw.sum())
            seg_left[redraw] = rng.integers(seg_len[0], seg_len[1] + 1, m)
            if runner is not None:
                use_teacher[redraw] = rng.random(m) < beta
        if log and (i + 1) % 250 == 0:
            log(f"  fly step {i + 1}/{steps} episodes done {len(episodes)} "
                f"({info['frames_rendered'] / max(time.time() - t_start, 1e-6):.0f} frames/s)")  # fmt: skip
    if writer is not None and rows:
        if defer_render:
            render_rows(renderer, rows, info, log)
        for k in range(0, len(rows), shard_steps):
            writer.write(rows[k : k + shard_steps])
    if not reset_done:  # evaluation cut before the time limit: count what is still flying as not crashed
        for e in np.flatnonzero(~s["done"]):
            finish(e, s)
    info["wall_s"] = time.time() - t_start
    if trace:
        info["trace"] = traces
    return episodes, info


def render_rows(renderer, rows: list[dict], info: dict, log=None, chunk: int = 32):
    """Fill ``frames`` of recorded steps with `render_trajectories`; gates are re-set at episode starts."""
    n, t = len(rows[0]["t"]), 0
    t0 = time.perf_counter()
    while t < len(rows):
        starts = np.flatnonzero(rows[t]["t"] == 0) if t else np.arange(n)
        if len(starts):
            renderer.set_gates(starts, rows[t]["gates_pos"][starts], rows[t]["gates_yaw"][starts])
        end = min(t + chunk, len(rows))
        for u in range(t + 1, end):  # stop the window before the next episode start of any slot
            if (rows[u]["t"] == 0).any():
                end = u
                break
        pos = np.stack([r["pos"] for r in rows[t:end]], 1)
        quat = np.stack([r["quat"] for r in rows[t:end]], 1)
        frames = renderer.render_trajectories(np.arange(n), pos, quat)
        for j, r in enumerate(rows[t:end]):
            r["frames"] = np.array(frames[:, j])
        info["frames_rendered"] += n * (end - t)
        if log and (t // chunk) != (end // chunk) and (end // chunk) % 8 == 0:
            log(f"  rendered {end}/{len(rows)} steps ({info['frames_rendered'] / (time.perf_counter() - t0):.0f} frames/s)")
        t = end
    info["render_s"] += time.perf_counter() - t0


def summarize(episodes: list[dict], policy_hz: float = 50.0) -> dict:
    """Metrics over finished episodes (definitions in scripts/eval_student.py)."""
    if not episodes:
        return dict(episodes=0)
    g = np.array([e["gates"] for e in episodes])
    out = np.array([e["outcome"] for e in episodes])
    length = np.array([e["length"] for e in episodes])
    laps = []
    for e in episodes:
        p = np.asarray(e["pass_steps"])
        if len(p) > 6:
            laps.append((p[6:] - p[:-6]) / policy_hz)
    laps = np.concatenate(laps) if laps else np.zeros(0)
    crash_tgt = np.bincount([e["crash_target"] for e in episodes if e["outcome"] > 0], minlength=6)
    err = np.array([e["act_err"] for e in episodes if e.get("has_err")])
    wts = np.array([e["length"] for e in episodes if e.get("has_err")], float)
    res = dict(
        episodes=len(episodes),
        frac_1_lap=float((g >= 6).mean()), frac_3_laps=float((g >= 18).mean()),
        frac_full_episode_no_crash=float((out == 0).mean()),
        gates_per_episode=float(g.mean()), gates_per_episode_median=float(np.median(g)),
        crash_rate=float((out > 0).mean()),
        crash_rate_by_cause={OUTCOMES[k]: float((out == k).mean()) for k in (1, 2, 3)},
        crashes_by_target_gate=crash_tgt.tolist(),
        episode_length_s=float(length.mean() / policy_hz),
        lap_time_s=float(laps.mean()) if len(laps) else None, n_laps=int(len(laps)),
        speed_m_s=float(np.average([e["speed"] for e in episodes], weights=np.maximum(length, 1))),
    )  # fmt: skip
    if len(err):
        mae = (err * wts[:, None]).sum(0) / wts.sum()
        res["action_mae_vs_teacher"] = float(mae.mean())
        res["action_mae_per_channel"] = mae.tolist()
    return res


def evaluate(fb, renderer, runner, steps: int | None = None, rng=None, trace: bool = False, writer=None, log=None):
    """Student alone (``runner``) or the teacher (``runner=None``, no rendering) for one episode per
    slot, no resets. -> (metrics, episodes, info)."""
    steps = steps or fb.cfg.max_steps
    eps, info = fly(fb, renderer, runner, steps=steps, beta=0.0, reset_done=False, rng=rng, trace=trace,
                    render=runner is not None, writer=writer, log=log)  # fmt: skip
    return summarize(eps, fb.cfg.policy_hz), eps, info


def append_jsonl(path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def add_student_args(ap):
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=90)
    ap.add_argument("--frames", type=int, default=3, help="frame history length")
    ap.add_argument("--no-prev-action", action="store_true")
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--channels", default="32,64,96,128,192")
    ap.add_argument("--prev-action-drop", type=float, default=0.0)


def cfg_from_args(args) -> st.StudentConfig:
    return st.StudentConfig(
        width=args.width, height=args.height, n_frames=args.frames, use_prev_action=not args.no_prev_action,
        aux_weight=args.aux_weight, lr=args.lr, batch=args.batch,
        channels=tuple(int(c) for c in args.channels.split(",")), prev_action_drop=args.prev_action_drop,
    )  # fmt: skip


def inference_speed(cfg: st.StudentConfig, params, batches=(1, 64), reps: int = 200) -> dict:
    """Wall time of one forward pass (frames already on the device), per batch size."""
    import jax.numpy as jnp

    net = st.StudentNet(cfg)
    f = jax.jit(lambda p, x, a: net.apply(p, x.astype(jnp.float32) / 255.0, a)[0])
    out = {}
    for b in batches:
        x = jnp.zeros((b, cfg.n_frames, cfg.height, cfg.width, 3), jnp.uint8)
        a = jnp.zeros((b, 4))
        f(params, x, a).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            f(params, x, a).block_until_ready()
        dt = (time.perf_counter() - t0) / reps
        out[f"batch_{b}"] = dict(ms_per_call=1e3 * dt, calls_per_s=1 / dt)
    return out
