"""Disk-backed DAgger data: `fpv.dagger.Dataset` without holding the frames in RAM, plus sequence sampling.

`DiskDataset` has the interface of `fpv.dagger.Dataset` (``add_dir``, ``blocks``, ``loaded``, ``len``, ``sizes``,
``sample``) and `DiskBlock` the attributes of `fpv.dagger.Block`. The small per-step arrays (labels, previous
actions, auxiliary targets, episode bookkeeping) are in RAM. The frames stay where `fly` wrote them: every shard
is an uncompressed ``np.savez`` file, so its ``frames`` member (uint8 [Ts, n, H, W, 3], time-major) is mapped in
place with a read-only `np.memmap` at that member's byte offset. `ShardFrames` presents the shards of one call as
the env-major flat array of `Block.frames` (frame of slot e at step s at index e * T + s) and reads only the
indexed frames; nothing is copied to another file and the whole array is never materialised. With the same RNG
state and weights, `sample` returns exactly what `Dataset.sample` returns for the same data.

`DiskDataset.sample_sequences` returns contiguous windows of one episode for a recurrent student, optionally with
the previous frame packed next to each current one (``n_frames=2``), see there.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

from fpv import student as st
from fpv.dagger import NOT_TRAINING, drop_page_cache, relabeller

SMALL = ("t", "ep", "valid", "label", "prev_action", "teacher_obs", "by_teacher")


# region frames
def _npz_member(path, key: str):
    """(byte offset, shape, dtype) of the array ``key`` stored uncompressed in the .npz ``path``."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}: {key} is compressed; shards are written uncompressed (np.savez)")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"{path}: bad zip local header for {key}")
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(f)
        read = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = read(f)
        if fortran:
            raise ValueError(f"{path}: {key} is Fortran-ordered")
        return f.tell(), shape, dtype


class ShardFrames:
    """Read-only, array-like env-major view of the frames of one `fly` call, mapped from its time-major shards.

    ``shape`` (n * T, H, W, 3), ``dtype`` uint8, ``len``. ``frames[idx]`` takes an int, a slice or an integer
    array of flat indices (flat index e * T + s = slot e, step s) and returns a plain ndarray of shape
    ``idx.shape + (H, W, 3)`` holding copies of only those frames.
    """

    def __init__(self, paths, T: int, n: int):
        self.maps, starts = [], [0]
        hw = None
        for p in paths:
            off, shape, dtype = _npz_member(p, "frames")
            hw = hw or tuple(shape[2:])
            if dtype != np.uint8 or tuple(shape[1:]) != (n, *hw):
                raise ValueError(f"{p}: frames {dtype} {shape}, expected uint8 [*, {n}, {', '.join(map(str, hw))}]")
            self.maps.append(np.memmap(p, np.uint8, "r", offset=off, shape=tuple(shape)))
            starts.append(starts[-1] + shape[0])
        if starts[-1] != T:
            raise ValueError(f"{paths[0].parent}: frames shards hold {starts[-1]} steps, expected {T}")
        self.starts = np.asarray(starts[:-1])  # first step of each shard
        self.T, self.n = T, n
        self.shape, self.dtype, self.ndim = (n * T, *hw), np.dtype(np.uint8), 1 + len(hw)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self)))
        idx = np.asarray(idx)
        if idx.dtype.kind not in "iu":
            raise IndexError(f"ShardFrames takes integer indices or slices, got {idx.dtype}")
        idx = np.where(idx < 0, idx + len(self), idx)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError(f"index out of range for ShardFrames of length {len(self)}")
        e, s = np.divmod(idx.reshape(-1), self.T)
        k = np.searchsorted(self.starts, s, side="right") - 1
        out = np.empty((e.size, *self.shape[1:]), self.dtype)
        for j in np.unique(k):
            m = k == j
            out[m] = self.maps[j][s[m] - self.starts[j], e[m]]
        return out.reshape(*idx.shape, *self.shape[1:])


# region data
class DiskBlock:
    """`fpv.dagger.Block` with frames on disk: one `fly` call, env-major and flattened, frame i belongs to the
    episode starting at start[i]; the last ``drop_before_crash`` frames before a crash are not training targets.
    ``frames`` is a `ShardFrames` over the call's shards.

    Beyond `Block`: ``eligible`` bool [n*T] (= frame is in ``valid_idx``) and ``seq_start`` [n*T], the first
    index of the contiguous recorded segment of frame i: like ``start``, but also broken where ``ep`` changes or
    the previous step is invalid, so a sequence never spans two episodes even in a malformed recording.
    """

    def __init__(self, paths, round_id: int, drop_before_crash: int = 8, max_steps: int = 1500, relabel=None):
        paths = sorted(paths)
        parts = []
        for p in paths:
            with np.load(p) as z:
                parts.append({k: z[k] for k in SMALL})  # members are read lazily: frames stay on disk
            drop_page_cache(p)
        cat = lambda k: np.concatenate([p[k] for p in parts], 0)  # noqa: E731
        t, valid, ep = cat("t"), cat("valid"), cat("ep")
        T, n = t.shape
        flat = lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1)).reshape(T * n, *a.shape[2:])  # noqa: E731
        self.frames = ShardFrames(paths, T, n)
        self.label, self.prev_action = flat(cat("label")), flat(cat("prev_action"))
        obs = flat(cat("teacher_obs"))
        self.aux = obs[:, st.AUX_IDX]
        if relabel is not None:
            self.label = relabel(obs)
        del obs
        self.by_teacher = flat(cat("by_teacher"))
        ar = np.arange(T)[:, None]
        first = (t == 0) | (ar == 0)  # an episode start, or the start of the recording
        idx = np.where(first, ar, 0)
        start_t = np.maximum.accumulate(idx, axis=0)  # [T, n] time index of the episode's first frame
        self.start = (np.swapaxes(start_t, 0, 1) + np.arange(n)[:, None] * T).reshape(-1)
        # index of the next episode boundary (a reset or the end of validity) after each step
        bound = np.where(first | ~valid, ar, 10 * T)
        bound[0] = np.where(valid[0], 10 * T, 0)
        nxt = np.minimum.accumulate(np.concatenate([bound[1:], np.full((1, n), 10 * T)])[::-1], axis=0)[::-1]
        to_end = nxt - ar  # 1 = last frame of its episode
        crash_soon = (to_end <= drop_before_crash) & (t + to_end - 1 < max_steps - 1)
        self.dropped = int((valid & crash_soon).sum())
        self.eligible = flat(valid & ~crash_soon)
        self.valid_idx = np.flatnonzero(self.eligible)
        brk = first.copy()
        brk[1:] |= (ep[1:] != ep[:-1]) | ~valid[:-1]
        seq_t = np.maximum.accumulate(np.where(brk, ar, 0), axis=0)
        self.seq_start = (np.swapaxes(seq_t, 0, 1) + np.arange(n)[:, None] * T).reshape(-1)
        self.round = round_id

    def __len__(self):
        return len(self.valid_idx)


class DiskDataset:
    def __init__(self, relabel: str | None = None):
        """``relabel``: teacher checkpoint whose actions replace the stored labels (teacher swap).
        Data directories are only read; frames are mapped from their shards in place."""
        self.blocks: list[DiskBlock] = []
        self.loaded: set[str] = set()
        self.relabel = relabeller(relabel) if relabel else None

    def add_dir(self, data_dir, rounds=None, extra: bool = False):
        """Load every not yet loaded `fly` call found in data_dir (optionally only some rounds).
        ``extra`` marks data of another run / renderer (see `fpv.dagger.train`'s ``extra_frac``)."""
        if (Path(data_dir) / NOT_TRAINING).exists():
            raise ValueError(f"{data_dir} holds evaluation flights, not training data")
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
            self.blocks.append(DiskBlock(paths, rid, relabel=self.relabel))
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

    def _counts(self, rng, batch: int, weights):
        w = np.array([len(b) for b in self.blocks], float) if weights is None else np.asarray(weights, float)
        return rng.multinomial(batch, w / w.sum())

    def sample(self, rng, batch: int, n_frames: int, weights=None):
        """-> frames uint8 [B, K, H, W, 3] (oldest first), prev_action, label, aux (as `Dataset.sample`)."""
        counts = self._counts(rng, batch, weights)
        out = [[], [], [], []]
        for b, c in zip(self.blocks, counts):
            if c == 0:
                continue
            i = b.valid_idx[rng.integers(0, len(b.valid_idx), c)]
            idx = np.maximum(i[:, None] - np.arange(n_frames - 1, -1, -1)[None], b.start[i][:, None])
            for o, v in zip(out, (b.frames[idx], b.prev_action[i], b.label[i], b.aux[i])):
                o.append(v)
        return tuple(np.concatenate(o) for o in out)

    def sample_sequences(self, rng, batch: int, length: int, burn_in: int = 0, weights=None, n_frames: int = 1):
        """Contiguous windows of T = burn_in + length steps of one recorded episode segment, oldest first.

        Blocks are drawn like `sample` (``weights`` per block, default its number of eligible targets), then the
        window's last step uniformly from the block's ``valid_idx``. A window reaching back past its segment start
        is left-padded with the segment's first step (frame and arrays) marked ``present=False``.
        ``n_frames=2`` packs each step's previous frame and its own frame (in that order) along the channel axis;
        the previous frame of a segment's first step (and of padding) is that step itself, also when the window
        starts mid-segment (the frame before the window is read, not clamped to the window).
        -> frames uint8 [B, T, H, W, 3 * n_frames], prev_action [B, T, 4], label [B, T, 4], aux [B, T, AUX_DIM],
           present bool [B, T] (False: padding, do not update the recurrent state),
           loss_mask bool [B, T] (present, an eligible target, and not among the first ``burn_in`` steps).
        """
        if length < 1 or burn_in < 0:
            raise ValueError(f"length {length} must be >= 1 and burn_in {burn_in} >= 0")
        if n_frames not in (1, 2):
            raise ValueError(f"n_frames {n_frames} must be 1 or 2")
        back = np.arange(burn_in + length - 1, -1, -1)[None]
        counts = self._counts(rng, batch, weights)
        out = [[], [], [], [], [], []]
        for b, c in zip(self.blocks, counts):
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
            for o, v in zip(out, (frames, b.prev_action[idx], b.label[idx], b.aux[idx], present, loss)):
                o.append(v)
        return tuple(np.concatenate(o) for o in out)
