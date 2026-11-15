"""Render the states of an existing DAgger data set again with another renderer (open loop).

    .venv-sim/bin/python scripts/rerender_dagger_shards.py --src /tmp/fpv_student/synth3 \
        --out runs/student/<run>/data --renderer airsim \
        --renderer-args '{"gpus":[0,1,2,3,4,5],"instances_per_gpu":4,"look":"stock"}' \
        --width 320 --height 180 --rounds 0,1,2,3 --steps 600

The trajectories of a DAgger run made on the fast synthetic renderer (teacher-flown with noise,
mixture-flown, student-flown; all labelled by the teacher) do not depend on the target renderer's
frames, so they can be rendered there with `render_trajectories`, which costs half as much per
frame as closed-loop rendering. For every source `fly` call of the chosen rounds, ``--passes``
disjoint sets of ``renderer.n`` of its parallel slots (chosen at random) are rendered for
``--steps`` steps from ``--offset`` and written as shards with the same round number (one output
call per pass). Source frames are not read. Passes already listed in ``<out>/rerender.jsonl`` are
skipped, so the script can be re-run after an interruption. The source call's FlightBatch seed (from
``<src>/seeds.jsonl``) is recorded in ``<out>/seeds.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fpv import dagger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default="airsim")
    ap.add_argument("--renderer-args", default="{}")
    ap.add_argument("--n-envs", type=int, default=24, help="slots (synthetic renderer only; airsim: from its args)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--rounds", default=None, help="comma list of source rounds (default all)")
    ap.add_argument("--steps", type=int, default=600, help="steps per source call to render")
    ap.add_argument("--offset", type=int, default=0, help="first step of the source call to use")
    ap.add_argument("--passes", type=int, default=1, help="disjoint slot sets rendered per source call")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    out = Path(args.out)
    rounds = None if args.rounds is None else {int(r) for r in args.rounds.split(",")}
    groups: dict[str, list[Path]] = {}
    for p in sorted(Path(args.src).glob("r*_c*_*.npz")):
        r, c, _ = p.name.split("_")
        if rounds is None or int(r[1:]) in rounds:
            groups.setdefault(f"{r}_{c}", []).append(p)
    done = set()
    if (out / "rerender.jsonl").exists():
        done = {(r["source"], r.get("pass", 0)) for r in map(json.loads, (out / "rerender.jsonl").read_text().splitlines())}
    todo = [(k, p) for k in groups for p in range(args.passes) if (k, p) not in done]
    src_meta = {}
    if (Path(args.src) / "seeds.jsonl").exists():
        src_meta = {r["call"]: r for r in map(json.loads, (Path(args.src) / "seeds.jsonl").read_text().splitlines())}
    if not todo:
        log("nothing to do")
        return
    renderer = dagger.make_renderer(args.renderer, args.n_envs, args.width, args.height, seed=args.seed,
                                    **json.loads(args.renderer_args))  # fmt: skip
    n = renderer.n
    fields = [f for f in dagger.REC_FIELDS if f != "frames"]
    stats = []
    for key, p_i in todo:
        parts = [np.load(p) for p in groups[key]]
        d = {f: np.concatenate([p[f] for p in parts], 0)[args.offset : args.offset + args.steps] for f in fields}
        T, n_src = d["t"].shape
        perm = np.random.default_rng([args.seed, int(key[1:4]), int(key[6:8])]).permutation(n_src)  # same per call
        cols = np.sort(perm[p_i * n : (p_i + 1) * n])
        assert len(cols) == n, f"{key}: {n_src} slots, not enough for pass {p_i}"
        rows = [{f: d[f][j][cols] for f in fields} for j in range(T)]
        info = dict(frames_rendered=0, render_s=0.0)
        dagger.render_rows(renderer, rows, info, log)
        meta = src_meta.get(key, {})
        writer = dagger.ShardWriter(out, int(key[1:4]), seed=meta.get("seed"), kind="rerender", source=f"{args.src}/{key}",
                                    source_kind=meta.get("kind"), renderer=args.renderer,
                                    highlight_target=dagger.highlight_on(renderer))  # fmt: skip
        for k in range(0, T, 100):
            writer.write(rows[k : k + 100])
        resets = int(sum((r["t"] == 0).sum() for r in rows[1:]))
        stats.append(dict(source=key, out_call=writer.prefix, frames=info["frames_rendered"], render_s=info["render_s"], resets=resets, **{"pass": p_i},
                          mean_pixel=float(np.mean([r["frames"].mean() for r in rows[::20]]))))  # fmt: skip
        log(f"{key} pass {p_i}: {info['frames_rendered']} frames in {info['render_s']:.0f} s "
            f"({info['frames_rendered'] / info['render_s']:.0f} frames/s), {resets} episode starts")  # fmt: skip
        dagger.append_jsonl(out / "rerender.jsonl", stats[-1])
    renderer.close()


if __name__ == "__main__":
    main()
