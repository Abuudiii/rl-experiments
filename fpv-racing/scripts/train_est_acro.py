"""Stage-4 estimated-observation policy on an acrobatic layout: `scripts/train_est.py` unchanged (same recipe,
options and outputs) with the acro layouts registered (the track comes from the ``--init`` teacher's config,
e.g. split_s with z_max 6 m) and the OPTIONAL dive shaping of `fpv.acro.ShapedRaceEnv` in the dynamics env.

    HIP_VISIBLE_DEVICES=2 .venv-rocm/bin/python scripts/train_est_acro.py --w-invert 0.2 -- \
        --variant rand --name est_acro_v5 --seed 410 --init runs/teacher/acro/ss_inv20/ckpt_latest.pkl \
        --set ppo.total_steps=1.8e9 est.place_mem_w=0.3 est.rescue_after_next=False

Arguments before ``--`` are this wrapper's (--w-invert, --invert-vz, --dive-gate); the rest go to train_est.py.
The shaping is written to <out>/<name>/acro_shaping.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fpv import acro  # noqa: E402

acro.register()

import train_est  # noqa: E402
from fpv.env_est import EstEnv  # noqa: E402


def main():
    i = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-invert", type=float, default=0.0)
    ap.add_argument("--invert-vz", type=float, default=0.0)
    ap.add_argument("--dive-gate", type=int, default=1)
    a = ap.parse_args(sys.argv[1:i])
    shaping = dict(w_invert=a.w_invert, invert_vz=a.invert_vz, dive_gate=a.dive_gate,
                   term="w_invert * max(0, -(body z . world z)) per step while target == dive_gate (fpv.acro.ShapedRaceEnv)")

    class AcroEstEnv(EstEnv):
        def __init__(self, cfg, ecfg, circuit=None, device=None):
            super().__init__(cfg, ecfg, circuit=circuit, device=device)
            self.race = acro.ShapedRaceEnv(cfg, circuit=circuit, device=device, w_invert=a.w_invert,
                                           dive_gate=a.dive_gate, invert_vz=a.invert_vz)  # fmt: skip

    train_est.EstEnv = AcroEstEnv
    sys.argv = [sys.argv[0], *sys.argv[i + 1 :]]
    rest = argparse.ArgumentParser(add_help=False)
    rest.add_argument("--name")
    rest.add_argument("--out", default=str(ROOT / "runs" / "percep" / "policy" / "train"))
    r, _ = rest.parse_known_args(sys.argv[1:])
    (Path(r.out) / r.name).mkdir(parents=True, exist_ok=True)
    (Path(r.out) / r.name / "acro_shaping.json").write_text(json.dumps(shaping, indent=1))
    train_est.main()


if __name__ == "__main__":
    main()
