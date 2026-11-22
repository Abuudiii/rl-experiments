# Perception pipeline: everything for the post

Camera frame -> gate detector (CNN) -> geometry + gate order (code) -> state estimate with IMU (code)
-> flight policy (PPO) -> sticks. No green next-gate highlight.

## Media

| File | What it is |
|---|---|
| `videos/slalom_unreal.mp4` | Slalom in Unreal, flown by the pipeline. 30 s, 26 gates (4 laps), no crash. |
| `videos/split_s_unreal.mp4` | Acrobatic split-S in Unreal, flown by the pipeline, with a side view. 30 s, 41 gates (6 laps), no crash, up to 158 deg tilt. |
| `flights_3d.mp4` | Six 3D chase panels: slalom, climb and dive, hairpin, split-S, circuit, clockwise. Pipeline flights in Unreal. |
| `architecture.png` | Pipeline diagram with three result charts. |
| `figures/` | The 13 figures on the model page (estimator data flow, training, tracks, results by track, noise, estimator versions, crash causes, detector, old students, split-S, Unreal by track, memory experiments). Made by `scripts/hf_figures.py` |
| `extra/` | Detector-view clips (what the detector saw and found); split-S video with the old chase camera. |

Both videos are the first flight from the fixed start (seed 900002, slot 0), not picked.
The `.mp4` files are not in git; they are on the model page: https://huggingface.co/Abuudiii/fpv-perception-pipeline

## Numbers

| | Result |
|---|---|
| Slalom, Unreal, 24 randomised flights | 20 of 24 three laps, 1 crash, 6.71 s laps |
| Split-S, Unreal, 24 randomised flights | 23 of 24 three laps, 1 crash, 4.67 s laps |
| Random unseen tracks, synthetic renderer, 2,048 flights | 93.8 % three laps (camera CNN 27.7 %, CNN + green gate about 70 %) |
| Six named tracks, synthetic renderer, 2,048 flights each | 92.5 % mean |
| Detector on Unreal frames, next gate found | 74.2 % before fine-tuning, 99.87 % after (90k Unreal frames) |
| Detector on upside-down split-S frames | 90.8 % before the acrobatic fine-tune, 98.3 % after |
| On-board compute per step (detector + estimator + policy) | 2.2 ms on an RTX PRO 4000, 20 ms budget |

## Caveats

- Unreal numbers: 24 flights each, one Unreal scene.
- The pipeline is given the gate order (nominal positions) before the flight; the old camera CNNs were not.
- Split-S uses its own policy and detector fine-tune; slalom uses the general ones.
- The flip exists because the split-S teacher was rewarded for going inverted; on speed alone it banks to about 100 deg.

## Where things come from

- Videos and flight records: `runs/percep/stage6/videos/<track>/`
- Results: `runs/percep/stage6/RESULTS.md` (Unreal), `runs/percep/stage5/RESULTS.md` (synthetic, closed loop),
  `runs/percep/policy/RESULTS.md` (policy), `runs/percep/detector/RESULTS.md` (detector)
- Checkpoints: detector `runs/percep/stage6/det_ft/B_all` (general), `C_acro` (split-S); policy
  `runs/percep/policy/train/est_rand_v5` (general), `est_acro_v5/ckpt_eval1.pkl` (split-S; the video and the
  23 of 24 result used this file, `ckpt_latest.pkl` next to it was never evaluated); teacher `runs/teacher/acro/ss_inv20`
- This folder's figures: `scripts/post_figures.py --out perception-pipeline`
