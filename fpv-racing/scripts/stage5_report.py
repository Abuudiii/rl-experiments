"""Stage-5 tables (markdown) from the evaluation JSONs, with 95 % Wilson intervals.

    python3 scripts/stage5_report.py > runs/percep/stage5/tables.md

Row sources: "s5:<dir>" = eval_stage5.py run outputs (<dir>/<block>[_name].json), "s4:<json>" = eval_est.py output
(random + scenarios in one file). Missing files are skipped.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S5 = ROOT / "runs/percep/stage5"
S4 = ROOT / "runs/percep/policy/eval"
LAYOUTS = ["circuit", "clockwise", "figure8", "slalom", "climb_dive", "sprint_hairpin"]
BLOCKS = ["random", *LAYOUTS]


def wilson(p, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(spec, suffix=""):
    kind, path = spec.split(":", 1)
    out = {}
    if kind == "s4":
        f = ROOT / path
        if not f.exists():
            return out
        r = json.loads(f.read_text())
        out["random"] = r.get("random")
        out |= r.get("scenarios", {})
    else:
        for b in BLOCKS:
            f = ROOT / path / f"{b}{suffix}.json"
            if f.exists():
                out[b] = json.loads(f.read_text())
    return {k: v for k, v in out.items() if v}


def pct(m, key):
    p, n = m[key], m["episodes"]
    lo, hi = wilson(p, n)
    return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def cells(m):
    lt = m["lap_time_s"]["flying_mean"]
    pw = m.get("perception", {}).get("next_pose_wrong_step_frac")
    return [str(m["episodes"]), pct(m, "frac_3_laps"), pct(m, "frac_1_lap"), pct(m, "crash_rate"),
            f"{lt:.2f}" if lt else "-", f"{100 * pw:.2f}" if pw is not None else "-"]  # fmt: skip


HEAD = "| system | flights | 3 laps % [95% CI] | 1 lap % [95% CI] | crash % [95% CI] | flying lap s | next-pose wrong % steps |"
SEP = "|---|---|---|---|---|---|---|"


def table(rows, block):
    print(HEAD)
    print(SEP)
    for label, spec, suf in rows:
        d = load(spec, suf)
        if block in d:
            print(f"| {label} | " + " | ".join(cells(d[block])) + " |")
    print()


def layout_mean(rows):
    print("| system | flights | 3 laps % (mean of 6) | 1 lap % | crash % | flying lap s | " + " | ".join(LAYOUTS) + " (3 laps %) |")
    print("|---|---|---|---|---|---|" + "---|" * len(LAYOUTS))
    for label, spec, suf in rows:
        d = load(spec, suf)
        if not all(b in d for b in LAYOUTS):
            continue
        ms = [d[b] for b in LAYOUTS]
        mean = lambda k: 100 * sum(m[k] for m in ms) / 6  # noqa: E731
        lt = sum(m["lap_time_s"]["flying_mean"] or 0 for m in ms) / 6
        print(f"| {label} | {sum(m['episodes'] for m in ms)} | {mean('frac_3_laps'):.1f} | {mean('frac_1_lap'):.1f} | "
              f"{mean('crash_rate'):.1f} | {lt:.2f} | " + " | ".join(f"{100 * m['frac_3_laps']:.1f}" for m in ms) + " |")  # fmt: skip
    print()


MAIN_ROWS = [
    ("teacher, true state (stage-4 run)", "s4:runs/percep/policy/eval/teacher/eval_perfect.json", ""),
    ("stage 4 perfect detections (stage-4 run)", "s4:runs/percep/policy/eval/est_rand_v5/eval_perfect.json", ""),
    ("stage 4 perfect, re-flown in the stage-5 harness (check a)", "s5:runs/percep/stage5/checks/a_true_perfect", ""),
    ("stage 4 x1 noise model (stage-4 run)", "s4:runs/percep/policy/eval/est_rand_v5/eval_x1.json", ""),
    ("stage 4 x1, re-flown in the stage-5 harness", "s5:runs/percep/stage5/eval/s4x1", ""),
    ("diagnostic: true corners (stage-4 perfect detections) + IMU x1", "s5:runs/percep/stage5/eval/s4_imu1", ""),
    ("**stage 5: rendered frames -> detector E thr 0.3**", "s5:runs/percep/stage5/eval/main", ""),
]
VARIANTS = {"s409": "policy seed 409", "known": "known first gate", "thr04": "detector threshold 0.4 (sensitivity)",
            "det_imu0": "diagnostic: detector with a perfect IMU (IMU noise x0)"}


def paired_cell(p, key="3laps"):
    d = p[key]
    return (f"{100 * d['a']:.1f} | {100 * d['b']:.1f} | {100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, "
            f"{100 * d['ci95'][1]:+.1f}] | {d['only_a']} / {d['only_b']}")  # fmt: skip


def variants():
    f = S5 / "diag.json"
    if not f.exists():
        return
    dg = json.loads(f.read_text())
    print("Paired on flights 0-511 of each stage-4 set (same flights, same seeds): a = stage 5 main (seed 407, thr 0.3),"
          " b = variant. diff = b - a with a 95 % interval of the paired per-flight differences; only a / only b ="
          " flights with three laps in one run but not the other.\n")  # fmt: skip
    for tag, label in VARIANTS.items():
        pv = dg["paired_variants_first512"].get(tag)
        if not pv:
            continue
        print(f"### {label}\n")
        print("| block | main 3 laps % | variant 3 laps % | diff [95% CI] | only main / only variant | main crash % | variant crash % | crash diff |")
        print("|---|---|---|---|---|---|---|---|")
        for b in BLOCKS:
            if b in pv:
                c = pv[b]["crash"]
                print(f"| {b} | {paired_cell(pv[b])} | {100 * c['a']:.1f} | {100 * c['b']:.1f} | {100 * c['diff']:+.1f} |")
        print()
    print("### stage 5 vs stage 4 on the same flights (stage 4 re-flown in the stage-5 harness; a = stage 4, b = stage 5)\n")
    print("| comparison | block | stage 4 3 laps % | stage 5 3 laps % | diff [95% CI] | only stage 4 / only stage 5 |")
    print("|---|---|---|---|---|---|")
    for k, v in dg["paired_stage4"].items():
        for b in BLOCKS:
            if b in v:
                print(f"| {k} | {b} | {paired_cell(v[b])} |")
    print()


def main():
    print("## Random tracks (stage-4 flights: seed 7000, 2048 x 30 s)\n")
    table(MAIN_ROWS, "random")
    print("## Six layouts (stage-4 flights: seed 7020, 2048 each)\n")
    layout_mean(MAIN_ROWS)
    for b in LAYOUTS:
        print(f"### {b}\n")
        table(MAIN_ROWS, b)
    print("## Variants (512 flights per set)\n")
    variants()


if __name__ == "__main__":
    sys.exit(main())
