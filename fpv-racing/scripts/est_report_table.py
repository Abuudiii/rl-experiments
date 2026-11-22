"""Markdown tables of the stage-4 evaluations (scripts/eval_est.py outputs).

    python3 scripts/est_report_table.py runs/percep/policy/eval > runs/percep/policy/eval/table.md

One row per (policy dir, eval file); columns per block (random tracks + six layouts): >= 3 laps, crash rate,
flying lap time, mean speed, crashes per flight-minute, share of steps with a wrong next-gate index / pose,
wrong-next events per flight-minute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    root = Path(sys.argv[1])
    rows = []
    for f in sorted(root.glob("*/eval_*.json")):
        d = json.loads(f.read_text())
        blocks = [("random", d["random"])] + list(d.get("scenarios", {}).items())
        for name, m in blocks:
            p = m["perception"]
            rows.append((f.parent.name, f.stem.removeprefix("eval_"), name, m["episodes"], m["frac_3_laps"], m["crash_rate"],
                         m["lap_time_s"]["flying_mean"], m["speed_m_s"]["mean"], m["crashes_per_flight_minute"],
                         p["wrong_next_step_frac"], p.get("next_pose_wrong_step_frac"),
                         p["wrong_next_events_per_flight_minute"]))  # fmt: skip
    print("| policy | condition | track | episodes | >= 3 laps | crash | lap s | speed m/s | crashes/min | wrong-next idx | "
          "wrong-next pose | wrong ev/min |")
    print("|" + "---|" * 12)
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6], 2)} | {fmt(r[7], 2)} | "
              f"{fmt(r[8])} | {fmt(r[9])} | {fmt(r[10])} | {fmt(r[11], 2)} |")


if __name__ == "__main__":
    main()
