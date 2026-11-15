"""Per-layout result table (markdown) from `scripts/eval_student.py` outputs.

    python scripts/student_scenario_table.py runs/student/<run>/best/eval.json \
        --baseline runs/student/<run>/best/baseline_ue_a.json [--out table.md]

Rows per layout: the student, the teacher on the same seeds (reference) and optionally a baseline
student, randomised and nominal settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ("layout", "pilot", "setting", "episodes", ">=3 laps", ">=1 lap", "crashes", "gate / ground / oob",
        "crashes by target gate", "crashes by time (0-2-4-8-12-16-20 s)", "lap s", "speed m/s", "MAE vs teacher")  # fmt: skip


def pct(x):
    return f"{100 * x:.1f}%"


def row(layout, pilot, setting, m):
    c = m["crash_rate_by_cause"]
    hist = m.get("crash_time_hist", {}).get("frac")
    n = m["episodes"]
    return [layout, pilot, setting, n, pct(m["frac_3_laps"]), pct(m["frac_1_lap"]), pct(m["crash_rate"]),
            " / ".join(str(round(c[k] * n)) for k in ("gate_crash", "ground", "out_of_bounds")),
            " ".join(map(str, m["crashes_by_target_gate"])),
            "-" if hist is None else " ".join(str(round(f * n)) for f in hist[:6]),
            "-" if m["lap_time_s"] is None else f"{m['lap_time_s']:.2f}", f"{m['speed_m_s']:.2f}",
            f"{m['action_mae_vs_teacher']:.3f}" if "action_mae_vs_teacher" in m else "-"]  # fmt: skip


def table(ev: dict, baseline: dict | None = None) -> str:
    out = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for layout, r in ev["scenarios"].items():
        for setting in ("randomised", "nominal"):
            for pilot, src in (("student", r["student"]), ("teacher", r["teacher_reference"]),
                               ("baseline", (baseline or {}).get("scenarios", {}).get(layout, {}).get("student", {}))):
                if setting in src:
                    out.append("| " + " | ".join(map(str, row(layout, pilot, setting, src[setting]))) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ev = json.loads(Path(args.eval).read_text())
    base = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    t = table(ev, base)
    print(t)
    if args.out:
        Path(args.out).write_text(t + "\n")


if __name__ == "__main__":
    main()
