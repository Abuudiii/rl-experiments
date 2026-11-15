"""Markdown table of `scripts/eval_student.py` results with 95% Wilson intervals for the rates.

    python scripts/student_final_table.py runs/student/<run>/eval.json [--other old=<eval.json>] [--out table.md]

One row per layout, pilot (student, teacher reference, other students) and setting (randomised,
nominal): episodes, three-lap rate, one-lap rate, crash rate (all causes, then gate / ground / out of
bounds), gates per episode (mean +- standard error), flying lap time [s], speed [m/s].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(p, n):
    lo, hi = wilson(p, n)
    return f"{100 * p:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def gates_se(eps):
    g = [e["gates"] for e in eps]
    m = sum(g) / len(g)
    sd = math.sqrt(sum((x - m) ** 2 for x in g) / max(len(g) - 1, 1))
    return f"{m:.1f} +- {sd / math.sqrt(len(g)):.1f}"


def rows(path: Path, label_student: str, teacher: bool):
    res = json.loads(path.read_text())
    eps = json.loads(path.with_suffix(".episodes.json").read_text())
    out = []
    for name, cell in res["scenarios"].items():
        for who, key in (("student", "student"), ("teacher", "teacher_reference")):
            if who == "teacher" and not teacher:
                continue
            for setting, m in cell.get(key, {}).items():
                n = m["episodes"]
                c = m["crash_rate_by_cause"]
                lab = label_student if who == "student" else "teacher (state)"
                lap = "-" if m["lap_time_s"] is None else f"{m['lap_time_s']:.2f}"
                out.append((name, setting, lab, n, rate(m["frac_3_laps"], n), rate(m["frac_1_lap"], n), rate(m["crash_rate"], n),
                            f"{100 * c['gate_crash']:.0f} / {100 * c['ground']:.0f} / {100 * c['out_of_bounds']:.0f}",
                            gates_se(eps[f"{name}/{who}_{setting}"]), lap, f"{m['speed_m_s']:.2f}"))  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval")
    ap.add_argument("--label", default="student (camera)")
    ap.add_argument("--other", nargs="*", default=[], help="label=path of other students' eval.json (same seed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    allrows = rows(Path(args.eval), args.label, True)
    for o in args.other:
        lab, p = o.split("=", 1)
        allrows += rows(Path(p), lab, False)
    order = {n: i for i, n in enumerate(dict.fromkeys(r[0] for r in allrows))}
    allrows.sort(key=lambda r: (order[r[0]], r[1] != "randomised", r[2]))
    head = ("layout", "setting", "pilot", "episodes", "3 laps % [95% CI]", "1 lap % [95% CI]", "crash % [95% CI]",
            "gate / ground / out %", "gates per episode +- se", "lap s", "speed m/s")  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in allrows]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
