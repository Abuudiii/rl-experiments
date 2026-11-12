"""Per-round learning table of DAgger runs (markdown) from ``<run>/rounds.jsonl``.

    python scripts/student_rounds_table.py runs/student/<run> [more runs] [--write]

``--write`` also stores the table as ``<run>/rounds_table.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = ("round", "beta", "new frames", "data frames", "collect s", "train s", "train MAE", ">=1 lap", ">=3 laps",
        "no crash", "gates/ep", "gate / ground / oob", "lap s", "MAE vs teacher", "eval frames", "eval s")  # fmt: skip


def table(run: Path) -> str:
    rows = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines()]
    out = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        m, c = r["eval"], r["eval"]["crash_rate_by_cause"]
        lap = "-" if m["lap_time_s"] is None else f"{m['lap_time_s']:.2f}"
        out.append("| " + " | ".join(str(v) for v in (
            r["round"], f"{r['beta']:.2f}", r.get("collect_frames", 0), r["data_frames"],
            f"{r.get('collect_wall_s', 0):.0f}", f"{r['train_wall_s']:.0f}", f"{r['train_mae']:.3f}",
            f"{100 * m['frac_1_lap']:.1f}%", f"{100 * m['frac_3_laps']:.1f}%",
            f"{100 * m['frac_full_episode_no_crash']:.1f}%", f"{m['gates_per_episode']:.1f}",
            f"{100 * c['gate_crash']:.0f} / {100 * c['ground']:.0f} / {100 * c['out_of_bounds']:.0f} %", lap,
            f"{m.get('action_mae_vs_teacher', float('nan')):.3f}", r["eval_frames"], f"{r['eval_wall_s']:.0f}",
        )) + " |")  # fmt: skip
    return "\n".join(out)


if __name__ == "__main__":
    write = "--write" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--"):
            continue
        t = table(Path(a))
        print(f"\n{a}\n{t}")
        if write:
            (Path(a) / "rounds_table.md").write_text(t + "\n")
