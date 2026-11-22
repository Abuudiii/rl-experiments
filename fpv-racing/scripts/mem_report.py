"""Per-round table of memory-student experiments: training rows (rounds.jsonl) joined with the held-out
evaluations of `scripts/mem_eval_round.py` (eval dir, one JSON per checkpoint).

    python3 scripts/mem_report.py runs/student/mem/stage1:runs/student/mem/eval/stage1 [more run:evaldir ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(run: Path, ev: Path):
    train = {}
    if (run / "rounds.jsonl").exists():
        for line in (run / "rounds.jsonl").read_text().splitlines():
            r = json.loads(line)
            train[r["round"]] = r
    out = []
    for f in sorted(ev.glob("*ckpt_r*.json")):
        e = json.loads(f.read_text())
        rnd = e.get("round")
        t = train.get(rnd, {})
        for layout, settings in e["layouts"].items():
            for setting, m in settings.items():
                bd = m["crash_breakdown"]
                crashes = sum(v["crashes"] for v in bd.values())
                out.append(dict(
                    run=run.name, round=rnd, layout=layout, setting=setting, n=m["episodes"],
                    three=m["frac_3_laps"], one=m["frac_1_lap"], crash=m["crash_rate"],
                    mae=m.get("action_mae_vs_teacher"), train_mae=t.get("train_mae"), beta=t.get("beta"),
                    frames=t.get("data_frames"),
                    by_target="/".join(str(bd[str(g)]["crashes"]) for g in range(6)),
                    wrong_heading=sum(v["wrong_heading"] for v in bd.values()) / max(crashes, 1),
                    near_cross=sum(v["near_crossing"] for v in bd.values()) / max(crashes, 1),
                    wrong_pass=m["wrong_gate_pass_episodes"],
                ))  # fmt: skip
    return out


def fmt(v, p=3):
    return "-" if v is None else f"{v:.{p}f}" if isinstance(v, float) else str(v)


def main():
    cols = ("run", "round", "layout", "setting", "n", "three", "one", "crash", "mae", "train_mae", "beta", "frames",
            "by_target", "wrong_heading", "near_cross", "wrong_pass")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for spec in sys.argv[1:]:
        run, ev = spec.split(":")
        for r in sorted(rows(Path(run), Path(ev)), key=lambda r: (r["layout"], r["setting"], r["round"])):
            print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
