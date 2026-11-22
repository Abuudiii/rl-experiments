"""Print matched round metrics and optionally export every layout cell as CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LAYOUTS = ("random", "circuit", "clockwise", "figure8", "slalom", "climb_dive", "sprint_hairpin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/student/ssm"))
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    arms = ("S", "B", "G", "S_seed2", "S_prev", "S_large")
    cells, rounds_by_arm = [], {}
    for arm in arms:
        directory = args.root / (f"baselines/{arm}" if arm in ("B", "G") else arm)
        path = directory / "rounds.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rounds_by_arm[arm] = {row["round"]: row for row in rows}
        meta_path = directory / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        for row in rows:
            metrics = dict(random=row["eval"], **row.get("eval_scenarios", {}))
            for layout in LAYOUTS:
                if layout not in metrics:
                    continue
                m = metrics[layout]
                cells.append(dict(arm=arm, node=meta.get("node", "historical_H100"), round=row["round"],
                                  beta=row["beta"], layout=layout, episodes=m["episodes"],
                                  episode_steps=meta.get("episode_steps", 1000),
                                  three_laps_pct=100 * m["frac_3_laps"], one_lap_pct=100 * m["frac_1_lap"],
                                  crash_pct=100 * m["crash_rate"], action_mae=m.get("action_mae_vs_teacher"),
                                  collect_seed=row.get("collect_seed"), eval_seed=row.get("eval_seed"),
                                  collect_frames=row["collect_frames"], train_steps=row["train_steps"],
                                  supervised_targets_per_batch=row.get("train_targets_per_batch", 256 if arm in ("B", "G") else None),
                                  gpu_peak_bytes=(row.get("device_memory") or {}).get("peak_bytes_in_use"),
                                  eligible_aggregate_frames=row["data_frames"], collect_s=row["collect_wall_s"],
                                  train_s=row["train_wall_s"], random_eval_s=row["eval_wall_s"],
                                  named_evals_s=row.get("eval_scenarios_wall_s"), total_round_s=row.get("round_wall_s")))
    print("Random tracks,256 episodes/round,20s: 3-lap / 1-lap / crash %")
    print("round | " + " | ".join(arms))
    for r in sorted(set().union(*(set(rows) for rows in rounds_by_arm.values()))):
        text = []
        for arm in arms:
            row = rounds_by_arm[arm].get(r)
            if row is None:
                text.append("—")
            else:
                m = row["eval"]
                text.append(f"{100*m['frac_3_laps']:.1f}/{100*m['frac_1_lap']:.1f}/{100*m['crash_rate']:.1f}")
        print(f"{r} | " + " | ".join(text))
    print("\nSix named layouts,64 episodes each,20s: S / B / G, each3-lap/1-lap/crash %")
    for r in sorted(rounds_by_arm["S"]):
        for layout in LAYOUTS[1:]:
            text = []
            for arm in ("S", "B", "G"):
                row = rounds_by_arm[arm].get(r)
                m = row.get("eval_scenarios", {}).get(layout) if row else None
                text.append(f"{100*m['frac_3_laps']:.1f}/{100*m['frac_1_lap']:.1f}/{100*m['crash_rate']:.1f}" if m else "—")
            print(f"{r:02d} {layout} | " + " | ".join(text))
    for arm in ("S", "S_seed2", "S_prev", "S_large"):
        p = args.root / arm / "best.json"
        if p.exists():
            best = json.loads(p.read_text())
            print(f"\n{arm} best round{best['best_round']['round']}, parameters={best['parameters']}: {best['inference']}")
            if "failures" in best:
                f = best["failures"]
                print("S failure seed/episodes:", f["seed"], f["episodes"])
                print("S crashes by target gate:", f["metrics"]["crashes_by_target_gate"])
                print("S crash shares/progress:", f["crash_share_by_target_gate"], f["crash_progress"])
                print("S wrong-gate geometric heading proxy:",
                      {k: v for k, v in f["wrong_gate_heading_proxy"].items() if k != "longest_heading_s"})
                print("S non-target opening crossings:", f["non_target_opening_crossings"])
    benchmarks = args.root / "baselines/benchmark.json"
    if benchmarks.exists():
        print("\nRecorded-baseline checkpoint inference:", benchmarks.read_text())
    if args.csv and cells:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.csv.with_suffix(".tmp")
        with temporary.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
        temporary.replace(args.csv)
        print("\nAll rounds/layouts, MAE, budgets and walltimes:", args.csv)
    print("\nB/G are recorded earlier H100 runs, not retrained. S exactly reproduces B's round3 restart; "
          "G has a continuing evaluation stream and, from round8, extra training data. "
          "Late S/G is not budget-matched; after round3 the held-out draws differ.")


if __name__ == "__main__":
    main()
