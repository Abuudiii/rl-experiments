"""Stage 6 tables: the pipeline in Unreal next to stage 5 (synthetic frames, same flights) and the published camera
student (README), with Wilson 95 % intervals.

    python scripts/stage6_report.py --eval runs/percep/stage6/eval/<run> --out runs/percep/stage6/tables.md

Reads <eval>/<layout>.json (randomised: stage-5 flights), <layout>_nominal.json (FlightBatch nominal flights),
<layout>_known.json (known first gate), random.json; stage-5 synthetic reruns on the same flights in
runs/percep/stage6/stage5_same_flights/<layout>_s96[_known].json (+ per-step records for the paired comparison, if
--s5-rec is given); stage 5's 2048-flight results in runs/percep/stage5/eval/main/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ["figure8", "climb_dive", "circuit", "clockwise", "slalom", "sprint_hairpin"]
# README "Camera student on the other tracks" (green next gate): three laps % randomised (n), crash %, lap s, nominal 3 laps %
STUDENT = {"figure8": (100.0, 96, 0.0, 6.74, 100.0), "climb_dive": (100.0, 96, 0.0, 5.10, 100.0),
           "circuit": (91.7, 48, 10.4, 4.97, 100.0), "slalom": (79.2, 48, 27.1, 6.62, 87.5),
           "clockwise": (72.9, 48, 33.3, 5.10, 100.0), "sprint_hairpin": (35.4, 48, 72.9, 6.44, 8.3)}  # fmt: skip


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def row(d):
    """(three-lap text, one lap %, crash text, lap s, next det %, err p50, fp/frame, n, k3) of a result JSON."""
    n = d["flights"]
    k3 = round(d["frac_3_laps"] * n)
    kc = round(d["crash_rate"] * n)
    lo, hi = wilson(k3, n)
    clo, chi = wilson(kc, n)
    ds = d.get("det_stats", {})
    lap = d.get("lap_time_s", {}).get("flying_mean")
    return dict(n=n, k3=k3, three=f"{100 * d['frac_3_laps']:.1f} [{lo:.0f}, {hi:.0f}]", one=f"{100 * d['frac_1_lap']:.1f}",
                crash=f"{100 * d['crash_rate']:.1f} [{clo:.0f}, {chi:.0f}]", lap="-" if lap is None else f"{lap:.2f}",
                next_det=f"{100 * ds.get('next_detected_frac_when_visible', float('nan')):.2f}",
                err=f"{ds.get('next_err_px', {}).get('p50', float('nan')):.3f}",
                fp=f"{ds.get('false_det_per_frame', float('nan')):.4f}")  # fmt: skip


def paired(ue_json, s5_rec_path):
    """Three laps per flight on the same flights: Unreal minus synthetic, with a Wald 95 % interval of the mean diff."""
    if s5_rec_path is None or not Path(s5_rec_path).exists():
        return None
    z = np.load(s5_rec_path)
    s5 = (z["passed"].sum(0) >= 18).astype(float)
    ue_g = np.asarray(ue_json["per_flight"]["gates"])
    idx = np.asarray(ue_json["idx"])
    ok = idx < len(s5)
    d = (ue_g[ok] >= 18).astype(float) - s5[idx[ok]]
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(n=int(len(d)), diff=100 * float(d.mean()), lo=100 * (d.mean() - 1.96 * se), hi=100 * (d.mean() + 1.96 * se),
                ue_only=int(((d > 0)).sum()), synth_only=int((d < 0).sum()))  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--eval-zero", default=None, help="the same with detector E (zero-shot), if run")
    ap.add_argument("--s5", default="runs/percep/stage6/stage5_same_flights")
    ap.add_argument("--s5-rec", default=None, help="dir of eval_stage5 --rec-dir npz files of the same-flight reruns")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = Path(a.eval)
    S5 = Path(a.s5)
    out, res = [], {}
    out.append("### Randomised layouts: Unreal frames vs synthetic frames on the same flights (stage-5 flights 0..n-1)\n")
    out.append("Three laps % [95 % Wilson] / one lap % / crash % [Wilson] / flying lap s / next gate detected % in flight / "
               "next-gate corner error p50 px / false detections per frame.\n")  # fmt: skip
    hdr = "| layout | n | Unreal, fine-tuned detector | Unreal, detector E (zero-shot) | synthetic frames, detector E (stage 5, same flights) | paired 3-lap diff Unreal - synthetic [95 %] | stage 5, all 2048 | published student (green gate) |"
    out += [hdr, "|---|---|---|---|---|---|---|---|"]
    for lay in LAYOUTS:
        ue = load(E / f"{lay}.json")
        if ue is None:
            continue
        r = row(ue)
        n = r["n"]
        zs = load(Path(a.eval_zero) / f"{lay}.json") if a.eval_zero else None
        s5 = load(S5 / f"{lay}_s96.json")
        s5r = None
        if s5 is not None:  # restrict the synthetic rerun to the same n flights via its per-step records
            rec = Path(a.s5_rec) / f"stage5_same_flights_{lay}_s96.npz" if a.s5_rec else None
            pr = paired(ue, rec)
            if rec is not None and rec.exists():
                z = np.load(rec)
                g = z["passed"].sum(0)[:n]
                k = int((g >= 18).sum())
                lo, hi = wilson(k, n)
                s5r = f"{100 * k / n:.1f} [{lo:.0f}, {hi:.0f}]"
            else:
                s5r = row(s5)["three"] + f" (n={s5['flights']})"
        else:
            pr = None
        full = load(ROOT / f"runs/percep/stage5/eval/main/{lay}.json")
        st = STUDENT[lay]
        slo, shi = wilson(round(st[0] * st[1] / 100), st[1])
        out.append(f"| {lay} | {n} | **{r['three']}** / {r['one']} / {r['crash']} / {r['lap']} / {r['next_det']} / {r['err']} / {r['fp']} | "
                   + (f"{row(zs)['three']} / {row(zs)['crash']} / {row(zs)['next_det']} (n={row(zs)['n']})" if zs else "-") + " | "
                   + (s5r or "-") + " | "
                   + (f"{pr['diff']:+.1f} [{pr['lo']:+.1f}, {pr['hi']:+.1f}] ({pr['ue_only']} only Unreal, {pr['synth_only']} only synthetic)" if pr else "-") + " | "
                   + (f"{100 * full['frac_3_laps']:.1f}" if full else "-") + " | "
                   + f"{st[0]:.1f} of {st[1]} [{slo:.0f}, {shi:.0f}], crash {st[2]:.1f}, lap {st[3]:.2f} |")  # fmt: skip
        res[lay] = dict(unreal=r, unreal_zero_shot=row(zs) if zs else None, stage5_same=s5r, paired=pr,
                        stage5_all_2048=full["frac_3_laps"] if full else None, student=st)  # fmt: skip
    if res:
        ks = [res[k]["unreal"] for k in res]
        out.append(f"| mean ({len(ks)} layouts) | | **{np.mean([k['k3'] / k['n'] for k in ks]) * 100:.1f}** | | | | | |")
    out.append("")
    for tag, title in (("_nominal", "Nominal layouts (FlightBatch nominal gates and physics, seed 950001, the students' protocol)"),
                       ("_known", "Known first gate (estimator told the first gate), same randomised flights"),
                       ("_nominal_known", "Nominal, known first gate")):  # fmt: skip
        rows = [(lay, load(E / f"{lay}{tag}.json")) for lay in LAYOUTS]
        rows = [(l_, d) for l_, d in rows if d]
        if not rows:
            continue
        out += [f"### {title}\n", "| layout | n | three laps % [Wilson] | one lap % | crash % [Wilson] | lap s | next det % | err p50 px | student nominal 3 laps % (24) |",
                "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for lay, d in rows:
            r = row(d)
            s5k = load(S5 / f"{lay}_s96_known.json") if tag == "_known" else None
            out.append(f"| {lay} | {r['n']} | **{r['three']}** | {r['one']} | {r['crash']} | {r['lap']} | {r['next_det']} | {r['err']} | "
                       + (f"{STUDENT[lay][4]:.1f}" if "nominal" in tag else (f"stage 5 same flights: {row(s5k)['three']} (n={s5k['flights']})" if s5k else "-")) + " |")  # fmt: skip
            res.setdefault(lay, {})[tag[1:]] = r
        out.append("")
    rnd = load(E / "random.json")
    if rnd:
        r = row(rnd)
        full = load(ROOT / "runs/percep/stage5/eval/main/random.json")
        out += ["### Random tracks (stage-5 flights, seed 7000)\n", "| n | three laps | one lap | crash | lap s | next det % | err p50 | stage 5 all 2048 |",
                "|---|---|---|---|---|---|---|---|",
                f"| {r['n']} | **{r['three']}** | {r['one']} | {r['crash']} | {r['lap']} | {r['next_det']} | {r['err']} | {100 * full['frac_3_laps']:.1f} |", ""]  # fmt: skip
        res["random"] = r
    Path(a.out).write_text("\n".join(out) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(out))


if __name__ == "__main__":
    main()
