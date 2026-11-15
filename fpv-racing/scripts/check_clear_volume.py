"""Is the track volume free of Blocks level geometry, and are gates anywhere in it visible and lit?

    /tmp/fpv_airsim/venv/bin/python scripts/check_clear_volume.py --gpus 4 5

At the renderer's scene placement (`fpv.render_airsim.SCENE_ORIGIN`):
  1. geometry: world-axis bounding boxes of every object in the level, and AirSim's occupancy voxel
     grid of a 14 m cube around the volume (gates moved away), checked against the clear volume
     x [-3, 10], y [-3.5, 9], z [0, 4] m (track frame; the floor at z = 0 is expected);
  2. gates at the extremes of the contract volume grown by 0.3 m (randomised layouts reach that far;
     all 8 corners, min / max height) and the six
     scenarios of `fpv.track.SCENARIOS`: an overview frame per layout and an FPV frame 1.5 m in front
     of every gate; gate pixels found vs predicted silhouette, gate colour / brightness per gate.
Writes checks/out/clear_volume.json and checks/out/clear_volume_sheet.jpg.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fpv import render_airsim as ra, track  # noqa: E402
from test_render_airsim import expected_mask, gate_bars, gate_mask  # noqa: E402

OUT = ROOT / "checks" / "out"
CLEAR_LO, CLEAR_HI = np.array([-3.0, -3.5, 0.0]), np.array([10.0, 9.0, 4.0])
# contract gate volume plus the 0.3 m by which randomised layouts may leave it
GATE_LO, GATE_HI = np.array([-1.0, -1.5, 0.7]) - 0.3, np.array([8.0, 7.0, 2.5]) + 0.3


def contract_box(b):
    """AirSim world-axis box (NED, metres) -> contract-frame (lo, hi)."""
    c, s = b["center"], b["size"]
    lo = np.array([c["x"] - s["x"] / 2, c["y"] - s["y"] / 2, c["z"] - s["z"] / 2])
    hi = np.array([c["x"] + s["x"] / 2, c["y"] + s["y"] / 2, c["z"] + s["z"] / 2])
    ox, oy, oz = ra.SCENE_ORIGIN
    # x -> x - ox, y_ned = -(y + oy) -> y = -y_ned - oy, z_ned = -(z + oz) -> z = -z_ned - oz
    return np.array([lo[0] - ox, -hi[1] - oy, -hi[2] - oz]), np.array([hi[0] - ox, -lo[1] - oy, -lo[2] - oz])


def geometry(r):
    k = r.n
    r.set_gates(np.arange(k), np.repeat([[[0.0, -60.0, 1.0]] * 6], k, 0), np.zeros((k, 6)))
    centre, size, res = (CLEAR_LO + CLEAR_HI) / 2, 14.0, 0.25
    p = r.probe_scene(0, centre, size, res)
    hits, nearest = [], []
    for name, b in p["objects"].items():
        if not isinstance(b, dict) or "center" not in b or b["size"]["x"] == 0 or name.startswith(("Gate", "Drone")):
            continue
        lo, hi = contract_box(b)
        gap = float(np.max(np.maximum(CLEAR_LO - hi, lo - CLEAR_HI)))
        nearest.append((round(gap, 2), name, lo.round(2).tolist(), hi.round(2).tolist()))
        if gap < 0 and not (name == "Ground" and hi[2] <= 1e-3):
            hits.append(name)
    nearest.sort()
    n = round(size / res)
    vox = p["voxels"].reshape(n, n, n)
    # find the vertical axis: the floor fills whole layers
    layer_fill = [np.moveaxis(vox, ax, 0).reshape(n, -1).mean(1) for ax in range(3)]
    zax = int(np.argmax([lf.max() for lf in layer_fill]))
    lf = layer_fill[zax]
    occ_layers = np.flatnonzero(lf > 0)
    z_lo = centre[2] - size / 2
    out = {"bounding_boxes": {"objects": len(p["objects"]), "intersecting_clear_volume": hits,
                              "nearest_to_volume": nearest[:6]},
           "voxels": {"cube_centre": centre.tolist(), "cube_side_m": size, "res_m": res, "occupied": int(vox.sum()),
                      "vertical_axis": zax, "occupied_layers": occ_layers.tolist(),
                      "layer_fill": lf.round(3).tolist()}}
    # the floor band is z in [-2, 0] (+ the boundary layer); which index direction is up decides
    for up in (1, -1):
        zc = z_lo + (np.arange(n) + 0.5) * res if up == 1 else z_lo + (n - np.arange(n) - 0.5) * res
        band = zc[occ_layers]
        if band.min() >= -2.0 - res and band.max() <= 0.0 + res:
            above = np.moveaxis(vox, zax, 0)[zc > res]
            out["voxels"].update(index_up=up, floor_band_z_m=[float(band.min()), float(band.max())],
                                 occupied_voxels_above_floor=int(above.sum()))
    out["clear"] = not hits and out["voxels"].get("occupied_voxels_above_floor", 1) == 0
    return out


def layouts():
    corners = [np.array(c, float) for c in itertools.product(*zip(GATE_LO, GATE_HI))]
    centre = (GATE_LO + GATE_HI) / 2
    yaw_to_centre = lambda p: math.atan2(centre[1] - p[1], centre[0] - p[0])  # noqa: E731 - face the middle
    out = {"corners_a": [(c, yaw_to_centre(c)) for c in corners[:6]],
           "corners_b": [(c, yaw_to_centre(c)) for c in corners[6:]]
                        + [(np.array([GATE_LO[0], centre[1], GATE_HI[2]]), 0.0), (np.array([GATE_HI[0], centre[1], GATE_LO[2]]), math.pi)]
                        + [(np.array([centre[0], GATE_LO[1], GATE_HI[2]]), math.pi / 2), (np.array([centre[0], GATE_HI[1], GATE_LO[2]]), -math.pi / 2)]}
    for name, gates in getattr(track, "SCENARIOS", {"circuit": track.CIRCUIT}).items():
        gp, gy = track.circuit_arrays(gates)
        out[name] = list(zip(gp.astype(float), gy.astype(float)))
    return out


def label(img, text):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5])
    a = ap.parse_args()
    W, H = 640, 360
    res, tiles = {"scene_origin": list(ra.SCENE_ORIGIN), "clear_volume": [CLEAR_LO.tolist(), CLEAR_HI.tolist()],
                  "gate_volume": [GATE_LO.tolist(), GATE_HI.tolist()]}, []
    L = layouts()
    with ra.AirsimRenderer(gpus=a.gpus[:1], instances_per_gpu=1, width=W, height=H, hfov_deg=ra.TP_HFOV_DEG,
                           mount_pitch_deg=0.0, drone_scale=ra.DRONE_SCALE) as r:
        res["geometry"] = geometry(r)
        print("GEOMETRY", json.dumps(res["geometry"]), flush=True)
        res["overview"] = {}
        for name, gates in L.items():
            gp, gy = np.array([g[0] for g in gates]), np.array([g[1] for g in gates])
            r.set_gates([0], gp[None], gy[None])
            pts = track.gate_corners(gp, gy, track.GATE_OUTER).reshape(-1, 3)
            cp, cq = ra.overview_camera(pts, W, H)
            f = r.render(cp[None], cq[None])[0]
            # gate pixels vs the gate silhouettes predicted for this camera
            pred = np.zeros((H, W), bool)
            for g in range(6):
                for bar in gate_bars(gp[g], gy[g]):
                    uv, z = ra.project_points(bar, cp, cq, W, H, ra.TP_HFOV_DEG)
                    m = np.zeros((H, W), np.uint8)
                    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
                    pred |= m.astype(bool)
            got = gate_mask(f)
            res["overview"][name] = {"camera_pos": cp.round(2).tolist(), "gate_px": int(got.sum()), "pred_px": int(pred.sum()),
                                     "iou": round(float((got & pred).sum() / max(1, (got | pred).sum())), 3)}
            tiles.append(label(f, f"overview {name}"))
    with ra.AirsimRenderer(gpus=a.gpus[-1:], instances_per_gpu=1, width=W, height=H) as r:  # FPV camera
        res["fpv"] = {}
        for name, gates in L.items():
            gp, gy = np.array([g[0] for g in gates]), np.array([g[1] for g in gates])
            r.set_gates([0], gp[None], gy[None])
            rows = []
            for g in range(6):
                n = np.array([math.cos(gy[g]), math.sin(gy[g]), 0.0])
                pos = gp[g] - 1.5 * n
                q = np.array([0, 0, math.sin(gy[g] / 2), math.cos(gy[g] / 2)])
                f = r.render(pos[None], q[None])[0]
                pred = expected_mask(gp, gy, [g], pos, q, W, H)
                got = gate_mask(f) & cv2.dilate(pred.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                col = f[got].mean(0) if got.any() else np.zeros(3)
                rows.append({"gate": g, "pos": gp[g].round(2).tolist(), "pred_px": int(pred.sum()), "found_px": int(got.sum()),
                             "iou": round(float((got & pred).sum() / max(1, (got | pred).sum())), 3),
                             "mean_rgb": col.round(1).tolist()})
                if name.startswith("corners") and g in (0, 5):
                    tiles.append(label(f, f"fpv {name} G{g + 1} z={gp[g][2]:.1f}"))
            res["fpv"][name] = rows
    allrows = [x for rows in res["fpv"].values() for x in rows]
    lum = np.array([np.mean(x["mean_rgb"]) for x in allrows])
    res["fpv_summary"] = {"gates": len(allrows), "iou_min": min(x["iou"] for x in allrows),
                          "iou_median": float(np.median([x["iou"] for x in allrows])),
                          "gate_brightness_min_median_max": [round(float(v), 1) for v in (lum.min(), np.median(lum), lum.max())]}
    res["overview_iou_min"] = min(v["iou"] for v in res["overview"].values())
    print("FPV", json.dumps(res["fpv_summary"]), "OVERVIEW_IOU_MIN", res["overview_iou_min"], flush=True)
    cols = 4
    rows_ = -(-len(tiles) // cols)
    sheet = np.zeros((rows_ * H // 2, cols * W // 2, 3), np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet[rr * H // 2:(rr + 1) * H // 2, cc * W // 2:(cc + 1) * W // 2] = cv2.resize(t, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "clear_volume_sheet.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / "clear_volume.json").write_text(json.dumps(res, indent=1, default=str))
    print("WROTE", OUT / "clear_volume.json", "CLEAR", res["geometry"]["clear"])


if __name__ == "__main__":
    main()
