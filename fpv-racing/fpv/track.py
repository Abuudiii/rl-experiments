"""Race circuit and gate geometry.

World frame: right-handed, x forward, y left, z up, metres. Yaw is about +z, 0 = facing +x.
A gate is ``{"pos": [x, y, z], "yaw": rad}``: ``pos`` is the centre of a square opening
(`GATE_OPENING` m), ``yaw`` is the direction of travel through it. The frame around the opening has
outer size `GATE_OUTER` m. The circuit is a closed loop flown G1..G6 and back to G1.

All helpers work on numpy and jax arrays alike and broadcast over leading axes, so one call covers
(n_envs, n_gates) per-environment layouts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GATE_OPENING = 0.4  # side of the square opening [m]
GATE_OUTER = 0.72  # outer side of the gate frame [m]

CIRCUIT = [
    {"pos": [2, 0, 1.0], "yaw": 0},
    {"pos": [5, 0.5, 1.3], "yaw": 0.5},
    {"pos": [6.5, 3, 1.0], "yaw": 1.571},
    {"pos": [5, 5.5, 1.4], "yaw": 2.6},
    {"pos": [2, 5.5, 1.0], "yaw": 3.142},
    {"pos": [0, 3, 1.2], "yaw": -1.571},
]

# FPV camera: at the drone centre, looking along body +x, pitched up, pinhole 16:9.
CAMERA = {"pitch_up_deg": 20.0, "hfov_deg": 110.0, "aspect": 16 / 9}


def _xp(a):
    if type(a).__module__.startswith("jax"):
        import jax.numpy as jnp

        return jnp
    return np


def circuit_arrays(circuit=CIRCUIT) -> tuple[np.ndarray, np.ndarray]:
    """Gate centres (G, 3) and yaws (G,) of a circuit as float32 numpy arrays."""
    pos = np.array([g["pos"] for g in circuit], dtype=np.float32)
    yaw = np.array([g["yaw"] for g in circuit], dtype=np.float32)
    return pos, yaw


def gate_axes(yaw):
    """Unit axes of a gate in the world frame: travel direction and left, each (..., 3). Up is +z."""
    xp = _xp(yaw)
    c, s, z = xp.cos(yaw), xp.sin(yaw), xp.zeros_like(yaw)
    return xp.stack([c, s, z], axis=-1), xp.stack([-s, c, z], axis=-1)


def gate_corners(pos, yaw, size: float = GATE_OPENING):
    """Corners of the square of side ``size`` centred on the gate, world frame, shape (..., 4, 3).

    Order as seen when flying through in the travel direction: top-left, top-right, bottom-right,
    bottom-left.
    """
    xp = _xp(pos)
    _, left = gate_axes(yaw)
    h = size / 2
    up = xp.asarray([0.0, 0.0, 1.0], dtype=pos.dtype)
    signs = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # (left, up)
    return xp.stack([pos + sl * h * left + su * h * up for sl, su in signs], axis=-2)


def gate_crossing(p0, p1, pos, yaw, half_open: float = GATE_OPENING / 2, half_frame: float = GATE_OUTER / 2):
    """Test the segment p0 -> p1 against the gate plane.

    Args:
        p0, p1: Segment end points (..., 3), previous and current drone position.
        pos, yaw: Gate centres (..., 3) and yaws (...,), broadcastable against p0 (add a gate axis
            to the points for per-environment layouts: ``p0[:, None, :]`` vs ``pos (N, G, 3)``).
        half_open: Half side of the region counted as the opening (shrink by the drone radius).
        half_frame: Half outer side of the region counted as the frame (grow by the drone radius).

    Returns:
        (passed, wrong_way, hit_frame) boolean arrays: crossed the plane inside the opening in the
        travel direction / inside the opening against the travel direction / on the frame (either
        direction).
    """
    xp = _xp(p0)
    normal, left = gate_axes(yaw)
    d0, d1 = p0 - pos, p1 - pos
    s0, s1 = (d0 * normal).sum(-1), (d1 * normal).sum(-1)
    fwd, bwd = (s0 < 0) & (s1 >= 0), (s0 >= 0) & (s1 < 0)
    denom = s0 - s1
    alpha = s0 / xp.where(xp.abs(denom) > 1e-9, denom, 1.0)
    d = d0 + alpha[..., None] * (d1 - d0)  # intersection with the gate plane, relative to centre
    r = xp.maximum(xp.abs((d * left).sum(-1)), xp.abs(d[..., 2]))
    inside, frame = r < half_open, (r >= half_open) & (r < half_frame)
    return fwd & inside, bwd & inside, (fwd | bwd) & frame


def track_dict(circuit=CIRCUIT) -> dict:
    pos, yaw = circuit_arrays(circuit)
    return {
        "convention": "world right-handed, x forward, y left, z up, metres; yaw about +z, 0 = +x; "
        "gate pos = centre of the opening, yaw = direction of travel; closed loop in list order",
        "gate_opening_m": GATE_OPENING,
        "gate_outer_m": GATE_OUTER,
        "camera": CAMERA,
        "gates": [
            {
                "pos": [float(v) for v in g["pos"]],
                "yaw": float(g["yaw"]),
                "opening_corners": gate_corners(pos[i], yaw[i]).round(5).tolist(),
                "outer_corners": gate_corners(pos[i], yaw[i], GATE_OUTER).round(5).tolist(),
            }
            for i, g in enumerate(circuit)
        ],
    }


def write_track_json(path: str | Path, circuit=CIRCUIT) -> Path:
    """Write the circuit (gates, sizes, corner positions, camera) so other tools load one layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(track_dict(circuit), indent=2))
    return path


def load_track_json(path: str | Path) -> list[dict]:
    """Gates of a ``track.json`` as a list of ``{"pos", "yaw"}``."""
    return [{"pos": g["pos"], "yaw": g["yaw"]} for g in json.loads(Path(path).read_text())["gates"]]


def camera_half_fov_tan() -> tuple[float, float]:
    """Tangents of the horizontal and vertical half field of view of the FPV camera."""
    th = math.tan(math.radians(CAMERA["hfov_deg"] / 2))
    return th, th / CAMERA["aspect"]


if __name__ == "__main__":
    print(write_track_json(Path(__file__).parent / "track.json"))
