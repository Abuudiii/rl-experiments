"""Race tracks and gate geometry.

World frame: right-handed, x forward, y left, z up, metres. Yaw is about +z, 0 = facing +x.
A gate is ``{"pos": [x, y, z], "yaw": rad}``: ``pos`` is the centre of a square opening
(`GATE_OPENING` m), ``yaw`` is the direction of travel through it. The frame around the opening has
outer size `GATE_OUTER` m. A track is a closed loop of 6 gates flown G1..G6 and back to G1.
`CIRCUIT` is the original track; `SCENARIOS` holds it with five more named layouts (held-out tests
for teachers trained on generated tracks, see `fpv.env`); `track_features` / `track_valid` measure
feasibility (spacing, turn angles against the camera field of view, clearance, ...).

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


def reverse_loop(gates: list[dict]) -> list[dict]:
    """The same gates flown the other way round, starting at the same first gate."""
    order = [gates[0], *gates[:0:-1]]
    return [{"pos": list(g["pos"]), "yaw": math.remainder(g["yaw"] + math.pi, 2 * math.pi)} for g in order]


# Named layouts, all inside the contract volume and satisfying `track_valid` (worst cases in
# `track_report`). They are not part of the training distribution of generated tracks.
SCENARIOS = {
    "circuit": CIRCUIT,
    "clockwise": reverse_loop(CIRCUIT),
    # Two lobes joined by two diagonals that cross over the centre 0.4 m apart in height; the right
    # lobe (G1-G3) turns clockwise, the left lobe (G4-G6) counter-clockwise.
    "figure8": [
        {"pos": [5.9, 5.1, 1.5], "yaw": -0.06},
        {"pos": [7.7, 2.85, 1.7], "yaw": -1.55},
        {"pos": [5.85, 0.4, 1.9], "yaw": -3.08},
        {"pos": [1.1, 5.1, 1.3], "yaw": -3.082},
        {"pos": [-0.7, 2.85, 1.0], "yaw": -1.592},
        {"pos": [1.15, 0.4, 0.9], "yaw": -0.062},
    ],
    # G1-G3 face +x, offset 1.4 m sideways and 0.6 m in height alternately: the path weaves left-right
    # (and up-down) through them, then returns counter-clockwise through three 90-degree turns. A closed
    # 6-gate loop must turn 360 deg, at most about 110 deg per leg with the next gate in view, so at
    # most two consecutive legs can be net-zero weaves.
    "slalom": [
        {"pos": [0.58, 0.3, 1.0], "yaw": 0.0},
        {"pos": [3.48, 1.7, 1.6], "yaw": 0.0},
        {"pos": [6.38, 0.3, 1.0], "yaw": 0.0},
        {"pos": [7.94, 1.86, 1.3], "yaw": 1.571},
        {"pos": [3.48, 6.31, 1.6], "yaw": 3.142},
        {"pos": [-0.98, 1.86, 1.3], "yaw": -1.571},
    ],
    # Stadium, counter-clockwise: climb 0.8 -> 2.45 m over the first straight and turn, dive 2.45 ->
    # 1.0 m along the back straight (slope 0.32).
    "climb_dive": [
        {"pos": [0.9, 0.0, 0.8], "yaw": 0.0},
        {"pos": [5.4, 0.0, 1.5], "yaw": 0.0},
        {"pos": [7.24, 1.84, 2.1], "yaw": 1.571},
        {"pos": [5.4, 3.68, 2.45], "yaw": 3.142},
        {"pos": [0.9, 3.68, 1.0], "yaw": 3.142},
        {"pos": [-0.94, 1.84, 0.8], "yaw": -1.571},
    ],
    # Clockwise: 5.1 m straight G1 -> G2, a 180-degree hairpin through G3 (legs of 2.1 m), a weave
    # down to G5 and a wide turn back up through G6.
    "sprint_hairpin": [
        {"pos": [0.7, 6.4, 1.2], "yaw": 0.0},
        {"pos": [5.8, 6.4, 1.2], "yaw": 0.0},
        {"pos": [7.28, 4.92, 1.4], "yaw": -1.571},
        {"pos": [5.8, 3.43, 1.6], "yaw": 3.142},
        {"pos": [2.3, 0.9, 1.4], "yaw": 3.052},
        {"pos": [-0.85, 4.8, 1.3], "yaw": 1.571},
    ],
}


def resolve(layout) -> list[dict]:
    """A layout given as a name in `SCENARIOS` or as a list of gates -> list of gates."""
    if isinstance(layout, str):
        if layout not in SCENARIOS:
            raise KeyError(f"unknown scenario {layout!r}; known: {', '.join(SCENARIOS)}")
        return SCENARIOS[layout]
    gates = [{"pos": [float(v) for v in g["pos"]], "yaw": float(g["yaw"])} for g in layout]
    assert len(gates) == len(CIRCUIT), f"a track has {len(CIRCUIT)} gates, got {len(gates)}"
    return gates


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


def _gates_json(circuit) -> list[dict]:
    pos, yaw = circuit_arrays(circuit)
    return [
        {
            "pos": [float(v) for v in g["pos"]],
            "yaw": float(g["yaw"]),
            "opening_corners": gate_corners(pos[i], yaw[i]).round(5).tolist(),
            "outer_corners": gate_corners(pos[i], yaw[i], GATE_OUTER).round(5).tolist(),
        }
        for i, g in enumerate(circuit)
    ]


def track_dict(circuit=CIRCUIT, scenarios: dict | None = SCENARIOS) -> dict:
    out = {
        "convention": "world right-handed, x forward, y left, z up, metres; yaw about +z, 0 = +x; "
        "gate pos = centre of the opening, yaw = direction of travel; closed loop in list order",
        "gate_opening_m": GATE_OPENING,
        "gate_outer_m": GATE_OUTER,
        "camera": CAMERA,
        "gates": _gates_json(circuit),
    }
    if scenarios:
        out["volume"] = {"lo": list(VOLUME_LO), "hi": list(VOLUME_HI)}
        out["scenarios"] = {name: {"gates": _gates_json(g)} for name, g in scenarios.items()}
    return out


def write_track_json(path: str | Path, circuit=CIRCUIT, scenarios: dict | None = SCENARIOS) -> Path:
    """Write the circuit (``gates``: sizes, corner positions, camera) and the named ``scenarios`` in
    the same per-gate format, so other tools load the same layouts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(track_dict(circuit, scenarios), indent=2))
    return path


def load_track_json(path: str | Path, scenario: str | None = None) -> list[dict]:
    """Gates of a ``track.json`` (the circuit, or the named scenario) as a list of ``{"pos", "yaw"}``."""
    d = json.loads(Path(path).read_text())
    gates = d["gates"] if scenario is None else d["scenarios"][scenario]["gates"]
    return [{"pos": g["pos"], "yaw": g["yaw"]} for g in gates]


def camera_half_fov_tan() -> tuple[float, float]:
    """Tangents of the horizontal and vertical half field of view of the FPV camera."""
    th = math.tan(math.radians(CAMERA["hfov_deg"] / 2))
    return th, th / CAMERA["aspect"]


# region track geometry: feasibility measures, shared by the scenarios and the generator

# Contract volume for gate centres [m] (world frame above)
VOLUME_LO = (-1.0, -1.5, 0.7)
VOLUME_HI = (8.0, 7.0, 2.5)

# Feasibility limits of a 6-gate loop for a Crazyflie-class drone at about 4 m/s whose camera must keep
# the next gate in view. Angles: exit = bearing to the next gate minus the gate's yaw (horizontal
# angle at which the next gate appears right after passing), entry = next gate's yaw minus that
# bearing. Reference path of a leg: cubic Hermite curve leaving along the gate's yaw and arriving
# along the next gate's yaw (tangent length = horizontal chord), height linear.
LIMITS = dict(
    spacing_min=2.0,  # m, consecutive gate centres (3D)
    spacing_max=7.0,
    exit_max=math.radians(58.0),  # next gate inside the 110 deg camera when leaving a gate
    entry_max=math.radians(60.0),
    radius_min=0.9,  # m, horizontal radius of curvature of the reference path
    slope_max=0.5,  # |dz| / horizontal distance of a leg
    clearance_min=0.9,  # m, reference path / 2 m run-up of a gate to every other gate centre
    separation_min=1.5,  # m, any two gate centres
    view_min=0.85,  # fraction of a leg's reference path with the next gate in the camera
)
CRUISE_PITCH_DEG = 10.0  # nose-down attitude assumed for the view measure (camera axis 10 deg up)
_N_LEG = 16  # samples per leg


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def leg_paths(pos, yaw):
    """Reference path of every leg i -> i+1: points, first and second derivative, each (..., G, K, 3)."""
    xp = _xp(pos)
    p0, p1 = pos, xp.roll(pos, -1, axis=-2)
    y0, y1 = yaw, xp.roll(yaw, -1, axis=-1)
    d = p1 - p0
    L = xp.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    m0 = xp.stack([L * xp.cos(y0), L * xp.sin(y0), d[..., 2]], -1)
    m1 = xp.stack([L * xp.cos(y1), L * xp.sin(y1), d[..., 2]], -1)
    t = xp.linspace(0.0, 1.0, _N_LEG, dtype=pos.dtype)[:, None]
    p0, p1, m0, m1 = (a[..., None, :] for a in (p0, p1, m0, m1))
    pt = (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1
    v = (6 * t**2 - 6 * t) * p0 + (3 * t**2 - 4 * t + 1) * m0 + (-6 * t**2 + 6 * t) * p1 + (3 * t**2 - 2 * t) * m1
    acc = (12 * t - 6) * p0 + (6 * t - 4) * m0 + (-12 * t + 6) * p1 + (6 * t - 2) * m1
    return pt, v, acc


def track_features(pos, yaw) -> dict:
    """Feasibility measures of closed loops, numpy or jax, batched over leading axes.

    Args: pos (..., G, 3) gate centres, yaw (..., G). Per-leg entries refer to leg i -> i+1 (mod G).
    Returns dict: spacing, exit, entry, slope, radius, view (..., G); clearance, separation (...,);
    lo, hi (..., 3) bounding box of the centres.
    """
    xp = _xp(pos)
    G = pos.shape[-2]
    p1 = xp.roll(pos, -1, axis=-2)
    d = p1 - pos
    horiz = xp.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    bearing = xp.arctan2(d[..., 1], d[..., 0])
    pt, v, acc = leg_paths(pos, yaw)
    vh = xp.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    curv = xp.abs(v[..., 0] * acc[..., 1] - v[..., 1] * acc[..., 0]) / (vh**3 + 1e-9)
    # Next gate in a camera that looks along the path tangent, pitched up by (camera - cruise) pitch
    heading = xp.arctan2(v[..., 1], v[..., 0])
    g = p1[..., None, :] - pt
    f = xp.cos(heading) * g[..., 0] + xp.sin(heading) * g[..., 1]
    lft = -xp.sin(heading) * g[..., 0] + xp.cos(heading) * g[..., 1]
    ph = math.radians(CAMERA["pitch_up_deg"] - CRUISE_PITCH_DEG)
    fc, uc = f * math.cos(ph) + g[..., 2] * math.sin(ph), -f * math.sin(ph) + g[..., 2] * math.cos(ph)
    th, tv = camera_half_fov_tan()
    seen = (fc > 0) & (xp.abs(lft) < th * fc) & (xp.abs(uc) < tv * fc)
    k_view = int(_N_LEG * 0.9)  # the last samples sit on the gate itself
    # Clearance: leg i's path and the run-up of gate i (2 m before it) against every other gate
    normal, _ = gate_axes(yaw)
    s = xp.linspace(0.0, 2.0, 9, dtype=pos.dtype)[:, None]
    runup = pos[..., None, :] - s * normal[..., None, :]  # (..., G, 9, 3)
    centres = pos[..., None, :, None, :]
    dist_leg = xp.sqrt(((pt[..., :, None, :, :] - centres) ** 2).sum(-1)).min(-1)  # (..., leg, gate)
    dist_run = xp.sqrt(((runup[..., :, None, :, :] - centres) ** 2).sum(-1)).min(-1)
    i = np.arange(G)
    own_leg = (i[:, None] == i[None, :]) | ((i[:, None] + 1) % G == i[None, :])
    own_run = i[:, None] == i[None, :]
    big = 1e3
    clear = xp.minimum(xp.where(own_leg, big, dist_leg).min((-1, -2)), xp.where(own_run, big, dist_run).min((-1, -2)))
    pair = xp.sqrt(((pos[..., :, None, :] - pos[..., None, :, :]) ** 2).sum(-1))
    sep = xp.where(i[:, None] == i[None, :], big, pair).min((-1, -2))
    return dict(
        spacing=xp.sqrt((d**2).sum(-1)),
        exit=_wrap(bearing - yaw),
        entry=_wrap(xp.roll(yaw, -1, axis=-1) - bearing),
        slope=d[..., 2] / (horiz + 1e-9),
        radius=1.0 / (curv.max(-1) + 1e-9),
        view=seen[..., :k_view].mean(-1),
        clearance=clear,
        separation=sep,
        lo=pos.min(-2),
        hi=pos.max(-2),
    )


def track_valid(f: dict, limits: dict = LIMITS):
    """Boolean (...,): every limit of `LIMITS` holds and all centres are inside the volume."""
    xp = _xp(f["spacing"])
    L = limits
    ok = (
        (f["spacing"] >= L["spacing_min"]) & (f["spacing"] <= L["spacing_max"])
        & (xp.abs(f["exit"]) <= L["exit_max"]) & (xp.abs(f["entry"]) <= L["entry_max"])
        & (f["radius"] >= L["radius_min"]) & (xp.abs(f["slope"]) <= L["slope_max"])
        & (f["view"] >= L["view_min"])
    ).all(-1)  # fmt: skip
    inside = ((f["lo"] >= xp.asarray(VOLUME_LO)) & (f["hi"] <= xp.asarray(VOLUME_HI))).all(-1)
    return ok & inside & (f["clearance"] >= L["clearance_min"]) & (f["separation"] >= L["separation_min"])


def track_report(gates) -> dict:
    """Readable feasibility summary of one layout (list of gates): worst case of every measure."""
    pos, yaw = circuit_arrays(gates)
    f = track_features(pos.astype(np.float64), yaw.astype(np.float64))
    turn = f["exit"] + f["entry"]  # heading change on each leg (left positive)
    return dict(
        valid=bool(track_valid(f)),
        spacing=[round(float(v), 2) for v in f["spacing"]],
        exit_deg=[round(float(v), 1) for v in np.degrees(f["exit"])],
        entry_deg=[round(float(v), 1) for v in np.degrees(f["entry"])],
        total_turn_deg=round(float(np.degrees(turn.sum())), 1),
        radius=[round(float(v), 2) for v in f["radius"]],
        slope=[round(float(v), 2) for v in f["slope"]],
        view=[round(float(v), 2) for v in f["view"]],
        clearance=round(float(f["clearance"]), 2),
        separation=round(float(f["separation"]), 2),
        height_span=round(float(np.ptp(pos[:, 2])), 2),
        lo=[round(float(v), 2) for v in f["lo"]], hi=[round(float(v), 2) for v in f["hi"]],
    )  # fmt: skip


# region procedural tracks (jax)


GEN_P_EIGHT = 0.35  # share of two-lobe (figure-eight) candidates at complexity 1
GEN_LOBE_DEG = 85.0  # angles of the outer two gates of a lobe around its centre, from the lobe axis
GEN_LOBE_HALF_SEP = (0.5, 1.0)  # lobe centres at +- this, in lobe radii
GEN_YAW_NOISE = 0.12  # rad, gate yaw around the bisector of the legs in and out
GEN_SIZE_FRAC = (0.7, 1.0)  # size of a loop as a fraction of the largest that fits the volume (two lobes: 0.9-1)
GEN_LEG_MIN = 2.1  # m, a candidate is grown until its shortest leg has this length (if it fits)


def sample_tracks(key, n: int, n_candidates: int = 32, complexity: float = 1.0, fallback=None, limits=LIMITS):
    """Draw ``n`` random closed 6-gate tracks inside the contract volume (jax, jittable).

    Candidates (polygon through the 6 gate centres, in order) come from two families:
      * loop (share 1 - 0.35 * complexity): gates around a centre at increasing angles with uneven
        gaps (factor 1 +- 0.5 * complexity) and radii (1 +- 0.4 * complexity). Uneven gaps give long
        straights into tight turns; alternating radii give concave corners, i.e. left-right weaves.
      * two lobes (share 0.35 * complexity): three gates clockwise around one centre (at about
        +85, 0, -85 deg from the lobe axis), then three counter-clockwise around another, centres
        1-2 lobe radii apart; the legs between the lobes cross (figure eight).
    Then: mirrored with probability 0.5 (direction of travel), squeezed across by U(0.35, 1) (at
    complexity 1), rotated at random, scaled to U(0.7, 1) (two lobes: U(0.9, 1)) of the largest
    size that fits the volume, but grown until the shortest leg is 2.1 m if that still fits, and
    placed at random in the volume. Gate yaw = bisector of the legs in and out (the direction of
    travel) + U(-0.12, 0.12) rad. Heights: a random first or second harmonic around the loop plus
    noise, span U(0, 1.8 m * complexity). complexity = 0 gives flat regular hexagons in either
    direction; intermediate values are a curriculum.

    ``n_candidates`` candidates per track; the first that satisfies `track_valid` (spacing, turn
    angles against the camera, curvature, slope, clearance of every leg and run-up from the other
    gates, separation, volume) is taken (at complexity 1 about 17% of candidates pass, so 32
    candidates fail together with probability about 0.3%). If none does, ``fallback`` (pos (n, 6, 3),
    yaw (n, 6); e.g. the previous episode's track) is used, else the last candidate. Returns
    (pos (n, 6, 3), yaw (n, 6), ok (n,) a valid candidate was found).
    """
    import jax
    import jax.numpy as jnp

    G, C, c = len(CIRCUIT), n_candidates, complexity
    assert G == 6
    ks = iter(jax.random.split(key, 24))

    def u(shape, lo=0.0, hi=1.0):
        return jax.random.uniform(next(ks), shape, minval=lo, maxval=hi)

    shp = (n, C)
    tau = 2 * math.pi
    # Loop around a centre
    w = 1 + u((*shp, G), -0.5, 0.5) * c
    ang = tau * (jnp.cumsum(w, -1) - w[..., :1]) / w.sum(-1, keepdims=True)
    rad = 1 + u((*shp, G), -0.4, 0.4) * c
    loop = jnp.stack([rad * jnp.cos(ang), rad * jnp.sin(ang)], -1)
    # Two lobes: A clockwise around (+half, 0), B counter-clockwise around (-half, 0)
    base = jnp.radians(jnp.asarray([GEN_LOBE_DEG, 0.0, -GEN_LOBE_DEG]))
    la = base + u((*shp, 3), -0.35, 0.35) * c
    lb = math.pi - base + u((*shp, 3), -0.35, 0.35) * c
    ra, rb = 1 + u((*shp, 3), -0.25, 0.25) * c, 1 + u((*shp, 3), -0.25, 0.25) * c
    half = u((*shp, 1), *GEN_LOBE_HALF_SEP)
    eight = jnp.concatenate([jnp.stack([half + ra * jnp.cos(la), ra * jnp.sin(la)], -1),
                             jnp.stack([-half + rb * jnp.cos(lb), rb * jnp.sin(lb)], -1)], -2)  # fmt: skip
    is8 = (u(shp) < GEN_P_EIGHT * c)[..., None, None]
    xy = jnp.where(is8, eight, loop)
    flip = jnp.where(u(shp) < 0.5, -1.0, 1.0)
    stretch = 1 - (1 - u(shp, 0.35, 1.0)) * c
    xy = xy * jnp.stack([jnp.ones(shp), flip * stretch], -1)[..., None, :]
    th = u((*shp, 1), -math.pi, math.pi)
    xy = jnp.stack([xy[..., 0] * jnp.cos(th) - xy[..., 1] * jnp.sin(th),
                    xy[..., 0] * jnp.sin(th) + xy[..., 1] * jnp.cos(th)], -1)  # fmt: skip
    # Gate yaw: bisector of the unit directions of the legs in and out
    d_out = jnp.roll(xy, -1, -2) - xy
    d_in = xy - jnp.roll(xy, 1, -2)
    leg = jnp.linalg.norm(d_out, axis=-1)
    bis = d_out / (leg[..., None] + 1e-9) + d_in / (jnp.linalg.norm(d_in, axis=-1, keepdims=True) + 1e-9)
    yaw = jnp.arctan2(bis[..., 1], bis[..., 0]) + u((*shp, G), -GEN_YAW_NOISE, GEN_YAW_NOISE)
    # Size and place
    lo, hi = jnp.asarray(VOLUME_LO), jnp.asarray(VOLUME_HI)
    size = hi - lo
    ext = jnp.ptp(xy, -2)  # (n, C, 2)
    fit = (size[:2] / (ext + 1e-6)).min(-1)
    size_frac = jnp.where(is8[..., 0, 0], u(shp, 0.9, 1.0), u(shp, *GEN_SIZE_FRAC))
    scale = jnp.minimum(jnp.maximum(fit * size_frac, GEN_LEG_MIN / (leg.min(-1) + 1e-6)), fit)
    xy = xy * scale[..., None, None]
    ext = ext * scale[..., None]
    xy = xy - xy.min(-2, keepdims=True) + lo[:2] + u((*shp, 1, 2)) * (size[:2] - ext)[..., None, :]
    k2 = (u((*shp, 1)) < 0.5) + 1.0
    ang = tau * k2 * jnp.arange(G) / G + u((*shp, 1), 0.0, tau)
    z = jnp.cos(ang) + u((*shp, G), -0.5, 0.5)
    span = u((*shp, 1)) * jnp.minimum(1.8 * c, size[2])
    z = (z - z.min(-1, keepdims=True)) / (jnp.ptp(z, -1, keepdims=True) + 1e-6) * span
    z = z + lo[2] + u((*shp, 1)) * (size[2] - span)
    pos = jnp.concatenate([xy, z[..., None]], -1)  # (n, C, G, 3)
    valid = track_valid(track_features(pos, yaw), limits)  # (n, C)
    ok = valid.any(-1)
    idx = jnp.where(ok, jnp.argmax(valid, -1), C - 1)
    pos = jnp.take_along_axis(pos, idx[:, None, None, None], 1)[:, 0]
    yaw = jnp.take_along_axis(yaw, idx[:, None, None], 1)[:, 0]
    if fallback is not None:
        pos, yaw = jnp.where(ok[:, None, None], pos, fallback[0]), jnp.where(ok[:, None], yaw, fallback[1])
    return pos, _wrap(yaw), ok


if __name__ == "__main__":
    print(write_track_json(Path(__file__).parent / "track.json"))
