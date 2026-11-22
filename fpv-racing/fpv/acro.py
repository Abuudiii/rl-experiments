"""Acrobatic layouts (kept out of `fpv.track.SCENARIOS`, so nothing that iterates the standard
scenarios changes). Same gate convention as `fpv.track`: ``pos`` = centre of the upright 0.4 m
opening, ``yaw`` = direction of travel; closed loop of six gates.

``split_s``: a high gate (4.5 m) heading +x; 1 m further on and 3.5 m lower, a low gate heading the
opposite way (-x), so the drone has to reverse direction while dropping 3.5 m (split-S: half-roll,
then pull through inverted); a second low gate 2.5 m behind it; then a climbing teardrop to the
south (gates at 1.5, 2.6, 3.7 m) back to the high gate. Flown with ``z_max`` = 6 m (`ACRO_ENV`);
the standard 3.5 m ceiling would end every episode at the high gate.

`register()` adds the layouts to `fpv.track.SCENARIOS` in the calling process only, so that
``EnvConfig(track="split_s")`` and checkpoints trained with it resolve; call it before building a
`RaceEnv` for these layouts. Nothing on disk or in other processes changes.

`ShapedRaceEnv`: OPTIONAL reward shaping for the dive (off unless ``w_invert`` > 0): while the next
gate is ``dive_gate`` (i.e. between passing the high gate and passing the low one), each step adds
``w_invert * max(0, -(body z . world z))``: nothing up to 90 deg of tilt, ``w_invert`` per step when
fully upside down. With ``invert_vz`` > 0 the term is further multiplied by
``clip(-vz / invert_vz, 0, 1)`` (vz = world vertical speed), so being upside down only pays while
diving (the inverted pull-through of a split-S), not while hanging at the top.
Everything else is `fpv.env.RaceEnv` unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

from fpv import track
from fpv.env import RaceEnv, quat_to_mat

LAYOUTS = {
    "split_s": [
        {"pos": [0.0, 0.0, 4.5], "yaw": 0.0},  # G1 high, heading +x
        {"pos": [1.0, 0.0, 1.0], "yaw": 3.1416},  # G2 low, heading -x, 3.5 m below
        {"pos": [-1.5, 0.0, 1.0], "yaw": 3.1416},  # G3 low, 2.5 m behind G2
        {"pos": [-4.6, -1.6, 1.5], "yaw": -1.9},  # G4 far end, turning south
        {"pos": [-2.9, -3.6, 2.6], "yaw": 0.3},  # G5 climbing, heading east
        {"pos": [-1.8, -1.5, 3.7], "yaw": 0.75},  # G6 climbing to the high gate
    ],
    # Deeper and tighter: high gate at 5.0 m, low gate only 0.3 m beyond it at 0.8 m (4.2 m drop, less
    # forward room for a sliced turn); the rest shifted to keep the spacing.
    "split_s_deep": [
        {"pos": [0.0, 0.0, 5.0], "yaw": 0.0},
        {"pos": [0.3, 0.0, 0.8], "yaw": 3.1416},
        {"pos": [-2.2, 0.0, 0.8], "yaw": 3.1416},
        {"pos": [-4.8, -1.6, 1.6], "yaw": -1.9},
        {"pos": [-3.1, -3.6, 2.9], "yaw": 0.3},
        {"pos": [-2.0, -1.5, 4.2], "yaw": 0.75},
    ],
}


class ShapedRaceEnv(RaceEnv):
    """`RaceEnv` plus the optional inversion term (module docstring)."""

    def __init__(self, cfg, circuit=None, device=None, w_invert: float = 0.0, dive_gate: int = 1,
                 invert_vz: float = 0.0):
        super().__init__(cfg, circuit=circuit, device=device)
        self.w_invert, self.dive_gate, self.invert_vz = float(w_invert), int(dive_gate), float(invert_vz)

    def step_core(self, state, action):
        target0 = state.target
        state, reward, events = super().step_core(state, action)
        if self.w_invert > 0:
            cz = quat_to_mat(state.data.states.quat[:, 0])[:, 2, 2]
            bonus = self.w_invert * jnp.maximum(-cz, 0.0) * (target0 == self.dive_gate) * ~events["terminated"]
            if self.invert_vz > 0:
                bonus = bonus * jnp.clip(-state.data.states.vel[:, 0, 2] / self.invert_vz, 0.0, 1.0)
            reward = reward + jnp.nan_to_num(bonus)
            state = state.replace(ep_return=state.ep_return + jnp.nan_to_num(bonus))
        return state, reward, events


# Environment options the layouts need on top of a config (out-of-bounds ceiling above the high gate)
ACRO_ENV = {"split_s": dict(z_max=6.0), "split_s_deep": dict(z_max=6.5)}


def register():
    """Make the acro layouts resolvable by name in this process (fpv.track.resolve / RaceEnv)."""
    for name, gates in LAYOUTS.items():
        track.SCENARIOS.setdefault(name, gates)
