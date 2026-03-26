
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import mujoco

from baseline.config import MPCConfig
from phases.mujoco_phase2_helpers import (
    LegBinding,
    ModelBindings,
    discover_model_bindings,
    foot_point_world,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    print_binding_summary,
)

FALLBACK_FOOT_LOCAL_OFFSET = np.array([0.0, 0.0, -0.20], dtype=float)


def cycle_time(cfg: MPCConfig) -> float:
    return cfg.stance_time + cfg.swing_time


def stance_fraction(cfg: MPCConfig) -> float:
    return cfg.stance_time / cycle_time(cfg)


def leg_phase_offset(leg_idx: int) -> float:
    # FL/RR together, FR/RL together
    offsets = [0.0, 0.5, 0.5, 0.0]
    return offsets[leg_idx]


def swing_phase_s(t: float, leg_idx: int, cfg: MPCConfig) -> float:
    cyc = cycle_time(cfg)
    st_frac = stance_fraction(cfg)
    phase = ((t / cyc) + leg_phase_offset(leg_idx)) % 1.0
    if phase < st_frac:
        return 0.0
    return float((phase - st_frac) / max(1.0 - st_frac, 1e-9))


def _leg_jacobian_at_point(m: mujoco.MjModel, d: mujoco.MjData, leg: LegBinding) -> np.ndarray:
    point_world = foot_point_world(d, leg, FALLBACK_FOOT_LOCAL_OFFSET)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point_world, leg.calf_body_id)
    return jacp[:, np.asarray(leg.dof_adrs, dtype=int)]


def _safe_leg_pinv(Jleg: np.ndarray) -> np.ndarray:
    # 3x3 in A1, but use pinv for robustness
    return np.linalg.pinv(Jleg, rcond=1e-4)


def compute_swing_delta_maps(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings: ModelBindings,
) -> None:
    """Store joint-space maps for +up and +forward foot motion.

    Each map converts a desired Cartesian foot displacement in meters into a
    joint displacement vector in actuator-space radians using dq = J^+ dp.
    """
    base_R = np.asarray(d.body(bindings.base_body_name).xmat, dtype=float).reshape(3, 3)
    e_fwd_world = base_R[:, 0].copy()
    e_up_world = np.array([0.0, 0.0, 1.0], dtype=float)

    for leg in bindings.leg_bindings:
        Jleg = _leg_jacobian_at_point(m, d, leg)
        Jpinv = _safe_leg_pinv(Jleg)
        dq_up = Jpinv @ e_up_world
        dq_fwd = Jpinv @ e_fwd_world

        # Mild regularization: cap absurd gains produced by near-singular poses.
        dq_up = np.clip(dq_up, -25.0, 25.0)
        dq_fwd = np.clip(dq_fwd, -25.0, 25.0)

        # Store on the binding object dynamically.
        leg.swing_dq_up = dq_up.astype(float)
        leg.swing_dq_fwd = dq_fwd.astype(float)
        leg.swing_anchor_q = None


def resolve_home_ctrl(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    if m.nu == 0:
        return np.zeros(0, dtype=float)
    if d.qpos.shape[0] >= 7 + m.nu:
        return np.asarray(d.qpos[7 : 7 + m.nu], dtype=float).copy()
    return np.asarray(d.ctrl, dtype=float).copy()


def _clip_targets_to_ctrlrange(m: mujoco.MjModel, actuator_ids: list[int], target: np.ndarray) -> np.ndarray:
    out = np.asarray(target, dtype=float).copy()
    for i, aid in enumerate(actuator_ids):
        if 0 <= aid < m.nu:
            lo, hi = np.asarray(m.actuator_ctrlrange[aid], dtype=float)
            if hi > lo:
                out[i] = float(np.clip(out[i], lo, hi))
    return out


def update_swing_anchors(d: mujoco.MjData, bindings: ModelBindings, prev_sched: np.ndarray, sched: np.ndarray) -> None:
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        is_swing = not bool(sched[leg_idx])
        was_stance = bool(prev_sched[leg_idx])
        if is_swing and was_stance:
            leg.swing_anchor_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        elif leg.swing_anchor_q is None and is_swing:
            leg.swing_anchor_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)


def build_ctrl_targets_phase3(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings: ModelBindings,
    home_ctrl: np.ndarray,
    scheduled_contact: np.ndarray,
    cfg: MPCConfig,
    clearance: float,
    step_len: float,
) -> np.ndarray:
    if home_ctrl.size == 0:
        return home_ctrl

    ctrl = home_ctrl.copy()
    t = float(d.time)

    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if not leg.actuator_ids or leg.home_joint_qpos is None:
            continue

        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        if bool(scheduled_contact[leg_idx]):
            # Keep stance actuators from fighting the externally applied GRF.
            target = current_q
        else:
            s = swing_phase_s(t, leg_idx, cfg)
            if leg.swing_anchor_q is None:
                leg.swing_anchor_q = current_q.copy()

            dq_up = getattr(leg, 'swing_dq_up', np.zeros_like(current_q))
            dq_fwd = getattr(leg, 'swing_dq_fwd', np.zeros_like(current_q))

            z_profile = math.sin(math.pi * s)                # 0 -> 1 -> 0
            x_profile = 0.5 - 0.5 * math.cos(math.pi * s)    # 0 -> 1
            target = leg.swing_anchor_q + clearance * z_profile * dq_up + step_len * x_profile * dq_fwd

            # Small return-to-home regularizer on ab/ad joints to reduce sideways drift.
            target = 0.90 * target + 0.10 * leg.home_joint_qpos

        target = _clip_targets_to_ctrlrange(m, leg.actuator_ids, target)
        ctrl[np.asarray(leg.actuator_ids, dtype=int)] = target

    return ctrl
