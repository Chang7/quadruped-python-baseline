
from __future__ import annotations

import numpy as np
import mujoco

from phases.mujoco_phase3_helpers import swing_phase_s
from phases.mujoco_phase5_helpers import (
    smoothstep01,
    is_rear_leg,
    swing_profiles_phase5,
    _leg_specific_params,
    _clip_targets_to_ctrlrange,
)


def build_ctrl_targets_phase10(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings,
    home_ctrl: np.ndarray,
    scheduled_contact: np.ndarray,
    actual_contact: np.ndarray,
    cfg,
    clearance: float,
    step_len_front: float,
    rear_step_scale: float,
    touchdown_depth_front: float,
    touchdown_depth_rear: float,
    touchdown_forward_front: float,
    touchdown_forward_rear: float,
    touchdown_search_window_front: float,
    touchdown_search_window_rear: float,
    rear_stance_press: float = 0.015,
    rear_stance_back: float = 0.008,
    front_stance_unload: float = 0.004,
) -> np.ndarray:
    """Phase-10 low-level target builder.

    Difference from phase-5:
      - swing and touchdown-search are retained
      - once actual contact is established in stance:
          * rear legs keep pressing slightly DOWN and BACK
          * front legs unload slightly UP
    This is designed for A1 position-servo actuators, where target=current_q
    tends to produce little active load transfer after first touchdown.
    """
    if home_ctrl.size == 0:
        return home_ctrl

    ctrl = home_ctrl.copy()
    t = float(d.time)

    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if not leg.actuator_ids or leg.home_joint_qpos is None:
            continue

        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        dq_up = getattr(leg, 'swing_dq_up', np.zeros_like(current_q))
        dq_fwd = getattr(leg, 'swing_dq_fwd', np.zeros_like(current_q))

        step_len_leg, td_depth_leg, td_fwd_leg, td_window_leg = _leg_specific_params(
            leg_idx,
            step_len_front,
            rear_step_scale,
            touchdown_depth_front,
            touchdown_depth_rear,
            touchdown_forward_front,
            touchdown_forward_rear,
            touchdown_search_window_front,
            touchdown_search_window_rear,
        )

        if not bool(scheduled_contact[leg_idx]):
            if getattr(leg, 'swing_anchor_q', None) is None:
                leg.swing_anchor_q = current_q.copy()

            s = swing_phase_s(t, leg_idx, cfg)
            x_prof, z_lift, z_drop = swing_profiles_phase5(s)
            target = (
                leg.swing_anchor_q
                + clearance * z_lift * dq_up
                + step_len_leg * x_prof * dq_fwd
                - 0.45 * td_depth_leg * z_drop * dq_up
            )
            target = 0.92 * target + 0.08 * leg.home_joint_qpos

        elif bool(actual_contact[leg_idx]):
            if is_rear_leg(leg_idx):
                # Once rear foot touches, continue pressing it DOWN and slightly BACK
                # to encourage real load transfer and forward support.
                target = current_q - rear_stance_press * dq_up - rear_stance_back * dq_fwd
                target = 0.98 * target + 0.02 * leg.home_joint_qpos
            else:
                # Front legs carry too much load in the current setup.
                # Bias them slightly upward to help shift support rearward.
                target = current_q + front_stance_unload * dq_up
                target = 0.98 * target + 0.02 * leg.home_joint_qpos

        else:
            if getattr(leg, 'stance_anchor_q', None) is None:
                leg.stance_anchor_q = current_q.copy()
            if getattr(leg, 'stance_start_time', None) is None:
                leg.stance_start_time = t

            tau = smoothstep01((t - float(leg.stance_start_time)) / max(td_window_leg, 1e-6))
            target = (
                leg.stance_anchor_q
                + td_fwd_leg * (0.20 + 0.80 * tau) * dq_fwd
                - td_depth_leg * (0.35 + 0.65 * tau) * dq_up
            )
            target = 0.95 * target + 0.05 * leg.home_joint_qpos

        target = _clip_targets_to_ctrlrange(m, leg.actuator_ids, target)
        ctrl[np.asarray(leg.actuator_ids, dtype=int)] = target

    return ctrl
