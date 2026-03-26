from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from phases.mujoco_phase2_helpers import foot_point_world
from phases.mujoco_phase3_helpers import FALLBACK_FOOT_LOCAL_OFFSET, compute_swing_delta_maps, swing_phase_s
from phases.mujoco_phase5_helpers import _clip_targets_to_ctrlrange, smoothstep01


def front_sign(leg_idx: int) -> float:
    return 1.0 if leg_idx in (0, 1) else -1.0


def side_sign(leg_idx: int) -> float:
    return 1.0 if leg_idx in (0, 2) else -1.0


def is_rear(leg_idx: int) -> bool:
    return leg_idx in (2, 3)


def init_phase11_state(m: mujoco.MjModel, d: mujoco.MjData, bindings) -> None:
    compute_swing_delta_maps(m, d, bindings)
    _compute_side_delta_maps(m, d, bindings)

    for leg in bindings.leg_bindings:
        leg.phase11_stance_anchor_world = foot_point_world(d, leg, FALLBACK_FOOT_LOCAL_OFFSET).copy()
        leg.phase11_swing_start_world = None
        leg.phase11_swing_target_world = None
        leg.phase11_touchdown_target_world = None
        leg.phase11_stance_start_time = 0.0
        leg.phase11_swing_start_time = None
        leg.phase11_touchdown_recorded = False
        leg.phase11_touchdown_delays = []
        leg.phase11_sched_stance_samples = 0
        leg.phase11_actual_contact_samples = 0
        leg.phase11_force_enabled_samples = 0
        leg.phase11_support_contact_samples = 0
        leg.phase11_last_q_target = None


def _compute_side_delta_maps(m: mujoco.MjModel, d: mujoco.MjData, bindings) -> None:
    base_R = np.asarray(d.body(bindings.base_body_name).xmat, dtype=float).reshape(3, 3)
    e_side_world = base_R[:, 1].copy()
    for leg in bindings.leg_bindings:
        Jleg = leg_jacobian_world(m, d, leg)
        Jpinv = np.linalg.pinv(Jleg, rcond=1e-4)
        dq_side = Jpinv @ e_side_world
        dq_side = np.clip(dq_side, -25.0, 25.0)
        leg.swing_dq_side = dq_side


def leg_jacobian_world(m: mujoco.MjModel, d: mujoco.MjData, leg) -> np.ndarray:
    point_world = foot_point_world(d, leg, FALLBACK_FOOT_LOCAL_OFFSET)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point_world, leg.calf_body_id)
    return jacp[:, np.asarray(leg.dof_adrs, dtype=int)]


def damped_ls_step(Jleg: np.ndarray, err: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    J = np.asarray(Jleg, dtype=float)
    e = np.asarray(err, dtype=float)
    A = J @ J.T + damping * np.eye(3)
    return J.T @ np.linalg.solve(A, e)


def nominal_foothold_world(d: mujoco.MjData, bindings, cfg, leg_idx: int, step_len_front: float, rear_step_scale: float,
                           gait_alpha: float, vx_err: float) -> np.ndarray:
    body = np.asarray(cfg.nominal_footholds_body[leg_idx], dtype=float).copy()
    step_len = float(step_len_front * (rear_step_scale if is_rear(leg_idx) else 1.0))
    # Mild Raibert-like forward placement using desired forward velocity error.
    body[0] += gait_alpha * (step_len + 0.06 * np.clip(vx_err, -0.35, 0.35))
    base = d.body(bindings.base_body_name)
    R = np.asarray(base.xmat, dtype=float).reshape(3, 3)
    p = np.asarray(base.xpos, dtype=float)
    return p + R @ body


def swing_target_world(start_world: np.ndarray, end_world: np.ndarray, s: float, clearance: float, touchdown_drop: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    p = (1.0 - s) * np.asarray(start_world, dtype=float) + s * np.asarray(end_world, dtype=float)
    lift = float(clearance) * 4.0 * s * (1.0 - s)
    late = smoothstep01((s - 0.68) / 0.32)
    p[2] = (1.0 - s) * start_world[2] + s * end_world[2] + lift - touchdown_drop * late
    return p


def build_ctrl_targets_phase11(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings,
    cfg,
    home_ctrl: np.ndarray,
    scheduled_contact: np.ndarray,
    actual_contact: np.ndarray,
    u_hold: np.ndarray,
    settle_time: float,
    gait_ramp_time: float,
    nominal_trunk_height: float,
    step_len_front: float,
    rear_step_scale: float,
    clearance: float,
    touchdown_depth_front: float,
    touchdown_depth_rear: float,
    touchdown_window_front: float,
    touchdown_window_rear: float,
    stance_press_front: float,
    stance_press_rear: float,
    rear_back_bias: float,
    front_unload: float,
    height_k: float,
    pitch_k: float,
    roll_k: float,
    speed_back_k: float,
    dq_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    if home_ctrl.size == 0:
        return home_ctrl.copy(), np.zeros(4, dtype=bool)

    ctrl = home_ctrl.copy()
    stance_enable = np.zeros(4, dtype=bool)

    base = d.body(bindings.base_body_name)
    base_R = np.asarray(base.xmat, dtype=float).reshape(3, 3)
    base_p = np.asarray(base.xpos, dtype=float).copy()
    vx = float(d.qvel[0])
    roll, pitch, yaw = _mat_to_rpy(base_R)
    height_err = float(nominal_trunk_height - base_p[2])

    gait_alpha = 0.0 if d.time < settle_time else smoothstep01((float(d.time) - settle_time) / max(gait_ramp_time, 1e-6))
    vx_err = float(cfg.desired_speed - vx)

    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if not leg.actuator_ids or leg.home_joint_qpos is None:
            continue

        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        current_p = foot_point_world(d, leg, FALLBACK_FOOT_LOCAL_OFFSET)
        dq_up = getattr(leg, 'swing_dq_up', np.zeros_like(current_q))
        dq_fwd = getattr(leg, 'swing_dq_fwd', np.zeros_like(current_q))
        dq_side = getattr(leg, 'swing_dq_side', np.zeros_like(current_q))
        load_ratio = float(np.clip(u_hold[3 * leg_idx + 2] / max(cfg.fz_max, 1e-6), 0.0, 1.0)) if u_hold.size >= 3 * leg_idx + 3 else 0.0

        if bool(scheduled_contact[leg_idx]):
            leg.phase11_sched_stance_samples += 1
            td_depth = touchdown_depth_rear if is_rear(leg_idx) else touchdown_depth_front
            td_window = touchdown_window_rear if is_rear(leg_idx) else touchdown_window_front

            if bool(actual_contact[leg_idx]):
                leg.phase11_actual_contact_samples += 1
                if leg.phase11_stance_anchor_world is None:
                    leg.phase11_stance_anchor_world = current_p.copy()
                if not leg.phase11_touchdown_recorded and leg.phase11_stance_start_time is not None:
                    leg.phase11_touchdown_delays.append(float(d.time) - float(leg.phase11_stance_start_time))
                    leg.phase11_touchdown_recorded = True

            if leg.phase11_stance_anchor_world is None:
                # Touchdown search: keep reaching toward the touchdown target, then lower.
                target = leg.phase11_touchdown_target_world
                if target is None:
                    target = nominal_foothold_world(d, bindings, cfg, leg_idx, step_len_front, rear_step_scale, gait_alpha, vx_err)
                tau = 0.0
                if leg.phase11_stance_start_time is not None:
                    tau = smoothstep01((float(d.time) - float(leg.phase11_stance_start_time)) / max(td_window, 1e-6))
                p_des = np.asarray(target, dtype=float).copy()
                p_des[2] -= td_depth * (0.30 + 0.70 * tau) + 0.50 * max(height_err, 0.0)
                if is_rear(leg_idx):
                    p_des[0] -= 0.50 * rear_back_bias * gait_alpha
                else:
                    p_des[2] += 0.40 * front_unload * gait_alpha
            else:
                p_des = np.asarray(leg.phase11_stance_anchor_world, dtype=float).copy()
                # Whole-body stabilization in foot space.
                press = (stance_press_rear if is_rear(leg_idx) else stance_press_front)
                press *= gait_alpha * (0.45 + 0.55 * load_ratio)
                p_des[2] -= height_k * height_err
                p_des[2] -= pitch_k * pitch * front_sign(leg_idx)
                p_des[2] -= roll_k * roll * side_sign(leg_idx)
                p_des[2] -= press
                if is_rear(leg_idx):
                    p_des[0] -= gait_alpha * (rear_back_bias + speed_back_k * max(vx_err, 0.0))
                else:
                    p_des[2] += front_unload * gait_alpha
                stance_enable[leg_idx] = True

            Jleg = leg_jacobian_world(m, d, leg)
            err = p_des - current_p
            dq = damped_ls_step(Jleg, err, damping=1e-3)
            dq = np.clip(dq, -dq_limit, dq_limit)
            q_target = current_q + dq
            q_target = 0.92 * q_target + 0.08 * leg.home_joint_qpos
        else:
            # Swing phase: move foot from liftoff point to a nominal forward target.
            s = swing_phase_s(max(float(d.time) - settle_time, 0.0), leg_idx, cfg)
            target_world = leg.phase11_swing_target_world
            if target_world is None:
                target_world = nominal_foothold_world(d, bindings, cfg, leg_idx, step_len_front, rear_step_scale, gait_alpha, vx_err)
            start_world = leg.phase11_swing_start_world if leg.phase11_swing_start_world is not None else current_p.copy()
            td_drop = (touchdown_depth_rear if is_rear(leg_idx) else touchdown_depth_front) * 0.35 * gait_alpha
            p_des = swing_target_world(start_world, target_world, s, clearance * gait_alpha, td_drop)
            # keep the swing leg somewhat home-regularized to reduce flailing
            Jleg = leg_jacobian_world(m, d, leg)
            err = p_des - current_p
            dq = damped_ls_step(Jleg, err, damping=1e-3)
            dq = np.clip(dq, -dq_limit, dq_limit)
            q_target = current_q + dq
            q_target = 0.85 * q_target + 0.15 * leg.home_joint_qpos

        q_target = _clip_targets_to_ctrlrange(m, leg.actuator_ids, q_target)
        ctrl[np.asarray(leg.actuator_ids, dtype=int)] = q_target
        leg.phase11_last_q_target = q_target.copy()

    return ctrl, stance_enable


def update_phase11_state_pre(d: mujoco.MjData, bindings, cfg, scheduled_contact: np.ndarray,
                             settle_time: float, step_len_front: float, rear_step_scale: float,
                             gait_alpha: float, vx_err: float, prev_sched: np.ndarray) -> None:
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        current_p = foot_point_world(d, leg, FALLBACK_FOOT_LOCAL_OFFSET)

        if bool(scheduled_contact[leg_idx]) and not bool(prev_sched[leg_idx]):
            leg.phase11_stance_start_time = float(d.time)
            leg.phase11_touchdown_recorded = False
            leg.phase11_touchdown_target_world = leg.phase11_swing_target_world.copy() if leg.phase11_swing_target_world is not None else current_p.copy()
            leg.phase11_stance_anchor_world = current_p.copy() if False else None

        elif (not bool(scheduled_contact[leg_idx])) and bool(prev_sched[leg_idx]):
            leg.phase11_swing_start_time = float(d.time)
            leg.phase11_swing_start_world = current_p.copy()
            leg.phase11_touchdown_target_world = None
            leg.phase11_stance_anchor_world = None
            leg.phase11_swing_target_world = nominal_foothold_world(
                d, bindings, cfg, leg_idx, step_len_front, rear_step_scale, gait_alpha, vx_err
            )

        elif not bool(scheduled_contact[leg_idx]) and leg.phase11_swing_target_world is None:
            leg.phase11_swing_start_world = current_p.copy()
            leg.phase11_swing_start_time = float(d.time)
            leg.phase11_swing_target_world = nominal_foothold_world(
                d, bindings, cfg, leg_idx, step_len_front, rear_step_scale, gait_alpha, vx_err
            )


def build_phase11_summary(log: dict, cfg, settle_time: float) -> dict:
    t = np.asarray(log['t'], dtype=float)
    x = np.asarray(log['x'], dtype=float)
    z = x[:, 2] if x.size else np.zeros(0)
    vx = x[:, 3] if x.size else np.zeros(0)
    actual = np.asarray(log['contact_actual'], dtype=bool)

    mask_after = t > max(1.0, settle_time)
    if mask_after.any():
        mean_vx = float(vx[mask_after].mean())
        mean_z = float(z[mask_after].mean())
        min_z = float(z[mask_after].min())
    else:
        mean_vx = 0.0
        mean_z = float(z.mean()) if z.size else 0.0
        min_z = float(z.min()) if z.size else 0.0

    collapse_threshold = 0.55 * float(log.get('nominal_trunk_height', mean_z if mean_z > 0 else 0.2))
    collapse_time = None
    if z.size:
        idx = np.where(z < collapse_threshold)[0]
        if idx.size:
            collapse_time = float(t[idx[0]])

    out = {
        'mean_vx_after_1s': mean_vx,
        'mean_trunk_height_after_1s': mean_z,
        'min_trunk_height_after_1s': min_z,
        'collapse_threshold': float(collapse_threshold),
        'collapse_time': collapse_time,
        'mean_actual_contact_ratio_per_leg': actual.mean(axis=0).tolist() if actual.size else [0.0] * 4,
    }
    return out


def write_phase11_summary(output_dir: str | Path, summary: dict) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'phase11_summary.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return str(path)


def _mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw
