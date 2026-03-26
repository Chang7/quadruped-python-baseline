from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco


def mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw


def body_velocity_world(m: mujoco.MjModel, d: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(body_id), vel6, 0)
    return vel6[3:6].copy(), vel6[0:3].copy()  # linear, angular in world


def apply_trunk_support_wrench_v2(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    trunk_body_id: int,
    target_height: float,
    target_yaw: float,
    desired_vx: float,
    *,
    enabled: bool = True,
    support_weight_frac: float = 0.88,
    target_x: float | None = None,
    target_y: float | None = None,
    x_k: float = 30.0,
    y_k: float = 18.0,
    height_k: float = 450.0,
    height_d: float = 55.0,
    vx_k: float = 18.0,
    vy_k: float = 18.0,
    roll_target: float = 0.0,
    pitch_target: float = 0.0,
    roll_k: float = 22.0,
    roll_d: float = 2.8,
    pitch_k: float = 20.0,
    pitch_d: float = 2.6,
    yaw_k: float = 6.0,
    yaw_d: float = 1.2,
    forward_bias: float = 0.0,
    max_force_xy: float = 18.0,
    max_force_z: float = 180.0,
    max_torque: float = 18.0,
) -> dict:
    out = {
        'force_world': np.zeros(3, dtype=float),
        'torque_world': np.zeros(3, dtype=float),
        'target_height': float(target_height),
        'desired_vx': float(desired_vx),
        'target_x': None if target_x is None else float(target_x),
        'target_y': None if target_y is None else float(target_y),
    }
    if not enabled:
        return out

    base = d.body(trunk_body_id)
    pos = np.asarray(base.xpos, dtype=float).copy()
    R = np.asarray(base.xmat, dtype=float).reshape(3, 3)
    roll, pitch, yaw = mat_to_rpy(R)
    lin_vel, ang_vel = body_velocity_world(m, d, trunk_body_id)

    total_mass = float(np.sum(np.asarray(m.body_mass, dtype=float)))
    gravity = float(abs(m.opt.gravity[2]))

    fz = support_weight_frac * total_mass * gravity + height_k * (float(target_height) - pos[2]) - height_d * lin_vel[2]
    fx = vx_k * (float(desired_vx) - lin_vel[0]) + float(forward_bias)
    if target_x is not None:
        fx += x_k * (float(target_x) - pos[0])
    fy = -vy_k * lin_vel[1]
    if target_y is not None:
        fy += y_k * (float(target_y) - pos[1])

    yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    tx = -roll_k * (roll - roll_target) - roll_d * ang_vel[0]
    ty = -pitch_k * (pitch - pitch_target) - pitch_d * ang_vel[1]
    tz = yaw_k * yaw_err - yaw_d * ang_vel[2]

    force_world = np.array([
        float(np.clip(fx, -max_force_xy, max_force_xy)),
        float(np.clip(fy, -max_force_xy, max_force_xy)),
        float(np.clip(fz, 0.0, max_force_z)),
    ], dtype=float)
    torque_world = np.array([
        float(np.clip(tx, -max_torque, max_torque)),
        float(np.clip(ty, -max_torque, max_torque)),
        float(np.clip(tz, -max_torque, max_torque)),
    ], dtype=float)

    mujoco.mj_applyFT(m, d, force_world, torque_world, pos, int(trunk_body_id), d.qfrc_applied)

    out.update({
        'force_world': force_world,
        'torque_world': torque_world,
        'roll': float(roll), 'pitch': float(pitch), 'yaw': float(yaw),
        'z': float(pos[2]), 'x': float(pos[0]), 'y': float(pos[1]),
        'vx': float(lin_vel[0]), 'vy': float(lin_vel[1]), 'vz': float(lin_vel[2]),
        'target_pitch': float(pitch_target), 'target_roll': float(roll_target),
        'forward_bias': float(forward_bias),
    })
    return out


def build_phase14_summary(log: dict, settle_time: float = 0.0) -> dict:
    t = np.asarray(log['t'], dtype=float)
    if t.size == 0:
        return {}
    x = np.asarray(log['x'], dtype=float)
    support_force = np.asarray(log['support_force_world'], dtype=float)
    support_tau = np.asarray(log['support_torque_world'], dtype=float)
    nominal_trunk_height = float(log.get('nominal_trunk_height', np.nan))

    mask = t >= max(1.0, float(settle_time))
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    z = x[:, 2]
    summary = {
        'mean_vx_after_1s': float(np.mean(x[mask, 3])) if x.ndim == 2 and x.shape[1] > 3 else None,
        'mean_trunk_height_after_1s': float(np.mean(z[mask])),
        'min_trunk_height_after_1s': float(np.min(z[mask])),
        'collapse_time': float(t[np.where(z < 0.22)[0][0]]) if np.any(z < 0.22) else None,
        'mean_support_force_world_after_1s': support_force[mask].mean(axis=0).tolist() if support_force.size else [0.0, 0.0, 0.0],
        'mean_support_torque_world_after_1s': support_tau[mask].mean(axis=0).tolist() if support_tau.size else [0.0, 0.0, 0.0],
        'nominal_trunk_height': nominal_trunk_height,
    }
    return summary


def write_phase14_summary(output_dir: str | Path, summary: dict) -> str:
    out = Path(output_dir) / 'phase14_summary.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(out)
