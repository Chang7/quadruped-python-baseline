
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from baseline.config import LEG_NAMES, MPCConfig
from experiments.low_level_realizer import (
    ModelBindings,
    body_pose_velocity,
    foot_point_world,
    leg_jacobian_world,
    damped_ls_step,
    clip_targets_to_ctrlrange,
    nominal_foothold_world,
    swing_target_world,
    actual_foot_contact_state,
    front_sign,
    side_sign,
    is_rear,
)


@dataclass
class QuasiStaticParams:
    startup_time: float = 0.45
    shift_time: float = 0.22
    swing_time: float = 0.22
    touchdown_wait: float = 0.16
    hold_time: float = 0.10

    desired_speed_cap: float = 0.06
    support_enabled: bool = True
    support_weight_start: float = 0.95
    support_weight_end: float = 0.95
    support_fade_start: float = 0.4
    support_fade_end: float = 6.0
    target_height_frac: float = 1.02
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_yaw: float = 0.0

    support_xy_k: float = 120.0
    support_xy_d: float = 20.0
    support_z_k: float = 650.0
    support_z_d: float = 85.0
    support_roll_k: float = 22.0
    support_roll_d: float = 3.2
    support_pitch_k: float = 24.0
    support_pitch_d: float = 3.2
    support_yaw_k: float = 5.0
    support_yaw_d: float = 1.0
    support_max_force_xy: float = 55.0
    support_max_force_z: float = 420.0
    support_max_torque: float = 20.0

    force_frame: str = "body"
    realization: str = "external"
    mpc_force_gain_start: float = 0.00
    mpc_force_gain_end: float = 0.10
    mpc_force_ramp_start: float = 1.2
    mpc_force_ramp_end: float = 3.2
    max_xy_over_fz: float = 0.25

    clearance: float = 0.060
    step_len_front: float = 0.040
    rear_step_scale: float = 0.85
    touchdown_depth_front: float = 0.012
    touchdown_depth_rear: float = 0.015
    touchdown_extra: float = 0.006
    dq_limit: float = 0.12
    damping_ls: float = 1e-4

    stance_press_front: float = 0.006
    stance_press_rear: float = 0.007
    stance_press_gain: float = 0.00025
    stance_fx_bias_gain: float = 0.00035
    stance_fy_bias_gain: float = 0.00030

    shift_x_mag: float = 0.010
    shift_y_mag: float = 0.012
    touchdown_confirm_hold: float = 0.06

    recovery_height_enter_frac: float = 0.85
    recovery_height_exit_frac: float = 0.96
    recovery_pitch_enter: float = 0.25
    recovery_roll_enter: float = 0.22
    recovery_pitch_exit: float = 0.12
    recovery_roll_exit: float = 0.10
    recovery_min_time: float = 0.12

    recovery_enable_after: float = 0.35
    recovery_force_exit_after: float = 0.35
    recovery_required_contacts: int = 4

    crawl_order: tuple[int, int, int, int] = (2, 3, 0, 1)  # RL, RR, FL, FR

    visual_step_boost: float = 1.0


def _quat_yaw_wrap(err: float) -> float:
    return float(np.arctan2(np.sin(err), np.cos(err)))


def _lerp(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t <= t0:
        return float(y0)
    if t >= t1:
        return float(y1)
    a = (t - t0) / max(t1 - t0, 1e-9)
    return float((1.0 - a) * y0 + a * y1)


def _support_weight(t: float, p: QuasiStaticParams) -> float:
    if not p.support_enabled:
        return 0.0
    return _lerp(t, p.support_fade_start, p.support_fade_end, p.support_weight_start, p.support_weight_end)


def _mpc_force_gain(t: float, p: QuasiStaticParams) -> float:
    return _lerp(t, p.mpc_force_ramp_start, p.mpc_force_ramp_end, p.mpc_force_gain_start, p.mpc_force_gain_end)


class ConfirmedQuasiStaticCrawl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, bindings: ModelBindings, cfg: MPCConfig, params: QuasiStaticParams):
        self.m = m
        self.d = d
        self.bindings = bindings
        self.cfg = cfg
        self.params = params

        pos, _, rpy, _, _ = body_pose_velocity(m, d, bindings.base_body_id)
        self.nominal_height = float(pos[2])
        self.target_height = float(params.target_height_frac * self.nominal_height)
        self.total_mass = float(np.sum(self.m.body_mass[1:]))
        self.g_mag = float(abs(self.m.opt.gravity[2])) if hasattr(self.m, "opt") else 9.81

        self.phase = "STARTUP"
        self.phase_start = 0.0
        self.leg_ptr = 0
        self.swing_leg: Optional[int] = None
        self.swing_start_world = None
        self.swing_target_world = None
        self.touchdown_target_world = None
        self.hold_contact_start = None

        self.recovery_count = 0
        self.recovery_enter_time = None

        self.home_ctrl = np.zeros(m.nu, dtype=float)
        for leg in self.bindings.leg_bindings:
            q = np.array([d.qpos[a] for a in leg.qpos_adrs], dtype=float)
            if leg.actuator_ids:
                self.home_ctrl[np.asarray(leg.actuator_ids, dtype=int)] = clip_targets_to_ctrlrange(m, leg.actuator_ids, q)

        self.stance_anchors = [foot_point_world(d, leg).copy() for leg in self.bindings.leg_bindings]
        self.body_target_xy = pos[:2].copy()
        self.body_target_yaw = float(rpy[2])

    def desired_speed(self) -> float:
        return min(float(self.cfg.desired_speed), float(self.params.desired_speed_cap))

    def current_sched(self) -> np.ndarray:
        sched = np.ones(4, dtype=bool)
        if self.phase in ("SWING", "TOUCHDOWN") and self.swing_leg is not None:
            sched[int(self.swing_leg)] = False
        return sched

    def rollout_sched(self) -> np.ndarray:
        sched = self.current_sched()
        return np.repeat(sched[None, :], self.cfg.horizon, axis=0)

    def _support_legs(self, swing_leg: int | None = None):
        if swing_leg is None:
            return [0, 1, 2, 3]
        return [i for i in range(4) if i != int(swing_leg)]

    def _compute_shift_target(self, swing_leg: int):
        stance_ids = self._support_legs(swing_leg)
        pts = np.vstack([self.stance_anchors[i] for i in stance_ids])
        ctr = pts[:, :2].mean(axis=0)
        # much smaller longitudinal shift; prefer lateral COM shift over aggressive fore/aft shift
        if is_rear(swing_leg):
            ctr[0] += 0.15 * self.params.shift_x_mag
        else:
            ctr[0] -= 0.10 * self.params.shift_x_mag
        ctr[1] += -side_sign(swing_leg) * self.params.shift_y_mag
        return ctr

    def _compute_swing_target(self, swing_leg: int):
        speed = self.desired_speed()
        step_len_front = self.params.step_len_front * self.params.visual_step_boost
        rear_scale = self.params.rear_step_scale
        tgt = nominal_foothold_world(
            self.d,
            self.bindings,
            self.cfg,
            swing_leg,
            gait_alpha=1.0,
            step_len_front=step_len_front,
            rear_step_scale=rear_scale,
            desired_vx=speed if speed != 0 else 0.10,
        )
        # Keep touchdown on the ground plane of current foot / anchor
        z_ref = min(self.stance_anchors[swing_leg][2], foot_point_world(self.d, self.bindings.leg_bindings[swing_leg])[2])
        tgt[2] = z_ref
        return tgt

    def _is_unhealthy(self, pos: np.ndarray, rpy: np.ndarray) -> bool:
        h_bad = pos[2] < self.params.recovery_height_enter_frac * self.nominal_height
        p_bad = abs(float(rpy[1])) > self.params.recovery_pitch_enter
        r_bad = abs(float(rpy[0])) > self.params.recovery_roll_enter
        return bool(h_bad or p_bad or r_bad)

    def _is_recovered(self, pos: np.ndarray, rpy: np.ndarray) -> bool:
        h_ok = pos[2] > self.params.recovery_height_exit_frac * self.nominal_height
        p_ok = abs(float(rpy[1])) < self.params.recovery_pitch_exit
        r_ok = abs(float(rpy[0])) < self.params.recovery_roll_exit
        return bool(h_ok and p_ok and r_ok)

    def step_state_machine(self, t: float, actual_contact: np.ndarray):
        pos, _, rpy, _, _ = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)

        if self.phase != "RECOVERY" and t >= self.params.recovery_enable_after and self._is_unhealthy(pos, rpy):
            self.phase = "RECOVERY"
            self.phase_start = t
            self.recovery_enter_time = t
            self.recovery_count += 1
            self.swing_leg = None

        if self.phase == "STARTUP":
            if t - self.phase_start >= self.params.startup_time:
                self.phase = "SHIFT"
                self.phase_start = t
                self.swing_leg = int(self.params.crawl_order[self.leg_ptr])
                self.body_target_xy = self._compute_shift_target(self.swing_leg)

        elif self.phase == "SHIFT":
            # refresh target slightly as body moves
            if self.swing_leg is not None:
                self.body_target_xy = self._compute_shift_target(self.swing_leg)
            if t - self.phase_start >= self.params.shift_time:
                self.phase = "SWING"
                self.phase_start = t
                self.swing_start_world = foot_point_world(self.d, self.bindings.leg_bindings[self.swing_leg]).copy()
                self.swing_target_world = self._compute_swing_target(self.swing_leg)
                self.touchdown_target_world = self.swing_target_world.copy()
                self.hold_contact_start = None

        elif self.phase == "SWING":
            if t - self.phase_start >= self.params.swing_time:
                self.phase = "TOUCHDOWN"
                self.phase_start = t

        elif self.phase == "TOUCHDOWN":
            if self.swing_leg is not None and bool(actual_contact[self.swing_leg]):
                if self.hold_contact_start is None:
                    self.hold_contact_start = t
                elif t - self.hold_contact_start >= self.params.touchdown_confirm_hold:
                    self.stance_anchors[self.swing_leg] = foot_point_world(self.d, self.bindings.leg_bindings[self.swing_leg]).copy()
                    self.phase = "HOLD"
                    self.phase_start = t
                    self.hold_contact_start = None
            else:
                self.hold_contact_start = None
                if t - self.phase_start >= self.params.touchdown_wait:
                    # accept current placement and continue; helps avoid deadlock
                    if self.swing_leg is not None:
                        self.stance_anchors[self.swing_leg] = foot_point_world(self.d, self.bindings.leg_bindings[self.swing_leg]).copy()
                    self.phase = "HOLD"
                    self.phase_start = t

        elif self.phase == "HOLD":
            if t - self.phase_start >= self.params.hold_time:
                self.leg_ptr = (self.leg_ptr + 1) % 4
                self.swing_leg = int(self.params.crawl_order[self.leg_ptr])
                self.phase = "SHIFT"
                self.phase_start = t
                self.body_target_xy = self._compute_shift_target(self.swing_leg)

        elif self.phase == "RECOVERY":
            # refresh anchors from all contacting feet to stop drift
            for i, leg in enumerate(self.bindings.leg_bindings):
                if bool(actual_contact[i]):
                    self.stance_anchors[i] = foot_point_world(self.d, leg).copy()
            self.swing_leg = None
            # keep body target near current pose, slightly forward to avoid sitting back
            self.body_target_xy = pos[:2].copy()
            self.body_target_xy[0] += 0.002
            ready_contacts = int(np.sum(actual_contact)) >= int(self.params.recovery_required_contacts)
            recovered = self._is_recovered(pos, rpy)
            timed_out = (t - self.phase_start) >= self.params.recovery_force_exit_after and ready_contacts
            if (t - self.phase_start) >= self.params.recovery_min_time and (recovered or timed_out):
                self.phase = "SHIFT"
                self.phase_start = t
                self.swing_leg = int(self.params.crawl_order[self.leg_ptr])
                self.body_target_xy = self._compute_shift_target(self.swing_leg)

    def swing_phase(self, t: float) -> float:
        if self.phase == "SWING":
            return float(np.clip((t - self.phase_start) / max(self.params.swing_time, 1e-6), 0.0, 1.0))
        if self.phase == "TOUCHDOWN":
            return 1.0
        return 0.0

    def build_ctrl_targets(self, t: float, u_cmd: np.ndarray, actual_contact: np.ndarray) -> np.ndarray:
        ctrl = self.home_ctrl.copy()
        pos, R, rpy, lin_vel, ang_vel = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        sched = self.current_sched()
        s = self.swing_phase(t)
        for leg in self.bindings.leg_bindings:
            i = leg.leg_idx
            foot_now = foot_point_world(self.d, leg)
            if sched[i]:
                tgt = self.stance_anchors[i].copy()
                fz = max(float(u_cmd[3 * i + 2]), 0.0)
                fx = float(u_cmd[3 * i + 0])
                fy = float(u_cmd[3 * i + 1])
                press = (self.params.stance_press_rear if is_rear(i) else self.params.stance_press_front) + self.params.stance_press_gain * min(fz, 80.0)
                tgt[2] -= press
                # very small MPC-informed horizontal bias on stance target
                tgt[0] -= self.params.stance_fx_bias_gain * fx
                tgt[1] -= self.params.stance_fy_bias_gain * fy
            else:
                depth = self.params.touchdown_depth_rear if is_rear(i) else self.params.touchdown_depth_front
                if self.phase == "SWING":
                    tgt = swing_target_world(
                        np.asarray(self.swing_start_world, dtype=float),
                        np.asarray(self.swing_target_world, dtype=float),
                        s,
                        self.params.clearance * self.params.visual_step_boost,
                        depth,
                    )
                else:
                    tgt = np.asarray(self.touchdown_target_world, dtype=float).copy()
                    tgt[2] -= self.params.touchdown_extra

            J = leg_jacobian_world(self.m, self.d, leg)
            err = tgt - foot_now
            dq = damped_ls_step(J, err, damping=self.params.damping_ls)
            dq = np.clip(dq, -self.params.dq_limit, self.params.dq_limit)
            q_cur = np.array([self.d.qpos[a] for a in leg.qpos_adrs], dtype=float)
            q_ref = q_cur + dq
            q_ref = clip_targets_to_ctrlrange(self.m, leg.actuator_ids, q_ref)
            if leg.actuator_ids:
                ctrl[np.asarray(leg.actuator_ids, dtype=int)] = q_ref
        return ctrl

    def apply_support(self, t: float) -> dict:
        pos, R, rpy, lin_vel, ang_vel = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        yaw = float(rpy[2])

        w = _support_weight(t, self.params)
        if self.phase == "RECOVERY":
            w = max(w, 1.0)

        target_pos = np.array([self.body_target_xy[0], self.body_target_xy[1], self.target_height], dtype=float)
        fx = w * (self.params.support_xy_k * (target_pos[0] - pos[0]) - self.params.support_xy_d * lin_vel[0])
        fy = w * (self.params.support_xy_k * (target_pos[1] - pos[1]) - self.params.support_xy_d * lin_vel[1])
        gravity_ff = self.total_mass * self.g_mag
        fz = w * (gravity_ff + self.params.support_z_k * (target_pos[2] - pos[2]) - self.params.support_z_d * lin_vel[2])

        tx = -w * (self.params.support_roll_k * (rpy[0] - self.params.target_roll) + self.params.support_roll_d * ang_vel[0])
        ty = -w * (self.params.support_pitch_k * (rpy[1] - self.params.target_pitch) + self.params.support_pitch_d * ang_vel[1])
        yaw_err = _quat_yaw_wrap(self.body_target_yaw - yaw)
        tz = w * (self.params.support_yaw_k * yaw_err - self.params.support_yaw_d * ang_vel[2])

        force = np.array([
            float(np.clip(fx, -self.params.support_max_force_xy, self.params.support_max_force_xy)),
            float(np.clip(fy, -self.params.support_max_force_xy, self.params.support_max_force_xy)),
            float(np.clip(fz, 0.0, self.params.support_max_force_z)),
        ], dtype=float)
        torque = np.array([
            float(np.clip(tx, -self.params.support_max_torque, self.params.support_max_torque)),
            float(np.clip(ty, -self.params.support_max_torque, self.params.support_max_torque)),
            float(np.clip(tz, -self.params.support_max_torque, self.params.support_max_torque)),
        ], dtype=float)

        mujoco.mj_applyFT(self.m, self.d, force, torque, pos, int(self.bindings.base_body_id), self.d.qfrc_applied)
        return {"force_world": force, "torque_world": torque, "weight": w, "phase": self.phase}

    def apply_mpc_forces(self, t: float, u_cmd: np.ndarray, actual_contact: np.ndarray) -> dict:
        gain = _mpc_force_gain(t, self.params)
        sched = self.current_sched()
        u_applied = np.zeros_like(u_cmd)
        force_enabled = np.zeros(4, dtype=bool)
        if gain <= 0.0:
            return {"u_applied": u_applied, "force_enabled": force_enabled, "gain": gain}
        pos, R_base, _, _, _ = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        for leg in self.bindings.leg_bindings:
            i = leg.leg_idx
            if not bool(sched[i]) or not bool(actual_contact[i]):
                continue
            f = np.asarray(u_cmd[3*i:3*i+3], dtype=float).copy()
            if self.params.force_frame == "body":
                f = R_base @ f
            f[0:2] = np.clip(f[0:2], -self.params.max_xy_over_fz * max(f[2], 0.0), self.params.max_xy_over_fz * max(f[2], 0.0))
            f *= gain
            if self.params.realization == "joint":
                J = leg_jacobian_world(self.m, self.d, leg)
                tau = J.T @ f
                for j, dof in enumerate(leg.dof_adrs):
                    self.d.qfrc_applied[dof] += tau[j]
            else:
                p = foot_point_world(self.d, leg)
                mujoco.mj_applyFT(self.m, self.d, f, np.zeros(3), p, int(leg.calf_body_id), self.d.qfrc_applied)
            u_applied[3*i:3*i+3] = f
            force_enabled[i] = True
        return {"u_applied": u_applied, "force_enabled": force_enabled, "gain": gain}

    def summary_health(self):
        pos, _, rpy, _, _ = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        health = 1.0
        health *= float(np.clip(pos[2] / max(self.nominal_height, 1e-6), 0.0, 1.0))
        health *= float(np.clip(1.0 - abs(rpy[1]) / max(self.params.recovery_pitch_enter, 1e-6), 0.0, 1.0))
        health *= float(np.clip(1.0 - abs(rpy[0]) / max(self.params.recovery_roll_enter, 1e-6), 0.0, 1.0))
        return health
