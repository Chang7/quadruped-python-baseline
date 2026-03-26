from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

from baseline.config import LEG_NAMES, MPCConfig
from experiments.low_level_realizer import (
    CleanLowLevelParams,
    CleanLowLevelRealizer,
    actual_foot_contact_state,
    body_pose_velocity,
    crawl_leg_state,
    foot_point_world,
    front_sign,
    is_rear,
    rollout_contact_schedule_crawl,
    rollout_contact_schedule_trot,
)


@dataclass
class AdaptiveSupervisorParams:
    startup_time: float = 0.45
    recovery_height_enter_frac: float = 0.72
    recovery_height_exit_frac: float = 0.82
    recovery_pitch_enter: float = 0.38
    recovery_pitch_exit: float = 0.18
    recovery_roll_enter: float = 0.28
    recovery_roll_exit: float = 0.15
    recovery_min_time: float = 0.28
    stable_hold_time: float = 0.20

    health_height_low_frac: float = 0.74
    health_height_high_frac: float = 0.92
    health_pitch_bad: float = 0.30
    health_roll_bad: float = 0.22

    speed_scale_min: float = 0.28
    force_scale_min: float = 0.20
    step_scale_min: float = 0.35
    drive_scale_min: float = 0.20

    support_bonus_walk: float = 0.18
    support_bonus_recovery: float = 0.70
    gait_clock_min_rate: float = 0.20


class AdaptiveSupervisor:
    def __init__(self, nominal_trunk_height: float, params: AdaptiveSupervisorParams):
        self.nominal_trunk_height = float(nominal_trunk_height)
        self.params = params
        self.mode: str = "STARTUP"
        self.state_enter_time: float = 0.0
        self.gait_clock: float = 0.0
        self.last_t: float | None = None
        self.stable_timer: float = 0.0
        self.recovery_count: int = 0

        self.health: float = 1.0
        self.speed_scale: float = 1.0
        self.force_scale: float = 1.0
        self.step_scale: float = 1.0
        self.drive_scale: float = 1.0
        self.support_bonus: float = 0.0
        self.front_rescue_bias: float = 0.0
        self.entered_recovery: bool = False
        self.just_exited_recovery: bool = False

    def _clip01(self, x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _compute_health(self, z: float, roll: float, pitch: float, actual_contact: np.ndarray) -> float:
        p = self.params
        z_low = p.health_height_low_frac * self.nominal_trunk_height
        z_high = p.health_height_high_frac * self.nominal_trunk_height
        hz = self._clip01((z - z_low) / max(z_high - z_low, 1e-9))
        hp = self._clip01(1.0 - abs(float(pitch)) / max(p.health_pitch_bad, 1e-9))
        hr = self._clip01(1.0 - abs(float(roll)) / max(p.health_roll_bad, 1e-9))
        front_contacts = int(actual_contact[0]) + int(actual_contact[1])
        rear_contacts = int(actual_contact[2]) + int(actual_contact[3])
        # penalize front-support loss more strongly because the failure mode has been rear-sitting / trunk collapse
        hc = 0.5 + 0.25 * min(front_contacts, 1) + 0.25 * min(rear_contacts, 1)
        h = min(hz, hp, hr)
        return float(np.clip(0.75 * h + 0.25 * hc, 0.0, 1.0))

    def _unstable(self, z: float, roll: float, pitch: float) -> bool:
        p = self.params
        return (
            z < p.recovery_height_enter_frac * self.nominal_trunk_height
            or abs(float(pitch)) > p.recovery_pitch_enter
            or abs(float(roll)) > p.recovery_roll_enter
        )

    def _stable(self, z: float, roll: float, pitch: float, actual_contact: np.ndarray) -> bool:
        p = self.params
        front_contacts = int(actual_contact[0]) + int(actual_contact[1])
        rear_contacts = int(actual_contact[2]) + int(actual_contact[3])
        return (
            z > p.recovery_height_exit_frac * self.nominal_trunk_height
            and abs(float(pitch)) < p.recovery_pitch_exit
            and abs(float(roll)) < p.recovery_roll_exit
            and front_contacts >= 1
            and rear_contacts >= 1
        )

    def update(self, t: float, z: float, roll: float, pitch: float, actual_contact: np.ndarray):
        self.entered_recovery = False
        self.just_exited_recovery = False
        if self.last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, float(t - self.last_t))
        self.last_t = float(t)

        self.health = self._compute_health(z, roll, pitch, actual_contact)
        p = self.params

        if self.mode == "STARTUP":
            if t >= p.startup_time and not self._unstable(z, roll, pitch):
                self.mode = "WALK"
                self.state_enter_time = float(t)
            elif t >= p.startup_time and self._unstable(z, roll, pitch):
                self.mode = "RECOVERY"
                self.state_enter_time = float(t)
                self.recovery_count += 1
                self.entered_recovery = True

        elif self.mode == "WALK":
            if self._unstable(z, roll, pitch):
                self.mode = "RECOVERY"
                self.state_enter_time = float(t)
                self.recovery_count += 1
                self.entered_recovery = True

        elif self.mode == "RECOVERY":
            if self._stable(z, roll, pitch, actual_contact):
                self.stable_timer += dt
            else:
                self.stable_timer = 0.0
            if (t - self.state_enter_time) >= p.recovery_min_time and self.stable_timer >= p.stable_hold_time:
                self.mode = "WALK"
                self.state_enter_time = float(t)
                self.stable_timer = 0.0
                self.just_exited_recovery = True

        # scales used by realizer
        if self.mode == "STARTUP":
            self.speed_scale = 0.0
            self.force_scale = 0.0
            self.step_scale = p.step_scale_min
            self.drive_scale = 0.0
            self.support_bonus = p.support_bonus_recovery
            self.front_rescue_bias = 0.0
        elif self.mode == "RECOVERY":
            self.speed_scale = 0.0
            self.force_scale = 0.0
            self.step_scale = p.step_scale_min
            self.drive_scale = 0.0
            self.support_bonus = p.support_bonus_recovery
            self.front_rescue_bias = 0.006
        else:
            h = self.health
            self.speed_scale = max(p.speed_scale_min, h)
            self.force_scale = max(p.force_scale_min, h)
            self.step_scale = max(p.step_scale_min, 0.55 + 0.45 * h)
            self.drive_scale = max(p.drive_scale_min, 0.50 + 0.50 * h)
            self.support_bonus = p.support_bonus_walk * (1.0 - h)
            front_contacts = int(actual_contact[0]) + int(actual_contact[1])
            rear_contacts = int(actual_contact[2]) + int(actual_contact[3])
            self.front_rescue_bias = 0.004 * max(0, rear_contacts - front_contacts)
            self.gait_clock += dt * max(p.gait_clock_min_rate, h)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "health": self.health,
            "speed_scale": self.speed_scale,
            "force_scale": self.force_scale,
            "step_scale": self.step_scale,
            "drive_scale": self.drive_scale,
            "support_bonus": self.support_bonus,
            "front_rescue_bias": self.front_rescue_bias,
            "recovery_count": self.recovery_count,
            "gait_clock": self.gait_clock,
        }


class AdaptiveCleanRealizer(CleanLowLevelRealizer):
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, bindings, cfg: MPCConfig, params: CleanLowLevelParams, supervisor: AdaptiveSupervisor):
        super().__init__(m, d, bindings, cfg, params)
        self.supervisor = supervisor

    def reanchor_all_legs(self):
        for leg in self.bindings.leg_bindings:
            p_now = foot_point_world(self.d, leg)
            leg.stance_anchor_world = p_now.copy()
            leg.swing_started = False
            leg.swing_start_time = None
            leg.swing_start_world = p_now.copy()
            leg.swing_target_world = p_now.copy()
            leg.touchdown_target_world = p_now.copy()
        self.prev_sched = np.ones(4, dtype=bool)

    def desired_speed(self) -> float:
        return super().desired_speed() * float(self.supervisor.speed_scale)

    def support_weight(self, t: float) -> float:
        base = super().support_weight(t)
        return float(np.clip(base + self.supervisor.support_bonus, 0.0, 1.25))

    def mpc_force_gain(self, t: float) -> float:
        return super().mpc_force_gain(t) * float(self.supervisor.force_scale)

    def schedule_now_and_rollout(self, t: float):
        if self.supervisor.mode != "WALK":
            return np.ones(4, dtype=bool), None, None, np.ones((self.cfg.horizon, 4), dtype=bool)
        if self.params.schedule == "crawl":
            now_sched, swing_leg, swing_s = crawl_leg_state(
                self.supervisor.gait_clock,
                self.params.crawl_phase_duration,
                self.params.crawl_swing_duration,
                self.params.crawl_order,
            )
            rollout = rollout_contact_schedule_crawl(
                self.supervisor.gait_clock,
                self.cfg,
                self.params.crawl_phase_duration,
                self.params.crawl_swing_duration,
                self.params.crawl_order,
            )
            return now_sched, swing_leg, swing_s, rollout
        return super().schedule_now_and_rollout(t)

    def build_control_targets(self, scheduled_contact: np.ndarray, actual_contact: np.ndarray, gait_alpha: float) -> np.ndarray:
        # Adjust a subset of parameters on the fly; always restore them before returning.
        saved = {
            "step_len_front": self.params.step_len_front,
            "rear_step_scale": self.params.rear_step_scale,
            "stance_drive_front": self.params.stance_drive_front,
            "stance_drive_rear": self.params.stance_drive_rear,
            "front_unload": self.params.front_unload,
            "stance_press_front": self.params.stance_press_front,
            "stance_press_rear": self.params.stance_press_rear,
            "visual_step_boost": self.params.visual_step_boost,
            "height_k": self.params.height_k,
            "target_height_frac": self.params.target_height_frac,
        }
        try:
            if self.supervisor.mode != "WALK":
                scheduled_contact = np.ones(4, dtype=bool)
                gait_alpha = 0.0
                self.params.step_len_front = 0.0
                self.params.rear_step_scale = 0.0
                self.params.stance_drive_front = 0.0
                self.params.stance_drive_rear = 0.0
                self.params.front_unload = 0.0
                self.params.stance_press_front = max(saved["stance_press_front"], 0.014)
                self.params.stance_press_rear = max(saved["stance_press_rear"], 0.014)
                self.params.visual_step_boost = 0.0
                self.params.height_k = max(saved["height_k"], 1.25)
                self.params.target_height_frac = max(saved["target_height_frac"], 0.97)
            else:
                h = float(self.supervisor.health)
                self.params.step_len_front = saved["step_len_front"] * self.supervisor.step_scale
                self.params.rear_step_scale = saved["rear_step_scale"] * (0.85 + 0.15 * self.supervisor.step_scale)
                self.params.stance_drive_front = saved["stance_drive_front"] * self.supervisor.drive_scale
                self.params.stance_drive_rear = saved["stance_drive_rear"] * self.supervisor.drive_scale
                self.params.front_unload = saved["front_unload"] - self.supervisor.front_rescue_bias
                self.params.stance_press_front = saved["stance_press_front"] + 0.5 * self.supervisor.front_rescue_bias
                self.params.stance_press_rear = max(0.5 * saved["stance_press_rear"], saved["stance_press_rear"] - 0.5 * self.supervisor.front_rescue_bias)
                self.params.visual_step_boost = saved["visual_step_boost"] * (0.9 + 0.4 * h)
                self.params.height_k = saved["height_k"] * (1.0 + 0.15 * (1.0 - h))
                gait_alpha = float(gait_alpha) * (0.7 + 0.3 * h)
            return super().build_control_targets(scheduled_contact, actual_contact, gait_alpha)
        finally:
            self.params.step_len_front = saved["step_len_front"]
            self.params.rear_step_scale = saved["rear_step_scale"]
            self.params.stance_drive_front = saved["stance_drive_front"]
            self.params.stance_drive_rear = saved["stance_drive_rear"]
            self.params.front_unload = saved["front_unload"]
            self.params.stance_press_front = saved["stance_press_front"]
            self.params.stance_press_rear = saved["stance_press_rear"]
            self.params.visual_step_boost = saved["visual_step_boost"]
            self.params.height_k = saved["height_k"]
            self.params.target_height_frac = saved["target_height_frac"]


def write_adaptive_summary(output_dir: str | Path, summary: dict[str, Any]) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "adaptive_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
