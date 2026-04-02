from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


BASE_PARAMS: dict[str, Any] = {
    "gait": "crawl",
    "preset": "conservative",
    "seconds": 5,
    "speed": 0.12,
    "lateral_speed": 0.0,
    "yaw_rate": 0.0,
    "seed": 0,
    "q_theta": 100000.0,
    "q_w": 10000.0,
    "r_u": 100.0,
    "du_xy_max": 1.0,
    "support_floor_ratio": 0.20,
    "grf_max_scale": 0.38,
    "joint_pd_scale": 0.25,
    "latched_joint_pd_scale": 0.25,
    "gait_step_freq": 0.38,
    "gait_duty_factor": 0.92,
    "stance_target_blend": 0.08,
    "contact_latch_budget_s": 0.10,
    "startup_full_stance_time_s": 0.28,
    "latched_force_scale": 0.95,
    "latched_release_phase_start": 0.18,
    "latched_release_phase_end": 0.58,
    "fy_scale": 1.0,
    "support_centroid_x_gain": 2.0,
    "support_centroid_y_gain": 10.0,
    "pre_swing_lookahead_steps": 5,
    "pre_swing_front_shift_scale": 1.5,
    "pre_swing_rear_shift_scale": 1.0,
    "pre_swing_gate_min_margin": 0.025,
    "front_pre_swing_gate_min_margin": 0.030,
    "rear_pre_swing_gate_min_margin": 0.025,
    "front_late_release_phase_threshold": 0.65,
    "front_late_release_min_margin": 0.015,
    "front_late_release_hold_s": 0.04,
    "front_late_release_forward_scale": 0.20,
    "support_margin_preview_s": 0.04,
    "front_late_release_extra_margin": 0.005,
    "front_late_release_pitch_guard": 0.30,
    "front_late_release_roll_guard": 0.35,
    "touchdown_reacquire_hold_s": 0.0,
    "front_touchdown_reacquire_hold_s": 0.14,
    "touchdown_reacquire_forward_scale": 0.20,
    "touchdown_reacquire_xy_blend": 0.0,
    "front_touchdown_reacquire_xy_blend": 0.70,
    "touchdown_reacquire_extra_depth": 0.0,
    "front_touchdown_reacquire_extra_depth": 0.016,
    "touchdown_reacquire_forward_bias": 0.0,
    "front_touchdown_reacquire_forward_bias": 0.014,
    "touchdown_confirm_hold_s": 0.0,
    "front_touchdown_confirm_hold_s": 0.12,
    "touchdown_confirm_forward_scale": 0.15,
    "touchdown_settle_hold_s": 0.0,
    "front_touchdown_settle_hold_s": 0.26,
    "touchdown_settle_forward_scale": 0.15,
    "touchdown_support_rear_floor_delta": 0.15,
    "touchdown_support_vertical_boost": 0.15,
    "touchdown_support_z_pos_gain_delta": 6.0,
    "touchdown_support_roll_angle_gain_delta": 6.0,
    "touchdown_support_roll_rate_gain_delta": 2.0,
    "touchdown_support_pitch_angle_gain_delta": 10.0,
    "touchdown_support_pitch_rate_gain_delta": 3.0,
    "touchdown_support_rear_joint_pd_scale": 0.10,
    "touchdown_support_anchor_xy_blend": 0.90,
    "touchdown_support_anchor_z_blend": 0.75,
    "full_contact_recovery_hold_s": 0.0,
    "full_contact_recovery_forward_scale": 1.0,
    "full_contact_recovery_roll_threshold": 0.0,
    "full_contact_recovery_pitch_threshold": 0.0,
    "full_contact_recovery_height_ratio": 0.0,
    "full_contact_recovery_recent_window_s": 0.0,
    "pre_swing_gate_hold_s": 0.12,
    "pre_swing_gate_forward_scale": 0.35,
    "vx_gain": 1.6,
    "vy_gain": 4.5,
    "z_pos_gain": 22.0,
    "z_vel_gain": 5.5,
    "min_vertical_force_scale": 0.55,
    "reduced_support_vertical_boost": 0.05,
    "roll_angle_gain": 32.0,
    "roll_rate_gain": 7.0,
    "pitch_angle_gain": 32.0,
    "pitch_rate_gain": 8.0,
    "rear_floor_base_scale": 0.65,
    "rear_floor_pitch_gain": 0.30,
    "pitch_rebalance_gain": 0.0,
    "pitch_rebalance_ref": 0.35,
    "front_latched_pitch_relief_gain": 0.0,
    "front_latched_rear_bias_gain": 0.0,
    "latched_swing_lift_ratio": 0.0,
    "latched_swing_tau_blend": 0.0,
    "virtual_unlatch_phase_threshold": 1.10,
    "virtual_unlatch_hold_s": 0.0,
}

SEARCH_SPACE: dict[str, list[Any]] = {
    "r_u": [80.0, 100.0, 120.0],
    "support_floor_ratio": [0.18, 0.20, 0.22],
    "grf_max_scale": [0.36, 0.38, 0.40],
    "joint_pd_scale": [0.20, 0.25, 0.30],
    "latched_joint_pd_scale": [0.20, 0.25, 0.30],
    "gait_step_freq": [0.38, 0.40, 0.42],
    "gait_duty_factor": [0.90, 0.92, 0.94],
    "stance_target_blend": [0.06, 0.08, 0.10],
    "contact_latch_budget_s": [0.08, 0.10, 0.12],
    "startup_full_stance_time_s": [0.24, 0.28, 0.32],
    "latched_force_scale": [0.90, 0.95],
    "fy_scale": [0.8, 1.0],
    "support_centroid_x_gain": [1.0, 2.0, 3.0],
    "support_centroid_y_gain": [8.0, 10.0, 12.0],
    "pre_swing_lookahead_steps": [4, 5, 6],
    "pre_swing_front_shift_scale": [1.5, 2.0, 2.5],
    "pre_swing_rear_shift_scale": [0.8, 1.0, 1.2],
    "pre_swing_gate_min_margin": [0.020, 0.025, 0.030],
    "front_pre_swing_gate_min_margin": [0.025, 0.030, 0.035],
    "rear_pre_swing_gate_min_margin": [0.020, 0.025, 0.030],
    "pre_swing_gate_hold_s": [0.10, 0.12, 0.14],
    "pre_swing_gate_forward_scale": [0.35, 0.45, 0.55, 1.0],
    "z_pos_gain": [20.0, 22.0, 24.0],
    "z_vel_gain": [5.0, 5.5, 6.0],
    "min_vertical_force_scale": [0.45, 0.50, 0.55],
    "reduced_support_vertical_boost": [0.0, 0.05, 0.10],
    "roll_angle_gain": [28.0, 30.0, 32.0],
    "roll_rate_gain": [6.0, 7.0, 8.0],
    "pitch_angle_gain": [28.0, 30.0, 32.0, 34.0],
    "pitch_rate_gain": [6.0, 8.0, 10.0],
    "rear_floor_base_scale": [0.65, 0.70, 0.75],
    "rear_floor_pitch_gain": [0.20, 0.30, 0.35],
    "pitch_rebalance_gain": [0.0, 0.08, 0.12],
    "front_latched_pitch_relief_gain": [0.0, 0.25, 0.45],
    "front_latched_rear_bias_gain": [0.0, 1.0, 1.8],
    "latched_swing_lift_ratio": [0.0, 0.06, 0.10],
    "latched_swing_tau_blend": [0.0, 0.05, 0.08],
    "front_late_release_phase_threshold": [0.60, 0.65, 0.72, 0.80],
    "front_late_release_min_margin": [0.010, 0.015, 0.020, 0.025],
    "front_late_release_hold_s": [0.02, 0.04, 0.06],
    "front_late_release_forward_scale": [0.05, 0.10, 0.15, 0.20],
    "support_margin_preview_s": [0.0, 0.02, 0.04, 0.06],
    "front_late_release_extra_margin": [0.0, 0.005, 0.010],
    "front_late_release_pitch_guard": [0.25, 0.30, 0.35],
    "front_late_release_roll_guard": [0.30, 0.35, 0.40],
    "touchdown_reacquire_hold_s": [0.0, 0.04, 0.08],
    "front_touchdown_reacquire_hold_s": [0.08, 0.12, 0.16],
    "touchdown_reacquire_forward_scale": [0.15, 0.25, 0.35],
    "touchdown_reacquire_xy_blend": [0.0, 0.25, 0.40],
    "front_touchdown_reacquire_xy_blend": [0.35, 0.55, 0.75],
    "touchdown_reacquire_extra_depth": [0.0, 0.004, 0.008],
    "front_touchdown_reacquire_extra_depth": [0.008, 0.012, 0.016],
    "touchdown_reacquire_forward_bias": [0.0, 0.004, 0.008],
    "front_touchdown_reacquire_forward_bias": [0.006, 0.010, 0.014],
    "touchdown_confirm_hold_s": [0.0, 0.03, 0.06],
    "front_touchdown_confirm_hold_s": [0.04, 0.06, 0.08],
    "touchdown_confirm_forward_scale": [0.25, 0.35, 0.45],
    "touchdown_settle_hold_s": [0.0, 0.04, 0.08],
    "front_touchdown_settle_hold_s": [0.04, 0.08, 0.12],
    "touchdown_settle_forward_scale": [0.35, 0.45, 0.55],
    "touchdown_support_rear_floor_delta": [0.0, 0.05, 0.10, 0.15],
    "touchdown_support_vertical_boost": [0.0, 0.05, 0.10, 0.15],
    "touchdown_support_z_pos_gain_delta": [0.0, 2.0, 4.0, 6.0],
    "touchdown_support_roll_angle_gain_delta": [0.0, 2.0, 4.0, 6.0],
    "touchdown_support_roll_rate_gain_delta": [0.0, 1.0, 2.0],
    "touchdown_support_pitch_angle_gain_delta": [0.0, 4.0, 8.0, 12.0],
    "touchdown_support_pitch_rate_gain_delta": [0.0, 1.0, 2.0, 3.0],
    "touchdown_support_rear_joint_pd_scale": [0.0, 0.05, 0.08, 0.12],
    "touchdown_support_anchor_xy_blend": [0.0, 0.50, 0.75, 0.90],
    "touchdown_support_anchor_z_blend": [0.0, 0.25, 0.50, 0.75],
    "full_contact_recovery_hold_s": [0.0, 0.08, 0.12, 0.16],
    "full_contact_recovery_forward_scale": [0.10, 0.15, 0.20, 1.0],
    "full_contact_recovery_roll_threshold": [0.18, 0.22, 0.26],
    "full_contact_recovery_pitch_threshold": [0.12, 0.15, 0.18],
    "full_contact_recovery_height_ratio": [0.70, 0.74, 0.78],
    "full_contact_recovery_recent_window_s": [0.0, 0.10, 0.14, 0.18],
    "virtual_unlatch_phase_threshold": [1.10],
    "virtual_unlatch_hold_s": [0.0],
}

SEED_CANDIDATES: list[dict[str, Any]] = [
    {},
    {"startup_full_stance_time_s": 0.24},
    {"startup_full_stance_time_s": 0.32},
    {"pre_swing_gate_forward_scale": 1.0},
    {"pre_swing_gate_forward_scale": 0.35},
    {"front_pre_swing_gate_min_margin": 0.025, "rear_pre_swing_gate_min_margin": 0.025},
    {"front_pre_swing_gate_min_margin": 0.030, "rear_pre_swing_gate_min_margin": 0.025},
    {"front_pre_swing_gate_min_margin": 0.035, "rear_pre_swing_gate_min_margin": 0.025},
    {"front_late_release_phase_threshold": 0.60, "front_late_release_min_margin": 0.015, "front_late_release_hold_s": 0.04, "front_late_release_forward_scale": 0.15},
    {"front_late_release_phase_threshold": 0.65, "front_late_release_min_margin": 0.015, "front_late_release_hold_s": 0.04, "front_late_release_forward_scale": 0.15},
    {"front_late_release_phase_threshold": 0.72, "front_late_release_min_margin": 0.020, "front_late_release_hold_s": 0.04, "front_late_release_forward_scale": 0.10},
    {"front_late_release_phase_threshold": 0.80, "front_late_release_min_margin": 0.020, "front_late_release_hold_s": 0.06, "front_late_release_forward_scale": 0.10},
    {"front_late_release_phase_threshold": 0.65, "front_late_release_min_margin": 0.020, "front_late_release_hold_s": 0.04, "front_late_release_forward_scale": 0.05, "support_margin_preview_s": 0.04},
    {"front_touchdown_reacquire_hold_s": 0.08, "touchdown_reacquire_forward_scale": 0.35},
    {"front_touchdown_reacquire_hold_s": 0.12, "touchdown_reacquire_forward_scale": 0.25, "front_touchdown_reacquire_xy_blend": 0.55, "front_touchdown_reacquire_extra_depth": 0.012},
    {"front_touchdown_reacquire_hold_s": 0.16, "touchdown_reacquire_forward_scale": 0.15, "front_touchdown_reacquire_xy_blend": 0.75, "front_touchdown_reacquire_extra_depth": 0.016},
    {"front_touchdown_reacquire_hold_s": 0.12, "front_touchdown_reacquire_xy_blend": 0.55, "front_touchdown_reacquire_extra_depth": 0.012, "front_touchdown_reacquire_forward_bias": 0.010, "front_touchdown_confirm_hold_s": 0.06, "touchdown_confirm_forward_scale": 0.35},
    {"front_touchdown_reacquire_hold_s": 0.16, "front_touchdown_reacquire_xy_blend": 0.75, "front_touchdown_reacquire_extra_depth": 0.016, "front_touchdown_reacquire_forward_bias": 0.014, "front_touchdown_confirm_hold_s": 0.08, "touchdown_confirm_forward_scale": 0.25},
    {"front_touchdown_reacquire_hold_s": 0.12, "touchdown_reacquire_forward_scale": 0.25, "front_touchdown_reacquire_xy_blend": 0.55, "front_touchdown_reacquire_extra_depth": 0.012, "front_touchdown_settle_hold_s": 0.08, "touchdown_settle_forward_scale": 0.45},
    {"front_touchdown_reacquire_hold_s": 0.12, "touchdown_reacquire_forward_scale": 0.15, "front_touchdown_reacquire_xy_blend": 0.55, "front_touchdown_reacquire_extra_depth": 0.012, "front_touchdown_settle_hold_s": 0.12, "touchdown_settle_forward_scale": 0.35},
    {"front_touchdown_confirm_hold_s": 0.08, "front_touchdown_settle_hold_s": 0.10, "touchdown_support_rear_floor_delta": 0.10, "touchdown_support_vertical_boost": 0.10, "touchdown_support_z_pos_gain_delta": 4.0, "touchdown_support_pitch_angle_gain_delta": 8.0, "touchdown_support_pitch_rate_gain_delta": 2.0},
    {"front_touchdown_confirm_hold_s": 0.08, "front_touchdown_settle_hold_s": 0.12, "touchdown_support_rear_floor_delta": 0.15, "touchdown_support_vertical_boost": 0.10, "touchdown_support_z_pos_gain_delta": 4.0, "touchdown_support_pitch_angle_gain_delta": 8.0, "touchdown_support_pitch_rate_gain_delta": 2.0, "touchdown_support_rear_joint_pd_scale": 0.08, "touchdown_support_anchor_xy_blend": 0.75, "touchdown_support_anchor_z_blend": 0.50},
    {"front_touchdown_confirm_hold_s": 0.06, "front_touchdown_settle_hold_s": 0.10, "touchdown_support_rear_floor_delta": 0.10, "touchdown_support_vertical_boost": 0.15, "touchdown_support_z_pos_gain_delta": 6.0, "touchdown_support_pitch_angle_gain_delta": 12.0, "touchdown_support_pitch_rate_gain_delta": 3.0, "touchdown_support_anchor_xy_blend": 0.90, "touchdown_support_anchor_z_blend": 0.75},
    {"front_touchdown_confirm_hold_s": 0.10, "front_touchdown_settle_hold_s": 0.22, "touchdown_confirm_forward_scale": 0.20, "touchdown_settle_forward_scale": 0.20, "full_contact_recovery_hold_s": 0.10, "full_contact_recovery_forward_scale": 0.20, "full_contact_recovery_roll_threshold": 0.22, "full_contact_recovery_pitch_threshold": 0.15, "full_contact_recovery_height_ratio": 0.74, "full_contact_recovery_recent_window_s": 0.14},
    {"front_touchdown_confirm_hold_s": 0.12, "front_touchdown_settle_hold_s": 0.26, "touchdown_confirm_forward_scale": 0.15, "touchdown_settle_forward_scale": 0.15, "touchdown_support_rear_floor_delta": 0.15, "touchdown_support_vertical_boost": 0.15, "touchdown_support_z_pos_gain_delta": 6.0, "touchdown_support_roll_angle_gain_delta": 6.0, "touchdown_support_roll_rate_gain_delta": 2.0, "touchdown_support_pitch_angle_gain_delta": 10.0, "touchdown_support_pitch_rate_gain_delta": 3.0, "touchdown_support_rear_joint_pd_scale": 0.10, "touchdown_support_anchor_xy_blend": 0.90, "touchdown_support_anchor_z_blend": 0.75, "full_contact_recovery_hold_s": 0.14, "full_contact_recovery_forward_scale": 0.15, "full_contact_recovery_roll_threshold": 0.20, "full_contact_recovery_pitch_threshold": 0.14, "full_contact_recovery_height_ratio": 0.76, "full_contact_recovery_recent_window_s": 0.18},
    {"support_centroid_x_gain": 3.0},
    {"gait_step_freq": 0.40, "support_centroid_x_gain": 3.0},
    {"gait_step_freq": 0.40, "support_centroid_x_gain": 3.0, "pre_swing_front_shift_scale": 1.5},
    {"gait_step_freq": 0.40, "support_centroid_x_gain": 3.0, "pre_swing_front_shift_scale": 2.0},
    {"gait_step_freq": 0.40, "support_centroid_x_gain": 3.0, "pre_swing_front_shift_scale": 2.5, "support_margin_preview_s": 0.04},
    {"support_centroid_x_gain": 3.0, "rear_floor_base_scale": 0.70, "rear_floor_pitch_gain": 0.30, "pitch_angle_gain": 32.0},
    {"pitch_rebalance_gain": 0.08, "rear_floor_base_scale": 0.70, "rear_floor_pitch_gain": 0.30},
    {"front_latched_pitch_relief_gain": 0.25, "front_latched_rear_bias_gain": 1.0, "latched_swing_lift_ratio": 0.06},
    {"front_latched_pitch_relief_gain": 0.45, "front_latched_rear_bias_gain": 1.8, "latched_swing_lift_ratio": 0.06},
    {"support_floor_ratio": 0.22},
    {"grf_max_scale": 0.40},
    {"gait_duty_factor": 0.94},
    {"gait_duty_factor": 0.90, "contact_latch_budget_s": 0.08},
    {"gait_step_freq": 0.36},
    {"gait_step_freq": 0.40},
    {"contact_latch_budget_s": 0.12},
    {"support_centroid_y_gain": 10.0},
    {"support_centroid_y_gain": 12.0},
    {"support_centroid_x_gain": 2.0},
    {"support_centroid_x_gain": 3.0, "z_pos_gain": 24.0},
    {"latched_force_scale": 0.90},
    {"joint_pd_scale": 0.20, "latched_joint_pd_scale": 0.20},
    {"joint_pd_scale": 0.30, "latched_joint_pd_scale": 0.30},
    {"z_pos_gain": 24.0, "z_vel_gain": 6.0, "rear_floor_base_scale": 0.70},
    {"roll_angle_gain": 32.0, "roll_rate_gain": 8.0},
    {"min_vertical_force_scale": 0.55, "reduced_support_vertical_boost": 0.10},
]


def _slugify_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    freq = f"{float(params['gait_step_freq']):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    duty = f"{float(params['gait_duty_factor']):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    fmax = f"{float(params['grf_max_scale']):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"f{freq}_d{duty}_fz{fmax}_{digest}"


def _candidate_score(summary: dict[str, Any]) -> float:
    quality = summary.get("quality_gate", {})
    passes = float(bool(quality.get("passes", False)))
    duration = float(summary.get("duration_s") or 0.0)
    mean_vx = float(summary.get("mean_vx") or 0.0)
    mean_vy = float(summary.get("mean_vy") or 0.0)
    mean_abs_vy = abs(mean_vy)
    mean_base_z = float(summary.get("mean_base_z") or 0.0)
    min_base_z = float(summary.get("min_base_z") or 0.0)
    ref_height = float(summary.get("ref_base_height") or 0.0)
    mean_abs_roll = float(summary.get("mean_abs_roll") or 0.0)
    mean_abs_pitch = float(summary.get("mean_abs_pitch") or 0.0)
    swing_steps = float(summary.get("steps_any_actual_swing") or summary.get("steps_any_current_swing") or 0.0)
    legs_with_current_swing = float(summary.get("legs_with_actual_swing") or summary.get("legs_with_current_swing") or 0.0)
    current_realization_total = float(summary.get("current_swing_realization_total") or 0.0)
    actual_realization_total = float(summary.get("actual_swing_realization_total") or 0.0)
    front_current_realization = float(summary.get("front_current_swing_realization_mean") or 0.0)
    front_actual_realization = float(summary.get("front_actual_swing_realization_mean") or 0.0)
    front_touchdown_reacquire = float(summary.get("front_touchdown_reacquire_mean") or 0.0)
    front_touchdown_confirm = float(summary.get("front_touchdown_confirm_mean") or 0.0)
    front_touchdown_support = float(summary.get("front_touchdown_support_mean") or 0.0)
    swing_ratio = 0.0
    current_swing_ratio = summary.get("actual_swing_ratio") or summary.get("current_swing_ratio")
    if isinstance(current_swing_ratio, dict):
        valid = [float(v) for v in current_swing_ratio.values() if v is not None]
        if valid:
            swing_ratio = max(valid)

    height_ratio = (mean_base_z / ref_height) if ref_height > 1e-9 else 0.0
    min_height_ratio = (min_base_z / ref_height) if ref_height > 1e-9 else 0.0

    invalid_keys = summary.get("meta", {}).get("invalid_contact_keys", []) or []
    invalid_penalty = 0.0
    invalid_text = " ".join(str(x) for x in invalid_keys).lower()
    if "trunk" in invalid_text:
        invalid_penalty += 150.0
    if "fr_hip" in invalid_text or "fl_hip" in invalid_text:
        invalid_penalty += 140.0
    elif "hip" in invalid_text:
        invalid_penalty += 90.0
    if "thigh" in invalid_text:
        invalid_penalty += 70.0
    front_branch_valid = front_actual_realization >= 0.10 and front_touchdown_reacquire >= 0.01
    if front_actual_realization < 0.10:
        invalid_penalty += 5000.0
    if front_actual_realization < 0.05:
        invalid_penalty += 14000.0
    if front_current_realization < 0.10:
        invalid_penalty += 2200.0
    if front_current_realization < 0.05:
        invalid_penalty += 5000.0
    if front_touchdown_reacquire < 0.01:
        invalid_penalty += 6000.0
    if duration >= 2.5 and front_touchdown_reacquire < 0.01:
        invalid_penalty += 15000.0
    if duration >= 2.0 and front_actual_realization < 0.10:
        invalid_penalty += 9000.0

    swing_coverage = legs_with_current_swing / 4.0
    duration_ratio = np.clip(duration / 5.0, 0.0, 1.0)
    forward_term = max(mean_vx, 0.0)

    return (
        1_000_000.0 * passes
        + 20_000.0 * float(front_branch_valid)
        + 10_000.0 * duration_ratio
        + 1_800.0 * height_ratio
        + 1_300.0 * min_height_ratio
        + 900.0 * forward_term
        + 350.0 * swing_ratio
        + 250.0 * swing_coverage
        + 0.5 * swing_steps
        + 1_100.0 * current_realization_total
        + 700.0 * actual_realization_total
        + 1_300.0 * front_current_realization
        + 1_100.0 * front_actual_realization
        + 2_000.0 * front_touchdown_reacquire
        + 1_200.0 * front_touchdown_confirm
        + 900.0 * front_touchdown_support
        - 300.0 * mean_abs_roll
        - 220.0 * mean_abs_pitch
        - 500.0 * mean_abs_vy
        - invalid_penalty
    )


def _run_candidate(
    repo_root: Path,
    artifact_dir: Path,
    params: dict[str, Any],
    save_media: bool,
    timeout_s: int,
) -> dict[str, Any]:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "simulation.run_linear_osqp",
        "--gait",
        str(params["gait"]),
        "--preset",
        str(params["preset"]),
        "--seconds",
        str(int(params["seconds"])),
        "--speed",
        str(params.get("speed", 0.12)),
        "--lateral-speed",
        str(params.get("lateral_speed", 0.0)),
        "--yaw-rate",
        str(params.get("yaw_rate", 0.0)),
        "--seed",
        str(int(params.get("seed", 0))),
        "--artifact-dir",
        str(artifact_dir),
        "--q-theta",
        str(params["q_theta"]),
        "--q-w",
        str(params["q_w"]),
        "--r-u",
        str(params["r_u"]),
        "--du-xy-max",
        str(params["du_xy_max"]),
        "--support-floor-ratio",
        str(params["support_floor_ratio"]),
        "--grf-max-scale",
        str(params["grf_max_scale"]),
        "--joint-pd-scale",
        str(params["joint_pd_scale"]),
        "--latched-joint-pd-scale",
        str(params["latched_joint_pd_scale"]),
        "--gait-step-freq",
        str(params["gait_step_freq"]),
        "--gait-duty-factor",
        str(params["gait_duty_factor"]),
        "--stance-target-blend",
        str(params["stance_target_blend"]),
        "--contact-latch-budget-s",
        str(params["contact_latch_budget_s"]),
        "--startup-full-stance-time-s",
        str(params["startup_full_stance_time_s"]),
        "--latched-force-scale",
        str(params["latched_force_scale"]),
        "--latched-release-phase-start",
        str(params["latched_release_phase_start"]),
        "--latched-release-phase-end",
        str(params["latched_release_phase_end"]),
        "--fy-scale",
        str(params["fy_scale"]),
        "--support-centroid-x-gain",
        str(params["support_centroid_x_gain"]),
        "--support-centroid-y-gain",
        str(params["support_centroid_y_gain"]),
        "--pre-swing-lookahead-steps",
        str(params["pre_swing_lookahead_steps"]),
        "--pre-swing-front-shift-scale",
        str(params["pre_swing_front_shift_scale"]),
        "--pre-swing-rear-shift-scale",
        str(params["pre_swing_rear_shift_scale"]),
        "--pre-swing-gate-min-margin",
        str(params["pre_swing_gate_min_margin"]),
        "--front-pre-swing-gate-min-margin",
        str(params["front_pre_swing_gate_min_margin"]),
        "--rear-pre-swing-gate-min-margin",
        str(params["rear_pre_swing_gate_min_margin"]),
        "--front-late-release-phase-threshold",
        str(params["front_late_release_phase_threshold"]),
        "--front-late-release-min-margin",
        str(params["front_late_release_min_margin"]),
        "--front-late-release-hold-s",
        str(params["front_late_release_hold_s"]),
        "--front-late-release-forward-scale",
        str(params["front_late_release_forward_scale"]),
        "--support-margin-preview-s",
        str(params["support_margin_preview_s"]),
        "--front-late-release-extra-margin",
        str(params["front_late_release_extra_margin"]),
        "--front-late-release-pitch-guard",
        str(params["front_late_release_pitch_guard"]),
        "--front-late-release-roll-guard",
        str(params["front_late_release_roll_guard"]),
        "--touchdown-reacquire-hold-s",
        str(params["touchdown_reacquire_hold_s"]),
        "--front-touchdown-reacquire-hold-s",
        str(params["front_touchdown_reacquire_hold_s"]),
        "--touchdown-reacquire-forward-scale",
        str(params["touchdown_reacquire_forward_scale"]),
        "--touchdown-reacquire-xy-blend",
        str(params["touchdown_reacquire_xy_blend"]),
        "--front-touchdown-reacquire-xy-blend",
        str(params["front_touchdown_reacquire_xy_blend"]),
        "--touchdown-reacquire-extra-depth",
        str(params["touchdown_reacquire_extra_depth"]),
        "--front-touchdown-reacquire-extra-depth",
        str(params["front_touchdown_reacquire_extra_depth"]),
        "--touchdown-reacquire-forward-bias",
        str(params["touchdown_reacquire_forward_bias"]),
        "--front-touchdown-reacquire-forward-bias",
        str(params["front_touchdown_reacquire_forward_bias"]),
        "--touchdown-confirm-hold-s",
        str(params["touchdown_confirm_hold_s"]),
        "--front-touchdown-confirm-hold-s",
        str(params["front_touchdown_confirm_hold_s"]),
        "--touchdown-confirm-forward-scale",
        str(params["touchdown_confirm_forward_scale"]),
        "--touchdown-settle-hold-s",
        str(params["touchdown_settle_hold_s"]),
        "--front-touchdown-settle-hold-s",
        str(params["front_touchdown_settle_hold_s"]),
        "--touchdown-settle-forward-scale",
        str(params["touchdown_settle_forward_scale"]),
        "--touchdown-support-rear-floor-delta",
        str(params["touchdown_support_rear_floor_delta"]),
        "--touchdown-support-vertical-boost",
        str(params["touchdown_support_vertical_boost"]),
        "--touchdown-support-z-pos-gain-delta",
        str(params["touchdown_support_z_pos_gain_delta"]),
        "--touchdown-support-roll-angle-gain-delta",
        str(params["touchdown_support_roll_angle_gain_delta"]),
        "--touchdown-support-roll-rate-gain-delta",
        str(params["touchdown_support_roll_rate_gain_delta"]),
        "--touchdown-support-pitch-angle-gain-delta",
        str(params["touchdown_support_pitch_angle_gain_delta"]),
        "--touchdown-support-pitch-rate-gain-delta",
        str(params["touchdown_support_pitch_rate_gain_delta"]),
        "--touchdown-support-rear-joint-pd-scale",
        str(params["touchdown_support_rear_joint_pd_scale"]),
        "--touchdown-support-anchor-xy-blend",
        str(params["touchdown_support_anchor_xy_blend"]),
        "--touchdown-support-anchor-z-blend",
        str(params["touchdown_support_anchor_z_blend"]),
        "--full-contact-recovery-hold-s",
        str(params["full_contact_recovery_hold_s"]),
        "--full-contact-recovery-forward-scale",
        str(params["full_contact_recovery_forward_scale"]),
        "--full-contact-recovery-roll-threshold",
        str(params["full_contact_recovery_roll_threshold"]),
        "--full-contact-recovery-pitch-threshold",
        str(params["full_contact_recovery_pitch_threshold"]),
        "--full-contact-recovery-height-ratio",
        str(params["full_contact_recovery_height_ratio"]),
        "--full-contact-recovery-recent-window-s",
        str(params["full_contact_recovery_recent_window_s"]),
        "--pre-swing-gate-hold-s",
        str(params["pre_swing_gate_hold_s"]),
        "--pre-swing-gate-forward-scale",
        str(params["pre_swing_gate_forward_scale"]),
        "--vx-gain",
        str(params["vx_gain"]),
        "--vy-gain",
        str(params["vy_gain"]),
        "--z-pos-gain",
        str(params["z_pos_gain"]),
        "--z-vel-gain",
        str(params["z_vel_gain"]),
        "--min-vertical-force-scale",
        str(params["min_vertical_force_scale"]),
        "--reduced-support-vertical-boost",
        str(params["reduced_support_vertical_boost"]),
        "--roll-angle-gain",
        str(params["roll_angle_gain"]),
        "--roll-rate-gain",
        str(params["roll_rate_gain"]),
        "--pitch-angle-gain",
        str(params["pitch_angle_gain"]),
        "--pitch-rate-gain",
        str(params["pitch_rate_gain"]),
        "--rear-floor-base-scale",
        str(params["rear_floor_base_scale"]),
        "--rear-floor-pitch-gain",
        str(params["rear_floor_pitch_gain"]),
        "--pitch-rebalance-gain",
        str(params["pitch_rebalance_gain"]),
        "--pitch-rebalance-ref",
        str(params["pitch_rebalance_ref"]),
        "--front-latched-pitch-relief-gain",
        str(params["front_latched_pitch_relief_gain"]),
        "--front-latched-rear-bias-gain",
        str(params["front_latched_rear_bias_gain"]),
        "--latched-swing-lift-ratio",
        str(params["latched_swing_lift_ratio"]),
        "--latched-swing-tau-blend",
        str(params["latched_swing_tau_blend"]),
        "--virtual-unlatch-phase-threshold",
        str(params["virtual_unlatch_phase_threshold"]),
        "--virtual-unlatch-hold-s",
        str(params["virtual_unlatch_hold_s"]),
        "--no-plots",
    ]
    if not save_media:
        cmd.append("--no-mp4")
    else:
        cmd.extend(["--recording-path", str(artifact_dir / "h5")])

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    summary_path = artifact_dir / "episode_000" / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"run failed with code {result.returncode}\nstdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_stdout_tail"] = result.stdout[-2000:]
    summary["_stderr_tail"] = result.stderr[-2000:]
    return summary


def _mutate_params(rng: random.Random, best_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(best_params)
    keys = list(SEARCH_SPACE.keys())
    n_changes = rng.randint(1, 3)
    for key in rng.sample(keys, n_changes):
        params[key] = rng.choice(SEARCH_SPACE[key])
    return params


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential long-running autotuner for the linear OSQP MuJoCo setup.")
    parser.add_argument("--minutes", type=float, default=95.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means unlimited until the time budget expires.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--base-dir", type=str, default="outputs/autotune_session")
    parser.add_argument("--candidate-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = (repo_root / args.base_dir).resolve()
    scratch_dir = base_dir / "scratch"
    best_dir = base_dir / "best_run"
    history_path = base_dir / "history.jsonl"
    status_path = base_dir / "status.json"

    base_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    deadline = time.time() + max(args.minutes, 0.0) * 60.0

    best_params = dict(BASE_PARAMS)
    best_summary: dict[str, Any] | None = None
    best_score = float("-inf")
    run_index = 0

    seed_queue = [dict(BASE_PARAMS) | seed for seed in SEED_CANDIDATES]

    while time.time() < deadline:
        if args.max_runs > 0 and run_index >= args.max_runs:
            break

        if seed_queue:
            candidate = seed_queue.pop(0)
        else:
            parent = best_params if best_summary is not None else BASE_PARAMS
            candidate = _mutate_params(rng, parent)

        run_index += 1
        run_slug = f"run_{run_index:04d}_{_slugify_params(candidate)}"
        candidate_dir = scratch_dir / run_slug
        started_at = datetime.now().isoformat(timespec="seconds")

        record: dict[str, Any] = {
            "run_index": run_index,
            "started_at": started_at,
            "params": candidate,
        }
        try:
            summary = _run_candidate(
                repo_root=repo_root,
                artifact_dir=candidate_dir,
                params=candidate,
                save_media=False,
                timeout_s=args.candidate_timeout_seconds,
            )
            score = _candidate_score(summary)
            record["score"] = score
            record["summary"] = summary
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")

            if score > best_score:
                best_score = score
                best_params = dict(candidate)
                best_summary = summary
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(candidate_dir, best_dir)
                _write_json(
                    status_path,
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "best_score": best_score,
                        "best_params": best_params,
                        "best_summary": best_summary,
                        "best_run_path": str(best_dir),
                        "completed_runs": run_index,
                        "best_has_media": False,
                    },
                )
            else:
                _write_json(
                    status_path,
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "best_score": best_score,
                        "best_params": best_params,
                        "best_summary": best_summary,
                        "best_run_path": str(best_dir),
                        "completed_runs": run_index,
                        "best_has_media": False,
                    },
                )
        except Exception as exc:
            record["error"] = str(exc)
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        finally:
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir, ignore_errors=True)

    if best_params is not None and best_summary is not None:
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_summary = _run_candidate(
            repo_root=repo_root,
            artifact_dir=best_dir,
            params=best_params,
            save_media=True,
            timeout_s=args.candidate_timeout_seconds,
        )
        best_score = _candidate_score(best_summary)

    final_payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "best_score": best_score,
        "best_params": best_params,
        "best_summary": best_summary,
        "best_run_path": str(best_dir),
        "completed_runs": run_index,
        "best_has_media": bool(best_summary is not None),
        "finished": True,
    }
    _write_json(status_path, final_payload)
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
