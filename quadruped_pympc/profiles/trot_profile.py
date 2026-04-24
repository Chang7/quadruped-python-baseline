"""Trot (and other dynamic-gait) tuning profile for the linear OSQP controller.

Importable from any entry point (MuJoCo simulation runner, ROS2 node) — does
not depend on argparse or simulation-specific modules.
"""
from __future__ import annotations


def robot_posture_offsets() -> dict[str, float]:
    """Robot-specific posture reference offsets.

    Each robot has a different persistent posture bias due to mass distribution
    and leg geometry.  These offsets are determined empirically by checking the
    signed mean roll/pitch in a straight 20 s run and compensating.
    """
    from quadruped_pympc.config import robot
    if robot == "go1":
        return {"roll_ref_offset": 0.01, "pitch_ref_offset": -0.05}
    # aliengo (default) and others
    return {"roll_ref_offset": 0.03, "pitch_ref_offset": -0.03}


def trot_conservative_profile() -> dict[str, float | int]:
    """Single trot profile used for the straight/turn/disturbance checks."""
    offsets = robot_posture_offsets()
    return {
        "command_smoothing": 0.0,
        "Q_theta_roll": 160000.0,
        "Q_theta_pitch": 240000.0,
        "Q_w_roll": 16000.0,
        "Q_w_pitch": 24000.0,
        "vy_gain": 6.0,
        "vx_gain": 2.3,
        "fy_scale": 1.0,
        "dynamic_fy_roll_gain": 0.0,
        "dynamic_fy_roll_ref": 0.18,
        "foothold_yaw_rate_scale": 0.0,
        "foothold_yaw_error_scale": 0.0,
        "grf_max_scale": 1.0,
        "stance_ramp_steps": 1,
        "joint_pd_scale": 0.10,
        "stance_joint_pd_scale": 0.05,
        "latched_joint_pd_scale": 0.10,
        "rear_floor_base_scale": 0.65,
        "rear_floor_pitch_gain": 0.20,
        "support_reference_mix": 0.85,
        "support_reference_xy_mix": 1.0,
        "min_vertical_force_scale": 1.0,
        "reduced_support_vertical_boost": 0.40,
        "du_xy_max": 10.0,
        "du_z_max": 25.0,
        "side_rebalance_gain": 0.0,
        "side_rebalance_ref": 0.20,
        "pitch_rebalance_gain": 0.0,
        "pitch_angle_gain": 40.0,
        "pitch_rate_gain": 12.0,
        "pitch_rebalance_ref": 0.20,
        "roll_ref_offset": offsets["roll_ref_offset"],
        "pitch_ref_offset": offsets["pitch_ref_offset"],
        "pre_swing_gate_min_margin": 0.0,
        "front_pre_swing_gate_min_margin": 0.0,
        "rear_pre_swing_gate_min_margin": 0.0,
        "pre_swing_gate_hold_s": 0.0,
        "contact_latch_steps": 0,
        "contact_latch_budget_s": 0.0,
        "virtual_unlatch_hold_s": 0.0,
    }


def dynamic_gait_profile_for(gait: str) -> dict[str, float | int]:
    if gait not in {"trot", "pace", "bound"}:
        return {}
    return trot_conservative_profile()
