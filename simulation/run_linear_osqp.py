from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from quadruped_pympc import config as cfg
    from quadruped_pympc.profiles import dynamic_gait_profile_for
    from simulation.crawl_preset import (
        add_crawl_allcontact_cli_args,
        add_crawl_recovery_cli_args,
        add_crawl_support_bridge_cli_args,
        apply_crawl_allcontact_cli_overrides,
        apply_crawl_recovery_cli_overrides,
        apply_crawl_support_bridge_cli_overrides,
        crawl_conservative_params,
    )
    from simulation.simulation import run_simulation
else:
    from quadruped_pympc import config as cfg
    from quadruped_pympc.profiles import dynamic_gait_profile_for
    from .crawl_preset import (
        add_crawl_allcontact_cli_args,
        add_crawl_recovery_cli_args,
        add_crawl_support_bridge_cli_args,
        apply_crawl_allcontact_cli_overrides,
        apply_crawl_recovery_cli_overrides,
        apply_crawl_support_bridge_cli_overrides,
        crawl_conservative_params,
    )
    from .simulation import run_simulation


# CLI flags that map directly to linear_osqp_params entries.
# (flag, type, param_key) -- param_key=None means use flag attr as key.
_CLI_PARAM_OVERRIDES: tuple[tuple[str, type, str | None], ...] = (
    ("--q-p", float, "Q_p"),
    ("--q-v", float, "Q_v"),
    ("--q-theta", float, "Q_theta"),
    ("--q-theta-roll", float, "Q_theta_roll"),
    ("--q-theta-pitch", float, "Q_theta_pitch"),
    ("--q-w", float, "Q_w"),
    ("--q-w-roll", float, "Q_w_roll"),
    ("--q-w-pitch", float, "Q_w_pitch"),
    ("--r-u", float, "R_u"),
    ("--cmd-alpha", float, "command_smoothing"),
    ("--du-xy-max", float, None),
    ("--du-z-max", float, None),
    ("--stance-ramp-steps", int, None),
    ("--fy-scale", float, None),
    ("--dynamic-fy-roll-gain", float, None),
    ("--dynamic-fy-roll-ref", float, None),
    ("--grf-max-scale", float, None),
    ("--support-floor-ratio", float, "support_force_floor_ratio"),
    ("--joint-pd-scale", float, None),
    ("--stance-joint-pd-scale", float, None),
    ("--latched-joint-pd-scale", float, None),
    ("--latched-release-phase-start", float, None),
    ("--latched-release-phase-end", float, None),
    ("--stance-target-blend", float, None),
    ("--latched-force-scale", float, None),
    ("--latched-floor-scale", float, None),
    ("--latched-same-side-receiver-scale", float, None),
    ("--latched-axle-receiver-scale", float, None),
    ("--latched-diagonal-receiver-scale", float, None),
    ("--latched-front-receiver-scale", float, None),
    ("--latched-rear-receiver-scale", float, None),
    ("--rear-all-contact-front-planted-latched-force-scale-target", float, None),
    ("--rear-all-contact-front-planted-latched-front-receiver-scale-target", float, None),
    ("--rear-all-contact-front-planted-latched-rear-receiver-scale-target", float, None),
    ("--rear-all-contact-front-planted-support-floor-delta", float, None),
    ("--rear-all-contact-front-planted-rear-floor-delta", float, None),
    ("--rear-all-contact-front-planted-z-pos-gain-delta", float, None),
    ("--rear-all-contact-front-planted-roll-angle-gain-delta", float, None),
    ("--rear-all-contact-front-planted-roll-rate-gain-delta", float, None),
    ("--rear-all-contact-front-planted-side-rebalance-delta", float, None),
    ("--front-latched-pitch-relief-gain", float, None),
    ("--front-latched-rear-bias-gain", float, None),
    ("--rear-floor-base-scale", float, None),
    ("--rear-floor-pitch-gain", float, None),
    ("--side-rebalance-gain", float, None),
    ("--side-rebalance-ref", float, None),
    ("--pitch-rebalance-gain", float, None),
    ("--pitch-rebalance-ref", float, None),
    ("--support-centroid-x-gain", float, None),
    ("--support-centroid-y-gain", float, None),
    ("--pre-swing-front-shift-scale", float, None),
    ("--pre-swing-rear-shift-scale", float, None),
    ("--support-reference-mix", float, None),
    ("--support-reference-xy-mix", float, None),
    ("--vx-gain", float, None),
    ("--vy-gain", float, None),
    ("--pre-swing-gate-min-margin", float, None),
    ("--pre-swing-gate-hold-s", float, None),
    ("--rear-pre-swing-gate-hold-s", float, None),
    ("--rear-pre-swing-guard-roll-threshold", float, None),
    ("--rear-pre-swing-guard-pitch-threshold", float, None),
    ("--rear-pre-swing-guard-height-ratio", float, None),
    ("--pre-swing-gate-forward-scale", float, None),
    ("--front-late-release-phase-threshold", float, None),
    ("--front-late-release-min-margin", float, None),
    ("--front-late-release-hold-steps", int, None),
    ("--front-late-release-hold-s", float, None),
    ("--front-late-release-forward-scale", float, None),
    ("--support-margin-preview-s", float, None),
    ("--front-late-release-extra-margin", float, None),
    ("--front-late-release-pitch-guard", float, None),
    ("--front-late-release-roll-guard", float, None),
    ("--touchdown-reacquire-hold-s", float, None),
    ("--front-touchdown-reacquire-hold-s", float, None),
    ("--rear-touchdown-reacquire-hold-s", float, None),
    ("--touchdown-reacquire-forward-scale", float, None),
    ("--touchdown-reacquire-xy-blend", float, None),
    ("--front-touchdown-reacquire-xy-blend", float, None),
    ("--rear-touchdown-reacquire-xy-blend", float, None),
    ("--touchdown-reacquire-extra-depth", float, None),
    ("--front-touchdown-reacquire-extra-depth", float, None),
    ("--rear-touchdown-reacquire-extra-depth", float, None),
    ("--touchdown-reacquire-forward-bias", float, None),
    ("--front-touchdown-reacquire-forward-bias", float, None),
    ("--rear-touchdown-reacquire-forward-bias", float, None),
    ("--rear-touchdown-reacquire-min-swing-time-s", float, None),
    ("--rear-touchdown-reacquire-max-xy-shift", float, None),
    ("--rear-touchdown-reacquire-min-phase", float, None),
    ("--rear-touchdown-reacquire-upward-vel-damping", float, None),
    ("--rear-touchdown-retry-descent-depth", float, None),
    ("--rear-touchdown-retry-descent-kp", float, None),
    ("--rear-touchdown-retry-descent-kd", float, None),
    ("--rear-touchdown-contact-debounce-s", float, None),
    ("--rear-touchdown-contact-min-phase", float, None),
    ("--rear-touchdown-contact-max-upward-vel", float, None),
    ("--rear-touchdown-contact-min-grf-z", float, None),
    ("--rear-touchdown-close-lock-hold-s", float, None),
    ("--rear-touchdown-reacquire-retire-stance-hold-s", float, None),
    ("--front-crawl-swing-height-scale", float, None),
    ("--rear-crawl-swing-height-scale", float, None),
    ("--stance-anchor-update-alpha", float, None),
    ("--front-stance-anchor-update-alpha", float, None),
    ("--touchdown-support-anchor-update-alpha", float, None),
    ("--front-touchdown-support-anchor-update-alpha", float, None),
    ("--touchdown-confirm-hold-s", float, None),
    ("--front-touchdown-confirm-hold-s", float, None),
    ("--rear-touchdown-confirm-hold-s", float, None),
    ("--touchdown-confirm-forward-scale", float, None),
    ("--touchdown-settle-hold-s", float, None),
    ("--front-touchdown-settle-hold-s", float, None),
    ("--rear-touchdown-settle-hold-s", float, None),
    ("--touchdown-settle-forward-scale", float, None),
    ("--touchdown-support-rear-floor-delta", float, None),
    ("--touchdown-support-vertical-boost", float, None),
    ("--touchdown-support-min-vertical-force-scale-delta", float, None),
    ("--touchdown-support-grf-max-scale-delta", float, None),
    ("--touchdown-support-z-pos-gain-delta", float, None),
    ("--touchdown-support-roll-angle-gain-delta", float, None),
    ("--touchdown-support-roll-rate-gain-delta", float, None),
    ("--touchdown-support-pitch-angle-gain-delta", float, None),
    ("--touchdown-support-pitch-rate-gain-delta", float, None),
    ("--touchdown-support-side-rebalance-delta", float, None),
    ("--touchdown-support-front-joint-pd-scale", float, None),
    ("--touchdown-support-rear-joint-pd-scale", float, None),
    ("--touchdown-support-anchor-xy-blend", float, None),
    ("--touchdown-support-anchor-z-blend", float, None),
    ("--rear-touchdown-support-anchor-update-alpha", float, None),
    ("--rear-touchdown-support-support-floor-delta", float, None),
    ("--rear-touchdown-support-vertical-boost", float, None),
    ("--rear-touchdown-support-min-vertical-force-scale-delta", float, None),
    ("--rear-touchdown-support-grf-max-scale-delta", float, None),
    ("--rear-touchdown-support-z-pos-gain-delta", float, None),
    ("--rear-touchdown-support-roll-angle-gain-delta", float, None),
    ("--rear-touchdown-support-roll-rate-gain-delta", float, None),
    ("--rear-touchdown-support-pitch-angle-gain-delta", float, None),
    ("--rear-touchdown-support-pitch-rate-gain-delta", float, None),
    ("--rear-touchdown-support-side-rebalance-delta", float, None),
    ("--rear-touchdown-support-front-joint-pd-scale", float, None),
    ("--rear-touchdown-support-rear-joint-pd-scale", float, None),
    ("--rear-post-touchdown-support-hold-s", float, None),
    ("--rear-post-touchdown-support-forward-scale", float, None),
    ("--rear-post-touchdown-support-height-ratio", float, None),
    ("--rear-post-touchdown-support-roll-threshold", float, None),
    ("--rear-post-touchdown-support-pitch-threshold", float, None),
    ("--rear-post-touchdown-support-min-grf-z", float, None),
    ("--rear-post-touchdown-support-min-rear-load-share", float, None),
    ("--z-pos-gain", float, None),
    ("--z-vel-gain", float, None),
    ("--min-vertical-force-scale", float, None),
    ("--reduced-support-vertical-boost", float, None),
    ("--roll-angle-gain", float, None),
    ("--roll-rate-gain", float, None),
    ("--pitch-angle-gain", float, None),
    ("--pitch-rate-gain", float, None),
    ("--yaw-angle-gain", float, None),
    ("--yaw-rate-gain", float, None),
    ("--foothold-yaw-rate-scale", float, None),
    ("--foothold-yaw-error-scale", float, None),
    ("--roll-ref-offset", float, None),
    ("--pitch-ref-offset", float, None),
    ("--latched-swing-xy-blend", float, None),
    ("--latched-swing-lift-ratio", float, None),
    ("--latched-swing-tau-blend", float, None),
    ("--contact-latch-steps", int, None),
    ("--contact-latch-budget-s", float, None),
    ("--front-contact-latch-steps", int, None),
    ("--front-contact-latch-budget-s", float, None),
    ("--rear-contact-latch-steps", int, None),
    ("--rear-contact-latch-budget-s", float, None),
    ("--startup-full-stance-time-s", float, None),
    ("--virtual-unlatch-phase-threshold", float, None),
    ("--virtual-unlatch-hold-s", float, None),
    ("--pre-swing-lookahead-steps", int, None),
    ("--front-pre-swing-gate-min-margin", float, None),
    ("--rear-pre-swing-gate-min-margin", float, None),
    ("--support-contact-confirm-hold-s", float, None),
    ("--front-support-contact-confirm-hold-s", float, None),
    ("--rear-support-contact-confirm-hold-s", float, None),
    ("--support-confirm-forward-scale", float, None),
    ("--front-swing-contact-release-timeout-s", float, None),
    ("--rear-swing-contact-release-timeout-s", float, None),
    ("--front-release-lift-height", float, None),
    ("--front-release-lift-kp", float, None),
    ("--front-release-lift-kd", float, None),
    ("--rear-release-lift-height", float, None),
    ("--rear-release-lift-kp", float, None),
    ("--rear-release-lift-kd", float, None),
    ("--rear-swing-release-support-hold-s", float, None),
    ("--rear-swing-release-forward-scale", float, None),
)


_DISTURBANCE_AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "roll": 3,
    "pitch": 4,
    "yaw": 5,
}


def _parse_disturbance_pulses(specs: list[str] | None) -> list[dict[str, object]]:
    """Parse repeated disturbance pulse specs of the form axis:time:duration:magnitude."""
    schedule: list[dict[str, object]] = []
    if not specs:
        return schedule

    for raw_spec in specs:
        parts = [part.strip().lower() for part in raw_spec.split(":")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid disturbance pulse '{raw_spec}'. Expected axis:time:duration:magnitude."
            )
        axis_name, time_s_text, duration_s_text, magnitude_text = parts
        if axis_name not in _DISTURBANCE_AXIS_TO_INDEX:
            valid = ", ".join(_DISTURBANCE_AXIS_TO_INDEX.keys())
            raise ValueError(f"Invalid disturbance axis '{axis_name}'. Expected one of: {valid}.")

        time_s = float(time_s_text)
        duration_s = max(float(duration_s_text), 1e-6)
        magnitude = float(magnitude_text)
        wrench = [0.0] * 6
        wrench[_DISTURBANCE_AXIS_TO_INDEX[axis_name]] = magnitude
        schedule.append(
            {
                "time_s": time_s,
                "duration_s": duration_s,
                "wrench": wrench,
            }
        )
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quadruped-PyMPC with artifact logging and optional linear OSQP controller.")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gait", type=str, default="crawl", choices=("crawl", "trot", "pace", "bound", "full_stance"))
    parser.add_argument("--speed", type=float, default=0.12, help="Forward command in normalized env units used by simulation.py")
    parser.add_argument("--lateral-speed", type=float, default=0.0, help="Optional lateral command in normalized env units used only for the controller reference.")
    parser.add_argument("--preset", type=str, default="conservative", choices=("conservative", "baseline"), help="Use conservative low-level-friendly defaults for first stable runs.")
    parser.add_argument(
        "--linear-osqp-params-json",
        type=str,
        default=None,
        help="Optional JSON file with linear_osqp_params overrides applied after the selected preset and before explicit CLI flags.",
    )
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--step-height", type=float, default=None, help="Override swing step height in meters.")
    parser.add_argument(
        "--disturbance-pulse",
        action="append",
        default=[],
        help=(
            "Repeatable smooth external-wrench pulse in the form axis:time:duration:magnitude, "
            "for example --disturbance-pulse x:0.5:0.2:4.0"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", type=str, default="linear_osqp", choices=("linear_osqp", "nominal", "input_rates", "sampling"))
    parser.add_argument("--render", action="store_true", help="Show the MuJoCo viewer (can segfault on exit on some setups).")
    parser.add_argument("--recording-path", type=str, default="", help="Optional H5 dataset output directory.")
    parser.add_argument("--artifact-dir", type=str, default="outputs/linear_osqp_artifacts", help="Directory for npz/summary/png/mp4 artifacts.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--keep-running-after-terminate", action="store_true", help="By default the run stops on first termination for easier debugging.")
    parser.add_argument("--random-reset-on-terminate", action="store_true")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--rear-touchdown-reacquire-force-until-contact", action="store_true", help="Keep rear controller-side swing active during planned stance until actual contact really returns.")
    parser.add_argument("--front-touchdown-reacquire-hold-current-xy", action="store_true", help="During front touchdown reacquire, hold the current foot xy and search mainly downward instead of chasing the nominal foothold.")
    parser.add_argument("--rear-touchdown-reacquire-hold-current-xy", action="store_true", help="During rear touchdown reacquire, hold the current foot xy and mainly search downward instead of chasing the nominal foothold.")
    parser.add_argument("--rear-crawl-disable-reflex-swing", action="store_true", help="Ignore rear crawl early-stance reflex swing shaping.")
    parser.add_argument("--rear-touchdown-confirm-keep-swing", action="store_true", help="Keep rear controller-side swing active through the rear touchdown confirmation window so a flaky first contact does not reset swing.")
    add_crawl_allcontact_cli_args(parser)
    add_crawl_support_bridge_cli_args(parser)
    add_crawl_recovery_cli_args(parser)
    parser.add_argument("--gait-step-freq", type=float, default=None, help="Override selected gait step frequency.")
    parser.add_argument("--gait-duty-factor", type=float, default=None, help="Override selected gait duty factor.")
    parser.add_argument("--front-stance-dropout-reacquire", action="store_true", help="Reopen a short touchdown confirm/settle window on front legs if actual contact returns after a brief planned-stance dropout.")
    parser.add_argument("--rear-stance-dropout-reacquire", action="store_true", help="Reopen a short touchdown confirm/settle window on rear legs if actual contact returns after a brief planned-stance dropout.")
    parser.add_argument("--ground-friction", type=float, default=None, help="Optional fixed ground friction coefficient for MuJoCo.")
    parser.add_argument("--contact-condim", type=int, default=None, help="Optional MuJoCo contact condim override for floor and foot geoms.")
    parser.add_argument("--contact-impratio", type=float, default=None, help="Optional MuJoCo impratio override.")
    parser.add_argument("--contact-torsional-friction", type=float, default=None, help="Optional torsional friction override for floor and foot geoms.")
    parser.add_argument("--contact-rolling-friction", type=float, default=None, help="Optional rolling friction override for floor and foot geoms.")
    # Register table-driven CLI overrides
    for _flag, _typ, _key in _CLI_PARAM_OVERRIDES:
        parser.add_argument(_flag, type=_typ, default=None)
    # Step-to-seconds convenience args (special conversion, not in table)
    parser.add_argument("--contact-latch-budget-steps", type=int, default=None)
    parser.add_argument("--startup-full-stance-steps", type=int, default=None)
    parser.add_argument("--virtual-unlatch-hold-steps", type=int, default=None)
    args = parser.parse_args()
    disturbance_schedule = _parse_disturbance_pulses(args.disturbance_pulse)

    cfg.mpc_params["type"] = args.controller
    cfg.mpc_params["horizon"] = args.horizon
    cfg.mpc_params["dt"] = args.dt
    cfg.mpc_params["mu"] = args.mu
    cfg.mpc_params["optimize_step_freq"] = False
    cfg.mpc_params["use_nonuniform_discretization"] = False

    if hasattr(cfg, "linear_osqp_params"):
        if args.preset == "conservative":
            conservative_params = {
                "Q_p": 2e4,
                "Q_v": 4e4,
                "Q_theta": 2e4,
                "Q_w": 2e3,
                "R_u": 5.0,
                "command_smoothing": 0.35,
                "du_xy_max": 2.5,
                "du_z_max": 3.5,
                "stance_ramp_steps": 6,
                "fy_scale": 0.15,
                "grf_max_scale": 0.35,
                "joint_pd_scale": 0.25,
                "stance_joint_pd_scale": 0.0,
                "latched_joint_pd_scale": 0.25,
                "latched_release_phase_start": 0.0,
                "latched_release_phase_end": 1.0,
                "stance_target_blend": 0.0,
                "latched_force_scale": 1.0,
                "latched_floor_scale": 1.0,
                "latched_same_side_receiver_scale": 1.0,
                "latched_axle_receiver_scale": 1.0,
                "latched_diagonal_receiver_scale": 1.0,
                "latched_front_receiver_scale": 1.0,
                "latched_rear_receiver_scale": 1.0,
                "front_latched_pitch_relief_gain": 0.0,
                "front_latched_rear_bias_gain": 0.0,
                "rear_floor_base_scale": 0.65,
                "rear_floor_pitch_gain": 0.20,
                "side_rebalance_gain": 0.0,
                "side_rebalance_ref": 0.35,
                "pitch_rebalance_gain": 0.0,
                "pitch_rebalance_ref": 0.35,
                "support_centroid_x_gain": 0.0,
                "support_centroid_y_gain": 0.0,
                "pre_swing_gate_min_margin": 0.015,
                "front_pre_swing_gate_min_margin": None,
                "rear_pre_swing_gate_min_margin": None,
                "support_contact_confirm_hold_s": 0.0,
                "front_support_contact_confirm_hold_s": None,
                "rear_support_contact_confirm_hold_s": None,
                "support_confirm_forward_scale": 1.0,
                "front_stance_dropout_reacquire": True,
                "rear_stance_dropout_reacquire": False,
                "front_late_release_phase_threshold": 1.1,
                "front_late_release_min_margin": None,
                "front_late_release_hold_s": None,
                "front_late_release_forward_scale": 1.0,
                "support_margin_preview_s": 0.0,
                "front_late_release_extra_margin": 0.0,
                "front_late_release_pitch_guard": None,
                "front_late_release_roll_guard": None,
                "touchdown_reacquire_hold_s": 0.0,
                "front_touchdown_reacquire_hold_s": None,
                "rear_touchdown_reacquire_hold_s": None,
                "touchdown_reacquire_forward_scale": 1.0,
                "touchdown_reacquire_xy_blend": 0.0,
                "front_touchdown_reacquire_xy_blend": None,
                "rear_touchdown_reacquire_xy_blend": None,
                "touchdown_reacquire_extra_depth": 0.0,
                "front_touchdown_reacquire_extra_depth": None,
                "rear_touchdown_reacquire_extra_depth": None,
                "touchdown_reacquire_forward_bias": 0.0,
                "front_touchdown_reacquire_forward_bias": None,
                "rear_touchdown_reacquire_forward_bias": None,
                "stance_anchor_update_alpha": 0.0,
                "front_stance_anchor_update_alpha": None,
                "rear_stance_anchor_update_alpha": None,
                "touchdown_confirm_hold_s": 0.0,
                "front_touchdown_confirm_hold_s": None,
                "rear_touchdown_confirm_hold_s": None,
                "touchdown_confirm_forward_scale": 1.0,
                "touchdown_settle_hold_s": 0.0,
                "front_touchdown_settle_hold_s": None,
                "rear_touchdown_settle_hold_s": None,
                "touchdown_settle_forward_scale": 1.0,
                "touchdown_support_rear_floor_delta": 0.0,
                "touchdown_support_vertical_boost": 0.0,
                "touchdown_support_min_vertical_force_scale_delta": 0.0,
                "touchdown_support_grf_max_scale_delta": 0.0,
                "touchdown_support_z_pos_gain_delta": 0.0,
                "touchdown_support_roll_angle_gain_delta": 0.0,
                "touchdown_support_roll_rate_gain_delta": 0.0,
                "touchdown_support_pitch_angle_gain_delta": 0.0,
                "touchdown_support_pitch_rate_gain_delta": 0.0,
                "touchdown_support_side_rebalance_delta": 0.0,
                "touchdown_support_front_joint_pd_scale": 0.0,
                "touchdown_support_rear_joint_pd_scale": 0.0,
                "touchdown_support_anchor_xy_blend": 0.0,
                "touchdown_support_anchor_z_blend": 0.0,
                "rear_touchdown_support_anchor_update_alpha": None,
                "rear_touchdown_support_support_floor_delta": 0.0,
                "rear_touchdown_support_vertical_boost": 0.0,
                "rear_touchdown_support_min_vertical_force_scale_delta": 0.0,
                "rear_touchdown_support_grf_max_scale_delta": 0.0,
                "rear_touchdown_support_z_pos_gain_delta": 0.0,
                "rear_touchdown_support_roll_angle_gain_delta": 0.0,
                "rear_touchdown_support_roll_rate_gain_delta": 0.0,
                "rear_touchdown_support_pitch_angle_gain_delta": 0.0,
                "rear_touchdown_support_pitch_rate_gain_delta": 0.0,
                "rear_touchdown_support_side_rebalance_delta": 0.0,
                "rear_touchdown_support_front_joint_pd_scale": 0.0,
                "rear_touchdown_support_rear_joint_pd_scale": 0.0,
                "touchdown_contact_vel_z_damping": 0.0,
                "rear_touchdown_contact_vel_z_damping": None,
                "front_margin_rescue_hold_s": 0.0,
                "front_margin_rescue_forward_scale": 1.0,
                "front_margin_rescue_min_margin": 0.0,
                "front_margin_rescue_margin_gap": 0.0,
                "front_margin_rescue_alpha_margin": 0.02,
                "front_margin_rescue_roll_threshold": None,
                "front_margin_rescue_pitch_threshold": None,
                "front_margin_rescue_height_ratio": 0.0,
                "front_margin_rescue_recent_swing_window_s": 0.0,
                "front_margin_rescue_require_all_contact": True,
                "rear_handoff_support_hold_s": 0.0,
                "rear_handoff_forward_scale": 1.0,
                "rear_handoff_lookahead_steps": 1,
                "rear_swing_bridge_hold_s": 0.0,
                "rear_swing_bridge_forward_scale": 1.0,
                "rear_swing_bridge_roll_threshold": None,
                "rear_swing_bridge_pitch_threshold": None,
                "rear_swing_bridge_height_ratio": 0.0,
                "rear_swing_bridge_recent_front_window_s": 0.0,
                "rear_swing_bridge_lookahead_steps": 1,
                "full_contact_recovery_hold_s": 0.0,
                "full_contact_recovery_forward_scale": 1.0,
                "full_contact_recovery_roll_threshold": None,
                "full_contact_recovery_pitch_threshold": None,
                "full_contact_recovery_height_ratio": 0.0,
                "full_contact_recovery_recent_window_s": 0.0,
                "full_contact_recovery_support_floor_delta": 0.0,
                "full_contact_recovery_min_vertical_force_scale_delta": 0.0,
                "full_contact_recovery_z_pos_gain_delta": 0.0,
                "full_contact_recovery_roll_angle_gain_delta": 0.0,
                "full_contact_recovery_roll_rate_gain_delta": 0.0,
                "full_contact_recovery_pitch_angle_gain_delta": 0.0,
                "full_contact_recovery_pitch_rate_gain_delta": 0.0,
                "full_contact_recovery_side_rebalance_delta": 0.0,
                "crawl_front_delayed_swing_recovery_hold_s": 0.0,
                "crawl_front_stance_support_tail_hold_s": 0.0,
                "rear_all_contact_stabilization_rear_floor_delta": 0.0,
                "rear_all_contact_stabilization_z_pos_gain_delta": 0.0,
                "rear_all_contact_stabilization_roll_angle_gain_delta": 0.0,
                "rear_all_contact_stabilization_roll_rate_gain_delta": 0.0,
                "rear_all_contact_stabilization_side_rebalance_delta": 0.0,
                "rear_all_contact_stabilization_front_anchor_z_blend": 0.0,
                "rear_all_contact_stabilization_rear_anchor_z_blend": 0.0,
                "rear_all_contact_stabilization_front_anchor_z_max_delta": 0.0,
                "rear_all_contact_stabilization_rear_anchor_z_max_delta": 0.0,
                "pre_swing_front_shift_scale": 1.0,
                "pre_swing_rear_shift_scale": 1.0,
                "pre_swing_gate_hold_s": 0.08,
                "pre_swing_gate_forward_scale": 1.0,
                "vx_gain": 1.6,
                "vy_gain": 4.5,
                "z_pos_gain": 20.0,
                "z_vel_gain": 5.0,
                "min_vertical_force_scale": 0.5,
                "reduced_support_vertical_boost": 0.0,
                "roll_angle_gain": 24.0,
                "roll_rate_gain": 6.0,
                "pitch_angle_gain": 28.0,
                "pitch_rate_gain": 8.0,
                "yaw_angle_gain": 4.0,
                "yaw_rate_gain": 1.5,
                "latched_swing_xy_blend": 0.0,
                "latched_swing_lift_ratio": 0.0,
                "latched_swing_tau_blend": 0.0,
                "contact_latch_steps": 6,
                "contact_latch_budget_s": 0.06,
                "front_contact_latch_steps": None,
                "front_contact_latch_budget_s": None,
                "startup_full_stance_time_s": 0.20,
                "virtual_unlatch_phase_threshold": 1.1,
                "virtual_unlatch_hold_s": 0.0,
                "pre_swing_lookahead_steps": 3,
            }
            if args.gait == "crawl":
                conservative_params.update(crawl_conservative_params())
            if args.gait in {"trot", "pace", "bound"}:
                conservative_params.update(dynamic_gait_profile_for(args.gait))
            cfg.linear_osqp_params.update(conservative_params)
        if args.linear_osqp_params_json:
            params_path = Path(args.linear_osqp_params_json).expanduser().resolve()
            with params_path.open("r", encoding="utf-8") as f:
                loaded_params = json.load(f)
            if not isinstance(loaded_params, dict):
                raise ValueError(
                    f"--linear-osqp-params-json must point to a JSON object, got {type(loaded_params).__name__}."
                )
            cfg.linear_osqp_params.update(loaded_params)
        if args.front_stance_dropout_reacquire:
            cfg.linear_osqp_params["front_stance_dropout_reacquire"] = True
        if args.rear_stance_dropout_reacquire:
            cfg.linear_osqp_params["rear_stance_dropout_reacquire"] = True
        if args.rear_touchdown_reacquire_force_until_contact:
            cfg.linear_osqp_params["rear_touchdown_reacquire_force_until_contact"] = True
        if args.front_touchdown_reacquire_hold_current_xy:
            cfg.linear_osqp_params["front_touchdown_reacquire_hold_current_xy"] = True
        if args.rear_touchdown_reacquire_hold_current_xy:
            cfg.linear_osqp_params["rear_touchdown_reacquire_hold_current_xy"] = True
        if args.rear_crawl_disable_reflex_swing:
            cfg.linear_osqp_params["rear_crawl_disable_reflex_swing"] = True
        if args.rear_touchdown_confirm_keep_swing:
            cfg.linear_osqp_params["rear_touchdown_confirm_keep_swing"] = True
        if args.rear_all_contact_stabilization_front_anchor_z_blend is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_front_anchor_z_blend"] = float(
                max(min(args.rear_all_contact_stabilization_front_anchor_z_blend, 1.0), 0.0)
            )
        if args.rear_all_contact_stabilization_rear_anchor_z_blend is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_rear_anchor_z_blend"] = float(
                max(min(args.rear_all_contact_stabilization_rear_anchor_z_blend, 1.0), 0.0)
            )
        if args.rear_all_contact_stabilization_front_anchor_z_max_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_front_anchor_z_max_delta"] = max(
                float(args.rear_all_contact_stabilization_front_anchor_z_max_delta),
                0.0,
            )
        if args.rear_all_contact_stabilization_rear_anchor_z_max_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_rear_anchor_z_max_delta"] = max(
                float(args.rear_all_contact_stabilization_rear_anchor_z_max_delta),
                0.0,
            )
        if args.rear_all_contact_post_recovery_tail_hold_s is not None:
            cfg.linear_osqp_params["rear_all_contact_post_recovery_tail_hold_s"] = max(
                float(args.rear_all_contact_post_recovery_tail_hold_s), 0.0
            )
        if args.rear_all_contact_release_tail_alpha_scale is not None:
            cfg.linear_osqp_params["rear_all_contact_release_tail_alpha_scale"] = float(
                max(min(args.rear_all_contact_release_tail_alpha_scale, 1.0), 0.0)
            )
        if args.rear_all_contact_post_recovery_front_late_alpha_scale is not None:
            cfg.linear_osqp_params["rear_all_contact_post_recovery_front_late_alpha_scale"] = float(
                max(min(args.rear_all_contact_post_recovery_front_late_alpha_scale, 1.0), 0.0)
            )
        apply_crawl_allcontact_cli_overrides(args, cfg.linear_osqp_params)
        apply_crawl_support_bridge_cli_overrides(args, cfg.linear_osqp_params)
        apply_crawl_recovery_cli_overrides(args, cfg.linear_osqp_params)
        # Table-driven CLI overrides
        for _flag, _typ, _key in _CLI_PARAM_OVERRIDES:
            _attr = _flag.lstrip("-").replace("-", "_")
            _val = getattr(args, _attr, None)
            if _val is not None:
                cfg.linear_osqp_params[_key or _attr] = _typ(_val)
        # Step-to-seconds conversions (special: store both forms)
        _dt = float(cfg.mpc_params["dt"])
        if args.contact_latch_budget_steps is not None:
            cfg.linear_osqp_params["contact_latch_budget_s"] = max(int(args.contact_latch_budget_steps), 0) * _dt
            cfg.linear_osqp_params["contact_latch_budget_steps"] = args.contact_latch_budget_steps
        if args.startup_full_stance_steps is not None:
            cfg.linear_osqp_params["startup_full_stance_time_s"] = max(int(args.startup_full_stance_steps), 0) * _dt
            cfg.linear_osqp_params["startup_full_stance_steps"] = args.startup_full_stance_steps
        if args.virtual_unlatch_hold_steps is not None:
            cfg.linear_osqp_params["virtual_unlatch_hold_s"] = max(int(args.virtual_unlatch_hold_steps), 0) * _dt
            cfg.linear_osqp_params["virtual_unlatch_hold_steps"] = args.virtual_unlatch_hold_steps
    cfg.simulation_params["gait"] = args.gait
    if args.step_height is not None:
        cfg.simulation_params["step_height"] = max(float(args.step_height), 0.0)
    elif args.preset == "conservative" and args.gait in {"trot", "pace", "bound"}:
        # A lower default step height keeps the dynamic-gait swing motion more
        # compact and consistently improved short-horizon trot quality.
        cfg.simulation_params["step_height"] = 0.04
    if args.preset == "conservative" and args.gait == "crawl":
        if args.gait_step_freq is None:
            cfg.simulation_params["gait_params"][args.gait]["step_freq"] = 0.40
        if args.gait_duty_factor is None:
            cfg.simulation_params["gait_params"][args.gait]["duty_factor"] = 0.90
    if args.gait_step_freq is not None:
        cfg.simulation_params["gait_params"][args.gait]["step_freq"] = args.gait_step_freq
    if args.gait_duty_factor is not None:
        cfg.simulation_params["gait_params"][args.gait]["duty_factor"] = args.gait_duty_factor
    cfg.simulation_params["mode"] = "forward" if abs(args.yaw_rate) < 1e-9 else "forward+rotate"
    cfg.simulation_params["mpc_frequency"] = int(round(1.0 / max(args.dt, 1e-6)))
    if args.contact_condim is not None:
        cfg.simulation_params["contact_condim_override"] = max(int(args.contact_condim), 1)
    if args.contact_impratio is not None:
        cfg.simulation_params["contact_impratio_override"] = max(float(args.contact_impratio), 1.0)
    if args.contact_torsional_friction is not None:
        cfg.simulation_params["contact_torsional_friction_override"] = max(float(args.contact_torsional_friction), 0.0)
    if args.contact_rolling_friction is not None:
        cfg.simulation_params["contact_rolling_friction_override"] = max(float(args.contact_rolling_friction), 0.0)

    recording = Path(args.recording_path) if args.recording_path else None
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    command_speed = math.hypot(float(args.speed), float(args.lateral_speed))
    controller_ref_base_lin_vel = (
        cfg.hip_height * float(args.speed),
        cfg.hip_height * float(args.lateral_speed),
        0.0,
    )
    controller_ref_base_ang_vel = (0.0, 0.0, float(args.yaw_rate))
    friction_coeff = (0.5, 1.0)
    if args.ground_friction is not None:
        friction_value = max(float(args.ground_friction), 1e-6)
        friction_coeff = (friction_value, friction_value)

    output = run_simulation(
        cfg,
        num_episodes=args.episodes,
        num_seconds_per_episode=args.seconds,
        ref_base_lin_vel=(command_speed, command_speed),
        ref_base_ang_vel=(args.yaw_rate, args.yaw_rate),
        friction_coeff=friction_coeff,
        base_vel_command_type="forward" if abs(args.yaw_rate) < 1e-9 else "forward+rotate",
        seed=args.seed,
        render=args.render,
        recording_path=recording,
        artifact_dir=artifact_dir,
        save_plots_flag=not args.no_plots,
        save_mp4_flag=not args.no_mp4,
        stop_on_terminate=not args.keep_running_after_terminate,
        random_reset_on_terminate=args.random_reset_on_terminate,
        controller_ref_base_lin_vel=controller_ref_base_lin_vel,
        controller_ref_base_ang_vel=controller_ref_base_ang_vel,
        disturbance_schedule=disturbance_schedule,
    )
    if output is not None:
        print(f"\nPrimary output path: {output}")


if __name__ == "__main__":
    main()
