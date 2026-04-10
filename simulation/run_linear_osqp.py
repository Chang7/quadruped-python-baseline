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
    from simulation.simulation import run_simulation
else:
    from quadruped_pympc import config as cfg
    from .simulation import run_simulation


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


def _dynamic_gait_conservative_profile() -> dict[str, float | int]:
    """Dynamic-gait profile that remains usable across straight, turn, and disturbance checks."""
    return {
        "command_smoothing": 0.0,
        # Axis-specific orientation / angular-rate Q weighting was the cleanest
        # way to improve generic trot posture without falling back to heavier
        # post-hoc heuristics. Pitch weighting gave the first large gain, and
        # additional roll weighting reduced the remaining turn/disturbance roll
        # drift while preserving forward tracking over longer horizons too.
        "Q_theta_roll": 160000.0,
        "Q_theta_pitch": 240000.0,
        "Q_w_roll": 16000.0,
        "Q_w_pitch": 24000.0,
        "vy_gain": 6.0,
        "vx_gain": 2.3,
        "fy_scale": 1.0,
        # Roll-proportional lateral force for centripetal roll compensation.
        # Tested 0.0/0.15/0.20/0.25: monotonic improvement on turn roll
        # (0.024->0.018) with no straight/disturbance regression. 0.25 matches
        # straight_tuned. Symmetric Q_theta_roll (160k->240k) was tested and
        # rejected: no turn roll effect, disturbance regression -- confirming
        # the gap is lateral force authority, not MPC cost weighting.
        "dynamic_fy_roll_gain": 0.25,
        "dynamic_fy_roll_ref": 0.18,
        # Turn-specific foothold yaw compensation helps the linear controller
        # generate a clearer turning geometry without affecting straight-line
        # gait logic.
        "foothold_yaw_rate_scale": 0.0,
        "foothold_yaw_error_scale": 0.0,
        "grf_max_scale": 1.0,
        "stance_ramp_steps": 1,
        "joint_pd_scale": 0.10,
        "stance_joint_pd_scale": 0.05,
        "latched_joint_pd_scale": 0.10,
        "rear_floor_base_scale": 0.65,
        "rear_floor_pitch_gain": 0.20,
        # A stronger support-wrench blend improves short-horizon turn and
        # disturbance posture quality without changing the straight-tuned
        # profile used for longer straight-line trot checks.
        "support_reference_mix": 0.85,
        # Keep the more posture-friendly vertical blend while allowing the
        # horizontal support reference to follow the solved wrench more
        # aggressively for better forward tracking.
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
        # The remaining dynamic-gait error is a steady posture bias (mean signed
        # pitch = mean |pitch|, not oscillating). roll_ref_offset=+0.03 was the
        # original fix. pitch_ref_offset was then swept from -0.01 to -0.03:
        # the stronger offset cuts mean |pitch| by 27-40% across all scenarios,
        # now beating stock pitch in every trot check.
        "roll_ref_offset": 0.03,
        # The remaining dynamic-gait error is a steady posture bias (mean signed
        # pitch = mean |pitch|, not oscillating). roll_ref_offset=+0.03 was the
        # original fix. pitch_ref_offset was then swept from -0.01 to -0.03:
        # the stronger offset cuts mean |pitch| by 27-40% across all scenarios,
        # now beating stock pitch in every trot check.
        "pitch_ref_offset": -0.03,
        "pre_swing_gate_min_margin": 0.0,
        "front_pre_swing_gate_min_margin": 0.0,
        "rear_pre_swing_gate_min_margin": 0.0,
        "pre_swing_gate_hold_s": 0.0,
        "contact_latch_steps": 0,
        "contact_latch_budget_s": 0.0,
        "virtual_unlatch_hold_s": 0.0,
    }


def _trot_straight_tuned_profile() -> dict[str, float | int]:
    """Straight-line trot profile tuned for longer-horizon forward tracking."""
    profile = _dynamic_gait_conservative_profile()
    profile.update(
        {
            "vx_gain": 5.5,
            "fy_scale": 0.35,
            "dynamic_fy_roll_gain": 0.25,
            "stance_joint_pd_scale": 0.10,
            "joint_pd_scale": 0.10,
            "latched_joint_pd_scale": 0.10,
            "rear_floor_base_scale": 0.50,
            "rear_floor_pitch_gain": 0.0,
            "support_reference_mix": 0.75,
            "support_reference_xy_mix": 1.0,
            "reduced_support_vertical_boost": 0.30,
            "pitch_angle_gain": 37.0,
            "pitch_rate_gain": 11.0,
            "support_centroid_x_gain": 1.0,
        }
    )
    return profile


def _dynamic_gait_profile_for(gait: str, trot_profile: str) -> dict[str, float | int]:
    if gait not in {"trot", "pace", "bound"}:
        return {}
    if gait == "trot" and trot_profile == "straight_tuned":
        return _trot_straight_tuned_profile()
    return _dynamic_gait_conservative_profile()


def _resolve_dynamic_trot_profile(
    gait: str,
    requested_profile: str,
    yaw_rate: float,
    lateral_speed: float,
    disturbance_schedule: list[dict[str, object]],
) -> str:
    """Resolve the effective trot dynamic profile from the requested mode.

    `auto` is intentionally conservative and deterministic:
    - straight-line trot without lateral/yaw/disturbance -> `straight_tuned`
    - otherwise -> `generic`
    """
    if gait != "trot" or requested_profile != "auto":
        return requested_profile
    if abs(float(yaw_rate)) < 1e-9 and abs(float(lateral_speed)) < 1e-9 and not disturbance_schedule:
        return "straight_tuned"
    return "generic"


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
    parser.add_argument(
        "--dynamic-trot-profile",
        type=str,
        default="auto",
        choices=("auto", "generic", "straight_tuned"),
        help="For trot only: use the generic all-scenario profile, the straight-line-tuned profile, or let auto pick between them from the command.",
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
    parser.add_argument("--q-p", type=float, default=None)
    parser.add_argument("--q-v", type=float, default=None)
    parser.add_argument("--q-theta", type=float, default=None)
    parser.add_argument("--q-theta-roll", type=float, default=None, help="Optional roll-axis override for the orientation-state weight.")
    parser.add_argument("--q-theta-pitch", type=float, default=None, help="Optional pitch-axis override for the orientation-state weight.")
    parser.add_argument("--q-w", type=float, default=None)
    parser.add_argument("--q-w-roll", type=float, default=None, help="Optional roll-axis override for the angular-rate-state weight.")
    parser.add_argument("--q-w-pitch", type=float, default=None, help="Optional pitch-axis override for the angular-rate-state weight.")
    parser.add_argument("--r-u", type=float, default=None)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--cmd-alpha", type=float, default=None, help="Override command smoothing factor.")
    parser.add_argument("--du-xy-max", type=float, default=None, help="Override per-step fx/fy slew limit [N].")
    parser.add_argument("--du-z-max", type=float, default=None, help="Override per-step fz slew limit [N].")
    parser.add_argument("--stance-ramp-steps", type=int, default=None, help="Override stance ramp length in controller steps.")
    parser.add_argument("--fy-scale", type=float, default=None, help="Override lateral-force scaling applied after solve.")
    parser.add_argument("--dynamic-fy-roll-gain", type=float, default=None, help="Extra lateral-force authority that ramps in with absolute roll during dynamic gaits.")
    parser.add_argument("--dynamic-fy-roll-ref", type=float, default=None, help="Roll angle [rad] that saturates the temporary dynamic lateral-force boost.")
    parser.add_argument("--grf-max-scale", type=float, default=None, help="Override effective normal-force upper bound as fraction of body weight budget.")
    parser.add_argument("--support-floor-ratio", type=float, default=None, help="Override minimum per-stance-leg normal force as fraction of body weight / n_stance.")
    parser.add_argument("--joint-pd-scale", type=float, default=None, help="Override additional low-level joint PD blend for linear_osqp.")
    parser.add_argument("--stance-joint-pd-scale", type=float, default=None, help="Override low-level joint PD blend kept on stance legs for linear_osqp.")
    parser.add_argument("--latched-joint-pd-scale", type=float, default=None, help="Override joint PD blend for planned-swing legs kept relatched in contact.")
    parser.add_argument("--latched-release-phase-start", type=float, default=None, help="Start of the swing-progress window where relatched planned-swing legs are relaxed.")
    parser.add_argument("--latched-release-phase-end", type=float, default=None, help="End of the swing-progress window where relatched planned-swing legs are relaxed.")
    parser.add_argument("--stance-target-blend", type=float, default=None, help="Blend stance-foot xy target toward the next foothold for linear_osqp.")
    parser.add_argument("--latched-force-scale", type=float, default=None, help="Scale force on planned-swing legs that remain relatched in contact.")
    parser.add_argument("--latched-floor-scale", type=float, default=None, help="Scale the minimum vertical-force floor on relatched planned-swing legs.")
    parser.add_argument("--latched-same-side-receiver-scale", type=float, default=None, help="Bias released load away from same-side support legs.")
    parser.add_argument("--latched-axle-receiver-scale", type=float, default=None, help="Bias released load toward the same front/rear pair as the relatched leg.")
    parser.add_argument("--latched-diagonal-receiver-scale", type=float, default=None, help="Bias released load toward the diagonal support leg.")
    parser.add_argument("--latched-front-receiver-scale", type=float, default=None, help="Extra weight for front support legs when redistributing released load.")
    parser.add_argument("--latched-rear-receiver-scale", type=float, default=None, help="Extra weight for rear support legs when redistributing released load.")
    parser.add_argument("--rear-all-contact-front-planted-latched-force-scale-target", type=float, default=None, help="During the narrow crawl front-planted tail, blend relatched planned-swing-leg force scale toward this target.")
    parser.add_argument("--rear-all-contact-front-planted-latched-front-receiver-scale-target", type=float, default=None, help="During the same crawl front-planted tail, blend released-load redistribution weight on front support legs toward this target.")
    parser.add_argument("--rear-all-contact-front-planted-latched-rear-receiver-scale-target", type=float, default=None, help="During the same crawl front-planted tail, blend released-load redistribution weight on rear support legs toward this target.")
    parser.add_argument("--rear-all-contact-front-planted-support-floor-delta", type=float, default=None, help="During the narrow crawl front-planted tail, temporarily raise the global stance support floor by this amount.")
    parser.add_argument("--rear-all-contact-front-planted-rear-floor-delta", type=float, default=None, help="During the same crawl front-planted tail, temporarily raise only the rear-load floor by this amount.")
    parser.add_argument("--rear-all-contact-front-planted-z-pos-gain-delta", type=float, default=None, help="During the same crawl front-planted tail, temporarily raise the base-height gain by this amount.")
    parser.add_argument("--rear-all-contact-front-planted-roll-angle-gain-delta", type=float, default=None, help="During the same crawl front-planted tail, temporarily raise the roll-angle gain by this amount.")
    parser.add_argument("--rear-all-contact-front-planted-roll-rate-gain-delta", type=float, default=None, help="During the same crawl front-planted tail, temporarily raise the roll-rate gain by this amount.")
    parser.add_argument("--rear-all-contact-front-planted-side-rebalance-delta", type=float, default=None, help="During the same crawl front-planted tail, temporarily raise the lateral support rebalance gain by this amount.")
    parser.add_argument("--front-latched-pitch-relief-gain", type=float, default=None, help="Extra unload gain applied only when a front planned-swing leg stays latched while the body pitches forward.")
    parser.add_argument("--front-latched-rear-bias-gain", type=float, default=None, help="Extra rearward redistribution applied only when a front planned-swing leg stays latched.")
    parser.add_argument("--rear-floor-base-scale", type=float, default=None, help="Baseline share of vertical load kept on rear support legs.")
    parser.add_argument("--rear-floor-pitch-gain", type=float, default=None, help="Additional rear-load bias as pitch error grows.")
    parser.add_argument("--side-rebalance-gain", type=float, default=None, help="Extra signed left/right load rebalance driven by roll sign.")
    parser.add_argument("--side-rebalance-ref", type=float, default=None, help="Roll angle in radians that saturates the signed left/right load rebalance.")
    parser.add_argument("--pitch-rebalance-gain", type=float, default=None, help="Post-solve extra rear-load transfer when the body pitches forward.")
    parser.add_argument("--pitch-rebalance-ref", type=float, default=None, help="Pitch angle in radians that saturates the post-solve rear-load transfer.")
    parser.add_argument("--support-centroid-x-gain", type=float, default=None, help="Move the body toward the front/rear center of the current support polygon during swing.")
    parser.add_argument("--support-centroid-y-gain", type=float, default=None, help="Move the body toward the left/right center of the current support polygon during swing.")
    parser.add_argument("--pre-swing-front-shift-scale", type=float, default=None, help="Scale pre-swing support-centroid shift when the upcoming swing leg is a front leg.")
    parser.add_argument("--pre-swing-rear-shift-scale", type=float, default=None, help="Scale pre-swing support-centroid shift when the upcoming swing leg is a rear leg.")
    parser.add_argument("--support-reference-mix", type=float, default=None, help="Blend factor between the equal-support guess and the solved support wrench reference.")
    parser.add_argument("--support-reference-xy-mix", type=float, default=None, help="Optional x/y-only blend factor between the equal-support guess and the solved support wrench reference.")
    parser.add_argument("--vx-gain", type=float, default=None, help="Forward velocity-error gain used in the desired body force heuristic.")
    parser.add_argument("--vy-gain", type=float, default=None, help="Lateral velocity-error gain used in the desired body force heuristic.")
    parser.add_argument("--pre-swing-gate-min-margin", type=float, default=None, help="Delay lift-off until the upcoming support polygon margin exceeds this value in meters.")
    parser.add_argument("--pre-swing-gate-hold-s", type=float, default=None, help="Maximum extra hold time in seconds before a scheduled swing leg is allowed to lift.")
    parser.add_argument("--rear-pre-swing-gate-hold-s", type=float, default=None, help="Optional rear-leg override for the pre-swing gate hold time.")
    parser.add_argument("--rear-pre-swing-guard-roll-threshold", type=float, default=None, help="Optional crawl-only rear-leg posture guard: delay rear swing opening when |roll| exceeds this threshold.")
    parser.add_argument("--rear-pre-swing-guard-pitch-threshold", type=float, default=None, help="Optional crawl-only rear-leg posture guard: delay rear swing opening when |pitch| exceeds this threshold.")
    parser.add_argument("--rear-pre-swing-guard-height-ratio", type=float, default=None, help="Optional crawl-only rear-leg posture guard: delay rear swing opening when base height / ref height falls below this ratio.")
    parser.add_argument("--pre-swing-gate-forward-scale", type=float, default=None, help="Scale forward reference velocity while a scheduled swing leg is being held by the support-margin gate.")
    parser.add_argument("--front-late-release-phase-threshold", type=float, default=None, help="Allow front planned-swing legs to open late once their swing phase exceeds this threshold and support is safe; >1 disables.")
    parser.add_argument("--front-late-release-min-margin", type=float, default=None, help="Required support margin for the front-only late-release path. Defaults to the front pre-swing gate margin.")
    parser.add_argument("--front-late-release-hold-steps", type=int, default=None, help="Legacy controller-step hold after front late release; converted internally using MPC dt.")
    parser.add_argument("--front-late-release-hold-s", type=float, default=None, help="Time in seconds to keep controller-side swing after front late release opens.")
    parser.add_argument("--front-late-release-forward-scale", type=float, default=None, help="Scale forward reference velocity while the front-only late-release path is active.")
    parser.add_argument("--support-margin-preview-s", type=float, default=None, help="Preview CoM xy motion by this many seconds when checking support-margin safety gates.")
    parser.add_argument("--front-late-release-extra-margin", type=float, default=None, help="Extra safety margin added on top of the front late-release minimum support margin.")
    parser.add_argument("--front-late-release-pitch-guard", type=float, default=None, help="Optional abs pitch limit in radians for opening the front-only late-release path.")
    parser.add_argument("--front-late-release-roll-guard", type=float, default=None, help="Optional abs roll limit in radians for opening the front-only late-release path.")
    parser.add_argument("--touchdown-reacquire-hold-s", type=float, default=None, help="Keep controller-side swing briefly after planned touchdown if the foot still has no actual contact.")
    parser.add_argument("--front-touchdown-reacquire-hold-s", type=float, default=None, help="Optional front-leg override for post-swing touchdown reacquire hold.")
    parser.add_argument("--rear-touchdown-reacquire-hold-s", type=float, default=None, help="Optional rear-leg override for post-swing touchdown reacquire hold.")
    parser.add_argument("--touchdown-reacquire-forward-scale", type=float, default=None, help="Scale forward reference velocity while waiting for actual touchdown reacquire.")
    parser.add_argument("--touchdown-reacquire-xy-blend", type=float, default=None, help="Blend touchdown reacquire target xy toward the stored touchdown position.")
    parser.add_argument("--front-touchdown-reacquire-xy-blend", type=float, default=None, help="Optional front-leg override for touchdown reacquire xy blend.")
    parser.add_argument("--rear-touchdown-reacquire-xy-blend", type=float, default=None, help="Optional rear-leg override for touchdown reacquire xy blend.")
    parser.add_argument("--touchdown-reacquire-extra-depth", type=float, default=None, help="Lower the touchdown reacquire target below the nominal touchdown height.")
    parser.add_argument("--front-touchdown-reacquire-extra-depth", type=float, default=None, help="Optional front-leg override for touchdown reacquire extra depth.")
    parser.add_argument("--rear-touchdown-reacquire-extra-depth", type=float, default=None, help="Optional rear-leg override for touchdown reacquire extra depth.")
    parser.add_argument("--touchdown-reacquire-forward-bias", type=float, default=None, help="Add a forward x bias to the touchdown reacquire target.")
    parser.add_argument("--front-touchdown-reacquire-forward-bias", type=float, default=None, help="Optional front-leg override for touchdown reacquire forward bias.")
    parser.add_argument("--rear-touchdown-reacquire-forward-bias", type=float, default=None, help="Optional rear-leg override for touchdown reacquire forward bias.")
    parser.add_argument("--rear-touchdown-reacquire-force-until-contact", action="store_true", help="Keep rear controller-side swing active during planned stance until actual contact really returns.")
    parser.add_argument("--rear-touchdown-reacquire-min-swing-time-s", type=float, default=None, help="Minimum rear swing time [s] before planned-stance reacquire support is allowed to take over.")
    parser.add_argument("--front-touchdown-reacquire-hold-current-xy", action="store_true", help="During front touchdown reacquire, hold the current foot xy and search mainly downward instead of chasing the nominal foothold.")
    parser.add_argument("--rear-touchdown-reacquire-hold-current-xy", action="store_true", help="During rear touchdown reacquire, hold the current foot xy and mainly search downward instead of chasing the nominal foothold.")
    parser.add_argument("--rear-touchdown-reacquire-max-xy-shift", type=float, default=None, help="Cap how far a rear touchdown reacquire target may move from the current foot xy.")
    parser.add_argument("--rear-touchdown-reacquire-min-phase", type=float, default=None, help="Minimum swing phase enforced during rear touchdown reacquire so the foot stays in the descent portion of swing.")
    parser.add_argument("--rear-touchdown-reacquire-upward-vel-damping", type=float, default=None, help="Extra damping applied only to upward rear foot z velocity while waiting for actual recontact.")
    parser.add_argument("--rear-touchdown-retry-descent-depth", type=float, default=None, help="Extra downward search depth used only after a rear false-close retry reopens swing.")
    parser.add_argument("--rear-touchdown-retry-descent-kp", type=float, default=None, help="Vertical task-space stiffness used during rear false-close retry descent assist.")
    parser.add_argument("--rear-touchdown-retry-descent-kd", type=float, default=None, help="Vertical task-space damping used during rear false-close retry descent assist.")
    parser.add_argument("--rear-touchdown-contact-debounce-s", type=float, default=None, help="Require rear actual contact to persist this long before closing controller-side swing during force-until-contact.")
    parser.add_argument("--rear-touchdown-contact-min-phase", type=float, default=None, help="Minimum rear swing phase required before controller-side rear contact may close during touchdown reacquire.")
    parser.add_argument("--rear-touchdown-contact-max-upward-vel", type=float, default=None, help="Maximum allowed upward rear foot z velocity [m/s] before controller-side rear contact may close during touchdown reacquire.")
    parser.add_argument("--rear-touchdown-contact-min-grf-z", type=float, default=None, help="Minimum upward world-frame GRF [N] required before rear touchdown is treated as truly load-bearing.")
    parser.add_argument("--rear-touchdown-close-lock-hold-s", type=float, default=None, help="After a rear late touchdown closes controller-side contact, keep that close sticky for this short window while actual contact persists.")
    parser.add_argument("--rear-touchdown-reacquire-retire-stance-hold-s", type=float, default=None, help="Retire stale rear reacquire state once rear planned/current/actual stance has stayed stable this long [s].")
    parser.add_argument("--rear-crawl-disable-reflex-swing", action="store_true", help="Ignore rear crawl early-stance reflex swing shaping.")
    parser.add_argument("--front-crawl-swing-height-scale", type=float, default=None, help="Scale front crawl swing vertical excursion relative to the nominal trajectory.")
    parser.add_argument("--rear-crawl-swing-height-scale", type=float, default=None, help="Scale rear crawl swing vertical excursion relative to the nominal trajectory.")
    parser.add_argument("--stance-anchor-update-alpha", type=float, default=None, help="Relax stance touchdown anchors toward the actual contacted foot position every step.")
    parser.add_argument("--front-stance-anchor-update-alpha", type=float, default=None, help="Optional front-leg override for stance anchor update alpha.")
    parser.add_argument("--touchdown-support-anchor-update-alpha", type=float, default=None, help="Relax stance touchdown anchors toward the actual contacted foot position only while touchdown support is active.")
    parser.add_argument("--front-touchdown-support-anchor-update-alpha", type=float, default=None, help="Optional front-leg override for touchdown-support anchor update alpha.")
    parser.add_argument("--touchdown-confirm-hold-s", type=float, default=None, help="Keep a short confirmation window after actual touchdown returns.")
    parser.add_argument("--front-touchdown-confirm-hold-s", type=float, default=None, help="Optional front-leg override for touchdown confirmation hold.")
    parser.add_argument("--rear-touchdown-confirm-hold-s", type=float, default=None, help="Optional rear-leg override for touchdown confirmation hold.")
    parser.add_argument("--rear-touchdown-confirm-keep-swing", action="store_true", help="Keep rear controller-side swing active through the rear touchdown confirmation window so a flaky first contact does not reset swing.")
    parser.add_argument("--touchdown-confirm-forward-scale", type=float, default=None, help="Scale forward reference velocity during the touchdown confirmation window.")
    parser.add_argument("--touchdown-settle-hold-s", type=float, default=None, help="Keep a short post-touchdown settling window after actual contact comes back.")
    parser.add_argument("--front-touchdown-settle-hold-s", type=float, default=None, help="Optional front-leg override for the post-touchdown settling window.")
    parser.add_argument("--rear-touchdown-settle-hold-s", type=float, default=None, help="Optional rear-leg override for the post-touchdown settling window.")
    parser.add_argument("--touchdown-settle-forward-scale", type=float, default=None, help="Scale forward reference velocity during the post-touchdown settling window.")
    parser.add_argument("--touchdown-support-rear-floor-delta", type=float, default=None, help="Temporarily increase rear-load floor while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-vertical-boost", type=float, default=None, help="Temporarily increase reduced-support vertical boost while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-min-vertical-force-scale-delta", type=float, default=None, help="Temporarily raise the minimum desired total vertical force while front touchdown support is active.")
    parser.add_argument("--touchdown-support-grf-max-scale-delta", type=float, default=None, help="Temporarily raise GRF headroom while front touchdown support is active.")
    parser.add_argument("--touchdown-support-z-pos-gain-delta", type=float, default=None, help="Temporarily increase base-height gain while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-roll-angle-gain-delta", type=float, default=None, help="Temporarily increase roll-angle gain while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-roll-rate-gain-delta", type=float, default=None, help="Temporarily increase roll-rate gain while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-pitch-angle-gain-delta", type=float, default=None, help="Temporarily increase pitch-angle gain while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-pitch-rate-gain-delta", type=float, default=None, help="Temporarily increase pitch-rate gain while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-side-rebalance-delta", type=float, default=None, help="Temporarily increase signed left/right load rebalance while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-front-joint-pd-scale", type=float, default=None, help="Extra low-gain joint PD applied only on front support legs while front touchdown support is active.")
    parser.add_argument("--touchdown-support-rear-joint-pd-scale", type=float, default=None, help="Extra low-gain joint PD applied only on rear support legs while front touchdown confirm/settle is active.")
    parser.add_argument("--touchdown-support-anchor-xy-blend", type=float, default=None, help="Blend a front touchdown support leg's stance target toward the actual contacted foot xy location.")
    parser.add_argument("--touchdown-support-anchor-z-blend", type=float, default=None, help="Blend a front touchdown support leg's stance target height toward the actual contacted foot z location.")
    parser.add_argument("--rear-touchdown-support-anchor-update-alpha", type=float, default=None, help="Relax rear stance anchors toward the actual foot only while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-support-floor-delta", type=float, default=None, help="Temporarily increase all-stance support floor while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-vertical-boost", type=float, default=None, help="Temporarily increase reduced-support vertical boost while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-min-vertical-force-scale-delta", type=float, default=None, help="Temporarily raise the minimum desired total vertical force while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-grf-max-scale-delta", type=float, default=None, help="Temporarily raise GRF headroom while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-z-pos-gain-delta", type=float, default=None, help="Temporarily increase base-height gain while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-roll-angle-gain-delta", type=float, default=None, help="Temporarily increase roll-angle gain while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-roll-rate-gain-delta", type=float, default=None, help="Temporarily increase roll-rate gain while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-pitch-angle-gain-delta", type=float, default=None, help="Temporarily increase pitch-angle gain while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-pitch-rate-gain-delta", type=float, default=None, help="Temporarily increase pitch-rate gain while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-side-rebalance-delta", type=float, default=None, help="Temporarily increase signed left/right load rebalance while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-front-joint-pd-scale", type=float, default=None, help="Extra low-gain joint PD applied only on front support legs while rear touchdown support is active.")
    parser.add_argument("--rear-touchdown-support-rear-joint-pd-scale", type=float, default=None, help="Extra low-gain joint PD applied only on rear support legs while rear touchdown support is active.")
    parser.add_argument("--rear-post-touchdown-support-hold-s", type=float, default=None, help="Keep a short rear-specific support tail alive after rear touchdown.")
    parser.add_argument("--rear-post-touchdown-support-forward-scale", type=float, default=None, help="Forward-command scale while rear post-touchdown support tail is active.")
    parser.add_argument("--rear-post-touchdown-support-height-ratio", type=float, default=None, help="Enable rear post-touchdown support tail while base height ratio stays below this value.")
    parser.add_argument("--rear-post-touchdown-support-roll-threshold", type=float, default=None, help="Enable rear post-touchdown support tail when |roll| exceeds this threshold.")
    parser.add_argument("--rear-post-touchdown-support-pitch-threshold", type=float, default=None, help="Enable rear post-touchdown support tail when |pitch| exceeds this threshold.")
    parser.add_argument("--rear-post-touchdown-support-min-grf-z", type=float, default=None, help="Keep rear post-touchdown support alive while the touched rear leg carries less than this vertical GRF [N].")
    parser.add_argument("--rear-post-touchdown-support-min-rear-load-share", type=float, default=None, help="Keep rear post-touchdown support alive while total rear vertical load share stays below this fraction.")
    parser.add_argument("--rear-all-contact-stabilization-hold-s", type=float, default=None, help="Rear-only late all-contact stabilization hold after a rear leg closes back into stance.")
    parser.add_argument("--rear-all-contact-stabilization-forward-scale", type=float, default=None, help="Forward-command scale while rear-only late all-contact stabilization is active.")
    parser.add_argument("--rear-all-contact-stabilization-front-alpha-scale", type=float, default=None, help="Clamp front touchdown-support alpha while rear-only late all-contact stabilization is active.")
    parser.add_argument("--rear-all-contact-stabilization-height-ratio", type=float, default=None, help="Enable rear-only late all-contact stabilization while base height ratio stays below this value.")
    parser.add_argument("--rear-all-contact-stabilization-roll-threshold", type=float, default=None, help="Enable rear-only late all-contact stabilization when |roll| exceeds this threshold.")
    parser.add_argument("--rear-all-contact-stabilization-pitch-threshold", type=float, default=None, help="Enable rear-only late all-contact stabilization when |pitch| exceeds this threshold.")
    parser.add_argument("--rear-late-seam-support-trigger-s", type=float, default=None, help="If a rear leg is already planned back in stance but controller-side contact is still open, wait this long before starting the late all-contact support window.")
    parser.add_argument("--rear-close-handoff-hold-s", type=float, default=None, help="Keep a short rear-leg handoff window alive right after a late planned-stance/current-contact close.")
    parser.add_argument("--rear-close-handoff-leg-floor-scale-delta", type=float, default=None, help="Temporarily raise the newly-closed rear leg vertical-load floor during the rear close-handoff window.")
    parser.add_argument("--rear-late-load-share-support-hold-s", type=float, default=None, help="Experimental asymmetric rear support hold [s] for the weaker rear leg during low-height crawl all-contact stance.")
    parser.add_argument("--rear-late-load-share-support-min-leg-share", type=float, default=None, help="Experimental minimum weaker-rear-leg share within the rear pair that triggers the asymmetric late rear support path.")
    parser.add_argument("--rear-late-load-share-support-height-ratio", type=float, default=None, help="Only arm the experimental late rear load-share support if base height is below this ratio of ref_z.")
    parser.add_argument("--rear-late-load-share-support-min-persist-s", type=float, default=None, help="Require the experimental weak-rear-leg condition to persist this long before late asymmetric rear support arms.")
    parser.add_argument("--rear-late-load-share-support-alpha-cap", type=float, default=None, help="Cap the experimental late asymmetric rear support alpha after persistence gating.")
    parser.add_argument("--rear-late-load-share-support-leg-floor-scale-delta", type=float, default=None, help="Temporarily raise only the weaker rear leg vertical-load floor during the late asymmetric rear support window.")
    parser.add_argument("--rear-all-contact-stabilization-min-rear-load-share", type=float, default=None, help="Keep rear-only late all-contact stabilization alive while total rear vertical load share stays below this fraction.")
    parser.add_argument("--rear-all-contact-stabilization-min-rear-leg-load-share", type=float, default=None, help="Keep rear-only late all-contact stabilization alive while the active rear leg load share stays below this fraction.")
    parser.add_argument("--rear-all-contact-stabilization-weak-leg-share-ref", type=float, default=None, help="During rear late all-contact stabilization, treat the weaker rear leg as under-supported below this rear-pair load-share fraction.")
    parser.add_argument("--rear-all-contact-stabilization-weak-leg-floor-delta", type=float, default=None, help="During rear late all-contact stabilization, temporarily raise only the weaker rear leg vertical-load floor by this scale.")
    parser.add_argument("--rear-all-contact-stabilization-weak-leg-height-ratio", type=float, default=None, help="Optional extra height-ratio gate for the weaker rear leg floor boost during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-weak-leg-tail-only", action="store_true", help="Only allow the weaker rear leg floor boost during the posture-only rear all-contact tail.")
    parser.add_argument("--crawl-front-planted-weak-rear-share-ref", type=float, default=None, help="During the narrow crawl front-planted seam, treat the weaker rear leg as under-supported below this rear-pair load-share fraction.")
    parser.add_argument("--crawl-front-planted-weak-rear-alpha-cap", type=float, default=None, help="Cap the dedicated front-planted weak-rear support alpha.")
    parser.add_argument("--rear-all-contact-stabilization-retrigger-limit", type=int, default=None, help="Allow this many late all-contact stabilization renewals after the initial rear touchdown trigger.")
    parser.add_argument("--rear-all-contact-stabilization-rear-floor-delta", type=float, default=None, help="Temporarily increase rear-load floor only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-z-pos-gain-delta", type=float, default=None, help="Temporarily increase base-height gain only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-roll-angle-gain-delta", type=float, default=None, help="Temporarily increase roll-angle gain only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-roll-rate-gain-delta", type=float, default=None, help="Temporarily increase roll-rate gain only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-side-rebalance-delta", type=float, default=None, help="Temporarily increase left/right load rebalance only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-front-anchor-z-blend", type=float, default=None, help="Blend front stance-foot target z toward the actual contacted foot height only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-rear-anchor-z-blend", type=float, default=None, help="Optional rear-leg counterpart for late all-contact stance z blending.")
    parser.add_argument("--rear-all-contact-stabilization-front-anchor-z-max-delta", type=float, default=None, help="Cap front stance-foot target z at actual z plus this margin only during rear late all-contact stabilization.")
    parser.add_argument("--rear-all-contact-stabilization-rear-anchor-z-max-delta", type=float, default=None, help="Optional rear-leg counterpart for late all-contact stance z capping.")
    parser.add_argument("--rear-all-contact-post-recovery-tail-hold-s", type=float, default=None, help="After late full-contact recovery ends, keep rear all-contact stabilization alive for this short tail if posture is still poor.")
    parser.add_argument("--rear-all-contact-release-tail-alpha-scale", type=float, default=None, help="Scale the short posture-only tail applied when rear all-contact stabilization drops out while posture is still poor.")
    parser.add_argument("--rear-all-contact-post-recovery-front-late-alpha-scale", type=float, default=None, help="When the post-recovery posture tail is triggered only by the late front seam, scale its rear all-contact alpha by this factor.")
    parser.add_argument("--front-rear-transition-guard-hold-s", type=float, default=None, help="Keep a brief front pre-swing guard alive after a rear transition seam starts to settle.")
    parser.add_argument("--front-rear-transition-guard-forward-scale", type=float, default=None, help="Scale forward reference velocity while the front rear-transition guard is active.")
    parser.add_argument("--front-rear-transition-guard-roll-threshold", type=float, default=None, help="Absolute roll threshold in radians that can trigger the front rear-transition guard.")
    parser.add_argument("--front-rear-transition-guard-pitch-threshold", type=float, default=None, help="Absolute pitch threshold in radians that can trigger the front rear-transition guard.")
    parser.add_argument("--front-rear-transition-guard-height-ratio", type=float, default=None, help="Trigger the front rear-transition guard if base height falls below this ratio of ref_z.")
    parser.add_argument("--front-rear-transition-guard-release-tail-s", type=float, default=None, help="Once the rear seam is protected again, cap the remaining front rear-transition guard hold to this short tail [s].")
    parser.add_argument("--front-rear-transition-guard-margin-release", type=float, default=None, help="Allow the front rear-transition guard to collapse early only after the front support margin itself has recovered above this threshold.")
    parser.add_argument("--touchdown-contact-vel-z-damping", type=float, default=None, help="Task-space vertical damping applied during touchdown support windows.")
    parser.add_argument("--front-touchdown-contact-vel-z-damping", type=float, default=None, help="Optional front-leg override for touchdown support vertical damping.")
    parser.add_argument("--rear-touchdown-contact-vel-z-damping", type=float, default=None, help="Optional rear-leg override for touchdown support vertical damping.")
    parser.add_argument("--front-margin-rescue-hold-s", type=float, default=None, help="Keep a brief late front-leg support rescue alive during unstable full-contact stance.")
    parser.add_argument("--front-margin-rescue-forward-scale", type=float, default=None, help="Scale forward reference velocity while late front-margin rescue is active.")
    parser.add_argument("--front-margin-rescue-min-margin", type=float, default=None, help="Trigger late front-margin rescue when the queried front support margin falls below this value.")
    parser.add_argument("--front-margin-rescue-margin-gap", type=float, default=None, help="Require the rescued front leg margin to be this much worse than the opposite front leg.")
    parser.add_argument("--front-margin-rescue-alpha-margin", type=float, default=None, help="Support alpha reaches 1 when the rescued front margin falls this far below the trigger threshold.")
    parser.add_argument("--front-margin-rescue-roll-threshold", type=float, default=None, help="Absolute roll threshold in radians required before late front-margin rescue may trigger.")
    parser.add_argument("--front-margin-rescue-pitch-threshold", type=float, default=None, help="Absolute pitch threshold in radians required before late front-margin rescue may trigger.")
    parser.add_argument("--front-margin-rescue-height-ratio", type=float, default=None, help="Trigger late front-margin rescue when base height falls below this ratio of ref_z.")
    parser.add_argument("--front-margin-rescue-recent-swing-window-s", type=float, default=None, help="Require that the same front leg swung within this recent window before late front-margin rescue may trigger.")
    parser.add_argument("--rear-handoff-support-hold-s", type=float, default=None, help="Keep front touchdown-style support alive briefly when a rear swing is about to start.")
    parser.add_argument("--rear-handoff-forward-scale", type=float, default=None, help="Scale forward reference velocity while the rear-handoff support extension is active.")
    parser.add_argument("--rear-handoff-lookahead-steps", type=int, default=None, help="How many horizon steps ahead to inspect for an upcoming rear swing before extending front touchdown support.")
    parser.add_argument("--rear-handoff-support-rear-alpha-scale", type=float, default=None, help="Blend in this much rear touchdown-support alpha during rear handoff support once only one rear stance leg remains.")
    parser.add_argument("--rear-swing-bridge-hold-s", type=float, default=None, help="Keep front touchdown-style support alive briefly into the late rear swing transition.")
    parser.add_argument("--rear-swing-bridge-forward-scale", type=float, default=None, help="Scale forward reference velocity while the rear-swing bridge is active.")
    parser.add_argument("--rear-swing-bridge-roll-threshold", type=float, default=None, help="Absolute roll threshold in radians that can trigger the rear-swing bridge.")
    parser.add_argument("--rear-swing-bridge-pitch-threshold", type=float, default=None, help="Absolute pitch threshold in radians that can trigger the rear-swing bridge.")
    parser.add_argument("--rear-swing-bridge-height-ratio", type=float, default=None, help="Trigger the rear-swing bridge if base height falls below this ratio of ref_z.")
    parser.add_argument("--rear-swing-bridge-recent-front-window-s", type=float, default=None, help="Require that front touchdown support happened within this recent time window before rear-swing bridging.")
    parser.add_argument("--rear-swing-bridge-lookahead-steps", type=int, default=None, help="How many horizon steps ahead to inspect for an upcoming rear swing when bridging support.")
    parser.add_argument("--rear-swing-bridge-allcontact-release-tail-s", type=float, default=None, help="Once the rear seam has fully closed again, cap the remaining rear-swing bridge hold to this short tail.")
    parser.add_argument("--rear-swing-bridge-rear-alpha-scale", type=float, default=None, help="Blend in this much rear touchdown-support alpha during the rear-swing bridge once only one rear stance leg remains.")
    parser.add_argument("--full-contact-recovery-hold-s", type=float, default=None, help="Keep touchdown-style support overrides active briefly during unstable four-foot contact.")
    parser.add_argument("--full-contact-recovery-forward-scale", type=float, default=None, help="Scale forward reference velocity while late full-contact recovery hold is active.")
    parser.add_argument("--full-contact-recovery-roll-threshold", type=float, default=None, help="Absolute roll threshold in radians that can trigger late full-contact recovery.")
    parser.add_argument("--full-contact-recovery-pitch-threshold", type=float, default=None, help="Absolute pitch threshold in radians that can trigger late full-contact recovery.")
    parser.add_argument("--full-contact-recovery-height-ratio", type=float, default=None, help="Trigger late full-contact recovery if base height falls below this ratio of ref_z.")
    parser.add_argument("--full-contact-recovery-recent-window-s", type=float, default=None, help="Require that front touchdown support was active within this recent time window before late full-contact recovery may trigger.")
    parser.add_argument("--full-contact-recovery-rear-support-scale", type=float, default=None, help="Blend rear support alpha into late full-contact recovery after a recent rear touchdown seam.")
    parser.add_argument("--full-contact-recovery-support-floor-delta", type=float, default=None, help="Temporarily increase support-floor ratio while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-z-pos-gain-delta", type=float, default=None, help="Temporarily increase base-height gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-roll-angle-gain-delta", type=float, default=None, help="Temporarily increase roll-angle gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-roll-rate-gain-delta", type=float, default=None, help="Temporarily increase roll-rate gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-pitch-angle-gain-delta", type=float, default=None, help="Temporarily increase pitch-angle gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-pitch-rate-gain-delta", type=float, default=None, help="Temporarily increase pitch-rate gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-side-rebalance-delta", type=float, default=None, help="Temporarily increase signed side-rebalance gain while late full-contact recovery is active.")
    parser.add_argument("--full-contact-recovery-allcontact-release-tail-s", type=float, default=None, help="Once all four feet and rear seam states are recovered again, cap the remaining late full-contact recovery hold to this short tail.")
    parser.add_argument("--crawl-front-delayed-swing-recovery-hold-s", type=float, default=None, help="In crawl, briefly keep late full-contact recovery alive when a front leg is nominally opening swing but is still actually/load-bearing in stance.")
    parser.add_argument("--crawl-front-delayed-swing-recovery-margin-threshold", type=float, default=None, help="Only extend crawl delayed front-swing recovery while the planned-swing front leg still has support margin at or below this threshold.")
    parser.add_argument("--crawl-front-delayed-swing-recovery-once-per-swing", dest="crawl_front_delayed_swing_recovery_once_per_swing", action="store_true", default=None, help="Allow delayed front-swing recovery to extend late full-contact recovery at most once per continuous front planned-swing window.")
    parser.add_argument("--no-crawl-front-delayed-swing-recovery-once-per-swing", dest="crawl_front_delayed_swing_recovery_once_per_swing", action="store_false", help="Allow delayed front-swing recovery to rearm multiple times within the same front planned-swing window.")
    parser.add_argument("--crawl-front-delayed-swing-recovery-release-tail-s", type=float, default=None, help="Once the planned-swing front leg has recovered support margin above the delayed-recovery threshold, cap the remaining late full-contact recovery hold to this short tail.")
    parser.add_argument("--crawl-front-delayed-swing-recovery-rearm-trigger-s", type=float, default=None, help="Only re-arm crawl delayed front-swing recovery when the remaining late full-contact recovery time is at or below this threshold.")
    parser.add_argument("--crawl-front-planted-swing-recovery-hold-s", type=float, default=None, help="In crawl, re-arm a short late full-contact recovery tail when a front leg is nominally in swing but is still physically planted near the end of recovery.")
    parser.add_argument("--crawl-front-planted-swing-recovery-margin-threshold", type=float, default=None, help="Only re-arm the planted-front-swing recovery tail while the planted planned-swing front leg support margin stays at or below this threshold.")
    parser.add_argument("--crawl-front-planted-swing-recovery-height-ratio", type=float, default=None, help="Only re-arm the planted-front-swing recovery tail while base height stays below this ratio of ref_z.")
    parser.add_argument("--crawl-front-planted-swing-recovery-roll-threshold", type=float, default=None, help="Optional abs roll threshold in radians required before the planted-front-swing recovery tail may arm.")
    parser.add_argument("--crawl-front-planted-swing-recovery-rearm-trigger-s", type=float, default=None, help="Only re-arm the planted-front-swing recovery tail once the existing full-contact recovery hold has decayed below this remaining time.")
    parser.add_argument("--crawl-front-planted-postdrop-recovery-hold-s", type=float, default=None, help="In crawl, re-arm a short late full-contact recovery chunk right after recovery drops if the same front planned-swing leg is still planted.")
    parser.add_argument("--crawl-front-planted-seam-support-hold-s", type=float, default=None, help="In crawl, keep a short dedicated support window alive when a front leg is planned swing but still planted during the low-height seam.")
    parser.add_argument("--crawl-front-stance-support-tail-hold-s", type=float, default=None, help="In crawl, keep the remaining front stance leg on touchdown-style support briefly after the opposite front leg actually opens swing.")
    parser.add_argument("--crawl-front-close-gap-support-hold-s", type=float, default=None, help="In crawl, keep the front stance-support tail alive while a front leg has already re-closed planned/current stance but actual contact has not yet returned.")
    parser.add_argument("--crawl-front-close-gap-keep-swing", type=int, choices=[0, 1], default=None, help="In crawl, keep a returning front leg on the swing side while planned/current stance has re-closed but actual contact has not yet returned.")
    parser.add_argument("--crawl-front-late-rearm-tail-hold-s", type=float, default=None, help="In crawl, re-arm a very short front stance-support tail after the original tail has expired but the same front swing is still sagging.")
    parser.add_argument("--crawl-front-late-rearm-budget-s", type=float, default=None, help="Maximum total extra late-rearm support budget that can be consumed within one continuous front planned-swing window.")
    parser.add_argument("--crawl-front-late-rearm-min-swing-time-s", type=float, default=None, help="Minimum front swing elapsed time before the late crawl tail re-arm may fire.")
    parser.add_argument("--crawl-front-late-rearm-min-negative-margin", type=float, default=None, help="Require the still-swinging front leg support margin to stay below -this value before the late crawl tail re-arm may fire.")
    parser.add_argument("--z-pos-gain", type=float, default=None, help="Base-height error gain used in the desired vertical force heuristic.")
    parser.add_argument("--z-vel-gain", type=float, default=None, help="Vertical velocity error gain used in the desired vertical force heuristic.")
    parser.add_argument("--min-vertical-force-scale", type=float, default=None, help="Minimum desired total vertical force as a fraction of body weight.")
    parser.add_argument("--reduced-support-vertical-boost", type=float, default=None, help="Extra body-weight-scaled lift request when fewer than four feet support the body.")
    parser.add_argument("--roll-angle-gain", type=float, default=None, help="Roll-angle feedback gain used in the desired torque heuristic.")
    parser.add_argument("--roll-rate-gain", type=float, default=None, help="Roll-rate feedback gain used in the desired torque heuristic.")
    parser.add_argument("--pitch-angle-gain", type=float, default=None, help="Pitch-angle feedback gain used in the desired torque heuristic.")
    parser.add_argument("--pitch-rate-gain", type=float, default=None, help="Pitch-rate feedback gain used in the desired torque heuristic.")
    parser.add_argument("--yaw-angle-gain", type=float, default=None, help="Yaw-angle feedback gain used in the desired torque heuristic.")
    parser.add_argument("--yaw-rate-gain", type=float, default=None, help="Yaw-rate feedback gain used in the desired torque heuristic.")
    parser.add_argument("--foothold-yaw-rate-scale", type=float, default=None, help="Raibert-style yaw-rate foothold compensation scale used by the linear path.")
    parser.add_argument("--foothold-yaw-error-scale", type=float, default=None, help="Additional yaw-rate tracking-error foothold compensation scale used by the linear path.")
    parser.add_argument("--roll-ref-offset", type=float, default=None, help="Constant roll reference bias [rad] added to the nominal WBInterface orientation reference.")
    parser.add_argument("--pitch-ref-offset", type=float, default=None, help="Constant pitch reference bias [rad] added to the nominal WBInterface orientation reference.")
    parser.add_argument("--latched-swing-xy-blend", type=float, default=None, help="Blend relatched planned-swing feet toward the swing xy trajectory during the release window.")
    parser.add_argument("--latched-swing-lift-ratio", type=float, default=None, help="Raise relatched swing legs by a fraction of step height while keeping stance support.")
    parser.add_argument("--latched-swing-tau-blend", type=float, default=None, help="Blend relatched planned-swing legs toward swing-space torque during the release window.")
    parser.add_argument("--contact-latch-steps", type=int, default=None, help="Base number of contact-sequence steps a planned-swing leg may stay relatched.")
    parser.add_argument("--contact-latch-budget-steps", type=int, default=None, help="Legacy controller-step budget for relatched support; converted internally using MPC dt.")
    parser.add_argument("--contact-latch-budget-s", type=float, default=None, help="Time budget in seconds for how long a scheduled swing leg may remain relatched in contact.")
    parser.add_argument("--front-contact-latch-steps", type=int, default=None, help="Optional front-leg override for the relatched planned-swing horizon.")
    parser.add_argument("--front-contact-latch-budget-s", type=float, default=None, help="Optional front-leg override for the relatched planned-swing budget in seconds.")
    parser.add_argument("--rear-contact-latch-steps", type=int, default=None, help="Optional rear-leg override for the relatched planned-swing horizon.")
    parser.add_argument("--rear-contact-latch-budget-s", type=float, default=None, help="Optional rear-leg override for the relatched planned-swing budget in seconds.")
    parser.add_argument("--startup-full-stance-steps", type=int, default=None, help="Legacy controller-step warmup before the gait starts; converted internally using MPC dt.")
    parser.add_argument("--startup-full-stance-time-s", type=float, default=None, help="Warmup duration in seconds with all four feet in stance before the gait starts.")
    parser.add_argument("--virtual-unlatch-phase-threshold", type=float, default=None, help="Force controller-side swing once a relatched leg passes this swing-phase threshold; >1 disables.")
    parser.add_argument("--virtual-unlatch-hold-steps", type=int, default=None, help="Legacy controller-step hold after virtual unlatch; converted internally using MPC dt.")
    parser.add_argument("--virtual-unlatch-hold-s", type=float, default=None, help="Time in seconds to keep controller-side swing after virtual unlatch triggers.")
    parser.add_argument("--pre-swing-lookahead-steps", type=int, default=None, help="How many MPC stages ahead to inspect upcoming support reduction when applying pre-swing centroid shift.")
    parser.add_argument("--gait-step-freq", type=float, default=None, help="Override selected gait step frequency.")
    parser.add_argument("--gait-duty-factor", type=float, default=None, help="Override selected gait duty factor.")
    parser.add_argument("--front-pre-swing-gate-min-margin", type=float, default=None, help="Override required support margin for front-leg lift-off.")
    parser.add_argument("--rear-pre-swing-gate-min-margin", type=float, default=None, help="Override required support margin for rear-leg lift-off.")
    parser.add_argument("--support-contact-confirm-hold-s", type=float, default=None, help="Require support contacts to stay ready for this long before allowing a scheduled swing.")
    parser.add_argument("--front-support-contact-confirm-hold-s", type=float, default=None, help="Optional front-leg override for support-contact confirmation hold.")
    parser.add_argument("--rear-support-contact-confirm-hold-s", type=float, default=None, help="Optional rear-leg override for support-contact confirmation hold.")
    parser.add_argument("--support-confirm-forward-scale", type=float, default=None, help="Scale forward reference velocity while support-contact confirmation is still pending.")
    parser.add_argument("--front-stance-dropout-reacquire", action="store_true", help="Reopen a short touchdown confirm/settle window on front legs if actual contact returns after a brief planned-stance dropout.")
    parser.add_argument("--rear-stance-dropout-reacquire", action="store_true", help="Reopen a short touchdown confirm/settle window on rear legs if actual contact returns after a brief planned-stance dropout.")
    parser.add_argument("--front-swing-contact-release-timeout-s", type=float, default=None, help="Force a front planned-swing leg open if physical contact persists longer than this timeout.")
    parser.add_argument("--rear-swing-contact-release-timeout-s", type=float, default=None, help="Force a rear planned-swing leg open if physical contact persists longer than this timeout.")
    parser.add_argument("--front-release-lift-height", type=float, default=None, help="Extra upward target [m] applied when a front planned-swing leg remains physically stuck in contact.")
    parser.add_argument("--front-release-lift-kp", type=float, default=None, help="Vertical task-space stiffness used by the front forced-release lift assist.")
    parser.add_argument("--front-release-lift-kd", type=float, default=None, help="Vertical task-space damping used by the front forced-release lift assist.")
    parser.add_argument("--rear-release-lift-height", type=float, default=None, help="Extra upward target [m] applied when a rear planned-swing leg remains physically stuck in contact.")
    parser.add_argument("--rear-release-lift-kp", type=float, default=None, help="Vertical task-space stiffness used by the rear forced-release lift assist.")
    parser.add_argument("--rear-release-lift-kd", type=float, default=None, help="Vertical task-space damping used by the rear forced-release lift assist.")
    parser.add_argument("--rear-swing-release-support-hold-s", type=float, default=None, help="Optional short support window after rear forced release; keep at zero unless explicitly testing it.")
    parser.add_argument("--rear-swing-release-forward-scale", type=float, default=None, help="Forward-reference scale used only while rear forced-release support is active.")
    parser.add_argument("--ground-friction", type=float, default=None, help="Optional fixed ground friction coefficient for MuJoCo.")
    parser.add_argument("--contact-condim", type=int, default=None, help="Optional MuJoCo contact condim override for floor and foot geoms.")
    parser.add_argument("--contact-impratio", type=float, default=None, help="Optional MuJoCo impratio override.")
    parser.add_argument("--contact-torsional-friction", type=float, default=None, help="Optional torsional friction override for floor and foot geoms.")
    parser.add_argument("--contact-rolling-friction", type=float, default=None, help="Optional rolling friction override for floor and foot geoms.")
    args = parser.parse_args()
    disturbance_schedule = _parse_disturbance_pulses(args.disturbance_pulse)
    selected_dynamic_trot_profile = _resolve_dynamic_trot_profile(
        gait=args.gait,
        requested_profile=args.dynamic_trot_profile,
        yaw_rate=args.yaw_rate,
        lateral_speed=args.lateral_speed,
        disturbance_schedule=disturbance_schedule,
    )

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
                # Crawl still needs extra rear-only transition help, but global
                # rescue-style overrides were masking the real problem. Keep
                # the preset focused on: (1) opening rear swing earlier, and
                # (2) holding rear controller-side swing until actual contact
                # really returns. The late single-rear-swing seam is mainly
                # carried by the front touchdown-support path (rear handoff /
                # rear bridge route through front_support_alpha), so those
                # overrides also need to be non-zero in crawl.
                conservative_params.update(
                    {
                        "joint_pd_scale": 0.50,
                        "stance_joint_pd_scale": 0.25,
                        "rear_pre_swing_gate_hold_s": 0.03,
                        "front_contact_latch_steps": 6,
                        "front_contact_latch_budget_s": 0.06,
                        "rear_contact_latch_steps": 6,
                        "rear_contact_latch_budget_s": 0.06,
                        # Keep the front forced-release structure available for
                        # targeted experiments, but leave it disabled by
                        # default. Aggressive front forced release shortened
                        # the best-known crawl run substantially and hid the
                        # genuine late front recontact seam.
                        "front_swing_contact_release_timeout_s": 0.0,
                        # The separate late front stuck-release hook remains
                        # available for experiments, but the default stays
                        # disabled because broad front force-release caused an
                        # early FR_hip collapse instead of helping the final
                        # low-height seam.
                        "crawl_front_stuck_swing_release_timeout_s": 0.0,
                        "crawl_front_stuck_swing_release_height_ratio": 0.0,
                        "crawl_front_stuck_swing_release_roll_threshold": None,
                        "crawl_front_stuck_swing_release_pitch_threshold": None,
                        # The most reliable crawl improvement so far has come
                        # from opening the rear controller-side swing earlier
                        # when the foot remains latched in contact. Values
                        # below 0.10 shortened the run again, while 0.11+ fell
                        # back toward the old front/rear seam failures.
                        "rear_swing_contact_release_timeout_s": 0.10,
                        "front_release_lift_height": 0.0,
                        "front_release_lift_kp": 0.0,
                        "front_release_lift_kd": 0.0,
                        "rear_release_lift_height": 0.012,
                        "rear_release_lift_kp": 260.0,
                        "rear_release_lift_kd": 18.0,
                        "rear_touchdown_reacquire_hold_s": 0.24,
                        "rear_touchdown_reacquire_extra_depth": 0.022,
                        "rear_touchdown_reacquire_force_until_contact": True,
                        "rear_touchdown_reacquire_min_swing_time_s": 0.14,
                        "rear_touchdown_reacquire_hold_current_xy": True,
                        "rear_touchdown_reacquire_max_xy_shift": 0.015,
                        "rear_touchdown_reacquire_min_phase": 0.55,
                        # Rear crawl failures still come from a weak first
                        # touchdown being accepted too early. Keep the
                        # controller-side rear swing open until the returning
                        # contact is a little more clearly load-bearing.
                        "rear_touchdown_contact_debounce_s": 0.015,
                        "rear_touchdown_contact_min_phase": 0.60,
                        "rear_touchdown_contact_max_upward_vel": 0.03,
                        "rear_touchdown_contact_min_grf_z": 8.0,
                        "rear_touchdown_reacquire_retire_stance_hold_s": 0.20,
                        "front_stance_dropout_reacquire": True,
                        "front_stance_dropout_support_hold_s": 0.12,
                        "front_stance_dropout_support_forward_scale": 0.20,
                        "rear_stance_dropout_reacquire": True,
                        "rear_pre_swing_guard_roll_threshold": 0.28,
                        "rear_pre_swing_guard_pitch_threshold": 0.14,
                        "rear_pre_swing_guard_height_ratio": 0.46,
                        "pre_swing_gate_hold_s": 0.04,
                        "rear_crawl_disable_reflex_swing": True,
                        "front_crawl_swing_height_scale": 1.0,
                        "rear_crawl_swing_height_scale": 0.25,
                        "support_force_floor_ratio": 0.10,
                        "touchdown_support_rear_floor_delta": 0.55,
                        "touchdown_support_vertical_boost": 0.22,
                        "touchdown_support_min_vertical_force_scale_delta": 0.0,
                        "touchdown_support_grf_max_scale_delta": 0.0,
                        "touchdown_support_z_pos_gain_delta": 6.0,
                        "touchdown_support_roll_angle_gain_delta": 7.0,
                        "touchdown_support_roll_rate_gain_delta": 2.5,
                        "touchdown_support_pitch_angle_gain_delta": 5.0,
                        "touchdown_support_pitch_rate_gain_delta": 2.0,
                        "touchdown_support_front_joint_pd_scale": 0.10,
                        "touchdown_support_rear_joint_pd_scale": 0.30,
                        # Let rear touchdown support persist slightly longer in
                        # crawl so the first real recontact can transfer load
                        # before the bridge/recovery path takes over.
                        "rear_touchdown_confirm_hold_s": 0.10,
                        "rear_touchdown_confirm_keep_swing": True,
                        "rear_touchdown_close_lock_hold_s": 0.0,
                        "rear_touchdown_settle_hold_s": 0.16,
                        "rear_post_touchdown_support_hold_s": 0.10,
                        "rear_post_touchdown_support_forward_scale": 1.0,
                        "rear_post_touchdown_support_height_ratio": 0.0,
                        "rear_post_touchdown_support_roll_threshold": 0.50,
                        "rear_post_touchdown_support_pitch_threshold": 0.16,
                        "rear_post_touchdown_support_min_grf_z": 0.0,
                        "rear_post_touchdown_support_min_rear_load_share": 0.28,
                        "rear_all_contact_stabilization_hold_s": 0.10,
                        "rear_all_contact_stabilization_forward_scale": 0.05,
                        "rear_all_contact_stabilization_front_alpha_scale": 0.5,
                        "rear_all_contact_stabilization_height_ratio": 0.58,
                        "rear_all_contact_stabilization_roll_threshold": 0.26,
                        "rear_all_contact_stabilization_pitch_threshold": 0.10,
                        "rear_all_contact_stabilization_preclose_pitch_threshold": 0.04,
                        "rear_all_contact_stabilization_preclose_vz_threshold": -0.08,
                        "rear_late_seam_support_trigger_s": 0.0,
                        # The current crawl codepath benefits from a slightly
                        # longer targeted rear close-handoff window than the
                        # earlier split-path sweep. Around 0.22 s is the best
                        # plateau so far once the newer late front seam logic is
                        # included.
                        "rear_close_handoff_hold_s": 0.22,
                        "rear_close_handoff_leg_floor_scale_delta": 0.20,
                        # In the remaining crawl failure, one rear leg keeps
                        # under-sharing load during the final low-height
                        # all-contact seam. A small dedicated weak-leg floor
                        # boost works better than renewing the broader
                        # close-handoff path; 0.10 was the local optimum over
                        # 0.09 / 0.10 / 0.11 and the earlier 0.08 / 0.12 / 0.16
                        # sweep.
                        "rear_late_load_share_support_hold_s": 0.20,
                        "rear_late_load_share_support_min_leg_share": 0.44,
                        "rear_late_load_share_support_height_ratio": 0.58,
                        "rear_late_load_share_support_min_persist_s": 0.04,
                        "rear_late_load_share_support_alpha_cap": 0.75,
                        "rear_late_load_share_support_leg_floor_scale_delta": 0.10,
                        "rear_all_contact_stabilization_min_rear_load_share": 0.18,
                        "rear_all_contact_stabilization_min_rear_leg_load_share": 0.0,
                        # The weak-leg sub-path within all-contact stabilization is
                        # disabled (share_ref=0.0 means condition never fires).
                        # Testing share_ref=0.40 caused severe regression (7.3s vs
                        # 13.54s baseline) because the threshold fires too broadly
                        # during normal alternating crawl stance, causing
                        # overcorrection and FR_hip invalid contact via side roll.
                        "rear_all_contact_stabilization_weak_leg_share_ref": 0.0,
                        "rear_all_contact_stabilization_weak_leg_floor_delta": 0.10,
                        "rear_all_contact_stabilization_weak_leg_height_ratio": 0.0,
                        "rear_all_contact_stabilization_weak_leg_tail_only": False,
                        "rear_all_contact_front_planted_support_floor_delta": 0.05,
                        "rear_all_contact_front_planted_rear_floor_delta": 0.20,
                        "rear_all_contact_front_planted_z_pos_gain_delta": 4.0,
                        "rear_all_contact_front_planted_roll_angle_gain_delta": 0.0,
                        "rear_all_contact_front_planted_roll_rate_gain_delta": 0.0,
                        "rear_all_contact_front_planted_side_rebalance_delta": 0.0,
                        "crawl_front_planted_weak_rear_share_ref": 0.42,
                        "crawl_front_planted_weak_rear_alpha_cap": 0.60,
                        "rear_all_contact_stabilization_retrigger_limit": 0,
                        "rear_all_contact_stabilization_rear_floor_delta": 0.55,
                        "rear_all_contact_stabilization_z_pos_gain_delta": 0.0,
                        "rear_all_contact_stabilization_roll_angle_gain_delta": 0.0,
                        "rear_all_contact_stabilization_roll_rate_gain_delta": 0.0,
                        "rear_all_contact_stabilization_side_rebalance_delta": 0.0,
                        "rear_all_contact_stabilization_front_anchor_z_blend": 0.0,
                        "rear_all_contact_stabilization_rear_anchor_z_blend": 0.0,
                        "rear_all_contact_stabilization_front_anchor_z_max_delta": 0.012,
                        "rear_all_contact_stabilization_rear_anchor_z_max_delta": 0.0,
                        "front_rear_transition_guard_hold_s": 0.26,
                        "front_rear_transition_guard_forward_scale": 0.20,
                        "front_rear_transition_guard_roll_threshold": 0.45,
                        "front_rear_transition_guard_pitch_threshold": 0.16,
                        "front_rear_transition_guard_height_ratio": 0.48,
                        "front_rear_transition_guard_release_tail_s": 0.04,
                        "front_rear_transition_guard_margin_release": 0.01,
                        "front_rear_transition_guard_post_recovery_hold_s": 0.0,
                        "rear_touchdown_support_support_floor_delta": 0.08,
                        "rear_touchdown_support_vertical_boost": 0.16,
                        "rear_touchdown_support_min_vertical_force_scale_delta": 0.0,
                        "rear_touchdown_support_grf_max_scale_delta": 0.0,
                        "rear_touchdown_support_z_pos_gain_delta": 4.0,
                        "rear_touchdown_support_roll_angle_gain_delta": 4.0,
                        "rear_touchdown_support_roll_rate_gain_delta": 1.5,
                        "rear_touchdown_support_pitch_angle_gain_delta": 6.0,
                        "rear_touchdown_support_pitch_rate_gain_delta": 2.0,
                        "rear_touchdown_support_side_rebalance_delta": 0.0,
                        "rear_touchdown_support_front_joint_pd_scale": 0.25,
                        "rear_touchdown_support_rear_joint_pd_scale": 0.10,
                        "rear_touchdown_contact_vel_z_damping": 20.0,
                        "reduced_support_vertical_boost": 0.30,
                        "rear_handoff_support_hold_s": 0.22,
                        "rear_handoff_forward_scale": 0.40,
                        "rear_handoff_lookahead_steps": 2,
                        "rear_handoff_support_rear_alpha_scale": 0.50,
                        "rear_swing_bridge_hold_s": 0.34,
                        "rear_swing_bridge_forward_scale": 0.40,
                        "rear_swing_bridge_roll_threshold": 0.12,
                        "rear_swing_bridge_pitch_threshold": 0.10,
                        "rear_swing_bridge_height_ratio": 0.84,
                        "rear_swing_bridge_rear_alpha_scale": 0.40,
                        "rear_swing_bridge_allcontact_release_tail_s": 0.0,
                        "rear_all_contact_post_recovery_tail_hold_s": 0.08,
                        "rear_all_contact_release_tail_alpha_scale": 0.0,
                        "full_contact_recovery_hold_s": 0.45,
                        "full_contact_recovery_forward_scale": 0.05,
                        "full_contact_recovery_roll_threshold": 0.12,
                        "full_contact_recovery_pitch_threshold": 0.06,
                        "full_contact_recovery_height_ratio": 0.88,
                        "full_contact_recovery_recent_window_s": 0.50,
                        "full_contact_recovery_rear_support_scale": 1.0,
                        "full_contact_recovery_allcontact_release_tail_s": 0.0,
                        "crawl_front_delayed_swing_recovery_hold_s": 0.14,
                        "crawl_front_delayed_swing_recovery_margin_threshold": 0.005,
                        "crawl_front_delayed_swing_recovery_once_per_swing": True,
                        "crawl_front_delayed_swing_recovery_release_tail_s": 0.0,
                        "crawl_front_delayed_swing_recovery_rearm_trigger_s": 0.0,
                        # When the existing full-contact recovery tail is
                        # about to expire but the front planned-swing leg is
                        # still physically planted, re-arm one short late
                        # recovery chunk instead of forcing release.
                        "crawl_front_planted_swing_recovery_hold_s": 0.06,
                        "crawl_front_planted_swing_recovery_margin_threshold": -0.010,
                        "crawl_front_planted_swing_recovery_height_ratio": 0.48,
                        "crawl_front_planted_swing_recovery_roll_threshold": 0.16,
                        "crawl_front_planted_swing_recovery_rearm_trigger_s": 0.02,
                        # The remaining best crawl failure happens right after
                        # late full-contact recovery drops while the same front
                        # planned-swing leg is still physically planted. Re-arm
                        # one short recovery chunk on that falling edge instead
                        # of broadening the earlier planted-front seam logic.
                        "crawl_front_planted_postdrop_recovery_hold_s": 0.06,
                        "crawl_front_planted_seam_support_hold_s": 0.0,
                        "crawl_front_planted_seam_keep_swing": False,
                        "crawl_front_stance_support_tail_hold_s": 0.10,
                        "crawl_front_stance_support_tail_forward_scale": 0.10,
                        "crawl_front_close_gap_support_hold_s": 0.03,
                        "crawl_front_late_rearm_tail_hold_s": 0.012,
                        "crawl_front_late_rearm_budget_s": 0.024,
                        "crawl_front_late_rearm_min_swing_time_s": 0.11,
                        "crawl_front_late_rearm_min_negative_margin": 0.02,
                        # Once all four feet are back on the ground, the remaining crawl
                        # failure is usually a low rolled posture that the existing
                        # touchdown gains do not quite lift back out of. Give the
                        # late all-contact recovery window its own mild height/posture
                        # bump instead of only reusing immediate-touchdown deltas.
                        "full_contact_recovery_support_floor_delta": 0.0,
                        "full_contact_recovery_z_pos_gain_delta": 0.0,
                        "full_contact_recovery_roll_angle_gain_delta": 0.0,
                        "full_contact_recovery_roll_rate_gain_delta": 0.0,
                        "full_contact_recovery_pitch_angle_gain_delta": 0.0,
                        "full_contact_recovery_pitch_rate_gain_delta": 0.0,
                        "full_contact_recovery_side_rebalance_delta": 0.0,
                    }
                )
            if args.gait in {"trot", "pace", "bound"}:
                conservative_params.update(_dynamic_gait_profile_for(args.gait, selected_dynamic_trot_profile))
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
        if args.q_p is not None:
            cfg.linear_osqp_params["Q_p"] = args.q_p
        if args.q_v is not None:
            cfg.linear_osqp_params["Q_v"] = args.q_v
        if args.q_theta is not None:
            cfg.linear_osqp_params["Q_theta"] = args.q_theta
        if args.q_theta_roll is not None:
            cfg.linear_osqp_params["Q_theta_roll"] = args.q_theta_roll
        if args.q_theta_pitch is not None:
            cfg.linear_osqp_params["Q_theta_pitch"] = args.q_theta_pitch
        if args.q_w is not None:
            cfg.linear_osqp_params["Q_w"] = args.q_w
        if args.q_w_roll is not None:
            cfg.linear_osqp_params["Q_w_roll"] = args.q_w_roll
        if args.q_w_pitch is not None:
            cfg.linear_osqp_params["Q_w_pitch"] = args.q_w_pitch
        if args.r_u is not None:
            cfg.linear_osqp_params["R_u"] = args.r_u
        if args.cmd_alpha is not None:
            cfg.linear_osqp_params["command_smoothing"] = args.cmd_alpha
        if args.du_xy_max is not None:
            cfg.linear_osqp_params["du_xy_max"] = args.du_xy_max
        if args.du_z_max is not None:
            cfg.linear_osqp_params["du_z_max"] = args.du_z_max
        if args.stance_ramp_steps is not None:
            cfg.linear_osqp_params["stance_ramp_steps"] = args.stance_ramp_steps
        if args.fy_scale is not None:
            cfg.linear_osqp_params["fy_scale"] = args.fy_scale
        if args.dynamic_fy_roll_gain is not None:
            cfg.linear_osqp_params["dynamic_fy_roll_gain"] = max(float(args.dynamic_fy_roll_gain), 0.0)
        if args.dynamic_fy_roll_ref is not None:
            cfg.linear_osqp_params["dynamic_fy_roll_ref"] = max(float(args.dynamic_fy_roll_ref), 1e-6)
        if args.grf_max_scale is not None:
            cfg.linear_osqp_params["grf_max_scale"] = args.grf_max_scale
        if args.support_floor_ratio is not None:
            cfg.linear_osqp_params["support_force_floor_ratio"] = args.support_floor_ratio
        if args.joint_pd_scale is not None:
            cfg.linear_osqp_params["joint_pd_scale"] = args.joint_pd_scale
        if args.stance_joint_pd_scale is not None:
            cfg.linear_osqp_params["stance_joint_pd_scale"] = args.stance_joint_pd_scale
        if args.latched_joint_pd_scale is not None:
            cfg.linear_osqp_params["latched_joint_pd_scale"] = args.latched_joint_pd_scale
        if args.latched_release_phase_start is not None:
            cfg.linear_osqp_params["latched_release_phase_start"] = args.latched_release_phase_start
        if args.latched_release_phase_end is not None:
            cfg.linear_osqp_params["latched_release_phase_end"] = args.latched_release_phase_end
        if args.stance_target_blend is not None:
            cfg.linear_osqp_params["stance_target_blend"] = args.stance_target_blend
        if args.latched_force_scale is not None:
            cfg.linear_osqp_params["latched_force_scale"] = args.latched_force_scale
        if args.latched_floor_scale is not None:
            cfg.linear_osqp_params["latched_floor_scale"] = args.latched_floor_scale
        if args.latched_same_side_receiver_scale is not None:
            cfg.linear_osqp_params["latched_same_side_receiver_scale"] = args.latched_same_side_receiver_scale
        if args.latched_axle_receiver_scale is not None:
            cfg.linear_osqp_params["latched_axle_receiver_scale"] = args.latched_axle_receiver_scale
        if args.latched_diagonal_receiver_scale is not None:
            cfg.linear_osqp_params["latched_diagonal_receiver_scale"] = args.latched_diagonal_receiver_scale
        if args.latched_front_receiver_scale is not None:
            cfg.linear_osqp_params["latched_front_receiver_scale"] = args.latched_front_receiver_scale
        if args.latched_rear_receiver_scale is not None:
            cfg.linear_osqp_params["latched_rear_receiver_scale"] = args.latched_rear_receiver_scale
        if args.rear_all_contact_front_planted_latched_force_scale_target is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_latched_force_scale_target"] = float(
                max(min(args.rear_all_contact_front_planted_latched_force_scale_target, 1.0), 0.0)
            )
        if args.rear_all_contact_front_planted_latched_front_receiver_scale_target is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_latched_front_receiver_scale_target"] = max(
                float(args.rear_all_contact_front_planted_latched_front_receiver_scale_target), 0.0
            )
        if args.rear_all_contact_front_planted_latched_rear_receiver_scale_target is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_latched_rear_receiver_scale_target"] = max(
                float(args.rear_all_contact_front_planted_latched_rear_receiver_scale_target), 0.0
            )
        if args.rear_all_contact_front_planted_support_floor_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_support_floor_delta"] = max(
                float(args.rear_all_contact_front_planted_support_floor_delta), 0.0
            )
        if args.rear_all_contact_front_planted_rear_floor_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_rear_floor_delta"] = max(
                float(args.rear_all_contact_front_planted_rear_floor_delta), 0.0
            )
        if args.rear_all_contact_front_planted_z_pos_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_z_pos_gain_delta"] = max(
                float(args.rear_all_contact_front_planted_z_pos_gain_delta), 0.0
            )
        if args.rear_all_contact_front_planted_roll_angle_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_roll_angle_gain_delta"] = max(
                float(args.rear_all_contact_front_planted_roll_angle_gain_delta), 0.0
            )
        if args.rear_all_contact_front_planted_roll_rate_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_roll_rate_gain_delta"] = max(
                float(args.rear_all_contact_front_planted_roll_rate_gain_delta), 0.0
            )
        if args.rear_all_contact_front_planted_side_rebalance_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_front_planted_side_rebalance_delta"] = max(
                float(args.rear_all_contact_front_planted_side_rebalance_delta), 0.0
            )
        if args.front_latched_pitch_relief_gain is not None:
            cfg.linear_osqp_params["front_latched_pitch_relief_gain"] = args.front_latched_pitch_relief_gain
        if args.front_latched_rear_bias_gain is not None:
            cfg.linear_osqp_params["front_latched_rear_bias_gain"] = args.front_latched_rear_bias_gain
        if args.rear_floor_base_scale is not None:
            cfg.linear_osqp_params["rear_floor_base_scale"] = args.rear_floor_base_scale
        if args.rear_floor_pitch_gain is not None:
            cfg.linear_osqp_params["rear_floor_pitch_gain"] = args.rear_floor_pitch_gain
        if args.side_rebalance_gain is not None:
            cfg.linear_osqp_params["side_rebalance_gain"] = max(float(args.side_rebalance_gain), 0.0)
        if args.side_rebalance_ref is not None:
            cfg.linear_osqp_params["side_rebalance_ref"] = max(float(args.side_rebalance_ref), 1e-6)
        if args.pitch_rebalance_gain is not None:
            cfg.linear_osqp_params["pitch_rebalance_gain"] = args.pitch_rebalance_gain
        if args.pitch_rebalance_ref is not None:
            cfg.linear_osqp_params["pitch_rebalance_ref"] = args.pitch_rebalance_ref
        if args.support_centroid_x_gain is not None:
            cfg.linear_osqp_params["support_centroid_x_gain"] = args.support_centroid_x_gain
        if args.support_centroid_y_gain is not None:
            cfg.linear_osqp_params["support_centroid_y_gain"] = args.support_centroid_y_gain
        if args.pre_swing_front_shift_scale is not None:
            cfg.linear_osqp_params["pre_swing_front_shift_scale"] = args.pre_swing_front_shift_scale
        if args.pre_swing_rear_shift_scale is not None:
            cfg.linear_osqp_params["pre_swing_rear_shift_scale"] = args.pre_swing_rear_shift_scale
        if args.support_reference_mix is not None:
            cfg.linear_osqp_params["support_reference_mix"] = max(min(float(args.support_reference_mix), 1.0), 0.0)
        if args.support_reference_xy_mix is not None:
            cfg.linear_osqp_params["support_reference_xy_mix"] = max(min(float(args.support_reference_xy_mix), 1.0), 0.0)
        if args.pre_swing_gate_min_margin is not None:
            cfg.linear_osqp_params["pre_swing_gate_min_margin"] = max(float(args.pre_swing_gate_min_margin), 0.0)
        if args.front_pre_swing_gate_min_margin is not None:
            cfg.linear_osqp_params["front_pre_swing_gate_min_margin"] = float(args.front_pre_swing_gate_min_margin)
        if args.rear_pre_swing_gate_min_margin is not None:
            cfg.linear_osqp_params["rear_pre_swing_gate_min_margin"] = float(args.rear_pre_swing_gate_min_margin)
        if args.support_contact_confirm_hold_s is not None:
            cfg.linear_osqp_params["support_contact_confirm_hold_s"] = max(float(args.support_contact_confirm_hold_s), 0.0)
        if args.front_support_contact_confirm_hold_s is not None:
            cfg.linear_osqp_params["front_support_contact_confirm_hold_s"] = max(
                float(args.front_support_contact_confirm_hold_s), 0.0
            )
        if args.rear_support_contact_confirm_hold_s is not None:
            cfg.linear_osqp_params["rear_support_contact_confirm_hold_s"] = max(
                float(args.rear_support_contact_confirm_hold_s), 0.0
            )
        if args.support_confirm_forward_scale is not None:
            cfg.linear_osqp_params["support_confirm_forward_scale"] = float(
                max(min(args.support_confirm_forward_scale, 1.0), 0.0)
            )
        if args.front_stance_dropout_reacquire:
            cfg.linear_osqp_params["front_stance_dropout_reacquire"] = True
        if args.rear_stance_dropout_reacquire:
            cfg.linear_osqp_params["rear_stance_dropout_reacquire"] = True
        if args.front_late_release_phase_threshold is not None:
            cfg.linear_osqp_params["front_late_release_phase_threshold"] = float(args.front_late_release_phase_threshold)
        if args.front_late_release_min_margin is not None:
            cfg.linear_osqp_params["front_late_release_min_margin"] = max(float(args.front_late_release_min_margin), 0.0)
        if args.front_late_release_hold_steps is not None:
            cfg.linear_osqp_params["front_late_release_hold_steps"] = max(int(args.front_late_release_hold_steps), 0)
        if args.front_late_release_hold_s is not None:
            cfg.linear_osqp_params["front_late_release_hold_s"] = max(float(args.front_late_release_hold_s), 0.0)
        if args.front_late_release_forward_scale is not None:
            cfg.linear_osqp_params["front_late_release_forward_scale"] = float(
                max(min(args.front_late_release_forward_scale, 1.0), 0.0)
            )
        if args.support_margin_preview_s is not None:
            cfg.linear_osqp_params["support_margin_preview_s"] = max(float(args.support_margin_preview_s), 0.0)
        if args.front_late_release_extra_margin is not None:
            cfg.linear_osqp_params["front_late_release_extra_margin"] = max(float(args.front_late_release_extra_margin), 0.0)
        if args.front_late_release_pitch_guard is not None:
            cfg.linear_osqp_params["front_late_release_pitch_guard"] = max(float(args.front_late_release_pitch_guard), 0.0)
        if args.front_late_release_roll_guard is not None:
            cfg.linear_osqp_params["front_late_release_roll_guard"] = max(float(args.front_late_release_roll_guard), 0.0)
        if args.touchdown_reacquire_hold_s is not None:
            cfg.linear_osqp_params["touchdown_reacquire_hold_s"] = max(float(args.touchdown_reacquire_hold_s), 0.0)
        if args.front_touchdown_reacquire_hold_s is not None:
            cfg.linear_osqp_params["front_touchdown_reacquire_hold_s"] = max(float(args.front_touchdown_reacquire_hold_s), 0.0)
        if args.rear_touchdown_reacquire_hold_s is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_hold_s"] = max(float(args.rear_touchdown_reacquire_hold_s), 0.0)
        if args.touchdown_reacquire_forward_scale is not None:
            cfg.linear_osqp_params["touchdown_reacquire_forward_scale"] = float(
                max(min(args.touchdown_reacquire_forward_scale, 1.0), 0.0)
            )
        if args.touchdown_reacquire_xy_blend is not None:
            cfg.linear_osqp_params["touchdown_reacquire_xy_blend"] = float(
                max(min(args.touchdown_reacquire_xy_blend, 1.0), 0.0)
            )
        if args.front_touchdown_reacquire_xy_blend is not None:
            cfg.linear_osqp_params["front_touchdown_reacquire_xy_blend"] = float(
                max(min(args.front_touchdown_reacquire_xy_blend, 1.0), 0.0)
            )
        if args.rear_touchdown_reacquire_xy_blend is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_xy_blend"] = float(
                max(min(args.rear_touchdown_reacquire_xy_blend, 1.0), 0.0)
            )
        if args.touchdown_reacquire_extra_depth is not None:
            cfg.linear_osqp_params["touchdown_reacquire_extra_depth"] = max(float(args.touchdown_reacquire_extra_depth), 0.0)
        if args.front_touchdown_reacquire_extra_depth is not None:
            cfg.linear_osqp_params["front_touchdown_reacquire_extra_depth"] = max(
                float(args.front_touchdown_reacquire_extra_depth),
                0.0,
            )
        if args.rear_touchdown_reacquire_extra_depth is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_extra_depth"] = max(
                float(args.rear_touchdown_reacquire_extra_depth),
                0.0,
            )
        if args.touchdown_reacquire_forward_bias is not None:
            cfg.linear_osqp_params["touchdown_reacquire_forward_bias"] = float(args.touchdown_reacquire_forward_bias)
        if args.front_touchdown_reacquire_forward_bias is not None:
            cfg.linear_osqp_params["front_touchdown_reacquire_forward_bias"] = float(args.front_touchdown_reacquire_forward_bias)
        if args.rear_touchdown_reacquire_forward_bias is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_forward_bias"] = float(args.rear_touchdown_reacquire_forward_bias)
        if args.rear_touchdown_reacquire_force_until_contact:
            cfg.linear_osqp_params["rear_touchdown_reacquire_force_until_contact"] = True
        if args.rear_touchdown_reacquire_min_swing_time_s is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_min_swing_time_s"] = max(
                float(args.rear_touchdown_reacquire_min_swing_time_s),
                0.0,
            )
        if args.front_touchdown_reacquire_hold_current_xy:
            cfg.linear_osqp_params["front_touchdown_reacquire_hold_current_xy"] = True
        if args.rear_touchdown_reacquire_hold_current_xy:
            cfg.linear_osqp_params["rear_touchdown_reacquire_hold_current_xy"] = True
        if args.rear_touchdown_reacquire_max_xy_shift is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_max_xy_shift"] = max(
                float(args.rear_touchdown_reacquire_max_xy_shift),
                0.0,
            )
        if args.rear_touchdown_reacquire_min_phase is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_min_phase"] = max(
                min(float(args.rear_touchdown_reacquire_min_phase), 1.0),
                0.0,
            )
        if args.rear_touchdown_reacquire_upward_vel_damping is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_upward_vel_damping"] = max(
                float(args.rear_touchdown_reacquire_upward_vel_damping),
                0.0,
            )
        if args.rear_touchdown_retry_descent_depth is not None:
            cfg.linear_osqp_params["rear_touchdown_retry_descent_depth"] = max(
                float(args.rear_touchdown_retry_descent_depth),
                0.0,
            )
        if args.rear_touchdown_retry_descent_kp is not None:
            cfg.linear_osqp_params["rear_touchdown_retry_descent_kp"] = max(
                float(args.rear_touchdown_retry_descent_kp),
                0.0,
            )
        if args.rear_touchdown_retry_descent_kd is not None:
            cfg.linear_osqp_params["rear_touchdown_retry_descent_kd"] = max(
                float(args.rear_touchdown_retry_descent_kd),
                0.0,
            )
        if args.rear_touchdown_contact_debounce_s is not None:
            cfg.linear_osqp_params["rear_touchdown_contact_debounce_s"] = max(
                float(args.rear_touchdown_contact_debounce_s),
                0.0,
            )
        if args.rear_touchdown_contact_min_phase is not None:
            cfg.linear_osqp_params["rear_touchdown_contact_min_phase"] = max(
                min(float(args.rear_touchdown_contact_min_phase), 1.0),
                0.0,
            )
        if args.rear_touchdown_contact_max_upward_vel is not None:
            cfg.linear_osqp_params["rear_touchdown_contact_max_upward_vel"] = float(
                args.rear_touchdown_contact_max_upward_vel
            )
        if args.rear_touchdown_contact_min_grf_z is not None:
            cfg.linear_osqp_params["rear_touchdown_contact_min_grf_z"] = max(
                float(args.rear_touchdown_contact_min_grf_z),
                0.0,
            )
        if args.rear_touchdown_close_lock_hold_s is not None:
            cfg.linear_osqp_params["rear_touchdown_close_lock_hold_s"] = max(
                float(args.rear_touchdown_close_lock_hold_s),
                0.0,
            )
        if args.rear_touchdown_reacquire_retire_stance_hold_s is not None:
            cfg.linear_osqp_params["rear_touchdown_reacquire_retire_stance_hold_s"] = max(
                float(args.rear_touchdown_reacquire_retire_stance_hold_s),
                0.0,
            )
        if args.rear_crawl_disable_reflex_swing:
            cfg.linear_osqp_params["rear_crawl_disable_reflex_swing"] = True
        if args.front_crawl_swing_height_scale is not None:
            cfg.linear_osqp_params["front_crawl_swing_height_scale"] = float(
                max(args.front_crawl_swing_height_scale, 0.0)
            )
        if args.rear_crawl_swing_height_scale is not None:
            cfg.linear_osqp_params["rear_crawl_swing_height_scale"] = float(
                max(args.rear_crawl_swing_height_scale, 0.0)
            )
        if args.stance_anchor_update_alpha is not None:
            cfg.linear_osqp_params["stance_anchor_update_alpha"] = float(
                max(min(args.stance_anchor_update_alpha, 1.0), 0.0)
            )
        if args.front_stance_anchor_update_alpha is not None:
            cfg.linear_osqp_params["front_stance_anchor_update_alpha"] = float(
                max(min(args.front_stance_anchor_update_alpha, 1.0), 0.0)
            )
        if args.touchdown_support_anchor_update_alpha is not None:
            cfg.linear_osqp_params["touchdown_support_anchor_update_alpha"] = float(
                max(min(args.touchdown_support_anchor_update_alpha, 1.0), 0.0)
            )
        if args.front_touchdown_support_anchor_update_alpha is not None:
            cfg.linear_osqp_params["front_touchdown_support_anchor_update_alpha"] = float(
                max(min(args.front_touchdown_support_anchor_update_alpha, 1.0), 0.0)
            )
        if args.touchdown_confirm_hold_s is not None:
            cfg.linear_osqp_params["touchdown_confirm_hold_s"] = max(float(args.touchdown_confirm_hold_s), 0.0)
        if args.front_touchdown_confirm_hold_s is not None:
            cfg.linear_osqp_params["front_touchdown_confirm_hold_s"] = max(
                float(args.front_touchdown_confirm_hold_s),
                0.0,
            )
        if args.rear_touchdown_confirm_hold_s is not None:
            cfg.linear_osqp_params["rear_touchdown_confirm_hold_s"] = max(
                float(args.rear_touchdown_confirm_hold_s),
                0.0,
            )
        if args.rear_touchdown_confirm_keep_swing:
            cfg.linear_osqp_params["rear_touchdown_confirm_keep_swing"] = True
        if args.touchdown_confirm_forward_scale is not None:
            cfg.linear_osqp_params["touchdown_confirm_forward_scale"] = float(
                max(min(args.touchdown_confirm_forward_scale, 1.0), 0.0)
            )
        if args.touchdown_settle_hold_s is not None:
            cfg.linear_osqp_params["touchdown_settle_hold_s"] = max(float(args.touchdown_settle_hold_s), 0.0)
        if args.front_touchdown_settle_hold_s is not None:
            cfg.linear_osqp_params["front_touchdown_settle_hold_s"] = max(float(args.front_touchdown_settle_hold_s), 0.0)
        if args.rear_touchdown_settle_hold_s is not None:
            cfg.linear_osqp_params["rear_touchdown_settle_hold_s"] = max(float(args.rear_touchdown_settle_hold_s), 0.0)
        if args.touchdown_settle_forward_scale is not None:
            cfg.linear_osqp_params["touchdown_settle_forward_scale"] = float(
                max(min(args.touchdown_settle_forward_scale, 1.0), 0.0)
            )
        if args.touchdown_support_rear_floor_delta is not None:
            cfg.linear_osqp_params["touchdown_support_rear_floor_delta"] = max(float(args.touchdown_support_rear_floor_delta), 0.0)
        if args.touchdown_support_vertical_boost is not None:
            cfg.linear_osqp_params["touchdown_support_vertical_boost"] = max(float(args.touchdown_support_vertical_boost), 0.0)
        if args.touchdown_support_min_vertical_force_scale_delta is not None:
            cfg.linear_osqp_params["touchdown_support_min_vertical_force_scale_delta"] = max(
                float(args.touchdown_support_min_vertical_force_scale_delta), 0.0
            )
        if args.touchdown_support_grf_max_scale_delta is not None:
            cfg.linear_osqp_params["touchdown_support_grf_max_scale_delta"] = max(
                float(args.touchdown_support_grf_max_scale_delta), 0.0
            )
        if args.touchdown_support_z_pos_gain_delta is not None:
            cfg.linear_osqp_params["touchdown_support_z_pos_gain_delta"] = max(float(args.touchdown_support_z_pos_gain_delta), 0.0)
        if args.touchdown_support_roll_angle_gain_delta is not None:
            cfg.linear_osqp_params["touchdown_support_roll_angle_gain_delta"] = max(float(args.touchdown_support_roll_angle_gain_delta), 0.0)
        if args.touchdown_support_roll_rate_gain_delta is not None:
            cfg.linear_osqp_params["touchdown_support_roll_rate_gain_delta"] = max(float(args.touchdown_support_roll_rate_gain_delta), 0.0)
        if args.touchdown_support_pitch_angle_gain_delta is not None:
            cfg.linear_osqp_params["touchdown_support_pitch_angle_gain_delta"] = max(float(args.touchdown_support_pitch_angle_gain_delta), 0.0)
        if args.touchdown_support_pitch_rate_gain_delta is not None:
            cfg.linear_osqp_params["touchdown_support_pitch_rate_gain_delta"] = max(float(args.touchdown_support_pitch_rate_gain_delta), 0.0)
        if args.touchdown_support_side_rebalance_delta is not None:
            cfg.linear_osqp_params["touchdown_support_side_rebalance_delta"] = max(float(args.touchdown_support_side_rebalance_delta), 0.0)
        if args.touchdown_support_front_joint_pd_scale is not None:
            cfg.linear_osqp_params["touchdown_support_front_joint_pd_scale"] = max(float(args.touchdown_support_front_joint_pd_scale), 0.0)
        if args.touchdown_support_rear_joint_pd_scale is not None:
            cfg.linear_osqp_params["touchdown_support_rear_joint_pd_scale"] = max(float(args.touchdown_support_rear_joint_pd_scale), 0.0)
        if args.touchdown_support_anchor_xy_blend is not None:
            cfg.linear_osqp_params["touchdown_support_anchor_xy_blend"] = float(
                max(min(args.touchdown_support_anchor_xy_blend, 1.0), 0.0)
            )
        if args.touchdown_support_anchor_z_blend is not None:
            cfg.linear_osqp_params["touchdown_support_anchor_z_blend"] = float(
                max(min(args.touchdown_support_anchor_z_blend, 1.0), 0.0)
            )
        if args.rear_touchdown_support_anchor_update_alpha is not None:
            cfg.linear_osqp_params["rear_touchdown_support_anchor_update_alpha"] = float(
                max(min(args.rear_touchdown_support_anchor_update_alpha, 1.0), 0.0)
            )
        if args.rear_touchdown_support_support_floor_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_support_floor_delta"] = max(
                float(args.rear_touchdown_support_support_floor_delta), 0.0
            )
        if args.rear_touchdown_support_vertical_boost is not None:
            cfg.linear_osqp_params["rear_touchdown_support_vertical_boost"] = max(
                float(args.rear_touchdown_support_vertical_boost), 0.0
            )
        if args.rear_touchdown_support_min_vertical_force_scale_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_min_vertical_force_scale_delta"] = max(
                float(args.rear_touchdown_support_min_vertical_force_scale_delta), 0.0
            )
        if args.rear_touchdown_support_grf_max_scale_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_grf_max_scale_delta"] = max(
                float(args.rear_touchdown_support_grf_max_scale_delta), 0.0
            )
        if args.rear_touchdown_support_z_pos_gain_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_z_pos_gain_delta"] = max(
                float(args.rear_touchdown_support_z_pos_gain_delta), 0.0
            )
        if args.rear_touchdown_support_roll_angle_gain_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_roll_angle_gain_delta"] = max(
                float(args.rear_touchdown_support_roll_angle_gain_delta), 0.0
            )
        if args.rear_touchdown_support_roll_rate_gain_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_roll_rate_gain_delta"] = max(
                float(args.rear_touchdown_support_roll_rate_gain_delta), 0.0
            )
        if args.rear_touchdown_support_pitch_angle_gain_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_pitch_angle_gain_delta"] = max(
                float(args.rear_touchdown_support_pitch_angle_gain_delta), 0.0
            )
        if args.rear_touchdown_support_pitch_rate_gain_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_pitch_rate_gain_delta"] = max(
                float(args.rear_touchdown_support_pitch_rate_gain_delta), 0.0
            )
        if args.rear_touchdown_support_side_rebalance_delta is not None:
            cfg.linear_osqp_params["rear_touchdown_support_side_rebalance_delta"] = max(
                float(args.rear_touchdown_support_side_rebalance_delta), 0.0
            )
        if args.rear_touchdown_support_front_joint_pd_scale is not None:
            cfg.linear_osqp_params["rear_touchdown_support_front_joint_pd_scale"] = max(
                float(args.rear_touchdown_support_front_joint_pd_scale), 0.0
            )
        if args.rear_touchdown_support_rear_joint_pd_scale is not None:
            cfg.linear_osqp_params["rear_touchdown_support_rear_joint_pd_scale"] = max(
                float(args.rear_touchdown_support_rear_joint_pd_scale), 0.0
            )
        if args.rear_post_touchdown_support_hold_s is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_hold_s"] = max(
                float(args.rear_post_touchdown_support_hold_s), 0.0
            )
        if args.rear_post_touchdown_support_forward_scale is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_forward_scale"] = float(
                max(min(args.rear_post_touchdown_support_forward_scale, 1.0), 0.0)
            )
        if args.rear_post_touchdown_support_height_ratio is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_height_ratio"] = max(
                float(args.rear_post_touchdown_support_height_ratio), 0.0
            )
        if args.rear_post_touchdown_support_roll_threshold is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_roll_threshold"] = max(
                float(args.rear_post_touchdown_support_roll_threshold), 0.0
            )
        if args.rear_post_touchdown_support_pitch_threshold is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_pitch_threshold"] = max(
                float(args.rear_post_touchdown_support_pitch_threshold), 0.0
            )
        if args.rear_post_touchdown_support_min_grf_z is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_min_grf_z"] = max(
                float(args.rear_post_touchdown_support_min_grf_z), 0.0
            )
        if args.rear_post_touchdown_support_min_rear_load_share is not None:
            cfg.linear_osqp_params["rear_post_touchdown_support_min_rear_load_share"] = max(
                float(args.rear_post_touchdown_support_min_rear_load_share), 0.0
            )
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
        if args.front_rear_transition_guard_hold_s is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_hold_s"] = max(
                float(args.front_rear_transition_guard_hold_s), 0.0
            )
        if args.front_rear_transition_guard_forward_scale is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_forward_scale"] = float(
                max(min(args.front_rear_transition_guard_forward_scale, 1.0), 0.0)
            )
        if args.front_rear_transition_guard_roll_threshold is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_roll_threshold"] = max(
                float(args.front_rear_transition_guard_roll_threshold), 0.0
            )
        if args.front_rear_transition_guard_pitch_threshold is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_pitch_threshold"] = max(
                float(args.front_rear_transition_guard_pitch_threshold), 0.0
            )
        if args.front_rear_transition_guard_height_ratio is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_height_ratio"] = max(
                float(args.front_rear_transition_guard_height_ratio), 0.0
            )
        if args.front_rear_transition_guard_release_tail_s is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_release_tail_s"] = max(
                float(args.front_rear_transition_guard_release_tail_s), 0.0
            )
        if args.front_rear_transition_guard_margin_release is not None:
            cfg.linear_osqp_params["front_rear_transition_guard_margin_release"] = max(
                float(args.front_rear_transition_guard_margin_release), 0.0
            )
        if args.rear_all_contact_stabilization_hold_s is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_hold_s"] = max(
                float(args.rear_all_contact_stabilization_hold_s), 0.0
            )
        if args.rear_all_contact_stabilization_forward_scale is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_forward_scale"] = float(
                max(min(args.rear_all_contact_stabilization_forward_scale, 1.0), 0.0)
            )
        if args.rear_all_contact_stabilization_front_alpha_scale is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_front_alpha_scale"] = float(
                max(min(args.rear_all_contact_stabilization_front_alpha_scale, 1.0), 0.0)
            )
        if args.rear_all_contact_stabilization_height_ratio is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_height_ratio"] = max(
                float(args.rear_all_contact_stabilization_height_ratio), 0.0
            )
        if args.rear_all_contact_stabilization_roll_threshold is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_roll_threshold"] = max(
                float(args.rear_all_contact_stabilization_roll_threshold), 0.0
            )
        if args.rear_all_contact_stabilization_pitch_threshold is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_pitch_threshold"] = max(
                float(args.rear_all_contact_stabilization_pitch_threshold), 0.0
            )
        if args.rear_late_seam_support_trigger_s is not None:
            cfg.linear_osqp_params["rear_late_seam_support_trigger_s"] = max(
                float(args.rear_late_seam_support_trigger_s), 0.0
            )
        if args.rear_close_handoff_hold_s is not None:
            cfg.linear_osqp_params["rear_close_handoff_hold_s"] = max(
                float(args.rear_close_handoff_hold_s), 0.0
            )
        if args.rear_close_handoff_leg_floor_scale_delta is not None:
            cfg.linear_osqp_params["rear_close_handoff_leg_floor_scale_delta"] = max(
                float(args.rear_close_handoff_leg_floor_scale_delta), 0.0
            )
        if args.rear_late_load_share_support_hold_s is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_hold_s"] = max(
                float(args.rear_late_load_share_support_hold_s),
                0.0,
            )
        if args.rear_late_load_share_support_min_leg_share is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_min_leg_share"] = max(
                float(args.rear_late_load_share_support_min_leg_share),
                0.0,
            )
        if args.rear_late_load_share_support_height_ratio is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_height_ratio"] = max(
                float(args.rear_late_load_share_support_height_ratio),
                0.0,
            )
        if args.rear_late_load_share_support_min_persist_s is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_min_persist_s"] = max(
                float(args.rear_late_load_share_support_min_persist_s),
                0.0,
            )
        if args.rear_late_load_share_support_alpha_cap is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_alpha_cap"] = float(
                np.clip(float(args.rear_late_load_share_support_alpha_cap), 0.0, 1.0)
            )
        if args.rear_late_load_share_support_leg_floor_scale_delta is not None:
            cfg.linear_osqp_params["rear_late_load_share_support_leg_floor_scale_delta"] = max(
                float(args.rear_late_load_share_support_leg_floor_scale_delta),
                0.0,
            )
        if args.rear_all_contact_stabilization_min_rear_load_share is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_min_rear_load_share"] = max(
                float(args.rear_all_contact_stabilization_min_rear_load_share), 0.0
            )
        if args.rear_all_contact_stabilization_min_rear_leg_load_share is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_min_rear_leg_load_share"] = max(
                float(args.rear_all_contact_stabilization_min_rear_leg_load_share), 0.0
            )
        if args.rear_all_contact_stabilization_weak_leg_share_ref is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_weak_leg_share_ref"] = max(
                float(args.rear_all_contact_stabilization_weak_leg_share_ref), 0.0
            )
        if args.rear_all_contact_stabilization_weak_leg_floor_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_weak_leg_floor_delta"] = max(
                float(args.rear_all_contact_stabilization_weak_leg_floor_delta), 0.0
            )
        if args.rear_all_contact_stabilization_weak_leg_height_ratio is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_weak_leg_height_ratio"] = max(
                float(args.rear_all_contact_stabilization_weak_leg_height_ratio), 0.0
            )
        if args.rear_all_contact_stabilization_weak_leg_tail_only:
            cfg.linear_osqp_params["rear_all_contact_stabilization_weak_leg_tail_only"] = True
        if args.crawl_front_planted_weak_rear_share_ref is not None:
            cfg.linear_osqp_params["crawl_front_planted_weak_rear_share_ref"] = max(
                float(args.crawl_front_planted_weak_rear_share_ref), 0.0
            )
        if args.crawl_front_planted_weak_rear_alpha_cap is not None:
            cfg.linear_osqp_params["crawl_front_planted_weak_rear_alpha_cap"] = float(
                np.clip(float(args.crawl_front_planted_weak_rear_alpha_cap), 0.0, 1.0)
            )
        if args.rear_all_contact_stabilization_retrigger_limit is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_retrigger_limit"] = max(
                int(args.rear_all_contact_stabilization_retrigger_limit),
                0,
            )
        if args.rear_all_contact_stabilization_rear_floor_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_rear_floor_delta"] = max(
                float(args.rear_all_contact_stabilization_rear_floor_delta), 0.0
            )
        if args.rear_all_contact_stabilization_z_pos_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_z_pos_gain_delta"] = max(
                float(args.rear_all_contact_stabilization_z_pos_gain_delta),
                0.0,
            )
        if args.rear_all_contact_stabilization_roll_angle_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_roll_angle_gain_delta"] = max(
                float(args.rear_all_contact_stabilization_roll_angle_gain_delta),
                0.0,
            )
        if args.rear_all_contact_stabilization_roll_rate_gain_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_roll_rate_gain_delta"] = max(
                float(args.rear_all_contact_stabilization_roll_rate_gain_delta),
                0.0,
            )
        if args.rear_all_contact_stabilization_side_rebalance_delta is not None:
            cfg.linear_osqp_params["rear_all_contact_stabilization_side_rebalance_delta"] = max(
                float(args.rear_all_contact_stabilization_side_rebalance_delta),
                0.0,
            )
        if args.touchdown_contact_vel_z_damping is not None:
            cfg.linear_osqp_params["touchdown_contact_vel_z_damping"] = max(
                float(args.touchdown_contact_vel_z_damping), 0.0
            )
        if args.front_touchdown_contact_vel_z_damping is not None:
            cfg.linear_osqp_params["front_touchdown_contact_vel_z_damping"] = max(
                float(args.front_touchdown_contact_vel_z_damping), 0.0
            )
        if args.rear_touchdown_contact_vel_z_damping is not None:
            cfg.linear_osqp_params["rear_touchdown_contact_vel_z_damping"] = max(
                float(args.rear_touchdown_contact_vel_z_damping), 0.0
            )
        if args.front_margin_rescue_hold_s is not None:
            cfg.linear_osqp_params["front_margin_rescue_hold_s"] = max(float(args.front_margin_rescue_hold_s), 0.0)
        if args.front_margin_rescue_forward_scale is not None:
            cfg.linear_osqp_params["front_margin_rescue_forward_scale"] = float(
                max(min(args.front_margin_rescue_forward_scale, 1.0), 0.0)
            )
        if args.front_margin_rescue_min_margin is not None:
            cfg.linear_osqp_params["front_margin_rescue_min_margin"] = float(args.front_margin_rescue_min_margin)
        if args.front_margin_rescue_margin_gap is not None:
            cfg.linear_osqp_params["front_margin_rescue_margin_gap"] = max(
                float(args.front_margin_rescue_margin_gap),
                0.0,
            )
        if args.front_margin_rescue_alpha_margin is not None:
            cfg.linear_osqp_params["front_margin_rescue_alpha_margin"] = max(
                float(args.front_margin_rescue_alpha_margin),
                1e-6,
            )
        if args.front_margin_rescue_roll_threshold is not None:
            cfg.linear_osqp_params["front_margin_rescue_roll_threshold"] = max(
                float(args.front_margin_rescue_roll_threshold), 0.0
            )
        if args.front_margin_rescue_pitch_threshold is not None:
            cfg.linear_osqp_params["front_margin_rescue_pitch_threshold"] = max(
                float(args.front_margin_rescue_pitch_threshold), 0.0
            )
        if args.front_margin_rescue_height_ratio is not None:
            cfg.linear_osqp_params["front_margin_rescue_height_ratio"] = max(
                float(args.front_margin_rescue_height_ratio), 0.0
            )
        if args.front_margin_rescue_recent_swing_window_s is not None:
            cfg.linear_osqp_params["front_margin_rescue_recent_swing_window_s"] = max(
                float(args.front_margin_rescue_recent_swing_window_s), 0.0
            )
        if args.rear_handoff_support_hold_s is not None:
            cfg.linear_osqp_params["rear_handoff_support_hold_s"] = max(float(args.rear_handoff_support_hold_s), 0.0)
        if args.rear_handoff_forward_scale is not None:
            cfg.linear_osqp_params["rear_handoff_forward_scale"] = float(
                max(min(args.rear_handoff_forward_scale, 1.0), 0.0)
            )
        if args.rear_handoff_lookahead_steps is not None:
            cfg.linear_osqp_params["rear_handoff_lookahead_steps"] = max(int(args.rear_handoff_lookahead_steps), 1)
        if args.rear_handoff_support_rear_alpha_scale is not None:
            cfg.linear_osqp_params["rear_handoff_support_rear_alpha_scale"] = float(
                max(min(args.rear_handoff_support_rear_alpha_scale, 1.0), 0.0)
            )
        if args.rear_swing_bridge_hold_s is not None:
            cfg.linear_osqp_params["rear_swing_bridge_hold_s"] = max(float(args.rear_swing_bridge_hold_s), 0.0)
        if args.rear_swing_bridge_forward_scale is not None:
            cfg.linear_osqp_params["rear_swing_bridge_forward_scale"] = float(
                max(min(args.rear_swing_bridge_forward_scale, 1.0), 0.0)
            )
        if args.rear_swing_bridge_roll_threshold is not None:
            cfg.linear_osqp_params["rear_swing_bridge_roll_threshold"] = max(
                float(args.rear_swing_bridge_roll_threshold), 0.0
            )
        if args.rear_swing_bridge_pitch_threshold is not None:
            cfg.linear_osqp_params["rear_swing_bridge_pitch_threshold"] = max(
                float(args.rear_swing_bridge_pitch_threshold), 0.0
            )
        if args.rear_swing_bridge_height_ratio is not None:
            cfg.linear_osqp_params["rear_swing_bridge_height_ratio"] = max(
                float(args.rear_swing_bridge_height_ratio), 0.0
            )
        if args.rear_swing_bridge_recent_front_window_s is not None:
            cfg.linear_osqp_params["rear_swing_bridge_recent_front_window_s"] = max(
                float(args.rear_swing_bridge_recent_front_window_s), 0.0
            )
        if args.rear_swing_bridge_lookahead_steps is not None:
            cfg.linear_osqp_params["rear_swing_bridge_lookahead_steps"] = max(
                int(args.rear_swing_bridge_lookahead_steps), 1
            )
        if args.rear_swing_bridge_allcontact_release_tail_s is not None:
            cfg.linear_osqp_params["rear_swing_bridge_allcontact_release_tail_s"] = max(
                float(args.rear_swing_bridge_allcontact_release_tail_s), 0.0
            )
        if args.rear_swing_bridge_rear_alpha_scale is not None:
            cfg.linear_osqp_params["rear_swing_bridge_rear_alpha_scale"] = float(
                max(min(args.rear_swing_bridge_rear_alpha_scale, 1.0), 0.0)
            )
        if args.front_swing_contact_release_timeout_s is not None:
            cfg.linear_osqp_params["front_swing_contact_release_timeout_s"] = max(
                float(args.front_swing_contact_release_timeout_s), 0.0
            )
        if args.rear_swing_contact_release_timeout_s is not None:
            cfg.linear_osqp_params["rear_swing_contact_release_timeout_s"] = max(
                float(args.rear_swing_contact_release_timeout_s), 0.0
            )
        if args.front_release_lift_height is not None:
            cfg.linear_osqp_params["front_release_lift_height"] = max(float(args.front_release_lift_height), 0.0)
        if args.front_release_lift_kp is not None:
            cfg.linear_osqp_params["front_release_lift_kp"] = max(float(args.front_release_lift_kp), 0.0)
        if args.front_release_lift_kd is not None:
            cfg.linear_osqp_params["front_release_lift_kd"] = max(float(args.front_release_lift_kd), 0.0)
        if args.rear_release_lift_height is not None:
            cfg.linear_osqp_params["rear_release_lift_height"] = max(float(args.rear_release_lift_height), 0.0)
        if args.rear_release_lift_kp is not None:
            cfg.linear_osqp_params["rear_release_lift_kp"] = max(float(args.rear_release_lift_kp), 0.0)
        if args.rear_release_lift_kd is not None:
            cfg.linear_osqp_params["rear_release_lift_kd"] = max(float(args.rear_release_lift_kd), 0.0)
        if args.rear_swing_release_support_hold_s is not None:
            cfg.linear_osqp_params["rear_swing_release_support_hold_s"] = max(
                float(args.rear_swing_release_support_hold_s), 0.0
            )
        if args.rear_swing_release_forward_scale is not None:
            cfg.linear_osqp_params["rear_swing_release_forward_scale"] = float(
                max(min(args.rear_swing_release_forward_scale, 1.0), 0.0)
            )
        if args.full_contact_recovery_hold_s is not None:
            cfg.linear_osqp_params["full_contact_recovery_hold_s"] = max(float(args.full_contact_recovery_hold_s), 0.0)
        if args.full_contact_recovery_forward_scale is not None:
            cfg.linear_osqp_params["full_contact_recovery_forward_scale"] = float(
                max(min(args.full_contact_recovery_forward_scale, 1.0), 0.0)
            )
        if args.full_contact_recovery_roll_threshold is not None:
            cfg.linear_osqp_params["full_contact_recovery_roll_threshold"] = max(
                float(args.full_contact_recovery_roll_threshold), 0.0
            )
        if args.full_contact_recovery_pitch_threshold is not None:
            cfg.linear_osqp_params["full_contact_recovery_pitch_threshold"] = max(
                float(args.full_contact_recovery_pitch_threshold), 0.0
            )
        if args.full_contact_recovery_height_ratio is not None:
            cfg.linear_osqp_params["full_contact_recovery_height_ratio"] = max(
                float(args.full_contact_recovery_height_ratio), 0.0
            )
        if args.full_contact_recovery_recent_window_s is not None:
            cfg.linear_osqp_params["full_contact_recovery_recent_window_s"] = max(
                float(args.full_contact_recovery_recent_window_s), 0.0
            )
        if args.full_contact_recovery_rear_support_scale is not None:
            cfg.linear_osqp_params["full_contact_recovery_rear_support_scale"] = float(
                max(min(args.full_contact_recovery_rear_support_scale, 1.0), 0.0)
            )
        if args.full_contact_recovery_support_floor_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_support_floor_delta"] = max(
                float(args.full_contact_recovery_support_floor_delta), 0.0
            )
        if args.full_contact_recovery_z_pos_gain_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_z_pos_gain_delta"] = max(
                float(args.full_contact_recovery_z_pos_gain_delta), 0.0
            )
        if args.full_contact_recovery_roll_angle_gain_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_roll_angle_gain_delta"] = max(
                float(args.full_contact_recovery_roll_angle_gain_delta), 0.0
            )
        if args.full_contact_recovery_roll_rate_gain_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_roll_rate_gain_delta"] = max(
                float(args.full_contact_recovery_roll_rate_gain_delta), 0.0
            )
        if args.full_contact_recovery_pitch_angle_gain_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_pitch_angle_gain_delta"] = max(
                float(args.full_contact_recovery_pitch_angle_gain_delta), 0.0
            )
        if args.full_contact_recovery_pitch_rate_gain_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_pitch_rate_gain_delta"] = max(
                float(args.full_contact_recovery_pitch_rate_gain_delta), 0.0
            )
        if args.full_contact_recovery_side_rebalance_delta is not None:
            cfg.linear_osqp_params["full_contact_recovery_side_rebalance_delta"] = max(
                float(args.full_contact_recovery_side_rebalance_delta), 0.0
            )
        if args.full_contact_recovery_allcontact_release_tail_s is not None:
            cfg.linear_osqp_params["full_contact_recovery_allcontact_release_tail_s"] = max(
                float(args.full_contact_recovery_allcontact_release_tail_s), 0.0
            )
        if args.crawl_front_delayed_swing_recovery_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_delayed_swing_recovery_hold_s"] = max(
                float(args.crawl_front_delayed_swing_recovery_hold_s), 0.0
            )
        if args.crawl_front_delayed_swing_recovery_margin_threshold is not None:
            cfg.linear_osqp_params["crawl_front_delayed_swing_recovery_margin_threshold"] = float(
                args.crawl_front_delayed_swing_recovery_margin_threshold
            )
        if args.crawl_front_delayed_swing_recovery_once_per_swing is not None:
            cfg.linear_osqp_params["crawl_front_delayed_swing_recovery_once_per_swing"] = bool(
                args.crawl_front_delayed_swing_recovery_once_per_swing
            )
        if args.crawl_front_delayed_swing_recovery_release_tail_s is not None:
            cfg.linear_osqp_params["crawl_front_delayed_swing_recovery_release_tail_s"] = max(
                float(args.crawl_front_delayed_swing_recovery_release_tail_s), 0.0
            )
        if args.crawl_front_delayed_swing_recovery_rearm_trigger_s is not None:
            cfg.linear_osqp_params["crawl_front_delayed_swing_recovery_rearm_trigger_s"] = max(
                float(args.crawl_front_delayed_swing_recovery_rearm_trigger_s), 0.0
            )
        if args.crawl_front_planted_swing_recovery_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_planted_swing_recovery_hold_s"] = max(
                float(args.crawl_front_planted_swing_recovery_hold_s), 0.0
            )
        if args.crawl_front_planted_swing_recovery_margin_threshold is not None:
            cfg.linear_osqp_params["crawl_front_planted_swing_recovery_margin_threshold"] = float(
                args.crawl_front_planted_swing_recovery_margin_threshold
            )
        if args.crawl_front_planted_swing_recovery_height_ratio is not None:
            cfg.linear_osqp_params["crawl_front_planted_swing_recovery_height_ratio"] = max(
                float(args.crawl_front_planted_swing_recovery_height_ratio), 0.0
            )
        if args.crawl_front_planted_swing_recovery_roll_threshold is not None:
            cfg.linear_osqp_params["crawl_front_planted_swing_recovery_roll_threshold"] = max(
                float(args.crawl_front_planted_swing_recovery_roll_threshold), 0.0
            )
        if args.crawl_front_planted_swing_recovery_rearm_trigger_s is not None:
            cfg.linear_osqp_params["crawl_front_planted_swing_recovery_rearm_trigger_s"] = max(
                float(args.crawl_front_planted_swing_recovery_rearm_trigger_s), 0.0
            )
        if args.crawl_front_planted_postdrop_recovery_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_planted_postdrop_recovery_hold_s"] = max(
                float(args.crawl_front_planted_postdrop_recovery_hold_s), 0.0
            )
        if args.crawl_front_planted_seam_support_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_planted_seam_support_hold_s"] = max(
                float(args.crawl_front_planted_seam_support_hold_s), 0.0
            )
        if args.crawl_front_stance_support_tail_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_stance_support_tail_hold_s"] = max(
                float(args.crawl_front_stance_support_tail_hold_s), 0.0
            )
        if args.crawl_front_close_gap_support_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_close_gap_support_hold_s"] = max(
                float(args.crawl_front_close_gap_support_hold_s), 0.0
            )
        if args.crawl_front_close_gap_keep_swing is not None:
            cfg.linear_osqp_params["crawl_front_close_gap_keep_swing"] = bool(
                int(args.crawl_front_close_gap_keep_swing)
            )
        if args.crawl_front_late_rearm_tail_hold_s is not None:
            cfg.linear_osqp_params["crawl_front_late_rearm_tail_hold_s"] = max(
                float(args.crawl_front_late_rearm_tail_hold_s), 0.0
            )
        if args.crawl_front_late_rearm_budget_s is not None:
            cfg.linear_osqp_params["crawl_front_late_rearm_budget_s"] = max(
                float(args.crawl_front_late_rearm_budget_s), 0.0
            )
        if args.crawl_front_late_rearm_min_swing_time_s is not None:
            cfg.linear_osqp_params["crawl_front_late_rearm_min_swing_time_s"] = max(
                float(args.crawl_front_late_rearm_min_swing_time_s), 0.0
            )
        if args.crawl_front_late_rearm_min_negative_margin is not None:
            cfg.linear_osqp_params["crawl_front_late_rearm_min_negative_margin"] = max(
                float(args.crawl_front_late_rearm_min_negative_margin), 0.0
            )
        if args.pre_swing_gate_hold_s is not None:
            cfg.linear_osqp_params["pre_swing_gate_hold_s"] = max(float(args.pre_swing_gate_hold_s), 0.0)
        if args.rear_pre_swing_gate_hold_s is not None:
            cfg.linear_osqp_params["rear_pre_swing_gate_hold_s"] = max(float(args.rear_pre_swing_gate_hold_s), 0.0)
        if args.rear_pre_swing_guard_roll_threshold is not None:
            cfg.linear_osqp_params["rear_pre_swing_guard_roll_threshold"] = max(
                float(args.rear_pre_swing_guard_roll_threshold),
                0.0,
            )
        if args.rear_pre_swing_guard_pitch_threshold is not None:
            cfg.linear_osqp_params["rear_pre_swing_guard_pitch_threshold"] = max(
                float(args.rear_pre_swing_guard_pitch_threshold),
                0.0,
            )
        if args.rear_pre_swing_guard_height_ratio is not None:
            cfg.linear_osqp_params["rear_pre_swing_guard_height_ratio"] = max(
                float(args.rear_pre_swing_guard_height_ratio),
                0.0,
            )
        if args.pre_swing_gate_forward_scale is not None:
            cfg.linear_osqp_params["pre_swing_gate_forward_scale"] = float(args.pre_swing_gate_forward_scale)
        if args.vx_gain is not None:
            cfg.linear_osqp_params["vx_gain"] = args.vx_gain
        if args.vy_gain is not None:
            cfg.linear_osqp_params["vy_gain"] = args.vy_gain
        if args.z_pos_gain is not None:
            cfg.linear_osqp_params["z_pos_gain"] = args.z_pos_gain
        if args.z_vel_gain is not None:
            cfg.linear_osqp_params["z_vel_gain"] = args.z_vel_gain
        if args.min_vertical_force_scale is not None:
            cfg.linear_osqp_params["min_vertical_force_scale"] = args.min_vertical_force_scale
        if args.reduced_support_vertical_boost is not None:
            cfg.linear_osqp_params["reduced_support_vertical_boost"] = args.reduced_support_vertical_boost
        if args.roll_angle_gain is not None:
            cfg.linear_osqp_params["roll_angle_gain"] = args.roll_angle_gain
        if args.roll_rate_gain is not None:
            cfg.linear_osqp_params["roll_rate_gain"] = args.roll_rate_gain
        if args.pitch_angle_gain is not None:
            cfg.linear_osqp_params["pitch_angle_gain"] = args.pitch_angle_gain
        if args.pitch_rate_gain is not None:
            cfg.linear_osqp_params["pitch_rate_gain"] = args.pitch_rate_gain
        if args.yaw_angle_gain is not None:
            cfg.linear_osqp_params["yaw_angle_gain"] = args.yaw_angle_gain
        if args.yaw_rate_gain is not None:
            cfg.linear_osqp_params["yaw_rate_gain"] = args.yaw_rate_gain
        if args.foothold_yaw_rate_scale is not None:
            cfg.linear_osqp_params["foothold_yaw_rate_scale"] = float(args.foothold_yaw_rate_scale)
        if args.foothold_yaw_error_scale is not None:
            cfg.linear_osqp_params["foothold_yaw_error_scale"] = float(args.foothold_yaw_error_scale)
        if args.roll_ref_offset is not None:
            cfg.linear_osqp_params["roll_ref_offset"] = args.roll_ref_offset
        if args.pitch_ref_offset is not None:
            cfg.linear_osqp_params["pitch_ref_offset"] = args.pitch_ref_offset
        if args.latched_swing_xy_blend is not None:
            cfg.linear_osqp_params["latched_swing_xy_blend"] = args.latched_swing_xy_blend
        if args.latched_swing_lift_ratio is not None:
            cfg.linear_osqp_params["latched_swing_lift_ratio"] = args.latched_swing_lift_ratio
        if args.latched_swing_tau_blend is not None:
            cfg.linear_osqp_params["latched_swing_tau_blend"] = args.latched_swing_tau_blend
        if args.contact_latch_steps is not None:
            cfg.linear_osqp_params["contact_latch_steps"] = args.contact_latch_steps
        if args.contact_latch_budget_s is not None:
            cfg.linear_osqp_params["contact_latch_budget_s"] = max(float(args.contact_latch_budget_s), 0.0)
        elif args.contact_latch_budget_steps is not None:
            cfg.linear_osqp_params["contact_latch_budget_s"] = max(int(args.contact_latch_budget_steps), 0) * float(cfg.mpc_params["dt"])
            cfg.linear_osqp_params["contact_latch_budget_steps"] = args.contact_latch_budget_steps
        if args.front_contact_latch_steps is not None:
            cfg.linear_osqp_params["front_contact_latch_steps"] = max(int(args.front_contact_latch_steps), 0)
        if args.front_contact_latch_budget_s is not None:
            cfg.linear_osqp_params["front_contact_latch_budget_s"] = max(float(args.front_contact_latch_budget_s), 0.0)
        if args.rear_contact_latch_steps is not None:
            cfg.linear_osqp_params["rear_contact_latch_steps"] = max(int(args.rear_contact_latch_steps), 0)
        if args.rear_contact_latch_budget_s is not None:
            cfg.linear_osqp_params["rear_contact_latch_budget_s"] = max(float(args.rear_contact_latch_budget_s), 0.0)
        if args.startup_full_stance_time_s is not None:
            cfg.linear_osqp_params["startup_full_stance_time_s"] = max(float(args.startup_full_stance_time_s), 0.0)
        elif args.startup_full_stance_steps is not None:
            cfg.linear_osqp_params["startup_full_stance_time_s"] = max(int(args.startup_full_stance_steps), 0) * float(cfg.mpc_params["dt"])
            cfg.linear_osqp_params["startup_full_stance_steps"] = args.startup_full_stance_steps
        if args.virtual_unlatch_phase_threshold is not None:
            cfg.linear_osqp_params["virtual_unlatch_phase_threshold"] = args.virtual_unlatch_phase_threshold
        if args.virtual_unlatch_hold_s is not None:
            cfg.linear_osqp_params["virtual_unlatch_hold_s"] = max(float(args.virtual_unlatch_hold_s), 0.0)
        elif args.virtual_unlatch_hold_steps is not None:
            cfg.linear_osqp_params["virtual_unlatch_hold_s"] = max(int(args.virtual_unlatch_hold_steps), 0) * float(cfg.mpc_params["dt"])
            cfg.linear_osqp_params["virtual_unlatch_hold_steps"] = args.virtual_unlatch_hold_steps
        if args.pre_swing_lookahead_steps is not None:
            cfg.linear_osqp_params["pre_swing_lookahead_steps"] = max(int(args.pre_swing_lookahead_steps), 0)

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
    if args.gait == "trot":
        print(f"Selected trot dynamic profile: {selected_dynamic_trot_profile}")
    if output is not None:
        print(f"\nPrimary output path: {output}")


if __name__ == "__main__":
    main()
