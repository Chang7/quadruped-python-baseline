"""Crawl-specific conservative preset parameters and CLI helpers.

These are diagnostic-scenario parameters for crawl contact-transition
debugging. They are not needed for trot deployment.
"""

from __future__ import annotations

import argparse


def crawl_conservative_params() -> dict:
    """Return crawl-specific parameter overrides for the conservative preset."""
    return {
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


def add_crawl_recovery_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register crawl-only late-recovery tuning arguments."""
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


def add_crawl_allcontact_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register crawl-only rear all-contact stabilization tuning arguments."""
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


def apply_crawl_recovery_cli_overrides(args: argparse.Namespace, params: dict) -> None:
    """Apply crawl-only late-recovery CLI overrides into linear_osqp_params."""
    if args.full_contact_recovery_hold_s is not None:
        params["full_contact_recovery_hold_s"] = max(float(args.full_contact_recovery_hold_s), 0.0)
    if args.full_contact_recovery_forward_scale is not None:
        params["full_contact_recovery_forward_scale"] = float(
            max(min(args.full_contact_recovery_forward_scale, 1.0), 0.0)
        )
    if args.full_contact_recovery_roll_threshold is not None:
        params["full_contact_recovery_roll_threshold"] = max(
            float(args.full_contact_recovery_roll_threshold), 0.0
        )
    if args.full_contact_recovery_pitch_threshold is not None:
        params["full_contact_recovery_pitch_threshold"] = max(
            float(args.full_contact_recovery_pitch_threshold), 0.0
        )
    if args.full_contact_recovery_height_ratio is not None:
        params["full_contact_recovery_height_ratio"] = max(
            float(args.full_contact_recovery_height_ratio), 0.0
        )
    if args.full_contact_recovery_recent_window_s is not None:
        params["full_contact_recovery_recent_window_s"] = max(
            float(args.full_contact_recovery_recent_window_s), 0.0
        )
    if args.full_contact_recovery_rear_support_scale is not None:
        params["full_contact_recovery_rear_support_scale"] = float(
            max(min(args.full_contact_recovery_rear_support_scale, 1.0), 0.0)
        )
    if args.full_contact_recovery_support_floor_delta is not None:
        params["full_contact_recovery_support_floor_delta"] = max(
            float(args.full_contact_recovery_support_floor_delta), 0.0
        )
    if args.full_contact_recovery_z_pos_gain_delta is not None:
        params["full_contact_recovery_z_pos_gain_delta"] = max(
            float(args.full_contact_recovery_z_pos_gain_delta), 0.0
        )
    if args.full_contact_recovery_roll_angle_gain_delta is not None:
        params["full_contact_recovery_roll_angle_gain_delta"] = max(
            float(args.full_contact_recovery_roll_angle_gain_delta), 0.0
        )
    if args.full_contact_recovery_roll_rate_gain_delta is not None:
        params["full_contact_recovery_roll_rate_gain_delta"] = max(
            float(args.full_contact_recovery_roll_rate_gain_delta), 0.0
        )
    if args.full_contact_recovery_pitch_angle_gain_delta is not None:
        params["full_contact_recovery_pitch_angle_gain_delta"] = max(
            float(args.full_contact_recovery_pitch_angle_gain_delta), 0.0
        )
    if args.full_contact_recovery_pitch_rate_gain_delta is not None:
        params["full_contact_recovery_pitch_rate_gain_delta"] = max(
            float(args.full_contact_recovery_pitch_rate_gain_delta), 0.0
        )
    if args.full_contact_recovery_side_rebalance_delta is not None:
        params["full_contact_recovery_side_rebalance_delta"] = max(
            float(args.full_contact_recovery_side_rebalance_delta), 0.0
        )
    if args.full_contact_recovery_allcontact_release_tail_s is not None:
        params["full_contact_recovery_allcontact_release_tail_s"] = max(
            float(args.full_contact_recovery_allcontact_release_tail_s), 0.0
        )
    if args.crawl_front_delayed_swing_recovery_hold_s is not None:
        params["crawl_front_delayed_swing_recovery_hold_s"] = max(
            float(args.crawl_front_delayed_swing_recovery_hold_s), 0.0
        )
    if args.crawl_front_delayed_swing_recovery_margin_threshold is not None:
        params["crawl_front_delayed_swing_recovery_margin_threshold"] = float(
            args.crawl_front_delayed_swing_recovery_margin_threshold
        )
    if args.crawl_front_delayed_swing_recovery_once_per_swing is not None:
        params["crawl_front_delayed_swing_recovery_once_per_swing"] = bool(
            args.crawl_front_delayed_swing_recovery_once_per_swing
        )
    if args.crawl_front_delayed_swing_recovery_release_tail_s is not None:
        params["crawl_front_delayed_swing_recovery_release_tail_s"] = max(
            float(args.crawl_front_delayed_swing_recovery_release_tail_s), 0.0
        )
    if args.crawl_front_delayed_swing_recovery_rearm_trigger_s is not None:
        params["crawl_front_delayed_swing_recovery_rearm_trigger_s"] = max(
            float(args.crawl_front_delayed_swing_recovery_rearm_trigger_s), 0.0
        )
    if args.crawl_front_planted_swing_recovery_hold_s is not None:
        params["crawl_front_planted_swing_recovery_hold_s"] = max(
            float(args.crawl_front_planted_swing_recovery_hold_s), 0.0
        )
    if args.crawl_front_planted_swing_recovery_margin_threshold is not None:
        params["crawl_front_planted_swing_recovery_margin_threshold"] = float(
            args.crawl_front_planted_swing_recovery_margin_threshold
        )
    if args.crawl_front_planted_swing_recovery_height_ratio is not None:
        params["crawl_front_planted_swing_recovery_height_ratio"] = max(
            float(args.crawl_front_planted_swing_recovery_height_ratio), 0.0
        )
    if args.crawl_front_planted_swing_recovery_roll_threshold is not None:
        params["crawl_front_planted_swing_recovery_roll_threshold"] = max(
            float(args.crawl_front_planted_swing_recovery_roll_threshold), 0.0
        )
    if args.crawl_front_planted_swing_recovery_rearm_trigger_s is not None:
        params["crawl_front_planted_swing_recovery_rearm_trigger_s"] = max(
            float(args.crawl_front_planted_swing_recovery_rearm_trigger_s), 0.0
        )
    if args.crawl_front_planted_postdrop_recovery_hold_s is not None:
        params["crawl_front_planted_postdrop_recovery_hold_s"] = max(
            float(args.crawl_front_planted_postdrop_recovery_hold_s), 0.0
        )
    if args.crawl_front_planted_seam_support_hold_s is not None:
        params["crawl_front_planted_seam_support_hold_s"] = max(
            float(args.crawl_front_planted_seam_support_hold_s), 0.0
        )
    if args.crawl_front_stance_support_tail_hold_s is not None:
        params["crawl_front_stance_support_tail_hold_s"] = max(
            float(args.crawl_front_stance_support_tail_hold_s), 0.0
        )
    if args.crawl_front_close_gap_support_hold_s is not None:
        params["crawl_front_close_gap_support_hold_s"] = max(
            float(args.crawl_front_close_gap_support_hold_s), 0.0
        )
    if args.crawl_front_close_gap_keep_swing is not None:
        params["crawl_front_close_gap_keep_swing"] = bool(int(args.crawl_front_close_gap_keep_swing))
    if args.crawl_front_late_rearm_tail_hold_s is not None:
        params["crawl_front_late_rearm_tail_hold_s"] = max(
            float(args.crawl_front_late_rearm_tail_hold_s), 0.0
        )
    if args.crawl_front_late_rearm_budget_s is not None:
        params["crawl_front_late_rearm_budget_s"] = max(
            float(args.crawl_front_late_rearm_budget_s), 0.0
        )
    if args.crawl_front_late_rearm_min_swing_time_s is not None:
        params["crawl_front_late_rearm_min_swing_time_s"] = max(
            float(args.crawl_front_late_rearm_min_swing_time_s), 0.0
        )
    if args.crawl_front_late_rearm_min_negative_margin is not None:
        params["crawl_front_late_rearm_min_negative_margin"] = max(
            float(args.crawl_front_late_rearm_min_negative_margin), 0.0
        )


def apply_crawl_allcontact_cli_overrides(args: argparse.Namespace, params: dict) -> None:
    """Apply crawl-only rear all-contact stabilization overrides."""
    if args.rear_all_contact_stabilization_hold_s is not None:
        params["rear_all_contact_stabilization_hold_s"] = max(
            float(args.rear_all_contact_stabilization_hold_s), 0.0
        )
    if args.rear_all_contact_stabilization_forward_scale is not None:
        params["rear_all_contact_stabilization_forward_scale"] = float(
            max(min(args.rear_all_contact_stabilization_forward_scale, 1.0), 0.0)
        )
    if args.rear_all_contact_stabilization_front_alpha_scale is not None:
        params["rear_all_contact_stabilization_front_alpha_scale"] = float(
            max(min(args.rear_all_contact_stabilization_front_alpha_scale, 1.0), 0.0)
        )
    if args.rear_all_contact_stabilization_height_ratio is not None:
        params["rear_all_contact_stabilization_height_ratio"] = max(
            float(args.rear_all_contact_stabilization_height_ratio), 0.0
        )
    if args.rear_all_contact_stabilization_roll_threshold is not None:
        params["rear_all_contact_stabilization_roll_threshold"] = max(
            float(args.rear_all_contact_stabilization_roll_threshold), 0.0
        )
    if args.rear_all_contact_stabilization_pitch_threshold is not None:
        params["rear_all_contact_stabilization_pitch_threshold"] = max(
            float(args.rear_all_contact_stabilization_pitch_threshold), 0.0
        )
    if args.rear_late_seam_support_trigger_s is not None:
        params["rear_late_seam_support_trigger_s"] = max(
            float(args.rear_late_seam_support_trigger_s), 0.0
        )
    if args.rear_close_handoff_hold_s is not None:
        params["rear_close_handoff_hold_s"] = max(
            float(args.rear_close_handoff_hold_s), 0.0
        )
    if args.rear_close_handoff_leg_floor_scale_delta is not None:
        params["rear_close_handoff_leg_floor_scale_delta"] = max(
            float(args.rear_close_handoff_leg_floor_scale_delta), 0.0
        )
    if args.rear_late_load_share_support_hold_s is not None:
        params["rear_late_load_share_support_hold_s"] = max(
            float(args.rear_late_load_share_support_hold_s), 0.0
        )
    if args.rear_late_load_share_support_min_leg_share is not None:
        params["rear_late_load_share_support_min_leg_share"] = max(
            float(args.rear_late_load_share_support_min_leg_share), 0.0
        )
    if args.rear_late_load_share_support_height_ratio is not None:
        params["rear_late_load_share_support_height_ratio"] = max(
            float(args.rear_late_load_share_support_height_ratio), 0.0
        )
    if args.rear_late_load_share_support_min_persist_s is not None:
        params["rear_late_load_share_support_min_persist_s"] = max(
            float(args.rear_late_load_share_support_min_persist_s), 0.0
        )
    if args.rear_late_load_share_support_alpha_cap is not None:
        params["rear_late_load_share_support_alpha_cap"] = float(
            max(min(float(args.rear_late_load_share_support_alpha_cap), 1.0), 0.0)
        )
    if args.rear_late_load_share_support_leg_floor_scale_delta is not None:
        params["rear_late_load_share_support_leg_floor_scale_delta"] = max(
            float(args.rear_late_load_share_support_leg_floor_scale_delta), 0.0
        )
    if args.rear_all_contact_stabilization_min_rear_load_share is not None:
        params["rear_all_contact_stabilization_min_rear_load_share"] = max(
            float(args.rear_all_contact_stabilization_min_rear_load_share), 0.0
        )
    if args.rear_all_contact_stabilization_min_rear_leg_load_share is not None:
        params["rear_all_contact_stabilization_min_rear_leg_load_share"] = max(
            float(args.rear_all_contact_stabilization_min_rear_leg_load_share), 0.0
        )
    if args.rear_all_contact_stabilization_weak_leg_share_ref is not None:
        params["rear_all_contact_stabilization_weak_leg_share_ref"] = max(
            float(args.rear_all_contact_stabilization_weak_leg_share_ref), 0.0
        )
    if args.rear_all_contact_stabilization_weak_leg_floor_delta is not None:
        params["rear_all_contact_stabilization_weak_leg_floor_delta"] = max(
            float(args.rear_all_contact_stabilization_weak_leg_floor_delta), 0.0
        )
    if args.rear_all_contact_stabilization_weak_leg_height_ratio is not None:
        params["rear_all_contact_stabilization_weak_leg_height_ratio"] = max(
            float(args.rear_all_contact_stabilization_weak_leg_height_ratio), 0.0
        )
    if args.rear_all_contact_stabilization_weak_leg_tail_only:
        params["rear_all_contact_stabilization_weak_leg_tail_only"] = True
    if args.crawl_front_planted_weak_rear_share_ref is not None:
        params["crawl_front_planted_weak_rear_share_ref"] = max(
            float(args.crawl_front_planted_weak_rear_share_ref), 0.0
        )
    if args.crawl_front_planted_weak_rear_alpha_cap is not None:
        params["crawl_front_planted_weak_rear_alpha_cap"] = float(
            max(min(float(args.crawl_front_planted_weak_rear_alpha_cap), 1.0), 0.0)
        )
    if args.rear_all_contact_stabilization_retrigger_limit is not None:
        params["rear_all_contact_stabilization_retrigger_limit"] = max(
            int(args.rear_all_contact_stabilization_retrigger_limit), 0
        )
    if args.rear_all_contact_stabilization_rear_floor_delta is not None:
        params["rear_all_contact_stabilization_rear_floor_delta"] = max(
            float(args.rear_all_contact_stabilization_rear_floor_delta), 0.0
        )
    if args.rear_all_contact_stabilization_z_pos_gain_delta is not None:
        params["rear_all_contact_stabilization_z_pos_gain_delta"] = max(
            float(args.rear_all_contact_stabilization_z_pos_gain_delta), 0.0
        )
    if args.rear_all_contact_stabilization_roll_angle_gain_delta is not None:
        params["rear_all_contact_stabilization_roll_angle_gain_delta"] = max(
            float(args.rear_all_contact_stabilization_roll_angle_gain_delta), 0.0
        )
    if args.rear_all_contact_stabilization_roll_rate_gain_delta is not None:
        params["rear_all_contact_stabilization_roll_rate_gain_delta"] = max(
            float(args.rear_all_contact_stabilization_roll_rate_gain_delta), 0.0
        )
    if args.rear_all_contact_stabilization_side_rebalance_delta is not None:
        params["rear_all_contact_stabilization_side_rebalance_delta"] = max(
            float(args.rear_all_contact_stabilization_side_rebalance_delta), 0.0
        )
    if args.rear_all_contact_stabilization_front_anchor_z_blend is not None:
        params["rear_all_contact_stabilization_front_anchor_z_blend"] = float(
            max(min(args.rear_all_contact_stabilization_front_anchor_z_blend, 1.0), 0.0)
        )
    if args.rear_all_contact_stabilization_rear_anchor_z_blend is not None:
        params["rear_all_contact_stabilization_rear_anchor_z_blend"] = float(
            max(min(args.rear_all_contact_stabilization_rear_anchor_z_blend, 1.0), 0.0)
        )
    if args.rear_all_contact_stabilization_front_anchor_z_max_delta is not None:
        params["rear_all_contact_stabilization_front_anchor_z_max_delta"] = max(
            float(args.rear_all_contact_stabilization_front_anchor_z_max_delta), 0.0
        )
    if args.rear_all_contact_stabilization_rear_anchor_z_max_delta is not None:
        params["rear_all_contact_stabilization_rear_anchor_z_max_delta"] = max(
            float(args.rear_all_contact_stabilization_rear_anchor_z_max_delta), 0.0
        )
    if args.rear_all_contact_post_recovery_tail_hold_s is not None:
        params["rear_all_contact_post_recovery_tail_hold_s"] = max(
            float(args.rear_all_contact_post_recovery_tail_hold_s), 0.0
        )
    if args.rear_all_contact_release_tail_alpha_scale is not None:
        params["rear_all_contact_release_tail_alpha_scale"] = max(
            float(args.rear_all_contact_release_tail_alpha_scale), 0.0
        )
    if args.rear_all_contact_post_recovery_front_late_alpha_scale is not None:
        params["rear_all_contact_post_recovery_front_late_alpha_scale"] = max(
            float(args.rear_all_contact_post_recovery_front_late_alpha_scale), 0.0
        )
    if args.front_rear_transition_guard_hold_s is not None:
        params["front_rear_transition_guard_hold_s"] = max(
            float(args.front_rear_transition_guard_hold_s), 0.0
        )
    if args.front_rear_transition_guard_forward_scale is not None:
        params["front_rear_transition_guard_forward_scale"] = float(
            max(min(float(args.front_rear_transition_guard_forward_scale), 1.0), 0.0)
        )
    if args.front_rear_transition_guard_roll_threshold is not None:
        params["front_rear_transition_guard_roll_threshold"] = max(
            float(args.front_rear_transition_guard_roll_threshold), 0.0
        )
    if args.front_rear_transition_guard_pitch_threshold is not None:
        params["front_rear_transition_guard_pitch_threshold"] = max(
            float(args.front_rear_transition_guard_pitch_threshold), 0.0
        )
    if args.front_rear_transition_guard_height_ratio is not None:
        params["front_rear_transition_guard_height_ratio"] = max(
            float(args.front_rear_transition_guard_height_ratio), 0.0
        )
    if args.front_rear_transition_guard_release_tail_s is not None:
        params["front_rear_transition_guard_release_tail_s"] = max(
            float(args.front_rear_transition_guard_release_tail_s), 0.0
        )
    if args.front_rear_transition_guard_margin_release is not None:
        params["front_rear_transition_guard_margin_release"] = max(
            float(args.front_rear_transition_guard_margin_release), 0.0
        )
