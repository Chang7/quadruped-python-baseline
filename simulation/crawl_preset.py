"""Crawl-specific conservative preset parameters.

These are diagnostic-scenario parameters for crawl contact-transition
debugging. They are not needed for trot deployment.
"""


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
