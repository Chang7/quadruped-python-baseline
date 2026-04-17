"""Linear timing parameter parsing, extracted from WBInterface."""
from __future__ import annotations

import numpy as np

from quadruped_pympc import config as cfg


class LinearTimingParamsMixin:
    """Mixin supplying _refresh_linear_timing_params to WBInterface."""

    def _refresh_linear_timing_params(self) -> None:
        params = getattr(cfg, 'linear_osqp_params', {})
        self.frg.freeze_world_z_during_contact_phases = bool(
            cfg.mpc_params['type'] == 'linear_osqp' and self.is_dynamic_gait
        )
        self.frg.yaw_rate_compensation_scale = float(params.get('foothold_yaw_rate_scale', 0.0))
        self.frg.yaw_error_compensation_scale = float(params.get('foothold_yaw_error_scale', 0.0))
        self.contact_latch_steps = int(params.get('contact_latch_steps', 6))
        self.startup_full_stance_time_s = self._resolve_duration_seconds(
            params, 'startup_full_stance_time_s', 'startup_full_stance_steps', step_dt=self.mpc_dt
        )
        self.contact_latch_budget_s = self._resolve_duration_seconds(
            params, 'contact_latch_budget_s', 'contact_latch_budget_steps', step_dt=self.mpc_dt
        )
        front_contact_latch_steps = params.get('front_contact_latch_steps', None)
        self.front_contact_latch_steps = max(
            int(self.contact_latch_steps if front_contact_latch_steps is None else front_contact_latch_steps),
            0,
        )
        front_contact_latch_budget_s = params.get('front_contact_latch_budget_s', None)
        self.front_contact_latch_budget_s = max(
            float(self.contact_latch_budget_s if front_contact_latch_budget_s is None else front_contact_latch_budget_s),
            0.0,
        )
        rear_contact_latch_steps = params.get('rear_contact_latch_steps', None)
        self.rear_contact_latch_steps = max(
            int(self.contact_latch_steps if rear_contact_latch_steps is None else rear_contact_latch_steps),
            0,
        )
        rear_contact_latch_budget_s = params.get('rear_contact_latch_budget_s', None)
        self.rear_contact_latch_budget_s = max(
            float(self.contact_latch_budget_s if rear_contact_latch_budget_s is None else rear_contact_latch_budget_s),
            0.0,
        )
        self.virtual_unlatch_hold_s = self._resolve_duration_seconds(
            params, 'virtual_unlatch_hold_s', 'virtual_unlatch_hold_steps', step_dt=self.mpc_dt
        )
        self.pre_swing_gate_min_margin = max(float(params.get('pre_swing_gate_min_margin', 0.0)), 0.0)
        front_margin = params.get('front_pre_swing_gate_min_margin', None)
        rear_margin = params.get('rear_pre_swing_gate_min_margin', None)
        # Allow per-leg overrides to go slightly negative for targeted diagnostics.
        # The shared/base margin remains nonnegative by default, so unchanged presets
        # behave exactly as before unless an explicit override is provided.
        self.front_pre_swing_gate_min_margin = float(
            self.pre_swing_gate_min_margin if front_margin is None else front_margin
        )
        self.rear_pre_swing_gate_min_margin = float(
            self.pre_swing_gate_min_margin if rear_margin is None else rear_margin
        )
        self.swing_contact_release_timeout_s = max(float(params.get('swing_contact_release_timeout_s', 0.0)), 0.0)
        front_release_timeout = params.get('front_swing_contact_release_timeout_s', None)
        rear_release_timeout = params.get('rear_swing_contact_release_timeout_s', None)
        self.front_swing_contact_release_timeout_s = max(
            float(self.swing_contact_release_timeout_s if front_release_timeout is None else front_release_timeout),
            0.0,
        )
        self.rear_swing_contact_release_timeout_s = max(
            float(self.swing_contact_release_timeout_s if rear_release_timeout is None else rear_release_timeout),
            0.0,
        )
        self._parse_crawl_params(params)
        self.front_release_lift_height = max(float(params.get('front_release_lift_height', 0.0)), 0.0)
        self.front_release_lift_kp = max(float(params.get('front_release_lift_kp', 0.0)), 0.0)
        self.front_release_lift_kd = max(float(params.get('front_release_lift_kd', 0.0)), 0.0)
        self.rear_release_lift_height = max(float(params.get('rear_release_lift_height', 0.0)), 0.0)
        self.rear_release_lift_kp = max(float(params.get('rear_release_lift_kp', 0.0)), 0.0)
        self.rear_release_lift_kd = max(float(params.get('rear_release_lift_kd', 0.0)), 0.0)
        self.support_contact_confirm_hold_s = max(float(params.get('support_contact_confirm_hold_s', 0.0)), 0.0)
        front_confirm_hold = params.get('front_support_contact_confirm_hold_s', None)
        rear_confirm_hold = params.get('rear_support_contact_confirm_hold_s', None)
        self.front_support_contact_confirm_hold_s = max(
            float(self.support_contact_confirm_hold_s if front_confirm_hold is None else front_confirm_hold),
            0.0,
        )
        self.rear_support_contact_confirm_hold_s = max(
            float(self.support_contact_confirm_hold_s if rear_confirm_hold is None else rear_confirm_hold),
            0.0,
        )
        self.support_confirm_min_contacts = max(int(params.get('support_confirm_min_contacts', 3)), 1)
        self.support_confirm_require_front_rear_span = bool(
            params.get('support_confirm_require_front_rear_span', True)
        )
        self.support_confirm_forward_scale = float(
            np.clip(params.get('support_confirm_forward_scale', 1.0), 0.0, 1.0)
        )
        self.front_stance_dropout_reacquire = bool(params.get('front_stance_dropout_reacquire', False))
        self.front_stance_dropout_support_hold_s = max(
            float(params.get('front_stance_dropout_support_hold_s', 0.0)),
            0.0,
        )
        self.front_stance_dropout_support_forward_scale = float(
            np.clip(params.get('front_stance_dropout_support_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_stance_dropout_reacquire = bool(params.get('rear_stance_dropout_reacquire', False))
        late_release_margin = params.get('front_late_release_min_margin', None)
        self.front_late_release_phase_threshold = float(
            params.get('front_late_release_phase_threshold', 1.1)
        )
        self.front_late_release_min_margin = max(
            float(self.front_pre_swing_gate_min_margin if late_release_margin is None else late_release_margin),
            0.0,
        )
        self.front_late_release_hold_s = self._resolve_duration_seconds(
            params,
            'front_late_release_hold_s',
            'front_late_release_hold_steps',
            step_dt=self.mpc_dt,
        )
        self.front_late_release_extra_margin = max(float(params.get('front_late_release_extra_margin', 0.0)), 0.0)
        pitch_guard = params.get('front_late_release_pitch_guard', None)
        roll_guard = params.get('front_late_release_roll_guard', None)
        self.front_late_release_pitch_guard = np.inf if pitch_guard is None else max(float(pitch_guard), 0.0)
        self.front_late_release_roll_guard = np.inf if roll_guard is None else max(float(roll_guard), 0.0)
        self.support_margin_preview_s = max(float(params.get('support_margin_preview_s', 0.0)), 0.0)
        self.touchdown_reacquire_hold_s = max(float(params.get('touchdown_reacquire_hold_s', 0.0)), 0.0)
        front_touchdown_hold = params.get('front_touchdown_reacquire_hold_s', None)
        rear_touchdown_hold = params.get('rear_touchdown_reacquire_hold_s', None)
        self.front_touchdown_reacquire_hold_s = max(
            float(self.touchdown_reacquire_hold_s if front_touchdown_hold is None else front_touchdown_hold),
            0.0,
        )
        self.rear_touchdown_reacquire_hold_s = max(
            float(self.touchdown_reacquire_hold_s if rear_touchdown_hold is None else rear_touchdown_hold),
            0.0,
        )
        self.touchdown_reacquire_forward_scale = float(
            np.clip(params.get('touchdown_reacquire_forward_scale', 1.0), 0.0, 1.0)
        )
        touchdown_xy_blend = params.get('touchdown_reacquire_xy_blend', 0.0)
        front_touchdown_xy_blend = params.get('front_touchdown_reacquire_xy_blend', None)
        rear_touchdown_xy_blend = params.get('rear_touchdown_reacquire_xy_blend', None)
        touchdown_extra_depth = params.get('touchdown_reacquire_extra_depth', 0.0)
        front_touchdown_extra_depth = params.get('front_touchdown_reacquire_extra_depth', None)
        rear_touchdown_extra_depth = params.get('rear_touchdown_reacquire_extra_depth', None)
        self.touchdown_reacquire_xy_blend = float(np.clip(touchdown_xy_blend, 0.0, 1.0))
        self.front_touchdown_reacquire_xy_blend = float(
            np.clip(
                self.touchdown_reacquire_xy_blend if front_touchdown_xy_blend is None else front_touchdown_xy_blend,
                0.0,
                1.0,
            )
        )
        self.rear_touchdown_reacquire_xy_blend = float(
            np.clip(
                self.touchdown_reacquire_xy_blend if rear_touchdown_xy_blend is None else rear_touchdown_xy_blend,
                0.0,
                1.0,
            )
        )
        self.touchdown_reacquire_extra_depth = max(float(touchdown_extra_depth), 0.0)
        self.front_touchdown_reacquire_extra_depth = max(
            float(
                self.touchdown_reacquire_extra_depth
                if front_touchdown_extra_depth is None
                else front_touchdown_extra_depth
            ),
            0.0,
        )
        self.rear_touchdown_reacquire_extra_depth = max(
            float(
                self.touchdown_reacquire_extra_depth
                if rear_touchdown_extra_depth is None
                else rear_touchdown_extra_depth
            ),
            0.0,
        )
        touchdown_forward_bias = params.get('touchdown_reacquire_forward_bias', 0.0)
        front_touchdown_forward_bias = params.get('front_touchdown_reacquire_forward_bias', None)
        rear_touchdown_forward_bias = params.get('rear_touchdown_reacquire_forward_bias', None)
        self.touchdown_reacquire_forward_bias = float(touchdown_forward_bias)
        self.front_touchdown_reacquire_forward_bias = float(
            self.touchdown_reacquire_forward_bias
            if front_touchdown_forward_bias is None
            else front_touchdown_forward_bias
        )
        self.rear_touchdown_reacquire_forward_bias = float(
            self.touchdown_reacquire_forward_bias
            if rear_touchdown_forward_bias is None
            else rear_touchdown_forward_bias
        )
        self.rear_touchdown_reacquire_force_until_contact = bool(
            params.get('rear_touchdown_reacquire_force_until_contact', False)
        )
        self.rear_touchdown_reacquire_min_swing_time_s = max(
            float(params.get('rear_touchdown_reacquire_min_swing_time_s', 0.0)),
            0.0,
        )
        self.front_touchdown_reacquire_hold_current_xy = bool(
            params.get('front_touchdown_reacquire_hold_current_xy', False)
        )
        self.rear_touchdown_reacquire_hold_current_xy = bool(
            params.get('rear_touchdown_reacquire_hold_current_xy', False)
        )
        self.rear_touchdown_reacquire_max_xy_shift = max(
            float(params.get('rear_touchdown_reacquire_max_xy_shift', 0.0)),
            0.0,
        )
        self.rear_touchdown_reacquire_min_phase = float(
            np.clip(params.get('rear_touchdown_reacquire_min_phase', 0.0), 0.0, 1.0)
        )
        self.rear_touchdown_reacquire_upward_vel_damping = max(
            float(params.get('rear_touchdown_reacquire_upward_vel_damping', 0.0)),
            0.0,
        )
        self.rear_touchdown_retry_descent_depth = max(
            float(params.get('rear_touchdown_retry_descent_depth', 0.0)),
            0.0,
        )
        self.rear_touchdown_retry_descent_kp = max(
            float(params.get('rear_touchdown_retry_descent_kp', 0.0)),
            0.0,
        )
        self.rear_touchdown_retry_descent_kd = max(
            float(params.get('rear_touchdown_retry_descent_kd', 0.0)),
            0.0,
        )
        self.rear_touchdown_contact_debounce_s = max(
            float(params.get('rear_touchdown_contact_debounce_s', 0.0)),
            0.0,
        )
        self.rear_touchdown_contact_min_phase = float(
            np.clip(params.get('rear_touchdown_contact_min_phase', 0.0), 0.0, 1.0)
        )
        rear_touchdown_contact_max_upward_vel = params.get('rear_touchdown_contact_max_upward_vel', None)
        self.rear_touchdown_contact_max_upward_vel = (
            np.inf
            if rear_touchdown_contact_max_upward_vel is None
            else float(rear_touchdown_contact_max_upward_vel)
        )
        self.rear_touchdown_contact_min_grf_z = max(
            float(params.get('rear_touchdown_contact_min_grf_z', 0.0)),
            0.0,
        )
        self.rear_touchdown_close_lock_hold_s = max(
            float(params.get('rear_touchdown_close_lock_hold_s', 0.0)),
            0.0,
        )
        self.rear_touchdown_reacquire_retire_stance_hold_s = max(
            float(params.get('rear_touchdown_reacquire_retire_stance_hold_s', 0.0)),
            0.0,
        )
        self.stance_anchor_update_alpha = float(
            np.clip(params.get('stance_anchor_update_alpha', 0.0), 0.0, 1.0)
        )
        front_anchor_update = params.get('front_stance_anchor_update_alpha', None)
        rear_anchor_update = params.get('rear_stance_anchor_update_alpha', None)
        self.front_stance_anchor_update_alpha = float(
            np.clip(
                self.stance_anchor_update_alpha if front_anchor_update is None else front_anchor_update,
                0.0,
                1.0,
            )
        )
        self.rear_stance_anchor_update_alpha = float(
            np.clip(
                self.stance_anchor_update_alpha if rear_anchor_update is None else rear_anchor_update,
                0.0,
                1.0,
            )
        )
        self.touchdown_support_anchor_update_alpha = float(
            np.clip(params.get('touchdown_support_anchor_update_alpha', 0.0), 0.0, 1.0)
        )
        front_touchdown_support_anchor_update = params.get('front_touchdown_support_anchor_update_alpha', None)
        rear_touchdown_support_anchor_update = params.get('rear_touchdown_support_anchor_update_alpha', None)
        self.front_touchdown_support_anchor_update_alpha = float(
            np.clip(
                self.touchdown_support_anchor_update_alpha
                if front_touchdown_support_anchor_update is None
                else front_touchdown_support_anchor_update,
                0.0,
                1.0,
            )
        )
        self.rear_touchdown_support_anchor_update_alpha = float(
            np.clip(
                self.touchdown_support_anchor_update_alpha
                if rear_touchdown_support_anchor_update is None
                else rear_touchdown_support_anchor_update,
                0.0,
                1.0,
            )
        )
        self.touchdown_confirm_hold_s = max(float(params.get('touchdown_confirm_hold_s', 0.0)), 0.0)
        front_touchdown_confirm_hold = params.get('front_touchdown_confirm_hold_s', None)
        rear_touchdown_confirm_hold = params.get('rear_touchdown_confirm_hold_s', None)
        self.front_touchdown_confirm_hold_s = max(
            float(self.touchdown_confirm_hold_s if front_touchdown_confirm_hold is None else front_touchdown_confirm_hold),
            0.0,
        )
        self.rear_touchdown_confirm_hold_s = max(
            float(self.touchdown_confirm_hold_s if rear_touchdown_confirm_hold is None else rear_touchdown_confirm_hold),
            0.0,
        )
        self.rear_touchdown_confirm_keep_swing = bool(
            params.get('rear_touchdown_confirm_keep_swing', False)
        )
        self.touchdown_confirm_forward_scale = float(
            np.clip(params.get('touchdown_confirm_forward_scale', 1.0), 0.0, 1.0)
        )
        self.touchdown_settle_hold_s = max(float(params.get('touchdown_settle_hold_s', 0.0)), 0.0)
        front_touchdown_settle_hold = params.get('front_touchdown_settle_hold_s', None)
        rear_touchdown_settle_hold = params.get('rear_touchdown_settle_hold_s', None)
        self.front_touchdown_settle_hold_s = max(
            float(self.touchdown_settle_hold_s if front_touchdown_settle_hold is None else front_touchdown_settle_hold),
            0.0,
        )
        self.rear_touchdown_settle_hold_s = max(
            float(self.touchdown_settle_hold_s if rear_touchdown_settle_hold is None else rear_touchdown_settle_hold),
            0.0,
        )
        self.touchdown_settle_forward_scale = float(
            np.clip(params.get('touchdown_settle_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_post_touchdown_support_hold_s = max(
            float(params.get('rear_post_touchdown_support_hold_s', 0.0)),
            0.0,
        )
        self.rear_post_touchdown_support_forward_scale = float(
            np.clip(params.get('rear_post_touchdown_support_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_post_touchdown_support_height_ratio = max(
            float(params.get('rear_post_touchdown_support_height_ratio', 0.0)),
            0.0,
        )
        post_roll = params.get('rear_post_touchdown_support_roll_threshold', None)
        post_pitch = params.get('rear_post_touchdown_support_pitch_threshold', None)
        self.rear_post_touchdown_support_roll_threshold = (
            np.inf if post_roll is None else max(float(post_roll), 0.0)
        )
        self.rear_post_touchdown_support_pitch_threshold = (
            np.inf if post_pitch is None else max(float(post_pitch), 0.0)
        )
        self.rear_post_touchdown_support_min_grf_z = max(
            float(params.get('rear_post_touchdown_support_min_grf_z', 0.0)),
            0.0,
        )
        self.rear_post_touchdown_support_min_rear_load_share = max(
            float(params.get('rear_post_touchdown_support_min_rear_load_share', 0.0)),
            0.0,
        )
        self.front_rear_transition_guard_hold_s = max(
            float(params.get('front_rear_transition_guard_hold_s', 0.0)),
            0.0,
        )
        self.front_rear_transition_guard_forward_scale = float(
            np.clip(params.get('front_rear_transition_guard_forward_scale', 1.0), 0.0, 1.0)
        )
        front_rear_guard_roll = params.get('front_rear_transition_guard_roll_threshold', None)
        front_rear_guard_pitch = params.get('front_rear_transition_guard_pitch_threshold', None)
        self.front_rear_transition_guard_roll_threshold = (
            np.inf if front_rear_guard_roll is None else max(float(front_rear_guard_roll), 0.0)
        )
        self.front_rear_transition_guard_pitch_threshold = (
            np.inf if front_rear_guard_pitch is None else max(float(front_rear_guard_pitch), 0.0)
        )
        self.front_rear_transition_guard_height_ratio = max(
            float(params.get('front_rear_transition_guard_height_ratio', 0.0)),
            0.0,
        )
        self.front_rear_transition_guard_release_tail_s = max(
            float(params.get('front_rear_transition_guard_release_tail_s', 0.0)),
            0.0,
        )
        self.front_rear_transition_guard_margin_release = max(
            float(params.get('front_rear_transition_guard_margin_release', 0.0)),
            0.0,
        )
        self.front_rear_transition_guard_post_recovery_hold_s = max(
            float(params.get('front_rear_transition_guard_post_recovery_hold_s', 0.0)),
            0.0,
        )
        self.touchdown_contact_vel_z_damping = max(float(params.get('touchdown_contact_vel_z_damping', 0.0)), 0.0)
        front_touchdown_contact_vel_z_damping = params.get('front_touchdown_contact_vel_z_damping', None)
        rear_touchdown_contact_vel_z_damping = params.get('rear_touchdown_contact_vel_z_damping', None)
        self.front_touchdown_contact_vel_z_damping = max(
            float(self.touchdown_contact_vel_z_damping if front_touchdown_contact_vel_z_damping is None else front_touchdown_contact_vel_z_damping),
            0.0,
        )
        self.rear_touchdown_contact_vel_z_damping = max(
            float(self.touchdown_contact_vel_z_damping if rear_touchdown_contact_vel_z_damping is None else rear_touchdown_contact_vel_z_damping),
            0.0,
        )
        self.front_margin_rescue_hold_s = max(float(params.get('front_margin_rescue_hold_s', 0.0)), 0.0)
        self.front_margin_rescue_forward_scale = float(
            np.clip(params.get('front_margin_rescue_forward_scale', 1.0), 0.0, 1.0)
        )
        self.front_margin_rescue_min_margin = float(params.get('front_margin_rescue_min_margin', 0.0))
        self.front_margin_rescue_margin_gap = max(
            float(params.get('front_margin_rescue_margin_gap', 0.0)),
            0.0,
        )
        self.front_margin_rescue_alpha_margin = max(
            float(params.get('front_margin_rescue_alpha_margin', 0.02)),
            1e-6,
        )
        rescue_roll = params.get('front_margin_rescue_roll_threshold', None)
        rescue_pitch = params.get('front_margin_rescue_pitch_threshold', None)
        self.front_margin_rescue_roll_threshold = (
            np.inf if rescue_roll is None else max(float(rescue_roll), 0.0)
        )
        self.front_margin_rescue_pitch_threshold = (
            np.inf if rescue_pitch is None else max(float(rescue_pitch), 0.0)
        )
        self.front_margin_rescue_height_ratio = max(float(params.get('front_margin_rescue_height_ratio', 0.0)), 0.0)
        self.front_margin_rescue_recent_swing_window_s = max(
            float(params.get('front_margin_rescue_recent_swing_window_s', 0.0)),
            0.0,
        )
        self.front_margin_rescue_require_all_contact = bool(
            params.get('front_margin_rescue_require_all_contact', True)
        )
        self.rear_handoff_support_hold_s = max(float(params.get('rear_handoff_support_hold_s', 0.0)), 0.0)
        self.rear_handoff_forward_scale = float(
            np.clip(params.get('rear_handoff_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_handoff_lookahead_steps = max(int(params.get('rear_handoff_lookahead_steps', 1)), 1)
        self.rear_handoff_support_rear_alpha_scale = float(
            np.clip(params.get('rear_handoff_support_rear_alpha_scale', 0.0), 0.0, 1.0)
        )
        self.rear_swing_bridge_hold_s = max(float(params.get('rear_swing_bridge_hold_s', 0.0)), 0.0)
        self.rear_swing_bridge_forward_scale = float(
            np.clip(params.get('rear_swing_bridge_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_swing_bridge_rear_alpha_scale = float(
            np.clip(params.get('rear_swing_bridge_rear_alpha_scale', 0.0), 0.0, 1.0)
        )
        self.rear_swing_release_support_hold_s = max(
            float(params.get('rear_swing_release_support_hold_s', 0.0)),
            0.0,
        )
        self.rear_swing_release_forward_scale = float(
            np.clip(params.get('rear_swing_release_forward_scale', 1.0), 0.0, 1.0)
        )
        bridge_roll = params.get('rear_swing_bridge_roll_threshold', None)
        bridge_pitch = params.get('rear_swing_bridge_pitch_threshold', None)
        self.rear_swing_bridge_roll_threshold = (
            np.inf if bridge_roll is None else max(float(bridge_roll), 0.0)
        )
        self.rear_swing_bridge_pitch_threshold = (
            np.inf if bridge_pitch is None else max(float(bridge_pitch), 0.0)
        )
        self.rear_swing_bridge_height_ratio = max(float(params.get('rear_swing_bridge_height_ratio', 0.0)), 0.0)
        self.rear_swing_bridge_recent_front_window_s = max(
            float(params.get('rear_swing_bridge_recent_front_window_s', 0.0)),
            0.0,
        )
        self.rear_swing_bridge_lookahead_steps = max(int(params.get('rear_swing_bridge_lookahead_steps', 1)), 1)
        self.rear_swing_bridge_allcontact_release_tail_s = max(
            float(params.get('rear_swing_bridge_allcontact_release_tail_s', 0.0)),
            0.0,
        )
        rear_pre_swing_guard_roll = params.get('rear_pre_swing_guard_roll_threshold', None)
        rear_pre_swing_guard_pitch = params.get('rear_pre_swing_guard_pitch_threshold', None)
        self.rear_pre_swing_guard_roll_threshold = (
            np.inf if rear_pre_swing_guard_roll is None else max(float(rear_pre_swing_guard_roll), 0.0)
        )
        self.rear_pre_swing_guard_pitch_threshold = (
            np.inf if rear_pre_swing_guard_pitch is None else max(float(rear_pre_swing_guard_pitch), 0.0)
        )
        self.rear_pre_swing_guard_height_ratio = max(
            float(params.get('rear_pre_swing_guard_height_ratio', 0.0)),
            0.0,
        )
        self.full_contact_recovery_hold_s = max(float(params.get('full_contact_recovery_hold_s', 0.0)), 0.0)
        self.full_contact_recovery_forward_scale = float(
            np.clip(params.get('full_contact_recovery_forward_scale', 1.0), 0.0, 1.0)
        )
        recovery_roll = params.get('full_contact_recovery_roll_threshold', None)
        recovery_pitch = params.get('full_contact_recovery_pitch_threshold', None)
        self.full_contact_recovery_roll_threshold = (
            np.inf if recovery_roll is None else max(float(recovery_roll), 0.0)
        )
        self.full_contact_recovery_pitch_threshold = (
            np.inf if recovery_pitch is None else max(float(recovery_pitch), 0.0)
        )
        self.full_contact_recovery_height_ratio = max(
            float(params.get('full_contact_recovery_height_ratio', 0.0)),
            0.0,
        )
        self.full_contact_recovery_recent_window_s = max(
            float(params.get('full_contact_recovery_recent_window_s', 0.0)),
            0.0,
        )
        self.full_contact_recovery_rear_support_scale = float(
            np.clip(params.get('full_contact_recovery_rear_support_scale', 0.0), 0.0, 1.0)
        )
        self.full_contact_recovery_allcontact_release_tail_s = max(
            float(params.get('full_contact_recovery_allcontact_release_tail_s', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_hold_s = max(
            float(params.get('rear_all_contact_stabilization_hold_s', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_forward_scale = float(
            np.clip(params.get('rear_all_contact_stabilization_forward_scale', 1.0), 0.0, 1.0)
        )
        self.rear_all_contact_stabilization_front_alpha_scale = float(
            np.clip(params.get('rear_all_contact_stabilization_front_alpha_scale', 1.0), 0.0, 1.0)
        )
        self.rear_all_contact_stabilization_height_ratio = max(
            float(params.get('rear_all_contact_stabilization_height_ratio', 0.0)),
            0.0,
        )
        rear_all_contact_roll = params.get('rear_all_contact_stabilization_roll_threshold', None)
        rear_all_contact_pitch = params.get('rear_all_contact_stabilization_pitch_threshold', None)
        rear_all_contact_preclose_pitch = params.get('rear_all_contact_stabilization_preclose_pitch_threshold', None)
        rear_all_contact_preclose_vz = params.get('rear_all_contact_stabilization_preclose_vz_threshold', None)
        self.rear_all_contact_stabilization_roll_threshold = (
            np.inf if rear_all_contact_roll is None else max(float(rear_all_contact_roll), 0.0)
        )
        self.rear_all_contact_stabilization_pitch_threshold = (
            np.inf if rear_all_contact_pitch is None else max(float(rear_all_contact_pitch), 0.0)
        )
        self.rear_all_contact_stabilization_preclose_pitch_threshold = (
            np.inf
            if rear_all_contact_preclose_pitch is None
            else max(float(rear_all_contact_preclose_pitch), 0.0)
        )
        self.rear_all_contact_stabilization_preclose_vz_threshold = (
            -np.inf
            if rear_all_contact_preclose_vz is None
            else float(rear_all_contact_preclose_vz)
        )
        self.rear_late_seam_support_trigger_s = max(
            float(params.get('rear_late_seam_support_trigger_s', 0.0)),
            0.0,
        )
        self.rear_close_handoff_hold_s = max(
            float(params.get('rear_close_handoff_hold_s', 0.0)),
            0.0,
        )
        self.rear_late_load_share_support_hold_s = max(
            float(params.get('rear_late_load_share_support_hold_s', 0.0)),
            0.0,
        )
        self.rear_late_load_share_support_min_leg_share = max(
            float(params.get('rear_late_load_share_support_min_leg_share', 0.0)),
            0.0,
        )
        self.rear_late_load_share_support_height_ratio = max(
            float(params.get('rear_late_load_share_support_height_ratio', 0.0)),
            0.0,
        )
        self.rear_late_load_share_support_min_persist_s = max(
            float(params.get('rear_late_load_share_support_min_persist_s', 0.0)),
            0.0,
        )
        self.rear_late_load_share_support_alpha_cap = float(
            np.clip(
                float(params.get('rear_late_load_share_support_alpha_cap', 1.0)),
                0.0,
                1.0,
            )
        )
        self.rear_all_contact_stabilization_min_rear_load_share = max(
            float(params.get('rear_all_contact_stabilization_min_rear_load_share', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_min_rear_leg_load_share = max(
            float(params.get('rear_all_contact_stabilization_min_rear_leg_load_share', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_weak_leg_share_ref = max(
            float(params.get('rear_all_contact_stabilization_weak_leg_share_ref', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_weak_leg_height_ratio = max(
            float(params.get('rear_all_contact_stabilization_weak_leg_height_ratio', 0.0)),
            0.0,
        )
        self.rear_all_contact_stabilization_weak_leg_tail_only = bool(
            params.get('rear_all_contact_stabilization_weak_leg_tail_only', False)
        )
        self.rear_all_contact_stabilization_retrigger_limit = max(
            int(params.get('rear_all_contact_stabilization_retrigger_limit', 0)),
            0,
        )
        self.rear_all_contact_post_recovery_tail_hold_s = max(
            float(params.get('rear_all_contact_post_recovery_tail_hold_s', 0.0)),
            0.0,
        )
        self.rear_all_contact_release_tail_alpha_scale = float(
            np.clip(
                params.get('rear_all_contact_release_tail_alpha_scale', 1.0),
                0.0,
                1.0,
            )
        )
        self.rear_all_contact_post_recovery_front_late_alpha_scale = float(
            np.clip(
                params.get('rear_all_contact_post_recovery_front_late_alpha_scale', 1.0),
                0.0,
                1.0,
            )
        )
        self.pre_swing_gate_hold_s = max(float(params.get('pre_swing_gate_hold_s', 0.0)), 0.0)
        rear_pre_swing_gate_hold_s = params.get('rear_pre_swing_gate_hold_s', None)
        self.rear_pre_swing_gate_hold_s = max(
            float(self.pre_swing_gate_hold_s if rear_pre_swing_gate_hold_s is None else rear_pre_swing_gate_hold_s),
            0.0,
        )

        # Dynamic two-leg-support gaits need faster swing opening and should
        # not inherit the conservative crawl-style latch/gating defaults.
        if getattr(self, 'is_dynamic_gait', False):
            self.contact_latch_steps = 0
            self.contact_latch_budget_s = 0.0
            self.front_contact_latch_steps = 0
            self.front_contact_latch_budget_s = 0.0
            self.rear_contact_latch_steps = 0
            self.rear_contact_latch_budget_s = 0.0
            self.virtual_unlatch_hold_s = 0.0
            self.pre_swing_gate_min_margin = 0.0
            self.front_pre_swing_gate_min_margin = 0.0
            self.rear_pre_swing_gate_min_margin = 0.0
            self.swing_contact_release_timeout_s = 0.0
            self.front_swing_contact_release_timeout_s = 0.0
            self.rear_swing_contact_release_timeout_s = 0.0
            self.support_contact_confirm_hold_s = 0.0
            self.front_support_contact_confirm_hold_s = 0.0
            self.rear_support_contact_confirm_hold_s = 0.0
            self.pre_swing_gate_hold_s = 0.0
            self.rear_pre_swing_gate_hold_s = 0.0
            self.front_rear_transition_guard_hold_s = 0.0
            self.front_rear_transition_guard_forward_scale = 1.0
            self.rear_swing_release_support_hold_s = 0.0
            self.rear_swing_release_forward_scale = 1.0
        if cfg.mpc_params['type'] != 'linear_osqp':
            # Keep stock controllers on the stock path: the crawl-specific
            # rear transition / recovery heuristics in this branch are meant
            # only for the custom linear_osqp controller.
            self.front_margin_rescue_hold_s = 0.0
            self.front_stance_dropout_support_hold_s = 0.0
            self.rear_handoff_support_hold_s = 0.0
            self.rear_swing_bridge_hold_s = 0.0
            self.rear_swing_release_support_hold_s = 0.0
            self.full_contact_recovery_hold_s = 0.0
            self.full_contact_recovery_recent_window_s = 0.0
            self.rear_touchdown_reacquire_hold_s = 0.0
            self.rear_touchdown_confirm_hold_s = 0.0
            self.rear_touchdown_settle_hold_s = 0.0
            self.rear_post_touchdown_support_hold_s = 0.0
            self.rear_all_contact_stabilization_hold_s = 0.0
            self.front_rear_transition_guard_hold_s = 0.0
        self._configure_rear_transition_manager()
        self._sync_rear_transition_debug_arrays()

