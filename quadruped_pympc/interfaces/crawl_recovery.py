"""Crawl-gait full-contact recovery logic, extracted from WBInterface."""
from __future__ import annotations

import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc import config as cfg


class CrawlRecoveryMixin:
    """Mixin supplying _process_crawl_recovery to WBInterface."""

    def _process_crawl_recovery(
        self,
        actual_contact,
        transition_contact,
        base_ori_euler_xyz: np.ndarray,
        base_pos_measured: np.ndarray,
        contact_sequence: np.ndarray,
        feet_pos: LegsAttr,
        com_pos_measured: np.ndarray,
        base_lin_vel: np.ndarray,
        simulation_dt: float,
        startup_full_stance_active: bool,
        prev_full_contact_recovery_active: bool,
        gate_forward_scale: float,
    ) -> tuple[float, np.ndarray, np.ndarray, bool, float, float, float, bool, bool, bool, np.ndarray]:
        actual_contact_array = np.asarray(actual_contact, dtype=int)
        transition_contact_array = np.asarray(transition_contact, dtype=int)
        previous_actual_contact_array = np.asarray(self.previous_actual_contact, dtype=int)
        front_support_margins = np.full(2, np.inf, dtype=float)
        front_late_posture_tail_candidate = False
        front_planted_posture_tail_candidate = False
        front_planted_posture_tail_trigger = False
        front_close_gap_keep_swing_mask = np.zeros(2, dtype=bool)
        all_contact_now = bool(np.all(transition_contact_array == 1))
        roll_mag = abs(float(base_ori_euler_xyz[0]))
        pitch_mag = abs(float(base_ori_euler_xyz[1]))
        ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
        height_ratio = float(base_pos_measured[2]) / ref_height

        recovery_hold_s = float(self.full_contact_recovery_hold_s)
        if recovery_hold_s <= 1e-9:
            self.full_contact_recovery_remaining_s = 0.0
            return (
                gate_forward_scale,
                actual_contact_array,
                front_support_margins,
                all_contact_now,
                roll_mag,
                pitch_mag,
                height_ratio,
                front_late_posture_tail_candidate,
                front_planted_posture_tail_candidate,
                front_planted_posture_tail_trigger,
                front_close_gap_keep_swing_mask,
            )

        # For late crawl seams, the foot-contact signal often returns one or
        # two control ticks before the stricter MuJoCo geometry contact closes
        # the current-contact state. Let full-contact recovery key off the
        # earlier transition contact so the all-contact stabilization window
        # can start as soon as the leg has physically re-touched.
        all_contact_prev = bool(np.all(previous_actual_contact_array == 1))
        recent_gate_ok = (
            float(self.full_contact_recovery_recent_window_s) <= 1e-9
            or float(self.front_touchdown_support_recent_remaining_s) > 1e-9
        )
        front_close_gap_reclose_now = bool(
            self.gait_name == 'crawl'
            and all_contact_now
            and (not all_contact_prev)
            and np.any(
                (np.asarray(self.planned_contact[0:2], dtype=int) == 1)
                & (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                & (previous_actual_contact_array[0:2] == 0)
                & (actual_contact_array[0:2] == 1)
            )
            and np.all(np.asarray(self.planned_contact[2:4], dtype=int) == 1)
            and np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
            and np.all(actual_contact_array[2:4] == 1)
        )
        recovery_posture_needed = bool(
            roll_mag >= float(self.full_contact_recovery_roll_threshold)
            or pitch_mag >= float(self.full_contact_recovery_pitch_threshold)
            or (
                float(self.full_contact_recovery_height_ratio) > 1e-9
                and height_ratio <= float(self.full_contact_recovery_height_ratio)
            )
        )
        recovery_trigger = (
            (not startup_full_stance_active)
            and recent_gate_ok
            and all_contact_now
            and (not all_contact_prev)
            and recovery_posture_needed
        )
        self.full_contact_recovery_trigger_debug = int(bool(recovery_trigger))
        if recovery_trigger:
            self.full_contact_recovery_remaining_s = max(
                float(self.full_contact_recovery_remaining_s),
                recovery_hold_s,
            )
        delayed_front_swing_recovery_hold_s = float(
            self.crawl_params.front_delayed_swing_recovery_hold_s
        )
        front_planned_swing = np.asarray(contact_sequence[0:2, 0], dtype=int) == 0
        delayed_front_margin_threshold = float(
            self.crawl_params.front_delayed_swing_recovery_margin_threshold
        )
        if delayed_front_swing_recovery_hold_s > 1e-9:
            delayed_front_recovery_once_per_swing = bool(
                self.crawl_params.front_delayed_swing_recovery_once_per_swing
            )
            if delayed_front_recovery_once_per_swing:
                self.front_delayed_swing_recovery_spent[np.logical_not(front_planned_swing)] = 0
            front_realized_stance = (
                (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                & (actual_contact_array[0:2] == 1)
            )
            front_support_margins = np.asarray(
                [
                    self._pre_swing_gate_margin(
                        leg_id,
                        actual_contact,
                        feet_pos,
                        com_pos_measured,
                        com_vel_xy=base_lin_vel[0:2],
                    )
                    for leg_id in range(2)
                ],
                dtype=float,
            )
            delayed_front_release_tail_s = float(
                self.crawl_params.front_delayed_swing_recovery_release_tail_s
            )
            delayed_front_rearm_trigger_s = float(
                self.crawl_params.front_delayed_swing_recovery_rearm_trigger_s
            )
            eligible_front_delayed_swing = (
                front_planned_swing
                & front_realized_stance
                & (front_support_margins <= delayed_front_margin_threshold)
            )
            if delayed_front_recovery_once_per_swing:
                eligible_front_delayed_swing &= (
                    np.asarray(self.front_delayed_swing_recovery_spent, dtype=int) == 0
                )
            delayed_front_recovery_near_expiry = bool(
                (prev_full_contact_recovery_active or float(self.full_contact_recovery_remaining_s) > 1e-9)
                and (
                    delayed_front_rearm_trigger_s <= 1e-9
                    or float(self.full_contact_recovery_remaining_s)
                    <= delayed_front_rearm_trigger_s + 1e-12
                )
            )
            front_delayed_swing_recovery = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and all_contact_now
                and recovery_posture_needed
                and delayed_front_recovery_near_expiry
                and np.any(eligible_front_delayed_swing)
            )
            self.front_delayed_swing_recovery_trigger_debug = int(bool(front_delayed_swing_recovery))
            if front_delayed_swing_recovery:
                self.full_contact_recovery_remaining_s = max(
                    float(self.full_contact_recovery_remaining_s),
                    delayed_front_swing_recovery_hold_s,
                )
                if delayed_front_recovery_once_per_swing:
                    self.front_delayed_swing_recovery_spent[eligible_front_delayed_swing] = 1
            if delayed_front_release_tail_s > 1e-9:
                releasable_front_delayed_swing = (
                    front_planned_swing
                    & front_realized_stance
                    & (front_support_margins > delayed_front_margin_threshold)
                )
                if np.any(releasable_front_delayed_swing):
                    self.full_contact_recovery_remaining_s = min(
                        float(self.full_contact_recovery_remaining_s),
                        delayed_front_release_tail_s,
                    )
        planted_front_recovery_hold_s = float(
            self.crawl_params.front_planted_swing_recovery_hold_s
        )
        if planted_front_recovery_hold_s > 1e-9:
            self.front_planted_swing_recovery_spent[np.logical_not(front_planned_swing)] = 0
            front_planted_late_leg = (
                front_planned_swing
                & (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                & (actual_contact_array[0:2] == 1)
            )
            front_planted_recovery_leg = np.asarray(front_planted_late_leg, dtype=bool).copy()
            front_planted_recovery_leg &= (
                np.asarray(self.front_planted_swing_recovery_spent, dtype=int) == 0
            )
            rear_contacts_stable = bool(
                np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                and np.all(actual_contact_array[2:4] == 1)
            )
            planted_front_height_ratio = float(
                self.crawl_params.front_planted_swing_recovery_height_ratio
            )
            planted_front_roll_threshold = self.crawl_params.front_planted_swing_recovery_roll_threshold
            planted_front_posture_bad = bool(
                (
                    planted_front_height_ratio > 1e-9
                    and float(height_ratio) <= float(planted_front_height_ratio)
                )
                or (
                    planted_front_roll_threshold is not None
                    and np.isfinite(float(planted_front_roll_threshold))
                    and float(roll_mag) >= float(planted_front_roll_threshold)
                )
            )
            planted_front_recovery_rearm_trigger_s = float(
                self.crawl_params.front_planted_swing_recovery_rearm_trigger_s
            )
            planted_front_recovery_near_expiry = bool(
                (prev_full_contact_recovery_active or float(self.full_contact_recovery_remaining_s) > 1e-9)
                and (
                    planted_front_recovery_rearm_trigger_s <= 1e-9
                    or float(self.full_contact_recovery_remaining_s)
                    <= planted_front_recovery_rearm_trigger_s + 1e-12
                )
            )
            planted_front_recovery_trigger = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and np.count_nonzero(front_planned_swing) == 1
                and rear_contacts_stable
                and planted_front_posture_bad
                and planted_front_recovery_near_expiry
                and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                and np.any(
                    front_planted_recovery_leg
                    & (
                        front_support_margins
                        <= float(self.crawl_params.front_planted_swing_recovery_margin_threshold)
                    )
                )
            )
            self.planted_front_recovery_trigger_debug = int(bool(planted_front_recovery_trigger))
            front_planted_posture_tail_candidate = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and np.count_nonzero(front_planned_swing) == 1
                and np.any(front_planted_late_leg)
                and rear_contacts_stable
                and planted_front_posture_bad
                and planted_front_recovery_near_expiry
                and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                and np.any(
                    front_planted_late_leg
                    & (
                        front_support_margins
                        <= float(self.crawl_params.front_planted_swing_recovery_margin_threshold)
                    )
                )
            )
            if planted_front_recovery_trigger:
                self.full_contact_recovery_remaining_s = max(
                    float(self.full_contact_recovery_remaining_s),
                    planted_front_recovery_hold_s,
                )
                self.front_planted_swing_recovery_spent[front_planned_swing] = 1
            planted_front_postdrop_recovery_hold_s = float(
                self.crawl_params.front_planted_postdrop_recovery_hold_s
            )
            planted_front_postdrop_recovery_trigger = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and planted_front_postdrop_recovery_hold_s > 1e-9
                and prev_full_contact_recovery_active
                and int(self.full_contact_recovery_active) == 0
                and np.count_nonzero(front_planned_swing) == 1
                and np.any(front_planted_late_leg)
                and rear_contacts_stable
                and planted_front_posture_bad
                and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                and np.any(
                    front_planted_late_leg
                    & (
                        front_support_margins
                        <= float(self.crawl_params.front_planted_swing_recovery_margin_threshold)
                    )
                )
            )
            self.planted_front_postdrop_recovery_trigger_debug = int(
                bool(planted_front_postdrop_recovery_trigger)
            )
            if planted_front_postdrop_recovery_trigger:
                self.full_contact_recovery_remaining_s = max(
                    float(self.full_contact_recovery_remaining_s),
                    planted_front_postdrop_recovery_hold_s,
                )
        else:
            front_planted_posture_tail_candidate = False
        front_stance_support_tail_hold_s = float(
            self.crawl_params.front_stance_support_tail_hold_s
        )
        if front_stance_support_tail_hold_s > 1e-9:
            front_current_swing_state = (
                front_planned_swing
                & (np.asarray(self.current_contact[0:2], dtype=int) == 0)
            )
            front_actual_swing_opened = (
                (previous_actual_contact_array[0:2] == 1)
                & (actual_contact_array[0:2] == 0)
            )
            front_stance_support_tail_trigger = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and recovery_posture_needed
                and (
                    prev_full_contact_recovery_active
                    or float(self.full_contact_recovery_remaining_s) > 1e-9
                )
                and np.any(front_planned_swing & front_actual_swing_opened)
            )
            if front_stance_support_tail_trigger:
                self.crawl_state.front_stance_support_tail_remaining_s = max(
                    float(self.crawl_state.front_stance_support_tail_remaining_s),
                    front_stance_support_tail_hold_s,
                )
        front_close_gap_support_hold_s = float(
            self.crawl_params.front_close_gap_support_hold_s
        )
        if front_close_gap_support_hold_s > 1e-9:
            front_close_gap_state = (
                (np.asarray(contact_sequence[0:2, 0], dtype=int) == 1)
                & (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                & (actual_contact_array[0:2] == 0)
            )
            front_close_gap_trigger = bool(
                (not startup_full_stance_active)
                and self.gait_name == 'crawl'
                and recovery_posture_needed
                and np.any(front_close_gap_state)
                and bool(np.all(actual_contact_array[2:4] == 1))
                and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
            )
            self.front_close_gap_trigger_debug = int(bool(front_close_gap_trigger))
            if front_close_gap_trigger:
                self.full_contact_recovery_remaining_s = max(
                    float(self.full_contact_recovery_remaining_s),
                    front_close_gap_support_hold_s,
                )
                self.crawl_state.front_stance_support_tail_remaining_s = max(
                    float(self.crawl_state.front_stance_support_tail_remaining_s),
                    front_close_gap_support_hold_s,
                )
                if bool(self.crawl_params.front_close_gap_keep_swing):
                    front_close_gap_keep_swing_mask = np.asarray(front_close_gap_state, dtype=bool).copy()
        front_late_posture_tail_candidate = False
        front_late_rearm_tail_hold_s = float(
            self.crawl_params.front_late_rearm_tail_hold_s
        )
        if front_late_rearm_tail_hold_s > 1e-9:
            self.front_late_rearm_used_s[np.logical_not(front_planned_swing)] = 0.0
            front_current_swing_state = (
                front_planned_swing
                & (np.asarray(self.current_contact[0:2], dtype=int) == 0)
            )
            if np.count_nonzero(front_current_swing_state) == 1:
                front_swing_leg_id = int(np.flatnonzero(front_current_swing_state)[0])
                front_swing_time = float(self.stc.swing_time[front_swing_leg_id])
                front_late_rearm_budget_s = max(
                    float(self.crawl_params.front_late_rearm_budget_s),
                    front_late_rearm_tail_hold_s,
                )
                front_late_rearm_remaining_budget_s = max(
                    0.0,
                    float(front_late_rearm_budget_s)
                    - float(self.front_late_rearm_used_s[front_swing_leg_id]),
                )
                front_late_rearm_chunk_s = min(
                    float(front_late_rearm_tail_hold_s),
                    float(front_late_rearm_remaining_budget_s),
                )
                front_late_rearm_trigger = bool(
                    (not startup_full_stance_active)
                    and self.gait_name == 'crawl'
                    and recovery_posture_needed
                    and bool(np.all(actual_contact_array[2:4] == 1))
                    and float(self.crawl_state.front_stance_support_tail_remaining_s) <= 1e-9
                    and float(self.full_contact_recovery_remaining_s) <= 1e-9
                    and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                    and front_swing_time
                    >= float(self.crawl_params.front_late_rearm_min_swing_time_s)
                    and float(front_support_margins[front_swing_leg_id])
                    <= -float(self.crawl_params.front_late_rearm_min_negative_margin)
                    and float(front_late_rearm_chunk_s) > 1e-9
                )
                self.front_late_rearm_trigger_debug = int(bool(front_late_rearm_trigger))
                front_late_posture_tail_candidate = bool(front_late_rearm_trigger)
                if front_late_rearm_trigger:
                    self.crawl_state.front_stance_support_tail_remaining_s = max(
                        float(self.crawl_state.front_stance_support_tail_remaining_s),
                        float(front_late_rearm_chunk_s),
                    )
                    self.front_late_rearm_used_s[front_swing_leg_id] = min(
                        float(front_late_rearm_budget_s),
                        float(self.front_late_rearm_used_s[front_swing_leg_id])
                        + float(front_late_rearm_chunk_s),
                    )
        if self.full_contact_recovery_remaining_s > 1e-9:
            self.full_contact_recovery_active = 1
            self.full_contact_recovery_alpha = 1.0
            gate_forward_scale = min(gate_forward_scale, float(self.full_contact_recovery_forward_scale))
            self.full_contact_recovery_remaining_s = max(
                0.0, float(self.full_contact_recovery_remaining_s) - float(simulation_dt)
            )
        else:
            self.full_contact_recovery_remaining_s = 0.0

        return (
            gate_forward_scale,
            actual_contact_array,
            front_support_margins,
            all_contact_now,
            roll_mag,
            pitch_mag,
            height_ratio,
            front_late_posture_tail_candidate,
            front_planted_posture_tail_candidate,
            front_planted_posture_tail_trigger,
            front_close_gap_keep_swing_mask,
        )
