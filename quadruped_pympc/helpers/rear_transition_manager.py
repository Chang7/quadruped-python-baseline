from __future__ import annotations

import numpy as np


class RearTransitionManager:
    """Small rear-leg transition state helper.

    The goal is not to replace the whole low-level stack, but to keep the
    rear touchdown / recontact bookkeeping in one place instead of scattering
    it across ``WBInterface``.
    """

    def __init__(self, mpc_dt: float) -> None:
        self.mpc_dt = max(float(mpc_dt), 1e-6)
        self.enabled = False
        self.contact_debounce_s = 0.0
        self.contact_min_phase = 0.0
        self.contact_max_upward_vel = np.inf
        self.contact_min_grf_z = 0.0
        self.reacquire_hold_s = 0.0
        self.reacquire_min_swing_time_s = 0.0
        self.reacquire_forward_scale = 1.0
        self.confirm_hold_s = 0.0
        self.confirm_forward_scale = 1.0
        self.settle_hold_s = 0.0
        self.settle_forward_scale = 1.0
        self.post_support_hold_s = 0.0
        self.post_support_forward_scale = 1.0
        self.post_support_height_ratio = 0.0
        self.post_support_roll_threshold = np.inf
        self.post_support_pitch_threshold = np.inf
        self.post_support_min_grf_z = 0.0
        self.post_support_min_rear_load_share = 0.0
        self.all_contact_stabilization_hold_s = 0.0
        self.all_contact_stabilization_forward_scale = 1.0
        self.all_contact_stabilization_front_alpha_scale = 1.0
        self.all_contact_stabilization_height_ratio = 0.0
        self.all_contact_stabilization_roll_threshold = np.inf
        self.all_contact_stabilization_pitch_threshold = np.inf
        self.all_contact_stabilization_min_rear_load_share = 0.0
        self.all_contact_stabilization_min_rear_leg_load_share = 0.0
        self.all_contact_stabilization_retrigger_limit = 0
        self.front_transition_guard_hold_s = 0.0
        self.front_transition_guard_forward_scale = 1.0
        self.front_transition_guard_roll_threshold = np.inf
        self.front_transition_guard_pitch_threshold = np.inf
        self.front_transition_guard_height_ratio = 0.0
        self.front_transition_guard_release_tail_s = 0.0
        self.pre_swing_guard_roll_threshold = np.inf
        self.pre_swing_guard_pitch_threshold = np.inf
        self.pre_swing_guard_height_ratio = 0.0
        self.confirm_keep_swing = False
        self.reset()

    def configure(
        self,
        *,
        enabled: bool,
        contact_debounce_s: float,
        contact_min_phase: float,
        contact_max_upward_vel: float,
        contact_min_grf_z: float,
        reacquire_hold_s: float,
        reacquire_min_swing_time_s: float,
        reacquire_forward_scale: float,
        confirm_hold_s: float,
        confirm_forward_scale: float,
        settle_hold_s: float,
        settle_forward_scale: float,
        post_support_hold_s: float,
        post_support_forward_scale: float,
        post_support_height_ratio: float,
        post_support_roll_threshold: float | None,
        post_support_pitch_threshold: float | None,
        post_support_min_grf_z: float,
        post_support_min_rear_load_share: float,
        all_contact_stabilization_hold_s: float,
        all_contact_stabilization_forward_scale: float,
        all_contact_stabilization_front_alpha_scale: float,
        all_contact_stabilization_height_ratio: float,
        all_contact_stabilization_roll_threshold: float | None,
        all_contact_stabilization_pitch_threshold: float | None,
        all_contact_stabilization_min_rear_load_share: float,
        all_contact_stabilization_min_rear_leg_load_share: float,
        all_contact_stabilization_retrigger_limit: int,
        front_transition_guard_hold_s: float,
        front_transition_guard_forward_scale: float,
        front_transition_guard_roll_threshold: float | None,
        front_transition_guard_pitch_threshold: float | None,
        front_transition_guard_height_ratio: float,
        front_transition_guard_release_tail_s: float,
        pre_swing_guard_roll_threshold: float | None,
        pre_swing_guard_pitch_threshold: float | None,
        pre_swing_guard_height_ratio: float,
        confirm_keep_swing: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.contact_debounce_s = max(float(contact_debounce_s), 0.0)
        self.contact_min_phase = float(np.clip(contact_min_phase, 0.0, 1.0))
        self.contact_max_upward_vel = (
            np.inf if contact_max_upward_vel is None else float(contact_max_upward_vel)
        )
        self.contact_min_grf_z = max(float(contact_min_grf_z), 0.0)
        self.reacquire_hold_s = max(float(reacquire_hold_s), 0.0)
        self.reacquire_min_swing_time_s = max(float(reacquire_min_swing_time_s), 0.0)
        self.reacquire_forward_scale = float(np.clip(reacquire_forward_scale, 0.0, 1.0))
        self.confirm_hold_s = max(float(confirm_hold_s), 0.0)
        self.confirm_forward_scale = float(np.clip(confirm_forward_scale, 0.0, 1.0))
        self.settle_hold_s = max(float(settle_hold_s), 0.0)
        self.settle_forward_scale = float(np.clip(settle_forward_scale, 0.0, 1.0))
        self.post_support_hold_s = max(float(post_support_hold_s), 0.0)
        self.post_support_forward_scale = float(np.clip(post_support_forward_scale, 0.0, 1.0))
        self.post_support_height_ratio = max(float(post_support_height_ratio), 0.0)
        self.post_support_roll_threshold = (
            np.inf if post_support_roll_threshold is None else max(float(post_support_roll_threshold), 0.0)
        )
        self.post_support_pitch_threshold = (
            np.inf if post_support_pitch_threshold is None else max(float(post_support_pitch_threshold), 0.0)
        )
        self.post_support_min_grf_z = max(float(post_support_min_grf_z), 0.0)
        self.post_support_min_rear_load_share = max(float(post_support_min_rear_load_share), 0.0)
        self.all_contact_stabilization_hold_s = max(float(all_contact_stabilization_hold_s), 0.0)
        self.all_contact_stabilization_forward_scale = float(
            np.clip(all_contact_stabilization_forward_scale, 0.0, 1.0)
        )
        self.all_contact_stabilization_front_alpha_scale = float(
            np.clip(all_contact_stabilization_front_alpha_scale, 0.0, 1.0)
        )
        self.all_contact_stabilization_height_ratio = max(float(all_contact_stabilization_height_ratio), 0.0)
        self.all_contact_stabilization_roll_threshold = (
            np.inf
            if all_contact_stabilization_roll_threshold is None
            else max(float(all_contact_stabilization_roll_threshold), 0.0)
        )
        self.all_contact_stabilization_pitch_threshold = (
            np.inf
            if all_contact_stabilization_pitch_threshold is None
            else max(float(all_contact_stabilization_pitch_threshold), 0.0)
        )
        self.all_contact_stabilization_min_rear_load_share = max(
            float(all_contact_stabilization_min_rear_load_share),
            0.0,
        )
        self.all_contact_stabilization_min_rear_leg_load_share = max(
            float(all_contact_stabilization_min_rear_leg_load_share),
            0.0,
        )
        self.all_contact_stabilization_retrigger_limit = max(int(all_contact_stabilization_retrigger_limit), 0)
        self.front_transition_guard_hold_s = max(float(front_transition_guard_hold_s), 0.0)
        self.front_transition_guard_forward_scale = float(
            np.clip(front_transition_guard_forward_scale, 0.0, 1.0)
        )
        self.front_transition_guard_roll_threshold = (
            np.inf
            if front_transition_guard_roll_threshold is None
            else max(float(front_transition_guard_roll_threshold), 0.0)
        )
        self.front_transition_guard_pitch_threshold = (
            np.inf
            if front_transition_guard_pitch_threshold is None
            else max(float(front_transition_guard_pitch_threshold), 0.0)
        )
        self.front_transition_guard_height_ratio = max(float(front_transition_guard_height_ratio), 0.0)
        self.front_transition_guard_release_tail_s = max(float(front_transition_guard_release_tail_s), 0.0)
        self.pre_swing_guard_roll_threshold = (
            np.inf if pre_swing_guard_roll_threshold is None else max(float(pre_swing_guard_roll_threshold), 0.0)
        )
        self.pre_swing_guard_pitch_threshold = (
            np.inf if pre_swing_guard_pitch_threshold is None else max(float(pre_swing_guard_pitch_threshold), 0.0)
        )
        self.pre_swing_guard_height_ratio = max(float(pre_swing_guard_height_ratio), 0.0)
        self.confirm_keep_swing = bool(confirm_keep_swing)

    def reset(self) -> None:
        self.actual_contact_elapsed_s = np.zeros(2, dtype=float)
        self.pending_confirm = np.zeros(2, dtype=int)
        self.post_support_remaining_s = np.zeros(2, dtype=float)
        self.all_contact_stabilization_remaining_s = np.zeros(2, dtype=float)
        self.all_contact_stabilization_retrigger_remaining = np.zeros(2, dtype=int)
        self.front_transition_guard_remaining_s = np.zeros(2, dtype=float)

    @staticmethod
    def _local_leg_index(global_leg_id: int) -> int:
        return int(global_leg_id) - 2

    def post_support_running(self, global_leg_id: int) -> bool:
        return bool(self.post_support_remaining_s[self._local_leg_index(global_leg_id)] > 1e-9)

    def all_contact_stabilization_running(self, global_leg_id: int) -> bool:
        return bool(self.all_contact_stabilization_remaining_s[self._local_leg_index(global_leg_id)] > 1e-9)

    def sync_debug_arrays(
        self,
        *,
        target_elapsed_s: np.ndarray | None = None,
        target_pending_confirm: np.ndarray | None = None,
    ) -> None:
        if target_elapsed_s is not None:
            target_elapsed_s[:] = 0.0
            target_elapsed_s[2:4] = np.asarray(self.actual_contact_elapsed_s, dtype=float)
        if target_pending_confirm is not None:
            target_pending_confirm[:] = 0
            target_pending_confirm[2:4] = np.asarray(self.pending_confirm, dtype=int)

    def update_actual_contact_elapsed(self, rear_contact_signal: np.ndarray, dt: float) -> None:
        rear_contact_signal = np.asarray(rear_contact_signal, dtype=int).reshape(-1)
        max_elapsed = max(float(self.contact_debounce_s), float(dt))
        for local_leg in range(min(2, rear_contact_signal.size)):
            if int(rear_contact_signal[local_leg]) == 1:
                self.actual_contact_elapsed_s[local_leg] = min(
                    float(self.actual_contact_elapsed_s[local_leg]) + float(dt),
                    max_elapsed,
                )
            else:
                self.actual_contact_elapsed_s[local_leg] = 0.0

    def clear_pending_confirm(self, global_leg_id: int) -> None:
        self.pending_confirm[self._local_leg_index(global_leg_id)] = 0

    def prime_pending_confirm(
        self,
        global_leg_id: int,
        *,
        planned_stance: bool,
        waiting_for_recontact: bool,
        contact_ready: bool,
    ) -> int:
        local_leg = self._local_leg_index(global_leg_id)
        if not planned_stance:
            self.pending_confirm[local_leg] = int(waiting_for_recontact and contact_ready)
        else:
            self.pending_confirm[local_leg] = 0
        return int(self.pending_confirm[local_leg])

    def contact_ready(
        self,
        global_leg_id: int,
        rear_contact_signal: np.ndarray,
        *,
        waiting_for_recontact: bool,
        swing_phase: float | None = None,
        current_foot_vz: float | None = None,
        foot_grf_world: np.ndarray | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        local_leg = self._local_leg_index(global_leg_id)
        rear_contact_signal = np.asarray(rear_contact_signal, dtype=int).reshape(-1)
        if local_leg >= rear_contact_signal.size:
            return False

        if self.contact_debounce_s <= 1e-9:
            ready = bool(rear_contact_signal[local_leg])
        else:
            ready = bool(rear_contact_signal[local_leg]) and (
                float(self.actual_contact_elapsed_s[local_leg]) + 1e-12 >= float(self.contact_debounce_s)
            )
        if (not ready) or (not waiting_for_recontact):
            return ready

        if swing_phase is not None and float(swing_phase) + 1e-12 < float(self.contact_min_phase):
            return False
        if current_foot_vz is not None and np.isfinite(self.contact_max_upward_vel):
            if float(current_foot_vz) > float(self.contact_max_upward_vel) + 1e-12:
                return False
        if self.contact_min_grf_z > 1e-9 and foot_grf_world is not None:
            foot_grf_world = np.asarray(foot_grf_world, dtype=float)
            if foot_grf_world.ndim == 2 and foot_grf_world.shape[0] > int(global_leg_id) and foot_grf_world.shape[1] >= 3:
                vertical_grf = max(float(foot_grf_world[int(global_leg_id), 2]), 0.0)
                if vertical_grf + 1e-12 < float(self.contact_min_grf_z):
                    return False
        return True

    def should_delay_reacquire(
        self,
        *,
        planned_stance: bool,
        waiting_for_recontact: bool,
        actual_contact: bool,
        swing_time: float,
    ) -> bool:
        return bool(
            planned_stance
            and waiting_for_recontact
            and (not actual_contact)
            and self.reacquire_min_swing_time_s > 1e-9
            and float(swing_time) + 1e-12 < float(self.reacquire_min_swing_time_s)
        )

    def update_reacquire_window(
        self,
        *,
        planned_stance: bool,
        waiting_for_recontact: bool,
        contact_ready: bool,
        current_elapsed_s: float,
        simulation_dt: float,
        horizon_steps: int,
    ) -> tuple[int, float, int, float]:
        if not self.enabled:
            return 0, 0.0, 0, 1.0
        if (
            (not planned_stance)
            or (not waiting_for_recontact)
            or contact_ready
            or self.reacquire_hold_s <= 1e-9
        ):
            return 0, 0.0, 0, 1.0

        next_elapsed = min(float(current_elapsed_s) + float(simulation_dt), float(self.reacquire_hold_s))
        remaining_hold_s = max(float(self.reacquire_hold_s) - next_elapsed, 0.0)
        hold_steps = max(int(np.floor(remaining_hold_s / self.mpc_dt)) + 1, 1)
        hold_steps = min(int(hold_steps), int(horizon_steps))
        return 1, next_elapsed, hold_steps, float(self.reacquire_forward_scale)

    def should_keep_confirm(
        self,
        global_leg_id: int,
        *,
        waiting_for_recontact: bool,
        planned_stance: bool,
        contact_ready: bool,
        prev_reacquire_active: bool,
        confirm_elapsed_s: float,
        stance_recontact: bool,
    ) -> bool:
        if not self.enabled:
            return False
        local_leg = self._local_leg_index(global_leg_id)
        keep_confirm = bool(stance_recontact or prev_reacquire_active or float(confirm_elapsed_s) > 1e-9)
        if not keep_confirm:
            keep_confirm = bool(
                waiting_for_recontact
                and planned_stance
                and contact_ready
                and int(self.pending_confirm[local_leg]) == 1
            )
        return keep_confirm

    def consume_confirm(self, global_leg_id: int, confirm_elapsed_s: float, simulation_dt: float) -> tuple[int, float, float]:
        if not self.enabled:
            return 0, 0.0, 1.0
        local_leg = self._local_leg_index(global_leg_id)
        self.pending_confirm[local_leg] = 0
        if self.confirm_hold_s <= 1e-9:
            return 0, 0.0, 1.0
        next_confirm_elapsed = min(float(confirm_elapsed_s) + float(simulation_dt), float(self.confirm_hold_s))
        next_confirm_elapsed = 0.0 if next_confirm_elapsed >= (float(self.confirm_hold_s) - 1e-12) else next_confirm_elapsed
        return 1, next_confirm_elapsed, float(self.confirm_forward_scale)

    def update_settle_window(
        self,
        *,
        planned_stance: bool,
        contact_ready: bool,
        prev_reacquire_active: bool,
        stance_recontact: bool,
        settle_remaining_s: float,
        simulation_dt: float,
    ) -> tuple[int, float, float]:
        if not self.enabled:
            return 0, 0.0, 1.0
        if (not planned_stance) or (not contact_ready):
            return 0, 0.0, 1.0

        next_remaining = float(settle_remaining_s)
        if prev_reacquire_active or stance_recontact:
            next_remaining = max(next_remaining, float(self.settle_hold_s))
        if next_remaining <= 1e-9:
            return 0, 0.0, 1.0

        next_remaining = max(0.0, next_remaining - float(simulation_dt))
        return 1, next_remaining, float(self.settle_forward_scale)

    def should_keep_swing_during_confirm(
        self,
        *,
        confirm_active: bool,
        contact_ready: bool,
    ) -> bool:
        if not self.enabled:
            return False
        return bool(self.confirm_keep_swing and confirm_active and contact_ready)

    def update_post_support_window(
        self,
        global_leg_id: int,
        *,
        trigger: bool,
        planned_stance: bool,
        actual_contact: bool,
        simulation_dt: float,
        height_ratio: float,
        roll_mag: float,
        pitch_mag: float,
        leg_grf_z: float,
        rear_load_share: float,
        recovery_active: bool,
    ) -> tuple[int, float]:
        if not self.enabled:
            return 0, 1.0
        local_leg = self._local_leg_index(global_leg_id)
        if trigger and self.post_support_hold_s > 1e-9:
            self.post_support_remaining_s[local_leg] = max(
                float(self.post_support_remaining_s[local_leg]),
                float(self.post_support_hold_s),
            )

        remaining = float(self.post_support_remaining_s[local_leg])
        if remaining <= 1e-9:
            self.post_support_remaining_s[local_leg] = 0.0
            return 0, 1.0
        if (not planned_stance) or (not actual_contact):
            self.post_support_remaining_s[local_leg] = 0.0
            return 0, 1.0

        needs_support = bool(recovery_active)
        if (not needs_support) and self.post_support_min_grf_z > 1e-9:
            needs_support = float(leg_grf_z) + 1e-12 < float(self.post_support_min_grf_z)
        if (not needs_support) and self.post_support_min_rear_load_share > 1e-9:
            needs_support = float(rear_load_share) + 1e-12 < float(self.post_support_min_rear_load_share)
        if (not needs_support) and self.post_support_height_ratio > 1e-9:
            needs_support = float(height_ratio) <= float(self.post_support_height_ratio)
        if (not needs_support) and np.isfinite(self.post_support_roll_threshold):
            needs_support = float(roll_mag) >= float(self.post_support_roll_threshold)
        if (not needs_support) and np.isfinite(self.post_support_pitch_threshold):
            needs_support = float(pitch_mag) >= float(self.post_support_pitch_threshold)

        self.post_support_remaining_s[local_leg] = max(0.0, remaining - float(simulation_dt))
        if not needs_support:
            return 0, 1.0
        return 1, float(self.post_support_forward_scale)

    def update_all_contact_stabilization_window(
        self,
        global_leg_id: int,
        *,
        trigger: bool,
        planned_stance: bool,
        actual_contact: bool,
        all_actual_contact: bool,
        simulation_dt: float,
        height_ratio: float,
        roll_mag: float,
        pitch_mag: float,
        rear_load_share: float,
        rear_leg_load_share: float,
    ) -> tuple[int, float, float]:
        if not self.enabled:
            return 0, 1.0, 1.0
        local_leg = self._local_leg_index(global_leg_id)
        if trigger and self.all_contact_stabilization_hold_s > 1e-9:
            self.all_contact_stabilization_remaining_s[local_leg] = max(
                float(self.all_contact_stabilization_remaining_s[local_leg]),
                float(self.all_contact_stabilization_hold_s),
            )
            self.all_contact_stabilization_retrigger_remaining[local_leg] = max(
                int(self.all_contact_stabilization_retrigger_remaining[local_leg]),
                int(self.all_contact_stabilization_retrigger_limit),
            )

        remaining = float(self.all_contact_stabilization_remaining_s[local_leg])
        if (not planned_stance) or (not actual_contact) or (not all_actual_contact):
            self.all_contact_stabilization_remaining_s[local_leg] = 0.0
            self.all_contact_stabilization_retrigger_remaining[local_leg] = 0
            return 0, 1.0, 1.0

        needs_support = False
        if self.all_contact_stabilization_min_rear_load_share > 1e-9:
            needs_support = float(rear_load_share) + 1e-12 < float(self.all_contact_stabilization_min_rear_load_share)
        if (not needs_support) and self.all_contact_stabilization_min_rear_leg_load_share > 1e-9:
            needs_support = float(rear_leg_load_share) + 1e-12 < float(
                self.all_contact_stabilization_min_rear_leg_load_share
            )
        if (not needs_support) and self.all_contact_stabilization_height_ratio > 1e-9:
            needs_support = float(height_ratio) <= float(self.all_contact_stabilization_height_ratio)
        if (not needs_support) and np.isfinite(self.all_contact_stabilization_roll_threshold):
            needs_support = float(roll_mag) >= float(self.all_contact_stabilization_roll_threshold)
        if (not needs_support) and np.isfinite(self.all_contact_stabilization_pitch_threshold):
            needs_support = float(pitch_mag) >= float(self.all_contact_stabilization_pitch_threshold)

        if remaining <= 1e-9:
            if (
                needs_support
                and self.all_contact_stabilization_hold_s > 1e-9
                and int(self.all_contact_stabilization_retrigger_remaining[local_leg]) > 0
            ):
                remaining = float(self.all_contact_stabilization_hold_s)
                self.all_contact_stabilization_remaining_s[local_leg] = remaining
                self.all_contact_stabilization_retrigger_remaining[local_leg] = max(
                    int(self.all_contact_stabilization_retrigger_remaining[local_leg]) - 1,
                    0,
                )
            else:
                self.all_contact_stabilization_remaining_s[local_leg] = 0.0
                self.all_contact_stabilization_retrigger_remaining[local_leg] = 0
                return 0, 1.0, 1.0

        self.all_contact_stabilization_remaining_s[local_leg] = max(
            0.0,
            remaining - float(simulation_dt),
        )
        if not needs_support:
            return 0, 1.0, 1.0
        return (
            1,
            float(self.all_contact_stabilization_forward_scale),
            float(self.all_contact_stabilization_front_alpha_scale),
        )

    def should_start_touchdown_support(
        self,
        *,
        gait_name: str,
        planned_stance: bool,
        waiting_for_recontact: bool,
        actual_contact: bool,
        previous_actual_contact: bool,
        contact_ready: bool,
    ) -> bool:
        if not self.enabled:
            return False
        if gait_name != 'crawl' or (not contact_ready):
            return False
        contact_returned_now = bool(actual_contact) and (not bool(previous_actual_contact))
        if (not planned_stance) and (waiting_for_recontact or contact_returned_now):
            return True
        if planned_stance and waiting_for_recontact and contact_returned_now:
            return True
        return False

    def should_accept_touchdown_as_stance(
        self,
        *,
        gait_name: str,
        planned_stance: bool,
        waiting_for_recontact: bool,
        contact_ready: bool,
    ) -> bool:
        if not self.enabled:
            return False
        return bool(
            gait_name == 'crawl'
            and (not planned_stance)
            and waiting_for_recontact
            and contact_ready
        )

    def should_accept_late_stance_contact(
        self,
        global_leg_id: int,
        rear_contact_signal: np.ndarray,
        *,
        gait_name: str,
        planned_stance: bool,
        waiting_for_recontact: bool,
        actual_contact: bool,
        previous_actual_contact: bool,
        recovery_active: bool,
        roll_mag: float,
        pitch_mag: float,
        height_ratio: float,
        current_foot_vz: float | None = None,
        foot_grf_world: np.ndarray | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        local_leg = self._local_leg_index(global_leg_id)
        rear_contact_signal = np.asarray(rear_contact_signal, dtype=int).reshape(-1)
        if local_leg >= rear_contact_signal.size:
            return False

        if gait_name != 'crawl':
            return False
        if not (
            planned_stance
            and waiting_for_recontact
            and actual_contact
            and previous_actual_contact
            and recovery_active
            and bool(rear_contact_signal[local_leg])
        ):
            return False
        posture_guard_active = False
        if self.pre_swing_guard_height_ratio > 1e-9:
            posture_guard_active = float(height_ratio) <= float(self.pre_swing_guard_height_ratio)
        if (not posture_guard_active) and np.isfinite(self.pre_swing_guard_roll_threshold):
            posture_guard_active = float(roll_mag) + 1e-12 >= float(self.pre_swing_guard_roll_threshold)
        if (not posture_guard_active) and np.isfinite(self.pre_swing_guard_pitch_threshold):
            posture_guard_active = float(pitch_mag) + 1e-12 >= float(self.pre_swing_guard_pitch_threshold)
        if not posture_guard_active:
            return False
        if self.contact_debounce_s > 1e-9 and (
            float(self.actual_contact_elapsed_s[local_leg]) + 1e-12 < float(self.contact_debounce_s)
        ):
            return False
        if current_foot_vz is not None and np.isfinite(self.contact_max_upward_vel):
            if float(current_foot_vz) > float(self.contact_max_upward_vel) + 1e-12:
                return False
        if self.contact_min_grf_z > 1e-9 and foot_grf_world is not None:
            foot_grf_world = np.asarray(foot_grf_world, dtype=float)
            if (
                foot_grf_world.ndim == 2
                and foot_grf_world.shape[0] > int(global_leg_id)
                and foot_grf_world.shape[1] >= 3
            ):
                vertical_grf = max(float(foot_grf_world[int(global_leg_id), 2]), 0.0)
                # Allow a very small GRF slack here. This override is only used
                # once the rear foot has already remained in contact through a
                # late recovery phase, and the main failure mode we observed was
                # flapping around the threshold by a few tenths of a Newton.
                required_grf = max(float(self.contact_min_grf_z) - 0.5, 0.0)
                if vertical_grf + 1e-12 < required_grf:
                    return False
        return True

    def pending_support_required(
        self,
        *,
        planned_stance: bool,
        waiting_for_recontact: bool,
        contact_ready: bool,
    ) -> bool:
        if not self.enabled:
            return False
        return bool(planned_stance and waiting_for_recontact and (not contact_ready))

    def should_delay_preswing_for_posture(
        self,
        *,
        gait_name: str,
        scheduled_swing: bool,
        current_contact: bool,
        actual_contact: bool,
        recovery_active: bool,
        roll_mag: float,
        pitch_mag: float,
        height_ratio: float,
    ) -> bool:
        if not self.enabled:
            return False
        if gait_name != 'crawl':
            return False
        if not (scheduled_swing and current_contact and actual_contact and recovery_active):
            return False
        if self.pre_swing_guard_height_ratio > 1e-9 and float(height_ratio) <= float(self.pre_swing_guard_height_ratio):
            return True
        if np.isfinite(self.pre_swing_guard_roll_threshold) and float(roll_mag) >= float(self.pre_swing_guard_roll_threshold):
            return True
        if np.isfinite(self.pre_swing_guard_pitch_threshold) and float(pitch_mag) >= float(self.pre_swing_guard_pitch_threshold):
            return True
        return False

    def should_delay_front_preswing_for_rear_transition(
        self,
        *,
        gait_name: str,
        scheduled_swing: bool,
        current_contact: bool,
        actual_contact: bool,
        rear_transition_active: bool,
        roll_mag: float,
        pitch_mag: float,
        height_ratio: float,
    ) -> bool:
        if not self.enabled:
            return False
        if gait_name != 'crawl':
            return False
        if self.front_transition_guard_hold_s <= 1e-9:
            return False
        if not (scheduled_swing and current_contact and actual_contact and rear_transition_active):
            return False
        if self.front_transition_guard_height_ratio > 1e-9 and float(height_ratio) <= float(
            self.front_transition_guard_height_ratio
        ):
            return True
        if np.isfinite(self.front_transition_guard_roll_threshold) and float(roll_mag) >= float(
            self.front_transition_guard_roll_threshold
        ):
            return True
        if np.isfinite(self.front_transition_guard_pitch_threshold) and float(pitch_mag) >= float(
            self.front_transition_guard_pitch_threshold
        ):
            return True
        return False

    def update_front_transition_guard_window(
        self,
        global_leg_id: int,
        *,
        gait_name: str,
        scheduled_swing: bool,
        current_contact: bool,
        actual_contact: bool,
        rear_transition_active: bool,
        roll_mag: float,
        pitch_mag: float,
        height_ratio: float,
        simulation_dt: float,
        rear_support_active: bool,
        rear_all_contact_active: bool,
        rear_contacts_stable: bool,
    ) -> tuple[int, float, float]:
        if not self.enabled:
            return 0, 0.0, 1.0

        local_leg = min(max(int(global_leg_id), 0), 1)
        hold_s = float(self.front_transition_guard_hold_s)
        if gait_name != 'crawl' or hold_s <= 1e-9:
            self.front_transition_guard_remaining_s[local_leg] = 0.0
            return 0, 0.0, 1.0

        if not (scheduled_swing and current_contact and actual_contact):
            self.front_transition_guard_remaining_s[local_leg] = 0.0
            return 0, 0.0, 1.0

        support_ready = bool(rear_contacts_stable)
        if support_ready:
            release_tail_s = min(
                max(float(self.front_transition_guard_release_tail_s), 0.0),
                hold_s,
            )
            if release_tail_s <= 1e-9:
                self.front_transition_guard_remaining_s[local_leg] = 0.0
                return 0, 0.0, 1.0
            self.front_transition_guard_remaining_s[local_leg] = min(
                float(self.front_transition_guard_remaining_s[local_leg]),
                release_tail_s,
            )

        trigger_now = self.should_delay_front_preswing_for_rear_transition(
            gait_name=gait_name,
            scheduled_swing=scheduled_swing,
            current_contact=current_contact,
            actual_contact=actual_contact,
            rear_transition_active=rear_transition_active,
            roll_mag=roll_mag,
            pitch_mag=pitch_mag,
            height_ratio=height_ratio,
        )
        if trigger_now and not support_ready:
            self.front_transition_guard_remaining_s[local_leg] = max(
                float(self.front_transition_guard_remaining_s[local_leg]),
                hold_s,
            )

        remaining_s = float(self.front_transition_guard_remaining_s[local_leg])
        if remaining_s <= 1e-9:
            self.front_transition_guard_remaining_s[local_leg] = 0.0
            return 0, 0.0, 1.0

        next_remaining_s = max(0.0, remaining_s - float(simulation_dt))
        self.front_transition_guard_remaining_s[local_leg] = next_remaining_s
        if next_remaining_s <= 1e-9 and not trigger_now:
            return 0, 0.0, 1.0
        return 1, next_remaining_s, float(self.front_transition_guard_forward_scale)
