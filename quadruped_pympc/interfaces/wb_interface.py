import copy
import time

import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr
from scipy.spatial.transform import Rotation as R

from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.foothold_reference_generator import FootholdReferenceGenerator
#from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_adam import InverseKinematicsNumeric
from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_mujoco import InverseKinematicsNumeric
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator
from quadruped_pympc.helpers.velocity_modulator import VelocityModulator
from quadruped_pympc.helpers.early_stance_detector import EarlyStanceDetector

if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
    from quadruped_pympc.helpers.visual_foothold_adaptation import VisualFootholdAdaptation


class WBInterface:
    """
    WBInterface is responsible for interfacing with the whole body controller of a quadruped robot.
    It initializes the necessary components for motion planning and control, including gait generation,
    swing trajectory control, and terrain estimation.
    """

    def __init__(self,
                 initial_feet_pos: LegsAttr,
                 legs_order: tuple[str, str, str, str] = ('FL', 'FR', 'RL', 'RR'),
                 feet_geom_id : LegsAttr = None):
        """Constructor of the WBInterface class

        Args:
            initial_feet_pos (LegsAttr): initial feet positions, otherwise they will be all zero
            legs_order (tuple[str, str, str, str], optional): order of the leg. Defaults to ('FL', 'FR', 'RL', 'RR').
        """

        mpc_dt = cfg.mpc_params['dt']
        self.mpc_dt = float(mpc_dt)
        horizon = cfg.mpc_params['horizon']
        self.legs_order = legs_order

        # Periodic gait generator --------------------------------------------------------------
        gait_name = cfg.simulation_params['gait']
        gait_params = cfg.simulation_params['gait_params'][gait_name]
        gait_type, duty_factor, step_frequency = (
            gait_params['type'],
            gait_params['duty_factor'],
            gait_params['step_freq'],
        )
        # Given the possibility to use nonuniform discretization,
        # we generate a contact sequence two times longer and with a dt half of the one of the mpc
        self.pgg = PeriodicGaitGenerator(
            duty_factor=duty_factor, step_freq=step_frequency, gait_type=gait_type, horizon=horizon
        )
        # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
        if cfg.mpc_params['use_nonuniform_discretization']:
            self.contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], mpc_dt]
            self.contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], horizon]
        else:
            self.contact_sequence_dts = [mpc_dt]
            self.contact_sequence_lenghts = [horizon]

        # Create the foothold reference generator ------------------------------------------------
        stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
        self.frg = FootholdReferenceGenerator(
            stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=initial_feet_pos
        )

        # Create swing trajectory generator ------------------------------------------------------
        self.step_height = cfg.simulation_params['step_height']
        swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
        position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
        velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
        swing_generator = cfg.simulation_params['swing_generator']
        self.stc = SwingTrajectoryController(
            step_height=self.step_height,
            swing_period=swing_period,
            position_gain_fb=position_gain_fb,
            velocity_gain_fb=velocity_gain_fb,
            generator=swing_generator,
        )
        self.last_des_foot_pos = LegsAttr(*[np.zeros((3,)) for _ in range(4)])
        

        # Terrain estimator -----------------------------------------------------------------------
        self.terrain_computation = TerrainEstimator()

        # Inverse Kinematics ---------------------------------------------------------------------
        self.ik = InverseKinematicsNumeric()

        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            # Visual foothold adaptation -------------------------------------------------------------
            self.vfa = VisualFootholdAdaptation(
                legs_order=self.legs_order, adaptation_strategy=cfg.simulation_params['visual_foothold_adaptation']
            )

        # Velocity modulator ---------------------------------------------------------------------
        self.vm = VelocityModulator()

        # Early Stance detector -------------------------------------------------------------------
        self.esd = EarlyStanceDetector(feet_geom_id)
        self.contact_latch_steps = 6
        self.startup_full_stance_time_s = 0.0
        self.contact_latch_budget_s = 0.0
        self.virtual_unlatch_hold_s = 0.0
        self.pre_swing_gate_min_margin = 0.0
        self.front_pre_swing_gate_min_margin = 0.0
        self.rear_pre_swing_gate_min_margin = 0.0
        self.support_contact_confirm_hold_s = 0.0
        self.front_support_contact_confirm_hold_s = 0.0
        self.rear_support_contact_confirm_hold_s = 0.0
        self.support_confirm_min_contacts = 3
        self.support_confirm_require_front_rear_span = True
        self.support_confirm_forward_scale = 1.0
        self.front_stance_dropout_reacquire = False
        self.front_late_release_phase_threshold = 1.1
        self.front_late_release_min_margin = 0.0
        self.front_late_release_hold_s = 0.0
        self.front_late_release_extra_margin = 0.0
        self.front_late_release_pitch_guard = np.inf
        self.front_late_release_roll_guard = np.inf
        self.support_margin_preview_s = 0.0
        self.touchdown_reacquire_hold_s = 0.0
        self.front_touchdown_reacquire_hold_s = 0.0
        self.touchdown_reacquire_forward_scale = 1.0
        self.touchdown_reacquire_xy_blend = 0.0
        self.front_touchdown_reacquire_xy_blend = 0.0
        self.touchdown_reacquire_extra_depth = 0.0
        self.front_touchdown_reacquire_extra_depth = 0.0
        self.touchdown_reacquire_forward_bias = 0.0
        self.front_touchdown_reacquire_forward_bias = 0.0
        self.stance_anchor_update_alpha = 0.0
        self.front_stance_anchor_update_alpha = 0.0
        self.touchdown_support_anchor_update_alpha = 0.0
        self.front_touchdown_support_anchor_update_alpha = 0.0
        self.touchdown_confirm_hold_s = 0.0
        self.front_touchdown_confirm_hold_s = 0.0
        self.touchdown_confirm_forward_scale = 1.0
        self.touchdown_settle_hold_s = 0.0
        self.front_touchdown_settle_hold_s = 0.0
        self.touchdown_settle_forward_scale = 1.0
        self.front_margin_rescue_hold_s = 0.0
        self.front_margin_rescue_forward_scale = 1.0
        self.front_margin_rescue_min_margin = 0.0
        self.front_margin_rescue_margin_gap = 0.0
        self.front_margin_rescue_alpha_margin = 0.02
        self.front_margin_rescue_roll_threshold = np.inf
        self.front_margin_rescue_pitch_threshold = np.inf
        self.front_margin_rescue_height_ratio = 0.0
        self.front_margin_rescue_recent_swing_window_s = 0.0
        self.front_margin_rescue_require_all_contact = True
        self.full_contact_recovery_hold_s = 0.0
        self.full_contact_recovery_forward_scale = 1.0
        self.full_contact_recovery_roll_threshold = np.inf
        self.full_contact_recovery_pitch_threshold = np.inf
        self.full_contact_recovery_height_ratio = 0.0
        self.full_contact_recovery_recent_window_s = 0.0
        self.pre_swing_gate_hold_s = 0.0
        self.startup_full_stance_elapsed_s = 0.0
        self.contact_latch_elapsed_s = np.zeros(4, dtype=float)
        self.pre_swing_gate_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_reacquire_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_settle_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_alpha = np.zeros(4, dtype=float)
        self.front_margin_rescue_recent_swing_remaining_s = np.zeros(2, dtype=float)
        self.touchdown_reacquire_armed = np.zeros(4, dtype=int)
        self.rear_handoff_support_remaining_s = 0.0
        self.rear_handoff_support_active = 0
        self.rear_handoff_support_mask = np.zeros(4, dtype=int)
        self.rear_swing_bridge_remaining_s = 0.0
        self.rear_swing_bridge_active = 0
        self.full_contact_recovery_remaining_s = 0.0
        self.front_touchdown_support_recent_remaining_s = 0.0
        self.rear_swing_bridge_recent_front_remaining_s = 0.0
        self.last_support_margin = np.full(4, np.nan, dtype=float)
        self.last_support_margin_query_xy = np.full((4, 2), np.nan, dtype=float)
        self.front_late_release_active = np.zeros(4, dtype=int)
        self.support_confirm_active = np.zeros(4, dtype=int)
        self.pre_swing_gate_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_active = np.zeros(4, dtype=int)
        self.touchdown_confirm_active = np.zeros(4, dtype=int)
        self.touchdown_settle_active = np.zeros(4, dtype=int)
        self.touchdown_support_active = np.zeros(4, dtype=int)
        self.front_margin_rescue_active = np.zeros(4, dtype=int)
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.last_gate_forward_scale = 1.0

        self.current_contact = np.array([1, 1, 1, 1])
        self.previous_contact = np.array([1, 1, 1, 1])
        self.planned_contact = np.array([1, 1, 1, 1])
        self.previous_actual_contact = np.array([0, 0, 0, 0])
        self.latched_swing_time = np.zeros(4, dtype=float)
        self.virtual_unlatch_hold_remaining_s = np.zeros(4, dtype=float)
        self._refresh_linear_timing_params()

    @staticmethod
    def _resolve_duration_seconds(
        params: dict,
        seconds_key: str,
        legacy_steps_key: str,
        *,
        step_dt: float,
        default_s: float = 0.0,
    ) -> float:
        seconds_val = params.get(seconds_key, None)
        if seconds_val is not None:
            return max(float(seconds_val), 0.0)

        legacy_val = params.get(legacy_steps_key, None)
        if legacy_val is not None:
            return max(float(legacy_val), 0.0) * max(float(step_dt), 1e-6)

        return max(float(default_s), 0.0)

    def _refresh_linear_timing_params(self) -> None:
        params = getattr(cfg, 'linear_osqp_params', {})
        self.contact_latch_steps = int(params.get('contact_latch_steps', 6))
        self.startup_full_stance_time_s = self._resolve_duration_seconds(
            params, 'startup_full_stance_time_s', 'startup_full_stance_steps', step_dt=self.mpc_dt
        )
        self.contact_latch_budget_s = self._resolve_duration_seconds(
            params, 'contact_latch_budget_s', 'contact_latch_budget_steps', step_dt=self.mpc_dt
        )
        self.virtual_unlatch_hold_s = self._resolve_duration_seconds(
            params, 'virtual_unlatch_hold_s', 'virtual_unlatch_hold_steps', step_dt=self.mpc_dt
        )
        self.pre_swing_gate_min_margin = max(float(params.get('pre_swing_gate_min_margin', 0.0)), 0.0)
        front_margin = params.get('front_pre_swing_gate_min_margin', None)
        rear_margin = params.get('rear_pre_swing_gate_min_margin', None)
        self.front_pre_swing_gate_min_margin = max(
            float(self.pre_swing_gate_min_margin if front_margin is None else front_margin),
            0.0,
        )
        self.rear_pre_swing_gate_min_margin = max(
            float(self.pre_swing_gate_min_margin if rear_margin is None else rear_margin),
            0.0,
        )
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
        self.front_touchdown_reacquire_hold_s = max(
            float(self.touchdown_reacquire_hold_s if front_touchdown_hold is None else front_touchdown_hold),
            0.0,
        )
        self.touchdown_reacquire_forward_scale = float(
            np.clip(params.get('touchdown_reacquire_forward_scale', 1.0), 0.0, 1.0)
        )
        touchdown_xy_blend = params.get('touchdown_reacquire_xy_blend', 0.0)
        front_touchdown_xy_blend = params.get('front_touchdown_reacquire_xy_blend', None)
        touchdown_extra_depth = params.get('touchdown_reacquire_extra_depth', 0.0)
        front_touchdown_extra_depth = params.get('front_touchdown_reacquire_extra_depth', None)
        self.touchdown_reacquire_xy_blend = float(np.clip(touchdown_xy_blend, 0.0, 1.0))
        self.front_touchdown_reacquire_xy_blend = float(
            np.clip(
                self.touchdown_reacquire_xy_blend if front_touchdown_xy_blend is None else front_touchdown_xy_blend,
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
        touchdown_forward_bias = params.get('touchdown_reacquire_forward_bias', 0.0)
        front_touchdown_forward_bias = params.get('front_touchdown_reacquire_forward_bias', None)
        self.touchdown_reacquire_forward_bias = float(touchdown_forward_bias)
        self.front_touchdown_reacquire_forward_bias = float(
            self.touchdown_reacquire_forward_bias
            if front_touchdown_forward_bias is None
            else front_touchdown_forward_bias
        )
        self.stance_anchor_update_alpha = float(
            np.clip(params.get('stance_anchor_update_alpha', 0.0), 0.0, 1.0)
        )
        front_anchor_update = params.get('front_stance_anchor_update_alpha', None)
        self.front_stance_anchor_update_alpha = float(
            np.clip(
                self.stance_anchor_update_alpha if front_anchor_update is None else front_anchor_update,
                0.0,
                1.0,
            )
        )
        self.touchdown_support_anchor_update_alpha = float(
            np.clip(params.get('touchdown_support_anchor_update_alpha', 0.0), 0.0, 1.0)
        )
        front_touchdown_support_anchor_update = params.get('front_touchdown_support_anchor_update_alpha', None)
        self.front_touchdown_support_anchor_update_alpha = float(
            np.clip(
                self.touchdown_support_anchor_update_alpha
                if front_touchdown_support_anchor_update is None
                else front_touchdown_support_anchor_update,
                0.0,
                1.0,
            )
        )
        self.touchdown_confirm_hold_s = max(float(params.get('touchdown_confirm_hold_s', 0.0)), 0.0)
        front_touchdown_confirm_hold = params.get('front_touchdown_confirm_hold_s', None)
        self.front_touchdown_confirm_hold_s = max(
            float(self.touchdown_confirm_hold_s if front_touchdown_confirm_hold is None else front_touchdown_confirm_hold),
            0.0,
        )
        self.touchdown_confirm_forward_scale = float(
            np.clip(params.get('touchdown_confirm_forward_scale', 1.0), 0.0, 1.0)
        )
        self.touchdown_settle_hold_s = max(float(params.get('touchdown_settle_hold_s', 0.0)), 0.0)
        front_touchdown_settle_hold = params.get('front_touchdown_settle_hold_s', None)
        self.front_touchdown_settle_hold_s = max(
            float(self.touchdown_settle_hold_s if front_touchdown_settle_hold is None else front_touchdown_settle_hold),
            0.0,
        )
        self.touchdown_settle_forward_scale = float(
            np.clip(params.get('touchdown_settle_forward_scale', 1.0), 0.0, 1.0)
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
        self.rear_swing_bridge_hold_s = max(float(params.get('rear_swing_bridge_hold_s', 0.0)), 0.0)
        self.rear_swing_bridge_forward_scale = float(
            np.clip(params.get('rear_swing_bridge_forward_scale', 1.0), 0.0, 1.0)
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
        self.pre_swing_gate_hold_s = max(float(params.get('pre_swing_gate_hold_s', 0.0)), 0.0)

    def _pre_swing_gate_required_margin(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_pre_swing_gate_min_margin)
        return float(self.rear_pre_swing_gate_min_margin)

    def _touchdown_reacquire_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_hold_s)
        return float(self.touchdown_reacquire_hold_s)

    def _support_contact_confirm_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_support_contact_confirm_hold_s)
        return float(self.rear_support_contact_confirm_hold_s)

    def _stance_anchor_update_alpha_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_stance_anchor_update_alpha)
        return float(self.stance_anchor_update_alpha)

    def _touchdown_support_anchor_update_alpha_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_support_anchor_update_alpha)
        return float(self.touchdown_support_anchor_update_alpha)

    def _touchdown_reacquire_xy_blend_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_xy_blend)
        return float(self.touchdown_reacquire_xy_blend)

    def _touchdown_reacquire_extra_depth_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_extra_depth)
        return float(self.touchdown_reacquire_extra_depth)

    def _touchdown_reacquire_forward_bias_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_forward_bias)
        return float(self.touchdown_reacquire_forward_bias)

    def _touchdown_confirm_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_confirm_hold_s)
        return float(self.touchdown_confirm_hold_s)

    def _touchdown_settle_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_settle_hold_s)
        return float(self.touchdown_settle_hold_s)

    def _front_margin_rescue_candidate_margin(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        feet_pos,
        com_pos: np.ndarray,
        com_vel_xy: np.ndarray,
        base_ori_euler_xyz: np.ndarray,
        base_pos_measured: np.ndarray,
    ) -> float | None:
        if int(leg_id) >= 2 or float(self.front_margin_rescue_hold_s) <= 1e-9:
            return None
        if int(self.planned_contact[leg_id]) != 1 or int(self.current_contact[leg_id]) != 1:
            return None
        if not bool(actual_contact[leg_id]):
            return None
        if self.front_margin_rescue_require_all_contact and not bool(np.all(np.asarray(actual_contact, dtype=int) == 1)):
            return None
        if (
            float(self.front_margin_rescue_recent_swing_window_s) > 1e-9
            and float(self.front_margin_rescue_recent_swing_remaining_s[leg_id]) <= 1e-9
        ):
            return None

        roll_mag = abs(float(base_ori_euler_xyz[0]))
        pitch_mag = abs(float(base_ori_euler_xyz[1]))
        ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
        height_ratio = float(base_pos_measured[2]) / ref_height

        danger = False
        if np.isfinite(self.front_margin_rescue_roll_threshold):
            danger = danger or roll_mag >= float(self.front_margin_rescue_roll_threshold)
        if np.isfinite(self.front_margin_rescue_pitch_threshold):
            danger = danger or pitch_mag >= float(self.front_margin_rescue_pitch_threshold)
        if float(self.front_margin_rescue_height_ratio) > 1e-9:
            danger = danger or height_ratio <= float(self.front_margin_rescue_height_ratio)
        if not danger:
            return None

        support_margin = self._pre_swing_gate_margin(
            leg_id,
            actual_contact,
            feet_pos,
            com_pos,
            com_vel_xy=com_vel_xy,
        )
        return float(support_margin)

    def _should_front_late_release(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        feet_pos: LegsAttr | None = None,
        com_pos: np.ndarray | None = None,
        com_vel_xy: np.ndarray | None = None,
        base_ori_euler_xyz: np.ndarray | None = None,
    ) -> bool:
        if int(leg_id) >= 2:
            return False
        if int(self.planned_contact[leg_id]) != 0 or int(actual_contact[leg_id]) != 1:
            return False

        threshold = float(self.front_late_release_phase_threshold)
        if threshold > 1.0:
            return False

        swing_period = max(float(self.stc.swing_period), 1e-6)
        phase = float(np.clip(self.latched_swing_time[leg_id] / swing_period, 0.0, 1.0))
        if phase < threshold:
            return False

        support_legs = np.flatnonzero(self.planned_contact == 1)
        if support_legs.size == 0:
            return False
        if not bool(np.all(np.asarray(actual_contact, dtype=int)[support_legs] == 1)):
            return False

        if base_ori_euler_xyz is not None:
            base_ori_euler_xyz = np.asarray(base_ori_euler_xyz, dtype=float).reshape(3)
            if abs(float(base_ori_euler_xyz[1])) > float(self.front_late_release_pitch_guard):
                return False
            if abs(float(base_ori_euler_xyz[0])) > float(self.front_late_release_roll_guard):
                return False

        required_margin = float(max(self.front_late_release_min_margin, 0.0) + max(self.front_late_release_extra_margin, 0.0))
        if required_margin > 1e-9:
            if feet_pos is None or com_pos is None:
                return False
            support_margin = self._pre_swing_gate_margin(leg_id, actual_contact, feet_pos, com_pos, com_vel_xy=com_vel_xy)
            if support_margin < required_margin:
                return False
        return True

    def _max_linear_latch_steps(self) -> int:
        swing_period = max(float(self.stc.swing_period), 1e-6)
        swing_ctrl_steps = max(int(np.floor(swing_period / max(self.mpc_dt, 1e-6))), 1)
        # Keep relatched support strictly shorter than the full swing so every swing
        # still has time to become an actual swing in the controller.
        return max(swing_ctrl_steps // 2, 1)

    def _max_linear_latch_budget_s(self) -> float:
        swing_period = max(float(self.stc.swing_period), 1e-6)
        return max(0.5 * swing_period, self.mpc_dt)

    def _effective_linear_latch_budget_s(self, requested_budget_s: float) -> float:
        max_budget_s = self._max_linear_latch_budget_s()
        if requested_budget_s <= 0.0:
            return max_budget_s
        return min(float(requested_budget_s), max_budget_s)

    def _foot_contact_from_mujoco(self, mujoco_contact) -> np.ndarray:
        actual_contact = np.zeros(4, dtype=int)
        if mujoco_contact is None or self.esd.feet_geom_id is None:
            return actual_contact

        for contact in mujoco_contact:
            for leg_id, leg_name in enumerate(self.legs_order):
                foot_geom_id = self.esd.feet_geom_id[leg_name]
                if foot_geom_id is None or foot_geom_id < 0:
                    continue
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    actual_contact[leg_id] = 1
        return actual_contact

    @staticmethod
    def _order_support_polygon(points_xy: np.ndarray) -> np.ndarray:
        points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        if points_xy.shape[0] <= 2:
            return points_xy
        center = np.mean(points_xy, axis=0)
        angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
        ordered = points_xy[np.argsort(angles)]
        signed_area2 = float(
            np.sum(ordered[:, 0] * np.roll(ordered[:, 1], -1) - ordered[:, 1] * np.roll(ordered[:, 0], -1))
        )
        if signed_area2 < 0.0:
            ordered = ordered[::-1]
        return ordered

    @staticmethod
    def _support_margin(points_xy: np.ndarray, query_xy: np.ndarray) -> float:
        points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        query_xy = np.asarray(query_xy, dtype=float).reshape(2)
        if points_xy.shape[0] < 3:
            return 0.0
        polygon = WBInterface._order_support_polygon(points_xy)
        margin = np.inf
        for idx in range(polygon.shape[0]):
            start = polygon[idx]
            end = polygon[(idx + 1) % polygon.shape[0]]
            edge = end - start
            edge_norm = float(np.linalg.norm(edge))
            if edge_norm <= 1e-9:
                continue
            rel = query_xy - start
            signed_distance = float((edge[0] * rel[1] - edge[1] * rel[0]) / edge_norm)
            margin = min(margin, signed_distance)
        if not np.isfinite(margin):
            return 0.0
        return float(margin)

    def _support_margin_query_xy(self, com_pos: np.ndarray, com_vel_xy: np.ndarray | None = None) -> np.ndarray:
        query_xy = np.asarray(com_pos[0:2], dtype=float).reshape(2).copy()
        if com_vel_xy is not None and float(self.support_margin_preview_s) > 1e-9:
            query_xy += float(self.support_margin_preview_s) * np.asarray(com_vel_xy[0:2], dtype=float).reshape(2)
        return query_xy

    def _pre_swing_gate_margin(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        feet_pos: LegsAttr,
        com_pos: np.ndarray,
        com_vel_xy: np.ndarray | None = None,
    ) -> float:
        actual_contact = np.asarray(actual_contact, dtype=int).reshape(4)
        support_legs = [idx for idx in range(4) if idx != leg_id and actual_contact[idx] == 1]
        if len(support_legs) < 3:
            self.last_support_margin[leg_id] = 0.0
            self.last_support_margin_query_xy[leg_id] = self._support_margin_query_xy(com_pos, com_vel_xy)
            return 0.0
        feet_xy = np.vstack(
            [np.asarray(getattr(feet_pos, self.legs_order[idx])[0:2], dtype=float).reshape(2) for idx in support_legs]
        )
        query_xy = self._support_margin_query_xy(com_pos, com_vel_xy)
        margin = self._support_margin(feet_xy, query_xy)
        self.last_support_margin[leg_id] = float(margin)
        self.last_support_margin_query_xy[leg_id] = np.asarray(query_xy, dtype=float).reshape(2)
        return float(margin)

    def _support_contacts_confirmed(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        scheduled_contact: np.ndarray,
    ) -> bool:
        actual_contact = np.asarray(actual_contact, dtype=int).reshape(4)
        scheduled_contact = np.asarray(scheduled_contact, dtype=int).reshape(4)
        support_ids = [idx for idx in range(4) if idx != int(leg_id) and int(scheduled_contact[idx]) == 1]
        if len(support_ids) == 0:
            return False

        actual_ids = [idx for idx in support_ids if int(actual_contact[idx]) == 1]
        required = min(max(int(self.support_confirm_min_contacts), 1), len(support_ids))
        if len(actual_ids) < required:
            return False

        if not bool(self.support_confirm_require_front_rear_span):
            return True

        scheduled_has_front = any(idx < 2 for idx in support_ids)
        scheduled_has_rear = any(idx >= 2 for idx in support_ids)
        actual_has_front = any(idx < 2 for idx in actual_ids)
        actual_has_rear = any(idx >= 2 for idx in actual_ids)
        if scheduled_has_front and scheduled_has_rear:
            return bool(actual_has_front and actual_has_rear)
        if scheduled_has_front:
            return bool(actual_has_front)
        if scheduled_has_rear:
            return bool(actual_has_rear)
        return True

    def get_latched_release_alpha(self, leg_id: int) -> float:
        if int(self.planned_contact[leg_id]) != 0 or int(self.current_contact[leg_id]) != 1:
            return 0.0

        params = getattr(cfg, 'linear_osqp_params', {})
        start = float(np.clip(params.get('latched_release_phase_start', 0.0), 0.0, 1.0))
        end = float(np.clip(params.get('latched_release_phase_end', 1.0), 0.0, 1.0))
        if start <= 0.0 and end >= 1.0:
            return 1.0
        if end <= start:
            return 0.0

        swing_period = max(float(self.stc.swing_period), 1e-6)
        phase = float(np.clip(self.latched_swing_time[leg_id] / swing_period, 0.0, 1.0))
        if phase <= start or phase >= end:
            return 0.0

        mid = 0.5 * (start + end)
        half_width = max(0.5 * (end - start), 1e-6)
        return float(np.clip(1.0 - abs(phase - mid) / half_width, 0.0, 1.0))

    def _latched_contact_horizon_steps(self, leg_id: int, base_latch_steps: int) -> int:
        params = getattr(cfg, 'linear_osqp_params', {})
        start = float(np.clip(params.get('latched_release_phase_start', 0.0), 0.0, 1.0))
        end = float(np.clip(params.get('latched_release_phase_end', 1.0), 0.0, 1.0))
        if base_latch_steps <= 0 or (start <= 0.0 and end >= 1.0) or end <= start:
            return max(int(base_latch_steps), 0)

        swing_period = max(float(self.stc.swing_period), 1e-6)
        phase = float(np.clip(self.latched_swing_time[leg_id] / swing_period, 0.0, 1.0))
        if phase <= start:
            return int(base_latch_steps)
        if phase >= end:
            return 0

        remaining = 1.0 - (phase - start) / max(end - start, 1e-6)
        return int(np.clip(np.floor(base_latch_steps * remaining), 0, base_latch_steps))

    def _should_virtual_unlatch(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        feet_pos: LegsAttr | None = None,
        com_pos: np.ndarray | None = None,
        com_vel_xy: np.ndarray | None = None,
    ) -> bool:
        params = getattr(cfg, 'linear_osqp_params', {})
        threshold = float(params.get('virtual_unlatch_phase_threshold', 1.1))
        if threshold > 1.0 or int(self.planned_contact[leg_id]) != 0 or int(actual_contact[leg_id]) != 1:
            return False

        swing_period = max(float(self.stc.swing_period), 1e-6)
        phase = float(np.clip(self.latched_swing_time[leg_id] / swing_period, 0.0, 1.0))
        if phase < threshold:
            return False

        support_legs = np.flatnonzero(self.planned_contact == 1)
        if support_legs.size == 0:
            return False
        if not bool(np.all(np.asarray(actual_contact, dtype=int)[support_legs] == 1)):
            return False

        required_margin = max(self._pre_swing_gate_required_margin(leg_id), 0.0)
        if required_margin > 1e-9 and feet_pos is not None and com_pos is not None:
            support_margin = self._pre_swing_gate_margin(leg_id, actual_contact, feet_pos, com_pos, com_vel_xy=com_vel_xy)
            if support_margin < required_margin:
                return False
        return True

    def _compute_swing_reference(self, leg_id: int, leg_name: str, touch_down) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        swing_time = float(np.clip(self.latched_swing_time[leg_id], 0.0, self.stc.swing_period))
        des_foot_pos, des_foot_vel, des_foot_acc = self.stc.swing_generator.compute_trajectory_references(
            swing_time,
            self.frg.lift_off_positions[leg_name],
            touch_down,
            self.esd.hitmoments[leg_name],
            self.esd.hitpoints[leg_name],
        )
        return (
            np.asarray(des_foot_pos, dtype=float).reshape(3),
            np.asarray(des_foot_vel, dtype=float).reshape(3),
            np.asarray(des_foot_acc, dtype=float).reshape(3),
        )

    def _compute_latched_swing_torque(
        self,
        leg_id: int,
        leg_name: str,
        q_dot: np.ndarray,
        J: np.ndarray,
        J_dot: np.ndarray,
        touch_down,
        foot_pos: np.ndarray,
        foot_vel: np.ndarray,
        h: np.ndarray,
        mass_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        des_foot_pos, des_foot_vel, des_foot_acc = self._compute_swing_reference(leg_id, leg_name, touch_down)
        err_pos = des_foot_pos - np.asarray(foot_pos, dtype=float).reshape(3)
        err_vel = des_foot_vel - np.asarray(foot_vel, dtype=float).reshape(3)
        acceleration = des_foot_acc + self.stc.position_gain_fb * err_pos + self.stc.velocity_gain_fb * err_vel

        tau_swing = J.T @ (self.stc.position_gain_fb * err_pos + self.stc.velocity_gain_fb * err_vel)
        if self.stc.use_feedback_linearization:
            tau_swing += mass_matrix @ np.linalg.pinv(J) @ (acceleration - J_dot @ q_dot) + h

        return np.asarray(tau_swing, dtype=float).reshape(3), des_foot_pos, des_foot_vel

    def update_state_and_reference(
        self,
        com_pos: np.ndarray,
        base_pos: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ori_euler_xyz: np.ndarray,
        base_ang_vel: np.ndarray,
        feet_pos: LegsAttr,
        hip_pos: LegsAttr,
        joints_pos: LegsAttr,
        heightmaps,
        legs_order: tuple[str, str, str, str],
        simulation_dt: float,
        ref_base_lin_vel: np.ndarray,
        ref_base_ang_vel: np.ndarray,
        mujoco_contact: np.ndarray = None,
    ) -> [dict, dict, list, LegsAttr, list, list, float, bool]:
        """Update the state and reference for the whole body controller, including the contact sequence, footholds, and terrain estimation.

        Args:
            com_pos (np.ndarray): center of mass position in world frame
            base_pos (np.ndarray): base position in world frame
            base_lin_vel (np.ndarray): base linear velocity in world frame
            base_ori_euler_xyz (np.ndarray): base orientation in euler angles in world frame
            base_ang_vel (np.ndarray): base angular velocity in base frame
            feet_pos (LegsAttr): feet positions in world frame
            hip_pos (LegsAttr): hip positions in world frame
            joints_pos (LegsAttr): joint positions
            heightmaps (dict): heightmaps for each leg
            legs_order (tuple[str, str, str, str]): order of the legs
            simulation_dt (float): simulation time step
            ref_base_lin_vel (np.ndarray): reference base linear velocity in world frame
            ref_base_ang_vel (np.ndarray): reference base angular velocity in world frame

        Returns:
            state_current (dict): dictionary of the state of the robot that is used in the mpc
            ref_state (dict):  dictionary of the reference state of the robot that is used in the mpc
            contact_sequence (np.ndarray): this is an array, containing the contact sequence of the robot in the future
            ref_feet_pos (LegsAttr): where to step in world frame
            ref_feet_constraints (LegsAttr): constraints for the footholds in the world frame
            step_height (float): step height
            optimize_swing (bool), boolean to inform that the robot is in the apex, hence we can optimize step freq.
        """

        # Estimate the terrain slope, height, and robot elevation -------------------------------------------------------
        base_pos_measured = np.asarray(base_pos, dtype=float).copy()
        com_pos_measured = np.asarray(com_pos, dtype=float).copy()
        terrain_roll, terrain_pitch, terrain_height, robot_height = self.terrain_computation.compute_terrain_estimation(
            base_position=base_pos,
            yaw=base_ori_euler_xyz[2],
            feet_pos=self.frg.lift_off_positions,
            current_contact=self.current_contact,
        )
        if cfg.mpc_params['type'] == 'linear_osqp':
            # The linear SRBD controller works directly with world-frame body height.
            # Replacing it with the terrain estimator's smoothed "robot_height" causes a
            # large fake height error at startup and pushes the robot into a crouch.
            base_pos = base_pos_measured
            com_pos = com_pos_measured
        else:
            base_pos[2] = robot_height
            com_pos[2] = robot_height #TODO, this is an error


        state_current = dict(
            position=com_pos + self.frg.com_pos_offset_w,  # manual com offset
            # position=base_pos,
            linear_velocity=base_lin_vel,
            orientation=base_ori_euler_xyz,
            angular_velocity=base_ang_vel,
            foot_FL=feet_pos.FL,
            foot_FR=feet_pos.FR,
            foot_RL=feet_pos.RL,
            foot_RR=feet_pos.RR,
            joint_FL=joints_pos.FL,
            joint_FR=joints_pos.FR,
            joint_RL=joints_pos.RL,
            joint_RR=joints_pos.RR,
        )

        # Modulate the desired velocity if the robot is in strange positions
        if self.vm.activated:
            ref_base_lin_vel, ref_base_ang_vel = self.vm.modulate_velocities(
                ref_base_lin_vel, ref_base_ang_vel, feet_pos, hip_pos
            )

        # Update the desired contact sequence ---------------------------
        if self.pgg.start_and_stop_activated:
            # stop the robot for energy efficency if there is no movement and its safe
            # only if activated by an an internal flag for now
            self.pgg.update_start_and_stop(
                feet_pos,
                hip_pos,
                self.frg.hip_offset,
                base_pos,
                base_ori_euler_xyz,
                base_lin_vel,
                base_ang_vel,
                ref_base_lin_vel,
                ref_base_ang_vel,
                self.current_contact,
            )

        startup_full_stance_active = (
            cfg.mpc_params['type'] == 'linear_osqp'
            and self.startup_full_stance_elapsed_s < max(float(self.startup_full_stance_time_s), 0.0)
        )
        if startup_full_stance_active:
            self.startup_full_stance_elapsed_s = min(
                self.startup_full_stance_elapsed_s + float(simulation_dt),
                max(float(self.startup_full_stance_time_s), 0.0),
            )
            contact_sequence = np.ones((4, self.pgg.horizon), dtype=float)
        else:
            self.pgg.run(simulation_dt, self.pgg.step_freq)
            contact_sequence = self.pgg.compute_contact_sequence(
                contact_sequence_dts=self.contact_sequence_dts, contact_sequence_lenghts=self.contact_sequence_lenghts
            )
        contact_sequence_for_mpc = copy.deepcopy(contact_sequence)
        if getattr(contact_sequence, "ndim", 0) == 2 and contact_sequence.shape[0] == 4:
            self.planned_contact = np.array(
                [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]]
            )
        actual_contact = self._foot_contact_from_mujoco(mujoco_contact)
        prev_touchdown_reacquire_active = np.asarray(self.touchdown_reacquire_active, dtype=int).copy()
        self.last_support_margin[:] = np.nan
        self.last_support_margin_query_xy[:] = np.nan
        self.front_late_release_active[:] = 0
        self.support_confirm_active[:] = 0
        self.pre_swing_gate_active[:] = 0
        self.touchdown_reacquire_active[:] = 0
        self.touchdown_confirm_active[:] = 0
        self.touchdown_settle_active[:] = 0
        self.touchdown_support_active[:] = 0
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.front_margin_rescue_active[:] = 0
        self.front_margin_rescue_alpha[:] = 0.0
        self.rear_handoff_support_active = 0
        self.rear_swing_bridge_active = 0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.last_gate_forward_scale = 1.0
        if getattr(contact_sequence, "ndim", 0) == 2 and contact_sequence.shape[0] == 4:
            if cfg.mpc_params['type'] == 'linear_osqp':
                gate_forward_scale = 1.0
                requested_gate_forward_scale = float(
                    np.clip(getattr(cfg, 'linear_osqp_params', {}).get('pre_swing_gate_forward_scale', 1.0), 0.0, 1.0)
                )
                requested_front_release_forward_scale = float(
                    np.clip(
                        getattr(cfg, 'linear_osqp_params', {}).get('front_late_release_forward_scale', 1.0),
                        0.0,
                        1.0,
                    )
                )
                recent_front_rescue_window_s = float(self.front_margin_rescue_recent_swing_window_s)
                for leg_id in range(2):
                    recent_swing_now = bool(
                        int(self.planned_contact[leg_id]) == 0
                        or int(self.current_contact[leg_id]) == 0
                        or not bool(actual_contact[leg_id])
                    )
                    if (not startup_full_stance_active) and recent_front_rescue_window_s > 1e-9 and recent_swing_now:
                        self.front_margin_rescue_recent_swing_remaining_s[leg_id] = recent_front_rescue_window_s
                    else:
                        self.front_margin_rescue_recent_swing_remaining_s[leg_id] = max(
                            0.0,
                            float(self.front_margin_rescue_recent_swing_remaining_s[leg_id]) - float(simulation_dt),
                        )
                for leg_id in range(4):
                    self._pre_swing_gate_margin(
                        leg_id,
                        actual_contact,
                        feet_pos,
                        com_pos_measured,
                        com_vel_xy=base_lin_vel[0:2],
                    )
                full_contact_now = bool(np.all(np.asarray(actual_contact, dtype=int) == 1))
                for leg_id in range(4):
                    scheduled_swing = bool(contact_sequence[leg_id][0] == 0)
                    confirm_hold_s = self._support_contact_confirm_hold_for_leg(leg_id)
                    if not (scheduled_swing and bool(actual_contact[leg_id]) and confirm_hold_s > 1e-9):
                        self.support_contact_confirm_elapsed_s[leg_id] = 0.0
                    else:
                        support_margin = self._pre_swing_gate_margin(
                            leg_id, actual_contact, feet_pos, com_pos_measured, com_vel_xy=base_lin_vel[0:2]
                        )
                        support_ready = self._support_contacts_confirmed(
                            leg_id, actual_contact, np.asarray(contact_sequence[:, 0], dtype=int)
                        )
                        margin_ready = support_margin >= self._pre_swing_gate_required_margin(leg_id)
                        if support_ready and margin_ready:
                            self.support_contact_confirm_elapsed_s[leg_id] = min(
                                self.support_contact_confirm_elapsed_s[leg_id] + float(simulation_dt),
                                float(confirm_hold_s),
                            )
                        else:
                            self.support_contact_confirm_elapsed_s[leg_id] = 0.0

                        if self.support_contact_confirm_elapsed_s[leg_id] + 1e-12 < float(confirm_hold_s):
                            self.support_confirm_active[leg_id] = 1
                            gate_forward_scale = min(gate_forward_scale, float(self.support_confirm_forward_scale))
                            remaining_confirm_s = max(
                                float(confirm_hold_s) - self.support_contact_confirm_elapsed_s[leg_id],
                                0.0,
                            )
                            confirm_steps = max(int(np.floor(remaining_confirm_s / max(self.mpc_dt, 1e-6))) + 1, 1)
                            confirm_steps = min(confirm_steps, contact_sequence.shape[1])
                            contact_sequence[leg_id][0:confirm_steps] = 1
                            continue

                    front_late_release = self._should_front_late_release(
                        leg_id,
                        actual_contact,
                        feet_pos=feet_pos,
                        com_pos=com_pos_measured,
                        com_vel_xy=base_lin_vel[0:2],
                        base_ori_euler_xyz=base_ori_euler_xyz,
                    )
                    if front_late_release:
                        self.front_late_release_active[leg_id] = 1
                        self.pre_swing_gate_elapsed_s[leg_id] = float(self.pre_swing_gate_hold_s)
                        if leg_id < 2:
                            gate_forward_scale = min(gate_forward_scale, requested_front_release_forward_scale)
                        continue
                    if not (
                        full_contact_now
                        and scheduled_swing
                        and bool(actual_contact[leg_id])
                        and float(self.pre_swing_gate_hold_s) > 1e-9
                        and self._pre_swing_gate_required_margin(leg_id) > 1e-9
                    ):
                        self.pre_swing_gate_elapsed_s[leg_id] = 0.0
                        continue

                    support_margin = self._pre_swing_gate_margin(
                        leg_id, actual_contact, feet_pos, com_pos_measured, com_vel_xy=base_lin_vel[0:2]
                    )
                    if support_margin >= self._pre_swing_gate_required_margin(leg_id):
                        self.pre_swing_gate_elapsed_s[leg_id] = 0.0
                        continue
                    if self.pre_swing_gate_elapsed_s[leg_id] >= float(self.pre_swing_gate_hold_s):
                        continue

                    gate_forward_scale = min(gate_forward_scale, requested_gate_forward_scale)
                    self.pre_swing_gate_active[leg_id] = 1
                    self.pre_swing_gate_elapsed_s[leg_id] = min(
                        self.pre_swing_gate_elapsed_s[leg_id] + float(simulation_dt),
                        float(self.pre_swing_gate_hold_s),
                    )
                    remaining_hold_s = max(
                        float(self.pre_swing_gate_hold_s) - self.pre_swing_gate_elapsed_s[leg_id],
                        0.0,
                    )
                    gate_steps = max(int(np.floor(remaining_hold_s / max(self.mpc_dt, 1e-6))) + 1, 1)
                    gate_steps = min(gate_steps, contact_sequence.shape[1])
                    contact_sequence[leg_id][0:gate_steps] = 1
            base_latch_steps = min(self.contact_latch_steps, contact_sequence.shape[1])
            latch_budget_s = float(self.contact_latch_budget_s)
            if cfg.mpc_params['type'] == 'linear_osqp':
                max_latch_steps = self._max_linear_latch_steps()
                base_latch_steps = min(base_latch_steps, max_latch_steps)
                latch_budget_s = self._effective_linear_latch_budget_s(latch_budget_s)
            for leg_id in range(4):
                scheduled_swing = contact_sequence[leg_id][0] == 0
                if not scheduled_swing or not actual_contact[leg_id]:
                    self.contact_latch_elapsed_s[leg_id] = 0.0
                    self.virtual_unlatch_hold_remaining_s[leg_id] = 0.0
                    continue

                if self._should_front_late_release(
                    leg_id,
                    actual_contact,
                    feet_pos=feet_pos,
                    com_pos=com_pos_measured,
                    com_vel_xy=base_lin_vel[0:2],
                    base_ori_euler_xyz=base_ori_euler_xyz,
                ):
                    self.front_late_release_active[leg_id] = 1
                    if latch_budget_s > 0.0:
                        self.contact_latch_elapsed_s[leg_id] = float(latch_budget_s)
                    if leg_id < 2:
                        gate_forward_scale = min(gate_forward_scale, requested_front_release_forward_scale)
                    self.virtual_unlatch_hold_remaining_s[leg_id] = max(
                        float(self.virtual_unlatch_hold_remaining_s[leg_id]),
                        float(self.front_late_release_hold_s),
                    )
                    continue

                if self.virtual_unlatch_hold_remaining_s[leg_id] > 0.0:
                    if leg_id < 2:
                        gate_forward_scale = min(gate_forward_scale, requested_front_release_forward_scale)
                    self.virtual_unlatch_hold_remaining_s[leg_id] = max(
                        0.0, self.virtual_unlatch_hold_remaining_s[leg_id] - float(simulation_dt)
                    )
                    continue

                if self._should_virtual_unlatch(
                    leg_id,
                    actual_contact,
                    feet_pos=feet_pos,
                    com_pos=com_pos_measured,
                    com_vel_xy=base_lin_vel[0:2],
                ):
                    self.virtual_unlatch_hold_remaining_s[leg_id] = float(self.virtual_unlatch_hold_s)
                    continue

                if latch_budget_s > 0.0 and self.contact_latch_elapsed_s[leg_id] >= latch_budget_s:
                    continue

                next_elapsed = self.contact_latch_elapsed_s[leg_id] + float(simulation_dt)
                if latch_budget_s > 0.0:
                    self.contact_latch_elapsed_s[leg_id] = min(next_elapsed, latch_budget_s)
                else:
                    self.contact_latch_elapsed_s[leg_id] = next_elapsed

                leg_latch_steps = self._latched_contact_horizon_steps(leg_id, base_latch_steps)
                if latch_budget_s > 0.0:
                    remaining_budget_s = max(latch_budget_s - self.contact_latch_elapsed_s[leg_id], 0.0)
                    remaining_budget_steps = max(int(np.floor(remaining_budget_s / max(self.mpc_dt, 1e-6))) + 1, 0)
                    leg_latch_steps = min(leg_latch_steps, remaining_budget_steps)
                if leg_latch_steps <= 0:
                    continue

                if actual_contact[leg_id] and scheduled_swing:
                    contact_sequence[leg_id][0:leg_latch_steps] = 1

            for leg_id in range(4):
                if int(self.planned_contact[leg_id]) == 0:
                    self.touchdown_reacquire_armed[leg_id] = 1

            for leg_id in range(4):
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                if not planned_stance:
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                if int(self.touchdown_reacquire_armed[leg_id]) != 1:
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                if bool(actual_contact[leg_id]):
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                hold_s = self._touchdown_reacquire_hold_for_leg(leg_id)
                if hold_s <= 1e-9:
                    continue

                self.touchdown_reacquire_active[leg_id] = 1
                gate_forward_scale = min(gate_forward_scale, float(self.touchdown_reacquire_forward_scale))

                self.touchdown_reacquire_elapsed_s[leg_id] = min(
                    self.touchdown_reacquire_elapsed_s[leg_id] + float(simulation_dt),
                    hold_s,
                )
                remaining_hold_s = max(hold_s - self.touchdown_reacquire_elapsed_s[leg_id], 0.0)
                hold_steps = max(int(np.floor(remaining_hold_s / max(self.mpc_dt, 1e-6))) + 1, 1)
                hold_steps = min(hold_steps, contact_sequence.shape[1])
                contact_sequence[leg_id][0:hold_steps] = 0

            for leg_id in range(4):
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                if not planned_stance or not bool(actual_contact[leg_id]):
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                confirm_hold_s = self._touchdown_confirm_hold_for_leg(leg_id)
                if confirm_hold_s <= 1e-9:
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                front_stance_recontact = bool(
                    self.front_stance_dropout_reacquire
                    and leg_id < 2
                    and (not startup_full_stance_active)
                    and int(self.planned_contact[leg_id]) == 1
                    and int(self.current_contact[leg_id]) == 1
                    and int(self.previous_actual_contact[leg_id]) == 0
                    and bool(actual_contact[leg_id])
                )
                keep_confirm = front_stance_recontact or bool(prev_touchdown_reacquire_active[leg_id]) or bool(
                    self.touchdown_confirm_elapsed_s[leg_id] > 1e-9
                )
                if not keep_confirm:
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                self.touchdown_confirm_active[leg_id] = 1
                gate_forward_scale = min(gate_forward_scale, float(self.touchdown_confirm_forward_scale))
                next_confirm_elapsed = min(
                    self.touchdown_confirm_elapsed_s[leg_id] + float(simulation_dt),
                    confirm_hold_s,
                )
                self.touchdown_confirm_elapsed_s[leg_id] = (
                    0.0 if next_confirm_elapsed >= (confirm_hold_s - 1e-12) else next_confirm_elapsed
                )

            for leg_id in range(4):
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                if not planned_stance or not bool(actual_contact[leg_id]):
                    self.touchdown_settle_remaining_s[leg_id] = 0.0
                    continue

                front_stance_recontact = bool(
                    self.front_stance_dropout_reacquire
                    and leg_id < 2
                    and (not startup_full_stance_active)
                    and int(self.planned_contact[leg_id]) == 1
                    and int(self.current_contact[leg_id]) == 1
                    and int(self.previous_actual_contact[leg_id]) == 0
                    and bool(actual_contact[leg_id])
                )
                if bool(prev_touchdown_reacquire_active[leg_id]) or front_stance_recontact:
                    self.touchdown_settle_remaining_s[leg_id] = max(
                        float(self.touchdown_settle_remaining_s[leg_id]),
                        self._touchdown_settle_hold_for_leg(leg_id),
                    )

                if self.touchdown_settle_remaining_s[leg_id] <= 1e-9:
                    continue

                self.touchdown_settle_active[leg_id] = 1
                gate_forward_scale = min(gate_forward_scale, float(self.touchdown_settle_forward_scale))
                self.touchdown_settle_remaining_s[leg_id] = max(
                    0.0, self.touchdown_settle_remaining_s[leg_id] - float(simulation_dt)
                )

            rescue_candidate_margin = np.full(2, np.nan, dtype=float)
            rescue_candidate_leg = -1
            for leg_id in range(2):
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                if not planned_stance or not bool(actual_contact[leg_id]):
                    self.front_margin_rescue_remaining_s[leg_id] = 0.0
                    self.front_margin_rescue_alpha[leg_id] = 0.0
                    continue

                candidate_margin = self._front_margin_rescue_candidate_margin(
                    leg_id,
                    actual_contact,
                    feet_pos=feet_pos,
                    com_pos=com_pos_measured,
                    com_vel_xy=base_lin_vel[0:2],
                    base_ori_euler_xyz=base_ori_euler_xyz,
                    base_pos_measured=base_pos_measured,
                )
                if candidate_margin is not None:
                    rescue_candidate_margin[leg_id] = float(candidate_margin)

            valid_front_candidates = np.flatnonzero(np.isfinite(rescue_candidate_margin))
            if valid_front_candidates.size > 0:
                rescue_candidate_leg = int(
                    valid_front_candidates[np.argmin(rescue_candidate_margin[valid_front_candidates])]
                )
                rescue_candidate_value = float(rescue_candidate_margin[rescue_candidate_leg])
                gap_ok = True
                other_leg = 1 - rescue_candidate_leg
                if np.isfinite(rescue_candidate_margin[other_leg]) and float(self.front_margin_rescue_margin_gap) > 1e-9:
                    gap_ok = rescue_candidate_value <= float(
                        rescue_candidate_margin[other_leg] - float(self.front_margin_rescue_margin_gap)
                    )
                if rescue_candidate_value <= float(self.front_margin_rescue_min_margin) and gap_ok:
                    self.front_margin_rescue_remaining_s[rescue_candidate_leg] = max(
                        float(self.front_margin_rescue_remaining_s[rescue_candidate_leg]),
                        float(self.front_margin_rescue_hold_s),
                    )
                    severity = float(self.front_margin_rescue_min_margin) - rescue_candidate_value
                    alpha = float(np.clip(severity / float(self.front_margin_rescue_alpha_margin), 0.0, 1.0))
                    self.front_margin_rescue_alpha[rescue_candidate_leg] = max(
                        float(self.front_margin_rescue_alpha[rescue_candidate_leg]),
                        alpha,
                    )

            active_front_rescue_leg = -1
            active_front_rescue_remaining = 0.0
            for leg_id in range(2):
                if float(self.front_margin_rescue_remaining_s[leg_id]) > active_front_rescue_remaining:
                    active_front_rescue_leg = leg_id
                    active_front_rescue_remaining = float(self.front_margin_rescue_remaining_s[leg_id])

            for leg_id in range(2):
                if leg_id != active_front_rescue_leg:
                    self.front_margin_rescue_remaining_s[leg_id] = 0.0
                    self.front_margin_rescue_alpha[leg_id] = 0.0
                    continue
                if self.front_margin_rescue_remaining_s[leg_id] <= 1e-9:
                    self.front_margin_rescue_alpha[leg_id] = 0.0
                    continue

                self.front_margin_rescue_active[leg_id] = 1
                rescue_alpha = float(np.clip(self.front_margin_rescue_alpha[leg_id], 0.0, 1.0))
                rescue_forward_scale = 1.0 - rescue_alpha * (1.0 - float(self.front_margin_rescue_forward_scale))
                gate_forward_scale = min(gate_forward_scale, float(np.clip(rescue_forward_scale, 0.0, 1.0)))
                self.front_margin_rescue_remaining_s[leg_id] = max(
                    0.0,
                    float(self.front_margin_rescue_remaining_s[leg_id]) - float(simulation_dt),
                )

            support_alpha = 0.0
            front_support_alpha = 0.0
            recovery_hold_s = float(self.full_contact_recovery_hold_s)
            if recovery_hold_s > 1e-9:
                roll_mag = abs(float(base_ori_euler_xyz[0]))
                pitch_mag = abs(float(base_ori_euler_xyz[1]))
                ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                height_ratio = float(base_pos_measured[2]) / ref_height
                recent_gate_ok = (
                    float(self.full_contact_recovery_recent_window_s) <= 1e-9
                    or float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                )
                recovery_trigger = (
                    (not startup_full_stance_active)
                    and recent_gate_ok
                    and bool(np.all(np.asarray(actual_contact, dtype=int) == 1))
                    and (
                        roll_mag >= float(self.full_contact_recovery_roll_threshold)
                        or pitch_mag >= float(self.full_contact_recovery_pitch_threshold)
                        or (
                            float(self.full_contact_recovery_height_ratio) > 1e-9
                            and height_ratio <= float(self.full_contact_recovery_height_ratio)
                        )
                    )
                )
                if recovery_trigger:
                    self.full_contact_recovery_remaining_s = max(
                        float(self.full_contact_recovery_remaining_s),
                        recovery_hold_s,
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
            else:
                self.full_contact_recovery_remaining_s = 0.0

            front_margin_rescue_bilateral = bool(np.any(np.asarray(self.front_margin_rescue_active[0:2], dtype=int) == 1))
            front_margin_rescue_alpha_max = float(
                np.clip(np.max(np.asarray(self.front_margin_rescue_alpha[0:2], dtype=float)), 0.0, 1.0)
            )
            for leg_id in range(4):
                touchdown_window_active = int(
                    int(self.touchdown_confirm_active[leg_id]) == 1
                    or int(self.touchdown_settle_active[leg_id]) == 1
                )
                margin_rescue_active = int(int(self.front_margin_rescue_active[leg_id]) == 1)
                support_active = int(touchdown_window_active == 1 or margin_rescue_active == 1)
                if leg_id < 2 and front_margin_rescue_bilateral and bool(actual_contact[leg_id]):
                    support_active = 1
                if (
                    leg_id < 2
                    and int(self.full_contact_recovery_active) == 1
                    and bool(actual_contact[leg_id])
                ):
                    support_active = 1
                self.touchdown_support_active[leg_id] = support_active
                if support_active:
                    leg_support_alpha = 1.0
                    if touchdown_window_active != 1 and leg_id < 2 and front_margin_rescue_bilateral:
                        leg_support_alpha = front_margin_rescue_alpha_max
                    elif touchdown_window_active != 1 and margin_rescue_active == 1:
                        leg_support_alpha = float(np.clip(self.front_margin_rescue_alpha[leg_id], 0.0, 1.0))
                    support_alpha = max(float(support_alpha), float(leg_support_alpha))
                    if leg_id < 2:
                        front_support_alpha = max(float(front_support_alpha), float(leg_support_alpha))
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                if (
                    planned_stance
                    and bool(actual_contact[leg_id])
                    and int(self.touchdown_reacquire_active[leg_id]) != 1
                    and int(self.touchdown_confirm_active[leg_id]) != 1
                    and int(self.touchdown_settle_active[leg_id]) != 1
                ):
                    self.touchdown_reacquire_armed[leg_id] = 0
            handoff_hold_s = float(self.rear_handoff_support_hold_s)
            if handoff_hold_s > 1e-9:
                native_front_support_active = front_support_alpha > 1e-9
                rear_handoff_lookahead = min(
                    max(int(self.rear_handoff_lookahead_steps), 1),
                    int(contact_sequence.shape[1]),
                )
                rear_swing_soon = bool(np.any(contact_sequence[2:4, :rear_handoff_lookahead] == 0))
                rear_transition_active = rear_swing_soon or bool(
                    np.any(np.asarray(self.current_contact[2:4], dtype=int) == 0)
                )
                if (not startup_full_stance_active) and front_support_alpha > 1e-9 and rear_swing_soon:
                    self.rear_handoff_support_mask[:] = 0
                    self.rear_handoff_support_mask[0:2] = np.asarray(
                        self.touchdown_support_active[0:2],
                        dtype=int,
                    )
                    self.rear_handoff_support_remaining_s = max(
                        float(self.rear_handoff_support_remaining_s),
                        handoff_hold_s,
                    )
                if self.rear_handoff_support_remaining_s > 1e-9 and rear_transition_active:
                    self.rear_handoff_support_active = 1
                    if not native_front_support_active:
                        gate_forward_scale = min(gate_forward_scale, float(self.rear_handoff_forward_scale))
                    self.rear_handoff_support_remaining_s = max(
                        0.0,
                        float(self.rear_handoff_support_remaining_s) - float(simulation_dt),
                    )
                    for leg_id in range(2):
                        if int(self.rear_handoff_support_mask[leg_id]) != 1:
                            continue
                        planned_stance = bool(contact_sequence[leg_id][0] == 1)
                        if planned_stance and bool(actual_contact[leg_id]):
                            self.touchdown_support_active[leg_id] = 1
                            support_alpha = 1.0
                            front_support_alpha = 1.0
                elif not rear_transition_active:
                    self.rear_handoff_support_remaining_s = 0.0
                    self.rear_handoff_support_mask[:] = 0
                else:
                    self.rear_handoff_support_remaining_s = 0.0
                    self.rear_handoff_support_mask[:] = 0
            else:
                self.rear_handoff_support_remaining_s = 0.0
                self.rear_handoff_support_mask[:] = 0
            if float(self.rear_swing_bridge_recent_front_window_s) > 1e-9:
                if front_support_alpha > 1e-9:
                    self.rear_swing_bridge_recent_front_remaining_s = float(self.rear_swing_bridge_recent_front_window_s)
                else:
                    self.rear_swing_bridge_recent_front_remaining_s = max(
                        0.0,
                        float(self.rear_swing_bridge_recent_front_remaining_s) - float(simulation_dt),
                    )
            else:
                self.rear_swing_bridge_recent_front_remaining_s = 0.0
            bridge_hold_s = float(self.rear_swing_bridge_hold_s)
            if bridge_hold_s > 1e-9:
                rear_lookahead = min(max(int(self.rear_swing_bridge_lookahead_steps), 1), int(contact_sequence.shape[1]))
                rear_planned_swing = bool(np.any(contact_sequence[2:4, :rear_lookahead] == 0))
                rear_current_swing = bool(np.any(np.asarray(self.current_contact[2:4], dtype=int) == 0))
                rear_pending_recontact = bool(
                    np.any(
                        (np.asarray(self.touchdown_reacquire_armed[2:4], dtype=int) == 1)
                        & (np.asarray(actual_contact[2:4], dtype=int) == 0)
                    )
                )
                recent_front_ok = (
                    front_support_alpha > 1e-9
                    or float(self.rear_swing_bridge_recent_front_remaining_s) > 1e-9
                )
                ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                height_ratio = float(base_pos_measured[2]) / ref_height
                bridge_trigger = (
                    (not startup_full_stance_active)
                    and recent_front_ok
                    and (rear_planned_swing or rear_current_swing or rear_pending_recontact)
                    and (
                        abs(float(base_ori_euler_xyz[0])) >= float(self.rear_swing_bridge_roll_threshold)
                        or abs(float(base_ori_euler_xyz[1])) >= float(self.rear_swing_bridge_pitch_threshold)
                        or (
                            float(self.rear_swing_bridge_height_ratio) > 1e-9
                            and height_ratio <= float(self.rear_swing_bridge_height_ratio)
                        )
                    )
                )
                if bridge_trigger:
                    self.rear_swing_bridge_remaining_s = max(
                        float(self.rear_swing_bridge_remaining_s),
                        bridge_hold_s,
                    )
                if self.rear_swing_bridge_remaining_s > 1e-9:
                    self.rear_swing_bridge_active = 1
                    gate_forward_scale = min(gate_forward_scale, float(self.rear_swing_bridge_forward_scale))
                    self.rear_swing_bridge_remaining_s = max(
                        0.0, float(self.rear_swing_bridge_remaining_s) - float(simulation_dt)
                    )
                else:
                    self.rear_swing_bridge_remaining_s = 0.0
            else:
                self.rear_swing_bridge_remaining_s = 0.0
            if int(self.rear_swing_bridge_active) == 1:
                for leg_id in range(2):
                    planned_stance = bool(contact_sequence[leg_id][0] == 1)
                    if planned_stance and bool(actual_contact[leg_id]):
                        self.touchdown_support_active[leg_id] = 1
                        support_alpha = 1.0
                        front_support_alpha = 1.0
            self.touchdown_support_alpha = float(support_alpha)
            self.front_touchdown_support_alpha = float(front_support_alpha)
            if float(self.full_contact_recovery_recent_window_s) > 1e-9:
                if front_support_alpha > 1e-9:
                    self.front_touchdown_support_recent_remaining_s = float(self.full_contact_recovery_recent_window_s)
                else:
                    self.front_touchdown_support_recent_remaining_s = max(
                        0.0,
                        float(self.front_touchdown_support_recent_remaining_s) - float(simulation_dt),
                    )
            else:
                self.front_touchdown_support_recent_remaining_s = 0.0

            self.last_gate_forward_scale = float(gate_forward_scale)
            if cfg.mpc_params['type'] == 'linear_osqp' and gate_forward_scale < 0.999:
                ref_base_lin_vel = np.array(ref_base_lin_vel, dtype=float, copy=True)
                ref_base_lin_vel[0] *= gate_forward_scale

        self.previous_contact = copy.deepcopy(self.current_contact)
        self.current_contact = np.array(
            [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]]
        )
        self.previous_actual_contact = np.asarray(actual_contact, dtype=int).copy()

        # Compute the reference for the footholds ---------------------------------------------------
        self.frg.update_lift_off_positions(
            self.previous_contact,
            self.current_contact,
            feet_pos,
            legs_order,
            self.pgg.gait_type,
            base_pos,
            base_ori_euler_xyz,
        )
        self.frg.update_touch_down_positions(
            self.previous_contact,
            self.current_contact,
            feet_pos,
            legs_order,
            self.pgg.gait_type,
            base_pos,
            base_ori_euler_xyz,
        )
        ref_feet_pos = self.frg.compute_footholds_reference(
            base_position=base_pos,
            base_ori_euler_xyz=base_ori_euler_xyz,
            base_xy_lin_vel=base_lin_vel[0:2],
            ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
            hips_position=hip_pos,
            com_height_nominal=cfg.simulation_params['ref_z'],
        )

        # Adjust the footholds given the terrain -----------------------------------------------------
        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            time_adaptation = time.time()
            if self.stc.check_apex_condition(self.current_contact, interval=0.01) and self.vfa.initialized == False:
                for leg_id, leg_name in enumerate(legs_order):
                    heightmaps[leg_name].update_height_map(ref_feet_pos[leg_name], yaw=base_ori_euler_xyz[2])
                self.vfa.compute_adaptation(
                    legs_order, ref_feet_pos, hip_pos, heightmaps, base_lin_vel, base_ori_euler_xyz, base_ang_vel
                )
                # print("Adaptation time: ", time.time() - time_adaptation)

            if self.stc.check_full_stance_condition(self.current_contact):
                self.vfa.reset()

            ref_feet_pos, ref_feet_constraints = self.vfa.get_footholds_adapted(ref_feet_pos)
        else:
            ref_feet_constraints = LegsAttr(FL=None, FR=None, RL=None, RR=None)

        

        # Update state reference ------------------------------------------------------------------------

        # Rotate the reference base linear velocity to the terrain frame
        ref_base_lin_vel = R.from_euler("xyz", [terrain_roll, terrain_pitch, 0]).as_matrix() @ ref_base_lin_vel
        if(terrain_pitch > 0.0):
            ref_base_lin_vel[2] = -ref_base_lin_vel[2]
        if(np.abs(terrain_pitch) > 0.2):
            ref_base_lin_vel[0] = ref_base_lin_vel[0]/2.
            ref_base_lin_vel[2] = ref_base_lin_vel[2]*2

        
        ref_pos = np.array([0, 0, cfg.hip_height])
        ref_pos[2] = cfg.simulation_params["ref_z"]# + terrain_height
        # Since the MPC close in CoM position, but usually we have desired height for the base,
        # we modify the reference to bring the base at the desired height and not the CoM
        ref_pos[2] -= base_pos[2] - (com_pos[2] + self.frg.com_pos_offset_w[2])


        if cfg.mpc_params['type'] != 'kinodynamic':
            ref_state = {}
            ref_state |= dict(
                ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                ref_foot_constraints_FL=ref_feet_constraints.FL,
                ref_foot_constraints_FR=ref_feet_constraints.FR,
                ref_foot_constraints_RL=ref_feet_constraints.RL,
                ref_foot_constraints_RR=ref_feet_constraints.RR,
                # Also update the reference base linear velocity and
                ref_linear_velocity=ref_base_lin_vel,
                ref_angular_velocity=ref_base_ang_vel,
                ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                ref_position=ref_pos,
                planned_contact_sequence=copy.deepcopy(contact_sequence_for_mpc),
                latched_release_alpha=np.array(
                    [self.get_latched_release_alpha(leg_id) for leg_id in range(4)],
                    dtype=float,
                ),
            )

        # -------------------------------------------------------------------------------------------------

        if cfg.mpc_params['optimize_step_freq']:
            # we can always optimize the step freq, or just at the apex of the swing
            # to avoid possible jittering in the solution
            optimize_swing = self.stc.check_touch_down_condition(self.current_contact, self.previous_contact, contact_sequence, lookahead=3)
        else:
            optimize_swing = 0

        return state_current, ref_state, contact_sequence, self.step_height, optimize_swing

    def compute_stance_and_swing_torque(
        self,
        simulation_dt: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        feet_jac: LegsAttr,
        feet_jac_dot: LegsAttr,
        feet_pos: LegsAttr,
        feet_vel: LegsAttr,
        legs_qfrc_passive: LegsAttr,
        legs_qfrc_bias: LegsAttr,
        legs_mass_matrix: LegsAttr,
        nmpc_GRFs: LegsAttr,
        nmpc_footholds: LegsAttr,
        legs_qpos_idx: LegsAttr,
        legs_qvel_idx: LegsAttr,
        tau: LegsAttr,
        optimize_swing: int,
        best_sample_freq: float,
        nmpc_joints_pos,
        nmpc_joints_vel,
        nmpc_joints_acc,
        nmpc_predicted_state,
        mujoco_contact: np.ndarray = None,
    ) -> LegsAttr:
        """Compute the stance and swing torque.

        Args:
            simulation_dt (float): simulation time step
            qvel (np.ndarray): joint velocities
            feet_jac (LegsAttr): feet jacobian
            feet_jac_dot (LegsAttr): derivative of the jacobian
            feet_pos (LegsAttr): feet positions in world frame
            feet_vel (LegsAttr): feet velocities in world frame
            legs_qfrc_passive (LegsAttr): passive forces and torques
            legs_qfrc_bias (LegsAttr): joint forces and torques
            legs_mass_matrix (LegsAttr): mass matrix of the legs
            nmpc_GRFs (LegsAttr): ground reaction forces from the MPC in world frame
            nmpc_footholds (LegsAttr): footholds from the MPC in world frame
            legs_qvel_idx (LegsAttr): joint velocities index
            tau (LegsAttr): joint torques
            optimize_swing (int): flag to signal that we need to update the swing trajectory time
            best_sample_freq (float): best sample frequency obtained from the
                                      sampling optimization or the batched ocp

        Returns:
            LegsAttr: joint torques
        """

        # If we have optimized the gait, we set all the timing parameters
        if optimize_swing == 1:
            self.pgg.step_freq = np.array([best_sample_freq])[0]
            self.frg.stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
            swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
            self.stc.regenerate_swing_trajectory_generator(step_height=self.step_height, swing_period=swing_period)

        
        # Update the Early Stance Detector for Reflexes
        self.esd.update_detection(feet_pos, self.last_des_foot_pos, lift_off=self.frg.lift_off_positions, touch_down=nmpc_footholds, 
                        swing_time=self.stc.swing_time, swing_period=self.stc.swing_period, 
                        current_contact=self.current_contact, previous_contact=self.previous_contact, mujoco_contact=mujoco_contact,
                        stc=self.stc)
        actual_contact = self._foot_contact_from_mujoco(mujoco_contact)
        if cfg.mpc_params['type'] == 'linear_osqp':
            for leg_id, leg_name in enumerate(self.legs_order):
                if (
                    int(self.current_contact[leg_id]) == 1
                    and int(self.planned_contact[leg_id]) == 1
                    and bool(actual_contact[leg_id])
                ):
                    anchor_alpha = self._stance_anchor_update_alpha_for_leg(leg_id)
                    if (
                        int(self.touchdown_support_active[leg_id]) == 1
                        or int(self.touchdown_confirm_active[leg_id]) == 1
                        or int(self.touchdown_settle_active[leg_id]) == 1
                    ):
                        anchor_alpha = max(
                            float(anchor_alpha),
                            self._touchdown_support_anchor_update_alpha_for_leg(leg_id),
                        )
                    if anchor_alpha <= 1e-9:
                        continue
                    anchor = np.asarray(self.frg.touch_down_positions[leg_name], dtype=float).copy()
                    foot_now = np.asarray(feet_pos[leg_name], dtype=float).copy()
                    self.frg.touch_down_positions[leg_name] = (
                        (1.0 - anchor_alpha) * anchor + anchor_alpha * foot_now
                    )


        # Compute Stance Torque ---------------------------------------------------------------------------
        tau.FL = -np.matmul(feet_jac.FL[:, legs_qvel_idx.FL].T, nmpc_GRFs.FL)
        tau.FR = -np.matmul(feet_jac.FR[:, legs_qvel_idx.FR].T, nmpc_GRFs.FR)
        tau.RL = -np.matmul(feet_jac.RL[:, legs_qvel_idx.RL].T, nmpc_GRFs.RL)
        tau.RR = -np.matmul(feet_jac.RR[:, legs_qvel_idx.RR].T, nmpc_GRFs.RR)

        self.stc.update_swing_time(self.current_contact, self.legs_order, simulation_dt)
        if cfg.mpc_params['type'] == 'linear_osqp':
            for leg_id in range(4):
                if int(self.planned_contact[leg_id]) == 0 and int(self.current_contact[leg_id]) == 1:
                    self.latched_swing_time[leg_id] = min(
                        self.latched_swing_time[leg_id] + simulation_dt, self.stc.swing_period
                    )
                else:
                    self.latched_swing_time[leg_id] = 0.0

        # Compute Swing Torque ------------------------------------------------------------------------------
        des_foot_pos = LegsAttr(*[np.zeros((3,)) for _ in range(4)])
        des_foot_vel = LegsAttr(*[np.zeros((3,)) for _ in range(4)])

        # The swing controller is in the end-effector space
        for leg_id, leg_name in enumerate(self.legs_order):
            is_latched_swing = False
            lift_ratio = 0.0
            if (
                self.current_contact[leg_id] == 0
            ):  # If in swing phase, compute the swing trajectory tracking control.
                swing_touch_down = np.asarray(nmpc_footholds[leg_name], dtype=float).copy()
                if int(self.touchdown_reacquire_active[leg_id]) == 1:
                    touchdown_ref = np.asarray(self.frg.touch_down_positions[leg_name], dtype=float).copy()
                    xy_blend = self._touchdown_reacquire_xy_blend_for_leg(leg_id)
                    extra_depth = self._touchdown_reacquire_extra_depth_for_leg(leg_id)
                    forward_bias = self._touchdown_reacquire_forward_bias_for_leg(leg_id)
                    if xy_blend > 0.0:
                        swing_touch_down[0:2] = (
                            (1.0 - xy_blend) * swing_touch_down[0:2]
                            + xy_blend * touchdown_ref[0:2]
                        )
                    if abs(float(forward_bias)) > 1e-9:
                        swing_touch_down[0] += float(forward_bias)
                    if extra_depth > 0.0:
                        current_foot_z = float(np.asarray(feet_pos[leg_name], dtype=float)[2])
                        swing_touch_down[2] = min(
                            float(swing_touch_down[2]),
                            float(touchdown_ref[2]),
                            current_foot_z,
                        ) - extra_depth
                tau[leg_name], des_foot_pos[leg_name], des_foot_vel[leg_name] = (
                    self.stc.compute_swing_control_cartesian_space(
                        leg_id=leg_id,
                        q_dot=qvel[legs_qvel_idx[leg_name]],
                        J=feet_jac[leg_name][:, legs_qvel_idx[leg_name]],
                        J_dot=feet_jac_dot[leg_name][:, legs_qvel_idx[leg_name]],
                        lift_off=self.frg.lift_off_positions[leg_name],
                        touch_down=swing_touch_down,
                        foot_pos=feet_pos[leg_name],
                        foot_vel=feet_vel[leg_name],
                        passive_force=legs_qfrc_passive[leg_name],
                        h=legs_qfrc_bias[leg_name],
                        mass_matrix=legs_mass_matrix[leg_name],
                        early_stance_hitmoments=self.esd.hitmoments[leg_name],
                        early_stance_hitpoints=self.esd.hitpoints[leg_name],
                    )
                )
            else:
                if cfg.mpc_params['type'] == 'linear_osqp':
                    # Holding the stance foot near its touchdown location avoids
                    # dragging grounded feet toward the next foothold target.
                    # A tiny xy blend can recover forward progression without
                    # reintroducing the full stance-foot dragging issue.
                    des_foot_pos[leg_name] = np.array(self.frg.touch_down_positions[leg_name], copy=True)
                    if int(self.touchdown_support_active[leg_id]) == 1:
                        anchor_xy_blend = float(
                            np.clip(
                                getattr(cfg, 'linear_osqp_params', {}).get('touchdown_support_anchor_xy_blend', 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        anchor_z_blend = float(
                            np.clip(
                                getattr(cfg, 'linear_osqp_params', {}).get('touchdown_support_anchor_z_blend', 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        actual_foot_pos = np.asarray(feet_pos[leg_name], dtype=float).copy()
                        if anchor_xy_blend > 0.0:
                            des_foot_pos[leg_name][0:2] = (
                                (1.0 - anchor_xy_blend) * des_foot_pos[leg_name][0:2]
                                + anchor_xy_blend * actual_foot_pos[0:2]
                            )
                        if anchor_z_blend > 0.0:
                            des_foot_pos[leg_name][2] = (
                                (1.0 - anchor_z_blend) * float(des_foot_pos[leg_name][2])
                                + anchor_z_blend * float(actual_foot_pos[2])
                            )
                    is_latched_swing = int(self.planned_contact[leg_id]) == 0
                    release_alpha = self.get_latched_release_alpha(leg_id) if is_latched_swing else 0.0
                    stance_blend = float(
                        np.clip(getattr(cfg, 'linear_osqp_params', {}).get('stance_target_blend', 0.0), 0.0, 1.0)
                    )
                    if stance_blend > 0.0:
                        des_foot_pos[leg_name][0:2] = (
                            (1.0 - stance_blend) * des_foot_pos[leg_name][0:2]
                            + stance_blend * np.asarray(nmpc_footholds[leg_name])[0:2]
                        )
                    if is_latched_swing:
                        latched_swing_pos, latched_swing_vel, _ = self._compute_swing_reference(
                            leg_id, leg_name, nmpc_footholds[leg_name]
                        )
                        xy_blend = float(
                            np.clip(
                                getattr(cfg, 'linear_osqp_params', {}).get('latched_swing_xy_blend', 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        lift_ratio = float(
                            np.clip(
                                getattr(cfg, 'linear_osqp_params', {}).get('latched_swing_lift_ratio', 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        if xy_blend > 0.0 or lift_ratio > 0.0:
                            xy_alpha = release_alpha * xy_blend
                            if xy_alpha > 0.0:
                                des_foot_pos[leg_name][0:2] = (
                                    (1.0 - xy_alpha) * des_foot_pos[leg_name][0:2]
                                    + xy_alpha * latched_swing_pos[0:2]
                                )
                                des_foot_vel[leg_name][0:2] = xy_alpha * latched_swing_vel[0:2]
                            lift_alpha = release_alpha * lift_ratio
                            if lift_alpha > 0.0:
                                des_foot_pos[leg_name][2] = (
                                    (1.0 - lift_alpha) * des_foot_pos[leg_name][2]
                                    + lift_alpha * float(latched_swing_pos[2])
                                )
                                des_foot_vel[leg_name][2] = lift_alpha * float(latched_swing_vel[2])
                        tau_blend = float(
                            np.clip(
                                getattr(cfg, 'linear_osqp_params', {}).get('latched_swing_tau_blend', 0.0),
                                0.0,
                                1.0,
                            )
                        )
                        tau_alpha = release_alpha * tau_blend
                        if tau_alpha > 0.0:
                            tau_swing, _, _ = self._compute_latched_swing_torque(
                                leg_id=leg_id,
                                leg_name=leg_name,
                                q_dot=qvel[legs_qvel_idx[leg_name]],
                                J=feet_jac[leg_name][:, legs_qvel_idx[leg_name]],
                                J_dot=feet_jac_dot[leg_name][:, legs_qvel_idx[leg_name]],
                                touch_down=nmpc_footholds[leg_name],
                                foot_pos=feet_pos[leg_name],
                                foot_vel=feet_vel[leg_name],
                                h=legs_qfrc_bias[leg_name],
                                mass_matrix=legs_mass_matrix[leg_name],
                            )
                            tau[leg_name] = (1.0 - tau_alpha) * tau[leg_name] + tau_alpha * tau_swing
                else:
                    des_foot_pos[leg_name] = nmpc_footholds[leg_name]
                if not (cfg.mpc_params['type'] == 'linear_osqp' and is_latched_swing and lift_ratio > 0.0):
                    des_foot_vel[leg_name] = des_foot_vel[leg_name] * 0.0


        self.last_des_foot_pos = des_foot_pos

        # Compensate for friction -------------------------------------------------------------
        if(self.stc.use_friction_compensation): #TODO fix this flag, is not only related to swing
            tau.FL -= legs_qfrc_passive.FL
            tau.FR -= legs_qfrc_passive.FR
            tau.RL -= legs_qfrc_passive.RL
            tau.RR -= legs_qfrc_passive.RR


        # Compute PD targets for the joints ----------------------------------------------------------------
        des_joints_pos = LegsAttr(*[np.zeros((3, 1)) for _ in range(4)])
        des_joints_vel = LegsAttr(*[np.zeros((3, 1)) for _ in range(4)])
        if cfg.mpc_params['type'] != 'kinodynamic':
            qpos_predicted = copy.deepcopy(qpos)
            # TODO use predicted rotation too
            # qpos_predicted[0:3] = nmpc_predicted_state[0:3]
            temp = self.ik.compute_solution(
                qpos_predicted, des_foot_pos.FL, des_foot_pos.FR, des_foot_pos.RL, des_foot_pos.RR
            )

            des_joints_pos.FL = np.array(temp[0:3]).reshape((3,))
            des_joints_pos.FR = np.array(temp[3:6]).reshape((3,))
            des_joints_pos.RL = np.array(temp[6:9]).reshape((3,))
            des_joints_pos.RR = np.array(temp[9:12]).reshape((3,))

            # TODO This should be done over the the desired joint positions jacobian
            des_joints_vel.FL = np.linalg.pinv(feet_jac.FL[:, legs_qvel_idx.FL]) @ des_foot_vel.FL
            des_joints_vel.FR = np.linalg.pinv(feet_jac.FR[:, legs_qvel_idx.FR]) @ des_foot_vel.FR
            des_joints_vel.RL = np.linalg.pinv(feet_jac.RL[:, legs_qvel_idx.RL]) @ des_foot_vel.RL
            des_joints_vel.RR = np.linalg.pinv(feet_jac.RR[:, legs_qvel_idx.RR]) @ des_foot_vel.RR

        else:
            # In the case of the kinodynamic model, we just use the NMPC predicted joints
            des_joints_pos = nmpc_joints_pos
            des_joints_pos = nmpc_joints_vel

        # Saturate of desired joint positions and velocities
        max_joints_pos_difference = 3.0
        max_joints_vel_difference = 10.0

        # Calculate the difference
        actual_joints_pos = LegsAttr(**{leg_name: qpos[legs_qpos_idx[leg_name]] for leg_name in self.legs_order})
        actual_joints_vel = LegsAttr(**{leg_name: qvel[legs_qvel_idx[leg_name]] for leg_name in self.legs_order})

        # Saturate the difference for each leg
        for leg in ["FL", "FR", "RL", "RR"]:
            joints_pos_difference = des_joints_pos[leg] - actual_joints_pos[leg]
            saturated_joints_pos_difference = np.clip(
                joints_pos_difference, -max_joints_pos_difference, max_joints_pos_difference
            )
            des_joints_pos[leg] = actual_joints_pos[leg] + saturated_joints_pos_difference

            joints_vel_difference = des_joints_vel[leg] - actual_joints_vel[leg]
            saturated_joints_vel_difference = np.clip(
                joints_vel_difference, -max_joints_vel_difference, max_joints_vel_difference
            )
            des_joints_vel[leg] = actual_joints_vel[leg] + saturated_joints_vel_difference

        return tau, des_joints_pos, des_joints_vel

    def reset(self, initial_feet_pos: LegsAttr):
        """Reset the whole body interface

        Args:
            initial_feet_pos (LegsAttr): initial feet positions
        """

        self.pgg.reset()
        # self.frg.reset()
        # self.stc.reset()
        # self.terrain_computation.reset()
        self.frg.lift_off_positions = initial_feet_pos
        if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
            self.vfa.reset()
        self.current_contact = np.array([1, 1, 1, 1])
        self.planned_contact = np.array([1, 1, 1, 1])
        self.previous_actual_contact = np.array([0, 0, 0, 0])
        self.latched_swing_time = np.zeros(4, dtype=float)
        self.virtual_unlatch_hold_remaining_s = np.zeros(4, dtype=float)
        self.contact_latch_elapsed_s = np.zeros(4, dtype=float)
        self.pre_swing_gate_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_reacquire_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_settle_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_alpha = np.zeros(4, dtype=float)
        self.front_margin_rescue_recent_swing_remaining_s = np.zeros(2, dtype=float)
        self.touchdown_reacquire_armed = np.zeros(4, dtype=int)
        self.rear_handoff_support_remaining_s = 0.0
        self.rear_handoff_support_active = 0
        self.rear_handoff_support_mask = np.zeros(4, dtype=int)
        self.rear_swing_bridge_remaining_s = 0.0
        self.rear_swing_bridge_active = 0
        self.last_support_margin = np.full(4, np.nan, dtype=float)
        self.last_support_margin_query_xy = np.full((4, 2), np.nan, dtype=float)
        self.front_late_release_active = np.zeros(4, dtype=int)
        self.support_confirm_active = np.zeros(4, dtype=int)
        self.pre_swing_gate_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_active = np.zeros(4, dtype=int)
        self.touchdown_confirm_active = np.zeros(4, dtype=int)
        self.touchdown_settle_active = np.zeros(4, dtype=int)
        self.touchdown_support_active = np.zeros(4, dtype=int)
        self.front_margin_rescue_active = np.zeros(4, dtype=int)
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.full_contact_recovery_remaining_s = 0.0
        self.front_touchdown_support_recent_remaining_s = 0.0
        self.rear_swing_bridge_recent_front_remaining_s = 0.0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.last_gate_forward_scale = 1.0
        self._refresh_linear_timing_params()
        self.startup_full_stance_elapsed_s = 0.0
        return
