import copy
import time
from dataclasses import dataclass

import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr
from scipy.spatial.transform import Rotation as R

from quadruped_pympc import config as cfg


# ---------------------------------------------------------------------------
# Crawl-specific config and state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrawlParams:
    """Crawl-gait-specific configuration parameters."""

    # Swing height / reflex
    rear_disable_reflex_swing: bool = False
    front_swing_height_scale: float = 1.0
    rear_swing_height_scale: float = 1.0
    # Stuck swing release
    front_stuck_swing_release_timeout_s: float = 0.0
    front_stuck_swing_release_height_ratio: float = 0.0
    front_stuck_swing_release_roll_threshold: float = float('inf')
    front_stuck_swing_release_pitch_threshold: float = float('inf')
    # Delayed swing recovery
    front_delayed_swing_recovery_hold_s: float = 0.0
    front_delayed_swing_recovery_margin_threshold: float = 0.0
    front_delayed_swing_recovery_once_per_swing: bool = False
    front_delayed_swing_recovery_release_tail_s: float = 0.0
    front_delayed_swing_recovery_rearm_trigger_s: float = 0.0
    # Planted swing recovery
    front_planted_swing_recovery_hold_s: float = 0.0
    front_planted_swing_recovery_margin_threshold: float = 0.0
    front_planted_swing_recovery_height_ratio: float = 0.0
    front_planted_swing_recovery_roll_threshold: object = None
    front_planted_swing_recovery_rearm_trigger_s: float = 0.0
    front_planted_postdrop_recovery_hold_s: float = 0.0
    # Planted seam support
    front_planted_seam_support_hold_s: float = 0.0
    front_planted_seam_keep_swing: bool = False
    # Stance support tail
    front_stance_support_tail_hold_s: float = 0.0
    front_stance_support_tail_forward_scale: float = 1.0
    # Close gap
    front_close_gap_support_hold_s: float = 0.0
    front_close_gap_keep_swing: bool = False
    # Late rearm
    front_late_rearm_tail_hold_s: float = 0.0
    front_late_rearm_budget_s: float = 0.0
    front_late_rearm_min_swing_time_s: float = 0.0
    front_late_rearm_min_negative_margin: float = 0.0
    # Weak rear
    front_planted_weak_rear_share_ref: float = 0.0
    front_planted_weak_rear_alpha_cap: float = 1.0


@dataclass
class CrawlState:
    """Crawl-gait-specific mutable runtime state (reset each episode)."""

    front_stance_support_tail_remaining_s: float = 0.0
    front_planted_seam_support_remaining_s: float = 0.0
    front_planted_seam_support_alpha: float = 0.0


# Spec for table-driven param parsing: (field_name, param_key, parse_kind)
# parse_kind: 'nonneg' = max(float,0), 'float' = float, 'bool' = bool,
#             'clip01' = clip(0,1), 'clip02' = clip(0,2),
#             'opt_inf' = inf-if-None else max(float,0),
#             'opt_none' = None-if-None else max(float,0)
_CRAWL_PARAM_SPEC = (
    ('rear_disable_reflex_swing', 'rear_crawl_disable_reflex_swing', 'bool'),
    ('front_swing_height_scale', 'front_crawl_swing_height_scale', 'clip02'),
    ('rear_swing_height_scale', 'rear_crawl_swing_height_scale', 'clip02'),
    ('front_stuck_swing_release_timeout_s', 'crawl_front_stuck_swing_release_timeout_s', 'nonneg'),
    ('front_stuck_swing_release_height_ratio', 'crawl_front_stuck_swing_release_height_ratio', 'nonneg'),
    ('front_stuck_swing_release_roll_threshold', 'crawl_front_stuck_swing_release_roll_threshold', 'opt_inf'),
    ('front_stuck_swing_release_pitch_threshold', 'crawl_front_stuck_swing_release_pitch_threshold', 'opt_inf'),
    ('front_delayed_swing_recovery_hold_s', 'crawl_front_delayed_swing_recovery_hold_s', 'nonneg'),
    ('front_delayed_swing_recovery_margin_threshold', 'crawl_front_delayed_swing_recovery_margin_threshold', 'float'),
    ('front_delayed_swing_recovery_once_per_swing', 'crawl_front_delayed_swing_recovery_once_per_swing', 'bool'),
    ('front_delayed_swing_recovery_release_tail_s', 'crawl_front_delayed_swing_recovery_release_tail_s', 'nonneg'),
    ('front_delayed_swing_recovery_rearm_trigger_s', 'crawl_front_delayed_swing_recovery_rearm_trigger_s', 'nonneg'),
    ('front_planted_swing_recovery_hold_s', 'crawl_front_planted_swing_recovery_hold_s', 'nonneg'),
    ('front_planted_swing_recovery_margin_threshold', 'crawl_front_planted_swing_recovery_margin_threshold', 'float'),
    ('front_planted_swing_recovery_height_ratio', 'crawl_front_planted_swing_recovery_height_ratio', 'nonneg'),
    ('front_planted_swing_recovery_roll_threshold', 'crawl_front_planted_swing_recovery_roll_threshold', 'opt_none'),
    ('front_planted_swing_recovery_rearm_trigger_s', 'crawl_front_planted_swing_recovery_rearm_trigger_s', 'nonneg'),
    ('front_planted_postdrop_recovery_hold_s', 'crawl_front_planted_postdrop_recovery_hold_s', 'nonneg'),
    ('front_planted_seam_support_hold_s', 'crawl_front_planted_seam_support_hold_s', 'nonneg'),
    ('front_planted_seam_keep_swing', 'crawl_front_planted_seam_keep_swing', 'bool'),
    ('front_stance_support_tail_hold_s', 'crawl_front_stance_support_tail_hold_s', 'nonneg'),
    ('front_stance_support_tail_forward_scale', 'crawl_front_stance_support_tail_forward_scale', 'clip01'),
    ('front_close_gap_support_hold_s', 'crawl_front_close_gap_support_hold_s', 'nonneg'),
    ('front_close_gap_keep_swing', 'crawl_front_close_gap_keep_swing', 'bool'),
    ('front_late_rearm_tail_hold_s', 'crawl_front_late_rearm_tail_hold_s', 'nonneg'),
    ('front_late_rearm_budget_s', 'crawl_front_late_rearm_budget_s', 'nonneg'),
    ('front_late_rearm_min_swing_time_s', 'crawl_front_late_rearm_min_swing_time_s', 'nonneg'),
    ('front_late_rearm_min_negative_margin', 'crawl_front_late_rearm_min_negative_margin', 'nonneg'),
    ('front_planted_weak_rear_share_ref', 'crawl_front_planted_weak_rear_share_ref', 'nonneg'),
    ('front_planted_weak_rear_alpha_cap', 'crawl_front_planted_weak_rear_alpha_cap', 'clip01'),
)
from quadruped_pympc.helpers.foothold_reference_generator import FootholdReferenceGenerator
#from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_adam import InverseKinematicsNumeric
from quadruped_pympc.helpers.inverse_kinematics.inverse_kinematics_numeric_mujoco import InverseKinematicsNumeric
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator
from quadruped_pympc.helpers.velocity_modulator import VelocityModulator
from quadruped_pympc.helpers.early_stance_detector import EarlyStanceDetector
from quadruped_pympc.helpers.rear_transition_manager import RearTransitionManager
from quadruped_pympc.interfaces.crawl_recovery import CrawlRecoveryMixin
from quadruped_pympc.interfaces.linear_timing_params import LinearTimingParamsMixin

if cfg.simulation_params['visual_foothold_adaptation'] != 'blind':
    from quadruped_pympc.helpers.visual_foothold_adaptation import VisualFootholdAdaptation


class WBInterface(CrawlRecoveryMixin, LinearTimingParamsMixin):
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
        self.gait_name = gait_name
        self.is_dynamic_gait = gait_name in ('trot', 'pace', 'bound')
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
            stance_time=stance_time,
            hip_height=cfg.hip_height,
            lift_off_positions=initial_feet_pos,
            # The z-freeze foothold anchoring was introduced to stabilize
            # dynamic trot turns. In crawl it over-constrains stance/swing
            # world-z updates and makes the late rear seam much harder to
            # recover, so keep it limited to dynamic gaits.
            freeze_world_z_during_contact_phases=(
                cfg.mpc_params['type'] == 'linear_osqp' and self.is_dynamic_gait
            ),
            yaw_rate_compensation_scale=0.0,
            yaw_error_compensation_scale=0.0,
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
        self.last_des_foot_vel = LegsAttr(*[np.zeros((3,)) for _ in range(4)])
        

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
        self.front_contact_latch_steps = 6
        self.front_contact_latch_budget_s = 0.0
        self.rear_contact_latch_steps = 6
        self.rear_contact_latch_budget_s = 0.0
        self.virtual_unlatch_hold_s = 0.0
        self.pre_swing_gate_min_margin = 0.0
        self.front_pre_swing_gate_min_margin = 0.0
        self.rear_pre_swing_gate_min_margin = 0.0
        self.rear_pre_swing_gate_hold_s = 0.0
        self.swing_contact_release_timeout_s = 0.0
        self.front_swing_contact_release_timeout_s = 0.0
        self.rear_swing_contact_release_timeout_s = 0.0
        self.front_release_lift_height = 0.0
        self.front_release_lift_kp = 0.0
        self.front_release_lift_kd = 0.0
        self.rear_release_lift_height = 0.0
        self.rear_release_lift_kp = 0.0
        self.rear_release_lift_kd = 0.0
        self.support_contact_confirm_hold_s = 0.0
        self.front_support_contact_confirm_hold_s = 0.0
        self.rear_support_contact_confirm_hold_s = 0.0
        self.support_confirm_min_contacts = 3
        self.support_confirm_require_front_rear_span = True
        self.support_confirm_forward_scale = 1.0
        self.front_stance_dropout_reacquire = False
        self.front_stance_dropout_support_hold_s = 0.0
        self.front_stance_dropout_support_forward_scale = 1.0
        self.rear_stance_dropout_reacquire = False
        self.front_late_release_phase_threshold = 1.1
        self.front_late_release_min_margin = 0.0
        self.front_late_release_hold_s = 0.0
        self.front_late_release_extra_margin = 0.0
        self.front_late_release_pitch_guard = np.inf
        self.front_late_release_roll_guard = np.inf
        self.support_margin_preview_s = 0.0
        self.touchdown_reacquire_hold_s = 0.0
        self.front_touchdown_reacquire_hold_s = 0.0
        self.rear_touchdown_reacquire_hold_s = 0.0
        self.touchdown_reacquire_forward_scale = 1.0
        self.touchdown_reacquire_xy_blend = 0.0
        self.front_touchdown_reacquire_xy_blend = 0.0
        self.rear_touchdown_reacquire_xy_blend = 0.0
        self.touchdown_reacquire_extra_depth = 0.0
        self.front_touchdown_reacquire_extra_depth = 0.0
        self.rear_touchdown_reacquire_extra_depth = 0.0
        self.touchdown_reacquire_forward_bias = 0.0
        self.front_touchdown_reacquire_forward_bias = 0.0
        self.rear_touchdown_reacquire_forward_bias = 0.0
        self.rear_touchdown_reacquire_force_until_contact = False
        self.rear_touchdown_reacquire_min_swing_time_s = 0.0
        self.front_touchdown_reacquire_hold_current_xy = False
        self.rear_touchdown_reacquire_hold_current_xy = False
        self.rear_touchdown_reacquire_max_xy_shift = 0.0
        self.rear_touchdown_reacquire_min_phase = 0.0
        self.rear_touchdown_reacquire_upward_vel_damping = 0.0
        self.rear_touchdown_retry_descent_depth = 0.0
        self.rear_touchdown_retry_descent_kp = 0.0
        self.rear_touchdown_retry_descent_kd = 0.0
        self.rear_touchdown_contact_debounce_s = 0.0
        self.rear_touchdown_contact_min_phase = 0.0
        self.rear_touchdown_contact_max_upward_vel = np.inf
        self.rear_touchdown_contact_min_grf_z = 0.0
        self.rear_touchdown_close_lock_hold_s = 0.0
        self.crawl_params = CrawlParams()
        self.crawl_state = CrawlState()
        self.stance_anchor_update_alpha = 0.0
        self.front_stance_anchor_update_alpha = 0.0
        self.rear_stance_anchor_update_alpha = 0.0
        self.touchdown_support_anchor_update_alpha = 0.0
        self.front_touchdown_support_anchor_update_alpha = 0.0
        self.rear_touchdown_support_anchor_update_alpha = 0.0
        self.touchdown_confirm_hold_s = 0.0
        self.front_touchdown_confirm_hold_s = 0.0
        self.rear_touchdown_confirm_hold_s = 0.0
        self.rear_touchdown_confirm_keep_swing = False
        self.touchdown_confirm_forward_scale = 1.0
        self.touchdown_settle_hold_s = 0.0
        self.front_touchdown_settle_hold_s = 0.0
        self.rear_touchdown_settle_hold_s = 0.0
        self.touchdown_settle_forward_scale = 1.0
        self.rear_post_touchdown_support_hold_s = 0.0
        self.rear_post_touchdown_support_forward_scale = 1.0
        self.rear_post_touchdown_support_height_ratio = 0.0
        self.rear_post_touchdown_support_roll_threshold = np.inf
        self.rear_post_touchdown_support_pitch_threshold = np.inf
        self.rear_post_touchdown_support_min_grf_z = 0.0
        self.rear_post_touchdown_support_min_rear_load_share = 0.0
        self.front_rear_transition_guard_hold_s = 0.0
        self.front_rear_transition_guard_forward_scale = 1.0
        self.front_rear_transition_guard_roll_threshold = np.inf
        self.front_rear_transition_guard_pitch_threshold = np.inf
        self.front_rear_transition_guard_height_ratio = 0.0
        self.front_rear_transition_guard_release_tail_s = 0.0
        self.front_rear_transition_guard_margin_release = 0.0
        self.front_rear_transition_guard_post_recovery_hold_s = 0.0
        self.rear_pre_swing_guard_roll_threshold = np.inf
        self.rear_pre_swing_guard_pitch_threshold = np.inf
        self.rear_pre_swing_guard_height_ratio = 0.0
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
        self.full_contact_recovery_rear_support_scale = 0.0
        self.pre_swing_gate_hold_s = 0.0
        self.startup_full_stance_elapsed_s = 0.0
        self.contact_latch_elapsed_s = np.zeros(4, dtype=float)
        self.pre_swing_gate_elapsed_s = np.zeros(4, dtype=float)
        self.swing_contact_release_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_wait_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_bypass_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_settle_remaining_s = np.zeros(4, dtype=float)
        self.rear_touchdown_actual_contact_elapsed_s = np.zeros(4, dtype=float)
        self.rear_late_seam_elapsed_s = np.zeros(4, dtype=float)
        self.rear_close_handoff_remaining_s = np.zeros(4, dtype=float)
        self.rear_close_handoff_alpha_scale = np.zeros(4, dtype=float)
        self.rear_late_load_share_remaining_s = np.zeros(4, dtype=float)
        self.rear_late_load_share_trigger_elapsed_s = np.zeros(4, dtype=float)
        self.rear_touchdown_close_lock_remaining_s = np.zeros(4, dtype=float)
        self.rear_touchdown_retry_count = np.zeros(4, dtype=int)
        self.rear_stable_stance_elapsed_s = np.zeros(4, dtype=float)
        self.previous_feet_pos_world = np.zeros((4, 3), dtype=float)
        self.previous_feet_pos_world_valid = False
        self.front_margin_rescue_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_alpha = np.zeros(4, dtype=float)
        self.front_margin_rescue_recent_swing_remaining_s = np.zeros(2, dtype=float)
        self.touchdown_reacquire_armed = np.zeros(4, dtype=int)
        self.rear_handoff_support_remaining_s = 0.0
        self.rear_handoff_support_active = 0
        self.rear_handoff_support_mask = np.zeros(4, dtype=int)
        self.rear_swing_bridge_remaining_s = 0.0
        self.rear_swing_bridge_active = 0
        self.rear_swing_release_support_remaining_s = 0.0
        self.rear_swing_release_support_active = 0
        self.full_contact_recovery_remaining_s = 0.0
        self.front_rear_transition_guard_post_recovery_remaining_s = 0.0
        self.rear_all_contact_post_recovery_remaining_s = 0.0
        self.rear_all_contact_post_recovery_alpha_scale = 1.0
        self.rear_all_contact_front_planted_tail_remaining_s = 0.0
        self.rear_all_contact_front_planted_tail_alpha_scale = 1.0
        self.front_delayed_swing_recovery_spent = np.zeros(2, dtype=int)
        self.front_planted_swing_recovery_spent = np.zeros(2, dtype=int)
        self.front_late_rearm_used_s = np.zeros(2, dtype=float)
        self.front_stance_dropout_support_remaining_s = np.zeros(2, dtype=float)
        self.front_touchdown_support_recent_remaining_s = 0.0
        self.rear_swing_bridge_recent_front_remaining_s = 0.0
        self.last_support_margin = np.full(4, np.nan, dtype=float)
        self.last_support_margin_query_xy = np.full((4, 2), np.nan, dtype=float)
        self.front_late_release_active = np.zeros(4, dtype=int)
        self.swing_contact_release_active = np.zeros(4, dtype=int)
        self.support_confirm_active = np.zeros(4, dtype=int)
        self.pre_swing_gate_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_active = np.zeros(4, dtype=int)
        self.touchdown_confirm_active = np.zeros(4, dtype=int)
        self.touchdown_settle_active = np.zeros(4, dtype=int)
        self.touchdown_support_active = np.zeros(4, dtype=int)
        self.rear_touchdown_pending_confirm = np.zeros(4, dtype=int)
        self.rear_retry_contact_signal_debug = np.zeros(4, dtype=int)
        self.rear_touchdown_contact_ready_debug = np.zeros(4, dtype=int)
        self.rear_late_stance_contact_ready_debug = np.zeros(4, dtype=int)
        self.rear_all_contact_support_needed_debug = np.zeros(4, dtype=int)
        self.rear_late_seam_support_active_debug = np.zeros(4, dtype=int)
        self.rear_close_handoff_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_alpha_debug = np.zeros(4, dtype=float)
        self.rear_late_load_share_candidate_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_candidate_alpha_debug = np.zeros(4, dtype=float)
        self.rear_late_load_share_trigger_enabled_debug = 0
        self.full_contact_recovery_trigger_debug = 0
        self.front_delayed_swing_recovery_trigger_debug = 0
        self.planted_front_recovery_trigger_debug = 0
        self.planted_front_postdrop_recovery_trigger_debug = 0
        self.front_close_gap_trigger_debug = 0
        self.front_late_rearm_trigger_debug = 0
        self.front_planted_posture_tail_trigger_debug = 0
        self.front_late_posture_tail_trigger_debug = 0
        self.front_margin_rescue_active = np.zeros(4, dtype=int)
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.rear_touchdown_support_alpha = 0.0
        self.rear_all_contact_stabilization_alpha = 0.0
        self.rear_all_contact_front_planted_tail_alpha = 0.0
        self.rear_all_contact_weak_leg_alpha = 0.0
        self.rear_all_contact_weak_leg_index = -1
        self.rear_close_handoff_alpha = 0.0
        self.rear_close_handoff_leg_index = -1
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        self.rear_all_contact_stabilization_min_rear_leg_load_share = 0.0
        self.rear_all_contact_stabilization_weak_leg_share_ref = 0.0
        self.rear_all_contact_stabilization_weak_leg_height_ratio = 0.0
        self.rear_all_contact_stabilization_weak_leg_tail_only = False
        self.rear_all_contact_stabilization_retrigger_limit = 0
        self.rear_all_contact_post_recovery_tail_hold_s = 0.0
        self.rear_all_contact_release_tail_alpha_scale = 1.0
        self.rear_late_seam_support_trigger_s = 0.0
        self.rear_close_handoff_hold_s = 0.0
        self.rear_late_load_share_support_hold_s = 0.0
        self.rear_late_load_share_support_min_leg_share = 0.0
        self.rear_late_load_share_support_height_ratio = 0.0
        self.rear_late_load_share_support_min_persist_s = 0.0
        self.rear_late_load_share_support_alpha_cap = 1.0
        self.touchdown_contact_vel_z_damping = 0.0
        self.front_touchdown_contact_vel_z_damping = 0.0
        self.rear_touchdown_contact_vel_z_damping = 0.0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.last_gate_forward_scale = 1.0

        self.current_contact = np.array([1, 1, 1, 1])
        self.previous_contact = np.array([1, 1, 1, 1])
        self.planned_contact = np.array([1, 1, 1, 1])
        self.previous_actual_contact = np.array([0, 0, 0, 0])
        self.latched_swing_time = np.zeros(4, dtype=float)
        self.virtual_unlatch_hold_remaining_s = np.zeros(4, dtype=float)
        self.rear_transition_manager = RearTransitionManager(self.mpc_dt)
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

    # _refresh_linear_timing_params is inherited from LinearTimingParamsMixin

    def _configure_rear_transition_manager(self) -> None:
        self.rear_transition_manager.configure(
            enabled=(cfg.mpc_params['type'] == 'linear_osqp'),
            contact_debounce_s=self.rear_touchdown_contact_debounce_s,
            contact_min_phase=self.rear_touchdown_contact_min_phase,
            contact_max_upward_vel=self.rear_touchdown_contact_max_upward_vel,
            contact_min_grf_z=self.rear_touchdown_contact_min_grf_z,
            reacquire_hold_s=self.rear_touchdown_reacquire_hold_s,
            reacquire_min_swing_time_s=self.rear_touchdown_reacquire_min_swing_time_s,
            reacquire_forward_scale=self.touchdown_reacquire_forward_scale,
            confirm_hold_s=self.rear_touchdown_confirm_hold_s,
            confirm_forward_scale=self.touchdown_confirm_forward_scale,
            settle_hold_s=self.rear_touchdown_settle_hold_s,
            settle_forward_scale=self.touchdown_settle_forward_scale,
            post_support_hold_s=self.rear_post_touchdown_support_hold_s,
            post_support_forward_scale=self.rear_post_touchdown_support_forward_scale,
            post_support_height_ratio=self.rear_post_touchdown_support_height_ratio,
            post_support_roll_threshold=self.rear_post_touchdown_support_roll_threshold,
            post_support_pitch_threshold=self.rear_post_touchdown_support_pitch_threshold,
            post_support_min_grf_z=self.rear_post_touchdown_support_min_grf_z,
            post_support_min_rear_load_share=self.rear_post_touchdown_support_min_rear_load_share,
            all_contact_stabilization_hold_s=max(
                float(getattr(self, 'rear_all_contact_stabilization_hold_s', 0.0)),
                0.0,
            ),
            all_contact_stabilization_forward_scale=float(
                np.clip(getattr(self, 'rear_all_contact_stabilization_forward_scale', 1.0), 0.0, 1.0)
            ),
            all_contact_stabilization_front_alpha_scale=float(
                np.clip(getattr(self, 'rear_all_contact_stabilization_front_alpha_scale', 1.0), 0.0, 1.0)
            ),
            all_contact_stabilization_height_ratio=max(
                float(getattr(self, 'rear_all_contact_stabilization_height_ratio', 0.0)),
                0.0,
            ),
            all_contact_stabilization_roll_threshold=getattr(
                self,
                'rear_all_contact_stabilization_roll_threshold',
                None,
            ),
            all_contact_stabilization_pitch_threshold=getattr(
                self,
                'rear_all_contact_stabilization_pitch_threshold',
                None,
            ),
            all_contact_stabilization_min_rear_load_share=max(
                float(getattr(self, 'rear_all_contact_stabilization_min_rear_load_share', 0.0)),
                0.0,
            ),
            all_contact_stabilization_min_rear_leg_load_share=max(
                float(getattr(self, 'rear_all_contact_stabilization_min_rear_leg_load_share', 0.0)),
                0.0,
            ),
            all_contact_stabilization_retrigger_limit=max(
                int(getattr(self, 'rear_all_contact_stabilization_retrigger_limit', 0)),
                0,
            ),
            front_transition_guard_hold_s=self.front_rear_transition_guard_hold_s,
            front_transition_guard_forward_scale=self.front_rear_transition_guard_forward_scale,
            front_transition_guard_roll_threshold=self.front_rear_transition_guard_roll_threshold,
            front_transition_guard_pitch_threshold=self.front_rear_transition_guard_pitch_threshold,
            front_transition_guard_height_ratio=self.front_rear_transition_guard_height_ratio,
            front_transition_guard_release_tail_s=self.front_rear_transition_guard_release_tail_s,
            pre_swing_guard_roll_threshold=self.rear_pre_swing_guard_roll_threshold,
            pre_swing_guard_pitch_threshold=self.rear_pre_swing_guard_pitch_threshold,
            pre_swing_guard_height_ratio=self.rear_pre_swing_guard_height_ratio,
            confirm_keep_swing=self.rear_touchdown_confirm_keep_swing,
        )

    def _sync_rear_transition_debug_arrays(self) -> None:
        self.rear_transition_manager.sync_debug_arrays(
            target_elapsed_s=self.rear_touchdown_actual_contact_elapsed_s,
            target_pending_confirm=self.rear_touchdown_pending_confirm,
        )

    def _clear_touchdown_reacquire_state(
        self,
        leg_id: int,
        *,
        clear_retry_count: bool = False,
    ) -> None:
        self.touchdown_reacquire_armed[leg_id] = 0
        self.touchdown_reacquire_active[leg_id] = 0
        self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
        if int(leg_id) >= 2:
            self.rear_transition_manager.clear_pending_confirm(leg_id)
            if clear_retry_count:
                self.rear_touchdown_retry_count[leg_id] = 0

    def _retire_rear_reacquire_after_stable_stance(
        self,
        actual_contact: np.ndarray,
        simulation_dt: float,
    ) -> None:
        hold_s = float(self.rear_touchdown_reacquire_retire_stance_hold_s)
        if hold_s <= 1e-9:
            self.rear_stable_stance_elapsed_s[2:4] = 0.0
            return
        for leg_id in range(2, 4):
            stable_planned_stance = int(self.planned_contact[leg_id]) == 1
            stable_controller_stance = int(self.current_contact[leg_id]) == 1
            stable_actual_stance = bool(actual_contact[leg_id])
            if stable_planned_stance and stable_controller_stance and stable_actual_stance:
                self.rear_stable_stance_elapsed_s[leg_id] += float(simulation_dt)
            else:
                self.rear_stable_stance_elapsed_s[leg_id] = 0.0
                continue
            if float(self.rear_stable_stance_elapsed_s[leg_id]) < hold_s:
                continue
            self._clear_touchdown_reacquire_state(leg_id, clear_retry_count=True)

    def _arm_rear_touchdown_close_lock(self, leg_id: int) -> None:
        if int(leg_id) < 2:
            return
        hold_s = float(self.rear_touchdown_close_lock_hold_s)
        if hold_s <= 1e-9:
            return
        self.rear_touchdown_close_lock_remaining_s[leg_id] = max(
            float(self.rear_touchdown_close_lock_remaining_s[leg_id]),
            hold_s,
        )

    def _pre_swing_gate_required_margin(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_pre_swing_gate_min_margin)
        return float(self.rear_pre_swing_gate_min_margin)

    def _pre_swing_gate_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.pre_swing_gate_hold_s)
        return float(self.rear_pre_swing_gate_hold_s)

    def _touchdown_reacquire_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_hold_s)
        return float(self.rear_touchdown_reacquire_hold_s)

    def _swing_contact_release_timeout_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_swing_contact_release_timeout_s)
        return float(self.rear_swing_contact_release_timeout_s)

    def _support_contact_confirm_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_support_contact_confirm_hold_s)
        return float(self.rear_support_contact_confirm_hold_s)

    def _support_contact_confirm_max_wait_for_leg(self, leg_id: int) -> float:
        # Support-confirm should help avoid premature lift-off, but it should
        # not block the whole swing window indefinitely if the confirmation
        # never fully clears. Reuse the pre-swing hold as extra wait budget.
        confirm_hold_s = self._support_contact_confirm_hold_for_leg(leg_id)
        extra_wait_s = max(float(self._pre_swing_gate_hold_for_leg(leg_id)), 0.0)
        return max(float(confirm_hold_s) + extra_wait_s, float(confirm_hold_s))

    def _stance_anchor_update_alpha_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_stance_anchor_update_alpha)
        return float(self.rear_stance_anchor_update_alpha)

    def _touchdown_support_anchor_update_alpha_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_support_anchor_update_alpha)
        return float(self.rear_touchdown_support_anchor_update_alpha)

    def _touchdown_reacquire_xy_blend_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_xy_blend)
        return float(self.rear_touchdown_reacquire_xy_blend)

    def _touchdown_reacquire_extra_depth_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_extra_depth)
        return float(self.rear_touchdown_reacquire_extra_depth)

    def _touchdown_reacquire_forward_bias_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_reacquire_forward_bias)
        return float(self.rear_touchdown_reacquire_forward_bias)

    def _touchdown_confirm_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_confirm_hold_s)
        return float(self.rear_touchdown_confirm_hold_s)

    def _touchdown_settle_hold_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_settle_hold_s)
        return float(self.rear_touchdown_settle_hold_s)

    def _touchdown_contact_ready_for_leg(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        *,
        foot_grf_world: np.ndarray | None = None,
        current_foot_vz: float | None = None,
        swing_phase: float | None = None,
        waiting_for_recontact: bool = False,
    ) -> bool:
        if int(leg_id) < 2:
            return bool(actual_contact[leg_id])
        return self._rear_touchdown_contact_ready(
            leg_id,
            actual_contact,
            foot_grf_world=foot_grf_world,
            current_foot_vz=current_foot_vz,
            swing_phase=swing_phase,
            waiting_for_recontact=waiting_for_recontact,
        )

    def _rear_touchdown_contact_ready(
        self,
        leg_id: int,
        actual_contact: np.ndarray,
        *,
        foot_grf_world: np.ndarray | None = None,
        current_foot_vz: float | None = None,
        swing_phase: float | None = None,
        waiting_for_recontact: bool = False,
    ) -> bool:
        if int(leg_id) < 2:
            return bool(actual_contact[leg_id])
        return self.rear_transition_manager.contact_ready(
            int(leg_id),
            np.asarray(actual_contact, dtype=int),
            waiting_for_recontact=waiting_for_recontact,
            swing_phase=swing_phase,
            current_foot_vz=current_foot_vz,
            foot_grf_world=foot_grf_world,
        )

    def _rear_waiting_for_recontact(
        self,
        leg_id: int,
        *,
        planned_stance: bool | None = None,
        actual_contact: np.ndarray | None = None,
        prev_reacquire_active: np.ndarray | None = None,
    ) -> bool:
        leg_id = int(leg_id)
        if leg_id < 2:
            return bool(int(self.touchdown_reacquire_armed[leg_id]) == 1)

        waiting = bool(
            int(self.touchdown_reacquire_armed[leg_id]) == 1
            or int(self.rear_touchdown_retry_count[leg_id]) > 0
        )
        if prev_reacquire_active is not None:
            waiting = waiting or bool(int(prev_reacquire_active[leg_id]) == 1)
        return waiting

    def _touchdown_contact_vel_z_damping_for_leg(self, leg_id: int) -> float:
        if int(leg_id) < 2:
            return float(self.front_touchdown_contact_vel_z_damping)
        return float(self.rear_touchdown_contact_vel_z_damping)

    def _stance_recontact_active(self, leg_id: int, actual_contact: np.ndarray, startup_full_stance_active: bool) -> bool:
        leg_id = int(leg_id)
        enabled = self.front_stance_dropout_reacquire if leg_id < 2 else self.rear_stance_dropout_reacquire
        if not bool(enabled):
            return False
        return bool(
            (not startup_full_stance_active)
            and int(self.planned_contact[leg_id]) == 1
            and int(self.current_contact[leg_id]) == 1
            and int(self.previous_actual_contact[leg_id]) == 0
            and bool(actual_contact[leg_id])
        )

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

    def _contact_latch_steps_for_leg(self, leg_id: int, base_latch_steps: int) -> int:
        if int(leg_id) < 2:
            return max(int(self.front_contact_latch_steps), 0)
        return max(int(self.rear_contact_latch_steps), 0)

    def _contact_latch_budget_s_for_leg(self, leg_id: int, base_budget_s: float) -> float:
        if int(leg_id) < 2:
            return float(self.front_contact_latch_budget_s)
        return float(self.rear_contact_latch_budget_s)

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

    def _foot_contact_from_signal(self, foot_contact_state) -> np.ndarray | None:
        if foot_contact_state is None:
            return None

        actual_contact = np.zeros(4, dtype=int)
        try:
            for leg_id, leg_name in enumerate(self.legs_order):
                value = foot_contact_state[leg_name]
                if isinstance(value, (np.ndarray, list, tuple)):
                    value = np.asarray(value).reshape(-1)[0]
                actual_contact[leg_id] = 1 if bool(value) else 0
        except Exception:
            try:
                arr = np.asarray(foot_contact_state, dtype=float).reshape(-1)
                if arr.size < 4:
                    return None
                actual_contact[:] = (arr[:4] > 0.5).astype(int)
            except Exception:
                return None
        return actual_contact

    def _foot_grf_from_signal(self, foot_grf_state) -> np.ndarray | None:
        if foot_grf_state is None:
            return None

        foot_grf = np.zeros((4, 3), dtype=float)
        try:
            for leg_id, leg_name in enumerate(self.legs_order):
                value = np.asarray(foot_grf_state[leg_name], dtype=float).reshape(-1)
                if value.size < 3:
                    return None
                foot_grf[leg_id, :] = value[:3]
        except Exception:
            try:
                arr = np.asarray(foot_grf_state, dtype=float)
                if arr.shape == (4, 3):
                    foot_grf[:, :] = arr
                else:
                    arr = arr.reshape(-1)
                    if arr.size < 12:
                        return None
                    foot_grf[:, :] = arr[:12].reshape(4, 3)
            except Exception:
                return None
        return foot_grf

    def _resolve_actual_contact(self, mujoco_contact=None, foot_contact_state=None) -> np.ndarray:
        signal_contact = self._foot_contact_from_signal(foot_contact_state)
        if signal_contact is not None:
            return signal_contact
        return self._foot_contact_from_mujoco(mujoco_contact)

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
        early_stance_hitmoment = self.esd.hitmoments[leg_name]
        early_stance_hitpoint = self.esd.hitpoints[leg_name]
        if (
            cfg.mpc_params['type'] == 'linear_osqp'
            and self.gait_name == 'crawl'
            and leg_id >= 2
            and bool(self.crawl_params.rear_disable_reflex_swing)
        ):
            early_stance_hitmoment = -1.0
            early_stance_hitpoint = None
        des_foot_pos, des_foot_vel, des_foot_acc = self.stc.swing_generator.compute_trajectory_references(
            swing_time,
            self.frg.lift_off_positions[leg_name],
            touch_down,
            early_stance_hitmoment,
            early_stance_hitpoint,
        )
        swing_height_scale = 1.0
        if cfg.mpc_params['type'] == 'linear_osqp' and self.gait_name == 'crawl':
            if leg_id < 2:
                swing_height_scale = float(self.crawl_params.front_swing_height_scale)
            else:
                swing_height_scale = float(self.crawl_params.rear_swing_height_scale)
        if abs(float(swing_height_scale) - 1.0) > 1e-3:
            lift_off = np.asarray(self.frg.lift_off_positions[leg_name], dtype=float).reshape(3)
            touch_down_arr = np.asarray(touch_down, dtype=float).reshape(3)
            phase = float(np.clip(swing_time / max(float(self.stc.swing_period), 1e-6), 0.0, 1.0))
            baseline_z = float((1.0 - phase) * lift_off[2] + phase * touch_down_arr[2])
            baseline_vz = float((touch_down_arr[2] - lift_off[2]) / max(float(self.stc.swing_period), 1e-6))
            scale = float(swing_height_scale)
            des_foot_pos = np.asarray(des_foot_pos, dtype=float).reshape(3)
            des_foot_vel = np.asarray(des_foot_vel, dtype=float).reshape(3)
            des_foot_acc = np.asarray(des_foot_acc, dtype=float).reshape(3)
            des_foot_pos[2] = baseline_z + scale * (float(des_foot_pos[2]) - baseline_z)
            des_foot_vel[2] = baseline_vz + scale * (float(des_foot_vel[2]) - baseline_vz)
            des_foot_acc[2] = scale * float(des_foot_acc[2])
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

    def _parse_crawl_params(self, params: dict) -> None:
        """Populate self.crawl_params from a flat params dict, table-driven."""
        cp = self.crawl_params
        for field_name, param_key, kind in _CRAWL_PARAM_SPEC:
            raw = params.get(param_key, None)
            if raw is None:
                # Missing or explicit-None: keep dataclass default.
                continue
            if kind == 'nonneg':
                setattr(cp, field_name, max(float(raw), 0.0))
            elif kind == 'float':
                setattr(cp, field_name, float(raw))
            elif kind == 'bool':
                setattr(cp, field_name, bool(raw))
            elif kind == 'clip01':
                setattr(cp, field_name, float(np.clip(raw, 0.0, 1.0)))
            elif kind == 'clip02':
                setattr(cp, field_name, float(np.clip(raw, 0.0, 2.0)))
            elif kind in ('opt_inf', 'opt_none'):
                setattr(cp, field_name, max(float(raw), 0.0))

    # _process_crawl_recovery is inherited from CrawlRecoveryMixin


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
        foot_contact_state = None,
        foot_grf_state = None,
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
        transition_contact = self._resolve_actual_contact(
            mujoco_contact=mujoco_contact,
            foot_contact_state=foot_contact_state,
        )
        self._retire_rear_reacquire_after_stable_stance(actual_contact, simulation_dt)
        prev_touchdown_reacquire_active = np.asarray(self.touchdown_reacquire_active, dtype=int).copy()
        prev_touchdown_confirm_active = np.asarray(self.touchdown_confirm_active, dtype=int).copy()
        prev_touchdown_settle_active = np.asarray(self.touchdown_settle_active, dtype=int).copy()
        prev_touchdown_support_active = np.asarray(self.touchdown_support_active, dtype=int).copy()
        prev_rear_handoff_support_active = bool(int(self.rear_handoff_support_active) == 1)
        prev_rear_swing_bridge_active = bool(int(self.rear_swing_bridge_active) == 1)
        prev_rear_swing_release_support_active = bool(int(self.rear_swing_release_support_active) == 1)
        prev_full_contact_recovery_active = bool(int(self.full_contact_recovery_active) == 1)
        prev_rear_all_contact_alpha = float(getattr(self, 'rear_all_contact_stabilization_alpha', 0.0))
        rear_transition_recovery_context_active = bool(
            prev_full_contact_recovery_active
            or prev_rear_handoff_support_active
            or prev_rear_swing_bridge_active
            or prev_rear_swing_release_support_active
            or np.any(np.asarray(prev_touchdown_reacquire_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_confirm_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_settle_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_support_active[2:4], dtype=int) == 1)
        )
        # Front pre-swing guarding should react to the rear transition seam itself,
        # not to the broader late recovery state after that seam has already closed.
        # Keeping the front guard tied to full-contact recovery made FL/FR planned
        # swing windows stay virtually closed for too long in crawl.
        front_guard_post_recovery_active = bool(
            float(getattr(self, 'front_rear_transition_guard_post_recovery_remaining_s', 0.0)) > 1e-9
        )
        front_guard_rear_transition_context_active = bool(
            prev_rear_handoff_support_active
            or prev_rear_swing_bridge_active
            or prev_rear_swing_release_support_active
            or np.any(np.asarray(self.current_contact[2:4], dtype=int) == 0)
            or np.any(np.asarray(prev_touchdown_reacquire_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_confirm_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_settle_active[2:4], dtype=int) == 1)
            or np.any(np.asarray(prev_touchdown_support_active[2:4], dtype=int) == 1)
            or front_guard_post_recovery_active
        )
        rear_touchdown_contact_signal = self._resolve_actual_contact(
            mujoco_contact=mujoco_contact,
            foot_contact_state=foot_contact_state,
        )
        foot_grf_world = self._foot_grf_from_signal(foot_grf_state)
        feet_pos_array = np.stack(
            [np.asarray(feet_pos[leg_name], dtype=float).reshape(3) for leg_name in self.legs_order],
            axis=0,
        )
        if self.previous_feet_pos_world_valid and simulation_dt > 1e-9:
            approx_feet_vel_world = (feet_pos_array - self.previous_feet_pos_world) / float(simulation_dt)
        else:
            approx_feet_vel_world = np.zeros((4, 3), dtype=float)
        rear_touchdown_contact_signal = np.asarray(rear_touchdown_contact_signal, dtype=int).copy()
        for leg_id in range(2, 4):
            waiting_for_recontact = self._rear_waiting_for_recontact(
                leg_id,
                prev_reacquire_active=prev_touchdown_reacquire_active,
            )
            if not (
                int(self.rear_touchdown_retry_count[leg_id]) > 0
                or waiting_for_recontact
                or int(prev_touchdown_reacquire_active[leg_id]) == 1
            ):
                continue
            if int(rear_touchdown_contact_signal[leg_id]) == 1 or not bool(actual_contact[leg_id]):
                continue
            if float(approx_feet_vel_world[leg_id, 2]) > float(self.rear_touchdown_contact_max_upward_vel):
                continue
            if foot_grf_world is not None:
                try:
                    vertical_grf = max(float(foot_grf_world[leg_id, 2]), 0.0)
                except Exception:
                    vertical_grf = 0.0
                required_grf = max(float(self.rear_touchdown_contact_min_grf_z) - 0.5, 0.0)
                if vertical_grf + 1e-12 < required_grf:
                    continue
            # In the late crawl seam, the external contact signal can stay low
            # even after MuJoCo already reports a clearly load-bearing rear foot.
            # Fuse that strong actual-contact evidence back into the retry path
            # so the rear leg is not kept artificially in recontact mode.
            rear_touchdown_contact_signal[leg_id] = 1
        rear_retry_contact_signal = np.asarray(actual_contact, dtype=int).copy()
        for leg_id in range(2, 4):
            waiting_for_recontact = self._rear_waiting_for_recontact(
                leg_id,
                prev_reacquire_active=prev_touchdown_reacquire_active,
            )
            if (
                int(self.rear_touchdown_retry_count[leg_id]) > 0
                or waiting_for_recontact
                or int(prev_touchdown_reacquire_active[leg_id]) == 1
            ):
                rear_retry_contact_signal[leg_id] = int(rear_touchdown_contact_signal[leg_id])
        self.last_support_margin[:] = np.nan
        self.last_support_margin_query_xy[:] = np.nan
        self.front_late_release_active[:] = 0
        self.swing_contact_release_active[:] = 0
        self.support_confirm_active[:] = 0
        self.pre_swing_gate_active[:] = 0
        self.touchdown_reacquire_active[:] = 0
        self.touchdown_confirm_active[:] = 0
        self.touchdown_settle_active[:] = 0
        self.touchdown_support_active[:] = 0
        self.rear_retry_contact_signal_debug[:] = 0
        self.rear_touchdown_contact_ready_debug[:] = 0
        self.rear_late_stance_contact_ready_debug[:] = 0
        self.rear_all_contact_support_needed_debug[:] = 0
        self.rear_late_seam_support_active_debug[:] = 0
        self.rear_close_handoff_active_debug[:] = 0
        self.rear_late_load_share_active_debug[:] = 0
        self.rear_late_load_share_alpha_debug[:] = 0.0
        self.rear_late_load_share_candidate_active_debug[:] = 0
        self.rear_late_load_share_candidate_alpha_debug[:] = 0.0
        self.rear_late_load_share_trigger_enabled_debug = 0
        self.full_contact_recovery_trigger_debug = 0
        self.front_delayed_swing_recovery_trigger_debug = 0
        self.planted_front_recovery_trigger_debug = 0
        self.planted_front_postdrop_recovery_trigger_debug = 0
        self.front_close_gap_trigger_debug = 0
        self.front_late_rearm_trigger_debug = 0
        self.front_planted_posture_tail_trigger_debug = 0
        self.front_late_posture_tail_trigger_debug = 0
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.rear_touchdown_support_alpha = 0.0
        self.rear_all_contact_stabilization_alpha = 0.0
        self.rear_all_contact_front_planted_tail_alpha = 0.0
        self.rear_all_contact_weak_leg_alpha = 0.0
        self.rear_all_contact_weak_leg_index = -1
        self.rear_close_handoff_alpha = 0.0
        self.rear_close_handoff_leg_index = -1
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        self.front_margin_rescue_active[:] = 0
        self.front_margin_rescue_alpha[:] = 0.0
        self.rear_handoff_support_active = 0
        self.rear_swing_bridge_active = 0
        self.rear_swing_release_support_active = 0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.front_stance_dropout_support_remaining_s[:] = 0.0
        self.last_gate_forward_scale = 1.0
        self.rear_retry_contact_signal_debug[:] = np.asarray(rear_retry_contact_signal, dtype=int)
        if getattr(contact_sequence, "ndim", 0) == 2 and contact_sequence.shape[0] == 4:
            gate_forward_scale = 1.0
            requested_gate_forward_scale = 1.0
            requested_front_release_forward_scale = 1.0
            if cfg.mpc_params['type'] == 'linear_osqp':
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
                    release_timeout_s = self._swing_contact_release_timeout_for_leg(leg_id)
                    planned_swing = bool(int(self.planned_contact[leg_id]) == 0)
                    stuck_in_contact = planned_swing and bool(actual_contact[leg_id]) and (not startup_full_stance_active)
                    if (
                        int(leg_id) < 2
                        and self.gait_name == 'crawl'
                        and release_timeout_s <= 1e-9
                        and float(getattr(self.crawl_params, 'front_stuck_swing_release_timeout_s', 0.0)) > 1e-9
                    ):
                        ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                        height_ratio = float(base_pos_measured[2]) / ref_height
                        roll_mag = abs(float(base_ori_euler_xyz[0]))
                        pitch_mag = abs(float(base_ori_euler_xyz[1]))
                        front_single_planned_swing = bool(
                            np.count_nonzero(np.asarray(contact_sequence[0:2, 0], dtype=int) == 0) == 1
                        )
                        rear_contacts_stable = bool(
                            np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                            and np.all(np.asarray(actual_contact[2:4], dtype=int) == 1)
                        )
                        recent_crawl_recovery = bool(
                            prev_full_contact_recovery_active
                            or prev_rear_all_contact_alpha > 1e-9
                            or np.any(np.asarray(prev_touchdown_support_active[2:4], dtype=int) == 1)
                            or float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                        )
                        posture_bad = bool(
                            (
                                float(getattr(self.crawl_params, 'front_stuck_swing_release_height_ratio', 0.0)) > 1e-9
                                and float(height_ratio)
                                <= float(getattr(self.crawl_params, 'front_stuck_swing_release_height_ratio', 0.0))
                            )
                            or (
                                np.isfinite(getattr(self.crawl_params, 'front_stuck_swing_release_roll_threshold', np.inf))
                                and float(roll_mag)
                                >= float(getattr(self.crawl_params, 'front_stuck_swing_release_roll_threshold', np.inf))
                            )
                            or (
                                np.isfinite(getattr(self.crawl_params, 'front_stuck_swing_release_pitch_threshold', np.inf))
                                and float(pitch_mag)
                                >= float(getattr(self.crawl_params, 'front_stuck_swing_release_pitch_threshold', np.inf))
                            )
                        )
                        if (
                            stuck_in_contact
                            and bool(int(self.current_contact[leg_id]) == 1)
                            and full_contact_now
                            and front_single_planned_swing
                            and rear_contacts_stable
                            and recent_crawl_recovery
                            and posture_bad
                        ):
                            release_timeout_s = float(
                                getattr(self.crawl_params, 'front_stuck_swing_release_timeout_s', 0.0)
                            )
                    if (not stuck_in_contact) or release_timeout_s <= 1e-9:
                        self.swing_contact_release_elapsed_s[leg_id] = 0.0
                        continue
                    self.swing_contact_release_elapsed_s[leg_id] = min(
                        self.swing_contact_release_elapsed_s[leg_id] + float(simulation_dt),
                        float(release_timeout_s),
                    )
                    if self.swing_contact_release_elapsed_s[leg_id] + 1e-12 >= float(release_timeout_s):
                        self.swing_contact_release_active[leg_id] = 1
                for leg_id in range(4):
                    scheduled_swing = bool(contact_sequence[leg_id][0] == 0)
                    confirm_hold_s = self._support_contact_confirm_hold_for_leg(leg_id)
                    gate_hold_s = self._pre_swing_gate_hold_for_leg(leg_id)
                    if int(self.swing_contact_release_active[leg_id]) == 1:
                        self.support_contact_confirm_elapsed_s[leg_id] = 0.0
                        self.support_contact_confirm_wait_s[leg_id] = 0.0
                        self.support_contact_confirm_bypass_active[leg_id] = 1
                        self.pre_swing_gate_elapsed_s[leg_id] = float(gate_hold_s)
                        continue
                    if not scheduled_swing:
                        self.support_contact_confirm_bypass_active[leg_id] = 0
                    if not (scheduled_swing and bool(actual_contact[leg_id]) and confirm_hold_s > 1e-9):
                        self.support_contact_confirm_elapsed_s[leg_id] = 0.0
                        self.support_contact_confirm_wait_s[leg_id] = 0.0
                    elif int(self.support_contact_confirm_bypass_active[leg_id]) != 1:
                        self.support_contact_confirm_wait_s[leg_id] = min(
                            self.support_contact_confirm_wait_s[leg_id] + float(simulation_dt),
                            self._support_contact_confirm_max_wait_for_leg(leg_id),
                        )
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

                        confirm_wait_expired = (
                            self.support_contact_confirm_wait_s[leg_id] + 1e-12
                            >= self._support_contact_confirm_max_wait_for_leg(leg_id)
                        )
                        if (
                            self.support_contact_confirm_elapsed_s[leg_id] + 1e-12 < float(confirm_hold_s)
                            and not confirm_wait_expired
                        ):
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
                        if confirm_wait_expired:
                            self.support_contact_confirm_bypass_active[leg_id] = 1
                            self.support_contact_confirm_elapsed_s[leg_id] = 0.0
                            self.support_contact_confirm_wait_s[leg_id] = 0.0

                    front_rear_transition_guard = False
                    if int(leg_id) < 2:
                        roll_mag = abs(float(base_ori_euler_xyz[0]))
                        pitch_mag = abs(float(base_ori_euler_xyz[1]))
                        ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                        height_ratio = float(base_pos_measured[2]) / ref_height
                        support_margin = self._pre_swing_gate_margin(
                            leg_id,
                            actual_contact,
                            feet_pos,
                            com_pos_measured,
                            com_vel_xy=base_lin_vel[0:2],
                        )
                        front_guard_release_ready = bool(
                            np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                            and np.all(np.asarray(actual_contact[2:4], dtype=int) == 1)
                            and float(support_margin)
                            >= float(getattr(self, 'front_rear_transition_guard_margin_release', 0.0))
                        )
                        (
                            front_rear_transition_guard,
                            front_transition_guard_remaining_s,
                            front_transition_guard_forward_scale,
                        ) = self.rear_transition_manager.update_front_transition_guard_window(
                            leg_id,
                            gait_name=self.gait_name,
                            scheduled_swing=scheduled_swing,
                            current_contact=bool(self.current_contact[leg_id]),
                            actual_contact=bool(actual_contact[leg_id]),
                            rear_transition_active=front_guard_rear_transition_context_active,
                            roll_mag=roll_mag,
                            pitch_mag=pitch_mag,
                            height_ratio=height_ratio,
                            simulation_dt=simulation_dt,
                            rear_support_active=bool(
                                np.any(np.asarray(prev_touchdown_support_active[2:4], dtype=int) == 1)
                            ),
                            rear_all_contact_active=bool(prev_rear_all_contact_alpha > 1e-9),
                            rear_contacts_stable=front_guard_release_ready,
                        )
                    if front_rear_transition_guard:
                        gate_forward_scale = min(
                            gate_forward_scale,
                            float(front_transition_guard_forward_scale),
                        )
                        self.pre_swing_gate_active[leg_id] = 1
                        guard_steps = max(
                            int(np.floor(float(front_transition_guard_remaining_s) / max(self.mpc_dt, 1e-6))) + 1,
                            1,
                        )
                        guard_steps = min(guard_steps, contact_sequence.shape[1])
                        contact_sequence[leg_id][0:guard_steps] = 1
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
                        self.pre_swing_gate_elapsed_s[leg_id] = float(gate_hold_s)
                        if leg_id < 2:
                            gate_forward_scale = min(gate_forward_scale, requested_front_release_forward_scale)
                        continue
                    rear_preswing_guard = False
                    if int(leg_id) >= 2:
                        roll_mag = abs(float(base_ori_euler_xyz[0]))
                        pitch_mag = abs(float(base_ori_euler_xyz[1]))
                        ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                        height_ratio = float(base_pos_measured[2]) / ref_height
                        rear_preswing_guard = self.rear_transition_manager.should_delay_preswing_for_posture(
                            gait_name=self.gait_name,
                            scheduled_swing=scheduled_swing,
                            current_contact=bool(self.current_contact[leg_id]),
                            actual_contact=bool(actual_contact[leg_id]),
                            recovery_active=rear_transition_recovery_context_active,
                            roll_mag=roll_mag,
                            pitch_mag=pitch_mag,
                            height_ratio=height_ratio,
                        )
                    if rear_preswing_guard and float(gate_hold_s) > 1e-9:
                        gate_forward_scale = min(gate_forward_scale, requested_gate_forward_scale)
                        self.pre_swing_gate_active[leg_id] = 1
                        self.pre_swing_gate_elapsed_s[leg_id] = min(
                            self.pre_swing_gate_elapsed_s[leg_id] + float(simulation_dt),
                            float(gate_hold_s),
                        )
                        remaining_hold_s = max(
                            float(gate_hold_s) - self.pre_swing_gate_elapsed_s[leg_id],
                            0.0,
                        )
                        gate_steps = max(int(np.floor(remaining_hold_s / max(self.mpc_dt, 1e-6))) + 1, 1)
                        gate_steps = min(gate_steps, contact_sequence.shape[1])
                        contact_sequence[leg_id][0:gate_steps] = 1
                        continue
                    # Pre-swing gating should only hold a leg before the controller
                    # has actually committed to swing. Once controller-side swing is
                    # already open, re-clamping it mid-flight shortens the effective
                    # swing window and hurts late rear recontact.
                    if not (
                        full_contact_now
                        and scheduled_swing
                        and bool(self.current_contact[leg_id])
                        and bool(actual_contact[leg_id])
                        and float(gate_hold_s) > 1e-9
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
                    if self.pre_swing_gate_elapsed_s[leg_id] >= float(gate_hold_s):
                        continue

                    gate_forward_scale = min(gate_forward_scale, requested_gate_forward_scale)
                    self.pre_swing_gate_active[leg_id] = 1
                    self.pre_swing_gate_elapsed_s[leg_id] = min(
                        self.pre_swing_gate_elapsed_s[leg_id] + float(simulation_dt),
                        float(gate_hold_s),
                    )
                    remaining_hold_s = max(
                        float(gate_hold_s) - self.pre_swing_gate_elapsed_s[leg_id],
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
                leg_latch_budget_s = self._contact_latch_budget_s_for_leg(leg_id, latch_budget_s)
                if cfg.mpc_params['type'] == 'linear_osqp':
                    leg_latch_budget_s = self._effective_linear_latch_budget_s(leg_latch_budget_s)
                leg_base_latch_steps = min(
                    self._contact_latch_steps_for_leg(leg_id, base_latch_steps),
                    contact_sequence.shape[1],
                )
                if cfg.mpc_params['type'] == 'linear_osqp':
                    leg_base_latch_steps = min(leg_base_latch_steps, self._max_linear_latch_steps())
                scheduled_swing = contact_sequence[leg_id][0] == 0
                if int(self.swing_contact_release_active[leg_id]) == 1 and scheduled_swing:
                    self.contact_latch_elapsed_s[leg_id] = max(
                        float(self.contact_latch_elapsed_s[leg_id]),
                        float(leg_latch_budget_s),
                    )
                    self.virtual_unlatch_hold_remaining_s[leg_id] = 0.0
                    continue
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
                    if leg_latch_budget_s > 0.0:
                        self.contact_latch_elapsed_s[leg_id] = float(leg_latch_budget_s)
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

                if int(leg_id) >= 2:
                    roll_mag = abs(float(base_ori_euler_xyz[0]))
                    pitch_mag = abs(float(base_ori_euler_xyz[1]))
                    ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                    height_ratio = float(base_pos_measured[2]) / ref_height
                    rear_late_release_guard = self.rear_transition_manager.should_delay_preswing_for_posture(
                        gait_name=self.gait_name,
                        scheduled_swing=bool(scheduled_swing),
                        current_contact=bool(self.current_contact[leg_id]),
                        actual_contact=bool(actual_contact[leg_id]),
                            recovery_active=rear_transition_recovery_context_active,
                        roll_mag=roll_mag,
                        pitch_mag=pitch_mag,
                        height_ratio=height_ratio,
                    )
                    if rear_late_release_guard:
                        gate_hold_s = self._pre_swing_gate_hold_for_leg(leg_id)
                        if gate_hold_s > 1e-9:
                            gate_forward_scale = min(gate_forward_scale, requested_gate_forward_scale)
                            self.pre_swing_gate_active[leg_id] = 1
                            self.pre_swing_gate_elapsed_s[leg_id] = min(
                                self.pre_swing_gate_elapsed_s[leg_id] + float(simulation_dt),
                                float(gate_hold_s),
                            )
                            remaining_hold_s = max(
                                float(gate_hold_s) - self.pre_swing_gate_elapsed_s[leg_id],
                                0.0,
                            )
                            guard_steps = max(int(np.floor(remaining_hold_s / max(self.mpc_dt, 1e-6))) + 1, 1)
                            guard_steps = min(guard_steps, contact_sequence.shape[1])
                            contact_sequence[leg_id][0:guard_steps] = 1
                            continue

                if leg_latch_budget_s > 0.0 and self.contact_latch_elapsed_s[leg_id] >= leg_latch_budget_s:
                    continue

                next_elapsed = self.contact_latch_elapsed_s[leg_id] + float(simulation_dt)
                if leg_latch_budget_s > 0.0:
                    self.contact_latch_elapsed_s[leg_id] = min(next_elapsed, leg_latch_budget_s)
                else:
                    self.contact_latch_elapsed_s[leg_id] = next_elapsed

                leg_latch_steps = self._latched_contact_horizon_steps(leg_id, leg_base_latch_steps)
                if leg_latch_budget_s > 0.0:
                    remaining_budget_s = max(leg_latch_budget_s - self.contact_latch_elapsed_s[leg_id], 0.0)
                    remaining_budget_steps = max(int(np.floor(remaining_budget_s / max(self.mpc_dt, 1e-6))) + 1, 0)
                    leg_latch_steps = min(leg_latch_steps, remaining_budget_steps)
                if leg_latch_steps <= 0:
                    continue

                if actual_contact[leg_id] and scheduled_swing:
                    contact_sequence[leg_id][0:leg_latch_steps] = 1

            for leg_id in range(4):
                if int(self.planned_contact[leg_id]) == 0:
                    self.touchdown_reacquire_armed[leg_id] = 1

            for leg_id in range(2, 4):
                planned_stance = bool(int(self.planned_contact[leg_id]) == 1)
                waiting_for_recontact = self._rear_waiting_for_recontact(
                    leg_id,
                    planned_stance=planned_stance,
                    actual_contact=actual_contact,
                    prev_reacquire_active=prev_touchdown_reacquire_active,
                )
                if not self.rear_transition_manager.should_delay_reacquire(
                    planned_stance=planned_stance,
                    waiting_for_recontact=waiting_for_recontact,
                    actual_contact=bool(actual_contact[leg_id]),
                    swing_time=float(self.stc.swing_time[leg_id]),
                ):
                    continue
                # Delay rear planned-stance reacquire until the swing leg has at
                # least entered its late descent phase; otherwise the controller
                # can "reacquire" while the foot is still on its way up.
                contact_sequence[leg_id][0] = 0
                self.touchdown_reacquire_elapsed_s[leg_id] = 0.0

            for leg_id in range(4):
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                waiting_for_recontact = self._rear_waiting_for_recontact(
                    leg_id,
                    planned_stance=planned_stance,
                    actual_contact=actual_contact,
                    prev_reacquire_active=prev_touchdown_reacquire_active,
                )
                contact_ready = self._touchdown_contact_ready_for_leg(
                    leg_id,
                    rear_retry_contact_signal if leg_id >= 2 else actual_contact,
                    foot_grf_world=foot_grf_world,
                    current_foot_vz=current_foot_vz,
                    swing_phase=swing_phase,
                    waiting_for_recontact=waiting_for_recontact,
                )
                if not planned_stance:
                    if int(leg_id) >= 2:
                        self.rear_transition_manager.prime_pending_confirm(
                            leg_id,
                            planned_stance=planned_stance,
                            waiting_for_recontact=waiting_for_recontact,
                            contact_ready=contact_ready,
                        )
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                if int(self.touchdown_reacquire_armed[leg_id]) != 1:
                    if int(leg_id) >= 2:
                        self.rear_transition_manager.clear_pending_confirm(leg_id)
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                if contact_ready:
                    self.touchdown_reacquire_elapsed_s[leg_id] = 0.0
                    continue
                if int(leg_id) >= 2:
                    self.rear_transition_manager.clear_pending_confirm(leg_id)
                hold_s = self._touchdown_reacquire_hold_for_leg(leg_id)
                if int(leg_id) >= 2:
                    (
                        reacquire_active,
                        next_reacquire_elapsed,
                        hold_steps,
                        reacquire_forward_scale,
                    ) = self.rear_transition_manager.update_reacquire_window(
                        planned_stance=planned_stance,
                        waiting_for_recontact=waiting_for_recontact,
                        contact_ready=contact_ready,
                        current_elapsed_s=float(self.touchdown_reacquire_elapsed_s[leg_id]),
                        simulation_dt=float(simulation_dt),
                        horizon_steps=int(contact_sequence.shape[1]),
                    )
                    self.touchdown_reacquire_active[leg_id] = int(reacquire_active)
                    self.touchdown_reacquire_elapsed_s[leg_id] = float(next_reacquire_elapsed)
                    if reacquire_active:
                        gate_forward_scale = min(gate_forward_scale, float(reacquire_forward_scale))
                        contact_sequence[leg_id][0:hold_steps] = 0
                    continue
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
                current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                waiting_for_recontact = self._rear_waiting_for_recontact(
                    leg_id,
                    planned_stance=planned_stance,
                    actual_contact=actual_contact,
                    prev_reacquire_active=prev_touchdown_reacquire_active,
                )
                contact_ready = self._touchdown_contact_ready_for_leg(
                    leg_id,
                    rear_retry_contact_signal if leg_id >= 2 else actual_contact,
                    foot_grf_world=foot_grf_world,
                    current_foot_vz=current_foot_vz,
                    swing_phase=swing_phase,
                    waiting_for_recontact=waiting_for_recontact,
                )
                if not planned_stance or not contact_ready:
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                confirm_hold_s = self._touchdown_confirm_hold_for_leg(leg_id)
                if confirm_hold_s <= 1e-9:
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                stance_recontact = self._stance_recontact_active(leg_id, actual_contact, startup_full_stance_active)
                keep_confirm = stance_recontact or bool(prev_touchdown_reacquire_active[leg_id]) or bool(
                    self.touchdown_confirm_elapsed_s[leg_id] > 1e-9
                )
                if not keep_confirm and int(leg_id) >= 2:
                    keep_confirm = self.rear_transition_manager.should_keep_confirm(
                        leg_id,
                        waiting_for_recontact=waiting_for_recontact,
                        planned_stance=planned_stance,
                        contact_ready=contact_ready,
                        prev_reacquire_active=bool(prev_touchdown_reacquire_active[leg_id]),
                        confirm_elapsed_s=float(self.touchdown_confirm_elapsed_s[leg_id]),
                        stance_recontact=stance_recontact,
                    )
                if not keep_confirm:
                    self.touchdown_confirm_elapsed_s[leg_id] = 0.0
                    continue

                self.touchdown_confirm_active[leg_id] = 1
                if int(leg_id) >= 2:
                    _, next_confirm_elapsed, confirm_forward_scale = self.rear_transition_manager.consume_confirm(
                        leg_id,
                        confirm_elapsed_s=float(self.touchdown_confirm_elapsed_s[leg_id]),
                        simulation_dt=float(simulation_dt),
                    )
                    gate_forward_scale = min(gate_forward_scale, float(confirm_forward_scale))
                    self.touchdown_confirm_elapsed_s[leg_id] = float(next_confirm_elapsed)
                    continue
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
                current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                waiting_for_recontact = self._rear_waiting_for_recontact(
                    leg_id,
                    planned_stance=planned_stance,
                    actual_contact=actual_contact,
                    prev_reacquire_active=prev_touchdown_reacquire_active,
                )
                contact_ready = self._touchdown_contact_ready_for_leg(
                    leg_id,
                    rear_retry_contact_signal if leg_id >= 2 else actual_contact,
                    foot_grf_world=foot_grf_world,
                    current_foot_vz=current_foot_vz,
                    swing_phase=swing_phase,
                    waiting_for_recontact=waiting_for_recontact,
                )
                if not planned_stance or not contact_ready:
                    self.touchdown_settle_remaining_s[leg_id] = 0.0
                    continue

                stance_recontact = self._stance_recontact_active(leg_id, actual_contact, startup_full_stance_active)
                if int(leg_id) >= 2:
                    settle_active, next_settle_remaining, settle_forward_scale = self.rear_transition_manager.update_settle_window(
                        planned_stance=planned_stance,
                        contact_ready=contact_ready,
                        prev_reacquire_active=bool(prev_touchdown_reacquire_active[leg_id]),
                        stance_recontact=stance_recontact,
                        settle_remaining_s=float(self.touchdown_settle_remaining_s[leg_id]),
                        simulation_dt=float(simulation_dt),
                    )
                    self.touchdown_settle_remaining_s[leg_id] = float(next_settle_remaining)
                    if not settle_active:
                        continue
                    self.touchdown_settle_active[leg_id] = 1
                    gate_forward_scale = min(gate_forward_scale, float(settle_forward_scale))
                    continue

                if bool(prev_touchdown_reacquire_active[leg_id]) or stance_recontact:
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

            if self.rear_touchdown_confirm_keep_swing:
                for leg_id in range(2, 4):
                    if int(self.touchdown_confirm_active[leg_id]) != 1:
                        continue
                    current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                    swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                    if not self._touchdown_contact_ready_for_leg(
                        leg_id,
                        rear_retry_contact_signal,
                        foot_grf_world=foot_grf_world,
                        current_foot_vz=current_foot_vz,
                        swing_phase=swing_phase,
                        waiting_for_recontact=True,
                    ) or not self.rear_transition_manager.should_keep_swing_during_confirm(
                        confirm_active=bool(int(self.touchdown_confirm_active[leg_id]) == 1),
                        contact_ready=True,
                    ):
                        continue
                    # A flaky first rear touchdown can briefly trip actual contact,
                    # reset swing_time to zero, and then re-open reacquire from the
                    # very start of the swing arc. Keep controller-side swing alive
                    # through the short confirm window so the rear foot continues
                    # descending instead of restarting from lift-off.
                    contact_sequence[leg_id][0] = 0

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
            rear_support_alpha = 0.0
            rear_all_contact_alpha = 0.0
            front_close_gap_keep_swing_mask = np.zeros(2, dtype=bool)
            front_planted_seam_keep_swing_mask = np.zeros(2, dtype=bool)
            (
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
            ) = self._process_crawl_recovery(
                actual_contact=actual_contact,
                transition_contact=transition_contact,
                base_ori_euler_xyz=base_ori_euler_xyz,
                base_pos_measured=base_pos_measured,
                contact_sequence=contact_sequence,
                feet_pos=feet_pos,
                com_pos_measured=com_pos_measured,
                base_lin_vel=base_lin_vel,
                simulation_dt=simulation_dt,
                startup_full_stance_active=startup_full_stance_active,
                prev_full_contact_recovery_active=prev_full_contact_recovery_active,
                gate_forward_scale=gate_forward_scale,
            )

            front_margin_rescue_bilateral = bool(np.any(np.asarray(self.front_margin_rescue_active[0:2], dtype=int) == 1))
            front_margin_rescue_alpha_max = float(
                np.clip(np.max(np.asarray(self.front_margin_rescue_alpha[0:2], dtype=float)), 0.0, 1.0)
            )
            self.rear_transition_manager.update_actual_contact_elapsed(
                np.asarray(rear_retry_contact_signal[2:4], dtype=int),
                float(simulation_dt),
            )
            self._sync_rear_transition_debug_arrays()
            front_dropout_support_hold_s = float(self.front_stance_dropout_support_hold_s)
            if front_dropout_support_hold_s > 1e-9:
                rear_transition_active_now = bool(
                    int(self.rear_handoff_support_active) == 1
                    or int(self.rear_swing_bridge_active) == 1
                    or int(self.rear_swing_release_support_active) == 1
                    or np.any(np.asarray(self.touchdown_reacquire_active[2:4], dtype=int) == 1)
                    or np.any(np.asarray(self.touchdown_confirm_active[2:4], dtype=int) == 1)
                    or np.any(np.asarray(self.touchdown_settle_active[2:4], dtype=int) == 1)
                    or np.any(np.asarray(self.touchdown_support_active[2:4], dtype=int) == 1)
                )
                for leg_id in range(2):
                    front_dropout = bool(
                        (not startup_full_stance_active)
                        and int(self.planned_contact[leg_id]) == 1
                        and int(self.current_contact[leg_id]) == 1
                        and bool(self.previous_actual_contact[leg_id])
                        and (not bool(actual_contact[leg_id]))
                        and (
                            rear_transition_active_now
                            or int(self.full_contact_recovery_active) == 1
                        )
                    )
                    front_return_gap = bool(
                        (not startup_full_stance_active)
                        and self.gait_name == 'crawl'
                        and int(self.planned_contact[leg_id]) == 1
                        and int(self.current_contact[leg_id]) == 0
                        and (not bool(actual_contact[leg_id]))
                    )
                    if front_dropout or front_return_gap:
                        self.front_stance_dropout_support_remaining_s[leg_id] = max(
                            float(self.front_stance_dropout_support_remaining_s[leg_id]),
                            front_dropout_support_hold_s,
                        )
            for leg_id in range(4):
                touchdown_window_active = int(
                    int(self.touchdown_confirm_active[leg_id]) == 1
                    or int(self.touchdown_settle_active[leg_id]) == 1
                )
                margin_rescue_active = int(int(self.front_margin_rescue_active[leg_id]) == 1)
                support_active = int(touchdown_window_active == 1 or margin_rescue_active == 1)
                front_dropout_support_active = bool(
                    leg_id < 2
                    and float(self.front_stance_dropout_support_remaining_s[leg_id]) > 1e-9
                    and int(self.planned_contact[leg_id]) == 1
                    and int(self.current_contact[leg_id]) == 1
                )
                if leg_id < 2 and front_margin_rescue_bilateral and bool(actual_contact[leg_id]):
                    support_active = 1
                if (
                    leg_id < 2
                    and int(self.full_contact_recovery_active) == 1
                    and bool(actual_contact[leg_id])
                ):
                    support_active = 1
                if front_dropout_support_active:
                    support_active = 1
                self.touchdown_support_active[leg_id] = support_active
                if support_active:
                    leg_support_alpha = 1.0
                    if touchdown_window_active != 1 and leg_id < 2 and front_margin_rescue_bilateral:
                        leg_support_alpha = front_margin_rescue_alpha_max
                    elif touchdown_window_active != 1 and margin_rescue_active == 1:
                        leg_support_alpha = float(np.clip(self.front_margin_rescue_alpha[leg_id], 0.0, 1.0))
                    elif front_dropout_support_active:
                        leg_support_alpha = 1.0
                    support_alpha = max(float(support_alpha), float(leg_support_alpha))
                    if leg_id < 2:
                        front_support_alpha = max(float(front_support_alpha), float(leg_support_alpha))
                    else:
                        rear_support_alpha = max(float(rear_support_alpha), float(leg_support_alpha))
                if front_dropout_support_active:
                    gate_forward_scale = min(
                        gate_forward_scale,
                        float(self.front_stance_dropout_support_forward_scale),
                    )
                    self.front_stance_dropout_support_remaining_s[leg_id] = max(
                        0.0,
                        float(self.front_stance_dropout_support_remaining_s[leg_id]) - float(simulation_dt),
                    )
                elif leg_id < 2:
                    self.front_stance_dropout_support_remaining_s[leg_id] = 0.0
                planned_stance = bool(contact_sequence[leg_id][0] == 1)
                leg_name = self.legs_order[leg_id]
                current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                if int(leg_id) >= 2:
                    waiting_for_recontact = self._rear_waiting_for_recontact(
                        leg_id,
                        planned_stance=planned_stance,
                        actual_contact=actual_contact,
                        prev_reacquire_active=prev_touchdown_reacquire_active,
                    )
                    late_stance_scheduled = bool(int(self.planned_contact[leg_id]) == 1)
                    rear_contact_returned_now = bool(actual_contact[leg_id]) and (
                        not bool(self.previous_actual_contact[leg_id])
                    )
                    pitch_mag = abs(float(base_ori_euler_xyz[1]))
                    roll_mag = abs(float(base_ori_euler_xyz[0]))
                    height_ratio = float(base_pos_measured[2]) / max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
                    rear_contact_ready = self._rear_touchdown_contact_ready(
                        leg_id,
                        rear_retry_contact_signal,
                        foot_grf_world=foot_grf_world,
                        current_foot_vz=current_foot_vz,
                        swing_phase=swing_phase,
                        waiting_for_recontact=waiting_for_recontact,
                    )
                    self.rear_touchdown_contact_ready_debug[leg_id] = int(bool(rear_contact_ready))
                    late_stance_contact_ready = self.rear_transition_manager.should_accept_late_stance_contact(
                        leg_id,
                        rear_retry_contact_signal,
                        gait_name=self.gait_name,
                        planned_stance=late_stance_scheduled,
                        waiting_for_recontact=waiting_for_recontact,
                        actual_contact=bool(actual_contact[leg_id]),
                        previous_actual_contact=bool(self.previous_actual_contact[leg_id]),
                        recovery_active=bool(rear_transition_recovery_context_active),
                        roll_mag=roll_mag,
                        pitch_mag=pitch_mag,
                        height_ratio=height_ratio,
                        current_foot_vz=current_foot_vz,
                        foot_grf_world=foot_grf_world,
                    )
                    self.rear_late_stance_contact_ready_debug[leg_id] = int(bool(late_stance_contact_ready))
                    if late_stance_scheduled and waiting_for_recontact and late_stance_contact_ready:
                        self._clear_touchdown_reacquire_state(leg_id, clear_retry_count=True)
                        self.touchdown_settle_remaining_s[leg_id] = max(
                            float(self.touchdown_settle_remaining_s[leg_id]),
                            self._touchdown_settle_hold_for_leg(leg_id),
                        )
                        self.touchdown_settle_active[leg_id] = 1
                        self.touchdown_support_active[leg_id] = 1
                        support_alpha = max(float(support_alpha), 1.0)
                        rear_support_alpha = max(float(rear_support_alpha), 1.0)
                        gate_forward_scale = min(
                            gate_forward_scale,
                            float(self.touchdown_settle_forward_scale),
                        )
                        contact_sequence[leg_id][0] = 1
                        planned_stance = True
                        rear_contact_ready = True
                    if self.rear_transition_manager.should_start_touchdown_support(
                        gait_name=self.gait_name,
                        planned_stance=planned_stance,
                        waiting_for_recontact=waiting_for_recontact,
                        actual_contact=bool(actual_contact[leg_id]),
                        previous_actual_contact=bool(self.previous_actual_contact[leg_id]),
                        contact_ready=rear_contact_ready,
                    ) and (not planned_stance):
                        # In the current crawl branch, a rear foot can regain real
                        # contact slightly before the controller-side swing window
                        # fully closes again. Do not snap the gait state here, but
                        # do reuse the rear touchdown-support window so the wrapper
                        # can start adding vertical/posture support immediately
                        # instead of waiting for the slower confirm/settle path.
                        self.touchdown_settle_remaining_s[leg_id] = max(
                            float(self.touchdown_settle_remaining_s[leg_id]),
                            self._touchdown_settle_hold_for_leg(leg_id),
                        )
                        self.touchdown_settle_active[leg_id] = 1
                        self.touchdown_support_active[leg_id] = 1
                        support_alpha = max(float(support_alpha), 1.0)
                        rear_support_alpha = max(float(rear_support_alpha), 1.0)
                        gate_forward_scale = min(gate_forward_scale, float(self.touchdown_settle_forward_scale))
                    if self.rear_transition_manager.should_accept_touchdown_as_stance(
                        gait_name=self.gait_name,
                        planned_stance=planned_stance,
                        waiting_for_recontact=waiting_for_recontact,
                        contact_ready=rear_contact_ready,
                    ):
                        # Once the rear foot has returned with a clearly valid
                        # contact signal, close the controller-side crawl swing
                        # immediately instead of keeping the leg in the slower
                        # reacquire path for another long seam.
                        contact_sequence[leg_id][0] = 1
                        # Only bridge the pre-stance case where the rear foot has
                        # already come back with valid contact but the gait schedule
                        # still marks swing for a short moment. Arming the same lock
                        # again after normal planned-stance close made crawl overly
                        # sticky and suppressed front swing.
                        self._arm_rear_touchdown_close_lock(leg_id)
                        self._clear_touchdown_reacquire_state(leg_id, clear_retry_count=True)
                        self.touchdown_settle_remaining_s[leg_id] = max(
                            float(self.touchdown_settle_remaining_s[leg_id]),
                            self._touchdown_settle_hold_for_leg(leg_id),
                        )
                        self.touchdown_settle_active[leg_id] = 1
                        self.touchdown_support_active[leg_id] = 1
                        support_alpha = max(float(support_alpha), 1.0)
                        rear_support_alpha = max(float(rear_support_alpha), 1.0)
                        gate_forward_scale = min(
                            gate_forward_scale,
                            float(self.touchdown_settle_forward_scale),
                        )
                        planned_stance = True
                    if planned_stance and rear_contact_ready:
                        # Once rear contact is re-established with enough debounce /
                        # GRF / downward motion evidence, close the controller-side
                        # swing immediately instead of keeping the leg stuck in the
                        # reacquire path behind confirm/settle bookkeeping.
                        if self.gait_name == 'crawl':
                            # When the rear foot finally becomes load-bearing again,
                            # re-arm the same short settle/support window here as
                            # well. Without this, crawl can still lose body height
                            # on the exact step where the controller-side swing
                            # closes, because the wrapper does not yet see a rear
                            # touchdown-support phase until later bookkeeping
                            # catches up.
                            if self.rear_transition_manager.should_start_touchdown_support(
                                gait_name=self.gait_name,
                                planned_stance=planned_stance,
                                waiting_for_recontact=waiting_for_recontact,
                                actual_contact=bool(actual_contact[leg_id]),
                                previous_actual_contact=bool(self.previous_actual_contact[leg_id]),
                                contact_ready=rear_contact_ready,
                            ):
                                self.touchdown_settle_remaining_s[leg_id] = max(
                                    float(self.touchdown_settle_remaining_s[leg_id]),
                                    self._touchdown_settle_hold_for_leg(leg_id),
                                )
                                self.touchdown_settle_active[leg_id] = 1
                                self.touchdown_support_active[leg_id] = 1
                                support_alpha = max(float(support_alpha), 1.0)
                                rear_support_alpha = max(float(rear_support_alpha), 1.0)
                                gate_forward_scale = min(
                                    gate_forward_scale,
                                    float(self.touchdown_settle_forward_scale),
                                )
                        self._clear_touchdown_reacquire_state(leg_id, clear_retry_count=True)
                        contact_sequence[leg_id][0] = 1
            # Rear recontact is the dominant failure mode in the current branch.
            # Even when controller-side swing is already closed again, there can
            # still be a short planned-stance gap where the rear foot is not yet
            # truly load-bearing. Keep the rear touchdown-support overrides alive
            # through that gap so the wrapper can temporarily boost the remaining
            # support legs instead of letting the body drop immediately.
            rear_pending_recontact_support = bool(np.any(np.asarray(self.touchdown_reacquire_active[2:4], dtype=int) == 1))
            if not rear_pending_recontact_support:
                for leg_id in range(2, 4):
                    planned_stance = bool(int(self.planned_contact[leg_id]) == 1)
                    waiting_for_recontact = self._rear_waiting_for_recontact(
                        leg_id,
                        planned_stance=planned_stance,
                        actual_contact=actual_contact,
                        prev_reacquire_active=prev_touchdown_reacquire_active,
                    )
                    current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                    swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                    rear_contact_ready = self._rear_touchdown_contact_ready(
                        leg_id,
                        rear_retry_contact_signal,
                        foot_grf_world=foot_grf_world,
                        current_foot_vz=current_foot_vz,
                        swing_phase=swing_phase,
                        waiting_for_recontact=True,
                    )
                    if not self.rear_transition_manager.pending_support_required(
                        planned_stance=planned_stance,
                        waiting_for_recontact=waiting_for_recontact,
                        contact_ready=rear_contact_ready,
                    ):
                        continue
                    rear_pending_recontact_support = True
                    break
            if rear_pending_recontact_support:
                support_alpha = max(float(support_alpha), 1.0)
                rear_support_alpha = max(float(rear_support_alpha), 1.0)
                gate_forward_scale = min(gate_forward_scale, float(self.touchdown_reacquire_forward_scale))
            rear_release_support_hold_s = float(self.rear_swing_release_support_hold_s)
            rear_forced_release_now = bool(np.any(np.asarray(self.swing_contact_release_active[2:4], dtype=int) == 1))
            rear_release_transition = bool(
                np.any(np.asarray(self.planned_contact[2:4], dtype=int) == 0)
                or np.any(np.asarray(self.current_contact[2:4], dtype=int) == 0)
            )
            if rear_release_support_hold_s > 1e-9:
                if rear_forced_release_now and (not startup_full_stance_active):
                    self.rear_swing_release_support_remaining_s = max(
                        float(self.rear_swing_release_support_remaining_s),
                        rear_release_support_hold_s,
                    )
                if self.rear_swing_release_support_remaining_s > 1e-9 and rear_release_transition:
                    self.rear_swing_release_support_active = 1
                    gate_forward_scale = min(gate_forward_scale, float(self.rear_swing_release_forward_scale))
                    self.rear_swing_release_support_remaining_s = max(
                        0.0,
                        float(self.rear_swing_release_support_remaining_s) - float(simulation_dt),
                    )
                    support_alpha = max(float(support_alpha), 1.0)
                    rear_support_alpha = max(float(rear_support_alpha), 1.0)
                    for leg_id in range(2):
                        planned_stance = bool(contact_sequence[leg_id][0] == 1)
                        if planned_stance and bool(actual_contact[leg_id]):
                            self.touchdown_support_active[leg_id] = 1
                            front_support_alpha = 1.0
                            support_alpha = 1.0
                elif not rear_release_transition:
                    self.rear_swing_release_support_remaining_s = 0.0
            else:
                self.rear_swing_release_support_remaining_s = 0.0
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
                front_pair_ready = bool(
                    np.all(np.asarray(actual_contact[0:2], dtype=int) == 1)
                    and np.all(np.asarray(contact_sequence[0:2, 0], dtype=int) == 1)
                )
                if (not startup_full_stance_active) and (front_support_alpha > 1e-9 or front_pair_ready) and rear_swing_soon:
                    self.rear_handoff_support_mask[:] = 0
                    if front_pair_ready:
                        self.rear_handoff_support_mask[0:2] = 1
                    else:
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
                    rear_single_support_ids = [
                        leg_id
                        for leg_id in range(2, 4)
                        if bool(contact_sequence[leg_id][0] == 1) and bool(actual_contact[leg_id])
                    ]
                    if len(rear_single_support_ids) == 1:
                        rear_handoff_rear_alpha = float(
                            getattr(self, 'rear_handoff_support_rear_alpha_scale', 0.0)
                        )
                        if rear_handoff_rear_alpha > 1e-9:
                            self.touchdown_support_active[rear_single_support_ids[0]] = 1
                            rear_support_alpha = max(float(rear_support_alpha), rear_handoff_rear_alpha)
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
                front_pair_ready = bool(
                    np.all(np.asarray(actual_contact[0:2], dtype=int) == 1)
                    and np.all(np.asarray(contact_sequence[0:2, 0], dtype=int) == 1)
                )
                recent_front_ok = (
                    front_support_alpha > 1e-9
                    or float(self.rear_swing_bridge_recent_front_remaining_s) > 1e-9
                    or front_pair_ready
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
                bridge_release_tail_s = float(
                    getattr(self, 'rear_swing_bridge_allcontact_release_tail_s', 0.0)
                )
                if bridge_release_tail_s > 1e-9:
                    rear_bridge_clear_now = bool(
                        all_contact_now
                        and np.all(np.asarray(contact_sequence[2:4, 0], dtype=int) == 1)
                        and np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                        and np.all(actual_contact_array[2:4] == 1)
                        and (not rear_planned_swing)
                        and (not rear_current_swing)
                        and (not rear_pending_recontact)
                        and (not np.any(np.asarray(self.touchdown_confirm_active[2:4], dtype=int) == 1))
                        and (not np.any(np.asarray(self.touchdown_settle_active[2:4], dtype=int) == 1))
                    )
                    if rear_bridge_clear_now and float(self.rear_swing_bridge_remaining_s) > bridge_release_tail_s:
                        self.rear_swing_bridge_remaining_s = min(
                            float(self.rear_swing_bridge_remaining_s),
                            bridge_release_tail_s,
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
                rear_single_support_ids = [
                    leg_id
                    for leg_id in range(2, 4)
                    if bool(contact_sequence[leg_id][0] == 1) and bool(actual_contact[leg_id])
                ]
                if len(rear_single_support_ids) == 1:
                    rear_bridge_rear_alpha = float(
                        getattr(self, 'rear_swing_bridge_rear_alpha_scale', 0.0)
                    )
                    if rear_bridge_rear_alpha > 1e-9:
                        self.touchdown_support_active[rear_single_support_ids[0]] = 1
                        rear_support_alpha = max(float(rear_support_alpha), rear_bridge_rear_alpha)
            if int(self.full_contact_recovery_active) == 1:
                # Full-contact recovery is meant to keep the front touchdown-style
                # support overrides alive briefly after all feet have come back.
                # Without routing a non-zero support alpha through the wrapper, the
                # controller slows down but never actually raises the extra height /
                # posture support gains that the recovery hold is supposed to reuse.
                support_alpha = max(float(support_alpha), float(self.full_contact_recovery_alpha))
                front_support_alpha = max(float(front_support_alpha), float(self.full_contact_recovery_alpha))
                rear_recovery_scale = float(self.full_contact_recovery_rear_support_scale)
                if rear_recovery_scale > 1e-9:
                    recent_rear_touchdown = bool(
                        np.any(np.asarray(prev_touchdown_support_active[2:4], dtype=int) == 1)
                        or np.any(np.asarray(self.touchdown_confirm_active[2:4], dtype=int) == 1)
                        or np.any(np.asarray(self.touchdown_settle_active[2:4], dtype=int) == 1)
                        or np.any(np.asarray(self.touchdown_reacquire_active[2:4], dtype=int) == 1)
                    )
                    # In crawl, once late full-contact recovery is active the robot
                    # is already in a low/posture-compromised all-contact state.
                    # Keep the rear touchdown support gains blended in during that
                    # window as well, instead of requiring a very recent rear event
                    # only. This makes the post-front-return stabilization path less
                    # front-biased after the rear seam has already been crossed.
                    use_rear_recovery = bool(recent_rear_touchdown)
                    if self.gait_name == 'crawl':
                        use_rear_recovery = True
                    if use_rear_recovery:
                        rear_support_alpha = max(
                            float(rear_support_alpha),
                            float(self.full_contact_recovery_alpha) * rear_recovery_scale,
                        )
                        if self.gait_name == 'crawl':
                            # During the late all-contact recovery seam, routing
                            # only a rear alpha through the wrapper is not enough:
                            # rear stance anchors and touchdown damping still stay
                            # off because the per-leg support-active path remains
                            # false. Reuse that same support-active route for rear
                            # legs while full-contact recovery is explicitly trying
                            # to stabilize a low/posture-poor all-contact state.
                            for leg_id in range(2, 4):
                                if bool(actual_contact[leg_id]) and bool(contact_sequence[leg_id][0] == 1):
                                    self.touchdown_support_active[leg_id] = 1
            # Rear touchdown support currently catches the immediate recontact,
            # but the remaining crawl failure appears one seam later: the rear
            # foot is already back in contact while the body is still low and
            # rolled, and the generic full-contact recovery then takes over with
            # front-biased support only. Keep a short rear-specific support tail
            # alive through that late load-transfer phase so the wrapper can
            # continue applying rear touchdown gains until posture recovers.
            roll_mag = abs(float(base_ori_euler_xyz[0]))
            pitch_mag = abs(float(base_ori_euler_xyz[1]))
            ref_height = max(float(cfg.simulation_params.get('ref_z', 0.0)), 1e-6)
            height_ratio = float(base_pos_measured[2]) / ref_height
            vertical_grf = np.maximum(np.asarray(foot_grf_world, dtype=float)[:, 2], 0.0)
            total_vertical_grf = max(float(np.sum(vertical_grf)), 1e-6)
            rear_vertical_grf = max(float(np.sum(vertical_grf[2:4])), 1e-6)
            rear_load_share = float(np.sum(vertical_grf[2:4])) / total_vertical_grf
            rear_pair_leg_shares = np.asarray(vertical_grf[2:4], dtype=float) / rear_vertical_grf
            weak_rear_leg_load_share = float(np.min(rear_pair_leg_shares))
            for leg_id in range(2, 4):
                rear_leg_load_share = float(rear_pair_leg_shares[leg_id - 2])
                recent_rear_touchdown = bool(
                    bool(actual_contact[leg_id]) and (not bool(self.previous_actual_contact[leg_id]))
                ) or bool(prev_touchdown_reacquire_active[leg_id]) or bool(int(self.touchdown_confirm_active[leg_id]) == 1) or bool(
                    int(self.touchdown_settle_active[leg_id]) == 1
                )
                direct_rear_support_active = bool(
                    recent_rear_touchdown
                    or bool(int(self.touchdown_reacquire_active[leg_id]) == 1)
                    or bool(int(self.touchdown_confirm_active[leg_id]) == 1)
                    or bool(int(self.touchdown_settle_active[leg_id]) == 1)
                )
                post_support_trigger = bool(
                    bool(prev_touchdown_support_active[leg_id])
                    and (not direct_rear_support_active)
                    and bool(contact_sequence[leg_id][0] == 1)
                    and bool(actual_contact[leg_id])
                    and (not self.rear_transition_manager.post_support_running(leg_id))
                )
                post_support_active, post_support_forward_scale = self.rear_transition_manager.update_post_support_window(
                    leg_id,
                    trigger=post_support_trigger,
                    planned_stance=bool(contact_sequence[leg_id][0] == 1),
                    actual_contact=bool(actual_contact[leg_id]),
                    simulation_dt=float(simulation_dt),
                    height_ratio=height_ratio,
                    roll_mag=roll_mag,
                    pitch_mag=pitch_mag,
                    leg_grf_z=float(vertical_grf[leg_id]),
                    rear_load_share=rear_load_share,
                    recovery_active=bool(int(self.full_contact_recovery_active) == 1),
                )
                if not post_support_active:
                    continue
                self.touchdown_support_active[leg_id] = 1
                support_alpha = max(float(support_alpha), 1.0)
                rear_support_alpha = max(float(rear_support_alpha), 1.0)
                gate_forward_scale = min(gate_forward_scale, float(post_support_forward_scale))
            if float(self.crawl_state.front_stance_support_tail_remaining_s) > 1e-9:
                gate_forward_scale = min(
                    gate_forward_scale,
                    float(getattr(self.crawl_params, 'front_stance_support_tail_forward_scale', 1.0)),
                )
                for leg_id in range(2):
                    if bool(contact_sequence[leg_id][0] == 1) and bool(actual_contact[leg_id]):
                        self.touchdown_support_active[leg_id] = 1
                        support_alpha = max(float(support_alpha), 1.0)
                        front_support_alpha = max(float(front_support_alpha), 1.0)
                self.crawl_state.front_stance_support_tail_remaining_s = max(
                    0.0,
                    float(self.crawl_state.front_stance_support_tail_remaining_s) - float(simulation_dt),
                )
            else:
                self.crawl_state.front_stance_support_tail_remaining_s = 0.0
            all_actual_contact = bool(np.all(np.asarray(actual_contact, dtype=int) == 1))
            for leg_id in range(2, 4):
                all_contact_support_needed = bool(
                    (
                        float(getattr(self, 'rear_all_contact_stabilization_min_rear_load_share', 0.0)) > 1e-9
                        and float(rear_load_share) + 1e-12
                        < float(getattr(self, 'rear_all_contact_stabilization_min_rear_load_share', 0.0))
                    )
                    or (
                        float(getattr(self, 'rear_all_contact_stabilization_min_rear_leg_load_share', 0.0)) > 1e-9
                        and float(rear_leg_load_share) + 1e-12
                        < float(getattr(self, 'rear_all_contact_stabilization_min_rear_leg_load_share', 0.0))
                    )
                    or (
                        float(getattr(self, 'rear_all_contact_stabilization_height_ratio', 0.0)) > 1e-9
                        and float(height_ratio)
                        <= float(getattr(self, 'rear_all_contact_stabilization_height_ratio', 0.0))
                    )
                    or (
                        np.isfinite(getattr(self, 'rear_all_contact_stabilization_roll_threshold', np.inf))
                        and float(roll_mag)
                        >= float(getattr(self, 'rear_all_contact_stabilization_roll_threshold', np.inf))
                    )
                    or (
                        np.isfinite(getattr(self, 'rear_all_contact_stabilization_pitch_threshold', np.inf))
                        and float(pitch_mag)
                        >= float(getattr(self, 'rear_all_contact_stabilization_pitch_threshold', np.inf))
                    )
                )
                self.rear_all_contact_support_needed_debug[leg_id] = int(bool(all_contact_support_needed))
                recent_rear_touchdown = bool(
                    bool(actual_contact[leg_id]) and (not bool(self.previous_actual_contact[leg_id]))
                ) or bool(prev_touchdown_reacquire_active[leg_id]) or bool(
                    int(self.touchdown_confirm_active[leg_id]) == 1
                ) or bool(int(self.touchdown_settle_active[leg_id]) == 1)
                seam_waiting_for_recontact = bool(
                    self._rear_waiting_for_recontact(
                        leg_id,
                        planned_stance=bool(contact_sequence[leg_id][0] == 1),
                        actual_contact=actual_contact,
                        prev_reacquire_active=prev_touchdown_reacquire_active,
                    )
                    or bool(int(self.touchdown_reacquire_active[leg_id]) == 1)
                )
                preclose_pitch_bad = bool(
                    np.isfinite(getattr(self, 'rear_all_contact_stabilization_preclose_pitch_threshold', np.inf))
                    and float(pitch_mag)
                    >= float(getattr(self, 'rear_all_contact_stabilization_preclose_pitch_threshold', np.inf))
                )
                preclose_descending_bad = bool(
                    float(base_lin_vel[2])
                    <= float(getattr(self, 'rear_all_contact_stabilization_preclose_vz_threshold', -np.inf))
                )
                late_stance_scheduled = bool(int(self.planned_contact[leg_id]) == 1)
                late_seam_open_with_actual = bool(
                    self.gait_name == 'crawl'
                    and late_stance_scheduled
                    and bool(actual_contact[leg_id])
                    and bool(self.current_contact[leg_id] == 0)
                    and all_actual_contact
                    and seam_waiting_for_recontact
                )
                if late_seam_open_with_actual:
                    self.rear_late_seam_elapsed_s[leg_id] = min(
                        float(self.rear_late_seam_elapsed_s[leg_id]) + float(simulation_dt),
                        max(float(self.rear_late_seam_support_trigger_s), float(simulation_dt)),
                    )
                else:
                    self.rear_late_seam_elapsed_s[leg_id] = 0.0
                # The remaining crawl failure happens in the short seam where the
                # rear foot is already back on the ground, all four feet are
                # physically in contact, but the controller-side rear contact is
                # still open. Start applying the late all-contact support shaping
                # through that pre-close seam instead of waiting for the current
                # contact latch to close one or two steps later.
                preclose_all_contact_active = bool(
                    self.gait_name == 'crawl'
                    and bool(contact_sequence[leg_id][0] == 1)
                    and bool(actual_contact[leg_id])
                    and bool(self.current_contact[leg_id] == 0)
                    and all_actual_contact
                    and all_contact_support_needed
                    and seam_waiting_for_recontact
                    and preclose_descending_bad
                    and preclose_pitch_bad
                )
                late_seam_trigger = bool(
                    late_seam_open_with_actual
                    and all_contact_support_needed
                    and float(self.rear_late_seam_support_trigger_s) > 1e-9
                    and float(self.rear_late_seam_elapsed_s[leg_id]) + 1e-12
                    >= float(self.rear_late_seam_support_trigger_s)
                    and (not self.rear_transition_manager.all_contact_stabilization_running(leg_id))
                )
                if preclose_all_contact_active:
                    self.touchdown_support_active[leg_id] = 1
                    support_alpha = max(float(support_alpha), 1.0)
                    rear_support_alpha = max(float(rear_support_alpha), 1.0)
                    rear_all_contact_alpha = max(float(rear_all_contact_alpha), 1.0)
                    gate_forward_scale = min(
                        gate_forward_scale,
                        float(getattr(self, 'rear_all_contact_stabilization_forward_scale', 1.0)),
                    )
                    if front_support_alpha > 1e-9:
                        front_support_alpha = min(
                            float(front_support_alpha),
                            float(getattr(self, 'rear_all_contact_stabilization_front_alpha_scale', 1.0)),
                        )
                all_contact_trigger = bool(
                    self.gait_name == 'crawl'
                    and bool(contact_sequence[leg_id][0] == 1)
                    and bool(actual_contact[leg_id])
                    and bool(self.current_contact[leg_id] == 0)
                    and all_actual_contact
                    and recent_rear_touchdown
                    and all_contact_support_needed
                    and (not self.rear_transition_manager.all_contact_stabilization_running(leg_id))
                )
                (
                    all_contact_active,
                    all_contact_forward_scale,
                    all_contact_front_alpha_scale,
                ) = self.rear_transition_manager.update_all_contact_stabilization_window(
                    leg_id,
                    trigger=bool(all_contact_trigger or late_seam_trigger),
                    planned_stance=late_stance_scheduled or late_seam_open_with_actual,
                    actual_contact=bool(actual_contact[leg_id]),
                    all_actual_contact=all_actual_contact,
                    simulation_dt=float(simulation_dt),
                    height_ratio=height_ratio,
                    roll_mag=roll_mag,
                    pitch_mag=pitch_mag,
                    rear_load_share=rear_load_share,
                    rear_leg_load_share=rear_leg_load_share,
                )
                self.rear_late_seam_support_active_debug[leg_id] = int(
                    bool(late_seam_open_with_actual and all_contact_active)
                )
                if not all_contact_active:
                    continue
                self.touchdown_support_active[leg_id] = 1
                support_alpha = max(float(support_alpha), 1.0)
                rear_support_alpha = max(float(rear_support_alpha), 1.0)
                rear_all_contact_alpha = max(float(rear_all_contact_alpha), 1.0)
                gate_forward_scale = min(gate_forward_scale, float(all_contact_forward_scale))
                if front_support_alpha > 1e-9:
                    front_support_alpha = min(
                        float(front_support_alpha),
                        float(all_contact_front_alpha_scale),
                    )
            if np.any(np.asarray(self.touchdown_support_active[2:4], dtype=int) == 1):
                rear_support_alpha = max(float(rear_support_alpha), float(support_alpha))
            self.touchdown_support_alpha = float(support_alpha)
            self.front_touchdown_support_alpha = float(front_support_alpha)
            self.rear_touchdown_support_alpha = float(rear_support_alpha)
            self.rear_all_contact_stabilization_alpha = float(rear_all_contact_alpha)
            planned_all_contact_now = bool(np.all(np.asarray(self.planned_contact, dtype=int) == 1))
            rear_all_contact_posture_needed = bool(
                (
                    float(getattr(self, 'rear_all_contact_stabilization_min_rear_load_share', 0.0)) > 1e-9
                    and float(rear_load_share) + 1e-12
                    < float(getattr(self, 'rear_all_contact_stabilization_min_rear_load_share', 0.0))
                )
                or (
                    float(getattr(self, 'rear_all_contact_stabilization_min_rear_leg_load_share', 0.0)) > 1e-9
                    and float(weak_rear_leg_load_share) + 1e-12
                    < float(getattr(self, 'rear_all_contact_stabilization_min_rear_leg_load_share', 0.0))
                )
                or (
                    float(getattr(self, 'rear_all_contact_stabilization_height_ratio', 0.0)) > 1e-9
                    and float(height_ratio)
                    <= float(getattr(self, 'rear_all_contact_stabilization_height_ratio', 0.0))
                )
                or (
                    np.isfinite(getattr(self, 'rear_all_contact_stabilization_roll_threshold', np.inf))
                    and float(roll_mag)
                    >= float(getattr(self, 'rear_all_contact_stabilization_roll_threshold', np.inf))
                )
                or (
                    np.isfinite(getattr(self, 'rear_all_contact_stabilization_pitch_threshold', np.inf))
                    and float(pitch_mag)
                    >= float(getattr(self, 'rear_all_contact_stabilization_pitch_threshold', np.inf))
                )
            )
            rear_all_contact_post_recovery_tail_hold_s = float(
                getattr(self, 'rear_all_contact_post_recovery_tail_hold_s', 0.0)
            )
            rear_all_contact_release_tail_trigger = bool(
                self.gait_name == 'crawl'
                and float(prev_rear_all_contact_alpha) > 1e-9
                and float(rear_all_contact_alpha) <= 1e-9
                and all_contact_now
                and planned_all_contact_now
                and bool(int(self.touchdown_settle_active[leg_id]) == 1)
                and rear_all_contact_posture_needed
            )
            if rear_all_contact_post_recovery_tail_hold_s > 1e-9:
                rear_all_contact_post_recovery_trigger = bool(
                    self.gait_name == 'crawl'
                    and prev_full_contact_recovery_active
                    and int(self.full_contact_recovery_active) == 0
                    and all_contact_now
                    and planned_all_contact_now
                    and rear_all_contact_posture_needed
                )
                planted_front_recovery_rearm_trigger_s = float(
                    getattr(self.crawl_params, 'front_planted_swing_recovery_rearm_trigger_s', 0.0)
                )
                front_planted_late_mask = (
                    (np.asarray(self.planned_contact[0:2], dtype=int) == 0)
                    & (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                    & (actual_contact_array[0:2] == 1)
                )
                rear_contacts_stable_now = bool(
                    np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                    and np.all(actual_contact_array[2:4] == 1)
                )
                planted_front_recovery_near_expiry_now = bool(
                    (
                        prev_full_contact_recovery_active
                        and int(self.full_contact_recovery_active) == 0
                    )
                    or (
                        (prev_full_contact_recovery_active or float(self.full_contact_recovery_remaining_s) > 1e-9)
                        and (
                            planted_front_recovery_rearm_trigger_s <= 1e-9
                            or float(self.full_contact_recovery_remaining_s)
                            <= planted_front_recovery_rearm_trigger_s + 1e-12
                        )
                    )
                )
                front_planted_posture_bad = bool(
                    (
                        float(getattr(self.crawl_params, 'front_planted_swing_recovery_height_ratio', 0.0)) > 1e-9
                        and float(height_ratio)
                        <= float(getattr(self.crawl_params, 'front_planted_swing_recovery_height_ratio', 0.0))
                    )
                    or (
                        getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', None) is not None
                        and np.isfinite(float(getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', 0.0)))
                        and float(roll_mag)
                        >= float(getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', 0.0))
                    )
                )
                front_planted_posture_tail_trigger = bool(
                    self.gait_name == 'crawl'
                    and rear_all_contact_posture_needed
                    and rear_contacts_stable_now
                    and np.count_nonzero(front_planted_late_mask) == 1
                    and planted_front_recovery_near_expiry_now
                    and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                    and np.any(
                        front_planted_late_mask
                        & (
                            front_support_margins
                            <= float(
                                getattr(
                                    self,
                                    'crawl_front_planted_swing_recovery_margin_threshold',
                                    0.0,
                                )
                            )
                        )
                    )
                )
                self.front_planted_posture_tail_trigger_debug = int(
                    bool(front_planted_posture_tail_trigger)
                )
                front_late_posture_tail_trigger = bool(
                    self.gait_name == 'crawl'
                    and (
                        front_late_posture_tail_candidate
                        or front_planted_posture_tail_candidate
                        or front_planted_posture_tail_trigger
                    )
                    and rear_all_contact_posture_needed
                )
                self.front_late_posture_tail_trigger_debug = int(bool(front_late_posture_tail_trigger))
                if (
                    rear_all_contact_post_recovery_trigger
                    or front_late_posture_tail_trigger
                    or rear_all_contact_release_tail_trigger
                ):
                    tail_alpha_scale = 1.0
                    if rear_all_contact_release_tail_trigger:
                        tail_alpha_scale = float(
                            getattr(self, 'rear_all_contact_release_tail_alpha_scale', 1.0)
                        )
                    elif (
                        front_late_posture_tail_trigger
                        and not rear_all_contact_post_recovery_trigger
                        and not rear_all_contact_release_tail_trigger
                    ):
                        tail_alpha_scale = float(
                            getattr(
                                self,
                                'rear_all_contact_post_recovery_front_late_alpha_scale',
                                1.0,
                            )
                        )
                    self.rear_all_contact_post_recovery_alpha_scale = float(
                        np.clip(tail_alpha_scale, 0.0, 1.0)
                    )
                    self.rear_all_contact_post_recovery_remaining_s = max(
                        float(self.rear_all_contact_post_recovery_remaining_s),
                        rear_all_contact_post_recovery_tail_hold_s,
                    )
                if front_planted_posture_tail_trigger:
                    self.rear_all_contact_front_planted_tail_alpha_scale = float(
                        np.clip(
                            getattr(
                                self,
                                'rear_all_contact_post_recovery_front_late_alpha_scale',
                                1.0,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    self.rear_all_contact_front_planted_tail_remaining_s = max(
                        float(self.rear_all_contact_front_planted_tail_remaining_s),
                        rear_all_contact_post_recovery_tail_hold_s,
                    )
                elif float(self.rear_all_contact_front_planted_tail_remaining_s) > 1e-9:
                    self.rear_all_contact_front_planted_tail_remaining_s = max(
                        0.0,
                        float(self.rear_all_contact_front_planted_tail_remaining_s) - float(simulation_dt),
                    )
                else:
                    self.rear_all_contact_front_planted_tail_remaining_s = 0.0
                    self.rear_all_contact_front_planted_tail_alpha_scale = 1.0
                if float(self.rear_all_contact_front_planted_tail_remaining_s) > 1e-9:
                    self.rear_all_contact_front_planted_tail_alpha = float(
                        np.clip(
                            float(self.rear_all_contact_front_planted_tail_alpha_scale)
                            * float(self.rear_all_contact_front_planted_tail_remaining_s)
                            / max(rear_all_contact_post_recovery_tail_hold_s, 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                else:
                    self.rear_all_contact_front_planted_tail_alpha = 0.0
                front_planted_seam_support_hold_s = float(
                    getattr(self.crawl_params, 'front_planted_seam_support_hold_s', 0.0)
                )
                if front_planted_seam_support_hold_s > 1e-9:
                    front_planted_post_recovery_drop_now = bool(
                        prev_full_contact_recovery_active
                        and int(self.full_contact_recovery_active) == 0
                    )
                    front_planted_seam_support_trigger = bool(
                        self.gait_name == 'crawl'
                        and front_planted_post_recovery_drop_now
                        and int(self.full_contact_recovery_active) == 0
                        and float(rear_all_contact_alpha) <= 1e-9
                        and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
                        and rear_contacts_stable_now
                        and np.count_nonzero(front_planted_late_mask) == 1
                        and front_planted_posture_bad
                        and np.any(
                            front_planted_late_mask
                            & (
                                front_support_margins
                                <= float(
                                    getattr(
                                        self,
                                        'crawl_front_planted_swing_recovery_margin_threshold',
                                        0.0,
                                    )
                                )
                            )
                        )
                    )
                    if front_planted_seam_support_trigger:
                        self.crawl_state.front_planted_seam_support_remaining_s = max(
                            float(self.crawl_state.front_planted_seam_support_remaining_s),
                            front_planted_seam_support_hold_s,
                        )
                    elif float(self.crawl_state.front_planted_seam_support_remaining_s) > 1e-9:
                        self.crawl_state.front_planted_seam_support_remaining_s = max(
                            0.0,
                            float(self.crawl_state.front_planted_seam_support_remaining_s) - float(simulation_dt),
                        )
                    else:
                        self.crawl_state.front_planted_seam_support_remaining_s = 0.0
                    if float(self.crawl_state.front_planted_seam_support_remaining_s) > 1e-9:
                        self.crawl_state.front_planted_seam_support_alpha = float(
                            np.clip(
                                float(self.crawl_state.front_planted_seam_support_remaining_s)
                                / max(front_planted_seam_support_hold_s, 1e-6),
                                0.0,
                                1.0,
                            )
                        )
                        if bool(getattr(self.crawl_params, 'front_planted_seam_keep_swing', False)):
                            front_planted_seam_keep_swing_mask = np.asarray(
                                front_planted_late_mask,
                                dtype=bool,
                            ).copy()
                    else:
                        self.crawl_state.front_planted_seam_support_alpha = 0.0
                else:
                    self.crawl_state.front_planted_seam_support_remaining_s = 0.0
                    self.crawl_state.front_planted_seam_support_alpha = 0.0
                if (
                    rear_all_contact_post_recovery_trigger
                    or front_late_posture_tail_trigger
                    or rear_all_contact_release_tail_trigger
                ):
                    pass
                elif float(self.rear_all_contact_post_recovery_remaining_s) > 1e-9:
                    self.rear_all_contact_post_recovery_remaining_s = max(
                        0.0,
                        float(self.rear_all_contact_post_recovery_remaining_s) - float(simulation_dt),
                    )
                else:
                    self.rear_all_contact_post_recovery_remaining_s = 0.0
                if float(self.rear_all_contact_post_recovery_remaining_s) > 1e-9:
                    # Keep a short, decaying posture-only tail after late full-contact
                    # recovery drops out. This preserves the rear all-contact height/roll
                    # shaping in the linear force path without re-extending the broader
                    # touchdown/recovery support windows that tend to suppress the next
                    # front swing.
                    rear_all_contact_alpha = max(
                        float(rear_all_contact_alpha),
                        float(
                                np.clip(
                                float(self.rear_all_contact_post_recovery_alpha_scale)
                                * float(self.rear_all_contact_post_recovery_remaining_s)
                                / max(rear_all_contact_post_recovery_tail_hold_s, 1e-6),
                                0.0,
                                1.0,
                            )
                        ),
                    )
            else:
                self.rear_all_contact_post_recovery_remaining_s = 0.0
                self.rear_all_contact_post_recovery_alpha_scale = 1.0
                self.rear_all_contact_front_planted_tail_remaining_s = 0.0
                self.rear_all_contact_front_planted_tail_alpha_scale = 1.0
                self.rear_all_contact_front_planted_tail_alpha = 0.0
                self.crawl_state.front_planted_seam_support_remaining_s = 0.0
                self.crawl_state.front_planted_seam_support_alpha = 0.0
            full_contact_recovery_allcontact_release_tail_s = float(
                getattr(self, 'full_contact_recovery_allcontact_release_tail_s', 0.0)
            )
            if full_contact_recovery_allcontact_release_tail_s > 1e-9:
                rear_transition_clear_now = bool(
                    np.all(np.asarray(contact_sequence[2:4, 0], dtype=int) == 1)
                    and np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                    and np.all(actual_contact_array[2:4] == 1)
                    and int(self.rear_handoff_support_active) == 0
                    and int(self.rear_swing_bridge_active) == 0
                    and int(self.rear_swing_release_support_active) == 0
                    and float(self.rear_close_handoff_alpha) <= 1e-9
                    and (not np.any(np.asarray(self.touchdown_reacquire_active[2:4], dtype=int) == 1))
                    and (not np.any(np.asarray(self.touchdown_confirm_active[2:4], dtype=int) == 1))
                    and (not np.any(np.asarray(self.touchdown_settle_active[2:4], dtype=int) == 1))
                    and (not np.any(np.asarray(self.touchdown_support_active[2:4], dtype=int) == 1))
                    and float(rear_all_contact_alpha) <= 1e-9
                )
                if (
                    self.gait_name == 'crawl'
                    and all_contact_now
                    and planned_all_contact_now
                    and rear_transition_clear_now
                    and float(self.full_contact_recovery_remaining_s)
                    > full_contact_recovery_allcontact_release_tail_s
                ):
                    self.full_contact_recovery_remaining_s = min(
                        float(self.full_contact_recovery_remaining_s),
                        full_contact_recovery_allcontact_release_tail_s,
                    )
            front_stance_support_tail_hold_s = float(
                getattr(self.crawl_params, 'front_stance_support_tail_hold_s', 0.0)
            )
            front_stance_release_tail_trigger = bool(
                self.gait_name == 'crawl'
                and float(front_stance_support_tail_hold_s) > 1e-9
                and rear_all_contact_release_tail_trigger
                and float(self.crawl_state.front_stance_support_tail_remaining_s) <= 1e-9
            )
            if front_stance_release_tail_trigger:
                self.crawl_state.front_stance_support_tail_remaining_s = max(
                    float(self.crawl_state.front_stance_support_tail_remaining_s),
                    float(front_stance_support_tail_hold_s),
                )
            self.rear_all_contact_weak_leg_alpha = 0.0
            self.rear_all_contact_weak_leg_index = -1
            weak_leg_share_ref = float(
                getattr(self, 'rear_all_contact_stabilization_weak_leg_share_ref', 0.0)
            )
            weak_leg_height_ratio = float(
                getattr(self, 'rear_all_contact_stabilization_weak_leg_height_ratio', 0.0)
            )
            weak_leg_tail_only = bool(
                getattr(self, 'rear_all_contact_stabilization_weak_leg_tail_only', False)
            )
            weak_leg_post_recovery_tail_active = bool(
                float(getattr(self, 'rear_all_contact_post_recovery_remaining_s', 0.0)) > 1e-9
            )
            weak_leg_late_recovery_only_active = bool(
                self.gait_name == 'crawl'
                and int(self.full_contact_recovery_active) == 1
                and float(rear_all_contact_alpha) <= 1e-9
                and all_contact_now
                and planned_all_contact_now
            )
            # Keep the dedicated weak-leg rear-floor assist narrowly scoped.
            # The broader "late recovery only" window helped earlier all-contact
            # cycles too aggressively and shortened the run before the final
            # low-height seam. When tail-only mode is requested, treat it as a
            # true post-recovery rear-all-contact tail rather than as a general
            # full-contact-recovery extension.
            weak_leg_tail_active = bool(
                weak_leg_post_recovery_tail_active
                if weak_leg_tail_only
                else (weak_leg_post_recovery_tail_active or weak_leg_late_recovery_only_active)
            )
            weak_leg_height_ok = bool(
                weak_leg_height_ratio <= 1e-9
                or float(height_ratio) <= weak_leg_height_ratio
            )
            weak_leg_support_alpha_source = float(rear_all_contact_alpha)
            if (not weak_leg_tail_only) and weak_leg_late_recovery_only_active:
                weak_leg_support_alpha_source = max(
                    float(weak_leg_support_alpha_source),
                    float(getattr(self, 'full_contact_recovery_alpha', 0.0)),
                )
            if (
                self.gait_name == 'crawl'
                and float(weak_leg_support_alpha_source) > 1e-9
                and weak_leg_share_ref > 1e-9
                and all_contact_now
                and planned_all_contact_now
                and weak_leg_height_ok
                and ((not weak_leg_tail_only) or weak_leg_tail_active)
            ):
                weak_rear_leg_local_idx = int(np.argmin(rear_pair_leg_shares))
                weak_rear_leg_id = int(2 + weak_rear_leg_local_idx)
                weak_rear_leg_share_deficit_alpha = float(
                    np.clip(
                        (weak_leg_share_ref - float(weak_rear_leg_load_share))
                        / max(weak_leg_share_ref, 0.20),
                        0.0,
                        1.0,
                    )
                )
                if (
                    weak_rear_leg_share_deficit_alpha > 1e-9
                    and bool(actual_contact[weak_rear_leg_id])
                    and bool(int(self.current_contact[weak_rear_leg_id]) == 1)
                    and bool(int(self.planned_contact[weak_rear_leg_id]) == 1)
                ):
                    weak_leg_activation_alpha = float(
                        float(weak_leg_support_alpha_source) * weak_rear_leg_share_deficit_alpha
                    )
                    if weak_leg_tail_only:
                        # In the narrow post-recovery tail we want the targeted
                        # weak-leg assist to be strong enough to matter before
                        # the low-height rear seam closes again. The old
                        # source*deficit product was too small and made even
                        # large floor-delta sweeps numerically inert.
                        weak_leg_activation_alpha = float(
                            np.clip(
                                float(weak_leg_support_alpha_source)
                                * max(
                                    weak_rear_leg_share_deficit_alpha,
                                    float(
                                        np.clip(
                                            weak_rear_leg_share_deficit_alpha / 0.10,
                                            0.0,
                                            1.0,
                                        )
                                    ),
                                ),
                                0.0,
                                1.0,
                            )
                        )
                    self.rear_all_contact_weak_leg_alpha = float(
                        np.clip(weak_leg_activation_alpha, 0.0, 1.0)
                    )
                    self.rear_all_contact_weak_leg_index = int(weak_rear_leg_id)
            front_planted_weak_share_ref = float(
                getattr(self.crawl_params, 'front_planted_weak_rear_share_ref', 0.0)
            )
            if (
                self.gait_name == 'crawl'
                and float(self.rear_all_contact_weak_leg_alpha) <= 1e-9
                and front_planted_weak_share_ref > 1e-9
                and int(self.full_contact_recovery_active) == 0
                and float(rear_all_contact_alpha) <= 1e-9
                and float(self.front_touchdown_support_recent_remaining_s) > 1e-9
            ):
                front_planted_late_mask = (
                    (np.asarray(self.planned_contact[0:2], dtype=int) == 0)
                    & (np.asarray(self.current_contact[0:2], dtype=int) == 1)
                    & (actual_contact_array[0:2] == 1)
                )
                rear_contacts_stable_for_front_planted = bool(
                    np.all(np.asarray(self.current_contact[2:4], dtype=int) == 1)
                    and np.all(actual_contact_array[2:4] == 1)
                )
                front_planted_support_margin_ok = bool(
                    np.any(
                        front_planted_late_mask
                        & (
                            front_support_margins
                            <= float(
                                getattr(
                                    self,
                                    'crawl_front_planted_swing_recovery_margin_threshold',
                                    0.0,
                                )
                            )
                        )
                    )
                )
                front_planted_posture_bad = bool(
                    (
                        float(getattr(self.crawl_params, 'front_planted_swing_recovery_height_ratio', 0.0)) > 1e-9
                        and float(height_ratio)
                        <= float(getattr(self.crawl_params, 'front_planted_swing_recovery_height_ratio', 0.0))
                    )
                    or (
                        getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', None) is not None
                        and np.isfinite(float(getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', 0.0)))
                        and float(roll_mag)
                        >= float(getattr(self.crawl_params, 'front_planted_swing_recovery_roll_threshold', 0.0))
                    )
                )
                if (
                    np.count_nonzero(front_planted_late_mask) == 1
                    and rear_contacts_stable_for_front_planted
                    and front_planted_support_margin_ok
                    and front_planted_posture_bad
                ):
                    weak_rear_leg_local_idx = int(np.argmin(rear_pair_leg_shares))
                    weak_rear_leg_id = int(2 + weak_rear_leg_local_idx)
                    weak_rear_leg_share_deficit_alpha = float(
                        np.clip(
                            (front_planted_weak_share_ref - float(weak_rear_leg_load_share))
                            / max(front_planted_weak_share_ref, 0.20),
                            0.0,
                            1.0,
                        )
                    )
                    if (
                        weak_rear_leg_share_deficit_alpha > 1e-9
                        and bool(actual_contact[weak_rear_leg_id])
                        and bool(int(self.current_contact[weak_rear_leg_id]) == 1)
                        and bool(int(self.planned_contact[weak_rear_leg_id]) == 1)
                    ):
                        front_recent_alpha = float(
                            np.clip(
                                float(self.front_touchdown_support_recent_remaining_s)
                                / max(float(self.full_contact_recovery_recent_window_s), 1e-6),
                                0.0,
                                1.0,
                            )
                        )
                        front_planted_weak_alpha = float(
                            np.clip(
                                front_recent_alpha
                                * max(
                                    weak_rear_leg_share_deficit_alpha,
                                    float(
                                        np.clip(
                                            weak_rear_leg_share_deficit_alpha / 0.10,
                                            0.0,
                                            1.0,
                                        )
                                    ),
                                ),
                                0.0,
                                float(getattr(self.crawl_params, 'front_planted_weak_rear_alpha_cap', 1.0)),
                            )
                        )
                        self.rear_all_contact_weak_leg_alpha = float(front_planted_weak_alpha)
                        self.rear_all_contact_weak_leg_index = int(weak_rear_leg_id)
            self.touchdown_support_alpha = float(support_alpha)
            self.front_touchdown_support_alpha = float(front_support_alpha)
            self.rear_touchdown_support_alpha = float(rear_support_alpha)
            self.rear_all_contact_stabilization_alpha = float(rear_all_contact_alpha)
            self._sync_rear_transition_debug_arrays()
            if float(self.full_contact_recovery_recent_window_s) > 1e-9:
                steady_all_contact_hold = bool(
                    all_contact_now
                    and planned_all_contact_now
                    and float(rear_all_contact_alpha) <= 1e-9
                    and float(self.crawl_state.front_stance_support_tail_remaining_s) <= 1e-9
                )
                if front_support_alpha > 1e-9 and (not steady_all_contact_hold):
                    self.front_touchdown_support_recent_remaining_s = float(self.full_contact_recovery_recent_window_s)
                else:
                    self.front_touchdown_support_recent_remaining_s = max(
                        0.0,
                        float(self.front_touchdown_support_recent_remaining_s) - float(simulation_dt),
                    )
            else:
                self.front_touchdown_support_recent_remaining_s = 0.0
            post_recovery_guard_hold_s = float(
                getattr(self, 'front_rear_transition_guard_post_recovery_hold_s', 0.0)
            )
            if post_recovery_guard_hold_s > 1e-9:
                post_recovery_guard_trigger = bool(
                    self.gait_name == 'crawl'
                    and (
                        (prev_full_contact_recovery_active and int(self.full_contact_recovery_active) == 0)
                        or (float(prev_rear_all_contact_alpha) > 1e-9 and float(rear_all_contact_alpha) <= 1e-9)
                    )
                )
                if post_recovery_guard_trigger:
                    self.front_rear_transition_guard_post_recovery_remaining_s = max(
                        float(self.front_rear_transition_guard_post_recovery_remaining_s),
                        post_recovery_guard_hold_s,
                    )
                else:
                    self.front_rear_transition_guard_post_recovery_remaining_s = max(
                        0.0,
                        float(self.front_rear_transition_guard_post_recovery_remaining_s) - float(simulation_dt),
                    )
            else:
                self.front_rear_transition_guard_post_recovery_remaining_s = 0.0

            self.last_gate_forward_scale = float(gate_forward_scale)
            if cfg.mpc_params['type'] == 'linear_osqp' and gate_forward_scale < 0.999:
                ref_base_lin_vel = np.array(ref_base_lin_vel, dtype=float, copy=True)
                ref_base_lin_vel[0] *= gate_forward_scale

        if cfg.mpc_params['type'] == 'linear_osqp' and self.rear_touchdown_reacquire_force_until_contact:
            for leg_id in range(2, 4):
                planned_stance = bool(int(self.planned_contact[leg_id]) == 1)
                waiting_for_recontact = self._rear_waiting_for_recontact(
                    leg_id,
                    planned_stance=planned_stance,
                    actual_contact=actual_contact,
                    prev_reacquire_active=prev_touchdown_reacquire_active,
                )
                current_foot_vz = float(approx_feet_vel_world[leg_id, 2])
                swing_phase = float(self.stc.swing_time[leg_id]) / max(float(self.stc.swing_period), 1e-6)
                rear_contact_ready = self._rear_touchdown_contact_ready(
                    leg_id,
                    rear_retry_contact_signal,
                    foot_grf_world=foot_grf_world,
                    current_foot_vz=current_foot_vz,
                    swing_phase=swing_phase,
                    waiting_for_recontact=True,
                )
                if planned_stance and waiting_for_recontact and (not rear_contact_ready):
                    contact_sequence[leg_id][0] = 0
                    min_swing_time_s = float(self.rear_touchdown_reacquire_min_swing_time_s)
                    ready_for_reacquire = (
                        min_swing_time_s <= 1e-9
                        or float(self.stc.swing_time[leg_id]) + 1e-12 >= min_swing_time_s
                    )
                    if ready_for_reacquire:
                        self.touchdown_reacquire_active[leg_id] = 1

        for leg_id in range(2, 4):
            remaining_s = float(self.rear_touchdown_close_lock_remaining_s[leg_id])
            if remaining_s <= 1e-9:
                self.rear_touchdown_close_lock_remaining_s[leg_id] = 0.0
                continue
            if bool(actual_contact[leg_id]):
                contact_sequence[leg_id][0] = 1
                self.rear_touchdown_close_lock_remaining_s[leg_id] = max(
                    0.0,
                    remaining_s - float(simulation_dt),
                )
            else:
                self.rear_touchdown_close_lock_remaining_s[leg_id] = 0.0

        if self.gait_name == 'crawl' and np.any(front_close_gap_keep_swing_mask):
            for leg_id in range(2):
                if not bool(front_close_gap_keep_swing_mask[leg_id]):
                    continue
                # When a front leg has already re-closed planned/current stance
                # but MuJoCo contact has not yet returned, keep the controller on
                # the swing side for one more control beat so the leg continues
                # descending instead of switching to stance too early.
                contact_sequence[leg_id][0] = 0

        if self.gait_name == 'crawl' and np.any(front_planted_seam_keep_swing_mask):
            for leg_id in range(2):
                if not bool(front_planted_seam_keep_swing_mask[leg_id]):
                    continue
                # After late full-contact recovery drops, a front leg can remain
                # physically planted even though it is already planned swing.
                # Keep that leg on the controller swing side for the short seam
                # tail so lift-off keeps progressing instead of re-latching
                # stance through the final low-height collapse window.
                contact_sequence[leg_id][0] = 0

        self.previous_contact = copy.deepcopy(self.current_contact)
        self.current_contact = np.array(
            [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]]
        )
        all_actual_contact_now = bool(np.all(np.asarray(actual_contact, dtype=int) == 1))
        for leg_id in range(2, 4):
            if int(self.planned_contact[leg_id]) == 0:
                self.rear_touchdown_retry_count[leg_id] = 0
                self.rear_close_handoff_remaining_s[leg_id] = 0.0
                continue
            if int(self.previous_contact[leg_id]) == 1 and int(self.current_contact[leg_id]) == 0:
                # This is a rear planned-stance reopen after a flaky first close.
                # The next reacquire should behave like a more vertical, more
                # committed search for ground rather than restarting a nominal swing.
                self.rear_touchdown_retry_count[leg_id] += 1
            elif int(self.current_contact[leg_id]) == 1 and bool(actual_contact[leg_id]):
                self.rear_touchdown_retry_count[leg_id] = 0
            close_after_late_actual_contact = bool(
                self.gait_name == 'crawl'
                and float(self.rear_close_handoff_hold_s) > 1e-9
                and int(self.previous_contact[leg_id]) == 0
                and int(self.current_contact[leg_id]) == 1
                and int(self.planned_contact[leg_id]) == 1
                and bool(actual_contact[leg_id])
                and bool(self.previous_actual_contact[leg_id])
                and all_actual_contact_now
            )
            if close_after_late_actual_contact:
                self.rear_close_handoff_remaining_s[leg_id] = max(
                    float(self.rear_close_handoff_remaining_s[leg_id]),
                    float(self.rear_close_handoff_hold_s),
                )
                self.rear_close_handoff_alpha_scale[leg_id] = max(
                    float(self.rear_close_handoff_alpha_scale[leg_id]),
                    1.0,
                )
            if not (
                int(self.current_contact[leg_id]) == 1
                and int(self.planned_contact[leg_id]) == 1
                and bool(actual_contact[leg_id])
            ):
                self.rear_close_handoff_remaining_s[leg_id] = 0.0
                self.rear_close_handoff_alpha_scale[leg_id] = 0.0
        self.rear_close_handoff_alpha = 0.0
        self.rear_close_handoff_leg_index = -1
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        late_load_share_hold_s = float(
            getattr(self, 'rear_late_load_share_support_hold_s', 0.0)
        )
        if not hasattr(self, 'rear_close_handoff_alpha_scale'):
            self.rear_close_handoff_alpha_scale = np.zeros(4, dtype=float)
        if not hasattr(self, 'rear_late_load_share_remaining_s'):
            self.rear_late_load_share_remaining_s = np.zeros(4, dtype=float)
        if not hasattr(self, 'rear_late_load_share_trigger_elapsed_s'):
            self.rear_late_load_share_trigger_elapsed_s = np.zeros(4, dtype=float)
        if not hasattr(self, 'rear_late_load_share_alpha_scale'):
            self.rear_late_load_share_alpha_scale = np.zeros(4, dtype=float)
        self.rear_late_load_share_active_debug[:] = 0
        self.rear_late_load_share_alpha_debug[:] = 0.0
        self.rear_late_load_share_candidate_active_debug[:] = 0
        self.rear_late_load_share_candidate_alpha_debug[:] = 0.0
        self.rear_late_load_share_trigger_enabled_debug = 0
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        if late_load_share_hold_s > 1e-9:
            front_planted_support_window_active = bool(
                float(getattr(self, 'rear_all_contact_front_planted_tail_alpha', 0.0)) > 1e-9
                or float(getattr(self.crawl_params, 'front_planted_seam_support_alpha', 0.0)) > 1e-9
            )
            late_load_share_trigger_enabled = bool(
                self.gait_name == 'crawl'
                and all_actual_contact_now
                and bool(np.all(np.asarray(self.planned_contact, dtype=int) == 1))
                and (
                    float(getattr(self, 'full_contact_recovery_active', 0.0)) > 0.5
                    or front_planted_support_window_active
                )
                and float(height_ratio)
                <= float(getattr(self, 'rear_late_load_share_support_height_ratio', 0.0))
                and rear_all_contact_posture_needed
            )
            self.rear_late_load_share_trigger_enabled_debug = int(late_load_share_trigger_enabled)
            late_load_share_candidate_active = np.zeros(4, dtype=int)
            late_load_share_candidate_alpha = np.zeros(4, dtype=float)
            if late_load_share_trigger_enabled:
                rear_vertical_grf = np.asarray(
                    [float(vertical_grf[2]), float(vertical_grf[3])],
                    dtype=float,
                )
                rear_total_vertical_grf = float(np.sum(rear_vertical_grf))
                if rear_total_vertical_grf > 1e-6:
                    # The late crawl collapse is driven more by left/right
                    # rear-pair imbalance than by total rear load alone.
                    rear_leg_shares = rear_vertical_grf / rear_total_vertical_grf
                    active_local_ids = np.flatnonzero(
                        np.asarray(self.rear_late_load_share_remaining_s[2:4], dtype=float) > 1e-9
                    )
                    if active_local_ids.size == 1:
                        # Once the late asymmetric rear support path is already
                        # active for one hind leg, keep that same leg latched
                        # until the window retires instead of letting the weak
                        # side bounce back and forth between RL/RR.
                        weak_local_idx = int(active_local_ids[0])
                    elif active_local_ids.size >= 2:
                        weak_local_idx = int(
                            active_local_ids[
                                np.argmax(
                                    np.asarray(
                                        self.rear_late_load_share_remaining_s[2:4],
                                        dtype=float,
                                    )[active_local_ids]
                                )
                            ]
                        )
                    else:
                        weak_local_idx = int(np.argmin(rear_leg_shares))
                    weak_leg_id = int(2 + weak_local_idx)
                    weak_leg_share = float(rear_leg_shares[weak_local_idx])
                    if (
                        weak_leg_share + 1e-12
                        < float(getattr(self, 'rear_late_load_share_support_min_leg_share', 0.0))
                        and bool(actual_contact[weak_leg_id])
                        and bool(int(self.current_contact[weak_leg_id]) == 1)
                        and bool(int(self.planned_contact[weak_leg_id]) == 1)
                    ):
                        # Allow this asymmetric leg-floor boost to coexist with
                        # the broader rear all-contact support path. The global
                        # rear-floor raise helps the whole hind pair, while this
                        # targeted alpha is intended to push extra load into the
                        # clearly under-sharing rear leg that still causes the
                        # late left/right roll collapse.
                        weak_leg_share_threshold = float(
                            getattr(self, 'rear_late_load_share_support_min_leg_share', 0.0)
                        )
                        # Scale the late asymmetric handoff by how clearly the
                        # weak rear leg is under-sharing load instead of
                        # immediately applying a full-strength override after a
                        # single threshold crossing.
                        weak_leg_share_deficit_alpha = float(
                            np.clip(
                                (weak_leg_share_threshold - weak_leg_share)
                                / max(weak_leg_share_threshold, 0.20),
                                0.0,
                                1.0,
                            )
                        )
                        late_load_share_candidate_active[weak_leg_id] = 1
                        late_load_share_candidate_alpha[weak_leg_id] = float(
                            np.clip(
                                weak_leg_share_deficit_alpha,
                                0.0,
                                float(
                                    getattr(
                                        self,
                                        'rear_late_load_share_support_alpha_cap',
                                        1.0,
                                    )
                                ),
                            )
                        )
            self.rear_late_load_share_candidate_active_debug[:] = late_load_share_candidate_active.astype(int)
            self.rear_late_load_share_candidate_alpha_debug[:] = late_load_share_candidate_alpha.astype(float)
            for leg_id in range(2, 4):
                if int(late_load_share_candidate_active[leg_id]) == 1:
                    self.rear_late_load_share_trigger_elapsed_s[leg_id] = min(
                        float(self.rear_late_load_share_trigger_elapsed_s[leg_id]) + float(simulation_dt),
                        max(
                            float(getattr(self, 'rear_late_load_share_support_min_persist_s', 0.0)),
                            late_load_share_hold_s,
                            float(simulation_dt),
                        ),
                    )
                else:
                    self.rear_late_load_share_trigger_elapsed_s[leg_id] = 0.0
                min_persist_s = float(
                    getattr(self, 'rear_late_load_share_support_min_persist_s', 0.0)
                )
                if (
                    int(late_load_share_candidate_active[leg_id]) == 1
                    and float(self.rear_late_load_share_trigger_elapsed_s[leg_id]) + 1e-12
                    >= max(min_persist_s, float(simulation_dt))
                ):
                    # Keep a dedicated weaker-leg crawl support window alive
                    # without re-arming the broader rear close-handoff path.
                    # The earlier shared path improved observability but ended
                    # up overextending the targeted rear handoff and regressed
                    # the long-horizon crawl baseline.
                    self.rear_late_load_share_remaining_s[leg_id] = max(
                        float(self.rear_late_load_share_remaining_s[leg_id]),
                        late_load_share_hold_s,
                    )
                    self.rear_late_load_share_alpha_scale[leg_id] = max(
                        float(self.rear_late_load_share_alpha_scale[leg_id]),
                        float(late_load_share_candidate_alpha[leg_id]),
                    )
                remaining_s = float(self.rear_late_load_share_remaining_s[leg_id])
                if remaining_s <= 1e-9:
                    self.rear_late_load_share_remaining_s[leg_id] = 0.0
                    self.rear_late_load_share_alpha_scale[leg_id] = 0.0
                    # Keep the trigger elapsed timer intact here so a weak-leg
                    # candidate can accumulate persistence before the support
                    # window is first armed.
                    continue
                if (
                    not bool(actual_contact[leg_id])
                    or not bool(int(self.current_contact[leg_id]) == 1)
                    or not bool(int(self.planned_contact[leg_id]) == 1)
                ):
                    self.rear_late_load_share_remaining_s[leg_id] = 0.0
                    self.rear_late_load_share_alpha_scale[leg_id] = 0.0
                    self.rear_late_load_share_trigger_elapsed_s[leg_id] = 0.0
                    continue
                time_alpha = float(
                    np.clip(
                        remaining_s / max(late_load_share_hold_s, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                alpha = float(
                    np.clip(
                        time_alpha * float(self.rear_late_load_share_alpha_scale[leg_id]),
                        0.0,
                        1.0,
                    )
                )
                self.rear_late_load_share_active_debug[leg_id] = int(alpha > 1e-9)
                self.rear_late_load_share_alpha_debug[leg_id] = float(alpha)
                if alpha > float(self.rear_late_load_share_alpha):
                    self.rear_late_load_share_alpha = alpha
                    self.rear_late_load_share_leg_index = int(leg_id)
                self.rear_late_load_share_remaining_s[leg_id] = max(
                    0.0,
                    remaining_s - float(simulation_dt),
                )
        else:
            self.rear_late_load_share_remaining_s[:] = 0.0
            self.rear_late_load_share_trigger_elapsed_s[:] = 0.0
            self.rear_late_load_share_alpha_scale[:] = 0.0
            self.rear_late_load_share_candidate_active_debug[:] = 0
            self.rear_late_load_share_candidate_alpha_debug[:] = 0.0
            self.rear_late_load_share_trigger_enabled_debug = 0
        for leg_id in range(2, 4):
            remaining_s = float(self.rear_close_handoff_remaining_s[leg_id])
            if remaining_s <= 1e-9 or float(self.rear_close_handoff_hold_s) <= 1e-9:
                self.rear_close_handoff_remaining_s[leg_id] = 0.0
                self.rear_close_handoff_alpha_scale[leg_id] = 0.0
                continue
            alpha_scale = float(
                np.clip(
                    float(self.rear_close_handoff_alpha_scale[leg_id]),
                    0.0,
                    1.0,
                )
            )
            alpha = float(
                np.clip(
                    (
                        remaining_s / max(float(self.rear_close_handoff_hold_s), 1e-6)
                    )
                    * alpha_scale,
                    0.0,
                    1.0,
                )
            )
            self.rear_close_handoff_active_debug[leg_id] = 1
            if alpha > float(self.rear_close_handoff_alpha):
                self.rear_close_handoff_alpha = alpha
                self.rear_close_handoff_leg_index = int(leg_id)
            self.rear_close_handoff_remaining_s[leg_id] = max(
                0.0,
                remaining_s - float(simulation_dt),
            )
            if float(self.rear_close_handoff_remaining_s[leg_id]) <= 1e-9:
                self.rear_close_handoff_alpha_scale[leg_id] = 0.0
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
            base_yaw_rate=float(base_ang_vel[2]),
            ref_base_yaw_rate=float(ref_base_ang_vel[2]),
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

        self.previous_feet_pos_world = feet_pos_array.copy()
        self.previous_feet_pos_world_valid = True

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
        foot_contact_state = None,
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
                    current_foot_xyz = np.asarray(feet_pos[leg_name], dtype=float).copy()
                    rear_retry_active = bool(leg_id >= 2 and int(self.rear_touchdown_retry_count[leg_id]) > 0)
                    xy_blend = self._touchdown_reacquire_xy_blend_for_leg(leg_id)
                    extra_depth = self._touchdown_reacquire_extra_depth_for_leg(leg_id)
                    forward_bias = self._touchdown_reacquire_forward_bias_for_leg(leg_id)
                    hold_current_xy = bool(
                        (leg_id < 2 and self.front_touchdown_reacquire_hold_current_xy)
                        or (leg_id >= 2 and self.rear_touchdown_reacquire_hold_current_xy)
                    )
                    if rear_retry_active:
                        hold_current_xy = True
                        extra_depth += 0.015
                    if hold_current_xy:
                        # When rear touchdown reacquire is late, holding the current
                        # xy and only pushing downward is often safer than chasing a
                        # stale nominal foothold that can drag the foot laterally.
                        swing_touch_down[0:2] = current_foot_xyz[0:2]
                    if (not hold_current_xy) and xy_blend > 0.0:
                        swing_touch_down[0:2] = (
                            (1.0 - xy_blend) * swing_touch_down[0:2]
                            + xy_blend * touchdown_ref[0:2]
                        )
                    if (not hold_current_xy) and abs(float(forward_bias)) > 1e-9:
                        swing_touch_down[0] += float(forward_bias)
                    max_xy_shift = float(self.rear_touchdown_reacquire_max_xy_shift)
                    if leg_id >= 2 and max_xy_shift > 1e-9:
                        delta_xy = np.asarray(swing_touch_down[0:2], dtype=float) - current_foot_xyz[0:2]
                        delta_norm = float(np.linalg.norm(delta_xy))
                        if delta_norm > max_xy_shift:
                            swing_touch_down[0:2] = (
                                current_foot_xyz[0:2] + (max_xy_shift / max(delta_norm, 1e-9)) * delta_xy
                            )
                    if extra_depth > 0.0:
                        current_foot_z = float(current_foot_xyz[2])
                        swing_touch_down[2] = min(
                            float(swing_touch_down[2]),
                            float(touchdown_ref[2]),
                            current_foot_z,
                        ) - extra_depth
                swing_time_override = None
                if leg_id >= 2 and int(self.touchdown_reacquire_active[leg_id]) == 1:
                    min_phase = float(self.rear_touchdown_reacquire_min_phase)
                    if int(self.rear_touchdown_retry_count[leg_id]) > 0:
                        min_phase = max(min_phase, 0.55)
                    if min_phase > 1e-9:
                        swing_time_override = max(
                            float(self.stc.swing_time[leg_id]),
                            min_phase * float(self.stc.swing_period),
                        )
                if cfg.mpc_params['type'] == 'linear_osqp' and self.gait_name == 'crawl':
                    if swing_time_override is not None:
                        self.latched_swing_time[leg_id] = min(
                            float(swing_time_override),
                            float(self.stc.swing_period),
                        )
                    tau[leg_name], des_foot_pos[leg_name], des_foot_vel[leg_name] = self._compute_latched_swing_torque(
                        leg_id=leg_id,
                        leg_name=leg_name,
                        q_dot=qvel[legs_qvel_idx[leg_name]],
                        J=feet_jac[leg_name][:, legs_qvel_idx[leg_name]],
                        J_dot=feet_jac_dot[leg_name][:, legs_qvel_idx[leg_name]],
                        touch_down=swing_touch_down,
                        foot_pos=feet_pos[leg_name],
                        foot_vel=feet_vel[leg_name],
                        h=legs_qfrc_bias[leg_name],
                        mass_matrix=legs_mass_matrix[leg_name],
                    )
                else:
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
                            swing_time_override=swing_time_override,
                        )
                    )
                if leg_id >= 2 and int(self.touchdown_reacquire_active[leg_id]) == 1:
                    upward_vel_damping = float(self.rear_touchdown_reacquire_upward_vel_damping)
                    if int(self.rear_touchdown_retry_count[leg_id]) > 0:
                        upward_vel_damping += 40.0
                    current_foot_vz = float(np.asarray(feet_vel[leg_name], dtype=float)[2])
                    if upward_vel_damping > 1e-9 and current_foot_vz > 0.0:
                        damping_force = np.array([0.0, 0.0, -upward_vel_damping * current_foot_vz], dtype=float)
                        tau[leg_name] += (
                            feet_jac[leg_name][:, legs_qvel_idx[leg_name]].T @ damping_force
                        )
                    if rear_retry_active:
                        retry_descent_depth = float(self.rear_touchdown_retry_descent_depth)
                        retry_descent_kp = float(self.rear_touchdown_retry_descent_kp)
                        retry_descent_kd = float(self.rear_touchdown_retry_descent_kd)
                        if retry_descent_depth > 1e-9 and (retry_descent_kp > 1e-9 or retry_descent_kd > 1e-9):
                            retry_target_z = min(
                                float(swing_touch_down[2]),
                                float(current_foot_xyz[2]) - retry_descent_depth,
                            )
                            z_err = retry_target_z - float(current_foot_xyz[2])
                            descent_force_z = retry_descent_kp * z_err - retry_descent_kd * current_foot_vz
                            if descent_force_z < 0.0:
                                descent_force = np.array([0.0, 0.0, descent_force_z], dtype=float)
                                tau[leg_name] += (
                                    feet_jac[leg_name][:, legs_qvel_idx[leg_name]].T @ descent_force
                                )
                                des_foot_pos[leg_name][2] = min(float(des_foot_pos[leg_name][2]), retry_target_z)
                                des_foot_vel[leg_name][2] = min(float(des_foot_vel[leg_name][2]), 0.0)
                if int(self.swing_contact_release_active[leg_id]) == 1 and bool(actual_contact[leg_id]):
                    if leg_id < 2:
                        release_lift_height = float(self.front_release_lift_height)
                        release_lift_kp = float(self.front_release_lift_kp)
                        release_lift_kd = float(self.front_release_lift_kd)
                    else:
                        release_lift_height = float(self.rear_release_lift_height)
                        release_lift_kp = float(self.rear_release_lift_kp)
                        release_lift_kd = float(self.rear_release_lift_kd)
                    if release_lift_height > 1e-9 and (release_lift_kp > 1e-9 or release_lift_kd > 1e-9):
                        current_foot_xyz = np.asarray(feet_pos[leg_name], dtype=float).reshape(3)
                        current_foot_vz = float(np.asarray(feet_vel[leg_name], dtype=float).reshape(3)[2])
                        release_target_z = max(
                            float(np.asarray(des_foot_pos[leg_name], dtype=float).reshape(3)[2]),
                            float(current_foot_xyz[2]) + release_lift_height,
                        )
                        z_err = release_target_z - float(current_foot_xyz[2])
                        lift_force_z = release_lift_kp * z_err - release_lift_kd * current_foot_vz
                        if lift_force_z > 0.0:
                            lift_force = np.array([0.0, 0.0, lift_force_z], dtype=float)
                            tau[leg_name] += (
                                feet_jac[leg_name][:, legs_qvel_idx[leg_name]].T @ lift_force
                            )
                            des_foot_pos[leg_name][2] = release_target_z
                            des_foot_vel[leg_name][2] = max(float(des_foot_vel[leg_name][2]), 0.0)
            else:
                if cfg.mpc_params['type'] == 'linear_osqp':
                    # Holding the stance foot near its touchdown location avoids
                    # dragging grounded feet toward the next foothold target.
                    # A tiny xy blend can recover forward progression without
                    # reintroducing the full stance-foot dragging issue.
                    des_foot_pos[leg_name] = np.array(self.frg.touch_down_positions[leg_name], copy=True)
                    actual_foot_pos = np.asarray(feet_pos[leg_name], dtype=float).copy()
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
                    rear_all_contact_alpha = float(getattr(self, 'rear_all_contact_stabilization_alpha', 0.0))
                    if rear_all_contact_alpha > 1e-9:
                        if leg_id < 2:
                            rear_all_contact_anchor_z_blend = float(
                                np.clip(
                                    getattr(
                                        cfg,
                                        'linear_osqp_params',
                                        {},
                                    ).get('rear_all_contact_stabilization_front_anchor_z_blend', 0.0),
                                    0.0,
                                    1.0,
                                )
                            )
                            rear_all_contact_anchor_z_max_delta = max(
                                float(
                                    getattr(
                                        cfg,
                                        'linear_osqp_params',
                                        {},
                                    ).get('rear_all_contact_stabilization_front_anchor_z_max_delta', 0.0)
                                ),
                                0.0,
                            )
                        else:
                            rear_all_contact_anchor_z_blend = float(
                                np.clip(
                                    getattr(
                                        cfg,
                                        'linear_osqp_params',
                                        {},
                                    ).get('rear_all_contact_stabilization_rear_anchor_z_blend', 0.0),
                                    0.0,
                                    1.0,
                                )
                            )
                            rear_all_contact_anchor_z_max_delta = max(
                                float(
                                    getattr(
                                        cfg,
                                        'linear_osqp_params',
                                        {},
                                    ).get('rear_all_contact_stabilization_rear_anchor_z_max_delta', 0.0)
                                ),
                                0.0,
                            )
                        if rear_all_contact_anchor_z_blend > 1e-9:
                            z_alpha = float(np.clip(rear_all_contact_alpha * rear_all_contact_anchor_z_blend, 0.0, 1.0))
                            des_foot_pos[leg_name][2] = (
                                (1.0 - z_alpha) * float(des_foot_pos[leg_name][2])
                                + z_alpha * float(actual_foot_pos[2])
                            )
                        if rear_all_contact_anchor_z_max_delta > 1e-9:
                            max_anchor_z = float(actual_foot_pos[2]) + float(rear_all_contact_anchor_z_max_delta)
                            if float(des_foot_pos[leg_name][2]) > max_anchor_z:
                                des_foot_pos[leg_name][2] = max_anchor_z
                    is_latched_swing = int(self.planned_contact[leg_id]) == 0
                    release_alpha = self.get_latched_release_alpha(leg_id) if is_latched_swing else 0.0
                    linear_params = getattr(cfg, 'linear_osqp_params', {})
                    stance_blend = float(
                        np.clip(linear_params.get('stance_target_blend', 0.0), 0.0, 1.0)
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


        if cfg.mpc_params['type'] == 'linear_osqp':
            for leg_id, leg_name in enumerate(self.legs_order):
                if int(self.touchdown_support_active[leg_id]) != 1:
                    continue
                if not bool(actual_contact[leg_id]):
                    continue
                z_damping = float(self._touchdown_contact_vel_z_damping_for_leg(leg_id))
                if z_damping <= 1e-9:
                    continue
                foot_vel_now = np.asarray(feet_vel[leg_name], dtype=float).reshape(3)
                damping_force = np.array([0.0, 0.0, -z_damping * float(foot_vel_now[2])], dtype=float)
                tau[leg_name] += feet_jac[leg_name][:, legs_qvel_idx[leg_name]].T @ damping_force

        self.last_des_foot_pos = des_foot_pos
        self.last_des_foot_vel = des_foot_vel

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
        self.swing_contact_release_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_wait_s = np.zeros(4, dtype=float)
        self.support_contact_confirm_bypass_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_confirm_elapsed_s = np.zeros(4, dtype=float)
        self.touchdown_settle_remaining_s = np.zeros(4, dtype=float)
        self.rear_touchdown_actual_contact_elapsed_s = np.zeros(4, dtype=float)
        self.rear_late_seam_elapsed_s = np.zeros(4, dtype=float)
        self.rear_close_handoff_remaining_s = np.zeros(4, dtype=float)
        self.rear_close_handoff_alpha_scale = np.zeros(4, dtype=float)
        self.rear_late_load_share_remaining_s = np.zeros(4, dtype=float)
        self.rear_late_load_share_trigger_elapsed_s = np.zeros(4, dtype=float)
        self.rear_touchdown_close_lock_remaining_s = np.zeros(4, dtype=float)
        self.rear_touchdown_retry_count = np.zeros(4, dtype=int)
        self.rear_stable_stance_elapsed_s = np.zeros(4, dtype=float)
        self.previous_feet_pos_world = np.stack(
            [np.asarray(initial_feet_pos[leg_name], dtype=float).reshape(3) for leg_name in self.legs_order],
            axis=0,
        )
        self.previous_feet_pos_world_valid = True
        self.front_margin_rescue_remaining_s = np.zeros(4, dtype=float)
        self.front_margin_rescue_alpha = np.zeros(4, dtype=float)
        self.front_margin_rescue_recent_swing_remaining_s = np.zeros(2, dtype=float)
        self.touchdown_reacquire_armed = np.zeros(4, dtype=int)
        self.rear_handoff_support_remaining_s = 0.0
        self.rear_handoff_support_active = 0
        self.rear_handoff_support_mask = np.zeros(4, dtype=int)
        self.rear_swing_bridge_remaining_s = 0.0
        self.rear_swing_bridge_active = 0
        self.rear_swing_release_support_remaining_s = 0.0
        self.rear_swing_release_support_active = 0
        self.last_support_margin = np.full(4, np.nan, dtype=float)
        self.last_support_margin_query_xy = np.full((4, 2), np.nan, dtype=float)
        self.front_late_release_active = np.zeros(4, dtype=int)
        self.swing_contact_release_active = np.zeros(4, dtype=int)
        self.support_confirm_active = np.zeros(4, dtype=int)
        self.pre_swing_gate_active = np.zeros(4, dtype=int)
        self.touchdown_reacquire_active = np.zeros(4, dtype=int)
        self.touchdown_confirm_active = np.zeros(4, dtype=int)
        self.touchdown_settle_active = np.zeros(4, dtype=int)
        self.touchdown_support_active = np.zeros(4, dtype=int)
        self.rear_touchdown_pending_confirm = np.zeros(4, dtype=int)
        self.rear_late_seam_support_active_debug = np.zeros(4, dtype=int)
        self.rear_close_handoff_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_alpha_debug = np.zeros(4, dtype=float)
        self.rear_late_load_share_candidate_active_debug = np.zeros(4, dtype=int)
        self.rear_late_load_share_candidate_alpha_debug = np.zeros(4, dtype=float)
        self.rear_late_load_share_trigger_enabled_debug = 0
        self.full_contact_recovery_trigger_debug = 0
        self.front_delayed_swing_recovery_trigger_debug = 0
        self.planted_front_recovery_trigger_debug = 0
        self.planted_front_postdrop_recovery_trigger_debug = 0
        self.front_close_gap_trigger_debug = 0
        self.front_late_rearm_trigger_debug = 0
        self.front_planted_posture_tail_trigger_debug = 0
        self.front_late_posture_tail_trigger_debug = 0
        self.front_margin_rescue_active = np.zeros(4, dtype=int)
        self.touchdown_support_alpha = 0.0
        self.front_touchdown_support_alpha = 0.0
        self.rear_touchdown_support_alpha = 0.0
        self.rear_all_contact_stabilization_alpha = 0.0
        self.rear_all_contact_front_planted_tail_alpha = 0.0
        self.crawl_state = CrawlState()
        self.rear_close_handoff_alpha = 0.0
        self.rear_close_handoff_leg_index = -1
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        self.touchdown_contact_vel_z_damping = 0.0
        self.front_touchdown_contact_vel_z_damping = 0.0
        self.rear_touchdown_contact_vel_z_damping = 0.0
        self.full_contact_recovery_remaining_s = 0.0
        self.front_rear_transition_guard_post_recovery_remaining_s = 0.0
        self.rear_all_contact_front_planted_tail_remaining_s = 0.0
        self.rear_all_contact_front_planted_tail_alpha_scale = 1.0
        self.front_delayed_swing_recovery_spent = np.zeros(2, dtype=int)
        self.front_planted_swing_recovery_spent = np.zeros(2, dtype=int)
        self.front_late_rearm_used_s = np.zeros(2, dtype=float)
        self.front_touchdown_support_recent_remaining_s = 0.0
        self.rear_swing_bridge_recent_front_remaining_s = 0.0
        self.full_contact_recovery_active = 0
        self.full_contact_recovery_alpha = 0.0
        self.last_gate_forward_scale = 1.0
        self.rear_transition_manager.reset()
        self._sync_rear_transition_debug_arrays()
        self.rear_retry_contact_signal_debug[:] = 0
        self.rear_touchdown_contact_ready_debug[:] = 0
        self.rear_late_stance_contact_ready_debug[:] = 0
        self.rear_all_contact_support_needed_debug[:] = 0
        self.rear_late_seam_support_active_debug[:] = 0
        self.rear_close_handoff_active_debug[:] = 0
        self.rear_late_load_share_active_debug[:] = 0
        self.rear_late_load_share_alpha_debug[:] = 0.0
        self.rear_late_load_share_candidate_active_debug[:] = 0
        self.rear_late_load_share_candidate_alpha_debug[:] = 0.0
        self.rear_late_load_share_trigger_enabled_debug = 0
        self.full_contact_recovery_trigger_debug = 0
        self.front_delayed_swing_recovery_trigger_debug = 0
        self.planted_front_recovery_trigger_debug = 0
        self.planted_front_postdrop_recovery_trigger_debug = 0
        self.front_close_gap_trigger_debug = 0
        self.front_late_rearm_trigger_debug = 0
        self.front_planted_posture_tail_trigger_debug = 0
        self.front_late_posture_tail_trigger_debug = 0
        self.rear_close_handoff_alpha = 0.0
        self.rear_close_handoff_leg_index = -1
        self.rear_late_load_share_alpha = 0.0
        self.rear_late_load_share_leg_index = -1
        self._refresh_linear_timing_params()
        self.startup_full_stance_elapsed_s = 0.0
        return
