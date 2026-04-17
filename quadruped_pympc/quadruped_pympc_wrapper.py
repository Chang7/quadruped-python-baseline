import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc import config as cfg
from quadruped_pympc.interfaces.srbd_batched_controller_interface import SRBDBatchedControllerInterface
from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
from quadruped_pympc.interfaces.wb_interface import WBInterface

_DEFAULT_OBS = ("ref_base_height", "ref_base_angles", "nmpc_GRFs", "nmpc_footholds", "swing_time")


class QuadrupedPyMPC_Wrapper:
    """A simple class wrapper of all the mpc submodules (swing, contact generator, mpc itself)."""

    def __init__(
        self,
        initial_feet_pos: LegsAttr,
        legs_order: tuple[str, str, str, str] = ('FL', 'FR', 'RL', 'RR'),
        feet_geom_id: LegsAttr = None,
        quadrupedpympc_observables_names: tuple[str, ...] = _DEFAULT_OBS,
    ):
        """Constructor of the QuadrupedPyMPC_Wrapper class.

        Args:
            initial_feet_pos (LegsAttr): initial feet positions, otherwise they will be all zero.
            legs_order (tuple[str, str, str, str], optional): order of the leg. Defaults to ('FL', 'FR', 'RL', 'RR').
            quadrupedpympc_observables_names (tuple[str, ...], optional): list of observable to save. Defaults to _DEFAULT_OBS.
        """

        self.mpc_frequency = cfg.simulation_params["mpc_frequency"]

        self.srbd_controller_interface = SRBDControllerInterface()

        if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['optimize_step_freq']:
            self.srbd_batched_controller_interface = SRBDBatchedControllerInterface()

        self.wb_interface = WBInterface(initial_feet_pos=initial_feet_pos(frame='world'), legs_order=legs_order, feet_geom_id =  feet_geom_id)

        self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_pos = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_acc = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_predicted_state = np.zeros(12)
        self.best_sample_freq = self.wb_interface.pgg.step_freq
        self.applied_linear_support_force_floor_ratio = 0.0
        self.applied_linear_rear_handoff_leg_index = -1
        self.applied_linear_rear_handoff_leg_floor_scale = 0.0
        self.applied_linear_latched_force_scale = 0.0
        self.applied_linear_latched_front_receiver_scale = 0.0
        self.applied_linear_latched_rear_receiver_scale = 0.0
        self.linear_solve_total_ms = 0.0
        self.linear_solve_setup_ms = 0.0
        self.linear_solve_wall_ms = 0.0
        self.linear_solve_iter = 0.0

        self.quadrupedpympc_observables_names = quadrupedpympc_observables_names
        self.quadrupedpympc_observables = {}

    @staticmethod
    def _apply_linear_touchdown_support_overrides(
        front_alpha: float,
        rear_alpha: float,
        recovery_alpha: float = 0.0,
        rear_all_contact_alpha: float = 0.0,
        rear_all_contact_front_planted_tail_alpha: float = 0.0,
        rear_all_contact_weak_leg_index: int = -1,
        rear_all_contact_weak_leg_alpha: float = 0.0,
        rear_close_handoff_leg_index: int = -1,
        rear_close_handoff_alpha: float = 0.0,
        rear_late_load_share_leg_index: int = -1,
        rear_late_load_share_alpha: float = 0.0,
    ) -> dict | None:
        if cfg.mpc_params['type'] != 'linear_osqp':
            return None
        front_alpha = float(np.clip(front_alpha, 0.0, 1.0))
        rear_alpha = float(np.clip(rear_alpha, 0.0, 1.0))
        recovery_alpha = float(np.clip(recovery_alpha, 0.0, 1.0))
        rear_all_contact_alpha = float(np.clip(rear_all_contact_alpha, 0.0, 1.0))
        rear_all_contact_front_planted_tail_alpha = float(
            np.clip(rear_all_contact_front_planted_tail_alpha, 0.0, 1.0)
        )
        rear_all_contact_weak_leg_alpha = float(np.clip(rear_all_contact_weak_leg_alpha, 0.0, 1.0))
        rear_close_handoff_alpha = float(np.clip(rear_close_handoff_alpha, 0.0, 1.0))
        rear_late_load_share_alpha = float(np.clip(rear_late_load_share_alpha, 0.0, 1.0))
        if max(
            front_alpha,
            rear_alpha,
            recovery_alpha,
            rear_all_contact_alpha,
            rear_all_contact_front_planted_tail_alpha,
            rear_all_contact_weak_leg_alpha,
            rear_close_handoff_alpha,
            rear_late_load_share_alpha,
        ) <= 1e-9:
            return None
        if not isinstance(getattr(cfg, 'linear_osqp_params', None), dict):
            return None

        backup = cfg.linear_osqp_params.copy()

        def _raise_from_backup(name: str, delta: float, lower: float = 0.0, upper: float | None = None) -> None:
            base = float(backup.get(name, cfg.linear_osqp_params.get(name, 0.0)))
            value = base + float(delta)
            value = max(lower, value)
            if upper is not None:
                value = min(upper, value)
            cfg.linear_osqp_params[name] = value

        def _blend_to_backup_target(
            name: str,
            target_name: str,
            alpha: float,
            lower: float = 0.0,
            upper: float | None = None,
        ) -> None:
            base = float(backup.get(name, cfg.linear_osqp_params.get(name, 0.0)))
            target = float(backup.get(target_name, base))
            value = base + float(alpha) * (target - base)
            value = max(lower, value)
            if upper is not None:
                value = min(upper, value)
            cfg.linear_osqp_params[name] = value

        if front_alpha > 1e-9:
            _raise_from_backup("rear_floor_base_scale", front_alpha * float(backup.get("touchdown_support_rear_floor_delta", 0.0)), lower=0.0, upper=1.5)
            _raise_from_backup("reduced_support_vertical_boost", front_alpha * float(backup.get("touchdown_support_vertical_boost", 0.0)), lower=0.0)
            _raise_from_backup(
                "min_vertical_force_scale",
                front_alpha * float(backup.get("touchdown_support_min_vertical_force_scale_delta", 0.0)),
                lower=0.0,
                upper=2.0,
            )
            _raise_from_backup(
                "grf_max_scale",
                front_alpha * float(backup.get("touchdown_support_grf_max_scale_delta", 0.0)),
                lower=0.0,
                upper=1.0,
            )
            _raise_from_backup("z_pos_gain", front_alpha * float(backup.get("touchdown_support_z_pos_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("roll_angle_gain", front_alpha * float(backup.get("touchdown_support_roll_angle_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("roll_rate_gain", front_alpha * float(backup.get("touchdown_support_roll_rate_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("pitch_angle_gain", front_alpha * float(backup.get("touchdown_support_pitch_angle_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("pitch_rate_gain", front_alpha * float(backup.get("touchdown_support_pitch_rate_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("side_rebalance_gain", front_alpha * float(backup.get("touchdown_support_side_rebalance_delta", 0.0)), lower=0.0)

        if rear_alpha > 1e-9:
            _raise_from_backup("support_force_floor_ratio", rear_alpha * float(backup.get("rear_touchdown_support_support_floor_delta", 0.0)), lower=0.0, upper=1.0)
            _raise_from_backup("reduced_support_vertical_boost", rear_alpha * float(backup.get("rear_touchdown_support_vertical_boost", 0.0)), lower=0.0)
            _raise_from_backup(
                "min_vertical_force_scale",
                rear_alpha * float(backup.get("rear_touchdown_support_min_vertical_force_scale_delta", 0.0)),
                lower=0.0,
                upper=2.0,
            )
            _raise_from_backup(
                "grf_max_scale",
                rear_alpha * float(backup.get("rear_touchdown_support_grf_max_scale_delta", 0.0)),
                lower=0.0,
                upper=1.0,
            )
            _raise_from_backup("z_pos_gain", rear_alpha * float(backup.get("rear_touchdown_support_z_pos_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("roll_angle_gain", rear_alpha * float(backup.get("rear_touchdown_support_roll_angle_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("roll_rate_gain", rear_alpha * float(backup.get("rear_touchdown_support_roll_rate_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("pitch_angle_gain", rear_alpha * float(backup.get("rear_touchdown_support_pitch_angle_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("pitch_rate_gain", rear_alpha * float(backup.get("rear_touchdown_support_pitch_rate_gain_delta", 0.0)), lower=0.0)
            _raise_from_backup("side_rebalance_gain", rear_alpha * float(backup.get("rear_touchdown_support_side_rebalance_delta", 0.0)), lower=0.0)

        if recovery_alpha > 1e-9:
            _raise_from_backup(
                "support_force_floor_ratio",
                recovery_alpha * float(backup.get("full_contact_recovery_support_floor_delta", 0.0)),
                lower=0.0,
                upper=1.0,
            )
            _raise_from_backup(
                "min_vertical_force_scale",
                recovery_alpha * float(backup.get("full_contact_recovery_min_vertical_force_scale_delta", 0.0)),
                lower=0.0,
                upper=2.0,
            )
            _raise_from_backup(
                "z_pos_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_z_pos_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_angle_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_roll_angle_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_rate_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_roll_rate_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "pitch_angle_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_pitch_angle_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "pitch_rate_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_pitch_rate_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "side_rebalance_gain",
                recovery_alpha * float(backup.get("full_contact_recovery_side_rebalance_delta", 0.0)),
                lower=0.0,
            )

        if rear_all_contact_alpha > 1e-9:
            _raise_from_backup(
                "rear_floor_base_scale",
                rear_all_contact_alpha * float(backup.get("rear_all_contact_stabilization_rear_floor_delta", 0.0)),
                lower=0.0,
                upper=1.5,
            )
            _raise_from_backup(
                "z_pos_gain",
                rear_all_contact_alpha * float(backup.get("rear_all_contact_stabilization_z_pos_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_angle_gain",
                rear_all_contact_alpha * float(backup.get("rear_all_contact_stabilization_roll_angle_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_rate_gain",
                rear_all_contact_alpha * float(backup.get("rear_all_contact_stabilization_roll_rate_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "side_rebalance_gain",
                rear_all_contact_alpha * float(backup.get("rear_all_contact_stabilization_side_rebalance_delta", 0.0)),
                lower=0.0,
            )

        if rear_all_contact_front_planted_tail_alpha > 1e-9:
            _raise_from_backup(
                "support_force_floor_ratio",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_support_floor_delta", 0.0)),
                lower=0.0,
                upper=1.0,
            )
            _raise_from_backup(
                "rear_floor_base_scale",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_rear_floor_delta", 0.0)),
                lower=0.0,
                upper=1.5,
            )
            _raise_from_backup(
                "z_pos_gain",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_z_pos_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_angle_gain",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_roll_angle_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "roll_rate_gain",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_roll_rate_gain_delta", 0.0)),
                lower=0.0,
            )
            _raise_from_backup(
                "side_rebalance_gain",
                rear_all_contact_front_planted_tail_alpha
                * float(backup.get("rear_all_contact_front_planted_side_rebalance_delta", 0.0)),
                lower=0.0,
            )
            _blend_to_backup_target(
                "latched_force_scale",
                "rear_all_contact_front_planted_latched_force_scale_target",
                rear_all_contact_front_planted_tail_alpha,
                lower=0.0,
                upper=1.0,
            )
            _blend_to_backup_target(
                "latched_front_receiver_scale",
                "rear_all_contact_front_planted_latched_front_receiver_scale_target",
                rear_all_contact_front_planted_tail_alpha,
                lower=0.0,
            )
            _blend_to_backup_target(
                "latched_rear_receiver_scale",
                "rear_all_contact_front_planted_latched_rear_receiver_scale_target",
                rear_all_contact_front_planted_tail_alpha,
                lower=0.0,
            )

        targeted_rear_leg_index = -1
        targeted_rear_leg_floor_delta = 0.0
        if rear_close_handoff_alpha > 1e-9 and int(rear_close_handoff_leg_index) in (2, 3):
            targeted_rear_leg_index = int(rear_close_handoff_leg_index)
            targeted_rear_leg_floor_delta = max(
                targeted_rear_leg_floor_delta,
                rear_close_handoff_alpha * float(backup.get("rear_close_handoff_leg_floor_scale_delta", 0.0)),
            )
        if rear_late_load_share_alpha > 1e-9 and int(rear_late_load_share_leg_index) in (2, 3):
            late_load_share_delta = (
                rear_late_load_share_alpha
                * float(backup.get("rear_late_load_share_support_leg_floor_scale_delta", 0.0))
            )
            if int(rear_late_load_share_leg_index) == targeted_rear_leg_index:
                targeted_rear_leg_floor_delta += late_load_share_delta
            elif late_load_share_delta > targeted_rear_leg_floor_delta:
                targeted_rear_leg_index = int(rear_late_load_share_leg_index)
                targeted_rear_leg_floor_delta = late_load_share_delta
        if rear_all_contact_weak_leg_alpha > 1e-9 and int(rear_all_contact_weak_leg_index) in (2, 3):
            weak_leg_delta = (
                rear_all_contact_weak_leg_alpha
                * float(backup.get("rear_all_contact_stabilization_weak_leg_floor_delta", 0.0))
            )
            if int(rear_all_contact_weak_leg_index) == targeted_rear_leg_index:
                targeted_rear_leg_floor_delta += weak_leg_delta
            elif weak_leg_delta > targeted_rear_leg_floor_delta:
                targeted_rear_leg_index = int(rear_all_contact_weak_leg_index)
                targeted_rear_leg_floor_delta = weak_leg_delta
        if targeted_rear_leg_floor_delta > 1e-9 and targeted_rear_leg_index in (2, 3):
            cfg.linear_osqp_params["rear_handoff_leg_index"] = int(targeted_rear_leg_index)
            _raise_from_backup(
                "rear_handoff_leg_floor_scale",
                targeted_rear_leg_floor_delta,
                lower=0.0,
                upper=2.0,
            )

        return backup

    def compute_actions(
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
        step_num: int,
        qpos: np.ndarray,
        qvel: np.ndarray,
        feet_jac: LegsAttr,
        feet_jac_dot: LegsAttr,
        feet_vel: LegsAttr,
        legs_qfrc_passive: LegsAttr,
        legs_qfrc_bias: LegsAttr,
        legs_mass_matrix: LegsAttr,
        legs_qpos_idx: LegsAttr,
        legs_qvel_idx: LegsAttr,
        tau: LegsAttr,
        inertia: np.ndarray,
        mujoco_contact: np.ndarray,
        foot_contact_state = None,
        foot_grf_state = None,
    ) -> LegsAttr:
        """Given the current state of the robot (and the reference),
            compute the torques to be applied to the motors.

        Args:
            com_pos (np.ndarray): center of mass position in
            base_pos (np.ndarray): base position in world frame
            base_lin_vel (np.ndarray): base velocity in world frame
            base_ori_euler_xyz (np.ndarray): base orientation in world frame
            base_ang_vel (np.ndarray): base angular velocity in base frame
            feet_pos (LegsAttr): locations of the feet in world frame
            hip_pos (LegsAttr): locations of the hip in world frame
            heightmaps (_type_): TODO
            legs_order (tuple[str, str, str, str]): order of the legs
            simulation_dt (float): simulation time step
            ref_base_lin_vel (np.ndarray): reference base linear velocity in world frame
            ref_base_ang_vel (np.ndarray): reference base angular velocity in world frame
            step_num (int): current step number of the environment
            qpos (np.ndarray): joint positions
            qvel (np.ndarray): joint velocities
            feet_jac (LegsAttr): jacobian of the feet
            feet_jac_dot (LegsAttr): derivative of the jacobian of the feet
            feet_vel (LegsAttr): velocity of the feet
            legs_qfrc_passive (LegsAttr): passive forces acting on the joints
            legs_qfrc_bias (LegsAttr): gravity compensation, coriolis and centrifugal forces
            legs_mass_matrix (LegsAttr): mass matrix of the legs
            legs_qvel_idx (LegsAttr): indices of the joint velocities
            tau (LegsAttr): joint torques
            inertia (np.ndarray): inertia matrix of the robot (CCRBI)

        Returns:
            LegsAttr: torques to be applied to the motors
        """

        # Update the state and reference -------------------------
        state_current, ref_state, contact_sequence, step_height, optimize_swing = (
            self.wb_interface.update_state_and_reference(
                com_pos,
                base_pos,
                base_lin_vel,
                base_ori_euler_xyz,
                base_ang_vel,
                feet_pos,
                hip_pos,
                joints_pos,
                heightmaps,
                legs_order,
                simulation_dt,
                ref_base_lin_vel,
                ref_base_ang_vel,
                mujoco_contact,
                foot_contact_state,
                foot_grf_state,
            )
        )

        # Solve OCP ---------------------------------------------------------------------------------------
        support_override_backup = None
        support_alpha = 0.0
        self.applied_linear_support_force_floor_ratio = 0.0
        self.applied_linear_rear_handoff_leg_index = -1
        self.applied_linear_rear_handoff_leg_floor_scale = 0.0
        self.applied_linear_latched_force_scale = 0.0
        self.applied_linear_latched_front_receiver_scale = 0.0
        self.applied_linear_latched_rear_receiver_scale = 0.0
        if step_num % round(1 / (self.mpc_frequency * simulation_dt)) == 0:
            if cfg.mpc_params['type'] == 'linear_osqp':
                front_support_alpha = float(getattr(self.wb_interface, 'front_touchdown_support_alpha', 0.0))
                rear_support_alpha = float(getattr(self.wb_interface, 'rear_touchdown_support_alpha', 0.0))
                recovery_alpha = float(getattr(self.wb_interface, 'full_contact_recovery_alpha', 0.0))
                rear_all_contact_alpha = float(
                    getattr(self.wb_interface, 'rear_all_contact_stabilization_alpha', 0.0)
                )
                rear_all_contact_front_planted_tail_alpha = float(
                    getattr(self.wb_interface, 'rear_all_contact_front_planted_tail_alpha', 0.0)
                )
                crawl_front_planted_seam_support_alpha = float(
                    getattr(self.wb_interface, 'crawl_front_planted_seam_support_alpha', 0.0)
                )
                front_planted_support_alpha = max(
                    rear_all_contact_front_planted_tail_alpha,
                    crawl_front_planted_seam_support_alpha,
                )
                rear_all_contact_weak_leg_index = int(
                    getattr(self.wb_interface, 'rear_all_contact_weak_leg_index', -1)
                )
                rear_all_contact_weak_leg_alpha = float(
                    getattr(self.wb_interface, 'rear_all_contact_weak_leg_alpha', 0.0)
                )
                rear_close_handoff_alpha = float(
                    getattr(self.wb_interface, 'rear_close_handoff_alpha', 0.0)
                )
                rear_close_handoff_leg_index = int(
                    getattr(self.wb_interface, 'rear_close_handoff_leg_index', -1)
                )
                rear_late_load_share_alpha = float(
                    getattr(self.wb_interface, 'rear_late_load_share_alpha', 0.0)
                )
                rear_late_load_share_leg_index = int(
                    getattr(self.wb_interface, 'rear_late_load_share_leg_index', -1)
                )
                support_override_backup = self._apply_linear_touchdown_support_overrides(
                    front_support_alpha,
                    rear_support_alpha,
                    recovery_alpha,
                    rear_all_contact_alpha,
                    front_planted_support_alpha,
                    rear_all_contact_weak_leg_index,
                    rear_all_contact_weak_leg_alpha,
                    rear_close_handoff_leg_index,
                    rear_close_handoff_alpha,
                    rear_late_load_share_leg_index,
                    rear_late_load_share_alpha,
                )
                if support_override_backup is not None:
                    self.applied_linear_support_force_floor_ratio = float(
                        cfg.linear_osqp_params.get("support_force_floor_ratio", 0.0)
                    )
                    self.applied_linear_rear_handoff_leg_index = int(
                        cfg.linear_osqp_params.get("rear_handoff_leg_index", -1)
                    )
                    self.applied_linear_rear_handoff_leg_floor_scale = float(
                        cfg.linear_osqp_params.get("rear_handoff_leg_floor_scale", 0.0)
                    )
                    self.applied_linear_latched_force_scale = float(
                        cfg.linear_osqp_params.get("latched_force_scale", 0.0)
                    )
                    self.applied_linear_latched_front_receiver_scale = float(
                        cfg.linear_osqp_params.get("latched_front_receiver_scale", 0.0)
                    )
                    self.applied_linear_latched_rear_receiver_scale = float(
                        cfg.linear_osqp_params.get("latched_rear_receiver_scale", 0.0)
                    )
            try:
                (
                    self.nmpc_GRFs,
                    self.nmpc_footholds,
                    self.nmpc_joints_pos,
                    self.nmpc_joints_vel,
                    self.nmpc_joints_acc,
                    self.best_sample_freq,
                    self.nmpc_predicted_state,
                ) = self.srbd_controller_interface.compute_control(
                    state_current,
                    ref_state,
                    contact_sequence,
                    inertia,
                    self.wb_interface.pgg.phase_signal,
                    self.wb_interface.pgg.step_freq,
                    optimize_swing,
                )
            finally:
                if support_override_backup is not None:
                    cfg.linear_osqp_params.clear()
                    cfg.linear_osqp_params.update(support_override_backup)

            if cfg.mpc_params['type'] == 'linear_osqp':
                controller = self.srbd_controller_interface.controller
                self.linear_solve_total_ms = float(getattr(controller, 'last_solve_total_ms', 0.0))
                self.linear_solve_setup_ms = float(getattr(controller, 'last_solve_setup_ms', 0.0))
                self.linear_solve_wall_ms = float(getattr(controller, 'last_solve_wall_ms', 0.0))
                self.linear_solve_iter = float(getattr(controller, 'last_solve_iter', 0))

            if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                # If the controller is gradient and is using RTI, we need to linearize the mpc after its computation
                # this helps to minize the delay between new state->control in a real case scenario.
                self.srbd_controller_interface.compute_RTI()

        # Update the gait
        if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['optimize_step_freq']:
            self.best_sample_freq = self.srbd_batched_controller_interface.optimize_gait(
                state_current,
                ref_state,
                inertia,
                self.wb_interface.pgg.phase_signal,
                self.wb_interface.pgg.step_freq,
                self.wb_interface.pgg.duty_factor,
                self.wb_interface.pgg.gait_type,
                optimize_swing,
            )

        # Compute Swing and Stance Torque ---------------------------------------------------------------------------
        tau, des_joints_pos, des_joints_vel = self.wb_interface.compute_stance_and_swing_torque(
            simulation_dt,
            qpos,
            qvel,
            feet_jac,
            feet_jac_dot,
            feet_pos,
            feet_vel,
            legs_qfrc_passive,
            legs_qfrc_bias,
            legs_mass_matrix,
            self.nmpc_GRFs,
            self.nmpc_footholds,
            legs_qpos_idx,
            legs_qvel_idx,
            tau,
            optimize_swing,
            self.best_sample_freq,
            self.nmpc_joints_pos,
            self.nmpc_joints_vel,
            self.nmpc_joints_acc,
            self.nmpc_predicted_state,
            mujoco_contact,
            foot_contact_state,
        )

        # Do some PD control over the joints (these values are normally passed
        # to a low-level motor controller, here we can try to simulate it)
        kp_joint_motor = cfg.simulation_params['impedence_joint_position_gain']
        kd_joint_motor = cfg.simulation_params['impedence_joint_velocity_gain']
        pd_scale = 0.0
        stance_pd_scale = 0.0
        latched_pd_scale = 0.0
        if cfg.mpc_params['type'] == 'linear_osqp':
            linear_params = getattr(cfg, 'linear_osqp_params', {})
            pd_scale = float(linear_params.get('joint_pd_scale', 0.0))
            stance_pd_scale = float(linear_params.get('stance_joint_pd_scale', 0.0))
            latched_pd_scale = float(linear_params.get('latched_joint_pd_scale', pd_scale))
        if pd_scale > 0.0 or stance_pd_scale > 0.0:
            for leg_id, leg in enumerate(legs_order):
                leg_pd_scale = stance_pd_scale
                if cfg.mpc_params['type'] == 'linear_osqp':
                    planned_contact = int(self.wb_interface.planned_contact[leg_id])
                    current_contact = int(self.wb_interface.current_contact[leg_id])
                    if planned_contact == 0 and current_contact == 0:
                        leg_pd_scale = pd_scale
                    elif planned_contact == 0 and current_contact == 1:
                        release_alpha = self.wb_interface.get_latched_release_alpha(leg_id)
                        leg_pd_scale = stance_pd_scale + release_alpha * (latched_pd_scale - stance_pd_scale)
                if leg_pd_scale <= 0.0:
                    continue
                tau[leg] += leg_pd_scale * (
                    kp_joint_motor * (des_joints_pos[leg] - qpos[legs_qpos_idx[leg]])
                    + kd_joint_motor * (des_joints_vel[leg] - qvel[legs_qvel_idx[leg]])
                )
        if cfg.mpc_params['type'] == 'linear_osqp':
            front_support_alpha = float(getattr(self.wb_interface, 'front_touchdown_support_alpha', 0.0))
            rear_support_alpha = float(getattr(self.wb_interface, 'rear_touchdown_support_alpha', 0.0))
            front_support_pd_scale = front_support_alpha * float(
                getattr(cfg, 'linear_osqp_params', {}).get('touchdown_support_front_joint_pd_scale', 0.0)
            )
            if front_support_pd_scale > 0.0:
                for leg_id, leg in enumerate(legs_order):
                    if leg_id >= 2:
                        continue
                    if int(self.wb_interface.current_contact[leg_id]) != 1:
                        continue
                    tau[leg] += front_support_pd_scale * (
                        kp_joint_motor * (des_joints_pos[leg] - qpos[legs_qpos_idx[leg]])
                        + kd_joint_motor * (des_joints_vel[leg] - qvel[legs_qvel_idx[leg]])
                    )
            rear_support_pd_scale = front_support_alpha * float(
                getattr(cfg, 'linear_osqp_params', {}).get('touchdown_support_rear_joint_pd_scale', 0.0)
            )
            if rear_support_pd_scale > 0.0:
                for leg_id, leg in enumerate(legs_order):
                    if leg_id < 2:
                        continue
                    if int(self.wb_interface.current_contact[leg_id]) != 1:
                        continue
                    tau[leg] += rear_support_pd_scale * (
                        kp_joint_motor * (des_joints_pos[leg] - qpos[legs_qpos_idx[leg]])
                        + kd_joint_motor * (des_joints_vel[leg] - qvel[legs_qvel_idx[leg]])
                    )
            front_support_pd_scale = rear_support_alpha * float(
                getattr(cfg, 'linear_osqp_params', {}).get('rear_touchdown_support_front_joint_pd_scale', 0.0)
            )
            if front_support_pd_scale > 0.0:
                for leg_id, leg in enumerate(legs_order):
                    if leg_id >= 2:
                        continue
                    if int(self.wb_interface.current_contact[leg_id]) != 1:
                        continue
                    tau[leg] += front_support_pd_scale * (
                        kp_joint_motor * (des_joints_pos[leg] - qpos[legs_qpos_idx[leg]])
                        + kd_joint_motor * (des_joints_vel[leg] - qvel[legs_qvel_idx[leg]])
                    )
            rear_support_pd_scale = rear_support_alpha * float(
                getattr(cfg, 'linear_osqp_params', {}).get('rear_touchdown_support_rear_joint_pd_scale', 0.0)
            )
            if rear_support_pd_scale > 0.0:
                for leg_id, leg in enumerate(legs_order):
                    if leg_id < 2:
                        continue
                    if int(self.wb_interface.current_contact[leg_id]) != 1:
                        continue
                    tau[leg] += rear_support_pd_scale * (
                        kp_joint_motor * (des_joints_pos[leg] - qpos[legs_qpos_idx[leg]])
                        + kd_joint_motor * (des_joints_vel[leg] - qvel[legs_qvel_idx[leg]])
                    )

        # Save some observables -------------------------------------------------------------------------------------
        self.quadrupedpympc_observables = {}
        for obs_name in self.quadrupedpympc_observables_names:
            if obs_name == 'ref_base_height':
                data = {'ref_base_height': ref_state['ref_position'][2]}
            elif obs_name == 'ref_base_angles':
                data = {'ref_base_angles': ref_state['ref_orientation']}
            elif obs_name == 'ref_feet_pos':
                ref_feet_pos = LegsAttr(
                    FL=ref_state['ref_foot_FL'].reshape(3, 1),
                    FR=ref_state['ref_foot_FR'].reshape(3, 1),
                    RL=ref_state['ref_foot_RL'].reshape(3, 1),
                    RR=ref_state['ref_foot_RR'].reshape(3, 1),
                )
                data = {'ref_feet_pos': ref_feet_pos}
            elif obs_name == 'des_foot_pos':
                data = {'des_foot_pos': self.wb_interface.last_des_foot_pos}
            elif obs_name == 'des_foot_vel':
                data = {'des_foot_vel': self.wb_interface.last_des_foot_vel}
            elif obs_name == 'ref_feet_constraints':
                ref_feet_constraints = LegsAttr(
                    FL=ref_state['ref_foot_FL_constraints'],
                    FR=ref_state['ref_foot_FR_constraints'],
                    RL=ref_state['ref_foot_RL_constraints'],
                    RR=ref_state['ref_foot_RR_constraints'],
                )
                data = {'ref_feet_constraints': ref_feet_constraints}
            elif obs_name == 'nmpc_GRFs':
                data = {'nmpc_GRFs': self.nmpc_GRFs}
            elif obs_name == 'nmpc_footholds':
                data = {'nmpc_footholds': self.nmpc_footholds}
            elif obs_name == 'swing_time':
                data = {'swing_time': self.wb_interface.stc.swing_time}
            elif obs_name == 'phase_signal':
                data = {'phase_signal': self.wb_interface.pgg._phase_signal}
            elif obs_name == 'lift_off_positions':
                data = {'lift_off_positions': self.wb_interface.frg.lift_off_positions}
            elif obs_name == 'planned_contact':
                data = {'planned_contact': np.asarray(self.wb_interface.planned_contact, dtype=float).copy()}
            elif obs_name == 'current_contact':
                data = {'current_contact': np.asarray(self.wb_interface.current_contact, dtype=float).copy()}
            elif obs_name == 'swing_contact_release_active':
                data = {'swing_contact_release_active': np.asarray(self.wb_interface.swing_contact_release_active, dtype=float).copy()}
            elif obs_name == 'latched_release_alpha':
                data = {
                    'latched_release_alpha': np.asarray(
                        [self.wb_interface.get_latched_release_alpha(leg_id) for leg_id in range(4)],
                        dtype=float,
                    )
                }
            elif obs_name == 'latched_swing_time':
                data = {'latched_swing_time': np.asarray(self.wb_interface.latched_swing_time, dtype=float).copy()}
            elif obs_name == 'support_margin':
                data = {'support_margin': np.asarray(self.wb_interface.last_support_margin, dtype=float).copy()}
            elif obs_name == 'support_confirm_active':
                data = {'support_confirm_active': np.asarray(self.wb_interface.support_confirm_active, dtype=float).copy()}
            elif obs_name == 'pre_swing_gate_active':
                data = {'pre_swing_gate_active': np.asarray(self.wb_interface.pre_swing_gate_active, dtype=float).copy()}
            elif obs_name == 'front_late_release_active':
                data = {'front_late_release_active': np.asarray(self.wb_interface.front_late_release_active, dtype=float).copy()}
            elif obs_name == 'touchdown_reacquire_active':
                data = {'touchdown_reacquire_active': np.asarray(self.wb_interface.touchdown_reacquire_active, dtype=float).copy()}
            elif obs_name == 'touchdown_confirm_active':
                data = {'touchdown_confirm_active': np.asarray(self.wb_interface.touchdown_confirm_active, dtype=float).copy()}
            elif obs_name == 'touchdown_settle_active':
                data = {'touchdown_settle_active': np.asarray(self.wb_interface.touchdown_settle_active, dtype=float).copy()}
            elif obs_name == 'touchdown_support_active':
                data = {'touchdown_support_active': np.asarray(self.wb_interface.touchdown_support_active, dtype=float).copy()}
            elif obs_name == 'rear_retry_contact_signal':
                data = {'rear_retry_contact_signal': np.asarray(self.wb_interface.rear_retry_contact_signal_debug, dtype=float).copy()}
            elif obs_name == 'rear_touchdown_contact_ready':
                data = {'rear_touchdown_contact_ready': np.asarray(self.wb_interface.rear_touchdown_contact_ready_debug, dtype=float).copy()}
            elif obs_name == 'rear_late_stance_contact_ready':
                data = {'rear_late_stance_contact_ready': np.asarray(self.wb_interface.rear_late_stance_contact_ready_debug, dtype=float).copy()}
            elif obs_name == 'rear_all_contact_support_needed':
                data = {'rear_all_contact_support_needed': np.asarray(self.wb_interface.rear_all_contact_support_needed_debug, dtype=float).copy()}
            elif obs_name == 'rear_late_seam_elapsed_s':
                data = {'rear_late_seam_elapsed_s': np.asarray(self.wb_interface.rear_late_seam_elapsed_s, dtype=float).copy()}
            elif obs_name == 'rear_late_seam_support_active':
                data = {'rear_late_seam_support_active': np.asarray(self.wb_interface.rear_late_seam_support_active_debug, dtype=float).copy()}
            elif obs_name == 'rear_close_handoff_active':
                data = {'rear_close_handoff_active': np.asarray(self.wb_interface.rear_close_handoff_active_debug, dtype=float).copy()}
            elif obs_name == 'rear_late_load_share_active':
                data = {
                    'rear_late_load_share_active': np.asarray(
                        self.wb_interface.rear_late_load_share_active_debug,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'rear_late_load_share_alpha':
                data = {
                    'rear_late_load_share_alpha': np.asarray(
                        self.wb_interface.rear_late_load_share_alpha_debug,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'rear_late_load_share_candidate_active':
                data = {
                    'rear_late_load_share_candidate_active': np.asarray(
                        self.wb_interface.rear_late_load_share_candidate_active_debug,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'rear_late_load_share_candidate_alpha':
                data = {
                    'rear_late_load_share_candidate_alpha': np.asarray(
                        self.wb_interface.rear_late_load_share_candidate_alpha_debug,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'rear_late_load_share_trigger_elapsed_s':
                data = {
                    'rear_late_load_share_trigger_elapsed_s': np.asarray(
                        self.wb_interface.rear_late_load_share_trigger_elapsed_s,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'rear_late_load_share_trigger_enabled':
                data = {
                    'rear_late_load_share_trigger_enabled': float(
                        getattr(self.wb_interface, 'rear_late_load_share_trigger_enabled_debug', 0.0)
                    )
                }
            elif obs_name == 'rear_close_handoff_alpha':
                data = {
                    'rear_close_handoff_alpha': float(
                        getattr(self.wb_interface, 'rear_close_handoff_alpha', 0.0)
                    )
                }
            elif obs_name == 'rear_close_handoff_leg_index':
                data = {
                    'rear_close_handoff_leg_index': float(
                        getattr(self.wb_interface, 'rear_close_handoff_leg_index', -1)
                    )
                }
            elif obs_name == 'rear_all_contact_weak_leg_alpha':
                data = {
                    'rear_all_contact_weak_leg_alpha': float(
                        getattr(self.wb_interface, 'rear_all_contact_weak_leg_alpha', 0.0)
                    )
                }
            elif obs_name == 'rear_all_contact_weak_leg_index':
                data = {
                    'rear_all_contact_weak_leg_index': float(
                        getattr(self.wb_interface, 'rear_all_contact_weak_leg_index', -1)
                    )
                }
            elif obs_name == 'applied_linear_support_force_floor_ratio':
                data = {
                    'applied_linear_support_force_floor_ratio': float(
                        getattr(self, 'applied_linear_support_force_floor_ratio', 0.0)
                    )
                }
            elif obs_name == 'applied_linear_rear_handoff_leg_index':
                data = {
                    'applied_linear_rear_handoff_leg_index': float(
                        getattr(self, 'applied_linear_rear_handoff_leg_index', -1)
                    )
                }
            elif obs_name == 'applied_linear_rear_handoff_leg_floor_scale':
                data = {
                    'applied_linear_rear_handoff_leg_floor_scale': float(
                        getattr(self, 'applied_linear_rear_handoff_leg_floor_scale', 0.0)
                    )
                }
            elif obs_name == 'applied_linear_latched_force_scale':
                data = {
                    'applied_linear_latched_force_scale': float(
                        getattr(self, 'applied_linear_latched_force_scale', 0.0)
                    )
                }
            elif obs_name == 'applied_linear_latched_front_receiver_scale':
                data = {
                    'applied_linear_latched_front_receiver_scale': float(
                        getattr(self, 'applied_linear_latched_front_receiver_scale', 0.0)
                    )
                }
            elif obs_name == 'applied_linear_latched_rear_receiver_scale':
                data = {
                    'applied_linear_latched_rear_receiver_scale': float(
                        getattr(self, 'applied_linear_latched_rear_receiver_scale', 0.0)
                    )
                }
            elif obs_name == 'rear_touchdown_actual_contact_elapsed_s':
                data = {'rear_touchdown_actual_contact_elapsed_s': np.asarray(self.wb_interface.rear_touchdown_actual_contact_elapsed_s, dtype=float).copy()}
            elif obs_name == 'rear_touchdown_pending_confirm':
                data = {'rear_touchdown_pending_confirm': np.asarray(self.wb_interface.rear_touchdown_pending_confirm, dtype=float).copy()}
            elif obs_name == 'front_margin_rescue_active':
                data = {'front_margin_rescue_active': np.asarray(self.wb_interface.front_margin_rescue_active, dtype=float).copy()}
            elif obs_name == 'front_margin_rescue_alpha':
                data = {'front_margin_rescue_alpha': np.asarray(self.wb_interface.front_margin_rescue_alpha, dtype=float).copy()}
            elif obs_name == 'touchdown_support_alpha':
                data = {'touchdown_support_alpha': float(self.wb_interface.touchdown_support_alpha)}
            elif obs_name == 'front_touchdown_support_alpha':
                data = {'front_touchdown_support_alpha': float(self.wb_interface.front_touchdown_support_alpha)}
            elif obs_name == 'rear_touchdown_support_alpha':
                data = {'rear_touchdown_support_alpha': float(self.wb_interface.rear_touchdown_support_alpha)}
            elif obs_name == 'rear_all_contact_stabilization_alpha':
                data = {
                    'rear_all_contact_stabilization_alpha': float(
                        getattr(self.wb_interface, 'rear_all_contact_stabilization_alpha', 0.0)
                    )
                }
            elif obs_name == 'rear_all_contact_front_planted_tail_alpha':
                data = {
                    'rear_all_contact_front_planted_tail_alpha': float(
                        getattr(self.wb_interface, 'rear_all_contact_front_planted_tail_alpha', 0.0)
                    )
                }
            elif obs_name == 'crawl_front_planted_seam_support_alpha':
                data = {
                    'crawl_front_planted_seam_support_alpha': float(
                        self.wb_interface.crawl_state.front_planted_seam_support_alpha
                    )
                }
            elif obs_name == 'rear_handoff_support_active':
                data = {'rear_handoff_support_active': float(getattr(self.wb_interface, 'rear_handoff_support_active', 0.0))}
            elif obs_name == 'rear_swing_bridge_active':
                data = {'rear_swing_bridge_active': float(getattr(self.wb_interface, 'rear_swing_bridge_active', 0.0))}
            elif obs_name == 'rear_swing_release_support_active':
                data = {'rear_swing_release_support_active': float(getattr(self.wb_interface, 'rear_swing_release_support_active', 0.0))}
            elif obs_name == 'full_contact_recovery_active':
                data = {'full_contact_recovery_active': float(self.wb_interface.full_contact_recovery_active)}
            elif obs_name == 'full_contact_recovery_alpha':
                data = {'full_contact_recovery_alpha': float(self.wb_interface.full_contact_recovery_alpha)}
            elif obs_name == 'full_contact_recovery_remaining_s':
                data = {'full_contact_recovery_remaining_s': float(self.wb_interface.full_contact_recovery_remaining_s)}
            elif obs_name == 'full_contact_recovery_trigger':
                data = {
                    'full_contact_recovery_trigger': float(
                        getattr(self.wb_interface, 'full_contact_recovery_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'front_delayed_swing_recovery_trigger':
                data = {
                    'front_delayed_swing_recovery_trigger': float(
                        getattr(self.wb_interface, 'front_delayed_swing_recovery_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'planted_front_recovery_trigger':
                data = {
                    'planted_front_recovery_trigger': float(
                        getattr(self.wb_interface, 'planted_front_recovery_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'planted_front_postdrop_recovery_trigger':
                data = {
                    'planted_front_postdrop_recovery_trigger': float(
                        getattr(self.wb_interface, 'planted_front_postdrop_recovery_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'front_close_gap_trigger':
                data = {
                    'front_close_gap_trigger': float(
                        getattr(self.wb_interface, 'front_close_gap_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'front_late_rearm_trigger':
                data = {
                    'front_late_rearm_trigger': float(
                        getattr(self.wb_interface, 'front_late_rearm_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'front_planted_posture_tail_trigger':
                data = {
                    'front_planted_posture_tail_trigger': float(
                        getattr(self.wb_interface, 'front_planted_posture_tail_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'front_late_posture_tail_trigger':
                data = {
                    'front_late_posture_tail_trigger': float(
                        getattr(self.wb_interface, 'front_late_posture_tail_trigger_debug', 0.0)
                    )
                }
            elif obs_name == 'crawl_front_stance_support_tail_remaining_s':
                data = {
                    'crawl_front_stance_support_tail_remaining_s': float(
                        self.wb_interface.crawl_state.front_stance_support_tail_remaining_s
                    )
                }
            elif obs_name == 'front_touchdown_support_recent_remaining_s':
                data = {
                    'front_touchdown_support_recent_remaining_s': float(
                        self.wb_interface.front_touchdown_support_recent_remaining_s
                    )
                }
            elif obs_name == 'front_delayed_swing_recovery_spent':
                data = {
                    'front_delayed_swing_recovery_spent': np.asarray(
                        self.wb_interface.front_delayed_swing_recovery_spent,
                        dtype=float,
                    ).copy()
                }
            elif obs_name == 'gate_forward_scale':
                data = {'gate_forward_scale': float(self.wb_interface.last_gate_forward_scale)}
            elif obs_name == 'linear_solve_total_ms':
                data = {'linear_solve_total_ms': float(getattr(self, 'linear_solve_total_ms', 0.0))}
            elif obs_name == 'linear_solve_setup_ms':
                data = {'linear_solve_setup_ms': float(getattr(self, 'linear_solve_setup_ms', 0.0))}
            elif obs_name == 'linear_solve_wall_ms':
                data = {'linear_solve_wall_ms': float(getattr(self, 'linear_solve_wall_ms', 0.0))}
            elif obs_name == 'linear_solve_iter':
                data = {'linear_solve_iter': float(getattr(self, 'linear_solve_iter', 0.0))}

            else:
                data = {}
                raise ValueError(f"Unknown observable name: {obs_name}")

            self.quadrupedpympc_observables.update(data)

        return tau

    def get_obs(self) -> dict:
        """Get some user-defined observables from withing the control loop.

        Returns:
            Dict: dictionary of observables
        """
        return self.quadrupedpympc_observables

    def reset(self, initial_feet_pos: LegsAttr):
        """Reset the controller."""

        self.wb_interface.reset(initial_feet_pos)
        self.srbd_controller_interface.controller.reset()
