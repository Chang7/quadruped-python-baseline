"""This file includes all of the configuration parameters for the MPC controllers
and of the internal simulations that can be launch from the folder /simulation.
"""
import numpy as np
from quadruped_pympc.helpers.quadruped_utils import GaitType

# These are used both for a real experiment and a simulation -----------
# These are the only attributes needed per quadruped, the rest can be computed automatically ----------------------
robot = 'aliengo'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah', 'spot'  # TODO: Load from robot_descriptions.py

from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config
robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
robot_leg_joints = robot_cfg.leg_joints
robot_feet_geom_names = robot_cfg.feet_geom_names
qpos0_js = robot_cfg.qpos0_js
hip_height = robot_cfg.hip_height

# ----------------------------------------------------------------------------------------------------------------
if (robot == 'go1'):
    mass = 12.019
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'go2'):
    mass = 15.019
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'aliengo'):
    mass = 24.637
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])

elif (robot == 'b2'):
    mass = 83.49
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])


elif (robot == 'hyqreal1'):
    mass = 108.40
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'hyqreal2'):
    mass = 126.69
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'mini_cheetah'):
    mass = 12.5
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'spot'):
    mass = 50.34
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])


gravity_constant = 9.81 # Exposed in case of different gravity conditions
# ----------------------------------------------------------------------------------------------------------------

mpc_params = {
    # 'nominal' optimized directly the GRF
    # 'input_rates' optimizes the delta GRF
    # 'sampling' is a gpu-based mpc that samples the GRF
    # 'collaborative' optimized directly the GRF and has a passive arm model inside
    # 'lyapunov' optimized directly the GRF and has a Lyapunov-based stability constraint
    # 'kinodynamic' sbrd with joints - experimental
    # 'linear_osqp' uses an external compact linear SRB force MPC with the same PyMPC low-level stack
    'type':                                    'nominal',

    # print the mpc info
    'verbose':                                 False,

    # horizon is the number of timesteps in the future that the mpc will optimize
    # dt is the discretization time used in the mpc
    'horizon':                                 12,
    'dt':                                      0.02,

    # GRF limits for each single leg
    "grf_max":                                 mass * gravity_constant,
    "grf_min":                                 0,
    'mu':                                      0.5,

    # this is used to have a smaller dt near the start of the horizon
    'use_nonuniform_discretization':           False,
    'horizon_fine_grained':                    2,
    'dt_fine_grained':                         0.01,

    # if this is true, we optimize the step frequency as well
    # for the sampling controller, this is done in the rollout
    # for the gradient-based controller, this is done with a batched version of the ocp
    'optimize_step_freq':                      False,
    'step_freq_available':                     [1.4, 2.0, 2.4],

    # ----- START properties only for the gradient-based mpc -----

    # this is used if you want to manually warm start the mpc
    'use_warm_start':                          False,

    # this enables integrators for height, linear velocities, roll and pitch
    'use_integrators':                         False,
    'alpha_integrator':                        0.1,
    'integrator_cap':                          [0.5, 0.2, 0.2, 0.0, 0.0, 1.0],

    # if this is off, the mpc will not optimize the footholds and will
    # use only the ones provided in the reference
    'use_foothold_optimization':               True,

    # this is set to false automatically is use_foothold_optimization is false
    # because in that case we cannot chose the footholds and foothold
    # constraints do not any make sense
    'use_foothold_constraints':                False,

    # works with all the mpc types except 'sampling'. In sim does not do much for now,
    # but in real it minizimes the delay between the mpc control and the state
    'use_RTI':                                 False,
    # If RTI is used, we can set the advance RTI-step! (Standard is the simpler RTI)
    # See https://arxiv.org/pdf/2403.07101.pdf
    'as_rti_type':                             "Standard",  # "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D", "Standard"
    'as_rti_iter':                             1,  # > 0, the higher the better, but slower computation!

    # This will force to use DDP instead of SQP, based on https://arxiv.org/abs/2403.10115.
    # Note that RTI is not compatible with DDP, and no state costraints for now are considered
    'use_DDP':                                 False,

    # this is used only in the case 'use_RTI' is false in a single mpc feedback loop.
    # More is better, but slower computation!
    'num_qp_iterations':                       1,

    # this is used to speeding up or robustify acados' solver (hpipm).
    'solver_mode':                             'balance',  # balance, robust, speed, crazy_speed


    # these is used only for the case 'input_rates', using as GRF not the actual state
    # of the robot of the predicted one. Can be activated to compensate
    # for the delay in the control loop on the real robot
    'use_input_prediction':                    False,

    # ONLY ONE CAN BE TRUE AT A TIME (only gradient)
    'use_static_stability':                    False,
    'use_zmp_stability':                       False,
    'trot_stability_margin':                   0.04,
    'pace_stability_margin':                   0.1,
    'crawl_stability_margin':                  0.04,  # in general, 0.02 is a good value

    # this is used to compensate for the external wrenches
    # you should provide explicitly this value in compute_control
    'external_wrenches_compensation':          True,
    'external_wrenches_compensation_num_step': 15,

    # this is used only in the case of collaborative mpc, to
    # compensate for the external wrench in the prediction (only collaborative)
    'passive_arm_compensation':                True,


    # Gain for Lyapunov-based MPC
    'K_z1': np.array([1, 1, 10]),
    'K_z2': np.array([1, 4, 10]),
    'residual_dynamics_upper_bound': 30,
    'use_residual_dynamics_decay': False,

    # ----- END properties for the gradient-based mpc -----


    # ----- START properties only for the sampling-based mpc -----

    # this is used only in the case 'sampling'.
    'sampling_method':                         'random_sampling',  # 'random_sampling', 'mppi', 'cem_mppi'
    'control_parametrization':                 'cubic_spline', # 'cubic_spline', 'linear_spline', 'zero_order'
    'num_splines':                             2,  # number of splines to use for the control parametrization
    'num_parallel_computations':               10000,  # More is better, but slower computation!
    'num_sampling_iterations':                 1,  # More is better, but slower computation!
    'device':                                  'gpu',  # 'gpu', 'cpu'
    # convariances for the sampling methods
    'sigma_cem_mppi':                          3,
    'sigma_mppi':                              3,
    'sigma_random_sampling':                   [0.2, 3, 10],
    'shift_solution':                          False,

    # ----- END properties for the sampling-based mpc -----
    }
# -----------------------------------------------------------------------

simulation_params = {
    'swing_generator':             'scipy',  # 'scipy', 'explicit'
    'swing_position_gain_fb':      500,
    'swing_velocity_gain_fb':      10,
    'impedence_joint_position_gain':  10.0,
    'impedence_joint_velocity_gain':  2.0,

    'step_height':                 0.2 * hip_height,  

    # Visual Foothold adapatation
    "visual_foothold_adaptation":  'blind', #'blind', 'height', 'vfa'

    # this is the integration time used in the simulator
    'dt':                          0.002,

    'gait':                        'trot',  # 'trot', 'pace', 'crawl', 'bound', 'full_stance'
    'gait_params':                 {'trot': {'step_freq': 1.4, 'duty_factor': 0.65, 'type': GaitType.TROT.value},
                                    'crawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.BACKDIAGONALCRAWL.value},
                                    'pace': {'step_freq': 1.4, 'duty_factor': 0.7, 'type': GaitType.PACE.value},
                                    'bound': {'step_freq': 1.8, 'duty_factor': 0.65, 'type': GaitType.BOUNDING.value},
                                    'full_stance': {'step_freq': 2, 'duty_factor': 0.65, 'type': GaitType.FULL_STANCE.value},
                                   },

    # This is used to activate or deactivate the reflexes upon contact detection
    'reflex_trigger_mode':       'tracking', # 'tracking', 'geom_contact', False
    'reflex_max_step_height':    0.5 * hip_height,  # this is the maximum step height that the robot can do if reflexes are enabled
    'reflex_next_steps_height_enhancement': False,
    'velocity_modulator': True,

    # velocity mode: human will give you the possibility to use the keyboard, the other are
    # forward only random linear-velocity, random will give you random linear-velocity and yaw-velocity
    'mode':                        'human',  # 'human', 'forward', 'random'
    'ref_z':                       hip_height,


    # the MPC will be called every 1/(mpc_frequency*dt) timesteps
    # this helps to evaluate more realistically the performance of the controller
    'mpc_frequency':               100,

    'use_inertia_recomputation':   True,

    'scene':                       'flat',  # flat, random_boxes, random_pyramids, perlin

    }
# -----------------------------------------------------------------------

# Linear-OSQP adapter parameters ---------------------------------------------------------
linear_osqp_params = {
    # Conservative defaults intended for first stable integration inside the PyMPC low-level stack.
    'Q_p': 2e4,
    'Q_v': 4e4,
    'Q_theta': 2e4,
    'Q_theta_roll': None,  # optional roll-axis override for the orientation-state weight
    'Q_theta_pitch': None,  # optional pitch-axis override for the orientation-state weight
    'Q_w': 2e3,
    'Q_w_roll': None,  # optional roll-axis override for the angular-rate-state weight
    'Q_w_pitch': None,  # optional pitch-axis override for the angular-rate-state weight
    'R_u': 5.0,
    # Post-solve conditioning to reduce stance transition shocks and lateral drift.
    'command_smoothing': 0.35,   # 0 -> no smoothing, 1 -> keep previous command
    'du_xy_max': 2.5,            # max per-step change in fx/fy [N]
    'du_z_max': 3.5,             # max per-step change in fz [N]
    'stance_ramp_steps': 6,      # ramp-up steps after a leg enters stance
    'fy_scale': 0.15,            # suppress lateral force during crawl-like debugging
    'dynamic_fy_roll_gain': 0.0,  # optional extra lateral-force authority that ramps in with absolute roll
    'dynamic_fy_roll_ref': 0.20,  # roll angle [rad] that saturates the temporary lateral-force boost
    'grf_max_scale': 0.35,       # effective fz upper bound as fraction of body weight per leg total budget
    'support_force_floor_ratio': 0.0,  # minimum share of body weight kept on each stance leg
    'joint_pd_scale': 0.25,      # blend-in low-level joint PD on swing legs to help realize foothold geometry in torque control
    'stance_joint_pd_scale': 0.0,  # optional reduced joint PD kept on stance legs while swing legs use the higher swing blend
    'latched_joint_pd_scale': 0.25,  # optional reduced joint PD for planned-swing legs kept relatched in contact
    'latched_release_phase_start': 0.0,  # optional swing-progress window start for relaxing relatched planned-swing legs
    'latched_release_phase_end': 1.0,  # optional swing-progress window end for relaxing relatched planned-swing legs
    'stance_target_blend': 0.0,  # blend stance-foot xy target toward the next foothold while keeping touchdown hold
    'latched_force_scale': 1.0,  # scale commanded force on planned-swing legs that remain relatched in contact
    'latched_floor_scale': 1.0,  # scale only the minimum vertical-force floor on relatched planned-swing legs
    'latched_same_side_receiver_scale': 1.0,  # redistribute released load away from same-side support legs when desired
    'latched_axle_receiver_scale': 1.0,  # bias released load toward the same front/rear pair of the relatched leg
    'latched_diagonal_receiver_scale': 1.0,  # bias released load toward the diagonal support leg
    'latched_front_receiver_scale': 1.0,  # extra weight applied to front support legs when redistributing released load
    'latched_rear_receiver_scale': 1.0,  # extra weight applied to rear support legs when redistributing released load
    'front_latched_pitch_relief_gain': 0.0,  # extra vertical-force relief applied only to front planned-swing legs that remain latched while pitching forward
    'front_latched_rear_bias_gain': 0.0,  # extra rearward redistribution applied only when a front planned-swing leg remains latched
    'rear_floor_base_scale': 0.65,  # baseline share of total vertical load that rear support legs should retain
    'rear_floor_pitch_gain': 0.20,  # additional rear-load bias as pitch error grows
    'side_rebalance_gain': 0.0,  # extra signed left/right load floor bias driven by roll sign to keep the falling side from collapsing
    'side_rebalance_ref': 0.35,  # roll angle [rad] that saturates the signed left/right load rebalance
    'pitch_rebalance_gain': 0.0,  # post-solve extra rear-load transfer when the body pitches forward
    'pitch_rebalance_ref': 0.35,  # pitch angle [rad] that saturates the post-solve rear-load transfer
    'support_centroid_x_gain': 0.0,  # move CoM toward the front/rear center of the current support polygon when swing is active
    'support_centroid_y_gain': 0.0,  # move CoM toward the left/right center of the current support polygon when swing is active
    'foothold_yaw_rate_scale': 0.0,  # Raibert-style yaw-rate foothold compensation scale used by the linear path
    'foothold_yaw_error_scale': 0.0,  # additional yaw-rate tracking-error foothold compensation scale
    'pre_swing_lookahead_steps': 3,  # inspect a few upcoming MPC stages so support-centroid shift can start before the first foot lifts
    'pre_swing_front_shift_scale': 1.0,  # amplify pre-swing support-centroid shift when the upcoming swing leg is a front leg
    'pre_swing_rear_shift_scale': 1.0,  # amplify pre-swing support-centroid shift when the upcoming swing leg is a rear leg
    'support_reference_xy_mix': None,  # optional x/y-only blend factor for the solved support-wrench reference; defaults to support_reference_mix
    'pre_swing_gate_min_margin': 0.015,  # keep a scheduled swing leg grounded until the upcoming 3-leg support polygon has this much margin
    'front_pre_swing_gate_min_margin': None,  # optional front-leg override for the required support margin before lift-off
    'rear_pre_swing_gate_min_margin': None,  # optional rear-leg override for the required support margin before lift-off
    'rear_pre_swing_gate_hold_s': None,  # optional rear-leg override for how long the pre-swing gate may keep lift-off delayed
    'rear_pre_swing_guard_roll_threshold': None,  # optional rear-leg posture guard: delay rear release when |roll| is above this during recovery
    'rear_pre_swing_guard_pitch_threshold': None,  # optional rear-leg posture guard: delay rear release when |pitch| is above this during recovery
    'rear_pre_swing_guard_height_ratio': 0.0,  # optional rear-leg posture guard: delay rear release when base height / ref height is this low
    'support_contact_confirm_hold_s': 0.0,  # require support contacts to stay ready this long before allowing a scheduled swing
    'front_support_contact_confirm_hold_s': None,  # optional front-leg override for support-contact confirmation hold
    'rear_support_contact_confirm_hold_s': None,  # optional rear-leg override for support-contact confirmation hold
    'support_confirm_min_contacts': 3,  # minimum supporting contacts that must stay ready before opening swing
    'support_confirm_require_front_rear_span': True,  # require both a front and rear support contact when both are scheduled
    'support_confirm_forward_scale': 1.0,  # scale forward reference velocity while support confirmation is still pending
    'front_stance_dropout_reacquire': False,  # reuse touchdown reacquire/confirm when a front stance foot unexpectedly loses actual contact
    'front_stance_dropout_support_hold_s': 0.0,  # keep front touchdown-style support alive briefly when a front stance foot drops actual contact
    'front_stance_dropout_support_forward_scale': 1.0,  # optional forward-speed scale during the front stance-dropout bridge window
    'rear_stance_dropout_reacquire': False,  # reuse touchdown reacquire/confirm when a rear stance foot unexpectedly loses actual contact
    'front_late_release_phase_threshold': 1.1,  # >1 disables; otherwise allow front planned-swing legs to open late once support is safe
    'front_late_release_min_margin': None,  # optional front-leg support margin required before the late-release path opens
    'front_late_release_hold_steps': 0,  # legacy controller-step hold after front late release opens
    'front_late_release_hold_s': None,  # controller-side swing hold in seconds after front late release opens
    'front_late_release_forward_scale': 1.0,  # scale forward reference velocity while the front-only late-release path is active
    'support_margin_preview_s': 0.0,  # preview CoM xy motion by this many seconds when checking support-margin safety gates
    'swing_contact_release_timeout_s': 0.0,  # if a planned swing leg remains physically stuck in contact, stop re-latching it forever after this timeout
    'front_swing_contact_release_timeout_s': None,  # optional front-leg override for planned-swing contact release timeout
    'rear_swing_contact_release_timeout_s': None,  # optional rear-leg override for planned-swing contact release timeout
    'front_release_lift_height': 0.0,  # temporary extra upward swing target [m] applied when a front swing leg remains physically stuck in contact
    'front_release_lift_kp': 0.0,  # task-space z gain used by the front forced-release lift assist
    'front_release_lift_kd': 0.0,  # task-space z damping used by the front forced-release lift assist
    'rear_release_lift_height': 0.0,  # temporary extra upward swing target [m] applied when a rear swing leg remains physically stuck in contact
    'rear_release_lift_kp': 0.0,  # task-space z gain used by the rear forced-release lift assist
    'rear_release_lift_kd': 0.0,  # task-space z damping used by the rear forced-release lift assist
    'front_late_release_extra_margin': 0.0,  # extra safety margin added on top of the front late-release minimum support margin
    'front_late_release_pitch_guard': None,  # optional abs pitch limit [rad] for opening the front late-release path
    'front_late_release_roll_guard': None,  # optional abs roll limit [rad] for opening the front late-release path
    'touchdown_reacquire_hold_s': 0.0,  # keep controller-side swing briefly after planned touchdown if the foot still has no actual contact
    'front_touchdown_reacquire_hold_s': None,  # optional front-leg override for post-swing touchdown reacquire hold
    'rear_touchdown_reacquire_hold_s': None,  # optional rear-leg override for post-swing touchdown reacquire hold
    'touchdown_reacquire_forward_scale': 1.0,  # scale forward reference velocity while waiting for actual touchdown reacquire
    'touchdown_reacquire_xy_blend': 0.0,  # blend touchdown reacquire target xy toward the stored touchdown position
    'front_touchdown_reacquire_xy_blend': None,  # optional front-leg override for touchdown reacquire xy blend
    'rear_touchdown_reacquire_xy_blend': None,  # optional rear-leg override for touchdown reacquire xy blend
    'touchdown_reacquire_extra_depth': 0.0,  # lower the touchdown reacquire target below the nominal touchdown height
    'front_touchdown_reacquire_extra_depth': None,  # optional front-leg override for touchdown reacquire extra depth
    'rear_touchdown_reacquire_extra_depth': None,  # optional rear-leg override for touchdown reacquire extra depth
    'touchdown_reacquire_forward_bias': 0.0,  # forward x bias added to the touchdown reacquire target
    'front_touchdown_reacquire_forward_bias': None,  # optional front-leg override for touchdown reacquire forward bias
    'rear_touchdown_reacquire_forward_bias': None,  # optional rear-leg override for touchdown reacquire forward bias
    'rear_touchdown_reacquire_force_until_contact': False,  # keep rear controller-side swing active during planned stance until actual contact truly returns
    'rear_touchdown_reacquire_min_swing_time_s': 0.0,  # optional minimum rear swing time [s] before planned-stance reacquire support is allowed to take over
    'rear_touchdown_reacquire_hold_current_xy': False,  # keep rear touchdown reacquire mostly vertical by holding the current foot xy instead of chasing the nominal foothold
    'rear_touchdown_reacquire_max_xy_shift': 0.0,  # optional cap [m] on how far a rear reacquire target may move away from the current foot xy
    'rear_touchdown_reacquire_min_phase': 0.0,  # optional minimum swing phase [0, 1] enforced during rear touchdown reacquire to bias the foot into late descent
    'rear_touchdown_reacquire_upward_vel_damping': 0.0,  # extra task-space damping [N*s/m] applied only to positive rear foot z velocity while waiting for actual recontact
    'rear_touchdown_retry_descent_depth': 0.0,  # extra downward search depth [m] applied only after a rear false-close retry opens swing again
    'rear_touchdown_retry_descent_kp': 0.0,  # vertical task-space stiffness [N/m] used during rear false-close retry descent assist
    'rear_touchdown_retry_descent_kd': 0.0,  # vertical task-space damping [N*s/m] used during rear false-close retry descent assist
    'rear_touchdown_contact_debounce_s': 0.0,  # require rear actual contact to persist this long before closing controller-side swing during force-until-contact
    'rear_touchdown_contact_min_phase': 0.0,  # require rear swing phase to reach at least this value before controller-side contact may close on touchdown
    'rear_touchdown_contact_max_upward_vel': None,  # optional max allowed upward rear foot z velocity [m/s] before controller-side contact may close on touchdown
    'rear_touchdown_contact_min_grf_z': 0.0,  # require at least this upward world-frame GRF [N] before rear touchdown is treated as truly load-bearing
    'rear_touchdown_reacquire_retire_stance_hold_s': 0.0,  # retire stale rear reacquire state once planned/current/actual stance has stayed stable this long [s]
    'rear_crawl_disable_reflex_swing': False,  # ignore early-stance reflex swing shaping on rear legs during crawl
    'front_crawl_swing_height_scale': 1.0,  # scale front-leg crawl swing vertical excursion relative to the nominal trajectory
    'rear_crawl_swing_height_scale': 1.0,  # scale rear-leg crawl swing vertical excursion relative to the nominal trajectory
    'stance_anchor_update_alpha': 0.0,  # continuously relax stance touchdown anchors toward actual contacted foot position
    'front_stance_anchor_update_alpha': None,  # optional front-leg override for stance anchor update alpha
    'rear_stance_anchor_update_alpha': None,  # optional rear-leg override for stance anchor update alpha
    'touchdown_support_anchor_update_alpha': 0.0,  # relax stance touchdown anchors toward actual foot position only while touchdown support is active
    'front_touchdown_support_anchor_update_alpha': None,  # optional front-leg override for touchdown-support anchor update alpha
    'rear_touchdown_support_anchor_update_alpha': None,  # optional rear-leg override for touchdown-support anchor update alpha
    'touchdown_confirm_hold_s': 0.0,  # keep a short confirmation window after actual touchdown returns
    'front_touchdown_confirm_hold_s': None,  # optional front-leg override for touchdown confirmation window
    'rear_touchdown_confirm_hold_s': None,  # optional rear-leg override for touchdown confirmation window
    'rear_touchdown_confirm_keep_swing': False,  # keep rear controller-side swing active during touchdown confirm so swing phase is not reset by a flaky first contact
    'touchdown_confirm_forward_scale': 1.0,  # scale forward reference velocity during the touchdown confirmation window
    'touchdown_settle_hold_s': 0.0,  # keep a short post-touchdown settling window after actual contact comes back
    'front_touchdown_settle_hold_s': None,  # optional front-leg override for the post-touchdown settling window
    'rear_touchdown_settle_hold_s': None,  # optional rear-leg override for the post-touchdown settling window
    'touchdown_settle_forward_scale': 1.0,  # scale forward reference velocity during the post-touchdown settling window
    'touchdown_support_rear_floor_delta': 0.0,  # temporary extra rear-load floor applied while front touchdown confirm/settle is active
    'touchdown_support_vertical_boost': 0.0,  # temporary extra body-weight-scaled lift request during front touchdown support
    'touchdown_support_min_vertical_force_scale_delta': 0.0,  # raise minimum vertical-force intent during front touchdown support
    'touchdown_support_grf_max_scale_delta': 0.0,  # temporarily raise GRF headroom during front touchdown support
    'touchdown_support_z_pos_gain_delta': 0.0,  # temporary extra height gain during front touchdown support
    'touchdown_support_roll_angle_gain_delta': 0.0,  # temporary extra roll-angle gain during front touchdown support
    'touchdown_support_roll_rate_gain_delta': 0.0,  # temporary extra roll-rate gain during front touchdown support
    'touchdown_support_pitch_angle_gain_delta': 0.0,  # temporary extra pitch-angle gain during front touchdown support
    'touchdown_support_pitch_rate_gain_delta': 0.0,  # temporary extra pitch-rate gain during front touchdown support
    'touchdown_support_side_rebalance_delta': 0.0,  # temporary extra signed left/right load rebalance during front touchdown support
    'touchdown_support_front_joint_pd_scale': 0.0,  # extra low-gain joint PD applied only on front support legs during front touchdown support
    'touchdown_support_rear_joint_pd_scale': 0.0,  # extra low-gain joint PD applied only on rear support legs during front touchdown support
    'touchdown_support_anchor_xy_blend': 0.0,  # blend a front touchdown support leg's stance target toward the actual contacted foot xy location
    'touchdown_support_anchor_z_blend': 0.0,  # blend a front touchdown support leg's stance target height toward the actual contacted foot z location
    'rear_touchdown_support_support_floor_delta': 0.0,  # temporary extra all-stance support floor applied while rear touchdown support is active
    'rear_touchdown_support_vertical_boost': 0.0,  # temporary extra body-weight-scaled lift request during rear touchdown support
    'rear_touchdown_support_min_vertical_force_scale_delta': 0.0,  # raise minimum vertical-force intent during rear touchdown support
    'rear_touchdown_support_grf_max_scale_delta': 0.0,  # temporarily raise GRF headroom during rear touchdown support
    'rear_touchdown_support_z_pos_gain_delta': 0.0,  # temporary extra height gain during rear touchdown support
    'rear_touchdown_support_roll_angle_gain_delta': 0.0,  # temporary extra roll-angle gain during rear touchdown support
    'rear_touchdown_support_roll_rate_gain_delta': 0.0,  # temporary extra roll-rate gain during rear touchdown support
    'rear_touchdown_support_pitch_angle_gain_delta': 0.0,  # temporary extra pitch-angle gain during rear touchdown support
    'rear_touchdown_support_pitch_rate_gain_delta': 0.0,  # temporary extra pitch-rate gain during rear touchdown support
    'rear_touchdown_support_side_rebalance_delta': 0.0,  # temporary extra signed left/right load rebalance during rear touchdown support
    'rear_touchdown_support_front_joint_pd_scale': 0.0,  # extra low-gain joint PD applied only on front support legs during rear touchdown support
    'rear_touchdown_support_rear_joint_pd_scale': 0.0,  # extra low-gain joint PD applied only on rear support legs during rear touchdown support
    'rear_post_touchdown_support_hold_s': 0.0,  # keep a short rear-specific support tail alive after a rear touchdown/recontact seam
    'rear_post_touchdown_support_forward_scale': 1.0,  # scale forward reference velocity while the rear post-touchdown support tail is active
    'rear_post_touchdown_support_height_ratio': 0.0,  # optionally keep rear post-touchdown support alive while base height stays this low
    'rear_post_touchdown_support_roll_threshold': None,  # optional abs roll threshold [rad] for keeping the rear post-touchdown support tail alive
    'rear_post_touchdown_support_pitch_threshold': None,  # optional abs pitch threshold [rad] for keeping the rear post-touchdown support tail alive
    'rear_post_touchdown_support_min_grf_z': 0.0,  # keep rear post-touchdown support alive while the touched rear leg carries less than this vertical GRF [N]
    'rear_post_touchdown_support_min_rear_load_share': 0.0,  # keep rear post-touchdown support alive while total rear vertical load share stays below this fraction
    'rear_all_contact_stabilization_hold_s': 0.0,  # short rear-specific late recovery window once a rear leg closes back into all-contact stance
    'rear_all_contact_stabilization_forward_scale': 1.0,  # scale forward reference velocity while the rear late all-contact stabilization window is active
    'rear_all_contact_stabilization_front_alpha_scale': 1.0,  # clamp front touchdown-support alpha while the rear late all-contact stabilization window is active
    'rear_all_contact_stabilization_height_ratio': 0.0,  # keep rear late all-contact stabilization alive while base height stays this low
    'rear_all_contact_stabilization_roll_threshold': None,  # optional abs roll threshold [rad] for keeping the rear late all-contact stabilization window alive
    'rear_all_contact_stabilization_pitch_threshold': None,  # optional abs pitch threshold [rad] for keeping the rear late all-contact stabilization window alive
    'rear_all_contact_stabilization_min_rear_load_share': 0.0,  # keep rear late all-contact stabilization alive while total rear vertical load share stays below this fraction
    'rear_all_contact_stabilization_min_rear_leg_load_share': 0.0,  # keep rear late all-contact stabilization alive while the active rear leg still carries less than this vertical load-share fraction
    'rear_all_contact_stabilization_retrigger_limit': 0,  # allow a small number of late all-contact stabilization renewals after the initial rear touchdown seam
    'rear_all_contact_stabilization_rear_floor_delta': 0.0,  # temporarily increase rear-load floor only during rear late all-contact stabilization
    'rear_all_contact_stabilization_front_anchor_z_blend': 0.0,  # blend front stance-target z toward the actual contacted foot height only during rear late all-contact stabilization
    'rear_all_contact_stabilization_rear_anchor_z_blend': 0.0,  # optional rear-leg counterpart for rear late all-contact stance-height blending
    'rear_all_contact_stabilization_front_anchor_z_max_delta': 0.0,  # cap front stance-target z at actual contacted foot z + this margin during rear late all-contact stabilization
    'rear_all_contact_stabilization_rear_anchor_z_max_delta': 0.0,  # optional rear-leg counterpart for late all-contact stance-target z capping
    'front_rear_transition_guard_hold_s': 0.0,  # briefly delay a front preswing when a rear touchdown seam is still unstable
    'front_rear_transition_guard_forward_scale': 1.0,  # scale forward reference velocity while the front preswing is being delayed by a rear transition seam
    'front_rear_transition_guard_roll_threshold': None,  # abs roll threshold [rad] for delaying a front preswing during a rear transition seam
    'front_rear_transition_guard_pitch_threshold': None,  # abs pitch threshold [rad] for delaying a front preswing during a rear transition seam
    'front_rear_transition_guard_height_ratio': 0.0,  # minimum base-height ratio below which a front preswing is delayed during a rear transition seam
    'touchdown_contact_vel_z_damping': 0.0,  # task-space vertical damping applied during touchdown support windows
    'front_touchdown_contact_vel_z_damping': None,  # optional front-leg override for touchdown vertical damping
    'rear_touchdown_contact_vel_z_damping': None,  # optional rear-leg override for touchdown vertical damping
    'front_margin_rescue_hold_s': 0.0,  # keep a brief late front-leg support rescue alive during unstable full-contact stance
    'front_margin_rescue_forward_scale': 1.0,  # scale forward reference velocity while late front-margin rescue is active
    'front_margin_rescue_min_margin': 0.0,  # trigger late front-margin rescue if the queried front support margin falls below this value
    'front_margin_rescue_margin_gap': 0.0,  # optionally require the rescued front leg margin to be this much worse than the opposite front leg
    'front_margin_rescue_alpha_margin': 0.02,  # support alpha reaches 1 when the rescued leg margin falls this far below the rescue threshold
    'front_margin_rescue_roll_threshold': None,  # optional abs roll threshold [rad] required before late front-margin rescue can trigger
    'front_margin_rescue_pitch_threshold': None,  # optional abs pitch threshold [rad] required before late front-margin rescue can trigger
    'front_margin_rescue_height_ratio': 0.0,  # optionally trigger late front-margin rescue when base height falls below this ratio of ref_z
    'front_margin_rescue_recent_swing_window_s': 0.0,  # require that the same front leg swung recently before late front-margin rescue may trigger
    'front_margin_rescue_require_all_contact': True,  # only allow late front-margin rescue when all feet are actually in contact
    'rear_handoff_support_hold_s': 0.0,  # keep front touchdown-style support alive briefly when a rear swing is about to start
    'rear_handoff_forward_scale': 1.0,  # scale forward reference velocity while the rear-handoff support extension is active
    'rear_handoff_lookahead_steps': 1,  # horizon steps to inspect for an upcoming rear swing before extending front touchdown support
    'rear_swing_bridge_hold_s': 0.0,  # keep front touchdown-style support alive briefly into the late rear swing transition
    'rear_swing_bridge_forward_scale': 1.0,  # scale forward reference velocity while the rear-swing bridge is active
    'rear_swing_bridge_roll_threshold': None,  # abs roll threshold [rad] that can trigger the rear-swing bridge
    'rear_swing_bridge_pitch_threshold': None,  # abs pitch threshold [rad] that can trigger the rear-swing bridge
    'rear_swing_bridge_height_ratio': 0.0,  # trigger the rear-swing bridge if base height falls below this ratio of ref_z
    'rear_swing_bridge_recent_front_window_s': 0.0,  # require that front touchdown support happened within this recent time window
    'rear_swing_bridge_lookahead_steps': 1,  # horizon steps to inspect for an upcoming rear swing when bridging support
    'rear_swing_release_support_hold_s': 0.0,  # keep temporary support overrides alive shortly after a rear leg is force-released into swing
    'rear_swing_release_forward_scale': 1.0,  # scale forward reference velocity while rear forced-release support is active
    'full_contact_recovery_hold_s': 0.0,  # keep touchdown-style support overrides alive for a short time once all feet are back in contact
    'full_contact_recovery_forward_scale': 1.0,  # scale forward reference velocity while the late full-contact recovery hold is active
    'full_contact_recovery_roll_threshold': None,  # abs roll threshold [rad] that can trigger the late full-contact recovery hold
    'full_contact_recovery_pitch_threshold': None,  # abs pitch threshold [rad] that can trigger the late full-contact recovery hold
    'full_contact_recovery_height_ratio': 0.0,  # trigger late full-contact recovery when base height falls below this ratio of ref_z
    'full_contact_recovery_recent_window_s': 0.0,  # optionally require that a front touchdown support window happened recently before enabling late full-contact recovery
    'full_contact_recovery_rear_support_scale': 0.0,  # optional rear-support alpha blended in during late full-contact recovery when a rear touchdown seam was active recently
    'crawl_front_delayed_swing_recovery_hold_s': 0.0,  # in crawl, briefly keep late full-contact recovery alive when a front leg is nominally opening swing but is still actually/load-bearing in stance
    'crawl_front_stance_support_tail_hold_s': 0.0,  # in crawl, keep the remaining front stance leg on touchdown-style support briefly after the opposite front leg actually opens swing
    'pre_swing_gate_hold_s': 0.08,  # maximum extra time to delay lift-off while waiting for enough support margin
    'pre_swing_gate_forward_scale': 1.0,  # scale forward reference velocity while lift-off is delayed by the pre-swing support-margin gate
    'vx_gain': 1.6,  # proportional gain from forward velocity error to desired body force
    'vy_gain': 4.5,  # proportional gain from lateral velocity error to desired body force
    'z_pos_gain': 20.0,  # proportional gain from base-height error to desired vertical force
    'z_vel_gain': 5.0,  # proportional gain from vertical velocity error to desired vertical force
    'min_vertical_force_scale': 0.5,  # minimum total desired vertical force as a fraction of body weight
    'reduced_support_vertical_boost': 0.0,  # extra body-weight-scaled lift request during 3-leg or 2-leg support
    'roll_angle_gain': 24.0,  # roll-angle feedback gain used in desired body torque
    'roll_rate_gain': 6.0,  # roll-rate feedback gain used in desired body torque
    'pitch_angle_gain': 28.0,  # pitch-angle feedback gain used in desired body torque
    'pitch_rate_gain': 8.0,  # pitch-rate feedback gain used in desired body torque
    'yaw_angle_gain': 4.0,  # yaw-angle feedback gain used in desired body torque
    'yaw_rate_gain': 1.5,  # yaw-rate feedback gain used in desired body torque
    'roll_ref_offset': 0.0,  # constant roll reference bias [rad] added on top of the WBInterface orientation reference
    'pitch_ref_offset': 0.0,  # constant pitch reference bias [rad] added on top of the WBInterface orientation reference
    'latched_swing_xy_blend': 0.0,  # blend relatched planned-swing feet toward the swing xy trajectory during the release window
    'latched_swing_lift_ratio': 0.0,  # raise a relatched swing leg a fraction of step height to help break contact
    'latched_swing_tau_blend': 0.0,  # blend a relatched planned-swing leg toward swing-space torque during the release window
    'contact_latch_steps': 6,  # base number of contact-sequence steps a planned-swing leg may stay relatched
    'rear_contact_latch_steps': None,  # optional rear-leg override for the planned-swing relatch horizon
    'contact_latch_budget_steps': 0,  # legacy controller-step budget; time-based config below takes precedence when set
    'contact_latch_budget_s': None,  # relatched-support budget in seconds; None falls back to legacy controller-step conversion
    'rear_contact_latch_budget_s': None,  # optional rear-leg override for the planned-swing relatch budget in seconds
    'startup_full_stance_steps': 15,  # legacy controller-step warmup; time-based config below takes precedence when set
    'startup_full_stance_time_s': None,  # full-stance warmup in seconds; None falls back to legacy controller-step conversion
    'virtual_unlatch_phase_threshold': 1.1,  # >1 disables; otherwise force controller-side swing once phase is advanced enough
    'virtual_unlatch_hold_steps': 0,  # legacy controller-step hold after virtual unlatch
    'virtual_unlatch_hold_s': None,  # controller-side swing hold in seconds after virtual unlatch; None uses legacy controller-step conversion
}
