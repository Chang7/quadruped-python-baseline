# Description: simulate the full quadruped model in MuJoCo and optionally save analysis artifacts.
from __future__ import annotations

import copy
import pathlib
import time
from os import PathLike
from pprint import pprint
from typing import Any

import mujoco
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

try:
    from .artifacts import append_step, finalize_log, init_run_log, save_npz, save_plots, save_summary, save_topdown_mp4, summarize_log
except ImportError:
    from simulation.artifacts import append_step, finalize_log, init_run_log, save_npz, save_plots, save_summary, save_topdown_mp4, summarize_log


def _init_mujoco_video(model, out_file: pathlib.Path, fps: int = 25, width: int = 960, height: int = 640):
    try:
        import cv2
    except ImportError:
        print("OpenCV not available; skipping MuJoCo scene mp4 export.")
        return None

    width = min(int(width), int(getattr(model.vis.global_, "offwidth", width)))
    height = min(int(height), int(getattr(model.vis.global_, "offheight", height)))

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        print(f"Unable to initialize MuJoCo offscreen renderer; skipping scene mp4 export. ({exc})")
        return None

    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        renderer.close()
        print(f"Unable to open video writer for {out_file}; skipping scene mp4 export.")
        return None

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = max(1.5, float(model.stat.extent) * 2.5)

    return {
        "renderer": renderer,
        "writer": writer,
        "camera": camera,
        "fps": int(fps),
        "next_time": 0.0,
        "path": out_file,
    }


def _write_mujoco_video_frame(video_state, data) -> None:
    renderer = video_state["renderer"]
    writer = video_state["writer"]
    camera = video_state["camera"]

    base_pos = np.asarray(data.qpos[:3], dtype=float).reshape(3)
    camera.lookat[:] = base_pos
    renderer.update_scene(data, camera=camera)
    rgb = np.asarray(renderer.render(), dtype=np.uint8)
    writer.write(rgb[:, :, ::-1])


def _close_mujoco_video(video_state):
    if video_state is None:
        return None
    video_state["writer"].release()
    video_state["renderer"].close()
    return video_state["path"]


def _apply_contact_overrides(model, simulation_params) -> None:
    if simulation_params is None:
        return

    condim_override = simulation_params.get("contact_condim_override", None)
    impratio_override = simulation_params.get("contact_impratio_override", None)
    torsional_override = simulation_params.get("contact_torsional_friction_override", None)
    rolling_override = simulation_params.get("contact_rolling_friction_override", None)

    if condim_override is None and impratio_override is None and torsional_override is None and rolling_override is None:
        return

    if impratio_override is not None:
        model.opt.impratio = max(float(impratio_override), 1.0)

    target_geoms = {"floor", "FL", "FR", "RL", "RR"}
    for geom_id in range(int(model.ngeom)):
        geom = model.geom(geom_id)
        name = (getattr(geom, "name", "") or "").strip()
        if name not in target_geoms:
            continue
        if condim_override is not None:
            model.geom_condim[geom_id] = int(condim_override)
        if torsional_override is not None:
            model.geom_friction[geom_id, 1] = max(float(torsional_override), 0.0)
        if rolling_override is not None:
            model.geom_friction[geom_id, 2] = max(float(rolling_override), 0.0)


def _compute_scheduled_disturbance_wrench(
    disturbance_schedule: list[dict[str, Any]] | None,
    sim_time: float,
) -> np.ndarray:
    """Return the current world-frame 6D wrench from user-scheduled disturbance pulses.

    Each pulse is a dict with:
    - time_s: pulse start time
    - duration_s: pulse duration
    - wrench: 6D world-frame wrench [Fx, Fy, Fz, Mx, My, Mz]

    A squared-sine envelope keeps the pulse smooth and zero at the ends.
    """
    if not disturbance_schedule:
        return np.zeros(6, dtype=float)

    wrench = np.zeros(6, dtype=float)
    now = float(sim_time)
    for pulse in disturbance_schedule:
        start = float(pulse.get("time_s", 0.0))
        duration = max(float(pulse.get("duration_s", 0.0)), 1e-6)
        phase = (now - start) / duration
        if phase < 0.0 or phase > 1.0:
            continue
        alpha = float(np.sin(np.pi * phase) ** 2)
        wrench += alpha * np.asarray(pulse.get("wrench", np.zeros(6)), dtype=float).reshape(6)
    return wrench


def run_simulation(
    qpympc_cfg,
    process=0,
    num_episodes=500,
    num_seconds_per_episode=60,
    ref_base_lin_vel=(0.0, 4.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="human",
    seed=0,
    render=True,
    recording_path: PathLike | None = None,
    artifact_dir: PathLike | None = None,
    save_plots_flag: bool = True,
    save_mp4_flag: bool = True,
    stop_on_terminate: bool = True,
    random_reset_on_terminate: bool = False,
    controller_ref_base_lin_vel: np.ndarray | None = None,
    controller_ref_base_ang_vel: np.ndarray | None = None,
    disturbance_schedule: list[dict[str, Any]] | None = None,
):
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # IMPORTANT: save actual observables; an empty tuple produces effectively empty trajectories.
    state_obs_names = list(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,
        ref_base_ang_vel=ref_base_ang_vel,
        ground_friction_coeff=friction_coeff,
        base_vel_command_type=base_vel_command_type,
        state_obs_names=tuple(state_obs_names),
    )
    pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant
    _apply_contact_overrides(env.mjModel, qpympc_cfg.simulation_params)

    fixed_ref_base_lin_vel = None
    if controller_ref_base_lin_vel is not None:
        fixed_ref_base_lin_vel = np.asarray(controller_ref_base_lin_vel, dtype=float).reshape(3)
    fixed_ref_base_ang_vel = None
    if controller_ref_base_ang_vel is not None:
        fixed_ref_base_ang_vel = np.asarray(controller_ref_base_ang_vel, dtype=float).reshape(3)

    ref_base_lin_vel_label = tuple(np.asarray(ref_base_lin_vel, dtype=float).reshape(-1).tolist())
    ref_base_ang_vel_label = tuple(np.asarray(ref_base_ang_vel, dtype=float).reshape(-1).tolist())
    if fixed_ref_base_lin_vel is not None:
        ref_base_lin_vel_label = tuple(np.round(fixed_ref_base_lin_vel, 6).tolist())
    if fixed_ref_base_ang_vel is not None:
        ref_base_ang_vel_label = tuple(np.round(fixed_ref_base_ang_vel, 6).tolist())

    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_heightmap = 0.04
        num_rows_heightmap = 7
        num_cols_heightmap = 7
        heightmaps = LegsAttr(
            FL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            FR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
        )
    else:
        heightmaps = None

    quadrupedpympc_observables_names = (
        "ref_base_height",
        "ref_base_angles",
        "ref_feet_pos",
        "des_foot_pos",
        "des_foot_vel",
        "nmpc_GRFs",
        "nmpc_footholds",
        "swing_time",
        "phase_signal",
        "lift_off_positions",
        "planned_contact",
        "current_contact",
        "swing_contact_release_active",
        "latched_release_alpha",
        "latched_swing_time",
        "support_margin",
        "support_confirm_active",
        "pre_swing_gate_active",
        "front_late_release_active",
        "touchdown_reacquire_active",
        "touchdown_confirm_active",
        "touchdown_settle_active",
        "touchdown_support_active",
        "rear_retry_contact_signal",
        "rear_touchdown_contact_ready",
        "rear_late_stance_contact_ready",
        "rear_all_contact_support_needed",
        "rear_late_seam_elapsed_s",
        "rear_late_seam_support_active",
        "rear_close_handoff_active",
        "rear_late_load_share_active",
        "rear_late_load_share_alpha",
        "rear_late_load_share_candidate_active",
        "rear_late_load_share_candidate_alpha",
        "rear_late_load_share_trigger_elapsed_s",
        "rear_late_load_share_trigger_enabled",
        "rear_close_handoff_alpha",
        "rear_close_handoff_leg_index",
        "rear_all_contact_weak_leg_alpha",
        "rear_all_contact_weak_leg_index",
        "applied_linear_support_force_floor_ratio",
        "applied_linear_rear_handoff_leg_index",
        "applied_linear_rear_handoff_leg_floor_scale",
        "applied_linear_latched_force_scale",
        "applied_linear_latched_front_receiver_scale",
        "applied_linear_latched_rear_receiver_scale",
        "rear_touchdown_actual_contact_elapsed_s",
        "rear_touchdown_pending_confirm",
        "front_margin_rescue_active",
        "front_margin_rescue_alpha",
        "touchdown_support_alpha",
        "rear_all_contact_stabilization_alpha",
        "rear_all_contact_front_planted_tail_alpha",
        "crawl_front_planted_seam_support_alpha",
        "rear_handoff_support_active",
        "rear_swing_bridge_active",
        "rear_swing_release_support_active",
        "full_contact_recovery_active",
        "full_contact_recovery_alpha",
        "full_contact_recovery_remaining_s",
        "full_contact_recovery_trigger",
        "front_delayed_swing_recovery_trigger",
        "planted_front_recovery_trigger",
        "planted_front_postdrop_recovery_trigger",
        "front_close_gap_trigger",
        "front_late_rearm_trigger",
        "front_planted_posture_tail_trigger",
        "front_late_posture_tail_trigger",
        "crawl_front_stance_support_tail_remaining_s",
        "front_touchdown_support_recent_remaining_s",
        "front_delayed_swing_recovery_spent",
        "gate_forward_scale",
    )
    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    if recording_path is not None:
        from gym_quadruped.utils.data.h5py import H5Writer

        root_path = pathlib.Path(recording_path)
        root_path.mkdir(parents=True, exist_ok=True)
        dataset_path = (
            root_path
            / f"{robot_name}/{scene_name}"
            / f"lin_vel={ref_base_lin_vel_label} ang_vel={ref_base_ang_vel_label} friction={friction_coeff}"
            / f"ep={num_episodes}_steps={int(num_seconds_per_episode // simulation_dt):d}.h5"
        )
        h5py_writer = H5Writer(file_path=dataset_path, env=env, extra_obs=None)
        print(f"\n Recording data to: {dataset_path.absolute()}")
    else:
        h5py_writer = None
        dataset_path = None

    artifact_root = pathlib.Path(artifact_dir) if artifact_dir else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    RENDER_FREQ = 30
    N_EPISODES = num_episodes
    N_STEPS_PER_EPISODE = int(num_seconds_per_episode // simulation_dt)
    last_render_time = time.time()

    for episode_num in range(N_EPISODES):
        if episode_num > 0:
            env.reset(random=random_reset_on_terminate)
            quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))

        ep_state_history, ep_time = [], []
        run_log = init_run_log(qpympc_cfg.mpc_params["type"], qpympc_cfg.simulation_params["gait"], robot_name, scene_name)
        terminated_in_episode = False
        truncated_in_episode = False
        ep_root = artifact_root / f"episode_{episode_num:03d}" if artifact_root is not None else None
        if ep_root is not None:
            ep_root.mkdir(parents=True, exist_ok=True)
        mujoco_video = _init_mujoco_video(env.mjModel, ep_root / "mujoco_scene.mp4") if (ep_root is not None and save_mp4_flag) else None
        if mujoco_video is not None:
            _write_mujoco_video_frame(mujoco_video, env.mjData)
            mujoco_video["next_time"] = 1.0 / float(mujoco_video["fps"])

        for _ in tqdm(range(N_STEPS_PER_EPISODE), desc=f"Ep:{episode_num:d}-steps:", total=N_STEPS_PER_EPISODE):
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame='world')
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            if fixed_ref_base_lin_vel is None:
                ref_base_lin_vel_now, ref_base_ang_vel_now = env.target_base_vel()
            else:
                ref_base_lin_vel_now = fixed_ref_base_lin_vel.copy()
                ref_base_ang_vel_now = (
                    fixed_ref_base_ang_vel.copy()
                    if fixed_ref_base_ang_vel is not None
                    else np.zeros(3, dtype=float)
                )
            inertia = env.get_base_inertia().flatten() if qpympc_cfg.simulation_params["use_inertia_recomputation"] else qpympc_cfg.inertia.flatten()
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx
            joints_pos = LegsAttr(
                FL=np.asarray(qpos[legs_qpos_idx.FL], dtype=float).copy(),
                FR=np.asarray(qpos[legs_qpos_idx.FR], dtype=float).copy(),
                RL=np.asarray(qpos[legs_qpos_idx.RL], dtype=float).copy(),
                RR=np.asarray(qpos[legs_qpos_idx.RR], dtype=float).copy(),
            )
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)
            try:
                foot_contact_state_now, _, foot_grf_state_now = env.feet_contact_state(ground_reaction_forces=True)
            except Exception:
                foot_grf_state_now = None
                try:
                    foot_contact_state_now, _ = env.feet_contact_state()
                except Exception:
                    foot_contact_state_now = None

            tau = quadrupedpympc_wrapper.compute_actions(
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
                ref_base_lin_vel_now,
                ref_base_ang_vel_now,
                env.step_num,
                qpos,
                qvel,
                feet_jac,
                feet_jac_dot,
                feet_vel,
                legs_qfrc_passive,
                legs_qfrc_bias,
                legs_mass_matrix,
                legs_qpos_idx,
                legs_qvel_idx,
                tau,
                inertia,
                env.mjData.contact,
                foot_contact_state_now,
                foot_grf_state_now,
            )
            for leg in ["FL", "FR", "RL", "RR"]:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            env.mjData.qfrc_applied[:6] = _compute_scheduled_disturbance_wrench(
                disturbance_schedule,
                env.simulation_time,
            )

            state, reward, is_terminated, is_truncated, info = env.step(action=action)
            ctrl_state = quadrupedpympc_wrapper.get_obs()
            if is_terminated or is_truncated:
                run_log["meta"]["termination_step"] = int(info.get("step_num", env.step_num))
                run_log["meta"]["termination_time"] = float(env.simulation_time)
                invalid_keys = list(info.get("invalid_contacts", {}).keys())
                if invalid_keys:
                    run_log["meta"]["invalid_contact_keys"] = invalid_keys

            # Re-read post-step signals for logging/plots.
            feet_pos_log = env.feet_pos(frame="world")
            feet_vel_log = env.feet_vel(frame='world')
            base_lin_vel_log = env.base_lin_vel(frame="world")
            base_ang_vel_log = env.base_ang_vel(frame="base")
            base_ori_euler_xyz_log = env.base_ori_euler_xyz
            base_pos_log = copy.deepcopy(env.base_pos)
            com_pos_log = copy.deepcopy(env.com)
            qpos_log, qvel_log = env.mjData.qpos.copy(), env.mjData.qvel.copy()
            if fixed_ref_base_lin_vel is None:
                ref_base_lin_vel_log, ref_base_ang_vel_log = env.target_base_vel()
            else:
                ref_base_lin_vel_log = fixed_ref_base_lin_vel.copy()
                ref_base_ang_vel_log = (
                    fixed_ref_base_ang_vel.copy()
                    if fixed_ref_base_ang_vel is not None
                    else np.zeros(3, dtype=float)
                )
            try:
                _, foot_contact_log, feet_GRF_log = env.feet_contact_state(ground_reaction_forces=True)
            except Exception:
                foot_contact_log = np.zeros(4)
                feet_GRF_log = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))

            ep_state_history.append(state)
            ep_time.append(env.simulation_time)
            append_step(
                run_log,
                sim_time=env.simulation_time,
                reward=reward,
                terminated=is_terminated,
                truncated=is_truncated,
                step_num=env.step_num,
                base_pos=base_pos_log,
                com_pos=com_pos_log,
                base_lin_vel=base_lin_vel_log,
                base_ang_vel=base_ang_vel_log,
                base_ori_euler_xyz=base_ori_euler_xyz_log,
                ref_base_lin_vel=ref_base_lin_vel_log,
                ref_base_ang_vel=ref_base_ang_vel_log,
                action=action,
                qpos=qpos_log,
                qvel=qvel_log,
                feet_pos=feet_pos_log,
                feet_vel=feet_vel_log,
                foot_contact=foot_contact_log,
                foot_grf=feet_GRF_log,
                ctrl_state=ctrl_state,
            )
            if mujoco_video is not None:
                while env.simulation_time + 1e-12 >= float(mujoco_video["next_time"]):
                    _write_mujoco_video_frame(mujoco_video, env.mjData)
                    mujoco_video["next_time"] += 1.0 / float(mujoco_video["fps"])

            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                feet_GRF_render = feet_GRF_log
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=ctrl_state["swing_time"][0],
                        FR=ctrl_state["swing_time"][1],
                        RL=ctrl_state["swing_time"][2],
                        RR=ctrl_state["swing_time"][3],
                    ),
                    lift_off_positions=ctrl_state["lift_off_positions"],
                    nmpc_footholds=ctrl_state["nmpc_footholds"],
                    ref_feet_pos=ctrl_state["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
                    for leg_name in legs_order:
                        data = heightmaps[leg_name].data
                        if data is not None:
                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                    heightmaps[leg_name].geom_ids[i, j] = render_sphere(
                                        viewer=env.viewer,
                                        position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                                        diameter=0.01,
                                        color=[0, 1, 0, 0.5],
                                        geom_id=heightmaps[leg_name].geom_ids[i, j],
                                    )

                for leg_name in legs_order:
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF_render[leg_name],
                        pos=feet_pos_log[leg_name],
                        scale=np.linalg.norm(feet_GRF_render[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )
                env.render()
                last_render_time = time.time()

            if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                if is_terminated:
                    print("Environment terminated")
                    terminated_in_episode = True
                if is_truncated:
                    truncated_in_episode = True
                if stop_on_terminate:
                    break
                env.reset(random=random_reset_on_terminate)
                quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))

        if h5py_writer is not None and ep_state_history:
            ep_obs_history = collate_obs(ep_state_history)
            ep_traj_time = np.asarray(ep_time)[:, np.newaxis]
            h5py_writer.append_trajectory(state_obs_traj=ep_obs_history, time=ep_traj_time)

        final_log = finalize_log(run_log)
        if artifact_root is not None:
            scene_video_path = _close_mujoco_video(mujoco_video)
            if scene_video_path is not None:
                print(f"Saved MuJoCo scene video to: {scene_video_path}")
            save_npz(final_log, ep_root / "run_log.npz")
            summary = summarize_log(final_log)
            save_summary(summary, ep_root / "summary.json")
            if save_plots_flag:
                save_plots(final_log, ep_root)
            if save_mp4_flag:
                save_topdown_mp4(final_log, ep_root / "topdown_motion.mp4")
            print(f"Saved artifacts to: {ep_root}")
        else:
            _close_mujoco_video(mujoco_video)

        if stop_on_terminate and (terminated_in_episode or truncated_in_episode):
            break

    env.close()
    if h5py_writer is not None:
        return h5py_writer.file_path
    if artifact_root is not None:
        return artifact_root
    return None


def collate_obs(list_of_dicts) -> dict[str, np.ndarray]:
    if not list_of_dicts:
        raise ValueError("Input list is empty.")
    keys = list(list_of_dicts[0].keys())
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}
    collated = {key: v[:, None] if v.ndim == 1 else v for key, v in collated.items()}
    return collated


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    qpympc_cfg = cfg
    run_simulation(qpympc_cfg=qpympc_cfg)
