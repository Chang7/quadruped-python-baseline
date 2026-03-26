from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import re
import traceback
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from baseline.config import make_config
from baseline.reference import rollout_reference
from baseline.model import build_prediction_model
from baseline.qp_builder import build_qp
from baseline.controller_osqp import MPCControllerOSQP
from baseline.plotting import plot_logs
from phases.mujoco_phase2_helpers import discover_model_bindings, foot_rel_world, mujoco_to_x, actual_contact_state, store_home_joint_qpos, print_binding_summary
from phases.mujoco_phase3_helpers import resolve_home_ctrl
from phases.mujoco_phase9_helpers import accumulate_actual_foot_and_support_candidates, disable_nonfoot_leg_collisions, save_phase9_plots
from phases.mujoco_visual_helpers import make_free_camera, should_capture, capture_rgb_frame, save_gif, save_mp4
from phases.mujoco_phase12_crawl_helpers import (
    init_phase12_state,
    crawl_leg_state,
    crawl_contact_schedule_rollout,
    update_phase12_state_pre,
    build_ctrl_targets_phase12,
    LEG_ORDER_NAME_TO_IDX,
)
from phases.mujoco_phase13_support_helpers import apply_trunk_support_wrench, build_phase13_summary, write_phase13_summary

DEFAULT_ORDER = (LEG_ORDER_NAME_TO_IDX['FR'], LEG_ORDER_NAME_TO_IDX['RL'], LEG_ORDER_NAME_TO_IDX['FL'], LEG_ORDER_NAME_TO_IDX['RR'])


def _create_renderer_with_fallback(m: mujoco.MjModel, width: int, height: int):
    req_w = int(width)
    req_h = int(height)
    try:
        return mujoco.Renderer(m, height=req_h, width=req_w), req_w, req_h, None
    except ValueError as exc:
        msg = str(exc)
        fw_match = re.search(r'framebuffer width\s+(\d+)', msg)
        fh_match = re.search(r'framebuffer height\s+(\d+)', msg)
        max_w = int(fw_match.group(1)) if fw_match else 640
        max_h = int(fh_match.group(1)) if fh_match else 480
        use_w = min(req_w, max_w)
        use_h = min(req_h, max_h)
        renderer = mujoco.Renderer(m, height=use_h, width=use_w)
        return renderer, use_w, use_h, msg


def run_mujoco_phase13_supported_visual(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    settle_time: float = 1.0,
    gait_ramp_time: float = 2.0,
    crawl_phase_duration: float = 0.34,
    swing_duration: float = 0.18,
    clearance: float = 0.045,
    step_len_front: float = 0.03,
    rear_step_scale: float = 0.80,
    touchdown_depth_front: float = 0.016,
    touchdown_depth_rear: float = 0.022,
    touchdown_window_front: float = 0.08,
    touchdown_window_rear: float = 0.10,
    stance_press_front: float = 0.010,
    stance_press_rear: float = 0.008,
    stance_drive_front: float = 0.002,
    stance_drive_rear: float = 0.002,
    front_unload: float = -0.006,
    height_k: float = 1.0,
    pitch_k: float = 0.04,
    roll_k: float = 0.03,
    pitch_sign: float = -1.0,
    roll_sign: float = 1.0,
    dq_limit: float = 0.08,
    disable_nonfoot_collision: bool = True,
    mpc_load_gain: float = 0.30,
    mpc_force_gain: float = 0.0,
    demo_speed_cap: float = 0.06,
    support_enabled: bool = True,
    support_weight_frac: float = 0.86,
    support_target_height: float | None = None,
    support_height_k: float = 450.0,
    support_height_d: float = 55.0,
    support_vx_k: float = 15.0,
    support_vy_k: float = 18.0,
    support_roll_k: float = 22.0,
    support_roll_d: float = 2.8,
    support_pitch_k: float = 20.0,
    support_pitch_d: float = 2.6,
    support_yaw_k: float = 6.0,
    support_yaw_d: float = 1.2,
    support_max_force_xy: float = 16.0,
    support_max_force_z: float = 180.0,
    support_max_torque: float = 18.0,
    save_gif_path: str | None = None,
    save_mp4_path: str | None = None,
    render_width: int = 640,
    render_height: int = 480,
    render_fps: int = 30,
    render_start_time: float = 0.0,
    render_end_time: float | None = None,
    camera_distance: float = 1.8,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -20.0,
    camera_lookat: tuple[float, float, float] | None = None,
):
    controller = MPCControllerOSQP(verbose=False)
    cfg_run = copy.deepcopy(cfg)
    cfg_run.desired_speed = min(float(cfg_run.desired_speed), float(demo_speed_cap))
    cfg_run.desired_accel = min(float(cfg_run.desired_accel), float(demo_speed_cap) / max(cfg_run.dt_mpc, 1e-6))

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    bindings = discover_model_bindings(m)
    print_binding_summary(bindings)
    if disable_nonfoot_collision:
        disabled = disable_nonfoot_leg_collisions(m, bindings)
        print('Disabled non-foot collision geoms:', disabled)
        mujoco.mj_forward(m, d)

    trunk_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bindings.base_body_name)
    store_home_joint_qpos(d, bindings)
    init_phase12_state(d, bindings)
    home_ctrl = resolve_home_ctrl(m, d)
    nominal_trunk_height = float(d.body(bindings.base_body_name).xpos[2])
    if support_target_height is None:
        support_target_height = min(max(0.28, 0.75 * nominal_trunk_height), nominal_trunk_height)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg_run.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg_run.nx, dtype=float)
    scheduled_contact_hold = np.ones(4, dtype=bool)
    prev_sched = np.ones(4, dtype=bool)
    current_swing_leg = None
    current_swing_phase = None

    log = {
        't': [], 'x': [], 'u': [], 'u_applied': [], 'contact': [], 'contact_actual': [],
        'contact_force_enabled': [], 'x_ref0': [],
        'actual_grf_a': [], 'actual_grf_b': [], 'actual_support_a': [], 'actual_support_b': [],
        'support_force_world': [], 'support_torque_world': [],
        'nominal_trunk_height': nominal_trunk_height,
    }

    captured_frames = []
    renderer = None
    actual_render_width = None
    actual_render_height = None
    if save_gif_path is not None or save_mp4_path is not None:
        renderer, actual_render_width, actual_render_height, fallback_msg = _create_renderer_with_fallback(m, render_width, render_height)
        if fallback_msg:
            print('Renderer fallback engaged:', fallback_msg.strip())

    def capture_if_needed(frame_count: int) -> int:
        nonlocal captured_frames
        if renderer is None:
            return frame_count
        if should_capture(d.time, frame_count, render_fps, render_start_time, render_end_time):
            cam = make_free_camera(m, d, lookat=camera_lookat, distance=camera_distance, azimuth=camera_azimuth, elevation=camera_elevation)
            captured_frames.append(capture_rgb_frame(renderer, d, cam))
            return frame_count + 1
        return frame_count

    frame_count = 0

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold, scheduled_contact_hold, prev_sched, current_swing_leg, current_swing_phase, frame_count
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_pre = actual_contact_state(m, d, bindings)

        eff_time = max(0.0, float(d.time) - settle_time)
        gait_alpha = 0.0 if d.time < settle_time else float(np.clip((d.time - settle_time) / max(gait_ramp_time, 1e-6), 0.0, 1.0))

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg_run, bindings.base_body_name)
            feet = foot_rel_world(d, bindings.base_body_name, bindings.leg_bindings, np.array([0.0, 0.0, -0.20], dtype=float))
            if d.time < settle_time:
                contact_schedule = np.ones((cfg_run.horizon, 4), dtype=bool)
                sched0 = np.ones(4, dtype=bool)
                swing_leg = None
                swing_phase = None
                x_ref = np.tile(x[None, :], (cfg_run.horizon + 1, 1))
            else:
                contact_schedule = crawl_contact_schedule_rollout(eff_time, cfg_run, crawl_phase_duration, swing_duration, DEFAULT_ORDER)
                sched0, swing_leg, swing_phase = crawl_leg_state(eff_time, crawl_phase_duration, swing_duration, DEFAULT_ORDER)
                x_ref = rollout_reference(eff_time, x, cfg_run)

            try:
                Ad_list, Bd_list = build_prediction_model(x_ref, feet, cfg_run)
                qp = build_qp(x_init=x, x_ref=x_ref, Ad_list=Ad_list, Bd_list=Bd_list, contact_schedule=contact_schedule, cfg=cfg_run)
                u_hold, _ = controller.solve(qp)
            except Exception:
                pass
            x_ref0_hold = x_ref[0].copy()
            scheduled_contact_hold = sched0.copy()
            current_swing_leg = swing_leg
            current_swing_phase = swing_phase
            next_mpc_time += cfg_run.dt_mpc

        update_phase12_state_pre(
            d=d, bindings=bindings, cfg=cfg_run,
            scheduled_contact=scheduled_contact_hold, prev_sched=prev_sched,
            step_len_front=step_len_front, rear_step_scale=rear_step_scale,
            gait_alpha=gait_alpha, vx_err=float(cfg_run.desired_speed - d.qvel[0]),
            demo_speed_cap=demo_speed_cap,
        )

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase12(
                m=m, d=d, bindings=bindings, cfg=cfg_run, home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold, actual_contact=actual_pre,
                swing_leg=current_swing_leg, swing_phase=current_swing_phase,
                u_hold=u_hold, nominal_trunk_height=float(support_target_height), gait_alpha=gait_alpha,
                settle_time=settle_time, phase_duration=crawl_phase_duration, swing_duration=swing_duration,
                step_len_front=step_len_front, rear_step_scale=rear_step_scale,
                clearance=clearance, touchdown_depth_front=touchdown_depth_front, touchdown_depth_rear=touchdown_depth_rear,
                touchdown_window_front=touchdown_window_front, touchdown_window_rear=touchdown_window_rear,
                stance_press_front=stance_press_front, stance_press_rear=stance_press_rear,
                stance_drive_front=stance_drive_front, stance_drive_rear=stance_drive_rear,
                front_unload=front_unload, height_k=height_k, pitch_k=pitch_k, roll_k=roll_k,
                pitch_sign=pitch_sign, roll_sign=roll_sign, dq_limit=dq_limit,
                mpc_load_gain=mpc_load_gain, demo_speed_cap=demo_speed_cap,
            )[0]

        d.qfrc_applied[:] = 0.0
        u_applied = np.zeros_like(u_hold)
        if mpc_force_gain > 0.0:
            base_R = d.body(bindings.base_body_name).xmat.reshape(3, 3).copy()
            for leg_i, binding in enumerate(bindings.leg_bindings):
                if not bool(scheduled_contact_hold[leg_i]) or not bool(actual_pre[leg_i]):
                    continue
                f = np.zeros(3, dtype=float)
                f[2] = float(np.clip(u_hold[3 * leg_i + 2], 0.0, cfg_run.fz_max)) * float(mpc_force_gain)
                f_world = base_R @ f
                jacp = np.zeros((3, m.nv), dtype=float)
                jacr = np.zeros((3, m.nv), dtype=float)
                p_world = np.asarray(d.geom_xpos[binding.foot_geom_id], dtype=float).copy() if binding.foot_geom_id is not None else np.asarray(d.xpos[binding.calf_body_id], dtype=float).copy()
                mujoco.mj_jac(m, d, jacp, jacr, p_world, binding.calf_body_id)
                d.qfrc_applied[:] += jacp.T @ f_world
                u_applied[3 * leg_i + 2] = f_world[2]

        support = apply_trunk_support_wrench(
            m, d, trunk_bid,
            target_height=float(support_target_height),
            target_yaw=float(x_ref0_hold[8]) if x_ref0_hold.size > 8 else 0.0,
            desired_vx=float(x_ref0_hold[3]) if x_ref0_hold.size > 3 else 0.0,
            enabled=support_enabled,
            support_weight_frac=support_weight_frac,
            height_k=support_height_k,
            height_d=support_height_d,
            vx_k=support_vx_k,
            vy_k=support_vy_k,
            roll_k=support_roll_k,
            roll_d=support_roll_d,
            pitch_k=support_pitch_k,
            pitch_d=support_pitch_d,
            yaw_k=support_yaw_k,
            yaw_d=support_yaw_d,
            max_force_xy=support_max_force_xy,
            max_force_z=support_max_force_z,
            max_torque=support_max_torque,
        )

        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_post = actual_contact_state(m, d, bindings)
        prev_sched = scheduled_contact_hold.copy()

        x_now = mujoco_to_x(d, cfg_run, bindings.base_body_name)
        foot_a, foot_b, support_a, support_b = accumulate_actual_foot_and_support_candidates(m, d, bindings)

        log['t'].append(float(d.time))
        log['x'].append(x_now.copy())
        log['u'].append(u_hold.copy())
        log['u_applied'].append(u_applied.copy())
        log['contact'].append(scheduled_contact_hold.copy())
        log['contact_actual'].append(actual_post.copy())
        log['contact_force_enabled'].append((scheduled_contact_hold & actual_post).copy())
        log['x_ref0'].append(x_ref0_hold.copy())
        log['actual_grf_a'].append(foot_a.copy())
        log['actual_grf_b'].append(foot_b.copy())
        log['actual_support_a'].append(support_a.copy())
        log['actual_support_b'].append(support_b.copy())
        log['support_force_world'].append(support['force_world'].copy())
        log['support_torque_world'].append(support['torque_world'].copy())

        nonlocal frame_count
        frame_count = capture_if_needed(frame_count)
        return d.time < cfg_run.sim_time

    if viewer:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running() and d.time < cfg_run.sim_time:
                keep_going = one_step()
                v.sync()
                if not keep_going:
                    break
    else:
        while d.time < cfg_run.sim_time:
            if not one_step():
                break

    saved = []
    summary = None
    if output_dir is not None:
        saved.extend(plot_logs(log, cfg_run, output_dir=output_dir))
        saved.extend(save_phase9_plots(log, output_dir=output_dir))
        summary = build_phase13_summary(log, settle_time=settle_time)
        summary.update({
            'settle_time': float(settle_time), 'gait_ramp_time': float(gait_ramp_time),
            'crawl_phase_duration': float(crawl_phase_duration), 'swing_duration': float(swing_duration),
            'disable_nonfoot_collision': bool(disable_nonfoot_collision), 'nominal_trunk_height': float(nominal_trunk_height),
            'support_enabled': bool(support_enabled), 'support_target_height': float(support_target_height),
            'support_weight_frac': float(support_weight_frac),
            'render_requested_width': int(render_width), 'render_requested_height': int(render_height),
            'render_actual_width': int(actual_render_width) if actual_render_width is not None else None,
            'render_actual_height': int(actual_render_height) if actual_render_height is not None else None,
            'demo_speed_cap': float(demo_speed_cap),
        })
        saved.append(write_phase13_summary(output_dir, summary))
        if save_gif_path is not None:
            Path(save_gif_path).parent.mkdir(parents=True, exist_ok=True)
            saved.append(save_gif(captured_frames, save_gif_path, fps=render_fps))
        if save_mp4_path is not None:
            Path(save_mp4_path).parent.mkdir(parents=True, exist_ok=True)
            saved.append(save_mp4(captured_frames, save_mp4_path, fps=render_fps))

    return log, saved, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Support-assisted MuJoCo visual demo using the same high-level Python MPC loop.')
    parser.add_argument('--scenario', type=str, default='straight_trot')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--settle-time', type=float, default=1.0)
    parser.add_argument('--gait-ramp-time', type=float, default=2.0)
    parser.add_argument('--crawl-phase-duration', type=float, default=0.34)
    parser.add_argument('--swing-duration', type=float, default=0.18)
    parser.add_argument('--clearance', type=float, default=0.045)
    parser.add_argument('--step-len-front', type=float, default=0.03)
    parser.add_argument('--rear-step-scale', type=float, default=0.80)
    parser.add_argument('--touchdown-depth-front', type=float, default=0.016)
    parser.add_argument('--touchdown-depth-rear', type=float, default=0.022)
    parser.add_argument('--touchdown-window-front', type=float, default=0.08)
    parser.add_argument('--touchdown-window-rear', type=float, default=0.10)
    parser.add_argument('--stance-press-front', type=float, default=0.010)
    parser.add_argument('--stance-press-rear', type=float, default=0.008)
    parser.add_argument('--stance-drive-front', type=float, default=0.002)
    parser.add_argument('--stance-drive-rear', type=float, default=0.002)
    parser.add_argument('--front-unload', type=float, default=-0.006)
    parser.add_argument('--height-k', type=float, default=1.0)
    parser.add_argument('--pitch-k', type=float, default=0.04)
    parser.add_argument('--roll-k', type=float, default=0.03)
    parser.add_argument('--pitch-sign', type=float, default=-1.0)
    parser.add_argument('--roll-sign', type=float, default=1.0)
    parser.add_argument('--dq-limit', type=float, default=0.08)
    parser.add_argument('--disable-nonfoot-collision', action='store_true')
    parser.add_argument('--mpc-load-gain', type=float, default=0.30)
    parser.add_argument('--mpc-force-gain', type=float, default=0.0)
    parser.add_argument('--demo-speed-cap', type=float, default=0.06)
    parser.add_argument('--support-enabled', action='store_true')
    parser.add_argument('--support-weight-frac', type=float, default=0.86)
    parser.add_argument('--support-target-height', type=float, default=None)
    parser.add_argument('--support-height-k', type=float, default=450.0)
    parser.add_argument('--support-height-d', type=float, default=55.0)
    parser.add_argument('--support-vx-k', type=float, default=15.0)
    parser.add_argument('--support-vy-k', type=float, default=18.0)
    parser.add_argument('--support-roll-k', type=float, default=22.0)
    parser.add_argument('--support-roll-d', type=float, default=2.8)
    parser.add_argument('--support-pitch-k', type=float, default=20.0)
    parser.add_argument('--support-pitch-d', type=float, default=2.6)
    parser.add_argument('--support-yaw-k', type=float, default=6.0)
    parser.add_argument('--support-yaw-d', type=float, default=1.2)
    parser.add_argument('--support-max-force-xy', type=float, default=16.0)
    parser.add_argument('--support-max-force-z', type=float, default=180.0)
    parser.add_argument('--support-max-torque', type=float, default=18.0)
    parser.add_argument('--save-gif', type=str, default=None)
    parser.add_argument('--save-mp4', type=str, default=None)
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--render-height', type=int, default=480)
    parser.add_argument('--render-fps', type=int, default=30)
    parser.add_argument('--render-start-time', type=float, default=0.0)
    parser.add_argument('--render-end-time', type=float, default=None)
    parser.add_argument('--camera-distance', type=float, default=1.8)
    parser.add_argument('--camera-azimuth', type=float, default=135.0)
    parser.add_argument('--camera-elevation', type=float, default=-20.0)
    args = parser.parse_args()
    cfg = make_config(args.scenario)
    try:
        _, saved, summary = run_mujoco_phase13_supported_visual(
            cfg=cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=args.output_dir,
            settle_time=args.settle_time,
            gait_ramp_time=args.gait_ramp_time,
            crawl_phase_duration=args.crawl_phase_duration,
            swing_duration=args.swing_duration,
            clearance=args.clearance,
            step_len_front=args.step_len_front,
            rear_step_scale=args.rear_step_scale,
            touchdown_depth_front=args.touchdown_depth_front,
            touchdown_depth_rear=args.touchdown_depth_rear,
            touchdown_window_front=args.touchdown_window_front,
            touchdown_window_rear=args.touchdown_window_rear,
            stance_press_front=args.stance_press_front,
            stance_press_rear=args.stance_press_rear,
            stance_drive_front=args.stance_drive_front,
            stance_drive_rear=args.stance_drive_rear,
            front_unload=args.front_unload,
            height_k=args.height_k,
            pitch_k=args.pitch_k,
            roll_k=args.roll_k,
            pitch_sign=args.pitch_sign,
            roll_sign=args.roll_sign,
            dq_limit=args.dq_limit,
            disable_nonfoot_collision=args.disable_nonfoot_collision,
            mpc_load_gain=args.mpc_load_gain,
            mpc_force_gain=args.mpc_force_gain,
            demo_speed_cap=args.demo_speed_cap,
            support_enabled=args.support_enabled,
            support_weight_frac=args.support_weight_frac,
            support_target_height=args.support_target_height,
            support_height_k=args.support_height_k,
            support_height_d=args.support_height_d,
            support_vx_k=args.support_vx_k,
            support_vy_k=args.support_vy_k,
            support_roll_k=args.support_roll_k,
            support_roll_d=args.support_roll_d,
            support_pitch_k=args.support_pitch_k,
            support_pitch_d=args.support_pitch_d,
            support_yaw_k=args.support_yaw_k,
            support_yaw_d=args.support_yaw_d,
            support_max_force_xy=args.support_max_force_xy,
            support_max_force_z=args.support_max_force_z,
            support_max_torque=args.support_max_torque,
            save_gif_path=args.save_gif,
            save_mp4_path=args.save_mp4,
            render_width=args.render_width,
            render_height=args.render_height,
            render_fps=args.render_fps,
            render_start_time=args.render_start_time,
            render_end_time=args.render_end_time,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        )
        print(f'MuJoCo phase-13 supported run finished for scenario: {cfg.scenario_name}')
        print(f'Model: {args.model}')
        if summary is not None:
            print('Mean vx after 1.0 s:', summary.get('mean_vx_after_1s'))
            print('Mean trunk height after 1.0 s:', summary.get('mean_trunk_height_after_1s'))
            print('Min trunk height after 1.0 s:', summary.get('min_trunk_height_after_1s'))
            print('Collapse time:', summary.get('collapse_time'))
            print('Mean support force world after 1.0 s:', summary.get('mean_support_force_world_after_1s'))
        if saved:
            print('Saved outputs:')
            for p in saved:
                print(' -', p)
    except Exception:
        print('MuJoCo phase-13 supported run failed. Full traceback below:')
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
