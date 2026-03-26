from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from baseline.config import make_config
from baseline.fsm import rollout_contact_schedule
from baseline.reference import rollout_reference
from baseline.model import build_prediction_model
from baseline.qp_builder import build_qp
from baseline.controller_osqp import MPCControllerOSQP
from baseline.plotting import plot_logs
from phases.mujoco_phase2_helpers import discover_model_bindings, foot_rel_world, mujoco_to_x, actual_contact_state, store_home_joint_qpos, print_binding_summary
from phases.mujoco_phase3_helpers import resolve_home_ctrl
from phases.mujoco_phase9_helpers import accumulate_actual_foot_and_support_candidates, disable_nonfoot_leg_collisions, save_phase9_plots
from phases.mujoco_visual_helpers import make_free_camera, should_capture, capture_rgb_frame, save_gif, save_mp4
from phases.mujoco_phase11_stabilized_helpers import (
    init_phase11_state,
    update_phase11_state_pre,
    build_ctrl_targets_phase11,
    build_phase11_summary,
    write_phase11_summary,
)


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


def run_mujoco_phase11_stabilized_visual(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    settle_time: float = 0.60,
    gait_ramp_time: float = 0.80,
    clearance: float = 0.06,
    step_len_front: float = 0.055,
    rear_step_scale: float = 0.72,
    touchdown_depth_front: float = 0.028,
    touchdown_depth_rear: float = 0.040,
    touchdown_window_front: float = 0.080,
    touchdown_window_rear: float = 0.120,
    stance_press_front: float = 0.006,
    stance_press_rear: float = 0.018,
    rear_back_bias: float = 0.010,
    front_unload: float = 0.002,
    height_k: float = 0.90,
    pitch_k: float = 0.030,
    roll_k: float = 0.020,
    speed_back_k: float = 0.020,
    dq_limit: float = 0.16,
    disable_nonfoot_collision: bool = True,
    # Optional small direct MPC-force feedforward; off by default for stability.
    mpc_force_gain: float = 0.0,
    # visual / recording
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
) -> tuple[dict, list[str], dict | None]:
    controller = MPCControllerOSQP(verbose=False)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    bindings = discover_model_bindings(m)
    print_binding_summary(bindings)
    if disable_nonfoot_collision:
        disabled = disable_nonfoot_leg_collisions(m, bindings)
        print('Disabled non-foot collision geoms:', disabled)
        mujoco.mj_forward(m, d)

    store_home_joint_qpos(d, bindings)
    init_phase11_state(m, d, bindings)
    home_ctrl = resolve_home_ctrl(m, d)
    nominal_trunk_height = float(d.body(bindings.base_body_name).xpos[2])

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.ones(4, dtype=bool)
    prev_sched = np.zeros(4, dtype=bool)

    log = {
        't': [],
        'x': [],
        'u': [],
        'u_applied': [],
        'contact': [],
        'contact_actual': [],
        'contact_force_enabled': [],
        'x_ref0': [],
        'actual_grf_a': [],
        'actual_grf_b': [],
        'actual_support_a': [],
        'actual_support_b': [],
        'nominal_trunk_height': nominal_trunk_height,
    }

    captured_frames: list[np.ndarray] = []
    renderer = None
    actual_render_width = None
    actual_render_height = None
    renderer_fallback_msg = None
    if save_gif_path is not None or save_mp4_path is not None:
        renderer, actual_render_width, actual_render_height, renderer_fallback_msg = _create_renderer_with_fallback(
            m, render_width, render_height
        )
        if renderer_fallback_msg:
            print('Renderer fallback engaged:', renderer_fallback_msg.strip())

    def capture_if_needed(frame_count: int) -> int:
        nonlocal captured_frames
        if renderer is None:
            return frame_count
        if should_capture(d.time, frame_count, render_fps, render_start_time, render_end_time):
            cam = make_free_camera(
                m, d,
                lookat=camera_lookat,
                distance=camera_distance,
                azimuth=camera_azimuth,
                elevation=camera_elevation,
            )
            captured_frames.append(capture_rgb_frame(renderer, d, cam))
            return frame_count + 1
        return frame_count

    frame_count = 0

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold, scheduled_contact_hold, prev_sched, frame_count

        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_pre = actual_contact_state(m, d, bindings)

        # Warm-start settle phase.
        eff_time = max(0.0, float(d.time) - settle_time)
        gait_alpha = 0.0 if d.time < settle_time else float(np.clip((d.time - settle_time) / max(gait_ramp_time, 1e-6), 0.0, 1.0))

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg, bindings.base_body_name)
            feet = foot_rel_world(d, bindings.base_body_name, bindings.leg_bindings, np.array([0.0, 0.0, -0.20], dtype=float))
            if d.time < settle_time:
                contact_schedule = np.ones((cfg.horizon, 4), dtype=bool)
                x_ref = np.tile(x[None, :], (cfg.horizon + 1, 1))
            else:
                contact_schedule = rollout_contact_schedule(eff_time, cfg)
                x_ref = rollout_reference(eff_time, x, cfg)

            try:
                Ad_list, Bd_list = build_prediction_model(x_ref, feet, cfg)
                qp = build_qp(
                    x_init=x,
                    x_ref=x_ref,
                    Ad_list=Ad_list,
                    Bd_list=Bd_list,
                    contact_schedule=contact_schedule,
                    cfg=cfg,
                )
                u_hold, _ = controller.solve(qp)
            except Exception:
                # If MPC solve has a transient issue, keep last value rather than exploding the low-level stabilizer.
                pass
            x_ref0_hold = x_ref[0].copy()
            scheduled_contact_hold = contact_schedule[0].copy() if d.time >= settle_time else np.ones(4, dtype=bool)
            next_mpc_time += cfg.dt_mpc

        x_now = mujoco_to_x(d, cfg, bindings.base_body_name)
        vx_err = float(cfg.desired_speed - x_now[3])

        update_phase11_state_pre(
            d=d,
            bindings=bindings,
            cfg=cfg,
            scheduled_contact=scheduled_contact_hold,
            settle_time=settle_time,
            step_len_front=step_len_front,
            rear_step_scale=rear_step_scale,
            gait_alpha=gait_alpha,
            vx_err=vx_err,
            prev_sched=prev_sched,
        )

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase11(
                m=m,
                d=d,
                bindings=bindings,
                cfg=cfg,
                home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold,
                actual_contact=actual_pre,
                u_hold=u_hold,
                settle_time=settle_time,
                gait_ramp_time=gait_ramp_time,
                nominal_trunk_height=nominal_trunk_height,
                step_len_front=step_len_front,
                rear_step_scale=rear_step_scale,
                clearance=clearance,
                touchdown_depth_front=touchdown_depth_front,
                touchdown_depth_rear=touchdown_depth_rear,
                touchdown_window_front=touchdown_window_front,
                touchdown_window_rear=touchdown_window_rear,
                stance_press_front=stance_press_front,
                stance_press_rear=stance_press_rear,
                rear_back_bias=rear_back_bias,
                front_unload=front_unload,
                height_k=height_k,
                pitch_k=pitch_k,
                roll_k=roll_k,
                speed_back_k=speed_back_k,
                dq_limit=dq_limit,
            )[0]

        # By default, do not inject external wrench: this runner prioritizes visible stability.
        d.qfrc_applied[:] = 0.0
        u_applied = np.zeros_like(u_hold)
        if mpc_force_gain > 0.0:
            # Optional small feedforward: spread only a tiny portion of vertical MPC load.
            base_R = d.body(bindings.base_body_name).xmat.reshape(3, 3).copy()
            for leg_i, binding in enumerate(bindings.leg_bindings):
                if not bool(scheduled_contact_hold[leg_i]) or not bool(actual_pre[leg_i]):
                    continue
                f = np.zeros(3, dtype=float)
                f[2] = float(np.clip(u_hold[3 * leg_i + 2], 0.0, cfg.fz_max)) * float(mpc_force_gain)
                f_world = base_R @ f
                p_world = np.asarray(d.geom_xpos[binding.foot_geom_id], dtype=float).copy() if binding.foot_geom_id is not None else np.asarray(d.xpos[binding.calf_body_id], dtype=float).copy()
                jacp = np.zeros((3, m.nv), dtype=float)
                jacr = np.zeros((3, m.nv), dtype=float)
                mujoco.mj_jac(m, d, jacp, jacr, p_world, binding.calf_body_id)
                d.qfrc_applied[:] += jacp.T @ f_world
                u_applied[3 * leg_i + 2] = f_world[2]

        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_post = actual_contact_state(m, d, bindings)
        prev_sched = scheduled_contact_hold.copy()

        x_now = mujoco_to_x(d, cfg, bindings.base_body_name)
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

        frame_count = capture_if_needed(frame_count)
        return d.time < cfg.sim_time

    if viewer:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running() and d.time < cfg.sim_time:
                keep_going = one_step()
                v.sync()
                if not keep_going:
                    break
    else:
        while d.time < cfg.sim_time:
            if not one_step():
                break

    saved = []
    summary = None
    if output_dir is not None:
        saved.extend(plot_logs(log, cfg, output_dir=output_dir))
        saved.extend(save_phase9_plots(log, output_dir=output_dir))
        summary = build_phase11_summary(log, cfg, settle_time=settle_time)
        summary.update({
            'settle_time': float(settle_time),
            'gait_ramp_time': float(gait_ramp_time),
            'disable_nonfoot_collision': bool(disable_nonfoot_collision),
            'nominal_trunk_height': float(nominal_trunk_height),
            'render_requested_width': int(render_width),
            'render_requested_height': int(render_height),
            'render_actual_width': int(actual_render_width) if actual_render_width is not None else None,
            'render_actual_height': int(actual_render_height) if actual_render_height is not None else None,
        })
        saved.append(write_phase11_summary(output_dir, summary))

        if save_gif_path is not None:
            Path(save_gif_path).parent.mkdir(parents=True, exist_ok=True)
            saved.append(save_gif(captured_frames, save_gif_path, fps=render_fps))
        if save_mp4_path is not None:
            Path(save_mp4_path).parent.mkdir(parents=True, exist_ok=True)
            saved.append(save_mp4(captured_frames, save_mp4_path, fps=render_fps))

    return log, saved, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Stabilization-oriented MuJoCo runner for visual debugging.')
    parser.add_argument('--scenario', type=str, default='straight_trot')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)

    parser.add_argument('--settle-time', type=float, default=0.60)
    parser.add_argument('--gait-ramp-time', type=float, default=0.80)
    parser.add_argument('--clearance', type=float, default=0.06)
    parser.add_argument('--step-len-front', type=float, default=0.055)
    parser.add_argument('--rear-step-scale', type=float, default=0.72)
    parser.add_argument('--touchdown-depth-front', type=float, default=0.028)
    parser.add_argument('--touchdown-depth-rear', type=float, default=0.040)
    parser.add_argument('--touchdown-window-front', type=float, default=0.080)
    parser.add_argument('--touchdown-window-rear', type=float, default=0.120)
    parser.add_argument('--stance-press-front', type=float, default=0.006)
    parser.add_argument('--stance-press-rear', type=float, default=0.018)
    parser.add_argument('--rear-back-bias', type=float, default=0.010)
    parser.add_argument('--front-unload', type=float, default=0.002)
    parser.add_argument('--height-k', type=float, default=0.90)
    parser.add_argument('--pitch-k', type=float, default=0.030)
    parser.add_argument('--roll-k', type=float, default=0.020)
    parser.add_argument('--speed-back-k', type=float, default=0.020)
    parser.add_argument('--dq-limit', type=float, default=0.16)
    parser.add_argument('--disable-nonfoot-collision', action='store_true')
    parser.add_argument('--mpc-force-gain', type=float, default=0.0)

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
        _, saved, summary = run_mujoco_phase11_stabilized_visual(
            cfg=cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=args.output_dir,
            settle_time=args.settle_time,
            gait_ramp_time=args.gait_ramp_time,
            clearance=args.clearance,
            step_len_front=args.step_len_front,
            rear_step_scale=args.rear_step_scale,
            touchdown_depth_front=args.touchdown_depth_front,
            touchdown_depth_rear=args.touchdown_depth_rear,
            touchdown_window_front=args.touchdown_window_front,
            touchdown_window_rear=args.touchdown_window_rear,
            stance_press_front=args.stance_press_front,
            stance_press_rear=args.stance_press_rear,
            rear_back_bias=args.rear_back_bias,
            front_unload=args.front_unload,
            height_k=args.height_k,
            pitch_k=args.pitch_k,
            roll_k=args.roll_k,
            speed_back_k=args.speed_back_k,
            dq_limit=args.dq_limit,
            disable_nonfoot_collision=args.disable_nonfoot_collision,
            mpc_force_gain=args.mpc_force_gain,
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
        print(f'MuJoCo phase-11 stabilized run finished for scenario: {cfg.scenario_name}')
        print(f'Model: {args.model}')
        if summary is not None:
            print('Mean vx after 1.0 s:', summary.get('mean_vx_after_1s'))
            print('Mean trunk height after 1.0 s:', summary.get('mean_trunk_height_after_1s'))
            print('Min trunk height after 1.0 s:', summary.get('min_trunk_height_after_1s'))
            print('Collapse time:', summary.get('collapse_time'))
        if saved:
            print('Saved outputs:')
            for p in saved:
                print(' -', p)
    except Exception:
        print('MuJoCo phase-11 stabilized run failed. Full traceback below:')
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
