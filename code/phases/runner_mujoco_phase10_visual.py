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
from phases.mujoco_phase5_helpers import (
    FALLBACK_FOOT_LOCAL_OFFSET,
    discover_model_bindings,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    compute_swing_delta_maps,
    resolve_home_ctrl,
    initialize_phase5_leg_state,
    update_phase5_state_pre,
    update_phase5_state_post,
    stance_force_enable_mask_phase5,
    print_binding_summary,
)
from phases.mujoco_phase7_helpers import leg_internal_force_to_qfrc
from phases.mujoco_phase9_helpers import (
    accumulate_actual_foot_and_support_candidates,
    build_phase9_summary,
    write_phase9_summary,
    save_phase9_plots,
    disable_nonfoot_leg_collisions,
)
from phases.mujoco_phase10_helpers import build_ctrl_targets_phase10
from phases.mujoco_visual_helpers import (
    make_free_camera,
    should_capture,
    capture_rgb_frame,
    save_gif,
    save_mp4,
)


def _transform_force(f: np.ndarray, d: mujoco.MjData, base_body_name: str, force_frame: str) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if force_frame == 'world':
        return f
    if force_frame == 'body':
        R = d.body(base_body_name).xmat.reshape(3, 3).copy()
        return R @ f
    raise ValueError(f'Unknown force_frame={force_frame}')


def _create_renderer_with_fallback(m: mujoco.MjModel, width: int, height: int):
    """Create an offscreen renderer, automatically clamping to the model framebuffer.

    MuJoCo offscreen framebuffer defaults to 640x480 unless the XML sets
    <visual><global offwidth=... offheight=...>. When a larger size is requested,
    Renderer raises ValueError. This helper retries with the framebuffer limit.
    """
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


def run_mujoco_phase10_visual(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    clearance: float = 0.05,
    step_len_front: float | None = None,
    rear_step_scale: float = 0.65,
    touchdown_depth_front: float = 0.035,
    touchdown_depth_rear: float = 0.05,
    touchdown_forward_front: float | None = None,
    touchdown_forward_rear: float = 0.004,
    touchdown_search_window_front: float | None = None,
    touchdown_search_window_rear: float = 0.10,
    stance_hold_time: float = 0.0,
    force_frame: str = 'body',
    fx_scale: float = 1.0,
    fy_scale: float = 1.0,
    fz_scale: float = 1.0,
    zero_tangential: bool = False,
    realization: str = 'external',
    disable_nonfoot_collision: bool = False,
    rear_stance_press: float = 0.015,
    rear_stance_back: float = 0.008,
    front_stance_unload: float = 0.004,
    # visual / recording
    save_gif_path: str | None = None,
    save_mp4_path: str | None = None,
    render_width: int = 960,
    render_height: int = 540,
    render_fps: int = 30,
    render_start_time: float = 0.0,
    render_end_time: float | None = None,
    camera_distance: float = 1.8,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -20.0,
    camera_lookat: tuple[float, float, float] | None = None,
    follow_trunk: bool = True,
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
    compute_swing_delta_maps(m, d, bindings)
    initialize_phase5_leg_state(bindings)
    home_ctrl = resolve_home_ctrl(m, d)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.zeros(4, dtype=bool)
    prev_sched = np.zeros(4, dtype=bool)

    if step_len_front is None:
        step_len_front = 0.05
    if touchdown_forward_front is None:
        touchdown_forward_front = max(0.010, 0.40 * step_len_front)
    if touchdown_search_window_front is None:
        touchdown_search_window_front = max(0.04, 0.60 * cfg.stance_time)

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
    }

    frames: list[np.ndarray] = []
    renderer = None
    cam = None
    actual_render_width = int(render_width)
    actual_render_height = int(render_height)
    renderer_resize_note = None
    if save_gif_path or save_mp4_path:
        renderer, actual_render_width, actual_render_height, renderer_resize_note = _create_renderer_with_fallback(
            m, int(render_width), int(render_height)
        )
        if renderer_resize_note is not None:
            print(
                f'Offscreen framebuffer smaller than requested render size; '
                f'using {actual_render_width}x{actual_render_height} instead.'
            )
        cam = make_free_camera(
            m,
            d,
            lookat=camera_lookat,
            distance=camera_distance,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
        )

    def maybe_capture() -> None:
        nonlocal frames
        if renderer is None or cam is None:
            return
        if follow_trunk:
            try:
                cam.lookat[:] = d.body(bindings.base_body_name).xpos.copy()
            except Exception:
                pass
        if should_capture(float(d.time), len(frames), int(render_fps), float(render_start_time), render_end_time):
            frames.append(capture_rgb_frame(renderer, d, cam))

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold, scheduled_contact_hold, prev_sched

        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_pre = actual_contact_state(m, d, bindings)

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg, bindings.base_body_name)
            feet = foot_rel_world(d, bindings.base_body_name, bindings.leg_bindings, FALLBACK_FOOT_LOCAL_OFFSET)

            contact_schedule = rollout_contact_schedule(float(d.time), cfg)
            x_ref = rollout_reference(float(d.time), x, cfg)
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
            x_ref0_hold = x_ref[0].copy()
            scheduled_contact_hold = contact_schedule[0].copy()
            next_mpc_time += cfg.dt_mpc

        update_phase5_state_pre(d, bindings, prev_sched, scheduled_contact_hold)

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase10(
                m=m,
                d=d,
                bindings=bindings,
                home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold,
                actual_contact=actual_pre,
                cfg=cfg,
                clearance=clearance,
                step_len_front=step_len_front,
                rear_step_scale=rear_step_scale,
                touchdown_depth_front=touchdown_depth_front,
                touchdown_depth_rear=touchdown_depth_rear,
                touchdown_forward_front=touchdown_forward_front,
                touchdown_forward_rear=touchdown_forward_rear,
                touchdown_search_window_front=touchdown_search_window_front,
                touchdown_search_window_rear=touchdown_search_window_rear,
                rear_stance_press=rear_stance_press,
                rear_stance_back=rear_stance_back,
                front_stance_unload=front_stance_unload,
            )

        force_enabled = stance_force_enable_mask_phase5(d, bindings, scheduled_contact_hold, actual_pre)
        d.qfrc_applied[:] = 0.0
        u_applied = np.zeros_like(u_hold)
        for leg_i, binding in enumerate(bindings.leg_bindings):
            if not bool(force_enabled[leg_i]):
                continue
            f = u_hold[3 * leg_i : 3 * leg_i + 3].copy()
            if zero_tangential:
                f[0] = 0.0
                f[1] = 0.0
            f[0] *= fx_scale
            f[1] *= fy_scale
            f[2] *= fz_scale
            f_world = _transform_force(f, d, bindings.base_body_name, force_frame)
            u_applied[3 * leg_i : 3 * leg_i + 3] = f_world
            point_world = d.xpos[binding.calf_body_id] + d.xmat[binding.calf_body_id].reshape(3, 3) @ (
                binding.foot_local_offset if getattr(binding, 'foot_local_offset', None) is not None else FALLBACK_FOOT_LOCAL_OFFSET
            )
            if realization == 'external':
                d.qfrc_applied[:] += force_to_qfrc(m, d, binding.calf_body_name, point_world, f_world)
            elif realization == 'joint':
                d.qfrc_applied[:] += leg_internal_force_to_qfrc(m, d, binding, point_world, f_world)
            else:
                raise ValueError(f'Unknown realization={realization}')

        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_post = actual_contact_state(m, d, bindings)
        update_phase5_state_post(d, bindings, scheduled_contact_hold, actual_post, force_enabled, stance_hold_time)
        prev_sched = scheduled_contact_hold.copy()

        x_now = mujoco_to_x(d, cfg, bindings.base_body_name)
        foot_a, foot_b, support_a, support_b = accumulate_actual_foot_and_support_candidates(m, d, bindings)

        log['t'].append(float(d.time))
        log['x'].append(x_now.copy())
        log['u'].append(u_hold.copy())
        log['u_applied'].append(u_applied.copy())
        log['contact'].append(scheduled_contact_hold.copy())
        log['contact_actual'].append(actual_post.copy())
        log['contact_force_enabled'].append(force_enabled.copy())
        log['x_ref0'].append(x_ref0_hold.copy())
        log['actual_grf_a'].append(foot_a.copy())
        log['actual_grf_b'].append(foot_b.copy())
        log['actual_support_a'].append(support_a.copy())
        log['actual_support_b'].append(support_b.copy())

        maybe_capture()
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
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved.extend(plot_logs(log, cfg, output_dir=output_dir))
        saved.extend(save_phase9_plots(log, output_dir=output_dir))
        summary = build_phase9_summary(log, bindings, disable_nonfoot_collision=disable_nonfoot_collision)
        summary['realization'] = realization
        summary['rear_stance_press'] = float(rear_stance_press)
        summary['rear_stance_back'] = float(rear_stance_back)
        summary['front_stance_unload'] = float(front_stance_unload)
        summary['requested_render_size'] = [int(render_width), int(render_height)]
        summary['actual_render_size'] = [int(actual_render_width), int(actual_render_height)]
        summary['renderer_resize_note'] = renderer_resize_note
        summary['requested_render_size'] = [int(render_width), int(render_height)]
        summary['actual_render_size'] = [int(actual_render_width), int(actual_render_height)]
        summary['renderer_resize_note'] = renderer_resize_note
        saved.append(write_phase9_summary(output_dir, summary))
    else:
        summary = build_phase9_summary(log, bindings, disable_nonfoot_collision=disable_nonfoot_collision)
        summary['realization'] = realization
        summary['rear_stance_press'] = float(rear_stance_press)
        summary['rear_stance_back'] = float(rear_stance_back)
        summary['front_stance_unload'] = float(front_stance_unload)
        summary['requested_render_size'] = [int(render_width), int(render_height)]
        summary['actual_render_size'] = [int(actual_render_width), int(actual_render_height)]
        summary['renderer_resize_note'] = renderer_resize_note

    if save_gif_path:
        Path(save_gif_path).parent.mkdir(parents=True, exist_ok=True)
        saved.append(save_gif(frames, save_gif_path, int(render_fps)))
    if save_mp4_path:
        Path(save_mp4_path).parent.mkdir(parents=True, exist_ok=True)
        saved.append(save_mp4(frames, save_mp4_path, int(render_fps)))

    return log, saved, summary


def main() -> int:
    parser = argparse.ArgumentParser(description='MuJoCo phase-10 visual runner: live viewer + GIF/MP4 recording')
    parser.add_argument('--model', required=True)
    parser.add_argument('--scenario', default='straight_trot')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--clearance', type=float, default=0.05)
    parser.add_argument('--step-len-front', type=float, default=0.05)
    parser.add_argument('--rear-step-scale', type=float, default=0.65)
    parser.add_argument('--touchdown-depth-front', type=float, default=0.035)
    parser.add_argument('--touchdown-depth-rear', type=float, default=0.05)
    parser.add_argument('--touchdown-forward-front', type=float, default=None)
    parser.add_argument('--touchdown-forward-rear', type=float, default=0.004)
    parser.add_argument('--touchdown-search-window-front', type=float, default=None)
    parser.add_argument('--touchdown-search-window-rear', type=float, default=0.10)
    parser.add_argument('--stance-hold-time', type=float, default=0.0)
    parser.add_argument('--force-frame', choices=['world', 'body'], default='body')
    parser.add_argument('--fx-scale', type=float, default=1.0)
    parser.add_argument('--fy-scale', type=float, default=1.0)
    parser.add_argument('--fz-scale', type=float, default=1.0)
    parser.add_argument('--zero-tangential', action='store_true')
    parser.add_argument('--realization', choices=['external', 'joint'], default='external')
    parser.add_argument('--disable-nonfoot-collision', action='store_true')
    parser.add_argument('--rear-stance-press', type=float, default=0.015)
    parser.add_argument('--rear-stance-back', type=float, default=0.008)
    parser.add_argument('--front-stance-unload', type=float, default=0.004)
    # visual / recording
    parser.add_argument('--save-gif', default=None)
    parser.add_argument('--save-mp4', default=None)
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--render-height', type=int, default=480)
    parser.add_argument('--render-fps', type=int, default=30)
    parser.add_argument('--render-start-time', type=float, default=0.0)
    parser.add_argument('--render-end-time', type=float, default=None)
    parser.add_argument('--camera-distance', type=float, default=1.8)
    parser.add_argument('--camera-azimuth', type=float, default=135.0)
    parser.add_argument('--camera-elevation', type=float, default=-20.0)
    parser.add_argument('--camera-lookat', nargs=3, type=float, default=None, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--no-follow-trunk', action='store_true')
    args = parser.parse_args()

    cfg = make_config(args.scenario)
    try:
        _, saved, summary = run_mujoco_phase10_visual(
            cfg=cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=args.output_dir,
            clearance=args.clearance,
            step_len_front=args.step_len_front,
            rear_step_scale=args.rear_step_scale,
            touchdown_depth_front=args.touchdown_depth_front,
            touchdown_depth_rear=args.touchdown_depth_rear,
            touchdown_forward_front=args.touchdown_forward_front,
            touchdown_forward_rear=args.touchdown_forward_rear,
            touchdown_search_window_front=args.touchdown_search_window_front,
            touchdown_search_window_rear=args.touchdown_search_window_rear,
            stance_hold_time=args.stance_hold_time,
            force_frame=args.force_frame,
            fx_scale=args.fx_scale,
            fy_scale=args.fy_scale,
            fz_scale=args.fz_scale,
            zero_tangential=args.zero_tangential,
            realization=args.realization,
            disable_nonfoot_collision=args.disable_nonfoot_collision,
            rear_stance_press=args.rear_stance_press,
            rear_stance_back=args.rear_stance_back,
            front_stance_unload=args.front_stance_unload,
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
            camera_lookat=tuple(args.camera_lookat) if args.camera_lookat is not None else None,
            follow_trunk=not args.no_follow_trunk,
        )
        print(f'MuJoCo phase-10 visual run finished for scenario: {args.scenario}')
        print(f'Model: {args.model}')
        if summary is not None:
            print(f"Mean mismatch ratio: {summary.get('mean_mismatch_ratio')}")
            print(f"Mean vx after 1.0 s: {summary.get('mean_vx_after_1s')}")
            print(f"Mean actual FOOT sum Fz after 1.0 s: {summary.get('mean_actual_sum_fz_after_1s')}")
            print(f"Mean actual SUPPORT sum Fz after 1.0 s: {summary.get('mean_actual_support_sum_fz_after_1s')}")
        if saved:
            print('Saved outputs:')
            for p in saved:
                print(f' - {p}')
        return 0
    except Exception:
        print('MuJoCo phase-10 visual run failed. Full traceback below:')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
