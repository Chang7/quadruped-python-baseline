
from __future__ import annotations

import argparse
import json
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

from experiments.low_level_realizer import (
    CleanLowLevelParams,
    CleanLowLevelRealizer,
    actual_foot_contact_state,
    discover_model_bindings,
    disable_nonfoot_leg_collisions,
    foot_rel_world,
    print_binding_summary,
    save_contact_plot,
    state_to_x,
)
from experiments.mujoco_clean_visual import (
    create_renderer_with_fallback,
    make_free_camera,
    should_capture,
    capture_rgb_frame,
    save_gif,
    save_mp4,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean MuJoCo integration runner for the Python quadruped MPC baseline.")
    p.add_argument("--scenario", type=str, default="straight_trot")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs_mujoco_clean")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--schedule", type=str, default="crawl", choices=["crawl", "trot"])
    p.add_argument("--disable-nonfoot-collision", action="store_true")

    p.add_argument("--settle-time", type=float, default=0.6)
    p.add_argument("--gait-ramp-time", type=float, default=1.2)
    p.add_argument("--desired-speed-cap", type=float, default=0.12)
    p.add_argument("--crawl-phase-duration", type=float, default=0.34)
    p.add_argument("--crawl-swing-duration", type=float, default=0.18)

    p.add_argument("--support-enabled", action="store_true")
    p.add_argument("--support-target-height-frac", type=float, default=0.90)
    p.add_argument("--support-weight-start", type=float, default=0.90)
    p.add_argument("--support-weight-end", type=float, default=0.10)
    p.add_argument("--support-fade-start", type=float, default=0.5)
    p.add_argument("--support-fade-end", type=float, default=1.8)

    p.add_argument("--mpc-force-gain-start", type=float, default=0.00)
    p.add_argument("--mpc-force-gain-end", type=float, default=0.28)
    p.add_argument("--mpc-force-ramp-start", type=float, default=0.4)
    p.add_argument("--mpc-force-ramp-end", type=float, default=1.4)
    p.add_argument("--force-frame", type=str, default="body", choices=["body", "world"])
    p.add_argument("--realization", type=str, default="external", choices=["external", "joint"])

    p.add_argument("--clearance", type=float, default=0.070)
    p.add_argument("--step-len-front", type=float, default=0.055)
    p.add_argument("--rear-step-scale", type=float, default=0.90)
    p.add_argument("--touchdown-depth-front", type=float, default=0.020)
    p.add_argument("--touchdown-depth-rear", type=float, default=0.025)
    p.add_argument("--dq-limit", type=float, default=0.16)
    p.add_argument("--visual-step-boost", type=float, default=1.0)

    p.add_argument("--stance-press-front", type=float, default=0.010)
    p.add_argument("--stance-press-rear", type=float, default=0.010)
    p.add_argument("--stance-drive-front", type=float, default=0.003)
    p.add_argument("--stance-drive-rear", type=float, default=0.004)
    p.add_argument("--front-unload", type=float, default=-0.001)
    p.add_argument("--height-k", type=float, default=1.0)
    p.add_argument("--pitch-k", type=float, default=0.04)
    p.add_argument("--roll-k", type=float, default=0.03)
    p.add_argument("--pitch-sign", type=float, default=-1.0)
    p.add_argument("--roll-sign", type=float, default=1.0)

    p.add_argument("--save-gif", type=str, default=None)
    p.add_argument("--save-mp4", type=str, default=None)
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--render-fps", type=int, default=30)
    p.add_argument("--render-start-time", type=float, default=0.0)
    p.add_argument("--render-end-time", type=float, default=None)
    p.add_argument("--camera-distance", type=float, default=1.8)
    p.add_argument("--camera-azimuth", type=float, default=135.0)
    p.add_argument("--camera-elevation", type=float, default=-20.0)
    return p


def run_clean(args):
    cfg = make_config(args.scenario)
    cfg.desired_speed = min(float(cfg.desired_speed), float(args.desired_speed_cap))
    cfg.desired_accel = min(float(cfg.desired_accel), float(args.desired_speed_cap) / max(cfg.dt_mpc, 1e-6))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    bindings = discover_model_bindings(m)
    print(print_binding_summary(bindings))
    if args.disable_nonfoot_collision:
        disabled = disable_nonfoot_leg_collisions(m, bindings)
        print("Disabled non-foot collision geoms:", disabled)
        mujoco.mj_forward(m, d)

    params = CleanLowLevelParams(
        schedule=args.schedule,
        settle_time=args.settle_time,
        gait_ramp_time=args.gait_ramp_time,
        desired_speed_cap=args.desired_speed_cap,
        crawl_phase_duration=args.crawl_phase_duration,
        crawl_swing_duration=args.crawl_swing_duration,
        support_enabled=bool(args.support_enabled),
        support_target_height_frac=args.support_target_height_frac,
        support_weight_start=args.support_weight_start,
        support_weight_end=args.support_weight_end,
        support_fade_start=args.support_fade_start,
        support_fade_end=args.support_fade_end,
        mpc_force_gain_start=args.mpc_force_gain_start,
        mpc_force_gain_end=args.mpc_force_gain_end,
        mpc_force_ramp_start=args.mpc_force_ramp_start,
        mpc_force_ramp_end=args.mpc_force_ramp_end,
        force_frame=args.force_frame,
        realization=args.realization,
        clearance=args.clearance,
        step_len_front=args.step_len_front,
        rear_step_scale=args.rear_step_scale,
        touchdown_depth_front=args.touchdown_depth_front,
        touchdown_depth_rear=args.touchdown_depth_rear,
        dq_limit=args.dq_limit,
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
        visual_step_boost=args.visual_step_boost,
        disable_nonfoot_collision=bool(args.disable_nonfoot_collision),
    )
    realizer = CleanLowLevelRealizer(m, d, bindings, cfg, params)

    controller = MPCControllerOSQP(verbose=False)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    sched_hold = np.ones(4, dtype=bool)
    swing_leg_hold = None
    swing_phase_hold = None

    log = {
        "t": [], "x": [], "u": [], "u_applied": [], "contact": [], "contact_actual": [], "x_ref0": [],
        "support_force_world": [], "support_torque_world": [], "nominal_trunk_height": realizer.nominal_trunk_height,
    }

    renderer = None
    frames = []
    frame_count = 0
    actual_render_size = None
    if args.save_gif is not None or args.save_mp4 is not None:
        renderer, w, h, fallback_msg = create_renderer_with_fallback(m, args.render_width, args.render_height)
        actual_render_size = (w, h)
        if fallback_msg:
            print("Renderer fallback engaged:", fallback_msg.strip())

    def capture_if_needed(frame_count_local: int) -> int:
        if renderer is None:
            return frame_count_local
        if should_capture(d.time, frame_count_local, args.render_fps, args.render_start_time, args.render_end_time):
            cam = make_free_camera(m, d, distance=args.camera_distance, azimuth=args.camera_azimuth, elevation=args.camera_elevation)
            frames.append(capture_rgb_frame(renderer, d, cam))
            return frame_count_local + 1
        return frame_count_local

    def one_step():
        nonlocal next_mpc_time, u_hold, x_ref0_hold, sched_hold, swing_leg_hold, swing_phase_hold, frame_count

        actual_contact = actual_foot_contact_state(m, d, bindings)
        pos_x = state_to_x(m, d, cfg, bindings.base_body_id)

        if d.time >= next_mpc_time - 1e-12:
            x_now = state_to_x(m, d, cfg, bindings.base_body_id)
            sched_now, swing_leg, swing_s, contact_rollout = realizer.schedule_now_and_rollout(float(d.time))
            feet = foot_rel_world(d, bindings)
            x_ref = rollout_reference(max(0.0, float(d.time) - params.settle_time), x_now, cfg)
            try:
                Ad_list, Bd_list = build_prediction_model(x_ref, feet, cfg)
                qp = build_qp(
                    x_init=x_now,
                    x_ref=x_ref,
                    Ad_list=Ad_list,
                    Bd_list=Bd_list,
                    contact_schedule=contact_rollout,
                    cfg=cfg,
                )
                u_hold, _ = controller.solve(qp)
            except Exception as exc:
                print("MPC solve warning:", str(exc))
            x_ref0_hold = x_ref[0].copy()
            sched_hold = sched_now.copy()
            swing_leg_hold = swing_leg
            swing_phase_hold = swing_s
            next_mpc_time += cfg.dt_mpc

        gait_alpha = 0.0 if d.time < params.settle_time else float(np.clip((d.time - params.settle_time) / max(params.gait_ramp_time, 1e-6), 0.0, 1.0))
        realizer.update_phase_state(sched_hold, gait_alpha)

        if m.nu > 0:
            d.ctrl[:] = realizer.build_control_targets(sched_hold, actual_contact, gait_alpha)

        d.qfrc_applied[:] = 0.0
        support_info = realizer.apply_support(target_yaw=float(cfg.desired_yaw))
        mpc_info = realizer.apply_mpc_forces(u_hold, sched_hold, actual_contact)

        mujoco.mj_step(m, d)

        x_post = state_to_x(m, d, cfg, bindings.base_body_id)
        actual_post = actual_foot_contact_state(m, d, bindings)

        log["t"].append(float(d.time))
        log["x"].append(x_post.copy())
        log["u"].append(u_hold.copy())
        log["u_applied"].append(mpc_info["u_applied"].copy())
        log["contact"].append(sched_hold.copy())
        log["contact_actual"].append(actual_post.copy())
        log["x_ref0"].append(x_ref0_hold.copy())
        log["support_force_world"].append(support_info["force_world"].copy())
        log["support_torque_world"].append(support_info["torque_world"].copy())

        frame_count = capture_if_needed(frame_count)

    if args.headless:
        while d.time < cfg.sim_time:
            one_step()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < cfg.sim_time:
                one_step()
                viewer.sync()

    # Save logs / plots
    plot_saved = plot_logs(log, cfg, str(output_dir))
    contact_plot = save_contact_plot(np.asarray(log["t"]), np.asarray(log["contact"]), np.asarray(log["contact_actual"]), str(output_dir / "fig_contact_schedule_vs_actual.png"))
    plot_saved.append(contact_plot)

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    support_force = np.asarray(log["support_force_world"], dtype=float)
    mask = t >= max(1.0, float(args.settle_time))
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    z = x[:, 2]
    summary = {
        "schedule": args.schedule,
        "mean_vx_after_1s": float(np.mean(x[mask, 3])),
        "mean_trunk_height_after_1s": float(np.mean(z[mask])),
        "min_trunk_height_after_1s": float(np.min(z[mask])),
        "collapse_time": None if not np.any(z < 0.22) else float(t[np.where(z < 0.22)[0][0]]),
        "mean_support_force_world_after_1s": support_force[mask].mean(axis=0).tolist() if support_force.size else [0.0, 0.0, 0.0],
        "mean_u_applied_after_1s": np.asarray(log["u_applied"])[mask].mean(axis=0).tolist() if log["u_applied"] else [0.0] * cfg.nu,
        "mean_actual_contact_ratio": np.asarray(log["contact_actual"], dtype=float)[mask].mean(axis=0).tolist(),
        "actual_render_size": None if actual_render_size is None else list(actual_render_size),
    }
    summary_path = output_dir / "clean_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    media_saved = []
    if args.save_gif is not None:
        Path(args.save_gif).parent.mkdir(parents=True, exist_ok=True)
        media_saved.append(save_gif(frames, args.save_gif, args.render_fps))
    if args.save_mp4 is not None:
        Path(args.save_mp4).parent.mkdir(parents=True, exist_ok=True)
        media_saved.append(save_mp4(frames, args.save_mp4, args.render_fps))

    print(f"MuJoCo clean run finished for scenario: {args.scenario}")
    print(f"Model: {args.model}")
    print("Mean vx after 1.0 s:", summary["mean_vx_after_1s"])
    print("Mean trunk height after 1.0 s:", summary["mean_trunk_height_after_1s"])
    print("Min trunk height after 1.0 s:", summary["min_trunk_height_after_1s"])
    print("Collapse time:", summary["collapse_time"])
    print("Saved outputs:")
    for p in plot_saved:
        print(" -", p)
    print(" -", str(summary_path))
    for p in media_saved:
        print(" -", p)
    return log, plot_saved, summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_clean(args)
    except Exception:
        print("MuJoCo clean run failed. Full traceback below:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
