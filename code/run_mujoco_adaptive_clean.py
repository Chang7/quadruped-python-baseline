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
    actual_foot_contact_state,
    body_pose_velocity,
    disable_nonfoot_leg_collisions,
    discover_model_bindings,
    foot_rel_world,
    print_binding_summary,
    save_contact_plot,
    state_to_x,
)
from experiments.adaptive_clean_helper import (
    AdaptiveCleanRealizer,
    AdaptiveSupervisor,
    AdaptiveSupervisorParams,
    write_adaptive_summary,
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
    p = argparse.ArgumentParser(description="Adaptive clean MuJoCo integration runner for the Python quadruped MPC baseline.")
    p.add_argument("--scenario", type=str, default="straight_trot")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs_mujoco_adaptive_clean")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--schedule", type=str, default="crawl", choices=["crawl", "trot"])
    p.add_argument("--disable-nonfoot-collision", action="store_true")

    # low-level / clean-realizer params
    p.add_argument("--settle-time", type=float, default=0.45)
    p.add_argument("--gait-ramp-time", type=float, default=1.0)
    p.add_argument("--desired-speed-cap", type=float, default=0.16)
    p.add_argument("--crawl-phase-duration", type=float, default=0.34)
    p.add_argument("--crawl-swing-duration", type=float, default=0.18)

    p.add_argument("--support-enabled", action="store_true")
    p.add_argument("--support-target-height-frac", type=float, default=0.88)
    p.add_argument("--support-weight-start", type=float, default=0.88)
    p.add_argument("--support-weight-end", type=float, default=0.12)
    p.add_argument("--support-fade-start", type=float, default=0.45)
    p.add_argument("--support-fade-end", type=float, default=1.8)
    p.add_argument("--mpc-force-gain-start", type=float, default=0.00)
    p.add_argument("--mpc-force-gain-end", type=float, default=0.32)
    p.add_argument("--mpc-force-ramp-start", type=float, default=0.40)
    p.add_argument("--mpc-force-ramp-end", type=float, default=1.40)
    p.add_argument("--force-frame", type=str, default="body", choices=["body", "world"])
    p.add_argument("--realization", type=str, default="external", choices=["external", "joint"])

    p.add_argument("--clearance", type=float, default=0.075)
    p.add_argument("--step-len-front", type=float, default=0.065)
    p.add_argument("--rear-step-scale", type=float, default=0.95)
    p.add_argument("--touchdown-depth-front", type=float, default=0.018)
    p.add_argument("--touchdown-depth-rear", type=float, default=0.022)
    p.add_argument("--dq-limit", type=float, default=0.18)
    p.add_argument("--visual-step-boost", type=float, default=1.0)

    p.add_argument("--stance-press-front", type=float, default=0.011)
    p.add_argument("--stance-press-rear", type=float, default=0.010)
    p.add_argument("--stance-drive-front", type=float, default=0.003)
    p.add_argument("--stance-drive-rear", type=float, default=0.004)
    p.add_argument("--front-unload", type=float, default=-0.001)
    p.add_argument("--height-k", type=float, default=1.0)
    p.add_argument("--pitch-k", type=float, default=0.04)
    p.add_argument("--roll-k", type=float, default=0.03)
    p.add_argument("--pitch-sign", type=float, default=-1.0)
    p.add_argument("--roll-sign", type=float, default=1.0)

    # adaptive supervisor params
    p.add_argument("--startup-time", type=float, default=0.45)
    p.add_argument("--recovery-height-enter-frac", type=float, default=0.72)
    p.add_argument("--recovery-height-exit-frac", type=float, default=0.82)
    p.add_argument("--recovery-pitch-enter", type=float, default=0.38)
    p.add_argument("--recovery-pitch-exit", type=float, default=0.18)
    p.add_argument("--recovery-roll-enter", type=float, default=0.28)
    p.add_argument("--recovery-roll-exit", type=float, default=0.15)
    p.add_argument("--recovery-min-time", type=float, default=0.28)
    p.add_argument("--stable-hold-time", type=float, default=0.20)
    p.add_argument("--health-height-low-frac", type=float, default=0.74)
    p.add_argument("--health-height-high-frac", type=float, default=0.92)
    p.add_argument("--health-pitch-bad", type=float, default=0.30)
    p.add_argument("--health-roll-bad", type=float, default=0.22)
    p.add_argument("--speed-scale-min", type=float, default=0.28)
    p.add_argument("--force-scale-min", type=float, default=0.20)
    p.add_argument("--step-scale-min", type=float, default=0.35)
    p.add_argument("--drive-scale-min", type=float, default=0.20)
    p.add_argument("--support-bonus-walk", type=float, default=0.18)
    p.add_argument("--support-bonus-recovery", type=float, default=0.70)
    p.add_argument("--gait-clock-min-rate", type=float, default=0.20)

    # media
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


def run_adaptive(args):
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

    base_params = CleanLowLevelParams(
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
    sup_params = AdaptiveSupervisorParams(
        startup_time=args.startup_time,
        recovery_height_enter_frac=args.recovery_height_enter_frac,
        recovery_height_exit_frac=args.recovery_height_exit_frac,
        recovery_pitch_enter=args.recovery_pitch_enter,
        recovery_pitch_exit=args.recovery_pitch_exit,
        recovery_roll_enter=args.recovery_roll_enter,
        recovery_roll_exit=args.recovery_roll_exit,
        recovery_min_time=args.recovery_min_time,
        stable_hold_time=args.stable_hold_time,
        health_height_low_frac=args.health_height_low_frac,
        health_height_high_frac=args.health_height_high_frac,
        health_pitch_bad=args.health_pitch_bad,
        health_roll_bad=args.health_roll_bad,
        speed_scale_min=args.speed_scale_min,
        force_scale_min=args.force_scale_min,
        step_scale_min=args.step_scale_min,
        drive_scale_min=args.drive_scale_min,
        support_bonus_walk=args.support_bonus_walk,
        support_bonus_recovery=args.support_bonus_recovery,
        gait_clock_min_rate=args.gait_clock_min_rate,
    )
    supervisor = AdaptiveSupervisor(float(d.body(bindings.base_body_name).xpos[2]), sup_params)
    realizer = AdaptiveCleanRealizer(m, d, bindings, cfg, base_params, supervisor)

    controller = MPCControllerOSQP(verbose=False)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    sched_hold = np.ones(4, dtype=bool)

    log = {
        "t": [], "x": [], "u": [], "u_applied": [], "contact": [], "contact_actual": [], "x_ref0": [],
        "support_force_world": [], "support_torque_world": [], "nominal_trunk_height": realizer.nominal_trunk_height,
        "mode": [], "health": [], "speed_scale": [], "force_scale": [], "step_scale": [], "drive_scale": [],
        "support_bonus": [], "front_rescue_bias": [], "gait_clock": [], "pitch": [], "roll": [], "z": [],
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
        nonlocal next_mpc_time, u_hold, x_ref0_hold, sched_hold, frame_count

        actual_contact = actual_foot_contact_state(m, d, bindings)
        x_now = state_to_x(m, d, cfg, bindings.base_body_id)
        roll, pitch, _ = x_now[6:9]
        supervisor.update(float(d.time), float(x_now[2]), float(roll), float(pitch), actual_contact)
        if supervisor.entered_recovery or supervisor.just_exited_recovery:
            realizer.reanchor_all_legs()
            next_mpc_time = min(next_mpc_time, float(d.time))

        if float(d.time) >= next_mpc_time - 1e-12:
            sched_now, _, _, contact_rollout = realizer.schedule_now_and_rollout(float(d.time))
            feet = foot_rel_world(d, bindings)
            x_ref = rollout_reference(max(0.0, float(d.time) - base_params.settle_time), x_now, cfg)
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
            next_mpc_time += cfg.dt_mpc

        gait_alpha = float(np.clip(supervisor.gait_clock / max(base_params.gait_ramp_time, 1e-6), 0.0, 1.0))
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
        sdict = supervisor.summary_dict()
        log["mode"].append(sdict["mode"])
        log["health"].append(float(sdict["health"]))
        log["speed_scale"].append(float(sdict["speed_scale"]))
        log["force_scale"].append(float(sdict["force_scale"]))
        log["step_scale"].append(float(sdict["step_scale"]))
        log["drive_scale"].append(float(sdict["drive_scale"]))
        log["support_bonus"].append(float(sdict["support_bonus"]))
        log["front_rescue_bias"].append(float(sdict["front_rescue_bias"]))
        log["gait_clock"].append(float(sdict["gait_clock"]))
        log["pitch"].append(float(x_post[7]))
        log["roll"].append(float(x_post[6]))
        log["z"].append(float(x_post[2]))

        return capture_if_needed

    if args.headless:
        while d.time < cfg.sim_time:
            frame_count = one_step()(frame_count)
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < cfg.sim_time:
                frame_count = one_step()(frame_count)
                viewer.sync()

    plot_saved = plot_logs(log, cfg, str(output_dir))
    contact_plot = save_contact_plot(
        np.asarray(log["t"]), np.asarray(log["contact"]), np.asarray(log["contact_actual"]),
        str(output_dir / "fig_contact_schedule_vs_actual.png")
    )
    plot_saved.append(contact_plot)

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    support_force = np.asarray(log["support_force_world"], dtype=float)
    after_mask = t >= max(1.0, float(args.startup_time))
    if not np.any(after_mask):
        after_mask = np.ones_like(t, dtype=bool)

    z = x[:, 2]
    pitch = x[:, 7]
    roll = x[:, 6]
    summary = {
        "mean_vx_after_1s": float(np.mean(x[after_mask, 3])),
        "mean_trunk_height_after_1s": float(np.mean(z[after_mask])),
        "min_trunk_height_after_1s": float(np.min(z[after_mask])),
        "mean_abs_pitch_after_1s": float(np.mean(np.abs(pitch[after_mask]))),
        "mean_abs_roll_after_1s": float(np.mean(np.abs(roll[after_mask]))),
        "collapse_time": None if not np.any(z < 0.22) else float(t[np.where(z < 0.22)[0][0]]),
        "mean_support_force_world_after_1s": support_force[after_mask].mean(axis=0).tolist() if support_force.size else [0.0, 0.0, 0.0],
        "mean_u_applied_after_1s": np.asarray(log["u_applied"])[after_mask].mean(axis=0).tolist() if log["u_applied"] else [0.0] * cfg.nu,
        "mean_actual_contact_ratio": np.asarray(log["contact_actual"], dtype=float)[after_mask].mean(axis=0).tolist(),
        "mode_counts": {m: int(sum(1 for v in log["mode"] if v == m)) for m in ["STARTUP", "WALK", "RECOVERY"]},
        "recovery_count": supervisor.recovery_count,
        "mean_health_after_1s": float(np.mean(np.asarray(log["health"], dtype=float)[after_mask])),
        "mean_speed_scale_after_1s": float(np.mean(np.asarray(log["speed_scale"], dtype=float)[after_mask])),
        "mean_force_scale_after_1s": float(np.mean(np.asarray(log["force_scale"], dtype=float)[after_mask])),
        "actual_render_size": None if actual_render_size is None else list(actual_render_size),
        "adaptive_params": vars(sup_params),
        "clean_params_subset": {
            "schedule": args.schedule,
            "settle_time": args.settle_time,
            "gait_ramp_time": args.gait_ramp_time,
            "desired_speed_cap": args.desired_speed_cap,
            "clearance": args.clearance,
            "step_len_front": args.step_len_front,
            "rear_step_scale": args.rear_step_scale,
            "dq_limit": args.dq_limit,
            "support_enabled": bool(args.support_enabled),
            "force_frame": args.force_frame,
            "realization": args.realization,
        },
    }
    summary_path = write_adaptive_summary(output_dir, summary)

    media_saved = []
    if args.save_gif is not None:
        Path(args.save_gif).parent.mkdir(parents=True, exist_ok=True)
        media_saved.append(save_gif(frames, args.save_gif, args.render_fps))
    if args.save_mp4 is not None:
        Path(args.save_mp4).parent.mkdir(parents=True, exist_ok=True)
        media_saved.append(save_mp4(frames, args.save_mp4, args.render_fps))

    print(f"Adaptive clean MuJoCo run finished for scenario: {args.scenario}")
    print(f"Model: {args.model}")
    print("Mean vx after 1.0 s:", summary["mean_vx_after_1s"])
    print("Mean trunk height after 1.0 s:", summary["mean_trunk_height_after_1s"])
    print("Min trunk height after 1.0 s:", summary["min_trunk_height_after_1s"])
    print("Mean abs pitch after 1.0 s:", summary["mean_abs_pitch_after_1s"])
    print("Mean abs roll after 1.0 s:", summary["mean_abs_roll_after_1s"])
    print("Collapse time:", summary["collapse_time"])
    print("Recovery count:", summary["recovery_count"])
    print("Saved outputs:")
    for p in plot_saved:
        print(" -", p)
    print(" -", summary_path)
    for p in media_saved:
        print(" -", p)
    return log, plot_saved, summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_adaptive(args)
    except Exception:
        print("Adaptive clean MuJoCo run failed. Full traceback below:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
