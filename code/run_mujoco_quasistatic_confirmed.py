
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

from experiments.low_level_realizer import (
    discover_model_bindings,
    print_binding_summary,
    disable_nonfoot_leg_collisions,
    state_to_x,
    actual_foot_contact_state,
    foot_rel_world,
    body_pose_velocity,
    save_contact_plot,
)
from experiments.quasistatic_confirmed_helper import QuasiStaticParams, ConfirmedQuasiStaticCrawl
from experiments.quasistatic_visual_helpers import (
    make_free_camera,
    create_renderer_with_fallback,
    should_capture,
    capture_rgb_frame,
    save_gif,
    save_mp4,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Confirmed quasi-static MuJoCo crawl using the Python MPC core.")
    p.add_argument("--scenario", type=str, default="straight_trot")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs_mujoco_quasistatic")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--disable-nonfoot-collision", action="store_true")

    # timing / gait
    p.add_argument("--startup-time", type=float, default=0.45)
    p.add_argument("--shift-time", type=float, default=0.22)
    p.add_argument("--swing-time", type=float, default=0.22)
    p.add_argument("--touchdown-wait", type=float, default=0.16)
    p.add_argument("--hold-time", type=float, default=0.10)
    p.add_argument("--desired-speed-cap", type=float, default=0.10)

    # support
    p.add_argument("--support-enabled", action="store_true")
    p.add_argument("--support-weight-start", type=float, default=0.95)
    p.add_argument("--support-weight-end", type=float, default=0.55)
    p.add_argument("--support-fade-start", type=float, default=0.4)
    p.add_argument("--support-fade-end", type=float, default=2.2)
    p.add_argument("--target-height-frac", type=float, default=0.98)
    p.add_argument("--support-xy-k", type=float, default=220.0)
    p.add_argument("--support-xy-d", type=float, default=36.0)
    p.add_argument("--support-z-k", type=float, default=520.0)
    p.add_argument("--support-z-d", type=float, default=60.0)
    p.add_argument("--support-roll-k", type=float, default=26.0)
    p.add_argument("--support-roll-d", type=float, default=3.2)
    p.add_argument("--support-pitch-k", type=float, default=22.0)
    p.add_argument("--support-pitch-d", type=float, default=2.8)
    p.add_argument("--support-yaw-k", type=float, default=5.0)
    p.add_argument("--support-yaw-d", type=float, default=1.0)
    p.add_argument("--support-max-force-xy", type=float, default=70.0)
    p.add_argument("--support-max-force-z", type=float, default=220.0)
    p.add_argument("--support-max-torque", type=float, default=20.0)
    p.add_argument("--target-pitch", type=float, default=0.0)
    p.add_argument("--target-roll", type=float, default=0.0)
    p.add_argument("--target-yaw", type=float, default=0.0)

    # MPC realization
    p.add_argument("--force-frame", type=str, default="body", choices=["body", "world"])
    p.add_argument("--realization", type=str, default="external", choices=["external", "joint"])
    p.add_argument("--mpc-force-gain-start", type=float, default=0.00)
    p.add_argument("--mpc-force-gain-end", type=float, default=0.22)
    p.add_argument("--mpc-force-ramp-start", type=float, default=0.8)
    p.add_argument("--mpc-force-ramp-end", type=float, default=2.4)
    p.add_argument("--max-xy-over-fz", type=float, default=0.35)

    # foot motion
    p.add_argument("--clearance", type=float, default=0.085)
    p.add_argument("--step-len-front", type=float, default=0.065)
    p.add_argument("--rear-step-scale", type=float, default=0.90)
    p.add_argument("--touchdown-depth-front", type=float, default=0.015)
    p.add_argument("--touchdown-depth-rear", type=float, default=0.020)
    p.add_argument("--touchdown-extra", type=float, default=0.010)
    p.add_argument("--dq-limit", type=float, default=0.18)
    p.add_argument("--visual-step-boost", type=float, default=1.0)

    # stance shaping
    p.add_argument("--stance-press-front", type=float, default=0.008)
    p.add_argument("--stance-press-rear", type=float, default=0.010)
    p.add_argument("--stance-press-gain", type=float, default=0.00045)
    p.add_argument("--stance-fx-bias-gain", type=float, default=0.0008)
    p.add_argument("--stance-fy-bias-gain", type=float, default=0.0006)
    p.add_argument("--shift-x-mag", type=float, default=0.020)
    p.add_argument("--shift-y-mag", type=float, default=0.020)
    p.add_argument("--touchdown-confirm-hold", type=float, default=0.06)

    # recovery
    p.add_argument("--recovery-height-enter-frac", type=float, default=0.78)
    p.add_argument("--recovery-height-exit-frac", type=float, default=0.90)
    p.add_argument("--recovery-pitch-enter", type=float, default=0.28)
    p.add_argument("--recovery-roll-enter", type=float, default=0.22)
    p.add_argument("--recovery-pitch-exit", type=float, default=0.12)
    p.add_argument("--recovery-roll-exit", type=float, default=0.10)
    p.add_argument("--recovery-min-time", type=float, default=0.18)

    # visual
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


def simple_plots(log: dict, output_dir: Path):
    import matplotlib.pyplot as plt

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    x_ref0 = np.asarray(log["x_ref0"], dtype=float)
    sched = np.asarray(log["contact"], dtype=bool)
    actual = np.asarray(log["contact_actual"], dtype=bool)
    support = np.asarray(log["support_force_world"], dtype=float)
    u_applied = np.asarray(log["u_applied"], dtype=float)
    phases = log["phase"]
    phase_to_y = {"STARTUP": 0, "SHIFT": 1, "SWING": 2, "TOUCHDOWN": 3, "HOLD": 4, "RECOVERY": 5}
    phase_y = np.array([phase_to_y.get(p, -1) for p in phases], dtype=float)

    # velocity tracking
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t, x[:, 3], label="vx actual")
    ax.plot(t, x_ref0[:, 3], "--", label="vx ref")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("vx [m/s]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_velocity_tracking.png", dpi=200)
    plt.close(fig)

    # trunk height + phase
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t, x[:, 2], label="trunk height")
    ax.plot(t, phase_y * 0.01 + x[:, 2].min(), alpha=0.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, phase_y, "k--", alpha=0.5, label="phase")
    ax2.set_yticks(list(phase_to_y.values()), list(phase_to_y.keys()))
    fig.tight_layout()
    fig.savefig(output_dir / "fig_trunk_height_phase.png", dpi=200)
    plt.close(fig)

    # support force z
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t, support[:, 2], label="support Fz")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("support Fz [N]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_support_force_z.png", dpi=200)
    plt.close(fig)

    # xy path
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot(x[:, 0], x[:, 1], label="base path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_xy_path.png", dpi=200)
    plt.close(fig)

    # commanded Fz per leg
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 6.0), sharex=True)
    names = ["FL", "FR", "RL", "RR"]
    for i, ax in enumerate(axes):
        ax.plot(t, u_applied[:, 3*i+2], label=f"{names[i]} applied Fz")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_leg_fz_subplots.png", dpi=200)
    plt.close(fig)

    save_contact_plot(t, sched, actual, output_dir / "fig_contact_schedule_vs_actual.png")


def run_quasistatic(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(args.scenario)
    cfg.desired_speed = min(float(cfg.desired_speed), float(args.desired_speed_cap))
    cfg.desired_accel = min(float(cfg.desired_accel), float(args.desired_speed_cap) / max(cfg.dt_mpc, 1e-6))

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    bindings = discover_model_bindings(m)
    print(print_binding_summary(bindings))
    if args.disable_nonfoot_collision:
        disabled = disable_nonfoot_leg_collisions(m, bindings)
        print("Disabled non-foot collision geoms:", disabled)
        mujoco.mj_forward(m, d)

    params = QuasiStaticParams(
        startup_time=args.startup_time,
        shift_time=args.shift_time,
        swing_time=args.swing_time,
        touchdown_wait=args.touchdown_wait,
        hold_time=args.hold_time,
        desired_speed_cap=args.desired_speed_cap,
        support_enabled=bool(args.support_enabled),
        support_weight_start=args.support_weight_start,
        support_weight_end=args.support_weight_end,
        support_fade_start=args.support_fade_start,
        support_fade_end=args.support_fade_end,
        target_height_frac=args.target_height_frac,
        target_pitch=args.target_pitch,
        target_roll=args.target_roll,
        target_yaw=args.target_yaw,
        support_xy_k=args.support_xy_k,
        support_xy_d=args.support_xy_d,
        support_z_k=args.support_z_k,
        support_z_d=args.support_z_d,
        support_roll_k=args.support_roll_k,
        support_roll_d=args.support_roll_d,
        support_pitch_k=args.support_pitch_k,
        support_pitch_d=args.support_pitch_d,
        support_yaw_k=args.support_yaw_k,
        support_yaw_d=args.support_yaw_d,
        support_max_force_xy=args.support_max_force_xy,
        support_max_force_z=args.support_max_force_z,
        support_max_torque=args.support_max_torque,
        force_frame=args.force_frame,
        realization=args.realization,
        mpc_force_gain_start=args.mpc_force_gain_start,
        mpc_force_gain_end=args.mpc_force_gain_end,
        mpc_force_ramp_start=args.mpc_force_ramp_start,
        mpc_force_ramp_end=args.mpc_force_ramp_end,
        max_xy_over_fz=args.max_xy_over_fz,
        clearance=args.clearance,
        step_len_front=args.step_len_front,
        rear_step_scale=args.rear_step_scale,
        touchdown_depth_front=args.touchdown_depth_front,
        touchdown_depth_rear=args.touchdown_depth_rear,
        touchdown_extra=args.touchdown_extra,
        dq_limit=args.dq_limit,
        stance_press_front=args.stance_press_front,
        stance_press_rear=args.stance_press_rear,
        stance_press_gain=args.stance_press_gain,
        stance_fx_bias_gain=args.stance_fx_bias_gain,
        stance_fy_bias_gain=args.stance_fy_bias_gain,
        shift_x_mag=args.shift_x_mag,
        shift_y_mag=args.shift_y_mag,
        touchdown_confirm_hold=args.touchdown_confirm_hold,
        recovery_height_enter_frac=args.recovery_height_enter_frac,
        recovery_height_exit_frac=args.recovery_height_exit_frac,
        recovery_pitch_enter=args.recovery_pitch_enter,
        recovery_roll_enter=args.recovery_roll_enter,
        recovery_pitch_exit=args.recovery_pitch_exit,
        recovery_roll_exit=args.recovery_roll_exit,
        recovery_min_time=args.recovery_min_time,
        visual_step_boost=args.visual_step_boost,
    )
    planner = ConfirmedQuasiStaticCrawl(m, d, bindings, cfg, params)
    controller = MPCControllerOSQP(verbose=False)

    renderer = None
    frames = []
    frame_count = 0
    actual_render_size = None
    if args.save_gif is not None or args.save_mp4 is not None:
        renderer, w, h, fallback_msg = create_renderer_with_fallback(m, args.render_width, args.render_height)
        actual_render_size = (w, h)
        if fallback_msg:
            print("Renderer fallback engaged:", fallback_msg.strip())

    log = {
        "t": [], "x": [], "u": [], "u_applied": [], "contact": [], "contact_actual": [], "x_ref0": [],
        "support_force_world": [], "phase": [], "health": [], "swing_leg": []
    }
    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)

    def capture_if_needed(frame_count_local: int) -> int:
        if renderer is None:
            return frame_count_local
        if should_capture(d.time, frame_count_local, args.render_fps, args.render_start_time, args.render_end_time):
            cam = make_free_camera(m, d, distance=args.camera_distance, azimuth=args.camera_azimuth, elevation=args.camera_elevation)
            frames.append(capture_rgb_frame(renderer, d, cam))
            return frame_count_local + 1
        return frame_count_local

    def mpc_step():
        nonlocal next_mpc_time, u_hold, x_ref0_hold
        x_now = state_to_x(m, d, cfg, bindings.base_body_id)
        x_ref = rollout_reference(max(0.0, float(d.time) - params.startup_time), x_now, cfg)
        feet = foot_rel_world(d, bindings)
        sched_rollout = planner.rollout_sched()
        Ad_list, Bd_list = build_prediction_model(x_ref, feet, cfg)
        qp = build_qp(
            x_init=x_now,
            x_ref=x_ref,
            Ad_list=Ad_list,
            Bd_list=Bd_list,
            contact_schedule=sched_rollout,
            cfg=cfg,
        )
        try:
            u_hold, _ = controller.solve(qp)
        except Exception as exc:
            print("MPC solve warning:", str(exc))
        x_ref0_hold = x_ref[0].copy()
        next_mpc_time += cfg.dt_mpc

    def step_once():
        nonlocal frame_count
        actual = actual_foot_contact_state(m, d, bindings)

        planner.step_state_machine(float(d.time), actual)

        if d.time >= next_mpc_time - 1e-12:
            mpc_step()

        if m.nu > 0:
            d.ctrl[:] = planner.build_ctrl_targets(float(d.time), u_hold, actual)

        d.qfrc_applied[:] = 0.0
        support_info = planner.apply_support(float(d.time))
        mpc_info = planner.apply_mpc_forces(float(d.time), u_hold, actual)

        mujoco.mj_step(m, d)

        x_post = state_to_x(m, d, cfg, bindings.base_body_id)
        actual_post = actual_foot_contact_state(m, d, bindings)

        log["t"].append(float(d.time))
        log["x"].append(x_post.copy())
        log["u"].append(u_hold.copy())
        log["u_applied"].append(mpc_info["u_applied"].copy())
        log["contact"].append(planner.current_sched().copy())
        log["contact_actual"].append(actual_post.copy())
        log["x_ref0"].append(x_ref0_hold.copy())
        log["support_force_world"].append(support_info["force_world"].copy())
        log["phase"].append(planner.phase)
        log["health"].append(float(planner.summary_health()))
        log["swing_leg"].append(-1 if planner.swing_leg is None else int(planner.swing_leg))

        frame_count = capture_if_needed(frame_count)

    if args.headless:
        while d.time < cfg.sim_time:
            step_once()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < cfg.sim_time:
                step_once()
                viewer.sync()

    # Save visuals
    if args.save_gif is not None and frames:
        save_gif(frames, args.save_gif, fps=args.render_fps)
    if args.save_mp4 is not None and frames:
        save_mp4(frames, args.save_mp4, fps=args.render_fps)

    simple_plots(log, output_dir)

    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    support = np.asarray(log["support_force_world"], dtype=float)
    actual = np.asarray(log["contact_actual"], dtype=bool)

    after = t > 1.0
    collapse_mask = x[:, 2] < 0.55 * planner.nominal_height
    collapse_time = None if not np.any(collapse_mask) else float(t[np.argmax(collapse_mask)])

    summary = {
        "mean_vx_after_1s": float(x[after, 3].mean()) if np.any(after) else float(x[:, 3].mean()),
        "mean_trunk_height_after_1s": float(x[after, 2].mean()) if np.any(after) else float(x[:, 2].mean()),
        "min_trunk_height_after_1s": float(x[after, 2].min()) if np.any(after) else float(x[:, 2].min()),
        "mean_abs_pitch_after_1s": float(np.abs(x[after, 7]).mean()) if np.any(after) else float(np.abs(x[:, 7]).mean()),
        "mean_abs_roll_after_1s": float(np.abs(x[after, 6]).mean()) if np.any(after) else float(np.abs(x[:, 6]).mean()),
        "collapse_time": collapse_time,
        "mean_support_force_world_after_1s": support[after].mean(axis=0).tolist() if np.any(after) else support.mean(axis=0).tolist(),
        "mean_actual_contact_ratio": actual.astype(float).mean(axis=0).tolist(),
        "mode_counts": {ph: int(sum(1 for p in log["phase"] if p == ph)) for ph in sorted(set(log["phase"]))},
        "recovery_count": int(planner.recovery_count),
        "mean_health_after_1s": float(np.asarray(log["health"], dtype=float)[after].mean()) if np.any(after) else float(np.mean(log["health"])),
        "actual_render_size": actual_render_size,
        "params": vars(args),
    }
    with open(output_dir / "quasistatic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_quasistatic(args)
        print("Confirmed quasi-static crawl run finished.")
        print("Model:", args.model)
        print("Mean vx after 1.0 s:", summary["mean_vx_after_1s"])
        print("Mean trunk height after 1.0 s:", summary["mean_trunk_height_after_1s"])
        print("Collapse time:", summary["collapse_time"])
        print("Recovery count:", summary["recovery_count"])
        print("Outputs saved to:", args.output_dir)
    except Exception:
        print("Confirmed quasi-static crawl run failed. Full traceback below:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
