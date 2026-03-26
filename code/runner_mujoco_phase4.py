
from __future__ import annotations

import argparse
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
from phases.mujoco_phase4_helpers import (
    FALLBACK_FOOT_LOCAL_OFFSET,
    discover_model_bindings,
    foot_point_world,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    compute_swing_delta_maps,
    resolve_home_ctrl,
    initialize_phase4_leg_state,
    update_phase4_state_pre,
    update_phase4_state_post,
    build_ctrl_targets_phase4,
    stance_force_enable_mask,
    build_phase4_summary,
    write_phase4_summary,
    print_binding_summary,
)


def run_mujoco_phase4(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    clearance: float = 0.05,
    step_len: float | None = None,
    touchdown_depth: float = 0.03,
    touchdown_forward: float | None = None,
    touchdown_search_window: float | None = None,
) -> tuple[dict, list[str], dict | None]:
    controller = MPCControllerOSQP(verbose=False)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)

    bindings = discover_model_bindings(m)
    print(print_binding_summary(bindings))

    home_ctrl = resolve_home_ctrl(m, d)

    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    store_home_joint_qpos(d, bindings)
    compute_swing_delta_maps(m, d, bindings)
    initialize_phase4_leg_state(bindings)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.zeros(4, dtype=bool)
    prev_sched = np.ones(4, dtype=bool)

    if step_len is None:
        step_len = max(0.03, min(0.10, 0.55 * cfg.desired_speed * cfg.swing_time))
    if touchdown_forward is None:
        touchdown_forward = max(0.01, min(0.04, 0.35 * step_len))
    if touchdown_search_window is None:
        touchdown_search_window = max(0.04, 0.60 * cfg.stance_time)

    log = {
        "t": [],
        "x": [],
        "u": [],
        "contact": [],
        "contact_actual": [],
        "contact_force_enabled": [],
        "x_ref0": [],
    }

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

        update_phase4_state_pre(d, bindings, prev_sched, scheduled_contact_hold)

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase4(
                m=m,
                d=d,
                bindings=bindings,
                home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold,
                actual_contact=actual_pre,
                cfg=cfg,
                clearance=clearance,
                step_len=step_len,
                touchdown_depth=touchdown_depth,
                touchdown_forward=touchdown_forward,
                touchdown_search_window=touchdown_search_window,
            )

        force_enabled = stance_force_enable_mask(scheduled_contact_hold, actual_pre)
        d.qfrc_applied[:] = 0.0
        for leg_i, binding in enumerate(bindings.leg_bindings):
            if not bool(force_enabled[leg_i]):
                continue
            f_world = u_hold[3 * leg_i : 3 * leg_i + 3]
            point_world = foot_point_world(d, binding, FALLBACK_FOOT_LOCAL_OFFSET)
            d.qfrc_applied[:] += force_to_qfrc(m, d, binding.calf_body_name, point_world, f_world)

        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        actual_post = actual_contact_state(m, d, bindings)
        update_phase4_state_post(d, bindings, scheduled_contact_hold, actual_post, force_enabled)
        prev_sched = scheduled_contact_hold.copy()

        x_now = mujoco_to_x(d, cfg, bindings.base_body_name)

        log["t"].append(float(d.time))
        log["x"].append(x_now.copy())
        log["u"].append(u_hold.copy())
        log["contact"].append(scheduled_contact_hold.copy())
        log["contact_actual"].append(actual_post.copy())
        log["contact_force_enabled"].append(force_enabled.copy())
        log["x_ref0"].append(x_ref0_hold.copy())

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
        saved = plot_logs(log, cfg, output_dir=output_dir)
        summary = build_phase4_summary(log, bindings)
        saved.append(write_phase4_summary(output_dir, summary))
    else:
        summary = build_phase4_summary(log, bindings)

    if summary is not None:
        print("Mean mismatch ratio:", round(summary["mean_mismatch_ratio"], 3))
        print("Mean vx after 1.0 s:", round(summary["mean_vx_after_1s"], 3))
        for item in summary["per_leg"]:
            print(
                f'{item["leg"]}: stance_success={item["stance_success_ratio"]:.3f}, '
                f'force_enabled={item["force_enabled_ratio"]:.3f}, '
                f'touchdown_delay_mean={item["touchdown_delay_mean_s"]}'
            )

    print(
        "Phase-4 params: "
        f"clearance={clearance:.3f} m, "
        f"step_len={step_len:.3f} m, "
        f"touchdown_depth={touchdown_depth:.3f} m, "
        f"touchdown_forward={touchdown_forward:.3f} m, "
        f"touchdown_search_window={touchdown_search_window:.3f} s"
    )
    return log, saved, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MuJoCo phase-4: touchdown-search + contact-gated stance force"
    )
    parser.add_argument("--scenario", default="straight_trot", choices=["straight_trot", "turn_pi_over_4"])
    parser.add_argument(
        "--model",
        default="./mujoco_menagerie/unitree_a1/scene.xml",
        help="Path to a MuJoCo MJCF scene (Menagerie scene.xml recommended)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--output-dir", default=None, help="Directory for saved plots")
    parser.add_argument("--clearance", type=float, default=0.05, help="Swing foot clearance [m]")
    parser.add_argument("--step-len", type=float, default=None, help="Swing foot forward placement [m]")
    parser.add_argument("--touchdown-depth", type=float, default=0.03, help="Downward search depth after scheduled stance begins [m]")
    parser.add_argument("--touchdown-forward", type=float, default=None, help="Small forward bias during touchdown search [m]")
    parser.add_argument(
        "--touchdown-search-window",
        type=float,
        default=None,
        help="Time window [s] over which touchdown search ramps in after scheduled stance begins",
    )
    args = parser.parse_args()

    try:
        cfg = make_config(args.scenario)
        outdir = args.output_dir or f"outputs_mujoco_phase4/{args.scenario}"
        _, saved, _ = run_mujoco_phase4(
            cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=outdir,
            clearance=args.clearance,
            step_len=args.step_len,
            touchdown_depth=args.touchdown_depth,
            touchdown_forward=args.touchdown_forward,
            touchdown_search_window=args.touchdown_search_window,
        )
        print(f"MuJoCo phase-4 run finished for scenario: {args.scenario}")
        print(f"Model: {Path(args.model).resolve()}")
        if saved:
            print("Saved outputs:")
            for p in saved:
                print(f" - {p}")
    except Exception:
        print("MuJoCo phase-4 run failed. Full traceback below:")
        traceback.print_exc()
