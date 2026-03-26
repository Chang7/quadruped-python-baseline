from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import traceback

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
    foot_point_world,
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
    build_ctrl_targets_phase5,
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


def _transform_force(f: np.ndarray, d: mujoco.MjData, base_body_name: str, force_frame: str) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if force_frame == 'world':
        return f
    if force_frame == 'body':
        R = d.body(base_body_name).xmat.reshape(3, 3).copy()
        return R @ f
    raise ValueError(f'Unknown force_frame={force_frame}')


def run_mujoco_phase9(
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
) -> tuple[dict, list[str], dict | None]:
    controller = MPCControllerOSQP(verbose=False)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Reset first so that keyframe / default qpos is loaded before we optionally edit contacts.
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)

    bindings = discover_model_bindings(m)
    disabled = []
    if disable_nonfoot_collision:
        disabled = disable_nonfoot_leg_collisions(m, bindings)
        # Refresh data after collision property edits.
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        else:
            mujoco.mj_resetData(m, d)
    print(print_binding_summary(bindings))
    if disabled:
        print('Disabled non-foot collision geoms:', disabled)

    home_ctrl = resolve_home_ctrl(m, d)

    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    store_home_joint_qpos(d, bindings)
    compute_swing_delta_maps(m, d, bindings)
    initialize_phase5_leg_state(bindings)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.zeros(4, dtype=bool)
    prev_sched = np.ones(4, dtype=bool)

    if step_len_front is None:
        step_len_front = max(0.03, min(0.08, 0.55 * cfg.desired_speed * cfg.swing_time))
    if touchdown_forward_front is None:
        touchdown_forward_front = max(0.012, min(0.03, 0.40 * step_len_front))
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
        # Foot-only actual GRF (compatible with phase-8 helpers / summary)
        'actual_grf_a': [],
        'actual_grf_b': [],
        # All support-geom actual GRF
        'actual_support_a': [],
        'actual_support_b': [],
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

        update_phase5_state_pre(d, bindings, prev_sched, scheduled_contact_hold)

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase5(
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
            point_world = foot_point_world(d, binding, FALLBACK_FOOT_LOCAL_OFFSET)
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
        summary = build_phase9_summary(log, bindings, disable_nonfoot_collision=disable_nonfoot_collision)
        summary['realization'] = realization
        saved.append(write_phase9_summary(output_dir, summary))
    else:
        summary = build_phase9_summary(log, bindings, disable_nonfoot_collision=disable_nonfoot_collision)
        summary['realization'] = realization

    if summary is not None:
        print('Mean mismatch ratio:', round(summary['mean_mismatch_ratio'], 3))
        print('Mean vx after 1.0 s:', round(summary['mean_vx_after_1s'], 3))
        print('Mean commanded sum Fx after 1.0 s:', round(summary['mean_sum_fx_after_1s'], 3))
        print('Mean actual FOOT sum Fx after 1.0 s:', round(summary['mean_actual_sum_fx_after_1s'], 3))
        print('Mean actual SUPPORT sum Fx after 1.0 s:', round(summary['mean_actual_support_sum_fx_after_1s'], 3))
        print('Mean commanded sum Fz after 1.0 s:', round(summary['mean_sum_fz_after_1s'], 3))
        print('Mean actual FOOT sum Fz after 1.0 s:', round(summary['mean_actual_sum_fz_after_1s'], 3))
        print('Mean actual SUPPORT sum Fz after 1.0 s:', round(summary['mean_actual_support_sum_fz_after_1s'], 3))
        print('Foot-only GRF sign convention:', summary['actual_grf_sign_convention'])
        print('All-support GRF sign convention:', summary['actual_support_sign_convention'])
        for item in summary['per_leg']:
            print(
                f"{item['leg']}: "
                f"cmd_fz={item['mean_fz_when_enabled']:.3f}, "
                f"foot_act_fz={item['mean_actual_fz_when_enabled']:.3f}, "
                f"support_act_fz={item['mean_actual_support_fz_when_enabled']:.3f}, "
                f"support_minus_foot_fz={item['support_minus_foot_fz_when_enabled']:.3f}, "
                f"stance_success={item['stance_success_ratio']:.3f}, "
                f"force_enabled={item['force_enabled_ratio']:.3f}, "
                f"touchdown_delay_mean={item['touchdown_delay_mean_s']}"
            )

    print(
        'Phase-9 params: '
        f'realization={realization}, '
        f'disable_nonfoot_collision={disable_nonfoot_collision}, '
        f'clearance={clearance:.3f} m, '
        f'step_len_front={step_len_front:.3f} m, '
        f'rear_step_scale={rear_step_scale:.3f}, '
        f'td_depth_front={touchdown_depth_front:.3f} m, '
        f'td_depth_rear={touchdown_depth_rear:.3f} m, '
        f'td_fwd_front={touchdown_forward_front:.3f} m, '
        f'td_fwd_rear={touchdown_forward_rear:.3f} m, '
        f'td_window_front={touchdown_search_window_front:.3f} s, '
        f'td_window_rear={touchdown_search_window_rear:.3f} s, '
        f'stance_hold_time={stance_hold_time:.3f} s, '
        f'force_frame={force_frame}, fx_scale={fx_scale:.3f}, fy_scale={fy_scale:.3f}, fz_scale={fz_scale:.3f}, '
        f'zero_tangential={zero_tangential}'
    )

    return log, saved, summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='MuJoCo phase-9 support-audit / foot-only-contact runner')
    p.add_argument('--scenario', type=str, default='straight_trot', choices=['straight_trot', 'turn_pi_over_4'])
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--clearance', type=float, default=0.05)
    p.add_argument('--step-len-front', type=float, default=None)
    p.add_argument('--rear-step-scale', type=float, default=0.65)
    p.add_argument('--touchdown-depth-front', type=float, default=0.035)
    p.add_argument('--touchdown-depth-rear', type=float, default=0.05)
    p.add_argument('--touchdown-forward-front', type=float, default=None)
    p.add_argument('--touchdown-forward-rear', type=float, default=0.004)
    p.add_argument('--touchdown-search-window-front', type=float, default=None)
    p.add_argument('--touchdown-search-window-rear', type=float, default=0.10)
    p.add_argument('--stance-hold-time', type=float, default=0.0)
    p.add_argument('--force-frame', type=str, default='body', choices=['world', 'body'])
    p.add_argument('--fx-scale', type=float, default=1.0)
    p.add_argument('--fy-scale', type=float, default=1.0)
    p.add_argument('--fz-scale', type=float, default=1.0)
    p.add_argument('--zero-tangential', action='store_true')
    p.add_argument('--realization', type=str, default='external', choices=['external', 'joint'])
    p.add_argument('--disable-nonfoot-collision', action='store_true')
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    cfg = make_config(args.scenario)
    try:
        _, saved, _ = run_mujoco_phase9(
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
        )
        print(f'MuJoCo phase-9 run finished for scenario: {args.scenario}')
        print(f'Model: {args.model}')
        if args.output_dir is not None:
            print('Saved outputs:')
            for s in saved:
                print(' -', s)
        return 0
    except Exception:
        print('MuJoCo phase-9 run failed. Full traceback below:')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
