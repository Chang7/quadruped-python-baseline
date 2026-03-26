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
from phases.mujoco_phase6_helpers import build_phase6_summary, write_phase6_summary
from phases.mujoco_phase7_helpers import leg_internal_force_to_qfrc


def _transform_force(f: np.ndarray, d: mujoco.MjData, base_body_name: str, force_frame: str) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if force_frame == 'world':
        return f
    if force_frame == 'body':
        R = d.body(base_body_name).xmat.reshape(3, 3).copy()
        return R @ f
    raise ValueError(f'Unknown force_frame={force_frame}')


def run_mujoco_phase7(
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
    realization: str = 'joint',
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

        log['t'].append(float(d.time))
        log['x'].append(x_now.copy())
        log['u'].append(u_hold.copy())
        log['u_applied'].append(u_applied.copy())
        log['contact'].append(scheduled_contact_hold.copy())
        log['contact_actual'].append(actual_post.copy())
        log['contact_force_enabled'].append(force_enabled.copy())
        log['x_ref0'].append(x_ref0_hold.copy())
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
        summary = build_phase6_summary(log, bindings)
        summary['realization'] = realization
        saved.append(write_phase6_summary(output_dir, summary))
    else:
        summary = build_phase6_summary(log, bindings)
        summary['realization'] = realization

    if summary is not None:
        print('Mean mismatch ratio:', round(summary['mean_mismatch_ratio'], 3))
        print('Mean vx after 1.0 s:', round(summary['mean_vx_after_1s'], 3))
        print('Mean sum Fx after 1.0 s:', round(summary['mean_sum_fx_after_1s'], 3))
        print('Mean sum Fy after 1.0 s:', round(summary['mean_sum_fy_after_1s'], 3))
        print('Mean sum Fz after 1.0 s:', round(summary['mean_sum_fz_after_1s'], 3))
        for item in summary['per_leg']:
            print(
                f"{item['leg']}: stance_success={item['stance_success_ratio']:.3f}, "
                f"force_enabled={item['force_enabled_ratio']:.3f}, "
                f"mean_fx={item['mean_fx_when_enabled']:.3f}, "
                f"mean_fy={item['mean_fy_when_enabled']:.3f}, "
                f"mean_fz={item['mean_fz_when_enabled']:.3f}, "
                f"touchdown_delay_mean={item['touchdown_delay_mean_s']}"
            )

    print(
        'Phase-7 params: '
        f'realization={realization}, '
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo phase-7: compare external foot wrench vs internal J^T f leg torque realization')
    parser.add_argument('--scenario', default='straight_trot', choices=['straight_trot', 'turn_pi_over_4'])
    parser.add_argument('--model', default='./mujoco_menagerie/unitree_a1/scene.xml')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output-dir', default='local_outputs/outputs_mujoco_phase7/run')
    parser.add_argument('--clearance', type=float, default=0.05)
    parser.add_argument('--step-len-front', type=float, default=None)
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
    parser.add_argument('--realization', choices=['external', 'joint'], default='joint')
    args = parser.parse_args()

    cfg = make_config(args.scenario)
    try:
        _, saved, _ = run_mujoco_phase7(
            cfg=cfg,
            model_path=args.model,
            viewer=(not args.headless),
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
        )
        print(f'MuJoCo phase-7 run finished for scenario: {args.scenario}')
        print(f'Model: {args.model}')
        print('Saved outputs:')
        for p in saved:
            print(f' - {p}')
    except Exception:
        print('MuJoCo phase-7 run failed. Full traceback below:')
        traceback.print_exc()
        raise
