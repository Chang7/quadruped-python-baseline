from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from baseline.config import make_config, IDX_G, IDX_TH, IDX_V, IDX_W, IDX_P
from baseline.fsm import rollout_contact_schedule
from baseline.reference import rollout_reference
from baseline.model import build_prediction_model
from baseline.qp_builder import build_qp
from baseline.controller_osqp import MPCControllerOSQP
from baseline.plotting import plot_logs

# Our baseline code uses [FL, FR, RL, RR].
LEG_ORDER = ["FL", "FR", "RL", "RR"]
BASE_BODY = "trunk"

# A convenient local point near the foot for the Unitree A1 Menagerie model.
# The calf body origin is near the knee; the foot contact point lies roughly below it.
FOOT_LOCAL_OFFSET = np.array([0.0, 0.0, -0.2], dtype=float)


def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw], dtype=float)


def foot_point_world(d: mujoco.MjData, leg: str) -> np.ndarray:
    body = d.body(f"{leg}_calf")
    R = body.xmat.reshape(3, 3)
    return body.xpos.copy() + R @ FOOT_LOCAL_OFFSET


def foot_rel_world(d: mujoco.MjData) -> np.ndarray:
    p_base = d.body(BASE_BODY).xpos.copy()
    feet = [foot_point_world(d, leg) - p_base for leg in LEG_ORDER]
    return np.vstack(feet)


def mujoco_to_x(d: mujoco.MjData, cfg) -> np.ndarray:
    x = np.zeros(cfg.nx, dtype=float)
    trunk = d.body(BASE_BODY)
    x[IDX_P] = trunk.xpos.copy()
    x[IDX_TH] = mat_to_rpy(trunk.xmat.reshape(3, 3))

    # MuJoCo free joint qvel layout is [linear(3), angular(3)].
    # Linear velocity is in world frame. Angular velocity is body-local.
    x[IDX_V] = d.qvel[:3].copy()
    x[IDX_W] = d.qvel[3:6].copy()
    x[IDX_G] = cfg.g
    return x


def force_to_qfrc(m: mujoco.MjModel, d: mujoco.MjData, body_name: str, point_world: np.ndarray, f_world: np.ndarray) -> np.ndarray:
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)

    # Jacobians are consistent after kinematics/comPos.
    mujoco.mj_jac(m, d, jacp, jacr, point_world, body_id)
    return jacp.T @ f_world


def run_mujoco(cfg, model_path: str, viewer: bool = True, output_dir: str | None = None) -> tuple[dict, list[str]]:
    controller = MPCControllerOSQP(verbose=False)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)

    # Hold the nominal posture with the model's position actuators, if present.
    home_ctrl = d.ctrl.copy() if m.nu > 0 else np.zeros(0, dtype=float)
    if m.nu > 0 and d.qpos.shape[0] >= 7 + m.nu:
        home_ctrl = d.qpos[7:7 + m.nu].copy()

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)

    log = {"t": [], "x": [], "u": [], "contact": [], "x_ref0": []}

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold

        # Keep posture actuators at the initial pose so the articulated robot does not collapse immediately.
        if m.nu > 0:
            d.ctrl[:] = home_ctrl

        # Make sure kinematics / COM quantities are current before building Jacobians.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg)
            feet = foot_rel_world(d)

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
            next_mpc_time += cfg.dt_mpc

        d.qfrc_applied[:] = 0.0
        for leg_i, leg in enumerate(LEG_ORDER):
            f_world = u_hold[3 * leg_i: 3 * leg_i + 3]
            point_world = foot_point_world(d, leg)
            d.qfrc_applied[:] += force_to_qfrc(m, d, f"{leg}_calf", point_world, f_world)

        mujoco.mj_step(m, d)

        x_now = mujoco_to_x(d, cfg)
        contact_now = rollout_contact_schedule(float(d.time), cfg)[0]

        log["t"].append(float(d.time))
        log["x"].append(x_now.copy())
        log["u"].append(u_hold.copy())
        log["contact"].append(contact_now.copy())
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
    if output_dir is not None:
        saved = plot_logs(log, cfg, output_dir=output_dir)
    return log, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the quadruped baseline MPC against a MuJoCo quadruped model")
    parser.add_argument("--scenario", default="straight_trot", choices=["straight_trot", "turn_pi_over_4"])
    parser.add_argument(
        "--model",
        default="./mujoco_menagerie/unitree_a1/scene.xml",
        help="Path to a MuJoCo MJCF scene (Menagerie scene.xml recommended)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--output-dir", default=None, help="Directory for saved plots")
    args = parser.parse_args()

    try:
        cfg = make_config(args.scenario)
        outdir = args.output_dir or f"outputs_mujoco/{args.scenario}"
        _, saved = run_mujoco(cfg, model_path=args.model, viewer=not args.headless, output_dir=outdir)
        print(f"MuJoCo run finished for scenario: {args.scenario}")
        print(f"Model: {Path(args.model).resolve()}")
        if saved:
            print("Saved figures:")
            for p in saved:
                print(f" - {p}")
    except Exception:
        print("MuJoCo run failed. Full traceback below:")
        traceback.print_exc()
