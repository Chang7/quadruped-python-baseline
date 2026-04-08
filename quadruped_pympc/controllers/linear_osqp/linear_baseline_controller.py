from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from quadruped_pympc import config as global_cfg

LEG_ORDER = ("FL", "FR", "RL", "RR")
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_W = slice(9, 12)
IDX_G = 12
NX = 13
NU = 12


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass
class LinearOSQPConfig:
    horizon: int
    dt: float
    mass: float
    inertia: np.ndarray
    gravity: float
    mu: float
    grf_min: float
    grf_max: float
    nominal_height: float
    Q_p: float = 1e6
    Q_v: float = 1e6
    Q_theta: float = 1e6
    Q_theta_roll: float | None = None
    Q_theta_pitch: float | None = None
    Q_w: float = 1e6
    Q_w_roll: float | None = None
    Q_w_pitch: float | None = None
    R_u: float = 1e1
    command_smoothing: float = 0.0
    du_xy_max: float = np.inf
    du_z_max: float = np.inf
    stance_ramp_steps: int = 1
    fy_scale: float = 1.0
    dynamic_fy_roll_gain: float = 0.0
    dynamic_fy_roll_ref: float = 0.20
    grf_max_scale: float = 1.0
    support_force_floor_ratio: float = 0.0
    latched_force_scale: float = 1.0
    latched_floor_scale: float = 1.0
    latched_same_side_receiver_scale: float = 1.0
    latched_axle_receiver_scale: float = 1.0
    latched_diagonal_receiver_scale: float = 1.0
    latched_front_receiver_scale: float = 1.0
    latched_rear_receiver_scale: float = 1.0
    front_latched_pitch_relief_gain: float = 0.0
    front_latched_rear_bias_gain: float = 0.0
    rear_floor_base_scale: float = 0.65
    rear_floor_pitch_gain: float = 0.20
    side_rebalance_gain: float = 0.0
    side_rebalance_ref: float = 0.35
    pitch_rebalance_gain: float = 0.0
    pitch_rebalance_ref: float = 0.35
    support_centroid_x_gain: float = 0.0
    support_centroid_y_gain: float = 0.0
    pre_swing_lookahead_steps: int = 3
    pre_swing_front_shift_scale: float = 1.0
    pre_swing_rear_shift_scale: float = 1.0
    support_reference_mix: float = 0.55
    support_reference_xy_mix: float | None = None
    vx_gain: float = 1.6
    vy_gain: float = 4.5
    z_pos_gain: float = 20.0
    z_vel_gain: float = 5.0
    min_vertical_force_scale: float = 0.5
    reduced_support_vertical_boost: float = 0.0
    roll_angle_gain: float = 24.0
    roll_rate_gain: float = 6.0
    pitch_angle_gain: float = 28.0
    pitch_rate_gain: float = 8.0
    yaw_angle_gain: float = 4.0
    yaw_rate_gain: float = 1.5
    roll_ref_offset: float = 0.0
    pitch_ref_offset: float = 0.0

    def Q(self) -> np.ndarray:
        q = np.zeros((NX, NX), dtype=float)
        # For locomotion we regulate height, not the absolute world x/y position.
        q[IDX_P, IDX_P] = np.diag([0.0, 0.0, self.Q_p])
        q[IDX_V, IDX_V] = np.eye(3) * self.Q_v
        q_theta_roll = self.Q_theta if self.Q_theta_roll is None else self.Q_theta_roll
        q_theta_pitch = self.Q_theta if self.Q_theta_pitch is None else self.Q_theta_pitch
        q_w_roll = self.Q_w if self.Q_w_roll is None else self.Q_w_roll
        q_w_pitch = self.Q_w if self.Q_w_pitch is None else self.Q_w_pitch
        q[IDX_TH, IDX_TH] = np.diag([q_theta_roll, q_theta_pitch, self.Q_theta])
        q[IDX_W, IDX_W] = np.diag([q_w_roll, q_w_pitch, self.Q_w])
        return q

    def QN(self) -> np.ndarray:
        return self.Q()

    def R(self) -> np.ndarray:
        return np.eye(NU, dtype=float) * self.R_u


def _get_linear_params() -> dict:
    default = {
        "Q_p": 2e4,
        "Q_v": 4e4,
        "Q_theta": 2e4,
        "Q_theta_roll": None,
        "Q_theta_pitch": None,
        "Q_w": 2e3,
        "Q_w_roll": None,
        "Q_w_pitch": None,
        "R_u": 5.0,
        "command_smoothing": 0.35,
        "du_xy_max": 2.5,
        "du_z_max": 3.5,
        "stance_ramp_steps": 6,
        "fy_scale": 0.15,
        "dynamic_fy_roll_gain": 0.0,
        "dynamic_fy_roll_ref": 0.20,
        "grf_max_scale": 0.35,
        "support_force_floor_ratio": 0.0,
        "latched_force_scale": 1.0,
        "latched_floor_scale": 1.0,
        "latched_same_side_receiver_scale": 1.0,
        "latched_axle_receiver_scale": 1.0,
        "latched_diagonal_receiver_scale": 1.0,
        "latched_front_receiver_scale": 1.0,
        "latched_rear_receiver_scale": 1.0,
        "front_latched_pitch_relief_gain": 0.0,
        "front_latched_rear_bias_gain": 0.0,
        "rear_floor_base_scale": 0.65,
        "rear_floor_pitch_gain": 0.20,
        "side_rebalance_gain": 0.0,
        "side_rebalance_ref": 0.35,
        "pitch_rebalance_gain": 0.0,
        "pitch_rebalance_ref": 0.35,
        "support_centroid_x_gain": 0.0,
        "support_centroid_y_gain": 0.0,
        "pre_swing_lookahead_steps": 3,
        "pre_swing_front_shift_scale": 1.0,
        "pre_swing_rear_shift_scale": 1.0,
        "support_reference_mix": 0.55,
        "support_reference_xy_mix": None,
        "vx_gain": 1.6,
        "vy_gain": 4.5,
        "z_pos_gain": 20.0,
        "z_vel_gain": 5.0,
        "min_vertical_force_scale": 0.5,
        "reduced_support_vertical_boost": 0.0,
        "roll_angle_gain": 24.0,
        "roll_rate_gain": 6.0,
        "pitch_angle_gain": 28.0,
        "pitch_rate_gain": 8.0,
        "yaw_angle_gain": 4.0,
        "yaw_rate_gain": 1.5,
        "roll_ref_offset": 0.0,
        "pitch_ref_offset": 0.0,
    }
    maybe = getattr(global_cfg, "linear_osqp_params", None)
    if isinstance(maybe, dict):
        out = default.copy()
        out.update(maybe)
        return out
    return default


class LinearSRBDController:
    """Drop-in OSQP linear MPC that plugs into the Quadruped-PyMPC stack.

    This controller intentionally reuses PyMPC's gait / foothold / swing / stance
    realization layers. It only replaces the SRBD control solve with a compact
    13-state, 12-input linear force MPC derived from the user's existing baseline.
    """

    def __init__(self) -> None:
        self.last_u = np.zeros(NU, dtype=float)
        self.last_status = "cold_start"
        self.prev_contact = np.zeros(4, dtype=bool)
        self.stance_age = np.zeros(4, dtype=int)

    def reset(self) -> None:
        self.last_u = np.zeros(NU, dtype=float)
        self.last_status = "reset"
        self.prev_contact = np.zeros(4, dtype=bool)
        self.stance_age = np.zeros(4, dtype=int)

    def _make_cfg(self, inertia: np.ndarray) -> LinearOSQPConfig:
        params = _get_linear_params()
        inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
        body_weight = float(global_cfg.mass) * float(global_cfg.gravity_constant)
        grf_max = min(float(global_cfg.mpc_params["grf_max"]), float(params.get("grf_max_scale", 1.0)) * body_weight)
        return LinearOSQPConfig(
            horizon=int(global_cfg.mpc_params["horizon"]),
            dt=float(global_cfg.mpc_params["dt"]),
            mass=float(global_cfg.mass),
            inertia=inertia,
            gravity=float(global_cfg.gravity_constant),
            mu=float(global_cfg.mpc_params["mu"]),
            grf_min=float(global_cfg.mpc_params["grf_min"]),
            grf_max=float(grf_max),
            nominal_height=float(global_cfg.simulation_params["ref_z"]),
            Q_p=float(params["Q_p"]),
            Q_v=float(params["Q_v"]),
            Q_theta=float(params["Q_theta"]),
            Q_theta_roll=(
                None if params.get("Q_theta_roll", None) is None else float(params.get("Q_theta_roll"))
            ),
            Q_theta_pitch=(
                None if params.get("Q_theta_pitch", None) is None else float(params.get("Q_theta_pitch"))
            ),
            Q_w=float(params["Q_w"]),
            Q_w_roll=(
                None if params.get("Q_w_roll", None) is None else float(params.get("Q_w_roll"))
            ),
            Q_w_pitch=(
                None if params.get("Q_w_pitch", None) is None else float(params.get("Q_w_pitch"))
            ),
            R_u=float(params["R_u"]),
            command_smoothing=float(params.get("command_smoothing", 0.0)),
            du_xy_max=float(params.get("du_xy_max", np.inf)),
            du_z_max=float(params.get("du_z_max", np.inf)),
            stance_ramp_steps=max(1, int(params.get("stance_ramp_steps", 1))),
            fy_scale=float(params.get("fy_scale", 1.0)),
            dynamic_fy_roll_gain=float(params.get("dynamic_fy_roll_gain", 0.0)),
            dynamic_fy_roll_ref=float(params.get("dynamic_fy_roll_ref", 0.20)),
            grf_max_scale=float(params.get("grf_max_scale", 1.0)),
            support_force_floor_ratio=float(params.get("support_force_floor_ratio", 0.0)),
            latched_force_scale=float(params.get("latched_force_scale", 1.0)),
            latched_floor_scale=float(params.get("latched_floor_scale", 1.0)),
            latched_same_side_receiver_scale=float(params.get("latched_same_side_receiver_scale", 1.0)),
            latched_axle_receiver_scale=float(params.get("latched_axle_receiver_scale", 1.0)),
            latched_diagonal_receiver_scale=float(params.get("latched_diagonal_receiver_scale", 1.0)),
            latched_front_receiver_scale=float(params.get("latched_front_receiver_scale", 1.0)),
            latched_rear_receiver_scale=float(params.get("latched_rear_receiver_scale", 1.0)),
            front_latched_pitch_relief_gain=float(params.get("front_latched_pitch_relief_gain", 0.0)),
            front_latched_rear_bias_gain=float(params.get("front_latched_rear_bias_gain", 0.0)),
            rear_floor_base_scale=float(params.get("rear_floor_base_scale", 0.65)),
            rear_floor_pitch_gain=float(params.get("rear_floor_pitch_gain", 0.20)),
            side_rebalance_gain=float(params.get("side_rebalance_gain", 0.0)),
            side_rebalance_ref=float(params.get("side_rebalance_ref", 0.35)),
            pitch_rebalance_gain=float(params.get("pitch_rebalance_gain", 0.0)),
            pitch_rebalance_ref=float(params.get("pitch_rebalance_ref", 0.35)),
            support_centroid_x_gain=float(params.get("support_centroid_x_gain", 0.0)),
            support_centroid_y_gain=float(params.get("support_centroid_y_gain", 0.0)),
            pre_swing_lookahead_steps=max(0, int(params.get("pre_swing_lookahead_steps", 3))),
            pre_swing_front_shift_scale=float(params.get("pre_swing_front_shift_scale", 1.0)),
            pre_swing_rear_shift_scale=float(params.get("pre_swing_rear_shift_scale", 1.0)),
            support_reference_mix=float(np.clip(params.get("support_reference_mix", 0.55), 0.0, 1.0)),
            support_reference_xy_mix=(
                None
                if params.get("support_reference_xy_mix", None) is None
                else float(np.clip(params.get("support_reference_xy_mix", 0.55), 0.0, 1.0))
            ),
            vx_gain=float(params.get("vx_gain", 1.6)),
            vy_gain=float(params.get("vy_gain", 4.5)),
            z_pos_gain=float(params.get("z_pos_gain", 20.0)),
            z_vel_gain=float(params.get("z_vel_gain", 5.0)),
            min_vertical_force_scale=float(params.get("min_vertical_force_scale", 0.5)),
            reduced_support_vertical_boost=float(params.get("reduced_support_vertical_boost", 0.0)),
            roll_angle_gain=float(params.get("roll_angle_gain", 24.0)),
            roll_rate_gain=float(params.get("roll_rate_gain", 6.0)),
            pitch_angle_gain=float(params.get("pitch_angle_gain", 28.0)),
            pitch_rate_gain=float(params.get("pitch_rate_gain", 8.0)),
            yaw_angle_gain=float(params.get("yaw_angle_gain", 4.0)),
            yaw_rate_gain=float(params.get("yaw_rate_gain", 1.5)),
            roll_ref_offset=float(params.get("roll_ref_offset", 0.0)),
            pitch_ref_offset=float(params.get("pitch_ref_offset", 0.0)),
        )

    @staticmethod
    def _state_to_vector(state_current: dict, gravity: float) -> np.ndarray:
        x = np.zeros((NX,), dtype=float)
        orientation = np.asarray(state_current["orientation"], dtype=float).reshape(3)
        x[IDX_P] = np.asarray(state_current["position"], dtype=float).reshape(3)
        x[IDX_V] = np.asarray(state_current["linear_velocity"], dtype=float).reshape(3)
        x[IDX_TH] = orientation
        x[IDX_W] = np.asarray(state_current["angular_velocity"], dtype=float).reshape(3)
        x[IDX_G] = gravity
        return x

    @staticmethod
    def _extract_footholds(ref_state: dict) -> np.ndarray:
        return np.vstack(
            [
                np.asarray(ref_state["ref_foot_FL"], dtype=float).reshape(3),
                np.asarray(ref_state["ref_foot_FR"], dtype=float).reshape(3),
                np.asarray(ref_state["ref_foot_RL"], dtype=float).reshape(3),
                np.asarray(ref_state["ref_foot_RR"], dtype=float).reshape(3),
            ]
        )

    @staticmethod
    def _contact_sequence_to_schedule(contact_sequence: np.ndarray, horizon: int) -> np.ndarray:
        arr = np.asarray(contact_sequence)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected contact_sequence shape: {arr.shape}")
        if arr.shape[0] == 4:
            arr = arr[:, :horizon].T
        elif arr.shape[1] == 4:
            arr = arr[:horizon, :]
        else:
            raise ValueError(f"Cannot interpret contact_sequence shape: {arr.shape}")
        if arr.shape[0] < horizon:
            pad = np.repeat(arr[-1:, :], horizon - arr.shape[0], axis=0)
            arr = np.vstack([arr, pad])
        return arr.astype(bool)

    @staticmethod
    def _foot_rel_world(state_current: dict) -> np.ndarray:
        com = np.asarray(state_current["position"], dtype=float).reshape(3)
        feet = np.vstack([
            np.asarray(state_current["foot_FL"], dtype=float).reshape(3),
            np.asarray(state_current["foot_FR"], dtype=float).reshape(3),
            np.asarray(state_current["foot_RL"], dtype=float).reshape(3),
            np.asarray(state_current["foot_RR"], dtype=float).reshape(3),
        ])
        return feet - com[None, :]

    @staticmethod
    def _rollout_reference(x_now: np.ndarray, ref_state: dict, cfg: LinearOSQPConfig) -> np.ndarray:
        ref_pos = np.asarray(ref_state["ref_position"], dtype=float).reshape(3)
        ref_vel = np.asarray(ref_state["ref_linear_velocity"], dtype=float).reshape(3)
        ref_ori = np.asarray(ref_state["ref_orientation"], dtype=float).reshape(3)
        ref_w = np.asarray(ref_state["ref_angular_velocity"], dtype=float).reshape(3)

        x_ref = np.zeros((cfg.horizon + 1, NX), dtype=float)
        x_ref[0] = x_now.copy()
        x_ref[:, IDX_G] = cfg.gravity

        # Follow PyMPC's reference semantics: keep desired height/orientation from
        # WBInterface, integrate the commanded body velocities across the horizon.
        p = np.asarray(x_now[IDX_P], dtype=float).copy()
        th = np.asarray(ref_ori, dtype=float).copy()
        th[0] += float(cfg.roll_ref_offset)
        th[1] += float(cfg.pitch_ref_offset)
        for k in range(cfg.horizon + 1):
            if k > 0:
                p = p + cfg.dt * ref_vel
                th = th + cfg.dt * ref_w
            x_ref[k, IDX_P] = p
            x_ref[k, 2] = ref_pos[2]
            x_ref[k, IDX_V] = ref_vel
            x_ref[k, IDX_TH] = th
            x_ref[k, IDX_W] = ref_w
        return x_ref

    @staticmethod
    def _continuous_matrices(x_nom: np.ndarray, foot_rel_world: np.ndarray, cfg: LinearOSQPConfig) -> Tuple[np.ndarray, np.ndarray]:
        ac = np.zeros((NX, NX), dtype=float)
        bc = np.zeros((NX, NU), dtype=float)

        orientation = np.asarray(x_nom[IDX_TH], dtype=float).reshape(3)
        roll = float(orientation[0])
        pitch = float(orientation[1])
        # Match the nominal centroidal model: Euler angle rates come from body-frame
        # angular velocity through the current xyz Euler kinematics.
        conj = np.eye(3, dtype=float)
        conj[1, 1] = np.cos(roll)
        conj[2, 2] = np.cos(pitch) * np.cos(roll)
        conj[2, 1] = -np.sin(roll)
        conj[0, 2] = -np.sin(pitch)
        conj[1, 2] = np.cos(pitch) * np.sin(roll)
        body_from_world = Rotation.from_euler("xyz", orientation).as_matrix().T
        ac[IDX_P, IDX_V] = np.eye(3)
        ac[IDX_TH, IDX_W] = np.linalg.inv(conj)
        ac[IDX_V, IDX_G] = np.array([0.0, 0.0, -1.0], dtype=float)

        inertia_inv = np.linalg.inv(cfg.inertia)
        for leg in range(4):
            cols = slice(3 * leg, 3 * leg + 3)
            r_i = np.asarray(foot_rel_world[leg], dtype=float).reshape(3)
            bc[IDX_V, cols] = np.eye(3) / cfg.mass
            bc[IDX_W, cols] = inertia_inv @ (body_from_world @ _skew(r_i))
        return ac, bc

    @staticmethod
    def _discretize(ac: np.ndarray, bc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ad = np.eye(ac.shape[0], dtype=float) + dt * ac
        bd = dt * bc
        return ad, bd

    def _prediction_model(self, x_ref: np.ndarray, foot_rel_world: np.ndarray, cfg: LinearOSQPConfig) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        ad_list: list[np.ndarray] = []
        bd_list: list[np.ndarray] = []
        for k in range(cfg.horizon):
            ac, bc = self._continuous_matrices(x_ref[k], foot_rel_world, cfg)
            ad, bd = self._discretize(ac, bc, cfg.dt)
            ad_list.append(ad)
            bd_list.append(bd)
        return ad_list, bd_list

    @staticmethod
    def _x_slice(k: int) -> slice:
        return slice(k * NX, (k + 1) * NX)

    @staticmethod
    def _u_slice(k: int, horizon: int) -> slice:
        xdim = (horizon + 1) * NX
        return slice(xdim + k * NU, xdim + (k + 1) * NU)

    @staticmethod
    def _leg_friction_block(mu: float) -> np.ndarray:
        return np.array(
            [
                [-1.0, 0.0, -mu],
                [1.0, 0.0, -mu],
                [0.0, -1.0, -mu],
                [0.0, 1.0, -mu],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _build_qp(
        self,
        x_init: np.ndarray,
        x_ref: np.ndarray,
        ad_list: list[np.ndarray],
        bd_list: list[np.ndarray],
        contact_schedule: np.ndarray,
        u_ref: np.ndarray,
        cfg: LinearOSQPConfig,
    ) -> tuple[sparse.csc_matrix, np.ndarray, sparse.csc_matrix, np.ndarray, np.ndarray]:
        horizon = cfg.horizon
        nvar = (horizon + 1) * NX + horizon * NU

        q = cfg.Q()
        qn = cfg.QN()
        r = cfg.R()

        p = sparse.block_diag(
            [
                sparse.kron(sparse.eye(horizon, format="csc"), 2.0 * sparse.csc_matrix(q)),
                2.0 * sparse.csc_matrix(qn),
                sparse.kron(sparse.eye(horizon, format="csc"), 2.0 * sparse.csc_matrix(r)),
            ],
            format="csc",
        )

        qx = np.concatenate(
            [-2.0 * (q @ x_ref[k]) for k in range(horizon)] + [-2.0 * (qn @ x_ref[horizon])]
        )
        qu = np.concatenate([-2.0 * (r @ u_ref[k]) for k in range(horizon)])
        q_vec = np.concatenate([qx, qu])

        rows: list[sparse.csc_matrix] = []
        lower: list[float] = []
        upper: list[float] = []

        a0 = sparse.lil_matrix((NX, nvar), dtype=float)
        a0[:, self._x_slice(0)] = np.eye(NX)
        rows.append(a0.tocsc())
        lower.extend(x_init.tolist())
        upper.extend(x_init.tolist())

        for k in range(horizon):
            aeq = sparse.lil_matrix((NX, nvar), dtype=float)
            aeq[:, self._x_slice(k)] = -ad_list[k]
            aeq[:, self._x_slice(k + 1)] = np.eye(NX)
            aeq[:, self._u_slice(k, horizon)] = -bd_list[k]
            rows.append(aeq.tocsc())
            lower.extend([0.0] * NX)
            upper.extend([0.0] * NX)

        f_leg = self._leg_friction_block(cfg.mu)
        b_leg = np.array([0.0, 0.0, 0.0, 0.0, -cfg.grf_min, cfg.grf_max], dtype=float)

        for k in range(horizon):
            u_block = self._u_slice(k, horizon)
            for leg in range(4):
                cols = slice(u_block.start + 3 * leg, u_block.start + 3 * leg + 3)
                if bool(contact_schedule[k, leg]):
                    aineq = sparse.lil_matrix((6, nvar), dtype=float)
                    aineq[:, cols] = f_leg
                    rows.append(aineq.tocsc())
                    lower.extend([-np.inf] * 6)
                    upper.extend(b_leg.tolist())
                else:
                    asw = sparse.lil_matrix((3, nvar), dtype=float)
                    asw[:, cols] = np.eye(3)
                    rows.append(asw.tocsc())
                    lower.extend([0.0, 0.0, 0.0])
                    upper.extend([0.0, 0.0, 0.0])

        a = sparse.vstack(rows, format="csc")
        l = np.asarray(lower, dtype=float)
        u = np.asarray(upper, dtype=float)
        return p, q_vec, a, l, u

    @staticmethod
    def _solve_osqp(
        p: sparse.csc_matrix,
        q_vec: np.ndarray,
        a: sparse.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
        warm_start: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        prob = osqp.OSQP()
        prob.setup(
            P=p,
            q=q_vec,
            A=a,
            l=l,
            u=u,
            warm_start=True,
            verbose=False,
            polish=False,
        )
        if warm_start.size == q_vec.size:
            prob.warm_start(x=warm_start)
        res = prob.solve()
        status = str(getattr(res.info, "status", "unknown"))
        if not status.startswith("solved"):
            raise RuntimeError(status)
        return np.asarray(res.x, dtype=float), status

    @staticmethod
    def _apply_slew_and_smoothing(u_des: np.ndarray, u_prev: np.ndarray, cfg: LinearOSQPConfig) -> np.ndarray:
        u = np.asarray(u_des, dtype=float).reshape(4, 3).copy()
        up = np.asarray(u_prev, dtype=float).reshape(4, 3)
        if np.isfinite(cfg.du_xy_max):
            du = np.clip(u[:, 0:2] - up[:, 0:2], -cfg.du_xy_max, cfg.du_xy_max)
            u[:, 0:2] = up[:, 0:2] + du
        if np.isfinite(cfg.du_z_max):
            duz = np.clip(u[:, 2] - up[:, 2], -cfg.du_z_max, cfg.du_z_max)
            u[:, 2] = up[:, 2] + duz
        if cfg.command_smoothing > 0.0:
            a = float(np.clip(cfg.command_smoothing, 0.0, 0.99))
            u = a * up + (1.0 - a) * u
        return u.reshape(NU)

    @staticmethod
    def _planned_contact_schedule(ref_state: dict, fallback: np.ndarray, horizon: int) -> np.ndarray:
        planned = ref_state.get("planned_contact_sequence")
        if planned is None:
            return np.asarray(fallback, dtype=bool)
        try:
            return LinearSRBDController._contact_sequence_to_schedule(np.asarray(planned), horizon)
        except Exception:
            return np.asarray(fallback, dtype=bool)

    @staticmethod
    def _latched_release_alpha_now(ref_state: dict) -> np.ndarray:
        alpha = ref_state.get("latched_release_alpha")
        if alpha is None:
            return np.ones(4, dtype=float)
        try:
            return np.clip(np.asarray(alpha, dtype=float).reshape(4), 0.0, 1.0)
        except Exception:
            return np.ones(4, dtype=float)

    @staticmethod
    def _scale_latched_support(
        u_cmd: np.ndarray,
        contact_now: np.ndarray,
        planned_now: np.ndarray,
        release_alpha: np.ndarray,
        pitch_now: float,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        u = np.asarray(u_cmd, dtype=float).reshape(4, 3).copy()
        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        planned_now = np.asarray(planned_now, dtype=bool).reshape(4)
        release_alpha = np.clip(np.asarray(release_alpha, dtype=float).reshape(4), 0.0, 1.0)
        latched = contact_now & (~planned_now)
        if not np.any(latched):
            return u.reshape(NU)

        scale = float(np.clip(cfg.latched_force_scale, 0.0, 1.0))
        front_pitch_relief_gain = float(max(cfg.front_latched_pitch_relief_gain, 0.0))
        front_rear_bias_gain = float(max(cfg.front_latched_rear_bias_gain, 0.0))
        if scale >= 0.999 and front_pitch_relief_gain <= 1e-9 and front_rear_bias_gain <= 1e-9:
            return u.reshape(NU)

        support_mask = contact_now & planned_now
        if not np.any(support_mask):
            support_mask = contact_now.copy()
            support_mask[latched] = False

        effective_scale = np.ones(4, dtype=float)
        effective_scale[latched] = 1.0 - release_alpha[latched] * (1.0 - scale)
        pitch_ratio = 0.0
        if front_pitch_relief_gain > 1e-9:
            pitch_ratio = float(
                np.clip(max(float(pitch_now), 0.0) / max(float(cfg.pitch_rebalance_ref), 1e-6), 0.0, 1.0)
            )
            if pitch_ratio > 0.0:
                front_latched = latched & np.array([True, True, False, False], dtype=bool)
                if np.any(front_latched):
                    extra_relief = np.clip(
                        release_alpha[front_latched] * front_pitch_relief_gain * pitch_ratio,
                        0.0,
                        0.95,
                    )
                    effective_scale[front_latched] *= (1.0 - extra_relief)
        u[latched, 2] *= effective_scale[latched]

        if np.any(support_mask):
            for leg in np.flatnonzero(latched):
                removed_vertical = float(u_cmd.reshape(4, 3)[leg, 2] - u[leg, 2])
                if removed_vertical <= 1e-9:
                    continue
                receiver_idx = np.flatnonzero(support_mask)
                if receiver_idx.size == 0:
                    break

                support_headroom = np.maximum(cfg.grf_max - u[receiver_idx, 2], 1e-6)
                receiver_weights = support_headroom.copy()
                same_side = (receiver_idx % 2) == (leg % 2)
                same_axle = (receiver_idx // 2) == (leg // 2)
                diagonal = ~(same_side | same_axle)
                receiver_front = receiver_idx < 2
                receiver_rear = receiver_idx >= 2
                receiver_weights[same_side] *= float(max(cfg.latched_same_side_receiver_scale, 0.0))
                receiver_weights[same_axle] *= float(max(cfg.latched_axle_receiver_scale, 0.0))
                receiver_weights[diagonal] *= float(max(cfg.latched_diagonal_receiver_scale, 0.0))
                receiver_weights[receiver_front] *= float(max(cfg.latched_front_receiver_scale, 0.0))
                receiver_weights[receiver_rear] *= float(max(cfg.latched_rear_receiver_scale, 0.0))
                if leg < 2 and front_rear_bias_gain > 1e-9 and pitch_ratio > 0.0:
                    rear_boost = 1.0 + front_rear_bias_gain * pitch_ratio
                    receiver_weights[receiver_rear] *= rear_boost
                    receiver_weights[receiver_front] /= rear_boost
                if float(np.sum(receiver_weights)) <= 1e-9:
                    receiver_weights = support_headroom
                if float(np.sum(receiver_weights)) <= 1e-9:
                    receiver_weights = np.ones_like(receiver_weights)
                u[receiver_idx, 2] += removed_vertical * (receiver_weights / float(np.sum(receiver_weights)))

        u[:, 2] = np.clip(u[:, 2], 0.0, cfg.grf_max)
        return u.reshape(NU)

    @staticmethod
    def _apply_contact_floors(
        u_cmd: np.ndarray,
        contact_now: np.ndarray,
        planned_now: np.ndarray,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        u = np.asarray(u_cmd, dtype=float).reshape(4, 3).copy()
        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        planned_now = np.asarray(planned_now, dtype=bool).reshape(4)
        if cfg.support_force_floor_ratio <= 0.0 or not np.any(contact_now):
            return u.reshape(NU)

        latched = contact_now & (~planned_now)
        support_mask = contact_now & planned_now
        n_support = int(np.count_nonzero(support_mask))
        if n_support <= 0:
            n_support = int(np.count_nonzero(contact_now))
            support_mask = contact_now.copy()
            latched = np.zeros(4, dtype=bool)
        support_floor = (cfg.support_force_floor_ratio * cfg.mass * cfg.gravity) / float(n_support)
        if np.any(support_mask):
            u[support_mask, 2] = np.maximum(u[support_mask, 2], support_floor)
        if np.any(latched):
            latched_floor_scale = float(np.clip(cfg.latched_floor_scale, 0.0, 1.0))
            u[latched, 2] = np.maximum(
                u[latched, 2], latched_floor_scale * support_floor
            )
        return u.reshape(NU)

    @staticmethod
    def _apply_pitch_rebalance(
        u_cmd: np.ndarray,
        contact_now: np.ndarray,
        pitch_now: float,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        gain = float(max(cfg.pitch_rebalance_gain, 0.0))
        if gain <= 0.0:
            return np.asarray(u_cmd, dtype=float).reshape(NU)

        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        active_legs = np.flatnonzero(contact_now)
        if active_legs.size <= 1:
            return np.asarray(u_cmd, dtype=float).reshape(NU)

        local_front = np.flatnonzero(active_legs < 2)
        local_rear = np.flatnonzero(active_legs >= 2)
        if local_front.size == 0 or local_rear.size == 0:
            return np.asarray(u_cmd, dtype=float).reshape(NU)

        pitch_ratio = float(np.clip(max(float(pitch_now), 0.0) / max(float(cfg.pitch_rebalance_ref), 1e-6), 0.0, 1.0))
        if pitch_ratio <= 0.0:
            return np.asarray(u_cmd, dtype=float).reshape(NU)

        u = np.asarray(u_cmd, dtype=float).reshape(4, 3).copy()
        fz_active = u[active_legs, 2].copy()
        total_fz = float(fz_active.sum())
        if total_fz <= 1e-9:
            return u.reshape(NU)

        support_floor = 0.0
        if cfg.support_force_floor_ratio > 0.0:
            support_floor = (cfg.support_force_floor_ratio * cfg.mass * cfg.gravity) / float(active_legs.size)
        lower = np.full_like(fz_active, max(cfg.grf_min, support_floor), dtype=float)

        rear_nominal = total_fz * float(local_rear.size) / float(active_legs.size)
        required_rear = rear_nominal * (1.0 + gain * pitch_ratio)
        fz_active = LinearSRBDController._transfer_group_load(fz_active, local_rear, local_front, required_rear, lower)

        total_after = float(fz_active.sum())
        if total_after > 1e-9:
            fz_active *= total_fz / total_after
        u[active_legs, 2] = np.clip(fz_active, lower, cfg.grf_max)
        return u.reshape(NU)

    def _apply_contact_conditioning(
        self,
        u_cmd: np.ndarray,
        contact_now: np.ndarray,
        planned_now: np.ndarray,
        release_alpha: np.ndarray,
        roll_now: float,
        pitch_now: float,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        u = np.asarray(u_cmd, dtype=float).reshape(4, 3).copy()
        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        for leg in range(4):
            if contact_now[leg]:
                if not self.prev_contact[leg]:
                    self.stance_age[leg] = 1
                else:
                    self.stance_age[leg] += 1
                ramp = min(1.0, self.stance_age[leg] / float(max(cfg.stance_ramp_steps, 1)))
                u[leg, :] *= ramp
            else:
                self.stance_age[leg] = 0
                u[leg, :] = 0.0
        self.prev_contact = contact_now.copy()
        fy_scale = float(cfg.fy_scale)
        dynamic_fy_roll_gain = float(max(cfg.dynamic_fy_roll_gain, 0.0))
        if dynamic_fy_roll_gain > 1e-9:
            roll_ratio = float(
                np.clip(abs(float(roll_now)) / max(float(cfg.dynamic_fy_roll_ref), 1e-6), 0.0, 1.0)
            )
            fy_scale = min(1.0, fy_scale + dynamic_fy_roll_gain * roll_ratio)
        u[:, 1] *= fy_scale
        u = self._scale_latched_support(u, contact_now, planned_now, release_alpha, pitch_now, cfg).reshape(4, 3)
        u[:, 2] = np.clip(u[:, 2], cfg.grf_min, cfg.grf_max)
        u = self._apply_contact_floors(u, contact_now, planned_now, cfg).reshape(4, 3)
        u = self._apply_pitch_rebalance(u, contact_now, pitch_now, cfg).reshape(4, 3)
        return u.reshape(NU)

    @staticmethod
    def _support_guess(contact_now: np.ndarray, cfg: LinearOSQPConfig) -> np.ndarray:
        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        n_stance = int(np.count_nonzero(contact_now))
        if n_stance <= 0:
            return np.zeros(NU, dtype=float)
        u = np.zeros((4, 3), dtype=float)
        u[contact_now, 2] = (cfg.mass * cfg.gravity) / float(n_stance)
        return u.reshape(NU)

    @staticmethod
    def _solve_regularized(system: np.ndarray, target: np.ndarray, reg: float, reference: np.ndarray) -> np.ndarray:
        h = system.T @ system + reg * np.eye(system.shape[1], dtype=float)
        rhs = system.T @ target + reg * reference
        try:
            return np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            return reference.copy()

    @staticmethod
    def _transfer_group_load(
        values: np.ndarray,
        receiver_idx: np.ndarray,
        donor_idx: np.ndarray,
        required_total: float,
        lower_bounds: np.ndarray,
    ) -> np.ndarray:
        if receiver_idx.size == 0 or donor_idx.size == 0:
            return values
        out = values.copy()
        receiver_total = float(out[receiver_idx].sum())
        if receiver_total >= required_total:
            return out
        donor_slack = np.maximum(out[donor_idx] - lower_bounds[donor_idx], 0.0)
        available = float(donor_slack.sum())
        needed = min(required_total - receiver_total, available)
        if needed <= 1e-9:
            return out
        if available > 1e-9:
            out[donor_idx] -= needed * (donor_slack / available)
        receiver_weights = out[receiver_idx].copy()
        if float(receiver_weights.sum()) <= 1e-9:
            receiver_weights = np.ones(receiver_idx.size, dtype=float)
        out[receiver_idx] += needed * (receiver_weights / float(receiver_weights.sum()))
        return out

    @staticmethod
    def _redistribute_vertical_load(
        fz_active: np.ndarray,
        active_legs: np.ndarray,
        foot_rel_world: np.ndarray,
        theta_err: np.ndarray,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        if active_legs.size <= 1:
            return fz_active

        total_fz = float(fz_active.sum())
        support_floor = 0.0
        if cfg.support_force_floor_ratio > 0.0:
            support_floor = (cfg.support_force_floor_ratio * cfg.mass * cfg.gravity) / float(active_legs.size)
        lower = np.full_like(fz_active, max(cfg.grf_min, support_floor), dtype=float)
        fz_active = np.maximum(fz_active, lower)

        local_front = np.flatnonzero(active_legs < 2)
        local_rear = np.flatnonzero(active_legs >= 2)
        if local_front.size > 0 and local_rear.size > 0:
            rear_nominal = total_fz * float(local_rear.size) / float(active_legs.size)
            rear_floor_scale = float(cfg.rear_floor_base_scale) + float(cfg.rear_floor_pitch_gain) * np.clip(
                abs(theta_err[1]) / 0.35, 0.0, 1.0
            )
            rear_floor = rear_floor_scale * rear_nominal
            fz_active = LinearSRBDController._transfer_group_load(fz_active, local_rear, local_front, rear_floor, lower)

        local_left = np.flatnonzero(np.isin(active_legs, (0, 2)))
        local_right = np.flatnonzero(np.isin(active_legs, (1, 3)))
        if local_left.size > 0 and local_right.size > 0:
            side_nominal = total_fz * min(local_left.size, local_right.size) / float(active_legs.size)
            side_floor_scale = 0.40 + 0.10 * np.clip(abs(theta_err[0]) / 0.45, 0.0, 1.0)
            left_floor = side_floor_scale * side_nominal
            right_floor = side_floor_scale * side_nominal
            fz_active = LinearSRBDController._transfer_group_load(fz_active, local_left, local_right, left_floor, lower)
            fz_active = LinearSRBDController._transfer_group_load(fz_active, local_right, local_left, right_floor, lower)
            side_rebalance_gain = float(max(cfg.side_rebalance_gain, 0.0))
            if side_rebalance_gain > 1e-9:
                roll_ratio = float(
                    np.clip(
                        float(theta_err[0]) / max(float(cfg.side_rebalance_ref), 1e-6),
                        -1.0,
                        1.0,
                    )
                )
                extra_transfer = side_rebalance_gain * abs(roll_ratio) * side_nominal
                # Body-frame +y points to the robot's left. With the standard
                # right-hand convention, positive roll means the robot is
                # falling toward the right side, so the extra load should be
                # shifted toward the right support legs rather than away from
                # them.
                if roll_ratio > 0.0:
                    right_target = float(fz_active[local_right].sum()) + extra_transfer
                    fz_active = LinearSRBDController._transfer_group_load(
                        fz_active,
                        local_right,
                        local_left,
                        right_target,
                        lower,
                    )
                elif roll_ratio < 0.0:
                    left_target = float(fz_active[local_left].sum()) + extra_transfer
                    fz_active = LinearSRBDController._transfer_group_load(
                        fz_active,
                        local_left,
                        local_right,
                        left_target,
                        lower,
                    )

        total_after = float(fz_active.sum())
        if total_after > 1e-9:
            fz_active *= total_fz / total_after
        return np.clip(fz_active, lower, cfg.grf_max)

    @staticmethod
    def _upcoming_support_center(
        foot_rel_world: np.ndarray,
        planned_schedule: np.ndarray | None,
        stage_index: int,
        current_support_count: int,
        lookahead_steps: int,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        if planned_schedule is None or lookahead_steps <= 0:
            return None, 0.0, None

        planned_schedule = np.asarray(planned_schedule, dtype=bool)
        if planned_schedule.ndim != 2 or planned_schedule.shape[1] != 4:
            return None, 0.0, None

        last_stage = min(planned_schedule.shape[0] - 1, int(stage_index) + int(lookahead_steps))
        current_contact = np.asarray(planned_schedule[int(stage_index)], dtype=bool).reshape(4)
        for future_idx in range(int(stage_index) + 1, last_stage + 1):
            future_contact = np.asarray(planned_schedule[future_idx], dtype=bool).reshape(4)
            future_legs = np.flatnonzero(future_contact)
            if 0 < future_legs.size < int(current_support_count):
                proximity = float(last_stage - future_idx + 1) / float(max(int(lookahead_steps), 1))
                future_center = np.mean(np.asarray(foot_rel_world[future_legs, 0:2], dtype=float), axis=0)
                upcoming_swing_legs = np.flatnonzero(np.logical_and(current_contact, np.logical_not(future_contact)))
                return future_center, float(np.clip(proximity, 0.0, 1.0)), upcoming_swing_legs

        return None, 0.0, None

    @staticmethod
    def _balance_reference(
        x_init: np.ndarray,
        x_ref: np.ndarray,
        foot_rel_world: np.ndarray,
        contact_now: np.ndarray,
        cfg: LinearOSQPConfig,
        planned_schedule: np.ndarray | None = None,
        stage_index: int = 0,
    ) -> np.ndarray:
        contact_now = np.asarray(contact_now, dtype=bool).reshape(4)
        active_legs = np.flatnonzero(contact_now)
        if active_legs.size == 0:
            return np.zeros(NU, dtype=float)

        u_guess = LinearSRBDController._support_guess(contact_now, cfg).reshape(4, 3)
        ori = np.asarray(x_init[IDX_TH], dtype=float).reshape(3)
        body_from_world = Rotation.from_euler("xyz", ori).as_matrix().T

        vel_err = np.asarray(x_ref[IDX_V] - x_init[IDX_V], dtype=float).reshape(3)
        theta_err = _wrap_angle(np.asarray(x_ref[IDX_TH] - x_init[IDX_TH], dtype=float).reshape(3))
        ang_err = np.asarray(x_ref[IDX_W] - x_init[IDX_W], dtype=float).reshape(3)
        z_err = float(x_ref[2] - x_init[2])
        support_center = np.mean(np.asarray(foot_rel_world[active_legs, 0:2], dtype=float), axis=0)

        desired_force = np.array(
            [
                cfg.mass * (float(cfg.vx_gain) * vel_err[0]),
                cfg.mass * (float(cfg.vy_gain) * vel_err[1]),
                cfg.mass * (cfg.gravity + float(cfg.z_pos_gain) * z_err + float(cfg.z_vel_gain) * vel_err[2]),
            ],
            dtype=float,
        )
        if active_legs.size < 4:
            desired_force[0] += cfg.mass * float(cfg.support_centroid_x_gain) * float(support_center[0])
            desired_force[1] += cfg.mass * float(cfg.support_centroid_y_gain) * float(support_center[1])
            reduced_support_ratio = float(4 - active_legs.size) / 3.0
            desired_force[2] += cfg.mass * cfg.gravity * float(cfg.reduced_support_vertical_boost) * reduced_support_ratio
        elif cfg.pre_swing_lookahead_steps > 0 and (abs(float(cfg.support_centroid_x_gain)) > 1e-9 or abs(float(cfg.support_centroid_y_gain)) > 1e-9):
            upcoming_center, proximity, upcoming_swing_legs = LinearSRBDController._upcoming_support_center(
                foot_rel_world,
                planned_schedule,
                stage_index,
                int(active_legs.size),
                int(cfg.pre_swing_lookahead_steps),
            )
            if upcoming_center is not None and proximity > 0.0:
                shift_scale = 1.0
                if upcoming_swing_legs is not None and np.size(upcoming_swing_legs) > 0:
                    upcoming_swing_legs = np.asarray(upcoming_swing_legs, dtype=int).reshape(-1)
                    if np.any(upcoming_swing_legs < 2):
                        shift_scale = float(cfg.pre_swing_front_shift_scale)
                    elif np.any(upcoming_swing_legs >= 2):
                        shift_scale = float(cfg.pre_swing_rear_shift_scale)
                desired_force[0] += cfg.mass * float(cfg.support_centroid_x_gain) * shift_scale * proximity * float(upcoming_center[0])
                desired_force[1] += cfg.mass * float(cfg.support_centroid_y_gain) * shift_scale * proximity * float(upcoming_center[1])
        desired_force[2] = max(desired_force[2], float(cfg.min_vertical_force_scale) * cfg.mass * cfg.gravity)

        desired_ang_acc = np.array(
            [
                float(cfg.roll_angle_gain) * theta_err[0] + float(cfg.roll_rate_gain) * ang_err[0],
                float(cfg.pitch_angle_gain) * theta_err[1] + float(cfg.pitch_rate_gain) * ang_err[1],
                float(cfg.yaw_angle_gain) * theta_err[2] + float(cfg.yaw_rate_gain) * ang_err[2],
            ],
            dtype=float,
        )
        desired_torque = cfg.inertia @ desired_ang_acc
        desired_wrench = np.concatenate([desired_force, desired_torque])

        a = np.zeros((6, 3 * active_legs.size), dtype=float)
        b = desired_wrench.copy()
        for idx, leg in enumerate(active_legs):
            cols = slice(3 * idx, 3 * idx + 3)
            r_i = np.asarray(foot_rel_world[leg], dtype=float).reshape(3)
            a[0:3, cols] = np.eye(3)
            a[3:6, cols] = body_from_world @ _skew(r_i)

        u_active = LinearSRBDController._solve_regularized(
            a,
            b,
            reg=2e-2,
            reference=u_guess[active_legs].reshape(-1),
        ).reshape(-1, 3)
        u_active[:, 2] = LinearSRBDController._redistribute_vertical_load(
            np.clip(u_active[:, 2], cfg.grf_min, cfg.grf_max),
            active_legs,
            foot_rel_world,
            theta_err,
            cfg,
        )

        for idx in range(u_active.shape[0]):
            fz = float(np.clip(u_active[idx, 2], max(cfg.grf_min, 1e-6), cfg.grf_max))
            fx_max = cfg.mu * fz
            fy_max = cfg.mu * fz
            u_active[idx, 0] = np.clip(u_active[idx, 0], -fx_max, fx_max)
            u_active[idx, 1] = np.clip(u_active[idx, 1], -fy_max, fy_max)
            u_active[idx, 2] = fz

        mix = float(np.clip(cfg.support_reference_mix, 0.0, 1.0))
        xy_mix = mix if cfg.support_reference_xy_mix is None else float(np.clip(cfg.support_reference_xy_mix, 0.0, 1.0))
        u = u_guess.copy()
        u[active_legs, 0:2] = (1.0 - xy_mix) * u_guess[active_legs, 0:2] + xy_mix * u_active[:, 0:2]
        u[active_legs, 2] = (1.0 - mix) * u_guess[active_legs, 2] + mix * u_active[:, 2]
        return u.reshape(NU)

    def _support_reference(
        self,
        x_init: np.ndarray,
        x_ref: np.ndarray,
        foot_rel_world: np.ndarray,
        contact_schedule: np.ndarray,
        planned_schedule: np.ndarray | None,
        cfg: LinearOSQPConfig,
    ) -> np.ndarray:
        u_ref = np.zeros((cfg.horizon, NU), dtype=float)
        for k in range(cfg.horizon):
            u_ref[k, :] = self._balance_reference(
                x_init,
                x_ref[k],
                foot_rel_world,
                contact_schedule[k],
                cfg,
                planned_schedule=planned_schedule,
                stage_index=k,
            )
        return u_ref

    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray,
        inertia: np.ndarray,
        external_wrenches: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        cfg = self._make_cfg(inertia)
        x_init = self._state_to_vector(state_current, gravity=cfg.gravity)
        x_ref = self._rollout_reference(x_init, ref_state, cfg)
        foot_rel_world = self._foot_rel_world(state_current)
        ad_list, bd_list = self._prediction_model(x_ref, foot_rel_world, cfg)
        contact_schedule = self._contact_sequence_to_schedule(contact_sequence, cfg.horizon)
        planned_schedule = self._planned_contact_schedule(ref_state, contact_schedule, cfg.horizon)
        release_alpha = self._latched_release_alpha_now(ref_state)
        u_ref = self._support_reference(x_init, x_ref, foot_rel_world, contact_schedule, planned_schedule, cfg)
        footholds = self._extract_footholds(ref_state)

        p, q_vec, a, l, u = self._build_qp(x_init, x_ref, ad_list, bd_list, contact_schedule, u_ref, cfg)

        startup_support = (not np.any(self.prev_contact)) and (not np.any(self.last_u))
        u_prev = self.last_u.copy()
        if startup_support:
            u_prev = self._support_guess(contact_schedule[0], cfg)
            self.prev_contact = np.asarray(contact_schedule[0], dtype=bool).copy()
            self.stance_age = np.where(self.prev_contact, max(cfg.stance_ramp_steps, 1), 0)

        warm = np.zeros(((cfg.horizon + 1) * NX + cfg.horizon * NU,), dtype=float)
        if u_ref.size == cfg.horizon * NU:
            warm[(cfg.horizon + 1) * NX :] = u_ref.reshape(-1)
        if u_prev.size == NU:
            warm[(cfg.horizon + 1) * NX : (cfg.horizon + 1) * NX + NU] = u_prev

        try:
            z, status = self._solve_osqp(p, q_vec, a, l, u, warm)
            u0_raw = z[(cfg.horizon + 1) * NX : (cfg.horizon + 1) * NX + NU].reshape(NU)
            u0 = self._apply_slew_and_smoothing(u0_raw, u_prev, cfg)
            u0 = self._apply_contact_conditioning(
                u0,
                contact_schedule[0],
                planned_schedule[0],
                release_alpha,
                float(x_init[IDX_TH][0]),
                float(x_init[IDX_TH][1]),
                cfg,
            )
            self.last_u = u0.copy()
            self.last_status = status
        except Exception as exc:  # pragma: no cover - fallback path for runtime robustness
            u0 = self._apply_contact_conditioning(
                u_prev.copy(),
                contact_schedule[0],
                planned_schedule[0],
                release_alpha,
                float(x_init[IDX_TH][0]),
                float(x_init[IDX_TH][1]),
                cfg,
            )
            self.last_u = u0.copy()
            self.last_status = f"fallback:{type(exc).__name__}:{exc}"

        x_pred = ad_list[0] @ x_init + bd_list[0] @ u0
        return u0, footholds, x_pred[:12], self.last_status
