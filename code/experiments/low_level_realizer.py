
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
import mujoco

from baseline.config import IDX_G, IDX_P, IDX_TH, IDX_V, IDX_W, LEG_NAMES, MPCConfig


FALLBACK_FOOT_LOCAL = np.array([0.0, 0.0, -0.20], dtype=float)


@dataclass
class LegBinding:
    leg: str
    leg_idx: int
    calf_body_name: str
    calf_body_id: int
    joint_ids: list[int]
    actuator_ids: list[int]
    qpos_adrs: list[int]
    dof_adrs: list[int]
    subtree_body_ids: set[int]
    geom_ids: set[int]
    foot_geom_id: int | None = None
    foot_geom_local: np.ndarray | None = None
    home_joint_qpos: np.ndarray | None = None

    stance_anchor_world: np.ndarray | None = None
    swing_start_world: np.ndarray | None = None
    swing_target_world: np.ndarray | None = None
    touchdown_target_world: np.ndarray | None = None
    swing_started: bool = False
    swing_start_time: float | None = None


@dataclass
class ModelBindings:
    base_body_name: str
    base_body_id: int
    floor_geom_ids: set[int]
    leg_bindings: list[LegBinding]


@dataclass
class CleanLowLevelParams:
    schedule: str = "crawl"
    crawl_order: tuple[int, int, int, int] = (1, 2, 0, 3)  # FR, RL, FL, RR
    crawl_phase_duration: float = 0.34
    crawl_swing_duration: float = 0.18

    settle_time: float = 0.6
    gait_ramp_time: float = 1.2
    desired_speed_cap: float = 0.12

    support_enabled: bool = True
    support_target_height_frac: float = 0.90
    support_target_pitch: float = 0.0
    support_target_roll: float = 0.0
    support_weight_start: float = 0.90
    support_weight_end: float = 0.10
    support_fade_start: float = 0.5
    support_fade_end: float = 1.8
    support_height_k: float = 380.0
    support_height_d: float = 48.0
    support_roll_k: float = 16.0
    support_roll_d: float = 2.0
    support_pitch_k: float = 14.0
    support_pitch_d: float = 1.8
    support_yaw_k: float = 4.0
    support_yaw_d: float = 0.8
    support_max_force_z: float = 160.0
    support_max_torque: float = 14.0

    mpc_force_gain_start: float = 0.00
    mpc_force_gain_end: float = 0.28
    mpc_force_ramp_start: float = 0.4
    mpc_force_ramp_end: float = 1.4
    force_frame: str = "body"
    realization: str = "external"
    max_xy_over_fz: float = 0.40

    clearance: float = 0.070
    step_len_front: float = 0.055
    rear_step_scale: float = 0.90
    touchdown_depth_front: float = 0.020
    touchdown_depth_rear: float = 0.025
    touchdown_window_front: float = 0.08
    touchdown_window_rear: float = 0.10
    dq_limit: float = 0.16

    target_height_frac: float = 0.95
    stance_press_front: float = 0.010
    stance_press_rear: float = 0.010
    stance_drive_front: float = 0.003
    stance_drive_rear: float = 0.004
    front_unload: float = -0.001
    height_k: float = 1.0
    pitch_k: float = 0.04
    roll_k: float = 0.03
    pitch_sign: float = -1.0
    roll_sign: float = 1.0

    visual_step_boost: float = 1.0

    disable_nonfoot_collision: bool = True


def mj_name_or_none(m: mujoco.MjModel, objtype, idx: int) -> str | None:
    name = mujoco.mj_id2name(m, objtype, idx)
    return None if name is None else str(name)


def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw], dtype=float)


def body_pose_velocity(m: mujoco.MjModel, d: mujoco.MjData, base_body_id: int):
    body = d.body(base_body_id)
    pos = np.asarray(body.xpos, dtype=float).copy()
    R = np.asarray(body.xmat, dtype=float).reshape(3, 3).copy()
    rpy = mat_to_rpy(R)
    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(base_body_id), vel6, 0)
    ang = vel6[0:3].copy()
    lin = vel6[3:6].copy()
    return pos, R, rpy, lin, ang


def state_to_x(m: mujoco.MjModel, d: mujoco.MjData, cfg: MPCConfig, base_body_id: int) -> np.ndarray:
    x = np.zeros(cfg.nx, dtype=float)
    pos, R, rpy, lin, ang = body_pose_velocity(m, d, base_body_id)
    x[IDX_P] = pos
    x[IDX_V] = lin
    x[IDX_TH] = rpy
    x[IDX_W] = ang
    x[IDX_G] = cfg.g
    return x


def _collect_subtree_bodies(m: mujoco.MjModel, root_body_id: int) -> set[int]:
    children: dict[int, list[int]] = {bid: [] for bid in range(m.nbody)}
    for bid in range(1, m.nbody):
        parent = int(m.body_parentid[bid])
        if 0 <= parent < m.nbody and parent != bid:
            children[parent].append(bid)
    out: set[int] = set()
    stack = [int(root_body_id)]
    while stack:
        bid = stack.pop()
        if bid in out:
            continue
        out.add(bid)
        stack.extend(children.get(bid, []))
    return out


def _geom_collision_enabled(m: mujoco.MjModel, gid: int) -> bool:
    return int(m.geom_contype[gid]) != 0 or int(m.geom_conaffinity[gid]) != 0


def find_base_body_name(m: mujoco.MjModel) -> tuple[str, int]:
    candidates = ["trunk", "base", "torso", "body", "base_link"]
    for target in candidates:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
        if bid != -1:
            return target, bid
    for i in range(m.nbody):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name and any(tok in name.lower() for tok in ["trunk", "base", "torso"]):
            return name, i
    raise RuntimeError("Could not identify the floating base body.")


def _find_leg_calf_body(m: mujoco.MjModel, leg: str) -> tuple[str, int]:
    preferred = [
        f"{leg}_calf", f"{leg}_lower_leg", f"{leg}_shank",
        f"{leg.lower()}_calf", f"{leg.lower()}_lower_leg", f"{leg.lower()}_shank",
    ]
    for name in preferred:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return name, bid
    for i in range(m.nbody):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        low = name.lower()
        if low.startswith(leg.lower()) and any(tok in low for tok in ["calf", "lower", "shank"]):
            return name, i
    raise RuntimeError(f"Could not identify calf body for {leg}.")


def _joint_sort_key(name: str) -> tuple[int, str]:
    low = name.lower()
    if any(tok in low for tok in ["abd", "abduction", "ab_ad"]):
        return (0, low)
    if any(tok in low for tok in ["hip", "thigh"]):
        return (1, low)
    if any(tok in low for tok in ["knee", "calf"]):
        return (2, low)
    return (99, low)


def _find_leg_joint_ids(m: mujoco.MjModel, leg: str) -> list[int]:
    joint_items = []
    for jid in range(m.njnt):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if name.lower().startswith(leg.lower()):
            joint_items.append((name, jid))
    if not joint_items:
        raise RuntimeError(f"Could not identify joints for {leg}.")
    joint_items.sort(key=lambda x: _joint_sort_key(x[0]))
    return [jid for _, jid in joint_items[:3]]


def _actuator_id_for_joint(m: mujoco.MjModel, joint_id: int) -> int | None:
    for aid in range(m.nu):
        if int(m.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(m.actuator_trnid[aid, 0]) == joint_id:
            return aid
    return None


def _find_leg_actuator_ids(m: mujoco.MjModel, joint_ids: list[int]) -> list[int]:
    out = []
    for jid in joint_ids:
        aid = _actuator_id_for_joint(m, jid)
        if aid is not None:
            out.append(aid)
    return out


def _find_floor_geom_ids(m: mujoco.MjModel) -> set[int]:
    out: set[int] = set()
    for gid in range(m.ngeom):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            out.add(gid)
        elif any(tok in name.lower() for tok in ["ground", "floor", "terrain"]):
            out.add(gid)
    return out


def _select_foot_geom(m: mujoco.MjModel, calf_body_id: int, geom_ids: set[int]) -> tuple[int | None, np.ndarray | None]:
    candidates = []
    for gid in sorted(geom_ids):
        if int(m.geom_bodyid[gid]) != int(calf_body_id):
            continue
        if not _geom_collision_enabled(m, gid):
            continue
        pos = np.asarray(m.geom_pos[gid], dtype=float).copy()
        gtype = int(m.geom_type[gid])
        sphere_bonus = -1.0 if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE) else 0.0
        candidates.append((float(pos[2]), sphere_bonus, gid, pos))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, gid, pos = candidates[0]
    return int(gid), pos


def discover_model_bindings(m: mujoco.MjModel) -> ModelBindings:
    base_name, base_id = find_base_body_name(m)
    floor_ids = _find_floor_geom_ids(m)
    leg_bindings: list[LegBinding] = []
    for leg_idx, leg in enumerate(LEG_NAMES):
        calf_name, calf_id = _find_leg_calf_body(m, leg)
        joint_ids = _find_leg_joint_ids(m, leg)
        actuator_ids = _find_leg_actuator_ids(m, joint_ids)
        qpos_adrs = [int(m.jnt_qposadr[j]) for j in joint_ids]
        dof_adrs = [int(m.jnt_dofadr[j]) for j in joint_ids]
        subtree_body_ids = _collect_subtree_bodies(m, calf_id)
        geom_ids = {gid for gid in range(m.ngeom) if int(m.geom_bodyid[gid]) in subtree_body_ids}
        foot_gid, foot_local = _select_foot_geom(m, calf_id, geom_ids)
        leg_bindings.append(
            LegBinding(
                leg=leg,
                leg_idx=leg_idx,
                calf_body_name=calf_name,
                calf_body_id=calf_id,
                joint_ids=joint_ids,
                actuator_ids=actuator_ids,
                qpos_adrs=qpos_adrs,
                dof_adrs=dof_adrs,
                subtree_body_ids=subtree_body_ids,
                geom_ids=geom_ids,
                foot_geom_id=foot_gid,
                foot_geom_local=foot_local,
            )
        )
    return ModelBindings(base_name, base_id, floor_ids, leg_bindings)


def print_binding_summary(bindings: ModelBindings) -> str:
    lines = [f"Base body: {bindings.base_body_name}", f"Floor geoms: {sorted(bindings.floor_geom_ids)}"]
    for leg in bindings.leg_bindings:
        lines.append(
            f"{leg.leg}: calf={leg.calf_body_name}, joints={leg.joint_ids}, actuators={leg.actuator_ids}, "
            f"geoms={sorted(leg.geom_ids)}, foot_geom={leg.foot_geom_id}, "
            f"foot_local={None if leg.foot_geom_local is None else np.round(leg.foot_geom_local, 4)}"
        )
    return "\n".join(lines)


def disable_nonfoot_leg_collisions(m: mujoco.MjModel, bindings: ModelBindings):
    disabled = []
    for leg in bindings.leg_bindings:
        for gid in sorted(leg.geom_ids):
            if gid == leg.foot_geom_id:
                continue
            if int(m.geom_bodyid[gid]) != leg.calf_body_id:
                continue
            if _geom_collision_enabled(m, gid):
                m.geom_contype[gid] = 0
                m.geom_conaffinity[gid] = 0
                disabled.append((leg.leg, gid))
    return disabled


def foot_point_world(d: mujoco.MjData, leg: LegBinding) -> np.ndarray:
    if leg.foot_geom_id is not None:
        return np.asarray(d.geom_xpos[leg.foot_geom_id], dtype=float).copy()
    body = d.body(leg.calf_body_name)
    R = np.asarray(body.xmat, dtype=float).reshape(3, 3)
    return np.asarray(body.xpos, dtype=float).copy() + R @ FALLBACK_FOOT_LOCAL


def foot_rel_world(d: mujoco.MjData, bindings: ModelBindings) -> np.ndarray:
    p_base = np.asarray(d.body(bindings.base_body_name).xpos, dtype=float).copy()
    feet = [foot_point_world(d, leg) - p_base for leg in bindings.leg_bindings]
    return np.vstack(feet)


def actual_foot_contact_state(m: mujoco.MjModel, d: mujoco.MjData, bindings: ModelBindings) -> np.ndarray:
    contact = np.zeros(4, dtype=bool)
    floor = bindings.floor_geom_ids
    if not floor:
        return contact
    for ci in range(d.ncon):
        con = d.contact[ci]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        for i, leg in enumerate(bindings.leg_bindings):
            if leg.foot_geom_id is None:
                ids = leg.geom_ids
            else:
                ids = {leg.foot_geom_id}
            if (g1 in ids and g2 in floor) or (g2 in ids and g1 in floor):
                contact[i] = True
    return contact


def store_home_qpos(d: mujoco.MjData, bindings: ModelBindings):
    for leg in bindings.leg_bindings:
        leg.home_joint_qpos = np.array([d.qpos[a] for a in leg.qpos_adrs], dtype=float)
        leg.stance_anchor_world = foot_point_world(d, leg).copy()
        leg.swing_start_world = foot_point_world(d, leg).copy()
        leg.swing_target_world = foot_point_world(d, leg).copy()
        leg.touchdown_target_world = foot_point_world(d, leg).copy()


def leg_jacobian_world(m: mujoco.MjModel, d: mujoco.MjData, leg: LegBinding) -> np.ndarray:
    point = foot_point_world(d, leg)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point, int(leg.calf_body_id))
    return jacp[:, np.asarray(leg.dof_adrs, dtype=int)]


def damped_ls_step(J: np.ndarray, err: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    A = J @ J.T + damping * np.eye(3)
    return J.T @ np.linalg.solve(A, err)


def clip_targets_to_ctrlrange(m: mujoco.MjModel, actuator_ids: Iterable[int], target: np.ndarray) -> np.ndarray:
    tgt = np.asarray(target, dtype=float).copy()
    aids = list(actuator_ids)
    if not aids:
        return tgt
    for i, aid in enumerate(aids):
        if int(m.actuator_ctrllimited[aid]):
            lo, hi = m.actuator_ctrlrange[aid]
            tgt[i] = float(np.clip(tgt[i], lo, hi))
    return tgt


def front_sign(leg_idx: int) -> float:
    return 1.0 if leg_idx in (0, 1) else -1.0


def side_sign(leg_idx: int) -> float:
    return 1.0 if leg_idx in (0, 2) else -1.0


def is_rear(leg_idx: int) -> bool:
    return leg_idx in (2, 3)


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def crawl_leg_state(t_eff: float, phase_duration: float, swing_duration: float, order: tuple[int, int, int, int]) -> tuple[np.ndarray, int | None, float | None]:
    cycle = 4.0 * phase_duration
    tau_cycle = float(np.mod(t_eff, cycle))
    slot = int(np.floor(tau_cycle / phase_duration))
    tau_slot = tau_cycle - slot * phase_duration
    sched = np.ones(4, dtype=bool)
    swing_leg = None
    swing_s = None
    if tau_slot < swing_duration:
        swing_leg = int(order[slot])
        swing_s = float(np.clip(tau_slot / max(swing_duration, 1e-6), 0.0, 1.0))
        sched[swing_leg] = False
    return sched, swing_leg, swing_s


def rollout_contact_schedule_crawl(t_eff: float, cfg: MPCConfig, phase_duration: float, swing_duration: float, order: tuple[int, int, int, int]) -> np.ndarray:
    out = np.ones((cfg.horizon, 4), dtype=bool)
    for k in range(cfg.horizon):
        sched, _, _ = crawl_leg_state(t_eff + k * cfg.dt_mpc, phase_duration, swing_duration, order)
        out[k] = sched
    return out


def rollout_contact_schedule_trot(t_eff: float, cfg: MPCConfig) -> np.ndarray:
    cycle = cfg.stance_time + cfg.swing_time
    stance_frac = cfg.stance_time / cycle
    phase_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)
    times = t_eff + cfg.dt_mpc * np.arange(cfg.horizon)
    rows = []
    for t in times:
        phase = ((t / cycle) + phase_offsets) % 1.0
        rows.append((phase < stance_frac).astype(bool))
    return np.vstack(rows)


def nominal_foothold_world(d: mujoco.MjData, bindings: ModelBindings, cfg: MPCConfig, leg_idx: int, gait_alpha: float, step_len_front: float, rear_step_scale: float, desired_vx: float) -> np.ndarray:
    body = np.asarray(cfg.nominal_footholds_body[leg_idx], dtype=float).copy()
    step_len = float(step_len_front * (rear_step_scale if is_rear(leg_idx) else 1.0))
    body[0] += gait_alpha * np.sign(desired_vx) * abs(step_len)
    base = d.body(bindings.base_body_name)
    R = np.asarray(base.xmat, dtype=float).reshape(3, 3)
    p = np.asarray(base.xpos, dtype=float)
    return p + R @ body


def swing_target_world(start_world: np.ndarray, end_world: np.ndarray, s: float, clearance: float, touchdown_depth: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    p = (1.0 - s) * np.asarray(start_world, dtype=float) + s * np.asarray(end_world, dtype=float)
    lift = float(clearance) * 4.0 * s * (1.0 - s)
    late = smoothstep01((s - 0.7) / 0.3)
    p[2] = (1.0 - s) * start_world[2] + s * end_world[2] + lift - touchdown_depth * late
    return p


class CleanLowLevelRealizer:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, bindings: ModelBindings, cfg: MPCConfig, params: CleanLowLevelParams):
        self.m = m
        self.d = d
        self.bindings = bindings
        self.cfg = cfg
        self.params = params
        store_home_qpos(d, bindings)
        self.nominal_trunk_height = float(d.body(bindings.base_body_name).xpos[2])
        self.support_target_height = float(params.support_target_height_frac * self.nominal_trunk_height)
        self.home_ctrl = self._resolve_home_ctrl()
        self.prev_sched = np.ones(4, dtype=bool)
        self.current_swing_leg = None
        self.current_swing_phase = None

    def _resolve_home_ctrl(self) -> np.ndarray:
        if self.m.nu == 0:
            return np.zeros(0, dtype=float)
        ctrl = np.zeros(self.m.nu, dtype=float)
        for leg in self.bindings.leg_bindings:
            if leg.home_joint_qpos is None or not leg.actuator_ids:
                continue
            q = clip_targets_to_ctrlrange(self.m, leg.actuator_ids, leg.home_joint_qpos)
            ctrl[np.asarray(leg.actuator_ids, dtype=int)] = q
        return ctrl

    def desired_speed(self) -> float:
        return min(float(self.cfg.desired_speed), float(self.params.desired_speed_cap))

    def support_weight(self, t: float) -> float:
        if not self.params.support_enabled:
            return 0.0
        return self._lerp(t, self.params.support_fade_start, self.params.support_fade_end,
                          self.params.support_weight_start, self.params.support_weight_end)

    def mpc_force_gain(self, t: float) -> float:
        return self._lerp(t, self.params.mpc_force_ramp_start, self.params.mpc_force_ramp_end,
                          self.params.mpc_force_gain_start, self.params.mpc_force_gain_end)

    @staticmethod
    def _lerp(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
        if t <= t0:
            return float(y0)
        if t >= t1:
            return float(y1)
        a = (t - t0) / max(t1 - t0, 1e-9)
        return float((1.0 - a) * y0 + a * y1)

    def schedule_now_and_rollout(self, t: float) -> tuple[np.ndarray, int | None, float | None, np.ndarray]:
        if self.params.schedule == "crawl":
            eff_time = max(0.0, t - self.params.settle_time)
            now_sched, swing_leg, swing_s = crawl_leg_state(
                eff_time, self.params.crawl_phase_duration, self.params.crawl_swing_duration, self.params.crawl_order
            )
            rollout = rollout_contact_schedule_crawl(
                eff_time, self.cfg, self.params.crawl_phase_duration, self.params.crawl_swing_duration, self.params.crawl_order
            )
            return now_sched, swing_leg, swing_s, rollout

        # fallback: trot schedule using config stance/swing times
        eff_time = max(0.0, t - self.params.settle_time)
        cycle = self.cfg.stance_time + self.cfg.swing_time
        stance_frac = self.cfg.stance_time / cycle
        phase_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)
        phase = ((eff_time / cycle) + phase_offsets) % 1.0
        now_sched = (phase < stance_frac).astype(bool)
        swing_leg = None
        swing_s = None
        if not np.all(now_sched):
            swing_leg = int(np.where(~now_sched)[0][0])
            swing_s = float((phase[swing_leg] - stance_frac) / max(1.0 - stance_frac, 1e-9))
        rollout = rollout_contact_schedule_trot(eff_time, self.cfg)
        return now_sched, swing_leg, swing_s, rollout

    def update_phase_state(self, scheduled_contact: np.ndarray, gait_alpha: float):
        desired_vx = self.desired_speed()
        for leg in self.bindings.leg_bindings:
            idx = leg.leg_idx
            was_stance = bool(self.prev_sched[idx])
            is_stance = bool(scheduled_contact[idx])
            p_now = foot_point_world(self.d, leg)

            if was_stance and not is_stance:
                leg.swing_started = True
                leg.swing_start_time = float(self.d.time)
                leg.swing_start_world = p_now.copy()
                leg.swing_target_world = nominal_foothold_world(
                    self.d, self.bindings, self.cfg, idx, gait_alpha,
                    self.params.step_len_front * self.params.visual_step_boost,
                    self.params.rear_step_scale,
                    desired_vx,
                )
                leg.touchdown_target_world = leg.swing_target_world.copy()

            if (not was_stance) and is_stance:
                leg.stance_anchor_world = p_now.copy()
                leg.swing_started = False

        self.prev_sched = scheduled_contact.copy()

    def build_control_targets(self, scheduled_contact: np.ndarray, actual_contact: np.ndarray, gait_alpha: float) -> np.ndarray:
        ctrl = self.home_ctrl.copy()
        pos, R, rpy, lin_vel, ang_vel = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        roll, pitch, yaw = rpy
        target_height = self.params.target_height_frac * self.nominal_trunk_height

        for leg in self.bindings.leg_bindings:
            if not leg.actuator_ids or leg.home_joint_qpos is None:
                continue
            idx = leg.leg_idx
            current_q = np.array([self.d.qpos[a] for a in leg.qpos_adrs], dtype=float)
            p_now = foot_point_world(self.d, leg)
            J = leg_jacobian_world(self.m, self.d, leg)
            front = front_sign(idx)
            side = side_sign(idx)
            dq = np.zeros(3, dtype=float)

            if bool(scheduled_contact[idx]):
                target = leg.stance_anchor_world.copy() if leg.stance_anchor_world is not None else p_now.copy()
                dz = 0.0
                dz -= self.params.height_k * (target_height - pos[2])
                dz += self.params.pitch_sign * self.params.pitch_k * front * pitch
                dz += self.params.roll_sign * self.params.roll_k * side * roll
                dz -= self.params.stance_press_rear if is_rear(idx) else self.params.stance_press_front
                if not is_rear(idx):
                    dz += self.params.front_unload
                target[2] += dz

                drive = self.params.stance_drive_rear if is_rear(idx) else self.params.stance_drive_front
                target[0] -= drive * np.sign(self.desired_speed()) * gait_alpha

                err = target - p_now
                dq = damped_ls_step(J, err, damping=1e-4)
            else:
                if leg.swing_target_world is None or leg.swing_start_world is None:
                    leg.swing_start_world = p_now.copy()
                    leg.swing_target_world = nominal_foothold_world(
                        self.d, self.bindings, self.cfg, idx, gait_alpha,
                        self.params.step_len_front * self.params.visual_step_boost,
                        self.params.rear_step_scale,
                        self.desired_speed(),
                    )
                # estimate swing phase if not explicitly tracked
                if leg.swing_start_time is None:
                    s = 0.0
                else:
                    elapsed = max(0.0, float(self.d.time) - float(leg.swing_start_time))
                    s = min(1.0, elapsed / max(self.params.crawl_swing_duration, 1e-6))
                touchdown_depth = self.params.touchdown_depth_rear if is_rear(idx) else self.params.touchdown_depth_front
                target = swing_target_world(
                    leg.swing_start_world,
                    leg.swing_target_world,
                    s,
                    self.params.clearance * self.params.visual_step_boost,
                    touchdown_depth,
                )
                err = target - p_now
                dq = damped_ls_step(J, err, damping=1e-4)

            dq = np.clip(dq, -self.params.dq_limit, self.params.dq_limit)
            q_target = current_q + dq
            q_target = clip_targets_to_ctrlrange(self.m, leg.actuator_ids, q_target)
            ctrl[np.asarray(leg.actuator_ids, dtype=int)] = q_target
        return ctrl

    def apply_support(self, target_yaw: float = 0.0) -> dict:
        w = self.support_weight(float(self.d.time))
        out = {"weight": w, "force_world": np.zeros(3), "torque_world": np.zeros(3)}
        if w <= 0.0:
            return out
        pos, R, rpy, lin_vel, ang_vel = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        roll, pitch, yaw = rpy
        total_mass = float(np.sum(np.asarray(self.m.body_mass, dtype=float)))
        gmag = float(abs(self.m.opt.gravity[2]))

        fz = w * (total_mass * gmag + self.params.support_height_k * (self.support_target_height - pos[2]) - self.params.support_height_d * lin_vel[2])
        fx = 0.0
        fy = 0.0
        tx = -w * (self.params.support_roll_k * (roll - self.params.support_target_roll) + self.params.support_roll_d * ang_vel[0])
        ty = -w * (self.params.support_pitch_k * (pitch - self.params.support_target_pitch) + self.params.support_pitch_d * ang_vel[1])
        yaw_err = float(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw)))
        tz = w * (self.params.support_yaw_k * yaw_err - self.params.support_yaw_d * ang_vel[2])

        force = np.array([0.0, 0.0, float(np.clip(fz, 0.0, self.params.support_max_force_z))], dtype=float)
        torque = np.array([
            float(np.clip(tx, -self.params.support_max_torque, self.params.support_max_torque)),
            float(np.clip(ty, -self.params.support_max_torque, self.params.support_max_torque)),
            float(np.clip(tz, -self.params.support_max_torque, self.params.support_max_torque)),
        ], dtype=float)

        mujoco.mj_applyFT(self.m, self.d, force, torque, pos, int(self.bindings.base_body_id), self.d.qfrc_applied)
        out.update({"force_world": force, "torque_world": torque})
        return out

    def apply_mpc_forces(self, u_hold: np.ndarray, scheduled_contact: np.ndarray, actual_contact: np.ndarray):
        g = self.mpc_force_gain(float(self.d.time))
        if g <= 0.0:
            return {"gain": g, "u_applied": np.zeros_like(u_hold), "force_enabled": np.zeros(4, dtype=bool)}
        pos, R_base, rpy, lin_vel, ang_vel = body_pose_velocity(self.m, self.d, self.bindings.base_body_id)
        u_applied = np.zeros_like(u_hold)
        force_enabled = np.zeros(4, dtype=bool)
        for leg in self.bindings.leg_bindings:
            i = leg.leg_idx
            if not bool(scheduled_contact[i]) or not bool(actual_contact[i]):
                continue
            f = np.asarray(u_hold[3*i:3*i+3], dtype=float).copy()
            if self.params.force_frame == "body":
                f = R_base @ f
            f[0:2] = np.clip(f[0:2], -self.params.max_xy_over_fz * max(f[2], 0.0), self.params.max_xy_over_fz * max(f[2], 0.0))
            f *= g
            if self.params.realization == "joint":
                J = leg_jacobian_world(self.m, self.d, leg)
                tau = J.T @ f
                for j, dof in enumerate(leg.dof_adrs):
                    self.d.qfrc_applied[dof] += tau[j]
            else:
                p = foot_point_world(self.d, leg)
                mujoco.mj_applyFT(self.m, self.d, f, np.zeros(3), p, int(leg.calf_body_id), self.d.qfrc_applied)
            u_applied[3*i:3*i+3] = f
            force_enabled[i] = True
        return {"gain": g, "u_applied": u_applied, "force_enabled": force_enabled}


def save_contact_plot(t: np.ndarray, sched: np.ndarray, actual: np.ndarray, output_path: str) -> str:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 5.8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, sched[:, i].astype(float), linewidth=2.0, label=f"{LEG_NAMES[i]} scheduled")
        ax.plot(t, actual[:, i].astype(float), "--", linewidth=1.8, label=f"{LEG_NAMES[i]} actual")
        ax.set_ylim(-0.1, 1.15)
        ax.set_ylabel("contact")
        ax.set_title(LEG_NAMES[i], loc="left", fontsize=10, pad=2)
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)
