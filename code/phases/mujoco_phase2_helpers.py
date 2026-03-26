from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import mujoco

from baseline.config import IDX_G, IDX_P, IDX_TH, IDX_V, IDX_W, LEG_NAMES, MPCConfig


@dataclass
class LegBinding:
    leg: str
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
    swing_lift_dir: np.ndarray | None = None


@dataclass
class ModelBindings:
    base_body_name: str
    base_body_id: int
    floor_geom_ids: set[int]
    leg_bindings: list[LegBinding]


def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw], dtype=float)


def mj_name_or_none(m: mujoco.MjModel, objtype, idx: int) -> str | None:
    name = mujoco.mj_id2name(m, objtype, idx)
    return None if name is None else str(name)


def find_base_body_name(m: mujoco.MjModel) -> tuple[str, int]:
    candidates = ["trunk", "base", "torso", "body", "base_link"]
    body_names = [(i, (mj_name_or_none(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")) for i in range(m.nbody)]

    for target in candidates:
        for i, name in body_names:
            if name == target:
                return name, i

    for i, name in body_names:
        if name and any(tok in name.lower() for tok in ["trunk", "base", "torso"]):
            return name, i

    raise RuntimeError("Could not identify the floating base body. Please set the base body name manually.")


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


def _find_leg_calf_body(m: mujoco.MjModel, leg: str) -> tuple[str, int]:
    preferred = [
        f"{leg}_calf",
        f"{leg}_lower_leg",
        f"{leg}_shank",
        f"{leg.lower()}_calf",
        f"{leg.lower()}_lower_leg",
        f"{leg.lower()}_shank",
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

    raise RuntimeError(f"Could not identify calf body for leg {leg}.")


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
    joint_items: list[tuple[str, int]] = []
    for jid in range(m.njnt):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        low = name.lower()
        if low.startswith(leg.lower()):
            joint_items.append((name, jid))

    if not joint_items:
        raise RuntimeError(f"Could not identify joints for leg {leg}.")

    joint_items.sort(key=lambda x: _joint_sort_key(x[0]))
    return [jid for _, jid in joint_items[:3]]


def _actuator_id_for_joint(m: mujoco.MjModel, joint_id: int) -> int | None:
    for aid in range(m.nu):
        trn_obj_id = int(m.actuator_trnid[aid, 0])
        trn_obj_kind = int(m.actuator_trntype[aid])
        if trn_obj_kind == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_obj_id == joint_id:
            return aid
    return None


def _find_leg_actuator_ids(m: mujoco.MjModel, joint_ids: list[int]) -> list[int]:
    actuator_ids: list[int] = []
    for jid in joint_ids:
        aid = _actuator_id_for_joint(m, jid)
        if aid is not None:
            actuator_ids.append(aid)
    return actuator_ids


def _find_floor_geom_ids(m: mujoco.MjModel) -> set[int]:
    out: set[int] = set()
    for gid in range(m.ngeom):
        name = mj_name_or_none(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            out.add(gid)
        elif any(tok in name.lower() for tok in ["ground", "floor", "terrain"]):
            out.add(gid)
    return out


def _geom_collision_enabled(m: mujoco.MjModel, gid: int) -> bool:
    return int(m.geom_contype[gid]) != 0 or int(m.geom_conaffinity[gid]) != 0


def _select_foot_geom(m: mujoco.MjModel, calf_body_id: int, geom_ids: set[int]) -> tuple[int | None, np.ndarray | None]:
    # Prefer collision-enabled geoms attached directly to the calf body.
    candidates: list[tuple[float, float, int, np.ndarray]] = []
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

    # Lower local z is more distal. Tie-break in favor of a sphere.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, gid, pos = candidates[0]
    return int(gid), pos


def discover_model_bindings(m: mujoco.MjModel) -> ModelBindings:
    base_body_name, base_body_id = find_base_body_name(m)
    floor_geom_ids = _find_floor_geom_ids(m)

    leg_bindings: list[LegBinding] = []
    for leg in LEG_NAMES:
        calf_body_name, calf_body_id = _find_leg_calf_body(m, leg)
        joint_ids = _find_leg_joint_ids(m, leg)
        actuator_ids = _find_leg_actuator_ids(m, joint_ids)
        qpos_adrs = [int(m.jnt_qposadr[jid]) for jid in joint_ids]
        dof_adrs = [int(m.jnt_dofadr[jid]) for jid in joint_ids]
        subtree_body_ids = _collect_subtree_bodies(m, calf_body_id)
        geom_ids = {gid for gid in range(m.ngeom) if int(m.geom_bodyid[gid]) in subtree_body_ids}
        foot_geom_id, foot_geom_local = _select_foot_geom(m, calf_body_id, geom_ids)
        leg_bindings.append(
            LegBinding(
                leg=leg,
                calf_body_name=calf_body_name,
                calf_body_id=calf_body_id,
                joint_ids=joint_ids,
                actuator_ids=actuator_ids,
                qpos_adrs=qpos_adrs,
                dof_adrs=dof_adrs,
                subtree_body_ids=subtree_body_ids,
                geom_ids=geom_ids,
                foot_geom_id=foot_geom_id,
                foot_geom_local=foot_geom_local,
            )
        )

    return ModelBindings(
        base_body_name=base_body_name,
        base_body_id=base_body_id,
        floor_geom_ids=floor_geom_ids,
        leg_bindings=leg_bindings,
    )


def foot_point_world(d: mujoco.MjData, binding: LegBinding, fallback_local_offset: np.ndarray) -> np.ndarray:
    if binding.foot_geom_id is not None:
        return np.asarray(d.geom_xpos[binding.foot_geom_id], dtype=float).copy()
    body = d.body(binding.calf_body_name)
    R = np.asarray(body.xmat, dtype=float).reshape(3, 3)
    return np.asarray(body.xpos, dtype=float).copy() + R @ fallback_local_offset


def foot_rel_world(
    d: mujoco.MjData,
    base_body_name: str,
    leg_bindings: Iterable[LegBinding],
    fallback_local_offset: np.ndarray,
) -> np.ndarray:
    p_base = np.asarray(d.body(base_body_name).xpos, dtype=float).copy()
    feet = [foot_point_world(d, binding, fallback_local_offset) - p_base for binding in leg_bindings]
    return np.vstack(feet)


def mujoco_to_x(d: mujoco.MjData, cfg: MPCConfig, base_body_name: str) -> np.ndarray:
    x = np.zeros(cfg.nx, dtype=float)
    base = d.body(base_body_name)
    x[IDX_P] = np.asarray(base.xpos, dtype=float).copy()
    x[IDX_TH] = mat_to_rpy(np.asarray(base.xmat, dtype=float).reshape(3, 3))
    x[IDX_V] = np.asarray(d.qvel[:3], dtype=float).copy()
    x[IDX_W] = np.asarray(d.qvel[3:6], dtype=float).copy()
    x[IDX_G] = cfg.g
    return x


def force_to_qfrc(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    body_name: str,
    point_world: np.ndarray,
    f_world: np.ndarray,
) -> np.ndarray:
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point_world, body_id)
    return jacp.T @ f_world


def actual_contact_state(m: mujoco.MjModel, d: mujoco.MjData, bindings: ModelBindings) -> np.ndarray:
    contact = np.zeros(4, dtype=bool)
    floor = bindings.floor_geom_ids
    if not floor:
        return contact

    for ci in range(d.ncon):
        con = d.contact[ci]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        for leg_idx, leg in enumerate(bindings.leg_bindings):
            foot_ids = {leg.foot_geom_id} if leg.foot_geom_id is not None else leg.geom_ids
            if (g1 in foot_ids and g2 in floor) or (g2 in foot_ids and g1 in floor):
                contact[leg_idx] = True
    return contact


def store_home_joint_qpos(d: mujoco.MjData, bindings: ModelBindings) -> None:
    for leg in bindings.leg_bindings:
        leg.home_joint_qpos = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)


def estimate_swing_lift_dir(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    leg: LegBinding,
    fallback_local_offset: np.ndarray,
) -> np.ndarray:
    if len(leg.dof_adrs) == 0:
        return np.zeros(0, dtype=float)

    point_world = foot_point_world(d, leg, fallback_local_offset)
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point_world, leg.calf_body_id)

    lift = []
    for dof in leg.dof_adrs:
        dz_dq = float(jacp[2, dof])
        if abs(dz_dq) < 1e-6:
            lift.append(0.0)
        else:
            lift.append(math.copysign(max(abs(dz_dq), 0.1), dz_dq))

    vec = np.asarray(lift, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        if vec.size == 3:
            vec = np.array([0.0, 0.5, -1.0], dtype=float)
        else:
            vec = np.ones_like(vec)
        norm = float(np.linalg.norm(vec))
    return vec / norm


def store_swing_lift_dirs(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings: ModelBindings,
    fallback_local_offset: np.ndarray,
) -> None:
    for leg in bindings.leg_bindings:
        leg.swing_lift_dir = estimate_swing_lift_dir(m, d, leg, fallback_local_offset)


def cycle_time(cfg: MPCConfig) -> float:
    return cfg.stance_time + cfg.swing_time


def stance_fraction(cfg: MPCConfig) -> float:
    return cfg.stance_time / cycle_time(cfg)


def leg_phase_offset(leg_idx: int) -> float:
    offsets = [0.0, 0.5, 0.5, 0.0]
    return offsets[leg_idx]


def swing_phase_alpha(t: float, leg_idx: int, cfg: MPCConfig) -> float:
    cyc = cycle_time(cfg)
    st_frac = stance_fraction(cfg)
    phase = ((t / cyc) + leg_phase_offset(leg_idx)) % 1.0
    if phase < st_frac:
        return 0.0
    s = (phase - st_frac) / max(1.0 - st_frac, 1e-9)
    return float(np.sin(np.pi * s))


def build_ctrl_targets_with_cfg(
    d: mujoco.MjData,
    bindings: ModelBindings,
    home_ctrl: np.ndarray,
    scheduled_contact: np.ndarray,
    swing_amp: float,
    cfg: MPCConfig,
) -> np.ndarray:
    if home_ctrl.size == 0:
        return home_ctrl

    ctrl = home_ctrl.copy()
    t = float(d.time)

    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if not leg.actuator_ids or leg.home_joint_qpos is None:
            continue

        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        if bool(scheduled_contact[leg_idx]):
            # Position servos should not fight the MPC GRF during stance.
            target = current_q
        else:
            alpha = swing_phase_alpha(t, leg_idx, cfg)
            lift_dir = leg.swing_lift_dir if leg.swing_lift_dir is not None else np.zeros_like(leg.home_joint_qpos)
            target = leg.home_joint_qpos + swing_amp * alpha * lift_dir

        ctrl[np.asarray(leg.actuator_ids, dtype=int)] = target

    return ctrl


def print_binding_summary(bindings: ModelBindings) -> str:
    lines = [f"Base body: {bindings.base_body_name}"]
    lines.append(f"Floor geoms: {sorted(bindings.floor_geom_ids)}")
    for leg in bindings.leg_bindings:
        lines.append(
            f"{leg.leg}: calf={leg.calf_body_name}, joints={leg.joint_ids}, "
            f"actuators={leg.actuator_ids}, geoms={sorted(leg.geom_ids)}, "
            f"foot_geom={leg.foot_geom_id}, foot_local={None if leg.foot_geom_local is None else np.round(leg.foot_geom_local, 4)}"
        )
    return "\n".join(lines)
