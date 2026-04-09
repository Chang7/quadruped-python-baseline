from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _leaf_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray(0.0, dtype=float)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            flat = [_leaf_to_numpy(v).reshape(-1)[0] if _leaf_to_numpy(v).size else 0.0 for v in value.ravel()]
            return np.asarray(flat, dtype=float).reshape(value.shape)
        return value.astype(float, copy=True)
    if isinstance(value, (float, int, bool, np.floating, np.integer, np.bool_)):
        return np.asarray(float(value), dtype=float)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.asarray(0.0, dtype=float)
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            # e.g. a list of contact objects -> treat as contact present
            return np.asarray(float(len(value) > 0), dtype=float)

    # MuJoCo contact structs and similar objects are not directly castable to float.
    # If it looks like a contact-like object, encode presence as 1.0.
    if any(hasattr(value, attr) for attr in ("dist", "geom1", "geom2", "frame", "pos")):
        return np.asarray(1.0, dtype=float)

    try:
        return np.asarray(value, dtype=float)
    except Exception:
        try:
            return np.asarray(float(bool(value)), dtype=float)
        except Exception:
            return np.asarray(0.0, dtype=float)


def _to_numpy(value: Any, legs_order=("FL", "FR", "RL", "RR")) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if isinstance(value, dict):
        if all(k in value for k in legs_order):
            return np.stack([_leaf_to_numpy(value[k]) for k in legs_order], axis=0)
        try:
            return np.asarray(value)
        except Exception:
            return np.asarray([])
    for attr in legs_order:
        if not hasattr(value, attr):
            break
    else:
        return np.stack([_leaf_to_numpy(getattr(value, attr)) for attr in legs_order], axis=0)
    return _leaf_to_numpy(value)


def init_run_log(controller_type: str, gait: str, robot: str, scene: str) -> dict[str, Any]:
    return {
        "meta": {
            "controller_type": controller_type,
            "gait": gait,
            "robot": robot,
            "scene": scene,
        },
        "time": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "step_num": [],
        "base_pos": [],
        "com_pos": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "base_ori_euler_xyz": [],
        "ref_base_lin_vel": [],
        "ref_base_ang_vel": [],
        "action": [],
        "qpos": [],
        "qvel": [],
        "feet_pos": [],
        "feet_vel": [],
        "des_foot_pos": [],
        "des_foot_vel": [],
        "foot_contact": [],
        "foot_grf": [],
        "phase_signal": [],
        "swing_time": [],
        "planned_contact": [],
        "current_contact": [],
        "swing_contact_release_active": [],
        "latched_release_alpha": [],
        "latched_swing_time": [],
        "support_margin": [],
        "support_confirm_active": [],
        "pre_swing_gate_active": [],
        "front_late_release_active": [],
        "touchdown_reacquire_active": [],
        "touchdown_confirm_active": [],
        "touchdown_settle_active": [],
        "touchdown_support_active": [],
        "rear_retry_contact_signal": [],
        "rear_touchdown_contact_ready": [],
        "rear_late_stance_contact_ready": [],
        "rear_all_contact_support_needed": [],
        "rear_late_seam_elapsed_s": [],
        "rear_late_seam_support_active": [],
        "rear_close_handoff_active": [],
        "rear_late_load_share_active": [],
        "rear_late_load_share_alpha": [],
        "rear_late_load_share_candidate_active": [],
        "rear_late_load_share_candidate_alpha": [],
        "rear_late_load_share_trigger_elapsed_s": [],
        "rear_late_load_share_trigger_enabled": [],
        "rear_close_handoff_alpha": [],
        "rear_close_handoff_leg_index": [],
        "rear_all_contact_weak_leg_alpha": [],
        "rear_all_contact_weak_leg_index": [],
        "applied_linear_support_force_floor_ratio": [],
        "applied_linear_rear_handoff_leg_index": [],
        "applied_linear_rear_handoff_leg_floor_scale": [],
        "applied_linear_latched_force_scale": [],
        "applied_linear_latched_front_receiver_scale": [],
        "applied_linear_latched_rear_receiver_scale": [],
        "rear_touchdown_actual_contact_elapsed_s": [],
        "rear_touchdown_pending_confirm": [],
        "front_margin_rescue_active": [],
        "front_margin_rescue_alpha": [],
        "touchdown_support_alpha": [],
        "rear_all_contact_stabilization_alpha": [],
        "rear_all_contact_front_planted_tail_alpha": [],
        "crawl_front_planted_seam_support_alpha": [],
        "rear_handoff_support_active": [],
        "rear_swing_bridge_active": [],
        "rear_swing_release_support_active": [],
        "full_contact_recovery_active": [],
        "full_contact_recovery_alpha": [],
        "full_contact_recovery_remaining_s": [],
        "full_contact_recovery_trigger": [],
        "front_delayed_swing_recovery_trigger": [],
        "planted_front_recovery_trigger": [],
        "planted_front_postdrop_recovery_trigger": [],
        "front_close_gap_trigger": [],
        "front_late_rearm_trigger": [],
        "front_planted_posture_tail_trigger": [],
        "front_late_posture_tail_trigger": [],
        "crawl_front_stance_support_tail_remaining_s": [],
        "front_touchdown_support_recent_remaining_s": [],
        "front_delayed_swing_recovery_spent": [],
        "gate_forward_scale": [],
        "nmpc_grfs": [],
        "nmpc_footholds": [],
        "ref_base_height": [],
    }


def append_step(
    log: dict[str, Any],
    *,
    sim_time: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    step_num: int,
    base_pos: np.ndarray,
    com_pos: np.ndarray,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    base_ori_euler_xyz: np.ndarray,
    ref_base_lin_vel: np.ndarray,
    ref_base_ang_vel: np.ndarray,
    action: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    feet_pos: Any,
    feet_vel: Any,
    foot_contact: Any,
    foot_grf: Any,
    ctrl_state: dict[str, Any] | None,
) -> None:
    log["time"].append(float(sim_time))
    log["reward"].append(float(reward))
    log["terminated"].append(bool(terminated))
    log["truncated"].append(bool(truncated))
    log["step_num"].append(int(step_num))
    log["base_pos"].append(np.asarray(base_pos, dtype=float).copy())
    log["com_pos"].append(np.asarray(com_pos, dtype=float).copy())
    log["base_lin_vel"].append(np.asarray(base_lin_vel, dtype=float).copy())
    log["base_ang_vel"].append(np.asarray(base_ang_vel, dtype=float).copy())
    log["base_ori_euler_xyz"].append(np.asarray(base_ori_euler_xyz, dtype=float).copy())
    log["ref_base_lin_vel"].append(np.asarray(ref_base_lin_vel, dtype=float).copy())
    log["ref_base_ang_vel"].append(np.asarray(ref_base_ang_vel, dtype=float).copy())
    log["action"].append(np.asarray(action, dtype=float).copy())
    log["qpos"].append(np.asarray(qpos, dtype=float).copy())
    log["qvel"].append(np.asarray(qvel, dtype=float).copy())
    log["feet_pos"].append(_to_numpy(feet_pos))
    log["feet_vel"].append(_to_numpy(feet_vel))
    log["foot_contact"].append(_to_numpy(foot_contact))
    log["foot_grf"].append(_to_numpy(foot_grf))

    ctrl_state = ctrl_state or {}
    if "phase_signal" in ctrl_state:
        log["phase_signal"].append(_to_numpy(ctrl_state["phase_signal"]))
    if "des_foot_pos" in ctrl_state:
        log["des_foot_pos"].append(_to_numpy(ctrl_state["des_foot_pos"]))
    if "des_foot_vel" in ctrl_state:
        log["des_foot_vel"].append(_to_numpy(ctrl_state["des_foot_vel"]))
    if "swing_time" in ctrl_state:
        log["swing_time"].append(_to_numpy(ctrl_state["swing_time"]))
    if "planned_contact" in ctrl_state:
        log["planned_contact"].append(_to_numpy(ctrl_state["planned_contact"]))
    if "current_contact" in ctrl_state:
        log["current_contact"].append(_to_numpy(ctrl_state["current_contact"]))
    if "swing_contact_release_active" in ctrl_state:
        log["swing_contact_release_active"].append(_to_numpy(ctrl_state["swing_contact_release_active"]))
    if "latched_release_alpha" in ctrl_state:
        log["latched_release_alpha"].append(_to_numpy(ctrl_state["latched_release_alpha"]))
    if "latched_swing_time" in ctrl_state:
        log["latched_swing_time"].append(_to_numpy(ctrl_state["latched_swing_time"]))
    if "support_margin" in ctrl_state:
        log["support_margin"].append(_to_numpy(ctrl_state["support_margin"]))
    if "support_confirm_active" in ctrl_state:
        log["support_confirm_active"].append(_to_numpy(ctrl_state["support_confirm_active"]))
    if "pre_swing_gate_active" in ctrl_state:
        log["pre_swing_gate_active"].append(_to_numpy(ctrl_state["pre_swing_gate_active"]))
    if "front_late_release_active" in ctrl_state:
        log["front_late_release_active"].append(_to_numpy(ctrl_state["front_late_release_active"]))
    if "touchdown_reacquire_active" in ctrl_state:
        log["touchdown_reacquire_active"].append(_to_numpy(ctrl_state["touchdown_reacquire_active"]))
    if "touchdown_confirm_active" in ctrl_state:
        log["touchdown_confirm_active"].append(_to_numpy(ctrl_state["touchdown_confirm_active"]))
    if "touchdown_settle_active" in ctrl_state:
        log["touchdown_settle_active"].append(_to_numpy(ctrl_state["touchdown_settle_active"]))
    if "touchdown_support_active" in ctrl_state:
        log["touchdown_support_active"].append(_to_numpy(ctrl_state["touchdown_support_active"]))
    if "rear_retry_contact_signal" in ctrl_state:
        log["rear_retry_contact_signal"].append(_to_numpy(ctrl_state["rear_retry_contact_signal"]))
    if "rear_touchdown_contact_ready" in ctrl_state:
        log["rear_touchdown_contact_ready"].append(_to_numpy(ctrl_state["rear_touchdown_contact_ready"]))
    if "rear_late_stance_contact_ready" in ctrl_state:
        log["rear_late_stance_contact_ready"].append(_to_numpy(ctrl_state["rear_late_stance_contact_ready"]))
    if "rear_all_contact_support_needed" in ctrl_state:
        log["rear_all_contact_support_needed"].append(_to_numpy(ctrl_state["rear_all_contact_support_needed"]))
    if "rear_late_seam_elapsed_s" in ctrl_state:
        log["rear_late_seam_elapsed_s"].append(_to_numpy(ctrl_state["rear_late_seam_elapsed_s"]))
    if "rear_late_seam_support_active" in ctrl_state:
        log["rear_late_seam_support_active"].append(_to_numpy(ctrl_state["rear_late_seam_support_active"]))
    if "rear_close_handoff_active" in ctrl_state:
        log["rear_close_handoff_active"].append(_to_numpy(ctrl_state["rear_close_handoff_active"]))
    if "rear_late_load_share_active" in ctrl_state:
        log["rear_late_load_share_active"].append(_to_numpy(ctrl_state["rear_late_load_share_active"]))
    if "rear_late_load_share_alpha" in ctrl_state:
        log["rear_late_load_share_alpha"].append(_to_numpy(ctrl_state["rear_late_load_share_alpha"]))
    if "rear_late_load_share_candidate_active" in ctrl_state:
        log["rear_late_load_share_candidate_active"].append(
            _to_numpy(ctrl_state["rear_late_load_share_candidate_active"])
        )
    if "rear_late_load_share_candidate_alpha" in ctrl_state:
        log["rear_late_load_share_candidate_alpha"].append(
            _to_numpy(ctrl_state["rear_late_load_share_candidate_alpha"])
        )
    if "rear_late_load_share_trigger_elapsed_s" in ctrl_state:
        log["rear_late_load_share_trigger_elapsed_s"].append(
            _to_numpy(ctrl_state["rear_late_load_share_trigger_elapsed_s"])
        )
    if "rear_late_load_share_trigger_enabled" in ctrl_state:
        log["rear_late_load_share_trigger_enabled"].append(
            float(ctrl_state["rear_late_load_share_trigger_enabled"])
        )
    if "rear_close_handoff_alpha" in ctrl_state:
        log["rear_close_handoff_alpha"].append(float(ctrl_state["rear_close_handoff_alpha"]))
    if "rear_close_handoff_leg_index" in ctrl_state:
        log["rear_close_handoff_leg_index"].append(float(ctrl_state["rear_close_handoff_leg_index"]))
    if "rear_all_contact_weak_leg_alpha" in ctrl_state:
        log["rear_all_contact_weak_leg_alpha"].append(float(ctrl_state["rear_all_contact_weak_leg_alpha"]))
    if "rear_all_contact_weak_leg_index" in ctrl_state:
        log["rear_all_contact_weak_leg_index"].append(float(ctrl_state["rear_all_contact_weak_leg_index"]))
    if "applied_linear_support_force_floor_ratio" in ctrl_state:
        log["applied_linear_support_force_floor_ratio"].append(
            float(ctrl_state["applied_linear_support_force_floor_ratio"])
        )
    if "applied_linear_rear_handoff_leg_index" in ctrl_state:
        log["applied_linear_rear_handoff_leg_index"].append(
            float(ctrl_state["applied_linear_rear_handoff_leg_index"])
        )
    if "applied_linear_rear_handoff_leg_floor_scale" in ctrl_state:
        log["applied_linear_rear_handoff_leg_floor_scale"].append(
            float(ctrl_state["applied_linear_rear_handoff_leg_floor_scale"])
        )
    if "applied_linear_latched_force_scale" in ctrl_state:
        log["applied_linear_latched_force_scale"].append(
            float(ctrl_state["applied_linear_latched_force_scale"])
        )
    if "applied_linear_latched_front_receiver_scale" in ctrl_state:
        log["applied_linear_latched_front_receiver_scale"].append(
            float(ctrl_state["applied_linear_latched_front_receiver_scale"])
        )
    if "applied_linear_latched_rear_receiver_scale" in ctrl_state:
        log["applied_linear_latched_rear_receiver_scale"].append(
            float(ctrl_state["applied_linear_latched_rear_receiver_scale"])
        )
    if "rear_touchdown_actual_contact_elapsed_s" in ctrl_state:
        log["rear_touchdown_actual_contact_elapsed_s"].append(_to_numpy(ctrl_state["rear_touchdown_actual_contact_elapsed_s"]))
    if "rear_touchdown_pending_confirm" in ctrl_state:
        log["rear_touchdown_pending_confirm"].append(_to_numpy(ctrl_state["rear_touchdown_pending_confirm"]))
    if "front_margin_rescue_active" in ctrl_state:
        log["front_margin_rescue_active"].append(_to_numpy(ctrl_state["front_margin_rescue_active"]))
    if "front_margin_rescue_alpha" in ctrl_state:
        log["front_margin_rescue_alpha"].append(_to_numpy(ctrl_state["front_margin_rescue_alpha"]))
    if "touchdown_support_alpha" in ctrl_state:
        log["touchdown_support_alpha"].append(float(ctrl_state["touchdown_support_alpha"]))
    if "rear_all_contact_stabilization_alpha" in ctrl_state:
        log["rear_all_contact_stabilization_alpha"].append(float(ctrl_state["rear_all_contact_stabilization_alpha"]))
    if "rear_all_contact_front_planted_tail_alpha" in ctrl_state:
        log["rear_all_contact_front_planted_tail_alpha"].append(
            float(ctrl_state["rear_all_contact_front_planted_tail_alpha"])
        )
    if "crawl_front_planted_seam_support_alpha" in ctrl_state:
        log["crawl_front_planted_seam_support_alpha"].append(
            float(ctrl_state["crawl_front_planted_seam_support_alpha"])
        )
    if "rear_handoff_support_active" in ctrl_state:
        log["rear_handoff_support_active"].append(float(ctrl_state["rear_handoff_support_active"]))
    if "rear_swing_bridge_active" in ctrl_state:
        log["rear_swing_bridge_active"].append(float(ctrl_state["rear_swing_bridge_active"]))
    if "rear_swing_release_support_active" in ctrl_state:
        log["rear_swing_release_support_active"].append(float(ctrl_state["rear_swing_release_support_active"]))
    if "full_contact_recovery_active" in ctrl_state:
        log["full_contact_recovery_active"].append(_to_numpy(ctrl_state["full_contact_recovery_active"]))
    if "full_contact_recovery_alpha" in ctrl_state:
        log["full_contact_recovery_alpha"].append(float(ctrl_state["full_contact_recovery_alpha"]))
    if "full_contact_recovery_remaining_s" in ctrl_state:
        log["full_contact_recovery_remaining_s"].append(float(ctrl_state["full_contact_recovery_remaining_s"]))
    if "full_contact_recovery_trigger" in ctrl_state:
        log["full_contact_recovery_trigger"].append(float(ctrl_state["full_contact_recovery_trigger"]))
    if "front_delayed_swing_recovery_trigger" in ctrl_state:
        log["front_delayed_swing_recovery_trigger"].append(
            float(ctrl_state["front_delayed_swing_recovery_trigger"])
        )
    if "planted_front_recovery_trigger" in ctrl_state:
        log["planted_front_recovery_trigger"].append(float(ctrl_state["planted_front_recovery_trigger"]))
    if "planted_front_postdrop_recovery_trigger" in ctrl_state:
        log["planted_front_postdrop_recovery_trigger"].append(
            float(ctrl_state["planted_front_postdrop_recovery_trigger"])
        )
    if "front_close_gap_trigger" in ctrl_state:
        log["front_close_gap_trigger"].append(float(ctrl_state["front_close_gap_trigger"]))
    if "front_late_rearm_trigger" in ctrl_state:
        log["front_late_rearm_trigger"].append(float(ctrl_state["front_late_rearm_trigger"]))
    if "front_planted_posture_tail_trigger" in ctrl_state:
        log["front_planted_posture_tail_trigger"].append(
            float(ctrl_state["front_planted_posture_tail_trigger"])
        )
    if "front_late_posture_tail_trigger" in ctrl_state:
        log["front_late_posture_tail_trigger"].append(float(ctrl_state["front_late_posture_tail_trigger"]))
    if "crawl_front_stance_support_tail_remaining_s" in ctrl_state:
        log["crawl_front_stance_support_tail_remaining_s"].append(
            float(ctrl_state["crawl_front_stance_support_tail_remaining_s"])
        )
    if "front_touchdown_support_recent_remaining_s" in ctrl_state:
        log["front_touchdown_support_recent_remaining_s"].append(
            float(ctrl_state["front_touchdown_support_recent_remaining_s"])
        )
    if "front_delayed_swing_recovery_spent" in ctrl_state:
        log["front_delayed_swing_recovery_spent"].append(
            _to_numpy(ctrl_state["front_delayed_swing_recovery_spent"])
        )
    if "gate_forward_scale" in ctrl_state:
        log["gate_forward_scale"].append(_to_numpy(ctrl_state["gate_forward_scale"]))
    if "nmpc_GRFs" in ctrl_state:
        log["nmpc_grfs"].append(_to_numpy(ctrl_state["nmpc_GRFs"]))
    if "nmpc_footholds" in ctrl_state:
        log["nmpc_footholds"].append(_to_numpy(ctrl_state["nmpc_footholds"]))
    if "ref_base_height" in ctrl_state:
        log["ref_base_height"].append(_to_numpy(ctrl_state["ref_base_height"]))


def finalize_log(log: dict[str, Any]) -> dict[str, Any]:
    out = {"meta": log["meta"]}
    for key, value in log.items():
        if key == "meta":
            continue
        if len(value) == 0:
            out[key] = np.asarray([])
            continue
        try:
            out[key] = np.stack([np.asarray(v) for v in value], axis=0)
        except Exception:
            out[key] = np.asarray(value, dtype=object)
    return out


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def _first_true_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def summarize_log(final_log: dict[str, Any]) -> dict[str, Any]:
    time = np.asarray(final_log.get("time", []), dtype=float).reshape(-1)
    base_pos = np.asarray(final_log.get("base_pos", []), dtype=float)
    base_lin_vel = np.asarray(final_log.get("base_lin_vel", []), dtype=float)
    euler = np.asarray(final_log.get("base_ori_euler_xyz", []), dtype=float)
    action = np.asarray(final_log.get("action", []), dtype=float)
    foot_contact = np.asarray(final_log.get("foot_contact", []), dtype=float)
    foot_grf = np.asarray(final_log.get("foot_grf", []), dtype=float)
    planned_contact = np.asarray(final_log.get("planned_contact", []), dtype=float)
    current_contact = np.asarray(final_log.get("current_contact", []), dtype=float)
    feet_pos = np.asarray(final_log.get("feet_pos", []), dtype=float)
    ref_base_height = np.asarray(final_log.get("ref_base_height", []), dtype=float).reshape(-1)
    support_margin = np.asarray(final_log.get("support_margin", []), dtype=float)
    support_confirm_active = np.asarray(final_log.get("support_confirm_active", []), dtype=float)
    pre_swing_gate_active = np.asarray(final_log.get("pre_swing_gate_active", []), dtype=float)
    front_late_release_active = np.asarray(final_log.get("front_late_release_active", []), dtype=float)
    touchdown_reacquire_active = np.asarray(final_log.get("touchdown_reacquire_active", []), dtype=float)
    touchdown_confirm_active = np.asarray(final_log.get("touchdown_confirm_active", []), dtype=float)
    touchdown_settle_active = np.asarray(final_log.get("touchdown_settle_active", []), dtype=float)
    touchdown_support_active = np.asarray(final_log.get("touchdown_support_active", []), dtype=float)
    front_margin_rescue_active = np.asarray(final_log.get("front_margin_rescue_active", []), dtype=float)
    front_margin_rescue_alpha = np.asarray(final_log.get("front_margin_rescue_alpha", []), dtype=float)
    touchdown_support_alpha = np.asarray(final_log.get("touchdown_support_alpha", []), dtype=float).reshape(-1)
    rear_all_contact_stabilization_alpha = np.asarray(
        final_log.get("rear_all_contact_stabilization_alpha", []),
        dtype=float,
    ).reshape(-1)
    rear_all_contact_front_planted_tail_alpha = np.asarray(
        final_log.get("rear_all_contact_front_planted_tail_alpha", []),
        dtype=float,
    ).reshape(-1)
    crawl_front_planted_seam_support_alpha = np.asarray(
        final_log.get("crawl_front_planted_seam_support_alpha", []),
        dtype=float,
    ).reshape(-1)
    rear_late_seam_elapsed_s = np.asarray(final_log.get("rear_late_seam_elapsed_s", []), dtype=float)
    rear_late_seam_support_active = np.asarray(final_log.get("rear_late_seam_support_active", []), dtype=float)
    rear_close_handoff_active = np.asarray(final_log.get("rear_close_handoff_active", []), dtype=float)
    rear_late_load_share_active = np.asarray(final_log.get("rear_late_load_share_active", []), dtype=float)
    rear_late_load_share_alpha = np.asarray(final_log.get("rear_late_load_share_alpha", []), dtype=float)
    rear_late_load_share_candidate_active = np.asarray(
        final_log.get("rear_late_load_share_candidate_active", []),
        dtype=float,
    )
    rear_late_load_share_candidate_alpha = np.asarray(
        final_log.get("rear_late_load_share_candidate_alpha", []),
        dtype=float,
    )
    rear_late_load_share_trigger_elapsed_s = np.asarray(
        final_log.get("rear_late_load_share_trigger_elapsed_s", []),
        dtype=float,
    )
    rear_late_load_share_trigger_enabled = np.asarray(
        final_log.get("rear_late_load_share_trigger_enabled", []),
        dtype=float,
    ).reshape(-1)
    rear_close_handoff_alpha = np.asarray(final_log.get("rear_close_handoff_alpha", []), dtype=float).reshape(-1)
    rear_close_handoff_leg_index = np.asarray(final_log.get("rear_close_handoff_leg_index", []), dtype=float).reshape(-1)
    rear_all_contact_weak_leg_alpha = np.asarray(
        final_log.get("rear_all_contact_weak_leg_alpha", []),
        dtype=float,
    ).reshape(-1)
    rear_all_contact_weak_leg_index = np.asarray(
        final_log.get("rear_all_contact_weak_leg_index", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_support_force_floor_ratio = np.asarray(
        final_log.get("applied_linear_support_force_floor_ratio", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_rear_handoff_leg_index = np.asarray(
        final_log.get("applied_linear_rear_handoff_leg_index", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_rear_handoff_leg_floor_scale = np.asarray(
        final_log.get("applied_linear_rear_handoff_leg_floor_scale", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_latched_force_scale = np.asarray(
        final_log.get("applied_linear_latched_force_scale", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_latched_front_receiver_scale = np.asarray(
        final_log.get("applied_linear_latched_front_receiver_scale", []),
        dtype=float,
    ).reshape(-1)
    applied_linear_latched_rear_receiver_scale = np.asarray(
        final_log.get("applied_linear_latched_rear_receiver_scale", []),
        dtype=float,
    ).reshape(-1)
    rear_all_contact_support_needed = np.asarray(
        final_log.get("rear_all_contact_support_needed", []),
        dtype=float,
    )
    rear_handoff_support_active = np.asarray(final_log.get("rear_handoff_support_active", []), dtype=float).reshape(-1)
    rear_swing_bridge_active = np.asarray(final_log.get("rear_swing_bridge_active", []), dtype=float).reshape(-1)
    full_contact_recovery_active = np.asarray(final_log.get("full_contact_recovery_active", []), dtype=float).reshape(-1)
    full_contact_recovery_alpha = np.asarray(final_log.get("full_contact_recovery_alpha", []), dtype=float).reshape(-1)
    gate_forward_scale = np.asarray(final_log.get("gate_forward_scale", []), dtype=float).reshape(-1)
    terminated = np.asarray(final_log.get("terminated", []), dtype=bool).reshape(-1)
    truncated = np.asarray(final_log.get("truncated", []), dtype=bool).reshape(-1)

    action_leg_norms = None
    if action.ndim == 2 and action.shape[1] >= 12:
        action_leg_norms = np.stack([
            np.linalg.norm(action[:, 0:3], axis=1),
            np.linalg.norm(action[:, 3:6], axis=1),
            np.linalg.norm(action[:, 6:9], axis=1),
            np.linalg.norm(action[:, 9:12], axis=1),
        ], axis=1)

    grf_leg_norms = None
    if foot_grf.ndim >= 3 and foot_grf.shape[1] >= 4:
        grf_leg_norms = np.linalg.norm(foot_grf[:, :4, :], axis=2)

    summary = {
        "meta": final_log.get("meta", {}),
        "n_steps": int(time.size),
        "duration_s": _safe_float(time[-1] - time[0]) if time.size >= 2 else _safe_float(time[0]) if time.size == 1 else None,
        "terminated_any": bool(terminated.any()) if terminated.size else False,
        "truncated_any": bool(truncated.any()) if truncated.size else False,
        "first_terminated_step": _first_true_index(terminated),
        "first_truncated_step": _first_true_index(truncated),
        "mean_vx": _safe_float(base_lin_vel[:, 0].mean()) if base_lin_vel.ndim == 2 and base_lin_vel.shape[1] >= 1 else None,
        "mean_vy": _safe_float(base_lin_vel[:, 1].mean()) if base_lin_vel.ndim == 2 and base_lin_vel.shape[1] >= 2 else None,
        "max_vx": _safe_float(base_lin_vel[:, 0].max()) if base_lin_vel.ndim == 2 and base_lin_vel.shape[1] >= 1 else None,
        "min_vx": _safe_float(base_lin_vel[:, 0].min()) if base_lin_vel.ndim == 2 and base_lin_vel.shape[1] >= 1 else None,
        "mean_base_z": _safe_float(base_pos[:, 2].mean()) if base_pos.ndim == 2 and base_pos.shape[1] >= 3 else None,
        "min_base_z": _safe_float(base_pos[:, 2].min()) if base_pos.ndim == 2 and base_pos.shape[1] >= 3 else None,
        "mean_abs_roll": _safe_float(np.abs(euler[:, 0]).mean()) if euler.ndim == 2 and euler.shape[1] >= 1 else None,
        "mean_abs_pitch": _safe_float(np.abs(euler[:, 1]).mean()) if euler.ndim == 2 and euler.shape[1] >= 2 else None,
        "max_abs_roll": _safe_float(np.abs(euler[:, 0]).max()) if euler.ndim == 2 and euler.shape[1] >= 1 else None,
        "max_abs_pitch": _safe_float(np.abs(euler[:, 1]).max()) if euler.ndim == 2 and euler.shape[1] >= 2 else None,
    }
    if foot_contact.ndim == 2 and foot_contact.shape[1] >= 4:
        summary["mean_contact_ratio"] = {
            "FL": _safe_float(foot_contact[:, 0].mean()),
            "FR": _safe_float(foot_contact[:, 1].mean()),
            "RL": _safe_float(foot_contact[:, 2].mean()),
            "RR": _safe_float(foot_contact[:, 3].mean()),
        }
        actual_swing_mask = foot_contact[:, :4] <= 0.5
        summary["steps_any_actual_swing"] = int(np.any(actual_swing_mask, axis=1).sum())
        summary["actual_swing_ratio"] = {
            "FL": _safe_float(actual_swing_mask[:, 0].mean()),
            "FR": _safe_float(actual_swing_mask[:, 1].mean()),
            "RL": _safe_float(actual_swing_mask[:, 2].mean()),
            "RR": _safe_float(actual_swing_mask[:, 3].mean()),
        }
        summary["legs_with_actual_swing"] = int(np.count_nonzero(np.any(actual_swing_mask, axis=0)))
    if grf_leg_norms is not None:
        summary["mean_grf_norm"] = {
            "FL": _safe_float(grf_leg_norms[:, 0].mean()),
            "FR": _safe_float(grf_leg_norms[:, 1].mean()),
            "RL": _safe_float(grf_leg_norms[:, 2].mean()),
            "RR": _safe_float(grf_leg_norms[:, 3].mean()),
        }
        rear_vertical_grf = np.maximum(foot_grf[:, 2:4, 2], 0.0)
        rear_total_vertical_grf = np.sum(rear_vertical_grf, axis=1)
        valid_rear_pair = rear_total_vertical_grf > 1e-6
        if np.any(valid_rear_pair):
            rear_pair_shares = np.zeros_like(rear_vertical_grf, dtype=float)
            rear_pair_shares[valid_rear_pair] = (
                rear_vertical_grf[valid_rear_pair]
                / rear_total_vertical_grf[valid_rear_pair, None]
            )
            weak_rear_leg_share = np.min(rear_pair_shares[valid_rear_pair], axis=1)
            weak_rear_leg_index = np.argmin(rear_pair_shares[valid_rear_pair], axis=1)
            summary["rear_pair_load_share_mean"] = {
                "RL": _safe_float(float(np.mean(rear_pair_shares[valid_rear_pair, 0]))),
                "RR": _safe_float(float(np.mean(rear_pair_shares[valid_rear_pair, 1]))),
            }
            summary["rear_pair_weak_leg_share_mean"] = _safe_float(float(np.mean(weak_rear_leg_share)))
            summary["rear_pair_weak_leg_share_p10"] = _safe_float(float(np.percentile(weak_rear_leg_share, 10)))
            summary["rear_pair_weaker_leg_ratio"] = {
                "RL": _safe_float(float(np.mean(weak_rear_leg_index == 0))),
                "RR": _safe_float(float(np.mean(weak_rear_leg_index == 1))),
            }

            if (
                rear_all_contact_support_needed.ndim == 2
                and rear_all_contact_support_needed.shape[0] == rear_vertical_grf.shape[0]
                and rear_all_contact_support_needed.shape[1] >= 4
            ):
                rear_need_mask = np.any(rear_all_contact_support_needed[:, 2:4] > 0.5, axis=1)
            else:
                rear_need_mask = np.zeros(rear_vertical_grf.shape[0], dtype=bool)
            need_valid_mask = valid_rear_pair & rear_need_mask
            if np.any(need_valid_mask):
                need_shares = rear_pair_shares[need_valid_mask]
                need_weak_share = np.min(need_shares, axis=1)
                need_weak_index = np.argmin(need_shares, axis=1)
                summary["rear_pair_load_share_when_needed_mean"] = {
                    "RL": _safe_float(float(np.mean(need_shares[:, 0]))),
                    "RR": _safe_float(float(np.mean(need_shares[:, 1]))),
                }
                summary["rear_pair_weak_leg_share_when_needed_mean"] = _safe_float(
                    float(np.mean(need_weak_share))
                )
                summary["rear_pair_weak_leg_share_when_needed_p10"] = _safe_float(
                    float(np.percentile(need_weak_share, 10))
                )
                summary["rear_pair_weaker_leg_when_needed_ratio"] = {
                    "RL": _safe_float(float(np.mean(need_weak_index == 0))),
                    "RR": _safe_float(float(np.mean(need_weak_index == 1))),
                }
                for tail_len in (600, 300):
                    tail_mask = need_valid_mask.copy()
                    tail_mask[: max(0, tail_mask.size - tail_len)] = False
                    if not np.any(tail_mask):
                        continue
                    tail_shares = rear_pair_shares[tail_mask]
                    tail_weak_share = np.min(tail_shares, axis=1)
                    tail_weak_index = np.argmin(tail_shares, axis=1)
                    summary[f"rear_pair_load_share_tail{tail_len}_mean"] = {
                        "RL": _safe_float(float(np.mean(tail_shares[:, 0]))),
                        "RR": _safe_float(float(np.mean(tail_shares[:, 1]))),
                    }
                    summary[f"rear_pair_weak_leg_share_tail{tail_len}_mean"] = _safe_float(
                        float(np.mean(tail_weak_share))
                    )
                    summary[f"rear_pair_weak_leg_share_tail{tail_len}_p10"] = _safe_float(
                        float(np.percentile(tail_weak_share, 10))
                    )
                    summary[f"rear_pair_weaker_leg_tail{tail_len}_ratio"] = {
                        "RL": _safe_float(float(np.mean(tail_weak_index == 0))),
                        "RR": _safe_float(float(np.mean(tail_weak_index == 1))),
                    }
    if action_leg_norms is not None:
        summary["mean_action_leg_norm"] = {
            "FL": _safe_float(action_leg_norms[:, 0].mean()),
            "FR": _safe_float(action_leg_norms[:, 1].mean()),
            "RL": _safe_float(action_leg_norms[:, 2].mean()),
            "RR": _safe_float(action_leg_norms[:, 3].mean()),
        }
    planned_swing_mask = None
    current_swing_mask = None
    actual_swing_mask = None

    if planned_contact.ndim == 2 and planned_contact.shape[1] >= 4:
        planned_swing_mask = planned_contact[:, :4] <= 0.5
        summary["steps_any_planned_swing"] = int(np.any(planned_swing_mask, axis=1).sum())
        summary["planned_swing_ratio"] = {
            "FL": _safe_float(planned_swing_mask[:, 0].mean()),
            "FR": _safe_float(planned_swing_mask[:, 1].mean()),
            "RL": _safe_float(planned_swing_mask[:, 2].mean()),
            "RR": _safe_float(planned_swing_mask[:, 3].mean()),
        }
        summary["legs_with_planned_swing"] = int(np.count_nonzero(np.any(planned_swing_mask, axis=0)))
    if current_contact.ndim == 2 and current_contact.shape[1] >= 4:
        current_swing_mask = current_contact[:, :4] <= 0.5
        summary["steps_any_current_swing"] = int(np.any(current_swing_mask, axis=1).sum())
        summary["current_swing_ratio"] = {
            "FL": _safe_float(current_swing_mask[:, 0].mean()),
            "FR": _safe_float(current_swing_mask[:, 1].mean()),
            "RL": _safe_float(current_swing_mask[:, 2].mean()),
            "RR": _safe_float(current_swing_mask[:, 3].mean()),
        }
        summary["legs_with_current_swing"] = int(np.count_nonzero(np.any(current_swing_mask, axis=0)))

    if foot_contact.ndim == 2 and foot_contact.shape[1] >= 4:
        actual_swing_mask = foot_contact[:, :4] <= 0.5

    if planned_swing_mask is not None:
        current_realization_by_leg: dict[str, float | None] = {}
        actual_realization_by_leg: dict[str, float | None] = {}
        leg_names = ("FL", "FR", "RL", "RR")
        current_overlap_total = 0
        actual_overlap_total = 0
        planned_total = int(planned_swing_mask.sum())
        for leg_idx, leg in enumerate(leg_names):
            planned_steps = int(planned_swing_mask[:, leg_idx].sum())
            if planned_steps <= 0:
                current_realization_by_leg[leg] = None
                actual_realization_by_leg[leg] = None
                continue

            if current_swing_mask is not None:
                current_overlap = int(np.logical_and(planned_swing_mask[:, leg_idx], current_swing_mask[:, leg_idx]).sum())
                current_realization_by_leg[leg] = _safe_float(float(current_overlap) / float(planned_steps))
                current_overlap_total += current_overlap
            else:
                current_realization_by_leg[leg] = None

            if actual_swing_mask is not None:
                actual_overlap = int(np.logical_and(planned_swing_mask[:, leg_idx], actual_swing_mask[:, leg_idx]).sum())
                actual_realization_by_leg[leg] = _safe_float(float(actual_overlap) / float(planned_steps))
                actual_overlap_total += actual_overlap
            else:
                actual_realization_by_leg[leg] = None

        summary["current_swing_realization_ratio"] = current_realization_by_leg
        summary["actual_swing_realization_ratio"] = actual_realization_by_leg

        if planned_total > 0:
            summary["current_swing_realization_total"] = _safe_float(float(current_overlap_total) / float(planned_total))
            summary["actual_swing_realization_total"] = _safe_float(float(actual_overlap_total) / float(planned_total))

        front_current = [current_realization_by_leg.get("FL"), current_realization_by_leg.get("FR")]
        front_actual = [actual_realization_by_leg.get("FL"), actual_realization_by_leg.get("FR")]
        front_current = [v for v in front_current if v is not None]
        front_actual = [v for v in front_actual if v is not None]
        if front_current:
            summary["front_current_swing_realization_mean"] = _safe_float(float(np.mean(front_current)))
        if front_actual:
            summary["front_actual_swing_realization_mean"] = _safe_float(float(np.mean(front_actual)))
    if feet_pos.ndim == 3 and feet_pos.shape[1] >= 4 and feet_pos.shape[2] >= 3:
        summary["foot_z_range"] = {
            "FL": _safe_float(feet_pos[:, 0, 2].max() - feet_pos[:, 0, 2].min()),
            "FR": _safe_float(feet_pos[:, 1, 2].max() - feet_pos[:, 1, 2].min()),
            "RL": _safe_float(feet_pos[:, 2, 2].max() - feet_pos[:, 2, 2].min()),
            "RR": _safe_float(feet_pos[:, 3, 2].max() - feet_pos[:, 3, 2].min()),
        }
    if support_margin.ndim == 2 and support_margin.shape[1] >= 4:
        support_margin_clean = np.where(np.isfinite(support_margin[:, :4]), support_margin[:, :4], np.nan)
        leg_names = ("FL", "FR", "RL", "RR")
        support_margin_mean = {}
        support_margin_min = {}
        for idx, leg in enumerate(leg_names):
            leg_vals = support_margin_clean[:, idx]
            valid_vals = leg_vals[np.isfinite(leg_vals)]
            support_margin_mean[leg] = _safe_float(valid_vals.mean()) if valid_vals.size else None
            support_margin_min[leg] = _safe_float(valid_vals.min()) if valid_vals.size else None
        summary["support_margin_mean"] = support_margin_mean
        summary["support_margin_min"] = support_margin_min
    if support_confirm_active.ndim == 2 and support_confirm_active.shape[1] >= 4:
        summary["support_confirm_ratio"] = {
            "FL": _safe_float(support_confirm_active[:, 0].mean()),
            "FR": _safe_float(support_confirm_active[:, 1].mean()),
            "RL": _safe_float(support_confirm_active[:, 2].mean()),
            "RR": _safe_float(support_confirm_active[:, 3].mean()),
        }
        front_vals = [summary["support_confirm_ratio"]["FL"], summary["support_confirm_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_support_confirm_mean"] = _safe_float(float(np.mean(front_vals)))
        rear_vals = [summary["support_confirm_ratio"]["RL"], summary["support_confirm_ratio"]["RR"]]
        rear_vals = [v for v in rear_vals if v is not None]
        if rear_vals:
            summary["rear_support_confirm_mean"] = _safe_float(float(np.mean(rear_vals)))
    if pre_swing_gate_active.ndim == 2 and pre_swing_gate_active.shape[1] >= 4:
        summary["pre_swing_gate_ratio"] = {
            "FL": _safe_float(pre_swing_gate_active[:, 0].mean()),
            "FR": _safe_float(pre_swing_gate_active[:, 1].mean()),
            "RL": _safe_float(pre_swing_gate_active[:, 2].mean()),
            "RR": _safe_float(pre_swing_gate_active[:, 3].mean()),
        }
    if front_late_release_active.ndim == 2 and front_late_release_active.shape[1] >= 4:
        summary["front_late_release_ratio"] = {
            "FL": _safe_float(front_late_release_active[:, 0].mean()),
            "FR": _safe_float(front_late_release_active[:, 1].mean()),
            "RL": _safe_float(front_late_release_active[:, 2].mean()),
            "RR": _safe_float(front_late_release_active[:, 3].mean()),
        }
        front_vals = [summary["front_late_release_ratio"]["FL"], summary["front_late_release_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_late_release_mean"] = _safe_float(float(np.mean(front_vals)))
    if touchdown_reacquire_active.ndim == 2 and touchdown_reacquire_active.shape[1] >= 4:
        summary["touchdown_reacquire_ratio"] = {
            "FL": _safe_float(touchdown_reacquire_active[:, 0].mean()),
            "FR": _safe_float(touchdown_reacquire_active[:, 1].mean()),
            "RL": _safe_float(touchdown_reacquire_active[:, 2].mean()),
            "RR": _safe_float(touchdown_reacquire_active[:, 3].mean()),
        }
        front_vals = [summary["touchdown_reacquire_ratio"]["FL"], summary["touchdown_reacquire_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_touchdown_reacquire_mean"] = _safe_float(float(np.mean(front_vals)))
        rear_vals = [summary["touchdown_reacquire_ratio"]["RL"], summary["touchdown_reacquire_ratio"]["RR"]]
        rear_vals = [v for v in rear_vals if v is not None]
        if rear_vals:
            summary["rear_touchdown_reacquire_mean"] = _safe_float(float(np.mean(rear_vals)))
    if touchdown_confirm_active.ndim == 2 and touchdown_confirm_active.shape[1] >= 4:
        summary["touchdown_confirm_ratio"] = {
            "FL": _safe_float(touchdown_confirm_active[:, 0].mean()),
            "FR": _safe_float(touchdown_confirm_active[:, 1].mean()),
            "RL": _safe_float(touchdown_confirm_active[:, 2].mean()),
            "RR": _safe_float(touchdown_confirm_active[:, 3].mean()),
        }
        front_vals = [summary["touchdown_confirm_ratio"]["FL"], summary["touchdown_confirm_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_touchdown_confirm_mean"] = _safe_float(float(np.mean(front_vals)))
        rear_vals = [summary["touchdown_confirm_ratio"]["RL"], summary["touchdown_confirm_ratio"]["RR"]]
        rear_vals = [v for v in rear_vals if v is not None]
        if rear_vals:
            summary["rear_touchdown_confirm_mean"] = _safe_float(float(np.mean(rear_vals)))
    if touchdown_settle_active.ndim == 2 and touchdown_settle_active.shape[1] >= 4:
        summary["touchdown_settle_ratio"] = {
            "FL": _safe_float(touchdown_settle_active[:, 0].mean()),
            "FR": _safe_float(touchdown_settle_active[:, 1].mean()),
            "RL": _safe_float(touchdown_settle_active[:, 2].mean()),
            "RR": _safe_float(touchdown_settle_active[:, 3].mean()),
        }
        front_vals = [summary["touchdown_settle_ratio"]["FL"], summary["touchdown_settle_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_touchdown_settle_mean"] = _safe_float(float(np.mean(front_vals)))
        rear_vals = [summary["touchdown_settle_ratio"]["RL"], summary["touchdown_settle_ratio"]["RR"]]
        rear_vals = [v for v in rear_vals if v is not None]
        if rear_vals:
            summary["rear_touchdown_settle_mean"] = _safe_float(float(np.mean(rear_vals)))
    if touchdown_support_active.ndim == 2 and touchdown_support_active.shape[1] >= 4:
        summary["touchdown_support_ratio"] = {
            "FL": _safe_float(touchdown_support_active[:, 0].mean()),
            "FR": _safe_float(touchdown_support_active[:, 1].mean()),
            "RL": _safe_float(touchdown_support_active[:, 2].mean()),
            "RR": _safe_float(touchdown_support_active[:, 3].mean()),
        }
        all_vals = [summary["touchdown_support_ratio"][leg] for leg in ("FL", "FR", "RL", "RR")]
        all_vals = [v for v in all_vals if v is not None]
        if all_vals:
            summary["touchdown_support_mean"] = _safe_float(float(np.mean(all_vals)))
        front_vals = [summary["touchdown_support_ratio"]["FL"], summary["touchdown_support_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_touchdown_support_mean"] = _safe_float(float(np.mean(front_vals)))
        rear_vals = [summary["touchdown_support_ratio"]["RL"], summary["touchdown_support_ratio"]["RR"]]
        rear_vals = [v for v in rear_vals if v is not None]
        if rear_vals:
            summary["rear_touchdown_support_mean"] = _safe_float(float(np.mean(rear_vals)))
    if front_margin_rescue_active.ndim == 2 and front_margin_rescue_active.shape[1] >= 4:
        summary["front_margin_rescue_ratio"] = {
            "FL": _safe_float(front_margin_rescue_active[:, 0].mean()),
            "FR": _safe_float(front_margin_rescue_active[:, 1].mean()),
            "RL": _safe_float(front_margin_rescue_active[:, 2].mean()),
            "RR": _safe_float(front_margin_rescue_active[:, 3].mean()),
        }
        front_vals = [summary["front_margin_rescue_ratio"]["FL"], summary["front_margin_rescue_ratio"]["FR"]]
        front_vals = [v for v in front_vals if v is not None]
        if front_vals:
            summary["front_margin_rescue_mean"] = _safe_float(float(np.mean(front_vals)))
    if front_margin_rescue_alpha.ndim == 2 and front_margin_rescue_alpha.shape[1] >= 4:
        summary["front_margin_rescue_alpha_mean"] = {
            "FL": _safe_float(float(np.nanmean(front_margin_rescue_alpha[:, 0]))),
            "FR": _safe_float(float(np.nanmean(front_margin_rescue_alpha[:, 1]))),
            "RL": _safe_float(float(np.nanmean(front_margin_rescue_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmean(front_margin_rescue_alpha[:, 3]))),
        }
        front_alpha_vals = [
            summary["front_margin_rescue_alpha_mean"]["FL"],
            summary["front_margin_rescue_alpha_mean"]["FR"],
        ]
        front_alpha_vals = [v for v in front_alpha_vals if v is not None]
        if front_alpha_vals:
            summary["front_margin_rescue_alpha_front_mean"] = _safe_float(float(np.mean(front_alpha_vals)))
        summary["front_margin_rescue_alpha_max"] = {
            "FL": _safe_float(float(np.nanmax(front_margin_rescue_alpha[:, 0]))),
            "FR": _safe_float(float(np.nanmax(front_margin_rescue_alpha[:, 1]))),
            "RL": _safe_float(float(np.nanmax(front_margin_rescue_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmax(front_margin_rescue_alpha[:, 3]))),
        }
    if touchdown_support_alpha.size:
        summary["touchdown_support_alpha_mean"] = _safe_float(float(np.nanmean(touchdown_support_alpha)))
        summary["touchdown_support_alpha_max"] = _safe_float(float(np.nanmax(touchdown_support_alpha)))
    if rear_all_contact_stabilization_alpha.size:
        summary["rear_all_contact_stabilization_alpha_mean"] = _safe_float(
            float(np.nanmean(rear_all_contact_stabilization_alpha))
        )
        summary["rear_all_contact_stabilization_alpha_max"] = _safe_float(
            float(np.nanmax(rear_all_contact_stabilization_alpha))
        )
    if rear_all_contact_front_planted_tail_alpha.size:
        summary["rear_all_contact_front_planted_tail_alpha_mean"] = _safe_float(
            float(np.nanmean(rear_all_contact_front_planted_tail_alpha))
        )
        summary["rear_all_contact_front_planted_tail_alpha_max"] = _safe_float(
            float(np.nanmax(rear_all_contact_front_planted_tail_alpha))
        )
    if crawl_front_planted_seam_support_alpha.size:
        summary["crawl_front_planted_seam_support_alpha_mean"] = _safe_float(
            float(np.nanmean(crawl_front_planted_seam_support_alpha))
        )
        summary["crawl_front_planted_seam_support_alpha_max"] = _safe_float(
            float(np.nanmax(crawl_front_planted_seam_support_alpha))
        )
    if rear_late_seam_elapsed_s.ndim == 2 and rear_late_seam_elapsed_s.shape[1] >= 4:
        summary["rear_late_seam_elapsed_max_s"] = {
            "RL": _safe_float(float(np.nanmax(rear_late_seam_elapsed_s[:, 2]))),
            "RR": _safe_float(float(np.nanmax(rear_late_seam_elapsed_s[:, 3]))),
        }
    if rear_late_seam_support_active.ndim == 2 and rear_late_seam_support_active.shape[1] >= 4:
        summary["rear_late_seam_support_ratio"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_seam_support_active[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_seam_support_active[:, 3]))),
        }
    if rear_close_handoff_active.ndim == 2 and rear_close_handoff_active.shape[1] >= 4:
        summary["rear_close_handoff_ratio"] = {
            "RL": _safe_float(float(np.nanmean(rear_close_handoff_active[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_close_handoff_active[:, 3]))),
        }
    if rear_late_load_share_active.ndim == 2 and rear_late_load_share_active.shape[1] >= 4:
        summary["rear_late_load_share_ratio"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_load_share_active[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_load_share_active[:, 3]))),
        }
    if rear_late_load_share_alpha.ndim == 2 and rear_late_load_share_alpha.shape[1] >= 4:
        summary["rear_late_load_share_alpha_mean"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_load_share_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_load_share_alpha[:, 3]))),
        }
        summary["rear_late_load_share_alpha_max"] = {
            "RL": _safe_float(float(np.nanmax(rear_late_load_share_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmax(rear_late_load_share_alpha[:, 3]))),
        }
    if (
        rear_late_load_share_candidate_active.ndim == 2
        and rear_late_load_share_candidate_active.shape[1] >= 4
    ):
        summary["rear_late_load_share_candidate_ratio"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_load_share_candidate_active[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_load_share_candidate_active[:, 3]))),
        }
    if (
        rear_late_load_share_candidate_alpha.ndim == 2
        and rear_late_load_share_candidate_alpha.shape[1] >= 4
    ):
        summary["rear_late_load_share_candidate_alpha_mean"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_load_share_candidate_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_load_share_candidate_alpha[:, 3]))),
        }
        summary["rear_late_load_share_candidate_alpha_max"] = {
            "RL": _safe_float(float(np.nanmax(rear_late_load_share_candidate_alpha[:, 2]))),
            "RR": _safe_float(float(np.nanmax(rear_late_load_share_candidate_alpha[:, 3]))),
        }
    if (
        rear_late_load_share_trigger_elapsed_s.ndim == 2
        and rear_late_load_share_trigger_elapsed_s.shape[1] >= 4
    ):
        summary["rear_late_load_share_trigger_elapsed_mean"] = {
            "RL": _safe_float(float(np.nanmean(rear_late_load_share_trigger_elapsed_s[:, 2]))),
            "RR": _safe_float(float(np.nanmean(rear_late_load_share_trigger_elapsed_s[:, 3]))),
        }
        summary["rear_late_load_share_trigger_elapsed_max"] = {
            "RL": _safe_float(float(np.nanmax(rear_late_load_share_trigger_elapsed_s[:, 2]))),
            "RR": _safe_float(float(np.nanmax(rear_late_load_share_trigger_elapsed_s[:, 3]))),
        }
    if rear_late_load_share_trigger_enabled.size:
        summary["rear_late_load_share_trigger_enabled_mean"] = _safe_float(
            float(np.nanmean(rear_late_load_share_trigger_enabled))
        )
        summary["rear_late_load_share_trigger_enabled_max"] = _safe_float(
            float(np.nanmax(rear_late_load_share_trigger_enabled))
        )
    if rear_close_handoff_alpha.size:
        summary["rear_close_handoff_alpha_mean"] = _safe_float(float(np.nanmean(rear_close_handoff_alpha)))
        summary["rear_close_handoff_alpha_max"] = _safe_float(float(np.nanmax(rear_close_handoff_alpha)))
    if rear_close_handoff_leg_index.size:
        summary["rear_close_handoff_leg_index_last"] = _safe_float(float(rear_close_handoff_leg_index[-1]))
    if rear_all_contact_weak_leg_alpha.size:
        summary["rear_all_contact_weak_leg_alpha_mean"] = _safe_float(
            float(np.nanmean(rear_all_contact_weak_leg_alpha))
        )
        summary["rear_all_contact_weak_leg_alpha_max"] = _safe_float(
            float(np.nanmax(rear_all_contact_weak_leg_alpha))
        )
    if rear_all_contact_weak_leg_index.size:
        summary["rear_all_contact_weak_leg_ratio"] = {
            "RL": _safe_float(float(np.nanmean(rear_all_contact_weak_leg_index == 2.0))),
            "RR": _safe_float(float(np.nanmean(rear_all_contact_weak_leg_index == 3.0))),
        }
        summary["rear_all_contact_weak_leg_index_last"] = _safe_float(
            float(rear_all_contact_weak_leg_index[-1])
        )
    if applied_linear_support_force_floor_ratio.size:
        summary["applied_linear_support_force_floor_ratio_mean"] = _safe_float(
            float(np.nanmean(applied_linear_support_force_floor_ratio))
        )
        summary["applied_linear_support_force_floor_ratio_max"] = _safe_float(
            float(np.nanmax(applied_linear_support_force_floor_ratio))
        )
    if applied_linear_rear_handoff_leg_index.size:
        summary["applied_linear_rear_handoff_leg_ratio"] = {
            "FL": _safe_float(float(np.nanmean(applied_linear_rear_handoff_leg_index == 0.0))),
            "FR": _safe_float(float(np.nanmean(applied_linear_rear_handoff_leg_index == 1.0))),
            "RL": _safe_float(float(np.nanmean(applied_linear_rear_handoff_leg_index == 2.0))),
            "RR": _safe_float(float(np.nanmean(applied_linear_rear_handoff_leg_index == 3.0))),
        }
        summary["applied_linear_rear_handoff_leg_index_last"] = _safe_float(
            float(applied_linear_rear_handoff_leg_index[-1])
        )
    if applied_linear_rear_handoff_leg_floor_scale.size:
        summary["applied_linear_rear_handoff_leg_floor_scale_mean"] = _safe_float(
            float(np.nanmean(applied_linear_rear_handoff_leg_floor_scale))
        )
        summary["applied_linear_rear_handoff_leg_floor_scale_max"] = _safe_float(
            float(np.nanmax(applied_linear_rear_handoff_leg_floor_scale))
        )
    if applied_linear_latched_force_scale.size:
        summary["applied_linear_latched_force_scale_mean"] = _safe_float(
            float(np.nanmean(applied_linear_latched_force_scale))
        )
        summary["applied_linear_latched_force_scale_min"] = _safe_float(
            float(np.nanmin(applied_linear_latched_force_scale))
        )
    if applied_linear_latched_front_receiver_scale.size:
        summary["applied_linear_latched_front_receiver_scale_mean"] = _safe_float(
            float(np.nanmean(applied_linear_latched_front_receiver_scale))
        )
        summary["applied_linear_latched_front_receiver_scale_min"] = _safe_float(
            float(np.nanmin(applied_linear_latched_front_receiver_scale))
        )
    if applied_linear_latched_rear_receiver_scale.size:
        summary["applied_linear_latched_rear_receiver_scale_mean"] = _safe_float(
            float(np.nanmean(applied_linear_latched_rear_receiver_scale))
        )
        summary["applied_linear_latched_rear_receiver_scale_max"] = _safe_float(
            float(np.nanmax(applied_linear_latched_rear_receiver_scale))
        )
    if rear_handoff_support_active.size:
        summary["rear_handoff_support_mean"] = _safe_float(float(np.nanmean(rear_handoff_support_active)))
        summary["rear_handoff_support_max"] = _safe_float(float(np.nanmax(rear_handoff_support_active)))
    if rear_swing_bridge_active.size:
        summary["rear_swing_bridge_mean"] = _safe_float(float(np.nanmean(rear_swing_bridge_active)))
        summary["rear_swing_bridge_max"] = _safe_float(float(np.nanmax(rear_swing_bridge_active)))
    if full_contact_recovery_active.size:
        summary["full_contact_recovery_mean"] = _safe_float(float(np.nanmean(full_contact_recovery_active)))
        summary["full_contact_recovery_max"] = _safe_float(float(np.nanmax(full_contact_recovery_active)))
    if full_contact_recovery_alpha.size:
        summary["full_contact_recovery_alpha_mean"] = _safe_float(float(np.nanmean(full_contact_recovery_alpha)))
        summary["full_contact_recovery_alpha_max"] = _safe_float(float(np.nanmax(full_contact_recovery_alpha)))
    if gate_forward_scale.size:
        summary["gate_forward_scale_mean"] = _safe_float(float(np.nanmean(gate_forward_scale)))
        summary["gate_forward_scale_min"] = _safe_float(float(np.nanmin(gate_forward_scale)))
    if ref_base_height.size:
        summary["ref_base_height"] = _safe_float(np.median(ref_base_height))

    planned_swing_ratio = summary.get("planned_swing_ratio", {})
    current_swing_ratio = summary.get("current_swing_ratio", {})
    actual_swing_ratio = summary.get("actual_swing_ratio", {})

    duration_s = summary.get("duration_s")
    mean_base_z = summary.get("mean_base_z")
    min_base_z = summary.get("min_base_z")
    mean_abs_pitch = summary.get("mean_abs_pitch")
    mean_vx = summary.get("mean_vx")
    steps_any_current_swing = summary.get("steps_any_current_swing")
    legs_with_current_swing = summary.get("legs_with_current_swing")
    steps_any_actual_swing = summary.get("steps_any_actual_swing")
    legs_with_actual_swing = summary.get("legs_with_actual_swing")
    ref_height = summary.get("ref_base_height")
    max_swing_ratio = None
    swing_ratio_source = actual_swing_ratio if isinstance(actual_swing_ratio, dict) and actual_swing_ratio else current_swing_ratio
    if isinstance(swing_ratio_source, dict) and swing_ratio_source:
        swing_values = [v for v in swing_ratio_source.values() if v is not None]
        if swing_values:
            max_swing_ratio = max(swing_values)

    swing_steps_for_gate = steps_any_actual_swing if steps_any_actual_swing is not None else steps_any_current_swing
    swing_legs_for_gate = legs_with_actual_swing if legs_with_actual_swing is not None else legs_with_current_swing

    quality_gate = {
        "duration_ge_5s": bool(duration_s is not None and duration_s >= 5.0),
        "mean_height_ratio_ge_0_75": bool(
            ref_height is not None and mean_base_z is not None and mean_base_z >= 0.75 * ref_height
        ),
        "min_height_ratio_ge_0_55": bool(
            ref_height is not None and min_base_z is not None and min_base_z >= 0.55 * ref_height
        ),
        "mean_abs_pitch_le_0_15": bool(mean_abs_pitch is not None and mean_abs_pitch <= 0.15),
        "mean_vx_ge_0_02": bool(mean_vx is not None and mean_vx >= 0.02),
        "max_leg_actual_swing_ratio_ge_0_05": bool(max_swing_ratio is not None and max_swing_ratio >= 0.05),
        "steps_any_actual_swing_ge_100": bool(
            swing_steps_for_gate is not None and swing_steps_for_gate >= 100
        ),
        "legs_with_actual_swing_ge_4": bool(
            swing_legs_for_gate is not None and swing_legs_for_gate >= 4
        ),
    }
    quality_gate["passes"] = bool(all(quality_gate.values()))
    summary["quality_gate"] = quality_gate
    return summary


def save_npz(final_log: dict[str, Any], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **{k: v for k, v in final_log.items() if k != "meta"}, meta=json.dumps(final_log.get("meta", {})))


def save_summary(summary: dict[str, Any], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_plots(final_log: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    t = np.asarray(final_log.get("time", []), dtype=float).reshape(-1)
    if t.size == 0:
        return saved

    base_lin_vel = np.asarray(final_log.get("base_lin_vel", []), dtype=float)
    base_pos = np.asarray(final_log.get("base_pos", []), dtype=float)
    euler = np.asarray(final_log.get("base_ori_euler_xyz", []), dtype=float)
    action = np.asarray(final_log.get("action", []), dtype=float)
    foot_contact = np.asarray(final_log.get("foot_contact", []), dtype=float)
    foot_grf = np.asarray(final_log.get("foot_grf", []), dtype=float)

    if base_lin_vel.ndim == 2 and base_lin_vel.shape[1] >= 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, base_lin_vel[:, 0], label="vx")
        if base_lin_vel.shape[1] >= 2:
            ax.plot(t, base_lin_vel[:, 1], label="vy")
        ax.set_title("Base Linear Velocity")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / "fig_velocity_tracking.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    if base_pos.ndim == 2 and base_pos.shape[1] >= 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, base_pos[:, 2])
        ax.set_title("Base Height")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("z [m]")
        ax.grid(True, alpha=0.3)
        path = out_dir / "fig_base_height.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(base_pos[:, 0], base_pos[:, 1], lw=2)
        ax.scatter(base_pos[0, 0], base_pos[0, 1], c="green", label="start")
        ax.scatter(base_pos[-1, 0], base_pos[-1, 1], c="red", label="end")
        ax.set_title("Base XY Path")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / "fig_xy_path.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    if euler.ndim == 2 and euler.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, euler[:, 0], label="roll")
        ax.plot(t, euler[:, 1], label="pitch")
        if euler.shape[1] >= 3:
            ax.plot(t, euler[:, 2], label="yaw")
        ax.set_title("Base Orientation (Euler XYZ)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / "fig_roll_pitch_yaw.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    if action.ndim == 2 and action.shape[1] >= 12:
        leg_norms = np.stack([
            np.linalg.norm(action[:, 0:3], axis=1),
            np.linalg.norm(action[:, 3:6], axis=1),
            np.linalg.norm(action[:, 6:9], axis=1),
            np.linalg.norm(action[:, 9:12], axis=1),
        ], axis=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            ax.plot(t, leg_norms[:, i], label=leg)
        ax.set_title("Per-leg Action Norm")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|tau| or |u|")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4)
        path = out_dir / "fig_action_leg_norms.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    if foot_contact.ndim == 2 and foot_contact.shape[1] >= 4:
        fig, axes = plt.subplots(4, 1, figsize=(9, 5), sharex=True)
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            axes[i].plot(t, foot_contact[:, i], drawstyle="steps-post")
            axes[i].set_ylabel(leg)
            axes[i].set_ylim(-0.1, 1.1)
            axes[i].grid(True, alpha=0.3)
        axes[0].set_title("Actual Foot Contact")
        axes[-1].set_xlabel("time [s]")
        path = out_dir / "fig_contact_timeline.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    if foot_grf.ndim >= 3 and foot_grf.shape[1] >= 4:
        norms = np.linalg.norm(foot_grf[:, :4, :], axis=2)
        fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            axes[i].plot(t, norms[:, i])
            axes[i].set_ylabel(leg)
            axes[i].grid(True, alpha=0.3)
        axes[0].set_title("Per-leg Ground Reaction Force Norm")
        axes[-1].set_xlabel("time [s]")
        path = out_dir / "fig_grf_norms.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        saved.append(path)

    return saved


def save_topdown_mp4(final_log: dict[str, Any], out_file: Path, fps: int = 25) -> Path | None:
    import shutil
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    base_pos = np.asarray(final_log.get("base_pos", []), dtype=float)
    feet_pos = np.asarray(final_log.get("feet_pos", []), dtype=float)
    time = np.asarray(final_log.get("time", []), dtype=float).reshape(-1)
    foot_contact = np.asarray(final_log.get("foot_contact", []), dtype=float)
    if base_pos.ndim != 2 or base_pos.shape[0] < 2 or feet_pos.ndim < 3:
        return None

    duration = max(float(time[-1] - time[0]) if len(time) >= 2 else 0.0, 1e-6)
    target_frames = min(400, max(60, int(duration * fps)))
    step = max(1, int(round(len(time) / target_frames)))
    idxs = np.arange(0, len(time), step)
    if idxs[-1] != len(time) - 1:
        idxs = np.append(idxs, len(time) - 1)

    x_all = np.concatenate([base_pos[:, 0], feet_pos[:, :, 0].reshape(-1)])
    y_all = np.concatenate([base_pos[:, 1], feet_pos[:, :, 1].reshape(-1)])
    x_pad = 0.15 * max(0.2, float(x_all.max() - x_all.min()))
    y_pad = 0.15 * max(0.2, float(y_all.max() - y_all.min()))

    fig, ax = plt.subplots(figsize=(6, 6))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def _draw_frame(idx: int) -> None:
        ax.cla()
        ax.set_title(f"Top-down motion | t={time[idx]:.2f}s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x_all.min() - x_pad), float(x_all.max() + x_pad))
        ax.set_ylim(float(y_all.min() - y_pad), float(y_all.max() + y_pad))
        ax.plot(base_pos[: idx + 1, 0], base_pos[: idx + 1, 1], color="tab:blue", lw=2, alpha=0.7)
        ax.scatter(base_pos[idx, 0], base_pos[idx, 1], color="tab:blue", s=70, label="base")
        fp = feet_pos[idx]
        if fp.ndim == 2 and fp.shape[0] >= 4:
            colors = ["tab:green", "tab:orange", "tab:red", "tab:purple"]
            labels = ["FL", "FR", "RL", "RR"]
            contacts = foot_contact[idx] if foot_contact.ndim == 2 and foot_contact.shape[1] >= 4 else np.zeros(4)
            for leg_i in range(4):
                marker = "o" if contacts[leg_i] > 0.5 else "x"
                ax.scatter(fp[leg_i, 0], fp[leg_i, 1], color=colors[leg_i], s=60, marker=marker, label=labels[leg_i])
                ax.plot([base_pos[idx, 0], fp[leg_i, 0]], [base_pos[idx, 1], fp[leg_i, 1]], color=colors[leg_i], alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        save_path = out_file
        try:
            with writer.saving(fig, str(save_path), dpi=140):
                for idx in idxs:
                    _draw_frame(idx)
                    writer.grab_frame()
            plt.close(fig)
            return save_path
        except FileNotFoundError:
            pass

    writer = PillowWriter(fps=max(1, fps // 2))
    save_path = out_file.with_suffix(".gif")
    with writer.saving(fig, str(save_path), dpi=140):
        for idx in idxs:
            _draw_frame(idx)
            writer.grab_frame()
    plt.close(fig)
    return save_path
