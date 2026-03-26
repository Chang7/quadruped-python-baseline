
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import mujoco

from baseline.config import MPCConfig, LEG_NAMES
from phases.mujoco_phase2_helpers import (
    LegBinding,
    ModelBindings,
    discover_model_bindings,
    foot_point_world,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    print_binding_summary,
)
from phases.mujoco_phase3_helpers import (
    FALLBACK_FOOT_LOCAL_OFFSET,
    cycle_time,
    stance_fraction,
    leg_phase_offset,
    swing_phase_s,
    compute_swing_delta_maps,
    resolve_home_ctrl,
)


def _clip_targets_to_ctrlrange(m: mujoco.MjModel, actuator_ids: list[int], target: np.ndarray) -> np.ndarray:
    out = np.asarray(target, dtype=float).copy()
    for i, aid in enumerate(actuator_ids):
        if 0 <= aid < m.nu:
            lo, hi = np.asarray(m.actuator_ctrlrange[aid], dtype=float)
            if hi > lo:
                out[i] = float(np.clip(out[i], lo, hi))
    return out


def smoothstep01(x: float) -> float:
    s = float(np.clip(x, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def initialize_phase4_leg_state(bindings: ModelBindings) -> None:
    for leg in bindings.leg_bindings:
        leg.swing_anchor_q = None
        leg.stance_anchor_q = None
        leg.stance_start_time = None
        leg.touchdown_recorded = False
        leg.touchdown_delays = []
        leg.sched_stance_samples = 0
        leg.actual_contact_samples = 0
        leg.force_enabled_samples = 0
        leg.touchdown_search_samples = 0


def update_phase4_state_pre(
    d: mujoco.MjData,
    bindings: ModelBindings,
    prev_sched: np.ndarray,
    sched: np.ndarray,
) -> None:
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)

        if not bool(sched[leg_idx]):
            if bool(prev_sched[leg_idx]) or leg.swing_anchor_q is None:
                leg.swing_anchor_q = current_q.copy()
            leg.stance_anchor_q = None
            leg.stance_start_time = None
            leg.touchdown_recorded = False
        else:
            if not bool(prev_sched[leg_idx]):
                leg.stance_anchor_q = current_q.copy()
                leg.stance_start_time = float(d.time)
                leg.touchdown_recorded = False
            elif leg.stance_anchor_q is None:
                leg.stance_anchor_q = current_q.copy()
                if leg.stance_start_time is None:
                    leg.stance_start_time = float(d.time)


def update_phase4_state_post(
    d: mujoco.MjData,
    bindings: ModelBindings,
    sched: np.ndarray,
    actual: np.ndarray,
    force_enabled: np.ndarray,
) -> None:
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if bool(sched[leg_idx]):
            leg.sched_stance_samples += 1
        if bool(actual[leg_idx]):
            leg.actual_contact_samples += 1
        if bool(force_enabled[leg_idx]):
            leg.force_enabled_samples += 1
        if bool(sched[leg_idx]) and not bool(actual[leg_idx]):
            leg.touchdown_search_samples += 1

        if bool(sched[leg_idx]) and bool(actual[leg_idx]) and not getattr(leg, "touchdown_recorded", False):
            t0 = getattr(leg, "stance_start_time", None)
            if t0 is not None:
                leg.touchdown_delays.append(float(max(0.0, d.time - t0)))
            leg.touchdown_recorded = True


def swing_profiles_phase4(s: float) -> tuple[float, float, float]:
    """Return (x_forward_profile, z_lift_profile, z_drop_profile)."""
    s = float(np.clip(s, 0.0, 1.0))
    x_prof = smoothstep01(s)

    if s < 0.45:
        z_lift = math.sin(0.5 * math.pi * (s / 0.45))
    elif s < 0.80:
        z_lift = math.cos(0.5 * math.pi * ((s - 0.45) / 0.35))
    else:
        z_lift = 0.0
    z_lift = max(0.0, float(z_lift))

    z_drop = smoothstep01((s - 0.78) / 0.22)
    return x_prof, z_lift, z_drop


def build_ctrl_targets_phase4(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    bindings: ModelBindings,
    home_ctrl: np.ndarray,
    scheduled_contact: np.ndarray,
    actual_contact: np.ndarray,
    cfg: MPCConfig,
    clearance: float,
    step_len: float,
    touchdown_depth: float,
    touchdown_forward: float,
    touchdown_search_window: float,
) -> np.ndarray:
    if home_ctrl.size == 0:
        return home_ctrl

    ctrl = home_ctrl.copy()
    t = float(d.time)

    for leg_idx, leg in enumerate(bindings.leg_bindings):
        if not leg.actuator_ids or leg.home_joint_qpos is None:
            continue

        current_q = np.array([d.qpos[adr] for adr in leg.qpos_adrs], dtype=float)
        dq_up = getattr(leg, "swing_dq_up", np.zeros_like(current_q))
        dq_fwd = getattr(leg, "swing_dq_fwd", np.zeros_like(current_q))

        if not bool(scheduled_contact[leg_idx]):
            if leg.swing_anchor_q is None:
                leg.swing_anchor_q = current_q.copy()

            s = swing_phase_s(t, leg_idx, cfg)
            x_prof, z_lift, z_drop = swing_profiles_phase4(s)

            target = (
                leg.swing_anchor_q
                + clearance * z_lift * dq_up
                + step_len * x_prof * dq_fwd
                - touchdown_depth * z_drop * dq_up
            )
            target = 0.92 * target + 0.08 * leg.home_joint_qpos

        elif bool(actual_contact[leg_idx]):
            # True stance: do not let the servo fight the externally applied MPC GRF.
            target = current_q

        else:
            # Scheduled stance, but the leg has not actually touched down yet.
            if leg.stance_anchor_q is None:
                leg.stance_anchor_q = current_q.copy()
            if leg.stance_start_time is None:
                leg.stance_start_time = t

            tau = smoothstep01((t - float(leg.stance_start_time)) / max(touchdown_search_window, 1e-6))
            target = (
                leg.stance_anchor_q
                + touchdown_forward * (0.25 + 0.75 * tau) * dq_fwd
                - touchdown_depth * (0.35 + 0.65 * tau) * dq_up
            )
            target = 0.95 * target + 0.05 * leg.home_joint_qpos

        target = _clip_targets_to_ctrlrange(m, leg.actuator_ids, target)
        ctrl[np.asarray(leg.actuator_ids, dtype=int)] = target

    return ctrl


def stance_force_enable_mask(scheduled_contact: np.ndarray, actual_contact: np.ndarray) -> np.ndarray:
    return np.logical_and(np.asarray(scheduled_contact, dtype=bool), np.asarray(actual_contact, dtype=bool))


def build_phase4_summary(log: dict, bindings: ModelBindings) -> dict:
    t = np.asarray(log["t"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    sched = np.asarray(log["contact"], dtype=bool)
    actual = np.asarray(log["contact_actual"], dtype=bool)
    enabled = np.asarray(log.get("contact_force_enabled", []), dtype=bool)

    after_mask = t >= 1.0
    if np.any(after_mask):
        vx_mean_after_1s = float(np.mean(x[after_mask, 3]))
    else:
        vx_mean_after_1s = float(np.mean(x[:, 3])) if x.size else 0.0

    per_leg = []
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        sched_ratio = float(np.mean(sched[:, leg_idx])) if sched.size else 0.0
        actual_ratio = float(np.mean(actual[:, leg_idx])) if actual.size else 0.0
        mismatch = float(np.mean(sched[:, leg_idx] != actual[:, leg_idx])) if sched.size else 0.0
        stance_success = (
            float(np.mean(actual[sched[:, leg_idx], leg_idx])) if np.any(sched[:, leg_idx]) else 0.0
        )
        enabled_ratio = float(np.mean(enabled[:, leg_idx])) if enabled.size else 0.0
        td = list(getattr(leg, "touchdown_delays", []))
        per_leg.append(
            {
                "leg": LEG_NAMES[leg_idx],
                "scheduled_stance_ratio": sched_ratio,
                "actual_contact_ratio": actual_ratio,
                "mismatch_ratio": mismatch,
                "stance_success_ratio": stance_success,
                "force_enabled_ratio": enabled_ratio,
                "touchdown_delay_mean_s": None if len(td) == 0 else float(np.mean(td)),
                "touchdown_delay_max_s": None if len(td) == 0 else float(np.max(td)),
                "touchdown_count": int(len(td)),
            }
        )

    summary = {
        "mean_mismatch_ratio": float(np.mean([item["mismatch_ratio"] for item in per_leg])) if per_leg else 0.0,
        "mean_vx_after_1s": vx_mean_after_1s,
        "per_leg": per_leg,
    }
    return summary


def write_phase4_summary(output_dir: str | Path, summary: dict) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "phase4_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)
