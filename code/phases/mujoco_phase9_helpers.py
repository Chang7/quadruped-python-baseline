from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mujoco

from baseline.config import LEG_NAMES
from phases.mujoco_phase8_helpers import build_phase8_summary, save_phase8_plots


def _leg_contact_geom_ids(m: mujoco.MjModel, bindings, foot_only: bool) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        ids: list[int] = []
        for gid in getattr(leg, 'geom_ids', []):
            gid = int(gid)
            if gid < 0 or gid >= m.ngeom:
                continue
            if int(m.geom_contype[gid]) == 0 or int(m.geom_conaffinity[gid]) == 0:
                continue
            if foot_only and leg.foot_geom_id is not None and gid != int(leg.foot_geom_id):
                continue
            ids.append(gid)
        if foot_only and leg.foot_geom_id is not None and int(leg.foot_geom_id) not in ids:
            gid = int(leg.foot_geom_id)
            if 0 <= gid < m.ngeom and int(m.geom_contype[gid]) > 0 and int(m.geom_conaffinity[gid]) > 0:
                ids.append(gid)
        out[leg_idx] = ids
    return out


def disable_nonfoot_leg_collisions(m: mujoco.MjModel, bindings) -> list[tuple[str, int]]:
    """Disable contact on collision-enabled non-foot leg geoms.

    Returns a list of (leg_name, geom_id) that were disabled.
    """
    disabled: list[tuple[str, int]] = []
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        foot_gid = None if leg.foot_geom_id is None else int(leg.foot_geom_id)
        for gid in getattr(leg, 'geom_ids', []):
            gid = int(gid)
            if gid == foot_gid:
                continue
            if gid < 0 or gid >= m.ngeom:
                continue
            if int(m.geom_contype[gid]) == 0 and int(m.geom_conaffinity[gid]) == 0:
                continue
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
            disabled.append((LEG_NAMES[leg_idx], gid))
    return disabled


def _accumulate_candidates_for_geom_sets(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    leg_geom_ids: dict[int, list[int]],
    floor_geom_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    out_a = np.zeros((len(leg_geom_ids), 3), dtype=float)
    out_b = np.zeros_like(out_a)
    wrench_cf = np.zeros(6, dtype=float)

    geom_to_leg: dict[int, int] = {}
    for leg_idx, gids in leg_geom_ids.items():
        for gid in gids:
            geom_to_leg[int(gid)] = int(leg_idx)

    for cid in range(int(d.ncon)):
        con = d.contact[cid]
        g1 = int(con.geom1)
        g2 = int(con.geom2)

        leg_idx = None
        sign_a = None
        if g1 in geom_to_leg and g2 in floor_geom_ids:
            leg_idx = geom_to_leg[g1]
            sign_a = 1.0
        elif g2 in geom_to_leg and g1 in floor_geom_ids:
            leg_idx = geom_to_leg[g2]
            sign_a = -1.0
        else:
            continue

        mujoco.mj_contactForce(m, d, cid, wrench_cf)
        f_cf = wrench_cf[:3].copy()
        R_rows = np.asarray(con.frame, dtype=float).reshape(3, 3)
        f_world = R_rows.T @ f_cf

        out_a[leg_idx] += sign_a * f_world
        out_b[leg_idx] -= sign_a * f_world

    return out_a, out_b


def accumulate_actual_foot_and_support_candidates(m: mujoco.MjModel, d: mujoco.MjData, bindings):
    floor_geom_ids = {int(gid) for gid in getattr(bindings, 'floor_geom_ids', [])}
    foot_sets = _leg_contact_geom_ids(m, bindings, foot_only=True)
    support_sets = _leg_contact_geom_ids(m, bindings, foot_only=False)

    foot_a, foot_b = _accumulate_candidates_for_geom_sets(m, d, foot_sets, floor_geom_ids)
    support_a, support_b = _accumulate_candidates_for_geom_sets(m, d, support_sets, floor_geom_ids)
    return foot_a, foot_b, support_a, support_b


def _choose_candidate(a: np.ndarray, b: np.ndarray, t: np.ndarray, t_start: float = 0.20):
    if a.size == 0 and b.size == 0:
        return np.zeros((len(t), len(LEG_NAMES), 3), dtype=float), 'none', {
            'candidate_a_mean_sum_fz_after_selection_window': 0.0,
            'candidate_b_mean_sum_fz_after_selection_window': 0.0,
        }
    if a.size == 0:
        return b, 'candidate_b_only', {
            'candidate_a_mean_sum_fz_after_selection_window': 0.0,
            'candidate_b_mean_sum_fz_after_selection_window': float(np.mean(np.sum(b[:, :, 2], axis=1))),
        }
    if b.size == 0:
        return a, 'candidate_a_only', {
            'candidate_a_mean_sum_fz_after_selection_window': float(np.mean(np.sum(a[:, :, 2], axis=1))),
            'candidate_b_mean_sum_fz_after_selection_window': 0.0,
        }

    mask = t >= t_start
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    mean_fz_a = float(np.mean(np.sum(a[:, :, 2], axis=1)[mask]))
    mean_fz_b = float(np.mean(np.sum(b[:, :, 2], axis=1)[mask]))
    if mean_fz_a >= mean_fz_b:
        return a, 'candidate_a(geom1_plus)', {
            'candidate_a_mean_sum_fz_after_selection_window': mean_fz_a,
            'candidate_b_mean_sum_fz_after_selection_window': mean_fz_b,
        }
    return b, 'candidate_b(geom1_minus)', {
        'candidate_a_mean_sum_fz_after_selection_window': mean_fz_a,
        'candidate_b_mean_sum_fz_after_selection_window': mean_fz_b,
    }


def build_phase9_summary(log: dict, bindings, disable_nonfoot_collision: bool = False) -> dict:
    # Reuse phase-8 summary for foot-only contacts using the standard keys.
    summary = build_phase8_summary(log, bindings)

    t = np.asarray(log.get('t', []), dtype=float)
    after_mask = t >= 1.0
    if t.size and not np.any(after_mask):
        after_mask = np.ones_like(t, dtype=bool)

    actual_contact = np.asarray(log.get('contact_actual', []), dtype=bool)
    enabled = np.asarray(log.get('contact_force_enabled', []), dtype=bool)

    support_a = np.asarray(log.get('actual_support_a', []), dtype=float)
    support_b = np.asarray(log.get('actual_support_b', []), dtype=float)
    support, support_convention, support_conv_stats = _choose_candidate(support_a, support_b, t)

    if support.size:
        sum_fx = np.sum(support[:, :, 0], axis=1)
        sum_fy = np.sum(support[:, :, 1], axis=1)
        sum_fz = np.sum(support[:, :, 2], axis=1)
        mean_support_sum_fx_after_1s = float(np.mean(sum_fx[after_mask]))
        mean_support_sum_fy_after_1s = float(np.mean(sum_fy[after_mask]))
        mean_support_sum_fz_after_1s = float(np.mean(sum_fz[after_mask]))
    else:
        mean_support_sum_fx_after_1s = 0.0
        mean_support_sum_fy_after_1s = 0.0
        mean_support_sum_fz_after_1s = 0.0

    per_leg = []
    foot_per_leg = summary['per_leg']
    for leg_idx, item in enumerate(foot_per_leg):
        leg_item = dict(item)
        if support.size:
            if actual_contact.size and np.any(actual_contact[:, leg_idx]):
                mask_contact = actual_contact[:, leg_idx]
            else:
                mask_contact = np.ones(support.shape[0], dtype=bool)
            if enabled.size and np.any(enabled[:, leg_idx]):
                mask_enabled = enabled[:, leg_idx]
            else:
                mask_enabled = mask_contact

            leg_item['mean_actual_support_fx_when_contact'] = float(np.mean(support[mask_contact, leg_idx, 0])) if np.any(mask_contact) else 0.0
            leg_item['mean_actual_support_fy_when_contact'] = float(np.mean(support[mask_contact, leg_idx, 1])) if np.any(mask_contact) else 0.0
            leg_item['mean_actual_support_fz_when_contact'] = float(np.mean(support[mask_contact, leg_idx, 2])) if np.any(mask_contact) else 0.0
            leg_item['mean_actual_support_fx_when_enabled'] = float(np.mean(support[mask_enabled, leg_idx, 0])) if np.any(mask_enabled) else 0.0
            leg_item['mean_actual_support_fy_when_enabled'] = float(np.mean(support[mask_enabled, leg_idx, 1])) if np.any(mask_enabled) else 0.0
            leg_item['mean_actual_support_fz_when_enabled'] = float(np.mean(support[mask_enabled, leg_idx, 2])) if np.any(mask_enabled) else 0.0
        else:
            leg_item['mean_actual_support_fx_when_contact'] = 0.0
            leg_item['mean_actual_support_fy_when_contact'] = 0.0
            leg_item['mean_actual_support_fz_when_contact'] = 0.0
            leg_item['mean_actual_support_fx_when_enabled'] = 0.0
            leg_item['mean_actual_support_fy_when_enabled'] = 0.0
            leg_item['mean_actual_support_fz_when_enabled'] = 0.0
        leg_item['support_minus_foot_fz_when_enabled'] = float(
            leg_item['mean_actual_support_fz_when_enabled'] - leg_item.get('mean_actual_fz_when_enabled', 0.0)
        )
        per_leg.append(leg_item)

    summary['per_leg'] = per_leg
    summary['disable_nonfoot_collision'] = bool(disable_nonfoot_collision)
    summary['actual_support_sign_convention'] = support_convention
    summary.update({f'support_{k}': v for k, v in support_conv_stats.items()})
    summary['mean_actual_support_sum_fx_after_1s'] = mean_support_sum_fx_after_1s
    summary['mean_actual_support_sum_fy_after_1s'] = mean_support_sum_fy_after_1s
    summary['mean_actual_support_sum_fz_after_1s'] = mean_support_sum_fz_after_1s
    summary['mean_support_minus_command_sum_fx_after_1s'] = float(mean_support_sum_fx_after_1s - summary['mean_sum_fx_after_1s'])
    summary['mean_support_minus_command_sum_fy_after_1s'] = float(mean_support_sum_fy_after_1s - summary['mean_sum_fy_after_1s'])
    summary['mean_support_minus_command_sum_fz_after_1s'] = float(mean_support_sum_fz_after_1s - summary['mean_sum_fz_after_1s'])
    return summary


def write_phase9_summary(output_dir: str | Path, summary: dict) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'phase9_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return str(path)


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def save_phase9_plots(log: dict, output_dir: str | Path) -> list[str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Reuse phase-8 plots first.
    saved.extend(save_phase8_plots(log, output_dir=outdir))

    t = np.asarray(log.get('t', []), dtype=float)
    cmd = np.asarray(log.get('u_applied', []), dtype=float)
    foot_a = np.asarray(log.get('actual_grf_a', []), dtype=float)
    foot_b = np.asarray(log.get('actual_grf_b', []), dtype=float)
    support_a = np.asarray(log.get('actual_support_a', []), dtype=float)
    support_b = np.asarray(log.get('actual_support_b', []), dtype=float)
    foot, foot_conv, _ = _choose_candidate(foot_a, foot_b, t)
    support, support_conv, _ = _choose_candidate(support_a, support_b, t)

    if t.size == 0:
        return saved

    cmd_sum = np.zeros((t.size, 3), dtype=float)
    if cmd.size:
        cmd_sum[:, 0] = np.sum(cmd[:, 0::3], axis=1)
        cmd_sum[:, 1] = np.sum(cmd[:, 1::3], axis=1)
        cmd_sum[:, 2] = np.sum(cmd[:, 2::3], axis=1)
    foot_sum = np.zeros((t.size, 3), dtype=float)
    if foot.size:
        foot_sum[:, 0] = np.sum(foot[:, :, 0], axis=1)
        foot_sum[:, 1] = np.sum(foot[:, :, 1], axis=1)
        foot_sum[:, 2] = np.sum(foot[:, :, 2], axis=1)
    support_sum = np.zeros((t.size, 3), dtype=float)
    if support.size:
        support_sum[:, 0] = np.sum(support[:, :, 0], axis=1)
        support_sum[:, 1] = np.sum(support[:, :, 1], axis=1)
        support_sum[:, 2] = np.sum(support[:, :, 2], axis=1)

    fig1, axes = plt.subplots(3, 1, figsize=(6.5, 6.8), sharex=True)
    labels = ['Fx', 'Fy', 'Fz']
    for j, ax in enumerate(axes):
        ax.plot(t, cmd_sum[:, j], linewidth=1.6, label=f'commanded sum {labels[j]}')
        ax.plot(t, foot_sum[:, j], '--', linewidth=1.5, label=f'foot-only actual sum {labels[j]}')
        ax.plot(t, support_sum[:, j], ':', linewidth=1.7, label=f'all-support actual sum {labels[j]}')
        ax.set_ylabel(f'{labels[j]} [N]')
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(frameon=True, fontsize=7)
    axes[-1].set_xlabel('time [s]')
    fig1.suptitle(f'Commanded vs actual support force (foot={foot_conv}, support={support_conv})', fontsize=10.5)
    fig1.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    saved.append(_save(fig1, outdir / 'fig_total_commanded_vs_foot_vs_support_force.png'))

    fig2, axes2 = plt.subplots(4, 1, figsize=(6.5, 7.2), sharex=True)
    for leg_idx, ax in enumerate(axes2):
        if foot.size:
            ax.plot(t, foot[:, leg_idx, 2], '--', linewidth=1.4, label=f'{LEG_NAMES[leg_idx]} foot-only actual $F_z$')
        if support.size:
            ax.plot(t, support[:, leg_idx, 2], ':', linewidth=1.6, label=f'{LEG_NAMES[leg_idx]} all-support actual $F_z$')
        ax.set_ylabel('Fz [N]')
        ax.set_title(LEG_NAMES[leg_idx], loc='left', fontsize=10, pad=2)
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(frameon=True, fontsize=7, loc='upper right')
    axes2[-1].set_xlabel('time [s]')
    fig2.tight_layout(h_pad=0.8)
    saved.append(_save(fig2, outdir / 'fig_per_leg_foot_vs_support_fz.png'))

    return saved
