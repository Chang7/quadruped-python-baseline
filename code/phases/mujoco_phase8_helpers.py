from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mujoco

from baseline.config import LEG_NAMES
from phases.mujoco_phase6_helpers import build_phase6_summary


def accumulate_actual_grf_candidates(m: mujoco.MjModel, d: mujoco.MjData, bindings) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate foot-floor contact forces for two possible sign conventions.

    MuJoCo's mj_contactForce returns the contact wrench in the contact frame, but the
    sign convention with respect to geom1 / geom2 is easy to misread in practice.
    To keep the logger robust, we accumulate *two* candidates:

    - candidate A: + when foot geom is geom1, - when foot geom is geom2
    - candidate B: the exact opposite

    Later, the summary picks whichever candidate yields positive mean total Fz on the
    robot after the initial transient.
    """
    foot_geom_to_leg = {
        int(leg.foot_geom_id): leg_idx
        for leg_idx, leg in enumerate(bindings.leg_bindings)
        if leg.foot_geom_id is not None
    }
    floor_geom_ids = {int(gid) for gid in bindings.floor_geom_ids}

    out_a = np.zeros((len(bindings.leg_bindings), 3), dtype=float)
    out_b = np.zeros_like(out_a)
    wrench_cf = np.zeros(6, dtype=float)

    for cid in range(int(d.ncon)):
        con = d.contact[cid]
        g1 = int(con.geom1)
        g2 = int(con.geom2)

        leg_idx = None
        sign_a = None
        if g1 in foot_geom_to_leg and g2 in floor_geom_ids:
            leg_idx = foot_geom_to_leg[g1]
            sign_a = 1.0
        elif g2 in foot_geom_to_leg and g1 in floor_geom_ids:
            leg_idx = foot_geom_to_leg[g2]
            sign_a = -1.0
        else:
            continue

        mujoco.mj_contactForce(m, d, cid, wrench_cf)
        f_cf = wrench_cf[:3].copy()

        # MuJoCo stores the contact basis vectors row-wise in con.frame.
        R_rows = np.asarray(con.frame, dtype=float).reshape(3, 3)
        f_world = R_rows.T @ f_cf

        out_a[leg_idx] += sign_a * f_world
        out_b[leg_idx] -= sign_a * f_world

    return out_a, out_b


def choose_actual_grf_candidate(log: dict, t_start: float = 0.20) -> tuple[np.ndarray, str, dict]:
    t = np.asarray(log.get('t', []), dtype=float)
    cand_a = np.asarray(log.get('actual_grf_a', []), dtype=float)
    cand_b = np.asarray(log.get('actual_grf_b', []), dtype=float)

    if cand_a.size == 0 and cand_b.size == 0:
        empty = np.zeros((len(t), len(LEG_NAMES), 3), dtype=float)
        return empty, 'none', {
            'candidate_a_mean_sum_fz_after_selection_window': 0.0,
            'candidate_b_mean_sum_fz_after_selection_window': 0.0,
        }

    if cand_a.size == 0:
        return cand_b, 'candidate_b_only', {
            'candidate_a_mean_sum_fz_after_selection_window': 0.0,
            'candidate_b_mean_sum_fz_after_selection_window': float(np.mean(np.sum(cand_b[:, :, 2], axis=1))),
        }
    if cand_b.size == 0:
        return cand_a, 'candidate_a_only', {
            'candidate_a_mean_sum_fz_after_selection_window': float(np.mean(np.sum(cand_a[:, :, 2], axis=1))),
            'candidate_b_mean_sum_fz_after_selection_window': 0.0,
        }

    mask = t >= t_start
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)

    sum_fz_a = np.sum(cand_a[:, :, 2], axis=1)
    sum_fz_b = np.sum(cand_b[:, :, 2], axis=1)
    mean_fz_a = float(np.mean(sum_fz_a[mask]))
    mean_fz_b = float(np.mean(sum_fz_b[mask]))

    # Pick the convention that yields more positive support force on the robot.
    if mean_fz_a >= mean_fz_b:
        chosen = cand_a
        label = 'candidate_a(foot_geom1_plus)'
    else:
        chosen = cand_b
        label = 'candidate_b(foot_geom1_minus)'

    return chosen, label, {
        'candidate_a_mean_sum_fz_after_selection_window': mean_fz_a,
        'candidate_b_mean_sum_fz_after_selection_window': mean_fz_b,
    }


def build_phase8_summary(log: dict, bindings) -> dict:
    summary = build_phase6_summary(log, bindings)

    t = np.asarray(log.get('t', []), dtype=float)
    after_mask = t >= 1.0
    if t.size and not np.any(after_mask):
        after_mask = np.ones_like(t, dtype=bool)

    actual_contact = np.asarray(log.get('contact_actual', []), dtype=bool)
    enabled = np.asarray(log.get('contact_force_enabled', []), dtype=bool)
    actual, convention, conv_stats = choose_actual_grf_candidate(log)

    if actual.size:
        sum_fx = np.sum(actual[:, :, 0], axis=1)
        sum_fy = np.sum(actual[:, :, 1], axis=1)
        sum_fz = np.sum(actual[:, :, 2], axis=1)
        mean_actual_sum_fx_after_1s = float(np.mean(sum_fx[after_mask]))
        mean_actual_sum_fy_after_1s = float(np.mean(sum_fy[after_mask]))
        mean_actual_sum_fz_after_1s = float(np.mean(sum_fz[after_mask]))
    else:
        mean_actual_sum_fx_after_1s = 0.0
        mean_actual_sum_fy_after_1s = 0.0
        mean_actual_sum_fz_after_1s = 0.0

    per_leg = []
    for leg_idx, item in enumerate(summary['per_leg']):
        leg_item = dict(item)
        if actual.size:
            if actual_contact.size and np.any(actual_contact[:, leg_idx]):
                mask_contact = actual_contact[:, leg_idx]
            else:
                mask_contact = np.ones(actual.shape[0], dtype=bool)

            if enabled.size and np.any(enabled[:, leg_idx]):
                mask_enabled = enabled[:, leg_idx]
            else:
                mask_enabled = mask_contact

            leg_item['mean_actual_fx_when_contact'] = float(np.mean(actual[mask_contact, leg_idx, 0])) if np.any(mask_contact) else 0.0
            leg_item['mean_actual_fy_when_contact'] = float(np.mean(actual[mask_contact, leg_idx, 1])) if np.any(mask_contact) else 0.0
            leg_item['mean_actual_fz_when_contact'] = float(np.mean(actual[mask_contact, leg_idx, 2])) if np.any(mask_contact) else 0.0

            leg_item['mean_actual_fx_when_enabled'] = float(np.mean(actual[mask_enabled, leg_idx, 0])) if np.any(mask_enabled) else 0.0
            leg_item['mean_actual_fy_when_enabled'] = float(np.mean(actual[mask_enabled, leg_idx, 1])) if np.any(mask_enabled) else 0.0
            leg_item['mean_actual_fz_when_enabled'] = float(np.mean(actual[mask_enabled, leg_idx, 2])) if np.any(mask_enabled) else 0.0
        else:
            leg_item['mean_actual_fx_when_contact'] = 0.0
            leg_item['mean_actual_fy_when_contact'] = 0.0
            leg_item['mean_actual_fz_when_contact'] = 0.0
            leg_item['mean_actual_fx_when_enabled'] = 0.0
            leg_item['mean_actual_fy_when_enabled'] = 0.0
            leg_item['mean_actual_fz_when_enabled'] = 0.0
        per_leg.append(leg_item)

    summary['per_leg'] = per_leg
    summary['actual_grf_sign_convention'] = convention
    summary.update(conv_stats)
    summary['mean_actual_sum_fx_after_1s'] = mean_actual_sum_fx_after_1s
    summary['mean_actual_sum_fy_after_1s'] = mean_actual_sum_fy_after_1s
    summary['mean_actual_sum_fz_after_1s'] = mean_actual_sum_fz_after_1s
    summary['mean_command_minus_actual_sum_fx_after_1s'] = float(summary['mean_sum_fx_after_1s'] - mean_actual_sum_fx_after_1s)
    summary['mean_command_minus_actual_sum_fy_after_1s'] = float(summary['mean_sum_fy_after_1s'] - mean_actual_sum_fy_after_1s)
    summary['mean_command_minus_actual_sum_fz_after_1s'] = float(summary['mean_sum_fz_after_1s'] - mean_actual_sum_fz_after_1s)
    return summary


def write_phase8_summary(output_dir: str | Path, summary: dict) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'phase8_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return str(path)


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def save_phase8_plots(log: dict, output_dir: str | Path) -> list[str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(log.get('t', []), dtype=float)
    cmd = np.asarray(log.get('u_applied', []), dtype=float)
    actual, convention, _ = choose_actual_grf_candidate(log)

    if t.size == 0:
        return []

    saved: list[str] = []

    cmd_sum = np.zeros((t.size, 3), dtype=float)
    if cmd.size:
        cmd_sum[:, 0] = np.sum(cmd[:, 0::3], axis=1)
        cmd_sum[:, 1] = np.sum(cmd[:, 1::3], axis=1)
        cmd_sum[:, 2] = np.sum(cmd[:, 2::3], axis=1)

    act_sum = np.zeros((t.size, 3), dtype=float)
    if actual.size:
        act_sum[:, 0] = np.sum(actual[:, :, 0], axis=1)
        act_sum[:, 1] = np.sum(actual[:, :, 1], axis=1)
        act_sum[:, 2] = np.sum(actual[:, :, 2], axis=1)

    fig1, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)
    labels = ['Fx', 'Fy', 'Fz']
    for j, ax in enumerate(axes):
        ax.plot(t, cmd_sum[:, j], linewidth=1.8, label=f'commanded sum {labels[j]}')
        ax.plot(t, act_sum[:, j], '--', linewidth=1.6, label=f'actual sum {labels[j]}')
        ax.set_ylabel(f'{labels[j]} [N]')
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(frameon=True, fontsize=8)
    axes[-1].set_xlabel('time [s]')
    fig1.suptitle(f'Commanded vs actual total contact force ({convention})', fontsize=11)
    fig1.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    saved.append(_save(fig1, outdir / 'fig_total_commanded_vs_actual_force.png'))

    fig2, axes2 = plt.subplots(4, 1, figsize=(6.4, 7.0), sharex=True)
    for leg_idx, ax in enumerate(axes2):
        if cmd.size:
            ax.plot(t, cmd[:, 3 * leg_idx + 0], linewidth=1.5, label=f'{LEG_NAMES[leg_idx]} commanded $F_x$')
        if actual.size:
            ax.plot(t, actual[:, leg_idx, 0], '--', linewidth=1.5, label=f'{LEG_NAMES[leg_idx]} actual $F_x$')
        ax.set_ylabel('Fx [N]')
        ax.set_title(LEG_NAMES[leg_idx], loc='left', fontsize=10, pad=2)
        ax.grid(alpha=0.30, linewidth=0.6)
        ax.legend(frameon=True, fontsize=8, loc='upper right')
    axes2[-1].set_xlabel('time [s]')
    fig2.tight_layout(h_pad=0.8)
    saved.append(_save(fig2, outdir / 'fig_per_leg_commanded_vs_actual_fx.png'))

    return saved
