
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from baseline.config import LEG_NAMES


def build_phase6_summary(log: dict, bindings) -> dict:
    t = np.asarray(log['t'], dtype=float)
    x = np.asarray(log['x'], dtype=float)
    sched = np.asarray(log['contact'], dtype=bool)
    actual = np.asarray(log['contact_actual'], dtype=bool)
    enabled = np.asarray(log.get('contact_force_enabled', []), dtype=bool)
    applied = np.asarray(log.get('u_applied', []), dtype=float)  # [T, 12]

    after_mask = t >= 1.0
    if x.size:
        vx_mean_after_1s = float(np.mean(x[after_mask, 3])) if np.any(after_mask) else float(np.mean(x[:, 3]))
    else:
        vx_mean_after_1s = 0.0

    if applied.size:
        sum_fx = np.sum(applied[:, 0::3], axis=1)
        sum_fy = np.sum(applied[:, 1::3], axis=1)
        sum_fz = np.sum(applied[:, 2::3], axis=1)
        mean_sum_fx_after_1s = float(np.mean(sum_fx[after_mask])) if np.any(after_mask) else float(np.mean(sum_fx))
        mean_sum_fy_after_1s = float(np.mean(sum_fy[after_mask])) if np.any(after_mask) else float(np.mean(sum_fy))
        mean_sum_fz_after_1s = float(np.mean(sum_fz[after_mask])) if np.any(after_mask) else float(np.mean(sum_fz))
    else:
        mean_sum_fx_after_1s = 0.0
        mean_sum_fy_after_1s = 0.0
        mean_sum_fz_after_1s = 0.0

    per_leg = []
    for leg_idx, leg in enumerate(bindings.leg_bindings):
        sched_ratio = float(np.mean(sched[:, leg_idx])) if sched.size else 0.0
        actual_ratio = float(np.mean(actual[:, leg_idx])) if actual.size else 0.0
        mismatch = float(np.mean(sched[:, leg_idx] != actual[:, leg_idx])) if sched.size else 0.0
        stance_success = float(np.mean(actual[sched[:, leg_idx], leg_idx])) if np.any(sched[:, leg_idx]) else 0.0
        enabled_ratio = float(np.mean(enabled[:, leg_idx])) if enabled.size else 0.0
        td = list(getattr(leg, 'touchdown_delays', []))

        if applied.size and enabled.size and np.any(enabled[:, leg_idx]):
            fx_enabled = float(np.mean(applied[enabled[:, leg_idx], 3*leg_idx + 0]))
            fy_enabled = float(np.mean(applied[enabled[:, leg_idx], 3*leg_idx + 1]))
            fz_enabled = float(np.mean(applied[enabled[:, leg_idx], 3*leg_idx + 2]))
        else:
            fx_enabled = 0.0
            fy_enabled = 0.0
            fz_enabled = 0.0

        per_leg.append({
            'leg': LEG_NAMES[leg_idx],
            'scheduled_stance_ratio': sched_ratio,
            'actual_contact_ratio': actual_ratio,
            'mismatch_ratio': mismatch,
            'stance_success_ratio': stance_success,
            'force_enabled_ratio': enabled_ratio,
            'touchdown_delay_mean_s': None if len(td) == 0 else float(np.mean(td)),
            'touchdown_delay_max_s': None if len(td) == 0 else float(np.max(td)),
            'touchdown_count': int(len(td)),
            'mean_fx_when_enabled': fx_enabled,
            'mean_fy_when_enabled': fy_enabled,
            'mean_fz_when_enabled': fz_enabled,
        })

    return {
        'mean_mismatch_ratio': float(np.mean([item['mismatch_ratio'] for item in per_leg])) if per_leg else 0.0,
        'mean_vx_after_1s': vx_mean_after_1s,
        'mean_sum_fx_after_1s': mean_sum_fx_after_1s,
        'mean_sum_fy_after_1s': mean_sum_fy_after_1s,
        'mean_sum_fz_after_1s': mean_sum_fz_after_1s,
        'per_leg': per_leg,
    }


def write_phase6_summary(output_dir: str | Path, summary: dict) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'phase6_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return str(path)
