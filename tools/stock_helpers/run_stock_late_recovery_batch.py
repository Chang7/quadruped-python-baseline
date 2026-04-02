import json, subprocess
from pathlib import Path
repo = Path(r'/mnt/c/Quadruped-PyMPC_linear_osqp_adapter/1.Quadruped-PyMPC-main/Quadruped-PyMPC-main')
base = repo / 'outputs'
cases = [
    ('stock_late_recovery_probe_b', ['--late-full-contact-recovery-roll-threshold','0.48','--late-full-contact-recovery-height-ratio','0.46','--late-full-contact-recovery-hold-s','0.05','--late-full-contact-recovery-forward-scale','0.50','--late-full-contact-recovery-lookahead-steps','6']),
    ('stock_late_recovery_probe_c', ['--late-full-contact-recovery-roll-threshold','0.46','--late-full-contact-recovery-height-ratio','0.46','--late-full-contact-recovery-hold-s','0.05','--late-full-contact-recovery-forward-scale','0.50','--late-full-contact-recovery-lookahead-steps','8']),
    ('stock_late_recovery_probe_d', ['--late-full-contact-recovery-roll-threshold','0.45','--late-full-contact-recovery-height-ratio','0.50','--late-full-contact-recovery-hold-s','0.04','--late-full-contact-recovery-forward-scale','0.55','--late-full-contact-recovery-lookahead-steps','8']),
    ('stock_late_recovery_probe_e', ['--late-full-contact-recovery-roll-threshold','0.44','--late-full-contact-recovery-height-ratio','0.50','--late-full-contact-recovery-hold-s','0.06','--late-full-contact-recovery-forward-scale','0.45','--late-full-contact-recovery-lookahead-steps','6']),
]
common = ['python','-m','simulation.run_linear_osqp','--controller','linear_osqp','--gait','crawl','--preset','conservative','--seconds','3','--no-plots','--no-mp4']
for name, extra in cases:
    cmd = ['wsl','bash','-lc', "cd /mnt/c/Quadruped-PyMPC_linear_osqp_adapter/1.Quadruped-PyMPC-main/Quadruped-PyMPC-main && source ../../venv/bin/activate && " + ' '.join(common + ['--artifact-dir', f'outputs/{name}'] + extra)]
    print('RUN', name)
    proc = subprocess.run(cmd)
    print('RET', proc.returncode)
    if proc.returncode != 0:
        continue
    summary = json.loads((base / name / 'episode_000' / 'summary.json').read_text())
    print(name, json.dumps({
        'duration_s': summary.get('duration_s'),
        'terminated_any': summary.get('terminated_any'),
        'invalid': summary.get('meta',{}).get('invalid_contact_keys'),
        'mean_vx': summary.get('mean_vx'),
        'actual_total': summary.get('actual_swing_realization_total'),
        'front_actual': summary.get('front_actual_swing_realization_mean'),
        'rr_actual_ratio': summary.get('actual_swing_ratio',{}).get('RR'),
        'late_recovery_mean': summary.get('late_full_contact_recovery_mean'),
        'gate_min': summary.get('gate_forward_scale_min'),
    }, ensure_ascii=False))
