#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt


def _walk(name, obj, out):
    if isinstance(obj, h5py.Dataset):
        out.append((name, obj.shape, str(obj.dtype)))


def summarize_h5(h5_path: Path):
    summary = []
    with h5py.File(h5_path, 'r') as f:
        f.visititems(lambda name, obj: _walk(name, obj, summary))
    return summary


def first_existing_key(f: h5py.File, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in f:
            return k
    return None


def try_load_vector(f: h5py.File, candidates: list[str]):
    k = first_existing_key(f, candidates)
    if k is None:
        return None, None
    arr = np.asarray(f[k])
    return k, arr


def plot_basic(h5_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        time_key, t = try_load_vector(f, ['time', 't', 'timestamp', 'timestamps'])
        state_key, x = try_load_vector(f, ['state', 'states', 'observation', 'observations'])
        action_key, u = try_load_vector(f, ['action', 'actions', 'tau', 'torque'])
        qpos_key, qpos = try_load_vector(f, ['qpos'])
        qvel_key, qvel = try_load_vector(f, ['qvel'])

        report = {
            'time_key': time_key,
            'state_key': state_key,
            'action_key': action_key,
            'qpos_key': qpos_key,
            'qvel_key': qvel_key,
        }
        (out_dir / 'key_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

        if t is None:
            # fallback: infer a sample index axis
            if x is not None:
                t = np.arange(len(x))
            elif qpos is not None:
                t = np.arange(len(qpos))
            elif u is not None:
                t = np.arange(len(u))
            else:
                t = np.arange(1)

        if x is not None and x.ndim == 2 and x.shape[0] == len(t):
            plt.figure(figsize=(10, 5))
            for i in range(min(x.shape[1], 6)):
                plt.plot(t, x[:, i], label=f'state[{i}]')
            plt.xlabel('time / index')
            plt.ylabel('value')
            plt.title('First state channels')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'states_preview.png', dpi=150)
            plt.close()

        if u is not None and u.ndim == 2:
            tu = t[:len(u)]
            plt.figure(figsize=(10, 5))
            for i in range(min(u.shape[1], 8)):
                plt.plot(tu, u[:, i], label=f'action[{i}]')
            plt.xlabel('time / index')
            plt.ylabel('value')
            plt.title('Action preview')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'actions_preview.png', dpi=150)
            plt.close()

        if qpos is not None and qpos.ndim == 2:
            tq = t[:len(qpos)]
            plt.figure(figsize=(10, 5))
            for i in range(min(qpos.shape[1], 8)):
                plt.plot(tq, qpos[:, i], label=f'qpos[{i}]')
            plt.xlabel('time / index')
            plt.ylabel('value')
            plt.title('qpos preview')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'qpos_preview.png', dpi=150)
            plt.close()

        if qvel is not None and qvel.ndim == 2:
            tv = t[:len(qvel)]
            plt.figure(figsize=(10, 5))
            for i in range(min(qvel.shape[1], 8)):
                plt.plot(tv, qvel[:, i], label=f'qvel[{i}]')
            plt.xlabel('time / index')
            plt.ylabel('value')
            plt.title('qvel preview')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'qvel_preview.png', dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inspect and preview a Quadruped-PyMPC .h5 recording.')
    parser.add_argument('h5_path', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('outputs_h5_inspect'))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_h5(args.h5_path)
    with open(args.out_dir / 'dataset_tree.txt', 'w', encoding='utf-8') as fp:
        for name, shape, dtype in summary:
            fp.write(f'{name}\tshape={shape}\tdtype={dtype}\n')

    plot_basic(args.h5_path, args.out_dir)
    print(f'Saved inspection outputs to: {args.out_dir}')
    print(f'Dataset tree: {args.out_dir / "dataset_tree.txt"}')
    print(f'Key report:    {args.out_dir / "key_report.json"}')


if __name__ == '__main__':
    main()
