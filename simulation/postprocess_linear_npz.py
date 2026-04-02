from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    from simulation.artifacts import save_plots, save_summary, save_topdown_mp4, summarize_log
else:
    from .artifacts import save_plots, save_summary, save_topdown_mp4, summarize_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild plots/mp4 from a saved run_log.npz artifact.")
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    final_log = {}
    for key in data.files:
        if key == "meta":
            final_log["meta"] = json.loads(str(data[key]))
        else:
            final_log[key] = data[key]

    summary = summarize_log(final_log)
    save_summary(summary, out_dir / "summary.json")
    if not args.no_plots:
        save_plots(final_log, out_dir)
    if not args.no_mp4:
        save_topdown_mp4(final_log, out_dir / "topdown_motion.mp4")
    print(f"Saved postprocessed artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
