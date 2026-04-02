from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-run the stock Quadruped-PyMPC repo with minimal overrides.")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--recording-path", type=str, default=None)
    parser.add_argument("--gait", type=str, default="trot")
    parser.add_argument("--mode", type=str, default="forward")
    parser.add_argument("--mpc-type", type=str, default="sampling")
    args = parser.parse_args()

    stock_root = Path(__file__).resolve().parents[1] / "1.Quadruped-PyMPC-main" / "Quadruped-PyMPC-main"
    if not stock_root.exists():
        print(f"stock_root_missing {stock_root}")
        return 2

    sys.path.insert(0, str(stock_root))

    import quadruped_pympc  # type: ignore
    from quadruped_pympc import config as cfg  # type: ignore
    from simulation.simulation import run_simulation  # type: ignore

    print(f"stock_package={quadruped_pympc.__file__}")

    qpympc_cfg = cfg
    qpympc_cfg.mpc_params["type"] = args.mpc_type
    qpympc_cfg.simulation_params["mode"] = args.mode
    qpympc_cfg.simulation_params["gait"] = args.gait
    qpympc_cfg.simulation_params["mpc_frequency"] = 100

    print(
        "smoke_cfg",
        {
            "mpc_type": qpympc_cfg.mpc_params["type"],
            "gait": qpympc_cfg.simulation_params["gait"],
            "mode": qpympc_cfg.simulation_params["mode"],
            "dt": qpympc_cfg.simulation_params["dt"],
            "recording_path": args.recording_path,
        },
    )

    run_simulation(
        qpympc_cfg=qpympc_cfg,
        num_episodes=args.episodes,
        num_seconds_per_episode=args.seconds,
        base_vel_command_type="forward",
        render=args.render,
        recording_path=args.recording_path,
    )
    print("smoke_run_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
