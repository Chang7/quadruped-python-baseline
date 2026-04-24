from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SETUP_BASH = ROOT / "ros2" / "msgs_ws" / "install" / "setup.bash"
COLLECTOR = ROOT / "ros2" / "collect_baseline_metrics.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "ros2_baseline"


def _rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _source_and_run(command: list[str]) -> str:
    return " && ".join(
        [
            f"cd {shlex.quote(str(ROOT))}",
            f"source {shlex.quote(_rel_posix(SETUP_BASH))}",
            _shell_join(command),
        ]
    )


def _spawn_bash(command: str, log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "wb")
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=(os.name == "posix"),
    )


def _stop_process(proc: subprocess.Popen[bytes] | None, sig: int, fallback_sig: int) -> None:
    if proc is None or proc.poll() is not None:
        return

    if os.name == "posix":
        os.killpg(proc.pid, sig)
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, fallback_sig)
    else:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a timed ROS2 baseline capture and write a JSON summary.")
    parser.add_argument("--controller", default="sampling")
    parser.add_argument("--gait", default="trot")
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--lateral-speed", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--scheduler-freq", type=float, default=500.0)
    parser.add_argument("--render-freq", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--renice", action="store_true")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).resolve() / f"{timestamp}_{args.controller}_{args.gait}"
    metrics_path = output_dir / "summary.json"
    simulator_log = output_dir / "simulator.log"
    controller_log = output_dir / "controller.log"
    collector_log = output_dir / "collector.log"

    simulator_cmd = [
        "python",
        "ros2/run_simulator.py",
        "--scheduler-freq",
        str(args.scheduler_freq),
        "--render-freq",
        str(args.render_freq),
    ]
    if args.no_render:
        simulator_cmd.append("--no-render")

    controller_cmd = [
        "python",
        "ros2/run_controller.py",
        "--controller",
        args.controller,
        "--gait",
        args.gait,
        "--auto-start",
        "--no-console",
        "--speed",
        str(args.speed),
        "--lateral-speed",
        str(args.lateral_speed),
        "--yaw-rate",
        str(args.yaw_rate),
    ]
    if args.renice:
        controller_cmd.append("--renice")

    collector_cmd = [
        "python",
        "ros2/collect_baseline_metrics.py",
        "--duration",
        str(args.duration),
        "--warmup",
        str(args.warmup),
        "--output",
        str(metrics_path),
    ]

    simulator_shell = _source_and_run(simulator_cmd)
    controller_shell = _source_and_run(controller_cmd)
    collector_shell = _source_and_run(collector_cmd)

    simulator_proc = None
    controller_proc = None
    collector_proc = None

    try:
        simulator_proc = _spawn_bash(simulator_shell, simulator_log)
        time.sleep(max(args.startup_delay, 0.0))
        controller_proc = _spawn_bash(controller_shell, controller_log)
        time.sleep(1.0)
        collector_proc = _spawn_bash(collector_shell, collector_log)

        exit_code = 0
        while True:
            sim_rc = simulator_proc.poll()
            ctrl_rc = controller_proc.poll()
            collect_rc = collector_proc.poll()

            if collect_rc is not None:
                exit_code = int(collect_rc)
                break

            if sim_rc is not None or ctrl_rc is not None:
                exit_code = sim_rc if sim_rc not in (None, 0) else ctrl_rc or 1
                break

            time.sleep(0.2)
    finally:
        _stop_process(collector_proc, signal.SIGINT, signal.SIGTERM)
        _stop_process(controller_proc, signal.SIGINT, signal.SIGTERM)
        _stop_process(simulator_proc, signal.SIGINT, signal.SIGTERM)

    payload: dict[str, object] = {
        "controller": args.controller,
        "gait": args.gait,
        "speed": args.speed,
        "lateral_speed": args.lateral_speed,
        "yaw_rate": args.yaw_rate,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "scheduler_freq_hz": args.scheduler_freq,
        "render_freq_hz": args.render_freq,
        "output_dir": str(output_dir),
        "simulator_log": str(simulator_log),
        "controller_log": str(controller_log),
        "collector_log": str(collector_log),
    }

    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["metrics"] = summary
    else:
        payload["metrics"] = None

    run_info_path = output_dir / "run_info.json"
    run_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(output_dir)
    print(run_info_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
