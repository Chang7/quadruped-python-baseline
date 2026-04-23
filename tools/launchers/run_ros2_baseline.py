from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRELIGHT = ROOT / "ros2" / "preflight.py"
BUILD_SCRIPT = ROOT / "ros2" / "build_msgs.sh"
SETUP_BASH = ROOT / "ros2" / "msgs_ws" / "install" / "setup.bash"


def _rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the ROS2 simulator/controller baseline from a ready bash shell.",
    )
    parser.add_argument("--controller", default="sampling")
    parser.add_argument("--gait", default="trot")
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--lateral-speed", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--scheduler-freq", type=float, default=500.0)
    parser.add_argument("--render-freq", type=float, default=30.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--renice", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after starting the simulator before launching the controller.",
    )
    return parser


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _source_and_run(command: list[str]) -> str:
    return " && ".join(
        [
            f"cd {shlex.quote(str(ROOT))}",
            f"source {shlex.quote(_rel_posix(SETUP_BASH))}",
            _shell_join(command),
        ]
    )


def _spawn_bash(command: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=ROOT,
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


def main() -> int:
    args = _build_arg_parser().parse_args()

    if shutil.which("bash") is None:
        raise SystemExit("bash is required to launch the ROS2 baseline.")

    if not args.skip_preflight:
        _run_checked([sys.executable, str(PRELIGHT), "--mode", "local"])

    if not args.skip_build and not SETUP_BASH.exists():
        _run_checked(["bash", _rel_posix(BUILD_SCRIPT)])

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

    simulator_shell = _source_and_run(simulator_cmd)
    controller_shell = _source_and_run(controller_cmd)

    print("[ros2-baseline] simulator:")
    print(f"  {simulator_shell}")
    print("[ros2-baseline] controller:")
    print(f"  {controller_shell}")

    if args.dry_run:
        return 0

    if not SETUP_BASH.exists():
        raise SystemExit(
            f"Missing {_rel_posix(SETUP_BASH)}. Build messages first or rerun without --skip-build."
        )

    simulator_proc = None
    controller_proc = None
    exit_code = 0

    try:
        simulator_proc = _spawn_bash(simulator_shell)
        time.sleep(max(args.startup_delay, 0.0))
        controller_proc = _spawn_bash(controller_shell)

        while True:
            simulator_rc = simulator_proc.poll()
            controller_rc = controller_proc.poll()
            if simulator_rc is not None or controller_rc is not None:
                exit_code = simulator_rc if simulator_rc not in (None, 0) else controller_rc or 0
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        _stop_process(controller_proc, signal.SIGINT, signal.SIGTERM)
        _stop_process(simulator_proc, signal.SIGINT, signal.SIGTERM)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
