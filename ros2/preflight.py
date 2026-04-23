from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROS2_DIR = ROOT / "ros2"
BUILD_SCRIPT = ROS2_DIR / "build_msgs.sh"
WORKSPACE_DIR = ROS2_DIR / "msgs_ws"
MSG_SETUP = WORKSPACE_DIR / "install" / "setup.bash"
DLS2_PACKAGE = WORKSPACE_DIR / "src" / "dls2_interface" / "package.xml"
COMPOSE_FILE = ROOT / "compose.yaml"
ENV_FILE = ROOT / "installation" / "mamba" / "integrated_gpu" / "mamba_environment_ros2_humble.yml"

REQUIRED_MODULES = (
    "rclpy",
    "mujoco",
    "gym_quadruped",
    "sensor_msgs",
)
MESSAGE_MODULES = ("dls2_interface",)


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _status(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def _print_kv(label: str, value: str) -> None:
    print(f"{label:<18} {value}")


def _rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _collect_local_status() -> dict[str, object]:
    bash_available = _has_command("bash")
    colcon_available = _has_command("colcon")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    required_modules = {name: _has_module(name) for name in REQUIRED_MODULES}
    message_modules = {name: _has_module(name) for name in MESSAGE_MODULES}

    can_build_messages = all(
        (
            bash_available,
            colcon_available,
            bool(conda_prefix),
            BUILD_SCRIPT.exists(),
            DLS2_PACKAGE.exists(),
        )
    )
    can_launch_nodes = all(required_modules.values()) and all(message_modules.values()) and MSG_SETUP.exists()

    return {
        "bash_available": bash_available,
        "colcon_available": colcon_available,
        "conda_prefix": conda_prefix,
        "required_modules": required_modules,
        "message_modules": message_modules,
        "can_build_messages": can_build_messages,
        "can_launch_nodes": can_launch_nodes,
    }


def _collect_docker_status() -> dict[str, object]:
    docker_available = _has_command("docker")
    can_start = docker_available and COMPOSE_FILE.exists()
    return {
        "docker_available": docker_available,
        "can_start": can_start,
    }


def _recommended_mode(requested_mode: str, docker_status: dict[str, object], local_status: dict[str, object]) -> str:
    if requested_mode != "auto":
        return requested_mode
    if docker_status["can_start"]:
        return "docker"
    return "local"


def _selected_mode_ready(mode: str, docker_status: dict[str, object], local_status: dict[str, object]) -> bool:
    if mode == "docker":
        return bool(docker_status["can_start"])
    return bool(local_status["can_build_messages"] or local_status["can_launch_nodes"])


def _print_recommendations(selected_mode: str, local_status: dict[str, object], docker_status: dict[str, object]) -> None:
    print()
    print("Recommended next commands")
    if selected_mode == "docker" and docker_status["can_start"]:
        print("  docker compose up --build -d pympc")
        print("  docker compose exec pympc bash")
        print("  cd /workspace")
        print("  python ros2/run_simulator.py --no-render")
        print("  python ros2/run_controller.py --controller sampling --gait trot --auto-start --speed 0.12 --no-console")
        return

    print(f"  mamba env create -f {_rel_posix(ENV_FILE)}")
    print("  conda activate quadruped_pympc_ros2_env")
    print("  pip install -e .")
    print(f"  bash {_rel_posix(BUILD_SCRIPT)}")
    print(f"  source {_rel_posix(MSG_SETUP)}")
    print("  python ros2/run_simulator.py --no-render")
    print("  python ros2/run_controller.py --controller sampling --gait trot --auto-start --speed 0.12 --no-console")
    if not local_status["can_launch_nodes"]:
        print("  python ros2/preflight.py --mode local")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the ROS2 baseline prerequisites are available.")
    parser.add_argument(
        "--mode",
        choices=("auto", "docker", "local"),
        default="auto",
        help="Which launch path to validate.",
    )
    args = parser.parse_args()

    docker_status = _collect_docker_status()
    local_status = _collect_local_status()
    selected_mode = _recommended_mode(args.mode, docker_status, local_status)

    print("ROS2 baseline preflight")
    _print_kv("repo", str(ROOT))
    _print_kv("platform", platform.platform())
    _print_kv("python", sys.executable)
    _print_kv("requested mode", args.mode)
    _print_kv("suggested mode", selected_mode)
    print()

    print("Workspace files")
    _print_kv("build script", _status(BUILD_SCRIPT.exists()))
    _print_kv("dls2_interface", _status(DLS2_PACKAGE.exists()))
    _print_kv("msg setup.bash", _status(MSG_SETUP.exists()))
    _print_kv("compose.yaml", _status(COMPOSE_FILE.exists()))
    _print_kv("mamba env file", _status(ENV_FILE.exists()))
    print()

    print("Commands")
    _print_kv("bash", _status(local_status["bash_available"]))
    _print_kv("colcon", _status(local_status["colcon_available"]))
    _print_kv("docker", _status(docker_status["docker_available"]))
    _print_kv("CONDA_PREFIX", local_status["conda_prefix"] or "MISSING")
    print()

    print("Python modules")
    for module_name, available in local_status["required_modules"].items():
        _print_kv(module_name, _status(available))
    for module_name, available in local_status["message_modules"].items():
        _print_kv(module_name, _status(available))
    print()

    print("Readiness")
    _print_kv("docker path", _status(bool(docker_status["can_start"])))
    _print_kv("local build", _status(bool(local_status["can_build_messages"])))
    _print_kv("local launch", _status(bool(local_status["can_launch_nodes"])))

    if platform.system() == "Windows" and not docker_status["can_start"] and not local_status["can_launch_nodes"]:
        print()
        print("Windows note")
        print("  This repo's ROS2 baseline is normally launched from Docker or a Linux/WSL conda shell.")

    _print_recommendations(selected_mode, local_status, docker_status)
    return 0 if _selected_mode_ready(selected_mode, docker_status, local_status) else 1


if __name__ == "__main__":
    raise SystemExit(main())
