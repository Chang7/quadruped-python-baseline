# ROS2 Baseline

This folder contains the upstream-style ROS2 simulator/controller bridge plus a
small baseline workflow for this adapter repo.

For the professor-facing baseline, the simplest target is:

- build `dls2_interface` messages,
- run `ros2/run_simulator.py`,
- run `ros2/run_controller.py --controller sampling --gait trot --auto-start`.

Using `sampling` keeps the stack on the stock controller path and avoids the
extra `acados` dependency needed by the gradient controllers.

## Preflight

Check which launch path is currently usable before trying to run nodes:

```bash
python ros2/preflight.py
```

For a local conda shell specifically:

```bash
python ros2/preflight.py --mode local
```

If you run this from a plain Windows host shell, expect the script to point you
to Docker or a Linux/WSL ROS2 environment unless the required modules are
already installed there.

## One-Command Launcher

Inside a ready Docker shell or local ROS2 conda shell, you can launch the
baseline pair together with:

```bash
python tools/launchers/run_ros2_baseline.py --speed 0.12 --no-render
```

For the turn baseline:

```bash
python tools/launchers/run_ros2_baseline.py --speed 0.12 --yaw-rate 0.3 --no-render
```

The launcher runs the same `run_simulator.py` and `run_controller.py` pair,
uses `--auto-start --no-console` for the controller, and builds ROS2 messages
first if `ros2/msgs_ws/install/setup.bash` is still missing.

## Option A: Docker

Build and start the container:

```bash
docker compose up --build -d pympc
```

Open two shells in the running container:

```bash
docker compose exec pympc bash
docker compose exec pympc bash
```

The image activates `quadruped_pympc_ros2_env` automatically and sources the
ROS2 message workspace if it has already been built.

Rebuild messages when needed:

```bash
cd /workspace
bash ros2/build_msgs.sh
source /workspace/ros2/msgs_ws/install/setup.bash
```

Run the simulator in shell A:

```bash
cd /workspace
python ros2/run_simulator.py --no-render
```

Run the stock sampling controller in shell B:

```bash
cd /workspace
python ros2/run_controller.py \
  --controller sampling \
  --gait trot \
  --auto-start \
  --speed 0.12 \
  --no-console
```

For a turn baseline, add `--yaw-rate 0.3`.

## Option B: Local Mamba/Conda

Create and activate the ROS2 environment:

```bash
mamba env create -f installation/mamba/integrated_gpu/mamba_environment_ros2_humble.yml
conda activate quadruped_pympc_ros2_env
```

Install the repo and build the ROS2 messages:

```bash
pip install -e .
bash ros2/build_msgs.sh
source ros2/msgs_ws/install/setup.bash
```

Then start the simulator and controller in two terminals with the same commands
shown in the Docker section.

## Useful Runtime Flags

These help commands require a ready ROS2 Python environment and a sourced
`ros2/msgs_ws/install/setup.bash`, because the modules are imported at startup.

Controller:

```bash
python ros2/run_controller.py --help
```

- `--controller sampling|linear_osqp|...`
- `--gait trot|pace|bound|crawl|full_stance`
- `--auto-start`
- `--speed`, `--lateral-speed`, `--yaw-rate`
- `--no-console`
- `--renice` for best-effort Linux priority tuning

Simulator:

```bash
python ros2/run_simulator.py --help
```

- `--scheduler-freq 500`
- `--render-freq 30`
- `--no-render`

## Notes

- The controller still subscribes to `joy`, but joystick input is optional for
  the scripted baseline above.
- `python ros2/preflight.py` is the quickest way to see whether the current
  shell can launch nodes, only build messages, or needs Docker first.
- `python tools/launchers/run_ros2_baseline.py --dry-run --skip-preflight --skip-build`
  prints the exact simulator/controller commands without starting them.
- If you rebuild `msgs_ws`, re-run `source ros2/msgs_ws/install/setup.bash` in
  every shell before launching nodes.
- Robot selection still follows `QUADRUPED_PYMPC_ROBOT`, for example:

```bash
export QUADRUPED_PYMPC_ROBOT=go1
```
