# Tools Layout

The active tools tree is intentionally small.

## Active contents

- `launchers/`
  - benchmark and reproducibility entry points such as
    `run_trot_benchmark_suite.py`
  - ROS2 baseline launcher: `run_ros2_baseline.py`
  - timed ROS2 baseline capture: `measure_ros2_baseline.py`

Older temporary sweep scripts were moved out of the active tools tree and are
now kept under:

- `../archive/legacy_tools/tools_legacy_sweeps/`

## Rule

- Keep only reusable launchers under `tools/`.
- Put one-off experiments and temporary sweeps under `archive/`, not here.
