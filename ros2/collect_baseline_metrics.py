from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from dls2_interface.msg import BaseState, TimeDebug
from rclpy.node import Node


def _quat_xyzw_to_euler_xyz(quat_xyzw: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = [float(value) for value in quat_xyzw]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_max_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(abs(value) for value in values))


class BaselineMetricsCollector(Node):
    def __init__(self, warmup_s: float, duration_s: float) -> None:
        super().__init__("ros2_baseline_metrics_collector")
        self.warmup_s = max(float(warmup_s), 0.0)
        self.duration_s = max(float(duration_s), 0.0)
        self.start_mono = time.perf_counter()
        self.end_mono = self.start_mono + self.warmup_s + self.duration_s

        self.base_state_total = 0
        self.time_debug_total = 0

        self.base_z_samples: list[float] = []
        self.roll_samples: list[float] = []
        self.pitch_samples: list[float] = []
        self.yaw_samples: list[float] = []
        self.time_mpc_samples: list[float] = []
        self.time_wbc_samples: list[float] = []

        self.base_state_subscription = self.create_subscription(BaseState, "/base_state", self._on_base_state, 10)
        self.time_debug_subscription = self.create_subscription(TimeDebug, "/time_debug", self._on_time_debug, 10)

    def _collecting(self) -> bool:
        now = time.perf_counter()
        return (self.start_mono + self.warmup_s) <= now <= self.end_mono

    def _on_base_state(self, msg: BaseState) -> None:
        self.base_state_total += 1
        if not self._collecting():
            return

        roll, pitch, yaw = _quat_xyzw_to_euler_xyz(msg.pose.orientation)
        self.base_z_samples.append(float(msg.pose.position[2]))
        self.roll_samples.append(roll)
        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)

    def _on_time_debug(self, msg: TimeDebug) -> None:
        self.time_debug_total += 1
        if not self._collecting():
            return

        self.time_mpc_samples.append(float(msg.time_mpc))
        self.time_wbc_samples.append(float(msg.time_wbc))

    def summary(self) -> dict[str, object]:
        mean_abs_roll = _safe_mean([abs(value) for value in self.roll_samples])
        mean_abs_pitch = _safe_mean([abs(value) for value in self.pitch_samples])

        return {
            "warmup_s": self.warmup_s,
            "duration_s": self.duration_s,
            "base_state_total": self.base_state_total,
            "time_debug_total": self.time_debug_total,
            "base_state_samples": len(self.base_z_samples),
            "time_debug_samples": len(self.time_mpc_samples),
            "mean_base_z": _safe_mean(self.base_z_samples),
            "mean_abs_roll": mean_abs_roll,
            "mean_abs_pitch": mean_abs_pitch,
            "max_abs_roll": _safe_max_abs(self.roll_samples),
            "max_abs_pitch": _safe_max_abs(self.pitch_samples),
            "mean_roll": _safe_mean(self.roll_samples),
            "mean_pitch": _safe_mean(self.pitch_samples),
            "mean_yaw": _safe_mean(self.yaw_samples),
            "time_mpc_mean_s": _safe_mean(self.time_mpc_samples),
            "time_mpc_max_s": max(self.time_mpc_samples) if self.time_mpc_samples else None,
            "time_wbc_mean_s": _safe_mean(self.time_wbc_samples),
            "time_wbc_max_s": max(self.time_wbc_samples) if self.time_wbc_samples else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect ROS2 baseline timing and posture metrics.")
    parser.add_argument("--duration", type=float, required=True, help="Active collection duration in seconds.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Warmup window to ignore before collecting.")
    parser.add_argument("--output", type=str, required=True, help="JSON output path.")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    collector = BaselineMetricsCollector(warmup_s=args.warmup, duration_s=args.duration)

    try:
        while rclpy.ok() and time.perf_counter() <= collector.end_mono:
            rclpy.spin_once(collector, timeout_sec=0.1)
    finally:
        summary = collector.summary()
        collector.destroy_node()
        rclpy.shutdown()

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["base_state_samples"] <= 0 or summary["time_debug_samples"] <= 0:
        print(output_path)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
