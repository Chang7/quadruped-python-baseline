"""Robot- and gait-specific tuning profiles for the linear OSQP controller.

These profiles are importable from any entry point (MuJoCo runner, ROS2 node)
without dragging in simulation-specific dependencies.
"""

from quadruped_pympc.profiles.trot_profile import (
    robot_posture_offsets,
    trot_conservative_profile,
    dynamic_gait_profile_for,
)

__all__ = [
    "robot_posture_offsets",
    "trot_conservative_profile",
    "dynamic_gait_profile_for",
]
