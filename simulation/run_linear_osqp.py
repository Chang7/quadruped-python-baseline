"""Compatibility wrapper for :mod:`mujoco_sim.run_linear_osqp`."""

from mujoco_sim.run_linear_osqp import *  # noqa: F401,F403
from mujoco_sim.run_linear_osqp import main


if __name__ == "__main__":
    main()

