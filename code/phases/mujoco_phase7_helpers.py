from __future__ import annotations

import numpy as np
import mujoco


def leg_internal_force_to_qfrc(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    binding,
    point_world: np.ndarray,
    f_world: np.ndarray,
) -> np.ndarray:
    """Map a Cartesian foot force to joint generalized forces for one 3-DoF leg only.

    This is an *internal actuation-style* realization: only the leg dofs receive
    the torque tau = J_leg^T f. The floating-base dofs are not directly forced.
    """
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jac(m, d, jacp, jacr, point_world, binding.calf_body_id)

    qfrc = np.zeros(m.nv, dtype=float)
    dofs = np.asarray(binding.dof_adrs, dtype=int)
    if dofs.size == 0:
        return qfrc
    tau_leg = jacp[:, dofs].T @ np.asarray(f_world, dtype=float)
    qfrc[dofs] = tau_leg
    return qfrc
