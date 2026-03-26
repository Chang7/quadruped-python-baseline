import numpy as np
from config import MPCConfig, IDX_P, IDX_V, IDX_TH, IDX_W, IDX_G


def rz(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def step_srb(
    x: np.ndarray,
    u: np.ndarray,
    foot_rel_world: np.ndarray,
    cfg: MPCConfig,
    dt: float,
) -> np.ndarray:
    """
    Very simple SRB plant step with forward Euler.
    """
    x_next = x.copy()

    p = x[IDX_P]
    v = x[IDX_V]
    theta = x[IDX_TH]
    w = x[IDX_W]
    g_scalar = float(x[IDX_G])

    forces = u.reshape(4, 3)

    total_force = np.sum(forces, axis=0)
    total_torque = np.zeros(3, dtype=float)
    for r_i, f_i in zip(foot_rel_world, forces):
        total_torque += np.cross(r_i, f_i)

    I_inv = np.linalg.inv(cfg.inertia_matrix())

    p_dot = v
    v_dot = total_force / cfg.mass + np.array([0.0, 0.0, -g_scalar])
    theta_dot = rz(theta[2]) @ w
    w_dot = I_inv @ total_torque

    x_next[IDX_P] = p + dt * p_dot
    x_next[IDX_V] = v + dt * v_dot
    x_next[IDX_TH] = theta + dt * theta_dot
    x_next[IDX_W] = w + dt * w_dot
    x_next[IDX_G] = g_scalar

    return x_next
