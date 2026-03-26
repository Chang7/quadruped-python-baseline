import numpy as np
from config import MPCConfig, IDX_P, IDX_V, IDX_TH, IDX_W, IDX_G


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,  x,   0.0],
    ], dtype=float)


def rz(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def build_continuous_matrices(
    x_nom: np.ndarray,
    foot_rel_world: np.ndarray,
    cfg: MPCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time 13-state, 12-input baseline.
    Input is always 12D; swing-leg zero-force is handled in constraints.
    """
    Ac = np.zeros((cfg.nx, cfg.nx), dtype=float)
    Bc = np.zeros((cfg.nx, cfg.nu), dtype=float)

    yaw = float(x_nom[IDX_TH][2])

    # p_dot = v
    Ac[IDX_P, IDX_V] = np.eye(3)

    # theta_dot ≈ Rz(yaw) w
    Ac[IDX_TH, IDX_W] = rz(yaw)

    # v_dot includes gravity state g as scalar in z direction
    # Sign convention: z-up, so gravity contributes negative z acceleration
    Ac[IDX_V, IDX_G] = np.array([0.0, 0.0, -1.0])

    I_inv = np.linalg.inv(cfg.inertia_matrix())

    for leg in range(4):
        cols = slice(3 * leg, 3 * leg + 3)
        r_i = foot_rel_world[leg]

        # v_dot += (1/m) * f_i
        Bc[IDX_V, cols] = np.eye(3) / cfg.mass

        # w_dot += I^{-1} (r_i x f_i)
        Bc[IDX_W, cols] = I_inv @ skew(r_i)

    return Ac, Bc


def discretize_forward_euler(
    Ac: np.ndarray,
    Bc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    Ad = np.eye(Ac.shape[0]) + dt * Ac
    Bd = dt * Bc
    return Ad, Bd


def build_prediction_model(
    x_ref: np.ndarray,
    foot_rel_world: np.ndarray,
    cfg: MPCConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Builds time-varying Ad[k], Bd[k].
    V1 simplification: foot positions remain fixed over the horizon.
    """
    Ad_list: list[np.ndarray] = []
    Bd_list: list[np.ndarray] = []

    for k in range(cfg.horizon):
        Ac, Bc = build_continuous_matrices(x_ref[k], foot_rel_world, cfg)
        Ad, Bd = discretize_forward_euler(Ac, Bc, cfg.dt_mpc)
        Ad_list.append(Ad)
        Bd_list.append(Bd)

    return Ad_list, Bd_list
