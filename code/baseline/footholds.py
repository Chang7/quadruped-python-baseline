import numpy as np
from baseline.config import MPCConfig, IDX_TH


def rz(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def nominal_foot_positions_world(x: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Returns 4x3 relative foot locations r_i in world frame, relative to COM.
    V1 simplification:
    - no Raibert update
    - just rotate nominal body-frame offsets by current yaw
    """
    yaw = float(x[IDX_TH][2])
    Rz = rz(yaw)
    return (Rz @ cfg.nominal_footholds_body.T).T
