import numpy as np
from config import MPCConfig, IDX_P, IDX_V, IDX_TH, IDX_W, IDX_G


def _speed_profile(t: float, cfg: MPCConfig) -> float:
    return min(cfg.desired_speed, cfg.desired_accel * max(t, 0.0))


def _yaw_profile(t: float, cfg: MPCConfig) -> tuple[float, float]:
    if cfg.scenario_name == "turn_pi_over_4":
        target = float(cfg.desired_yaw)
        rate = abs(float(cfg.desired_yaw_rate))
        sign = 1.0 if target >= 0.0 else -1.0
        yaw_mag = min(abs(target), rate * max(t, 0.0))
        yaw_ref = sign * yaw_mag
        yaw_rate_ref = sign * rate if yaw_mag < abs(target) - 1e-12 else 0.0
        return yaw_ref, yaw_rate_ref

    # Straight-line baseline
    return float(cfg.desired_yaw), float(cfg.desired_yaw_rate)


def rollout_reference(t0: float, x_now: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Builds a horizon reference x_ref with shape (N+1, 13).
    Straight-line scenario:
      - forward motion with acceleration ramp
      - zero yaw command
    Turning scenario:
      - forward motion with the velocity direction aligned to the yaw reference
      - yaw commanded toward pi/4 (paper-inspired baseline)
    """
    N = cfg.horizon
    x_ref = np.zeros((N + 1, cfg.nx), dtype=float)

    x_ref[0, :] = x_now.copy()
    x_ref[:, IDX_G] = cfg.g
    x_ref[:, 2] = cfg.nominal_height

    for k in range(N + 1):
        tk = t0 + k * cfg.dt_mpc
        speed = _speed_profile(tk, cfg)
        yaw_ref, yaw_rate_ref = _yaw_profile(tk, cfg)

        vx = speed * np.cos(yaw_ref)
        vy = speed * np.sin(yaw_ref)

        if k == 0:
            x_ref[k, IDX_V] = np.array([vx, vy, 0.0])
            x_ref[k, IDX_TH] = np.array([0.0, 0.0, yaw_ref])
            x_ref[k, IDX_W] = np.array([0.0, 0.0, yaw_rate_ref])
        else:
            x_ref[k, IDX_P] = x_ref[k - 1, IDX_P] + cfg.dt_mpc * x_ref[k - 1, IDX_V]
            x_ref[k, 2] = cfg.nominal_height
            x_ref[k, IDX_V] = np.array([vx, vy, 0.0])
            x_ref[k, IDX_TH] = np.array([0.0, 0.0, yaw_ref])
            x_ref[k, IDX_W] = np.array([0.0, 0.0, yaw_rate_ref])

    return x_ref
