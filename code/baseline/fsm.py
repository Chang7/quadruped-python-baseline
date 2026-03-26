import numpy as np
from baseline.config import MPCConfig


def _cycle_time(cfg: MPCConfig) -> float:
    return cfg.stance_time + cfg.swing_time


def _stance_fraction(cfg: MPCConfig) -> float:
    return cfg.stance_time / _cycle_time(cfg)


def trot_contact_row(t: float, cfg: MPCConfig) -> np.ndarray:
    """
    Returns a boolean array of shape (4,) for legs [FL, FR, RL, RR].
    Trot: diagonal legs share the same phase.
    """
    cycle = _cycle_time(cfg)
    stance_frac = _stance_fraction(cfg)

    # FL & RR together, FR & RL together
    phase_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)
    phase = ((t / cycle) + phase_offsets) % 1.0
    contact = phase < stance_frac
    return contact.astype(bool)


def rollout_contact_schedule(t0: float, cfg: MPCConfig) -> np.ndarray:
    """
    Returns shape (N, 4)
    """
    times = t0 + cfg.dt_mpc * np.arange(cfg.horizon)
    return np.vstack([trot_contact_row(t, cfg) for t in times])
