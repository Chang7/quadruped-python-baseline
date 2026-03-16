from dataclasses import dataclass, field
import numpy as np

# State layout: [p(3), v(3), theta(3), w(3), g(1)]
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_W = slice(9, 12)
IDX_G = 12

LEG_NAMES = ("FL", "FR", "RL", "RR")


@dataclass
class MPCConfig:
    nx: int = 13
    nu: int = 12  # 4 legs * 3 force components

    # Simulation / prediction
    horizon: int = 15
    dt_mpc: float = 0.02
    dt_sim: float = 0.02
    sim_time: float = 3.0
    scenario_name: str = "straight_trot"

    # Robot parameters (paper / MATLAB baseline values)
    mass: float = 5.5
    Ixx: float = 0.026
    Iyy: float = 0.112
    Izz: float = 0.075
    g: float = 9.81
    mu: float = 1.0
    nominal_height: float = 0.2

    # Gait parameters: paper straight-line trot baseline
    stance_time: float = 0.1
    swing_time: float = 0.18

    # Reference command
    desired_speed: float = 0.5
    desired_accel: float = 0.5
    desired_yaw: float = 0.0
    desired_yaw_rate: float = 0.0

    # Force bounds
    fz_min: float = 0.0
    fz_max: float = 60.0

    # Cost weights
    Q_p: float = 1e6
    Q_v: float = 1e6
    Q_theta: float = 1e6
    Q_w: float = 1e6
    R_u: float = 1e1

    # Nominal foot locations relative to COM in body frame
    # MATLAB get_params.m uses pf34 with y offsets ±0.094 m.
    # Order: FL, FR, RL, RR
    nominal_footholds_body: np.ndarray = field(
        default_factory=lambda: np.array([
            [ 0.15,  0.094, -0.20],
            [ 0.15, -0.094, -0.20],
            [-0.15,  0.094, -0.20],
            [-0.15, -0.094, -0.20],
        ], dtype=float)
    )

    def inertia_matrix(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    def Q(self) -> np.ndarray:
        Q = np.zeros((self.nx, self.nx))
        Q[IDX_P, IDX_P] = np.eye(3) * self.Q_p
        Q[IDX_V, IDX_V] = np.eye(3) * self.Q_v
        Q[IDX_TH, IDX_TH] = np.eye(3) * self.Q_theta
        Q[IDX_W, IDX_W] = np.eye(3) * self.Q_w
        return Q

    def QN(self) -> np.ndarray:
        return self.Q()

    def R(self) -> np.ndarray:
        return np.eye(self.nu) * self.R_u


def make_config(scenario: str = "straight_trot") -> MPCConfig:
    cfg = MPCConfig()
    cfg.scenario_name = scenario

    if scenario == "straight_trot":
        cfg.desired_speed = 0.5
        cfg.desired_accel = 0.5
        cfg.desired_yaw = 0.0
        cfg.desired_yaw_rate = 0.0
        cfg.sim_time = 3.0
    elif scenario == "turn_pi_over_4":
        # Paper-inspired turning case: heading to pi/4 while maintaining constant forward speed.
        cfg.desired_speed = 0.35
        cfg.desired_accel = 0.5
        cfg.desired_yaw = float(np.pi / 4.0)
        cfg.desired_yaw_rate = 0.45
        cfg.sim_time = 3.0
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return cfg
