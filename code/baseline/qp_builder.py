from dataclasses import dataclass
import numpy as np
from scipy import sparse
from baseline.config import MPCConfig


@dataclass
class QPData:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    l: np.ndarray
    u: np.ndarray
    nx: int
    nu: int
    N: int


def _x_slice(k: int, nx: int) -> slice:
    return slice(k * nx, (k + 1) * nx)


def _u_slice(k: int, nx: int, nu: int, N: int) -> slice:
    xdim = (N + 1) * nx
    return slice(xdim + k * nu, xdim + (k + 1) * nu)


def leg_friction_block(mu: float) -> np.ndarray:
    """
    Implements the linearized friction-pyramid block:
        |fx| <= mu * fz
        |fy| <= mu * fz
        fz_min <= fz <= fz_max
    """
    return np.array([
        [-1.0,  0.0, -mu],
        [ 1.0,  0.0, -mu],
        [ 0.0, -1.0, -mu],
        [ 0.0,  1.0, -mu],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0,  1.0],
    ], dtype=float)


def build_qp(
    x_init: np.ndarray,
    x_ref: np.ndarray,
    Ad_list: list[np.ndarray],
    Bd_list: list[np.ndarray],
    contact_schedule: np.ndarray,
    cfg: MPCConfig,
) -> QPData:
    """
    Sparse stacked OSQP form:
        z = [x_0, ..., x_N, u_0, ..., u_{N-1}]
    """
    nx, nu, N = cfg.nx, cfg.nu, cfg.horizon
    nvar = (N + 1) * nx + N * nu

    Q = cfg.Q()
    QN = cfg.QN()
    R = cfg.R()

    # Objective: 1/2 z^T P z + q^T z
    P_blocks = [
        sparse.kron(sparse.eye(N, format="csc"), 2.0 * sparse.csc_matrix(Q)),
        2.0 * sparse.csc_matrix(QN),
        sparse.kron(sparse.eye(N, format="csc"), 2.0 * sparse.csc_matrix(R)),
    ]
    P = sparse.block_diag(P_blocks, format="csc")

    qx = np.concatenate(
        [-2.0 * (Q @ x_ref[k]) for k in range(N)] +
        [-2.0 * (QN @ x_ref[N])]
    )
    qu = np.zeros(N * nu, dtype=float)
    q = np.concatenate([qx, qu])

    A_rows: list[sparse.csc_matrix] = []
    l_list: list[float] = []
    u_list: list[float] = []

    # 1) Initial state equality: x_0 = x_init
    A0 = sparse.lil_matrix((nx, nvar), dtype=float)
    A0[:, _x_slice(0, nx)] = np.eye(nx)
    A_rows.append(A0.tocsc())
    l_list.extend(x_init.tolist())
    u_list.extend(x_init.tolist())

    # 2) Dynamics equalities: x_{k+1} - Ad x_k - Bd u_k = 0
    for k in range(N):
        Aeq = sparse.lil_matrix((nx, nvar), dtype=float)
        Aeq[:, _x_slice(k, nx)] = -Ad_list[k]
        Aeq[:, _x_slice(k + 1, nx)] = np.eye(nx)
        Aeq[:, _u_slice(k, nx, nu, N)] = -Bd_list[k]

        A_rows.append(Aeq.tocsc())
        l_list.extend([0.0] * nx)
        u_list.extend([0.0] * nx)

    # 3) Input constraints
    F_leg = leg_friction_block(cfg.mu)
    b_leg = np.array([0.0, 0.0, 0.0, 0.0, -cfg.fz_min, cfg.fz_max], dtype=float)

    for k in range(N):
        u_block = _u_slice(k, nx, nu, N)

        for leg in range(4):
            cols = slice(u_block.start + 3 * leg, u_block.start + 3 * leg + 3)

            if bool(contact_schedule[k, leg]):
                # Stance leg: friction pyramid + normal force bounds
                Aineq = sparse.lil_matrix((6, nvar), dtype=float)
                Aineq[:, cols] = F_leg
                A_rows.append(Aineq.tocsc())

                l_list.extend([-np.inf] * 6)
                u_list.extend(b_leg.tolist())
            else:
                # Swing leg: force = 0
                Asw = sparse.lil_matrix((3, nvar), dtype=float)
                Asw[:, cols] = np.eye(3)
                A_rows.append(Asw.tocsc())

                l_list.extend([0.0, 0.0, 0.0])
                u_list.extend([0.0, 0.0, 0.0])

    A = sparse.vstack(A_rows, format="csc")
    l = np.asarray(l_list, dtype=float)
    u = np.asarray(u_list, dtype=float)

    return QPData(P=P, q=q, A=A, l=l, u=u, nx=nx, nu=nu, N=N)
