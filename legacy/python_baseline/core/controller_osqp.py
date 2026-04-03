import osqp
import numpy as np
from qp_builder import QPData


class MPCControllerOSQP:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def solve(self, qp: QPData) -> tuple[np.ndarray, object]:
        prob = osqp.OSQP()
        prob.setup(
            P=qp.P,
            q=qp.q,
            A=qp.A,
            l=qp.l,
            u=qp.u,
            warm_start=True,
            verbose=self.verbose,
            polish=False,
        )

        res = prob.solve()
        status = getattr(res.info, "status", "")
        if not str(status).startswith("solved"):
            raise RuntimeError(f"OSQP failed with status: {status}")

        start = (qp.N + 1) * qp.nx
        stop = start + qp.nu
        u0 = np.asarray(res.x[start:stop], dtype=float).reshape(-1)

        return u0, res
