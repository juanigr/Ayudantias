"""Conversión a Python de ``LP_WB.m``."""

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def LP_WB(D, Q, M, R, S, display=True, method="highs"):
    """Resuelve el baricentro de Wasserstein como un programa lineal.

    El original llama a ``linprogGurobi``. Esta versión usa
    ``scipy.optimize.linprog`` con HiGHS, por lo que no requiere Gurobi.

    Retorna ``(p, val, cpu, m, n)`` como la función MATLAB.
    """
    start = perf_counter()
    M = int(M)
    R = int(R)
    S = np.asarray(S, dtype=int).reshape(-1)

    if len(D) != M or len(Q) != M or S.size != M:
        raise ValueError("M debe coincidir con las longitudes de D, Q y S.")

    costs = []
    constraint_blocks = []
    barycenter_blocks = []
    rhs = []

    for index in range(M):
        s = int(S[index])
        distance = np.asarray(D[index], dtype=float)
        q = np.asarray(Q[index], dtype=float).reshape(-1)

        if distance.shape != (R, s):
            raise ValueError(
                f"D[{index}] debe tener forma {(R, s)}; "
                f"tiene {distance.shape}."
            )
        if q.size != s:
            raise ValueError(f"Q[{index}] debe tener longitud {s}.")

        costs.append(distance.reshape(R * s, order="F"))

        column_sums = sparse.kron(
            sparse.eye(s, format="csr"), np.ones((1, R)), format="csr"
        )
        row_sums = sparse.kron(
            np.ones((1, s)), sparse.eye(R, format="csr"), format="csr"
        )
        constraint_blocks.append(
            sparse.vstack((column_sums, row_sums), format="csr")
        )
        barycenter_blocks.append(
            sparse.vstack(
                (sparse.csr_matrix((s, R)), -sparse.eye(R, format="csr")),
                format="csr",
            )
        )
        rhs.extend((q, np.zeros(R, dtype=float)))

    objective = np.concatenate((*costs, np.zeros(R, dtype=float)))
    transport_constraints = sparse.block_diag(
        constraint_blocks, format="csr"
    )
    barycenter_constraints = sparse.vstack(
        barycenter_blocks, format="csr"
    )
    A_eq = sparse.hstack(
        (transport_constraints, barycenter_constraints), format="csr"
    )
    b_eq = np.concatenate(rhs)

    rows, columns = A_eq.shape
    result = linprog(
        objective,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method=method,
        options={"disp": bool(display)},
    )

    if not result.success:
        raise RuntimeError(
            f"El programa lineal no pudo resolverse: {result.message}"
        )

    p = result.x[-R:]
    cpu = perf_counter() - start
    return p, float(result.fun), cpu, rows, columns


lp_wb = LP_WB

__all__ = ["LP_WB", "lp_wb"]
