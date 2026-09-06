"""Conversión a Python de ``GetPlan.m``."""

import numpy as np


def _array_module(array):
    """Detecta NumPy o CuPy sin exigir CuPy en equipos que usan CPU."""
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp
    return np


def GetPlan(p, q, R=None, S=None):
    """Construye un plan con marginales ``p`` y ``q``.

    Es la traducción de

    ``pi = repmat(p, 1, S)/S + repmat(q', R, 1)/R - 1/(R*S)``.

    Si ``p`` y ``q`` son vectores de probabilidad, las sumas por fila de
    ``pi`` son ``p`` y las sumas por columna son ``q``.
    """
    xp = _array_module(p)
    p = xp.asarray(p, dtype=float).reshape(-1)
    q = xp.asarray(q, dtype=float).reshape(-1)

    if R is None:
        R = p.size
    if S is None:
        S = q.size
    R = int(R)
    S = int(S)

    if p.size != R:
        raise ValueError(f"p debe tener longitud R={R}; tiene {p.size}.")
    if q.size != S:
        raise ValueError(f"q debe tener longitud S={S}; tiene {q.size}.")

    return p[:, None] / S + q[None, :] / R - 1.0 / (R * S)


get_plan = GetPlan

__all__ = ["GetPlan", "get_plan"]
