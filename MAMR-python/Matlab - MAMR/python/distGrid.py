"""Conversión a Python de ``distGrid.m``."""

import numpy as np
from scipy.spatial.distance import cdist


def distGrid(K, M):
    """Calcula las distancias euclidianas entre dos grillas.

    La primera grilla tiene paso ``1/M`` y la segunda tiene paso uno. El
    resultado posee dimensiones

    ``((M * (K - 1) + 1)**2, K**2)``.

    Esta implementación conserva el orden por columnas de MATLAB.
    """
    K = int(K)
    M = int(M)
    if K < 1:
        raise ValueError("K debe ser un entero positivo.")
    if M < 1:
        raise ValueError("M debe ser un entero positivo.")

    fine_size = M * (K - 1) + 1
    fine_axis = np.linspace(1.0, float(K), fine_size)
    base_axis = np.arange(1.0, K + 1.0)

    fine_x, fine_y = np.meshgrid(fine_axis, fine_axis, indexing="xy")
    base_x, base_y = np.meshgrid(base_axis, base_axis, indexing="xy")

    fine_points = np.column_stack(
        (fine_x.ravel(order="F"), fine_y.ravel(order="F"))
    )
    base_points = np.column_stack(
        (base_x.ravel(order="F"), base_y.ravel(order="F"))
    )

    return cdist(fine_points, base_points, metric="euclidean")


dist_grid = distGrid

__all__ = ["distGrid", "dist_grid"]
