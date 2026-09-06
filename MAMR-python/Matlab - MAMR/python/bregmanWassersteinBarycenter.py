"""Conversión a Python de ``bregmanWassersteinBarycenter.m``."""

from pathlib import Path
from time import perf_counter

import numpy as np

from salvaPNG import salvaPNG


def _backend(use_gpu):
    if not use_gpu:
        return np
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "useGPU=True requiere instalar CuPy con soporte para CUDA."
        ) from exc
    return cp


def _to_numpy(array, xp):
    return array if xp is np else xp.asnumpy(array)


def _save_barycenter(c, kn, iteration, shown_time, output_dir, xp):
    import matplotlib.pyplot as plt

    image = _to_numpy(1.0 - c, xp).reshape((kn, kn), order="F")
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="hot", aspect="auto", origin="upper")
    ax.set_title(f"k={iteration}, t={shown_time}")
    salvaPNG(fig, Path(output_dir) / f"{shown_time}-IBP.png")
    plt.close(fig)


def bregmanWassersteinBarycenter(
    C,
    M,
    MaxCPU=60.0,
    lambda_=300.0,
    useGPU=False,
    toleranceDifference=1e-3,
    weights=None,
    save_figures=True,
    output_dir="Fig",
    PrintEvery=5.0,
):
    """Calcula un baricentro de Wasserstein mediante iteraciones de Bregman.

    Parameters
    ----------
    C : array, shape (d, n)
        Matriz cuyas columnas son histogramas.
    M : array, shape (d, d)
        Matriz de costos o métrica base.
    MaxCPU : float
        Tiempo máximo de ejecución, en segundos.
    lambda_ : float
        Parámetro llamado ``lambda`` en MATLAB. En Python se agrega el guion
        bajo porque ``lambda`` es una palabra reservada.

    Returns
    -------
    c, MVP, objectives
        Los tres resultados de la función MATLAB. ``objectives`` permanece
        vacío porque el archivo original nunca agrega valores a ese vector.

    Notes
    -----
    El archivo MATLAB recibe ``toleranceDifference``, pero su condición de
    parada por tolerancia está comentada. Esta conversión conserva ese
    comportamiento y detiene el proceso por ``MaxCPU``.
    """
    del toleranceDifference  # Conservado por compatibilidad con MATLAB.

    start = perf_counter()
    xp = _backend(useGPU)
    C = xp.asarray(C, dtype=float)
    metric = xp.asarray(M, dtype=float)

    if C.ndim != 2 or metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("C debe ser 2D y M debe ser una matriz cuadrada.")
    d, n = C.shape
    if metric.shape != (d, d):
        raise ValueError("M debe tener dimensiones (C.shape[0], C.shape[0]).")

    if weights is None:
        weights = xp.ones(n, dtype=float) / n
    else:
        weights = xp.asarray(weights, dtype=float).reshape(n)

    kernel = xp.exp(-float(lambda_) * metric)
    kernel = xp.maximum(kernel, 1e-300)

    column_sums = xp.sum(kernel, axis=0)
    UKv = kernel @ (C / column_sums[:, None])
    UKv = xp.maximum(UKv, 1e-300)
    geometric_mean = xp.exp(xp.log(UKv) @ weights)
    u = geometric_mean[:, None] / UKv

    count = 0
    matrix_vector = 0
    mvp = []
    objectives = np.array([], dtype=float)
    next_print = 0.0
    cpu = 0.0
    iteration = 0

    kn = round(np.sqrt(d))
    if save_figures and kn * kn != d:
        raise ValueError(
            "El número de filas de C debe ser un cuadrado perfecto para "
            "generar las imágenes."
        )

    while cpu <= MaxCPU:
        iteration += 1
        cpu = perf_counter() - start

        if cpu >= next_print:
            differ = xp.sum(xp.std(UKv, axis=1, ddof=0))
            differ_value = float(_to_numpy(differ, xp))
            c = xp.mean(UKv, axis=1)
            print(f"k = {count:5d}, differ = {differ_value:5.2e}")
            if save_figures:
                _save_barycenter(
                    c, kn, iteration, next_print, output_dir, xp
                )
            next_print += PrintEvery

        denominator = kernel.T @ u
        denominator = xp.maximum(denominator, 1e-300)
        UKv = u * (kernel @ (C / denominator))
        UKv = xp.maximum(UKv, 1e-300)
        matrix_vector += 2
        count += 1

        geometric_mean = xp.exp(xp.log(UKv) @ weights)
        u = u * geometric_mean[:, None] / UKv
        matrix_vector += 1
        mvp.append(matrix_vector)

    cpu = perf_counter() - start
    c = xp.mean(UKv, axis=1)
    if save_figures:
        _save_barycenter(c, kn, iteration, round(cpu), output_dir, xp)

    return c, np.asarray(mvp, dtype=int), objectives


bregman_wasserstein_barycenter = bregmanWassersteinBarycenter

__all__ = [
    "bregmanWassersteinBarycenter",
    "bregman_wasserstein_barycenter",
]
