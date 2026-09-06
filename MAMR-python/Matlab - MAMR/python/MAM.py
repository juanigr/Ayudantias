"""Conversión a Python de ``MAM.m``.

La función mantiene el orden de argumentos y de resultados del código MATLAB.
Usa NumPy en CPU y, opcionalmente, CuPy en GPU.
"""

from pathlib import Path
from time import perf_counter

import numpy as np


def _backend(use_gpu):
    """Devuelve NumPy o CuPy según el valor de ``use_gpu``."""
    if not use_gpu:
        return np

    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "UseGPU=True requiere instalar CuPy con soporte para la versión "
            "de CUDA disponible en el equipo."
        ) from exc
    return cp


def _to_numpy(x, xp):
    """Copia un arreglo a CPU únicamente cuando se está usando CuPy."""
    return x if xp is np else xp.asnumpy(x)


def _default_get_plan(p, q):
    """Traducción exacta de la función MATLAB ``GetPlan``."""
    r = p.shape[0]
    s = q.shape[0]
    return p[:, None] / s + q[None, :] / r - 1.0 / (r * s)


def _project_simplex_columns(y, xp):
    """Proyecta cada columna de ``y`` sobre el simplex de probabilidad."""
    y_sorted = xp.sort(y, axis=0)[::-1, :]
    cumulative = xp.cumsum(y_sorted, axis=0)
    indices = xp.arange(1, y.shape[0] + 1, dtype=y.dtype)[:, None]
    threshold = xp.max((cumulative - 1.0) / indices, axis=0)
    return xp.maximum(y - threshold[None, :], 0.0)


def _save_image(p, kn, iteration, shown_time, suffix, output_dir, xp):
    """Equivalente de ``imagesc`` y ``salvaPNG`` del archivo original."""
    import matplotlib.pyplot as plt

    image = _to_numpy(1.0 - p, xp).reshape((kn, kn), order="F")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="hot", aspect="auto", origin="upper")
    ax.set_title(f"k={iteration}, t={shown_time}")
    fig.savefig(output_dir / f"{shown_time}-{suffix}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def MAM(
    d,
    q,
    M,
    R,
    S,
    p,
    rho,
    UseGPU=False,
    tol=1e-6,
    MaxCPU=60.0,
    PrintEvery=5.0,
    get_plan=None,
    save_figures=True,
    output_dir="Fig",
):
    """Ejecuta el algoritmo MAM.

    Parámetros
    ----------
    d : secuencia de matrices
        ``d[m]`` debe tener dimensiones ``(R, S[m])``.
    q : secuencia de vectores
        ``q[m]`` debe tener longitud ``S[m]``.
    M, R : int
        Cantidad de medidas y tamaño del vector ``p``, respectivamente.
    S : secuencia de int
        Tamaños de los vectores almacenados en ``q``.
    p : arreglo de longitud R
        Punto inicial.
    rho : float
        Parámetro positivo del algoritmo.

    Los restantes parámetros corresponden a los del archivo MATLAB. Si se
    dispone de la función ``GetPlan`` original, puede pasarse su equivalente
    mediante ``get_plan``. En caso contrario se utiliza internamente la
    traducción exacta de la función MATLAB ``GetPlan``.

    Retorna
    -------
    p, val, cpu, theta
        Los mismos cuatro resultados que la función MATLAB.
    """
    start = perf_counter()
    xp = _backend(UseGPU)

    M = int(M)
    R = int(R)
    S = np.asarray(S, dtype=int).reshape(-1)
    if len(S) != M or len(d) != M or len(q) != M:
        raise ValueError("M debe coincidir con las longitudes de d, q y S.")
    if rho == 0:
        raise ValueError("rho debe ser distinto de cero.")

    kn = round(np.sqrt(R))
    if kn * kn != R:
        raise ValueError("R debe ser un cuadrado perfecto para generar imágenes.")

    p = xp.asarray(p, dtype=float).reshape(R)
    q = [xp.asarray(q[m], dtype=float).reshape(S[m]) for m in range(M)]
    d = [xp.asarray(d[m], dtype=float).reshape(R, S[m]) for m in range(M)]

    if get_plan is None:
        theta = [_default_get_plan(p, q[m]) for m in range(M)]
    else:
        theta = [
            xp.asarray(get_plan(p, q[m], R, int(S[m])), dtype=float).reshape(
                R, S[m]
            )
            for m in range(M)
        ]
    pk = [p.copy() for _ in range(M)]

    weights = 1.0 / S.astype(float)
    weights /= weights.sum()
    avg = p.copy()
    val = 0.0

    next_print = 0.0
    cpu = 0.0
    iteration = 0
    nx = xp.inf

    while cpu <= MaxCPU:
        iteration += 1
        cpu = perf_counter() - start

        if cpu >= next_print:
            nx = xp.linalg.norm(p - avg)
            nx_value = float(_to_numpy(nx, xp))
            print(
                f"k = {iteration:5d}, |pk-pkk| = {nx_value:5.2e}, "
                f"cpu = {cpu:5.0f}"
            )
            if save_figures:
                _save_image(
                    p, kn, iteration, next_print, "MAM", output_dir, xp
                )
            if nx_value <= tol:
                break
            next_print += PrintEvery

        p = avg.copy()
        avg = xp.zeros(R, dtype=float)

        for m in range(M):
            correction = 2.0 * (p - pk[m])[:, None] / S[m]
            y = (theta[m] + correction - d[m] / rho) / q[m][None, :]
            pihat = _project_simplex_columns(y, xp) * q[m][None, :]
            theta[m] = pihat - (p - pk[m])[:, None] / S[m]
            pk[m] = xp.sum(theta[m], axis=1)
            avg += weights[m] * pk[m]

    cpu = perf_counter() - start
    nx_value = float(_to_numpy(nx, xp))
    print(
        f"k = {iteration:5d}, |pk-pkk| = {nx_value:5.2e}, "
        f"cpu = {cpu:5.0f}"
    )

    if save_figures:
        _save_image(
            p, kn, iteration, round(cpu), "MAM", output_dir, xp
        )

    return p, val, cpu, theta


__all__ = ["MAM"]
