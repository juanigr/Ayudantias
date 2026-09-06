"""Conversión a Python de ``MAMH.m``.

La función mantiene el orden de argumentos y de resultados del código MATLAB.
Usa NumPy en CPU y, opcionalmente, CuPy en GPU.
"""

from pathlib import Path
from time import perf_counter

import numpy as np


def _backend(use_gpu):
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
    return x if xp is np else xp.asnumpy(x)


def _default_get_plan(p, q):
    """Plan inicial independiente con marginales ``p`` y ``q``."""
    return p[:, None] * q[None, :]


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


def MAMH(
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
    """Ejecuta el algoritmo MAMH traducido desde MATLAB.

    ``d[m]`` debe tener dimensiones ``(R, S[m])`` y ``q[m]`` debe tener
    longitud ``S[m]``. Si no se entrega ``get_plan``, el plan inicial es el
    producto exterior ``p[:, None] * q[m][None, :]``.

    Retorna ``(p, val, cpu, theta)``, igual que la función original.
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

    lbd = 40.0
    aux = 2.0 * lbd - 1.0
    d = [lbd * d[m] / rho for m in range(M)]

    avg = p.copy()
    val = 0.0
    next_print = 0.0
    cpu = 0.0
    iteration = 0
    nx = xp.inf

    while cpu <= MaxCPU:
        iteration += 1
        cpu = perf_counter() - start

        projected_avg = avg - (xp.sum(avg) - 1.0) / R
        if cpu >= next_print:
            nx = xp.linalg.norm(p - projected_avg)
            nx_value = float(_to_numpy(nx, xp))
            print(
                f"k = {iteration:5d}, |pk-pkk| = {nx_value:5.2e}, "
                f"cpu = {cpu:5.0f}"
            )
            if save_figures:
                _save_image(
                    p, kn, iteration, next_print, "MAMH", output_dir, xp
                )
            if nx_value <= tol:
                break
            next_print += PrintEvery

        p = projected_avg
        val = 0.0
        avg = xp.zeros(R, dtype=float)

        for m in range(M):
            qm = xp.sum(theta[m], axis=0)
            beta = (
                (p - pk[m])[:, None] / S[m]
                + (q[m] - qm)[None, :] / R
                + (xp.sum(qm) - 1.0) / (R * S[m])
            )
            theta[m] = xp.maximum(
                theta[m] + aux * beta - d[m], -beta
            )
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
            p, kn, iteration, round(cpu), "MAMH", output_dir, xp
        )

    return p, val, cpu, theta


__all__ = ["MAMH"]
