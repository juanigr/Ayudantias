"""Conversión a Python de ``main.m``.

Este es el punto de entrada del proyecto. Debe estar en la misma carpeta que
los demás módulos convertidos y que las carpetas ``dataPeyre`` o
``dataAltschuler``.
"""

from pathlib import Path

import numpy as np

from GetPlan import GetPlan
from LP_WB import LP_WB
from MAM import MAM
from MAMH import MAMH
from MAMR import MAMR
from bregmanWassersteinBarycenter import bregmanWassersteinBarycenter
from distGrid import distGrid
from salvaPNG import salvaPNG


def _to_numpy(array):
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(array)
    return np.asarray(array)


def _matlab_round_nonnegative(values):
    """Equivale a ``round`` de MATLAB para las coordenadas no negativas."""
    return np.floor(values + 0.5)


def _read_histograms(base_dir, dataset, number_images, grid_size,
                     save_figures, output_dir):
    if dataset not in (1, 2):
        raise ValueError("dataset debe ser 1 (Peyre) o 2 (Altschuler).")

    import matplotlib.pyplot as plt

    Q = np.zeros((grid_size * grid_size, number_images), dtype=float)
    fig = None
    axes = None
    if save_figures:
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        axes = axes.ravel()

    for index in range(number_images):
        number = index + 1
        if dataset == 1:
            path = base_dir / "dataPeyre" / f"{number}.csv"
            data = np.loadtxt(path, delimiter=",")
            data = np.atleast_2d(data).astype(float)
            total = data[:, 2].sum()
            if total <= 0:
                raise ValueError(f"Las masas de {path.name} no suman positivo.")
            data[:, 2] /= total
        else:
            path = base_dir / "dataAltschuler" / f"{number}.txt"
            data = np.loadtxt(path)
            data = np.atleast_2d(data).astype(float)
            data[:, :2] = _matlab_round_nonnegative(
                grid_size * data[:, :2]
            )

        image = np.zeros((grid_size, grid_size), dtype=float)
        for row in data:
            matlab_i = int(row[0])
            matlab_j = max(int(row[1]), 1)
            if not (1 <= matlab_i <= grid_size and 1 <= matlab_j <= grid_size):
                raise IndexError(
                    f"Coordenada ({matlab_i}, {matlab_j}) fuera de la "
                    f"grilla en {path}."
                )
            image[matlab_i - 1, matlab_j - 1] = row[2]

        Q[:, index] = image.reshape(-1, order="F")

        if save_figures and index < 25:
            axes[index].imshow(1.0 - image, cmap="hot", origin="upper")
            axes[index].set_xticks([])
            axes[index].set_yticks([])
            axes[index].set_title(str(number))

    if save_figures:
        for axis in axes[min(number_images, 25):]:
            axis.axis("off")
        fig.tight_layout()
        salvaPNG(fig, output_dir / "input-histograms.png")
        plt.close(fig)

    return Q


def main(
    methods=(3,),
    dataset=1,
    exact_wb=False,
    use_gpu=False,
    max_cpu=30.0,
    print_every=5.0,
    tol=-np.inf,
    number_images=10,
    grid_size=60,
    rho=5e3,
    lambda_=300.0,
    base_dir=None,
    output_dir=None,
    save_figures=True,
):
    """Ejecuta uno o más métodos del programa original.

    Métodos: 1=IBP, 2=MAM, 3=MAM-R, 4=MAM-H y 5=LP.
    ``base_dir`` debe contener ``dataPeyre`` o ``dataAltschuler``.
    """
    if isinstance(methods, (int, np.integer)):
        methods = (int(methods),)
    methods = tuple(int(method) for method in methods)
    if any(method not in (1, 2, 3, 4, 5) for method in methods):
        raise ValueError("Los métodos posibles son 1, 2, 3, 4 y 5.")

    base_dir = (
        Path(base_dir) if base_dir is not None
        else Path(__file__).resolve().parent
    )
    output_dir = (
        Path(output_dir) if output_dir is not None else base_dir / "Fig"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    number_images = int(number_images)
    grid_size = int(grid_size)

    print("Computing distance...")
    if exact_wb:
        kn = number_images * (grid_size - 1) + 1
        R = kn * kn
        D = distGrid(grid_size, number_images) ** 2 / (grid_size**2)
    else:
        kn = grid_size
        R = grid_size * grid_size
        D = distGrid(grid_size, 1) ** 2 / (grid_size**2)

    print("Reading data...")
    Q = _read_histograms(
        base_dir,
        dataset,
        number_images,
        grid_size,
        save_figures,
        output_dir,
    )

    results = {}
    for method in methods:
        if method == 1:
            if exact_wb:
                raise ValueError(
                    "El método IBP requiere que D y Q tengan igual número "
                    "de filas; use exact_wb=False."
                )
            print("Running IBP!")
            median = np.median(D)
            if median <= 0:
                raise ValueError("La mediana de D debe ser positiva.")
            p, _, _ = bregmanWassersteinBarycenter(
                Q,
                D / median,
                max_cpu,
                lambda_,
                use_gpu,
                tol,
                save_figures=save_figures,
                output_dir=output_dir,
                PrintEvery=print_every,
            )
        else:
            print("Arranging data...")
            supports = []
            q = []
            distances = []
            p_initial = np.ones(R, dtype=float) / R

            for index in range(number_images):
                mask = Q[:, index] > 1e-15
                support_size = int(mask.sum())
                if support_size == 0:
                    raise ValueError(
                        f"El histograma {index + 1} tiene soporte vacío."
                    )
                supports.append(support_size)
                distances.append(D[:, mask])
                q_vector = Q[mask, index].copy()
                q.append(q_vector / q_vector.sum())

            supports = np.asarray(supports, dtype=int)
            common = dict(
                d=distances,
                q=q,
                M=number_images,
                R=R,
                S=supports,
                p=p_initial,
                rho=rho,
                UseGPU=use_gpu,
                tol=tol,
                MaxCPU=max_cpu,
                PrintEvery=print_every,
                get_plan=GetPlan,
                save_figures=save_figures,
                output_dir=output_dir,
            )

            if method == 2:
                print("Running MAM!")
                p, _, _, _ = MAM(**common)
            elif method == 3:
                print("Running MAM-R!")
                p, _, _, _ = MAMR(**common)
            elif method == 4:
                print("Running MAM-H!")
                p, _, _, _ = MAMH(**common)
            else:
                print("Running LP with SciPy/HiGHS!")
                p, _, _, _, _ = LP_WB(
                    distances,
                    q,
                    number_images,
                    R,
                    supports,
                )

        p_array = _to_numpy(p).reshape((kn, kn), order="F")
        results[method] = p_array

        if save_figures:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.imshow(1.0 - p_array, cmap="hot", origin="upper")
            ax.set_title(f"Final method {method}")
            salvaPNG(fig, output_dir / f"Final-met-{method}.png")
            plt.close(fig)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baricentro de Wasserstein: conversión de main.m"
    )
    parser.add_argument("--method", type=int, nargs="+", default=[3])
    parser.add_argument("--dataset", type=int, choices=(1, 2), default=1)
    parser.add_argument("--exact-wb", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--max-cpu", type=float, default=30.0)
    parser.add_argument("--print-every", type=float, default=5.0)
    parser.add_argument("--tol", type=float, default=-np.inf)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--rho", type=float, default=5e3)
    parser.add_argument("--lambda-value", type=float, default=300.0)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    main(
        methods=args.method,
        dataset=args.dataset,
        exact_wb=args.exact_wb,
        use_gpu=args.use_gpu,
        max_cpu=args.max_cpu,
        print_every=args.print_every,
        tol=args.tol,
        number_images=args.images,
        grid_size=args.grid_size,
        rho=args.rho,
        lambda_=args.lambda_value,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
    )
