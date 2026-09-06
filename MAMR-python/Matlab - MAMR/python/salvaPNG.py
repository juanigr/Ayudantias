"""Reemplazo en Python de la función ausente ``salvaPNG.m``."""

from pathlib import Path


def salvaPNG(fig, name, dpi=150):
    """Guarda una figura de Matplotlib en formato PNG.

    La carpeta de destino se crea automáticamente cuando no existe.
    """
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


save_png = salvaPNG

__all__ = ["salvaPNG", "save_png"]
