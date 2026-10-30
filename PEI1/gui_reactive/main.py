"""Punto de entrada de la GUI de PEI 1 (simulador interactivo de actitud satelital).

Se puede lanzar de cualquiera de estas formas:
    python PEI1/gui_reactive/main.py          (desde la raíz del repositorio)
    python -m gui_reactive.main               (desde la carpeta PEI1)
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

if __package__ in (None, ""):
	# Ejecutado como script: se añade PEI1 al path para encontrar los grupos
	# numerical_engine, space_physics y gui_reactive.
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	from gui_reactive.app import SatelliteAttitudeApp
else:
	from .app import SatelliteAttitudeApp


def main() -> None:
	root = tk.Tk()
	SatelliteAttitudeApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
