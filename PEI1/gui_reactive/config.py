"""Capa de configuración: constantes de la GUI y esquemas numéricos disponibles."""

from __future__ import annotations

from typing import Callable

import matplotlib
import numpy as np
from matplotlib import font_manager
from numpy.typing import NDArray

from numerical_engine.numerical_schemes import Crank_Nicolson, Euler, Inverse_Euler, RK4


Array = NDArray[np.float64]
Scheme = Callable[[Array, float, float, Callable], Array]
SCHEMES: dict[str, tuple[Scheme, int]] = {
	"Euler": (Euler, 1),
	"Euler inverso": (Inverse_Euler, 1),
	"Crank-Nicolson": (Crank_Nicolson, 2),
	"Runge-Kutta 4": (RK4, 4),
}
SCHEME_COLORS = {"Euler": "#d95d39", "Euler inverso": "#8a4fa3", "Crank-Nicolson": "#3767a5", "Runge-Kutta 4": "#167d73"}
# Los Combobox de Tk no pueden mostrar LaTeX: se usan subíndices Unicode.
AXIS_NAMES = ("I₁ (mínimo)", "I₂ (intermedio)", "I₃ (máximo)")
AXIS_COLORS = ("#d95d39", "#167d73", "#3767a5")
MAX_STEPS = 20000
DT_MAX = 1.0
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])
BACKGROUND = "#f3f5f4"
CARD = "#ffffff"
STABLE_COLOR = "#1f7a4d"
UNSTABLE_COLOR = "#b3412e"
PAGE_TITLES = (
	"Satélite y Condiciones Iniciales",
	"Esquemas Numéricos",
	"Integración numérica",
	"Error Estimado",
)

# Fórmulas y variables en formato LaTeX (mathtext de Matplotlib) con tipografía Computer Modern,
# tanto en las gráficas como en las etiquetas de Tk (widgets.MathLabel).
matplotlib.rcParams["mathtext.fontset"] = "cm"
UI_FONT = "Segoe UI" if "Segoe UI" in {font.name for font in font_manager.fontManager.ttflist} else "DejaVu Sans"
