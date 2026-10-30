"""Capa de configuración: constantes de la GUI y esquemas numéricos disponibles."""

from __future__ import annotations

from typing import Callable

import numpy as np
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
AXIS_NAMES = ("I1 (mínimo)", "I2 (intermedio)", "I3 (máximo)")
AXIS_COLORS = ("#d95d39", "#167d73", "#3767a5")
MAX_STEPS = 20000
DT_MAX = 1.0
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])
BACKGROUND = "#f3f5f4"
CARD = "#ffffff"
STABLE_COLOR = "#1f7a4d"
UNSTABLE_COLOR = "#b3412e"
