"""Capa de post-proceso y geometría: solo visualización, sin física ni numérica propia."""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from numerical_engine.numerical_stability_region import create_mesh, stability_region
from space_physics.Euler_equations import Euler_Equations

from .config import LIGHT, SCHEMES, Array


# ================================================================================================
# GEOMETRY AND POST-PROCESSING HELPERS (no physics or numerics here, only visualisation)
# ================================================================================================

def box_semiaxes(inertia: Array) -> tuple[Array, bool]:
	"""Semiejes de la caja homogénea con momentos (I1, I2, I3): a_i² ∝ I_j + I_k − I_i.

	El eje de inercia mínima es el más largo. Para que siga pareciendo un CubeSat, ningún
	lado baja del 40 % del mayor; devuelve también si ha hecho falta ese recorte (lo que
	ocurre siempre que I3 ≥ I1 + I2, donde no existe sólido real).
	"""
	i1, i2, i3 = inertia
	squared = np.array([i2 + i3 - i1, i1 + i3 - i2, i1 + i2 - i3], dtype=float)
	semiaxes = np.sqrt(np.clip(squared, 0.0, None))
	semiaxes = 0.48 * semiaxes / semiaxes.max()
	clamped = bool(np.any(semiaxes < 0.4 * 0.48))
	return np.maximum(semiaxes, 0.4 * 0.48), clamped


def _quad(center: Array, u: Array, v: Array) -> Array:
	"""Rectángulo de centro `center` y semilados u, v."""
	return np.array([center - u - v, center + u - v, center + u + v, center - u + v])


def _prism(center: Array, half: Array) -> list[Array]:
	corners = np.array([center + half * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
	return [corners[list(quad)] for quad in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))]


def satellite_model(inertia: Array) -> tuple[list[Array], Array, list[tuple[Array, str, float]], bool]:
	"""CubeSat en ejes cuerpo, con el eje 1 (inercia mínima) como eje largo.

	Bus de aluminio con raíles en las cuatro aristas largas, células solares en las caras
	laterales, dos alas solares desplegables, cámara, antena de parche y cuatro antenas
	UHF de cinta en configuración turnstile. Devuelve (caras, colores RGB, líneas, recortado).
	"""
	(a, b, c), clamped = box_semiaxes(inertia)
	faces: list[Array] = []
	colors: list[tuple[float, float, float]] = []
	aluminium, rail, cell, substrate = (0.72, 0.74, 0.77), (0.42, 0.44, 0.47), (0.10, 0.17, 0.38), (0.30, 0.31, 0.33)
	e1, e2, e3 = np.eye(3)

	def add(polygons: list[Array] | Array, color: tuple[float, float, float]) -> None:
		for polygon in polygons if isinstance(polygons, list) else [polygons]:
			faces.append(polygon)
			colors.append(color)

	add(_prism(np.zeros(3), np.array([a, b, c])), aluminium)
	# Raíles (salen un poco por los extremos, como en el estándar CubeSat).
	rail_half = 0.035
	for sy in (-1, 1):
		for sz in (-1, 1):
			add(_prism(np.array([0.0, sy * b, sz * c]), np.array([a + 0.035, rail_half, rail_half])), rail)

	# Células solares en las caras ±e2 y ±e3, una por «U» a lo largo del eje largo.
	units = 3
	pitch = 2 * a / units
	for k in range(units):
		x = -a + pitch * (k + 0.5)
		for sign in (-1, 1):
			add(_quad(np.array([x, sign * (b + 0.004), 0.0]), 0.42 * pitch * e1, 0.72 * c * e3), cell)
			add(_quad(np.array([x, 0.0, sign * (c + 0.004)]), 0.42 * pitch * e1, 0.72 * b * e2), cell)

	# Alas desplegables abisagradas en las aristas +e3 de las caras ±e2, en el plano 1-2.
	span, rows = 1.5 * b + 0.15, 2
	for sign in (-1, 1):
		y0 = sign * (b + rail_half)
		center = np.array([0.0, y0 + sign * span / 2, c])
		add(_quad(center, (a + 0.02) * e1, span / 2 * e2), substrate)
		for k in range(units):
			for row in range(rows):
				x = -a + pitch * (k + 0.5)
				y = y0 + sign * span * (row + 0.5) / rows
				add(_quad(np.array([x, y, c + 0.006]), 0.44 * pitch * e1, 0.44 * span / rows * e2), cell)

	# Cámara en −e1 y antena de parche S-band en −e3.
	angles = np.linspace(0.0, 2 * np.pi, 13)
	radius = 0.45 * min(b, c)
	lens = np.column_stack([np.full_like(angles, -a - 0.006), radius * np.cos(angles), radius * np.sin(angles)])
	add(lens, (0.06, 0.06, 0.08))
	add(_quad(np.array([0.55 * a, 0.0, -c - 0.008]), 0.22 * a * e1, 0.45 * b * e2), (0.93, 0.93, 0.90))

	# Antenas UHF (turnstile) en la cara +e1: se dibujan como líneas, visibles desde cualquier vista.
	nan = [np.nan] * 3
	tip = 0.55
	antennas = []
	for direction in (e2, -e2, e3, -e3):
		root = np.array([a + 0.01, 0.0, 0.0]) + direction * 0.6 * min(b, c)
		antennas += [root, root + direction * tip + np.array([0.05, 0.0, 0.0]), nan]
	lines = [(np.array(antennas, dtype=float), "#1d1f22", 1.7)]
	return faces, np.asarray(colors), lines, clamped


def shaded_colors(faces: list[Array], base_colors: Array) -> Array:
	"""Sombreado lambertiano (doble cara) respecto a una luz fija en ejes inerciales."""
	normals = np.array([np.cross(face[1] - face[0], face[2] - face[0]) for face in faces])
	normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15
	intensity = 0.42 + 0.58 * np.abs(normals @ LIGHT)
	rgb = np.clip(base_colors * intensity[:, None], 0.0, 1.0)
	return np.column_stack([rgb, np.ones(len(faces))])


def ellipsoid(radii: Array, nu: int = 32, nv: int = 20) -> tuple[Array, Array, Array]:
	u = np.linspace(0, 2 * np.pi, nu)
	v = np.linspace(0, np.pi, nv)
	return (
		radii[0] * np.outer(np.cos(u), np.sin(v)),
		radii[1] * np.outer(np.sin(u), np.sin(v)),
		radii[2] * np.outer(np.ones_like(u), np.cos(v)),
	)


def format_spectrum(eigenvalues: Array) -> str:
	"""LaTeX de «0, ±1.814» (silla) o «0, ±1.814 i» (centro)."""
	dominant = eigenvalues[np.argmax(np.abs(eigenvalues))]
	if abs(dominant) < 1e-12:
		return r"$0,\ 0,\ 0$"
	if abs(dominant.real) >= abs(dominant.imag):
		return rf"$0,\ \pm {abs(dominant.real):.3f}$"
	return rf"$0,\ \pm {abs(dominant.imag):.3f}\,i$"


def latex_sci(value: float, digits: int = 2) -> str:
	"""Número en notación científica LaTeX (sin $): 3.21e-03 → 3.21\\times10^{-3}."""
	if not np.isfinite(value):
		return r"\mathrm{—}"
	if value == 0:
		return "0"
	mantissa, exponent = f"{value:.{digits}e}".split("e")
	return rf"{mantissa}\times10^{{{int(exponent)}}}"


def error_summary(error: Array, solution: Array, times: Array) -> dict:
	"""Máximos del error de Richardson E(t) (N x 3) frente a la solución ω(t) del mismo esquema."""
	norm = np.linalg.norm(error, axis=1)
	finite = np.isfinite(norm)
	if not np.any(finite):
		return {"max": np.nan, "time": np.nan, "relative": np.nan, "components": np.full(3, np.nan), "final": np.nan}
	index = int(np.nanargmax(np.where(finite, norm, -np.inf)))
	with np.errstate(divide="ignore", invalid="ignore"):
		relative = norm / np.linalg.norm(solution, axis=1)
	return {
		"max": float(norm[index]),
		"time": float(times[index]),
		"relative": float(np.nanmax(relative)) if np.any(np.isfinite(relative)) else np.nan,
		"components": np.nanmax(np.abs(error), axis=0),
		"final": float(norm[-1]),
	}


def log_safe(values: Array) -> Array:
	"""Ceros exactos → NaN para poder dibujar en escala logarítmica."""
	return np.where(values > 0, values, np.nan)


@lru_cache(maxsize=16)
def stability_map(name: str, extent: float) -> tuple[Array, Array, Array]:
	"""|R(z)| de stability_region (numerical_engine) en una malla de create_mesh, en caché."""
	X, Y = create_mesh((-1.25 * extent, 0.75 * extent), (-extent, extent), 140)
	_, R = stability_region(SCHEMES[name][0], X, Y)
	return X, Y, np.abs(R)


def amplification_at(name: str, points: Array) -> Array:
	"""|R(z)| de stability_region en los puntos z = λΔt."""
	return np.array([
		abs(stability_region(SCHEMES[name][0], np.array([[z.real]]), np.array([[z.imag]]))[1][0, 0])
		for z in points
	])


def first_invalid(solution: Array) -> int:
	"""Primer índice no finito (divergencia numérica) o len(solution)."""
	bad = ~np.all(np.isfinite(solution), axis=1)
	return int(np.argmax(bad)) if bad.any() else len(solution)


def set_inertias(inertia: Array) -> None:
	"""Fija las inercias globales de space_physics que lee Euler_Equations(omega, t)."""
	physics = sys.modules[Euler_Equations.__module__]
	physics.I1, physics.I2, physics.I3 = (float(value) for value in inertia)
