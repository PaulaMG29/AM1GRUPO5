"""Interactive satellite attitude simulator for PEI 1."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

try:
	from .numerical_engine import (
		Cauchy_Problem,
		Crank_Nicolson,
		Euler,
		RK4,
		Richardson_Error,
	)
	from .space_physics import Euler_Equations, L, T, exact_Jacobian
except ImportError:
	from numerical_engine import (
		Cauchy_Problem,
		Crank_Nicolson,
		Euler,
		RK4,
		Richardson_Error,
	)
	from space_physics import Euler_Equations, L, T, exact_Jacobian


Array = NDArray[np.float64]
Scheme = Callable[[Array, float, float, Callable], Array]
SCHEMES: dict[str, tuple[Scheme, int]] = {
	"Euler": (Euler, 1),
	"Crank-Nicolson": (Crank_Nicolson, 2),
	"Runge-Kutta 4": (RK4, 4),
}
SCHEME_COLORS = {"Euler": "#d95d39", "Crank-Nicolson": "#3767a5", "Runge-Kutta 4": "#167d73"}
AXIS_NAMES = ("I1 (mínimo)", "I2 (intermedio)", "I3 (máximo)")
AXIS_COLORS = ("#d95d39", "#167d73", "#3767a5")
MAX_STEPS = 20000
DT_MAX = 1.0
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])
BACKGROUND = "#f3f5f4"
CARD = "#ffffff"
STABLE_COLOR = "#1f7a4d"
UNSTABLE_COLOR = "#b3412e"


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


def propagate_attitude(solution: Array, times: Array) -> Array:
	"""Matrices de actitud R(t) (cuerpo → inercial) con dR/dt = R [ω]×.

	Cada paso usa la fórmula de Rodrigues con la velocidad angular media del intervalo,
	por lo que R permanece ortogonal y la cinemática es de segundo orden.
	"""
	rotations = np.empty((len(times), 3, 3))
	rotations[0] = np.eye(3)
	for n in range(len(times) - 1):
		rotation_vector = 0.5 * (solution[n] + solution[n + 1]) * (times[n + 1] - times[n])
		angle = float(np.linalg.norm(rotation_vector))
		if angle < 1e-14:
			rotations[n + 1] = rotations[n]
			continue
		x, y, z = rotation_vector / angle
		skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
		increment = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
		rotations[n + 1] = rotations[n] @ increment
	return rotations


def ellipsoid(radii: Array, nu: int = 32, nv: int = 20) -> tuple[Array, Array, Array]:
	u = np.linspace(0, 2 * np.pi, nu)
	v = np.linspace(0, np.pi, nv)
	return (
		radii[0] * np.outer(np.cos(u), np.sin(v)),
		radii[1] * np.outer(np.sin(u), np.sin(v)),
		radii[2] * np.outer(np.ones_like(u), np.cos(v)),
	)


def separatrix_curves(inertia: Array, momentum: float) -> list[Array]:
	"""Separatriz analítica sobre el elipsoide |L| = momentum (nivel 2T = L²/I2).

	Es la intersección del elipsoide de momento con los planos
	√(I1(I2−I1)) ω1 = ±√(I3(I3−I2)) ω3: dos elipses que pasan por ±ω*(2).
	"""
	i1, i2, i3 = inertia
	if momentum <= 0 or not (i1 < i2 < i3):
		return []
	slope = np.sqrt(i1 * (i2 - i1) / (i3 * (i3 - i2)))
	theta = np.linspace(0.0, 2 * np.pi, 361)
	omega1 = momentum * np.cos(theta) / np.sqrt(i1**2 + (i3 * slope) ** 2)
	omega2 = momentum * np.sin(theta) / i2
	return [np.column_stack([omega1, omega2, sign * slope * omega1]) for sign in (-1.0, 1.0)]


def equilibrium_spectrum(inertia: Array, omega_rate: float, axis_index: int) -> tuple[Array, bool]:
	"""Autovalores de exact_Jacobian en ω*(k) = Ω e_k y si el equilibrio es inestable."""
	equilibrium = np.zeros(3)
	equilibrium[axis_index] = omega_rate
	eigenvalues = np.linalg.eigvals(exact_Jacobian(*inertia, equilibrium))
	return eigenvalues, bool(np.max(eigenvalues.real) > 1e-9)


def format_spectrum(eigenvalues: Array) -> str:
	"""«0, ±1.814» (silla) o «0, ±1.814 i» (centro)."""
	dominant = eigenvalues[np.argmax(np.abs(eigenvalues))]
	if abs(dominant) < 1e-12:
		return "0, 0, 0"
	if abs(dominant.real) >= abs(dominant.imag):
		return f"0,  ±{abs(dominant.real):.3f}"
	return f"0,  ±{abs(dominant.imag):.3f} i"


def amplification_factor(name: str, z: Array) -> Array:
	"""R(z) de la ecuación test u' = λu, z = λΔt."""
	if name == "Euler":
		return 1 + z
	if name == "Crank-Nicolson":
		return (1 + z / 2) / (1 - z / 2)
	return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def first_invalid(solution: Array) -> int:
	"""Primer índice no finito (divergencia numérica) o len(solution)."""
	bad = ~np.all(np.isfinite(solution), axis=1)
	return int(np.argmax(bad)) if bad.any() else len(solution)


def relative_drift(values: Array) -> Array:
	"""|q(t) − q(0)| / |q(0)|, con los ceros exactos sustituidos por NaN para la escala log."""
	drift = np.abs(values - values[0]) / max(abs(values[0]), 1e-300)
	return np.where(drift > 0, drift, np.nan)


class _PlotRecorder:
	"""Sustituto de matplotlib.pyplot que guarda lo que se iba a pintar sin abrir ventanas.

	estimate_scheme_order (numerical_engine) pinta ln(E) frente a ln(N) con plt.show(), que
	bloquearía la GUI; así se reutiliza ese código tal cual y se recuperan sus datos.
	"""

	def __init__(self) -> None:
		self.x: Array = np.array([])
		self.y: Array = np.array([])

	def plot(self, x, y, *_args, **_kwargs) -> None:
		self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

	def __getattr__(self, _name: str) -> Callable:
		return lambda *_args, **_kwargs: None


_ENGINE_LOCK = threading.Lock()


def richardson_with_order(scheme: Scheme, initial: Array, field: Callable, times: Array) -> tuple[Array, float, Array, Array]:
	"""Richardson_Error de numerical_engine (con su estimate_scheme_order).

	Devuelve (error, orden estimado, ln N, ln E) de la recta de convergencia.
	"""
	engine = sys.modules[Richardson_Error.__module__]
	recorder = _PlotRecorder()
	with _ENGINE_LOCK:
		original = engine.plt
		engine.plt = recorder
		try:
			with np.errstate(all="ignore"):
				error, order = Richardson_Error(scheme, initial, field, times)
		finally:
			engine.plt = original
	return np.asarray(error, dtype=float), float(order), recorder.x, recorder.y


# ================================================================================================
# GUI
# ================================================================================================

class SatelliteAttitudeApp:
	"""Three-view Tkinter dashboard backed by the PEI1 numerical modules."""

	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("PEI 1 | Dinámica de actitud satelital")
		self.root.geometry("1480x940")
		self.root.minsize(1120, 760)
		self.root.configure(background=BACKGROUND)

		self.inertias = [0.5, 1.0, 1.5]
		self.omega_degrees = 180.0
		self.spin_axis = AXIS_NAMES[1]
		self.log_perturbation = -3.0
		self.dt = 0.02
		self.final_time = 20.0
		self.scheme_name = "Runge-Kutta 4"
		self.results: dict | None = None
		self.menu_open = True
		self.current_page = 0
		self.animation_object: animation.FuncAnimation | None = None
		self.animation_paused = False
		self.busy = False
		self._redraw_job: str | None = None
		self._syncing_inertias = False

		style = ttk.Style()
		style.theme_use("clam")
		style.configure("TFrame", background=BACKGROUND)
		style.configure("TLabel", background=BACKGROUND, foreground="#172b2a")
		style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#173b38")
		style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#245b55")
		style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#243b39")
		style.configure("Status.TLabel", font=("Segoe UI", 9, "italic"), foreground="#245b55")
		style.configure("TButton", padding=(10, 7))
		style.configure("Accent.TButton", background="#167d73", foreground="white")

		self.shell = ttk.Frame(root)
		self.shell.pack(fill="both", expand=True)
		self.shell.rowconfigure(0, weight=1)
		self.shell.columnconfigure(1, weight=1)
		self.sidebar = ttk.Frame(self.shell, padding=(12, 14), width=220)
		self.sidebar.grid(row=0, column=0, sticky="ns")
		self.sidebar.grid_propagate(False)
		self.content = ttk.Frame(self.shell, padding=(18, 14, 18, 6))
		self.content.grid(row=0, column=1, sticky="nsew")
		self.content.rowconfigure(1, weight=1)
		self.content.columnconfigure(0, weight=1)
		self.status_var = tk.StringVar()
		ttk.Label(self.shell, textvariable=self.status_var, style="Status.TLabel", padding=(18, 0, 18, 6)).grid(
			row=1, column=1, sticky="w"
		)
		self._build_sidebar()
		self.show_page(0)

	def _build_sidebar(self) -> None:
		for child in self.sidebar.winfo_children():
			child.destroy()
		self.sidebar.configure(width=220 if self.menu_open else 54)
		self.sidebar.grid_propagate(False)
		arrow = "‹" if self.menu_open else "›"
		ttk.Button(self.sidebar, text=arrow, width=3, command=self._toggle_menu).pack(anchor="w", pady=(0, 22))
		if self.menu_open:
			ttk.Label(self.sidebar, text="PEI 1", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
			for index, title in enumerate((
				"Satélite y Condiciones Iniciales",
				"Esquemas Numéricos",
				"Integración y Error",
			)):
				ttk.Button(
					self.sidebar,
					text=title,
					command=lambda page=index: self.show_page(page),
				).pack(fill="x", anchor="w", pady=4)

	def _toggle_menu(self) -> None:
		self.menu_open = not self.menu_open
		self._build_sidebar()

	def _clear_content(self) -> None:
		if self.animation_object is not None:
			self.animation_object.event_source.stop()
		self.animation_object = None
		for child in self.content.winfo_children():
			child.destroy()

	def show_page(self, page: int) -> None:
		if self.busy:
			return
		self.current_page = page
		self._clear_content()
		for row in range(1, 4):
			self.content.rowconfigure(row, weight=0)
		self.content.rowconfigure(1, weight=1)
		titles = (
			"Satélite y Condiciones Iniciales",
			"Esquemas Numéricos",
			"Integración y Error",
		)
		ttk.Label(self.content, text=titles[page], style="Title.TLabel").grid(
			row=0, column=0, sticky="w", pady=(0, 10)
		)
		if page == 0:
			self._build_initial_page()
		elif page == 1:
			self._build_scheme_page()
		else:
			self._build_results_page()

	def _make_canvas(self, parent, figure: Figure) -> FigureCanvasTkAgg:
		canvas = FigureCanvasTkAgg(figure, master=parent)
		widget = canvas.get_tk_widget()
		# Tamaño pedido mínimo: el reparto lo decide la rejilla, no figsize (evita bucles de resize).
		widget.configure(width=1, height=1, background=BACKGROUND, highlightthickness=0)
		widget.pack(fill="both", expand=True)
		return canvas

	@staticmethod
	def _card(parent, title: str) -> tk.Frame:
		card = tk.Frame(parent, background=CARD, highlightbackground="#cfdcd9", highlightthickness=1, padx=14, pady=10)
		tk.Label(card, text=title, background=CARD, foreground="#173b38", font=("Segoe UI", 11, "bold")).grid(
			row=0, column=0, columnspan=4, sticky="w", pady=(0, 6)
		)
		return card

	def _run_in_background(self, message: str, work: Callable[[], None], done: Callable[[], None]) -> None:
		"""Ejecuta `work` en un hilo (solo NumPy, nada de Tk) y luego `done` en el hilo de Tk."""
		self.busy = True
		self.status_var.set(f"⏳ {message}")
		self.root.configure(cursor="watch")
		outcome: dict = {}

		def target() -> None:
			try:
				work()
			except Exception as error:  # noqa: BLE001 - se muestra al usuario
				outcome["error"] = error

		worker = threading.Thread(target=target, daemon=True)
		worker.start()

		def poll() -> None:
			if worker.is_alive():
				self.root.after(120, poll)
				return
			self.busy = False
			self.status_var.set("")
			self.root.configure(cursor="")
			if "error" in outcome:
				messagebox.showerror("No se pudo calcular", str(outcome["error"]), parent=self.root)
				return
			done()

		self.root.after(120, poll)

	# --------------------------------------------------------------------------------------------
	# Page 1: satellite and initial conditions
	# --------------------------------------------------------------------------------------------

	def _axis_index(self) -> int:
		return AXIS_NAMES.index(self.spin_axis)

	def _equilibrium(self) -> Array:
		"""Equilibrio ω*(k) = Ω e_k (giro puro)."""
		omega = np.zeros(3, dtype=float)
		omega[self._axis_index()] = np.deg2rad(self.omega_degrees)
		return omega

	def _initial_omega(self) -> Array:
		"""ω(0) = ω*(k) + δω, con δω = ε |Ω| en las dos componentes transversales.

		Sin δω el giro puro es un punto fijo exacto de las ecuaciones de Euler y ningún
		esquema se separa de él, por lo que nunca se vería la inestabilidad del eje I2.
		"""
		omega = self._equilibrium()
		perturbation = 10.0**self.log_perturbation * abs(np.deg2rad(self.omega_degrees))
		for index in range(3):
			if index != self._axis_index():
				omega[index] = perturbation
		return omega

	def _schedule_initial_redraw(self, _value: str | None = None) -> None:
		if self._redraw_job is not None:
			self.root.after_cancel(self._redraw_job)
		self._redraw_job = self.root.after(80, self._redraw_initial_plots)

	def _redraw_initial_plots(self) -> None:
		self._redraw_job = None
		if self.current_page != 0 or not hasattr(self, "initial_axes"):
			return
		self._draw_satellite(self.initial_axes[0])
		self._draw_invariant_ellipsoids(self.initial_axes[1])
		self._update_spectrum_card()
		self.initial_canvas.draw_idle()

	def _build_initial_page(self) -> None:
		charts = ttk.Frame(self.content)
		charts.grid(row=1, column=0, sticky="nsew")
		# Ejes en posiciones fijas (sin constrained_layout): el tamaño de los gráficos no cambia
		# al mover los sliders y el zoom deja sitio para las etiquetas de los ejes.
		figure = Figure(figsize=(12, 5.0), dpi=95, facecolor=BACKGROUND)
		satellite_axis = figure.add_axes((0.0, 0.0, 0.5, 0.93), projection="3d")
		invariant_axis = figure.add_axes((0.5, 0.0, 0.5, 0.93), projection="3d")
		self.initial_axes = (satellite_axis, invariant_axis)
		self._draw_satellite(satellite_axis)
		self._draw_invariant_ellipsoids(invariant_axis)
		self.initial_canvas = self._make_canvas(charts, figure)

		controls = ttk.Frame(self.content, padding=(0, 8, 0, 0))
		controls.grid(row=2, column=0, sticky="ew")
		controls.columnconfigure((0, 1, 2), weight=1, uniform="controls")
		self.inertia_controls = []
		for index, value in enumerate(self.inertias):
			control = self._slider_control(
				controls,
				f"I{index + 1}  (kg m²)",
				value,
				0.1,
				2.0,
				0.01,
				lambda raw, i=index: self._set_inertia(i, raw),
			)
			control.grid(row=0, column=index, sticky="ew", padx=5)
			self.inertia_controls.append(control)

		condition = ttk.Frame(controls)
		condition.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
		condition.columnconfigure((0, 1), weight=2)
		condition.columnconfigure(2, weight=1)
		self._slider_control(
			condition,
			"Ω inicial  (grados/s, el signo da el sentido de giro)",
			self.omega_degrees,
			-360.0,
			360.0,
			1.0,
			self._set_omega,
		).grid(row=0, column=0, sticky="ew", padx=5)
		self._slider_control(
			condition,
			"Perturbación transversal  log10(ε)   [δω = ε |Ω|]",
			self.log_perturbation,
			-10.0,
			-1.0,
			0.1,
			self._set_perturbation,
			fmt="{:.1f}",
		).grid(row=0, column=1, sticky="ew", padx=5)
		axis_frame = ttk.Frame(condition, padding=(10, 0))
		axis_frame.grid(row=0, column=2, sticky="ew")
		ttk.Label(axis_frame, text="Eje del giro nominal", style="Section.TLabel").pack(anchor="w")
		self.axis_var = tk.StringVar(value=self.spin_axis)
		axis_box = ttk.Combobox(axis_frame, textvariable=self.axis_var, values=AXIS_NAMES, state="readonly")
		axis_box.pack(fill="x", pady=(5, 0))
		axis_box.bind("<<ComboboxSelected>>", self._set_spin_axis)

		self._build_spectrum_card(controls).grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=(10, 0))
		self._update_spectrum_card()

	def _build_spectrum_card(self, parent) -> tk.Frame:
		card = self._card(parent, "Linealización espectral en los equilibrios  ω*(k) = Ω·e_k   (autovalores de exact_Jacobian)")
		card.columnconfigure(2, weight=1)
		header_font = ("Segoe UI", 9, "bold")
		for column, text in enumerate(("Equilibrio", "Eje de giro", "Autovalores λ (s⁻¹)", "Clasificación")):
			tk.Label(card, text=text, background=CARD, foreground="#5b6d6a", font=header_font).grid(
				row=1, column=column, sticky="w", padx=(0, 24)
			)
		self.spectrum_rows = []
		for index in range(3):
			labels = [
				tk.Label(card, background=CARD, font=("Segoe UI", 10), anchor="w")
				for _ in range(4)
			]
			for column, label in enumerate(labels):
				label.grid(row=index + 2, column=column, sticky="ew", padx=(0, 24), ipady=2)
			self.spectrum_rows.append(labels)
		self.flip_var = tk.StringVar()
		tk.Label(card, textvariable=self.flip_var, background=CARD, foreground="#173b38", font=("Segoe UI", 9, "italic")).grid(
			row=5, column=0, columnspan=4, sticky="w", pady=(6, 0)
		)
		return card

	def _update_spectrum_card(self) -> None:
		if not hasattr(self, "spectrum_rows"):
			return
		inertia = np.asarray(self.inertias)
		rate = np.deg2rad(self.omega_degrees)
		axis_names = ("e1 · I mínima", "e2 · I intermedia", "e3 · I máxima")
		for index, labels in enumerate(self.spectrum_rows):
			eigenvalues, unstable = equilibrium_spectrum(inertia, rate, index)
			selected = index == self._axis_index()
			background = "#e6f2ef" if selected else CARD
			texts = (
				f"{'▶ ' if selected else '   '}ω*({index + 1})",
				axis_names[index],
				format_spectrum(eigenvalues),
				"● Silla hiperbólica · inestable" if unstable else "● Centro elíptico · estable (Lyapunov)",
			)
			for column, (label, text) in enumerate(zip(labels, texts)):
				label.configure(
					text=text,
					background=background,
					foreground=(UNSTABLE_COLOR if unstable else STABLE_COLOR) if column == 3 else "#172b2a",
					font=("Segoe UI", 10, "bold" if selected else "normal"),
				)
		eigenvalues, unstable = equilibrium_spectrum(inertia, rate, self._axis_index())
		growth = float(np.max(eigenvalues.real))
		if unstable:
			flip = np.log(1 / 10.0**self.log_perturbation) / growth
			self.flip_var.set(
				f"Perturbación ∝ e^(λt) con λ = {growth:.3f} s⁻¹  →  primer volteo estimado en t ≈ ln(1/ε)/λ = {flip:.2f} s"
			)
		elif abs(rate) > 0:
			self.flip_var.set("Equilibrio estable: la perturbación oscila (polhodia cerrada alrededor de ω*) sin crecer.")
		else:
			self.flip_var.set("Ω = 0: no hay giro y todos los autovalores son nulos.")

	def _slider_control(
		self,
		parent,
		label: str,
		value: float,
		lower: float,
		upper: float,
		resolution: float,
		callback: Callable[[float], None],
		fmt: str | None = None,
	) -> ttk.Frame:
		if fmt is None:
			decimals = max(0, int(np.ceil(-np.log10(resolution))))
			fmt = f"{{:.{decimals}f}}"
		group = ttk.Frame(parent)
		group.columnconfigure(0, weight=1)
		ttk.Label(group, text=label, style="Section.TLabel").grid(row=0, column=0, sticky="w")
		value_var = tk.StringVar(value=fmt.format(value))
		value_entry = ttk.Entry(group, textvariable=value_var, width=8, justify="right")
		value_entry.grid(row=0, column=1, sticky="e")
		scale_var = tk.DoubleVar(value=value)
		scale = tk.Scale(
			group,
			variable=scale_var,
			from_=lower,
			to=upper,
			resolution=resolution,
			orient="horizontal",
			showvalue=False,
			highlightthickness=0,
			troughcolor="#d5e3e0",
			activebackground="#167d73",
			command=lambda raw: self._slider_changed(raw, value_var, callback, fmt),
		)
		scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 0))

		def commit_entry(_event: tk.Event) -> None:
			try:
				value = float(value_var.get().replace(",", "."))
			except ValueError:
				value_var.set(fmt.format(scale_var.get()))
				return
			value = min(upper, max(lower, value))
			scale_var.set(value)
			value_var.set(fmt.format(value))
			callback(value)

		value_entry.bind("<Return>", commit_entry)
		value_entry.bind("<FocusOut>", commit_entry)
		group.scale_var = scale_var  # type: ignore[attr-defined]
		group.value_var = value_var  # type: ignore[attr-defined]
		return group

	@staticmethod
	def _slider_changed(raw: str, value_var: tk.StringVar, callback: Callable[[float], None], fmt: str) -> None:
		value = float(raw)
		value_var.set(fmt.format(value))
		callback(value)

	def _set_inertia(self, index: int, value: float) -> None:
		if self._syncing_inertias:
			return
		gap = 0.01
		values = self.inertias.copy()
		values[index] = value
		if index == 0:
			values[0] = min(values[0], 2.0 - 2 * gap)
			values[1] = max(values[1], values[0] + gap)
			values[2] = max(values[2], values[1] + gap)
			if values[2] > 2.0:
				values[2] = 2.0
				values[1] = min(values[1], values[2] - gap)
				values[0] = min(values[0], values[1] - gap)
		elif index == 1:
			values[1] = min(max(values[1], values[0] + gap), values[2] - gap)
		else:
			values[2] = max(values[2], values[1] + gap)
		values = [min(2.0, max(0.1, item)) for item in values]
		self.inertias = values
		self._syncing_inertias = True
		for control, current in zip(self.inertia_controls, values):
			control.scale_var.set(current)  # type: ignore[attr-defined]
			control.value_var.set(f"{current:.2f}")  # type: ignore[attr-defined]
		self._syncing_inertias = False
		self._schedule_initial_redraw()

	def _set_omega(self, value: float) -> None:
		self.omega_degrees = min(360.0, max(-360.0, value))
		self._schedule_initial_redraw()

	def _set_perturbation(self, value: float) -> None:
		self.log_perturbation = min(-1.0, max(-10.0, value))
		self._schedule_initial_redraw()

	def _set_spin_axis(self, _event: tk.Event | None = None) -> None:
		self.spin_axis = self.axis_var.get()
		self._schedule_initial_redraw()

	@staticmethod
	def _style_3d_axis(axis, labels: tuple[str, str, str], extent: float) -> None:
		axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		axis.set_xlabel(labels[0], labelpad=2)
		axis.set_ylabel(labels[1], labelpad=2)
		axis.set_zlabel(labels[2], labelpad=2)
		axis.locator_params(nbins=5)
		axis.tick_params(labelsize=8, pad=0)
		axis.set_box_aspect((1, 1, 1), zoom=0.86)
		for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
			pane.set_pane_color((0.97, 0.98, 0.98, 1.0))

	def _draw_satellite(self, axis) -> None:
		axis.clear()
		inertia = np.asarray(self.inertias, dtype=float)
		faces, base_colors, lines, clamped = satellite_model(inertia)
		title = "CubeSat 3U (bus con proporciones de la caja de inercias I1, I2, I3)"
		if clamped:
			title += "\nproporciones recortadas para el dibujo" + (" (I3 ≥ I1 + I2: no realizable)" if inertia[2] >= inertia[0] + inertia[1] else "")
		axis.set_title(title, fontsize=10, color="#173b38")
		axis.add_collection3d(Poly3DCollection(
			faces, facecolors=shaded_colors(faces, base_colors), edgecolors=(0.15, 0.15, 0.17, 0.45), linewidths=0.35,
		))
		for segments, color, width in lines:
			axis.plot(*segments.T, color=color, linewidth=width)
		extent = 0.95
		for index, color in enumerate(AXIS_COLORS):
			vector = np.zeros(3)
			vector[index] = extent * 0.97
			axis.plot(*zip(np.zeros(3), vector), color=color, linewidth=1.6, alpha=0.9)
			axis.text(*(vector * 1.04), f"e{index + 1}", color=color, fontsize=10, fontweight="bold")
		spin = np.zeros(3)
		spin[self._axis_index()] = np.sign(self.omega_degrees or 1.0) * extent * 0.8
		axis.quiver(0, 0, 0, *spin, color="#111111", linewidth=1.8, arrow_length_ratio=0.15)
		axis.text(*(spin * 0.62 + 0.07), "Ω", color="#111111", fontsize=11, fontweight="bold")
		self._style_3d_axis(axis, ("Eje 1", "Eje 2", "Eje 3"), extent)
		axis.view_init(elev=24, azim=35)

	def _draw_invariant_ellipsoids(self, axis) -> None:
		axis.clear()
		axis.set_title("Elipsoides 2T y L², separatriz y equilibrios", fontsize=10, color="#173b38")
		inertia = np.asarray(self.inertias)
		omega = self._initial_omega()
		energy = 2 * float(T(*self.inertias, omega))
		momentum = float(L(*self.inertias, omega))
		labels = ("ω1 (rad/s)", "ω2 (rad/s)", "ω3 (rad/s)")
		if energy > 0 and momentum > 0:
			energy_radii = np.sqrt(energy / inertia)
			momentum_radii = momentum / inertia
			axis.plot_surface(*ellipsoid(energy_radii, 36, 24), color="#e09b49", alpha=0.22, linewidth=0, shade=True)
			axis.plot_surface(*ellipsoid(momentum_radii, 36, 24), color="#167d73", alpha=0.20, linewidth=0, shade=True)
			axis.plot([], [], [], color="#e09b49", label="Energía 2T")
			axis.plot([], [], [], color="#167d73", label="Momento L²")
			for index, curve in enumerate(separatrix_curves(inertia, momentum)):
				axis.plot(*curve.T, color="#c34c32", linestyle="--", linewidth=1.1, label="Separatriz" if index == 0 else None)
			for index, color in enumerate(AXIS_COLORS):
				points = np.zeros((2, 3))
				points[:, index] = (momentum / inertia[index], -momentum / inertia[index])
				axis.scatter(*points.T, color=color, s=22, depthshade=False)
			axis.scatter(*omega, color="#111111", s=40, marker="*", depthshade=False, label="ω(0)")
			axis.legend(loc="upper left", fontsize=8)
			extent = 1.1 * max(float(np.max(energy_radii)), float(np.max(momentum_radii)))
		else:
			axis.text2D(0.22, 0.55, "El giro nulo no define elipsoides", transform=axis.transAxes)
			extent = 1.0
		self._style_3d_axis(axis, labels, extent)
		axis.view_init(elev=22, azim=35)

	# --------------------------------------------------------------------------------------------
	# Page 2: numerical schemes
	# --------------------------------------------------------------------------------------------

	def _build_scheme_page(self) -> None:
		controls = ttk.Frame(self.content, padding=(0, 0, 0, 10))
		controls.grid(row=1, column=0, sticky="ew")
		controls.columnconfigure(0, weight=2)
		controls.columnconfigure(1, weight=1)
		controls.columnconfigure(2, weight=1)
		self._slider_control(
			controls, "Paso temporal Δt (s)", self.dt, 0.001, DT_MAX, 0.001, self._set_dt
		).grid(row=0, column=0, sticky="ew", padx=5)
		final_frame = ttk.Frame(controls, padding=5)
		final_frame.grid(row=0, column=1, sticky="ew")
		ttk.Label(final_frame, text="Tiempo final t_f (s)", style="Section.TLabel").pack(anchor="w")
		self.final_time_var = tk.StringVar(value=f"{self.final_time:.2f}")
		final_entry = ttk.Entry(final_frame, textvariable=self.final_time_var)
		final_entry.pack(fill="x", pady=(5, 0))
		final_entry.bind("<KeyRelease>", lambda _event: self._update_scheme_summary())
		scheme_frame = ttk.Frame(controls, padding=5)
		scheme_frame.grid(row=0, column=2, sticky="ew")
		ttk.Label(scheme_frame, text="Esquema de integración", style="Section.TLabel").pack(anchor="w")
		self.scheme_var = tk.StringVar(value=self.scheme_name)
		scheme_box = ttk.Combobox(
			scheme_frame,
			textvariable=self.scheme_var,
			values=tuple(SCHEMES),
			state="readonly",
		)
		scheme_box.pack(fill="x", pady=(5, 0))
		scheme_box.bind("<<ComboboxSelected>>", lambda _event: self._update_scheme_summary())
		ttk.Button(controls, text="Ejecutar", style="Accent.TButton", command=self._execute).grid(
			row=0, column=3, padx=(10, 5), pady=(19, 0), sticky="e"
		)

		summary = self._card(self.content, "Resumen de lo que se va a graficar")
		summary.grid(row=2, column=0, sticky="ew", pady=(0, 8))
		summary.columnconfigure((1, 3), weight=1, uniform="summary")
		self.summary_rows = []
		for index in range(8):
			row, column = index % 4 + 1, 2 * (index // 4)
			bullet = tk.Label(summary, text="•", background=CARD, foreground="#167d73", font=("Segoe UI", 12, "bold"))
			text = tk.Label(summary, background=CARD, foreground="#172b2a", font=("Segoe UI", 10), justify="left", anchor="w", wraplength=540)
			bullet.grid(row=row, column=column, sticky="nw", padx=(0 if column == 0 else 18, 6), pady=2)
			text.grid(row=row, column=column + 1, sticky="nw", pady=2)
			self.summary_rows.append((bullet, text))

		self.content.rowconfigure(1, weight=0)
		self.content.rowconfigure(3, weight=1)
		chart_frame = ttk.Frame(self.content)
		chart_frame.grid(row=3, column=0, sticky="nsew")
		figure = Figure(figsize=(12.0, 4.6), dpi=95, facecolor=BACKGROUND)
		axes = [figure.add_axes((0.05 + 0.33 * index, 0.12, 0.28, 0.78)) for index in range(3)]
		self.stability_figure = figure
		self.stability_axes = axes
		self.stability_canvas = self._make_canvas(chart_frame, figure)
		self._draw_stability_regions()

	def _set_dt(self, value: float) -> None:
		self.dt = min(DT_MAX, max(0.001, value))
		if self.current_page == 1:
			self._draw_stability_regions()

	def _stability_points(self) -> tuple[Array, bool]:
		eigenvalues, unstable = equilibrium_spectrum(
			np.asarray(self.inertias), np.deg2rad(self.omega_degrees), self._axis_index()
		)
		return eigenvalues * self.dt, unstable

	def _draw_stability_regions(self) -> None:
		if not hasattr(self, "stability_axes"):
			return
		points, _ = self._stability_points()
		extent = max(4.0, 1.25 * float(np.max(np.abs(points))))
		x = np.linspace(-1.25 * extent, 0.75 * extent, 220)
		y = np.linspace(-extent, extent, 220)
		xx, yy = np.meshgrid(x, y)
		z = xx + 1j * yy
		for axis, name in zip(self.stability_axes, SCHEMES):
			axis.clear()
			amplification = amplification_factor(name, z)
			stable = np.abs(amplification) <= 1.0
			axis.contourf(xx, yy, stable.astype(float), levels=[0.5, 1.5], colors=["#a9d7cf"], alpha=0.9)
			axis.contour(xx, yy, np.abs(amplification), levels=[1.0], colors=["#203b38"], linewidths=1.1)
			axis.axhline(0, color="#879794", linewidth=0.7)
			axis.axvline(0, color="#879794", linewidth=0.7)
			axis.scatter(points.real, points.imag, color="#c34c32", marker="x", s=48, zorder=3, label="λ·Δt")
			axis.set(title=name, xlabel="Re(z)", ylabel="Im(z)", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect="equal")
			axis.title.set_fontweight("bold" if name == self.scheme_var.get() else "normal")
			axis.grid(alpha=0.16)
			axis.legend(loc="lower left", fontsize=8)
		self.stability_canvas.draw_idle()
		self._update_scheme_summary()

	def _update_scheme_summary(self) -> None:
		if not hasattr(self, "summary_rows"):
			return
		points, unstable = self._stability_points()
		axis_index = self._axis_index()
		eigenvalues = points / self.dt
		scheme = self.scheme_var.get()
		try:
			final_time = float(self.final_time_var.get().replace(",", "."))
			steps = int(np.ceil(final_time / self.dt)) if final_time > 0 else 0
			integration = f"t ∈ [0, {final_time:g}] s con N = {steps} pasos de Δt = {self.dt:.3f} s"
			if steps > MAX_STEPS:
				integration += f"  ⚠ supera el máximo de {MAX_STEPS} pasos"
		except ValueError:
			integration = "t_f no es un número válido"
		dominant = float(np.max(np.abs(points)))
		ratios = "  ·  ".join(
			f"{name}: {float(np.max(np.abs(amplification_factor(name, points)))):.5f}" for name in SCHEMES
		)
		initial = self._initial_omega()
		items = (
			f"Esquema seleccionado: {scheme} (orden teórico {SCHEMES[scheme][1]})",
			f"Equilibrio nominal ω*({axis_index + 1}) = Ω·e{axis_index + 1}, con Ω = {self.omega_degrees:.0f} °/s",
			f"Autovalores λ = {format_spectrum(eigenvalues)} s⁻¹  →  "
			+ ("silla hiperbólica (inestable)" if unstable else "centro elíptico (estable)"),
			f"Puntos z = λ·Δt en el plano complejo: |z| máx = {dominant:.4f}",
			f"Factor de amplificación máx |R(z)|  →  {ratios}",
			"|R| > 1 amplifica la perturbación; en la silla es la inestabilidad física, en un centro es error del esquema",
			integration,
			f"Condición inicial ω(0) = ({initial[0]:.4g}, {initial[1]:.4g}, {initial[2]:.4g}) rad/s",
		)
		for (_bullet, label), text in zip(self.summary_rows, items):
			label.configure(text=text)

	# --------------------------------------------------------------------------------------------
	# Integration
	# --------------------------------------------------------------------------------------------

	def _execute(self) -> None:
		if self.busy:
			return
		try:
			final_time = float(self.final_time_var.get().replace(",", "."))
			if not np.isfinite(final_time) or final_time <= 0:
				raise ValueError("t_f debe ser un número positivo.")
			if self.dt <= 0:
				raise ValueError("Δt debe ser positivo.")
			initial = self._initial_omega()
			if np.linalg.norm(initial) == 0:
				raise ValueError("Selecciona Ω ≠ 0 para integrar el movimiento.")
			steps = max(1, int(np.ceil(final_time / self.dt)))
			if steps > MAX_STEPS:
				raise ValueError(f"La combinación t_f/Δt supera {MAX_STEPS} pasos.")
			if steps < 4:
				raise ValueError("Hacen falta al menos 4 pasos (reduce Δt o aumenta t_f).")
		except ValueError as error:
			messagebox.showerror("No se pudo integrar", str(error), parent=self.root)
			return
		self.scheme_name = self.scheme_var.get()
		self.final_time = final_time
		inertias = np.asarray(self.inertias, dtype=float).copy()
		times = np.linspace(0.0, final_time, steps + 1)
		results = {
			"times": times,
			"initial": initial,
			"inertias": inertias,
			"axis_index": self._axis_index(),
			"omega_degrees": self.omega_degrees,
			"log_perturbation": self.log_perturbation,
			"dt": float(times[1] - times[0]),
			"field": lambda omega, _time: Euler_Equations(*inertias, omega),
			"runs": {},
		}

		def work() -> None:
			# Se integran los tres esquemas para poder conmutar entre ellos y comparar derivas.
			for name in SCHEMES:
				self._run_scheme(results, name)
			if not results["runs"][self.scheme_name].get("failed"):
				self._richardson(results, self.scheme_name)

		def done() -> None:
			failed = results["runs"][self.scheme_name].get("failed")
			if failed:
				messagebox.showerror("No se pudo integrar", failed, parent=self.root)
				return
			self.results = results
			self.show_page(2)

		self._run_in_background("Integrando con Euler, Crank-Nicolson y RK4 y estimando el orden por Richardson…", work, done)

	@staticmethod
	def _run_scheme(results: dict, name: str) -> dict:
		"""Cauchy_Problem con un esquema + invariantes 2T y L² + actitud. Se guarda en caché."""
		runs = results["runs"]
		if name in runs:
			return runs[name]
		scheme, _order = SCHEMES[name]
		times = results["times"]
		inertia = results["inertias"]
		try:
			with np.errstate(all="ignore"):
				solution, _ = Cauchy_Problem(scheme, results["initial"], results["field"], times)
		except (ValueError, np.linalg.LinAlgError) as error:
			runs[name] = {"failed": f"{name}: {error}"}
			return runs[name]
		valid = first_invalid(solution)
		if valid < 2:
			runs[name] = {"failed": f"{name}: la solución diverge desde el primer paso."}
			return runs[name]
		solution, run_times = solution[:valid], times[:valid]
		energy = 2 * np.array([T(*inertia, omega) for omega in solution])
		momentum = np.array([L(*inertia, omega) for omega in solution]) ** 2
		runs[name] = {
			"solution": solution,
			"times": run_times,
			"diverged": valid < len(times),
			"energy_drift": relative_drift(energy),
			"momentum_drift": relative_drift(momentum),
			"rotations": propagate_attitude(solution, run_times),
		}
		return runs[name]

	@staticmethod
	def _richardson(results: dict, name: str) -> dict:
		"""Richardson_Error + estimate_scheme_order de numerical_engine (en caché)."""
		run = results["runs"][name]
		if "richardson" in run:
			return run
		scheme, _order = SCHEMES[name]
		times = results["times"]
		try:
			error, order, log_n, log_e = richardson_with_order(scheme, results["initial"], results["field"], times)
		except (ValueError, np.linalg.LinAlgError, TypeError) as failure:
			error, order, log_n, log_e = np.full((len(times), 3), np.nan), np.nan, np.array([]), np.array([])
			run["richardson_failed"] = str(failure)
		run["richardson"] = error
		run["order_estimate"] = order
		run["convergence"] = (log_n, log_e)
		return run

	# --------------------------------------------------------------------------------------------
	# Page 3: results
	# --------------------------------------------------------------------------------------------

	def _build_results_page(self) -> None:
		if self.results is None:
			ttk.Label(self.content, text="Configura los parámetros y pulsa «Ejecutar» en «Esquemas Numéricos».").grid(
				row=1, column=0, sticky="nw"
			)
			return
		toolbar = ttk.Frame(self.content)
		toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
		ttk.Label(toolbar, text="Esquema visualizado", style="Section.TLabel").pack(side="left")
		self.view_scheme_var = tk.StringVar(value=self.scheme_name)
		selector = ttk.Combobox(
			toolbar, textvariable=self.view_scheme_var, values=tuple(SCHEMES), state="readonly", width=18
		)
		selector.pack(side="left", padx=(8, 16))
		selector.bind("<<ComboboxSelected>>", self._switch_scheme)
		self.pause_button = ttk.Button(toolbar, text="Pausa", command=self._toggle_animation)
		self.pause_button.pack(side="left")
		self.summary_var = tk.StringVar()
		ttk.Label(toolbar, textvariable=self.summary_var, style="Info.TLabel").pack(side="left", padx=16)
		self.content.rowconfigure(1, weight=0)
		self.content.rowconfigure(2, weight=1)
		self.results_frame = ttk.Frame(self.content)
		self.results_frame.grid(row=2, column=0, sticky="nsew")
		self._render_results()

	def _switch_scheme(self, _event: tk.Event | None = None) -> None:
		if self.busy or self.results is None:
			self.view_scheme_var.set(self.scheme_name)
			return
		name = self.view_scheme_var.get()
		run = self._run_scheme(self.results, name)
		if run.get("failed"):
			messagebox.showerror("Esquema no disponible", run["failed"], parent=self.root)
			self.view_scheme_var.set(self.scheme_name)
			return
		results = self.results

		def done() -> None:
			self.scheme_name = name
			if self.current_page == 2:
				self._render_results()

		self._run_in_background(f"Estimando el orden de {name} por Richardson…", lambda: self._richardson(results, name), done)

	def _toggle_animation(self) -> None:
		if self.animation_object is None:
			return
		if self.animation_paused:
			self.animation_object.resume()
			self.pause_button.configure(text="Pausa")
		else:
			self.animation_object.pause()
			self.pause_button.configure(text="Reanudar")
		self.animation_paused = not self.animation_paused

	def _render_results(self) -> None:
		if self.animation_object is not None:
			self.animation_object.event_source.stop()
			self.animation_object = None
		self.animation_paused = False
		self.pause_button.configure(text="Pausa")
		for child in self.results_frame.winfo_children():
			child.destroy()
		assert self.results is not None
		name = self.scheme_name
		run = self.results["runs"][name]
		_, order = SCHEMES[name]
		solution = run["solution"]
		times = run["times"]
		inertia = self.results["inertias"]

		# Dos lienzos: el satélite se repinta en cada fotograma; el análisis solo actualiza
		# por blitting el marcador de fases y los cursores temporales (sin repintar superficies).
		self.results_frame.rowconfigure(0, weight=1)
		self.results_frame.columnconfigure(0, weight=3, uniform="results")
		self.results_frame.columnconfigure(1, weight=7, uniform="results")
		attitude_figure = Figure(figsize=(4.4, 7.6), dpi=92, facecolor=BACKGROUND, constrained_layout=True)
		attitude_grid = attitude_figure.add_gridspec(2, 1, height_ratios=(1.9, 1.0))
		attitude_axis = attitude_figure.add_subplot(attitude_grid[0], projection="3d")
		angle_axis = attitude_figure.add_subplot(attitude_grid[1])
		figure = Figure(figsize=(9.6, 7.6), dpi=92, facecolor=BACKGROUND, constrained_layout=True)
		grid = figure.add_gridspec(2, 4, height_ratios=(1.2, 0.9))
		phase_axis = figure.add_subplot(grid[0, 0:2], projection="3d")
		omega_axis = figure.add_subplot(grid[0, 2:4])
		error_axis = figure.add_subplot(grid[1, 0])
		convergence_axis = figure.add_subplot(grid[1, 1])
		energy_axis = figure.add_subplot(grid[1, 2])
		momentum_axis = figure.add_subplot(grid[1, 3])

		marker = self._draw_phase_portrait(phase_axis, solution, inertia)
		cursors = [self._draw_omega(omega_axis, solution, times)]
		estimate = run["order_estimate"]
		cursors.append(self._draw_richardson(error_axis, run["richardson"][: len(times)], times, estimate))
		self._draw_convergence(convergence_axis, *run["convergence"], estimate, order)
		cursors.extend(self._draw_invariants(energy_axis, momentum_axis, name))

		divergence = "  ·  ¡divergió numéricamente!" if run["diverged"] else ""
		figure.suptitle(
			f"{name} (orden teórico {order}, estimado {estimate:.2f})  |  Δt = {float(self.results['dt']):.4g} s  |  "
			f"Ω = {self.results['omega_degrees']:.0f} °/s  |  ε = 10^{self.results['log_perturbation']:.1f}{divergence}",
			color="#173b38", fontsize=11,
		)
		axis_index = self.results["axis_index"]
		flips = np.count_nonzero(np.diff(np.sign(solution[:, axis_index])) != 0)
		energy_drift = run["energy_drift"]
		momentum_drift = run["momentum_drift"]
		self.summary_var.set(
			f"Cambios de signo de ω{axis_index + 1}: {flips}   ·   "
			f"max ΔT/T0 = {np.nanmax(energy_drift) if np.any(np.isfinite(energy_drift)) else 0:.2e}   ·   "
			f"max ΔL²/L0² = {np.nanmax(momentum_drift) if np.any(np.isfinite(momentum_drift)) else 0:.2e}"
		)
		for frame_column, current in enumerate((attitude_figure, figure)):
			holder = ttk.Frame(self.results_frame)
			holder.grid(row=0, column=frame_column, sticky="nsew")
			holder.pack_propagate(False)
			self._make_canvas(holder, current)
		self._animate_attitude(attitude_axis, angle_axis, run, inertia, [marker, *cursors])

	def _draw_phase_portrait(self, axis, solution: Array, inertia: Array):
		axis.set_title("Espacio de fases (ω1, ω2, ω3) y superficies de Poinsot", fontsize=10)
		initial = np.asarray(self.results["initial"])
		energy = 2 * float(T(*inertia, initial))
		momentum = float(L(*inertia, initial))
		if energy > 0 and momentum > 0:
			energy_radii = np.sqrt(energy / inertia)
			momentum_radii = momentum / inertia
			axis.plot_surface(*ellipsoid(energy_radii), color="#e09b49", alpha=0.10, linewidth=0)
			axis.plot_surface(*ellipsoid(momentum_radii), color="#167d73", alpha=0.11, linewidth=0)
			for index, curve in enumerate(separatrix_curves(inertia, momentum)):
				axis.plot(*curve.T, color="#c34c32", linewidth=1.0, linestyle="--", label="Separatriz crítica" if index == 0 else None)
			for index, color in enumerate(AXIS_COLORS):
				points = np.zeros((2, 3))
				points[:, index] = (momentum / inertia[index], -momentum / inertia[index])
				axis.scatter(*points.T, color=color, s=18, depthshade=False)
			extent = 1.1 * float(np.max(momentum_radii))
			axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		axis.plot(*solution.T, color="#173b38", linewidth=1.2, label="Polhodia ω(t)")
		(marker,) = axis.plot(*solution[0], marker="o", color="#c34c32", markersize=6, linestyle="")
		axis.set_xlabel("ω1 (rad/s)")
		axis.set_ylabel("ω2 (rad/s)")
		axis.set_zlabel("ω3 (rad/s)")
		axis.set_box_aspect((1, 1, 1), zoom=0.88)
		axis.legend(fontsize=7, loc="upper left")
		return marker

	@staticmethod
	def _draw_omega(axis, solution: Array, times: Array):
		for index, color in enumerate(AXIS_COLORS):
			axis.plot(times, solution[:, index], color=color, linewidth=1.3, label=f"ω{index + 1}")
		axis.set(title="Velocidad angular en ejes cuerpo ω(t)", xlabel="Tiempo (s)", ylabel="ω (rad/s)")
		axis.grid(alpha=0.22)
		axis.legend(fontsize=7, loc="upper right", ncol=3)
		return axis.axvline(times[0], color="#111111", linewidth=0.8, alpha=0.6)

	@staticmethod
	def _draw_richardson(axis, error: Array, times: Array, order: float):
		with np.errstate(divide="ignore", invalid="ignore"):
			for index, color in enumerate(AXIS_COLORS):
				component = np.abs(error[:, index])
				axis.semilogy(times, np.where(component > 0, component, np.nan), color=color, linewidth=0.9, alpha=0.75, label=f"|E{index + 1}|")
			total = np.linalg.norm(error, axis=1)
			axis.semilogy(times, np.where(total > 0, total, np.nan), color="#111111", linewidth=1.4, label="‖E‖")
		axis.set(
			title=f"Error de Richardson (q = {order:.2f})\nE = (U_Δt/2 − U_Δt) / (1 − 2^−q)",
			xlabel="Tiempo (s)", ylabel="Error estimado (rad/s)",
		)
		axis.title.set_fontsize(9)
		axis.grid(alpha=0.22, which="both")
		axis.legend(fontsize=7, loc="lower right", ncol=2)
		return axis.axvline(times[0], color="#111111", linewidth=0.8, alpha=0.6)

	@staticmethod
	def _draw_convergence(axis, log_n: Array, log_e: Array, estimate: float, order: int) -> None:
		"""Recta ln E – ln N de estimate_scheme_order (numerical_engine)."""
		mask = np.isfinite(log_n) & np.isfinite(log_e)
		axis.plot(log_n[mask], log_e[mask], "o-", color="#173b38", linewidth=1.3, label="ln‖U_2N − U_N‖∞")
		if np.count_nonzero(mask) >= 2:
			fit = np.polyfit(log_n[mask], log_e[mask], 1)
			axis.plot(log_n[mask], np.polyval(fit, log_n[mask]), "--", color="#c34c32", linewidth=1.0, label=f"ajuste: pendiente {fit[0]:.2f}")
		axis.set(
			title=f"Convergencia (estimate_scheme_order)\norden estimado q = {estimate:.2f}  ·  teórico {order}",
			xlabel="ln(N)", ylabel="ln(E)",
		)
		axis.title.set_fontsize(9)
		axis.grid(alpha=0.22)
		axis.legend(fontsize=7, loc="upper right")

	def _draw_invariants(self, energy_axis, momentum_axis, selected: str) -> list:
		for name in SCHEMES:
			run = self.results["runs"].get(name)
			if run is None or run.get("failed"):
				continue
			width, alpha = (1.8, 1.0) if name == selected else (0.9, 0.55)
			energy_axis.semilogy(run["times"], run["energy_drift"], color=SCHEME_COLORS[name], linewidth=width, alpha=alpha, label=name)
			momentum_axis.semilogy(run["times"], run["momentum_drift"], color=SCHEME_COLORS[name], linewidth=width, alpha=alpha, label=name)
		energy_axis.set(title="Deriva de energía ΔT(t)\n|2T(t) − 2T₀| / 2T₀", xlabel="Tiempo (s)", ylabel="ΔT relativa")
		momentum_axis.set(title="Deriva de momento ΔL(t)\n|L²(t) − L₀²| / L₀²", xlabel="Tiempo (s)", ylabel="ΔL relativa")
		cursors = []
		for axis in (energy_axis, momentum_axis):
			axis.title.set_fontsize(9)
			axis.grid(alpha=0.22, which="both")
			axis.legend(fontsize=7, loc="lower right")
			cursors.append(axis.axvline(0.0, color="#111111", linewidth=0.8, alpha=0.6))
		return cursors

	def _animate_attitude(self, axis, angle_axis, run: dict, inertia: Array, synced: list) -> None:
		solution, times, rotations = run["solution"], run["times"], run["rotations"]
		marker, *cursors = synced
		analysis_figure = marker.axes.figure
		analysis_canvas = analysis_figure.canvas
		background: dict = {}
		for artist in synced:
			artist.set_animated(True)

		def blit_synced() -> None:
			if "image" not in background:
				return
			analysis_canvas.restore_region(background["image"])
			for artist in synced:
				artist.axes.draw_artist(artist)
			analysis_canvas.blit(analysis_figure.bbox)

		def capture_background(_event) -> None:
			background["image"] = analysis_canvas.copy_from_bbox(analysis_figure.bbox)
			blit_synced()

		analysis_canvas.mpl_connect("draw_event", capture_background)
		axis_index = self.results["axis_index"]
		extent = 1.0
		axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		axis.set_box_aspect((1, 1, 1))
		axis.set_xlabel("X inercial")
		axis.set_ylabel("Y inercial")
		axis.set_zlabel("Z inercial")
		axis.view_init(elev=22, azim=35)

		faces, base_colors, lines, _ = satellite_model(inertia)
		body = Poly3DCollection(faces, edgecolors=(0.15, 0.15, 0.17, 0.45), linewidths=0.35)
		axis.add_collection3d(body)
		line_artists = [(segments, axis.plot([], [], [], color=color, linewidth=width)[0]) for segments, color, width in lines]
		axis_lines = [axis.plot([], [], [], color=color, linewidth=1.8, label=f"e{i + 1}")[0] for i, color in enumerate(AXIS_COLORS)]
		(trace,) = axis.plot([], [], [], color=AXIS_COLORS[axis_index], linewidth=0.8, alpha=0.55)
		# L es constante en ejes inerciales: L_in = R(0) I ω(0) con R(0) = Id.
		momentum = inertia * solution[0]
		momentum_dir = momentum / np.linalg.norm(momentum) * extent * 0.95
		axis.plot(*zip(-momentum_dir, momentum_dir), color="#111111", linestyle="--", linewidth=1.0, label="L (inercial, fijo)")
		axis.legend(fontsize=7, loc="upper left")

		stride = max(1, len(times) // 480)
		frame_indices = np.arange(0, len(times), stride)
		if frame_indices[-1] != len(times) - 1:
			frame_indices = np.append(frame_indices, len(times) - 1)
		tips = rotations[:, :, axis_index] * extent * 0.95
		# Ángulo entre el eje nominal de giro (en inercial) y L: 0° ↔ 180° en cada volteo.
		angles = np.degrees(np.arccos(np.clip(rotations[:, :, axis_index] @ (momentum_dir / np.linalg.norm(momentum_dir)), -1, 1)))
		angle_axis.plot(times, angles, color=AXIS_COLORS[axis_index], linewidth=1.4)
		angle_axis.set(
			title=f"Ángulo entre e{axis_index + 1} y L (inercial)", xlabel="Tiempo (s)", ylabel="grados",
			ylim=(-5, 185), yticks=(0, 45, 90, 135, 180),
		)
		angle_axis.title.set_fontsize(9)
		angle_axis.grid(alpha=0.22)
		angle_cursor = angle_axis.axvline(times[0], color="#111111", linewidth=0.8, alpha=0.6)

		def draw_frame(frame: int):
			step = frame_indices[frame]
			rotation = rotations[step]
			rotated = [face @ rotation.T for face in faces]
			body.set_verts(rotated)
			body.set_facecolor(shaded_colors(rotated, base_colors))
			for segments, artist in line_artists:
				moved = segments @ rotation.T
				artist.set_data_3d(moved[:, 0], moved[:, 1], moved[:, 2])
			for index, line in enumerate(axis_lines):
				tip = rotation[:, index] * extent * 0.95
				line.set_data_3d([0, tip[0]], [0, tip[1]], [0, tip[2]])
			trace.set_data_3d(tips[: step + 1, 0], tips[: step + 1, 1], tips[: step + 1, 2])
			marker.set_data_3d([solution[step, 0]], [solution[step, 1]], [solution[step, 2]])
			for cursor in (*cursors, angle_cursor):
				cursor.set_xdata([times[step], times[step]])
			axis.set_title(
				f"Orientación del satélite · t = {times[step]:.2f} s\nángulo(e{axis_index + 1}, L) = {angles[step]:5.1f}°",
				fontsize=10,
			)
			if frame % 2 == 0:
				blit_synced()
			return [body, trace, angle_cursor, *axis_lines]

		draw_frame(0)
		self.animation_object = animation.FuncAnimation(
			axis.figure,
			draw_frame,
			frames=len(frame_indices),
			interval=40,
			blit=False,
			repeat=True,
		)
		analysis_canvas.draw_idle()


def main() -> None:
	root = tk.Tk()
	SatelliteAttitudeApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
