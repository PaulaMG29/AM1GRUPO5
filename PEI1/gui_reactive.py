"""Interactive satellite attitude simulator for PEI 1."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
AXIS_NAMES = ("I1 (mínimo)", "I2 (intermedio)", "I3 (máximo)")


class SatelliteAttitudeApp:
	"""Three-view Tkinter dashboard backed by the PEI1 numerical modules."""

	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("PEI 1 | Dinámica de actitud satelital")
		self.root.geometry("1440x920")
		self.root.minsize(1120, 760)
		self.root.configure(background="#f3f5f4")

		self.inertias = [0.5, 1.0, 1.5]
		self.omega_degrees = 180.0
		self.spin_axis = AXIS_NAMES[1]
		self.dt = 0.02
		self.final_time = 12.0
		self.scheme_name = "Runge-Kutta 4"
		self.results: dict[str, Array | float | str] | None = None
		self.menu_open = True
		self.current_page = 0
		self.animation_object: animation.FuncAnimation | None = None
		self._redraw_job: str | None = None
		self._syncing_inertias = False

		style = ttk.Style()
		style.theme_use("clam")
		style.configure("TFrame", background="#f3f5f4")
		style.configure("TLabel", background="#f3f5f4", foreground="#172b2a")
		style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#173b38")
		style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#245b55")
		style.configure("TButton", padding=(10, 7))
		style.configure("Accent.TButton", background="#167d73", foreground="white")

		self.shell = ttk.Frame(root)
		self.shell.pack(fill="both", expand=True)
		self.shell.rowconfigure(0, weight=1)
		self.shell.columnconfigure(1, weight=1)
		self.sidebar = ttk.Frame(self.shell, padding=(12, 14), width=220)
		self.sidebar.grid(row=0, column=0, sticky="ns")
		self.sidebar.grid_propagate(False)
		self.content = ttk.Frame(self.shell, padding=(18, 14, 18, 16))
		self.content.grid(row=0, column=1, sticky="nsew")
		self.content.rowconfigure(1, weight=1)
		self.content.columnconfigure(0, weight=1)
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
		self.current_page = page
		self._clear_content()
		self.content.rowconfigure(1, weight=1)
		self.content.rowconfigure(2, weight=0)
		titles = (
			"Satélite y Condiciones Iniciales",
			"Esquemas Numéricos",
			"Integración y Error",
		)
		ttk.Label(self.content, text=titles[page], style="Title.TLabel").grid(
			row=0, column=0, sticky="w", pady=(0, 12)
		)
		if page == 0:
			self._build_initial_page()
		elif page == 1:
			self._build_scheme_page()
		else:
			self._build_results_page()

	def _make_canvas(self, parent: ttk.Frame, figure: Figure) -> FigureCanvasTkAgg:
		canvas = FigureCanvasTkAgg(figure, master=parent)
		canvas.get_tk_widget().pack(fill="both", expand=True)
		return canvas

	def _initial_omega(self) -> Array:
		omega = np.zeros(3, dtype=float)
		omega[AXIS_NAMES.index(self.spin_axis)] = np.deg2rad(self.omega_degrees)
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
		self.initial_canvas.draw_idle()

	def _build_initial_page(self) -> None:
		charts = ttk.Frame(self.content)
		charts.grid(row=1, column=0, sticky="nsew")
		charts.rowconfigure(0, weight=1)
		charts.columnconfigure((0, 1), weight=1, uniform="initial-charts")
		figure = Figure(figsize=(12, 5.2), dpi=95, facecolor="#f3f5f4", constrained_layout=True)
		satellite_axis = figure.add_subplot(1, 2, 1, projection="3d")
		invariant_axis = figure.add_subplot(1, 2, 2, projection="3d")
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
		condition.columnconfigure(0, weight=2)
		condition.columnconfigure(1, weight=1)
		self._slider_control(
			condition,
			"Ω inicial  (grados/s)",
			self.omega_degrees,
			0.0,
			360.0,
			1.0,
			self._set_omega,
		).grid(row=0, column=0, sticky="ew", padx=5)
		axis_frame = ttk.Frame(condition, padding=(10, 0))
		axis_frame.grid(row=0, column=1, sticky="ew")
		ttk.Label(axis_frame, text="Eje del giro puro", style="Section.TLabel").pack(anchor="w")
		self.axis_var = tk.StringVar(value=self.spin_axis)
		axis_box = ttk.Combobox(axis_frame, textvariable=self.axis_var, values=AXIS_NAMES, state="readonly")
		axis_box.pack(fill="x", pady=(5, 0))
		axis_box.bind("<<ComboboxSelected>>", self._set_spin_axis)

	def _slider_control(
		self,
		parent: ttk.Frame,
		label: str,
		value: float,
		lower: float,
		upper: float,
		resolution: float,
		callback: Callable[[float], None],
	) -> ttk.Frame:
		group = ttk.Frame(parent)
		group.columnconfigure(0, weight=1)
		ttk.Label(group, text=label, style="Section.TLabel").grid(row=0, column=0, sticky="w")
		value_var = tk.StringVar(value=f"{value:.2f}")
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
			command=lambda raw: self._slider_changed(raw, value_var, callback),
		)
		scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 0))

		def commit_entry(_event: tk.Event) -> None:
			try:
				value = float(value_var.get().replace(",", "."))
			except ValueError:
				value_var.set(f"{scale_var.get():.2f}")
				return
			value = min(upper, max(lower, value))
			scale_var.set(value)
			value_var.set(f"{value:.2f}")
			callback(value)

		value_entry.bind("<Return>", commit_entry)
		value_entry.bind("<FocusOut>", commit_entry)
		group.scale_var = scale_var  # type: ignore[attr-defined]
		group.value_var = value_var  # type: ignore[attr-defined]
		return group

	@staticmethod
	def _slider_changed(raw: str, value_var: tk.StringVar, callback: Callable[[float], None]) -> None:
		value = float(raw)
		value_var.set(f"{value:.2f}")
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
		self.omega_degrees = min(360.0, max(0.0, value))
		self._schedule_initial_redraw()

	def _set_spin_axis(self, _event: tk.Event | None = None) -> None:
		self.spin_axis = self.axis_var.get()
		self._schedule_initial_redraw()

	def _draw_satellite(self, axis) -> None:
		axis.clear()
		axis.set_title("Satélite y ejes principales", fontsize=11, color="#173b38")
		dimensions = np.sqrt(np.asarray(self.inertias, dtype=float))
		dimensions = 0.7 + 0.8 * dimensions / dimensions.max()
		x, y, z = dimensions / 2
		vertices = np.array([
			[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
			[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
		])
		edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
		for first, second in edges:
			axis.plot(*zip(vertices[first], vertices[second]), color="#23786f", linewidth=2)
		colors = ("#d95d39", "#167d73", "#3767a5")
		for index, color in enumerate(colors):
			vector = np.zeros(3)
			vector[index] = dimensions[index] * 0.9
			axis.quiver(0, 0, 0, *vector, color=color, linewidth=1.8, arrow_length_ratio=0.12)
			label_position = vector * 1.12
			axis.text(*label_position, f"I{index + 1}", color=color, fontsize=10)
		extent = max(dimensions) * 0.85
		axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		axis.set_xlabel("Eje 1")
		axis.set_ylabel("Eje 2")
		axis.set_zlabel("Eje 3")
		axis.set_box_aspect((1, 1, 1))
		axis.view_init(elev=24, azim=35)

	def _draw_invariant_ellipsoids(self, axis) -> None:
		axis.clear()
		axis.set_title("Elipsoides de energía y momento", fontsize=11, color="#173b38")
		inertia = np.asarray(self.inertias)
		omega = self._initial_omega()
		energy = 2 * float(T(*self.inertias, omega))
		momentum = float(L(*self.inertias, omega))
		if energy > 0 and momentum > 0:
			u = np.linspace(0, 2 * np.pi, 36)
			v = np.linspace(0, np.pi, 24)
			unit_x = np.outer(np.cos(u), np.sin(v))
			unit_y = np.outer(np.sin(u), np.sin(v))
			unit_z = np.outer(np.ones_like(u), np.cos(v))
			energy_radii = np.sqrt(energy / inertia)
			momentum_radii = momentum / inertia
			axis.plot_surface(
				unit_x * energy_radii[0], unit_y * energy_radii[1], unit_z * energy_radii[2],
				color="#e09b49", alpha=0.22, linewidth=0, shade=True,
			)
			axis.plot_surface(
				unit_x * momentum_radii[0], unit_y * momentum_radii[1], unit_z * momentum_radii[2],
				color="#167d73", alpha=0.25, linewidth=0, shade=True,
			)
			axis.plot([], [], [], color="#e09b49", label="Energía 2T")
			axis.plot([], [], [], color="#167d73", label="Momento |L|")
			axis.legend(loc="upper left", fontsize=8)
			extent = 1.15 * max(float(np.max(energy_radii)), float(np.max(momentum_radii)))
			for index, color in enumerate(("#d95d39", "#167d73", "#3767a5")):
				vector = np.zeros(3)
				vector[index] = extent * 0.78
				axis.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.1)
				axis.text(*(vector * 1.08), f"ω{index + 1}", color=color, fontsize=9)
			axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		else:
			axis.text2D(0.22, 0.55, "El giro nulo no define elipsoides", transform=axis.transAxes)
			axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
		axis.set_xlabel("ω1 (rad/s)")
		axis.set_ylabel("ω2 (rad/s)")
		axis.set_zlabel("ω3 (rad/s)")
		axis.set_box_aspect((1, 1, 1))
		axis.view_init(elev=22, azim=35)

	def _build_scheme_page(self) -> None:
		controls = ttk.Frame(self.content, padding=(0, 0, 0, 10))
		controls.grid(row=1, column=0, sticky="ew")
		controls.columnconfigure(0, weight=2)
		controls.columnconfigure(1, weight=1)
		controls.columnconfigure(2, weight=1)
		self._slider_control(
			controls, "Paso temporal Δt (s)", self.dt, 0.001, 0.2, 0.001, self._set_dt
		).grid(row=0, column=0, sticky="ew", padx=5)
		final_frame = ttk.Frame(controls, padding=5)
		final_frame.grid(row=0, column=1, sticky="ew")
		ttk.Label(final_frame, text="Tiempo final t_f (s)", style="Section.TLabel").pack(anchor="w")
		self.final_time_var = tk.StringVar(value=f"{self.final_time:.2f}")
		ttk.Entry(final_frame, textvariable=self.final_time_var).pack(fill="x", pady=(5, 0))
		scheme_frame = ttk.Frame(controls, padding=5)
		scheme_frame.grid(row=0, column=2, sticky="ew")
		ttk.Label(scheme_frame, text="Esquema de integración", style="Section.TLabel").pack(anchor="w")
		self.scheme_var = tk.StringVar(value=self.scheme_name)
		ttk.Combobox(
			scheme_frame,
			textvariable=self.scheme_var,
			values=tuple(SCHEMES),
			state="readonly",
		).pack(fill="x", pady=(5, 0))
		ttk.Button(controls, text="Ejecutar", style="Accent.TButton", command=self._execute).grid(
			row=0, column=3, padx=(10, 5), pady=(19, 0), sticky="e"
		)

		figure = Figure(figsize=(12.2, 6.4), dpi=95, facecolor="#f3f5f4", constrained_layout=True)
		axes = [figure.add_subplot(1, 3, index + 1) for index in range(3)]
		self.stability_figure = figure
		self.stability_axes = axes
		self.content.rowconfigure(1, weight=0)
		self.content.rowconfigure(2, weight=1)
		chart_frame = ttk.Frame(self.content)
		chart_frame.grid(row=2, column=0, sticky="nsew")
		self.stability_canvas = self._make_canvas(chart_frame, figure)
		self._draw_stability_regions()

	def _set_dt(self, value: float) -> None:
		self.dt = min(0.2, max(0.001, value))
		if self.current_page == 1:
			self._draw_stability_regions()

	def _draw_stability_regions(self) -> None:
		if not hasattr(self, "stability_axes"):
			return
		x = np.linspace(-5.0, 3.0, 180)
		y = np.linspace(-4.0, 4.0, 180)
		xx, yy = np.meshgrid(x, y)
		z = xx + 1j * yy
		amplifications = (
			1 + z,
			(1 + z / 2) / (1 - z / 2),
			1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24,
		)
		jacobian = exact_Jacobian(*self.inertias, self._initial_omega())
		eigenvalues = np.linalg.eigvals(jacobian) * self.dt
		for axis, name, amplification in zip(self.stability_axes, SCHEMES, amplifications):
			axis.clear()
			stable = np.abs(amplification) <= 1.0
			axis.contourf(xx, yy, stable.astype(float), levels=[0.5, 1.5], colors=["#a9d7cf"], alpha=0.9)
			axis.contour(xx, yy, np.abs(amplification), levels=[1.0], colors=["#203b38"], linewidths=1.1)
			axis.axhline(0, color="#879794", linewidth=0.7)
			axis.axvline(0, color="#879794", linewidth=0.7)
			axis.scatter(eigenvalues.real, eigenvalues.imag, color="#c34c32", marker="x", s=48, label="λ(exact_Jacobian) Δt")
			axis.set(title=name, xlabel="Re(z)", ylabel="Im(z)", xlim=(-5, 3), ylim=(-4, 4), aspect="equal")
			axis.grid(alpha=0.16)
			axis.legend(loc="lower left", fontsize=7)
		self.stability_canvas.draw_idle()

	def _execute(self) -> None:
		try:
			final_time = float(self.final_time_var.get().replace(",", "."))
			if not np.isfinite(final_time) or final_time <= 0:
				raise ValueError("t_f debe ser un número positivo.")
			if self.dt <= 0:
				raise ValueError("Δt debe ser positivo.")
			self.scheme_name = self.scheme_var.get()
			scheme, order = SCHEMES[self.scheme_name]
			initial = self._initial_omega()
			if np.linalg.norm(initial) == 0:
				raise ValueError("Selecciona Ω > 0 para integrar el movimiento.")
			steps = max(1, int(np.ceil(final_time / self.dt)))
			if steps > 20000:
				raise ValueError("La combinación t_f/Δt supera 20 000 pasos.")
			times = np.linspace(0.0, final_time, steps + 1)
			field = lambda omega, _time: Euler_Equations(*self.inertias, omega)
			solution, times = Cauchy_Problem(scheme, initial, field, times)
			error = Richardson_Error(scheme, order, initial, field, times)
			self.final_time = final_time
			self.results = {
				"solution": solution,
				"times": times,
				"error": error,
				"initial": initial,
				"inertias": np.asarray(self.inertias, dtype=float).copy(),
				"scheme": self.scheme_name,
				"dt": float(times[1] - times[0]),
			}
			self.show_page(2)
		except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
			messagebox.showerror("No se pudo integrar", str(error), parent=self.root)

	def _build_results_page(self) -> None:
		if self.results is None:
			ttk.Label(self.content, text="Configura los parámetros y pulsa «Ejecutar».").grid(
				row=1, column=0, sticky="nw"
			)
			return
		solution = np.asarray(self.results["solution"])
		times = np.asarray(self.results["times"])
		error = np.asarray(self.results["error"])
		inertia = np.asarray(self.results["inertias"])
		figure = Figure(figsize=(13.0, 7.8), dpi=95, facecolor="#f3f5f4", constrained_layout=True)
		grid = figure.add_gridspec(2, 6, height_ratios=(1.15, 0.85))
		phase_axis = figure.add_subplot(grid[0, 0:2], projection="3d")
		attitude_axis = figure.add_subplot(grid[0, 2:4], projection="3d")
		error_axis = figure.add_subplot(grid[0, 4:6])
		energy_axis = figure.add_subplot(grid[1, 0:3])
		momentum_axis = figure.add_subplot(grid[1, 3:6])
		self._draw_phase_portrait(phase_axis, solution, times, inertia)
		self._draw_invariants(energy_axis, momentum_axis, solution, times, inertia)
		error_norm = np.linalg.norm(error, axis=1)
		error_axis.plot(times, error_norm, color="#c34c32", linewidth=1.6)
		error_axis.set(title="Error de Richardson", xlabel="Tiempo (s)", ylabel="||e_R|| (rad/s)")
		error_axis.grid(alpha=0.22)
		figure.suptitle(f"{self.results['scheme']}  |  Δt = {float(self.results['dt']):.4g} s", color="#173b38", fontsize=13)
		chart_frame = ttk.Frame(self.content)
		chart_frame.grid(row=1, column=0, sticky="nsew")
		self._make_canvas(chart_frame, figure)
		self._animate_attitude(attitude_axis, solution, times, inertia)

	def _draw_phase_portrait(self, axis, solution: Array, times: Array, inertia: Array) -> None:
		axis.set_title("Espacio de fases y superficies de Poinsot", fontsize=10)
		initial = np.asarray(self.results["initial"])
		energy = 2 * float(T(*inertia, initial))
		momentum = float(L(*inertia, initial))
		if energy > 0 and momentum > 0:
			u = np.linspace(0, 2 * np.pi, 32)
			v = np.linspace(0, np.pi, 20)
			unit_x = np.outer(np.cos(u), np.sin(v))
			unit_y = np.outer(np.sin(u), np.sin(v))
			unit_z = np.outer(np.ones_like(u), np.cos(v))
			energy_radii = np.sqrt(energy / inertia)
			momentum_radii = momentum / inertia
			axis.plot_surface(unit_x * energy_radii[0], unit_y * energy_radii[1], unit_z * energy_radii[2], color="#e09b49", alpha=0.12, linewidth=0)
			axis.plot_surface(unit_x * momentum_radii[0], unit_y * momentum_radii[1], unit_z * momentum_radii[2], color="#167d73", alpha=0.13, linewidth=0)
			separatrix = self._separatrix(inertia, momentum, times, float(self.results["dt"]))
			if separatrix is not None:
				axis.plot(*separatrix.T, color="#c34c32", linewidth=1.0, linestyle="--", label="Separatriz crítica")
		axis.plot(*solution.T, color="#173b38", linewidth=1.3, label="Cauchy problem")
		axis.set_xlabel("ω1 (rad/s)")
		axis.set_ylabel("ω2 (rad/s)")
		axis.set_zlabel("ω3 (rad/s)")
		axis.set_box_aspect((1, 1, 1))
		axis.legend(fontsize=7, loc="upper left")

	def _separatrix(self, inertia: Array, momentum: float, times: Array, dt: float) -> Array | None:
		if momentum <= 1e-12:
			return None
		i1, i2, i3 = inertia
		omega1 = max(momentum / i1 * 1e-3, 1e-7)
		omega3 = np.sqrt(i1 * (i2 - i1) / (i3 * (i3 - i2))) * omega1
		omega2_squared = (momentum**2 - (i1 * omega1) ** 2 - (i3 * omega3) ** 2) / i2**2
		if omega2_squared <= 0:
			return None
		seed = np.array([omega1, np.sqrt(omega2_squared), omega3], dtype=float)
		horizon = max(float(times[-1]), 1.0) * 1.5
		count = min(2400, max(400, int(np.ceil(horizon / max(dt, 1e-4)))))
		sep_times = np.linspace(0.0, horizon, count + 1)
		field = lambda omega, _time: Euler_Equations(*inertia, omega)
		try:
			curve, _ = Cauchy_Problem(RK4, seed, field, sep_times)
			return curve
		except (ValueError, FloatingPointError):
			return None

	def _draw_invariants(
		self,
		energy_axis,
		momentum_axis,
		solution: Array,
		times: Array,
		inertia: Array,
	) -> None:
		energies = np.array([T(*inertia, omega) for omega in solution])
		momenta = np.array([L(*inertia, omega) for omega in solution])
		energy_axis.plot(times, energies, color="#d28a39", linewidth=1.4)
		energy_axis.set(title="Energía cinética T(t)", xlabel="Tiempo (s)", ylabel="T (J)")
		energy_axis.grid(alpha=0.22)
		momentum_axis.plot(times, momenta, color="#167d73", linewidth=1.4)
		momentum_axis.set(title="Módulo del momento |L(t)|", xlabel="Tiempo (s)", ylabel="|L| (kg m²/s)")
		momentum_axis.grid(alpha=0.22)

	@staticmethod
	def _rotation_increment(rotation_vector: Array) -> Array:
		angle = float(np.linalg.norm(rotation_vector))
		if angle < 1e-12:
			return np.eye(3)
		x, y, z = rotation_vector / angle
		skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
		return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

	def _animate_attitude(self, axis, solution: Array, times: Array, inertia: Array) -> None:
		axis.set_title("Orientación del satélite", fontsize=10)
		axis.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), zlim=(-1.8, 1.8))
		axis.set_box_aspect((1, 1, 1))
		axis.set_xlabel("X")
		axis.set_ylabel("Y")
		axis.set_zlabel("Z")
		axis.view_init(elev=24, azim=35)
		dimensions = np.sqrt(inertia)
		dimensions = 0.7 + 0.7 * dimensions / dimensions.max()
		half = dimensions / 2
		vertices = np.array([
			[-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
			[half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
			[-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
			[half[0], half[1], half[2]], [-half[0], half[1], half[2]],
		])
		edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
		stride = max(1, len(times) // 360)
		frame_indices = np.arange(0, len(times), stride)
		if frame_indices[-1] != len(times) - 1:
			frame_indices = np.append(frame_indices, len(times) - 1)
		rotations = []
		rotation = np.eye(3)
		frame_set = set(frame_indices.tolist())
		for index in range(len(times)):
			if index in frame_set:
				rotations.append(rotation.copy())
			if index + 1 < len(times):
				rotation = rotation @ self._rotation_increment(solution[index] * (times[index + 1] - times[index]))

		def draw_frame(frame: int):
			for artist in list(axis.collections) + list(axis.lines):
				artist.remove()
			rotated = vertices @ rotations[frame].T
			for first, second in edges:
				axis.plot(*zip(rotated[first], rotated[second]), color="#167d73", linewidth=2)
			for index, color in enumerate(("#d95d39", "#167d73", "#3767a5")):
				vector = rotations[frame] @ np.eye(3)[index] * 1.15
				axis.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.12)
			axis.set_title(f"Orientación del satélite · t = {times[frame_indices[frame]]:.2f} s", fontsize=10)
			return axis.lines

		draw_frame(0)
		self.animation_object = animation.FuncAnimation(
			axis.figure,
			draw_frame,
			frames=len(frame_indices),
			interval=45,
			blit=False,
			repeat=True,
		)
		axis.figure.canvas.draw()


def main() -> None:
	root = tk.Tk()
	SatelliteAttitudeApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
