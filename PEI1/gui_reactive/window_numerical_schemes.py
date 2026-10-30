"""Ventana 2: «Esquemas Numéricos» (Δt, t_f, esquema, regiones de estabilidad) y botón «Ejecutar»."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.figure import Figure

from numerical_engine.linear_solvers import partition
from space_physics.equilibrium_espectrum import equilibrium_spectrum

from .config import BACKGROUND, CARD, DT_MAX, MAX_STEPS, SCHEMES, Array
from .helpers import amplification_at, format_spectrum, stability_map
from .simulation import richardson, run_scheme
from .widgets import make_canvas, make_card, slider_control


class NumericalSchemesWindow:
	"""Métodos de la ventana 2; se combinan en SatelliteAttitudeApp (app.py), que guarda el estado."""

	# --------------------------------------------------------------------------------------------
	# Page 2: numerical schemes
	# --------------------------------------------------------------------------------------------

	def _build_scheme_page(self) -> None:
		controls = ttk.Frame(self.content, padding=(0, 0, 0, 10))
		controls.grid(row=1, column=0, sticky="ew")
		controls.columnconfigure(0, weight=2)
		controls.columnconfigure(1, weight=1)
		controls.columnconfigure(2, weight=1)
		slider_control(
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

		summary = make_card(self.content, "Resumen de lo que se va a graficar")
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
		# Un panel por esquema de SCHEMES (Euler, Euler inverso, Crank-Nicolson y RK4).
		width = 0.92 / len(SCHEMES)
		axes = [figure.add_axes((0.05 + width * index, 0.12, 0.78 * width, 0.78)) for index in range(len(SCHEMES))]
		self.stability_figure = figure
		self.stability_axes = axes
		self.stability_canvas = make_canvas(chart_frame, figure)
		self._draw_stability_regions()

	def _set_dt(self, value: float) -> None:
		self.dt = min(DT_MAX, max(0.001, value))
		if self.current_page == 1:
			self._draw_stability_regions()

	def _stability_points(self) -> tuple[Array, bool]:
		eigenvalues, unstable = equilibrium_spectrum(*self.inertias, np.deg2rad(self.omega_degrees), self._axis_index())
		return eigenvalues * self.dt, unstable

	def _draw_stability_regions(self) -> None:
		if not hasattr(self, "stability_axes"):
			return
		points, _ = self._stability_points()
		# Extensión redondeada a 0.5 para reutilizar las mallas en caché al mover Δt.
		extent = max(4.0, float(np.ceil(2.5 * float(np.max(np.abs(points))))) / 2)
		for axis, name in zip(self.stability_axes, SCHEMES):
			axis.clear()
			xx, yy, modulus = stability_map(name, extent)
			axis.contourf(xx, yy, (modulus <= 1.0).astype(float), levels=[0.5, 1.5], colors=["#a9d7cf"], alpha=0.9)
			axis.contour(xx, yy, modulus, levels=[1.0], colors=["#203b38"], linewidths=1.1)
			axis.axhline(0, color="#879794", linewidth=0.7)
			axis.axvline(0, color="#879794", linewidth=0.7)
			axis.scatter(points.real, points.imag, color="#c34c32", marker="x", s=48, zorder=3, label="λ·Δt")
			axis.set(title=name, xlabel="Re(z)", ylabel="Im(z)", xlim=(xx[0, 0], xx[0, -1]), ylim=(yy[0, 0], yy[-1, 0]), aspect="equal")
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
			f"{name}: {float(np.max(amplification_at(name, points))):.5f}" for name in SCHEMES
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
		times = partition(0.0, final_time, steps)
		results = {
			"times": times,
			"initial": initial,
			"inertias": inertias,
			"axis_index": self._axis_index(),
			"omega_degrees": self.omega_degrees,
			"log_perturbation": self.log_perturbation,
			"dt": float(times[1] - times[0]),
			"runs": {},
		}

		def work() -> None:
			# Se integran todos los esquemas para poder conmutar entre ellos y comparar derivas.
			for name in SCHEMES:
				run_scheme(results, name)
			if not results["runs"][self.scheme_name].get("failed"):
				richardson(results, self.scheme_name)

		def done() -> None:
			failed = results["runs"][self.scheme_name].get("failed")
			if failed:
				messagebox.showerror("No se pudo integrar", failed, parent=self.root)
				return
			self.results = results
			self.show_page(2)

		self._run_in_background(f"Integrando con {', '.join(SCHEMES)} y estimando el orden por Richardson…", work, done)
