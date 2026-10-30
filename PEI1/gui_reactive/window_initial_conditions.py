"""Ventana 1: «Satélite y Condiciones Iniciales» (inercias, giro nominal, perturbación y espectro)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from space_physics.angular_momentum import L
from space_physics.equilibrium_espectrum import equilibrium_spectrum
from space_physics.kinetic_energy import T
from space_physics.separatrix import separatrix

from .config import AXIS_COLORS, AXIS_NAMES, BACKGROUND, CARD, STABLE_COLOR, UNSTABLE_COLOR, Array
from .helpers import ellipsoid, format_spectrum, satellite_model, shaded_colors
from .widgets import make_canvas, make_card, slider_control


class InitialConditionsWindow:
	"""Métodos de la ventana 1; se combinan en SatelliteAttitudeApp (app.py), que guarda el estado."""

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
		self.initial_canvas = make_canvas(charts, figure)

		controls = ttk.Frame(self.content, padding=(0, 8, 0, 0))
		controls.grid(row=2, column=0, sticky="ew")
		controls.columnconfigure((0, 1, 2), weight=1, uniform="controls")
		self.inertia_controls = []
		for index, value in enumerate(self.inertias):
			control = slider_control(
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
		slider_control(
			condition,
			"Ω inicial  (grados/s, el signo da el sentido de giro)",
			self.omega_degrees,
			-360.0,
			360.0,
			1.0,
			self._set_omega,
		).grid(row=0, column=0, sticky="ew", padx=5)
		slider_control(
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
		card = make_card(parent, "Linealización espectral en los equilibrios  ω*(k) = Ω·e_k   (autovalores de exact_Jacobian)")
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
			eigenvalues, unstable = equilibrium_spectrum(*inertia, rate, index)
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
		eigenvalues, unstable = equilibrium_spectrum(*inertia, rate, self._axis_index())
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
			for index, curve in enumerate(separatrix(*inertia, momentum)):
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
