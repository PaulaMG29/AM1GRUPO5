"""Ventana 3: «Integración y Error» (actitud animada, fases, ω(t), Richardson, convergencia e invariantes)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from space_physics.angular_momentum import L
from space_physics.kinetic_energy import T
from space_physics.separatrix import separatrix

from .config import AXIS_COLORS, BACKGROUND, SCHEME_COLORS, SCHEMES, Array
from .helpers import ellipsoid, satellite_model, shaded_colors
from .simulation import richardson, run_scheme
from .widgets import make_canvas


class IntegrationErrorWindow:
	"""Métodos de la ventana 3; se combinan en SatelliteAttitudeApp (app.py), que guarda el estado."""

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
		run = run_scheme(self.results, name)
		if run.get("failed"):
			messagebox.showerror("Esquema no disponible", run["failed"], parent=self.root)
			self.view_scheme_var.set(self.scheme_name)
			return
		results = self.results

		def done() -> None:
			self.scheme_name = name
			if self.current_page == 2:
				self._render_results()

		self._run_in_background(f"Estimando el orden de {name} por Richardson…", lambda: richardson(results, name), done)

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
			make_canvas(holder, current)
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
			for index, curve in enumerate(separatrix(*inertia, momentum)):
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
	def _draw_convergence(axis, log_n: Array, log_e: Array, used: Array, estimate: float, order: int) -> None:
		"""Puntos ln E – ln N de estimate_scheme_order, separando los que usó en su ajuste."""
		finite = np.isfinite(log_n) & np.isfinite(log_e)
		used = used.astype(bool) & finite
		axis.plot(log_n[finite], log_e[finite], "-", color="#173b38", linewidth=1.0, alpha=0.6)
		axis.plot(log_n[used], log_e[used], "o", color="#173b38", markersize=6, label="usados en el ajuste")
		discarded = finite & ~used
		if np.any(discarded):
			axis.plot(log_n[discarded], log_e[discarded], "o", markerfacecolor="white", markeredgecolor="#173b38", markersize=6, label="descartados")
		if np.count_nonzero(used) >= 2 and np.isfinite(estimate):
			# Recta de pendiente −q por los puntos usados (solo para dibujarla).
			intercept = float(np.mean(log_e[used] + estimate * log_n[used]))
			axis.plot(log_n[finite], intercept - estimate * log_n[finite], "--", color="#c34c32", linewidth=1.0, label=f"pendiente −q = {-estimate:.2f}")
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
