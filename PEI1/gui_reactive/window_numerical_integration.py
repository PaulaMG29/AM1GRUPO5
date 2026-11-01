"""Ventana 3: «Integración numérica» (actitud animada, ángulo con L, espacio de fases y ω(t))."""

from __future__ import annotations

from tkinter import ttk

import numpy as np
from matplotlib import animation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from space_physics.angular_momentum import L
from space_physics.kinetic_energy import T
from space_physics.separatrix import separatrix

from .config import AXIS_COLORS, BACKGROUND, Array
from .helpers import ellipsoid, satellite_model, shaded_colors
from .widgets import make_canvas

POLHODE_MARKER = "#8a8f94"


class NumericalIntegrationWindow:
	"""Métodos de la ventana 3; se combinan en SatelliteAttitudeApp (app.py), que guarda el estado."""

	# --------------------------------------------------------------------------------------------
	# Page 3: numerical integration
	# --------------------------------------------------------------------------------------------

	def _build_results_page(self) -> None:
		if self._build_results_toolbar(with_pause=True) is None:
			return
		self._render_results()

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
		solution = run["solution"]
		times = run["times"]
		inertia = self.results["inertias"]

		# Dos lienzos del mismo tamaño y con las mismas filas:
		#   izquierda  → orientación del satélite (con su estado en viñetas) y ángulo entre e_k y L,
		#   derecha    → espacio de fases (polhodia) y ω(t).
		# El satélite se repinta en cada fotograma; a la derecha solo se actualizan por blitting
		# el marcador de la polhodia y el cursor temporal (sin repintar superficies).
		self.results_frame.rowconfigure(0, weight=1)
		self.results_frame.columnconfigure((0, 1), weight=1, uniform="results")
		height_ratios = (1.75, 1.0)
		attitude_figure = Figure(figsize=(6.4, 7.6), dpi=92, facecolor=BACKGROUND, constrained_layout=True)
		attitude_grid = attitude_figure.add_gridspec(2, 2, height_ratios=height_ratios, width_ratios=(3.0, 1.0))
		attitude_axis = attitude_figure.add_subplot(attitude_grid[0, 0], projection="3d")
		info_axis = attitude_figure.add_subplot(attitude_grid[0, 1])
		angle_axis = attitude_figure.add_subplot(attitude_grid[1, :])
		figure = Figure(figsize=(6.4, 7.6), dpi=92, facecolor=BACKGROUND, constrained_layout=True)
		grid = figure.add_gridspec(2, 1, height_ratios=height_ratios)
		phase_axis = figure.add_subplot(grid[0], projection="3d")
		omega_axis = figure.add_subplot(grid[1])

		marker = self._draw_phase_portrait(phase_axis, solution, inertia)
		cursor = self._draw_omega(omega_axis, solution, times)

		axis_index = self.results["axis_index"]
		flips = np.count_nonzero(np.diff(np.sign(solution[:, axis_index])) != 0)
		divergence = "   ·   ¡divergió numéricamente!" if run["diverged"] else ""
		self.summary_label.set(
			rf"$\Delta t = {float(self.results['dt']):.4g}$ s   ·   $\Omega = {self.results['omega_degrees']:.0f}^\circ$/s   ·   "
			rf"$\varepsilon = 10^{{{self.results['log_perturbation']:.1f}}}$   ·   "
			rf"cambios de signo de $\omega_{axis_index + 1}$: {flips}{divergence}"
		)
		for frame_column, current in enumerate((attitude_figure, figure)):
			holder = ttk.Frame(self.results_frame)
			holder.grid(row=0, column=frame_column, sticky="nsew")
			holder.pack_propagate(False)
			make_canvas(holder, current)
		self._animate_attitude(attitude_axis, info_axis, angle_axis, run, inertia, [marker, cursor])

	def _draw_phase_portrait(self, axis, solution: Array, inertia: Array):
		axis.set_title(r"Espacio de fases $(\omega_1, \omega_2, \omega_3)$ y superficies de Poinsot", fontsize=10)
		initial = np.asarray(self.results["initial"])
		energy = 2 * float(T(*inertia, initial))
		momentum = float(L(*inertia, initial))
		if energy > 0 and momentum > 0:
			energy_radii = np.sqrt(energy / inertia)
			momentum_radii = momentum / inertia
			axis.plot_surface(*ellipsoid(energy_radii), color="#e09b49", alpha=0.10, linewidth=0)
			axis.plot_surface(*ellipsoid(momentum_radii), color="#167d73", alpha=0.11, linewidth=0)
			axis.plot([], [], [], color="#e09b49", label=r"Energía $2T$")
			axis.plot([], [], [], color="#167d73", label=r"Momento $L^2$")
			for index, curve in enumerate(separatrix(*inertia, momentum)):
				axis.plot(*curve.T, color="#c34c32", linewidth=1.0, linestyle="--", label="Separatriz crítica" if index == 0 else None)
			for index, color in enumerate(AXIS_COLORS):
				points = np.zeros((2, 3))
				points[:, index] = (momentum / inertia[index], -momentum / inertia[index])
				axis.scatter(*points.T, color=color, s=18, depthshade=False)
			extent = 1.1 * float(np.max(momentum_radii))
			axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
		axis.plot(*solution.T, color="#173b38", linewidth=1.2, label=r"Polhodia $\omega(t)$")
		(marker,) = axis.plot(
			*solution[0], marker="o", color=POLHODE_MARKER, markeredgecolor="#3d4247", markersize=7, linestyle="",
			label=r"$\omega$ actual",
		)
		axis.set_xlabel(r"$\omega_1$ (rad/s)")
		axis.set_ylabel(r"$\omega_2$ (rad/s)")
		axis.set_zlabel(r"$\omega_3$ (rad/s)")
		axis.set_box_aspect((1, 1, 1), zoom=0.88)
		axis.legend(fontsize=7, loc="upper left")
		return marker

	@staticmethod
	def _draw_omega(axis, solution: Array, times: Array):
		for index, color in enumerate(AXIS_COLORS):
			axis.plot(times, solution[:, index], color=color, linewidth=1.3, label=rf"$\omega_{index + 1}$")
		axis.set(title=r"Velocidad angular en ejes cuerpo $\omega(t)$", xlabel=r"Tiempo $t$ (s)", ylabel=r"$\omega$ (rad/s)")
		axis.title.set_fontsize(10)
		axis.grid(alpha=0.22)
		axis.legend(fontsize=8, loc="upper right", ncol=3)
		return axis.axvline(times[0], color="#111111", linewidth=0.8, alpha=0.6)

	def _animate_attitude(self, axis, info_axis, angle_axis, run: dict, inertia: Array, synced: list) -> None:
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
		axis.set_xlabel(r"$X$ inercial")
		axis.set_ylabel(r"$Y$ inercial")
		axis.set_zlabel(r"$Z$ inercial")
		axis.view_init(elev=22, azim=35)
		axis.set(xticks=(-1, 0, 1), yticks=(-1, 0, 1), zticks=(-1, 0, 1))
		axis.tick_params(labelsize=8, pad=0)
		axis.set_title("Orientación del satélite", fontsize=10)

		faces, base_colors, lines, _ = satellite_model(inertia)
		body = Poly3DCollection(faces, edgecolors=(0.15, 0.15, 0.17, 0.45), linewidths=0.35)
		axis.add_collection3d(body)
		line_artists = [(segments, axis.plot([], [], [], color=color, linewidth=width)[0]) for segments, color, width in lines]
		axis_lines = [
			axis.plot([], [], [], color=color, linewidth=1.8, label=rf"$\mathbf{{e}}_{i + 1}$")[0]
			for i, color in enumerate(AXIS_COLORS)
		]
		(trace,) = axis.plot([], [], [], color=AXIS_COLORS[axis_index], linewidth=0.8, alpha=0.55)
		# L es constante en ejes inerciales: L_in = R(0) I ω(0) con R(0) = Id.
		momentum = inertia * solution[0]
		momentum_dir = momentum / np.linalg.norm(momentum) * extent * 0.95
		axis.plot(*zip(-momentum_dir, momentum_dir), color="#111111", linestyle="--", linewidth=1.0, label=r"$\mathbf{L}$ (inercial, fijo)")
		axis.legend(fontsize=8, loc="upper left")

		# Estado del instante actual en viñetas, a la derecha de la orientación.
		info_axis.axis("off")
		info_axis.text(0.0, 0.80, "Estado actual", transform=info_axis.transAxes, fontsize=10, fontweight="bold", color="#173b38")
		time_text = info_axis.text(0.0, 0.68, "", transform=info_axis.transAxes, fontsize=10, color="#172b2a")
		angle_text = info_axis.text(0.0, 0.58, "", transform=info_axis.transAxes, fontsize=10, color="#172b2a")
		info_axis.text(
			0.0, 0.40,
			rf"$\angle(\mathbf{{e}}_{axis_index + 1}, \mathbf{{L}})$: $0^\circ$ o $180^\circ$" + "\n"
			+ "eje de giro alineado con " + r"$\mathbf{L}$" + ";\nun salto entre ambos\nes un volteo.",
			transform=info_axis.transAxes, fontsize=8, color="#5b6d6a", va="top", linespacing=1.4,
		)

		stride = max(1, len(times) // 480)
		frame_indices = np.arange(0, len(times), stride)
		if frame_indices[-1] != len(times) - 1:
			frame_indices = np.append(frame_indices, len(times) - 1)
		tips = rotations[:, :, axis_index] * extent * 0.95
		# Ángulo entre el eje nominal de giro (en inercial) y L: 0° ↔ 180° en cada volteo.
		angles = np.degrees(np.arccos(np.clip(rotations[:, :, axis_index] @ (momentum_dir / np.linalg.norm(momentum_dir)), -1, 1)))
		angle_axis.plot(times, angles, color=AXIS_COLORS[axis_index], linewidth=1.4)
		angle_axis.set(
			title=rf"Ángulo entre $\mathbf{{e}}_{axis_index + 1}$ y $\mathbf{{L}}$ (inercial)", xlabel=r"Tiempo $t$ (s)",
			ylabel=r"$\angle(\mathbf{e}, \mathbf{L})$ ($^\circ$)", ylim=(-5, 185), yticks=(0, 45, 90, 135, 180),
		)
		angle_axis.title.set_fontsize(10)
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
			time_text.set_text(rf"•  $t = {times[step]:.2f}$ s")
			angle_text.set_text(rf"•  $\angle(\mathbf{{e}}_{axis_index + 1}, \mathbf{{L}}) = {angles[step]:.1f}^\circ$")
			if frame % 2 == 0:
				blit_synced()
			return [body, trace, angle_cursor, time_text, angle_text, *axis_lines]

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
