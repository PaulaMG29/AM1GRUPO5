"""Ventana 4: «Error Estimado» (error máximo, error de Richardson, convergencia y deriva de invariantes)."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure

from .config import AXIS_COLORS, BACKGROUND, CARD, SCHEME_COLORS, SCHEMES, Array
from .helpers import error_summary, latex_sci
from .widgets import MathLabel, make_canvas, make_card


class EstimatedErrorWindow:
	"""Métodos de la ventana 4; se combinan en SatelliteAttitudeApp (app.py), que guarda el estado."""

	# --------------------------------------------------------------------------------------------
	# Page 4: estimated error
	# --------------------------------------------------------------------------------------------

	def _build_error_page(self) -> None:
		if self._build_results_toolbar(with_pause=False) is None:
			return
		self._render_error()

	def _render_error(self) -> None:
		for child in self.results_frame.winfo_children():
			child.destroy()
		assert self.results is not None
		name = self.scheme_name
		run = self.results["runs"][name]
		_, order = SCHEMES[name]
		times = run["times"]
		error = run["richardson"][: len(times)]
		estimate = run["order_estimate"]
		self.summary_label.set(
			rf"{name}   ·   orden teórico $p = {order}$, estimado $q = {estimate:.2f}$   ·   "
			rf"$\Delta t = {float(self.results['dt']):.4g}$ s"
		)

		# Arriba: tarjeta con el error máximo. Debajo, 2 x 2: el error y su orden en la fila
		# superior, y la deriva de los invariantes (física) en la inferior.
		self.results_frame.columnconfigure(0, weight=1)
		self.results_frame.rowconfigure(0, weight=0)
		self.results_frame.rowconfigure(1, weight=1)
		self._build_max_error_card(self.results_frame, run, error, times).grid(row=0, column=0, sticky="ew", pady=(0, 8))

		figure = Figure(figsize=(12.0, 6.4), dpi=92, facecolor=BACKGROUND, constrained_layout=True)
		grid = figure.add_gridspec(2, 2)
		error_axis = figure.add_subplot(grid[0, 0])
		convergence_grid = grid[0, 1].subgridspec(1, 2, width_ratios=(2.6, 1.0))
		convergence_axis = figure.add_subplot(convergence_grid[0])
		convergence_info = figure.add_subplot(convergence_grid[1])
		energy_axis = figure.add_subplot(grid[1, 0])
		momentum_axis = figure.add_subplot(grid[1, 1])

		self._draw_richardson(error_axis, error, times, estimate)
		self._draw_convergence(convergence_axis, convergence_info, *run["convergence"], estimate, order)
		self._draw_invariants(energy_axis, momentum_axis, name)

		holder = tk.Frame(self.results_frame, background=BACKGROUND)
		holder.grid(row=1, column=0, sticky="nsew")
		holder.pack_propagate(False)
		make_canvas(holder, figure).draw_idle()

	def _build_max_error_card(self, parent, run: dict, error: Array, times: Array) -> tk.Frame:
		card = make_card(
			parent, r"Error máximo estimado por Richardson   $\mathbf{E}(t) \approx \omega_{\mathrm{exacta}}(t) - \omega_{\Delta t}(t)$"
		)
		if run.get("richardson_failed"):
			MathLabel(card, text=f"No se pudo estimar el error: {run['richardson_failed']}", background=CARD, color="#b3412e").grid(
				row=1, column=0, sticky="w"
			)
			return card
		summary = error_summary(error, run["solution"], times)
		components = summary["components"]
		energy, momentum = run["energy_drift"], run["momentum_drift"]
		relative = summary["relative"]
		# (título, fórmula, valor, nota) de cada recuadro.
		tiles = (
			(
				"Error máximo", r"$\max_t\,\Vert\mathbf{E}(t)\Vert$",
				rf"${latex_sci(summary['max'])}$ rad/s", rf"en $t = {summary['time']:.2f}$ s",
			),
			(
				"Error relativo máximo", r"$\max_t\,\Vert\mathbf{E}\Vert\,/\,\Vert\omega\Vert$",
				rf"${latex_sci(relative)}$", self._percent(relative),
			),
			(
				"Máximo por componente", r"$\max_t\,|E_i|$  (rad/s)",
				"\n".join(rf"$|E_{index + 1}| = {latex_sci(value)}$" for index, value in enumerate(components)), "",
			),
			(
				"Error al final", r"$\Vert\mathbf{E}(t_f)\Vert$",
				rf"${latex_sci(summary['final'])}$ rad/s", rf"$t_f = {times[-1]:.2f}$ s",
			),
			(
				"Deriva de invariantes", r"$\max_t\,\Delta T$,  $\max_t\,\Delta L$",
				rf"$\Delta T = {latex_sci(self._nanmax(energy))}$" + "\n" + rf"$\Delta L = {latex_sci(self._nanmax(momentum))}$", "",
			),
		)
		tile_color = "#f1f7f6"
		for column, (title, formula, value, note) in enumerate(tiles):
			card.columnconfigure(column, weight=1, uniform="tiles")
			tile = tk.Frame(card, background=tile_color, highlightbackground="#d7e5e2", highlightthickness=1, padx=12, pady=8)
			tile.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
			MathLabel(tile, text=title, background=tile_color, color="#245b55", size=9, weight="bold").pack(anchor="w")
			MathLabel(tile, text=formula, background=tile_color, color="#5b6d6a", size=10).pack(anchor="w")
			multiline = "\n" in value
			MathLabel(tile, text=value, background=tile_color, color="#173b38", size=11 if multiline else 15).pack(anchor="w", pady=(4, 0))
			if note:
				MathLabel(tile, text=note, background=tile_color, color="#5b6d6a", size=9).pack(anchor="w", pady=(2, 0))
		return card

	@staticmethod
	def _percent(relative: float) -> str:
		"""«= 9.87 %» (o en notación científica si es muy pequeño o muy grande)."""
		if not np.isfinite(relative):
			return ""
		percent = 100 * relative
		value = f"{percent:.3g}" if 1e-2 <= percent < 1e3 else latex_sci(percent)
		return rf"$= {value}\ \%$"

	@staticmethod
	def _nanmax(values: Array) -> float:
		return float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.0

	@staticmethod
	def _draw_richardson(axis, error: Array, times: Array, order: float) -> None:
		with np.errstate(divide="ignore", invalid="ignore"):
			for index, color in enumerate(AXIS_COLORS):
				component = np.abs(error[:, index])
				axis.semilogy(times, np.where(component > 0, component, np.nan), color=color, linewidth=0.9, alpha=0.75, label=rf"$|E_{index + 1}|$")
			total = np.linalg.norm(error, axis=1)
			axis.semilogy(times, np.where(total > 0, total, np.nan), color="#111111", linewidth=1.4, label=r"$\Vert\mathbf{E}\Vert$")
		axis.set(
			title=(
				rf"Error de Richardson ($q = {order:.2f}$):   "
				r"$\mathbf{E} = \dfrac{\mathbf{U}_{\Delta t/2} - \mathbf{U}_{\Delta t}}{1 - 2^{-q}}$"
			),
			xlabel=r"Tiempo $t$ (s)", ylabel="Error estimado (rad/s)",
		)
		axis.title.set_fontsize(10)
		axis.grid(alpha=0.22, which="both")
		axis.legend(fontsize=8, loc="lower right", ncol=2)

	@staticmethod
	def _draw_convergence(axis, info_axis, log_n: Array, log_e: Array, used: Array, estimate: float, order: int) -> None:
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
			axis.plot(log_n[finite], intercept - estimate * log_n[finite], "--", color="#c34c32", linewidth=1.0, label=rf"pendiente $-q = {-estimate:.2f}$")
		axis.set(title="Convergencia del esquema", xlabel=r"$\ln N$", ylabel=r"$\ln E$")
		axis.title.set_fontsize(10)
		axis.grid(alpha=0.22)
		axis.legend(fontsize=8, loc="upper right")

		# Orden estimado y teórico en viñetas, a la derecha de la gráfica.
		info_axis.axis("off")
		bullets = (
			("Orden del esquema", dict(fontweight="bold", color="#173b38")),
			(rf"•  Estimado: $q = {estimate:.2f}$", dict(color="#172b2a")),
			(rf"•  Teórico: $p = {order}$", dict(color="#172b2a")),
			(rf"•  $|q - p| = {abs(estimate - order):.2f}$", dict(color="#172b2a")),
			(rf"•  Puntos del ajuste: {int(np.count_nonzero(used))} de {int(np.count_nonzero(finite))}", dict(color="#5b6d6a")),
		)
		for row, (text, style) in enumerate(bullets):
			info_axis.text(0.02, 0.88 - 0.13 * row, text, transform=info_axis.transAxes, fontsize=10 if row < 4 else 9, **style)

	def _draw_invariants(self, energy_axis, momentum_axis, selected: str) -> None:
		for name in SCHEMES:
			run = self.results["runs"].get(name)
			if run is None or run.get("failed"):
				continue
			width, alpha = (1.8, 1.0) if name == selected else (0.9, 0.55)
			energy_axis.semilogy(run["times"], run["energy_drift"], color=SCHEME_COLORS[name], linewidth=width, alpha=alpha, label=name)
			momentum_axis.semilogy(run["times"], run["momentum_drift"], color=SCHEME_COLORS[name], linewidth=width, alpha=alpha, label=name)
		energy_axis.set(
			title=r"Deriva de energía:   $\Delta T(t) = |2T(t) - 2T_0|\,/\,2T_0$", xlabel=r"Tiempo $t$ (s)", ylabel=r"$\Delta T$ relativa"
		)
		momentum_axis.set(
			title=r"Deriva de momento:   $\Delta L(t) = |L^2(t) - L_0^2|\,/\,L_0^2$", xlabel=r"Tiempo $t$ (s)", ylabel=r"$\Delta L$ relativa"
		)
		for axis in (energy_axis, momentum_axis):
			axis.title.set_fontsize(10)
			axis.grid(alpha=0.22, which="both")
			axis.legend(fontsize=8, loc="lower right", ncol=2)
