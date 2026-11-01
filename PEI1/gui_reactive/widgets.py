"""Capa de widgets reutilizables: etiquetas LaTeX, lienzos de Matplotlib, tarjetas y sliders con entrada."""

from __future__ import annotations

import base64
import io
import tkinter as tk
from functools import lru_cache
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import BACKGROUND, CARD, UI_FONT


@lru_cache(maxsize=1024)
def _math_png(text: str, size: float, color: str, weight: str, style: str, dpi: float) -> bytes:
	"""PNG transparente con `text` renderizado por mathtext (admite $...$ y varias líneas)."""
	figure = Figure(dpi=dpi)
	canvas = FigureCanvasAgg(figure)
	label = figure.text(
		0, 0, text, fontsize=size, color=color, fontweight=weight, fontstyle=style, family=UI_FONT,
		ha="left", va="bottom", linespacing=1.35,
	)
	box = label.get_window_extent(canvas.get_renderer())
	pad = 2.0
	width, height = box.width + 2 * pad, box.height + 2 * pad
	figure.set_size_inches(width / dpi, height / dpi)
	label.set_position((pad / width, pad / height))
	buffer = io.BytesIO()
	figure.savefig(buffer, format="png", dpi=dpi, transparent=True)
	return buffer.getvalue()


class MathLabel(tk.Label):
	"""Etiqueta de Tk que muestra texto con fórmulas LaTeX ($...$), renderizado como imagen.

	Tk no sabe dibujar LaTeX, así que el texto se pasa por el mathtext de Matplotlib y se
	muestra como PhotoImage. Se usa igual que un Label: MathLabel(parent, text=..., size=10).
	"""

	def __init__(
		self, parent, text: str = "", size: float = 10, color: str = "#172b2a", weight: str = "normal",
		style: str = "normal", background: str = BACKGROUND, **options,
	) -> None:
		super().__init__(parent, background=background, borderwidth=0, **options)
		self._style = {"size": size, "color": color, "weight": weight, "style": style}
		self._text: str | None = None
		self.set(text)

	def set(self, text: str, **style) -> None:
		"""Cambia el texto y, opcionalmente, size, color, weight o style."""
		style = {**self._style, **style}
		if text == self._text and style == self._style:
			return
		self._text, self._style = text, style
		if not text:
			self._image = None
			self.configure(image="")
			return
		png = _math_png(text, style["size"], style["color"], style["weight"], style["style"], self.winfo_fpixels("1i"))
		self._image = tk.PhotoImage(master=self, data=base64.b64encode(png))
		self.configure(image=self._image)


def make_canvas(parent, figure: Figure) -> FigureCanvasTkAgg:
	canvas = FigureCanvasTkAgg(figure, master=parent)
	widget = canvas.get_tk_widget()
	# Tamaño pedido mínimo: el reparto lo decide la rejilla, no figsize (evita bucles de resize).
	widget.configure(width=1, height=1, background=BACKGROUND, highlightthickness=0)
	widget.pack(fill="both", expand=True)
	return canvas


def make_card(parent, title: str) -> tk.Frame:
	card = tk.Frame(parent, background=CARD, highlightbackground="#cfdcd9", highlightthickness=1, padx=14, pady=10)
	MathLabel(card, text=title, background=CARD, color="#173b38", size=11, weight="bold").grid(
		row=0, column=0, columnspan=4, sticky="w", pady=(0, 6)
	)
	return card


def slider_control(
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
	MathLabel(group, text=label, color="#245b55", weight="bold").grid(row=0, column=0, sticky="w")
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
		command=lambda raw: _slider_changed(raw, value_var, callback, fmt),
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


def _slider_changed(raw: str, value_var: tk.StringVar, callback: Callable[[float], None], fmt: str) -> None:
	value = float(raw)
	value_var.set(fmt.format(value))
	callback(value)
