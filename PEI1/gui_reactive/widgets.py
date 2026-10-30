"""Capa de widgets reutilizables: lienzos de Matplotlib, tarjetas y sliders con entrada."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import BACKGROUND, CARD


def make_canvas(parent, figure: Figure) -> FigureCanvasTkAgg:
	canvas = FigureCanvasTkAgg(figure, master=parent)
	widget = canvas.get_tk_widget()
	# Tamaño pedido mínimo: el reparto lo decide la rejilla, no figsize (evita bucles de resize).
	widget.configure(width=1, height=1, background=BACKGROUND, highlightthickness=0)
	widget.pack(fill="both", expand=True)
	return canvas


def make_card(parent, title: str) -> tk.Frame:
	card = tk.Frame(parent, background=CARD, highlightbackground="#cfdcd9", highlightthickness=1, padx=14, pady=10)
	tk.Label(card, text=title, background=CARD, foreground="#173b38", font=("Segoe UI", 11, "bold")).grid(
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
