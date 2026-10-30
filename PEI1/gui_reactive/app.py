"""Carcasa de la aplicación: ventana principal, menú lateral, navegación entre ventanas y estado."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from matplotlib import animation

from .config import AXIS_NAMES, BACKGROUND
from .window_initial_conditions import InitialConditionsWindow
from .window_integration_error import IntegrationErrorWindow
from .window_numerical_schemes import NumericalSchemesWindow


class SatelliteAttitudeApp(InitialConditionsWindow, NumericalSchemesWindow, IntegrationErrorWindow):
	"""Three-view Tkinter dashboard backed by the PEI1 numerical modules.

	Esta clase es la «carcasa»: estado compartido, menú lateral, navegación y cálculo en segundo
	plano. El contenido de cada ventana se hereda de su propio fichero window_*.py.
	"""

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
