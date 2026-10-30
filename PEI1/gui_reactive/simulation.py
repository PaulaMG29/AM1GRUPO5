"""Capa de simulación: conecta la GUI con numerical_engine y space_physics (sin Tkinter)."""

from __future__ import annotations

import numpy as np

from numerical_engine.Cauchy_problem import Cauchy_Problem
from numerical_engine.Richardson_error import Richardson_Error, estimate_scheme_order
from space_physics.Euler_equations import Euler_Equations
from space_physics.attitude_equations import Attitude_Equations, nearest_rotation
from space_physics.invariants_drift import invariants_drift

from .config import SCHEMES
from .helpers import first_invalid, log_safe, set_inertias


def run_scheme(results: dict, name: str) -> dict:
	"""Cauchy_Problem de Attitude_Equations (ω y R) + invariantes 2T y L². Se guarda en caché."""
	runs = results["runs"]
	if name in runs:
		return runs[name]
	scheme, _order = SCHEMES[name]
	times = results["times"]
	inertia = results["inertias"]
	set_inertias(inertia)
	# Estado U = [ω, R] con R(0) = Id: ω sale igual que con Euler_Equations y R da la actitud.
	initial_state = np.concatenate([results["initial"], np.eye(3).reshape(9)])
	try:
		with np.errstate(all="ignore"):
			states, _ = Cauchy_Problem(scheme, initial_state, Attitude_Equations, times)
	except (ValueError, np.linalg.LinAlgError) as error:
		runs[name] = {"failed": f"{name}: {error}"}
		return runs[name]
	valid = first_invalid(states)
	if valid < 2:
		runs[name] = {"failed": f"{name}: la solución diverge desde el primer paso."}
		return runs[name]
	states, run_times = states[:valid], times[:valid]
	solution = states[:, 0:3]
	with np.errstate(all="ignore"):
		energy_drift, momentum_drift = invariants_drift(*inertia, solution)
	runs[name] = {
		"solution": solution,
		"times": run_times,
		"diverged": valid < len(times),
		"energy_drift": log_safe(energy_drift),
		"momentum_drift": log_safe(momentum_drift),
		# Para dibujar se proyecta R sobre la rotación más cercana (el esquema no conserva RᵀR = I).
		"rotations": np.array([nearest_rotation(state[3:12].reshape(3, 3)) for state in states]),
	}
	return runs[name]


def richardson(results: dict, name: str) -> dict:
	"""Richardson_Error + estimate_scheme_order de numerical_engine (en caché)."""
	run = results["runs"][name]
	if "richardson" in run:
		return run
	scheme, _order = SCHEMES[name]
	times = results["times"]
	# Las inercias de esta ejecución (los sliders pueden haber cambiado desde entonces).
	set_inertias(results["inertias"])
	try:
		with np.errstate(all="ignore"):
			order, log_n, log_e, used = estimate_scheme_order(scheme, results["initial"], Euler_Equations, times)
			error, order = Richardson_Error(scheme, results["initial"], Euler_Equations, times, Orden=order)
	except (ValueError, np.linalg.LinAlgError, TypeError) as failure:
		error, order = np.full((len(times), 3), np.nan), np.nan
		log_n, log_e, used = np.array([]), np.array([]), np.array([], dtype=bool)
		run["richardson_failed"] = str(failure)
	run["richardson"] = error
	run["order_estimate"] = order
	run["convergence"] = (log_n, log_e, used)
	return run
