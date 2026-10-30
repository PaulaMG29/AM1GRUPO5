from numpy import array, zeros, abs, inf, log, polyfit, median
from typing import Callable
from numpy.linalg import norm

from .linear_solvers import partition
from .Cauchy_problem import Cauchy_Problem

#================================================================================================
# ERROR ESTIMATION
#================================================================================================

def estimate_scheme_order(Esquema: Callable, U0: array, F: Callable, t: array) -> tuple:
    """
    Devuelve (order, log(N_values), log(errors), valid_points):
    valid_points marca los puntos usados en el ajuste lineal (región asintótica).
    """
    number_meshes = 5
    N_values = zeros(number_meshes, dtype=int)
    for i in range(N_values.size):
        N_values[i] = (len(t) - 1) * 2**(i - 2)

    errors = []
    for N in N_values:
        # malla gruesa
        t1 = partition(t[0], t[-1], N)
        # malla fina
        t2 = partition(t[0], t[-1], 2 * N)
        U1, t1 = Cauchy_Problem(Esquema, U0, F, t1)
        U2, t2 = Cauchy_Problem(Esquema, U0, F, t2)

        U2 = U2[::2]

        E = U2 - U1
        errors.append(norm(E, inf))

    errors = array(errors, dtype=float)
    x = log(N_values)
    y = log(errors)
    finite_errors = (errors > 0) & (errors < inf)

    # Entre cada par de mallas consecutivas se calcula una
    # estimación local del orden:
    #     p = -Δ(log(E)) / Δ(log(N))
    # Si el método está en régimen asintótico, estos valores
    # deberían ser aproximadamente constantes.

    interval_indices = [
        i for i in range(len(errors) - 1) if finite_errors[i] and finite_errors[i + 1]
    ]
    if not interval_indices:
        raise ValueError("No hay errores positivos consecutivos para estimar el orden.")
    local_orders = array([
        -(y[i + 1] - y[i]) / (x[i + 1] - x[i])
        for i in interval_indices
    ])

    # Se toma la mediana de los órdenes locales porque es poco
    # sensible a valores atípicos producidos por:
    #   - Mallas demasiado gruesas (error de truncamiento alto)
    #   - Mallas demasiado finas (error de redondeo)

    p_ref = median(local_orders)

    # Se consideran válidos únicamente los intervalos cuyo
    # orden local difiere menos de un 10 % del orden de
    # referencia.
 
    tol = 0.10 * abs(p_ref)
    valid_intervals = abs(local_orders - p_ref) < tol

    # Si un intervalo es válido, se conservan los dos puntos
    # extremos que lo definen.

    valid_points = zeros(len(N_values), dtype=bool)
    for interval, i in enumerate(interval_indices):
        if valid_intervals[interval]:
            valid_points[i] = True
            valid_points[i + 1] = True

    x_fit = x[valid_points]
    y_fit = y[valid_points]

    # Ajuste lineal en la región asintótica; si no hay intervalos
    # seleccionables, la mediana local evita un ajuste vacío.
    if len(x_fit) >= 2:
        coeffs = polyfit(x_fit, y_fit, 1)
        order = -coeffs[0]
    else:
        order = p_ref

    return order, x, y, valid_points

def Richardson_Error(Esquema: Callable, U0: array, F: Callable, t: array, Orden: float = None)-> array:
    """
    Orden: orden del esquema; si no se da, se estima con estimate_scheme_order.
    """
    t1 = t
    t2 = partition(t[0], t[-1], 2 * (len(t) - 1))
    Dt = t[1] - t[0]
    # Solución con Dt:
    U1, t_1 = Cauchy_Problem(Esquema, U0, F, t1)
    # Solución con Dt /2:
    U2, t_2 = Cauchy_Problem(Esquema, U0, F, t2)

    # Comparamos cada 2 puntos de la solución con Dt/2
    U2 = U2[::2]

    if Orden is None:
        Orden, _, _, _ = estimate_scheme_order(Esquema, U0, F, t)
    Error = (U2 - U1) / (Dt**Orden - (Dt/2)**Orden) * Dt**Orden
    
    return Error, Orden
