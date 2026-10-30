"""
Grupo numerical_engine: motor numérico dividido en capas.

    linear_solvers              partition
    non_linear_solvers          our_jacobian, our_Newton
    numerical_schemes           Euler, RK4, Crank_Nicolson, Inverse_Euler
    Cauchy_problem              Cauchy_Problem
    Richardson_error            estimate_scheme_order, Richardson_Error
    numerical_stability_region  create_mesh, stability_region

Se reexportan aquí para que `from numerical_engine import X` siga funcionando igual que
con el antiguo numerical_engine.py.
"""

from .linear_solvers import partition
from .non_linear_solvers import our_jacobian, our_Newton
from .numerical_schemes import Euler, RK4, Crank_Nicolson, Inverse_Euler
from .Cauchy_problem import Cauchy_Problem
from .Richardson_error import estimate_scheme_order, Richardson_Error
from .numerical_stability_region import create_mesh, stability_region
