"""
Grupo space_physics: física del sólido rígido dividida en capas.

    Euler_equations        Euler_Equations (e inercias globales I1, I2, I3)
    exact_jacobian         exact_Jacobian
    kinetic_energy         T
    angular_momentum       L
    attitude_equations     Attitude_Equations, nearest_rotation
    separatrix             separatrix
    equilibrium_espectrum  equilibrium_spectrum
    invariants_drift       invariants_drift

Se reexportan aquí para que `from space_physics import X` siga funcionando igual que con el
antiguo space_physics.py. Las inercias globales viven en space_physics.Euler_equations.
"""

from .Euler_equations import Euler_Equations
from .exact_jacobian import exact_Jacobian
from .kinetic_energy import T
from .angular_momentum import L
from .attitude_equations import Attitude_Equations, nearest_rotation
from .separatrix import separatrix
from .equilibrium_espectrum import equilibrium_spectrum
from .invariants_drift import invariants_drift
