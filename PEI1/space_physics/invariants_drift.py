from numpy import array, abs

from .kinetic_energy import T
from .angular_momentum import L

def invariants_drift(I1: float, I2: float, I3: float, omega: array) -> tuple:
    """
    Verificación de los invariantes a lo largo de una solución omega (N x 3).

    Retorna:
    (|2T(t) - 2T0| / 2T0, |L²(t) - L0²| / L0²)
    """
    energy = array([2 * T(I1, I2, I3, w) for w in omega])
    momentum = array([L(I1, I2, I3, w)**2 for w in omega])
    return abs(energy - energy[0]) / abs(energy[0]), abs(momentum - momentum[0]) / abs(momentum[0])
