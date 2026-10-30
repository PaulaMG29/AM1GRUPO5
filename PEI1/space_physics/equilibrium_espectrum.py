from numpy import zeros
from numpy.linalg import eigvals

from .exact_jacobian import exact_Jacobian

def equilibrium_spectrum(I1: float, I2: float, I3: float, Omega: float, k: int) -> tuple:
    """
    Linealización en el equilibrio omega*(k) = Omega e_k (k = 0, 1, 2).

    Retorna:
    (autovalores de exact_Jacobian, True si hay algún autovalor con parte real > 0 -> inestable)
    """
    omega = zeros(3)
    omega[k] = Omega
    eigenvalues = eigvals(exact_Jacobian(I1, I2, I3, omega))
    return eigenvalues, bool(max(eigenvalues.real) > 1e-9)
