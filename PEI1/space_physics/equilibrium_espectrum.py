from numpy import zeros
from numpy.linalg import eigvals
from typing import Callable

from .Euler_equations import Euler_Equations

def equilibrium_spectrum(I1: float, I2: float, I3: float, Omega: float, k: int, jacobian: Callable) -> tuple:
    """
    Linealización en el equilibrio omega*(k) = Omega e_k (k = 0, 1, 2).

    jacobian: procedimiento jacobian(F, U, t) -> matriz jacobiana de F en U. Se recibe como
    argumento (la GUI pasa our_jacobian de numerical_engine) para que el grupo físico no
    dependa del grupo matemático.

    Retorna:
    (autovalores del jacobiano de Euler_Equations,
     True si hay algún autovalor con parte real > 0 -> inestable)

    Las inercias se pasan explícitamente a Euler_Equations: no se tocan las globales, que
    pueden estar usándose en una integración en segundo plano.
    """
    omega = zeros(3)
    omega[k] = Omega
    def F(U, t):
        return Euler_Equations(U, t, (I1, I2, I3))
    eigenvalues = eigvals(jacobian(F, omega, 0.0))
    return eigenvalues, bool(max(eigenvalues.real) > 1e-9)
