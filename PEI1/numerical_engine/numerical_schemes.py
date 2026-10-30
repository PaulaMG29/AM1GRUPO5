from numpy import array
from typing import Callable

from .non_linear_solvers import our_Newton

#================================================================================================
# NUMERICAL SCHEMES
#================================================================================================

def Euler(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    return U + Dt * F(U, t)

def RK4(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    k1 = F(U, t)
    k2 = F(U + Dt*k1/2, t+Dt/2)
    k3 = F(U + Dt*k2/2, t+Dt/2)
    k4 = F(U + Dt*k3, t+Dt)
    return U + Dt*(k1+2*k2+2*k3+k4)/6

def Crank_Nicolson(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    """
    G = function
    U = seed
    """
    def G(Y, t):
        return Y + A - Dt/2*(F(Y, t + Dt))
    A = - U - Dt/2 * F(U, t)
    return our_Newton(G, U, 0, 1e-6, 100)

def Inverse_Euler(U: array, t: float, Dt: float, F: Callable[[array,float],array]) -> array:
    """
    G = function
    U = seed

    U^{n+1} es la raíz de G(Y) = Y - U^n - Dt F(Y, t + Dt), resuelta con our_Newton
    (como en Crank_Nicolson) en lugar de la iteración de punto fijo de hito2.
    """
    def G(Y, t):
        return Y + A - Dt * F(Y, t + Dt)
    A = - U
    return our_Newton(G, U, 0, 1e-6, 100)
