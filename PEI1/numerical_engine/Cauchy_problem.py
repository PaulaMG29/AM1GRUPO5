from numpy import array, zeros
from typing import Callable

#================================================================================================
# CAUCHY PROBLEM
#================================================================================================

def Cauchy_Problem(Esquema: Callable, U0: array, F: Callable, t: array) -> array:
    """
    t   : time domain (vector con todos los instantes temporales)
    """
    Nv = len(U0)
    N = len(t) - 1
    U = zeros((N+1, Nv))
    U[0, :] = U0
    for n in range(0, N):
        U[n+1, :] = Esquema(U[n, :], t[n], t[n+1] - t[n], F)
    return U, t
