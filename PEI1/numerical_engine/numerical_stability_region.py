from numpy import array, linspace, meshgrid, abs, zeros_like
from typing import Callable

#================================================================================================
# NUMERICAL STABILITY REGION
#================================================================================================

def create_mesh(xlim:tuple, ylim:tuple, N:int):
    x = linspace(xlim[0], xlim[1], N)
    y = linspace(ylim[0], ylim[1], N)
    X, Y = meshgrid(x, y)
    return X, Y

def stability_region(scheme:Callable, X: array, Y: array):
    N = X.shape[0]
    Z = X + 1j*Y
    R = zeros_like(Z, dtype=complex)
    for i in range(N):
        for j in range(N):
            z = Z[i,j]
            # Ecuación test, solo para sacar la región de estabilidad, no es parte del esquema
            # omega = Dt * lamda
            # Como Dt = 1, entonces omega = lamda, y la ecuación test es U'= F = z*U
            def F(U, t):
                return z*U
            # Tomando U0 = 1, el valor obtenido tras un paso (U1) es 
            # directamente el factor de amplificación R(z).
            U0 = array([1+0j])
            # U1 = scheme(U, t, Dt, F)
            U1 = scheme(U0, 0.0, 1.0, F)
            R[i,j] = U1[0]
    # La región de estabilidad viene dada por los puntos que 
    # cumplen |R(z)| ≤ 1.
    stable = abs(R) <= 1
    return stable, R
